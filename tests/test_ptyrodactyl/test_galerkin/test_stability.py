"""Tests for :mod:`ptyrodactyl.galerkin.stability`."""

import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction
from itertools import product
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple

import ptyrodactyl.galerkin.sources as sources_module
import ptyrodactyl.galerkin.stability as stability_module
from ptyrodactyl.galerkin import (
    cgls_solve,
    check_galerkin_absorber_floor,
    create_galerkin_target,
    create_host_checked_galerkin_target,
    create_matched_galerkin_source,
    invoke_galerkin_stability,
    lsqr_solve,
)
from ptyrodactyl.galerkin.sources import (
    build_represented_focused_galerkin_source,
    build_represented_plane_galerkin_source,
)
from ptyrodactyl.galerkin.stability import (
    check_represented_galerkin_absorber_floor,
    invoke_represented_galerkin_stability,
)
from ptyrodactyl.types import (
    GalerkinCertificateReason,
    GalerkinRepresentedSource,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    GalerkinSource,
    GalerkinSourceBranch,
    GalerkinStabilityDisposition,
    GalerkinStabilityFailure,
    GalerkinStabilityProof,
    GalerkinStabilityResult,
    GalerkinStabilityRoute,
    GalerkinTargetManifest,
    create_galerkin_product_support,
    create_galerkin_solve_result,
    create_galerkin_source,
    create_potential_3d,
)
from tests._galerkin_target_fixture import (
    TARGET_CAP_SCALE,
    TARGET_VOLTAGE_KV,
    checked_acquisition,
    production_target,
    production_vacuum_target,
)

from .test_sources import (
    _manifest as _represented_manifest,
)
from .test_sources import (
    _position as _represented_position,
)
from .test_sources import (
    _source_kwargs as _represented_source_kwargs,
)
from .test_system import _dense_target

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _submitted_result(field: jnp.ndarray) -> GalerkinSolveResult:
    """Create one submitted state with intentionally untrusted diagnostics."""
    zeros = jnp.zeros_like(field)
    return create_galerkin_solve_result(
        field=field,
        residual=zeros,
        residual_norm=0.0,
        normal_residual_norm=0.0,
        recurrence_residual_norm=0.0,
        iterations=1,
        operator_applications=4,
        status=GalerkinSolveStatus.CONVERGED,
        converged=True,
        method=GalerkinSolveMethod.CGLS,
        certificate_reason=GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND,
    )


def _case(
    *,
    field_offset: complex = 0.0,
) -> Tuple[GalerkinTargetManifest, GalerkinSource, GalerkinSolveResult]:
    """Create one exact finite vacuum result for Route-A invocation."""
    manifest = production_vacuum_target()
    incident = jnp.array(
        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        dtype=jnp.complex128,
    )
    source = create_matched_galerkin_source(manifest, incident)
    field = incident.at[0].add(field_offset)
    solve_result = _submitted_result(field)
    return manifest, source, solve_result


def _manifest_from_state_tuples(
    state_tuples: list[Tuple[int, int, int]],
    *,
    target_name: str,
) -> GalerkinTargetManifest:
    """Create one canonical vacuum target for an arbitrary finite support."""
    state_set = set(state_tuples)
    interaction_tuples = state_set | {
        tuple(-component for component in index) for index in state_set
    }
    absorber_tuples = {
        tuple(left[axis] - right[axis] for axis in range(3))
        for left in state_tuples
        for right in state_tuples
    } | set(product(range(-1, 2), repeat=3))
    multiplier_tuples = interaction_tuples | absorber_tuples
    work_tuples = {
        tuple(
            left_axis + right_axis
            for left_axis, right_axis in zip(left, right)
        )
        for left in state_tuples
        for right in multiplier_tuples
    }
    quotient_tuples = (
        state_set | interaction_tuples | absorber_tuples | work_tuples
    )
    work_shape = tuple(
        max(index[axis] for index in quotient_tuples)
        - min(index[axis] for index in quotient_tuples)
        + 3
        for axis in range(3)
    )
    support = create_galerkin_product_support(
        state_indices=jnp.asarray(state_tuples, dtype=jnp.int64),
        interaction_indices=jnp.asarray(
            sorted(interaction_tuples), dtype=jnp.int64
        ),
        absorber_indices=jnp.asarray(sorted(absorber_tuples), dtype=jnp.int64),
        work_indices=jnp.asarray(sorted(work_tuples), dtype=jnp.int64),
        work_shape=work_shape,
    )
    maximum_indices = tuple(
        max(abs(index[axis]) for index in interaction_tuples)
        for axis in range(3)
    )
    nx, ny, nz = tuple(4 * maximum + 5 for maximum in maximum_indices)
    box_lengths = (float(nx), float(ny), float(nz))
    potential = create_potential_3d(
        jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        voxel_size=(1.0, 1.0, 1.0),
        box_size=box_lengths,
        origin=(0.0, 0.0, 0.0),
        producer="stability-vacuum-support-fixture-v1",
        provenance_hash="e" * 64,
        coefficient_normalization="VC-1 periodic trigonometric mean DFT",
        band_limit=0.45,
    )
    eligibility = checked_acquisition(support, box_lengths)
    manifest = create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name=target_name,
    )
    return manifest


def _thirty_three_mode_manifest() -> GalerkinTargetManifest:
    """Create one canonical target immediately above the Gershgorin bound."""
    state_tuples = [(index, 0, 0) for index in range(-16, 17)]
    return _manifest_from_state_tuples(
        state_tuples, target_name="thirty-three-mode-cosine-box"
    )


def _infinite_delta_manifest() -> GalerkinTargetManifest:
    """Create one valid direct-Route-A target with RM-S2 ``delta_H=inf``."""
    state_tuples = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    absorber_tuples = set(product(range(-1, 2), repeat=3)) | {
        (-2, 0, 0),
        (2, 0, 0),
    }
    work_tuples = {
        tuple(
            left_axis + right_axis
            for left_axis, right_axis in zip(left, right)
        )
        for left in state_tuples
        for right in absorber_tuples
    }
    support = create_galerkin_product_support(
        state_indices=jnp.asarray(state_tuples, dtype=jnp.int64),
        interaction_indices=jnp.asarray(state_tuples, dtype=jnp.int64),
        absorber_indices=jnp.asarray(sorted(absorber_tuples), dtype=jnp.int64),
        work_indices=jnp.asarray(sorted(work_tuples), dtype=jnp.int64),
        work_shape=(7, 3, 3),
    )
    potential = create_potential_3d(
        jnp.asarray([[[1.01e308, -1.01e308, 0.0]]], dtype=jnp.float64),
        voxel_size=(1.0, 1.0, 1.0),
        box_size=(3.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        producer="stability-infinite-delta-fixture-v1",
        provenance_hash="f" * 64,
        coefficient_normalization="VC-1 periodic trigonometric mean DFT",
        band_limit=0.4,
    )
    eligibility = checked_acquisition(support, potential.box_size)
    manifest = create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name="direct-route-with-infinite-delta-h",
    )
    return manifest


def _zero_submission(
    manifest: GalerkinTargetManifest,
) -> Tuple[GalerkinSource, GalerkinSolveResult]:
    """Create one exact zero source and submitted state."""
    dimension = manifest.support.state_indices.shape[0]
    zeros = jnp.zeros(dimension, dtype=jnp.complex128)
    source = create_galerkin_source(
        incident_field=zeros,
        incident_source=zeros,
        additional_source=zeros,
        total_source=zeros,
        scattered_source=zeros,
        branch=GalerkinSourceBranch.FINITE_MATCHED,
    )
    solve_result = _submitted_result(zeros)
    return source, solve_result


def _represented_plane_source(
    manifest: GalerkinTargetManifest,
) -> GalerkinRepresentedSource:
    """Build one eligible represented plane source with an added RHS."""
    position = _represented_position(manifest, (0, 0, 0))
    additional = (
        jnp.zeros(
            (manifest.support.state_indices.shape[0],), dtype=jnp.complex128
        )
        .at[-1]
        .set(0.125 + 0.0625j)
    )
    source = build_represented_plane_galerkin_source(
        manifest=manifest,
        state_position=position,
        aperture_weight=jnp.asarray(2.0 - 0.5j, dtype=jnp.complex128),
        target_reduced_flux=jnp.asarray(2.5, dtype=jnp.float64),
        aberration_phase=jnp.asarray(0.125, dtype=jnp.float64),
        additional_source=additional,
        **_represented_source_kwargs(),
    )
    jax.block_until_ready(source)
    return source


class TestGalerkinStabilityInvocation:
    """Verify exact Route-A checking and per-result dispositions.

    :see: :func:`ptyrodactyl.galerkin.check_galerkin_absorber_floor`
    :see: :func:`ptyrodactyl.galerkin.invoke_galerkin_stability`
    """

    def test_exact_route_a_pass_bounds_true_dense_state_error(self) -> None:
        """Enclose the independently solved dense error."""
        state_budget = 1.0e-4
        manifest, source, solve_result = _case(field_offset=1.0e-8 + 0.0j)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=state_budget
        )
        result = invoke_galerkin_stability(
            manifest,
            source,
            solve_result,
            proof,
            state_budget=state_budget,
        )
        dense = _dense_target(manifest)
        exact = np.linalg.solve(dense, np.asarray(source.total_source))
        true_error = np.linalg.norm(exact - np.asarray(solve_result.field))

        assert proof.failure is GalerkinStabilityFailure.NONE
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )
        assert proof.route is GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX
        assert float(result.lower_singular_bound) == 201.0 / 1024.0
        algebraic_floor = Fraction(
            proof.algebraic_floor_numerator,
            proof.algebraic_floor_denominator,
        )
        transferred_floor = Fraction(
            proof.transferred_floor_numerator,
            proof.transferred_floor_denominator,
        )
        delta_h = Fraction.from_float(
            float(
                manifest.fixed_linear_error_ledger.fixed_linear_operator_error_bound
            )
        )
        assert algebraic_floor == Fraction(201, 1024)
        assert transferred_floor == algebraic_floor - delta_h
        assert proof.transferred_floor_finite
        assert (
            Fraction(proof.floor_numerator, proof.floor_denominator)
            == algebraic_floor
        )
        assert proof.exact_target_residual_finite
        assert proof.rhs_target == stability_module._RHS_TARGET
        assert proof.residual_scope == stability_module._RESIDUAL_SCOPE
        assert true_error <= float(result.state_error_upper_bound)
        assert float(result.state_error_upper_bound) <= state_budget

    @pytest.mark.parametrize(
        ("solve", "method"),
        [
            (cgls_solve, GalerkinSolveMethod.CGLS),
            (lsqr_solve, GalerkinSolveMethod.LSQR),
        ],
        ids=["cgls", "lsqr"],
    )
    def test_retained_solver_result_traverses_checker_and_invocation(
        self,
        solve: Callable[..., GalerkinSolveResult],
        method: GalerkinSolveMethod,
    ) -> None:
        """Invoke stability for each solve with an independent fixed budget."""
        preregistered_state_budget = 1.0e-5
        manifest = production_vacuum_target()
        incident = jnp.array(
            [0.2 - 0.1j, 1.0 + 0.0j, -0.3 + 0.4j],
            dtype=jnp.complex128,
        )
        source = create_matched_galerkin_source(manifest, incident)
        solve_result = solve(
            manifest,
            source.total_source,
            max_iterations=20,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
        )
        proof = check_galerkin_absorber_floor(
            manifest,
            source,
            solve_result,
            state_budget=preregistered_state_budget,
        )
        result = invoke_galerkin_stability(
            manifest,
            source,
            solve_result,
            proof,
            state_budget=preregistered_state_budget,
        )

        assert bool(solve_result.converged)
        assert solve_result.method is method
        assert proof.failure is GalerkinStabilityFailure.NONE
        assert proof.result_digest == result.result_digest
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )
        assert result.failure is GalerkinStabilityFailure.NONE
        assert float(result.state_budget) == preregistered_state_budget
        assert float(result.state_error_upper_bound) <= float(
            result.state_budget
        )

    def test_direct_exact_floor_survives_negative_perturbative_margin(
        self,
    ) -> None:
        """Use the exact shared CAP even when ``s_alg-delta_H`` is negative."""
        manifest = production_target()
        source, solve_result = _zero_submission(manifest)
        proof = check_galerkin_absorber_floor(
            manifest,
            source,
            solve_result,
            state_budget=1.0,
        )
        result = invoke_galerkin_stability(
            manifest,
            source,
            solve_result,
            proof,
            state_budget=1.0,
        )

        algebraic_floor = Fraction(
            proof.algebraic_floor_numerator,
            proof.algebraic_floor_denominator,
        )
        transferred_floor = Fraction(
            proof.transferred_floor_numerator,
            proof.transferred_floor_denominator,
        )
        exact_floor = Fraction(
            proof.floor_numerator,
            proof.floor_denominator,
        )
        assert transferred_floor < 0
        assert exact_floor == algebraic_floor > 0
        assert proof.failure is GalerkinStabilityFailure.NONE
        assert proof.exact_target_residual_finite
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )

    def test_direct_floor_survives_infinite_delta_but_state_lift_fails_typed(
        self,
    ) -> None:
        """Separate matrix proof from a noncertifying residual lift."""
        manifest = _infinite_delta_manifest()
        source, zero_result = _zero_submission(manifest)
        zero_proof = check_galerkin_absorber_floor(
            manifest,
            source,
            zero_result,
            state_budget=1.0,
        )
        zero_invocation = invoke_galerkin_stability(
            manifest,
            source,
            zero_result,
            zero_proof,
            state_budget=1.0,
        )
        nonzero_result = _submitted_result(
            jnp.ones(
                (manifest.support.state_indices.shape[0],),
                dtype=jnp.complex128,
            )
        )
        nonzero_proof = check_galerkin_absorber_floor(
            manifest,
            source,
            nonzero_result,
            state_budget=1.0,
        )
        nonzero_invocation = invoke_galerkin_stability(
            manifest,
            source,
            nonzero_result,
            nonzero_proof,
            state_budget=1.0,
        )

        assert jnp.isinf(
            manifest.fixed_linear_error_ledger.fixed_linear_operator_error_bound
        )
        assert zero_proof.failure is GalerkinStabilityFailure.NONE
        assert not zero_proof.transferred_floor_finite
        assert zero_proof.floor_numerator > 0
        assert zero_proof.exact_target_residual_finite
        assert (
            zero_invocation.disposition
            is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )
        assert nonzero_proof.failure is GalerkinStabilityFailure.NONE
        assert nonzero_proof.floor_numerator > 0
        assert not nonzero_proof.exact_target_residual_finite
        assert (
            nonzero_invocation.disposition
            is GalerkinStabilityDisposition.REJECTED
        )
        assert (
            nonzero_invocation.failure
            is GalerkinStabilityFailure.NO_FINITE_EXACT_TARGET_RESIDUAL_BOUND
        )

    def test_exact_checker_uses_box_floor_above_gershgorin_limit(
        self,
    ) -> None:
        """Certify 33 modes without constructing a Gershgorin proof."""
        manifest = _thirty_three_mode_manifest()
        zeros = jnp.zeros(33, dtype=jnp.complex128)
        source = create_galerkin_source(
            incident_field=zeros,
            incident_source=zeros,
            additional_source=zeros,
            total_source=zeros,
            scattered_source=zeros,
            branch=GalerkinSourceBranch.FINITE_MATCHED,
        )
        solve_result = _submitted_result(zeros)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0
        )
        result = invoke_galerkin_stability(
            manifest,
            source,
            solve_result,
            proof,
            state_budget=1.0,
        )

        assert manifest.support.state_indices.shape == (33, 3)
        expected_absorber_floor = (
            Fraction(1)
            - (Fraction(1) - Fraction(9, 4 * 34**2)) * Fraction(1, 2) ** 2
        )
        expected_floor = Fraction(1, 4) * expected_absorber_floor
        assert (
            Fraction(proof.floor_numerator, proof.floor_denominator)
            == expected_floor
        )
        assert proof.route is GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX
        assert proof.failure is GalerkinStabilityFailure.NONE
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )
        assert result.route is proof.route
        assert result.result_digest == proof.result_digest

    def test_rational_box_floor_is_sharp_on_two_by_two_by_two_box(
        self,
    ) -> None:
        """Match the exact rectangular eigenvalue without trigonometry."""
        state_tuples = list(product(range(2), repeat=3))
        state_indices = np.asarray(state_tuples, dtype=np.int64)
        floor = stability_module._rational_cosine_shell_box_floor(
            state_indices
        )

        differences = state_indices[:, None, :] - state_indices[None, :, :]
        axis_coefficients = np.where(
            differences == 0,
            0.5,
            np.where(np.abs(differences) == 1, 0.25, 0.0),
        )
        cosine_gramian = np.prod(axis_coefficients, axis=-1)
        absorber = np.eye(len(state_tuples)) - cosine_gramian

        assert floor == Fraction(37, 64)
        assert np.linalg.eigvalsh(absorber)[0] == pytest.approx(
            float(floor), abs=2.0e-15
        )

    def test_box_floor_recovers_when_dense_gershgorin_is_zero(self) -> None:
        """Prove positivity for a full three-cube with an interior row."""
        manifest = _manifest_from_state_tuples(
            list(product(range(-1, 2), repeat=3)),
            target_name="three-cube-box-floor",
        )
        source, solve_result = _zero_submission(manifest)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0
        )
        result = invoke_galerkin_stability(
            manifest, source, solve_result, proof, state_budget=1.0
        )

        expected_absorber_floor = Fraction(95769, 262144)
        expected_floor = Fraction(1, 4) * expected_absorber_floor
        assert (
            Fraction(proof.floor_numerator, proof.floor_denominator)
            == expected_floor
        )
        assert proof.route is GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX
        assert proof.failure is GalerkinStabilityFailure.NONE
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )

    def test_sparse_support_retains_stronger_gershgorin_floor(self) -> None:
        """Do not weaken a disconnected sparse support to its large box."""
        manifest = _manifest_from_state_tuples(
            [(0, 0, 0), (2, 2, 2)],
            target_name="sparse-gershgorin-floor",
        )
        source, solve_result = _zero_submission(manifest)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0
        )
        result = invoke_galerkin_stability(
            manifest, source, solve_result, proof, state_budget=1.0
        )

        assert Fraction(
            proof.floor_numerator, proof.floor_denominator
        ) == Fraction(7, 32)
        assert proof.route is GalerkinStabilityRoute.ABSORBER_FLOOR_GERSHGORIN
        assert result.route is proof.route
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )

    def test_huge_span_floor_stays_positive_without_transcendentals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use Python integers and Fractions where floating cosine cancels."""
        coordinate = np.iinfo(np.int64).max // 4
        state_indices = np.asarray(
            [
                (-coordinate, -coordinate, -coordinate),
                (coordinate, coordinate, coordinate),
            ],
            dtype=np.int64,
        )

        def reject_cosine(_value: float) -> float:
            raise AssertionError("the exact checker must not evaluate cosine")

        monkeypatch.setattr(stability_module.math, "cos", reject_cosine)
        floor = stability_module._rational_cosine_shell_box_floor(
            state_indices
        )

        assert floor > 0
        assert float(floor) > 0.0
        assert floor.numerator.bit_length() < 512
        assert floor.denominator.bit_length() < 512

    def test_budget_miss_is_a_typed_fallback(self) -> None:
        """Retain the matrix proof while stripping certification language."""
        manifest, source, solve_result = _case(field_offset=0.1 + 0.05j)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0e-12
        )
        result = invoke_galerkin_stability(
            manifest, source, solve_result, proof, state_budget=1.0e-12
        )

        assert (
            result.disposition is GalerkinStabilityDisposition.TYPED_FALLBACK
        )
        assert result.failure is GalerkinStabilityFailure.STATE_BUDGET_MISSED
        assert float(result.lower_singular_bound) > 0.0
        assert float(result.state_error_upper_bound) > float(
            result.state_budget
        )

    def test_proof_mutation_rejects_instead_of_trusting_producer_fields(
        self,
    ) -> None:
        """Reject a producer mutation of the reconstructed proof."""
        manifest, source, solve_result = _case()
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0e-12
        )
        mutated = replace(proof, target_digest="0" * 64)
        result = invoke_galerkin_stability(
            manifest, source, solve_result, mutated, state_budget=1.0e-12
        )

        assert result.disposition is GalerkinStabilityDisposition.REJECTED
        assert result.failure is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH

    def test_selected_floor_route_is_hash_bound_provenance(self) -> None:
        """Reject relabeling a cosine-box proof as a Gershgorin proof."""
        manifest, source, solve_result = _case()
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0
        )
        mutated = replace(
            proof,
            route=GalerkinStabilityRoute.ABSORBER_FLOOR_GERSHGORIN,
        )
        result = invoke_galerkin_stability(
            manifest, source, solve_result, mutated, state_budget=1.0
        )

        assert proof.route is GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX
        assert result.disposition is GalerkinStabilityDisposition.REJECTED
        assert result.failure is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH

    def test_manifest_and_result_mutations_invalidate_bound_proof(
        self,
    ) -> None:
        """Bind the proof to both the canonical target and submitted state."""
        manifest, source, solve_result = _case()
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0e-12
        )
        changed_manifest = eqx.tree_at(
            lambda target: target.cap_scale,
            manifest,
            jnp.asarray(0.5),
        )
        changed_result = eqx.tree_at(
            lambda result: result.field,
            solve_result,
            solve_result.field.at[2].add(0.01j),
        )
        changed_source = eqx.tree_at(
            lambda value: value.total_source,
            source,
            source.total_source.at[1].add(0.01),
        )
        changed_diagnostic = eqx.tree_at(
            lambda result: result.residual,
            solve_result,
            solve_result.residual.at[0].add(0.01 + 0.02j),
        )

        target_rejection = invoke_galerkin_stability(
            changed_manifest,
            source,
            solve_result,
            proof,
            state_budget=1.0e-12,
        )
        result_rejection = invoke_galerkin_stability(
            manifest,
            source,
            changed_result,
            proof,
            state_budget=1.0e-12,
        )
        source_rejection = invoke_galerkin_stability(
            manifest,
            changed_source,
            solve_result,
            proof,
            state_budget=1.0e-12,
        )
        diagnostic_rejection = invoke_galerkin_stability(
            manifest,
            source,
            changed_diagnostic,
            proof,
            state_budget=1.0e-12,
        )
        assert (
            target_rejection.failure
            is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH
        )
        assert (
            result_rejection.failure
            is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH
        )
        assert (
            source_rejection.failure
            is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH
        )
        assert (
            diagnostic_rejection.failure
            is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH
        )

    def test_target_payload_binds_every_declared_nested_field_once(
        self,
    ) -> None:
        """Bind stored nested fields while omitting duplicate properties."""
        manifest = production_vacuum_target()
        payload = stability_module._target_payload(manifest)

        def assert_complete(module: object, stored: object) -> None:
            """Match each dataclass field recursively to one payload key."""
            assert is_dataclass(module)
            assert isinstance(stored, dict)
            declared = {field.name for field in fields(module)}
            stored_mapping = cast(Dict[str, object], stored)
            stored_fields = cast(Dict[str, object], stored_mapping["fields"])
            assert isinstance(stored_fields, dict)
            assert set(stored_fields) == declared
            for field in fields(module):
                value = getattr(module, field.name)
                if is_dataclass(value) and not isinstance(value, type):
                    assert_complete(value, stored_fields[field.name])

        assert_complete(manifest, payload)
        top_level = cast(Dict[str, object], payload["fields"])
        assert isinstance(top_level, dict)
        assert "support" not in top_level
        assert "potential" not in top_level
        realization_payload = cast(Dict[str, object], top_level["realization"])
        realization_fields = cast(
            Dict[str, object], realization_payload["fields"]
        )
        potential_payload = cast(
            Dict[str, object], realization_fields["potential"]
        )
        potential_fields = cast(Dict[str, object], potential_payload["fields"])
        assert "volume" in potential_fields
        volume_payload = cast(Dict[str, object], potential_fields["volume"])
        array_payload = cast(Dict[str, object], volume_payload["array"])
        assert array_payload["bytes"]

    def test_canonical_rebuild_rejects_forgery_in_every_derived_branch(
        self,
    ) -> None:
        """Rerun every producer instead of trusting nested evidence leaves."""
        manifest = production_vacuum_target()
        changed_potential = eqx.tree_at(
            lambda potential: potential.volume,
            manifest.potential,
            manifest.potential.volume.at[0, 0, 0].set(1.0),
        )
        potential_forgery = replace(
            manifest,
            realization=replace(
                manifest.realization,
                potential=changed_potential,
            ),
        )
        changed_submission = eqx.tree_at(
            lambda acquisition: acquisition.carrier,
            manifest.acquisition,
            manifest.carrier.at[1].set(0.01),
        )
        submission_forgery = replace(
            manifest,
            realization=replace(
                manifest.realization,
                support_eligibility=replace(
                    manifest.support_eligibility,
                    manifest=changed_submission,
                ),
            ),
        )
        result_forgery = replace(
            manifest,
            realization=replace(
                manifest.realization,
                support_eligibility=eqx.tree_at(
                    lambda result: result.support_eligible,
                    manifest.support_eligibility,
                    jnp.asarray(False),
                ),
            ),
        )
        ceiling_forgery = replace(
            manifest,
            realization=replace(
                manifest.realization,
                support_eligibility=replace(
                    manifest.support_eligibility,
                    max_binary_pair_checks=(
                        manifest.support_eligibility.max_binary_pair_checks + 1
                    ),
                ),
            ),
        )
        realization_forgery = eqx.tree_at(
            lambda target: target.realization.coefficient_error_bounds,
            manifest,
            manifest.realization.coefficient_error_bounds.at[0].add(1.0),
        )
        ledger_forgery = eqx.tree_at(
            lambda target: (
                target.fixed_linear_error_ledger.fixed_linear_operator_error_bound
            ),
            manifest,
            manifest.fixed_linear_error_ledger.fixed_linear_operator_error_bound
            + 1.0,
        )
        target_forgery = eqx.tree_at(
            lambda target: target.interaction_coefficients,
            manifest,
            manifest.interaction_coefficients.at[0].add(1.0),
        )

        for forged in (
            potential_forgery,
            submission_forgery,
            result_forgery,
            ceiling_forgery,
            realization_forgery,
            ledger_forgery,
            target_forgery,
        ):
            assert stability_module._target_digest(
                forged
            ) != stability_module._target_digest(manifest)
            assert not stability_module._manifest_is_canonical(forged)

    def test_canonical_rebuild_replays_direct_host_certificate_route(
        self,
    ) -> None:
        """Accept finite direct evidence and preserve typed budget failure."""
        base = production_vacuum_target()
        finite = create_host_checked_galerkin_target(
            base.potential,
            base.support_eligibility,
            accelerating_voltage_kv=base.accelerating_voltage_kv,
            cap_scale=base.cap_scale,
            target_name="host-checked-vacuum",
            maximum_direct_terms=1_000,
        )
        failed = create_host_checked_galerkin_target(
            base.potential,
            base.support_eligibility,
            accelerating_voltage_kv=base.accelerating_voltage_kv,
            cap_scale=base.cap_scale,
            target_name="host-budget-failure-vacuum",
            maximum_direct_terms=1,
        )
        incident = jnp.asarray(
            (0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j),
            dtype=jnp.complex128,
        )
        failed_source = create_matched_galerkin_source(failed, incident)
        failed_result = _submitted_result(incident)
        failed_proof = check_galerkin_absorber_floor(
            failed,
            failed_source,
            failed_result,
            state_budget=1.0,
        )
        failed_invocation = invoke_galerkin_stability(
            failed,
            failed_source,
            failed_result,
            failed_proof,
            state_budget=1.0,
        )

        assert stability_module._manifest_is_canonical(finite)
        assert stability_module._manifest_is_canonical(failed)
        assert not failed_proof.exact_target_residual_finite
        assert (
            failed_invocation.failure
            is GalerkinStabilityFailure.NO_FINITE_EXACT_TARGET_RESIDUAL_BOUND
        )

    def test_canonical_rebuild_rejects_same_shape_cross_potential_ledger(
        self,
    ) -> None:
        """A valid RM-S2 ledger cannot migrate to another voxel target."""
        first = production_target()
        changed_potential = eqx.tree_at(
            lambda potential: potential.volume,
            first.potential,
            1.125 * first.potential.volume,
        )
        second = create_galerkin_target(
            changed_potential,
            first.support_eligibility,
            accelerating_voltage_kv=first.accelerating_voltage_kv,
            cap_scale=first.cap_scale,
            target_name="same-shape-second-potential",
        )
        swapped = replace(
            second,
            fixed_linear_error_ledger=first.fixed_linear_error_ledger,
        )

        assert stability_module._manifest_is_canonical(first)
        assert stability_module._manifest_is_canonical(second)
        assert not stability_module._manifest_is_canonical(swapped)

    def test_proof_budget_mutations_reject_against_preregistration(
        self,
    ) -> None:
        """Reject representable and overflowing proof-budget mutations."""
        manifest, source, solve_result = _case()
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0e-12
        )
        representable_numerator, representable_denominator = (
            1.0e300
        ).as_integer_ratio()
        for numerator, denominator in (
            (representable_numerator, representable_denominator),
            (10**10_000, 1),
        ):
            mutated = replace(
                proof,
                state_budget_numerator=numerator,
                state_budget_denominator=denominator,
            )

            result = invoke_galerkin_stability(
                manifest,
                source,
                solve_result,
                mutated,
                state_budget=1.0e-12,
            )

            assert result.disposition is GalerkinStabilityDisposition.REJECTED
            assert (
                result.failure
                is GalerkinStabilityFailure.PROOF_RECORD_MISMATCH
            )

    def test_exact_bound_reporting_is_outward_and_range_fail_closed(
        self,
    ) -> None:
        """Handle large residual squares and reject nonrepresentable bounds."""
        base = production_vacuum_target()

        def build_manifest(
            cap_scale: jax.Array,
        ) -> GalerkinTargetManifest:
            """Build the canonical target with one dynamic CAP scale."""
            manifest = create_galerkin_target(
                base.potential,
                base.support_eligibility,
                accelerating_voltage_kv=base.accelerating_voltage_kv,
                cap_scale=cap_scale,
                target_name="underflowing-reported-floor",
            )
            return manifest

        subnormal_cap: jax.Array = jnp.nextafter(0.0, 1.0)
        with pytest.raises(
            _RUNTIME_ERRORS,
            match="cap_scale must be finite and preserve",
        ):
            jax.block_until_ready(build_manifest(subnormal_cap))
        with pytest.raises(
            _RUNTIME_ERRORS,
            match="cap_scale must be finite and preserve",
        ):
            jax.block_until_ready(jax.jit(build_manifest)(subnormal_cap))

        zeros = jnp.zeros(3, dtype=jnp.complex128)
        zero_source = create_galerkin_source(
            incident_field=zeros,
            incident_source=zeros,
            additional_source=zeros,
            total_source=zeros,
            scattered_source=zeros,
            branch=GalerkinSourceBranch.FINITE_MATCHED,
        )
        zero_result = _submitted_result(zeros)

        large_field_result = _submitted_result(
            jnp.array([1.0e200 + 0.0j, 0.0, 0.0])
        )
        residual_proof = check_galerkin_absorber_floor(
            base, zero_source, large_field_result, state_budget=1.0
        )
        residual_result = invoke_galerkin_stability(
            base,
            zero_source,
            large_field_result,
            residual_proof,
            state_budget=1.0,
        )
        overflowing_field_result = _submitted_result(
            jnp.array([1.7e308 + 0.0j, 0.0, 0.0])
        )
        overflowing_proof = check_galerkin_absorber_floor(
            base, zero_source, overflowing_field_result, state_budget=1.0
        )
        overflowing_result = invoke_galerkin_stability(
            base,
            zero_source,
            overflowing_field_result,
            overflowing_proof,
            state_budget=1.0,
        )

        assert float(residual_result.residual_upper_bound) > 1.0e199
        assert (
            residual_result.disposition
            is GalerkinStabilityDisposition.TYPED_FALLBACK
        )
        assert (
            overflowing_result.disposition
            is GalerkinStabilityDisposition.REJECTED
        )
        assert (
            overflowing_result.failure
            is GalerkinStabilityFailure.ARITHMETIC_RANGE_FAILURE
        )

    def test_tiny_exact_residual_has_finite_outward_bound(self) -> None:
        """Report a finite upward bound when the residual square underflows."""
        manifest = production_vacuum_target()
        zeros = jnp.zeros(3, dtype=jnp.complex128)
        right_hand_side = jnp.array(
            [1.0e-200 + 0.0j, 0.0, 0.0], dtype=jnp.complex128
        )
        source = create_galerkin_source(
            incident_field=zeros,
            incident_source=zeros,
            additional_source=right_hand_side,
            total_source=right_hand_side,
            scattered_source=right_hand_side,
            branch=GalerkinSourceBranch.FINITE_MATCHED,
        )
        solve_result = _submitted_result(zeros)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=1.0
        )
        result = invoke_galerkin_stability(
            manifest,
            source,
            solve_result,
            proof,
            state_budget=1.0,
        )
        exact_one_component_norm = abs(
            float(np.asarray(right_hand_side)[0].real)
        )

        assert proof.residual_squared_numerator > 0
        assert np.isfinite(float(result.residual_upper_bound))
        assert float(result.residual_upper_bound) >= exact_one_component_norm
        assert float(result.residual_upper_bound) < 2.0e-200
        assert np.isfinite(float(result.state_error_upper_bound))
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )

    def test_subnormal_state_budget_is_rejected_at_the_host_boundary(
        self,
    ) -> None:
        """Reject a positive budget that XLA would flush before invocation."""
        manifest, source, solve_result = _case(field_offset=1.0e-8)
        unsafe_budget = float(1.0e-310)
        with pytest.raises(ValueError, match="smallest normal float64"):
            check_galerkin_absorber_floor(
                manifest,
                source,
                solve_result,
                state_budget=unsafe_budget,
            )
        safe_budget = float(np.finfo(np.float64).tiny)
        proof = check_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget=safe_budget
        )
        result = invoke_galerkin_stability(
            manifest,
            source,
            solve_result,
            proof,
            state_budget=safe_budget,
        )

        assert result.disposition in {
            GalerkinStabilityDisposition.OPERATIONAL_PASS,
            GalerkinStabilityDisposition.TYPED_FALLBACK,
        }

    def test_compiled_manifest_is_canonical_for_exact_invocation(self) -> None:
        """Accept factory-identical eager and compiled target payloads."""
        eager, _, _ = _case()

        def rebuild(volume: jax.Array) -> GalerkinTargetManifest:
            """Rebuild the target through the documented compiled factory."""
            potential = eqx.tree_at(
                lambda candidate: candidate.volume,
                eager.potential,
                volume,
            )
            manifest = create_galerkin_target(
                potential,
                eager.support_eligibility,
                accelerating_voltage_kv=eager.accelerating_voltage_kv,
                cap_scale=eager.cap_scale,
                target_name=eager.target_name,
            )
            return manifest

        compiled = jax.jit(rebuild)(eager.potential.volume)
        incident = jnp.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j])
        source = create_matched_galerkin_source(compiled, incident)
        solve_result = _submitted_result(incident)
        proof = check_galerkin_absorber_floor(
            compiled, source, solve_result, state_budget=1.0
        )

        assert proof.failure is GalerkinStabilityFailure.NONE

    def test_checker_fails_closed_on_wrong_width_carrier_fields(self) -> None:
        """Reject forged arrays outside the canonical binary64 payload."""
        manifest, source, solve_result = _case()
        narrow_manifest = eqx.tree_at(
            lambda target: (
                target.realization.support_eligibility.manifest.carrier
            ),
            manifest,
            manifest.carrier.astype(jnp.float32),
        )
        narrow_source = eqx.tree_at(
            lambda value: value.total_source,
            source,
            source.total_source.astype(jnp.complex64),
        )
        narrow_result = eqx.tree_at(
            lambda result: result.field,
            solve_result,
            solve_result.field.astype(jnp.complex64),
        )

        operator_proof = check_galerkin_absorber_floor(
            narrow_manifest,
            source,
            solve_result,
            state_budget=1.0,
        )
        source_proof = check_galerkin_absorber_floor(
            manifest,
            narrow_source,
            solve_result,
            state_budget=1.0,
        )
        result_proof = check_galerkin_absorber_floor(
            manifest,
            source,
            narrow_result,
            state_budget=1.0,
        )

        assert (
            operator_proof.failure
            is GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT
        )
        assert (
            source_proof.failure
            is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
        )
        assert (
            result_proof.failure
            is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
        )

    def test_wrong_width_result_fails_closed_without_pytest_import_hook(
        self,
    ) -> None:
        """Exercise the dtype guard in an undecorated interpreter."""
        script = """
            import equinox as eqx
            import jax.numpy as jnp

            from ptyrodactyl.galerkin import check_galerkin_absorber_floor
            from ptyrodactyl.types import GalerkinStabilityFailure
            from tests.test_ptyrodactyl.test_galerkin.test_stability import (
                _case,
            )

            manifest, source, solve_result = _case()
            narrow_result = eqx.tree_at(
                lambda result: result.field,
                solve_result,
                solve_result.field.astype(jnp.complex64),
            )
            proof = check_galerkin_absorber_floor(
                manifest,
                source,
                narrow_result,
                state_budget=1.0,
            )
            assert (
                proof.failure
                is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
            )
            print("wrong-width-rejected")
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            check=False,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "wrong-width-rejected" in result.stdout

    def test_fresh_invalid_manifest_and_submission_fail_closed(self) -> None:
        """Reject invalid fresh inputs rather than certifying their product."""
        manifest, source, solve_result = _case()
        invalid_manifest = eqx.tree_at(
            lambda target: (
                target.cap_scale,
                target.absorber_coefficients,
            ),
            manifest,
            (
                jnp.asarray(-0.25),
                -manifest.absorber_coefficients,
            ),
        )
        invalid_result = eqx.tree_at(
            lambda result: result.field,
            solve_result,
            solve_result.field.at[0].set(jnp.nan + 0.0j),
        )
        invalid_source = eqx.tree_at(
            lambda value: value.incident_field,
            source,
            source.incident_field.at[0].set(jnp.nan + 0.0j),
        )
        invalid_status = eqx.tree_at(
            lambda result: result.status,
            solve_result,
            jnp.asarray(999, dtype=jnp.int32),
        )
        inconsistent_convergence = eqx.tree_at(
            lambda result: result.converged,
            solve_result,
            jnp.asarray(False),
        )
        invalid_reported_residual = eqx.tree_at(
            lambda result: result.residual,
            solve_result,
            solve_result.residual.at[0].set(jnp.nan + 0.0j),
        )
        invalid_precision = replace(manifest, precision="mutated-precision")

        operator_proof = check_galerkin_absorber_floor(
            invalid_manifest, source, solve_result, state_budget=1.0
        )
        submission_proof = check_galerkin_absorber_floor(
            manifest, source, invalid_result, state_budget=1.0
        )
        operator_result = invoke_galerkin_stability(
            invalid_manifest,
            source,
            solve_result,
            operator_proof,
            state_budget=1.0,
        )
        submission_result = invoke_galerkin_stability(
            manifest,
            source,
            invalid_result,
            submission_proof,
            state_budget=1.0,
        )
        precision_proof = check_galerkin_absorber_floor(
            invalid_precision, source, solve_result, state_budget=1.0
        )

        assert (
            operator_result.failure
            is GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT
        )
        assert (
            submission_result.failure
            is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
        )
        assert (
            precision_proof.failure
            is GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT
        )
        malformed_submissions = (
            (invalid_source, solve_result),
            (source, invalid_status),
            (source, inconsistent_convergence),
            (source, invalid_reported_residual),
        )
        for malformed_source, malformed_result in malformed_submissions:
            malformed_proof = check_galerkin_absorber_floor(
                manifest,
                malformed_source,
                malformed_result,
                state_budget=1.0,
            )
            malformed_invocation = invoke_galerkin_stability(
                manifest,
                malformed_source,
                malformed_result,
                malformed_proof,
                state_budget=1.0,
            )
            assert (
                malformed_proof.failure
                is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
            )
            assert (
                malformed_invocation.failure
                is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
            )


# fmt: off
class TestRepresentedSourceStabilityInvocation:
    """Verify the explicit RM-S3 source-error stability route.

:see: :func:`ptyrodactyl.galerkin.check_represented_galerkin_absorber_floor`
:see: :func:`ptyrodactyl.galerkin.invoke_represented_galerkin_stability`
    """
# fmt: on

    def test_eligible_source_replays_and_encloses_rational_dense_oracles(
        self,
    ) -> None:
        """Add ``delta_S`` once and enclose exact-rational/dense checks."""
        manifest = _represented_manifest()
        source = _represented_plane_source(manifest)
        zeros = jnp.zeros_like(source.actions.total_source)
        solve_result = _submitted_result(zeros)
        state_budget = 1.0e100

        proof = check_represented_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget
        )
        replay = check_represented_galerkin_absorber_floor(
            manifest, source, solve_result, state_budget
        )
        result = invoke_represented_galerkin_stability(
            manifest, source, solve_result, proof, state_budget
        )

        assert proof.failure is GalerkinStabilityFailure.NONE
        assert stability_module._proof_payload(proof) == (
            stability_module._proof_payload(replay)
        )
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )
        source_error = Fraction.from_float(
            float(
                source.error_enclosure.exact_target_total_source_error_upper_bound
            )
        )
        assert (
            Fraction(
                proof.source_error_upper_numerator,
                proof.source_error_upper_denominator,
            )
            == source_error
        )
        assert proof.source_error_finite
        assert proof.source_error_route == source.error_enclosure.route.value
        assert proof.source_error_scope == source.error_enclosure.error_scope
        assert proof.rhs_target == stability_module._REPRESENTED_RHS_TARGET
        assert proof.residual_scope == (
            stability_module._REPRESENTED_RESIDUAL_SCOPE
        )

        _, _, exact_algebraic_target = stability_module._target_matrices(
            manifest
        )
        residual_squared = stability_module._residual_squared(
            exact_algebraic_target,
            source.actions.total_source,
            solve_result,
        )
        field_norm_squared = stability_module._vector_norm_squared(
            solve_result.field
        )
        expected_residual, expected_finite = (
            stability_module._lift_exact_target_residual_up(
                residual_squared,
                field_norm_squared,
                float(
                    manifest.fixed_linear_error_ledger.fixed_linear_operator_error_bound
                ),
                source_error,
                True,
            )
        )
        assert expected_finite
        assert (
            Fraction(
                proof.exact_target_residual_upper_numerator,
                proof.exact_target_residual_upper_denominator,
            )
            == expected_residual
        )

        dense_state = np.linalg.solve(
            _dense_target(manifest), np.asarray(source.actions.total_source)
        )
        assert np.linalg.norm(dense_state) <= float(
            result.state_error_upper_bound
        )

    def test_forged_represented_source_fails_full_payload_rebuild(
        self,
    ) -> None:
        """Reject a total-source mutation despite otherwise intact evidence."""
        manifest = _represented_manifest()
        source = _represented_plane_source(manifest)
        forged = eqx.tree_at(
            lambda value: value.actions.total_source,
            source,
            source.actions.total_source.at[0].add(0.25 + 0.125j),
        )
        solve_result = _submitted_result(
            jnp.zeros_like(source.actions.total_source)
        )

        proof = check_represented_galerkin_absorber_floor(
            manifest, forged, solve_result, 1.0
        )
        result = invoke_represented_galerkin_stability(
            manifest, forged, solve_result, proof, 1.0
        )

        assert (
            proof.failure
            is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
        )
        assert (
            result.failure
            is GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT
        )

    def test_canonical_noneligible_source_has_typed_rejection(self) -> None:
        """Distinguish valid represented evidence from RM-S3 eligibility."""
        manifest = _represented_manifest()
        weights = jnp.zeros(
            (manifest.support.state_indices.shape[0],),
            dtype=jnp.complex128,
        )
        weights = weights.at[_represented_position(manifest, (0, 0, 0))].set(
            1.0 + 0.25j
        )
        weights = weights.at[_represented_position(manifest, (1, 0, -1))].set(
            -0.5 + 0.75j
        )
        source = build_represented_focused_galerkin_source(
            manifest=manifest,
            aperture_weights=weights,
            target_reduced_flux=jnp.asarray(3.0, dtype=jnp.float64),
            aberration_phases=jnp.zeros_like(jnp.real(weights)),
            **_represented_source_kwargs(),
        )
        jax.block_until_ready(source)
        solve_result = _submitted_result(
            jnp.zeros_like(source.actions.total_source)
        )

        proof = check_represented_galerkin_absorber_floor(
            manifest, source, solve_result, 1.0
        )
        result = invoke_represented_galerkin_stability(
            manifest, source, solve_result, proof, 1.0
        )

        assert not source.rm_s3_eligible
        assert (
            proof.failure is GalerkinStabilityFailure.SOURCE_NOT_RM_S3_ELIGIBLE
        )
        assert (
            result.failure
            is GalerkinStabilityFailure.SOURCE_NOT_RM_S3_ELIGIBLE
        )

    def test_nonfinite_source_error_has_typed_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail closed when the rebuilt source has no finite ``delta_S``."""
        monkeypatch.setattr(
            sources_module,
            "_all_normal_arithmetic_supported",
            lambda: jnp.asarray(False, dtype=jnp.bool_),
        )
        manifest = _represented_manifest()
        source = _represented_plane_source(manifest)
        solve_result = _submitted_result(
            jnp.zeros_like(source.actions.total_source)
        )

        proof = check_represented_galerkin_absorber_floor(
            manifest, source, solve_result, 1.0
        )
        result = invoke_represented_galerkin_stability(
            manifest, source, solve_result, proof, 1.0
        )

        assert jnp.isinf(
            source.error_enclosure.exact_target_total_source_error_upper_bound
        )
        assert not proof.source_error_finite
        expected_failure = (
            GalerkinStabilityFailure.NO_FINITE_EXACT_TARGET_SOURCE_ERROR_BOUND
        )
        assert proof.failure is expected_failure
        assert result.failure is expected_failure
