"""Tests for :mod:`ptyrodactyl.born.stability`."""

from collections.abc import Callable
from dataclasses import replace
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptyrodactyl.born import (
    cgls_solve,
    check_galerkin_absorber_floor,
    create_matched_galerkin_source,
    invoke_galerkin_stability,
    lsqr_solve,
)
from ptyrodactyl.tools import relativistic_wavelength_ang
from ptyrodactyl.types import (
    GalerkinCertificateReason,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    GalerkinSource,
    GalerkinSourceBranch,
    GalerkinStabilityDisposition,
    GalerkinStabilityFailure,
    GalerkinStabilityProof,
    GalerkinStabilityResult,
    GalerkinTargetManifest,
    create_galerkin_product_support,
    create_galerkin_solve_result,
    create_galerkin_source,
    create_galerkin_stability_proof,
    create_galerkin_target_manifest,
)

from .test_system import _dense_target, _manifest

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
) -> tuple[GalerkinTargetManifest, GalerkinSource, GalerkinSolveResult]:
    """Create one exact finite vacuum result for Route-A invocation."""
    manifest = _manifest(interaction=False)
    incident = jnp.array(
        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        dtype=jnp.complex128,
    )
    source = create_matched_galerkin_source(manifest, incident)
    field = incident.at[0].add(field_offset)
    solve_result = _submitted_result(field)
    return manifest, source, solve_result


def _dimension_limit_manifest() -> GalerkinTargetManifest:
    """Create one canonical target immediately above the exact bound."""
    state_tuples = [(index, 0, 0) for index in range(-16, 17)]
    absorber_tuples = set((index, 0, 0) for index in range(-32, 33)) | set(
        product(range(-1, 2), repeat=3)
    )
    work_tuples = {
        tuple(
            left_axis + right_axis
            for left_axis, right_axis in zip(left, right)
        )
        for left in state_tuples
        for right in absorber_tuples
    }
    support = create_galerkin_product_support(
        state_indices=jnp.asarray(state_tuples),
        interaction_indices=jnp.asarray([[0, 0, 0]]),
        absorber_indices=jnp.asarray(sorted(absorber_tuples)),
        work_indices=jnp.asarray(sorted(work_tuples)),
        work_shape=(97, 3, 3),
    )
    voltage_kv = jnp.asarray(200.0)
    wavenumber = 2.0 * jnp.pi / relativistic_wavelength_ang(voltage_kv)
    manifest = create_galerkin_target_manifest(
        support=support,
        preterminal_indices=support.state_indices,
        voltage_coefficients=jnp.zeros(1, dtype=jnp.complex128),
        box_lengths=jnp.asarray([5.0, 6.0, 7.0]),
        carrier=jnp.asarray([0.0, 0.0, wavenumber]),
        accelerating_voltage_kv=voltage_kv,
        cap_scale=jnp.asarray(0.25),
        target_name="dimension-limit-boundary",
    )
    return manifest


class TestGalerkinStabilityInvocation:
    """Verify exact Route-A checking and per-result dispositions.

    :see: :func:`ptyrodactyl.born.check_galerkin_absorber_floor`
    :see: :func:`ptyrodactyl.born.invoke_galerkin_stability`
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
        assert float(result.lower_singular_bound) == 0.1875
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
        preregistered_state_budget = 1.0e-9
        manifest = _manifest()
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

    def test_exact_checker_rejects_canonical_thirty_three_mode_target(
        self,
    ) -> None:
        """Fail closed immediately above the 32-mode exact checker bound."""
        manifest = _dimension_limit_manifest()
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
        assert (
            proof.failure is GalerkinStabilityFailure.CHECKER_DIMENSION_LIMIT
        )
        assert result.disposition is GalerkinStabilityDisposition.REJECTED
        assert (
            result.failure is GalerkinStabilityFailure.CHECKER_DIMENSION_LIMIT
        )
        assert result.result_digest == proof.result_digest

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
        mutated = create_galerkin_stability_proof(
            target_digest="0" * 64,
            result_digest=proof.result_digest,
            floor_numerator=proof.floor_numerator,
            floor_denominator=proof.floor_denominator,
            residual_squared_numerator=proof.residual_squared_numerator,
            residual_squared_denominator=proof.residual_squared_denominator,
            state_budget_numerator=proof.state_budget_numerator,
            state_budget_denominator=proof.state_budget_denominator,
            route=proof.route,
            failure=proof.failure,
            checker_id=proof.checker_id,
        )
        result = invoke_galerkin_stability(
            manifest, source, solve_result, mutated, state_budget=1.0e-12
        )

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
            mutated = create_galerkin_stability_proof(
                target_digest=proof.target_digest,
                result_digest=proof.result_digest,
                floor_numerator=proof.floor_numerator,
                floor_denominator=proof.floor_denominator,
                residual_squared_numerator=proof.residual_squared_numerator,
                residual_squared_denominator=(
                    proof.residual_squared_denominator
                ),
                state_budget_numerator=numerator,
                state_budget_denominator=denominator,
                route=proof.route,
                failure=proof.failure,
                checker_id=proof.checker_id,
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
        base = _manifest(interaction=False)

        def build_manifest(
            cap_scale: jax.Array,
        ) -> GalerkinTargetManifest:
            """Build the canonical target with one dynamic CAP scale."""
            manifest = create_galerkin_target_manifest(
                support=base.support,
                preterminal_indices=base.preterminal_indices,
                voltage_coefficients=base.voltage_coefficients,
                box_lengths=base.box_lengths,
                carrier=base.carrier,
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
        manifest = _manifest(interaction=False)
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

        def rebuild(phi: jax.Array) -> GalerkinTargetManifest:
            """Rebuild the target through the documented compiled factory."""
            manifest = create_galerkin_target_manifest(
                support=eager.support,
                preterminal_indices=eager.preterminal_indices,
                voltage_coefficients=phi,
                box_lengths=eager.box_lengths,
                carrier=eager.carrier,
                accelerating_voltage_kv=eager.accelerating_voltage_kv,
                cap_scale=eager.cap_scale,
                target_name=eager.target_name,
            )
            return manifest

        compiled = jax.jit(rebuild)(eager.voltage_coefficients)
        incident = jnp.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j])
        source = create_matched_galerkin_source(compiled, incident)
        solve_result = _submitted_result(incident)
        proof = check_galerkin_absorber_floor(
            compiled, source, solve_result, state_budget=1.0
        )

        assert proof.failure is GalerkinStabilityFailure.NONE

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
