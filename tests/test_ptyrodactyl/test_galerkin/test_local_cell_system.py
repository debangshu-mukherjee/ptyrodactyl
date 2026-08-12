r"""Tests for the completed solver-ready ``LOCAL_CELL_LVT1`` target."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.engine as galerkin_engine
import ptyrodactyl.galerkin.local_cell_system as local_system
from ptyrodactyl._tools import stored_value_payload, upward_add
from ptyrodactyl.galerkin.absorber import certify_axial_cap_floor
from ptyrodactyl.galerkin.acquisition import (
    check_galerkin_acquisition_support,
)
from ptyrodactyl.galerkin.free_geometry import (
    FreeGeometryEnclosure,
    enclose_free_geometry,
    transfer_exact_carrier_acquisition,
)
from ptyrodactyl.galerkin.local_cell_system import (
    apply_local_cell_galerkin_target,
    apply_local_cell_galerkin_target_adjoint,
    compose_local_cell_galerkin_target,
    prepare_local_cell_galerkin_target,
)
from ptyrodactyl.types import C_LIGHT, E_CHARGE, HBAR, M_E
from ptyrodactyl.types.acquisition_types import (
    GalerkinBackwardDisposition,
    GalerkinTerminalSide,
)
from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellTargetManifest,
)
from tests._galerkin_target_fixture import (
    checked_acquisition,
    production_target,
)
from tests.test_ptyrodactyl.test_galerkin.test_absorber import (
    _successful_cap_fixture,
)
from tests.test_ptyrodactyl.test_galerkin.test_acquisition import _manifest


@pytest.fixture(scope="module")
def cap_proof():
    """Return one shared fully replayed L4 proof."""
    return _successful_cap_fixture()[2]


@pytest.fixture(scope="module")
def local_target(cap_proof) -> GalerkinLocalCellTargetManifest:
    """Compose one public target once for this focused wall."""
    return compose_local_cell_galerkin_target(
        cap_proof,
        target_name="  local-cell-system-test  ",
    )


def _dense_terms(
    target: GalerkinLocalCellTargetManifest,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build independent dense D/R/B/H matrices from frozen pair maps."""
    compression = target.compression
    cap_certificate = target.cap_floor_proof.coefficient_certificate
    state_count = target.state_indices.shape[0]
    interaction_positions = np.asarray(
        compression.state_pair_interaction_positions
    )
    absorber_positions = np.asarray(
        cap_certificate.state_pair_absorber_positions
    )
    interaction_coefficients = np.asarray(compression.interaction_coefficients)
    absorber_coefficients = np.asarray(target.absorber_coefficients)
    scale = float(target.algebraic_cap_scale)
    interaction = np.asarray(
        [
            [
                interaction_coefficients[
                    interaction_positions[row * state_count + column]
                ]
                for column in range(state_count)
            ]
            for row in range(state_count)
        ],
        dtype=np.complex128,
    )
    absorber = np.asarray(
        [
            [
                scale
                * absorber_coefficients[
                    absorber_positions[row * state_count + column]
                ]
                for column in range(state_count)
            ]
            for row in range(state_count)
        ],
        dtype=np.complex128,
    )
    free = np.diag(np.asarray(target.free_diagonal, dtype=np.complex128))
    operator = free - interaction - 1j * absorber
    return free, interaction, absorber, operator


def _exact_decimal_k0(voltage_kv: float) -> Decimal:
    """Evaluate exact stored-input SC.2 independently with Decimal."""
    with localcontext() as context:
        context.prec = 100
        voltage = Decimal.from_float(voltage_kv) * Decimal(1000)
        mass = Decimal.from_float(float(np.asarray(M_E)))
        charge = Decimal.from_float(float(np.asarray(E_CHARGE)))
        speed = Decimal.from_float(float(np.asarray(C_LIGHT)))
        hbar = Decimal.from_float(float(np.asarray(HBAR)))
        squared = (
            Decimal(2)
            * mass
            * charge
            * voltage
            / (hbar * hbar)
            * (
                Decimal(1)
                + charge * voltage / (Decimal(2) * mass * speed * speed)
            )
            / (Decimal(10) ** 20)
        )
        return +squared.sqrt()


def _zero_error_geometry(
    acquisition,
) -> FreeGeometryEnclosure:
    """Create a route-neutral zero-error carrier transfer fixture."""
    state_count = acquisition.manifest.support.state_indices.shape[0]
    zeros_state = jnp.zeros((state_count,), dtype=jnp.float64)
    zeros_carrier = jnp.zeros((3,), dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return FreeGeometryEnclosure(
        algebraic_free_diagonal=zeros_state,
        exact_wavenumber_lower_bound=one,
        exact_wavenumber_upper_bound=one,
        wavenumber_error_bound=zero,
        exact_carrier_lower_bounds=zeros_carrier,
        exact_carrier_upper_bounds=zeros_carrier,
        carrier_component_error_bounds=zeros_carrier,
        exact_free_diagonal_lower_bounds=zeros_state,
        exact_free_diagonal_upper_bounds=zeros_state,
        free_diagonal_error_bounds=zeros_state,
        free_operator_error_bound=zero,
        exact_geometry_target="test exact geometry",
        algebraic_geometry_realization="test algebraic geometry",
        free_geometry_digest="1" * 64,
    )


def test_exact_k0_carrier_free_diagonal_and_binary64_index_gate(
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Check SC.2/SC.8/SC.23 and enforce the canonical 2^52 bound."""
    ledger = local_target.fixed_linear_error_ledger
    exact_k0 = _exact_decimal_k0(float(local_target.accelerating_voltage_kv))
    assert (
        Decimal.from_float(float(ledger.exact_wavenumber_lower_bound))
        <= exact_k0
        <= Decimal.from_float(float(ledger.exact_wavenumber_upper_bound))
    )
    assert (
        Decimal.from_float(float(ledger.exact_carrier_lower_bounds[0]))
        <= exact_k0
        <= Decimal.from_float(float(ledger.exact_carrier_upper_bounds[0]))
    )
    assert_array_equal(ledger.exact_carrier_lower_bounds[1:], [0.0, 0.0])
    assert_array_equal(ledger.exact_carrier_upper_bounds[1:], [0.0, 0.0])

    with localcontext() as context:
        context.prec = 100
        pi = Decimal(
            "3.141592653589793238462643383279502884197169399375105820974944"
        )
        length = Decimal.from_float(float(local_target.box_lengths[0]))
        for position, row in enumerate(np.asarray(local_target.state_indices)):
            mode = Decimal(int(row[0]))
            offset = mode / length
            exact_d = (
                Decimal(4) * pi * exact_k0 * offset
                + Decimal(4) * pi * pi * offset * offset
            )
            lower = Decimal.from_float(
                float(ledger.exact_free_diagonal_lower_bounds[position])
            )
            upper = Decimal.from_float(
                float(ledger.exact_free_diagonal_upper_bounds[position])
            )
            assert lower <= exact_d <= upper
            stored = Decimal.from_float(
                float(ledger.algebraic_free_diagonal[position])
            )
            error = Decimal.from_float(
                float(ledger.free_diagonal_error_bounds[position])
            )
            assert abs(stored - exact_d) <= error

    acquisition = local_target.support_eligibility
    exact_limit = jnp.asarray(
        [[2**52, 0, 0], [-(2**52), 0, 0]], dtype=jnp.int64
    )
    accepted = enclose_free_geometry(
        exact_limit,
        acquisition,
        local_target.accelerating_voltage_kv,
    )
    assert bool(jnp.all(jnp.isfinite(accepted.algebraic_free_diagonal)))
    for unsafe_value in (2**52 + 1, np.iinfo(np.int64).max):
        unsafe = jnp.asarray([[unsafe_value, 0, 0]], dtype=jnp.int64)
        with pytest.raises(ValueError, match="safe-index bound"):
            enclose_free_geometry(
                unsafe,
                acquisition,
                local_target.accelerating_voltage_kv,
            )


def test_exact_carrier_transfer_covers_projection_and_all_sectors(
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Preserve symbolic exact rows and reject ambiguous/rounded sectors."""
    geometry = enclose_free_geometry(
        local_target.state_indices,
        local_target.support_eligibility,
        local_target.accelerating_voltage_kv,
    )
    exact = transfer_exact_carrier_acquisition(
        local_target.support_eligibility,
        geometry,
    )
    assert_array_equal(exact.incident_shell_defect_bounds, [0.0])
    assert_array_equal(exact.outgoing_shell_defect_bounds, [0.0])
    assert_array_equal(exact.incident_projection_error_bounds, [0.0])
    assert_array_equal(exact.outgoing_projection_error_bounds, [0.0])

    projected_acquisition = checked_acquisition(
        local_target.support,
        (
            float(local_target.box_lengths[0]),
            float(local_target.box_lengths[1]),
            float(local_target.box_lengths[2]),
        ),
        projected_offset=(1.0e-4, 0.0, 0.0),
    )
    projected_geometry = enclose_free_geometry(
        local_target.state_indices,
        projected_acquisition,
        local_target.accelerating_voltage_kv,
    )
    projected = transfer_exact_carrier_acquisition(
        projected_acquisition,
        projected_geometry,
    )
    assert float(projected.incident_projection_error_bounds[0]) > 0.0
    assert float(projected.incident_shell_defect_bounds[0]) > 0.0

    transverse = jnp.asarray([10.0, 0.0, 0.0], dtype=jnp.float64)
    grazing = check_galerkin_acquisition_support(
        _manifest(
            carrier=transverse,
            wavenumber=jnp.asarray(10.0, dtype=jnp.float64),
            disposition=GalerkinBackwardDisposition.REPRESENTED,
            exclusion_basis="",
        )
    )
    grazing_geometry = _zero_error_geometry(grazing)
    transfer_exact_carrier_acquisition(grazing, grazing_geometry)
    perturbed_geometry = grazing_geometry._replace(
        carrier_component_error_bounds=jnp.asarray(
            [0.0, 0.0, jnp.finfo(jnp.float64).tiny],
            dtype=jnp.float64,
        )
    )
    with pytest.raises(ValueError, match="represented sector"):
        transfer_exact_carrier_acquisition(grazing, perturbed_geometry)

    omitted = check_galerkin_acquisition_support(
        _manifest(
            carrier=jnp.asarray([0.0, 0.0, 10.0], dtype=jnp.float64),
            wavenumber=jnp.asarray(10.0, dtype=jnp.float64),
            omitted=jnp.asarray([[0, 0, -2]], dtype=jnp.int64),
        )
    )
    assert bool(omitted.support_eligible)
    assert bool(omitted.omitted_backward_mask[0])
    transfer_exact_carrier_acquisition(omitted, _zero_error_geometry(omitted))

    for unsafe_value in (2**52 + 1, np.iinfo(np.int64).max):
        unsafe_omitted = eqx.tree_at(
            lambda value: value.manifest.deliberately_omitted_indices,
            local_target.support_eligibility,
            jnp.asarray([[0, 0, unsafe_value]], dtype=jnp.int64),
        )
        with pytest.raises(ValueError, match="deliberately_omitted_indices"):
            transfer_exact_carrier_acquisition(unsafe_omitted, geometry)

    ambiguous = check_galerkin_acquisition_support(
        _manifest(
            carrier=jnp.asarray([0.0, 0.0, 2.0 * np.pi], dtype=jnp.float64),
            wavenumber=jnp.asarray(2.0 * np.pi, dtype=jnp.float64),
        )
    )
    with pytest.raises(ValueError, match="eligible acquisition"):
        transfer_exact_carrier_acquisition(
            ambiguous,
            _zero_error_geometry(ambiguous),
        )


def test_dense_forward_adjoint_dot_jit_and_vjp(
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Match independent dense H/H-star and JAX complex VJP convention.

    :see: :func:`ptyrodactyl.galerkin.apply_local_cell_galerkin_target`
    :see: :func:`ptyrodactyl.galerkin.apply_local_cell_galerkin_target_adjoint`
    """
    _, _, _, dense = _dense_terms(local_target)
    field = jnp.asarray(
        [1.0 + 0.2j, -0.5 + 0.7j, 0.3 - 0.4j], dtype=jnp.complex128
    )
    cotangent = jnp.asarray(
        [-0.2 + 0.8j, 1.1 - 0.3j, 0.4 + 0.6j], dtype=jnp.complex128
    )
    forward = apply_local_cell_galerkin_target(local_target, field)
    adjoint = apply_local_cell_galerkin_target_adjoint(local_target, cotangent)
    assert_allclose(forward, dense @ np.asarray(field), rtol=3e-15, atol=3e-13)
    assert_allclose(
        adjoint,
        dense.conj().T @ np.asarray(cotangent),
        rtol=3e-15,
        atol=3e-13,
    )
    assert_allclose(
        jnp.vdot(forward, cotangent),
        jnp.vdot(field, adjoint),
        rtol=4e-15,
        atol=4e-13,
    )

    def closed(value):
        """Close over the prepared target for JAX transformation."""
        return apply_local_cell_galerkin_target(local_target, value)

    assert_allclose(jax.jit(closed)(field), forward, rtol=0.0, atol=0.0)
    _, pullback = jax.vjp(closed, field)
    vjp_adjoint = jnp.conj(pullback(jnp.conj(cotangent))[0])
    assert_allclose(vjp_adjoint, adjoint, rtol=4e-15, atol=4e-13)


def test_engine_dispatch_and_original_residual_match_dense(
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Keep LOCAL_CELL_LVT1 disjoint from legacy and sparse COO routes."""
    _, _, _, dense = _dense_terms(local_target)
    field = jnp.asarray(
        [0.4 + 0.1j, -0.2 + 0.3j, 0.15 - 0.25j],
        dtype=jnp.complex128,
    )
    source = jnp.asarray(
        [1.1 - 0.2j, -0.7 + 0.4j, 0.6 + 0.8j],
        dtype=jnp.complex128,
    )
    expected_forward = dense @ np.asarray(field)
    expected_adjoint = dense.conj().T @ np.asarray(field)
    forward = galerkin_engine.apply_galerkin_operator(local_target, field)
    adjoint = galerkin_engine.apply_galerkin_adjoint(local_target, field)
    assert_allclose(forward, expected_forward, rtol=3e-15, atol=3e-13)
    assert_allclose(adjoint, expected_adjoint, rtol=3e-15, atol=3e-13)
    assert_allclose(
        jax.jit(
            lambda value: galerkin_engine.apply_galerkin_operator(
                local_target, value
            )
        )(field),
        expected_forward,
        rtol=3e-15,
        atol=3e-13,
    )
    assert_allclose(
        jax.jit(
            lambda value: galerkin_engine.apply_galerkin_adjoint(
                local_target, value
            )
        )(field),
        expected_adjoint,
        rtol=3e-15,
        atol=3e-13,
    )

    expected_forward_residual = np.asarray(source) - expected_forward
    expected_adjoint_residual = np.asarray(source) - expected_adjoint
    forward_residual, forward_norm = (
        galerkin_engine.evaluate_galerkin_residual(local_target, field, source)
    )
    adjoint_residual, adjoint_norm = (
        galerkin_engine.evaluate_galerkin_adjoint_residual(
            local_target, field, source
        )
    )
    assert_allclose(
        forward_residual,
        expected_forward_residual,
        rtol=3e-15,
        atol=3e-13,
    )
    assert_allclose(
        adjoint_residual,
        expected_adjoint_residual,
        rtol=3e-15,
        atol=3e-13,
    )
    assert_allclose(forward_norm, np.linalg.norm(expected_forward_residual))
    assert_allclose(adjoint_norm, np.linalg.norm(expected_adjoint_residual))

    for use_adjoint, expected in (
        (False, expected_forward_residual),
        (True, expected_adjoint_residual),
    ):
        residual, norm = galerkin_engine._solver_original_residual(
            local_target,
            field,
            source,
            adjoint=use_adjoint,
        )
        assert_allclose(residual, expected, rtol=3e-15, atol=3e-13)
        assert_allclose(norm, np.linalg.norm(expected))


def test_engine_cgls_and_lsqr_match_dense_local_target(
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Solve the nontrivial prepared local target through both engines."""
    _, _, _, dense = _dense_terms(local_target)
    source = jnp.asarray(
        [0.7 - 0.2j, -0.4 + 0.6j, 0.3 + 0.1j],
        dtype=jnp.complex128,
    )
    expected = np.linalg.solve(dense, np.asarray(source))
    for solver in (galerkin_engine.cgls_solve, galerkin_engine.lsqr_solve):
        result = solver(
            local_target,
            source,
            max_iterations=96,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
        )
        assert bool(result.converged), solver.__name__
        assert_allclose(result.field, expected, rtol=3e-8, atol=3e-9)
        expected_residual = np.asarray(source) - dense @ np.asarray(
            result.field
        )
        assert_allclose(
            result.residual,
            expected_residual,
            rtol=2e-8,
            atol=2e-10,
        )
        assert_allclose(
            result.residual_norm,
            np.linalg.norm(expected_residual),
            rtol=2e-8,
            atol=2e-10,
        )


def test_engine_implicit_source_vjp_accepts_prepared_local_target(
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Exercise the primal/custom-VJP PyTree seam with LOCAL_CELL_LVT1."""
    _, _, _, dense = _dense_terms(local_target)
    source = jnp.asarray(
        [0.7 - 0.2j, -0.4 + 0.6j, 0.3 + 0.1j],
        dtype=jnp.complex128,
    )

    def source_loss(candidate_source):
        """Return a real dense-state norm through the implicit root."""
        field = galerkin_engine.implicit_galerkin_solve(
            local_target,
            candidate_source,
            max_iterations=96,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
        )
        return 0.5 * jnp.real(jnp.vdot(field, field))

    def dense_loss(candidate_source: np.ndarray) -> float:
        """Return the independent dense counterpart of ``source_loss``."""
        field = np.linalg.solve(dense, candidate_source)
        return float(0.5 * np.real(np.vdot(field, field)))

    gradient = jax.grad(source_loss)(source)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    step = 2.0e-4
    for direction in (
        np.asarray([0.0, 1.0, 0.0], dtype=np.complex128),
        np.asarray([0.0, 1.0j, 0.0], dtype=np.complex128),
    ):
        finite_difference = (
            dense_loss(np.asarray(source) + step * direction)
            - dense_loss(np.asarray(source) - step * direction)
        ) / (2.0 * step)
        directional = float(
            jnp.real(jnp.sum(gradient * jnp.asarray(direction)))
        )
        assert_allclose(
            directional,
            finite_difference,
            rtol=5e-5,
            atol=5e-7,
        )


def test_delta_total_once_and_floor_failure_remains_solver_ready(
    cap_proof,
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Charge D/R/B once and admit a finite-B Gram noncertificate.

    :see: :func:`ptyrodactyl.galerkin.compose_local_cell_galerkin_target`
    """
    ledger = local_target.fixed_linear_error_ledger
    expected = upward_add(
        upward_add(
            ledger.free_operator_error_bound,
            ledger.interaction_operator_error_bound,
        ),
        ledger.cap_operator_error_bound,
    )
    assert float(ledger.fixed_linear_operator_error_bound) == float(expected)
    assert float(ledger.interaction_operator_error_bound) == float(
        local_target.compression.fixed_interaction_error_bound
    )
    certificate = local_target.cap_floor_proof.coefficient_certificate
    assert float(ledger.absorber_operator_error_bound) == float(
        certificate.absorber_operator_error_bound
    )
    assert float(ledger.cap_scale_error_bound) == float(
        local_target.cap_floor_proof.scale_error_bound
    )
    assert float(ledger.cap_operator_error_bound) == float(
        local_target.cap_floor_proof.physical_operator_error_upper_bound
    )
    duplicated = upward_add(expected, ledger.absorber_operator_error_bound)
    assert float(duplicated) > float(expected)
    assert "excludes LVT9 tail" in ledger.error_scope

    failed_floor = certify_axial_cap_floor(
        cap_proof.coefficient_certificate,
        maximum_gram_degree=1,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    assert not bool(failed_floor.exact_target_floor_eligible)
    assert not bool(failed_floor.realized_floor_eligible)
    assert bool(jnp.isfinite(failed_floor.physical_operator_error_upper_bound))
    admitted = compose_local_cell_galerkin_target(
        failed_floor,
        target_name="finite-B-without-floor",
    )
    assert bool(admitted.fixed_linear_error_ledger.finite_certificate)
    assert float(
        admitted.fixed_linear_error_ledger.cap_operator_error_bound
    ) == float(failed_floor.physical_operator_error_upper_bound)


def test_digest_split_and_canonical_target_name(
    cap_proof,
    local_target: GalerkinLocalCellTargetManifest,
) -> None:
    """Exclude proof/terminal/source context from exact operator identity."""
    canonical = local_system._compose_prepared(
        cap_proof, "local-cell-system-test"
    )
    padded = local_system._compose_prepared(
        cap_proof, "  local-cell-system-test  "
    )
    renamed = local_system._compose_prepared(cap_proof, "renamed-target")
    assert canonical.target_name == padded.target_name
    assert stored_value_payload(canonical) == stored_value_payload(padded)
    assert renamed.target_digest == canonical.target_digest
    assert (
        renamed.manifest_evidence_digest != canonical.manifest_evidence_digest
    )

    geometry = enclose_free_geometry(
        local_target.state_indices,
        local_target.support_eligibility,
        local_target.accelerating_voltage_kv,
    )
    transfer = transfer_exact_carrier_acquisition(
        local_target.support_eligibility,
        geometry,
    )
    proof_budget = eqx.tree_at(
        lambda value: value.maximum_gram_work,
        cap_proof,
        cap_proof.maximum_gram_work + 1,
    )
    certificate = cap_proof.coefficient_certificate
    absorber = certificate.absorber
    core = absorber.interaction_core
    compression = core.compression
    realization = compression.realization
    eligibility = realization.support_eligibility
    acquisition = eligibility.manifest
    context_manifest = replace(
        acquisition,
        incident_indices=jnp.asarray([[1, 0, 0]], dtype=jnp.int64),
        elastic_outgoing_indices=jnp.asarray([[-1, 0, 0]], dtype=jnp.int64),
        outgoing_physical_wavevectors=(
            acquisition.outgoing_physical_wavevectors
            + jnp.asarray([[1.0e-6, 0.0, 0.0]], dtype=jnp.float64)
        ),
        terminal_side=GalerkinTerminalSide.NEGATIVE,
    )
    context_eligibility = replace(eligibility, manifest=context_manifest)
    context_realization = replace(
        realization, support_eligibility=context_eligibility
    )
    context_compression = replace(compression, realization=context_realization)
    context_core = replace(core, compression=context_compression)
    context_absorber = replace(absorber, interaction_core=context_core)
    context_certificate = replace(certificate, absorber=context_absorber)
    acquisition_context = replace(
        cap_proof, coefficient_certificate=context_certificate
    )
    for contextual in (proof_budget, acquisition_context):
        assert local_system._target_digest(contextual, geometry) == (
            canonical.target_digest
        )
        contextual_evidence = local_system._manifest_evidence_digest(
            contextual,
            geometry,
            transfer,
            local_target.fixed_linear_error_ledger,
            canonical.target_digest,
            canonical.target_name,
        )
        assert contextual_evidence != canonical.manifest_evidence_digest

    operator_mutation = eqx.tree_at(
        lambda value: (
            value.coefficient_certificate.absorber.absorber_coefficients
        ),
        cap_proof,
        cap_proof.coefficient_certificate.absorber.absorber_coefficients.at[
            0
        ].add(1.0e-6),
    )
    assert local_system._target_digest(operator_mutation, geometry) != (
        canonical.target_digest
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "free_operator_error_bound",
        "interaction_operator_error_bound",
        "cap_operator_error_bound",
        "fixed_linear_operator_error_bound",
    ],
)
def test_prepare_rejects_each_rehashed_delta_forgery(
    field_name: str,
    cap_proof,
    local_target: GalerkinLocalCellTargetManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruct D/R/B/H instead of trusting a self-rehashed ledger."""
    monkeypatch.setattr(
        local_system,
        "prepare_axial_cap_floor",
        lambda submitted: cap_proof,
    )
    ledger = local_target.fixed_linear_error_ledger
    forged_ledger = replace(
        ledger,
        **{
            field_name: getattr(ledger, field_name)
            + jnp.asarray(1.0e-12, dtype=jnp.float64),
            "ledger_digest": "a" * 64,
        },
    )
    forged = replace(local_target, fixed_linear_error_ledger=forged_ledger)
    with pytest.raises(ValueError, match="full operator/evidence replay"):
        prepare_local_cell_galerkin_target(forged)


def test_prepare_rejects_nested_and_manifest_forgery_and_legacy(
    cap_proof,
    local_target: GalerkinLocalCellTargetManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject nested/digest, DIRECT, finite, and VC-1 seams.

    :see: :func:`ptyrodactyl.galerkin.prepare_local_cell_galerkin_target`
    """
    nested = replace(
        local_target,
        cap_floor_proof=replace(cap_proof, proof_digest="b" * 64),
    )
    with pytest.raises(ValueError, match="full host replay"):
        prepare_local_cell_galerkin_target(nested)

    monkeypatch.setattr(
        local_system,
        "prepare_axial_cap_floor",
        lambda submitted: cap_proof,
    )
    for field_name, replacement in (
        ("target_digest", "c" * 64),
        ("manifest_evidence_digest", "d" * 64),
    ):
        forged = replace(local_target, **{field_name: replacement})
        with pytest.raises(ValueError, match="full operator/evidence replay"):
            prepare_local_cell_galerkin_target(forged)

    semantic_ledger = replace(
        local_target.fixed_linear_error_ledger,
        error_scope="forged standalone ledger scope",
        ledger_digest="e" * 64,
    )
    semantic = replace(
        local_target,
        fixed_linear_error_ledger=semantic_ledger,
    )
    with pytest.raises(ValueError, match="full operator/evidence replay"):
        prepare_local_cell_galerkin_target(semantic)

    nonfinite = eqx.tree_at(
        lambda value: value.coefficient_certificate.finite_certificate,
        cap_proof,
        jnp.asarray(False),
    )
    with pytest.raises(ValueError, match="LVT.31 CAP coefficient"):
        local_system._compose_prepared(nonfinite, "nonfinite")

    certificate = cap_proof.coefficient_certificate
    absorber = certificate.absorber
    core = absorber.interaction_core
    compression = core.compression
    realization = compression.realization
    triangle_realization = replace(
        realization,
        error_route=type(realization.error_route).TRIANGLE_FALLBACK,
    )
    triangle_compression = replace(
        compression, realization=triangle_realization
    )
    triangle_core = replace(core, compression=triangle_compression)
    triangle_absorber = replace(absorber, interaction_core=triangle_core)
    triangle_certificate = replace(certificate, absorber=triangle_absorber)
    triangle = replace(cap_proof, coefficient_certificate=triangle_certificate)
    with pytest.raises(ValueError, match="DIRECT LVT.13"):
        local_system._compose_prepared(triangle, "triangle")


def test_prepare_explicitly_rejects_legacy_target() -> None:
    """Reject a VC-1 carrier inside prepare with the route-specific error."""
    with pytest.raises(TypeError, match="legacy targets"):
        prepare_local_cell_galerkin_target(production_target())
