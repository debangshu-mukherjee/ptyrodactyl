"""Tests for :mod:`ptyrodactyl.galerkin.terminal`.

Extended Summary
----------------
These tests bind the matrix-free coordinate trace, normal derivative, actual
adjoints, Hermitian current action, fiber cross terms, exact-current interval,
orientation, compiled execution, and deliberately unavailable downstream
physical contracts.
"""

from dataclasses import replace
from decimal import Decimal, getcontext
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple

import ptyrodactyl.galerkin.terminal as terminal_module
from ptyrodactyl.galerkin.acquisition import check_galerkin_acquisition_support
from ptyrodactyl.galerkin.system import create_galerkin_target
from ptyrodactyl.galerkin.terminal import (
    apply_galerkin_terminal_current,
    apply_galerkin_terminal_normal_derivative,
    apply_galerkin_terminal_normal_derivative_adjoint,
    apply_galerkin_terminal_trace,
    apply_galerkin_terminal_trace_adjoint,
    certify_galerkin_terminal_current_operator,
    enclose_galerkin_terminal_current,
    enclose_galerkin_terminal_current_action,
    evaluate_galerkin_terminal_current,
    prepare_galerkin_terminal_current_diagnostic,
)
from ptyrodactyl.types import C_LIGHT, E_CHARGE, HBAR, M_E
from ptyrodactyl.types.acquisition_types import (
    GalerkinBackwardDisposition,
    GalerkinTerminalSide,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.galerkin_types import GalerkinTargetManifest
from ptyrodactyl.types.terminal_types import (
    GalerkinCurrentOperatorCertificate,
    GalerkinCurrentOperatorFailure,
    GalerkinTerminalCurrentActionFailure,
    GalerkinTerminalCurrentScope,
)
from tests._galerkin_target_fixture import (
    TARGET_CAP_SCALE,
    TARGET_VOLTAGE_KV,
    checked_acquisition,
    periodic_target_potential,
    production_target,
)

getcontext().prec = 100

_PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923"
)


@pytest.fixture(scope="module")
def target() -> GalerkinTargetManifest:
    """Build one shared nontrivial coordinate-terminal target."""
    return production_target()


@pytest.fixture(scope="module")
def selected_sector_target() -> GalerkinTargetManifest:
    """Build an eligible target with an unselected retained fiber."""
    return _build_selected_sector_target(GalerkinTerminalSide.POSITIVE)


def _build_selected_sector_target(
    terminal_side: GalerkinTerminalSide,
) -> GalerkinTargetManifest:
    """Build a factory-owned target with one selected transverse fiber."""
    state = jnp.asarray(
        [
            (normal, transverse, 0)
            for transverse in (0, 1)
            for normal in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    interaction = jnp.asarray(
        [(normal, 0, 0) for normal in range(-1, 2)],
        dtype=jnp.int64,
    )
    absorber = jnp.asarray(
        [
            (normal, transverse, third)
            for normal in range(-2, 3)
            for transverse in range(-1, 2)
            for third in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    work = jnp.asarray(
        [
            (normal, transverse, third)
            for normal in range(-3, 4)
            for transverse in range(-1, 3)
            for third in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    support: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=state,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(7, 5, 3),
    )
    potential = periodic_target_potential()
    negative_side = terminal_side is GalerkinTerminalSide.NEGATIVE
    full_eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_side=terminal_side,
        backward_disposition=(
            GalerkinBackwardDisposition.REPRESENTED
            if negative_side
            else GalerkinBackwardDisposition.EXCLUDED
        ),
        claims_backscatter=negative_side,
    )
    selected_preterminal = state[state[:, 1] == 0]
    selected_manifest = replace(
        full_eligibility.manifest,
        preterminal_indices=selected_preterminal,
        transverse_indices=jnp.asarray(((0, 0),), dtype=jnp.int64),
    )
    selected_eligibility = check_galerkin_acquisition_support(
        selected_manifest
    )
    assert bool(selected_eligibility.support_eligible)
    result: GalerkinTargetManifest = create_galerkin_target(
        potential,
        selected_eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name=f"selected-terminal-sector-{terminal_side.value}-target",
    )
    return result


@pytest.fixture(scope="module")
def selected_operator_certificate(
    selected_sector_target: GalerkinTargetManifest,
) -> GalerkinCurrentOperatorCertificate:
    """Build one authenticated uniform selected-sector certificate."""
    target = selected_sector_target
    indices = target.support.state_indices
    field = (
        0.25
        + 0.1 * indices[:, 0].astype(jnp.float64)
        + 1j * (0.2 - 0.05 * indices[:, 0].astype(jnp.float64))
    ).astype(jnp.complex128)
    diagnostic = enclose_galerkin_terminal_current(target, field)
    return certify_galerkin_terminal_current_operator(diagnostic)


class TestCoordinateTerminal:
    """Bind every public coordinate-terminal map to this test module.

    :see: :func:`ptyrodactyl.galerkin.apply_galerkin_terminal_current`
    :see: :func:`ptyrodactyl.galerkin.\
apply_galerkin_terminal_normal_derivative`
    :see: :func:`ptyrodactyl.galerkin.\
apply_galerkin_terminal_normal_derivative_adjoint`
    :see: :func:`ptyrodactyl.galerkin.apply_galerkin_terminal_trace`
    :see: :func:`ptyrodactyl.galerkin.apply_galerkin_terminal_trace_adjoint`
    :see: :func:`ptyrodactyl.galerkin.enclose_galerkin_terminal_current`
    :see: :func:`ptyrodactyl.galerkin.evaluate_galerkin_terminal_current`
    """

    def test_maps_have_actual_adjoints_and_hermitian_current(
        self,
        target: GalerkinTargetManifest,
    ) -> None:
        """Verify T*, N*, and self-adjoint F in their frozen metrics."""
        state = jnp.asarray(
            (1.0 + 2.0j, -0.3 + 0.7j, 0.5 - 0.2j),
            dtype=jnp.complex128,
        )
        second_state = jnp.asarray(
            (-0.4 + 0.1j, 0.8 - 0.6j, 0.2 + 0.9j),
            dtype=jnp.complex128,
        )
        terminal = jnp.asarray((0.6 - 0.35j,), dtype=jnp.complex128)

        trace_left = jnp.vdot(
            apply_galerkin_terminal_trace(target, state), terminal
        )
        trace_right = jnp.vdot(
            state, apply_galerkin_terminal_trace_adjoint(target, terminal)
        )
        normal_left = jnp.vdot(
            apply_galerkin_terminal_normal_derivative(target, state), terminal
        )
        normal_right = jnp.vdot(
            state,
            apply_galerkin_terminal_normal_derivative_adjoint(
                target, terminal
            ),
        )
        current_left = jnp.vdot(
            apply_galerkin_terminal_current(target, state), second_state
        )
        current_right = jnp.vdot(
            state,
            apply_galerkin_terminal_current(target, second_state),
        )

        np.testing.assert_allclose(trace_left, trace_right, rtol=2e-15)
        np.testing.assert_allclose(normal_left, normal_right, rtol=2e-15)
        np.testing.assert_allclose(current_left, current_right, rtol=2e-15)

    def test_current_retains_nontrivial_same_fiber_cross_terms(
        self,
        target: GalerkinTargetManifest,
    ) -> None:
        """Require sum-before-product normal-frequency interference."""
        field = jnp.asarray(
            (1.0 + 2.0j, -0.3 + 0.7j, 0.5 - 0.2j),
            dtype=jnp.complex128,
        )
        axis = target.acquisition.terminal_axis
        length = target.box_lengths[axis]
        indices = target.support.state_indices[:, axis]
        wavevectors = (
            target.carrier[axis]
            + 2.0 * jnp.pi * indices.astype(jnp.float64) / length
        )
        summed_field = jnp.sum(field)
        weighted_sum = jnp.sum(wavevectors * field)
        expected = jnp.real(jnp.conj(summed_field) * weighted_sum) / length
        diagonal_only = jnp.sum(wavevectors * jnp.abs(field) ** 2) / length
        expected_action = jnp.stack(
            [
                jnp.sum((wavevector + wavevectors) * field) / (2.0 * length)
                for wavevector in wavevectors
            ]
        )

        action = apply_galerkin_terminal_current(target, field)
        current = evaluate_galerkin_terminal_current(target, field)

        np.testing.assert_allclose(action, expected_action, rtol=2e-15)
        np.testing.assert_allclose(current, expected, rtol=2e-15)
        assert not bool(jnp.isclose(current, diagonal_only, rtol=1e-8))

    def test_unselected_retained_fiber_is_not_total_plane_current(
        self,
        selected_sector_target: GalerkinTargetManifest,
    ) -> None:
        """Prove the diagnostic excludes an unselected ``K_u`` fiber."""
        target = selected_sector_target
        state_indices = target.support.state_indices
        selected = state_indices[:, 1] == 0
        unselected = state_indices[:, 1] == 1
        selected_field = jnp.where(
            selected & (state_indices[:, 0] == 0),
            2.0 + 0.0j,
            0.0 + 0.0j,
        ).astype(jnp.complex128)
        unselected_field = jnp.where(
            unselected & (state_indices[:, 0] == 0),
            1.0 + 0.0j,
            0.0 + 0.0j,
        ).astype(jnp.complex128)
        combined_field = selected_field + unselected_field

        unselected_trace = apply_galerkin_terminal_trace(
            target, unselected_field
        )
        unselected_normal = apply_galerkin_terminal_normal_derivative(
            target, unselected_field
        )
        unselected_action = apply_galerkin_terminal_current(
            target, unselected_field
        )
        combined_action = apply_galerkin_terminal_current(
            target, combined_field
        )
        selected_action = apply_galerkin_terminal_current(
            target, selected_field
        )
        evidence = enclose_galerkin_terminal_current(target, unselected_field)

        np.testing.assert_array_equal(unselected_trace, 0.0 + 0.0j)
        np.testing.assert_array_equal(unselected_normal, 0.0 + 0.0j)
        np.testing.assert_array_equal(unselected_action, 0.0 + 0.0j)
        np.testing.assert_array_equal(combined_action, selected_action)
        np.testing.assert_array_equal(combined_action[unselected], 0.0 + 0.0j)
        assert float(evidence.exact_reduced_current_lower_bound) <= 0.0
        assert float(evidence.exact_reduced_current_upper_bound) >= 0.0
        full_plane_current = float(target.carrier[0] / target.box_lengths[0])
        assert full_plane_current > float(
            evidence.exact_reduced_current_upper_bound
        )
        assert evidence.current_scope is (
            GalerkinTerminalCurrentScope.SELECTED_ACQUISITION_FIBER_SECTOR
        )
        assert "selected by acquisition K_d" in evidence.current_target
        assert "does not claim total/full-plane current" in (
            evidence.eligibility_scope
        )

    def test_opposite_face_reverses_normal_and_current_only(
        self,
        target: GalerkinTargetManifest,
    ) -> None:
        """Reverse oriented N, F, and current while preserving T."""
        positive = target
        negative = _opposite_side_target(positive)
        field = jnp.asarray(
            (0.2 + 1.0j, -0.7 + 0.4j, 1.1 - 0.5j),
            dtype=jnp.complex128,
        )

        positive_trace = apply_galerkin_terminal_trace(positive, field)
        negative_trace = apply_galerkin_terminal_trace(negative, field)
        positive_normal = apply_galerkin_terminal_normal_derivative(
            positive, field
        )
        negative_normal = apply_galerkin_terminal_normal_derivative(
            negative, field
        )
        positive_action = apply_galerkin_terminal_current(positive, field)
        negative_action = apply_galerkin_terminal_current(negative, field)
        positive_evidence = enclose_galerkin_terminal_current(positive, field)
        negative_evidence = enclose_galerkin_terminal_current(negative, field)

        np.testing.assert_array_equal(negative_trace, positive_trace)
        np.testing.assert_array_equal(negative_normal, -positive_normal)
        np.testing.assert_array_equal(negative_action, -positive_action)
        np.testing.assert_allclose(
            negative_evidence.reduced_current,
            -positive_evidence.reduced_current,
            rtol=2e-15,
        )
        np.testing.assert_allclose(
            negative_evidence.exact_reduced_current_lower_bound,
            -positive_evidence.exact_reduced_current_upper_bound,
            rtol=2e-15,
        )
        np.testing.assert_allclose(
            negative_evidence.exact_reduced_current_upper_bound,
            -positive_evidence.exact_reduced_current_lower_bound,
            rtol=2e-15,
        )

    def test_exact_current_interval_contains_independent_decimal_oracle(
        self,
        target: GalerkinTargetManifest,
    ) -> None:
        """Contain exact constants, pi, and stored coefficients."""
        field = jnp.asarray(
            (1.0 + 2.0j, -0.3 + 0.7j, 0.5 - 0.2j),
            dtype=jnp.complex128,
        )

        evidence = enclose_galerkin_terminal_current(target, field)
        exact = _decimal_exact_current(target, field)
        lower = Decimal.from_float(
            float(evidence.exact_reduced_current_lower_bound)
        )
        upper = Decimal.from_float(
            float(evidence.exact_reduced_current_upper_bound)
        )

        assert lower <= exact <= upper
        assert bool(evidence.current_diagnostic_eligible)
        assert not bool(evidence.vacuum_branch_eligible)
        assert not bool(evidence.detector_eligible)
        rounded_error = abs(
            Decimal.from_float(float(evidence.reduced_current)) - exact
        )
        assert rounded_error <= Decimal.from_float(
            float(evidence.reduced_current_error_upper_bound)
        )

    def test_compiled_maps_and_enclosure_match_eager(
        self,
        target: GalerkinTargetManifest,
    ) -> None:
        """Keep all terminal actions and bounded evidence JIT-compatible."""
        field = jnp.asarray(
            (0.75 - 0.5j, 0.2 + 0.4j, -0.1 + 1.2j),
            dtype=jnp.complex128,
        )
        eager_action = apply_galerkin_terminal_current(target, field)
        eager_evidence = enclose_galerkin_terminal_current(target, field)

        compiled_action = jax.jit(
            lambda value: apply_galerkin_terminal_current(target, value)
        )(field)
        compiled_evidence = jax.jit(
            lambda value: enclose_galerkin_terminal_current(target, value)
        )(field)

        np.testing.assert_allclose(compiled_action, eager_action, rtol=1e-15)
        np.testing.assert_allclose(
            compiled_evidence.current_action,
            eager_evidence.current_action,
            rtol=1e-15,
        )
        np.testing.assert_array_equal(
            compiled_evidence.exact_reduced_current_lower_bound,
            eager_evidence.exact_reduced_current_lower_bound,
        )
        np.testing.assert_array_equal(
            compiled_evidence.exact_reduced_current_upper_bound,
            eager_evidence.exact_reduced_current_upper_bound,
        )
        exact = _decimal_exact_current(target, field)
        eager_lower = Decimal.from_float(
            float(eager_evidence.exact_reduced_current_lower_bound)
        )
        eager_upper = Decimal.from_float(
            float(eager_evidence.exact_reduced_current_upper_bound)
        )
        compiled_lower = Decimal.from_float(
            float(compiled_evidence.exact_reduced_current_lower_bound)
        )
        compiled_upper = Decimal.from_float(
            float(compiled_evidence.exact_reduced_current_upper_bound)
        )
        assert eager_lower <= exact <= eager_upper
        assert compiled_lower <= exact <= compiled_upper


class TestCurrentOperatorCertificate:
    """Bind the stronger uniform current-operator evidence to a dense wall.

    :see: :func:`ptyrodactyl.galerkin.\
certify_galerkin_terminal_current_operator`
    :see: :func:`ptyrodactyl.galerkin.\
enclose_galerkin_terminal_current_action`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_galerkin_terminal_current_diagnostic`
    """

    def test_dense_exact_and_frozen_operators_fit_every_uniform_bound(
        self,
        selected_operator_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Independently assemble T, N, F and verify the LVT.55a wall."""
        certificate = selected_operator_certificate
        target = certificate.diagnostic.target
        exact_trace, exact_normal = _decimal_exact_terminal_matrices(target)
        frozen_trace, frozen_normal = _frozen_terminal_matrices(certificate)
        exact_current = _dense_current(exact_trace, exact_normal)
        frozen_current = _dense_current(frozen_trace, frozen_normal)

        trace_error = np.linalg.norm(frozen_trace - exact_trace, ord=2)
        normal_error = np.linalg.norm(frozen_normal - exact_normal, ord=2)
        current_error = np.linalg.norm(frozen_current - exact_current, ord=2)
        assert trace_error <= float(
            certificate.trace_operator_error_upper_bound
        )
        assert normal_error <= float(
            certificate.normal_operator_error_upper_bound
        )
        assert current_error <= float(
            certificate.current_operator_error_upper_bound
        )
        assert np.linalg.norm(exact_trace, ord=2) <= float(
            certificate.exact_trace_operator_norm_upper_bound
        )
        assert np.linalg.norm(exact_normal, ord=2) <= float(
            certificate.exact_normal_operator_norm_upper_bound
        )
        np.testing.assert_allclose(
            exact_current, exact_current.conj().T, rtol=0.0, atol=2e-15
        )
        np.testing.assert_allclose(
            frozen_current, frozen_current.conj().T, rtol=0.0, atol=2e-15
        )

        rows = _selected_terminal_rows(target)
        selected_positions = np.flatnonzero(rows >= 0)
        unselected_positions = np.flatnonzero(rows < 0)
        first_row_positions = selected_positions[rows[selected_positions] == 0]
        assert first_row_positions.size >= 2
        first = int(first_row_positions[0])
        second = int(first_row_positions[1])
        assert exact_current[first, second] != 0.0
        np.testing.assert_array_equal(
            frozen_current[unselected_positions, :], 0.0 + 0.0j
        )
        np.testing.assert_array_equal(
            frozen_current[:, unselected_positions], 0.0 + 0.0j
        )

        exact_scale = _decimal_exact_number_current_scale(target)
        scale_lower = Decimal.from_float(
            float(certificate.exact_number_current_scale_lower_bound)
        )
        scale_upper = Decimal.from_float(
            float(certificate.exact_number_current_scale_upper_bound)
        )
        assert scale_lower <= exact_scale <= scale_upper
        stored_scale_error = abs(
            Decimal.from_float(float(certificate.number_current_scale))
            - exact_scale
        )
        assert stored_scale_error <= Decimal.from_float(
            float(certificate.number_current_scale_error_upper_bound)
        )
        assert bool(certificate.current_operator_eligible)
        assert int(certificate.current_operator_failure_mask) == int(
            GalerkinCurrentOperatorFailure.NONE
        )
        assert certificate.terminal_plane_coordinate == 0.0
        assert certificate.current_scope is (
            GalerkinTerminalCurrentScope.SELECTED_ACQUISITION_FIBER_SECTOR
        )
        assert not hasattr(certificate, "vacuum_branch_eligible")
        assert not hasattr(certificate, "detector_eligible")
        assert not bool(certificate.diagnostic.vacuum_branch_eligible)
        assert not bool(certificate.diagnostic.detector_eligible)
        assert "excludes unselected K_u" in certificate.eligibility_scope
        assert "additionally requires" in certificate.eligibility_scope
        assert "finite_certificate" in certificate.eligibility_scope

    def test_per_call_action_contains_exact_dyadic_dense_action_and_forgery(
        self,
        selected_operator_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Contain exact frozen arithmetic and reject a safe-looking forge."""
        certificate = selected_operator_certificate
        target = certificate.diagnostic.target
        rows = _selected_terminal_rows(target)
        selected_positions = np.flatnonzero(rows >= 0)
        selected_position = int(selected_positions[0])
        subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
        host_field = np.zeros(rows.shape[0], dtype=np.complex128)
        normal_indices = np.asarray(target.support.state_indices)[
            :, target.acquisition.terminal_axis
        ]
        for position in selected_positions:
            normal_index = float(normal_indices[position])
            host_field[position] = complex(
                0.3 - 0.07 * normal_index,
                0.4 + 0.11 * normal_index,
            )
        host_field[selected_position] = complex(subnormal, -subnormal)
        field = jnp.asarray(host_field, dtype=jnp.complex128)
        assert np.asarray(field)[selected_position] != 0.0 + 0.0j
        enclosure = enclose_galerkin_terminal_current_action(
            certificate, field
        )
        exact_action = _exact_dyadic_frozen_action(certificate, field)
        total_squared_error = Fraction(0)
        for index, (exact_real, exact_imag) in enumerate(exact_action):
            real_lower = Fraction.from_float(
                float(enclosure.algebraic_action_real_lower_bounds[index])
            )
            real_upper = Fraction.from_float(
                float(enclosure.algebraic_action_real_upper_bounds[index])
            )
            imag_lower = Fraction.from_float(
                float(enclosure.algebraic_action_imag_lower_bounds[index])
            )
            imag_upper = Fraction.from_float(
                float(enclosure.algebraic_action_imag_upper_bounds[index])
            )
            assert real_lower <= exact_real <= real_upper
            assert imag_lower <= exact_imag <= imag_upper
            rounded = complex(np.asarray(enclosure.production_action)[index])
            real_error = Fraction.from_float(rounded.real) - exact_real
            imag_error = Fraction.from_float(rounded.imag) - exact_imag
            component_squared = (
                real_error * real_error + imag_error * imag_error
            )
            total_squared_error += component_squared
            component_error = _decimal_sqrt_fraction(component_squared)
            assert component_error <= Decimal.from_float(
                float(enclosure.component_error_bounds[index])
            )
        total_error = _decimal_sqrt_fraction(total_squared_error)
        assert total_error <= Decimal.from_float(
            float(enclosure.action_error_bound)
        )
        assert bool(enclosure.finite_certificate)
        assert int(enclosure.failure_mask) == int(
            GalerkinTerminalCurrentActionFailure.NONE
        )

        forged = eqx.tree_at(
            lambda record: record.current_operator_error_upper_bound,
            certificate,
            certificate.current_operator_error_upper_bound * 2.0,
        )
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="canonical replay authentication",
        ):
            forged_enclosure = enclose_galerkin_terminal_current_action(
                forged, field
            )
            jax.block_until_ready(forged_enclosure)

    def test_side_reversal_preserves_bounds_and_reverses_frozen_operator(
        self,
        selected_operator_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Reverse N and F without changing T, Cj, or norm bounds."""
        positive = selected_operator_certificate
        negative_target = _build_selected_sector_target(
            GalerkinTerminalSide.NEGATIVE
        )
        negative_diagnostic = enclose_galerkin_terminal_current(
            negative_target, positive.diagnostic.submitted_field
        )
        negative = certify_galerkin_terminal_current_operator(
            negative_diagnostic
        )
        positive_trace, positive_normal = _frozen_terminal_matrices(positive)
        negative_trace, negative_normal = _frozen_terminal_matrices(negative)

        np.testing.assert_array_equal(negative_trace, positive_trace)
        np.testing.assert_array_equal(negative_normal, -positive_normal)
        np.testing.assert_array_equal(
            _dense_current(negative_trace, negative_normal),
            -_dense_current(positive_trace, positive_normal),
        )
        np.testing.assert_array_equal(
            negative.trace_operator_error_upper_bound,
            positive.trace_operator_error_upper_bound,
        )
        np.testing.assert_array_equal(
            negative.normal_operator_error_upper_bound,
            positive.normal_operator_error_upper_bound,
        )
        np.testing.assert_array_equal(
            negative.current_operator_error_upper_bound,
            positive.current_operator_error_upper_bound,
        )
        np.testing.assert_array_equal(
            negative.number_current_scale, positive.number_current_scale
        )

    def test_nested_target_forgery_and_arithmetic_ineligibility_fail_closed(
        self,
        selected_operator_certificate: GalerkinCurrentOperatorCertificate,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject target forgery and type an arithmetic noncertificate."""
        certificate = selected_operator_certificate
        target = certificate.diagnostic.target
        forged_target = eqx.tree_at(
            lambda manifest: (
                manifest.fixed_linear_error_ledger.exact_carrier_lower_bounds
            ),
            target,
            target.fixed_linear_error_ledger.exact_carrier_lower_bounds - 1.0,
        )
        forged_diagnostic = enclose_galerkin_terminal_current(
            forged_target, certificate.diagnostic.submitted_field
        )
        with pytest.raises(
            ValueError, match="target failed canonical reconstruction"
        ):
            certify_galerkin_terminal_current_operator(forged_diagnostic)

        monkeypatch.setattr(
            terminal_module,
            "all_normal_arithmetic_supported",
            lambda: jnp.asarray(False),
        )
        ineligible_diagnostic = enclose_galerkin_terminal_current(
            target, certificate.diagnostic.submitted_field
        )
        ineligible = certify_galerkin_terminal_current_operator(
            ineligible_diagnostic
        )
        expected = (
            GalerkinCurrentOperatorFailure.CURRENT_DIAGNOSTIC_INELIGIBLE
            | GalerkinCurrentOperatorFailure.ARITHMETIC_ENVIRONMENT_UNSUPPORTED
        )
        assert not bool(ineligible.current_operator_eligible)
        assert int(ineligible.current_operator_failure_mask) == int(expected)
        assert not bool(ineligible.diagnostic.vacuum_branch_eligible)
        assert not bool(ineligible.diagnostic.detector_eligible)


def _opposite_side_target(
    target: GalerkinTargetManifest,
) -> GalerkinTargetManifest:
    """Return the identical target with opposite face orientation."""
    side = (
        GalerkinTerminalSide.NEGATIVE
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else GalerkinTerminalSide.POSITIVE
    )
    acquisition = replace(target.acquisition, terminal_side=side)
    support_eligibility = replace(
        target.support_eligibility, manifest=acquisition
    )
    realization = replace(
        target.realization, support_eligibility=support_eligibility
    )
    opposite: GalerkinTargetManifest = replace(target, realization=realization)
    return opposite


def _selected_terminal_rows(target: GalerkinTargetManifest) -> np.ndarray:
    """Map each state coefficient to a selected terminal row or minus one."""
    axis = target.acquisition.terminal_axis
    transverse_axes = tuple(index for index in range(3) if index != axis)
    terminal_rows = {
        tuple(int(component) for component in transverse): row
        for row, transverse in enumerate(
            np.asarray(target.acquisition.transverse_indices)
        )
    }
    rows = np.full(target.support.state_indices.shape[0], -1, dtype=np.int64)
    for position, state_index in enumerate(
        np.asarray(target.support.state_indices)
    ):
        transverse = tuple(
            int(state_index[index]) for index in transverse_axes
        )
        rows[position] = terminal_rows.get(transverse, -1)
    return rows


def _decimal_exact_terminal_matrices(
    target: GalerkinTargetManifest,
) -> Tuple[np.ndarray, np.ndarray]:
    """Independently assemble exact-target T and N through Decimal data."""
    state_size = target.support.state_indices.shape[0]
    terminal_size = target.acquisition.transverse_indices.shape[0]
    trace = np.zeros((terminal_size, state_size), dtype=np.complex128)
    normal = np.zeros((terminal_size, state_size), dtype=np.complex128)
    axis = target.acquisition.terminal_axis
    length = _decimal_from_float(np.asarray(target.box_lengths)[axis])
    normalization = Decimal(1) / length.sqrt()
    carrier = _decimal_exact_wavenumber()
    sign = (
        Decimal(1)
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else Decimal(-1)
    )
    rows = _selected_terminal_rows(target)
    normal_indices = np.asarray(target.support.state_indices)[:, axis]
    for position, (row, normal_index) in enumerate(
        zip(rows, normal_indices, strict=True)
    ):
        if row < 0:
            continue
        wavevector = sign * (
            carrier + Decimal(2) * _PI * Decimal(int(normal_index)) / length
        )
        trace[row, position] = complex(float(normalization), 0.0)
        normal[row, position] = complex(0.0, float(normalization * wavevector))
    return trace, normal


def _frozen_terminal_matrices(
    certificate: GalerkinCurrentOperatorCertificate,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense-assemble the stored frozen dyadic T and N independently."""
    target = certificate.diagnostic.target
    state_size = target.support.state_indices.shape[0]
    terminal_size = target.acquisition.transverse_indices.shape[0]
    trace = np.zeros((terminal_size, state_size), dtype=np.complex128)
    normal = np.zeros((terminal_size, state_size), dtype=np.complex128)
    rows = _selected_terminal_rows(target)
    trace_coefficients = np.asarray(certificate.trace_frozen_coefficients)
    normal_coefficients = np.asarray(certificate.normal_frozen_coefficients)
    for position, row in enumerate(rows):
        if row >= 0:
            trace[row, position] = trace_coefficients[position]
            normal[row, position] = normal_coefficients[position]
    return trace, normal


def _dense_current(trace: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Assemble the Hermitian dyadic current from independent T and N."""
    current: np.ndarray = (
        trace.conj().T @ normal - normal.conj().T @ trace
    ) / (2.0j)
    return current


def _decimal_exact_number_current_scale(
    target: GalerkinTargetManifest,
) -> Decimal:
    """Evaluate exact SC.35c from the exact stored binary64 constants."""
    mass = _decimal_from_float(M_E)
    charge = _decimal_from_float(E_CHARGE)
    speed = _decimal_from_float(C_LIGHT)
    hbar = _decimal_from_float(HBAR)
    voltage = _decimal_from_float(target.accelerating_voltage_kv) * Decimal(
        1000
    )
    speed_squared = speed * speed
    scale: Decimal = (
        hbar
        * speed_squared
        / (mass * speed_squared + charge * voltage)
        * Decimal("1e20")
    )
    return scale


def _fraction_complex(
    value: complex | np.complex128,
) -> Tuple[Fraction, Fraction]:
    """Interpret both stored binary64 complex components as exact dyadics."""
    scalar = complex(value)
    return Fraction.from_float(scalar.real), Fraction.from_float(scalar.imag)


def _fraction_complex_add(
    left: Tuple[Fraction, Fraction],
    right: Tuple[Fraction, Fraction],
) -> Tuple[Fraction, Fraction]:
    """Add exact complex rational pairs."""
    return left[0] + right[0], left[1] + right[1]


def _fraction_complex_subtract(
    left: Tuple[Fraction, Fraction],
    right: Tuple[Fraction, Fraction],
) -> Tuple[Fraction, Fraction]:
    """Subtract exact complex rational pairs."""
    return left[0] - right[0], left[1] - right[1]


def _fraction_complex_multiply(
    left: Tuple[Fraction, Fraction],
    right: Tuple[Fraction, Fraction],
) -> Tuple[Fraction, Fraction]:
    """Multiply exact complex rational pairs."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _fraction_complex_conjugate(
    value: Tuple[Fraction, Fraction],
) -> Tuple[Fraction, Fraction]:
    """Conjugate one exact complex rational pair."""
    return value[0], -value[1]


def _exact_dyadic_frozen_action(
    certificate: GalerkinCurrentOperatorCertificate,
    field: jax.Array,
) -> list[Tuple[Fraction, Fraction]]:
    """Dense-evaluate frozen F in exact rational arithmetic."""
    rows = _selected_terminal_rows(certificate.diagnostic.target)
    trace = [
        _fraction_complex(value)
        for value in np.asarray(certificate.trace_frozen_coefficients)
    ]
    normal = [
        _fraction_complex(value)
        for value in np.asarray(certificate.normal_frozen_coefficients)
    ]
    state = [_fraction_complex(value) for value in np.asarray(field)]
    inverse_two_i = (Fraction(0), Fraction(-1, 2))
    action: list[Tuple[Fraction, Fraction]] = []
    for left_position, left_row in enumerate(rows):
        total = (Fraction(0), Fraction(0))
        if left_row >= 0:
            for right_position, right_row in enumerate(rows):
                if right_row != left_row:
                    continue
                trace_normal = _fraction_complex_multiply(
                    _fraction_complex_conjugate(trace[left_position]),
                    normal[right_position],
                )
                normal_trace = _fraction_complex_multiply(
                    _fraction_complex_conjugate(normal[left_position]),
                    trace[right_position],
                )
                current_entry = _fraction_complex_multiply(
                    _fraction_complex_subtract(trace_normal, normal_trace),
                    inverse_two_i,
                )
                total = _fraction_complex_add(
                    total,
                    _fraction_complex_multiply(
                        current_entry, state[right_position]
                    ),
                )
        action.append(total)
    return action


def _decimal_sqrt_fraction(value: Fraction) -> Decimal:
    """Evaluate a nonnegative exact rational square root at high precision."""
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    result: Decimal = decimal.sqrt()
    return result


def _decimal_from_float(value: float | np.float64 | jax.Array) -> Decimal:
    """Return the exact decimal value of one stored binary64 number."""
    fraction = Fraction.from_float(float(value))
    decimal: Decimal = Decimal(fraction.numerator) / Decimal(
        fraction.denominator
    )
    return decimal


def _decimal_exact_wavenumber() -> Decimal:
    """Evaluate exact SC.2 from exact manifested binary64 constants."""
    mass = _decimal_from_float(M_E)
    charge = _decimal_from_float(E_CHARGE)
    speed = _decimal_from_float(C_LIGHT)
    hbar = _decimal_from_float(HBAR)
    voltage = _decimal_from_float(TARGET_VOLTAGE_KV) * Decimal(1000)
    prefactor = Decimal(2) * mass * charge / (hbar * hbar)
    rest_energy = mass * speed * speed
    correction = Decimal(1) + charge * voltage / (Decimal(2) * rest_energy)
    wavenumber: Decimal = (
        prefactor * voltage * correction * Decimal("1e-20")
    ).sqrt()
    return wavenumber


def _decimal_exact_current(
    target: GalerkinTargetManifest,
    field: jax.Array,
) -> Decimal:
    """Evaluate the exact same-fiber current independently."""
    axis = target.acquisition.terminal_axis
    length = _decimal_from_float(np.asarray(target.box_lengths)[axis])
    carrier = _decimal_exact_wavenumber()
    stored_field = np.asarray(field)
    summed_real = sum(
        (_decimal_from_float(value.real) for value in stored_field),
        Decimal(0),
    )
    summed_imag = sum(
        (_decimal_from_float(value.imag) for value in stored_field),
        Decimal(0),
    )
    weighted_real = Decimal(0)
    weighted_imag = Decimal(0)
    for index, value in zip(
        np.asarray(target.support.state_indices)[:, axis],
        stored_field,
        strict=True,
    ):
        wavevector = carrier + Decimal(2) * _PI * Decimal(int(index)) / length
        weighted_real += wavevector * _decimal_from_float(value.real)
        weighted_imag += wavevector * _decimal_from_float(value.imag)
    exact: Decimal = (
        summed_real * weighted_real + summed_imag * weighted_imag
    ) / length
    if target.acquisition.terminal_side is GalerkinTerminalSide.NEGATIVE:
        exact = -exact
    return exact
