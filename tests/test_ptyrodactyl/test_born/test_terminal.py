"""Tests for :mod:`ptyrodactyl.born.terminal`.

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptyrodactyl.born.acquisition import check_galerkin_acquisition_support
from ptyrodactyl.born.system import create_galerkin_target
from ptyrodactyl.born.terminal import (
    apply_galerkin_terminal_current,
    apply_galerkin_terminal_normal_derivative,
    apply_galerkin_terminal_normal_derivative_adjoint,
    apply_galerkin_terminal_trace,
    apply_galerkin_terminal_trace_adjoint,
    enclose_galerkin_terminal_current,
    evaluate_galerkin_terminal_current,
)
from ptyrodactyl.types.acquisition_types import GalerkinTerminalSide
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.constants import C_LIGHT, E_CHARGE, HBAR, M_E
from ptyrodactyl.types.galerkin_types import GalerkinTargetManifest
from ptyrodactyl.types.terminal_types import GalerkinTerminalCurrentScope
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
    full_eligibility = checked_acquisition(support, potential.box_size)
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
        target_name="selected-terminal-sector-target",
    )
    return result


class TestCoordinateTerminal:
    """Bind every public coordinate-terminal map to this test module.

    :see: :func:`ptyrodactyl.born.apply_galerkin_terminal_current`
    :see: :func:`ptyrodactyl.born.apply_galerkin_terminal_normal_derivative`
    :see: :func:`ptyrodactyl.born.\
apply_galerkin_terminal_normal_derivative_adjoint`
    :see: :func:`ptyrodactyl.born.apply_galerkin_terminal_trace`
    :see: :func:`ptyrodactyl.born.apply_galerkin_terminal_trace_adjoint`
    :see: :func:`ptyrodactyl.born.enclose_galerkin_terminal_current`
    :see: :func:`ptyrodactyl.born.evaluate_galerkin_terminal_current`
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
