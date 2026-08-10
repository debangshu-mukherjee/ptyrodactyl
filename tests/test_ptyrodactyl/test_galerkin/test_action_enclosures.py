"""Tests for :mod:`ptyrodactyl.galerkin.action_enclosures`.

Extended Summary
----------------
These tests compare the outward RM-S2 per-state evidence with exact rational
dense actions assembled independently from the stored binary target data.
They cover forward and actual-adjoint signs, cancellation, compiled execution,
typed overflow fallback, and the nondifferentiable evidence boundary.
"""

from decimal import Decimal, getcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple

import ptyrodactyl.galerkin.action_enclosures as action_enclosures
from ptyrodactyl._interval import (
    _interval_add,
    _interval_multiply,
    _is_nonzero_subnormal,
    _minimum_normal,
    _point_interval,
)
from ptyrodactyl.galerkin.action_enclosures import (
    _action_arithmetic_environment_probes,
    enclose_galerkin_residual,
    enclose_galerkin_target_action,
)
from ptyrodactyl.types.action_error_types import (
    GalerkinActionDirection,
    GalerkinActionErrorRoute,
)
from ptyrodactyl.types.galerkin_types import GalerkinTargetManifest
from tests._galerkin_target_fixture import production_target

getcontext().prec = 120

type _RationalComplex = Tuple[Fraction, Fraction]


class TestActionEnclosures:
    """Bind public action and residual evaluators to this test module.

    :see: :func:`ptyrodactyl.galerkin.enclose_galerkin_residual`
    :see: :func:`ptyrodactyl.galerkin.enclose_galerkin_target_action`
    """

    def test_public_evaluators_are_distinct_callables(self) -> None:
        """Keep action and independent-residual APIs distinct."""
        assert callable(enclose_galerkin_target_action)
        assert callable(enclose_galerkin_residual)
        assert enclose_galerkin_target_action is not enclose_galerkin_residual


def _rational_float(value: float | np.float64) -> Fraction:
    """Return the exact rational represented by one binary64 value."""
    return Fraction.from_float(float(value))


def _rational_complex(value: complex | np.complex128) -> _RationalComplex:
    """Return exact rationals for one stored complex binary64 value."""
    stored = complex(value)
    return (_rational_float(stored.real), _rational_float(stored.imag))


def _complex_add(
    left: _RationalComplex,
    right: _RationalComplex,
) -> _RationalComplex:
    """Add two exact rational complex values."""
    return (left[0] + right[0], left[1] + right[1])


def _complex_subtract(
    left: _RationalComplex,
    right: _RationalComplex,
) -> _RationalComplex:
    """Subtract two exact rational complex values."""
    return (left[0] - right[0], left[1] - right[1])


def _complex_multiply(
    left: _RationalComplex,
    right: _RationalComplex,
) -> _RationalComplex:
    """Multiply two exact rational complex values."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _complex_conjugate(value: _RationalComplex) -> _RationalComplex:
    """Conjugate one exact rational complex value."""
    return (value[0], -value[1])


def _decimal(value: Fraction) -> Decimal:
    """Convert one exact fraction to the active high-precision decimal."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def _exact_dense_matrix(
    target: GalerkinTargetManifest,
) -> list[list[_RationalComplex]]:
    """Assemble exact-real ``H_alg`` independently from stored coefficients."""
    state = np.asarray(target.support.state_indices, dtype=np.int64)
    interaction = {
        tuple(index): _rational_complex(value)
        for index, value in zip(
            np.asarray(target.support.interaction_indices),
            np.asarray(target.interaction_coefficients),
            strict=True,
        )
    }
    absorber = {
        tuple(index): _rational_complex(value)
        for index, value in zip(
            np.asarray(target.support.absorber_indices),
            np.asarray(target.absorber_coefficients),
            strict=True,
        )
    }
    cap = (
        _rational_float(float(np.asarray(target.cap_scale))),
        Fraction(0),
    )
    minus_i = (Fraction(0), Fraction(-1))
    zero = (Fraction(0), Fraction(0))
    matrix: list[list[_RationalComplex]] = []
    for row_position, row_index in enumerate(state):
        matrix_row: list[_RationalComplex] = []
        for column_position, column_index in enumerate(state):
            difference = tuple(row_index - column_index)
            free = (
                _rational_float(np.asarray(target.free_diagonal)[row_position])
                if row_position == column_position
                else Fraction(0),
                Fraction(0),
            )
            interaction_entry = interaction.get(difference, zero)
            absorber_entry = absorber.get(difference, zero)
            cap_entry = _complex_multiply(
                minus_i,
                _complex_multiply(cap, absorber_entry),
            )
            matrix_row.append(
                _complex_add(
                    _complex_subtract(free, interaction_entry),
                    cap_entry,
                )
            )
        matrix.append(matrix_row)
    return matrix


def _exact_action(
    target: GalerkinTargetManifest,
    field: jax.Array,
    *,
    adjoint: bool,
) -> list[_RationalComplex]:
    """Apply the exact rational dense forward or conjugate transpose."""
    matrix = _exact_dense_matrix(target)
    if adjoint:
        matrix = [
            [
                _complex_conjugate(matrix[column][row])
                for column in range(len(matrix))
            ]
            for row in range(len(matrix))
        ]
    exact_field = [_rational_complex(value) for value in np.asarray(field)]
    result: list[_RationalComplex] = []
    for row in matrix:
        value = (Fraction(0), Fraction(0))
        for coefficient, field_value in zip(row, exact_field, strict=True):
            value = _complex_add(
                value,
                _complex_multiply(coefficient, field_value),
            )
        result.append(value)
    return result


def _exact_error_norm(
    stored: jax.Array,
    exact: list[_RationalComplex],
) -> Decimal:
    """Return a high-precision norm of stored-minus-exact components."""
    squared = Decimal(0)
    for stored_value, exact_value in zip(
        np.asarray(stored), exact, strict=True
    ):
        difference = _complex_subtract(
            _rational_complex(stored_value), exact_value
        )
        squared += _decimal(difference[0]) ** 2
        squared += _decimal(difference[1]) ** 2
    return squared.sqrt()


def _decimal_bound(value: jax.Array) -> Decimal:
    """Interpret a reported finite or infinite binary64 bound exactly."""
    return Decimal.from_float(float(value))


def _fraction_bound(value: jax.Array) -> Fraction:
    """Interpret one finite binary64 interval endpoint exactly."""
    return Fraction.from_float(float(value))


def _assert_fraction_contained(
    interval: Tuple[jax.Array, jax.Array],
    exact: Fraction,
) -> None:
    """Require one exact rational to lie in a finite outward interval."""
    assert _fraction_bound(interval[0]) <= exact
    assert exact <= _fraction_bound(interval[1])
    assert not bool(_is_nonzero_subnormal(interval[0]))
    assert not bool(_is_nonzero_subnormal(interval[1]))


@pytest.fixture(scope="module")
def target() -> GalerkinTargetManifest:
    """Build one shared nontrivial production target."""
    return production_target()


def test_ftz_safe_primitives_contain_exact_fraction_oracles() -> None:
    """Contain subnormal points, products, and cancellation under FTZ."""
    tiny = _minimum_normal()
    minimum_subnormal = jnp.asarray(
        float.fromhex("0x0.0000000000001p-1022"), dtype=jnp.float64
    )
    negative_minimum_subnormal = -minimum_subnormal
    half = jnp.asarray(0.5, dtype=jnp.float64)
    next_tiny = jnp.asarray(
        float.fromhex("0x1.0000000000001p-1022"), dtype=jnp.float64
    )

    positive_point = _point_interval(minimum_subnormal)
    negative_point = _point_interval(negative_minimum_subnormal)
    underflow_product = _interval_multiply(
        _point_interval(tiny), _point_interval(half)
    )
    cancellation = _interval_add(
        _point_interval(tiny), _point_interval(-next_tiny)
    )

    _assert_fraction_contained(
        positive_point, Fraction.from_float(float(minimum_subnormal))
    )
    _assert_fraction_contained(
        negative_point,
        Fraction.from_float(float(negative_minimum_subnormal)),
    )
    _assert_fraction_contained(
        underflow_product,
        Fraction.from_float(float(tiny)) * Fraction(1, 2),
    )
    _assert_fraction_contained(
        cancellation,
        Fraction.from_float(float(tiny))
        - Fraction.from_float(float(next_tiny)),
    )


def test_normal_arithmetic_probe_is_independent_of_gradual_underflow() -> None:
    """Require normal primitives without requiring subnormal support."""
    normal_supported, gradual_supported = (
        _action_arithmetic_environment_probes()
    )

    assert bool(normal_supported)
    assert isinstance(bool(gradual_supported), bool)


@pytest.mark.parametrize("adjoint", [False, True])
def test_action_interval_contains_exact_dense_target(
    target: GalerkinTargetManifest,
    *,
    adjoint: bool,
) -> None:
    """Enclose exact rational forward and actual-adjoint dense actions."""
    field = jnp.asarray(
        (1.125 - 0.75j, -0.3125 + 0.6875j, 0.21875 - 0.40625j),
        dtype=jnp.complex128,
    )
    result = enclose_galerkin_target_action(
        target,
        field,
        adjoint=adjoint,
    )
    exact = _exact_action(target, field, adjoint=adjoint)
    expected_direction = (
        GalerkinActionDirection.ADJOINT
        if adjoint
        else GalerkinActionDirection.FORWARD
    )

    assert result.direction is expected_direction
    assert (
        result.route
        is GalerkinActionErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE
    )
    assert bool(result.arithmetic_environment_supported)
    assert bool(result.finite_certificate)
    np.testing.assert_array_equal(np.asarray(result.submitted_field), field)
    for position, exact_value in enumerate(exact):
        exact_real = _decimal(exact_value[0])
        exact_imag = _decimal(exact_value[1])
        assert (
            _decimal_bound(result.algebraic_action_real_lower_bounds[position])
            <= exact_real
        )
        assert exact_real <= _decimal_bound(
            result.algebraic_action_real_upper_bounds[position]
        )
        assert (
            _decimal_bound(result.algebraic_action_imag_lower_bounds[position])
            <= exact_imag
        )
        assert exact_imag <= _decimal_bound(
            result.algebraic_action_imag_upper_bounds[position]
        )

    production_error = _exact_error_norm(result.production_action, exact)
    direct_error = _exact_error_norm(result.independent_direct_action, exact)
    assert production_error <= _decimal_bound(result.action_error_bound)
    assert direct_error <= _decimal_bound(
        result.direct_algebraic_action_error_bound
    )
    field_norm = _exact_error_norm(
        jnp.zeros_like(field),
        [_rational_complex(value) for value in np.asarray(field)],
    )
    assert _decimal_bound(result.field_norm_lower_bound) <= field_norm
    assert field_norm <= _decimal_bound(result.field_norm_upper_bound)
    assert production_error <= (
        _decimal_bound(result.per_state_relative_action_error_bound)
        * field_norm
    )


def test_zero_state_has_exact_zero_relative_action_bound(
    target: GalerkinTargetManifest,
) -> None:
    """Preserve the only sound zero-state relative-bound special case."""
    field = jnp.zeros(
        (target.support.state_indices.shape[0],), dtype=jnp.complex128
    )

    result = enclose_galerkin_target_action(target, field)

    np.testing.assert_array_equal(result.production_action, field)
    np.testing.assert_array_equal(result.independent_direct_action, field)
    assert float(result.action_error_bound) == 0.0
    assert float(result.per_state_relative_action_error_bound) == 0.0
    assert bool(result.finite_certificate)


def test_unsupported_normal_environment_is_typed_noncertificate(
    target: GalerkinTargetManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a failed normal-arithmetic probe to positive-infinity evidence."""
    monkeypatch.setattr(
        action_enclosures,
        "_action_arithmetic_environment_probes",
        lambda: (
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(False, dtype=jnp.bool_),
        ),
    )
    field = jnp.asarray(
        (0.5 + 0.25j, -0.75 + 0.125j, 0.375 - 0.5j),
        dtype=jnp.complex128,
    )

    result = enclose_galerkin_target_action(target, field)

    assert not bool(result.arithmetic_environment_supported)
    assert not bool(result.gradual_underflow_supported)
    assert not bool(result.finite_certificate)
    assert jnp.isinf(result.action_error_bound)
    assert result.action_error_bound > 0.0
    assert jnp.isinf(result.per_state_relative_action_error_bound)


@pytest.mark.parametrize("adjoint", [False, True])
def test_independent_residual_encloses_adversarial_cancellation(
    target: GalerkinTargetManifest,
    *,
    adjoint: bool,
) -> None:
    """Enclose a residual created by nearly canceling source and action."""
    field = jnp.asarray(
        (0.875 + 0.4375j, -0.625 + 0.1875j, 0.28125 - 0.53125j),
        dtype=jnp.complex128,
    )
    action = enclose_galerkin_target_action(
        target, field, adjoint=adjoint
    ).independent_direct_action
    perturbed_real = jnp.nextafter(jnp.real(action), jnp.inf)
    source = perturbed_real + 1j * jnp.imag(action)

    result = enclose_galerkin_residual(
        target,
        field,
        source,
        adjoint=adjoint,
    )
    exact_action = _exact_action(target, field, adjoint=adjoint)
    exact_source = [_rational_complex(value) for value in np.asarray(source)]
    exact_residual = [
        _complex_subtract(source_value, action_value)
        for source_value, action_value in zip(
            exact_source, exact_action, strict=True
        )
    ]
    evaluator_error = _exact_error_norm(result.formed_residual, exact_residual)
    exact_residual_norm = _exact_error_norm(
        jnp.zeros_like(result.formed_residual),
        exact_residual,
    )

    assert result.direction is (
        GalerkinActionDirection.ADJOINT
        if adjoint
        else GalerkinActionDirection.FORWARD
    )
    assert bool(result.finite_certificate)
    np.testing.assert_array_equal(result.submitted_field, field)
    np.testing.assert_array_equal(result.algebraic_source, source)
    assert evaluator_error <= _decimal_bound(
        result.residual_evaluator_error_bound
    )
    assert _decimal_bound(
        result.formed_residual_norm_lower_bound
    ) <= _exact_error_norm(
        jnp.zeros_like(result.formed_residual),
        [
            _rational_complex(value)
            for value in np.asarray(result.formed_residual)
        ],
    )
    assert exact_residual_norm <= _decimal_bound(
        result.algebraic_residual_norm_upper_bound
    )


def test_residual_overflow_is_typed_noncertificate(
    target: GalerkinTargetManifest,
) -> None:
    """Retain infinity as a typed residual noncertificate, never zero."""
    state_size = target.support.state_indices.shape[0]
    field = jnp.zeros((state_size,), dtype=jnp.complex128)
    source = jnp.full(
        (state_size,),
        jnp.finfo(jnp.float64).max + 0.0j,
        dtype=jnp.complex128,
    )

    result = enclose_galerkin_residual(target, field, source)

    assert not bool(result.finite_certificate)
    assert jnp.isinf(result.algebraic_residual_norm_upper_bound)
    assert result.algebraic_residual_norm_upper_bound > 0.0


def test_action_and_residual_compile_and_have_no_evidence_tangents(
    target: GalerkinTargetManifest,
) -> None:
    """Compile both routes and stop every evidence tangent."""
    field = jnp.asarray(
        (0.75 + 0.125j, -0.5 + 0.25j, 0.375 - 0.625j),
        dtype=jnp.complex128,
    )
    source = jnp.asarray(
        (0.25 - 0.5j, 0.125 + 0.375j, -0.75 + 0.625j),
        dtype=jnp.complex128,
    )
    compiled_action = jax.jit(
        lambda value: enclose_galerkin_target_action(
            target, value, adjoint=True
        )
    )(field)
    compiled_residual = jax.jit(
        lambda value, right_hand_side: enclose_galerkin_residual(
            target,
            value,
            right_hand_side,
            adjoint=True,
        )
    )(field, source)
    eager_action = enclose_galerkin_target_action(target, field, adjoint=True)
    eager_residual = enclose_galerkin_residual(
        target, field, source, adjoint=True
    )

    np.testing.assert_array_equal(
        compiled_action.production_action,
        eager_action.production_action,
    )
    np.testing.assert_array_equal(
        compiled_residual.formed_residual,
        eager_residual.formed_residual,
    )
    _, action_tangent = jax.jvp(
        lambda value: enclose_galerkin_target_action(
            target, value
        ).action_error_bound,
        (field,),
        (jnp.ones_like(field),),
    )
    _, residual_tangent = jax.jvp(
        lambda value, right_hand_side: enclose_galerkin_residual(
            target, value, right_hand_side
        ).algebraic_residual_norm_upper_bound,
        (field, source),
        (jnp.ones_like(field), jnp.ones_like(source)),
    )
    assert float(action_tangent) == 0.0
    assert float(residual_tangent) == 0.0
