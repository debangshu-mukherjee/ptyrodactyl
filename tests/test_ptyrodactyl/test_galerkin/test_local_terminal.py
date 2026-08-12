r"""Tests for authenticated local coordinate-terminal current operators."""

from __future__ import annotations

import functools
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.local_terminal as local_terminal
from ptyrodactyl._tools import (
    RootEnclosureError,
    fraction_upper_float,
    sqrt_fraction_upper,
)
from ptyrodactyl.galerkin.absorber import (
    certify_axial_cap_floor,
    certify_axial_cell_absorber,
    realize_axial_cell_absorber,
)
from ptyrodactyl.galerkin.acquisition import (
    check_galerkin_acquisition_support,
)
from ptyrodactyl.galerkin.local_cell import (
    realize_local_cell_galerkin_potential,
)
from ptyrodactyl.galerkin.local_cell_certification import (
    certify_local_cell_galerkin_potential,
)
from ptyrodactyl.galerkin.local_cell_interaction import (
    certify_local_cell_exact_compression,
    create_local_cell_interaction_core,
)
from ptyrodactyl.galerkin.local_cell_system import (
    compose_local_cell_galerkin_target,
    prepare_local_cell_galerkin_target,
)
from ptyrodactyl.galerkin.local_terminal import (
    apply_local_terminal_current,
    apply_local_terminal_normal_derivative,
    apply_local_terminal_normal_derivative_adjoint,
    apply_local_terminal_trace,
    apply_local_terminal_trace_adjoint,
    certify_local_terminal_current_operator,
    enclose_local_terminal_current,
    enclose_local_terminal_current_action,
    prepare_local_terminal_current,
    prepare_local_terminal_current_action,
    prepare_local_terminal_current_operator,
)
from ptyrodactyl.types import C_LIGHT, E_CHARGE, HBAR, M_E
from ptyrodactyl.types.acquisition_types import (
    GalerkinBackwardDisposition,
    GalerkinTerminalSide,
)
from ptyrodactyl.types.born_potential_types import (
    create_galerkin_product_support,
)
from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellTargetManifest,
)
from ptyrodactyl.types.local_cell_types import (
    create_local_cell_potential_3d,
)
from ptyrodactyl.types.local_terminal_types import (
    GalerkinLocalCurrentOperatorCertificate,
    GalerkinLocalCurrentOperatorFailure,
    GalerkinLocalTerminalActionFailure,
    GalerkinLocalTerminalCurrentFailure,
    GalerkinLocalTerminalScope,
    GalerkinPreparedLocalCurrentOperator,
)
from tests._galerkin_target_fixture import checked_acquisition
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_represented_sources as represented_tests,
)

_BUDGET = 64
_COORDINATE = np.float64(0.2857142857142857)
_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944"
    "5923078164062862089986280348253421170679"
)

type _DecimalComplex = tuple[Decimal, Decimal]
type _RationalComplex = tuple[Fraction, Fraction]


def _decimal_float(value: object) -> Decimal:
    """Return the exact Decimal value of one stored binary64 scalar."""
    return Decimal.from_float(float(np.asarray(value)))


def _decimal_sin_cos(angle: Decimal) -> tuple[Decimal, Decimal]:
    """Evaluate sine and cosine independently with a high-precision series."""
    two_pi = Decimal(2) * _PI
    reduced = angle % two_pi
    if reduced > _PI:
        reduced -= two_pi
    squared = reduced * reduced
    sine = reduced
    sine_term = reduced
    cosine = Decimal(1)
    cosine_term = Decimal(1)
    for order in range(1, 90):
        sine_term *= -squared / Decimal((2 * order) * (2 * order + 1))
        cosine_term *= -squared / Decimal((2 * order - 1) * (2 * order))
        sine += sine_term
        cosine += cosine_term
    return +sine, +cosine


def _decimal_add(
    left: _DecimalComplex, right: _DecimalComplex
) -> _DecimalComplex:
    """Add two high-precision complex pairs."""
    return left[0] + right[0], left[1] + right[1]


def _decimal_multiply(
    left: _DecimalComplex, right: _DecimalComplex
) -> _DecimalComplex:
    """Multiply two high-precision complex pairs."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _rational_add(
    left: _RationalComplex, right: _RationalComplex
) -> _RationalComplex:
    """Add two exact stored-value complex pairs."""
    return left[0] + right[0], left[1] + right[1]


def _rational_multiply(
    left: _RationalComplex, right: _RationalComplex
) -> _RationalComplex:
    """Multiply two exact stored-value complex pairs."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _rational_complex(value: complex) -> _RationalComplex:
    """Interpret one stored complex128 value exactly."""
    return (
        Fraction.from_float(float(np.real(value))),
        Fraction.from_float(float(np.imag(value))),
    )


def _dense_operators(
    certificate: GalerkinLocalCurrentOperatorCertificate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble independent dense matrices from the frozen public payload."""
    state_size = certificate.target.state_indices.shape[0]
    fiber_size = certificate.scope_transverse_indices.shape[0]
    trace = np.zeros((fiber_size, state_size), dtype=np.complex128)
    normal = np.zeros_like(trace)
    rows = np.asarray(certificate.state_to_fiber_rows)
    selected = np.asarray(certificate.selected_state_mask)
    for column in range(state_size):
        if bool(selected[column]):
            trace[rows[column], column] = np.asarray(
                certificate.trace_frozen_coefficients
            )[column]
            normal[rows[column], column] = np.asarray(
                certificate.normal_frozen_coefficients
            )[column]
    current = (trace.conj().T @ normal - normal.conj().T @ trace) / (2.0j)
    return trace, normal, current


def _assert_dense_actions(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: jax.Array,
    prepared: GalerkinPreparedLocalCurrentOperator | None = None,
) -> None:
    """Check every public frozen map against literal dense matrices."""
    checked_prepared = (
        local_terminal._make_prepared_local_current_operator(certificate)
        if prepared is None
        else prepared
    )
    trace, normal, current = _dense_operators(certificate)
    terminal = jnp.asarray(
        np.linspace(0.2, 0.8, trace.shape[0])
        + 1.0j * np.linspace(-0.3, 0.1, trace.shape[0]),
        dtype=jnp.complex128,
    )
    assert_allclose(
        apply_local_terminal_trace(checked_prepared, field), trace @ field
    )
    assert_allclose(
        apply_local_terminal_normal_derivative(checked_prepared, field),
        normal @ field,
    )
    assert_allclose(
        apply_local_terminal_trace_adjoint(checked_prepared, terminal),
        trace.conj().T @ terminal,
    )
    assert_allclose(
        apply_local_terminal_normal_derivative_adjoint(
            checked_prepared, terminal
        ),
        normal.conj().T @ terminal,
    )
    assert_allclose(
        apply_local_terminal_current(checked_prepared, field), current @ field
    )
    assert_allclose(current, current.conj().T, rtol=0.0, atol=2.0e-15)


@functools.lru_cache(maxsize=1)
def _positive_axis_zero_certificate() -> (
    GalerkinLocalCurrentOperatorCertificate
):
    """Return one nonzero-coordinate positive-x full-scope operator."""
    certificate = certify_local_terminal_current_operator(
        represented_tests._target("local-terminal-positive-axis-zero"),
        terminal_plane_coordinate=_COORDINATE,
        current_scope=GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        maximum_direct_pairs=_BUDGET,
    )
    assert bool(certificate.current_operator_eligible)
    return certificate


@functools.lru_cache(maxsize=1)
def _positive_axis_zero_prepared() -> GalerkinPreparedLocalCurrentOperator:
    """Cross the public operator replay boundary exactly once."""
    return prepare_local_terminal_current_operator(
        _positive_axis_zero_certificate(), maximum_direct_pairs=_BUDGET
    )


def _positive_field() -> np.ndarray:
    """Return the shared exact stored positive-axis submitted state."""
    return np.asarray(
        (0.7 - 0.2j, -0.3 + 0.8j, 0.4 + 0.1j),
        dtype=np.complex128,
    )


@functools.lru_cache(maxsize=1)
def _positive_action_enclosure():
    """Cross the public frozen-action enclosure boundary exactly once."""
    return enclose_local_terminal_current_action(
        _positive_axis_zero_certificate(),
        _positive_field(),
        maximum_direct_pairs=_BUDGET,
    )


@functools.lru_cache(maxsize=1)
def _positive_current_diagnostic():
    """Cross the public exact-current enclosure boundary exactly once."""
    return enclose_local_terminal_current(
        _positive_axis_zero_certificate(),
        _positive_field(),
        maximum_direct_pairs=_BUDGET,
    )


@functools.lru_cache(maxsize=1)
def _negative_axis_two_target() -> GalerkinLocalCellTargetManifest:
    """Build one genuine negative-z target with an omitted retained fiber."""
    state = jnp.asarray(
        [
            (0, transverse, normal)
            for transverse in (0, 1)
            for normal in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    interaction = jnp.asarray(
        [
            (0, transverse, normal)
            for transverse in range(-1, 2)
            for normal in range(-2, 3)
        ],
        dtype=jnp.int64,
    )
    absorber = jnp.asarray(
        [
            (0, transverse, normal)
            for transverse in range(-1, 2)
            for normal in range(-5, 6)
        ],
        dtype=jnp.int64,
    )
    work = jnp.asarray(
        [
            (0, transverse, normal)
            for transverse in range(-2, 3)
            for normal in range(-6, 7)
        ],
        dtype=jnp.int64,
    )
    support = create_galerkin_product_support(
        state_indices=state,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(1, 5, 13),
    )
    cells = jnp.zeros((3, 2, 1), dtype=jnp.float64)
    potential = create_local_cell_potential_3d(
        cells,
        cell_size=(1.0, 1.0, 1.0),
        box_size=(1.0, 2.0, 3.0),
        cell_center_origin=(0.2, 0.1, 0.125),
        reference_value=0.0,
        reference_semantics="local terminal negative-z exact-zero reference",
        producer="local-terminal-selected-sector-test-v1",
        provenance_hash="b" * 64,
        producer_coefficient_normalization="producer metadata only",
        producer_bandwidth=1.0,
    )
    full = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
        terminal_side=GalerkinTerminalSide.NEGATIVE,
        backward_disposition=GalerkinBackwardDisposition.REPRESENTED,
        claims_backscatter=True,
    )
    selected_preterminal = state[state[:, 1] == 0]
    selected_manifest = replace(
        full.manifest,
        preterminal_indices=selected_preterminal,
        transverse_indices=jnp.asarray(((0, 0),), dtype=jnp.int64),
    )
    selected = check_galerkin_acquisition_support(selected_manifest)
    assert bool(selected.support_eligible)
    realization = realize_local_cell_galerkin_potential(potential, selected)
    certificate = certify_local_cell_galerkin_potential(
        realization, maximum_direct_terms=1000
    )
    compression = certify_local_cell_exact_compression(
        certificate, accelerating_voltage_kv=200.0
    )
    core = create_local_cell_interaction_core(compression)
    absorber_realization = realize_axial_cell_absorber(
        core,
        jnp.asarray([1.0, 0.0, 0.5], dtype=jnp.float64),
        terminal_axis=2,
        plateau_start=0,
        plateau_count=1,
        plateau_floor=jnp.asarray(1.0, dtype=jnp.float64),
        zero_start=1,
        zero_count=1,
        exact_cap_scale=jnp.asarray(0.25, dtype=jnp.float64),
    )
    absorber_certificate = certify_axial_cell_absorber(absorber_realization)
    proof = certify_axial_cap_floor(
        absorber_certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    return compose_local_cell_galerkin_target(
        proof, target_name="local-terminal-negative-axis-two"
    )


@functools.lru_cache(maxsize=1)
def _negative_axis_two_prepared_target() -> GalerkinLocalCellTargetManifest:
    """Replay the second public parent exactly once for both scopes."""
    return prepare_local_cell_galerkin_target(_negative_axis_two_target())


@functools.lru_cache(maxsize=2)
def _negative_axis_two_certificate(
    scope: GalerkinLocalTerminalScope,
) -> GalerkinLocalCurrentOperatorCertificate:
    """Certify an arithmetic-only scope from the one replayed parent."""
    return local_terminal._certify_prepared_operator(
        _negative_axis_two_prepared_target(),
        np.float64(-0.375),
        scope,
        _BUDGET,
    )


def _exact_axis_aligned_coefficients(
    certificate: GalerkinLocalCurrentOperatorCertificate,
) -> tuple[list[_DecimalComplex], list[_DecimalComplex]]:
    """Evaluate exact axis-aligned T/N coefficients with Decimal arithmetic."""
    target = certificate.target
    axis = target.acquisition.terminal_axis
    side = (
        Decimal(1)
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else Decimal(-1)
    )
    with localcontext() as context:
        context.prec = 100
        mass = _decimal_float(M_E)
        charge = _decimal_float(E_CHARGE)
        speed = _decimal_float(C_LIGHT)
        hbar = _decimal_float(HBAR)
        voltage = _decimal_float(target.accelerating_voltage_kv) * Decimal(
            1000
        )
        k_zero = (
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
        ).sqrt()
        length = _decimal_float(target.box_lengths[axis])
        coordinate = _decimal_float(certificate.terminal_plane_coordinate)
        normalization = Decimal(1) / length.sqrt()
        trace: list[_DecimalComplex] = []
        normal: list[_DecimalComplex] = []
        indices = np.asarray(target.state_indices)[:, axis]
        selected = np.asarray(certificate.selected_state_mask)
        for position, index in enumerate(indices):
            if not bool(selected[position]):
                trace.append((Decimal(0), Decimal(0)))
                normal.append((Decimal(0), Decimal(0)))
                continue
            angle = (
                Decimal(2) * _PI * Decimal(int(index)) * coordinate / length
            )
            sine, cosine = _decimal_sin_cos(angle)
            trace_value = (normalization * cosine, normalization * sine)
            wavevector = side * (
                k_zero + Decimal(2) * _PI * Decimal(int(index)) / length
            )
            trace.append(trace_value)
            normal.append(
                (-wavevector * trace_value[1], wavevector * trace_value[0])
            )
    return trace, normal


def _contains_decimal(lower: object, upper: object, value: Decimal) -> bool:
    """Check one exact Decimal value against stored binary64 endpoints."""
    return _decimal_float(lower) <= value <= _decimal_float(upper)


def test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity() -> None:
    """Match T/N/F, actual adjoints, JIT, and VJP at nonzero xi.

    :see: :func:`ptyrodactyl.galerkin.apply_local_terminal_current`
    :see: :func:`ptyrodactyl.galerkin.apply_local_terminal_normal_derivative`
    :see: :func:`ptyrodactyl.galerkin.\
apply_local_terminal_normal_derivative_adjoint`
    :see: :func:`ptyrodactyl.galerkin.apply_local_terminal_trace`
    :see: :func:`ptyrodactyl.galerkin.apply_local_terminal_trace_adjoint`
    :see: :func:`ptyrodactyl.galerkin.\
certify_local_terminal_current_operator`
    """
    certificate = _positive_axis_zero_certificate()
    assert certificate.terminal_axis == 0
    assert certificate.terminal_side is GalerkinTerminalSide.POSITIVE
    assert float(certificate.terminal_plane_coordinate) != 0.0
    field = jnp.asarray(_positive_field())
    prepared = _positive_axis_zero_prepared()
    _assert_dense_actions(certificate, field, prepared)
    _, _, dense_current = _dense_operators(certificate)

    def frozen_action(values):
        """Close over the host-prepared trust marker."""
        return apply_local_terminal_current(prepared, values)

    compiled = jax.jit(frozen_action)(field)
    assert_allclose(compiled, dense_current @ field)
    cotangent = jnp.asarray(
        (-0.1 + 0.4j, 0.6 - 0.2j, 0.3 + 0.7j),
        dtype=jnp.complex128,
    )
    _, pullback = jax.vjp(frozen_action, field)
    _, dense_pullback = jax.vjp(
        lambda values: jnp.asarray(dense_current) @ values, field
    )
    assert_allclose(pullback(cotangent)[0], dense_pullback(cotangent)[0])


def test_action_and_exact_current_intervals_match_independent_oracles() -> (
    None
):
    """Enclose high-precision T/N, frozen F action, norms, and current.

    :see: :func:`ptyrodactyl.galerkin.enclose_local_terminal_current`
    :see: :func:`ptyrodactyl.galerkin.enclose_local_terminal_current_action`
    """
    certificate = _positive_axis_zero_certificate()
    field = _positive_field()
    exact_trace, exact_normal = _exact_axis_aligned_coefficients(certificate)
    trace_rectangles = certificate.exact_trace_coefficient_rectangles
    normal_rectangles = certificate.exact_normal_coefficient_rectangles
    for index, (trace_value, normal_value) in enumerate(
        zip(exact_trace, exact_normal, strict=True)
    ):
        assert _contains_decimal(
            trace_rectangles.real_lower_bounds[index],
            trace_rectangles.real_upper_bounds[index],
            trace_value[0],
        )
        assert _contains_decimal(
            trace_rectangles.imag_lower_bounds[index],
            trace_rectangles.imag_upper_bounds[index],
            trace_value[1],
        )
        assert _contains_decimal(
            normal_rectangles.real_lower_bounds[index],
            normal_rectangles.real_upper_bounds[index],
            normal_value[0],
        )
        assert _contains_decimal(
            normal_rectangles.imag_lower_bounds[index],
            normal_rectangles.imag_upper_bounds[index],
            normal_value[1],
        )

    rows = np.asarray(certificate.state_to_fiber_rows)
    fiber_size = certificate.scope_transverse_indices.shape[0]
    with localcontext() as context:
        context.prec = 100
        trace_norm = max(
            sum(
                value[0] * value[0] + value[1] * value[1]
                for row, value in zip(rows, exact_trace, strict=True)
                if row == fiber
            ).sqrt()
            for fiber in range(fiber_size)
        )
        normal_norm = max(
            sum(
                value[0] * value[0] + value[1] * value[1]
                for row, value in zip(rows, exact_normal, strict=True)
                if row == fiber
            ).sqrt()
            for fiber in range(fiber_size)
        )
    assert (
        _decimal_float(certificate.exact_trace_operator_norm_upper_bound)
        >= trace_norm
    )
    assert (
        _decimal_float(certificate.exact_normal_operator_norm_upper_bound)
        >= normal_norm
    )

    action = _positive_action_enclosure()
    assert bool(action.current_action_eligible)
    frozen_trace = [
        _rational_complex(value)
        for value in np.asarray(certificate.trace_frozen_coefficients)
    ]
    frozen_normal = [
        _rational_complex(value)
        for value in np.asarray(certificate.normal_frozen_coefficients)
    ]
    exact_field = [_rational_complex(value) for value in field]
    trace_sums = [(Fraction(0), Fraction(0)) for _ in range(fiber_size)]
    normal_sums = [(Fraction(0), Fraction(0)) for _ in range(fiber_size)]
    for index, value in enumerate(exact_field):
        row = int(rows[index])
        trace_sums[row] = _rational_add(
            trace_sums[row], _rational_multiply(frozen_trace[index], value)
        )
        normal_sums[row] = _rational_add(
            normal_sums[row], _rational_multiply(frozen_normal[index], value)
        )
    exact_action: list[_RationalComplex] = []
    minus_half_i = (Fraction(0), Fraction(-1, 2))
    for index in range(field.shape[0]):
        row = int(rows[index])
        conjugate_trace = (frozen_trace[index][0], -frozen_trace[index][1])
        conjugate_normal = (
            frozen_normal[index][0],
            -frozen_normal[index][1],
        )
        left = _rational_multiply(conjugate_trace, normal_sums[row])
        right = _rational_multiply(conjugate_normal, trace_sums[row])
        exact_action.append(
            _rational_multiply(
                (left[0] - right[0], left[1] - right[1]), minus_half_i
            )
        )
    action_rectangles = action.frozen_action_rectangles
    squared_error = Fraction(0)
    for index, value in enumerate(exact_action):
        assert (
            Fraction.from_float(
                float(action_rectangles.real_lower_bounds[index])
            )
            <= value[0]
            <= Fraction.from_float(
                float(action_rectangles.real_upper_bounds[index])
            )
        )
        assert (
            Fraction.from_float(
                float(action_rectangles.imag_lower_bounds[index])
            )
            <= value[1]
            <= Fraction.from_float(
                float(action_rectangles.imag_upper_bounds[index])
            )
        )
        stored = _rational_complex(np.asarray(action.production_action)[index])
        component_squared = (stored[0] - value[0]) ** 2 + (
            stored[1] - value[1]
        ) ** 2
        squared_error += component_squared
        assert (
            component_squared
            <= Fraction.from_float(float(action.component_error_bounds[index]))
            ** 2
        )
    assert (
        squared_error
        <= Fraction.from_float(float(action.action_error_upper_bound)) ** 2
    )

    diagnostic = _positive_current_diagnostic()
    assert bool(diagnostic.current_diagnostic_eligible)
    exact_field_decimal = [
        (_decimal_float(value.real), _decimal_float(value.imag))
        for value in field
    ]
    trace_values = [(Decimal(0), Decimal(0)) for _ in range(fiber_size)]
    normal_values = [(Decimal(0), Decimal(0)) for _ in range(fiber_size)]
    for index, value in enumerate(exact_field_decimal):
        row = int(rows[index])
        trace_values[row] = _decimal_add(
            trace_values[row], _decimal_multiply(exact_trace[index], value)
        )
        normal_values[row] = _decimal_add(
            normal_values[row], _decimal_multiply(exact_normal[index], value)
        )
    exact_current = sum(
        trace[0] * normal[1] - trace[1] * normal[0]
        for trace, normal in zip(trace_values, normal_values, strict=True)
    )
    assert _contains_decimal(
        diagnostic.exact_reduced_current_lower_bound,
        diagnostic.exact_reduced_current_upper_bound,
        exact_current,
    )
    assert float(diagnostic.reduced_current) == pytest.approx(
        float(np.real(np.vdot(field, np.asarray(action.production_action)))),
        rel=0.0,
        abs=0.0,
    )
    assert int(action.direct_work_count) == 2 * field.size
    assert int(diagnostic.direct_work_count) == 3 * field.size + fiber_size
    assert "excludes uniform epsilon_F" in action.error_scope
    assert "never added" in diagnostic.error_scope

    trace_error = float(certificate.trace_operator_error_upper_bound)
    normal_error = float(certificate.normal_operator_error_upper_bound)
    exact_trace_bound = float(
        certificate.exact_trace_operator_norm_upper_bound
    )
    exact_normal_bound = float(
        certificate.exact_normal_operator_norm_upper_bound
    )

    def upward(value: float) -> float:
        """Move one positive rounded operation one binary64 neighbor up."""
        return float(np.nextafter(np.float64(value), np.float64(np.inf)))

    expected_current_error = upward(
        upward(trace_error * exact_normal_bound)
        + upward(upward(exact_trace_bound + trace_error) * normal_error)
    )
    assert float(certificate.current_operator_error_upper_bound) == (
        expected_current_error
    )
    assert certificate.fixed_linear_error_formula.count("LVT.55a5") == 1
    with localcontext() as context:
        context.prec = 100
        speed = _decimal_float(C_LIGHT)
        exact_scale = (
            _decimal_float(HBAR)
            * speed
            * speed
            * (Decimal(10) ** 20)
            / (
                _decimal_float(M_E) * speed * speed
                + _decimal_float(E_CHARGE)
                * _decimal_float(certificate.target.accelerating_voltage_kv)
                * Decimal(1000)
            )
        )
    assert _contains_decimal(
        certificate.exact_number_current_scale_lower_bound,
        certificate.exact_number_current_scale_upper_bound,
        exact_scale,
    )


def test_full_and_selected_scopes_omit_unselected_sector_current() -> None:
    """Keep a negative-z omitted fiber in full scope and absent in selected.

    :see: :class:`ptyrodactyl.types.GalerkinLocalTerminalScope`
    """
    target = _negative_axis_two_prepared_target()
    full = _negative_axis_two_certificate(
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS
    )
    selected = _negative_axis_two_certificate(
        GalerkinLocalTerminalScope.SELECTED_PRETERMINAL_FIBERS
    )
    assert bool(full.current_operator_eligible)
    assert bool(selected.current_operator_eligible)
    assert full.terminal_axis == selected.terminal_axis == 2
    assert full.terminal_side is GalerkinTerminalSide.NEGATIVE
    assert selected.terminal_side is GalerkinTerminalSide.NEGATIVE
    state = np.asarray(target.state_indices)
    omitted = state[:, 1] == 1
    field = jnp.asarray(
        np.where(omitted, 0.7 + 0.3j * (state[:, 2] + 2), 0.0 + 0.0j),
        dtype=jnp.complex128,
    )
    full_prepared = local_terminal._make_prepared_local_current_operator(full)
    selected_prepared = local_terminal._make_prepared_local_current_operator(
        selected
    )
    _assert_dense_actions(full, field, full_prepared)
    _assert_dense_actions(selected, field, selected_prepared)
    full_action = np.asarray(
        apply_local_terminal_current(full_prepared, field)
    )
    selected_action = np.asarray(
        apply_local_terminal_current(selected_prepared, field)
    )
    assert np.linalg.norm(full_action) > 1.0e-6
    assert_array_equal(selected_action, 0.0j)
    assert np.any(np.asarray(full.selected_state_mask)[omitted])
    assert not np.any(np.asarray(selected.selected_state_mask)[omitted])
    _, exact_selected_normal = _exact_axis_aligned_coefficients(selected)
    selected_position = int(
        np.flatnonzero(np.asarray(selected.selected_state_mask))[0]
    )
    selected_rectangles = selected.exact_normal_coefficient_rectangles
    selected_normal = exact_selected_normal[selected_position]
    assert _contains_decimal(
        selected_rectangles.real_lower_bounds[selected_position],
        selected_rectangles.real_upper_bounds[selected_position],
        selected_normal[0],
    )
    assert _contains_decimal(
        selected_rectangles.imag_lower_bounds[selected_position],
        selected_rectangles.imag_upper_bounds[selected_position],
        selected_normal[1],
    )
    full_current = local_terminal._enclose_current_prepared(
        full_prepared, field, _BUDGET
    )
    selected_current = local_terminal._enclose_current_prepared(
        selected_prepared, field, _BUDGET
    )
    assert abs(float(full_current.reduced_current)) > 1.0e-6
    assert float(selected_current.reduced_current) == 0.0
    assert float(selected_current.exact_reduced_current_lower_bound) <= 0.0
    assert float(selected_current.exact_reduced_current_upper_bound) >= 0.0


def test_prepare_rejects_operator_action_current_and_parent_forgeries() -> (
    None
):
    """Replay all evidence and keep a directly built wrapper nonauthoritative.

    :see: :func:`ptyrodactyl.galerkin.prepare_local_terminal_current_operator`
    :see: :func:`ptyrodactyl.galerkin.prepare_local_terminal_current_action`
    :see: :func:`ptyrodactyl.galerkin.prepare_local_terminal_current`
    """
    certificate = _positive_axis_zero_certificate()
    field = jnp.asarray(_positive_field())
    with pytest.raises(TypeCheckError):
        apply_local_terminal_current(certificate, field)
    forged_wrapper = GalerkinPreparedLocalCurrentOperator(
        certificate=certificate
    )
    canonical_prepared = _positive_axis_zero_prepared()
    assert_allclose(
        apply_local_terminal_current(forged_wrapper, field),
        apply_local_terminal_current(canonical_prepared, field),
    )
    with pytest.raises(TypeCheckError):
        enclose_local_terminal_current_action(
            forged_wrapper,  # type: ignore[arg-type]
            field,
            maximum_direct_pairs=_BUDGET,
        )

    forged_operator = eqx.tree_at(
        lambda value: value.maximum_direct_pairs,
        certificate,
        certificate.maximum_direct_pairs + 1,
    )
    with pytest.raises(ValueError, match="complete target/operator/policy"):
        prepare_local_terminal_current_operator(
            forged_operator, maximum_direct_pairs=_BUDGET
        )
    action = _positive_action_enclosure()
    assert (
        prepare_local_terminal_current_action(
            action, maximum_direct_pairs=_BUDGET
        ).action_evidence_digest
        == action.action_evidence_digest
    )
    forged_action = eqx.tree_at(
        lambda value: value.production_action,
        action,
        action.production_action.at[0].add(1.0 + 0.0j),
    )
    with pytest.raises(ValueError, match="complete host replay"):
        prepare_local_terminal_current_action(
            forged_action, maximum_direct_pairs=_BUDGET
        )
    diagnostic = _positive_current_diagnostic()
    assert (
        prepare_local_terminal_current(
            diagnostic, maximum_direct_pairs=_BUDGET
        ).diagnostic_evidence_digest
        == diagnostic.diagnostic_evidence_digest
    )
    forged_current = eqx.tree_at(
        lambda value: value.exact_reduced_current_upper_bound,
        diagnostic,
        diagnostic.exact_reduced_current_upper_bound + 1.0,
    )
    with pytest.raises(ValueError, match="complete host replay"):
        prepare_local_terminal_current(
            forged_current, maximum_direct_pairs=_BUDGET
        )


def test_typed_work_root_range_and_subnormal_dispositions(monkeypatch) -> None:
    """Exercise distinct policy, root, host range, and subnormal outcomes."""
    canonical = _positive_axis_zero_certificate()
    target = canonical.target
    budget_failed = local_terminal._certify_prepared_operator(
        target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        5,
    )
    assert budget_failed.failure is (
        GalerkinLocalCurrentOperatorFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    assert not bool(budget_failed.current_operator_eligible)
    diagnostic_budget = int(canonical.action_work_count)
    boundary_certificate = local_terminal._certify_prepared_operator(
        target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        diagnostic_budget,
    )
    assert bool(boundary_certificate.current_operator_eligible)
    boundary_prepared = local_terminal._make_prepared_local_current_operator(
        boundary_certificate
    )
    current_failed = local_terminal._enclose_current_prepared(
        boundary_prepared,
        np.asarray((1.0 + 0.0j,) * 3, dtype=np.complex128),
        diagnostic_budget,
    )
    assert current_failed.failure is (
        GalerkinLocalTerminalCurrentFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )

    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    with pytest.raises(ValueError, match="normal-or-zero"):
        certify_local_terminal_current_operator(
            target,
            terminal_plane_coordinate=subnormal,
            current_scope=GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
            maximum_direct_pairs=_BUDGET,
        )
    subnormal_action = local_terminal._enclose_action_prepared(
        _positive_axis_zero_prepared(),
        np.asarray((subnormal + 0.0j, 1.0 + 0.0j, 0.0j)),
        _BUDGET,
    )
    assert subnormal_action.failure is (
        GalerkinLocalTerminalActionFailure.ARITHMETIC_RANGE_FAILURE
    )

    def fail_root(_turn):
        """Force the typed rational-turn root-enclosure path."""
        raise RootEnclosureError("forced local-terminal root failure")

    monkeypatch.setattr(local_terminal, "rational_turn_exponential", fail_root)
    root_failed = local_terminal._certify_prepared_operator(
        target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        _BUDGET,
    )
    assert root_failed.failure is (
        GalerkinLocalCurrentOperatorFailure.ROOT_ENCLOSURE_FAILURE
    )
    monkeypatch.undo()

    monkeypatch.setattr(
        local_terminal, "_normal_or_zero", lambda _value: False
    )
    range_failed = local_terminal._certify_prepared_operator(
        target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        _BUDGET,
    )
    assert range_failed.failure is (
        GalerkinLocalCurrentOperatorFailure.ARITHMETIC_RANGE_FAILURE
    )
    monkeypatch.undo()

    monkeypatch.setattr(
        local_terminal, "host_binary64_supported", lambda: False
    )
    host_failed = local_terminal._certify_prepared_operator(
        target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        _BUDGET,
    )
    assert host_failed.failure is (
        GalerkinLocalCurrentOperatorFailure.HOST_ARITHMETIC_UNSUPPORTED
    )
    monkeypatch.undo()

    oversized = np.iinfo(np.int64).max + 1
    monkeypatch.setattr(
        local_terminal,
        "_work_counts",
        lambda _state, _fiber: (oversized, oversized + 1),
    )
    overflow = local_terminal._certify_prepared_operator(
        target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        _BUDGET,
    )
    assert overflow.failure is (
        GalerkinLocalCurrentOperatorFailure.DIRECT_WORK_COUNT_OVERFLOW
    )
    assert int(overflow.action_work_count) == 0
    assert overflow.action_work_count_exact == str(oversized)


def test_nonsquare_length_normal_norm_uses_outward_reciprocal_root() -> None:
    """Regress the lower-biased division by an upper square-root bound."""
    length_value = np.float64(4.021897810218978)
    length = Fraction.from_float(float(length_value))
    normalized_magnitude = (
        local_terminal._normalized_wavevector_magnitude_upper(
            (Fraction(1), Fraction(1)), length
        )
    )
    reported = Decimal.from_float(
        fraction_upper_float(
            sqrt_fraction_upper(normalized_magnitude * normalized_magnitude)
        )
    )
    with localcontext() as context:
        context.prec = 160
        dense_row_norm = (
            Decimal(1)
            / (Decimal(length.numerator) / Decimal(length.denominator)).sqrt()
        )
        old_rational = sqrt_fraction_upper(
            (Fraction(1) / sqrt_fraction_upper(length)) ** 2
        )
        correct_rational = Decimal(normalized_magnitude.numerator) / Decimal(
            normalized_magnitude.denominator
        )
        old_rational_value = Decimal(old_rational.numerator) / Decimal(
            old_rational.denominator
        )
    assert reported >= dense_row_norm
    assert correct_rational >= dense_row_norm
    assert old_rational_value < dense_row_norm


def test_identity_digest_is_separate_from_policy_evidence() -> None:
    """Keep frozen operator identity stable while policy evidence changes."""
    first = _positive_axis_zero_certificate()
    second = local_terminal._certify_prepared_operator(
        first.target,
        _COORDINATE,
        GalerkinLocalTerminalScope.FULL_STATE_FIBERS,
        _BUDGET + 1,
    )
    assert first.operator_identity_digest == second.operator_identity_digest
    assert first.operator_evidence_digest != second.operator_evidence_digest
    assert_allclose(
        first.trace_frozen_coefficients,
        second.trace_frozen_coefficients,
        rtol=0.0,
        atol=0.0,
    )
