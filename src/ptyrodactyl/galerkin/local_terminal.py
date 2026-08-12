r"""Build authenticated local coordinate trace and current operators.

Extended Summary
----------------
This leaf consumes only the completed ``LOCAL_CELL_LVT1`` target route.  It
binds an exact stored plane coordinate and either the complete retained
transverse-fiber set or the acquisition-selected complete preterminal set.
Frozen carrier-stripped trace and side-oriented normal matrices expose their
literal coefficient adjoints and the implicit Hermitian LVT.55a current.
Host evidence separately certifies uniform exact-target error, a frozen
per-call action, and one direct exact-target current scalar.

Routine Listings
----------------
:func:`apply_local_terminal_current`
    Apply the implicit actual frozen Hermitian current matrix.
:func:`apply_local_terminal_normal_derivative`
    Apply the side-oriented frozen physical normal trace.
:func:`apply_local_terminal_normal_derivative_adjoint`
    Apply the actual conjugate transpose of the frozen normal trace.
:func:`apply_local_terminal_trace`
    Apply the frozen carrier-stripped trace at the bound coordinate.
:func:`apply_local_terminal_trace_adjoint`
    Apply the actual conjugate transpose of the frozen trace.
:func:`certify_local_terminal_current_operator`
    Replay a local target and certify one scoped coordinate operator.
:func:`enclose_local_terminal_current`
    Replay the operator and enclose one direct exact-target current.
:func:`enclose_local_terminal_current_action`
    Replay the operator and enclose one frozen current action.
:func:`prepare_local_terminal_current`
    Replay complete operator, action, exact current, policy, and evidence.
:func:`prepare_local_terminal_current_action`
    Replay complete operator, field, frozen action, policy, and evidence.
:func:`prepare_local_terminal_current_operator`
    Replay raw operator storage and return the prepared JIT capability.

Notes
-----
No carrier or function in this leaf asserts a vacuum branch or detector
claim.  Uniform ``epsilon_F`` is stored only in the operator certificate.
Per-call frozen arithmetic and the direct exact-target scalar interval are
exclusive routes and never add ``epsilon_F`` again.
"""

from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Complex, Complex128, jaxtyped

from ptyrodactyl._tools import (
    ComplexRectangle,
    RootEnclosureError,
    all_normal_arithmetic_supported,
    coefficient_error_fraction,
    complex_rectangle_multiply,
    conjugate_rectangle,
    fraction_lower_float,
    fraction_upper_float,
    has_subnormal_components,
    host_binary64_supported,
    mathematical_pi_interval,
    pairwise_rectangle_sum,
    rational_turn_exponential,
    real_interval_product,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
    upward_add,
    upward_multiply,
)
from ptyrodactyl.types import C_LIGHT, E_CHARGE, HBAR, M_E
from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellTargetManifest,
)
from ptyrodactyl.types.local_terminal_types import (
    GalerkinLocalCoordinateCauchyCurrent,
    GalerkinLocalCurrentOperatorCertificate,
    GalerkinLocalCurrentOperatorFailure,
    GalerkinLocalTerminalActionFailure,
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalCurrentActionEnclosure,
    GalerkinLocalTerminalCurrentFailure,
    GalerkinLocalTerminalScope,
    GalerkinPreparedLocalCurrentOperator,
    _make_local_coordinate_cauchy_current,
    _make_local_current_operator_certificate,
    _make_local_terminal_current_action_enclosure,
    _make_prepared_local_current_operator,
)

from .local_cell_system import prepare_local_cell_galerkin_target

_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_SPACE_DIMENSIONS: int = 3
_TRANSVERSE_AXES: tuple[tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))
_TRACE_FORMULA: str = (
    "T_xi[h,p]=L_n^(-1/2) exp(+2 pi i p xi/L_n), carrier stripped"
)
_NORMAL_FORMULA: str = (
    "N_xi^(s)[h,p]=i s(k_i,n+2 pi p/L_n) T_xi[h,p], using exact "
    "local-ledger carrier evidence"
)
_CURRENT_FORMULA: str = "F_xi^(s)=(T_xi^* N_xi^(s)-N_xi^(s)* T_xi)/(2i)"
_FIXED_ERROR_FORMULA: str = (
    "epsilon_F=up(epsilon_T ||N|| + (||T||+epsilon_T) epsilon_N); "
    "LVT.55a5 evaluated once"
)
_CURRENT_NORMALIZATION: str = (
    "SC.35c C_j=hbar*c^2/(m_e*c^2+e*U0), converted to square "
    "Angstroms per second"
)
_COEFFICIENT_METRICS: str = (
    "state SC.12 box-L2 orthonormal Euclidean coefficients; trace "
    "transverse-plane-L2 orthonormal Euclidean coefficients"
)
_ELIGIBILITY_SCOPE: str = (
    "uniform scoped signed finite current operator at one exact stored plane; "
    "excludes vacuum branches, outgoing extraction, detectors, state error, "
    "and continuum error"
)
_ACTION_TARGET: str = (
    "exact-real action of the stored frozen dyadic implicit Hermitian F matrix"
)
_ACTION_ERROR_SCOPE: str = (
    "rounded-call to frozen-matrix Euclidean error only; excludes uniform "
    "epsilon_F, exact-target transfer, state error, and continuum error"
)
_CURRENT_TARGET: str = (
    "direct exact-target oriented reduced-current scalar for the submitted "
    "stored state, plane, and declared complete transverse-fiber scope"
)
_CURRENT_ERROR_SCOPE: str = (
    "direct exact-target scalar interval replaces any epsilon_F state-norm "
    "transfer and is never added to that route"
)
_NORMALIZATION_FAILURE = (
    GalerkinLocalCurrentOperatorFailure.CURRENT_NORMALIZATION_UNENCLOSED
)


def _checked_maximum_direct_pairs(value: object) -> int:
    """PRIVATE: Validate one independent signed-int64 work budget.

    Parameters
    ----------
    value : object
        Candidate positive Python or NumPy integer.

    Returns
    -------
    budget : int
        Positive signed-int64 work budget.

    Raises
    ------
    TypeError
        If the candidate is boolean or not an integer.
    ValueError
        If the integer is outside positive signed-int64 range.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError("maximum_direct_pairs must be an integer")
    budget = int(value)
    if budget <= 0 or budget > _MAXIMUM_SIGNED_INT64:
        raise ValueError("maximum_direct_pairs must be positive signed int64")
    return budget


def _checked_coordinate(value: object) -> np.float64:
    """PRIVATE: Validate one exact stored finite normal-or-zero float64 plane.

    Parameters
    ----------
    value : object
        Candidate scalar binary64 coordinate.

    Returns
    -------
    coordinate : np.float64
        Exact stored plane coordinate.

    Raises
    ------
    TypeError
        If the input is not scalar binary64 floating data.
    ValueError
        If the coordinate is nonfinite or subnormal.
    """
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(np.float64):
        raise TypeError("terminal_plane_coordinate must be scalar float64")
    coordinate = np.float64(array)
    if not np.isfinite(coordinate) or (
        coordinate != 0.0 and abs(coordinate) < np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "terminal_plane_coordinate must be finite normal-or-zero float64"
        )
    return coordinate


def _checked_scope(
    value: GalerkinLocalTerminalScope | str,
) -> GalerkinLocalTerminalScope:
    """PRIVATE: Validate one explicit current-scope enum.

    Parameters
    ----------
    value : GalerkinLocalTerminalScope | str
        Candidate exact scope enum or enum value.

    Returns
    -------
    scope : GalerkinLocalTerminalScope
        Canonical complete-fiber scope.

    Raises
    ------
    ValueError
        If ``value`` does not name one supported scope.
    """
    scope = GalerkinLocalTerminalScope(value)
    return scope  # noqa: RET504


def _scope_mapping(
    target: GalerkinLocalCellTargetManifest,
    scope: GalerkinLocalTerminalScope,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PRIVATE: Derive canonical fibers, safe rows, and selected-state mask.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local-cell target.
    scope : GalerkinLocalTerminalScope
        Full-state or selected-preterminal complete-fiber scope.

    Returns
    -------
    fibers : np.ndarray
        Canonically ordered transverse integer fibers with shape ``(f, 2)``.
    rows : np.ndarray
        Safe scoped-fiber row for every retained state coefficient.
    selected : np.ndarray
        Exact selected-state membership mask.
    """
    axis = target.acquisition.terminal_axis
    transverse_axes = _TRANSVERSE_AXES[axis]
    state = np.asarray(jax.device_get(target.state_indices), dtype=np.int64)
    state_transverse = state[:, transverse_axes]
    if scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS:
        fibers = np.unique(state_transverse, axis=0)
    else:
        fibers = np.asarray(
            jax.device_get(target.acquisition.transverse_indices),
            dtype=np.int64,
        )
    row_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(fibers)
    }
    rows = np.zeros((state.shape[0],), dtype=np.int64)
    selected = np.zeros((state.shape[0],), dtype=np.bool_)
    for index, transverse in enumerate(state_transverse):
        row = row_lookup.get(tuple(int(value) for value in transverse))
        if row is not None:
            rows[index] = row
            selected[index] = True
    return fibers, rows, selected


def _work_counts(state_size: int, fiber_size: int) -> tuple[int, int]:
    """PRIVATE: Return exact linear action and current operation counts.

    Parameters
    ----------
    state_size : int
        Retained state coefficient count traversed by each pass.
    fiber_size : int
        Scoped fiber count traversed by the scalar reduction.

    Returns
    -------
    action_count : int
        Exact ``2 n`` aggregate-plus-adjoint action count.
    diagnostic_count : int
        Exact ``3 n + f`` action-plus-direct-current count.
    """
    action_count = 2 * state_size
    diagnostic_count = 3 * state_size + fiber_size
    return action_count, diagnostic_count


def _rounded_coefficients(
    target: GalerkinLocalCellTargetManifest,
    coordinate: np.float64,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """PRIVATE: Evaluate the actual frozen coordinate ``T/N`` coefficients.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local-cell target.
    coordinate : np.float64
        Exact stored plane coordinate.
    selected : np.ndarray
        Scoped selected-state mask.

    Returns
    -------
    trace : np.ndarray
        Actual frozen carrier-stripped trace coefficients.
    normal : np.ndarray
        Actual frozen side-oriented normal coefficients.
    """
    axis = target.acquisition.terminal_axis
    side = (
        1.0 if target.acquisition.terminal_side.value == "positive" else -1.0
    )
    indices = target.state_indices[:, axis].astype(jnp.float64)
    length = target.box_lengths[axis]
    phase = jnp.exp(2.0j * jnp.pi * indices * jnp.asarray(coordinate) / length)
    normalization = jax.lax.rsqrt(length)
    trace_all = normalization * phase
    wavevector = side * (
        target.carrier[axis] + 2.0 * jnp.pi * indices / length
    )
    normal_all = 1.0j * wavevector * trace_all
    mask = jnp.asarray(selected)
    trace = np.asarray(
        jax.device_get(jnp.where(mask, trace_all, 0.0 + 0.0j)),
        dtype=np.complex128,
    )
    normal = np.asarray(
        jax.device_get(jnp.where(mask, normal_all, 0.0 + 0.0j)),
        dtype=np.complex128,
    )
    return trace, normal


def _real_interval_add(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    """PRIVATE: Add two exact rational real intervals.

    Parameters
    ----------
    left : tuple[Fraction, Fraction]
        First ordered rational interval.
    right : tuple[Fraction, Fraction]
        Second ordered rational interval.

    Returns
    -------
    lower : Fraction
        Exact lower endpoint of the Minkowski sum.
    upper : Fraction
        Exact upper endpoint of the Minkowski sum.
    """
    result = (left[0] + right[0], left[1] + right[1])
    return result  # noqa: RET504


def _negate_rectangle(value: ComplexRectangle) -> ComplexRectangle:
    """PRIVATE: Negate one exact rational complex rectangle.

    Parameters
    ----------
    value : ComplexRectangle
        Ordered real and imaginary component bounds.

    Returns
    -------
    result : ComplexRectangle
        Negated rectangle with endpoint order retained.
    """
    result = (-value[1], -value[0], -value[3], -value[2])
    return result  # noqa: RET504


def _point_rectangle(value: np.complex128) -> ComplexRectangle:
    """PRIVATE: Convert one finite complex128 point to exact rationals.

    Parameters
    ----------
    value : np.complex128
        Exact stored binary64 complex value.

    Returns
    -------
    rectangle : ComplexRectangle
        Degenerate exact rational rectangle.
    """
    real = Fraction.from_float(float(np.real(value)))
    imag = Fraction.from_float(float(np.imag(value)))
    rectangle = (real, real, imag, imag)
    return rectangle  # noqa: RET504


def _fraction_rectangle(
    rectangles: GalerkinLocalTerminalComplexRectangles,
    index: int,
) -> ComplexRectangle:
    """PRIVATE: Recover one outward binary64 rectangle as exact rationals.

    Parameters
    ----------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Stored componentwise rectangle arrays.
    index : int
        State component index.

    Returns
    -------
    rectangle : ComplexRectangle
        Exact dyadic interpretation of the stored endpoints.
    """
    rectangle = tuple(
        Fraction.from_float(float(np.asarray(values)[index]))
        for values in rectangles
    )
    return rectangle  # type: ignore[return-value]  # noqa: RET504


def _exact_normal_intervals(
    target: GalerkinLocalCellTargetManifest,
) -> list[tuple[Fraction, Fraction]]:
    """PRIVATE: Build exact-ledger side-oriented normal-wavevector intervals.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Local target carrying exact carrier endpoints and box geometry.

    Returns
    -------
    intervals : list[tuple[Fraction, Fraction]]
        Inclusive exact-target oriented normal-wavevector intervals.
    """
    axis = target.acquisition.terminal_axis
    side = 1 if target.acquisition.terminal_side.value == "positive" else -1
    length = Fraction.from_float(float(np.asarray(target.box_lengths[axis])))
    ledger = target.fixed_linear_error_ledger
    carrier = (
        Fraction.from_float(
            float(np.asarray(ledger.exact_carrier_lower_bounds[axis]))
        ),
        Fraction.from_float(
            float(np.asarray(ledger.exact_carrier_upper_bounds[axis]))
        ),
    )
    pi_values = mathematical_pi_interval()
    pi_interval = (
        Fraction.from_float(float(np.asarray(pi_values[0]))),
        Fraction.from_float(float(np.asarray(pi_values[1]))),
    )
    indices = np.asarray(
        jax.device_get(target.state_indices[:, axis]), dtype=np.int64
    )
    intervals: list[tuple[Fraction, Fraction]] = []
    for index in indices:
        reciprocal = Fraction(2 * int(index), 1) / length
        offset = real_interval_product((reciprocal, reciprocal), pi_interval)
        wavevector = _real_interval_add(carrier, offset)
        if side < 0:
            wavevector = (-wavevector[1], -wavevector[0])
        intervals.append(wavevector)
    return intervals


def _exact_coefficient_rectangles(  # noqa: PLR0913
    target: GalerkinLocalCellTargetManifest,
    coordinate: np.float64,
    selected: np.ndarray,
    frozen_trace: np.ndarray,
    frozen_normal: np.ndarray,
) -> tuple[
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalComplexRectangles,
    np.ndarray,
    np.ndarray,
    list[tuple[Fraction, Fraction]],
    list[ComplexRectangle],
    list[ComplexRectangle],
]:
    """PRIVATE: Enclose exact coordinate coefficients and frozen errors.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local target.
    coordinate : np.float64
        Exact stored coordinate.
    selected : np.ndarray
        Scoped selected-state mask.
    frozen_trace : np.ndarray
        Actual frozen trace coefficients.
    frozen_normal : np.ndarray
        Actual frozen normal coefficients.

    Returns
    -------
    trace_rectangles : GalerkinLocalTerminalComplexRectangles
        Outward exact-target trace rectangles.
    normal_rectangles : GalerkinLocalTerminalComplexRectangles
        Outward exact-target normal rectangles.
    trace_errors : np.ndarray
        Frozen-to-exact trace coefficient errors.
    normal_errors : np.ndarray
        Frozen-to-exact normal coefficient errors.
    wavevectors : list[tuple[Fraction, Fraction]]
        Exact-ledger oriented normal-wavevector intervals.
    trace_rational : list[ComplexRectangle]
        Exact rational trace rectangles before float conversion.
    normal_rational : list[ComplexRectangle]
        Exact rational normal rectangles before float conversion.
    """
    axis = target.acquisition.terminal_axis
    length = Fraction.from_float(float(np.asarray(target.box_lengths[axis])))
    xi = Fraction.from_float(float(coordinate))
    normalization = (
        Fraction(1, 1) / sqrt_fraction_upper(length),
        sqrt_fraction_upper(Fraction(1, 1) / length),
    )
    normalization_rectangle: ComplexRectangle = (
        normalization[0],
        normalization[1],
        Fraction(0),
        Fraction(0),
    )
    wavevectors = _exact_normal_intervals(target)
    normal_indices = np.asarray(
        jax.device_get(target.state_indices[:, axis]), dtype=np.int64
    )
    zero: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    trace_rational: list[ComplexRectangle] = []
    normal_rational: list[ComplexRectangle] = []
    for state_index, normal_index in enumerate(normal_indices):
        if not bool(selected[state_index]):
            trace_rational.append(zero)
            normal_rational.append(zero)
            continue
        phase = rational_turn_exponential(
            -Fraction(int(normal_index)) * xi / length
        )
        trace_rectangle = complex_rectangle_multiply(
            normalization_rectangle, phase
        )
        q_lower, q_upper = wavevectors[state_index]
        iq_rectangle: ComplexRectangle = (
            Fraction(0),
            Fraction(0),
            q_lower,
            q_upper,
        )
        normal_rectangle = complex_rectangle_multiply(
            iq_rectangle, trace_rectangle
        )
        trace_rational.append(trace_rectangle)
        normal_rational.append(normal_rectangle)

    def convert(
        values: list[ComplexRectangle],
    ) -> GalerkinLocalTerminalComplexRectangles:
        """Convert rational rectangles to outward binary64 arrays."""
        columns = list(zip(*values, strict=True))
        converted = (
            np.asarray(
                [fraction_lower_float(value) for value in columns[0]],
                dtype=np.float64,
            ),
            np.asarray(
                [fraction_upper_float(value) for value in columns[1]],
                dtype=np.float64,
            ),
            np.asarray(
                [fraction_lower_float(value) for value in columns[2]],
                dtype=np.float64,
            ),
            np.asarray(
                [fraction_upper_float(value) for value in columns[3]],
                dtype=np.float64,
            ),
        )
        return GalerkinLocalTerminalComplexRectangles(
            *(jnp.asarray(value) for value in converted)
        )

    trace_errors = np.asarray(
        [
            fraction_upper_float(coefficient_error_fraction(value, rectangle))
            for value, rectangle in zip(
                frozen_trace, trace_rational, strict=True
            )
        ],
        dtype=np.float64,
    )
    normal_errors = np.asarray(
        [
            fraction_upper_float(coefficient_error_fraction(value, rectangle))
            for value, rectangle in zip(
                frozen_normal, normal_rational, strict=True
            )
        ],
        dtype=np.float64,
    )
    return (
        convert(trace_rational),
        convert(normal_rational),
        trace_errors,
        normal_errors,
        wavevectors,
        trace_rational,
        normal_rational,
    )


def _fiber_root_upper(
    rows: np.ndarray,
    selected: np.ndarray,
    fiber_size: int,
    magnitudes: list[Fraction],
) -> float:
    """PRIVATE: Bound the maximum scoped fiber Euclidean row norm.

    Parameters
    ----------
    rows : np.ndarray
        Safe scoped-fiber row per state coefficient.
    selected : np.ndarray
        Scoped selected-state mask.
    fiber_size : int
        Number of scoped transverse fibers.
    magnitudes : list[Fraction]
        Per-state nonnegative coefficient-magnitude bounds.

    Returns
    -------
    upper : float
        Outward binary64 maximum fiber-row norm bound.
    """
    squared = [Fraction(0) for _ in range(fiber_size)]
    for index, magnitude in enumerate(magnitudes):
        if bool(selected[index]):
            squared[int(rows[index])] += magnitude * magnitude
    upper_fraction = max(sqrt_fraction_upper(value) for value in squared)
    upper = fraction_upper_float(upper_fraction)
    return upper  # noqa: RET504


def _normalized_wavevector_magnitude_upper(
    interval: tuple[Fraction, Fraction],
    length: Fraction,
) -> Fraction:
    """PRIVATE: Bound one normalized exact normal-wavevector magnitude.

    Parameters
    ----------
    interval : tuple[Fraction, Fraction]
        Exact oriented wavevector interval.
    length : Fraction
        Positive exact stored terminal box length.

    Returns
    -------
    upper : Fraction
        Outward rational bound for ``max(abs(q))/sqrt(length)``.
    """
    upper = max(abs(interval[0]), abs(interval[1])) * sqrt_fraction_upper(
        Fraction(1, 1) / length
    )
    return upper  # noqa: RET504


def _operator_reports(
    target: GalerkinLocalCellTargetManifest,
    rows: np.ndarray,
    selected: np.ndarray,
    fiber_size: int,
    wavevectors: list[tuple[Fraction, Fraction]],
    trace_errors: np.ndarray,
    normal_errors: np.ndarray,
) -> tuple[np.float64, ...]:
    """PRIVATE: Build exact norms, fixed errors, and SC.35c evidence.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local target.
    rows : np.ndarray
        Safe scoped-fiber row per state coefficient.
    selected : np.ndarray
        Scoped selected-state mask.
    fiber_size : int
        Scoped transverse-fiber count.
    wavevectors : list[tuple[Fraction, Fraction]]
        Exact-ledger oriented wavevector intervals.
    trace_errors : np.ndarray
        Per-state trace coefficient errors.
    normal_errors : np.ndarray
        Per-state normal coefficient errors.

    Returns
    -------
    reports : tuple[np.float64, ...]
        Nine exact-norm, fixed-error, and SC.35c reports.
    """
    axis = target.acquisition.terminal_axis
    length = Fraction.from_float(float(np.asarray(target.box_lengths[axis])))
    counts = [0 for _ in range(fiber_size)]
    normal_magnitudes: list[Fraction] = []
    for index, interval in enumerate(wavevectors):
        if bool(selected[index]):
            counts[int(rows[index])] += 1
        normal_magnitudes.append(
            _normalized_wavevector_magnitude_upper(interval, length)
        )
    exact_trace = fraction_upper_float(
        sqrt_fraction_upper(Fraction(max(counts), 1) / length)
    )
    exact_normal = _fiber_root_upper(
        rows, selected, fiber_size, normal_magnitudes
    )
    trace_error_fractions = [
        Fraction.from_float(float(value)) for value in trace_errors
    ]
    normal_error_fractions = [
        Fraction.from_float(float(value)) for value in normal_errors
    ]
    trace_error = _fiber_root_upper(
        rows, selected, fiber_size, trace_error_fractions
    )
    normal_error = _fiber_root_upper(
        rows, selected, fiber_size, normal_error_fractions
    )
    trace_error_array = jnp.asarray(trace_error, dtype=jnp.float64)
    exact_normal_array = jnp.asarray(exact_normal, dtype=jnp.float64)
    exact_trace_array = jnp.asarray(exact_trace, dtype=jnp.float64)
    normal_error_array = jnp.asarray(normal_error, dtype=jnp.float64)
    current_error = upward_add(
        upward_multiply(trace_error_array, exact_normal_array),
        upward_multiply(
            upward_add(exact_trace_array, trace_error_array),
            normal_error_array,
        ),
    )
    one_thousand = Fraction(1000)
    angstrom_squared = Fraction(10**20)
    mass = Fraction.from_float(float(np.asarray(M_E)))
    charge = Fraction.from_float(float(np.asarray(E_CHARGE)))
    speed = Fraction.from_float(float(np.asarray(C_LIGHT)))
    hbar = Fraction.from_float(float(np.asarray(HBAR)))
    voltage = Fraction.from_float(
        float(np.asarray(target.accelerating_voltage_kv))
    )
    exact_scale = (
        hbar
        * speed
        * speed
        * angstrom_squared
        / (mass * speed * speed + charge * voltage * one_thousand)
    )
    scale_lower = fraction_lower_float(exact_scale)
    scale_upper = fraction_upper_float(exact_scale)
    stored_scale = np.float64(
        float(np.asarray(HBAR))
        * float(np.asarray(C_LIGHT)) ** 2
        * 1.0e20
        / (
            float(np.asarray(M_E)) * float(np.asarray(C_LIGHT)) ** 2
            + float(np.asarray(E_CHARGE))
            * float(np.asarray(target.accelerating_voltage_kv))
            * 1000.0
        )
    )
    scale_point = Fraction.from_float(float(stored_scale))
    scale_error = fraction_upper_float(
        max(abs(scale_point - exact_scale), abs(scale_point - exact_scale))
    )
    reports = (
        np.float64(exact_trace),
        np.float64(exact_normal),
        np.float64(trace_error),
        np.float64(normal_error),
        np.float64(current_error),
        stored_scale,
        np.float64(scale_lower),
        np.float64(scale_upper),
        np.float64(scale_error),
    )
    return reports  # noqa: RET504


def _normal_or_zero(values: object) -> bool:
    """PRIVATE: Check finite normal-range array components or exact zeros.

    Parameters
    ----------
    values : object
        Candidate NumPy/JAX scalar or array.

    Returns
    -------
    valid : bool
        Whether every component is finite and non-subnormal.
    """
    array = jnp.asarray(values)
    valid = bool(jnp.all(jnp.isfinite(array))) and not bool(
        has_subnormal_components(array)
    )
    return valid  # noqa: RET504


def _sentinel_rectangles(size: int) -> GalerkinLocalTerminalComplexRectangles:
    """PRIVATE: Return fail-closed unbounded rectangles.

    Parameters
    ----------
    size : int
        Retained state length.

    Returns
    -------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Componentwise ``[-inf, +inf]`` sentinels.
    """
    lower = jnp.full((size,), -jnp.inf, dtype=jnp.float64)
    upper = jnp.full((size,), jnp.inf, dtype=jnp.float64)
    rectangles = GalerkinLocalTerminalComplexRectangles(
        lower, upper, lower, upper
    )
    return rectangles  # noqa: RET504


def _certify_prepared_operator(  # noqa: PLR0912,PLR0915
    target: GalerkinLocalCellTargetManifest,
    coordinate: np.float64,
    scope: GalerkinLocalTerminalScope,
    maximum_direct_pairs: int,
) -> GalerkinLocalCurrentOperatorCertificate:
    """PRIVATE: Build one operator record from a replayed local target.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local-cell target.
    coordinate : np.float64
        Exact stored terminal plane.
    scope : GalerkinLocalTerminalScope
        Explicit complete transverse-fiber scope.
    maximum_direct_pairs : int
        Independent signed-int64 linear-work budget.

    Returns
    -------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Provisional uniform operator certificate or typed noncertificate.
    """
    fibers, rows, selected = _scope_mapping(target, scope)
    state_size = target.state_indices.shape[0]
    fiber_size = fibers.shape[0]
    action_count, diagnostic_count = _work_counts(state_size, fiber_size)
    count_overflow = (
        action_count > _MAXIMUM_SIGNED_INT64
        or diagnostic_count > _MAXIMUM_SIGNED_INT64
    )
    stored_action_count = 0 if count_overflow else action_count
    stored_diagnostic_count = 0 if count_overflow else diagnostic_count
    host_supported = host_binary64_supported()
    normal_supported = bool(all_normal_arithmetic_supported())
    selected_complete = (
        scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
        or bool(target.support_eligibility.terminal_fiber_complete)
    )
    target_eligible = bool(target.fixed_linear_error_ledger.finite_certificate)
    if not target_eligible:
        failure = (
            GalerkinLocalCurrentOperatorFailure.TARGET_FIXED_LINEAR_INELIGIBLE
        )
    elif not selected_complete:
        failure = GalerkinLocalCurrentOperatorFailure.TERMINAL_FIBER_INCOMPLETE
    elif count_overflow:
        failure = (
            GalerkinLocalCurrentOperatorFailure.DIRECT_WORK_COUNT_OVERFLOW
        )
    elif not host_supported or not normal_supported:
        failure = (
            GalerkinLocalCurrentOperatorFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
    elif action_count > maximum_direct_pairs:
        failure = (
            GalerkinLocalCurrentOperatorFailure.DIRECT_WORK_BUDGET_EXCEEDED
        )
    else:
        failure = GalerkinLocalCurrentOperatorFailure.NONE

    frozen_trace, frozen_normal = _rounded_coefficients(
        target, coordinate, selected
    )
    trace_rectangles = _sentinel_rectangles(state_size)
    normal_rectangles = _sentinel_rectangles(state_size)
    trace_errors = np.full((state_size,), np.inf, dtype=np.float64)
    normal_errors = np.full((state_size,), np.inf, dtype=np.float64)
    reports = tuple(np.float64(np.inf) for _ in range(9))
    if failure is GalerkinLocalCurrentOperatorFailure.NONE:
        try:
            (
                trace_rectangles,
                normal_rectangles,
                trace_errors,
                normal_errors,
                wavevectors,
                _,
                _,
            ) = _exact_coefficient_rectangles(
                target,
                coordinate,
                selected,
                frozen_trace,
                frozen_normal,
            )
            reports = _operator_reports(
                target,
                rows,
                selected,
                fiber_size,
                wavevectors,
                trace_errors,
                normal_errors,
            )
        except (RootEnclosureError, ValueError, ZeroDivisionError):
            failure = (
                GalerkinLocalCurrentOperatorFailure.ROOT_ENCLOSURE_FAILURE
            )
    if failure is GalerkinLocalCurrentOperatorFailure.NONE:
        evidence_arrays = (
            frozen_trace,
            frozen_normal,
            *trace_rectangles,
            *normal_rectangles,
            trace_errors,
            normal_errors,
            *reports,
        )
        if not all(_normal_or_zero(value) for value in evidence_arrays):
            failure = (
                GalerkinLocalCurrentOperatorFailure.ARITHMETIC_RANGE_FAILURE
            )
        elif reports[5] <= 0.0 or reports[6] <= 0.0 or reports[6] > reports[7]:
            failure = _NORMALIZATION_FAILURE
    eligible = failure is GalerkinLocalCurrentOperatorFailure.NONE
    identity_digest = sha256(
        {
            "domain": "ptyrodactyl.local_terminal.operator.identity.v1",
            "target_digest": target.target_digest,
            "coordinate": stored_value_payload(np.asarray(coordinate)),
            "scope": scope.value,
            "terminal_axis": target.acquisition.terminal_axis,
            "terminal_side": target.acquisition.terminal_side.value,
            "scope_fibers": stored_value_payload(fibers),
            "rows": stored_value_payload(rows),
            "selected": stored_value_payload(selected),
            "frozen_trace": stored_value_payload(frozen_trace),
            "frozen_normal": stored_value_payload(frozen_normal),
            "trace_formula": _TRACE_FORMULA,
            "normal_formula": _NORMAL_FORMULA,
            "current_formula": _CURRENT_FORMULA,
        }
    )
    evidence_digest = sha256(
        {
            "domain": "ptyrodactyl.local_terminal.operator.evidence.v1",
            "identity_digest": identity_digest,
            "parent_target_evidence_digest": target.manifest_evidence_digest,
            "trace_rectangles": stored_value_payload(trace_rectangles),
            "normal_rectangles": stored_value_payload(normal_rectangles),
            "trace_errors": stored_value_payload(trace_errors),
            "normal_errors": stored_value_payload(normal_errors),
            "reports": stored_value_payload(reports),
            "action_work_count_exact": str(action_count),
            "diagnostic_work_count_exact": str(diagnostic_count),
            "maximum_direct_pairs": maximum_direct_pairs,
            "host_supported": host_supported,
            "normal_supported": normal_supported,
            "failure": failure.value,
            "fixed_error_formula": _FIXED_ERROR_FORMULA,
            "current_normalization": _CURRENT_NORMALIZATION,
        }
    )
    stopped = jax.tree.map(
        jax.lax.stop_gradient,
        (
            jnp.asarray(coordinate),
            jnp.asarray(fibers),
            jnp.asarray(rows),
            jnp.asarray(selected),
            jnp.asarray(frozen_trace),
            jnp.asarray(frozen_normal),
            trace_rectangles,
            normal_rectangles,
            jnp.asarray(trace_errors),
            jnp.asarray(normal_errors),
            *(jnp.asarray(value) for value in reports),
            jnp.asarray(stored_action_count, dtype=jnp.int64),
            jnp.asarray(stored_diagnostic_count, dtype=jnp.int64),
            jnp.asarray(maximum_direct_pairs, dtype=jnp.int64),
            jnp.asarray(host_supported),
            jnp.asarray(normal_supported),
            jnp.asarray(eligible),
        ),
    )
    certificate = _make_local_current_operator_certificate(
        target,
        stopped[0],
        (stopped[1], stopped[2], stopped[3]),
        (stopped[4], stopped[5]),
        (stopped[6], stopped[7]),
        (stopped[8], stopped[9]),
        tuple(stopped[10:19]),
        (stopped[19], stopped[20], stopped[21]),
        (stopped[22], stopped[23], stopped[24]),
        terminal_axis=target.acquisition.terminal_axis,
        terminal_side=target.acquisition.terminal_side,
        current_scope=scope,
        failure=failure,
        action_work_count_exact=str(action_count),
        current_diagnostic_work_count_exact=str(diagnostic_count),
        declarations=(
            _TRACE_FORMULA,
            _NORMAL_FORMULA,
            _CURRENT_FORMULA,
            _FIXED_ERROR_FORMULA,
            _CURRENT_NORMALIZATION,
            _COEFFICIENT_METRICS,
            _ELIGIBILITY_SCOPE,
        ),
        digests=(
            target.target_digest,
            target.manifest_evidence_digest,
            identity_digest,
            evidence_digest,
        ),
    )
    return certificate  # noqa: RET504


def _checked_prepared(
    prepared: GalerkinPreparedLocalCurrentOperator,
) -> GalerkinLocalCurrentOperatorCertificate:
    """PRIVATE: Extract the explicit prepared-only frozen-action capability.

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Caller-held marker returned by host operator preparation.

    Returns
    -------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Nested operator record used by frozen JIT actions.

    Raises
    ------
    TypeError
        If raw target or certificate storage is supplied instead.

    Notes
    -----
    Python construction of the wrapper is not cryptographic authentication.
    Scientific host entry points ignore wrappers and replay raw certificates;
    transform callers must close only over the host preparer's return value.
    """
    if not isinstance(prepared, GalerkinPreparedLocalCurrentOperator):
        raise TypeError(
            "frozen local-terminal actions require the prepared-only "
            "operator wrapper"
        )
    certificate = prepared.certificate
    return certificate  # noqa: RET504


def _checked_state(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one transform-time retained-state vector.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Prepared wrapper's nested operator certificate.
    field : Complex[Array, "..."]
        Candidate complex retained-state coefficients.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite normal-range complex128 state vector.

    Raises
    ------
    ValueError
        If rank or retained-state length is invalid.
    equinox.EquinoxRuntimeError
        If the operator is ineligible or a component is nonfinite/subnormal.
    """
    values = jnp.asarray(field, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("field must be one-dimensional")
    if values.shape != certificate.trace_frozen_coefficients.shape:
        raise ValueError("field must match the retained state")
    checked = eqx.error_if(
        values,
        (~certificate.current_operator_eligible)
        | jnp.any(~jnp.isfinite(values))
        | has_subnormal_components(values),
        "prepared operator and field must be eligible finite normal-range "
        "data",
    )
    return checked  # noqa: RET504


def _checked_terminal_vector(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    terminal_field: Complex[Array, "..."],
) -> Complex128[Array, " f"]:
    """PRIVATE: Validate one transform-time scoped terminal vector.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Prepared wrapper's nested operator certificate.
    terminal_field : Complex[Array, "..."]
        Candidate scoped transverse-fiber coefficients.

    Returns
    -------
    checked : Complex128[Array, " f"]
        Finite normal-range complex128 terminal vector.

    Raises
    ------
    ValueError
        If rank or scoped-fiber length is invalid.
    equinox.EquinoxRuntimeError
        If the operator is ineligible or a component is nonfinite/subnormal.
    """
    values = jnp.asarray(terminal_field, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("terminal_field must be one-dimensional")
    if values.shape[0] != certificate.scope_transverse_indices.shape[0]:
        raise ValueError("terminal_field must match scoped fibers")
    checked = eqx.error_if(
        values,
        (~certificate.current_operator_eligible)
        | jnp.any(~jnp.isfinite(values))
        | has_subnormal_components(values),
        "prepared operator and terminal field must be eligible finite "
        "normal-range data",
    )
    return checked  # noqa: RET504


def _raw_trace(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: Complex128[Array, " n"],
) -> Complex128[Array, " f"]:
    """PRIVATE: Apply the actual frozen trace without output rejection.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Frozen coordinate-operator evidence.
    field : Complex128[Array, " n"]
        Retained-state coefficients.

    Returns
    -------
    trace : Complex128[Array, " f"]
        Scoped carrier-stripped trace coefficients.
    """
    contributions = jnp.where(
        certificate.selected_state_mask,
        certificate.trace_frozen_coefficients * field,
        0.0 + 0.0j,
    )
    fiber_size = certificate.scope_transverse_indices.shape[0]
    trace = (
        jnp.zeros((fiber_size,), dtype=jnp.complex128)
        .at[certificate.state_to_fiber_rows]
        .add(contributions)
    )
    return trace  # noqa: RET504


def _raw_normal(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: Complex128[Array, " n"],
) -> Complex128[Array, " f"]:
    """PRIVATE: Apply the actual frozen oriented normal trace.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Frozen coordinate-operator evidence.
    field : Complex128[Array, " n"]
        Retained-state coefficients.

    Returns
    -------
    normal : Complex128[Array, " f"]
        Scoped side-oriented normal-derivative coefficients.
    """
    contributions = jnp.where(
        certificate.selected_state_mask,
        certificate.normal_frozen_coefficients * field,
        0.0 + 0.0j,
    )
    fiber_size = certificate.scope_transverse_indices.shape[0]
    normal = (
        jnp.zeros((fiber_size,), dtype=jnp.complex128)
        .at[certificate.state_to_fiber_rows]
        .add(contributions)
    )
    return normal  # noqa: RET504


def _raw_trace_adjoint(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    terminal_field: Complex128[Array, " f"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the literal adjoint of the frozen trace matrix.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Frozen coordinate-operator evidence.
    terminal_field : Complex128[Array, " f"]
        Scoped transverse-fiber coefficients.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        Retained-state conjugate-transpose action.
    """
    gathered = terminal_field[certificate.state_to_fiber_rows]
    adjoint = jnp.where(
        certificate.selected_state_mask,
        jnp.conj(certificate.trace_frozen_coefficients) * gathered,
        0.0 + 0.0j,
    )
    return adjoint  # noqa: RET504


def _raw_normal_adjoint(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    terminal_field: Complex128[Array, " f"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the literal adjoint of the frozen normal matrix.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Frozen coordinate-operator evidence.
    terminal_field : Complex128[Array, " f"]
        Scoped transverse-fiber coefficients.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        Retained-state conjugate-transpose action.
    """
    gathered = terminal_field[certificate.state_to_fiber_rows]
    adjoint = jnp.where(
        certificate.selected_state_mask,
        jnp.conj(certificate.normal_frozen_coefficients) * gathered,
        0.0 + 0.0j,
    )
    return adjoint  # noqa: RET504


def _raw_current_action(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: Complex128[Array, " n"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the implicit actual frozen Hermitian current matrix.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Frozen coordinate-operator evidence.
    field : Complex128[Array, " n"]
        Retained-state coefficients.

    Returns
    -------
    action : Complex128[Array, " n"]
        ``(T* N-N* T) field/(2i)`` using actual frozen adjoints.
    """
    trace = _raw_trace(certificate, field)
    normal = _raw_normal(certificate, field)
    action = (
        _raw_trace_adjoint(certificate, normal)
        - _raw_normal_adjoint(certificate, trace)
    ) / (2.0j)
    return action  # noqa: RET504


def _checked_output(values: Array, name: str) -> Array:
    """PRIVATE: Reject nonfinite or subnormal frozen-action outputs.

    Parameters
    ----------
    values : Array
        Candidate real or complex transform output.
    name : str
        Diagnostic name for runtime rejection.

    Returns
    -------
    checked : Array
        Finite normal-range output.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If one nonzero component is nonfinite or subnormal.
    """
    checked = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | has_subnormal_components(values),
        f"{name} left finite normal binary64 range",
    )
    return checked  # noqa: RET504


def _host_state(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: object,
) -> np.ndarray:
    """PRIVATE: Convert one host enclosure input without hiding range failure.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Parent operator fixing the retained-state length.
    field : object
        Candidate complex floating vector.

    Returns
    -------
    values : np.ndarray
        Complex128 host state, possibly nonfinite/subnormal for typed failure.

    Raises
    ------
    TypeError
        If the input is not complex floating data.
    ValueError
        If rank or retained-state length is invalid.
    """
    array = np.asarray(field)
    if array.dtype.kind != "c":
        raise TypeError("field must use complex floating data")
    if array.ndim != 1:
        raise ValueError("field must be one-dimensional")
    if array.shape != certificate.trace_frozen_coefficients.shape:
        raise ValueError("field must match the retained state")
    values = np.asarray(array, dtype=np.complex128)
    return values  # noqa: RET504


def _convert_rational_rectangles(
    values: list[ComplexRectangle],
) -> GalerkinLocalTerminalComplexRectangles:
    """PRIVATE: Convert rational complex rectangles outward to binary64.

    Parameters
    ----------
    values : list[ComplexRectangle]
        Ordered rational component rectangles.

    Returns
    -------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Outward finite-normal-or-zero binary64 endpoints when representable.
    """
    columns = list(zip(*values, strict=True))
    rectangles = GalerkinLocalTerminalComplexRectangles(
        jnp.asarray([fraction_lower_float(value) for value in columns[0]]),
        jnp.asarray([fraction_upper_float(value) for value in columns[1]]),
        jnp.asarray([fraction_lower_float(value) for value in columns[2]]),
        jnp.asarray([fraction_upper_float(value) for value in columns[3]]),
    )
    return rectangles  # noqa: RET504


def _exact_frozen_action_rectangles(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: np.ndarray,
) -> tuple[GalerkinLocalTerminalComplexRectangles, list[ComplexRectangle]]:
    """PRIVATE: Evaluate the frozen implicit action in exact dyadic arithmetic.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Canonical frozen ``T/N`` coefficient evidence.
    field : np.ndarray
        Exact stored submitted complex128 state.

    Returns
    -------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Outward exact-real frozen-action rectangles.
    rational : list[ComplexRectangle]
        Exact rational action rectangles before float conversion.
    """
    rows = np.asarray(certificate.state_to_fiber_rows)
    selected = np.asarray(certificate.selected_state_mask)
    trace_coefficients = np.asarray(certificate.trace_frozen_coefficients)
    normal_coefficients = np.asarray(certificate.normal_frozen_coefficients)
    fiber_size = certificate.scope_transverse_indices.shape[0]
    trace_terms: list[list[ComplexRectangle]] = [[] for _ in range(fiber_size)]
    normal_terms: list[list[ComplexRectangle]] = [
        [] for _ in range(fiber_size)
    ]
    zero = _point_rectangle(np.complex128(0.0 + 0.0j))
    for index, value in enumerate(field):
        if not bool(selected[index]):
            continue
        row = int(rows[index])
        point = _point_rectangle(np.complex128(value))
        trace_terms[row].append(
            complex_rectangle_multiply(
                _point_rectangle(np.complex128(trace_coefficients[index])),
                point,
            )
        )
        normal_terms[row].append(
            complex_rectangle_multiply(
                _point_rectangle(np.complex128(normal_coefficients[index])),
                point,
            )
        )
    trace_sums = [
        pairwise_rectangle_sum(terms) if terms else zero
        for terms in trace_terms
    ]
    normal_sums = [
        pairwise_rectangle_sum(terms) if terms else zero
        for terms in normal_terms
    ]
    negative_half_i: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(-1, 2),
        Fraction(-1, 2),
    )
    rational: list[ComplexRectangle] = []
    for index in range(field.shape[0]):
        if not bool(selected[index]):
            rational.append(zero)
            continue
        row = int(rows[index])
        trace_normal = complex_rectangle_multiply(
            conjugate_rectangle(
                _point_rectangle(np.complex128(trace_coefficients[index]))
            ),
            normal_sums[row],
        )
        normal_trace = complex_rectangle_multiply(
            conjugate_rectangle(
                _point_rectangle(np.complex128(normal_coefficients[index]))
            ),
            trace_sums[row],
        )
        difference = pairwise_rectangle_sum(
            (trace_normal, _negate_rectangle(normal_trace))
        )
        rational.append(
            complex_rectangle_multiply(difference, negative_half_i)
        )
    return _convert_rational_rectangles(rational), rational


def _action_error_evidence(
    production: np.ndarray,
    rational: list[ComplexRectangle],
) -> tuple[np.ndarray, np.float64]:
    """PRIVATE: Bound rounded-to-frozen action error without uniform transfer.

    Parameters
    ----------
    production : np.ndarray
        Rounded implicit frozen action.
    rational : list[ComplexRectangle]
        Exact-real frozen-action rectangles.

    Returns
    -------
    component_errors : np.ndarray
        Outward per-component complex error bounds.
    action_error : np.float64
        Outward Euclidean vector error bound.
    """
    exact_errors = [
        coefficient_error_fraction(np.complex128(value), rectangle)
        for value, rectangle in zip(production, rational, strict=True)
    ]
    component_errors = np.asarray(
        [fraction_upper_float(value) for value in exact_errors],
        dtype=np.float64,
    )
    squared = sum((value * value for value in exact_errors), Fraction(0))
    action_error = np.float64(
        fraction_upper_float(sqrt_fraction_upper(squared))
    )
    return component_errors, action_error


def _enclose_action_prepared(  # noqa: PLR0912
    prepared: GalerkinPreparedLocalCurrentOperator,
    field: object,
    maximum_direct_pairs: int,
) -> GalerkinLocalTerminalCurrentActionEnclosure:
    """PRIVATE: Enclose one action after exactly one operator replay.

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Host-replayed operator wrapper.
    field : object
        Candidate submitted complex state.
    maximum_direct_pairs : int
        Independently supplied signed-int64 work policy.

    Returns
    -------
    enclosure : GalerkinLocalTerminalCurrentActionEnclosure
        Frozen-action evidence or typed noncertificate.
    """
    certificate = prepared.certificate
    values = _host_state(certificate, field)
    state_size = values.shape[0]
    action_count, _ = _work_counts(
        state_size, certificate.scope_transverse_indices.shape[0]
    )
    count_overflow = action_count > _MAXIMUM_SIGNED_INT64
    host_supported = host_binary64_supported()
    field_range = _normal_or_zero(values)
    if not bool(certificate.current_operator_eligible):
        failure = GalerkinLocalTerminalActionFailure.OPERATOR_NONCERTIFICATE
    elif count_overflow:
        failure = GalerkinLocalTerminalActionFailure.DIRECT_WORK_COUNT_OVERFLOW
    elif not host_supported:
        failure = (
            GalerkinLocalTerminalActionFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
    elif action_count > maximum_direct_pairs:
        failure = (
            GalerkinLocalTerminalActionFailure.DIRECT_WORK_BUDGET_EXCEEDED
        )
    elif not field_range:
        failure = GalerkinLocalTerminalActionFailure.ARITHMETIC_RANGE_FAILURE
    else:
        failure = GalerkinLocalTerminalActionFailure.NONE
    production = np.zeros((state_size,), dtype=np.complex128)
    rectangles = _sentinel_rectangles(state_size)
    component_errors = np.full((state_size,), np.inf, dtype=np.float64)
    action_error = np.float64(np.inf)
    if failure is GalerkinLocalTerminalActionFailure.NONE:
        production = np.asarray(
            jax.device_get(
                _raw_current_action(
                    certificate, jnp.asarray(values, dtype=jnp.complex128)
                )
            ),
            dtype=np.complex128,
        )
        try:
            rectangles, rational = _exact_frozen_action_rectangles(
                certificate, values
            )
            component_errors, action_error = _action_error_evidence(
                production, rational
            )
        except (RootEnclosureError, ValueError, ZeroDivisionError):
            failure = (
                GalerkinLocalTerminalActionFailure.ARITHMETIC_RANGE_FAILURE
            )
    if failure is GalerkinLocalTerminalActionFailure.NONE and not all(
        _normal_or_zero(value)
        for value in (
            production,
            *rectangles,
            component_errors,
            action_error,
        )
    ):
        failure = GalerkinLocalTerminalActionFailure.ARITHMETIC_RANGE_FAILURE
    eligible = failure is GalerkinLocalTerminalActionFailure.NONE
    state_digest = sha256(
        {
            "domain": "ptyrodactyl.local_terminal.action.identity.v1",
            "operator_identity_digest": certificate.operator_identity_digest,
            "submitted_field": stored_value_payload(values),
        }
    )
    evidence_digest = sha256(
        {
            "domain": "ptyrodactyl.local_terminal.action.evidence.v1",
            "operator_evidence_digest": certificate.operator_evidence_digest,
            "state_identity_digest": state_digest,
            "production_action": stored_value_payload(production),
            "rectangles": stored_value_payload(rectangles),
            "component_errors": stored_value_payload(component_errors),
            "action_error": stored_value_payload(action_error),
            "direct_work_count_exact": str(action_count),
            "maximum_direct_pairs": maximum_direct_pairs,
            "host_supported": host_supported,
            "failure": failure.value,
            "error_scope": _ACTION_ERROR_SCOPE,
        }
    )
    stored_count = 0 if count_overflow else action_count
    enclosure = _make_local_terminal_current_action_enclosure(
        certificate,
        jax.lax.stop_gradient(jnp.asarray(values)),
        jax.lax.stop_gradient(jnp.asarray(production)),
        jax.tree.map(jax.lax.stop_gradient, rectangles),
        jax.lax.stop_gradient(jnp.asarray(component_errors)),
        jax.lax.stop_gradient(jnp.asarray(action_error)),
        (
            jnp.asarray(stored_count, dtype=jnp.int64),
            jnp.asarray(maximum_direct_pairs, dtype=jnp.int64),
        ),
        (jnp.asarray(host_supported), jnp.asarray(eligible)),
        failure=failure,
        direct_work_count_exact=str(action_count),
        exact_action_target=_ACTION_TARGET,
        error_scope=_ACTION_ERROR_SCOPE,
        state_identity_digest=state_digest,
        action_evidence_digest=evidence_digest,
    )
    return enclosure  # noqa: RET504


def _exact_target_current_interval(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: np.ndarray,
) -> tuple[Fraction, Fraction]:
    """PRIVATE: Directly enclose one exact-target scoped current scalar.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Canonical exact-target ``T/N`` coefficient rectangles.
    field : np.ndarray
        Exact stored submitted complex128 state.

    Returns
    -------
    lower : Fraction
        Exact rational lower endpoint of ``Im(<T u,N u>)``.
    upper : Fraction
        Exact rational upper endpoint of ``Im(<T u,N u>)``.
    """
    rows = np.asarray(certificate.state_to_fiber_rows)
    selected = np.asarray(certificate.selected_state_mask)
    fiber_size = certificate.scope_transverse_indices.shape[0]
    trace_terms: list[list[ComplexRectangle]] = [[] for _ in range(fiber_size)]
    normal_terms: list[list[ComplexRectangle]] = [
        [] for _ in range(fiber_size)
    ]
    zero = _point_rectangle(np.complex128(0.0 + 0.0j))
    for index, value in enumerate(field):
        if not bool(selected[index]):
            continue
        row = int(rows[index])
        point = _point_rectangle(np.complex128(value))
        trace_terms[row].append(
            complex_rectangle_multiply(
                _fraction_rectangle(
                    certificate.exact_trace_coefficient_rectangles, index
                ),
                point,
            )
        )
        normal_terms[row].append(
            complex_rectangle_multiply(
                _fraction_rectangle(
                    certificate.exact_normal_coefficient_rectangles, index
                ),
                point,
            )
        )
    trace_sums = [
        pairwise_rectangle_sum(terms) if terms else zero
        for terms in trace_terms
    ]
    normal_sums = [
        pairwise_rectangle_sum(terms) if terms else zero
        for terms in normal_terms
    ]
    products = [
        complex_rectangle_multiply(conjugate_rectangle(trace), normal)
        for trace, normal in zip(trace_sums, normal_sums, strict=True)
    ]
    lower = sum((value[2] for value in products), Fraction(0))
    upper = sum((value[3] for value in products), Fraction(0))
    return lower, upper


def _current_error_upper(
    rounded: np.float64,
    interval: tuple[Fraction, Fraction],
) -> np.float64:
    """PRIVATE: Bound a rounded current point against an exact interval.

    Parameters
    ----------
    rounded : np.float64
        Rounded frozen-current quadratic scalar.
    interval : tuple[Fraction, Fraction]
        Direct exact-target current interval.

    Returns
    -------
    upper : np.float64
        Outward maximum endpoint discrepancy.
    """
    point = Fraction.from_float(float(rounded))
    difference = max(abs(point - interval[0]), abs(point - interval[1]))
    upper = np.float64(fraction_upper_float(difference))
    return upper  # noqa: RET504


def _enclose_current_prepared(  # noqa: PLR0912
    prepared: GalerkinPreparedLocalCurrentOperator,
    field: object,
    maximum_direct_pairs: int,
) -> GalerkinLocalCoordinateCauchyCurrent:
    """PRIVATE: Enclose one exact-target current after one operator replay.

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Host-replayed operator wrapper.
    field : object
        Candidate submitted complex state.
    maximum_direct_pairs : int
        Independently supplied signed-int64 work policy.

    Returns
    -------
    diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Direct exact-target scoped current evidence or typed noncertificate.
    """
    certificate = prepared.certificate
    values = _host_state(certificate, field)
    action = _enclose_action_prepared(prepared, values, maximum_direct_pairs)
    state_size = values.shape[0]
    fiber_size = certificate.scope_transverse_indices.shape[0]
    _, diagnostic_count = _work_counts(state_size, fiber_size)
    count_overflow = diagnostic_count > _MAXIMUM_SIGNED_INT64
    host_supported = host_binary64_supported()
    if not bool(certificate.current_operator_eligible):
        failure = GalerkinLocalTerminalCurrentFailure.OPERATOR_NONCERTIFICATE
    elif not bool(action.current_action_eligible):
        failure = GalerkinLocalTerminalCurrentFailure.ACTION_NONCERTIFICATE
    elif count_overflow:
        failure = (
            GalerkinLocalTerminalCurrentFailure.DIRECT_WORK_COUNT_OVERFLOW
        )
    elif not host_supported:
        failure = (
            GalerkinLocalTerminalCurrentFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
    elif diagnostic_count > maximum_direct_pairs:
        failure = (
            GalerkinLocalTerminalCurrentFailure.DIRECT_WORK_BUDGET_EXCEEDED
        )
    else:
        failure = GalerkinLocalTerminalCurrentFailure.NONE
    trace = np.zeros((fiber_size,), dtype=np.complex128)
    normal = np.zeros((fiber_size,), dtype=np.complex128)
    rounded = np.float64(np.inf)
    lower = np.float64(-np.inf)
    upper = np.float64(np.inf)
    error = np.float64(np.inf)
    if failure is GalerkinLocalTerminalCurrentFailure.NONE:
        jax_field = jnp.asarray(values, dtype=jnp.complex128)
        trace = np.asarray(
            jax.device_get(_raw_trace(certificate, jax_field)),
            dtype=np.complex128,
        )
        normal = np.asarray(
            jax.device_get(_raw_normal(certificate, jax_field)),
            dtype=np.complex128,
        )
        rounded = np.float64(
            np.real(np.vdot(values, np.asarray(action.production_action)))
        )
        try:
            exact_interval = _exact_target_current_interval(
                certificate, values
            )
            lower = np.float64(fraction_lower_float(exact_interval[0]))
            upper = np.float64(fraction_upper_float(exact_interval[1]))
            error = _current_error_upper(rounded, exact_interval)
        except (RootEnclosureError, ValueError, ZeroDivisionError):
            failure = (
                GalerkinLocalTerminalCurrentFailure.ARITHMETIC_RANGE_FAILURE
            )
    if failure is GalerkinLocalTerminalCurrentFailure.NONE and not all(
        _normal_or_zero(value)
        for value in (trace, normal, rounded, lower, upper, error)
    ):
        failure = GalerkinLocalTerminalCurrentFailure.ARITHMETIC_RANGE_FAILURE
    eligible = failure is GalerkinLocalTerminalCurrentFailure.NONE
    digest = sha256(
        {
            "domain": "ptyrodactyl.local_terminal.current.evidence.v1",
            "action_evidence_digest": action.action_evidence_digest,
            "trace": stored_value_payload(trace),
            "normal": stored_value_payload(normal),
            "rounded_current": stored_value_payload(rounded),
            "exact_current_interval": stored_value_payload((lower, upper)),
            "current_error": stored_value_payload(error),
            "direct_work_count_exact": str(diagnostic_count),
            "maximum_direct_pairs": maximum_direct_pairs,
            "host_supported": host_supported,
            "failure": failure.value,
            "current_target": _CURRENT_TARGET,
            "error_scope": _CURRENT_ERROR_SCOPE,
        }
    )
    stored_count = 0 if count_overflow else diagnostic_count
    diagnostic = _make_local_coordinate_cauchy_current(
        action,
        jax.lax.stop_gradient(jnp.asarray(trace)),
        jax.lax.stop_gradient(jnp.asarray(normal)),
        (
            jnp.asarray(rounded),
            jnp.asarray(lower),
            jnp.asarray(upper),
            jnp.asarray(error),
        ),
        (
            jnp.asarray(stored_count, dtype=jnp.int64),
            jnp.asarray(maximum_direct_pairs, dtype=jnp.int64),
        ),
        (jnp.asarray(host_supported), jnp.asarray(eligible)),
        failure=failure,
        direct_work_count_exact=str(diagnostic_count),
        current_target=_CURRENT_TARGET,
        error_scope=_CURRENT_ERROR_SCOPE,
        diagnostic_evidence_digest=digest,
    )
    return diagnostic  # noqa: RET504


def certify_local_terminal_current_operator(
    target: GalerkinLocalCellTargetManifest,
    *,
    terminal_plane_coordinate: object,
    current_scope: GalerkinLocalTerminalScope | str,
    maximum_direct_pairs: object,
) -> GalerkinLocalCurrentOperatorCertificate:
    """Replay a local target and certify one scoped coordinate operator.

    :see: :func:`~.test_local_terminal.\
test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity`

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Completed local-cell target to authenticate in full.
    terminal_plane_coordinate : object
        Exact stored finite normal-or-zero scalar float64 plane coordinate.
    current_scope : GalerkinLocalTerminalScope | str
        Full-state or selected-preterminal complete transverse-fiber scope.
    maximum_direct_pairs : object
        Independent positive signed-int64 linear-work budget.

    Returns
    -------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Provisional raw uniform operator evidence or typed noncertificate.

    Raises
    ------
    TypeError
        If the target is not the local-cell route or an input type is invalid.
    ValueError
        If target replay, coordinate, scope, or budget validation fails.
    """
    if not isinstance(target, GalerkinLocalCellTargetManifest):
        raise TypeError(
            "target must be GalerkinLocalCellTargetManifest; legacy target "
            "coercion is forbidden"
        )
    coordinate = _checked_coordinate(terminal_plane_coordinate)
    scope = _checked_scope(current_scope)
    budget = _checked_maximum_direct_pairs(maximum_direct_pairs)
    prepared_target = prepare_local_cell_galerkin_target(target)
    certificate = _certify_prepared_operator(
        prepared_target, coordinate, scope, budget
    )
    return certificate  # noqa: RET504


def prepare_local_terminal_current_operator(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    *,
    maximum_direct_pairs: object,
) -> GalerkinPreparedLocalCurrentOperator:
    """Replay raw operator storage and return the prepared JIT capability.

    :see: :func:`~.test_local_terminal.\
test_prepare_rejects_operator_action_current_and_parent_forgeries`

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Raw provisional operator storage to authenticate in full.
    maximum_direct_pairs : object
        Independently supplied positive signed-int64 work policy.

    Returns
    -------
    prepared : GalerkinPreparedLocalCurrentOperator
        Explicit host-replayed capability accepted by frozen actions.

    Raises
    ------
    TypeError
        If raw storage has the wrong carrier type.
    ValueError
        If target, policy, operator, evidence, or digest replay differs.

    Notes
    -----
    The wrapper is an explicit caller trust marker, not an unforgeable Python
    token.  Host scientific boundaries replay the nested raw certificate and
    never accept a caller-constructed wrapper as authentication.
    """
    if not isinstance(certificate, GalerkinLocalCurrentOperatorCertificate):
        raise TypeError("certificate must be raw local operator storage")
    budget = _checked_maximum_direct_pairs(maximum_direct_pairs)
    prepared_target = prepare_local_cell_galerkin_target(certificate.target)
    canonical = _certify_prepared_operator(
        prepared_target,
        _checked_coordinate(certificate.terminal_plane_coordinate),
        _checked_scope(certificate.current_scope),
        budget,
    )
    if stored_value_payload(canonical) != stored_value_payload(certificate):
        raise ValueError(
            "local terminal operator failed complete target/operator/policy "
            "replay"
        )
    prepared = _make_prepared_local_current_operator(canonical)
    return prepared  # noqa: RET504


@jaxtyped(typechecker=beartype)
def apply_local_terminal_trace(
    prepared: GalerkinPreparedLocalCurrentOperator,
    field: Complex[Array, "..."],
) -> Complex128[Array, " f"]:
    """Apply the frozen carrier-stripped trace at the bound coordinate.

    :see: :func:`~.test_local_terminal.\
test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity`

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Wrapper returned by host operator preparation.
    field : Complex[Array, "..."]
        Retained-state coefficients.

    Returns
    -------
    trace : Complex128[Array, " f"]
        Scoped carrier-stripped coordinate trace.

    Raises
    ------
    TypeError
        If raw target/certificate storage is passed in place of preparation.
    ValueError
        If field rank or length is invalid.
    equinox.EquinoxRuntimeError
        If operator, input, or output is ineligible/non-normal.
    """
    certificate = _checked_prepared(prepared)
    checked = _checked_state(certificate, field)
    trace: Complex128[Array, " f"] = _checked_output(
        _raw_trace(certificate, checked), "local terminal trace"
    )
    return trace


@jaxtyped(typechecker=beartype)
def apply_local_terminal_trace_adjoint(
    prepared: GalerkinPreparedLocalCurrentOperator,
    terminal_field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the actual conjugate transpose of the frozen trace.

    :see: :func:`~.test_local_terminal.\
test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity`

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Wrapper returned by host operator preparation.
    terminal_field : Complex[Array, "..."]
        Scoped transverse-fiber coefficients.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        Retained-state trace-adjoint action.

    Raises
    ------
    TypeError
        If raw storage is passed instead of the prepared wrapper.
    ValueError
        If terminal rank or length is invalid.
    equinox.EquinoxRuntimeError
        If operator, input, or output is ineligible/non-normal.
    """
    certificate = _checked_prepared(prepared)
    checked = _checked_terminal_vector(certificate, terminal_field)
    adjoint: Complex128[Array, " n"] = _checked_output(
        _raw_trace_adjoint(certificate, checked),
        "local terminal trace adjoint",
    )
    return adjoint


@jaxtyped(typechecker=beartype)
def apply_local_terminal_normal_derivative(
    prepared: GalerkinPreparedLocalCurrentOperator,
    field: Complex[Array, "..."],
) -> Complex128[Array, " f"]:
    """Apply the side-oriented frozen physical normal trace.

    :see: :func:`~.test_local_terminal.\
test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity`

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Wrapper returned by host operator preparation.
    field : Complex[Array, "..."]
        Retained-state coefficients.

    Returns
    -------
    normal : Complex128[Array, " f"]
        Scoped oriented normal-derivative trace.

    Raises
    ------
    TypeError
        If raw storage is passed instead of the prepared wrapper.
    ValueError
        If field rank or length is invalid.
    equinox.EquinoxRuntimeError
        If operator, input, or output is ineligible/non-normal.
    """
    certificate = _checked_prepared(prepared)
    checked = _checked_state(certificate, field)
    normal: Complex128[Array, " f"] = _checked_output(
        _raw_normal(certificate, checked), "local terminal normal derivative"
    )
    return normal


@jaxtyped(typechecker=beartype)
def apply_local_terminal_normal_derivative_adjoint(
    prepared: GalerkinPreparedLocalCurrentOperator,
    terminal_field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the actual conjugate transpose of the frozen normal trace.

    :see: :func:`~.test_local_terminal.\
test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity`

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Wrapper returned by host operator preparation.
    terminal_field : Complex[Array, "..."]
        Scoped transverse-fiber coefficients.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        Retained-state normal-trace-adjoint action.

    Raises
    ------
    TypeError
        If raw storage is passed instead of the prepared wrapper.
    ValueError
        If terminal rank or length is invalid.
    equinox.EquinoxRuntimeError
        If operator, input, or output is ineligible/non-normal.
    """
    certificate = _checked_prepared(prepared)
    checked = _checked_terminal_vector(certificate, terminal_field)
    adjoint: Complex128[Array, " n"] = _checked_output(
        _raw_normal_adjoint(certificate, checked),
        "local terminal normal derivative adjoint",
    )
    return adjoint


@jaxtyped(typechecker=beartype)
def apply_local_terminal_current(
    prepared: GalerkinPreparedLocalCurrentOperator,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the implicit actual frozen Hermitian current matrix.

    :see: :func:`~.test_local_terminal.\
test_nonzero_coordinate_dense_actions_adjoints_and_hermiticity`

    Parameters
    ----------
    prepared : GalerkinPreparedLocalCurrentOperator
        Wrapper returned by host operator preparation.
    field : Complex[Array, "..."]
        Retained-state coefficients.

    Returns
    -------
    action : Complex128[Array, " n"]
        ``(T* N-N* T) field/(2i)``.

    Raises
    ------
    TypeError
        If raw storage is passed instead of the prepared wrapper.
    ValueError
        If field rank or length is invalid.
    equinox.EquinoxRuntimeError
        If operator, input, or output is ineligible/non-normal.
    """
    certificate = _checked_prepared(prepared)
    checked = _checked_state(certificate, field)
    action: Complex128[Array, " n"] = _checked_output(
        _raw_current_action(certificate, checked), "local terminal current"
    )
    return action


def enclose_local_terminal_current_action(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: object,
    *,
    maximum_direct_pairs: object,
) -> GalerkinLocalTerminalCurrentActionEnclosure:
    """Replay the operator and enclose one frozen current action.

    :see: :func:`~.test_local_terminal.\
test_action_and_exact_current_intervals_match_independent_oracles`

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Raw operator storage to replay; prepared wrappers are not accepted.
    field : object
        Submitted complex retained-state vector.
    maximum_direct_pairs : object
        Independent positive signed-int64 work policy.

    Returns
    -------
    enclosure : GalerkinLocalTerminalCurrentActionEnclosure
        Frozen-action arithmetic evidence or typed noncertificate.

    Raises
    ------
    TypeError
        If the operator, field dtype, or work policy has the wrong type.
    ValueError
        If operator replay or field shape fails.
    """
    if not isinstance(certificate, GalerkinLocalCurrentOperatorCertificate):
        raise TypeError("certificate must be raw local operator storage")
    budget = _checked_maximum_direct_pairs(maximum_direct_pairs)
    prepared = prepare_local_terminal_current_operator(
        certificate, maximum_direct_pairs=budget
    )
    enclosure = _enclose_action_prepared(prepared, field, budget)
    return enclosure  # noqa: RET504


def prepare_local_terminal_current_action(
    enclosure: GalerkinLocalTerminalCurrentActionEnclosure,
    *,
    maximum_direct_pairs: object,
) -> GalerkinLocalTerminalCurrentActionEnclosure:
    """Replay complete operator, field, frozen action, policy, and evidence.

    :see: :func:`~.test_local_terminal.\
test_prepare_rejects_operator_action_current_and_parent_forgeries`

    Parameters
    ----------
    enclosure : GalerkinLocalTerminalCurrentActionEnclosure
        Raw per-call action storage to authenticate.
    maximum_direct_pairs : object
        Independently supplied positive signed-int64 work policy.

    Returns
    -------
    canonical : GalerkinLocalTerminalCurrentActionEnclosure
        Fresh canonically reconstructed action evidence.

    Raises
    ------
    TypeError
        If the enclosure or policy has the wrong type.
    ValueError
        If any parent, action, work, outcome, or digest payload differs.
    """
    if not isinstance(enclosure, GalerkinLocalTerminalCurrentActionEnclosure):
        raise TypeError("enclosure has the wrong local action type")
    budget = _checked_maximum_direct_pairs(maximum_direct_pairs)
    canonical = enclose_local_terminal_current_action(
        enclosure.certificate,
        enclosure.submitted_field,
        maximum_direct_pairs=budget,
    )
    if stored_value_payload(canonical) != stored_value_payload(enclosure):
        raise ValueError("local terminal action failed complete host replay")
    return canonical


def enclose_local_terminal_current(
    certificate: GalerkinLocalCurrentOperatorCertificate,
    field: object,
    *,
    maximum_direct_pairs: object,
) -> GalerkinLocalCoordinateCauchyCurrent:
    """Replay the operator and enclose one direct exact-target current.

    :see: :func:`~.test_local_terminal.\
test_action_and_exact_current_intervals_match_independent_oracles`

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Raw operator storage to replay; prepared wrappers are not accepted.
    field : object
        Submitted complex retained-state vector.
    maximum_direct_pairs : object
        Independent positive signed-int64 work policy.

    Returns
    -------
    diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Scoped exact-target current evidence or typed noncertificate.

    Raises
    ------
    TypeError
        If operator, field dtype, or work policy has the wrong type.
    ValueError
        If operator replay or field shape fails.
    """
    if not isinstance(certificate, GalerkinLocalCurrentOperatorCertificate):
        raise TypeError("certificate must be raw local operator storage")
    budget = _checked_maximum_direct_pairs(maximum_direct_pairs)
    prepared = prepare_local_terminal_current_operator(
        certificate, maximum_direct_pairs=budget
    )
    diagnostic = _enclose_current_prepared(prepared, field, budget)
    return diagnostic  # noqa: RET504


def prepare_local_terminal_current(
    diagnostic: GalerkinLocalCoordinateCauchyCurrent,
    *,
    maximum_direct_pairs: object,
) -> GalerkinLocalCoordinateCauchyCurrent:
    """Replay complete operator, action, exact current, policy, and evidence.

    :see: :func:`~.test_local_terminal.\
test_prepare_rejects_operator_action_current_and_parent_forgeries`

    Parameters
    ----------
    diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Raw submitted-state current storage to authenticate.
    maximum_direct_pairs : object
        Independently supplied positive signed-int64 work policy.

    Returns
    -------
    canonical : GalerkinLocalCoordinateCauchyCurrent
        Fresh canonically reconstructed current diagnostic.

    Raises
    ------
    TypeError
        If the diagnostic or policy has the wrong type.
    ValueError
        If parent, action, current, policy, or digest replay differs.
    """
    if not isinstance(diagnostic, GalerkinLocalCoordinateCauchyCurrent):
        raise TypeError("diagnostic has the wrong local current type")
    budget = _checked_maximum_direct_pairs(maximum_direct_pairs)
    canonical = enclose_local_terminal_current(
        diagnostic.action_enclosure.certificate,
        diagnostic.action_enclosure.submitted_field,
        maximum_direct_pairs=budget,
    )
    if stored_value_payload(canonical) != stored_value_payload(diagnostic):
        raise ValueError("local terminal current failed complete host replay")
    return canonical


__all__: list[str] = [
    "apply_local_terminal_current",
    "apply_local_terminal_normal_derivative",
    "apply_local_terminal_normal_derivative_adjoint",
    "apply_local_terminal_trace",
    "apply_local_terminal_trace_adjoint",
    "certify_local_terminal_current_operator",
    "enclose_local_terminal_current",
    "enclose_local_terminal_current_action",
    "prepare_local_terminal_current",
    "prepare_local_terminal_current_action",
    "prepare_local_terminal_current_operator",
]
