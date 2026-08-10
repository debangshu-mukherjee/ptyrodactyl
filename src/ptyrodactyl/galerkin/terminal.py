r"""Apply and enclose the coordinate Cauchy/current diagnostic.

Extended Summary
----------------
This module realizes the bounded RM-S4a coordinate terminal attached to the
acquisition-selected ``K_d`` transverse-fiber sector of one canonical scalar
target.  It applies the selected field trace ``T``, the oriented physical
normal-derivative trace ``N``, their actual coefficient-metric adjoints, and
the Hermitian reduced-current action
``F u=(T* N-N* T)u/(2i)`` without assembling dense matrices.  A separate
evidence path certifies only the submitted-state exact normalized-carrier
current with shared FTZ-safe interval arithmetic; it does not certify a
uniform exact operator or rounded-action error bound.

Routine Listings
----------------
:func:`apply_galerkin_terminal_current`
    Apply the Hermitian selected-fiber current action.
:func:`apply_galerkin_terminal_normal_derivative`
    Apply the selected oriented normal-derivative trace.
:func:`apply_galerkin_terminal_normal_derivative_adjoint`
    Apply the selected coefficient-metric normal-trace adjoint.
:func:`apply_galerkin_terminal_trace`
    Apply the selected coordinate field trace.
:func:`apply_galerkin_terminal_trace_adjoint`
    Apply the selected coefficient-metric field-trace adjoint.
:func:`enclose_galerkin_terminal_current`
    Enclose one submitted-state exact selected-fiber current.
:func:`evaluate_galerkin_terminal_current`
    Evaluate the rounded selected-fiber current quadratic form.

Notes
-----
The selected terminal consists only of complete retained normal-frequency
fibers whose transverse indices are declared by the acquisition ``K_d`` set.
Modes in ``K_u`` on unselected transverse fibers are annihilated.  Therefore
this diagnostic is not a total full-plane current unless a separate coverage
argument identifies ``K_d`` with every retained transverse fiber; this route
does not make that identity claim.  The trace coordinate is zero, and the
declared side reverses ``N`` and ``F``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float64,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import (
    RealInterval,
    all_normal_arithmetic_supported,
    arithmetic_environment_probes,
    interval_add,
    interval_divide_positive,
    interval_multiply,
    interval_subtract,
    mathematical_pi_interval,
    point_interval,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportStatus,
    GalerkinCoordinateCauchyCurrent,
    GalerkinDetectorFailure,
    GalerkinTargetManifest,
    GalerkinTerminalCurrentFailure,
    GalerkinTerminalCurrentRoute,
    GalerkinTerminalSide,
    GalerkinVacuumBranchFailure,
    create_galerkin_coordinate_cauchy_current,
    scalar_int,
)

from ._direct_interval import (
    _complex_interval_add,
    _complex_interval_conjugate,
    _complex_interval_multiply,
    _complex_point_interval,
    _ComplexInterval,
)

_SPACE_DIMENSIONS: int = 3
_TRANSVERSE_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))
_ROUTE: GalerkinTerminalCurrentRoute = (
    GalerkinTerminalCurrentRoute.FTZ_SAFE_EXACT_CARRIER_CAUCHY
)
_COEFFICIENT_METRICS: str = (
    "state: SC.12 box-L2 orthonormal Euclidean complex coefficients; "
    "terminal: transverse-plane-L2 orthonormal Euclidean complex "
    "coefficients"
)
_CURRENT_TARGET: str = (
    "exact-real oriented coordinate-plane reduced current at xi=0 summed "
    "only over complete retained normal-frequency fibers selected by "
    "acquisition K_d, for the exact normalized SC-1 carrier, exact "
    "manifested box, mathematical pi, and exact stored binary64 submitted "
    "coefficients"
)
_ELIGIBILITY_SCOPE: str = (
    "per-submitted-state RM-S4a exact selected-K_d-fiber-sector current "
    "scalar enclosure; excludes unselected K_u transverse fibers and does "
    "not claim total/full-plane current identity; the rounded Hermitian F "
    "action has no uniform exact-operator/action-error certificate; excludes "
    "a compact local vacuum slab, outgoing branch extraction, detector "
    "pixels, absolute number-flux calibration, solver-to-observable "
    "transfer, and continuum accuracy"
)
_VACUUM_FAILURE: GalerkinVacuumBranchFailure = (
    GalerkinVacuumBranchFailure.NO_COMPACT_LOCAL_VACUUM_SLAB_CONTRACT
)
_DETECTOR_FAILURE: GalerkinDetectorFailure = (
    GalerkinDetectorFailure.NO_VACUUM_BRANCH
    | GalerkinDetectorFailure.NO_OUTGOING_EXTRACTION
    | GalerkinDetectorFailure.NO_PIXEL_RESPONSE
)


def _checked_state(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Convert and validate one retained-state vector.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target that fixes the state length.
    field : Complex[Array, "..."]
        Candidate retained-state coefficients.
    name : str
        Field name used in validation messages.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite binary64-complex state vector.

    Raises
    ------
    ValueError
        If the candidate is not one-dimensional or has the wrong length.
    equinox.EquinoxRuntimeError
        If one component is nonfinite.
    """
    values: Complex128[Array, " n"] = jnp.asarray(field, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if values.shape[0] != target.support.state_indices.shape[0]:
        raise ValueError(f"{name} must match K_u")
    checked: Complex128[Array, " n"] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        f"{name} must be finite",
    )
    return checked


def _checked_terminal_vector(
    target: GalerkinTargetManifest,
    values: Complex[Array, "..."],
    name: str,
) -> Complex128[Array, " t"]:
    """PRIVATE: Convert and validate one transverse terminal vector.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target that fixes the selected terminal fibers.
    values : Complex[Array, "..."]
        Candidate transverse-plane coefficients.
    name : str
        Field name used in validation messages.

    Returns
    -------
    checked : Complex128[Array, " t"]
        Finite binary64-complex terminal vector.

    Raises
    ------
    ValueError
        If the candidate is not one-dimensional or has the wrong length.
    equinox.EquinoxRuntimeError
        If one component is nonfinite.
    """
    vector: Complex128[Array, " t"] = jnp.asarray(values, dtype=jnp.complex128)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
    if vector.shape[0] != terminal_size:
        raise ValueError(f"{name} must match the selected terminal fibers")
    checked: Complex128[Array, " t"] = eqx.error_if(
        vector,
        jnp.any(~jnp.isfinite(vector)),
        f"{name} must be finite",
    )
    return checked


def _terminal_row_map(
    target: GalerkinTargetManifest,
) -> Tuple[Int64[Array, " n"], Bool[Array, " n"]]:
    """PRIVATE: Map retained modes to selected transverse terminal fibers.

    Implementation Logic
    --------------------
    1. Remove the static normal coordinate from state indices.
    2. Encode transverse indices in the endpoint-safe work quotient.
    3. Search selected-fiber keys and confirm exact integer equality.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target with checked state and terminal supports.

    Returns
    -------
    rows : Int64[Array, " n"]
        Safe selected-fiber row for every state coefficient.
    selected : Bool[Array, " n"]
        Whether the state coefficient belongs to the selected terminal.

    Notes
    -----
    The exact-equality check prevents a quotient-key collision from becoming
    terminal membership.  Complete-fiber eligibility is supplied by RM-S1.
    """
    axis: int = target.acquisition.terminal_axis
    transverse_axes: Tuple[int, int] = _TRANSVERSE_AXES[axis]
    state_transverse: Int64[Array, "n 2"] = target.support.state_indices[
        :, transverse_axes
    ]
    terminal_transverse: Int64[Array, "t 2"] = (
        target.acquisition.transverse_indices
    )
    work_shape: Tuple[int, int, int] = target.support.work_shape
    first_modulus: int = work_shape[transverse_axes[0]]
    second_modulus: int = work_shape[transverse_axes[1]]
    moduli: Int64[Array, " 2"] = jnp.asarray(
        (first_modulus, second_modulus), dtype=jnp.int64
    )
    terminal_residues: Int64[Array, "t 2"] = jnp.mod(
        terminal_transverse, moduli
    )
    terminal_keys: Int64[Array, " t"] = (
        terminal_residues[:, 0] * second_modulus + terminal_residues[:, 1]
    )
    order: Int64[Array, " t"] = jnp.argsort(terminal_keys)
    sorted_keys: Int64[Array, " t"] = terminal_keys[order]
    state_residues: Int64[Array, "n 2"] = jnp.mod(state_transverse, moduli)
    state_keys: Int64[Array, " n"] = (
        state_residues[:, 0] * second_modulus + state_residues[:, 1]
    )
    locations: Int64[Array, " n"] = jnp.searchsorted(
        sorted_keys, state_keys, side="left"
    )
    clipped_locations: Int64[Array, " n"] = jnp.clip(
        locations, 0, terminal_transverse.shape[0] - 1
    )
    rows: Int64[Array, " n"] = order[clipped_locations]
    selected: Bool[Array, " n"] = (
        (locations < terminal_transverse.shape[0])
        & (sorted_keys[clipped_locations] == state_keys)
        & jnp.all(terminal_transverse[rows] == state_transverse, axis=1)
    )
    result: Tuple[Int64[Array, " n"], Bool[Array, " n"]] = (
        rows,
        selected,
    )
    return result


def _trace_normalization(
    target: GalerkinTargetManifest,
) -> Float64[Array, ""]:
    """PRIVATE: Return the coordinate trace normalization.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target with exact stored positive box lengths.

    Returns
    -------
    normalization : Float64[Array, ""]
        Rounded ``L_n**(-1/2)`` in inverse square-root Angstroms.
    """
    axis: int = target.acquisition.terminal_axis
    normalization: Float64[Array, ""] = jax.lax.rsqrt(target.box_lengths[axis])
    return normalization


def _rounded_oriented_normal_wavevectors(
    target: GalerkinTargetManifest,
) -> Float64[Array, " n"]:
    """PRIVATE: Evaluate stored-geometry oriented normal wavevectors.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target with stored algebraic carrier and box geometry.

    Returns
    -------
    wavevectors : Float64[Array, " n"]
        Rounded ``s(k_i,n + 2 pi g_n/L_n)`` values.
    """
    axis: int = target.acquisition.terminal_axis
    side_sign: float = (
        1.0
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else -1.0
    )
    indices: Float64[Array, " n"] = target.support.state_indices[
        :, axis
    ].astype(jnp.float64)
    offsets: Float64[Array, " n"] = (
        2.0 * jnp.pi * indices / target.box_lengths[axis]
    )
    wavevectors: Float64[Array, " n"] = side_sign * (
        target.carrier[axis] + offsets
    )
    return wavevectors


def _raw_trace(
    target: GalerkinTargetManifest,
    field: Complex128[Array, " n"],
) -> Complex128[Array, " t"]:
    """PRIVATE: Apply ``T`` without post-action finite rejection.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical coordinate-terminal target.
    field : Complex128[Array, " n"]
        Finite retained-state coefficients.

    Returns
    -------
    trace : Complex128[Array, " t"]
        Rounded transverse field-trace coefficients.
    """
    rows, selected = _terminal_row_map(target)
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
    contributions: Complex128[Array, " n"] = jnp.where(
        selected,
        _trace_normalization(target) * field,
        0.0 + 0.0j,
    )
    trace: Complex128[Array, " t"] = (
        jnp.zeros((terminal_size,), dtype=jnp.complex128)
        .at[rows]
        .add(contributions)
    )
    return trace


def _raw_trace_adjoint(
    target: GalerkinTargetManifest,
    terminal_field: Complex128[Array, " t"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply ``T*`` without post-action finite rejection.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical coordinate-terminal target.
    terminal_field : Complex128[Array, " t"]
        Finite transverse-plane coefficients.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        Rounded actual coefficient-metric trace adjoint.
    """
    rows, selected = _terminal_row_map(target)
    gathered: Complex128[Array, " n"] = terminal_field[rows]
    adjoint: Complex128[Array, " n"] = jnp.where(
        selected,
        _trace_normalization(target) * gathered,
        0.0 + 0.0j,
    )
    return adjoint


def _raw_normal_derivative(
    target: GalerkinTargetManifest,
    field: Complex128[Array, " n"],
) -> Complex128[Array, " t"]:
    """PRIVATE: Apply ``N`` without post-action finite rejection.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical coordinate-terminal target.
    field : Complex128[Array, " n"]
        Finite retained-state coefficients.

    Returns
    -------
    normal_trace : Complex128[Array, " t"]
        Rounded oriented physical normal-derivative coefficients.
    """
    rows, selected = _terminal_row_map(target)
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
    multipliers: Complex128[Array, " n"] = (
        1j
        * _trace_normalization(target)
        * _rounded_oriented_normal_wavevectors(target)
    )
    contributions: Complex128[Array, " n"] = jnp.where(
        selected,
        multipliers * field,
        0.0 + 0.0j,
    )
    normal_trace: Complex128[Array, " t"] = (
        jnp.zeros((terminal_size,), dtype=jnp.complex128)
        .at[rows]
        .add(contributions)
    )
    return normal_trace


def _raw_normal_derivative_adjoint(
    target: GalerkinTargetManifest,
    terminal_field: Complex128[Array, " t"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply ``N*`` without post-action finite rejection.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical coordinate-terminal target.
    terminal_field : Complex128[Array, " t"]
        Finite transverse-plane coefficients.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        Rounded actual coefficient-metric normal-trace adjoint.
    """
    rows, selected = _terminal_row_map(target)
    multipliers: Complex128[Array, " n"] = (
        -1j
        * _trace_normalization(target)
        * _rounded_oriented_normal_wavevectors(target)
    )
    gathered: Complex128[Array, " n"] = terminal_field[rows]
    adjoint: Complex128[Array, " n"] = jnp.where(
        selected,
        multipliers * gathered,
        0.0 + 0.0j,
    )
    return adjoint


def _raw_current_action(
    target: GalerkinTargetManifest,
    field: Complex128[Array, " n"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply ``F=(T*N-N*T)/(2i)`` without finite rejection.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical coordinate-terminal target.
    field : Complex128[Array, " n"]
        Finite retained-state coefficients.

    Returns
    -------
    action : Complex128[Array, " n"]
        Rounded Hermitian reduced-current action.
    """
    trace: Complex128[Array, " t"] = _raw_trace(target, field)
    normal_trace: Complex128[Array, " t"] = _raw_normal_derivative(
        target, field
    )
    trace_normal: Complex128[Array, " n"] = _raw_trace_adjoint(
        target, normal_trace
    )
    normal_trace_term: Complex128[Array, " n"] = (
        _raw_normal_derivative_adjoint(target, trace)
    )
    action: Complex128[Array, " n"] = (trace_normal - normal_trace_term) / (
        2.0j
    )
    return action


def _checked_action(
    action: Complex128[Array, " n"],
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Reject a nonfinite public terminal-map action.

    Parameters
    ----------
    action : Complex128[Array, " n"]
        Candidate public terminal-map output.
    name : str
        Action name used in the runtime rejection message.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite terminal-map output.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If one output component is nonfinite.
    """
    checked: Complex128[Array, " n"] = eqx.error_if(
        action,
        jnp.any(~jnp.isfinite(action)),
        f"{name} must be finite",
    )
    return checked


def _exact_oriented_normal_wavevector_interval(
    target: GalerkinTargetManifest,
) -> RealInterval:
    """PRIVATE: Enclose exact-target oriented normal wavevectors.

    Implementation Logic
    --------------------
    1. Enclose exact reciprocal offsets from integer indices and box length.
    2. Multiply by an outward mathematical ``2 pi`` interval.
    3. Add the RM-S2 exact normalized-carrier interval and orient by side.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Target carrying exact-carrier and exact-box evidence.

    Returns
    -------
    interval : RealInterval
        Inclusive ``s(k_i,n+2 pi g_n/L_n)`` endpoints.
    """
    axis: int = target.acquisition.terminal_axis
    indices: Float64[Array, " n"] = target.support.state_indices[
        :, axis
    ].astype(jnp.float64)
    reciprocal: RealInterval = interval_divide_positive(
        point_interval(indices),
        point_interval(target.box_lengths[axis]),
    )
    two_pi: RealInterval = interval_multiply(
        point_interval(jnp.asarray(2.0, dtype=jnp.float64)),
        mathematical_pi_interval(),
    )
    offsets: RealInterval = interval_multiply(reciprocal, two_pi)
    ledger = target.fixed_linear_error_ledger
    carrier: RealInterval = (
        jnp.broadcast_to(
            ledger.exact_carrier_lower_bounds[axis], indices.shape
        ),
        jnp.broadcast_to(
            ledger.exact_carrier_upper_bounds[axis], indices.shape
        ),
    )
    unoriented: RealInterval = interval_add(carrier, offsets)
    if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE:
        interval: RealInterval = unoriented
    else:
        interval = (-unoriented[1], -unoriented[0])
    return interval


def _exact_current_interval(
    target: GalerkinTargetManifest,
    field: Complex128[Array, " n"],
) -> RealInterval:
    """PRIVATE: Enclose the exact normalized-carrier coordinate current.

    Implementation Logic
    --------------------
    1. Accumulate the complex field sum and wavevector-weighted sum within
       every selected transverse fiber.
    2. Form ``Re(conj(sum u) * sum(q u))`` only after each fiber sum, thereby
       retaining every normal-frequency interference cross term.
    3. Sum the fiber currents and divide outwardly by the exact box length.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target and exact geometry evidence.
    field : Complex128[Array, " n"]
        Exact stored binary64 submitted state.

    Returns
    -------
    interval : RealInterval
        Inclusive exact oriented reduced-current endpoints.

    Notes
    -----
    This sum-before-product evaluation implements
    ``Im(<T u,N u>)`` without explicitly interval-evaluating square roots.
    The exact algebraic cancellation leaves ``L_n**-1``.
    """
    rows, selected = _terminal_row_map(target)
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
    wavevectors: RealInterval = _exact_oriented_normal_wavevector_interval(
        target
    )
    zeros: Float64[Array, " t"] = jnp.zeros(
        (terminal_size,), dtype=jnp.float64
    )
    initial: Tuple[_ComplexInterval, _ComplexInterval] = (
        (zeros, zeros, zeros, zeros),
        (zeros, zeros, zeros, zeros),
    )

    def add_state(
        index: scalar_int,
        accumulator: Tuple[_ComplexInterval, _ComplexInterval],
    ) -> Tuple[_ComplexInterval, _ComplexInterval]:
        """Accumulate one selected state coefficient into its exact fiber."""
        row: scalar_int = rows[index]
        included: Bool[Array, ""] = selected[index]
        value: _ComplexInterval = _complex_point_interval(field[index])
        selected_value: _ComplexInterval = (
            jnp.where(included, value[0], 0.0),
            jnp.where(included, value[1], 0.0),
            jnp.where(included, value[2], 0.0),
            jnp.where(included, value[3], 0.0),
        )
        wavevector: _ComplexInterval = (
            wavevectors[0][index],
            wavevectors[1][index],
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        weighted_value: _ComplexInterval = _complex_interval_multiply(
            wavevector,
            selected_value,
        )
        prior_sum: _ComplexInterval = (
            accumulator[0][0][row],
            accumulator[0][1][row],
            accumulator[0][2][row],
            accumulator[0][3][row],
        )
        prior_weighted: _ComplexInterval = (
            accumulator[1][0][row],
            accumulator[1][1][row],
            accumulator[1][2][row],
            accumulator[1][3][row],
        )
        updated_sum: _ComplexInterval = _complex_interval_add(
            prior_sum, selected_value
        )
        updated_weighted: _ComplexInterval = _complex_interval_add(
            prior_weighted, weighted_value
        )
        result: Tuple[_ComplexInterval, _ComplexInterval] = (
            tuple(
                component.at[row].set(value_component)
                for component, value_component in zip(
                    accumulator[0], updated_sum, strict=True
                )
            ),
            tuple(
                component.at[row].set(value_component)
                for component, value_component in zip(
                    accumulator[1], updated_weighted, strict=True
                )
            ),
        )
        return result

    fiber_sum: _ComplexInterval
    weighted_fiber_sum: _ComplexInterval
    fiber_sum, weighted_fiber_sum = lax.fori_loop(
        0,
        field.shape[0],
        add_state,
        initial,
    )
    fiber_products: _ComplexInterval = _complex_interval_multiply(
        _complex_interval_conjugate(fiber_sum),
        weighted_fiber_sum,
    )
    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)

    def add_fiber(
        index: scalar_int,
        accumulator: RealInterval,
    ) -> RealInterval:
        """Accumulate one outward real fiber-current interval."""
        contribution: RealInterval = (
            fiber_products[0][index],
            fiber_products[1][index],
        )
        updated: RealInterval = interval_add(accumulator, contribution)
        return updated

    summed: RealInterval = lax.fori_loop(
        0,
        terminal_size,
        add_fiber,
        (zero, zero),
    )
    axis: int = target.acquisition.terminal_axis
    interval: RealInterval = interval_divide_positive(
        summed,
        point_interval(target.box_lengths[axis]),
    )
    return interval


def _current_error_upper_bound(
    rounded_current: Float64[Array, ""],
    exact_interval: RealInterval,
) -> Float64[Array, ""]:
    """PRIVATE: Bound rounded-current distance from the exact interval.

    Parameters
    ----------
    rounded_current : Float64[Array, ""]
        Rounded current quadratic in inverse Angstroms.
    exact_interval : RealInterval
        Inclusive exact current interval in inverse Angstroms.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward maximum absolute endpoint discrepancy.
    """
    difference: RealInterval = interval_subtract(
        point_interval(rounded_current), exact_interval
    )
    upper: Float64[Array, ""] = jnp.maximum(
        jnp.abs(difference[0]), jnp.abs(difference[1])
    )
    return upper


@jaxtyped(typechecker=beartype)
def apply_galerkin_terminal_trace(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " t"]:
    """Apply the selected coordinate field trace.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target with a selected complete coordinate terminal.
    field : Complex[Array, "..."]
        Retained-state coefficient vector.

    Returns
    -------
    trace : Complex128[Array, " t"]
        Transverse-plane orthonormal coefficients ``T u``.

    Raises
    ------
    ValueError
        If the state rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the input or rounded trace is nonfinite.
    """
    checked_field: Complex128[Array, " n"] = _checked_state(
        target, field, "field"
    )
    raw_trace: Complex128[Array, " t"] = _raw_trace(target, checked_field)
    trace: Complex128[Array, " t"] = _checked_action(raw_trace, "trace")
    return trace


@jaxtyped(typechecker=beartype)
def apply_galerkin_terminal_trace_adjoint(
    target: GalerkinTargetManifest,
    terminal_field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the selected coefficient-metric field-trace adjoint.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target fixing the state and terminal coefficient metrics.
    terminal_field : Complex[Array, "..."]
        Transverse-plane coefficient vector.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        State coefficients ``T* y`` under the declared Euclidean metrics.

    Raises
    ------
    ValueError
        If the terminal vector rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the input or rounded adjoint is nonfinite.
    """
    checked_terminal: Complex128[Array, " t"] = _checked_terminal_vector(
        target, terminal_field, "terminal_field"
    )
    raw_adjoint: Complex128[Array, " n"] = _raw_trace_adjoint(
        target, checked_terminal
    )
    adjoint: Complex128[Array, " n"] = _checked_action(
        raw_adjoint, "trace adjoint"
    )
    return adjoint


@jaxtyped(typechecker=beartype)
def apply_galerkin_terminal_normal_derivative(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " t"]:
    """Apply the selected oriented normal-derivative trace.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target with carrier, box, axis, and oriented side.
    field : Complex[Array, "..."]
        Retained-state coefficient vector.

    Returns
    -------
    normal_trace : Complex128[Array, " t"]
        Oriented normal-derivative coefficients ``N u``.

    Raises
    ------
    ValueError
        If the state rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the input or rounded normal trace is nonfinite.
    """
    checked_field: Complex128[Array, " n"] = _checked_state(
        target, field, "field"
    )
    raw_trace: Complex128[Array, " t"] = _raw_normal_derivative(
        target, checked_field
    )
    normal_trace: Complex128[Array, " t"] = _checked_action(
        raw_trace, "normal-derivative trace"
    )
    return normal_trace


@jaxtyped(typechecker=beartype)
def apply_galerkin_terminal_normal_derivative_adjoint(
    target: GalerkinTargetManifest,
    terminal_field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the selected coefficient-metric normal-trace adjoint.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target fixing the state and terminal coefficient metrics.
    terminal_field : Complex[Array, "..."]
        Transverse-plane coefficient vector.

    Returns
    -------
    adjoint : Complex128[Array, " n"]
        State coefficients ``N* y`` under the declared Euclidean metrics.

    Raises
    ------
    ValueError
        If the terminal vector rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the input or rounded adjoint is nonfinite.
    """
    checked_terminal: Complex128[Array, " t"] = _checked_terminal_vector(
        target, terminal_field, "terminal_field"
    )
    raw_adjoint: Complex128[Array, " n"] = _raw_normal_derivative_adjoint(
        target, checked_terminal
    )
    adjoint: Complex128[Array, " n"] = _checked_action(
        raw_adjoint, "normal-derivative trace adjoint"
    )
    return adjoint


@jaxtyped(typechecker=beartype)
def apply_galerkin_terminal_current(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the Hermitian selected-fiber current action.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target fixing ``T``, ``N``, and both coefficient metrics.
    field : Complex[Array, "..."]
        Retained-state coefficient vector.

    Returns
    -------
    current_action : Complex128[Array, " n"]
        Linear action ``F u`` for ``F=(T* N-N* T)/(2i)``.

    Raises
    ------
    ValueError
        If the state rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the input or rounded current action is nonfinite.

    Notes
    -----
    This function returns a linear action, not the scalar flux.  Use
    :func:`evaluate_galerkin_terminal_current` for the quadratic form.
    """
    checked_field: Complex128[Array, " n"] = _checked_state(
        target, field, "field"
    )
    raw_action: Complex128[Array, " n"] = _raw_current_action(
        target, checked_field
    )
    current_action: Complex128[Array, " n"] = _checked_action(
        raw_action, "terminal current action"
    )
    return current_action


@jaxtyped(typechecker=beartype)
def evaluate_galerkin_terminal_current(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Float64[Array, ""]:
    """Evaluate the rounded selected-fiber current quadratic form.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target fixing the Hermitian current operator.
    field : Complex[Array, "..."]
        Retained-state coefficient vector.

    Returns
    -------
    reduced_current : Float64[Array, ""]
        Rounded ``Re(<u,F u>)`` in inverse Angstroms.

    Raises
    ------
    ValueError
        If the state rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the input, current action, or quadratic form is nonfinite.
    """
    checked_field: Complex128[Array, " n"] = _checked_state(
        target, field, "field"
    )
    current_action: Complex128[Array, " n"] = _checked_action(
        _raw_current_action(target, checked_field),
        "terminal current action",
    )
    raw_current: Float64[Array, ""] = jnp.real(
        jnp.vdot(checked_field, current_action)
    )
    reduced_current: Float64[Array, ""] = eqx.error_if(
        raw_current,
        ~jnp.isfinite(raw_current),
        "terminal reduced current must be finite",
    )
    return reduced_current


@jaxtyped(typechecker=beartype)
def enclose_galerkin_terminal_current(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> GalerkinCoordinateCauchyCurrent:
    """Enclose one submitted-state exact selected-fiber current.

    :see: :class:`~.test_terminal.TestCoordinateTerminal`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical scalar target and RM-S2 exact-geometry evidence.
    field : Complex[Array, "..."]
        Submitted retained-state coefficient vector.

    Returns
    -------
    diagnostic : GalerkinCoordinateCauchyCurrent
        Bound traces, current action, exact interval, and separately named
        current/vacuum/detector eligibility.

    Raises
    ------
    ValueError
        If the state rank or length is invalid.
    equinox.EquinoxRuntimeError
        If the submitted field is nonfinite or carrier construction fails.

    Notes
    -----
    Positive infinity is a typed current noncertificate.  It never becomes a
    physical vacuum-branch or detector claim.
    """
    checked_field: Complex128[Array, " n"] = _checked_state(
        target, field, "field"
    )
    trace: Complex128[Array, " t"] = _raw_trace(target, checked_field)
    normal_trace: Complex128[Array, " t"] = _raw_normal_derivative(
        target, checked_field
    )
    action: Complex128[Array, " n"] = _raw_current_action(
        target, checked_field
    )
    rounded_current: Float64[Array, ""] = jnp.real(
        jnp.vdot(checked_field, action)
    )
    exact_interval: RealInterval = _exact_current_interval(
        target, checked_field
    )
    error_upper: Float64[Array, ""] = _current_error_upper_bound(
        rounded_current, exact_interval
    )
    probes = arithmetic_environment_probes()
    arithmetic_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    gradual_supported: Bool[Array, ""] = probes[-1]
    support_eligible: Bool[Array, ""] = (
        target.support_eligibility.status
        == int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
    ) & target.support_eligibility.support_eligible
    fiber_complete: Bool[Array, ""] = (
        target.support_eligibility.terminal_fiber_complete
    )
    finite_evidence: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(trace))
        & jnp.all(jnp.isfinite(normal_trace))
        & jnp.all(jnp.isfinite(action))
        & jnp.isfinite(rounded_current)
        & jnp.isfinite(exact_interval[0])
        & jnp.isfinite(exact_interval[1])
        & jnp.isfinite(error_upper)
    )
    zero: Int64[Array, ""] = jnp.asarray(
        int(GalerkinTerminalCurrentFailure.NONE), dtype=jnp.int64
    )
    failure_mask: Int64[Array, ""] = zero
    failure_mask = jnp.bitwise_or(
        failure_mask,
        jnp.where(
            support_eligible,
            zero,
            int(GalerkinTerminalCurrentFailure.SUPPORT_INELIGIBLE),
        ),
    )
    failure_mask = jnp.bitwise_or(
        failure_mask,
        jnp.where(
            fiber_complete,
            zero,
            int(GalerkinTerminalCurrentFailure.TERMINAL_FIBER_INCOMPLETE),
        ),
    )
    failure_mask = jnp.bitwise_or(
        failure_mask,
        jnp.where(
            arithmetic_supported,
            zero,
            int(
                GalerkinTerminalCurrentFailure.ARITHMETIC_ENVIRONMENT_UNSUPPORTED
            ),
        ),
    )
    failure_mask = jnp.bitwise_or(
        failure_mask,
        jnp.where(
            finite_evidence,
            zero,
            int(GalerkinTerminalCurrentFailure.NONFINITE_CURRENT_EVIDENCE),
        ),
    )
    current_eligible: Bool[Array, ""] = failure_mask == int(
        GalerkinTerminalCurrentFailure.NONE
    )
    stopped_evidence = jax.tree.map(
        jax.lax.stop_gradient,
        (
            trace,
            normal_trace,
            action,
            rounded_current,
            exact_interval[0],
            exact_interval[1],
            error_upper,
            arithmetic_supported,
            gradual_supported,
            current_eligible,
            failure_mask,
        ),
    )
    diagnostic: GalerkinCoordinateCauchyCurrent = (
        create_galerkin_coordinate_cauchy_current(
            target=target,
            submitted_field=jax.lax.stop_gradient(checked_field),
            trace_coefficients=stopped_evidence[0],
            normal_derivative_coefficients=stopped_evidence[1],
            current_action=stopped_evidence[2],
            reduced_current=stopped_evidence[3],
            exact_reduced_current_lower_bound=stopped_evidence[4],
            exact_reduced_current_upper_bound=stopped_evidence[5],
            reduced_current_error_upper_bound=stopped_evidence[6],
            arithmetic_environment_supported=stopped_evidence[7],
            gradual_underflow_supported=stopped_evidence[8],
            current_diagnostic_eligible=stopped_evidence[9],
            current_diagnostic_failure_mask=stopped_evidence[10],
            vacuum_branch_eligible=jnp.asarray(False),
            detector_eligible=jnp.asarray(False),
            terminal_axis=target.acquisition.terminal_axis,
            terminal_side=target.acquisition.terminal_side,
            route=_ROUTE,
            vacuum_branch_failure=_VACUUM_FAILURE,
            detector_failure=_DETECTOR_FAILURE,
            coefficient_metrics=_COEFFICIENT_METRICS,
            current_target=_CURRENT_TARGET,
            eligibility_scope=_ELIGIBILITY_SCOPE,
        )
    )
    return diagnostic


__all__: list[str] = [
    "apply_galerkin_terminal_current",
    "apply_galerkin_terminal_normal_derivative",
    "apply_galerkin_terminal_normal_derivative_adjoint",
    "apply_galerkin_terminal_trace",
    "apply_galerkin_terminal_trace_adjoint",
    "enclose_galerkin_terminal_current",
    "evaluate_galerkin_terminal_current",
]
