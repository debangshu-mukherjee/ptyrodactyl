r"""Apply and enclose the coordinate Cauchy/current diagnostic.

Extended Summary
----------------
This module realizes the bounded RM-S4a coordinate terminal attached to the
acquisition-selected ``K_d`` transverse-fiber sector of one canonical scalar
target.  It applies the selected field trace ``T``, the oriented physical
normal-derivative trace ``N``, their actual coefficient-metric adjoints, and
the Hermitian reduced-current action
``F u=(T* N-N* T)u/(2i)`` without assembling dense matrices.  The evidence
ladder keeps the weaker submitted-state exact-current diagnostic separate
from a uniform frozen-operator certificate and an independently replayed
per-call frozen-action enclosure.

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
:func:`certify_galerkin_terminal_current_operator`
    Certify the uniform frozen selected-sector current operator.
:func:`enclose_galerkin_terminal_current`
    Enclose one submitted-state exact selected-fiber current.
:func:`enclose_galerkin_terminal_current_action`
    Enclose one frozen current action after certificate authentication.
:func:`evaluate_galerkin_terminal_current`
    Evaluate the rounded selected-fiber current quadratic form.
:func:`prepare_galerkin_terminal_current_diagnostic`
    Host-authenticate one provisional coordinate-current diagnostic.

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
    interval_sqrt,
    interval_square,
    interval_subtract,
    mathematical_pi_interval,
    point_interval,
)
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    HBAR,
    M_E,
    GalerkinAcquisitionSupportStatus,
    GalerkinCoordinateCauchyCurrent,
    GalerkinCurrentOperatorCertificate,
    GalerkinCurrentOperatorFailure,
    GalerkinDetectorFailure,
    GalerkinTargetManifest,
    GalerkinTerminalCurrentActionEnclosure,
    GalerkinTerminalCurrentActionFailure,
    GalerkinTerminalCurrentFailure,
    GalerkinTerminalCurrentRoute,
    GalerkinTerminalSide,
    GalerkinVacuumBranchFailure,
    create_galerkin_coordinate_cauchy_current,
    create_galerkin_current_operator_certificate,
    create_galerkin_terminal_current_action_enclosure,
    scalar_int,
)

from ._direct_interval import (
    _complex_interval_add,
    _complex_interval_conjugate,
    _complex_interval_multiply,
    _complex_point_interval,
    _ComplexInterval,
)
from .stability import _manifest_is_canonical

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
    "provisional transform-compatible per-submitted-state RM-S4a exact "
    "selected-K_d-fiber-sector current scalar enclosure; non-authoritative "
    "until host canonical-target reconstruction and exact diagnostic replay; "
    "excludes unselected K_u transverse fibers and does not claim "
    "total/full-plane current identity; the rounded Hermitian F action has no "
    "uniform exact-operator/action-error certificate; excludes a compact "
    "local vacuum slab, outgoing branch extraction, detector pixels, "
    "absolute number-flux calibration, solver-to-observable transfer, and "
    "continuum accuracy"
)
_FIXED_LINEAR_TARGET: str = (
    "exact selected-K_d coordinate T, oriented N, and Hermitian F at xi=0 "
    "for the exact normalized SC-1 carrier, exact manifested box, and "
    "mathematical pi"
)
_PER_CALL_ACTION_ROUTE: str = (
    "FTZ-safe exact-real interval evaluation of the frozen dyadic T/N/F "
    "matrix action, independently of the rounded public action"
)
_CURRENT_NORMALIZATION: str = (
    "SC.35c C_j=hbar/m_*=hbar*c^2/(m_e*c^2+e*U0) converted from square "
    "metres per second to square Angstroms per second"
)
_OPERATOR_ELIGIBILITY_SCOPE: str = (
    "uniform RM-S4 LVT.55a2 selected-K_d-fiber-sector signed current "
    "operator capability at xi=0; every scientific submitted-field result "
    "additionally requires that call's action enclosure finite_certificate; "
    "with independently replayed frozen per-call action enclosures; excludes "
    "unselected K_u transverse fibers, total/full-plane identity, vacuum "
    "branches, outgoing extraction, and detectors"
)
_ACTION_TARGET: str = (
    "exact-real action of the stored frozen dyadic current matrix "
    "Fhat=(That* Nhat-Nhat* That)/(2i) on the exact stored binary64 field"
)
_ACTION_ERROR_SCOPE: str = (
    "per-call Euclidean coefficient error from rounded public action to the "
    "exact-real frozen-matrix action; excludes the uniform frozen-to-exact "
    "target operator error, state error, and continuum error"
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
        Rounded current quadratic in inverse-square Angstroms.
    exact_interval : RealInterval
        Inclusive exact current interval in inverse-square Angstroms.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward maximum absolute endpoint discrepancy in inverse-square
        Angstroms.
    """
    difference: RealInterval = interval_subtract(
        point_interval(rounded_current), exact_interval
    )
    upper: Float64[Array, ""] = jnp.maximum(
        jnp.abs(difference[0]), jnp.abs(difference[1])
    )
    return upper


def _trace_normalization_interval(
    target: GalerkinTargetManifest,
) -> RealInterval:
    """PRIVATE: Enclose the exact coordinate trace normalization.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical scalar target supplying the terminal-axis box length.

    Returns
    -------
    interval : RealInterval
        Outward interval for the inverse square root of that box length.
    """
    axis: int = target.acquisition.terminal_axis
    root: RealInterval = interval_sqrt(
        point_interval(target.box_lengths[axis])
    )
    one: Float64[Array, ""] = jnp.asarray(1.0, dtype=jnp.float64)
    interval: RealInterval = interval_divide_positive(
        point_interval(one), root
    )
    return interval


def _distance_from_interval(
    point: Float64[Array, "..."],
    interval: RealInterval,
) -> Float64[Array, "..."]:
    """PRIVATE: Bound a stored real point's distance from an interval.

    Parameters
    ----------
    point : Float64[Array, "..."]
        Stored real point or array of points.
    interval : RealInterval
        Inclusive exact-real interval with shape broadcastable to ``point``.

    Returns
    -------
    upper : Float64[Array, "..."]
        Outward maximum absolute endpoint discrepancy.
    """
    difference: RealInterval = interval_subtract(
        point_interval(point), interval
    )
    upper: Float64[Array, "..."] = jnp.maximum(
        jnp.abs(difference[0]), jnp.abs(difference[1])
    )
    return upper


def _fiber_operator_norm_upper(
    target: GalerkinTargetManifest,
    coefficient_magnitude_upper_bounds: Float64[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Bound the maximum selected-fiber row Euclidean norm.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical scalar target defining selected terminal fibers.
    coefficient_magnitude_upper_bounds : Float64[Array, " n"]
        Per-state outward coefficient-magnitude bounds.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward maximum selected-fiber row Euclidean norm.
    """
    rows, selected = _terminal_row_map(target)
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
    zeros: Float64[Array, " t"] = jnp.zeros(
        (terminal_size,), dtype=jnp.float64
    )
    initial: RealInterval = (zeros, zeros)

    def add_coefficient(
        index: scalar_int,
        accumulator: RealInterval,
    ) -> RealInterval:
        """Accumulate one outward squared coefficient into its fiber."""
        magnitude: Float64[Array, ""] = jnp.where(
            selected[index], coefficient_magnitude_upper_bounds[index], 0.0
        )
        contribution: RealInterval = interval_square(point_interval(magnitude))
        row: scalar_int = rows[index]
        prior: RealInterval = (accumulator[0][row], accumulator[1][row])
        updated: RealInterval = interval_add(prior, contribution)
        result: RealInterval = (
            accumulator[0].at[row].set(updated[0]),
            accumulator[1].at[row].set(updated[1]),
        )
        return result

    squared_norms: RealInterval = lax.fori_loop(
        0,
        coefficient_magnitude_upper_bounds.shape[0],
        add_coefficient,
        initial,
    )
    norm_intervals: RealInterval = interval_sqrt(squared_norms)
    upper: Float64[Array, ""] = jnp.max(norm_intervals[1])
    return upper


def _frozen_terminal_coefficient_evidence(
    target: GalerkinTargetManifest,
) -> Tuple[
    Complex128[Array, " n"],
    Complex128[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
]:
    """PRIVATE: Build frozen T/N rows and exact-target error evidence.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical scalar target defining the selected terminal sector.

    Returns
    -------
    trace_coefficients : Complex128[Array, " n"]
        Frozen rounded trace coefficients in retained-state order.
    normal_coefficients : Complex128[Array, " n"]
        Frozen rounded oriented normal-derivative coefficients.
    trace_errors : Float64[Array, " n"]
        Per-state trace-coefficient error upper bounds.
    normal_errors : Float64[Array, " n"]
        Per-state normal-coefficient error upper bounds.
    exact_trace_magnitudes : Float64[Array, " n"]
        Per-state exact trace-coefficient magnitude upper bounds.
    exact_normal_magnitudes : Float64[Array, " n"]
        Per-state exact normal-coefficient magnitude upper bounds.
    """
    _, selected = _terminal_row_map(target)
    trace_normalization: Float64[Array, ""] = _trace_normalization(target)
    wavevectors: Float64[Array, " n"] = _rounded_oriented_normal_wavevectors(
        target
    )
    trace_coefficients: Complex128[Array, " n"] = jnp.where(
        selected,
        jnp.asarray(trace_normalization, dtype=jnp.complex128),
        0.0 + 0.0j,
    )
    normal_coefficients: Complex128[Array, " n"] = jnp.where(
        selected,
        1j * trace_normalization * wavevectors,
        0.0 + 0.0j,
    )

    trace_interval_scalar: RealInterval = _trace_normalization_interval(target)
    state_shape: Tuple[int, ...] = wavevectors.shape
    trace_interval: RealInterval = (
        jnp.broadcast_to(trace_interval_scalar[0], state_shape),
        jnp.broadcast_to(trace_interval_scalar[1], state_shape),
    )
    exact_wavevectors: RealInterval = (
        _exact_oriented_normal_wavevector_interval(target)
    )
    exact_normal: RealInterval = interval_multiply(
        trace_interval, exact_wavevectors
    )
    trace_errors: Float64[Array, " n"] = jnp.where(
        selected,
        _distance_from_interval(jnp.real(trace_coefficients), trace_interval),
        0.0,
    )
    normal_errors: Float64[Array, " n"] = jnp.where(
        selected,
        _distance_from_interval(jnp.imag(normal_coefficients), exact_normal),
        0.0,
    )
    exact_trace_magnitudes: Float64[Array, " n"] = jnp.where(
        selected,
        jnp.maximum(jnp.abs(trace_interval[0]), jnp.abs(trace_interval[1])),
        0.0,
    )
    exact_normal_magnitudes: Float64[Array, " n"] = jnp.where(
        selected,
        jnp.maximum(jnp.abs(exact_normal[0]), jnp.abs(exact_normal[1])),
        0.0,
    )
    evidence: Tuple[
        Complex128[Array, " n"],
        Complex128[Array, " n"],
        Float64[Array, " n"],
        Float64[Array, " n"],
        Float64[Array, " n"],
        Float64[Array, " n"],
    ] = (
        trace_coefficients,
        normal_coefficients,
        trace_errors,
        normal_errors,
        exact_trace_magnitudes,
        exact_normal_magnitudes,
    )
    return evidence


def _current_operator_error_upper(
    trace_error: Float64[Array, ""],
    normal_error: Float64[Array, ""],
    exact_trace_norm: Float64[Array, ""],
    exact_normal_norm: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the outward LVT.55a5 operator bound.

    Parameters
    ----------
    trace_error : Float64[Array, ""]
        Trace-operator error upper bound.
    normal_error : Float64[Array, ""]
        Normal-operator error upper bound.
    exact_trace_norm : Float64[Array, ""]
        Exact trace-operator norm upper bound.
    exact_normal_norm : Float64[Array, ""]
        Exact normal-operator norm upper bound.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward current-operator error upper bound.
    """
    first: RealInterval = interval_multiply(
        point_interval(trace_error), point_interval(exact_normal_norm)
    )
    trace_plus_error: RealInterval = interval_add(
        point_interval(exact_trace_norm), point_interval(trace_error)
    )
    second: RealInterval = interval_multiply(
        trace_plus_error, point_interval(normal_error)
    )
    upper: Float64[Array, ""] = interval_add(first, second)[1]
    return upper


def _number_current_scale_enclosure(
    target: GalerkinTargetManifest,
) -> Tuple[Float64[Array, ""], RealInterval, Float64[Array, ""]]:
    """PRIVATE: Enclose SC.35c in square Angstroms per second.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target supplying the accelerating voltage.

    Returns
    -------
    stored : Float64[Array, ""]
        Rounded SC.35c number-current scale.
    exact_interval : RealInterval
        Outward exact-real enclosure of the SC.35c scale.
    error : Float64[Array, ""]
        Outward distance of ``stored`` from ``exact_interval``.
    """
    one_thousand: Float64[Array, ""] = jnp.asarray(1000.0, dtype=jnp.float64)
    angstrom_squared_per_square_metre: Float64[Array, ""] = jnp.asarray(
        1.0e20, dtype=jnp.float64
    )
    mass: Float64[Array, ""] = jnp.asarray(M_E, dtype=jnp.float64)
    charge: Float64[Array, ""] = jnp.asarray(E_CHARGE, dtype=jnp.float64)
    speed: Float64[Array, ""] = jnp.asarray(C_LIGHT, dtype=jnp.float64)
    hbar: Float64[Array, ""] = jnp.asarray(HBAR, dtype=jnp.float64)
    voltage_kv: Float64[Array, ""] = target.accelerating_voltage_kv
    voltage_volts: Float64[Array, ""] = voltage_kv * one_thousand
    speed_squared: Float64[Array, ""] = speed * speed
    denominator: Float64[Array, ""] = (
        mass * speed_squared + charge * voltage_volts
    )
    stored: Float64[Array, ""] = (
        hbar * speed_squared / denominator * angstrom_squared_per_square_metre
    )

    speed_squared_interval: RealInterval = interval_square(
        point_interval(speed)
    )
    voltage_interval: RealInterval = interval_multiply(
        point_interval(voltage_kv), point_interval(one_thousand)
    )
    rest_energy: RealInterval = interval_multiply(
        point_interval(mass), speed_squared_interval
    )
    kinetic_energy: RealInterval = interval_multiply(
        point_interval(charge), voltage_interval
    )
    exact_denominator: RealInterval = interval_add(rest_energy, kinetic_energy)
    exact_numerator: RealInterval = interval_multiply(
        point_interval(hbar), speed_squared_interval
    )
    si_interval: RealInterval = interval_divide_positive(
        exact_numerator, exact_denominator
    )
    exact_interval: RealInterval = interval_multiply(
        si_interval, point_interval(angstrom_squared_per_square_metre)
    )
    error: Float64[Array, ""] = _distance_from_interval(stored, exact_interval)
    enclosure: Tuple[Float64[Array, ""], RealInterval, Float64[Array, ""]] = (
        stored,
        exact_interval,
        error,
    )
    return enclosure


def _authenticated_current_diagnostic(
    diagnostic: GalerkinCoordinateCauchyCurrent,
) -> GalerkinCoordinateCauchyCurrent:
    """PRIVATE: Replay and authenticate a public current-diagnostic record.

    Parameters
    ----------
    diagnostic : GalerkinCoordinateCauchyCurrent
        Submitted provisional coordinate-current diagnostic.

    Returns
    -------
    authenticated : GalerkinCoordinateCauchyCurrent
        Canonically replayed diagnostic with authenticated stored values.

    Raises
    ------
    ValueError
        If the nested target fails canonical reconstruction.
    """
    if not _manifest_is_canonical(diagnostic.target):
        raise ValueError(
            "current diagnostic target failed canonical reconstruction"
        )
    canonical: GalerkinCoordinateCauchyCurrent = (
        enclose_galerkin_terminal_current(
            diagnostic.target, diagnostic.submitted_field
        )
    )
    same_payload: Bool[Array, ""] = jnp.asarray(
        eqx.tree_equal(diagnostic, canonical, typematch=True)
    )
    checked_current: Float64[Array, ""] = eqx.error_if(
        canonical.reduced_current,
        ~same_payload,
        "current diagnostic failed canonical replay authentication",
    )
    authenticated: GalerkinCoordinateCauchyCurrent = eqx.tree_at(
        lambda record: record.reduced_current,
        canonical,
        checked_current,
    )
    return authenticated


def _frozen_current_action_interval(
    certificate: GalerkinCurrentOperatorCertificate,
    field: Complex128[Array, " n"],
) -> _ComplexInterval:
    """PRIVATE: Enclose the exact-real frozen dyadic F action.

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Authenticated frozen terminal-current operator evidence.
    field : Complex128[Array, " n"]
        Exact stored retained-state coefficients.

    Returns
    -------
    selected_action : _ComplexInterval
        Componentwise exact-real frozen-action rectangles.
    """
    target: GalerkinTargetManifest = certificate.diagnostic.target
    rows, selected = _terminal_row_map(target)
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
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
        """Accumulate one frozen trace and normal contribution."""
        row: scalar_int = rows[index]
        included: Bool[Array, ""] = selected[index]
        value: _ComplexInterval = _complex_point_interval(field[index])
        trace_coefficient: _ComplexInterval = _complex_point_interval(
            certificate.trace_frozen_coefficients[index]
        )
        normal_coefficient: _ComplexInterval = _complex_point_interval(
            certificate.normal_frozen_coefficients[index]
        )
        trace_value: _ComplexInterval = _complex_interval_multiply(
            trace_coefficient, value
        )
        normal_value: _ComplexInterval = _complex_interval_multiply(
            normal_coefficient, value
        )
        trace_value = (
            jnp.where(included, trace_value[0], 0.0),
            jnp.where(included, trace_value[1], 0.0),
            jnp.where(included, trace_value[2], 0.0),
            jnp.where(included, trace_value[3], 0.0),
        )
        normal_value = (
            jnp.where(included, normal_value[0], 0.0),
            jnp.where(included, normal_value[1], 0.0),
            jnp.where(included, normal_value[2], 0.0),
            jnp.where(included, normal_value[3], 0.0),
        )
        prior_trace: _ComplexInterval = (
            accumulator[0][0][row],
            accumulator[0][1][row],
            accumulator[0][2][row],
            accumulator[0][3][row],
        )
        prior_normal: _ComplexInterval = (
            accumulator[1][0][row],
            accumulator[1][1][row],
            accumulator[1][2][row],
            accumulator[1][3][row],
        )
        updated_trace: _ComplexInterval = _complex_interval_add(
            prior_trace, trace_value
        )
        updated_normal: _ComplexInterval = _complex_interval_add(
            prior_normal, normal_value
        )
        result: Tuple[_ComplexInterval, _ComplexInterval] = (
            tuple(
                component.at[row].set(updated)
                for component, updated in zip(
                    accumulator[0], updated_trace, strict=True
                )
            ),
            tuple(
                component.at[row].set(updated)
                for component, updated in zip(
                    accumulator[1], updated_normal, strict=True
                )
            ),
        )
        return result

    trace_sum: _ComplexInterval
    normal_sum: _ComplexInterval
    trace_sum, normal_sum = lax.fori_loop(
        0, field.shape[0], add_state, initial
    )
    trace_coefficients: _ComplexInterval = _complex_point_interval(
        certificate.trace_frozen_coefficients
    )
    normal_coefficients: _ComplexInterval = _complex_point_interval(
        certificate.normal_frozen_coefficients
    )
    gathered_trace: _ComplexInterval = tuple(
        component[rows] for component in trace_sum
    )
    gathered_normal: _ComplexInterval = tuple(
        component[rows] for component in normal_sum
    )
    trace_normal: _ComplexInterval = _complex_interval_multiply(
        _complex_interval_conjugate(trace_coefficients), gathered_normal
    )
    normal_trace: _ComplexInterval = _complex_interval_multiply(
        _complex_interval_conjugate(normal_coefficients), gathered_trace
    )
    negative_normal_trace: _ComplexInterval = (
        -normal_trace[1],
        -normal_trace[0],
        -normal_trace[3],
        -normal_trace[2],
    )
    difference: _ComplexInterval = _complex_interval_add(
        trace_normal, negative_normal_trace
    )
    action: _ComplexInterval = _complex_interval_multiply(
        difference,
        _complex_point_interval(jnp.asarray(-0.5j, dtype=jnp.complex128)),
    )
    selected_action: _ComplexInterval = (
        jnp.where(selected, action[0], 0.0),
        jnp.where(selected, action[1], 0.0),
        jnp.where(selected, action[2], 0.0),
        jnp.where(selected, action[3], 0.0),
    )
    return selected_action


def _complex_rectangle_error_bounds(
    rounded: Complex128[Array, " n"],
    exact: _ComplexInterval,
) -> Float64[Array, " n"]:
    """PRIVATE: Bound point-to-rectangle componentwise distances.

    Parameters
    ----------
    rounded : Complex128[Array, " n"]
        Rounded complex vector.
    exact : _ComplexInterval
        Componentwise exact-real complex rectangles.

    Returns
    -------
    upper : Float64[Array, " n"]
        Per-component outward Euclidean distance upper bounds.
    """
    real_distance: Float64[Array, " n"] = _distance_from_interval(
        jnp.real(rounded), (exact[0], exact[1])
    )
    imag_distance: Float64[Array, " n"] = _distance_from_interval(
        jnp.imag(rounded), (exact[2], exact[3])
    )
    squared: RealInterval = interval_add(
        interval_square(point_interval(real_distance)),
        interval_square(point_interval(imag_distance)),
    )
    upper: Float64[Array, " n"] = interval_sqrt(squared)[1]
    return upper


def _vector_l2_upper(
    component_magnitude_upper_bounds: Float64[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Bound one vector Euclidean norm outwardly.

    Parameters
    ----------
    component_magnitude_upper_bounds : Float64[Array, " n"]
        Per-component nonnegative magnitude upper bounds.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward Euclidean norm upper bound.
    """
    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)

    def add_component(
        index: scalar_int,
        accumulator: RealInterval,
    ) -> RealInterval:
        """Accumulate one outward squared component bound."""
        squared: RealInterval = interval_square(
            point_interval(component_magnitude_upper_bounds[index])
        )
        updated: RealInterval = interval_add(accumulator, squared)
        return updated

    squared_norm: RealInterval = lax.fori_loop(
        0,
        component_magnitude_upper_bounds.shape[0],
        add_component,
        (zero, zero),
    )
    upper: Float64[Array, ""] = interval_sqrt(squared_norm)[1]
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
        Rounded ``Re(<u,F u>)`` in inverse-square Angstroms.

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


@jaxtyped(typechecker=beartype)
def prepare_galerkin_terminal_current_diagnostic(
    diagnostic: GalerkinCoordinateCauchyCurrent,
) -> GalerkinCoordinateCauchyCurrent:
    """Host-authenticate one provisional coordinate-current diagnostic.

    :see: :class:`~.test_terminal.TestCurrentOperatorCertificate`

    Parameters
    ----------
    diagnostic : GalerkinCoordinateCauchyCurrent
        Provisional transform-compatible diagnostic to authenticate.

    Returns
    -------
    authenticated : GalerkinCoordinateCauchyCurrent
        Canonically reconstructed target and exact-replayed diagnostic.

    Raises
    ------
    ValueError
        If the nested target fails canonical host reconstruction.
    equinox.EquinoxRuntimeError
        If any diagnostic payload differs from canonical replay.

    Notes
    -----
    :func:`enclose_galerkin_terminal_current` remains JIT-compatible and
    produces provisional evidence.  This explicit non-JIT preparation step
    establishes the target and record trust boundary required by every
    stronger scientific consumer.
    """
    authenticated: GalerkinCoordinateCauchyCurrent = (
        _authenticated_current_diagnostic(diagnostic)
    )
    return authenticated


@jaxtyped(typechecker=beartype)
def certify_galerkin_terminal_current_operator(
    diagnostic: GalerkinCoordinateCauchyCurrent,
) -> GalerkinCurrentOperatorCertificate:
    """Certify the uniform frozen selected-sector current operator.

    :see: :class:`~.test_terminal.TestCurrentOperatorCertificate`

    Parameters
    ----------
    diagnostic : GalerkinCoordinateCauchyCurrent
        Provisional weaker submitted-state diagnostic.  This function first
        host-authenticates its target and payload, which then fix the selected
        fiber scope, axis, side, and ``xi=0`` plane.

    Returns
    -------
    certificate : GalerkinCurrentOperatorCertificate
        Uniform LVT.55a4--LVT.55a5 frozen-operator evidence and enclosed
        SC.35c number-current normalization.

    Raises
    ------
    ValueError
        If the nested target fails host-side canonical reconstruction.
    equinox.EquinoxRuntimeError
        If the public diagnostic payload fails canonical replay.

    Notes
    -----
    The returned public carrier is storage, not proof by possession.  Every
    scientific consumer must replay this host-side producer and compare the
    complete payload before using it.  Canonical target reconstruction is a
    deliberate non-JIT authentication boundary.  This first bounded route is
    restricted to the acquisition-selected ``K_d`` fibers at ``xi=0`` and
    makes no vacuum or detector claim.
    """
    authenticated_diagnostic: GalerkinCoordinateCauchyCurrent = (
        prepare_galerkin_terminal_current_diagnostic(diagnostic)
    )
    target: GalerkinTargetManifest = authenticated_diagnostic.target
    (
        trace_coefficients,
        normal_coefficients,
        trace_coefficient_errors,
        normal_coefficient_errors,
        exact_trace_magnitudes,
        exact_normal_magnitudes,
    ) = _frozen_terminal_coefficient_evidence(target)
    exact_trace_norm: Float64[Array, ""] = _fiber_operator_norm_upper(
        target, exact_trace_magnitudes
    )
    exact_normal_norm: Float64[Array, ""] = _fiber_operator_norm_upper(
        target, exact_normal_magnitudes
    )
    trace_error: Float64[Array, ""] = _fiber_operator_norm_upper(
        target, trace_coefficient_errors
    )
    normal_error: Float64[Array, ""] = _fiber_operator_norm_upper(
        target, normal_coefficient_errors
    )
    current_error: Float64[Array, ""] = _current_operator_error_upper(
        trace_error,
        normal_error,
        exact_trace_norm,
        exact_normal_norm,
    )
    current_scale: Float64[Array, ""]
    exact_current_scale: RealInterval
    current_scale_error: Float64[Array, ""]
    current_scale, exact_current_scale, current_scale_error = (
        _number_current_scale_enclosure(target)
    )
    probes = arithmetic_environment_probes()
    arithmetic_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    gradual_supported: Bool[Array, ""] = probes[-1]
    finite_operator_evidence: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(trace_coefficients))
        & jnp.all(jnp.isfinite(normal_coefficients))
        & jnp.all(jnp.isfinite(trace_coefficient_errors))
        & jnp.all(jnp.isfinite(normal_coefficient_errors))
        & jnp.isfinite(exact_trace_norm)
        & jnp.isfinite(exact_normal_norm)
        & jnp.isfinite(trace_error)
        & jnp.isfinite(normal_error)
        & jnp.isfinite(current_error)
        & jnp.all(trace_coefficient_errors >= 0.0)
        & jnp.all(normal_coefficient_errors >= 0.0)
        & (exact_trace_norm >= 0.0)
        & (exact_normal_norm >= 0.0)
        & (trace_error >= 0.0)
        & (normal_error >= 0.0)
        & (current_error >= 0.0)
    )
    normalization_enclosed: Bool[Array, ""] = (
        jnp.isfinite(current_scale)
        & jnp.isfinite(exact_current_scale[0])
        & jnp.isfinite(exact_current_scale[1])
        & jnp.isfinite(current_scale_error)
        & (current_scale > 0.0)
        & (exact_current_scale[0] > 0.0)
        & (exact_current_scale[0] <= exact_current_scale[1])
        & (
            current_scale_error
            >= _distance_from_interval(current_scale, exact_current_scale)
        )
    )
    zero: Int64[Array, ""] = jnp.asarray(
        int(GalerkinCurrentOperatorFailure.NONE), dtype=jnp.int64
    )
    failure_mask: Int64[Array, ""] = zero
    for passed, reason in (
        (
            authenticated_diagnostic.current_diagnostic_eligible,
            GalerkinCurrentOperatorFailure.CURRENT_DIAGNOSTIC_INELIGIBLE,
        ),
        (
            target.fixed_linear_error_ledger.finite_certificate,
            GalerkinCurrentOperatorFailure.FIXED_LINEAR_CERTIFICATE_INELIGIBLE,
        ),
        (
            arithmetic_supported,
            GalerkinCurrentOperatorFailure.ARITHMETIC_ENVIRONMENT_UNSUPPORTED,
        ),
        (
            finite_operator_evidence,
            GalerkinCurrentOperatorFailure.NONFINITE_OPERATOR_EVIDENCE,
        ),
        (
            normalization_enclosed,
            GalerkinCurrentOperatorFailure.CURRENT_NORMALIZATION_UNENCLOSED,
        ),
    ):
        failure_mask = jnp.bitwise_or(
            failure_mask,
            jnp.where(passed, zero, int(reason)),
        )
    stopped = jax.tree.map(
        jax.lax.stop_gradient,
        (
            trace_coefficients,
            normal_coefficients,
            trace_coefficient_errors,
            normal_coefficient_errors,
            exact_trace_norm,
            exact_normal_norm,
            trace_error,
            normal_error,
            current_error,
            current_scale,
            exact_current_scale[0],
            exact_current_scale[1],
            current_scale_error,
            arithmetic_supported,
            gradual_supported,
            failure_mask,
        ),
    )
    certificate: GalerkinCurrentOperatorCertificate = (
        create_galerkin_current_operator_certificate(
            diagnostic=authenticated_diagnostic,
            trace_frozen_coefficients=stopped[0],
            normal_frozen_coefficients=stopped[1],
            trace_coefficient_error_bounds=stopped[2],
            normal_coefficient_error_bounds=stopped[3],
            exact_trace_operator_norm_upper_bound=stopped[4],
            exact_normal_operator_norm_upper_bound=stopped[5],
            trace_operator_error_upper_bound=stopped[6],
            normal_operator_error_upper_bound=stopped[7],
            current_operator_error_upper_bound=stopped[8],
            number_current_scale=stopped[9],
            exact_number_current_scale_lower_bound=stopped[10],
            exact_number_current_scale_upper_bound=stopped[11],
            number_current_scale_error_upper_bound=stopped[12],
            terminal_plane_coordinate=jnp.asarray(0.0, dtype=jnp.float64),
            arithmetic_environment_supported=stopped[13],
            gradual_underflow_supported=stopped[14],
            current_operator_failure_mask=stopped[15],
            current_scope=authenticated_diagnostic.current_scope,
            route=_ROUTE,
            coefficient_metrics=_COEFFICIENT_METRICS,
            fixed_linear_target=_FIXED_LINEAR_TARGET,
            per_call_action_route=_PER_CALL_ACTION_ROUTE,
            current_normalization=_CURRENT_NORMALIZATION,
            eligibility_scope=_OPERATOR_ELIGIBILITY_SCOPE,
        )
    )
    return certificate


def _authenticated_current_operator_certificate(
    certificate: GalerkinCurrentOperatorCertificate,
) -> GalerkinCurrentOperatorCertificate:
    """PRIVATE: Replay and authenticate a public operator certificate.

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Submitted provisional frozen current-operator certificate.

    Returns
    -------
    authenticated : GalerkinCurrentOperatorCertificate
        Canonically replayed certificate with authenticated stored values.
    """
    canonical: GalerkinCurrentOperatorCertificate = (
        certify_galerkin_terminal_current_operator(certificate.diagnostic)
    )
    same_payload: Bool[Array, ""] = jnp.asarray(
        eqx.tree_equal(certificate, canonical, typematch=True)
    )
    checked_error: Float64[Array, ""] = eqx.error_if(
        canonical.current_operator_error_upper_bound,
        ~same_payload,
        "current-operator certificate failed canonical replay authentication",
    )
    authenticated: GalerkinCurrentOperatorCertificate = eqx.tree_at(
        lambda record: record.current_operator_error_upper_bound,
        canonical,
        checked_error,
    )
    return authenticated


@jaxtyped(typechecker=beartype)
def enclose_galerkin_terminal_current_action(
    certificate: GalerkinCurrentOperatorCertificate,
    field: Complex[Array, "..."],
) -> GalerkinTerminalCurrentActionEnclosure:
    """Enclose one frozen current action after certificate authentication.

    :see: :class:`~.test_terminal.TestCurrentOperatorCertificate`

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Public storage record, authenticated by complete canonical replay
        before any field action is evaluated.
    field : Complex[Array, "..."]
        Exact stored binary64 retained-state coefficients.

    Returns
    -------
    enclosure : GalerkinTerminalCurrentActionEnclosure
        Rounded public action, exact-real frozen-action rectangles, and an
        outward Euclidean action-error bound.

    Raises
    ------
    ValueError
        If canonical target reconstruction fails or the state rank or length
        is invalid.
    equinox.EquinoxRuntimeError
        If the certificate is forged or the submitted field is nonfinite.

    Notes
    -----
    The action enclosure accounts only for arithmetic around the frozen
    dyadic matrix.  The separate uniform frozen-to-exact target difference
    remains in ``certificate.current_operator_error_upper_bound``.  Complete
    public-certificate replay includes host-side target reconstruction, so
    this authenticated evidence entry point is deliberately not JIT-able.
    Scientific use requires both ``certificate.current_operator_eligible``
    and this call's ``finite_certificate``; neither predicate substitutes for
    the other.
    """
    authenticated: GalerkinCurrentOperatorCertificate = (
        _authenticated_current_operator_certificate(certificate)
    )
    target: GalerkinTargetManifest = authenticated.diagnostic.target
    checked_field: Complex128[Array, " n"] = _checked_state(
        target, field, "field"
    )
    production_action: Complex128[Array, " n"] = _raw_current_action(
        target, checked_field
    )
    exact_action: _ComplexInterval = _frozen_current_action_interval(
        authenticated, checked_field
    )
    component_errors: Float64[Array, " n"] = _complex_rectangle_error_bounds(
        production_action, exact_action
    )
    action_error: Float64[Array, ""] = _vector_l2_upper(component_errors)
    probes = arithmetic_environment_probes()
    arithmetic_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    gradual_supported: Bool[Array, ""] = probes[-1]
    finite_rectangles: Bool[Array, ""] = jnp.asarray(True)
    for component in exact_action:
        finite_rectangles = finite_rectangles & jnp.all(
            jnp.isfinite(component)
        )
    finite_evidence: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(production_action))
        & finite_rectangles
        & jnp.all(jnp.isfinite(component_errors))
        & jnp.isfinite(action_error)
        & jnp.all(component_errors >= 0.0)
        & (action_error >= 0.0)
    )
    zero: Int64[Array, ""] = jnp.asarray(
        int(GalerkinTerminalCurrentActionFailure.NONE), dtype=jnp.int64
    )
    failure_mask: Int64[Array, ""] = zero
    for passed, reason in (
        (
            authenticated.current_operator_eligible,
            GalerkinTerminalCurrentActionFailure.OPERATOR_INELIGIBLE,
        ),
        (
            arithmetic_supported,
            GalerkinTerminalCurrentActionFailure.ARITHMETIC_ENVIRONMENT_UNSUPPORTED,
        ),
        (
            finite_evidence,
            GalerkinTerminalCurrentActionFailure.NONFINITE_ACTION_EVIDENCE,
        ),
    ):
        failure_mask = jnp.bitwise_or(
            failure_mask,
            jnp.where(passed, zero, int(reason)),
        )
    stopped = jax.tree.map(
        jax.lax.stop_gradient,
        (
            checked_field,
            production_action,
            exact_action,
            component_errors,
            action_error,
            arithmetic_supported,
            gradual_supported,
            failure_mask,
        ),
    )
    enclosure: GalerkinTerminalCurrentActionEnclosure = (
        create_galerkin_terminal_current_action_enclosure(
            certificate=authenticated,
            submitted_field=stopped[0],
            production_action=stopped[1],
            algebraic_action_real_lower_bounds=stopped[2][0],
            algebraic_action_real_upper_bounds=stopped[2][1],
            algebraic_action_imag_lower_bounds=stopped[2][2],
            algebraic_action_imag_upper_bounds=stopped[2][3],
            component_error_bounds=stopped[3],
            action_error_bound=stopped[4],
            arithmetic_environment_supported=stopped[5],
            gradual_underflow_supported=stopped[6],
            failure_mask=stopped[7],
            route=_ROUTE,
            exact_action_target=_ACTION_TARGET,
            coefficient_norm="Euclidean complex state-coefficient l2 norm",
            error_scope=_ACTION_ERROR_SCOPE,
        )
    )
    return enclosure


__all__: list[str] = [
    "apply_galerkin_terminal_current",
    "apply_galerkin_terminal_normal_derivative",
    "apply_galerkin_terminal_normal_derivative_adjoint",
    "apply_galerkin_terminal_trace",
    "apply_galerkin_terminal_trace_adjoint",
    "certify_galerkin_terminal_current_operator",
    "enclose_galerkin_terminal_current",
    "enclose_galerkin_terminal_current_action",
    "evaluate_galerkin_terminal_current",
    "prepare_galerkin_terminal_current_diagnostic",
]
