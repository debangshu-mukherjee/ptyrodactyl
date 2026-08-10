r"""Define bounded coordinate-terminal current evidence.

Extended Summary
----------------
This module owns the typed RM-S4a selected-sector coordinate Cauchy/current
diagnostic.  The carrier binds one submitted state to its canonical scalar
target, acquisition-selected transverse fibers, coordinate axis, oriented
side, matrix-free field and normal-derivative traces, Hermitian current
action, and an FTZ-safe exact-carrier current enclosure.  It keeps this
per-state selected-sector claim explicitly separate from total full-plane
current, physical vacuum branch extraction, and detector eligibility.

Routine Listings
----------------
:class:`GalerkinCoordinateCauchyCurrent`
    Store one bounded selected-sector coordinate-current diagnostic.
:class:`GalerkinDetectorFailure`
    Store the unavailable detector-contract reasons.
:class:`GalerkinTerminalCurrentFailure`
    Enumerate fail-closed coordinate-current predicate bits.
:class:`GalerkinTerminalCurrentRoute`
    Store the coordinate-current enclosure route.
:class:`GalerkinTerminalCurrentScope`
    Store the factory-owned selected-fiber current scope.
:class:`GalerkinVacuumBranchFailure`
    Store the unavailable physical vacuum-branch reason.
:func:`create_galerkin_coordinate_cauchy_current`
    Create a validated bounded selected-sector current diagnostic.

Notes
-----
The bounded evidence certifies an exact submitted-state current only over
complete retained normal-frequency fibers whose transverse indices belong to
the acquisition's selected ``K_d`` set.  It excludes every other transverse
fiber in ``K_u`` and therefore is not automatically a total full-plane
current.  It also does not certify the rounded current action as an exact
operator and is not an outgoing-wave map.
"""

from enum import Enum, IntFlag

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from .acquisition_types import GalerkinTerminalSide
from .galerkin_types import GalerkinTargetManifest


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

    Parameters
    ----------
    condition : bool
        Structural failure predicate.
    message : str
        Exception message for the rejected contract.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


class GalerkinDetectorFailure(IntFlag):
    """Store the unavailable detector-contract reasons.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Attributes
    ----------
    NONE : int
        No detector-contract predicate failed.
    NO_VACUUM_BRANCH : int
        No certified physical vacuum branch feeds a detector entrance.
    NO_OUTGOING_EXTRACTION : int
        No forward/backward Cauchy branch separation is certified.
    NO_PIXEL_RESPONSE : int
        No positive pixel operator or calibrated response is bound.
    """

    NONE = 0
    NO_VACUUM_BRANCH = 1 << 0
    NO_OUTGOING_EXTRACTION = 1 << 1
    NO_PIXEL_RESPONSE = 1 << 2


class GalerkinTerminalCurrentFailure(IntFlag):
    """Enumerate fail-closed coordinate-current predicate bits.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Attributes
    ----------
    NONE : int
        Every implemented coordinate-current predicate passed.
    SUPPORT_INELIGIBLE : int
        The bound acquisition is not RM-S1 support eligible.
    TERMINAL_FIBER_INCOMPLETE : int
        The selected coordinate terminal lacks a complete retained fiber.
    ARITHMETIC_ENVIRONMENT_UNSUPPORTED : int
        A load-bearing normal binary64 arithmetic probe failed.
    NONFINITE_CURRENT_EVIDENCE : int
        The rounded action or exact-current enclosure is nonfinite.
    """

    NONE = 0
    SUPPORT_INELIGIBLE = 1 << 0
    TERMINAL_FIBER_INCOMPLETE = 1 << 1
    ARITHMETIC_ENVIRONMENT_UNSUPPORTED = 1 << 2
    NONFINITE_CURRENT_EVIDENCE = 1 << 3


class GalerkinTerminalCurrentRoute(str, Enum):
    """Store the coordinate-current enclosure route.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Attributes
    ----------
    FTZ_SAFE_EXACT_CARRIER_CAUCHY : str
        Enclose the exact normalized-carrier Cauchy current with shared
        FTZ-safe outward binary64 intervals.
    """

    FTZ_SAFE_EXACT_CARRIER_CAUCHY = (
        "rm_s4a_ftz_safe_exact_carrier_coordinate_cauchy"
    )


class GalerkinTerminalCurrentScope(str, Enum):
    """Store the factory-owned selected-fiber current scope.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Attributes
    ----------
    SELECTED_ACQUISITION_FIBER_SECTOR : str
        Sum only complete retained normal-frequency fibers selected by the
        acquisition ``K_d`` transverse-index set.
    """

    SELECTED_ACQUISITION_FIBER_SECTOR = "selected_acquisition_kd_fiber_sector"


class GalerkinVacuumBranchFailure(str, Enum):
    """Store the unavailable physical vacuum-branch reason.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Attributes
    ----------
    NO_COMPACT_LOCAL_VACUUM_SLAB_CONTRACT : str
        The trigonometric target has no exact compact-local source-free slab
        contract on which physical vacuum branches can be separated.
    """

    NO_COMPACT_LOCAL_VACUUM_SLAB_CONTRACT = (
        "no_compact_local_vacuum_slab_contract"
    )


class GalerkinCoordinateCauchyCurrent(eqx.Module):
    """Store one bounded selected-sector coordinate-current diagnostic.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Attributes
    ----------
    target : GalerkinTargetManifest
        Canonical scalar target that owns the state and terminal geometry.
    submitted_field : Complex128[Array, " n"]
        Exact stored binary64 state used by this diagnostic.
    trace_coefficients : Complex128[Array, " t"]
        Matrix-free coordinate field trace ``T u`` in transverse-plane
        orthonormal coefficients.
    normal_derivative_coefficients : Complex128[Array, " t"]
        Oriented physical normal-derivative trace ``N u``.
    current_action : Complex128[Array, " n"]
        Hermitian reduced-current action ``F u`` with
        ``F=(T* N-N* T)/(2i)``.
    reduced_current : Float64[Array, ""]
        Rounded quadratic diagnostic ``Re(<u,F u>)``.
    exact_reduced_current_lower_bound : Float64[Array, ""]
        Inclusive lower endpoint for the exact normalized-carrier current.
    exact_reduced_current_upper_bound : Float64[Array, ""]
        Inclusive upper endpoint for the exact normalized-carrier current.
    reduced_current_error_upper_bound : Float64[Array, ""]
        Outward distance bound from the rounded current to the exact interval.
    arithmetic_environment_supported : Bool[Array, ""]
        Whether every load-bearing normal arithmetic probe passed.
    gradual_underflow_supported : Bool[Array, ""]
        Diagnostic gradual-underflow probe result.
    current_diagnostic_eligible : Bool[Array, ""]
        Whether this submitted-state reduced-current diagnostic is eligible.
    current_diagnostic_failure_mask : Int64[Array, ""]
        Bitwise :class:`GalerkinTerminalCurrentFailure` payload.
    vacuum_branch_eligible : Bool[Array, ""]
        Always false for this trigonometric coordinate-current route.
    detector_eligible : Bool[Array, ""]
        Always false because no outgoing branch or pixel map is bound.
    terminal_axis : int
        Static coordinate normal axis copied from the target acquisition.
    terminal_side : GalerkinTerminalSide
        Static oriented coordinate face copied from the acquisition.
    current_scope : GalerkinTerminalCurrentScope
        Factory-owned identity of the selected ``K_d``-fiber sector.
    route : GalerkinTerminalCurrentRoute
        Static exact-current enclosure route.
    vacuum_branch_failure : GalerkinVacuumBranchFailure
        Static reason why a physical vacuum branch is unavailable.
    detector_failure : GalerkinDetectorFailure
        Static detector-contract failure flags.
    coefficient_metrics : str
        Static state/trace inner-product declaration.
    current_target : str
        Static exact-real current target declaration.
    eligibility_scope : str
        Static boundary of the submitted-state diagnostic eligibility claim.

    Notes
    -----
    ``current_diagnostic_eligible`` certifies only the submitted-state scalar
    enclosure over acquisition-selected ``K_d`` fibers.  It does not certify
    equality with a total full-plane current or a uniform operator/action
    error bound for the rounded ``current_action``.  The two separately named
    downstream eligibility fields are deliberately false.
    """

    target: GalerkinTargetManifest
    submitted_field: Complex128[Array, " n"]
    trace_coefficients: Complex128[Array, " t"]
    normal_derivative_coefficients: Complex128[Array, " t"]
    current_action: Complex128[Array, " n"]
    reduced_current: Float64[Array, ""]
    exact_reduced_current_lower_bound: Float64[Array, ""]
    exact_reduced_current_upper_bound: Float64[Array, ""]
    reduced_current_error_upper_bound: Float64[Array, ""]
    arithmetic_environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    current_diagnostic_eligible: Bool[Array, ""]
    current_diagnostic_failure_mask: Int64[Array, ""]
    vacuum_branch_eligible: Bool[Array, ""]
    detector_eligible: Bool[Array, ""]
    terminal_axis: int = eqx.field(static=True)
    terminal_side: GalerkinTerminalSide = eqx.field(static=True)
    current_scope: GalerkinTerminalCurrentScope = eqx.field(static=True)
    route: GalerkinTerminalCurrentRoute = eqx.field(static=True)
    vacuum_branch_failure: GalerkinVacuumBranchFailure = eqx.field(static=True)
    detector_failure: GalerkinDetectorFailure = eqx.field(static=True)
    coefficient_metrics: str = eqx.field(static=True)
    current_target: str = eqx.field(static=True)
    eligibility_scope: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_coordinate_cauchy_current(  # noqa: PLR0913
    target: GalerkinTargetManifest,
    submitted_field: Complex[Array, "..."],
    trace_coefficients: Complex[Array, "..."],
    normal_derivative_coefficients: Complex[Array, "..."],
    current_action: Complex[Array, "..."],
    reduced_current: Float[Array, ""],
    exact_reduced_current_lower_bound: Float[Array, ""],
    exact_reduced_current_upper_bound: Float[Array, ""],
    reduced_current_error_upper_bound: Float[Array, ""],
    arithmetic_environment_supported: Bool[Array, ""],
    gradual_underflow_supported: Bool[Array, ""],
    current_diagnostic_eligible: Bool[Array, ""],
    current_diagnostic_failure_mask: Int[Array, ""],
    vacuum_branch_eligible: Bool[Array, ""],
    detector_eligible: Bool[Array, ""],
    *,
    terminal_axis: int,
    terminal_side: GalerkinTerminalSide,
    route: GalerkinTerminalCurrentRoute,
    vacuum_branch_failure: GalerkinVacuumBranchFailure,
    detector_failure: GalerkinDetectorFailure,
    coefficient_metrics: str,
    current_target: str,
    eligibility_scope: str,
) -> GalerkinCoordinateCauchyCurrent:
    """Create a validated bounded selected-sector current diagnostic.

    :see: :class:`~.test_terminal_types.TestCoordinateTerminalTypes`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target owning the state and terminal geometry.
    submitted_field : Complex[Array, "..."]
        Candidate retained-state coefficient vector.
    trace_coefficients : Complex[Array, "..."]
        Candidate transverse field trace.
    normal_derivative_coefficients : Complex[Array, "..."]
        Candidate oriented transverse normal-derivative trace.
    current_action : Complex[Array, "..."]
        Candidate Hermitian current action.
    reduced_current : Float[Array, ""]
        Candidate rounded reduced-current scalar.
    exact_reduced_current_lower_bound : Float[Array, ""]
        Candidate inclusive exact-current lower endpoint.
    exact_reduced_current_upper_bound : Float[Array, ""]
        Candidate inclusive exact-current upper endpoint.
    reduced_current_error_upper_bound : Float[Array, ""]
        Candidate non-negative rounded-to-exact current error bound.
    arithmetic_environment_supported : Bool[Array, ""]
        Load-bearing arithmetic probe result.
    gradual_underflow_supported : Bool[Array, ""]
        Diagnostic gradual-underflow probe result.
    current_diagnostic_eligible : Bool[Array, ""]
        Candidate submitted-state current-diagnostic eligibility.
    current_diagnostic_failure_mask : Int[Array, ""]
        Candidate current-diagnostic failure bits.
    vacuum_branch_eligible : Bool[Array, ""]
        Must be false for this route.
    detector_eligible : Bool[Array, ""]
        Must be false for this route.
    terminal_axis : int
        Static terminal axis that must equal the acquisition axis.
    terminal_side : GalerkinTerminalSide
        Static side that must equal the acquisition side.
    route : GalerkinTerminalCurrentRoute
        Static exact-current enclosure route.
    vacuum_branch_failure : GalerkinVacuumBranchFailure
        Static physical-vacuum failure reason.
    detector_failure : GalerkinDetectorFailure
        Static detector-contract failure flags.
    coefficient_metrics : str
        Nonempty state/trace metric declaration.
    current_target : str
        Nonempty exact-real current target declaration.
    eligibility_scope : str
        Nonempty claim-boundary declaration.

    Returns
    -------
    diagnostic : GalerkinCoordinateCauchyCurrent
        Structurally validated current evidence.

    Raises
    ------
    ValueError
        If an array rank, vector length, static binding, or declaration is
        invalid.
    equinox.EquinoxRuntimeError
        If the evidence ordering, eligibility logic, or mandatory downstream
        ineligibility fails closed.
    """
    state_size: int = target.support.state_indices.shape[0]
    terminal_size: int = target.acquisition.transverse_indices.shape[0]
    field: Complex128[Array, " n"] = jnp.asarray(
        submitted_field, dtype=jnp.complex128
    )
    trace: Complex128[Array, " t"] = jnp.asarray(
        trace_coefficients, dtype=jnp.complex128
    )
    normal_trace: Complex128[Array, " t"] = jnp.asarray(
        normal_derivative_coefficients, dtype=jnp.complex128
    )
    action: Complex128[Array, " n"] = jnp.asarray(
        current_action, dtype=jnp.complex128
    )
    _raise_if(field.ndim != 1, "submitted_field must be 1D")
    _raise_if(field.shape[0] != state_size, "submitted_field must match K_u")
    for values, name in (
        (trace, "trace_coefficients"),
        (normal_trace, "normal_derivative_coefficients"),
    ):
        _raise_if(values.ndim != 1, f"{name} must be 1D")
        _raise_if(
            values.shape[0] != terminal_size,
            f"{name} must match K_d fibers",
        )
    _raise_if(action.ndim != 1, "current_action must be 1D")
    _raise_if(action.shape[0] != state_size, "current_action must match K_u")
    _raise_if(
        terminal_axis != target.acquisition.terminal_axis,
        "terminal_axis must match target acquisition",
    )
    _raise_if(
        terminal_side is not target.acquisition.terminal_side,
        "terminal_side must match target acquisition",
    )
    for declaration, name in (
        (coefficient_metrics, "coefficient_metrics"),
        (current_target, "current_target"),
        (eligibility_scope, "eligibility_scope"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")

    rounded: Float64[Array, ""] = jnp.asarray(
        reduced_current, dtype=jnp.float64
    )
    lower: Float64[Array, ""] = jnp.asarray(
        exact_reduced_current_lower_bound, dtype=jnp.float64
    )
    upper: Float64[Array, ""] = jnp.asarray(
        exact_reduced_current_upper_bound, dtype=jnp.float64
    )
    error: Float64[Array, ""] = jnp.asarray(
        reduced_current_error_upper_bound, dtype=jnp.float64
    )
    failure_mask: Int64[Array, ""] = jnp.asarray(
        current_diagnostic_failure_mask, dtype=jnp.int64
    )
    eligible: Bool[Array, ""] = jnp.asarray(current_diagnostic_eligible)
    arithmetic_supported: Bool[Array, ""] = jnp.asarray(
        arithmetic_environment_supported
    )
    gradual_supported: Bool[Array, ""] = jnp.asarray(
        gradual_underflow_supported
    )
    vacuum_eligible: Bool[Array, ""] = jnp.asarray(vacuum_branch_eligible)
    detector_is_eligible: Bool[Array, ""] = jnp.asarray(detector_eligible)

    for value, name in (
        (rounded, "reduced_current"),
        (lower, "exact_reduced_current_lower_bound"),
        (upper, "exact_reduced_current_upper_bound"),
        (error, "reduced_current_error_upper_bound"),
        (failure_mask, "current_diagnostic_failure_mask"),
        (eligible, "current_diagnostic_eligible"),
        (arithmetic_supported, "arithmetic_environment_supported"),
        (gradual_supported, "gradual_underflow_supported"),
        (vacuum_eligible, "vacuum_branch_eligible"),
        (detector_is_eligible, "detector_eligible"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")

    finite_payload: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(field))
        & jnp.all(jnp.isfinite(trace))
        & jnp.all(jnp.isfinite(normal_trace))
        & jnp.all(jnp.isfinite(action))
        & jnp.isfinite(rounded)
        & jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & jnp.isfinite(error)
    )
    checked_field: Complex128[Array, " n"] = eqx.error_if(
        field,
        jnp.any(~jnp.isfinite(field)),
        "submitted_field must be finite",
    )
    checked_lower: Float64[Array, ""] = eqx.error_if(
        lower,
        (lower > upper)
        | jnp.isnan(lower)
        | jnp.isnan(upper)
        | jnp.isnan(error)
        | (error < 0.0),
        "current interval must be ordered and its error non-negative",
    )
    allowed_mask: int = int(
        GalerkinTerminalCurrentFailure.SUPPORT_INELIGIBLE
        | GalerkinTerminalCurrentFailure.TERMINAL_FIBER_INCOMPLETE
        | GalerkinTerminalCurrentFailure.ARITHMETIC_ENVIRONMENT_UNSUPPORTED
        | GalerkinTerminalCurrentFailure.NONFINITE_CURRENT_EVIDENCE
    )
    checked_failure: Int64[Array, ""] = eqx.error_if(
        failure_mask,
        (failure_mask < 0)
        | (
            jnp.bitwise_and(
                failure_mask,
                jnp.asarray(~allowed_mask, dtype=jnp.int64),
            )
            != 0
        )
        | (
            eligible
            != (failure_mask == int(GalerkinTerminalCurrentFailure.NONE))
        )
        | (eligible & ~finite_payload)
        | vacuum_eligible
        | detector_is_eligible,
        "eligibility flags must agree with typed current and downstream "
        "failures",
    )
    _raise_if(
        vacuum_branch_failure
        is not (
            GalerkinVacuumBranchFailure.NO_COMPACT_LOCAL_VACUUM_SLAB_CONTRACT
        ),
        "this route requires the compact-local vacuum-slab failure",
    )
    required_detector_failures: GalerkinDetectorFailure = (
        GalerkinDetectorFailure.NO_VACUUM_BRANCH
        | GalerkinDetectorFailure.NO_OUTGOING_EXTRACTION
        | GalerkinDetectorFailure.NO_PIXEL_RESPONSE
    )
    _raise_if(
        detector_failure != required_detector_failures,
        "this route requires every unavailable detector-contract reason",
    )

    diagnostic: GalerkinCoordinateCauchyCurrent = (
        GalerkinCoordinateCauchyCurrent(
            target=target,
            submitted_field=checked_field,
            trace_coefficients=trace,
            normal_derivative_coefficients=normal_trace,
            current_action=action,
            reduced_current=rounded,
            exact_reduced_current_lower_bound=checked_lower,
            exact_reduced_current_upper_bound=upper,
            reduced_current_error_upper_bound=error,
            arithmetic_environment_supported=arithmetic_supported,
            gradual_underflow_supported=gradual_supported,
            current_diagnostic_eligible=eligible,
            current_diagnostic_failure_mask=checked_failure,
            vacuum_branch_eligible=vacuum_eligible,
            detector_eligible=detector_is_eligible,
            terminal_axis=terminal_axis,
            terminal_side=terminal_side,
            current_scope=(
                GalerkinTerminalCurrentScope.SELECTED_ACQUISITION_FIBER_SECTOR
            ),
            route=route,
            vacuum_branch_failure=vacuum_branch_failure,
            detector_failure=detector_failure,
            coefficient_metrics=coefficient_metrics,
            current_target=current_target,
            eligibility_scope=eligibility_scope,
        )
    )
    return diagnostic


__all__: list[str] = [
    "GalerkinCoordinateCauchyCurrent",
    "GalerkinDetectorFailure",
    "GalerkinTerminalCurrentFailure",
    "GalerkinTerminalCurrentRoute",
    "GalerkinTerminalCurrentScope",
    "GalerkinVacuumBranchFailure",
    "create_galerkin_coordinate_cauchy_current",
]
