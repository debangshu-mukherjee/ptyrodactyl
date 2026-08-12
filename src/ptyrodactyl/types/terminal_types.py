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
:class:`GalerkinCurrentOperatorCertificate`
    Store uniform selected-sector LVT.55a operator evidence.
:class:`GalerkinCurrentOperatorFailure`
    Enumerate fail-closed uniform current-operator predicate bits.
:class:`GalerkinDetectorFailure`
    Store the unavailable detector-contract reasons.
:class:`GalerkinTerminalCurrentActionEnclosure`
    Store one per-call frozen-current action enclosure.
:class:`GalerkinTerminalCurrentActionFailure`
    Enumerate fail-closed per-call current-action predicate bits.
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
:func:`create_galerkin_current_operator_certificate`
    Create validated uniform selected-sector current-operator evidence.
:func:`create_galerkin_terminal_current_action_enclosure`
    Create one validated per-call frozen-current action enclosure.

Notes
-----
The transform-compatible diagnostic is provisional until its owning host
preparer reconstructs the nested target and exact-replays its whole payload.
Once authenticated, it certifies an exact submitted-state current only over
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


class GalerkinCurrentOperatorFailure(IntFlag):
    """Enumerate fail-closed uniform current-operator predicate bits.

    :see: :class:`~.test_terminal_types.TestCurrentOperatorTypes`

    Attributes
    ----------
    NONE : int
        Every implemented LVT.55a operator predicate passed.
    CURRENT_DIAGNOSTIC_INELIGIBLE : int
        The nested submitted-state current diagnostic is ineligible.
    FIXED_LINEAR_CERTIFICATE_INELIGIBLE : int
        The parent target lacks finite RM-S2 fixed-linear evidence.
    ARITHMETIC_ENVIRONMENT_UNSUPPORTED : int
        The required normal binary64 arithmetic probes failed.
    NONFINITE_OPERATOR_EVIDENCE : int
        One uniform trace, normal, current, or action-route bound is nonfinite.
    CURRENT_NORMALIZATION_UNENCLOSED : int
        The physical number-current scale lacks a positive finite enclosure.
    """

    NONE = 0
    CURRENT_DIAGNOSTIC_INELIGIBLE = 1 << 0
    FIXED_LINEAR_CERTIFICATE_INELIGIBLE = 1 << 1
    ARITHMETIC_ENVIRONMENT_UNSUPPORTED = 1 << 2
    NONFINITE_OPERATOR_EVIDENCE = 1 << 3
    CURRENT_NORMALIZATION_UNENCLOSED = 1 << 4


class GalerkinTerminalCurrentActionFailure(IntFlag):
    """Enumerate fail-closed per-call current-action predicate bits.

    :see: :class:`~.test_terminal_types.TestCurrentOperatorTypes`

    Attributes
    ----------
    NONE : int
        The submitted action has a finite frozen-matrix enclosure.
    OPERATOR_INELIGIBLE : int
        The parent uniform current-operator certificate is ineligible.
    ARITHMETIC_ENVIRONMENT_UNSUPPORTED : int
        The required normal binary64 arithmetic probes failed.
    NONFINITE_ACTION_EVIDENCE : int
        The rounded action, interval, or norm bound is nonfinite.
    """

    NONE = 0
    OPERATOR_INELIGIBLE = 1 << 0
    ARITHMETIC_ENVIRONMENT_UNSUPPORTED = 1 << 1
    NONFINITE_ACTION_EVIDENCE = 1 << 2


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
        orthonormal coefficients with inverse square-root Angstrom units.
    normal_derivative_coefficients : Complex128[Array, " t"]
        Oriented physical normal-derivative trace ``N u`` with inverse
        three-halves Angstrom units.
    current_action : Complex128[Array, " n"]
        Hermitian reduced-current action ``F u`` with
        ``F=(T* N-N* T)/(2i)`` in inverse-square Angstroms.
    reduced_current : Float64[Array, ""]
        Rounded quadratic diagnostic ``Re(<u,F u>)`` in inverse-square
        Angstroms.
    exact_reduced_current_lower_bound : Float64[Array, ""]
        Inclusive lower endpoint for the exact normalized-carrier current in
        inverse-square Angstroms.
    exact_reduced_current_upper_bound : Float64[Array, ""]
        Inclusive upper endpoint for the exact normalized-carrier current in
        inverse-square Angstroms.
    reduced_current_error_upper_bound : Float64[Array, ""]
        Outward distance bound from the rounded current to the exact interval
        in inverse-square Angstroms.
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
    ``current_diagnostic_eligible`` is provisional transform-compatible
    evidence for the submitted-state scalar enclosure over
    acquisition-selected ``K_d`` fibers.  A scientific consumer must first
    host-reconstruct the nested target and exact-replay the whole diagnostic;
    raw public-carrier possession is non-authoritative.  Even after that
    preparation it does not certify equality with a total full-plane current
    or a uniform operator/action error bound for the rounded
    ``current_action``.  The two separately named downstream eligibility
    fields are deliberately false.
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


class GalerkinCurrentOperatorCertificate(eqx.Module):
    """Store uniform selected-sector LVT.55a operator evidence.

    :see: :class:`~.test_terminal_types.TestCurrentOperatorTypes`

    The carrier is the second, strictly stronger level of the RM-S4 terminal
    ladder.  It nests one accepted submitted-state diagnostic, binds the
    frozen dyadic ``T`` and ``N`` coefficient rows used by the rounded action,
    and stores independent uniform LVT.55a4--LVT.55a5 enclosures.  It also
    encloses the physical ``C_j`` normalization.  It has no vacuum-branch or
    detector eligibility field.  Its public fields are a storage record, not
    an authentication boundary: every scientific consumer must replay the
    canonical producer and reject any payload mismatch before using it.  A
    true ``current_operator_eligible`` stores uniform route capability only;
    each submitted-field result additionally requires that call's action
    enclosure ``finite_certificate``.  Trace coefficients have inverse
    square-root Angstrom units, normal coefficients inverse three-halves
    Angstrom units, and every current-operator error has inverse-square
    Angstrom units.  ``number_current_scale`` has square Angstroms per second,
    so its product with the current quadratic has inverse seconds.
    """

    diagnostic: GalerkinCoordinateCauchyCurrent
    trace_frozen_coefficients: Complex128[Array, " n"]
    normal_frozen_coefficients: Complex128[Array, " n"]
    trace_coefficient_error_bounds: Float64[Array, " n"]
    normal_coefficient_error_bounds: Float64[Array, " n"]
    exact_trace_operator_norm_upper_bound: Float64[Array, ""]
    exact_normal_operator_norm_upper_bound: Float64[Array, ""]
    trace_operator_error_upper_bound: Float64[Array, ""]
    normal_operator_error_upper_bound: Float64[Array, ""]
    current_operator_error_upper_bound: Float64[Array, ""]
    number_current_scale: Float64[Array, ""]
    exact_number_current_scale_lower_bound: Float64[Array, ""]
    exact_number_current_scale_upper_bound: Float64[Array, ""]
    number_current_scale_error_upper_bound: Float64[Array, ""]
    terminal_plane_coordinate: Float64[Array, ""]
    arithmetic_environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    current_operator_eligible: Bool[Array, ""]
    current_operator_failure_mask: Int64[Array, ""]
    current_scope: GalerkinTerminalCurrentScope = eqx.field(static=True)
    route: GalerkinTerminalCurrentRoute = eqx.field(static=True)
    coefficient_metrics: str = eqx.field(static=True)
    fixed_linear_target: str = eqx.field(static=True)
    per_call_action_route: str = eqx.field(static=True)
    current_normalization: str = eqx.field(static=True)
    eligibility_scope: str = eqx.field(static=True)


class GalerkinTerminalCurrentActionEnclosure(eqx.Module):
    """Store one per-call frozen-current action enclosure.

    :see: :class:`~.test_terminal_types.TestCurrentOperatorTypes`

    The component rectangles enclose the exact-real action of the frozen
    dyadic current matrix.  ``action_error_bound`` is the Euclidean distance
    from the rounded public action to that matrix action for this submitted
    field.  It is distinct from the uniform exact-target operator difference
    stored by :class:`GalerkinCurrentOperatorCertificate`.  Public instances
    are storage records: only the owning terminal encloser's canonical replay
    establishes scientific evidence; the exported representation factory is
    not an authentication boundary.

    The action rectangles, component errors, and aggregate error have
    inverse-square Angstrom units for dimensionless normalized state
    coefficients.
    """

    certificate: GalerkinCurrentOperatorCertificate
    submitted_field: Complex128[Array, " n"]
    production_action: Complex128[Array, " n"]
    algebraic_action_real_lower_bounds: Float64[Array, " n"]
    algebraic_action_real_upper_bounds: Float64[Array, " n"]
    algebraic_action_imag_lower_bounds: Float64[Array, " n"]
    algebraic_action_imag_upper_bounds: Float64[Array, " n"]
    component_error_bounds: Float64[Array, " n"]
    action_error_bound: Float64[Array, ""]
    arithmetic_environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    finite_certificate: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    route: GalerkinTerminalCurrentRoute = eqx.field(static=True)
    exact_action_target: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)


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


@jaxtyped(typechecker=beartype)
def create_galerkin_current_operator_certificate(  # noqa: PLR0913,PLR0915
    diagnostic: GalerkinCoordinateCauchyCurrent,
    trace_frozen_coefficients: Complex[Array, "..."],
    normal_frozen_coefficients: Complex[Array, "..."],
    trace_coefficient_error_bounds: Float[Array, "..."],
    normal_coefficient_error_bounds: Float[Array, "..."],
    exact_trace_operator_norm_upper_bound: Float[Array, ""],
    exact_normal_operator_norm_upper_bound: Float[Array, ""],
    trace_operator_error_upper_bound: Float[Array, ""],
    normal_operator_error_upper_bound: Float[Array, ""],
    current_operator_error_upper_bound: Float[Array, ""],
    number_current_scale: Float[Array, ""],
    exact_number_current_scale_lower_bound: Float[Array, ""],
    exact_number_current_scale_upper_bound: Float[Array, ""],
    number_current_scale_error_upper_bound: Float[Array, ""],
    terminal_plane_coordinate: Float[Array, ""],
    arithmetic_environment_supported: Bool[Array, ""],
    gradual_underflow_supported: Bool[Array, ""],
    current_operator_failure_mask: Int[Array, ""],
    *,
    current_scope: GalerkinTerminalCurrentScope,
    route: GalerkinTerminalCurrentRoute,
    coefficient_metrics: str,
    fixed_linear_target: str,
    per_call_action_route: str,
    current_normalization: str,
    eligibility_scope: str,
) -> GalerkinCurrentOperatorCertificate:
    """Create validated uniform selected-sector current-operator evidence.

    :see: :class:`~.test_terminal_types.TestCurrentOperatorTypes`

    The eligibility bit is factory-derived from the exact typed failure mask;
    callers cannot supply it independently.  This bounded first route fixes
    ``xi=0`` and the acquisition-selected fiber scope.  It certifies neither a
    vacuum branch nor a detector.

    Returns
    -------
    certificate : GalerkinCurrentOperatorCertificate
        Structurally validated public storage record.
    """
    state_size: int = diagnostic.target.support.state_indices.shape[0]
    trace_coefficients: Complex128[Array, " n"] = jnp.asarray(
        trace_frozen_coefficients, dtype=jnp.complex128
    )
    normal_coefficients: Complex128[Array, " n"] = jnp.asarray(
        normal_frozen_coefficients, dtype=jnp.complex128
    )
    trace_errors: Float64[Array, " n"] = jnp.asarray(
        trace_coefficient_error_bounds, dtype=jnp.float64
    )
    normal_errors: Float64[Array, " n"] = jnp.asarray(
        normal_coefficient_error_bounds, dtype=jnp.float64
    )
    for values, name in (
        (trace_coefficients, "trace_frozen_coefficients"),
        (normal_coefficients, "normal_frozen_coefficients"),
        (trace_errors, "trace_coefficient_error_bounds"),
        (normal_errors, "normal_coefficient_error_bounds"),
    ):
        _raise_if(values.ndim != 1, f"{name} must be 1D")
        _raise_if(values.shape[0] != state_size, f"{name} must match K_u")

    trace_norm: Float64[Array, ""] = jnp.asarray(
        exact_trace_operator_norm_upper_bound, dtype=jnp.float64
    )
    normal_norm: Float64[Array, ""] = jnp.asarray(
        exact_normal_operator_norm_upper_bound, dtype=jnp.float64
    )
    trace_error: Float64[Array, ""] = jnp.asarray(
        trace_operator_error_upper_bound, dtype=jnp.float64
    )
    normal_error: Float64[Array, ""] = jnp.asarray(
        normal_operator_error_upper_bound, dtype=jnp.float64
    )
    current_error: Float64[Array, ""] = jnp.asarray(
        current_operator_error_upper_bound, dtype=jnp.float64
    )
    current_scale: Float64[Array, ""] = jnp.asarray(
        number_current_scale, dtype=jnp.float64
    )
    scale_lower: Float64[Array, ""] = jnp.asarray(
        exact_number_current_scale_lower_bound, dtype=jnp.float64
    )
    scale_upper: Float64[Array, ""] = jnp.asarray(
        exact_number_current_scale_upper_bound, dtype=jnp.float64
    )
    scale_error: Float64[Array, ""] = jnp.asarray(
        number_current_scale_error_upper_bound, dtype=jnp.float64
    )
    plane: Float64[Array, ""] = jnp.asarray(
        terminal_plane_coordinate, dtype=jnp.float64
    )
    arithmetic_supported: Bool[Array, ""] = jnp.asarray(
        arithmetic_environment_supported
    )
    gradual_supported: Bool[Array, ""] = jnp.asarray(
        gradual_underflow_supported
    )
    failure_mask: Int64[Array, ""] = jnp.asarray(
        current_operator_failure_mask, dtype=jnp.int64
    )
    for value, name in (
        (trace_norm, "exact_trace_operator_norm_upper_bound"),
        (normal_norm, "exact_normal_operator_norm_upper_bound"),
        (trace_error, "trace_operator_error_upper_bound"),
        (normal_error, "normal_operator_error_upper_bound"),
        (current_error, "current_operator_error_upper_bound"),
        (current_scale, "number_current_scale"),
        (scale_lower, "exact_number_current_scale_lower_bound"),
        (scale_upper, "exact_number_current_scale_upper_bound"),
        (scale_error, "number_current_scale_error_upper_bound"),
        (plane, "terminal_plane_coordinate"),
        (arithmetic_supported, "arithmetic_environment_supported"),
        (gradual_supported, "gradual_underflow_supported"),
        (failure_mask, "current_operator_failure_mask"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")

    _raise_if(
        current_scope
        is not GalerkinTerminalCurrentScope.SELECTED_ACQUISITION_FIBER_SECTOR,
        "bounded current-operator route requires selected K_d fiber scope",
    )
    _raise_if(
        current_scope is not diagnostic.current_scope,
        "current_scope must match the nested diagnostic",
    )
    _raise_if(
        route
        is not (GalerkinTerminalCurrentRoute.FTZ_SAFE_EXACT_CARRIER_CAUCHY),
        "current-operator route must match the coordinate-current route",
    )
    for declaration, name in (
        (coefficient_metrics, "coefficient_metrics"),
        (fixed_linear_target, "fixed_linear_target"),
        (per_call_action_route, "per_call_action_route"),
        (current_normalization, "current_normalization"),
        (eligibility_scope, "eligibility_scope"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")

    nonnegative_bounds: Bool[Array, ""] = (
        jnp.all(trace_errors >= 0.0)
        & jnp.all(normal_errors >= 0.0)
        & (trace_norm >= 0.0)
        & (normal_norm >= 0.0)
        & (trace_error >= 0.0)
        & (normal_error >= 0.0)
        & (current_error >= 0.0)
    )
    finite_operator_evidence: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(trace_coefficients))
        & jnp.all(jnp.isfinite(normal_coefficients))
        & jnp.all(jnp.isfinite(trace_errors))
        & jnp.all(jnp.isfinite(normal_errors))
        & jnp.isfinite(trace_norm)
        & jnp.isfinite(normal_norm)
        & jnp.isfinite(trace_error)
        & jnp.isfinite(normal_error)
        & jnp.isfinite(current_error)
        & jnp.isfinite(plane)
        & (plane == 0.0)
        & nonnegative_bounds
    )
    scale_distance: Float64[Array, ""] = jnp.maximum(
        jnp.abs(current_scale - scale_lower),
        jnp.abs(current_scale - scale_upper),
    )
    normalization_enclosed: Bool[Array, ""] = (
        jnp.isfinite(current_scale)
        & jnp.isfinite(scale_lower)
        & jnp.isfinite(scale_upper)
        & jnp.isfinite(scale_error)
        & (current_scale > 0.0)
        & (scale_lower > 0.0)
        & (scale_lower <= scale_upper)
        & (scale_error >= scale_distance)
    )
    diagnostic_eligible: Bool[Array, ""] = (
        diagnostic.current_diagnostic_eligible
    )
    fixed_linear_eligible: Bool[Array, ""] = (
        diagnostic.target.fixed_linear_error_ledger.finite_certificate
    )
    zero: Int64[Array, ""] = jnp.asarray(
        int(GalerkinCurrentOperatorFailure.NONE), dtype=jnp.int64
    )
    expected_mask: Int64[Array, ""] = zero
    for passed, reason in (
        (
            diagnostic_eligible,
            GalerkinCurrentOperatorFailure.CURRENT_DIAGNOSTIC_INELIGIBLE,
        ),
        (
            fixed_linear_eligible,
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
        expected_mask = jnp.bitwise_or(
            expected_mask,
            jnp.where(passed, zero, int(reason)),
        )
    checked_failure: Int64[Array, ""] = eqx.error_if(
        failure_mask,
        failure_mask != expected_mask,
        "current-operator failure mask must equal reconstructed predicates",
    )
    eligible: Bool[Array, ""] = checked_failure == int(
        GalerkinCurrentOperatorFailure.NONE
    )
    certificate: GalerkinCurrentOperatorCertificate = (
        GalerkinCurrentOperatorCertificate(
            diagnostic=diagnostic,
            trace_frozen_coefficients=trace_coefficients,
            normal_frozen_coefficients=normal_coefficients,
            trace_coefficient_error_bounds=trace_errors,
            normal_coefficient_error_bounds=normal_errors,
            exact_trace_operator_norm_upper_bound=trace_norm,
            exact_normal_operator_norm_upper_bound=normal_norm,
            trace_operator_error_upper_bound=trace_error,
            normal_operator_error_upper_bound=normal_error,
            current_operator_error_upper_bound=current_error,
            number_current_scale=current_scale,
            exact_number_current_scale_lower_bound=scale_lower,
            exact_number_current_scale_upper_bound=scale_upper,
            number_current_scale_error_upper_bound=scale_error,
            terminal_plane_coordinate=plane,
            arithmetic_environment_supported=arithmetic_supported,
            gradual_underflow_supported=gradual_supported,
            current_operator_eligible=eligible,
            current_operator_failure_mask=checked_failure,
            current_scope=current_scope,
            route=route,
            coefficient_metrics=coefficient_metrics.strip(),
            fixed_linear_target=fixed_linear_target.strip(),
            per_call_action_route=per_call_action_route.strip(),
            current_normalization=current_normalization.strip(),
            eligibility_scope=eligibility_scope.strip(),
        )
    )
    return certificate


@jaxtyped(typechecker=beartype)
def create_galerkin_terminal_current_action_enclosure(  # noqa: PLR0913,PLR0915
    certificate: GalerkinCurrentOperatorCertificate,
    submitted_field: Complex[Array, "..."],
    production_action: Complex[Array, "..."],
    algebraic_action_real_lower_bounds: Float[Array, "..."],
    algebraic_action_real_upper_bounds: Float[Array, "..."],
    algebraic_action_imag_lower_bounds: Float[Array, "..."],
    algebraic_action_imag_upper_bounds: Float[Array, "..."],
    component_error_bounds: Float[Array, "..."],
    action_error_bound: Float[Array, ""],
    arithmetic_environment_supported: Bool[Array, ""],
    gradual_underflow_supported: Bool[Array, ""],
    failure_mask: Int[Array, ""],
    *,
    route: GalerkinTerminalCurrentRoute,
    exact_action_target: str,
    coefficient_norm: str,
    error_scope: str,
) -> GalerkinTerminalCurrentActionEnclosure:
    """Create one validated per-call frozen-current action enclosure.

    :see: :class:`~.test_terminal_types.TestCurrentOperatorTypes`

    This factory checks representation invariants only.  The canonical
    terminal producer computes the load-bearing aggregate norm with shared
    FTZ-safe interval arithmetic, and every scientific consumer must replay
    that producer rather than trusting a public carrier by possession.

    Returns
    -------
    enclosure : GalerkinTerminalCurrentActionEnclosure
        Structurally validated public storage record.
    """
    state_size: int = (
        certificate.diagnostic.target.support.state_indices.shape[0]
    )
    field: Complex128[Array, " n"] = jnp.asarray(
        submitted_field, dtype=jnp.complex128
    )
    action: Complex128[Array, " n"] = jnp.asarray(
        production_action, dtype=jnp.complex128
    )
    real_lower: Float64[Array, " n"] = jnp.asarray(
        algebraic_action_real_lower_bounds, dtype=jnp.float64
    )
    real_upper: Float64[Array, " n"] = jnp.asarray(
        algebraic_action_real_upper_bounds, dtype=jnp.float64
    )
    imag_lower: Float64[Array, " n"] = jnp.asarray(
        algebraic_action_imag_lower_bounds, dtype=jnp.float64
    )
    imag_upper: Float64[Array, " n"] = jnp.asarray(
        algebraic_action_imag_upper_bounds, dtype=jnp.float64
    )
    component_errors: Float64[Array, " n"] = jnp.asarray(
        component_error_bounds, dtype=jnp.float64
    )
    for values, name in (
        (field, "submitted_field"),
        (action, "production_action"),
        (real_lower, "algebraic_action_real_lower_bounds"),
        (real_upper, "algebraic_action_real_upper_bounds"),
        (imag_lower, "algebraic_action_imag_lower_bounds"),
        (imag_upper, "algebraic_action_imag_upper_bounds"),
        (component_errors, "component_error_bounds"),
    ):
        _raise_if(values.ndim != 1, f"{name} must be 1D")
        _raise_if(values.shape[0] != state_size, f"{name} must match K_u")
    error: Float64[Array, ""] = jnp.asarray(
        action_error_bound, dtype=jnp.float64
    )
    arithmetic_supported: Bool[Array, ""] = jnp.asarray(
        arithmetic_environment_supported
    )
    gradual_supported: Bool[Array, ""] = jnp.asarray(
        gradual_underflow_supported
    )
    submitted_failure: Int64[Array, ""] = jnp.asarray(
        failure_mask, dtype=jnp.int64
    )
    for value, name in (
        (error, "action_error_bound"),
        (arithmetic_supported, "arithmetic_environment_supported"),
        (gradual_supported, "gradual_underflow_supported"),
        (submitted_failure, "failure_mask"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")
    _raise_if(route is not certificate.route, "route must match certificate")
    for declaration, name in (
        (exact_action_target, "exact_action_target"),
        (coefficient_norm, "coefficient_norm"),
        (error_scope, "error_scope"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")

    ordered: Bool[Array, ""] = jnp.all(
        (real_lower <= real_upper) & (imag_lower <= imag_upper)
    )
    finite_evidence: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(field))
        & jnp.all(jnp.isfinite(action))
        & jnp.all(jnp.isfinite(real_lower))
        & jnp.all(jnp.isfinite(real_upper))
        & jnp.all(jnp.isfinite(imag_lower))
        & jnp.all(jnp.isfinite(imag_upper))
        & jnp.all(jnp.isfinite(component_errors))
        & jnp.isfinite(error)
        & jnp.all(component_errors >= 0.0)
        & (error >= 0.0)
        & ordered
    )
    real_distance: Float64[Array, " n"] = jnp.maximum(
        jnp.abs(jnp.real(action) - real_lower),
        jnp.abs(jnp.real(action) - real_upper),
    )
    imag_distance: Float64[Array, " n"] = jnp.maximum(
        jnp.abs(jnp.imag(action) - imag_lower),
        jnp.abs(jnp.imag(action) - imag_upper),
    )
    component_coverage: Bool[Array, ""] = jnp.all(
        component_errors >= jnp.hypot(real_distance, imag_distance)
    )
    component_floor_coverage: Bool[Array, ""] = error >= jnp.max(
        component_errors
    )
    finite_evidence = (
        finite_evidence & component_coverage & component_floor_coverage
    )
    zero: Int64[Array, ""] = jnp.asarray(
        int(GalerkinTerminalCurrentActionFailure.NONE), dtype=jnp.int64
    )
    expected_mask: Int64[Array, ""] = zero
    for passed, reason in (
        (
            certificate.current_operator_eligible,
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
        expected_mask = jnp.bitwise_or(
            expected_mask,
            jnp.where(passed, zero, int(reason)),
        )
    checked_failure: Int64[Array, ""] = eqx.error_if(
        submitted_failure,
        submitted_failure != expected_mask,
        "current-action failure mask must equal reconstructed predicates",
    )
    finite_certificate: Bool[Array, ""] = checked_failure == int(
        GalerkinTerminalCurrentActionFailure.NONE
    )
    enclosure: GalerkinTerminalCurrentActionEnclosure = (
        GalerkinTerminalCurrentActionEnclosure(
            certificate=certificate,
            submitted_field=field,
            production_action=action,
            algebraic_action_real_lower_bounds=real_lower,
            algebraic_action_real_upper_bounds=real_upper,
            algebraic_action_imag_lower_bounds=imag_lower,
            algebraic_action_imag_upper_bounds=imag_upper,
            component_error_bounds=component_errors,
            action_error_bound=error,
            arithmetic_environment_supported=arithmetic_supported,
            gradual_underflow_supported=gradual_supported,
            finite_certificate=finite_certificate,
            failure_mask=checked_failure,
            route=route,
            exact_action_target=exact_action_target.strip(),
            coefficient_norm=coefficient_norm.strip(),
            error_scope=error_scope.strip(),
        )
    )
    return enclosure


__all__: list[str] = [
    "GalerkinCoordinateCauchyCurrent",
    "GalerkinCurrentOperatorCertificate",
    "GalerkinCurrentOperatorFailure",
    "GalerkinDetectorFailure",
    "GalerkinTerminalCurrentActionEnclosure",
    "GalerkinTerminalCurrentActionFailure",
    "GalerkinTerminalCurrentFailure",
    "GalerkinTerminalCurrentRoute",
    "GalerkinTerminalCurrentScope",
    "GalerkinVacuumBranchFailure",
    "create_galerkin_coordinate_cauchy_current",
    "create_galerkin_current_operator_certificate",
    "create_galerkin_terminal_current_action_enclosure",
]
