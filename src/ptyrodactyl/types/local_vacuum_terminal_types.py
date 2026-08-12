r"""Define composed local vacuum-terminal evidence carriers.

Extended Summary
----------------
These host evidence carriers compose one replayed local projection defect
with both exact slab-plane current diagnostics, strict physical vacuum roots,
homogeneous Cauchy propagators, independent endpoint/forced mismatch routes,
and the nonsymmetrized projection-work cut balance.  They distinguish a
plane-defined continuation from exact native-sector and native-slab claims.
No carrier asserts detector eligibility.

Routine Listings
----------------
:class:`GalerkinLocalVacuumBranchEvidence`
    Store exact submitted-state Cauchy and branch mismatch evidence.
:class:`GalerkinLocalVacuumCutBalance`
    Store the independent reduced-current and defect-work cross-check.
:class:`GalerkinLocalVacuumHalfSpaceDisposition`
    Classify exact-zero, nonzero, or unresolved excluded branch content.
:class:`GalerkinLocalVacuumTerminalCertificate`
    Store one composed LVT.39--LVT.56 local vacuum terminal certificate.
:class:`GalerkinLocalVacuumTerminalDisposition`
    Select one honest plane-defined or exact-native continuation claim.
:class:`GalerkinLocalVacuumTerminalEntireEvidence`
    Store exact phase and forced-integral helper resource evidence.
:class:`GalerkinLocalVacuumTerminalFailure`
    Enumerate simultaneous local vacuum-terminal noncertificate outcomes.
"""

from __future__ import annotations

import math
from enum import IntFlag, StrEnum
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex128, Float64, Int64

from ptyrodactyl._tools import (
    EntireEnclosureFailure,
    EntireWorkTranscript,
    fraction_upper_float,
    has_subnormal_components,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)

from .acquisition_types import GalerkinTerminalSide
from .local_projection_types import GalerkinLocalProjectionDefectCertificate
from .local_terminal_types import (
    GalerkinLocalCoordinateCauchyCurrent,
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalScope,
)
from .local_vacuum_propagation_types import (
    GalerkinLocalVacuumPropagationFailure,
    GalerkinLocalVacuumPropagator,
    GalerkinLocalVacuumRootCertificate,
    GalerkinLocalVacuumRootClass,
    _validate_local_vacuum_propagator,
    _validate_local_vacuum_root_certificate,
)

_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_SHA256_HEX_LENGTH: int = 64
_CAUCHY_COMPONENT_COUNT: int = 2
_BRANCH_ROLE_COUNT: int = 2
_RECTANGLE_ROLE_COUNT: int = 2
_CAUCHY_EVIDENCE_COUNT: int = 5
_BRANCH_RECTANGLE_COUNT: int = 4
_PRODUCTION_EVIDENCE_COUNT: int = 12
_PLANE_MISMATCH_BOUND_COUNT: int = 3
_CROSSCHECK_MASK_COUNT: int = 2
_WORK_EVIDENCE_COUNT: int = 2
_BRANCH_ELIGIBILITY_COUNT: int = 3
_CUT_ELIGIBILITY_COUNT: int = 3
_TERMINAL_ELIGIBILITY_COUNT: int = 4
_CUT_REPORT_COUNT: int = 6
_PREDICTION_BRANCH_ROLE: int = 0
_ENTIRE_POLICY_COUNT: int = 5
_HULL_ALGORITHM: str = "outward_binary64_normal_hull_v1"
_MAXIMUM_BINARY64_RATIONAL_BITS: int = 1024

_PRODUCTION_TO_SUBMITTED_AMPLITUDE_SCOPE: str = (
    "the frozen defining-plane point and its direct exact-x amplitude error "
    "use only the stored state x; the one-time dyadic hull widening is "
    "already consumed through the defining rectangle and is never added as "
    "a separate error; endpoint and forced-integral rectangles contain no "
    "L6 state-radius transfer"
)
_STATE_RADIUS_AMPLITUDE_SCOPE: str = (
    "defining-plane state transfer uses the exact terminal amplitude map "
    "norm times the replayed L6 state radius B; it excludes projection "
    "D0 transfer and every LVT.55 plane mismatch"
)
_EXACT_STATE_AMPLITUDE_SCOPE: str = (
    "defining-plane exact-state amplitude error is the direct frozen-point "
    "to exact-x error plus the terminal-map state-radius transfer exactly "
    "once; it excludes every LVT.55 plane mismatch"
)
_SUBMITTED_PLANE_MISMATCH_SCOPE: str = (
    "endpoint and forced-integral LVT.55 mismatch for the exact stored "
    "submitted state x only"
)
_PROJECTION_STATE_TRANSFER_MISMATCH_SCOPE: str = (
    "plane/native mismatch transfer uses only projection ||D0,h|| B and is "
    "not a defining-plane amplitude state-transfer term"
)
_PROJECTION_TOTAL_MISMATCH_SCOPE: str = (
    "exact-state plane/native mismatch uses projection total E_f(x,B), with "
    "its ||D0,h|| B contribution included exactly once"
)
_ROOT_REALIZATION_SCOPE: str = (
    "nearest-float root and outward interval distance use the raw replayed "
    "root interval and are audit evidence already absorbed by the direct "
    "defining-branch rectangle error; neither root error nor dyadic hull "
    "widening is added separately to E_a"
)
_PREDICTION_BRANCH_ROLE_SCOPE: str = (
    "role 0 is outward for propagating roots, decaying for evanescent roots, "
    "and field t for grazing roots; grazing role 1 is side-oriented "
    "derivative nu"
)
_AMPLITUDE_DEPENDENCY_SCOPE: str = (
    "LVT.56 defining-plane E_a combines frozen-production error with exact "
    "terminal-map state transfer once; projection total E_f is a separate "
    "plane/native mismatch summand charged only for another-plane or native "
    "invariance and never recombined with ||D0||B"
)
_COMPLETION_SCOPE: str = (
    "scoped local vacuum Cauchy branches and exact-native dispositions only; "
    "excludes positive-port selection, detector eligibility, quadrature, "
    "dose, response, calibration, likelihood, and continuum accuracy"
)
_ENTIRE_HELPER_SCOPE: str = (
    "exact per-kernel physical-phase exp, forced-integral phi1, and grazing "
    "phi2 helper transcripts; excludes root/propagator work and direct linear "
    "rectangle terms"
)

type _RectanglePair = tuple[
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalComplexRectangles,
]
type _PropagationFailure = (
    EntireEnclosureFailure | GalerkinLocalVacuumPropagationFailure | None
)
type _CauchyEvidence = tuple[
    _RectanglePair,
    _RectanglePair,
    _RectanglePair,
    _RectanglePair,
    _RectanglePair,
]
type _BranchRectangles = tuple[
    _RectanglePair,
    _RectanglePair,
    _RectanglePair,
    _RectanglePair,
]
type _ProductionEvidence = tuple[
    Float64[Array, " f"],
    Float64[Array, " f"],
    Complex128[Array, "f 2"],
    Complex128[Array, "f 2"],
    Float64[Array, "f 2"],
    Float64[Array, "f 2"],
    Float64[Array, "f 2"],
    Float64[Array, "f 2"],
    Float64[Array, "f 2"],
    Float64[Array, ""],
    Float64[Array, ""],
    Float64[Array, ""],
]
type _PlaneMismatchBounds = tuple[
    Float64[Array, "f 2"],
    Float64[Array, "f 2"],
    Float64[Array, "f 2"],
]
type _CrosscheckMasks = tuple[
    Bool[Array, "f 2"],
    Bool[Array, "f 2"],
]
type _WorkEvidence = tuple[Int64[Array, ""], Int64[Array, ""]]
type _BranchEligibility = tuple[
    Bool[Array, ""], Bool[Array, ""], Bool[Array, ""]
]
type _HalfSpaceDispositions = tuple[
    GalerkinLocalVacuumHalfSpaceDisposition, ...
]
type _CutEligibility = tuple[Bool[Array, ""], Bool[Array, ""], Bool[Array, ""]]
type _TerminalEligibility = tuple[
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
]


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise for one structural carrier failure.

    Parameters
    ----------
    condition : bool
        Whether the structural failure is present.
    message : str
        Diagnostic for the rejected carrier.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 value.

    Parameters
    ----------
    value : str
        Candidate hexadecimal digest.

    Returns
    -------
    valid : bool
        Whether ``value`` is canonical lowercase SHA-256 text.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


def _normal_or_zero(values: Array) -> bool:
    """PRIVATE: Check finite normal-range components or exact zeros.

    Parameters
    ----------
    values : Array
        Candidate real binary64 scalar or array.

    Returns
    -------
    valid : bool
        Whether every component is finite and non-subnormal.
    """
    valid: bool = bool(jnp.all(jnp.isfinite(values))) and not bool(
        has_subnormal_components(values)
    )
    return valid


class GalerkinLocalVacuumTerminalDisposition(StrEnum):
    """Select one honest plane-defined or exact-native continuation claim.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_terminal_enums_are_explicit_and_disjoint`
    """

    PLANE_DEFINED_FREE_CONTINUATION = "plane_defined_free_continuation"
    NATIVE_ZERO_DEFECT_TERMINAL_SECTOR = "native_zero_defect_terminal_sector"
    NATIVE_ZERO_DEFECT_SLAB = "native_zero_defect_slab"


class GalerkinLocalVacuumHalfSpaceDisposition(StrEnum):
    """Classify exact-zero, nonzero, or unresolved excluded branch content.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_terminal_enums_are_explicit_and_disjoint`
    """

    PROPAGATING_INWARD_EXACT_ZERO = "propagating_inward_exact_zero"
    PROPAGATING_INWARD_PROVABLY_NONZERO = "propagating_inward_provably_nonzero"
    PROPAGATING_INWARD_UNRESOLVED = "propagating_inward_unresolved"
    EVANESCENT_GROWING_EXACT_ZERO = "evanescent_growing_exact_zero"
    EVANESCENT_GROWING_PROVABLY_NONZERO = "evanescent_growing_provably_nonzero"
    EVANESCENT_GROWING_UNRESOLVED = "evanescent_growing_unresolved"
    GRAZING_DERIVATIVE_EXACT_ZERO = "grazing_derivative_exact_zero"
    GRAZING_DERIVATIVE_PROVABLY_NONZERO = "grazing_derivative_provably_nonzero"
    GRAZING_DERIVATIVE_UNRESOLVED = "grazing_derivative_unresolved"
    ROOT_UNCLASSIFIED = "root_unclassified"


class GalerkinLocalVacuumTerminalFailure(IntFlag):
    """Enumerate simultaneous local vacuum-terminal noncertificate outcomes.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_terminal_enums_are_explicit_and_disjoint`
    """

    NONE = 0
    ZERO_SLAB_NONCERTIFICATE = 1 << 0
    PROJECTION_NONCERTIFICATE = 1 << 1
    CURRENT_DIAGNOSTIC_NONCERTIFICATE = 1 << 2
    CURRENT_OPERATOR_NONCERTIFICATE = 1 << 3
    CURRENT_ACTION_NONCERTIFICATE = 1 << 4
    ROOT_UNCLASSIFIED = 1 << 5
    ROOT_PROPAGATOR_FAILURE = 1 << 6
    CAUCHY_CROSSCHECK_EMPTY = 1 << 7
    BRANCH_CROSSCHECK_EMPTY = 1 << 8
    CUT_BALANCE_CROSSCHECK_EMPTY = 1 << 9
    NATIVE_STRUCTURAL_ZERO_UNAVAILABLE = 1 << 10
    DISPOSITION_SCOPE_MISMATCH = 1 << 11
    HOST_ARITHMETIC_UNSUPPORTED = 1 << 12
    DIRECT_WORK_BUDGET_EXCEEDED = 1 << 13
    DIRECT_WORK_COUNT_OVERFLOW = 1 << 14
    ARITHMETIC_RANGE_FAILURE = 1 << 15
    ENTIRE_HELPER_ENCLOSURE_FAILURE = 1 << 16
    DIRECT_RATIONAL_SIZE_FAILURE = 1 << 17


class GalerkinLocalVacuumTerminalEntireEvidence(eqx.Module):
    """Store exact phase and forced-integral helper resource evidence.

    Successful calls retain their complete ``EntireWorkTranscript``. Failed
    calls retain the typed helper reason and exact completed-work count at the
    same deterministic kernel label.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_branch_and_cut_carriers_keep_routes_separate`
    """

    helper_attempted: Bool[Array, ""]
    helper_eligible: Bool[Array, ""]
    kernel_labels: tuple[str, ...] = eqx.field(static=True)
    transcripts: tuple[EntireWorkTranscript | None, ...] = eqx.field(
        static=True
    )
    failure_reasons: tuple[EntireEnclosureFailure | None, ...] = eqx.field(
        static=True
    )
    failure_work_counts: tuple[int, ...] = eqx.field(static=True)
    precision_bits: int = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_range_reductions: int = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    total_series_terms: int = eqx.field(static=True)
    total_range_reductions: int = eqx.field(static=True)
    total_root_enclosures: int = eqx.field(static=True)
    total_rectangle_products: int = eqx.field(static=True)
    total_reciprocal_steps: int = eqx.field(static=True)
    total_exact_work_count: int = eqx.field(static=True)
    helper_scope: str = eqx.field(static=True)
    helper_evidence_digest: str = eqx.field(static=True)


class GalerkinLocalVacuumBranchEvidence(eqx.Module):
    """Store exact submitted-state Cauchy and branch mismatch evidence.

    The endpoint and forced-integral rectangles use the submitted stored
    state ``x``.  Three plane-mismatch arrays separately use submitted,
    projection ``||D0||B``, and total ``E_f(x, B)`` reports.  The distinct
    LVT.56 bridge owns a concrete role-zero production point, direct point-to-x
    error, exact terminal-map state-radius transfer, and exact-once total.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_branch_and_cut_carriers_keep_routes_separate`
    """

    root_certificates: tuple[GalerkinLocalVacuumRootCertificate | None, ...]
    propagators: tuple[GalerkinLocalVacuumPropagator | None, ...]
    root_failure_reasons: tuple[_PropagationFailure, ...] = eqx.field(
        static=True
    )
    root_failure_work_counts: tuple[int, ...] = eqx.field(static=True)
    propagator_failure_reasons: tuple[_PropagationFailure, ...] = eqx.field(
        static=True
    )
    propagator_failure_work_counts: tuple[int, ...] = eqx.field(static=True)
    entire_evidence: GalerkinLocalVacuumTerminalEntireEvidence
    inner_cauchy_rectangles: _RectanglePair
    outer_cauchy_rectangles: _RectanglePair
    endpoint_cauchy_mismatch_rectangles: _RectanglePair
    forced_cauchy_mismatch_rectangles: _RectanglePair
    certified_cauchy_mismatch_rectangles: _RectanglePair
    defining_branch_rectangles: _RectanglePair
    endpoint_branch_mismatch_rectangles: _RectanglePair
    forced_branch_mismatch_rectangles: _RectanglePair
    certified_branch_mismatch_rectangles: _RectanglePair
    submitted_state_branch_mismatch_upper_bounds: Float64[Array, "f 2"]
    projection_state_transfer_branch_mismatch_upper_bounds: Float64[
        Array, "f 2"
    ]
    projection_total_branch_mismatch_upper_bounds: Float64[Array, "f 2"]
    frozen_positive_root_realizations: Float64[Array, " f"]
    frozen_positive_root_error_bounds: Float64[Array, " f"]
    physical_phase_realizations: Complex128[Array, "f 2"]
    frozen_defining_branch_points: Complex128[Array, "f 2"]
    production_to_submitted_amplitude_error_bounds: Float64[Array, "f 2"]
    state_radius_amplitude_error_bounds: Float64[Array, "f 2"]
    exact_state_total_amplitude_error_bounds: Float64[Array, "f 2"]
    production_amplitude_norm_upper_bounds: Float64[Array, "f 2"]
    exact_state_amplitude_norm_upper_bounds: Float64[Array, "f 2"]
    production_prediction_l2_norm_upper_bound: Float64[Array, ""]
    exact_state_prediction_error_l2_upper_bound: Float64[Array, ""]
    exact_state_prediction_l2_norm_upper_bound: Float64[Array, ""]
    cauchy_crosscheck_mask: Bool[Array, "f 2"]
    branch_crosscheck_mask: Bool[Array, "f 2"]
    direct_work_count: Int64[Array, ""]
    maximum_direct_terms: Int64[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    normal_arithmetic_eligible: Bool[Array, ""]
    branch_evidence_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    half_space_dispositions: _HalfSpaceDispositions = eqx.field(static=True)
    prediction_branch_role: int = eqx.field(static=True)
    prediction_branch_role_scope: str = eqx.field(static=True)
    direct_work_count_exact: str = eqx.field(static=True)
    maximum_root_work: int = eqx.field(static=True)
    maximum_propagator_interval_work: int = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    direct_rational_peak_bits: int = eqx.field(static=True)
    direct_rational_work_count_exact: str = eqx.field(static=True)
    direct_rational_failure: EntireEnclosureFailure | None = eqx.field(
        static=True
    )
    hull_algorithm: str = eqx.field(static=True)
    hull_attempted_endpoint_count: int = eqx.field(static=True)
    hull_completed_endpoint_count: int = eqx.field(static=True)
    hull_input_peak_bits: int = eqx.field(static=True)
    hull_output_peak_bits: int = eqx.field(static=True)
    hull_normal_floor_count: int = eqx.field(static=True)
    hull_range_failure: bool = eqx.field(static=True)
    hull_evidence_digest: str = eqx.field(static=True)
    direct_work_formula: str = eqx.field(static=True)
    physical_root_formula: str = eqx.field(static=True)
    root_realization_formula: str = eqx.field(static=True)
    root_realization_scope: str = eqx.field(static=True)
    physical_cauchy_formula: str = eqx.field(static=True)
    endpoint_mismatch_formula: str = eqx.field(static=True)
    forced_mismatch_formula: str = eqx.field(static=True)
    plane_mismatch_bound_formula: str = eqx.field(static=True)
    amplitude_error_formula: str = eqx.field(static=True)
    amplitude_norm_formula: str = eqx.field(static=True)
    production_to_submitted_amplitude_scope: str = eqx.field(static=True)
    state_radius_amplitude_scope: str = eqx.field(static=True)
    exact_state_amplitude_scope: str = eqx.field(static=True)
    submitted_plane_mismatch_scope: str = eqx.field(static=True)
    projection_state_transfer_mismatch_scope: str = eqx.field(static=True)
    projection_total_mismatch_scope: str = eqx.field(static=True)
    helper_policy_digest: str = eqx.field(static=True)
    physical_root_identity_digest: str = eqx.field(static=True)
    cauchy_evidence_digest: str = eqx.field(static=True)
    branch_evidence_digest: str = eqx.field(static=True)


class GalerkinLocalVacuumCutBalance(eqx.Module):
    """Store the independent reduced-current and defect-work cross-check.

    The defect-work interval is formed from ``G diag(d)`` without Hermitian
    symmetrization.  The side-oriented outer-minus-inner reduced current is
    the same positive-coordinate cut difference for either terminal side.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_branch_and_cut_carriers_keep_routes_separate`
    """

    current_difference_lower_bound: Float64[Array, ""]
    current_difference_upper_bound: Float64[Array, ""]
    negative_defect_work_lower_bound: Float64[Array, ""]
    negative_defect_work_upper_bound: Float64[Array, ""]
    certified_balance_lower_bound: Float64[Array, ""]
    certified_balance_upper_bound: Float64[Array, ""]
    direct_work_count: Int64[Array, ""]
    maximum_direct_pairs: Int64[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    normal_arithmetic_eligible: Bool[Array, ""]
    cut_balance_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    direct_work_count_exact: str = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    direct_rational_peak_bits: int = eqx.field(static=True)
    direct_rational_work_count_exact: str = eqx.field(static=True)
    direct_rational_failure: EntireEnclosureFailure | None = eqx.field(
        static=True
    )
    direct_work_formula: str = eqx.field(static=True)
    current_difference_formula: str = eqx.field(static=True)
    defect_work_formula: str = eqx.field(static=True)
    balance_scope: str = eqx.field(static=True)
    cut_balance_digest: str = eqx.field(static=True)


class GalerkinLocalVacuumTerminalCertificate(eqx.Module):
    """Store one composed LVT.39--LVT.56 local vacuum terminal certificate.

    Vacuum eligibility is stronger than distinct both-plane current
    diagnostic, operator, and submitted-action eligibility.  It additionally
    requires final zero-slab readiness, a finite projection bound, classified
    roots, both mismatch cross-checks, the cut-balance intersection, and the
    requested honest continuation disposition.  Detector eligibility is
    deliberately absent.

    :see: :func:`~.test_local_vacuum_terminal_types.\
test_local_vacuum_terminal_certificate_owns_no_detector_claim`
    """

    projection_certificate: GalerkinLocalProjectionDefectCertificate
    inner_current_diagnostic: GalerkinLocalCoordinateCauchyCurrent
    outer_current_diagnostic: GalerkinLocalCoordinateCauchyCurrent
    branch_evidence: GalerkinLocalVacuumBranchEvidence
    cut_balance: GalerkinLocalVacuumCutBalance
    defining_plane_coordinate: Float64[Array, ""]
    comparison_plane_coordinate: Float64[Array, ""]
    current_diagnostic_eligible: Bool[Array, ""]
    current_operator_eligible: Bool[Array, ""]
    current_action_eligible: Bool[Array, ""]
    vacuum_branch_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    terminal_axis: int = eqx.field(static=True)
    terminal_side: GalerkinTerminalSide = eqx.field(static=True)
    terminal_scope: GalerkinLocalTerminalScope = eqx.field(static=True)
    disposition: GalerkinLocalVacuumTerminalDisposition = eqx.field(
        static=True
    )
    amplitude_dependency_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    state_identity_digest: str = eqx.field(static=True)
    projection_identity_digest: str = eqx.field(static=True)
    parent_projection_certificate_digest: str = eqx.field(static=True)
    inner_terminal_evidence_digest: str = eqx.field(static=True)
    outer_terminal_evidence_digest: str = eqx.field(static=True)
    branch_evidence_digest: str = eqx.field(static=True)
    cut_balance_digest: str = eqx.field(static=True)
    terminal_identity_digest: str = eqx.field(static=True)
    terminal_evidence_digest: str = eqx.field(static=True)


def _validate_rectangles(
    rectangles: GalerkinLocalTerminalComplexRectangles,
    size: int,
    name: str,
    *,
    require_normal: bool,
) -> None:
    """PRIVATE: Validate one componentwise complex rectangle vector.

    Parameters
    ----------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Candidate componentwise complex rectangles.
    size : int
        Required vector length.
    name : str
        Diagnostic evidence name.
    require_normal : bool
        Whether every endpoint must be finite normal-or-zero. This static
        keyword changes host validation but does not enter a JIT trace.

    Raises
    ------
    TypeError
        If ``rectangles`` has the wrong carrier type.
    ValueError
        If shapes, dtypes, ordering, NaNs, or range are invalid.
    """
    if not isinstance(rectangles, GalerkinLocalTerminalComplexRectangles):
        raise TypeError(f"{name} has the wrong rectangle carrier type")
    values = tuple(jnp.asarray(value) for value in rectangles)
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.float64) or value.shape != (size,)
            for value in values
        ),
        f"{name} must contain four float64 vectors of length {size}",
    )
    _raise_if(
        bool(jnp.any(values[0] > values[1]))
        or bool(jnp.any(values[2] > values[3]))
        or any(bool(jnp.any(jnp.isnan(value))) for value in values),
        f"{name} must be ordered and contain no NaN",
    )
    _raise_if(
        any(bool(has_subnormal_components(value)) for value in values),
        f"{name} must contain no subnormal endpoint",
    )
    if require_normal:
        _raise_if(
            any(not _normal_or_zero(value) for value in values),
            f"{name} must be finite normal-or-zero",
        )


def _validate_rectangle_pair(
    rectangles: _RectanglePair,
    size: int,
    name: str,
    *,
    require_normal: bool,
) -> None:
    """PRIVATE: Validate two ordered rectangle-vector roles.

    Parameters
    ----------
    rectangles : _RectanglePair
        Candidate field/normal or primary/secondary rectangle roles.
    size : int
        Required vector length.
    name : str
        Diagnostic evidence name.
    require_normal : bool
        Whether all endpoints must be finite normal-or-zero. This static
        keyword changes host validation but does not enter a JIT trace.

    Raises
    ------
    TypeError
        If the submitted pair structure is invalid.
    ValueError
        If either rectangle role is structurally invalid.
    """
    if (
        not isinstance(rectangles, tuple)
        or len(rectangles) != _RECTANGLE_ROLE_COUNT
    ):
        raise TypeError(f"{name} must contain exactly two rectangle roles")
    for index, value in enumerate(rectangles):
        _validate_rectangles(
            value,
            size,
            f"{name}[{index}]",
            require_normal=require_normal,
        )


def _validate_crosscheck(
    endpoint: _RectanglePair,
    forced: _RectanglePair,
    certified: _RectanglePair,
    submitted_mask: Bool[Array, "f 2"],
    size: int,
    name: str,
) -> tuple[bool, bool]:
    """PRIVATE: Validate endpoint/forced intersections and their mask.

    Parameters
    ----------
    endpoint : _RectanglePair
        Endpoint-route rectangle roles.
    forced : _RectanglePair
        Forced-integral rectangle roles.
    certified : _RectanglePair
        Stored intersections, with unbounded sentinels for empty entries.
    submitted_mask : Bool[Array, "f 2"]
        Stored per-fiber, per-role nonempty-intersection mask.
    size : int
        Scoped transverse-fiber count.
    name : str
        Diagnostic cross-check name.

    Returns
    -------
    complete : bool
        Whether every route pair is available and intersects.
    any_disjoint : bool
        Whether any two available route rectangles are genuinely disjoint.

    Raises
    ------
    ValueError
        If the mask or stored intersections differ from direct reconstruction.
    """
    for label, rectangles in (
        ("endpoint", endpoint),
        ("forced", forced),
        ("certified", certified),
    ):
        _validate_rectangle_pair(
            rectangles,
            size,
            f"{name} {label}",
            require_normal=False,
        )
    mask = jnp.asarray(submitted_mask)
    _raise_if(
        mask.dtype != jnp.dtype(jnp.bool_) or mask.shape != (size, 2),
        f"{name} mask must be bool (f, 2)",
    )
    expected_columns: list[np.ndarray] = []
    disjoint_columns: list[np.ndarray] = []
    for role in range(_RECTANGLE_ROLE_COUNT):
        left = tuple(np.asarray(value) for value in endpoint[role])
        right = tuple(np.asarray(value) for value in forced[role])
        left_available = np.all(
            np.stack([np.isfinite(value) for value in left]), axis=0
        )
        right_available = np.all(
            np.stack([np.isfinite(value) for value in right]), axis=0
        )
        geometric_overlap = (
            np.maximum(left[0], right[0]) <= np.minimum(left[1], right[1])
        ) & (np.maximum(left[2], right[2]) <= np.minimum(left[3], right[3]))
        both_available = left_available & right_available
        expected = geometric_overlap & both_available
        disjoint_columns.append(both_available & ~geometric_overlap)
        expected_columns.append(expected)
        expected_values = (
            np.where(expected, np.maximum(left[0], right[0]), -np.inf),
            np.where(expected, np.minimum(left[1], right[1]), np.inf),
            np.where(expected, np.maximum(left[2], right[2]), -np.inf),
            np.where(expected, np.minimum(left[3], right[3]), np.inf),
        )
        stored = tuple(np.asarray(value) for value in certified[role])
        _raise_if(
            any(
                not np.array_equal(value, expected_value)
                for value, expected_value in zip(
                    stored, expected_values, strict=True
                )
            ),
            f"{name} certified rectangles are not exact intersections",
        )
    expected_mask = np.stack(expected_columns, axis=1)
    _raise_if(
        not np.array_equal(np.asarray(mask), expected_mask),
        f"{name} mask disagrees with endpoint/forced intersections",
    )
    complete = bool(np.all(expected_mask))
    any_disjoint = bool(np.any(np.stack(disjoint_columns, axis=1)))
    return complete, any_disjoint


def _component_zero_status(
    rectangles: GalerkinLocalTerminalComplexRectangles,
    index: int,
) -> str:
    """PRIVATE: Classify one complex rectangle against exact zero.

    Parameters
    ----------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Componentwise complex rectangles.
    index : int
        Fiber index to classify.

    Returns
    -------
    status : str
        One of ``exact_zero``, ``provably_nonzero``, or ``unresolved``.
    """
    values = tuple(float(np.asarray(value)[index]) for value in rectangles)
    if all(value == 0.0 for value in values):
        status = "exact_zero"
    elif (
        values[0] > 0.0
        or values[1] < 0.0
        or values[2] > 0.0
        or values[3] < 0.0
    ):
        status = "provably_nonzero"
    else:
        status = "unresolved"
    return status


def _checked_failure(
    failure_mask: Int64[Array, ""],
) -> GalerkinLocalVacuumTerminalFailure:
    """PRIVATE: Parse one scalar known-bit terminal failure mask.

    Parameters
    ----------
    failure_mask : Int64[Array, ""]
        Candidate simultaneous failure bit mask.

    Returns
    -------
    failure : GalerkinLocalVacuumTerminalFailure
        Canonical known-bit failure value.

    Raises
    ------
    ValueError
        If storage is not scalar int64 or contains unknown bits.
    """
    submitted = jnp.asarray(failure_mask)
    _raise_if(
        submitted.dtype != jnp.dtype(jnp.int64) or submitted.shape != (),
        "failure_mask must be a scalar int64",
    )
    known = 0
    for reason in GalerkinLocalVacuumTerminalFailure:
        known |= int(reason)
    value = int(np.asarray(submitted))
    _raise_if(
        value < 0 or bool(value & ~known),
        "failure_mask has unknown bits",
    )
    return GalerkinLocalVacuumTerminalFailure(value)


def _validate_work(
    work: _WorkEvidence,
    direct_work_count_exact: str,
    failure: GalerkinLocalVacuumTerminalFailure,
) -> int:
    """PRIVATE: Validate exact, stored, and budgeted direct work evidence.

    Parameters
    ----------
    work : _WorkEvidence
        Stored signed-int64 direct count and independent positive policy.
    direct_work_count_exact : str
        Canonical arbitrary-precision decimal count transcript.
    failure : GalerkinLocalVacuumTerminalFailure
        Typed simultaneous outcome owning work bits.

    Returns
    -------
    exact_count : int
        Parsed nonnegative exact direct-work count.

    Raises
    ------
    TypeError
        If work is not the required count/policy tuple.
    ValueError
        If dtypes, transcript, overflow, or budget disposition disagree.
    """
    if not isinstance(work, tuple) or len(work) != _WORK_EVIDENCE_COUNT:
        raise TypeError("direct work evidence must contain count and policy")
    values = tuple(jnp.asarray(value) for value in work)
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.int64) or value.shape != ()
            for value in values
        ),
        "direct work evidence must use scalar int64 storage",
    )
    budget = int(np.asarray(values[1]))
    _raise_if(budget <= 0, "direct work policy must be positive")
    try:
        exact_count = int(direct_work_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "direct work transcript must be canonical decimal"
        ) from error
    _raise_if(
        exact_count < 0 or str(exact_count) != direct_work_count_exact,
        "direct work transcript must be canonical nonnegative decimal",
    )
    stored_count = int(np.asarray(values[0]))
    overflow = bool(
        failure & GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
    )
    budget_exceeded = bool(
        failure
        & GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    _raise_if(
        overflow != (exact_count > _MAXIMUM_SIGNED_INT64),
        "work-overflow bit disagrees with the exact transcript",
    )
    _raise_if(
        budget_exceeded != (exact_count > budget),
        "direct-work budget bit disagrees with exact work",
    )
    if overflow:
        _raise_if(
            stored_count != 0,
            "overflow must preserve exact work and store zero int64 count",
        )
    else:
        _raise_if(
            stored_count != exact_count,
            "stored direct work must equal its signed-int64 transcript",
        )
    return exact_count


def _expected_half_space_disposition(
    root_class: GalerkinLocalVacuumRootClass,
    status: str,
) -> GalerkinLocalVacuumHalfSpaceDisposition:
    """PRIVATE: Map one classified root and zero status to its branch status.

    Parameters
    ----------
    root_class : GalerkinLocalVacuumRootClass
        Strict physical vacuum-root class.
    status : str
        Exact-zero, provably-nonzero, or unresolved rectangle status.

    Returns
    -------
    disposition : GalerkinLocalVacuumHalfSpaceDisposition
        Required excluded-branch or grazing-derivative disposition.

    Raises
    ------
    ValueError
        If ``root_class`` or ``status`` is not a classified combination.
    """
    prefix = {
        GalerkinLocalVacuumRootClass.PROPAGATING: "PROPAGATING_INWARD",
        GalerkinLocalVacuumRootClass.EVANESCENT: "EVANESCENT_GROWING",
        GalerkinLocalVacuumRootClass.GRAZING: "GRAZING_DERIVATIVE",
    }.get(root_class)
    _raise_if(prefix is None, "half-space status requires a classified root")
    suffix = {
        "exact_zero": "EXACT_ZERO",
        "provably_nonzero": "PROVABLY_NONZERO",
        "unresolved": "UNRESOLVED",
    }.get(status)
    _raise_if(suffix is None, "unknown half-space rectangle status")
    return GalerkinLocalVacuumHalfSpaceDisposition[f"{prefix}_{suffix}"]


def _complex_norm_upper(value: complex) -> float:
    """PRIVATE: Return an outward binary64 norm of one frozen complex value.

    Parameters
    ----------
    value : complex
        Finite exact stored complex binary64 point.

    Returns
    -------
    upper : float
        Canonical outward binary64 Euclidean magnitude.
    """
    real = Fraction.from_float(float(np.real(value)))
    imag = Fraction.from_float(float(np.imag(value)))
    return fraction_upper_float(sqrt_fraction_upper(real * real + imag * imag))


def _sum_upper(left: float, right: float) -> float:
    """PRIVATE: Return the canonical outward sum of two binary64 uppers.

    Parameters
    ----------
    left : float
        First finite nonnegative exact binary64 addend.
    right : float
        Second finite nonnegative exact binary64 addend.

    Returns
    -------
    upper : float
        Canonical outward binary64 exact sum.
    """
    exact = Fraction.from_float(left) + Fraction.from_float(right)
    return fraction_upper_float(exact)


def _real_vector_l2_upper(values: np.ndarray) -> float:
    """PRIVATE: Return the canonical outward l2 norm of real uppers.

    Parameters
    ----------
    values : np.ndarray
        Finite nonnegative exact binary64 component uppers.

    Returns
    -------
    upper : float
        Canonical outward binary64 vector norm.
    """
    squared = sum(
        (
            Fraction.from_float(float(value))
            * Fraction.from_float(float(value))
            for value in values
        ),
        start=Fraction(0),
    )
    return fraction_upper_float(sqrt_fraction_upper(squared))


def _complex_vector_l2_upper(values: np.ndarray) -> float:
    """PRIVATE: Return the canonical outward l2 norm of complex points.

    Parameters
    ----------
    values : np.ndarray
        Finite exact stored complex binary64 vector.

    Returns
    -------
    upper : float
        Canonical outward binary64 vector norm.
    """
    squared = Fraction(0)
    for value in values:
        real = Fraction.from_float(float(np.real(value)))
        imag = Fraction.from_float(float(np.imag(value)))
        squared += real * real + imag * imag
    return fraction_upper_float(sqrt_fraction_upper(squared))


def _point_rectangle_error_upper(
    point: complex,
    rectangles: GalerkinLocalTerminalComplexRectangles,
    index: int,
) -> float:
    """PRIVATE: Bound one frozen point against one exact branch rectangle.

    Parameters
    ----------
    point : complex
        Frozen production branch component.
    rectangles : GalerkinLocalTerminalComplexRectangles
        Exact submitted-state branch rectangles.
    index : int
        Scoped fiber row.

    Returns
    -------
    upper : float
        Canonical outward maximum corner distance.
    """
    point_real = Fraction.from_float(float(np.real(point)))
    point_imag = Fraction.from_float(float(np.imag(point)))
    real_endpoints = (
        Fraction.from_float(float(np.asarray(rectangles[0])[index])),
        Fraction.from_float(float(np.asarray(rectangles[1])[index])),
    )
    imag_endpoints = (
        Fraction.from_float(float(np.asarray(rectangles[2])[index])),
        Fraction.from_float(float(np.asarray(rectangles[3])[index])),
    )
    real_error = max(abs(value - point_real) for value in real_endpoints)
    imag_error = max(abs(value - point_imag) for value in imag_endpoints)
    squared = real_error * real_error + imag_error * imag_error
    return fraction_upper_float(sqrt_fraction_upper(squared))


def _validate_production_evidence(
    production_evidence: _ProductionEvidence,
    defining_branches: _RectanglePair,
    fiber_size: int,
    *,
    require_finite_errors: bool,
) -> tuple[Array, ...]:
    """PRIVATE: Validate the concrete LVT.56 point and exact-once errors.

    Parameters
    ----------
    production_evidence : _ProductionEvidence
        Frozen roots, phases, branch points, amplitude errors, and norms.
    defining_branches : _RectanglePair
        Exact submitted-state defining-plane branch rectangles.
    fiber_size : int
        Scoped transverse-fiber count.
    require_finite_errors : bool
        Whether eligible errors and norms must be finite. This static keyword
        changes host validation but does not enter a JIT trace.

    Returns
    -------
    values : tuple[Array, ...]
        Canonically converted frozen production evidence arrays.

    Raises
    ------
    TypeError
        If the tuple structure is invalid.
    ValueError
        If shapes, dtypes, point errors, exact-once totals, or norms disagree.
    """
    if (
        not isinstance(production_evidence, tuple)
        or len(production_evidence) != _PRODUCTION_EVIDENCE_COUNT
    ):
        raise TypeError("production evidence has the wrong tuple structure")
    values = tuple(jnp.asarray(value) for value in production_evidence)
    roots, root_errors, phases, points = values[:4]
    component_reports = values[4:9]
    vector_reports = values[9:]
    _raise_if(
        roots.dtype != jnp.dtype(jnp.float64)
        or roots.shape != (fiber_size,)
        or root_errors.dtype != jnp.dtype(jnp.float64)
        or root_errors.shape != (fiber_size,)
        or any(
            value.dtype != jnp.dtype(jnp.complex128)
            or value.shape != (fiber_size, _BRANCH_ROLE_COUNT)
            for value in (phases, points)
        )
        or any(
            value.dtype != jnp.dtype(jnp.float64)
            or value.shape != (fiber_size, _BRANCH_ROLE_COUNT)
            for value in component_reports
        )
        or any(
            value.dtype != jnp.dtype(jnp.float64) or value.shape != ()
            for value in vector_reports
        ),
        "production evidence shapes or dtypes are invalid",
    )
    _raise_if(
        any(
            not _normal_or_zero(value)
            for value in (roots, root_errors, phases, points)
        )
        or bool(jnp.any(root_errors < 0.0)),
        "frozen roots, root errors, phases, and branch points must be finite "
        "normal-or-zero",
    )
    _raise_if(
        any(
            bool(jnp.any(jnp.isnan(value)))
            or bool(jnp.any(value < 0.0))
            or bool(has_subnormal_components(value))
            for value in (*component_reports, *vector_reports)
        ),
        "amplitude errors and norms must be nonnegative and non-subnormal",
    )
    production_errors, state_errors, total_errors, point_norms, total_norms = (
        np.asarray(value) for value in component_reports
    )
    production_l2, total_error_l2, exact_state_l2 = (
        float(np.asarray(value)) for value in vector_reports
    )
    if require_finite_errors:
        _raise_if(
            any(
                not _normal_or_zero(value)
                for value in (*component_reports, *vector_reports)
            ),
            "eligible errors and norms must be finite normal-or-zero",
        )
        point_values = np.asarray(points)
        for fiber in range(fiber_size):
            for role in range(_BRANCH_ROLE_COUNT):
                point = complex(point_values[fiber, role])
                expected_error = _point_rectangle_error_upper(
                    point,
                    defining_branches[role],
                    fiber,
                )
                _raise_if(
                    float(production_errors[fiber, role]) != expected_error,
                    "production-to-x error must equal the direct rectangle "
                    "distance",
                )
                expected_total = _sum_upper(
                    expected_error,
                    float(state_errors[fiber, role]),
                )
                _raise_if(
                    float(total_errors[fiber, role]) != expected_total,
                    "exact-state amplitude error must compose each route once",
                )
                expected_point_norm = _complex_norm_upper(point)
                _raise_if(
                    float(point_norms[fiber, role]) != expected_point_norm,
                    "production amplitude norm must match the frozen point",
                )
                expected_total_norm = _sum_upper(
                    expected_point_norm,
                    expected_total,
                )
                _raise_if(
                    float(total_norms[fiber, role]) != expected_total_norm,
                    "exact-state amplitude norm must add its total error once",
                )
        expected_production_l2 = _complex_vector_l2_upper(
            point_values[:, _PREDICTION_BRANCH_ROLE]
        )
        expected_total_error_l2 = _real_vector_l2_upper(
            total_errors[:, _PREDICTION_BRANCH_ROLE]
        )
        _raise_if(
            production_l2 != expected_production_l2,
            "production prediction l2 norm must reduce branch role zero",
        )
        _raise_if(
            total_error_l2 != expected_total_error_l2,
            "prediction error l2 norm must reduce total errors once",
        )
        _raise_if(
            exact_state_l2
            != _sum_upper(expected_production_l2, expected_total_error_l2),
            "exact-state prediction norm must add its vector error once",
        )
    return values


def _validate_plane_mismatch_bounds(
    bounds: _PlaneMismatchBounds,
    fiber_size: int,
    *,
    require_finite: bool,
) -> tuple[Array, Array, Array]:
    """PRIVATE: Validate the three distinct plane-mismatch bounds.

    Parameters
    ----------
    bounds : _PlaneMismatchBounds
        Submitted-x forcing, projection ``||D0||B``, and total ``E_f`` bounds.
    fiber_size : int
        Scoped transverse-fiber count.
    require_finite : bool
        Whether eligible bounds must be finite. This static keyword changes
        host validation but does not enter a JIT trace.

    Returns
    -------
    submitted : Array
        Submitted-state forcing mismatch bounds.
    transfer : Array
        Projection state-radius transfer mismatch bounds.
    total : Array
        Exact-state total projection mismatch bounds.

    Raises
    ------
    TypeError
        If the tuple structure is invalid.
    ValueError
        If shapes, dtypes, range, or exact-once composition disagree.
    """
    if (
        not isinstance(bounds, tuple)
        or len(bounds) != _PLANE_MISMATCH_BOUND_COUNT
    ):
        raise TypeError("plane mismatch bounds have the wrong tuple structure")
    values = tuple(jnp.asarray(value) for value in bounds)
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.float64)
            or value.shape != (fiber_size, _BRANCH_ROLE_COUNT)
            for value in values
        ),
        "plane mismatch bounds must be float64 (f, 2) arrays",
    )
    _raise_if(
        any(
            bool(jnp.any(jnp.isnan(value)))
            or bool(jnp.any(value < 0.0))
            or bool(has_subnormal_components(value))
            for value in values
        ),
        "plane mismatch bounds must be nonnegative and non-subnormal",
    )
    if require_finite:
        _raise_if(
            any(not _normal_or_zero(value) for value in values),
            "eligible plane mismatch bounds must be finite normal-or-zero",
        )
        submitted, transfer, total = (np.asarray(value) for value in values)
        for fiber in range(fiber_size):
            for role in range(_BRANCH_ROLE_COUNT):
                expected = _sum_upper(
                    float(submitted[fiber, role]),
                    float(transfer[fiber, role]),
                )
                _raise_if(
                    float(total[fiber, role]) != expected,
                    "projection total mismatch must compose each term once",
                )
    return values[0], values[1], values[2]


def _validate_root_realization(
    root: GalerkinLocalVacuumRootCertificate | None,
    realization: float,
    realization_error: float,
    phases: np.ndarray,
    *,
    eligible: bool,
) -> None:
    """PRIVATE: Validate one rounded positive root and phase realization.

    Parameters
    ----------
    root : GalerkinLocalVacuumRootCertificate | None
        Replayed strict physical root evidence or a typed missing root.
    realization : float
        Frozen nearest-binary64 midpoint realization.
    realization_error : float
        Outward distance from the realization to the full root interval.
    phases : np.ndarray
        Frozen inner/outer physical phase realizations.
    eligible : bool
        Whether full branch evidence is eligible. This static keyword changes
        host validation but does not enter a JIT trace.

    Raises
    ------
    ValueError
        If classification, sentinel, rounding, error, or phases disagree.
    """
    phase_sentinel = bool(np.all(phases == 0.0))
    _raise_if(
        bool(np.any(phases == 0.0)) and not phase_sentinel,
        "physical phases must be both nonzero or one typed sentinel pair",
    )
    if (
        root is None
        or root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED
    ):
        _raise_if(
            realization != 0.0
            or realization_error != 0.0
            or not phase_sentinel,
            "unclassified roots require zero production sentinels",
        )
        return
    interval = root.root_interval
    if interval is None:
        raise ValueError("classified root has no positive interval")
    if root.classification is GalerkinLocalVacuumRootClass.GRAZING:
        canonical = realization == 0.0 and realization_error == 0.0
    else:
        midpoint = (interval.lower + interval.upper) / 2
        try:
            expected_root = float(midpoint)
        except OverflowError:
            expected_root = 0.0
            finite_normal_root = False
        else:
            finite_normal_root = (
                math.isfinite(expected_root)
                and expected_root > 0.0
                and _normal_or_zero(jnp.asarray(expected_root))
            )
        if finite_normal_root:
            exact_root = Fraction.from_float(expected_root)
            expected_error = fraction_upper_float(
                max(
                    abs(exact_root - interval.lower),
                    abs(exact_root - interval.upper),
                )
            )
            canonical = (
                realization == expected_root
                and realization_error == expected_error
                and _normal_or_zero(jnp.asarray(expected_error))
            )
        else:
            canonical = False
    sentinel = (
        realization == 0.0 and realization_error == 0.0 and phase_sentinel
    )
    _raise_if(
        eligible and (not canonical or phase_sentinel),
        "eligible root realization/error/phases are not canonical",
    )
    _raise_if(
        not eligible and not sentinel and (not canonical or phase_sentinel),
        "noncertificate root data is neither canonical nor a typed sentinel",
    )


def _make_local_vacuum_terminal_entire_evidence(
    kernel_labels: tuple[str, ...],
    transcripts: tuple[EntireWorkTranscript | None, ...],
    failure_reasons: tuple[EntireEnclosureFailure | None, ...],
    failure_work_counts: tuple[int, ...],
    policies: tuple[int, int, int, int, int],
    helper_attempted: Bool[Array, ""],
    helper_eligible: Bool[Array, ""],
    *,
    helper_evidence_digest: str,
) -> GalerkinLocalVacuumTerminalEntireEvidence:
    """PRIVATE: Construct exact per-kernel entire-helper evidence.

    Parameters
    ----------
    kernel_labels : tuple[str, ...]
        Deterministic ordered phase/exp/phi1/phi2 invocation labels.
    transcripts : tuple[EntireWorkTranscript | None, ...]
        Complete successful-call transcripts at the same labels.
    failure_reasons : tuple[EntireEnclosureFailure | None, ...]
        Typed helper failures at unsuccessful labels.
    failure_work_counts : tuple[int, ...]
        Exact completed-work counts for failed calls and zero for successes.
    policies : tuple[int, int, int, int, int]
        Precision, term, work, range-reduction, and rational-bit policies.
    helper_attempted : Bool[Array, ""]
        Whether the deterministic helper schedule was issued.
    helper_eligible : Bool[Array, ""]
        Whether every labeled helper call completed successfully.
    helper_evidence_digest : str
        Complete ordered helper evidence digest.

    Returns
    -------
    evidence : GalerkinLocalVacuumTerminalEntireEvidence
        Structurally validated aggregate helper transcript.

    Raises
    ------
    TypeError
        If tuple, policy, transcript, or failure types are invalid.
    ValueError
        If policies, labels, success/failure pairing, totals, or digest fail.
    """
    sequences = (
        kernel_labels,
        transcripts,
        failure_reasons,
        failure_work_counts,
    )
    if any(not isinstance(value, tuple) for value in sequences):
        raise TypeError("entire-helper evidence must use tuple storage")
    size = len(kernel_labels)
    _raise_if(
        any(len(value) != size for value in sequences[1:]),
        "entire-helper evidence lengths must be equal",
    )
    _raise_if(
        any(
            not isinstance(value, str) or not value.strip()
            for value in kernel_labels
        )
        or len(set(kernel_labels)) != size,
        "entire-helper kernel labels must be unique nonempty strings",
    )
    if (
        not isinstance(policies, tuple)
        or len(policies) != _ENTIRE_POLICY_COUNT
    ):
        raise TypeError(
            "entire-helper policies have the wrong tuple structure"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in policies
    ):
        raise TypeError("entire-helper policies must be Python integers")
    precision, terms, work, reductions, rational_bits = policies
    _raise_if(
        precision <= 0
        or terms <= 0
        or work <= 0
        or reductions < 0
        or rational_bits <= 1
        or any(value > _MAXIMUM_SIGNED_INT64 for value in policies),
        "entire-helper policies are outside signed-int64 ranges",
    )
    attempted = jnp.asarray(helper_attempted)
    flag = jnp.asarray(helper_eligible)
    _raise_if(
        attempted.dtype != jnp.dtype(jnp.bool_)
        or attempted.shape != ()
        or flag.dtype != jnp.dtype(jnp.bool_)
        or flag.shape != (),
        "helper predicates must be scalar bool",
    )
    attempted_value = bool(np.asarray(attempted))
    _raise_if(
        attempted_value != (size > 0),
        "helper_attempted disagrees with the deterministic call schedule",
    )
    totals = [0, 0, 0, 0, 0, 0]
    failed = False
    for label, transcript, reason, failure_work in zip(
        kernel_labels,
        transcripts,
        failure_reasons,
        failure_work_counts,
        strict=True,
    ):
        if isinstance(failure_work, bool) or not isinstance(failure_work, int):
            raise TypeError(
                "helper failure work counts must be Python integers"
            )
        _raise_if(failure_work < 0, "helper failure work cannot be negative")
        if reason is None:
            if not isinstance(transcript, EntireWorkTranscript):
                raise TypeError("successful helper call requires a transcript")
            _raise_if(
                failure_work != 0,
                "successful helper work sentinel must be zero",
            )
            expected_algorithm = (
                "exact_fraction_complex_exprel_v1"
                if "phi1" in label
                else (
                    "exact_fraction_complex_phi2_v1"
                    if "phi2" in label
                    else "exact_fraction_complex_exp_v1"
                )
            )
            _raise_if(
                transcript.algorithm != expected_algorithm
                or (
                    transcript.precision_bits,
                    transcript.maximum_terms,
                    transcript.maximum_work,
                    transcript.maximum_range_reductions,
                    transcript.maximum_rational_bits,
                )
                != policies,
                "helper transcript algorithm or policy differs from its label",
            )
            counts = (
                transcript.series_terms,
                transcript.range_reductions,
                transcript.root_enclosures,
                transcript.rectangle_products,
                transcript.reciprocal_steps,
                transcript.exact_work_count,
            )
            _raise_if(
                any(value < 0 for value in counts),
                "helper transcript counts cannot be negative",
            )
            totals = [
                left + right
                for left, right in zip(totals, counts, strict=True)
            ]
        else:
            if not isinstance(reason, EntireEnclosureFailure):
                raise TypeError("helper failure has the wrong typed reason")
            _raise_if(
                transcript is not None,
                "failed helper call cannot own a success transcript",
            )
            failed = True
            totals[5] += failure_work
    _raise_if(
        bool(np.asarray(flag)) != (attempted_value and not failed),
        "helper eligibility disagrees with typed per-kernel failures",
    )
    _raise_if(
        not _valid_digest(helper_evidence_digest),
        "helper_evidence_digest must be SHA-256",
    )
    return GalerkinLocalVacuumTerminalEntireEvidence(
        helper_attempted=attempted,
        helper_eligible=flag,
        kernel_labels=kernel_labels,
        transcripts=transcripts,
        failure_reasons=failure_reasons,
        failure_work_counts=failure_work_counts,
        precision_bits=precision,
        maximum_terms=terms,
        maximum_work=work,
        maximum_range_reductions=reductions,
        maximum_rational_bits=rational_bits,
        total_series_terms=totals[0],
        total_range_reductions=totals[1],
        total_root_enclosures=totals[2],
        total_rectangle_products=totals[3],
        total_reciprocal_steps=totals[4],
        total_exact_work_count=totals[5],
        helper_scope=_ENTIRE_HELPER_SCOPE,
        helper_evidence_digest=helper_evidence_digest,
    )


def _make_local_vacuum_branch_evidence(  # noqa: PLR0912,PLR0913,PLR0915
    projection_certificate: GalerkinLocalProjectionDefectCertificate,
    root_certificates: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    propagators: tuple[GalerkinLocalVacuumPropagator | None, ...],
    root_failure_reasons: tuple[_PropagationFailure, ...],
    root_failure_work_counts: tuple[int, ...],
    propagator_failure_reasons: tuple[_PropagationFailure, ...],
    propagator_failure_work_counts: tuple[int, ...],
    entire_evidence: GalerkinLocalVacuumTerminalEntireEvidence,
    cauchy_evidence: _CauchyEvidence,
    branch_rectangles: _BranchRectangles,
    plane_mismatch_bounds: _PlaneMismatchBounds,
    production_evidence: _ProductionEvidence,
    crosscheck_masks: _CrosscheckMasks,
    work: _WorkEvidence,
    eligibility: _BranchEligibility,
    failure_mask: Int64[Array, ""],
    *,
    half_space_dispositions: _HalfSpaceDispositions,
    direct_work_count_exact: str,
    maximum_root_work: int,
    maximum_propagator_interval_work: int,
    maximum_rational_bits: int,
    direct_rational_peak_bits: int,
    direct_rational_work_count_exact: str,
    direct_rational_failure: EntireEnclosureFailure | None,
    hull_algorithm: str,
    hull_attempted_endpoint_count: int,
    hull_completed_endpoint_count: int,
    hull_input_peak_bits: int,
    hull_output_peak_bits: int,
    hull_normal_floor_count: int,
    hull_range_failure: bool,
    hull_evidence_digest: str,
    direct_work_formula: str,
    physical_root_formula: str,
    root_realization_formula: str,
    physical_cauchy_formula: str,
    endpoint_mismatch_formula: str,
    forced_mismatch_formula: str,
    plane_mismatch_bound_formula: str,
    amplitude_error_formula: str,
    amplitude_norm_formula: str,
    helper_policy_digest: str,
    physical_root_identity_digest: str,
    cauchy_evidence_digest: str,
    branch_evidence_digest: str,
) -> GalerkinLocalVacuumBranchEvidence:
    """PRIVATE: Construct and validate one scoped vacuum branch record.

    Parameters
    ----------
    projection_certificate : GalerkinLocalProjectionDefectCertificate
        Canonical projection parent fixing scope, slab, state, and ``E_f``.
    root_certificates : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Ordered physical root evidence or typed missing entries.
    propagators : tuple[GalerkinLocalVacuumPropagator | None, ...]
        Ordered homogeneous propagators or typed missing entries.
    root_failure_reasons : tuple[_PropagationFailure, ...]
        Per-fiber typed root-helper failures, otherwise none.
    root_failure_work_counts : tuple[int, ...]
        Exact completed root-helper work at failures and zero on success.
    propagator_failure_reasons : tuple[_PropagationFailure, ...]
        Per-fiber typed propagator-helper failures, otherwise none or skip.
    propagator_failure_work_counts : tuple[int, ...]
        Exact completed propagator-helper work at failures and zero otherwise.
    entire_evidence : GalerkinLocalVacuumTerminalEntireEvidence
        Exact per-kernel phase and forced-integral helper evidence.
    cauchy_evidence : _CauchyEvidence
        Inner, outer, endpoint, forced, and certified Cauchy rectangles.
    branch_rectangles : _BranchRectangles
        Defining, endpoint, forced, and certified two-role branch rectangles.
    plane_mismatch_bounds : _PlaneMismatchBounds
        Submitted-x, projection state-transfer, and projection total bounds.
    production_evidence : _ProductionEvidence
        Frozen roots, phases, defining branch point, direct/state/total
        amplitude errors, and production/exact-state norm uppers.
    crosscheck_masks : _CrosscheckMasks
        Cauchy and branch endpoint/forced intersection masks.
    work : _WorkEvidence
        Stored direct count and independent policy.
    eligibility : _BranchEligibility
        Host, normal-arithmetic, and branch-evidence predicates.
    failure_mask : Int64[Array, ""]
        Typed branch-construction outcomes.
    half_space_dispositions : _HalfSpaceDispositions
        Per-fiber excluded-branch or grazing-derivative statuses.
    direct_work_count_exact : str
        Canonical arbitrary-precision direct-work transcript.
    maximum_root_work : int
        Independent per-fiber strict root-helper work policy.
    maximum_propagator_interval_work : int
        Independent per-fiber post-entire propagator work policy.
    maximum_rational_bits : int
        Independent source-local exact-rational endpoint bit policy.
    direct_rational_peak_bits : int
        Largest reduced numerator or denominator retained by direct arithmetic
        or submitted to the fixed dyadic-hull boundary.  Hull endpoint
        conversions do not increment ``direct_rational_work_count_exact``;
        their completed work is ``hull_completed_endpoint_count``.
    direct_rational_work_count_exact : str
        Canonical arbitrary-precision direct rational-operation transcript.
    direct_rational_failure : EntireEnclosureFailure | None
        Typed direct-arithmetic or dyadic-hull-input rational-size outcome,
        otherwise none.
    hull_algorithm : str
        Fixed outward normal-binary64 dyadic-hull algorithm identifier.
    hull_attempted_endpoint_count : int
        Number of exact endpoints submitted to the hull converter.
    hull_completed_endpoint_count : int
        Number of endpoints converted, range-checked, and proved enclosing.
    hull_input_peak_bits : int
        Largest numerator or denominator bit length entering the hull.
    hull_output_peak_bits : int
        Largest numerator or denominator bit length after dyadic hulling.
    hull_normal_floor_count : int
        Number of nonzero endpoints underflowed to zero or widened from a
        directed subnormal to the normal floor.
    hull_range_failure : bool
        Whether any attempted hull endpoint lacked a normal finite enclosure.
    hull_evidence_digest : str
        Digest of the fixed hull algorithm and aggregate transcript.
    direct_work_formula : str
        Exact linear-pass operation-count declaration.
    physical_root_formula : str
        Authenticated physical LVT.39 reconstruction declaration.
    root_realization_formula : str
        Nearest-midpoint float and outward audit-only root-error declaration.
    physical_cauchy_formula : str
        Carrier-restored, side-oriented Cauchy declaration.
    endpoint_mismatch_formula : str
        Endpoint-minus-propagated-inner LVT.44 declaration.
    forced_mismatch_formula : str
        Independent finite forced-integral declaration.
    plane_mismatch_bound_formula : str
        Separate LVT.55 submitted/``||D0||B``/total declaration.
    amplitude_error_formula : str
        LVT.48/LVT.52--LVT.54 exact-once amplitude-error declaration.
    amplitude_norm_formula : str
        LVT.56 frozen-point plus total-error norm declaration.
    helper_policy_digest : str
        Digest of independently supplied pure-helper policies.
    physical_root_identity_digest : str
        Target, scope, and physical-root identity digest.
    cauchy_evidence_digest : str
        Complete submitted-state Cauchy evidence digest.
    branch_evidence_digest : str
        Complete branch evidence digest.

    Returns
    -------
    evidence : GalerkinLocalVacuumBranchEvidence
        Structurally validated scoped branch evidence.

    Raises
    ------
    TypeError
        If parents, roots, propagators, or dispositions have wrong types.
    ValueError
        If shapes, routes, bounds, work, outcomes, or digests disagree.
    """
    if not isinstance(
        projection_certificate, GalerkinLocalProjectionDefectCertificate
    ):
        raise TypeError("projection_certificate has the wrong carrier type")
    if not isinstance(
        entire_evidence, GalerkinLocalVacuumTerminalEntireEvidence
    ):
        raise TypeError("entire_evidence has the wrong carrier type")
    checked_entire = _make_local_vacuum_terminal_entire_evidence(
        entire_evidence.kernel_labels,
        entire_evidence.transcripts,
        entire_evidence.failure_reasons,
        entire_evidence.failure_work_counts,
        (
            entire_evidence.precision_bits,
            entire_evidence.maximum_terms,
            entire_evidence.maximum_work,
            entire_evidence.maximum_range_reductions,
            entire_evidence.maximum_rational_bits,
        ),
        entire_evidence.helper_attempted,
        entire_evidence.helper_eligible,
        helper_evidence_digest=entire_evidence.helper_evidence_digest,
    )
    _raise_if(
        stored_value_payload(checked_entire)
        != stored_value_payload(entire_evidence),
        "entire_evidence aggregates disagree with per-kernel transcripts",
    )
    fiber_size = projection_certificate.scope_transverse_indices.shape[0]
    structures = (
        (cauchy_evidence, _CAUCHY_EVIDENCE_COUNT, "Cauchy evidence"),
        (
            branch_rectangles,
            _BRANCH_RECTANGLE_COUNT,
            "branch rectangles",
        ),
        (
            plane_mismatch_bounds,
            _PLANE_MISMATCH_BOUND_COUNT,
            "plane mismatch bounds",
        ),
        (crosscheck_masks, _CROSSCHECK_MASK_COUNT, "cross-check masks"),
        (eligibility, _BRANCH_ELIGIBILITY_COUNT, "branch predicates"),
    )
    for values, required, name in structures:
        if not isinstance(values, tuple) or len(values) != required:
            raise TypeError(f"{name} has the wrong tuple structure")
    _raise_if(
        len(root_certificates) != fiber_size
        or len(propagators) != fiber_size
        or len(root_failure_reasons) != fiber_size
        or len(root_failure_work_counts) != fiber_size
        or len(propagator_failure_reasons) != fiber_size
        or len(propagator_failure_work_counts) != fiber_size
        or len(half_space_dispositions) != fiber_size,
        "root, propagator, failure, and disposition evidence must match "
        "fibers",
    )
    _raise_if(
        any(
            not isinstance(value, GalerkinLocalVacuumHalfSpaceDisposition)
            for value in half_space_dispositions
        ),
        "half-space dispositions have the wrong enum type",
    )
    failure = _checked_failure(failure_mask)
    allowed = (
        GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.ROOT_UNCLASSIFIED
        | GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
        | GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    _raise_if(bool(failure & ~allowed), "branch evidence owns unrelated bits")
    _raise_if(
        bool(
            failure
            & GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        )
        != (not bool(projection_certificate.finite_projection_bound_eligible)),
        "projection failure bit disagrees with the parent finite bound",
    )
    _validate_work(work, direct_work_count_exact, failure)
    policies = (
        maximum_root_work,
        maximum_propagator_interval_work,
        maximum_rational_bits,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in policies
    ):
        raise TypeError(
            "root, interval, and rational policies must be integers"
        )
    if isinstance(direct_rational_peak_bits, bool) or not isinstance(
        direct_rational_peak_bits, int
    ):
        raise TypeError("direct_rational_peak_bits must be a Python integer")
    _raise_if(
        any(value <= 0 or value > _MAXIMUM_SIGNED_INT64 for value in policies)
        or maximum_rational_bits <= 1
        or direct_rational_peak_bits < 0,
        "direct rational policy or peak is outside its admitted range",
    )
    try:
        rational_work = int(direct_rational_work_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "direct rational work transcript must be canonical decimal"
        ) from error
    _raise_if(
        rational_work < 0
        or str(rational_work) != direct_rational_work_count_exact,
        "direct rational work transcript must be canonical nonnegative "
        "decimal",
    )
    rational_failed = bool(
        failure
        & GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    if direct_rational_failure is not None and not isinstance(
        direct_rational_failure, EntireEnclosureFailure
    ):
        raise TypeError("direct_rational_failure has the wrong enum type")
    _raise_if(
        rational_failed
        != (
            direct_rational_failure
            is EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
        ),
        "direct rational failure bit disagrees with its typed transcript",
    )
    _raise_if(
        (
            not rational_failed
            and direct_rational_peak_bits > maximum_rational_bits
        )
        or (
            rational_failed
            and direct_rational_peak_bits <= maximum_rational_bits
        ),
        "direct rational peak disagrees with its retained-bit policy",
    )
    hull_counts = (
        hull_attempted_endpoint_count,
        hull_completed_endpoint_count,
        hull_input_peak_bits,
        hull_output_peak_bits,
        hull_normal_floor_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in hull_counts
    ):
        raise TypeError("dyadic hull counts and peaks must be Python integers")
    if not isinstance(hull_algorithm, str):
        raise TypeError("hull_algorithm must be a string")
    if not isinstance(hull_range_failure, bool):
        raise TypeError("hull_range_failure must be a Python bool")
    _raise_if(
        hull_algorithm != _HULL_ALGORITHM,
        "dyadic hull algorithm is not canonical",
    )
    hull_size_failed = hull_input_peak_bits > maximum_rational_bits
    _raise_if(
        any(value < 0 for value in hull_counts)
        or hull_completed_endpoint_count > hull_attempted_endpoint_count
        or (
            not hull_range_failure
            and not hull_size_failed
            and hull_completed_endpoint_count != hull_attempted_endpoint_count
        )
        or (
            hull_size_failed
            and (
                not rational_failed
                or hull_completed_endpoint_count
                >= hull_attempted_endpoint_count
                or (
                    not hull_range_failure
                    and hull_completed_endpoint_count + 1
                    != hull_attempted_endpoint_count
                )
            )
        )
        or hull_normal_floor_count > hull_completed_endpoint_count
        or hull_output_peak_bits > _MAXIMUM_BINARY64_RATIONAL_BITS
        or direct_rational_peak_bits < hull_input_peak_bits,
        "dyadic hull transcript is internally inconsistent",
    )
    _raise_if(
        hull_range_failure
        and not bool(
            failure
            & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        ),
        "dyadic hull range failure requires the arithmetic-range bit",
    )
    expected_hull_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.hull.v1",
            "algorithm": hull_algorithm,
            "maximum_rational_bits": maximum_rational_bits,
            "attempted_endpoints": hull_attempted_endpoint_count,
            "completed_endpoints": hull_completed_endpoint_count,
            "input_peak_bits": hull_input_peak_bits,
            "output_peak_bits": hull_output_peak_bits,
            "normal_floor_count": hull_normal_floor_count,
            "range_failure": hull_range_failure,
        }
    )
    _raise_if(
        hull_evidence_digest != expected_hull_digest,
        "dyadic hull evidence digest disagrees with its transcript",
    )
    helper_failure = (
        GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
    )
    _raise_if(
        bool(failure & helper_failure)
        != (
            bool(entire_evidence.helper_attempted)
            and not bool(entire_evidence.helper_eligible)
        ),
        "entire-helper failure bit disagrees with nested evidence",
    )

    (
        inner_cauchy,
        outer_cauchy,
        endpoint_cauchy,
        forced_cauchy,
        certified_cauchy,
    ) = cauchy_evidence
    (
        defining_branches,
        endpoint_branches,
        forced_branches,
        certified_branches,
    ) = branch_rectangles
    _validate_rectangle_pair(
        defining_branches,
        fiber_size,
        "defining branches",
        require_normal=False,
    )

    roots_classified = True
    propagator_failure = False
    secondary = defining_branches[1]
    skipped_root_failures = (
        GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    for index, (
        root,
        propagator,
        root_reason,
        root_work,
        propagator_reason,
        propagator_work,
        disposition,
    ) in enumerate(
        zip(
            root_certificates,
            propagators,
            root_failure_reasons,
            root_failure_work_counts,
            propagator_failure_reasons,
            propagator_failure_work_counts,
            half_space_dispositions,
            strict=True,
        )
    ):
        admitted_failure_types = (
            EntireEnclosureFailure,
            GalerkinLocalVacuumPropagationFailure,
        )
        for reason, failure_work, name in (
            (root_reason, root_work, "root"),
            (propagator_reason, propagator_work, "propagator"),
        ):
            if reason is not None and not isinstance(
                reason, admitted_failure_types
            ):
                raise TypeError(f"{name} failure has the wrong typed reason")
            if isinstance(failure_work, bool) or not isinstance(
                failure_work, int
            ):
                raise TypeError(
                    f"{name} failure work must be a Python integer"
                )
            _raise_if(
                failure_work < 0,
                f"{name} failure work cannot be negative",
            )
            _raise_if(
                reason is None and failure_work != 0,
                f"{name} reason/work success pairing is inconsistent",
            )
        if root is None:
            roots_classified = False
            skipped_before_root = bool(failure & skipped_root_failures)
            propagator_failure = propagator_failure or root_reason is not None
            _raise_if(
                propagator is not None
                or (root_reason is None and not skipped_before_root)
                or propagator_reason is not None
                or disposition
                is not (
                    GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED
                ),
                "missing root requires a missing propagator and status",
            )
            continue
        _validate_local_vacuum_root_certificate(root)
        _raise_if(
            root_reason is not None,
            "stored root cannot also own a root-helper failure",
        )
        _raise_if(
            root.work_transcript.maximum_work != maximum_root_work
            or root.work_transcript.maximum_rational_bits
            != maximum_rational_bits,
            "root helper policies differ from branch policies",
        )
        if root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED:
            roots_classified = False
            _raise_if(
                propagator is not None
                or propagator_reason is not None
                or disposition
                is not (
                    GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED
                ),
                "unclassified root cannot own a propagator or branch status",
            )
            continue
        if propagator is None:
            propagator_failure = True
            _raise_if(
                propagator_reason is None
                or disposition
                is not (
                    GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED
                ),
                "missing propagator must remain explicitly unclassified",
            )
            continue
        _validate_local_vacuum_propagator(propagator)
        _raise_if(
            propagator_reason is not None,
            "stored propagator cannot also own a helper failure",
        )
        _raise_if(
            propagator.interval_work_transcript.maximum_work
            != maximum_propagator_interval_work
            or propagator.interval_work_transcript.maximum_rational_bits
            != maximum_rational_bits
            or propagator.precision_bits != entire_evidence.precision_bits
            or propagator.maximum_terms != entire_evidence.maximum_terms
            or propagator.maximum_entire_work != entire_evidence.maximum_work
            or propagator.maximum_range_reductions
            != entire_evidence.maximum_range_reductions,
            "propagator policies differ from branch helper policies",
        )
        _raise_if(
            propagator.root_certificate.root_identity_digest
            != root.root_identity_digest
            or propagator.root_certificate.root_evidence_digest
            != root.root_evidence_digest,
            "propagator root digests must match the physical root evidence",
        )
        status = _component_zero_status(secondary, index)
        expected = _expected_half_space_disposition(
            root.classification,
            status,
        )
        _raise_if(
            disposition is not expected,
            "half-space status disagrees with its rectangle",
        )

    _raise_if(
        bool(failure & GalerkinLocalVacuumTerminalFailure.ROOT_UNCLASSIFIED)
        != (not roots_classified),
        "unclassified-root bit disagrees with root evidence",
    )
    _raise_if(
        bool(
            failure
            & GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
        )
        != propagator_failure,
        "root/propagator failure bit disagrees with helper evidence",
    )
    cauchy_mask, branch_mask = (
        jnp.asarray(value) for value in crosscheck_masks
    )
    _validate_rectangle_pair(
        inner_cauchy, fiber_size, "inner Cauchy", require_normal=False
    )
    _validate_rectangle_pair(
        outer_cauchy, fiber_size, "outer Cauchy", require_normal=False
    )
    cauchy_complete, cauchy_disjoint = _validate_crosscheck(
        endpoint_cauchy,
        forced_cauchy,
        certified_cauchy,
        cauchy_mask,
        fiber_size,
        "Cauchy mismatch",
    )
    branch_complete, branch_disjoint = _validate_crosscheck(
        endpoint_branches,
        forced_branches,
        certified_branches,
        branch_mask,
        fiber_size,
        "branch mismatch",
    )
    _raise_if(
        bool(
            failure
            & GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
        )
        != cauchy_disjoint,
        "Cauchy cross-check bit disagrees with its mask",
    )
    _raise_if(
        bool(
            failure
            & GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
        )
        != branch_disjoint,
        "branch cross-check bit disagrees with its mask",
    )

    host_ok, normal_ok, submitted_eligible = (
        bool(np.asarray(value)) for value in eligibility
    )
    _raise_if(
        any(
            jnp.asarray(value).dtype != jnp.dtype(jnp.bool_)
            or jnp.asarray(value).shape != ()
            for value in eligibility
        ),
        "branch predicates must be scalar bool",
    )
    host_failed = bool(
        failure
        & GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
    )
    _raise_if(
        host_failed != (not host_ok),
        "host-arithmetic bit disagrees with branch predicates",
    )
    range_failed = bool(
        failure & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    _raise_if(
        range_failed != (not normal_ok),
        "arithmetic-range bit disagrees with branch predicates",
    )
    fatal = (
        GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.ROOT_UNCLASSIFIED
        | GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
        | GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    expected_eligible = (
        not bool(failure & fatal) and cauchy_complete and branch_complete
    )
    _raise_if(
        submitted_eligible != expected_eligible,
        "branch eligibility disagrees with typed outcomes",
    )
    production_values = _validate_production_evidence(
        production_evidence,
        defining_branches,
        fiber_size,
        require_finite_errors=submitted_eligible,
    )
    submitted_mismatch, projection_transfer, projection_total = (
        _validate_plane_mismatch_bounds(
            plane_mismatch_bounds,
            fiber_size,
            require_finite=submitted_eligible,
        )
    )
    (
        frozen_roots,
        frozen_root_errors,
        physical_phases,
        frozen_points,
        production_errors,
        state_errors,
        total_errors,
        point_norms,
        total_norms,
        production_l2,
        total_error_l2,
        exact_state_l2,
    ) = production_values
    frozen_root_values = np.asarray(frozen_roots)
    frozen_root_error_values = np.asarray(frozen_root_errors)
    phase_values = np.asarray(physical_phases)
    for index, root in enumerate(root_certificates):
        _validate_root_realization(
            root,
            float(frozen_root_values[index]),
            float(frozen_root_error_values[index]),
            phase_values[index],
            eligible=submitted_eligible,
        )
    if submitted_eligible:
        for pair in (
            inner_cauchy,
            outer_cauchy,
            endpoint_cauchy,
            forced_cauchy,
            certified_cauchy,
            defining_branches,
            endpoint_branches,
            forced_branches,
            certified_branches,
        ):
            _validate_rectangle_pair(
                pair, fiber_size, "eligible branch", require_normal=True
            )
    for text, name in (
        (direct_work_formula, "direct_work_formula"),
        (physical_root_formula, "physical_root_formula"),
        (root_realization_formula, "root_realization_formula"),
        (physical_cauchy_formula, "physical_cauchy_formula"),
        (endpoint_mismatch_formula, "endpoint_mismatch_formula"),
        (forced_mismatch_formula, "forced_mismatch_formula"),
        (plane_mismatch_bound_formula, "plane_mismatch_bound_formula"),
        (amplitude_error_formula, "amplitude_error_formula"),
        (amplitude_norm_formula, "amplitude_norm_formula"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (hull_evidence_digest, "hull_evidence_digest"),
        (helper_policy_digest, "helper_policy_digest"),
        (physical_root_identity_digest, "physical_root_identity_digest"),
        (cauchy_evidence_digest, "cauchy_evidence_digest"),
        (branch_evidence_digest, "branch_evidence_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")

    return GalerkinLocalVacuumBranchEvidence(
        root_certificates=root_certificates,
        propagators=propagators,
        root_failure_reasons=root_failure_reasons,
        root_failure_work_counts=root_failure_work_counts,
        propagator_failure_reasons=propagator_failure_reasons,
        propagator_failure_work_counts=propagator_failure_work_counts,
        entire_evidence=entire_evidence,
        inner_cauchy_rectangles=inner_cauchy,
        outer_cauchy_rectangles=outer_cauchy,
        endpoint_cauchy_mismatch_rectangles=endpoint_cauchy,
        forced_cauchy_mismatch_rectangles=forced_cauchy,
        certified_cauchy_mismatch_rectangles=certified_cauchy,
        defining_branch_rectangles=defining_branches,
        endpoint_branch_mismatch_rectangles=endpoint_branches,
        forced_branch_mismatch_rectangles=forced_branches,
        certified_branch_mismatch_rectangles=certified_branches,
        submitted_state_branch_mismatch_upper_bounds=submitted_mismatch,
        projection_state_transfer_branch_mismatch_upper_bounds=(
            projection_transfer
        ),
        projection_total_branch_mismatch_upper_bounds=projection_total,
        frozen_positive_root_realizations=frozen_roots,
        frozen_positive_root_error_bounds=frozen_root_errors,
        physical_phase_realizations=physical_phases,
        frozen_defining_branch_points=frozen_points,
        production_to_submitted_amplitude_error_bounds=production_errors,
        state_radius_amplitude_error_bounds=state_errors,
        exact_state_total_amplitude_error_bounds=total_errors,
        production_amplitude_norm_upper_bounds=point_norms,
        exact_state_amplitude_norm_upper_bounds=total_norms,
        production_prediction_l2_norm_upper_bound=production_l2,
        exact_state_prediction_error_l2_upper_bound=total_error_l2,
        exact_state_prediction_l2_norm_upper_bound=exact_state_l2,
        cauchy_crosscheck_mask=cauchy_mask,
        branch_crosscheck_mask=branch_mask,
        direct_work_count=jnp.asarray(work[0]),
        maximum_direct_terms=jnp.asarray(work[1]),
        host_binary64_eligible=jnp.asarray(eligibility[0]),
        normal_arithmetic_eligible=jnp.asarray(eligibility[1]),
        branch_evidence_eligible=jnp.asarray(eligibility[2]),
        failure_mask=jnp.asarray(failure_mask),
        half_space_dispositions=half_space_dispositions,
        prediction_branch_role=_PREDICTION_BRANCH_ROLE,
        prediction_branch_role_scope=_PREDICTION_BRANCH_ROLE_SCOPE,
        direct_work_count_exact=direct_work_count_exact,
        maximum_root_work=maximum_root_work,
        maximum_propagator_interval_work=maximum_propagator_interval_work,
        maximum_rational_bits=maximum_rational_bits,
        direct_rational_peak_bits=direct_rational_peak_bits,
        direct_rational_work_count_exact=direct_rational_work_count_exact,
        direct_rational_failure=direct_rational_failure,
        hull_algorithm=hull_algorithm,
        hull_attempted_endpoint_count=hull_attempted_endpoint_count,
        hull_completed_endpoint_count=hull_completed_endpoint_count,
        hull_input_peak_bits=hull_input_peak_bits,
        hull_output_peak_bits=hull_output_peak_bits,
        hull_normal_floor_count=hull_normal_floor_count,
        hull_range_failure=hull_range_failure,
        hull_evidence_digest=hull_evidence_digest,
        direct_work_formula=direct_work_formula.strip(),
        physical_root_formula=physical_root_formula.strip(),
        root_realization_formula=root_realization_formula.strip(),
        root_realization_scope=_ROOT_REALIZATION_SCOPE,
        physical_cauchy_formula=physical_cauchy_formula.strip(),
        endpoint_mismatch_formula=endpoint_mismatch_formula.strip(),
        forced_mismatch_formula=forced_mismatch_formula.strip(),
        plane_mismatch_bound_formula=plane_mismatch_bound_formula.strip(),
        amplitude_error_formula=amplitude_error_formula.strip(),
        amplitude_norm_formula=amplitude_norm_formula.strip(),
        production_to_submitted_amplitude_scope=(
            _PRODUCTION_TO_SUBMITTED_AMPLITUDE_SCOPE
        ),
        state_radius_amplitude_scope=_STATE_RADIUS_AMPLITUDE_SCOPE,
        exact_state_amplitude_scope=_EXACT_STATE_AMPLITUDE_SCOPE,
        submitted_plane_mismatch_scope=_SUBMITTED_PLANE_MISMATCH_SCOPE,
        projection_state_transfer_mismatch_scope=(
            _PROJECTION_STATE_TRANSFER_MISMATCH_SCOPE
        ),
        projection_total_mismatch_scope=_PROJECTION_TOTAL_MISMATCH_SCOPE,
        helper_policy_digest=helper_policy_digest,
        physical_root_identity_digest=physical_root_identity_digest,
        cauchy_evidence_digest=cauchy_evidence_digest,
        branch_evidence_digest=branch_evidence_digest,
    )


def _make_local_vacuum_cut_balance(  # noqa: PLR0913,PLR0915
    reports: tuple[Float64[Array, ""], ...],
    work: _WorkEvidence,
    flags: _CutEligibility,
    failure_mask: Int64[Array, ""],
    *,
    direct_work_count_exact: str,
    maximum_rational_bits: int,
    direct_rational_peak_bits: int,
    direct_rational_work_count_exact: str,
    direct_rational_failure: EntireEnclosureFailure | None,
    direct_work_formula: str,
    current_difference_formula: str,
    defect_work_formula: str,
    balance_scope: str,
    cut_balance_digest: str,
) -> GalerkinLocalVacuumCutBalance:
    """PRIVATE: Construct and validate one independent cut-balance record.

    Parameters
    ----------
    reports : tuple[Float64[Array, ""], ...]
        Current, defect-work, and certified intersection endpoints.
    work : _WorkEvidence
        Stored direct count and independent pair policy.
    flags : _CutEligibility
        Host, normal-arithmetic, and cut-balance predicates.
    failure_mask : Int64[Array, ""]
        Typed cut-balance outcomes.
    direct_work_count_exact : str
        Canonical arbitrary-precision direct-work transcript.
    maximum_rational_bits : int
        Independent cut-work exact-rational endpoint bit policy.
    direct_rational_peak_bits : int
        Largest reduced numerator or denominator retained by cut arithmetic.
    direct_rational_work_count_exact : str
        Canonical arbitrary-precision cut rational-operation transcript.
    direct_rational_failure : EntireEnclosureFailure | None
        Typed cut rational-size outcome, otherwise none.
    direct_work_formula : str
        Exact nonsymmetrized pair-operation count declaration.
    current_difference_formula : str
        Side-oriented outer-minus-inner current declaration.
    defect_work_formula : str
        Nonsymmetrized exact ``G diag(d)`` work declaration.
    balance_scope : str
        Explicit full or selected complete-fiber balance scope.
    cut_balance_digest : str
        Complete cut-balance evidence digest.

    Returns
    -------
    balance : GalerkinLocalVacuumCutBalance
        Structurally validated independent cut-balance record.

    Raises
    ------
    TypeError
        If predicate tuple structure is invalid.
    ValueError
        If reports, work, predicates, outcome, or digest disagree.
    """
    _raise_if(
        len(reports) != _CUT_REPORT_COUNT,
        "cut balance requires current, work, and intersection endpoints",
    )
    values = tuple(jnp.asarray(value) for value in reports)
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.float64) or value.shape != ()
            for value in values
        )
        or any(bool(jnp.isnan(value)) for value in values),
        "cut-balance reports must be scalar float64 and not NaN",
    )
    _raise_if(
        any(bool(has_subnormal_components(value)) for value in values),
        "cut-balance reports must contain no subnormal",
    )
    _raise_if(
        bool(values[0] > values[1])
        or bool(values[2] > values[3])
        or bool(values[4] > values[5]),
        "cut-balance intervals must be ordered",
    )
    current_available = bool(jnp.all(jnp.isfinite(jnp.stack(values[:2]))))
    work_available = bool(jnp.all(jnp.isfinite(jnp.stack(values[2:4]))))
    both_routes_available = current_available and work_available
    expected_lower = max(float(values[0]), float(values[2]))
    expected_upper = min(float(values[1]), float(values[3]))
    overlaps = both_routes_available and expected_lower <= expected_upper
    disjoint = both_routes_available and expected_lower > expected_upper
    stored_lower = float(values[4])
    stored_upper = float(values[5])
    if overlaps:
        _raise_if(
            stored_lower != expected_lower or stored_upper != expected_upper,
            "certified cut balance must equal the interval intersection",
        )
    else:
        _raise_if(
            not math.isinf(stored_lower)
            or stored_lower >= 0.0
            or not math.isinf(stored_upper)
            or stored_upper <= 0.0,
            "empty cut balance must use the unbounded sentinel",
        )
    failure = _checked_failure(failure_mask)
    allowed = (
        GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    _raise_if(bool(failure & ~allowed), "cut balance owns unrelated bits")
    _validate_work(work, direct_work_count_exact, failure)
    if isinstance(maximum_rational_bits, bool) or not isinstance(
        maximum_rational_bits, int
    ):
        raise TypeError("maximum_rational_bits must be a Python integer")
    if isinstance(direct_rational_peak_bits, bool) or not isinstance(
        direct_rational_peak_bits, int
    ):
        raise TypeError("direct_rational_peak_bits must be a Python integer")
    _raise_if(
        maximum_rational_bits <= 1
        or maximum_rational_bits > _MAXIMUM_SIGNED_INT64
        or direct_rational_peak_bits < 0,
        "cut rational policy or peak is outside its admitted range",
    )
    try:
        rational_work = int(direct_rational_work_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "cut rational work transcript must be canonical decimal"
        ) from error
    _raise_if(
        rational_work < 0
        or str(rational_work) != direct_rational_work_count_exact,
        "cut rational work transcript must be canonical nonnegative decimal",
    )
    rational_failed = bool(
        failure
        & GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    if direct_rational_failure is not None and not isinstance(
        direct_rational_failure, EntireEnclosureFailure
    ):
        raise TypeError("direct_rational_failure has the wrong enum type")
    _raise_if(
        rational_failed
        != (
            direct_rational_failure
            is EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
        ),
        "cut rational failure bit disagrees with its typed transcript",
    )
    _raise_if(
        (
            not rational_failed
            and direct_rational_peak_bits > maximum_rational_bits
        )
        or (
            rational_failed
            and direct_rational_peak_bits <= maximum_rational_bits
        ),
        "cut rational peak disagrees with its retained-bit policy",
    )
    if not isinstance(flags, tuple) or len(flags) != _CUT_ELIGIBILITY_COUNT:
        raise TypeError(
            "cut-balance predicates have the wrong tuple structure"
        )
    host, normal, submitted_eligible = (
        bool(np.asarray(value)) for value in flags
    )
    _raise_if(
        any(
            jnp.asarray(value).dtype != jnp.dtype(jnp.bool_)
            or jnp.asarray(value).shape != ()
            for value in flags
        ),
        "cut-balance predicates must be scalar bool",
    )
    _raise_if(
        bool(
            failure
            & GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
        )
        != disjoint,
        "cut-balance cross-check bit disagrees with its intervals",
    )
    host_failed = bool(
        failure
        & GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
    )
    _raise_if(host_failed != (not host), "cut-balance host bit disagrees")
    range_failed = bool(
        failure & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    _raise_if(
        range_failed != (not normal),
        "cut-balance range bit disagrees with its predicate",
    )
    fatal = (
        GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    expected_eligible = not bool(failure & fatal)
    _raise_if(
        submitted_eligible != expected_eligible,
        "cut-balance eligibility disagrees with typed outcomes",
    )
    if submitted_eligible:
        _raise_if(
            any(not _normal_or_zero(value) for value in values),
            "eligible cut-balance reports must be finite normal-or-zero",
        )
    for text, name in (
        (direct_work_formula, "direct_work_formula"),
        (current_difference_formula, "current_difference_formula"),
        (defect_work_formula, "defect_work_formula"),
        (balance_scope, "balance_scope"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    _raise_if(
        not _valid_digest(cut_balance_digest),
        "cut_balance_digest must be SHA-256",
    )
    return GalerkinLocalVacuumCutBalance(
        current_difference_lower_bound=values[0],
        current_difference_upper_bound=values[1],
        negative_defect_work_lower_bound=values[2],
        negative_defect_work_upper_bound=values[3],
        certified_balance_lower_bound=values[4],
        certified_balance_upper_bound=values[5],
        direct_work_count=jnp.asarray(work[0]),
        maximum_direct_pairs=jnp.asarray(work[1]),
        host_binary64_eligible=jnp.asarray(flags[0]),
        normal_arithmetic_eligible=jnp.asarray(flags[1]),
        cut_balance_eligible=jnp.asarray(flags[2]),
        failure_mask=jnp.asarray(failure_mask),
        direct_work_count_exact=direct_work_count_exact,
        maximum_rational_bits=maximum_rational_bits,
        direct_rational_peak_bits=direct_rational_peak_bits,
        direct_rational_work_count_exact=direct_rational_work_count_exact,
        direct_rational_failure=direct_rational_failure,
        direct_work_formula=direct_work_formula.strip(),
        current_difference_formula=current_difference_formula.strip(),
        defect_work_formula=defect_work_formula.strip(),
        balance_scope=balance_scope.strip(),
        cut_balance_digest=cut_balance_digest,
    )


def _make_local_vacuum_terminal_certificate(  # noqa: PLR0912,PLR0913,PLR0915
    projection_certificate: GalerkinLocalProjectionDefectCertificate,
    inner_current_diagnostic: GalerkinLocalCoordinateCauchyCurrent,
    outer_current_diagnostic: GalerkinLocalCoordinateCauchyCurrent,
    branch_evidence: GalerkinLocalVacuumBranchEvidence,
    cut_balance: GalerkinLocalVacuumCutBalance,
    coordinates: tuple[Float64[Array, ""], Float64[Array, ""]],
    eligibility: _TerminalEligibility,
    failure_mask: Int64[Array, ""],
    *,
    terminal_axis: int,
    terminal_side: GalerkinTerminalSide,
    terminal_scope: GalerkinLocalTerminalScope,
    disposition: GalerkinLocalVacuumTerminalDisposition,
    target_digest: str,
    source_digest: str,
    state_identity_digest: str,
    projection_identity_digest: str,
    parent_projection_certificate_digest: str,
    inner_terminal_evidence_digest: str,
    outer_terminal_evidence_digest: str,
    branch_evidence_digest: str,
    cut_balance_digest: str,
    terminal_identity_digest: str,
    terminal_evidence_digest: str,
) -> GalerkinLocalVacuumTerminalCertificate:
    """PRIVATE: Construct and validate one composed vacuum terminal record.

    Parameters
    ----------
    projection_certificate : GalerkinLocalProjectionDefectCertificate
        Fully replayed projection parent.
    inner_current_diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt inner-plane L7 submitted-state diagnostic.
    outer_current_diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt terminal-side outer-plane L7 diagnostic.
    branch_evidence : GalerkinLocalVacuumBranchEvidence
        Physical roots, propagators, Cauchy data, and mismatch evidence.
    cut_balance : GalerkinLocalVacuumCutBalance
        Independent nonsymmetrized defect-work/current cross-check.
    coordinates : tuple[Float64[Array, ""], Float64[Array, ""]]
        Defining outer and comparison inner exact stored coordinates.
    eligibility : _TerminalEligibility
        Current-diagnostic, current-operator, submitted-action, and final
        vacuum-branch predicates.
    failure_mask : Int64[Array, ""]
        Complete simultaneous terminal outcomes.
    terminal_axis : int
        Target-owned physical terminal axis.
    terminal_side : GalerkinTerminalSide
        Target-owned outward terminal side.
    terminal_scope : GalerkinLocalTerminalScope
        Full or selected complete transverse-fiber scope.
    disposition : GalerkinLocalVacuumTerminalDisposition
        Requested honest continuation claim.
    target_digest : str
        Bound local target identity digest.
    source_digest : str
        Bound represented-source identity digest.
    state_identity_digest : str
        Bound submitted-state identity digest.
    projection_identity_digest : str
        Bound projection identity digest.
    parent_projection_certificate_digest : str
        Bound complete projection evidence digest.
    inner_terminal_evidence_digest : str
        Bound inner L7 diagnostic evidence digest.
    outer_terminal_evidence_digest : str
        Bound outer L7 diagnostic evidence digest.
    branch_evidence_digest : str
        Bound branch evidence digest.
    cut_balance_digest : str
        Bound cut-balance evidence digest.
    terminal_identity_digest : str
        Scope, slab, disposition, and state identity digest.
    terminal_evidence_digest : str
        Complete vacuum-terminal evidence digest.

    Returns
    -------
    certificate : GalerkinLocalVacuumTerminalCertificate
        Structurally validated composed vacuum terminal record.

    Raises
    ------
    TypeError
        If parents, enums, or nested evidence have wrong carrier types.
    ValueError
        If parent, scope, plane, outcome, eligibility, or digest disagrees.
    """
    if not isinstance(
        projection_certificate, GalerkinLocalProjectionDefectCertificate
    ):
        raise TypeError("projection_certificate has the wrong carrier type")
    if not isinstance(
        inner_current_diagnostic, GalerkinLocalCoordinateCauchyCurrent
    ) or not isinstance(
        outer_current_diagnostic, GalerkinLocalCoordinateCauchyCurrent
    ):
        raise TypeError(
            "both plane diagnostics must use local current carriers"
        )
    if not isinstance(branch_evidence, GalerkinLocalVacuumBranchEvidence):
        raise TypeError("branch_evidence has the wrong carrier type")
    if not isinstance(cut_balance, GalerkinLocalVacuumCutBalance):
        raise TypeError("cut_balance has the wrong carrier type")
    if not isinstance(terminal_side, GalerkinTerminalSide):
        raise TypeError("terminal_side has the wrong enum type")
    if not isinstance(terminal_scope, GalerkinLocalTerminalScope):
        raise TypeError("terminal_scope has the wrong enum type")
    if not isinstance(disposition, GalerkinLocalVacuumTerminalDisposition):
        raise TypeError("disposition has the wrong enum type")

    zero_slab = projection_certificate.zero_slab_certificate
    represented = zero_slab.represented_source_certificate
    source = represented.source
    target = source.target
    stability = projection_certificate.stability_result
    expected_axis = target.acquisition.terminal_axis
    expected_side = target.acquisition.terminal_side
    _raise_if(
        isinstance(terminal_axis, bool)
        or not isinstance(terminal_axis, int)
        or terminal_axis != expected_axis
        or terminal_side is not expected_side
        or terminal_scope is not projection_certificate.projection_scope,
        "terminal axis, side, and scope must match the projection parent",
    )

    coordinate_values = tuple(jnp.asarray(value) for value in coordinates)
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.float64) or value.shape != ()
            for value in coordinate_values
        ),
        "terminal coordinates must be scalar float64",
    )
    _raise_if(
        any(not _normal_or_zero(value) for value in coordinate_values),
        "terminal coordinates must be finite normal-or-zero",
    )
    lower = np.asarray(zero_slab.slab_lower_coordinate)
    upper = np.asarray(zero_slab.slab_upper_coordinate)
    expected_outer, expected_inner = (
        (upper, lower)
        if terminal_side is GalerkinTerminalSide.POSITIVE
        else (lower, upper)
    )
    _raise_if(
        not np.array_equal(np.asarray(coordinate_values[0]), expected_outer)
        or not np.array_equal(
            np.asarray(coordinate_values[1]), expected_inner
        ),
        "defining/comparison planes must be the terminal-side slab endpoints",
    )

    projection_target_payload = stored_value_payload(target)
    state = np.asarray(stability.solve_result.field)
    diagnostics = (inner_current_diagnostic, outer_current_diagnostic)
    expected_coordinates = (expected_inner, expected_outer)
    for diagnostic, expected_coordinate in zip(
        diagnostics, expected_coordinates, strict=True
    ):
        action = diagnostic.action_enclosure
        operator = action.certificate
        _raise_if(
            stored_value_payload(operator.target) != projection_target_payload
            or not np.array_equal(np.asarray(action.submitted_field), state)
            or operator.current_scope is not terminal_scope
            or not np.array_equal(
                np.asarray(operator.scope_transverse_indices),
                np.asarray(projection_certificate.scope_transverse_indices),
            )
            or not np.array_equal(
                np.asarray(operator.state_to_fiber_rows),
                np.asarray(projection_certificate.state_to_fiber_rows),
            )
            or not np.array_equal(
                np.asarray(operator.selected_state_mask),
                np.asarray(projection_certificate.selected_state_mask),
            )
            or not np.array_equal(
                np.asarray(operator.terminal_plane_coordinate),
                expected_coordinate,
            ),
            "plane diagnostic target, state, scope, mapping, or coordinate "
            "differs",
        )
    _raise_if(
        not np.array_equal(
            np.asarray(inner_current_diagnostic.maximum_direct_pairs),
            np.asarray(outer_current_diagnostic.maximum_direct_pairs),
        ),
        "both plane diagnostics must bind one terminal work policy",
    )

    flags = tuple(jnp.asarray(value) for value in eligibility)
    if (
        not isinstance(eligibility, tuple)
        or len(eligibility) != _TERMINAL_ELIGIBILITY_COUNT
    ):
        raise TypeError("terminal predicates have the wrong tuple structure")
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.bool_) or value.shape != ()
            for value in flags
        ),
        "terminal predicates must be scalar bool",
    )
    diagnostic_ready = all(
        bool(value.current_diagnostic_eligible) for value in diagnostics
    )
    operator_ready = all(
        bool(value.action_enclosure.certificate.current_operator_eligible)
        for value in diagnostics
    )
    action_ready = all(
        bool(value.action_enclosure.current_action_eligible)
        for value in diagnostics
    )
    _raise_if(
        bool(flags[0]) != diagnostic_ready
        or bool(flags[1]) != operator_ready
        or bool(flags[2]) != action_ready,
        "diagnostic/operator/action predicates disagree with both planes",
    )

    failure = _checked_failure(failure_mask)
    branch_failure = _checked_failure(branch_evidence.failure_mask)
    cut_failure = _checked_failure(cut_balance.failure_mask)
    expected_failure = branch_failure | cut_failure
    failure_type = GalerkinLocalVacuumTerminalFailure
    zero_ready = bool(zero_slab.terminal_zero_slab_eligible)
    projection_ready = bool(
        projection_certificate.finite_projection_bound_eligible
    )
    if not zero_ready:
        expected_failure |= (
            GalerkinLocalVacuumTerminalFailure.ZERO_SLAB_NONCERTIFICATE
        )
    if not projection_ready:
        expected_failure |= (
            GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        )
    if not diagnostic_ready:
        diagnostic_failure = failure_type.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        expected_failure |= diagnostic_failure
    if not operator_ready:
        expected_failure |= (
            GalerkinLocalVacuumTerminalFailure.CURRENT_OPERATOR_NONCERTIFICATE
        )
    if not action_ready:
        expected_failure |= (
            GalerkinLocalVacuumTerminalFailure.CURRENT_ACTION_NONCERTIFICATE
        )

    selected_disposition = disposition is (
        GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_TERMINAL_SECTOR
    )
    full_disposition = disposition is (
        GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_SLAB
    )
    scope_matches = (
        not selected_disposition
        or terminal_scope
        is GalerkinLocalTerminalScope.SELECTED_PRETERMINAL_FIBERS
    ) and (
        not full_disposition
        or terminal_scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
    )
    if not scope_matches:
        expected_failure |= (
            GalerkinLocalVacuumTerminalFailure.DISPOSITION_SCOPE_MISMATCH
        )
    native = selected_disposition or full_disposition
    structural = bool(projection_certificate.structural_exact_zero_eligible)
    if native and not structural:
        structural_failure = failure_type.NATIVE_STRUCTURAL_ZERO_UNAVAILABLE
        expected_failure |= structural_failure
    _raise_if(
        failure != expected_failure,
        "terminal failure mask disagrees with parents and disposition",
    )
    disposition_ready = scope_matches and (not native or structural)
    expected_vacuum = (
        zero_ready
        and projection_ready
        and diagnostic_ready
        and operator_ready
        and action_ready
        and bool(branch_evidence.branch_evidence_eligible)
        and bool(jnp.all(branch_evidence.cauchy_crosscheck_mask))
        and bool(jnp.all(branch_evidence.branch_crosscheck_mask))
        and bool(cut_balance.cut_balance_eligible)
        and disposition_ready
    )
    _raise_if(
        bool(flags[3]) != expected_vacuum,
        "vacuum-branch eligibility disagrees with the complete ladder",
    )

    for value, expected, name in (
        (target_digest, target.target_digest, "target_digest"),
        (source_digest, source.source_digest, "source_digest"),
        (
            state_identity_digest,
            stability.result_identity_digest,
            "state_identity_digest",
        ),
        (
            projection_identity_digest,
            projection_certificate.projection_identity_digest,
            "projection_identity_digest",
        ),
        (
            parent_projection_certificate_digest,
            projection_certificate.certificate_digest,
            "parent_projection_certificate_digest",
        ),
        (
            inner_terminal_evidence_digest,
            inner_current_diagnostic.diagnostic_evidence_digest,
            "inner_terminal_evidence_digest",
        ),
        (
            outer_terminal_evidence_digest,
            outer_current_diagnostic.diagnostic_evidence_digest,
            "outer_terminal_evidence_digest",
        ),
        (
            branch_evidence_digest,
            branch_evidence.branch_evidence_digest,
            "branch_evidence_digest",
        ),
        (
            cut_balance_digest,
            cut_balance.cut_balance_digest,
            "cut_balance_digest",
        ),
    ):
        _raise_if(value != expected, f"{name} disagrees with its parent")
    for digest, name in (
        (terminal_identity_digest, "terminal_identity_digest"),
        (terminal_evidence_digest, "terminal_evidence_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")

    return GalerkinLocalVacuumTerminalCertificate(
        projection_certificate=projection_certificate,
        inner_current_diagnostic=inner_current_diagnostic,
        outer_current_diagnostic=outer_current_diagnostic,
        branch_evidence=branch_evidence,
        cut_balance=cut_balance,
        defining_plane_coordinate=coordinate_values[0],
        comparison_plane_coordinate=coordinate_values[1],
        current_diagnostic_eligible=flags[0],
        current_operator_eligible=flags[1],
        current_action_eligible=flags[2],
        vacuum_branch_eligible=flags[3],
        failure_mask=jnp.asarray(failure_mask),
        terminal_axis=terminal_axis,
        terminal_side=terminal_side,
        terminal_scope=terminal_scope,
        disposition=disposition,
        amplitude_dependency_scope=_AMPLITUDE_DEPENDENCY_SCOPE,
        completion_scope=_COMPLETION_SCOPE,
        target_digest=target_digest,
        source_digest=source_digest,
        state_identity_digest=state_identity_digest,
        projection_identity_digest=projection_identity_digest,
        parent_projection_certificate_digest=(
            parent_projection_certificate_digest
        ),
        inner_terminal_evidence_digest=inner_terminal_evidence_digest,
        outer_terminal_evidence_digest=outer_terminal_evidence_digest,
        branch_evidence_digest=branch_evidence_digest,
        cut_balance_digest=cut_balance_digest,
        terminal_identity_digest=terminal_identity_digest,
        terminal_evidence_digest=terminal_evidence_digest,
    )


__all__: list[str] = [
    "GalerkinLocalVacuumBranchEvidence",
    "GalerkinLocalVacuumCutBalance",
    "GalerkinLocalVacuumHalfSpaceDisposition",
    "GalerkinLocalVacuumTerminalCertificate",
    "GalerkinLocalVacuumTerminalDisposition",
    "GalerkinLocalVacuumTerminalEntireEvidence",
    "GalerkinLocalVacuumTerminalFailure",
]
