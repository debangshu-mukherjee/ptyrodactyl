r"""Define disjoint local positive-port and detector evidence carriers.

Extended Summary
----------------
These carriers form the RM-S4 L9 eligibility ladder downstream of one local
vacuum terminal.  A positive port preserves exact branch dispositions; a
passive ideal-pixel record preserves current, coordinate-Jacobian,
quadrature, and aperture factors separately; a detector record combines a
tuple of mutually incoherent modes only after each mode's quadratic form;
and a likelihood record keeps the probability law and pointwise NLL
eligibility distinct.  Derivative eligibility belongs to the separate RM-I1
chart and is deliberately absent here.

The classes are storage records rather than authorities.  Only canonical
producers in :mod:`ptyrodactyl.galerkin.detector` may construct eligible
instances after replaying every parent and rebuilding every digest.

Routine Listings
----------------
:class:`GalerkinLocalCensoredPoissonDetector`
    Store one fixed nonnegative pre-gain censored-count detector map.
:class:`GalerkinLocalCensoredPoissonDetectorInputManifest`
    Bind primitive detector inputs and nested pixel manifests.
:class:`GalerkinLocalCensoredPoissonLikelihood`
    Store one pre-gain censored-Poisson likelihood enclosure.
:class:`GalerkinLocalDetectorCoordinateConvention`
    Select one RM-S4 coordinate and amplitude normalization pair.
:class:`GalerkinLocalDetectorFailure`
    Enumerate simultaneous local detector noncertificate outcomes.
:class:`GalerkinLocalDetectorHelperCall`
    Name one channel-specific censored-Poisson helper invocation.
:class:`GalerkinLocalDetectorHelperFailureEvidence`
    Store one replayable channel-specific helper failure.
:class:`GalerkinLocalDetectorLikelihoodStage`
    Fix the stochastic law before deterministic gain and offset.
:class:`GalerkinLocalDetectorProductionStage`
    Name every stopped-evidence production-point stage.
:class:`GalerkinLocalDetectorRationalInterval`
    Store one exact rational detector interval.
:class:`GalerkinLocalDetectorRealProductionTrace`
    Bind one rounded real production point to its exact raw enclosure.
:class:`GalerkinLocalDetectorWorkTranscript`
    Store bounded exact detector-composition work evidence.
:class:`GalerkinLocalPassivePixelForms`
    Store one mode's positive and passive diagonal ideal-pixel forms.
:class:`GalerkinLocalPassivePixelInputManifest`
    Bind independent upstream replay policy and primitive pixel inputs.
:class:`GalerkinLocalPositivePortBranchDisposition`
    Record each classified fiber's retained and excluded branch treatment.
:class:`GalerkinLocalPositivePortCertificate`
    Store one projected outward positive-port certificate.
:class:`GalerkinLocalPositivePortRoute`
    Select one explicit positive-port branch-disposition route.
"""

from __future__ import annotations

from dataclasses import fields, replace
from enum import IntFlag, StrEnum
from fractions import Fraction
from hashlib import sha256 as bytes_sha256
from math import gcd

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype.typing import cast
from jaxtyping import Array, Bool, Complex128, Float64, Int64

from ptyrodactyl._tools import (
    CensoredPoissonEnclosureFailure,
    CensoredPoissonWorkTranscript,
    EntireEnclosureFailure,
    EntireWorkTranscript,
    fraction_lower_float,
    fraction_upper_float,
    sha256,
    stored_value_payload,
)

from .local_vacuum_propagation_types import (
    GalerkinLocalVacuumRootClass,
    _validate_entire_transcript,
)
from .local_vacuum_terminal_types import (
    GalerkinLocalVacuumHalfSpaceDisposition,
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
)


class GalerkinLocalDetectorCoordinateConvention(StrEnum):
    """Select one RM-S4 coordinate and amplitude normalization pair.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_enums_are_complete_explicit_and_disjoint`
    """

    ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES = (
        "angular_wavenumber_amplitude_in_angular_coordinates"
    )
    ANGULAR_WAVENUMBER_AMPLITUDE_IN_CYCLIC_COORDINATES = (
        "angular_wavenumber_amplitude_in_cyclic_coordinates"
    )
    NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES = (
        "native_cyclic_amplitude_in_cyclic_coordinates"
    )
    ANGULAR_WAVENUMBER_AMPLITUDE_IN_SOLID_ANGLE = (
        "angular_wavenumber_amplitude_in_solid_angle"
    )


class GalerkinLocalPositivePortRoute(StrEnum):
    """Select one explicit positive-port branch-disposition route.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_enums_are_complete_explicit_and_disjoint`
    """

    PROJECTED_OUTWARD_PROPAGATING = "projected_outward_propagating"
    OUTGOING_RADIATION = "outgoing_radiation"


class GalerkinLocalPositivePortBranchDisposition(StrEnum):
    """Record each classified fiber's retained and excluded branch treatment.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_enums_are_complete_explicit_and_disjoint`
    """

    PROPAGATING_OUTWARD_RETAINED_INWARD_EXACT_ZERO = (
        "propagating_outward_retained_inward_exact_zero"
    )
    PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_PROVABLY_NONZERO = (
        "propagating_outward_retained_inward_projected_provably_nonzero"
    )
    PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_UNRESOLVED = (
        "propagating_outward_retained_inward_projected_unresolved"
    )
    EVANESCENT_DECAYING_ZERO_WEIGHT_GROWING_EXACT_ZERO = (
        "evanescent_decaying_zero_weight_growing_exact_zero"
    )
    EVANESCENT_GROWING_REJECTED = "evanescent_growing_rejected"
    GRAZING_CONSTANT_ZERO_WEIGHT_DERIVATIVE_EXACT_ZERO = (
        "grazing_constant_zero_weight_derivative_exact_zero"
    )
    GRAZING_DERIVATIVE_REJECTED = "grazing_derivative_rejected"
    ROOT_UNCLASSIFIED = "root_unclassified"


class GalerkinLocalDetectorLikelihoodStage(StrEnum):
    """Fix the stochastic law before deterministic gain and offset.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_enums_are_complete_explicit_and_disjoint`
    """

    PRE_GAIN_CENSORED_COUNTS = "pre_gain_censored_counts"


class GalerkinLocalDetectorHelperCall(StrEnum):
    """Name one channel-specific censored-Poisson helper invocation.

    :see: :func:`~.test_local_detector_types.\
test_helper_failure_evidence_is_owner_call_and_channel_bound`
    """

    EXACT_STATE_CENSORED_MEAN = "exact_state_censored_mean"
    PRODUCTION_CENSORED_MEAN = "production_censored_mean"
    PRODUCTION_PROBABILITY = "production_probability"
    ADMITTED_HULL_PROBABILITY = "admitted_hull_probability"
    PRODUCTION_NLL = "production_nll"
    ADMITTED_HULL_NLL = "admitted_hull_nll"


class GalerkinLocalDetectorProductionStage(StrEnum):
    """Name every stopped-evidence production-point stage.

    A trace stage records a rounded binary64 point beside the exact rational
    interval from which its audit hull is formed.  The point is allowed to
    lie outside a narrow raw interval; the canonical hull and point-to-raw
    distance retain that rounding discrepancy explicitly.

    :see: :func:`~.test_local_detector_types.\
test_exact_interval_and_production_trace_are_owned_and_digest_bound`
    """

    L8_ROLE_ZERO_AMPLITUDE = "l8_role_zero_amplitude"
    POSITIVE_PORT_AMPLITUDE = "positive_port_amplitude"
    COORDINATE_FACTOR = "coordinate_factor"
    PIXEL_FORM_DIAGONAL = "pixel_form_diagonal"
    MODE_PRODUCTION_QUADRATIC = "mode_production_quadratic"
    MODE_PIXEL_FRACTION = "mode_pixel_fraction"
    ENSEMBLE_WEIGHT = "ensemble_weight"
    INCIDENT_DOSE = "incident_dose"
    IDEAL_ARRIVAL_MEAN = "ideal_arrival_mean"
    PRE_GAIN_RESPONSE_MEAN = "pre_gain_response_mean"
    CENSORED_COUNT_MEAN = "censored_count_mean"
    CENSORED_PROBABILITY = "censored_probability"
    CENSORED_NLL = "censored_nll"
    POST_CENSOR_DIGITIZED_MEAN = "post_censor_digitized_mean"


class GalerkinLocalDetectorFailure(IntFlag):
    """Enumerate simultaneous local detector noncertificate outcomes.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_enums_are_complete_explicit_and_disjoint`
    """

    NONE = 0
    VACUUM_TERMINAL_NONCERTIFICATE = 1 << 0
    ROOT_UNCLASSIFIED = 1 << 1
    PROPAGATING_INWARD_NOT_EXACT_ZERO = 1 << 2
    EVANESCENT_GROWING_NOT_EXACT_ZERO = 1 << 3
    GRAZING_DERIVATIVE_NOT_EXACT_ZERO = 1 << 4
    INCIDENT_FLUX_NONPOSITIVE = 1 << 5
    PIXEL_FORM_NONPOSITIVE = 1 << 7
    PIXEL_FORM_NONPASSIVE = 1 << 8
    PRODUCTION_POINT_HULL_FAILURE = 1 << 10
    ENSEMBLE_WEIGHT_INVALID = 1 << 11
    DOSE_INVALID = 1 << 12
    RESPONSE_NONPOSITIVE = 1 << 13
    RESPONSE_NOT_SUBSTOCHASTIC = 1 << 14
    CALIBRATION_INVALID = 1 << 15
    COUNT_DOMAIN_INVALID = 1 << 16
    POISSON_ENCLOSURE_FAILURE = 1 << 17
    NESTED_HELPER_FAILURE = 1 << 18
    NLL_UNAVAILABLE = 1 << 19
    EXACT_WORK_BUDGET_EXCEEDED = 1 << 21
    EXACT_WORK_COUNT_OVERFLOW = 1 << 22
    RATIONAL_SIZE_LIMIT = 1 << 23
    ARITHMETIC_RANGE_FAILURE = 1 << 24


class GalerkinLocalDetectorRationalInterval(eqx.Module):
    """Store one exact rational detector interval.

    :see: :func:`~.test_local_detector_types.\
test_exact_interval_and_production_trace_are_owned_and_digest_bound`
    """

    lower_numerator: int = eqx.field(static=True)
    lower_denominator: int = eqx.field(static=True)
    upper_numerator: int = eqx.field(static=True)
    upper_denominator: int = eqx.field(static=True)

    @property
    def lower(self) -> Fraction:
        """Return the exact lower endpoint."""
        result: Fraction = Fraction(
            self.lower_numerator, self.lower_denominator
        )
        return result

    @property
    def upper(self) -> Fraction:
        """Return the exact upper endpoint."""
        result: Fraction = Fraction(
            self.upper_numerator, self.upper_denominator
        )
        return result


type _DetectorIntervals = tuple[GalerkinLocalDetectorRationalInterval, ...]
type _OptionalDetectorIntervals = tuple[
    GalerkinLocalDetectorRationalInterval | None, ...
]
type _ModePixelIntervals = tuple[_DetectorIntervals, ...]
type _PixelFormIntervals = tuple[_DetectorIntervals, ...]
type _DetectorIntervalCarrier = GalerkinLocalDetectorRationalInterval

_DETECTOR_WORK_ALGORITHM: str = "exact_fraction_local_detector_v1"
_DETECTOR_NESTED_PARENT_WORK_SCOPE: str = (
    "nested_parent_work_count_exact sums authenticated L8 root, propagator, "
    "entire-helper, branch-direct, cut-direct, and nested L9 detector work; "
    "it excludes unmetered replay control flow"
)
_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_MAXIMUM_SIGNED_INT64: int = (1 << 63) - 1
_MATRIX_DIMENSIONS: int = 2
_SHA256_HEX_LENGTH: int = 64
_TRACE_DTYPE: str = "float64"
_EXP_ENTIRE_ALGORITHM: str = "exact_fraction_real_exp_v1"
_LOG_ENTIRE_ALGORITHM: str = "exact_fraction_real_log_atanh_pow2_v1"
_NO_DERIVATIVE_SCOPE: str = (
    "stopped RM-S4 detector evidence; derivative eligibility belongs only "
    "to an independently invoked RM-I1 chart"
)
_POSITIVE_PORT_BRANCH_SCOPE: str = (
    "L8 defining-plane role 0: outward propagating, decaying evanescent, "
    "or constant grazing field"
)
_POSITIVE_PORT_EXACT_STATE_SCOPE: str = (
    "L8 role-zero production-to-exact-x plus terminal-map state-radius "
    "transfer, composed exactly once"
)
_POSITIVE_PORT_ROOT_AUDIT_SCOPE: str = (
    "root realization affects rounded coordinate/Q points; exact root "
    "intervals widen Q and are never added to role-zero E_a"
)
_POSITIVE_PORT_COMPLETION_SCOPE: str = (
    "projected positive port and strictly stronger outgoing-radiation claim "
    "remain disjoint"
)


class GalerkinLocalDetectorRealProductionTrace(eqx.Module):
    """Bind one rounded real production point to its exact raw enclosure.

    ``point`` is flattened in C order and ``logical_shape`` restores the
    scientific array shape.  A point need not lie in ``raw_intervals``: the
    exact point-to-raw distance and the union hull make that rounding event
    explicit.  A point outside the stored union hull is never admissible.

    :see: :func:`~.test_local_detector_types.\
test_exact_interval_and_production_trace_are_owned_and_digest_bound`
    """

    point: Float64[Array, " n"]
    point_to_raw_absolute_error_upper_bounds: Float64[Array, " n"]
    certified_hull_lower_bounds: Float64[Array, " n"]
    certified_hull_upper_bounds: Float64[Array, " n"]
    raw_intervals: _DetectorIntervals = eqx.field(static=True)
    exact_point_intervals: _DetectorIntervals = eqx.field(static=True)
    stage: GalerkinLocalDetectorProductionStage = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    logical_shape: tuple[int, ...] = eqx.field(static=True)
    point_dtype: str = eqx.field(static=True)
    point_bytes_digest: str = eqx.field(static=True)
    raw_interval_digest: str = eqx.field(static=True)
    trace_digest: str = eqx.field(static=True)


type _DetectorProductionTraces = tuple[
    GalerkinLocalDetectorRealProductionTrace, ...
]


class GalerkinLocalPassivePixelInputManifest(eqx.Module):
    """Bind independent upstream replay policy and primitive pixel inputs.

    :see: :func:`~.test_local_detector_types.\
test_input_manifests_enforce_dtype_hard_caps_and_defer_child_caps`
    """

    maximum_state_error: Float64[Array, ""]
    node_to_pixel: Int64[Array, " f"]
    quadrature_weight_points: Float64[Array, " f"]
    aperture_efficiency_points: Float64[Array, " f"]
    route: GalerkinLocalPositivePortRoute = eqx.field(static=True)
    terminal_disposition: GalerkinLocalVacuumTerminalDisposition = eqx.field(
        static=True
    )
    maximum_stability_direct_pairs: int = eqx.field(static=True)
    maximum_gram_pairs: int = eqx.field(static=True)
    maximum_terminal_direct_pairs: int = eqx.field(static=True)
    maximum_branch_direct_terms: int = eqx.field(static=True)
    maximum_cut_direct_pairs: int = eqx.field(static=True)
    maximum_root_work: int = eqx.field(static=True)
    precision_bits: int = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    maximum_entire_work: int = eqx.field(static=True)
    maximum_range_reductions: int = eqx.field(static=True)
    maximum_interval_work: int = eqx.field(static=True)
    maximum_l8_rational_bits: int = eqx.field(static=True)
    coordinate_convention: GalerkinLocalDetectorCoordinateConvention = (
        eqx.field(static=True)
    )
    quadrature_weight_intervals: _DetectorIntervals = eqx.field(static=True)
    aperture_efficiency_intervals: _DetectorIntervals = eqx.field(static=True)
    pixel_count: int = eqx.field(static=True)
    maximum_detector_work: int = eqx.field(static=True)
    maximum_detector_rational_bits: int = eqx.field(static=True)
    manifest_digest: str = eqx.field(static=True)


class GalerkinLocalCensoredPoissonDetectorInputManifest(eqx.Module):
    """Bind primitive detector inputs and nested pixel manifests.

    :see: :func:`~.test_local_detector_types.\
test_input_manifests_enforce_dtype_hard_caps_and_defer_child_caps`
    """

    pixel_inputs: tuple[GalerkinLocalPassivePixelInputManifest, ...]
    response_matrix: Float64[Array, "r p"]
    pre_gain_background: Float64[Array, " r"]
    deterministic_gain: Float64[Array, " r"]
    electronic_offset: Float64[Array, " r"]
    count_ceilings: Int64[Array, " r"]
    fit_mask: Bool[Array, " r"]
    incident_electron_count_point: Float64[Array, ""]
    ensemble_weight_numerators: tuple[int, ...] = eqx.field(static=True)
    ensemble_weight_denominators: tuple[int, ...] = eqx.field(static=True)
    incident_electron_count_interval: GalerkinLocalDetectorRationalInterval = (
        eqx.field(static=True)
    )
    calibration_provenance: str = eqx.field(static=True)
    maximum_detector_work: int = eqx.field(static=True)
    maximum_detector_rational_bits: int = eqx.field(static=True)
    maximum_count_ceiling: int = eqx.field(static=True)
    maximum_poisson_work: int = eqx.field(static=True)
    maximum_poisson_rational_bits: int = eqx.field(static=True)
    exp_precision_bits: int = eqx.field(static=True)
    maximum_exp_terms: int = eqx.field(static=True)
    maximum_exp_work: int = eqx.field(static=True)
    maximum_exp_range_reductions: int = eqx.field(static=True)
    manifest_digest: str = eqx.field(static=True)


class GalerkinLocalDetectorHelperFailureEvidence(eqx.Module):
    """Store one replayable channel-specific helper failure.

    The call kind and channel index prevent equal-looking failures from being
    swapped.  ``attempted`` and ``planned`` remain exact decimal strings so a
    signed-int64 preflight overflow is itself replayable.  Nested exponential
    or logarithm evidence is kept separately from local Poisson work.

    :see: :func:`~.test_local_detector_types.\
test_helper_failure_evidence_is_owner_call_and_channel_bound`
    """

    call: GalerkinLocalDetectorHelperCall = eqx.field(static=True)
    channel_index: int = eqx.field(static=True)
    failure: CensoredPoissonEnclosureFailure = eqx.field(static=True)
    nested_kernel: str | None = eqx.field(static=True)
    nested_failure: EntireEnclosureFailure | None = eqx.field(static=True)
    prior_exp_transcripts: tuple[EntireWorkTranscript, ...] = eqx.field(
        static=True
    )
    prior_log_transcripts: tuple[EntireWorkTranscript, ...] = eqx.field(
        static=True
    )
    local_exact_work_count_exact: str = eqx.field(static=True)
    nested_exact_work_count_exact: str | None = eqx.field(static=True)
    nested_attempted_exact_work_count_exact: str | None = eqx.field(
        static=True
    )
    planned_exact_work_count_exact: str = eqx.field(static=True)
    attempted_exact_work_count_exact: str = eqx.field(static=True)
    failure_digest: str = eqx.field(static=True)


type _OptionalHelperFailures = tuple[
    GalerkinLocalDetectorHelperFailureEvidence | None, ...
]


class GalerkinLocalDetectorWorkTranscript(eqx.Module):
    """Store bounded exact detector-composition work evidence.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_carrier_schema_exposes_full_staged_evidence_no_gradient`
    """

    algorithm: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    coordinate_factor_count: int = eqx.field(static=True)
    pixel_product_count: int = eqx.field(static=True)
    mode_quadratic_count: int = eqx.field(static=True)
    ensemble_product_count: int = eqx.field(static=True)
    response_product_count: int = eqx.field(static=True)
    production_trace_count: int = eqx.field(static=True)
    hull_endpoint_count: int = eqx.field(static=True)
    nested_production_trace_count: int = eqx.field(static=True)
    nested_hull_endpoint_count: int = eqx.field(static=True)
    exact_work_count: int = eqx.field(static=True)
    rational_peak_bits: int = eqx.field(static=True)
    nested_parent_work_count_exact: str = eqx.field(static=True)
    nested_helper_work_count_exact: str = eqx.field(static=True)
    nested_parent_work_scope: str = eqx.field(static=True)
    planned_exact_work_count_exact: str = eqx.field(static=True)
    attempted_exact_work_count_exact: str = eqx.field(static=True)
    completed_successfully: bool = eqx.field(static=True)
    arithmetic_failure: GalerkinLocalDetectorFailure = eqx.field(static=True)
    scientific_failure: GalerkinLocalDetectorFailure = eqx.field(static=True)
    preflight_failed: bool = eqx.field(static=True)
    count_overflow: bool = eqx.field(static=True)


class GalerkinLocalPositivePortCertificate(eqx.Module):
    """Store one projected outward positive-port certificate.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_carrier_schema_exposes_full_staged_evidence_no_gradient`
    """

    terminal_certificate: GalerkinLocalVacuumTerminalCertificate
    production_amplitudes: Complex128[Array, " f"]
    exact_state_total_amplitude_error_bounds: Float64[Array, " f"]
    production_prediction_l2_norm_upper_bound: Float64[Array, ""]
    exact_state_prediction_error_l2_upper_bound: Float64[Array, ""]
    production_root_realizations: Float64[Array, " f"]
    production_root_error_upper_bounds: Float64[Array, " f"]
    retained_propagating_mask: Bool[Array, " f"]
    zero_weight_mask: Bool[Array, " f"]
    positive_port_eligible: Bool[Array, ""]
    outgoing_radiation_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    production_traces: _DetectorProductionTraces
    exact_root_intervals: tuple[
        GalerkinLocalDetectorRationalInterval | None, ...
    ] = eqx.field(static=True)
    parent_half_space_dispositions: tuple[
        GalerkinLocalVacuumHalfSpaceDisposition, ...
    ] = eqx.field(static=True)
    branch_dispositions: tuple[
        GalerkinLocalPositivePortBranchDisposition, ...
    ] = eqx.field(static=True)
    route: GalerkinLocalPositivePortRoute = eqx.field(static=True)
    branch_role: int = eqx.field(static=True)
    branch_scope: str = eqx.field(static=True)
    exact_state_amplitude_scope: str = eqx.field(static=True)
    root_realization_audit_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    state_identity_digest: str = eqx.field(static=True)
    parent_terminal_identity_digest: str = eqx.field(static=True)
    parent_terminal_evidence_digest: str = eqx.field(static=True)
    port_identity_digest: str = eqx.field(static=True)
    port_evidence_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


class GalerkinLocalPassivePixelForms(eqx.Module):
    """Store one mode's positive and passive diagonal ideal-pixel forms.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_carrier_schema_exposes_full_staged_evidence_no_gradient`
    """

    positive_port: GalerkinLocalPositivePortCertificate
    node_to_pixel: Int64[Array, " f"]
    production_evidence_available: Bool[Array, ""]
    positive_forms_eligible: Bool[Array, ""]
    passive_forms_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    quadrature_weights: Float64[Array, " f"]
    aperture_efficiencies: Float64[Array, " f"]
    production_traces: _DetectorProductionTraces
    current_weight_intervals: _DetectorIntervals = eqx.field(static=True)
    amplitude_scale_interval: GalerkinLocalDetectorRationalInterval = (
        eqx.field(static=True)
    )
    coordinate_jacobian_intervals: _DetectorIntervals = eqx.field(static=True)
    quadrature_weight_intervals: _DetectorIntervals = eqx.field(static=True)
    aperture_efficiency_intervals: _DetectorIntervals = eqx.field(static=True)
    outward_form_diagonal_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    pixel_form_diagonal_intervals: _PixelFormIntervals = eqx.field(static=True)
    outward_minus_pixel_form_diagonal_intervals: _DetectorIntervals = (
        eqx.field(static=True)
    )
    production_outward_quadratic_interval: _DetectorIntervalCarrier = (
        eqx.field(static=True)
    )
    outward_form_norm_upper_interval: _DetectorIntervalCarrier = eqx.field(
        static=True
    )
    outward_production_realization_error_upper_interval: _DetectorIntervalCarrier = eqx.field(  # noqa: E501
        static=True
    )
    outward_state_radius_incremental_error_upper_interval: _DetectorIntervalCarrier = eqx.field(  # noqa: E501
        static=True
    )
    outward_combined_exact_state_error_upper_interval: _DetectorIntervalCarrier = eqx.field(  # noqa: E501
        static=True
    )
    exact_state_outward_flux_interval: _DetectorIntervalCarrier = eqx.field(
        static=True
    )
    production_quadratic_intervals: _DetectorIntervals = eqx.field(static=True)
    pixel_form_norm_upper_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    production_to_exact_x_amplitude_error_interval: _DetectorIntervalCarrier = eqx.field(  # noqa: E501
        static=True
    )
    state_radius_amplitude_error_interval: _DetectorIntervalCarrier = (
        eqx.field(static=True)
    )
    exact_state_amplitude_error_interval: _DetectorIntervalCarrier = eqx.field(
        static=True
    )
    production_amplitude_norm_interval: _DetectorIntervalCarrier = eqx.field(
        static=True
    )
    production_realization_error_upper_intervals: _DetectorIntervals = (
        eqx.field(static=True)
    )
    state_radius_incremental_error_upper_intervals: _DetectorIntervals = (
        eqx.field(static=True)
    )
    combined_exact_state_error_upper_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    exact_state_pixel_flux_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    coordinate_convention: GalerkinLocalDetectorCoordinateConvention = (
        eqx.field(static=True)
    )
    pixel_count: int = eqx.field(static=True)
    work_transcript: GalerkinLocalDetectorWorkTranscript = eqx.field(
        static=True
    )
    coordinate_factor_scope: str = eqx.field(static=True)
    pixel_form_scope: str = eqx.field(static=True)
    lvt56_error_scope: str = eqx.field(static=True)
    passivity_margin_scope: str = eqx.field(static=True)
    no_experimental_validity_scope: str = eqx.field(static=True)
    parent_port_certificate_digest: str = eqx.field(static=True)
    input_manifest_digest: str = eqx.field(static=True)
    pixel_model_identity_digest: str = eqx.field(static=True)
    pixel_model_evidence_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


class GalerkinLocalCensoredPoissonDetector(eqx.Module):
    """Store one fixed nonnegative pre-gain censored-count detector map.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_carrier_schema_exposes_full_staged_evidence_no_gradient`
    """

    pixel_forms: tuple[GalerkinLocalPassivePixelForms, ...]
    response_matrix: Float64[Array, "r p"]
    pre_gain_background: Float64[Array, " r"]
    deterministic_gain: Float64[Array, " r"]
    electronic_offset: Float64[Array, " r"]
    count_ceilings: Int64[Array, " r"]
    fit_mask: Bool[Array, " r"]
    incident_electron_count_point: Float64[Array, ""]
    production_evidence_available: Bool[Array, ""]
    exact_state_censored_mean_evidence_available: Bool[Array, ""]
    production_censored_mean_evidence_available: Bool[Array, ""]
    detector_eligible: Bool[Array, ""]
    likelihood_law_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    production_traces: _DetectorProductionTraces
    mode_target_digests: tuple[str, ...] = eqx.field(static=True)
    mode_source_digests: tuple[str, ...] = eqx.field(static=True)
    mode_state_identity_digests: tuple[str, ...] = eqx.field(static=True)
    mode_state_radius_intervals: _OptionalDetectorIntervals = eqx.field(
        static=True
    )
    mode_state_radius_provenance_digests: tuple[str, ...] = eqx.field(
        static=True
    )
    mode_port_certificate_digests: tuple[str, ...] = eqx.field(static=True)
    mode_pixel_evidence_digests: tuple[str, ...] = eqx.field(static=True)
    mode_state_binding_digest: str = eqx.field(static=True)
    ensemble_weight_numerators: tuple[int, ...] = eqx.field(static=True)
    ensemble_weight_denominators: tuple[int, ...] = eqx.field(static=True)
    incident_reduced_flux_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    mode_exact_state_pixel_flux_intervals: _ModePixelIntervals = eqx.field(
        static=True
    )
    mode_production_quadratic_intervals: _ModePixelIntervals = eqx.field(
        static=True
    )
    mode_pixel_form_norm_upper_intervals: _ModePixelIntervals = eqx.field(
        static=True
    )
    mode_production_to_exact_x_amplitude_error_intervals: _DetectorIntervals = eqx.field(  # noqa: E501
        static=True
    )
    mode_state_radius_amplitude_error_intervals: _DetectorIntervals = (
        eqx.field(static=True)
    )
    mode_exact_state_amplitude_error_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    mode_production_amplitude_norm_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    mode_production_realization_error_upper_intervals: _ModePixelIntervals = (
        eqx.field(static=True)
    )
    mode_state_radius_incremental_error_upper_intervals: _ModePixelIntervals = eqx.field(  # noqa: E501
        static=True
    )
    mode_combined_exact_state_error_upper_intervals: _ModePixelIntervals = (
        eqx.field(static=True)
    )
    mode_outward_passivity_margin_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    mode_pixel_fraction_intervals: _ModePixelIntervals = eqx.field(static=True)
    ideal_arrival_mean_intervals: _DetectorIntervals = eqx.field(static=True)
    production_pre_gain_mean_point_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    exact_state_pre_gain_mean_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    censored_mean_intervals: _DetectorIntervals = eqx.field(static=True)
    expected_digitized_mean_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    incident_electron_count: GalerkinLocalDetectorRationalInterval = eqx.field(
        static=True
    )
    likelihood_stage: GalerkinLocalDetectorLikelihoodStage = eqx.field(
        static=True
    )
    work_transcript: GalerkinLocalDetectorWorkTranscript = eqx.field(
        static=True
    )
    censored_mean_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    censored_mean_failures: _OptionalHelperFailures = eqx.field(static=True)
    production_censored_mean_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    production_censored_mean_failures: _OptionalHelperFailures = eqx.field(
        static=True
    )
    maximum_count_ceiling: int = eqx.field(static=True)
    maximum_poisson_work: int = eqx.field(static=True)
    maximum_poisson_rational_bits: int = eqx.field(static=True)
    exp_precision_bits: int = eqx.field(static=True)
    maximum_exp_terms: int = eqx.field(static=True)
    maximum_exp_work: int = eqx.field(static=True)
    maximum_exp_range_reductions: int = eqx.field(static=True)
    flux_normalization_scope: str = eqx.field(static=True)
    ensemble_scope: str = eqx.field(static=True)
    response_scope: str = eqx.field(static=True)
    calibration_provenance: str = eqx.field(static=True)
    no_experimental_validity_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    input_manifest_digest: str = eqx.field(static=True)
    detector_model_identity_digest: str = eqx.field(static=True)
    detector_model_evidence_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


class GalerkinLocalCensoredPoissonLikelihood(eqx.Module):
    """Store one pre-gain censored-Poisson likelihood enclosure.

    :see: :func:`~.test_local_detector_types.\
test_local_detector_carrier_schema_exposes_full_staged_evidence_no_gradient`
    """

    detector: GalerkinLocalCensoredPoissonDetector
    observed_counts: Int64[Array, " r"]
    likelihood_evidence_available: Bool[Array, ""]
    likelihood_law_eligible: Bool[Array, ""]
    nll_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    production_traces: _DetectorProductionTraces
    admitted_pre_gain_mean_hull_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    production_probability_point_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    admitted_hull_probability_intervals: _DetectorIntervals = eqx.field(
        static=True
    )
    fitted_probability_positive_floor_intervals: tuple[
        GalerkinLocalDetectorRationalInterval | None, ...
    ] = eqx.field(static=True)
    production_nll_point_intervals: tuple[
        GalerkinLocalDetectorRationalInterval | None, ...
    ] = eqx.field(static=True)
    admitted_hull_nll_intervals: tuple[
        GalerkinLocalDetectorRationalInterval | None, ...
    ] = eqx.field(static=True)
    total_nll_interval: GalerkinLocalDetectorRationalInterval | None = (
        eqx.field(static=True)
    )
    production_probability_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    production_probability_failures: _OptionalHelperFailures = eqx.field(
        static=True
    )
    admitted_hull_probability_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    admitted_hull_probability_failures: _OptionalHelperFailures = eqx.field(
        static=True
    )
    production_nll_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    production_nll_failures: _OptionalHelperFailures = eqx.field(static=True)
    admitted_hull_nll_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    admitted_hull_nll_failures: _OptionalHelperFailures = eqx.field(
        static=True
    )
    work_transcript: GalerkinLocalDetectorWorkTranscript = eqx.field(
        static=True
    )
    log_precision_bits: int = eqx.field(static=True)
    maximum_log_terms: int = eqx.field(static=True)
    maximum_log_work: int = eqx.field(static=True)
    maximum_log_range_reductions: int = eqx.field(static=True)
    likelihood_scope: str = eqx.field(static=True)
    nll_scope: str = eqx.field(static=True)
    no_derivative_scope: str = eqx.field(static=True)
    parent_detector_certificate_digest: str = eqx.field(static=True)
    likelihood_identity_digest: str = eqx.field(static=True)
    likelihood_evidence_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


type _DetectorDigestCarrier = (
    GalerkinLocalPositivePortCertificate
    | GalerkinLocalPassivePixelForms
    | GalerkinLocalCensoredPoissonDetector
    | GalerkinLocalCensoredPoissonLikelihood
)
type _DetectorInputManifest = (
    GalerkinLocalPassivePixelInputManifest
    | GalerkinLocalCensoredPoissonDetectorInputManifest
)


def _make_local_censored_poisson_detector_candidate(
    **values: object,
) -> GalerkinLocalCensoredPoissonDetector:
    """PRIVATE: Construct one unsealed detector candidate.

    Parameters
    ----------
    **values : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalCensoredPoissonDetector
        Canonical derived result.
    """
    return GalerkinLocalCensoredPoissonDetector(**values)  # type: ignore[arg-type]


def _make_local_censored_poisson_detector_input_manifest_candidate(
    **values: object,
) -> GalerkinLocalCensoredPoissonDetectorInputManifest:
    """PRIVATE: Construct one unsealed detector-input candidate.

    Parameters
    ----------
    **values : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalCensoredPoissonDetectorInputManifest
        Canonical derived result.
    """
    return GalerkinLocalCensoredPoissonDetectorInputManifest(**values)  # type: ignore[arg-type]


def _make_local_censored_poisson_likelihood_candidate(
    **values: object,
) -> GalerkinLocalCensoredPoissonLikelihood:
    """PRIVATE: Construct one unsealed likelihood candidate.

    Parameters
    ----------
    **values : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalCensoredPoissonLikelihood
        Canonical derived result.
    """
    return GalerkinLocalCensoredPoissonLikelihood(**values)  # type: ignore[arg-type]


def _make_local_passive_pixel_forms_candidate(
    **values: object,
) -> GalerkinLocalPassivePixelForms:
    """PRIVATE: Construct one unsealed passive-pixel candidate.

    Parameters
    ----------
    **values : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPassivePixelForms
        Canonical derived result.
    """
    return GalerkinLocalPassivePixelForms(**values)  # type: ignore[arg-type]


def _make_local_passive_pixel_input_manifest_candidate(
    **values: object,
) -> GalerkinLocalPassivePixelInputManifest:
    """PRIVATE: Construct one unsealed pixel-input candidate.

    Parameters
    ----------
    **values : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPassivePixelInputManifest
        Canonical derived result.
    """
    return GalerkinLocalPassivePixelInputManifest(**values)  # type: ignore[arg-type]


def _make_local_positive_port_candidate(
    **values: object,
) -> GalerkinLocalPositivePortCertificate:
    """PRIVATE: Construct one unsealed positive-port candidate.

    Parameters
    ----------
    **values : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPositivePortCertificate
        Canonical derived result.
    """
    return GalerkinLocalPositivePortCertificate(**values)  # type: ignore[arg-type]


def _validate_local_detector_rational_interval(
    interval: object,
) -> GalerkinLocalDetectorRationalInterval:
    """PRIVATE: Validate one exact local-detector interval carrier.

    Parameters
    ----------
    interval : object
        Required canonical input.

    Returns
    -------
    validated : GalerkinLocalDetectorRationalInterval
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    _validate_raw_local_detector_interval_storage(interval)
    validated = cast(GalerkinLocalDetectorRationalInterval, interval)
    return validated  # noqa: RET504


def _validate_raw_local_detector_interval_storage(  # noqa: PLR0912
    interval: object,
) -> int:
    """PRIVATE: Validate canonical raw interval storage and return peak bits.

    Parameters
    ----------
    interval : object
        Required canonical input.

    Returns
    -------
    peak : int
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(interval, GalerkinLocalDetectorRationalInterval):
        raise TypeError("local detector interval has the wrong carrier type")
    stored = (
        interval.lower_numerator,
        interval.lower_denominator,
        interval.upper_numerator,
        interval.upper_denominator,
    )
    if any(type(value) is not int for value in stored):
        raise TypeError("local detector interval storage must use Python ints")
    if interval.lower_denominator <= 0 or interval.upper_denominator <= 0:
        raise ValueError(
            "local detector interval denominators must be positive"
        )
    peak = max(abs(value).bit_length() for value in stored)
    if peak > _HARD_MAXIMUM_RATIONAL_BITS:
        raise ValueError("local detector interval exceeds the hard bit cap")
    endpoint_pairs = (
        (interval.lower_numerator, interval.lower_denominator),
        (interval.upper_numerator, interval.upper_denominator),
    )
    if any(
        gcd(abs(numerator), denominator) != 1
        for numerator, denominator in endpoint_pairs
    ):
        raise ValueError("local detector interval endpoints must be reduced")
    if (
        interval.lower_numerator * interval.upper_denominator
        > interval.upper_numerator * interval.lower_denominator
    ):
        raise ValueError("local detector interval endpoints must be ordered")
    return peak


def _make_local_detector_rational_interval(
    lower: Fraction,
    upper: Fraction,
) -> GalerkinLocalDetectorRationalInterval:
    """PRIVATE: Construct one validated exact local-detector interval.

    Parameters
    ----------
    lower : Fraction
        Exact lower endpoint.
    upper : Fraction
        Exact upper endpoint.

    Returns
    -------
    validated : GalerkinLocalDetectorRationalInterval
        Validated ordered exact interval carrier.

    Raises
    ------
    TypeError
        If either endpoint is not a Fraction.
    ValueError
        If ordering or the hard rational-size invariant fails.
    """
    if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
        raise TypeError("local detector interval endpoints must be Fractions")
    interval = GalerkinLocalDetectorRationalInterval(
        lower_numerator=lower.numerator,
        lower_denominator=lower.denominator,
        upper_numerator=upper.numerator,
        upper_denominator=upper.denominator,
    )
    validated: GalerkinLocalDetectorRationalInterval = (
        _validate_local_detector_rational_interval(interval)
    )
    return validated


def _normal_or_zero(values: np.ndarray) -> bool:
    """PRIVATE: Check that binary64 components are finite normal-or-zero.

    Parameters
    ----------
    values : np.ndarray
        Required canonical input.

    Returns
    -------
    valid : bool
        Canonical derived result.
    """
    finite = np.isfinite(values)
    magnitude = np.abs(values)
    valid: bool = bool(
        np.all(
            finite
            & ((magnitude == 0.0) | (magnitude >= np.finfo(np.float64).tiny))
        )
    )
    return valid


def _normal_zero_or_infinity(values: np.ndarray) -> bool:
    """PRIVATE: Check for normal, zero, or infinite components.

    Parameters
    ----------
    values : np.ndarray
        Required canonical input.

    Returns
    -------
    valid : bool
        Canonical derived result.
    """
    magnitude = np.abs(values)
    valid: bool = bool(
        np.all(
            ~np.isnan(values)
            & (
                (magnitude == 0.0)
                | np.isinf(magnitude)
                | (magnitude >= np.finfo(np.float64).tiny)
            )
        )
    )
    return valid


def _valid_digest(value: object) -> bool:
    """PRIVATE: Check for a lowercase SHA-256 hexadecimal digest.

    Parameters
    ----------
    value : object
        Required canonical input.

    Returns
    -------
    result : bool
        Canonical derived result.
    """
    if type(value) is not str or len(value) != _SHA256_HEX_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _checked_logical_shape(
    logical_shape: object, point_size: int
) -> tuple[int, ...]:
    """PRIVATE: Validate a logical shape against a flat point size.

    Parameters
    ----------
    logical_shape : object
        Required canonical input.
    point_size : int
        Required canonical input.

    Returns
    -------
    checked : tuple[int, ...]
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(logical_shape, tuple) or any(
        type(value) is not int or value < 0 for value in logical_shape
    ):
        raise TypeError(
            "local detector production logical_shape must be nonnegative ints"
        )
    checked = cast(tuple[int, ...], logical_shape)
    size = 1
    for value in checked:
        size *= value
    if size != point_size:
        raise ValueError(
            "local detector production logical_shape disagrees with point"
        )
    return checked


def _production_trace_expected(  # noqa: PLR0913
    raw_intervals: _DetectorIntervals,
    point: np.ndarray,
    stage: GalerkinLocalDetectorProductionStage,
    quantity: str,
    logical_shape: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    _DetectorIntervals,
    str,
    str,
    str,
]:
    """PRIVATE: Return canonical point errors, union hulls, and trace digests.

    Parameters
    ----------
    raw_intervals : _DetectorIntervals
        Required canonical input.
    point : np.ndarray
        Required canonical input.
    stage : GalerkinLocalDetectorProductionStage
        Required canonical input.
    quantity : str
        Required canonical input.
    logical_shape : tuple[int, ...]
        Required canonical input.

    Returns
    -------
    error_array : np.ndarray
        Canonical derived result.
    lower_array : np.ndarray
        Canonical derived result.
    upper_array : np.ndarray
        Canonical derived result.
    result_3 : _DetectorIntervals
        Canonical derived result.
    point_digest : str
        Canonical derived result.
    raw_digest : str
        Canonical derived result.
    trace_digest : str
        Canonical derived result.
    """
    point_values = np.asarray(point, dtype=np.float64).reshape(-1)
    errors: list[float] = []
    hull_lowers: list[float] = []
    hull_uppers: list[float] = []
    exact_points: list[GalerkinLocalDetectorRationalInterval] = []
    raw_payload: list[tuple[int, int, int, int]] = []
    for value, raw in zip(point_values, raw_intervals, strict=True):
        interval = _validate_local_detector_rational_interval(raw)
        point_fraction = Fraction.from_float(float(value))
        distance = max(
            abs(point_fraction - interval.lower),
            abs(point_fraction - interval.upper),
        )
        errors.append(fraction_upper_float(distance))
        hull_lowers.append(
            fraction_lower_float(min(interval.lower, point_fraction))
        )
        hull_uppers.append(
            fraction_upper_float(max(interval.upper, point_fraction))
        )
        exact_points.append(
            _make_local_detector_rational_interval(
                point_fraction, point_fraction
            )
        )
        raw_payload.append(
            (
                interval.lower_numerator,
                interval.lower_denominator,
                interval.upper_numerator,
                interval.upper_denominator,
            )
        )
    error_array = np.asarray(errors, dtype=np.float64)
    lower_array = np.asarray(hull_lowers, dtype=np.float64)
    upper_array = np.asarray(hull_uppers, dtype=np.float64)
    raw_digest = sha256(
        {
            "domain": "ptyrodactyl.local_detector.production_raw.v1",
            "stage": stage.value,
            "quantity": quantity,
            "logical_shape": logical_shape,
            "raw_intervals": tuple(raw_payload),
        }
    )
    point_digest = bytes_sha256(
        np.ascontiguousarray(point_values).tobytes(order="C")
    ).hexdigest()
    trace_digest = sha256(
        {
            "domain": "ptyrodactyl.local_detector.production_trace.v1",
            "stage": stage.value,
            "quantity": quantity,
            "logical_shape": logical_shape,
            "point_dtype": _TRACE_DTYPE,
            "point_bytes_digest": point_digest,
            "point": stored_value_payload(point_values),
            "exact_point_intervals": stored_value_payload(tuple(exact_points)),
            "point_to_raw_error": stored_value_payload(error_array),
            "certified_hull_lower": stored_value_payload(lower_array),
            "certified_hull_upper": stored_value_payload(upper_array),
            "raw_interval_digest": raw_digest,
        }
    )
    return (
        error_array,
        lower_array,
        upper_array,
        tuple(exact_points),
        point_digest,
        raw_digest,
        trace_digest,
    )


def _validate_local_detector_real_production_trace(  # noqa: PLR0912
    trace: GalerkinLocalDetectorRealProductionTrace,
) -> GalerkinLocalDetectorRealProductionTrace:
    """PRIVATE: Authenticate one exact-raw/rounded-point production trace.

    Parameters
    ----------
    trace : GalerkinLocalDetectorRealProductionTrace
        Required canonical input.

    Returns
    -------
    trace : GalerkinLocalDetectorRealProductionTrace
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(trace, GalerkinLocalDetectorRealProductionTrace):
        raise TypeError("local detector production trace has the wrong type")
    if not isinstance(trace.stage, GalerkinLocalDetectorProductionStage):
        raise TypeError("local detector production stage has the wrong type")
    if type(trace.quantity) is not str or not trace.quantity.strip():
        raise ValueError("local detector production quantity must be nonempty")
    point = np.asarray(trace.point)
    error = np.asarray(trace.point_to_raw_absolute_error_upper_bounds)
    lower = np.asarray(trace.certified_hull_lower_bounds)
    upper = np.asarray(trace.certified_hull_upper_bounds)
    if point.dtype != np.dtype(np.float64) or point.ndim != 1:
        raise ValueError(
            "local detector production point must be flat float64"
        )
    size = point.shape[0]
    if (
        error.dtype != np.dtype(np.float64)
        or lower.dtype != np.dtype(np.float64)
        or upper.dtype != np.dtype(np.float64)
        or error.shape != (size,)
        or lower.shape != (size,)
        or upper.shape != (size,)
    ):
        raise ValueError("local detector production audit arrays must match")
    if (
        not isinstance(trace.raw_intervals, tuple)
        or len(trace.raw_intervals) != size
    ):
        raise ValueError("local detector production raw intervals must match")
    if (
        not isinstance(trace.exact_point_intervals, tuple)
        or len(trace.exact_point_intervals) != size
    ):
        raise ValueError("local detector exact point intervals must match")
    logical_shape = _checked_logical_shape(trace.logical_shape, size)
    if trace.point_dtype != _TRACE_DTYPE:
        raise ValueError("local detector production dtype is not canonical")
    if not all(
        _valid_digest(value)
        for value in (
            trace.point_bytes_digest,
            trace.raw_interval_digest,
            trace.trace_digest,
        )
    ):
        raise ValueError("local detector production digests must be SHA-256")
    if not all(
        _normal_or_zero(values) for values in (point, error, lower, upper)
    ):
        raise ValueError(
            "local detector production values must be finite normal-or-zero"
        )
    if bool(np.any(error < 0.0)) or bool(np.any(lower > upper)):
        raise ValueError("local detector production audit bounds are invalid")
    expected = _production_trace_expected(
        trace.raw_intervals,
        point,
        trace.stage,
        trace.quantity,
        logical_shape,
    )
    expected_arrays = expected[:3]
    if any(
        not np.array_equal(stored, canonical)
        for stored, canonical in zip(
            (error, lower, upper), expected_arrays, strict=True
        )
    ):
        raise ValueError(
            "local detector point/raw error or certified hull is noncanonical"
        )
    if any(
        stored.lower != canonical.lower or stored.upper != canonical.upper
        for stored, canonical in zip(
            trace.exact_point_intervals, expected[3], strict=True
        )
    ):
        raise ValueError("local detector exact rounded points disagree")
    if (
        trace.point_bytes_digest != expected[4]
        or trace.raw_interval_digest != expected[5]
        or trace.trace_digest != expected[6]
    ):
        raise ValueError("local detector production trace digest disagrees")
    point_fractions = tuple(
        Fraction.from_float(float(value)) for value in point
    )
    for index, point_fraction in enumerate(point_fractions):
        stored_lower = Fraction.from_float(float(lower[index]))
        stored_upper = Fraction.from_float(float(upper[index]))
        if not (stored_lower <= point_fraction <= stored_upper):
            raise ValueError(
                "local detector production point lies outside certified hull"
            )
    return trace


def _make_local_detector_real_production_trace(
    raw_intervals: _DetectorIntervals,
    point: object,
    *,
    stage: GalerkinLocalDetectorProductionStage,
    quantity: str,
    logical_shape: tuple[int, ...],
) -> GalerkinLocalDetectorRealProductionTrace:
    """PRIVATE: Build one canonical exact-raw/rounded-point trace.

    Parameters
    ----------
    raw_intervals : _DetectorIntervals
        Required canonical input.
    point : object
        Required canonical input.
    stage : GalerkinLocalDetectorProductionStage
        Required canonical input.
    quantity : str
        Required canonical input.
    logical_shape : tuple[int, ...]
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalDetectorRealProductionTrace
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(stage, GalerkinLocalDetectorProductionStage):
        raise TypeError("local detector production stage has the wrong type")
    if type(quantity) is not str or not quantity.strip():
        raise ValueError("local detector production quantity must be nonempty")
    point_array = np.asarray(point)
    if point_array.dtype != np.dtype(np.float64):
        raise TypeError("local detector production point must be float64")
    flat = np.ascontiguousarray(point_array.reshape(-1))
    if not _normal_or_zero(flat):
        raise ValueError(
            "local detector production point must be finite normal-or-zero"
        )
    if not isinstance(raw_intervals, tuple) or len(raw_intervals) != flat.size:
        raise ValueError("local detector production raw intervals must match")
    checked_shape = _checked_logical_shape(logical_shape, flat.size)
    expected = _production_trace_expected(
        raw_intervals, flat, stage, quantity.strip(), checked_shape
    )
    trace = GalerkinLocalDetectorRealProductionTrace(
        point=jnp.asarray(flat, dtype=jnp.float64),
        point_to_raw_absolute_error_upper_bounds=jnp.asarray(
            expected[0], dtype=jnp.float64
        ),
        certified_hull_lower_bounds=jnp.asarray(
            expected[1], dtype=jnp.float64
        ),
        certified_hull_upper_bounds=jnp.asarray(
            expected[2], dtype=jnp.float64
        ),
        raw_intervals=raw_intervals,
        exact_point_intervals=expected[3],
        stage=stage,
        quantity=quantity.strip(),
        logical_shape=checked_shape,
        point_dtype=_TRACE_DTYPE,
        point_bytes_digest=expected[4],
        raw_interval_digest=expected[5],
        trace_digest=expected[6],
    )
    return _validate_local_detector_real_production_trace(trace)


def _validate_local_detector_work_transcript(  # noqa: PLR0912
    transcript: GalerkinLocalDetectorWorkTranscript,
) -> GalerkinLocalDetectorWorkTranscript:
    """PRIVATE: Validate one exact local-detector work transcript.

    Parameters
    ----------
    transcript : GalerkinLocalDetectorWorkTranscript
        Candidate bounded-work transcript.

    Returns
    -------
    validated : GalerkinLocalDetectorWorkTranscript
        Validated deterministic work evidence.

    Raises
    ------
    TypeError
        If the transcript or a policy/counter has the wrong exact type.
    ValueError
        If its algorithm, policies, counters, or retained peak disagree.
    """
    if not isinstance(transcript, GalerkinLocalDetectorWorkTranscript):
        raise TypeError("local detector work has the wrong carrier type")
    counters = (
        transcript.maximum_work,
        transcript.maximum_rational_bits,
        transcript.coordinate_factor_count,
        transcript.pixel_product_count,
        transcript.mode_quadratic_count,
        transcript.ensemble_product_count,
        transcript.response_product_count,
        transcript.production_trace_count,
        transcript.hull_endpoint_count,
        transcript.nested_production_trace_count,
        transcript.nested_hull_endpoint_count,
        transcript.exact_work_count,
        transcript.rational_peak_bits,
    )
    if type(transcript.algorithm) is not str or any(
        type(value) is not int for value in counters
    ):
        raise TypeError("local detector work fields have invalid exact types")
    if (
        type(transcript.completed_successfully) is not bool
        or type(transcript.preflight_failed) is not bool
        or type(transcript.count_overflow) is not bool
    ):
        raise TypeError("local detector work status fields must be bools")
    if not isinstance(
        transcript.arithmetic_failure, GalerkinLocalDetectorFailure
    ) or not isinstance(
        transcript.scientific_failure, GalerkinLocalDetectorFailure
    ):
        raise TypeError("local detector work failure has the wrong enum type")
    exact_counts = (
        transcript.nested_parent_work_count_exact,
        transcript.nested_helper_work_count_exact,
        transcript.planned_exact_work_count_exact,
        transcript.attempted_exact_work_count_exact,
    )
    if any(
        type(value) is not str
        or not value
        or (len(value) > 1 and value.startswith("0"))
        or not value.isdecimal()
        for value in exact_counts
    ):
        raise ValueError("local detector exact work counts are noncanonical")
    nested_parent, nested_helper, planned, attempted = (
        int(value) for value in exact_counts
    )
    if transcript.algorithm != _DETECTOR_WORK_ALGORITHM:
        raise ValueError("local detector work algorithm is not canonical")
    if (
        transcript.nested_parent_work_scope
        != _DETECTOR_NESTED_PARENT_WORK_SCOPE
    ):
        raise ValueError(
            "local detector nested-parent work scope is not canonical"
        )
    if (
        transcript.maximum_work <= 0
        or transcript.maximum_work > _MAXIMUM_SIGNED_INT64
        or transcript.maximum_rational_bits <= 1
        or transcript.maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
        or any(value < 0 for value in counters[2:])
        or transcript.exact_work_count > transcript.maximum_work
    ):
        raise ValueError(
            "local detector work policies or counters are invalid"
        )
    expected_overflow = planned > _MAXIMUM_SIGNED_INT64
    if transcript.count_overflow != expected_overflow:
        raise ValueError("local detector work overflow flag disagrees")
    if transcript.preflight_failed:
        expected_preflight_failure = (
            GalerkinLocalDetectorFailure.EXACT_WORK_COUNT_OVERFLOW
            if expected_overflow
            else GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED
        )
        if (
            transcript.completed_successfully
            or transcript.exact_work_count != 0
            or attempted != planned
            or (not expected_overflow and planned <= transcript.maximum_work)
            or any(value != 0 for value in counters[2:9])
            or transcript.rational_peak_bits != 0
            or transcript.arithmetic_failure is not expected_preflight_failure
            or transcript.scientific_failure
            is not GalerkinLocalDetectorFailure.NONE
        ):
            raise ValueError("local detector failed preflight is inconsistent")
    elif transcript.completed_successfully:
        if (
            transcript.arithmetic_failure
            is not GalerkinLocalDetectorFailure.NONE
            or transcript.scientific_failure
            is not GalerkinLocalDetectorFailure.NONE
            or expected_overflow
            or planned != transcript.exact_work_count
            or attempted != transcript.exact_work_count
            or planned > transcript.maximum_work
            or transcript.rational_peak_bits > transcript.maximum_rational_bits
        ):
            raise ValueError(
                "local detector completed work evidence is inconsistent"
            )
    else:
        partial_failures = (
            GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT,
            GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE,
            GalerkinLocalDetectorFailure.PRODUCTION_POINT_HULL_FAILURE,
            GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE,
            GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE,
        )
        scientific_failures = (
            GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE,
            GalerkinLocalDetectorFailure.PIXEL_FORM_NONPASSIVE,
            GalerkinLocalDetectorFailure.ENSEMBLE_WEIGHT_INVALID,
            GalerkinLocalDetectorFailure.DOSE_INVALID,
            GalerkinLocalDetectorFailure.RESPONSE_NOT_SUBSTOCHASTIC,
            GalerkinLocalDetectorFailure.NLL_UNAVAILABLE,
        )
        arithmetic_stop = transcript.arithmetic_failure in partial_failures
        scientific_stop = transcript.scientific_failure in scientific_failures
        rational_failed = transcript.arithmetic_failure is (
            GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT
        )
        if (
            arithmetic_stop == scientific_stop
            or expected_overflow
            or planned > transcript.maximum_work
            or attempted < transcript.exact_work_count
            or attempted > planned
            or transcript.exact_work_count > transcript.maximum_work
            or (
                rational_failed
                and transcript.rational_peak_bits
                <= transcript.maximum_rational_bits
            )
            or (
                not rational_failed
                and transcript.rational_peak_bits
                > transcript.maximum_rational_bits
            )
        ):
            raise ValueError(
                "local detector failed work evidence is inconsistent"
            )
    validated: GalerkinLocalDetectorWorkTranscript = transcript
    return validated


def _canonical_nonnegative_decimal(value: object, name: str) -> str:
    """PRIVATE: Validate one canonical nonnegative decimal integer string.

    Parameters
    ----------
    value : object
        Required canonical input.
    name : str
        Required canonical input.

    Returns
    -------
    value : str
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    if (
        type(value) is not str
        or not value
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError(f"{name} must be a canonical nonnegative decimal")
    return value


def _validate_prior_entire_transcripts(
    prior_exp_transcripts: object,
    prior_log_transcripts: object,
) -> None:
    """PRIVATE: Validate completed nested entire-helper prefix transcripts.

    Parameters
    ----------
    prior_exp_transcripts : object
        Required canonical input.
    prior_log_transcripts : object
        Required canonical input.

    Raises
    ------
    TypeError
        If a lane or any resource field has the wrong exact type.
    ValueError
        If a transcript violates its resource or algorithm lane contract.
    """
    if (
        type(prior_exp_transcripts) is not tuple
        or type(prior_log_transcripts) is not tuple
    ):
        raise TypeError(
            "local detector prior nested transcripts must be tuples"
        )
    for transcripts, algorithm, lane in (
        (prior_exp_transcripts, _EXP_ENTIRE_ALGORITHM, "exp"),
        (prior_log_transcripts, _LOG_ENTIRE_ALGORITHM, "log"),
    ):
        for transcript in transcripts:
            if type(transcript) is not EntireWorkTranscript:
                raise TypeError(f"prior {lane} transcript has the wrong type")
            integer_fields = (
                transcript.precision_bits,
                transcript.maximum_terms,
                transcript.maximum_work,
                transcript.maximum_range_reductions,
                transcript.maximum_rational_bits,
                transcript.series_terms,
                transcript.range_reductions,
                transcript.root_enclosures,
                transcript.rectangle_products,
                transcript.reciprocal_steps,
                transcript.exact_work_count,
            )
            if any(type(value) is not int for value in integer_fields):
                raise TypeError(
                    "prior entire transcript resources must use Python ints"
                )
            _validate_entire_transcript(transcript)
            if transcript.algorithm != algorithm:
                raise ValueError(
                    f"prior {lane} transcript algorithm disagrees"
                )
            if transcript.maximum_rational_bits <= 1:
                raise ValueError(
                    "prior entire transcript rational policy must exceed one"
                )
            if (
                transcript.precision_bits + 1
                > transcript.maximum_rational_bits
            ):
                raise ValueError(
                    "local detector prior entire transcript precision exceeds "
                    "its rational policy"
                )


def _helper_failure_digest(
    evidence: GalerkinLocalDetectorHelperFailureEvidence,
) -> str:
    """PRIVATE: Return the canonical digest for one helper-failure record.

    Parameters
    ----------
    evidence : GalerkinLocalDetectorHelperFailureEvidence
        Required canonical input.

    Returns
    -------
    result : str
        Canonical derived result.
    """
    return sha256(
        {
            "domain": "ptyrodactyl.local_detector.helper_failure.v1",
            "call": evidence.call.value,
            "channel_index": evidence.channel_index,
            "failure": evidence.failure.value,
            "nested_kernel": evidence.nested_kernel,
            "nested_failure": (
                None
                if evidence.nested_failure is None
                else evidence.nested_failure.value
            ),
            "prior_exp_transcripts": stored_value_payload(
                evidence.prior_exp_transcripts
            ),
            "prior_log_transcripts": stored_value_payload(
                evidence.prior_log_transcripts
            ),
            "local_exact_work_count_exact": (
                evidence.local_exact_work_count_exact
            ),
            "nested_exact_work_count_exact": (
                evidence.nested_exact_work_count_exact
            ),
            "nested_attempted_exact_work_count_exact": (
                evidence.nested_attempted_exact_work_count_exact
            ),
            "planned_exact_work_count_exact": (
                evidence.planned_exact_work_count_exact
            ),
            "attempted_exact_work_count_exact": (
                evidence.attempted_exact_work_count_exact
            ),
        }
    )


def _validate_local_detector_helper_failure_evidence(  # noqa: PLR0912
    evidence: GalerkinLocalDetectorHelperFailureEvidence,
) -> GalerkinLocalDetectorHelperFailureEvidence:
    """PRIVATE: Validate one channel- and helper-bound failure record.

    Parameters
    ----------
    evidence : GalerkinLocalDetectorHelperFailureEvidence
        Required canonical input.

    Returns
    -------
    evidence : GalerkinLocalDetectorHelperFailureEvidence
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(evidence, GalerkinLocalDetectorHelperFailureEvidence):
        raise TypeError("local detector helper failure has the wrong type")
    if not isinstance(evidence.call, GalerkinLocalDetectorHelperCall):
        raise TypeError("local detector helper call has the wrong enum type")
    if type(evidence.channel_index) is not int or evidence.channel_index < 0:
        raise ValueError("local detector helper channel must be nonnegative")
    if not isinstance(evidence.failure, CensoredPoissonEnclosureFailure):
        raise TypeError(
            "local detector helper failure has the wrong enum type"
        )
    _validate_prior_entire_transcripts(
        evidence.prior_exp_transcripts,
        evidence.prior_log_transcripts,
    )
    local = int(
        _canonical_nonnegative_decimal(
            evidence.local_exact_work_count_exact,
            "local_exact_work_count_exact",
        )
    )
    planned = int(
        _canonical_nonnegative_decimal(
            evidence.planned_exact_work_count_exact,
            "planned_exact_work_count_exact",
        )
    )
    attempted = int(
        _canonical_nonnegative_decimal(
            evidence.attempted_exact_work_count_exact,
            "attempted_exact_work_count_exact",
        )
    )
    if planned < attempted or attempted < local:
        raise ValueError(
            "local detector helper failure counts are inconsistent"
        )
    if (
        evidence.failure
        is CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED
    ):
        if planned != attempted or attempted <= local:
            raise ValueError("helper work-budget failure counts disagree")
    elif planned != attempted or attempted != local:
        raise ValueError("nonbudget helper failure counts must be completed")
    nested_present = evidence.nested_kernel is not None
    if nested_present:
        if evidence.nested_kernel not in ("exp", "log"):
            raise ValueError("local detector nested helper kernel is invalid")
        if (
            not isinstance(evidence.nested_failure, EntireEnclosureFailure)
            or evidence.nested_exact_work_count_exact is None
            or evidence.nested_attempted_exact_work_count_exact is None
        ):
            raise ValueError(
                "local detector nested failure evidence is incomplete"
            )
        _canonical_nonnegative_decimal(
            evidence.nested_exact_work_count_exact,
            "nested_exact_work_count_exact",
        )
        nested_completed = int(evidence.nested_exact_work_count_exact)
        nested_attempted = int(
            _canonical_nonnegative_decimal(
                evidence.nested_attempted_exact_work_count_exact,
                "nested_attempted_exact_work_count_exact",
            )
        )
        if (
            evidence.nested_failure
            is EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
        ):
            if nested_attempted <= nested_completed:
                raise ValueError("nested work-budget counts disagree")
        elif nested_attempted != nested_completed:
            raise ValueError("nonbudget nested counts must be completed")
    elif (
        evidence.nested_failure is not None
        or evidence.nested_exact_work_count_exact is not None
        or evidence.nested_attempted_exact_work_count_exact is not None
    ):
        raise ValueError(
            "local detector nested failure sentinel is inconsistent"
        )
    nested_outer = {
        CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE: "exp",
        CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE: "log",
    }
    required_kernel = nested_outer.get(evidence.failure)
    if required_kernel is None and nested_present:
        raise ValueError(
            "nonnested helper failure cannot carry nested evidence"
        )
    if (
        required_kernel is not None
        and evidence.nested_kernel != required_kernel
    ):
        raise ValueError(
            f"{required_kernel} failure requires matching nested evidence"
        )
    if not _valid_digest(evidence.failure_digest):
        raise ValueError("local detector helper failure digest is malformed")
    if evidence.failure_digest != _helper_failure_digest(evidence):
        raise ValueError("local detector helper failure digest disagrees")
    return evidence


def _make_local_detector_helper_failure_evidence(  # noqa: PLR0913
    *,
    call: GalerkinLocalDetectorHelperCall,
    channel_index: int,
    failure: CensoredPoissonEnclosureFailure,
    local_exact_work_count: int,
    nested_kernel: str | None,
    nested_failure: EntireEnclosureFailure | None,
    nested_exact_work_count: int | None,
    nested_attempted_exact_work_count: int | None = None,
    prior_exp_transcripts: object = (),
    prior_log_transcripts: object = (),
    planned_exact_work_count: int | None = None,
    attempted_exact_work_count: int | None = None,
) -> GalerkinLocalDetectorHelperFailureEvidence:
    """PRIVATE: Construct one canonical replayable helper failure record.

    Parameters
    ----------
    call : GalerkinLocalDetectorHelperCall
        Required canonical input.
    channel_index : int
        Required canonical input.
    failure : CensoredPoissonEnclosureFailure
        Required canonical input.
    local_exact_work_count : int
        Required canonical input.
    nested_kernel : str | None
        Required canonical input.
    nested_failure : EntireEnclosureFailure | None
        Required canonical input.
    nested_exact_work_count : int | None
        Required canonical input.
    nested_attempted_exact_work_count : int | None
        Optional input; the signature supplies its default.
    prior_exp_transcripts : object
        Optional input; the signature supplies its default.
    prior_log_transcripts : object
        Optional input; the signature supplies its default.
    planned_exact_work_count : int | None
        Optional input; the signature supplies its default.
    attempted_exact_work_count : int | None
        Optional input; the signature supplies its default.

    Returns
    -------
    result : GalerkinLocalDetectorHelperFailureEvidence
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    counts = (
        local_exact_work_count,
        nested_exact_work_count,
        nested_attempted_exact_work_count,
        planned_exact_work_count,
        attempted_exact_work_count,
    )
    if any(value is not None and type(value) is not int for value in counts):
        raise TypeError(
            "local detector helper work counts must be Python ints"
        )
    if local_exact_work_count < 0 or any(
        value is not None and value < 0
        for value in (
            nested_exact_work_count,
            nested_attempted_exact_work_count,
        )
    ):
        raise ValueError(
            "local detector helper work counts must be nonnegative"
        )
    _validate_prior_entire_transcripts(
        prior_exp_transcripts,
        prior_log_transcripts,
    )
    canonical_exp_transcripts = cast(
        tuple[EntireWorkTranscript, ...], prior_exp_transcripts
    )
    canonical_log_transcripts = cast(
        tuple[EntireWorkTranscript, ...], prior_log_transcripts
    )
    attempted = (
        local_exact_work_count
        if attempted_exact_work_count is None
        else attempted_exact_work_count
    )
    planned = (
        attempted
        if planned_exact_work_count is None
        else planned_exact_work_count
    )
    evidence = GalerkinLocalDetectorHelperFailureEvidence(
        call=call,
        channel_index=channel_index,
        failure=failure,
        nested_kernel=nested_kernel,
        nested_failure=nested_failure,
        prior_exp_transcripts=canonical_exp_transcripts,
        prior_log_transcripts=canonical_log_transcripts,
        local_exact_work_count_exact=str(local_exact_work_count),
        nested_exact_work_count_exact=(
            None
            if nested_exact_work_count is None
            else str(nested_exact_work_count)
        ),
        nested_attempted_exact_work_count_exact=(
            None
            if nested_attempted_exact_work_count is None
            else str(nested_attempted_exact_work_count)
        ),
        planned_exact_work_count_exact=str(planned),
        attempted_exact_work_count_exact=str(attempted),
        failure_digest="0" * 64,
    )
    sealed = replace(evidence, failure_digest=_helper_failure_digest(evidence))
    return _validate_local_detector_helper_failure_evidence(sealed)


def _make_local_detector_work_transcript(  # noqa: PLR0913
    *,
    algorithm: str,
    maximum_work: int,
    maximum_rational_bits: int,
    coordinate_factor_count: int,
    pixel_product_count: int,
    mode_quadratic_count: int,
    ensemble_product_count: int,
    response_product_count: int,
    exact_work_count: int,
    rational_peak_bits: int,
    production_trace_count: int = 0,
    hull_endpoint_count: int = 0,
    nested_production_trace_count: int = 0,
    nested_hull_endpoint_count: int = 0,
    nested_helper_work_count_exact: str = "0",
    nested_parent_work_count_exact: str = "0",
    planned_exact_work_count_exact: str | None = None,
    attempted_exact_work_count_exact: str | None = None,
    completed_successfully: bool = True,
    arithmetic_failure: GalerkinLocalDetectorFailure = (
        GalerkinLocalDetectorFailure.NONE
    ),
    scientific_failure: GalerkinLocalDetectorFailure = (
        GalerkinLocalDetectorFailure.NONE
    ),
    preflight_failed: bool = False,
    count_overflow: bool | None = None,
) -> GalerkinLocalDetectorWorkTranscript:
    """PRIVATE: Construct one validated local-detector work transcript.

    Parameters
    ----------
    algorithm : str
        Required canonical input.
    maximum_work : int
        Required canonical input.
    maximum_rational_bits : int
        Required canonical input.
    coordinate_factor_count : int
        Required canonical input.
    pixel_product_count : int
        Required canonical input.
    mode_quadratic_count : int
        Required canonical input.
    ensemble_product_count : int
        Required canonical input.
    response_product_count : int
        Required canonical input.
    exact_work_count : int
        Required canonical input.
    rational_peak_bits : int
        Required canonical input.
    production_trace_count : int
        Optional input; the signature supplies its default.
    hull_endpoint_count : int
        Optional input; the signature supplies its default.
    nested_production_trace_count : int
        Optional input; the signature supplies its default.
    nested_hull_endpoint_count : int
        Optional input; the signature supplies its default.
    nested_helper_work_count_exact : str
        Optional input; the signature supplies its default.
    nested_parent_work_count_exact : str
        Optional input; the signature supplies its default.
    planned_exact_work_count_exact : str | None
        Optional input; the signature supplies its default.
    attempted_exact_work_count_exact : str | None
        Optional input; the signature supplies its default.
    completed_successfully : bool
        Optional input; the signature supplies its default.
    arithmetic_failure : GalerkinLocalDetectorFailure
        Optional input; the signature supplies its default.
    scientific_failure : GalerkinLocalDetectorFailure
        Optional input; the signature supplies its default.
    preflight_failed : bool
        Optional input; the signature supplies its default.
    count_overflow : bool | None
        Optional input; the signature supplies its default.

    Returns
    -------
    validated : GalerkinLocalDetectorWorkTranscript
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    planned = (
        str(exact_work_count)
        if planned_exact_work_count_exact is None
        else planned_exact_work_count_exact
    )
    attempted = (
        str(exact_work_count)
        if attempted_exact_work_count_exact is None
        else attempted_exact_work_count_exact
    )
    overflow = (
        int(planned) > _MAXIMUM_SIGNED_INT64
        if count_overflow is None
        and type(planned) is str
        and planned.isdecimal()
        else count_overflow
    )
    if type(overflow) is not bool:
        raise TypeError("count_overflow must be a Python bool")
    transcript = GalerkinLocalDetectorWorkTranscript(
        algorithm=algorithm,
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        coordinate_factor_count=coordinate_factor_count,
        pixel_product_count=pixel_product_count,
        mode_quadratic_count=mode_quadratic_count,
        ensemble_product_count=ensemble_product_count,
        response_product_count=response_product_count,
        production_trace_count=production_trace_count,
        hull_endpoint_count=hull_endpoint_count,
        nested_production_trace_count=nested_production_trace_count,
        nested_hull_endpoint_count=nested_hull_endpoint_count,
        exact_work_count=exact_work_count,
        rational_peak_bits=rational_peak_bits,
        nested_parent_work_count_exact=nested_parent_work_count_exact,
        nested_helper_work_count_exact=nested_helper_work_count_exact,
        nested_parent_work_scope=_DETECTOR_NESTED_PARENT_WORK_SCOPE,
        planned_exact_work_count_exact=planned,
        attempted_exact_work_count_exact=attempted,
        completed_successfully=completed_successfully,
        arithmetic_failure=arithmetic_failure,
        scientific_failure=scientific_failure,
        preflight_failed=preflight_failed,
        count_overflow=overflow,
    )
    validated: GalerkinLocalDetectorWorkTranscript = (
        _validate_local_detector_work_transcript(transcript)
    )
    return validated


def _checked_scalar(
    value: object, dtype: np.dtype[np.generic], name: str
) -> np.ndarray:
    """PRIVATE: Validate one concrete scalar array with an exact dtype.

    Parameters
    ----------
    value : object
        Required canonical input.
    dtype : np.dtype[np.generic]
        Required canonical input.
    name : str
        Required canonical input.

    Returns
    -------
    array : np.ndarray
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    array = np.asarray(value)
    if array.shape != () or array.dtype != dtype:
        raise ValueError(f"{name} must be one scalar {dtype.name} array")
    return array


def _checked_vector(
    value: object,
    dtype: np.dtype[np.generic],
    size: int,
    name: str,
) -> np.ndarray:
    """PRIVATE: Validate one concrete vector with an exact dtype and size.

    Parameters
    ----------
    value : object
        Required canonical input.
    dtype : np.dtype[np.generic]
        Required canonical input.
    size : int
        Required canonical input.
    name : str
        Required canonical input.

    Returns
    -------
    array : np.ndarray
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype != dtype:
        raise ValueError(
            f"{name} must have shape {(size,)} and dtype {dtype.name}"
        )
    return array


def _same_array_bytes(left: object, right: object) -> bool:
    """PRIVATE: Return exact dtype, shape, and C-order byte equality.

    Parameters
    ----------
    left : object
        Required canonical input.
    right : object
        Required canonical input.

    Returns
    -------
    same : bool
        Canonical derived result.
    """
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    same: bool = (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
    )
    return same


def _validate_failure_scalar(value: object) -> GalerkinLocalDetectorFailure:
    """PRIVATE: Validate one scalar int64 local-detector failure mask.

    Parameters
    ----------
    value : object
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalDetectorFailure
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    array = _checked_scalar(value, np.dtype(np.int64), "failure_mask")
    integer = int(array)
    known = 0
    for failure in GalerkinLocalDetectorFailure:
        known |= int(failure)
    if integer < 0 or integer & ~known:
        raise ValueError("local detector failure mask has unknown bits")
    return GalerkinLocalDetectorFailure(integer)


def _validate_production_traces(
    traces: _DetectorProductionTraces,
) -> _DetectorProductionTraces:
    """PRIVATE: Validate ordered, uniquely named production traces.

    Parameters
    ----------
    traces : _DetectorProductionTraces
        Required canonical input.

    Returns
    -------
    checked : _DetectorProductionTraces
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(traces, tuple):
        raise TypeError("local detector production traces must be a tuple")
    checked = tuple(
        _validate_local_detector_real_production_trace(trace)
        for trace in traces
    )
    keys = tuple((trace.stage, trace.quantity) for trace in checked)
    if len(set(keys)) != len(keys):
        raise ValueError("local detector production trace keys must be unique")
    return checked


def _validate_helper_outcomes(
    transcripts: object,
    failures: object,
    size: int,
    allowed_calls: tuple[GalerkinLocalDetectorHelperCall, ...],
    name: str,
) -> None:
    """PRIVATE: Validate aligned per-channel helper outcomes.

    Parameters
    ----------
    transcripts : object
        Required canonical input.
    failures : object
        Required canonical input.
    size : int
        Required canonical input.
    allowed_calls : tuple[GalerkinLocalDetectorHelperCall, ...]
        Required canonical input.
    name : str
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    TypeError
        If the canonical contract is violated.
    """
    if (
        not isinstance(transcripts, tuple)
        or not isinstance(failures, tuple)
        or len(transcripts) != size
        or len(failures) != size
    ):
        raise ValueError(f"{name} helper outcome counts disagree")
    for channel, (transcript, failure) in enumerate(
        zip(transcripts, failures, strict=True)
    ):
        if (transcript is None) == (failure is None):
            raise ValueError(
                f"{name} requires exactly one transcript or failure"
            )
        if transcript is not None and not isinstance(
            transcript, CensoredPoissonWorkTranscript
        ):
            raise TypeError(f"{name} transcript has the wrong type")
        if failure is not None:
            checked = _validate_local_detector_helper_failure_evidence(failure)
            if (
                checked.channel_index != channel
                or checked.call not in allowed_calls
            ):
                raise ValueError(f"{name} helper failure binding disagrees")


def _validate_intervals(
    values: object,
    size: int,
    name: str,
    *,
    nonnegative: bool,
    positive: bool = False,
) -> _DetectorIntervals:
    """PRIVATE: Validate one fixed-size exact interval tuple and sign contract.

    Parameters
    ----------
    values : object
        Required canonical input.
    size : int
        Required canonical input.
    name : str
        Required canonical input.
    nonnegative : bool
        Required canonical input.
    positive : bool
        Optional input; the signature supplies its default.

    Returns
    -------
    checked : _DetectorIntervals
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(values, tuple) or len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} intervals")
    checked = tuple(
        _validate_local_detector_rational_interval(value) for value in values
    )
    if nonnegative and any(value.lower < 0 for value in checked):
        raise ValueError(f"{name} must be nonnegative")
    if positive and any(value.lower <= 0 for value in checked):
        raise ValueError(f"{name} must be strictly positive")
    return checked


def _expected_carrier_digests(
    value: _DetectorDigestCarrier,
    *,
    domain: str,
    digest_names: tuple[str, str, str],
    identity_names: tuple[str, ...],
) -> tuple[str, str, str]:
    """PRIVATE: Derive canonical identity, evidence, and certificate digests.

    Parameters
    ----------
    value : _DetectorDigestCarrier
        Required canonical input.
    domain : str
        Required canonical input.
    digest_names : tuple[str, str, str]
        Required canonical input.
    identity_names : tuple[str, ...]
        Required canonical input.

    Returns
    -------
    identity : str
        Canonical derived result.
    evidence : str
        Canonical derived result.
    certificate : str
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    declared = {field.name for field in fields(value)}
    if any(name not in declared for name in (*digest_names, *identity_names)):
        raise ValueError("local detector digest schema is incomplete")
    identity = sha256(
        {
            "domain": f"{domain}.identity.v1",
            "fields": {
                name: stored_value_payload(getattr(value, name))
                for name in identity_names
            },
        }
    )
    evidence = sha256(
        {
            "domain": f"{domain}.evidence.v1",
            "identity": identity,
            "fields": {
                field.name: stored_value_payload(getattr(value, field.name))
                for field in fields(value)
                if field.name not in digest_names
            },
        }
    )
    certificate = sha256(
        {
            "domain": f"{domain}.certificate.v1",
            "identity": identity,
            "evidence": evidence,
        }
    )
    return identity, evidence, certificate


def _validate_digest_triplet(
    stored: tuple[str, str, str], expected: tuple[str, str, str]
) -> None:
    """PRIVATE: Reject malformed or noncanonical carrier digest triplets.

    Parameters
    ----------
    stored : tuple[str, str, str]
        Required canonical input.
    expected : tuple[str, str, str]
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    if not all(_valid_digest(value) for value in stored):
        raise ValueError("local detector carrier digests must be SHA-256")
    if stored != expected:
        raise ValueError("local detector carrier digest triplet disagrees")


def _local_detector_input_manifest_digest(
    value: _DetectorInputManifest, domain: str
) -> str:
    """PRIVATE: Return one canonical primitive-input manifest digest.

    Parameters
    ----------
    value : _DetectorInputManifest
        Required canonical input.
    domain : str
        Required canonical input.

    Returns
    -------
    result : str
        Canonical derived result.
    """
    return sha256(
        {
            "domain": domain,
            "fields": {
                field.name: stored_value_payload(getattr(value, field.name))
                for field in fields(value)
                if field.name != "manifest_digest"
            },
        }
    )


def _validate_local_passive_pixel_input_manifest(
    manifest: GalerkinLocalPassivePixelInputManifest,
) -> GalerkinLocalPassivePixelInputManifest:
    """PRIVATE: Validate one independent primitive passive-pixel manifest.

    Parameters
    ----------
    manifest : GalerkinLocalPassivePixelInputManifest
        Required canonical input.

    Returns
    -------
    manifest : GalerkinLocalPassivePixelInputManifest
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(manifest, GalerkinLocalPassivePixelInputManifest):
        raise TypeError(
            "local passive-pixel input manifest has the wrong type"
        )
    if not isinstance(
        manifest.route, GalerkinLocalPositivePortRoute
    ) or not isinstance(
        manifest.terminal_disposition,
        GalerkinLocalVacuumTerminalDisposition,
    ):
        raise TypeError("local passive-pixel upstream route is invalid")
    state_error = _checked_scalar(
        manifest.maximum_state_error,
        np.dtype(np.float64),
        "maximum_state_error",
    )
    if not _normal_or_zero(state_error) or float(state_error) <= 0.0:
        raise ValueError(
            "maximum_state_error must be finite positive binary64"
        )
    mapping = np.asarray(manifest.node_to_pixel)
    quadrature = np.asarray(manifest.quadrature_weight_points)
    aperture = np.asarray(manifest.aperture_efficiency_points)
    if (
        mapping.ndim != 1
        or mapping.dtype != np.dtype(np.int64)
        or quadrature.shape != mapping.shape
        or quadrature.dtype != np.dtype(np.float64)
        or aperture.shape != mapping.shape
        or aperture.dtype != np.dtype(np.float64)
        or mapping.size == 0
    ):
        raise ValueError("local passive-pixel primitive vectors disagree")
    if (
        type(manifest.pixel_count) is not int
        or manifest.pixel_count <= 0
        or bool(np.any(mapping < -1))
        or bool(np.any(mapping >= manifest.pixel_count))
        or not _normal_or_zero(quadrature)
        or not _normal_or_zero(aperture)
    ):
        raise ValueError("local passive-pixel primitive values are invalid")
    positive_policies = (
        manifest.maximum_stability_direct_pairs,
        manifest.maximum_gram_pairs,
        manifest.maximum_terminal_direct_pairs,
        manifest.maximum_branch_direct_terms,
        manifest.maximum_cut_direct_pairs,
        manifest.maximum_root_work,
        manifest.precision_bits,
        manifest.maximum_terms,
        manifest.maximum_entire_work,
        manifest.maximum_interval_work,
        manifest.maximum_l8_rational_bits,
        manifest.maximum_detector_work,
        manifest.maximum_detector_rational_bits,
    )
    if (
        any(type(value) is not int for value in positive_policies)
        or any(
            value <= 0 or value > _MAXIMUM_SIGNED_INT64
            for value in positive_policies
        )
        or type(manifest.maximum_range_reductions) is not int
        or manifest.maximum_range_reductions < 0
        or manifest.maximum_range_reductions > _MAXIMUM_SIGNED_INT64
        or manifest.maximum_l8_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
        or manifest.maximum_detector_rational_bits
        > _HARD_MAXIMUM_RATIONAL_BITS
        or manifest.maximum_l8_rational_bits <= 1
        or manifest.maximum_detector_rational_bits <= 1
    ):
        raise ValueError("local passive-pixel resource policy is invalid")
    interval_groups = (
        manifest.quadrature_weight_intervals,
        manifest.aperture_efficiency_intervals,
    )
    if any(
        not isinstance(values, tuple) or len(values) != mapping.size
        for values in interval_groups
    ):
        raise ValueError("local passive-pixel interval counts disagree")
    interval_peak = max(
        (
            _validate_raw_local_detector_interval_storage(value)
            for values in interval_groups
            for value in values
        ),
        default=0,
    )
    if interval_peak <= manifest.maximum_detector_rational_bits:
        for values, name in (
            (
                manifest.quadrature_weight_intervals,
                "quadrature_weight_intervals",
            ),
            (
                manifest.aperture_efficiency_intervals,
                "aperture_efficiency_intervals",
            ),
        ):
            _validate_intervals(values, mapping.size, name, nonnegative=False)
    if not isinstance(
        manifest.coordinate_convention,
        GalerkinLocalDetectorCoordinateConvention,
    ):
        raise TypeError("local passive-pixel coordinate convention is invalid")
    expected = _local_detector_input_manifest_digest(
        manifest, "ptyrodactyl.local_detector.pixel_input.v1"
    )
    if (
        not _valid_digest(manifest.manifest_digest)
        or manifest.manifest_digest != expected
    ):
        raise ValueError("local passive-pixel input digest disagrees")
    return manifest


def _make_local_passive_pixel_input_manifest(
    manifest: GalerkinLocalPassivePixelInputManifest,
) -> GalerkinLocalPassivePixelInputManifest:
    """PRIVATE: Seal one primitive passive-pixel input manifest.

    Parameters
    ----------
    manifest : GalerkinLocalPassivePixelInputManifest
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPassivePixelInputManifest
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    if not isinstance(manifest, GalerkinLocalPassivePixelInputManifest):
        raise TypeError(
            "local passive-pixel input manifest has the wrong type"
        )
    for values in (
        manifest.quadrature_weight_intervals,
        manifest.aperture_efficiency_intervals,
    ):
        if isinstance(values, tuple):
            for value in values:
                _validate_raw_local_detector_interval_storage(value)
    sealed = replace(
        manifest,
        manifest_digest=_local_detector_input_manifest_digest(
            manifest, "ptyrodactyl.local_detector.pixel_input.v1"
        ),
    )
    return _validate_local_passive_pixel_input_manifest(sealed)


def _validate_local_censored_poisson_detector_input_manifest(  # noqa: PLR0912
    manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
) -> GalerkinLocalCensoredPoissonDetectorInputManifest:
    """PRIVATE: Validate one independent primitive detector input manifest.

    Parameters
    ----------
    manifest : GalerkinLocalCensoredPoissonDetectorInputManifest
        Required canonical input.

    Returns
    -------
    manifest : GalerkinLocalCensoredPoissonDetectorInputManifest
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(
        manifest, GalerkinLocalCensoredPoissonDetectorInputManifest
    ):
        raise TypeError("local detector input manifest has the wrong type")
    if (
        not isinstance(manifest.pixel_inputs, tuple)
        or not manifest.pixel_inputs
    ):
        raise ValueError("local detector input requires pixel manifests")
    pixel_inputs = tuple(
        _validate_local_passive_pixel_input_manifest(value)
        for value in manifest.pixel_inputs
    )
    pixel_count = pixel_inputs[0].pixel_count
    if any(value.pixel_count != pixel_count for value in pixel_inputs):
        raise ValueError("local detector pixel manifests disagree")
    response = np.asarray(manifest.response_matrix)
    if (
        response.dtype != np.dtype(np.float64)
        or response.ndim != _MATRIX_DIMENSIONS
        or response.shape[0] == 0
        or response.shape[1] != pixel_count
    ):
        raise ValueError("local detector input response shape disagrees")
    channel_count = response.shape[0]
    calibration_arrays = [response]
    for value, dtype, name in (
        (manifest.pre_gain_background, np.dtype(np.float64), "background"),
        (manifest.deterministic_gain, np.dtype(np.float64), "gain"),
        (manifest.electronic_offset, np.dtype(np.float64), "offset"),
        (manifest.count_ceilings, np.dtype(np.int64), "count ceilings"),
        (manifest.fit_mask, np.dtype(np.bool_), "fit mask"),
    ):
        checked = _checked_vector(value, dtype, channel_count, name)
        if dtype == np.dtype(np.float64):
            calibration_arrays.append(checked)
    if any(not _normal_or_zero(value) for value in calibration_arrays):
        raise ValueError(
            "local detector calibration storage is outside binary64"
        )
    point = _checked_scalar(
        manifest.incident_electron_count_point,
        np.dtype(np.float64),
        "incident_electron_count_point",
    )
    if not _normal_or_zero(point):
        raise ValueError("incident electron count point is outside binary64")
    mode_count = len(pixel_inputs)
    if (
        len(manifest.ensemble_weight_numerators) != mode_count
        or len(manifest.ensemble_weight_denominators) != mode_count
        or any(
            type(value) is not int
            for value in (
                *manifest.ensemble_weight_numerators,
                *manifest.ensemble_weight_denominators,
            )
        )
        or any(value <= 0 for value in manifest.ensemble_weight_denominators)
    ):
        raise ValueError("local detector exact ensemble weights are invalid")
    if (
        max(
            (
                abs(value).bit_length()
                for value in (
                    *manifest.ensemble_weight_numerators,
                    *manifest.ensemble_weight_denominators,
                )
            ),
            default=0,
        )
        > _HARD_MAXIMUM_RATIONAL_BITS
    ):
        raise ValueError(
            "local detector ensemble weights exceed the hard bit cap"
        )
    positive_policies = (
        manifest.maximum_detector_work,
        manifest.maximum_detector_rational_bits,
        manifest.maximum_count_ceiling,
        manifest.maximum_poisson_work,
        manifest.maximum_poisson_rational_bits,
        manifest.exp_precision_bits,
        manifest.maximum_exp_terms,
        manifest.maximum_exp_work,
    )
    if (
        any(type(value) is not int for value in positive_policies)
        or any(
            value <= 0 or value > _MAXIMUM_SIGNED_INT64
            for value in positive_policies
        )
        or type(manifest.maximum_exp_range_reductions) is not int
        or manifest.maximum_exp_range_reductions < 0
        or manifest.maximum_exp_range_reductions > _MAXIMUM_SIGNED_INT64
        or manifest.maximum_detector_rational_bits
        > _HARD_MAXIMUM_RATIONAL_BITS
        or manifest.maximum_detector_rational_bits <= 1
        or manifest.maximum_poisson_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
        or manifest.maximum_poisson_rational_bits <= 1
        or type(manifest.calibration_provenance) is not str
        or not manifest.calibration_provenance.strip()
    ):
        raise ValueError(
            "local detector input policies/provenance are invalid"
        )
    dose_peak = _validate_raw_local_detector_interval_storage(
        manifest.incident_electron_count_interval
    )
    if dose_peak <= manifest.maximum_detector_rational_bits:
        _validate_local_detector_rational_interval(
            manifest.incident_electron_count_interval
        )
    expected = _local_detector_input_manifest_digest(
        manifest, "ptyrodactyl.local_detector.detector_input.v1"
    )
    if (
        not _valid_digest(manifest.manifest_digest)
        or manifest.manifest_digest != expected
    ):
        raise ValueError("local detector input manifest digest disagrees")
    return manifest


def _make_local_censored_poisson_detector_input_manifest(
    manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
) -> GalerkinLocalCensoredPoissonDetectorInputManifest:
    """PRIVATE: Seal one primitive detector input manifest.

    Parameters
    ----------
    manifest : GalerkinLocalCensoredPoissonDetectorInputManifest
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalCensoredPoissonDetectorInputManifest
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(
        manifest, GalerkinLocalCensoredPoissonDetectorInputManifest
    ):
        raise TypeError("local detector input manifest has the wrong type")
    if isinstance(manifest.pixel_inputs, tuple):
        for pixel in manifest.pixel_inputs:
            if isinstance(pixel, GalerkinLocalPassivePixelInputManifest):
                for values in (
                    pixel.quadrature_weight_intervals,
                    pixel.aperture_efficiency_intervals,
                ):
                    if isinstance(values, tuple):
                        for value in values:
                            _validate_raw_local_detector_interval_storage(
                                value
                            )
    _validate_raw_local_detector_interval_storage(
        manifest.incident_electron_count_interval
    )
    weights = (
        *manifest.ensemble_weight_numerators,
        *manifest.ensemble_weight_denominators,
    )
    if all(type(value) is int for value in weights) and any(
        abs(value).bit_length() > _HARD_MAXIMUM_RATIONAL_BITS
        for value in weights
    ):
        raise ValueError(
            "local detector ensemble weights exceed the hard bit cap"
        )
    sealed = replace(
        manifest,
        manifest_digest=_local_detector_input_manifest_digest(
            manifest, "ptyrodactyl.local_detector.detector_input.v1"
        ),
    )
    return _validate_local_censored_poisson_detector_input_manifest(sealed)


def _expected_port_branches(  # noqa: PLR0912, PLR0915
    terminal: GalerkinLocalVacuumTerminalCertificate,
    route: GalerkinLocalPositivePortRoute,
) -> tuple[
    tuple[GalerkinLocalPositivePortBranchDisposition, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    GalerkinLocalDetectorFailure,
    bool,
    bool,
]:
    """PRIVATE: Derive the disjoint projected-port and radiation dispositions.

    Parameters
    ----------
    terminal : GalerkinLocalVacuumTerminalCertificate
        Required canonical input.
    route : GalerkinLocalPositivePortRoute
        Required canonical input.

    Returns
    -------
    result_0 : tuple[GalerkinLocalPositivePortBranchDisposition, ...]
        Canonical derived result.
    result_1 : tuple[bool, ...]
        Canonical derived result.
    result_2 : tuple[bool, ...]
        Canonical derived result.
    failure : GalerkinLocalDetectorFailure
        Canonical derived result.
    projected : bool
        Canonical derived result.
    outgoing : bool
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    branch = terminal.branch_evidence
    roots = branch.root_certificates
    half_spaces = branch.half_space_dispositions
    dispositions: list[GalerkinLocalPositivePortBranchDisposition] = []
    retained: list[bool] = []
    zero_weight: list[bool] = []
    branch_disposition = GalerkinLocalPositivePortBranchDisposition
    detector_failure = GalerkinLocalDetectorFailure
    failure = GalerkinLocalDetectorFailure.NONE
    if not bool(np.asarray(terminal.vacuum_branch_eligible)):
        failure |= GalerkinLocalDetectorFailure.VACUUM_TERMINAL_NONCERTIFICATE
    all_propagating_inward_zero = True
    for root, half_space in zip(roots, half_spaces, strict=True):
        if (
            root is None
            or root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED
            or half_space
            is GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED
        ):
            dispositions.append(
                GalerkinLocalPositivePortBranchDisposition.ROOT_UNCLASSIFIED
            )
            retained.append(False)
            zero_weight.append(True)
            failure |= GalerkinLocalDetectorFailure.ROOT_UNCLASSIFIED
            continue
        if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
            retained.append(True)
            zero_weight.append(False)
            if half_space is (
                GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_EXACT_ZERO
            ):
                disposition = branch_disposition.PROPAGATING_OUTWARD_RETAINED_INWARD_EXACT_ZERO  # noqa: E501
            elif half_space is (
                GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_PROVABLY_NONZERO
            ):
                disposition = branch_disposition.PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_PROVABLY_NONZERO  # noqa: E501
                all_propagating_inward_zero = False
            elif half_space is (
                GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_UNRESOLVED
            ):
                disposition = branch_disposition.PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_UNRESOLVED  # noqa: E501
                all_propagating_inward_zero = False
            else:
                raise ValueError(
                    "propagating root has a nonpropagating half-space status"
                )
            dispositions.append(disposition)
            continue
        retained.append(False)
        zero_weight.append(True)
        if root.classification is GalerkinLocalVacuumRootClass.EVANESCENT:
            if half_space is (
                GalerkinLocalVacuumHalfSpaceDisposition.EVANESCENT_GROWING_EXACT_ZERO
            ):
                dispositions.append(
                    GalerkinLocalPositivePortBranchDisposition.EVANESCENT_DECAYING_ZERO_WEIGHT_GROWING_EXACT_ZERO
                )
            elif half_space in (
                GalerkinLocalVacuumHalfSpaceDisposition.EVANESCENT_GROWING_PROVABLY_NONZERO,
                GalerkinLocalVacuumHalfSpaceDisposition.EVANESCENT_GROWING_UNRESOLVED,
            ):
                dispositions.append(
                    GalerkinLocalPositivePortBranchDisposition.EVANESCENT_GROWING_REJECTED
                )
                failure |= detector_failure.EVANESCENT_GROWING_NOT_EXACT_ZERO
            else:
                raise ValueError(
                    "evanescent root has a nonevanescent half-space status"
                )
        elif root.classification is GalerkinLocalVacuumRootClass.GRAZING:
            if half_space is (
                GalerkinLocalVacuumHalfSpaceDisposition.GRAZING_DERIVATIVE_EXACT_ZERO
            ):
                dispositions.append(
                    GalerkinLocalPositivePortBranchDisposition.GRAZING_CONSTANT_ZERO_WEIGHT_DERIVATIVE_EXACT_ZERO
                )
            elif half_space in (
                GalerkinLocalVacuumHalfSpaceDisposition.GRAZING_DERIVATIVE_PROVABLY_NONZERO,
                GalerkinLocalVacuumHalfSpaceDisposition.GRAZING_DERIVATIVE_UNRESOLVED,
            ):
                dispositions.append(
                    GalerkinLocalPositivePortBranchDisposition.GRAZING_DERIVATIVE_REJECTED
                )
                failure |= detector_failure.GRAZING_DERIVATIVE_NOT_EXACT_ZERO
            else:
                raise ValueError(
                    "grazing root has a nongrazing half-space status"
                )
        else:
            raise ValueError("local detector root class is unsupported")
    fatal_projected = (
        GalerkinLocalDetectorFailure.VACUUM_TERMINAL_NONCERTIFICATE
        | GalerkinLocalDetectorFailure.ROOT_UNCLASSIFIED
        | GalerkinLocalDetectorFailure.EVANESCENT_GROWING_NOT_EXACT_ZERO
        | GalerkinLocalDetectorFailure.GRAZING_DERIVATIVE_NOT_EXACT_ZERO
    )
    projected = not bool(failure & fatal_projected)
    outgoing = (
        route is GalerkinLocalPositivePortRoute.OUTGOING_RADIATION
        and projected
        and all_propagating_inward_zero
    )
    if (
        route is GalerkinLocalPositivePortRoute.OUTGOING_RADIATION
        and not all_propagating_inward_zero
    ):
        failure |= (
            GalerkinLocalDetectorFailure.PROPAGATING_INWARD_NOT_EXACT_ZERO
        )
    return (
        tuple(dispositions),
        tuple(retained),
        tuple(zero_weight),
        failure,
        projected,
        outgoing,
    )


def _validate_local_positive_port_certificate(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalPositivePortCertificate,
) -> GalerkinLocalPositivePortCertificate:
    """PRIVATE: Authenticate one local projected positive-port carrier.

    Parameters
    ----------
    certificate : GalerkinLocalPositivePortCertificate
        Required canonical input.

    Returns
    -------
    certificate : GalerkinLocalPositivePortCertificate
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalPositivePortCertificate):
        raise TypeError("local positive port has the wrong carrier type")
    terminal = certificate.terminal_certificate
    if not isinstance(terminal, GalerkinLocalVacuumTerminalCertificate):
        raise TypeError("local positive port terminal has the wrong type")
    if not isinstance(certificate.route, GalerkinLocalPositivePortRoute):
        raise TypeError("local positive port route has the wrong enum type")
    branch = terminal.branch_evidence
    size = len(branch.root_certificates)
    amplitudes = _checked_vector(
        certificate.production_amplitudes,
        np.dtype(np.complex128),
        size,
        "production_amplitudes",
    )
    errors = _checked_vector(
        certificate.exact_state_total_amplitude_error_bounds,
        np.dtype(np.float64),
        size,
        "exact_state_total_amplitude_error_bounds",
    )
    roots = _checked_vector(
        certificate.production_root_realizations,
        np.dtype(np.float64),
        size,
        "production_root_realizations",
    )
    root_errors = _checked_vector(
        certificate.production_root_error_upper_bounds,
        np.dtype(np.float64),
        size,
        "production_root_error_upper_bounds",
    )
    retained = _checked_vector(
        certificate.retained_propagating_mask,
        np.dtype(np.bool_),
        size,
        "retained_propagating_mask",
    )
    zero_weight = _checked_vector(
        certificate.zero_weight_mask,
        np.dtype(np.bool_),
        size,
        "zero_weight_mask",
    )
    parent_eligible = bool(np.asarray(terminal.vacuum_branch_eligible))
    for value, name in (
        (
            certificate.production_prediction_l2_norm_upper_bound,
            "production_prediction_l2_norm_upper_bound",
        ),
        (
            certificate.exact_state_prediction_error_l2_upper_bound,
            "exact_state_prediction_error_l2_upper_bound",
        ),
    ):
        scalar = _checked_scalar(value, np.dtype(np.float64), name)
        if (
            not _normal_zero_or_infinity(scalar)
            or float(scalar) < 0.0
            or (parent_eligible and not np.isfinite(float(scalar)))
        ):
            raise ValueError(
                f"{name} must be a canonical nonnegative availability value"
            )
    if (
        not _normal_zero_or_infinity(amplitudes.view(np.float64))
        or not _normal_zero_or_infinity(errors)
        or not _normal_zero_or_infinity(roots)
        or not _normal_zero_or_infinity(root_errors)
        or bool(np.any(errors < 0.0))
        or bool(np.any(root_errors < 0.0))
        or (
            parent_eligible
            and not all(
                _normal_or_zero(values)
                for values in (
                    amplitudes.view(np.float64),
                    errors,
                    roots,
                    root_errors,
                )
            )
        )
    ):
        raise ValueError("local positive-port production arrays are invalid")
    if (
        not _same_array_bytes(
            amplitudes,
            np.asarray(branch.frozen_defining_branch_points)[:, 0],
        )
        or not _same_array_bytes(
            errors,
            np.asarray(branch.exact_state_total_amplitude_error_bounds)[:, 0],
        )
        or not _same_array_bytes(
            roots, np.asarray(branch.frozen_positive_root_realizations)
        )
        or not _same_array_bytes(
            root_errors, np.asarray(branch.frozen_positive_root_error_bounds)
        )
        or not _same_array_bytes(
            certificate.production_prediction_l2_norm_upper_bound,
            branch.production_prediction_l2_norm_upper_bound,
        )
        or not _same_array_bytes(
            certificate.exact_state_prediction_error_l2_upper_bound,
            branch.exact_state_prediction_error_l2_upper_bound,
        )
    ):
        raise ValueError(
            "local positive-port copied L8 production evidence disagrees"
        )
    expected = _expected_port_branches(terminal, certificate.route)
    if certificate.branch_dispositions != expected[0]:
        raise ValueError("local positive-port branch dispositions disagree")
    if not np.array_equal(
        retained, np.asarray(expected[1])
    ) or not np.array_equal(zero_weight, np.asarray(expected[2])):
        raise ValueError("local positive-port retained/zero masks disagree")
    failure = _validate_failure_scalar(certificate.failure_mask)
    if failure != expected[3]:
        raise ValueError("local positive-port failure mask disagrees")
    positive = _checked_scalar(
        certificate.positive_port_eligible,
        np.dtype(np.bool_),
        "positive_port_eligible",
    )
    outgoing = _checked_scalar(
        certificate.outgoing_radiation_eligible,
        np.dtype(np.bool_),
        "outgoing_radiation_eligible",
    )
    if bool(positive) != expected[4] or bool(outgoing) != expected[5]:
        raise ValueError("local positive-port predicates disagree")
    if (
        certificate.parent_half_space_dispositions
        != branch.half_space_dispositions
        or certificate.branch_role != branch.prediction_branch_role
        or certificate.branch_role != 0
        or certificate.branch_scope != _POSITIVE_PORT_BRANCH_SCOPE
        or certificate.exact_state_amplitude_scope
        != _POSITIVE_PORT_EXACT_STATE_SCOPE
        or certificate.root_realization_audit_scope
        != _POSITIVE_PORT_ROOT_AUDIT_SCOPE
        or certificate.completion_scope != _POSITIVE_PORT_COMPLETION_SCOPE
    ):
        raise ValueError(
            "local positive-port scope or parent status disagrees"
        )
    if len(certificate.exact_root_intervals) != size:
        raise ValueError("local positive-port root interval count disagrees")
    for stored, root in zip(
        certificate.exact_root_intervals,
        branch.root_certificates,
        strict=True,
    ):
        expected_root = None if root is None else root.root_interval
        if expected_root is None:
            if stored is not None:
                raise ValueError("local positive-port root sentinel disagrees")
        elif stored is None or not isinstance(
            stored, GalerkinLocalDetectorRationalInterval
        ):
            raise ValueError(
                "local positive-port exact root interval disagrees"
            )
        else:
            _validate_local_detector_rational_interval(stored)
            if (
                stored.lower != expected_root.lower
                or stored.upper != expected_root.upper
            ):
                raise ValueError(
                    "local positive-port exact root interval disagrees"
                )
    traces = _validate_production_traces(certificate.production_traces)
    defining = branch.defining_branch_rectangles[0]
    rectangle_columns = tuple(np.asarray(value) for value in defining)
    trace_available = all(
        _normal_or_zero(values)
        for values in (
            amplitudes.view(np.float64),
            *rectangle_columns,
        )
    )
    real_raw = (
        tuple(
            _make_local_detector_rational_interval(
                Fraction.from_float(float(rectangle_columns[0][index])),
                Fraction.from_float(float(rectangle_columns[1][index])),
            )
            for index in range(size)
        )
        if trace_available
        else ()
    )
    imag_raw = (
        tuple(
            _make_local_detector_rational_interval(
                Fraction.from_float(float(rectangle_columns[2][index])),
                Fraction.from_float(float(rectangle_columns[3][index])),
            )
            for index in range(size)
        )
        if trace_available
        else ()
    )
    expected_traces = (
        (
            _make_local_detector_real_production_trace(
                real_raw,
                np.asarray(np.real(amplitudes), dtype=np.float64),
                stage=GalerkinLocalDetectorProductionStage.L8_ROLE_ZERO_AMPLITUDE,
                quantity="l8_role_zero_amplitude.real",
                logical_shape=(size,),
            ),
            _make_local_detector_real_production_trace(
                imag_raw,
                np.asarray(np.imag(amplitudes), dtype=np.float64),
                stage=GalerkinLocalDetectorProductionStage.L8_ROLE_ZERO_AMPLITUDE,
                quantity="l8_role_zero_amplitude.imag",
                logical_shape=(size,),
            ),
            _make_local_detector_real_production_trace(
                tuple(
                    _make_local_detector_rational_interval(
                        Fraction.from_float(float(value)),
                        Fraction.from_float(float(value)),
                    )
                    for value in np.real(amplitudes)
                ),
                np.asarray(np.real(amplitudes), dtype=np.float64),
                stage=GalerkinLocalDetectorProductionStage.POSITIVE_PORT_AMPLITUDE,
                quantity="positive_port_amplitude.real",
                logical_shape=(size,),
            ),
            _make_local_detector_real_production_trace(
                tuple(
                    _make_local_detector_rational_interval(
                        Fraction.from_float(float(value)),
                        Fraction.from_float(float(value)),
                    )
                    for value in np.imag(amplitudes)
                ),
                np.asarray(np.imag(amplitudes), dtype=np.float64),
                stage=GalerkinLocalDetectorProductionStage.POSITIVE_PORT_AMPLITUDE,
                quantity="positive_port_amplitude.imag",
                logical_shape=(size,),
            ),
        )
        if trace_available
        else ()
    )
    if len(traces) != len(expected_traces) or any(
        stored.trace_digest != expected_trace.trace_digest
        for stored, expected_trace in zip(traces, expected_traces, strict=True)
    ):
        raise ValueError("local positive-port L8 production traces disagree")
    if (
        certificate.target_digest != terminal.target_digest
        or certificate.source_digest != terminal.source_digest
        or certificate.state_identity_digest != terminal.state_identity_digest
        or certificate.parent_terminal_identity_digest
        != terminal.terminal_identity_digest
        or certificate.parent_terminal_evidence_digest
        != terminal.terminal_evidence_digest
    ):
        raise ValueError(
            "local positive-port parent identity binding disagrees"
        )
    digest_names = (
        "port_identity_digest",
        "port_evidence_digest",
        "certificate_digest",
    )
    expected_digests = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.positive_port",
        digest_names=digest_names,
        identity_names=(
            "target_digest",
            "source_digest",
            "state_identity_digest",
            "parent_terminal_identity_digest",
            "route",
            "branch_role",
        ),
    )
    _validate_digest_triplet(
        tuple(getattr(certificate, name) for name in digest_names),
        expected_digests,
    )
    return certificate


def _make_local_positive_port_certificate(
    certificate: GalerkinLocalPositivePortCertificate,
) -> GalerkinLocalPositivePortCertificate:
    """PRIVATE: Seal and validate one owner-constructed positive port.

    Parameters
    ----------
    certificate : GalerkinLocalPositivePortCertificate
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPositivePortCertificate
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalPositivePortCertificate):
        raise TypeError("local positive port has the wrong carrier type")
    digest_names = (
        "port_identity_digest",
        "port_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.positive_port",
        digest_names=digest_names,
        identity_names=(
            "target_digest",
            "source_digest",
            "state_identity_digest",
            "parent_terminal_identity_digest",
            "route",
            "branch_role",
        ),
    )
    sealed = replace(
        certificate,
        port_identity_digest=expected[0],
        port_evidence_digest=expected[1],
        certificate_digest=expected[2],
    )
    return _validate_local_positive_port_certificate(sealed)


def _validate_available_pixel_intervals(
    certificate: GalerkinLocalPassivePixelForms,
    fiber_size: int,
    pixel_count: int,
) -> None:
    """PRIVATE: Validate the shaped interval DAG of an available pixel result.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.
    fiber_size : int
        Required canonical input.
    pixel_count : int
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    for values, name in (
        (certificate.current_weight_intervals, "current_weight_intervals"),
        (
            certificate.coordinate_jacobian_intervals,
            "coordinate_jacobian_intervals",
        ),
        (
            certificate.quadrature_weight_intervals,
            "quadrature_weight_intervals",
        ),
        (
            certificate.aperture_efficiency_intervals,
            "aperture_efficiency_intervals",
        ),
        (
            certificate.outward_form_diagonal_intervals,
            "outward_form_diagonal_intervals",
        ),
        (
            certificate.outward_minus_pixel_form_diagonal_intervals,
            "outward_minus_pixel_form_diagonal_intervals",
        ),
    ):
        _validate_intervals(values, fiber_size, name, nonnegative=True)
    _validate_intervals(
        (certificate.amplitude_scale_interval,),
        1,
        "amplitude_scale_interval",
        nonnegative=True,
        positive=True,
    )
    if (
        not isinstance(certificate.pixel_form_diagonal_intervals, tuple)
        or len(certificate.pixel_form_diagonal_intervals) != pixel_count
    ):
        raise ValueError("local detector pixel form row count disagrees")
    for row in certificate.pixel_form_diagonal_intervals:
        _validate_intervals(
            row, fiber_size, "pixel_form_diagonal_intervals", nonnegative=True
        )
    for values, name in (
        (
            certificate.production_quadratic_intervals,
            "production_quadratic_intervals",
        ),
        (
            certificate.pixel_form_norm_upper_intervals,
            "pixel_form_norm_upper_intervals",
        ),
        (
            certificate.production_realization_error_upper_intervals,
            "production_realization_error_upper_intervals",
        ),
        (
            certificate.state_radius_incremental_error_upper_intervals,
            "state_radius_incremental_error_upper_intervals",
        ),
        (
            certificate.combined_exact_state_error_upper_intervals,
            "combined_exact_state_error_upper_intervals",
        ),
        (
            certificate.exact_state_pixel_flux_intervals,
            "exact_state_pixel_flux_intervals",
        ),
    ):
        _validate_intervals(values, pixel_count, name, nonnegative=True)


def _validate_local_passive_pixel_forms(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalPassivePixelForms,
) -> GalerkinLocalPassivePixelForms:
    """PRIVATE: Authenticate one positive passive ideal-pixel form carrier.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.

    Returns
    -------
    certificate : GalerkinLocalPassivePixelForms
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalPassivePixelForms):
        raise TypeError("local passive pixel forms have the wrong type")
    port = _validate_local_positive_port_certificate(certificate.positive_port)
    if (
        type(certificate.pixel_count) is not int
        or certificate.pixel_count <= 0
    ):
        raise ValueError("local detector pixel_count must be positive")
    fiber_size = np.asarray(port.production_amplitudes).shape[0]
    pixel_count = certificate.pixel_count
    mapping = _checked_vector(
        certificate.node_to_pixel,
        np.dtype(np.int64),
        fiber_size,
        "node_to_pixel",
    )
    if bool(np.any(mapping < -1)) or bool(np.any(mapping >= pixel_count)):
        raise ValueError("local detector node_to_pixel is outside its domain")
    quadrature = _checked_vector(
        certificate.quadrature_weights,
        np.dtype(np.float64),
        fiber_size,
        "quadrature_weights",
    )
    aperture = _checked_vector(
        certificate.aperture_efficiencies,
        np.dtype(np.float64),
        fiber_size,
        "aperture_efficiencies",
    )
    if not _normal_or_zero(quadrature) or not _normal_or_zero(aperture):
        raise ValueError(
            "local detector quadrature/aperture storage is outside binary64"
        )
    available = bool(
        _checked_scalar(
            certificate.production_evidence_available,
            np.dtype(np.bool_),
            "production_evidence_available",
        )
    )
    scalar_intervals = (
        certificate.production_outward_quadratic_interval,
        certificate.outward_form_norm_upper_interval,
        certificate.outward_production_realization_error_upper_interval,
        certificate.outward_state_radius_incremental_error_upper_interval,
        certificate.outward_combined_exact_state_error_upper_interval,
        certificate.exact_state_outward_flux_interval,
        certificate.production_to_exact_x_amplitude_error_interval,
        certificate.state_radius_amplitude_error_interval,
        certificate.exact_state_amplitude_error_interval,
        certificate.production_amplitude_norm_interval,
    )
    _validate_intervals(
        scalar_intervals,
        len(scalar_intervals),
        "local detector scalar pixel evidence",
        nonnegative=True,
    )
    _validate_local_detector_work_transcript(certificate.work_transcript)
    if available:
        _validate_available_pixel_intervals(
            certificate, fiber_size, pixel_count
        )
    else:
        primitive_interval_groups = (
            (
                certificate.quadrature_weight_intervals,
                "quadrature_weight_intervals",
            ),
            (
                certificate.aperture_efficiency_intervals,
                "aperture_efficiency_intervals",
            ),
        )
        if any(
            not isinstance(values, tuple) or len(values) != fiber_size
            for values, _name in primitive_interval_groups
        ):
            raise ValueError(
                "unavailable pixel primitive interval counts disagree"
            )
        primitive_peak = max(
            (
                _validate_raw_local_detector_interval_storage(value)
                for values, _name in primitive_interval_groups
                for value in values
            ),
            default=0,
        )
        if primitive_peak <= certificate.work_transcript.maximum_rational_bits:
            for values, name in primitive_interval_groups:
                _validate_intervals(
                    values, fiber_size, name, nonnegative=False
                )
        sentinel_tuples = (
            certificate.current_weight_intervals,
            certificate.coordinate_jacobian_intervals,
            certificate.outward_form_diagonal_intervals,
            certificate.pixel_form_diagonal_intervals,
            certificate.outward_minus_pixel_form_diagonal_intervals,
            certificate.production_quadratic_intervals,
            certificate.pixel_form_norm_upper_intervals,
            certificate.production_realization_error_upper_intervals,
            certificate.state_radius_incremental_error_upper_intervals,
            certificate.combined_exact_state_error_upper_intervals,
            certificate.exact_state_pixel_flux_intervals,
        )
        if any(value != () for value in sentinel_tuples):
            raise ValueError(
                "unavailable pixel evidence must use empty tuples"
            )
        _validate_intervals(
            (certificate.amplitude_scale_interval,),
            1,
            "unavailable amplitude scale sentinel",
            nonnegative=True,
        )
    _validate_production_traces(certificate.production_traces)
    for value, name in (
        (
            certificate.production_evidence_available,
            "production_evidence_available",
        ),
        (certificate.positive_forms_eligible, "positive_forms_eligible"),
        (certificate.passive_forms_eligible, "passive_forms_eligible"),
    ):
        _checked_scalar(value, np.dtype(np.bool_), name)
    _validate_failure_scalar(certificate.failure_mask)
    if not isinstance(
        certificate.coordinate_convention,
        GalerkinLocalDetectorCoordinateConvention,
    ):
        raise TypeError("local detector coordinate convention is invalid")
    if any(
        type(value) is not str or not value.strip()
        for value in (
            certificate.coordinate_factor_scope,
            certificate.pixel_form_scope,
            certificate.lvt56_error_scope,
            certificate.passivity_margin_scope,
            certificate.no_experimental_validity_scope,
        )
    ):
        raise ValueError("local detector pixel scopes must be nonempty")
    if not _valid_digest(certificate.input_manifest_digest):
        raise ValueError(
            "local detector pixel input-manifest digest is invalid"
        )
    if certificate.parent_port_certificate_digest != port.certificate_digest:
        raise ValueError("local detector pixel parent digest disagrees")
    # Import locally to keep module initialization acyclic while making the
    # type owner replay the sole pure detector arithmetic implementation.
    from ptyrodactyl.galerkin.detector import (  # noqa: PLC0415
        _expected_local_passive_pixel_evidence,
    )

    canonical = _expected_local_passive_pixel_evidence(certificate)
    interval_fields = (
        "current_weight_intervals",
        "amplitude_scale_interval",
        "coordinate_jacobian_intervals",
        "quadrature_weight_intervals",
        "aperture_efficiency_intervals",
        "outward_form_diagonal_intervals",
        "pixel_form_diagonal_intervals",
        "outward_minus_pixel_form_diagonal_intervals",
        "production_outward_quadratic_interval",
        "outward_form_norm_upper_interval",
        "outward_production_realization_error_upper_interval",
        "outward_state_radius_incremental_error_upper_interval",
        "outward_combined_exact_state_error_upper_interval",
        "exact_state_outward_flux_interval",
        "production_quadratic_intervals",
        "pixel_form_norm_upper_intervals",
        "production_to_exact_x_amplitude_error_interval",
        "state_radius_amplitude_error_interval",
        "exact_state_amplitude_error_interval",
        "production_amplitude_norm_interval",
        "production_realization_error_upper_intervals",
        "state_radius_incremental_error_upper_intervals",
        "combined_exact_state_error_upper_intervals",
        "exact_state_pixel_flux_intervals",
    )
    if any(
        stored_value_payload(getattr(certificate, name))
        != stored_value_payload(canonical[name])
        for name in interval_fields
    ):
        raise ValueError("local detector pixel arithmetic evidence disagrees")
    canonical_traces = cast(
        _DetectorProductionTraces, canonical["production_traces"]
    )
    if tuple(
        trace.trace_digest for trace in certificate.production_traces
    ) != tuple(trace.trace_digest for trace in canonical_traces):
        raise ValueError(
            "local detector pixel production trace chain disagrees"
        )
    if stored_value_payload(
        certificate.work_transcript
    ) != stored_value_payload(canonical["work_transcript"]):
        raise ValueError("local detector pixel exact-work evidence disagrees")
    if available != canonical["production_evidence_available"]:
        raise ValueError(
            "local detector pixel availability predicate disagrees"
        )
    for name in ("positive_forms_eligible", "passive_forms_eligible"):
        if bool(np.asarray(getattr(certificate, name))) != canonical[name]:
            raise ValueError(f"local detector {name} predicate disagrees")
    if int(np.asarray(certificate.failure_mask)) != canonical["failure_mask"]:
        raise ValueError("local detector pixel failure mask disagrees")
    for name in (
        "coordinate_factor_scope",
        "pixel_form_scope",
        "lvt56_error_scope",
        "passivity_margin_scope",
        "no_experimental_validity_scope",
    ):
        if getattr(certificate, name) != canonical[name]:
            raise ValueError(f"local detector canonical {name} disagrees")
    digest_names = (
        "pixel_model_identity_digest",
        "pixel_model_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.passive_pixel_forms",
        digest_names=digest_names,
        identity_names=(
            "parent_port_certificate_digest",
            "input_manifest_digest",
            "coordinate_convention",
            "node_to_pixel",
            "quadrature_weight_intervals",
            "quadrature_weights",
            "aperture_efficiency_intervals",
            "aperture_efficiencies",
            "pixel_count",
        ),
    )
    _validate_digest_triplet(
        tuple(getattr(certificate, name) for name in digest_names), expected
    )
    return certificate


def _make_local_passive_pixel_forms(
    certificate: GalerkinLocalPassivePixelForms,
) -> GalerkinLocalPassivePixelForms:
    """PRIVATE: Seal and validate owner-constructed passive pixel forms.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPassivePixelForms
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If primitive interval storage is not a canonical tuple.
    """
    if not isinstance(certificate, GalerkinLocalPassivePixelForms):
        raise TypeError("local passive pixel forms have the wrong type")
    for values in (
        certificate.quadrature_weight_intervals,
        certificate.aperture_efficiency_intervals,
    ):
        if not isinstance(values, tuple):
            raise ValueError(
                "local detector primitive pixel intervals must be tuples"
            )
        for value in values:
            _validate_raw_local_detector_interval_storage(value)
    digest_names = (
        "pixel_model_identity_digest",
        "pixel_model_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.passive_pixel_forms",
        digest_names=digest_names,
        identity_names=(
            "parent_port_certificate_digest",
            "input_manifest_digest",
            "coordinate_convention",
            "node_to_pixel",
            "quadrature_weight_intervals",
            "quadrature_weights",
            "aperture_efficiency_intervals",
            "aperture_efficiencies",
            "pixel_count",
        ),
    )
    sealed = replace(
        certificate,
        pixel_model_identity_digest=expected[0],
        pixel_model_evidence_digest=expected[1],
        certificate_digest=expected[2],
    )
    return _validate_local_passive_pixel_forms(sealed)


def _validate_mode_pixel_intervals(
    values: object,
    mode_count: int,
    pixel_count: int,
    name: str,
) -> _ModePixelIntervals:
    """PRIVATE: Validate one ordered mode-by-pixel nonnegative interval table.

    Parameters
    ----------
    values : object
        Required canonical input.
    mode_count : int
        Required canonical input.
    pixel_count : int
        Required canonical input.
    name : str
        Required canonical input.

    Returns
    -------
    result : _ModePixelIntervals
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(values, tuple) or len(values) != mode_count:
        raise ValueError(f"{name} mode count disagrees")
    return tuple(
        _validate_intervals(row, pixel_count, name, nonnegative=True)
        for row in values
    )


def _expected_mode_state_binding(
    pixel_forms: tuple[GalerkinLocalPassivePixelForms, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    _OptionalDetectorIntervals,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
]:
    """PRIVATE: Derive ordered state/source authority through nested L8.

    Parameters
    ----------
    pixel_forms : tuple[GalerkinLocalPassivePixelForms, ...]
        Required canonical input.

    Returns
    -------
    result_0 : tuple[str, ...]
        Canonical derived result.
    result_1 : tuple[str, ...]
        Canonical derived result.
    result_2 : tuple[str, ...]
        Canonical derived result.
    result_3 : _OptionalDetectorIntervals
        Canonical derived result.
    result_4 : tuple[str, ...]
        Canonical derived result.
    result_5 : tuple[str, ...]
        Canonical derived result.
    result_6 : tuple[str, ...]
        Canonical derived result.
    binding : str
        Canonical derived result.
    """
    targets: list[str] = []
    sources: list[str] = []
    states: list[str] = []
    radii: list[GalerkinLocalDetectorRationalInterval | None] = []
    radius_provenance: list[str] = []
    ports: list[str] = []
    pixels: list[str] = []
    for pixel in pixel_forms:
        port = pixel.positive_port
        terminal = port.terminal_certificate
        projection = terminal.projection_certificate
        radius_value = float(np.asarray(projection.state_radius_upper_bound))
        targets.append(terminal.target_digest)
        sources.append(terminal.source_digest)
        states.append(terminal.state_identity_digest)
        radii.append(
            None
            if not np.isfinite(radius_value) or radius_value < 0.0
            else _make_local_detector_rational_interval(
                Fraction.from_float(radius_value),
                Fraction.from_float(radius_value),
            )
        )
        radius_provenance.append(terminal.parent_projection_certificate_digest)
        ports.append(port.certificate_digest)
        pixels.append(pixel.pixel_model_evidence_digest)
    binding = sha256(
        {
            "domain": "ptyrodactyl.local_detector.mode_state_binding.v1",
            "target_digests": tuple(targets),
            "source_digests": tuple(sources),
            "state_identity_digests": tuple(states),
            "state_radius_intervals": stored_value_payload(tuple(radii)),
            "state_radius_provenance_digests": tuple(radius_provenance),
            "port_certificate_digests": tuple(ports),
            "pixel_evidence_digests": tuple(pixels),
        }
    )
    return (
        tuple(targets),
        tuple(sources),
        tuple(states),
        tuple(radii),
        tuple(radius_provenance),
        tuple(ports),
        tuple(pixels),
        binding,
    )


def _validate_available_detector_intervals(
    certificate: GalerkinLocalCensoredPoissonDetector,
    mode_count: int,
    pixel_count: int,
    channel_count: int,
) -> None:
    """PRIVATE: Validate an available detector interval DAG.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.
    mode_count : int
        Required canonical input.
    pixel_count : int
        Required canonical input.
    channel_count : int
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    _validate_intervals(
        certificate.incident_reduced_flux_intervals,
        mode_count,
        "incident_reduced_flux_intervals",
        nonnegative=True,
        positive=True,
    )
    for values, name in (
        (
            certificate.mode_production_to_exact_x_amplitude_error_intervals,
            "mode_production_to_exact_x_amplitude_error_intervals",
        ),
        (
            certificate.mode_state_radius_amplitude_error_intervals,
            "mode_state_radius_amplitude_error_intervals",
        ),
        (
            certificate.mode_exact_state_amplitude_error_intervals,
            "mode_exact_state_amplitude_error_intervals",
        ),
        (
            certificate.mode_production_amplitude_norm_intervals,
            "mode_production_amplitude_norm_intervals",
        ),
        (
            certificate.mode_outward_passivity_margin_intervals,
            "mode_outward_passivity_margin_intervals",
        ),
    ):
        _validate_intervals(values, mode_count, name, nonnegative=True)
    for values, name in (
        (
            certificate.mode_exact_state_pixel_flux_intervals,
            "mode_exact_state_pixel_flux_intervals",
        ),
        (
            certificate.mode_production_quadratic_intervals,
            "mode_production_quadratic_intervals",
        ),
        (
            certificate.mode_pixel_form_norm_upper_intervals,
            "mode_pixel_form_norm_upper_intervals",
        ),
        (
            certificate.mode_production_realization_error_upper_intervals,
            "mode_production_realization_error_upper_intervals",
        ),
        (
            certificate.mode_state_radius_incremental_error_upper_intervals,
            "mode_state_radius_incremental_error_upper_intervals",
        ),
        (
            certificate.mode_combined_exact_state_error_upper_intervals,
            "mode_combined_exact_state_error_upper_intervals",
        ),
        (
            certificate.mode_pixel_fraction_intervals,
            "mode_pixel_fraction_intervals",
        ),
    ):
        _validate_mode_pixel_intervals(values, mode_count, pixel_count, name)
    for values, name in (
        (
            certificate.ideal_arrival_mean_intervals,
            "ideal_arrival_mean_intervals",
        ),
        (
            certificate.production_pre_gain_mean_point_intervals,
            "production_pre_gain_mean_point_intervals",
        ),
        (
            certificate.exact_state_pre_gain_mean_intervals,
            "exact_state_pre_gain_mean_intervals",
        ),
        (certificate.censored_mean_intervals, "censored_mean_intervals"),
        (
            certificate.expected_digitized_mean_intervals,
            "expected_digitized_mean_intervals",
        ),
    ):
        checked = _validate_intervals(
            values, channel_count, name, nonnegative=False
        )
        if name != "expected_digitized_mean_intervals" and any(
            value.lower < 0 for value in checked
        ):
            raise ValueError(f"{name} must be nonnegative")
    if any(
        value.lower != value.upper
        for value in certificate.production_pre_gain_mean_point_intervals
    ):
        raise ValueError(
            "production pre-gain point intervals must be singletons"
        )


def _validate_local_censored_poisson_detector(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalCensoredPoissonDetector,
) -> GalerkinLocalCensoredPoissonDetector:
    """PRIVATE: Authenticate one fixed censored-Poisson detector carrier.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.

    Returns
    -------
    certificate : GalerkinLocalCensoredPoissonDetector
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalCensoredPoissonDetector):
        raise TypeError("local censored-Poisson detector has the wrong type")
    if (
        not isinstance(certificate.pixel_forms, tuple)
        or not certificate.pixel_forms
    ):
        raise ValueError("local detector requires at least one mode")
    pixels = tuple(
        _validate_local_passive_pixel_forms(value)
        for value in certificate.pixel_forms
    )
    mode_count = len(pixels)
    pixel_count = pixels[0].pixel_count
    if any(value.pixel_count != pixel_count for value in pixels):
        raise ValueError("local detector modes must share one pixel count")
    expected_binding = _expected_mode_state_binding(pixels)
    stored_binding: tuple[object, ...] = (
        certificate.mode_target_digests,
        certificate.mode_source_digests,
        certificate.mode_state_identity_digests,
        certificate.mode_state_radius_intervals,
        certificate.mode_state_radius_provenance_digests,
        certificate.mode_port_certificate_digests,
        certificate.mode_pixel_evidence_digests,
        certificate.mode_state_binding_digest,
    )
    if stored_binding != expected_binding:
        raise ValueError("local detector ordered mode-state binding disagrees")
    if len(set(certificate.mode_target_digests)) != 1:
        raise ValueError("local detector modes must share one target identity")
    if certificate.target_digest != certificate.mode_target_digests[0]:
        raise ValueError("local detector target digest disagrees with modes")
    response = np.asarray(certificate.response_matrix)
    if (
        response.dtype != np.dtype(np.float64)
        or response.ndim != _MATRIX_DIMENSIONS
    ):
        raise ValueError("local detector response must be one float64 matrix")
    channel_count, response_pixels = response.shape
    if channel_count == 0 or response_pixels != pixel_count:
        raise ValueError("local detector response dimensions disagree")
    background = _checked_vector(
        certificate.pre_gain_background,
        np.dtype(np.float64),
        channel_count,
        "pre_gain_background",
    )
    gain = _checked_vector(
        certificate.deterministic_gain,
        np.dtype(np.float64),
        channel_count,
        "deterministic_gain",
    )
    offset = _checked_vector(
        certificate.electronic_offset,
        np.dtype(np.float64),
        channel_count,
        "electronic_offset",
    )
    _checked_vector(
        certificate.count_ceilings,
        np.dtype(np.int64),
        channel_count,
        "count_ceilings",
    )
    _checked_vector(
        certificate.fit_mask,
        np.dtype(np.bool_),
        channel_count,
        "fit_mask",
    )
    dose_point = _checked_scalar(
        certificate.incident_electron_count_point,
        np.dtype(np.float64),
        "incident_electron_count_point",
    )
    if not _normal_or_zero(dose_point):
        raise ValueError("incident_electron_count_point is outside binary64")
    available = bool(
        _checked_scalar(
            certificate.production_evidence_available,
            np.dtype(np.bool_),
            "production_evidence_available",
        )
    )
    if (
        not _normal_or_zero(response)
        or not _normal_or_zero(background)
        or not _normal_or_zero(gain)
        or not _normal_or_zero(offset)
    ):
        raise ValueError(
            "local detector calibration storage is outside binary64"
        )
    if (
        not isinstance(certificate.ensemble_weight_numerators, tuple)
        or not isinstance(certificate.ensemble_weight_denominators, tuple)
        or len(certificate.ensemble_weight_numerators) != mode_count
        or len(certificate.ensemble_weight_denominators) != mode_count
        or any(
            type(value) is not int
            for value in (
                *certificate.ensemble_weight_numerators,
                *certificate.ensemble_weight_denominators,
            )
        )
        or any(
            value <= 0 for value in certificate.ensemble_weight_denominators
        )
    ):
        raise ValueError("local detector ensemble weight storage is invalid")
    if (
        max(
            (
                abs(value).bit_length()
                for value in (
                    *certificate.ensemble_weight_numerators,
                    *certificate.ensemble_weight_denominators,
                )
            ),
            default=0,
        )
        > _HARD_MAXIMUM_RATIONAL_BITS
    ):
        raise ValueError(
            "local detector ensemble weights exceed the hard bit cap"
        )
    if available:
        _validate_available_detector_intervals(
            certificate, mode_count, pixel_count, channel_count
        )
    else:
        unavailable_fields = (
            "incident_reduced_flux_intervals",
            "mode_exact_state_pixel_flux_intervals",
            "mode_production_quadratic_intervals",
            "mode_pixel_form_norm_upper_intervals",
            "mode_production_to_exact_x_amplitude_error_intervals",
            "mode_state_radius_amplitude_error_intervals",
            "mode_exact_state_amplitude_error_intervals",
            "mode_production_amplitude_norm_intervals",
            "mode_production_realization_error_upper_intervals",
            "mode_state_radius_incremental_error_upper_intervals",
            "mode_combined_exact_state_error_upper_intervals",
            "mode_outward_passivity_margin_intervals",
            "mode_pixel_fraction_intervals",
            "ideal_arrival_mean_intervals",
            "production_pre_gain_mean_point_intervals",
            "exact_state_pre_gain_mean_intervals",
            "censored_mean_intervals",
            "expected_digitized_mean_intervals",
        )
        if any(
            getattr(certificate, name) != () for name in unavailable_fields
        ):
            raise ValueError(
                "unavailable detector evidence must use empty tuples"
            )
    _validate_local_detector_work_transcript(certificate.work_transcript)
    dose = certificate.incident_electron_count
    dose_bits = _validate_raw_local_detector_interval_storage(dose)
    if dose_bits <= certificate.work_transcript.maximum_rational_bits:
        _validate_local_detector_rational_interval(dose)
    if not isinstance(
        certificate.likelihood_stage, GalerkinLocalDetectorLikelihoodStage
    ):
        raise TypeError("local detector likelihood stage is invalid")
    _validate_production_traces(certificate.production_traces)
    if available:
        _validate_helper_outcomes(
            certificate.censored_mean_transcripts,
            certificate.censored_mean_failures,
            channel_count,
            (GalerkinLocalDetectorHelperCall.EXACT_STATE_CENSORED_MEAN,),
            "exact-state censored mean",
        )
        _validate_helper_outcomes(
            certificate.production_censored_mean_transcripts,
            certificate.production_censored_mean_failures,
            channel_count,
            (GalerkinLocalDetectorHelperCall.PRODUCTION_CENSORED_MEAN,),
            "production censored mean",
        )
    elif any(
        value != ()
        for value in (
            certificate.censored_mean_transcripts,
            certificate.censored_mean_failures,
            certificate.production_censored_mean_transcripts,
            certificate.production_censored_mean_failures,
        )
    ):
        _validate_helper_outcomes(
            certificate.censored_mean_transcripts,
            certificate.censored_mean_failures,
            channel_count,
            (GalerkinLocalDetectorHelperCall.EXACT_STATE_CENSORED_MEAN,),
            "partial exact-state censored mean",
        )
        _validate_helper_outcomes(
            certificate.production_censored_mean_transcripts,
            certificate.production_censored_mean_failures,
            channel_count,
            (GalerkinLocalDetectorHelperCall.PRODUCTION_CENSORED_MEAN,),
            "partial production censored mean",
        )
    policies = (
        certificate.maximum_count_ceiling,
        certificate.maximum_poisson_work,
        certificate.maximum_poisson_rational_bits,
        certificate.exp_precision_bits,
        certificate.maximum_exp_terms,
        certificate.maximum_exp_work,
        certificate.maximum_exp_range_reductions,
    )
    if (
        any(type(value) is not int for value in policies)
        or any(value <= 0 for value in policies[:-1])
        or policies[-1] < 0
        or any(value > _MAXIMUM_SIGNED_INT64 for value in policies)
    ):
        raise ValueError("local detector Poisson policies are invalid")
    if (
        certificate.maximum_poisson_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
        or certificate.maximum_poisson_rational_bits <= 1
    ):
        raise ValueError("local detector Poisson policy does not cover model")
    for value, name in (
        (
            certificate.production_evidence_available,
            "production_evidence_available",
        ),
        (
            certificate.exact_state_censored_mean_evidence_available,
            "exact_state_censored_mean_evidence_available",
        ),
        (
            certificate.production_censored_mean_evidence_available,
            "production_censored_mean_evidence_available",
        ),
        (certificate.detector_eligible, "detector_eligible"),
        (certificate.likelihood_law_eligible, "likelihood_law_eligible"),
    ):
        _checked_scalar(value, np.dtype(np.bool_), name)
    _validate_failure_scalar(certificate.failure_mask)
    if any(
        type(value) is not str or not value.strip()
        for value in (
            certificate.flux_normalization_scope,
            certificate.ensemble_scope,
            certificate.response_scope,
            certificate.calibration_provenance,
            certificate.no_experimental_validity_scope,
        )
    ):
        raise ValueError("local detector scopes/provenance must be nonempty")
    if not _valid_digest(certificate.input_manifest_digest):
        raise ValueError("local detector input-manifest digest is invalid")
    if certificate.likelihood_stage is not (
        GalerkinLocalDetectorLikelihoodStage.PRE_GAIN_CENSORED_COUNTS
    ):
        raise ValueError("local detector likelihood stage is not canonical")
    from ptyrodactyl.galerkin.detector import (  # noqa: PLC0415
        _expected_local_censored_poisson_detector,
    )

    canonical = _expected_local_censored_poisson_detector(certificate)
    derived_fields = (
        "incident_reduced_flux_intervals",
        "mode_exact_state_pixel_flux_intervals",
        "mode_production_quadratic_intervals",
        "mode_pixel_form_norm_upper_intervals",
        "mode_production_to_exact_x_amplitude_error_intervals",
        "mode_state_radius_amplitude_error_intervals",
        "mode_exact_state_amplitude_error_intervals",
        "mode_production_amplitude_norm_intervals",
        "mode_production_realization_error_upper_intervals",
        "mode_state_radius_incremental_error_upper_intervals",
        "mode_combined_exact_state_error_upper_intervals",
        "mode_outward_passivity_margin_intervals",
        "mode_pixel_fraction_intervals",
        "ideal_arrival_mean_intervals",
        "production_pre_gain_mean_point_intervals",
        "exact_state_pre_gain_mean_intervals",
        "censored_mean_intervals",
        "expected_digitized_mean_intervals",
        "work_transcript",
        "censored_mean_transcripts",
        "censored_mean_failures",
        "production_censored_mean_transcripts",
        "production_censored_mean_failures",
    )
    if any(
        stored_value_payload(getattr(certificate, name))
        != stored_value_payload(canonical[name])
        for name in derived_fields
    ):
        raise ValueError("local detector recomputed evidence disagrees")
    canonical_traces = cast(
        _DetectorProductionTraces, canonical["production_traces"]
    )
    if tuple(
        trace.trace_digest for trace in certificate.production_traces
    ) != tuple(trace.trace_digest for trace in canonical_traces):
        raise ValueError("local detector production trace chain disagrees")
    if available != canonical["production_evidence_available"]:
        raise ValueError("local detector production availability disagrees")
    for name in (
        "exact_state_censored_mean_evidence_available",
        "production_censored_mean_evidence_available",
        "detector_eligible",
        "likelihood_law_eligible",
    ):
        if bool(np.asarray(getattr(certificate, name))) != canonical[name]:
            raise ValueError(f"local detector {name} predicate disagrees")
    if int(np.asarray(certificate.failure_mask)) != canonical["failure_mask"]:
        raise ValueError("local detector canonical failure mask disagrees")
    for name in (
        "flux_normalization_scope",
        "ensemble_scope",
        "response_scope",
        "no_experimental_validity_scope",
    ):
        if getattr(certificate, name) != canonical[name]:
            raise ValueError(f"local detector canonical {name} disagrees")
    digest_names = (
        "detector_model_identity_digest",
        "detector_model_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.censored_poisson_detector",
        digest_names=digest_names,
        identity_names=(
            "mode_state_binding_digest",
            "input_manifest_digest",
            "ensemble_weight_numerators",
            "ensemble_weight_denominators",
            "response_matrix",
            "pre_gain_background",
            "deterministic_gain",
            "electronic_offset",
            "count_ceilings",
            "fit_mask",
            "incident_electron_count",
            "incident_electron_count_point",
            "calibration_provenance",
        ),
    )
    _validate_digest_triplet(
        tuple(getattr(certificate, name) for name in digest_names), expected
    )
    return certificate


def _make_local_censored_poisson_detector(
    certificate: GalerkinLocalCensoredPoissonDetector,
) -> GalerkinLocalCensoredPoissonDetector:
    """PRIVATE: Seal and validate one owner-constructed detector carrier.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalCensoredPoissonDetector
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalCensoredPoissonDetector):
        raise TypeError("local censored-Poisson detector has the wrong type")
    digest_names = (
        "detector_model_identity_digest",
        "detector_model_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.censored_poisson_detector",
        digest_names=digest_names,
        identity_names=(
            "mode_state_binding_digest",
            "input_manifest_digest",
            "ensemble_weight_numerators",
            "ensemble_weight_denominators",
            "response_matrix",
            "pre_gain_background",
            "deterministic_gain",
            "electronic_offset",
            "count_ceilings",
            "fit_mask",
            "incident_electron_count",
            "incident_electron_count_point",
            "calibration_provenance",
        ),
    )
    sealed = replace(
        certificate,
        detector_model_identity_digest=expected[0],
        detector_model_evidence_digest=expected[1],
        certificate_digest=expected[2],
    )
    return _validate_local_censored_poisson_detector(sealed)


def _validate_likelihood_optional_intervals(
    values: object,
    *,
    fit_mask: tuple[bool, ...],
    name: str,
    positive: bool,
) -> None:
    """PRIVATE: Validate one channel-aligned fit-only interval tuple.

    Parameters
    ----------
    values : object
        Required canonical input.
    fit_mask : tuple[bool, ...]
        Required canonical input.
    name : str
        Required canonical input.
    positive : bool
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(values, tuple) or len(values) != len(fit_mask):
        raise ValueError(f"{name} must be channel aligned")
    for fitted, value in zip(fit_mask, values, strict=True):
        if not fitted:
            if value is not None:
                raise ValueError(f"unfitted {name} entries must be None")
            continue
        if value is None:
            raise ValueError(f"fitted {name} entries cannot be None")
        checked = _validate_local_detector_rational_interval(value)
        if checked.lower < 0 or (positive and checked.lower <= 0):
            raise ValueError(f"{name} has an invalid sign")


def _validate_fit_helper_outcomes(
    transcripts: object,
    failures: object,
    *,
    fit_mask: tuple[bool, ...],
    call: GalerkinLocalDetectorHelperCall,
    name: str,
) -> None:
    """PRIVATE: Validate optional fit-only helper invocation outcomes.

    Parameters
    ----------
    transcripts : object
        Required canonical input.
    failures : object
        Required canonical input.
    fit_mask : tuple[bool, ...]
        Required canonical input.
    call : GalerkinLocalDetectorHelperCall
        Required canonical input.
    name : str
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    TypeError
        If the canonical contract is violated.
    """
    if transcripts == () and failures == ():
        return
    if (
        not isinstance(transcripts, tuple)
        or not isinstance(failures, tuple)
        or len(transcripts) != len(fit_mask)
        or len(failures) != len(fit_mask)
    ):
        raise ValueError(f"{name} helper outcomes must be channel aligned")
    for channel, (fitted, transcript, failure) in enumerate(
        zip(fit_mask, transcripts, failures, strict=True)
    ):
        if not fitted:
            if transcript is not None or failure is not None:
                raise ValueError(f"unfitted {name} outcomes must be inactive")
            continue
        if (transcript is None) == (failure is None):
            raise ValueError(f"fitted {name} requires one helper outcome")
        if transcript is not None and not isinstance(
            transcript, CensoredPoissonWorkTranscript
        ):
            raise TypeError(f"{name} transcript has the wrong type")
        if failure is not None:
            checked = _validate_local_detector_helper_failure_evidence(failure)
            if checked.channel_index != channel or checked.call is not call:
                raise ValueError(f"{name} helper failure binding disagrees")


def _validate_local_censored_poisson_likelihood(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalCensoredPoissonLikelihood,
) -> GalerkinLocalCensoredPoissonLikelihood:
    """PRIVATE: Authenticate one admitted-hull censored-Poisson likelihood.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonLikelihood
        Required canonical input.

    Returns
    -------
    certificate : GalerkinLocalCensoredPoissonLikelihood
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    ValueError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalCensoredPoissonLikelihood):
        raise TypeError("local censored-Poisson likelihood has the wrong type")
    detector = _validate_local_censored_poisson_detector(certificate.detector)
    channel_count = np.asarray(detector.response_matrix).shape[0]
    _checked_vector(
        certificate.observed_counts,
        np.dtype(np.int64),
        channel_count,
        "observed_counts",
    )
    fit_mask = tuple(bool(value) for value in np.asarray(detector.fit_mask))
    available = bool(
        _checked_scalar(
            certificate.likelihood_evidence_available,
            np.dtype(np.bool_),
            "likelihood_evidence_available",
        )
    )
    law_eligible = bool(
        _checked_scalar(
            certificate.likelihood_law_eligible,
            np.dtype(np.bool_),
            "likelihood_law_eligible",
        )
    )
    nll_eligible = bool(
        _checked_scalar(
            certificate.nll_eligible,
            np.dtype(np.bool_),
            "nll_eligible",
        )
    )
    _validate_failure_scalar(certificate.failure_mask)
    _validate_local_detector_work_transcript(certificate.work_transcript)
    traces = _validate_production_traces(certificate.production_traces)
    policies = (
        certificate.log_precision_bits,
        certificate.maximum_log_terms,
        certificate.maximum_log_work,
        certificate.maximum_log_range_reductions,
    )
    if (
        any(type(value) is not int for value in policies)
        or any(value <= 0 for value in policies[:-1])
        or policies[-1] < 0
        or any(value > _MAXIMUM_SIGNED_INT64 for value in policies)
    ):
        raise ValueError("local detector logarithm policies are invalid")
    if (
        certificate.parent_detector_certificate_digest
        != detector.certificate_digest
    ):
        raise ValueError("likelihood parent detector digest disagrees")
    if any(
        type(value) is not str or not value.strip()
        for value in (
            certificate.likelihood_scope,
            certificate.nll_scope,
            certificate.no_derivative_scope,
        )
    ):
        raise ValueError("likelihood scopes must be nonempty strings")
    probability_outcomes_present = any(
        value != ()
        for value in (
            certificate.production_probability_transcripts,
            certificate.production_probability_failures,
            certificate.admitted_hull_probability_transcripts,
            certificate.admitted_hull_probability_failures,
        )
    )
    if probability_outcomes_present:
        _validate_helper_outcomes(
            certificate.production_probability_transcripts,
            certificate.production_probability_failures,
            channel_count,
            (GalerkinLocalDetectorHelperCall.PRODUCTION_PROBABILITY,),
            "production probability",
        )
        _validate_helper_outcomes(
            certificate.admitted_hull_probability_transcripts,
            certificate.admitted_hull_probability_failures,
            channel_count,
            (GalerkinLocalDetectorHelperCall.ADMITTED_HULL_PROBABILITY,),
            "admitted-hull probability",
        )
    _validate_fit_helper_outcomes(
        certificate.production_nll_transcripts,
        certificate.production_nll_failures,
        fit_mask=fit_mask,
        call=GalerkinLocalDetectorHelperCall.PRODUCTION_NLL,
        name="production NLL",
    )
    _validate_fit_helper_outcomes(
        certificate.admitted_hull_nll_transcripts,
        certificate.admitted_hull_nll_failures,
        fit_mask=fit_mask,
        call=GalerkinLocalDetectorHelperCall.ADMITTED_HULL_NLL,
        name="admitted-hull NLL",
    )
    if available:
        admitted = _validate_intervals(
            certificate.admitted_pre_gain_mean_hull_intervals,
            channel_count,
            "admitted_pre_gain_mean_hull_intervals",
            nonnegative=True,
        )
        probability_points = _validate_intervals(
            certificate.production_probability_point_intervals,
            channel_count,
            "production_probability_point_intervals",
            nonnegative=True,
        )
        admitted_probabilities = _validate_intervals(
            certificate.admitted_hull_probability_intervals,
            channel_count,
            "admitted_hull_probability_intervals",
            nonnegative=True,
        )
        if any(value.lower != value.upper for value in probability_points):
            raise ValueError(
                "production probability points must be singletons"
            )
        if any(value.upper > 1 for value in admitted_probabilities):
            raise ValueError("censored probabilities cannot exceed one")
        if len(certificate.fitted_probability_positive_floor_intervals) != (
            channel_count
        ):
            raise ValueError(
                "likelihood probability floors must be channel aligned"
            )
        for fitted, floor, probability in zip(
            fit_mask,
            certificate.fitted_probability_positive_floor_intervals,
            admitted_probabilities,
            strict=True,
        ):
            if not fitted:
                if floor is not None:
                    raise ValueError(
                        "unfitted probability floors must be None"
                    )
            elif floor is not None:
                checked = _validate_local_detector_rational_interval(floor)
                if (
                    checked.lower <= 0
                    or checked.lower != checked.upper
                    or checked.lower != probability.lower
                ):
                    raise ValueError(
                        "fitted probability floor is not canonical"
                    )
        if nll_eligible:
            _validate_likelihood_optional_intervals(
                certificate.production_nll_point_intervals,
                fit_mask=fit_mask,
                name="production NLL points",
                positive=False,
            )
            _validate_likelihood_optional_intervals(
                certificate.admitted_hull_nll_intervals,
                fit_mask=fit_mask,
                name="admitted-hull NLL intervals",
                positive=False,
            )
            if any(
                value is not None and value.lower != value.upper
                for value in certificate.production_nll_point_intervals
            ):
                raise ValueError("production NLL points must be singletons")
            if certificate.total_nll_interval is None:
                raise ValueError("eligible likelihood requires total NLL")
            total = _validate_local_detector_rational_interval(
                certificate.total_nll_interval
            )
            if total.lower < 0:
                raise ValueError("total NLL must be nonnegative")
        elif (
            certificate.production_nll_point_intervals != ()
            or certificate.admitted_hull_nll_intervals != ()
            or certificate.total_nll_interval is not None
        ):
            raise ValueError(
                "unavailable NLL evidence must use empty sentinels"
            )
        if not probability_outcomes_present:
            raise ValueError(
                "available likelihood requires probability outcomes"
            )
        if not law_eligible:
            raise ValueError(
                "available probability evidence must establish its law"
            )
        if not admitted:
            raise ValueError("available likelihood admitted hulls are empty")
    else:
        unavailable_fields = (
            certificate.admitted_pre_gain_mean_hull_intervals,
            certificate.production_probability_point_intervals,
            certificate.admitted_hull_probability_intervals,
            certificate.fitted_probability_positive_floor_intervals,
            certificate.production_nll_point_intervals,
            certificate.admitted_hull_nll_intervals,
        )
        if any(value != () for value in unavailable_fields) or (
            certificate.total_nll_interval is not None
        ):
            raise ValueError("unavailable likelihood values must be empty")
        if law_eligible or nll_eligible:
            raise ValueError("unavailable likelihood cannot claim eligibility")
    if nll_eligible and not law_eligible:
        raise ValueError(
            "NLL eligibility requires probability-law eligibility"
        )
    if traces and any(
        trace.stage
        not in (
            GalerkinLocalDetectorProductionStage.CENSORED_PROBABILITY,
            GalerkinLocalDetectorProductionStage.CENSORED_NLL,
        )
        for trace in traces
    ):
        raise ValueError("likelihood trace has a nonlikelihood stage")
    from ptyrodactyl.galerkin.detector import (  # noqa: PLC0415
        _expected_local_censored_poisson_likelihood,
    )

    canonical = _expected_local_censored_poisson_likelihood(certificate)
    derived_fields = (
        "admitted_pre_gain_mean_hull_intervals",
        "production_probability_point_intervals",
        "admitted_hull_probability_intervals",
        "fitted_probability_positive_floor_intervals",
        "production_nll_point_intervals",
        "admitted_hull_nll_intervals",
        "total_nll_interval",
        "production_probability_transcripts",
        "production_probability_failures",
        "admitted_hull_probability_transcripts",
        "admitted_hull_probability_failures",
        "production_nll_transcripts",
        "production_nll_failures",
        "admitted_hull_nll_transcripts",
        "admitted_hull_nll_failures",
        "work_transcript",
    )
    if any(
        stored_value_payload(getattr(certificate, name))
        != stored_value_payload(canonical[name])
        for name in derived_fields
    ):
        raise ValueError("local likelihood recomputed evidence disagrees")
    canonical_traces = cast(
        _DetectorProductionTraces, canonical["production_traces"]
    )
    if tuple(
        trace.trace_digest for trace in certificate.production_traces
    ) != tuple(trace.trace_digest for trace in canonical_traces):
        raise ValueError("local likelihood production trace chain disagrees")
    for name in (
        "likelihood_evidence_available",
        "likelihood_law_eligible",
        "nll_eligible",
    ):
        if bool(np.asarray(getattr(certificate, name))) != canonical[name]:
            raise ValueError(f"local likelihood {name} predicate disagrees")
    if int(np.asarray(certificate.failure_mask)) != canonical["failure_mask"]:
        raise ValueError("local likelihood failure mask disagrees")
    for name in ("likelihood_scope", "nll_scope", "no_derivative_scope"):
        if getattr(certificate, name) != canonical[name]:
            raise ValueError(f"local likelihood canonical {name} disagrees")
    digest_names = (
        "likelihood_identity_digest",
        "likelihood_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.censored_poisson_likelihood",
        digest_names=digest_names,
        identity_names=(
            "parent_detector_certificate_digest",
            "observed_counts",
            "log_precision_bits",
            "maximum_log_terms",
            "maximum_log_work",
            "maximum_log_range_reductions",
        ),
    )
    _validate_digest_triplet(
        tuple(getattr(certificate, name) for name in digest_names), expected
    )
    return certificate


def _make_local_censored_poisson_likelihood(
    certificate: GalerkinLocalCensoredPoissonLikelihood,
) -> GalerkinLocalCensoredPoissonLikelihood:
    """PRIVATE: Seal and validate one owner-constructed likelihood carrier.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonLikelihood
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalCensoredPoissonLikelihood
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    if not isinstance(certificate, GalerkinLocalCensoredPoissonLikelihood):
        raise TypeError("local censored-Poisson likelihood has the wrong type")
    digest_names = (
        "likelihood_identity_digest",
        "likelihood_evidence_digest",
        "certificate_digest",
    )
    expected = _expected_carrier_digests(
        certificate,
        domain="ptyrodactyl.local_detector.censored_poisson_likelihood",
        digest_names=digest_names,
        identity_names=(
            "parent_detector_certificate_digest",
            "observed_counts",
            "log_precision_bits",
            "maximum_log_terms",
            "maximum_log_work",
            "maximum_log_range_reductions",
        ),
    )
    sealed = replace(
        certificate,
        likelihood_identity_digest=expected[0],
        likelihood_evidence_digest=expected[1],
        certificate_digest=expected[2],
    )
    return _validate_local_censored_poisson_likelihood(sealed)


__all__: list[str] = [
    "GalerkinLocalCensoredPoissonDetector",
    "GalerkinLocalCensoredPoissonDetectorInputManifest",
    "GalerkinLocalCensoredPoissonLikelihood",
    "GalerkinLocalDetectorCoordinateConvention",
    "GalerkinLocalDetectorFailure",
    "GalerkinLocalDetectorHelperCall",
    "GalerkinLocalDetectorHelperFailureEvidence",
    "GalerkinLocalDetectorLikelihoodStage",
    "GalerkinLocalDetectorProductionStage",
    "GalerkinLocalDetectorRationalInterval",
    "GalerkinLocalDetectorRealProductionTrace",
    "GalerkinLocalDetectorWorkTranscript",
    "GalerkinLocalPassivePixelForms",
    "GalerkinLocalPassivePixelInputManifest",
    "GalerkinLocalPositivePortBranchDisposition",
    "GalerkinLocalPositivePortCertificate",
    "GalerkinLocalPositivePortRoute",
]
