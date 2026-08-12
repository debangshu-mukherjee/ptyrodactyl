r"""Define owned RM-S5 residual-to-observable evidence carriers.

Extended Summary
----------------
These carriers bind one replayed RM-S4 censored-Poisson likelihood to the
layered RM-S5 mean, law, and pointwise-NLL conclusions.  Mean and law
evidence is derived only from the nested detector.  Likelihood evidence is
used only for the separately gated direct and score NLL routes.

Routine Listings
----------------
:class:`GalerkinLocalResidualObservableCertificate`
    Store one layered residual-to-observable certificate.
:class:`GalerkinLocalResidualObservableFailure`
    Enumerate typed RM-S5 noncertificate outcomes.
:class:`GalerkinLocalResidualObservableHelperCall`
    Name one bounded RM-S5 nested helper invocation.
:class:`GalerkinLocalResidualObservableHelperFailureEvidence`
    Store one replayable RM-S5 helper failure.
:class:`GalerkinLocalResidualObservableInputManifest`
    Bind the L9 replay inputs and every RM-S5 resource policy.
:class:`GalerkinLocalResidualObservableLayer`
    Name the strongest retained RM-S5 evidence layer.
:class:`GalerkinLocalResidualObservableRoute`
    Name the selected pointwise-NLL error route.
:class:`GalerkinLocalResidualObservableScope`
    Distinguish the full law from its fixed fitted projection.
:class:`GalerkinLocalResidualObservableWorkTranscript`
    Store staged bounded exact-work evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntFlag, StrEnum
from fractions import Fraction

import equinox as eqx
import jax
import numpy as np
from beartype.typing import Any
from jaxtyping import Array, Bool, Int64

from ptyrodactyl._tools import (
    CensoredPoissonEnclosureError,
    CensoredPoissonEnclosureFailure,
    CensoredPoissonWorkTranscript,
    EntireEnclosureError,
    EntireEnclosureFailure,
    EntireWorkTranscript,
    enclose_censored_poisson_probability,
    enclose_real_exp,
    sha256,
    stored_value_payload,
)

from .local_detector_types import (
    GalerkinLocalCensoredPoissonDetectorInputManifest,
    GalerkinLocalCensoredPoissonLikelihood,
    GalerkinLocalDetectorProductionStage,
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalDetectorRealProductionTrace,
    _make_local_detector_rational_interval,
    _validate_local_censored_poisson_detector_input_manifest,
    _validate_local_censored_poisson_likelihood,
    _validate_local_detector_rational_interval,
    _validate_local_detector_real_production_trace,
    _validate_prior_entire_transcripts,
)

_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_MAXIMUM_SIGNED_INT64: int = (1 << 63) - 1
_SHA256_HEX_LENGTH: int = 64
_RESIDUAL_OBSERVABLE_WORK_ALGORITHM: str = (
    "exact_fraction_local_residual_observable_v1"
)
_NESTED_PARENT_WORK_SCOPE: str = (
    "nested_parent_work_count_exact authenticates the complete replayed L9 "
    "likelihood work tree; local RM-S5 arithmetic is counted separately"
)
_MEAN_SCOPE: str = (
    "detector-derived pre-gain exact-state/production hulls and endpoint "
    "distances; likelihood hulls are consistency checks only"
)
_LAW_SCOPE: str = (
    "independent-Poisson full-channel law and its fixed fit-mask projection; "
    "linear TV remains valid when exponential tightening is unavailable"
)
_NLL_SCOPE: str = (
    "global direct admitted-hull NLL error and fitted score-Lipschitz error; "
    "the tighter available route is selected"
)
_RESOURCE_SCOPE: str = (
    "mean, law, direct-NLL, and score-NLL stages stop independently at exact "
    "resource boundaries and retain every completed weaker layer"
)
_NO_SCIENTIFIC_CLAIM_SCOPE: str = (
    "RM-S5 algebraic error evidence makes no calibration, shot-noise, "
    "detectability, resolution, or physical-model claim"
)


class _ResidualObservableArithmeticError(ArithmeticError):
    """PRIVATE: Report one exact local rational-size stop."""

    exact_work_count: int
    rational_peak_bits: int

    def __init__(self, exact_work_count: int, rational_peak_bits: int) -> None:
        super().__init__("residual-observable rational size limit exceeded")
        self.exact_work_count = exact_work_count
        self.rational_peak_bits = rational_peak_bits


@dataclass
class _ResidualObservableLedger:
    """PRIVATE: Meter and bound every issued local exact transaction."""

    maximum_rational_bits: int
    exact_work_count: int = 0
    rational_peak_bits: int = 0

    def _retain(self, *values: Fraction) -> None:
        """Retain exact results and stop after the issuing transaction."""
        for value in values:
            self.rational_peak_bits = max(
                self.rational_peak_bits,
                1,
                abs(value.numerator).bit_length(),
                value.denominator.bit_length(),
            )
        if self.rational_peak_bits > self.maximum_rational_bits:
            raise _ResidualObservableArithmeticError(
                self.exact_work_count, self.rational_peak_bits
            )

    def scan_intervals(self, values: object) -> None:
        """Check raw primitive storage without materializing a Fraction."""
        self.rational_peak_bits = max(
            self.rational_peak_bits, _intervals_peak_bits(values)
        )
        if self.rational_peak_bits > self.maximum_rational_bits:
            raise _ResidualObservableArithmeticError(
                self.exact_work_count, self.rational_peak_bits
            )

    def commit(self, *values: Fraction) -> None:
        """Charge one canonical compound exact transaction."""
        self.exact_work_count += 1
        self._retain(*values)

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        """Charge and return one exact addition."""
        result = left + right
        self.commit(result)
        return result

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        """Charge and return one exact subtraction."""
        result = left - right
        self.commit(result)
        return result

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        """Charge and return one exact multiplication."""
        result = left * right
        self.commit(result)
        return result

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        """Charge and return one exact division by a positive value."""
        result = numerator / denominator
        self.commit(result)
        return result

    def score_accumulate(
        self,
        factor: Fraction,
        distance: Fraction,
        rounding: Fraction,
        total: Fraction,
    ) -> tuple[Fraction, Fraction]:
        """Charge the two canonical per-channel score transactions."""
        product = self.multiply(factor, distance)
        term = product + rounding
        updated_total = total + term
        self.commit(term, updated_total)
        return term, updated_total


class GalerkinLocalResidualObservableLayer(StrEnum):
    """Name the strongest retained RM-S5 evidence layer.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_enums_are_explicit_disjoint_and_layered`
    """

    UNAVAILABLE = "unavailable"
    STATE = "state"
    MEAN = "mean"
    LAW = "law"
    POINTWISE_NLL = "pointwise_nll"


class GalerkinLocalResidualObservableScope(StrEnum):
    """Distinguish the full law from its fixed fitted projection.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_enums_are_explicit_disjoint_and_layered`
    """

    FULL_CHANNEL_LAW = "full_channel_law"
    FIXED_FIT_PROJECTION = "fixed_fit_projection"


class GalerkinLocalResidualObservableRoute(StrEnum):
    """Name the selected pointwise-NLL error route.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_enums_are_explicit_disjoint_and_layered`
    """

    UNAVAILABLE = "unavailable"
    DIRECT_ADMITTED_HULL = "direct_admitted_hull"
    SCORE_LIPSCHITZ = "score_lipschitz"
    TIED = "tied"


class GalerkinLocalResidualObservableHelperCall(StrEnum):
    """Name one bounded RM-S5 nested helper invocation.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_enums_are_explicit_disjoint_and_layered`
    """

    FULL_TV_EXPONENTIAL = "full_tv_exponential"
    FITTED_TV_EXPONENTIAL = "fitted_tv_exponential"
    SATURATED_SCORE_MASS = "saturated_score_mass"


class GalerkinLocalResidualObservableFailure(IntFlag):
    """Enumerate typed RM-S5 noncertificate outcomes.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_enums_are_explicit_disjoint_and_layered`
    """

    NONE = 0
    PARENT_STATE_NONCERTIFICATE = 1 << 0
    PARENT_MEAN_NONCERTIFICATE = 1 << 1
    PARENT_LAW_NONCERTIFICATE = 1 << 2
    ADMITTED_MEAN_HULL_UNAVAILABLE = 1 << 3
    PRODUCTION_MEAN_NONSINGLETON = 1 << 4
    PRODUCTION_MEAN_OUTSIDE_HULL = 1 << 5
    DIRECT_NLL_UNAVAILABLE = 1 << 6
    SCORE_MEAN_FLOOR_UNAVAILABLE = 1 << 7
    SCORE_PROBABILITY_FLOOR_UNAVAILABLE = 1 << 8
    EXPONENTIAL_ENCLOSURE_FAILURE = 1 << 9
    POISSON_ENCLOSURE_FAILURE = 1 << 10
    NESTED_HELPER_FAILURE = 1 << 11
    EXACT_WORK_BUDGET_EXCEEDED = 1 << 12
    EXACT_WORK_COUNT_OVERFLOW = 1 << 13
    RATIONAL_SIZE_LIMIT = 1 << 14
    ARITHMETIC_RANGE_FAILURE = 1 << 15


_KNOWN_RESIDUAL_OBSERVABLE_FAILURE_MASK: int = sum(
    int(member) for member in GalerkinLocalResidualObservableFailure
)


type _Intervals = tuple[GalerkinLocalDetectorRationalInterval, ...]
type _OptionalIntervals = tuple[
    GalerkinLocalDetectorRationalInterval | None, ...
]


class GalerkinLocalResidualObservableInputManifest(eqx.Module):
    """Bind the L9 replay inputs and every RM-S5 resource policy.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_manifest_is_owner_sealed_and_policy_complete`
    """

    detector_input_manifest: GalerkinLocalCensoredPoissonDetectorInputManifest
    observed_counts: Int64[Array, " r"]
    maximum_detector_work: int = eqx.field(static=True)
    maximum_detector_rational_bits: int = eqx.field(static=True)
    log_precision_bits: int = eqx.field(static=True)
    maximum_log_terms: int = eqx.field(static=True)
    maximum_log_work: int = eqx.field(static=True)
    maximum_log_range_reductions: int = eqx.field(static=True)
    tv_exp_precision_bits: int = eqx.field(static=True)
    maximum_tv_exp_terms: int = eqx.field(static=True)
    maximum_tv_exp_work: int = eqx.field(static=True)
    maximum_tv_exp_range_reductions: int = eqx.field(static=True)
    maximum_residual_observable_work: int = eqx.field(static=True)
    maximum_residual_observable_rational_bits: int = eqx.field(static=True)
    manifest_digest: str = eqx.field(static=True)


class GalerkinLocalResidualObservableHelperFailureEvidence(eqx.Module):
    """Store one replayable RM-S5 helper failure.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_work_and_helper_evidence_bind_partial_stages`
    """

    call: GalerkinLocalResidualObservableHelperCall = eqx.field(static=True)
    channel_index: int | None = eqx.field(static=True)
    entire_failure: EntireEnclosureFailure | None = eqx.field(static=True)
    poisson_failure: CensoredPoissonEnclosureFailure | None = eqx.field(
        static=True
    )
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


class GalerkinLocalResidualObservableWorkTranscript(eqx.Module):
    """Store staged bounded exact-work evidence.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_work_and_helper_evidence_bind_partial_stages`
    """

    algorithm: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    fitted_channel_count: int = eqx.field(static=True)
    zero_observation_positive_ceiling_count: int = eqx.field(static=True)
    interior_observation_count: int = eqx.field(static=True)
    saturated_positive_ceiling_count: int = eqx.field(static=True)
    mean_exact_work_count: int = eqx.field(static=True)
    law_exact_work_count: int = eqx.field(static=True)
    direct_nll_exact_work_count: int = eqx.field(static=True)
    score_nll_exact_work_count: int = eqx.field(static=True)
    exact_work_count: int = eqx.field(static=True)
    rational_peak_bits: int = eqx.field(static=True)
    nested_parent_work_count_exact: str = eqx.field(static=True)
    nested_helper_work_count_exact: str = eqx.field(static=True)
    planned_mean_exact_work_count_exact: str = eqx.field(static=True)
    planned_law_exact_work_count_exact: str = eqx.field(static=True)
    planned_direct_nll_exact_work_count_exact: str = eqx.field(static=True)
    planned_score_nll_exact_work_count_exact: str = eqx.field(static=True)
    attempted_exact_work_count_exact: str = eqx.field(static=True)
    completed_layer: GalerkinLocalResidualObservableLayer = eqx.field(
        static=True
    )
    completed_successfully: bool = eqx.field(static=True)
    failure: GalerkinLocalResidualObservableFailure = eqx.field(static=True)
    preflight_failed: bool = eqx.field(static=True)
    count_overflow: bool = eqx.field(static=True)
    nested_parent_work_scope: str = eqx.field(static=True)


class GalerkinLocalResidualObservableCertificate(eqx.Module):
    """Store one layered residual-to-observable certificate.

    :see: :func:`~.test_local_residual_observable_types.\
test_residual_observable_certificate_schema_is_layered_and_has_no_gradient`
    """

    parent_likelihood: GalerkinLocalCensoredPoissonLikelihood
    input_manifest: GalerkinLocalResidualObservableInputManifest
    state_evidence_available: Bool[Array, ""]
    mean_evidence_available: Bool[Array, ""]
    law_evidence_available: Bool[Array, ""]
    full_law_evidence_available: Bool[Array, ""]
    fitted_law_evidence_available: Bool[Array, ""]
    direct_nll_evidence_available: Bool[Array, ""]
    score_nll_evidence_available: Bool[Array, ""]
    selected_nll_evidence_available: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    admitted_pre_gain_mean_hull_intervals: _Intervals | None = eqx.field(
        static=True
    )
    channel_mean_error_bound_intervals: _Intervals | None = eqx.field(
        static=True
    )
    full_mean_l1_error_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    fitted_mean_l1_error_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    full_linear_tv_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    fitted_linear_tv_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    full_exponential_tv_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    fitted_exponential_tv_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    full_selected_tv_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    fitted_selected_tv_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    production_fitted_total_nll_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    direct_nll_error_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    score_lipschitz_factor_intervals: _OptionalIntervals = eqx.field(
        static=True
    )
    score_rounding_error_intervals: _OptionalIntervals = eqx.field(static=True)
    score_term_error_intervals: _OptionalIntervals = eqx.field(static=True)
    score_nll_error_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    selected_nll_error_bound_interval: (
        GalerkinLocalDetectorRationalInterval | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    saturated_predecessor_mass_upper_intervals: _OptionalIntervals = eqx.field(
        static=True
    )
    saturated_tail_probability_floor_intervals: _OptionalIntervals = eqx.field(
        static=True
    )
    full_tv_exp_transcript: EntireWorkTranscript | None = eqx.field(
        static=True
    )
    fitted_tv_exp_transcript: EntireWorkTranscript | None = eqx.field(
        static=True
    )
    full_tv_exp_failure: (
        GalerkinLocalResidualObservableHelperFailureEvidence | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    fitted_tv_exp_failure: (
        GalerkinLocalResidualObservableHelperFailureEvidence | None
    ) = eqx.field(  # noqa: E501
        static=True
    )
    saturated_probability_transcripts: tuple[
        CensoredPoissonWorkTranscript | None, ...
    ] = eqx.field(static=True)
    saturated_probability_failures: tuple[
        GalerkinLocalResidualObservableHelperFailureEvidence | None, ...
    ] = eqx.field(static=True)
    work_transcript: GalerkinLocalResidualObservableWorkTranscript = eqx.field(
        static=True
    )
    strongest_layer: GalerkinLocalResidualObservableLayer = eqx.field(
        static=True
    )
    selected_nll_route: GalerkinLocalResidualObservableRoute = eqx.field(
        static=True
    )
    full_law_scope: GalerkinLocalResidualObservableScope = eqx.field(
        static=True
    )
    fitted_law_scope: GalerkinLocalResidualObservableScope = eqx.field(
        static=True
    )
    mean_scope: str = eqx.field(static=True)
    law_scope: str = eqx.field(static=True)
    nll_scope: str = eqx.field(static=True)
    resource_scope: str = eqx.field(static=True)
    no_scientific_claim_scope: str = eqx.field(static=True)
    parent_detector_certificate_digest: str = eqx.field(static=True)
    parent_detector_input_manifest_digest: str = eqx.field(static=True)
    parent_likelihood_certificate_digest: str = eqx.field(static=True)
    input_manifest_digest: str = eqx.field(static=True)
    observable_identity_digest: str = eqx.field(static=True)
    observable_evidence_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


def _valid_digest(value: object) -> bool:
    """PRIVATE: Return whether ``value`` is one lowercase SHA-256 digest.

    Parameters
    ----------
    value : object
        Required canonical input.

    Returns
    -------
    result : bool
        Whether the value is canonical.
    """
    return (
        type(value) is str
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_concrete_array_carrier(value: object) -> bool:
    """PRIVATE: Return whether ``value`` is one admitted array carrier.

    Parameters
    ----------
    value : object
        Candidate NumPy or JAX array.

    Returns
    -------
    result : bool
        Whether the carrier preserves declared array identity.
    """
    return type(value) is np.ndarray or isinstance(value, jax.Array)


def _canonical_decimal(value: object, name: str) -> int:
    """PRIVATE: Decode one canonical nonnegative decimal integer.

    Parameters
    ----------
    value : object
        Required canonical input.
    name : str
        Field name used in diagnostics.

    Returns
    -------
    result : int
        Canonical decoded value.

    Raises
    ------
    ValueError
        If the value is not a canonical decimal string.
    """
    if (
        type(value) is not str
        or not value
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError(f"{name} must be a canonical nonnegative decimal")
    return int(value)


def _manifest_digest(
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> str:
    """PRIVATE: Return the canonical RM-S5 primitive-manifest digest.

    Parameters
    ----------
    manifest : GalerkinLocalResidualObservableInputManifest
        Required canonical input.

    Returns
    -------
    result : str
        Canonical digest.
    """
    return sha256(
        {
            "domain": "ptyrodactyl.local_residual_observable.input.v1",
            "fields": {
                field.name: stored_value_payload(getattr(manifest, field.name))
                for field in fields(manifest)
                if field.name != "manifest_digest"
            },
        }
    )


def _make_local_residual_observable_input_manifest_candidate(  # noqa: PLR0913
    *,
    detector_input_manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
    observed_counts: Int64[Array, " r"],
    maximum_detector_work: int,
    maximum_detector_rational_bits: int,
    log_precision_bits: int,
    maximum_log_terms: int,
    maximum_log_work: int,
    maximum_log_range_reductions: int,
    tv_exp_precision_bits: int,
    maximum_tv_exp_terms: int,
    maximum_tv_exp_work: int,
    maximum_tv_exp_range_reductions: int,
    maximum_residual_observable_work: int,
    maximum_residual_observable_rational_bits: int,
    manifest_digest: str,
) -> GalerkinLocalResidualObservableInputManifest:
    """PRIVATE: Construct one unsealed RM-S5 input-manifest candidate.

    Parameters
    ----------
    detector_input_manifest : GalerkinLocalCensoredPoissonDetectorInputManifest
        Required canonical input.
    observed_counts : Int64[Array, " r"]
        Required canonical input.
    maximum_detector_work : int
        Required canonical input.
    maximum_detector_rational_bits : int
        Required canonical input.
    log_precision_bits : int
        Required canonical input.
    maximum_log_terms : int
        Required canonical input.
    maximum_log_work : int
        Required canonical input.
    maximum_log_range_reductions : int
        Required canonical input.
    tv_exp_precision_bits : int
        Required canonical input.
    maximum_tv_exp_terms : int
        Required canonical input.
    maximum_tv_exp_work : int
        Required canonical input.
    maximum_tv_exp_range_reductions : int
        Required canonical input.
    maximum_residual_observable_work : int
        Required canonical input.
    maximum_residual_observable_rational_bits : int
        Required canonical input.
    manifest_digest : str
        Placeholder digest replaced by the owner sealer.

    Returns
    -------
    result : GalerkinLocalResidualObservableInputManifest
        Unsealed candidate.
    """
    return GalerkinLocalResidualObservableInputManifest(
        detector_input_manifest=detector_input_manifest,
        observed_counts=observed_counts,
        maximum_detector_work=maximum_detector_work,
        maximum_detector_rational_bits=maximum_detector_rational_bits,
        log_precision_bits=log_precision_bits,
        maximum_log_terms=maximum_log_terms,
        maximum_log_work=maximum_log_work,
        maximum_log_range_reductions=maximum_log_range_reductions,
        tv_exp_precision_bits=tv_exp_precision_bits,
        maximum_tv_exp_terms=maximum_tv_exp_terms,
        maximum_tv_exp_work=maximum_tv_exp_work,
        maximum_tv_exp_range_reductions=maximum_tv_exp_range_reductions,
        maximum_residual_observable_work=(maximum_residual_observable_work),
        maximum_residual_observable_rational_bits=(
            maximum_residual_observable_rational_bits
        ),
        manifest_digest=manifest_digest,
    )


def _prevalidate_local_residual_observable_input_manifest(
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> None:
    """PRIVATE: Validate manifest primitives before digest serialization.

    Parameters
    ----------
    manifest : GalerkinLocalResidualObservableInputManifest
        Candidate manifest.

    Raises
    ------
    TypeError
        If a carrier, array, or policy has the wrong exact type.
    ValueError
        If a shape, domain, policy, or digest is invalid.
    """
    if type(manifest) is not GalerkinLocalResidualObservableInputManifest:
        raise TypeError("residual-observable manifest has the wrong type")
    detector_manifest = (
        _validate_local_censored_poisson_detector_input_manifest(
            manifest.detector_input_manifest
        )
    )
    if not _is_concrete_array_carrier(manifest.observed_counts):
        raise TypeError(
            "residual-observable observed counts must be an array carrier"
        )
    observed = np.asarray(manifest.observed_counts)
    ceilings = np.asarray(detector_manifest.count_ceilings)
    if (
        observed.dtype != np.dtype(np.int64)
        or observed.ndim != 1
        or observed.shape != ceilings.shape
    ):
        raise ValueError(
            "residual-observable observed counts have invalid shape or dtype"
        )
    if bool(np.any(observed < 0)) or bool(np.any(observed > ceilings)):
        raise ValueError(
            "residual-observable observed counts exceed their ceilings"
        )


def _validate_local_residual_observable_input_manifest(
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> GalerkinLocalResidualObservableInputManifest:
    """PRIVATE: Validate one independently replayable RM-S5 manifest.

    Parameters
    ----------
    manifest : GalerkinLocalResidualObservableInputManifest
        Candidate manifest.

    Returns
    -------
    result : GalerkinLocalResidualObservableInputManifest
        Structurally validated manifest.

    Raises
    ------
    TypeError
        If a carrier, array, or policy has the wrong exact type.
    ValueError
        If a shape, domain, policy, or digest is invalid.
    """
    _prevalidate_local_residual_observable_input_manifest(manifest)
    policies = (
        manifest.maximum_detector_work,
        manifest.maximum_detector_rational_bits,
        manifest.log_precision_bits,
        manifest.maximum_log_terms,
        manifest.maximum_log_work,
        manifest.maximum_log_range_reductions,
        manifest.tv_exp_precision_bits,
        manifest.maximum_tv_exp_terms,
        manifest.maximum_tv_exp_work,
        manifest.maximum_tv_exp_range_reductions,
        manifest.maximum_residual_observable_work,
        manifest.maximum_residual_observable_rational_bits,
    )
    if any(type(value) is not int for value in policies):
        raise TypeError("residual-observable policies must use Python ints")
    positive = (
        manifest.maximum_detector_work,
        manifest.log_precision_bits,
        manifest.maximum_log_terms,
        manifest.maximum_log_work,
        manifest.tv_exp_precision_bits,
        manifest.maximum_tv_exp_terms,
        manifest.maximum_tv_exp_work,
        manifest.maximum_residual_observable_work,
    )
    reductions = (
        manifest.maximum_log_range_reductions,
        manifest.maximum_tv_exp_range_reductions,
    )
    rational_policies = (
        manifest.maximum_detector_rational_bits,
        manifest.maximum_residual_observable_rational_bits,
    )
    if (
        any(value <= 0 or value > _MAXIMUM_SIGNED_INT64 for value in positive)
        or any(
            value < 0 or value > _MAXIMUM_SIGNED_INT64 for value in reductions
        )
        or any(
            value <= 1 or value > _HARD_MAXIMUM_RATIONAL_BITS
            for value in rational_policies
        )
        or manifest.log_precision_bits + 1
        > manifest.maximum_detector_rational_bits
        or manifest.tv_exp_precision_bits + 1
        > manifest.maximum_residual_observable_rational_bits
    ):
        raise ValueError("residual-observable policies are invalid")
    expected = _manifest_digest(manifest)
    if not _valid_digest(manifest.manifest_digest):
        raise ValueError("residual-observable manifest digest is not SHA-256")
    if manifest.manifest_digest != expected:
        raise ValueError("residual-observable manifest digest disagrees")
    return manifest


def _make_local_residual_observable_input_manifest(
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> GalerkinLocalResidualObservableInputManifest:
    """PRIVATE: Seal and validate one owner-constructed RM-S5 manifest.

    Parameters
    ----------
    manifest : GalerkinLocalResidualObservableInputManifest
        Unsealed owner candidate.

    Returns
    -------
    result : GalerkinLocalResidualObservableInputManifest
        Canonical sealed manifest.

    Raises
    ------
    TypeError
        If the candidate has the wrong exact type.
    ValueError
        If its primitive structure or policy is invalid.
    """
    if type(manifest) is not GalerkinLocalResidualObservableInputManifest:
        raise TypeError("residual-observable manifest has the wrong type")
    _prevalidate_local_residual_observable_input_manifest(manifest)
    sealed = replace(manifest, manifest_digest=_manifest_digest(manifest))
    return _validate_local_residual_observable_input_manifest(sealed)


def _helper_failure_digest(
    evidence: GalerkinLocalResidualObservableHelperFailureEvidence,
) -> str:
    """PRIVATE: Return the canonical digest for one helper failure.

    Parameters
    ----------
    evidence : GalerkinLocalResidualObservableHelperFailureEvidence
        Required canonical input.

    Returns
    -------
    result : str
        Canonical digest.
    """
    return sha256(
        {
            "domain": "ptyrodactyl.local_residual_observable.helper.v1",
            "fields": {
                field.name: stored_value_payload(getattr(evidence, field.name))
                for field in fields(evidence)
                if field.name != "failure_digest"
            },
        }
    )


def _validate_residual_observable_entire_transcripts(
    prior_exp_transcripts: object,
    prior_log_transcripts: object,
) -> None:
    """PRIVATE: Validate exact transcript lanes without type erasure.

    Parameters
    ----------
    prior_exp_transcripts : object
        Candidate completed exponential transcript tuple.
    prior_log_transcripts : object
        Candidate completed logarithm transcript tuple.

    Raises
    ------
    TypeError
        If a tuple, transcript, or algorithm has the wrong exact type.
    ValueError
        If the authoritative entire-transcript contract is violated.
    """
    if (
        type(prior_exp_transcripts) is not tuple
        or type(prior_log_transcripts) is not tuple
    ):
        raise TypeError(
            "residual-observable entire transcript lanes must be exact tuples"
        )
    for transcript in prior_exp_transcripts + prior_log_transcripts:
        if type(transcript) is not EntireWorkTranscript:
            raise TypeError(
                "residual-observable entire transcript has the wrong type"
            )
        if type(transcript.algorithm) is not str:
            raise TypeError(
                "residual-observable entire algorithm must be an exact str"
            )
    _validate_prior_entire_transcripts(
        prior_exp_transcripts,
        prior_log_transcripts,
    )


def _prevalidate_local_residual_observable_helper_failure(
    evidence: GalerkinLocalResidualObservableHelperFailureEvidence,
) -> None:
    """PRIVATE: Validate helper shapes before digest serialization.

    Parameters
    ----------
    evidence : GalerkinLocalResidualObservableHelperFailureEvidence
        Candidate helper failure.

    Raises
    ------
    TypeError
        If the carrier, enum, channel, or transcript tuple has a wrong exact
        type.
    """
    if (
        type(evidence)
        is not GalerkinLocalResidualObservableHelperFailureEvidence
    ):
        raise TypeError("residual-observable helper failure has wrong type")
    if type(evidence.call) is not GalerkinLocalResidualObservableHelperCall:
        raise TypeError("residual-observable helper call has wrong enum type")
    if evidence.channel_index is not None and (
        type(evidence.channel_index) is not int or evidence.channel_index < 0
    ):
        raise TypeError("residual-observable helper channel is invalid")
    if (
        type(evidence.prior_exp_transcripts) is not tuple
        or type(evidence.prior_log_transcripts) is not tuple
    ):
        raise TypeError(
            "residual-observable helper transcript lanes must be exact tuples"
        )
    _validate_residual_observable_entire_transcripts(
        evidence.prior_exp_transcripts,
        evidence.prior_log_transcripts,
    )


def _make_local_residual_observable_helper_failure_candidate(  # noqa: PLR0913
    *,
    call: GalerkinLocalResidualObservableHelperCall,
    channel_index: int | None,
    entire_failure: EntireEnclosureFailure | None,
    poisson_failure: CensoredPoissonEnclosureFailure | None,
    nested_kernel: str | None,
    nested_failure: EntireEnclosureFailure | None,
    prior_exp_transcripts: tuple[EntireWorkTranscript, ...],
    prior_log_transcripts: tuple[EntireWorkTranscript, ...],
    local_exact_work_count_exact: str,
    nested_exact_work_count_exact: str | None,
    nested_attempted_exact_work_count_exact: str | None,
    planned_exact_work_count_exact: str,
    attempted_exact_work_count_exact: str,
    failure_digest: str,
) -> GalerkinLocalResidualObservableHelperFailureEvidence:
    """PRIVATE: Construct one unsealed helper-failure candidate.

    Parameters
    ----------
    call : GalerkinLocalResidualObservableHelperCall
        Required canonical input.
    channel_index : int | None
        Optional channel binding for the saturated-probability lane.
    entire_failure : EntireEnclosureFailure | None
        Direct exponential failure, when applicable.
    poisson_failure : CensoredPoissonEnclosureFailure | None
        Direct Poisson failure, when applicable.
    nested_kernel : str | None
        Nested entire-kernel name supplied by a Poisson failure.
    nested_failure : EntireEnclosureFailure | None
        Nested entire failure supplied by a Poisson failure.
    prior_exp_transcripts : tuple[EntireWorkTranscript, ...]
        Completed exponential prefixes.
    prior_log_transcripts : tuple[EntireWorkTranscript, ...]
        Completed logarithm prefixes.
    local_exact_work_count_exact : str
        Completed local work as a canonical decimal.
    nested_exact_work_count_exact : str | None
        Completed nested work, when a nested helper failed.
    nested_attempted_exact_work_count_exact : str | None
        Attempted nested work, when a nested helper failed.
    planned_exact_work_count_exact : str
        Planned helper work as a canonical decimal.
    attempted_exact_work_count_exact : str
        Attempted helper work as a canonical decimal.
    failure_digest : str
        Placeholder digest replaced by the owner sealer.

    Returns
    -------
    result : GalerkinLocalResidualObservableHelperFailureEvidence
        Unsealed candidate.
    """
    return GalerkinLocalResidualObservableHelperFailureEvidence(
        call=call,
        channel_index=channel_index,
        entire_failure=entire_failure,
        poisson_failure=poisson_failure,
        nested_kernel=nested_kernel,
        nested_failure=nested_failure,
        prior_exp_transcripts=prior_exp_transcripts,
        prior_log_transcripts=prior_log_transcripts,
        local_exact_work_count_exact=local_exact_work_count_exact,
        nested_exact_work_count_exact=nested_exact_work_count_exact,
        nested_attempted_exact_work_count_exact=(
            nested_attempted_exact_work_count_exact
        ),
        planned_exact_work_count_exact=planned_exact_work_count_exact,
        attempted_exact_work_count_exact=attempted_exact_work_count_exact,
        failure_digest=failure_digest,
    )


def _validate_local_residual_observable_helper_failure(  # noqa: PLR0912
    evidence: GalerkinLocalResidualObservableHelperFailureEvidence,
) -> GalerkinLocalResidualObservableHelperFailureEvidence:
    """PRIVATE: Validate one replayable nested-helper failure.

    Parameters
    ----------
    evidence : GalerkinLocalResidualObservableHelperFailureEvidence
        Candidate failure evidence.

    Returns
    -------
    result : GalerkinLocalResidualObservableHelperFailureEvidence
        Validated failure evidence.

    Raises
    ------
    TypeError
        If an enum, channel, or transcript has the wrong exact type.
    ValueError
        If lane binding, counts, nested evidence, or digest is invalid.
    """
    _prevalidate_local_residual_observable_helper_failure(evidence)
    exponential_lane = evidence.call in (
        GalerkinLocalResidualObservableHelperCall.FULL_TV_EXPONENTIAL,
        GalerkinLocalResidualObservableHelperCall.FITTED_TV_EXPONENTIAL,
    )
    if exponential_lane:
        if (
            evidence.channel_index is not None
            or type(evidence.entire_failure) is not EntireEnclosureFailure
            or evidence.poisson_failure is not None
            or evidence.nested_kernel is not None
            or evidence.nested_failure is not None
            or evidence.prior_exp_transcripts
            or evidence.prior_log_transcripts
            or evidence.nested_exact_work_count_exact is not None
            or evidence.nested_attempted_exact_work_count_exact is not None
        ):
            raise ValueError(
                "residual-observable exponential failure lane disagrees"
            )
    else:
        if (
            evidence.channel_index is None
            or evidence.entire_failure is not None
            or type(evidence.poisson_failure)
            is not CensoredPoissonEnclosureFailure
        ):
            raise ValueError(
                "residual-observable Poisson failure lane disagrees"
            )
        nested_present = evidence.nested_failure is not None
        nested_shapes = (
            evidence.nested_kernel is not None,
            evidence.nested_exact_work_count_exact is not None,
            evidence.nested_attempted_exact_work_count_exact is not None,
        )
        if any(present != nested_present for present in nested_shapes):
            raise ValueError(
                "residual-observable nested failure is incomplete"
            )
        nested_outer = {
            CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE: (
                "exp"
            ),
            CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE: (
                "log"
            ),
        }
        required_kernel = nested_outer.get(evidence.poisson_failure)
        nested_required = required_kernel is not None
        if nested_present != nested_required:
            raise ValueError(
                "residual-observable nested failure trigger disagrees"
            )
        if nested_present and (
            type(evidence.nested_kernel) is not str
            or evidence.nested_kernel != required_kernel
            or type(evidence.nested_failure) is not EntireEnclosureFailure
        ):
            raise ValueError(
                "residual-observable nested helper kernel is not canonical"
            )
        if nested_present and (
            (required_kernel == "exp" and evidence.prior_log_transcripts)
            or (required_kernel == "log" and evidence.prior_exp_transcripts)
        ):
            raise ValueError(
                "residual-observable Poisson nested transcript lane disagrees"
            )
        if not nested_present and evidence.prior_log_transcripts:
            raise ValueError(
                "residual-observable Poisson helper has no logarithm lane"
            )
    local = _canonical_decimal(
        evidence.local_exact_work_count_exact,
        "local_exact_work_count_exact",
    )
    planned = _canonical_decimal(
        evidence.planned_exact_work_count_exact,
        "planned_exact_work_count_exact",
    )
    attempted = _canonical_decimal(
        evidence.attempted_exact_work_count_exact,
        "attempted_exact_work_count_exact",
    )
    work_budget_failure = (
        evidence.entire_failure is EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
        if exponential_lane
        else evidence.poisson_failure
        is CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    expected_outer_counts = (
        planned == attempted and attempted > local
        if work_budget_failure
        else planned == attempted == local
    )
    if not expected_outer_counts:
        raise ValueError("residual-observable helper work counts disagree")
    if evidence.nested_exact_work_count_exact is not None:
        nested = _canonical_decimal(
            evidence.nested_exact_work_count_exact,
            "nested_exact_work_count_exact",
        )
        nested_attempted = _canonical_decimal(
            evidence.nested_attempted_exact_work_count_exact,
            "nested_attempted_exact_work_count_exact",
        )
        nested_budget_failure = evidence.nested_failure is (
            EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
        )
        expected_nested_counts = (
            nested_attempted > nested
            if nested_budget_failure
            else nested_attempted == nested
        )
        if not expected_nested_counts:
            raise ValueError("residual-observable nested work counts disagree")
    expected = _helper_failure_digest(evidence)
    if not _valid_digest(evidence.failure_digest) or (
        evidence.failure_digest != expected
    ):
        raise ValueError("residual-observable helper failure digest disagrees")
    return evidence


def _make_local_residual_observable_helper_failure(
    evidence: GalerkinLocalResidualObservableHelperFailureEvidence,
) -> GalerkinLocalResidualObservableHelperFailureEvidence:
    """PRIVATE: Seal and validate one owner-constructed helper failure.

    Parameters
    ----------
    evidence : GalerkinLocalResidualObservableHelperFailureEvidence
        Unsealed owner candidate.

    Returns
    -------
    result : GalerkinLocalResidualObservableHelperFailureEvidence
        Canonical sealed failure evidence.
    """
    _prevalidate_local_residual_observable_helper_failure(evidence)
    sealed = replace(evidence, failure_digest=_helper_failure_digest(evidence))
    return _validate_local_residual_observable_helper_failure(sealed)


def _make_local_residual_observable_work_transcript_candidate(  # noqa: PLR0913
    **values: Any,
) -> GalerkinLocalResidualObservableWorkTranscript:
    """PRIVATE: Construct one exact-field work-transcript candidate.

    Parameters
    ----------
    **values : Any
        Exact declared work-transcript fields.

    Returns
    -------
    result : GalerkinLocalResidualObservableWorkTranscript
        Unvalidated candidate.
    """
    result: GalerkinLocalResidualObservableWorkTranscript = (
        GalerkinLocalResidualObservableWorkTranscript(**values)
    )
    return result  # noqa: RET504


def _validate_local_residual_observable_work_causality(  # noqa: PLR0912
    transcript: GalerkinLocalResidualObservableWorkTranscript,
    *,
    mean: int,
    law: int,
    direct: int,
    score: int,
    attempted: int,
) -> None:
    """PRIVATE: Couple typed failures to their retained stage boundary.

    Parameters
    ----------
    transcript : GalerkinLocalResidualObservableWorkTranscript
        Structurally validated staged work.
    mean : int
        Canonical mean-stage plan.
    law : int
        Canonical law-stage plan.
    direct : int
        Canonical direct-NLL-stage plan.
    score : int
        Canonical score-NLL-stage plan.
    attempted : int
        Canonical attempted local work.

    Raises
    ------
    ValueError
        If the failure bits cannot cause the stored retained prefix.
    """
    failure = transcript.failure
    counters = (
        transcript.mean_exact_work_count,
        transcript.law_exact_work_count,
        transcript.direct_nll_exact_work_count,
        transcript.score_nll_exact_work_count,
    )
    layer = transcript.completed_layer
    failure_type = GalerkinLocalResidualObservableFailure
    exp_bit = failure_type.EXPONENTIAL_ENCLOSURE_FAILURE
    exp_failed = bool(failure & exp_bit)
    primary = failure & ~exp_bit
    parent_mean_bits = (
        failure_type.PARENT_MEAN_NONCERTIFICATE
        | failure_type.ADMITTED_MEAN_HULL_UNAVAILABLE
        | failure_type.PRODUCTION_MEAN_NONSINGLETON
        | failure_type.PRODUCTION_MEAN_OUTSIDE_HULL
    )
    score_floor_bits = (
        failure_type.SCORE_MEAN_FLOOR_UNAVAILABLE
        | failure_type.SCORE_PROBABILITY_FLOOR_UNAVAILABLE
    )
    resource_bits = (
        failure_type.RATIONAL_SIZE_LIMIT
        | failure_type.ARITHMETIC_RANGE_FAILURE
    )
    preflight_bits = (
        failure_type.EXACT_WORK_BUDGET_EXCEEDED
        | failure_type.EXACT_WORK_COUNT_OVERFLOW
    )

    if transcript.preflight_failed:
        return
    if failure & preflight_bits:
        raise ValueError(
            "residual-observable resource preflight failure is misplaced"
        )
    if exp_failed and (
        transcript.law_exact_work_count != law
        or layer
        not in (
            GalerkinLocalResidualObservableLayer.LAW,
            GalerkinLocalResidualObservableLayer.POINTWISE_NLL,
        )
    ):
        raise ValueError(
            "residual-observable exponential failure precedes the law layer"
        )

    causal = False
    if primary == failure_type.NONE and exp_failed:
        causal = (
            layer is GalerkinLocalResidualObservableLayer.LAW
            and counters == (mean, law, law, law)
            and attempted == law
        ) or (
            layer is GalerkinLocalResidualObservableLayer.POINTWISE_NLL
            and counters == (mean, law, direct, score)
            and attempted == score
        )
    elif primary == failure_type.PARENT_STATE_NONCERTIFICATE:
        causal = (
            not exp_failed
            and counters == (0, 0, 0, 0)
            and attempted == 0
            and layer is GalerkinLocalResidualObservableLayer.UNAVAILABLE
        )
    elif (
        primary != failure_type.NONE
        and not (primary & ~parent_mean_bits)
        and int(primary).bit_count() == 1
    ):
        causal = (
            not exp_failed
            and counters == (0, 0, 0, 0)
            and attempted == 0
            and layer is GalerkinLocalResidualObservableLayer.STATE
        )
    elif primary == failure_type.PARENT_LAW_NONCERTIFICATE:
        causal = (
            not exp_failed
            and counters == (mean, mean, mean, mean)
            and attempted == mean
            and layer is GalerkinLocalResidualObservableLayer.MEAN
        )
    elif primary == failure_type.DIRECT_NLL_UNAVAILABLE:
        causal = (
            counters == (mean, law, law, law)
            and attempted == law
            and layer is GalerkinLocalResidualObservableLayer.LAW
        )
    elif primary in (
        failure_type.POISSON_ENCLOSURE_FAILURE,
        failure_type.POISSON_ENCLOSURE_FAILURE
        | failure_type.NESTED_HELPER_FAILURE,
    ):
        causal = (
            counters == (mean, law, direct, direct)
            and attempted == score
            and layer is GalerkinLocalResidualObservableLayer.POINTWISE_NLL
        )
    elif primary & score_floor_bits:
        remaining = primary & ~score_floor_bits
        causal = (
            counters == (mean, law, direct, direct)
            and layer is GalerkinLocalResidualObservableLayer.POINTWISE_NLL
            and remaining
            in (
                failure_type.NONE,
                failure_type.RATIONAL_SIZE_LIMIT,
            )
            and (
                attempted == transcript.exact_work_count
                if remaining == failure_type.RATIONAL_SIZE_LIMIT
                else attempted == score
            )
        )
    elif primary in (
        failure_type.RATIONAL_SIZE_LIMIT,
        failure_type.ARITHMETIC_RANGE_FAILURE,
    ):
        causal = attempted == transcript.exact_work_count
    elif primary == failure_type.NONE and failure == failure_type.NONE:
        causal = True
    elif primary & resource_bits:
        causal = False

    if not causal:
        raise ValueError(
            "residual-observable failure does not cause its retained stage"
        )


def _validate_local_residual_observable_work_transcript(
    transcript: GalerkinLocalResidualObservableWorkTranscript,
) -> GalerkinLocalResidualObservableWorkTranscript:
    """PRIVATE: Validate one staged RM-S5 work transcript.

    Parameters
    ----------
    transcript : GalerkinLocalResidualObservableWorkTranscript
        Candidate transcript.

    Returns
    -------
    result : GalerkinLocalResidualObservableWorkTranscript
        Validated work evidence.

    Raises
    ------
    TypeError
        If any policy, counter, status, or enum has the wrong exact type.
    ValueError
        If policies, staged counts, completion, or failure disagree.
    """
    if type(transcript) is not GalerkinLocalResidualObservableWorkTranscript:
        raise TypeError("residual-observable work has the wrong type")
    integer_fields = (
        transcript.maximum_work,
        transcript.maximum_rational_bits,
        transcript.channel_count,
        transcript.fitted_channel_count,
        transcript.zero_observation_positive_ceiling_count,
        transcript.interior_observation_count,
        transcript.saturated_positive_ceiling_count,
        transcript.mean_exact_work_count,
        transcript.law_exact_work_count,
        transcript.direct_nll_exact_work_count,
        transcript.score_nll_exact_work_count,
        transcript.exact_work_count,
        transcript.rational_peak_bits,
    )
    if type(transcript.algorithm) is not str or any(
        type(value) is not int for value in integer_fields
    ):
        raise TypeError("residual-observable work fields have invalid types")
    if (
        type(transcript.completed_successfully) is not bool
        or type(transcript.preflight_failed) is not bool
        or type(transcript.count_overflow) is not bool
        or type(transcript.completed_layer)
        is not GalerkinLocalResidualObservableLayer
        or type(transcript.failure)
        is not GalerkinLocalResidualObservableFailure
        or type(transcript.nested_parent_work_scope) is not str
    ):
        raise TypeError("residual-observable work status has invalid types")
    exact_fields = (
        transcript.nested_parent_work_count_exact,
        transcript.nested_helper_work_count_exact,
        transcript.planned_mean_exact_work_count_exact,
        transcript.planned_law_exact_work_count_exact,
        transcript.planned_direct_nll_exact_work_count_exact,
        transcript.planned_score_nll_exact_work_count_exact,
        transcript.attempted_exact_work_count_exact,
    )
    nested_parent, nested_helper, mean, law, direct, score, attempted = (
        _canonical_decimal(value, "residual-observable work count")
        for value in exact_fields
    )
    del nested_parent, nested_helper
    expected_mean = (
        3 * transcript.channel_count + transcript.fitted_channel_count
    )
    expected_law = expected_mean + 4
    expected_direct = (
        3 * transcript.channel_count + 2 * transcript.fitted_channel_count + 6
    )
    expected_score = (
        3 * transcript.channel_count
        + 4 * transcript.fitted_channel_count
        + 6
        + 2 * transcript.zero_observation_positive_ceiling_count
        + 7 * transcript.interior_observation_count
        + 4 * transcript.saturated_positive_ceiling_count
    )
    if (
        transcript.algorithm != _RESIDUAL_OBSERVABLE_WORK_ALGORITHM
        or transcript.nested_parent_work_scope != _NESTED_PARENT_WORK_SCOPE
        or transcript.maximum_work <= 0
        or transcript.maximum_work > _MAXIMUM_SIGNED_INT64
        or transcript.maximum_rational_bits <= 1
        or transcript.maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
        or transcript.rational_peak_bits > 3 * _HARD_MAXIMUM_RATIONAL_BITS + 2
        or int(transcript.failure) & ~_KNOWN_RESIDUAL_OBSERVABLE_FAILURE_MASK
        or any(value < 0 for value in integer_fields[2:])
        or transcript.fitted_channel_count > transcript.channel_count
        or (
            transcript.zero_observation_positive_ceiling_count
            + transcript.interior_observation_count
            + transcript.saturated_positive_ceiling_count
            > transcript.fitted_channel_count
        )
        or (mean, law, direct, score)
        != (expected_mean, expected_law, expected_direct, expected_score)
        or not (mean <= law <= direct <= score)
        or transcript.mean_exact_work_count > mean
        or transcript.law_exact_work_count > law
        or transcript.direct_nll_exact_work_count > direct
        or transcript.score_nll_exact_work_count > score
        or not (
            transcript.mean_exact_work_count
            <= transcript.law_exact_work_count
            <= transcript.direct_nll_exact_work_count
            <= transcript.score_nll_exact_work_count
        )
        or transcript.score_nll_exact_work_count != transcript.exact_work_count
        or transcript.exact_work_count > transcript.maximum_work
        or attempted < transcript.exact_work_count
        or attempted > score
    ):
        raise ValueError("residual-observable work evidence is inconsistent")
    expected_overflow = score > _MAXIMUM_SIGNED_INT64
    if transcript.count_overflow != expected_overflow:
        raise ValueError("residual-observable overflow flag disagrees")
    if transcript.preflight_failed:
        expected_failure = (
            GalerkinLocalResidualObservableFailure.EXACT_WORK_COUNT_OVERFLOW
            if expected_overflow
            else GalerkinLocalResidualObservableFailure.EXACT_WORK_BUDGET_EXCEEDED  # noqa: E501
        )
        if (
            transcript.completed_successfully
            or transcript.exact_work_count != 0
            or attempted != score
            or transcript.rational_peak_bits != 0
            or transcript.completed_layer
            is not GalerkinLocalResidualObservableLayer.STATE
            or transcript.failure is not expected_failure
            or (not expected_overflow and score <= transcript.maximum_work)
        ):
            raise ValueError("residual-observable preflight stop disagrees")
    else:
        counters = (
            transcript.mean_exact_work_count,
            transcript.law_exact_work_count,
            transcript.direct_nll_exact_work_count,
            transcript.score_nll_exact_work_count,
        )
        incomplete_mean = counters[0] < mean
        incomplete_law = counters[1] < law
        incomplete_direct = counters[2] < direct
        rational_failed = bool(
            transcript.failure
            & GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
        )
        phase_shape_valid = (
            (
                incomplete_mean
                and counters[1:] == (counters[0],) * 3
                and transcript.completed_layer
                in (
                    GalerkinLocalResidualObservableLayer.UNAVAILABLE,
                    GalerkinLocalResidualObservableLayer.STATE,
                )
            )
            or (
                not incomplete_mean
                and incomplete_law
                and counters[0] == mean
                and counters[2:] == (counters[1],) * 2
                and transcript.completed_layer
                in (
                    GalerkinLocalResidualObservableLayer.STATE,
                    GalerkinLocalResidualObservableLayer.MEAN,
                )
                and (
                    transcript.completed_layer
                    is GalerkinLocalResidualObservableLayer.MEAN
                    or (rational_failed and counters[1] == mean)
                )
            )
            or (
                not incomplete_law
                and incomplete_direct
                and counters[:2] == (mean, law)
                and counters[3] == counters[2]
                and transcript.completed_layer
                in (
                    GalerkinLocalResidualObservableLayer.MEAN,
                    GalerkinLocalResidualObservableLayer.LAW,
                )
                and (
                    transcript.completed_layer
                    is GalerkinLocalResidualObservableLayer.LAW
                    or (rational_failed and counters[2] == law)
                )
            )
            or (
                not incomplete_direct
                and counters[:3] == (mean, law, direct)
                and transcript.completed_layer
                in (
                    GalerkinLocalResidualObservableLayer.LAW,
                    GalerkinLocalResidualObservableLayer.POINTWISE_NLL,
                )
            )
        )
        if (
            not phase_shape_valid
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
                "residual-observable stopped stage or rational peak disagrees"
            )
    _validate_local_residual_observable_work_causality(
        transcript,
        mean=mean,
        law=law,
        direct=direct,
        score=score,
        attempted=attempted,
    )
    if transcript.completed_successfully:
        if (
            transcript.completed_layer
            is not GalerkinLocalResidualObservableLayer.POINTWISE_NLL
            or transcript.failure
            is not GalerkinLocalResidualObservableFailure.NONE
            or transcript.exact_work_count != score
            or transcript.rational_peak_bits > transcript.maximum_rational_bits
        ):
            raise ValueError("residual-observable success work disagrees")
    elif transcript.failure == GalerkinLocalResidualObservableFailure.NONE:
        raise ValueError("residual-observable stopped work lacks a failure")
    return transcript


def _make_local_residual_observable_work_transcript(
    transcript: GalerkinLocalResidualObservableWorkTranscript,
) -> GalerkinLocalResidualObservableWorkTranscript:
    """PRIVATE: Validate one owner-constructed work transcript.

    Parameters
    ----------
    transcript : GalerkinLocalResidualObservableWorkTranscript
        Owner-constructed candidate.

    Returns
    -------
    result : GalerkinLocalResidualObservableWorkTranscript
        Canonical validated work evidence.
    """
    return _validate_local_residual_observable_work_transcript(transcript)


def _checked_intervals(
    values: object,
    *,
    size: int,
    name: str,
    optional_items: bool = False,
) -> tuple[GalerkinLocalDetectorRationalInterval | None, ...]:
    """PRIVATE: Validate one exact-size tuple of detector intervals.

    Parameters
    ----------
    values : object
        Required tuple input.
    size : int
        Required tuple length.
    name : str
        Field name used in diagnostics.
    optional_items : bool
        Whether individual ``None`` sentinels are allowed.

    Returns
    -------
    result : tuple[GalerkinLocalDetectorRationalInterval | None, ...]
        Validated interval tuple.

    Raises
    ------
    TypeError
        If the outer container or an interval has the wrong exact type.
    ValueError
        If its length or endpoint domain is invalid.
    """
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(values) != size:
        raise ValueError(f"{name} has the wrong length")
    checked: list[GalerkinLocalDetectorRationalInterval | None] = []
    for value in values:
        if value is None:
            if not optional_items:
                raise ValueError(f"{name} does not allow missing intervals")
            checked.append(None)
        else:
            if type(value) is not GalerkinLocalDetectorRationalInterval:
                raise TypeError(f"{name} interval has the wrong exact type")
            interval = _validate_local_detector_rational_interval(value)
            if interval.lower < 0:
                raise ValueError(f"{name} must be nonnegative")
            checked.append(interval)
    return tuple(checked)


def _checked_optional_interval(
    value: object, name: str
) -> GalerkinLocalDetectorRationalInterval | None:
    """PRIVATE: Validate one optional nonnegative exact interval.

    Parameters
    ----------
    value : object
        Required optional input.
    name : str
        Field name used in diagnostics.

    Returns
    -------
    result : GalerkinLocalDetectorRationalInterval | None
        Validated interval or sentinel.

    Raises
    ------
    ValueError
        If an interval has a negative lower endpoint.
    """
    if value is None:
        return None
    if type(value) is not GalerkinLocalDetectorRationalInterval:
        raise TypeError(f"{name} interval has the wrong exact type")
    checked = _validate_local_detector_rational_interval(value)
    if checked.lower < 0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _same_interval(
    left: GalerkinLocalDetectorRationalInterval,
    right: GalerkinLocalDetectorRationalInterval,
) -> bool:
    """PRIVATE: Return whether two exact interval storages are identical.

    Parameters
    ----------
    left : GalerkinLocalDetectorRationalInterval
        First canonical interval.
    right : GalerkinLocalDetectorRationalInterval
        Second canonical interval.

    Returns
    -------
    result : bool
        Whether every stored endpoint field agrees.
    """
    return (
        left.lower_numerator == right.lower_numerator
        and left.lower_denominator == right.lower_denominator
        and left.upper_numerator == right.upper_numerator
        and left.upper_denominator == right.upper_denominator
    )


def _interval_peak_bits(
    interval: GalerkinLocalDetectorRationalInterval,
) -> int:
    """PRIVATE: Return the largest stored endpoint-component bit length.

    Parameters
    ----------
    interval : GalerkinLocalDetectorRationalInterval
        Structurally validated exact interval.

    Returns
    -------
    result : int
        Largest numerator or denominator bit length.
    """
    return max(
        1,
        abs(interval.lower_numerator).bit_length(),
        interval.lower_denominator.bit_length(),
        abs(interval.upper_numerator).bit_length(),
        interval.upper_denominator.bit_length(),
    )


def _intervals_peak_bits(values: object) -> int:
    """PRIVATE: Return the largest bit length in nested optional intervals.

    Parameters
    ----------
    values : object
        One interval, a nested exact tuple, or ``None``.

    Returns
    -------
    result : int
        Largest retained endpoint-component bit length.
    """
    if values is None:
        return 0
    if type(values) is GalerkinLocalDetectorRationalInterval:
        return _interval_peak_bits(values)
    if type(values) is tuple:
        return max(
            (_intervals_peak_bits(value) for value in values), default=0
        )
    raise TypeError("residual-observable interval peak input is invalid")


def _point_interval(value: Fraction) -> GalerkinLocalDetectorRationalInterval:
    """PRIVATE: Store one exact rational singleton through the L9 owner.

    Parameters
    ----------
    value : Fraction
        Exact singleton value.

    Returns
    -------
    result : GalerkinLocalDetectorRationalInterval
        Canonical singleton interval.
    """
    return _make_local_detector_rational_interval(value, value)


def _interval_payload_equal(left: object, right: object) -> bool:
    """PRIVATE: Compare arbitrary stored carriers without properties.

    Parameters
    ----------
    left : object
        First stored value.
    right : object
        Second stored value.

    Returns
    -------
    result : bool
        Whether their canonical stored-value payloads agree.
    """
    return stored_value_payload(left) == stored_value_payload(right)


def _state_authority_available(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
) -> bool:
    """PRIVATE: Recompute authenticated L6 state-radius authority per mode.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.

    Returns
    -------
    result : bool
        Whether every detector mode owns eligible aligned state authority.
    """
    detector = likelihood.detector
    mode_count = len(detector.pixel_forms)
    aligned = (
        detector.mode_state_radius_intervals,
        detector.mode_state_radius_provenance_digests,
        detector.mode_port_certificate_digests,
        detector.mode_pixel_evidence_digests,
    )
    if mode_count == 0 or any(len(values) != mode_count for values in aligned):
        return False
    for index, pixel in enumerate(detector.pixel_forms):
        terminal = pixel.positive_port.terminal_certificate
        projection = terminal.projection_certificate
        proof = projection.stability_result.proof
        stored_radius = detector.mode_state_radius_intervals[index]
        if (
            not bool(np.asarray(projection.finite_projection_bound_eligible))
            or not bool(np.asarray(proof.state_radius_eligible))
            or stored_radius is None
            or detector.mode_state_radius_provenance_digests[index]
            != terminal.parent_projection_certificate_digest
            or detector.mode_port_certificate_digests[index]
            != pixel.positive_port.certificate_digest
            or detector.mode_pixel_evidence_digests[index]
            != pixel.pixel_model_evidence_digest
        ):
            return False
        radius_value = float(np.asarray(projection.state_radius_upper_bound))
        if not np.isfinite(radius_value) or radius_value < 0.0:
            return False
        expected_radius = _point_interval(Fraction.from_float(radius_value))
        if not _same_interval(stored_radius, expected_radius):
            return False
    return True


def _derived_detector_mean_hulls(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
) -> tuple[GalerkinLocalDetectorRationalInterval, ...] | None:
    """PRIVATE: Derive admitted mean hulls solely from the nested detector.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.

    Returns
    -------
    result : tuple[GalerkinLocalDetectorRationalInterval, ...] | None
        Detector-derived exact hulls, or ``None`` when unavailable.
    """
    detector = likelihood.detector
    channel_count = np.asarray(detector.response_matrix).shape[0]
    production = detector.production_pre_gain_mean_point_intervals
    exact = detector.exact_state_pre_gain_mean_intervals
    if len(production) != channel_count or len(exact) != channel_count:
        return None
    result: list[GalerkinLocalDetectorRationalInterval] = []
    for point, state in zip(production, exact, strict=True):
        checked_point = _validate_local_detector_rational_interval(point)
        checked_state = _validate_local_detector_rational_interval(state)
        if checked_point.lower != checked_point.upper:
            return None
        lower = min(checked_point.lower, checked_state.lower)
        upper = max(checked_point.upper, checked_state.upper)
        result.append(_make_local_detector_rational_interval(lower, upper))
    return tuple(result)


def _derived_detector_mean_evidence(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
) -> (
    tuple[
        _Intervals,
        _Intervals,
        GalerkinLocalDetectorRationalInterval,
        GalerkinLocalDetectorRationalInterval,
    ]
    | None
):
    """PRIVATE: Recompute the complete detector-derived mean evidence.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.

    Returns
    -------
    result : tuple or None
        Hulls, channel errors, and full/fitted L1 singleton bounds, or
        ``None`` when the detector leaves are unavailable or nonsingleton.
    """
    detector = likelihood.detector
    channel_count = np.asarray(detector.response_matrix).shape[0]
    hulls = _derived_detector_mean_hulls(likelihood)
    if hulls is None:
        return None
    production = detector.production_pre_gain_mean_point_intervals
    if len(production) != channel_count:
        return None
    errors: list[GalerkinLocalDetectorRationalInterval] = []
    full = Fraction()
    fitted = Fraction()
    fit_mask = tuple(bool(value) for value in np.asarray(detector.fit_mask))
    for is_fitted, point, hull in zip(
        fit_mask, production, hulls, strict=True
    ):
        checked_point = _validate_local_detector_rational_interval(point)
        if checked_point.lower != checked_point.upper:
            return None
        error = max(
            abs(checked_point.lower - hull.lower),
            abs(checked_point.lower - hull.upper),
        )
        errors.append(_point_interval(error))
        full += error
        if is_fitted:
            fitted += error
    return (
        hulls,
        tuple(errors),
        _point_interval(full),
        _point_interval(fitted),
    )


def _mean_leaf_structure_available(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
) -> tuple[bool, bool]:
    """PRIVATE: Check mean-leaf shape and singleton storage without Fractions.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.

    Returns
    -------
    aligned : bool
        Whether production and exact-state lanes are channel aligned.
    singleton : bool
        Whether every production interval is an exact stored singleton.
    """
    detector = likelihood.detector
    channel_count = np.asarray(detector.response_matrix).shape[0]
    production = detector.production_pre_gain_mean_point_intervals
    exact = detector.exact_state_pre_gain_mean_intervals
    aligned = (
        type(production) is tuple
        and type(exact) is tuple
        and len(production) == channel_count
        and len(exact) == channel_count
        and all(
            type(value) is GalerkinLocalDetectorRationalInterval
            for value in production + exact
        )
    )
    singleton = aligned and all(
        value.lower_numerator == value.upper_numerator
        and value.lower_denominator == value.upper_denominator
        for value in production
    )
    return aligned, singleton


def _derived_detector_mean_evidence_checked(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
    ledger: _ResidualObservableLedger,
) -> tuple[
    _Intervals,
    _Intervals,
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalDetectorRationalInterval,
]:
    """PRIVATE: Derive mean evidence through the exact local work ledger.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.
    ledger : _ResidualObservableLedger
        RM-S5 rational-size and exact-work authority.

    Returns
    -------
    hulls : _Intervals
        Detector-derived admitted pre-gain hulls.
    errors : _Intervals
        Exact singleton endpoint-distance bounds.
    full : GalerkinLocalDetectorRationalInterval
        Full-channel L1 singleton bound.
    fitted : GalerkinLocalDetectorRationalInterval
        Fixed-fit L1 singleton bound.
    """
    detector = likelihood.detector
    production = detector.production_pre_gain_mean_point_intervals
    exact = detector.exact_state_pre_gain_mean_intervals
    ledger.scan_intervals((production, exact))
    fit_mask = tuple(bool(value) for value in np.asarray(detector.fit_mask))
    hulls: list[GalerkinLocalDetectorRationalInterval] = []
    errors: list[GalerkinLocalDetectorRationalInterval] = []
    full = Fraction()
    fitted = Fraction()
    for is_fitted, point, state in zip(
        fit_mask, production, exact, strict=True
    ):
        point_value = point.lower
        lower = min(point_value, state.lower)
        upper = max(point_value, state.upper)
        lower_distance = abs(ledger.subtract(point_value, lower))
        upper_distance = abs(ledger.subtract(upper, point_value))
        distance = max(lower_distance, upper_distance)
        full = ledger.add(full, distance)
        if is_fitted:
            fitted = ledger.add(fitted, distance)
        hulls.append(_make_local_detector_rational_interval(lower, upper))
        errors.append(_point_interval(distance))
    return (
        tuple(hulls),
        tuple(errors),
        _point_interval(full),
        _point_interval(fitted),
    )


def _entire_helper_failure_from_error(
    call: GalerkinLocalResidualObservableHelperCall,
    error: EntireEnclosureError,
) -> GalerkinLocalResidualObservableHelperFailureEvidence:
    """PRIVATE: Bind one direct exponential error to its exact call lane.

    Parameters
    ----------
    call : GalerkinLocalResidualObservableHelperCall
        Full- or fitted-TV helper lane.
    error : EntireEnclosureError
        Typed bounded helper failure.

    Returns
    -------
    result : GalerkinLocalResidualObservableHelperFailureEvidence
        Canonical sealed failure evidence.
    """
    return _make_local_residual_observable_helper_failure(
        _make_local_residual_observable_helper_failure_candidate(
            call=call,
            channel_index=None,
            entire_failure=error.failure,
            poisson_failure=None,
            nested_kernel=None,
            nested_failure=None,
            prior_exp_transcripts=(),
            prior_log_transcripts=(),
            local_exact_work_count_exact=str(error.exact_work_count),
            nested_exact_work_count_exact=None,
            nested_attempted_exact_work_count_exact=None,
            planned_exact_work_count_exact=str(
                error.attempted_exact_work_count
            ),
            attempted_exact_work_count_exact=str(
                error.attempted_exact_work_count
            ),
            failure_digest="0" * _SHA256_HEX_LENGTH,
        )
    )


def _poisson_helper_failure_from_error(
    channel: int,
    error: CensoredPoissonEnclosureError,
) -> GalerkinLocalResidualObservableHelperFailureEvidence:
    """PRIVATE: Bind one predecessor-probability error to its channel.

    Parameters
    ----------
    channel : int
        Saturated fitted channel index.
    error : CensoredPoissonEnclosureError
        Typed bounded helper failure.

    Returns
    -------
    result : GalerkinLocalResidualObservableHelperFailureEvidence
        Canonical sealed failure evidence.
    """
    return _make_local_residual_observable_helper_failure(
        _make_local_residual_observable_helper_failure_candidate(
            call=GalerkinLocalResidualObservableHelperCall.SATURATED_SCORE_MASS,
            channel_index=channel,
            entire_failure=None,
            poisson_failure=error.failure,
            nested_kernel=error.nested_kernel,
            nested_failure=error.nested_failure,
            prior_exp_transcripts=error.prior_exp_transcripts,
            prior_log_transcripts=error.prior_log_transcripts,
            local_exact_work_count_exact=str(error.exact_work_count),
            nested_exact_work_count_exact=(
                None
                if error.nested_exact_work_count is None
                else str(error.nested_exact_work_count)
            ),
            nested_attempted_exact_work_count_exact=(
                None
                if error.nested_attempted_exact_work_count is None
                else str(error.nested_attempted_exact_work_count)
            ),
            planned_exact_work_count_exact=str(
                error.attempted_exact_work_count
            ),
            attempted_exact_work_count_exact=str(
                error.attempted_exact_work_count
            ),
            failure_digest="0" * _SHA256_HEX_LENGTH,
        )
    )


def _tv_bound_evidence(
    eta: Fraction,
    *,
    call: GalerkinLocalResidualObservableHelperCall,
    manifest: GalerkinLocalResidualObservableInputManifest,
    ledger: _ResidualObservableLedger | None = None,
) -> tuple[
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalDetectorRationalInterval | None,
    GalerkinLocalDetectorRationalInterval,
    EntireWorkTranscript | None,
    GalerkinLocalResidualObservableHelperFailureEvidence | None,
]:
    """PRIVATE: Recompute one linear and optional exponential TV bound.

    Parameters
    ----------
    eta : Fraction
        Exact L1 mean-error sum.
    call : GalerkinLocalResidualObservableHelperCall
        Full or fitted exponential helper lane.
    manifest : GalerkinLocalResidualObservableInputManifest
        Bound RM-S5 helper policies.
    ledger : _ResidualObservableLedger | None, optional
        Exact local transaction ledger.  ``None`` keeps this pure helper
        convenient for parent-free arithmetic tests.

    Returns
    -------
    linear : GalerkinLocalDetectorRationalInterval
        Canonical linear bound.
    exponential : GalerkinLocalDetectorRationalInterval | None
        Exponential tightening when available.
    selected : GalerkinLocalDetectorRationalInterval
        Tighter retained bound; linear always survives helper failure.
    transcript : EntireWorkTranscript | None
        Successful exact helper transcript.
    failure : GalerkinLocalResidualObservableHelperFailureEvidence | None
        Typed helper failure evidence.
    """
    linear = _point_interval(min(Fraction(1), eta))
    negative_eta = -eta
    if ledger is not None:
        ledger.commit(negative_eta)
    try:
        exponential_raw, transcript = enclose_real_exp(
            (negative_eta, negative_eta),
            precision_bits=manifest.tv_exp_precision_bits,
            maximum_terms=manifest.maximum_tv_exp_terms,
            maximum_work=manifest.maximum_tv_exp_work,
            maximum_range_reductions=(
                manifest.maximum_tv_exp_range_reductions
            ),
            maximum_rational_bits=(
                manifest.maximum_residual_observable_rational_bits
            ),
        )
    except EntireEnclosureError as error:
        if ledger is not None:
            ledger.commit(linear.upper, negative_eta)
        return (
            linear,
            None,
            linear,
            None,
            _entire_helper_failure_from_error(call, error),
        )
    lower = max(Fraction(), Fraction(1) - exponential_raw[1])
    upper = min(Fraction(1), Fraction(1) - exponential_raw[0])
    if ledger is not None:
        ledger.commit(lower, upper)
    exponential = _make_local_detector_rational_interval(lower, upper)
    selected = exponential if exponential.upper < linear.upper else linear
    return linear, exponential, selected, transcript, None


def _production_nll_trace(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
    channel: int,
    production_point: GalerkinLocalDetectorRationalInterval,
    ledger: _ResidualObservableLedger | None = None,
) -> GalerkinLocalDetectorRealProductionTrace:
    """PRIVATE: Return the unique authenticated production-NLL trace.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.
    channel : int
        Fitted channel index.
    production_point : GalerkinLocalDetectorRationalInterval
        Authenticated production NLL singleton.
    ledger : _ResidualObservableLedger | None, optional
        S5 raw-storage scanner, when invoked inside score replay.

    Returns
    -------
    result : GalerkinLocalDetectorRealProductionTrace
        Unique canonical scalar trace.

    Raises
    ------
    ValueError
        If the unique trace, point binding, or scalar shape disagrees.
    """
    quantity = f"production_nll.channel_{channel}"
    traces = tuple(
        trace
        for trace in likelihood.production_traces
        if trace.stage is GalerkinLocalDetectorProductionStage.CENSORED_NLL
        and trace.quantity == quantity
    )
    if len(traces) != 1:
        raise ValueError("production NLL trace is not unique")
    raw_trace = traces[0]
    if ledger is not None:
        ledger.scan_intervals(
            (
                production_point,
                raw_trace.exact_point_intervals,
                raw_trace.raw_intervals,
            )
        )
    trace = _validate_local_detector_real_production_trace(raw_trace)
    if (
        trace.logical_shape != ()
        or len(trace.raw_intervals) != 1
        or len(trace.exact_point_intervals) != 1
        or not _same_interval(trace.exact_point_intervals[0], production_point)
    ):
        raise ValueError("production NLL trace point binding disagrees")
    return trace


def _trace_rounding_error(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
    channel: int,
    production_point: GalerkinLocalDetectorRationalInterval,
    ledger: _ResidualObservableLedger | None = None,
) -> Fraction:
    """PRIVATE: Recompute one raw production-NLL rounding radius.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.
    channel : int
        Fitted channel index.
    production_point : GalerkinLocalDetectorRationalInterval
        Authenticated production NLL singleton.
    ledger : _ResidualObservableLedger | None, optional
        Exact score-stage work and rational-size ledger.

    Returns
    -------
    result : Fraction
        Exact maximum distance to the unique raw-trace endpoints.
    """
    trace = _production_nll_trace(
        likelihood, channel, production_point, ledger
    )
    raw = trace.raw_intervals[0]
    if ledger is not None:
        ledger.scan_intervals(
            (
                production_point,
                trace.exact_point_intervals,
                trace.raw_intervals,
            )
        )
    point = production_point.lower
    if ledger is None:
        return max(abs(point - raw.lower), abs(point - raw.upper))
    lower_distance = abs(ledger.subtract(point, raw.lower))
    upper_distance = abs(ledger.subtract(point, raw.upper))
    return max(lower_distance, upper_distance)


def _nested_parent_work(
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
) -> int:
    """PRIVATE: Sum the complete authenticated L9 likelihood work tree.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Structurally validated parent likelihood.

    Returns
    -------
    result : int
        Complete local, parent, and helper work count.
    """
    work = likelihood.work_transcript
    return (
        work.exact_work_count
        + _canonical_decimal(
            work.nested_parent_work_count_exact,
            "parent nested-parent work",
        )
        + _canonical_decimal(
            work.nested_helper_work_count_exact,
            "parent nested-helper work",
        )
    )


def _entire_transcript_work(transcript: EntireWorkTranscript | None) -> int:
    """PRIVATE: Return completed work in one successful entire helper.

    Parameters
    ----------
    transcript : EntireWorkTranscript | None
        Optional successful transcript.

    Returns
    -------
    result : int
        Completed exact work.
    """
    return 0 if transcript is None else transcript.exact_work_count


def _poisson_transcript_work(
    transcript: CensoredPoissonWorkTranscript | None,
) -> int:
    """PRIVATE: Return local and nested work in one Poisson helper.

    Parameters
    ----------
    transcript : CensoredPoissonWorkTranscript | None
        Optional successful transcript.

    Returns
    -------
    result : int
        Completed exact work across the helper tree.
    """
    if transcript is None:
        return 0
    return transcript.exact_work_count + sum(
        item.exact_work_count
        for item in transcript.exp_transcripts + transcript.log_transcripts
    )


def _validate_saturated_probability_transcript(
    transcript: CensoredPoissonWorkTranscript,
    *,
    manifest: GalerkinLocalResidualObservableInputManifest,
    observed_count: int,
    count_ceiling: int,
) -> None:
    """PRIVATE: Validate one successful predecessor-probability transcript.

    Parameters
    ----------
    transcript : CensoredPoissonWorkTranscript
        Candidate successful helper transcript.
    manifest : GalerkinLocalResidualObservableInputManifest
        Bound L9 probability policies.
    observed_count : int
        Canonical predecessor observation ``c - 1``.
    count_ceiling : int
        Positive saturated count ceiling ``c``.

    Raises
    ------
    TypeError
        If the transcript or any resource field has the wrong exact type.
    ValueError
        If algorithms, policies, counters, or nested lanes disagree.
    """
    if type(transcript) is not CensoredPoissonWorkTranscript:
        raise TypeError(
            "saturated probability transcript has the wrong exact type"
        )
    integer_fields = (
        transcript.maximum_count_ceiling,
        transcript.maximum_work,
        transcript.maximum_rational_bits,
        transcript.exp_precision_bits,
        transcript.maximum_exp_terms,
        transcript.maximum_exp_work,
        transcript.maximum_exp_range_reductions,
        transcript.log_precision_bits,
        transcript.maximum_log_terms,
        transcript.maximum_log_work,
        transcript.maximum_log_range_reductions,
        transcript.count_ceiling,
        transcript.observed_count,
        transcript.polynomial_terms,
        transcript.endpoint_evaluations,
        transcript.critical_point_evaluations,
        transcript.direct_tail_lower_evaluations,
        transcript.exact_work_count,
    )
    if type(transcript.algorithm) is not str or any(
        type(value) is not int for value in integer_fields
    ):
        raise TypeError(
            "saturated probability transcript resources must use exact ints"
        )
    detector_manifest = manifest.detector_input_manifest
    if (
        transcript.algorithm
        != "exact_fraction_censored_poisson_probability_v1"
        or transcript.maximum_count_ceiling
        != detector_manifest.maximum_count_ceiling
        or transcript.maximum_work != detector_manifest.maximum_poisson_work
        or transcript.maximum_rational_bits
        != detector_manifest.maximum_poisson_rational_bits
        or transcript.exp_precision_bits
        != detector_manifest.exp_precision_bits
        or transcript.maximum_exp_terms != detector_manifest.maximum_exp_terms
        or transcript.maximum_exp_work != detector_manifest.maximum_exp_work
        or transcript.maximum_exp_range_reductions
        != detector_manifest.maximum_exp_range_reductions
        or (
            transcript.log_precision_bits,
            transcript.maximum_log_terms,
            transcript.maximum_log_work,
            transcript.maximum_log_range_reductions,
        )
        != (0, 0, 0, 0)
        or transcript.observed_count != observed_count
        or transcript.count_ceiling != count_ceiling
        or any(value < 0 for value in integer_fields[13:])
        or transcript.exact_work_count > transcript.maximum_work
        or type(transcript.exp_transcripts) is not tuple
        or type(transcript.log_transcripts) is not tuple
        or transcript.log_transcripts
    ):
        raise ValueError(
            "saturated probability transcript structure disagrees"
        )
    _validate_residual_observable_entire_transcripts(
        transcript.exp_transcripts, ()
    )
    for nested in transcript.exp_transcripts:
        if (
            nested.precision_bits != transcript.exp_precision_bits
            or nested.maximum_terms != transcript.maximum_exp_terms
            or nested.maximum_work != transcript.maximum_exp_work
            or nested.maximum_range_reductions
            != transcript.maximum_exp_range_reductions
            or nested.maximum_rational_bits != transcript.maximum_rational_bits
        ):
            raise ValueError(
                "saturated probability nested exponential policy disagrees"
            )


def _helper_failure_work(
    failure: GalerkinLocalResidualObservableHelperFailureEvidence | None,
) -> int:
    """PRIVATE: Return completed local, prefix, and nested failure work.

    Parameters
    ----------
    failure : GalerkinLocalResidualObservableHelperFailureEvidence | None
        Optional typed helper failure.

    Returns
    -------
    result : int
        Completed helper work across every retained lane.
    """
    if failure is None:
        return 0
    result = _canonical_decimal(
        failure.local_exact_work_count_exact,
        "helper local work",
    )
    result += sum(
        item.exact_work_count
        for item in (
            failure.prior_exp_transcripts + failure.prior_log_transcripts
        )
    )
    if failure.nested_exact_work_count_exact is not None:
        result += _canonical_decimal(
            failure.nested_exact_work_count_exact,
            "helper nested work",
        )
    return result


def _work_plan_counts(
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """PRIVATE: Derive category counts and every exact local stage plan.

    Parameters
    ----------
    manifest : GalerkinLocalResidualObservableInputManifest
        Validated observed counts and detector manifest.

    Returns
    -------
    result : tuple[int, ...]
        ``r, f, z, i, s`` followed by mean, law, direct, and score plans.
    """
    observed = tuple(
        int(value) for value in np.asarray(manifest.observed_counts)
    )
    detector_manifest = manifest.detector_input_manifest
    ceilings = tuple(
        int(value) for value in np.asarray(detector_manifest.count_ceilings)
    )
    fit_mask = tuple(
        bool(value) for value in np.asarray(detector_manifest.fit_mask)
    )
    channel_count = len(observed)
    fitted_count = sum(fit_mask)
    zero_count = sum(
        fitted and ceiling > 0 and count == 0
        for fitted, count, ceiling in zip(
            fit_mask, observed, ceilings, strict=True
        )
    )
    interior_count = sum(
        fitted and 0 < count < ceiling
        for fitted, count, ceiling in zip(
            fit_mask, observed, ceilings, strict=True
        )
    )
    saturated_count = sum(
        fitted and ceiling > 0 and count == ceiling
        for fitted, count, ceiling in zip(
            fit_mask, observed, ceilings, strict=True
        )
    )
    mean = 3 * channel_count + fitted_count
    law = mean + 4
    direct = 3 * channel_count + 2 * fitted_count + 6
    score = (
        3 * channel_count
        + 4 * fitted_count
        + 6
        + 2 * zero_count
        + 7 * interior_count
        + 4 * saturated_count
    )
    return (
        channel_count,
        fitted_count,
        zero_count,
        interior_count,
        saturated_count,
        mean,
        law,
        direct,
        score,
    )


def _empty_channel_lanes(channel_count: int) -> tuple[None, ...]:
    """PRIVATE: Return one exact channel-aligned tuple of sentinels.

    Parameters
    ----------
    channel_count : int
        Required lane length.

    Returns
    -------
    result : tuple[None, ...]
        Canonical empty lane.
    """
    return (None,) * channel_count


def _expected_local_residual_observable_evidence(  # noqa: C901, PLR0912, PLR0915
    likelihood: GalerkinLocalCensoredPoissonLikelihood,
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> dict[str, Any]:
    """PRIVATE: Independently replay the complete RM-S5 evidence DAG.

    Parameters
    ----------
    likelihood : GalerkinLocalCensoredPoissonLikelihood
        Authenticated replayed L9 likelihood.
    manifest : GalerkinLocalResidualObservableInputManifest
        Authenticated RM-S5 primitive and resource manifest.

    Returns
    -------
    result : dict[str, Any]
        Canonical values for every derived certificate field.

    Raises
    ------
    ValueError
        If authenticated parent leaves disagree with their RM-S5 bindings.
    """
    likelihood = _validate_local_censored_poisson_likelihood(likelihood)
    manifest = _validate_local_residual_observable_input_manifest(manifest)
    detector = likelihood.detector
    detector_manifest = manifest.detector_input_manifest
    plans = _work_plan_counts(manifest)
    (
        channel_count,
        fitted_count,
        zero_count,
        interior_count,
        saturated_count,
    ) = plans[:5]
    mean_plan, law_plan, direct_plan, score_plan = plans[5:]
    fit_mask = tuple(bool(value) for value in np.asarray(detector.fit_mask))
    observed = tuple(
        int(value) for value in np.asarray(manifest.observed_counts)
    )
    ceilings = tuple(
        int(value) for value in np.asarray(detector.count_ceilings)
    )
    empty = _empty_channel_lanes(channel_count)
    failure_type = GalerkinLocalResidualObservableFailure
    result: dict[str, Any] = {
        "state_evidence_available": False,
        "mean_evidence_available": False,
        "law_evidence_available": False,
        "full_law_evidence_available": False,
        "fitted_law_evidence_available": False,
        "direct_nll_evidence_available": False,
        "score_nll_evidence_available": False,
        "selected_nll_evidence_available": False,
        "failure_mask": GalerkinLocalResidualObservableFailure.NONE,
        "admitted_pre_gain_mean_hull_intervals": None,
        "channel_mean_error_bound_intervals": None,
        "full_mean_l1_error_bound_interval": None,
        "fitted_mean_l1_error_bound_interval": None,
        "full_linear_tv_bound_interval": None,
        "fitted_linear_tv_bound_interval": None,
        "full_exponential_tv_bound_interval": None,
        "fitted_exponential_tv_bound_interval": None,
        "full_selected_tv_bound_interval": None,
        "fitted_selected_tv_bound_interval": None,
        "production_fitted_total_nll_interval": None,
        "direct_nll_error_bound_interval": None,
        "score_lipschitz_factor_intervals": empty,
        "score_rounding_error_intervals": empty,
        "score_term_error_intervals": empty,
        "score_nll_error_bound_interval": None,
        "selected_nll_error_bound_interval": None,
        "saturated_predecessor_mass_upper_intervals": empty,
        "saturated_tail_probability_floor_intervals": empty,
        "full_tv_exp_transcript": None,
        "fitted_tv_exp_transcript": None,
        "full_tv_exp_failure": None,
        "fitted_tv_exp_failure": None,
        "saturated_probability_transcripts": empty,
        "saturated_probability_failures": empty,
        "strongest_layer": GalerkinLocalResidualObservableLayer.UNAVAILABLE,
        "selected_nll_route": GalerkinLocalResidualObservableRoute.UNAVAILABLE,
    }
    failure = GalerkinLocalResidualObservableFailure.NONE
    counters = [0, 0, 0, 0]
    attempted = 0
    preflight_failed = False
    count_overflow = score_plan > _MAXIMUM_SIGNED_INT64
    ledger = _ResidualObservableLedger(
        manifest.maximum_residual_observable_rational_bits
    )

    state_available = _state_authority_available(likelihood)
    result["state_evidence_available"] = state_available
    if state_available:
        result["strongest_layer"] = GalerkinLocalResidualObservableLayer.STATE
    else:
        failure |= (
            GalerkinLocalResidualObservableFailure.PARENT_STATE_NONCERTIFICATE
        )

    mean_aligned, mean_singleton = _mean_leaf_structure_available(likelihood)
    parent_mean_gate = (
        bool(np.asarray(detector.production_evidence_available))
        and bool(np.asarray(detector.detector_eligible))
        and mean_aligned
    )
    if state_available and not parent_mean_gate:
        failure |= (
            GalerkinLocalResidualObservableFailure.PARENT_MEAN_NONCERTIFICATE
        )
    elif state_available and not mean_singleton:
        failure |= (
            GalerkinLocalResidualObservableFailure.PRODUCTION_MEAN_NONSINGLETON
        )

    if state_available and parent_mean_gate and mean_singleton:
        if (
            count_overflow
            or score_plan > manifest.maximum_residual_observable_work
        ):
            failure |= (
                GalerkinLocalResidualObservableFailure.EXACT_WORK_COUNT_OVERFLOW
                if count_overflow
                else (
                    GalerkinLocalResidualObservableFailure.EXACT_WORK_BUDGET_EXCEEDED
                )
            )
            attempted = score_plan
            preflight_failed = True
        else:
            try:
                (
                    hulls,
                    errors,
                    full_l1,
                    fitted_l1,
                ) = _derived_detector_mean_evidence_checked(likelihood, ledger)
            except _ResidualObservableArithmeticError as error:
                failure |= (
                    GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
                )
                counters[:] = [error.exact_work_count] * 4
                attempted = error.exact_work_count
            else:
                if ledger.exact_work_count != mean_plan:
                    raise ValueError(
                        "residual-observable mean work formula disagrees"
                    )
                result.update(
                    {
                        "mean_evidence_available": True,
                        "admitted_pre_gain_mean_hull_intervals": hulls,
                        "channel_mean_error_bound_intervals": errors,
                        "full_mean_l1_error_bound_interval": full_l1,
                        "fitted_mean_l1_error_bound_interval": fitted_l1,
                        "strongest_layer": (
                            GalerkinLocalResidualObservableLayer.MEAN
                        ),
                    }
                )
                counters[:] = [mean_plan] * 4
                attempted = mean_plan
                if likelihood.admitted_pre_gain_mean_hull_intervals:
                    parent_hulls = _checked_intervals(
                        likelihood.admitted_pre_gain_mean_hull_intervals,
                        size=channel_count,
                        name="parent likelihood admitted hulls",
                    )
                    if any(
                        left is None or not _same_interval(left, right)
                        for left, right in zip(
                            parent_hulls, hulls, strict=True
                        )
                    ):
                        raise ValueError(
                            "parent likelihood admitted hulls disagree with "
                            "detector"
                        )

    law_available = bool(result["mean_evidence_available"]) and bool(
        np.asarray(detector.likelihood_law_eligible)
    )
    if bool(result["mean_evidence_available"]) and not law_available:
        failure |= (
            GalerkinLocalResidualObservableFailure.PARENT_LAW_NONCERTIFICATE
        )
    if law_available:
        full_eta = result["full_mean_l1_error_bound_interval"].lower
        fitted_eta = result["fitted_mean_l1_error_bound_interval"].lower
        try:
            full_tv = _tv_bound_evidence(
                full_eta,
                call=(
                    GalerkinLocalResidualObservableHelperCall.FULL_TV_EXPONENTIAL
                ),
                manifest=manifest,
                ledger=ledger,
            )
            fitted_tv = _tv_bound_evidence(
                fitted_eta,
                call=(
                    GalerkinLocalResidualObservableHelperCall.FITTED_TV_EXPONENTIAL
                ),
                manifest=manifest,
                ledger=ledger,
            )
        except _ResidualObservableArithmeticError as error:
            failure |= (
                GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
            )
            counters[1:] = [error.exact_work_count] * 3
            attempted = error.exact_work_count
        else:
            if ledger.exact_work_count != law_plan:
                raise ValueError(
                    "residual-observable law work formula disagrees"
                )
            result.update(
                {
                    "law_evidence_available": True,
                    "full_law_evidence_available": True,
                    "fitted_law_evidence_available": True,
                    "full_linear_tv_bound_interval": full_tv[0],
                    "fitted_linear_tv_bound_interval": fitted_tv[0],
                    "full_exponential_tv_bound_interval": full_tv[1],
                    "fitted_exponential_tv_bound_interval": fitted_tv[1],
                    "full_selected_tv_bound_interval": full_tv[2],
                    "fitted_selected_tv_bound_interval": fitted_tv[2],
                    "full_tv_exp_transcript": full_tv[3],
                    "fitted_tv_exp_transcript": fitted_tv[3],
                    "full_tv_exp_failure": full_tv[4],
                    "fitted_tv_exp_failure": fitted_tv[4],
                    "strongest_layer": (
                        GalerkinLocalResidualObservableLayer.LAW
                    ),
                }
            )
            counters[1:] = [law_plan] * 3
            attempted = law_plan
            if full_tv[4] is not None or fitted_tv[4] is not None:
                failure |= failure_type.EXPONENTIAL_ENCLOSURE_FAILURE

    nll_gate = bool(result["law_evidence_available"]) and bool(
        np.asarray(likelihood.nll_eligible)
    )
    direct_available = False
    if bool(result["law_evidence_available"]) and not nll_gate:
        failure |= (
            GalerkinLocalResidualObservableFailure.DIRECT_NLL_UNAVAILABLE
        )
    if nll_gate:
        production_points = likelihood.production_nll_point_intervals
        if (
            type(production_points) is not tuple
            or len(production_points) != channel_count
        ):
            raise ValueError(
                "parent production NLL points are not channel aligned"
            )
        fitted_points: list[GalerkinLocalDetectorRationalInterval] = []
        for fitted, point in zip(fit_mask, production_points, strict=True):
            if not fitted:
                if point is not None:
                    raise ValueError(
                        "unfitted production NLL point is not None"
                    )
                continue
            if point is None:
                raise ValueError("fitted production NLL point is unavailable")
            if type(point) is not GalerkinLocalDetectorRationalInterval:
                raise TypeError(
                    "fitted production NLL point has the wrong exact type"
                )
            if (
                point.lower_numerator != point.upper_numerator
                or point.lower_denominator != point.upper_denominator
            ):
                raise ValueError("production NLL point is not a singleton")
            fitted_points.append(point)
        parent_total_raw = likelihood.total_nll_interval
        if type(parent_total_raw) is not GalerkinLocalDetectorRationalInterval:
            raise TypeError("parent total NLL has the wrong exact type")
        try:
            ledger.scan_intervals((production_points, parent_total_raw))
        except _ResidualObservableArithmeticError as error:
            failure |= (
                GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
            )
            counters[2:] = [error.exact_work_count] * 2
            attempted = error.exact_work_count
            parent_total = None
        else:
            parent_total = _validate_local_detector_rational_interval(
                parent_total_raw
            )
        if parent_total is not None:
            validated_points: list[GalerkinLocalDetectorRationalInterval] = []
            for point in fitted_points:
                checked = _validate_local_detector_rational_interval(point)
                if checked.lower != checked.upper:
                    raise ValueError("production NLL point is not a singleton")
                validated_points.append(checked)
            try:
                production_total = Fraction()
                for point in validated_points:
                    production_total = ledger.add(
                        production_total, point.lower
                    )
                lower_distance = abs(
                    ledger.subtract(production_total, parent_total.lower)
                )
                upper_distance = abs(
                    ledger.subtract(production_total, parent_total.upper)
                )
                direct_error = max(lower_distance, upper_distance)
            except _ResidualObservableArithmeticError as error:
                failure |= (
                    GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
                )
                counters[2:] = [error.exact_work_count] * 2
                attempted = error.exact_work_count
            else:
                if ledger.exact_work_count != direct_plan:
                    raise ValueError(
                        "residual-observable direct-NLL work formula disagrees"
                    )
                production_total_interval = _point_interval(production_total)
                direct_error_interval = _point_interval(direct_error)
                direct_available = True
                result.update(
                    {
                        "direct_nll_evidence_available": True,
                        "production_fitted_total_nll_interval": (
                            production_total_interval
                        ),
                        "direct_nll_error_bound_interval": (
                            direct_error_interval
                        ),
                    }
                )
                counters[2:] = [direct_plan] * 2
                attempted = direct_plan

    score_available = False
    score_fatal = bool(
        failure & GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
    )
    if nll_gate and not score_fatal:
        predecessors: list[GalerkinLocalDetectorRationalInterval | None] = (
            list(empty)
        )
        tail_floors: list[GalerkinLocalDetectorRationalInterval | None] = list(
            empty
        )
        poisson_transcripts: list[CensoredPoissonWorkTranscript | None] = list(
            empty
        )
        poisson_failures: list[
            GalerkinLocalResidualObservableHelperFailureEvidence | None
        ] = list(empty)
        score_blocked = False
        score_rational_error: _ResidualObservableArithmeticError | None = None
        # Scientific score prerequisites are checked before any helper or
        # score-only local transaction, so a stop cannot retain partial terms.
        mean_hulls = result["admitted_pre_gain_mean_hull_intervals"]
        for channel, (fitted, count, ceiling) in enumerate(
            zip(fit_mask, observed, ceilings, strict=True)
        ):
            if not fitted:
                continue
            if 0 < count < ceiling and mean_hulls[channel].lower <= 0:
                failure |= failure_type.SCORE_MEAN_FLOOR_UNAVAILABLE
                score_blocked = True
            if ceiling > 0 and count == ceiling:
                floor = likelihood.fitted_probability_positive_floor_intervals[
                    channel
                ]
                if floor is None:
                    failure |= failure_type.SCORE_PROBABILITY_FLOOR_UNAVAILABLE
                    score_blocked = True
                    continue
                try:
                    ledger.scan_intervals(floor)
                except _ResidualObservableArithmeticError as error:
                    score_rational_error = error
                    score_blocked = True
                    break
                checked_floor = _validate_local_detector_rational_interval(
                    floor
                )
                if (
                    checked_floor.lower <= 0
                    or checked_floor.lower != checked_floor.upper
                ):
                    failure |= failure_type.SCORE_PROBABILITY_FLOOR_UNAVAILABLE
                    score_blocked = True
                else:
                    tail_floors[channel] = checked_floor
        # Trace inputs are raw-scanned before any score arithmetic.
        if not score_blocked:
            for channel, (fitted, ceiling) in enumerate(
                zip(fit_mask, ceilings, strict=True)
            ):
                if not fitted or ceiling == 0:
                    continue
                point = likelihood.production_nll_point_intervals[channel]
                if point is None:
                    raise ValueError(
                        "fitted production NLL point is unavailable"
                    )
                try:
                    _production_nll_trace(likelihood, channel, point, ledger)
                except _ResidualObservableArithmeticError as error:
                    score_rational_error = error
                    score_blocked = True
                    break
        # Saturated predecessor helpers run in fitted-channel order before
        # any score-local transaction.
        for channel, (fitted, count, ceiling) in enumerate(
            zip(fit_mask, observed, ceilings, strict=True)
        ):
            if score_blocked or not fitted or ceiling == 0 or count != ceiling:
                continue
            checked_floor = tail_floors[channel]
            hull = mean_hulls[channel]
            try:
                probability, transcript = enclose_censored_poisson_probability(
                    (hull.lower, hull.upper),
                    ceiling - 1,
                    ceiling,
                    maximum_count_ceiling=detector_manifest.maximum_count_ceiling,
                    maximum_work=detector_manifest.maximum_poisson_work,
                    maximum_rational_bits=detector_manifest.maximum_poisson_rational_bits,
                    exp_precision_bits=detector_manifest.exp_precision_bits,
                    maximum_exp_terms=detector_manifest.maximum_exp_terms,
                    maximum_exp_work=detector_manifest.maximum_exp_work,
                    maximum_exp_range_reductions=(
                        detector_manifest.maximum_exp_range_reductions
                    ),
                )
            except CensoredPoissonEnclosureError as error:
                poisson_failures[channel] = _poisson_helper_failure_from_error(
                    channel, error
                )
                failure |= failure_type.POISSON_ENCLOSURE_FAILURE
                if error.nested_failure is not None:
                    failure |= failure_type.NESTED_HELPER_FAILURE
                score_blocked = True
                break
            predecessors[channel] = _point_interval(probability[1])
            poisson_transcripts[channel] = transcript
            try:
                ledger.scan_intervals(predecessors[channel])
            except _ResidualObservableArithmeticError as error:
                score_rational_error = error
                score_blocked = True
                break
        result.update(
            {
                "saturated_predecessor_mass_upper_intervals": tuple(
                    predecessors
                ),
                "saturated_tail_probability_floor_intervals": tuple(
                    tail_floors
                ),
                "saturated_probability_transcripts": tuple(
                    poisson_transcripts
                ),
                "saturated_probability_failures": tuple(poisson_failures),
            }
        )
        if score_blocked:
            if score_rational_error is not None:
                failure |= (
                    GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
                )
                counters[3] = score_rational_error.exact_work_count
                attempted = score_rational_error.exact_work_count
            else:
                counters[3] = direct_plan if direct_available else law_plan
                attempted = score_plan
        else:
            factors: list[GalerkinLocalDetectorRationalInterval | None] = list(
                empty
            )
            rounding: list[GalerkinLocalDetectorRationalInterval | None] = (
                list(empty)
            )
            terms: list[GalerkinLocalDetectorRationalInterval | None] = list(
                empty
            )
            score_total = Fraction()
            for channel, (fitted, count, ceiling) in enumerate(
                zip(fit_mask, observed, ceilings, strict=True)
            ):
                if not fitted:
                    continue
                try:
                    if ceiling == 0:
                        factor = rho = Fraction()
                    else:
                        production_point = (
                            likelihood.production_nll_point_intervals[channel]
                        )
                        if production_point is None:
                            raise ValueError(
                                "fitted production NLL point is unavailable"
                            )
                        rho = _trace_rounding_error(
                            likelihood, channel, production_point, ledger
                        )
                        if count == 0:
                            factor = Fraction(1)
                        elif count < ceiling:
                            hull = result[
                                "admitted_pre_gain_mean_hull_intervals"
                            ][channel]
                            count_value = Fraction(count)
                            lower_ratio = ledger.divide(
                                count_value, hull.lower
                            )
                            upper_ratio = ledger.divide(
                                count_value, hull.upper
                            )
                            factor = max(
                                abs(ledger.subtract(Fraction(1), lower_ratio)),
                                abs(ledger.subtract(Fraction(1), upper_ratio)),
                            )
                            ledger.commit(factor)
                        else:
                            predecessor = predecessors[channel]
                            floor = tail_floors[channel]
                            if predecessor is None or floor is None:
                                raise ValueError(
                                    "saturated score helper evidence is "
                                    "incomplete"
                                )
                            factor = ledger.divide(
                                predecessor.upper, floor.lower
                            )
                            ledger.commit(factor)
                    distance = result["channel_mean_error_bound_intervals"][
                        channel
                    ].lower
                    if ceiling == 0:
                        term = Fraction()
                        score_total = ledger.add(score_total, term)
                        ledger.commit(term)
                    elif count == 0:
                        term = ledger.add(distance, rho)
                        score_total = ledger.add(score_total, term)
                    else:
                        term, score_total = ledger.score_accumulate(
                            factor, distance, rho, score_total
                        )
                except _ResidualObservableArithmeticError as error:
                    score_rational_error = error
                    score_blocked = True
                    break
                factors[channel] = _point_interval(factor)
                rounding[channel] = _point_interval(rho)
                terms[channel] = _point_interval(term)
            result.update(
                {
                    "score_lipschitz_factor_intervals": tuple(factors),
                    "score_rounding_error_intervals": tuple(rounding),
                    "score_term_error_intervals": tuple(terms),
                }
            )
            if score_blocked:
                failure |= (
                    GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
                )
                if score_rational_error is None:
                    raise ValueError(
                        "residual-observable score stop lacks work evidence"
                    )
                counters[3] = score_rational_error.exact_work_count
                attempted = score_rational_error.exact_work_count
                result.update(
                    {
                        "score_lipschitz_factor_intervals": empty,
                        "score_rounding_error_intervals": empty,
                        "score_term_error_intervals": empty,
                    }
                )
            else:
                if ledger.exact_work_count != score_plan:
                    raise ValueError(
                        "residual-observable score work formula disagrees"
                    )
                score_interval = _point_interval(score_total)
                score_available = True
                result["score_nll_evidence_available"] = True
                result["score_nll_error_bound_interval"] = score_interval
                counters[3] = score_plan
                attempted = score_plan

    if direct_available or score_available:
        result["selected_nll_evidence_available"] = True
        result["strongest_layer"] = (
            GalerkinLocalResidualObservableLayer.POINTWISE_NLL
        )
        direct_bound = result["direct_nll_error_bound_interval"]
        score_bound = result["score_nll_error_bound_interval"]
        if direct_bound is None:
            selected = score_bound
            route = GalerkinLocalResidualObservableRoute.SCORE_LIPSCHITZ
        elif score_bound is None or direct_bound.upper < score_bound.upper:
            selected = direct_bound
            route = GalerkinLocalResidualObservableRoute.DIRECT_ADMITTED_HULL
        elif score_bound.upper < direct_bound.upper:
            selected = score_bound
            route = GalerkinLocalResidualObservableRoute.SCORE_LIPSCHITZ
        else:
            selected = direct_bound
            route = GalerkinLocalResidualObservableRoute.TIED
        result["selected_nll_error_bound_interval"] = selected
        result["selected_nll_route"] = route

    result["failure_mask"] = failure
    helper_work = sum(
        (
            _entire_transcript_work(result["full_tv_exp_transcript"]),
            _entire_transcript_work(result["fitted_tv_exp_transcript"]),
            _helper_failure_work(result["full_tv_exp_failure"]),
            _helper_failure_work(result["fitted_tv_exp_failure"]),
            *(
                _poisson_transcript_work(value)
                for value in result["saturated_probability_transcripts"]
            ),
            *(
                _helper_failure_work(value)
                for value in result["saturated_probability_failures"]
            ),
        )
    )
    completed_layer = result["strongest_layer"]
    exact_work = counters[3]
    work = _make_local_residual_observable_work_transcript(
        _make_local_residual_observable_work_transcript_candidate(
            algorithm=_RESIDUAL_OBSERVABLE_WORK_ALGORITHM,
            maximum_work=manifest.maximum_residual_observable_work,
            maximum_rational_bits=manifest.maximum_residual_observable_rational_bits,
            channel_count=channel_count,
            fitted_channel_count=fitted_count,
            zero_observation_positive_ceiling_count=zero_count,
            interior_observation_count=interior_count,
            saturated_positive_ceiling_count=saturated_count,
            mean_exact_work_count=counters[0],
            law_exact_work_count=counters[1],
            direct_nll_exact_work_count=counters[2],
            score_nll_exact_work_count=counters[3],
            exact_work_count=exact_work,
            rational_peak_bits=ledger.rational_peak_bits,
            nested_parent_work_count_exact=str(
                _nested_parent_work(likelihood)
            ),
            nested_helper_work_count_exact=str(helper_work),
            planned_mean_exact_work_count_exact=str(mean_plan),
            planned_law_exact_work_count_exact=str(law_plan),
            planned_direct_nll_exact_work_count_exact=str(direct_plan),
            planned_score_nll_exact_work_count_exact=str(score_plan),
            attempted_exact_work_count_exact=str(attempted),
            completed_layer=completed_layer,
            completed_successfully=(
                failure == GalerkinLocalResidualObservableFailure.NONE
                and completed_layer
                is GalerkinLocalResidualObservableLayer.POINTWISE_NLL
            ),
            failure=failure,
            preflight_failed=preflight_failed,
            count_overflow=count_overflow,
            nested_parent_work_scope=_NESTED_PARENT_WORK_SCOPE,
        )
    )
    result["work_transcript"] = work
    return result


def _certificate_digests(
    certificate: GalerkinLocalResidualObservableCertificate,
) -> tuple[str, str, str]:
    """PRIVATE: Derive identity, evidence, and certificate digests.

    Parameters
    ----------
    certificate : GalerkinLocalResidualObservableCertificate
        Required canonical input.

    Returns
    -------
    identity : str
        Canonical identity digest.
    evidence : str
        Canonical evidence digest.
    certificate_digest : str
        Canonical certificate digest.
    """
    digest_names = {
        "observable_identity_digest",
        "observable_evidence_digest",
        "certificate_digest",
    }
    identity_names = (
        "parent_detector_certificate_digest",
        "parent_detector_input_manifest_digest",
        "parent_likelihood_certificate_digest",
        "input_manifest_digest",
    )
    identity = sha256(
        {
            "domain": "ptyrodactyl.local_residual_observable.identity.v1",
            "fields": {
                name: stored_value_payload(getattr(certificate, name))
                for name in identity_names
            },
        }
    )
    evidence = sha256(
        {
            "domain": "ptyrodactyl.local_residual_observable.evidence.v1",
            "identity": identity,
            "fields": {
                field.name: stored_value_payload(
                    getattr(certificate, field.name)
                )
                for field in fields(certificate)
                if field.name not in digest_names
            },
        }
    )
    certificate_digest = sha256(
        {
            "domain": "ptyrodactyl.local_residual_observable.certificate.v1",
            "identity": identity,
            "evidence": evidence,
        }
    )
    return identity, evidence, certificate_digest


def _make_local_residual_observable_certificate_candidate(
    **values: Any,
) -> GalerkinLocalResidualObservableCertificate:
    """PRIVATE: Construct one exact-field certificate candidate.

    Parameters
    ----------
    **values : Any
        Exact declared certificate fields.

    Returns
    -------
    result : GalerkinLocalResidualObservableCertificate
        Unsealed candidate.
    """
    result: GalerkinLocalResidualObservableCertificate = (
        GalerkinLocalResidualObservableCertificate(**values)
    )
    return result  # noqa: RET504


def _expected_certificate_fields_checked(  # noqa: C901, PLR0912, PLR0915
    certificate: GalerkinLocalResidualObservableCertificate,
) -> dict[str, Any]:
    """PRIVATE: Validate primitives and exact-compare the derived RM-S5 DAG.

    Parameters
    ----------
    certificate : GalerkinLocalResidualObservableCertificate
        Candidate certificate, whose digest fields may still be placeholders.

    Returns
    -------
    result : dict[str, Any]
        Independently recomputed derived fields.

    Raises
    ------
    TypeError
        If a carrier, scalar, tuple, or enum has the wrong exact type.
    ValueError
        If parent bindings, scopes, or any derived field disagree.
    """
    if type(certificate) is not GalerkinLocalResidualObservableCertificate:
        raise TypeError("residual-observable certificate has the wrong type")
    likelihood = _validate_local_censored_poisson_likelihood(
        certificate.parent_likelihood
    )
    manifest = _validate_local_residual_observable_input_manifest(
        certificate.input_manifest
    )
    detector = likelihood.detector
    channel_count = np.asarray(detector.response_matrix).shape[0]
    flag_names = (
        "state_evidence_available",
        "mean_evidence_available",
        "law_evidence_available",
        "full_law_evidence_available",
        "fitted_law_evidence_available",
        "direct_nll_evidence_available",
        "score_nll_evidence_available",
        "selected_nll_evidence_available",
    )
    for name in flag_names:
        stored_value = getattr(certificate, name)
        if not _is_concrete_array_carrier(stored_value):
            raise TypeError(
                "residual-observable availability must be an array carrier"
            )
        value = np.asarray(stored_value)
        if value.dtype != np.dtype(np.bool_) or value.shape != ():
            raise TypeError("residual-observable availability must be bool")
    if not _is_concrete_array_carrier(certificate.failure_mask):
        raise TypeError(
            "residual-observable failure mask must be an array carrier"
        )
    failure_array = np.asarray(certificate.failure_mask)
    if failure_array.dtype != np.dtype(np.int64) or failure_array.shape != ():
        raise TypeError("residual-observable failure mask must be int64")
    if int(failure_array) & ~_KNOWN_RESIDUAL_OBSERVABLE_FAILURE_MASK:
        raise ValueError("residual-observable failure mask has unknown bits")
    if (
        type(certificate.strongest_layer)
        is not GalerkinLocalResidualObservableLayer
        or type(certificate.selected_nll_route)
        is not GalerkinLocalResidualObservableRoute
        or type(certificate.full_law_scope)
        is not GalerkinLocalResidualObservableScope
        or type(certificate.fitted_law_scope)
        is not GalerkinLocalResidualObservableScope
    ):
        raise TypeError("residual-observable enum field has wrong type")
    scope_names = (
        "mean_scope",
        "law_scope",
        "nll_scope",
        "resource_scope",
        "no_scientific_claim_scope",
    )
    if any(
        type(getattr(certificate, name)) is not str for name in scope_names
    ):
        raise TypeError("residual-observable scopes must be exact strings")
    if (
        certificate.full_law_scope
        is not GalerkinLocalResidualObservableScope.FULL_CHANNEL_LAW
        or certificate.fitted_law_scope
        is not GalerkinLocalResidualObservableScope.FIXED_FIT_PROJECTION
        or certificate.mean_scope != _MEAN_SCOPE
        or certificate.law_scope != _LAW_SCOPE
        or certificate.nll_scope != _NLL_SCOPE
        or certificate.resource_scope != _RESOURCE_SCOPE
        or certificate.no_scientific_claim_scope != _NO_SCIENTIFIC_CLAIM_SCOPE
    ):
        raise ValueError("residual-observable canonical scope disagrees")
    parent_digests = (
        certificate.parent_detector_certificate_digest,
        certificate.parent_detector_input_manifest_digest,
        certificate.parent_likelihood_certificate_digest,
        certificate.input_manifest_digest,
    )
    if not all(_valid_digest(value) for value in parent_digests):
        raise ValueError("residual-observable parent digest is not SHA-256")
    if parent_digests != (
        detector.certificate_digest,
        detector.input_manifest_digest,
        likelihood.certificate_digest,
        manifest.manifest_digest,
    ):
        raise ValueError("residual-observable parent provenance disagrees")
    if (
        manifest.detector_input_manifest.manifest_digest
        != detector.input_manifest_digest
        or not bool(
            eqx.tree_equal(
                manifest.observed_counts,
                likelihood.observed_counts,
                typematch=True,
            )
        )
        or manifest.maximum_detector_work
        != likelihood.work_transcript.maximum_work
        or manifest.maximum_detector_rational_bits
        != likelihood.work_transcript.maximum_rational_bits
        or manifest.log_precision_bits != likelihood.log_precision_bits
        or manifest.maximum_log_terms != likelihood.maximum_log_terms
        or manifest.maximum_log_work != likelihood.maximum_log_work
        or manifest.maximum_log_range_reductions
        != likelihood.maximum_log_range_reductions
    ):
        raise ValueError(
            "residual-observable input manifest and parent replay disagree"
        )
    optional_interval_tuple_names = (
        "score_lipschitz_factor_intervals",
        "score_rounding_error_intervals",
        "score_term_error_intervals",
        "saturated_predecessor_mass_upper_intervals",
        "saturated_tail_probability_floor_intervals",
    )
    for name in (
        "admitted_pre_gain_mean_hull_intervals",
        "channel_mean_error_bound_intervals",
    ):
        values = getattr(certificate, name)
        if values is not None:
            _checked_intervals(
                values,
                size=channel_count,
                name=name,
            )
    for name in optional_interval_tuple_names:
        _checked_intervals(
            getattr(certificate, name),
            size=channel_count,
            name=name,
            optional_items=True,
        )
    for name in (
        "saturated_probability_transcripts",
        "saturated_probability_failures",
    ):
        values = getattr(certificate, name)
        if type(values) is not tuple:
            raise TypeError(f"{name} must be an exact tuple")
        if len(values) != channel_count:
            raise ValueError(f"{name} has the wrong length")
    for transcript in (
        certificate.full_tv_exp_transcript,
        certificate.fitted_tv_exp_transcript,
    ):
        if transcript is not None:
            _validate_residual_observable_entire_transcripts((transcript,), ())
    observed = tuple(
        int(value) for value in np.asarray(manifest.observed_counts)
    )
    ceilings = tuple(
        int(value) for value in np.asarray(detector.count_ceilings)
    )
    for channel, transcript in enumerate(
        certificate.saturated_probability_transcripts
    ):
        if transcript is not None:
            _validate_saturated_probability_transcript(
                transcript,
                manifest=manifest,
                observed_count=ceilings[channel] - 1,
                count_ceiling=ceilings[channel],
            )
            if (
                observed[channel] != ceilings[channel]
                or ceilings[channel] <= 0
            ):
                raise ValueError(
                    "saturated probability transcript channel disagrees"
                )
    for helper in (
        certificate.full_tv_exp_failure,
        certificate.fitted_tv_exp_failure,
        *certificate.saturated_probability_failures,
    ):
        if helper is not None:
            _validate_local_residual_observable_helper_failure(helper)
    _validate_local_residual_observable_work_transcript(
        certificate.work_transcript
    )
    expected = _expected_local_residual_observable_evidence(
        likelihood, manifest
    )
    for name in flag_names:
        if bool(np.asarray(getattr(certificate, name))) is not expected[name]:
            raise ValueError(f"residual-observable {name} disagrees")
    expected_failure = expected["failure_mask"]
    if int(failure_array) != int(expected_failure):
        raise ValueError("residual-observable failure mask disagrees")
    derived_names = tuple(
        name for name in expected if name not in (*flag_names, "failure_mask")
    )
    for name in derived_names:
        if not _interval_payload_equal(
            getattr(certificate, name), expected[name]
        ):
            raise ValueError(f"residual-observable derived {name} disagrees")
    return expected


def _validate_local_residual_observable_certificate(
    certificate: GalerkinLocalResidualObservableCertificate,
) -> GalerkinLocalResidualObservableCertificate:
    """PRIVATE: Validate one independently replayed and sealed certificate.

    Parameters
    ----------
    certificate : GalerkinLocalResidualObservableCertificate
        Candidate certificate.

    Returns
    -------
    result : GalerkinLocalResidualObservableCertificate
        Canonical validated certificate.

    Raises
    ------
    TypeError
        If a carrier or primitive field has the wrong exact type.
    ValueError
        If replayed evidence, provenance, work, or digests disagree.
    """
    _expected_certificate_fields_checked(certificate)
    expected_digests = _certificate_digests(certificate)
    stored_digests = (
        certificate.observable_identity_digest,
        certificate.observable_evidence_digest,
        certificate.certificate_digest,
    )
    if not all(_valid_digest(value) for value in stored_digests) or (
        stored_digests != expected_digests
    ):
        raise ValueError("residual-observable certificate digest disagrees")
    return certificate


def _make_local_residual_observable_certificate(
    certificate: GalerkinLocalResidualObservableCertificate,
) -> GalerkinLocalResidualObservableCertificate:
    """PRIVATE: Seal and validate one owner-constructed RM-S5 certificate.

    Parameters
    ----------
    certificate : GalerkinLocalResidualObservableCertificate
        Unsealed owner candidate.

    Returns
    -------
    result : GalerkinLocalResidualObservableCertificate
        Canonical sealed certificate.

    Raises
    ------
    TypeError
        If the candidate has the wrong exact carrier type.
    """
    _expected_certificate_fields_checked(certificate)
    digests = _certificate_digests(certificate)
    sealed = replace(
        certificate,
        observable_identity_digest=digests[0],
        observable_evidence_digest=digests[1],
        certificate_digest=digests[2],
    )
    return _validate_local_residual_observable_certificate(sealed)


__all__: list[str] = [
    "GalerkinLocalResidualObservableCertificate",
    "GalerkinLocalResidualObservableFailure",
    "GalerkinLocalResidualObservableHelperCall",
    "GalerkinLocalResidualObservableHelperFailureEvidence",
    "GalerkinLocalResidualObservableInputManifest",
    "GalerkinLocalResidualObservableLayer",
    "GalerkinLocalResidualObservableRoute",
    "GalerkinLocalResidualObservableScope",
    "GalerkinLocalResidualObservableWorkTranscript",
]
