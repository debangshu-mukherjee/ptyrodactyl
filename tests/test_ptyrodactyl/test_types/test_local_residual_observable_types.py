r"""Tests for owned RM-S5 residual-to-observable evidence carriers."""

from __future__ import annotations

import dataclasses
from enum import IntFlag, StrEnum
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from ptyrodactyl._tools import (
    CensoredPoissonEnclosureFailure,
    EntireEnclosureFailure,
    EntireWorkTranscript,
)
from ptyrodactyl.galerkin.detector import (
    create_local_censored_poisson_detector_input_manifest,
)
from ptyrodactyl.types.local_residual_observable_types import (
    _NESTED_PARENT_WORK_SCOPE,
    _RESIDUAL_OBSERVABLE_WORK_ALGORITHM,
    GalerkinLocalResidualObservableCertificate,
    GalerkinLocalResidualObservableFailure,
    GalerkinLocalResidualObservableHelperCall,
    GalerkinLocalResidualObservableHelperFailureEvidence,
    GalerkinLocalResidualObservableInputManifest,
    GalerkinLocalResidualObservableLayer,
    GalerkinLocalResidualObservableRoute,
    GalerkinLocalResidualObservableScope,
    GalerkinLocalResidualObservableWorkTranscript,
    _make_local_residual_observable_helper_failure,
    _make_local_residual_observable_helper_failure_candidate,
    _make_local_residual_observable_input_manifest,
    _make_local_residual_observable_input_manifest_candidate,
    _make_local_residual_observable_work_transcript,
    _make_local_residual_observable_work_transcript_candidate,
    _validate_local_residual_observable_helper_failure,
    _validate_local_residual_observable_input_manifest,
    _validate_local_residual_observable_work_transcript,
)
from tests.test_ptyrodactyl.test_types import (
    test_local_detector_types as l9_type_tests,
)


def _fields(value: type[object]) -> set[str]:
    """Return the exact declared carrier field names."""
    return {field.name for field in dataclasses.fields(value)}


def _detector_manifest():
    """Return one small parent-free nested L9 detector manifest."""
    return create_local_censored_poisson_detector_input_manifest(
        pixel_inputs=(l9_type_tests._pixel_manifest(),),
        ensemble_weight_numerators=(1,),
        ensemble_weight_denominators=(1,),
        incident_electron_count_interval=l9_type_tests._interval(1),
        incident_electron_count_point=np.asarray(1.0, dtype=np.float64),
        response_matrix=np.asarray([[1.0]], dtype=np.float64),
        pre_gain_background=np.asarray([0.0], dtype=np.float64),
        deterministic_gain=np.asarray([1.0], dtype=np.float64),
        electronic_offset=np.asarray([0.0], dtype=np.float64),
        count_ceilings=np.asarray([2], dtype=np.int64),
        fit_mask=np.asarray([True], dtype=np.bool_),
        calibration_provenance="parent-free RM-S5 type fixture",
        maximum_detector_work=10_000,
        maximum_detector_rational_bits=256,
        maximum_count_ceiling=8,
        maximum_poisson_work=10_000,
        maximum_poisson_rational_bits=256,
        exp_precision_bits=64,
        maximum_exp_terms=256,
        maximum_exp_work=10_000,
        maximum_exp_range_reductions=64,
    )


def _manifest_candidate():
    """Return one unsealed parent-free RM-S5 manifest."""
    return _make_local_residual_observable_input_manifest_candidate(
        detector_input_manifest=_detector_manifest(),
        observed_counts=jnp.asarray([1], dtype=jnp.int64),
        maximum_detector_work=10_000,
        maximum_detector_rational_bits=256,
        log_precision_bits=64,
        maximum_log_terms=256,
        maximum_log_work=10_000,
        maximum_log_range_reductions=64,
        tv_exp_precision_bits=64,
        maximum_tv_exp_terms=256,
        maximum_tv_exp_work=10_000,
        maximum_tv_exp_range_reductions=64,
        maximum_residual_observable_work=10_000,
        maximum_residual_observable_rational_bits=256,
        manifest_digest="0" * 64,
    )


def test_residual_observable_enums_are_explicit_disjoint_and_layered() -> None:
    """Freeze the RM-S5 layer, scope, route, helper, and failure domains.

    :see: :class:`ptyrodactyl.types.GalerkinLocalResidualObservableLayer`
    :see: :class:`ptyrodactyl.types.GalerkinLocalResidualObservableScope`
    :see: :class:`ptyrodactyl.types.GalerkinLocalResidualObservableRoute`
    :see: :class:`ptyrodactyl.types.GalerkinLocalResidualObservableHelperCall`
    :see: :class:`ptyrodactyl.types.GalerkinLocalResidualObservableFailure`
    """
    assert {
        item.name: item.value for item in GalerkinLocalResidualObservableLayer
    } == {
        "UNAVAILABLE": "unavailable",
        "STATE": "state",
        "MEAN": "mean",
        "LAW": "law",
        "POINTWISE_NLL": "pointwise_nll",
    }
    assert {
        item.name: item.value for item in GalerkinLocalResidualObservableScope
    } == {
        "FULL_CHANNEL_LAW": "full_channel_law",
        "FIXED_FIT_PROJECTION": "fixed_fit_projection",
    }
    assert {
        item.name: item.value for item in GalerkinLocalResidualObservableRoute
    } == {
        "UNAVAILABLE": "unavailable",
        "DIRECT_ADMITTED_HULL": "direct_admitted_hull",
        "SCORE_LIPSCHITZ": "score_lipschitz",
        "TIED": "tied",
    }
    assert {
        item.name: item.value
        for item in GalerkinLocalResidualObservableHelperCall
    } == {
        "FULL_TV_EXPONENTIAL": "full_tv_exponential",
        "FITTED_TV_EXPONENTIAL": "fitted_tv_exponential",
        "SATURATED_SCORE_MASS": "saturated_score_mass",
    }
    assert issubclass(GalerkinLocalResidualObservableFailure, IntFlag)
    assert {
        item.name: int(item) for item in GalerkinLocalResidualObservableFailure
    } == {
        "PARENT_STATE_NONCERTIFICATE": 1 << 0,
        "PARENT_MEAN_NONCERTIFICATE": 1 << 1,
        "PARENT_LAW_NONCERTIFICATE": 1 << 2,
        "ADMITTED_MEAN_HULL_UNAVAILABLE": 1 << 3,
        "PRODUCTION_MEAN_NONSINGLETON": 1 << 4,
        "PRODUCTION_MEAN_OUTSIDE_HULL": 1 << 5,
        "DIRECT_NLL_UNAVAILABLE": 1 << 6,
        "SCORE_MEAN_FLOOR_UNAVAILABLE": 1 << 7,
        "SCORE_PROBABILITY_FLOOR_UNAVAILABLE": 1 << 8,
        "EXPONENTIAL_ENCLOSURE_FAILURE": 1 << 9,
        "POISSON_ENCLOSURE_FAILURE": 1 << 10,
        "NESTED_HELPER_FAILURE": 1 << 11,
        "EXACT_WORK_BUDGET_EXCEEDED": 1 << 12,
        "EXACT_WORK_COUNT_OVERFLOW": 1 << 13,
        "RATIONAL_SIZE_LIMIT": 1 << 14,
        "ARITHMETIC_RANGE_FAILURE": 1 << 15,
    }


def test_residual_observable_manifest_is_owner_sealed_and_policy_complete() -> (  # noqa: E501
    None
):
    """Freeze the nested L9 replay, TV, log, and RM-S5 policy surface.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalResidualObservableInputManifest`
    """
    assert _fields(GalerkinLocalResidualObservableInputManifest) == {
        "detector_input_manifest",
        "observed_counts",
        "maximum_detector_work",
        "maximum_detector_rational_bits",
        "log_precision_bits",
        "maximum_log_terms",
        "maximum_log_work",
        "maximum_log_range_reductions",
        "tv_exp_precision_bits",
        "maximum_tv_exp_terms",
        "maximum_tv_exp_work",
        "maximum_tv_exp_range_reductions",
        "maximum_residual_observable_work",
        "maximum_residual_observable_rational_bits",
        "manifest_digest",
    }
    sealed = _make_local_residual_observable_input_manifest(
        _manifest_candidate()
    )
    assert _validate_local_residual_observable_input_manifest(sealed) is sealed
    assert len(sealed.manifest_digest) == 64
    changed = dataclasses.replace(
        sealed, observed_counts=np.asarray([0], dtype=np.int64)
    )
    with pytest.raises(ValueError, match="digest disagrees"):
        _validate_local_residual_observable_input_manifest(changed)
    resealed = _make_local_residual_observable_input_manifest(changed)
    assert resealed.manifest_digest != sealed.manifest_digest

    with pytest.raises(ValueError, match="shape or dtype"):
        _make_local_residual_observable_input_manifest(
            dataclasses.replace(
                sealed, observed_counts=np.asarray([1], dtype=np.int32)
            )
        )
    for observed in (
        cast(jnp.ndarray, (1,)),
        cast(jnp.ndarray, [1]),
    ):
        with pytest.raises(TypeError, match="array carrier"):
            _make_local_residual_observable_input_manifest(
                dataclasses.replace(sealed, observed_counts=observed)
            )
    with pytest.raises(ValueError, match="ceilings"):
        _make_local_residual_observable_input_manifest(
            dataclasses.replace(
                sealed, observed_counts=np.asarray([3], dtype=np.int64)
            )
        )
    for name in (
        "maximum_detector_work",
        "maximum_detector_rational_bits",
        "log_precision_bits",
        "maximum_log_terms",
        "maximum_log_work",
        "maximum_log_range_reductions",
        "tv_exp_precision_bits",
        "maximum_tv_exp_terms",
        "maximum_tv_exp_work",
        "maximum_tv_exp_range_reductions",
        "maximum_residual_observable_work",
        "maximum_residual_observable_rational_bits",
    ):
        with pytest.raises(TypeError, match="Python ints"):
            _make_local_residual_observable_input_manifest(
                dataclasses.replace(sealed, **{name: True})
            )
    for name in (
        "maximum_detector_work",
        "log_precision_bits",
        "maximum_log_terms",
        "maximum_log_work",
        "tv_exp_precision_bits",
        "maximum_tv_exp_terms",
        "maximum_tv_exp_work",
        "maximum_residual_observable_work",
    ):
        with pytest.raises(ValueError, match="policies"):
            _make_local_residual_observable_input_manifest(
                dataclasses.replace(sealed, **{name: 0})
            )
    for name in (
        "maximum_log_range_reductions",
        "maximum_tv_exp_range_reductions",
    ):
        with pytest.raises(ValueError, match="policies"):
            _make_local_residual_observable_input_manifest(
                dataclasses.replace(sealed, **{name: -1})
            )
    for name in (
        "maximum_detector_rational_bits",
        "maximum_residual_observable_rational_bits",
    ):
        for invalid in (1, 1_048_577):
            with pytest.raises(ValueError, match="policies"):
                _make_local_residual_observable_input_manifest(
                    dataclasses.replace(sealed, **{name: invalid})
                )
    for values in (
        {"log_precision_bits": 256},
        {"tv_exp_precision_bits": 256},
    ):
        with pytest.raises(ValueError, match="policies"):
            _make_local_residual_observable_input_manifest(
                dataclasses.replace(sealed, **values)
            )
    for name in (
        "maximum_detector_work",
        "maximum_detector_rational_bits",
        "log_precision_bits",
        "maximum_log_terms",
        "maximum_log_work",
        "maximum_log_range_reductions",
        "tv_exp_precision_bits",
        "maximum_tv_exp_terms",
        "maximum_tv_exp_work",
        "maximum_tv_exp_range_reductions",
        "maximum_residual_observable_work",
        "maximum_residual_observable_rational_bits",
    ):
        with pytest.raises(ValueError, match="policies"):
            _make_local_residual_observable_input_manifest(
                dataclasses.replace(sealed, **{name: 1 << 63})
            )


def test_residual_observable_certificate_schema_is_layered_and_has_no_gradient() -> (  # noqa: E501
    None
):
    """Require every mean, law, NLL, helper, scope, and provenance layer.

    :see: :class:`ptyrodactyl.types.GalerkinLocalResidualObservableCertificate`
    """
    names = _fields(GalerkinLocalResidualObservableCertificate)
    required = {
        "parent_likelihood",
        "input_manifest",
        "state_evidence_available",
        "mean_evidence_available",
        "law_evidence_available",
        "full_law_evidence_available",
        "fitted_law_evidence_available",
        "direct_nll_evidence_available",
        "score_nll_evidence_available",
        "selected_nll_evidence_available",
        "failure_mask",
        "admitted_pre_gain_mean_hull_intervals",
        "channel_mean_error_bound_intervals",
        "full_mean_l1_error_bound_interval",
        "fitted_mean_l1_error_bound_interval",
        "full_linear_tv_bound_interval",
        "fitted_linear_tv_bound_interval",
        "full_exponential_tv_bound_interval",
        "fitted_exponential_tv_bound_interval",
        "full_selected_tv_bound_interval",
        "fitted_selected_tv_bound_interval",
        "production_fitted_total_nll_interval",
        "direct_nll_error_bound_interval",
        "score_lipschitz_factor_intervals",
        "score_rounding_error_intervals",
        "score_term_error_intervals",
        "score_nll_error_bound_interval",
        "selected_nll_error_bound_interval",
        "saturated_predecessor_mass_upper_intervals",
        "saturated_tail_probability_floor_intervals",
        "full_tv_exp_transcript",
        "fitted_tv_exp_transcript",
        "full_tv_exp_failure",
        "fitted_tv_exp_failure",
        "saturated_probability_transcripts",
        "saturated_probability_failures",
        "work_transcript",
        "strongest_layer",
        "selected_nll_route",
        "full_law_scope",
        "fitted_law_scope",
        "mean_scope",
        "law_scope",
        "nll_scope",
        "resource_scope",
        "no_scientific_claim_scope",
        "parent_detector_certificate_digest",
        "parent_detector_input_manifest_digest",
        "parent_likelihood_certificate_digest",
        "input_manifest_digest",
        "observable_identity_digest",
        "observable_evidence_digest",
        "certificate_digest",
    }
    assert names == required
    assert not any(
        token in name
        for name in names
        for token in ("gradient", "jacobian", "shot_noise", "detectability")
    )


def test_residual_observable_work_and_helper_evidence_bind_partial_stages() -> (  # noqa: E501
    None
):
    """Authenticate helper lanes and the exact staged RM-S5 resource DAG.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalResidualObservableHelperFailureEvidence`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalResidualObservableWorkTranscript`
    """
    assert _fields(GalerkinLocalResidualObservableHelperFailureEvidence) == {
        "call",
        "channel_index",
        "entire_failure",
        "poisson_failure",
        "nested_kernel",
        "nested_failure",
        "prior_exp_transcripts",
        "prior_log_transcripts",
        "local_exact_work_count_exact",
        "nested_exact_work_count_exact",
        "nested_attempted_exact_work_count_exact",
        "planned_exact_work_count_exact",
        "attempted_exact_work_count_exact",
        "failure_digest",
    }
    assert _fields(GalerkinLocalResidualObservableWorkTranscript) == {
        "algorithm",
        "maximum_work",
        "maximum_rational_bits",
        "channel_count",
        "fitted_channel_count",
        "zero_observation_positive_ceiling_count",
        "interior_observation_count",
        "saturated_positive_ceiling_count",
        "mean_exact_work_count",
        "law_exact_work_count",
        "direct_nll_exact_work_count",
        "score_nll_exact_work_count",
        "exact_work_count",
        "rational_peak_bits",
        "nested_parent_work_count_exact",
        "nested_helper_work_count_exact",
        "planned_mean_exact_work_count_exact",
        "planned_law_exact_work_count_exact",
        "planned_direct_nll_exact_work_count_exact",
        "planned_score_nll_exact_work_count_exact",
        "attempted_exact_work_count_exact",
        "completed_layer",
        "completed_successfully",
        "failure",
        "preflight_failed",
        "count_overflow",
        "nested_parent_work_scope",
    }

    exponential = _make_local_residual_observable_helper_failure(
        _make_local_residual_observable_helper_failure_candidate(
            call=(
                GalerkinLocalResidualObservableHelperCall.FULL_TV_EXPONENTIAL
            ),
            channel_index=None,
            entire_failure=EntireEnclosureFailure.WORK_BUDGET_EXCEEDED,
            poisson_failure=None,
            nested_kernel=None,
            nested_failure=None,
            prior_exp_transcripts=(),
            prior_log_transcripts=(),
            local_exact_work_count_exact="3",
            nested_exact_work_count_exact=None,
            nested_attempted_exact_work_count_exact=None,
            planned_exact_work_count_exact="5",
            attempted_exact_work_count_exact="5",
            failure_digest="0" * 64,
        )
    )
    assert (
        _validate_local_residual_observable_helper_failure(exponential)
        is exponential
    )
    assert len(exponential.failure_digest) == 64
    with pytest.raises(ValueError, match="digest disagrees"):
        _validate_local_residual_observable_helper_failure(
            dataclasses.replace(
                exponential,
                call=(
                    GalerkinLocalResidualObservableHelperCall.FITTED_TV_EXPONENTIAL
                ),
            )
        )
    with pytest.raises(ValueError, match="lane disagrees"):
        _make_local_residual_observable_helper_failure(
            dataclasses.replace(
                exponential,
                call=(
                    GalerkinLocalResidualObservableHelperCall.SATURATED_SCORE_MASS
                ),
            )
        )

    prefix = EntireWorkTranscript(
        algorithm="exact_fraction_real_exp_v1",
        precision_bits=64,
        maximum_terms=256,
        maximum_work=100,
        maximum_range_reductions=64,
        maximum_rational_bits=256,
        series_terms=1,
        range_reductions=0,
        root_enclosures=0,
        rectangle_products=0,
        reciprocal_steps=0,
        exact_work_count=5,
    )
    log_prefix = dataclasses.replace(
        prefix,
        algorithm="exact_fraction_real_log_atanh_pow2_v1",
        exact_work_count=7,
    )
    poisson = _make_local_residual_observable_helper_failure(
        _make_local_residual_observable_helper_failure_candidate(
            call=(
                GalerkinLocalResidualObservableHelperCall.SATURATED_SCORE_MASS
            ),
            channel_index=2,
            entire_failure=None,
            poisson_failure=(
                CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
            ),
            nested_kernel="exp",
            nested_failure=EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
            prior_exp_transcripts=(prefix,),
            prior_log_transcripts=(),
            local_exact_work_count_exact="3",
            nested_exact_work_count_exact="2",
            nested_attempted_exact_work_count_exact="2",
            planned_exact_work_count_exact="3",
            attempted_exact_work_count_exact="3",
            failure_digest="0" * 64,
        )
    )
    assert poisson.prior_exp_transcripts == (prefix,)
    assert poisson.prior_log_transcripts == ()
    assert poisson.nested_exact_work_count_exact == "2"
    assert poisson.nested_attempted_exact_work_count_exact == "2"
    logarithm = _make_local_residual_observable_helper_failure(
        dataclasses.replace(
            poisson,
            poisson_failure=(
                CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE
            ),
            nested_kernel="log",
            prior_exp_transcripts=(),
            prior_log_transcripts=(log_prefix,),
        )
    )
    assert logarithm.nested_kernel == "log"
    assert logarithm.prior_exp_transcripts == ()
    assert logarithm.prior_log_transcripts == (log_prefix,)
    with pytest.raises(ValueError, match="nested failure is incomplete"):
        _make_local_residual_observable_helper_failure(
            dataclasses.replace(poisson, nested_kernel=None)
        )
    with pytest.raises(ValueError, match="work counts disagree"):
        _make_local_residual_observable_helper_failure(
            dataclasses.replace(
                poisson, nested_attempted_exact_work_count_exact="1"
            )
        )
    with pytest.raises(TypeError, match="tuples"):
        _make_local_residual_observable_helper_failure(
            dataclasses.replace(poisson, prior_exp_transcripts=[prefix])
        )
    for values, message in (
        ({"prior_exp_transcripts": (log_prefix,)}, "exp.*algorithm"),
        ({"prior_log_transcripts": (prefix,)}, "log.*algorithm"),
        ({"prior_log_transcripts": (log_prefix,)}, "lane disagrees"),
    ):
        with pytest.raises(ValueError, match=message):
            _make_local_residual_observable_helper_failure(
                dataclasses.replace(poisson, **values)
            )

    for values in (
        {"nested_kernel": None},
        {"nested_kernel": "exp"},
        {"nested_failure": None},
        {"nested_exact_work_count_exact": None},
        {"prior_exp_transcripts": (prefix,)},
        {"prior_log_transcripts": (prefix,)},
    ):
        with pytest.raises(ValueError):
            _make_local_residual_observable_helper_failure(
                dataclasses.replace(logarithm, **values)
            )
    with pytest.raises(ValueError):
        _make_local_residual_observable_helper_failure(
            dataclasses.replace(
                poisson,
                poisson_failure=(
                    CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE
                ),
            )
        )

    class StringSubclass(str):
        """Expose equality-preserving string identity adversaries."""

    for carrier in (
        dataclasses.replace(
            poisson,
            nested_kernel=StringSubclass("exp"),
        ),
        dataclasses.replace(
            poisson,
            prior_exp_transcripts=(
                dataclasses.replace(
                    prefix,
                    algorithm=StringSubclass("exact_fraction_real_exp_v1"),
                ),
            ),
        ),
        dataclasses.replace(
            logarithm,
            prior_log_transcripts=(
                dataclasses.replace(
                    log_prefix,
                    algorithm=StringSubclass(
                        "exact_fraction_real_log_atanh_pow2_v1"
                    ),
                ),
            ),
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            _make_local_residual_observable_helper_failure(carrier)

    class TranscriptTuple(tuple[EntireWorkTranscript, ...]):
        """Expose an exact-tuple-subclass adversary."""

    for values in (
        {
            "prior_exp_transcripts": cast(
                tuple[EntireWorkTranscript, ...], TranscriptTuple((prefix,))
            )
        },
        {"prior_exp_transcripts": (cast(EntireWorkTranscript, object()),)},
    ):
        with pytest.raises(TypeError):
            _make_local_residual_observable_helper_failure(
                dataclasses.replace(poisson, **values)
            )

    for name in (
        "precision_bits",
        "maximum_terms",
        "maximum_work",
        "maximum_range_reductions",
        "maximum_rational_bits",
        "series_terms",
        "range_reductions",
        "root_enclosures",
        "rectangle_products",
        "reciprocal_steps",
        "exact_work_count",
    ):
        with pytest.raises(TypeError):
            _make_local_residual_observable_helper_failure(
                dataclasses.replace(
                    poisson,
                    prior_exp_transcripts=(
                        dataclasses.replace(prefix, **{name: True}),
                    ),
                )
            )
    for malformed in (
        dataclasses.replace(prefix, precision_bits=True),
        dataclasses.replace(prefix, maximum_rational_bits=1),
        dataclasses.replace(prefix, exact_work_count=101),
        dataclasses.replace(
            prefix,
            precision_bits=256,
            maximum_rational_bits=256,
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            _make_local_residual_observable_helper_failure(
                dataclasses.replace(
                    poisson, prior_exp_transcripts=(malformed,)
                )
            )

    work_values = {
        "algorithm": _RESIDUAL_OBSERVABLE_WORK_ALGORITHM,
        "maximum_work": 1_000,
        "maximum_rational_bits": 256,
        "channel_count": 5,
        "fitted_channel_count": 4,
        "zero_observation_positive_ceiling_count": 1,
        "interior_observation_count": 1,
        "saturated_positive_ceiling_count": 1,
        "mean_exact_work_count": 19,
        "law_exact_work_count": 23,
        "direct_nll_exact_work_count": 29,
        "score_nll_exact_work_count": 50,
        "exact_work_count": 50,
        "rational_peak_bits": 8,
        "nested_parent_work_count_exact": "123",
        "nested_helper_work_count_exact": "456",
        "planned_mean_exact_work_count_exact": "19",
        "planned_law_exact_work_count_exact": "23",
        "planned_direct_nll_exact_work_count_exact": "29",
        "planned_score_nll_exact_work_count_exact": "50",
        "attempted_exact_work_count_exact": "50",
        "completed_layer": GalerkinLocalResidualObservableLayer.POINTWISE_NLL,
        "completed_successfully": True,
        "failure": GalerkinLocalResidualObservableFailure.NONE,
        "preflight_failed": False,
        "count_overflow": False,
        "nested_parent_work_scope": _NESTED_PARENT_WORK_SCOPE,
    }
    success = _make_local_residual_observable_work_transcript(
        _make_local_residual_observable_work_transcript_candidate(
            **work_values
        )
    )
    assert (
        _validate_local_residual_observable_work_transcript(success) is success
    )
    assert (
        success.mean_exact_work_count,
        success.law_exact_work_count,
        success.direct_nll_exact_work_count,
        success.score_nll_exact_work_count,
        success.exact_work_count,
    ) == (19, 23, 29, 50, 50)

    partials = tuple(
        _make_local_residual_observable_work_transcript(
            dataclasses.replace(
                success,
                mean_exact_work_count=counters[0],
                law_exact_work_count=counters[1],
                direct_nll_exact_work_count=counters[2],
                score_nll_exact_work_count=counters[3],
                exact_work_count=counters[3],
                attempted_exact_work_count_exact=str(counters[3]),
                completed_layer=layer,
                completed_successfully=False,
                failure=(
                    GalerkinLocalResidualObservableFailure.ARITHMETIC_RANGE_FAILURE
                ),
            )
        )
        for counters, layer in (
            ((7, 7, 7, 7), GalerkinLocalResidualObservableLayer.STATE),
            ((19, 21, 21, 21), GalerkinLocalResidualObservableLayer.MEAN),
            ((19, 23, 25, 25), GalerkinLocalResidualObservableLayer.LAW),
            (
                (19, 23, 29, 40),
                GalerkinLocalResidualObservableLayer.POINTWISE_NLL,
            ),
        )
    )
    assert tuple(value.exact_work_count for value in partials) == (
        7,
        21,
        25,
        40,
    )
    for partial, wrong_layer in zip(
        partials,
        (
            GalerkinLocalResidualObservableLayer.MEAN,
            GalerkinLocalResidualObservableLayer.LAW,
            GalerkinLocalResidualObservableLayer.POINTWISE_NLL,
            GalerkinLocalResidualObservableLayer.MEAN,
        ),
        strict=True,
    ):
        with pytest.raises(ValueError, match="stage"):
            _make_local_residual_observable_work_transcript(
                dataclasses.replace(partial, completed_layer=wrong_layer)
            )

    budget_preflight = _make_local_residual_observable_work_transcript(
        dataclasses.replace(
            success,
            maximum_work=49,
            mean_exact_work_count=0,
            law_exact_work_count=0,
            direct_nll_exact_work_count=0,
            score_nll_exact_work_count=0,
            exact_work_count=0,
            rational_peak_bits=0,
            attempted_exact_work_count_exact="50",
            completed_layer=GalerkinLocalResidualObservableLayer.STATE,
            completed_successfully=False,
            failure=(
                GalerkinLocalResidualObservableFailure.EXACT_WORK_BUDGET_EXCEEDED
            ),
            preflight_failed=True,
        )
    )
    assert budget_preflight.preflight_failed
    assert budget_preflight.exact_work_count == 0
    for values in (
        {"attempted_exact_work_count_exact": "0"},
        {"rational_peak_bits": 1},
        {"preflight_failed": False},
    ):
        with pytest.raises(ValueError):
            _make_local_residual_observable_work_transcript(
                dataclasses.replace(budget_preflight, **values)
            )

    channel_count = (1 << 63) // 3 + 1
    planned_mean = 3 * channel_count
    planned_law = planned_mean + 4
    planned_score = planned_mean + 6
    overflow_preflight = _make_local_residual_observable_work_transcript(
        _make_local_residual_observable_work_transcript_candidate(
            **{
                **work_values,
                "maximum_work": 1_000,
                "channel_count": channel_count,
                "fitted_channel_count": 0,
                "zero_observation_positive_ceiling_count": 0,
                "interior_observation_count": 0,
                "saturated_positive_ceiling_count": 0,
                "mean_exact_work_count": 0,
                "law_exact_work_count": 0,
                "direct_nll_exact_work_count": 0,
                "score_nll_exact_work_count": 0,
                "exact_work_count": 0,
                "rational_peak_bits": 0,
                "planned_mean_exact_work_count_exact": str(planned_mean),
                "planned_law_exact_work_count_exact": str(planned_law),
                "planned_direct_nll_exact_work_count_exact": str(
                    planned_score
                ),
                "planned_score_nll_exact_work_count_exact": str(planned_score),
                "attempted_exact_work_count_exact": str(planned_score),
                "completed_layer": GalerkinLocalResidualObservableLayer.STATE,
                "completed_successfully": False,
                "failure": (
                    GalerkinLocalResidualObservableFailure.EXACT_WORK_COUNT_OVERFLOW
                ),
                "preflight_failed": True,
                "count_overflow": True,
            }
        )
    )
    assert overflow_preflight.count_overflow
    assert int(overflow_preflight.planned_score_nll_exact_work_count_exact) > (
        (1 << 63) - 1
    )
    with pytest.raises(ValueError, match="inconsistent"):
        _make_local_residual_observable_work_transcript(
            dataclasses.replace(
                success,
                planned_score_nll_exact_work_count_exact="49",
                attempted_exact_work_count_exact="49",
                score_nll_exact_work_count=49,
                exact_work_count=49,
            )
        )
    with pytest.raises(ValueError, match="preflight"):
        _make_local_residual_observable_work_transcript(
            dataclasses.replace(
                success,
                mean_exact_work_count=0,
                law_exact_work_count=0,
                direct_nll_exact_work_count=0,
                score_nll_exact_work_count=0,
                exact_work_count=0,
                rational_peak_bits=0,
                attempted_exact_work_count_exact="50",
                completed_layer=GalerkinLocalResidualObservableLayer.STATE,
                completed_successfully=False,
                failure=(
                    GalerkinLocalResidualObservableFailure.EXACT_WORK_BUDGET_EXCEEDED
                ),
                preflight_failed=True,
            )
        )
    with pytest.raises(ValueError, match="rational peak"):
        _make_local_residual_observable_work_transcript(
            dataclasses.replace(
                partials[1],
                failure=GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT,
            )
        )

    hard_bits = 1_048_576
    maximum_post_issue_peak = 3 * hard_bits + 2
    rational_stop = _make_local_residual_observable_work_transcript(
        dataclasses.replace(
            partials[1],
            maximum_rational_bits=hard_bits,
            rational_peak_bits=maximum_post_issue_peak,
            failure=GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT,
        )
    )
    assert rational_stop.rational_peak_bits == maximum_post_issue_peak
    with pytest.raises(ValueError):
        _make_local_residual_observable_work_transcript(
            dataclasses.replace(
                rational_stop,
                rational_peak_bits=maximum_post_issue_peak + 1,
            )
        )

    nonfatal_exponential = _make_local_residual_observable_work_transcript(
        dataclasses.replace(
            success,
            completed_successfully=False,
            failure=(
                GalerkinLocalResidualObservableFailure.EXPONENTIAL_ENCLOSURE_FAILURE
            ),
        )
    )
    assert nonfatal_exponential.exact_work_count == 50
    score_helper_stop = _make_local_residual_observable_work_transcript(
        dataclasses.replace(
            success,
            score_nll_exact_work_count=29,
            exact_work_count=29,
            attempted_exact_work_count_exact="50",
            completed_successfully=False,
            failure=(
                GalerkinLocalResidualObservableFailure.POISSON_ENCLOSURE_FAILURE
                | GalerkinLocalResidualObservableFailure.NESTED_HELPER_FAILURE
            ),
        )
    )
    assert score_helper_stop.direct_nll_exact_work_count == 29

    full_parent_failure = dataclasses.replace(
        success,
        completed_successfully=False,
        failure=(
            GalerkinLocalResidualObservableFailure.PARENT_STATE_NONCERTIFICATE
        ),
    )
    zero_exponential = dataclasses.replace(
        success,
        mean_exact_work_count=0,
        law_exact_work_count=0,
        direct_nll_exact_work_count=0,
        score_nll_exact_work_count=0,
        exact_work_count=0,
        rational_peak_bits=0,
        attempted_exact_work_count_exact="0",
        completed_layer=GalerkinLocalResidualObservableLayer.UNAVAILABLE,
        completed_successfully=False,
        failure=(
            GalerkinLocalResidualObservableFailure.EXPONENTIAL_ENCLOSURE_FAILURE
        ),
    )
    for invalid in (full_parent_failure, zero_exponential):
        with pytest.raises(ValueError, match="cause|precedes"):
            _make_local_residual_observable_work_transcript(invalid)

    class LayerAlias(StrEnum):
        POINTWISE_NLL = "pointwise_nll"

    class FailureAlias(IntFlag):
        NONE = 0

    class StringSubclass(str):
        """Expose equality-preserving work-scope identity adversaries."""

    for values in (
        {
            "completed_layer": cast(
                GalerkinLocalResidualObservableLayer,
                LayerAlias.POINTWISE_NLL,
            )
        },
        {
            "failure": cast(
                GalerkinLocalResidualObservableFailure,
                FailureAlias.NONE,
            )
        },
        {
            "nested_parent_work_scope": StringSubclass(
                _NESTED_PARENT_WORK_SCOPE
            )
        },
    ):
        with pytest.raises(TypeError):
            _make_local_residual_observable_work_transcript(
                dataclasses.replace(success, **values)
            )
    with pytest.raises(ValueError):
        _make_local_residual_observable_work_transcript(
            dataclasses.replace(
                success,
                completed_successfully=False,
                failure=GalerkinLocalResidualObservableFailure(1 << 20),
            )
        )


__all__: list[str] = []
