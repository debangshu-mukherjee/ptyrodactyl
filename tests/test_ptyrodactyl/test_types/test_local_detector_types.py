r"""Tests for the owned local positive-port and detector evidence carriers."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from ptyrodactyl._tools import (
    CensoredPoissonEnclosureFailure,
    EntireWorkTranscript,
)
from ptyrodactyl.galerkin.detector import (
    create_local_censored_poisson_detector_input_manifest,
    create_local_passive_pixel_input_manifest,
)
from ptyrodactyl.types.local_detector_types import (
    GalerkinLocalCensoredPoissonDetector,
    GalerkinLocalCensoredPoissonDetectorInputManifest,
    GalerkinLocalCensoredPoissonLikelihood,
    GalerkinLocalDetectorCoordinateConvention,
    GalerkinLocalDetectorFailure,
    GalerkinLocalDetectorHelperCall,
    GalerkinLocalDetectorHelperFailureEvidence,
    GalerkinLocalDetectorLikelihoodStage,
    GalerkinLocalDetectorProductionStage,
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalDetectorRealProductionTrace,
    GalerkinLocalDetectorWorkTranscript,
    GalerkinLocalPassivePixelForms,
    GalerkinLocalPassivePixelInputManifest,
    GalerkinLocalPositivePortBranchDisposition,
    GalerkinLocalPositivePortCertificate,
    GalerkinLocalPositivePortRoute,
    _make_local_censored_poisson_detector_input_manifest,
    _make_local_detector_helper_failure_evidence,
    _make_local_detector_rational_interval,
    _make_local_detector_real_production_trace,
    _make_local_detector_work_transcript,
    _make_local_passive_pixel_input_manifest,
    _validate_local_detector_helper_failure_evidence,
    _validate_local_detector_real_production_trace,
    _validate_local_passive_pixel_input_manifest,
)
from ptyrodactyl.types.local_vacuum_terminal_types import (
    GalerkinLocalVacuumTerminalDisposition,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def _interval(
    lower: int | Fraction, upper: int | Fraction | None = None
) -> GalerkinLocalDetectorRationalInterval:
    """Return one exact detector interval."""
    low = Fraction(lower)
    high = low if upper is None else Fraction(upper)
    return _make_local_detector_rational_interval(low, high)


def _pixel_manifest(
    *,
    quadrature: GalerkinLocalDetectorRationalInterval | None = None,
    maximum_detector_rational_bits: int = 64,
) -> GalerkinLocalPassivePixelInputManifest:
    """Build one small parent-free primitive pixel manifest."""
    return create_local_passive_pixel_input_manifest(
        maximum_state_error=np.asarray(0.125, dtype=np.float64),
        node_to_pixel=np.asarray([0], dtype=np.int64),
        quadrature_weight_intervals=(
            _interval(1) if quadrature is None else quadrature,
        ),
        quadrature_weight_points=np.asarray([1.0], dtype=np.float64),
        aperture_efficiency_intervals=(_interval(Fraction(3, 4)),),
        aperture_efficiency_points=np.asarray([0.75], dtype=np.float64),
        route=GalerkinLocalPositivePortRoute.OUTGOING_RADIATION,
        disposition=(
            GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_SLAB
        ),
        coordinate_convention=(
            GalerkinLocalDetectorCoordinateConvention.NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES
        ),
        pixel_count=1,
        maximum_stability_direct_pairs=3,
        maximum_gram_pairs=1,
        maximum_terminal_direct_pairs=3,
        maximum_branch_direct_terms=3,
        maximum_cut_direct_pairs=3,
        maximum_root_work=100,
        precision_bits=96,
        maximum_terms=64,
        maximum_entire_work=100,
        maximum_range_reductions=8,
        maximum_interval_work=100,
        maximum_l8_rational_bits=64,
        maximum_detector_work=1_000,
        maximum_detector_rational_bits=maximum_detector_rational_bits,
    )


def test_local_detector_enums_are_complete_explicit_and_disjoint() -> None:
    """Freeze every L9 route, helper, production stage, and failure bit.

    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorCoordinateConvention`
    :see: :class:`ptyrodactyl.types.GalerkinLocalPositivePortRoute`
    :see: :class:`ptyrodactyl.types.GalerkinLocalPositivePortBranchDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorLikelihoodStage`
    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorFailure`
    """
    assert {
        item.name: item.value
        for item in GalerkinLocalDetectorCoordinateConvention
    } == {
        "ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES": (
            "angular_wavenumber_amplitude_in_angular_coordinates"
        ),
        "ANGULAR_WAVENUMBER_AMPLITUDE_IN_CYCLIC_COORDINATES": (
            "angular_wavenumber_amplitude_in_cyclic_coordinates"
        ),
        "NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES": (
            "native_cyclic_amplitude_in_cyclic_coordinates"
        ),
        "ANGULAR_WAVENUMBER_AMPLITUDE_IN_SOLID_ANGLE": (
            "angular_wavenumber_amplitude_in_solid_angle"
        ),
    }
    assert {
        item.name: item.value for item in GalerkinLocalPositivePortRoute
    } == {
        "PROJECTED_OUTWARD_PROPAGATING": "projected_outward_propagating",
        "OUTGOING_RADIATION": "outgoing_radiation",
    }
    assert {
        item.name: item.value
        for item in GalerkinLocalPositivePortBranchDisposition
    } == {
        "PROPAGATING_OUTWARD_RETAINED_INWARD_EXACT_ZERO": (
            "propagating_outward_retained_inward_exact_zero"
        ),
        "PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_PROVABLY_NONZERO": (
            "propagating_outward_retained_inward_projected_provably_nonzero"
        ),
        "PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_UNRESOLVED": (
            "propagating_outward_retained_inward_projected_unresolved"
        ),
        "EVANESCENT_DECAYING_ZERO_WEIGHT_GROWING_EXACT_ZERO": (
            "evanescent_decaying_zero_weight_growing_exact_zero"
        ),
        "EVANESCENT_GROWING_REJECTED": "evanescent_growing_rejected",
        "GRAZING_CONSTANT_ZERO_WEIGHT_DERIVATIVE_EXACT_ZERO": (
            "grazing_constant_zero_weight_derivative_exact_zero"
        ),
        "GRAZING_DERIVATIVE_REJECTED": "grazing_derivative_rejected",
        "ROOT_UNCLASSIFIED": "root_unclassified",
    }
    assert {
        item.name: item.value for item in GalerkinLocalDetectorLikelihoodStage
    } == {"PRE_GAIN_CENSORED_COUNTS": "pre_gain_censored_counts"}
    assert {
        item.name: item.value for item in GalerkinLocalDetectorHelperCall
    } == {
        "EXACT_STATE_CENSORED_MEAN": "exact_state_censored_mean",
        "PRODUCTION_CENSORED_MEAN": "production_censored_mean",
        "PRODUCTION_PROBABILITY": "production_probability",
        "ADMITTED_HULL_PROBABILITY": "admitted_hull_probability",
        "PRODUCTION_NLL": "production_nll",
        "ADMITTED_HULL_NLL": "admitted_hull_nll",
    }
    assert [item.value for item in GalerkinLocalDetectorProductionStage] == [
        "l8_role_zero_amplitude",
        "positive_port_amplitude",
        "coordinate_factor",
        "pixel_form_diagonal",
        "mode_production_quadratic",
        "mode_pixel_fraction",
        "ensemble_weight",
        "incident_dose",
        "ideal_arrival_mean",
        "pre_gain_response_mean",
        "censored_count_mean",
        "censored_probability",
        "censored_nll",
        "post_censor_digitized_mean",
    ]
    assert {item.name: int(item) for item in GalerkinLocalDetectorFailure} == {
        "VACUUM_TERMINAL_NONCERTIFICATE": 1 << 0,
        "ROOT_UNCLASSIFIED": 1 << 1,
        "PROPAGATING_INWARD_NOT_EXACT_ZERO": 1 << 2,
        "EVANESCENT_GROWING_NOT_EXACT_ZERO": 1 << 3,
        "GRAZING_DERIVATIVE_NOT_EXACT_ZERO": 1 << 4,
        "INCIDENT_FLUX_NONPOSITIVE": 1 << 5,
        "PIXEL_FORM_NONPOSITIVE": 1 << 7,
        "PIXEL_FORM_NONPASSIVE": 1 << 8,
        "PRODUCTION_POINT_HULL_FAILURE": 1 << 10,
        "ENSEMBLE_WEIGHT_INVALID": 1 << 11,
        "DOSE_INVALID": 1 << 12,
        "RESPONSE_NONPOSITIVE": 1 << 13,
        "RESPONSE_NOT_SUBSTOCHASTIC": 1 << 14,
        "CALIBRATION_INVALID": 1 << 15,
        "COUNT_DOMAIN_INVALID": 1 << 16,
        "POISSON_ENCLOSURE_FAILURE": 1 << 17,
        "NESTED_HELPER_FAILURE": 1 << 18,
        "NLL_UNAVAILABLE": 1 << 19,
        "EXACT_WORK_BUDGET_EXCEEDED": 1 << 21,
        "EXACT_WORK_COUNT_OVERFLOW": 1 << 22,
        "RATIONAL_SIZE_LIMIT": 1 << 23,
        "ARITHMETIC_RANGE_FAILURE": 1 << 24,
    }
    assert not hasattr(
        GalerkinLocalDetectorFailure, "RM_S5_GRADIENT_UNAVAILABLE"
    )
    assert not hasattr(GalerkinLocalDetectorFailure, "PARENT_REPLAY_FAILURE")


def test_local_detector_carrier_schema_exposes_full_staged_evidence_no_gradient() -> (  # noqa: E501
    None
):
    """Freeze manifests, traces, LVT.56, state bindings, and NLL layers.

    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorWorkTranscript`
    :see: :class:`ptyrodactyl.types.GalerkinLocalPositivePortCertificate`
    :see: :class:`ptyrodactyl.types.GalerkinLocalPassivePixelForms`
    :see: :class:`ptyrodactyl.types.GalerkinLocalCensoredPoissonDetector`
    :see: :class:`ptyrodactyl.types.GalerkinLocalCensoredPoissonLikelihood`
    """
    assert _fields(GalerkinLocalDetectorRealProductionTrace) == {
        "point",
        "point_to_raw_absolute_error_upper_bounds",
        "certified_hull_lower_bounds",
        "certified_hull_upper_bounds",
        "raw_intervals",
        "exact_point_intervals",
        "stage",
        "quantity",
        "logical_shape",
        "point_dtype",
        "point_bytes_digest",
        "raw_interval_digest",
        "trace_digest",
    }
    assert {
        "maximum_l8_rational_bits",
        "maximum_detector_rational_bits",
        "quadrature_weight_intervals",
        "aperture_efficiency_intervals",
        "manifest_digest",
    } <= _fields(GalerkinLocalPassivePixelInputManifest)
    assert {
        "pixel_inputs",
        "incident_electron_count_interval",
        "maximum_detector_rational_bits",
        "maximum_poisson_rational_bits",
        "fit_mask",
        "manifest_digest",
    } <= _fields(GalerkinLocalCensoredPoissonDetectorInputManifest)
    port = _fields(GalerkinLocalPositivePortCertificate)
    assert {
        "terminal_certificate",
        "production_amplitudes",
        "exact_state_total_amplitude_error_bounds",
        "production_root_realizations",
        "production_root_error_upper_bounds",
        "exact_root_intervals",
        "retained_propagating_mask",
        "zero_weight_mask",
        "branch_dispositions",
        "production_traces",
        "positive_port_eligible",
        "outgoing_radiation_eligible",
        "state_identity_digest",
        "parent_terminal_identity_digest",
        "parent_terminal_evidence_digest",
        "port_identity_digest",
        "port_evidence_digest",
        "certificate_digest",
    } <= port
    pixels = _fields(GalerkinLocalPassivePixelForms)
    assert {
        "current_weight_intervals",
        "amplitude_scale_interval",
        "coordinate_jacobian_intervals",
        "outward_form_diagonal_intervals",
        "pixel_form_diagonal_intervals",
        "outward_minus_pixel_form_diagonal_intervals",
        "production_outward_quadratic_interval",
        "outward_form_norm_upper_interval",
        "outward_production_realization_error_upper_interval",
        "outward_state_radius_incremental_error_upper_interval",
        "outward_combined_exact_state_error_upper_interval",
        "production_to_exact_x_amplitude_error_interval",
        "state_radius_amplitude_error_interval",
        "exact_state_amplitude_error_interval",
        "production_realization_error_upper_intervals",
        "state_radius_incremental_error_upper_intervals",
        "combined_exact_state_error_upper_intervals",
        "exact_state_pixel_flux_intervals",
        "production_traces",
        "work_transcript",
        "input_manifest_digest",
        "pixel_model_identity_digest",
        "pixel_model_evidence_digest",
    } <= pixels
    detector = _fields(GalerkinLocalCensoredPoissonDetector)
    assert {
        "pixel_forms",
        "mode_target_digests",
        "mode_source_digests",
        "mode_state_identity_digests",
        "mode_state_radius_intervals",
        "mode_state_radius_provenance_digests",
        "mode_state_binding_digest",
        "mode_exact_state_pixel_flux_intervals",
        "mode_production_quadratic_intervals",
        "mode_pixel_fraction_intervals",
        "ideal_arrival_mean_intervals",
        "production_pre_gain_mean_point_intervals",
        "exact_state_pre_gain_mean_intervals",
        "censored_mean_intervals",
        "expected_digitized_mean_intervals",
        "censored_mean_transcripts",
        "censored_mean_failures",
        "production_censored_mean_transcripts",
        "production_censored_mean_failures",
        "production_traces",
        "work_transcript",
        "likelihood_stage",
        "likelihood_law_eligible",
    } <= detector
    likelihood = _fields(GalerkinLocalCensoredPoissonLikelihood)
    assert {
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
        "production_traces",
        "work_transcript",
        "no_derivative_scope",
    } <= likelihood
    all_fields = port | pixels | detector | likelihood
    forbidden = {
        "source_certificates",
        "gradient_eligible",
        "gradient_scope",
        "epsilon_probability_floor",
        "experimental_validity_eligible",
        "post_gain_likelihood",
        "detector_gradient_certificate",
        "legacy_terminal",
        "terminal_model_error_upper_intervals",
    }
    assert forbidden.isdisjoint(all_fields)


def test_exact_interval_and_production_trace_are_owned_and_digest_bound() -> (
    None
):
    """Audit a rounded point outside its raw singleton without hiding it.

    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorProductionStage`
    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorRationalInterval`
    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorRealProductionTrace`
    """
    raw = _interval(Fraction(1, 10))
    trace = _make_local_detector_real_production_trace(
        (raw,),
        np.asarray([0.1], dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.INCIDENT_DOSE,
        quantity="dose",
        logical_shape=(),
    )
    assert isinstance(trace, GalerkinLocalDetectorRealProductionTrace)
    assert raw.lower == raw.upper == Fraction(1, 10)
    assert trace.exact_point_intervals[0].lower == Fraction.from_float(0.1)
    assert float(trace.point_to_raw_absolute_error_upper_bounds[0]) > 0.0
    assert float(trace.certified_hull_lower_bounds[0]) <= 0.1
    assert float(trace.certified_hull_upper_bounds[0]) >= 0.1
    assert trace.point_dtype == "float64"
    assert len(trace.point_bytes_digest) == len(trace.trace_digest) == 64
    with pytest.raises(TypeError, match="float64"):
        _make_local_detector_real_production_trace(
            (raw,),
            np.asarray([0.1], dtype=np.float32),
            stage=GalerkinLocalDetectorProductionStage.INCIDENT_DOSE,
            quantity="dose",
            logical_shape=(),
        )
    with pytest.raises(ValueError, match="digest disagrees"):
        _validate_local_detector_real_production_trace(
            dataclasses.replace(trace, trace_digest="0" * 64)
        )
    with pytest.raises(ValueError, match="ordered"):
        _make_local_detector_rational_interval(Fraction(2), Fraction(1))
    with pytest.raises(TypeError, match="Fraction"):
        _make_local_detector_rational_interval(0, Fraction(1))  # type: ignore[arg-type]


def test_input_manifests_enforce_dtype_hard_caps_and_defer_child_caps() -> (
    None
):
    """Reject malformed storage while retaining over-child-policy evidence.

    :see: :class:`ptyrodactyl.types.GalerkinLocalPassivePixelInputManifest`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalCensoredPoissonDetectorInputManifest`
    """
    with pytest.raises(TypeError, match="node_to_pixel.*int64"):
        create_local_passive_pixel_input_manifest(
            maximum_state_error=np.asarray(0.125, dtype=np.float64),
            node_to_pixel=np.asarray([0], dtype=np.int32),
            quadrature_weight_intervals=(_interval(1),),
            quadrature_weight_points=np.asarray([1.0], dtype=np.float64),
            aperture_efficiency_intervals=(_interval(1),),
            aperture_efficiency_points=np.asarray([1.0], dtype=np.float64),
            route=GalerkinLocalPositivePortRoute.OUTGOING_RADIATION,
            disposition=(
                GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_SLAB
            ),
            coordinate_convention=(
                GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES
            ),
            pixel_count=1,
        )
    denominator = (1 << 20) + 1
    over_child = _interval(Fraction(1, denominator))
    pixel = _pixel_manifest(
        quadrature=over_child, maximum_detector_rational_bits=16
    )
    assert pixel.quadrature_weight_intervals[
        0
    ].lower_denominator.bit_length() > (pixel.maximum_detector_rational_bits)
    assert len(pixel.manifest_digest) == 64
    with pytest.raises(ValueError, match="hard bit cap"):
        _validate_local_passive_pixel_input_manifest(
            dataclasses.replace(
                pixel,
                quadrature_weight_intervals=(
                    GalerkinLocalDetectorRationalInterval(
                        lower_numerator=1,
                        lower_denominator=1 << 1_048_576,
                        upper_numerator=1,
                        upper_denominator=1 << 1_048_576,
                    ),
                ),
            )
        )

    dose = _interval(Fraction(1, denominator))
    detector_manifest = create_local_censored_poisson_detector_input_manifest(
        pixel_inputs=(pixel,),
        ensemble_weight_numerators=(1,),
        ensemble_weight_denominators=(1,),
        incident_electron_count_interval=dose,
        incident_electron_count_point=np.asarray(1.0, dtype=np.float64),
        response_matrix=np.asarray([[1.0]], dtype=np.float64),
        pre_gain_background=np.asarray([0.0], dtype=np.float64),
        deterministic_gain=np.asarray([2.0], dtype=np.float64),
        electronic_offset=np.asarray([-1.0], dtype=np.float64),
        count_ceilings=np.asarray([3], dtype=np.int64),
        fit_mask=np.asarray([True], dtype=np.bool_),
        calibration_provenance="unit-test calibration",
        maximum_detector_rational_bits=16,
        maximum_poisson_rational_bits=32,
    )
    assert detector_manifest.incident_electron_count_interval is dose
    assert dose.lower_denominator.bit_length() > (
        detector_manifest.maximum_detector_rational_bits
    )
    assert detector_manifest.pixel_inputs == (pixel,)
    malformed_intervals = (
        (
            GalerkinLocalDetectorRationalInterval(
                lower_numerator=2,
                lower_denominator=denominator,
                upper_numerator=1,
                upper_denominator=denominator,
            ),
            ValueError,
            "ordered",
        ),
        (
            GalerkinLocalDetectorRationalInterval(
                lower_numerator=2,
                lower_denominator=2 * denominator,
                upper_numerator=2,
                upper_denominator=2 * denominator,
            ),
            ValueError,
            "reduced",
        ),
        (
            GalerkinLocalDetectorRationalInterval(
                lower_numerator=0,
                lower_denominator=denominator,
                upper_numerator=0,
                upper_denominator=denominator,
            ),
            ValueError,
            "reduced",
        ),
        (
            GalerkinLocalDetectorRationalInterval(
                lower_numerator=True,
                lower_denominator=denominator,
                upper_numerator=1,
                upper_denominator=denominator,
            ),
            TypeError,
            "Python ints",
        ),
    )
    for malformed, error, message in malformed_intervals:
        with pytest.raises(error, match=message):
            _make_local_passive_pixel_input_manifest(
                dataclasses.replace(
                    pixel, quadrature_weight_intervals=(malformed,)
                )
            )
        with pytest.raises(error, match=message):
            _make_local_censored_poisson_detector_input_manifest(
                dataclasses.replace(
                    detector_manifest,
                    incident_electron_count_interval=malformed,
                )
            )
    negative_pixel = _pixel_manifest(quadrature=_interval(-1))
    negative_detector = create_local_censored_poisson_detector_input_manifest(
        pixel_inputs=(negative_pixel,),
        ensemble_weight_numerators=(1,),
        ensemble_weight_denominators=(1,),
        incident_electron_count_interval=_interval(-1),
        incident_electron_count_point=np.asarray(-1.0, dtype=np.float64),
        response_matrix=np.asarray([[1.0]], dtype=np.float64),
        pre_gain_background=np.asarray([0.0], dtype=np.float64),
        deterministic_gain=np.asarray([1.0], dtype=np.float64),
        electronic_offset=np.asarray([0.0], dtype=np.float64),
        count_ceilings=np.asarray([3], dtype=np.int64),
        fit_mask=np.asarray([True], dtype=np.bool_),
        calibration_provenance="negative scientific-stop control",
    )
    assert len(negative_pixel.manifest_digest) == 64
    assert len(negative_detector.manifest_digest) == 64
    with pytest.raises(TypeError, match="response_matrix.*float64"):
        create_local_censored_poisson_detector_input_manifest(
            pixel_inputs=(pixel,),
            ensemble_weight_numerators=(1,),
            ensemble_weight_denominators=(1,),
            incident_electron_count_interval=_interval(1),
            incident_electron_count_point=np.asarray(1.0, dtype=np.float64),
            response_matrix=np.asarray([[1.0]], dtype=np.float32),
            pre_gain_background=np.asarray([0.0], dtype=np.float64),
            deterministic_gain=np.asarray([1.0], dtype=np.float64),
            electronic_offset=np.asarray([0.0], dtype=np.float64),
            count_ceilings=np.asarray([3], dtype=np.int64),
            fit_mask=np.asarray([True], dtype=np.bool_),
            calibration_provenance="unit-test calibration",
        )


def test_work_factory_authenticates_success_preflight_and_partial_stops() -> (
    None
):
    """Preserve exact nested counts and three disjoint transcript outcomes."""
    success = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=100,
        maximum_rational_bits=64,
        coordinate_factor_count=3,
        pixel_product_count=4,
        mode_quadratic_count=5,
        ensemble_product_count=2,
        response_product_count=6,
        production_trace_count=7,
        hull_endpoint_count=14,
        nested_production_trace_count=11,
        nested_hull_endpoint_count=22,
        exact_work_count=42,
        rational_peak_bits=17,
        nested_parent_work_count_exact="123",
        nested_helper_work_count_exact="456",
    )
    assert success.completed_successfully
    assert success.planned_exact_work_count_exact == "42"
    assert success.attempted_exact_work_count_exact == "42"
    assert success.nested_parent_work_count_exact == "123"
    assert success.nested_helper_work_count_exact == "456"

    preflight = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=10,
        maximum_rational_bits=64,
        coordinate_factor_count=0,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        exact_work_count=0,
        rational_peak_bits=0,
        planned_exact_work_count_exact="11",
        attempted_exact_work_count_exact="11",
        completed_successfully=False,
        arithmetic_failure=(
            GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED
        ),
        preflight_failed=True,
    )
    assert preflight.preflight_failed and not preflight.completed_successfully
    assert preflight.exact_work_count == 0

    partial = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=100,
        maximum_rational_bits=8,
        coordinate_factor_count=1,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        exact_work_count=3,
        rational_peak_bits=9,
        planned_exact_work_count_exact="10",
        attempted_exact_work_count_exact="4",
        completed_successfully=False,
        arithmetic_failure=GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT,
    )
    assert not partial.preflight_failed
    assert partial.rational_peak_bits > partial.maximum_rational_bits
    with pytest.raises(ValueError, match="algorithm"):
        _make_local_detector_work_transcript(  # pragma: no cover
            algorithm="forged",
            maximum_work=100,
            maximum_rational_bits=64,
            coordinate_factor_count=0,
            pixel_product_count=0,
            mode_quadratic_count=0,
            ensemble_product_count=0,
            response_product_count=0,
            exact_work_count=0,
            rational_peak_bits=0,
        )


def test_helper_failure_evidence_is_owner_call_and_channel_bound() -> None:
    """Authenticate partial nested-helper outcomes without generic sentinels.

    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorHelperCall`
    :see: :class:`ptyrodactyl.types.GalerkinLocalDetectorHelperFailureEvidence`
    """
    exp_prefix = EntireWorkTranscript(
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
        exp_prefix,
        algorithm="exact_fraction_real_log_atanh_pow2_v1",
        exact_work_count=7,
    )
    evidence = _make_local_detector_helper_failure_evidence(
        call=GalerkinLocalDetectorHelperCall.ADMITTED_HULL_PROBABILITY,
        channel_index=2,
        failure=CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED,
        local_exact_work_count=3,
        nested_kernel=None,
        nested_failure=None,
        nested_exact_work_count=None,
        prior_exp_transcripts=(exp_prefix,),
        prior_log_transcripts=(log_prefix,),
        planned_exact_work_count=5,
        attempted_exact_work_count=5,
    )
    assert isinstance(evidence, GalerkinLocalDetectorHelperFailureEvidence)
    assert evidence.channel_index == 2
    assert evidence.call is (
        GalerkinLocalDetectorHelperCall.ADMITTED_HULL_PROBABILITY
    )
    assert evidence.local_exact_work_count_exact == "3"
    assert evidence.planned_exact_work_count_exact == "5"
    assert evidence.prior_exp_transcripts == (exp_prefix,)
    assert evidence.prior_log_transcripts == (log_prefix,)
    assert len(evidence.failure_digest) == 64
    with pytest.raises(ValueError, match="digest disagrees"):
        _validate_local_detector_helper_failure_evidence(
            dataclasses.replace(evidence, channel_index=1)
        )
    with pytest.raises(ValueError, match="counts disagree"):
        _make_local_detector_helper_failure_evidence(
            call=GalerkinLocalDetectorHelperCall.PRODUCTION_NLL,
            channel_index=0,
            failure=CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED,
            local_exact_work_count=3,
            nested_kernel=None,
            nested_failure=None,
            nested_exact_work_count=None,
        )

    def seal_prefixes(
        *,
        exp: object = (),
        log: object = (),
    ) -> GalerkinLocalDetectorHelperFailureEvidence:
        return _make_local_detector_helper_failure_evidence(
            call=GalerkinLocalDetectorHelperCall.PRODUCTION_PROBABILITY,
            channel_index=0,
            failure=CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED,
            local_exact_work_count=3,
            nested_kernel=None,
            nested_failure=None,
            nested_exact_work_count=None,
            prior_exp_transcripts=exp,
            prior_log_transcripts=log,
            planned_exact_work_count=5,
            attempted_exact_work_count=5,
        )

    class TranscriptTuple(tuple[EntireWorkTranscript, ...]):
        """Deliberately noncanonical tuple subclass."""

    for non_tuple in ([exp_prefix], TranscriptTuple((exp_prefix,))):
        with pytest.raises(TypeError, match="tuples"):
            seal_prefixes(exp=non_tuple)
    with pytest.raises(TypeError, match="wrong type"):
        seal_prefixes(exp=(object(),))

    with pytest.raises(ValueError, match="exp.*algorithm"):
        seal_prefixes(exp=(log_prefix,))
    with pytest.raises(ValueError, match="log.*algorithm"):
        seal_prefixes(log=(exp_prefix,))
    with pytest.raises(ValueError, match="exp.*algorithm"):
        seal_prefixes(exp=(dataclasses.replace(exp_prefix, algorithm="bad"),))
    with pytest.raises(ValueError, match="log.*algorithm"):
        seal_prefixes(log=(dataclasses.replace(log_prefix, algorithm="bad"),))

    integer_fields = (
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
    )
    positive_fields = (
        "precision_bits",
        "maximum_terms",
        "maximum_work",
        "maximum_rational_bits",
    )
    for field_name in integer_fields:
        with pytest.raises(TypeError, match="Python ints"):
            seal_prefixes(
                exp=(dataclasses.replace(exp_prefix, **{field_name: True}),)
            )
        for invalid in (-1, 1 << 63):
            with pytest.raises(ValueError, match="policies|resource"):
                seal_prefixes(
                    exp=(
                        dataclasses.replace(
                            exp_prefix, **{field_name: invalid}
                        ),
                    )
                )
    for field_name in positive_fields:
        with pytest.raises(ValueError, match="policies"):
            seal_prefixes(
                exp=(dataclasses.replace(exp_prefix, **{field_name: 0}),)
            )
    for malformed in (
        dataclasses.replace(exp_prefix, maximum_rational_bits=1),
        dataclasses.replace(exp_prefix, maximum_rational_bits=1_048_577),
        dataclasses.replace(exp_prefix, exact_work_count=101),
        dataclasses.replace(
            exp_prefix,
            precision_bits=256,
            maximum_rational_bits=256,
        ),
    ):
        with pytest.raises(ValueError, match="rational|resource|precision"):
            seal_prefixes(exp=(malformed,))


__all__: list[str] = []
