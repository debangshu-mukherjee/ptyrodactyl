r"""Tests for composed local vacuum-terminal evidence carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_vacuum_terminal_types import (
    GalerkinLocalVacuumBranchEvidence,
    GalerkinLocalVacuumCutBalance,
    GalerkinLocalVacuumHalfSpaceDisposition,
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
    GalerkinLocalVacuumTerminalEntireEvidence,
    GalerkinLocalVacuumTerminalFailure,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_local_vacuum_terminal_enums_are_explicit_and_disjoint() -> None:
    """Freeze honest continuation claims and three-way branch statuses.

    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumHalfSpaceDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumTerminalDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumTerminalFailure`
    """
    assert {
        item.name: item.value
        for item in GalerkinLocalVacuumTerminalDisposition
    } == {
        "PLANE_DEFINED_FREE_CONTINUATION": ("plane_defined_free_continuation"),
        "NATIVE_ZERO_DEFECT_TERMINAL_SECTOR": (
            "native_zero_defect_terminal_sector"
        ),
        "NATIVE_ZERO_DEFECT_SLAB": "native_zero_defect_slab",
    }
    assert {
        item.name: item.value
        for item in GalerkinLocalVacuumHalfSpaceDisposition
    } == {
        "PROPAGATING_INWARD_EXACT_ZERO": "propagating_inward_exact_zero",
        "PROPAGATING_INWARD_PROVABLY_NONZERO": (
            "propagating_inward_provably_nonzero"
        ),
        "PROPAGATING_INWARD_UNRESOLVED": "propagating_inward_unresolved",
        "EVANESCENT_GROWING_EXACT_ZERO": "evanescent_growing_exact_zero",
        "EVANESCENT_GROWING_PROVABLY_NONZERO": (
            "evanescent_growing_provably_nonzero"
        ),
        "EVANESCENT_GROWING_UNRESOLVED": "evanescent_growing_unresolved",
        "GRAZING_DERIVATIVE_EXACT_ZERO": "grazing_derivative_exact_zero",
        "GRAZING_DERIVATIVE_PROVABLY_NONZERO": (
            "grazing_derivative_provably_nonzero"
        ),
        "GRAZING_DERIVATIVE_UNRESOLVED": ("grazing_derivative_unresolved"),
        "ROOT_UNCLASSIFIED": "root_unclassified",
    }
    assert {
        item.name: int(item) for item in GalerkinLocalVacuumTerminalFailure
    } == {
        "ZERO_SLAB_NONCERTIFICATE": 1 << 0,
        "PROJECTION_NONCERTIFICATE": 1 << 1,
        "CURRENT_DIAGNOSTIC_NONCERTIFICATE": 1 << 2,
        "CURRENT_OPERATOR_NONCERTIFICATE": 1 << 3,
        "CURRENT_ACTION_NONCERTIFICATE": 1 << 4,
        "ROOT_UNCLASSIFIED": 1 << 5,
        "ROOT_PROPAGATOR_FAILURE": 1 << 6,
        "CAUCHY_CROSSCHECK_EMPTY": 1 << 7,
        "BRANCH_CROSSCHECK_EMPTY": 1 << 8,
        "CUT_BALANCE_CROSSCHECK_EMPTY": 1 << 9,
        "NATIVE_STRUCTURAL_ZERO_UNAVAILABLE": 1 << 10,
        "DISPOSITION_SCOPE_MISMATCH": 1 << 11,
        "HOST_ARITHMETIC_UNSUPPORTED": 1 << 12,
        "DIRECT_WORK_BUDGET_EXCEEDED": 1 << 13,
        "DIRECT_WORK_COUNT_OVERFLOW": 1 << 14,
        "ARITHMETIC_RANGE_FAILURE": 1 << 15,
        "ENTIRE_HELPER_ENCLOSURE_FAILURE": 1 << 16,
        "DIRECT_RATIONAL_SIZE_FAILURE": 1 << 17,
    }


def test_local_vacuum_branch_and_cut_carriers_keep_routes_separate() -> None:
    """Keep submitted x, state transfer, total mismatch, and cut routes apart.

    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumBranchEvidence`
    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumCutBalance`
    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumTerminalEntireEvidence`
    """
    branch = _fields(GalerkinLocalVacuumBranchEvidence)
    helper = _fields(GalerkinLocalVacuumTerminalEntireEvidence)
    assert {
        "helper_attempted",
        "helper_eligible",
        "kernel_labels",
        "transcripts",
        "failure_reasons",
        "failure_work_counts",
        "total_series_terms",
        "total_range_reductions",
        "total_root_enclosures",
        "total_rectangle_products",
        "total_reciprocal_steps",
        "total_exact_work_count",
        "helper_evidence_digest",
    } <= helper
    assert {
        "root_certificates",
        "propagators",
        "root_failure_reasons",
        "root_failure_work_counts",
        "propagator_failure_reasons",
        "propagator_failure_work_counts",
        "entire_evidence",
        "inner_cauchy_rectangles",
        "outer_cauchy_rectangles",
        "endpoint_cauchy_mismatch_rectangles",
        "forced_cauchy_mismatch_rectangles",
        "certified_cauchy_mismatch_rectangles",
        "defining_branch_rectangles",
        "endpoint_branch_mismatch_rectangles",
        "forced_branch_mismatch_rectangles",
        "certified_branch_mismatch_rectangles",
        "submitted_state_branch_mismatch_upper_bounds",
        "projection_state_transfer_branch_mismatch_upper_bounds",
        "projection_total_branch_mismatch_upper_bounds",
        "frozen_positive_root_realizations",
        "frozen_positive_root_error_bounds",
        "physical_phase_realizations",
        "frozen_defining_branch_points",
        "production_to_submitted_amplitude_error_bounds",
        "state_radius_amplitude_error_bounds",
        "exact_state_total_amplitude_error_bounds",
        "production_amplitude_norm_upper_bounds",
        "exact_state_amplitude_norm_upper_bounds",
        "production_prediction_l2_norm_upper_bound",
        "exact_state_prediction_error_l2_upper_bound",
        "exact_state_prediction_l2_norm_upper_bound",
        "prediction_branch_role",
        "prediction_branch_role_scope",
        "cauchy_crosscheck_mask",
        "branch_crosscheck_mask",
        "half_space_dispositions",
        "failure_mask",
        "maximum_root_work",
        "maximum_propagator_interval_work",
        "maximum_rational_bits",
        "direct_rational_peak_bits",
        "direct_rational_work_count_exact",
        "direct_rational_failure",
        "hull_algorithm",
        "hull_attempted_endpoint_count",
        "hull_completed_endpoint_count",
        "hull_input_peak_bits",
        "hull_output_peak_bits",
        "hull_normal_floor_count",
        "hull_range_failure",
        "hull_evidence_digest",
        "production_to_submitted_amplitude_scope",
        "state_radius_amplitude_scope",
        "exact_state_amplitude_scope",
        "submitted_plane_mismatch_scope",
        "projection_state_transfer_mismatch_scope",
        "projection_total_mismatch_scope",
        "root_realization_scope",
        "helper_policy_digest",
        "physical_root_identity_digest",
        "cauchy_evidence_digest",
        "branch_evidence_digest",
    } <= branch
    cut = _fields(GalerkinLocalVacuumCutBalance)
    assert {
        "current_difference_lower_bound",
        "current_difference_upper_bound",
        "negative_defect_work_lower_bound",
        "negative_defect_work_upper_bound",
        "certified_balance_lower_bound",
        "certified_balance_upper_bound",
        "normal_arithmetic_eligible",
        "cut_balance_eligible",
        "failure_mask",
        "maximum_rational_bits",
        "direct_rational_peak_bits",
        "direct_rational_work_count_exact",
        "direct_rational_failure",
        "current_difference_formula",
        "defect_work_formula",
        "cut_balance_digest",
    } <= cut
    assert {
        "symmetrized_defect_work",
        "tolerance_zero",
        "detector_eligible",
    }.isdisjoint(branch | cut)


def test_local_vacuum_terminal_certificate_owns_no_detector_claim() -> None:
    """Freeze the composed parent ladder before any detector composition.

    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumTerminalCertificate`
    """
    fields = _fields(GalerkinLocalVacuumTerminalCertificate)
    assert {
        "projection_certificate",
        "inner_current_diagnostic",
        "outer_current_diagnostic",
        "branch_evidence",
        "cut_balance",
        "defining_plane_coordinate",
        "comparison_plane_coordinate",
        "current_diagnostic_eligible",
        "current_operator_eligible",
        "current_action_eligible",
        "vacuum_branch_eligible",
        "failure_mask",
        "terminal_axis",
        "terminal_side",
        "terminal_scope",
        "disposition",
        "amplitude_dependency_scope",
        "completion_scope",
        "parent_projection_certificate_digest",
        "terminal_identity_digest",
        "terminal_evidence_digest",
    } <= fields
    assert {
        "detector_eligible",
        "pixel_form",
        "quadrature",
        "dose",
        "response",
        "calibration",
        "likelihood",
    }.isdisjoint(fields)


__all__: list[str] = []
