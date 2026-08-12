r"""Tests for exact local projection-defect evidence carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_projection_types import (
    GalerkinLocalProjectionDefectCertificate,
    GalerkinLocalProjectionDefectFailure,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_local_projection_failure_bits_and_carrier_fields_are_disjoint() -> (
    None
):
    """Freeze independent structural, policy, and arithmetic evidence.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalProjectionDefectCertificate`
    :see: :class:`ptyrodactyl.types.GalerkinLocalProjectionDefectFailure`
    """
    assert {
        item.name: int(item) for item in GalerkinLocalProjectionDefectFailure
    } == {
        "ZERO_SLAB_NONCERTIFICATE": 1 << 0,
        "PARENT_SOURCE_EVIDENCE_MISMATCH": 1 << 1,
        "STATE_RADIUS_UNAVAILABLE": 1 << 2,
        "OPERATIONAL_STATE_BUDGET_MISSED": 1 << 3,
        "TERMINAL_SCOPE_INCOMPLETE": 1 << 4,
        "STRUCTURAL_EXACT_ZERO_UNAVAILABLE": 1 << 5,
        "HOST_ARITHMETIC_UNSUPPORTED": 1 << 6,
        "GRAM_PAIR_BUDGET_EXCEEDED": 1 << 7,
        "GRAM_PAIR_COUNT_OVERFLOW": 1 << 8,
        "ROOT_ENCLOSURE_FAILURE": 1 << 9,
        "ARITHMETIC_RANGE_FAILURE": 1 << 10,
    }
    fields = _fields(GalerkinLocalProjectionDefectCertificate)
    assert {
        "zero_slab_certificate",
        "stability_result",
        "scope_transverse_indices",
        "state_to_fiber_rows",
        "selected_state_mask",
        "exact_free_diagonal_lower_bounds",
        "exact_free_diagonal_upper_bounds",
        "structural_exact_zero_state_mask",
        "structural_exact_zero_fiber_mask",
        "gram_real_lower_bounds",
        "gram_real_upper_bounds",
        "gram_imag_lower_bounds",
        "gram_imag_upper_bounds",
        "measured_defect_squared_lower_bounds",
        "measured_defect_squared_upper_bounds",
        "measured_defect_upper_bounds",
        "operator_squared_norm_upper_bounds",
        "operator_norm_upper_bounds",
        "state_error_transfer_upper_bounds",
        "total_defect_upper_bounds",
        "state_radius_upper_bound",
        "maximum_state_error",
        "direct_pair_count",
        "maximum_gram_pairs",
        "maximum_stability_direct_pairs",
        "host_binary64_eligible",
        "normal_arithmetic_eligible",
        "structural_exact_zero_eligible",
        "finite_projection_bound_eligible",
        "operational_budget_eligible",
        "failure_mask",
        "projection_scope",
        "maximum_state_error_numerator",
        "maximum_state_error_denominator",
        "direct_pair_count_exact",
        "gram_formula",
        "measurement_formula",
        "operator_bound_formula",
        "state_lift_formula",
        "precision_transcript",
        "error_scope",
        "completion_scope",
        "parent_target_evidence_digest",
        "parent_source_evidence_digest",
        "parent_represented_certificate_digest",
        "parent_zero_slab_certificate_digest",
        "parent_stability_result_identity_digest",
        "parent_stability_result_evidence_digest",
        "state_identity_digest",
        "projection_identity_digest",
        "arithmetic_environment_digest",
        "gram_transcript_digest",
        "certificate_digest",
    } <= fields
    assert {
        "maximum_state_error",
        "maximum_gram_pairs",
        "maximum_stability_direct_pairs",
    }.isdisjoint(
        {
            "direct_pair_count",
            "state_radius_upper_bound",
            "total_defect_upper_bounds",
        }
    )
    assert {
        "sampled_quadrature",
        "terminal_propagator",
        "detector_response",
        "cancellation_eligible",
        "exact_spectral_norm",
    }.isdisjoint(fields)


__all__: list[str] = []
