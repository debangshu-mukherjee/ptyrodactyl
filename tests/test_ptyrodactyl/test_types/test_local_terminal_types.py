"""Tests for disjoint local coordinate-terminal current carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_terminal_types import (
    GalerkinLocalCoordinateCauchyCurrent,
    GalerkinLocalCurrentOperatorCertificate,
    GalerkinLocalCurrentOperatorFailure,
    GalerkinLocalTerminalActionFailure,
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalCurrentActionEnclosure,
    GalerkinLocalTerminalCurrentFailure,
    GalerkinLocalTerminalScope,
    GalerkinPreparedLocalCurrentOperator,
)


def _fields(carrier: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(carrier)}


def test_local_terminal_enums_and_carrier_boundaries() -> None:
    """Freeze the two scopes and three distinct typed outcome surfaces.

    :see: :class:`ptyrodactyl.types.GalerkinLocalTerminalScope`
    :see: :class:`ptyrodactyl.types.GalerkinLocalCurrentOperatorFailure`
    :see: :class:`ptyrodactyl.types.GalerkinLocalTerminalActionFailure`
    :see: :class:`ptyrodactyl.types.GalerkinLocalTerminalComplexRectangles`
    :see: :class:`ptyrodactyl.types.GalerkinLocalTerminalCurrentFailure`
    """
    assert {item.value for item in GalerkinLocalTerminalScope} == {
        "full_state_fibers",
        "selected_preterminal_fibers",
    }
    assert {item.value for item in GalerkinLocalCurrentOperatorFailure} == {
        "none",
        "target_fixed_linear_ineligible",
        "terminal_fiber_incomplete",
        "host_arithmetic_unsupported",
        "direct_work_budget_exceeded",
        "direct_work_count_overflow",
        "root_enclosure_failure",
        "arithmetic_range_failure",
        "current_normalization_unenclosed",
    }
    assert {item.value for item in GalerkinLocalTerminalActionFailure} == {
        "none",
        "operator_noncertificate",
        "host_arithmetic_unsupported",
        "direct_work_budget_exceeded",
        "direct_work_count_overflow",
        "arithmetic_range_failure",
    }
    assert {item.value for item in GalerkinLocalTerminalCurrentFailure} == {
        "none",
        "operator_noncertificate",
        "action_noncertificate",
        "host_arithmetic_unsupported",
        "direct_work_budget_exceeded",
        "direct_work_count_overflow",
        "arithmetic_range_failure",
    }


def test_local_terminal_operator_carrier_rejects_forged_shapes_and_digests() -> (  # noqa: E501
    None
):
    """Freeze operator/action/current ownership without downstream claims.

    :see: :class:`ptyrodactyl.types.GalerkinLocalCurrentOperatorCertificate`
    :see: :class:`ptyrodactyl.types.GalerkinPreparedLocalCurrentOperator`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalTerminalCurrentActionEnclosure`
    :see: :class:`ptyrodactyl.types.GalerkinLocalCoordinateCauchyCurrent`
    """
    rectangles = set(GalerkinLocalTerminalComplexRectangles._fields)
    operator = _fields(GalerkinLocalCurrentOperatorCertificate)
    prepared = _fields(GalerkinPreparedLocalCurrentOperator)
    action = _fields(GalerkinLocalTerminalCurrentActionEnclosure)
    current = _fields(GalerkinLocalCoordinateCauchyCurrent)
    assert rectangles == {
        "real_lower_bounds",
        "real_upper_bounds",
        "imag_lower_bounds",
        "imag_upper_bounds",
    }
    assert prepared == {"certificate"}
    assert {
        "target",
        "terminal_plane_coordinate",
        "scope_transverse_indices",
        "state_to_fiber_rows",
        "selected_state_mask",
        "trace_frozen_coefficients",
        "normal_frozen_coefficients",
        "exact_trace_coefficient_rectangles",
        "exact_normal_coefficient_rectangles",
        "current_operator_error_upper_bound",
        "number_current_scale",
        "action_work_count_exact",
        "current_diagnostic_work_count_exact",
        "operator_identity_digest",
        "operator_evidence_digest",
    } <= operator
    assert {
        "certificate",
        "submitted_field",
        "production_action",
        "frozen_action_rectangles",
        "action_error_upper_bound",
        "direct_work_count_exact",
        "state_identity_digest",
        "action_evidence_digest",
    } <= action
    assert {
        "action_enclosure",
        "trace_coefficients",
        "normal_derivative_coefficients",
        "reduced_current",
        "exact_reduced_current_lower_bound",
        "exact_reduced_current_upper_bound",
        "reduced_current_error_upper_bound",
        "direct_work_count_exact",
        "diagnostic_evidence_digest",
    } <= current
    forbidden = {
        "vacuum_branch_eligible",
        "detector_eligible",
        "detector",
        "legacy_target",
        "solver_residual",
        "projection_defect",
    }
    assert forbidden.isdisjoint(operator | prepared | action | current)
