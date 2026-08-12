r"""Tests for bounded local represented-source stability carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_stability_types import (
    GalerkinLocalStabilityDisposition,
    GalerkinLocalStabilityFailure,
    GalerkinLocalStabilityProof,
    GalerkinLocalStabilityResult,
    GalerkinLocalStabilityRoute,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_local_stability_routes_dispositions_and_failures_are_disjoint() -> (
    None
):
    """Freeze the local-only route and typed outcome surface.

    :see: :class:`ptyrodactyl.types.GalerkinLocalStabilityDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinLocalStabilityFailure`
    :see: :class:`ptyrodactyl.types.GalerkinLocalStabilityRoute`
    """
    assert {item.value for item in GalerkinLocalStabilityRoute} == {
        "local_exact_axial_cap_floor"
    }
    assert {item.value for item in GalerkinLocalStabilityDisposition} == {
        "operational_pass",
        "finite_state_radius_fallback",
        "matrix_floor_only",
        "rejected",
    }
    assert {item.value for item in GalerkinLocalStabilityFailure} == {
        "none",
        "state_budget_missed",
        "source_noncertificate",
        "exact_target_floor_unavailable",
        "nonpositive_exact_target_floor",
        "direct_work_budget_exceeded",
        "direct_work_count_overflow",
        "root_enclosure_failure",
        "host_arithmetic_unsupported",
        "arithmetic_range_failure",
    }


def test_local_stability_carriers_bind_full_parents_and_transcripts() -> None:
    """Freeze proof ownership and the result's full nested trust root.

    :see: :class:`ptyrodactyl.types.GalerkinLocalStabilityProof`
    :see: :class:`ptyrodactyl.types.GalerkinLocalStabilityResult`
    """
    proof = _fields(GalerkinLocalStabilityProof)
    result = _fields(GalerkinLocalStabilityResult)
    assert {
        "lower_singular_bound",
        "algebraic_residual_upper_bound",
        "fixed_linear_state_transfer_upper_bound",
        "total_source_error_upper_bound",
        "exact_target_residual_upper_bound",
        "state_radius_upper_bound",
        "maximum_state_error",
        "direct_work_count",
        "maximum_direct_pairs",
        "matrix_floor_eligible",
        "state_radius_eligible",
        "operational_state_eligible",
        "exact_floor_numerator",
        "residual_squared_numerator",
        "field_norm_squared_numerator",
        "exact_target_residual_upper_numerator",
        "state_radius_upper_numerator",
        "result_identity_digest",
        "proof_evidence_digest",
    } <= proof
    assert result == {
        "certificate",
        "solve_result",
        "proof",
        "result_identity_digest",
        "result_evidence_digest",
        "completion_scope",
    }
    forbidden = {
        "legacy_manifest",
        "legacy_source",
        "realized_floor",
        "solver_residual_used_as_bound",
        "terminal",
        "detector",
    }
    assert forbidden.isdisjoint(proof | result)
