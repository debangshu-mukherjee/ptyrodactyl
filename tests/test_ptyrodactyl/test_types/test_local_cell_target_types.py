"""Tests for disjoint solver-ready local-cell target carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellFixedLinearErrorLedger,
    GalerkinLocalCellTargetManifest,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_local_cell_target_carriers_keep_fixed_linear_scope_disjoint() -> None:
    """Freeze the LVT target/ledger ownership boundary.

    :see: :class:`ptyrodactyl.types.GalerkinLocalCellFixedLinearErrorLedger`
    :see: :class:`ptyrodactyl.types.GalerkinLocalCellTargetManifest`
    """
    ledger = _fields(GalerkinLocalCellFixedLinearErrorLedger)
    manifest = _fields(GalerkinLocalCellTargetManifest)
    assert {
        "algebraic_free_diagonal",
        "free_operator_error_bound",
        "interaction_operator_error_bound",
        "absorber_operator_error_bound",
        "cap_scale_error_bound",
        "cap_operator_error_bound",
        "fixed_linear_operator_error_bound",
        "free_geometry_digest",
        "parent_operator_digest",
        "ledger_digest",
    } <= ledger
    assert {
        "cap_floor_proof",
        "fixed_linear_error_ledger",
        "exact_target_incident_full_offset_max",
        "exact_target_outgoing_full_offset_max",
        "target_route",
        "target_digest",
        "manifest_evidence_digest",
        "target_name",
    } <= manifest
    forbidden = {
        "tail_error",
        "source",
        "right_hand_side",
        "per_call_error",
        "gram_error",
        "terminal",
        "potential",
    }
    assert forbidden.isdisjoint(ledger | manifest)


def test_local_cell_manifest_has_no_legacy_potential_compatibility() -> None:
    """Expose only ``local_potential`` for the disjoint LVT-1 route."""
    assert isinstance(
        GalerkinLocalCellTargetManifest.local_potential,
        property,
    )
    assert not hasattr(GalerkinLocalCellTargetManifest, "potential")
