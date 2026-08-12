r"""Tests for exact LVT.21--LVT.22 local zero-slab carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalVacuumReference,
    GalerkinLocalZeroSlabCertificate,
    GalerkinLocalZeroSlabFailure,
)


def _field_names(carrier: type[object]) -> set[str]:
    """Return the declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(carrier)}


def test_zero_slab_failure_bits_and_reference_are_disjoint() -> None:
    """Freeze the canonical vacuum declaration and simultaneous failure bits.

    :see: :class:`ptyrodactyl.types.GalerkinLocalVacuumReference`
    :see: :class:`ptyrodactyl.types.GalerkinLocalZeroSlabFailure`
    """
    assert GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value == (
        "stored zero is the vacuum value used by exact SC.2 k0 and SC.8 k_i"
    )
    bits = [
        int(reason)
        for reason in GalerkinLocalZeroSlabFailure
        if reason is not GalerkinLocalZeroSlabFailure.NONE
    ]
    assert len(bits) == 10
    assert len(set(bits)) == len(bits)
    assert all(bit > 0 and bit & (bit - 1) == 0 for bit in bits)
    assert int(GalerkinLocalZeroSlabFailure.NONE) == 0


def test_zero_slab_carrier_separates_exact_and_projection_predicates() -> None:
    """Keep exact spatial absence independent of finite projection evidence.

    :see: :class:`ptyrodactyl.types.GalerkinLocalZeroSlabCertificate`
    """
    fields = _field_names(GalerkinLocalZeroSlabCertificate)
    assert {
        "represented_source_certificate",
        "periodic_layer_indices",
        "potential_layer_zero_mask",
        "cap_layer_zero_mask",
        "additional_source_layer_zero_mask",
        "incident_exact_shell_mask",
        "exact_spatial_source_zero_eligible",
        "exact_zero_slab_eligible",
        "projection_match_eligible",
        "terminal_zero_slab_eligible",
        "failure_mask",
    } <= fields
    assert {
        "unwrapped_layer_start",
        "unwrapped_layer_stop",
        "cap_zero_block_lift",
        "slab_lower_numerator",
        "slab_lower_denominator",
        "slab_upper_numerator",
        "slab_upper_denominator",
        "layer_union_lower_numerator",
        "layer_union_lower_denominator",
        "layer_union_upper_numerator",
        "layer_union_upper_denominator",
    } <= fields
    assert {
        "target_digest",
        "parent_target_evidence_digest",
        "represented_source_digest",
        "parent_source_evidence_digest",
        "parent_represented_certificate_digest",
        "slab_digest",
        "certificate_digest",
    } <= fields
    forbidden = {
        "state",
        "solver_residual",
        "projection_defect",
        "cauchy_mismatch",
        "terminal_current",
        "vacuum_branch_eligible",
    }
    assert forbidden.isdisjoint(fields)
