r"""Tests for disjoint axial local-cell CAP evidence carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types import (
    GalerkinAxialCapCoefficientCertificate,
    GalerkinAxialCapCoefficientFailure,
    GalerkinAxialCapExactFloorFailure,
    GalerkinAxialCapFloorProof,
    GalerkinAxialCapRealizedFloorFailure,
    GalerkinAxialCapRealizedFloorRoute,
    GalerkinAxialCellAbsorber,
)


def _field_names(carrier: type[object]) -> set[str]:
    """Return the declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(carrier)}


def test_absorber_carriers_own_only_the_l4_evidence_layers() -> None:
    """Keep profile, coefficient, and floor evidence explicitly disjoint.

    :see: :class:`ptyrodactyl.types.GalerkinAxialCellAbsorber`
    :see: :class:`ptyrodactyl.types.GalerkinAxialCapCoefficientCertificate`
    :see: :class:`ptyrodactyl.types.GalerkinAxialCapFloorProof`
    """
    absorber_fields = _field_names(GalerkinAxialCellAbsorber)
    certificate_fields = _field_names(GalerkinAxialCapCoefficientCertificate)
    proof_fields = _field_names(GalerkinAxialCapFloorProof)

    assert {
        "interaction_core",
        "layer_values",
        "absorber_coefficients",
        "signed_absorber_positions",
        "source_digest",
        "operator_digest",
    } <= absorber_fields
    assert {
        "difference_absorber_positions",
        "difference_multiplicities",
        "state_pair_absorber_positions",
        "absorber_operator_error_bound",
        "certificate_digest",
    } <= certificate_fields
    assert {
        "exact_target_failure",
        "realized_floor_failure",
        "gram_subinterval_numerator",
        "gram_subinterval_denominator",
        "gram_precision_bits",
        "ldl_iteration_count",
        "gram_work_count",
        "maximum_gram_work",
        "gram_transcript_digest",
    } <= proof_fields
    forbidden = {
        "free_diagonal",
        "right_hand_side",
        "source",
        "terminal",
        "hamiltonian",
        "solve",
    }
    assert forbidden.isdisjoint(
        absorber_fields | certificate_fields | proof_fields
    )


def test_exact_and_realized_failure_spaces_cannot_be_conflated() -> None:
    """Freeze independent LVT.29a and coefficient-dependent LVT.32 statuses.

    :see: :class:`ptyrodactyl.types.GalerkinAxialCapCoefficientFailure`
    :see: :class:`ptyrodactyl.types.GalerkinAxialCapExactFloorFailure`
    :see: :class:`ptyrodactyl.types.GalerkinAxialCapRealizedFloorFailure`
    :see: :class:`ptyrodactyl.types.GalerkinAxialCapRealizedFloorRoute`
    """
    exact_values = {
        failure.value for failure in GalerkinAxialCapExactFloorFailure
    }
    realized_values = {
        failure.value for failure in GalerkinAxialCapRealizedFloorFailure
    }
    assert "coefficient_certificate_not_finite" not in exact_values
    assert "coefficient_certificate_not_finite" in realized_values
    assert "gram_degree_budget_exceeded" in exact_values
    assert "gram_degree_budget_exceeded" not in realized_values
    assert {route.value for route in GalerkinAxialCapRealizedFloorRoute} == {
        "exact_frozen_scale_lvt32a",
        "scale_transfer_lvt32b",
    }
    assert GalerkinAxialCapCoefficientFailure.NONE.value == "none"
