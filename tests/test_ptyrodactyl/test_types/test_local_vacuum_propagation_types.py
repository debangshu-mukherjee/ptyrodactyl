r"""Tests for exact local vacuum propagation evidence carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_vacuum_propagation_types import (
    GalerkinLocalVacuumPropagationError,
    GalerkinLocalVacuumPropagationFailure,
    GalerkinLocalVacuumPropagator,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRootCertificate,
    GalerkinLocalVacuumRootClass,
    GalerkinLocalVacuumWorkTranscript,
    GalerkinLocalVacuumZeroWitness,
    GalerkinLocalVacuumZeroWitnessRoute,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_vacuum_root_enums_and_carrier_fields_are_disjoint() -> None:
    """Freeze strict branch values, typed failures, and exact evidence fields.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumPropagationError`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumPropagationFailure`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumRationalInterval`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumRootClass`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumWorkTranscript`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumZeroWitnessRoute`
    """
    root_classes = {
        item.name: item.value for item in GalerkinLocalVacuumRootClass
    }
    assert root_classes == {
        "PROPAGATING": "propagating",
        "EVANESCENT": "evanescent",
        "GRAZING": "grazing",
        "UNCLASSIFIED": "unclassified",
    }
    assert {
        item.name: item.value for item in GalerkinLocalVacuumZeroWitnessRoute
    } == {
        "EXACT_RATIONAL_DIFFERENCE": "exact_rational_difference",
        "SYMBOLIC_NORMAL_FORM_DIFFERENCE": ("symbolic_normal_form_difference"),
    }
    assert {
        item.name: item.value for item in GalerkinLocalVacuumPropagationFailure
    } == {
        "ZERO_WITNESS_INCONSISTENT": "zero_witness_inconsistent",
        "ROOT_UNCLASSIFIED": "root_unclassified",
    }
    assert {
        "lower_numerator",
        "lower_denominator",
        "upper_numerator",
        "upper_denominator",
    } == _fields(GalerkinLocalVacuumRationalInterval)
    assert {
        "algorithm",
        "maximum_work",
        "maximum_rational_bits",
        "additions",
        "subtractions",
        "multiplications",
        "divisions",
        "root_enclosures",
        "exact_work_count",
    } == _fields(GalerkinLocalVacuumWorkTranscript)


def test_zero_witness_scope_is_formal_and_parent_free() -> None:
    """Keep formal equality separate from authenticated physical LVT.39.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumRootCertificate`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumZeroWitness`
    """
    witness_fields = _fields(GalerkinLocalVacuumZeroWitness)
    assert {
        "left_normal_form",
        "right_normal_form",
        "maximum_rational_bits",
        "route",
        "witness_formula",
        "trust_scope",
        "witness_digest",
    } == witness_fields
    root_fields = _fields(GalerkinLocalVacuumRootCertificate)
    assert {
        "q_interval",
        "zero_witness",
        "root_interval",
        "work_transcript",
        "classification",
        "classification_formula",
        "root_formula",
        "witness_scope",
        "completion_scope",
        "root_identity_digest",
        "root_evidence_digest",
    } == root_fields
    forbidden = {
        "target",
        "projection_certificate",
        "branch_amplitude",
        "forced_integral",
        "terminal_disposition",
        "detector_eligible",
    }
    assert forbidden.isdisjoint(witness_fields | root_fields)


def test_vacuum_propagator_keeps_exact_entries_and_helper_transcript() -> None:
    """Pin row-major exact matrix storage and disallow downstream claims.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalVacuumPropagator`
    """
    fields = _fields(GalerkinLocalVacuumPropagator)
    assert {
        "root_certificate",
        "entries",
        "entire_transcript",
        "interval_work_transcript",
        "distance_numerator",
        "distance_denominator",
        "precision_bits",
        "maximum_terms",
        "maximum_entire_work",
        "maximum_range_reductions",
        "propagator_formula",
        "trust_scope",
        "completion_scope",
        "propagator_identity_digest",
        "propagator_evidence_digest",
    } == fields
    assert {
        "projection_parent",
        "mismatch",
        "branch_amplitudes",
        "terminal_eligible",
        "detector_claim",
    }.isdisjoint(fields)


__all__: list[str] = []
