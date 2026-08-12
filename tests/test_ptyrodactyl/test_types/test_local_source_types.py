"""Tests for the disjoint LVT.20 local additional-source carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_source_types import (
    GalerkinLocalAdditionalSource,
    GalerkinLocalAdditionalSourceCertificate,
    GalerkinLocalAdditionalSourceCertificateFailure,
    GalerkinLocalAdditionalSourceRoute,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_local_source_routes_and_failures_are_typed_and_disjoint() -> None:
    """Freeze the two source routes and direct-certificate outcomes.

    :see: :class:`ptyrodactyl.types.GalerkinLocalAdditionalSourceRoute`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalAdditionalSourceCertificateFailure`
    """
    assert {item.value for item in GalerkinLocalAdditionalSourceRoute} == {
        "zero",
        "local_cell",
    }
    assert {
        item.value for item in GalerkinLocalAdditionalSourceCertificateFailure
    } == {
        "none",
        "host_arithmetic_unsupported",
        "work_budget_exceeded",
        "root_enclosure_failure",
        "arithmetic_range_failure",
    }


def test_local_source_carriers_own_only_lvt20c_evidence() -> None:
    """Exclude represented, matched, slab, and terminal state.

    :see: :class:`ptyrodactyl.types.GalerkinLocalAdditionalSource`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalAdditionalSourceCertificate`
    """
    source = _fields(GalerkinLocalAdditionalSource)
    certificate = _fields(GalerkinLocalAdditionalSourceCertificate)
    assert {
        "target",
        "source_cell_values",
        "algebraic_additional_source",
        "algebraic_volume_sqrt",
        "route",
        "target_digest",
        "parent_target_evidence_digest",
        "source_digest",
        "realization_digest",
    } <= source
    assert {
        "source",
        "exact_source_real_lower_bounds",
        "exact_source_real_upper_bounds",
        "exact_source_imag_lower_bounds",
        "exact_source_imag_upper_bounds",
        "component_error_bounds",
        "additional_source_error_upper_bound",
        "direct_term_count",
        "maximum_direct_terms",
        "failure",
        "certificate_digest",
    } <= certificate
    forbidden = {
        "represented_source",
        "incident_field",
        "matched_source",
        "total_source",
        "slab",
        "terminal",
        "vacuum_branch_eligible",
        "potential",
    }
    assert forbidden.isdisjoint(source | certificate)
