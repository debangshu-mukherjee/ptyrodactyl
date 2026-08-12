"""Tests for disjoint represented ``LOCAL_CELL_LVT1`` source carriers."""

from __future__ import annotations

import dataclasses

from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalComplexRectangles,
    GalerkinLocalRepresentedSource,
    GalerkinLocalRepresentedSourceActions,
    GalerkinLocalRepresentedSourceCertificate,
    GalerkinLocalRepresentedSourceFailure,
    GalerkinLocalRepresentedSourceKind,
    GalerkinLocalRepresentedSourceModes,
    GalerkinLocalSourceAxis,
    GalerkinLocalSourcePhaseConvention,
)


def _fields(value: type[object]) -> set[str]:
    """Return declared Equinox/dataclass field names."""
    return {field.name for field in dataclasses.fields(value)}


def test_local_represented_routes_axes_phases_and_failures_are_disjoint() -> (
    None
):
    """Freeze the represented-source route and typed noncertificate surface.

    :see: :class:`ptyrodactyl.types.GalerkinLocalRepresentedSourceFailure`
    :see: :class:`ptyrodactyl.types.GalerkinLocalRepresentedSourceKind`
    :see: :class:`ptyrodactyl.types.GalerkinLocalSourceAxis`
    :see: :class:`ptyrodactyl.types.GalerkinLocalSourcePhaseConvention`
    """
    assert {item.value for item in GalerkinLocalRepresentedSourceKind} == {
        "local_represented_plane_mode",
        "local_represented_coherent_focused",
    }
    assert {int(item) for item in GalerkinLocalSourceAxis} == {0, 1, 2}
    assert {item.value for item in GalerkinLocalSourcePhaseConvention} == {
        "local_physical_kappa_scan_source_plus_aberration"
    }
    assert {item.value for item in GalerkinLocalRepresentedSourceFailure} == {
        "none",
        "additional_source_noncertificate",
        "terminal_orientation_unsupported",
        "undeclared_incident_mode",
        "nonexact_incident_disposition",
        "exact_shell_failure",
        "nonforward_or_grazing",
        "duplicate_transverse_fiber",
        "nonpositive_exact_flux",
        "host_arithmetic_unsupported",
        "direct_work_budget_exceeded",
        "root_enclosure_failure",
        "arithmetic_range_failure",
    }


def test_local_represented_carriers_own_direct_dbrsmtc_evidence_only() -> None:
    """Freeze parent, mode, action, rectangle, and ledger ownership.

    :see: :class:`ptyrodactyl.types.GalerkinLocalComplexRectangles`
    :see: :class:`ptyrodactyl.types.GalerkinLocalRepresentedSource`
    :see: :class:`ptyrodactyl.types.GalerkinLocalRepresentedSourceActions`
    :see: :class:`ptyrodactyl.types.GalerkinLocalRepresentedSourceCertificate`
    :see: :class:`ptyrodactyl.types.GalerkinLocalRepresentedSourceModes`
    """
    modes = set(GalerkinLocalRepresentedSourceModes._fields)
    actions = set(GalerkinLocalRepresentedSourceActions._fields)
    rectangles = set(GalerkinLocalComplexRectangles._fields)
    source = _fields(GalerkinLocalRepresentedSource)
    certificate = _fields(GalerkinLocalRepresentedSourceCertificate)
    assert {
        "aperture_weights",
        "phased_coefficients",
        "incident_field",
        "exact_shell_mask",
        "exact_incident_disposition_mask",
        "exact_reduced_flux_lower_bound",
        "target_reduced_flux_discrepancy_upper_bound",
    } <= modes
    assert actions == {
        "free_action",
        "physical_cap_action",
        "interaction_action",
        "additional_source",
        "vacuum_matched_source",
        "total_source",
        "scattered_source",
    }
    assert rectangles == {
        "real_lower_bounds",
        "real_upper_bounds",
        "imag_lower_bounds",
        "imag_upper_bounds",
    }
    assert {
        "target",
        "additional_source_certificate",
        "modes",
        "actions",
        "normal_axis",
        "source_digest",
        "source_evidence_digest",
    } <= source
    assert {
        "free_rectangles",
        "physical_cap_rectangles",
        "interaction_rectangles",
        "additional_source_rectangles",
        "vacuum_matched_rectangles",
        "total_source_rectangles",
        "scattered_source_rectangles",
        "free_component_error_bounds",
        "physical_cap_component_error_bounds",
        "interaction_component_error_bounds",
        "additional_source_component_error_bounds",
        "vacuum_matched_component_error_bounds",
        "total_source_component_error_bounds",
        "scattered_source_component_error_bounds",
        "incident_field_norm_upper_bound",
        "certificate_digest",
    } <= certificate
    forbidden = {
        "legacy_target",
        "potential",
        "raw_additional_source",
        "gram_error",
        "delta_H",
        "solver_residual",
        "slab",
        "terminal",
        "detector",
    }
    assert forbidden.isdisjoint(modes | actions | source | certificate)
