"""Tests for disjoint LVT exact-compression and interaction-core carriers."""

import dataclasses

from ptyrodactyl.types.local_cell_interaction_types import (
    GalerkinLocalCellCompressionFailure,
    GalerkinLocalCellExactCompression,
    GalerkinLocalCellInteractionCore,
)


def test_compression_failures_are_unique_and_fail_closed() -> None:
    """Keep every typed noncertificate distinct from successful evidence."""
    values = [failure.value for failure in GalerkinLocalCellCompressionFailure]
    assert len(values) == len(set(values))
    assert GalerkinLocalCellCompressionFailure.NONE.value == "none"
    assert all(value != "none" for value in values[1:])


def test_exact_compression_owns_only_lvt14_through_lvt18_fields() -> None:
    """Exclude CAP, source, terminal, free-diagonal, and solver-ready seams."""
    names = {
        field.name
        for field in dataclasses.fields(GalerkinLocalCellExactCompression)
    }
    required = {
        "realization",
        "product_support",
        "difference_indices",
        "difference_interaction_positions",
        "difference_multiplicities",
        "state_pair_interaction_positions",
        "interaction_coupling",
        "interaction_coefficients",
        "interaction_coefficient_error_bounds",
        "fixed_interaction_error_bound",
        "operator_digest",
        "certificate_digest",
    }
    assert required <= names
    forbidden_fragments = (
        "absorber",
        "cap_",
        "detector",
        "free_diagonal",
        "source",
        "stability",
        "terminal",
        "tail",
    )
    assert not {
        name
        for name in names
        if any(fragment in name for fragment in forbidden_fragments)
    }


def test_interaction_core_is_disjoint_and_non_solver_ready() -> None:
    """Keep the L3 action core distinct from a completed target manifest."""
    names = {
        field.name
        for field in dataclasses.fields(GalerkinLocalCellInteractionCore)
    }
    assert names == {
        "compression",
        "action_route",
        "adjoint_route",
        "completion_scope",
        "operator_digest",
    }
    assert not hasattr(GalerkinLocalCellInteractionCore, "free_diagonal")
    assert not hasattr(GalerkinLocalCellInteractionCore, "cap_scale")
