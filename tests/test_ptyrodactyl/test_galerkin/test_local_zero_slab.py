r"""Tests for exact LVT.21--LVT.22 zero-slab leaf logic."""

from __future__ import annotations

import functools
import inspect
from dataclasses import replace
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import ptyrodactyl.galerkin.local_represented_sources as represented_sources
import ptyrodactyl.galerkin.local_sources as local_sources
import ptyrodactyl.galerkin.local_zero_slab as zero_slab
from ptyrodactyl.galerkin.absorber import (
    certify_axial_cap_floor,
    certify_axial_cell_absorber,
    realize_axial_cell_absorber,
)
from ptyrodactyl.galerkin.local_cell_system import (
    compose_local_cell_galerkin_target,
)
from ptyrodactyl.galerkin.local_zero_slab import (
    _derive_layer_lift,
    _eligibility_predicates,
    _failure_mask,
    _incident_predicates,
    _layer_zero_mask,
    _vacuum_reference_eligible,
    certify_local_zero_slab,
    prepare_local_zero_slab_certificate,
)
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalRepresentedSourceCertificate,
    GalerkinLocalRepresentedSourceKind,
    GalerkinLocalSourcePhaseConvention,
)
from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalVacuumReference,
    GalerkinLocalZeroSlabFailure,
)
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_represented_sources as represented_tests,
)

_PHASE = GalerkinLocalSourcePhaseConvention.PHYSICAL_WAVEVECTOR
_LOWER = np.float64(0.75)
_UPPER = np.float64(1.25)


@functools.lru_cache(maxsize=4)
def _zero_additional(name: str = "zero-slab-target"):
    """Return one target-identical exact ZERO LVT.20 certificate."""
    source = local_sources._realize_zero_prepared(
        represented_tests._target(name)
    )
    return local_sources._certify_canonical_source(source, 9)


@functools.lru_cache(maxsize=4)
def _layer_selective_additional(name: str = "zero-slab-target"):
    """Return q nonzero outside and exactly zero on terminal layer one."""
    cells = jnp.asarray(
        [[[1.0 + 2.0j, 0.0 + 0.0j, -3.0j]]], dtype=jnp.complex128
    )
    source = local_sources._realize_local_cell_prepared(
        represented_tests._target(name), cells
    )
    return local_sources._certify_canonical_source(source, 9)


def _compose_source(name: str, additional):
    """Compose the fixture's declared exact-shell zero-mode plane source."""
    target = represented_tests._target(name)
    zero_position = int(
        np.flatnonzero(np.all(np.asarray(target.state_indices) == 0, axis=1))[
            0
        ]
    )
    size = target.state_indices.shape[0]
    weights = (
        jnp.zeros((size,), dtype=jnp.complex128)
        .at[zero_position]
        .set(0.75 - 0.25j)
    )
    aberrations = (
        jnp.zeros((size,), dtype=jnp.float64).at[zero_position].set(0.3)
    )
    return represented_sources._compose_prepared(
        target,
        additional,
        weights,
        jnp.asarray(1.25, dtype=jnp.float64),
        scan_position=jnp.asarray([0.0, 0.2, -0.1], dtype=jnp.float64),
        aberration_phases=aberrations,
        source_plane_coordinate=jnp.asarray(0.125, dtype=jnp.float64),
        kind=GalerkinLocalRepresentedSourceKind.PLANE_MODE,
        phase_convention=_PHASE,
        source_name=f"{name}-plane",
    )


@functools.lru_cache(maxsize=4)
def _represented_zero(
    name: str = "zero-slab-target",
) -> GalerkinLocalRepresentedSourceCertificate:
    """Return one finite exact-shell represented source with ZERO q."""
    source = _compose_source(name, _zero_additional(name))
    return represented_sources._certify_canonical(source, 21)


@functools.lru_cache(maxsize=2)
def _represented_layer_selective(
    name: str = "zero-slab-target",
) -> GalerkinLocalRepresentedSourceCertificate:
    """Return one finite source whose local q is zero only on layer one."""
    source = _compose_source(name, _layer_selective_additional(name))
    return represented_sources._certify_canonical(source, 21)


@functools.lru_cache(maxsize=1)
def _canonical_slab():
    """Cross the complete public zero-slab trust boundary exactly once."""
    return certify_local_zero_slab(
        _represented_zero(),
        slab_lower_coordinate=_LOWER,
        slab_upper_coordinate=_UPPER,
    )


@functools.lru_cache(maxsize=1)
def _wrapped_represented() -> GalerkinLocalRepresentedSourceCertificate:
    """Build a real wrapped L4 zero block over periodic layers two and zero."""
    fixture_proof = represented_tests._cap_proof()
    core = fixture_proof.coefficient_certificate.absorber.interaction_core
    absorber = realize_axial_cell_absorber(
        core,
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64),
        terminal_axis=0,
        plateau_start=1,
        plateau_count=1,
        plateau_floor=jnp.asarray(1.0, dtype=jnp.float64),
        zero_start=2,
        zero_count=2,
        exact_cap_scale=jnp.asarray(0.25, dtype=jnp.float64),
    )
    cap_certificate = certify_axial_cell_absorber(absorber)
    proof = certify_axial_cap_floor(
        cap_certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    target = compose_local_cell_galerkin_target(
        proof, target_name="zero-slab-wrapped-target"
    )
    additional = local_sources._certify_canonical_source(
        local_sources._realize_zero_prepared(target), 9
    )
    zero_position = int(
        np.flatnonzero(np.all(np.asarray(target.state_indices) == 0, axis=1))[
            0
        ]
    )
    size = target.state_indices.shape[0]
    weights = (
        jnp.zeros((size,), dtype=jnp.complex128)
        .at[zero_position]
        .set(0.5 + 0.25j)
    )
    source = represented_sources._compose_prepared(
        target,
        additional,
        weights,
        jnp.asarray(1.0, dtype=jnp.float64),
        scan_position=jnp.zeros((3,), dtype=jnp.float64),
        aberration_phases=jnp.zeros((size,), dtype=jnp.float64),
        source_plane_coordinate=jnp.asarray(0.0, dtype=jnp.float64),
        kind=GalerkinLocalRepresentedSourceKind.PLANE_MODE,
        phase_convention=_PHASE,
        source_name="zero-slab-wrapped-plane",
    )
    return represented_sources._certify_canonical(source, 21)


def _replace_target_potential(certificate, **changes):
    """Replace the target potential after one genuine parent build."""
    source = certificate.source
    target = source.target
    proof = target.cap_floor_proof
    coefficient = proof.coefficient_certificate
    absorber = coefficient.absorber
    core = absorber.interaction_core
    compression = core.compression
    realization = compression.realization
    potential = replace(realization.local_potential, **changes)
    realization = replace(realization, local_potential=potential)
    compression = replace(compression, realization=realization)
    core = replace(core, compression=compression)
    absorber = replace(absorber, interaction_core=core)
    coefficient = replace(coefficient, absorber=absorber)
    proof = replace(proof, coefficient_certificate=coefficient)
    target = replace(target, cap_floor_proof=proof)
    return replace(certificate, source=replace(source, target=target))


def _replace_target_absorber(certificate, **changes):
    """White-box replace exact L4 profile evidence for predicate testing."""
    source = certificate.source
    target = source.target
    proof = target.cap_floor_proof
    coefficient = proof.coefficient_certificate
    absorber = replace(coefficient.absorber, **changes)
    coefficient = replace(coefficient, absorber=absorber)
    proof = replace(proof, coefficient_certificate=coefficient)
    target = replace(target, cap_floor_proof=proof)
    return replace(certificate, source=replace(source, target=target))


def _replace_free_intervals(certificate, lower, upper):
    """White-box replace exact free-diagonal intervals for shell tests."""
    source = certificate.source
    target = source.target
    ledger = replace(
        target.fixed_linear_error_ledger,
        exact_free_diagonal_lower_bounds=lower,
        exact_free_diagonal_upper_bounds=upper,
    )
    target = replace(target, fixed_linear_error_ledger=ledger)
    return replace(certificate, source=replace(source, target=target))


def _replace_additional_cells(certificate, cells):
    """White-box replace q cells without changing rounded source actions."""
    source = certificate.source
    additional_certificate = source.additional_source_certificate
    additional = replace(
        additional_certificate.source, source_cell_values=cells
    )
    additional_certificate = replace(additional_certificate, source=additional)
    source = replace(
        source, additional_source_certificate=additional_certificate
    )
    return replace(certificate, source=source)


def test_exact_guarded_layer_lift_handles_wrapping_and_faces() -> None:
    """Check exact face guards and canonical wrapped integer lifts.

    :see: :func:`ptyrodactyl.galerkin.certify_local_zero_slab`
    """
    face_guard = _derive_layer_lift(
        Fraction(1),
        Fraction(2),
        Fraction(1, 2),
        Fraction(4),
        4,
        0,
        3,
    )
    assert (face_guard.start, face_guard.stop) == (0, 3)
    assert face_guard.union_lower == 0
    assert face_guard.union_upper == 3
    np.testing.assert_array_equal(face_guard.periodic_indices, [0, 1, 2])
    assert face_guard.cap_zero_block_contains_layers

    positive_lift = _derive_layer_lift(
        Fraction(13, 4),
        Fraction(19, 4),
        Fraction(1, 2),
        Fraction(4),
        4,
        3,
        2,
    )
    negative_lift = _derive_layer_lift(
        Fraction(-3, 4),
        Fraction(3, 4),
        Fraction(1, 2),
        Fraction(4),
        4,
        3,
        2,
    )
    assert (positive_lift.start, positive_lift.stop) == (3, 5)
    assert (negative_lift.start, negative_lift.stop) == (-1, 1)
    np.testing.assert_array_equal(positive_lift.periodic_indices, [3, 0])
    np.testing.assert_array_equal(negative_lift.periodic_indices, [3, 0])
    assert positive_lift.cap_zero_block_lift == 0
    assert negative_lift.cap_zero_block_lift == -1
    assert positive_lift.cap_zero_block_contains_layers
    assert negative_lift.cap_zero_block_contains_layers


def test_wrapped_cap_containment_uses_one_integer_lift_not_modulo_only() -> (
    None
):
    """Reject a modulo-zero set that no single lifted L4 block contains."""
    modulo_only = _derive_layer_lift(
        Fraction(13, 4),
        Fraction(19, 4),
        Fraction(1, 2),
        Fraction(4),
        4,
        0,
        4,
    )
    np.testing.assert_array_equal(modulo_only.periodic_indices, [3, 0])
    assert modulo_only.cap_zero_block_lift == 0
    assert not modulo_only.cap_zero_block_contains_layers


def test_layer_zero_mask_maps_physical_xyz_to_anisotropic_zyx_storage() -> (
    None
):
    """Check all physical axes against anisotropic zyx cell storage."""
    periodic_indices = np.asarray([0, 1], dtype=np.int64)
    for physical_axis in range(3):
        values = np.zeros((2, 3, 4), dtype=np.complex128)
        storage_axis = 2 - physical_axis
        nonzero_position = [0, 0, 0]
        nonzero_position[storage_axis] = 1
        values[tuple(nonzero_position)] = 1.0 + 1.0j
        zero_mask = _layer_zero_mask(
            values,
            physical_axis,
            periodic_indices,
        )
        np.testing.assert_array_equal(zero_mask, [True, False])


def test_vacuum_reference_is_exact_signed_zero_and_literal_only() -> None:
    """Reject offsets and merely similar vacuum-reference wording."""
    canonical = GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value
    assert _vacuum_reference_eligible(0.0, canonical)
    assert _vacuum_reference_eligible(-0.0, canonical)
    assert not _vacuum_reference_eligible(
        float(np.nextafter(np.float64(0.0), np.float64(1.0))),
        canonical,
    )
    assert not _vacuum_reference_eligible(0.0, "vacuum zero")


def test_spatial_zero_remains_independent_of_projection_match() -> None:
    """Retain exact LVT.22 when the represented direct certificate fails."""
    predicates = _eligibility_predicates(
        cap_block_contains=True,
        active_consistent=True,
        vacuum_reference=True,
        potential_zero=True,
        cap_layers_zero=True,
        incident_free_zero=True,
        additional_zero=True,
        projection_match=False,
    )
    assert predicates.exact_spatial_source_zero_eligible
    assert predicates.exact_zero_slab_eligible
    assert not predicates.projection_match_eligible
    assert not predicates.terminal_zero_slab_eligible

    active = np.asarray([True], dtype=np.bool_)
    failure = _failure_mask(
        vacuum_reference=True,
        potential_zero=True,
        cap_block_contains=True,
        cap_layers_zero=True,
        additional_zero=True,
        active_consistent=True,
        active=active,
        declared=active,
        exact_disposition=active,
        exact_shell=active,
        projection_match=False,
    )
    assert failure is (
        GalerkinLocalZeroSlabFailure.REPRESENTED_SOURCE_NONCERTIFICATE
    )

    inconsistent = _eligibility_predicates(
        cap_block_contains=True,
        active_consistent=False,
        vacuum_reference=True,
        potential_zero=True,
        cap_layers_zero=True,
        incident_free_zero=True,
        additional_zero=True,
        projection_match=True,
    )
    assert not inconsistent.incident_free_zero_eligible
    assert not inconsistent.exact_spatial_source_zero_eligible
    assert not inconsistent.terminal_zero_slab_eligible

    inactive = np.asarray([False, False], dtype=np.bool_)
    active = np.asarray([True, False], dtype=np.bool_)
    incident_failures = _failure_mask(
        vacuum_reference=True,
        potential_zero=True,
        cap_block_contains=True,
        cap_layers_zero=True,
        additional_zero=True,
        active_consistent=False,
        active=active,
        declared=inactive,
        exact_disposition=inactive,
        exact_shell=inactive,
        projection_match=True,
    )
    expected_incident_failures = (
        GalerkinLocalZeroSlabFailure.INCIDENT_ACTIVE_MASK_MISMATCH
        | GalerkinLocalZeroSlabFailure.UNDECLARED_INCIDENT_MODE
        | GalerkinLocalZeroSlabFailure.NONEXACT_INCIDENT_DISPOSITION
        | GalerkinLocalZeroSlabFailure.ACTIVE_INCIDENT_OFF_SHELL
    )
    assert incident_failures == expected_incident_failures


def test_target_backed_zero_and_layer_selective_local_cell_success() -> None:
    """Certify genuine ZERO and q-zero-on-slab LVT.20 parent routes."""
    zero = _canonical_slab()
    assert zero.failure_mask == int(GalerkinLocalZeroSlabFailure.NONE)
    assert bool(zero.vacuum_reference_eligible)
    assert bool(zero.potential_zero_eligible)
    assert bool(zero.cap_zero_eligible)
    assert bool(zero.incident_free_zero_eligible)
    assert bool(zero.additional_source_zero_eligible)
    assert bool(zero.exact_spatial_source_zero_eligible)
    assert bool(zero.exact_zero_slab_eligible)
    assert bool(zero.projection_match_eligible)
    assert bool(zero.terminal_zero_slab_eligible)
    np.testing.assert_array_equal(zero.periodic_layer_indices, [1])
    assert (zero.unwrapped_layer_start, zero.unwrapped_layer_stop) == (
        "1",
        "2",
    )
    assert prepare_local_zero_slab_certificate(zero).certificate_digest == (
        zero.certificate_digest
    )

    local = zero_slab._certify_prepared(
        _represented_layer_selective(), _LOWER, _UPPER
    )
    cells = np.asarray(
        local.represented_source_certificate.source.additional_source_certificate.source.source_cell_values
    )
    assert cells[0, 0, 0] != 0.0
    assert cells[0, 0, 1] == 0.0
    assert cells[0, 0, 2] != 0.0
    np.testing.assert_array_equal(
        local.additional_source_layer_zero_mask, [True]
    )
    assert bool(local.terminal_zero_slab_eligible)
    assert local.slab_digest != zero.slab_digest

    alternate_parent = represented_sources._certify_canonical(
        _represented_zero().source, 22
    )
    alternate = zero_slab._certify_prepared(alternate_parent, _LOWER, _UPPER)
    assert alternate.slab_digest == zero.slab_digest
    assert alternate.certificate_digest != zero.certificate_digest


def test_target_backed_wrapped_slab_uses_one_authenticated_cap_lift() -> None:
    """Certify layers two then zero under one genuine wrapped L4 block."""
    wrapped = zero_slab._certify_prepared(
        _wrapped_represented(), np.float64(2.0), np.float64(3.0)
    )
    np.testing.assert_array_equal(wrapped.periodic_layer_indices, [2, 0])
    assert (wrapped.unwrapped_layer_start, wrapped.unwrapped_layer_stop) == (
        "2",
        "4",
    )
    assert wrapped.cap_zero_block_lift == "0"
    assert bool(wrapped.cap_zero_block_contains_layers)
    assert bool(wrapped.cap_zero_eligible)
    assert bool(wrapped.terminal_zero_slab_eligible)
    assert len(wrapped.certificate_digest) == 64


def test_exact_spatial_zero_survives_replayed_direct_noncertificates() -> None:
    """Separate LVT.22 facts from represented/LVT.20 direct work evidence."""
    represented_budget = represented_sources._certify_canonical(
        _represented_zero().source, 1
    )
    slab = certify_local_zero_slab(
        represented_budget,
        slab_lower_coordinate=_LOWER,
        slab_upper_coordinate=_UPPER,
    )
    assert bool(slab.exact_zero_slab_eligible)
    assert not bool(slab.projection_match_eligible)
    assert not bool(slab.terminal_zero_slab_eligible)
    assert int(slab.failure_mask) == int(
        GalerkinLocalZeroSlabFailure.REPRESENTED_SOURCE_NONCERTIFICATE
    )

    additional_budget = local_sources._certify_canonical_source(
        _layer_selective_additional().source, 1
    )
    represented_source = _compose_source("zero-slab-target", additional_budget)
    represented = represented_sources._certify_canonical(
        represented_source, 21
    )
    slab = zero_slab._certify_prepared(represented, _LOWER, _UPPER)
    assert bool(slab.additional_source_zero_eligible)
    assert bool(slab.exact_zero_slab_eligible)
    assert not bool(slab.projection_match_eligible)
    assert not bool(slab.terminal_zero_slab_eligible)


@pytest.mark.parametrize("component", ["potential", "cap", "q_real", "q_imag"])
def test_one_bit_spatial_factor_adversaries_fail_without_cancellation(
    component: str,
) -> None:
    """Reject a one-bit nonzero in each separately proved LVT.22 factor."""
    one_bit = np.nextafter(np.float64(0.0), np.float64(1.0))
    base = (
        _represented_layer_selective()
        if component.startswith("q_")
        else _represented_zero()
    )
    if component == "potential":
        potential_values = base.source.target.local_potential.cell_values
        values = potential_values.at[0, 0, 1].set(one_bit)
        changed = _replace_target_potential(base, cell_values=values)
    elif component == "cap":
        absorber = (
            base.source.target.cap_floor_proof.coefficient_certificate.absorber
        )
        values = absorber.layer_values.at[1].set(one_bit)
        changed = _replace_target_absorber(base, layer_values=values)
    else:
        cells = (
            base.source.additional_source_certificate.source.source_cell_values
        )
        value = one_bit + 0.0j if component == "q_real" else 1j * one_bit
        changed = _replace_additional_cells(base, cells.at[0, 0, 1].set(value))
    slab = zero_slab._certify_prepared(changed, _LOWER, _UPPER)
    assert not bool(slab.exact_zero_slab_eligible)
    assert not bool(slab.terminal_zero_slab_eligible)
    expected = {
        "potential": GalerkinLocalZeroSlabFailure.POTENTIAL_NONZERO,
        "cap": GalerkinLocalZeroSlabFailure.CAP_NONZERO,
        "q_real": GalerkinLocalZeroSlabFailure.ADDITIONAL_SOURCE_NONZERO,
        "q_imag": GalerkinLocalZeroSlabFailure.ADDITIONAL_SOURCE_NONZERO,
    }[component]
    assert int(slab.failure_mask) & int(expected)


@pytest.mark.parametrize(
    ("reference_value", "reference_semantics"),
    [
        (
            np.nextafter(np.float64(0.0), np.float64(1.0)),
            GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value,
        ),
        (np.float64(0.0), "unresolved zero reference"),
        (np.float64(0.0), "material-relative zero reference"),
    ],
)
def test_reference_offset_or_nonvacuum_meaning_cannot_claim_vacuum(
    reference_value: np.float64,
    reference_semantics: str,
) -> None:
    """Require both numeric zero and the exact SC.2/SC.8 vacuum literal."""
    changed = _replace_target_potential(
        _represented_zero(),
        reference_value=float(reference_value),
        reference_semantics=reference_semantics,
    )
    slab = zero_slab._certify_prepared(changed, _LOWER, _UPPER)
    assert bool(slab.potential_zero_eligible)
    assert not bool(slab.vacuum_reference_eligible)
    assert not bool(slab.exact_zero_slab_eligible)
    assert int(slab.failure_mask) & int(
        GalerkinLocalZeroSlabFailure.VACUUM_REFERENCE_UNDECLARED
    )


def test_active_off_shell_fails_while_inactive_off_shell_is_allowed() -> None:
    """Use exact free intervals, never rounded D values or tolerances."""
    base = _represented_zero()
    active = int(np.flatnonzero(np.asarray(base.source.modes.active_mask))[0])
    inactive = int(
        np.flatnonzero(~np.asarray(base.source.modes.active_mask))[0]
    )
    ledger = base.source.target.fixed_linear_error_ledger

    lower = ledger.exact_free_diagonal_lower_bounds.at[active].set(-1.0e-30)
    upper = ledger.exact_free_diagonal_upper_bounds.at[active].set(1.0e-30)
    active_off_shell = _replace_free_intervals(base, lower, upper)
    incident = _incident_predicates(active_off_shell)
    assert not bool(incident[3][active])
    slab = zero_slab._certify_prepared(active_off_shell, _LOWER, _UPPER)
    assert not bool(slab.incident_free_zero_eligible)
    assert int(slab.failure_mask) & int(
        GalerkinLocalZeroSlabFailure.ACTIVE_INCIDENT_OFF_SHELL
    )

    lower = ledger.exact_free_diagonal_lower_bounds.at[inactive].set(1.0)
    upper = ledger.exact_free_diagonal_upper_bounds.at[inactive].set(1.0)
    inactive_off_shell = _replace_free_intervals(base, lower, upper)
    slab = zero_slab._certify_prepared(inactive_off_shell, _LOWER, _UPPER)
    assert bool(slab.incident_free_zero_eligible)
    assert bool(slab.terminal_zero_slab_eligible)


def test_total_source_cancellation_is_never_a_spatial_zero_oracle() -> None:
    """Reject q nonzero on the slab even if a total vector is forged zero."""
    base = _represented_layer_selective()
    cells = base.source.additional_source_certificate.source.source_cell_values
    changed = _replace_additional_cells(
        base, cells.at[0, 0, 1].set(1.0 - 2.0j)
    )
    source = changed.source
    cancelled_actions = source.actions._replace(
        total_source=jnp.zeros_like(source.actions.total_source)
    )
    changed = replace(
        changed, source=replace(source, actions=cancelled_actions)
    )
    slab = zero_slab._certify_prepared(changed, _LOWER, _UPPER)
    assert not bool(slab.additional_source_zero_eligible)
    assert not bool(slab.exact_zero_slab_eligible)


def test_geometry_range_face_and_full_transverse_adversaries() -> None:
    """Reject invalid widths and require complete guarded cell layers."""
    parent = _represented_zero()
    for lower, upper in (
        (_LOWER, _LOWER),
        (_UPPER, _LOWER),
        (np.float64(0.0), np.float64(3.25)),
    ):
        with pytest.raises(ValueError, match="width"):
            zero_slab._certify_prepared(parent, lower, upper)
    with pytest.raises(ValueError, match="normal-or-zero"):
        zero_slab._certify_prepared(
            parent,
            np.nextafter(np.float64(0.0), np.float64(1.0)),
            _UPPER,
        )

    face_guard = _derive_layer_lift(
        Fraction(1),
        Fraction(3, 2),
        Fraction(1, 2),
        Fraction(4),
        4,
        1,
        1,
    )
    assert (face_guard.start, face_guard.stop) == (0, 2)
    assert not face_guard.cap_zero_block_contains_layers

    for physical_axis in range(3):
        storage_axis = 2 - physical_axis
        values = np.zeros((2, 3, 4), dtype=np.float64)
        position = [0, 0, 0]
        position[storage_axis] = 1
        for transverse_axis in range(3):
            if transverse_axis != storage_axis:
                position[transverse_axis] = values.shape[transverse_axis] - 1
        values[tuple(position)] = 1.0
        mask = _layer_zero_mask(
            values,
            physical_axis,
            np.asarray([0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(mask, [True, False])


def test_signed_zero_is_eligible_but_remains_identity_distinct() -> None:
    """Treat signed q zeros numerically while digesting their stored bytes."""
    target = represented_tests._target("zero-slab-target")
    positive_cells = np.zeros((1, 1, 3), dtype=np.complex128)
    negative_cells = positive_cells.copy()
    negative_cells[0, 0, 1] = complex(-0.0, 0.0)
    additional_certificates = []
    slabs = []
    for name, cells in (
        ("positive-zero", positive_cells),
        ("negative-zero", negative_cells),
    ):
        additional_source = local_sources._realize_local_cell_prepared(
            target, jnp.asarray(cells)
        )
        additional = local_sources._certify_canonical_source(
            additional_source, 9
        )
        additional_certificates.append(additional)
        represented = represented_sources._certify_canonical(
            _compose_source("zero-slab-target", additional), 21
        )
        slab = zero_slab._certify_prepared(represented, _LOWER, _UPPER)
        assert bool(slab.additional_source_zero_eligible), name
        assert bool(slab.terminal_zero_slab_eligible), name
        slabs.append(slab)
    assert additional_certificates[0].source.source_digest != (
        additional_certificates[1].source.source_digest
    )
    assert slabs[0].slab_digest != slabs[1].slab_digest


def test_complete_replay_rejects_geometry_mask_digest_and_cross_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject local edits, valid-looking rehashes, and parent cross-pairs."""
    canonical = _canonical_slab()
    with monkeypatch.context() as patch:
        patch.setattr(
            zero_slab,
            "_prepare_represented_source_certificate",
            lambda _: canonical.represented_source_certificate,
        )
        for forged in (
            replace(canonical, unwrapped_layer_start="0"),
            replace(
                canonical,
                potential_layer_zero_mask=(
                    canonical.potential_layer_zero_mask.at[0].set(False)
                ),
            ),
            replace(
                canonical,
                failure_mask=jnp.asarray(
                    int(GalerkinLocalZeroSlabFailure.POTENTIAL_NONZERO),
                    dtype=jnp.int64,
                ),
            ),
            replace(canonical, certificate_digest="a" * 64),
        ):
            with pytest.raises(ValueError, match="complete host replay"):
                prepare_local_zero_slab_certificate(forged)

    cross_parent = zero_slab._certify_prepared(
        _represented_zero("zero-slab-cross-parent"), _LOWER, _UPPER
    )
    assert canonical.target_digest == cross_parent.target_digest
    assert canonical.slab_digest == cross_parent.slab_digest
    assert canonical.represented_source_digest == (
        cross_parent.represented_source_digest
    )
    assert canonical.parent_target_evidence_digest != (
        cross_parent.parent_target_evidence_digest
    )

    forged_potential_mask = np.asarray(
        canonical.potential_layer_zero_mask
    ).copy()
    forged_potential_mask[0] = False
    forged_predicates = (
        bool(canonical.cap_zero_block_contains_layers),
        bool(canonical.incident_active_mask_consistent),
        bool(canonical.vacuum_reference_eligible),
        False,
        bool(canonical.cap_zero_eligible),
        bool(canonical.incident_free_zero_eligible),
        bool(canonical.additional_source_zero_eligible),
        bool(canonical.exact_spatial_source_zero_eligible),
        False,
        bool(canonical.projection_match_eligible),
        False,
    )
    forged_failure = GalerkinLocalZeroSlabFailure.POTENTIAL_NONZERO
    rehashed_digest = zero_slab._certificate_digest(
        cross_parent.represented_source_certificate,
        canonical.slab_digest,
        (
            forged_potential_mask,
            np.asarray(canonical.cap_layer_zero_mask),
            np.asarray(canonical.additional_source_layer_zero_mask),
        ),
        (
            np.asarray(canonical.incident_active_mask),
            np.asarray(canonical.incident_declared_mask),
            np.asarray(canonical.incident_exact_disposition_mask),
            np.asarray(canonical.incident_exact_shell_mask),
            bool(canonical.incident_active_mask_consistent),
        ),
        forged_predicates,
        forged_failure,
    )
    self_rehashed_cross_parent = replace(
        canonical,
        represented_source_certificate=(
            cross_parent.represented_source_certificate
        ),
        potential_layer_zero_mask=jnp.asarray(forged_potential_mask),
        potential_zero_eligible=jnp.asarray(False),
        exact_zero_slab_eligible=jnp.asarray(False),
        terminal_zero_slab_eligible=jnp.asarray(False),
        failure_mask=jnp.asarray(int(forged_failure), dtype=jnp.int64),
        parent_target_evidence_digest=(
            cross_parent.parent_target_evidence_digest
        ),
        represented_source_digest=cross_parent.represented_source_digest,
        parent_source_evidence_digest=(
            cross_parent.parent_source_evidence_digest
        ),
        parent_represented_certificate_digest=(
            cross_parent.parent_represented_certificate_digest
        ),
        certificate_digest=rehashed_digest,
    )
    with pytest.raises(ValueError, match="complete host replay"):
        prepare_local_zero_slab_certificate(self_rehashed_cross_parent)


def test_zero_slab_public_boundary_consumes_only_represented_certificate() -> (
    None
):
    """Exclude raw source, target, vector, and redundant layer inputs.

    :see: :func:`ptyrodactyl.galerkin.prepare_local_zero_slab_certificate`
    """
    certify_parameters = inspect.signature(certify_local_zero_slab).parameters
    assert tuple(certify_parameters) == (
        "represented_source_certificate",
        "slab_lower_coordinate",
        "slab_upper_coordinate",
    )
    assert certify_parameters["slab_lower_coordinate"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert certify_parameters["slab_upper_coordinate"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    prepare_parameters = inspect.signature(
        prepare_local_zero_slab_certificate
    ).parameters
    assert tuple(prepare_parameters) == ("certificate",)
