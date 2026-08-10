"""Tests for :mod:`ptyrodactyl.galerkin.acquisition`.

Extended Summary
----------------
These tests compare the production constant-memory RM-S1 predicates with a
small independent Python-set oracle. They separately exercise structural,
support-ineligible, and support-eligible outcomes and audit the physical
direction, sector, and geometry evidence.
"""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple

from ptyrodactyl.galerkin.acquisition import (
    check_galerkin_acquisition_support,
)
from ptyrodactyl.types import acquisition_types
from ptyrodactyl.types.acquisition_types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportFailure,
    GalerkinAcquisitionSupportStatus,
    GalerkinBackwardDisposition,
    GalerkinCarrierOverlapDisposition,
    GalerkinCarrierOwnership,
    GalerkinCarrierTargetRoute,
    GalerkinDirectionDisposition,
    GalerkinEndpointConvention,
    GalerkinTerminalSide,
    create_galerkin_acquisition_manifest,
)
from ptyrodactyl.types.born_potential_types import GalerkinProductSupport


def _indices(
    x_values: range | Tuple[int, ...],
    z_values: range | Tuple[int, ...],
) -> jax.Array:
    """Return exact three-dimensional indices on the ``y=0`` plane."""
    return jnp.asarray(
        [
            (x_value, 0, z_value)
            for x_value in x_values
            for z_value in z_values
        ],
        dtype=jnp.int64,
    )


def _support(
    *,
    interaction: jax.Array | None = None,
    absorber: jax.Array | None = None,
    work: jax.Array | None = None,
) -> GalerkinProductSupport:
    """Return one small exact RM-S1 support submission."""
    return GalerkinProductSupport(
        state_indices=_indices(range(-1, 2), range(-1, 2)),
        interaction_indices=(
            _indices((0,), range(-1, 2))
            if interaction is None
            else jnp.asarray(interaction, dtype=jnp.int64)
        ),
        absorber_indices=(
            _indices(range(-2, 3), range(-2, 3))
            if absorber is None
            else jnp.asarray(absorber, dtype=jnp.int64)
        ),
        work_indices=(
            _indices(range(-1, 2), range(-2, 3))
            if work is None
            else jnp.asarray(work, dtype=jnp.int64)
        ),
        work_shape=(5, 1, 5),
    )


def _canonical_wavevectors(
    indices: jax.Array,
    carrier: jax.Array,
    box_lengths: jax.Array,
) -> jax.Array:
    """Return the production canonical binary64 coefficient realization."""
    return (
        carrier[None, :]
        + (2.0 * jnp.asarray(jnp.pi, dtype=jnp.float64))
        * indices.astype(jnp.float64)
        / box_lengths[None, :]
    )


def _manifest(  # noqa: PLR0913
    *,
    support: GalerkinProductSupport | None = None,
    incident: jax.Array | None = None,
    outgoing: jax.Array | None = None,
    preterminal: jax.Array | None = None,
    transverse: jax.Array | None = None,
    omitted: jax.Array | None = None,
    incident_wavevectors: jax.Array | None = None,
    outgoing_wavevectors: jax.Array | None = None,
    incident_dispositions: jax.Array | None = None,
    outgoing_dispositions: jax.Array | None = None,
    incident_shell_bounds: jax.Array | None = None,
    outgoing_shell_bounds: jax.Array | None = None,
    incident_projection_bounds: jax.Array | None = None,
    outgoing_projection_bounds: jax.Array | None = None,
    carrier: jax.Array | None = None,
    box_lengths: jax.Array | None = None,
    wavenumber: jax.Array | None = None,
    carrier_shell_bound: jax.Array | None = None,
    shell_tolerance: jax.Array | None = None,
    terminal_axis: int = 2,
    terminal_side: GalerkinTerminalSide = GalerkinTerminalSide.POSITIVE,
    disposition: GalerkinBackwardDisposition = (
        GalerkinBackwardDisposition.EXCLUDED
    ),
    exclusion_basis: str = "forward-sector projection; no backscatter claim",
    claims_backscatter: bool = False,
) -> GalerkinAcquisitionManifest:
    """Return one complete bounded acquisition support submission."""
    incident_indices = (
        jnp.asarray([[0, 0, 0]], dtype=jnp.int64)
        if incident is None
        else jnp.asarray(incident, dtype=jnp.int64)
    )
    outgoing_indices = (
        jnp.asarray([[0, 0, 0]], dtype=jnp.int64)
        if outgoing is None
        else jnp.asarray(outgoing, dtype=jnp.int64)
    )
    carrier_array = (
        jnp.asarray([0.0, 0.0, 10.0], dtype=jnp.float64)
        if carrier is None
        else jnp.asarray(carrier, dtype=jnp.float64)
    )
    box_array = (
        jnp.ones((3,), dtype=jnp.float64)
        if box_lengths is None
        else jnp.asarray(box_lengths, dtype=jnp.float64)
    )
    incident_physical = (
        _canonical_wavevectors(incident_indices, carrier_array, box_array)
        if incident_wavevectors is None
        else jnp.asarray(incident_wavevectors, dtype=jnp.float64)
    )
    outgoing_physical = (
        _canonical_wavevectors(outgoing_indices, carrier_array, box_array)
        if outgoing_wavevectors is None
        else jnp.asarray(outgoing_wavevectors, dtype=jnp.float64)
    )
    incident_count = incident_indices.shape[0]
    outgoing_count = outgoing_indices.shape[0]

    return create_galerkin_acquisition_manifest(
        _support() if support is None else support,
        incident_indices,
        outgoing_indices,
        (_indices((0,), range(-1, 2)) if preterminal is None else preterminal),
        (
            jnp.asarray([[0, 0]], dtype=jnp.int64)
            if transverse is None
            else transverse
        ),
        (jnp.zeros((0, 3), dtype=jnp.int64) if omitted is None else omitted),
        incident_physical_wavevectors=incident_physical,
        outgoing_physical_wavevectors=outgoing_physical,
        incident_direction_dispositions=(
            jnp.full(
                (incident_count,),
                GalerkinDirectionDisposition.EXACT_COEFFICIENT,
                dtype=jnp.int32,
            )
            if incident_dispositions is None
            else incident_dispositions
        ),
        outgoing_direction_dispositions=(
            jnp.full(
                (outgoing_count,),
                GalerkinDirectionDisposition.EXACT_COEFFICIENT,
                dtype=jnp.int32,
            )
            if outgoing_dispositions is None
            else outgoing_dispositions
        ),
        incident_on_shell_defect_bounds=(
            jnp.full((incident_count,), 1.0e-10, dtype=jnp.float64)
            if incident_shell_bounds is None
            else incident_shell_bounds
        ),
        outgoing_on_shell_defect_bounds=(
            jnp.full((outgoing_count,), 1.0e-10, dtype=jnp.float64)
            if outgoing_shell_bounds is None
            else outgoing_shell_bounds
        ),
        incident_projection_error_bounds=(
            jnp.zeros((incident_count,), dtype=jnp.float64)
            if incident_projection_bounds is None
            else incident_projection_bounds
        ),
        outgoing_projection_error_bounds=(
            jnp.zeros((outgoing_count,), dtype=jnp.float64)
            if outgoing_projection_bounds is None
            else outgoing_projection_bounds
        ),
        carrier=carrier_array,
        box_lengths=box_array,
        wavenumber=(
            jnp.linalg.norm(carrier_array)
            if wavenumber is None
            else wavenumber
        ),
        carrier_on_shell_defect_bound=(
            jnp.asarray(1.0e-10, dtype=jnp.float64)
            if carrier_shell_bound is None
            else carrier_shell_bound
        ),
        on_shell_defect_tolerance=(
            jnp.asarray(1.0e-8, dtype=jnp.float64)
            if shell_tolerance is None
            else shell_tolerance
        ),
        terminal_axis=terminal_axis,
        terminal_side=terminal_side,
        carrier_id="forward_0",
        carrier_ownership=(
            GalerkinCarrierOwnership.INDEPENDENT_SINGLE_CARRIER
        ),
        carrier_overlap_disposition=(
            GalerkinCarrierOverlapDisposition.NO_OTHER_CARRIER_BLOCKS
        ),
        carrier_target_route=(
            GalerkinCarrierTargetRoute.NORMALIZE_FROM_ACCELERATING_VOLTAGE
        ),
        endpoint_convention=GalerkinEndpointConvention.SIGNED_HALF_OPEN,
        backward_disposition=disposition,
        backward_exclusion_basis=exclusion_basis,
        claims_backscatter=claims_backscatter,
    )


def _as_set(values: jax.Array) -> set[Tuple[int, ...]]:
    """Convert one bounded integer array to an independent Python set."""
    return {
        tuple(int(component) for component in row)
        for row in np.asarray(values)
    }


def _bounded_oracle(
    manifest: GalerkinAcquisitionManifest,
) -> Dict[str, bool | int]:
    """Evaluate the core RM-S1 relations with independent Python sets."""
    state = _as_set(manifest.support.state_indices)
    interaction = _as_set(manifest.support.interaction_indices)
    absorber = _as_set(manifest.support.absorber_indices)
    work = _as_set(manifest.support.work_indices)
    incident = _as_set(manifest.incident_indices)
    outgoing = _as_set(manifest.elastic_outgoing_indices)
    preterminal = _as_set(manifest.preterminal_indices)
    transverse = _as_set(manifest.transverse_indices)

    transfers = {
        tuple(left - right for left, right in zip(output, source, strict=True))
        for output in preterminal
        for source in incident
    }
    absorber_differences = {
        tuple(left - right for left, right in zip(first, second, strict=True))
        for first in state
        for second in state
    }
    work_products = {
        tuple(left + right for left, right in zip(first, second, strict=True))
        for first in state
        for second in interaction
    }
    expected_preterminal = {
        point
        for point in state
        if tuple(
            component
            for axis, component in enumerate(point)
            if axis != manifest.terminal_axis
        )
        in transverse
    }
    result: Dict[str, bool | int] = {
        "incident_in_state": incident <= state,
        "outgoing_in_preterminal": outgoing <= preterminal,
        "preterminal_in_state": preterminal <= state,
        "direct_transfers_represented": transfers <= interaction,
        "absorber_differences_represented": absorber_differences <= absorber,
        "work_products_represented": work_products <= work,
        "interaction_sign_symmetric": {
            tuple(-component for component in point) for point in interaction
        }
        <= interaction,
        "absorber_sign_symmetric": {
            tuple(-component for component in point) for point in absorber
        }
        <= absorber,
        "terminal_fiber_complete": preterminal == expected_preterminal,
        "direct_transfer_pair_count": len(preterminal) * len(incident),
        "represented_direct_transfer_pair_count": sum(
            tuple(
                left - right
                for left, right in zip(output, source, strict=True)
            )
            in interaction
            for output in preterminal
            for source in incident
        ),
    }
    return result


def _has_failure(
    mask: jax.Array, failure: GalerkinAcquisitionSupportFailure
) -> bool:
    """Return whether one typed failure bit is present."""
    return bool(int(mask) & int(failure))


class TestGalerkinAcquisitionSupport:
    (
        """Verify the production RM-S1 finite-support eligibility seam.

    :see: :func:`ptyrodactyl.galerkin.check_galerkin_acquisition_support`
    :see: :class:`ptyrodactyl.types."""
        """GalerkinAcquisitionSupportFailure`
    :see: :class:`ptyrodactyl.types."""
        """GalerkinAcquisitionSupportResult`
    :see: :class:`ptyrodactyl.types."""
        """GalerkinAcquisitionSupportStatus`
    :see: :class:`ptyrodactyl.types.GalerkinAcquisitionManifest`
    :see: :class:`ptyrodactyl.types.GalerkinBackwardDisposition`
    :see: :class:`ptyrodactyl.types."""
        """GalerkinCarrierOverlapDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinCarrierOwnership`
    :see: :class:`ptyrodactyl.types.GalerkinCarrierTargetRoute`
    :see: :class:`ptyrodactyl.types.GalerkinDirectionDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinEndpointConvention`
    :see: :class:`ptyrodactyl.types.GalerkinTerminalSide`
    :see: :func:`ptyrodactyl.types.create_galerkin_acquisition_manifest`
    """
    )

    def test_positive_manifest_is_support_eligible_and_exact_width(
        self,
    ) -> None:
        """Bind every required set, declaration, predicate, and count."""
        manifest = _manifest()
        result = check_galerkin_acquisition_support(manifest)

        for values in (
            manifest.incident_indices,
            manifest.elastic_outgoing_indices,
            manifest.preterminal_indices,
            manifest.transverse_indices,
            manifest.deliberately_omitted_indices,
        ):
            assert values.dtype == jnp.int64
        for values in (
            manifest.incident_physical_wavevectors,
            manifest.outgoing_physical_wavevectors,
            manifest.carrier,
            manifest.box_lengths,
            manifest.wavenumber,
            result.carrier_shell_defect_upper_bound,
            result.incident_projection_error_upper_bounds,
        ):
            assert values.dtype == jnp.float64
        assert manifest.incident_direction_dispositions.dtype == jnp.int32
        assert result.status.dtype == jnp.int32
        assert result.failure_mask.dtype == jnp.int64
        assert result.direct_transfer_pair_count.dtype == jnp.int64
        assert result.status == (
            GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE
        )
        assert result.structural_valid
        assert result.support_eligible
        assert result.failure_mask == 0
        assert result.direct_transfer_pair_count == 3
        assert result.represented_direct_transfer_pair_count == 3
        assert result.max_binary_pair_checks == 20_000_000
        assert jnp.all(result.state_forward_mask)
        assert not jnp.any(result.state_grazing_mask)
        assert not jnp.any(result.state_backward_mask)
        assert not jnp.any(result.state_ambiguous_mask)
        assert jnp.all(result.state_oriented_normal_interval_lower > 0.0)
        assert jnp.all(
            result.state_oriented_normal_interval_upper
            >= result.state_oriented_normal_interval_lower
        )
        assert manifest.carrier_id == "forward_0"
        assert manifest.direct_transfer_rule == "K_d-Q_in subset K_chi"
        assert manifest.terminal_axis == 2
        assert manifest.terminal_side == GalerkinTerminalSide.POSITIVE
        assert manifest.carrier_target_route == (
            GalerkinCarrierTargetRoute.NORMALIZE_FROM_ACCELERATING_VOLTAGE
        )

    def test_core_predicates_match_small_independent_oracle(self) -> None:
        """Match an independent Python-set oracle on adversarial data."""
        cases = (
            _manifest(),
            _manifest(incident=jnp.asarray([[2, 0, 0]], dtype=jnp.int64)),
            _manifest(outgoing=jnp.asarray([[1, 0, 0]], dtype=jnp.int64)),
            _manifest(
                preterminal=_indices((0,), (-1, 0)),
                outgoing=jnp.asarray([[0, 0, 0]], dtype=jnp.int64),
            ),
            _manifest(support=_support(interaction=_indices((0,), (0,)))),
            _manifest(
                support=_support(absorber=_indices(range(-1, 2), range(-2, 3)))
            ),
            _manifest(
                support=_support(
                    work=_indices(range(-1, 2), range(-2, 3))[:-1]
                )
            ),
        )
        checked_fields = (
            "incident_in_state",
            "outgoing_in_preterminal",
            "preterminal_in_state",
            "direct_transfers_represented",
            "absorber_differences_represented",
            "work_products_represented",
            "interaction_sign_symmetric",
            "absorber_sign_symmetric",
            "terminal_fiber_complete",
            "direct_transfer_pair_count",
            "represented_direct_transfer_pair_count",
        )

        for manifest in cases:
            expected = _bounded_oracle(manifest)
            actual = check_galerkin_acquisition_support(manifest)
            for field in checked_fields:
                assert int(getattr(actual, field)) == int(expected[field])

    def test_duplicate_submission_is_structurally_invalid(self) -> None:
        """Distinguish a repeated set entry from support ineligibility."""
        duplicate = jnp.asarray([[0, 0, 0], [0, 0, 0]], dtype=jnp.int64)
        result = check_galerkin_acquisition_support(
            _manifest(incident=duplicate)
        )

        assert not result.structural_valid
        assert not result.support_eligible
        assert result.status == (
            GalerkinAcquisitionSupportStatus.STRUCTURALLY_INVALID
        )
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.DUPLICATE_INDEX,
        )

    def test_noncanonical_endpoint_is_structurally_invalid(self) -> None:
        """Reject the positive endpoint outside a five-point half-open axis."""
        result = check_galerkin_acquisition_support(
            _manifest(incident=jnp.asarray([[3, 0, 0]], dtype=jnp.int64))
        )

        assert result.status == (
            GalerkinAcquisitionSupportStatus.STRUCTURALLY_INVALID
        )
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.ENDPOINT_CONFLICT,
        )

    def test_missing_mode_is_structural_but_support_ineligible(self) -> None:
        """Classify a valid set that misses state and transfer support."""
        result = check_galerkin_acquisition_support(
            _manifest(incident=jnp.asarray([[2, 0, 0]], dtype=jnp.int64))
        )

        assert result.structural_valid
        assert not result.support_eligible
        assert result.status == (
            GalerkinAcquisitionSupportStatus.SUPPORT_INELIGIBLE
        )
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.INCIDENT_OUTSIDE_STATE,
        )
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.DIRECT_TRANSFER_MISSING,
        )

    def test_incomplete_preterminal_fiber_fails_closed(self) -> None:
        """Require every normal coefficient in a selected transverse mode."""
        result = check_galerkin_acquisition_support(
            _manifest(
                preterminal=_indices((0,), (-1, 0)),
                outgoing=jnp.asarray([[0, 0, 0]], dtype=jnp.int64),
            )
        )

        assert not result.terminal_fiber_complete
        assert result.status == (
            GalerkinAcquisitionSupportStatus.SUPPORT_INELIGIBLE
        )
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.TERMINAL_FIBER_MISMATCH,
        )

    def test_backward_declaration_uses_complete_computed_masks(self) -> None:
        """Admit a represented backward sector and reject a false claim."""
        carrier = jnp.asarray([0.0, 0.0, 5.0], dtype=jnp.float64)
        represented = check_galerkin_acquisition_support(
            _manifest(
                carrier=carrier,
                wavenumber=jnp.asarray(5.0, dtype=jnp.float64),
                disposition=GalerkinBackwardDisposition.REPRESENTED,
                exclusion_basis="",
            )
        )
        contradicted = check_galerkin_acquisition_support(
            _manifest(claims_backscatter=True)
        )

        assert represented.backward_disposition_valid
        assert represented.support_eligible
        assert int(jnp.sum(represented.state_backward_mask)) == 3
        assert int(jnp.sum(represented.state_forward_mask)) == 6
        assert not contradicted.backward_disposition_valid
        assert contradicted.status == (
            GalerkinAcquisitionSupportStatus.SUPPORT_INELIGIBLE
        )

    def test_deliberately_omitted_backward_modes_are_explicit(self) -> None:
        """Classify omitted modes and reject an overstated scatter claim."""
        carrier = jnp.asarray([0.0, 0.0, 5.0], dtype=jnp.float64)
        omitted = jnp.asarray([[0, 0, -2]], dtype=jnp.int64)
        result = check_galerkin_acquisition_support(
            _manifest(
                carrier=carrier,
                wavenumber=jnp.asarray(5.0, dtype=jnp.float64),
                omitted=omitted,
                disposition=GalerkinBackwardDisposition.REPRESENTED,
                exclusion_basis="",
                claims_backscatter=True,
            )
        )

        assert result.omitted_mask_valid
        assert bool(result.omitted_backward_mask[0])
        assert not result.backward_disposition_valid
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.BACKWARD_DISPOSITION_INVALID,
        )

    def test_grazing_is_exact_and_interval_ambiguity_fails_closed(
        self,
    ) -> None:
        """Never turn a zero-containing interval into tolerance grazing."""
        transverse_carrier = jnp.asarray([10.0, 0.0, 0.0], dtype=jnp.float64)
        grazing = check_galerkin_acquisition_support(
            _manifest(
                carrier=transverse_carrier,
                wavenumber=jnp.asarray(10.0, dtype=jnp.float64),
                disposition=GalerkinBackwardDisposition.REPRESENTED,
                exclusion_basis="",
            )
        )
        cancellation_carrier = jnp.asarray(
            [0.0, 0.0, 2.0 * np.pi], dtype=jnp.float64
        )
        ambiguous = check_galerkin_acquisition_support(
            _manifest(
                carrier=cancellation_carrier,
                wavenumber=cancellation_carrier[2],
            )
        )

        assert int(jnp.sum(grazing.state_grazing_mask)) == 3
        assert not jnp.any(grazing.state_ambiguous_mask)
        assert jnp.all(
            grazing.state_oriented_normal_interval_lower[
                grazing.state_grazing_mask
            ]
            == 0.0
        )
        assert jnp.all(
            grazing.state_oriented_normal_interval_upper[
                grazing.state_grazing_mask
            ]
            == 0.0
        )
        assert grazing.support_eligible
        assert jnp.any(ambiguous.state_ambiguous_mask)
        assert jnp.all(
            ambiguous.state_oriented_normal_interval_lower[
                ambiguous.state_ambiguous_mask
            ]
            <= 0.0
        )
        assert jnp.all(
            ambiguous.state_oriented_normal_interval_upper[
                ambiguous.state_ambiguous_mask
            ]
            >= 0.0
        )
        assert not ambiguous.sector_classification_complete
        assert _has_failure(
            ambiguous.failure_mask,
            GalerkinAcquisitionSupportFailure.SECTOR_CLASSIFICATION_AMBIGUOUS,
        )

    def test_projected_direction_requires_outward_error_evidence(self) -> None:
        """Admit a bounded projection and reject an understated bound."""
        projected_wavevector = jnp.asarray(
            [[0.0, 0.0, 9.999]], dtype=jnp.float64
        )
        projected_code = jnp.asarray(
            [GalerkinDirectionDisposition.PROJECTED], dtype=jnp.int32
        )
        admitted = check_galerkin_acquisition_support(
            _manifest(
                incident_wavevectors=projected_wavevector,
                incident_dispositions=projected_code,
                incident_shell_bounds=jnp.asarray([0.1], dtype=jnp.float64),
                shell_tolerance=jnp.asarray(0.1, dtype=jnp.float64),
                incident_projection_bounds=jnp.asarray(
                    [0.01], dtype=jnp.float64
                ),
            )
        )
        understated = check_galerkin_acquisition_support(
            _manifest(
                incident_wavevectors=projected_wavevector,
                incident_dispositions=projected_code,
                incident_shell_bounds=jnp.asarray([0.1], dtype=jnp.float64),
                shell_tolerance=jnp.asarray(0.1, dtype=jnp.float64),
                incident_projection_bounds=jnp.asarray(
                    [1.0e-6], dtype=jnp.float64
                ),
            )
        )

        assert admitted.direction_evidence_valid
        assert admitted.support_eligible
        assert admitted.incident_projection_error_upper_bounds[0] > 0.0
        assert admitted.incident_projection_error_upper_bounds[0] < 0.01
        assert not understated.direction_evidence_valid
        assert _has_failure(
            understated.failure_mask,
            GalerkinAcquisitionSupportFailure.DIRECTION_EVIDENCE_INVALID,
        )

    def test_exact_direction_requires_canonical_binary64_roundtrip(
        self,
    ) -> None:
        """Keep exact lattice identity primary and reject altered metadata."""
        altered = jnp.asarray([[0.0, 0.0, 9.999]], dtype=jnp.float64)
        result = check_galerkin_acquisition_support(
            _manifest(
                incident_wavevectors=altered,
                incident_shell_bounds=jnp.asarray([0.1], dtype=jnp.float64),
                shell_tolerance=jnp.asarray(0.1, dtype=jnp.float64),
            )
        )

        assert not result.direction_evidence_valid
        assert _has_failure(
            result.failure_mask,
            GalerkinAcquisitionSupportFailure.DIRECTION_EVIDENCE_INVALID,
        )

    def test_nonzero_elastic_coefficient_is_projected_without_witness(
        self,
    ) -> None:
        """Fail closed on a nonzero exact-shell claim without a witness."""
        outgoing = jnp.asarray([[0, 0, 1]], dtype=jnp.int64)
        exact = check_galerkin_acquisition_support(
            _manifest(
                outgoing=outgoing,
                outgoing_shell_bounds=jnp.asarray([200.0], dtype=jnp.float64),
                shell_tolerance=jnp.asarray(200.0, dtype=jnp.float64),
            )
        )
        projected = check_galerkin_acquisition_support(
            _manifest(
                outgoing=outgoing,
                outgoing_dispositions=jnp.asarray(
                    [GalerkinDirectionDisposition.PROJECTED], dtype=jnp.int32
                ),
                outgoing_shell_bounds=jnp.asarray([200.0], dtype=jnp.float64),
                outgoing_projection_bounds=jnp.asarray(
                    [1.0e-10], dtype=jnp.float64
                ),
                shell_tolerance=jnp.asarray(200.0, dtype=jnp.float64),
            )
        )

        assert not exact.direction_evidence_valid
        assert _has_failure(
            exact.failure_mask,
            GalerkinAcquisitionSupportFailure.DIRECTION_EVIDENCE_INVALID,
        )
        assert projected.direction_evidence_valid
        assert projected.support_eligible

    def test_on_shell_diagnostic_is_recomputed_not_trusted(self) -> None:
        """Reject a zero bound when outward arithmetic is positive."""
        result = check_galerkin_acquisition_support(
            _manifest(
                incident_shell_bounds=jnp.asarray([0.0], dtype=jnp.float64)
            )
        )

        assert result.incident_shell_defect_upper_bounds[0] > 0.0
        assert not result.direction_evidence_valid

    def test_finite_geometry_maxima_cover_s1_16_and_s1_17(self) -> None:
        """Bind full and transverse outgoing and transfer maxima."""
        outgoing_wavevector = jnp.asarray([[6.0, 0.0, 8.0]], dtype=jnp.float64)
        projected_code = jnp.asarray(
            [GalerkinDirectionDisposition.PROJECTED], dtype=jnp.int32
        )
        result = check_galerkin_acquisition_support(
            _manifest(
                outgoing_wavevectors=outgoing_wavevector,
                outgoing_dispositions=projected_code,
                outgoing_projection_bounds=jnp.asarray(
                    [10.0], dtype=jnp.float64
                ),
            )
        )
        expected_transverse = 6.0 / (2.0 * np.pi)
        expected_full = np.sqrt(40.0) / (2.0 * np.pi)

        assert result.support_eligible
        for actual, expected in (
            (result.outgoing_transverse_offset_max, expected_transverse),
            (result.transfer_transverse_max, expected_transverse),
            (result.outgoing_full_offset_max, expected_full),
            (result.transfer_full_max, expected_full),
        ):
            assert float(actual) >= expected
            assert float(actual) == pytest.approx(expected, abs=1.0e-12)

    def test_transverse_maximum_is_perpendicular_to_carrier(self) -> None:
        """Use the carrier direction rather than the coordinate terminal."""
        carrier = jnp.asarray([6.0, 0.0, 8.0], dtype=jnp.float64)
        outgoing_wavevector = jnp.asarray(
            [[10.0, 0.0, 0.0]], dtype=jnp.float64
        )
        result = check_galerkin_acquisition_support(
            _manifest(
                carrier=carrier,
                wavenumber=jnp.asarray(10.0, dtype=jnp.float64),
                outgoing_wavevectors=outgoing_wavevector,
                outgoing_dispositions=jnp.asarray(
                    [GalerkinDirectionDisposition.PROJECTED], dtype=jnp.int32
                ),
                outgoing_projection_bounds=jnp.asarray(
                    [10.0], dtype=jnp.float64
                ),
            )
        )
        expected_transverse = 8.0 / (2.0 * np.pi)

        assert result.support_eligible
        assert float(result.outgoing_transverse_offset_max) >= (
            expected_transverse
        )
        assert float(result.outgoing_transverse_offset_max) == pytest.approx(
            expected_transverse, abs=1.0e-12
        )

    def test_checker_is_jit_compatible(self) -> None:
        """Compile every data-dependent finite-set and evidence predicate."""
        manifest = _manifest()
        eager = check_galerkin_acquisition_support(manifest)
        compiled = eqx.filter_jit(check_galerkin_acquisition_support)(manifest)

        assert compiled.status == eager.status
        assert compiled.failure_mask == eager.failure_mask
        assert compiled.direct_transfer_pair_count == (
            eager.direct_transfer_pair_count
        )
        assert jnp.array_equal(
            compiled.state_forward_mask, eager.state_forward_mask
        )
        assert compiled.transfer_full_max == eager.transfer_full_max

    def test_checker_interval_evidence_has_zero_jvp(self) -> None:
        """Stop acquisition-certificate tangents before outward arithmetic."""

        def projection_bound(carrier_z: jax.Array) -> jax.Array:
            carrier = jnp.stack(
                (
                    jnp.asarray(0.0, dtype=jnp.float64),
                    jnp.asarray(0.0, dtype=jnp.float64),
                    carrier_z,
                )
            )
            result = check_galerkin_acquisition_support(
                _manifest(carrier=carrier)
            )
            return result.incident_projection_error_upper_bounds[0]

        carrier_z = jnp.asarray(10.0, dtype=jnp.float64)
        eager = projection_bound(carrier_z)
        compiled = jax.jit(projection_bound)(carrier_z)
        _, tangent = jax.jvp(
            projection_bound,
            (carrier_z,),
            (jnp.asarray(1.0, dtype=jnp.float64),),
        )

        assert compiled == eager
        assert tangent == 0.0

    def test_unsupported_normal_arithmetic_rejects_acquisition(
        self,
        monkeypatch,
    ) -> None:
        """Reject acquisition evidence when a required probe fails."""
        interval_core = importlib.import_module("ptyrodactyl._interval")

        def unsupported_normal_arithmetic() -> jax.Array:
            return jnp.asarray(False)

        monkeypatch.setattr(
            interval_core,
            "_all_normal_arithmetic_supported",
            unsupported_normal_arithmetic,
        )
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="oriented normal intervals must be finite and ordered",
        ):
            result = check_galerkin_acquisition_support(_manifest())
            jax.block_until_ready(result)

    def test_factory_rejects_malformed_rank_and_zero_carrier(self) -> None:
        """Reject data that cannot denote one supported carrier frame."""
        with pytest.raises(ValueError, match="incident_indices"):
            _manifest(incident=jnp.asarray([0, 0, 0], dtype=jnp.int64))

        with pytest.raises(
            eqx.EquinoxRuntimeError, match="carrier must be nonzero"
        ):
            _manifest(
                carrier=jnp.zeros((3,), dtype=jnp.float64),
                wavenumber=jnp.asarray(1.0, dtype=jnp.float64),
            )

    def test_result_status_and_failure_mask_are_checker_derived(self) -> None:
        """Do not expose a public factory that accepts forged aggregates."""
        assert "create_galerkin_acquisition_support_result" not in (
            acquisition_types.__all__
        )
        assert not hasattr(
            acquisition_types,
            "create_galerkin_acquisition_support_result",
        )
