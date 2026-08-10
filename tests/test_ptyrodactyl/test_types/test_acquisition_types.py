"""Tests for :mod:`ptyrodactyl.types.acquisition_types`."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.types.acquisition_types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportFailure,
    GalerkinAcquisitionSupportResult,
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


def _minimal_manifest() -> GalerkinAcquisitionManifest:
    """Create one minimal structurally shaped support manifest."""
    zero_index = jnp.zeros((1, 3), dtype=jnp.int32)
    support = GalerkinProductSupport(
        state_indices=zero_index.astype(jnp.int64),
        interaction_indices=zero_index.astype(jnp.int64),
        absorber_indices=zero_index.astype(jnp.int64),
        work_indices=zero_index.astype(jnp.int64),
        work_shape=(1, 1, 1),
    )
    carrier = jnp.asarray([0.0, 0.0, 10.0], dtype=jnp.float32)
    manifest = create_galerkin_acquisition_manifest(
        support,
        zero_index,
        zero_index,
        zero_index,
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.zeros((0, 3), dtype=jnp.int32),
        incident_physical_wavevectors=carrier[None, :],
        outgoing_physical_wavevectors=carrier[None, :],
        incident_direction_dispositions=jnp.asarray(
            [GalerkinDirectionDisposition.EXACT_COEFFICIENT], dtype=jnp.int16
        ),
        outgoing_direction_dispositions=jnp.asarray(
            [GalerkinDirectionDisposition.EXACT_COEFFICIENT], dtype=jnp.int16
        ),
        incident_on_shell_defect_bounds=jnp.asarray([1.0e-10]),
        outgoing_on_shell_defect_bounds=jnp.asarray([1.0e-10]),
        incident_projection_error_bounds=jnp.asarray([0.0]),
        outgoing_projection_error_bounds=jnp.asarray([0.0]),
        carrier=carrier,
        box_lengths=jnp.ones((3,), dtype=jnp.float32),
        wavenumber=jnp.asarray(10.0, dtype=jnp.float32),
        carrier_on_shell_defect_bound=jnp.asarray(1.0e-10),
        on_shell_defect_tolerance=jnp.asarray(1.0e-8),
        terminal_axis=2,
        terminal_side=GalerkinTerminalSide.POSITIVE,
        carrier_id="minimal_forward",
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
        backward_disposition=GalerkinBackwardDisposition.EXCLUDED,
        backward_exclusion_basis="forward-sector projection",
        claims_backscatter=False,
    )
    return manifest


class TestGalerkinAcquisitionTypes:
    """Verify the typed RM-S1 support-core vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinAcquisitionManifest`
    :see: :class:`ptyrodactyl.types.GalerkinAcquisitionSupportFailure`
    :see: :class:`ptyrodactyl.types.GalerkinAcquisitionSupportResult`
    :see: :class:`ptyrodactyl.types.GalerkinAcquisitionSupportStatus`
    :see: :class:`ptyrodactyl.types.GalerkinBackwardDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinCarrierOverlapDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinCarrierOwnership`
    :see: :class:`ptyrodactyl.types.GalerkinCarrierTargetRoute`
    :see: :class:`ptyrodactyl.types.GalerkinDirectionDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinEndpointConvention`
    :see: :class:`ptyrodactyl.types.GalerkinTerminalSide`
    :see: :func:`ptyrodactyl.types.create_galerkin_acquisition_manifest`
    """

    def test_static_vocabulary_is_zero_legacy_and_support_scoped(self) -> None:
        """Freeze names that prevent inference of full detector eligibility."""
        assert GalerkinAcquisitionSupportStatus.STRUCTURALLY_INVALID == 0
        assert GalerkinAcquisitionSupportStatus.SUPPORT_INELIGIBLE == 1
        assert GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE == 2
        assert GalerkinAcquisitionSupportFailure.NONE == 0
        assert GalerkinDirectionDisposition.EXACT_COEFFICIENT == 0
        assert GalerkinDirectionDisposition.PROJECTED == 1
        assert GalerkinCarrierTargetRoute.NORMALIZE_FROM_ACCELERATING_VOLTAGE
        assert GalerkinBackwardDisposition.EXCLUDED.value == "excluded"
        assert GalerkinEndpointConvention.SIGNED_HALF_OPEN.value == (
            "signed_half_open"
        )

    def test_manifest_factory_canonicalizes_exact_storage_widths(self) -> None:
        """Canonicalize submitted integer, float, and disposition arrays."""
        manifest = _minimal_manifest()
        jax.block_until_ready(manifest)

        for field_name in (
            "incident_indices",
            "elastic_outgoing_indices",
            "preterminal_indices",
            "transverse_indices",
            "deliberately_omitted_indices",
        ):
            assert getattr(manifest, field_name).dtype == jnp.int64
        for field_name in (
            "incident_physical_wavevectors",
            "outgoing_physical_wavevectors",
            "carrier",
            "box_lengths",
            "wavenumber",
            "carrier_on_shell_defect_bound",
            "on_shell_defect_tolerance",
        ):
            assert getattr(manifest, field_name).dtype == jnp.float64
        assert manifest.incident_direction_dispositions.dtype == jnp.int32
        assert manifest.outgoing_direction_dispositions.dtype == jnp.int32

    def test_public_carrier_annotations_pin_widths_and_support_status(
        self,
    ) -> None:
        """Keep numerical result fields exact-width and support-scoped."""
        assert GalerkinAcquisitionManifest.__annotations__[
            "carrier"
        ].dtypes == ("float64",)
        assert GalerkinAcquisitionSupportResult.__annotations__[
            "status"
        ].dtypes == ("int32",)
        assert GalerkinAcquisitionSupportResult.__annotations__[
            "failure_mask"
        ].dtypes == ("int64",)
        assert GalerkinAcquisitionSupportResult.__annotations__[
            "support_eligible"
        ].dtypes == ("bool", "bool_")

    def test_manifest_factory_rejects_zero_algebraic_carrier(self) -> None:
        """Require a nonzero seed for exact target-side normalization."""
        manifest = _minimal_manifest()
        with pytest.raises(
            (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError, ValueError),
            match="carrier must be nonzero",
        ):
            invalid = create_galerkin_acquisition_manifest(
                manifest.support,
                manifest.incident_indices,
                manifest.elastic_outgoing_indices,
                manifest.preterminal_indices,
                manifest.transverse_indices,
                manifest.deliberately_omitted_indices,
                incident_physical_wavevectors=(
                    manifest.incident_physical_wavevectors
                ),
                outgoing_physical_wavevectors=(
                    manifest.outgoing_physical_wavevectors
                ),
                incident_direction_dispositions=(
                    manifest.incident_direction_dispositions
                ),
                outgoing_direction_dispositions=(
                    manifest.outgoing_direction_dispositions
                ),
                incident_on_shell_defect_bounds=(
                    manifest.incident_on_shell_defect_bounds
                ),
                outgoing_on_shell_defect_bounds=(
                    manifest.outgoing_on_shell_defect_bounds
                ),
                incident_projection_error_bounds=(
                    manifest.incident_projection_error_bounds
                ),
                outgoing_projection_error_bounds=(
                    manifest.outgoing_projection_error_bounds
                ),
                carrier=jnp.zeros((3,), dtype=jnp.float64),
                box_lengths=manifest.box_lengths,
                wavenumber=manifest.wavenumber,
                carrier_on_shell_defect_bound=(
                    manifest.carrier_on_shell_defect_bound
                ),
                on_shell_defect_tolerance=manifest.on_shell_defect_tolerance,
                terminal_axis=manifest.terminal_axis,
                terminal_side=manifest.terminal_side,
                carrier_id=manifest.carrier_id,
                carrier_ownership=manifest.carrier_ownership,
                carrier_overlap_disposition=(
                    manifest.carrier_overlap_disposition
                ),
                carrier_target_route=manifest.carrier_target_route,
                endpoint_convention=manifest.endpoint_convention,
                backward_disposition=manifest.backward_disposition,
                backward_exclusion_basis=manifest.backward_exclusion_basis,
                claims_backscatter=manifest.claims_backscatter,
            )
            jax.block_until_ready(invalid)
