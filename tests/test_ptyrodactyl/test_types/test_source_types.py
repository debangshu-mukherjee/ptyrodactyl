"""Tests for :mod:`ptyrodactyl.types.source_types`.

Extended Summary
----------------
These tests freeze the RM-S3 source vocabulary and verify canonical action
storage, the complete six-stage representation ledger, finite direct-interval
source certificates, and typed infinity fallback.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Dict

from ptyrodactyl.types.source_types import (
    GalerkinRepresentedSourceKind,
    GalerkinSourceActions,
    GalerkinSourceAxis,
    GalerkinSourceErrorEnclosure,
    GalerkinSourceErrorRoute,
    GalerkinSourceModes,
    GalerkinSourcePhaseConvention,
    GalerkinSourceRepresentationLedger,
    GalerkinSourceRepresentationRoute,
    GalerkinStoredShellRoute,
    create_galerkin_source_actions,
    create_galerkin_source_error_enclosure,
    create_galerkin_source_ledger,
    create_galerkin_source_modes,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _error_kwargs(
    action_bound: float,
    exact_bound: float,
    *,
    environment_supported: bool = True,
) -> Dict[str, object]:
    """Build one complete source-error factory argument dictionary."""
    direct = jnp.zeros((2,), dtype=jnp.complex128)
    return {
        "free_action_error_upper_bound": action_bound,
        "cap_action_error_upper_bound": action_bound,
        "matched_source_error_upper_bound": action_bound,
        "interaction_action_error_upper_bound": action_bound,
        "total_source_error_upper_bound": action_bound,
        "scattered_source_error_upper_bound": action_bound,
        "independent_direct_cap_action": direct,
        "independent_direct_interaction_action": direct,
        "incident_field_norm_upper_bound": 1.0,
        "free_target_transfer_error_upper_bound": 0.0,
        "cap_target_transfer_error_upper_bound": 0.0,
        "interaction_target_transfer_error_upper_bound": 0.0,
        "exact_target_matched_source_error_upper_bound": exact_bound,
        "exact_target_total_source_error_upper_bound": exact_bound,
        "exact_target_scattered_source_error_upper_bound": exact_bound,
        "arithmetic_environment_supported": jnp.asarray(
            environment_supported, dtype=jnp.bool_
        ),
        "gradual_underflow_supported": jnp.asarray(True, dtype=jnp.bool_),
    }


def _mode_kwargs(flux_discrepancy: float) -> Dict[str, object]:
    """Build one minimal represented-mode factory argument dictionary."""
    coefficient = jnp.ones((1,), dtype=jnp.complex128)
    return {
        "aperture_weights": coefficient,
        "phased_coefficients": coefficient,
        "incident_field": coefficient,
        "physical_wavevectors": jnp.asarray(
            ((0.0, 0.0, 1.0),), dtype=jnp.float64
        ),
        "shell_defects": jnp.zeros((1,), dtype=jnp.float64),
        "exact_free_diagonal_lower_bounds": jnp.zeros((1,), dtype=jnp.float64),
        "exact_free_diagonal_upper_bounds": jnp.zeros((1,), dtype=jnp.float64),
        "exact_normal_wavevector_lower_bounds": jnp.ones(
            (1,), dtype=jnp.float64
        ),
        "exact_normal_wavevector_upper_bounds": jnp.ones(
            (1,), dtype=jnp.float64
        ),
        "active_mask": jnp.ones((1,), dtype=jnp.bool_),
        "forward_mask": jnp.ones((1,), dtype=jnp.bool_),
        "grazing_mask": jnp.zeros((1,), dtype=jnp.bool_),
        "backward_mask": jnp.zeros((1,), dtype=jnp.bool_),
        "scan_position": jnp.zeros((3,), dtype=jnp.float64),
        "aberration_phases": jnp.zeros((1,), dtype=jnp.float64),
        "source_plane_coordinate": 0.0,
        "shell_defect_tolerance": 0.0,
        "aperture_reduced_flux": 1.0,
        "input_reduced_flux": 1.0,
        "target_reduced_flux": 1.0,
        "output_reduced_flux": 1.0,
        "flux_normalization": 1.0,
        "exact_reduced_flux_lower_bound": 0.9,
        "exact_reduced_flux_upper_bound": 1.1,
        "target_reduced_flux_discrepancy_upper_bound": flux_discrepancy,
        "normal_axis": GalerkinSourceAxis.Z,
        "phase_convention": (
            GalerkinSourcePhaseConvention.PHYSICAL_WAVEVECTOR
        ),
        "stored_shell_route": GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL,
    }


class TestGalerkinSourceVocabulary:
    """Verify represented stored-shell RM-S3 carrier vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinRepresentedSource`
    :see: :class:`ptyrodactyl.types.GalerkinRepresentedSourceKind`
    :see: :class:`ptyrodactyl.types.GalerkinSourceActions`
    :see: :class:`ptyrodactyl.types.GalerkinSourceAxis`
    :see: :class:`ptyrodactyl.types.GalerkinSourceErrorEnclosure`
    :see: :class:`ptyrodactyl.types.GalerkinSourceErrorRoute`
    :see: :class:`ptyrodactyl.types.GalerkinSourceModes`
    :see: :class:`ptyrodactyl.types.GalerkinSourcePhaseConvention`
    :see: :class:`ptyrodactyl.types.GalerkinSourceRepresentationLedger`
    :see: :class:`ptyrodactyl.types.GalerkinSourceRepresentationRoute`
    :see: :class:`ptyrodactyl.types.GalerkinStoredShellRoute`
    :see: :func:`ptyrodactyl.types.create_galerkin_source_actions`
    :see: :func:`ptyrodactyl.types.create_galerkin_source_error_enclosure`
    :see: :func:`ptyrodactyl.types.create_galerkin_source_ledger`
    :see: :func:`ptyrodactyl.types.create_galerkin_source_modes`
    :see: :func:`ptyrodactyl.types.create_represented_galerkin_source`
    """

    def test_enums_freeze_geometry_branch_and_error_vocabulary(self) -> None:
        """Freeze every static source convention used by the narrow branch."""
        assert GalerkinSourceAxis.X.value == 0
        assert GalerkinSourceAxis.Y.value == 1
        assert GalerkinSourceAxis.Z.value == 2
        assert (
            GalerkinRepresentedSourceKind.PLANE_MODE.value
            == "represented_plane_mode"
        )
        assert (
            GalerkinRepresentedSourceKind.COHERENT_FOCUSED.value
            == "represented_coherent_focused"
        )
        assert (
            GalerkinSourcePhaseConvention.PHYSICAL_WAVEVECTOR.value
            == "physical_kappa_scan_source_plus_aberration"
        )
        assert (
            GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL.value
            == "exact_stored_free_diagonal"
        )
        assert (
            GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET.value
            == "exact_periodic_finite_target"
        )
        assert (
            GalerkinSourceErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE.value
            == "rm_s3_ftz_safe_direct_interval_bridge"
        )
        assert (
            GalerkinSourceErrorRoute.NONCERTIFIED_INFINITY.value
            == "typed_noncertificate_infinity"
        )

    def test_representation_ledger_keeps_all_six_terms_distinct(self) -> None:
        """Store exact represented stages separately from algebraic error."""
        ledger = create_galerkin_source_ledger(
            box_error_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            carrier_error_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            window_error_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            preband_error_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            band_error_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            algebraic_error_upper_bound=jnp.asarray(0.0),
            route=(
                GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET
            ),
        )
        jax.block_until_ready(ledger)

        assert ledger.box_error_upper_bound.dtype == jnp.float64
        assert ledger.carrier_error_upper_bound.dtype == jnp.float64
        assert ledger.window_error_upper_bound.dtype == jnp.float64
        assert ledger.preband_error_upper_bound.dtype == jnp.float64
        assert ledger.band_error_upper_bound.dtype == jnp.float64
        assert ledger.algebraic_error_upper_bound.dtype == jnp.float64
        assert ledger.algebraic_error_upper_bound == 0.0
        assert GalerkinSourceRepresentationLedger.__annotations__[
            "algebraic_error_upper_bound"
        ].dtypes == ("float64",)

    def test_exact_representation_route_rejects_silent_nonzero_stage(
        self,
    ) -> None:
        """Forbid nonzero errors under the exact finite-target route."""
        with pytest.raises(_RUNTIME_ERRORS, match="must be exactly zero"):
            ledger = create_galerkin_source_ledger(
                box_error_upper_bound=0.0,
                carrier_error_upper_bound=0.0,
                window_error_upper_bound=0.1,
                preband_error_upper_bound=0.0,
                band_error_upper_bound=0.0,
                algebraic_error_upper_bound=0.0,
                route=(
                    GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET
                ),
            )
            jax.block_until_ready(ledger)

    def test_error_enclosure_requires_typed_infinity_noncertificate(
        self,
    ) -> None:
        """Reject finite bounds under the noncertificate label."""
        enclosure = create_galerkin_source_error_enclosure(
            **_error_kwargs(jnp.inf, jnp.inf),
            route=GalerkinSourceErrorRoute.NONCERTIFIED_INFINITY,
        )
        jax.block_until_ready(enclosure)
        assert isinstance(enclosure, GalerkinSourceErrorEnclosure)
        assert jnp.isinf(enclosure.free_action_error_upper_bound)
        assert jnp.isinf(enclosure.cap_action_error_upper_bound)
        assert jnp.isinf(enclosure.matched_source_error_upper_bound)
        assert not enclosure.finite_certificate
        assert GalerkinSourceErrorEnclosure.__annotations__[
            "matched_source_error_upper_bound"
        ].dtypes == ("float64",)

        with pytest.raises(_RUNTIME_ERRORS, match="requires infinity"):
            invalid = create_galerkin_source_error_enclosure(
                **(
                    _error_kwargs(jnp.inf, jnp.inf)
                    | {"free_action_error_upper_bound": 0.0}
                ),
                route=GalerkinSourceErrorRoute.NONCERTIFIED_INFINITY,
            )
            jax.block_until_ready(invalid)

    def test_error_enclosure_admits_finite_direct_interval_certificate(
        self,
    ) -> None:
        """Store finite algebraic and exact-target source error layers."""
        enclosure = create_galerkin_source_error_enclosure(
            **_error_kwargs(2.0e-12, 3.0e-12),
            route=(GalerkinSourceErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE),
        )
        jax.block_until_ready(enclosure)

        assert enclosure.finite_certificate
        assert enclosure.arithmetic_environment_supported
        assert enclosure.matched_source_error_upper_bound == 2.0e-12
        assert (
            enclosure.exact_target_matched_source_error_upper_bound == 3.0e-12
        )
        assert "excludes full delta_H" in enclosure.error_scope

        unsupported = create_galerkin_source_error_enclosure(
            **_error_kwargs(
                2.0e-12,
                3.0e-12,
                environment_supported=False,
            ),
            route=(GalerkinSourceErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE),
        )
        jax.block_until_ready(unsupported)
        assert not unsupported.finite_certificate

    def test_source_modes_require_outward_requested_flux_discrepancy(
        self,
    ) -> None:
        """Reject a nominal flux interval paired with an understated error."""
        modes = create_galerkin_source_modes(**_mode_kwargs(0.11))
        jax.block_until_ready(modes)
        assert modes.target_reduced_flux_discrepancy_upper_bound == 0.11

        with pytest.raises(_RUNTIME_ERRORS, match="represented source modes"):
            invalid = create_galerkin_source_modes(**_mode_kwargs(0.01))
            jax.block_until_ready(invalid)

    def test_source_actions_canonicalize_and_validate_exact_dtypes(
        self,
    ) -> None:
        """Convert valid lower-width action vectors to complex128 storage."""
        zero = jnp.zeros((2,), dtype=jnp.complex64)
        one = jnp.ones((2,), dtype=jnp.complex64)
        actions = create_galerkin_source_actions(
            free_action=zero,
            cap_action=one,
            interaction_action=one,
            incident_source=-1j * one,
            additional_source=zero,
            total_source=-1j * one,
            scattered_source=one,
        )
        jax.block_until_ready(actions)
        assert isinstance(actions, GalerkinSourceActions)
        for field_name in GalerkinSourceActions.__annotations__:
            value = getattr(actions, field_name)
            assert value.dtype == jnp.complex128
            assert GalerkinSourceActions.__annotations__[
                field_name
            ].dtypes == ("complex128",)

    def test_source_actions_reject_shape_and_numeric_failures(self) -> None:
        """Reject mismatched, non-finite, and non-vector action data."""
        valid = jnp.zeros((2,), dtype=jnp.complex128)
        with pytest.raises(ValueError, match="length 2"):
            create_galerkin_source_actions(
                free_action=valid,
                cap_action=jnp.zeros((1,), dtype=jnp.complex128),
                interaction_action=valid,
                incident_source=valid,
                additional_source=valid,
                total_source=valid,
                scattered_source=valid,
            )
        with pytest.raises(_RUNTIME_ERRORS, match="must be finite"):
            actions = create_galerkin_source_actions(
                free_action=valid,
                cap_action=valid,
                interaction_action=valid,
                incident_source=valid.at[0].set(jnp.nan + 0.0j),
                additional_source=valid,
                total_source=valid,
                scattered_source=valid,
            )
            jax.block_until_ready(actions)

    def test_public_source_annotations_are_width_exact(self) -> None:
        """Keep every numerical source-carrier annotation width qualified."""
        for field_name in (
            "aperture_weights",
            "phased_coefficients",
            "incident_field",
        ):
            assert GalerkinSourceModes.__annotations__[field_name].dtypes == (
                "complex128",
            )
        for field_name in (
            "physical_wavevectors",
            "shell_defects",
            "exact_free_diagonal_lower_bounds",
            "exact_free_diagonal_upper_bounds",
            "exact_normal_wavevector_lower_bounds",
            "exact_normal_wavevector_upper_bounds",
            "source_plane_coordinate",
            "shell_defect_tolerance",
            "aperture_reduced_flux",
            "input_reduced_flux",
            "target_reduced_flux",
            "output_reduced_flux",
            "flux_normalization",
            "exact_reduced_flux_lower_bound",
            "exact_reduced_flux_upper_bound",
            "target_reduced_flux_discrepancy_upper_bound",
        ):
            assert GalerkinSourceModes.__annotations__[field_name].dtypes == (
                "float64",
            )
        for field_name in (
            "free_action_error_upper_bound",
            "cap_action_error_upper_bound",
            "matched_source_error_upper_bound",
            "interaction_action_error_upper_bound",
            "total_source_error_upper_bound",
            "scattered_source_error_upper_bound",
            "incident_field_norm_upper_bound",
            "free_target_transfer_error_upper_bound",
            "cap_target_transfer_error_upper_bound",
            "interaction_target_transfer_error_upper_bound",
            "exact_target_matched_source_error_upper_bound",
            "exact_target_total_source_error_upper_bound",
            "exact_target_scattered_source_error_upper_bound",
        ):
            assert GalerkinSourceErrorEnclosure.__annotations__[
                field_name
            ].dtypes == ("float64",)
