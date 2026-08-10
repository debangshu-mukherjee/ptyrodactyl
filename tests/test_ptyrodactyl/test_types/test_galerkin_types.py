"""Tests for :mod:`ptyrodactyl.types.galerkin_types`."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.born import (
    create_galerkin_target,
    create_host_checked_galerkin_target,
)
from ptyrodactyl.types import (
    GalerkinPhysicalResidual,
    GalerkinSource,
    GalerkinSourceBranch,
    GalerkinStabilityDisposition,
    GalerkinStabilityFailure,
    GalerkinStabilityProof,
    GalerkinStabilityResult,
    GalerkinStabilityRoute,
    GalerkinTargetManifest,
    create_galerkin_physical_residual,
    create_galerkin_source,
    create_galerkin_stability_proof,
    create_galerkin_stability_result,
)
from tests._galerkin_target_fixture import production_target


def _canonical_target_from_lower_width_inputs() -> GalerkinTargetManifest:
    """Create one target whose valid public inputs use narrower dtypes."""
    base = production_target()
    return create_galerkin_target(
        base.potential,
        base.support_eligibility,
        accelerating_voltage_kv=jnp.asarray(200.0, dtype=jnp.float32),
        cap_scale=jnp.asarray(0.25, dtype=jnp.float32),
        target_name="canonical-dtype-target",
    )


class TestGalerkinProductionCarriers:
    """Verify the production Galerkin carrier vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinPhysicalResidual`
    :see: :class:`ptyrodactyl.types.GalerkinSource`
    :see: :class:`ptyrodactyl.types.GalerkinSourceBranch`
    :see: :class:`ptyrodactyl.types.GalerkinStabilityDisposition`
    :see: :class:`ptyrodactyl.types.GalerkinStabilityFailure`
    :see: :class:`ptyrodactyl.types.GalerkinStabilityProof`
    :see: :class:`ptyrodactyl.types.GalerkinStabilityResult`
    :see: :class:`ptyrodactyl.types.GalerkinStabilityRoute`
    :see: :class:`ptyrodactyl.types.GalerkinTargetManifest`
    :see: :func:`ptyrodactyl.types.create_galerkin_physical_residual`
    :see: :func:`ptyrodactyl.types.create_galerkin_source`
    :see: :func:`ptyrodactyl.types.create_galerkin_stability_proof`
    :see: :func:`ptyrodactyl.types.create_galerkin_stability_result`
    :see: :func:`ptyrodactyl.born.create_galerkin_target`
    :see: :func:`ptyrodactyl.born.create_host_checked_galerkin_target`
    """

    def test_enums_freeze_source_and_stability_vocabulary(self) -> None:
        """Freeze the public source, route, disposition, and failure values."""
        assert GalerkinSourceBranch.FINITE_MATCHED.value == "finite_matched"
        assert GalerkinStabilityRoute.ABSORBER_FLOOR.value == "absorber_floor"
        assert (
            GalerkinStabilityRoute.ABSORBER_FLOOR_GERSHGORIN.value
            == "absorber_floor_gershgorin"
        )
        assert (
            GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX.value
            == "absorber_floor_cosine_box"
        )
        assert (
            GalerkinStabilityDisposition.OPERATIONAL_PASS.value
            == "operational_pass"
        )
        assert (
            GalerkinStabilityDisposition.TYPED_FALLBACK.value
            == "typed_fallback"
        )
        assert GalerkinStabilityDisposition.REJECTED.value == "rejected"
        assert GalerkinStabilityFailure.NONE.value == "none"
        assert (
            GalerkinStabilityFailure.ARITHMETIC_RANGE_FAILURE.value
            == "arithmetic_range_failure"
        )
        assert (
            GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT.value
            == "invalid_submission_contract"
        )
        assert (
            GalerkinStabilityFailure.PROOF_RECORD_MISMATCH.value
            == "proof_record_mismatch"
        )

    def test_factories_canonicalize_storage_dtypes(self) -> None:
        """Store accepted lower-width inputs in the declared exact dtypes."""
        target = _canonical_target_from_lower_width_inputs()
        source = create_galerkin_source(
            incident_field=jnp.ones((1,), dtype=jnp.complex64),
            incident_source=jnp.ones((1,), dtype=jnp.complex64),
            additional_source=jnp.zeros((1,), dtype=jnp.complex64),
            total_source=jnp.ones((1,), dtype=jnp.complex64),
            scattered_source=jnp.zeros((1,), dtype=jnp.complex64),
            branch=GalerkinSourceBranch.FINITE_MATCHED,
        )
        residual = create_galerkin_physical_residual(
            residual=jnp.zeros((1,), dtype=jnp.complex64),
            residual_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )
        stability = create_galerkin_stability_result(
            lower_singular_bound=jnp.asarray(1.0, dtype=jnp.float32),
            residual_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            state_error_upper_bound=jnp.asarray(0.0, dtype=jnp.float32),
            state_budget=jnp.asarray(1.0, dtype=jnp.float32),
            route=GalerkinStabilityRoute.ABSORBER_FLOOR,
            disposition=GalerkinStabilityDisposition.OPERATIONAL_PASS,
            failure=GalerkinStabilityFailure.NONE,
            target_digest="target",
            result_digest="result",
            checker_id="checker",
        )

        assert target.preterminal_indices.dtype == jnp.int64
        assert target.voltage_coefficients.dtype == jnp.complex128
        assert target.interaction_coefficients.dtype == jnp.complex128
        assert target.interaction_coupling.dtype == jnp.float64
        assert target.absorber_coefficients.dtype == jnp.complex128
        assert target.free_diagonal.dtype == jnp.float64
        assert target.carrier.dtype == jnp.float64
        assert target.box_lengths.dtype == jnp.float64
        assert target.wavenumber.dtype == jnp.float64
        assert target.accelerating_voltage_kv.dtype == jnp.float64
        assert target.cap_scale.dtype == jnp.float64
        assert target.realization is not None
        assert target.fixed_linear_error_ledger is not None
        assert (
            target.support_eligibility
            is target.realization.support_eligibility
        )
        assert target.acquisition is target.support_eligibility.manifest
        for value in (
            source.incident_field,
            source.incident_source,
            source.additional_source,
            source.total_source,
            source.scattered_source,
            residual.residual,
        ):
            assert value.dtype == jnp.complex128
        assert residual.residual_norm.dtype == jnp.float64
        assert stability.lower_singular_bound.dtype == jnp.float64
        assert stability.residual_upper_bound.dtype == jnp.float64
        assert stability.state_error_upper_bound.dtype == jnp.float64
        assert stability.state_budget.dtype == jnp.float64
        annotation_contracts = {
            GalerkinTargetManifest: {
                "interaction_coefficients": "complex128",
                "interaction_coupling": "float64",
                "absorber_coefficients": "complex128",
                "exact_target_incident_full_offset_max": "float64",
                "exact_target_outgoing_full_offset_max": "float64",
                "exact_target_incident_shell_defect_bounds": "float64",
                "exact_target_outgoing_shell_defect_bounds": "float64",
                "exact_target_incident_projection_error_bounds": "float64",
                "exact_target_outgoing_projection_error_bounds": "float64",
                "accelerating_voltage_kv": "float64",
                "cap_scale": "float64",
            },
            GalerkinSource: {
                "incident_field": "complex128",
                "incident_source": "complex128",
                "additional_source": "complex128",
                "total_source": "complex128",
                "scattered_source": "complex128",
            },
            GalerkinPhysicalResidual: {
                "residual": "complex128",
                "residual_norm": "float64",
            },
            GalerkinStabilityResult: {
                "lower_singular_bound": "float64",
                "residual_upper_bound": "float64",
                "state_error_upper_bound": "float64",
                "state_budget": "float64",
            },
        }
        for carrier_type, field_contracts in annotation_contracts.items():
            for field_name, expected_dtype in field_contracts.items():
                annotation = carrier_type.__annotations__[field_name]
                assert annotation.dtypes == (expected_dtype,)

    def test_target_factory_retains_infinite_rm_s2_noncertificate(
        self,
    ) -> None:
        """Keep unrelated positive-infinite RM-S2 bounds as typed evidence."""
        base = production_target()
        target = create_host_checked_galerkin_target(
            base.potential,
            base.support_eligibility,
            accelerating_voltage_kv=base.accelerating_voltage_kv,
            cap_scale=base.cap_scale,
            target_name="explicit-rm-s2-noncertificate",
            maximum_direct_terms=1,
        )
        jax.block_until_ready(target)

        assert jnp.isinf(
            target.fixed_linear_error_ledger.interaction_operator_error_bound
        )
        assert not target.fixed_linear_error_ledger.finite_certificate

    def test_factories_normalize_and_validate_static_enum_labels(self) -> None:
        """Normalize valid labels and reject invalid static enum contracts."""
        zeros = jnp.zeros((1,), dtype=jnp.complex128)
        source: GalerkinSource = create_galerkin_source(
            incident_field=zeros,
            incident_source=zeros,
            additional_source=zeros,
            total_source=zeros,
            scattered_source=zeros,
            branch="finite_matched",
        )
        proof: GalerkinStabilityProof = create_galerkin_stability_proof(
            target_digest="target",
            result_digest="result",
            algebraic_floor_numerator=1,
            algebraic_floor_denominator=1,
            transferred_floor_numerator=0,
            transferred_floor_denominator=1,
            transferred_floor_finite=True,
            floor_numerator=1,
            floor_denominator=1,
            residual_squared_numerator=0,
            residual_squared_denominator=1,
            field_norm_squared_numerator=0,
            field_norm_squared_denominator=1,
            exact_target_residual_upper_numerator=0,
            exact_target_residual_upper_denominator=1,
            exact_target_residual_finite=True,
            source_error_upper_numerator=0,
            source_error_upper_denominator=1,
            source_error_finite=True,
            state_budget_numerator=1,
            state_budget_denominator=1,
            route="absorber_floor",
            failure="none",
            checker_id="checker",
            rhs_target="stored-source",
            residual_scope="independent-residual",
            source_error_route="stored-source-exact",
            source_error_scope="no source approximation",
        )
        result: GalerkinStabilityResult = create_galerkin_stability_result(
            lower_singular_bound=1.0,
            residual_upper_bound=0.0,
            state_error_upper_bound=0.0,
            state_budget=1.0,
            route="absorber_floor",
            disposition="operational_pass",
            failure="none",
            target_digest="target",
            result_digest="result",
            checker_id="checker",
        )

        assert source.branch is GalerkinSourceBranch.FINITE_MATCHED
        assert proof.route is GalerkinStabilityRoute.ABSORBER_FLOOR
        assert proof.failure is GalerkinStabilityFailure.NONE
        assert (
            result.disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS
        )
        with pytest.raises(ValueError):
            create_galerkin_source(
                incident_field=zeros,
                incident_source=zeros,
                additional_source=zeros,
                total_source=zeros,
                scattered_source=zeros,
                branch="unknown-source",
            )
        with pytest.raises(ValueError):
            create_galerkin_stability_result(
                lower_singular_bound=1.0,
                residual_upper_bound=0.0,
                state_error_upper_bound=0.0,
                state_budget=1.0,
                route="absorber_floor",
                disposition="operational_pass",
                failure="state_budget_missed",
                target_digest="target",
                result_digest="result",
                checker_id="checker",
            )
