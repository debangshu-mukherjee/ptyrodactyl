"""Tests for :mod:`ptyrodactyl.types.galerkin_types`."""

import jax.numpy as jnp
import pytest

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
    create_galerkin_target_manifest,
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
    :see: :func:`ptyrodactyl.types.create_galerkin_target_manifest`
    """

    def test_enums_freeze_source_and_stability_vocabulary(self) -> None:
        """Freeze the public source, route, disposition, and failure values."""
        assert GalerkinSourceBranch.FINITE_MATCHED.value == "finite_matched"
        assert GalerkinStabilityRoute.ABSORBER_FLOOR.value == "absorber_floor"
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
            floor_numerator=1,
            floor_denominator=1,
            residual_squared_numerator=0,
            residual_squared_denominator=1,
            state_budget_numerator=1,
            state_budget_denominator=1,
            route="absorber_floor",
            failure="none",
            checker_id="checker",
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
