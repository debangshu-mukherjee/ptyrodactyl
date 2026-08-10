"""Tests for :mod:`ptyrodactyl.born.system`."""

import inspect
from collections.abc import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptyrodactyl.born import (
    apply_galerkin_adjoint,
    apply_galerkin_operator,
    apply_galerkin_target,
    apply_galerkin_target_adjoint,
    build_cosine_shell_absorber_coefficients,
    build_interaction_coefficients,
    cgls_solve,
    create_host_checked_galerkin_target,
    create_matched_galerkin_source,
    evaluate_physical_galerkin_adjoint_residual,
    evaluate_physical_galerkin_residual,
    lsqr_solve,
)
from ptyrodactyl.born.acquisition import (
    check_galerkin_acquisition_support,
)
from ptyrodactyl.born.system import create_galerkin_target
from ptyrodactyl.tools import helmholtz_coupling, relativistic_wavelength_ang
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportStatus,
    GalerkinPotentialCertificateFailure,
    GalerkinPotentialErrorRoute,
    GalerkinProductSupport,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    GalerkinTargetManifest,
    create_galerkin_product_support,
    create_potential_3d,
)
from tests._galerkin_target_fixture import (
    TARGET_CAP_SCALE,
    TARGET_VOLTAGE_KV,
    checked_acquisition,
    periodic_target_potential,
    production_target,
    target_support,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _support() -> GalerkinProductSupport:
    """Create one three-mode support with independent coefficient bands."""
    support: GalerkinProductSupport = target_support()
    return support


def _manifest(*, interaction: bool = True) -> GalerkinTargetManifest:
    """Create one on-axis manifested target with an analytic absorber."""
    if interaction:
        manifest: GalerkinTargetManifest = production_target()
        return manifest
    potential = periodic_target_potential()
    vacuum = eqx.tree_at(
        lambda candidate: candidate.volume,
        potential,
        jnp.zeros_like(potential.volume),
    )
    eligibility = checked_acquisition(_support(), vacuum.box_size)
    manifest = create_galerkin_target(
        vacuum,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name="three-mode-on-axis-vacuum",
    )
    return manifest


def _one_mode_manifest(
    cap_scale: float | jax.Array = 0.25,
) -> GalerkinTargetManifest:
    """Create one analytic one-mode target for arithmetic-range tests."""
    profile_indices = jnp.array(
        [
            [first, second, third]
            for first in range(-1, 2)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ],
        dtype=jnp.int32,
    )
    state_indices = jnp.zeros((1, 3), dtype=jnp.int32)
    support = create_galerkin_product_support(
        state_indices=state_indices,
        interaction_indices=state_indices,
        absorber_indices=profile_indices,
        work_indices=profile_indices,
        work_shape=(3, 3, 3),
    )
    potential = create_potential_3d(
        jnp.zeros((3, 3, 3), dtype=jnp.float64),
        voxel_size=(1.0, 1.0, 1.0),
        box_size=(3.0, 3.0, 3.0),
        origin=(0.0, 0.0, 0.0),
        producer="one-mode-system-fixture-v1",
        provenance_hash="e" * 64,
        coefficient_normalization="VC-1 periodic trigonometric mean DFT",
        band_limit=0.2,
    )
    eligibility = checked_acquisition(support, potential.box_size)
    manifest = create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=cap_scale,
        target_name="one-mode-on-axis",
    )
    return manifest


def _tilted_manifest() -> GalerkinTargetManifest:
    """Create one mixed-axis target with a tilted on-shell carrier."""
    potential = periodic_target_potential()
    support = _support()
    eligibility = checked_acquisition(
        support,
        potential.box_size,
        carrier_direction=(1.0, -0.018, 0.031),
    )
    manifest: GalerkinTargetManifest = create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=0.23,
        target_name="three-mode-tilted",
    )
    return manifest


def _dense_target(manifest: GalerkinTargetManifest) -> np.ndarray:
    """Assemble an independent dense target from coefficient dictionaries."""
    state = np.asarray(manifest.support.state_indices)
    interaction_map = {
        tuple(index): value
        for index, value in zip(
            np.asarray(manifest.support.interaction_indices),
            np.asarray(manifest.interaction_coefficients),
            strict=True,
        )
    }
    absorber_map = {
        tuple(index): value
        for index, value in zip(
            np.asarray(manifest.support.absorber_indices),
            np.asarray(manifest.absorber_coefficients),
            strict=True,
        )
    }
    interaction = np.array(
        [
            [interaction_map.get(tuple(row - column), 0.0) for column in state]
            for row in state
        ],
        dtype=np.complex128,
    )
    absorber = np.array(
        [
            [absorber_map[tuple(row - column)] for column in state]
            for row in state
        ],
        dtype=np.complex128,
    )
    target = (
        np.diag(np.asarray(manifest.free_diagonal))
        - interaction
        - 1j * float(manifest.cap_scale) * absorber
    )
    return target


class TestScalarGalerkinSystem:
    """Verify manifested SC-1 target, source, and residual contracts.

    :see: :class:`ptyrodactyl.types.GalerkinPhysicalResidual`
    :see: :class:`ptyrodactyl.types.GalerkinSource`
    :see: :class:`ptyrodactyl.types.GalerkinSourceBranch`
    :see: :class:`ptyrodactyl.types.GalerkinTargetManifest`
    :see: :func:`ptyrodactyl.types.create_galerkin_physical_residual`
    :see: :func:`ptyrodactyl.types.create_galerkin_source`
    :see: :func:`ptyrodactyl.born.apply_galerkin_target`
    :see: :func:`ptyrodactyl.born.apply_galerkin_target_adjoint`
    :see: :func:`ptyrodactyl.born.create_galerkin_target`
    :see: :func:`ptyrodactyl.born.create_host_checked_galerkin_target`
    :see: :func:`ptyrodactyl.born.create_matched_galerkin_source`
    :see: :func:`ptyrodactyl.born.evaluate_physical_galerkin_adjoint_residual`
    :see: :func:`ptyrodactyl.born.evaluate_physical_galerkin_residual`
    """

    def test_production_target_signature_has_no_raw_coefficient_seam(
        self,
    ) -> None:
        """Freeze the sole Potential3D-to-checked-support builder signature.

        :see: :func:`ptyrodactyl.born.system.create_galerkin_target`
        """
        parameters = inspect.signature(create_galerkin_target).parameters

        assert tuple(parameters) == (
            "potential",
            "support_eligibility",
            "accelerating_voltage_kv",
            "cap_scale",
            "target_name",
        )
        for forbidden in (
            "support",
            "preterminal_indices",
            "voltage_coefficients",
            "carrier",
            "box_lengths",
            "wavenumber",
        ):
            assert forbidden not in parameters
        with pytest.raises(TypeError):
            create_galerkin_target(
                support=_support(),
                preterminal_indices=_support().state_indices,
                voltage_coefficients=jnp.zeros((3,), dtype=jnp.complex128),
                accelerating_voltage_kv=TARGET_VOLTAGE_KV,
                cap_scale=TARGET_CAP_SCALE,
                target_name="forbidden-raw-path",
            )

    def test_host_certificate_is_consumed_before_rm_s2_construction(
        self,
    ) -> None:
        """Put useful VC.17 evidence on an explicit production target path."""
        potential = periodic_target_potential()
        eligibility = checked_acquisition(_support(), potential.box_size)
        fallback = create_galerkin_target(
            potential,
            eligibility,
            accelerating_voltage_kv=TARGET_VOLTAGE_KV,
            cap_scale=TARGET_CAP_SCALE,
            target_name="triangle-target",
        )
        checked = create_host_checked_galerkin_target(
            potential,
            eligibility,
            accelerating_voltage_kv=TARGET_VOLTAGE_KV,
            cap_scale=TARGET_CAP_SCALE,
            target_name="host-checked-target",
            maximum_direct_terms=1_000,
        )
        failed = create_host_checked_galerkin_target(
            potential,
            eligibility,
            accelerating_voltage_kv=TARGET_VOLTAGE_KV,
            cap_scale=TARGET_CAP_SCALE,
            target_name="host-budget-failure-target",
            maximum_direct_terms=1,
        )
        jax.block_until_ready((fallback, checked, failed))

        certificate = checked.realization.coefficient_certificate
        failed_certificate = failed.realization.coefficient_certificate
        assert certificate is not None
        assert bool(certificate.finite_certificate)
        assert (
            checked.realization.error_route
            is GalerkinPotentialErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
        )
        np.testing.assert_array_equal(
            checked.voltage_coefficients,
            fallback.voltage_coefficients,
        )
        checked_bound = (
            checked.fixed_linear_error_ledger.fixed_linear_operator_error_bound
        )
        fallback_ledger = fallback.fixed_linear_error_ledger
        fallback_bound = fallback_ledger.fixed_linear_operator_error_bound
        assert checked_bound < fallback_bound
        assert failed_certificate is not None
        assert not bool(failed_certificate.finite_certificate)
        assert (
            failed_certificate.failure
            is GalerkinPotentialCertificateFailure.WORK_BUDGET_EXCEEDED
        )
        assert jnp.isinf(
            failed.fixed_linear_error_ledger.fixed_linear_operator_error_bound
        )

    def test_target_rechecks_ineligible_and_forged_support_results(
        self,
    ) -> None:
        """Reject honest ineligibility and forged aggregate eligibility."""
        potential = periodic_target_potential()
        valid = checked_acquisition(_support(), potential.box_size)
        invalid_manifest = eqx.tree_at(
            lambda manifest: manifest.preterminal_indices,
            valid.manifest,
            jnp.zeros((1, 3), dtype=jnp.int64),
        )
        ineligible = check_galerkin_acquisition_support(invalid_manifest)
        forged = eqx.tree_at(
            lambda result: (result.status, result.support_eligible),
            ineligible,
            (
                jnp.asarray(
                    GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE,
                    dtype=jnp.int32,
                ),
                jnp.asarray(True),
            ),
        )

        assert not ineligible.support_eligible
        for submitted in (ineligible, forged):
            with pytest.raises(_RUNTIME_ERRORS, match="independently"):
                target = create_galerkin_target(
                    potential,
                    submitted,
                    accelerating_voltage_kv=TARGET_VOLTAGE_KV,
                    cap_scale=TARGET_CAP_SCALE,
                    target_name="rejected-support",
                )
                jax.block_until_ready(target)

    def test_target_rejects_box_and_nominal_wavenumber_mismatches(
        self,
    ) -> None:
        """Require exact voxel/acquisition boxes and canonical voltage k0."""
        potential = periodic_target_potential()
        box_mismatch = checked_acquisition(
            _support(),
            (6.0, 3.0, 3.0),
        )
        wavenumber_mismatch = checked_acquisition(
            _support(),
            potential.box_size,
            voltage_kv=300.0,
        )

        with pytest.raises(_RUNTIME_ERRORS, match="box lengths"):
            target = create_galerkin_target(
                potential,
                box_mismatch,
                accelerating_voltage_kv=TARGET_VOLTAGE_KV,
                cap_scale=TARGET_CAP_SCALE,
                target_name="box-mismatch",
            )
            jax.block_until_ready(target)
        with pytest.raises(_RUNTIME_ERRORS, match="canonical voltage"):
            target = create_galerkin_target(
                potential,
                wavenumber_mismatch,
                accelerating_voltage_kv=TARGET_VOLTAGE_KV,
                cap_scale=TARGET_CAP_SCALE,
                target_name="wavenumber-mismatch",
            )
            jax.block_until_ready(target)

    def test_exact_and_projected_rows_bind_exact_target_geometry(self) -> None:
        """Keep zero rows symbolic and inflate only projected evidence."""
        exact = production_target()
        potential = periodic_target_potential()
        projected_support = checked_acquisition(
            _support(),
            potential.box_size,
            projected_offset=(0.0, 0.1, 0.0),
        )
        projected = create_galerkin_target(
            potential,
            projected_support,
            accelerating_voltage_kv=TARGET_VOLTAGE_KV,
            cap_scale=TARGET_CAP_SCALE,
            target_name="projected-direction-target",
        )
        jax.block_until_ready((exact, projected))

        np.testing.assert_array_equal(
            exact.exact_target_incident_shell_defect_bounds,
            jnp.zeros((1,), dtype=jnp.float64),
        )
        np.testing.assert_array_equal(
            exact.exact_target_incident_projection_error_bounds,
            jnp.zeros((1,), dtype=jnp.float64),
        )
        assert exact.incident_full_offset_max == 0.0
        assert (
            projected.exact_target_incident_shell_defect_bounds[0]
            >= projected_support.incident_shell_defect_upper_bounds[0]
        )
        assert (
            projected.exact_target_incident_projection_error_bounds[0]
            >= projected_support.incident_projection_error_upper_bounds[0]
        )
        assert (
            projected.incident_full_offset_max
            > projected_support.incident_full_offset_max
        )

    def test_target_source_and_residual_canonicalize_complex64_inputs(
        self,
    ) -> None:
        """Canonicalize the production physics seam to binary64 complex."""
        manifest = _manifest()
        field = jnp.asarray(
            [0.2 + 0.1j, -0.3 + 0.05j, 0.4 - 0.2j],
            dtype=jnp.complex64,
        )
        additional = jnp.asarray(
            [0.01 - 0.02j, 0.03 + 0.01j, -0.02 + 0.04j],
            dtype=jnp.complex64,
        )

        action = apply_galerkin_target(manifest, field)
        adjoint_action = apply_galerkin_target_adjoint(manifest, field)
        matched = create_matched_galerkin_source(
            manifest,
            field,
            additional,
        )
        residual = evaluate_physical_galerkin_residual(
            manifest,
            field,
            additional,
        )
        adjoint_residual = evaluate_physical_galerkin_adjoint_residual(
            manifest,
            field,
            additional,
        )

        assert action.dtype == jnp.complex128
        assert adjoint_action.dtype == jnp.complex128
        for value in (
            matched.incident_field,
            matched.incident_source,
            matched.additional_source,
            matched.total_source,
            matched.scattered_source,
            residual.residual,
            adjoint_residual.residual,
        ):
            assert value.dtype == jnp.complex128
        assert residual.residual_norm.dtype == jnp.float64
        assert adjoint_residual.residual_norm.dtype == jnp.float64

    def test_manifest_derives_voltage_consistent_shifted_diagonal(
        self,
    ) -> None:
        """Bind SC.2 voltage, the on-shell carrier, and SC.22 diagonal."""
        manifest = _manifest()
        expected_k0 = 2.0 * jnp.pi / relativistic_wavelength_ang(200.0)
        chex.assert_trees_all_close(manifest.wavenumber, expected_k0)
        assert manifest.contract_version == "SC-1"
        assert manifest.coefficient_normalization == "SC.13b"
        assert manifest.precision == (
            "float64/complex128; voltage-derived coupling and interaction "
            "use canonical 50-mantissa-bit rounding"
        )
        assert manifest.absorber_profile == "analytic_cosine_shell_v1"
        assert "exact SC.13b" in manifest.absorber_coefficient_provenance
        chex.assert_trees_all_equal(
            manifest.absorber_coefficients,
            build_cosine_shell_absorber_coefficients(manifest.support),
        )

    def test_target_and_adjoint_match_independent_dense_matrices(self) -> None:
        """Match H and H* against a direct nonnormal dense construction."""
        manifest = _manifest()
        field = jnp.array(
            [0.4 + 0.2j, -0.1 + 0.5j, 0.7 - 0.3j],
            dtype=jnp.complex128,
        )
        dense = _dense_target(manifest)
        forward = apply_galerkin_target(manifest, field)
        adjoint = apply_galerkin_target_adjoint(manifest, field)
        np.testing.assert_allclose(
            forward, dense @ np.asarray(field), atol=1e-11
        )
        np.testing.assert_allclose(
            adjoint, dense.conj().T @ np.asarray(field), atol=1e-11
        )
        compiled = jax.jit(apply_galerkin_target)(manifest, field)
        chex.assert_trees_all_close(compiled, forward, atol=1e-11)

    def test_tilted_target_and_adjoint_match_dense_and_dot_oracles(
        self,
    ) -> None:
        """Match tilted H/H* to dense actions and the complex dot identity."""
        manifest = _tilted_manifest()
        field = jnp.array(
            [0.4 + 0.2j, -0.1 + 0.5j, 0.7 - 0.3j],
            dtype=jnp.complex128,
        )
        probe = jnp.array(
            [-0.2 + 0.1j, 0.3 - 0.6j, 0.15 + 0.4j],
            dtype=jnp.complex128,
        )
        dense = _dense_target(manifest)
        forward = apply_galerkin_operator(manifest, field)
        adjoint = apply_galerkin_adjoint(manifest, probe)

        assert abs(float(manifest.carrier[0])) > 0.0
        assert abs(float(manifest.carrier[1])) > 0.0
        np.testing.assert_allclose(
            forward,
            dense @ np.asarray(field),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            adjoint,
            dense.conj().T @ np.asarray(probe),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            np.vdot(np.asarray(probe), np.asarray(forward)),
            np.vdot(np.asarray(adjoint), np.asarray(field)),
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_matched_source_reproduces_incident_field_in_vacuum(self) -> None:
        """Verify finite RM-S3 S_inc=H0 v and exact vacuum reproduction."""
        manifest = _manifest(interaction=False)
        incident = jnp.array(
            [0.2 - 0.1j, 1.0 + 0.0j, -0.3 + 0.4j],
            dtype=jnp.complex128,
        )
        source = create_matched_galerkin_source(manifest, incident)
        residual = evaluate_physical_galerkin_residual(
            manifest, incident, source.total_source
        )
        chex.assert_trees_all_close(
            residual.residual, jnp.zeros(3), atol=1e-12
        )
        np.testing.assert_allclose(residual.residual_norm, 0.0, atol=1e-12)
        chex.assert_trees_all_close(source.additional_source, jnp.zeros(3))

    def test_physical_residuals_use_direct_forward_and_adjoint_targets(
        self,
    ) -> None:
        """Recompute residuals independently of FFT and recurrence paths."""
        manifest = _manifest()
        field = jnp.array(
            [0.2 + 0.3j, -0.5 + 0.1j, 0.7 - 0.2j],
            dtype=jnp.complex128,
        )
        source = jnp.array(
            [0.8 - 0.1j, 0.2 + 0.4j, -0.3 + 0.5j],
            dtype=jnp.complex128,
        )
        dense = _dense_target(manifest)
        forward = evaluate_physical_galerkin_residual(manifest, field, source)
        adjoint = evaluate_physical_galerkin_adjoint_residual(
            manifest, field, source
        )
        expected_forward = np.asarray(source) - dense @ np.asarray(field)
        expected_adjoint = np.asarray(source) - dense.conj().T @ np.asarray(
            field
        )
        np.testing.assert_allclose(
            forward.residual, expected_forward, atol=1e-13
        )
        np.testing.assert_allclose(
            adjoint.residual, expected_adjoint, atol=1e-13
        )

    @pytest.mark.parametrize(
        ("solve", "method", "applications_per_iteration"),
        [
            (cgls_solve, GalerkinSolveMethod.CGLS, 3),
            (lsqr_solve, GalerkinSolveMethod.LSQR, 4),
        ],
        ids=["cgls", "lsqr"],
    )
    def test_manifested_solvers_compile_and_retain_direct_residuals(
        self,
        solve: Callable[..., GalerkinSolveResult],
        method: GalerkinSolveMethod,
        applications_per_iteration: int,
    ) -> None:
        """Match eager/JIT solves and account for every manifested action."""
        manifest = _manifest()
        source = jnp.array(
            [0.8 - 0.1j, 0.2 + 0.4j, -0.3 + 0.5j],
            dtype=jnp.complex128,
        )
        dense_solution = np.linalg.solve(
            _dense_target(manifest), np.asarray(source)
        )

        def solve_manifested(
            target: GalerkinTargetManifest,
            right_hand_side: jax.Array,
        ) -> GalerkinSolveResult:
            """Solve one manifested target with fixed numerical controls."""
            result = solve(
                target,
                right_hand_side,
                max_iterations=20,
                relative_tolerance=1.0e-12,
                absolute_tolerance=1.0e-14,
            )
            return result

        eager = solve_manifested(manifest, source)
        compiled = jax.jit(solve_manifested)(manifest, source)
        for result in (eager, compiled):
            physical = evaluate_physical_galerkin_residual(
                manifest, result.field, source
            )

            assert bool(result.converged)
            assert int(result.status) == int(GalerkinSolveStatus.CONVERGED)
            assert result.method is method
            assert int(result.iterations) > 0
            assert int(result.operator_applications) == (
                4 + applications_per_iteration * int(result.iterations)
            )
            np.testing.assert_allclose(
                result.field, dense_solution, rtol=1.0e-11, atol=1.0e-11
            )
            np.testing.assert_array_equal(result.residual, physical.residual)
            np.testing.assert_array_equal(
                result.residual_norm, physical.residual_norm
            )
            assert float(result.residual_norm) <= 1.0e-12
        chex.assert_trees_all_close(
            compiled.field,
            eager.field,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_direct_residual_mismatch_has_a_truthful_typed_status(
        self,
    ) -> None:
        """Do not mislabel a rounded-action/direct-residual disagreement."""
        manifest = _manifest()
        initial_field = jnp.array(
            [0.4 + 0.2j, -0.1 + 0.5j, 0.7 - 0.3j],
            dtype=jnp.complex128,
        )
        source = apply_galerkin_target(manifest, initial_field)
        physical = evaluate_physical_galerkin_residual(
            manifest, initial_field, source
        )
        assert float(physical.residual_norm) > 0.0

        for solve in (cgls_solve, lsqr_solve):
            result = solve(
                manifest,
                source,
                initial_field=initial_field,
                max_iterations=5,
                relative_tolerance=0.0,
                absolute_tolerance=0.0,
            )

            assert not bool(result.converged)
            assert int(result.status) == int(
                GalerkinSolveStatus.RESIDUAL_MISMATCH
            )
            assert int(result.iterations) == 0
            assert int(result.operator_applications) == 4
            assert float(result.residual_norm) > 0.0

    @pytest.mark.parametrize("magnitude", [1.0e-200, 1.0e200])
    def test_residual_norm_and_solver_failure_are_scale_safe(
        self,
        magnitude: float,
    ) -> None:
        """Avoid false zero or overflowing norms for finite residuals."""
        manifest = _manifest()
        source = jnp.array([magnitude + 0.0j, 0.0, 0.0], dtype=jnp.complex128)
        zeros = jnp.zeros(3, dtype=jnp.complex128)
        physical = evaluate_physical_galerkin_residual(manifest, zeros, source)

        np.testing.assert_allclose(
            physical.residual_norm, magnitude, rtol=1.0e-15
        )
        expected_statuses = (
            GalerkinSolveStatus.MAX_ITERATIONS,
            GalerkinSolveStatus.MAX_ITERATIONS,
        )
        for solve, expected_status in zip(
            (cgls_solve, lsqr_solve), expected_statuses, strict=True
        ):
            result = solve(
                manifest,
                source,
                max_iterations=5,
                relative_tolerance=0.0,
                absolute_tolerance=0.0,
            )
            assert not bool(result.converged)
            assert int(result.status) == int(expected_status)
            assert float(result.residual_norm) > 0.0

    def test_target_binds_same_exact_voltage_target_eager_and_jit(
        self,
    ) -> None:
        """Bind one exact target while enclosing either rounded realization."""
        eager = _manifest()

        def rebuild(volume: jax.Array) -> GalerkinTargetManifest:
            """Rebuild the same target with dynamic voxel values."""
            potential = eqx.tree_at(
                lambda candidate: candidate.volume,
                eager.potential,
                volume,
            )
            manifest = create_galerkin_target(
                potential,
                eager.support_eligibility,
                accelerating_voltage_kv=eager.accelerating_voltage_kv,
                cap_scale=eager.cap_scale,
                target_name=eager.target_name,
            )
            return manifest

        compiled = jax.jit(rebuild)(eager.potential.volume)
        _, volume_tangent = jax.jvp(
            lambda volume: rebuild(volume).interaction_coefficients,
            (eager.potential.volume,),
            (jnp.ones_like(eager.potential.volume),),
        )
        eager_builder = build_interaction_coefficients(
            eager.support,
            eager.voltage_coefficients,
            eager.accelerating_voltage_kv,
        )
        compiled_builder = jax.jit(build_interaction_coefficients)(
            eager.support,
            compiled.voltage_coefficients,
            eager.accelerating_voltage_kv,
        )
        eager_coupling = helmholtz_coupling(eager.accelerating_voltage_kv)
        compiled_coupling = jax.jit(helmholtz_coupling)(
            eager.accelerating_voltage_kv
        )

        np.testing.assert_allclose(
            eager.voltage_coefficients,
            compiled.voltage_coefficients,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        assert bool(
            jnp.all(
                jnp.abs(
                    eager.voltage_coefficients - compiled.voltage_coefficients
                )
                <= (
                    eager.realization.coefficient_error_bounds
                    + compiled.realization.coefficient_error_bounds
                )
            )
        )
        np.testing.assert_array_equal(
            eager.interaction_coefficients, eager_builder
        )
        np.testing.assert_array_equal(
            compiled.interaction_coefficients, compiled_builder
        )
        eager_ledger = eager.fixed_linear_error_ledger
        compiled_ledger = compiled.fixed_linear_error_ledger
        eager_interaction_errors = (
            eager_ledger.interaction_coefficient_error_bounds
        )
        compiled_interaction_errors = (
            compiled_ledger.interaction_coefficient_error_bounds
        )
        assert bool(
            jnp.all(
                jnp.abs(
                    eager.interaction_coefficients
                    - compiled.interaction_coefficients
                )
                <= (eager_interaction_errors + compiled_interaction_errors)
            )
        )
        np.testing.assert_array_equal(
            eager.interaction_coupling, compiled.interaction_coupling
        )
        np.testing.assert_array_equal(
            eager.interaction_coupling, eager_coupling
        )
        np.testing.assert_array_equal(eager_coupling, compiled_coupling)
        assert bool(jnp.all(jnp.isfinite(volume_tangent)))
        assert bool(jnp.any(jnp.abs(volume_tangent) > 0.0))
        assert "50-mantissa-bit" in eager.precision
        assert "50 mantissa bits" in eager.interaction_coefficient_provenance

    def test_target_cap_boundary_preserves_the_analytic_absorber(
        self,
    ) -> None:
        """Reject a low CAP and retain a normal action at the boundary."""
        base = _one_mode_manifest()
        boundary = 64.0 * jnp.finfo(jnp.float64).tiny
        below = jnp.nextafter(boundary, 0.0)

        def rebuild(cap_scale: jax.Array) -> GalerkinTargetManifest:
            """Rebuild the fixed target with one dynamic CAP scale."""
            manifest = create_galerkin_target(
                base.potential,
                base.support_eligibility,
                accelerating_voltage_kv=base.accelerating_voltage_kv,
                cap_scale=cap_scale,
                target_name="cap-boundary-target",
            )
            return manifest

        for build in (rebuild, jax.jit(rebuild)):
            with pytest.raises(_RUNTIME_ERRORS, match="cap_scale must"):
                jax.block_until_ready(build(below))
        manifest = rebuild(boundary)
        center = jnp.ones(1, dtype=jnp.complex128)
        action = apply_galerkin_target(manifest, center)

        assert np.any(np.asarray(action) != 0.0)
        assert np.all(np.isfinite(np.asarray(action)))

    def test_subnormal_rhs_fails_closed_in_residuals_and_solvers(self) -> None:
        """Reject stored nonzero RHS components that CPU arithmetic flushes."""
        manifest = _manifest()
        zeros = jnp.zeros(3, dtype=jnp.complex128)
        source = jnp.array([1.0e-308 + 0.0j, 0.0, 0.0])

        residual_calls = (
            lambda: evaluate_physical_galerkin_residual(
                manifest, zeros, source
            ),
            lambda: jax.jit(evaluate_physical_galerkin_residual)(
                manifest, zeros, source
            ),
        )
        for call in residual_calls:
            with pytest.raises(_RUNTIME_ERRORS, match="subnormal"):
                jax.block_until_ready(call())
        for solve in (cgls_solve, lsqr_solve):
            solve_once = lambda rhs, method=solve: method(  # noqa: E731
                manifest,
                rhs,
                max_iterations=5,
                relative_tolerance=0.0,
                absolute_tolerance=0.0,
            )
            for call in (solve_once, jax.jit(solve_once)):
                with pytest.raises(_RUNTIME_ERRORS, match="subnormal"):
                    jax.block_until_ready(call(source))

    def test_normal_input_whose_target_action_flushes_to_zero_is_rejected(
        self,
    ) -> None:
        """Reject loss of an injective target or matched-source action."""
        manifest = _manifest(interaction=False)
        incident = jnp.array(
            [0.0, jnp.finfo(jnp.float64).tiny, 0.0],
            dtype=jnp.complex128,
        )
        zeros = jnp.zeros_like(incident)
        calls = (
            lambda: apply_galerkin_target(manifest, incident),
            lambda: apply_galerkin_target_adjoint(manifest, incident),
            lambda: create_matched_galerkin_source(manifest, incident),
            lambda: evaluate_physical_galerkin_residual(
                manifest, incident, zeros
            ),
        )
        for call in calls:
            for checked_call in (call, jax.jit(call)):
                with pytest.raises(_RUNTIME_ERRORS, match="retain a nonzero"):
                    jax.block_until_ready(checked_call())

    def test_adjacent_normal_residual_cancellation_fails_closed(self) -> None:
        """Reject a subnormal residual that rounded subtraction loses."""
        manifest = _one_mode_manifest()
        absorber_diagonal = 7.0 / 8.0
        field_value = (
            8.0 * np.finfo(np.float64).tiny / (0.25 * absorber_diagonal)
        )
        field = jnp.asarray([field_value + 0.0j], dtype=jnp.complex128)
        action = complex(np.asarray(apply_galerkin_target(manifest, field))[0])
        source_value = complex(
            action.real,
            np.nextafter(action.imag, -np.inf),
        )
        source = jnp.asarray([source_value], dtype=jnp.complex128)

        assert source_value - action != 0.0j
        assert abs(source_value.imag) >= np.finfo(np.float64).tiny

        def residual_dynamic(
            candidate_field: jax.Array,
            candidate_source: jax.Array,
        ):
            """Evaluate the adjacent-normal residual with dynamic leaves."""
            residual = evaluate_physical_galerkin_residual(
                manifest,
                candidate_field,
                candidate_source,
            )
            return residual

        residual_calls = (
            lambda: residual_dynamic(field, source),
            lambda: jax.jit(residual_dynamic)(field, source),
            jax.jit(lambda: residual_dynamic(field, source)),
        )
        for call in residual_calls:
            with pytest.raises(_RUNTIME_ERRORS, match="subtraction lost"):
                jax.block_until_ready(call())

        for solve in (cgls_solve, lsqr_solve):

            def solve_dynamic(
                candidate_field: jax.Array,
                candidate_source: jax.Array,
                method: Callable[..., GalerkinSolveResult] = solve,
            ) -> GalerkinSolveResult:
                """Solve from one adjacent-normal residual pair."""
                result = method(
                    manifest,
                    candidate_source,
                    initial_field=candidate_field,
                    max_iterations=2,
                    relative_tolerance=0.0,
                    absolute_tolerance=0.0,
                )
                return result

            solve_calls = (
                lambda: solve_dynamic(field, source),
                lambda: jax.jit(solve_dynamic)(field, source),
                jax.jit(lambda: solve_dynamic(field, source)),
            )
            for call in solve_calls:
                with pytest.raises(_RUNTIME_ERRORS, match="subtraction lost"):
                    jax.block_until_ready(call())

    def test_adjacent_normal_source_addition_fails_closed(self) -> None:
        """Reject a matched-source sum whose exact subnormal value is lost."""
        manifest = _one_mode_manifest()
        absorber_diagonal = 7.0 / 8.0
        field_value = (
            8.0 * np.finfo(np.float64).tiny / (0.25 * absorber_diagonal)
        )
        incident = jnp.asarray([field_value + 0.0j], dtype=jnp.complex128)
        base_source = create_matched_galerkin_source(manifest, incident)
        incident_source = complex(np.asarray(base_source.incident_source)[0])
        additional_value = complex(
            -incident_source.real,
            np.nextafter(-incident_source.imag, np.inf),
        )
        additional = jnp.asarray([additional_value], dtype=jnp.complex128)

        assert incident_source + additional_value != 0.0j
        assert abs(additional_value.imag) >= np.finfo(np.float64).tiny

        def source_dynamic(
            candidate_incident: jax.Array,
            candidate_additional: jax.Array,
        ):
            """Construct the adjacent-normal matched-source sum."""
            source = create_matched_galerkin_source(
                manifest,
                candidate_incident,
                candidate_additional,
            )
            return source

        source_calls = (
            lambda: source_dynamic(incident, additional),
            lambda: jax.jit(source_dynamic)(incident, additional),
            jax.jit(lambda: source_dynamic(incident, additional)),
        )
        for call in source_calls:
            with pytest.raises(_RUNTIME_ERRORS, match="addition lost"):
                jax.block_until_ready(call())

    def test_manifest_actions_reject_nonfinite_derived_outputs(self) -> None:
        """Reject overflow from finite target fields in eager and JIT paths."""
        manifest = _manifest()
        field = jnp.full(3, 1.0e308 + 0.0j, dtype=jnp.complex128)
        for action in (apply_galerkin_target, apply_galerkin_target_adjoint):
            for call in (action, jax.jit(action)):
                with pytest.raises(_RUNTIME_ERRORS, match="must be finite"):
                    jax.block_until_ready(call(manifest, field))
