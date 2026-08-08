"""Tests for :mod:`ptyrodactyl.born.system`."""

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
    create_matched_galerkin_source,
    evaluate_physical_galerkin_adjoint_residual,
    evaluate_physical_galerkin_residual,
    lsqr_solve,
)
from ptyrodactyl.tools import helmholtz_coupling, relativistic_wavelength_ang
from ptyrodactyl.types import (
    GalerkinProductSupport,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    GalerkinTargetManifest,
    create_galerkin_product_support,
    create_galerkin_target_manifest,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _support() -> GalerkinProductSupport:
    """Create one three-mode support with independent coefficient bands."""
    state = jnp.array([[0, 0, -1], [0, 0, 0], [0, 0, 1]], dtype=jnp.int32)
    interaction = jnp.array(
        [[0, 0, -1], [0, 0, 0], [0, 0, 1]], dtype=jnp.int32
    )
    absorber = jnp.array(
        [
            [first, second, third]
            for first in range(-1, 2)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ]
        + [[0, 0, -2], [0, 0, 2]],
        dtype=jnp.int32,
    )
    work = jnp.array(
        [
            [first, second, third]
            for first in range(-1, 2)
            for second in range(-1, 2)
            for third in range(-3, 4)
        ],
        dtype=jnp.int32,
    )
    return create_galerkin_product_support(
        state,
        interaction,
        absorber,
        work,
        (3, 3, 7),
    )


def _manifest(*, interaction: bool = True) -> GalerkinTargetManifest:
    """Create one on-axis manifested target with an analytic absorber."""
    support = _support()
    voltage_kv = jnp.asarray(200.0, dtype=jnp.float64)
    k0 = 2.0 * jnp.pi / relativistic_wavelength_ang(voltage_kv)
    voltage_coefficients = (
        jnp.array(
            [0.02 - 0.01j, 0.10 + 0.0j, 0.02 + 0.01j],
            dtype=jnp.complex128,
        )
        if interaction
        else jnp.zeros((3,), dtype=jnp.complex128)
    )
    return create_galerkin_target_manifest(
        support=support,
        preterminal_indices=support.state_indices,
        voltage_coefficients=voltage_coefficients,
        box_lengths=jnp.array([5.0, 6.0, 7.0], dtype=jnp.float64),
        carrier=jnp.array([0.0, 0.0, k0], dtype=jnp.float64),
        accelerating_voltage_kv=voltage_kv,
        cap_scale=jnp.asarray(0.25, dtype=jnp.float64),
        target_name="three-mode-on-axis",
    )


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
    voltage_kv = jnp.asarray(200.0, dtype=jnp.float64)
    wavenumber = 2.0 * jnp.pi / relativistic_wavelength_ang(voltage_kv)
    manifest = create_galerkin_target_manifest(
        support=support,
        preterminal_indices=state_indices,
        voltage_coefficients=jnp.zeros(1, dtype=jnp.complex128),
        box_lengths=jnp.array([5.0, 6.0, 7.0], dtype=jnp.float64),
        carrier=jnp.array([0.0, 0.0, wavenumber], dtype=jnp.float64),
        accelerating_voltage_kv=voltage_kv,
        cap_scale=cap_scale,
        target_name="one-mode-on-axis",
    )
    return manifest


def _tilted_manifest() -> GalerkinTargetManifest:
    """Create one mixed-axis target with a tilted on-shell carrier."""
    state = jnp.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    absorber = jnp.array(
        [
            [first, second, third]
            for first in range(-1, 2)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ]
        + [[-2, 0, 0], [2, 0, 0]]
    )
    work = jnp.array(
        [
            [first, second, third]
            for first in range(-3, 4)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ]
    )
    support = create_galerkin_product_support(
        state_indices=state,
        interaction_indices=state,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(7, 3, 3),
    )
    voltage_kv = jnp.asarray(200.0, dtype=jnp.float64)
    k0 = 2.0 * jnp.pi / relativistic_wavelength_ang(voltage_kv)
    carrier_x = jnp.asarray(0.31, dtype=jnp.float64)
    carrier_y = jnp.asarray(-0.18, dtype=jnp.float64)
    carrier_z = jnp.sqrt(k0**2 - carrier_x**2 - carrier_y**2)
    return create_galerkin_target_manifest(
        support=support,
        preterminal_indices=support.state_indices,
        voltage_coefficients=jnp.array(
            [0.045 - 0.025j, 0.22 + 0.0j, 0.045 + 0.025j],
            dtype=jnp.complex128,
        ),
        box_lengths=jnp.array([5.0, 6.0, 7.0], dtype=jnp.float64),
        carrier=jnp.stack((carrier_x, carrier_y, carrier_z)),
        accelerating_voltage_kv=voltage_kv,
        cap_scale=jnp.asarray(0.23, dtype=jnp.float64),
        target_name="three-mode-tilted",
    )


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
    :see: :func:`ptyrodactyl.types.create_galerkin_target_manifest`
    :see: :func:`ptyrodactyl.born.apply_galerkin_target`
    :see: :func:`ptyrodactyl.born.apply_galerkin_target_adjoint`
    :see: :func:`ptyrodactyl.born.create_matched_galerkin_source`
    :see: :func:`ptyrodactyl.born.evaluate_physical_galerkin_adjoint_residual`
    :see: :func:`ptyrodactyl.born.evaluate_physical_galerkin_residual`
    """

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

    @pytest.mark.parametrize(
        ("box_lengths", "voltage_kv", "carrier", "message"),
        [
            (
                jnp.array([1.0, 1.0, 1.0e-300]),
                jnp.asarray(200.0),
                None,
                "free_diagonal",
            ),
            (
                jnp.ones(3),
                jnp.asarray(1.0e308),
                jnp.zeros(3),
                "voltage-derived (interaction coupling|wavenumber)",
            ),
        ],
    )
    def test_manifest_rejects_nonfinite_derived_physics_eager_and_compiled(
        self,
        box_lengths: jax.Array,
        voltage_kv: jax.Array,
        carrier: jax.Array | None,
        message: str,
    ) -> None:
        """Reject overflow in voltage-derived or shifted-free quantities."""
        support = _support()
        if carrier is None:
            k0 = 2.0 * jnp.pi / relativistic_wavelength_ang(voltage_kv)
            carrier = jnp.array([0.0, 0.0, k0])

        def build(box, voltage, wavevector):
            return create_galerkin_target_manifest(
                support=support,
                preterminal_indices=support.state_indices,
                voltage_coefficients=jnp.zeros(3, dtype=jnp.complex128),
                box_lengths=box,
                carrier=wavevector,
                accelerating_voltage_kv=voltage,
                cap_scale=0.25,
                target_name="overflowing-manifest",
            )

        with pytest.raises(_RUNTIME_ERRORS, match=message):
            jax.block_until_ready(build(box_lengths, voltage_kv, carrier))
        with pytest.raises(_RUNTIME_ERRORS, match=message):
            jax.block_until_ready(
                jax.jit(build)(box_lengths, voltage_kv, carrier)
            )

    def test_manifest_binds_bit_exact_voltage_coupling_eager_and_jit(
        self,
    ) -> None:
        """Bind phi, voltage, coupling, chi, and canonical rounding bytes."""
        eager = _manifest()

        def rebuild(phi: jax.Array) -> GalerkinTargetManifest:
            """Rebuild the same target with dynamic voltage coefficients."""
            manifest = create_galerkin_target_manifest(
                support=eager.support,
                preterminal_indices=eager.preterminal_indices,
                voltage_coefficients=phi,
                box_lengths=eager.box_lengths,
                carrier=eager.carrier,
                accelerating_voltage_kv=eager.accelerating_voltage_kv,
                cap_scale=eager.cap_scale,
                target_name=eager.target_name,
            )
            return manifest

        compiled = jax.jit(rebuild)(eager.voltage_coefficients)
        eager_builder = build_interaction_coefficients(
            eager.support,
            eager.voltage_coefficients,
            eager.accelerating_voltage_kv,
        )
        compiled_builder = jax.jit(build_interaction_coefficients)(
            eager.support,
            eager.voltage_coefficients,
            eager.accelerating_voltage_kv,
        )
        eager_coupling = helmholtz_coupling(eager.accelerating_voltage_kv)
        compiled_coupling = jax.jit(helmholtz_coupling)(
            eager.accelerating_voltage_kv
        )

        np.testing.assert_array_equal(
            eager.interaction_coefficients, compiled.interaction_coefficients
        )
        np.testing.assert_array_equal(
            eager.interaction_coefficients, eager_builder
        )
        np.testing.assert_array_equal(
            compiled.interaction_coefficients, compiled_builder
        )
        np.testing.assert_array_equal(
            eager.interaction_coupling, compiled.interaction_coupling
        )
        np.testing.assert_array_equal(
            eager.interaction_coupling, eager_coupling
        )
        np.testing.assert_array_equal(eager_coupling, compiled_coupling)
        assert "50-mantissa-bit" in eager.precision
        assert "50 mantissa bits" in eager.interaction_coefficient_provenance

    def test_manifest_cap_boundary_preserves_the_analytic_absorber(
        self,
    ) -> None:
        """Reject a low CAP and retain a normal action at the boundary."""
        base = _one_mode_manifest()
        boundary = 64.0 * jnp.finfo(jnp.float64).tiny
        below = jnp.nextafter(boundary, 0.0)

        def rebuild(cap_scale: jax.Array) -> GalerkinTargetManifest:
            """Rebuild the fixed target with one dynamic CAP scale."""
            manifest = create_galerkin_target_manifest(
                support=base.support,
                preterminal_indices=base.preterminal_indices,
                voltage_coefficients=jnp.zeros(1, dtype=jnp.complex128),
                box_lengths=base.box_lengths,
                carrier=base.carrier,
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
