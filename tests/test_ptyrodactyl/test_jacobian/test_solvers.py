"""Tests for :mod:`ptyrodactyl.jacobian.solvers`.

Extended Summary
----------------
Exercises closed-form solver updates, convergence-state freezing, and Krylov
tridiagonalisation against independently known small systems.

:see: :func:`ptyrodactyl.jacobian.conjugate_gradient`
:see: :func:`ptyrodactyl.jacobian.effective_nullspace_dimension`
:see: :func:`ptyrodactyl.jacobian.gauss_newton_solve`
:see: :func:`ptyrodactyl.jacobian.gauss_newton_step`
:see: :func:`ptyrodactyl.jacobian.lanczos_tridiagonal`
:see: :func:`ptyrodactyl.jacobian.levenberg_marquardt_solve`
:see: :func:`ptyrodactyl.jacobian.levenberg_marquardt_step`
:see: :func:`ptyrodactyl.jacobian.singular_spectrum`
"""

import jax
import jax.numpy as jnp
from beartype.typing import Tuple

from ptyrodactyl.jacobian import (
    conjugate_gradient,
    gauss_newton_solve,
    lanczos_tridiagonal,
    levenberg_marquardt_solve,
)
from ptyrodactyl.jacobian._treemath import _tree_dot
from ptyrodactyl.types import GNState, LMState


def test_tree_dot_uses_real_hermitian_inner_product() -> None:
    """Complex trees use the positive-definite real Hermitian product."""
    tree_a = {
        "complex": jnp.array([1.0 + 2.0j, -3.0 + 0.5j]),
        "real": jnp.array([2.0, -1.0]),
    }
    tree_b = {
        "complex": jnp.array([0.5 - 1.0j, 2.0 + 4.0j]),
        "real": jnp.array([-3.0, 5.0]),
    }

    expected = jnp.real(
        jnp.vdot(tree_a["complex"], tree_b["complex"])
        + jnp.vdot(tree_a["real"], tree_b["real"])
    )

    assert jnp.issubdtype(_tree_dot(tree_a, tree_b).dtype, jnp.floating)
    assert jnp.allclose(_tree_dot(tree_a, tree_b), expected)
    assert _tree_dot(tree_a, tree_a) > 0.0


def test_conjugate_gradient_tiny_quadratic_smoke() -> None:
    """Exercise CG on a one-step tiny diagonal system."""

    def linear_operator(vector):
        return 2.0 * vector

    rhs = jnp.array([2.0, 4.0])
    x0 = jnp.zeros_like(rhs)
    solution, iterations = conjugate_gradient(
        linear_operator,
        rhs,
        x0,
        max_iterations=1,
        tolerance=0.0,
    )

    assert iterations == 1
    assert jnp.all(jnp.isfinite(solution))
    assert jnp.allclose(solution, rhs / 2.0)


def test_conjugate_gradient_complex_hermitian_system() -> None:
    """CG solves a positive-definite system with complex-valued leaves."""

    def linear_operator(vector):
        return jnp.array([2.0, 5.0]) * vector

    rhs = jnp.array([2.0 + 4.0j, 10.0 - 5.0j])
    solution, iterations = conjugate_gradient(
        linear_operator,
        rhs,
        jnp.zeros_like(rhs),
        max_iterations=4,
        tolerance=1e-12,
    )

    assert iterations <= 4
    assert jnp.allclose(solution, rhs / jnp.array([2.0, 5.0]))


def test_gauss_newton_retains_newly_converged_update() -> None:
    """Retain the exact first GN update in eager and compiled execution.

    :see: :func:`ptyrodactyl.jacobian.gauss_newton_solve`
    """

    def forward_fn(params: jax.Array) -> jax.Array:
        prediction: jax.Array = 2.0 * params
        return prediction

    def solve(params: jax.Array) -> Tuple[jax.Array, GNState]:
        """Solve the fixed one-dimensional linear inverse problem."""
        solve_result: Tuple[jax.Array, GNState] = gauss_newton_solve(
            forward_fn,
            params,
            jnp.array([2.0]),
            max_iterations=4,
            tolerance=1e-10,
            cg_max_iterations=1,
            cg_tolerance=0.0,
        )
        return solve_result

    params_init = jnp.array([0.0])
    params_final, final_state = solve(params_init)
    compiled_params, compiled_state = jax.jit(solve)(params_init)

    assert jnp.allclose(params_final, jnp.array([1.0]), atol=1e-12)
    assert jnp.allclose(final_state.params, params_final)
    assert final_state.residual_norm < 1e-10
    assert final_state.iteration == 1
    assert jnp.allclose(compiled_params, params_final, atol=1e-12)
    assert jnp.allclose(compiled_state.params, final_state.params, atol=1e-12)
    assert jnp.allclose(
        compiled_state.residual_norm,
        final_state.residual_norm,
        atol=1e-12,
    )
    assert compiled_state.iteration == final_state.iteration


def test_levenberg_marquardt_retains_first_converged_update() -> None:
    """Retain the first accepted LM step in eager and compiled execution.

    :see: :func:`ptyrodactyl.jacobian.levenberg_marquardt_solve`
    """

    def forward_fn(params: jax.Array) -> jax.Array:
        prediction: jax.Array = 2.0 * params
        return prediction

    damping_init = 1e-3

    def solve(params: jax.Array) -> Tuple[jax.Array, LMState]:
        """Solve the fixed damped one-dimensional inverse problem."""
        solve_result: Tuple[jax.Array, LMState] = levenberg_marquardt_solve(
            forward_fn,
            params,
            jnp.array([2.0]),
            max_iterations=4,
            tolerance=1e-3,
            damping_init=damping_init,
            cg_max_iterations=1,
            cg_tolerance=0.0,
        )
        return solve_result

    params_init = jnp.array([0.0])
    params_final, final_state = solve(params_init)
    compiled_params, compiled_state = jax.jit(solve)(params_init)
    expected_params = jnp.array([4.0 / (4.0 + damping_init)])
    expected_residual = jnp.linalg.norm(2.0 * expected_params - 2.0)

    assert jnp.allclose(params_final, expected_params, atol=1e-12)
    assert jnp.allclose(final_state.params, expected_params, atol=1e-12)
    assert jnp.allclose(
        final_state.residual_norm,
        expected_residual,
        atol=1e-12,
    )
    assert final_state.residual_norm < 1e-3
    assert jnp.allclose(final_state.damping, damping_init / 3.0)
    assert final_state.iteration == 1
    assert jnp.allclose(compiled_params, params_final, atol=1e-12)
    assert jnp.allclose(compiled_state.params, final_state.params, atol=1e-12)
    assert jnp.allclose(
        compiled_state.residual_norm,
        final_state.residual_norm,
        atol=1e-12,
    )
    assert jnp.allclose(compiled_state.damping, final_state.damping)
    assert compiled_state.iteration == final_state.iteration


def test_already_converged_nonlinear_solver_states_remain_stable() -> None:
    """Freeze parameters, scalars, and iteration for converged GN/LM states.

    :see: :func:`ptyrodactyl.jacobian.gauss_newton_solve`
    :see: :func:`ptyrodactyl.jacobian.levenberg_marquardt_solve`
    """

    def forward_fn(params: jax.Array) -> jax.Array:
        prediction: jax.Array = 2.0 * params
        return prediction

    params_init = jnp.array([1.0])
    data = jnp.array([2.0])
    gn_params, gn_state = gauss_newton_solve(
        forward_fn,
        params_init,
        data,
        max_iterations=4,
        tolerance=1e-10,
        cg_max_iterations=1,
    )
    lm_params, lm_state = levenberg_marquardt_solve(
        forward_fn,
        params_init,
        data,
        max_iterations=4,
        tolerance=1e-10,
        damping_init=0.25,
        cg_max_iterations=1,
    )

    assert jnp.array_equal(gn_params, params_init)
    assert jnp.array_equal(gn_state.params, params_init)
    assert gn_state.residual_norm == 0.0
    assert gn_state.iteration == 0
    assert jnp.array_equal(lm_params, params_init)
    assert jnp.array_equal(lm_state.params, params_init)
    assert lm_state.residual_norm == 0.0
    assert lm_state.damping == 0.25
    assert lm_state.iteration == 0


def test_lanczos_matches_diagonal_operator_eager_and_jit() -> None:
    """Recover a full diagonal spectrum with finite eager and JIT outputs.

    :see: :func:`ptyrodactyl.jacobian.lanczos_tridiagonal`
    """
    diagonal = jnp.array([1.0, 2.0, 4.0])
    initial_vector = jnp.ones(3)

    def diagonal_operator(vector: jax.Array) -> jax.Array:
        operated: jax.Array = diagonal * vector
        return operated

    def run_lanczos(
        vector: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        tridiagonal: Tuple[jax.Array, jax.Array] = lanczos_tridiagonal(
            diagonal_operator,
            vector,
            num_iterations=3,
        )
        return tridiagonal

    alpha, beta = run_lanczos(initial_vector)
    alpha_jit, beta_jit = jax.jit(run_lanczos)(initial_vector)
    tridiagonal_matrix = (
        jnp.diag(alpha) + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
    )

    assert alpha.shape == (3,)
    assert beta.shape == (2,)
    assert jnp.all(jnp.isfinite(alpha))
    assert jnp.all(jnp.isfinite(beta))
    assert jnp.allclose(alpha_jit, alpha)
    assert jnp.allclose(beta_jit, beta)
    assert jnp.allclose(alpha[0], 7.0 / 3.0)
    assert jnp.allclose(beta[0], jnp.sqrt(14.0) / 3.0)
    assert jnp.allclose(
        jnp.linalg.eigvalsh(tridiagonal_matrix),
        diagonal,
        atol=1e-6,
    )
