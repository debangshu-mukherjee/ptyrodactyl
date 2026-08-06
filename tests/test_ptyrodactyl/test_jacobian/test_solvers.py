"""Tests for :mod:`ptyrodactyl.jacobian.solvers`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/solvers.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import jax.numpy as jnp

from ptyrodactyl.jacobian._treemath import _tree_dot
from ptyrodactyl.jacobian.solvers import conjugate_gradient


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
