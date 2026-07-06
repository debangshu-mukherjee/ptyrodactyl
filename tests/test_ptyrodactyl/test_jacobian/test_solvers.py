"""Tests for :mod:`ptyrodactyl.jacobian.solvers`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/solvers.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import jax.numpy as jnp

from ptyrodactyl.jacobian.solvers import conjugate_gradient


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
