"""Regression tests for the historical complex optimizer oracle.

The hand-written optimizers are retained as a migration oracle until Plan 11
replaces them. These tests pin their existing Plan-02 behavior; they do not
promote that legacy implementation over the corrected Plan-11 formulation.
"""

from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ptyrodactyl.tools.optimizers import complex_adam, wirtinger_grad

_ORACLE_PATH = (
    Path(__file__).parents[2] / "test_data" / "plan01_wirtinger_oracle.npz"
)


def _load_oracle() -> dict[str, np.ndarray]:
    """Load independent copies of every stored optimizer-oracle array."""
    with np.load(_ORACLE_PATH) as archive:
        oracle = {name: archive[name].copy() for name in archive.files}
    return oracle


def _assert_bit_equal(actual: Array, expected: np.ndarray) -> None:
    """Assert both tree equality and exact stored-array equality."""
    chex.assert_trees_all_close(
        actual,
        jnp.asarray(expected),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_wirtinger_grad_matches_known_complex_quadratic() -> None:
    """Match the closed-form derivative of a real complex-domain objective."""
    oracle = _load_oracle()
    params = jnp.asarray(oracle["params0"])

    def real_quadratic(value: Array) -> Array:
        """Return the squared norm of the real component of ``value``."""
        objective = jnp.sum(jnp.real(value) ** 2)
        return objective

    actual = wirtinger_grad(real_quadratic)(params)
    expected = np.real(oracle["params0"]).astype(np.complex128)

    _assert_bit_equal(actual, expected)


def test_complex_adam_two_step_matches_plan01_oracle() -> None:
    """Replay the two default-parameter Adam steps captured before Plan 02."""
    oracle = _load_oracle()
    params0 = jnp.asarray(oracle["params0"])
    grads = jnp.asarray(oracle["grads"])
    zeros = jnp.zeros_like(params0)

    params1, (moment1, variance1, step1) = complex_adam(
        params0,
        grads,
        (zeros, zeros, 0),
    )
    params2, (moment2, variance2, step2) = complex_adam(
        params1,
        grads * 0.5,
        (moment1, variance1, step1),
    )

    assert step1 == 1
    assert step2 == 2
    _assert_bit_equal(params1, oracle["p1"])
    _assert_bit_equal(params2, oracle["p2"])
    _assert_bit_equal(moment2, oracle["m2"])
    _assert_bit_equal(variance2, oracle["v2"])
