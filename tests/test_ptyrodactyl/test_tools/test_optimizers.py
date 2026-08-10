"""Regression tests for the historical complex optimizer oracle.

The tests pin the retained complex-gradient convention against an immutable
historical artifact. They do not promote that implementation as a new default.

:see: :class:`ptyrodactyl.tools.LRSchedulerState`
:see: :class:`ptyrodactyl.tools.Optimizer`
:see: :class:`ptyrodactyl.tools.OptimizerState`
:see: :func:`ptyrodactyl.tools.adagrad_update`
:see: :func:`ptyrodactyl.tools.adam_update`
:see: :func:`ptyrodactyl.tools.complex_adagrad`
:see: :func:`ptyrodactyl.tools.complex_rmsprop`
:see: :func:`ptyrodactyl.tools.create_cosine_scheduler`
:see: :func:`ptyrodactyl.tools.create_step_scheduler`
:see: :func:`ptyrodactyl.tools.create_warmup_cosine_scheduler`
:see: :func:`ptyrodactyl.tools.init_adagrad`
:see: :func:`ptyrodactyl.tools.init_adam`
:see: :func:`ptyrodactyl.tools.init_rmsprop`
:see: :func:`ptyrodactyl.tools.init_scheduler_state`
:see: :func:`ptyrodactyl.tools.rmsprop_update`
"""

from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict
from jaxtyping import Array, Shaped
from numpy.typing import NDArray

from ptyrodactyl.tools.optimizers import complex_adam, wirtinger_grad

_ORACLE_PATH = (
    Path(__file__).parents[2] / "test_data" / "plan01_wirtinger_oracle.npz"
)


def _load_oracle() -> Dict[str, Shaped[NDArray, "..."]]:
    """Load independent copies of every stored optimizer-oracle array."""
    with np.load(_ORACLE_PATH) as archive:
        oracle = {name: archive[name].copy() for name in archive.files}
    return oracle


def _assert_bit_equal(
    actual: Array,
    expected: Shaped[NDArray, "..."],
) -> None:
    """Assert both tree equality and exact stored-array equality."""
    chex.assert_trees_all_close(
        actual,
        jnp.asarray(expected),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_wirtinger_grad_matches_known_complex_quadratic() -> None:
    """Match the closed-form derivative of a real complex-domain objective.

    :see: :func:`ptyrodactyl.tools.wirtinger_grad`
    """
    oracle = _load_oracle()
    params = jnp.asarray(oracle["params0"])

    def real_quadratic(value: Array) -> Array:
        """Return the squared norm of the real component of ``value``."""
        objective = jnp.sum(jnp.real(value) ** 2)
        return objective

    actual = wirtinger_grad(real_quadratic)(params)
    expected = np.real(oracle["params0"]).astype(np.complex128)

    _assert_bit_equal(actual, expected)


def test_complex_adam_two_step_matches_historical_oracle() -> None:
    """Replay two captured default-parameter Adam steps exactly.

    :see: :func:`ptyrodactyl.tools.complex_adam`
    """
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
