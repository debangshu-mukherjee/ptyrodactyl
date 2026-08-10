"""Test validated form-factor parameter carriers.

:see: :func:`ptyrodactyl.types.create_kirkland_parameters`
:see: :func:`ptyrodactyl.types.create_lobato_parameters`
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from equinox import EquinoxRuntimeError

from ptyrodactyl.types import (
    KirklandParameters,
    LobatoParameters,
    create_kirkland_parameters,
    create_lobato_parameters,
)

_TRACED_ERROR_TYPES: Tuple[type[Exception], ...] = (
    EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _lobato_inputs() -> Tuple[jax.Array, jax.Array]:
    """Return nontrivial valid Lobato coefficients."""
    amplitudes = jnp.array(
        [1.0, -0.25, 0.125, -0.0625, 0.03125],
        dtype=jnp.float64,
    )
    scales = jnp.array([0.05, 0.2, 0.8, 3.2, 12.8], dtype=jnp.float64)
    return amplitudes, scales


def _kirkland_inputs() -> Tuple[jax.Array, ...]:
    """Return nontrivial valid Kirkland coefficients."""
    return (
        jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        jnp.array([0.4, 0.5, 0.6], dtype=jnp.float64),
        jnp.array([0.7, 0.8, 0.9], dtype=jnp.float64),
        jnp.array([1.0, 1.1, 1.2], dtype=jnp.float64),
    )


def test_factories_create_float64_equinox_pytrees_under_jit() -> None:
    """Valid factories preserve every coefficient as a dynamic JAX leaf.

    :see: :class:`ptyrodactyl.types.KirklandParameters`
    :see: :class:`ptyrodactyl.types.LobatoParameters`
    """
    lobato = jax.jit(create_lobato_parameters)(*_lobato_inputs())
    kirkland = jax.jit(create_kirkland_parameters)(*_kirkland_inputs())
    jax.block_until_ready((lobato.amplitudes, kirkland.gaussian_scales))

    assert isinstance(lobato, LobatoParameters)
    assert isinstance(lobato, eqx.Module)
    assert isinstance(kirkland, KirklandParameters)
    assert isinstance(kirkland, eqx.Module)
    assert len(jax.tree_util.tree_leaves(lobato)) == 2
    assert len(jax.tree_util.tree_leaves(kirkland)) == 4
    for leaf in jax.tree_util.tree_leaves((lobato, kirkland)):
        assert leaf.dtype == jnp.float64


@pytest.mark.parametrize("bad_shape", [(4,), (6,), (1, 5)])
def test_lobato_factory_rejects_wrong_static_shapes(
    bad_shape: Tuple[int, ...],
) -> None:
    """Both Lobato vectors must have exactly five entries."""
    amplitudes, scales = _lobato_inputs()
    with pytest.raises(ValueError, match=r"amplitudes.*shape \(5,\)"):
        create_lobato_parameters(jnp.ones(bad_shape), scales)
    with pytest.raises(ValueError, match=r"scales.*shape \(5,\)"):
        create_lobato_parameters(amplitudes, jnp.ones(bad_shape))


@pytest.mark.parametrize("field_index", range(4))
def test_kirkland_factory_rejects_wrong_static_shapes(
    field_index: int,
) -> None:
    """Every Kirkland vector must have exactly three entries."""
    inputs = list(_kirkland_inputs())
    inputs[field_index] = jnp.ones((4,), dtype=jnp.float64)

    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        create_kirkland_parameters(*inputs)


@pytest.mark.parametrize("invalid_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_factories_reject_nonfinite_coefficients(
    invalid_value: jax.Array,
) -> None:
    """Traced numerical validation rejects nonfinite amplitudes and scales."""
    lobato_amplitudes, lobato_scales = _lobato_inputs()
    bad_lobato = lobato_amplitudes.at[2].set(invalid_value)
    with pytest.raises(EquinoxRuntimeError, match="amplitudes"):
        create_lobato_parameters(bad_lobato, lobato_scales)

    kirkland_inputs = list(_kirkland_inputs())
    kirkland_inputs[3] = kirkland_inputs[3].at[1].set(invalid_value)
    with pytest.raises(EquinoxRuntimeError, match="gaussian_scales"):
        create_kirkland_parameters(*kirkland_inputs)


@pytest.mark.parametrize("invalid_scale", [0.0, -1.0])
def test_factories_reject_nonpositive_scales(invalid_scale: float) -> None:
    """Every physical width/scale coefficient must be strictly positive."""
    lobato_amplitudes, lobato_scales = _lobato_inputs()
    bad_lobato_scales = lobato_scales.at[0].set(invalid_scale)
    with pytest.raises(EquinoxRuntimeError, match="strictly positive"):
        create_lobato_parameters(lobato_amplitudes, bad_lobato_scales)

    kirkland_inputs = list(_kirkland_inputs())
    kirkland_inputs[1] = kirkland_inputs[1].at[0].set(invalid_scale)
    with pytest.raises(EquinoxRuntimeError, match="strictly positive"):
        create_kirkland_parameters(*kirkland_inputs)


def test_lobato_numerical_validation_executes_for_traced_inputs() -> None:
    """A JIT trace cannot bypass runtime coefficient validation."""
    amplitudes, scales = _lobato_inputs()
    bad_scales = scales.at[3].set(0.0)
    compiled_factory = jax.jit(create_lobato_parameters)

    with pytest.raises(_TRACED_ERROR_TYPES, match="strictly positive"):
        result = compiled_factory(amplitudes, bad_scales)
        jax.block_until_ready(result.scales)


def test_factory_carriers_have_finite_coefficient_gradients() -> None:
    """Carrier construction preserves gradients through all dynamic fields."""
    lobato_amplitudes, lobato_scales = _lobato_inputs()

    def lobato_loss(amplitudes: jax.Array, scales: jax.Array) -> jax.Array:
        params = create_lobato_parameters(amplitudes, scales)
        return jnp.sum(jnp.square(params.amplitudes) / params.scales)

    lobato_gradients = jax.grad(lobato_loss, argnums=(0, 1))(
        lobato_amplitudes,
        lobato_scales,
    )

    def kirkland_loss(*coefficients: jax.Array) -> jax.Array:
        params = create_kirkland_parameters(*coefficients)
        return jnp.sum(
            params.lorentzian_amplitudes / params.lorentzian_scales
        ) + jnp.sum(params.gaussian_amplitudes * params.gaussian_scales)

    kirkland_gradients = jax.grad(
        kirkland_loss,
        argnums=(0, 1, 2, 3),
    )(*_kirkland_inputs())

    for gradient in (*lobato_gradients, *kirkland_gradients):
        assert np.all(np.isfinite(np.asarray(gradient)))
