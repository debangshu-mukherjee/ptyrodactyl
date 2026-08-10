"""Tests for the private aggregate's scalar-physics implementation seams."""

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jax import lax
from numpy.testing import assert_array_equal

import ptyrodactyl._tools.physics as physics_leaf
from ptyrodactyl._tools import (
    coupled_interaction_value,
    helmholtz_coupling_value,
)


def _physics_inputs() -> Tuple[jax.Array, ...]:
    """Return canonical binary64 scalar inputs for a 300 kV electron."""
    return (
        jnp.asarray(300.0, dtype=jnp.float64),
        jnp.asarray(9.1093837015e-31, dtype=jnp.float64),
        jnp.asarray(1.602176634e-19, dtype=jnp.float64),
        jnp.asarray(2.99792458e8, dtype=jnp.float64),
        jnp.asarray(6.62607015e-34, dtype=jnp.float64),
    )


def test_physics_shared_seams_are_owned_by_physics_leaf() -> None:
    """Keep aggregate physics seams identical to their private leaf owner."""
    assert helmholtz_coupling_value is physics_leaf.helmholtz_coupling_value
    assert coupled_interaction_value is physics_leaf.coupled_interaction_value


def test_canonical_coupling_dtype_and_bits_agree_eager_and_compiled() -> None:
    """Freeze the rounded binary64 coupling across eager and JIT routes."""
    inputs = _physics_inputs()
    eager = helmholtz_coupling_value(*inputs)
    compiled = jax.jit(helmholtz_coupling_value)(*inputs)

    assert eager.shape == ()
    assert eager.dtype == jnp.float64
    assert compiled.dtype == jnp.float64
    assert int(np.asarray(eager).view(np.uint64).item()) == 0x3FDAA8EA661713D8
    assert_array_equal(compiled, eager)


def test_coupled_interaction_output_agrees_eager_and_compiled() -> None:
    """Return the canonical coupling and rounded complex interaction."""
    coefficients = jnp.asarray(
        (0.3 + 0.0j, 0.04 + 0.01j, -0.02 + 0.005j),
        dtype=jnp.complex128,
    )
    inputs = _physics_inputs()
    eager_coupling, eager_interaction = coupled_interaction_value(
        coefficients,
        *inputs,
    )
    compiled_coupling, compiled_interaction = jax.jit(
        coupled_interaction_value
    )(coefficients, *inputs)

    unrounded = eager_coupling * coefficients
    expected = lax.reduce_precision(
        jnp.real(unrounded),
        exponent_bits=11,
        mantissa_bits=50,
    ) + 1j * lax.reduce_precision(
        jnp.imag(unrounded),
        exponent_bits=11,
        mantissa_bits=50,
    )

    assert eager_coupling.shape == ()
    assert eager_coupling.dtype == jnp.float64
    assert eager_interaction.shape == coefficients.shape
    assert eager_interaction.dtype == jnp.complex128
    assert_array_equal(eager_interaction, expected)
    assert_array_equal(compiled_coupling, eager_coupling)
    assert_array_equal(compiled_interaction, eager_interaction)
