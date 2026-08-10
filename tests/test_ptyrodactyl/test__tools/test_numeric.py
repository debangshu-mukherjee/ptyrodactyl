"""Tests for the private aggregate's numeric implementation seams."""

import jax
import jax.numpy as jnp
import numpy as np

import ptyrodactyl._tools.numeric as numeric_leaf
from ptyrodactyl._tools import (
    has_lost_nonzero_components,
    has_lost_subtraction,
    has_nonzero_components,
    has_subnormal_components,
)


def _boolean(value: jax.Array) -> bool:
    """Return one scalar predicate result as a host boolean."""
    return bool(np.asarray(value))


def test_numeric_shared_seams_are_owned_by_numeric_leaf() -> None:
    """Keep aggregate numeric seams identical to their private leaf owner."""
    assert has_subnormal_components is numeric_leaf.has_subnormal_components
    assert has_nonzero_components is numeric_leaf.has_nonzero_components
    assert has_lost_subtraction is numeric_leaf.has_lost_subtraction
    assert (
        has_lost_nonzero_components is numeric_leaf.has_lost_nonzero_components
    )


def test_normal_and_subnormal_predicates_agree_eager_and_compiled() -> None:
    """Classify exact component bits consistently through JIT."""
    zeros = jnp.asarray((0.0, -0.0), dtype=jnp.float64)
    normal = jnp.asarray((0.0, 1.0, -2.0), dtype=jnp.float64)
    subnormal = jnp.asarray(
        (
            float.fromhex("0x0.0000000000001p-1022"),
            -float.fromhex("0x0.0000000000001p-1022"),
        ),
        dtype=jnp.float64,
    )

    classify = jax.jit(
        lambda values: (
            has_subnormal_components(values),
            has_nonzero_components(values),
        )
    )

    assert not _boolean(has_nonzero_components(zeros))
    assert not _boolean(has_subnormal_components(normal))
    assert _boolean(has_nonzero_components(normal))
    assert _boolean(has_subnormal_components(subnormal))
    assert tuple(_boolean(value) for value in classify(normal)) == (
        False,
        True,
    )
    assert tuple(_boolean(value) for value in classify(subnormal)) == (
        True,
        True,
    )


def test_lost_subtraction_predicate_agrees_eager_and_compiled() -> None:
    """Detect an unequal stored pair whose supplied difference became zero."""
    left = jnp.asarray((1.0,), dtype=jnp.float64)
    right = jnp.asarray(
        (float.fromhex("0x1.0000000000001p+0"),),
        dtype=jnp.float64,
    )
    zero_difference = jnp.zeros_like(left)
    exact_difference = jnp.asarray(
        (-float.fromhex("0x1.0000000000000p-52"),),
        dtype=jnp.float64,
    )
    compiled = jax.jit(has_lost_subtraction)

    assert _boolean(has_lost_subtraction(left, right, zero_difference))
    assert _boolean(compiled(left, right, zero_difference))
    assert not _boolean(has_lost_subtraction(left, right, exact_difference))
    assert not _boolean(compiled(left, right, exact_difference))


def test_lost_nonzero_component_predicate_covers_zero_and_subnormal() -> None:
    """Detect normal source components mapped to zero or subnormal bits."""
    source = jnp.asarray((1.0 + 2.0j,), dtype=jnp.complex128)
    lost_imaginary = jnp.asarray((1.0 + 0.0j,), dtype=jnp.complex128)
    preserved = jnp.asarray((1.0 + 2.0j,), dtype=jnp.complex128)
    subnormal_real = jnp.asarray(
        (complex(float.fromhex("0x0.0000000000001p-1022"), 2.0),),
        dtype=jnp.complex128,
    )
    compiled = jax.jit(has_lost_nonzero_components)

    assert _boolean(has_lost_nonzero_components(source, lost_imaginary))
    assert _boolean(compiled(source, lost_imaginary))
    assert _boolean(has_lost_nonzero_components(source, subnormal_real))
    assert _boolean(compiled(source, subnormal_real))
    assert not _boolean(has_lost_nonzero_components(source, preserved))
    assert not _boolean(compiled(source, preserved))
