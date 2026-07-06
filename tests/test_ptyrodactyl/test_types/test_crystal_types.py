"""Tests for :mod:`ptyrodactyl.types.crystal_types`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/types/crystal_types.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import jax
import jax.numpy as jnp

from ptyrodactyl.types import CrystalData, create_crystal_data


def test_create_crystal_data_jit_compiles_and_runs() -> None:
    """JIT-compile and execute the valid CrystalData factory path."""
    jitted_create = jax.jit(create_crystal_data)
    crystal_data: CrystalData = jitted_create(
        positions=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float64),
        atomic_numbers=jnp.array([14], dtype=jnp.int32),
    )
    jax.block_until_ready(crystal_data.positions)

    assert crystal_data.positions.shape == (1, 3)
    assert crystal_data.atomic_numbers.shape == (1,)
