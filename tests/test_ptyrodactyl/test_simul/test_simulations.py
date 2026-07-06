"""Tests for :mod:`ptyrodactyl.simul.simulations`."""
# ruff: noqa: E402, I001

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.simul.simulations import (
    annular_detector,
    decompose_beam_to_modes,
)
from ptyrodactyl.types import create_calibrated_array, create_stem4d


def test_annular_detector_static_scan_shape_and_jit() -> None:
    """Annular detector reshapes with a static raster and JIT-compiles."""
    positions = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    data = jnp.arange(6 * 4 * 4, dtype=jnp.float64).reshape(6, 4, 4)
    stem4d = create_stem4d(data, 0.5, 0.25, positions, 80.0)
    collection_angles = jnp.array([0.0, 1000.0], dtype=jnp.float64)
    expected = np.array(
        [[120.0, 376.0, 632.0], [888.0, 1144.0, 1400.0]],
        dtype=np.float64,
    )

    result = annular_detector(stem4d, collection_angles, (2, 3))
    jitted = jax.jit(annular_detector, static_argnames=("scan_shape",))(
        stem4d, collection_angles, (2, 3)
    )

    assert np.array_equal(np.asarray(result.data_array), expected)
    assert np.array_equal(np.asarray(jitted.data_array), expected)
    assert np.array_equal(
        np.asarray(result.data_array), np.asarray(jitted.data_array)
    )


def test_decompose_beam_to_modes_requires_key() -> None:
    """Beam decomposition no longer has an implicit PRNG key."""
    beam = create_calibrated_array(
        (jnp.arange(16, dtype=jnp.float64).reshape(4, 4) + 1j).astype(
            jnp.complex128
        ),
        0.5,
        0.5,
        True,
    )

    with pytest.raises(TypeError):
        decompose_beam_to_modes(beam, 3)


def test_decompose_beam_to_modes_fixed_key_reproducible() -> None:
    """A fixed key gives reproducible modes and old key-zero values."""
    beam = create_calibrated_array(
        (jnp.arange(16, dtype=jnp.float64).reshape(4, 4) + 1j).astype(
            jnp.complex128
        ),
        0.5,
        0.5,
        True,
    )

    first = decompose_beam_to_modes(
        beam, 3, jax.random.PRNGKey(0), first_mode_weight=0.6
    )
    second = decompose_beam_to_modes(
        beam, 3, jax.random.PRNGKey(0), first_mode_weight=0.6
    )

    assert np.array_equal(np.asarray(first.modes), np.asarray(second.modes))
    assert np.array_equal(
        np.asarray(first.weights), np.asarray(second.weights)
    )
    assert np.array_equal(np.asarray(first.calib), np.asarray(second.calib))
    assert np.array_equal(
        np.asarray(first.modes[0, 0, 0]),
        np.asarray(-0.26771966341457853 + 0.199488434683742j),
    )
    assert np.array_equal(
        np.asarray(first.weights),
        np.asarray([0.6, 0.2, 0.2], dtype=np.float64),
    )
