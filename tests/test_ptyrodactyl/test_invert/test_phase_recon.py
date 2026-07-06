"""Tests for :mod:`ptyrodactyl.invert.phase_recon`."""
# ruff: noqa: E402, I001

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.invert.phase_recon import (
    multi_slice_multi_modal,
    single_slice_multi_modal,
    single_slice_poscorrected,
    single_slice_ptychography,
)
from ptyrodactyl.types import (
    ProbeModes,
    create_calibrated_array,
    create_stem4d,
)


def _tiny_recon_inputs():
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 5)
    shape = (4, 4)
    positions = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    experimental = create_stem4d(
        jnp.ones((2, *shape), dtype=jnp.float64),
        1.0,
        0.25,
        positions,
        80.0,
    )
    potential_array = (
        jax.random.normal(keys[0], shape, dtype=jnp.float64) * 0.01
    ).astype(jnp.complex128)
    potential = create_calibrated_array(potential_array, 1.0, 1.0, True)
    beam_array = (
        jax.random.normal(keys[1], shape, dtype=jnp.float64)
        + 1j * jax.random.normal(keys[2], shape, dtype=jnp.float64)
    ).astype(jnp.complex128)
    beam = create_calibrated_array(beam_array, 1.0, 1.0, True)
    mode_array = jnp.stack([beam_array, beam_array * (0.5 + 0.1j)], axis=-1)
    modes = ProbeModes(
        modes=mode_array,
        weights=jnp.array([0.7, 0.3], dtype=jnp.float64),
        calib=jnp.array(1.0, dtype=jnp.float64),
    )
    return experimental, potential, beam, potential_array, beam_array, modes


def _assert_scalar_equal(actual, expected) -> None:
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_single_slice_ptychography_runs_and_regresses() -> None:
    """The base reconstructor runs and returns fixed-seed values."""
    experimental, potential, beam, *_ = _tiny_recon_inputs()

    output = single_slice_ptychography(
        experimental,
        potential,
        beam,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[2].shape == (4, 4, 2)
    assert output[3].shape == (4, 4, 2)
    _assert_scalar_equal(output[0].data_array[0, 0], 0.028663108252426836 + 0j)
    _assert_scalar_equal(
        output[1].data_array[0, 0],
        -1.4996974644913912 - 1.4834405203573693j,
    )
    _assert_scalar_equal(output[2][0, 0, 0], 0.027663112105862656 + 0j)


def test_single_slice_poscorrected_regresses() -> None:
    """Position-corrected single-slice reconstruction is fixed-seed stable."""
    experimental, potential, beam, *_ = _tiny_recon_inputs()

    output = single_slice_poscorrected(
        experimental,
        potential,
        beam,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[2].shape == (2, 2)
    assert output[5].shape == (2, 2, 2)
    _assert_scalar_equal(output[0].data_array[0, 0], 0.028663116254880484 + 0j)
    _assert_scalar_equal(
        output[1].data_array[0, 0],
        -1.4996974644877181 - 1.4834405203501158j,
    )
    _assert_scalar_equal(output[2][0, 0], 0.0020012170316727577)


def test_single_slice_multi_modal_regresses() -> None:
    """Multi-modal single-slice reconstruction is fixed-seed stable."""
    experimental, _, _, potential_array, _, modes = _tiny_recon_inputs()

    output = single_slice_multi_modal(
        experimental,
        potential_array,
        modes,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[3].shape == (4, 4, 2)
    assert output[4].shape == (4, 4, 2, 2)
    _assert_scalar_equal(output[0][0, 0], 0.028663118115104917 + 0j)
    _assert_scalar_equal(
        output[1].modes[0, 0, 0],
        -1.4996978807671886 - 1.4834412513458781j,
    )
    _assert_scalar_equal(output[2][0, 0], 0.002001215397622949)


def test_multi_slice_multi_modal_regresses() -> None:
    """Multi-slice multi-modal reconstruction is fixed-seed stable."""
    experimental, _, _, potential_array, beam_array, _ = _tiny_recon_inputs()

    output = multi_slice_multi_modal(
        experimental,
        potential_array,
        beam_array,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[3].shape == (4, 4, 2)
    assert output[4].shape == (4, 4, 2)
    _assert_scalar_equal(output[0][0, 0], 0.028663176363479325 + 0j)
    _assert_scalar_equal(
        output[1][0, 0],
        -1.4996974644560432 - 1.4834405202874856j,
    )
    _assert_scalar_equal(output[2][0, 0], 0.020009807739047515)
