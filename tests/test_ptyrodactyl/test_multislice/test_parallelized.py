"""Tests for :mod:`ptyrodactyl.multislice.parallelized`."""
# ruff: noqa: E402, I001

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.multislice.parallelized import (
    cbed_amplitude_from_atoms,
    cbed_image_from_atoms,
    stem4d_sharded,
)
from ptyrodactyl.multislice.simulations import probe_modes_to_distribution
from ptyrodactyl.types import (
    create_atomic_slice_data,
    create_detector_config,
    create_microscope_config,
    create_probe_modes,
)


_GRID_SIZE = 4
_CALIB_ANG = jnp.asarray(0.5, dtype=jnp.float64)
_VOLTAGE_KV = jnp.asarray(80.0, dtype=jnp.float64)


def _distinct_modes():
    mode_zero = jnp.zeros(
        (_GRID_SIZE, _GRID_SIZE),
        dtype=jnp.complex128,
    ).at[0, 0].set(1.0 + 0.25j)
    mode_one = jnp.ones(
        (_GRID_SIZE, _GRID_SIZE),
        dtype=jnp.complex128,
    ) * (0.5 - 0.75j)
    return jnp.stack((mode_zero, mode_one), axis=-1)


def _atom_backed_inputs():
    sample = create_atomic_slice_data(
        atom_coords=jnp.asarray([[0.0, 0.0, 0.5]], dtype=jnp.float64),
        atom_types=jnp.asarray([0], dtype=jnp.int32),
        slice_z_bounds=jnp.asarray([[0.0, 1.0]], dtype=jnp.float64),
        atom_potentials=jnp.zeros(
            (1, _GRID_SIZE, _GRID_SIZE),
            dtype=jnp.float64,
        ),
    )
    microscope = create_microscope_config(
        voltage_kv=_VOLTAGE_KV,
        aperture_mrad=20.0,
        probe_shape=(_GRID_SIZE, _GRID_SIZE),
    )
    detector = create_detector_config(
        real_space_calib_ang=_CALIB_ANG,
        scan_positions_ang=jnp.asarray([[0.0, 0.0]], dtype=jnp.float64),
    )
    return sample, microscope, detector


def _cbed_from_atoms(beam, mode_distribution, sample):
    return cbed_image_from_atoms(
        beam=beam,
        mode_distribution=mode_distribution,
        atom_coords=sample.atom_coords,
        atom_types=sample.atom_types,
        slice_z_bounds=sample.slice_z_bounds,
        atom_potentials=sample.atom_potentials,
        voltage_kv=_VOLTAGE_KV,
        calib_ang=_CALIB_ANG,
        atom_mask=sample.atom_mask,
    )


def test_cbed_from_atoms_retains_complex_seam_and_single_mode_bit_identity():
    """Atom-backed CBED reduces only after retaining complex mode fields."""
    sample, _, _ = _atom_backed_inputs()
    modes = _distinct_modes()[..., :1]
    probe = create_probe_modes(
        modes,
        jnp.ones((1,), dtype=jnp.float64),
        _CALIB_ANG,
    )
    distribution = probe_modes_to_distribution(probe)

    amplitudes = cbed_amplitude_from_atoms(
        beam=modes,
        atom_coords=sample.atom_coords,
        atom_types=sample.atom_types,
        slice_z_bounds=sample.slice_z_bounds,
        atom_potentials=sample.atom_potentials,
        voltage_kv=_VOLTAGE_KV,
        calib_ang=_CALIB_ANG,
        atom_mask=sample.atom_mask,
    )
    image = _cbed_from_atoms(modes, distribution, sample)
    legacy_single_mode = jnp.abs(amplitudes[..., 0]) ** 2

    assert jnp.issubdtype(amplitudes.dtype, jnp.complexfloating)
    assert amplitudes.shape == (_GRID_SIZE, _GRID_SIZE, 1)
    assert np.array_equal(
        np.asarray(image),
        np.asarray(legacy_single_mode),
    )


def test_stem4d_sharded_honors_weights_and_is_jittable():
    """Distinct modes selected by [1, 0] and [0, 1] stay distinct."""
    sample, microscope, detector = _atom_backed_inputs()
    modes = _distinct_modes()
    first_probe = create_probe_modes(
        modes,
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        _CALIB_ANG,
    )
    second_probe = create_probe_modes(
        modes,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        _CALIB_ANG,
    )

    first = stem4d_sharded(first_probe, sample, microscope, detector)
    second = stem4d_sharded(second_probe, sample, microscope, detector)
    first_jitted = jax.jit(stem4d_sharded)(
        first_probe,
        sample,
        microscope,
        detector,
    )

    first_expected = _cbed_from_atoms(
        modes,
        probe_modes_to_distribution(first_probe),
        sample,
    )
    second_expected = _cbed_from_atoms(
        modes,
        probe_modes_to_distribution(second_probe),
        sample,
    )

    assert np.array_equal(
        np.asarray(first.data[0]),
        np.asarray(first_expected),
    )
    assert np.array_equal(
        np.asarray(second.data[0]),
        np.asarray(second_expected),
    )
    assert not np.array_equal(np.asarray(first.data), np.asarray(second.data))
    assert np.array_equal(
        np.asarray(first_jitted.data),
        np.asarray(first.data),
    )


def test_stem4d_sharded_has_finite_nonzero_mode_weight_gradient():
    """The explicit sharded mode distribution remains differentiable."""
    sample, microscope, detector = _atom_backed_inputs()
    modes = _distinct_modes()

    def loss(weights):
        probe = create_probe_modes(modes, weights, _CALIB_ANG)
        result = stem4d_sharded(probe, sample, microscope, detector)
        return jnp.sum(result.data)

    gradient = jax.grad(loss)(
        jnp.asarray([0.4, 0.6], dtype=jnp.float64),
    )

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)
