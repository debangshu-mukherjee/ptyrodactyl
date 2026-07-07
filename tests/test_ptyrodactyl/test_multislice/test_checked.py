"""Tests for :mod:`ptyrodactyl.multislice.checked`."""
# ruff: noqa: E402, I001

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import EquinoxRuntimeError

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.multislice import (
    checked_cbed_image,
    checked_make_probe,
    checked_stem4d_sharded,
    checked_stem_4d,
)
from ptyrodactyl.multislice.parallelized import stem4d_sharded
from ptyrodactyl.multislice.simulations import cbed_image, make_probe, stem_4d
from ptyrodactyl.types import (
    create_atomic_slice_data,
    create_detector_config,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
)


_GRID_SIZE = 16
_IMAGE_SIZE = jnp.array([_GRID_SIZE, _GRID_SIZE], dtype=jnp.int32)
_APERTURE = jnp.asarray(12.0, dtype=jnp.float64)
_VOLTAGE = jnp.asarray(80.0, dtype=jnp.float64)
_CALIB_PM = jnp.asarray(50.0, dtype=jnp.float64)
_CALIB_ANG = jnp.asarray(0.5, dtype=jnp.float64)
_SLICE_THICKNESS = jnp.asarray(1.0, dtype=jnp.float64)
_POSITIONS = jnp.array(
    [[0.0, 0.0], [3.0, 4.0]],
    dtype=jnp.float64,
)


def _assert_tree_array_equal(actual, expected):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)

    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        assert np.array_equal(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
        )


def _assert_finite(value):
    assert np.all(np.isfinite(np.asarray(value)))


def _assert_tree_allclose(actual, expected):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)

    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=0.0,
            atol=1.0e-15,
        )


def _potential_data(scale=1.0):
    yy = jnp.arange(_GRID_SIZE, dtype=jnp.float64).reshape(_GRID_SIZE, 1, 1)
    xx = jnp.arange(_GRID_SIZE, dtype=jnp.float64).reshape(1, _GRID_SIZE, 1)
    zz = jnp.arange(2, dtype=jnp.float64).reshape(1, 1, 2)
    return scale * 1.0e-4 * (yy + (2.0 * xx) + (3.0 * zz))


def _valid_potential_slices(scale=1.0):
    return create_potential_slices(
        _potential_data(scale),
        _SLICE_THICKNESS,
        _CALIB_ANG,
    )


def _microscope(defocus=0.0):
    return create_microscope_config(
        voltage_kv=_VOLTAGE,
        aperture_mrad=_APERTURE,
        defocus_ang=defocus,
        probe_shape=(_GRID_SIZE, _GRID_SIZE),
    )


def _detector(scan_positions_px=None, scan_positions_ang=None):
    return create_detector_config(
        real_space_calib_ang=_CALIB_ANG,
        probe_calibration_pm=_CALIB_PM,
        scan_positions_px=scan_positions_px,
        scan_positions_ang=scan_positions_ang,
    )


def _valid_probe_modes(defocus=0.0):
    probe = make_probe(
        _microscope(defocus=defocus),
        _detector(),
    )
    return create_probe_modes(
        probe[..., jnp.newaxis],
        jnp.ones((1,), dtype=jnp.float64),
        _CALIB_ANG,
    )


def _sharded_inputs(scale=1.0):
    probe_modes = _valid_probe_modes().modes
    scan_positions_ang = jnp.array(
        [[0.0, 0.0], [1.0, 1.5]],
        dtype=jnp.float64,
    )
    atom_coords = jnp.array(
        [[2.0, 2.5, 0.25], [4.5, 5.0, 1.25]],
        dtype=jnp.float64,
    )
    atom_types = jnp.array([0, 0], dtype=jnp.int32)
    slice_z_bounds = jnp.array(
        [[0.0, 1.0], [1.0, 2.0]],
        dtype=jnp.float64,
    )
    atom_potentials = _potential_data(scale)[..., 0][jnp.newaxis, ...]
    return (
        probe_modes,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
    )


def test_checked_make_probe_transparent_jit_grad_and_raises():
    microscope = _microscope()
    detector = _detector()
    expected = make_probe(microscope, detector)
    actual = checked_make_probe(microscope, detector)
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(
        lambda defocus: checked_make_probe(
            _microscope(defocus=defocus),
            detector,
        )
    )(jnp.asarray(3.0, dtype=jnp.float64))
    expected_jitted = make_probe(
        _microscope(defocus=jnp.asarray(3.0, dtype=jnp.float64)),
        detector,
    )
    _assert_tree_allclose(jitted, expected_jitted)

    grad_value = jax.grad(
        lambda defocus: jnp.sum(
            jnp.abs(
                checked_make_probe(
                    _microscope(defocus=defocus),
                    detector,
                )
            )
            ** 2
        )
    )(jnp.asarray(0.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    with pytest.raises(
        EquinoxRuntimeError,
        match="voltage_kv must be positive",
    ):
        checked_make_probe(
            create_microscope_config(
                voltage_kv=jnp.asarray(-80.0, dtype=jnp.float64),
                aperture_mrad=_APERTURE,
                probe_shape=(_GRID_SIZE, _GRID_SIZE),
            ),
            detector,
        )


def test_checked_cbed_image_transparent_jit_grad_vmap_and_raises():
    pot_slices = _valid_potential_slices()
    beam = _valid_probe_modes()
    microscope = _microscope()

    expected = cbed_image(pot_slices, beam, microscope)
    actual = checked_cbed_image(pot_slices, beam, microscope)
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(checked_cbed_image)(pot_slices, beam, microscope)
    _assert_tree_array_equal(jitted, expected)

    grad_value = jax.grad(
        lambda scale: jnp.sum(
            checked_cbed_image(
                _valid_potential_slices(scale),
                beam,
                microscope,
            ).data_array
        )
    )(jnp.asarray(1.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    batched_slices = jnp.stack(
        [pot_slices.slices, pot_slices.slices * 0.25],
        axis=0,
    )
    vmapped = jax.vmap(
        lambda slices: checked_cbed_image(
            create_potential_slices(
                slices,
                pot_slices.slice_thickness,
                pot_slices.calib,
            ),
            beam,
            microscope,
        ).data_array
    )(batched_slices)
    assert vmapped.shape == (2, _GRID_SIZE, _GRID_SIZE)
    _assert_finite(vmapped)

    bad_pot_slices = eqx.tree_at(
        lambda value: value.slices,
        pot_slices,
        pot_slices.slices.at[0, 0, 0].set(jnp.nan),
    )
    with pytest.raises(
        EquinoxRuntimeError,
        match="pot_slices.slices contain non-finite values",
    ):
        checked_cbed_image(bad_pot_slices, beam, microscope)


def test_checked_stem_4d_transparent_jit_grad_and_raises():
    pot_slice = _valid_potential_slices()
    beam = _valid_probe_modes()
    microscope = _microscope()
    detector = _detector(scan_positions_px=_POSITIONS)

    expected = stem_4d(pot_slice, beam, microscope, detector)
    actual = checked_stem_4d(
        pot_slice,
        beam,
        microscope,
        detector,
    )
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(checked_stem_4d)(
        pot_slice,
        beam,
        microscope,
        detector,
    )
    _assert_tree_array_equal(jitted, expected)

    grad_value = jax.grad(
        lambda scale: jnp.sum(
            checked_stem_4d(
                _valid_potential_slices(scale),
                beam,
                microscope,
                detector,
            ).data
        )
    )(jnp.asarray(1.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    bad_positions = _POSITIONS.at[1, 0].set(_GRID_SIZE)
    bad_detector = _detector(scan_positions_px=bad_positions)
    with pytest.raises(
        EquinoxRuntimeError,
        match="positions must be within pot_slice grid bounds",
    ):
        checked_stem_4d(
            pot_slice,
            beam,
            microscope,
            bad_detector,
        )


def test_checked_stem4d_sharded_transparent_jit_grad_and_raises():
    (
        probe_mode_array,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
    ) = _sharded_inputs()
    probe_modes = create_probe_modes(
        probe_mode_array,
        jnp.ones((probe_mode_array.shape[-1],), dtype=jnp.float64),
        _CALIB_ANG,
    )
    sample = create_atomic_slice_data(
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
    )
    microscope = _microscope()
    detector = _detector(scan_positions_ang=scan_positions_ang)

    expected = stem4d_sharded(
        probe_modes,
        sample,
        microscope,
        detector,
    )
    actual = checked_stem4d_sharded(
        probe_modes,
        sample,
        microscope,
        detector,
    )
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(checked_stem4d_sharded)(
        probe_modes,
        sample,
        microscope,
        detector,
    )
    _assert_tree_array_equal(jitted, expected)

    grad_value = jax.grad(
        lambda scale: jnp.sum(
            checked_stem4d_sharded(
                probe_modes,
                create_atomic_slice_data(
                    atom_coords,
                    atom_types,
                    slice_z_bounds,
                    atom_potentials * scale,
                ),
                microscope,
                detector,
            ).data
        )
    )(jnp.asarray(1.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    bad_scan_positions = scan_positions_ang.at[1, 0].set(8.1)
    bad_detector = _detector(scan_positions_ang=bad_scan_positions)
    with pytest.raises(
        EquinoxRuntimeError,
        match="scan_positions_ang must be within atom_potentials grid bounds",
    ):
        checked_stem4d_sharded(
            probe_modes,
            sample,
            microscope,
            bad_detector,
        )
