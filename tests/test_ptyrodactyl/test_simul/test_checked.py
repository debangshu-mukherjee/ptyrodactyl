"""Tests for :mod:`ptyrodactyl.simul.checked`."""
# ruff: noqa: E402, I001

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from equinox import EquinoxRuntimeError

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.simul import (
    checked_cbed,
    checked_make_probe,
    checked_stem4d_sharded,
    checked_stem_4d,
)
from ptyrodactyl.simul.parallelized import stem4d_sharded
from ptyrodactyl.simul.simulations import cbed, make_probe, stem_4d
from ptyrodactyl.types import PotentialSlices, ProbeModes


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
    return PotentialSlices(
        slices=_potential_data(scale),
        slice_thickness=_SLICE_THICKNESS,
        calib=_CALIB_ANG,
    )


def _valid_probe_modes(defocus=0.0):
    probe = make_probe(
        _APERTURE,
        _VOLTAGE,
        _IMAGE_SIZE,
        _CALIB_PM,
        defocus=defocus,
    )
    return ProbeModes(
        modes=probe[..., jnp.newaxis],
        weights=jnp.ones((1,), dtype=jnp.float64),
        calib=_CALIB_ANG,
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
    expected = make_probe(_APERTURE, _VOLTAGE, _IMAGE_SIZE, _CALIB_PM)
    actual = checked_make_probe(_APERTURE, _VOLTAGE, _IMAGE_SIZE, _CALIB_PM)
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(
        lambda defocus: checked_make_probe(
            _APERTURE,
            _VOLTAGE,
            _IMAGE_SIZE,
            _CALIB_PM,
            defocus=defocus,
        )
    )(jnp.asarray(3.0, dtype=jnp.float64))
    expected_jitted = make_probe(
        _APERTURE,
        _VOLTAGE,
        _IMAGE_SIZE,
        _CALIB_PM,
        defocus=jnp.asarray(3.0, dtype=jnp.float64),
    )
    _assert_tree_allclose(jitted, expected_jitted)

    grad_value = jax.grad(
        lambda defocus: jnp.sum(
            jnp.abs(
                checked_make_probe(
                    _APERTURE,
                    _VOLTAGE,
                    _IMAGE_SIZE,
                    _CALIB_PM,
                    defocus=defocus,
                )
            )
            ** 2
        )
    )(jnp.asarray(0.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    with pytest.raises(EquinoxRuntimeError, match="voltage must be positive"):
        checked_make_probe(
            _APERTURE,
            jnp.asarray(-80.0, dtype=jnp.float64),
            _IMAGE_SIZE,
            _CALIB_PM,
        )


def test_checked_cbed_transparent_jit_grad_vmap_and_raises():
    pot_slices = _valid_potential_slices()
    beam = _valid_probe_modes()

    expected = cbed(pot_slices, beam, _VOLTAGE)
    actual = checked_cbed(pot_slices, beam, _VOLTAGE)
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(checked_cbed)(pot_slices, beam, _VOLTAGE)
    _assert_tree_array_equal(jitted, expected)

    grad_value = jax.grad(
        lambda scale: jnp.sum(
            checked_cbed(
                _valid_potential_slices(scale),
                beam,
                _VOLTAGE,
            ).data_array
        )
    )(jnp.asarray(1.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    batched_slices = jnp.stack(
        [pot_slices.slices, pot_slices.slices * 0.25],
        axis=0,
    )
    vmapped = jax.vmap(
        lambda slices: checked_cbed(
            PotentialSlices(
                slices=slices,
                slice_thickness=pot_slices.slice_thickness,
                calib=pot_slices.calib,
            ),
            beam,
            _VOLTAGE,
        ).data_array
    )(batched_slices)
    assert vmapped.shape == (2, _GRID_SIZE, _GRID_SIZE)
    _assert_finite(vmapped)

    bad_pot_slices = PotentialSlices(
        slices=pot_slices.slices.at[0, 0, 0].set(jnp.nan),
        slice_thickness=pot_slices.slice_thickness,
        calib=pot_slices.calib,
    )
    with pytest.raises(
        EquinoxRuntimeError,
        match="pot_slices.slices contain non-finite values",
    ):
        checked_cbed(bad_pot_slices, beam, _VOLTAGE)


def test_checked_stem_4d_transparent_jit_grad_and_raises():
    pot_slice = _valid_potential_slices()
    beam = _valid_probe_modes()

    expected = stem_4d(pot_slice, beam, _POSITIONS, _VOLTAGE, _CALIB_ANG)
    actual = checked_stem_4d(
        pot_slice,
        beam,
        _POSITIONS,
        _VOLTAGE,
        _CALIB_ANG,
    )
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(checked_stem_4d)(
        pot_slice,
        beam,
        _POSITIONS,
        _VOLTAGE,
        _CALIB_ANG,
    )
    _assert_tree_array_equal(jitted, expected)

    grad_value = jax.grad(
        lambda scale: jnp.sum(
            checked_stem_4d(
                _valid_potential_slices(scale),
                beam,
                _POSITIONS,
                _VOLTAGE,
                _CALIB_ANG,
            ).data
        )
    )(jnp.asarray(1.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    bad_positions = _POSITIONS.at[1, 0].set(_GRID_SIZE)
    with pytest.raises(
        EquinoxRuntimeError,
        match="positions must be within pot_slice grid bounds",
    ):
        checked_stem_4d(
            pot_slice,
            beam,
            bad_positions,
            _VOLTAGE,
            _CALIB_ANG,
        )


def test_checked_stem4d_sharded_transparent_jit_grad_and_raises():
    (
        probe_modes,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
    ) = _sharded_inputs()

    expected = stem4d_sharded(
        probe_modes,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
        _VOLTAGE,
        _CALIB_ANG,
    )
    actual = checked_stem4d_sharded(
        probe_modes,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
        _VOLTAGE,
        _CALIB_ANG,
    )
    _assert_tree_array_equal(actual, expected)

    jitted = jax.jit(checked_stem4d_sharded)(
        probe_modes,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
        _VOLTAGE,
        _CALIB_ANG,
    )
    _assert_tree_array_equal(jitted, expected)

    grad_value = jax.grad(
        lambda scale: jnp.sum(
            checked_stem4d_sharded(
                probe_modes,
                scan_positions_ang,
                atom_coords,
                atom_types,
                slice_z_bounds,
                atom_potentials * scale,
                _VOLTAGE,
                _CALIB_ANG,
            ).data
        )
    )(jnp.asarray(1.0, dtype=jnp.float64))
    _assert_finite(grad_value)

    bad_scan_positions = scan_positions_ang.at[1, 0].set(8.1)
    with pytest.raises(
        EquinoxRuntimeError,
        match="scan_positions_ang must be within atom_potentials grid bounds",
    ):
        checked_stem4d_sharded(
            probe_modes,
            bad_scan_positions,
            atom_coords,
            atom_types,
            slice_z_bounds,
            atom_potentials,
            _VOLTAGE,
            _CALIB_ANG,
        )
