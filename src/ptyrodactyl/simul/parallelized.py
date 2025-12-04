"""Parallelized simulation functions for distributed electron microscopy.

Extended Summary
----------------
This module provides sharded versions of simulation functions that leverage
JAX's distributed computing capabilities for large-scale electron microscopy
simulations. Functions accept pre-sharded arrays for efficient parallel
execution across multiple devices.

Routine Listings
----------------
_compute_slice_potential : function, internal
    Compute potential slice on-the-fly by summing atom type contributions.
_cbed_from_potential_slices : function, internal
    Compute CBED pattern with on-the-fly potential slice generation.
stem4d_sharded : function
    Generate 4D-STEM data from sharded beams and atom coordinates.

Notes
-----
All functions are fully JAX-safe and JIT-compilable. They are designed for
use with JAX's pjit/shard_map for distributed execution across TPU/GPU pods.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from ptyrodactyl.tools import (
    STEM4D,
    ScalarFloat,
    ScalarInt,
    ScalarNumeric,
    make_stem4d,
)

from .simulations import (
    propagation_func,
    transmission_func,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
@jax.jit
def _compute_slice_potential(
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    z_min: ScalarFloat,
    z_max: ScalarFloat,
    atom_potentials: Float[Array, "T H W"],
    grid_shape: Tuple[int, int],
    calib_ang: ScalarFloat,
) -> Float[Array, "H W"]:
    """Compute potential slice on-the-fly by summing atom type contributions.

    Parameters
    ----------
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in angstroms with columns (x, y, z).
    atom_types : Int[Array, " N"]
        Atom type indices (0-indexed) for each atom.
    z_min : ScalarFloat
        Minimum z coordinate for this slice in angstroms.
    z_max : ScalarFloat
        Maximum z coordinate for this slice in angstroms.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each atom type.
        T is the number of unique atom types.
    grid_shape : Tuple[int, int]
        Output grid shape (height, width).
    calib_ang : ScalarFloat
        Pixel size in angstroms.

    Returns
    -------
    slice_potential : Float[Array, "H W"]
        The computed potential slice.
    """
    h: int
    w: int
    h, w = grid_shape
    num_types: int = atom_potentials.shape[0]

    in_slice: Float[Array, " N"] = (
        (atom_coords[:, 2] >= z_min) & (atom_coords[:, 2] < z_max)
    ).astype(jnp.float64)

    def _process_atom_type(
        atom_type_idx: ScalarInt,
    ) -> Float[Array, "H W"]:
        """Process contribution from a single atom type."""
        type_mask: Float[Array, " N"] = (
            atom_types == atom_type_idx
        ) * in_slice

        x_pixels: Float[Array, " N"] = atom_coords[:, 0] / calib_ang
        y_pixels: Float[Array, " N"] = atom_coords[:, 1] / calib_ang

        x_idx: Int[Array, " N"] = jnp.floor(x_pixels).astype(jnp.int32) % w
        y_idx: Int[Array, " N"] = jnp.floor(y_pixels).astype(jnp.int32) % h

        positions_grid: Float[Array, "H W"] = jnp.zeros(
            (h, w), dtype=jnp.float64
        )
        positions_grid = positions_grid.at[y_idx, x_idx].add(type_mask)

        positions_k: Complex[Array, "H W"] = jnp.fft.fft2(positions_grid)
        potential_k: Complex[Array, "H W"] = jnp.fft.fft2(
            atom_potentials[atom_type_idx]
        )
        convolved_k: Complex[Array, "H W"] = positions_k * potential_k
        convolved: Float[Array, "H W"] = jnp.real(jnp.fft.ifft2(convolved_k))

        return convolved

    type_contributions: Float[Array, "T H W"] = jax.vmap(_process_atom_type)(
        jnp.arange(num_types)
    )
    slice_potential: Float[Array, "H W"] = jnp.sum(type_contributions, axis=0)

    return slice_potential


@jaxtyped(typechecker=beartype)
@jax.jit
def _cbed_from_potential_slices(
    beam: Complex[Array, "H W M"],
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    voltage_kv: ScalarNumeric,
    calib_ang: ScalarFloat,
) -> Float[Array, "H W"]:
    """Compute CBED pattern with on-the-fly potential slice generation.

    Parameters
    ----------
    beam : Complex[Array, "H W M"]
        Electron beam modes in real space.
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in angstroms with columns (x, y, z).
    atom_types : Int[Array, " N"]
        Atom type indices (0-indexed) for each atom.
    slice_z_bounds : Float[Array, "S 2"]
        Z boundaries for each slice with columns (z_min, z_max).
        S is the number of slices.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each atom type.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    calib_ang : ScalarFloat
        Pixel size in angstroms.

    Returns
    -------
    cbed_pattern : Float[Array, "H W"]
        The computed CBED intensity pattern.
    """
    h: int
    w: int
    h, w = beam.shape[0], beam.shape[1]
    num_slices: int = slice_z_bounds.shape[0]
    grid_shape: Tuple[int, int] = (h, w)

    slice_thickness: Float[Array, " "] = (
        slice_z_bounds[0, 1] - slice_z_bounds[0, 0]
    )

    propagator: Complex[Array, "H W"] = propagation_func(
        h, w, slice_thickness, voltage_kv, calib_ang
    )

    init_wave: Complex[Array, "H W M"] = beam

    def _scan_fn(
        carry: Complex[Array, "H W M"], slice_idx: ScalarInt
    ) -> Tuple[Complex[Array, "H W M"], None]:
        """Propagate wave through a single potential slice."""
        wave: Complex[Array, "H W M"] = carry

        z_min: Float[Array, " "] = slice_z_bounds[slice_idx, 0]
        z_max: Float[Array, " "] = slice_z_bounds[slice_idx, 1]

        pot_slice: Float[Array, "H W"] = _compute_slice_potential(
            atom_coords,
            atom_types,
            z_min,
            z_max,
            atom_potentials,
            grid_shape,
            calib_ang,
        )

        trans_slice: Complex[Array, "H W"] = transmission_func(
            pot_slice, voltage_kv
        )
        wave = wave * trans_slice[..., jnp.newaxis]

        def _propagate(w: Complex[Array, "H W M"]) -> Complex[Array, "H W M"]:
            """Apply Fresnel propagation in Fourier space."""
            w_k: Complex[Array, "H W M"] = jnp.fft.fft2(w, axes=(0, 1))
            w_k = w_k * propagator[..., jnp.newaxis]
            return jnp.fft.ifft2(w_k, axes=(0, 1))

        is_last_slice: jnp.bool_ = slice_idx == num_slices - 1
        wave = lax.cond(is_last_slice, lambda w: w, _propagate, wave)

        return wave, None

    final_wave: Complex[Array, "H W M"]
    final_wave, _ = lax.scan(_scan_fn, init_wave, jnp.arange(num_slices))

    fourier_pattern: Complex[Array, "H W M"] = jnp.fft.fftshift(
        jnp.fft.fft2(final_wave, axes=(0, 1)), axes=(0, 1)
    )
    intensity_per_mode: Float[Array, "H W M"] = jnp.square(
        jnp.abs(fourier_pattern)
    )
    cbed_pattern: Float[Array, "H W"] = jnp.sum(intensity_per_mode, axis=-1)

    return cbed_pattern


@jaxtyped(typechecker=beartype)
@jax.jit
def stem4d_sharded(
    sharded_multimodal_beams: Complex[Array, "P H W M"],
    sharded_positions: Num[Array, "P 2"],
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    voltage_kv: ScalarNumeric,
    calib_ang: ScalarFloat,
) -> STEM4D:
    """Generate 4D-STEM data from sharded beams with on-the-fly slices.

    This function accepts pre-shifted beams and computes CBED patterns for
    each position using on-the-fly potential slice generation. Designed for
    efficient parallel execution with JAX sharding.

    Parameters
    ----------
    sharded_multimodal_beams : Complex[Array, "P H W M"]
        Pre-shifted electron beam modes for each scan position.
        P is number of positions, H and W are image dimensions, M is modes.
        Beams are sharded along the first axis (P)
    sharded_positions : Num[Array, "P 2"]
        Scan positions in pixels with columns (y, x).
        Positions are sharded along the first axis (P)
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in angstroms with columns (x, y, z).
        N is the total number of atoms.
    atom_types : Int[Array, " N"]
        Atom type indices (0-indexed) for each atom, maps to atom_potentials.
    slice_z_bounds : Float[Array, "S 2"]
        Z boundaries for each slice with columns (z_min, z_max).
        S is the number of slices through the sample.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each unique atom type.
        T is the number of unique atom types in the sample.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    calib_ang : ScalarFloat
        Real space pixel size in angstroms.

    Returns
    -------
    stem4d_data_sharded : STEM4D
        Complete 4D-STEM dataset containing:
        - data : Float[Array, "P H W"]
            Diffraction patterns for each scan position
            Sharded along the first axis (P).
        - real_space_calib : Float[Array, " "]
            Real space calibration in angstroms per pixel
            Not sharded, scalar value.
        - fourier_space_calib : Float[Array, " "]
            Fourier space calibration in inverse angstroms per pixel
            Not sharded, scalar value.
        - scan_positions : Float[Array, "P 2"]
            Scan positions in angstroms.
            Sharded along the first axis (P).
        - voltage_kv : Float[Array, " "]
            Accelerating voltage in kilovolts.
            Not sharded, scalar value.

    Notes
    -----
    This function generates potential slices on-the-fly rather than
    pre-computing them, enabling memory-efficient simulation of thick samples.

    Algorithm:
    1. For each scan position, compute CBED with on-the-fly slice generation
    2. Each slice potential is computed by:
       - Selecting atoms within the slice z-range
       - Scattering atom positions to a grid by type
       - FFT-convolving with precomputed atomic potentials
       - Summing contributions from all atom types
    3. Propagate beam through all slices using multislice algorithm
    4. Return STEM4D PyTree with all data and calibrations

    The function is fully JIT-compilable and designed for use with JAX's
    sharding primitives (pjit, shard_map) for distributed execution.
    """
    h: int = sharded_multimodal_beams.shape[1]

    def _process_single_position(pos_idx: ScalarInt) -> Float[Array, "H W"]:
        """Compute CBED pattern for a single beam position."""
        current_beam: Complex[Array, "H W M"] = sharded_multimodal_beams[
            pos_idx
        ]

        cbed_pattern: Float[Array, "H W"] = _cbed_from_potential_slices(
            beam=current_beam,
            atom_coords=atom_coords,
            atom_types=atom_types,
            slice_z_bounds=slice_z_bounds,
            atom_potentials=atom_potentials,
            voltage_kv=voltage_kv,
            calib_ang=calib_ang,
        )
        return cbed_pattern

    cbed_patterns: Float[Array, "P H W"] = jax.vmap(_process_single_position)(
        jnp.arange(sharded_multimodal_beams.shape[0])
    )

    real_space_fov: Float[Array, " "] = jnp.asarray(h * calib_ang)
    fourier_calib: Float[Array, " "] = 1.0 / real_space_fov

    scan_positions_ang: Float[Array, "P 2"] = sharded_positions * calib_ang

    stem4d_data_sharded: STEM4D = make_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kv=voltage_kv,
    )

    return stem4d_data_sharded
