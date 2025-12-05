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
clip_cbed : function
    Clip CBED patterns to mrad extent and resize to target shape.
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
from beartype.typing import Optional, Tuple
from jax import lax
from jax.image import resize
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Complex, Float, Int, jaxtyped

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
def clip_cbed(
    cbed: Float[Array, "H W"],
    fourier_calib_inv_ang: ScalarFloat,
    voltage_kv: ScalarNumeric,
    extent_mrad: ScalarFloat,
    output_shape: Tuple[int, int],
) -> Float[Array, "Ho Wo"]:
    """Clip CBED pattern to mrad extent and resize to target shape.

    Parameters
    ----------
    cbed : Float[Array, "H W"]
        Input CBED pattern (fftshifted, centered).
    fourier_calib_inv_ang : ScalarFloat
        Fourier space calibration in inverse angstroms per pixel.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    extent_mrad : ScalarFloat
        Half-angle extent in milliradians (radius from center).
    output_shape : Tuple[int, int]
        Target output shape (height, width).

    Returns
    -------
    clipped_cbed : Float[Array, "Ho Wo"]
        Clipped and resized CBED pattern.
    """
    h: int = cbed.shape[0]
    w: int = cbed.shape[1]

    wavelength_ang: Float[Array, " "] = 12.2643 / jnp.sqrt(
        voltage_kv * (1.0 + 0.978459e-3 * voltage_kv)
    )
    mrad_per_inv_ang: Float[Array, " "] = wavelength_ang * 1000.0

    extent_inv_ang: Float[Array, " "] = extent_mrad / mrad_per_inv_ang
    extent_pixels: Int[Array, " "] = jnp.ceil(
        extent_inv_ang / fourier_calib_inv_ang
    ).astype(jnp.int32)

    center_y: int = h // 2
    center_x: int = w // 2

    y_start: Int[Array, " "] = jnp.maximum(0, center_y - extent_pixels)
    y_end: Int[Array, " "] = jnp.minimum(h, center_y + extent_pixels)
    x_start: Int[Array, " "] = jnp.maximum(0, center_x - extent_pixels)
    x_end: Int[Array, " "] = jnp.minimum(w, center_x + extent_pixels)

    clipped: Float[Array, "Hc Wc"] = lax.dynamic_slice(
        cbed,
        (y_start, x_start),
        (y_end - y_start, x_end - x_start),
    )

    resized: Float[Array, "Ho Wo"] = resize(
        clipped,
        output_shape,
        method="linear",
    )

    return resized


@jaxtyped(typechecker=beartype)
def stem4d_sharded(
    probe_modes: Complex[Array, "H W M"],
    scan_positions_ang: Float[Array, "P 2"],
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    voltage_kv: ScalarNumeric,
    calib_ang: ScalarFloat,
    mesh: Optional[Mesh] = None,
) -> STEM4D:
    """Generate 4D-STEM data with on-the-fly beam shifting and slices.

    This function accepts base probe modes and scan positions, then shifts the
    beams on-the-fly for each position. Potential slices are also generated
    on-the-fly, enabling memory-efficient simulation of large datasets.

    Parameters
    ----------
    probe_modes : Complex[Array, "H W M"]
        Base electron probe modes (unshifted).
        H and W are image dimensions, M is number of modes.
    scan_positions_ang : Float[Array, "P 2"]
        Scan positions in angstroms with columns (y, x).
        P is number of positions. Can be sharded along the first axis.
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
    mesh : Optional[Mesh]
        JAX device mesh for multi-GPU parallelism. If provided, uses shard_map
        to distribute computation across devices. If None, uses single-device
        vmap.

    Returns
    -------
    stem4d_data_sharded : STEM4D
        Complete 4D-STEM dataset containing:
        - data : Float[Array, "P H W"]
            Diffraction patterns for each scan position
            Sharded along the first axis (P) if input was sharded.
        - real_space_calib : Float[Array, " "]
            Real space calibration in angstroms per pixel.
        - fourier_space_calib : Float[Array, " "]
            Fourier space calibration in inverse angstroms per pixel.
        - scan_positions : Float[Array, "P 2"]
            Scan positions in angstroms.
        - voltage_kv : Float[Array, " "]
            Accelerating voltage in kilovolts.

    Notes
    -----
    This function generates both beam shifts and potential slices on-the-fly
    rather than pre-computing them, enabling memory-efficient simulation of
    large scan grids and thick samples.

    Algorithm:
    1. Pre-compute probe in Fourier space and frequency grids (once)
    2. For each scan position:
       a. Apply Fourier shift to probe modes for current position
       b. Compute CBED with on-the-fly slice generation
    3. Each slice potential is computed by:
       - Selecting atoms within the slice z-range
       - Scattering atom positions to a grid by type
       - FFT-convolving with precomputed atomic potentials
       - Summing contributions from all atom types
    4. Propagate beam through all slices using multislice algorithm
    5. Return STEM4D PyTree with all data and calibrations

    The function is fully JIT-compilable and designed for use with JAX's
    sharding primitives for distributed execution.

    See Also
    --------
    clip_cbed : Clip and resize CBED patterns to target mrad extent and shape.
    """
    h: int = probe_modes.shape[0]
    w: int = probe_modes.shape[1]

    probe_k: Complex[Array, "H W M"] = jnp.fft.fft2(probe_modes, axes=(0, 1))
    qy: Float[Array, " H"] = jnp.fft.fftfreq(h, d=calib_ang)
    qx: Float[Array, " W"] = jnp.fft.fftfreq(w, d=calib_ang)
    qya: Float[Array, "H W"]
    qxa: Float[Array, "H W"]
    qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")

    def _shift_probe(
        position_ang: Float[Array, " 2"]
    ) -> Complex[Array, "H W M"]:
        """Shift probe modes to a single position using Fourier phase ramp."""
        y_shift: ScalarFloat = position_ang[0]
        x_shift: ScalarFloat = position_ang[1]
        phase: Float[Array, "H W"] = (
            -2.0 * jnp.pi * ((qya * y_shift) + (qxa * x_shift))
        )
        phase_shift: Complex[Array, "H W"] = jnp.exp(1j * phase)
        shifted_k: Complex[Array, "H W M"] = probe_k * phase_shift[..., None]
        shifted_beam: Complex[Array, "H W M"] = jnp.fft.ifft2(
            shifted_k, axes=(0, 1)
        )
        return shifted_beam

    def _process_single_position(
        position_ang: Float[Array, " 2"]
    ) -> Float[Array, "H W"]:
        """Compute CBED pattern for a single scan position."""
        current_beam: Complex[Array, "H W M"] = _shift_probe(position_ang)

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

    def _process_batch(
        positions_batch: Float[Array, "B 2"]
    ) -> Float[Array, "B H W"]:
        """Process a batch of positions (one shard)."""
        return jax.vmap(_process_single_position)(positions_batch)

    if mesh is not None:
        sharded_compute = jax.shard_map(
            _process_batch,
            mesh=mesh,
            in_specs=(PartitionSpec("p", None),),
            out_specs=PartitionSpec("p", None, None),
        )
        cbed_patterns: Float[Array, "P H W"] = sharded_compute(
            scan_positions_ang
        )
    else:
        cbed_patterns = jax.vmap(_process_single_position)(scan_positions_ang)

    real_space_fov: Float[Array, " "] = jnp.asarray(h * calib_ang)
    fourier_calib: Float[Array, " "] = 1.0 / real_space_fov

    stem4d_data_sharded: STEM4D = make_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kv=voltage_kv,
    )

    return stem4d_data_sharded
