"""Parallelized simulation functions for distributed microscopy.

Extended Summary
----------------
This module provides sharded versions of simulation functions
that leverage JAX's distributed computing capabilities for
large-scale electron microscopy simulations. Functions accept
pre-sharded arrays for efficient parallel execution across
multiple devices.

Routine Listings
----------------
:func:`_compute_slice_potential`
    Compute potential slice on-the-fly by summing atom type
    contributions.
:func:`_cbed_from_potential_slices`
    Compute CBED pattern with on-the-fly potential slice
    generation.
:func:`clip_cbed`
    Clip CBED patterns to mrad extent and resize to target
    shape.
:func:`stem4d_sharded`
    Generate 4D-STEM data from sharded beams and atom
    coordinates.

Notes
-----
All functions are fully JAX-safe and JIT-compilable. They are
designed for use with JAX's ``shard_map`` for distributed
execution across TPU/GPU pods.
"""

from functools import partial

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
@partial(jax.jit, static_argnames=["grid_shape"])
def _compute_slice_potential(
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    z_min: ScalarFloat,
    z_max: ScalarFloat,
    atom_potentials: Float[Array, "T H W"],
    grid_shape: Tuple[int, int],
    calib_ang: ScalarFloat,
    atom_mask: Optional[Float[Array, " N"]] = None,
) -> Float[Array, "H W"]:
    """Compute a potential slice by summing atom type contributions.

    Extended Summary
    ----------------
    Generates a single potential slice on-the-fly by selecting
    atoms within the z-range, scattering their positions onto
    a grid per atom type, and FFT-convolving with precomputed
    atomic potentials.

    Implementation Logic
    --------------------
    1. **Select atoms in z-range** --
       Mask atoms with ``z_min <= z < z_max``, optionally
       combined with ``atom_mask``.
    2. **Per-type convolution** --
       For each atom type, scatter positions to a delta grid,
       FFT-convolve with the precomputed potential kernel.
    3. **Sum contributions** --
       Sum convolved results across all atom types.

    Parameters
    ----------
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in Angstroms, columns ``(x, y, z)``.
    atom_types : Int[Array, " N"]
        Atom type indices (0-indexed) for each atom.
    z_min : ScalarFloat
        Minimum z coordinate for this slice in Angstroms.
    z_max : ScalarFloat
        Maximum z coordinate for this slice in Angstroms.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each atom type.
        T is the number of unique atom types.
    grid_shape : Tuple[int, int]
        Output grid shape ``(height, width)``.
    calib_ang : ScalarFloat
        Pixel size in Angstroms.
    atom_mask : Optional[Float[Array, " N"]]
        Mask for atoms to include (1.0 = include,
        0.0 = exclude). If ``None``, all atoms are included.

    Returns
    -------
    slice_potential : Float[Array, "H W"]
        The computed potential slice in Kirkland units.
    """
    h: int
    w: int
    h, w = grid_shape
    num_types: int = atom_potentials.shape[0]

    in_slice: Float[Array, " N"] = (
        (atom_coords[:, 2] >= z_min) & (atom_coords[:, 2] < z_max)
    ).astype(jnp.float64)

    if atom_mask is not None:
        in_slice = in_slice * atom_mask

    def _process_atom_type(
        atom_type_idx: ScalarInt,
    ) -> Float[Array, "H W"]:
        """Compute potential contribution from one atom type.

        Parameters
        ----------
        atom_type_idx : ScalarInt
            Index into ``atom_potentials`` for this type.

        Returns
        -------
        convolved : Float[Array, "H W"]
            FFT-convolved potential contribution.
        """
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
    atom_mask: Optional[Float[Array, " N"]] = None,
) -> Float[Array, "H W"]:
    """Compute CBED pattern with on-the-fly slice generation.

    Extended Summary
    ----------------
    Propagates electron beam modes through the sample using
    the multislice algorithm. Potential slices are generated
    on-the-fly via :func:`_compute_slice_potential` rather
    than pre-computed, enabling memory-efficient simulation.

    Implementation Logic
    --------------------
    1. **Build propagator** --
       Compute Fresnel propagator from slice thickness via
       :func:`~ptyrodactyl.simul.simulations.propagation_func`.
    2. **Scan over slices** --
       For each slice, generate the potential on-the-fly,
       compute the transmission function, multiply the wave,
       and propagate (skip propagation on the last slice).
    3. **Compute intensity** --
       FFT to Fourier space, square modulus, sum over modes.

    Parameters
    ----------
    beam : Complex[Array, "H W M"]
        Electron beam modes in real space.
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in Angstroms, columns ``(x, y, z)``.
    atom_types : Int[Array, " N"]
        Atom type indices (0-indexed) for each atom.
    slice_z_bounds : Float[Array, "S 2"]
        Z boundaries for each slice, columns
        ``(z_min, z_max)`` in Angstroms. S is the number of
        slices.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials per atom type.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    calib_ang : ScalarFloat
        Pixel size in Angstroms.
    atom_mask : Optional[Float[Array, " N"]]
        Mask for atoms (1.0 = include, 0.0 = exclude). Used
        by tiled workflows to exclude atoms outside the
        current tile. If ``None``, all atoms are included.

    Returns
    -------
    cbed_pattern : Float[Array, "H W"]
        Computed CBED intensity pattern (sum over modes).
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
        """Propagate wave through one potential slice.

        Parameters
        ----------
        carry : Complex[Array, "H W M"]
            Current wave state.
        slice_idx : ScalarInt
            Index of the current slice.

        Returns
        -------
        wave : Complex[Array, "H W M"]
            Updated wave after transmission and propagation.
        None
            No stacked output.
        """
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
            atom_mask,
        )

        trans_slice: Complex[Array, "H W"] = transmission_func(
            pot_slice, voltage_kv
        )
        wave = wave * trans_slice[..., jnp.newaxis]

        def _propagate(w: Complex[Array, "H W M"]) -> Complex[Array, "H W M"]:
            """Apply Fresnel propagation in Fourier space.

            Parameters
            ----------
            w : Complex[Array, "H W M"]
                Wave in real space.

            Returns
            -------
            Complex[Array, "H W M"]
                Wave after Fresnel propagation.
            """
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
    """Clip CBED pattern to mrad extent and resize.

    Extended Summary
    ----------------
    Extracts the central region of a CBED pattern corresponding
    to a given angular extent in milliradians, then resizes to
    the target output shape using bilinear interpolation.

    Implementation Logic
    --------------------
    1. **Convert mrad to pixels** --
       Use wavelength and Fourier calibration to convert
       ``extent_mrad`` to a pixel radius.
    2. **Extract central crop** --
       ``lax.dynamic_slice`` around the pattern center.
    3. **Resize** --
       Bilinear resize to ``output_shape``.

    Parameters
    ----------
    cbed : Float[Array, "H W"]
        Input CBED pattern (fftshifted, centered).
    fourier_calib_inv_ang : ScalarFloat
        Fourier space calibration in inverse Angstroms per
        pixel.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    extent_mrad : ScalarFloat
        Half-angle extent in milliradians (radius from
        center).
    output_shape : Tuple[int, int]
        Target output shape ``(height, width)``.

    Returns
    -------
    resized : Float[Array, "Ho Wo"]
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

    Extended Summary
    ----------------
    Accepts base probe modes and scan positions, then shifts
    the beams on-the-fly for each position. Potential slices
    are also generated on-the-fly, enabling memory-efficient
    simulation of large datasets. Fully JIT-compilable and
    designed for use with JAX's sharding primitives.

    Implementation Logic
    --------------------
    1. **Pre-compute Fourier quantities** --
       FFT the probe and build frequency grids (once).
    2. **Per-position processing** --
       For each scan position, apply a Fourier phase ramp to
       shift the probe, then compute the CBED pattern via
       :func:`_cbed_from_potential_slices`.
    3. **Distributed execution** --
       If *mesh* is provided, use ``jax.shard_map`` to
       distribute positions across devices; otherwise use
       ``jax.vmap``.
    4. **Build output** --
       Return :class:`~ptyrodactyl.tools.STEM4D` PyTree with
       data, calibrations, and scan positions.

    Parameters
    ----------
    probe_modes : Complex[Array, "H W M"]
        Base electron probe modes (unshifted). H and W are
        image dimensions, M is number of modes.
    scan_positions_ang : Float[Array, "P 2"]
        Scan positions in Angstroms, columns ``(y, x)``.
        P is the number of positions. Can be sharded along
        the first axis.
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in Angstroms, columns ``(x, y, z)``.
    atom_types : Int[Array, " N"]
        Atom type indices (0-indexed), maps to
        *atom_potentials*.
    slice_z_bounds : Float[Array, "S 2"]
        Z boundaries per slice, columns ``(z_min, z_max)``
        in Angstroms.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each unique
        atom type.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    calib_ang : ScalarFloat
        Real-space pixel size in Angstroms.
    mesh : Optional[Mesh]
        JAX device mesh for multi-GPU parallelism. If
        provided, uses ``shard_map``. If ``None``, uses
        single-device ``vmap``.

    Returns
    -------
    stem4d_data_sharded : STEM4D
        Complete 4D-STEM dataset containing diffraction
        patterns, real- and Fourier-space calibrations,
        scan positions, and accelerating voltage.

    See Also
    --------
    :func:`clip_cbed` : Clip and resize CBED patterns to
        target mrad extent and shape.
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
        """Shift probe modes via Fourier phase ramp.

        Parameters
        ----------
        position_ang : Float[Array, " 2"]
            Target position ``(y, x)`` in Angstroms.

        Returns
        -------
        shifted_beam : Complex[Array, "H W M"]
            Probe modes shifted to the target position.
        """
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
        """Compute CBED pattern for a single scan position.

        Parameters
        ----------
        position_ang : Float[Array, " 2"]
            Scan position ``(y, x)`` in Angstroms.

        Returns
        -------
        cbed_pattern : Float[Array, "H W"]
            CBED intensity pattern at this position.
        """
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
        """Process a batch of positions (one shard).

        Parameters
        ----------
        positions_batch : Float[Array, "B 2"]
            Batch of scan positions in Angstroms.

        Returns
        -------
        Float[Array, "B H W"]
            CBED patterns for the batch.
        """
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


__all__: list[str] = [
    "clip_cbed",
    "stem4d_sharded",
]
