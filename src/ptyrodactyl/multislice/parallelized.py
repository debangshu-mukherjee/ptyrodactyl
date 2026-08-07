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
:func:`cbed_amplitude_from_atoms`
    Compute CBED amplitudes with on-the-fly potential slice generation.
:func:`cbed_image_from_atoms`
    Compute CBED intensity with on-the-fly potential slice generation.
:func:`stem4d_sharded`
    Generate 4D-STEM data from sharded beams and atom
    coordinates.

Notes
-----
All functions are fully JAX-safe and JIT-compilable. They are
designed for use with JAX's ``shard_map`` for distributed
execution across TPU/GPU pods.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from ptyrodactyl.multislice.simulations import (
    _cbed_amplitude_from_slice_provider,
    probe_modes_to_distribution,
)
from ptyrodactyl.types import (
    STEM4D,
    AtomicSliceData,
    DetectorConfig,
    Distribution,
    MicroscopeConfig,
    ProbeModes,
    create_stem4d,
    scalar_float,
    scalar_int,
    scalar_num,
)

from .reduce import apply_distribution


@jax.jit(static_argnames=["grid_shape"])
@jaxtyped(typechecker=beartype)
def _compute_slice_potential(
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    z_min: scalar_float,
    z_max: scalar_float,
    atom_potentials: Float[Array, "T H W"],
    grid_shape: Tuple[int, int],
    calib_ang: scalar_float,
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
    z_min : scalar_float
        Minimum z coordinate for this slice in Angstroms.
    z_max : scalar_float
        Maximum z coordinate for this slice in Angstroms.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each atom type.
        T is the number of unique atom types.
    grid_shape : Tuple[int, int]
        Output grid shape ``(height, width)``.
    calib_ang : scalar_float
        Pixel size in Angstroms.
    atom_mask : Optional[Float[Array, " N"]]
        Mask for atoms to include (1.0 = include,
        0.0 = exclude). If ``None``, all atoms are included.

    Returns
    -------
    slice_potential : Float[Array, "H W"]
        The computed projected potential slice in volt-Angstroms.
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
        atom_type_idx: scalar_int,
    ) -> Float[Array, "H W"]:
        """Compute potential contribution from one atom type.

        Parameters
        ----------
        atom_type_idx : scalar_int
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


@jax.jit
@jaxtyped(typechecker=beartype)
def cbed_amplitude_from_atoms(
    beam: Complex[Array, "H W M"],
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
    atom_mask: Optional[Float[Array, " N"]] = None,
) -> Complex[Array, "H W M"]:
    """Compute CBED detector amplitudes with on-the-fly slice generation.

    :see: cbed_image_from_atoms, stem4d_sharded.
    """
    h: int = beam.shape[0]
    w: int = beam.shape[1]
    grid_shape: Tuple[int, int] = (h, w)
    num_slices: int = slice_z_bounds.shape[0]
    slice_thickness: Float[Array, " "] = (
        slice_z_bounds[0, 1] - slice_z_bounds[0, 0]
    )

    def _slice_at(slice_idx: scalar_int) -> Float[Array, "H W"]:
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
        return pot_slice

    detector_amplitude: Complex[Array, "H W M"] = (
        _cbed_amplitude_from_slice_provider(
            beam,
            num_slices,
            slice_thickness,
            voltage_kv,
            calib_ang,
            _slice_at,
        )
    )
    return detector_amplitude


@jax.jit
@jaxtyped(typechecker=beartype)
def cbed_image_from_atoms(
    beam: Complex[Array, "H W M"],
    mode_distribution: Distribution,
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
    atom_mask: Optional[Float[Array, " N"]] = None,
) -> Float[Array, "H W"]:
    """Compute CBED intensity from atom slices through the reducer.

    ``mode_distribution`` indexes the retained mode axis of the complex
    amplitude exactly once. The shared reducer then performs the only
    detector ``|.|^2`` and weighted mode reduction on this path.

    :see: cbed_amplitude_from_atoms, stem4d_sharded.
    """
    amplitudes: Complex[Array, "H W M"] = cbed_amplitude_from_atoms(
        beam=beam,
        atom_coords=atom_coords,
        atom_types=atom_types,
        slice_z_bounds=slice_z_bounds,
        atom_potentials=atom_potentials,
        voltage_kv=voltage_kv,
        calib_ang=calib_ang,
        atom_mask=atom_mask,
    )

    def _mode_amplitude(
        sample: Float[Array, " D"],
    ) -> Complex[Array, "H W"]:
        mode_idx: Int[Array, ""] = sample[0].astype(jnp.int32)
        amplitude: Complex[Array, "H W"] = amplitudes[..., mode_idx]
        return amplitude

    cbed_pattern: Float[Array, "H W"] = apply_distribution(
        mode_distribution,
        _mode_amplitude,
    )
    return cbed_pattern


@jaxtyped(typechecker=beartype)
def stem4d_sharded(
    probe_modes: ProbeModes,
    sample: AtomicSliceData,
    microscope: MicroscopeConfig,
    detector: DetectorConfig,
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
       :func:`cbed_image_from_atoms`.
    3. **Distributed execution** --
       If *mesh* is provided, use ``jax.shard_map`` to
       distribute positions across devices; otherwise use
       ``jax.vmap``.
    4. **Build output** --
       Return :class:`~ptyrodactyl.types.STEM4D` PyTree with
       data, calibrations, and scan positions.

    Parameters
    ----------
    probe_modes : ProbeModes
        Base electron probe modes (unshifted).
    sample : AtomicSliceData
        Atom coordinates, type indices, z bounds, and potential kernels.
    microscope : MicroscopeConfig
        Microscope voltage and ensemble configuration.
    detector : DetectorConfig
        Scan positions in Angstroms and real-space calibration.
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

    :see: cbed_image_from_atoms, checked_stem4d_sharded.
    """
    if detector.scan_positions_ang is None:
        raise ValueError("detector.scan_positions_ang is required")

    scan_positions_ang: Float[Array, "P 2"] = detector.scan_positions_ang
    voltage_kv: Float[Array, ""] = microscope.voltage_kv
    calib_ang: Float[Array, ""] = detector.real_space_calib_ang
    h: int = probe_modes.modes.shape[0]
    w: int = probe_modes.modes.shape[1]

    probe_k: Complex[Array, "H W M"] = jnp.fft.fft2(
        probe_modes.modes,
        axes=(0, 1),
    )
    mode_distribution: Distribution = probe_modes_to_distribution(
        probe_modes,
    )
    qy: Float[Array, " H"] = jnp.fft.fftfreq(h, d=calib_ang)
    qx: Float[Array, " W"] = jnp.fft.fftfreq(w, d=calib_ang)
    qya: Float[Array, "H W"]
    qxa: Float[Array, "H W"]
    qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")

    def _shift_probe(
        position_ang: Float[Array, " 2"],
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
        y_shift: scalar_float = position_ang[0]
        x_shift: scalar_float = position_ang[1]
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
        position_ang: Float[Array, " 2"],
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

        cbed_pattern: Float[Array, "H W"] = cbed_image_from_atoms(
            beam=current_beam,
            mode_distribution=mode_distribution,
            atom_coords=sample.atom_coords,
            atom_types=sample.atom_types,
            slice_z_bounds=sample.slice_z_bounds,
            atom_potentials=sample.atom_potentials,
            voltage_kv=voltage_kv,
            calib_ang=calib_ang,
            atom_mask=sample.atom_mask,
        )
        return cbed_pattern

    def _process_batch(
        positions_batch: Float[Array, "B 2"],
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

    stem4d_data_sharded: STEM4D = create_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kv=microscope.voltage_kv,
    )

    return stem4d_data_sharded


__all__: list[str] = [
    "cbed_amplitude_from_atoms",
    "cbed_image_from_atoms",
    "stem4d_sharded",
]
