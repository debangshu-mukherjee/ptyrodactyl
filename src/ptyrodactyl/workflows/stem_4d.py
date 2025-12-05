"""High-level workflows for electron microscopy simulations.

Extended Summary
----------------
This module provides complete workflows that combine multiple simulation
steps into convenient functions for common use cases.

Routine Listings
----------------
_estimate_memory_gb : function, internal
    Estimate memory requirements for 4D-STEM simulation in GB.
_get_device_memory_gb : function, internal
    Get available memory on the first JAX device in GB.
crystal2stem4d : function
    4D-STEM simulation from CrystalData with automatic sharding.

Notes
-----
Workflows are designed as convenience functions that chain together
lower-level simulation functions from the simulations and atom_potentials
modules.
"""

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax.sharding import Mesh
from jaxtyping import Array, Complex, Float, Int

from ptyrodactyl.simul import (
    clip_cbed,
    make_probe,
    single_atom_potential,
    stem4d_sharded,
)
from ptyrodactyl.tools import (
    STEM4D,
    CrystalData,
    ScalarFloat,
    ScalarNumeric,
    make_stem4d,
)

jax.config.update("jax_enable_x64", True)

_LARGE_POSITION_THRESHOLD: int = 100


def _estimate_memory_gb(
    num_positions: int,
    height: int,
    width: int,
    num_modes: int,
    num_slices: int,
) -> float:
    """Estimate memory requirements for 4D-STEM simulation in GB.

    Computes estimated memory usage based on array sizes:
    - Beams: P x H x W x M complex128 (16 bytes per element)
    - CBED patterns: P x H x W float64 (8 bytes per element)
    - Potential slices: H x W x S float64 (8 bytes per element)

    Applies a 2.5x overhead factor for FFT working memory and intermediates.

    Parameters
    ----------
    num_positions : int
        Number of scan positions.
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    num_modes : int
        Number of probe modes.
    num_slices : int
        Number of potential slices.

    Returns
    -------
    memory_gb : float
        Estimated memory requirement in gigabytes.
    """
    bytes_per_complex128: int = 16
    bytes_per_float64: int = 8
    beams_bytes: int = (
        num_positions * height * width * num_modes * bytes_per_complex128
    )
    cbed_bytes: int = num_positions * height * width * bytes_per_float64
    slices_bytes: int = height * width * num_slices * bytes_per_float64
    overhead_factor: float = 2.5
    total_bytes: float = (
        beams_bytes + cbed_bytes + slices_bytes
    ) * overhead_factor
    memory_gb: float = total_bytes / (1024**3)
    return memory_gb


def _get_device_memory_gb() -> float:
    """Get available memory on the first JAX device in GB.

    Attempts to query memory_stats from the first JAX device. This works
    for GPU/TPU devices that expose memory information via the bytes_limit
    key. Falls back to 16.0 GB for CPU or devices where memory stats are
    unavailable.

    Returns
    -------
    memory_gb : float
        Available device memory in gigabytes.
        Returns 16.0 as default if unable to determine.
    """
    try:
        devices = jax.devices()
        if len(devices) > 0:
            device = devices[0]
            if hasattr(device, "memory_stats"):
                stats = device.memory_stats()
                if stats and "bytes_limit" in stats:
                    return stats["bytes_limit"] / (1024**3)
        return 16.0
    except Exception:  # noqa: BLE001
        return 16.0


@beartype
def crystal2stem4d(  # noqa: PLR0913, PLR0915
    crystal_data: CrystalData,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
    cbed_aperture_mrad: ScalarNumeric,
    cbed_extent_mrad: ScalarFloat = 50.0,
    cbed_shape: Tuple[int, int] = (256, 256),
    real_space_pixel_size_ang: ScalarFloat = 0.02,
    slice_thickness: ScalarFloat = 1.0,
    num_modes: int = 1,
    probe_defocus: ScalarNumeric = 0.0,
    probe_c3: ScalarNumeric = 0.0,
    probe_c5: ScalarNumeric = 0.0,
    padding: float = 4.0,
    supersampling: int = 4,
    force_parallel: Optional[bool] = None,
) -> STEM4D:
    """4D-STEM simulation from crystal data with automatic sharding.

    Takes a CrystalData PyTree, generates electron probe, computes atomic
    potentials on-the-fly, and runs the 4D-STEM simulation. Automatically
    shards data across devices when beneficial. Output CBED patterns are
    clipped to the specified mrad extent and resized to the target shape.

    Parameters
    ----------
    crystal_data : CrystalData
        Crystal structure data containing atomic positions and numbers.
    scan_positions : Float[Array, "P 2"]
        Array of (y, x) scan positions in Angstroms.
        P is the number of scan positions.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    cbed_aperture_mrad : ScalarNumeric
        Probe aperture size in milliradians.
    cbed_extent_mrad : ScalarFloat, optional
        Half-angle extent of output CBED in milliradians. Default is 50.0.
    cbed_shape : Tuple[int, int], optional
        Output CBED shape (height, width). Default is (256, 256).
    real_space_pixel_size_ang : ScalarFloat, optional
        Real space pixel size in Angstroms for simulation. Default is 0.02
        (2 pm), which provides fine sampling for accurate multislice.
    slice_thickness : ScalarFloat, optional
        Thickness of each slice in Angstroms. Default is 1.0.
    num_modes : int, optional
        Number of probe modes for partial coherence. Default is 1.
    probe_defocus : ScalarNumeric, optional
        Probe defocus in Angstroms. Default is 0.0.
    probe_c3 : ScalarNumeric, optional
        Third-order spherical aberration in Angstroms. Default is 0.0.
    probe_c5 : ScalarNumeric, optional
        Fifth-order spherical aberration in Angstroms. Default is 0.0.
    padding : float, optional
        Padding in Angstroms for potential calculation. Default is 4.0.
    supersampling : int, optional
        Supersampling factor for atomic potentials. Default is 4.
    force_parallel : bool, optional
        If True, force sharding across devices. If False, no sharding.
        If None (default), automatically select based on resources.

    Returns
    -------
    stem4d_result : STEM4D
        Complete 4D-STEM dataset containing:
        - data : Float[Array, "P Ho Wo"]
            Clipped and resized diffraction patterns
        - real_space_calib : Float[Array, " "]
            Real space calibration in angstroms per pixel
        - fourier_space_calib : Float[Array, " "]
            Fourier space calibration in inverse angstroms per pixel
        - scan_positions : Float[Array, "P 2"]
            Scan positions in angstroms
        - voltage_kv : Float[Array, " "]
            Accelerating voltage in kilovolts

    Notes
    -----
    The simulation grid is determined by the sample FOV and
    real_space_pixel_size_ang. Output CBEDs are clipped to cbed_extent_mrad
    and resized to cbed_shape.

    Selection criteria for sharding (when force_parallel is None):
    1. Multiple devices available (GPU/TPU), OR
    2. Estimated memory exceeds 50% of single device memory, OR
    3. Large number of scan positions (>100)

    Algorithm:
    1. Calculate grid dimensions (H, W, S) directly from coordinate ranges
    2. Generate electron probe with specified aberrations
    3. Pre-shift beams to all scan positions using shift_beam_fourier
    4. Compute slice z-boundaries from z-coordinate range and thickness
    5. Precompute 2D atomic potentials for each unique atom type
    6. Optionally shard data across devices based on use_parallel flag
    7. Run stem4d_sharded to get raw CBED patterns (on-the-fly slice gen)
    8. Clip CBEDs to cbed_extent_mrad and resize to cbed_shape

    See Also
    --------
    stem4d_sharded : Low-level JAX-safe 4D-STEM function with on-the-fly slices
    single_atom_potential : Computes 2D atomic potentials for each type.
    shift_beam_fourier : Pre-shifts beams to scan positions.
    make_probe : Creates electron probe with aberrations.
    """
    x_coords: Float[Array, " N"] = crystal_data.positions[:, 0]
    y_coords: Float[Array, " N"] = crystal_data.positions[:, 1]
    z_coords: Float[Array, " N"] = crystal_data.positions[:, 2]

    x_range: float = float(jnp.max(x_coords) - jnp.min(x_coords)) + 2 * padding
    y_range: float = float(jnp.max(y_coords) - jnp.min(y_coords)) + 2 * padding
    z_range: float = float(jnp.max(z_coords) - jnp.min(z_coords))

    width: int = int(np.ceil(x_range / real_space_pixel_size_ang))
    height: int = int(np.ceil(y_range / real_space_pixel_size_ang))
    num_slices: int = max(1, int(np.ceil(z_range / slice_thickness)))

    devices = jax.devices()
    num_devices: int = len(devices)
    num_positions: int = scan_positions.shape[0]

    if force_parallel is not None:
        use_parallel: bool = force_parallel
    else:
        device_memory_gb: float = _get_device_memory_gb()
        est_memory_gb: float = _estimate_memory_gb(
            num_positions=num_positions,
            height=height,
            width=width,
            num_modes=num_modes,
            num_slices=num_slices,
        )
        memory_threshold: float = device_memory_gb * 0.5
        large_positions: bool = num_positions > _LARGE_POSITION_THRESHOLD

        use_parallel = (
            (num_devices > 1)
            or (est_memory_gb > memory_threshold)
            or large_positions
        )

    image_size: Int[Array, " 2"] = jnp.array([height, width])
    probe: Complex[Array, "H W"] = make_probe(
        aperture=cbed_aperture_mrad,
        voltage=voltage_kv,
        image_size=image_size,
        calibration_pm=real_space_pixel_size_ang * 100.0,
        defocus=probe_defocus,
        c3=probe_c3,
        c5=probe_c5,
    )

    if num_modes > 1:
        modes: Complex[Array, "H W M"] = jnp.stack(
            [probe] * num_modes, axis=-1
        )
    else:
        modes = probe[..., jnp.newaxis]

    z_min: float = float(jnp.min(z_coords))
    slice_boundaries: list[list[float]] = []
    for i in range(num_slices):
        z_start: float = z_min + i * float(slice_thickness)
        z_end: float = z_start + float(slice_thickness)
        slice_boundaries.append([z_start, z_end])

    slice_z_bounds: Float[Array, "S 2"] = jnp.array(
        slice_boundaries, dtype=jnp.float64
    )

    unique_atoms: list[int] = sorted(
        {int(x) for x in crystal_data.atomic_numbers}
    )
    atom_type_map: dict[int, int] = {
        atom_num: idx for idx, atom_num in enumerate(unique_atoms)
    }

    atom_potentials_list: list[Float[Array, "H W"]] = []
    for atom_num in unique_atoms:
        pot: Float[Array, "H W"] = single_atom_potential(
            atom_no=atom_num,
            pixel_size=real_space_pixel_size_ang,
            grid_shape=(height, width),
            center_coords=jnp.array([0.0, 0.0]),
            supersampling=supersampling,
        )
        atom_potentials_list.append(pot)
    atom_potentials: Float[Array, "T H W"] = jnp.stack(
        atom_potentials_list, axis=0
    )

    atom_types: Int[Array, " N"] = jnp.array(
        [atom_type_map[int(x)] for x in crystal_data.atomic_numbers],
        dtype=jnp.int32,
    )
    atom_coords: Float[Array, "N 3"] = crystal_data.positions

    if use_parallel:
        mesh = Mesh(np.array(devices), axis_names=("p",))
        raw_stem4d: STEM4D = stem4d_sharded(
            modes,
            scan_positions,
            atom_coords,
            atom_types,
            slice_z_bounds,
            atom_potentials,
            voltage_kv,
            real_space_pixel_size_ang,
            mesh=mesh,
        )
    else:
        raw_stem4d = stem4d_sharded(
            modes,
            scan_positions,
            atom_coords,
            atom_types,
            slice_z_bounds,
            atom_potentials,
            voltage_kv,
            real_space_pixel_size_ang,
        )
    fourier_calib_inv_ang: Float[Array, " "] = raw_stem4d.fourier_space_calib

    def _clip_single_cbed(cbed: Float[Array, "H W"]) -> Float[Array, "Ho Wo"]:
        """Clip and resize a single CBED pattern."""
        return clip_cbed(
            cbed=cbed,
            fourier_calib_inv_ang=fourier_calib_inv_ang,
            voltage_kv=voltage_kv,
            extent_mrad=cbed_extent_mrad,
            output_shape=cbed_shape,
        )

    clipped_cbeds: Float[Array, "P Ho Wo"] = jax.vmap(_clip_single_cbed)(
        raw_stem4d.data
    )
    wavelength_ang: float = 12.2643 / np.sqrt(
        float(voltage_kv) * (1.0 + 0.978459e-3 * float(voltage_kv))
    )
    mrad_per_inv_ang: float = wavelength_ang * 1000.0
    output_fourier_calib_mrad: float = (
        2.0 * float(cbed_extent_mrad) / cbed_shape[0]
    )
    output_fourier_calib_inv_ang: float = (
        output_fourier_calib_mrad / mrad_per_inv_ang
    )
    stem4d_result: STEM4D = make_stem4d(
        data=clipped_cbeds,
        real_space_calib=real_space_pixel_size_ang,
        fourier_space_calib=output_fourier_calib_inv_ang,
        scan_positions=raw_stem4d.scan_positions,
        voltage_kv=voltage_kv,
    )
    return stem4d_result
