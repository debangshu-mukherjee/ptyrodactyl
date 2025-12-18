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
crystal2stem4d_tiled : function
    Tiled 4D-STEM simulation for large samples with fixed memory per tile.

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
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from ptyrodactyl.simul import (
    make_probe,
    single_atom_potential,
    stem4d_sharded,
    wavelength_ang,
)
from ptyrodactyl.simul.parallelized import _cbed_from_potential_slices
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


@jaxtyped(typechecker=beartype)
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

    potential_extent_ang: float = 10.0
    kernel_size: int = int(
        np.ceil(potential_extent_ang / real_space_pixel_size_ang)
    )
    if kernel_size % 2 == 0:
        kernel_size += 1

    atom_potentials_list: list[Float[Array, "H W"]] = []
    for atom_num in unique_atoms:
        small_pot: Float[Array, "K K"] = single_atom_potential(
            atom_no=atom_num,
            pixel_size=real_space_pixel_size_ang,
            grid_shape=(kernel_size, kernel_size),
            center_coords=jnp.array([0.0, 0.0]),
            supersampling=supersampling,
        )
        padded_pot: Float[Array, "H W"] = jnp.zeros(
            (height, width), dtype=jnp.float64
        )
        half_k: int = kernel_size // 2
        padded_pot = padded_pot.at[:half_k + 1, :half_k + 1].set(
            small_pot[half_k:, half_k:]
        )
        padded_pot = padded_pot.at[:half_k + 1, -half_k:].set(
            small_pot[half_k:, :half_k]
        )
        padded_pot = padded_pot.at[-half_k:, :half_k + 1].set(
            small_pot[:half_k, half_k:]
        )
        padded_pot = padded_pot.at[-half_k:, -half_k:].set(
            small_pot[:half_k, :half_k]
        )
        atom_potentials_list.append(padded_pot)
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
    fourier_calib_inv_ang: float = float(raw_stem4d.fourier_space_calib)
    wavelength_ang_clip: float = 12.2643 / np.sqrt(
        float(voltage_kv) * (1.0 + 0.978459e-3 * float(voltage_kv))
    )
    mrad_per_inv_ang_clip: float = wavelength_ang_clip * 1000.0
    extent_inv_ang: float = float(cbed_extent_mrad) / mrad_per_inv_ang_clip
    extent_pixels: int = int(np.ceil(extent_inv_ang / fourier_calib_inv_ang))

    raw_h: int = raw_stem4d.data.shape[1]
    raw_w: int = raw_stem4d.data.shape[2]
    center_y: int = raw_h // 2
    center_x: int = raw_w // 2
    y_start: int = max(0, center_y - extent_pixels)
    y_end: int = min(raw_h, center_y + extent_pixels)
    x_start: int = max(0, center_x - extent_pixels)
    x_end: int = min(raw_w, center_x + extent_pixels)
    clip_h: int = y_end - y_start
    clip_w: int = x_end - x_start

    def _clip_single_cbed(cbed: Float[Array, "H W"]) -> Float[Array, "Ho Wo"]:
        """Clip and resize a single CBED pattern."""
        clipped: Float[Array, "Hc Wc"] = lax.dynamic_slice(
            cbed, (y_start, x_start), (clip_h, clip_w)
        )
        return jax.image.resize(clipped, cbed_shape, method="linear")

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


_DEFAULT_TILE_SIZE_ANG: float = 40.0
_DEFAULT_GRID_PIXELS: int = 4096
_DEFAULT_PIXEL_SIZE_ANG: float = 0.02


@jaxtyped(typechecker=beartype)
def crystal2stem4d_tiled(  # noqa: PLR0913, PLR0915
    crystal_data: CrystalData,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
    cbed_aperture_mrad: ScalarNumeric,
    cbed_extent_mrad: ScalarFloat = 50.0,
    cbed_shape: Tuple[int, int] = (256, 256),
    tile_size_ang: float = _DEFAULT_TILE_SIZE_ANG,
    grid_pixels: int = _DEFAULT_GRID_PIXELS,
    pixel_size_ang: float = _DEFAULT_PIXEL_SIZE_ANG,
    fourier_pixels: Optional[int] = None,
    slice_thickness: ScalarFloat = 1.0,
    num_modes: int = 1,
    probe_defocus: ScalarNumeric = 0.0,
    probe_c3: ScalarNumeric = 0.0,
    probe_c5: ScalarNumeric = 0.0,
    supersampling: int = 4,
) -> STEM4D:
    """Tiled 4D-STEM simulation for arbitrarily large samples.

    Divides the sample into tiles and computes CBEDs for each tile
    independently. This approach has O(1) memory per tile regardless
    of total sample size, enabling simulation of very large fields of view.

    Parameters
    ----------
    crystal_data : CrystalData
        Crystal structure data containing atomic positions and numbers.
    scan_positions : Float[Array, "P 2"]
        Array of (y, x) scan positions in Angstroms.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    cbed_aperture_mrad : ScalarNumeric
        Probe aperture size in milliradians.
    cbed_extent_mrad : ScalarFloat, optional
        Half-angle extent of output CBED in milliradians. Default is 50.0.
    cbed_shape : Tuple[int, int], optional
        Output CBED shape (height, width). Default is (256, 256).
    tile_size_ang : float, optional
        Active scan region size per tile in Angstroms. Default is 40.0 (4nm).
        Beams scan within this central region of each tile.
    grid_pixels : int, optional
        Grid size in pixels (should be power of 2). Default is 4096.
        Total grid covers tile_size + 2*padding.
    pixel_size_ang : float, optional
        Pixel size in Angstroms. Default is 0.02 (2pm).
    fourier_pixels : int, optional
        FFT grid size for Fourier-space sampling. If None, automatically
        calculated from cbed_extent_mrad and cbed_shape to ensure proper
        sampling. Larger values give finer Fourier resolution. Must be
        >= grid_pixels.
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
    supersampling : int, optional
        Supersampling factor for atomic potentials. Default is 4.

    Returns
    -------
    stem4d_result : STEM4D
        Complete 4D-STEM dataset with uniform CBED patterns.

    Notes
    -----
    Tiling scheme:
    - Each tile has a fixed grid of grid_pixels x grid_pixels
    - The central tile_size_ang x tile_size_ang region is the active scan area
    - Surrounding padding accommodates beam spread from defocus
    - Atoms within the full grid region are extracted for each tile

    The padding is computed as:
        padding = (grid_pixels * pixel_size_ang - tile_size_ang) / 2

    For default values (4096 pixels, 0.02 Å, 40 Å tile):
        total_grid = 4096 * 0.02 = 81.92 Å
        padding = (81.92 - 40) / 2 = 20.96 Å per side

    This provides ~21 Å padding, sufficient for:
        - 50nm defocus with 30mrad aperture: spread = 1.5 Å
        - Atom potential extent: ~5 Å
        - Large safety margin for probe tails

    Algorithm:
    1. Compute grid parameters and padding
    2. Precompute atom potentials (small kernel, shared across tiles)
    3. Generate probe on fixed-size grid
    4. For each scan position:
       a. Determine which tile it belongs to
       b. Extract atoms within tile region
       c. Compute local beam shift within tile
       d. Run multislice for that position
    5. Clip and resize all CBEDs uniformly
    """
    grid_size_ang: float = grid_pixels * pixel_size_ang
    padding_ang: float = (grid_size_ang - tile_size_ang) / 2.0

    if padding_ang < 0:
        msg = (
            f"Tile size ({tile_size_ang} Å) exceeds grid size "
            f"({grid_size_ang} Å). Increase grid_pixels or decrease "
            f"tile_size_ang."
        )
        raise ValueError(msg)

    # Calculate fft_pixels for proper Fourier sampling
    fft_pixels: int
    if fourier_pixels is None:
        wavelength: float = float(wavelength_ang(voltage_kv))
        target_mrad_per_pixel: float = (
            2.0 * float(cbed_extent_mrad) / float(cbed_shape[0])
        )
        target_inv_ang_per_pixel: float = (
            target_mrad_per_pixel / (wavelength * 1000.0)
        )
        required_grid_size_ang: float = 1.0 / target_inv_ang_per_pixel
        computed_fourier_pixels: int = int(
            np.ceil(required_grid_size_ang / pixel_size_ang)
        )
        fft_pixels = 1
        while fft_pixels < computed_fourier_pixels:
            fft_pixels *= 2
        fft_pixels = max(fft_pixels, grid_pixels)
    else:
        if fourier_pixels < grid_pixels:
            msg = (
                f"fourier_pixels ({fourier_pixels}) must be >= grid_pixels "
                f"({grid_pixels})."
            )
            raise ValueError(msg)
        fft_pixels = fourier_pixels

    z_coords: Float[Array, " N"] = crystal_data.positions[:, 2]
    z_min: float = float(jnp.min(z_coords))
    z_max: float = float(jnp.max(z_coords))
    z_range: float = z_max - z_min
    num_slices: int = max(1, int(np.ceil(z_range / slice_thickness)))

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

    potential_extent_ang: float = 10.0
    kernel_size: int = int(np.ceil(potential_extent_ang / pixel_size_ang))
    if kernel_size % 2 == 0:
        kernel_size += 1

    atom_potentials_list: list[Float[Array, "H W"]] = []
    for atom_num in unique_atoms:
        small_pot: Float[Array, "K K"] = single_atom_potential(
            atom_no=atom_num,
            pixel_size=pixel_size_ang,
            grid_shape=(kernel_size, kernel_size),
            center_coords=jnp.array([0.0, 0.0]),
            supersampling=supersampling,
        )
        padded_pot: Float[Array, "H W"] = jnp.zeros(
            (fft_pixels, fft_pixels), dtype=jnp.float64
        )
        half_k: int = kernel_size // 2
        padded_pot = padded_pot.at[:half_k + 1, :half_k + 1].set(
            small_pot[half_k:, half_k:]
        )
        padded_pot = padded_pot.at[:half_k + 1, -half_k:].set(
            small_pot[half_k:, :half_k]
        )
        padded_pot = padded_pot.at[-half_k:, :half_k + 1].set(
            small_pot[:half_k, half_k:]
        )
        padded_pot = padded_pot.at[-half_k:, -half_k:].set(
            small_pot[:half_k, :half_k]
        )
        atom_potentials_list.append(padded_pot)
    atom_potentials: Float[Array, "T H W"] = jnp.stack(
        atom_potentials_list, axis=0
    )

    atom_types_full: Int[Array, " N"] = jnp.array(
        [atom_type_map[int(x)] for x in crystal_data.atomic_numbers],
        dtype=jnp.int32,
    )
    atom_coords_full: Float[Array, "N 3"] = crystal_data.positions

    fft_image_size: Int[Array, " 2"] = jnp.array([fft_pixels, fft_pixels])
    probe: Complex[Array, "H W"] = make_probe(
        aperture=cbed_aperture_mrad,
        voltage=voltage_kv,
        image_size=fft_image_size,
        calibration_pm=pixel_size_ang * 100.0,
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

    devices = jax.devices()
    num_devices: int = len(devices)
    mesh = Mesh(np.array(devices), axis_names=("p",))

    tile_center_offset: float = grid_size_ang / 2.0

    def _process_single_position(
        position_ang: Float[Array, " 2"],
    ) -> Float[Array, "H W"]:
        """Process a single scan position within its tile."""
        pos_y: ScalarFloat = position_ang[0]
        pos_x: ScalarFloat = position_ang[1]

        tile_min_y: Float[Array, " "] = pos_y - tile_center_offset
        tile_max_y: Float[Array, " "] = pos_y + tile_center_offset
        tile_min_x: Float[Array, " "] = pos_x - tile_center_offset
        tile_max_x: Float[Array, " "] = pos_x + tile_center_offset

        in_tile: Float[Array, " N"] = (
            (atom_coords_full[:, 0] >= tile_min_x)
            & (atom_coords_full[:, 0] < tile_max_x)
            & (atom_coords_full[:, 1] >= tile_min_y)
            & (atom_coords_full[:, 1] < tile_max_y)
        ).astype(jnp.float64)

        # Convert to local tile coordinates (don't multiply by mask -
        # the mask is passed separately to exclude out-of-tile atoms)
        local_x: Float[Array, " N"] = atom_coords_full[:, 0] - tile_min_x
        local_y: Float[Array, " N"] = atom_coords_full[:, 1] - tile_min_y
        local_z: Float[Array, " N"] = atom_coords_full[:, 2]

        local_coords: Float[Array, "N 3"] = jnp.stack(
            [local_x, local_y, local_z], axis=-1
        )

        beam_local_y: Float[Array, " "] = tile_center_offset
        beam_local_x: Float[Array, " "] = tile_center_offset
        beam_position: Float[Array, " 2"] = jnp.array(
            [beam_local_y, beam_local_x]
        )

        h: int = fft_pixels
        w: int = fft_pixels
        probe_k: Complex[Array, "H W M"] = jnp.fft.fft2(modes, axes=(0, 1))
        qy: Float[Array, " H"] = jnp.fft.fftfreq(h, d=pixel_size_ang)
        qx: Float[Array, " W"] = jnp.fft.fftfreq(w, d=pixel_size_ang)
        qya: Float[Array, "H W"]
        qxa: Float[Array, "H W"]
        qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")

        y_shift: ScalarFloat = beam_position[0]
        x_shift: ScalarFloat = beam_position[1]
        phase: Float[Array, "H W"] = (
            -2.0 * jnp.pi * ((qya * y_shift) + (qxa * x_shift))
        )
        phase_shift: Complex[Array, "H W"] = jnp.exp(1j * phase)
        shifted_k: Complex[Array, "H W M"] = probe_k * phase_shift[..., None]
        shifted_beam: Complex[Array, "H W M"] = jnp.fft.ifft2(
            shifted_k, axes=(0, 1)
        )

        cbed: Float[Array, "H W"] = _cbed_from_potential_slices(
            beam=shifted_beam,
            atom_coords=local_coords,
            atom_types=atom_types_full,
            slice_z_bounds=slice_z_bounds,
            atom_potentials=atom_potentials,
            voltage_kv=voltage_kv,
            calib_ang=pixel_size_ang,
            atom_mask=in_tile,
        )
        return cbed

    num_positions: int = scan_positions.shape[0]

    if num_devices > 1 and num_positions >= num_devices:
        in_sharding = NamedSharding(mesh, PartitionSpec("p", None))

        @jax.jit
        def _compute_all_cbeds(
            positions: Float[Array, "P 2"],
        ) -> Float[Array, "P H W"]:
            """Compute CBEDs for all positions with sharding."""
            return jax.vmap(_process_single_position)(positions)

        sharded_positions = jax.device_put(scan_positions, in_sharding)
        raw_cbeds: Float[Array, "P H W"] = _compute_all_cbeds(
            sharded_positions
        )
    else:
        raw_cbeds = jax.vmap(_process_single_position)(scan_positions)

    fft_grid_size_ang: float = fft_pixels * pixel_size_ang
    fourier_calib_inv_ang: float = 1.0 / fft_grid_size_ang
    e_lambda_ang: float = float(wavelength_ang(voltage_kv))
    mrad_per_inv_ang: float = e_lambda_ang * 1000.0
    extent_inv_ang: float = float(cbed_extent_mrad) / mrad_per_inv_ang
    extent_pixels: int = int(np.ceil(extent_inv_ang / fourier_calib_inv_ang))

    center: int = fft_pixels // 2
    y_start: int = max(0, center - extent_pixels)
    y_end: int = min(fft_pixels, center + extent_pixels)
    x_start: int = max(0, center - extent_pixels)
    x_end: int = min(fft_pixels, center + extent_pixels)
    clip_h: int = y_end - y_start
    clip_w: int = x_end - x_start

    def _clip_single_cbed(cbed: Float[Array, "H W"]) -> Float[Array, "Ho Wo"]:
        """Clip and resize a single CBED pattern."""
        clipped: Float[Array, "Hc Wc"] = lax.dynamic_slice(
            cbed, (y_start, x_start), (clip_h, clip_w)
        )
        return jax.image.resize(clipped, cbed_shape, method="linear")

    clipped_cbeds: Float[Array, "P Ho Wo"] = jax.vmap(_clip_single_cbed)(
        raw_cbeds
    )

    output_fourier_calib_mrad: float = (
        2.0 * float(cbed_extent_mrad) / cbed_shape[0]
    )
    output_fourier_calib_inv_ang: float = (
        output_fourier_calib_mrad / mrad_per_inv_ang
    )

    stem4d_result: STEM4D = make_stem4d(
        data=clipped_cbeds,
        real_space_calib=pixel_size_ang,
        fourier_space_calib=output_fourier_calib_inv_ang,
        scan_positions=scan_positions,
        voltage_kv=voltage_kv,
    )
    return stem4d_result
