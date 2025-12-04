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
    Smart dispatcher for 4D-STEM simulation, auto-selects implementation.
crystal2stem4d_parallel : function
    Parallel sharded 4D-STEM simulation for large-scale computations.
crystal2stem4d_single : function
    Single-device 4D-STEM simulation from crystal structure file.

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
from beartype.typing import Optional
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from ptyrodactyl.simul import (
    kirkland_potentials_crystal,
    make_probe,
    parse_crystal,
    shift_beam_fourier,
    single_atom_potential,
    stem4d_sharded,
    stem_4d,
)
from ptyrodactyl.tools import (
    STEM4D,
    CrystalData,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarNumeric,
    make_probe_modes,
)

jax.config.update("jax_enable_x64", True)

_LARGE_POSITION_THRESHOLD: int = 100


@jaxtyped(typechecker=beartype)
def crystal2stem4d_single(  # noqa: PLR0913
    crystal_filepath: str,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
    cbed_pixel_size_ang: ScalarFloat,
    cbed_aperture_mrad: ScalarNumeric,
    slice_thickness: ScalarFloat = 1.0,
    num_modes: int = 1,
    probe_defocus: Optional[ScalarNumeric] = 0.0,
    probe_c3: Optional[ScalarNumeric] = 0.0,
    probe_c5: Optional[ScalarNumeric] = 0.0,
    padding: float = 4.0,
    supersampling: int = 4,
) -> STEM4D:
    """Single-device 4D-STEM simulation from a crystal structure file.

    Loads a crystal structure file (XYZ or POSCAR), generates Kirkland
    potential slices, creates an electron probe with specified aberrations,
    and runs the 4D-STEM simulation on a single device.

    Parameters
    ----------
    crystal_filepath : str
        Path to the crystal structure file (.xyz, POSCAR, or CONTCAR).
    scan_positions : Float[Array, "P 2"]
        Array of (y, x) scan positions in Angstroms.
        P is the number of scan positions.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    cbed_pixel_size_ang : ScalarFloat
        Real space pixel size in Angstroms for the calculation.
    cbed_aperture_mrad : ScalarNumeric
        Probe aperture size in milliradians.
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

    Returns
    -------
    stem4d_data : STEM4D
        Complete 4D-STEM dataset containing:
        - data : Float[Array, "P H W"]
            Diffraction patterns for each scan position
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
    This function runs on a single device without sharding. Use
    crystal2stem4d_parallel for multi-device execution or crystal2stem4d for
    automatic selection.

    Algorithm:
    1. Load crystal structure from file using parse_crystal
    2. Generate Kirkland potential slices from atomic coordinates
    3. Extract grid dimensions (H, W) from potential slice shape
    4. Create electron probe with specified aberrations
    5. Create multimodal probe with equal weights if num_modes > 1,
       otherwise wrap single probe as 1-mode array
    6. Convert scan positions from Angstroms to pixels
    7. AOT compile stem_4d with concrete input shapes
    8. Run compiled stem_4d and return calibrated STEM4D result

    See Also
    --------
    crystal2stem4d : Smart dispatcher that auto-selects implementation.
    crystal2stem4d_parallel : Parallel sharded implementation for multi-device.
    stem_4d : Low-level 4D-STEM simulation function.
    kirkland_potentials_crystal : Generates potential slices from crystal data.
    make_probe : Creates electron probe with aberrations.
    """
    crystal_data: CrystalData = parse_crystal(crystal_filepath)
    potential_slices: PotentialSlices = kirkland_potentials_crystal(
        crystal_data=crystal_data,
        pixel_size=cbed_pixel_size_ang,
        slice_thickness=slice_thickness,
        padding=padding,
        supersampling=supersampling,
    )
    image_height: Int[Array, ""] = jnp.asarray(
        potential_slices.slices.shape[0], dtype=jnp.int32
    )
    image_width: Int[Array, ""] = jnp.asarray(
        potential_slices.slices.shape[1], dtype=jnp.int32
    )
    image_size: Int[Array, " 2"] = jnp.array([image_height, image_width])
    probe: Complex[Array, "H W"] = make_probe(
        aperture=cbed_aperture_mrad,
        voltage=voltage_kv,
        image_size=image_size,
        calibration_pm=cbed_pixel_size_ang * 100.0,
        defocus=probe_defocus,
        c3=probe_c3,
        c5=probe_c5,
    )

    if num_modes > 1:
        modes: Complex[Array, "H W M"] = jnp.stack(
            [probe] * num_modes, axis=-1
        )
        weights: Float[Array, " M"] = jnp.ones(num_modes) / num_modes
    else:
        modes = probe[..., jnp.newaxis]
        weights = jnp.array([1.0])
    probe_modes: ProbeModes = make_probe_modes(
        modes=modes,
        weights=weights,
        calib=cbed_pixel_size_ang,
    )
    scan_positions_pixels: Float[Array, "P 2"] = (
        scan_positions / cbed_pixel_size_ang
    )
    stem_4d_compiled = stem_4d.lower(
        potential_slices,
        probe_modes,
        scan_positions_pixels,
        voltage_kv,
        cbed_pixel_size_ang,
    ).compile()
    stem4d_data: STEM4D = stem_4d_compiled(
        potential_slices,
        probe_modes,
        scan_positions_pixels,
        voltage_kv,
        cbed_pixel_size_ang,
    )
    return stem4d_data


@beartype
def crystal2stem4d_parallel(  # noqa: PLR0913
    crystal_filepath: str,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
    cbed_pixel_size_ang: ScalarFloat,
    cbed_aperture_mrad: ScalarNumeric,
    slice_thickness: ScalarFloat = 1.0,
    num_modes: int = 1,
    probe_defocus: ScalarNumeric = 0.0,
    probe_c3: ScalarNumeric = 0.0,
    probe_c5: ScalarNumeric = 0.0,
    padding: float = 4.0,
    supersampling: int = 4,
) -> STEM4D:
    """Parallel sharded 4D-STEM simulation for large-scale computations.

    This workflow loads a crystal structure file (XYZ or POSCAR), generates
    beams, computes atomic potentials, sets up sharding across available
    devices, and runs the sharded 4D-STEM simulation.

    Parameters
    ----------
    crystal_filepath : str
        Path to the crystal structure file (.xyz, POSCAR, or CONTCAR).
    scan_positions : Float[Array, "P 2"]
        Array of (y, x) scan positions in Angstroms.
        P is the number of scan positions.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    cbed_pixel_size_ang : ScalarFloat
        Real space pixel size in Angstroms for the calculation.
    cbed_aperture_mrad : ScalarNumeric
        Probe aperture size in milliradians.
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

    Returns
    -------
    stem4d_result : STEM4D
        Complete 4D-STEM dataset with sharded data containing:
        - data : Float[Array, "P H W"]
            Diffraction patterns for each scan position
            (sharded along P axis)
        - real_space_calib : Float[Array, " "]
            Real space calibration in angstroms per pixel
        - fourier_space_calib : Float[Array, " "]
            Fourier space calibration in inverse angstroms per pixel
        - scan_positions : Float[Array, "P 2"]
            Scan positions in angstroms (sharded along P axis)
        - voltage_kv : Float[Array, " "]
            Accelerating voltage in kilovolts

    Notes
    -----
    This function shards beams and positions along the P axis for parallel
    execution across multiple devices. Use crystal2stem4d_single for
    single-device execution or crystal2stem4d for automatic selection.

    Algorithm:
    1. Load crystal data from file using parse_crystal
    2. Calculate grid dimensions (H, W, S) directly from coordinate ranges
    3. Generate electron probe with specified aberrations
    4. Create multimodal probe if num_modes > 1, otherwise single mode
    5. Convert scan positions from Angstroms to pixels
    6. Pre-shift beams to all scan positions using shift_beam_fourier
    7. Compute slice z-boundaries from z-coordinate range and thickness
    8. Extract unique atom types and build 0-indexed type mapping
    9. Precompute 2D atomic potentials for each unique atom type
    10. Map atomic numbers to type indices for stem4d_sharded
    11. Create JAX mesh for sharding along position axis ("p")
    12. Define sharding specs: beams (P,None,None,None), positions (P,None),
        atom_coords/slice_bounds (None,None), atom_types (None),
        atom_potentials (None,None,None) for replication
    13. Distribute arrays to devices using jax.device_put
    14. AOT compile stem4d_sharded with concrete sharded input shapes
    15. Run compiled stem4d_sharded and return sharded STEM4D result

    See Also
    --------
    crystal2stem4d : Smart dispatcher that auto-selects implementation.
    crystal2stem4d_single : Single-device implementation without sharding.
    stem4d_sharded : Low-level JAX-safe sharded 4D-STEM function.
    single_atom_potential : Computes 2D atomic potentials for each type.
    shift_beam_fourier : Pre-shifts beams to scan positions.
    make_probe : Creates electron probe with aberrations.
    """
    crystal_data: CrystalData = parse_crystal(crystal_filepath)
    x_coords: Float[Array, " N"] = crystal_data.positions[:, 0]
    y_coords: Float[Array, " N"] = crystal_data.positions[:, 1]
    z_coords: Float[Array, " N"] = crystal_data.positions[:, 2]
    x_range: float = float(jnp.max(x_coords) - jnp.min(x_coords)) + 2 * padding
    y_range: float = float(jnp.max(y_coords) - jnp.min(y_coords)) + 2 * padding
    z_range: float = float(jnp.max(z_coords) - jnp.min(z_coords))
    width: int = int(np.ceil(x_range / cbed_pixel_size_ang))
    height: int = int(np.ceil(y_range / cbed_pixel_size_ang))
    num_slices: int = max(1, int(np.ceil(z_range / slice_thickness)))
    image_size: Int[Array, " 2"] = jnp.array([height, width])
    probe: Complex[Array, "H W"] = make_probe(
        aperture=cbed_aperture_mrad,
        voltage=voltage_kv,
        image_size=image_size,
        calibration_pm=cbed_pixel_size_ang * 100.0,
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

    scan_positions_pixels: Float[Array, "P 2"] = (
        scan_positions / cbed_pixel_size_ang
    )
    shifted_beams: Complex[Array, "P H W M"] = shift_beam_fourier(
        beam=modes,
        pos=scan_positions_pixels,
        calib_ang=cbed_pixel_size_ang,
    )
    z_coords: Float[Array, " N"] = crystal_data.positions[:, 2]
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
            pixel_size=cbed_pixel_size_ang,
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
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("p",))
    beam_sharding = NamedSharding(mesh, PartitionSpec("p", None, None, None))
    pos_sharding = NamedSharding(mesh, PartitionSpec("p", None))
    replicated_3d = NamedSharding(mesh, PartitionSpec(None, None, None))
    replicated_1d = NamedSharding(mesh, PartitionSpec(None))
    replicated_2d = NamedSharding(mesh, PartitionSpec(None, None))
    sharded_beams = jax.device_put(shifted_beams, beam_sharding)
    sharded_positions = jax.device_put(scan_positions_pixels, pos_sharding)
    replicated_atom_coords = jax.device_put(atom_coords, replicated_2d)
    replicated_atom_types = jax.device_put(atom_types, replicated_1d)
    replicated_slice_bounds = jax.device_put(slice_z_bounds, replicated_2d)
    replicated_potentials = jax.device_put(atom_potentials, replicated_3d)
    stem4d_sharded_compiled = stem4d_sharded.lower(
        sharded_beams,
        sharded_positions,
        replicated_atom_coords,
        replicated_atom_types,
        replicated_slice_bounds,
        replicated_potentials,
        voltage_kv,
        cbed_pixel_size_ang,
    ).compile()
    stem4d_result: STEM4D = stem4d_sharded_compiled(
        sharded_beams,
        sharded_positions,
        replicated_atom_coords,
        replicated_atom_types,
        replicated_slice_bounds,
        replicated_potentials,
        voltage_kv,
        cbed_pixel_size_ang,
    )
    return stem4d_result


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
def crystal2stem4d(  # noqa: PLR0913
    crystal_filepath: str,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
    cbed_pixel_size_ang: ScalarFloat,
    cbed_aperture_mrad: ScalarNumeric,
    slice_thickness: ScalarFloat = 1.0,
    num_modes: int = 1,
    probe_defocus: ScalarNumeric = 0.0,
    probe_c3: ScalarNumeric = 0.0,
    probe_c5: ScalarNumeric = 0.0,
    padding: float = 4.0,
    supersampling: int = 4,
    force_parallel: Optional[bool] = None,
) -> STEM4D:
    """Smart dispatcher for 4D-STEM simulation from crystal structure file.

    Automatically selects between single-device and parallel implementations
    based on available devices and estimated memory requirements.

    Parameters
    ----------
    crystal_filepath : str
        Path to the crystal structure file (.xyz, POSCAR, or CONTCAR).
    scan_positions : Float[Array, "P 2"]
        Array of (y, x) scan positions in Angstroms.
        P is the number of scan positions.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    cbed_pixel_size_ang : ScalarFloat
        Real space pixel size in Angstroms for the calculation.
    cbed_aperture_mrad : ScalarNumeric
        Probe aperture size in milliradians.
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
        If True, force parallel execution. If False, force single-device.
        If None (default), automatically select based on resources.

    Returns
    -------
    stem4d_data : STEM4D
        Complete 4D-STEM dataset containing:
        - data : Float[Array, "P H W"]
            Diffraction patterns for each scan position
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
    Selection criteria for parallel execution (when force_parallel is None):
    1. Multiple devices available (GPU/TPU), OR
    2. Estimated memory exceeds 50% of single device memory, OR
    3. Large number of scan positions (>100, defined by
       _LARGE_POSITION_THRESHOLD)

    Use force_parallel=True/False to override automatic selection.

    Algorithm:
    1. Query available JAX devices and device memory
    2. Load crystal structure to estimate grid dimensions
    3. Compute coordinate ranges with padding for x/y, raw for z
    4. Estimate grid size (width, height, slices) from ranges
    5. Estimate memory requirements using _estimate_memory_gb
    6. If force_parallel is set, use that choice directly
    7. Otherwise, use parallel if: multi-device OR high memory OR many
       positions exceeds _LARGE_POSITION_THRESHOLD
    8. Call crystal2stem4d_parallel or crystal2stem4d_single accordingly

    See Also
    --------
    crystal2stem4d_single : Single-device implementation without sharding.
    crystal2stem4d_parallel : Parallel sharded implementation for multi-device.
    stem_4d : Low-level single-device 4D-STEM simulation function.
    stem4d_sharded : Low-level JAX-safe sharded 4D-STEM function.
    """
    devices = jax.devices()
    num_devices: int = len(devices)
    device_memory_gb: float = _get_device_memory_gb()
    crystal_data: CrystalData = parse_crystal(crystal_filepath)
    x_coords: Float[Array, " N"] = crystal_data.positions[:, 0]
    y_coords: Float[Array, " N"] = crystal_data.positions[:, 1]
    z_coords: Float[Array, " N"] = crystal_data.positions[:, 2]
    x_range: float = float(jnp.max(x_coords) - jnp.min(x_coords)) + 2 * padding
    y_range: float = float(jnp.max(y_coords) - jnp.min(y_coords)) + 2 * padding
    z_range: float = float(jnp.max(z_coords) - jnp.min(z_coords))
    est_width: int = int(np.ceil(x_range / cbed_pixel_size_ang))
    est_height: int = int(np.ceil(y_range / cbed_pixel_size_ang))
    est_slices: int = int(np.ceil(z_range / slice_thickness))
    num_positions: int = scan_positions.shape[0]
    est_memory_gb: float = _estimate_memory_gb(
        num_positions=num_positions,
        height=est_height,
        width=est_width,
        num_modes=num_modes,
        num_slices=est_slices,
    )
    if force_parallel is not None:
        use_parallel: bool = force_parallel
    else:
        memory_threshold: float = device_memory_gb * 0.5
        large_positions: bool = num_positions > _LARGE_POSITION_THRESHOLD

        use_parallel = (
            (num_devices > 1)
            or (est_memory_gb > memory_threshold)
            or large_positions
        )
    if use_parallel:
        return crystal2stem4d_parallel(
            crystal_filepath=crystal_filepath,
            scan_positions=scan_positions,
            voltage_kv=voltage_kv,
            cbed_pixel_size_ang=cbed_pixel_size_ang,
            cbed_aperture_mrad=cbed_aperture_mrad,
            slice_thickness=slice_thickness,
            num_modes=num_modes,
            probe_defocus=probe_defocus,
            probe_c3=probe_c3,
            probe_c5=probe_c5,
            padding=padding,
            supersampling=supersampling,
        )
    return crystal2stem4d_single(
        crystal_filepath=crystal_filepath,
        scan_positions=scan_positions,
        voltage_kv=voltage_kv,
        cbed_pixel_size_ang=cbed_pixel_size_ang,
        cbed_aperture_mrad=cbed_aperture_mrad,
        slice_thickness=slice_thickness,
        num_modes=num_modes,
        probe_defocus=probe_defocus,
        probe_c3=probe_c3,
        probe_c5=probe_c5,
        padding=padding,
        supersampling=supersampling,
    )
