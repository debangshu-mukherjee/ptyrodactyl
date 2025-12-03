"""High-level workflows for electron microscopy simulations.

Extended Summary
----------------
This module provides complete workflows that combine multiple simulation
steps into convenient functions for common use cases.

Routine Listings
----------------
xyz_to_4d_stem : function
    Simulates 4D-STEM data from an XYZ structure file.

Notes
-----
Workflows are designed as convenience functions that chain together
lower-level simulation functions from the simulations and atom_potentials
modules.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from ptyrodactyl.tools import (
    STEM4D,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarNumeric,
    XYZData,
    make_probe_modes,
)

from ptyrodactyl.simul import (
    kirkland_potentials_xyz, 
    make_probe, 
    parse_xyz, 
    stem_4d,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def xyz_to_4d_stem(
    xyz_filepath: str,
    slice_thickness: ScalarFloat,
    lateral_extent: ScalarFloat,
    cbed_aperture_mrad: ScalarNumeric,
    voltage_kv: ScalarNumeric,
    scan_positions: Float[Array, "P 2"],
    cbed_pixel_size_ang: ScalarFloat,
    probe_defocus: Optional[ScalarNumeric] = 0.0,
    probe_c3: Optional[ScalarNumeric] = 0.0,
    probe_c5: Optional[ScalarNumeric] = 0.0,
) -> STEM4D:
    """Complete workflow to simulate 4D-STEM data from an XYZ structure file.

    Parameters
    ----------
    xyz_filepath : str
        Path to the XYZ file containing atomic structure.
    slice_thickness : ScalarFloat
        Thickness of each slice in Angstroms for multislice calculation.
    lateral_extent : ScalarFloat
        Minimum lateral extent in Angstroms for periodic boundaries.
        The structure will be repeated to ensure at least this extent.
    cbed_aperture_mrad : ScalarNumeric
        Probe aperture size in milliradians.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    scan_positions : Float[Array, "P 2"]
        Array of (y, x) scan positions in Angstroms where P is number of positions.
    cbed_pixel_size_ang : ScalarFloat
        Real space pixel size in Angstroms for the calculation.
    probe_defocus : ScalarNumeric, optional
        Probe defocus in Angstroms. Default is 0.0.
    probe_c3 : ScalarNumeric, optional
        Third-order spherical aberration in Angstroms. Default is 0.0.
    probe_c5 : ScalarNumeric, optional
        Fifth-order spherical aberration in Angstroms. Default is 0.0.

    Returns
    -------
    STEM4D
        Complete 4D-STEM dataset containing:
        - Diffraction patterns for each scan position
        - Real and Fourier space calibrations
        - Scan positions in Angstroms
        - Accelerating voltage

    Notes
    -----
    This function loads the structure, calculates appropriate repeats based
    on thickness and lateral extents, generates Kirkland potentials, creates
    a probe, and simulates CBED patterns at multiple scan positions.

    Algorithm:
    - Load XYZ structure from file
    - Calculate repeats needed:
        - Z repeats based on total thickness / lattice c parameter
        - XY repeats based on lateral_extent / lattice a,b parameters
    - Generate Kirkland potentials with calculated repeats
    - Create probe with specified aberrations
    - Generate scan positions grid
    - Run 4D-STEM simulation
    - Return calibrated 4D data
    """
    xyz_data: XYZData = parse_xyz(xyz_filepath)
    if xyz_data.lattice is not None:
        a_length: Float[Array, " "] = jnp.linalg.norm(xyz_data.lattice[0])
        b_length: Float[Array, " "] = jnp.linalg.norm(xyz_data.lattice[1])
        c_length: Float[Array, " "] = jnp.linalg.norm(xyz_data.lattice[2])
        repeat_x: Int[Array, " "] = jnp.ceil(lateral_extent / a_length).astype(
            jnp.int32
        )
        repeat_y: Int[Array, " "] = jnp.ceil(lateral_extent / b_length).astype(
            jnp.int32
        )
        z_coords: Float[Array, " N"] = xyz_data.positions[:, 2]
        z_min: Float[Array, " "] = jnp.min(z_coords)
        z_max: Float[Array, " "] = jnp.max(z_coords)
        structure_thickness: Float[Array, " "] = z_max - z_min
        total_thickness_needed: Float[Array, " "] = (
            structure_thickness + slice_thickness
        )
        repeat_z: Int[Array, " "] = jnp.ceil(
            total_thickness_needed / c_length
        ).astype(jnp.int32)
        repeats: Int[Array, " 3"] = jnp.array([repeat_x, repeat_y, repeat_z])
    else:
        repeats: Int[Array, " 3"] = jnp.array([1, 1, 1])

    potential_slices: PotentialSlices = kirkland_potentials_xyz(
        xyz_data=xyz_data,
        pixel_size=cbed_pixel_size_ang,
        slice_thickness=slice_thickness,
        repeats=repeats,
        padding=4.0,
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
    probe_modes: ProbeModes = make_probe_modes(
        modes=probe[..., jnp.newaxis],
        weights=jnp.array([1.0]),
        calib=cbed_pixel_size_ang,
    )
    scan_positions_pixels: Float[Array, "P 2"] = (
        scan_positions / cbed_pixel_size_ang
    )
    stem4d_data: STEM4D = stem_4d(
        pot_slice=potential_slices,
        beam=probe_modes,
        positions=scan_positions_pixels,
        voltage_kV=voltage_kv,
        calib_ang=cbed_pixel_size_ang,
    )
    return stem4d_data
