"""
Module: electrons.workflows
---------------------------
High-level workflows for electron microscopy simulations.

This module provides complete workflows that combine multiple simulation
steps into convenient functions for common use cases.

Functions
---------
- `xyz_to_4d_stem`:
    Simulates 4D-STEM data from an XYZ structure file
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from .atom_potentials import kirkland_potentials_XYZ
from .electron_types import (STEM4D, PotentialSlices, ProbeModes, XYZData,
                             make_probe_modes, make_stem4d, scalar_float,
                             scalar_numeric)
from .preprocessing import parse_xyz
from .simulations import make_probe, stem_4D

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def xyz_to_4d_stem(
    xyz_filepath: str,
    slice_thickness: scalar_float,
    lateral_extent: scalar_float,
    cbed_aperture_mrad: scalar_numeric,
    voltage_kV: scalar_numeric,
    scan_positions: Float[Array, "P 2"],
    cbed_pixel_size_ang: scalar_float,
    probe_defocus: Optional[scalar_numeric] = 0.0,
    probe_c3: Optional[scalar_numeric] = 0.0,
    probe_c5: Optional[scalar_numeric] = 0.0,
) -> STEM4D:
    """
    Description
    -----------
    Complete workflow to simulate 4D-STEM data from an XYZ structure file.
    This function loads the structure, calculates appropriate repeats based
    on thickness and lateral extents, generates Kirkland potentials, creates
    a probe, and simulates CBED patterns at multiple scan positions.

    Parameters
    ----------
    - `xyz_filepath` (str):
        Path to the XYZ file containing atomic structure
    - `slice_thickness` (scalar_float):
        Thickness of each slice in Angstroms for multislice calculation
    - `lateral_extent` (scalar_float):
        Minimum lateral extent in Angstroms for periodic boundaries.
        The structure will be repeated to ensure at least this extent.
    - `cbed_aperture_mrad` (scalar_numeric):
        Probe aperture size in milliradians
    - `voltage_kV` (scalar_numeric):
        Accelerating voltage in kilovolts
    - `scan_positions` (Float[Array, "P 2"]):
        Array of (y, x) scan positions in Angstroms where P is number of positions
    - `cbed_pixel_size_ang` (scalar_float):
        Real space pixel size in Angstroms for the calculation
    - `probe_defocus` (Optional[scalar_numeric]):
        Probe defocus in Angstroms. Default is 0.0
    - `probe_c3` (Optional[scalar_numeric]):
        Third-order spherical aberration in Angstroms. Default is 0.0
    - `probe_c5` (Optional[scalar_numeric]):
        Fifth-order spherical aberration in Angstroms. Default is 0.0

    Returns
    -------
    - `stem4d_data` (STEM4D):
        Complete 4D-STEM dataset containing:
        - Diffraction patterns for each scan position
        - Real and Fourier space calibrations
        - Scan positions in Angstroms
        - Accelerating voltage

    Flow
    ----
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
    # Load XYZ structure
    xyz_data: XYZData = parse_xyz(xyz_filepath)

    # Calculate repeats based on thickness and lateral extent
    if xyz_data.lattice is not None:
        # Get lattice parameters
        a_length: Float[Array, ""] = jnp.linalg.norm(xyz_data.lattice[0])
        b_length: Float[Array, ""] = jnp.linalg.norm(xyz_data.lattice[1])
        c_length: Float[Array, ""] = jnp.linalg.norm(xyz_data.lattice[2])

        # Calculate repeats
        repeat_x: Int[Array, ""] = jnp.ceil(lateral_extent / a_length).astype(jnp.int32)
        repeat_y: Int[Array, ""] = jnp.ceil(lateral_extent / b_length).astype(jnp.int32)

        # For z-direction, we need to consider the total thickness needed
        # Get z-range of atoms
        z_coords: Float[Array, "N"] = xyz_data.positions[:, 2]
        z_min: Float[Array, ""] = jnp.min(z_coords)
        z_max: Float[Array, ""] = jnp.max(z_coords)
        structure_thickness: Float[Array, ""] = z_max - z_min

        # If we need more thickness than one unit cell provides
        total_thickness_needed: Float[Array, ""] = structure_thickness + slice_thickness
        repeat_z: Int[Array, ""] = jnp.ceil(total_thickness_needed / c_length).astype(
            jnp.int32
        )

        repeats: Int[Array, "3"] = jnp.array([repeat_x, repeat_y, repeat_z])
    else:
        # No lattice information, use default no repeats
        repeats: Int[Array, "3"] = jnp.array([1, 1, 1])

    # Generate Kirkland potentials
    potential_slices: PotentialSlices = kirkland_potentials_XYZ(
        xyz_data=xyz_data,
        pixel_size=cbed_pixel_size_ang,
        slice_thickness=slice_thickness,
        repeats=repeats,
        padding=4.0,  # Default padding
    )

    # Get image size from potential slices
    image_height: int = potential_slices.slices.shape[0]
    image_width: int = potential_slices.slices.shape[1]
    image_size: Int[Array, "2"] = jnp.array([image_height, image_width])

    # Create probe
    probe: Complex[Array, "H W"] = make_probe(
        aperture=cbed_aperture_mrad,
        voltage=voltage_kV,
        image_size=image_size,
        calibration_pm=cbed_pixel_size_ang * 100.0,  # Convert Angstroms to picometers
        defocus=probe_defocus,
        c3=probe_c3,
        c5=probe_c5,
    )
    probe_modes: ProbeModes = make_probe_modes(
        modes=probe[..., jnp.newaxis],
        weights=jnp.array([1.0]),  # Single mode with weight 1
        calib=cbed_pixel_size_ang,
    )
    scan_positions_pixels: Float[Array, "P 2"] = scan_positions / cbed_pixel_size_ang
    stem4d_data: STEM4D = stem_4D(
        pot_slice=potential_slices,
        beam=probe_modes,
        positions=scan_positions_pixels,
        voltage_kV=voltage_kV,
        calib_ang=cbed_pixel_size_ang,
    )

    return stem4d_data
