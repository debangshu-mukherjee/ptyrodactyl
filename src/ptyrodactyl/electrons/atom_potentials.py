"""
Module: electrons.atom_potentials
---------------------------------
Functions for calculating atomic potentials and performing transformations
on crystal structures.

Functions
---------
- `contrast_stretch`:
    Rescales intensity values of image series between specified percentiles
- `single_atom_potential`:
    Calculates the projected potential of a single atom using Kirkland scattering factors
- `expand_periodic_images_minimal`:
    Expands periodic images of a crystal structure to cover a given threshold distance
- `kirkland_potentials_XYZ`:
    Converts XYZData structure to PotentialSlices using FFT-based atomic positioning

Internal Functions
------------------
These functions are not exported and are used internally by the module.

- `_slice_atoms`:
    Partitions atoms into slices along the z-axis and sorts them by slice number
- `_bessel_kv`:
    Computes the modified Bessel function of the second kind
- `_compute_min_repeats`:
    Determines the minimum number of repeats needed to cover a given threshold distance
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Real, jaxtyped

from .electron_types import (PotentialSlices, ProbeModes, XYZData,
                             make_potential_slices, make_probe_modes,
                             scalar_float, scalar_int, scalar_numeric)
from .geometry import (reciprocal_lattice, rotate_structure, rotmatrix_axis,
                       rotmatrix_vectors)
from .preprocessing import kirkland_potentials
from .simulations import make_probe, stem_4D

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def contrast_stretch(
    series: Union[Float[Array, "H W"], Float[Array, "N H W"]],
    p1: float,
    p2: float,
) -> Union[Float[Array, "H W"], Float[Array, "N H W"]]:
    """
    Description
    -----------
    Rescales intensity values of image series between specified percentiles
    using pure JAX operations. Handles both 2D single images and 3D image stacks.

    Parameters
    ----------
    - `series` (Union[Float[Array, "H W"], Float[Array, "N H W"]]):
        Input image or stack of images to process
    - `p1` (float):
        Lower percentile for intensity rescaling
    - `p2` (float):
        Upper percentile for intensity rescaling

    Returns
    -------
    - `transformed` (Union[Float[Array, "H W"], Float[Array, "N H W"]]):
        Intensity-rescaled image(s) with same shape as input

    Flow
    ----
    - Handle dimension expansion for 2D inputs
    - Compute percentiles for each image independently
    - Apply rescaling transformation using vectorized operations
    - Return result with original shape
    """
    original_shape: Tuple[int, int] = series.shape
    series_reshaped: Float[Array, "N H W"] = jnp.where(
        len(original_shape) == 2, series[jnp.newaxis, :, :], series
    )

    def rescale_single_image(image: Float[Array, "H W"]) -> Float[Array, "H W"]:
        flattened: Float[Array, "HW"] = image.flatten()
        lower_bound: scalar_float = jnp.percentile(flattened, p1)
        upper_bound: scalar_float = jnp.percentile(flattened, p2)
        clipped_image: Float[Array, "H W"] = jnp.clip(image, lower_bound, upper_bound)
        range_val: scalar_float = upper_bound - lower_bound
        rescaled_image: Float[Array, "H W"] = jnp.where(
            range_val > 0, (clipped_image - lower_bound) / range_val, clipped_image
        )
        return rescaled_image

    transformed: Float[Array, "N H W"] = jax.vmap(rescale_single_image)(series_reshaped)
    final_result: Union[Float[Array, "H W"], Float[Array, "N H W"]] = jnp.where(
        len(original_shape) == 2, transformed[0], transformed
    )
    return final_result


@jaxtyped(typechecker=beartype)
def _bessel_kv(v: scalar_float, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Description
    -----------
    Computes the modified Bessel function of the second kind
    K_v(x) for real order v > 0 and x > 0,
    using a numerically stable and differentiable
    JAX-compatible approximation.

    Parameters
    ----------
    - `v` (scalar_float):
        Order of the Bessel function
    - `x` (Float[Array, "..."]):
        Positive real input array

    Returns
    -------
    - `k_v` (Float[Array, "..."]):
        Approximated values of K_v(x)

    Notes
    -----
    - Valid for v >= 0 and x > 0
    - Supports broadcasting and autodiff
    - JIT-safe and VMAP-safe
    """
    v: Float[Array, ""] = jnp.asarray(v)
    x: Float[Array, "..."] = jnp.asarray(x)
    dtype: jnp.dtype = x.dtype

    def k0_small(x: Float[Array, "..."]) -> Float[Array, "..."]:
        i0: Float[Array, "..."] = jax.scipy.special.i0(x)
        coeffs: Float[Array, "7"] = jnp.array(
            [
                -0.57721566,
                0.42278420,
                0.23069756,
                0.03488590,
                0.00262698,
                0.00010750,
                0.00000740,
            ],
            dtype=dtype,
        )
        x2: Float[Array, "..."] = (x * x) / 4.0
        powers: Float[Array, "... 7"] = jnp.power(x2[..., jnp.newaxis], jnp.arange(7))
        poly: Float[Array, "..."] = jnp.sum(coeffs * powers, axis=-1)
        return -jnp.log(x / 2.0) * i0 + poly

    def k0_large(x: Float[Array, "..."]) -> Float[Array, "..."]:
        coeffs: Float[Array, "7"] = jnp.array(
            [
                1.25331414,
                -0.07832358,
                0.02189568,
                -0.01062446,
                0.00587872,
                -0.00251540,
                0.00053208,
            ],
            dtype=dtype,
        )
        z: Float[Array, "..."] = 1.0 / x
        powers: Float[Array, "... 7"] = jnp.power(z[..., jnp.newaxis], jnp.arange(7))
        poly: Float[Array, "..."] = jnp.sum(coeffs * powers, axis=-1)

        return jnp.exp(-x) * poly / jnp.sqrt(x)

    k0_result: Float[Array, "..."] = jnp.where(x <= 1.0, k0_small(x), k0_large(x))
    return jnp.where(v == 0.0, k0_result, jnp.zeros_like(x, dtype=dtype))


@jaxtyped(typechecker=beartype)
def single_atom_potential(
    atom_no: scalar_int,
    pixel_size: scalar_float,
    grid_shape: Optional[Tuple[scalar_int, scalar_int]] = None,
    center_coords: Optional[Float[Array, "2"]] = None,
    supersampling: Optional[scalar_int] = 16,
    potential_extent: Optional[scalar_float] = 4.0,
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Calculate the projected potential of a single atom using Kirkland scattering factors.
    The potential can be centered at arbitrary coordinates within a custom grid.

    Parameters
    ----------
    - `atom_no` (scalar_int):
        Atomic number of the atom whose potential is being calculated
    - `pixel_size` (scalar_float):
        Real space pixel size in Ångstroms
    - `grid_shape` (Tuple[scalar_int, scalar_int], optional):
        Shape of the output grid (height, width). If None, calculated from potential_extent
    - `center_coords` (Float[Array, "2"], optional):
        (x, y) coordinates in Ångstroms where atom should be centered.
        If None, centers at grid center
    - `supersampling` (scalar_int, optional):
        Supersampling factor for increased accuracy. Default is 16
    - `potential_extent` (scalar_float, optional):
        Distance in Ångstroms from atom center to calculate potential. Default is 4.0 Å

    Returns
    -------
    - `potential` (Float[Array, "h w"]):
        Projected potential matrix with atom centered at specified coordinates

    Flow
    ----
    - Initialize physical constants:
        - a0 = 0.5292 Å (Bohr radius)
        - ek = 14.4 eV·Å (electron charge squared divided by 4πε₀)
        - Calculate prefactors for Bessel (term1) and Gaussian (term2) contributions
    - Load Kirkland scattering parameters:
        - Extract 12 parameters for the specified atom from preloaded Kirkland data
        - Parameters alternate between amplitudes and reciprocal space widths
    - Determine grid dimensions:
        - If grid_shape provided: use it directly, multiplied by supersampling
        - If grid_shape is None: calculate from potential_extent to ensure full coverage
        - Calculate step size as pixel_size divided by supersampling factor
    - Set atom center position:
        - If center_coords provided: use (x, y) coordinates directly
        - If center_coords is None: place atom at origin (0, 0)

    - Generate coordinate grids:
        - Create x and y coordinate arrays centered around the atom position
        - Account for supersampling in coordinate spacing
        - Use meshgrid to create 2D coordinate arrays
    - Calculate radial distances:
        - Compute distance from each grid point to the atom center
        - r = sqrt((x - center_x)² + (y - center_y)²)
    - Evaluate Bessel function contributions:
        - Calculate three Bessel K₀ terms using the first 6 Kirkland parameters
        - Each term: amplitude * K₀(2π * sqrt(width) * r)
        - Sum all three terms and multiply by term1 prefactor
    - Evaluate Gaussian contributions:
        - Calculate three Gaussian terms using the last 6 Kirkland parameters
        - Each term: (amplitude/width) * exp(-π²/width * r²)
        - Sum all three terms and multiply by term2 prefactor
    - Combine contributions:
        - Total potential = Bessel contributions + Gaussian contributions
        - Result is supersampled potential on fine grid

    - Downsample to target resolution:
        - Reshape array to group supersampling pixels together
        - Average over supersampling dimensions
        - Crop to exact target dimensions if necessary
    - Return the final potential array at the requested resolution
    """
    a0: Float[Array, ""] = jnp.asarray(0.5292)
    ek: Float[Array, ""] = jnp.asarray(14.4)
    term1: Float[Array, ""] = 4.0 * (jnp.pi**2) * a0 * ek
    term2: Float[Array, ""] = 2.0 * (jnp.pi**2) * a0 * ek
    kirkland_array: Float[Array, "103 12"] = kirkland_potentials()
    kirk_params: Float[Array, "12"] = jax.lax.dynamic_slice(
        kirkland_array, (atom_no - 1, 0), (1, 12)
    )[0]
    step_size: Float[Array, ""] = pixel_size / supersampling
    if grid_shape is None:
        grid_extent: Float[Array, ""] = potential_extent
        n_points: Int[Array, ""] = jnp.ceil(2.0 * grid_extent / step_size).astype(
            jnp.int32
        )
        grid_height: Int[Array, ""] = n_points
        grid_width: Int[Array, ""] = n_points
    else:
        grid_height: Int[Array, ""] = jnp.asarray(
            grid_shape[0] * supersampling, dtype=jnp.int32
        )
        grid_width: Int[Array, ""] = jnp.asarray(
            grid_shape[1] * supersampling, dtype=jnp.int32
        )
    if center_coords is None:
        center_x: Float[Array, ""] = 0.0
        center_y: Float[Array, ""] = 0.0
    else:
        center_x: Float[Array, ""] = center_coords[0]
        center_y: Float[Array, ""] = center_coords[1]
    y_coords: Float[Array, "h"] = (
        jnp.arange(grid_height) - grid_height // 2
    ) * step_size + center_y
    x_coords: Float[Array, "w"] = (
        jnp.arange(grid_width) - grid_width // 2
    ) * step_size + center_x
    ya: Float[Array, "h w"]
    xa: Float[Array, "h w"]
    ya, xa = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    r: Float[Array, "h w"] = jnp.sqrt((xa - center_x) ** 2 + (ya - center_y) ** 2)
    bessel_term1: Float[Array, "h w"] = kirk_params[0] * _bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[1]) * r
    )
    bessel_term2: Float[Array, "h w"] = kirk_params[2] * _bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[3]) * r
    )
    bessel_term3: Float[Array, "h w"] = kirk_params[4] * _bessel_kv(
        0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[5]) * r
    )
    part1: Float[Array, "h w"] = term1 * (bessel_term1 + bessel_term2 + bessel_term3)
    gauss_term1: Float[Array, "h w"] = (kirk_params[6] / kirk_params[7]) * jnp.exp(
        -(jnp.pi**2 / kirk_params[7]) * r**2
    )
    gauss_term2: Float[Array, "h w"] = (kirk_params[8] / kirk_params[9]) * jnp.exp(
        -(jnp.pi**2 / kirk_params[9]) * r**2
    )
    gauss_term3: Float[Array, "h w"] = (kirk_params[10] / kirk_params[11]) * jnp.exp(
        -(jnp.pi**2 / kirk_params[11]) * r**2
    )
    part2: Float[Array, "h w"] = term2 * (gauss_term1 + gauss_term2 + gauss_term3)
    supersampled_potential: Float[Array, "h w"] = part1 + part2
    if grid_shape is None:
        target_height: Int[Array, ""] = grid_height // supersampling
        target_width: Int[Array, ""] = grid_width // supersampling
    else:
        target_height: Int[Array, ""] = jnp.asarray(grid_shape[0], dtype=jnp.int32)
        target_width: Int[Array, ""] = jnp.asarray(grid_shape[1], dtype=jnp.int32)
    height: Int[Array, ""] = jnp.asarray(
        supersampled_potential.shape[0], dtype=jnp.int32
    )
    width: Int[Array, ""] = jnp.asarray(
        supersampled_potential.shape[1], dtype=jnp.int32
    )
    new_height: Int[Array, ""] = (height // supersampling) * supersampling
    new_width: Int[Array, ""] = (width // supersampling) * supersampling
    cropped: Float[Array, "h_crop w_crop"] = jax.lax.dynamic_slice(
        supersampled_potential, (0, 0), (new_height, new_width)
    )
    reshaped: Float[Array, "h_new supersampling w_new supersampling"] = cropped.reshape(
        new_height // supersampling,
        supersampling,
        new_width // supersampling,
        supersampling,
    )
    potential: Float[Array, "h_new w_new"] = jnp.mean(reshaped, axis=(1, 3))
    potential_resized: Float[Array, "h w"] = jax.lax.dynamic_slice(
        potential, (0, 0), (target_height, target_width)
    )
    return potential_resized


@jaxtyped(typechecker=beartype)
def _compute_min_repeats(
    cell: Float[Array, "3 3"], threshold_nm: scalar_float
) -> Tuple[int, int, int]:
    """
    Description
    -----------
    Internal function to compute the minimal number of unit cell repeats along each
    lattice vector direction such that the resulting supercell dimensions exceed a
    specified threshold distance. This is used to ensure periodic images are included
    for accurate potential calculations.

    Parameters
    ----------
    - `cell` (Float[Array, "3 3"]):
        Real-space unit cell matrix where rows represent lattice vectors a1, a2, a3
    - `threshold_nm` (scalar_float):
        Minimum required length in nanometers for the supercell along each direction

    Returns
    -------
    - `n_repeats` (Tuple[int, int, int]):
        Number of repeats (nx, ny, nz) needed along each lattice vector direction

    Flow
    ----
    - Calculate lattice vector lengths:
        - Compute the norm of each row in the cell matrix
        - This gives the physical length of each lattice vector in nm

    - Determine minimal repeats:
        - For each direction, divide threshold by lattice vector length
        - Use ceiling function to ensure we exceed the threshold
        - Convert to integers for use as repeat counts

    - Return repeat counts:
        - Package the three repeat values as a tuple
        - These values will be used to construct supercells that include
          sufficient periodic images for accurate calculations
    """
    lengths: Float[Array, "3"] = jnp.linalg.norm(cell, axis=1)  # shape (3,)
    n_repeats: Int[Array, "3"] = jnp.ceil(threshold_nm / lengths).astype(int)
    return tuple(n_repeats)


@jaxtyped(typechecker=beartype)
def expand_periodic_images_minimal(
    coords: Float[Array, "N 4"], cell: Float[Array, "3 3"], threshold_nm: scalar_float
) -> Tuple[Float[Array, "M 4"], Tuple[int, int, int]]:
    """
    Expand coordinates in all directions just enough to exceed (twice of) a minimum
    bounding box size along each axis.

    Parameters:
    - coords: (N, 4)
    - cell: (3, 3) lattice matrix (rows = a1, a2, a3)
    - threshold_nm: float

    Returns:
    - expanded_coords: (M, 4)
    - nx, ny, nz: number of repeats used in each direction
    """
    nx: int
    ny: int
    nz: int
    nx, ny, nz = _compute_min_repeats(cell, threshold_nm)
    nz = 0  # Set nz to 0 for 2D expansion

    i: Int[Array, "2nx+1"] = jnp.arange(-nx, nx + 1)
    j: Int[Array, "2ny+1"] = jnp.arange(-ny, ny + 1)
    k: Int[Array, "2nz+1"] = jnp.arange(-nz, nz + 1)

    ii: Int[Array, "2nx+1 2ny+1 2nz+1"]
    jj: Int[Array, "2nx+1 2ny+1 2nz+1"]
    kk: Int[Array, "2nx+1 2ny+1 2nz+1"]
    ii, jj, kk = jnp.meshgrid(i, j, k, indexing="ij")
    shifts: Int[Array, "M 3"] = jnp.stack(
        [ii.ravel(), jj.ravel(), kk.ravel()], axis=-1
    )  # (M, 3)
    # print(shifts)
    shift_vectors: Float[Array, "M 3"] = shifts @ cell  # (M, 3)

    def shift_all_atoms(shift_vec: Float[Array, "3"]) -> Float[Array, "N 4"]:
        atom_numbers: Float[Array, "N 1"] = coords[:, 0:1]
        new_coords: Float[Array, "N 4"] = jnp.hstack(
            (atom_numbers, coords[:, 1:4] + shift_vec)
        )
        return new_coords

    expanded_coords: Float[Array, "M N 4"] = jax.vmap(shift_all_atoms)(
        shift_vectors
    )  # (M, N, 4)
    # print("expanded_coords shape", expanded_coords.shape)
    return expanded_coords.reshape(-1, 4), (nx, ny, nz)


import jax.numpy as jnp


@jaxtyped(typechecker=beartype)
def _slice_atoms(
    coords: Float[Array, "N 3"],
    atom_numbers: Int[Array, "N"],
    slice_thickness: scalar_numeric,
) -> Float[Array, "N 4"]:
    """
    Description
    -----------
    Partitions atoms into slices along the z-axis and returns them sorted by slice number.
    This internal function is used to organize atomic positions for slice-by-slice
    potential calculations in electron microscopy simulations.

    Parameters
    ----------
    - `coords` (Float[Array, "N 3"]):
        Atomic positions with shape (N, 3) where columns represent x, y, z coordinates
        in Angstroms
    - `atom_numbers` (Int[Array, "N"]):
        Atomic numbers for each of the N atoms, used to identify element types
    - `slice_thickness` (scalar_numeric):
        Thickness of each slice in Angstroms. Can be float, int, or 0-dimensional
        JAX array

    Returns
    -------
    - `sorted_atoms` (Float[Array, "N 4"]):
        Array with shape (N, 4) containing [x, y, slice_num, atom_number] for each atom,
        sorted by ascending slice number. Slice numbers start from 0.

    Flow
    ----
    - Extract z-coordinates and find minimum and maximum z values
    - Calculate slice index for each atom based on its z-position:
        - Atoms are assigned to slices using floor division: (z - z_min) / slice_thickness
        - This ensures atoms at z_min are in slice 0
    - Construct output array with x, y positions, slice numbers, and atom numbers
    - Sort atoms by their slice indices to group atoms within the same slice
    - Return the sorted array for efficient slice-by-slice processing

    Notes
    -----
    - The number of slices is implicitly ceil((z_max - z_min) / slice_thickness)
    - Atoms exactly at slice boundaries are assigned to the lower slice
    - All arrays are JAX arrays for compatibility with JIT compilation
    """
    z_coords: Float[Array, "N"] = coords[:, 2]
    z_min: scalar_float = jnp.min(z_coords)
    slice_indices: Real[Array, "N"] = jnp.floor((z_coords - z_min) / slice_thickness)
    sorted_atoms_presort: Float[Array, "N 4"] = jnp.column_stack(
        [
            coords[:, 0],
            coords[:, 1],
            slice_indices.astype(jnp.float32),
            atom_numbers.astype(jnp.float32),
        ]
    )
    sorted_order: Real[Array, "N"] = jnp.argsort(slice_indices)
    sorted_atoms: Float[Array, "N 4"] = sorted_atoms_presort[sorted_order]
    return sorted_atoms


@jaxtyped(typechecker=beartype)
def kirkland_potentials_XYZ(
    xyz_data: XYZData,
    pixel_size: scalar_float,
    slice_thickness: scalar_float = 1.0,
    padding: scalar_float = 4.0,
) -> PotentialSlices:
    """
    Description
    -----------
    Converts XYZData structure to PotentialSlices by calculating atomic potentials
    and assembling them into slices using FFT shifts for precise positioning.

    Parameters
    ----------
    - `xyz_data` (XYZData):
        Input structure containing atomic positions and numbers
    - `pixel_size` (scalar_float):
        Size of each pixel in Angstroms (becomes calib in PotentialSlices)
    - `slice_thickness` (scalar_float, optional):
        Thickness of each slice in Angstroms. Default is 1.0
    - `padding` (scalar_float, optional):
        Padding in Angstroms added to all sides. Default is 4.0

    Returns
    -------
    - `potential_slices` (PotentialSlices):
        Sliced potentials with wraparound artifacts removed

    Flow
    ----
    - Extract positions and atomic numbers from XYZData
    - Use _slice_atoms to partition atoms by z-coordinate into slices
    - Calculate extents with padding to determine uniform slice dimensions
    - Pre-calculate single atom potentials for each unique atomic species
    - For each slice:
        - Create zeros array with padded dimensions
        - For each atom in slice:
            - Place appropriate atomic potential at center
            - FFT shift to actual position
            - Accumulate contributions
    - Crop padding and return PotentialSlices
    """
    positions: Float[Array, "N 3"] = xyz_data.positions
    atomic_numbers: Int[Array, "N"] = xyz_data.atomic_numbers
    sliced_atoms: Float[Array, "N 4"] = _slice_atoms(
        coords=positions,
        atom_numbers=atomic_numbers,
        slice_thickness=slice_thickness,
    )
    x_coords: Float[Array, "N"] = sliced_atoms[:, 0]
    y_coords: Float[Array, "N"] = sliced_atoms[:, 1]
    slice_indices: Int[Array, "N"] = sliced_atoms[:, 2].astype(jnp.int32)
    atom_nums: Int[Array, "N"] = sliced_atoms[:, 3].astype(jnp.int32)
    x_min: scalar_float = jnp.min(x_coords) - padding
    x_max: scalar_float = jnp.max(x_coords) + padding
    y_min: scalar_float = jnp.min(y_coords) - padding
    y_max: scalar_float = jnp.max(y_coords) + padding
    width: Int[Array, ""] = jnp.ceil((x_max - x_min) / pixel_size).astype(jnp.int32)
    height: Int[Array, ""] = jnp.ceil((y_max - y_min) / pixel_size).astype(jnp.int32)
    unique_atoms: Int[Array, "n_unique"] = jnp.unique(atom_nums)

    def calc_single_potential(atom_no: scalar_int) -> Float[Array, "h w"]:
        return single_atom_potential(
            atom_no=atom_no,
            pixel_size=pixel_size,
            grid_shape=(height, width),
            center_coords=jnp.array([0.0, 0.0]),
            supersampling=16,
            potential_extent=4.0,
        )

    atomic_potentials: Float[Array, "n_unique h w"] = jax.vmap(calc_single_potential)(
        unique_atoms
    )
    max_atom_num = jnp.max(unique_atoms)
    atom_to_idx_array = jnp.full(max_atom_num + 1, -1, dtype=jnp.int32)
    atom_to_idx_array = atom_to_idx_array.at[unique_atoms].set(
        jnp.arange(len(unique_atoms))
    )
    n_slices: int = int(jnp.max(slice_indices) + 1)
    all_slices: Float[Array, "h w n_slices"] = jnp.zeros(
        (height, width, n_slices), dtype=jnp.float32
    )
    ky: Float[Array, "h 1"] = jnp.fft.fftfreq(height, d=1.0).reshape(-1, 1)
    kx: Float[Array, "1 w"] = jnp.fft.fftfreq(width, d=1.0).reshape(1, -1)

    def process_single_slice(slice_idx: int) -> Float[Array, "h w"]:
        atoms_mask: Bool[Array, "N"] = slice_indices == slice_idx
        slice_x: Float[Array, "n_atoms"] = x_coords[atoms_mask]
        slice_y: Float[Array, "n_atoms"] = y_coords[atoms_mask]
        slice_atoms: Int[Array, "n_atoms"] = atom_nums[atoms_mask]
        slice_potential: Float[Array, "h w"] = jnp.zeros(
            (height, width), dtype=jnp.float32
        )
        pixel_x: Float[Array, "n_atoms"] = (slice_x - x_min) / pixel_size
        pixel_y: Float[Array, "n_atoms"] = (slice_y - y_min) / pixel_size
        center_x: float = width / 2.0
        center_y: float = height / 2.0
        shift_x: Float[Array, "n_atoms"] = pixel_x - center_x
        shift_y: Float[Array, "n_atoms"] = pixel_y - center_y

        def add_atom_contribution(
            carry: Float[Array, "h w"],
            atom_data: Tuple[scalar_float, scalar_float, scalar_int],
        ) -> Tuple[Float[Array, "h w"], None]:
            slice_pot, sx, sy, atom_no = carry, *atom_data
            atom_idx = atom_to_idx_array[atom_no]
            atom_pot: Float[Array, "h w"] = atomic_potentials[atom_idx]
            phase: Complex[Array, "h w"] = jnp.exp(2j * jnp.pi * (kx * sx + ky * sy))
            atom_pot_fft: Complex[Array, "h w"] = jnp.fft.fft2(atom_pot)
            shifted_fft: Complex[Array, "h w"] = atom_pot_fft * phase
            shifted_pot: Float[Array, "h w"] = jnp.real(jnp.fft.ifft2(shifted_fft))
            return slice_pot + shifted_pot, None

        slice_potential, _ = jax.lax.scan(
            add_atom_contribution,
            slice_potential,
            (shift_x, shift_y, slice_atoms),
        )
        return slice_potential

    slice_indices_array = jnp.arange(n_slices)
    processed_slices = jax.vmap(process_single_slice)(slice_indices_array)
    all_slices = processed_slices.transpose(1, 2, 0)
    crop_pixels: int = int(jnp.round(padding / pixel_size))
    cropped_slices: Float[Array, "h_crop w_crop n_slices"] = all_slices[
        crop_pixels:-crop_pixels, crop_pixels:-crop_pixels, :
    ]
    pot_slices: PotentialSlices = make_potential_slices(
        slices=cropped_slices,
        slice_thickness=slice_thickness,
        calib=pixel_size,
    )
    return pot_slices
