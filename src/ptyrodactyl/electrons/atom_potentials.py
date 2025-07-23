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

Internal Functions
------------------
These functions are not exported and are used internally by the module.

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
from jaxtyping import Array, Float, Int, jaxtyped

from .electron_types import (PotentialSlices, ProbeModes,
                             make_potential_slices, make_probe_modes,
                             scalar_float, scalar_int)
from .geometry import (reciprocal_lattice, rotate_structure, rotmatrix_axis,
                       rotmatrix_vectors)
from .preprocessing import kirkland_potentials
from .simulations import make_probe, stem_4D


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
    # Compute norms of lattice vectors
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
def slice_atoms(
    coords: Float[Array, "N 4"], slice_thickness: scalar_float
) -> Tuple[Int[Array, "N"], Int[Array, "n_slices"], scalar_float, scalar_float]:
    """
    Assign atoms to slices and group them using sorted indices.

    Returns:
    - grouped_indices: (N,) reordered atom indices
    - slice_bounds: list of start indices for each slice in grouped_indices
    - z_min, z_max
    """
    z_coords: Float[Array, "N"] = coords[:, 3]
    z_min: scalar_float = jnp.min(z_coords)
    z_max: scalar_float = jnp.max(z_coords)
    n_slices: scalar_int = jnp.ceil((z_max - z_min) / slice_thickness).astype(int)

    # Slice index for each atom
    slice_indices: Int[Array, "N"] = jnp.floor(
        (z_coords - z_min) / slice_thickness
    ).astype(int)

    # Sort by slice index
    sorted_order: Int[Array, "N"] = jnp.argsort(slice_indices)
    sorted_slice_indices: Int[Array, "N"] = slice_indices[sorted_order]

    # Count how many atoms per slice
    slice_counts: Int[Array, "n_slices"] = jnp.bincount(
        sorted_slice_indices, length=n_slices
    )

    # Compute slice start positions (cumulative sum)
    slice_bounds: Int[Array, "n_slices"] = jnp.cumsum(
        jnp.pad(slice_counts[:-1], (1, 0))
    )  # Start index of each slice

    return sorted_order, slice_bounds, z_min, z_max


@jaxtyped(typechecker=beartype)
def build_slice_potential(
    coords: Float[Array, "N 4"],
    canvas_shape: Tuple[int, int],
    minmaxes: Tuple[float, float, float, float],
    pixel_size: scalar_float,
    atomic_potential_fn: Union[Float[Array, "atom_types h w"], np.ndarray],
) -> np.ndarray:
    """
    Sum 2D atomic potentials into a slice canvas, clipping contributions
    that fall outside the canvas bounds.

    Parameters:
    - coords: (N, 4) array of atomic xy positions in angstroms
    - canvas_shape: (H, W)
    - minmaxes: (x_min, x_max, y_min, y_max) in angstroms
    - pixel_size: angstroms per pixel
    - atomic_potential_fn: () -> (h, w) 2D array centered at (0,0)

    Returns:
    - (H, W) 2D potential array
    """
    H: int
    W: int
    H, W = canvas_shape
    canvas: np.ndarray = np.zeros((H, W))
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_min, x_max, y_min, y_max = minmaxes

    def add_single_atom(canvas: np.ndarray, line: np.ndarray) -> np.ndarray:
        # Compute center position in pixels

        i_center: int = np.floor((line[1] - x_min) / pixel_size).astype(int)
        j_center: int = np.floor((line[2] - y_min) / pixel_size).astype(int)
        atom_pot: np.ndarray = atomic_potential_fn[
            np.round(line[0]).astype(int)
        ]  # (h, w)

        h: int
        w: int
        h, w = atom_pot.shape
        half_h: int = h // 2
        half_w: int = w // 2

        # Bounds in canvas
        i_start: int = i_center - half_h
        i_end: int = i_center + half_h
        j_start: int = j_center - half_w
        j_end: int = j_center + half_w

        # Compute valid overlapping region between atom_pot and canvas
        i_start_clip: int = np.maximum(i_start, 0)
        i_end_clip: int = np.minimum(i_end, H)
        j_start_clip: int = np.maximum(j_start, 0)
        j_end_clip: int = np.minimum(j_end, W)

        # Corresponding indices in atom_pot
        ai_start: int = i_start_clip - i_start
        ai_end: int = ai_start + (i_end_clip - i_start_clip)
        aj_start: int = j_start_clip - j_start
        aj_end: int = aj_start + (j_end_clip - j_start_clip)
        _h: int = ai_end - ai_start
        _w: int = aj_end - aj_start

        slice_shape: np.ndarray = np.array([_h, _w])
        pot_start: np.ndarray = np.array([ai_start, aj_start])
        clip_start: np.ndarray = np.array([i_start_clip, j_start_clip])

        # Add clipped portion
        # clipped_atom_pot = jax.lax.dynamic_slice(atom_pot, pot_start, slice_shape)
        clipped_atom_pot: np.ndarray = atom_pot[
            pot_start[0] : pot_start[0] + slice_shape[0],
            pot_start[1] : pot_start[1] + slice_shape[1],
        ]
        # original_values = jax.lax.dynamic_slice(canvas, clip_start, slice_shape)
        canvas[
            clip_start[0] : clip_start[0] + slice_shape[0],
            clip_start[1] : clip_start[1] + slice_shape[1],
        ] += clipped_atom_pot
        # new_values = original_values + clipped_atom_pot
        # canvas = jax.lax.dynamic_update_slice(canvas, new_values, clip_start)
        return canvas

    # Loop over all atoms (could be vmapped, but usually small per slice)
    for line in coords:
        canvas = add_single_atom(canvas, line)

    # canvas = jax.lax.fori_loop(0, coords.shape[0], lambda i, c: add_single_atom(c, i), canvas)

    return canvas


@jaxtyped(typechecker=beartype)
def build_slice_wrapper(
    coords: Float[Array, "N 4"],
    sorted_order: Int[Array, "N"],
    slice_bounds: Int[Array, "n_slices"],
    kirkland_jax: Union[Float[Array, "atom_types h w"], np.ndarray],
    pixel_size: scalar_float = 0.1,
) -> list:
    x_max: scalar_float = jnp.max(coords[:, 1])
    x_min: scalar_float = jnp.min(coords[:, 1])
    y_max: scalar_float = jnp.max(coords[:, 2])
    y_min: scalar_float = jnp.min(coords[:, 2])
    H: scalar_int = jnp.ceil((x_max - x_min) / pixel_size).astype(int)
    W: scalar_int = jnp.ceil((y_max - y_min) / pixel_size).astype(int)

    def build_slice_i(i: int) -> Union[Float[Array, "H W"], np.ndarray]:
        i = int(i)
        atoms_in_slice_i: Int[Array, "n_atoms"] = sorted_order[
            slice_bounds[i] : slice_bounds[i + 1]
        ]
        if len(atoms_in_slice_i) == 0:
            return jnp.zeros((H, W))
        coords_in_slice: Float[Array, "n_atoms 4"] = coords[atoms_in_slice_i]
        canvas: np.ndarray = build_slice_potential(
            coords_in_slice,
            (H, W),
            (x_min, x_max, y_min, y_max),
            pixel_size,
            kirkland_jax,
        )
        return canvas

    return [build_slice_i(i) for i in range(len(slice_bounds) - 1)]


@jaxtyped(typechecker=beartype)
def overall_wrapper(
    atoms: Union[np.ndarray, Float[Array, "N 4"]],
    metadata: dict,
    zone_hkl: Union[np.ndarray, Float[Array, "3"]],
    theta: scalar_float,
    pixel_size: scalar_float,
    kirkland_jax: Union[Float[Array, "atom_types h w"], np.ndarray],
    poss: list = [[0, 0]],
) -> Tuple[
    Float[Array, "n_positions H W"], list, Float[Array, "M 4"], Float[Array, "3 3"]
]:
    tic: float = time.time()
    atoms_jnp: Float[Array, "N 4"] = jnp.asarray(atoms, dtype=jnp.float32)
    metadata_jnp: Float[Array, "3 3"] = jnp.asarray(
        metadata["lattice"], dtype=jnp.float32
    )
    expanded_coords: Float[Array, "M 4"]
    nx: int
    ny: int
    nz: int
    expanded_coords, (nx, ny, nz) = expand_periodic_images_minimal(
        atoms_jnp, metadata_jnp, 10
    )
    expanded_coords = jnp.hstack(
        (
            expanded_coords[:, 0:1],
            expanded_coords[:, 1:4] - jnp.mean(expanded_coords[:, 1:4], axis=0),
        )
    )  # Center the coordinates
    recip: Float[Array, "3 3"] = reciprocal_lattice(metadata["lattice"])
    zone_vector: Float[Array, "3"] = zone_hkl @ recip
    rotation: Float[Array, "3 3"] = rotmatrix_vectors(zone_vector, jnp.array(zone_hkl))
    rotated_coords: Float[Array, "M 4"]
    rotated_cell: Float[Array, "3 3"]
    rotated_coords, rotated_cell = rotate_structure(
        expanded_coords, metadata_jnp, rotation, theta
    )
    # i-x-h, j-y-w

    sorted_coords: Int[Array, "M"]
    slice_bounds: Int[Array, "n_slices"]
    z_min: scalar_float
    z_max: scalar_float
    sorted_coords, slice_bounds, z_min, z_max = slice_atoms(
        rotated_coords, slice_thickness=1
    )  # in Angstrom
    slices: list = build_slice_wrapper(
        rotated_coords, sorted_coords, slice_bounds, kirkland_jax, pixel_size
    )

    slices_array: np.ndarray = np.array(slices)
    # chop slices_array down to squares
    if slices_array.shape[2] > slices_array.shape[1]:
        slices_array = slices_array[
            :,
            :,
            slices_array.shape[2] // 2
            - slices_array.shape[1] // 2 : slices_array.shape[2] // 2
            + slices_array.shape[1] // 2,
        ]
    else:
        slices_array = slices_array[
            :,
            slices_array.shape[1] // 2
            - slices_array.shape[2] // 2 : slices_array.shape[1] // 2
            + slices_array.shape[2] // 2,
            :,
        ]
    probe: Float[Array, "H W 1"] = make_probe(
        aperture=5,
        voltage=100,
        defocus=0.0,
        image_size=jnp.array(slices_array[0].shape),
        calibration_pm=pixel_size * 100,
        # defocus = -0.0,
        # c3=0.0,
        # c5=0.0
    )
    probe = probe[:, :, jnp.newaxis]

    # Reorganize the slices so that the third dimension becomes the first dimension
    slices_array = np.moveaxis(slices_array, 0, -1)

    slices_array: Float[Array, "H W n_slices"] = jnp.asarray(
        slices_array, dtype=jnp.complex64
    )

    sigma: scalar_float = 0.001  # /(V*Angstrom) at 100 kV

    phase_shift: Float[Array, "H W n_slices"] = jnp.exp(-1j * sigma * slices_array)

    pot_slices: PotentialSlices = make_potential_slices(phase_shift, 1.0, 0.1)
    beam: ProbeModes = make_probe_modes(probe, jnp.array([1.0]), calib=0.1)
    toc: float = time.time()
    print("Time taken for preprocessing:", toc - tic, "seconds")

    # cbed = cbed(slices_array, probe, jnp.asarray([[200.0, 200.0]]), jnp.asarray(bin_size, dtype= jnp.float16), jnp.asarray(100), 0.1)
    poss: np.ndarray = np.array(poss)
    center_pixel: np.ndarray = np.array(
        [slices_array.shape[0] // 2, slices_array.shape[1] // 2]
    )
    pixel_shifts: np.ndarray = poss / pixel_size
    pixels: Float[Array, "n_positions 2"] = jnp.asarray(
        pixel_shifts + center_pixel, dtype=jnp.float32
    )

    # cbed_pattern = jax.jit(cbed)(pot_slices, beam, 100.0)
    cbed_patterns: Float[Array, "n_positions H W"] = stem_4D(
        pot_slice=pot_slices,
        beam=beam,
        positions=pixels,
        voltage_kV=100.0,
        calib_ang=pixel_size,
    )
    print("time taken for cbed calculation:", time.time() - toc, "seconds")

    return cbed_patterns, slices, rotated_coords, rotated_cell
