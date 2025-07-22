"""
Module: electrons.atom_potentials
---------------------------------
Functions for calculating atomic potentials and performing transformations
on crystal structures.

Functions
---------
- `contrast_stretch`:
    Rescales intensity values of image series between specified percentiles
- `atomic_potential`:
    Calculates the projected potential of a single atom using Kirkland scattering factors
- `rotation_matrix_from_vectors`:
    Returns a proper rotation matrix that rotates v1 to v2
- `rotation_matrix_about_axis`:
    Returns a rotation matrix that rotates around a given axis by an angle theta
- `rotate_structure`:
    Rotates a crystal structure by a given angle about a specified axis
- `reciprocal_lattice`:
    Computes the reciprocal lattice vectors for a given unit cell
- `compute_min_repeats`:
    Determines the minimum number of repeats needed to cover a given threshold distance
- `expand_periodic_images_minimal`:
    Expands periodic images of a crystal structure to cover a given threshold distance

Internal Functions
------------------
These functions are not exported and are used internally by the module.

- `_bessel_kv`:
    Computes the modified Bessel function of the second kind
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Float, Int, jaxtyped

from .electron_types import (PotentialSlices, ProbeModes,
                             make_potential_slices, make_probe_modes,
                             scalar_float, scalar_int)
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
    # Convert to JAX arrays, preserving dtype
    v = jnp.asarray(v)
    x = jnp.asarray(x)

    # Get the dtype to use throughout
    dtype = x.dtype

    def k0_small(x):
        # For K_0(x) when x <= 1, use series expansion
        # Based on Numerical Recipes formulation
        i0 = jax.scipy.special.i0(x)
        coeffs = jnp.array(
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

        x2 = x * x / 4.0
        poly = coeffs[0]
        xn = 1.0
        for i, c in enumerate(coeffs[1:]):
            xn *= x2
            poly += c * xn

        return -jnp.log(x / 2.0) * i0 + poly

    def k0_large(x):
        # For K_0(x) when x > 1, use asymptotic expansion
        # Based on Numerical Recipes formulation
        coeffs = jnp.array(
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

        z = 1.0 / x
        poly = coeffs[0]
        zn = 1.0
        for c in coeffs[1:]:
            zn *= z
            poly += c * zn

        return jnp.exp(-x) * poly / jnp.sqrt(x)

    # For now, just handle v=0 case which is what's used in the code
    # Use jnp.where for vectorized conditional
    k0_result = jnp.where(x <= 1.0, k0_small(x), k0_large(x))

    # Return k0_result if v == 0, otherwise zeros
    return jnp.where(v == 0.0, k0_result, jnp.zeros_like(x, dtype=dtype))


@jaxtyped(typechecker=beartype)
def atomic_potential(
    atom_no: scalar_int,
    pixel_size: scalar_float,
    grid_shape: Optional[Tuple[scalar_int, scalar_int]] = None,
    center_coords: Optional[Float[Array, "2"]] = None,
    sampling: Optional[scalar_int] = 16,
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
    - `sampling` (scalar_int, optional):
        Supersampling factor for increased accuracy. Default is 16
    - `potential_extent` (scalar_float, optional):
        Distance in Ångstroms from atom center to calculate potential. Default is 4.0 Å
    - `datafile` (str, optional):
        Path to CSV file containing Kirkland scattering factors

    Returns
    -------
    - `potential` (Float[Array, "h w"]):
        Projected potential matrix with atom centered at specified coordinates

    Flow
    ----
    - Define physical constants and load Kirkland parameters
    - Determine grid size and center coordinates
    - Calculate step size for supersampling
    - Create coordinate grid with atom centered at specified position
    - Calculate radial distances from atom center
    - Compute Bessel and Gaussian terms using Kirkland parameters
    - Combine terms to get total potential
    - Downsample to target resolution using average pooling
    - Return final potential matrix
    """
    a0: Float[Array, ""] = jnp.asarray(0.5292)
    ek: Float[Array, ""] = jnp.asarray(14.4)
    term1: Float[Array, ""] = 4.0 * (jnp.pi**2) * a0 * ek
    term2: Float[Array, ""] = 2.0 * (jnp.pi**2) * a0 * ek
    kirkland_array: Float[Array, "103 12"] = kirkland_potentials()
    kirk_params: Float[Array, "12"] = kirkland_array[atom_no - 1, :]
    step_size: Float[Array, ""] = pixel_size / sampling
    if grid_shape is None:
        grid_extent: Float[Array, ""] = potential_extent
        n_points: Int[Array, ""] = jnp.ceil(2.0 * grid_extent / step_size).astype(
            jnp.int32
        )
        grid_height: Int[Array, ""] = n_points
        grid_width: Int[Array, ""] = n_points
    else:
        grid_height: Int[Array, ""] = jnp.asarray(
            grid_shape[0] * sampling, dtype=jnp.int32
        )
        grid_width: Int[Array, ""] = jnp.asarray(
            grid_shape[1] * sampling, dtype=jnp.int32
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
        target_height: Int[Array, ""] = grid_height // sampling
        target_width: Int[Array, ""] = grid_width // sampling
    else:
        target_height: Int[Array, ""] = jnp.asarray(grid_shape[0], dtype=jnp.int32)
        target_width: Int[Array, ""] = jnp.asarray(grid_shape[1], dtype=jnp.int32)
    height: Int[Array, ""] = supersampled_potential.shape[0]
    width: Int[Array, ""] = supersampled_potential.shape[1]
    new_height: Int[Array, ""] = (height // sampling) * sampling
    new_width: Int[Array, ""] = (width // sampling) * sampling
    cropped: Float[Array, "h_crop w_crop"] = supersampled_potential[
        :new_height, :new_width
    ]
    reshaped: Float[Array, "h_new sampling w_new sampling"] = cropped.reshape(
        new_height // sampling, sampling, new_width // sampling, sampling
    )
    potential: Float[Array, "h_new w_new"] = jnp.mean(reshaped, axis=(1, 3))
    potential_resized: Float[Array, "h w"] = potential[:target_height, :target_width]
    return potential_resized


def rotation_matrix_from_vectors(v1, v2):
    """
    Return a proper rotation matrix that rotates v1 to v2.
    Uses the classic Rodrigues rotation formula.
    """
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)

    cross = jnp.cross(v1, v2)
    dot = jnp.dot(v1, v2)
    sin_theta = jnp.linalg.norm(cross)

    def fallback_parallel():
        return jnp.eye(3)

    def fallback_opposite():
        # Pick any orthogonal axis to v1
        ortho = jnp.where(
            jnp.abs(v1[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
        )
        axis = jnp.cross(v1, ortho)
        axis = axis / jnp.linalg.norm(axis)
        K = jnp.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return jnp.eye(3) + 2 * K @ K  # 180° rotation

    def compute():
        axis = cross / sin_theta
        K = jnp.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return jnp.eye(3) + sin_theta * K + (1 - dot) * (K @ K)

    is_parallel = sin_theta < 1e-8
    is_opposite = dot < -0.9999

    return jax.lax.cond(
        is_parallel,
        lambda: jax.lax.cond(is_opposite, fallback_opposite, fallback_parallel),
        compute,
    )


def rotation_matrix_about_axis(axis, theta):
    """
    Return a rotation matrix that rotates around a given axis by an angle theta.
    Uses the Rodrigues' rotation formula.
    """
    axis = axis / jnp.linalg.norm(axis)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    ux, uy, uz = axis

    return jnp.array(
        [
            [
                cos_theta + ux**2 * (1 - cos_theta),
                ux * uy * (1 - cos_theta) - uz * sin_theta,
                ux * uz * (1 - cos_theta) + uy * sin_theta,
            ],
            [
                uy * ux * (1 - cos_theta) + uz * sin_theta,
                cos_theta + uy**2 * (1 - cos_theta),
                uy * uz * (1 - cos_theta) - ux * sin_theta,
            ],
            [
                uz * ux * (1 - cos_theta) - uy * sin_theta,
                uz * uy * (1 - cos_theta) + ux * sin_theta,
                cos_theta + uz**2 * (1 - cos_theta),
            ],
        ]
    )


def rotate_structure(coords, cell, R, theta=0):
    """
    Rotate atomic coordinates and unit cell.
    - coords: (N, 4)
    - cell: (3, 3)
    - R: (3, 3) rotation matrix
    - theta: rotation angle for in-plane rotation (not used in this function, but can be useful for future extensions)
    """
    rotated_coords = coords[:, 1:4] @ R.T
    rotated_coords = jnp.hstack(
        (coords[:, 0:1], rotated_coords)
    )  # Keep the first column (atom IDs)
    rotated_cell = cell @ R.T
    if theta != 0:
        # Apply in-plane rotation if needed
        in_plane_rotation = rotation_matrix_about_axis(jnp.array([0, 0, 1]), theta)
        rotated_coords_in_plane = rotated_coords[:, 1:4] @ in_plane_rotation.T
        rotated_coords = jnp.hstack((rotated_coords[:, 0:1], rotated_coords_in_plane))
    return rotated_coords, rotated_cell


def reciprocal_lattice(cell):
    """
    Compute reciprocal lattice vectors (rows) from real-space cell (3x3).
    """
    a1, a2, a3 = cell
    V = jnp.dot(a1, jnp.cross(a2, a3))
    b1 = 2 * jnp.pi * jnp.cross(a2, a3) / V
    b2 = 2 * jnp.pi * jnp.cross(a3, a1) / V
    b3 = 2 * jnp.pi * jnp.cross(a1, a2) / V
    return jnp.stack([b1, b2, b3])


def compute_min_repeats(cell, threshold_nm):
    """
    Compute the minimal number of repeats along each lattice vector
    so that the resulting supercell length exceeds `threshold_nm`.

    Parameters:
    - cell: (3, 3) real-space lattice (rows = a1, a2, a3)
    - threshold_nm: float, length threshold in nm

    Returns:
    - nx, ny, nz: integers
    """
    # Compute norms of lattice vectors
    lengths = jnp.linalg.norm(cell, axis=1)  # shape (3,)
    n_repeats = jnp.ceil(threshold_nm / lengths).astype(int)
    return tuple(n_repeats)


def expand_periodic_images_minimal(coords, cell, threshold_nm):
    """
    Expand coordinates in all directions just enough to exceed (twice of) a minimum
    bounding box size along each axis.

    Parameters:
    - coords: (N, 4)
    - cell: (3, 3) lattice matrix (rows = a1, a2, a3)
    - threshold_nm: float

    Returns:
    - expanded_coords: (M, 3)
    - nx, ny, nz: number of repeats used in each direction
    """
    nx, ny, nz = compute_min_repeats(cell, threshold_nm)
    nz = 0  # Set nz to 0 for 2D expansion

    i = jnp.arange(-nx, nx + 1)
    j = jnp.arange(-ny, ny + 1)
    k = jnp.arange(-nz, nz + 1)

    ii, jj, kk = jnp.meshgrid(i, j, k, indexing="ij")
    shifts = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1)  # (M, 3)
    # print(shifts)
    shift_vectors = shifts @ cell  # (M, 3)

    def shift_all_atoms(shift_vec):
        atom_numbers = coords[:, 0:1]
        new_coords = jnp.hstack((atom_numbers, coords[:, 1:4] + shift_vec))
        return new_coords

    expanded_coords = jax.vmap(shift_all_atoms)(shift_vectors)  # (M, N, 3)
    # print("expanded_coords shape", expanded_coords.shape)
    return expanded_coords.reshape(-1, 4), (nx, ny, nz)


import jax.numpy as jnp


def slice_atoms(coords, slice_thickness):
    """
    Assign atoms to slices and group them using sorted indices.

    Returns:
    - grouped_indices: (N,) reordered atom indices
    - slice_bounds: list of start indices for each slice in grouped_indices
    - z_min, z_max
    """
    z_coords = coords[:, 3]
    z_min = jnp.min(z_coords)
    z_max = jnp.max(z_coords)
    n_slices = jnp.ceil((z_max - z_min) / slice_thickness).astype(int)

    # Slice index for each atom
    slice_indices = jnp.floor((z_coords - z_min) / slice_thickness).astype(int)

    # Sort by slice index
    sorted_order = jnp.argsort(slice_indices)
    sorted_slice_indices = slice_indices[sorted_order]

    # Count how many atoms per slice
    slice_counts = jnp.bincount(sorted_slice_indices, length=n_slices)

    # Compute slice start positions (cumulative sum)
    slice_bounds = jnp.cumsum(
        jnp.pad(slice_counts[:-1], (1, 0))
    )  # Start index of each slice

    return sorted_order, slice_bounds, z_min, z_max


def build_slice_potential(
    coords, canvas_shape, minmaxes, pixel_size, atomic_potential_fn
):
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
    H, W = canvas_shape
    canvas = np.zeros((H, W))
    x_min, x_max, y_min, y_max = minmaxes

    def add_single_atom(canvas, line):
        # Compute center position in pixels

        i_center = np.floor((line[1] - x_min) / pixel_size).astype(int)
        j_center = np.floor((line[2] - y_min) / pixel_size).astype(int)
        atom_pot = atomic_potential_fn[np.round(line[0]).astype(int)]  # (h, w)

        h, w = atom_pot.shape
        half_h = h // 2
        half_w = w // 2

        # Bounds in canvas
        i_start = i_center - half_h
        i_end = i_center + half_h
        j_start = j_center - half_w
        j_end = j_center + half_w

        # Compute valid overlapping region between atom_pot and canvas
        i_start_clip = np.maximum(i_start, 0)
        i_end_clip = np.minimum(i_end, H)
        j_start_clip = np.maximum(j_start, 0)
        j_end_clip = np.minimum(j_end, W)

        # Corresponding indices in atom_pot
        ai_start = i_start_clip - i_start
        ai_end = ai_start + (i_end_clip - i_start_clip)
        aj_start = j_start_clip - j_start
        aj_end = aj_start + (j_end_clip - j_start_clip)
        _h = ai_end - ai_start
        _w = aj_end - aj_start

        slice_shape = np.array([_h, _w])
        pot_start = np.array([ai_start, aj_start])
        clip_start = np.array([i_start_clip, j_start_clip])

        # Add clipped portion
        # clipped_atom_pot = jax.lax.dynamic_slice(atom_pot, pot_start, slice_shape)
        clipped_atom_pot = atom_pot[
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


def build_slice_wrapper(
    coords, sorted_order, slice_bounds, kirkland_jax, pixel_size=0.1
):
    x_max = jnp.max(coords[:, 1])
    x_min = jnp.min(coords[:, 1])
    y_max = jnp.max(coords[:, 2])
    y_min = jnp.min(coords[:, 2])
    H = jnp.ceil((x_max - x_min) / pixel_size).astype(int)
    W = jnp.ceil((y_max - y_min) / pixel_size).astype(int)

    def build_slice_i(i):
        i = int(i)
        atoms_in_slice_i = sorted_order[slice_bounds[i] : slice_bounds[i + 1]]
        if len(atoms_in_slice_i) == 0:
            return jnp.zeros((H, W))
        coords_in_slice = coords[atoms_in_slice_i]
        canvas = build_slice_potential(
            coords_in_slice,
            (H, W),
            (x_min, x_max, y_min, y_max),
            pixel_size,
            kirkland_jax,
        )
        return canvas

    return [build_slice_i(i) for i in range(len(slice_bounds) - 1)]


def overall_wrapper(
    atoms, metadata, zone_hkl, theta, pixel_size, kirkland_jax, poss=[[0, 0]]
):
    tic = time.time()
    atoms_jnp = jnp.asarray(atoms, dtype=jnp.float32)
    metadata_jnp = jnp.asarray(metadata["lattice"], dtype=jnp.float32)
    expanded_coords, (nx, ny, nz) = expand_periodic_images_minimal(
        atoms_jnp, metadata_jnp, 10
    )
    expanded_coords = jnp.hstack(
        (
            expanded_coords[:, 0:1],
            expanded_coords[:, 1:4] - jnp.mean(expanded_coords[:, 1:4], axis=0),
        )
    )  # Center the coordinates
    recip = reciprocal_lattice(metadata["lattice"])
    zone_vector = zone_hkl @ recip
    rotation = rotation_matrix_from_vectors(zone_vector, jnp.array(zone_hkl))
    rotated_coords, rotated_cell = rotate_structure(
        expanded_coords, metadata_jnp, rotation, theta
    )
    # i-x-h, j-y-w

    sorted_coords, slice_bounds, z_min, z_max = slice_atoms(
        rotated_coords, slice_thickness=1
    )  # in Angstrom
    slices = build_slice_wrapper(
        rotated_coords, sorted_coords, slice_bounds, kirkland_jax, pixel_size
    )

    slices_array = np.array(slices)
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
    probe = make_probe(
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

    slices_array = jnp.asarray(slices_array, dtype=jnp.complex64)

    sigma = 0.001  # /(V*Angstrom) at 100 kV

    phase_shift = jnp.exp(-1j * sigma * slices_array)

    pot_slices: PotentialSlices = make_potential_slices(phase_shift, 1.0, 0.1)
    beam: ProbeModes = make_probe_modes(probe, jnp.array([1.0]), calib=0.1)
    toc = time.time()
    print("Time taken for preprocessing:", toc - tic, "seconds")

    # cbed = cbed(slices_array, probe, jnp.asarray([[200.0, 200.0]]), jnp.asarray(bin_size, dtype= jnp.float16), jnp.asarray(100), 0.1)
    poss = np.array(poss)
    center_pixel = np.array([slices_array.shape[0] // 2, slices_array.shape[1] // 2])
    pixel_shifts = poss / pixel_size
    pixels = jnp.asarray(pixel_shifts + center_pixel, dtype=jnp.float32)

    # cbed_pattern = jax.jit(cbed)(pot_slices, beam, 100.0)
    cbed_patterns = stem_4D(
        pot_slice=pot_slices,
        beam=beam,
        positions=pixels,
        voltage_kV=100.0,
        calib_ang=pixel_size,
    )
    print("time taken for cbed calculation:", time.time() - toc, "seconds")

    return cbed_patterns, slices, rotated_coords, rotated_cell
