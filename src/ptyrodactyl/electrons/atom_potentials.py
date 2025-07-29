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
    Calculates the projected potential of a single atom using Kirkland
    scattering factors
- `kirkland_potentials_XYZ`:
    Converts XYZData structure to PotentialSlices using FFT-based atomic
    positioning
- `bessel_kv`:
    Computes the modified Bessel function of the second kind K_v(x)

Internal Functions
------------------
These functions are not exported and are used internally by the module.

- `_slice_atoms`:
    Partitions atoms into slices along the z-axis and sorts them by slice number
- `_compute_min_repeats`:
    Determines the minimum number of repeats needed to cover a given
    threshold distance
- `_expand_periodic_images`:
    Expands periodic images of a crystal structure to cover a given
    threshold distance
"""

import time

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Real, jaxtyped

from .electron_types import (PotentialSlices, XYZData, make_potential_slices,
                             scalar_float, scalar_int, scalar_numeric)
from .preprocessing import kirkland_potentials

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
    using pure JAX operations. Handles both 2D single images and 3D image
    stacks.

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
        lower_bound: Float[Array, ""] = jnp.percentile(flattened, p1)
        upper_bound: Float[Array, ""] = jnp.percentile(flattened, p2)
        clipped_image: Float[Array, "H W"] = jnp.clip(image, lower_bound, upper_bound)
        range_val: Float[Array, ""] = upper_bound - lower_bound
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
@jax.jit
def bessel_kv(v: scalar_float, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Description
    -----------
    Computes the modified Bessel function of the second kind
    K_v(x) for real order v >= 0 and x > 0,
    using a numerically stable and differentiable
    JAX-compatible approximation.

    Parameters
    ----------
    - `v` (scalar_float):
        Order of the Bessel function (v >= 0)
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
    - Uses series expansion for small x (x <= 2.0) and asymptotic expansion
      for large x
    - For non-integer v, uses the reflection formula:
      K_v = π/(2sin(πv)) * (I_{-v} - I_v)
    - For integer v, uses specialized series expansions and recurrence relations
    - Special exact formula for v = 0.5: K_{1/2}(x) = sqrt(π/(2x)) * exp(-x)
    - The transition point between small and large x approximations is set
      at x = 2.0
    """
    v: Float[Array, ""] = jnp.asarray(v)
    x: Float[Array, "..."] = jnp.asarray(x)
    dtype: jnp.dtype = x.dtype

    def iv_series(
        v_order: scalar_float, x_val: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        """Compute I_v(x) using series expansion"""
        x_half: Float[Array, "..."] = x_val / 2.0
        x_half_v: Float[Array, "..."] = jnp.power(x_half, v_order)
        x2_quarter: Float[Array, "..."] = (x_val * x_val) / 4.0

        max_terms: int = 20
        k_arr: Float[Array, "20"] = jnp.arange(max_terms, dtype=dtype)

        gamma_v_plus_1: Float[Array, ""] = jax.scipy.special.gamma(v_order + 1)
        gamma_terms: Float[Array, "20"] = jax.scipy.special.gamma(k_arr + v_order + 1)
        factorial_terms: Float[Array, "20"] = jax.scipy.special.factorial(k_arr)

        powers: Float[Array, "... 20"] = jnp.power(x2_quarter[..., jnp.newaxis], k_arr)
        series_terms: Float[Array, "... 20"] = powers / (
            factorial_terms * gamma_terms / gamma_v_plus_1
        )

        result: Float[Array, "..."] = (
            x_half_v / gamma_v_plus_1 * jnp.sum(series_terms, axis=-1)
        )
        return result

    def kv_small(v: scalar_float, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Series expansion for small x using the relation with I_v"""
        v_int: Float[Array, ""] = jnp.round(v)
        is_integer: Bool[Array, ""] = jnp.abs(v - v_int) < 1e-10

        def non_integer_case():
            iv_pos: Float[Array, "..."] = iv_series(v, x)
            iv_neg: Float[Array, "..."] = iv_series(-v, x)
            sin_piv: Float[Array, ""] = jnp.sin(jnp.pi * v)
            pi_over_2sin: Float[Array, ""] = jnp.pi / (2.0 * sin_piv)
            iv_diff: Float[Array, "..."] = iv_neg - iv_pos
            result: Float[Array, "..."] = jnp.where(
                jnp.abs(sin_piv) > 1e-10, pi_over_2sin * iv_diff, 0.0
            )
            return result

        def integer_case():
            n: Int[Array, ""] = jnp.abs(v_int).astype(jnp.int32)

            def k0_series():
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
                powers: Float[Array, "... 7"] = jnp.power(
                    x2[..., jnp.newaxis], jnp.arange(7)
                )
                poly: Float[Array, "..."] = jnp.sum(coeffs * powers, axis=-1)
                log_term: Float[Array, "..."] = -jnp.log(x / 2.0) * i0
                result: Float[Array, "..."] = log_term + poly
                return result

            def kn_series():
                k0: Float[Array, "..."] = k0_series()

                i1: Float[Array, "..."] = jax.scipy.special.i1(x)
                k1_coeffs: Float[Array, "5"] = jnp.array(
                    [1.0, -0.5, 0.0625, -0.03125, 0.0234375], dtype=dtype
                )
                x2: Float[Array, "..."] = (x * x) / 4.0
                k1_powers: Float[Array, "... 5"] = jnp.power(
                    x2[..., jnp.newaxis], jnp.arange(5)
                )
                k1_poly: Float[Array, "..."] = jnp.sum(k1_coeffs * k1_powers, axis=-1)
                log_i1_term: Float[Array, "..."] = -jnp.log(x / 2.0) * i1
                k1: Float[Array, "..."] = log_i1_term + k1_poly / x

                def recurrence_step(carry, i):
                    k_prev2, k_prev1 = carry
                    two_i_over_x: Float[Array, "..."] = 2.0 * i / x
                    k_curr: Float[Array, "..."] = two_i_over_x * k_prev1 + k_prev2
                    return (k_prev1, k_curr), k_curr

                def compute_kn():
                    init = (k0, k1)
                    max_n = 20  # Maximum order we support
                    indices = jnp.arange(1, max_n, dtype=jnp.float32)

                    def masked_step(carry, i):
                        k_prev2, k_prev1 = carry
                        mask = i < n
                        two_i_over_x: Float[Array, "..."] = 2.0 * i / x
                        k_curr: Float[Array, "..."] = two_i_over_x * k_prev1 + k_prev2
                        # Only update if within our target range
                        k_curr = jnp.where(mask, k_curr, k_prev1)
                        return (k_prev1, k_curr), k_curr

                    carry, k_vals = jax.lax.scan(masked_step, init, indices)
                    final_k: Float[Array, "..."] = carry[1]
                    return final_k

                kn_result: Float[Array, "..."] = jnp.where(
                    n == 0, k0, jnp.where(n == 1, k1, compute_kn())
                )
                return kn_result

            pos_v_result: Float[Array, "..."] = jnp.where(
                v >= 0, kn_series(), kn_series()
            )
            return pos_v_result

        small_x_result: Float[Array, "..."] = jnp.where(
            is_integer, integer_case(), non_integer_case()
        )
        return small_x_result

    def kv_large(v: scalar_float, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Asymptotic expansion for large x"""
        sqrt_term: Float[Array, "..."] = jnp.sqrt(jnp.pi / (2.0 * x))
        exp_term: Float[Array, "..."] = jnp.exp(-x)

        v2: Float[Array, ""] = v * v
        four_v2: Float[Array, ""] = 4.0 * v2
        a0: Float[Array, ""] = 1.0
        a1: Float[Array, ""] = (four_v2 - 1.0) / 8.0
        a2: Float[Array, ""] = (four_v2 - 1.0) * (four_v2 - 9.0) / (2.0 * 64.0)
        a3: Float[Array, ""] = (
            (four_v2 - 1.0) * (four_v2 - 9.0) * (four_v2 - 25.0) / (6.0 * 512.0)
        )
        a4: Float[Array, ""] = (
            (four_v2 - 1.0)
            * (four_v2 - 9.0)
            * (four_v2 - 25.0)
            * (four_v2 - 49.0)
            / (24.0 * 4096.0)
        )

        z: Float[Array, "..."] = 1.0 / x
        poly: Float[Array, "..."] = a0 + z * (a1 + z * (a2 + z * (a3 + z * a4)))

        large_x_result: Float[Array, "..."] = sqrt_term * exp_term * poly
        return large_x_result

    def k_half(x: Float[Array, "..."]) -> Float[Array, "..."]:
        sqrt_pi_over_2x: Float[Array, "..."] = jnp.sqrt(jnp.pi / (2.0 * x))
        exp_neg_x: Float[Array, "..."] = jnp.exp(-x)
        k_half_result: Float[Array, "..."] = sqrt_pi_over_2x * exp_neg_x
        return k_half_result

    small_x_vals: Float[Array, "..."] = kv_small(v, x)
    large_x_vals: Float[Array, "..."] = kv_large(v, x)
    general_result: Float[Array, "..."] = jnp.where(
        x <= 2.0, small_x_vals, large_x_vals
    )

    k_half_vals: Float[Array, "..."] = k_half(x)
    is_half: Bool[Array, ""] = jnp.abs(v - 0.5) < 1e-10
    final_result: Float[Array, "..."] = jnp.where(is_half, k_half_vals, general_result)

    return final_result


@jaxtyped(typechecker=beartype)
def single_atom_potential(
    atom_no: scalar_int,
    pixel_size: scalar_float,
    grid_shape: Optional[Tuple[scalar_int, scalar_int]] = None,
    center_coords: Optional[Float[Array, "2"]] = None,
    supersampling: Optional[scalar_int] = 4,
    potential_extent: Optional[scalar_float] = 4.0,
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Calculate the projected potential of a single atom using Kirkland
    scattering factors.
    The potential can be centered at arbitrary coordinates within a custom grid.

    Parameters
    ----------
    - `atom_no` (scalar_int):
        Atomic number of the atom whose potential is being calculated
    - `pixel_size` (scalar_float):
        Real space pixel size in Ångstroms
    - `grid_shape` (Tuple[scalar_int, scalar_int], optional):
        Shape of the output grid (height, width). If None, calculated from
        potential_extent
    - `center_coords` (Float[Array, "2"], optional):
        (x, y) coordinates in Ångstroms where atom should be centered.
        If None, centers at grid center
    - `supersampling` (scalar_int, optional):
        Supersampling factor for increased accuracy. Default is 16
    - `potential_extent` (scalar_float, optional):
        Distance in Ångstroms from atom center to calculate potential.
        Default is 4.0 Å

    Returns
    -------
    - `potential` (Float[Array, "h w"]):
        Projected potential matrix with atom centered at specified coordinates

    Flow
    ----
    - Initialize physical constants:
        - a0 = 0.5292 Å (Bohr radius)
        - ek = 14.4 eV·Å (electron charge squared divided by 4πε₀)
        - Calculate prefactors for Bessel (term1) and Gaussian (term2)
          contributions
    - Load Kirkland scattering parameters:
        - Extract 12 parameters for the specified atom from preloaded
          Kirkland data
        - Parameters alternate between amplitudes and reciprocal space widths
    - Determine grid dimensions:
        - If grid_shape provided: use it directly, multiplied by supersampling
        - If grid_shape is None: calculate from potential_extent to ensure
          full coverage
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
    atom_idx: Int[Array, ""] = (atom_no - 1).astype(jnp.int32)
    kirk_params: Float[Array, "12"] = jax.lax.dynamic_slice(
        kirkland_array, (atom_idx, jnp.int32(0)), (1, 12)
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
    # Add small epsilon to avoid r=0 which causes NaN in Bessel K_0(0)
    epsilon = 1e-10
    r: Float[Array, "h w"] = jnp.sqrt(
        (xa - center_x) ** 2 + (ya - center_y) ** 2 + epsilon
    )
    bessel_term1: Float[Array, "h w"] = kirk_params[0] * bessel_kv(
        0.0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[1]) * r
    )
    bessel_term2: Float[Array, "h w"] = kirk_params[2] * bessel_kv(
        0.0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[3]) * r
    )
    bessel_term3: Float[Array, "h w"] = kirk_params[4] * bessel_kv(
        0.0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[5]) * r
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


# JIT compile single_atom_potential with static arguments
single_atom_potential = jax.jit(
    single_atom_potential, static_argnames=["grid_shape", "supersampling"]
)


@jaxtyped(typechecker=beartype)
def _compute_min_repeats(
    cell: Float[Array, "3 3"], threshold_nm: scalar_float
) -> Tuple[int, int, int]:
    """
    Description
    -----------
    Internal function to compute the minimal number of unit cell repeats
    along each lattice vector direction such that the resulting supercell
    dimensions exceed a specified threshold distance. This is used to ensure
    periodic images are included
    for accurate potential calculations.

    Parameters
    ----------
    - `cell` (Float[Array, "3 3"]):
        Real-space unit cell matrix where rows represent lattice vectors
        a1, a2, a3
    - `threshold_nm` (scalar_float):
        Minimum required length in nanometers for the supercell along each
        direction

    Returns
    -------
    - `n_repeats` (Tuple[int, int, int]):
        Number of repeats (nx, ny, nz) needed along each lattice vector
        direction

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
    lengths: Float[Array, "3"] = jnp.linalg.norm(cell, axis=1)
    repeat_ratios: Float[Array, "3"] = threshold_nm / lengths
    n_repeats_float: Float[Array, "3"] = jnp.ceil(repeat_ratios)
    n_repeats: Int[Array, "3"] = n_repeats_float.astype(int)
    n_repeats_tuple: Tuple[int, int, int] = tuple(n_repeats)
    return n_repeats_tuple


@jaxtyped(typechecker=beartype)
def _expand_periodic_images(
    coords: Float[Array, "N 4"], cell: Float[Array, "3 3"], threshold_nm: scalar_float
) -> Tuple[Float[Array, "M 4"], Tuple[int, int, int]]:
    """
    Expand coordinates in all directions just enough to exceed (twice of) a
    minimum
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
    nz = 0

    i: Int[Array, "2nx+1"] = jnp.arange(-nx, nx + 1)
    j: Int[Array, "2ny+1"] = jnp.arange(-ny, ny + 1)
    k: Int[Array, "2nz+1"] = jnp.arange(-nz, nz + 1)

    ii: Int[Array, "2nx+1 2ny+1 2nz+1"]
    jj: Int[Array, "2nx+1 2ny+1 2nz+1"]
    kk: Int[Array, "2nx+1 2ny+1 2nz+1"]
    ii, jj, kk = jnp.meshgrid(i, j, k, indexing="ij")
    shifts: Int[Array, "M 3"] = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1)
    shift_vectors: Float[Array, "M 3"] = shifts @ cell

    def shift_all_atoms(shift_vec: Float[Array, "3"]) -> Float[Array, "N 4"]:
        atom_numbers: Float[Array, "N 1"] = coords[:, 0:1]
        positions: Float[Array, "N 3"] = coords[:, 1:4]
        shifted_positions: Float[Array, "N 3"] = positions + shift_vec
        new_coords: Float[Array, "N 4"] = jnp.hstack((atom_numbers, shifted_positions))
        return new_coords

    expanded_coords: Float[Array, "M N 4"] = jax.vmap(shift_all_atoms)(shift_vectors)
    final_coords: Float[Array, "M 4"] = expanded_coords.reshape(-1, 4)
    repeat_counts: Tuple[int, int, int] = (nx, ny, nz)
    return final_coords, repeat_counts


@jaxtyped(typechecker=beartype)
def _slice_atoms(
    coords: Float[Array, "N 3"],
    atom_numbers: Int[Array, "N"],
    slice_thickness: scalar_numeric,
) -> Float[Array, "N 4"]:
    """
    Description
    -----------
    Partitions atoms into slices along the z-axis and returns them sorted by
    slice number. This internal function is used to organize atomic positions
    for slice-by-slice
    potential calculations in electron microscopy simulations.

    Parameters
    ----------
    - `coords` (Float[Array, "N 3"]):
        Atomic positions with shape (N, 3) where columns represent x, y, z
        coordinates
        in Angstroms
    - `atom_numbers` (Int[Array, "N"]):
        Atomic numbers for each of the N atoms, used to identify element types
    - `slice_thickness` (scalar_numeric):
        Thickness of each slice in Angstroms. Can be float, int, or
        0-dimensional
        JAX array

    Returns
    -------
    - `sorted_atoms` (Float[Array, "N 4"]):
        Array with shape (N, 4) containing [x, y, slice_num, atom_number]
        for each atom,
        sorted by ascending slice number. Slice numbers start from 0.

    Flow
    ----
    - Extract z-coordinates and find minimum and maximum z values
    - Calculate slice index for each atom based on its z-position:
        - Atoms are assigned to slices using floor division:
          (z - z_min) / slice_thickness
        - This ensures atoms at z_min are in slice 0
    - Construct output array with x, y positions, slice numbers, and atom
      numbers
    - Sort atoms by their slice indices to group atoms within the same slice
    - Return the sorted array for efficient slice-by-slice processing

    Notes
    -----
    - The number of slices is implicitly ceil((z_max - z_min) / slice_thickness)
    - Atoms exactly at slice boundaries are assigned to the lower slice
    - All arrays are JAX arrays for compatibility with JIT compilation
    """
    z_coords: Float[Array, "N"] = coords[:, 2]
    z_min: Float[Array, ""] = jnp.min(z_coords)
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
    slice_thickness: Optional[scalar_float] = 1.0,
    repeats: Optional[Int[Array, "3"]] = jnp.array([1, 1, 1]),
    padding: Optional[scalar_float] = 4.0,
    supersampling: Optional[scalar_int] = 4,
) -> PotentialSlices:
    """
    Description
    -----------
    Converts XYZData structure to PotentialSlices by calculating atomic
    potentials
    and assembling them into slices using FFT shifts for precise positioning.

    Parameters
    ----------
    - `xyz_data` (XYZData):
        Input structure containing atomic positions and numbers
    - `pixel_size` (scalar_float):
        Size of each pixel in Angstroms (becomes calib in PotentialSlices)
    - `slice_thickness` (scalar_float, optional):
        Thickness of each slice in Angstroms. Default is 1.0
    - `repeats` (Int[Array, "3"], optional):
        Number of unit cell repeats in [x, y, z] directions. Default is
        [1, 1, 1], which means no repeating.
        Requires xyz_data.lattice to be provided for repeating the structure.
    - `padding` (scalar_float, optional):
        Padding in Angstroms added to all sides. Default is 4.0

    Returns
    -------
    - `potential_slices` (PotentialSlices):
        Sliced potentials with wraparound artifacts removed

    Flow
    ----
    - Extract atomic positions, atomic numbers, and lattice from the input
      XYZData structure
    - If repeats > [1,1,1], tile the structure using the lattice vectors to
      create a supercell
    - Partition atoms into slices along the z-axis using _slice_atoms,
      assigning each atom to a slice based on its z-coordinate and the
      specified slice_thickness
    - Compute the minimum and maximum x and y coordinates of all atoms, add
      padding, and determine the grid size (width, height) in pixels
    - Identify all unique atomic numbers present in the structure
    - For each unique atomic number, precompute a single-atom projected
      potential using single_atom_potential (centered at the origin, with the
      correct grid size and pixel size)
    - Build a lookup array to map atomic numbers to their corresponding
      precomputed potential indices
    - For each slice:
        - Initialize a zero grid for the slice
        - For each atom in the slice:
            - Place the corresponding atomic potential at the atom's (x, y)
              position using FFT-based shifting for subpixel accuracy
            - Accumulate all atomic contributions for the slice
    - Remove the extra padding from the edges of the grid to obtain the
      final region of interest
    - Return a PotentialSlices object containing the 3D array of potential
      slices, the slice thickness, and the pixel size
    """
    positions: Float[Array, "N 3"] = xyz_data.positions
    atomic_numbers: Int[Array, "N"] = xyz_data.atomic_numbers
    lattice: Float[Array, "3 3"] = xyz_data.lattice

    def apply_repeats_with_lattice(
        positions: Float[Array, "N 3"],
        atomic_numbers: Int[Array, "N"],
        lattice: Float[Array, "3 3"],
    ) -> Tuple[Float[Array, "M 3"], Int[Array, "M"]]:
        """Apply periodic repeats to positions and atomic numbers."""
        nx: Int[Array, ""] = repeats[0]
        ny: Int[Array, ""] = repeats[1]
        nz: Int[Array, ""] = repeats[2]

        max_n: int = 20
        ix: Int[Array, "max_n"] = jnp.arange(max_n)
        iy: Int[Array, "max_n"] = jnp.arange(max_n)
        iz: Int[Array, "max_n"] = jnp.arange(max_n)

        mask_x: Bool[Array, "max_n"] = ix < nx
        mask_y: Bool[Array, "max_n"] = iy < ny
        mask_z: Bool[Array, "max_n"] = iz < nz

        ixx: Int[Array, "max_n max_n max_n"]
        iyy: Int[Array, "max_n max_n max_n"]
        izz: Int[Array, "max_n max_n max_n"]
        ixx, iyy, izz = jnp.meshgrid(ix, iy, iz, indexing="ij")

        mask_x_expanded: Bool[Array, "max_n 1 1"] = mask_x[:, None, None]
        mask_y_expanded: Bool[Array, "1 max_n 1"] = mask_y[None, :, None]
        mask_z_expanded: Bool[Array, "1 1 max_n"] = mask_z[None, None, :]
        mask_3d: Bool[Array, "max_n max_n max_n"] = (
            mask_x_expanded & mask_y_expanded & mask_z_expanded
        )

        ixx_flat: Int[Array, "max_n^3"] = ixx.ravel()
        iyy_flat: Int[Array, "max_n^3"] = iyy.ravel()
        izz_flat: Int[Array, "max_n^3"] = izz.ravel()
        shift_indices: Int[Array, "max_n^3 3"] = jnp.stack(
            [ixx_flat, iyy_flat, izz_flat], axis=-1
        )
        mask_flat: Bool[Array, "max_n^3"] = mask_3d.ravel()

        mask_float: Float[Array, "max_n^3"] = mask_flat.astype(jnp.float32)

        shift_indices_float: Float[Array, "max_n^3 3"] = shift_indices.astype(
            jnp.float32
        )
        mask_expanded: Float[Array, "max_n^3 1"] = mask_float[:, None]
        shift_indices_masked: Float[Array, "max_n^3 3"] = (
            shift_indices_float * mask_expanded
        )

        shift_vectors: Float[Array, "max_n^3 3"] = shift_indices_masked @ lattice

        n_atoms: int = positions.shape[0]
        max_shifts: int = max_n * max_n * max_n

        positions_expanded: Float[Array, "1 N 3"] = positions[None, :, :]
        positions_broadcast: Float[Array, "max_n^3 N 3"] = jnp.broadcast_to(
            positions_expanded, (max_shifts, n_atoms, 3)
        )
        shift_vectors_expanded: Float[Array, "max_n^3 1 3"] = shift_vectors[:, None, :]
        shifts_broadcast: Float[Array, "max_n^3 N 3"] = jnp.broadcast_to(
            shift_vectors_expanded, (max_shifts, n_atoms, 3)
        )

        repeated_positions: Float[Array, "max_n^3 N 3"] = (
            positions_broadcast + shifts_broadcast
        )

        total_atoms: int = max_shifts * n_atoms
        repeated_positions_flat: Float[Array, "max_n^3*N 3"] = (
            repeated_positions.reshape(total_atoms, 3)
        )

        atom_mask: Bool[Array, "max_n^3*N"] = jnp.repeat(mask_flat, n_atoms)
        atom_mask_float: Float[Array, "max_n^3*N"] = atom_mask.astype(jnp.float32)

        atom_mask_expanded: Float[Array, "max_n^3*N 1"] = atom_mask_float[:, None]
        repeated_positions_masked: Float[Array, "max_n^3*N 3"] = (
            repeated_positions_flat * atom_mask_expanded
        )

        atomic_numbers_tiled: Int[Array, "max_n^3*N"] = jnp.tile(
            atomic_numbers, max_shifts
        )
        atom_mask_int: Int[Array, "max_n^3*N"] = atom_mask.astype(jnp.int32)
        repeated_atomic_numbers_masked: Int[Array, "max_n^3*N"] = (
            atomic_numbers_tiled * atom_mask_int
        )

        return (repeated_positions_masked, repeated_atomic_numbers_masked)

    def return_unchanged(
        positions: Float[Array, "N 3"],
        atomic_numbers: Int[Array, "N"],
        lattice: Float[Array, "3 3"],
    ) -> Tuple[Float[Array, "max_n^3*N 3"], Int[Array, "max_n^3*N"]]:
        """Return positions and atomic numbers unchanged but in the same shape as apply_repeats."""
        n_atoms: int = positions.shape[0]
        max_n: int = 20
        max_shifts: int = max_n * max_n * max_n
        max_total: int = max_shifts * n_atoms

        positions_padded: Float[Array, "max_n^3*N 3"] = jnp.zeros((max_total, 3))
        atomic_numbers_padded: Int[Array, "max_n^3*N"] = jnp.zeros(
            max_total, dtype=jnp.int32
        )

        positions_padded = positions_padded.at[:n_atoms].set(positions)
        atomic_numbers_padded = atomic_numbers_padded.at[:n_atoms].set(atomic_numbers)

        return (positions_padded, atomic_numbers_padded)

    positions, atomic_numbers = jax.lax.cond(
        jnp.any(repeats > 1),
        lambda pos, an, lat: apply_repeats_with_lattice(pos, an, lat),
        lambda pos, an, lat: return_unchanged(pos, an, lat),
        positions,
        atomic_numbers,
        lattice,
    )

    sliced_atoms: Float[Array, "N 4"] = _slice_atoms(
        coords=positions,
        atom_numbers=atomic_numbers,
        slice_thickness=slice_thickness,
    )
    x_coords: Float[Array, "N"] = sliced_atoms[:, 0]
    y_coords: Float[Array, "N"] = sliced_atoms[:, 1]
    slice_indices: Int[Array, "N"] = sliced_atoms[:, 2].astype(jnp.int32)
    atom_nums: Int[Array, "N"] = sliced_atoms[:, 3].astype(jnp.int32)
    x_coords_min: Float[Array, ""] = jnp.min(x_coords)
    x_coords_max: Float[Array, ""] = jnp.max(x_coords)
    y_coords_min: Float[Array, ""] = jnp.min(y_coords)
    y_coords_max: Float[Array, ""] = jnp.max(y_coords)
    x_min: Float[Array, ""] = x_coords_min - padding
    x_max: Float[Array, ""] = x_coords_max + padding
    y_min: Float[Array, ""] = y_coords_min - padding
    y_max: Float[Array, ""] = y_coords_max + padding
    x_range: Float[Array, ""] = x_max - x_min
    y_range: Float[Array, ""] = y_max - y_min
    width_float: Float[Array, ""] = jnp.ceil(x_range / pixel_size)
    height_float: Float[Array, ""] = jnp.ceil(y_range / pixel_size)
    width: Int[Array, ""] = width_float.astype(jnp.int32)
    height: Int[Array, ""] = height_float.astype(jnp.int32)
    # Use size parameter for JIT compatibility - max 118 elements in periodic table
    unique_atoms: Int[Array, "118"] = jnp.unique(atom_nums, size=118, fill_value=-1)
    # Create mask for valid (non-fill) atoms
    valid_mask = unique_atoms >= 0
    n_unique_atoms = jnp.sum(valid_mask)

    # Convert height and width to Python integers for use in the function
    height_int = int(height)
    width_int = int(width)

    # Create a specialized version of single_atom_potential for this specific grid size
    @jax.jit
    def calc_single_potential_fixed_grid(
        atom_no: scalar_int, is_valid: Bool
    ) -> Float[Array, "h w"]:
        # Calculate potential only for valid atoms, return zeros for padding
        potential = single_atom_potential(
            atom_no=atom_no,
            pixel_size=pixel_size,
            grid_shape=(height_int, width_int),
            center_coords=jnp.array([0.0, 0.0]),
            supersampling=supersampling,
            potential_extent=4.0,
        )
        # Return potential if valid, zeros otherwise
        return jnp.where(is_valid, potential, jnp.zeros((height_int, width_int)))

    # Calculate potentials for all 118 slots (padded with zeros)
    atomic_potentials: Float[Array, "118 h w"] = jax.vmap(
        calc_single_potential_fixed_grid
    )(unique_atoms, valid_mask)
    atom_to_idx_array: Int[Array, "119"] = jnp.full(119, -1, dtype=jnp.int32)

    # Create mapping for only the unique atoms we actually have
    # Use where to only set indices for valid atoms
    indices = jnp.arange(118, dtype=jnp.int32)
    atom_indices = jnp.where(valid_mask, unique_atoms, -1)

    # Build the mapping array using a scan
    def update_mapping(carry, idx_atom):
        mapping_array = carry
        idx, atom = idx_atom
        # Only update if atom is valid (>= 0)
        mapping_array = jnp.where(
            atom >= 0, mapping_array.at[atom].set(idx), mapping_array
        )
        return mapping_array, None

    atom_to_idx_array, _ = jax.lax.scan(
        update_mapping, atom_to_idx_array, (indices, atom_indices)
    )
    max_slice_idx: Int[Array, ""] = jnp.max(slice_indices).astype(jnp.int32)
    n_slices: Int[Array, ""] = max_slice_idx + 1
    all_slices: Float[Array, "h w n_slices"] = jnp.zeros(
        (height, width, n_slices), dtype=jnp.float32
    )
    ky: Float[Array, "h 1"] = jnp.fft.fftfreq(height, d=1.0).reshape(-1, 1)
    kx: Float[Array, "1 w"] = jnp.fft.fftfreq(width, d=1.0).reshape(1, -1)

    def process_single_slice(slice_idx: int) -> Float[Array, "h w"]:
        slice_potential: Float[Array, "h w"] = jnp.zeros(
            (height, width), dtype=jnp.float32
        )
        center_x: float = width / 2.0
        center_y: float = height / 2.0

        def add_atom_contribution(
            carry: Float[Array, "h w"],
            atom_data: Tuple[scalar_float, scalar_float, scalar_int, scalar_int],
        ) -> Tuple[Float[Array, "h w"], None]:
            slice_pot: Float[Array, "h w"] = carry
            x: scalar_float
            y: scalar_float
            atom_no: scalar_int
            atom_slice_idx: scalar_int
            x, y, atom_no, atom_slice_idx = atom_data

            # Only add contribution if atom is in current slice
            x_offset: scalar_float = x - x_min
            y_offset: scalar_float = y - y_min
            pixel_x: scalar_float = x_offset / pixel_size
            pixel_y: scalar_float = y_offset / pixel_size
            shift_x: scalar_float = pixel_x - center_x
            shift_y: scalar_float = pixel_y - center_y

            atom_idx: int = atom_to_idx_array[atom_no]
            atom_pot: Float[Array, "h w"] = atomic_potentials[atom_idx]
            kx_sx: Float[Array, "h w"] = kx * shift_x
            ky_sy: Float[Array, "h w"] = ky * shift_y
            phase_arg: Float[Array, "h w"] = kx_sx + ky_sy
            phase: Complex[Array, "h w"] = jnp.exp(2j * jnp.pi * phase_arg)
            atom_pot_fft: Complex[Array, "h w"] = jnp.fft.fft2(atom_pot)
            shifted_fft: Complex[Array, "h w"] = atom_pot_fft * phase
            shifted_pot: Float[Array, "h w"] = jnp.real(jnp.fft.ifft2(shifted_fft))

            # Add contribution only if atom is in current slice
            contribution: Float[Array, "h w"] = jnp.where(
                atom_slice_idx == slice_idx, shifted_pot, jnp.zeros_like(shifted_pot)
            ).astype(jnp.float32)
            updated_pot: Float[Array, "h w"] = (slice_pot + contribution).astype(
                jnp.float32
            )
            return updated_pot, None

        slice_potential, _ = jax.lax.scan(
            add_atom_contribution,
            slice_potential,
            (x_coords, y_coords, atom_nums, slice_indices),
        )
        return slice_potential

    slice_indices_array: Int[Array, "n_slices"] = jnp.arange(n_slices)
    processed_slices: Float[Array, "n_slices h w"] = jax.vmap(process_single_slice)(
        slice_indices_array
    )
    all_slices: Float[Array, "h w n_slices"] = processed_slices.transpose(1, 2, 0)
    padding_pixels_float: Float[Array, ""] = jnp.round(padding / pixel_size)
    crop_pixels: int = int(padding_pixels_float)
    cropped_slices: Float[Array, "h_crop w_crop n_slices"] = all_slices[
        crop_pixels:-crop_pixels, crop_pixels:-crop_pixels, :
    ]
    pot_slices: PotentialSlices = make_potential_slices(
        slices=cropped_slices,
        slice_thickness=slice_thickness,
        calib=pixel_size,
    )
    return pot_slices
