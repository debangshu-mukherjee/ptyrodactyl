"""Atomic potential calculations for electron microscopy.

Extended Summary
----------------
Functions for calculating projected atomic potentials using
Kirkland scattering factors and assembling them into potential
slices for multislice simulations. Supports periodic boundary
handling and FFT-based sub-pixel atom positioning.

Routine Listings
----------------
:func:`contrast_stretch`
    Rescale intensity values between specified percentiles.
:func:`_bessel_iv_series`
    Series expansion for modified Bessel function I_v(x).
:func:`_bessel_k0_series`
    Series expansion for K_0(x).
:func:`_bessel_kn_recurrence`
    Recurrence relation for K_n(x).
:func:`_bessel_kv_small_non_integer`
    K_v(x) for small x and non-integer v.
:func:`_bessel_kv_small_integer`
    K_v(x) for small x and integer v.
:func:`_bessel_kv_large`
    Asymptotic expansion for K_v(x) at large x.
:func:`_bessel_k_half`
    Exact formula for K_{1/2}(x).
:func:`bessel_kv`
    Modified Bessel function of the second kind K_v(x).
:func:`_calculate_bessel_contributions`
    Bessel contributions to the atomic potential.
:func:`_calculate_gaussian_contributions`
    Gaussian contributions to the atomic potential.
:func:`_downsample_potential`
    Downsample supersampled potential to target resolution.
:func:`single_atom_potential`
    Projected potential of a single atom via Kirkland
    parameterization.
:func:`_slice_atoms`
    Partition atoms into slices along the z-axis.
:func:`_compute_grid_dimensions`
    Compute grid dimensions from coordinate ranges.
:func:`_process_all_slices`
    Assemble all potential slices from atomic contributions.
:func:`_build_shift_masks`
    Build shift indices and masks for periodic repeats.
:func:`_tile_positions_with_shifts`
    Tile positions and atomic numbers with shift vectors.
:func:`_apply_repeats_or_return`
    Apply periodic repeats or return unchanged positions.
:func:`_build_potential_lookup`
    Build lookup table for atomic potentials.
:func:`kirkland_potentials_crystal`
    Convert :class:`~ptyrodactyl.tools.CrystalData` to
    :class:`~ptyrodactyl.tools.PotentialSlices`.

Notes
-----
Internal functions (prefixed with underscore) handle slice
partitioning, periodic image expansion, Bessel function
calculations, and potential lookup tables.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Real, jaxtyped

from ptyrodactyl.tools import (
    CrystalData,
    PotentialSlices,
    ScalarFloat,
    ScalarInt,
    ScalarNumeric,
    make_potential_slices,
)

from .preprocessing import kirkland_potentials

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
@jax.jit
def contrast_stretch(
    series: Union[Float[Array, " H W"], Float[Array, " N H W"]],
    p1: float,
    p2: float,
) -> Union[Float[Array, " H W"], Float[Array, " N H W"]]:
    """Rescale image intensity between specified percentiles.

    Extended Summary
    ----------------
    Clips pixel values to the ``[p1, p2]`` percentile range
    and linearly rescales to ``[0, 1]``. Handles both single
    images and stacks via ``jax.vmap``.

    Implementation Logic
    --------------------
    1. **Expand dims** -- Promote 2D to 3D if needed.
    2. **Per-image percentiles** -- Compute lower/upper
       bounds from ``jnp.percentile``.
    3. **Clip and rescale** -- Linear map to ``[0, 1]``.
    4. **Restore shape** -- Squeeze back to 2D if input
       was 2D.

    Parameters
    ----------
    series : Float[Array, " H W"] | Float[Array, " N H W"]
        Input image or image stack.
    p1 : float
        Lower percentile (0--100).
    p2 : float
        Upper percentile (0--100).

    Returns
    -------
    final_result : Float[Array, " H W"] | Float[Array, " N H W"]
        Rescaled image(s) with same shape as input.
    """
    original_shape: Tuple[int, ...] = series.shape
    is_2d_image: int = 2
    series_reshaped: Float[Array, " N H W"] = jnp.where(
        len(original_shape) == is_2d_image, series[jnp.newaxis, :, :], series
    )

    def _rescale_single_image(
        image: Float[Array, " H W"],
    ) -> Float[Array, " H W"]:
        """Rescale one image via percentile-based stretching.

        Parameters
        ----------
        image : Float[Array, " H W"]
            Single image to rescale.

        Returns
        -------
        rescaled_image : Float[Array, " H W"]
            Image rescaled to ``[0, 1]``.
        """
        flattened: Float[Array, " HW"] = image.flatten()
        lower_bound: Float[Array, ""] = jnp.percentile(flattened, p1)
        upper_bound: Float[Array, ""] = jnp.percentile(flattened, p2)
        clipped_image: Float[Array, " H W"] = jnp.clip(
            image, lower_bound, upper_bound
        )
        range_val: Float[Array, ""] = upper_bound - lower_bound
        rescaled_image: Float[Array, " H W"] = jnp.where(
            range_val > 0,
            (clipped_image - lower_bound) / range_val,
            clipped_image,
        )
        return rescaled_image

    transformed: Float[Array, " N H W"] = jax.vmap(_rescale_single_image)(
        series_reshaped
    )
    is_2d_image: int = 2
    final_result: Union[Float[Array, " H W"], Float[Array, " N H W"]] = (
        jnp.where(
            len(original_shape) == is_2d_image, transformed[0], transformed
        )
    )
    return final_result


def _bessel_iv_series(
    v_order: ScalarFloat, x_val: Float[Array, " ..."], dtype: jnp.dtype
) -> Float[Array, " ..."]:
    r"""Compute :math:`I_v(x)` via 20-term series expansion.

    Parameters
    ----------
    v_order : ScalarFloat
        Order of the Bessel function.
    x_val : Float[Array, " ..."]
        Positive real input values.
    dtype : jnp.dtype
        Data type for intermediate computation.

    Returns
    -------
    result : Float[Array, " ..."]
        Approximation of :math:`I_v(x)`.
    """
    x_half: Float[Array, " ..."] = x_val / 2.0
    x_half_v: Float[Array, " ..."] = jnp.power(x_half, v_order)
    x2_quarter: Float[Array, " ..."] = (x_val * x_val) / 4.0

    max_terms: int = 20
    k_arr: Float[Array, " 20"] = jnp.arange(max_terms, dtype=dtype)

    gamma_v_plus_1: Float[Array, ""] = jax.scipy.special.gamma(v_order + 1)
    gamma_terms: Float[Array, " 20"] = jax.scipy.special.gamma(
        k_arr + v_order + 1
    )
    factorial_terms: Float[Array, " 20"] = jax.scipy.special.factorial(k_arr)

    powers: Float[Array, " ... 20"] = jnp.power(
        x2_quarter[..., jnp.newaxis], k_arr
    )
    series_terms: Float[Array, " ... 20"] = powers / (
        factorial_terms * gamma_terms / gamma_v_plus_1
    )

    result: Float[Array, " ..."] = (
        x_half_v / gamma_v_plus_1 * jnp.sum(series_terms, axis=-1)
    )
    return result


def _bessel_k0_series(
    x: Float[Array, " ..."], dtype: jnp.dtype
) -> Float[Array, " ..."]:
    r"""Compute :math:`K_0(x)` via polynomial series expansion.

    Parameters
    ----------
    x : Float[Array, " ..."]
        Positive real input values.
    dtype : jnp.dtype
        Data type for coefficients.

    Returns
    -------
    result : Float[Array, " ..."]
        Approximation of :math:`K_0(x)`.
    """
    i0: Float[Array, " ..."] = jax.scipy.special.i0(x)
    coeffs: Float[Array, " 7"] = jnp.array(
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
    x2: Float[Array, " ..."] = (x * x) / 4.0
    powers: Float[Array, " ... 7"] = jnp.power(
        x2[..., jnp.newaxis], jnp.arange(7)
    )
    poly: Float[Array, " ..."] = jnp.sum(coeffs * powers, axis=-1)
    log_term: Float[Array, " ..."] = -jnp.log(x / 2.0) * i0
    result: Float[Array, " ..."] = log_term + poly
    return result


def _bessel_kn_recurrence(
    n: Int[Array, ""],
    x: Float[Array, " ..."],
    k0: Float[Array, " ..."],
    k1: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Compute :math:`K_n(x)` via forward recurrence.

    Parameters
    ----------
    n : Int[Array, ""]
        Target integer order.
    x : Float[Array, " ..."]
        Positive real input values.
    k0 : Float[Array, " ..."]
        Pre-computed :math:`K_0(x)`.
    k1 : Float[Array, " ..."]
        Pre-computed :math:`K_1(x)`.

    Returns
    -------
    kn_result : Float[Array, " ..."]
        :math:`K_n(x)` values.
    """

    def _compute_kn() -> Float[Array, " ..."]:
        """Forward recurrence from K_0 and K_1 up to K_n.

        Returns
        -------
        final_k : Float[Array, " ..."]
            K_n(x) computed via masked recurrence.
        """
        init = (k0, k1)
        max_n = 20
        indices = jnp.arange(1, max_n, dtype=jnp.float32)

        def masked_step(
            carry: Tuple[Float[Array, " ..."], Float[Array, " ..."]],
            i: Float[Array, ""],
        ) -> Tuple[
            Tuple[Float[Array, " ..."], Float[Array, " ..."]],
            Float[Array, " ..."],
        ]:
            """One step of Bessel recurrence with masking.

            Parameters
            ----------
            carry : tuple
                ``(k_prev2, k_prev1)`` from previous steps.
            i : Float[Array, ""]
                Current recurrence index.

            Returns
            -------
            tuple
                Updated ``(k_prev1, k_curr)`` and ``k_curr``.
            """
            k_prev2, k_prev1 = carry
            mask = i < n
            two_i_over_x: Float[Array, " ..."] = 2.0 * i / x
            k_curr: Float[Array, " ..."] = two_i_over_x * k_prev1 + k_prev2
            k_curr = jnp.where(mask, k_curr, k_prev1)
            return (k_prev1, k_curr), k_curr

        carry, k_vals = jax.lax.scan(masked_step, init, indices)
        final_k: Float[Array, " ..."] = carry[1]
        return final_k

    kn_result: Float[Array, " ..."] = jnp.where(
        n == 0, k0, jnp.where(n == 1, k1, _compute_kn())
    )
    return kn_result


def _bessel_kv_small_non_integer(
    v: ScalarFloat, x: Float[Array, " ..."], dtype: jnp.dtype
) -> Float[Array, " ..."]:
    r"""Compute :math:`K_v(x)` for small x and non-integer v.

    Uses the reflection formula
    :math:`K_v = \frac{\pi}{2\sin(\pi v)}(I_{-v} - I_v)`.

    Parameters
    ----------
    v : ScalarFloat
        Non-integer order.
    x : Float[Array, " ..."]
        Positive real input (small regime, x <= 2).
    dtype : jnp.dtype
        Data type for computation.

    Returns
    -------
    result : Float[Array, " ..."]
        Approximation of :math:`K_v(x)`.
    """
    error_bound: Float[Array, ""] = jnp.asarray(1e-10)
    iv_pos: Float[Array, " ..."] = _bessel_iv_series(v, x, dtype)
    iv_neg: Float[Array, " ..."] = _bessel_iv_series(-v, x, dtype)
    sin_piv: Float[Array, ""] = jnp.sin(jnp.pi * v)
    pi_over_2sin: Float[Array, ""] = jnp.pi / (2.0 * sin_piv)
    iv_diff: Float[Array, " ..."] = iv_neg - iv_pos
    result: Float[Array, " ..."] = jnp.where(
        jnp.abs(sin_piv) > error_bound, pi_over_2sin * iv_diff, 0.0
    )
    return result


def _bessel_kv_small_integer(
    v: Float[Array, ""], x: Float[Array, " ..."], dtype: jnp.dtype
) -> Float[Array, " ..."]:
    r"""Compute :math:`K_v(x)` for small x and integer v.

    Uses specialised series for :math:`K_0`, :math:`K_1`,
    and forward recurrence for higher integer orders.

    Parameters
    ----------
    v : Float[Array, ""]
        Order (must be close to an integer).
    x : Float[Array, " ..."]
        Positive real input (small regime, x <= 2).
    dtype : jnp.dtype
        Data type for computation.

    Returns
    -------
    pos_v_result : Float[Array, " ..."]
        Approximation of :math:`K_n(x)`.
    """
    v_int: Float[Array, ""] = jnp.round(v)
    n: Int[Array, ""] = jnp.abs(v_int).astype(jnp.int32)

    k0: Float[Array, " ..."] = _bessel_k0_series(x, dtype)

    i1: Float[Array, " ..."] = jax.scipy.special.i1(x)
    k1_coeffs: Float[Array, " 5"] = jnp.array(
        [1.0, -0.5, 0.0625, -0.03125, 0.0234375], dtype=dtype
    )
    x2: Float[Array, " ..."] = (x * x) / 4.0
    k1_powers: Float[Array, " ... 5"] = jnp.power(
        x2[..., jnp.newaxis], jnp.arange(5)
    )
    k1_poly: Float[Array, " ..."] = jnp.sum(k1_coeffs * k1_powers, axis=-1)
    log_i1_term: Float[Array, " ..."] = -jnp.log(x / 2.0) * i1
    k1: Float[Array, " ..."] = log_i1_term + k1_poly / x

    kn_result: Float[Array, " ..."] = _bessel_kn_recurrence(
        n, x, k0, k1
    )
    pos_v_result: Float[Array, " ..."] = jnp.where(
        v >= 0, kn_result, kn_result
    )
    return pos_v_result


def _bessel_kv_large(
    v: ScalarFloat, x: Float[Array, " ..."]
) -> Float[Array, " ..."]:
    r"""Asymptotic expansion for :math:`K_v(x)` at large x.

    Uses a 5-term asymptotic series valid for ``x > 2``.

    Parameters
    ----------
    v : ScalarFloat
        Order of the Bessel function.
    x : Float[Array, " ..."]
        Positive real input (large regime, x > 2).

    Returns
    -------
    large_x_result : Float[Array, " ..."]
        Asymptotic approximation of :math:`K_v(x)`.
    """
    sqrt_term: Float[Array, " ..."] = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_term: Float[Array, " ..."] = jnp.exp(-x)

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

    z: Float[Array, " ..."] = 1.0 / x
    poly: Float[Array, " ..."] = a0 + z * (a1 + z * (a2 + z * (a3 + z * a4)))

    large_x_result: Float[Array, " ..."] = sqrt_term * exp_term * poly
    return large_x_result


def _bessel_k_half(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Exact formula :math:`K_{1/2}(x)=\sqrt{\pi/(2x)}\,e^{-x}`.

    Parameters
    ----------
    x : Float[Array, " ..."]
        Positive real input.

    Returns
    -------
    k_half_result : Float[Array, " ..."]
        Exact :math:`K_{1/2}(x)` values.
    """
    sqrt_pi_over_2x: Float[Array, " ..."] = jnp.sqrt(jnp.pi / (2.0 * x))
    exp_neg_x: Float[Array, " ..."] = jnp.exp(-x)
    k_half_result: Float[Array, " ..."] = sqrt_pi_over_2x * exp_neg_x
    return k_half_result


@jaxtyped(typechecker=beartype)
@jax.jit
def bessel_kv(v: ScalarFloat, x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Compute the modified Bessel function :math:`K_v(x)`.

    Extended Summary
    ----------------
    JAX-compatible, numerically stable, and differentiable
    approximation of :math:`K_v(x)` for real order
    :math:`v \geq 0` and :math:`x > 0`. Supports broadcasting,
    autodiff, JIT, and vmap.

    Implementation Logic
    --------------------
    1. **Classify order** --
       Determine if *v* is integer, half-integer, or general.
    2. **Small-x branch** (x <= 2) --
       Series expansion: reflection formula for non-integer v,
       specialised K_0/K_1 series + recurrence for integer v.
    3. **Large-x branch** (x > 2) --
       5-term asymptotic expansion.
    4. **Combine** --
       ``jnp.where`` selects the appropriate branch.
    5. **Half-integer shortcut** --
       Exact formula for v = 0.5.

    Parameters
    ----------
    v : ScalarFloat
        Order of the Bessel function (:math:`v \geq 0`).
    x : Float[Array, " ..."]
        Positive real input array.

    Returns
    -------
    final_result : Float[Array, " ..."]
        Approximated values of :math:`K_v(x)`.

    Notes
    -----
    The transition between small- and large-x approximations
    is at ``x = 2.0``. For integer orders ``n > 1``, forward
    recurrence with masked updates is used.
    """
    v: Float[Array, ""] = jnp.asarray(v)
    x: Float[Array, " ..."] = jnp.asarray(x)
    dtype: jnp.dtype = x.dtype

    v_int: Float[Array, ""] = jnp.round(v)
    epsilon_tolerance: float = 1e-10
    is_integer: Bool[Array, ""] = jnp.abs(v - v_int) < epsilon_tolerance

    small_x_non_int: Float[Array, " ..."] = _bessel_kv_small_non_integer(
        v, x, dtype
    )
    small_x_int: Float[Array, " ..."] = _bessel_kv_small_integer(v, x, dtype)
    small_x_vals: Float[Array, " ..."] = jnp.where(
        is_integer, small_x_int, small_x_non_int
    )

    large_x_vals: Float[Array, " ..."] = _bessel_kv_large(v, x)

    small_x_threshold: float = 2.0
    general_result: Float[Array, " ..."] = jnp.where(
        x <= small_x_threshold, small_x_vals, large_x_vals
    )

    k_half_vals: Float[Array, " ..."] = _bessel_k_half(x)
    is_half: Bool[Array, ""] = jnp.abs(v - 0.5) < epsilon_tolerance
    final_result: Float[Array, " ..."] = jnp.where(
        is_half, k_half_vals, general_result
    )

    return final_result


def _calculate_bessel_contributions(
    kirk_params: Float[Array, " 12"],
    r: Float[Array, " h w"],
    term1: Float[Array, ""],
) -> Float[Array, " h w"]:
    r"""Evaluate the three Bessel :math:`K_0` terms of the Kirkland potential.

    Parameters
    ----------
    kirk_params : Float[Array, " 12"]
        Kirkland parameters for one element (first 6 used).
    r : Float[Array, " h w"]
        Radial distance grid in Angstroms.
    term1 : Float[Array, ""]
        Prefactor :math:`4\pi^2 a_0 e_k`.

    Returns
    -------
    Float[Array, " h w"]
        Sum of three Bessel contributions scaled by *term1*.
    """
    bessel_term1: Float[Array, " h w"] = kirk_params[0] * bessel_kv(
        0.0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[1]) * r
    )
    bessel_term2: Float[Array, " h w"] = kirk_params[2] * bessel_kv(
        0.0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[3]) * r
    )
    bessel_term3: Float[Array, " h w"] = kirk_params[4] * bessel_kv(
        0.0, 2.0 * jnp.pi * jnp.sqrt(kirk_params[5]) * r
    )
    return term1 * (bessel_term1 + bessel_term2 + bessel_term3)


def _calculate_gaussian_contributions(
    kirk_params: Float[Array, " 12"],
    r: Float[Array, " h w"],
    term2: Float[Array, ""],
) -> Float[Array, " h w"]:
    r"""Evaluate the three Gaussian terms of the Kirkland potential.

    Parameters
    ----------
    kirk_params : Float[Array, " 12"]
        Kirkland parameters for one element (last 6 used).
    r : Float[Array, " h w"]
        Radial distance grid in Angstroms.
    term2 : Float[Array, ""]
        Prefactor :math:`2\pi^2 a_0 e_k`.

    Returns
    -------
    Float[Array, " h w"]
        Sum of three Gaussian contributions scaled by
        *term2*.
    """
    gauss_term1: Float[Array, " h w"] = (
        kirk_params[6] / kirk_params[7]
    ) * jnp.exp(-(jnp.pi**2 / kirk_params[7]) * r**2)
    gauss_term2: Float[Array, " h w"] = (
        kirk_params[8] / kirk_params[9]
    ) * jnp.exp(-(jnp.pi**2 / kirk_params[9]) * r**2)
    gauss_term3: Float[Array, " h w"] = (
        kirk_params[10] / kirk_params[11]
    ) * jnp.exp(-(jnp.pi**2 / kirk_params[11]) * r**2)
    return term2 * (gauss_term1 + gauss_term2 + gauss_term3)


def _downsample_potential(
    supersampled_potential: Float[Array, " h w"],
    supersampling: int,
    target_height: int,
    target_width: int,
) -> Float[Array, " h w"]:
    """Downsample supersampled potential to target resolution.

    Parameters
    ----------
    supersampled_potential : Float[Array, " h w"]
        Potential on the fine (supersampled) grid.
    supersampling : int
        Supersampling factor used during computation.
    target_height : int
        Desired output height in pixels.
    target_width : int
        Desired output width in pixels.

    Returns
    -------
    potential_resized : Float[Array, " h w"]
        Potential averaged down to target resolution.
    """
    height: int = supersampled_potential.shape[0]
    width: int = supersampled_potential.shape[1]
    new_height: int = (height // supersampling) * supersampling
    new_width: int = (width // supersampling) * supersampling

    cropped: Float[Array, " h_crop w_crop"] = jax.lax.dynamic_slice(
        supersampled_potential, (0, 0), (new_height, new_width)
    )

    reshaped: Float[Array, " h_new supersampling w_new supersampling"] = (
        cropped.reshape(
            new_height // supersampling,
            supersampling,
            new_width // supersampling,
            supersampling,
        )
    )

    potential: Float[Array, " h_new w_new"] = jnp.mean(reshaped, axis=(1, 3))
    potential_resized: Float[Array, " h w"] = jax.lax.dynamic_slice(
        potential, (0, 0), (target_height, target_width)
    )
    return potential_resized


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=["grid_shape", "supersampling"])
def single_atom_potential(
    atom_no: ScalarInt,
    pixel_size: ScalarFloat,
    grid_shape: Tuple[int, int],
    center_coords: Optional[Float[Array, " 2"]] = None,
    supersampling: int = 4,
) -> Float[Array, " h w"]:
    r"""Compute projected potential of a single atom.

    Extended Summary
    ----------------
    Uses the Kirkland parameterization of electron scattering
    factors, which decomposes the projected potential into
    three Bessel :math:`K_0` terms and three Gaussian terms:

    .. math::

        V(r) = 4\pi^2 a_0 e_k \sum_{i=1}^{3}
        a_i\,K_0(2\pi\sqrt{b_i}\,r)
        + 2\pi^2 a_0 e_k \sum_{i=1}^{3}
        \frac{c_i}{d_i}\,
        \exp\!\left(-\frac{\pi^2}{d_i}\,r^2\right)

    Implementation Logic
    --------------------
    1. **Physical constants** --
       Bohr radius :math:`a_0 = 0.5292` Angstroms,
       :math:`e_k = 14.4` eV Angstroms.
    2. **Load Kirkland parameters** --
       12 coefficients for the specified element.
    3. **Supersampled coordinate grid** --
       ``grid_shape * supersampling`` with step size
       ``pixel_size / supersampling``.
    4. **Radial distances** --
       ``r = sqrt(dx^2 + dy^2 + eps)`` to avoid NaN at
       the origin.
    5. **Evaluate potential** --
       :func:`_calculate_bessel_contributions` +
       :func:`_calculate_gaussian_contributions`.
    6. **Downsample** --
       Average over supersampling pixels via
       :func:`_downsample_potential`.

    Parameters
    ----------
    atom_no : ScalarInt
        Atomic number (1-indexed).
    pixel_size : ScalarFloat
        Real-space pixel size in Angstroms.
    grid_shape : Tuple[int, int]
        Output grid shape ``(height, width)``.
    center_coords : Float[Array, " 2"], optional
        ``(x, y)`` position in Angstroms to center the atom.
        If ``None``, centers at grid origin.
    supersampling : int, optional
        Supersampling factor. Default is 4.

    Returns
    -------
    potential_resized : Float[Array, " h w"]
        Projected potential at the target resolution in
        Kirkland units.
    """
    a0: Float[Array, ""] = jnp.asarray(0.5292)
    ek: Float[Array, ""] = jnp.asarray(14.4)
    term1: Float[Array, ""] = 4.0 * (jnp.pi**2) * a0 * ek
    term2: Float[Array, ""] = 2.0 * (jnp.pi**2) * a0 * ek
    kirkland_array: Float[Array, " 103 12"] = kirkland_potentials()
    atom_idx: Int[Array, ""] = (atom_no - 1).astype(jnp.int32)
    kirk_params: Float[Array, " 12"] = jax.lax.dynamic_slice(
        kirkland_array, (atom_idx, jnp.int32(0)), (1, 12)
    )[0]
    step_size: Float[Array, ""] = pixel_size / supersampling
    grid_height: int = grid_shape[0] * supersampling
    grid_width: int = grid_shape[1] * supersampling
    if center_coords is None:
        center_x: Float[Array, ""] = 0.0
        center_y: Float[Array, ""] = 0.0
    else:
        center_x: Float[Array, ""] = center_coords[0]
        center_y: Float[Array, ""] = center_coords[1]
    y_coords: Float[Array, " h"] = (
        jnp.arange(grid_height) - grid_height // 2
    ) * step_size + center_y
    x_coords: Float[Array, " w"] = (
        jnp.arange(grid_width) - grid_width // 2
    ) * step_size + center_x
    ya: Float[Array, " h w"]
    xa: Float[Array, " h w"]
    ya, xa = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    epsilon: float = 1e-10
    r: Float[Array, " h w"] = jnp.sqrt(
        (xa - center_x) ** 2 + (ya - center_y) ** 2 + epsilon
    )

    part1: Float[Array, " h w"] = _calculate_bessel_contributions(
        kirk_params, r, term1
    )
    part2: Float[Array, " h w"] = _calculate_gaussian_contributions(
        kirk_params, r, term2
    )
    supersampled_potential: Float[Array, " h w"] = part1 + part2

    target_height: int = grid_shape[0]
    target_width: int = grid_shape[1]

    potential_resized: Float[Array, " h w"] = _downsample_potential(
        supersampled_potential, supersampling, target_height, target_width
    )
    return potential_resized


# JIT compile single_atom_potential with static arguments
single_atom_potential = jax.jit(
    single_atom_potential, static_argnames=["grid_shape", "supersampling"]
)


@jaxtyped(typechecker=beartype)
def _slice_atoms(
    coords: Float[Array, " N 3"],
    atom_numbers: Int[Array, " N"],
    slice_thickness: ScalarNumeric,
) -> Float[Array, " N 4"]:
    """Partition atoms into slices along the z-axis.

    Extended Summary
    ----------------
    Assigns each atom to a slice based on its z-coordinate and
    the specified slice thickness, then sorts by slice index
    for efficient slice-by-slice processing.

    Implementation Logic
    --------------------
    1. **Compute slice indices** --
       ``floor((z - z_min) / slice_thickness)``.
    2. **Build output array** --
       ``[x, y, slice_idx, atom_number]`` per atom.
    3. **Sort by slice** --
       ``argsort`` on slice indices.

    Parameters
    ----------
    coords : Float[Array, " N 3"]
        Atomic positions ``(x, y, z)`` in Angstroms.
    atom_numbers : Int[Array, " N"]
        Atomic numbers for each atom.
    slice_thickness : ScalarNumeric
        Thickness of each slice in Angstroms.

    Returns
    -------
    sorted_atoms : Float[Array, " N 4"]
        ``[x, y, slice_idx, atom_number]`` per atom, sorted
        by ascending slice index (0-based).

    Notes
    -----
    Atoms at slice boundaries are assigned to the lower slice.
    All arrays are JAX arrays for JIT compatibility.
    """
    z_coords: Float[Array, " N"] = coords[:, 2]
    z_min: Float[Array, ""] = jnp.min(z_coords)
    slice_indices: Real[Array, " N"] = jnp.floor(
        (z_coords - z_min) / slice_thickness
    )
    sorted_atoms_presort: Float[Array, " N 4"] = jnp.column_stack(
        [
            coords[:, 0],
            coords[:, 1],
            slice_indices.astype(jnp.float32),
            atom_numbers.astype(jnp.float32),
        ]
    )
    sorted_order: Real[Array, " N"] = jnp.argsort(slice_indices)
    sorted_atoms: Float[Array, " N 4"] = sorted_atoms_presort[sorted_order]
    return sorted_atoms


default_repeats: Int[Array, " 3"] = jnp.array([1, 1, 1])


@jaxtyped(typechecker=beartype)
def _compute_grid_dimensions(
    x_coords: Float[Array, " N"],
    y_coords: Float[Array, " N"],
    padding: ScalarFloat,
    pixel_size: ScalarFloat,
    grid_height: Optional[int] = None,
    grid_width: Optional[int] = None,
) -> Tuple[Float[Array, ""], Float[Array, ""], int, int]:
    """Compute grid dimensions and coordinate origins.

    Parameters
    ----------
    x_coords : Float[Array, " N"]
        X coordinates of all atoms in Angstroms.
    y_coords : Float[Array, " N"]
        Y coordinates of all atoms in Angstroms.
    padding : ScalarFloat
        Padding added to each side in Angstroms.
    pixel_size : ScalarFloat
        Pixel size in Angstroms.
    grid_height : int, optional
        Fixed grid height (for JIT). If ``None``, computed
        from coordinate range.
    grid_width : int, optional
        Fixed grid width (for JIT). If ``None``, computed
        from coordinate range.

    Returns
    -------
    x_min : Float[Array, ""]
        Minimum x with padding in Angstroms.
    y_min : Float[Array, ""]
        Minimum y with padding in Angstroms.
    width : int
        Grid width in pixels.
    height : int
        Grid height in pixels.
    """
    x_coords_min: Float[Array, ""] = jnp.min(x_coords)
    y_coords_min: Float[Array, ""] = jnp.min(y_coords)
    x_min: Float[Array, ""] = x_coords_min - padding
    y_min: Float[Array, ""] = y_coords_min - padding

    if grid_height is not None and grid_width is not None:
        return x_min, y_min, grid_width, grid_height

    x_coords_max: Float[Array, ""] = jnp.max(x_coords)
    y_coords_max: Float[Array, ""] = jnp.max(y_coords)
    x_max: Float[Array, ""] = x_coords_max + padding
    y_max: Float[Array, ""] = y_coords_max + padding
    x_range: Float[Array, ""] = x_max - x_min
    y_range: Float[Array, ""] = y_max - y_min
    width_float: Float[Array, ""] = jnp.ceil(x_range / pixel_size)
    height_float: Float[Array, ""] = jnp.ceil(y_range / pixel_size)
    width: Int[Array, ""] = width_float.astype(jnp.int32)
    height: Int[Array, ""] = height_float.astype(jnp.int32)
    width_int: int = int(width)
    height_int: int = int(height)
    return x_min, y_min, width_int, height_int


@jaxtyped(typechecker=beartype)
def _process_all_slices(
    atom_data: Tuple[
        Float[Array, " N"],  # x_coords
        Float[Array, " N"],  # y_coords
        Int[Array, " N"],  # atom_nums
        Int[Array, " N"],  # slice_indices
    ],
    potential_data: Tuple[
        Float[Array, " 118 h w"],  # atomic_potentials
        Int[Array, " 119"],  # atom_to_idx_array
    ],
    grid_params: Tuple[
        ScalarFloat,  # x_min
        ScalarFloat,  # y_min
        ScalarFloat,  # pixel_size
        int,  # height
        int,  # width
    ],
    num_slices: Optional[int] = None,
) -> Float[Array, " h w n_slices"]:
    """Assemble all potential slices from atomic contributions.

    Extended Summary
    ----------------
    For each slice, iterates over all atoms (via ``lax.scan``)
    and accumulates FFT-shifted atomic potentials. Slices are
    processed in parallel via ``jax.vmap``.

    Parameters
    ----------
    atom_data : tuple
        ``(x_coords, y_coords, atom_nums, slice_indices)``
        arrays of length N.
    potential_data : tuple
        ``(atomic_potentials, atom_to_idx_array)`` -- lookup
        table of precomputed potentials.
    grid_params : tuple
        ``(x_min, y_min, pixel_size, height, width)``.
    num_slices : int, optional
        Fixed number of slices (for JIT). If ``None``,
        computed from max slice index.

    Returns
    -------
    all_slices : Float[Array, " h w n_slices"]
        3D array of potential slices.
    """
    x_coords, y_coords, atom_nums, slice_indices = atom_data
    atomic_potentials, atom_to_idx_array = potential_data
    x_min, y_min, pixel_size, height, width = grid_params

    if num_slices is not None:
        n_slices: int = num_slices
    else:
        max_slice_idx: Int[Array, ""] = jnp.max(slice_indices).astype(
            jnp.int32
        )
        n_slices: int = int(max_slice_idx + 1)
    all_slices: Float[Array, " h w n_slices"] = jnp.zeros(
        (height, width, n_slices), dtype=jnp.float32
    )
    ky: Float[Array, " h 1"] = jnp.fft.fftfreq(height, d=1.0).reshape(-1, 1)
    kx: Float[Array, " 1 w"] = jnp.fft.fftfreq(width, d=1.0).reshape(1, -1)

    def _process_single_slice(slice_idx: int) -> Float[Array, " h w"]:
        """Accumulate potentials for atoms in one slice.

        Parameters
        ----------
        slice_idx : int
            Index of the slice to process.

        Returns
        -------
        slice_potential : Float[Array, " h w"]
            Accumulated potential for this slice.
        """
        slice_potential: Float[Array, " h w"] = jnp.zeros(
            (height, width), dtype=jnp.float32
        )
        center_x: float = width / 2.0
        center_y: float = height / 2.0

        def _add_atom_contribution(
            carry: Float[Array, " h w"],
            atom_data: Tuple[ScalarFloat, ScalarFloat, ScalarInt, ScalarInt],
        ) -> Tuple[Float[Array, " h w"], None]:
            """Add one atom's FFT-shifted potential to the slice.

            Parameters
            ----------
            carry : Float[Array, " h w"]
                Running slice potential.
            atom_data : tuple
                ``(x, y, atom_no, atom_slice_idx)``.

            Returns
            -------
            updated_pot : Float[Array, " h w"]
                Slice potential with this atom added.
            None
                No stacked output.
            """
            slice_pot: Float[Array, " h w"] = carry
            x: ScalarFloat
            y: ScalarFloat
            atom_no: ScalarInt
            atom_slice_idx: ScalarInt
            x, y, atom_no, atom_slice_idx = atom_data

            x_offset: ScalarFloat = x - x_min
            y_offset: ScalarFloat = y - y_min
            pixel_x: ScalarFloat = x_offset / pixel_size
            pixel_y: ScalarFloat = y_offset / pixel_size
            shift_x: ScalarFloat = pixel_x - center_x
            shift_y: ScalarFloat = pixel_y - center_y

            atom_idx: int = atom_to_idx_array[atom_no]
            atom_pot: Float[Array, " h w"] = atomic_potentials[atom_idx]
            kx_sx: Float[Array, " h w"] = kx * shift_x
            ky_sy: Float[Array, " h w"] = ky * shift_y
            phase_arg: Float[Array, " h w"] = kx_sx + ky_sy
            phase: Complex[Array, " h w"] = jnp.exp(2j * jnp.pi * phase_arg)
            atom_pot_fft: Complex[Array, " h w"] = jnp.fft.fft2(atom_pot)
            shifted_fft: Complex[Array, " h w"] = atom_pot_fft * phase
            shifted_pot: Float[Array, " h w"] = jnp.real(
                jnp.fft.ifft2(shifted_fft)
            )

            contribution: Float[Array, " h w"] = jnp.where(
                atom_slice_idx == slice_idx,
                shifted_pot,
                jnp.zeros_like(shifted_pot),
            ).astype(jnp.float32)
            updated_pot: Float[Array, " h w"] = (
                slice_pot + contribution
            ).astype(jnp.float32)
            return updated_pot, None

        slice_potential, _ = jax.lax.scan(
            _add_atom_contribution,
            slice_potential,
            (x_coords, y_coords, atom_nums, slice_indices),
        )
        return slice_potential

    slice_indices_array: Int[Array, " n_slices"] = jnp.arange(n_slices)
    processed_slices: Float[Array, "n_slices h w"] = jax.vmap(
        _process_single_slice
    )(slice_indices_array)
    all_slices: Float[Array, " h w n_slices"] = processed_slices.transpose(
        1, 2, 0
    )
    return all_slices


@jaxtyped(typechecker=beartype)
def _build_shift_masks(
    repeats: Int[Array, " 3"],
    max_n: int = 20,
) -> Tuple[Bool[Array, " max_n^3"], Int[Array, " max_n^3 3"]]:
    """Build shift indices and validity masks for periodic repeats.

    Parameters
    ----------
    repeats : Int[Array, " 3"]
        Number of repeats in ``(x, y, z)``.
    max_n : int, optional
        Maximum repeat count per axis. Default is 20.

    Returns
    -------
    mask_flat : Bool[Array, " max_n^3"]
        Validity mask for each shift combination.
    shift_indices : Int[Array, " max_n^3 3"]
        Integer shift indices ``(ix, iy, iz)``.
    """
    nx: Int[Array, ""] = repeats[0]
    ny: Int[Array, ""] = repeats[1]
    nz: Int[Array, ""] = repeats[2]

    ix: Int[Array, " max_n"] = jnp.arange(max_n)
    iy: Int[Array, " max_n"] = jnp.arange(max_n)
    iz: Int[Array, " max_n"] = jnp.arange(max_n)

    mask_x: Bool[Array, " max_n"] = ix < nx
    mask_y: Bool[Array, " max_n"] = iy < ny
    mask_z: Bool[Array, " max_n"] = iz < nz

    ixx: Int[Array, " max_n max_n max_n"]
    iyy: Int[Array, " max_n max_n max_n"]
    izz: Int[Array, " max_n max_n max_n"]
    ixx, iyy, izz = jnp.meshgrid(ix, iy, iz, indexing="ij")

    mask_x_expanded: Bool[Array, " max_n 1 1"] = mask_x[:, None, None]
    mask_y_expanded: Bool[Array, " 1 max_n 1"] = mask_y[None, :, None]
    mask_z_expanded: Bool[Array, " 1 1 max_n"] = mask_z[None, None, :]
    mask_3d: Bool[Array, " max_n max_n max_n"] = (
        mask_x_expanded & mask_y_expanded & mask_z_expanded
    )

    ixx_flat: Int[Array, " max_n^3"] = ixx.ravel()
    iyy_flat: Int[Array, " max_n^3"] = iyy.ravel()
    izz_flat: Int[Array, " max_n^3"] = izz.ravel()
    shift_indices: Int[Array, " max_n^3 3"] = jnp.stack(
        [ixx_flat, iyy_flat, izz_flat], axis=-1
    )
    mask_flat: Bool[Array, " max_n^3"] = mask_3d.ravel()

    return mask_flat, shift_indices


@jaxtyped(typechecker=beartype)
def _tile_positions_with_shifts(
    positions: Float[Array, " N 3"],
    atomic_numbers: Int[Array, " N"],
    shift_vectors: Float[Array, "max_n^3 3"],
    mask_flat: Bool[Array, " max_n^3"],
) -> Tuple[Float[Array, "max_n^3*N 3"], Int[Array, " max_n^3*N"]]:
    """Tile atomic positions by adding shift vectors.

    Parameters
    ----------
    positions : Float[Array, " N 3"]
        Original positions in Angstroms.
    atomic_numbers : Int[Array, " N"]
        Atomic numbers for the original atoms.
    shift_vectors : Float[Array, "max_n^3 3"]
        Lattice shift vectors in Angstroms.
    mask_flat : Bool[Array, " max_n^3"]
        Validity mask for each shift.

    Returns
    -------
    repeated_positions_masked : Float[Array, "max_n^3*N 3"]
        Tiled positions (invalid ones zeroed out).
    repeated_atomic_numbers_masked : Int[Array, " max_n^3*N"]
        Tiled atomic numbers (invalid ones zeroed).
    """
    n_atoms: int = positions.shape[0]
    max_n: int = 20
    max_shifts: int = max_n * max_n * max_n

    positions_expanded: Float[Array, " 1 N 3"] = positions[None, :, :]
    positions_broadcast: Float[Array, "max_n^3 N 3"] = jnp.broadcast_to(
        positions_expanded, (max_shifts, n_atoms, 3)
    )
    shift_vectors_expanded: Float[Array, "max_n^3 1 3"] = shift_vectors[
        :, None, :
    ]
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

    atom_mask: Bool[Array, " max_n^3*N"] = jnp.repeat(mask_flat, n_atoms)
    atom_mask_float: Float[Array, " max_n^3*N"] = atom_mask.astype(jnp.float32)
    atom_mask_expanded: Float[Array, "max_n^3*N 1"] = atom_mask_float[:, None]
    repeated_positions_masked: Float[Array, "max_n^3*N 3"] = (
        repeated_positions_flat * atom_mask_expanded
    )

    atomic_numbers_tiled: Int[Array, " max_n^3*N"] = jnp.tile(
        atomic_numbers, max_shifts
    )
    atom_mask_int: Int[Array, " max_n^3*N"] = atom_mask.astype(jnp.int32)
    repeated_atomic_numbers_masked: Int[Array, " max_n^3*N"] = (
        atomic_numbers_tiled * atom_mask_int
    )

    return (repeated_positions_masked, repeated_atomic_numbers_masked)


@jaxtyped(typechecker=beartype)
def _apply_repeats_or_return(
    positions: Float[Array, " N 3"],
    atomic_numbers: Int[Array, " N"],
    lattice: Float[Array, " 3 3"],
    repeats: Int[Array, " 3"],
) -> Tuple[Float[Array, " M 3"], Int[Array, " M"]]:
    """Apply periodic repeats or return unchanged positions.

    Parameters
    ----------
    positions : Float[Array, " N 3"]
        Atomic positions in Angstroms.
    atomic_numbers : Int[Array, " N"]
        Atomic numbers.
    lattice : Float[Array, " 3 3"]
        Lattice vectors in Angstroms.
    repeats : Int[Array, " 3"]
        Number of repeats ``(nx, ny, nz)``. ``[1,1,1]``
        means no repeating.

    Returns
    -------
    positions_out : Float[Array, " M 3"]
        Tiled (or padded) positions.
    atomic_numbers_out : Int[Array, " M"]
        Tiled (or padded) atomic numbers.
    """

    def _apply_repeats_with_lattice(
        positions: Float[Array, " N 3"],
        atomic_numbers: Int[Array, " N"],
        lattice: Float[Array, " 3 3"],
    ) -> Tuple[Float[Array, " M 3"], Int[Array, " M"]]:
        """Tile positions using lattice vectors.

        Parameters
        ----------
        positions : Float[Array, " N 3"]
            Original positions.
        atomic_numbers : Int[Array, " N"]
            Original atomic numbers.
        lattice : Float[Array, " 3 3"]
            Lattice vectors.

        Returns
        -------
        tuple
            Tiled positions and atomic numbers.
        """
        mask_flat: Bool[Array, " M"]
        shift_indices: Int[Array, " M 3"]
        mask_flat, shift_indices = _build_shift_masks(repeats)

        mask_float: Float[Array, " M"] = mask_flat.astype(jnp.float32)
        shift_indices_float: Float[Array, " M 3"] = shift_indices.astype(
            jnp.float32
        )
        mask_expanded: Float[Array, " M 1"] = mask_float[:, None]
        shift_indices_masked: Float[Array, " M 3"] = (
            shift_indices_float * mask_expanded
        )
        shift_vectors: Float[Array, " M 3"] = shift_indices_masked @ lattice

        return _tile_positions_with_shifts(
            positions, atomic_numbers, shift_vectors, mask_flat
        )

    def _return_unchanged(
        positions: Float[Array, " N 3"],
        atomic_numbers: Int[Array, " N"],
    ) -> Tuple[Float[Array, " M 3"], Int[Array, " M"]]:
        """Return positions padded to match tiled shape.

        Parameters
        ----------
        positions : Float[Array, " N 3"]
            Original positions.
        atomic_numbers : Int[Array, " N"]
            Original atomic numbers.

        Returns
        -------
        tuple
            Zero-padded positions and atomic numbers.
        """
        n_atoms: int = positions.shape[0]
        max_n: int = 20
        max_shifts: int = max_n * max_n * max_n
        max_total: int = max_shifts * n_atoms

        positions_padded: Float[Array, " M 3"] = jnp.zeros((max_total, 3))
        atomic_numbers_padded: Int[Array, " M"] = jnp.zeros(
            max_total, dtype=jnp.int32
        )

        positions_padded = positions_padded.at[:n_atoms].set(positions)
        atomic_numbers_padded = atomic_numbers_padded.at[:n_atoms].set(
            atomic_numbers
        )

        return (positions_padded, atomic_numbers_padded)

    return jax.lax.cond(
        jnp.any(repeats > 1),
        lambda pos, an, lat: _apply_repeats_with_lattice(pos, an, lat),
        lambda pos, an, _: _return_unchanged(pos, an),
        positions,
        atomic_numbers,
        lattice,
    )


@jaxtyped(typechecker=beartype)
def _build_potential_lookup(
    atom_nums: Int[Array, " N"],
    height: int,
    width: int,
    pixel_size: ScalarFloat,
    supersampling: ScalarInt,
) -> Tuple[Float[Array, " 118 h w"], Int[Array, " 119"]]:
    """Build lookup table of precomputed atomic potentials.

    Parameters
    ----------
    atom_nums : Int[Array, " N"]
        Atomic numbers present in the structure.
    height : int
        Grid height in pixels.
    width : int
        Grid width in pixels.
    pixel_size : ScalarFloat
        Pixel size in Angstroms.
    supersampling : ScalarInt
        Supersampling factor.

    Returns
    -------
    atomic_potentials : Float[Array, " 118 h w"]
        Precomputed potentials for up to 118 elements.
    atom_to_idx_array : Int[Array, " 119"]
        Mapping from atomic number to index in
        *atomic_potentials*.
    """
    unique_atoms: Int[Array, " 118"] = jnp.unique(
        atom_nums, size=118, fill_value=-1
    )
    valid_mask: Bool[Array, " 118"] = unique_atoms >= 0

    @jax.jit
    def _calc_single_potential_fixed_grid(
        atom_no: ScalarInt, is_valid: Bool
    ) -> Float[Array, " h w"]:
        """Compute potential for one atom type on the fixed grid.

        Parameters
        ----------
        atom_no : ScalarInt
            Atomic number.
        is_valid : Bool
            Whether this slot contains a real element.

        Returns
        -------
        Float[Array, " h w"]
            Potential (zeros if invalid).
        """
        potential = single_atom_potential(
            atom_no=atom_no,
            pixel_size=pixel_size,
            grid_shape=(height, width),
            center_coords=jnp.array([0.0, 0.0]),
            supersampling=supersampling,
            potential_extent=4.0,
        )
        return jnp.where(is_valid, potential, jnp.zeros((height, width)))

    atomic_potentials: Float[Array, " 118 h w"] = jax.vmap(
        _calc_single_potential_fixed_grid
    )(unique_atoms, valid_mask)
    atom_to_idx_array: Int[Array, " 119"] = jnp.full(119, -1, dtype=jnp.int32)

    indices: Int[Array, " 118"] = jnp.arange(118, dtype=jnp.int32)
    atom_indices: Int[Array, " 118"] = jnp.where(valid_mask, unique_atoms, -1)

    def _update_mapping2(
        carry: Int[Array, " 119"], idx_atom: Tuple[ScalarInt, ScalarInt]
    ) -> Tuple[Int[Array, " 119"], None]:
        """Update atomic-number-to-index mapping.

        Parameters
        ----------
        carry : Int[Array, " 119"]
            Current mapping array.
        idx_atom : tuple
            ``(index, atomic_number)`` pair.

        Returns
        -------
        mapping_array : Int[Array, " 119"]
            Updated mapping.
        None
            No stacked output.
        """
        mapping_array: Int[Array, " 119"] = carry
        idx: ScalarInt
        atom: ScalarInt
        idx, atom = idx_atom
        mapping_array = jnp.where(
            atom >= 0, mapping_array.at[atom].set(idx), mapping_array
        )
        return mapping_array, None

    atom_to_idx_array, _ = jax.lax.scan(
        _update_mapping2, atom_to_idx_array, (indices, atom_indices)
    )
    return atomic_potentials, atom_to_idx_array


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=["grid_shape", "supersampling"])
def kirkland_potentials_crystal(
    crystal_data: CrystalData,
    pixel_size: ScalarFloat,
    slice_thickness: ScalarFloat = 1.0,
    repeats: Int[Array, " 3"] = default_repeats,
    padding: ScalarFloat = 4.0,
    supersampling: ScalarInt = 4,
    grid_shape: Optional[Tuple[int, int, int]] = None,
) -> PotentialSlices:
    """Convert :class:`~ptyrodactyl.tools.CrystalData` to potential slices.

    Extended Summary
    ----------------
    Calculates atomic potentials and assembles them into slices
    using FFT-based sub-pixel positioning. Supports periodic
    tiling and padding to avoid wraparound artefacts.

    Implementation Logic
    --------------------
    1. **Tile structure** --
       If ``repeats > [1,1,1]``, replicate atoms using
       lattice vectors via :func:`_apply_repeats_or_return`.
    2. **Partition into slices** --
       :func:`_slice_atoms` assigns each atom to a z-slice.
    3. **Compute grid dimensions** --
       :func:`_compute_grid_dimensions` with padding.
    4. **Build potential lookup** --
       :func:`_build_potential_lookup` precomputes one
       potential kernel per unique element.
    5. **Assemble slices** --
       :func:`_process_all_slices` places each atom's
       potential at its ``(x, y)`` position via FFT shifting
       and accumulates per slice.
    6. **Crop padding** --
       Remove border pixels to eliminate wraparound.

    Parameters
    ----------
    crystal_data : CrystalData
        Input crystal structure.
    pixel_size : ScalarFloat
        Pixel size in Angstroms.
    slice_thickness : ScalarFloat, optional
        Thickness per slice in Angstroms. Default is 1.0.
    repeats : Int[Array, " 3"], optional
        Unit cell repeats ``[nx, ny, nz]``. Default
        ``[1, 1, 1]``.
    padding : ScalarFloat, optional
        Padding on each side in Angstroms. Default is 4.0.
    supersampling : ScalarInt, optional
        Supersampling factor. Default is 4.
    grid_shape : Tuple[int, int, int], optional
        Static ``(height, width, n_slices)`` for JIT. If
        ``None``, dimensions are computed dynamically.

    Returns
    -------
    pot_slices : PotentialSlices
        Sliced potentials with wraparound artefacts removed.

    Notes
    -----
    For JIT compilation, provide *grid_shape*. Compute it as::

        height = ceil((y_range + 2*padding) / pixel_size)
        width  = ceil((x_range + 2*padding) / pixel_size)
        n_slices = ceil(z_extent / slice_thickness)

    See Also
    --------
    :func:`single_atom_potential` : Projected potential for one
        atom.
    :func:`~ptyrodactyl.tools.make_potential_slices` : Factory
        for :class:`~ptyrodactyl.tools.PotentialSlices`.
    """
    positions: Float[Array, " N 3"] = crystal_data.positions
    atomic_numbers: Int[Array, " N"] = crystal_data.atomic_numbers
    lattice: Float[Array, " 3 3"] = crystal_data.lattice

    positions, atomic_numbers = _apply_repeats_or_return(
        positions, atomic_numbers, lattice, repeats
    )

    sliced_atoms: Float[Array, " N 4"] = _slice_atoms(
        coords=positions,
        atom_numbers=atomic_numbers,
        slice_thickness=slice_thickness,
    )
    x_coords: Float[Array, " N"] = sliced_atoms[:, 0]
    y_coords: Float[Array, " N"] = sliced_atoms[:, 1]
    slice_indices: Int[Array, " N"] = sliced_atoms[:, 2].astype(jnp.int32)
    atom_nums: Int[Array, " N"] = sliced_atoms[:, 3].astype(jnp.int32)

    grid_height: Optional[int] = None
    grid_width: Optional[int] = None
    num_slices: Optional[int] = None
    if grid_shape is not None:
        grid_height, grid_width, num_slices = grid_shape

    x_min: Float[Array, ""]
    y_min: Float[Array, ""]
    width: int
    height: int
    x_min, y_min, width, height = _compute_grid_dimensions(
        x_coords, y_coords, padding, pixel_size, grid_height, grid_width
    )

    atomic_potentials: Float[Array, " 118 h w"]
    atom_to_idx_array: Int[Array, " 119"]
    atomic_potentials, atom_to_idx_array = _build_potential_lookup(
        atom_nums, height, width, pixel_size, supersampling
    )

    all_slices: Float[Array, " h w n_slices"] = _process_all_slices(
        (x_coords, y_coords, atom_nums, slice_indices),
        (atomic_potentials, atom_to_idx_array),
        (x_min, y_min, pixel_size, height, width),
        num_slices,
    )

    crop_pixels: int = int(jnp.round(padding / pixel_size))
    output_height: int = height - 2 * crop_pixels
    output_width: int = width - 2 * crop_pixels
    n_slices_out: int = all_slices.shape[2]
    cropped_slices: Float[Array, " h_crop w_crop n_slices"] = (
        jax.lax.dynamic_slice(
            all_slices,
            (crop_pixels, crop_pixels, 0),
            (output_height, output_width, n_slices_out),
        )
    )
    pot_slices: PotentialSlices = make_potential_slices(
        slices=cropped_slices,
        slice_thickness=slice_thickness,
        calib=pixel_size,
    )
    return pot_slices
