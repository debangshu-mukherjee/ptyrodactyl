"""Plotting and visualization helpers.

Extended Summary
----------------
This module provides plotting-oriented helpers for presentation
transforms and visualization colormaps.

Routine Listings
----------------
:func:`clip_cbed`
    Clip CBED patterns to mrad extent and resize to target
    shape.
:func:`contrast_stretch`
    Rescale image intensity between specified percentiles.
:func:`create_phosphor_colormap`
    Create custom colormap simulating phosphor screen appearance.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Literal, Sequence, Tuple, Union
from jax import lax
from jax.image import resize
from jaxtyping import Array, Float, Int, jaxtyped
from matplotlib.colors import LinearSegmentedColormap

from ptyrodactyl.tools.constants import relativistic_wavelength_ang
from ptyrodactyl.types import scalar_float, scalar_num


@jaxtyped(typechecker=beartype)
@jax.jit
def clip_cbed(
    cbed: Float[Array, "H W"],
    fourier_calib_inv_ang: scalar_float,
    voltage_kv: scalar_num,
    extent_mrad: scalar_float,
    output_shape: Tuple[int, int],
) -> Float[Array, "Ho Wo"]:
    """Clip CBED pattern to mrad extent and resize.

    Extended Summary
    ----------------
    Extracts the central region of a CBED pattern corresponding
    to a given angular extent in milliradians, then resizes to
    the target output shape using bilinear interpolation.

    Implementation Logic
    --------------------
    1. **Convert mrad to pixels** --
       Use wavelength and Fourier calibration to convert
       ``extent_mrad`` to a pixel radius.
    2. **Extract central crop** --
       ``lax.dynamic_slice`` around the pattern center.
    3. **Resize** --
       Bilinear resize to ``output_shape``.

    Parameters
    ----------
    cbed : Float[Array, "H W"]
        Input CBED pattern (fftshifted, centered).
    fourier_calib_inv_ang : scalar_float
        Fourier space calibration in inverse Angstroms per
        pixel.
    voltage_kv : scalar_num
        Accelerating voltage in kilovolts.
    extent_mrad : scalar_float
        Half-angle extent in milliradians (radius from
        center).
    output_shape : Tuple[int, int]
        Target output shape ``(height, width)``.

    Returns
    -------
    resized : Float[Array, "Ho Wo"]
        Clipped and resized CBED pattern.

    :see: contrast_stretch, create_phosphor_colormap.
    """
    h: int = cbed.shape[0]
    w: int = cbed.shape[1]

    wavelength_ang: Float[Array, " "] = relativistic_wavelength_ang(voltage_kv)
    mrad_per_inv_ang: Float[Array, " "] = wavelength_ang * 1000.0

    extent_inv_ang: Float[Array, " "] = extent_mrad / mrad_per_inv_ang
    extent_pixels: Int[Array, " "] = jnp.ceil(
        extent_inv_ang / fourier_calib_inv_ang
    ).astype(jnp.int32)

    center_y: int = h // 2
    center_x: int = w // 2

    y_start: Int[Array, " "] = jnp.maximum(0, center_y - extent_pixels)
    y_end: Int[Array, " "] = jnp.minimum(h, center_y + extent_pixels)
    x_start: Int[Array, " "] = jnp.maximum(0, center_x - extent_pixels)
    x_end: Int[Array, " "] = jnp.minimum(w, center_x + extent_pixels)

    clipped: Float[Array, "Hc Wc"] = lax.dynamic_slice(
        cbed,
        (y_start, x_start),
        (y_end - y_start, x_end - x_start),
    )

    resized: Float[Array, "Ho Wo"] = resize(
        clipped,
        output_shape,
        method="linear",
    )

    return resized


@jaxtyped(typechecker=beartype)
@jax.jit
def contrast_stretch(
    series: Union[Float[Array, " H W"], Float[Array, " N H W"]],
    p1: scalar_float,
    p2: scalar_float,
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
    p1 : scalar_float
        Lower percentile (0--100).
    p2 : scalar_float
        Upper percentile (0--100).

    Returns
    -------
    final_result : Float[Array, " H W"] | Float[Array, " N H W"]
        Rescaled image(s) with same shape as input.

    :see: clip_cbed, create_phosphor_colormap.
    """
    original_shape: Tuple[int, ...] = series.shape
    is_2d_image: int = 2
    if len(original_shape) == is_2d_image:
        series_reshaped: Float[Array, " N H W"] = series[jnp.newaxis, :, :]
    else:
        series_reshaped = series

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
    if len(original_shape) == is_2d_image:
        final_result: Union[Float[Array, " H W"], Float[Array, " N H W"]] = (
            transformed[0]
        )
    else:
        final_result = transformed
    return final_result


@jaxtyped(typechecker=beartype)
def create_phosphor_colormap(
    name: str = "phosphor",
) -> LinearSegmentedColormap:
    """Create a custom colormap that simulates a phosphor screen appearance.

    The colormap transitions from black through a bright phosphorescent green,
    with a slight white bloom at maximum intensity.

    Parameters
    ----------
    name : str, optional
        Name for the colormap. Default is 'phosphor'.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Custom phosphor screen colormap.

    Notes
    -----
    1. **Define Color Anchors** --
       Set transition points and RGB values from black
       through dark green, bright green, lighter green,
       to white bloom.
    2. **Extract Channel Data** --
       Separate positions and RGB values from color
       definitions into individual channel lists.
    3. **Build Segment Dict** --
       Create color channel definitions for red, green,
       and blue as required by LinearSegmentedColormap.
    4. **Construct Colormap** --
       Create and return LinearSegmentedColormap with
       the custom color segment dictionary.

    :see: contrast_stretch, clip_cbed.
    """
    colors: List[Tuple[float, Tuple[float, float, float]]] = [
        (0.0, (0.0, 0.0, 0.0)),
        (0.4, (0.0, 0.05, 0.0)),
        (0.7, (0.15, 0.85, 0.15)),
        (0.9, (0.45, 0.95, 0.45)),
        (1.0, (0.8, 1.0, 0.8)),
    ]
    positions: List[float] = [x[0] for x in colors]
    rgb_values: List[Tuple[float, float, float]] = [x[1] for x in colors]
    red: List[Tuple[float, float, float]] = [
        (pos, rgb[0], rgb[0])
        for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    green: List[Tuple[float, float, float]] = [
        (pos, rgb[1], rgb[1])
        for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    blue: List[Tuple[float, float, float]] = [
        (pos, rgb[2], rgb[2])
        for pos, rgb in zip(positions, rgb_values, strict=True)
    ]
    alpha: List[Tuple[float, float, float]] = [
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    segment_data: Dict[
        Literal["red", "green", "blue", "alpha"],
        Sequence[Tuple[float, ...]],
    ] = {"red": red, "green": green, "blue": blue, "alpha": alpha}
    cmap: LinearSegmentedColormap = LinearSegmentedColormap(
        name,
        segment_data,
    )
    return cmap


__all__: list[str] = [
    "clip_cbed",
    "contrast_stretch",
    "create_phosphor_colormap",
]
