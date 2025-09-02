"""
Module: ptyrodactyl.photons.helper
----------------------------------
Utility functions for optical propagation.

Functions
---------
- `create_spatial_grid`:
    Creates a 2D spatial grid for optical propagation
- `normalize_field`:
    Normalizes a complex field to unit power
- `add_phase_screen`:
    Adds a phase screen to a complex field
- `field_intensity`:
    Calculates intensity from a complex field
- `scale_pixel`:
    Rescales OpticalWavefront pixel size while keeping array shape fixed
"""

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, Int, Num

from ptyrodactyl._decorators import beartype, jaxtyped

from .photon_types import (OpticalWavefront, make_optical_wavefront,
                           scalar_float)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def create_spatial_grid(
    diameter: Num[Array, " "],
    num_points: Int[Array, " "],
) -> Tuple[Float[Array, "N N"], Float[Array, "N N"]]:
    """Create a 2D spatial grid for optical propagation.

    Parameters
    ----------
    diameter : Num[Array, " "]
        Physical size of the grid in meters.
    num_points : Int[Array, " "]
        Number of points in each dimension.

    Returns
    -------
    Tuple[Float[Array, "N N"], Float[Array, "N N"]]
        Tuple of meshgrid arrays (X, Y) representing spatial coordinates.

    Notes
    -----
    Algorithm:
    - Create a linear space of points along the x-axis
    - Create a linear space of points along the y-axis
    - Create a meshgrid of spatial coordinates
    - Return the meshgrid
    """
    x: Float[Array, " N"] = jnp.linspace(-diameter / 2, diameter / 2, num_points)
    y: Float[Array, " N"] = jnp.linspace(-diameter / 2, diameter / 2, num_points)
    xx: Float[Array, "N N"]
    yy: Float[Array, "N N"]
    xx, yy = jnp.meshgrid(x, y)
    return (xx, yy)


@jaxtyped(typechecker=beartype)
def normalize_field(field: Complex[Array, "H W"]) -> Complex[Array, "H W"]:
    """Normalize complex field to unit power.

    Parameters
    ----------
    field : Complex[Array, "H W"]
        Input complex field.

    Returns
    -------
    Complex[Array, "H W"]
        Normalized complex field.

    Notes
    -----
    Algorithm:
    - Calculate the power of the field as the sum of the square of the absolute value of the field
    - Normalize the field by dividing by the square root of the power
    - Return the normalized field
    """
    power: Float[Array, " "] = jnp.sum(jnp.abs(field) ** 2)
    normalized_field: Complex[Array, "H W"] = field / jnp.sqrt(power)
    return normalized_field


@jaxtyped(typechecker=beartype)
def add_phase_screen(
    field: Num[Array, "H W"],
    phase: Float[Array, "H W"],
) -> Complex[Array, "H W"]:
    """Add a phase screen to a complex field.

    Parameters
    ----------
    field : Num[Array, "H W"]
        Input complex field.
    phase : Float[Array, "H W"]
        Phase screen to add.

    Returns
    -------
    Complex[Array, "H W"]
        Field with phase screen added.

    Notes
    -----
    Algorithm:
    - Multiply the input field by the exponential of the phase screen
    - Return the screened field
    """
    screened_field: Complex[Array, "H W"] = field * jnp.exp(1j * phase)
    return screened_field


@jaxtyped(typechecker=beartype)
def field_intensity(field: Complex[Array, "H W"]) -> Float[Array, "H W"]:
    """Calculate intensity from complex field.

    Parameters
    ----------
    field : Complex[Array, "H W"]
        Input complex field.

    Returns
    -------
    Float[Array, "H W"]
        Intensity of the field.

    Notes
    -----
    Algorithm:
    - Calculate the intensity as the square of the absolute value of the field
    - Return the intensity
    """
    intensity: Float[Array, "H W"] = jnp.abs(field) ** 2
    return intensity


@jaxtyped(typechecker=beartype)
def scale_pixel(
    wavefront: OpticalWavefront,
    new_dx: scalar_float,
) -> OpticalWavefront:
    """Rescale OpticalWavefront pixel size while keeping array shape fixed.

    JAX-compatible (jit/vmap-safe). Crops or pads to preserve shape.

    Parameters
    ----------
    wavefront : OpticalWavefront
        OpticalWavefront to be resized.
    new_dx : scalar_float
        New pixel size (meters).

    Returns
    -------
    OpticalWavefront
        Resized OpticalWavefront with updated pixel size
        and resized field, which is of the same size as
        the original field.
    """
    field: Complex[Array, "H W"] = wavefront.field
    old_dx: scalar_float = wavefront.dx
    H: int
    W: int
    H, W = field.shape
    scale: scalar_float = new_dx / old_dx
    current_fov_h: scalar_float = H * old_dx
    current_fov_w: scalar_float = W * old_dx
    new_fov_h: scalar_float = H * new_dx
    new_fov_w: scalar_float = W * new_dx

    def smaller_pixel_size(field: Complex[Array, "H W"]):
        """
        If the new pixel size is smaller than the old one,
        then the new FOV is smaller too at the same field
        size. So we will first find the new smaller FOV,
        and crop to that size with the current pixel size.
        Then we will resize to the new pizel size with the
        cropped FOV so that the size of the field remains
        the same.
        So here the order is crop, then resize.
        """
        new_H: Int[Array, " "] = jnp.floor(new_fov_h / old_dx).astype(int)
        new_W: Int[Array, " "] = jnp.floor(new_fov_w / old_dx).astype(int)
        start_h: Int[Array, " "] = jnp.floor(
            (current_fov_h - new_fov_h) / (2 * old_dx)
        ).astype(int)
        start_w: Int[Array, " "] = jnp.floor(
            (current_fov_w - new_fov_w) / (2 * old_dx)
        ).astype(int)
        cropped: Complex[Array, "new_H new_W"] = jax.lax.dynamic_slice(
            field, (start_h, start_w), (new_H, new_W)
        )
        resized: Complex[Array, "H W"] = jax.image.resize(
            cropped,
            (H, W),
            method="linear",
            antialias=True,
        )
        return resized

    def larger_pixel_size(field: Complex[Array, "H W"]):
        """
        If the new pixel size is larger than the old one,
        then the new FOV of the final field is larger too
        at the same field size. So we will need to first
        get the current FOV data with the new pixel size,
        which will be smaller than the current field size.
        Following this, we need to pad out to fill the
        field.
        So here the order is resize then pad.
        """
        data_minimia_h: Float[Array, " "] = jnp.min(jnp.abs(field))
        new_H: Int[Array, " "] = jnp.floor(current_fov_h / new_dx).astype(int)
        new_W: Int[Array, " "] = jnp.floor(current_fov_w / new_dx).astype(int)
        resized: Complex[Array, "H W"] = jax.image.resize(
            field,
            (new_H, new_W),
            method="linear",
            antialias=True,
        )
        pad_h_0: Int[Array, " "] = jnp.floor((H - new_H) / 2).astype(int)
        pad_h_1: Int[Array, " "] = H - (new_H + pad_h_0)
        pad_w_0: Int[Array, " "] = jnp.floor((W - new_W) / 2).astype(int)
        pad_w_1: Int[Array, " "] = W - (new_W + pad_w_0)
        return jnp.pad(
            resized,
            ((pad_h_0, pad_h_1), (pad_w_0, pad_w_1)),
            mode="constant",
            constant_values=data_minimia_h,
        )

    resized_field = jax.lax.cond(
        scale > 1.0, larger_pixel_size, smaller_pixel_size, field
    )
    return make_optical_wavefront(
        field=resized_field,
        dx=new_dx,
        wavelength=wavefront.wavelength,
        z=wavefront.z,
    )
