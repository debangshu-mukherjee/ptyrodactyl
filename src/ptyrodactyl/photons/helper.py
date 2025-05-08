"""
Module: photons.helper
---------------------------
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
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def create_spatial_grid(
    diameter: Num[Array, ""], num_points: Int[Array, ""]
) -> Tuple[Float[Array, "N N"], Float[Array, "N N"]]:
    """
    Description
    -----------
    Create a 2D spatial grid for optical propagation.

    Parameters
    ----------
    - `diameter` (Num[Array, ""]):
        Physical size of the grid in meters
    - `num_points` (Int[Array, ""]):
        Number of points in each dimension

    Returns
    -------
    - Tuple of meshgrid arrays (X, Y) representing spatial coordinates

    Flow
    ----
    - Create a linear space of points along the x-axis
    - Create a linear space of points along the y-axis
    - Create a meshgrid of spatial coordinates
    - Return the meshgrid
    """
    x: Float[Array, "N"] = jnp.linspace(-diameter / 2, diameter / 2, num_points)
    y: Float[Array, "N"] = jnp.linspace(-diameter / 2, diameter / 2, num_points)
    xx: Float[Array, "N N"]
    yy: Float[Array, "N N"]
    xx, yy = jnp.meshgrid(x, y)
    return (xx, yy)


@jaxtyped(typechecker=beartype)
def normalize_field(field: Complex[Array, "H W"]) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Normalize complex field to unit power

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field

    Returns
    -------
    - `normalized_field` (Complex[Array, "H W"]):
        Normalized complex field

    Flow
    ----
    - Calculate the power of the field as the sum of the square of the absolute value of the field
    - Normalize the field by dividing by the square root of the power
    - Return the normalized field
    """
    power: Float[Array, ""] = jnp.sum(jnp.abs(field) ** 2)
    normalized_field: Complex[Array, "H W"] = field / jnp.sqrt(power)
    return normalized_field


@jaxtyped(typechecker=beartype)
def add_phase_screen(
    field: Num[Array, "H W"],
    phase: Float[Array, "H W"],
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Add a phase screen to a complex field,
    as:

    .. math::
        $field \times \exp(i \cdot phase)$.

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field
    - `phase` (Float[Array, "H W"]):
        Phase screen to add

    Returns
    -------
    - `screened_field` (Complex[Array, "H W"]):
        Field with phase screen added

    Flow
    ----
    - Multiply the input field by the exponential of the phase screen
    - Return the screened field
    """
    screened_field: Complex[Array, "H W"] = field * jnp.exp(1j * phase)
    return screened_field


@jaxtyped(typechecker=beartype)
def field_intensity(field: Complex[Array, "H W"]) -> Float[Array, "H W"]:
    """
    Description
    -----------
    Calculate intensity from complex field

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field

    Returns
    -------
    - `intensity` (Float[Array, "H W"]):
        Intensity of the field

    Flow
    ----
    - Calculate the intensity as the square of the absolute value of the field
    - Return the intensity
    """
    intensity: Float[Array, "H W"] = jnp.abs(field) ** 2
    return intensity
