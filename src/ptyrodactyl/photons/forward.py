"""
Module: photons.forward
---------------------------
Codes for optical propagation through lenses and optical elements.

Functions
---------
- `lens_propagation`:
    Propagates an optical wavefront through a lens
- `linear_interaction`:
    Propagates an optical wavefront through a sample using linear interaction
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from .helper import add_phase_screen
from .lenses import create_lens_phase
from .photon_types import (LensParams, OpticalWavefront, SampleFunction,
                           make_optical_wavefront)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def lens_propagation(incoming: OpticalWavefront, lens: LensParams) -> OpticalWavefront:
    """
    Description
    -----------
    Propagate an optical wavefront through a lens.
    The lens is modeled as a thin lens with a given focal length and diameter.

    Parameters
    ----------
    - `incoming` (OpticalWavefront):
        The incoming optical wavefront
    - `lens` (LensParams):
        The lens parameters including focal length and diameter

    Returns
    -------
    - `outgoing` (OpticalWavefront):
        The propagated optical wavefront after passing through the lens

    Flow
    ----
    - Create a meshgrid of coordinates based on the incoming wavefront's shape and pixel size.
    - Calculate the phase profile and transmission function of the lens.
    - Apply the phase screen to the incoming wavefront's field.
    - Return the new optical wavefront with the updated field, wavelength, and pixel size.
    """
    H: int
    W: int
    H, W = incoming.field.shape
    x: Float[Array, W] = jnp.linspace(-W // 2, W // 2 - 1, W) * incoming.dx
    y: Float[Array, H] = jnp.linspace(-H // 2, H // 2 - 1, H) * incoming.dx
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    X, Y = jnp.meshgrid(x, y)

    phase_profile: Float[Array, "H W"]
    transmission: Float[Array, "H W"]
    phase_profile, transmission = create_lens_phase(X, Y, lens, incoming.wavelength)
    transmitted_field: Complex[Array, "H W"] = add_phase_screen(
        incoming.field * transmission,
        phase_profile,
    )
    outgoing: OpticalWavefront = make_optical_wavefront(
        field=transmitted_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return outgoing


@jaxtyped(typechecker=beartype)
def linear_interaction(
    sample: SampleFunction,
    light: OpticalWavefront,
) -> OpticalWavefront:
    """
    Description
    -----------
    Propagate an optical wavefront through a sample using linear interaction.
    The sample is modeled as a complex function that modifies the incoming wavefront.

    Parameters
    ----------
    - `sample` (SampleFunction):
        The sample function representing the optical properties of the sample
    - `light` (OpticalWavefront):
        The incoming optical wavefront

    Returns
    -------
    - `interacted` (OpticalWavefront):
        The propagated optical wavefront after passing through the sample

    """
    new_field: Complex[Array, "H W"] = sample.sample * light.field
    interacted: OpticalWavefront = make_optical_wavefront(
        field=new_field,
        wavelength=light.wavelength,
        dx=light.dx,
        z_position=light.z_position,
    )
    return interacted
