"""
Module: photons.forward
---------------------------
Codes for optical propagation through lenses and optical elements.

Functions
---------
- `lens_propagation`:
    Propagates an optical wavefront through a lens
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from .helper import add_phase_screen
from .lenses import create_lens_phase
from .photon_types import (LensParams, OpticalWavefront, make_lens_params,
                           make_optical_wavefront, scalar_num)

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
    x: Float[Array, "W"] = jnp.linspace(-W // 2, W // 2 - 1, W) * incoming.dx
    y: Float[Array, "H"] = jnp.linspace(-H // 2, H // 2 - 1, H) * incoming.dx
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    X, Y = jnp.meshgrid(x, y)

    phase_profile: Float[Array, "H W"]
    transmission: Float[Array, "H W"]
    phase_profile, transmission = create_lens_phase(X, Y, lens, incoming.wavelength)
    transmitted_field: Complex[Array, "H W"] = add_phase_screen(
        incoming.field * transmission, phase_profile
    )
    outgoing: OpticalWavefront = make_optical_wavefront(
        field=transmitted_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return outgoing
