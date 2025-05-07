import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from .helper import add_phase_screen
from .lenses import create_lens_phase
from .types import LensParams, OpticalWavefront, scalar_num

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
    - `OpticalWavefront`:
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

    return OpticalWavefront(
        field=transmitted_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def zoom_wavefront(
    wavefront: OpticalWavefront, zoom_factor: scalar_num
) -> OpticalWavefront:
    """
    Zoom an optical wavefront by a specified factor.
    Key is this returns the same sized array as the
    original wavefront.

    Parameters
    ----------
    - `wavefront` (OpticalWavefront):
        Incoming optical wavefront.
    - `zoom_factor` (scalar_num):
        Zoom factor (greater than 1 to zoom in, less than 1 to zoom out).

    Returns
    -------
    - `zoomed_wavefront` (OpticalWavefront):
        Zoomed optical wavefront of the same spatial dimensions.

    Flow
    ----
    - Calculate the new dimensions of the zoomed wavefront.
    - Resize the wavefront field using Lanczos interpolation.
    - Crop the resized field to match the original dimensions.
    - Return the new optical wavefront with the updated field, wavelength,
    and pixel size.
    """
    H: int
    W: int
    H, W = wavefront.field.shape
    H_zoom: int = int(H * zoom_factor)
    W_zoom: int = int(W * zoom_factor)
    zoomed_field: Complex[Array, "H_zoom W_zoom"] = jax.image.resize(
        image=wavefront.field, shape=(H_zoom, W_zoom), method="lanczos5"
    )
    start_H: int = (H_zoom - H) // 2
    start_W: int = (W_zoom - W) // 2
    zoom_cropped: Complex[Array, "H W"] = jax.lax.dynamic_slice(
        zoomed_field, (start_H, start_W), (H, W)
    )
    return OpticalWavefront(
        field=zoom_cropped,
        wavelength=wavefront.wavelength,
        dx=wavefront.dx / zoom_factor,
        z_position=wavefront.z_position,
    )
