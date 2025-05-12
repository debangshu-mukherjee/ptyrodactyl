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
- `simple_diffractogram`:
    Calculates the diffractogram of a sample using a simple model
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Complex, Float, jaxtyped

from .helper import add_phase_screen, field_intensity, scale_pixel
from .lenses import create_lens_phase
from .optics import circular_aperture, fraunhofer_prop, optical_zoom
from .photon_types import (Diffractogram, LensParams, OpticalWavefront,
                           SampleFunction, make_diffractogram,
                           make_optical_wavefront, scalar_float)

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


@jaxtyped(typechecker=beartype)
def simple_diffractogram(
    sample_cut: SampleFunction,
    lightwave: OpticalWavefront,
    zoom_factor: scalar_float,
    aperture_diameter: scalar_float,
    travel_distance: scalar_float,
    camera_pixel_size: scalar_float,
    aperture_center: Optional[Float[Array, "2"]] = None,
) -> Diffractogram:
    """
    Description
    -----------
    Calculate the diffractogram of a sample using a simple model.
    The lightwave interacts with the sample linearly, and is then
    zoomed optically. Following this it interacts with a circular
    aperture before propagating to the camera plane.
    The camera image is then scaled to the pixel size of the camera.
    The diffractogram is created from the camera image.

    Parameters
    ----------
    - `sample_cut` (SampleFunction):
        The sample function representing the optical properties of the sample
    - `lightwave` (OpticalWavefront):
        The incoming optical wavefront
    - `zoom_factor` (scalar_float):
        The zoom factor for the optical system
    - `aperture_diameter` (scalar_float):
        The diameter of the aperture in meters
    - `travel_distance` (scalar_float):
        The distance traveled by the light in meters
    - `camera_pixel_size` (scalar_float):
        The pixel size of the camera in meters
    - `aperture_center` (Optional[Float[Array, "2"]]):
        The center of the aperture in pixels

    Returns
    -------
    - `diffractogram` (Diffractogram):
        The calculated diffractogram of the sample

    Flow
    ----
    - Propagate the lightwave through the sample using linear interaction
    - Apply optical zoom to the wavefront
    - Apply a circular aperture to the zoomed wavefront
    - Propagate the wavefront to the camera plane using Fraunhofer propagation
    - Scale the pixel size of the camera image
    - Calculate the field intensity of the camera image
    - Create a diffractogram from the camera image
    """
    at_sample_plane: OpticalWavefront = linear_interaction(
        sample=sample_cut,
        light=lightwave,
    )
    zoomed_wave: OpticalWavefront = optical_zoom(at_sample_plane, zoom_factor)
    after_aperture: OpticalWavefront = circular_aperture(
        zoomed_wave, aperture_diameter, aperture_center
    )
    at_camera: OpticalWavefront = fraunhofer_prop(after_aperture, travel_distance)
    at_camera_scaled: OpticalWavefront = scale_pixel(
        at_camera,
        camera_pixel_size,
    )
    scaled_camera_image: Float[Array, "H W"] = field_intensity(at_camera_scaled.field)
    diffractogram: Diffractogram = make_diffractogram(
        image=scaled_camera_image,
        wavelength=at_camera_scaled.wavelength,
        dx=at_camera_scaled.dx,
    )
    return diffractogram
