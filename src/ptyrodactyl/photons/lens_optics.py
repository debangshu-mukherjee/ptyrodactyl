"""
Module: ptyrodactyl.photons.lens_optics
---------------------------------------
Codes for optical propgation steps.

Functions
---------
- `angular_spectrum_prop`:
    Propagates a complex optical field using the angular spectrum method
- `fresnel_prop`:
    Propagates a complex optical field using the Fresnel approximation
- `fraunhofer_prop`:
    Propagates a complex optical field using the Fraunhofer approximation
- `digital_zoom`:
    Zooms an optical wavefront by a specified factor
- `optical_zoom`:
    Modifies the calibration of an optical wavefront without changing its field
"""

import jax
import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Bool, Complex, Float

from ptyrodactyl._decorators import beartype, jaxtyped

from .helper import add_phase_screen
from .photon_types import (
    OpticalWavefront,
    make_optical_wavefront,
    scalar_float,
    scalar_integer,
    scalar_numeric,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def angular_spectrum_prop(
    incoming: OpticalWavefront,
    z_move: scalar_numeric,
    refractive_index: Optional[scalar_numeric] = 1.0,
) -> OpticalWavefront:
    """Propagate a complex field using the angular spectrum method.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:

        field : Complex[Array, "H W"]
            Input complex field
        wavelength : Float[Array, ""]
            Wavelength of light in meters
        dx : Float[Array, ""]
            Grid spacing in meters
        z_position : Float[Array, ""]
            Wave front position in meters
    z_move : scalar_numeric
        Propagation distance in meters
        This is in free space.
    refractive_index : Optional[scalar_numeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    OpticalWavefront
        Propagated wave front

    Notes
    -----
    Algorithm:

    - Get the shape of the input field
    - Calculate the wavenumber
    - Compute the path length
    - Create spatial frequency coordinates
    - Compute the squared spatial frequencies
    - Angular spectrum transfer function
    - Ensure evanescent waves are properly handled
    - Fourier transform of the input field
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Return the propagated field
    """
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]
    wavenumber: Float[Array, ""] = 2 * jnp.pi / incoming.wavelength
    path_length = refractive_index * z_move
    fx: Float[Array, " H"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " W"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    FSQ: Float[Array, "H W"] = (FX**2) + (FY**2)
    asp_transfer: Complex[Array, ""] = jnp.exp(
        1j * wavenumber * path_length * jnp.sqrt(1 - (incoming.wavelength**2) * FSQ),
    )
    evanescent_mask: Bool[Array, " H W"] = (1 / incoming.wavelength) ** 2 >= FSQ
    H_mask: Complex[Array, "H W"] = asp_transfer * evanescent_mask
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, "H W"] = field_ft * H_mask
    propagated_field: Complex[Array, "H W"] = jnp.fft.ifft2(propagated_ft)
    propagated: OpticalWavefront = make_optical_wavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fresnel_prop(
    incoming: OpticalWavefront,
    z_move: scalar_numeric,
    refractive_index: Optional[scalar_numeric] = 1.0,
) -> OpticalWavefront:
    """Propagate a complex field using the Fresnel approximation.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:

        field : Complex[Array, "H W"]
            Input complex field
        wavelength : Float[Array, ""]
            Wavelength of light in meters
        dx : Float[Array, ""]
            Grid spacing in meters
        z_position : Float[Array, ""]
            Wave front position in meters
    z_move : scalar_numeric
        Propagation distance in meters
        This is in free space.
    refractive_index : Optional[scalar_numeric], optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    OpticalWavefront
        Propagated wave front

    Notes
    -----
    Algorithm:

    - Calculate the wavenumber
    - Create spatial coordinates
    - Quadratic phase factor for Fresnel approximation (pre-free-space propagation)
    - Apply quadratic phase to the input field
    - Compute Fourier transform of the input field
    - Compute spatial frequency coordinates
    - Transfer function for Fresnel propagation
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Final quadratic phase factor (post-free-space propagation)
    - Apply final quadratic phase factor
    - Return the propagated field
    """
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]
    k: Float[Array, ""] = (2 * jnp.pi) / incoming.wavelength
    x: Float[Array, " H"] = jnp.arange(-nx // 2, nx // 2) * incoming.dx
    y: Float[Array, " W"] = jnp.arange(-ny // 2, ny // 2) * incoming.dx
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    X, Y = jnp.meshgrid(x, y)
    path_length = refractive_index * z_move
    quadratic_phase: Float[Array, "H W"] = k / (2 * path_length) * (X**2 + Y**2)
    field_with_phase: Complex[Array, "H W"] = add_phase_screen(
        incoming.field,
        quadratic_phase,
    )
    field_ft: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(field_with_phase)),
    )
    fx: Float[Array, " H"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " W"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    transfer_phase: Float[Array, "H W"] = (
        (-1) * jnp.pi * incoming.wavelength * path_length * (FX**2 + FY**2)
    )
    propagated_ft: Complex[Array, "H W"] = add_phase_screen(field_ft, transfer_phase)
    propagated_field: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(propagated_ft)),
    )
    final_quadratic_phase: Float[Array, "H W"] = k / (2 * path_length) * (X**2 + Y**2)
    final_propagated_field: Complex[Array, "H W"] = jnp.fft.ifftshift(
        add_phase_screen(propagated_field, final_quadratic_phase),
    )
    propagated: OpticalWavefront = make_optical_wavefront(
        field=final_propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fraunhofer_prop(
    incoming: OpticalWavefront,
    z_move: scalar_float,
    refractive_index: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """Propagate a complex field using the Fraunhofer approximation.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with the following parameters:

        field : Complex[Array, "H W"]
            Input complex field
        wavelength : Float[Array, ""]
            Wavelength of light in meters
        dx : Float[Array, ""]
            Grid spacing in meters
        z_position : Float[Array, ""]
            Wave front position in meters
    z_move : scalar_float
        Propagation distance in meters.
        This is in free space.
    refractive_index : scalar_float, optional
        Index of refraction of the medium. Default is 1.0 (vacuum).

    Returns
    -------
    OpticalWavefront
        Propagated wave front

    Notes
    -----
    Algorithm:

    - Get the shape of the input field
    - Calculate the spatial frequency coordinates
    - Create the meshgrid of spatial frequencies
    - Compute the transfer function for Fraunhofer propagation
    - Compute the Fourier transform of the input field
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Return the propagated field
    """
    ny: scalar_integer = incoming.field.shape[0]
    nx: scalar_integer = incoming.field.shape[1]
    fx: Float[Array, " H"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, " W"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    path_length = refractive_index * z_move
    H: Complex[Array, "H W"] = jnp.exp(
        -1j * jnp.pi * incoming.wavelength * path_length * (FX**2 + FY**2),
    ) / (1j * incoming.wavelength * path_length)
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, "H W"] = field_ft * H
    propagated_field: Complex[Array, "H W"] = jnp.fft.ifft2(propagated_ft)
    propagated: OpticalWavefront = make_optical_wavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + path_length,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def digital_zoom(
    wavefront: OpticalWavefront,
    zoom_factor: scalar_numeric,
) -> OpticalWavefront:
    """Zoom an optical wavefront by a specified factor.

    Key is this returns the same sized array as the original wavefront.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Incoming optical wavefront.
    zoom_factor : scalar_numeric
        Zoom factor (greater than 1 to zoom in, less than 1 to zoom out).

    Returns
    -------
    OpticalWavefront
        Zoomed optical wavefront of the same spatial dimensions.

    Notes
    -----
    Algorithm:

    - Calculate the new dimensions of the zoomed wavefront.
    - Resize the wavefront field using cubic interpolation.
    - Crop the resized field to match the original dimensions.
    - Return the new optical wavefront with the updated field, wavelength,
      and pixel size.
    """
    H: int
    W: int
    H, W = wavefront.field.shape
    H_cut: int = int(H / zoom_factor)
    W_cut: int = int(W / zoom_factor)
    start_H: int = (H - H_cut) // 2
    start_W: int = (W - W_cut) // 2
    cut_field: Complex[Array, "H_cut W_cut"] = jax.lax.dynamic_slice(
        wavefront.field,
        (start_H, start_W),
        (H_cut, W_cut),
    )
    zoomed_field: Complex[Array, "H W"] = jax.image.resize(
        image=cut_field,
        shape=(H, W),
        method="trilinear",
    )
    zoomed_wavefront: OpticalWavefront = make_optical_wavefront(
        field=zoomed_field,
        wavelength=wavefront.wavelength,
        dx=wavefront.dx / zoom_factor,
        z_position=wavefront.z_position,
    )
    return zoomed_wavefront


@jaxtyped(typechecker=beartype)
def optical_zoom(
    wavefront: OpticalWavefront,
    zoom_factor: scalar_numeric,
) -> OpticalWavefront:
    """This is the optical zoom function that only modifies the calibration and leaves everything else the same.

    Parameters
    ----------
    wavefront : OpticalWavefront
        Incoming optical wavefront.
    zoom_factor : scalar_numeric
        Zoom factor (greater than 1 to zoom in, less than 1 to zoom out).

    Returns
    -------
    OpticalWavefront
        Zoomed optical wavefront of the same spatial dimensions.
    """
    new_dx = wavefront.dx * zoom_factor
    zoomed_wavefront: OpticalWavefront = make_optical_wavefront(
        field=wavefront.field,
        wavelength=wavefront.wavelength,
        dx=new_dx,
        z_position=wavefront.z_position,
    )
    return zoomed_wavefront
