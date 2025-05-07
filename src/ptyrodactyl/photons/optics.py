import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

from .helper import add_phase_screen
from .types import OpticalWavefront, scalar_int, scalar_num

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def angular_spectrum_prop(
    incoming: OpticalWavefront, z_move: scalar_num
) -> OpticalWavefront:
    """
    Description
    -----------
    Propagate a complex field using the angular spectrum method.

    Parameters
    ----------
    - `incoming` (OpticalWavefront)
        PyTree with the following parameters:
        - `field` (Complex[Array, "H W"]):
            Input complex field
        - `wavelength` (Float[Array, ""]):
            Wavelength of light in meters
        - `dx` (Float[Array, ""]):
            Grid spacing in meters
        - `z_position` (Float[Array, ""]):
            Wave front position in meters
    - `z_move` (scalar_num):
        Propagation distance in meters

    Returns
    -------
    - `propagated` (OpticalWavefront):
        Propagated wave front

    Flow
    ----
    - Get the shape of the input field
    - Calculate the wavenumber
    - Spatial frequency coordinates
    - Compute the squared spatial frequencies
    - Angular spectrum transfer function
    - Ensure evanescent waves are properly handled
    - Fourier transform of the input field
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Return the propagated field
    """
    ny: scalar_int = incoming.field.shape[0]
    nx: scalar_int = incoming.field.shape[1]
    wavenumber: Float[Array, ""] = 2 * jnp.pi / incoming.wavelength
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    FSQ: Float[Array, "H W"] = (FX**2) + (FY**2)
    H: Complex[Array, ""] = jnp.exp(
        1j * wavenumber * z_move * jnp.sqrt(1 - (incoming.wavelength**2) * FSQ)
    )
    evanescent_mask: Bool[Array, "H W"] = FSQ <= (1 / incoming.wavelength) ** 2
    H_mask: Complex[Array, "H W"] = H * evanescent_mask
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, "H W"] = field_ft * H_mask
    propagated_field: Complex[Array, "H W"] = jnp.fft.ifft2(propagated_ft)
    propagated = OpticalWavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + z_move,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fresnel_prop(incoming: OpticalWavefront, z_move: scalar_num) -> OpticalWavefront:
    """
    Description
    -----------
    Propagate a complex field using the Fresnel approximation.

    Parameters
    ----------
    - `incoming` (OpticalWavefront)
        PyTree with the following parameters:
        - `field` (Complex[Array, "H W"]):
            Input complex field
        - `wavelength` (Float[Array, ""]):
            Wavelength of light in meters
        - `dx` (Float[Array, ""]):
            Grid spacing in meters
        - `z_position` (Float[Array, ""]):
            Wave front position in meters
    - `z_move` (scalar_num):
        Propagation distance in meters

    Returns
    -------
    - `propagated` (OpticalWavefront):
        Propagated wave front

    Flow
    ----
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
    ny: scalar_int = incoming.field.shape[0]
    nx: scalar_int = incoming.field.shape[1]
    k: Float[Array, ""] = (2 * jnp.pi) / incoming.wavelength
    x: Float[Array, "H"] = jnp.arange(-nx // 2, nx // 2) * incoming.dx
    y: Float[Array, "W"] = jnp.arange(-ny // 2, ny // 2) * incoming.dx
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    X, Y = jnp.meshgrid(x, y)
    quadratic_phase: Float[Array, "H W"] = k / (2 * z_move) * (X**2 + Y**2)
    field_with_phase: Complex[Array, "H W"] = add_phase_screen(
        incoming.field, quadratic_phase
    )
    field_ft: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(field_with_phase))
    )
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    transfer_phase: Float[Array, "H W"] = (
        (-1) * jnp.pi * incoming.wavelength * z_move * (FX**2 + FY**2)
    )
    propagated_ft: Complex[Array, "H W"] = add_phase_screen(field_ft, transfer_phase)
    propagated_field: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(propagated_ft))
    )
    final_quadratic_phase: Float[Array, "H W"] = k / (2 * z_move) * (X**2 + Y**2)
    final_propagated_field: Complex[Array, "H W"] = add_phase_screen(
        propagated_field, final_quadratic_phase
    )
    propagated = OpticalWavefront(
        field=final_propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + z_move,
    )
    return propagated


@jaxtyped(typechecker=beartype)
def fraunhofer_prop(incoming: OpticalWavefront, z_move: scalar_num) -> OpticalWavefront:
    """
    Description
    -----------
    Propagate a complex field using the Fraunhofer approximation.

    Parameters
    ----------
    - `incoming` (OpticalWavefront)
        PyTree with the following parameters:
        - `field` (Complex[Array, "H W"]):
            Input complex field
        - `wavelength` (Float[Array, ""]):
            Wavelength of light in meters
        - `dx` (Float[Array, ""]):
            Grid spacing in meters
        - `z_position` (Float[Array, ""]):
            Wave front position in meters
    - `z_move` (scalar_num):
        Propagation distance in meters

    Returns
    -------
    - `propagated` (OpticalWavefront):
        Propagated wave front

    Flow
    ----
    - Get the shape of the input field
    - Calculate the spatial frequency coordinates
    - Create the meshgrid of spatial frequencies
    - Compute the transfer function for Fraunhofer propagation
    - Compute the Fourier transform of the input field
    - Apply the transfer function in the Fourier domain
    - Inverse Fourier transform to get the propagated field
    - Return the propagated field
    """
    ny: scalar_int = incoming.field.shape[0]
    nx: scalar_int = incoming.field.shape[1]
    k = 2 * jnp.pi / incoming.wavelength
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=incoming.dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=incoming.dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    H: Complex[Array, "H W"] = jnp.exp(
        -1j * jnp.pi * incoming.wavelength * z_move * (FX**2 + FY**2)
    ) / (1j * incoming.wavelength * z_move)
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(incoming.field)
    propagated_ft: Complex[Array, "H W"] = field_ft * H
    propagated_field: Complex[Array, "H W"] = jnp.fft.ifft2(propagated_ft)
    propagated = OpticalWavefront(
        field=propagated_field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position + z_move,
    )
    return propagated
