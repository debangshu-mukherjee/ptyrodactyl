import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped
from .helper import add_phase_screen

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def angular_spectrum_prop(
    field: Complex[Array, "H W"],
    z: Num[Array, ""],
    dx: Float[Array, ""],
    wavelength: Float[Array, ""],
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Propagate a complex field using the angular spectrum method.

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field
    - `z` (Num[Array, ""]):
        Propagation distance in meters
    - `dx` (Float[Array, ""]):
        Grid spacing in meters
    - `wavelength` (Float[Array, ""]):
        Wavelength of light in meters

    Returns
    -------
    - `propagated_field` (Complex[Array, "H W"]):
        Propagated complex field

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
    ny: Int[Array, ""] = field.shape[0]
    nx: Int[Array, ""] = field.shape[1]
    wavenumber: Float[Array, ""] = 2 * jnp.pi / wavelength
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)
    FSQ: Float[Array, "H W"] = (FX**2) + (FY**2)
    H: Complex[Array, ""] = jnp.exp(1j * wavenumber * z * jnp.sqrt(1 - (wavelength**2) * FSQ))
    evanescent_mask: Bool[Array, "H W"] = FSQ <= (1 / wavelength) ** 2
    H_mask: Complex[Array, "H W"] = H * evanescent_mask
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(field)
    propagated_ft: Complex[Array, "H W"] = field_ft * H_mask
    propagated_field: Complex[Array, "H W"] = jnp.fft.ifft2(propagated_ft)
    return propagated_field


@jaxtyped(typechecker=beartype)
def fresnel_prop(
    field: Complex[Array, "H W"],
    z: Num[Array, ""],
    dx: Float[Array, ""],
    wavelength: Float[Array, ""],
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Propagate a complex field using the Fresnel approximation.

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field
    - `z` (Num[Array, ""]):
        Propagation distance in meters
    - `dx` (Float[Array, ""]):
        Grid spacing in meters
    - `wavelength` (Float[Array, ""]):
        Wavelength of light in meters

    Returns
    -------
    - `final_propagated_field` (Complex[Array, "H W"]):
        Propagated complex field

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
    # Get the shape of the input field
    ny: Int[Array, ""] = field.shape[0]
    nx: Int[Array, ""] = field.shape[1]

    # Calculate the wavenumber
    k: Float[Array, ""] = (2 * jnp.pi) / wavelength  # Wavenumber

    # Create spatial coordinates
    x: Float[Array, "H"] = jnp.arange(-nx // 2, nx // 2) * dx
    y: Float[Array, "W"] = jnp.arange(-ny // 2, ny // 2) * dx
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    X, Y = jnp.meshgrid(x, y)

    # Quadratic phase factor for Fresnel approximation (pre-free-space propagation)
    quadratic_phase: Float[Array, "H W"] = k / (2 * z) * (X**2 + Y**2)

    # Apply quadratic phase to the input field
    field_with_phase: Complex[Array, "H W"] = add_phase_screen(
        field, quadratic_phase
    )

    # Compute Fourier transform of the input field
    field_ft: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(field_with_phase))
    )

    # Compute spatial frequency coordinates
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)

    # Transfer function for Fresnel propagation
    transfer_phase: Float[Array, "H W"] = (
        (-1) * jnp.pi * wavelength * z * (FX**2 + FY**2)
    )

    # Apply the transfer function in the Fourier domain
    propagated_ft: Complex[Array, "H W"] = add_phase_screen(
        field_ft, transfer_phase
    )

    # Inverse Fourier transform to get the propagated field
    propagated_field: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(propagated_ft))
    )

    # Final quadratic phase factor (post-free-space propagation)
    final_quadratic_phase: Float[Array, "H W"] = k / (2 * z) * (X**2 + Y**2)

    # Apply final quadratic phase factor
    final_propagated_field: Complex[Array, "H W"] = add_phase_screen(
        propagated_field, final_quadratic_phase
    )

    # Return the propagated field
    return final_propagated_field


@jaxtyped(typechecker=beartype)
def fraunhofer_prop(
    field: Complex[Array, "H W"],
    z: Num[Array, ""],
    dx: Float[Array, ""],
    wavelength: Float[Array, ""],
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Propagate a complex field using the Fraunhofer approximation.

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field
    - `z` (Num[Array, ""]):
        Propagation distance in meters
    - `dx` (Float[Array, ""]):
        Grid spacing in meters
    - `wavelength` (Float[Array, ""]):
        Wavelength of light in meters

    Returns
    -------
    - `propagated_field` (Complex[Array, "H W"]):
        Propagated complex field

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
    # Get the shape of the input field
    ny: Int[Array, ""] = field.shape[0]
    nx: Int[Array, ""] = field.shape[1]

    # Calculate the spatial frequency coordinates
    k = 2 * jnp.pi / wavelength  # Wavenumber
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=dx)

    # Create the meshgrid of spatial frequencies
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)

    # Compute the transfer function for Fraunhofer propagation
    H: Complex[Array, "H W"] = jnp.exp(
        -1j * jnp.pi * wavelength * z * (FX**2 + FY**2)
    ) / (1j * wavelength * z)

    # Compute the Fourier transform of the input field
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(field)

    # Apply the transfer function in the Fourier domain
    propagated_ft: Complex[Array, "H W"] = field_ft * H

    # Inverse Fourier transform to get the propagated field
    propagated_field: Complex[Array, "H W"] = jnp.fft.ifft2(propagated_ft)

    # Return the propagated field
    return propagated_field