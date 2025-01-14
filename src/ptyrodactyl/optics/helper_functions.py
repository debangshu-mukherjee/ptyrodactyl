import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

import ptyrodactyl.optics as pto

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class GridParams(NamedTuple):
    """
    Description
    -----------
    PyTree structure for computational grid parameters

    Attributes
    ----------
    - `X` (Float[Array, "H W"]):
        Spatial grid in the x-direction
    - `Y` (Float[Array, "H W"]):
        Spatial grid in the y-direction
    - `phase_profile` (Float[Array, "H W"]):
        Phase profile of the optical field
    - `transmission` (Float[Array, "H W"]):
        Transmission profile of the optical field

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX transformations
    like jit, grad, and vmap. The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    phase_profile: Float[Array, "H W"]
    transmission: Float[Array, "H W"]

    def tree_flatten(self):
        # Return a tuple of arrays (the children) and None (the auxiliary data)
        return (
            (
                self.X,
                self.Y,
                self.phase_profile,
                self.transmission,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the NamedTuple from flattened data
        return cls(*children)


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
    # Get the shape of the input field
    ny: Int[Array, ""] = field.shape[0]
    nx: Int[Array, ""] = field.shape[1]

    # Calculate the wavenumber
    k: Float[Array, ""] = 2 * jnp.pi / wavelength  # Wavenumber

    # Spatial frequency coordinates
    fx: Float[Array, "H"] = jnp.fft.fftfreq(nx, d=dx)
    fy: Float[Array, "W"] = jnp.fft.fftfreq(ny, d=dx)
    FX: Float[Array, "H W"]
    FY: Float[Array, "H W"]
    FX, FY = jnp.meshgrid(fx, fy)

    # Compute the squared spatial frequencies
    FSQ: Float[Array, "H W"] = (FX**2) + (FY**2)

    # Angular spectrum transfer function
    H: Complex[Array, ""] = jnp.exp(1j * k * z * jnp.sqrt(1 - (wavelength**2) * FSQ))

    # Ensure evanescent waves are properly handled
    evanescent_mask: Bool[Array, "H W"] = FSQ <= (1 / wavelength) ** 2
    H_mask: Complex[Array, "H W"] = H * evanescent_mask

    # Fourier transform of the input field
    field_ft: Complex[Array, "H W"] = jnp.fft.fft2(field)

    # Apply the transfer function in the Fourier domain
    propagated_ft: Complex[Array, "H W"] = field_ft * H_mask

    # Inverse Fourier transform to get the propagated field
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
    field_with_phase: Complex[Array, "H W"] = pto.add_phase_screen(
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
    propagated_ft: Complex[Array, "H W"] = pto.add_phase_screen(
        field_ft, transfer_phase
    )

    # Inverse Fourier transform to get the propagated field
    propagated_field: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(propagated_ft))
    )

    # Final quadratic phase factor (post-free-space propagation)
    final_quadratic_phase: Float[Array, "H W"] = k / (2 * z) * (X**2 + Y**2)

    # Apply final quadratic phase factor
    final_propagated_field: Complex[Array, "H W"] = pto.add_phase_screen(
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
