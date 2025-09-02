"""
Module: ptyrodactyl.photons.apertures
-------------------------------------
Aperture and apodizer elements for shaping optical wavefronts.

Functions
---------
- `circular_aperture`:
    Applies a circular aperture (optionally offset) with uniform transmittivity.
- `rectangular_aperture`:
    Applies an axis-aligned rectangular aperture with uniform transmittivity.
- `annular_aperture`:
    Applies a concentric ring (donut) aperture between inner/outer diameters.
- `variable_transmission_aperture`:
    Applies an arbitrary transmission mask (array or callable), including
    common apodizers such as Gaussian or super-Gaussian.
- `gaussian_apodizer`:
    Applies a Gaussian apodizer (smooth transmission mask) to the wavefront.
- `supergaussian_apodizer`:
    Applies a super-Gaussian apodizer (smooth transmission mask) to the wavefront.
- `gaussian_apodizer_elliptical`:
    Applies a Gaussian apodizer (smooth transmission mask) to the wavefront.
- `supergaussian_apodizer_elliptical`:
    Applies a super-Gaussian apodizer (smooth transmission mask) to the wavefront.
"""

import jax
import jax.numpy as jnp
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Bool, Float

from ptyrodactyl._decorators import beartype, jaxtyped

from .photon_types import (OpticalWavefront, make_optical_wavefront,
                           scalar_float, scalar_numeric)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def _xy_grids(
    nx: int, ny: int, dx: float
) -> Tuple[Float[Array, " H W"], Float[Array, " H W"]]:
    """
    Internal helper to create centered spatial coordinate grids (in meters).

    Parameters
    ----------
    nx : int
        Number of grid points along x-axis
    ny : int
        Number of grid points along y-axis
    dx : float
        Grid spacing in meters

    Returns
    -------
    X : Float[Array, "H W"]
        X coordinate grid in meters
    Y : Float[Array, "H W"]
        Y coordinate grid in meters
    """
    x: Float[Array, " W"] = jnp.arange(-nx // 2, nx // 2) * dx
    y: Float[Array, " H"] = jnp.arange(-ny // 2, ny // 2) * dx
    xx: Float[Array, "H W"]
    yy: Float[Array, "H W"]
    xx, yy = jnp.meshgrid(x, y)
    return (xx, yy)


@jaxtyped(typechecker=beartype)
def circular_aperture(
    incoming: OpticalWavefront,
    diameter: scalar_float,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply a circular aperture to the incoming wavefront.
    The aperture is defined by its physical diameter and (optional) center.

    Parameters
    ----------
    incoming : OpticalWavefront
        PyTree with:

        - field : Complex[Array, "H W"]
            Complex input field
        - wavelength : Float[Array, ""]
            Wavelength in meters
        - dx : Float[Array, ""]
            Pixel size in meters
        - z_position : Float[Array, ""]
            Axial position in meters
    diameter : scalar_float
        Aperture diameter in meters
    center : Optional[Float[Array, " 2"]], optional
        Physical center [x0, y0] of the aperture in meters, by default [0, 0]
    transmittivity : Optional[scalar_float], optional
        Uniform transmittivity inside the aperture (0..1), by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying the circular aperture

    Notes
    -----
    - Build centered (x, y) grids in meters
    - Compute radial distance from the specified center
    - Create a binary mask for r <= diameter/2
    - Multiply by transmittivity (clipped to [0, 1])
    - Apply to the complex field and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    r: Float[Array, " H W"] = jnp.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    inside: Bool[Array, " H W"] = r <= (diameter / 2.0)
    t = jnp.clip(jnp.asarray(transmittivity, dtype=float), 0.0, 1.0)
    transmission: Float[Array, " H W"] = inside.astype(float) * t
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * transmission,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def rectangular_aperture(
    incoming: OpticalWavefront,
    width: scalar_float,
    height: scalar_float,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply an axis-aligned rectangular aperture to the incoming wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree (see `circular_aperture`)
    width : scalar_float
        Rectangle width along x in meters
    height : scalar_float
        Rectangle height along y in meters
    center : Optional[Float[Array, " 2"]], optional
        Rectangle center [x0, y0] in meters, by default [0, 0]
    transmittivity : Optional[scalar_float], optional
        Uniform transmittivity inside the rectangle (0..1), by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying the rectangular aperture

    Notes
    -----
    - Build centered (x, y) grids in meters
    - Compute half-width/half-height and an inside-rectangle mask
    - Multiply by transmittivity (clipped)
    - Apply to the complex field and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    hx = width / 2.0
    hy = height / 2.0
    inside_x: Bool[Array, " H W"] = ((x0 - hx) <= xx) & ((x0 + hx) >= xx)
    inside_y: Bool[Array, " H W"] = ((y0 - hy) <= yy) & ((y0 + hy) >= yy)
    inside: Bool[Array, " H W"] = inside_x & inside_y
    t = jnp.clip(jnp.asarray(transmittivity, dtype=float), 0.0, 1.0)
    transmission: Float[Array, " H W"] = inside.astype(float) * t
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * transmission,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def annular_aperture(
    incoming: OpticalWavefront,
    inner_diameter: scalar_float,
    outer_diameter: scalar_float,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply an annular (ring) aperture with inner and outer diameters.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree (see `circular_aperture`)
    inner_diameter : scalar_float
        Inner blocked diameter in meters
    outer_diameter : scalar_float
        Outer clear aperture diameter in meters
    center : Optional[Float[Array, " 2"]], optional
        Ring center [x0, y0] in meters, by default [0, 0]
    transmittivity : Optional[scalar_float], optional
        Uniform transmittivity in the ring (0..1), by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying the annular aperture

    Notes
    -----
    - Build centered (x, y) grids in meters
    - Compute radial distance from center
    - Create mask for inner_radius < r <= outer_radius
    - Multiply by transmittivity (clipped), apply, and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    r: Float[Array, " H W"] = jnp.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    r_in = inner_diameter / 2.0
    r_out = outer_diameter / 2.0
    ring: Bool[Array, " H W"] = (r > r_in) & (r <= r_out)
    t = jnp.clip(jnp.asarray(transmittivity, dtype=float), 0.0, 1.0)
    transmission: Float[Array, " H W"] = ring.astype(float) * t
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * transmission,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def variable_transmission_aperture(
    incoming: OpticalWavefront,
    transmission: Union[scalar_float, Float[Array, " ..."]],
) -> OpticalWavefront:
    """
    Apply an arbitrary (spatially varying) transmission to the wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront PyTree
    transmission : Union[scalar_float, Float[Array, " H W"]]
        Precomputed transmission map (0..1) with shape "H W", or a scalar
        attenuation factor for uniform transmission

    Returns
    -------
    OpticalWavefront
        Wavefront after applying the transmission

    Examples
    --------
    Uniform attenuation::
    >>> wf2 = variable_transmission_aperture(wf, 0.5)  # 50% transmission

    Spatially varying transmission::
    >>> tmap = create_transmission_map(...)  # Shape (H, W)
    >>> wf2 = variable_transmission_aperture(wf, tmap)

    Notes
    -----
    - For scalar transmission: applies uniform attenuation
    - For array transmission: applies spatially varying transmission map
    - Transmission values are clipped to [0, 1]
    - This function is fully JAX-compatible and uses jax.lax.cond
    """
    trans = jnp.asarray(transmission, dtype=float)

    def apply_scalar_transmission() -> OpticalWavefront:
        t = jnp.clip(trans, 0.0, 1.0)
        return make_optical_wavefront(
            field=incoming.field * t,
            wavelength=incoming.wavelength,
            dx=incoming.dx,
            z_position=incoming.z_position,
        )

    def apply_array_transmission() -> OpticalWavefront:
        tmap = jnp.clip(trans, 0.0, 1.0)
        return make_optical_wavefront(
            field=incoming.field * tmap,
            wavelength=incoming.wavelength,
            dx=incoming.dx,
            z_position=incoming.z_position,
        )

    return jax.lax.cond(
        trans.ndim == 0, apply_scalar_transmission, apply_array_transmission
    )


@jaxtyped(typechecker=beartype)
def gaussian_apodizer(
    incoming: OpticalWavefront,
    sigma: scalar_float,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    peak_transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply a Gaussian apodizer (smooth transmission mask) to the wavefront.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront
    sigma : scalar_float
        Gaussian width parameter in meters
    center : Optional[Float[Array, " 2"]], optional
        Physical center [x0, y0] of the Gaussian in meters, by default [0, 0]
    peak_transmittivity : Optional[scalar_float], optional
        Maximum transmission at the Gaussian center, by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying Gaussian apodization

    Notes
    -----
    - Build centered (x, y) grids
    - Compute squared radial distance from center
    - Evaluate Gaussian exp(-r^2 / (2*sigma^2))
    - Scale by peak transmittivity, clip to [0,1]
    - Multiply with incoming field and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    r2: Float[Array, " H W"] = (xx - x0) ** 2 + (yy - y0) ** 2
    gauss: Float[Array, " H W"] = jnp.exp(-r2 / (2.0 * sigma**2))
    tmap: Float[Array, " H W"] = jnp.clip(gauss * peak_transmittivity, 0.0, 1.0)
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def supergaussian_apodizer(
    incoming: OpticalWavefront,
    sigma: scalar_float,
    m: scalar_numeric,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    peak_transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply a super-Gaussian apodizer to the wavefront.
    Transmission profile: exp(- (r^2 / sigma^2)^m ).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront
    sigma : scalar_float
        Width parameter in meters (sets the roll-off scale)
    m : scalar_numeric
        Super-Gaussian order (m=1 → Gaussian, m>1 → flatter top)
    center : Optional[Float[Array, " 2"]], optional
        Physical center [x0, y0] of the profile, by default [0, 0]
    peak_transmittivity : Optional[scalar_float], optional
        Maximum transmission at the center, by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying super-Gaussian apodization

    Notes
    -----
    - Build centered (x, y) grids
    - Compute squared radial distance from center
    - Evaluate exp(- (r^2 / sigma^2)^m )
    - Scale by peak transmittivity, clip to [0,1]
    - Multiply with incoming field and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    r2: Float[Array, " H W"] = (xx - x0) ** 2 + (yy - y0) ** 2
    super_gauss: Float[Array, " H W"] = jnp.exp(-((r2 / (sigma**2)) ** m))
    tmap: Float[Array, " H W"] = jnp.clip(super_gauss * peak_transmittivity, 0.0, 1.0)
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def gaussian_apodizer_elliptical(
    incoming: OpticalWavefront,
    sigma_x: scalar_float,
    sigma_y: scalar_float,
    theta: Optional[scalar_float] = 0.0,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    peak_transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply an elliptical Gaussian apodizer to the wavefront, with optional rotation.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront
    sigma_x : scalar_float
        Gaussian width along the x'-axis (meters) after rotation by `theta`
    sigma_y : scalar_float
        Gaussian width along the y'-axis (meters) after rotation by `theta`
    theta : Optional[scalar_float], optional
        Rotation angle in radians (counter-clockwise), by default 0.0
    center : Optional[Float[Array, " 2"]], optional
        Physical center [x0, y0] in meters, by default [0, 0]
    peak_transmittivity : Optional[scalar_float], optional
        Maximum transmission at the center, by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying elliptical Gaussian apodization

    Notes
    -----
    - Build centered (x, y) grids
    - Translate by `center`, rotate by `theta` → (x', y')
    - Evaluate exp(-0.5 * ( (x'/sigma_x)^2 + (y'/sigma_y)^2 ))
    - Scale by `peak_transmittivity`, clip to [0, 1]
    - Multiply with incoming field and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    xc = xx - x0
    yc = yy - y0
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    xp = ct * xc + st * yc
    yp = -st * xc + ct * yc
    arg = (xp / sigma_x) ** 2 + (yp / sigma_y) ** 2
    gauss = jnp.exp(-0.5 * arg)
    tmap = jnp.clip(gauss * peak_transmittivity, 0.0, 1.0)
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured


@jaxtyped(typechecker=beartype)
def supergaussian_apodizer_elliptical(
    incoming: OpticalWavefront,
    sigma_x: scalar_float,
    sigma_y: scalar_float,
    m: scalar_numeric,
    theta: Optional[scalar_float] = 0.0,
    center: Optional[Float[Array, " 2"]] = jnp.zeros(2),
    peak_transmittivity: Optional[scalar_float] = 1.0,
) -> OpticalWavefront:
    """
    Apply an elliptical super-Gaussian apodizer with optional rotation.
    Transmission profile: exp( - ( (x'/sigma_x)^2 + (y'/sigma_y)^2 )^m ).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input optical wavefront
    sigma_x : scalar_float
        Width along x' (meters) after rotation by `theta`
    sigma_y : scalar_float
        Width along y' (meters) after rotation by `theta`
    m : scalar_numeric
        Super-Gaussian order (m=1 → Gaussian; m>1 → flatter top, sharper edges)
    theta : Optional[scalar_float], optional
        Rotation angle in radians (counter-clockwise), by default 0.0
    center : Optional[Float[Array, " 2"]], optional
        Physical center [x0, y0] in meters, by default [0, 0]
    peak_transmittivity : Optional[scalar_float], optional
        Maximum transmission at the center, by default 1.0

    Returns
    -------
    OpticalWavefront
        Wavefront after applying elliptical super-Gaussian apodization

    Notes
    -----
    - Build centered (x, y) grids
    - Translate by `center`, rotate by `theta` → (x', y')
    - Evaluate exp( - ( (x'/sigma_x)^2 + (y'/sigma_y)^2 )^m )
    - Scale by `peak_transmittivity`, clip to [0, 1]
    - Multiply with incoming field and return
    """
    ny: int = incoming.field.shape[0]
    nx: int = incoming.field.shape[1]
    xx, yy = _xy_grids(nx, ny, float(incoming.dx))
    x0, y0 = center[0], center[1]
    xc = xx - x0
    yc = yy - y0
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    xp = ct * xc + st * yc
    yp = -st * xc + ct * yc
    base = (xp / sigma_x) ** 2 + (yp / sigma_y) ** 2
    super_gauss = jnp.exp(-(base**m))
    tmap = jnp.clip(super_gauss * peak_transmittivity, 0.0, 1.0)
    apertured: OpticalWavefront = make_optical_wavefront(
        field=incoming.field * tmap,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return apertured
