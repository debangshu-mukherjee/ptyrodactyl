"""
Module: photons.lenses
---------------------------
Optics model for simulation of optical lenses.

Functions
---------
- `lens_thickness_profile`:
    Calculates the thickness profile of a lens
- `lens_focal_length`:
    Calculates the focal length of a lens using the lensmaker's equation
- `create_lens_phase`:
    Creates the phase profile and transmission mask for a lens
- `propagate_through_lens`:
    Propagates a field through a lens
- `double_convex_lens`:
    Creates parameters for a double convex lens
- `double_concave_lens`:
    Creates parameters for a double concave lens
- `plano_convex_lens`:
    Creates parameters for a plano-convex lens
- `plano_concave_lens`:
    Creates parameters for a plano-concave lens
- `meniscus_lens`:
    Creates parameters for a meniscus (concavo-convex) lens
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from .helper import add_phase_screen
from .photon_types import (LensParams, make_lens_params, scalar_float,
                           scalar_num)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def lens_thickness_profile(
    r: Float[Array, "H W"],
    R1: Float[Array, ""],
    R2: Float[Array, ""],
    center_thickness: Float[Array, ""],
    diameter: Float[Array, ""],
) -> Float[Array, "H W"]:
    """
    Description
    -----------
    Calculate the thickness profile of a lens.

    Parameters
    ----------
    - `r` (Float[Array, "H W"]):
        Radial distance from the optical axis
    - `R1` (Float[Array, ""]):
        Radius of curvature of the first surface
    - `R2` (Float[Array, ""]):
        Radius of curvature of the second surface
    - `center_thickness` (Float[Array, ""]):
        Thickness at the center of the lens
    - `diameter` (Float[Array, ""]):
        Diameter of the lens

    Returns
    -------
    - `thickness` (Float[Array, "H W"]):
        Thickness profile of the lens

    Flow
    ----
    - Calculate surface sag for both surfaces
    - Combine sags with center thickness
    - Apply aperture mask
    - Return thickness profile
    """
    sag1: Float[Array, "H W"] = jnp.where(
        r <= diameter / 2, R1 - jnp.sqrt(jnp.maximum(R1**2 - r**2, 0.0)), 0.0
    )
    sag2: Float[Array, "H W"] = jnp.where(
        r <= diameter / 2, R2 - jnp.sqrt(jnp.maximum(R2**2 - r**2, 0.0)), 0.0
    )
    thickness: Float[Array, "H W"] = jnp.where(
        r <= diameter / 2, center_thickness + sag1 - sag2, 0.0
    )
    return thickness


@jaxtyped(typechecker=beartype)
def lens_focal_length(
    n: scalar_float,
    R1: scalar_num,
    R2: scalar_num,
) -> scalar_float:
    """
    Description
    -----------
    Calculate the focal length of a lens using the lensmaker's equation.

    Parameters
    ----------
    - `n` (scalar_float):
        Refractive index of the lens material
    - `R1` (scalar_num):
        Radius of curvature of the first surface (positive for convex)
    - `R2` (scalar_num):
        Radius of curvature of the second surface (positive for convex)

    Returns
    -------
    - `f` (scalar_float):
        Focal length of the lens

    Flow
    ----
    - Apply the lensmaker's equation
    - Return the calculated focal length
    """
    f: Float[Array, ""] = 1.0 / ((n - 1.0) * (1.0 / R1 - 1.0 / R2))
    return f


@jaxtyped(typechecker=beartype)
def create_lens_phase(
    X: Float[Array, "H W"],
    Y: Float[Array, "H W"],
    params: LensParams,
    wavelength: scalar_float,
) -> Tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """
    Description
    -----------
    Create the phase profile and transmission mask for a lens.

    Parameters
    ----------
    - `X` (Float[Array, "H W"]):
        X coordinates grid
    - `Y` (Float[Array, "H W"]):
        Y coordinates grid
    - `params` (LensParams):
        Lens parameters
    - `wavelength` (scalar_float):
        Wavelength of light

    Returns
    -------
    - `phase_profile` (Float[Array, "H W"]):
        Phase profile of the lens
    - `transmission` (Float[Array, "H W"]):
        Transmission mask of the lens

    Flow
    ----
    - Calculate radial coordinates
    - Calculate thickness profile
    - Calculate phase profile
    - Create transmission mask
    - Return phase and transmission
    """
    r: Float[Array, "H W"] = jnp.sqrt(X**2 + Y**2)
    thickness: Float[Array, "H W"] = lens_thickness_profile(
        r, params.R1, params.R2, params.center_thickness, params.diameter
    )
    k: Float[Array, ""] = 2 * jnp.pi / wavelength
    phase_profile: Float[Array, "H W"] = k * (params.n - 1) * thickness
    transmission: Float[Array, "H W"] = (r <= params.diameter / 2).astype(float)
    return (phase_profile, transmission)


@jaxtyped(typechecker=beartype)
def propagate_through_lens(
    field: Complex[Array, "H W"],
    phase_profile: Float[Array, "H W"],
    transmission: Float[Array, "H W"],
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Propagate a field through a lens.

    Parameters
    ----------
    - `field` (Complex[Array, "H W"]):
        Input complex field
    - `phase_profile` (Float[Array, "H W"]):
        Phase profile of the lens
    - `transmission` (Float[Array, "H W"]):
        Transmission mask of the lens

    Returns
    -------
    - `output_field` (Complex[Array, "H W"]):
        Field after passing through the lens

    Flow
    ----
    - Apply transmission mask
    - Add phase profile
    - Return modified field
    """
    output_field: Complex[Array, "H W"] = add_phase_screen(
        field * transmission, phase_profile
    )
    return output_field


@jaxtyped(typechecker=beartype)
def double_convex_lens(
    focal_length: Float[Array, ""],
    diameter: Float[Array, ""],
    n: Float[Array, ""],
    center_thickness: Float[Array, ""],
    R_ratio: Optional[Float[Array, ""]] = jnp.array(1.0),
) -> LensParams:
    """
    Description
    -----------
    Create parameters for a double convex lens.

    Parameters
    ----------
    - `focal_length` (Float[Array, ""]):
        Desired focal length
    - `diameter` (Float[Array, ""]):
        Lens diameter
    - `n` (Float[Array, ""]):
        Refractive index
    - `center_thickness` (Float[Array, ""]):
        Center thickness
    - `R_ratio` (Optional[Float[Array, ""]]):
        Ratio of R2/R1.
        default is 1.0 for symmetric lens

    Returns
    -------
    - `params` (LensParams):
        Lens parameters

    Flow
    ----
    - Calculate R1 using lensmaker's equation
    - Calculate R2 using R_ratio
    - Create and return LensParams
    """
    R1: Float[Array, ""] = focal_length * (n - 1) * (1 + R_ratio) / 2
    R2: Float[Array, ""] = R1 * R_ratio
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )
    return params


def double_concave_lens(
    focal_length: Float[Array, ""],
    diameter: Float[Array, ""],
    n: Float[Array, ""],
    center_thickness: Float[Array, ""],
    R_ratio: Optional[Float[Array, ""]] = jnp.array(1.0),
) -> LensParams:
    """
    Description
    -----------
    Create parameters for a double concave lens.

    Parameters
    ----------
    - `focal_length` (Float[Array, ""]):
        Desired focal length
    - `diameter` (Float[Array, ""]):
        Lens diameter
    - `n` (Float[Array, ""]):
        Refractive index
    - `center_thickness` (Float[Array, ""]):
        Center thickness
    - `R_ratio` (Optional[Float[Array, ""]]):
        Ratio of R2/R1.
        default is 1.0 for symmetric lens

    Returns
    -------
    - `params` (LensParams):
        Lens parameters

    Flow
    ----
    - Calculate R1 using lensmaker's equation
    - Calculate R2 using R_ratio
    - Create and return LensParams
    """
    R1: Float[Array, ""] = focal_length * (n - 1) * (1 + R_ratio) / 2
    R2: Float[Array, ""] = R1 * R_ratio
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=-abs(R1),
        R2=-abs(R2),
    )
    return params


@jaxtyped(typechecker=beartype)
def plano_convex_lens(
    focal_length: Float[Array, ""],
    diameter: Float[Array, ""],
    n: Float[Array, ""],
    center_thickness: Float[Array, ""],
    convex_first: Optional[Bool[Array, ""]] = jnp.array(True),
) -> LensParams:
    """
    Description
    -----------
    Create parameters for a plano-convex lens.

    Parameters
    ----------
    - `focal_length` (Float[Array, ""]):
        Desired focal length
    - `diameter` (Float[Array, ""]):
        Lens diameter
    - `n` (Float[Array, ""]):
        Refractive index
    - `center_thickness` (Float[Array, ""]):
        Center thickness
    - `R_ratio` (Optional[Float[Array, ""]]):
        Ratio of R2/R1.
        default is 1.0 for symmetric lens
    - `convex_first` (Optional[Bool[Array, ""]]):
        If True, first surface is convex.
        Default: True

    Returns
    -------
    - `params` (LensParams):
        Lens parameters

    Flow
    ----
    - Calculate R for curved surface
    - Set other R to infinity (flat surface)
    - Create and return LensParams
    """
    R: Float[Array, ""] = focal_length * (n - 1)
    R1: Float[Array, ""] = jnp.where(convex_first, R, jnp.inf)
    R2: Float[Array, ""] = jnp.where(convex_first, jnp.inf, R)
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )
    return params


@jaxtyped(typechecker=beartype)
def plano_concave_lens(
    focal_length: Float[Array, ""],
    diameter: Float[Array, ""],
    n: Float[Array, ""],
    center_thickness: Float[Array, ""],
    concave_first: Optional[Bool[Array, ""]] = jnp.array(True),
) -> LensParams:
    """
    Description
    -----------
    Create parameters for a plano-concave lens.

    Parameters
    ----------
    - `focal_length` (Float[Array, ""]):
        Desired focal length
    - `diameter` (Float[Array, ""]):
        Lens diameter
    - `n` (Float[Array, ""]):
        Refractive index
    - `center_thickness` (Float[Array, ""]):
        Center thickness
    - `R_ratio` (Optional[Float[Array, ""]]):
        Ratio of R2/R1.
        default is 1.0 for symmetric lens
    - `concave_first` (Optional[Bool[Array, ""]]):
        If True, first surface is concave (default: True)

    Returns
    -------
    - `params` (LensParams):
        Lens parameters

    Flow
    ----
    - Calculate R for curved surface
    - Set other R to infinity (flat surface)
    - Create and return LensParams
    """
    R: Float[Array, ""] = -abs(focal_length * (n - 1))
    R1: Float[Array, ""] = jnp.where(concave_first, R, jnp.inf)
    R2: Float[Array, ""] = jnp.where(concave_first, jnp.inf, R)
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )
    return params


@jaxtyped(typechecker=beartype)
def meniscus_lens(
    focal_length: Float[Array, ""],
    diameter: Float[Array, ""],
    n: Float[Array, ""],
    center_thickness: Float[Array, ""],
    R_ratio: Float[Array, ""],
    convex_first: Optional[Bool[Array, ""]] = jnp.array(True),
) -> LensParams:
    """
    Description
    -----------
    Create parameters for a meniscus (concavo-convex) lens.
    For a meniscus lens, one surface is convex (positive R)
    and one is concave (negative R).

    Parameters
    ----------
    - `focal_length` (Float[Array, ""]):
        Desired focal length in meters
    - `diameter` (Float[Array, ""]):
        Lens diameter in meters
    - `n` (Float[Array, ""]):
        Refractive index of lens material
    - `center_thickness` (Float[Array, ""]):
        Center thickness in meters
    - `R_ratio` (Float[Array, ""]):
        Absolute ratio of R2/R1
    - `convex_first` (Bool[Array, ""]):
        If True, first surface is convex (default: True)

    Returns
    -------
    - `params` (LensParams):
        Lens parameters

    Flow
    ----
    - Calculate magnitude of R1 using lensmaker's equation
    - Calculate R2 magnitude using R_ratio
    - Assign correct signs based on convex_first
    - Create and return LensParams
    """
    R1_mag: Float[Array, ""] = (
        focal_length * (n - 1) * (1 - R_ratio) / (1 if convex_first else -1)
    )
    R2_mag: Float[Array, ""] = abs(R1_mag * R_ratio)
    R1: Float[Array, ""] = jnp.where(
        convex_first,
        abs(R1_mag),
        -abs(R1_mag),
    )
    R2: Float[Array, ""] = jnp.where(
        convex_first,
        -abs(R2_mag),
        abs(R2_mag),
    )
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )
    return params
