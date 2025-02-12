import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

import ptyrodactyl.optics as pto

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class LensParams(NamedTuple):
    """
    Description
    -----------
    PyTree structure for lens parameters

    Attributes
    ----------
    - `focal_length` (Float[Array, ""]):
        Focal length of the lens in meters
    - `diameter` (Float[Array, ""]):
        Diameter of the lens in meters
    - `n` (Float[Array, ""]):
        Refractive index of the lens material
    - `center_thickness` (Float[Array, ""]):
        Thickness at the center of the lens in meters
    - `R1` (Float[Array, ""]):
        Radius of curvature of the first surface in meters (positive for convex)
    - `R2` (Float[Array, ""]):
        Radius of curvature of the second surface in meters (positive for convex)

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX transformations
    like jit, grad, and vmap. The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    focal_length: Float[Array, ""]
    diameter: Float[Array, ""]
    n: Float[Array, ""]
    center_thickness: Float[Array, ""]
    R1: Float[Array, ""]
    R2: Float[Array, ""]

    def tree_flatten(self):
        # Return a tuple of arrays (the children) and None (the auxiliary data)
        return (
            (
                self.focal_length,
                self.diameter,
                self.n,
                self.center_thickness,
                self.R1,
                self.R2,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the NamedTuple from flattened data
        return cls(*children)


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
    # Calculate surface sags
    sag1: Float[Array, "H W"] = jnp.where(
        r <= diameter / 2, R1 - jnp.sqrt(jnp.maximum(R1**2 - r**2, 0.0)), 0.0
    )

    sag2: Float[Array, "H W"] = jnp.where(
        r <= diameter / 2, R2 - jnp.sqrt(jnp.maximum(R2**2 - r**2, 0.0)), 0.0
    )

    # Calculate total thickness profile
    thickness: Float[Array, "H W"] = jnp.where(
        r <= diameter / 2, center_thickness + sag1 - sag2, 0.0
    )

    return thickness


@jaxtyped(typechecker=beartype)
def lens_focal_length(
    n: Float[Array, ""],
    R1: Float[Array, ""],
    R2: Float[Array, ""],
) -> Float[Array, ""]:
    """
    Description
    -----------
    Calculate the focal length of a lens using the lensmaker's equation.

    Parameters
    ----------
    - `n` (Float[Array, ""]):
        Refractive index of the lens material
    - `R1` (Float[Array, ""]):
        Radius of curvature of the first surface (positive for convex)
    - `R2` (Float[Array, ""]):
        Radius of curvature of the second surface (positive for convex)

    Returns
    -------
    - `f` (Float[Array, ""]):
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
    wavelength: Float[Array, ""],
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
    - `wavelength` (Float[Array, ""]):
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
    # Calculate radial coordinates
    r: Float[Array, "H W"] = jnp.sqrt(X**2 + Y**2)

    # Calculate thickness profile
    thickness: Float[Array, "H W"] = pto.calculate_thickness_profile(
        r, params.R1, params.R2, params.center_thickness, params.diameter
    )

    # Calculate phase profile
    k: Float[Array, ""] = 2 * jnp.pi / wavelength
    phase_profile: Float[Array, "H W"] = k * (params.n - 1) * thickness

    # Create transmission mask
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
    output_field: Complex[Array, "H W"] = pto.add_phase_screen(
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
    # For a double convex lens, both R1 and R2 are positive
    R1: Float[Array, ""] = focal_length * (n - 1) * (1 + R_ratio) / 2
    R2: Float[Array, ""] = R1 * R_ratio

    return LensParams(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )


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
    # For a double concave lens, both R1 and R2 are negative
    R1: Float[Array, ""] = focal_length * (n - 1) * (1 + R_ratio) / 2
    R2: Float[Array, ""] = R1 * R_ratio

    return LensParams(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=-abs(R1),  # Ensure negative
        R2=-abs(R2),  # Ensure negative
    )


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

    # Assign R to first or second surface based on convex_first
    R1: Float[Array, ""] = jnp.where(convex_first, R, jnp.inf)
    R2: Float[Array, ""] = jnp.where(convex_first, jnp.inf, R)

    return LensParams(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )


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

    # Assign R to first or second surface based on concave_first
    R1: Float[Array, ""] = jnp.where(concave_first, R, jnp.inf)
    R2: Float[Array, ""] = jnp.where(concave_first, jnp.inf, R)

    return LensParams(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )


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
    # Calculate absolute values of radii
    # Using lensmaker's equation: 1/f = (n-1)(1/R1 - 1/R2)
    R1_mag: Float[Array, ""] = (
        focal_length * (n - 1) * (1 - R_ratio) / (1 if convex_first else -1)
    )
    R2_mag: Float[Array, ""] = abs(R1_mag * R_ratio)

    # Assign signs based on which surface is convex
    R1: Float[Array, ""] = jnp.where(
        convex_first,
        abs(R1_mag),  # Convex first surface (positive)
        -abs(R1_mag),  # Concave first surface (negative)
    )

    R2: Float[Array, ""] = jnp.where(
        convex_first,
        -abs(R2_mag),  # Concave second surface (negative)
        abs(R2_mag),  # Convex second surface (positive)
    )

    return LensParams(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )
