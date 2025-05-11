"""
Module: photons.photon_types
----------------------------
Data structures and type definitions for optical ptychography.

Type Aliases
------------
- `scalar_float`: 
    Type alias for float or Float array of 0 dimensions
- `scalar_int`: 
    Type alias for int or Integer array of 0 dimensions
- `scalar_num`: 
    Type alias for numeric types (int, float or Num array)
    Num Array has 0 dimensions
- `non_jax_number`: 
    Type alias for non-JAX numeric types (int, float)

Classes
-------
- `LensParams`: 
    A named tuple for lens parameters
- `GridParams`: 
    A named tuple for computational grid parameters
- `OpticalWavefront`: 
    A named tuple for representing an optical wavefront
- `MicroscopeData`:
    A named tuple for storing 3D or 4D microscope image data
- `Diffractogram`:
    A named tuple for storing a single diffraction pattern

Factory Functions
----------------
- `make_lens_params`: 
    Creates a LensParams instance with runtime type checking
- `make_grid_params`: 
    Creates a GridParams instance with runtime type checking
- `make_optical_wavefront`: 
    Creates an OpticalWavefront instance with runtime type checking
- `make_microscope_data`:
    Creates a MicroscopeData instance with runtime type checking
- `make_diffractogram`:
    Creates a Diffractogram instance with runtime type checking

    Note: Always use these factory functions instead of directly instantiating the 
    NamedTuple classes to ensure proper runtime type checking of the contents.
"""

from beartype import beartype
from beartype.typing import NamedTuple, TypeAlias, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Complex, Float, Integer, Num, jaxtyped

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]


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
        return cls(*children)


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
    This class is registered as a PyTree node, making it
    compatible with JAX transformations like jit, grad, and vmap.
    The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    phase_profile: Float[Array, "H W"]
    transmission: Float[Array, "H W"]

    def tree_flatten(self):
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
        return cls(*children)


@register_pytree_node_class
class OpticalWavefront(NamedTuple):
    """
    Description
    -----------
    PyTree structure for representing an optical wavefront.

    Attributes
    ----------
    - `field` (Complex[Array, "H W"]):
        Complex amplitude of the optical field.
    - `wavelength` (scalar_float):
        Wavelength of the optical wavefront in meters.
    - `dx` (scalar_float):
        Spatial sampling interval (grid spacing) in meters.
    - `z_position` (scalar_float]):
        Axial position of the wavefront along the propagation direction in meters.
    """

    field: Complex[Array, "H W"]
    wavelength: scalar_float
    dx: scalar_float
    z_position: scalar_float

    def tree_flatten(self):
        return (
            (
                self.field,
                self.wavelength,
                self.dx,
                self.z_position,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class MicroscopeData(NamedTuple):
    """
    Description
    -----------
    PyTree structure for representing an 3D or 4D microscope image.

    Attributes
    ----------
    - `image_data` (Float[Array, "P H W"] | Float[Array, "X Y H W"]):
        3D or 4D image data representing the optical field.
    - `wavelength` (scalar_float):
        Wavelength of the optical wavefront in meters.
    - `dx` (scalar_float):
        Spatial sampling interval (grid spacing) in meters.
    """

    image_data: Union[Float[Array, "P H W"], Float[Array, "X Y H W"]]
    wavelength: scalar_float
    dx: scalar_float

    def tree_flatten(self):
        return (
            (
                self.image_data,
                self.wavelength,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class Diffractogram(NamedTuple):
    """
    Description
    -----------
    PyTree structure for representing a single diffractogram.

    Attributes
    ----------
    - `image` (Float[Array, "H W"]):
        Image data.
    - `wavelength` (scalar_float):
        Wavelength of the optical wavefront in meters.
    - `dx` (scalar_float):
        Spatial sampling interval (grid spacing) in meters.
    """

    image: Float[Array, "H W"]
    wavelength: scalar_float
    dx: scalar_float

    def tree_flatten(self):
        return (
            (
                self.image,
                self.wavelength,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_lens_params(
    focal_length: Float[Array, ""],
    diameter: Float[Array, ""],
    n: Float[Array, ""],
    center_thickness: Float[Array, ""],
    R1: Float[Array, ""],
    R2: Float[Array, ""],
) -> LensParams:
    """
    Factory function for LensParams with runtime type-checking.

    Parameters
    ----------
    - `focal_length` : Float[Array, ""]
    - `diameter` : Float[Array, ""]
    - `n` : Float[Array, ""]
    - `center_thickness` : Float[Array, ""]
    - `R1` : Float[Array, ""]
    - `R2` : Float[Array, ""]

    Returns
    -------
    - `LensParams` instance
    """
    return LensParams(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        R1=R1,
        R2=R2,
    )


@jaxtyped(typechecker=beartype)
def make_grid_params(
    X: Float[Array, "H W"],
    Y: Float[Array, "H W"],
    phase_profile: Float[Array, "H W"],
    transmission: Float[Array, "H W"],
) -> GridParams:
    """
    Factory function for GridParams with runtime type-checking.

    Parameters
    ----------
    - `X` : Float[Array, "H W"]
    - `Y` : Float[Array, "H W"]
    - `phase_profile` : Float[Array, "H W"]
    - `transmission` : Float[Array, "H W"]

    Returns
    -------
    - `GridParams` instance
    """
    return GridParams(X=X, Y=Y, phase_profile=phase_profile, transmission=transmission)


@jaxtyped(typechecker=beartype)
def make_optical_wavefront(
    field: Complex[Array, "H W"],
    wavelength: Float[Array, ""],
    dx: Float[Array, ""],
    z_position: Float[Array, ""],
) -> OpticalWavefront:
    """
    Factory function for OpticalWavefront with runtime type-checking.

    Parameters
    ----------
    - `field` : Complex[Array, "H W"]
    - `wavelength` : Float[Array, ""]
    - `dx` : Float[Array, ""]
    - `z_position` : Float[Array, ""]

    Returns
    -------
    - `OpticalWavefront` instance
    """
    return OpticalWavefront(
        field=field, wavelength=wavelength, dx=dx, z_position=z_position
    )

@jaxtyped(typechecker=beartype)
def make_microscope_data(
    image_data: Union[Float[Array, "P H W"], Float[Array, "X Y H W"]],
    wavelength: Float[Array, ""],
    dx: Float[Array, ""],
) -> MicroscopeData:
    """
    Factory function for MicroscopeData with runtime type-checking.

    Parameters
    ----------
    - `image_data` : Union[Float[Array, "P H W"], Float[Array, "X Y H W"]]
    - `wavelength` : Float[Array, ""]
    - `dx` : Float[Array, ""]

    Returns
    -------
    - `MicroscopeData` instance
    """
    return MicroscopeData(image_data=image_data, wavelength=wavelength, dx=dx)

@jaxtyped(typechecker=beartype)
def make_diffractogram(
    image: Float[Array, "H W"],
    wavelength: Float[Array, ""],
    dx: Float[Array, ""],
) -> Diffractogram:
    """
    Factory function for Diffractogram with runtime type-checking.

    Parameters
    ----------
    - `image` : Float[Array, "H W"]
    - `wavelength` : Float[Array, ""]
    - `dx` : Float[Array, ""]

    Returns
    -------
    - `Diffractogram` instance
    """
    return Diffractogram(image=image, wavelength=wavelength, dx=dx)