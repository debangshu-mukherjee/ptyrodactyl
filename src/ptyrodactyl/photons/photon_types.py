"""
Module: ptyrodactyl.photons.photon_types
----------------------------------------
Data structures and type definitions for optical ptychography.

Type Aliases
------------
- `scalar_float`:
    A type alias for float or Float[Array, " "]
- `scalar_integer`:
    A type alias for int or Int[Array, " "]
- `scalar_complex`:
    A type alias for complex or Complex[Array, " "]
- `scalar_numeric`:
    A type alias for int, float, complex or Num[Array, " "]
- `non_jax_number`:
    A type alias for int, float or complex

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
- `SampleFunction`:
    A named tuple for representing a sample function
- `Diffractogram`:
    A named tuple for storing a single diffraction pattern

Factory Functions
-----------------
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
- `make_sample_function`:
    Creates a SampleFunction instance with runtime type checking

Notes
-----
Always use these factory functions instead of directly instantiating the
NamedTuple classes to ensure proper runtime type checking of the contents.
"""

import jax
import jax.numpy as jnp
from beartype.typing import NamedTuple, Tuple, TypeAlias, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Complex, Float, Int, Num

from ptyrodactyl._decorators import beartype, jaxtyped

jax.config.update("jax_enable_x64", True)

scalar_float: TypeAlias = Union[float, Float[Array, " "]]
scalar_integer: TypeAlias = Union[int, Int[Array, " "]]
scalar_complex: TypeAlias = Union[complex, Complex[Array, " "]]
scalar_numeric: TypeAlias = Union[int, float, complex, Num[Array, " "]]
non_jax_number: TypeAlias = Union[int, float, complex]


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class LensParams(NamedTuple):
    """PyTree structure for lens parameters.

    Attributes
    ----------
    focal_length : scalar_float
        Focal length of the lens in meters
    diameter : scalar_float
        Diameter of the lens in meters
    n : scalar_float
        Refractive index of the lens material
    center_thickness : scalar_float
        Thickness at the center of the lens in meters
    r1 : scalar_float
        Radius of curvature of the first surface in meters (positive for convex)
    r2 : scalar_float
        Radius of curvature of the second surface in meters (positive for convex)

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX transformations
    like jit, grad, and vmap. The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    focal_length: scalar_float
    diameter: scalar_float
    n: scalar_float
    center_thickness: scalar_float
    r1: scalar_float
    r2: scalar_float

    def tree_flatten(self):
        return (
            (
                self.focal_length,
                self.diameter,
                self.n,
                self.center_thickness,
                self.r1,
                self.r2,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class GridParams(NamedTuple):
    """PyTree structure for computational grid parameters.

    Attributes
    ----------
    xx : Float[Array, "H W"]
        Spatial grid in the x-direction
    yy : Float[Array, "H W"]
        Spatial grid in the y-direction
    phase_profile : Float[Array, "H W"]
        Phase profile of the optical field
    transmission : Float[Array, "H W"]
        Transmission profile of the optical field

    Notes
    -----
    This class is registered as a PyTree node, making it
    compatible with JAX transformations like jit, grad, and vmap.
    The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    xx: Float[Array, "H W"]
    yy: Float[Array, "H W"]
    phase_profile: Float[Array, "H W"]
    transmission: Float[Array, "H W"]

    def tree_flatten(self):
        return (
            (
                self.xx,
                self.yy,
                self.phase_profile,
                self.transmission,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class OpticalWavefront(NamedTuple):
    """PyTree structure for representing an optical wavefront.

    Attributes
    ----------
    field : Complex[Array, "H W"]
        Complex amplitude of the optical field.
    wavelength : scalar_float
        Wavelength of the optical wavefront in meters.
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters.
    z_position : scalar_float
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


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class MicroscopeData(NamedTuple):
    """PyTree structure for representing an 3D or 4D microscope image.

    Attributes
    ----------
    image_data : Float[Array, "P H W"] | Float[Array, "X Y H W"]
        3D or 4D image data representing the optical field.
    positions : Num[Array, " P 2"]
        Positions of the images during collection.
    wavelength : scalar_float
        Wavelength of the optical wavefront in meters.
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters.
    """

    image_data: Union[Float[Array, "P H W"], Float[Array, "X Y H W"]]
    positions: Num[Array, " P 2"]
    wavelength: scalar_float
    dx: scalar_float

    def tree_flatten(self):
        return (
            (
                self.image_data,
                self.positions,
                self.wavelength,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class SampleFunction(NamedTuple):
    """PyTree structure for representing a sample function.

    Attributes
    ----------
    sample : Complex[Array, "H W"]
        The sample function.
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters.
    """

    sample: Complex[Array, "H W"]
    dx: scalar_float

    def tree_flatten(self):
        return (
            (
                self.sample,
                self.dx,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class Diffractogram(NamedTuple):
    """PyTree structure for representing a single diffractogram.

    Attributes
    ----------
    image : Float[Array, "H W"]
        Image data.
    wavelength : scalar_float
        Wavelength of the optical wavefront in meters.
    dx : scalar_float
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
    focal_length: scalar_float,
    diameter: scalar_float,
    n: scalar_float,
    center_thickness: scalar_float,
    r1: scalar_float,
    r2: scalar_float,
) -> LensParams:
    """JAX-safe factory function for LensParams with data validation.

    Parameters
    ----------
    focal_length : scalar_float
        Focal length of the lens in meters
    diameter : scalar_float
        Diameter of the lens in meters
    n : scalar_float
        Refractive index of the lens material
    center_thickness : scalar_float
        Thickness at the center of the lens in meters
    r1 : scalar_float
        Radius of curvature of the first surface in meters (positive for convex)
    r2 : scalar_float
        Radius of curvature of the second surface in meters (positive for convex)

    Returns
    -------
    LensParams
        Validated lens parameters instance

    Raises
    ------
    ValueError
        If parameters are invalid or out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate parameters:
        - Check focal_length is positive
        - Check diameter is positive
        - Check refractive index is positive
        - Check center_thickness is positive
        - Check radii are finite
    - Create and return LensParams instance
    """
    focal_length = jnp.asarray(focal_length, dtype=jnp.float64)
    diameter = jnp.asarray(diameter, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    center_thickness = jnp.asarray(center_thickness, dtype=jnp.float64)
    r1 = jnp.asarray(r1, dtype=jnp.float64)
    r2 = jnp.asarray(r2, dtype=jnp.float64)

    def validate_and_create():
        def check_focal_length():
            return lax.cond(
                focal_length > 0,
                lambda: focal_length,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: focal_length, lambda: focal_length)
                ),
            )

        def check_diameter():
            return lax.cond(
                diameter > 0,
                lambda: diameter,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: diameter, lambda: diameter)
                ),
            )

        def check_refractive_index():
            return lax.cond(
                n > 0,
                lambda: n,
                lambda: lax.stop_gradient(lax.cond(False, lambda: n, lambda: n)),
            )

        def check_center_thickness():
            return lax.cond(
                center_thickness > 0,
                lambda: center_thickness,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: center_thickness, lambda: center_thickness)
                ),
            )

        def check_radii_finite():
            return lax.cond(
                jnp.logical_and(jnp.isfinite(r1), jnp.isfinite(r2)),
                lambda: (r1, r2),
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: (r1, r2), lambda: (r1, r2))
                ),
            )

        check_focal_length()
        check_diameter()
        check_refractive_index()
        check_center_thickness()
        check_radii_finite()

        return LensParams(
            focal_length=focal_length,
            diameter=diameter,
            n=n,
            center_thickness=center_thickness,
            r1=r1,
            r2=r2,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_grid_params(
    xx: Float[Array, "H W"],
    yy: Float[Array, "H W"],
    phase_profile: Float[Array, "H W"],
    transmission: Float[Array, "H W"],
) -> GridParams:
    """JAX-safe factory function for GridParams with data validation.

    Parameters
    ----------
    xx : Float[Array, "H W"]
        Spatial grid in the x-direction
    yy : Float[Array, "H W"]
        Spatial grid in the y-direction
    phase_profile : Float[Array, "H W"]
        Phase profile of the optical field
    transmission : Float[Array, "H W"]
        Transmission profile of the optical field

    Returns
    -------
    GridParams
        Validated grid parameters instance

    Raises
    ------
    ValueError
        If array shapes are inconsistent or data is invalid

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate array shapes:
        - Check all arrays are 2D
        - Check all arrays have the same shape
    - Validate data:
        - Ensure transmission values are between 0 and 1
        - Ensure phase values are finite
        - Ensure grid coordinates are finite
    - Create and return GridParams instance
    """
    xx = jnp.asarray(xx, dtype=jnp.float64)
    yy = jnp.asarray(yy, dtype=jnp.float64)
    phase_profile = jnp.asarray(phase_profile, dtype=jnp.float64)
    transmission = jnp.asarray(transmission, dtype=jnp.float64)

    def validate_and_create():
        H, W = xx.shape

        def check_2d_arrays():
            return lax.cond(
                jnp.logical_and(
                    jnp.logical_and(xx.ndim == 2, yy.ndim == 2),
                    jnp.logical_and(phase_profile.ndim == 2, transmission.ndim == 2),
                ),
                lambda: (xx, yy, phase_profile, transmission),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (xx, yy, phase_profile, transmission),
                        lambda: (xx, yy, phase_profile, transmission),
                    )
                ),
            )

        def check_same_shape():
            return lax.cond(
                jnp.logical_and(
                    jnp.logical_and(xx.shape == (H, W), yy.shape == (H, W)),
                    jnp.logical_and(
                        phase_profile.shape == (H, W), transmission.shape == (H, W)
                    ),
                ),
                lambda: (xx, yy, phase_profile, transmission),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (xx, yy, phase_profile, transmission),
                        lambda: (xx, yy, phase_profile, transmission),
                    )
                ),
            )

        def check_transmission_range():
            return lax.cond(
                jnp.logical_and(jnp.all(transmission >= 0), jnp.all(transmission <= 1)),
                lambda: transmission,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: transmission, lambda: transmission)
                ),
            )

        def check_phase_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(phase_profile)),
                lambda: phase_profile,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: phase_profile, lambda: phase_profile)
                ),
            )

        def check_grid_finite():
            return lax.cond(
                jnp.logical_and(jnp.all(jnp.isfinite(xx)), jnp.all(jnp.isfinite(yy))),
                lambda: (xx, yy),
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: (xx, yy), lambda: (xx, yy))
                ),
            )

        check_2d_arrays()
        check_same_shape()
        check_transmission_range()
        check_phase_finite()
        check_grid_finite()

        return GridParams(
            xx=xx,
            yy=yy,
            phase_profile=phase_profile,
            transmission=transmission,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_optical_wavefront(
    field: Complex[Array, " hh ww"],
    wavelength: scalar_float,
    dx: scalar_float,
    z_position: scalar_float,
) -> OpticalWavefront:
    """JAX-safe factory function for OpticalWavefront with data validation.

    Parameters
    ----------
    field : Complex[Array, " hh ww"]
        Complex amplitude of the optical field
    wavelength : scalar_float
        Wavelength of the optical wavefront in meters
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters
    z_position : scalar_float
        Axial position of the wavefront along the propagation direction in meters

    Returns
    -------
    OpticalWavefront
        Validated optical wavefront instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate field array:
        - Check it's 2D
        - Ensure all values are finite
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
        - Check z_position is finite
    - Create and return OpticalWavefront instance
    """
    field: Complex[Array, " hh ww"] = jnp.asarray(field, dtype=jnp.complex128)
    wavelength: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    z_position: Float[Array, " "] = jnp.asarray(z_position, dtype=jnp.float64)

    def validate_and_create():
        def check_2d_field():
            return lax.cond(
                field.ndim == 2,
                lambda: field,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: field, lambda: field)
                ),
            )

        def check_field_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(field)),
                lambda: field,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: field, lambda: field)
                ),
            )

        def check_wavelength():
            return lax.cond(
                wavelength > 0,
                lambda: wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelength, lambda: wavelength)
                ),
            )

        def check_dx():
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(lax.cond(False, lambda: dx, lambda: dx)),
            )

        def check_z_position():
            return lax.cond(
                jnp.isfinite(z_position),
                lambda: z_position,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: z_position, lambda: z_position)
                ),
            )

        check_2d_field()
        check_field_finite()
        check_wavelength()
        check_dx()
        check_z_position()

        return OpticalWavefront(
            field=field,
            wavelength=wavelength,
            dx=dx,
            z_position=z_position,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_microscope_data(
    image_data: Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]],
    positions: Num[Array, " pp 2"],
    wavelength: scalar_float,
    dx: scalar_float,
) -> MicroscopeData:
    """JAX-safe factory function for MicroscopeData with data validation.

    Parameters
    ----------
    image_data : Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]]
        3D or 4D image data representing the optical field
    positions : Num[Array, " pp 2"]
        Positions of the images during collection
    wavelength : scalar_float
        Wavelength of the optical wavefront in meters
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters

    Returns
    -------
    MicroscopeData
        Validated microscope data instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate image_data:
        - Check it's 3D or 4D
        - Ensure all values are finite and non-negative
    - Validate positions:
        - Check it's 2D with shape (pp, 2)
        - Ensure all values are finite
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
    - Validate consistency:
        - Check P matches between image_data and positions
    - Create and return MicroscopeData instance
    """
    image_data: Union[Float[Array, " pp hh ww"], Float[Array, " xx yy hh ww"]] = (
        jnp.asarray(image_data, dtype=jnp.float64)
    )
    positions: Num[Array, " pp 2"] = jnp.asarray(positions, dtype=jnp.float64)
    wavelength: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    expected_image_dim = 2
    expected_diffractogram_dim_3d: int = 3
    expected_diffractogram_dim_4d: int = 4

    def validate_and_create() -> MicroscopeData:
        def check_image_dimensions() -> (
            Union[Float[Array, "P H W"], Float[Array, "X Y H W"]]
        ):
            return lax.cond(
                jnp.logical_or(
                    image_data.ndim == expected_diffractogram_dim_3d,
                    image_data.ndim == expected_diffractogram_dim_4d,
                ),
                lambda: image_data,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image_data, lambda: image_data)
                ),
            )

        def check_image_finite() -> (
            Union[Float[Array, "P H W"], Float[Array, "X Y H W"]]
        ):
            return lax.cond(
                jnp.all(jnp.isfinite(image_data)),
                lambda: image_data,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image_data, lambda: image_data)
                ),
            )

        def check_image_nonnegative() -> (
            Union[Float[Array, "P H W"], Float[Array, "X Y H W"]]
        ):
            return lax.cond(
                jnp.all(image_data >= 0),
                lambda: image_data,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image_data, lambda: image_data)
                ),
            )

        def check_positions_shape() -> Num[Array, " P 2"]:
            return lax.cond(
                positions.shape[1] == expected_image_dim,
                lambda: positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: positions, lambda: positions)
                ),
            )

        def check_positions_finite() -> Num[Array, " P 2"]:
            return lax.cond(
                jnp.all(jnp.isfinite(positions)),
                lambda: positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: positions, lambda: positions)
                ),
            )

        def check_wavelength() -> Float[Array, " "]:
            return lax.cond(
                wavelength > 0,
                lambda: wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelength, lambda: wavelength)
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(lax.cond(False, lambda: dx, lambda: dx)),
            )

        def check_consistency() -> Tuple[
            Union[Float[Array, "P H W"], Float[Array, "X Y H W"]],
            Num[Array, " P 2"],
        ]:
            pp = positions.shape[0]

            def check_3d_consistency() -> Tuple[
                Union[Float[Array, "pp H W"], Float[Array, "X Y H W"]],
                Num[Array, " pp 2"],
            ]:
                return lax.cond(
                    image_data.shape[0] == pp,
                    lambda: (image_data, positions),
                    lambda: lax.stop_gradient(
                        lax.cond(
                            False,
                            lambda: (image_data, positions),
                            lambda: (image_data, positions),
                        )
                    ),
                )

            def check_4d_consistency() -> Tuple[
                Union[Float[Array, "P H W"], Float[Array, "X Y H W"]],
                Num[Array, " P 2"],
            ]:
                return lax.cond(
                    image_data.shape[0] * image_data.shape[1] == pp,
                    lambda: (image_data, positions),
                    lambda: lax.stop_gradient(
                        lax.cond(
                            False,
                            lambda: (image_data, positions),
                            lambda: (image_data, positions),
                        )
                    ),
                )

            return lax.cond(
                image_data.ndim == expected_image_dim,
                check_3d_consistency,
                check_4d_consistency,
            )

        check_image_dimensions()
        check_image_finite()
        check_image_nonnegative()
        check_positions_shape()
        check_positions_finite()
        check_wavelength()
        check_dx()
        check_consistency()

        return MicroscopeData(
            image_data=image_data,
            positions=positions,
            wavelength=wavelength,
            dx=dx,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_diffractogram(
    image: Float[Array, "H W"],
    wavelength: scalar_float,
    dx: scalar_float,
) -> Diffractogram:
    """JAX-safe factory function for Diffractogram with data validation.

    Parameters
    ----------
    image : Float[Array, "H W"]
        Image data
    wavelength : scalar_float
        Wavelength of the optical wavefront in meters
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters

    Returns
    -------
    Diffractogram
        Validated diffractogram instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate image array:
        - Check it's 2D
        - Ensure all values are finite and non-negative
    - Validate parameters:
        - Check wavelength is positive
        - Check dx is positive
    - Create and return Diffractogram instance
    """
    image: Float[Array, " H W"] = jnp.asarray(image, dtype=jnp.float64)
    wavelength: Float[Array, " "] = jnp.asarray(wavelength, dtype=jnp.float64)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    expected_sample_dim: int = 2

    def validate_and_create() -> Diffractogram:
        def check_2d_image() -> Float[Array, " H W"]:
            return lax.cond(
                image.ndim == expected_sample_dim,
                lambda: image,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image, lambda: image)
                ),
            )

        def check_image_finite() -> Float[Array, " H W"]:
            return lax.cond(
                jnp.all(jnp.isfinite(image)),
                lambda: image,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image, lambda: image)
                ),
            )

        def check_image_nonnegative() -> Float[Array, " H W"]:
            return lax.cond(
                jnp.all(image >= 0),
                lambda: image,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: image, lambda: image)
                ),
            )

        def check_wavelength() -> Float[Array, " "]:
            return lax.cond(
                wavelength > 0,
                lambda: wavelength,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: wavelength, lambda: wavelength)
                ),
            )

        def check_dx() -> Float[Array, " "]:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(lax.cond(False, lambda: dx, lambda: dx)),
            )

        check_2d_image()
        check_image_finite()
        check_image_nonnegative()
        check_wavelength()
        check_dx()

        return Diffractogram(
            image=image,
            wavelength=wavelength,
            dx=dx,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_sample_function(
    sample: Complex[Array, "H W"],
    dx: scalar_float,
) -> SampleFunction:
    """JAX-safe factory function for SampleFunction with data validation.

    Parameters
    ----------
    sample : Complex[Array, "H W"]
        The sample function
    dx : scalar_float
        Spatial sampling interval (grid spacing) in meters

    Returns
    -------
    SampleFunction
        Validated sample function instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Algorithm:

    - Convert inputs to JAX arrays
    - Validate sample array:
        - Check it's 2D
        - Ensure all values are finite
    - Validate parameters:
        - Check dx is positive
    - Create and return SampleFunction instance
    """
    sample: Complex[Array, "H W"] = jnp.asarray(sample, dtype=jnp.complex128)
    dx: Float[Array, " "] = jnp.asarray(dx, dtype=jnp.float64)
    expected_sample_dim: int = 2

    def validate_and_create() -> SampleFunction:
        def check_2d_sample() -> Complex[Array, "H W"]:
            return lax.cond(
                sample.ndim == expected_sample_dim,
                lambda: sample,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: sample, lambda: sample)
                ),
            )

        def check_sample_finite() -> Complex[Array, "H W"]:
            return lax.cond(
                jnp.all(jnp.isfinite(sample)),
                lambda: sample,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: sample, lambda: sample)
                ),
            )

        def check_dx() -> scalar_float:
            return lax.cond(
                dx > 0,
                lambda: dx,
                lambda: lax.stop_gradient(lax.cond(False, lambda: dx, lambda: dx)),
            )

        check_2d_sample()
        check_sample_finite()
        check_dx()

        return SampleFunction(
            sample=sample,
            dx=dx,
        )

    return validate_and_create()
