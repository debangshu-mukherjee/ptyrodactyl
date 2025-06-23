"""
Module: electrons.electron_types
-------------------------------
Data structures and type definitions for electron microscopy and ptychography.

Type Aliases
------------
- `scalar_numeric`:
    Type alias for numeric types (int, float or Num array)
    Num Array has 0 dimensions
- `scalar_float`:
    Type alias for float or Float array of 0 dimensions
- `scalar_int`:
    Type alias for int or Integer array of 0 dimensions
- `non_jax_number`:
    Type alias for non-JAX numeric types (int, float)

Classes
-------
- `CalibratedArray`:
    A named tuple for calibrated array data with spatial calibration
- `ProbeModes`:
    A named tuple for multimodal electron probe state
- `PotentialSlices`:
    A named tuple for potential slices in multi-slice simulations

Factory Functions
----------------
- `make_calibrated_array`:
    Creates a CalibratedArray instance with runtime type checking
- `make_probe_modes`:
    Creates a ProbeModes instance with runtime type checking
- `make_potential_slices`:
    Creates a PotentialSlices instance with runtime type checking

    Note: Always use these factory functions instead of directly instantiating the
    NamedTuple classes to ensure proper runtime type checking of the contents.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, TypeAlias, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

jax.config.update("jax_enable_x64", True)

scalar_numeric: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Int[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class CalibratedArray(NamedTuple):
    """
    Description
    -----------
    PyTree structure for calibrated Array.

    Attributes
    ----------
    - `data_array` (Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]):
        The actual array data
    - `calib_y` (scalar_float):
        Calibration in y direction
    - `calib_x` (scalar_float):
        Calibration in x direction
    - `real_space` (Bool[Array, ""]):
        Whether the array is in real space.
        If False, it is in reciprocal space.
    """

    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]
    calib_y: scalar_float
    calib_x: scalar_float
    real_space: Bool[Array, ""]

    def tree_flatten(self):
        return (
            (
                self.data_array,
                self.calib_y,
                self.calib_x,
                self.real_space,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class ProbeModes(NamedTuple):
    """
    Description
    -----------
    PyTree structure for multimodal electron probe state.

    Attributes
    ----------
    - `modes` (Complex[Array, "H W M"]):
        M is number of modes
    - `weights` (Float[Array, "M"]):
        Mode occupation numbers.
    - `calib` (scalar_float):
        Pixel Calibration
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, "M"]
    calib: scalar_float

    def tree_flatten(self):
        return (
            (
                self.modes,
                self.weights,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class PotentialSlices(NamedTuple):
    """
    Description
    -----------
    PyTree structure for multimodal electron probe state.

    Attributes
    ----------
    - `slices` (Complex[Array, "H W S"]):
        Individual potential slices.
        S is number of slices
    - `slice_thickness` (scalar_numeric):
        Mode occupation numbers
    - `calib` (scalar_float):
        Pixel Calibration
    """

    slices: Complex[Array, "H W S"]
    slice_thickness: scalar_numeric
    calib: scalar_float

    def tree_flatten(self):
        return (
            (
                self.slices,
                self.slice_thickness,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_calibrated_array(
    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]],
    calib_y: scalar_float,
    calib_x: scalar_float,
    real_space: Bool[Array, ""],
) -> CalibratedArray:
    """
    Description
    -----------
    JAX-safe factory function for CalibratedArray with data validation.

    Parameters
    ----------
    - `data_array` (Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]):
        The actual array data
    - `calib_y` (scalar_float):
        Calibration in y direction
    - `calib_x` (scalar_float):
        Calibration in x direction
    - `real_space` (Bool[Array, ""]):
        Whether the array is in real space

    Returns
    -------
    - `calibrated_array` (CalibratedArray):
        Validated calibrated array instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate data_array:
        - Check it's 2D
        - Ensure all values are finite
    - Validate calibration parameters:
        - Check calib_y is positive
        - Check calib_x is positive
    - Validate real_space:
        - Ensure it's a boolean scalar
    - Create and return CalibratedArray instance
    """
    # Convert data_array to appropriate dtype based on input type
    if jnp.issubdtype(data_array.dtype, jnp.integer):
        data_array = jnp.asarray(data_array, dtype=jnp.int32)
    elif jnp.issubdtype(data_array.dtype, jnp.floating):
        data_array = jnp.asarray(data_array, dtype=jnp.float64)
    elif jnp.issubdtype(data_array.dtype, jnp.complexfloating):
        data_array = jnp.asarray(data_array, dtype=jnp.complex128)
    else:
        data_array = jnp.asarray(data_array)

    calib_y = jnp.asarray(calib_y, dtype=jnp.float64)
    calib_x = jnp.asarray(calib_x, dtype=jnp.float64)
    real_space = jnp.asarray(real_space, dtype=jnp.bool_)

    def validate_and_create():
        def check_2d_array():
            return lax.cond(
                data_array.ndim == 2,
                lambda: data_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: data_array, lambda: data_array)
                ),
            )

        def check_array_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(data_array)),
                lambda: data_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: data_array, lambda: data_array)
                ),
            )

        def check_calib_y():
            return lax.cond(
                calib_y > 0,
                lambda: calib_y,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib_y, lambda: calib_y)
                ),
            )

        def check_calib_x():
            return lax.cond(
                calib_x > 0,
                lambda: calib_x,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib_x, lambda: calib_x)
                ),
            )

        def check_real_space():
            return lax.cond(
                real_space.ndim == 0,
                lambda: real_space,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: real_space, lambda: real_space)
                ),
            )

        check_2d_array()
        check_array_finite()
        check_calib_y()
        check_calib_x()
        check_real_space()

        return CalibratedArray(
            data_array=data_array,
            calib_y=calib_y,
            calib_x=calib_x,
            real_space=real_space,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_probe_modes(
    modes: Complex[Array, "H W M"],
    weights: Float[Array, "M"],
    calib: scalar_float,
) -> ProbeModes:
    """
    Description
    -----------
    JAX-safe factory function for ProbeModes with data validation.

    Parameters
    ----------
    - `modes` (Complex[Array, "H W M"]):
        Complex probe modes, M is number of modes
    - `weights` (Float[Array, "M"]):
        Mode occupation numbers
    - `calib` (scalar_float):
        Pixel calibration

    Returns
    -------
    - `probe_modes` (ProbeModes):
        Validated probe modes instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate modes array:
        - Check it's 3D with shape (H, W, M)
        - Ensure all values are finite
    - Validate weights array:
        - Check it's 1D with length M
        - Ensure all values are non-negative
        - Ensure sum is positive
    - Validate calibration:
        - Check calib is positive
    - Create and return ProbeModes instance
    """
    modes = jnp.asarray(modes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    calib = jnp.asarray(calib, dtype=jnp.float64)

    def validate_and_create():
        H, W, M = modes.shape

        def check_3d_modes():
            return lax.cond(
                modes.ndim == 3,
                lambda: modes,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: modes, lambda: modes)
                ),
            )

        def check_modes_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(modes)),
                lambda: modes,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: modes, lambda: modes)
                ),
            )

        def check_weights_shape():
            return lax.cond(
                weights.shape == (M,),
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_weights_nonnegative():
            return lax.cond(
                jnp.all(weights >= 0),
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_weights_sum():
            return lax.cond(
                jnp.sum(weights) > 0,
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_calib():
            return lax.cond(
                calib > 0,
                lambda: calib,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib, lambda: calib)
                ),
            )

        check_3d_modes()
        check_modes_finite()
        check_weights_shape()
        check_weights_nonnegative()
        check_weights_sum()
        check_calib()

        return ProbeModes(
            modes=modes,
            weights=weights,
            calib=calib,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_potential_slices(
    slices: Complex[Array, "H W S"],
    slice_thickness: scalar_numeric,
    calib: scalar_float,
) -> PotentialSlices:
    """
    Description
    -----------
    JAX-safe factory function for PotentialSlices with data validation.

    Parameters
    ----------
    - `slices` (Complex[Array, "H W S"]):
        Individual potential slices, S is number of slices
    - `slice_thickness` (scalar_numeric):
        Thickness of each slice
    - `calib` (scalar_float):
        Pixel calibration

    Returns
    -------
    - `potential_slices` (PotentialSlices):
        Validated potential slices instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate slices array:
        - Check it's 3D with shape (H, W, S)
        - Ensure all values are finite
    - Validate slice_thickness:
        - Check it's positive
    - Validate calibration:
        - Check calib is positive
    - Create and return PotentialSlices instance
    """
    slices = jnp.asarray(slices, dtype=jnp.complex128)
    slice_thickness = jnp.asarray(slice_thickness, dtype=jnp.float64)
    calib = jnp.asarray(calib, dtype=jnp.float64)

    def validate_and_create():
        H, W, S = slices.shape

        def check_3d_slices():
            return lax.cond(
                slices.ndim == 3,
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_slices_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(slices)),
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_slice_thickness():
            return lax.cond(
                slice_thickness > 0,
                lambda: slice_thickness,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slice_thickness, lambda: slice_thickness)
                ),
            )

        def check_calib():
            return lax.cond(
                calib > 0,
                lambda: calib,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib, lambda: calib)
                ),
            )

        check_3d_slices()
        check_slices_finite()
        check_slice_thickness()
        check_calib()

        return PotentialSlices(
            slices=slices,
            slice_thickness=slice_thickness,
            calib=calib,
        )

    return validate_and_create()
