import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import NamedTuple, Optional, TypeAlias, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

jax.config.update("jax_enable_x64", True)

scalar_numeric: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Int[Array, ""]]


@jaxtyped(typechecker=typechecker)
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


@jaxtyped(typechecker=typechecker)
@register_pytree_node_class
class ProbeModes(NamedTuple):
    """
    Description
    -----------
    PyTree structure for multimodal electron probe state.

    Attributes
    ----------
    - `modes` (Complex[Array, "H W *M"]):
        M is number of modes
    - `weights` (Optional[Float[Array, "M"]]):
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


@jaxtyped(typechecker=typechecker)
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

    slices: Complex[Array, "H W *S"]
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
