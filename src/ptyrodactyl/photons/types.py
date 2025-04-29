from beartype.typing import NamedTuple, Union, TypeAlias
from jaxtyping import Array, Float, Integer, Num
from jax.tree_util import register_pytree_node_class

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