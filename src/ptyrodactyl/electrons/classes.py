import jax
from beartype.typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class CalibratedArray(NamedTuple):
    """
    Description
    -----------
    PyTree structure for calibrated Array.

    Attributes
    ----------
    - `data_array` (Num[Array, "H W"] | Num[Array, "W W"]):
        The actual array data
    - `calib_y` (Float[Array, ""]):
        Calibration in y direction
    - `calib_x` (Float[Array, ""]):
        Calibration in x direction
    - `real_space` (Bool[Array, ""]):
        Whether the array is in real space.
        If False, it is in reciprocal space.
    """

    data_array: Num[Array, "H W"] | Num[Array, "W W"]
    calib_y: Float[Array, ""]
    calib_x: Float[Array, ""]
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


@register_pytree_node_class
class ProbeState(NamedTuple):
    """ "
    Description
    -----------
    PyTree structure for electron probe state.

    Attributes
    ----------
    - `modes` (Complex[Array, "H W M"]):
        M is number of modes
    - `weights` (Float[Array, "M"]):
        Mode occupation numbers
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, "M"]

    def tree_flatten(self):
        return (
            (
                self.modes,
                self.weights,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class MixedQuantumStates(NamedTuple):
    """ "
    Description
    -----------
    PyTree structure for mixed probe quantum states.

    Attributes
    ----------
    - `states` (Complex[Array, "H W N"]):
        N different states
    - `weights` (Float[Array, "M"]):
        Occupation probabilities
    """

    states: Complex[Array, "H W N"]
    probabilities: Float[Array, "N"]

    def tree_flatten(self):
        return (
            (
                self.states,
                self.probabilities,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class MixedStateParams(NamedTuple):
    """ "
    Description
    -----------
    PyTree structure for mixed probe quantum states.

    Attributes
    ----------
    - `num_modes` (Int[Array, ""]):
        number of modes
    - `mode_weights` (Float[Array, "M"]):
        Weights for each mode
    """

    num_modes: Int[Array, ""]
    mode_weights: Float[Array, "M"]

    def tree_flatten(self):
        return (
            (
                self.num_modes,
                self.mode_weights,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
