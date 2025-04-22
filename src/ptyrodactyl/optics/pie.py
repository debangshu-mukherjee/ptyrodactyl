import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

import ptyrodactyl.optics as pto

jax.config.update("jax_enable_x64", True)
