import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, TypeAlias, Union
from jax import lax
from jaxtyping import (Array, Bool, Complex, Complex128, Float, Int, Num,
                       PRNGKeyArray, jaxtyped)

import ptyrodactyl.electrons as pte

jax.config.update("jax_enable_x64", True)


scalar_numeric: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Int[Array, ""]]
