"""
Module: photons.engine
---------------------------
Engine framework for optical simulations and ptychography.

This module is a placeholder for building a simulation engine 
that will orchestrate the various optical components.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

import ptyrodactyl.photons as pto

jax.config.update("jax_enable_x64", True)
