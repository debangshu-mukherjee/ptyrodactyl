"""
Module: electrons.preprocessing
------------------------------
Data preprocessing utilities for electron microscopy and ptychography.

This module contains utilities for preprocessing electron microscopy data
before analysis or reconstruction. Currently includes type definitions
for scalar numeric types used throughout the electrons module.
"""

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, TypeAlias, Union
from jax import lax
from jaxtyping import (Array, Bool, Complex, Complex128, Float, Int, Num,
                       PRNGKeyArray, jaxtyped)

jax.config.update("jax_enable_x64", True)
