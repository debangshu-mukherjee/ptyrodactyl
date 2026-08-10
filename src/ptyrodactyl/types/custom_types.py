"""Custom type aliases and vocabularies for scalar and image data.

Extended Summary
----------------
This module defines shared scalar and image type aliases plus small
static vocabularies for JAX-compatible electron microscopy code.
Scalar aliases accept standard Python scalars and 0-dimensional JAX
arrays so public APIs can be called naturally from Python while
remaining traceable under JAX transformations.

Routine Listings
----------------
:class:`LossType`
    Store static loss-function selection.
:obj:`float_jax_image`
    Type alias for 2D JAX float array (H, W).
:obj:`float_np_image`
    Type alias for 2D numpy float array (H, W).
:obj:`int_jax_image`
    Type alias for 2D JAX integer array (H, W).
:obj:`int_np_image`
    Type alias for 2D numpy integer array (H, W).
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar array).
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX scalar array).

Notes
-----
These aliases are re-exported from :mod:`ptyrodactyl.types` as the
canonical type import path for the package. They are intentionally
width-polymorphic. Canonical carrier fields and post-conversion arrays use
width-qualified jaxtyping dtypes such as ``Float64`` or ``Int64`` instead.
An exact dtype annotation asserts a contract; it does not convert a value.
"""

from enum import Enum

from beartype.typing import TypeAlias, Union
from jaxtyping import Array, Bool, Float, Int, Num
from numpy.typing import NDArray


class LossType(str, Enum):
    """Store static loss-function selection.

    :see: :mod:`~.test_custom_types`

    Attributes
    ----------
    MAE : str
        Mean absolute error.
    MSE : str
        Mean squared error.
    RMSE : str
        Root mean squared error.
    """

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"


scalar_float: TypeAlias = Union[float, Float[Array, " "]]
"""Scalar float accepted as a Python float or 0-dimensional JAX array.

:see: :mod:`~.test_custom_types`
"""

scalar_int: TypeAlias = Union[int, Int[Array, " "]]
"""Scalar integer accepted as a Python int or 0-dimensional JAX array.

:see: :mod:`~.test_custom_types`
"""

scalar_bool: TypeAlias = Union[bool, Bool[Array, " "]]
"""Scalar boolean accepted as a Python bool or 0-dimensional JAX array.

:see: :mod:`~.test_custom_types`
"""

scalar_num: TypeAlias = Union[int, float, Num[Array, " "]]
"""Scalar numeric value accepted as Python numeric or JAX scalar array.

:see: :mod:`~.test_custom_types`
"""

non_jax_number: TypeAlias = Union[int, float]
"""Non-JAX numeric scalar accepted as a Python int or float.

:see: :mod:`~.test_custom_types`
"""

float_jax_image: TypeAlias = Float[Array, " H W"]
"""2-dimensional JAX floating-point image array.

:see: :mod:`~.test_custom_types`
"""

int_jax_image: TypeAlias = Int[Array, " H W"]
"""2-dimensional JAX integer image array.

:see: :mod:`~.test_custom_types`
"""

float_np_image: TypeAlias = Float[NDArray, " H W"]
"""2-dimensional numpy floating-point image array.

:see: :mod:`~.test_custom_types`
"""

int_np_image: TypeAlias = Int[NDArray, " H W"]
"""2-dimensional numpy integer image array.

:see: :mod:`~.test_custom_types`
"""

__all__: list[str] = [
    "LossType",
    "float_jax_image",
    "float_np_image",
    "int_jax_image",
    "int_np_image",
    "non_jax_number",
    "scalar_bool",
    "scalar_float",
    "scalar_int",
    "scalar_num",
]
