"""Tests for :mod:`ptyrodactyl.types.custom_types`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/types/custom_types.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.

:see: :class:`ptyrodactyl.types.LossType`
:see: :obj:`ptyrodactyl.types.float_jax_image`
:see: :obj:`ptyrodactyl.types.float_np_image`
:see: :obj:`ptyrodactyl.types.int_jax_image`
:see: :obj:`ptyrodactyl.types.int_np_image`
:see: :obj:`ptyrodactyl.types.non_jax_number`
:see: :obj:`ptyrodactyl.types.scalar_bool`
:see: :obj:`ptyrodactyl.types.scalar_float`
:see: :obj:`ptyrodactyl.types.scalar_int`
:see: :obj:`ptyrodactyl.types.scalar_num`
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped

import ptyrodactyl.types as types
from ptyrodactyl.types import custom_types


def test_custom_types_resolve_through_public_package() -> None:
    """Prove each shared vocabulary object has one canonical export."""
    for symbol in custom_types.__all__:
        assert getattr(types, symbol) is getattr(custom_types, symbol)


def test_shared_input_aliases_remain_width_polymorphic() -> None:
    """Keep broad scalar and image aliases valid for float32 inputs.

    The exact-width carrier policy starts after explicit conversion. This
    check prevents the shared input vocabulary from becoming a hidden cast or
    a double-precision-only API.

    :see: :obj:`ptyrodactyl.types.scalar_float`
    """

    @jaxtyped(typechecker=beartype)
    def scalar_identity(value: types.scalar_float) -> types.scalar_float:
        result: types.scalar_float = value
        return result

    @jaxtyped(typechecker=beartype)
    def image_identity(
        values: types.float_jax_image,
    ) -> types.float_jax_image:
        result: types.float_jax_image = values
        return result

    scalar = jnp.asarray(1.0, dtype=jnp.float32)
    image = jnp.ones((2, 3), dtype=jnp.float32)
    scalar_result = scalar_identity(scalar)
    image_result = image_identity(image)

    assert scalar_result.dtype == jnp.float32
    assert image_result.dtype == jnp.float32
