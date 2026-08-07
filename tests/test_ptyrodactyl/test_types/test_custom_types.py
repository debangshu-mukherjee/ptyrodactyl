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

import ptyrodactyl.types as types
from ptyrodactyl.types import custom_types


def test_custom_types_resolve_through_public_package() -> None:
    """Prove each shared vocabulary object has one canonical export."""
    for symbol in custom_types.__all__:
        assert getattr(types, symbol) is getattr(custom_types, symbol)
