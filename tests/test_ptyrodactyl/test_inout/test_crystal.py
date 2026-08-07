"""Test :mod:`ptyrodactyl.inout.crystal`.

:see: :func:`ptyrodactyl.inout.parse_crystal`
"""

import ptyrodactyl.inout as inout
from ptyrodactyl.inout import crystal


def test_parse_crystal_resolves_through_public_package() -> None:
    """Prove the dispatcher has one canonical public package export.

    :see: :func:`ptyrodactyl.inout.parse_crystal`
    """
    assert inout.parse_crystal is crystal.parse_crystal
