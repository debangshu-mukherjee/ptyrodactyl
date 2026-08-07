"""Test :mod:`ptyrodactyl.inout.poscar`.

:see: :func:`ptyrodactyl.inout.parse_poscar`
"""

import ptyrodactyl.inout as inout
from ptyrodactyl.inout import poscar


def test_parse_poscar_resolves_through_public_package() -> None:
    """Prove the POSCAR parser has one canonical public package export.

    :see: :func:`ptyrodactyl.inout.parse_poscar`
    """
    assert inout.parse_poscar is poscar.parse_poscar
