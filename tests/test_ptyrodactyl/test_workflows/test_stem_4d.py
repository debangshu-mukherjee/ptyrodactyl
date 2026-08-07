"""Tests for :mod:`ptyrodactyl.workflows.stem_4d`.

Extended Summary
----------------
This module verifies high-level 4D-STEM compositions and mirrors
``src/ptyrodactyl/workflows/stem_4d.py``.

:see: :func:`ptyrodactyl.workflows.crystal2stem4d_tiled`
:see: :func:`ptyrodactyl.workflows.crystal2stem4d`
"""

import ptyrodactyl.workflows as workflows
from ptyrodactyl.workflows import stem_4d


def test_stem_4d_exports_resolve_through_public_package() -> None:
    """Prove each 4D-STEM workflow has one canonical package export."""
    for symbol in stem_4d.__all__:
        assert getattr(workflows, symbol) is getattr(stem_4d, symbol)
