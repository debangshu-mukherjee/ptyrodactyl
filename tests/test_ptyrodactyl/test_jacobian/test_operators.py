"""Tests for :mod:`ptyrodactyl.jacobian.operators`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/operators.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.

:see: :func:`ptyrodactyl.jacobian.hvp_gauss_newton`
:see: :func:`ptyrodactyl.jacobian.jtj_operator`
:see: :func:`ptyrodactyl.jacobian.jvp_operator`
:see: :func:`ptyrodactyl.jacobian.vjp_operator`
"""

import ptyrodactyl.jacobian as jacobian
from ptyrodactyl.jacobian import operators


def test_operator_exports_resolve_through_public_package() -> None:
    """Prove each matrix-free operator has one canonical package export."""
    for symbol in operators.__all__:
        assert getattr(jacobian, symbol) is getattr(operators, symbol)
