"""Tests for :mod:`ptyrodactyl.jacobian.fisher`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/fisher.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.

:see: :func:`ptyrodactyl.jacobian.a_optimality`
:see: :func:`ptyrodactyl.jacobian.condition_number`
:see: :func:`ptyrodactyl.jacobian.d_optimality`
:see: :func:`ptyrodactyl.jacobian.e_optimality`
:see: :func:`ptyrodactyl.jacobian.effective_fisher`
:see: :func:`ptyrodactyl.jacobian.fisher_diagonal`
:see: :func:`ptyrodactyl.jacobian.fisher_eigenspectrum`
:see: :func:`ptyrodactyl.jacobian.fisher_information_operator`
:see: :func:`ptyrodactyl.jacobian.fisher_information`
:see: :func:`ptyrodactyl.jacobian.information_gain`
:see: :func:`ptyrodactyl.jacobian.optimal_weights_e_criterion`
:see: :func:`ptyrodactyl.jacobian.schur_complement`
:see: :func:`ptyrodactyl.jacobian.stack_fisher`
"""

import ptyrodactyl.jacobian as jacobian
from ptyrodactyl.jacobian import fisher


def test_fisher_exports_resolve_through_public_package() -> None:
    """Prove each Fisher helper resolves through its canonical package."""
    for symbol in fisher.__all__:
        assert getattr(jacobian, symbol) is getattr(fisher, symbol)
