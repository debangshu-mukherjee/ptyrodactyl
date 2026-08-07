"""Tests for :mod:`ptyrodactyl.jacobian.gauge`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/gauge.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.

:see: :func:`ptyrodactyl.jacobian.decompose_gauge_observable`
:see: :func:`ptyrodactyl.jacobian.effective_rank`
:see: :func:`ptyrodactyl.jacobian.gauge_invariant_norm`
:see: :func:`ptyrodactyl.jacobian.gauge_orbit_distance`
:see: :func:`ptyrodactyl.jacobian.nullspace_vectors_lanczos`
:see: :func:`ptyrodactyl.jacobian.project_to_nullspace`
:see: :func:`ptyrodactyl.jacobian.project_to_observable`
:see: :func:`ptyrodactyl.jacobian.random_gauge_direction`
"""

import ptyrodactyl.jacobian as jacobian
from ptyrodactyl.jacobian import gauge


def test_gauge_exports_resolve_through_public_package() -> None:
    """Prove each gauge helper resolves through its canonical package."""
    for symbol in gauge.__all__:
        assert getattr(jacobian, symbol) is getattr(gauge, symbol)
