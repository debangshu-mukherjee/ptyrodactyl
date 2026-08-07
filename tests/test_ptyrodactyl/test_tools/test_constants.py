"""Tests for :mod:`ptyrodactyl.tools.constants`.

Extended Summary
----------------
Regression wall for the derived relativistic-optics functions. The
phase interaction parameter and the Helmholtz coupling are pinned
against published reference values (Kirkland, Advanced Computing in
Electron Microscopy, Table 2.1) and against each other through the
exact identity :math:`\\sigma_H = 2 k_0 \\sigma`. These pins exist
because a previous implementation used :math:`\\hbar^2` where
:math:`h^2` is required, inflating sigma by :math:`(2\\pi)^2`; any
reappearance of that class of bug must fail loudly here.

:see: :func:`ptyrodactyl.tools.relativistic_mass`
:see: :func:`ptyrodactyl.tools.relativistic_wavelength_ang`
"""

import jax.numpy as jnp
import pytest

import ptyrodactyl.tools
from ptyrodactyl.tools import (
    helmholtz_coupling,
    phase_interaction_parameter,
    relativistic_wavelength_ang,
)
from ptyrodactyl.types import C_LIGHT, E_CHARGE, M_E


class TestPhaseInteractionParameter:
    """Pins sigma = 2*pi*m*e*lambda/h**2 in rad/(V·Angstrom).

    :see: :func:`ptyrodactyl.tools.phase_interaction_parameter`
    """

    @pytest.mark.parametrize(
        ("voltage_kv", "expected"),
        [
            (100.0, 0.92440e-3),
            (200.0, 0.72884e-3),
            (300.0, 0.65262e-3),
        ],
    )
    def test_reference_values(
        self, voltage_kv: float, expected: float
    ) -> None:
        """Match published sigma in rad/(V Angstrom) within 1e-4 relative.

        Compare each parameterized voltage with its tabulated value.
        """
        sigma = phase_interaction_parameter(voltage_kv)
        assert sigma == pytest.approx(expected, rel=1e-4)

    def test_matches_closed_form(self) -> None:
        """Cross-check against the independent closed form.

        sigma = (2*pi / (lambda * V)) * (m0c^2 + eV) / (2*m0c^2 + eV),
        the algebraically equivalent expression used by the multislice
        transmission function.
        """
        voltage_kv = 100.0
        lam = relativistic_wavelength_ang(voltage_kv)
        ev = jnp.float64(E_CHARGE) * voltage_kv * 1e3
        m0c2 = jnp.float64(M_E) * jnp.square(jnp.float64(C_LIGHT))
        closed_form = (2.0 * jnp.pi / (lam * voltage_kv * 1e3)) * (
            (m0c2 + ev) / (2.0 * m0c2 + ev)
        )
        sigma = phase_interaction_parameter(voltage_kv)
        assert sigma == pytest.approx(float(closed_form), rel=1e-12)

    def test_hbar_bug_regression(self) -> None:
        """The hbar**2 bug inflated sigma by (2*pi)**2 ~ 39.48."""
        sigma = float(phase_interaction_parameter(100.0))
        assert sigma < 2e-3, (
            "sigma is orders of magnitude too large: the hbar^2 vs h^2 "
            "bug has reappeared"
        )

    def test_zero_dim_float64(self) -> None:
        """Prove sigma at 100 kV is rank-zero float64 by shape and dtype."""
        sigma = phase_interaction_parameter(100.0)
        assert sigma.shape == ()
        assert sigma.dtype == jnp.float64

    def test_monotonically_decreasing_in_voltage(self) -> None:
        """Prove sigma decreases from 60 to 300 kV.

        Require every adjacent sampled-voltage difference to be negative.
        """
        voltages = jnp.asarray([60.0, 100.0, 200.0, 300.0])
        sigmas = jnp.stack([phase_interaction_parameter(v) for v in voltages])
        assert bool(jnp.all(jnp.diff(sigmas) < 0.0))


class TestHelmholtzCoupling:
    """Pins sigma_H = 2*m*e/hbar**2 in 1/(V·Angstrom^2).

    :see: :func:`ptyrodactyl.tools.helmholtz_coupling`
    """

    @pytest.mark.parametrize(
        ("voltage_kv", "expected"),
        [
            (100.0, 0.31383),
            (300.0, 0.41656),
        ],
    )
    def test_reference_values(
        self, voltage_kv: float, expected: float
    ) -> None:
        """Match sigma_H in 1/(V Angstrom^2) within 1e-4 relative.

        Compare each parameterized voltage with its tabulated value.
        """
        sigma_h = helmholtz_coupling(voltage_kv)
        assert sigma_h == pytest.approx(expected, rel=1e-4)

    def test_equals_twice_k0_times_sigma(self) -> None:
        """Exact identity sigma_H = 2*k0*sigma with k0 = 2*pi/lambda."""
        for voltage_kv in (60.0, 100.0, 300.0):
            lam = relativistic_wavelength_ang(voltage_kv)
            k0 = 2.0 * jnp.pi / lam
            sigma = phase_interaction_parameter(voltage_kv)
            sigma_h = helmholtz_coupling(voltage_kv)
            assert sigma_h == pytest.approx(float(2.0 * k0 * sigma), rel=1e-12)

    def test_linear_in_voltage(self) -> None:
        """sigma_H = (2*m0*e/hbar^2)*(1 + U0/511) is affine in U0."""
        s100 = helmholtz_coupling(100.0)
        s200 = helmholtz_coupling(200.0)
        s300 = helmholtz_coupling(300.0)
        assert float(s300 - s200) == pytest.approx(
            float(s200 - s100), rel=1e-12
        )

    def test_zero_dim_float64(self) -> None:
        """Prove sigma_H at 100 kV is rank-zero float64 by shape and dtype."""
        sigma_h = helmholtz_coupling(100.0)
        assert sigma_h.shape == ()
        assert sigma_h.dtype == jnp.float64


def test_buggy_name_removed() -> None:
    """The (2*pi)^2-inflated `interaction_parameter` must stay gone."""
    assert not hasattr(ptyrodactyl.tools, "interaction_parameter")
