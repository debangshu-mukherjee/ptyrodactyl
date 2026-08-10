"""Tests for :mod:`ptyrodactyl.types.constants`.

Extended Summary
----------------
Pins the weak-typing contract of the physical constants: each constant is a
0-dimensional ``float64`` JAX array with ``weak_type=True``, so arithmetic
with lower-precision arrays does not promote them (the constants behave like
Python scalars in dtype promotion). Also pins exact values and the
``__all__`` surface.

:see: :obj:`ptyrodactyl.types.C_LIGHT`
:see: :obj:`ptyrodactyl.types.E_CHARGE`
:see: :obj:`ptyrodactyl.types.H_PLANCK`
:see: :obj:`ptyrodactyl.types.M0C2_EV`
:see: :obj:`ptyrodactyl.types.M_E`
:see: :obj:`ptyrodactyl.types.MOTT_BETHE_VOLT_ANGSTROM_SQ`
"""

import jax
import jax.numpy as jnp
import pytest

import ptyrodactyl.types
from ptyrodactyl.types import (
    A_BOHR,
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M0C2_EV,
    M_E,
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    helmholtz_coupling,
    phase_interaction_parameter,
    relativistic_mass,
    relativistic_wavelength_ang,
)

_ALL_CONSTANTS = {
    "HBAR": (HBAR, 1.054571817e-34),
    "H_PLANCK": (H_PLANCK, 6.62607015e-34),
    "M_E": (M_E, 9.1093837015e-31),
    "E_CHARGE": (E_CHARGE, 1.602176634e-19),
    "C_LIGHT": (C_LIGHT, 2.99792458e8),
    "A_BOHR": (A_BOHR, 0.529177210903),
    "M0C2_EV": (M0C2_EV, 510998.95),
    "MOTT_BETHE_VOLT_ANGSTROM_SQ": (MOTT_BETHE_VOLT_ANGSTROM_SQ, 47.87801),
}


class TestConstantsContract:
    """Validate dtype, weak typing, and values of every physical constant.

    :see: :obj:`ptyrodactyl.types.A_BOHR`
    :see: :obj:`ptyrodactyl.types.HBAR`
    """

    @pytest.mark.parametrize("name", sorted(_ALL_CONSTANTS))
    def test_float64_zero_dim_weak(self, name: str) -> None:
        """Each constant is a weak-typed 0-d float64 JAX array.

        Notes
        -----
        Asserts dtype ``float64`` (the import-time guard's contract), zero
        rank, and ``weak_type=True`` (the promotion-safety contract).
        """
        const, _ = _ALL_CONSTANTS[name]
        const_arr = jnp.asarray(const)
        assert const_arr.dtype == jnp.float64
        assert const_arr.ndim == 0
        assert const_arr.weak_type

    @pytest.mark.parametrize("name", sorted(_ALL_CONSTANTS))
    def test_exact_value(self, name: str) -> None:
        """Each constant carries its literature value exactly.

        Notes
        -----
        Bit-exact comparison against the defining Python float literal.
        """
        const, value = _ALL_CONSTANTS[name]
        assert float(const) == value

    def test_no_promotion_of_float32(self) -> None:
        """Weak typing keeps float32 arrays float32 under arithmetic.

        Notes
        -----
        The decisive promotion check: a strongly-typed float64 constant
        would silently promote float32 operands; weak typing must not.
        """
        x32 = jnp.ones(3, dtype=jnp.float32)
        assert (A_BOHR * x32).dtype == jnp.float32

    def test_no_promotion_of_complex64(self) -> None:
        """Weak typing keeps complex64 arrays complex64 under arithmetic.

        Notes
        -----
        Same contract as the float32 check, for the complex pair.
        """
        c64 = jnp.ones(3, dtype=jnp.complex64)
        assert (HBAR * c64).dtype == jnp.complex64


class TestHelmholtzCoupling:
    """Pin the volumetric coupling and its scalar-output contract.

    :see: :func:`ptyrodactyl.types.helmholtz_coupling`
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
        """Match published coupling values within 1e-4 relative."""
        sigma_h = helmholtz_coupling(voltage_kv)
        assert sigma_h == pytest.approx(expected, rel=1e-4)

    def test_equals_twice_k0_times_sigma(self) -> None:
        """Match the exact identity sigma_H = 2*k0*sigma."""
        for voltage_kv in (60.0, 100.0, 300.0):
            lam = relativistic_wavelength_ang(voltage_kv)
            k0 = 2.0 * jnp.pi / lam
            sigma = phase_interaction_parameter(voltage_kv)
            sigma_h = helmholtz_coupling(voltage_kv)
            assert sigma_h == pytest.approx(float(2.0 * k0 * sigma), rel=1e-12)

    def test_linear_in_voltage(self) -> None:
        """Require the coupling to be affine in accelerating voltage."""
        s100 = helmholtz_coupling(100.0)
        s200 = helmholtz_coupling(200.0)
        s300 = helmholtz_coupling(300.0)
        assert float(s300 - s200) == pytest.approx(
            float(s200 - s100), rel=1e-12
        )

    def test_zero_dim_float64(self) -> None:
        """Require a rank-zero float64 result at 100 kV."""
        sigma_h = helmholtz_coupling(100.0)
        assert sigma_h.shape == ()
        assert sigma_h.dtype == jnp.float64

    def test_float32_input_has_float64_eager_and_compiled_output(self) -> None:
        """Canonicalize a lower-width boundary value to float64."""
        voltage = jnp.asarray(200.0, dtype=jnp.float32)

        eager = helmholtz_coupling(voltage)
        compiled = jax.jit(helmholtz_coupling)(voltage)

        assert eager.dtype == jnp.float64
        assert compiled.dtype == jnp.float64
        assert helmholtz_coupling.__annotations__["return"].dtypes == (
            "float64",
        )


class TestPhaseInteractionParameter:
    """Pin the phase coupling and its scalar-output contract.

    :see: :func:`ptyrodactyl.types.phase_interaction_parameter`
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
        """Match published phase parameters within 1e-4 relative."""
        sigma = phase_interaction_parameter(voltage_kv)
        assert sigma == pytest.approx(expected, rel=1e-4)

    def test_matches_closed_form(self) -> None:
        """Cross-check against an algebraically independent closed form."""
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
        """Keep the former hbar-squared inflation defect absent."""
        sigma = float(phase_interaction_parameter(100.0))
        assert sigma < 2e-3

    def test_zero_dim_float64(self) -> None:
        """Require a rank-zero float64 result at 100 kV."""
        sigma = phase_interaction_parameter(100.0)
        assert sigma.shape == ()
        assert sigma.dtype == jnp.float64

    def test_monotonically_decreasing_in_voltage(self) -> None:
        """Require sigma to decrease from 60 to 300 kV."""
        voltages = jnp.asarray([60.0, 100.0, 200.0, 300.0])
        sigmas = jnp.stack([phase_interaction_parameter(v) for v in voltages])
        assert bool(jnp.all(jnp.diff(sigmas) < 0.0))


class TestRelativisticMass:
    """Pin the relativistic-mass formula and monotonic behavior.

    :see: :func:`ptyrodactyl.types.relativistic_mass`
    """

    def test_zero_voltage_equals_rest_mass(self) -> None:
        """Recover the canonical rest mass at zero voltage."""
        assert relativistic_mass(0.0) == pytest.approx(float(M_E))

    def test_increases_with_voltage(self) -> None:
        """Require relativistic mass to increase with voltage."""
        assert relativistic_mass(300.0) > relativistic_mass(100.0)


class TestRelativisticWavelength:
    """Pin the relativistic wavelength shape and voltage dependence.

    :see: :func:`ptyrodactyl.types.relativistic_wavelength_ang`
    """

    def test_reference_value_at_200_kv(self) -> None:
        """Match the standard 200 kV wavelength in Angstroms."""
        wavelength = relativistic_wavelength_ang(200.0)
        assert wavelength == pytest.approx(0.02507934, rel=1e-6)

    def test_decreases_with_voltage(self) -> None:
        """Require wavelength to decrease with accelerating voltage."""
        assert relativistic_wavelength_ang(
            300.0
        ) < relativistic_wavelength_ang(100.0)


def test_all_derived_optics_functions_canonicalize_to_float64() -> None:
    """Pin eager, compiled, and annotated binary64 result contracts."""
    voltage = jnp.asarray(200.0, dtype=jnp.float32)
    functions = (
        phase_interaction_parameter,
        relativistic_mass,
        relativistic_wavelength_ang,
    )

    for function in functions:
        assert function(voltage).dtype == jnp.float64
        assert jax.jit(function)(voltage).dtype == jnp.float64
        assert function.__annotations__["return"].dtypes == ("float64",)


def test_buggy_name_removed() -> None:
    """Keep the formerly inflated interaction_parameter name absent."""
    assert not hasattr(ptyrodactyl.types, "interaction_parameter")
