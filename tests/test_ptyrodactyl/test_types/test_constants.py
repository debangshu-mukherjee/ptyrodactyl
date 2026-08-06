"""Tests for :mod:`ptyrodactyl.types.constants`.

Extended Summary
----------------
Pins the weak-typing contract of the physical constants: each constant is a
0-dimensional ``float64`` JAX array with ``weak_type=True``, so arithmetic
with lower-precision arrays does not promote them (the constants behave like
Python scalars in dtype promotion). Also pins exact values and the
``__all__`` surface.
"""

import jax.numpy as jnp
import pytest

from ptyrodactyl.types import (
    A_BOHR,
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M0C2_EV,
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    M_E,
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

    :see: :mod:`ptyrodactyl.types.constants`
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
