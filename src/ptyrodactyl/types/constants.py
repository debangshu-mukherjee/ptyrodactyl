"""Physical constants for electron microscopy.

Extended Summary
----------------
This module defines the shared physical constants used by
ptyrodactyl's electron microscopy calculations. Constants are
stored as module-level Python floats so computational modules can
cast them to JAX arrays only at the point of numerical use.

Routine Listings
----------------
:obj:`HBAR`
    Reduced Planck constant in J·s.
:obj:`H_PLANCK`
    Planck constant in J·s.
:obj:`M_E`
    Electron rest mass in kg.
:obj:`E_CHARGE`
    Elementary charge in C.
:obj:`C_LIGHT`
    Speed of light in m/s.
:obj:`A_BOHR`
    Bohr radius in Angstroms.
:obj:`M0C2_EV`
    Electron rest energy in eV.
"""

from beartype.typing import Final

HBAR: Final[float] = 1.054571817e-34
"""Reduced Planck constant in J·s."""

H_PLANCK: Final[float] = 6.62607015e-34
"""Planck constant in J·s."""

M_E: Final[float] = 9.1093837015e-31
"""Electron rest mass in kg."""

E_CHARGE: Final[float] = 1.602176634e-19
"""Elementary charge in C."""

C_LIGHT: Final[float] = 2.99792458e8
"""Speed of light in m/s."""

A_BOHR: Final[float] = 0.529177210903
"""Bohr radius in Angstroms."""

M0C2_EV: Final[float] = 510998.95
"""Electron rest energy in eV."""

__all__: list[str] = [
    "A_BOHR",
    "C_LIGHT",
    "E_CHARGE",
    "H_PLANCK",
    "HBAR",
    "M0C2_EV",
    "M_E",
]
