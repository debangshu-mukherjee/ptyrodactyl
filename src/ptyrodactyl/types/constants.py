"""Physical constants for electron microscopy.

Extended Summary
----------------
This module defines the shared physical constants used by
ptyrodactyl's electron microscopy calculations. Constants are
materialized as **0-dimensional, weak-typed JAX arrays** at import
time: they are created from Python float literals (never with an
explicit ``dtype=``), so JAX preserves weak typing and the constants
promote exactly like Python scalars (``HBAR * x`` leaves a
``float32``/``complex64`` array's dtype untouched). The package
``__init__`` enables ``jax_enable_x64`` before any submodule import,
so these constants always materialize as ``float64``; the import-time
guard below turns any future ordering regression into a loud
``ImportError`` instead of silently truncated ``float32`` physics.

Because they are arrays, these constants are **not hashable** and must
never be used as JIT static arguments.

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
:obj:`MOTT_BETHE_VOLT_ANGSTROM_SQ`
    Mott-Bethe constant h²/(2π m₀ e) in V·Å².
"""

import jax.numpy as jnp
from beartype.typing import Final

from .custom_types import scalar_float

HBAR: Final[scalar_float] = jnp.asarray(1.054571817e-34)
"""Reduced Planck constant in J·s."""

H_PLANCK: Final[scalar_float] = jnp.asarray(6.62607015e-34)
"""Planck constant in J·s."""

M_E: Final[scalar_float] = jnp.asarray(9.1093837015e-31)
"""Electron rest mass in kg."""

E_CHARGE: Final[scalar_float] = jnp.asarray(1.602176634e-19)
"""Elementary charge in C."""

C_LIGHT: Final[scalar_float] = jnp.asarray(2.99792458e8)
"""Speed of light in m/s."""

A_BOHR: Final[scalar_float] = jnp.asarray(0.529177210903)
"""Bohr radius in Angstroms."""

M0C2_EV: Final[scalar_float] = jnp.asarray(510998.95)
"""Electron rest energy in eV."""

MOTT_BETHE_VOLT_ANGSTROM_SQ: Final[scalar_float] = jnp.asarray(47.87801)
"""Mott-Bethe constant h²/(2π m₀ e) in V·Å² (potential Fourier convention)."""

if HBAR.dtype != jnp.float64:
    raise ImportError(
        "ptyrodactyl.types.constants materialized under float32: "
        "jax_enable_x64 must be set before this module is imported "
        "(ptyrodactyl/__init__.py owns that ordering — see CONTRIBUTING)."
    )

__all__: list[str] = [
    "A_BOHR",
    "C_LIGHT",
    "E_CHARGE",
    "H_PLANCK",
    "HBAR",
    "M0C2_EV",
    "MOTT_BETHE_VOLT_ANGSTROM_SQ",
    "M_E",
]
