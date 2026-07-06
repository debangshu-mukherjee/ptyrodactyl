"""Canonical public type and constant exports.

Extended Summary
----------------
This package is the canonical import surface for shared type
aliases and physical constants used throughout ptyrodactyl.

Routine Listings
----------------
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar
    array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar
    array).
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar
    array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX
    scalar array).
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:obj:`float_jax_image`
    Type alias for 2D JAX float array (H, W).
:obj:`int_jax_image`
    Type alias for 2D JAX integer array (H, W).
:obj:`float_np_image`
    Type alias for 2D numpy float array (H, W).
:obj:`int_np_image`
    Type alias for 2D numpy integer array (H, W).
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

from .constants import A_BOHR, C_LIGHT, E_CHARGE, H_PLANCK, HBAR, M0C2_EV, M_E
from .custom_types import (
    float_jax_image,
    float_np_image,
    int_jax_image,
    int_np_image,
    non_jax_number,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_num,
)

__all__: list[str] = [
    "A_BOHR",
    "C_LIGHT",
    "E_CHARGE",
    "H_PLANCK",
    "HBAR",
    "M0C2_EV",
    "M_E",
    "float_jax_image",
    "float_np_image",
    "int_jax_image",
    "int_np_image",
    "non_jax_number",
    "scalar_bool",
    "scalar_float",
    "scalar_int",
    "scalar_num",
]
