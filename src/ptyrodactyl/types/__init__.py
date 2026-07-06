"""Canonical public type and constant exports.

Extended Summary
----------------
This package is the canonical import surface for shared type
aliases and physical constants used throughout ptyrodactyl.

The submodules are organized as follows:

- :mod:`custom_types`
    Custom type aliases for scalar and image data.
- :mod:`constants`
    Physical constants for electron microscopy.
- :mod:`electron_types`
    Electron microscopy carriers and factories.
- :mod:`crystal_types`
    Crystal carriers and factories.
- :mod:`jacobian_types`
    Jacobian parameter and solver-state carriers.

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
:obj:`MOTT_BETHE_VOLT_ANGSTROM_SQ`
    Mott-Bethe constant h²/(2π m₀ e) in V·Å².
:class:`CalibratedArray`
    Calibrated array data with spatial calibration.
:class:`ProbeModes`
    Multimodal electron probe state.
:class:`PotentialSlices`
    Potential slices for multi-slice simulations.
:class:`STEM4D`
    4D-STEM data with diffraction patterns, calibrations, and parameters.
:class:`CrystalStructure`
    Crystal structure with fractional and Cartesian coordinates.
:class:`CrystalData`
    Crystal data with atomic positions, lattice vectors, and metadata.
:class:`ExitWaveParams`
    Complex exit wave array.
:class:`AberrationParams`
    Zernike coefficients and soft aperture cutoff.
:class:`GeometryParams`
    Rotation angle, centre offset, ellipticity.
:class:`PositionParams`
    Per-scan-point position corrections.
:class:`ProbeModeParams`
    Probe mode weights and shapes.
:class:`PtychoParams`
    Combined parameter container for all blocks.
:class:`FisherState`
    State container for iterative Fisher computation.
:class:`CGState`
    State container for conjugate gradient iteration.
:class:`GNState`
    State container for Gauss-Newton iteration.
:class:`LMState`
    State container for Levenberg-Marquardt iteration.
:class:`LanczosState`
    State container for Lanczos tridiagonalisation.
:func:`create_calibrated_array`
    Create a CalibratedArray with runtime validation.
:func:`create_probe_modes`
    Create a ProbeModes with runtime validation.
:func:`create_potential_slices`
    Create a PotentialSlices with runtime validation.
:func:`create_stem4d`
    Create a STEM4D with runtime validation.
:func:`create_crystal_structure`
    Create a CrystalStructure with runtime validation.
:func:`create_crystal_data`
    Create a CrystalData with runtime validation.
:func:`create_ptycho_params`
    Construct combined PtychoParams from components.
"""

from .constants import (
    A_BOHR,
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M0C2_EV,
    M_E,
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
)
from .crystal_types import (
    CrystalData,
    CrystalStructure,
    create_crystal_data,
    create_crystal_structure,
)
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
from .electron_types import (
    STEM4D,
    CalibratedArray,
    PotentialSlices,
    ProbeModes,
    create_calibrated_array,
    create_potential_slices,
    create_probe_modes,
    create_stem4d,
)
from .jacobian_types import (
    AberrationParams,
    CGState,
    ExitWaveParams,
    FisherState,
    GeometryParams,
    GNState,
    LanczosState,
    LMState,
    PositionParams,
    ProbeModeParams,
    PtychoParams,
    create_ptycho_params,
)

__all__: list[str] = [
    "A_BOHR",
    "AberrationParams",
    "C_LIGHT",
    "CGState",
    "CalibratedArray",
    "CrystalData",
    "CrystalStructure",
    "E_CHARGE",
    "ExitWaveParams",
    "FisherState",
    "GNState",
    "GeometryParams",
    "H_PLANCK",
    "HBAR",
    "LMState",
    "LanczosState",
    "M0C2_EV",
    "MOTT_BETHE_VOLT_ANGSTROM_SQ",
    "M_E",
    "PositionParams",
    "PotentialSlices",
    "ProbeModeParams",
    "ProbeModes",
    "PtychoParams",
    "STEM4D",
    "create_calibrated_array",
    "create_crystal_data",
    "create_crystal_structure",
    "create_potential_slices",
    "create_ptycho_params",
    "create_probe_modes",
    "create_stem4d",
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
