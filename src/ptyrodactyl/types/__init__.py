"""Canonical public type and constant exports.

Extended Summary
----------------
This package is the canonical import surface for shared type
aliases and physical constants used throughout ptyrodactyl.

The submodules are organized as follows:

- :mod:`constants`
    Physical constants for electron microscopy.
- :mod:`crystal_types`
    Crystal carriers and factories.
- :mod:`custom_types`
    Custom type aliases for scalar and image data.
- :mod:`distributions`
    Probability distribution types for ptyrodactyl ensembles.
- :mod:`electron_types`
    Electron microscopy carriers and factories.
- :mod:`jacobian_types`
    Jacobian parameter and solver-state carriers.
- :mod:`recon_types`
    Reconstruction problem and result carriers.

Routine Listings
----------------
:class:`AberrationParams`
    Zernike coefficients and soft aperture cutoff.
:class:`CGState`
    State container for conjugate gradient iteration.
:class:`CalibratedArray`
    Calibrated array data with spatial calibration.
:class:`CrystalData`
    Crystal data with atomic positions, lattice vectors, and metadata.
:class:`CrystalStructure`
    Crystal structure with fractional and Cartesian coordinates.
:class:`Distribution`
    Generic weighted-sample distribution carrier.
:class:`ExitWaveParams`
    Complex exit wave array.
:class:`FisherState`
    State container for iterative Fisher computation.
:class:`GNState`
    State container for Gauss-Newton iteration.
:class:`GeometryParams`
    Rotation angle, centre offset, ellipticity.
:class:`LMState`
    State container for Levenberg-Marquardt iteration.
:class:`LanczosState`
    State container for Lanczos tridiagonalisation.
:class:`LaplaceUncertainty`
    Laplace-approximation uncertainty carrier.
:class:`PositionParams`
    Per-scan-point position corrections.
:class:`PosteriorSamples`
    Posterior sample diagnostics carrier.
:class:`PotentialSlices`
    Potential slices for multi-slice simulations.
:class:`ProbeModeParams`
    Probe mode weights and shapes.
:class:`ProbeModes`
    Multimodal electron probe state.
:class:`PtychoParams`
    Combined parameter container for all blocks.
:class:`ReconProblem`
    Reconstruction inverse-problem carrier.
:class:`ReconResult`
    Reconstruction solver-result carrier.
:class:`ReductionMode`
    Reduction-mode enum for distribution collapse.
:class:`STEM4D`
    4D-STEM data with diffraction patterns, calibrations, and parameters.
:func:`create_calibrated_array`
    Create a CalibratedArray with runtime validation.
:func:`create_crystal_data`
    Create a CrystalData with runtime validation.
:func:`create_crystal_structure`
    Create a CrystalStructure with runtime validation.
:func:`create_distribution`
    Create a validated Distribution.
:func:`create_laplace_uncertainty`
    Create a LaplaceUncertainty with runtime validation.
:func:`create_posterior_samples`
    Create PosteriorSamples with runtime validation.
:func:`create_potential_slices`
    Create a PotentialSlices with runtime validation.
:func:`create_probe_modes`
    Create a ProbeModes with runtime validation.
:func:`create_ptycho_params`
    Construct combined PtychoParams from components.
:func:`create_recon_problem`
    Create a ReconProblem with runtime validation.
:func:`create_recon_result`
    Create a ReconResult with runtime validation.
:func:`create_stem4d`
    Create a STEM4D with runtime validation.
:func:`create_trivial_distribution`
    Create the trivial single-sample distribution.
:obj:`A_BOHR`
    Bohr radius in Angstroms.
:obj:`C_LIGHT`
    Speed of light in m/s.
:obj:`E_CHARGE`
    Elementary charge in C.
:obj:`HBAR`
    Reduced Planck constant in J·s.
:obj:`H_PLANCK`
    Planck constant in J·s.
:obj:`M0C2_EV`
    Electron rest energy in eV.
:obj:`MOTT_BETHE_VOLT_ANGSTROM_SQ`
    Mott-Bethe constant h²/(2π m₀ e) in V·Å².
:obj:`M_E`
    Electron rest mass in kg.
:obj:`TRIVIAL`
    Trivial (single-sample, unit-weight) distribution constant.
:obj:`TRIVIAL_DISTRIBUTION`
    Alias of the trivial distribution constant.
:obj:`float_jax_image`
    Type alias for 2D JAX float array (H, W).
:obj:`float_np_image`
    Type alias for 2D numpy float array (H, W).
:obj:`int_jax_image`
    Type alias for 2D JAX integer array (H, W).
:obj:`int_np_image`
    Type alias for 2D numpy integer array (H, W).
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar
    array).
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar
    array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar
    array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX
    scalar array).
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
from .distributions import (
    TRIVIAL,
    TRIVIAL_DISTRIBUTION,
    Distribution,
    ReductionMode,
    create_distribution,
    create_trivial_distribution,
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
from .recon_types import (
    LaplaceUncertainty,
    PosteriorSamples,
    ReconProblem,
    ReconResult,
    create_laplace_uncertainty,
    create_posterior_samples,
    create_recon_problem,
    create_recon_result,
)

__all__: list[str] = [
    "A_BOHR",
    "AberrationParams",
    "CGState",
    "C_LIGHT",
    "CalibratedArray",
    "CrystalData",
    "CrystalStructure",
    "Distribution",
    "E_CHARGE",
    "ExitWaveParams",
    "FisherState",
    "GNState",
    "GeometryParams",
    "HBAR",
    "H_PLANCK",
    "LMState",
    "LanczosState",
    "LaplaceUncertainty",
    "M0C2_EV",
    "MOTT_BETHE_VOLT_ANGSTROM_SQ",
    "M_E",
    "PositionParams",
    "PosteriorSamples",
    "PotentialSlices",
    "ProbeModeParams",
    "ProbeModes",
    "PtychoParams",
    "ReconProblem",
    "ReconResult",
    "ReductionMode",
    "STEM4D",
    "TRIVIAL",
    "TRIVIAL_DISTRIBUTION",
    "create_calibrated_array",
    "create_crystal_data",
    "create_crystal_structure",
    "create_distribution",
    "create_laplace_uncertainty",
    "create_posterior_samples",
    "create_potential_slices",
    "create_probe_modes",
    "create_ptycho_params",
    "create_recon_problem",
    "create_recon_result",
    "create_stem4d",
    "create_trivial_distribution",
    "distributions",
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
