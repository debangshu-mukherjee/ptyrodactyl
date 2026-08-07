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
- :mod:`form_factor_types`
    Atomic form-factor coefficient carriers and factories.
- :mod:`jacobian_types`
    Jacobian parameter and solver-state carriers.
- :mod:`potential_types`
    Three-dimensional scalar-potential carrier and factory.
- :mod:`recon_types`
    Reconstruction problem and result carriers.

Routine Listings
----------------
:class:`AberrationParams`
    Store probe aberration parameters.
:class:`AtomicSliceData`
    Store on-the-fly atomic slice inputs for sharded multislice.
:class:`AxisUpdate`
    Store additive distribution-axis deltas for one kernel evaluation.
:class:`CalibratedArray`
    Store calibrated array data with spatial calibration.
:class:`CGState`
    Store conjugate gradient iteration state.
:class:`CrystalData`
    Store parsed crystal data and static metadata.
:class:`CrystalStructure`
    Store fractional and Cartesian crystal coordinates.
:class:`DetectorConfig`
    Store detector, scan, and calibration configuration.
:class:`Distribution`
    Store a generic weighted distribution over latent samples.
:class:`EnsembleAxes`
    Store optional ensemble distributions for forward simulators.
:class:`ExitWaveParams`
    Store complex exit-wave parameters.
:class:`FisherState`
    Store state for iterative Fisher computation.
:class:`GeometryParams`
    Store geometric calibration parameters.
:class:`GNState`
    Store Gauss-Newton iteration state.
:class:`KirklandParameters`
    Store Kirkland coefficients for one element.
:class:`LanczosState`
    Store Lanczos tridiagonalisation state.
:class:`LaplaceUncertainty`
    Store a Laplace-approximation uncertainty summary.
:class:`LMState`
    Store Levenberg-Marquardt iteration state.
:class:`LobatoParameters`
    Store Lobato--Van Dyck coefficients for one element.
:class:`LossType`
    Store static loss-function selection.
:class:`MicroscopeConfig`
    Store microscope voltage and probe-aberration configuration.
:class:`OptimizableBlock`
    Store static optimizable ptychography block names.
:class:`PositionParams`
    Store scan position error parameters.
:class:`PosteriorSamples`
    Store posterior samples and diagnostics.
:class:`Potential3D`
    Store a band-limited scalar electrostatic potential in volts.
:class:`PotentialSlices`
    Store potential slices for multislice simulations.
:class:`ProbeModeParams`
    Store probe mode parameters for partial coherence.
:class:`ProbeModes`
    Store multimodal electron probe data.
:class:`PtychoParams`
    Store all ptychographic parameter blocks.
:class:`ReconProblem`
    Store a reconstruction inverse-problem contract.
:class:`ReconResult`
    Store a reconstruction solver result.
:class:`ReductionMode`
    Store static ensemble reduction mode.
:class:`STEM4D`
    Store 4D-STEM diffraction data.
:func:`combine_axis_updates`
    Sum a tuple of additive axis-update carriers.
:func:`create_atomic_slice_data`
    Create AtomicSliceData with structural and runtime validation.
:func:`create_axis_update`
    Create an AxisUpdate with structural and runtime validation.
:func:`create_calibrated_array`
    Create a CalibratedArray with runtime validation.
:func:`create_cg_state`
    Create a validated conjugate gradient iteration state.
:func:`create_crystal_data`
    Create a CrystalData with runtime validation.
:func:`create_crystal_structure`
    Create a CrystalStructure with runtime validation.
:func:`create_detector_config`
    Create a DetectorConfig with structural and runtime validation.
:func:`create_distribution`
    Create a Distribution with validated probability weights.
:func:`create_ensemble_axes`
    Create EnsembleAxes with structural validation.
:func:`create_fisher_state`
    Create a validated Fisher computation state.
:func:`create_gn_state`
    Create a validated Gauss-Newton iteration state.
:func:`create_kirkland_parameters`
    Create validated Kirkland coefficients for one element.
:func:`create_lanczos_state`
    Create a validated Lanczos tridiagonalisation state.
:func:`create_laplace_uncertainty`
    Create a LaplaceUncertainty with runtime validation.
:func:`create_lm_state`
    Create a validated Levenberg-Marquardt iteration state.
:func:`create_lobato_parameters`
    Create validated Lobato--Van Dyck coefficients for one element.
:func:`create_microscope_config`
    Create a MicroscopeConfig with structural and runtime validation.
:func:`create_posterior_samples`
    Create PosteriorSamples with runtime validation.
:func:`create_potential_3d`
    Create a validated three-dimensional electrostatic potential.
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
    Create the one-sample identity distribution.
:obj:`A_BOHR`
    Bohr radius in Angstroms.
:obj:`C_LIGHT`
    Speed of light in m/s.
:obj:`E_CHARGE`
    Elementary charge in C.
:obj:`float_jax_image`
    Type alias for 2D JAX float array (H, W).
:obj:`float_np_image`
    Type alias for 2D numpy float array (H, W).
:obj:`H_PLANCK`
    Planck constant in J·s.
:obj:`HBAR`
    Reduced Planck constant in J·s.
:obj:`int_jax_image`
    Type alias for 2D JAX integer array (H, W).
:obj:`int_np_image`
    Type alias for 2D numpy integer array (H, W).
:obj:`M0C2_EV`
    Electron rest energy in eV.
:obj:`M_E`
    Electron rest mass in kg.
:obj:`MOTT_BETHE_VOLT_ANGSTROM_SQ`
    Mott-Bethe constant h²/(2π m₀ e) in V·Å².
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar array).
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX scalar array).
:obj:`TRIVIAL`
    Short alias for ``TRIVIAL_DISTRIBUTION``.
:obj:`TRIVIAL_DISTRIBUTION`
    Identity one-sample distribution.

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
    LossType,
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
    AtomicSliceData,
    AxisUpdate,
    CalibratedArray,
    DetectorConfig,
    EnsembleAxes,
    MicroscopeConfig,
    PotentialSlices,
    ProbeModes,
    combine_axis_updates,
    create_atomic_slice_data,
    create_axis_update,
    create_calibrated_array,
    create_detector_config,
    create_ensemble_axes,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
    create_stem4d,
)
from .form_factor_types import (
    KirklandParameters,
    LobatoParameters,
    create_kirkland_parameters,
    create_lobato_parameters,
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
    OptimizableBlock,
    PositionParams,
    ProbeModeParams,
    PtychoParams,
    create_cg_state,
    create_fisher_state,
    create_gn_state,
    create_lanczos_state,
    create_lm_state,
    create_ptycho_params,
)
from .potential_types import Potential3D, create_potential_3d
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
    "AxisUpdate",
    "AtomicSliceData",
    "CGState",
    "C_LIGHT",
    "CalibratedArray",
    "CrystalData",
    "CrystalStructure",
    "Distribution",
    "DetectorConfig",
    "E_CHARGE",
    "EnsembleAxes",
    "ExitWaveParams",
    "FisherState",
    "GNState",
    "GeometryParams",
    "HBAR",
    "H_PLANCK",
    "KirklandParameters",
    "LMState",
    "LanczosState",
    "LaplaceUncertainty",
    "LobatoParameters",
    "LossType",
    "M0C2_EV",
    "MOTT_BETHE_VOLT_ANGSTROM_SQ",
    "M_E",
    "MicroscopeConfig",
    "OptimizableBlock",
    "PositionParams",
    "PosteriorSamples",
    "Potential3D",
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
    "combine_axis_updates",
    "create_axis_update",
    "create_atomic_slice_data",
    "create_calibrated_array",
    "create_cg_state",
    "create_crystal_data",
    "create_crystal_structure",
    "create_distribution",
    "create_detector_config",
    "create_ensemble_axes",
    "create_fisher_state",
    "create_gn_state",
    "create_kirkland_parameters",
    "create_lanczos_state",
    "create_laplace_uncertainty",
    "create_lm_state",
    "create_lobato_parameters",
    "create_microscope_config",
    "create_posterior_samples",
    "create_potential_3d",
    "create_potential_slices",
    "create_probe_modes",
    "create_ptycho_params",
    "create_recon_problem",
    "create_recon_result",
    "create_stem4d",
    "create_trivial_distribution",
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
