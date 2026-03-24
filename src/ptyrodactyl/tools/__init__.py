"""Utility tools for JAX ptychography.

Extended Summary
----------------
This package contains essential utilities for complex-valued
optimization, loss functions, and parallel processing in
ptychography applications. All functions are JAX-compatible and
support automatic differentiation. This includes an implementation
of the Wirtinger derivatives, which are used for creating complex
valued Adam, Adagrad and RMSprop optimizers.

Routine Listings
----------------
:class:`CalibratedArray`
    Calibrated array data with spatial calibration.
:class:`CrystalData`
    Crystal data with atomic positions, lattice vectors,
    and metadata.
:class:`CrystalStructure`
    Crystal structure with fractional and Cartesian coordinates.
:class:`LRSchedulerState`
    Learning rate scheduler state.
:class:`Optimizer`
    Optimizer configuration.
:class:`OptimizerState`
    Optimizer state for training.
:class:`PotentialSlices`
    Potential slices for multi-slice simulations.
:class:`ProbeModes`
    Multimodal electron probe state.
:class:`STEM4D`
    4D-STEM data with diffraction patterns, calibrations,
    and parameters.
:data:`NonJaxNumber`
    Non-JAX numeric types (int, float).
:data:`ScalarFloat`
    Float or 0-dimensional Float array.
:data:`ScalarInt`
    Int or 0-dimensional Int array.
:data:`ScalarNumeric`
    Numeric types (int, float, or 0-dimensional Num array).
:data:`A_BOHR`
    Bohr radius in Angstroms.
:data:`C_LIGHT`
    Speed of light in m/s.
:data:`E_CHARGE`
    Elementary charge in C.
:data:`H_PLANCK`
    Planck constant in J·s.
:data:`HBAR`
    Reduced Planck constant in J·s.
:data:`M0C2_EV`
    Electron rest energy in eV.
:data:`M_E`
    Electron rest mass in kg.
:func:`adagrad_update`
    Adagrad parameter update step.
:func:`adam_update`
    Adam parameter update step.
:func:`complex_adagrad`
    Adagrad optimizer with Wirtinger derivatives for complex
    parameters.
:func:`complex_adam`
    Adam optimizer with Wirtinger derivatives for complex
    parameters.
:func:`complex_rmsprop`
    RMSprop optimizer with Wirtinger derivatives for complex
    parameters.
:func:`create_cosine_scheduler`
    Create cosine annealing learning rate scheduler.
:func:`create_loss_function`
    Factory function to create custom loss functions.
:func:`create_step_scheduler`
    Create step decay learning rate scheduler.
:func:`create_warmup_cosine_scheduler`
    Create warmup cosine annealing learning rate scheduler.
:func:`init_adagrad`
    Initialize Adagrad optimizer state.
:func:`init_adam`
    Initialize Adam optimizer state.
:func:`init_rmsprop`
    Initialize RMSprop optimizer state.
:func:`init_scheduler_state`
    Initialize learning rate scheduler state.
:func:`interaction_parameter`
    Interaction parameter sigma in 1/(V·Angstrom).
:func:`make_calibrated_array`
    Creates a CalibratedArray with runtime type checking.
:func:`make_crystal_data`
    Creates a CrystalData with runtime type checking.
:func:`make_crystal_structure`
    Creates a CrystalStructure with runtime type checking.
:func:`make_potential_slices`
    Creates a PotentialSlices with runtime type checking.
:func:`make_probe_modes`
    Creates a ProbeModes with runtime type checking.
:func:`make_stem4d`
    Creates a STEM4D with runtime type checking.
:func:`relativistic_mass`
    Relativistic electron mass in kg.
:func:`relativistic_wavelength_ang`
    Relativistic electron wavelength in Angstroms.
:func:`rmsprop_update`
    RMSprop parameter update step.
:func:`shard_array`
    Shard arrays across multiple devices for parallel
    processing.
:func:`wirtinger_grad`
    Compute Wirtinger gradients for complex-valued
    optimization.

Notes
-----
All optimizers and loss functions support JAX transformations
including jit compilation, automatic differentiation, and
vectorized mapping.
"""

from .constants import (
    A_BOHR,
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M0C2_EV,
    M_E,
    interaction_parameter,
    relativistic_mass,
    relativistic_wavelength_ang,
)
from .electron_types import (
    STEM4D,
    CalibratedArray,
    CrystalData,
    CrystalStructure,
    NonJaxNumber,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarInt,
    ScalarNumeric,
)
from .factory import (
    make_calibrated_array,
    make_crystal_data,
    make_crystal_structure,
    make_potential_slices,
    make_probe_modes,
    make_stem4d,
)
from .loss_functions import create_loss_function
from .optimizers import (
    LRSchedulerState,
    Optimizer,
    OptimizerState,
    adagrad_update,
    adam_update,
    complex_adagrad,
    complex_adam,
    complex_rmsprop,
    create_cosine_scheduler,
    create_step_scheduler,
    create_warmup_cosine_scheduler,
    init_adagrad,
    init_adam,
    init_rmsprop,
    init_scheduler_state,
    rmsprop_update,
    wirtinger_grad,
)
from .parallel import shard_array

__all__: list[str] = [
    "A_BOHR",
    "C_LIGHT",
    "CalibratedArray",
    "CrystalData",
    "CrystalStructure",
    "E_CHARGE",
    "H_PLANCK",
    "HBAR",
    "LRSchedulerState",
    "M0C2_EV",
    "M_E",
    "NonJaxNumber",
    "Optimizer",
    "OptimizerState",
    "PotentialSlices",
    "ProbeModes",
    "ScalarFloat",
    "ScalarInt",
    "ScalarNumeric",
    "STEM4D",
    "adagrad_update",
    "adam_update",
    "complex_adagrad",
    "complex_adam",
    "complex_rmsprop",
    "create_cosine_scheduler",
    "create_loss_function",
    "create_step_scheduler",
    "create_warmup_cosine_scheduler",
    "init_adagrad",
    "init_adam",
    "init_rmsprop",
    "init_scheduler_state",
    "interaction_parameter",
    "make_calibrated_array",
    "make_crystal_data",
    "make_crystal_structure",
    "make_potential_slices",
    "make_probe_modes",
    "make_stem4d",
    "relativistic_mass",
    "relativistic_wavelength_ang",
    "rmsprop_update",
    "shard_array",
    "wirtinger_grad",
]
