"""Utility tools for JAX ptychography.

Extended Summary
----------------
This package contains essential utilities for complex-valued optimization,
loss functions, and parallel processing in ptychography applications.
All functions are JAX-compatible and support automatic differentiation.
This includes an implementation of the Wirtinger derivatives, which
are used for creating complex valued Adam, Adagrad and RMSprop optimizers.

Submodules
----------
electron_types
    Data structures and type definitions for electron microscopy including
    CalibratedArray, ProbeModes, PotentialSlices, and CrystalData PyTrees.
factory
    Factory functions for validating data before PyTree loading.
loss_functions
    Loss function implementations for ptychography including MAE, MSE, and RMSE
    with support for complex-valued data and custom loss function creation.
optimizers
    Complex-valued optimizers with Wirtinger derivatives including Adam,
    Adagrad, and RMSprop, plus learning rate schedulers for training.
parallel
    Parallel processing utilities for sharding arrays across multiple devices
    and distributed computing in ptychography workflows.

Routine Listings
----------------
CalibratedArray : PyTree
    Calibrated array data with spatial calibration.
CrystalData : PyTree
    Crystal data with atomic positions, lattice vectors, and metadata.
CrystalStructure : PyTree
    Crystal structure with fractional and Cartesian coordinates.
LRSchedulerState : NamedTuple
    Learning rate scheduler state.
NonJaxNumber : TypeAlias
    Non-JAX numeric types (int, float).
Optimizer : NamedTuple
    Optimizer configuration.
OptimizerState : NamedTuple
    Optimizer state for training.
PotentialSlices : PyTree
    Potential slices for multi-slice simulations.
ProbeModes : PyTree
    Multimodal electron probe state.
ScalarFloat : TypeAlias
    Float or 0-dimensional Float array.
ScalarInt : TypeAlias
    Int or 0-dimensional Int array.
ScalarNumeric : TypeAlias
    Numeric types (int, float, or 0-dimensional Num array).
STEM4D : PyTree
    4D-STEM data with diffraction patterns, calibrations, and parameters.
adagrad_update : function
    Adagrad parameter update step.
adam_update : function
    Adam parameter update step.
complex_adagrad : function
    Adagrad optimizer with Wirtinger derivatives for complex parameters.
complex_adam : function
    Adam optimizer with Wirtinger derivatives for complex parameters.
complex_rmsprop : function
    RMSprop optimizer with Wirtinger derivatives for complex parameters.
create_cosine_scheduler : function
    Create cosine annealing learning rate scheduler.
create_loss_function : function
    Factory function to create custom loss functions.
create_step_scheduler : function
    Create step decay learning rate scheduler.
create_warmup_cosine_scheduler : function
    Create warmup cosine annealing learning rate scheduler.
init_adagrad : function
    Initialize Adagrad optimizer state.
init_adam : function
    Initialize Adam optimizer state.
init_rmsprop : function
    Initialize RMSprop optimizer state.
init_scheduler_state : function
    Initialize learning rate scheduler state.
make_calibrated_array : function
    Creates a CalibratedArray instance with runtime type checking.
make_crystal_data : function
    Creates a CrystalData instance with runtime type checking.
make_crystal_structure : function
    Creates a CrystalStructure instance with runtime type checking.
make_potential_slices : function
    Creates a PotentialSlices instance with runtime type checking.
make_probe_modes : function
    Creates a ProbeModes instance with runtime type checking.
make_stem4d : function
    Creates a STEM4D instance with runtime type checking.
rmsprop_update : function
    RMSprop parameter update step.
shard_array : function
    Shard arrays across multiple devices for parallel processing.
wirtinger_grad : function
    Compute Wirtinger gradients for complex-valued optimization.

Notes
-----
All optimizers and loss functions support JAX transformations including
jit compilation, automatic differentiation, and vectorized mapping.
"""

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
    "CalibratedArray",
    "CrystalData",
    "CrystalStructure",
    "LRSchedulerState",
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
    "make_calibrated_array",
    "make_crystal_data",
    "make_crystal_structure",
    "make_potential_slices",
    "make_probe_modes",
    "make_stem4d",
    "rmsprop_update",
    "shard_array",
    "wirtinger_grad",
]
