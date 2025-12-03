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
    CalibratedArray, ProbeModes, PotentialSlices, and XYZData PyTrees.
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
create_loss_function : function
    Factory function to create custom loss functions.
wirtinger_grad : function
    Compute Wirtinger gradients for complex-valued optimization.
complex_adam : function
    Adam optimizer with Wirtinger derivatives for complex parameters.
complex_adagrad : function
    Adagrad optimizer with Wirtinger derivatives for complex parameters.
complex_rmsprop : function
    RMSprop optimizer with Wirtinger derivatives for complex parameters.
shard_array : function
    Shard arrays across multiple devices for parallel processing.

Notes
-----
All optimizers and loss functions support JAX transformations including
jit compilation, automatic differentiation, and vectorized mapping.
"""

from .electron_types import (
    STEM4D,
    CalibratedArray,
    CrystalStructure,
    NonJaxNumber,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarInt,
    ScalarNumeric,
    XYZData,
    make_calibrated_array,
    make_crystal_structure,
    make_potential_slices,
    make_probe_modes,
    make_stem4d,
    make_xyz_data,
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
    "STEM4D",
    "CalibratedArray",
    "CrystalStructure",
    "PotentialSlices",
    "ProbeModes",
    "XYZData",
    "make_calibrated_array",
    "make_crystal_structure",
    "make_potential_slices",
    "make_probe_modes",
    "make_stem4d",
    "make_xyz_data",
    "NonJaxNumber",
    "ScalarFloat",
    "ScalarInt",
    "ScalarNumeric",
    "create_loss_function",
    "LRSchedulerState",
    "Optimizer",
    "OptimizerState",
    "adam_update",
    "adagrad_update",
    "complex_adam",
    "complex_adagrad",
    "complex_rmsprop",
    "rmsprop_update",
    "wirtinger_grad",
    "init_adam",
    "init_adagrad",
    "init_rmsprop",
    "init_scheduler_state",
    "create_cosine_scheduler",
    "create_step_scheduler",
    "create_warmup_cosine_scheduler",
    "shard_array",
]
