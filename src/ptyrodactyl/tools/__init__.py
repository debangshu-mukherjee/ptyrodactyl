"""
Module: ptyrodactyl.tools
-------------------------
Utility tools for JAX ptychography - supporting light, electrons and X-rays.

This package contains essential utilities for complex-valued optimization,
loss functions, and parallel processing in ptychography applications.
All functions are JAX-compatible and support automatic differentiation.
This includes an implementation of the Wirtinger derivatives, which
are used for creating complex valued Adam, Adagrad and RMSprop optimizers.

Submodules
----------
- `loss_functions`:
    Loss function implementations for ptychography including MAE, MSE, and RMSE
    with support for complex-valued data and custom loss function creation
- `optimizers`:
    Complex-valued optimizers with Wirtinger derivatives including Adam,
    Adagrad, and RMSprop, plus learning rate schedulers for training
- `parallel`:
    Parallel processing utilities for sharding arrays across multiple devices
    and distributed computing in ptychography workflows
"""

from .loss_functions import create_loss_function
from .optimizers import (LRSchedulerState, Optimizer, OptimizerState,
                         adagrad_update, adam_update, complex_adagrad,
                         complex_adam, complex_rmsprop,
                         create_cosine_scheduler, create_step_scheduler,
                         create_warmup_cosine_scheduler, init_adagrad,
                         init_adam, init_rmsprop, init_scheduler_state,
                         rmsprop_update, wirtinger_grad)
from .parallel import shard_array

__all__: list[str] = [
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
