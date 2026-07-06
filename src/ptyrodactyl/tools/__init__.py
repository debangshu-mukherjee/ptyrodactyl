"""Numerical utilities for ptychography workflows.

Extended Summary
----------------
This package exposes numerical helpers for loss construction,
complex-valued optimization, derived electron-optics quantities, and
array sharding. Shared carriers, scalar aliases, constants, and
validated factories are exported only from :mod:`ptyrodactyl.types`.

The submodules are organized as follows:

- :mod:`caching`
    Persistent XLA compilation cache configuration.
- :mod:`constants`
    Derived electron-optics quantities.
- :mod:`loss_functions`
    Loss function implementations for ptychography optimization.
- :mod:`optimizers`
    Complex-valued optimizers with Wirtinger derivatives.
- :mod:`parallel`
    Parallel processing utilities for distributed ptychography.

Routine Listings
----------------
:class:`LRSchedulerState`
    Learning rate scheduler state.
:class:`Optimizer`
    Optimizer configuration.
:class:`OptimizerState`
    Optimizer state for training.
:func:`adagrad_update`
    Adagrad parameter update step.
:func:`adam_update`
    Adam parameter update step.
:func:`complex_adagrad`
    Adagrad optimizer with Wirtinger derivatives for complex parameters.
:func:`complex_adam`
    Adam optimizer with Wirtinger derivatives for complex parameters.
:func:`complex_rmsprop`
    RMSprop optimizer with Wirtinger derivatives for complex parameters.
:func:`create_cosine_scheduler`
    Create cosine annealing learning rate scheduler.
:func:`create_loss_function`
    Factory that creates a JIT-compiled loss function.
:func:`create_step_scheduler`
    Create step decay learning rate scheduler.
:func:`create_warmup_cosine_scheduler`
    Create warmup cosine annealing learning rate scheduler.
:func:`enable_compilation_cache`
    Point JAX's persistent compilation cache at a directory.
:func:`helmholtz_coupling`
    Helmholtz potential coupling sigma_H in 1/(V·Angstrom^2).
:func:`init_adagrad`
    Initialize Adagrad optimizer state.
:func:`init_adam`
    Initialize Adam optimizer state.
:func:`init_rmsprop`
    Initialize RMSprop optimizer state.
:func:`init_scheduler_state`
    Initialize learning rate scheduler state.
:func:`phase_interaction_parameter`
    Phase interaction parameter sigma in rad/(V·Angstrom).
:func:`relativistic_mass`
    Relativistic electron mass in kg.
:func:`relativistic_wavelength_ang`
    Relativistic electron wavelength in Angstroms.
:func:`rmsprop_update`
    RMSprop parameter update step.
:func:`shard_array`
    Shard arrays across multiple devices for parallel processing.
:func:`wirtinger_grad`
    Compute Wirtinger gradients for complex-valued optimization.

Notes
-----
All exported functions are JAX-compatible and designed for use with
``jit``, ``grad``, and ``vmap`` where applicable.
"""

from .caching import enable_compilation_cache
from .constants import (
    helmholtz_coupling,
    phase_interaction_parameter,
    relativistic_mass,
    relativistic_wavelength_ang,
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
    "LRSchedulerState",
    "Optimizer",
    "OptimizerState",
    "adagrad_update",
    "adam_update",
    "complex_adagrad",
    "complex_adam",
    "complex_rmsprop",
    "create_cosine_scheduler",
    "create_loss_function",
    "create_step_scheduler",
    "create_warmup_cosine_scheduler",
    "enable_compilation_cache",
    "helmholtz_coupling",
    "init_adagrad",
    "init_adam",
    "init_rmsprop",
    "init_scheduler_state",
    "phase_interaction_parameter",
    "relativistic_mass",
    "relativistic_wavelength_ang",
    "rmsprop_update",
    "shard_array",
    "wirtinger_grad",
]
