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
    State maintained by learning rate schedulers.
:class:`Optimizer`
    Optimizer configuration pairing init and update callables.
:class:`OptimizerState`
    State maintained by optimizers.
:func:`adagrad_update`
    Update parameters using Adagrad with Wirtinger derivatives.
:func:`adam_update`
    Update parameters using Adam with Wirtinger derivatives.
:func:`complex_adagrad`
    Perform one step of complex-valued Adagrad.
:func:`complex_adam`
    Perform one step of complex-valued Adam.
:func:`complex_rmsprop`
    Perform one step of complex-valued RMSprop.
:func:`create_cosine_scheduler`
    Create a cosine annealing learning rate scheduler.
:func:`create_loss_function`
    Create a JIT-compiled loss function for ptychography.
:func:`create_step_scheduler`
    Create a step decay learning rate scheduler.
:func:`create_warmup_cosine_scheduler`
    Create a warmup-then-cosine-decay scheduler.
:func:`enable_compilation_cache`
    Enable JAX's persistent compilation cache.
:func:`helmholtz_coupling`
    Helmholtz potential coupling sigma_H in 1/(V·Angstrom^2).
:func:`init_adagrad`
    Initialise Adagrad optimizer state.
:func:`init_adam`
    Initialise Adam optimizer state.
:func:`init_rmsprop`
    Initialise RMSprop optimizer state.
:func:`init_scheduler_state`
    Initialise scheduler state with a given learning rate.
:func:`phase_interaction_parameter`
    Phase interaction parameter sigma in rad/(V·Angstrom).
:func:`relativistic_mass`
    Relativistic electron mass in kg.
:func:`relativistic_wavelength_ang`
    Relativistic electron wavelength in Angstroms.
:func:`rmsprop_update`
    Update parameters using RMSprop with Wirtinger derivatives.
:func:`shard_array`
    Shard an array across specified axes and devices.
:func:`wirtinger_grad`
    Compute the Wirtinger gradient of a real-valued function.

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
