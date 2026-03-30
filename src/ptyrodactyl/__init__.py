"""Differentiable electron microscopy forward and inverse problems.

Extended Summary
----------------
A comprehensive toolkit for electron ptychography simulations
and reconstructions using JAX for automatic differentiation and
GPU acceleration. All functions are fully differentiable and
JIT-compilable, supporting ``jax.jit``, ``jax.grad``,
``jax.vmap``, and other JAX transformations. Complex-valued
optimization is handled via Wirtinger calculus, and distributed
computing is supported through device mesh parallelism. Type
safety is enforced with jaxtyping and beartype.

Routine Listings
----------------
- :mod:`born`
    Convergent Born series simulations.
- :mod:`invert`
    Electron microscopy reconstructions, ptychography and
    focal series.
- :mod:`jacobian`
    Jacobian computation submodule.
- :mod:`simul`
    Electron microscopy simulations including 4D-STEM, CBED,
    and multislice.
- :mod:`tools`
    Utility tools for optimization, loss functions, and
    parallel processing including complex-valued optimizers
    with Wirtinger derivatives.
- :mod:`workflows`
    High-level workflows combining simulation steps for common
    use cases such as simulating 4D-STEM data from XYZ
    structure files.

Notes
-----
All functions are optimized for JAX transformations and support
both CPU and GPU execution. For best performance, use JIT
compilation and consider using the provided factory functions
for input validation and float64 casting.

Multi-node distributed execution is supported via
``jax.distributed.initialize()``. To enable, set the
environment variable ``PTYRODACTYL_DISTRIBUTED=1`` before
launching with ``srun`` or equivalent. An optional
``PTYRODACTYL_COORDINATOR_ADDRESS`` environment variable
overrides automatic SLURM coordinator detection, which is
required on some ROCm clusters.
"""

import os
from importlib.metadata import version

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

import jax  # noqa: E402

if (
    os.environ.get("PTYRODACTYL_DISTRIBUTED", "0") == "1"
    and int(os.environ.get("SLURM_NTASKS", "1")) > 1
):
    coordinator_address: str | None = os.environ.get(
        "PTYRODACTYL_COORDINATOR_ADDRESS"
    )
    if coordinator_address is not None:
        jax.distributed.initialize(coordinator_address=coordinator_address)
    else:
        jax.distributed.initialize()

jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402, I001
    born,
    invert,
    jacobian,
    simul,
    tools,
    workflows,
)

__version__: str = version("ptyrodactyl")

__all__: list[str] = [
    "born",
    "invert",
    "jacobian",
    "simul",
    "tools",
    "workflows",
]