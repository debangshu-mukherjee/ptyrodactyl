"""Differentiable electron microscopy forward and inverse problems.

Extended Summary
----------------
A comprehensive toolkit for electron ptychography simulations and
reconstructions using JAX for automatic differentiation and acceleration.
All functions are fully differentiable and JIT-compilable.

Submodules
----------
invert
    Electron microscopy reconstructions, ptychography and focal series.
simul
    Electron microscopy simulations including 4D-STEM, CBED, and multislice.
tools
    Utility tools for optimization, loss functions, and parallel processing
    including complex-valued optimizers with Wirtinger derivatives.
workflows
    High-level workflows combining simulation steps for common use cases
    such as simulating 4D-STEM data from XYZ structure files.

Key Features
------------
- JAX-compatible:
    All functions support jit, grad, vmap, and other JAX transformations.
- Automatic differentiation:
    Full support for gradient-based optimization.
- Complex-valued optimization:
    Wirtinger calculus for complex parameters.
- Parallel processing:
    Device mesh support for distributed computing.
- Type safety:
    Comprehensive type checking with jaxtyping and beartype.

Notes
-----
This package is designed for electron microscopy simulations and 
reconstructions. All functions are optimized for JAX transformations and 
support both CPU and GPU execution. For best performance, use JIT compilation
and consider using the provided factory functions for data validation.
"""

import os

# Enable multi-threaded CPU execution for JAX (must be set before JAX import)
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

from . import invert, simul, tools, workflows

__all__: list[str] = [
    "invert",
    "simul",
    "tools",
    "workflows",
]
