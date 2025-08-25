"""
Module: ptyrodactyl
-------------------
Ptychography through differentiable programming in JAX.

A comprehensive toolkit for ptychography simulations and reconstructions
using JAX for automatic differentiation and acceleration. Supports both
optical and electron microscopy applications with fully differentiable
and JIT-compilable functions.

Submodules
----------
- `electrons`:
    Electron microscopy simulation and ptychography reconstruction
    including CBED patterns, 4D-STEM data generation, and inverse algorithms
- `photons`:
    Optical microscopy simulation and ptychography reconstruction
    including wavefront propagation, lens optics, and optical ptychography
- `tools`:
    Utility tools for optimization, loss functions, and parallel processing
    including complex-valued optimizers with Wirtinger derivatives

Key Features
------------
- JAX-compatible: All functions support jit, grad, vmap, and other JAX transformations
- Automatic differentiation: Full support for gradient-based optimization
- Complex-valued optimization: Wirtinger calculus for complex parameters
- Multi-modal support: Handles both single and multi-modal probes
- Parallel processing: Device mesh support for distributed computing
- Type safety: Comprehensive type checking with jaxtyping and beartype

Examples
--------
Basic usage for electron ptychography:
    >>> from ptyrodactyl.electrons import stem_4D, single_slice_ptychography
    >>> # Generate 4D-STEM data
    >>> data = stem_4D(potential, probe, positions)
    >>> # Reconstruct sample
    >>> reconstructed = single_slice_ptychography(data, initial_guess, positions)

Basic usage for optical ptychography:
    >>> from ptyrodactyl.photons import simple_microscope, simple_microscope_ptychography
    >>> # Simulate optical data
    >>> data = simple_microscope(sample, wavefront, positions)
    >>> # Reconstruct sample
    >>> reconstructed = simple_microscope_ptychography(data, initial_guess, positions)

Notes
-----
This package is designed for research and development in ptychography.
All functions are optimized for JAX transformations and support both
CPU and GPU execution. For best performance, use JIT compilation
and consider using the provided factory functions for data validation.
"""

from . import electrons, photons, tools

__all__: list[str] = [
    "electrons",
    "photons",
    "tools",
]
