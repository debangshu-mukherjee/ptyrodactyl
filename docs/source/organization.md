# Package Organization

## Overview

Ptyrodactyl is a typed and tested JAX-based library for electron microscopy and ptychography, utilizing JAX's capabilities for multi-device computation and autodifferentiation. The package is organized into four main modules: `simul` for electron microscopy simulations, `invert` for reconstruction algorithms, `tools` for common utilities including optimizers and loss functions, and `workflows` for high-level meta-functions that orchestrate complete simulation pipelines.

## Module Structure

### **ptyrodactyl.simul**

JAX-based electron microscopy simulation toolkit for ptychography and 4D-STEM.

### **ptyrodactyl.invert**

Inverse reconstruction algorithms for electron ptychography including single-slice, position-corrected, and multi-modal reconstruction methods.

### **ptyrodactyl.tools**

Common utilities and shared data structures used throughout the package, including complex-valued optimizers with Wirtinger derivatives.

### **ptyrodactyl.workflows**

High-level workflow functions that combine primitives from `simul` and data structures from `tools` into complete end-to-end pipelines. These meta-functions provide convenient interfaces for common tasks like running full 4D-STEM simulations from structure files.

## Design Principles

### 1. **JAX-First Architecture**
All functions are designed to be:
- **Differentiable**: Full support for `jax.grad`
- **JIT-compilable**: Optimized with `jax.jit`
- **Vectorizable**: Compatible with `jax.vmap`

### 2. **Type Safety**
- Type hints using `jaxtyping`
- Runtime type checking with `beartype`
- Using `PyTrees` for data containers
- `PyTrees` are loaded with type-checked factory functions.

## File Organization

The package structure is organized for clarity and maintainability:

```text
src/ptyrodactyl/
├── __init__.py           # Top-level exports
├── simul/
│   ├── __init__.py       # Simul module exports
│   ├── atom_potentials.py # Atomic potential calculations
│   ├── geometry.py       # Geometric transformations
│   ├── preprocessing.py  # Data preprocessing utilities
│   └── simulations.py    # Forward simulation functions
├── invert/
│   ├── __init__.py       # Invert module exports
│   └── phase_recon.py    # Phase reconstruction algorithms
├── tools/
│   ├── __init__.py       # Tools module exports
│   ├── electron_types.py # Data structures and type definitions
│   ├── loss_functions.py # Loss function definitions
│   ├── optimizers.py     # Complex-valued optimizers
│   └── parallel.py       # Parallel processing utilities
└── workflows/
    ├── __init__.py       # Workflows module exports
    └── stem_4d.py        # End-to-end 4D-STEM workflows
```

## Extension Points

The package is designed to be extensible:

1. **Custom Loss Functions**: Implement new loss functions following the pattern in `tools.loss_functions`
2. **New Optimizers**: Add optimizers with Wirtinger derivative support
3. **Additional Reconstructions**: Build on base reconstruction algorithms in `invert.phase_recon`
4. **Custom Workflows**: Combine existing functions for specific use cases

## Future Directions

The package architecture supports future extensions:
- Real-time microscopy inversion
- Additional electron microscopy modalities
- Machine learning-enhanced reconstructions
