# Package Organization

## Overview

Ptyrodactyl is organized into three main modules, each serving a specific purpose in ptychographic reconstruction and simulation. The package follows a clean, hierarchical structure that separates concerns while maintaining ease of use through a well-designed public API.

## Module Structure

### **ptyrodactyl.electrons**
The electron microscopy module provides comprehensive tools for electron ptychography, 4D-STEM simulations, and phase reconstruction.

#### Key Components:
- **Data Types & Structures**
  - `CalibratedArray`: Calibrated array data with spatial calibration
  - `ProbeModes`: Multi-modal electron probe states
  - `PotentialSlices`: Potential slices for multi-slice simulations
  - `CrystalStructure`: Crystal structure with fractional and Cartesian coordinates
  - `XYZData`: Parsed XYZ file data with atomic positions
  - `STEM4D`: 4D-STEM data with diffraction patterns and calibrations

- **Simulation Functions**
  - `stem_4d`: Generate 4D-STEM data from potential and probe
  - `stem_4d_parallel`: Parallel version for multiple probe positions
  - `cbed`: Convergent beam electron diffraction patterns
  - `make_probe`: Create electron probes with aberrations
  - `transmission_func`: Transmission function for potential slices
  - `propagation_func`: Fresnel propagation in Fourier space
  - `wavelength_ang`: Calculate electron wavelength from voltage

- **Phase Reconstruction**
  - `single_slice_ptychography`: Single-slice reconstruction
  - `single_slice_poscorrected`: Position-corrected reconstruction
  - `single_slice_multi_modal`: Multi-modal probe reconstruction
  - `multi_slice_multi_modal`: Multi-slice with multi-modal probes

- **Preprocessing & Utilities**
  - `parse_xyz`: Parse XYZ structure files
  - `kirkland_potentials`: Generate Kirkland atomic potentials
  - `atomic_symbol`: Convert atomic numbers to symbols
  - `rotmatrix_vectors`: Create rotation matrices
  - `reciprocal_lattice`: Calculate reciprocal lattice vectors

- **Workflows**
  - `xyz_to_4d_stem`: Complete pipeline from XYZ to 4D-STEM data

### **ptyrodactyl.photons**
The optical microscopy module handles optical ptychography, wavefront propagation, and lens simulations.

#### Key Components:
- **Data Types & Structures**
  - `OpticalWavefront`: Complex optical field with wavelength and position
  - `LensParams`: Physical lens parameters (focal length, diameter, etc.)
  - `GridParams`: Computational grid for optical elements
  - `MicroscopeData`: Microscopy data with positions and wavelength
  - `SampleFunction`: Sample transmission/phase functions
  - `Diffractogram`: Diffraction pattern data

- **Optical Elements**
  - `apply_lens`: Apply lens phase transformation
  - `apply_aperture`: Apply aperture transmission
  - `apply_beam_splitter`: Beam splitter operations
  - `apply_waveplate`: Wave plate transformations
  - `circular_aperture`: Generate circular apertures
  - `rectangular_aperture`: Generate rectangular apertures

- **Propagation & Simulation**
  - `fresnel_propagate`: Fresnel propagation
  - `angular_spectrum_propagate`: Angular spectrum method
  - `simple_microscope`: Simulate optical microscopy
  - `scanning_microscope`: Scanning microscopy simulation

- **Phase Reconstruction**
  - `simple_microscope_ptychography`: Basic optical ptychography
  - `scanning_microscope_ptychography`: Scanning ptychography

- **Lens Optics**
  - `lens_thickness`: Calculate lens thickness profiles
  - `lens_phase`: Phase transformation for lenses
  - `lens_transmission`: Complex transmission functions

### **ptyrodactyl.tools**
Utility module providing optimization tools and helper functions for both electron and photon modules.

#### Key Components:
- **Optimizers**
  - `Optimizer`: Base optimizer class with Wirtinger derivatives
  - `adam_update`: ADAM optimizer for complex variables
  - `adagrad_update`: AdaGrad optimizer
  - `rmsprop_update`: RMSProp optimizer
  - `sgd_update`: Stochastic gradient descent
  - Learning rate schedulers (exponential, cosine, step)

- **Loss Functions**
  - `mse_loss`: Mean squared error
  - `mae_loss`: Mean absolute error
  - `poisson_loss`: Poisson likelihood
  - `amplitude_loss`: Amplitude-only loss
  - `create_loss_function`: Factory for loss functions

- **Parallel Processing**
  - `create_device_mesh`: Setup for distributed computing
  - `shard_data`: Data sharding across devices
  - `gather_results`: Collect distributed results

## Design Principles

### 1. **JAX-First Architecture**
All functions are designed to be:
- **Differentiable**: Full support for `jax.grad`
- **JIT-compilable**: Optimized with `jax.jit`
- **Vectorizable**: Compatible with `jax.vmap`
- **Device-agnostic**: Run on CPU, GPU, or TPU

### 2. **Type Safety**
- Comprehensive type hints using `jaxtyping`
- Runtime type checking with `beartype`
- Clear array shape specifications

### 3. **Functional Programming**
- Pure functions without side effects
- Immutable data structures (NamedTuples)
- Composable operations

### 4. **Factory Pattern**
All data structures have corresponding factory functions:
- `make_calibrated_array`
- `make_probe_modes`
- `make_potential_slices`
- `make_crystal_structure`
- `make_xyz_data`
- `make_stem4d`

These factories provide:
- Runtime validation
- Type checking
- Default value handling
- JAX array conversion

## File Organization

While the public API presents three clean modules, the internal structure is organized for maintainability:

```
src/ptyrodactyl/
├── __init__.py           # Top-level exports
├── electrons/
│   ├── __init__.py       # Electron module exports
│   ├── electron_types.py # Data structures
│   ├── simulations.py    # Forward simulations
│   ├── phase_recon.py    # Inverse algorithms
│   ├── preprocessing.py  # Data preparation
│   ├── atom_potentials.py # Atomic potentials
│   ├── geometry.py       # Geometric operations
│   └── workflows.py      # Complete pipelines
├── photons/
│   ├── __init__.py       # Photon module exports
│   ├── photon_types.py   # Data structures
│   ├── apertures.py      # Aperture functions
│   ├── elements.py       # Optical elements
│   ├── engine.py         # Propagation engine
│   ├── helper.py         # Utility functions
│   ├── invertor.py       # Reconstruction algorithms
│   ├── lens_optics.py    # Lens calculations
│   ├── lenses.py         # Lens implementations
│   └── microscope.py     # Microscopy simulations
└── tools/
    ├── __init__.py       # Tools module exports
    ├── optimizers.py     # Optimization algorithms
    ├── loss_functions.py # Loss function definitions
    └── parallel.py       # Parallel processing utilities
```

## Import Patterns

### Public API Usage
Users should import from the three main modules:

```python
# Import from main modules
from ptyrodactyl.electrons import stem_4d, single_slice_ptychography
from ptyrodactyl.photons import simple_microscope, OpticalWavefront
from ptyrodactyl.tools import adam_update, mse_loss

# Import entire modules
import ptyrodactyl.electrons as electrons
import ptyrodactyl.photons as photons
import ptyrodactyl.tools as tools
```

### Internal Implementation
The `__init__.py` files handle internal imports and expose a clean API:

```python
# electrons/__init__.py example
from .electron_types import (
    CalibratedArray, ProbeModes, STEM4D,
    make_calibrated_array, make_probe_modes, make_stem4d
)
from .simulations import (
    stem_4d, cbed, make_probe, transmission_func
)
from .phase_recon import (
    single_slice_ptychography, single_slice_poscorrected
)
# ... etc
```

## Best Practices

### 1. **Use Factory Functions**
Always create data structures through factory functions:
```python
# Good
stem_data = make_stem4d(data, real_calib, fourier_calib, positions, voltage)

# Avoid
stem_data = STEM4D(data, real_calib, fourier_calib, positions, voltage)
```

### 2. **Leverage JAX Transformations**
```python
# JIT compilation for performance
@jax.jit
def reconstruct(data, probe, positions):
    return single_slice_ptychography(data, probe, positions)

# Automatic differentiation
grad_fn = jax.grad(loss_function)

# Vectorization
batched_stem = jax.vmap(stem_4d, in_axes=(None, None, 0))
```

### 3. **Type Annotations**
Use type hints for clarity:
```python
def process_data(
    data: Float[Array, "H W"],
    calib: scalar_float
) -> CalibratedArray:
    return make_calibrated_array(data, calib, calib, True)
```

### 4. **Composable Operations**
Build complex operations from simple functions:
```python
# Compose multiple operations
def full_reconstruction(raw_data, initial_guess):
    # Preprocess
    data = preprocess(raw_data)
    
    # Simulate forward model
    simulated = stem_4d(initial_guess, probe, positions)
    
    # Reconstruct
    result = single_slice_ptychography(data, initial_guess)
    
    return result
```

## Performance Considerations

### Memory Management
- Use `jax.checkpoint` for memory-intensive operations
- Leverage `jax.lax.scan` for sequential operations
- Prefer in-place updates with `.at[].set()` for large arrays

### Parallelization
- Use `jax.pmap` for data parallelism
- Implement sharding strategies for large datasets
- Utilize device mesh for distributed computing

### Optimization
- JIT-compile hot paths
- Batch operations with `vmap`
- Use appropriate precision (float32 vs float64)

## Extension Points

The package is designed to be extensible:

1. **Custom Loss Functions**: Implement new loss functions following the pattern in `tools.loss_functions`
2. **New Optimizers**: Add optimizers with Wirtinger derivative support
3. **Additional Reconstructions**: Build on base reconstruction algorithms
4. **Custom Workflows**: Combine existing functions for specific use cases

## Dependencies

### Core Dependencies
- **JAX**: Automatic differentiation and JIT compilation
- **NumPy**: Array operations (via JAX)
- **jaxtyping**: Type annotations for JAX arrays
- **beartype**: Runtime type checking

### Optional Dependencies
- **matplotlib**: Visualization (for examples)
- **scipy**: Additional scientific computing tools
- **h5py**: HDF5 file I/O

## Future Directions

The package architecture supports future extensions:
- Additional reconstruction algorithms
- GPU-optimized kernels
- Real-time processing pipelines
- Integration with experimental data formats
- Machine learning-enhanced reconstructions