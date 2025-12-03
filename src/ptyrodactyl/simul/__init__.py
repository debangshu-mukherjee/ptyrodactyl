"""JAX-based electron microscopy simulation toolkit.

Extended Summary
----------------
This package implements various electron microscopy components and propagation
models with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
atom_potentials
    Functions for generating atomic potentials and slices from coordinates.
geometry
    Geometric transformations and operations for crystal structures.
preprocessing
    Data preprocessing utilities and type definitions for microscopy data.
simulations
    Forward simulation functions for electron beam propagation, CBED
    patterns, and 4D-STEM data generation with aberration calculations.

Routine Listings
----------------
stem_4d : function
    Generate 4D-STEM data from potential slices and probe.
make_probe : function
    Create electron probe with specified aberrations.
cbed : function
    Generate convergent beam electron diffraction patterns.
transmission_func : function
    Compute transmission function for a potential slice.
propagation_func : function
    Compute Fresnel propagation function.
kirkland_potentials_xyz : function
    Generate atomic potentials from XYZ coordinates using Kirkland parameters.
parse_xyz : function
    Parse XYZ file and return validated structure data.

Notes
-----
All simulation functions are JAX-compatible and support automatic
differentiation. The module is designed to be extensible for new
simulation methods and can be used for both forward modeling and
gradient-based reconstruction algorithms.
"""

from .atom_potentials import (
    bessel_kv,
    contrast_stretch,
    kirkland_potentials_xyz,
    single_atom_potential,
)
from .geometry import (
    reciprocal_lattice,
    rotate_structure,
    rotmatrix_axis,
    rotmatrix_vectors,
)
from .preprocessing import atomic_symbol, kirkland_potentials, parse_xyz
from .simulations import (
    aberration,
    annular_detector,
    cbed,
    decompose_beam_to_modes,
    fourier_calib,
    fourier_coords,
    make_probe,
    propagation_func,
    shift_beam_fourier,
    stem_4d,
    stem_4d_parallel,
    stem_4d_sharded,
    transmission_func,
    wavelength_ang,
)

__all__: list[str] = [
    "bessel_kv",
    "contrast_stretch",
    "kirkland_potentials_xyz",
    "single_atom_potential",    
    "reciprocal_lattice",
    "rotate_structure",
    "rotmatrix_axis",
    "rotmatrix_vectors",
    "atomic_symbol",
    "kirkland_potentials",
    "parse_xyz",
    "aberration",
    "annular_detector",
    "cbed",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "make_probe",
    "propagation_func",
    "shift_beam_fourier",
    "stem_4d",
    "stem_4d_parallel",
    "stem_4d_sharded",
    "transmission_func",
    "wavelength_ang",
]
