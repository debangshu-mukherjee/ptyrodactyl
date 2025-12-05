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
parallelized
    Sharded simulation functions for distributed computing.
preprocessing
    Data preprocessing utilities and type definitions for microscopy data.
simulations
    Forward simulation functions for electron beam propagation, CBED
    patterns, and 4D-STEM data generation with aberration calculations.

Routine Listings
----------------
aberration : function
    Calculate aberration phase from aberration coefficients.
annular_detector : function
    Create annular detector mask for STEM imaging.
atomic_symbol : function
    Convert atomic number to chemical symbol.
bessel_kv : function
    Modified Bessel function of second kind.
cbed : function
    Generate convergent beam electron diffraction patterns.
clip_cbed : function
    Clip CBED patterns to mrad extent and resize to target shape.
contrast_stretch : function
    Contrast stretch for visualization.
decompose_beam_to_modes : function
    Decompose electron beam into orthogonal modes.
fourier_calib : function
    Calculate Fourier space calibration from real space.
fourier_coords : function
    Generate Fourier space coordinate arrays.
kirkland_potentials : function
    Kirkland atomic potential parameters lookup.
kirkland_potentials_crystal : function
    Generate atomic potentials from crystal data using Kirkland parameters.
make_probe : function
    Create electron probe with specified aberrations.
parse_crystal : function
    Parse XYZ or POSCAR file, auto-detecting format, returns CrystalData.
parse_poscar : function
    Parse VASP POSCAR file and return validated structure data.
parse_xyz : function
    Parse XYZ file and return validated structure data.
propagation_func : function
    Compute Fresnel propagation function.
reciprocal_lattice : function
    Calculate reciprocal lattice vectors from real space lattice.
rotate_structure : function
    Rotate crystal structure by specified angles.
rotmatrix_axis : function
    Create rotation matrix from axis and angle.
rotmatrix_vectors : function
    Create rotation matrix from two vectors.
shift_beam_fourier : function
    Shift beam in Fourier space.
single_atom_potential : function
    Calculate single atom potential using Kirkland parameterization.
stem_4d : function
    Generate 4D-STEM data from potential slices and probe.
stem4d_sharded : function
    Generate 4D-STEM data from sharded beams with on-the-fly slice generation.
transmission_func : function
    Compute transmission function for a potential slice.
wavelength_ang : function
    Calculate electron wavelength in Angstroms from accelerating voltage.

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
    kirkland_potentials_crystal,
    single_atom_potential,
)
from .geometry import (
    reciprocal_lattice,
    rotate_structure,
    rotmatrix_axis,
    rotmatrix_vectors,
)
from .parallelized import clip_cbed, stem4d_sharded
from .preprocessing import (
    atomic_symbol,
    kirkland_potentials,
    parse_crystal,
    parse_poscar,
    parse_xyz,
)
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
    transmission_func,
    wavelength_ang,
)

__all__: list[str] = [
    "aberration",
    "annular_detector",
    "atomic_symbol",
    "bessel_kv",
    "cbed",
    "clip_cbed",
    "contrast_stretch",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "kirkland_potentials",
    "kirkland_potentials_crystal",
    "make_probe",
    "parse_crystal",
    "parse_poscar",
    "parse_xyz",
    "propagation_func",
    "reciprocal_lattice",
    "rotate_structure",
    "rotmatrix_axis",
    "rotmatrix_vectors",
    "shift_beam_fourier",
    "single_atom_potential",
    "stem_4d",
    "stem4d_sharded",
    "transmission_func",
    "wavelength_ang",
]
