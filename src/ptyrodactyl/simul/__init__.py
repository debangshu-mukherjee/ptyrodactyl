"""JAX-based electron microscopy simulation toolkit.

Extended Summary
----------------
This package implements various electron microscopy components
and propagation models with JAX for automatic differentiation
and acceleration. All functions are fully differentiable and
JIT-compilable.

Submodules
----------
atom_potentials
    Functions for generating atomic potentials and slices from
    coordinates.
geometry
    Geometric transformations and operations for crystal
    structures.
parallelized
    Sharded simulation functions for distributed computing.
preprocessing
    Data preprocessing utilities and type definitions for
    microscopy data.
simulations
    Forward simulation functions for electron beam propagation,
    CBED patterns, and 4D-STEM data generation with aberration
    calculations.

Routine Listings
----------------
:func:`aberration`
    Calculate aberration phase from aberration coefficients.
:func:`annular_detector`
    Create annular detector mask for STEM imaging.
:func:`atomic_symbol`
    Convert atomic number to chemical symbol.
:func:`bessel_kv`
    Modified Bessel function of the second kind.
:func:`cbed`
    Generate convergent beam electron diffraction patterns.
:func:`clip_cbed`
    Clip CBED patterns to mrad extent and resize to target
    shape.
:func:`contrast_stretch`
    Contrast stretch for visualization.
:func:`decompose_beam_to_modes`
    Decompose electron beam into orthogonal modes.
:func:`fourier_calib`
    Calculate Fourier space calibration from real space.
:func:`fourier_coords`
    Generate Fourier space coordinate arrays.
:func:`kirkland_potentials`
    Kirkland atomic potential parameters lookup.
:func:`kirkland_potentials_crystal`
    Generate atomic potentials from crystal data using
    Kirkland parameters.
:func:`make_probe`
    Create electron probe with specified aberrations.
:func:`parse_crystal`
    Parse XYZ or POSCAR file, auto-detecting format,
    returns :class:`~ptyrodactyl.tools.CrystalData`.
:func:`parse_poscar`
    Parse VASP POSCAR file and return validated structure
    data.
:func:`parse_xyz`
    Parse XYZ file and return validated structure data.
:func:`propagation_func`
    Compute Fresnel propagation function.
:func:`reciprocal_lattice`
    Calculate reciprocal lattice vectors from real space
    lattice.
:func:`rotate_structure`
    Rotate crystal structure by specified angles.
:func:`rotmatrix_axis`
    Create rotation matrix from axis and angle.
:func:`rotmatrix_vectors`
    Create rotation matrix from two vectors.
:func:`shift_beam_fourier`
    Shift beam in Fourier space.
:func:`single_atom_potential`
    Calculate single atom potential using Kirkland
    parameterization.
:func:`stem_4d`
    Generate 4D-STEM data from potential slices and probe.
:func:`stem4d_sharded`
    Generate 4D-STEM data from sharded beams with on-the-fly
    slice generation.
:func:`tilt_crystal`
    Tilt :class:`~ptyrodactyl.tools.CrystalData` by alpha
    and beta angles (TEM stage-like tilts).
:func:`transmission_func`
    Compute transmission function for a potential slice.
:func:`wavelength_ang`
    Calculate electron wavelength in Angstroms from
    accelerating voltage.

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
    tilt_crystal,
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
    "tilt_crystal",
    "transmission_func",
    "wavelength_ang",
]
