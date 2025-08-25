"""
Module: ptyrodactyl.electrons
------------------------------
JAX-based electron microscopy simulation toolkit for ptychography and 4D-STEM.

This package implements various electron microscopy components and propagation models
with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
- `atom_potentials`:
    Functions for generating atomic potentials and slices from atomic coordinates
- `geometry`:
    Geometric transformations and operations for crystal structures
- `electron_types`:
    Data structures and type definitions for electron microscopy including
    CalibratedArray, ProbeModes, and PotentialSlices
- `phase_recon`:
    Inverse algorithms for ptychography reconstruction including single-slice,
    position-corrected, and multi-modal reconstruction methods
- `preprocessing`:
    Data preprocessing utilities and type definitions for electron microscopy data
- `simulations`:
    Forward simulation functions for electron beam propagation, CBED patterns,
    and 4D-STEM data generation including aberration calculations and probe creation
- `workflows`:
    High-level workflows that combine multiple simulation steps for common use cases
    such as simulating 4D-STEM data from XYZ structure files
"""

from .atom_potentials import (
    bessel_kv,
    contrast_stretch,
    kirkland_potentials_xyz,
    single_atom_potential,
)
from .electron_types import (
    STEM4D,
    CalibratedArray,
    CrystalStructure,
    PotentialSlices,
    ProbeModes,
    XYZData,
    make_calibrated_array,
    make_crystal_structure,
    make_potential_slices,
    make_probe_modes,
    make_stem4d,
    make_xyz_data,
    non_jax_number,
    scalar_float,
    scalar_int,
    scalar_numeric,
)
from .geometry import (
    reciprocal_lattice,
    rotate_structure,
    rotmatrix_axis,
    rotmatrix_vectors,
)
from .phase_recon import (
    multi_slice_multi_modal,
    single_slice_multi_modal,
    single_slice_poscorrected,
    single_slice_ptychography,
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
from .workflows import xyz_to_4d_stem

__all__: list[str] = [
    "aberration",
    "annular_detector",
    "atomic_symbol",
    "kirkland_potentials",
    "parse_xyz",
    "contrast_stretch",
    "single_atom_potential",
    "kirkland_potentials_xyz",
    "bessel_kv",
    "rotmatrix_vectors",
    "rotmatrix_axis",
    "rotate_structure",
    "reciprocal_lattice",
    "cbed",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "make_probe",
    "propagation_func",
    "shift_beam_fourier",
    "stem_4d",
    "stem_4d_sharded",
    "stem_4d_parallel",
    "transmission_func",
    "wavelength_ang",
    "multi_slice_multi_modal",
    "single_slice_multi_modal",
    "single_slice_poscorrected",
    "single_slice_ptychography",
    "CalibratedArray",
    "PotentialSlices",
    "ProbeModes",
    "STEM4D",
    "CrystalStructure",
    "XYZData",
    "make_calibrated_array",
    "make_potential_slices",
    "make_probe_modes",
    "make_stem4d",
    "make_crystal_structure",
    "make_xyz_data",
    "non_jax_number",
    "scalar_float",
    "scalar_int",
    "scalar_numeric",
    "xyz_to_4d_stem",
]
