"""
Module: ptyrodactyl.electrons
-----------------------------
JAX-based electron microscopy simulation toolkit for ptychography and 4D-STEM.

This package implements various electron microscopy components and propagation models
with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
- `forward`:
    Forward simulation functions for electron beam propagation, CBED patterns,
    and 4D-STEM data generation including aberration calculations and probe creation
- `inverse`:
    Inverse algorithms for ptychography reconstruction including single-slice,
    position-corrected, and multi-modal reconstruction methods
- `preprocessing`:
    Data preprocessing utilities and type definitions for electron microscopy data
- `electron_types`:
    Data structures and type definitions for electron microscopy including
    CalibratedArray, ProbeModes, and PotentialSlices
"""

from .electron_types import (CalibratedArray, PotentialSlices, ProbeModes,
                             make_calibrated_array, make_potential_slices,
                             make_probe_modes, non_jax_number, scalar_float,
                             scalar_int, scalar_numeric)
from .forward import (aberration, cbed, decompose_beam_to_modes, fourier_calib,
                      fourier_coords, make_probe, propagation_func,
                      shift_beam_fourier, stem_4D, transmission_func,
                      wavelength_ang)
from .inverse import (get_optimizer, multi_slice_multi_modal,
                      single_slice_multi_modal, single_slice_poscorrected,
                      single_slice_ptychography)

__all__ = [
    "aberration",
    "cbed",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "make_probe",
    "propagation_func",
    "shift_beam_fourier",
    "stem_4D",
    "transmission_func",
    "wavelength_ang",
    "get_optimizer",
    "multi_slice_multi_modal",
    "single_slice_multi_modal",
    "single_slice_poscorrected",
    "single_slice_ptychography",
    "CalibratedArray",
    "PotentialSlices",
    "ProbeModes",
    "make_calibrated_array",
    "make_potential_slices",
    "make_probe_modes",
    "non_jax_number",
    "scalar_float",
    "scalar_int",
    "scalar_numeric",
]
