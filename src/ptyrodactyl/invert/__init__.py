"""
Module: ptyrodactyl.invert
--------------------------
Inverse reconstruction algorithms for electron ptychography.

This module contains functions for reconstructing sample potentials from
experimental ptychography data using various optimization algorithms.
All functions support both single-slice and multi-slice reconstructions,
with options for position correction and multi-modal probe handling.

Submodules
----------
- `phase_recon`:
    Inverse algorithms for ptychography reconstruction including single-slice,
    position-corrected, and multi-modal reconstruction methods

Functions
---------
- `single_slice_ptychography`:
    Performs single-slice ptychography reconstruction
- `single_slice_poscorrected`:
    Performs single-slice reconstruction with position correction
- `single_slice_multi_modal`:
    Performs single-slice reconstruction with multi-modal probe
- `multi_slice_multi_modal`:
    Performs multi-slice reconstruction with multi-modal probe

Notes
-----
All reconstruction functions use JAX-compatible optimizers and support
automatic differentiation. The functions are designed to work with
experimental data and can handle various noise levels and experimental
conditions. Input data should be properly preprocessed and validated
using the factory functions from the simul.electron_types module.
"""

from .phase_recon import (
    multi_slice_multi_modal,
    single_slice_multi_modal,
    single_slice_poscorrected,
    single_slice_ptychography,
)

__all__: list[str] = [
    "multi_slice_multi_modal",
    "single_slice_multi_modal",
    "single_slice_poscorrected",
    "single_slice_ptychography",
]
