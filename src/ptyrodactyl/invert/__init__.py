"""Inverse reconstruction algorithms for electron ptychography.

Extended Summary
----------------
Provides gradient-based optimization routines for reconstructing
sample electrostatic potentials and electron probe functions from
experimental 4D-STEM ptychographic datasets. All functions use
JAX-compatible optimizers and support automatic differentiation.
Supports single-slice and multi-slice reconstructions with
options for position correction and multi-modal probe handling.

Routine Listings
----------------
:func:`multi_slice_multi_modal`
    Multi-slice reconstruction with position correction.
:func:`single_slice_multi_modal`
    Single-slice reconstruction with multi-modal probe and
    position correction.
:func:`single_slice_poscorrected`
    Single-slice reconstruction with position correction.
:func:`single_slice_ptychography`
    Single-slice ptychography reconstruction of potential and
    beam.

Notes
-----
All reconstruction functions use JAX-compatible optimizers and
support automatic differentiation. Input data should be properly
preprocessed and validated using the factory functions from the
:mod:`ptyrodactyl.tools` module.
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
