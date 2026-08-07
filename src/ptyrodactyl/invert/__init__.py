"""Inverse reconstruction algorithms for electron ptychography.

Extended Summary
----------------
Provides gradient-based optimization routines for reconstructing
sample electrostatic potentials and electron probe functions from
experimental 4D-STEM ptychographic datasets. All functions use
JAX-compatible optimizers and support automatic differentiation.
Supports single-slice and multi-slice reconstructions with
options for position correction and multi-modal probe handling.

The submodules are organized as follows:

- :mod:`phase_recon`
    Inverse reconstruction algorithms for electron ptychography.

Routine Listings
----------------
:func:`multi_slice_multi_modal`
    Reconstruct potential, beam, and positions with multi-slice.
:func:`single_slice_multi_modal`
    Reconstruct potential, multi-modal beam, and positions.
:func:`single_slice_poscorrected`
    Reconstruct potential, beam, and positions from 4D-STEM data.
:func:`single_slice_ptychography`
    Reconstruct potential and beam from 4D-STEM data.
:obj:`OPTIMIZERS`
    Registry mapping optimizer name strings to
    :class:`~ptyrodactyl.tools.Optimizer` instances.

Notes
-----
All reconstruction functions use JAX-compatible optimizers and
support automatic differentiation. Input data should be properly
preprocessed and validated using the factory functions from the
:mod:`ptyrodactyl.tools` module.
"""

from .phase_recon import (
    OPTIMIZERS,
    multi_slice_multi_modal,
    single_slice_multi_modal,
    single_slice_poscorrected,
    single_slice_ptychography,
)

__all__: list[str] = [
    "OPTIMIZERS",
    "multi_slice_multi_modal",
    "single_slice_multi_modal",
    "single_slice_poscorrected",
    "single_slice_ptychography",
]
