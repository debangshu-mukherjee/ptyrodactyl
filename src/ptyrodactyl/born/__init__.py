"""Convergent Born series simulations.

Extended Summary
----------------
The submodules are organized as follows:

- :mod:`green`
    Fourier-space Green's function for the homogeneous Helmholtz equation.

Routine Listings
----------------
:func:`convergence_parameter`
    Compute the convergence parameter from the scattering potential.
:func:`green_function_fourier`
    Construct the Fourier-space Green's function.
:func:`reciprocal_coords`
    Construct 3-D reciprocal-space coordinate arrays.
:func:`wavenumber_background`
    Compute the optimal background wavenumber squared.

"""

from .green import (
    convergence_parameter,
    green_function_fourier,
    reciprocal_coords,
    wavenumber_background,
)

__all__ = [
    "convergence_parameter",
    "green_function_fourier",
    "reciprocal_coords",
    "wavenumber_background",
]
