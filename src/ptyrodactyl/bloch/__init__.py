"""Bloch-wave simulations for dynamical electron diffraction.

Extended Summary
----------------
The submodules are organized as follows:

- :mod:`bloch_forward`
    Bloch-wave forward solver for dynamical electron diffraction.

Routine Listings
----------------
:func:`bloch_beam_amplitudes`
    Compute beam amplitudes at one thickness.
:func:`bloch_thickness_series`
    Compute amplitudes across uniform thickness steps.
:func:`excitation_errors`
    Compute Ewald-sphere excitation errors.
:func:`extinction_distance`
    Compute the two-beam extinction distance.
:func:`fourier_potential_from_grid`
    Sample Fourier potential coefficients from a grid.
:func:`scattering_matrix`
    Propagate beam amplitudes with a matrix exponential.
:func:`structure_matrix`
    Assemble the Bloch dynamical structure matrix.
:func:`two_beam_pendellosung`
    Evaluate the two-beam Pendellosung amplitudes.

"""

from .bloch_forward import (
    bloch_beam_amplitudes,
    bloch_thickness_series,
    excitation_errors,
    extinction_distance,
    fourier_potential_from_grid,
    scattering_matrix,
    structure_matrix,
    two_beam_pendellosung,
)

__all__ = [
    "bloch_beam_amplitudes",
    "bloch_thickness_series",
    "excitation_errors",
    "extinction_distance",
    "fourier_potential_from_grid",
    "scattering_matrix",
    "structure_matrix",
    "two_beam_pendellosung",
]
