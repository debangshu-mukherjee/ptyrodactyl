"""Convergent Born series simulations.

Routine Listings
----------------
:func:`structure_matrix`:
    Assemble the dynamical structure matrix A_gh from Fourier potential and excitation errors.
:func:`excitation_errors`:
    Excitation error s_g for each beam given tilt and beam energy.
:func:`scattering_matrix`:
    Scattering matrix S(t) = expm(i pi lambda A t) for a given thickness.
:func:`bloch_beam_amplitudes`:
    Beam amplitudes psi_g at one thickness from an incident-beam boundary condition.
:func:`bloch_thickness_series`:
    Beam amplitudes across a stack of thicknesses via lax.scan on the propagator.
:func:`two_beam_pendellosung`:
    Closed-form two-beam Pendellosung amplitudes for analytic validation.
:func:`extinction_distance`:
    Two-beam extinction distance xi_g from a single Fourier coefficient.
:func:`fourier_potential_from_grid`:
    Complex Fourier potential coefficients U_g sampled from a real-space potential array.
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
