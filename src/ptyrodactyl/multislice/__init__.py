"""Multislice-family forward simulation toolkit.

Extended Summary
----------------
This subpackage owns multislice forward simulation for electron microscopy,
including CBED amplitudes and intensities, 4D-STEM data generation, sharded
atom-slice entry points, checked wrappers, distribution producers, and
late detector reducers. It is named for the algorithm family rather than a
generic simulation bucket; :mod:`ptyrodactyl.born` remains a sibling
subpackage for convergent Born series simulations.

Submodules
----------
- :mod:`atom_potentials`
    Functions for generating atomic potentials and slices from
    coordinates.
- :mod:`checked`
    JIT-safe validating wrappers for simulation kernels.
- :mod:`parallelized`
    Sharded simulation functions for distributed computing.
- :mod:`producers`
    Distribution producers and CBED axis binders.
- :mod:`reduce`
    Distribution-axis reducers for detector intensity formation.
- :mod:`simulations`
    Forward simulation functions for electron beam propagation,
    CBED patterns, and 4D-STEM data generation with aberration
    calculations.

The submodules are organized as follows:

- :mod:`atom_potentials`
    Atomic potential calculations for electron microscopy.
- :mod:`checked`
    JIT-safe validating wrappers for simulation kernels.
- :mod:`parallelized`
    Parallelized simulation functions for distributed microscopy.
- :mod:`producers`
    Distribution producers and CBED axis binders.
- :mod:`reduce`
    Distribution-axis reducers for detector intensity formation.
- :mod:`simulations`
    Forward simulation functions for electron microscopy.

Routine Listings
----------------
:func:`aberration`
    Calculate aberration phase for the electron probe.
:func:`annular_detector`
    Simulate annular detector for STEM imaging.
:func:`apply_distribution`
    Reduce one weighted distribution axis to detector intensity.
:func:`apply_distributions`
    Reduce multiple weighted distribution axes to detector
    intensity.
:func:`bessel_kv`
    Modified Bessel function of the second kind K_v(x).
:func:`bind_cbed_axes`
    Bind distribution cursor rows to the single-mode CBED
    amplitude kernel.
:func:`cbed_amplitude`
    Simulate complex CBED detector amplitudes.
:func:`cbed_image`
    Simulate convergent beam electron diffraction intensity patterns.
:func:`checked_cbed_image`
    Validate CBED inputs and run the bare CBED intensity kernel.
:func:`checked_make_probe`
    Validate probe-construction inputs and run the bare probe
    kernel.
:func:`checked_stem4d_sharded`
    Validate sharded 4D-STEM inputs and run the bare sharded
    4D-STEM kernel.
:func:`checked_stem_4d`
    Validate 4D-STEM inputs and run the bare 4D-STEM kernel.
:func:`coherence_to_distribution`
    Build the incoherent chromatic/angular coherence distribution.
:func:`decompose_beam_to_modes`
    Decompose electron beam into orthogonal modes.
:func:`fourier_calib`
    Calculate Fourier space calibration from real space.
:func:`fourier_coords`
    Generate Fourier space coordinate arrays.
:func:`kirkland_potentials_crystal`
    Convert :class:`~ptyrodactyl.types.CrystalData` to
    :class:`~ptyrodactyl.types.PotentialSlices`.
:func:`make_probe`
    Create electron probe with specified aberrations.
:func:`position_jitter_to_distribution`
    Build the incoherent two-dimensional position-jitter
    distribution.
:func:`probe_modes_to_distribution`
    Return the explicit incoherent distribution for probe modes.
:func:`propagation_func`
    Compute Fresnel propagation function.
:func:`shift_beam_fourier`
    Shift electron beam in Fourier space for scanning.
:func:`single_atom_potential`
    Projected potential of a single atom via Kirkland
    parameterization.
:func:`stem4d_sharded`
    Generate 4D-STEM data from sharded beams and atom
    coordinates.
:func:`stem_4d`
    Generate 4D-STEM data with multiple probe positions.
:func:`transmission_func`
    Calculate transmission function for a potential slice.

Notes
-----
All simulation functions are JAX-compatible and support automatic
differentiation. The module is designed to be extensible for new
simulation methods and can be used for both forward modeling and
gradient-based reconstruction algorithms.
"""

from ptyrodactyl.multislice.simulations import (
    aberration,
    annular_detector,
    bind_cbed_axes,
    cbed_amplitude,
    cbed_image,
    decompose_beam_to_modes,
    fourier_calib,
    fourier_coords,
    make_probe,
    probe_modes_to_distribution,
    propagation_func,
    shift_beam_fourier,
    stem_4d,
    transmission_func,
)

from .atom_potentials import (
    bessel_kv,
    kirkland_potentials_crystal,
    single_atom_potential,
)
from .checked import (
    checked_cbed_image,
    checked_make_probe,
    checked_stem4d_sharded,
    checked_stem_4d,
)
from .parallelized import stem4d_sharded
from .producers import (
    coherence_to_distribution,
    position_jitter_to_distribution,
)
from .reduce import apply_distribution, apply_distributions

__all__: list[str] = [
    "aberration",
    "annular_detector",
    "apply_distribution",
    "apply_distributions",
    "bessel_kv",
    "bind_cbed_axes",
    "cbed_amplitude",
    "cbed_image",
    "checked_cbed_image",
    "checked_make_probe",
    "checked_stem4d_sharded",
    "checked_stem_4d",
    "coherence_to_distribution",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "kirkland_potentials_crystal",
    "make_probe",
    "position_jitter_to_distribution",
    "probe_modes_to_distribution",
    "propagation_func",
    "shift_beam_fourier",
    "single_atom_potential",
    "stem_4d",
    "stem4d_sharded",
    "transmission_func",
]
