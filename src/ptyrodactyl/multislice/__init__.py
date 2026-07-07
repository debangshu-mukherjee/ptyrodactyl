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
    Calculate aberration phase from aberration coefficients.
:func:`annular_detector`
    Create annular detector mask for STEM imaging.
:func:`apply_distribution`
    Reduce one weighted distribution axis to detector intensity.
:func:`apply_distributions`
    Reduce multiple weighted distribution axes to detector
    intensity.
:func:`bessel_kv`
    Modified Bessel function of the second kind.
:func:`bind_cbed_axes`
    Bind distribution cursor rows to the single-mode CBED
    amplitude kernel.
:func:`cbed_amplitude`
    Generate complex convergent beam electron diffraction amplitudes.
:func:`cbed_image`
    Generate convergent beam electron diffraction intensity patterns.
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
    Generate atomic potentials from crystal data using
    Kirkland parameters.
:func:`make_probe`
    Create electron probe with specified aberrations.
:func:`position_jitter_to_distribution`
    Build the incoherent two-dimensional position-jitter
    distribution.
:func:`propagation_func`
    Compute Fresnel propagation function.
:func:`shift_beam_fourier`
    Shift beam in Fourier space.
:func:`single_atom_potential`
    Calculate single atom potential using Kirkland
    parameterization.
:func:`stem4d_sharded`
    Generate 4D-STEM data from sharded beams with on-the-fly
    slice generation.
:func:`stem_4d`
    Generate 4D-STEM data from potential slices and probe.
:func:`transmission_func`
    Compute transmission function for a potential slice.

Notes
-----
All simulation functions are JAX-compatible and support automatic
differentiation. The module is designed to be extensible for new
simulation methods and can be used for both forward modeling and
gradient-based reconstruction algorithms.
"""

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
    bind_cbed_axes,
    coherence_to_distribution,
    position_jitter_to_distribution,
)
from .reduce import apply_distribution, apply_distributions
from ptyrodactyl.multislice.simulations import (
    aberration,
    annular_detector,
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
