"""Multislice-family forward simulation toolkit.

Extended Summary
----------------
This subpackage owns multislice forward simulation for electron microscopy,
including CBED amplitudes and intensities, 4D-STEM data generation, sharded
atom-slice entry points, checked wrappers, distribution producers, and
late detector reducers. It is named for the algorithm family rather than a
generic simulation bucket. :mod:`ptyrodactyl.galerkin` and
:mod:`ptyrodactyl.born` remain distinct sibling forward families.

The submodules are organized as follows:

- :mod:`atom_potentials`
    Atomic potential calculations for electron microscopy.
- :mod:`checked`
    JIT-safe validating wrappers for simulation kernels.
- :mod:`form_factors`
    Atomic form factors and projected potentials.
- :mod:`multislice_recon`
    Gradient-based reconstruction algorithms for multislice data.
- :mod:`parallelized`
    Parallelized simulation functions for distributed microscopy.
- :mod:`potential_volume`
    Band-limited volumetric independent-atom potential builders.
- :mod:`producers`
    Distribution producers and CBED axis binders.
- :mod:`reduce`
    Distribution-axis reducers for detector intensity formation.
- :mod:`simulations`
    Forward simulation functions for electron microscopy.

Routine Listings
----------------
:func:`aberration`
    Calculate the aberration phase for the electron probe.
:func:`annular_detector`
    Integrate 4D-STEM data with an annular detector.
:func:`apply_distribution`
    Apply the late detector reduction for one distribution axis.
:func:`apply_distributions`
    Apply the late detector reduction for multiple distribution axes.
:func:`atomic_form_factor`
    Evaluate an atomic form factor with Lobato as the default.
:func:`bessel_kv`
    Compute the modified Bessel function :math:`K_v(x)`.
:func:`bind_cbed_axes`
    Return a CBED amplitude closure bound to distribution-axis columns.
:func:`cbed_amplitude`
    Return complex CBED detector fields for each probe mode.
:func:`cbed_amplitude_from_atoms`
    Compute CBED detector amplitudes with on-the-fly slice generation.
:func:`cbed_image`
    Simulate a CBED intensity image via explicit mode reduction.
:func:`cbed_image_from_atoms`
    Compute CBED intensity from atom slices through the reducer.
:func:`checked_cbed_image`
    Validate CBED inputs and run the bare CBED intensity kernel.
:func:`checked_make_probe`
    Validate probe-construction inputs and run the bare probe kernel.
:func:`checked_stem4d_sharded`
    Validate sharded 4D-STEM inputs and run the bare sharded kernel.
:func:`checked_stem_4d`
    Validate 4D-STEM inputs and run the bare 4D-STEM kernel.
:func:`coherence_to_distribution`
    Build an incoherent chromatic/angular coherence distribution.
:func:`crystal_potential_slices`
    Convert :class:`~ptyrodactyl.types.CrystalData` to potential slices.
:func:`crystal_potential_volume`
    Build a full 3D IAM voltage field from atom positions.
:func:`decompose_beam_to_modes`
    Decompose an electron beam into orthogonal modes.
:func:`fourier_calib`
    Compute Fourier-space calibration from real-space parameters.
:func:`fourier_coords`
    Generate Fourier space coordinate arrays.
:func:`kirkland_form_factor`
    Evaluate the Kirkland electron form factor.
:func:`kirkland_projected_potential`
    Evaluate the Kirkland projected electrostatic potential.
:func:`lobato_bandlimited_peak`
    Evaluate the on-nucleus peak of a band-limited Lobato potential.
:func:`lobato_form_factor`
    Evaluate the Lobato--Van Dyck electron form factor.
:func:`lobato_projected_potential`
    Evaluate the Lobato--Van Dyck projected electrostatic potential.
:func:`make_probe`
    Create an electron probe with spherical aberrations.
:func:`multi_slice_multi_modal`
    Reconstruct potential, beam, and positions with multi-slice.
:func:`position_jitter_to_distribution`
    Build an incoherent two-dimensional position-jitter distribution.
:func:`probe_modes_to_distribution`
    Return the explicit incoherent distribution for probe modes.
:func:`projected_atom_potential`
    Evaluate an atomic projected potential with Lobato as the default.
:func:`propagation_func`
    Compute the Fresnel propagation function for multislice.
:func:`shift_beam_fourier`
    Shift beam to new position(s) via Fourier phase ramp.
:func:`single_atom_potential`
    Compute projected potential of a single atom.
:func:`single_atom_potential_3d`
    Build one band-limited three-dimensional atomic potential.
:func:`single_slice_multi_modal`
    Reconstruct potential, multi-modal beam, and positions.
:func:`single_slice_poscorrected`
    Reconstruct potential, beam, and positions from 4D-STEM data.
:func:`single_slice_ptychography`
    Reconstruct potential and beam from 4D-STEM data.
:func:`stem4d_sharded`
    Generate 4D-STEM data with on-the-fly beam shifting and slices.
:func:`stem_4d`
    Generate 4D-STEM data at multiple probe positions.
:func:`transmission_func`
    Calculate the complex transmission function of a potential slice.
:obj:`OPTIMIZERS`
    Registry mapping optimizer name strings to Optax
    gradient-transformation factories.

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
    crystal_potential_slices,
    single_atom_potential,
)
from .checked import (
    checked_cbed_image,
    checked_make_probe,
    checked_stem4d_sharded,
    checked_stem_4d,
)
from .form_factors import (
    atomic_form_factor,
    kirkland_form_factor,
    kirkland_projected_potential,
    lobato_bandlimited_peak,
    lobato_form_factor,
    lobato_projected_potential,
    projected_atom_potential,
)
from .multislice_recon import (
    OPTIMIZERS,
    multi_slice_multi_modal,
    single_slice_multi_modal,
    single_slice_poscorrected,
    single_slice_ptychography,
)
from .parallelized import (
    cbed_amplitude_from_atoms,
    cbed_image_from_atoms,
    stem4d_sharded,
)
from .potential_volume import (
    crystal_potential_volume,
    single_atom_potential_3d,
)
from .producers import (
    coherence_to_distribution,
    position_jitter_to_distribution,
)
from .reduce import apply_distribution, apply_distributions

__all__: list[str] = [
    "OPTIMIZERS",
    "aberration",
    "annular_detector",
    "apply_distribution",
    "apply_distributions",
    "atomic_form_factor",
    "bessel_kv",
    "bind_cbed_axes",
    "cbed_amplitude",
    "cbed_amplitude_from_atoms",
    "cbed_image",
    "cbed_image_from_atoms",
    "checked_cbed_image",
    "checked_make_probe",
    "checked_stem4d_sharded",
    "checked_stem_4d",
    "coherence_to_distribution",
    "crystal_potential_slices",
    "crystal_potential_volume",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "kirkland_form_factor",
    "kirkland_projected_potential",
    "lobato_bandlimited_peak",
    "lobato_form_factor",
    "lobato_projected_potential",
    "make_probe",
    "multi_slice_multi_modal",
    "position_jitter_to_distribution",
    "probe_modes_to_distribution",
    "projected_atom_potential",
    "propagation_func",
    "shift_beam_fourier",
    "single_atom_potential",
    "single_atom_potential_3d",
    "single_slice_multi_modal",
    "single_slice_poscorrected",
    "single_slice_ptychography",
    "stem_4d",
    "stem4d_sharded",
    "transmission_func",
]
