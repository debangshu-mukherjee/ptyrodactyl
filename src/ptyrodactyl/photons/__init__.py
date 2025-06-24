"""
Module: ptyrodactyl.photons
---------------------------
JAX-based optical simulation toolkit for light microscopes and ptychography.

This package implements various optical components and propagation models
with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
- `engine`:
   Ptychographic Iterative Engine (PIE) based codes.
- `helper`:
    Utility functions for creating grids, phase manipulation, and field calculations
- `invertor`:
    Inversion algorithms for phase retrieval and ptychography.
- `lens_optics`:
    Optical propagation functions including angular spectrum, Fresnel, and Fraunhofer methods
- `lenses`:
    Models for various lens types and their optical properties
- `microscope`:
    Forward propagation of light through optical elements
- `photon_types`:
    Data structures and type definitions for optical propagation
"""

from .engine import (epie_optical, single_pie_iteration, single_pie_sequential,
                     single_pie_vmap)
from .helper import (add_phase_screen, create_spatial_grid, field_intensity,
                     normalize_field, scale_pixel)
from .invertor import get_optimizer, simple_microscope_ptychography
from .lens_optics import (angular_spectrum_prop, circular_aperture,
                          digital_zoom, fraunhofer_prop, fresnel_prop,
                          optical_zoom)
from .lenses import (create_lens_phase, double_concave_lens,
                     double_convex_lens, lens_focal_length,
                     lens_thickness_profile, meniscus_lens, plano_concave_lens,
                     plano_convex_lens, propagate_through_lens)
from .microscope import (lens_propagation, linear_interaction,
                         simple_diffractogram, simple_microscope)
from .photon_types import (Diffractogram, GridParams, LensParams,
                           MicroscopeData, OpticalWavefront, SampleFunction,
                           make_diffractogram, make_grid_params,
                           make_lens_params, make_microscope_data,
                           make_optical_wavefront, make_sample_function,
                           non_jax_number, scalar_float, scalar_int,
                           scalar_num)

__all__ = [
    "epie_optical",
    "single_pie_iteration",
    "single_pie_vmap",
    "single_pie_sequential",
    "add_phase_screen",
    "create_spatial_grid",
    "field_intensity",
    "normalize_field",
    "scale_pixel",
    "get_optimizer",
    "simple_microscope_ptychography",
    "angular_spectrum_prop",
    "circular_aperture",
    "digital_zoom",
    "fraunhofer_prop",
    "fresnel_prop",
    "optical_zoom",
    "create_lens_phase",
    "double_concave_lens",
    "double_convex_lens",
    "lens_focal_length",
    "lens_thickness_profile",
    "meniscus_lens",
    "plano_concave_lens",
    "plano_convex_lens",
    "propagate_through_lens",
    "lens_propagation",
    "linear_interaction",
    "simple_diffractogram",
    "simple_microscope",
    "Diffractogram",
    "GridParams",
    "LensParams",
    "MicroscopeData",
    "OpticalWavefront",
    "SampleFunction",
    "make_diffractogram",
    "make_grid_params",
    "make_lens_params",
    "make_microscope_data",
    "make_optical_wavefront",
    "make_sample_function",
    "non_jax_number",
    "scalar_float",
    "scalar_int",
    "scalar_num",
]
