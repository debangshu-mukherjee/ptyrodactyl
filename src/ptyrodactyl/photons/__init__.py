"""
Module: ptyrodactyl.photons
==================================================
JAX-based optical simulation toolkit for light microscopes and ptychography.

This package implements various optical components and propagation models
with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
- `engine`: 
    Framework for building complete optical simulation pipelines
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

.. currentmodule:: ptyrodactyl.photons
"""

from .engine import *
from .helper import *
from .invertor import *
from .lens_optics import *
from .lenses import *
from .microscope import *
from .photon_types import *
