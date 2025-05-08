"""
Module: ptyrodactyl.photons
==================================================
JAX-based optical simulation toolkit for light microscopes and ptychography.

This package implements various optical components and propagation models
with JAX for automatic differentiation and acceleration. All functions
are fully differentiable and JIT-compilable.

Submodules
----------
- `optics`: 
    Optical propagation functions including angular spectrum, Fresnel, and Fraunhofer methods
- `lenses`: 
    Models for various lens types and their optical properties
- `forward`: 
    Forward propagation of light through optical elements
- `helper`: 
    Utility functions for creating grids, phase manipulation, and field calculations
- `photon_types`: 
    Data structures and type definitions for optical propagation
- `engine`: 
    Framework for building complete optical simulation pipelines

.. currentmodule:: ptyrodactyl.photons
"""

from .engine import *
from .forward import *
from .helper import *
from .lenses import *
from .optics import *
from .photon_types import *
