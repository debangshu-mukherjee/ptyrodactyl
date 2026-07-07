"""Unit-cell geometry and crystallographic helper exports.

Extended Summary
----------------
This package owns unit-cell geometry and crystallographic helpers
without depending on simulation or multislice modules.

Routine Listings
----------------
:func:`reciprocal_lattice`
    Compute reciprocal lattice vectors from real-space unit
    cell.
:func:`rotate_structure`
    Apply rotation transformations to crystal structures.
:func:`rotmatrix_axis`
    Generate a rotation matrix for rotation around an arbitrary
    axis.
:func:`rotmatrix_vectors`
    Compute a rotation matrix that rotates one vector to align
    with another.
:func:`tilt_crystal`
    Tilt :class:`~ptyrodactyl.types.CrystalData` by alpha and
    beta angles (TEM stage-like tilts).
"""

from .unitcell import (
    reciprocal_lattice,
    rotate_structure,
    rotmatrix_axis,
    rotmatrix_vectors,
    tilt_crystal,
)

__all__: list[str] = [
    "reciprocal_lattice",
    "rotate_structure",
    "rotmatrix_axis",
    "rotmatrix_vectors",
    "tilt_crystal",
]
