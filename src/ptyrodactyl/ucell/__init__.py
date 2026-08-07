"""Unit-cell geometry and crystallographic helper exports.

Extended Summary
----------------
This package owns unit-cell geometry and crystallographic helpers
without depending on simulation or multislice modules.

The submodules are organized as follows:

- :mod:`unitcell`
    Unit-cell rotation and reciprocal-lattice helpers.

Routine Listings
----------------
:func:`reciprocal_lattice`
    Compute reciprocal lattice vectors from a real-space cell.
:func:`rotate_structure`
    Apply rotation to a crystal structure.
:func:`rotmatrix_axis`
    Generate a rotation matrix around an arbitrary axis.
:func:`rotmatrix_vectors`
    Compute a rotation matrix that rotates v1 to align with v2.
:func:`tilt_crystal`
    Tilt :class:`~ptyrodactyl.types.CrystalData` by alpha and beta.

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
