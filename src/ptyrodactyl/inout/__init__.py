"""Crystal-structure ingestion and lookup exports.

Extended Summary
----------------
This package owns file-ingest helpers for crystal structures and
related lookup data used by simulations and workflows.

The submodules are organized as follows:

- :mod:`crystal`
    Crystal-structure parser dispatcher.
- :mod:`poscar`
    VASP POSCAR crystal-structure parsing utilities.
- :mod:`xyz`
    XYZ crystal-structure parsing and atomic data lookups.

Routine Listings
----------------
:func:`atomic_symbol`
    Return atomic number for a given atomic symbol string.
:func:`kirkland_potentials`
    Return preloaded Kirkland potential parameters.
:func:`parse_crystal`
    Parse XYZ or POSCAR file, auto-detecting format, returns CrystalData.
:func:`parse_poscar`
    Parse a VASP POSCAR file and return a validated CrystalData PyTree.
:func:`parse_xyz`
    Parse an XYZ file and return a validated CrystalData PyTree.
"""

from .crystal import parse_crystal
from .poscar import parse_poscar
from .xyz import atomic_symbol, kirkland_potentials, parse_xyz

__all__: list[str] = [
    "atomic_symbol",
    "kirkland_potentials",
    "parse_crystal",
    "parse_poscar",
    "parse_xyz",
]
