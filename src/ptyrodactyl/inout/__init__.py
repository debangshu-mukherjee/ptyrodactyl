"""Host-world ingestion and HDF5 archive exports.

Extended Summary
----------------
This package owns file-ingest helpers for crystal structures, related lookup
data, and the versioned Stage-0 scalar-potential HDF5 archive.

The submodules are organized as follows:

- :mod:`crystal`
    Crystal-structure parser dispatcher.
- :mod:`form_factor_data`
    Bundled Lobato--Van Dyck and Kirkland coefficient tables.
- :mod:`hdf5`
    Scalar-potential HDF5 ingest and emit codec.
- :mod:`poscar`
    VASP POSCAR crystal-structure parsing utilities.
- :mod:`xyz`
    XYZ crystal-structure parsing and atomic data lookups.

Routine Listings
----------------
:class:`HDF5SchemaError`
    Report an incompatible or malformed ptyrodactyl HDF5 archive.
:func:`atomic_symbol`
    Return atomic number for a given atomic symbol string.
:func:`kirkland_potentials`
    Return preloaded Kirkland potential parameters.
:func:`load_from_h5`
    Load one validated scalar-potential carrier from HDF5.
:func:`lobato_potentials`
    Return preloaded Lobato--Van Dyck potential parameters.
:func:`parse_crystal`
    Parse XYZ or POSCAR file, auto-detecting format, returns CrystalData.
:func:`parse_poscar`
    Parse a VASP POSCAR file and return a validated CrystalData PyTree.
:func:`parse_xyz`
    Parse an XYZ file and return a validated CrystalData PyTree.
:func:`save_to_h5`
    Save one validated scalar-potential carrier to HDF5.
"""

from .crystal import parse_crystal
from .form_factor_data import kirkland_potentials, lobato_potentials
from .hdf5 import HDF5SchemaError, load_from_h5, save_to_h5
from .poscar import parse_poscar
from .xyz import atomic_symbol, parse_xyz

__all__: list[str] = [
    "HDF5SchemaError",
    "atomic_symbol",
    "kirkland_potentials",
    "load_from_h5",
    "lobato_potentials",
    "parse_crystal",
    "parse_poscar",
    "parse_xyz",
    "save_to_h5",
]
