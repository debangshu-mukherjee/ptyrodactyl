"""Data preprocessing utilities for electron microscopy.

Extended Summary
----------------
This module contains utilities for preprocessing electron
microscopy data before analysis or reconstruction, including
XYZ and POSCAR file parsing and atomic data lookups.

Routine Listings
----------------
:func:`_load_atomic_numbers`
    Load atomic number mapping from JSON file.
:data:`_ATOMIC_NUMBERS`
    Module-level dict mapping symbols to atomic numbers.
:func:`atomic_symbol`
    Return atomic number for a given atomic symbol string.
:func:`_load_kirkland_csv`
    Load Kirkland potential parameters from CSV file.
:data:`_KIRKLAND_POTENTIALS`
    Module-level JAX array of Kirkland parameters.
:func:`kirkland_potentials`
    Return preloaded Kirkland scattering factors as JAX
    array.
:func:`_parse_xyz_metadata`
    Extract metadata from the XYZ comment line.
:func:`parse_xyz`
    Parse an XYZ file and return validated
    :class:`~ptyrodactyl.tools.CrystalData`.
:func:`parse_poscar`
    Parse a VASP POSCAR file and return validated
    :class:`~ptyrodactyl.tools.CrystalData`.
:func:`_extract_elements_from_comment`
    Extract element symbols from POSCAR comment line.
:func:`parse_crystal`
    Parse XYZ or POSCAR file, auto-detecting format.

Notes
-----
Internal functions (prefixed with underscore) handle loading
atomic number mappings, Kirkland potentials from CSV, and
parsing XYZ metadata.
"""

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Union
from jaxtyping import Array, Float, Int, jaxtyped

from ptyrodactyl.tools import CrystalData, ScalarInt, make_crystal_data

_KIRKLAND_PATH: Path = (
    Path(__file__).resolve().parent / "luggage" / "Kirkland_Potentials.csv"
)
_ATOMS_PATH: Path = (
    Path(__file__).resolve().parent / "luggage" / "atom_numbers.json"
)

jax.config.update("jax_enable_x64", True)


@beartype
def _load_atomic_numbers(
    json_path: Optional[Path] = _ATOMS_PATH,
) -> Dict[str, int]:
    """Load atomic number mapping from JSON file in manifest folder.

    Parameters
    ----------
    json_path : Path, optional
        Custom path to JSON file, defaults to module path.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping atomic symbols to atomic numbers.

    Raises
    ------
    FileNotFoundError
        If JSON file is not found.
    json.JSONDecodeError
        If JSON file is malformed.

    Notes
    -----
    Uses pathlib for OS-independent path handling.
    """
    file_path: Path = json_path if json_path is not None else _ATOMS_PATH
    with open(file_path, encoding="utf-8") as file:
        atomic_data: Dict[str, int] = json.load(file)
    return atomic_data


_ATOMIC_NUMBERS: Dict[str, int] = _load_atomic_numbers()


@jaxtyped(typechecker=beartype)
def atomic_symbol(symbol_string: str) -> ScalarInt:
    """Return atomic number for a given atomic symbol string.

    Implementation Logic
    --------------------
    1. **Strip and normalize** --
       Remove whitespace and capitalize the symbol.
    2. **Look up** --
       Find atomic number in the preloaded
       :data:`_ATOMIC_NUMBERS` mapping.

    Parameters
    ----------
    symbol_string : str
        Chemical symbol for the element (e.g., ``"H"``,
        ``"He"``, ``"Li"``).

    Returns
    -------
    atomic_number : ScalarInt
        Atomic number corresponding to the symbol.

    Raises
    ------
    KeyError
        If atomic symbol is not found in the mapping.
    ValueError
        If atomic symbol is empty.
    """
    cleaned_symbol: str = symbol_string.strip()

    if not cleaned_symbol:
        raise ValueError("Atomic symbol cannot be empty")

    normalized_symbol: str = cleaned_symbol.capitalize()
    if normalized_symbol not in _ATOMIC_NUMBERS:
        available_symbols: str = ", ".join(sorted(_ATOMIC_NUMBERS.keys()))
        raise KeyError(
            f"Atomic symbol '{symbol_string}' not found. Available symbols: {available_symbols}"
        )

    atomic_number: ScalarInt = _ATOMIC_NUMBERS[normalized_symbol]
    return atomic_number


@jaxtyped(typechecker=beartype)
def _load_kirkland_csv(
    file_path: Optional[Path] = _KIRKLAND_PATH,
) -> Float[Array, "103 12"]:
    """Load Kirkland potential parameters from CSV file.

    Parameters
    ----------
    file_path : Path, optional
        Custom path to CSV file, defaults to module path.

    Returns
    -------
    Float[Array, "103 12"]
        Kirkland potential parameters as JAX array.

    Raises
    ------
    FileNotFoundError
        If CSV file is not found.
    ValueError
        If CSV dimensions are incorrect.

    Notes
    -----
    Uses numpy to load CSV then converts to JAX array for performance.
    """
    kirkland_numpy: np.ndarray = np.loadtxt(  # type: ignore[call-overload]
        file_path, delimiter=",", dtype=np.float64
    )
    if kirkland_numpy.shape != (103, 12):
        raise ValueError(
            f"Expected CSV shape (103, 12), got {kirkland_numpy.shape}"
        )
    kirkland_data: Float[Array, "103 12"] = jnp.asarray(
        kirkland_numpy, dtype=jnp.float64
    )
    return kirkland_data


_KIRKLAND_POTENTIALS: Float[Array, "103 12"] = _load_kirkland_csv()


@jaxtyped(typechecker=beartype)
def kirkland_potentials() -> Float[Array, "103 12"]:
    """Return preloaded Kirkland potential parameters.

    Returns
    -------
    kirkland_data : Float[Array, "103 12"]
        Kirkland potential parameters for elements 1--103.

    Notes
    -----
    Data is loaded once at module import time from
    :data:`_KIRKLAND_POTENTIALS`. No file I/O on each call.
    """
    return _KIRKLAND_POTENTIALS


@beartype
def _parse_xyz_metadata(line: str) -> Dict[str, Any]:
    """Extract metadata from the XYZ comment line.

    Parameters
    ----------
    line : str
        Second line of the XYZ file (comment/metadata).

    Returns
    -------
    Dict[str, Any]
        Parsed metadata with optional keys: lattice,
        stress, energy, properties.

    Raises
    ------
    ValueError
        If lattice or stress tensor dimensions are incorrect.
    """
    metadata: Dict[str, Any] = {}
    max_xyz_columns: int = 9
    lattice_match: Optional[re.Match[str]] = re.search(
        r'Lattice="([^"]+)"', line
    )
    if lattice_match:
        values: List[float] = list(map(float, lattice_match.group(1).split()))
        if len(values) != max_xyz_columns:
            raise ValueError("Lattice must contain 9 values")
        metadata["lattice"] = jnp.array(values, dtype=jnp.float64).reshape(
            3, 3
        )

    stress_match: Optional[re.Match[str]] = re.search(
        r'stress="([^"]+)"', line
    )
    if stress_match:
        values: List[float] = list(map(float, stress_match.group(1).split()))
        if len(values) != max_xyz_columns:
            raise ValueError("Stress tensor must contain 9 values")
        metadata["stress"] = jnp.array(values, dtype=jnp.float64).reshape(3, 3)

    energy_match: Optional[re.Match[str]] = re.search(
        r"energy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line
    )
    if energy_match:
        metadata["energy"] = float(energy_match.group(1))

    props_match: Optional[re.Match[str]] = re.search(
        r"Properties=([^ ]+)", line
    )
    if props_match:
        raw_props: str = props_match.group(1)
        parts: List[str] = raw_props.split(":")
        props: List[Dict[str, Union[str, int]]] = []
        for i in range(0, len(parts), 3):
            props.append(
                {
                    "name": parts[i],
                    "type": parts[i + 1],
                    "count": int(parts[i + 2]),
                }
            )
        metadata["properties"] = props

    return metadata


@jaxtyped(typechecker=beartype)
def parse_xyz(file_path: Union[str, Path]) -> CrystalData:
    """Parse an XYZ file and return a validated CrystalData PyTree.

    Parameters
    ----------
    file_path : str or Path
        Path to the XYZ file.

    Returns
    -------
    crystal_data : CrystalData
        Validated JAX-compatible structure with all contents from the XYZ file.

    Raises
    ------
    ValueError
        If file format is invalid or contains inconsistent data.
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    Supports both atomic symbols (e.g., ``"H"``, ``"Fe"``)
    and atomic numbers (e.g., ``"1"``, ``"26"``) in the first
    column of atom data.
    """
    with open(file_path, encoding="utf-8") as f:
        lines: List[str] = f.readlines()
    too_small: int = 2
    if len(lines) < too_small:
        raise ValueError("Invalid XYZ file: fewer than 2 lines.")

    try:
        num_atoms: int = int(lines[0].strip())
    except ValueError as err:
        raise ValueError(
            "First line must be the number of atoms (int)."
        ) from err

    comment: str = lines[1].strip()
    metadata: Dict[str, Any] = _parse_xyz_metadata(comment)

    if len(lines) < 2 + num_atoms:
        raise ValueError(
            f"Expected {num_atoms} atoms, found only {len(lines) - 2}."
        )

    positions: List[List[float]] = []
    atomic_numbers: List[int] = []
    columns_normal: int = 4
    columns_extra: int = 5
    for ii in range(2, 2 + num_atoms):
        parts: List[str] = lines[ii].split()
        if len(parts) not in {4, 5, 6, 7}:
            raise ValueError(
                f"Line {ii + 1} has unexpected format: {lines[ii].strip()}"
            )

        if len(parts) == columns_normal:
            symbol: str
            x: str
            y: str
            z: str
            symbol, x, y, z = parts
        elif len(parts) == columns_extra:
            _: str
            symbol, x, y, z = parts
        else:
            symbol, x, y, z = parts[:4]

        positions.append([float(x), float(y), float(z)])

        # Handle both atomic symbols and atomic numbers
        try:
            # Try to parse as an integer (atomic number)
            atomic_num: int = int(symbol)
            atomic_numbers.append(atomic_num)
        except ValueError:
            # Not a number, treat as atomic symbol
            atomic_numbers.append(atomic_symbol(symbol))

    positions_arr: Float[Array, " N 3"] = jnp.array(
        positions, dtype=jnp.float64
    )
    atomic_z_arr: Int[Array, " N"] = jnp.array(atomic_numbers, dtype=jnp.int32)

    return make_crystal_data(
        positions=positions_arr,
        atomic_numbers=atomic_z_arr,
        lattice=metadata.get("lattice"),
        stress=metadata.get("stress"),
        energy=metadata.get("energy"),
        properties=metadata.get("properties"),
        comment=comment,
    )


_NUM_LATTICE_COMPONENTS: int = 3
_NUM_POSITION_COMPONENTS: int = 3


@jaxtyped(typechecker=beartype)
def parse_poscar(  # noqa: PLR0912, PLR0915
    file_path: Union[str, Path],
) -> CrystalData:
    """Parse a VASP POSCAR file and return a validated CrystalData PyTree.

    Supports VASP 5+ format with element symbols on line 6, as well as
    older VASP 4 format where element symbols must be inferred from
    the comment line.

    Parameters
    ----------
    file_path : str or Path
        Path to the POSCAR/CONTCAR file.

    Returns
    -------
    crystal_data : CrystalData
        Validated JAX-compatible structure containing:
        - positions : Float[Array, "N 3"]
            Cartesian coordinates in Angstroms
        - atomic_numbers : Int[Array, " N"]
            Atomic numbers for each atom
        - lattice : Float[Array, "3 3"]
            Lattice vectors in Angstroms
        - comment : str
            First line of the POSCAR file

    Raises
    ------
    ValueError
        If file format is invalid, element symbols are missing,
        or atom counts don't match positions.
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    POSCAR format (lines):
    1. Comment, 2. Scaling factor, 3--5. Lattice vectors,
    6. Element symbols (VASP 5+) or counts (VASP 4),
    7. Counts (if line 6 has symbols), 8. Optional
    ``Selective dynamics``, 9. Coordinate type, 10+. Positions.

    Implementation Logic
    --------------------
    1. **Read file** -- Load all lines.
    2. **Parse header** -- Comment and scaling factor.
    3. **Parse lattice** -- 3x3 vectors, apply scaling.
    4. **Detect VASP version** -- Letters on line 6 indicate
       VASP 5+ with explicit element symbols.
    5. **Parse elements and counts** -- Extract symbols and
       per-element atom counts.
    6. **Handle selective dynamics** -- Skip if present.
    7. **Parse coordinates** -- Direct (fractional) or
       Cartesian. Convert fractional to Cartesian via
       ``positions @ lattice``.
    8. **Build output** -- Construct atomic numbers array
       and return
       :class:`~ptyrodactyl.tools.CrystalData` PyTree.
    """
    with open(file_path, encoding="utf-8") as f:
        lines: List[str] = f.readlines()

    min_lines: int = 8
    if len(lines) < min_lines:
        raise ValueError(
            f"Invalid POSCAR: expected at least {min_lines} lines, "
            f"got {len(lines)}."
        )

    comment: str = lines[0].strip()

    try:
        scale: float = float(lines[1].strip())
    except ValueError as err:
        raise ValueError(
            "Line 2 must be the universal scaling factor (float)."
        ) from err

    lattice_rows: List[List[float]] = []
    for i in range(2, 5):
        parts: List[str] = lines[i].split()
        if len(parts) != _NUM_LATTICE_COMPONENTS:
            raise ValueError(
                f"Line {i + 1} must have {_NUM_LATTICE_COMPONENTS} lattice "
                f"vector components, got {len(parts)}."
            )
        lattice_rows.append([float(x) for x in parts])

    lattice: Float[Array, "3 3"] = (
        jnp.array(lattice_rows, dtype=jnp.float64) * scale
    )

    line_6: str = lines[5].strip()
    has_symbols: bool = any(c.isalpha() for c in line_6)

    if has_symbols:
        element_symbols: List[str] = line_6.split()
        counts_line: str = lines[6].strip()
        atom_counts: List[int] = [int(x) for x in counts_line.split()]
        next_line_idx: int = 7
    else:
        atom_counts = [int(x) for x in line_6.split()]
        element_symbols = _extract_elements_from_comment(comment)
        if len(element_symbols) != len(atom_counts):
            raise ValueError(
                "VASP 4 format detected but cannot determine element "
                "symbols. Use VASP 5+ format with element symbols on line 6."
            )
        next_line_idx = 6

    if len(element_symbols) != len(atom_counts):
        raise ValueError(
            f"Number of element symbols ({len(element_symbols)}) does not "
            f"match number of atom counts ({len(atom_counts)})."
        )

    coord_line: str = lines[next_line_idx].strip()
    if coord_line.lower().startswith("s"):
        next_line_idx += 1
        coord_line = lines[next_line_idx].strip()

    is_direct: bool = coord_line.lower().startswith("d")
    next_line_idx += 1

    total_atoms: int = sum(atom_counts)
    if len(lines) < next_line_idx + total_atoms:
        raise ValueError(
            f"Expected {total_atoms} atom positions, but file has only "
            f"{len(lines) - next_line_idx} remaining lines."
        )

    positions_list: List[List[float]] = []
    for i in range(next_line_idx, next_line_idx + total_atoms):
        parts = lines[i].split()
        if len(parts) < _NUM_POSITION_COMPONENTS:
            raise ValueError(
                f"Line {i + 1} must have at least "
                f"{_NUM_POSITION_COMPONENTS} position coordinates."
            )
        positions_list.append(
            [float(parts[0]), float(parts[1]), float(parts[2])]
        )

    positions_arr: Float[Array, "N 3"] = jnp.array(
        positions_list, dtype=jnp.float64
    )

    if is_direct:
        positions_arr = positions_arr @ lattice

    atomic_numbers_list: List[int] = []
    for symbol, count in zip(element_symbols, atom_counts, strict=True):
        atom_num: int = atomic_symbol(symbol)
        atomic_numbers_list.extend([atom_num] * count)

    atomic_z_arr: Int[Array, " N"] = jnp.array(
        atomic_numbers_list, dtype=jnp.int32
    )

    return make_crystal_data(
        positions=positions_arr,
        atomic_numbers=atomic_z_arr,
        lattice=lattice,
        comment=comment,
    )


@beartype
def _extract_elements_from_comment(comment: str) -> List[str]:
    """Extract element symbols from POSCAR comment line.

    Attempts to find element symbols in the comment line for VASP 4
    format files where symbols are not on a dedicated line.

    Parameters
    ----------
    comment : str
        The first line of the POSCAR file.

    Returns
    -------
    elements : List[str]
        List of element symbols found in the comment.

    Notes
    -----
    This is a heuristic approach that looks for capitalized words
    that match known element symbols.
    """
    words: List[str] = comment.split()
    elements: List[str] = []
    for word in words:
        cleaned: str = re.sub(r"[^A-Za-z]", "", word)
        if cleaned:
            normalized: str = cleaned.capitalize()
            if normalized in _ATOMIC_NUMBERS:
                elements.append(normalized)
    return elements


@jaxtyped(typechecker=beartype)
def parse_crystal(file_path: Union[str, Path]) -> CrystalData:
    """Parse XYZ or POSCAR file, auto-detecting format, returns CrystalData.

    Automatically detects whether the input file is an XYZ or POSCAR/CONTCAR
    file based on file extension and calls the appropriate parser.

    Parameters
    ----------
    file_path : str or Path
        Path to the crystal structure file (.xyz, POSCAR, or CONTCAR).

    Returns
    -------
    crystal_data : CrystalData
        Validated JAX-compatible structure with atomic positions and numbers.

    Raises
    ------
    ValueError
        If file format cannot be determined or is unsupported.
    FileNotFoundError
        If the specified file does not exist.

    Implementation Logic
    --------------------
    1. **Check extension** --
       ``.xyz`` dispatches to :func:`parse_xyz`.
    2. **Check filename** --
       Names containing ``POSCAR`` or ``CONTCAR`` dispatch
       to :func:`parse_poscar`.
    3. **Content heuristic** --
       If the first line parses as an integer, assume XYZ;
       otherwise fall back to POSCAR.

    Notes
    -----
    Supported formats: XYZ (``.xyz``), VASP POSCAR/CONTCAR.

    See Also
    --------
    :func:`parse_xyz` : Parser for XYZ format files.
    :func:`parse_poscar` : Parser for VASP POSCAR/CONTCAR
        files.
    """
    path: Path = Path(file_path)
    filename: str = path.name.lower()
    suffix: str = path.suffix.lower()

    if suffix == ".xyz":
        return parse_xyz(file_path)

    if "poscar" in filename or "contcar" in filename:
        return parse_poscar(file_path)

    with open(path, encoding="utf-8") as f:
        first_line: str = f.readline().strip()

    try:
        int(first_line)
        return parse_xyz(file_path)
    except ValueError:
        pass

    return parse_poscar(file_path)
