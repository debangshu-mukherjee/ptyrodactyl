"""VASP POSCAR crystal-structure parsing utilities.

Extended Summary
----------------
This module parses VASP POSCAR and CONTCAR files into the
canonical :class:`~ptyrodactyl.types.CrystalData` carrier.

Routine Listings
----------------
:func:`parse_poscar`
    Parse a VASP POSCAR file and return a validated CrystalData PyTree.

"""

from pathlib import Path

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Union
from jaxtyping import Array, Float, Int, jaxtyped

from ptyrodactyl.types import CrystalData, create_crystal_data

from .xyz import _extract_elements_from_comment, atomic_symbol

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

    :see: :func:`~.test_parse_poscar_resolves_through_public_package`

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
       :class:`~ptyrodactyl.types.CrystalData` PyTree.

    :see: parse_crystal, parse_xyz, atomic_symbol.
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

    crystal_data: CrystalData = create_crystal_data(
        positions=positions_arr,
        atomic_numbers=atomic_z_arr,
        lattice=lattice,
        comment=comment,
    )
    return crystal_data


__all__: list[str] = [
    "parse_poscar",
]
