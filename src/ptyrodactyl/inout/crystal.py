"""Crystal-structure parser dispatcher.

Extended Summary
----------------
This module dispatches crystal structure files to the XYZ or POSCAR
parser based on filename, extension, and the first line of the file.

Routine Listings
----------------
:func:`parse_crystal`
    Parse XYZ or POSCAR file, auto-detecting format, returns CrystalData.
"""

from pathlib import Path

from beartype import beartype
from beartype.typing import Union
from jaxtyping import jaxtyped

from ptyrodactyl.types import CrystalData

from .poscar import parse_poscar
from .xyz import parse_xyz


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
    
    :see: parse_xyz, parse_poscar.
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


__all__: list[str] = [
    "parse_crystal",
]
