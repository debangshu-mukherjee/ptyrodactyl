"""
Module: electrons.preprocessing
-------------------------------
Data preprocessing utilities for electron microscopy and ptychography.

This module contains utilities for preprocessing electron microscopy data
before analysis or reconstruction. Currently includes type definitions
for scalar numeric types used throughout the electrons module.

Functions
---------
- `atomic_symbol`:
    Returns atomic number for given atomic symbol string.
- `kirkland_potentials`:
    Loads Kirkland scattering factors from CSV file.
- `parse_xyz`:
    Parses an XYZ file and returns a list of atoms with their
    element symbols and 3D coordinates.

Internal Functions
------------------
These functions are not exported and are used internally by the module.

- `_load_atomic_numbers`:
    Loads atomic number mapping from JSON file in manifest folder.
- `_load_kirkland_potentials`:
    Loads Kirkland scattering factors from CSV file.
- `_parse_xyz_metadata`:
    Internal function to extract metadata from the XYZ comment line.
"""

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Optional, Union
from jaxtyping import Array, Float, Int, jaxtyped

from .electron_types import XYZData, make_xyz_data, scalar_int

_KIRKLAND_PATH: Path = (
    Path(__file__).resolve().parent / "luggage" / "Kirkland_Potentials.csv"
)
_ATOMS_PATH: Path = Path(__file__).resolve().parent / "luggage" / "atom_numbers.json"

jax.config.update("jax_enable_x64", True)


def _load_atomic_numbers(json_path: Optional[Path] = _ATOMS_PATH) -> Dict[str, int]:
    """
    Description
    -----------
    Loads atomic number mapping from JSON file in manifest folder.
    Uses pathlib for OS-independent path handling.

    Parameters
    ----------
    - `json_path` (Optional[Path]):
        Custom path to JSON file, defaults to module path

    Returns
    -------
    - `atomic_data` (Dict[str, int]):
        Dictionary mapping atomic symbols to atomic numbers

    Raises
    ------
    - FileNotFoundError:
        If JSON file is not found
    - json.JSONDecodeError:
        If JSON file is malformed
    """
    file_path: Path = json_path if json_path is not None else _ATOMS_PATH
    with open(file_path, "r", encoding="utf-8") as file:
        atomic_data: Dict[str, int] = json.load(file)
    return atomic_data


_ATOMIC_NUMBERS: Dict[str, int] = _load_atomic_numbers()


@jaxtyped(typechecker=beartype)
def atomic_symbol(atomic_symbol: str) -> scalar_int:
    """
    Description
    -----------
    Returns atomic number for given atomic symbol string.
    Uses preloaded atomic number mapping for fast lookup.

    Parameters
    ----------
    - `atomic_symbol` (str):
        Chemical symbol for the element (e.g., "H", "He", "Li")

    Returns
    -------
    - `atomic_number` (scalar_int):
        Atomic number corresponding to the symbol

    Raises
    ------
    - KeyError:
        If atomic symbol is not found in the mapping
    - TypeError:
        If input is not a string

    Flow
    ----
    - Validate input is string
    - Strip whitespace and ensure proper case
    - Look up atomic number in preloaded mapping
    - Return atomic number as scalar integer
    """
    cleaned_symbol: str = atomic_symbol.strip()

    if not cleaned_symbol:
        raise ValueError("Atomic symbol cannot be empty")

    normalized_symbol: str = cleaned_symbol.capitalize()
    if normalized_symbol not in _ATOMIC_NUMBERS:
        available_symbols: str = ", ".join(sorted(_ATOMIC_NUMBERS.keys()))
        raise KeyError(
            f"Atomic symbol '{atomic_symbol}' not found. "
            f"Available symbols: {available_symbols}"
        )

    atomic_number: scalar_int = _ATOMIC_NUMBERS[normalized_symbol]
    return atomic_number


def _load_kirkland_csv(
    file_path: Optional[Path] = _KIRKLAND_PATH,
) -> Float[Array, "103 12"]:
    """
    Description
    -----------
    Loads Kirkland potential parameters from CSV file.
    Uses numpy to load CSV then converts to JAX array for performance.

    Parameters
    ----------
    - `csv_path` (Optional[Path]):
        Custom path to CSV file, defaults to module path

    Returns
    -------
    - `kirkland_data` (Float[Array, "103 12"]):
        Kirkland potential parameters as JAX array

    Raises
    ------
    - FileNotFoundError:
        If CSV file is not found
    - ValueError:
        If CSV dimensions are incorrect
    """

    kirkland_numpy: np.ndarray = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
    if kirkland_numpy.shape != (103, 12):
        raise ValueError(f"Expected CSV shape (103, 12), got {kirkland_numpy.shape}")
    kirkland_data: Float[Array, "103 12"] = jnp.asarray(
        kirkland_numpy, dtype=jnp.float64
    )
    return kirkland_data


_KIRKLAND_POTENTIALS: Float[Array, "103 12"] = _load_kirkland_csv()


@jaxtyped(typechecker=beartype)
def kirkland_potentials() -> Float[Array, "103 12"]:
    """
    Description
    -----------
    Returns preloaded Kirkland potential parameters as JAX array.
    Data is loaded once at module import for optimal performance.

    Returns
    -------
    - `kirkland_potentials` (Float[Array, "103 12"]):
        Kirkland potential parameters for elements 1-103

    Flow
    ----
    - Return preloaded JAX array from module-level cache
    - No file I/O operations for fast access
    """
    return _KIRKLAND_POTENTIALS


def _parse_xyz_metadata(line: str) -> Dict[str, Any]:
    """
    Internal function to extract metadata from the XYZ comment line.

    Parameters
    ----------
    - `line` (str):
        Second line of the XYZ file (comment/metadata).

    Returns
    -------
    - `metadata` (dict):
        Parsed metadata with optional keys: lattice, stress, energy, properties.

    Raises
    ------
    - ValueError:
        If lattice or stress tensor dimensions are incorrect
    """
    metadata = {}

    lattice_match = re.search(r'Lattice="([^"]+)"', line)
    if lattice_match:
        values = list(map(float, lattice_match.group(1).split()))
        if len(values) != 9:
            raise ValueError("Lattice must contain 9 values")
        metadata["lattice"] = jnp.array(values, dtype=jnp.float64).reshape(3, 3)

    stress_match = re.search(r'stress="([^"]+)"', line)
    if stress_match:
        values = list(map(float, stress_match.group(1).split()))
        if len(values) != 9:
            raise ValueError("Stress tensor must contain 9 values")
        metadata["stress"] = jnp.array(values, dtype=jnp.float64).reshape(3, 3)

    energy_match = re.search(r"energy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
    if energy_match:
        metadata["energy"] = float(energy_match.group(1))

    props_match = re.search(r"Properties=([^ ]+)", line)
    if props_match:
        raw_props = props_match.group(1)
        parts = raw_props.split(":")
        props = []
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
def parse_xyz(file_path: Union[str, Path]) -> XYZData:
    """
    Parses an XYZ file and returns a validated XYZData PyTree.

    Parameters
    ----------
    - `file_path` (str or Path):
        Path to the XYZ file.

    Returns
    -------
    - `XYZData`:
        Validated JAX-compatible structure with all contents from the XYZ file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("Invalid XYZ file: fewer than 2 lines.")

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("First line must be the number of atoms (int).")

    comment = lines[1].strip()
    metadata = _parse_xyz_metadata(comment)

    if len(lines) < 2 + num_atoms:
        raise ValueError(f"Expected {num_atoms} atoms, found only {len(lines) - 2}.")

    positions = []
    atomic_numbers = []

    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) not in {4, 5, 6, 7}:
            raise ValueError(f"Line {i + 1} has unexpected format: {lines[i].strip()}")

        if len(parts) == 4:
            symbol, x, y, z = parts
        elif len(parts) == 5:
            _, symbol, x, y, z = parts
        else:
            symbol, x, y, z = parts[:4]

        positions.append([float(x), float(y), float(z)])
        atomic_numbers.append(atomic_symbol(symbol))

    positions_arr: Float[Array, "N 3"] = jnp.array(positions, dtype=jnp.float64)
    atomic_Z_arr: Int[Array, "N"] = jnp.array(atomic_numbers, dtype=jnp.int32)

    return make_xyz_data(
        positions=positions_arr,
        atomic_numbers=atomic_Z_arr,
        lattice=metadata.get("lattice"),
        stress=metadata.get("stress"),
        energy=metadata.get("energy"),
        properties=metadata.get("properties"),
        comment=comment,
    )
