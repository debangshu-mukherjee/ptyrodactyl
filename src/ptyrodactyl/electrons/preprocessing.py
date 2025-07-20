"""
Module: electrons.preprocessing
-------------------------------
Data preprocessing utilities for electron microscopy and ptychography.

This module contains utilities for preprocessing electron microscopy data
before analysis or reconstruction. Currently includes type definitions
for scalar numeric types used throughout the electrons module.
"""

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, TypeAlias, Union
from jax import lax
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Int,
    Num,
    PRNGKeyArray,
    jaxtyped,
)

jax.config.update("jax_enable_x64", True)

def parse_xyz(file_path):
    """
    Parses an XYZ file and returns a list of atoms with their element symbols and 3D coordinates.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        atoms (list of dict): List of atoms, each as a dictionary with keys 'element', 'x', 'y', 'z'.
        comment (str): The comment line in the XYZ file.
    """
    atoms = []
    periodic_table = {"C": 5, "Bi": 82, "S": 15, "Mo": 41, "Se": 33, "H": 0}
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("Invalid XYZ file: fewer than 2 lines.")

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("First line must contain the number of atoms.")

    comment = lines[1].strip()

    if len(lines) < 2 + num_atoms:
        raise ValueError(
            f"Expected {num_atoms} atoms, but file has only {len(lines)-2} atom lines."
        )

    atoms = np.zeros((num_atoms, 4))
    metadata = parse_xyz_metadata(lines[1])
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) != 4 and len(parts) != 5 and len(parts) != 7 and len(parts) != 6:
            raise ValueError(
                f"Line {i+1} does not have correct number of entries: {lines[i]}"
            )
        if len(parts) == 4:
            element, x, y, z = parts
        elif len(parts) == 7:
            element, x, y, z = parts[:4]
        elif len(parts) == 5:
            _, element, x, y, z = parts
        elif len(parts) == 6:
            element, x, y, z = parts[:4]
        atoms[i - 2] = [periodic_table[element], float(x), float(y), float(z)]

    return atoms, metadata, comment


def parse_xyz_metadata(line):
    metadata = {}

    # Extract lattice
    lattice_match = re.search(r'Lattice="([^"]+)"', line)
    if lattice_match:
        lattice_values = list(map(float, lattice_match.group(1).split()))
        if len(lattice_values) != 9:
            raise ValueError("Lattice must contain 9 values")
        metadata["lattice"] = np.array(lattice_values).reshape((3, 3))

    # Extract stress
    stress_match = re.search(r'stress="([^"]+)"', line)
    if stress_match:
        stress_values = list(map(float, stress_match.group(1).split()))
        if len(stress_values) != 9:
            raise ValueError("Stress tensor must contain 9 values")
        metadata["stress"] = np.array(stress_values).reshape((3, 3))

    # Extract energy
    energy_match = re.search(r"energy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
    if energy_match:
        metadata["energy"] = float(energy_match.group(1))

    # Extract properties
    props_match = re.search(r"Properties=([^ ]+)", line)
    if props_match:
        raw_props = props_match.group(1)
        # Format: name:type:count
        prop_fields = raw_props.split(":")
        props = []
        for i in range(0, len(prop_fields), 3):
            props.append(
                {
                    "name": prop_fields[i],
                    "type": prop_fields[i + 1],
                    "count": int(prop_fields[i + 2]),
                }
            )
        metadata["properties"] = props

    return metadata
