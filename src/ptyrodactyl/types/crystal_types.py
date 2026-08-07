"""Define crystal carriers and factories.

Extended Summary
----------------
This module defines canonical Equinox PyTree carriers for crystal
structure coordinates and parsed crystal data. Dynamic numeric fields
remain JAX leaves, while CrystalData metadata is stored as static
JSON-encodable Python data.

Routine Listings
----------------
:class:`CrystalData`
    Store parsed crystal data and static metadata.
:class:`CrystalStructure`
    Store fractional and Cartesian crystal coordinates.
:func:`create_crystal_data`
    Create a CrystalData with runtime validation.
:func:`create_crystal_structure`
    Create a CrystalStructure with runtime validation.

Notes
-----
The only static fields are ``CrystalData.properties`` and
``CrystalData.comment``. Optional array fields remain dynamic PyTree leaves
when present.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union
from jaxtyping import Array, Float, Int, Num, jaxtyped

from .custom_types import scalar_float


def _raise_if(condition: bool, message: str) -> None:
    """Raise ValueError when a structural condition is true."""
    if condition:
        raise ValueError(message)


class CrystalStructure(eqx.Module):
    """Store fractional and Cartesian crystal coordinates.

    :see: :mod:`~.test_crystal_types`

    Attributes
    ----------
    frac_positions : Float[Array, "* 4"]
        Fractional positions with atomic numbers in the fourth column.
    cart_positions : Num[Array, "* 4"]
        Cartesian positions with atomic numbers in the fourth column.
    cell_lengths : Num[Array, " 3"]
        Unit-cell lengths [a, b, c] in Angstroms.
    cell_angles : Num[Array, " 3"]
        Unit-cell angles [alpha, beta, gamma] in degrees.

    See Also
    --------
    :func:`create_crystal_structure`
        Create and validate a :class:`CrystalStructure`.
    """

    frac_positions: Float[Array, "* 4"]
    cart_positions: Num[Array, "* 4"]
    cell_lengths: Num[Array, " 3"]
    cell_angles: Num[Array, " 3"]


class CrystalData(eqx.Module):
    """Store parsed crystal data and static metadata.

    :see: :func:`~.test_create_crystal_data_jit_compiles_and_runs`

    Attributes
    ----------
    positions : Float[Array, " N 3"]
        Cartesian positions in Angstroms.
    atomic_numbers : Int[Array, " N"]
        Atomic numbers corresponding to each atom.
    lattice : Optional[Float[Array, "3 3"]]
        Lattice vectors in Angstroms, or None.
    stress : Optional[Float[Array, "3 3"]]
        Stress tensor, or None.
    energy : Optional[scalar_float]
        Total energy in eV, or None.
    properties : Optional[List[Dict[str, Union[str, int]]]]
        Static JSON-encodable per-atom metadata.
    comment : Optional[str]
        Static source comment string.

    See Also
    --------
    :func:`create_crystal_data`
        Create and validate a :class:`CrystalData`.
    """

    positions: Float[Array, " N 3"]
    atomic_numbers: Int[Array, " N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[scalar_float]
    properties: Optional[List[Dict[str, Union[str, int]]]] = eqx.field(
        default=None,
        static=True,
    )
    comment: Optional[str] = eqx.field(default=None, static=True)


@jaxtyped(typechecker=beartype)
def create_crystal_structure(
    frac_positions: Float[Array, "..."],
    cart_positions: Num[Array, "..."],
    cell_lengths: Num[Array, "..."],
    cell_angles: Num[Array, "..."],
) -> CrystalStructure:
    """Create a CrystalStructure with runtime validation.

    :see: :mod:`~.test_crystal_types`

    Parameters
    ----------
    frac_positions : Float[Array, "..."]
        Fractional coordinates and atomic numbers. Must have shape ``(N, 4)``.
    cart_positions : Num[Array, "..."]
        Cartesian coordinates and atomic numbers. Must have shape ``(N, 4)``.
    cell_lengths : Num[Array, "..."]
        Unit-cell lengths in Angstroms. Must have shape ``(3,)``.
    cell_angles : Num[Array, "..."]
        Unit-cell angles in degrees. Must have shape ``(3,)``.

    Returns
    -------
    crystal_structure : CrystalStructure
        Validated crystal structure.

    Raises
    ------
    ValueError
        If ranks or shapes are invalid.

    Notes
    -----
    1. Convert inputs to JAX arrays.
    2. Validate coordinate, length, and angle structures.
    3. Require matching atomic numbers, positive lengths, and angles in the
       open interval (0, 180) with traced error checks.
    4. Create and return a CrystalStructure.
    """
    frac_arr: Float[Array, "* 4"] = jnp.asarray(
        frac_positions,
        dtype=jnp.float64,
    )
    cart_arr: Num[Array, "* 4"] = jnp.asarray(cart_positions)
    lengths_arr: Num[Array, " 3"] = jnp.asarray(cell_lengths)
    angles_arr: Num[Array, " 3"] = jnp.asarray(cell_angles)

    max_cols: int = 4
    num_cell_params: int = 3
    coord_rank: int = 2
    _raise_if(frac_arr.ndim != coord_rank, "frac_positions must be 2D")
    _raise_if(cart_arr.ndim != coord_rank, "cart_positions must be 2D")
    _raise_if(
        frac_arr.shape[1] != max_cols,
        "frac_positions must have shape (N, 4)",
    )
    _raise_if(
        cart_arr.shape[1] != max_cols,
        "cart_positions must have shape (N, 4)",
    )
    _raise_if(
        lengths_arr.shape != (num_cell_params,),
        "cell_lengths must have shape (3,)",
    )
    _raise_if(
        angles_arr.shape != (num_cell_params,),
        "cell_angles must have shape (3,)",
    )
    _raise_if(
        frac_arr.shape[0] != cart_arr.shape[0],
        "frac_positions and cart_positions length differ",
    )

    checked_frac: Float[Array, "* 4"] = eqx.error_if(
        frac_arr,
        jnp.any(~jnp.isfinite(frac_arr)),
        "frac_positions contain non-finite values",
    )
    checked_cart: Num[Array, "* 4"] = eqx.error_if(
        cart_arr,
        jnp.any(~jnp.isfinite(cart_arr)),
        "cart_positions contain non-finite values",
    )
    checked_frac = eqx.error_if(
        checked_frac,
        jnp.any(checked_frac[:, 3] != checked_cart[:, 3]),
        "atomic numbers must match between frac and cart positions",
    )
    checked_lengths: Num[Array, " 3"] = eqx.error_if(
        lengths_arr,
        jnp.any(~jnp.isfinite(lengths_arr)),
        "cell_lengths contain non-finite values",
    )
    checked_lengths = eqx.error_if(
        checked_lengths,
        jnp.any(checked_lengths <= 0),
        "cell_lengths must be positive",
    )
    checked_angles: Num[Array, " 3"] = eqx.error_if(
        angles_arr,
        jnp.any(~jnp.isfinite(angles_arr)),
        "cell_angles contain non-finite values",
    )
    max_angle_degrees: int = 180
    checked_angles = eqx.error_if(
        checked_angles,
        jnp.any((checked_angles <= 0) | (checked_angles >= max_angle_degrees)),
        "cell_angles must be between 0 and 180 degrees",
    )
    crystal_structure: CrystalStructure = CrystalStructure(
        frac_positions=checked_frac,
        cart_positions=checked_cart,
        cell_lengths=checked_lengths,
        cell_angles=checked_angles,
    )
    return crystal_structure


@jaxtyped(typechecker=beartype)
def create_crystal_data(
    positions: Float[Array, "..."],
    atomic_numbers: Int[Array, "..."],
    lattice: Optional[Float[Array, "..."]] = None,
    stress: Optional[Float[Array, "..."]] = None,
    energy: Optional[scalar_float] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> CrystalData:
    """Create a CrystalData with runtime validation.

    :see: :func:`~.test_create_crystal_data_jit_compiles_and_runs`

    Parameters
    ----------
    positions : Float[Array, "..."]
        Cartesian positions in Angstroms. Must have shape ``(N, 3)``.
    atomic_numbers : Int[Array, "..."]
        Atomic numbers corresponding to each atom. Must have shape ``(N,)``.
    lattice : Optional[Float[Array, "..."]], optional
        Lattice vectors. Must have shape ``(3, 3)`` when provided. Defaults
        to the identity matrix.
    stress : Optional[Float[Array, "..."]], optional
        Stress tensor. Must have shape ``(3, 3)`` when provided.
    energy : Optional[scalar_float], optional
        Total energy in eV.
    properties : Optional[List[Dict[str, Union[str, int]]]], optional
        Static JSON-encodable per-atom metadata.
    comment : Optional[str], optional
        Static source comment string.

    Returns
    -------
    crystal_data : CrystalData
        Validated crystal data.

    Raises
    ------
    ValueError
        If ranks or shapes are invalid.

    Notes
    -----
    1. Convert numeric inputs to JAX arrays.
    2. Validate required and optional array structures.
    3. Require finite numeric arrays, non-negative atomic numbers, and finite
       energy with traced error checks.
    4. Create and return a CrystalData with static metadata.
    """
    positions_arr: Float[Array, " N 3"] = jnp.asarray(
        positions,
        dtype=jnp.float64,
    )
    atomic_numbers_arr: Int[Array, " N"] = jnp.asarray(
        atomic_numbers,
        dtype=jnp.int32,
    )
    lattice_arr: Optional[Float[Array, "3 3"]]
    if lattice is None:
        lattice_arr = jnp.eye(3, dtype=jnp.float64)
    else:
        lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)

    stress_arr: Optional[Float[Array, "3 3"]]
    if stress is None:
        stress_arr = None
    else:
        stress_arr = jnp.asarray(stress, dtype=jnp.float64)

    energy_arr: Optional[Float[Array, ""]]
    if energy is None:
        energy_arr = None
    else:
        energy_arr = jnp.asarray(energy, dtype=jnp.float64)

    max_position_cols: int = 3
    positions_rank: int = 2
    atom_rank: int = 1
    matrix_shape: tuple[int, int] = (3, 3)
    scalar_shape: tuple[()] = ()
    _raise_if(positions_arr.ndim != positions_rank, "positions must be 2D")
    _raise_if(
        positions_arr.shape[1] != max_position_cols,
        "positions must have shape (N, 3)",
    )
    _raise_if(
        atomic_numbers_arr.ndim != atom_rank,
        "atomic_numbers must be 1D",
    )
    _raise_if(
        atomic_numbers_arr.shape[0] != positions_arr.shape[0],
        "atomic_numbers must have shape (N,)",
    )
    _raise_if(
        lattice_arr is not None and lattice_arr.shape != matrix_shape,
        "lattice must have shape (3, 3)",
    )
    _raise_if(
        stress_arr is not None and stress_arr.shape != matrix_shape,
        "stress must have shape (3, 3)",
    )
    _raise_if(
        energy_arr is not None and energy_arr.shape != scalar_shape,
        "energy must be a scalar",
    )

    checked_positions: Float[Array, " N 3"] = eqx.error_if(
        positions_arr,
        jnp.any(~jnp.isfinite(positions_arr)),
        "positions contain non-finite values",
    )
    checked_atomic_numbers: Int[Array, " N"] = eqx.error_if(
        atomic_numbers_arr,
        jnp.any(atomic_numbers_arr < 0),
        "atomic_numbers must be non-negative",
    )
    checked_lattice: Optional[Float[Array, "3 3"]] = (
        None
        if lattice_arr is None
        else eqx.error_if(
            lattice_arr,
            jnp.any(~jnp.isfinite(lattice_arr)),
            "lattice contains non-finite values",
        )
    )
    checked_stress: Optional[Float[Array, "3 3"]] = (
        None
        if stress_arr is None
        else eqx.error_if(
            stress_arr,
            jnp.any(~jnp.isfinite(stress_arr)),
            "stress contains non-finite values",
        )
    )
    checked_energy: Optional[Float[Array, ""]] = (
        None
        if energy_arr is None
        else eqx.error_if(
            energy_arr,
            ~jnp.isfinite(energy_arr),
            "energy must be finite",
        )
    )
    crystal_data: CrystalData = CrystalData(
        positions=checked_positions,
        atomic_numbers=checked_atomic_numbers,
        lattice=checked_lattice,
        stress=checked_stress,
        energy=checked_energy,
        properties=properties,
        comment=comment,
    )
    return crystal_data


__all__: list[str] = [
    "CrystalData",
    "CrystalStructure",
    "create_crystal_data",
    "create_crystal_structure",
]
