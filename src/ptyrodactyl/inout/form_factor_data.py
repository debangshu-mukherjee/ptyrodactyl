"""Load bundled atomic form-factor coefficient tables.

Extended Summary
----------------
This module is the single host-world home for the Lobato--Van Dyck and
Kirkland neutral-atom coefficient tables. Both assets cover H through Lr
(Z = 1--103), are validated as they are loaded, and are converted once to
float64 JAX arrays at module import.

Routine Listings
----------------
:func:`kirkland_potentials`
    Return preloaded Kirkland potential parameters.
:func:`lobato_potentials`
    Return preloaded Lobato--Van Dyck potential parameters.

References
----------
Lobato, I. and Van Dyck, D. (2014), *Acta Crystallographica A* 70,
636--649, DOI: 10.1107/S205327331401643X.

Kirkland, E. J., *Advanced Computing in Electron Microscopy*.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped
from numpy.typing import NDArray

_ELEMENT_COUNT: int = 103
_KIRKLAND_COEFFICIENT_COUNT: int = 12
_LOBATO_COEFFICIENT_COUNT: int = 10
_LUGGAGE_DIR: Path = Path(__file__).resolve().parent / "luggage"
_KIRKLAND_PATH: Path = _LUGGAGE_DIR / "Kirkland_Potentials.csv"
_LOBATO_PATH: Path = _LUGGAGE_DIR / "Lobato_van_Dyck.csv"


def _validate_coefficients(
    coefficients: Float[NDArray, "rows columns"],
    *,
    expected_shape: tuple[int, int],
    scale_columns: slice,
    table_name: str,
) -> None:
    """Validate one host-side coefficient matrix before JAX conversion."""
    if coefficients.shape != expected_shape:
        raise ValueError(
            f"Expected {table_name} CSV shape {expected_shape}, "
            f"got {coefficients.shape}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{table_name} CSV contains non-finite values")
    if np.any(coefficients[:, scale_columns] <= 0):
        raise ValueError(f"{table_name} scale coefficients must be positive")


@jaxtyped(typechecker=beartype)
def _load_kirkland_csv(
    file_path: Path | None = None,
) -> Float[Array, "103 12"]:
    """Load and validate the Kirkland coefficient table.

    Parameters
    ----------
    file_path : Path | None, optional
        Alternate CSV path. The bundled table is used when omitted.

    Returns
    -------
    kirkland_data : Float[Array, "103 12"]
        Float64 coefficient matrix in interleaved
        ``a1, b1, ..., c3, d3`` order.

    Raises
    ------
    ValueError
        If the table has the wrong shape or invalid coefficients.
    """
    resolved_path: Path = _KIRKLAND_PATH if file_path is None else file_path
    kirkland_numpy: Float[NDArray, "103 12"] = np.loadtxt(
        resolved_path,
        delimiter=",",
        dtype=np.float64,
    )
    _validate_coefficients(
        kirkland_numpy,
        expected_shape=(_ELEMENT_COUNT, _KIRKLAND_COEFFICIENT_COUNT),
        scale_columns=slice(1, None, 2),
        table_name="Kirkland",
    )
    kirkland_data: Float[Array, "103 12"] = jnp.asarray(
        kirkland_numpy,
        dtype=jnp.float64,
    )
    return kirkland_data


@jaxtyped(typechecker=beartype)
def _load_lobato_csv(
    file_path: Path | None = None,
) -> Float[Array, "103 10"]:
    """Load and validate the Lobato--Van Dyck coefficient table.

    Parameters
    ----------
    file_path : Path | None, optional
        Alternate CSV path. The bundled table is used when omitted.

    Returns
    -------
    lobato_data : Float[Array, "103 10"]
        Float64 coefficient matrix interleaved as
        ``a1, b1, ..., a5, b5``.

    Raises
    ------
    ValueError
        If element numbering, shape, or coefficients are invalid.

    Notes
    -----
    The source asset groups its five amplitudes before its five scales. This
    loader interleaves each pair once so all downstream consumers use the
    canonical row representation.
    """
    resolved_path: Path = _LOBATO_PATH if file_path is None else file_path
    atomic_numbers: Int[NDArray, "103"] = np.loadtxt(
        resolved_path,
        delimiter=",",
        dtype=np.int64,
        skiprows=1,
        usecols=(0,),
    )
    grouped: Float[NDArray, "103 10"] = np.loadtxt(
        resolved_path,
        delimiter=",",
        dtype=np.float64,
        skiprows=1,
        usecols=range(2, 12),
    )
    expected_atomic_numbers: Int[NDArray, "103"] = np.arange(
        1,
        _ELEMENT_COUNT + 1,
        dtype=np.int64,
    )
    if not np.array_equal(atomic_numbers, expected_atomic_numbers):
        raise ValueError(
            "Lobato CSV must contain atomic numbers 1 through 103"
        )
    _validate_coefficients(
        grouped,
        expected_shape=(_ELEMENT_COUNT, _LOBATO_COEFFICIENT_COUNT),
        scale_columns=slice(5, None),
        table_name="Lobato",
    )

    interleaved: Float[NDArray, "103 10"] = np.empty(
        (_ELEMENT_COUNT, _LOBATO_COEFFICIENT_COUNT),
        dtype=np.float64,
    )
    interleaved[:, 0::2] = grouped[:, :5]
    interleaved[:, 1::2] = grouped[:, 5:]
    lobato_data: Float[Array, "103 10"] = jnp.asarray(
        interleaved,
        dtype=jnp.float64,
    )
    return lobato_data


_KIRKLAND_POTENTIALS: Float[Array, "103 12"] = _load_kirkland_csv()
_LOBATO_POTENTIALS: Float[Array, "103 10"] = _load_lobato_csv()


@jaxtyped(typechecker=beartype)
def kirkland_potentials() -> Float[Array, "103 12"]:
    """Return preloaded Kirkland potential parameters.

    :see: :mod:`~.test_form_factor_data`

    Returns
    -------
    kirkland_data : Float[Array, "103 12"]
        Parameters for elements 1--103 in interleaved pair order.

    Notes
    -----
    The returned object is the module-level array loaded at import; repeated
    calls perform no file I/O or copy.
    """
    kirkland_data: Float[Array, "103 12"] = _KIRKLAND_POTENTIALS
    return kirkland_data


@jaxtyped(typechecker=beartype)
def lobato_potentials() -> Float[Array, "103 10"]:
    """Return preloaded Lobato--Van Dyck potential parameters.

    :see: :mod:`~.test_form_factor_data`

    Returns
    -------
    lobato_data : Float[Array, "103 10"]
        Parameters for elements 1--103 interleaved as
        ``a1, b1, ..., a5, b5``.

    Notes
    -----
    The returned object is the module-level array loaded at import; repeated
    calls perform no file I/O or copy.
    """
    lobato_data: Float[Array, "103 10"] = _LOBATO_POTENTIALS
    return lobato_data


__all__: list[str] = [
    "kirkland_potentials",
    "lobato_potentials",
]
