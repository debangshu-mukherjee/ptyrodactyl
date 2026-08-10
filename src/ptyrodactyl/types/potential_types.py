"""Define the canonical scalar three-dimensional potential carrier.

Extended Summary
----------------
This module defines the voltage-field interchange carrier shared by the IAM
and later ab-initio potential producers.  The sampled volume is the only
dynamic PyTree leaf.  Geometry, physical-reference, boundary, provenance,
and Fourier-normalization declarations are static metadata so a compiled
forward model cannot silently reinterpret the field.

Routine Listings
----------------
:class:`Potential3D`
    Store a band-limited scalar electrostatic potential in volts.
:func:`create_potential_3d`
    Create a validated three-dimensional electrostatic potential.

Notes
-----
The volume storage order is ``(z, y, x)``.  Geometry tuples use physical
``(x, y, z)`` order and lengths are measured in Angstroms.  Potential values
and the declared additive reference are measured in volts.
"""

import math
import re
from collections.abc import Sequence
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float64, Num, jaxtyped

from .custom_types import scalar_num

_CUBE_RANK: int = 3
_SHA256_HEX_LENGTH: int = 64
_XYZ_SIZE: int = 3

type _StaticXYZ = Sequence[float] | Num[Array, " 3"]


class Potential3D(eqx.Module):
    """Store a band-limited scalar electrostatic potential in volts.

    :see: :mod:`~.test_potential_types`

    Attributes
    ----------
    volume : Float64[Array, "nz ny nx"]
        Sampled electrostatic potential in volts.  This is the carrier's only
        dynamic PyTree leaf.
    voxel_size : Tuple[float, float, float]
        Canonical binary64 diagnostic spacing ``(Lx / Nx, Ly / Ny,
        Lz / Nz)`` in Angstroms.  The exact finite-target grid is owned by
        ``box_size`` and the integer volume shape, not by multiplying these
        rounded values back by the shape.
    box_size : Tuple[float, float, float]
        Static periodic box lengths ``(Lx, Ly, Lz)`` in Angstroms.
    origin : Tuple[float, float, float]
        Static physical coordinate of sample ``volume[0, 0, 0]``, in
        Angstroms and ``(x, y, z)`` order.
    units : str
        Static potential units. The canonical value is exactly ``"V"``;
        energy units such as ``"eV"`` are rejected by the factory.
    reference_value : float
        Static additive electrostatic reference value in volts.
    reference_semantics : str
        Static physical meaning of the additive reference. It must be
        explicit; an unspecified or silently mean-zeroed reference is
        invalid.
    boundary : str
        Static boundary convention used to construct the field.
    producer : str
        Static name and version of the potential producer.
    provenance_hash : str
        Static lower-case SHA-256 digest identifying producer coefficients.
    coefficient_normalization : str
        Static continuous-transform and discrete-FFT normalization.
    band_limit : float
        Static spherical potential band limit in cycles per Angstrom.

    See Also
    --------
    :func:`create_potential_3d`
        Create and validate a :class:`Potential3D`.
    """

    volume: Float64[Array, "nz ny nx"]
    voxel_size: Tuple[float, float, float] = eqx.field(static=True)
    box_size: Tuple[float, float, float] = eqx.field(static=True)
    origin: Tuple[float, float, float] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    reference_value: float = eqx.field(static=True)
    reference_semantics: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    provenance_hash: str = eqx.field(static=True)
    coefficient_normalization: str = eqx.field(static=True)
    band_limit: float = eqx.field(static=True)


def _xyz_tuple(values: _StaticXYZ, name: str) -> Tuple[float, float, float]:
    """PRIVATE: Convert and validate one physical ``(x, y, z)`` tuple.

    Parameters
    ----------
    values : _StaticXYZ
        Three real physical-coordinate values in Angstroms.
    name : str
        Field name included in validation errors.

    Returns
    -------
    x : float
        Finite physical x value in Angstroms.
    y : float
        Finite physical y value in Angstroms.
    z : float
        Finite physical z value in Angstroms.

    Raises
    ------
    ValueError
        If ``values`` does not contain exactly three finite real numbers.
    """
    if isinstance(values, str | bytes) or len(values) != _XYZ_SIZE:
        raise ValueError(f"{name} must contain exactly three values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} values must be real numbers")
    values_array: Num[Array, " 3"] = jnp.asarray(values)
    if jnp.issubdtype(values_array.dtype, jnp.bool_):
        raise ValueError(f"{name} values must be real numbers")
    try:
        converted_values: Tuple[float, ...] = tuple(
            float(value) for value in values
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be real numbers") from error
    x: float = converted_values[0]
    y: float = converted_values[1]
    z: float = converted_values[2]
    result: Tuple[float, float, float] = (x, y, z)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


def _nonempty_text(value: str, name: str) -> str:
    """PRIVATE: Validate and normalize one required text declaration.

    Parameters
    ----------
    value : str
        Static text to strip and validate.
    name : str
        Field name included in the validation error.

    Returns
    -------
    result : str
        Stripped non-empty text.

    Raises
    ------
    ValueError
        If ``value`` is not a string or contains only whitespace.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result: str = value.strip()
    return result


@jaxtyped(typechecker=beartype)
def create_potential_3d(  # noqa: PLR0912, PLR0913, PLR0915
    volume: Num[Array, "..."],
    voxel_size: _StaticXYZ,
    box_size: _StaticXYZ,
    origin: _StaticXYZ,
    *,
    units: str = "V",
    reference_value: scalar_num = 0.0,
    reference_semantics: str = "isolated-neutral-atom vacuum zero at infinity",
    boundary: str = "periodic",
    producer: str,
    provenance_hash: str,
    coefficient_normalization: str,
    band_limit: scalar_num,
) -> Potential3D:
    """Create a validated three-dimensional electrostatic potential.

    :see: :mod:`~.test_potential_types`

    Parameters
    ----------
    volume : Num[Array, "..."]
        Real sampled voltage field with shape ``(nz, ny, nx)``.
    voxel_size : Sequence[float]
        Submitted voxel spacing ``(dx, dy, dz)`` in Angstroms.  It is checked
        against the grid spacing implied by ``box_size`` and the integer
        volume shape; the returned carrier stores that canonical binary64
        quotient as diagnostic metadata.
    box_size : Sequence[float]
        Authoritative exact-target box lengths ``(Lx, Ly, Lz)`` in
        Angstroms.  Together with the integer volume shape they define sample
        coordinates and the physical voxel metric.
    origin : Sequence[float]
        Coordinate of the first sample in ``(x, y, z)`` order, in Angstroms.
    units : str, optional
        Potential units.  Must be exactly ``"V"``.  Default: ``"V"``.
    reference_value : float, optional
        Declared additive reference value in volts.  Default: ``0.0``.
    reference_semantics : str, optional
        Physical reference declaration.  Default: the vacuum zero of the
        isolated-neutral-atom IAM.
    boundary : str, optional
        Boundary convention.  Default: ``"periodic"``.
    producer : str
        Producer name and version.
    provenance_hash : str
        SHA-256 coefficient/source digest, optionally prefixed by
        ``"sha256:"``.
    coefficient_normalization : str
        Continuous Fourier and discrete FFT normalization declaration.
    band_limit : float
        Spherical band limit in cycles per Angstrom.

    Returns
    -------
    potential : Potential3D
        Validated potential carrier with ``volume`` as its only dynamic leaf.

    Raises
    ------
    ValueError
        If geometry or metadata are missing, inconsistent, ambiguous, or use
        non-voltage units, or if the volume is not a finite real 3D array.

    Notes
    -----
    Structural and metadata checks run eagerly.  The data-dependent finite
    check is attached with :func:`equinox.error_if`, so it also executes under
    JAX transformations.
    """
    raw_volume: Num[Array, "..."] = jnp.asarray(volume)
    if jnp.issubdtype(raw_volume.dtype, jnp.complexfloating):
        raise ValueError("volume must be a real electrostatic potential")
    volume_arr: Float64[Array, "nz ny nx"] = raw_volume.astype(jnp.float64)
    if volume_arr.ndim != _CUBE_RANK:
        raise ValueError("volume must have shape (nz, ny, nx)")
    if any(size <= 0 for size in volume_arr.shape):
        raise ValueError("volume dimensions must be positive")

    voxel_xyz: Tuple[float, float, float] = _xyz_tuple(
        voxel_size,
        "voxel_size",
    )
    box_xyz: Tuple[float, float, float] = _xyz_tuple(box_size, "box_size")
    origin_xyz: Tuple[float, float, float] = _xyz_tuple(origin, "origin")
    if any(value <= 0.0 for value in voxel_xyz):
        raise ValueError("voxel_size values must be positive")
    if any(value <= 0.0 for value in box_xyz):
        raise ValueError("box_size values must be positive")

    nz: int
    ny: int
    nx: int
    nz, ny, nx = volume_arr.shape
    canonical_voxel_xyz: Tuple[float, float, float] = (
        box_xyz[0] / nx,
        box_xyz[1] / ny,
        box_xyz[2] / nz,
    )
    if not all(
        math.isfinite(value) and value > 0.0 for value in canonical_voxel_xyz
    ):
        raise ValueError(
            "box_size / (nx, ny, nz) must remain a positive finite "
            "binary64 diagnostic spacing"
        )
    if not all(
        math.isclose(submitted, canonical, rel_tol=1e-12, abs_tol=1e-12)
        for submitted, canonical in zip(
            voxel_xyz,
            canonical_voxel_xyz,
            strict=True,
        )
    ):
        raise ValueError(
            "box_size must equal voxel_size * (nx, ny, nz) in xyz order"
        )

    if units != "V":
        raise ValueError(
            "units must be exactly 'V'; potential energy units are invalid"
        )
    reference_array: Num[Array, ""] = jnp.asarray(reference_value)
    if isinstance(reference_value, bool) or jnp.issubdtype(
        reference_array.dtype,
        jnp.bool_,
    ):
        raise ValueError("reference_value must be a finite voltage")
    reference_float: float = float(reference_value)
    if not math.isfinite(reference_float):
        raise ValueError("reference_value must be a finite voltage")
    reference_text: str = _nonempty_text(
        reference_semantics,
        "reference_semantics",
    )
    normalized_reference: str = re.sub(
        r"[^a-z0-9]+",
        " ",
        reference_text.casefold(),
    ).strip()
    ambiguous_reference_tokens: set[str] = {
        "none",
        "tbd",
        "unknown",
        "unspecified",
    }
    ambiguous_reference_phrases: Tuple[str, ...] = (
        "not declared",
        "not specified",
        "not stated",
        "to be determined",
    )
    reference_tokens: set[str] = set(normalized_reference.split())
    if (
        normalized_reference == "n a"
        or reference_tokens & ambiguous_reference_tokens
        or any(
            phrase in normalized_reference
            for phrase in ambiguous_reference_phrases
        )
    ):
        raise ValueError("reference_semantics must state a physical reference")

    boundary_text: str = _nonempty_text(boundary, "boundary")
    producer_text: str = _nonempty_text(producer, "producer")
    normalization_text: str = _nonempty_text(
        coefficient_normalization,
        "coefficient_normalization",
    )
    provenance_text: str = _nonempty_text(
        provenance_hash,
        "provenance_hash",
    ).lower()
    if provenance_text.startswith("sha256:"):
        provenance_text = provenance_text.removeprefix("sha256:")
    if len(provenance_text) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in provenance_text
    ):
        raise ValueError(
            "provenance_hash must be a SHA-256 hexadecimal digest"
        )

    band_limit_array: Num[Array, ""] = jnp.asarray(band_limit)
    if isinstance(band_limit, bool) or jnp.issubdtype(
        band_limit_array.dtype,
        jnp.bool_,
    ):
        raise ValueError("band_limit must be positive and finite")
    band_limit_float: float = float(band_limit)
    if not math.isfinite(band_limit_float) or band_limit_float <= 0.0:
        raise ValueError("band_limit must be positive and finite")
    common_nyquist: Fraction = min(
        Fraction(count, 1) / (2 * Fraction.from_float(length))
        for count, length in zip(
            (nx, ny, nz),
            box_xyz,
            strict=True,
        )
    )
    if Fraction.from_float(band_limit_float) > common_nyquist:
        raise ValueError(
            "band_limit exceeds the common representable Nyquist frequency"
        )

    checked_volume: Float64[Array, "nz ny nx"] = eqx.error_if(
        volume_arr,
        jnp.any(~jnp.isfinite(volume_arr)),
        "volume contains non-finite values",
    )
    potential: Potential3D = Potential3D(
        volume=checked_volume,
        voxel_size=canonical_voxel_xyz,
        box_size=box_xyz,
        origin=origin_xyz,
        units=units,
        reference_value=reference_float,
        reference_semantics=reference_text,
        boundary=boundary_text,
        producer=producer_text,
        provenance_hash=provenance_text,
        coefficient_normalization=normalization_text,
        band_limit=band_limit_float,
    )
    return potential


__all__: list[str] = ["Potential3D", "create_potential_3d"]
