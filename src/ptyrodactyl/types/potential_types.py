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
    Band-limited scalar electrostatic potential on an orthogonal grid.
:func:`create_potential_3d`
    Create a Potential3D with structural and traced-value validation.

Notes
-----
The volume storage order is ``(z, y, x)``.  Geometry tuples use physical
``(x, y, z)`` order and lengths are measured in Angstroms.  Potential values
and the declared additive reference are measured in volts.
"""

import math
import re
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Num, jaxtyped

from .custom_types import scalar_num

_CUBE_RANK: int = 3
_SHA256_HEX_LENGTH: int = 64
_XYZ_SIZE: int = 3

type _StaticXYZ = Sequence[float] | Num[Array, " 3"]


class Potential3D(eqx.Module):
    """Store a band-limited scalar electrostatic potential in volts.

    Attributes
    ----------
    volume : Float[Array, "nz ny nx"]
        Sampled electrostatic potential in volts.  This is the carrier's only
        dynamic PyTree leaf.
    voxel_size : tuple[float, float, float]
        Voxel spacing ``(dx, dy, dz)`` in Angstroms.
    box_size : tuple[float, float, float]
        Periodic box lengths ``(Lx, Ly, Lz)`` in Angstroms.
    origin : tuple[float, float, float]
        Physical coordinate of sample ``volume[0, 0, 0]``, in Angstroms and
        ``(x, y, z)`` order.
    units : str
        Potential units.  The canonical value is exactly ``"V"``; energy
        units such as ``"eV"`` are rejected by the factory.
    reference_value : float
        Additive electrostatic reference value in volts.
    reference_semantics : str
        Physical meaning of the additive reference.  It must be explicit;
        an unspecified or silently mean-zeroed reference is invalid.
    boundary : str
        Boundary convention used to construct the field.
    producer : str
        Name and version of the potential producer.
    provenance_hash : str
        Lower-case SHA-256 digest identifying the producer coefficients.
    coefficient_normalization : str
        Declared continuous-transform and discrete-FFT normalization.
    band_limit : float
        Spherical potential band limit in cycles per Angstrom.
    """

    volume: Float[Array, "nz ny nx"]
    voxel_size: tuple[float, float, float] = eqx.field(static=True)
    box_size: tuple[float, float, float] = eqx.field(static=True)
    origin: tuple[float, float, float] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    reference_value: float = eqx.field(static=True)
    reference_semantics: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    provenance_hash: str = eqx.field(static=True)
    coefficient_normalization: str = eqx.field(static=True)
    band_limit: float = eqx.field(static=True)


def _xyz_tuple(values: _StaticXYZ, name: str) -> tuple[float, float, float]:
    """Convert and validate one static physical ``(x, y, z)`` tuple."""
    if isinstance(values, str | bytes) or len(values) != _XYZ_SIZE:
        raise ValueError(f"{name} must contain exactly three values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} values must be real numbers")
    values_array: Num[Array, " 3"] = jnp.asarray(values)
    if jnp.issubdtype(values_array.dtype, jnp.bool_):
        raise ValueError(f"{name} values must be real numbers")
    try:
        result: tuple[float, float, float] = tuple(
            float(value) for value in values
        )  # type: ignore[assignment]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be real numbers") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


def _nonempty_text(value: str, name: str) -> str:
    """Validate and normalize one required static text declaration."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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

    Parameters
    ----------
    volume : Num[Array, "..."]
        Real sampled voltage field with shape ``(nz, ny, nx)``.
    voxel_size : Sequence[float]
        Voxel spacing ``(dx, dy, dz)`` in Angstroms.
    box_size : Sequence[float]
        Box lengths ``(Lx, Ly, Lz)`` in Angstroms.  They must agree with the
        volume shape and voxel spacing.
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
    volume_arr: Float[Array, "nz ny nx"] = raw_volume.astype(jnp.float64)
    if volume_arr.ndim != _CUBE_RANK:
        raise ValueError("volume must have shape (nz, ny, nx)")
    if any(size <= 0 for size in volume_arr.shape):
        raise ValueError("volume dimensions must be positive")

    voxel_xyz: tuple[float, float, float] = _xyz_tuple(
        voxel_size,
        "voxel_size",
    )
    box_xyz: tuple[float, float, float] = _xyz_tuple(box_size, "box_size")
    origin_xyz: tuple[float, float, float] = _xyz_tuple(origin, "origin")
    if any(value <= 0.0 for value in voxel_xyz):
        raise ValueError("voxel_size values must be positive")
    if any(value <= 0.0 for value in box_xyz):
        raise ValueError("box_size values must be positive")

    nz: int
    ny: int
    nx: int
    nz, ny, nx = volume_arr.shape
    expected_box: tuple[float, float, float] = (
        nx * voxel_xyz[0],
        ny * voxel_xyz[1],
        nz * voxel_xyz[2],
    )
    if not all(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(box_xyz, expected_box, strict=True)
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
    ambiguous_reference_phrases: tuple[str, ...] = (
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
    common_nyquist: float = min(0.5 / spacing for spacing in voxel_xyz)
    if band_limit_float > common_nyquist * (1.0 + 1e-12):
        raise ValueError(
            "band_limit exceeds the common representable Nyquist frequency"
        )

    checked_volume: Float[Array, "nz ny nx"] = eqx.error_if(
        volume_arr,
        jnp.any(~jnp.isfinite(volume_arr)),
        "volume contains non-finite values",
    )
    potential: Potential3D = Potential3D(
        volume=checked_volume,
        voxel_size=voxel_xyz,
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
