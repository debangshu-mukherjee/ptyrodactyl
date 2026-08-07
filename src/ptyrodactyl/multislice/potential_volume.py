"""Build band-limited three-dimensional independent-atom potentials.

Extended Summary
----------------
This module constructs a true volumetric electrostatic field from analytic
atomic form factors.  Atoms are translated by reciprocal-space phase ramps,
so their positions remain continuous differentiable parameters; no rounded
voxel deposition or projection is used.  The inverse DFT is normalized to the
continuous Fourier convention used by the SC-1 scalar contract.

Routine Listings
----------------
:func:`crystal_potential_volume`
    Superimpose atomic voltage fields and return a Potential3D carrier.
:func:`single_atom_potential_3d`
    Build one band-limited atomic voltage field on an orthogonal grid.

Notes
-----
Array storage and ``grid_shape`` use ``(z, y, x)`` order.  Physical vectors,
``voxel_size``, ``origin``, and atom positions use ``(x, y, z)`` order.  The
reciprocal coordinates are cycles per Angstrom, while the form-factor public
API accepts angular reciprocal coordinates and therefore receives
``2 * pi * |g|``.
"""

import math
import operator
from collections.abc import Sequence
from numbers import Real

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, Num, jaxtyped

from ..types import (
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    CrystalData,
    CrystalStructure,
    Potential3D,
    create_potential_3d,
    scalar_int,
    scalar_num,
)
from .form_factors import atomic_form_factor

_COEFFICIENT_NORMALIZATION: str = (
    "continuous phi_tilde(g)=C_MB*f(g); periodic c_g=phi_tilde/box_volume; "
    "JAX ifftn scaled by 1/voxel_volume; g in cycles/angstrom; "
    "strict spherical cutoff"
)
_KIRKLAND_PROVENANCE_SHA256: str = (
    "79d0dd198a688ee5c3009db7deb9963abbbea2974734193e2ceb1ea07c16b226"
)
_LOBATO_PROVENANCE_SHA256: str = (
    "84fb662f029dda26488c8f85a9bdecf2ffa9adafad4c3fd7a16f3112110938c0"
)
_MAX_ATOMIC_NUMBER: int = 103
_MATRIX_RANK: int = 2
_MIN_ATOMIC_NUMBER: int = 1
_POSITION_COLUMNS: int = 4
_XYZ_SIZE: int = 3

type _CrystalInput = CrystalData | CrystalStructure
type _GridShapeInput = Sequence[int] | Int[Array, " 3"]
type _StaticXYZ = Sequence[float] | Num[Array, " 3"]
type _VoxelSizeInput = scalar_num | _StaticXYZ


def _grid_shape_tuple(grid_shape: _GridShapeInput) -> tuple[int, int, int]:
    """Validate a static ``(nz, ny, nx)`` grid shape."""
    if isinstance(grid_shape, str | bytes) or len(grid_shape) != _XYZ_SIZE:
        raise ValueError("grid_shape must contain exactly (nz, ny, nx)")
    shape_values: list[int] = []
    for value in grid_shape:
        if isinstance(value, bool):
            raise ValueError("grid_shape values must be positive integers")
        try:
            integer_value: int = operator.index(value)
        except TypeError as error:
            raise ValueError(
                "grid_shape values must be positive integers"
            ) from error
        if integer_value <= 0:
            raise ValueError("grid_shape values must be positive integers")
        shape_values.append(integer_value)
    nz: int
    ny: int
    nx: int
    nz, ny, nx = shape_values
    return (nz, ny, nx)


def _voxel_size_tuple(
    voxel_size: _VoxelSizeInput,
) -> tuple[float, float, float]:
    """Normalize scalar or xyz voxel spacing to a positive static tuple."""
    if isinstance(voxel_size, bool) or (
        isinstance(voxel_size, Sequence)
        and any(isinstance(value, bool) for value in voxel_size)
    ):
        raise ValueError("voxel_size values must not be booleans")
    if isinstance(voxel_size, Real) and not isinstance(voxel_size, bool):
        scalar_spacing: float = float(voxel_size)
        values: tuple[float, float, float] = (
            scalar_spacing,
            scalar_spacing,
            scalar_spacing,
        )
    else:
        voxel_array: Num[Array, "..."] = jnp.asarray(voxel_size)
        if jnp.issubdtype(voxel_array.dtype, jnp.bool_):
            raise ValueError("voxel_size values must not be booleans")
        if voxel_array.shape == ():
            scalar_spacing = float(voxel_array)
            values = (
                scalar_spacing,
                scalar_spacing,
                scalar_spacing,
            )
            if not math.isfinite(scalar_spacing) or scalar_spacing <= 0.0:
                raise ValueError(
                    "voxel_size values must be positive and finite"
                )
            return values
        if voxel_array.shape != (_XYZ_SIZE,):
            raise ValueError(
                "voxel_size must be a scalar or contain exactly (dx, dy, dz)"
            )
        try:
            voxel_values: tuple[float, ...] = tuple(
                float(value) for value in voxel_array
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "voxel_size values must be real numbers"
            ) from error
        values = (voxel_values[0], voxel_values[1], voxel_values[2])
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("voxel_size values must be positive and finite")
    return values


def _origin_tuple(origin: _StaticXYZ) -> tuple[float, float, float]:
    """Validate a static physical ``(x, y, z)`` origin."""
    if isinstance(origin, str | bytes) or len(origin) != _XYZ_SIZE:
        raise ValueError("origin must contain exactly (x, y, z)")
    if any(isinstance(value, bool) for value in origin):
        raise ValueError("origin values must not be booleans")
    origin_array: Num[Array, " 3"] = jnp.asarray(origin)
    if jnp.issubdtype(origin_array.dtype, jnp.bool_):
        raise ValueError("origin values must not be booleans")
    try:
        result: tuple[float, float, float] = tuple(
            float(value) for value in origin
        )  # type: ignore[assignment]
    except (TypeError, ValueError) as error:
        raise ValueError("origin values must be real numbers") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError("origin values must be finite")
    return result


def _band_limit_value(
    band_limit: scalar_num | None,
    voxel_size: tuple[float, float, float],
) -> float:
    """Validate or choose the common spherical Nyquist band limit."""
    common_nyquist: float = min(0.5 / spacing for spacing in voxel_size)
    if band_limit is None:
        return common_nyquist
    band_limit_array: Num[Array, ""] = jnp.asarray(band_limit)
    if isinstance(band_limit, bool) or jnp.issubdtype(
        band_limit_array.dtype,
        jnp.bool_,
    ):
        raise ValueError("band_limit must be positive and finite")
    value: float = float(band_limit)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("band_limit must be positive and finite")
    if value > common_nyquist * (1.0 + 1e-12):
        raise ValueError(
            "band_limit exceeds the common representable Nyquist frequency"
        )
    return value


def _checked_atomic_numbers(
    atomic_numbers: Array,
) -> Int[Array, " N"]:
    """Reject non-integral or out-of-table atomic numbers under tracing."""
    raw_numbers: Array = jnp.asarray(atomic_numbers)
    if raw_numbers.ndim != 1:
        raise ValueError("atomic_numbers must have shape (N,)")
    if jnp.issubdtype(raw_numbers.dtype, jnp.bool_):
        raise ValueError("atomic_numbers must be integers, not booleans")
    if jnp.issubdtype(raw_numbers.dtype, jnp.complexfloating):
        raise ValueError("atomic_numbers must be real integers")
    checked_numbers: Num[Array, " N"] = eqx.error_if(
        raw_numbers,
        jnp.any(~jnp.isfinite(raw_numbers)),
        "atomic_numbers contain non-finite values",
    )
    checked_numbers = eqx.error_if(
        checked_numbers,
        jnp.any(checked_numbers != jnp.floor(checked_numbers)),
        "atomic_numbers must be integers",
    )
    checked_numbers = eqx.error_if(
        checked_numbers,
        jnp.any(
            (checked_numbers < _MIN_ATOMIC_NUMBER)
            | (checked_numbers > _MAX_ATOMIC_NUMBER)
        ),
        "atomic_numbers must be in the inclusive range [1, 103]",
    )
    return checked_numbers.astype(jnp.int32)


def _checked_positions(
    positions: Num[Array, "N 3"],
) -> Float[Array, "N 3"]:
    """Coerce xyz atom positions and attach a traced finite-value check."""
    raw_positions: Num[Array, "..."] = jnp.asarray(positions)
    if jnp.issubdtype(raw_positions.dtype, jnp.complexfloating):
        raise ValueError("positions must be real Cartesian coordinates")
    positions_arr: Float[Array, "N 3"] = raw_positions.astype(jnp.float64)
    if (
        positions_arr.ndim != _MATRIX_RANK
        or positions_arr.shape[1] != _XYZ_SIZE
    ):
        raise ValueError("positions must have shape (N, 3)")
    checked: Float[Array, "N 3"] = eqx.error_if(
        positions_arr,
        jnp.any(~jnp.isfinite(positions_arr)),
        "positions contain non-finite values",
    )
    return checked


def _checked_single_atomic_number(atom_no: scalar_int) -> Int[Array, ""]:
    """Validate one scalar atomic number without clipping traced values."""
    raw_number: Num[Array, ""] = jnp.asarray(atom_no)
    if raw_number.shape != ():
        raise ValueError("atom_no must be a scalar")
    checked: Int[Array, " 1"] = _checked_atomic_numbers(raw_number[None])
    return checked[0]


def _reciprocal_grid(
    voxel_size: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
) -> tuple[
    Float[Array, "nz ny nx"],
    Float[Array, "nz ny nx"],
    Float[Array, "nz ny nx"],
    Float[Array, "nz ny nx"],
]:
    """Return Cartesian DFT frequencies and their magnitude in cycles/Å."""
    dx: float
    dy: float
    dz: float
    dx, dy, dz = voxel_size
    nz: int
    ny: int
    nx: int
    nz, ny, nx = grid_shape
    gx_1d: Float[Array, " nx"] = jnp.fft.fftfreq(nx, d=dx)
    gy_1d: Float[Array, " ny"] = jnp.fft.fftfreq(ny, d=dy)
    gz_1d: Float[Array, " nz"] = jnp.fft.fftfreq(nz, d=dz)
    gz: Float[Array, "nz ny nx"]
    gy: Float[Array, "nz ny nx"]
    gx: Float[Array, "nz ny nx"]
    gz, gy, gx = jnp.meshgrid(gz_1d, gy_1d, gx_1d, indexing="ij")
    magnitude: Float[Array, "nz ny nx"] = jnp.sqrt(gx * gx + gy * gy + gz * gz)
    return gx, gy, gz, magnitude


def _potential_from_atoms(
    atomic_numbers: Array,
    positions: Num[Array, "N 3"],
    voxel_size: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    band_limit: float,
    parameterization: str,
) -> Float[Array, "nz ny nx"]:
    """Evaluate the periodic Fourier series for an atomic superposition."""
    checked_numbers: Int[Array, " N"] = _checked_atomic_numbers(atomic_numbers)
    checked_positions: Float[Array, "N 3"] = _checked_positions(positions)
    if checked_numbers.shape[0] != checked_positions.shape[0]:
        raise ValueError("atomic_numbers and positions must have equal length")

    gx: Float[Array, "nz ny nx"]
    gy: Float[Array, "nz ny nx"]
    gz: Float[Array, "nz ny nx"]
    g_magnitude: Float[Array, "nz ny nx"]
    gx, gy, gz, g_magnitude = _reciprocal_grid(voxel_size, grid_shape)
    spectral_mask: Float[Array, "nz ny nx"] = (
        g_magnitude < band_limit
    ).astype(jnp.float64)
    angular_magnitude: Float[Array, "nz ny nx"] = 2.0 * jnp.pi * g_magnitude
    origin_arr: Float[Array, " 3"] = jnp.asarray(origin, dtype=jnp.float64)

    initial_coefficients: jax.Array = jnp.zeros(
        grid_shape,
        dtype=jnp.complex128,
    )

    def _add_atom(
        coefficients: jax.Array,
        atom: tuple[Int[Array, ""], Float[Array, " 3"]],
    ) -> tuple[jax.Array, None]:
        """Accumulate one continuously translated atomic Fourier field."""
        atomic_number: Int[Array, ""]
        position: Float[Array, " 3"]
        atomic_number, position = atom
        form_factor: Float[Array, "nz ny nx"] = atomic_form_factor(
            atomic_number,
            angular_magnitude,
            parameterization=parameterization,
        )
        relative_position: Float[Array, " 3"] = position - origin_arr
        phase_argument: Float[Array, "nz ny nx"] = (
            gx * relative_position[0]
            + gy * relative_position[1]
            + gz * relative_position[2]
        )
        phase: jax.Array = jnp.exp(-2.0j * jnp.pi * phase_argument)
        atom_coefficients: jax.Array = (
            MOTT_BETHE_VOLT_ANGSTROM_SQ * form_factor * spectral_mask * phase
        )
        return coefficients + atom_coefficients, None

    fourier_amplitudes: jax.Array
    fourier_amplitudes, _ = jax.lax.scan(
        _add_atom,
        initial_coefficients,
        (checked_numbers, checked_positions),
    )
    voxel_volume: float = voxel_size[0] * voxel_size[1] * voxel_size[2]
    volume: Float[Array, "nz ny nx"] = jnp.real(
        jnp.fft.ifftn(fourier_amplitudes)
    ) / jnp.asarray(voxel_volume, dtype=jnp.float64)
    return volume


@jaxtyped(typechecker=beartype)
def single_atom_potential_3d(
    atom_no: scalar_int,
    voxel_size: _VoxelSizeInput,
    grid_shape: _GridShapeInput,
    center_coords: Float[Array, " 3"] | None = None,
    *,
    origin: _StaticXYZ = (0.0, 0.0, 0.0),
    band_limit: scalar_num | None = None,
    parameterization: str = "lobato",
) -> Float[Array, "nz ny nx"]:
    """Build one band-limited three-dimensional atomic potential.

    Parameters
    ----------
    atom_no : scalar_int
        Atomic number in the inclusive range 1--103.
    voxel_size : Real | Sequence[float]
        Scalar isotropic spacing or ``(dx, dy, dz)`` spacing in Angstroms.
    grid_shape : Sequence[int]
        Static volume shape ``(nz, ny, nx)``.
    center_coords : Float[Array, " 3"] | None, optional
        Continuous atom position ``(x, y, z)`` in Angstroms.  ``None`` places
        the atom at the geometric center of the periodic box.
    origin : Sequence[float], optional
        Static coordinate of sample ``[0, 0, 0]``.  Default: ``(0, 0, 0)``.
    band_limit : float | None, optional
        Strict spherical cutoff in cycles per Angstrom.  ``None`` uses the
        common Nyquist frequency.  Modes exactly on the cutoff are excluded
        so an even-grid Nyquist endpoint cannot break Hermitian symmetry.
    parameterization : str, optional
        Atomic form-factor model, ``"lobato"`` (default) or ``"kirkland"``.

    Returns
    -------
    volume : Float[Array, "nz ny nx"]
        Band-limited electrostatic potential samples in volts.

    Raises
    ------
    ValueError
        If static geometry is invalid, an atomic number is outside 1--103,
        or coordinates are non-finite.

    Notes
    -----
    The reciprocal amplitude is

    ``Phi_tilde(g) = C_MB f_Z(g) exp(-2 pi i g . r_atom)``.

    JAX ``ifftn`` supplies ``1 / N``; division by the voxel volume therefore
    yields ``1 / box_volume`` per Fourier-series mode.  The zero mode is
    retained, preserving the physical IAM vacuum reference and its nonzero
    discrete mean.
    """
    voxel_xyz: tuple[float, float, float] = _voxel_size_tuple(voxel_size)
    shape_zyx: tuple[int, int, int] = _grid_shape_tuple(grid_shape)
    origin_xyz: tuple[float, float, float] = _origin_tuple(origin)
    cutoff: float = _band_limit_value(band_limit, voxel_xyz)
    checked_number: Int[Array, ""] = _checked_single_atomic_number(atom_no)

    if center_coords is None:
        nz: int
        ny: int
        nx: int
        nz, ny, nx = shape_zyx
        box_xyz: tuple[float, float, float] = (
            nx * voxel_xyz[0],
            ny * voxel_xyz[1],
            nz * voxel_xyz[2],
        )
        atom_position: Float[Array, " 3"] = jnp.asarray(
            tuple(
                origin_value + 0.5 * box_value
                for origin_value, box_value in zip(
                    origin_xyz,
                    box_xyz,
                    strict=True,
                )
            ),
            dtype=jnp.float64,
        )
    else:
        center_arr: Num[Array, "..."] = jnp.asarray(center_coords)
        if center_arr.shape != (_XYZ_SIZE,):
            raise ValueError("center_coords must have shape (3,)")
        atom_position = _checked_positions(center_arr[None, :])[0]

    volume: Float[Array, "nz ny nx"] = _potential_from_atoms(
        checked_number[None],
        atom_position[None, :],
        voxel_xyz,
        shape_zyx,
        origin_xyz,
        cutoff,
        parameterization,
    )
    return volume


def _crystal_arrays(
    crystal: _CrystalInput,
) -> tuple[Float[Array, "N 3"], Array]:
    """Extract xyz positions and atomic numbers from either carrier."""
    if isinstance(crystal, CrystalData):
        return crystal.positions, crystal.atomic_numbers
    if isinstance(crystal, CrystalStructure):
        cartesian: Num[Array, "..."] = jnp.asarray(crystal.cart_positions)
        if (
            cartesian.ndim != _MATRIX_RANK
            or cartesian.shape[1] != _POSITION_COLUMNS
        ):
            raise ValueError("crystal.cart_positions must have shape (N, 4)")
        return cartesian[:, :3].astype(jnp.float64), cartesian[:, 3]
    raise TypeError("crystal must be CrystalData or CrystalStructure")


def _infer_grid_geometry(
    crystal: _CrystalInput,
    voxel_size: tuple[float, float, float],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Infer a cell-preserving orthogonal grid and actual voxel spacing."""
    if isinstance(crystal, CrystalStructure):
        angles = tuple(float(value) for value in crystal.cell_angles)
        if not all(
            math.isclose(value, 90.0, abs_tol=1e-8) for value in angles
        ):
            raise ValueError(
                "automatic grid inference requires an orthogonal crystal cell"
            )
        lengths_xyz: tuple[float, float, float] = tuple(
            float(value) for value in crystal.cell_lengths
        )  # type: ignore[assignment]
    else:
        if crystal.lattice is None:
            raise ValueError(
                "automatic grid inference requires crystal.lattice"
            )
        lattice = jnp.asarray(crystal.lattice, dtype=jnp.float64)
        diagonal = jnp.diag(jnp.diag(lattice))
        if not bool(jnp.allclose(lattice, diagonal, rtol=0.0, atol=1e-10)):
            raise ValueError(
                "automatic grid inference requires an axis-aligned "
                "orthogonal lattice"
            )
        lengths_xyz = (
            float(abs(lattice[0, 0])),
            float(abs(lattice[1, 1])),
            float(abs(lattice[2, 2])),
        )
    if not all(math.isfinite(value) and value > 0.0 for value in lengths_xyz):
        raise ValueError("crystal cell lengths must be positive and finite")

    nx: int = math.ceil(lengths_xyz[0] / voxel_size[0])
    ny: int = math.ceil(lengths_xyz[1] / voxel_size[1])
    nz: int = math.ceil(lengths_xyz[2] / voxel_size[2])
    actual_voxel_size: tuple[float, float, float] = (
        lengths_xyz[0] / nx,
        lengths_xyz[1] / ny,
        lengths_xyz[2] / nz,
    )
    return (nz, ny, nx), actual_voxel_size


@jaxtyped(typechecker=beartype)
def crystal_potential_volume(
    crystal: _CrystalInput,
    voxel_size: _VoxelSizeInput,
    grid_shape: _GridShapeInput | None = None,
    *,
    origin: _StaticXYZ = (0.0, 0.0, 0.0),
    band_limit: scalar_num | None = None,
    parameterization: str = "lobato",
) -> Potential3D:
    """Build a full 3D IAM voltage field from atom positions.

    Parameters
    ----------
    crystal : CrystalData | CrystalStructure
        Crystal carrier containing Cartesian atom positions and species.
    voxel_size : Real | Sequence[float]
        Scalar isotropic spacing or ``(dx, dy, dz)`` in Angstroms. When
        ``grid_shape`` is omitted, these are maximum requested spacings;
        the actual spacings are reduced as needed to preserve the crystal
        cell lengths exactly.
    grid_shape : Sequence[int] | None, optional
        Static ``(nz, ny, nx)`` volume shape.  If omitted, infer it from an
        axis-aligned orthogonal lattice (or orthogonal CrystalStructure cell).
        Supply it explicitly when differentiating or compiling over crystals.
    origin : Sequence[float], optional
        Coordinate of sample ``volume[0, 0, 0]`` in ``(x, y, z)`` order.
    band_limit : float | None, optional
        Strict spherical cutoff in cycles per Angstrom.  ``None`` selects the
        common Nyquist frequency.
    parameterization : str, optional
        Atomic form-factor model.  Default: ``"lobato"``.

    Returns
    -------
    potential : Potential3D
        Validated band-limited scalar potential with voltage/reference,
        geometry, boundary, normalization, and provenance metadata.

    Raises
    ------
    ValueError
        If geometry, positions, species, or the selected band are invalid.
    TypeError
        If ``crystal`` is not a canonical supported crystal carrier.

    Notes
    -----
    Every atom contributes to every retained three-dimensional Fourier mode.
    Its continuous position enters only through an analytic phase ramp, so
    directional derivatives with respect to all three position components are
    finite.  The builder performs no projection, slicing, or mean removal.
    """
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")
    requested_voxel_xyz: tuple[float, float, float] = _voxel_size_tuple(
        voxel_size
    )
    shape_zyx: tuple[int, int, int]
    voxel_xyz: tuple[float, float, float]
    if grid_shape is None:
        shape_zyx, voxel_xyz = _infer_grid_geometry(
            crystal,
            requested_voxel_xyz,
        )
    else:
        shape_zyx = _grid_shape_tuple(grid_shape)
        voxel_xyz = requested_voxel_xyz
    origin_xyz: tuple[float, float, float] = _origin_tuple(origin)
    cutoff: float = _band_limit_value(band_limit, voxel_xyz)
    positions: Float[Array, "N 3"]
    atomic_numbers: Array
    positions, atomic_numbers = _crystal_arrays(crystal)
    volume: Float[Array, "nz ny nx"] = _potential_from_atoms(
        atomic_numbers,
        positions,
        voxel_xyz,
        shape_zyx,
        origin_xyz,
        cutoff,
        parameterization,
    )

    nz: int
    ny: int
    nx: int
    nz, ny, nx = shape_zyx
    box_xyz: tuple[float, float, float] = (
        nx * voxel_xyz[0],
        ny * voxel_xyz[1],
        nz * voxel_xyz[2],
    )
    provenance_hash: str = (
        _LOBATO_PROVENANCE_SHA256
        if parameterization == "lobato"
        else _KIRKLAND_PROVENANCE_SHA256
    )
    potential: Potential3D = create_potential_3d(
        volume=volume,
        voxel_size=voxel_xyz,
        box_size=box_xyz,
        origin=origin_xyz,
        units="V",
        reference_value=0.0,
        reference_semantics=(
            "isolated-neutral-atom vacuum zero at infinity; periodic IAM "
            "zero Fourier mode retained"
        ),
        boundary="periodic orthogonal box",
        producer=f"{parameterization} independent-atom model",
        provenance_hash=provenance_hash,
        coefficient_normalization=_COEFFICIENT_NORMALIZATION,
        band_limit=cutoff,
    )
    return potential


__all__: list[str] = [
    "crystal_potential_volume",
    "single_atom_potential_3d",
]
