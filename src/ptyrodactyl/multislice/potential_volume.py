"""Build band-limited three-dimensional independent-atom potentials.

Extended Summary
----------------
This module constructs a true volumetric electrostatic field from analytic
atomic form factors.  Atoms are translated by reciprocal-space phase ramps,
so their positions remain continuous differentiable parameters; no rounded
voxel deposition or projection is used. The inverse DFT uses the continuous
Fourier convention required by the scalar-potential contract.

Routine Listings
----------------
:func:`crystal_potential_volume`
    Build a full 3D IAM voltage field from atom positions.
:func:`single_atom_potential_3d`
    Build one band-limited three-dimensional atomic potential.

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
from beartype.typing import Tuple
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


def _grid_shape_tuple(grid_shape: _GridShapeInput) -> Tuple[int, int, int]:
    """PRIVATE: Validate a static ``(nz, ny, nx)`` grid shape.

    Parameters
    ----------
    grid_shape : _GridShapeInput
        Candidate grid shape in storage-axis order.

    Returns
    -------
    nz : int
        Positive storage-axis depth.
    ny : int
        Positive storage-axis height.
    nx : int
        Positive storage-axis width.

    Raises
    ------
    ValueError
        If the shape does not contain exactly three positive integers.
    """
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
    result: Tuple[int, int, int] = (nz, ny, nx)
    return result


def _voxel_size_tuple(
    voxel_size: _VoxelSizeInput,
) -> Tuple[float, float, float]:
    """PRIVATE: Normalize voxel spacing to a positive static xyz tuple.

    Parameters
    ----------
    voxel_size : _VoxelSizeInput
        Scalar spacing or physical ``(dx, dy, dz)`` spacing in Angstroms.

    Returns
    -------
    dx : float
        Positive finite x spacing in Angstroms.
    dy : float
        Positive finite y spacing in Angstroms.
    dz : float
        Positive finite z spacing in Angstroms.

    Raises
    ------
    ValueError
        If a value is Boolean, non-real, non-finite, non-positive, or has an
        unsupported shape.
    """
    if isinstance(voxel_size, bool) or (
        isinstance(voxel_size, Sequence)
        and any(isinstance(value, bool) for value in voxel_size)
    ):
        raise ValueError("voxel_size values must not be booleans")
    if isinstance(voxel_size, Real) and not isinstance(voxel_size, bool):
        scalar_spacing: float = float(voxel_size)
        dx: float = scalar_spacing
        dy: float = scalar_spacing
        dz: float = scalar_spacing
        values: Tuple[float, float, float] = (
            dx,
            dy,
            dz,
        )
    else:
        voxel_array: Num[Array, "..."] = jnp.asarray(voxel_size)
        if jnp.issubdtype(voxel_array.dtype, jnp.bool_):
            raise ValueError("voxel_size values must not be booleans")
        if voxel_array.shape == ():
            scalar_spacing = float(voxel_array)
            dx = scalar_spacing
            dy = scalar_spacing
            dz = scalar_spacing
            values = (
                dx,
                dy,
                dz,
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
            voxel_values: Tuple[float, ...] = tuple(
                float(value) for value in voxel_array
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "voxel_size values must be real numbers"
            ) from error
        dx = voxel_values[0]
        dy = voxel_values[1]
        dz = voxel_values[2]
        values = (dx, dy, dz)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("voxel_size values must be positive and finite")
    return values


def _origin_tuple(origin: _StaticXYZ) -> Tuple[float, float, float]:
    """PRIVATE: Validate a static physical ``(x, y, z)`` origin.

    Parameters
    ----------
    origin : _StaticXYZ
        Physical ``(x, y, z)`` origin in Angstroms.

    Returns
    -------
    x : float
        Finite physical x origin in Angstroms.
    y : float
        Finite physical y origin in Angstroms.
    z : float
        Finite physical z origin in Angstroms.

    Raises
    ------
    ValueError
        If the origin is not three finite real non-Boolean values.
    """
    if isinstance(origin, str | bytes) or len(origin) != _XYZ_SIZE:
        raise ValueError("origin must contain exactly (x, y, z)")
    if any(isinstance(value, bool) for value in origin):
        raise ValueError("origin values must not be booleans")
    origin_array: Num[Array, " 3"] = jnp.asarray(origin)
    if jnp.issubdtype(origin_array.dtype, jnp.bool_):
        raise ValueError("origin values must not be booleans")
    try:
        origin_values: Tuple[float, ...] = tuple(
            float(value) for value in origin
        )
    except (TypeError, ValueError) as error:
        raise ValueError("origin values must be real numbers") from error
    x: float = origin_values[0]
    y: float = origin_values[1]
    z: float = origin_values[2]
    result: Tuple[float, float, float] = (x, y, z)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("origin values must be finite")
    return result


def _band_limit_value(
    band_limit: scalar_num | None,
    voxel_size: Tuple[float, float, float],
) -> float:
    """PRIVATE: Validate or choose the common spherical Nyquist band limit.

    Parameters
    ----------
    band_limit : scalar_num | None
        Spherical cutoff in cycles per Angstrom, or ``None`` to use the
        common Nyquist limit. Default is ``None``.
    voxel_size : Tuple[float, float, float]
        Physical ``(dx, dy, dz)`` spacing in Angstroms.

    Returns
    -------
    value : float
        Validated cutoff, or the common Nyquist limit when omitted.

    Raises
    ------
    ValueError
        If the cutoff is Boolean, non-finite, non-positive, or exceeds the
        common Nyquist limit.

    Notes
    -----
    The default is the smallest axis-wise Nyquist frequency, which defines a
    sphere representable on every physical grid axis.
    """
    common_nyquist: float = min(0.5 / spacing for spacing in voxel_size)
    if band_limit is None:
        value: float = common_nyquist
        return value
    band_limit_array: Num[Array, ""] = jnp.asarray(band_limit)
    if isinstance(band_limit, bool) or jnp.issubdtype(
        band_limit_array.dtype,
        jnp.bool_,
    ):
        raise ValueError("band_limit must be positive and finite")
    value = float(band_limit)
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
    """PRIVATE: Reject invalid atomic numbers under tracing.

    Parameters
    ----------
    atomic_numbers : Array
        One-dimensional atomic-number values.

    Returns
    -------
    result : Int[Array, " N"]
        Validated atomic numbers stored as signed 32-bit integers.

    Raises
    ------
    ValueError
        If the array is not one-dimensional, is Boolean, or is complex.
    equinox.EquinoxRuntimeError
        If a traced value is non-finite, non-integral, or outside ``[1, 103]``.
    """
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
    result: Int[Array, " N"] = checked_numbers.astype(jnp.int32)
    return result


def _checked_positions(
    positions: Num[Array, "N 3"],
) -> Float[Array, "N 3"]:
    """PRIVATE: Coerce xyz atom positions and attach a finite-value check.

    Parameters
    ----------
    positions : Num[Array, "N 3"]
        Cartesian atom positions in Angstroms.

    Returns
    -------
    checked : Float[Array, "N 3"]
        Binary64 positions with the traced finite-value check attached.

    Raises
    ------
    ValueError
        If positions are complex or do not have shape ``(N, 3)``.
    equinox.EquinoxRuntimeError
        If a traced coordinate is non-finite.
    """
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
    """PRIVATE: Validate one atomic number without clipping traced values.

    Parameters
    ----------
    atom_no : scalar_int
        Candidate one-based atomic number.

    Returns
    -------
    result : Int[Array, ""]
        Validated scalar atomic number.

    Raises
    ------
    ValueError
        If ``atom_no`` is not scalar or has an invalid structural dtype.
    equinox.EquinoxRuntimeError
        If a traced value is non-finite, non-integral, or outside ``[1, 103]``.
    """
    raw_number: Num[Array, ""] = jnp.asarray(atom_no)
    if raw_number.shape != ():
        raise ValueError("atom_no must be a scalar")
    checked: Int[Array, " 1"] = _checked_atomic_numbers(raw_number[None])
    result: Int[Array, ""] = checked[0]
    return result


def _reciprocal_grid(
    voxel_size: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
) -> Tuple[
    Float[Array, "nz ny nx"],
    Float[Array, "nz ny nx"],
    Float[Array, "nz ny nx"],
    Float[Array, "nz ny nx"],
]:
    """PRIVATE: Return Cartesian DFT frequencies and their magnitude.

    Parameters
    ----------
    voxel_size : Tuple[float, float, float]
        Physical ``(dx, dy, dz)`` spacing in Angstroms.
    grid_shape : Tuple[int, int, int]
        Storage shape ``(nz, ny, nx)``.

    Returns
    -------
    gx : Float[Array, "nz ny nx"]
        X frequency in cycles per Angstrom.
    gy : Float[Array, "nz ny nx"]
        Y frequency in cycles per Angstrom.
    gz : Float[Array, "nz ny nx"]
        Z frequency in cycles per Angstrom.
    magnitude : Float[Array, "nz ny nx"]
        Cartesian frequency magnitude in cycles per Angstrom.

    Notes
    -----
    ``fftfreq`` defines cycles per Angstrom here; callers multiply the
    magnitude by ``2 * pi`` before evaluating angular-frequency form factors.
    """
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
    result: Tuple[
        Float[Array, "nz ny nx"],
        Float[Array, "nz ny nx"],
        Float[Array, "nz ny nx"],
        Float[Array, "nz ny nx"],
    ] = (gx, gy, gz, magnitude)
    return result


def _potential_from_atoms(
    atomic_numbers: Array,
    positions: Num[Array, "N 3"],
    voxel_size: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    band_limit: float,
    parameterization: str,
) -> Float[Array, "nz ny nx"]:
    """PRIVATE: Evaluate the periodic Fourier series for all atoms.

    Parameters
    ----------
    atomic_numbers : Array
        One-dimensional atomic numbers.
    positions : Num[Array, "N 3"]
        Cartesian atom positions in Angstroms.
    voxel_size : Tuple[float, float, float]
        Physical ``(dx, dy, dz)`` spacing in Angstroms.
    grid_shape : Tuple[int, int, int]
        Storage shape ``(nz, ny, nx)``.
    origin : Tuple[float, float, float]
        Physical ``(x, y, z)`` origin in Angstroms.
    band_limit : float
        Spherical cutoff in cycles per Angstrom.
    parameterization : str
        Independent-atom form-factor parameterization.

    Returns
    -------
    volume : Float[Array, "nz ny nx"]
        Periodic electrostatic potential in volts.

    Raises
    ------
    ValueError
        If the atomic-number and position arrays have different lengths or
        fail structural validation.
    equinox.EquinoxRuntimeError
        If traced atom data fails value validation.

    Notes
    -----
    Each atom contributes
    ``C_MB * f_Z(2 pi |g|) * exp(-2 pi i g . (r - origin))`` inside the strict
    spherical spectral mask. The inverse transform normalization is applied
    by the caller after accumulation.
    """
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
        atom: Tuple[Int[Array, ""], Float[Array, " 3"]],
    ) -> Tuple[jax.Array, None]:
        """PRIVATE: Accumulate one translated atomic Fourier field.

        Parameters
        ----------
        coefficients : jax.Array
            Running complex Fourier coefficients in volt-Angstrom cubed.
        atom : Tuple[Int[Array, ""], Float[Array, " 3"]]
            Atomic number and physical xyz position in Angstroms.

        Returns
        -------
        updated_coefficients : jax.Array
            Fourier coefficients in volt-Angstrom cubed with the atom
            contribution added.
        auxiliary : None
            No stacked scan output.
        """
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
        updated_coefficients: jax.Array = coefficients + atom_coefficients
        auxiliary: None = None
        result: Tuple[jax.Array, None] = updated_coefficients, auxiliary
        return result

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

    :see: :mod:`~.test_potential_volume`

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
    voxel_xyz: Tuple[float, float, float] = _voxel_size_tuple(voxel_size)
    shape_zyx: Tuple[int, int, int] = _grid_shape_tuple(grid_shape)
    origin_xyz: Tuple[float, float, float] = _origin_tuple(origin)
    cutoff: float = _band_limit_value(band_limit, voxel_xyz)
    checked_number: Int[Array, ""] = _checked_single_atomic_number(atom_no)

    if center_coords is None:
        nz: int
        ny: int
        nx: int
        nz, ny, nx = shape_zyx
        box_xyz: Tuple[float, float, float] = (
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
) -> Tuple[Float[Array, "N 3"], Array]:
    """PRIVATE: Extract xyz positions and atomic numbers from either carrier.

    Parameters
    ----------
    crystal : _CrystalInput
        Crystal carrier containing Cartesian atom data.

    Returns
    -------
    positions : Float[Array, "N 3"]
        Cartesian xyz positions in Angstroms.
    atomic_numbers : Array
        Atomic numbers aligned with ``positions``.

    Raises
    ------
    ValueError
        If ``CrystalStructure.cart_positions`` does not have shape ``(N, 4)``.
    TypeError
        If ``crystal`` is not a supported carrier.
    """
    if isinstance(crystal, CrystalData):
        result: Tuple[Float[Array, "N 3"], Array] = (
            crystal.positions,
            crystal.atomic_numbers,
        )
        return result
    if isinstance(crystal, CrystalStructure):
        cartesian: Num[Array, "..."] = jnp.asarray(crystal.cart_positions)
        if (
            cartesian.ndim != _MATRIX_RANK
            or cartesian.shape[1] != _POSITION_COLUMNS
        ):
            raise ValueError("crystal.cart_positions must have shape (N, 4)")
        result: Tuple[Float[Array, "N 3"], Array] = (
            cartesian[:, :3].astype(jnp.float64),
            cartesian[:, 3],
        )
        return result
    raise TypeError("crystal must be CrystalData or CrystalStructure")


def _infer_grid_geometry(
    crystal: _CrystalInput,
    voxel_size: Tuple[float, float, float],
) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """PRIVATE: Infer an orthogonal grid and its actual voxel spacing.

    Parameters
    ----------
    crystal : _CrystalInput
        Crystal carrier with cell geometry.
    voxel_size : Tuple[float, float, float]
        Requested maximum ``(dx, dy, dz)`` spacing in Angstroms.

    Returns
    -------
    grid_shape : Tuple[int, int, int]
        Cell-preserving storage shape ``(nz, ny, nx)``.
    actual_voxel_size : Tuple[float, float, float]
        Actual physical ``(dx, dy, dz)`` spacing in Angstroms.

    Raises
    ------
    ValueError
        If cell geometry is absent, non-orthogonal, non-axis-aligned, or
        contains a non-positive or non-finite length.
    """
    if isinstance(crystal, CrystalStructure):
        angles = tuple(float(value) for value in crystal.cell_angles)
        if not all(
            math.isclose(value, 90.0, abs_tol=1e-8) for value in angles
        ):
            raise ValueError(
                "automatic grid inference requires an orthogonal crystal cell"
            )
        lengths_xyz: Tuple[float, float, float] = tuple(
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
    actual_voxel_size: Tuple[float, float, float] = (
        lengths_xyz[0] / nx,
        lengths_xyz[1] / ny,
        lengths_xyz[2] / nz,
    )
    result: Tuple[Tuple[int, int, int], Tuple[float, float, float]] = (
        (nz, ny, nx),
        actual_voxel_size,
    )
    return result


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

    :see: :mod:`~.test_potential_volume`

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
    requested_voxel_xyz: Tuple[float, float, float] = _voxel_size_tuple(
        voxel_size
    )
    shape_zyx: Tuple[int, int, int]
    voxel_xyz: Tuple[float, float, float]
    if grid_shape is None:
        shape_zyx, voxel_xyz = _infer_grid_geometry(
            crystal,
            requested_voxel_xyz,
        )
    else:
        shape_zyx = _grid_shape_tuple(grid_shape)
        voxel_xyz = requested_voxel_xyz
    origin_xyz: Tuple[float, float, float] = _origin_tuple(origin)
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
    box_xyz: Tuple[float, float, float] = (
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
