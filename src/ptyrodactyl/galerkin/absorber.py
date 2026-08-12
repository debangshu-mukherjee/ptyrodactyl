r"""Realize, certify, and apply the LVT-1 axial local-cell CAP.

Extended Summary
----------------
This leaf authenticates an L3 local-cell interaction core, realizes the
axis-only LVT.23 profile on the same grid, directly encloses the exact LVT.24
coefficients, proves exact LVT.26 difference coverage and LVT.31 operator
error, and constructs a verified support-only LVT.29a floor.  Exact-target
and realized-matrix eligibility remain separate.  The action callables apply
the physical frozen matrix ``B_alg = epsilon_alg A_alg`` and make no per-call
floating-point error claim.

Routine Listings
----------------
:func:`apply_axial_physical_cap`
    Apply the frozen physical algebraic CAP ``B_alg``.
:func:`apply_axial_physical_cap_adjoint`
    Apply the formal matrix adjoint of frozen physical ``B_alg``.
:func:`certify_axial_cap_floor`
    Certify exact LVT.29a and, independently, realized LVT.32 floors.
:func:`certify_axial_cell_absorber`
    Certify a finite Hermitian approximant against exact LVT.24.
:func:`prepare_axial_cap_floor`
    Replay all nested public evidence before transform-compatible use.
:func:`realize_axial_cell_absorber`
    Realize one canonical Hermitian LVT.24 coefficient approximant.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Iterable, Tuple
from jax.core import Tracer
from jaxtyping import (
    Array,
    Complex,
    Complex128,
    Float,
    Float64,
    Int64,
    Shaped,
    jaxtyped,
)
from numpy.typing import NDArray

from ptyrodactyl._tools import (
    ComplexRectangle,
    RootEnclosureError,
    complex_rectangle_multiply,
    conjugate_rectangle,
    fraction_from_float,
    fraction_lower_float,
    fraction_upper_float,
    has_subnormal_components,
    host_binary64_supported,
    normalized_sinc_integer_ratio,
    pairwise_rectangle_sum,
    rational_turn_exponential,
    scale_complex_rectangle,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types import (
    GalerkinAxialCapCoefficientCertificate,
    GalerkinAxialCapCoefficientFailure,
    GalerkinAxialCapExactFloorFailure,
    GalerkinAxialCapFloorProof,
    GalerkinAxialCapRealizedFloorFailure,
    GalerkinAxialCapRealizedFloorRoute,
    GalerkinAxialCellAbsorber,
    GalerkinLocalCellInteractionCore,
    _make_axial_cap_coefficient_certificate,
    _make_axial_cap_floor_proof,
    _make_axial_cell_absorber,
)

from .local_cell_interaction import prepare_local_cell_interaction_core

_ABSORBER_COMPLETION_SCOPE: str = (
    "axial_CAP_only; no source, free diagonal, terminal, or solver target"
)
_ARITHMETIC: str = (
    "direct exact-rational LVT.24 rectangles with outward binary64 storage"
)
_CERTIFICATE_DOMAIN: str = "ptyrodactyl.local_cell.lvt24_lvt31_certificate.v1"
_COEFFICIENT_FORMULA: str = (
    "LVT.24 axis-only cell coefficient with symbolic transverse zeros v1"
)
_COEFFICIENT_EXACT_TARGET: str = (
    "exact LVT.24 coefficients of authenticated LVT.23 profile"
)
_COVERAGE_CLAIM: str = (
    "exact ordinary-integer Du subset Ia with row-major Iu-pair map and "
    "LVT.17 multiplicities"
)
_DEFAULT_GRAM_PRECISION_BITS: int = 48
_DEFAULT_LDL_ITERATION_COUNT: int = 64
_DEFAULT_MAXIMUM_DIRECT_TERMS: int = 2_000_000
_DEFAULT_MAXIMUM_GRAM_DEGREE: int = 64
_DEFAULT_MAXIMUM_GRAM_WORK: int = 50_000_000
_DEFAULT_MAXIMUM_STATE_PAIRS: int = 20_000_000
_EXACT_FLOOR_TARGET: str = (
    "exact LVT.29a target epsilon_CAP*a_P*mu_P in inverse-square Angstroms "
    "on one fixed finite support"
)
_EXACT_PROFILE_TARGET: str = (
    "LVT.23 exact binary64 layer values as cellwise-constant real profile"
)
_GRAM_PROOF_ROUTE: str = (
    "LVT.30 support span and principal-submatrix interlacing; strict "
    "rational subinterval; exact-rational Hermitian midpoint LDL-star plus "
    "Frobenius-Weyl v1"
)
_GRAM_TRANSCRIPT_DOMAIN: str = (
    "ptyrodactyl.local_cell.lvt30_gram_ldl_weyl_transcript.v1"
)
_GRAM_WORK_SCOPE: str = (
    "versioned exact preflight upper count of abstract host work units: "
    "rational trig-series terms plus exact LDL-star trial cubic units; "
    "excludes Python big-integer bit complexity"
)
_HERMITIAN_APPROXIMANT_CLAIM: str = (
    "actual finite complex128 Ia approximant has exact numeric ordinary "
    "signed-index Hermitian symmetry; provenance unrestricted"
)
_MAXIMUM_GRAM_PRECISION_BITS: int = 256
_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_MINIMUM_GRAM_PRECISION_BITS: int = 12
_OPERATOR_DIGEST_DOMAIN: str = "ptyrodactyl.local_cell.lvt24_cap_operator.v1"
_OPERATOR_ERROR_SCOPE: str = (
    "fixed exact-Hermitian A_alg minus exact LVT.24 compression only"
)
_PER_CALL_ARITHMETIC_EXCLUSION: str = (
    "excludes every per-call scale multiply, coefficient multiply, "
    "accumulation, transform, and output rounding error"
)
_PROOF_DIGEST_DOMAIN: str = "ptyrodactyl.local_cell.lvt29_lvt32_floor.v1"
_REALIZED_FLOOR_SCOPE: str = (
    "fixed exact-real B_alg matrix only; no per-call arithmetic, box, band, "
    "CAP-removal, reflection, outgoing, or continuum claim"
)
_SCALE_SEMANTICS: str = (
    "epsilon_CAP and epsilon_alg are positive exact stored binary64 reals; "
    "both have inverse-square Angstrom (Angstrom^-2) units; route 32a iff "
    "exactly equal, otherwise route 32b"
)
_SOURCE_DIGEST_DOMAIN: str = "ptyrodactyl.local_cell.lvt23_cap_source.v1"
_SPACE_DIMENSIONS: int = 3
_ZERO_DIGEST: str = "0" * 64

type _ComplexQ = Tuple[Fraction, Fraction]
type _FractionInterval = Tuple[Fraction, Fraction]
type _FractionIntervalPair = Tuple[_FractionInterval, _FractionInterval]
type _HostCoefficients = Complex128[NDArray, " a"]
type _HostAbsorberIndices = Int64[NDArray, "a 3"]
type _HostDifferenceIndices = Int64[NDArray, "d 3"]
type _HostDifferenceVector = Int64[NDArray, " d"]
type _HostFloatVector = Float64[NDArray, " n"]
type _HostIndices = Int64[NDArray, "n 3"]
type _HostIntVector = Int64[NDArray, " n"]
type _HostMode = Int64[NDArray, " 3"]
type _HostPairVector = Int64[NDArray, " p"]
type _HostStateIndices = Int64[NDArray, "s 3"]
type _DifferenceEvidence = (
    Tuple[
        _HostDifferenceIndices,
        _HostDifferenceVector,
        _HostDifferenceVector,
        _HostPairVector,
    ]
    | None
)


class _GramAttempt(TypedDict):
    """Type one exact Gram-proof attempt payload."""

    failure: GalerkinAxialCapExactFloorFailure
    work_count: int
    midpoint_shift: Fraction | None
    entry_error: Fraction | None
    gram_lower: Fraction | None
    transcript_digest: str


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject JAX tracers at an explicit host boundary.

    Parameters
    ----------
    value : object
        Internal value used by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError(
            "axial CAP certification requires concrete host values"
        )


def _checked_budget(value: int, name: str) -> int:
    """PRIVATE: Validate one positive signed-64-bit budget.

    Parameters
    ----------
    value : int
        Internal value used by this helper.
    name : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > _MAXIMUM_SIGNED_INT64
    ):
        raise ValueError(f"{name} must be a positive signed-64-bit integer")
    _return_value: int = value
    return _return_value


def _checked_nonnegative_budget(value: int, name: str) -> int:
    """PRIVATE: Validate one nonnegative signed-64-bit budget.

    Parameters
    ----------
    value : int
        Internal value used by this helper.
    name : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAXIMUM_SIGNED_INT64
    ):
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    _return_value: int = value
    return _return_value


def _checked_precision(value: int) -> int:
    """PRIVATE: Validate the bounded exact-rational Gram precision.

    Parameters
    ----------
    value : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    checked = _checked_budget(value, "gram_precision_bits")
    if (
        checked < _MINIMUM_GRAM_PRECISION_BITS
        or checked > _MAXIMUM_GRAM_PRECISION_BITS
    ):
        raise ValueError("gram_precision_bits must lie in [12, 256]")
    _return_value: int = checked
    return _return_value


def _host_exact_array(
    value: jax.Array,
    dtype: np.dtype,
    name: str,
) -> Shaped[NDArray, "..."]:
    """PRIVATE: Transfer an array while requiring exact dtype.

    Parameters
    ----------
    value : jax.Array
        Internal value used by this helper.
    dtype : np.dtype
        Internal value used by this helper.
    name : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : Shaped[NDArray, '...']
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    array = np.asarray(jax.device_get(value))
    if array.dtype != dtype:
        raise ValueError(f"{name} must have exact {dtype.name} dtype")
    _return_value: Shaped[NDArray, "..."] = array
    return _return_value


def _host_exact_scalar(
    value: jax.Array,
    dtype: np.dtype,
    name: str,
) -> bool | float | int:
    """PRIVATE: Transfer one exact-dtype scalar.

    Parameters
    ----------
    value : jax.Array
        Internal value used by this helper.
    dtype : np.dtype
        Internal value used by this helper.
    name : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : bool | float | int
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    array = _host_exact_array(value, dtype, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    _return_value: bool | float | int = array.item()
    return _return_value


def _checked_float_scalar(value: object, name: str) -> np.float64:
    """PRIVATE: Require one finite positive exact binary64 scalar.

    Parameters
    ----------
    value : object
        Internal value used by this helper.
    name : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : np.float64
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    array = np.asarray(jax.device_get(value))
    if (
        array.shape != ()
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array)
        or float(array) <= 0.0
        or float(array) < np.finfo(np.float64).tiny
    ):
        raise ValueError(
            f"{name} must be an exact finite positive normal float64 scalar"
        )
    _return_value: np.float64 = np.float64(array)
    return _return_value


def _checked_layer_values(value: object) -> Float64[NDArray, " l"]:
    """PRIVATE: Require a finite normal-or-zero float64 layer vector.

    Parameters
    ----------
    value : object
        Internal value used by this helper.

    Returns
    -------
    _return_value : Float64[NDArray, ' l']
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    array = np.asarray(jax.device_get(value))
    if (
        array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or not array.size
    ):
        raise ValueError(
            "layer_values must be one nonempty exact float64 vector"
        )
    tiny = np.finfo(np.float64).tiny
    if (
        not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
        or np.any((array != 0.0) & (np.abs(array) < tiny))
    ):
        raise ValueError(
            "layer_values must be finite normal-or-zero values in [0, 1]"
        )
    _return_value: Float64[NDArray, " l"] = array
    return _return_value


def _checked_indices(value: jax.Array, name: str) -> _HostIndices:
    """PRIVATE: Require one nonempty exact int64 support matrix.

    Parameters
    ----------
    value : jax.Array
        Internal value used by this helper.
    name : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : _HostIndices
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    raw = _host_exact_array(value, np.dtype(np.int64), name)
    if (
        raw.ndim != 2  # noqa: PLR2004
        or raw.shape[1:] != (_SPACE_DIMENSIONS,)
        or not raw.shape[0]
    ):
        raise ValueError(f"{name} must have shape (n, 3)")
    _return_value: _HostIndices = raw
    return _return_value


def _checked_terminal_axis(
    core: GalerkinLocalCellInteractionCore,
    terminal_axis: int,
) -> int:
    """PRIVATE: Bind one canonical axis to the nested acquisition.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    acquisition = core.compression.realization.support_eligibility.manifest
    if (
        isinstance(terminal_axis, bool)
        or not isinstance(terminal_axis, int)
        or terminal_axis not in range(_SPACE_DIMENSIONS)
        or terminal_axis != acquisition.terminal_axis
    ):
        raise ValueError(
            "terminal_axis must be 0, 1, or 2 and match the nested acquisition"
        )
    _return_value: int = terminal_axis
    return _return_value


def _check_grid_layer_count(
    core: GalerkinLocalCellInteractionCore,
    layer_values: Float64[NDArray, " l"],
    terminal_axis: int,
) -> None:
    """PRIVATE: Bind profile length to the same local-cell grid.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Internal value used by this helper.
    layer_values : Float64[NDArray, ' l']
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    cell_values = core.compression.realization.local_potential.cell_values
    grid_shape_xyz = tuple(reversed(cell_values.shape))
    if layer_values.shape != (grid_shape_xyz[terminal_axis],):
        raise ValueError(
            "layer_values must match the nested local-cell grid along "
            "terminal_axis"
        )


def _mode_tuple(row: _HostMode) -> Tuple[int, int, int]:
    """PRIVATE: Convert one reciprocal-index row to Python integers.

    Parameters
    ----------
    row : _HostMode
        Internal value used by this helper.

    Returns
    -------
    result_0 : int
        Internal result produced by this helper.
    result_1 : int
        Internal result produced by this helper.
    result_2 : int
        Internal result produced by this helper.
    """
    _return_value: Tuple[int, int, int] = (
        int(row[0]),
        int(row[1]),
        int(row[2]),
    )
    return _return_value


def _canonical_signed_mode(mode: Tuple[int, int, int]) -> bool:
    """PRIVATE: Select one ordinary-Hermitian signed-pair representative.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Internal value used by this helper.

    Returns
    -------
    _return_value : bool
        Internal result produced by this helper.
    """
    for component in mode:
        if component != 0:
            _return_value: bool = component > 0
            return _return_value
    _return_value: bool = True
    return _return_value


def _signed_position_map(
    indices: _HostIndices,
) -> Tuple[list[Tuple[int, int, int]], _HostIntVector]:
    """PRIVATE: Build and validate exact ordinary signed-index positions.

    Parameters
    ----------
    indices : _HostIndices
        Internal value used by this helper.

    Returns
    -------
    modes : list[Tuple[int, int, int]]
        Internal result produced by this helper.
    positions : _HostIntVector
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    minimum = np.iinfo(np.int64).min
    if np.any(indices == minimum):
        raise ValueError("absorber_indices contain an unnegatable integer")
    modes = [_mode_tuple(row) for row in indices]
    position_by_mode = {mode: position for position, mode in enumerate(modes)}
    if len(position_by_mode) != len(modes):
        raise ValueError("absorber_indices must be unique")
    positions = np.empty((len(modes),), dtype=np.int64)
    for position, mode in enumerate(modes):
        opposite = (-mode[0], -mode[1], -mode[2])
        if opposite not in position_by_mode:
            raise ValueError(
                "absorber_indices must be ordinarily sign symmetric"
            )
        positions[position] = position_by_mode[opposite]
    _return_value: Tuple[list[Tuple[int, int, int]], _HostIntVector] = (
        modes,
        positions,
    )
    return _return_value


def _checked_coefficients(
    value: jax.Array,
    modes: list[Tuple[int, int, int]],
    signed_positions: _HostIntVector,
) -> _HostCoefficients:
    """PRIVATE: Validate one finite exact-Hermitian complex128 approximant.

    Parameters
    ----------
    value : jax.Array
        Internal value used by this helper.
    modes : list[Tuple[int, int, int]]
        Internal value used by this helper.
    signed_positions : _HostIntVector
        Internal value used by this helper.

    Returns
    -------
    _return_value : _HostCoefficients
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    raw = _host_exact_array(
        value, np.dtype(np.complex128), "absorber_coefficients"
    )
    if raw.ndim != 1 or raw.shape != (len(modes),):
        raise ValueError("absorber_coefficients must match ordered I_a")
    tiny = np.finfo(np.float64).tiny
    subnormal = ((raw.real != 0.0) & (np.abs(raw.real) < tiny)) | (
        (raw.imag != 0.0) & (np.abs(raw.imag) < tiny)
    )
    if not np.all(np.isfinite(raw)) or np.any(subnormal):
        raise ValueError(
            "absorber_coefficients must be finite normal-or-zero complex128"
        )
    for position, mode in enumerate(modes):
        opposite_position = int(signed_positions[position])
        if raw[opposite_position] != np.complex128(
            np.conjugate(raw[position])
        ):
            raise ValueError(
                "absorber_coefficients must store exact Hermitian pairs"
            )
        if mode == (0, 0, 0) and float(raw[position].imag) != 0.0:
            raise ValueError("the absorber zero mode must be exactly real")
    _return_value: _HostCoefficients = raw
    return _return_value


def _source_digest(
    core: GalerkinLocalCellInteractionCore,
    layer_values: Float64[NDArray, " l"],
    plateau_floor: np.float64,
    exact_cap_scale: np.float64,
    algebraic_cap_scale: np.float64,
    *,
    terminal_axis: int,
    plateau_start: int,
    plateau_count: int,
    zero_start: int,
    zero_count: int,
) -> str:
    """PRIVATE: Digest the CAP source declaration without approximant.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Internal value used by this helper.
    layer_values : Float64[NDArray, ' l']
        Internal value used by this helper.
    plateau_floor : np.float64
        Internal value used by this helper.
    exact_cap_scale : np.float64
        Internal value used by this helper.
    algebraic_cap_scale : np.float64
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.
    plateau_start : int
        Internal value used by this helper.
    plateau_count : int
        Internal value used by this helper.
    zero_start : int
        Internal value used by this helper.
    zero_count : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : str
        Internal result produced by this helper.
    """
    _return_value: str = sha256(
        {
            "domain": _SOURCE_DIGEST_DOMAIN,
            "l3_operator_digest": core.operator_digest,
            "layer_values": stored_value_payload(layer_values),
            "plateau_floor": stored_value_payload(plateau_floor),
            "exact_cap_scale": stored_value_payload(exact_cap_scale),
            "algebraic_cap_scale": stored_value_payload(algebraic_cap_scale),
            "terminal_axis": terminal_axis,
            "plateau_start": plateau_start,
            "plateau_count": plateau_count,
            "zero_start": zero_start,
            "zero_count": zero_count,
            "exact_profile_target": _EXACT_PROFILE_TARGET,
            "coefficient_formula": _COEFFICIENT_FORMULA,
            "scale_semantics": _SCALE_SEMANTICS,
        }
    )
    return _return_value


def _operator_digest(
    source_digest: str,
    absorber_indices: _HostIndices,
    signed_positions: _HostIntVector,
    coefficients: _HostCoefficients,
) -> str:
    """PRIVATE: Digest exact operator identity including approximant bytes.

    Parameters
    ----------
    source_digest : str
        Internal value used by this helper.
    absorber_indices : _HostIndices
        Internal value used by this helper.
    signed_positions : _HostIntVector
        Internal value used by this helper.
    coefficients : _HostCoefficients
        Internal value used by this helper.

    Returns
    -------
    _return_value : str
        Internal result produced by this helper.
    """
    _return_value: str = sha256(
        {
            "domain": _OPERATOR_DIGEST_DOMAIN,
            "source_digest": source_digest,
            "absorber_indices": stored_value_payload(absorber_indices),
            "signed_absorber_positions": stored_value_payload(
                signed_positions
            ),
            "absorber_coefficients": stored_value_payload(coefficients),
            "hermitian_approximant_claim": _HERMITIAN_APPROXIMANT_CLAIM,
        }
    )
    return _return_value


def _rounded_lvt24_coefficients(
    layer_values: Float64[NDArray, " l"],
    modes: list[Tuple[int, int, int]],
    signed_positions: _HostIntVector,
    origin: float,
    length: float,
    terminal_axis: int,
) -> _HostCoefficients:
    """PRIVATE: Realize a canonical rounded and postprojected LVT.24 array.

    Parameters
    ----------
    layer_values : Float64[NDArray, ' l']
        Internal value used by this helper.
    modes : list[Tuple[int, int, int]]
        Internal value used by this helper.
    signed_positions : _HostIntVector
        Internal value used by this helper.
    origin : float
        Internal value used by this helper.
    length : float
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : _HostCoefficients
        Internal result produced by this helper.
    """
    layer_count = layer_values.shape[0]
    coefficients = np.zeros((len(modes),), dtype=np.complex128)
    rows = np.arange(layer_count, dtype=np.float64)
    for position, mode in enumerate(modes):
        if not _canonical_signed_mode(mode):
            continue
        if any(
            component != 0
            for axis, component in enumerate(mode)
            if axis != terminal_axis
        ):
            continue
        normal_mode = mode[terminal_axis]
        if normal_mode != 0 and normal_mode % layer_count == 0:
            continue
        if normal_mode == 0:
            value = np.complex128(np.sum(layer_values) / layer_count)
        else:
            phase = np.exp(
                np.complex128(-2j * np.pi * normal_mode) * rows / layer_count
            )
            series = np.sum(layer_values.astype(np.complex128) * phase)
            shape = np.sinc(normal_mode / layer_count)
            origin_phase = np.exp(
                np.complex128(-2j * np.pi * normal_mode * origin / length)
            )
            value = np.complex128(shape * origin_phase * series / layer_count)
        opposite_position = int(signed_positions[position])
        if opposite_position == position:
            coefficients[position] = np.complex128(float(value.real) + 0.0j)
        else:
            coefficients[position] = value
            coefficients[opposite_position] = np.complex128(
                np.conjugate(value)
            )
    _return_value: _HostCoefficients = coefficients
    return _return_value


def _build_absorber(  # noqa: PLR0913
    core: GalerkinLocalCellInteractionCore,
    layer_values: Float64[NDArray, " l"],
    plateau_floor: np.float64,
    exact_cap_scale: np.float64,
    algebraic_cap_scale: np.float64,
    coefficients: _HostCoefficients,
    signed_positions: _HostIntVector,
    *,
    terminal_axis: int,
    plateau_start: int,
    plateau_count: int,
    zero_start: int,
    zero_count: int,
) -> GalerkinAxialCellAbsorber:
    """PRIVATE: Construct a canonical absorber around coefficient bytes.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Internal value used by this helper.
    layer_values : Float64[NDArray, ' l']
        Internal value used by this helper.
    plateau_floor : np.float64
        Internal value used by this helper.
    exact_cap_scale : np.float64
        Internal value used by this helper.
    algebraic_cap_scale : np.float64
        Internal value used by this helper.
    coefficients : _HostCoefficients
        Internal value used by this helper.
    signed_positions : _HostIntVector
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.
    plateau_start : int
        Internal value used by this helper.
    plateau_count : int
        Internal value used by this helper.
    zero_start : int
        Internal value used by this helper.
    zero_count : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCellAbsorber
        Internal result produced by this helper.
    """
    indices = _checked_indices(
        core.support.absorber_indices, "absorber_indices"
    )
    source_digest = _source_digest(
        core,
        layer_values,
        plateau_floor,
        exact_cap_scale,
        algebraic_cap_scale,
        terminal_axis=terminal_axis,
        plateau_start=plateau_start,
        plateau_count=plateau_count,
        zero_start=zero_start,
        zero_count=zero_count,
    )
    operator_digest = _operator_digest(
        source_digest, indices, signed_positions, coefficients
    )
    absorber = _make_axial_cell_absorber(
        core,
        jnp.asarray(layer_values),
        jnp.asarray(plateau_floor),
        jnp.asarray(exact_cap_scale),
        jnp.asarray(algebraic_cap_scale),
        jnp.asarray(coefficients),
        jnp.asarray(signed_positions),
        terminal_axis=terminal_axis,
        plateau_start=plateau_start,
        plateau_count=plateau_count,
        zero_start=zero_start,
        zero_count=zero_count,
        exact_profile_target=_EXACT_PROFILE_TARGET,
        coefficient_formula=_COEFFICIENT_FORMULA,
        hermitian_approximant_claim=_HERMITIAN_APPROXIMANT_CLAIM,
        scale_semantics=_SCALE_SEMANTICS,
        completion_scope=_ABSORBER_COMPLETION_SCOPE,
        source_digest=source_digest,
        operator_digest=operator_digest,
    )
    jax.block_until_ready(absorber)
    _return_value: GalerkinAxialCellAbsorber = absorber
    return _return_value  # noqa: RET504


def _canonical_absorber(
    submitted: GalerkinAxialCellAbsorber,
) -> GalerkinAxialCellAbsorber:
    """PRIVATE: Rebuild L3 data and canonicalize an actual approximant.

    Parameters
    ----------
    submitted : GalerkinAxialCellAbsorber
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCellAbsorber
        Internal result produced by this helper.
    """
    _assert_concrete(submitted)
    core = prepare_local_cell_interaction_core(submitted.interaction_core)
    terminal_axis = _checked_terminal_axis(core, submitted.terminal_axis)
    layer_values = _checked_layer_values(submitted.layer_values)
    _check_grid_layer_count(core, layer_values, terminal_axis)
    plateau_floor = _checked_float_scalar(
        submitted.plateau_floor, "plateau_floor"
    )
    exact_scale = _checked_float_scalar(
        submitted.exact_cap_scale, "exact_cap_scale"
    )
    algebraic_scale = _checked_float_scalar(
        submitted.algebraic_cap_scale, "algebraic_cap_scale"
    )
    indices = _checked_indices(
        core.support.absorber_indices, "absorber_indices"
    )
    modes, signed_positions = _signed_position_map(indices)
    coefficients = _checked_coefficients(
        submitted.absorber_coefficients, modes, signed_positions
    )
    canonical = _build_absorber(
        core,
        layer_values,
        plateau_floor,
        exact_scale,
        algebraic_scale,
        coefficients,
        signed_positions,
        terminal_axis=terminal_axis,
        plateau_start=submitted.plateau_start,
        plateau_count=submitted.plateau_count,
        zero_start=submitted.zero_start,
        zero_count=submitted.zero_count,
    )
    _return_value: GalerkinAxialCellAbsorber = canonical
    return _return_value  # noqa: RET504


@jaxtyped(typechecker=beartype)
def realize_axial_cell_absorber(  # noqa: PLR0913
    interaction_core: GalerkinLocalCellInteractionCore,
    layer_values: Float[Array, " l"],
    *,
    terminal_axis: int,
    plateau_start: int,
    plateau_count: int,
    plateau_floor: float | jax.Array | Shaped[NDArray, ""],
    zero_start: int,
    zero_count: int,
    exact_cap_scale: float | jax.Array | Shaped[NDArray, ""],
    algebraic_cap_scale: float | jax.Array | Shaped[NDArray, ""] | None = None,
) -> GalerkinAxialCellAbsorber:
    """Realize one canonical Hermitian LVT.24 coefficient approximant.

    :see: :func:`~.test_absorber.\
test_axis_two_anisotropic_wrapped_hard_plateau_and_wide_modes`

    Parameters
    ----------
    interaction_core : GalerkinLocalCellInteractionCore
        Concrete accepted L3 core, fully replayed before CAP construction.
    layer_values : Float[Array, " l"]
        Exact float64 dimensionless axis-layer values in ``[0, 1]``.
    terminal_axis : int
        Physical xyz axis, required to equal the nested acquisition axis.
    plateau_start : int
        First periodic full-face plateau layer.
    plateau_count : int
        Positive count of consecutive periodic plateau layers.
    plateau_floor : float or jax.Array or Shaped[NDArray, ""]
        Positive float64 ``a_P`` attained on every plateau layer.
    zero_start : int
        First periodic exact-zero layer.
    zero_count : int
        Positive count of consecutive exact-zero layers.
    exact_cap_scale : float or jax.Array or Shaped[NDArray, ""]
        Positive exact stored binary64 physical target scale in inverse-square
        Angstroms (Angstrom^-2).
    algebraic_cap_scale : float or jax.Array or Shaped[NDArray, ""] or None,
        optional
        Positive exact stored binary64 frozen-matrix scale in inverse-square
        Angstroms (Angstrom^-2). ``None`` copies ``exact_cap_scale`` exactly.

    Returns
    -------
    absorber : GalerkinAxialCellAbsorber
        Non-solver-ready profile and canonical rounded Hermitian approximant.

    Raises
    ------
    ValueError
        If nested replay, storage, axis, grid, block, scale, or range checks
        fail.
    """
    _assert_concrete(
        (interaction_core, layer_values, exact_cap_scale, algebraic_cap_scale)
    )
    core = prepare_local_cell_interaction_core(interaction_core)
    checked_axis = _checked_terminal_axis(core, terminal_axis)
    values = _checked_layer_values(layer_values)
    _check_grid_layer_count(core, values, checked_axis)
    exact_scale = _checked_float_scalar(exact_cap_scale, "exact_cap_scale")
    algebraic_input = (
        exact_cap_scale if algebraic_cap_scale is None else algebraic_cap_scale
    )
    algebraic_scale = _checked_float_scalar(
        algebraic_input, "algebraic_cap_scale"
    )
    floor = _checked_float_scalar(plateau_floor, "plateau_floor")
    indices = _checked_indices(
        core.support.absorber_indices, "absorber_indices"
    )
    modes, signed_positions = _signed_position_map(indices)
    local_potential = core.compression.realization.local_potential
    origin = float(local_potential.cell_center_origin[checked_axis])
    length = float(local_potential.box_size[checked_axis])
    coefficients = _rounded_lvt24_coefficients(
        values,
        modes,
        signed_positions,
        origin,
        length,
        checked_axis,
    )
    _checked_coefficients(jnp.asarray(coefficients), modes, signed_positions)
    absorber: GalerkinAxialCellAbsorber = _build_absorber(
        core,
        values,
        floor,
        exact_scale,
        algebraic_scale,
        coefficients,
        signed_positions,
        terminal_axis=checked_axis,
        plateau_start=plateau_start,
        plateau_count=plateau_count,
        zero_start=zero_start,
        zero_count=zero_count,
    )
    return absorber  # noqa: RET504


def _difference_evidence(
    state_indices: _HostStateIndices,
    absorber_indices: _HostAbsorberIndices,
) -> _DifferenceEvidence:
    """PRIVATE: Build exact Du, multiplicities, and pair-to-Ia map.

    Parameters
    ----------
    state_indices : _HostStateIndices
        Internal value used by this helper.
    absorber_indices : _HostAbsorberIndices
        Internal value used by this helper.

    Returns
    -------
    _return_value : _DifferenceEvidence
        Internal result produced by this helper.

    Raises
    ------
    AssertionError
        If an internal validation or arithmetic check fails.
    """
    absorber_modes = [_mode_tuple(row) for row in absorber_indices]
    absorber_position = {
        mode: position for position, mode in enumerate(absorber_modes)
    }
    pair_count = state_indices.shape[0] ** 2
    pair_positions = np.empty((pair_count,), dtype=np.int64)
    multiplicity_by_position: Dict[int, int] = {}
    flat_position = 0
    for left in state_indices:
        left_mode = _mode_tuple(left)
        for right in state_indices:
            right_mode = _mode_tuple(right)
            difference = (
                left_mode[0] - right_mode[0],
                left_mode[1] - right_mode[1],
                left_mode[2] - right_mode[2],
            )
            if any(
                value < np.iinfo(np.int64).min
                or value > np.iinfo(np.int64).max
                for value in difference
            ):
                _return_value: _DifferenceEvidence = None
                return _return_value
            position = absorber_position.get(difference)
            if position is None:
                _return_value: _DifferenceEvidence = None
                return _return_value
            pair_positions[flat_position] = position
            multiplicity_by_position[position] = (
                multiplicity_by_position.get(position, 0) + 1
            )
            flat_position += 1
    ordered_positions = sorted(
        multiplicity_by_position,
        key=lambda position: absorber_modes[position],
    )
    positions = np.asarray(ordered_positions, dtype=np.int64)
    differences = np.asarray(
        [absorber_modes[position] for position in ordered_positions],
        dtype=np.int64,
    )
    multiplicities = np.asarray(
        [multiplicity_by_position[position] for position in ordered_positions],
        dtype=np.int64,
    )
    if int(sum(int(value) for value in multiplicities)) != pair_count:
        raise AssertionError("absorber multiplicities do not sum to n squared")
    _return_value: _DifferenceEvidence = (
        differences,
        positions,
        multiplicities,
        pair_positions,
    )
    return _return_value


def _symbolic_exact_zero(
    mode: Tuple[int, int, int],
    terminal_axis: int,
    layer_count: int,
) -> bool:
    """PRIVATE: Identify LVT.24 transverse or nonzero sinc zeros.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.
    layer_count : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : bool
        Internal result produced by this helper.
    """
    transverse = any(
        component != 0
        for axis, component in enumerate(mode)
        if axis != terminal_axis
    )
    normal_mode = mode[terminal_axis]
    _return_value: bool = transverse or (
        normal_mode != 0 and normal_mode % layer_count == 0
    )
    return _return_value


def _direct_term_count(
    modes: list[Tuple[int, int, int]],
    terminal_axis: int,
    layer_count: int,
) -> int:
    """PRIVATE: Count canonical signed-mode layer terms before expansion.

    Parameters
    ----------
    modes : list[Tuple[int, int, int]]
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.
    layer_count : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.
    """
    canonical_count = sum(
        _canonical_signed_mode(mode)
        and not _symbolic_exact_zero(mode, terminal_axis, layer_count)
        for mode in modes
    )
    _return_value: int = canonical_count * layer_count
    return _return_value


def _exact_mode_rectangle(
    layer_values: Float64[NDArray, " l"],
    normal_mode: int,
    origin: float,
    length: float,
) -> ComplexRectangle:
    """PRIVATE: Enclose one canonical axial LVT.24 coefficient.

    Parameters
    ----------
    layer_values : Float64[NDArray, ' l']
        Internal value used by this helper.
    normal_mode : int
        Internal value used by this helper.
    origin : float
        Internal value used by this helper.
    length : float
        Internal value used by this helper.

    Returns
    -------
    _return_value : ComplexRectangle
        Internal result produced by this helper.
    """
    layer_count = layer_values.shape[0]
    terms: Iterable[ComplexRectangle] = (
        scale_complex_rectangle(
            rational_turn_exponential(
                Fraction(normal_mode * row, layer_count)
            ),
            fraction_from_float(float(value)),
        )
        for row, value in enumerate(layer_values)
    )
    series = pairwise_rectangle_sum(terms)
    sinc = normalized_sinc_integer_ratio(normal_mode, layer_count)
    shaped = complex_rectangle_multiply(
        series,
        (sinc[0], sinc[1], Fraction(0), Fraction(0)),
    )
    origin_turn = (
        Fraction(normal_mode)
        * fraction_from_float(origin)
        / fraction_from_float(length)
    )
    phased = complex_rectangle_multiply(
        shaped, rational_turn_exponential(origin_turn)
    )
    _return_value: ComplexRectangle = scale_complex_rectangle(
        phased, Fraction(1, layer_count)
    )
    return _return_value


def _exact_rectangles(
    layer_values: Float64[NDArray, " l"],
    modes: list[Tuple[int, int, int]],
    signed_positions: _HostIntVector,
    origin: float,
    length: float,
    terminal_axis: int,
) -> list[ComplexRectangle]:
    """PRIVATE: Enclose all ordered Ia coefficients with symbolic zeros.

    Parameters
    ----------
    layer_values : Float64[NDArray, ' l']
        Internal value used by this helper.
    modes : list[Tuple[int, int, int]]
        Internal value used by this helper.
    signed_positions : _HostIntVector
        Internal value used by this helper.
    origin : float
        Internal value used by this helper.
    length : float
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : list[ComplexRectangle]
        Internal result produced by this helper.
    """
    zero: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    rectangles = [zero for _ in modes]
    layer_count = layer_values.shape[0]
    for position, mode in enumerate(modes):
        if not _canonical_signed_mode(mode):
            continue
        if _symbolic_exact_zero(mode, terminal_axis, layer_count):
            rectangle = zero
        else:
            rectangle = _exact_mode_rectangle(
                layer_values,
                mode[terminal_axis],
                origin,
                length,
            )
        opposite_position = int(signed_positions[position])
        rectangles[position] = rectangle
        rectangles[opposite_position] = conjugate_rectangle(rectangle)
    _return_value: list[ComplexRectangle] = rectangles
    return _return_value


def _euclidean_error(
    coefficient: np.complex128,
    rectangle: ComplexRectangle,
) -> Fraction:
    """PRIVATE: Bound a complex point by the farthest rectangle corner.

    Parameters
    ----------
    coefficient : np.complex128
        Internal value used by this helper.
    rectangle : ComplexRectangle
        Internal value used by this helper.

    Returns
    -------
    _return_value : Fraction
        Internal result produced by this helper.
    """
    real = fraction_from_float(float(coefficient.real))
    imaginary = fraction_from_float(float(coefficient.imag))
    real_gap = max(abs(real - rectangle[0]), abs(real - rectangle[1]))
    imaginary_gap = max(
        abs(imaginary - rectangle[2]),
        abs(imaginary - rectangle[3]),
    )
    _return_value: Fraction = sqrt_fraction_upper(
        real_gap * real_gap + imaginary_gap * imaginary_gap
    )
    return _return_value


def _coefficient_evidence_payload(  # noqa: PLR0913
    absorber: GalerkinAxialCellAbsorber,
    real_lower: Float64[NDArray, " a"],
    real_upper: Float64[NDArray, " a"],
    imag_lower: Float64[NDArray, " a"],
    imag_upper: Float64[NDArray, " a"],
    errors: Float64[NDArray, " a"],
    differences: Int64[NDArray, "d 3"],
    positions: Int64[NDArray, " d"],
    multiplicities: Int64[NDArray, " d"],
    pair_positions: Int64[NDArray, " s"],
    operator_error: np.float64,
    term_count: int,
    pair_count: int,
    term_budget: int,
    pair_budget: int,
    failure: GalerkinAxialCapCoefficientFailure,
) -> Dict[str, object]:
    """PRIVATE: Build complete replayable coefficient-certificate evidence.

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Internal value used by this helper.
    real_lower : Float64[NDArray, ' a']
        Internal value used by this helper.
    real_upper : Float64[NDArray, ' a']
        Internal value used by this helper.
    imag_lower : Float64[NDArray, ' a']
        Internal value used by this helper.
    imag_upper : Float64[NDArray, ' a']
        Internal value used by this helper.
    errors : Float64[NDArray, ' a']
        Internal value used by this helper.
    differences : Int64[NDArray, 'd 3']
        Internal value used by this helper.
    positions : Int64[NDArray, ' d']
        Internal value used by this helper.
    multiplicities : Int64[NDArray, ' d']
        Internal value used by this helper.
    pair_positions : Int64[NDArray, ' s']
        Internal value used by this helper.
    operator_error : np.float64
        Internal value used by this helper.
    term_count : int
        Internal value used by this helper.
    pair_count : int
        Internal value used by this helper.
    term_budget : int
        Internal value used by this helper.
    pair_budget : int
        Internal value used by this helper.
    failure : GalerkinAxialCapCoefficientFailure
        Internal value used by this helper.

    Returns
    -------
    _return_value : Dict[str, object]
        Internal result produced by this helper.
    """
    _return_value: Dict[str, object] = {
        "parent_operator_digest": absorber.operator_digest,
        "source_digest": absorber.source_digest,
        "real_lower": stored_value_payload(real_lower),
        "real_upper": stored_value_payload(real_upper),
        "imag_lower": stored_value_payload(imag_lower),
        "imag_upper": stored_value_payload(imag_upper),
        "coefficient_errors": stored_value_payload(errors),
        "difference_indices": stored_value_payload(differences),
        "difference_absorber_positions": stored_value_payload(positions),
        "difference_multiplicities": stored_value_payload(multiplicities),
        "state_pair_absorber_positions": stored_value_payload(pair_positions),
        "absorber_operator_error_bound": stored_value_payload(operator_error),
        "finite_certificate": failure
        is GalerkinAxialCapCoefficientFailure.NONE,
        "direct_term_count": term_count,
        "state_pair_count": pair_count,
        "maximum_direct_terms": term_budget,
        "maximum_state_pairs": pair_budget,
        "failure": failure.value,
        "exact_target": _COEFFICIENT_EXACT_TARGET,
        "arithmetic": _ARITHMETIC,
        "coverage_claim": _COVERAGE_CLAIM,
        "operator_error_scope": _OPERATOR_ERROR_SCOPE,
        "per_call_arithmetic_exclusion": _PER_CALL_ARITHMETIC_EXCLUSION,
    }
    return _return_value


def _make_coefficient_result(  # noqa: PLR0913
    absorber: GalerkinAxialCellAbsorber,
    real_lower: Float64[NDArray, " a"],
    real_upper: Float64[NDArray, " a"],
    imag_lower: Float64[NDArray, " a"],
    imag_upper: Float64[NDArray, " a"],
    errors: Float64[NDArray, " a"],
    differences: Int64[NDArray, "d 3"],
    positions: Int64[NDArray, " d"],
    multiplicities: Int64[NDArray, " d"],
    pair_positions: Int64[NDArray, " s"],
    operator_error: np.float64,
    term_count: int,
    pair_count: int,
    term_budget: int,
    pair_budget: int,
    failure: GalerkinAxialCapCoefficientFailure,
) -> GalerkinAxialCapCoefficientCertificate:
    """PRIVATE: Store one coefficient certificate or typed noncertificate.

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Internal value used by this helper.
    real_lower : Float64[NDArray, ' a']
        Internal value used by this helper.
    real_upper : Float64[NDArray, ' a']
        Internal value used by this helper.
    imag_lower : Float64[NDArray, ' a']
        Internal value used by this helper.
    imag_upper : Float64[NDArray, ' a']
        Internal value used by this helper.
    errors : Float64[NDArray, ' a']
        Internal value used by this helper.
    differences : Int64[NDArray, 'd 3']
        Internal value used by this helper.
    positions : Int64[NDArray, ' d']
        Internal value used by this helper.
    multiplicities : Int64[NDArray, ' d']
        Internal value used by this helper.
    pair_positions : Int64[NDArray, ' s']
        Internal value used by this helper.
    operator_error : np.float64
        Internal value used by this helper.
    term_count : int
        Internal value used by this helper.
    pair_count : int
        Internal value used by this helper.
    term_budget : int
        Internal value used by this helper.
    pair_budget : int
        Internal value used by this helper.
    failure : GalerkinAxialCapCoefficientFailure
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapCoefficientCertificate
        Internal result produced by this helper.
    """
    evidence = _coefficient_evidence_payload(
        absorber,
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        errors,
        differences,
        positions,
        multiplicities,
        pair_positions,
        operator_error,
        term_count,
        pair_count,
        term_budget,
        pair_budget,
        failure,
    )
    digest = sha256({"domain": _CERTIFICATE_DOMAIN, "evidence": evidence})
    certificate = _make_axial_cap_coefficient_certificate(
        absorber,
        jnp.asarray(real_lower),
        jnp.asarray(real_upper),
        jnp.asarray(imag_lower),
        jnp.asarray(imag_upper),
        jnp.asarray(errors),
        jnp.asarray(differences),
        jnp.asarray(positions),
        jnp.asarray(multiplicities),
        jnp.asarray(pair_positions),
        jnp.asarray(operator_error),
        jnp.asarray(failure is GalerkinAxialCapCoefficientFailure.NONE),
        jnp.asarray(term_count, dtype=jnp.int64),
        jnp.asarray(pair_count, dtype=jnp.int64),
        jnp.asarray(term_budget, dtype=jnp.int64),
        jnp.asarray(pair_budget, dtype=jnp.int64),
        failure=failure,
        exact_target=_COEFFICIENT_EXACT_TARGET,
        arithmetic=_ARITHMETIC,
        coverage_claim=_COVERAGE_CLAIM,
        operator_error_scope=_OPERATOR_ERROR_SCOPE,
        per_call_arithmetic_exclusion=_PER_CALL_ARITHMETIC_EXCLUSION,
        parent_operator_digest=absorber.operator_digest,
        certificate_digest=digest,
    )
    jax.block_until_ready(certificate)
    _return_value: GalerkinAxialCapCoefficientCertificate = certificate
    return _return_value


def _failure_coefficient_result(
    absorber: GalerkinAxialCellAbsorber,
    term_count: int,
    pair_count: int,
    term_budget: int,
    pair_budget: int,
    failure: GalerkinAxialCapCoefficientFailure,
    evidence: _DifferenceEvidence = None,
) -> GalerkinAxialCapCoefficientCertificate:
    """PRIVATE: Store one all-infinite typed coefficient noncertificate.

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Internal value used by this helper.
    term_count : int
        Internal value used by this helper.
    pair_count : int
        Internal value used by this helper.
    term_budget : int
        Internal value used by this helper.
    pair_budget : int
        Internal value used by this helper.
    failure : GalerkinAxialCapCoefficientFailure
        Internal value used by this helper.
    evidence : _DifferenceEvidence
        Internal value used by this helper. Default is fixed by the signature.

    Returns
    -------
    _return_value : GalerkinAxialCapCoefficientCertificate
        Internal result produced by this helper.
    """
    absorber_count = absorber.support.absorber_indices.shape[0]
    lower = np.full((absorber_count,), -np.inf, dtype=np.float64)
    upper = np.full((absorber_count,), np.inf, dtype=np.float64)
    errors = np.full((absorber_count,), np.inf, dtype=np.float64)
    if evidence is None:
        differences = np.zeros((0, 3), dtype=np.int64)
        positions = np.zeros((0,), dtype=np.int64)
        multiplicities = np.zeros((0,), dtype=np.int64)
        pair_positions = np.zeros((0,), dtype=np.int64)
    else:
        differences, positions, multiplicities, pair_positions = evidence
    _return_value: GalerkinAxialCapCoefficientCertificate = (
        _make_coefficient_result(
            absorber,
            lower,
            upper,
            lower,
            upper,
            errors,
            differences,
            positions,
            multiplicities,
            pair_positions,
            np.float64(np.inf),
            term_count,
            pair_count,
            term_budget,
            pair_budget,
            failure,
        )
    )
    return _return_value


def _certify_axial_cell_absorber_impl(  # noqa: PLR0911
    submitted: GalerkinAxialCellAbsorber,
    *,
    maximum_direct_terms: int,
    maximum_state_pairs: int,
) -> GalerkinAxialCapCoefficientCertificate:
    """PRIVATE: Build LVT.24/LVT.26/LVT.31 evidence without recursion.

    Parameters
    ----------
    submitted : GalerkinAxialCellAbsorber
        Internal value used by this helper.
    maximum_direct_terms : int
        Internal value used by this helper.
    maximum_state_pairs : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapCoefficientCertificate
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    absorber = _canonical_absorber(submitted)
    indices = _checked_indices(
        absorber.support.absorber_indices, "absorber_indices"
    )
    state = _checked_indices(absorber.support.state_indices, "state_indices")
    modes, signed_positions = _signed_position_map(indices)
    coefficients = _checked_coefficients(
        absorber.absorber_coefficients, modes, signed_positions
    )
    layer_values = _checked_layer_values(absorber.layer_values)
    term_count = _direct_term_count(
        modes, absorber.terminal_axis, layer_values.shape[0]
    )
    pair_count = state.shape[0] ** 2
    if (
        term_count > _MAXIMUM_SIGNED_INT64
        or pair_count > _MAXIMUM_SIGNED_INT64
    ):
        raise ValueError(
            "exact axial CAP work counts must fit signed int64 storage"
        )
    if term_count > maximum_direct_terms:
        _return_value: GalerkinAxialCapCoefficientCertificate = (
            _failure_coefficient_result(
                absorber,
                term_count,
                pair_count,
                maximum_direct_terms,
                maximum_state_pairs,
                GalerkinAxialCapCoefficientFailure.DIRECT_TERM_BUDGET_EXCEEDED,
            )
        )
        return _return_value
    if pair_count > maximum_state_pairs:
        _return_value: GalerkinAxialCapCoefficientCertificate = (
            _failure_coefficient_result(
                absorber,
                term_count,
                pair_count,
                maximum_direct_terms,
                maximum_state_pairs,
                GalerkinAxialCapCoefficientFailure.STATE_PAIR_BUDGET_EXCEEDED,
            )
        )
        return _return_value
    evidence = _difference_evidence(state, indices)
    if evidence is None:
        _return_value: GalerkinAxialCapCoefficientCertificate = (
            _failure_coefficient_result(
                absorber,
                term_count,
                pair_count,
                maximum_direct_terms,
                maximum_state_pairs,
                GalerkinAxialCapCoefficientFailure.DIFFERENCE_COVERAGE_MISSING,
            )
        )
        return _return_value
    if not host_binary64_supported():
        _return_value: GalerkinAxialCapCoefficientCertificate = (
            _failure_coefficient_result(
                absorber,
                term_count,
                pair_count,
                maximum_direct_terms,
                maximum_state_pairs,
                GalerkinAxialCapCoefficientFailure.HOST_ARITHMETIC_UNSUPPORTED,
                evidence,
            )
        )
        return _return_value
    local_potential = (
        absorber.interaction_core.compression.realization.local_potential
    )
    origin = float(local_potential.cell_center_origin[absorber.terminal_axis])
    length = float(local_potential.box_size[absorber.terminal_axis])
    try:
        rectangles = _exact_rectangles(
            layer_values,
            modes,
            signed_positions,
            origin,
            length,
            absorber.terminal_axis,
        )
    except RootEnclosureError:
        _return_value: GalerkinAxialCapCoefficientCertificate = (
            _failure_coefficient_result(
                absorber,
                term_count,
                pair_count,
                maximum_direct_terms,
                maximum_state_pairs,
                GalerkinAxialCapCoefficientFailure.ROOT_ENCLOSURE_FAILURE,
                evidence,
            )
        )
        return _return_value
    error_fractions = [
        _euclidean_error(coefficient, rectangle)
        for coefficient, rectangle in zip(
            coefficients, rectangles, strict=True
        )
    ]
    real_lower = np.asarray(
        [fraction_lower_float(rectangle[0]) for rectangle in rectangles],
        dtype=np.float64,
    )
    real_upper = np.asarray(
        [fraction_upper_float(rectangle[1]) for rectangle in rectangles],
        dtype=np.float64,
    )
    imag_lower = np.asarray(
        [fraction_lower_float(rectangle[2]) for rectangle in rectangles],
        dtype=np.float64,
    )
    imag_upper = np.asarray(
        [fraction_upper_float(rectangle[3]) for rectangle in rectangles],
        dtype=np.float64,
    )
    errors = np.asarray(
        [fraction_upper_float(error) for error in error_fractions],
        dtype=np.float64,
    )
    differences, positions, multiplicities, pair_positions = evidence
    radicand: Fraction = sum(
        (
            int(multiplicity)
            * fraction_from_float(float(errors[int(position)])) ** 2
            for position, multiplicity in zip(
                positions, multiplicities, strict=True
            )
        ),
        start=Fraction(0),
    )
    operator_error_fraction = sqrt_fraction_upper(radicand)
    operator_error = np.float64(fraction_upper_float(operator_error_fraction))
    endpoint_arrays = (real_lower, real_upper, imag_lower, imag_upper, errors)
    if not all(
        np.all(np.isfinite(value)) for value in endpoint_arrays
    ) or not np.isfinite(operator_error):
        _return_value: GalerkinAxialCapCoefficientCertificate = (
            _failure_coefficient_result(
                absorber,
                term_count,
                pair_count,
                maximum_direct_terms,
                maximum_state_pairs,
                GalerkinAxialCapCoefficientFailure.ARITHMETIC_RANGE_FAILURE,
                evidence,
            )
        )
        return _return_value
    _return_value: GalerkinAxialCapCoefficientCertificate = (
        _make_coefficient_result(
            absorber,
            real_lower,
            real_upper,
            imag_lower,
            imag_upper,
            errors,
            differences,
            positions,
            multiplicities,
            pair_positions,
            operator_error,
            term_count,
            pair_count,
            maximum_direct_terms,
            maximum_state_pairs,
            GalerkinAxialCapCoefficientFailure.NONE,
        )
    )
    return _return_value


def _certificate_budgets(
    certificate: GalerkinAxialCapCoefficientCertificate,
) -> Tuple[int, int]:
    """PRIVATE: Read exact stored coefficient-certificate budgets.

    Parameters
    ----------
    certificate : GalerkinAxialCapCoefficientCertificate
        Internal value used by this helper.

    Returns
    -------
    result_0 : int
        Internal result produced by this helper.
    result_1 : int
        Internal result produced by this helper.
    """
    term_budget = int(
        _host_exact_scalar(
            certificate.maximum_direct_terms,
            np.dtype(np.int64),
            "maximum_direct_terms",
        )
    )
    pair_budget = int(
        _host_exact_scalar(
            certificate.maximum_state_pairs,
            np.dtype(np.int64),
            "maximum_state_pairs",
        )
    )
    _return_value: Tuple[int, int] = (
        _checked_budget(term_budget, "maximum_direct_terms"),
        _checked_budget(pair_budget, "maximum_state_pairs"),
    )
    return _return_value


def _authenticate_coefficient_certificate(
    certificate: GalerkinAxialCapCoefficientCertificate,
) -> GalerkinAxialCapCoefficientCertificate:
    """PRIVATE: Fully replay one public LVT.24/LVT.31 certificate.

    Parameters
    ----------
    certificate : GalerkinAxialCapCoefficientCertificate
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapCoefficientCertificate
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    _assert_concrete(certificate)
    term_budget, pair_budget = _certificate_budgets(certificate)
    canonical = _certify_axial_cell_absorber_impl(
        certificate.absorber,
        maximum_direct_terms=term_budget,
        maximum_state_pairs=pair_budget,
    )
    if stored_value_payload(canonical) != stored_value_payload(certificate):
        raise ValueError(
            "axial CAP coefficient certificate does not match full host replay"
        )
    _return_value: GalerkinAxialCapCoefficientCertificate = canonical
    return _return_value


@jaxtyped(typechecker=beartype)
def certify_axial_cell_absorber(
    absorber: GalerkinAxialCellAbsorber,
    *,
    maximum_direct_terms: int = _DEFAULT_MAXIMUM_DIRECT_TERMS,
    maximum_state_pairs: int = _DEFAULT_MAXIMUM_STATE_PAIRS,
) -> GalerkinAxialCapCoefficientCertificate:
    """Certify a finite Hermitian approximant against exact LVT.24.

    :see: :func:`~.test_absorber.\
test_lvt24_rectangles_du_mapping_and_lvt31_transfer`

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Concrete axial profile and actual complex128 Hermitian approximant.
        The coefficient bytes need not originate from the realization helper.
    maximum_direct_terms : int, optional
        Maximum canonical-mode--layer terms. Default: ``2_000_000``.
    maximum_state_pairs : int, optional
        Maximum ordered state pairs. Default: ``20_000_000``.

    Returns
    -------
    certificate : GalerkinAxialCapCoefficientCertificate
        Finite direct rectangles, exact Du maps, and LVT.31 error, or one
        typed all-infinite noncertificate.

    Raises
    ------
    ValueError
        If budgets, nested replay, primitive storage, or Hermitian symmetry
        fail.

    Notes
    -----
    This certifies the submitted exact-Hermitian approximation point and does
    not claim FFT or realization-helper provenance.
    """
    term_budget = _checked_budget(maximum_direct_terms, "maximum_direct_terms")
    pair_budget = _checked_budget(maximum_state_pairs, "maximum_state_pairs")
    certificate: GalerkinAxialCapCoefficientCertificate = (
        _certify_axial_cell_absorber_impl(
            absorber,
            maximum_direct_terms=term_budget,
            maximum_state_pairs=pair_budget,
        )
    )
    return certificate  # noqa: RET504


def _complex_add(left: _ComplexQ, right: _ComplexQ) -> _ComplexQ:
    """PRIVATE: Add two exact-rational complex pairs.

    Parameters
    ----------
    left : _ComplexQ
        Internal value used by this helper.
    right : _ComplexQ
        Internal value used by this helper.

    Returns
    -------
    _return_value : _ComplexQ
        Internal result produced by this helper.
    """
    _return_value: _ComplexQ = (left[0] + right[0], left[1] + right[1])
    return _return_value


def _complex_subtract(left: _ComplexQ, right: _ComplexQ) -> _ComplexQ:
    """PRIVATE: Subtract two exact-rational complex pairs.

    Parameters
    ----------
    left : _ComplexQ
        Internal value used by this helper.
    right : _ComplexQ
        Internal value used by this helper.

    Returns
    -------
    _return_value : _ComplexQ
        Internal result produced by this helper.
    """
    _return_value: _ComplexQ = (left[0] - right[0], left[1] - right[1])
    return _return_value


def _complex_multiply(left: _ComplexQ, right: _ComplexQ) -> _ComplexQ:
    """PRIVATE: Multiply two exact-rational complex pairs.

    Parameters
    ----------
    left : _ComplexQ
        Internal value used by this helper.
    right : _ComplexQ
        Internal value used by this helper.

    Returns
    -------
    _return_value : _ComplexQ
        Internal result produced by this helper.
    """
    _return_value: _ComplexQ = (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )
    return _return_value


def _complex_conjugate(value: _ComplexQ) -> _ComplexQ:
    """PRIVATE: Conjugate one exact-rational complex pair.

    Parameters
    ----------
    value : _ComplexQ
        Internal value used by this helper.

    Returns
    -------
    _return_value : _ComplexQ
        Internal result produced by this helper.
    """
    _return_value: _ComplexQ = (value[0], -value[1])
    return _return_value


def _complex_scale(value: _ComplexQ, scalar: Fraction) -> _ComplexQ:
    """PRIVATE: Scale one exact-rational complex pair.

    Parameters
    ----------
    value : _ComplexQ
        Internal value used by this helper.
    scalar : Fraction
        Internal value used by this helper.

    Returns
    -------
    _return_value : _ComplexQ
        Internal result produced by this helper.
    """
    _return_value: _ComplexQ = (value[0] * scalar, value[1] * scalar)
    return _return_value


def _arctangent_interval(
    inverse: int,
    terms: int,
) -> Tuple[Fraction, Fraction]:
    """PRIVATE: Enclose ``atan(1/inverse)`` by alternating sums.

    Parameters
    ----------
    inverse : int
        Internal value used by this helper.
    terms : int
        Internal value used by this helper.

    Returns
    -------
    result_0 : Fraction
        Internal result produced by this helper.
    result_1 : Fraction
        Internal result produced by this helper.
    """
    total = Fraction(0)
    previous = total
    for term_index in range(terms):
        previous = total
        term = Fraction(
            1,
            (2 * term_index + 1) * inverse ** (2 * term_index + 1),
        )
        total = total + term if term_index % 2 == 0 else total - term
    _return_value: Tuple[Fraction, Fraction] = (
        min(previous, total),
        max(previous, total),
    )
    return _return_value


def _pi_interval(terms: int) -> Tuple[Fraction, Fraction]:
    """PRIVATE: Enclose mathematical pi through Machin's identity.

    Parameters
    ----------
    terms : int
        Internal value used by this helper.

    Returns
    -------
    result_0 : Fraction
        Internal result produced by this helper.
    result_1 : Fraction
        Internal result produced by this helper.
    """
    first_lower, first_upper = _arctangent_interval(5, terms)
    second_lower, second_upper = _arctangent_interval(239, terms)
    _return_value: Tuple[Fraction, Fraction] = (
        16 * first_lower - 4 * second_upper,
        16 * first_upper - 4 * second_lower,
    )
    return _return_value


def _first_quadrant_sine_cosine(
    turns: Fraction,
    pi_bounds: Tuple[Fraction, Fraction],
    terms: int,
) -> Tuple[Tuple[Fraction, Fraction], Tuple[Fraction, Fraction]]:
    """PRIVATE: Enclose sine and cosine on exact turns in ``[0, 1/4]``.

    Parameters
    ----------
    turns : Fraction
        Internal value used by this helper.
    pi_bounds : Tuple[Fraction, Fraction]
        Internal value used by this helper.
    terms : int
        Internal value used by this helper.

    Returns
    -------
    result_0 : Tuple[Fraction, Fraction]
        Internal result produced by this helper.
    result_1 : Tuple[Fraction, Fraction]
        Internal result produced by this helper.
    """
    pi_lower, pi_upper = pi_bounds
    angle_lower = 2 * turns * pi_lower
    angle_upper = 2 * turns * pi_upper
    sine_lower = Fraction(0)
    sine_upper = Fraction(0)
    cosine_lower = Fraction(0)
    cosine_upper = Fraction(0)
    for term_index in range(terms):
        sine_denominator = math.factorial(2 * term_index + 1)
        cosine_denominator = math.factorial(2 * term_index)
        if term_index % 2 == 0:
            sine_lower += (
                angle_lower ** (2 * term_index + 1) / sine_denominator
            )
            sine_upper += (
                angle_upper ** (2 * term_index + 1) / sine_denominator
            )
            cosine_lower += (
                angle_lower ** (2 * term_index) / cosine_denominator
            )
            cosine_upper += (
                angle_upper ** (2 * term_index) / cosine_denominator
            )
        else:
            sine_lower -= (
                angle_upper ** (2 * term_index + 1) / sine_denominator
            )
            sine_upper -= (
                angle_lower ** (2 * term_index + 1) / sine_denominator
            )
            cosine_lower -= (
                angle_upper ** (2 * term_index) / cosine_denominator
            )
            cosine_upper -= (
                angle_lower ** (2 * term_index) / cosine_denominator
            )
    term_index = terms
    sine_remainder = angle_upper ** (2 * term_index + 1) / math.factorial(
        2 * term_index + 1
    )
    cosine_remainder = angle_upper ** (2 * term_index) / math.factorial(
        2 * term_index
    )
    if term_index % 2 == 0:
        sine_upper += sine_remainder
        cosine_upper += cosine_remainder
    else:
        sine_lower -= sine_remainder
        cosine_lower -= cosine_remainder
    _return_value: Tuple[
        Tuple[Fraction, Fraction], Tuple[Fraction, Fraction]
    ] = (
        (sine_lower, sine_upper),
        (cosine_lower, cosine_upper),
    )
    return _return_value


def _sine_cosine_turns(
    turns: Fraction,
    pi_bounds: Tuple[Fraction, Fraction],
    terms: int,
) -> _FractionIntervalPair:
    """PRIVATE: Enclose sine and cosine for exact rational turns.

    Parameters
    ----------
    turns : Fraction
        Internal value used by this helper.
    pi_bounds : Tuple[Fraction, Fraction]
        Internal value used by this helper.
    terms : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : _FractionIntervalPair
        Internal result produced by this helper.
    """
    reduced = turns % 1
    quadrant = int(reduced * 4)
    offset = reduced - Fraction(quadrant, 4)
    sine, cosine = _first_quadrant_sine_cosine(offset, pi_bounds, terms)
    if quadrant == 0:
        _return_value: _FractionIntervalPair = sine, cosine
        return _return_value
    if quadrant == 1:
        _return_value: _FractionIntervalPair = cosine, (-sine[1], -sine[0])
        return _return_value
    if quadrant == 2:  # noqa: PLR2004
        _return_value: _FractionIntervalPair = (
            (-sine[1], -sine[0]),
            (
                -cosine[1],
                -cosine[0],
            ),
        )
        return _return_value
    _return_value: _FractionIntervalPair = (-cosine[1], -cosine[0]), sine
    return _return_value


def _upper_square_root(value: Fraction, bits: int) -> Fraction:
    """PRIVATE: Bound a rational square root above on a dyadic grid.

    Parameters
    ----------
    value : Fraction
        Internal value used by this helper.
    bits : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : Fraction
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    if value < 0:
        raise ValueError("square-root input must be nonnegative")
    if value == 0:
        _return_value: Fraction = Fraction(0)
        return _return_value
    scaled_numerator = value.numerator << (2 * bits)
    quotient = scaled_numerator // value.denominator
    root = math.isqrt(quotient)
    if root * root * value.denominator < scaled_numerator:
        root += 1
    _return_value: Fraction = Fraction(root, 1 << bits)
    return _return_value


def _nearest_dyadic(value: Fraction, bits: int) -> Fraction:
    """PRIVATE: Round a rational to nearest on one exact dyadic grid.

    Parameters
    ----------
    value : Fraction
        Internal value used by this helper.
    bits : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : Fraction
        Internal result produced by this helper.
    """
    scale = 1 << bits
    scaled = value * scale
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder >= scaled.denominator:
        quotient += 1
    _return_value: Fraction = Fraction(quotient, scale)
    return _return_value


def _positive_definite_shift(
    matrix: list[list[_ComplexQ]],
    shift: Fraction,
) -> list[Fraction] | None:
    """PRIVATE: Prove ``matrix - shift I`` positive by exact LDL-star.

    Parameters
    ----------
    matrix : list[list[_ComplexQ]]
        Internal value used by this helper.
    shift : Fraction
        Internal value used by this helper.

    Returns
    -------
    _return_value : list[Fraction] | None
        Internal result produced by this helper.
    """
    size = len(matrix)
    lower = [
        [(Fraction(0), Fraction(0)) for _ in range(size)] for _ in range(size)
    ]
    diagonal: list[Fraction] = []
    for row in range(size):
        if matrix[row][row][1] != 0:
            _return_value: list[Fraction] | None = None
            return _return_value
        pivot = matrix[row][row][0] - shift
        for previous in range(row):
            pivot -= diagonal[previous] * (
                lower[row][previous][0] ** 2 + lower[row][previous][1] ** 2
            )
        if pivot <= 0:
            _return_value: list[Fraction] | None = None
            return _return_value
        diagonal.append(pivot)
        lower[row][row] = (Fraction(1), Fraction(0))
        for next_row in range(row + 1, size):
            value = matrix[next_row][row]
            for previous in range(row):
                product = _complex_multiply(
                    lower[next_row][previous],
                    _complex_conjugate(lower[row][previous]),
                )
                value = _complex_subtract(
                    value,
                    _complex_scale(product, diagonal[previous]),
                )
            lower[next_row][row] = _complex_scale(value, 1 / pivot)
    _return_value: list[Fraction] | None = diagonal
    return _return_value


def _fraction_payload(value: Fraction) -> Dict[str, str]:
    """PRIVATE: Serialize one exact rational canonically.

    Parameters
    ----------
    value : Fraction
        Internal value used by this helper.

    Returns
    -------
    _return_value : Dict[str, str]
        Internal result produced by this helper.
    """
    _return_value: Dict[str, str] = {
        "numerator_hex": hex(value.numerator),
        "denominator_hex": hex(value.denominator),
    }
    return _return_value


def _complex_fraction_payload(value: _ComplexQ) -> Dict[str, object]:
    """PRIVATE: Serialize one exact-rational complex pair.

    Parameters
    ----------
    value : _ComplexQ
        Internal value used by this helper.

    Returns
    -------
    _return_value : Dict[str, object]
        Internal result produced by this helper.
    """
    _return_value: Dict[str, object] = {
        "real": _fraction_payload(value[0]),
        "imag": _fraction_payload(value[1]),
    }
    return _return_value


def _gram_work_count(
    degree: int,
    precision_bits: int,
    ldl_iterations: int,
) -> int:
    """PRIVATE: Count versioned abstract exact-Gram host work units.

    Parameters
    ----------
    degree : int
        Internal value used by this helper.
    precision_bits : int
        Internal value used by this helper.
    ldl_iterations : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.
    """
    size = degree + 1
    upper_entries = size * (size - 1) // 2
    trig_units = upper_entries * (4 * precision_bits + 24)
    ldl_units_per_trial = size**3 + size**2 + size
    ldl_units = (ldl_iterations + 2) * ldl_units_per_trial
    midpoint_units = 12 * upper_entries + size
    _return_value: int = trig_units + ldl_units + midpoint_units
    return _return_value


def _gram_attempt(  # noqa: PLR0915
    degree: int,
    delta: Fraction,
    precision_bits: int,
    ldl_iterations: int,
    maximum_gram_degree: int,
    maximum_gram_work: int,
) -> _GramAttempt:
    """PRIVATE: Attempt the exact LDL-star/Frobenius-Weyl Gram proof.

    Parameters
    ----------
    degree : int
        Internal value used by this helper.
    delta : Fraction
        Internal value used by this helper.
    precision_bits : int
        Internal value used by this helper.
    ldl_iterations : int
        Internal value used by this helper.
    maximum_gram_degree : int
        Internal value used by this helper.
    maximum_gram_work : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : _GramAttempt
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    AssertionError
        If an internal validation or arithmetic check fails.
    """
    work_count = _gram_work_count(degree, precision_bits, ldl_iterations)
    if work_count > _MAXIMUM_SIGNED_INT64:
        raise ValueError("exact gram_work_count must fit signed int64 storage")
    base_payload: Dict[str, object] = {
        "degree": degree,
        "delta": _fraction_payload(delta),
        "gram_subinterval_width": stored_value_payload(
            np.float64(fraction_lower_float(delta))
        ),
        "precision_bits": precision_bits,
        "ldl_iterations": ldl_iterations,
        "gram_work_count": work_count,
        "maximum_gram_degree": maximum_gram_degree,
        "maximum_gram_work": maximum_gram_work,
        "gram_proof_route": _GRAM_PROOF_ROUTE,
        "gram_work_scope": _GRAM_WORK_SCOPE,
    }
    if degree > maximum_gram_degree:
        failure = GalerkinAxialCapExactFloorFailure.GRAM_DEGREE_BUDGET_EXCEEDED
        payload = {**base_payload, "failure": failure.value}
        _return_value: _GramAttempt = {
            "failure": failure,
            "work_count": work_count,
            "midpoint_shift": None,
            "entry_error": None,
            "gram_lower": None,
            "transcript_digest": sha256(
                {"domain": _GRAM_TRANSCRIPT_DOMAIN, "transcript": payload}
            ),
        }
        return _return_value
    if work_count > maximum_gram_work:
        failure = GalerkinAxialCapExactFloorFailure.GRAM_WORK_BUDGET_EXCEEDED
        payload = {**base_payload, "failure": failure.value}
        _return_value: _GramAttempt = {
            "failure": failure,
            "work_count": work_count,
            "midpoint_shift": None,
            "entry_error": None,
            "gram_lower": None,
            "transcript_digest": sha256(
                {"domain": _GRAM_TRANSCRIPT_DOMAIN, "transcript": payload}
            ),
        }
        return _return_value
    if not host_binary64_supported():
        failure = GalerkinAxialCapExactFloorFailure.HOST_ARITHMETIC_UNSUPPORTED
        payload = {**base_payload, "failure": failure.value}
        _return_value: _GramAttempt = {
            "failure": failure,
            "work_count": work_count,
            "midpoint_shift": None,
            "entry_error": None,
            "gram_lower": None,
            "transcript_digest": sha256(
                {"domain": _GRAM_TRANSCRIPT_DOMAIN, "transcript": payload}
            ),
        }
        return _return_value
    size = degree + 1
    midpoint = [
        [(Fraction(0), Fraction(0)) for _ in range(size)] for _ in range(size)
    ]
    rectangles: list[Dict[str, object]] = []
    error_squared = Fraction(0)
    try:
        pi_bounds = _pi_interval(precision_bits)
        for row in range(size):
            midpoint[row][row] = (delta, Fraction(0))
            for column in range(row + 1, size):
                separation = column - row
                sine, cosine = _sine_cosine_turns(
                    separation * delta,
                    pi_bounds,
                    precision_bits,
                )
                denominator_lower = 2 * separation * pi_bounds[0]
                denominator_upper = 2 * separation * pi_bounds[1]
                real_candidates = [
                    sine[endpoint] / denominator
                    for endpoint in (0, 1)
                    for denominator in (denominator_lower, denominator_upper)
                ]
                one_minus_cosine = (1 - cosine[1], 1 - cosine[0])
                imag_candidates = [
                    one_minus_cosine[endpoint] / denominator
                    for endpoint in (0, 1)
                    for denominator in (denominator_lower, denominator_upper)
                ]
                real_interval = (min(real_candidates), max(real_candidates))
                imag_interval = (min(imag_candidates), max(imag_candidates))
                real_center = sum(real_interval, Fraction(0)) / 2
                imag_center = sum(imag_interval, Fraction(0)) / 2
                real_midpoint = _nearest_dyadic(real_center, precision_bits)
                imag_midpoint = _nearest_dyadic(imag_center, precision_bits)
                midpoint[row][column] = (real_midpoint, imag_midpoint)
                midpoint[column][row] = (real_midpoint, -imag_midpoint)
                real_radius = (real_interval[1] - real_interval[0]) / 2 + abs(
                    real_midpoint - real_center
                )
                imag_radius = (imag_interval[1] - imag_interval[0]) / 2 + abs(
                    imag_midpoint - imag_center
                )
                error_squared += 2 * (
                    real_radius * real_radius + imag_radius * imag_radius
                )
                rectangles.append(
                    {
                        "row": row,
                        "column": column,
                        "real_lower": _fraction_payload(real_interval[0]),
                        "real_upper": _fraction_payload(real_interval[1]),
                        "imag_lower": _fraction_payload(imag_interval[0]),
                        "imag_upper": _fraction_payload(imag_interval[1]),
                        "midpoint": _complex_fraction_payload(
                            midpoint[row][column]
                        ),
                    }
                )
        entry_error = _upper_square_root(error_squared, precision_bits + 16)
        if _positive_definite_shift(midpoint, Fraction(0)) is None:
            failure = GalerkinAxialCapExactFloorFailure.GRAM_ARITHMETIC_FAILURE
            payload = {
                **base_payload,
                "failure": failure.value,
                "rectangles": rectangles,
                "entry_error": _fraction_payload(entry_error),
            }
            _return_value: _GramAttempt = {
                "failure": failure,
                "work_count": work_count,
                "midpoint_shift": None,
                "entry_error": entry_error,
                "gram_lower": None,
                "transcript_digest": sha256(
                    {"domain": _GRAM_TRANSCRIPT_DOMAIN, "transcript": payload}
                ),
            }
            return _return_value
        lower_shift = Fraction(0)
        upper_shift = delta
        for _ in range(ldl_iterations):
            trial = (lower_shift + upper_shift) / 2
            if _positive_definite_shift(midpoint, trial) is not None:
                lower_shift = trial
            else:
                upper_shift = trial
        pivots = _positive_definite_shift(midpoint, lower_shift)
        if pivots is None:
            raise AssertionError(
                "accepted exact LDL-star shift lost positivity"
            )
        gram_lower = lower_shift - entry_error
    except (ArithmeticError, OverflowError, ValueError):
        failure = GalerkinAxialCapExactFloorFailure.GRAM_ARITHMETIC_FAILURE
        payload = {**base_payload, "failure": failure.value}
        _return_value: _GramAttempt = {
            "failure": failure,
            "work_count": work_count,
            "midpoint_shift": None,
            "entry_error": None,
            "gram_lower": None,
            "transcript_digest": sha256(
                {"domain": _GRAM_TRANSCRIPT_DOMAIN, "transcript": payload}
            ),
        }
        return _return_value
    failure = (
        GalerkinAxialCapExactFloorFailure.NONE
        if gram_lower > 0
        else GalerkinAxialCapExactFloorFailure.GRAM_LOWER_BOUND_NONPOSITIVE
    )
    payload = {
        **base_payload,
        "failure": failure.value,
        "rectangles": rectangles,
        "midpoint_shift_lower": _fraction_payload(lower_shift),
        "entry_frobenius_error_upper": _fraction_payload(entry_error),
        "plateau_gram_lower": _fraction_payload(gram_lower),
        "final_ldl_pivots": [_fraction_payload(pivot) for pivot in pivots],
        "weyl_identity": "lambda_min(G)>=lambda_min(M)-||G-M||_F",
    }
    _return_value: _GramAttempt = {
        "failure": failure,
        "work_count": work_count,
        "midpoint_shift": lower_shift,
        "entry_error": entry_error,
        "gram_lower": gram_lower,
        "transcript_digest": sha256(
            {"domain": _GRAM_TRANSCRIPT_DOMAIN, "transcript": payload}
        ),
    }
    return _return_value


def _floor_evidence_payload(  # noqa: PLR0913
    certificate: GalerkinAxialCapCoefficientCertificate,
    degree: int,
    delta: Fraction,
    midpoint_shift: np.float64,
    entry_error: np.float64,
    gram_lower: np.float64,
    exact_dimensionless: np.float64,
    exact_physical: np.float64,
    realized_dimensionless: np.float64,
    scale_error: np.float64,
    physical_error: np.float64,
    realized_physical: np.float64,
    maximum_gram_degree: int,
    precision_bits: int,
    ldl_iterations: int,
    work_count: int,
    maximum_gram_work: int,
    exact_failure: GalerkinAxialCapExactFloorFailure,
    realized_failure: GalerkinAxialCapRealizedFloorFailure,
    route: GalerkinAxialCapRealizedFloorRoute,
    transcript_digest: str,
) -> Dict[str, object]:
    """PRIVATE: Build complete replayable LVT.29--LVT.32 evidence.

    Parameters
    ----------
    certificate : GalerkinAxialCapCoefficientCertificate
        Internal value used by this helper.
    degree : int
        Internal value used by this helper.
    delta : Fraction
        Internal value used by this helper.
    midpoint_shift : np.float64
        Internal value used by this helper.
    entry_error : np.float64
        Internal value used by this helper.
    gram_lower : np.float64
        Internal value used by this helper.
    exact_dimensionless : np.float64
        Internal value used by this helper.
    exact_physical : np.float64
        Internal value used by this helper.
    realized_dimensionless : np.float64
        Internal value used by this helper.
    scale_error : np.float64
        Internal value used by this helper.
    physical_error : np.float64
        Internal value used by this helper.
    realized_physical : np.float64
        Internal value used by this helper.
    maximum_gram_degree : int
        Internal value used by this helper.
    precision_bits : int
        Internal value used by this helper.
    ldl_iterations : int
        Internal value used by this helper.
    work_count : int
        Internal value used by this helper.
    maximum_gram_work : int
        Internal value used by this helper.
    exact_failure : GalerkinAxialCapExactFloorFailure
        Internal value used by this helper.
    realized_failure : GalerkinAxialCapRealizedFloorFailure
        Internal value used by this helper.
    route : GalerkinAxialCapRealizedFloorRoute
        Internal value used by this helper.
    transcript_digest : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : Dict[str, object]
        Internal result produced by this helper.
    """
    _return_value: Dict[str, object] = {
        "parent_certificate_digest": certificate.certificate_digest,
        "degree": degree,
        "delta": _fraction_payload(delta),
        "gram_midpoint_shift_lower_bound": stored_value_payload(
            midpoint_shift
        ),
        "gram_entry_frobenius_error_upper_bound": stored_value_payload(
            entry_error
        ),
        "plateau_gram_lower_bound": stored_value_payload(gram_lower),
        "dimensionless_exact_floor_lower_bound": stored_value_payload(
            exact_dimensionless
        ),
        "exact_target_physical_floor_lower_bound": stored_value_payload(
            exact_physical
        ),
        "realized_dimensionless_floor_lower_bound": stored_value_payload(
            realized_dimensionless
        ),
        "scale_error_bound": stored_value_payload(scale_error),
        "physical_operator_error_upper_bound": stored_value_payload(
            physical_error
        ),
        "realized_physical_floor_lower_bound": stored_value_payload(
            realized_physical
        ),
        "exact_target_floor_eligible": (
            exact_failure is GalerkinAxialCapExactFloorFailure.NONE
        ),
        "realized_floor_eligible": (
            exact_failure is GalerkinAxialCapExactFloorFailure.NONE
            and realized_failure is GalerkinAxialCapRealizedFloorFailure.NONE
        ),
        "maximum_gram_degree": maximum_gram_degree,
        "gram_precision_bits": precision_bits,
        "ldl_iteration_count": ldl_iterations,
        "gram_work_count": work_count,
        "maximum_gram_work": maximum_gram_work,
        "exact_target_failure": exact_failure.value,
        "realized_floor_failure": realized_failure.value,
        "realized_floor_route": route.value,
        "gram_transcript_digest": transcript_digest,
        "exact_floor_target": _EXACT_FLOOR_TARGET,
        "gram_proof_route": _GRAM_PROOF_ROUTE,
        "gram_work_scope": _GRAM_WORK_SCOPE,
        "realized_floor_scope": _REALIZED_FLOOR_SCOPE,
        "completion_scope": _ABSORBER_COMPLETION_SCOPE,
    }
    return _return_value


def _make_floor_result(  # noqa: PLR0913
    certificate: GalerkinAxialCapCoefficientCertificate,
    degree: int,
    delta: Fraction,
    midpoint_shift: np.float64,
    entry_error: np.float64,
    gram_lower: np.float64,
    exact_dimensionless: np.float64,
    exact_physical: np.float64,
    realized_dimensionless: np.float64,
    scale_error: np.float64,
    physical_error: np.float64,
    realized_physical: np.float64,
    maximum_gram_degree: int,
    precision_bits: int,
    ldl_iterations: int,
    work_count: int,
    maximum_gram_work: int,
    exact_failure: GalerkinAxialCapExactFloorFailure,
    realized_failure: GalerkinAxialCapRealizedFloorFailure,
    route: GalerkinAxialCapRealizedFloorRoute,
    transcript_digest: str,
) -> GalerkinAxialCapFloorProof:
    """PRIVATE: Store one exact/realized floor proof attempt.

    Parameters
    ----------
    certificate : GalerkinAxialCapCoefficientCertificate
        Internal value used by this helper.
    degree : int
        Internal value used by this helper.
    delta : Fraction
        Internal value used by this helper.
    midpoint_shift : np.float64
        Internal value used by this helper.
    entry_error : np.float64
        Internal value used by this helper.
    gram_lower : np.float64
        Internal value used by this helper.
    exact_dimensionless : np.float64
        Internal value used by this helper.
    exact_physical : np.float64
        Internal value used by this helper.
    realized_dimensionless : np.float64
        Internal value used by this helper.
    scale_error : np.float64
        Internal value used by this helper.
    physical_error : np.float64
        Internal value used by this helper.
    realized_physical : np.float64
        Internal value used by this helper.
    maximum_gram_degree : int
        Internal value used by this helper.
    precision_bits : int
        Internal value used by this helper.
    ldl_iterations : int
        Internal value used by this helper.
    work_count : int
        Internal value used by this helper.
    maximum_gram_work : int
        Internal value used by this helper.
    exact_failure : GalerkinAxialCapExactFloorFailure
        Internal value used by this helper.
    realized_failure : GalerkinAxialCapRealizedFloorFailure
        Internal value used by this helper.
    route : GalerkinAxialCapRealizedFloorRoute
        Internal value used by this helper.
    transcript_digest : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapFloorProof
        Internal result produced by this helper.
    """
    evidence = _floor_evidence_payload(
        certificate,
        degree,
        delta,
        midpoint_shift,
        entry_error,
        gram_lower,
        exact_dimensionless,
        exact_physical,
        realized_dimensionless,
        scale_error,
        physical_error,
        realized_physical,
        maximum_gram_degree,
        precision_bits,
        ldl_iterations,
        work_count,
        maximum_gram_work,
        exact_failure,
        realized_failure,
        route,
        transcript_digest,
    )
    proof_digest = sha256(
        {"domain": _PROOF_DIGEST_DOMAIN, "evidence": evidence}
    )
    width = np.float64(fraction_lower_float(delta))
    proof = _make_axial_cap_floor_proof(
        certificate,
        jnp.asarray(degree, dtype=jnp.int64),
        jnp.asarray(width),
        jnp.asarray(midpoint_shift),
        jnp.asarray(entry_error),
        jnp.asarray(gram_lower),
        jnp.asarray(exact_dimensionless),
        jnp.asarray(exact_physical),
        jnp.asarray(realized_dimensionless),
        jnp.asarray(scale_error),
        jnp.asarray(physical_error),
        jnp.asarray(realized_physical),
        jnp.asarray(exact_failure is GalerkinAxialCapExactFloorFailure.NONE),
        jnp.asarray(
            exact_failure is GalerkinAxialCapExactFloorFailure.NONE
            and realized_failure is GalerkinAxialCapRealizedFloorFailure.NONE
        ),
        jnp.asarray(maximum_gram_degree, dtype=jnp.int64),
        jnp.asarray(precision_bits, dtype=jnp.int64),
        jnp.asarray(ldl_iterations, dtype=jnp.int64),
        jnp.asarray(work_count, dtype=jnp.int64),
        jnp.asarray(maximum_gram_work, dtype=jnp.int64),
        exact_target_failure=exact_failure,
        realized_floor_failure=realized_failure,
        realized_floor_route=route,
        gram_subinterval_numerator=str(delta.numerator),
        gram_subinterval_denominator=str(delta.denominator),
        exact_floor_target=_EXACT_FLOOR_TARGET,
        gram_proof_route=_GRAM_PROOF_ROUTE,
        gram_work_scope=_GRAM_WORK_SCOPE,
        realized_floor_scope=_REALIZED_FLOOR_SCOPE,
        completion_scope=_ABSORBER_COMPLETION_SCOPE,
        parent_certificate_digest=certificate.certificate_digest,
        gram_transcript_digest=transcript_digest,
        proof_digest=proof_digest,
    )
    jax.block_until_ready(proof)
    _return_value: GalerkinAxialCapFloorProof = proof
    return _return_value


def _gram_degree(absorber: GalerkinAxialCellAbsorber) -> int:
    """PRIVATE: Compute exact LVT.30 normal span from authenticated Iu.

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Internal value used by this helper.

    Returns
    -------
    _return_value : int
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    state = _checked_indices(absorber.support.state_indices, "state_indices")
    normal_values = [int(value) for value in state[:, absorber.terminal_axis]]
    degree = max(normal_values) - min(normal_values)
    if degree > _MAXIMUM_SIGNED_INT64:
        raise ValueError("exact gram_degree must fit signed int64 storage")
    _return_value: int = degree
    return _return_value


def _gram_subinterval(absorber: GalerkinAxialCellAbsorber) -> Fraction:
    """PRIVATE: Select one strict exact subinterval inside the plateau.

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Internal value used by this helper.

    Returns
    -------
    _return_value : Fraction
        Internal result produced by this helper.

    Raises
    ------
    AssertionError
        If an internal validation or arithmetic check fails.
    """
    layer_count = absorber.layer_values.shape[0]
    plateau_width = Fraction(absorber.plateau_count, layer_count)
    delta = min(plateau_width / 2, Fraction(1, 4))
    if not 0 < delta < min(plateau_width, Fraction(1, 2)):
        raise AssertionError("canonical Gram subinterval is not strict")
    _return_value: Fraction = delta
    return _return_value


def _certify_axial_cap_floor_impl(  # noqa: PLR0912,PLR0915
    submitted: GalerkinAxialCapCoefficientCertificate,
    *,
    maximum_gram_degree: int,
    gram_precision_bits: int,
    ldl_iteration_count: int,
    maximum_gram_work: int,
) -> GalerkinAxialCapFloorProof:
    """PRIVATE: Build LVT.29--LVT.32 evidence without recursive replay.

    Parameters
    ----------
    submitted : GalerkinAxialCapCoefficientCertificate
        Internal value used by this helper.
    maximum_gram_degree : int
        Internal value used by this helper.
    gram_precision_bits : int
        Internal value used by this helper.
    ldl_iteration_count : int
        Internal value used by this helper.
    maximum_gram_work : int
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapFloorProof
        Internal result produced by this helper.

    Raises
    ------
    AssertionError
        If an internal validation or arithmetic check fails.
    """
    certificate = _authenticate_coefficient_certificate(submitted)
    absorber = certificate.absorber
    degree = _gram_degree(absorber)
    delta = _gram_subinterval(absorber)
    attempt: _GramAttempt = _gram_attempt(
        degree,
        delta,
        gram_precision_bits,
        ldl_iteration_count,
        maximum_gram_degree,
        maximum_gram_work,
    )
    exact_failure: GalerkinAxialCapExactFloorFailure = attempt["failure"]
    work_count = int(attempt["work_count"])
    transcript_digest = str(attempt["transcript_digest"])
    exact_scale = fraction_from_float(
        float(
            _host_exact_scalar(
                absorber.exact_cap_scale,
                np.dtype(np.float64),
                "exact_cap_scale",
            )
        )
    )
    algebraic_scale = fraction_from_float(
        float(
            _host_exact_scalar(
                absorber.algebraic_cap_scale,
                np.dtype(np.float64),
                "algebraic_cap_scale",
            )
        )
    )
    route = (
        GalerkinAxialCapRealizedFloorRoute.EXACT_FROZEN_SCALE_LVT32A
        if algebraic_scale == exact_scale
        else GalerkinAxialCapRealizedFloorRoute.SCALE_TRANSFER_LVT32B
    )
    scale_error_fraction = abs(algebraic_scale - exact_scale)
    scale_error = np.float64(fraction_upper_float(scale_error_fraction))
    coefficient_is_finite = (
        certificate.failure is GalerkinAxialCapCoefficientFailure.NONE
        and bool(
            _host_exact_scalar(
                certificate.finite_certificate,
                np.dtype(np.bool_),
                "finite_certificate",
            )
        )
    )
    delta_a: Fraction | None = None
    if coefficient_is_finite:
        delta_a = fraction_from_float(
            float(
                _host_exact_scalar(
                    certificate.absorber_operator_error_bound,
                    np.dtype(np.float64),
                    "absorber_operator_error_bound",
                )
            )
        )
        if (
            route
            is GalerkinAxialCapRealizedFloorRoute.EXACT_FROZEN_SCALE_LVT32A
        ):
            physical_error_fraction = algebraic_scale * delta_a
        else:
            physical_error_fraction = (
                exact_scale + scale_error_fraction
            ) * delta_a + scale_error_fraction
        physical_error = np.float64(
            fraction_upper_float(physical_error_fraction)
        )
    else:
        physical_error_fraction = None
        physical_error = np.float64(np.inf)
    if exact_failure is not GalerkinAxialCapExactFloorFailure.NONE:
        midpoint_value = attempt["midpoint_shift"]
        entry_value = attempt["entry_error"]
        midpoint = np.float64(
            0.0
            if not isinstance(midpoint_value, Fraction)
            else fraction_lower_float(midpoint_value)
        )
        entry_error = np.float64(
            np.inf
            if not isinstance(entry_value, Fraction)
            else fraction_upper_float(entry_value)
        )
        _return_value: GalerkinAxialCapFloorProof = _make_floor_result(
            certificate,
            degree,
            delta,
            midpoint,
            entry_error,
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(-np.inf),
            scale_error,
            physical_error,
            np.float64(-np.inf),
            maximum_gram_degree,
            gram_precision_bits,
            ldl_iteration_count,
            work_count,
            maximum_gram_work,
            exact_failure,
            GalerkinAxialCapRealizedFloorFailure.EXACT_TARGET_FLOOR_NOT_FINITE,
            route,
            transcript_digest,
        )
        return _return_value
    midpoint_fraction = attempt["midpoint_shift"]
    entry_error_fraction = attempt["entry_error"]
    gram_lower_fraction = attempt["gram_lower"]
    if (
        not isinstance(midpoint_fraction, Fraction)
        or not isinstance(entry_error_fraction, Fraction)
        or not isinstance(gram_lower_fraction, Fraction)
    ):
        raise AssertionError(
            "successful Gram attempt omitted rational evidence"
        )
    plateau_floor = fraction_from_float(
        float(
            _host_exact_scalar(
                absorber.plateau_floor,
                np.dtype(np.float64),
                "plateau_floor",
            )
        )
    )
    exact_dimensionless_fraction = plateau_floor * gram_lower_fraction
    exact_physical_fraction = exact_scale * exact_dimensionless_fraction
    midpoint = np.float64(fraction_lower_float(midpoint_fraction))
    entry_error = np.float64(fraction_upper_float(entry_error_fraction))
    gram_lower = np.float64(fraction_lower_float(gram_lower_fraction))
    exact_dimensionless = np.float64(
        fraction_lower_float(exact_dimensionless_fraction)
    )
    exact_physical = np.float64(fraction_lower_float(exact_physical_fraction))
    if (
        not np.isfinite(midpoint)
        or not np.isfinite(entry_error)
        or not np.isfinite(gram_lower)
        or gram_lower <= 0.0
        or not np.isfinite(exact_dimensionless)
        or exact_dimensionless <= 0.0
        or not np.isfinite(exact_physical)
        or exact_physical <= 0.0
    ):
        _return_value: GalerkinAxialCapFloorProof = _make_floor_result(
            certificate,
            degree,
            delta,
            midpoint,
            entry_error,
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(-np.inf),
            scale_error,
            physical_error,
            np.float64(-np.inf),
            maximum_gram_degree,
            gram_precision_bits,
            ldl_iteration_count,
            work_count,
            maximum_gram_work,
            GalerkinAxialCapExactFloorFailure.ARITHMETIC_RANGE_FAILURE,
            GalerkinAxialCapRealizedFloorFailure.EXACT_TARGET_FLOOR_NOT_FINITE,
            route,
            transcript_digest,
        )
        return _return_value
    if not coefficient_is_finite:
        _return_value: GalerkinAxialCapFloorProof = _make_floor_result(
            certificate,
            degree,
            delta,
            midpoint,
            entry_error,
            gram_lower,
            exact_dimensionless,
            exact_physical,
            np.float64(-np.inf),
            scale_error,
            np.float64(np.inf),
            np.float64(-np.inf),
            maximum_gram_degree,
            gram_precision_bits,
            ldl_iteration_count,
            work_count,
            maximum_gram_work,
            GalerkinAxialCapExactFloorFailure.NONE,
            GalerkinAxialCapRealizedFloorFailure.COEFFICIENT_CERTIFICATE_NOT_FINITE,
            route,
            transcript_digest,
        )
        return _return_value
    if not isinstance(delta_a, Fraction) or not isinstance(
        physical_error_fraction, Fraction
    ):
        raise AssertionError(
            "finite coefficient evidence omitted its physical error transfer"
        )
    realized_dimensionless_fraction = exact_dimensionless_fraction - delta_a
    if route is GalerkinAxialCapRealizedFloorRoute.EXACT_FROZEN_SCALE_LVT32A:
        realized_physical_fraction = (
            algebraic_scale * realized_dimensionless_fraction
        )
    else:
        realized_physical_fraction = (
            exact_physical_fraction - physical_error_fraction
        )
    realized_dimensionless = np.float64(
        fraction_lower_float(realized_dimensionless_fraction)
    )
    realized_physical = np.float64(
        fraction_lower_float(realized_physical_fraction)
    )
    if realized_dimensionless_fraction <= 0:
        realized_failure = GalerkinAxialCapRealizedFloorFailure.REALIZED_DIMENSIONLESS_FLOOR_NONPOSITIVE  # noqa: E501
    elif realized_physical_fraction <= 0:
        realized_failure = GalerkinAxialCapRealizedFloorFailure.REALIZED_PHYSICAL_FLOOR_NONPOSITIVE  # noqa: E501
    elif (
        not np.isfinite(realized_dimensionless)
        or realized_dimensionless <= 0.0
        or not np.isfinite(physical_error)
        or not np.isfinite(realized_physical)
        or realized_physical <= 0.0
    ):
        realized_failure = (
            GalerkinAxialCapRealizedFloorFailure.ARITHMETIC_RANGE_FAILURE
        )
    else:
        realized_failure = GalerkinAxialCapRealizedFloorFailure.NONE
    _return_value: GalerkinAxialCapFloorProof = _make_floor_result(
        certificate,
        degree,
        delta,
        midpoint,
        entry_error,
        gram_lower,
        exact_dimensionless,
        exact_physical,
        realized_dimensionless,
        scale_error,
        physical_error,
        realized_physical,
        maximum_gram_degree,
        gram_precision_bits,
        ldl_iteration_count,
        work_count,
        maximum_gram_work,
        GalerkinAxialCapExactFloorFailure.NONE,
        realized_failure,
        route,
        transcript_digest,
    )
    return _return_value


@jaxtyped(typechecker=beartype)
def certify_axial_cap_floor(
    coefficient_certificate: GalerkinAxialCapCoefficientCertificate,
    *,
    maximum_gram_degree: int = _DEFAULT_MAXIMUM_GRAM_DEGREE,
    gram_precision_bits: int = _DEFAULT_GRAM_PRECISION_BITS,
    ldl_iteration_count: int = _DEFAULT_LDL_ITERATION_COUNT,
    maximum_gram_work: int = _DEFAULT_MAXIMUM_GRAM_WORK,
) -> GalerkinAxialCapFloorProof:
    """Certify exact LVT.29a and, independently, realized LVT.32 floors.

    :see: :func:`~.test_absorber.\
test_verified_floor_routes_are_separate_and_exact_target_is_positive`

    Parameters
    ----------
    coefficient_certificate : GalerkinAxialCapCoefficientCertificate
        Concrete coefficient attempt, fully replay-authenticated first. A
        typed noncertificate may still support an exact-target floor.
    maximum_gram_degree : int, optional
        Maximum LVT.30 support span. Default: ``64``.
    gram_precision_bits : int, optional
        Exact rational trig/midpoint precision in ``[12, 256]``.
        Default: ``48``.
    ldl_iteration_count : int, optional
        Exact LDL-star bisection count. Default: ``64``.
    maximum_gram_work : int, optional
        Maximum versioned abstract exact-Gram host work units.
        Default: ``50_000_000``.

    Returns
    -------
    proof : GalerkinAxialCapFloorProof
        Replayable support-only proof or typed exact and/or realized failure.

    Raises
    ------
    ValueError
        If budgets, nested certificate replay, or exact count storage fail.

    Notes
    -----
    The proof uses the consecutive degree-``d_*`` subinterval Gramian and
    Cauchy interlacing, never a floating eigensolver. Route LVT.32a is selected
    exactly when the two stored binary64 scale reals are equal; otherwise the
    single LVT.32b transfer is selected. The exact-target floor does not depend
    on coefficient-certificate success.
    """
    degree_budget = _checked_nonnegative_budget(
        maximum_gram_degree, "maximum_gram_degree"
    )
    precision = _checked_precision(gram_precision_bits)
    iterations = _checked_budget(ldl_iteration_count, "ldl_iteration_count")
    work_budget = _checked_budget(maximum_gram_work, "maximum_gram_work")
    proof: GalerkinAxialCapFloorProof = _certify_axial_cap_floor_impl(
        coefficient_certificate,
        maximum_gram_degree=degree_budget,
        gram_precision_bits=precision,
        ldl_iteration_count=iterations,
        maximum_gram_work=work_budget,
    )
    return proof  # noqa: RET504


def _proof_parameters(
    proof: GalerkinAxialCapFloorProof,
) -> Tuple[int, int, int, int]:
    """PRIVATE: Read exact stored Gram-proof invocation parameters.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Internal value used by this helper.

    Returns
    -------
    result_0 : int
        Internal result produced by this helper.
    result_1 : int
        Internal result produced by this helper.
    result_2 : int
        Internal result produced by this helper.
    result_3 : int
        Internal result produced by this helper.
    """
    degree_budget = int(
        _host_exact_scalar(
            proof.maximum_gram_degree,
            np.dtype(np.int64),
            "maximum_gram_degree",
        )
    )
    precision = int(
        _host_exact_scalar(
            proof.gram_precision_bits,
            np.dtype(np.int64),
            "gram_precision_bits",
        )
    )
    iterations = int(
        _host_exact_scalar(
            proof.ldl_iteration_count,
            np.dtype(np.int64),
            "ldl_iteration_count",
        )
    )
    work_budget = int(
        _host_exact_scalar(
            proof.maximum_gram_work,
            np.dtype(np.int64),
            "maximum_gram_work",
        )
    )
    _return_value: Tuple[int, int, int, int] = (
        _checked_nonnegative_budget(degree_budget, "maximum_gram_degree"),
        _checked_precision(precision),
        _checked_budget(iterations, "ldl_iteration_count"),
        _checked_budget(work_budget, "maximum_gram_work"),
    )
    return _return_value


def prepare_axial_cap_floor(
    proof: GalerkinAxialCapFloorProof,
) -> GalerkinAxialCapFloorProof:
    """Replay all nested public evidence before transform-compatible use.

    :see: :func:`~.test_absorber.test_prepare_rejects_forged_nested_pair_map`

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Submitted exact/realized floor evidence.

    Returns
    -------
    prepared : GalerkinAxialCapFloorProof
        Fresh canonical proof with exact stored-value identity to the input.

    Raises
    ------
    ValueError
        If any nested source, approximant, rectangle, map, budget, exact
        rational, route, transcript, status, or digest differs from replay.

    Notes
    -----
    Call this after deserialization or another trust boundary. The action
    functions intentionally omit host replay so they remain JAX-transform
    compatible when closed over this prepared value.
    """
    _assert_concrete(proof)
    degree_budget, precision, iterations, work_budget = _proof_parameters(
        proof
    )
    prepared: GalerkinAxialCapFloorProof = _certify_axial_cap_floor_impl(
        proof.coefficient_certificate,
        maximum_gram_degree=degree_budget,
        gram_precision_bits=precision,
        ldl_iteration_count=iterations,
        maximum_gram_work=work_budget,
    )
    if stored_value_payload(prepared) != stored_value_payload(proof):
        raise ValueError(
            "axial CAP floor proof does not match full host replay"
        )
    return prepared


def _checked_action_field(
    proof: GalerkinAxialCapFloorProof,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one physical CAP action input without host replay.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Internal value used by this helper.
    field : Complex[Array, '...']
        Internal value used by this helper.

    Returns
    -------
    checked : Complex128[Array, ' n']
        Internal result produced by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    certificate = proof.coefficient_certificate
    absorber = certificate.absorber
    if (
        absorber.coefficient_formula != _COEFFICIENT_FORMULA
        or absorber.hermitian_approximant_claim != _HERMITIAN_APPROXIMANT_CLAIM
        or absorber.scale_semantics != _SCALE_SEMANTICS
        or absorber.completion_scope != _ABSORBER_COMPLETION_SCOPE
        or certificate.parent_operator_digest != absorber.operator_digest
        or certificate.operator_error_scope != _OPERATOR_ERROR_SCOPE
        or certificate.per_call_arithmetic_exclusion
        != _PER_CALL_ARITHMETIC_EXCLUSION
        or proof.parent_certificate_digest != certificate.certificate_digest
        or proof.realized_floor_scope != _REALIZED_FLOOR_SCOPE
        or proof.completion_scope != _ABSORBER_COMPLETION_SCOPE
    ):
        raise ValueError("axial CAP action semantics are noncanonical")
    values: Complex128[Array, " n"] = jnp.asarray(field, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("field must be 1D")
    state_count = absorber.support.state_indices.shape[0]
    if values.shape != (state_count,):
        raise ValueError("field must match the retained state support")
    checked: Complex128[Array, " n"] = eqx.error_if(
        values,
        (~certificate.finite_certificate)
        | jnp.any(~jnp.isfinite(values))
        | has_subnormal_components(values),
        "physical CAP action requires finite coefficient evidence and a "
        "finite normal-range field",
    )
    return checked


def _apply_physical_cap(
    proof: GalerkinAxialCapFloorProof,
    field: Complex128[Array, " n"],
    *,
    adjoint: bool,
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply physical B_alg forward or reverse-conjugate adjoint.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Internal value used by this helper.
    field : Complex128[Array, ' n']
        Internal value used by this helper.
    adjoint : bool
        Internal value used by this helper.

    Returns
    -------
    checked : Complex128[Array, ' n']
        Internal result produced by this helper.
    """
    certificate = proof.coefficient_certificate
    absorber = certificate.absorber
    coefficients = absorber.absorber_coefficients
    scale = absorber.algebraic_cap_scale
    pair_positions = certificate.state_pair_absorber_positions
    state_count = absorber.support.state_indices.shape[0]
    pair_count = state_count * state_count

    def add_pair(
        flat_position: Int64[Array, ""],
        accumulator: Complex128[Array, " n"],
    ) -> Complex128[Array, " n"]:
        """Accumulate one exact state pair without materializing a matrix."""
        row = flat_position // state_count
        column = flat_position % state_count
        coefficient = scale * coefficients[pair_positions[flat_position]]
        if adjoint:
            updated = accumulator.at[column].add(
                jnp.conj(coefficient) * field[row]
            )
        else:
            updated = accumulator.at[row].add(coefficient * field[column])
        _return_value: Complex128[Array, " n"] = updated
        return _return_value

    initial: Complex128[Array, " n"] = jnp.zeros(
        (state_count,), dtype=jnp.complex128
    )
    applied: Complex128[Array, " n"] = jax.lax.fori_loop(
        0, pair_count, add_pair, initial
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        applied,
        jnp.any(~jnp.isfinite(applied)),
        "physical CAP action left finite binary64 range",
    )
    return checked


@jaxtyped(typechecker=beartype)
def apply_axial_physical_cap(
    proof: GalerkinAxialCapFloorProof,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the frozen physical algebraic CAP ``B_alg``.

    :see: :func:`~.test_absorber.\
test_physical_actions_match_dense_matrix_and_formal_adjoint`

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Prepared evidence containing a finite coefficient certificate.
    field : Complex[Array, "..."]
        Retained state coefficient vector.

    Returns
    -------
    physical_cap : Complex128[Array, " n"]
        Rounded action of ``epsilon_alg A_alg`` in state-support order.

    Notes
    -----
    This transform-compatible callable applies the physical frozen matrix,
    not the dimensionless ``A_alg``. LVT.31/LVT.32 concern fixed exact-real
    matrices and explicitly exclude per-call multiplication, accumulation,
    transform, and output rounding error.
    """
    checked: Complex128[Array, " n"] = _checked_action_field(proof, field)
    physical_cap: Complex128[Array, " n"] = _apply_physical_cap(
        proof, checked, adjoint=False
    )
    return physical_cap


@jaxtyped(typechecker=beartype)
def apply_axial_physical_cap_adjoint(
    proof: GalerkinAxialCapFloorProof,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the formal matrix adjoint of frozen physical ``B_alg``.

    :see: :func:`~.test_absorber.\
test_physical_actions_match_dense_matrix_and_formal_adjoint`

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Prepared evidence containing a finite coefficient certificate.
    field : Complex[Array, "..."]
        Retained adjoint-state coefficient vector.

    Returns
    -------
    physical_cap_adjoint : Complex128[Array, " n"]
        Explicit reverse-pair conjugated algebraic-matrix action.

    Notes
    -----
    Exact Hermitian coefficient symmetry makes the mathematical matrix
    Hermitian, but this routine still implements the formal reverse-conjugate
    adjoint. It makes no bitwise or per-call error claim.
    """
    checked: Complex128[Array, " n"] = _checked_action_field(proof, field)
    physical_cap_adjoint: Complex128[Array, " n"] = _apply_physical_cap(
        proof, checked, adjoint=True
    )
    return physical_cap_adjoint


__all__: list[str] = [
    "apply_axial_physical_cap",
    "apply_axial_physical_cap_adjoint",
    "certify_axial_cap_floor",
    "certify_axial_cell_absorber",
    "prepare_axial_cap_floor",
    "realize_axial_cell_absorber",
]
