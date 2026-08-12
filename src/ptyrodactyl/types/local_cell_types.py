r"""Define the disjoint LVT-1 local-cell potential and realization carriers.

Extended Summary
----------------
This module owns the exact finite-target identity for real voltages that are
constant on periodic, cell-centered, half-open rectangular cells.  It does
not reinterpret :class:`Potential3D`: point-sample/band-limited semantics and
local-cell semantics have distinct public carriers and route identities.

Routine Listings
----------------
:class:`GalerkinLocalCellCertificateFailure`
    Store the outcome of one direct local-cell certificate attempt.
:class:`GalerkinLocalCellCoefficientCertificate`
    Store independently enclosed exact local-cell coefficients.
:class:`GalerkinLocalCellErrorRoute`
    Store the outward local-cell coefficient-error route.
:class:`GalerkinLocalCellPotentialRealization`
    Store one LVT-1 local-cell coefficient realization.
:class:`GalerkinLocalCellTailEnclosure`
    Store one authenticated LVT.9 full Fourier-tail enclosure.
:class:`GalerkinLocalCellTailFailure`
    Store the outcome of one LVT.9 enclosure attempt.
:class:`GalerkinVoxelTargetRoute`
    Store the disjoint voxel finite-target interpretation.
:class:`LocalCellPotential3D`
    Store real voltages constant on periodic rectangular cells.
:func:`create_local_cell_potential_3d`
    Create a validated periodic local-cell voltage field.

Notes
-----
The cell array uses storage order ``(z, y, x)``. Geometry tuples and integer
reciprocal indices use physical ``(x, y, z)`` order. The authoritative cell
widths are the exact-model quotients ``box_size / (nx, ny, nz)``; the stored
``cell_size`` tuple is diagnostic metadata only.
"""

import math
import re
from collections.abc import Sequence
from enum import Enum
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int,
    Int64,
    Num,
    jaxtyped,
)

from .acquisition_types import (
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
)
from .born_potential_types import GalerkinProductSupport
from .custom_types import scalar_num

_CELL_RANK: int = 3
_CELL_SUPPORT_CONVENTION: str = (
    "cell-centered periodic half-open rectangles [-Delta/2, Delta/2)"
)
_CELL_VALUE_SEMANTICS: str = (
    "complete real voltage constant on each manifested cell"
)
_COEFFICIENT_FORMULA: str = "LVT.7 local-cell SC.13b coefficient v1"
_COEFFICIENT_INDEX_CONVENTION: str = (
    "unwrapped integer mode; modular DFT bin only; no sampled Nyquist gate"
)
_OUTPUT_COEFFICIENT_NORMALIZATION: str = (
    "SC.13b mean DFT times centered-cell sinc and physical-origin phase"
)
_PRODUCER_BANDWIDTH_ROLE: str = (
    "producer metadata only; not an LVT-1 coefficient cutoff"
)
_SHA256_HEX_LENGTH: int = 64
_VOXEL_METRIC: str = "box-volume-over-cell-count weighted real L2"
_XYZ_SIZE: int = 3

type _StaticXYZ = Sequence[float] | Num[Array, " 3"]


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

    Parameters
    ----------
    condition : bool
        Structural failure predicate.
    message : str
        Exception message used when the predicate is true.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _xyz_tuple(values: _StaticXYZ, name: str) -> Tuple[float, float, float]:
    """PRIVATE: Convert and validate one physical ``(x, y, z)`` tuple.

    Parameters
    ----------
    values : _StaticXYZ
        Three submitted physical-coordinate values.
    name : str
        Field name used in validation errors.

    Returns
    -------
    x : float
        Finite physical x value.
    y : float
        Finite physical y value.
    z : float
        Finite physical z value.

    Raises
    ------
    ValueError
        If the input is not exactly three finite real numbers.
    """
    if isinstance(values, str | bytes) or len(values) != _XYZ_SIZE:
        raise ValueError(f"{name} must contain exactly three values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} values must be real numbers")
    values_array: Num[Array, " 3"] = jnp.asarray(values)
    if jnp.issubdtype(values_array.dtype, jnp.bool_):
        raise ValueError(f"{name} values must be real numbers")
    try:
        converted: Tuple[float, ...] = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be real numbers") from error
    x: float = converted[0]
    y: float = converted[1]
    z: float = converted[2]
    result: Tuple[float, float, float] = (x, y, z)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


def _nonempty_text(value: str, name: str) -> str:
    """PRIVATE: Validate and strip one required static declaration.

    Parameters
    ----------
    value : str
        Submitted text.
    name : str
        Field name used in the validation error.

    Returns
    -------
    result : str
        Stripped nonempty text.

    Raises
    ------
    ValueError
        If the value is not a nonempty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result: str = value.strip()
    return result


def _canonical_periodic_origin(
    origin: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """PRIVATE: Reduce one binary64 origin exactly modulo its box.

    Parameters
    ----------
    origin : Tuple[float, float, float]
        Finite submitted physical cell-center origin.
    box_size : Tuple[float, float, float]
        Positive finite periodic box lengths.

    Returns
    -------
    x : float
        Binary64 representative in the first axis's half-open interval.
    y : float
        Binary64 representative in the second axis's half-open interval.
    z : float
        Binary64 representative in the third axis's half-open interval.

    Notes
    -----
    Binary64 inputs are exact dyadic rationals. Fraction arithmetic prevents
    huge whole-box shifts from reaching trigonometric range reduction.
    """
    reduced: list[float] = []
    for coordinate, length in zip(origin, box_size, strict=True):
        exact_remainder: Fraction = Fraction.from_float(
            coordinate
        ) % Fraction.from_float(length)
        rounded_remainder: float = float(exact_remainder)
        if rounded_remainder >= length:
            rounded_remainder = 0.0
        reduced.append(rounded_remainder)
    canonical: Tuple[float, float, float] = (
        reduced[0],
        reduced[1],
        reduced[2],
    )
    return canonical


class GalerkinVoxelTargetRoute(str, Enum):
    """Store the disjoint voxel finite-target interpretation.

    :see: :class:`~.test_local_cell_types.TestLocalCellRouteTypes`

    Attributes
    ----------
    LOCAL_CELL_LVT1 : str
        Piecewise-constant periodic cell field governed by LVT.7.
    TRIGONOMETRIC_VC1 : str
        Periodic trigonometric point-sample target governed by VC-1.
    """

    LOCAL_CELL_LVT1 = "local_cell_lvt1"
    TRIGONOMETRIC_VC1 = "trigonometric_vc1"


class GalerkinLocalCellErrorRoute(str, Enum):
    """Store the outward local-cell coefficient-error route.

    :see: :class:`~.test_local_cell_types.TestLocalCellRouteTypes`

    Attributes
    ----------
    DIRECT_PAIRWISE_HOST_INTERVAL : str
        Exact-rational direct LVT.7 enclosure with pairwise accumulation.
    TRIANGLE_FALLBACK : str
        Backend-independent triangle bound around the rounded LVT.7 result.
    """

    DIRECT_PAIRWISE_HOST_INTERVAL = "lvt1_direct_pairwise_host_interval"
    TRIANGLE_FALLBACK = "lvt1_triangle_fallback"


class GalerkinLocalCellCertificateFailure(str, Enum):
    """Store the outcome of one direct local-cell certificate attempt.

    :see: :class:`GalerkinLocalCellCoefficientCertificate`
    :see: :class:`~.test_local_cell_types.\
TestLocalCellCoefficientCertificateTypes`

    Attributes
    ----------
    NONE : str
        Every exact-target rectangle and error bound is finite.
    HOST_ARITHMETIC_UNSUPPORTED : str
        The host failed a required IEEE-754 binary64 capability probe.
    WORK_BUDGET_EXCEEDED : str
        The declared direct-term budget was insufficient.
    ROOT_ENCLOSURE_FAILURE : str
        Exact rational-turn phase or sinc construction failed closed.
    ARITHMETIC_RANGE_FAILURE : str
        An exact endpoint or error radius could not be stored outward.
    """

    NONE = "none"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalCellTailFailure(str, Enum):
    """Store the outcome of one LVT.9 enclosure attempt.

    :see: :class:`GalerkinLocalCellTailEnclosure`
    :see: :class:`~.test_local_cell_types.TestLocalCellTailEnclosureTypes`

    Attributes
    ----------
    NONE : str
        The authenticated LVT.13 parent produced a finite LVT.9 enclosure.
    PARENT_CERTIFICATE_NOT_FINITE : str
        The replay-authenticated LVT.13 parent is a typed noncertificate.
    PARSEVAL_CONTRADICTION : str
        The retained-energy lower bound exceeds the exact cell energy.
    ARITHMETIC_RANGE_FAILURE : str
        A finite exact rational endpoint is not representable by finite
        outward binary64 evidence.
    """

    NONE = "none"
    PARENT_CERTIFICATE_NOT_FINITE = "parent_certificate_not_finite"
    PARSEVAL_CONTRADICTION = "parseval_contradiction"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalCellCoefficientCertificate(eqx.Module):
    """Store independently enclosed exact local-cell coefficients.

    :see: :class:`~.test_local_cell_types.\
TestLocalCellCoefficientCertificateTypes`

    Attributes
    ----------
    exact_coefficient_real_lower_bounds : Float64[Array, " p"]
        Outward lower real endpoints for exact pre-projection LVT.7.
    exact_coefficient_real_upper_bounds : Float64[Array, " p"]
        Outward upper real endpoints for exact pre-projection LVT.7.
    exact_coefficient_imag_lower_bounds : Float64[Array, " p"]
        Outward lower imaginary endpoints for exact pre-projection LVT.7.
    exact_coefficient_imag_upper_bounds : Float64[Array, " p"]
        Outward upper imaginary endpoints for exact pre-projection LVT.7.
    finite_certificate : Bool[Array, ""]
        Whether every stored exact-target rectangle endpoint is finite.
    direct_term_count : Int64[Array, ""]
        Number of canonical-mode--cell terms expanded by the checker.
    maximum_direct_terms : Int64[Array, ""]
        Positive caller-declared direct work budget.
    failure : GalerkinLocalCellCertificateFailure
        Static typed certificate outcome.
    exact_target : str
        Static exact-target declaration.
    arithmetic : str
        Static host-arithmetic declaration.
    direct_term_count_route : str
        Static algorithm/version defining which direct cell terms are counted.
    coefficient_formula : str
        Static exact LVT.7 formula identifier.
    local_potential_digest : str
        Canonical digest of every declared local-potential field.
    requested_support_digest : str
        Canonical digest of the independently rebuilt requested support.
    stored_coefficients_digest : str
        Canonical digest of the actual stored coefficient dtype, shape, bytes,
        route, formula, and ordering context.
    realization_digest : str
        Parent identity binding source, support, formula, and coefficients.
    certificate_digest : str
        Child identity additionally binding rectangles, direct errors, budget,
        and outcome.

    Notes
    -----
    This public carrier is forgeable storage. Scientific consumers must run
    the host authenticator owned by ``galerkin.local_cell_certification``
    before using it; a checksum is not proof by construction.
    """

    exact_coefficient_real_lower_bounds: Float64[Array, " p"]
    exact_coefficient_real_upper_bounds: Float64[Array, " p"]
    exact_coefficient_imag_lower_bounds: Float64[Array, " p"]
    exact_coefficient_imag_upper_bounds: Float64[Array, " p"]
    finite_certificate: Bool[Array, ""]
    direct_term_count: Int64[Array, ""]
    maximum_direct_terms: Int64[Array, ""]
    failure: GalerkinLocalCellCertificateFailure = eqx.field(static=True)
    exact_target: str = eqx.field(static=True)
    arithmetic: str = eqx.field(static=True)
    direct_term_count_route: str = eqx.field(static=True)
    coefficient_formula: str = eqx.field(static=True)
    local_potential_digest: str = eqx.field(static=True)
    requested_support_digest: str = eqx.field(static=True)
    stored_coefficients_digest: str = eqx.field(static=True)
    realization_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def _make_local_cell_certificate(  # noqa: PLR0913
    exact_coefficient_real_lower_bounds: Float[Array, "..."],
    exact_coefficient_real_upper_bounds: Float[Array, "..."],
    exact_coefficient_imag_lower_bounds: Float[Array, "..."],
    exact_coefficient_imag_upper_bounds: Float[Array, "..."],
    finite_certificate: Bool[Array, ""],
    direct_term_count: Int[Array, ""],
    maximum_direct_terms: Int[Array, ""],
    *,
    failure: GalerkinLocalCellCertificateFailure,
    exact_target: str,
    arithmetic: str,
    direct_term_count_route: str,
    coefficient_formula: str,
    local_potential_digest: str,
    requested_support_digest: str,
    stored_coefficients_digest: str,
    realization_digest: str,
    certificate_digest: str,
) -> GalerkinLocalCellCoefficientCertificate:
    """PRIVATE: Store one internally computed direct host certificate.

    Parameters
    ----------
    exact_coefficient_real_lower_bounds : Float[Array, "..."]
        Submitted real lower endpoints.
    exact_coefficient_real_upper_bounds : Float[Array, "..."]
        Submitted real upper endpoints.
    exact_coefficient_imag_lower_bounds : Float[Array, "..."]
        Submitted imaginary lower endpoints.
    exact_coefficient_imag_upper_bounds : Float[Array, "..."]
        Submitted imaginary upper endpoints.
    finite_certificate : Bool[Array, ""]
        Whether every exact-target rectangle endpoint is finite.
    direct_term_count : Int[Array, ""]
        Expanded canonical-mode--cell term count.
    maximum_direct_terms : Int[Array, ""]
        Positive direct work budget.
    failure : GalerkinLocalCellCertificateFailure
        Typed certificate outcome.
    exact_target : str
        Nonempty exact-target declaration.
    arithmetic : str
        Nonempty host-arithmetic declaration.
    direct_term_count_route : str
        Nonempty direct-term counting algorithm and version.
    coefficient_formula : str
        Nonempty exact formula identifier.
    local_potential_digest : str
        Canonical local-potential digest.
    requested_support_digest : str
        Canonical requested-support digest.
    stored_coefficients_digest : str
        Canonical actual-coefficient payload digest.
    realization_digest : str
        Parent realization identity.
    certificate_digest : str
        Complete child certificate identity.

    Returns
    -------
    certificate : GalerkinLocalCellCoefficientCertificate
        Structurally validated host-certificate storage.

    Raises
    ------
    ValueError
        If ranks, shapes, scalar fields, declarations, or digests are invalid.
    equinox.EquinoxRuntimeError
        If endpoints are NaN, cross, or contradict the typed outcome.
    """
    real_lower: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_real_lower_bounds,
        dtype=jnp.float64,
    )
    real_upper: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_real_upper_bounds,
        dtype=jnp.float64,
    )
    imag_lower: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_imag_lower_bounds,
        dtype=jnp.float64,
    )
    imag_upper: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_imag_upper_bounds,
        dtype=jnp.float64,
    )
    finite: Bool[Array, ""] = jnp.asarray(
        finite_certificate,
        dtype=jnp.bool_,
    )
    term_count: Int64[Array, ""] = jnp.asarray(
        direct_term_count,
        dtype=jnp.int64,
    )
    term_budget: Int64[Array, ""] = jnp.asarray(
        maximum_direct_terms,
        dtype=jnp.int64,
    )
    endpoint_arrays: Tuple[Float64[Array, " p"], ...] = (
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
    )
    _raise_if(
        any(array.ndim != 1 for array in endpoint_arrays),
        "exact coefficient endpoints must be 1D",
    )
    _raise_if(
        any(array.shape != real_lower.shape for array in endpoint_arrays),
        "exact coefficient endpoint arrays must share one shape",
    )
    for value, name in (
        (finite, "finite_certificate"),
        (term_count, "direct_term_count"),
        (term_budget, "maximum_direct_terms"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")
    for value, name in (
        (exact_target, "exact_target"),
        (arithmetic, "arithmetic"),
        (direct_term_count_route, "direct_term_count_route"),
        (coefficient_formula, "coefficient_formula"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")
    for value, name in (
        (local_potential_digest, "local_potential_digest"),
        (requested_support_digest, "requested_support_digest"),
        (stored_coefficients_digest, "stored_coefficients_digest"),
        (realization_digest, "realization_digest"),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(
            len(value) != _SHA256_HEX_LENGTH
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value),
            f"{name} must be a lowercase SHA-256 digest",
        )

    invalid_endpoints: Bool[Array, ""] = (
        jnp.any(jnp.isnan(real_lower))
        | jnp.any(jnp.isnan(real_upper))
        | jnp.any(jnp.isnan(imag_lower))
        | jnp.any(jnp.isnan(imag_upper))
        | jnp.any(real_lower > real_upper)
        | jnp.any(imag_lower > imag_upper)
    )
    all_endpoints_finite: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(real_lower))
        & jnp.all(jnp.isfinite(real_upper))
        & jnp.all(jnp.isfinite(imag_lower))
        & jnp.all(jnp.isfinite(imag_upper))
    )
    failure_is_none: bool = failure is GalerkinLocalCellCertificateFailure.NONE
    contradiction: Bool[Array, ""] = (
        (finite != all_endpoints_finite)
        | (finite != failure_is_none)
        | (term_count < 0)
        | (term_budget <= 0)
        | (finite & (term_count > term_budget))
    )
    checked_real_lower: Float64[Array, " p"] = eqx.error_if(
        real_lower,
        invalid_endpoints | contradiction,
        "local-cell certificate endpoints or outcome are inconsistent",
    )
    certificate: GalerkinLocalCellCoefficientCertificate = (
        GalerkinLocalCellCoefficientCertificate(
            exact_coefficient_real_lower_bounds=checked_real_lower,
            exact_coefficient_real_upper_bounds=real_upper,
            exact_coefficient_imag_lower_bounds=imag_lower,
            exact_coefficient_imag_upper_bounds=imag_upper,
            finite_certificate=finite,
            direct_term_count=term_count,
            maximum_direct_terms=term_budget,
            failure=failure,
            exact_target=exact_target.strip(),
            arithmetic=arithmetic.strip(),
            direct_term_count_route=direct_term_count_route.strip(),
            coefficient_formula=coefficient_formula.strip(),
            local_potential_digest=local_potential_digest,
            requested_support_digest=requested_support_digest,
            stored_coefficients_digest=stored_coefficients_digest,
            realization_digest=realization_digest,
            certificate_digest=certificate_digest,
        )
    )
    return certificate


class LocalCellPotential3D(eqx.Module):
    """Store real voltages constant on periodic rectangular cells.

    :see: :class:`~.test_local_cell_types.TestLocalCellPotential3D`

    Attributes
    ----------
    cell_values : Float64[Array, "nz ny nx"]
        Complete real voltage assigned to each exact rectangular cell.
    cell_size : Tuple[float, float, float]
        Diagnostic binary64 quotients ``box_size / (nx, ny, nz)``.
    box_size : Tuple[float, float, float]
        Authoritative periodic box lengths in Angstroms.
    cell_center_origin : Tuple[float, float, float]
        Canonical periodic center of cell ``(0, 0, 0)`` in Angstroms, stored
        componentwise in ``[0, box_size)``.
    units : str
        Exact voltage unit identifier, always ``"V"``.
    reference_value : float
        Declared additive electrostatic reference in volts.
    reference_semantics : str
        Physical meaning of the additive reference.
    boundary : str
        Exact periodic boundary declaration.
    producer : str
        Producer name and version.
    provenance_hash : str
        Lowercase SHA-256 producer/source digest.
    producer_coefficient_normalization : str
        Producer-side transform normalization metadata.
    producer_bandwidth : float
        Positive finite producer bandwidth metadata. It is not an exact
        local-cell coefficient cutoff.
    target_route : GalerkinVoxelTargetRoute
        Hardcoded disjoint finite-target route.
    cell_value_semantics : str
        Hardcoded constant-cell value semantics.
    cell_support_convention : str
        Hardcoded centered half-open periodic cell convention.
    producer_bandwidth_role : str
        Hardcoded metadata-only bandwidth disposition.
    coefficient_formula : str
        Hardcoded exact local-cell coefficient formula identifier.
    """

    cell_values: Float64[Array, "nz ny nx"]
    cell_size: Tuple[float, float, float] = eqx.field(static=True)
    box_size: Tuple[float, float, float] = eqx.field(static=True)
    cell_center_origin: Tuple[float, float, float] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    reference_value: float = eqx.field(static=True)
    reference_semantics: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    provenance_hash: str = eqx.field(static=True)
    producer_coefficient_normalization: str = eqx.field(static=True)
    producer_bandwidth: float = eqx.field(static=True)
    target_route: GalerkinVoxelTargetRoute = eqx.field(static=True)
    cell_value_semantics: str = eqx.field(static=True)
    cell_support_convention: str = eqx.field(static=True)
    producer_bandwidth_role: str = eqx.field(static=True)
    coefficient_formula: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_local_cell_potential_3d(  # noqa: PLR0912, PLR0913, PLR0915
    cell_values: Num[Array, "..."],
    cell_size: _StaticXYZ,
    box_size: _StaticXYZ,
    cell_center_origin: _StaticXYZ,
    *,
    units: str = "V",
    reference_value: scalar_num = 0.0,
    reference_semantics: str,
    boundary: str = "periodic",
    producer: str,
    provenance_hash: str,
    producer_coefficient_normalization: str,
    producer_bandwidth: scalar_num,
) -> LocalCellPotential3D:
    """Create a validated periodic local-cell voltage field.

    :see: :class:`~.test_local_cell_types.TestLocalCellPotential3D`

    Parameters
    ----------
    cell_values : Num[Array, "..."]
        Real cell voltages in storage order ``(z, y, x)``.
    cell_size : Sequence[float]
        Submitted diagnostic cell widths in physical ``(x, y, z)`` order.
    box_size : Sequence[float]
        Authoritative box lengths in physical ``(x, y, z)`` order.
    cell_center_origin : Sequence[float]
        Physical center of cell ``(0, 0, 0)``. The factory reduces it exactly
        modulo the box before one binary64 rounding.
    units : str, optional
        Potential units, exactly ``"V"``. Default: ``"V"``.
    reference_value : scalar_num, optional
        Declared additive voltage reference. Default: ``0.0``.
    reference_semantics : str
        Explicit physical-reference declaration supplied by the producer.
    boundary : str, optional
        Boundary convention, exactly ``"periodic"``. Default: ``"periodic"``.
    producer : str
        Producer name and version.
    provenance_hash : str
        SHA-256 producer/source digest, optionally prefixed by ``"sha256:"``.
    producer_coefficient_normalization : str
        Producer-side transform normalization metadata.
    producer_bandwidth : scalar_num
        Positive finite producer bandwidth metadata. It is recorded but is
        never used as an LVT-1 support or Nyquist gate.

    Returns
    -------
    local_potential : LocalCellPotential3D
        Validated disjoint local-cell finite-target payload.

    Raises
    ------
    ValueError
        If geometry, metadata, units, reference, boundary, or array structure
        is invalid.
    equinox.EquinoxRuntimeError
        If a cell value is non-finite, including under JAX transformations.
    """
    raw_values: Num[Array, "..."] = jnp.asarray(cell_values)
    if jnp.issubdtype(raw_values.dtype, jnp.complexfloating):
        raise ValueError("cell_values must be real voltages")
    values: Float64[Array, "nz ny nx"] = raw_values.astype(jnp.float64)
    if values.ndim != _CELL_RANK:
        raise ValueError("cell_values must have shape (nz, ny, nx)")
    if any(size <= 0 for size in values.shape):
        raise ValueError("cell_values dimensions must be positive")

    cell_xyz: Tuple[float, float, float] = _xyz_tuple(cell_size, "cell_size")
    box_xyz: Tuple[float, float, float] = _xyz_tuple(box_size, "box_size")
    origin_xyz: Tuple[float, float, float] = _xyz_tuple(
        cell_center_origin,
        "cell_center_origin",
    )
    if any(value <= 0.0 for value in cell_xyz):
        raise ValueError("cell_size values must be positive")
    if any(value <= 0.0 for value in box_xyz):
        raise ValueError("box_size values must be positive")
    canonical_origin_xyz: Tuple[float, float, float] = (
        _canonical_periodic_origin(origin_xyz, box_xyz)
    )

    nz: int
    ny: int
    nx: int
    nz, ny, nx = values.shape
    canonical_cell_xyz: Tuple[float, float, float] = (
        box_xyz[0] / nx,
        box_xyz[1] / ny,
        box_xyz[2] / nz,
    )
    if not all(
        math.isfinite(value) and value > 0.0 for value in canonical_cell_xyz
    ):
        raise ValueError(
            "box_size / (nx, ny, nz) must remain positive and finite"
        )
    if not all(
        math.isclose(submitted, canonical, rel_tol=1e-12, abs_tol=1e-12)
        for submitted, canonical in zip(
            cell_xyz,
            canonical_cell_xyz,
            strict=True,
        )
    ):
        raise ValueError(
            "box_size must equal cell_size * (nx, ny, nz) in xyz order"
        )

    if units != "V":
        raise ValueError("units must be exactly 'V'")
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
    ambiguous_tokens: set[str] = {
        "none",
        "tbd",
        "unknown",
        "unspecified",
    }
    ambiguous_phrases: Tuple[str, ...] = (
        "not declared",
        "not specified",
        "not stated",
        "to be determined",
    )
    if (
        normalized_reference == "n a"
        or set(normalized_reference.split()) & ambiguous_tokens
        or any(phrase in normalized_reference for phrase in ambiguous_phrases)
    ):
        raise ValueError("reference_semantics must state a physical reference")

    if boundary != "periodic":
        raise ValueError("boundary must be exactly 'periodic' for LVT-1")
    producer_text: str = _nonempty_text(producer, "producer")
    normalization_text: str = _nonempty_text(
        producer_coefficient_normalization,
        "producer_coefficient_normalization",
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

    bandwidth_array: Num[Array, ""] = jnp.asarray(producer_bandwidth)
    if isinstance(producer_bandwidth, bool) or jnp.issubdtype(
        bandwidth_array.dtype,
        jnp.bool_,
    ):
        raise ValueError("producer_bandwidth must be positive and finite")
    bandwidth_float: float = float(producer_bandwidth)
    if not math.isfinite(bandwidth_float) or bandwidth_float <= 0.0:
        raise ValueError("producer_bandwidth must be positive and finite")

    checked_values: Float64[Array, "nz ny nx"] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        "cell_values contain non-finite values",
    )
    local_potential: LocalCellPotential3D = LocalCellPotential3D(
        cell_values=checked_values,
        cell_size=canonical_cell_xyz,
        box_size=box_xyz,
        cell_center_origin=canonical_origin_xyz,
        units="V",
        reference_value=reference_float,
        reference_semantics=reference_text,
        boundary="periodic",
        producer=producer_text,
        provenance_hash=provenance_text,
        producer_coefficient_normalization=normalization_text,
        producer_bandwidth=bandwidth_float,
        target_route=GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1,
        cell_value_semantics=_CELL_VALUE_SEMANTICS,
        cell_support_convention=_CELL_SUPPORT_CONVENTION,
        producer_bandwidth_role=_PRODUCER_BANDWIDTH_ROLE,
        coefficient_formula=_COEFFICIENT_FORMULA,
    )
    return local_potential


class GalerkinLocalCellPotentialRealization(eqx.Module):
    """Store one LVT-1 local-cell coefficient realization.

    :see: :class:`~.test_local_cell_types.TestLocalCellRealizationTypes`

    Attributes
    ----------
    local_potential : LocalCellPotential3D
        Disjoint exact local-cell voltage payload.
    support_eligibility : GalerkinAcquisitionSupportResult
        Independently checked finite acquisition-support artifact.
    voltage_coefficients : Complex128[Array, " p"]
        Stored coefficient approximant on the ordered interaction support.
        The triangle route stores the rounded LVT.7 callable output; the
        direct route stores the actual finite exact-Hermitian approximant
        bounded against exact LVT.7 by its LVT.13 certificate.
    coefficient_error_bounds : Float64[Array, " p"]
        Outward per-coefficient errors relative to exact LVT.7.
    coefficient_certificate : GalerkinLocalCellCoefficientCertificate or None
        Optional direct host evidence. ``None`` denotes the triangle route.
    target_route : GalerkinVoxelTargetRoute
        Hardcoded local-cell route identity.
    error_route : GalerkinLocalCellErrorRoute
        Static coefficient-error route.
    output_coefficient_normalization : str
        Static SC.13b output normalization.
    coefficient_index_convention : str
        Static unwrapped-index convention.
    voxel_metric : str
        Static real physical cell metric.
    coefficient_formula : str
        Static exact coefficient formula identifier.

    Notes
    -----
    This public Equinox carrier is storage, not proof by construction.
    Map-only consumers rebuild canonical source/support inputs and ignore
    submitted coefficient/error leaves. A later coefficient consumer must
    validate independent rectangles and a bound digest; eager/JIT numerical
    replay is not a bitwise integrity mechanism.
    """

    local_potential: LocalCellPotential3D
    support_eligibility: GalerkinAcquisitionSupportResult
    voltage_coefficients: Complex128[Array, " p"]
    coefficient_error_bounds: Float64[Array, " p"]
    target_route: GalerkinVoxelTargetRoute = eqx.field(static=True)
    error_route: GalerkinLocalCellErrorRoute = eqx.field(static=True)
    output_coefficient_normalization: str = eqx.field(static=True)
    coefficient_index_convention: str = eqx.field(static=True)
    voxel_metric: str = eqx.field(static=True)
    coefficient_formula: str = eqx.field(static=True)
    coefficient_certificate: GalerkinLocalCellCoefficientCertificate | None = (
        None
    )

    @property
    def support(self) -> GalerkinProductSupport:
        """Return the checked finite product support without duplicating it."""
        support: GalerkinProductSupport = (
            self.support_eligibility.manifest.support
        )
        return support


@jaxtyped(typechecker=beartype)
def _create_local_cell_realization(
    local_potential: LocalCellPotential3D,
    support_eligibility: GalerkinAcquisitionSupportResult,
    voltage_coefficients: Complex[Array, "..."],
    coefficient_error_bounds: Float[Array, "..."],
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Store an internally computed LVT-1 realization.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Exact local-cell source payload.
    support_eligibility : GalerkinAcquisitionSupportResult
        Checked support artifact defining coefficient order.
    voltage_coefficients : Complex[Array, "..."]
        Internally produced rounded LVT.7/SC.13b callable output for the
        triangle route.
    coefficient_error_bounds : Float[Array, "..."]
        Non-negative outward componentwise coefficient errors.

    Returns
    -------
    realization : GalerkinLocalCellPotentialRealization
        Validated disjoint local-cell coefficients and stopped evidence.

    Raises
    ------
    ValueError
        If route identity, semantics, ranks, or shapes are invalid.
    equinox.EquinoxRuntimeError
        If support is ineligible, coefficients are non-finite, or error bounds
        are NaN or negative.
    """
    _raise_if(
        local_potential.target_route
        is not GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1,
        "local_potential must use LOCAL_CELL_LVT1",
    )
    _raise_if(
        local_potential.cell_value_semantics != _CELL_VALUE_SEMANTICS
        or local_potential.cell_support_convention != _CELL_SUPPORT_CONVENTION
        or local_potential.producer_bandwidth_role != _PRODUCER_BANDWIDTH_ROLE
        or local_potential.coefficient_formula != _COEFFICIENT_FORMULA,
        "local_potential has noncanonical local-cell semantics",
    )
    coefficients: Complex128[Array, " p"] = jnp.asarray(
        voltage_coefficients,
        dtype=jnp.complex128,
    )
    errors: Float64[Array, " p"] = jnp.asarray(
        coefficient_error_bounds,
        dtype=jnp.float64,
    )
    support: GalerkinProductSupport = support_eligibility.manifest.support
    expected_size: int = support.interaction_indices.shape[0]
    _raise_if(coefficients.ndim != 1, "voltage_coefficients must be 1D")
    _raise_if(
        coefficients.shape[0] != expected_size,
        "voltage_coefficients must match interaction support",
    )
    _raise_if(
        errors.shape != coefficients.shape,
        "coefficient_error_bounds must match voltage_coefficients",
    )

    eligible: Bool[Array, ""] = (
        support_eligibility.status
        == int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
    ) & support_eligibility.support_eligible
    checked_coefficients: Complex128[Array, " p"] = eqx.error_if(
        coefficients,
        (~eligible) | jnp.any(~jnp.isfinite(coefficients)),
        "support must be eligible and voltage_coefficients finite",
    )
    checked_errors: Float64[Array, " p"] = eqx.error_if(
        errors,
        jnp.any(jnp.isnan(errors)) | jnp.any(errors < 0.0),
        "coefficient_error_bounds must be non-negative and not NaN",
    )
    realization: GalerkinLocalCellPotentialRealization = (
        GalerkinLocalCellPotentialRealization(
            local_potential=local_potential,
            support_eligibility=support_eligibility,
            voltage_coefficients=checked_coefficients,
            coefficient_error_bounds=checked_errors,
            target_route=GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1,
            error_route=GalerkinLocalCellErrorRoute.TRIANGLE_FALLBACK,
            output_coefficient_normalization=(
                _OUTPUT_COEFFICIENT_NORMALIZATION
            ),
            coefficient_index_convention=_COEFFICIENT_INDEX_CONVENTION,
            voxel_metric=_VOXEL_METRIC,
            coefficient_formula=_COEFFICIENT_FORMULA,
            coefficient_certificate=None,
        )
    )
    return realization


@jaxtyped(typechecker=beartype)
def _create_direct_local_cell_realization(
    realization: GalerkinLocalCellPotentialRealization,
    coefficient_error_bounds: Float[Array, "..."],
    coefficient_certificate: GalerkinLocalCellCoefficientCertificate,
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Attach one internally computed direct certificate and errors.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete canonical source/support and actual finite Hermitian
        coefficient approximant.
    coefficient_error_bounds : Float[Array, "..."]
        Direct outward Euclidean LVT.13 errors in support order.
    coefficient_certificate : GalerkinLocalCellCoefficientCertificate
        Direct exact-target rectangles and bound payload identity.

    Returns
    -------
    refined : GalerkinLocalCellPotentialRealization
        Direct-route realization with jointly checked evidence.

    Raises
    ------
    ValueError
        If formulas or evidence-array shapes are inconsistent.
    equinox.EquinoxRuntimeError
        If errors are NaN, negative, or contradict the certificate outcome.

    Notes
    -----
    This private joint factory prevents a direct route without a certificate
    and prevents finite certificate status from being attached to infinite
    parent errors. Public Equinox storage remains forgeable; digest
    authentication is still required before scientific consumption.
    """
    errors: Float64[Array, " p"] = jnp.asarray(
        coefficient_error_bounds,
        dtype=jnp.float64,
    )
    expected_shape: Tuple[int, ...] = realization.voltage_coefficients.shape
    _raise_if(errors.ndim != 1, "coefficient_error_bounds must be 1D")
    _raise_if(
        errors.shape != expected_shape,
        "coefficient_error_bounds must match voltage_coefficients",
    )
    endpoint_shape: Tuple[int, ...] = (
        coefficient_certificate.exact_coefficient_real_lower_bounds.shape
    )
    _raise_if(
        endpoint_shape != expected_shape,
        "certificate endpoints must match voltage_coefficients",
    )
    _raise_if(
        coefficient_certificate.coefficient_formula
        != realization.coefficient_formula,
        "certificate formula must match the realization formula",
    )
    finite_errors: Bool[Array, ""] = jnp.all(jnp.isfinite(errors))
    all_positive_infinity: Bool[Array, ""] = jnp.all(jnp.isposinf(errors))
    certificate_finite: Bool[Array, ""] = (
        coefficient_certificate.finite_certificate
    )
    failure_is_none: bool = (
        coefficient_certificate.failure
        is GalerkinLocalCellCertificateFailure.NONE
    )
    contradiction: Bool[Array, ""] = (
        jnp.any(jnp.isnan(errors))
        | jnp.any(errors < 0.0)
        | (certificate_finite != finite_errors)
        | (certificate_finite != failure_is_none)
        | ((~certificate_finite) & (~all_positive_infinity))
    )
    checked_errors: Float64[Array, " p"] = eqx.error_if(
        errors,
        contradiction,
        "direct local-cell errors contradict certificate outcome",
    )
    refined: GalerkinLocalCellPotentialRealization = (
        GalerkinLocalCellPotentialRealization(
            local_potential=realization.local_potential,
            support_eligibility=realization.support_eligibility,
            voltage_coefficients=realization.voltage_coefficients,
            coefficient_error_bounds=checked_errors,
            target_route=realization.target_route,
            error_route=(
                GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
            ),
            output_coefficient_normalization=(
                realization.output_coefficient_normalization
            ),
            coefficient_index_convention=(
                realization.coefficient_index_convention
            ),
            voxel_metric=realization.voxel_metric,
            coefficient_formula=realization.coefficient_formula,
            coefficient_certificate=coefficient_certificate,
        )
    )
    return refined


class GalerkinLocalCellTailEnclosure(eqx.Module):
    """Store one authenticated LVT.9 full Fourier-tail enclosure.

    :see: :class:`~.test_local_cell_types.TestLocalCellTailEnclosureTypes`

    Attributes
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Replay-authenticated DIRECT LVT.13 parent realization.
    squared_tail_lower_bound : Float64[Array, ""]
        Outward lower endpoint for the squared box-L2 tail norm.
    squared_tail_upper_bound : Float64[Array, ""]
        Outward upper endpoint for the squared box-L2 tail norm.
    tail_l2_lower_bound : Float64[Array, ""]
        Outward lower endpoint for the box-L2 tail norm.
    tail_l2_upper_bound : Float64[Array, ""]
        Outward upper endpoint for the box-L2 tail norm.
    finite_enclosure : Bool[Array, ""]
        Whether all four LVT.9 endpoints are finite.
    failure : GalerkinLocalCellTailFailure
        Static typed tail-enclosure outcome.
    parent_certificate_failure : GalerkinLocalCellCertificateFailure
        Exact typed outcome propagated from the authenticated parent.
    exact_target : str
        Static LVT.9 target declaration.
    arithmetic : str
        Static exact-rational and outward-conversion declaration.
    parent_certificate_digest : str
        Authenticated DIRECT LVT.13 parent identity.
    tail_enclosure_digest : str
        Complete child evidence identity.

    Notes
    -----
    A parent noncertificate or tail arithmetic failure is represented by both
    intervals ``[0, +inf]``. This public carrier is forgeable storage;
    scientific consumers must replay the private LVT.9 authenticator.
    """

    realization: GalerkinLocalCellPotentialRealization
    squared_tail_lower_bound: Float64[Array, ""]
    squared_tail_upper_bound: Float64[Array, ""]
    tail_l2_lower_bound: Float64[Array, ""]
    tail_l2_upper_bound: Float64[Array, ""]
    finite_enclosure: Bool[Array, ""]
    failure: GalerkinLocalCellTailFailure = eqx.field(static=True)
    parent_certificate_failure: GalerkinLocalCellCertificateFailure = (
        eqx.field(static=True)
    )
    exact_target: str = eqx.field(static=True)
    arithmetic: str = eqx.field(static=True)
    parent_certificate_digest: str = eqx.field(static=True)
    tail_enclosure_digest: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def _make_local_cell_tail_enclosure(  # noqa: PLR0913
    realization: GalerkinLocalCellPotentialRealization,
    squared_tail_lower_bound: Float[Array, ""],
    squared_tail_upper_bound: Float[Array, ""],
    tail_l2_lower_bound: Float[Array, ""],
    tail_l2_upper_bound: Float[Array, ""],
    finite_enclosure: Bool[Array, ""],
    *,
    failure: GalerkinLocalCellTailFailure,
    parent_certificate_failure: GalerkinLocalCellCertificateFailure,
    exact_target: str,
    arithmetic: str,
    parent_certificate_digest: str,
    tail_enclosure_digest: str,
) -> GalerkinLocalCellTailEnclosure:
    """PRIVATE: Jointly validate and store one LVT.9 enclosure attempt.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Replay-authenticated DIRECT LVT.13 parent realization.
    squared_tail_lower_bound : Float[Array, ""]
        Submitted squared-tail lower endpoint.
    squared_tail_upper_bound : Float[Array, ""]
        Submitted squared-tail upper endpoint.
    tail_l2_lower_bound : Float[Array, ""]
        Submitted tail-norm lower endpoint.
    tail_l2_upper_bound : Float[Array, ""]
        Submitted tail-norm upper endpoint.
    finite_enclosure : Bool[Array, ""]
        Whether all submitted endpoints are finite.
    failure : GalerkinLocalCellTailFailure
        Typed tail-enclosure outcome.
    parent_certificate_failure : GalerkinLocalCellCertificateFailure
        Typed outcome of the authenticated parent.
    exact_target : str
        Nonempty LVT.9 target declaration.
    arithmetic : str
        Nonempty arithmetic declaration.
    parent_certificate_digest : str
        Authenticated DIRECT LVT.13 parent identity.
    tail_enclosure_digest : str
        Complete child evidence identity.

    Returns
    -------
    enclosure : GalerkinLocalCellTailEnclosure
        Structurally checked LVT.9 evidence storage.

    Raises
    ------
    ValueError
        If parent binding, declarations, scalar shapes, or digests are
        invalid.
    equinox.EquinoxRuntimeError
        If numeric endpoints contradict the typed outcome.
    """
    squared_lower: Float64[Array, ""] = jnp.asarray(
        squared_tail_lower_bound,
        dtype=jnp.float64,
    )
    squared_upper: Float64[Array, ""] = jnp.asarray(
        squared_tail_upper_bound,
        dtype=jnp.float64,
    )
    norm_lower: Float64[Array, ""] = jnp.asarray(
        tail_l2_lower_bound,
        dtype=jnp.float64,
    )
    norm_upper: Float64[Array, ""] = jnp.asarray(
        tail_l2_upper_bound,
        dtype=jnp.float64,
    )
    finite: Bool[Array, ""] = jnp.asarray(
        finite_enclosure,
        dtype=jnp.bool_,
    )
    scalar_fields = (
        squared_lower,
        squared_upper,
        norm_lower,
        norm_upper,
        finite,
    )
    _raise_if(
        any(value.shape != () for value in scalar_fields),
        "local-cell tail fields must be scalars",
    )
    _raise_if(not exact_target.strip(), "exact_target must be nonempty")
    _raise_if(not arithmetic.strip(), "arithmetic must be nonempty")
    for value, name in (
        (parent_certificate_digest, "parent_certificate_digest"),
        (tail_enclosure_digest, "tail_enclosure_digest"),
    ):
        _raise_if(
            not isinstance(value, str)
            or len(value) != _SHA256_HEX_LENGTH
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value),
            f"{name} must be a lowercase SHA-256 digest",
        )
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("tail enclosure requires DIRECT LVT.13")
    _raise_if(
        certificate.certificate_digest != parent_certificate_digest,
        "tail parent digest must match the realization certificate",
    )
    _raise_if(
        certificate.failure is not parent_certificate_failure,
        "tail parent failure must match the realization certificate",
    )

    failure_is_none: bool = failure is GalerkinLocalCellTailFailure.NONE
    parent_is_none: bool = (
        parent_certificate_failure is GalerkinLocalCellCertificateFailure.NONE
    )
    parent_relation_invalid: bool = (
        (failure is GalerkinLocalCellTailFailure.NONE and not parent_is_none)
        or (
            failure
            is GalerkinLocalCellTailFailure.PARENT_CERTIFICATE_NOT_FINITE
            and parent_is_none
        )
        or (
            failure
            not in (
                GalerkinLocalCellTailFailure.NONE,
                GalerkinLocalCellTailFailure.PARENT_CERTIFICATE_NOT_FINITE,
            )
            and not parent_is_none
        )
    )
    invalid: Bool[Array, ""] = (
        jnp.isnan(squared_lower)
        | jnp.isnan(squared_upper)
        | jnp.isnan(norm_lower)
        | jnp.isnan(norm_upper)
        | (squared_lower < 0.0)
        | (norm_lower < 0.0)
        | (squared_lower > squared_upper)
        | (norm_lower > norm_upper)
        | (finite != failure_is_none)
        | parent_relation_invalid
    )
    success_invalid: Bool[Array, ""] = finite & (
        (~jnp.isfinite(squared_lower))
        | (~jnp.isfinite(squared_upper))
        | (~jnp.isfinite(norm_lower))
        | (~jnp.isfinite(norm_upper))
    )
    failure_invalid: Bool[Array, ""] = (~finite) & (
        (squared_lower != 0.0)
        | (~jnp.isposinf(squared_upper))
        | (norm_lower != 0.0)
        | (~jnp.isposinf(norm_upper))
    )
    checked_squared_lower: Float64[Array, ""] = eqx.error_if(
        squared_lower,
        invalid | success_invalid | failure_invalid,
        "local-cell tail fields contradict their typed outcome",
    )
    enclosure: GalerkinLocalCellTailEnclosure = GalerkinLocalCellTailEnclosure(
        realization=realization,
        squared_tail_lower_bound=checked_squared_lower,
        squared_tail_upper_bound=squared_upper,
        tail_l2_lower_bound=norm_lower,
        tail_l2_upper_bound=norm_upper,
        finite_enclosure=finite,
        failure=failure,
        parent_certificate_failure=parent_certificate_failure,
        exact_target=exact_target.strip(),
        arithmetic=arithmetic.strip(),
        parent_certificate_digest=parent_certificate_digest,
        tail_enclosure_digest=tail_enclosure_digest,
    )
    return enclosure


__all__: list[str] = [
    "GalerkinLocalCellCertificateFailure",
    "GalerkinLocalCellCoefficientCertificate",
    "GalerkinLocalCellErrorRoute",
    "GalerkinLocalCellPotentialRealization",
    "GalerkinLocalCellTailEnclosure",
    "GalerkinLocalCellTailFailure",
    "GalerkinVoxelTargetRoute",
    "LocalCellPotential3D",
    "create_local_cell_potential_3d",
]
