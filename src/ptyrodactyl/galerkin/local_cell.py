r"""Realize the disjoint LVT-1 periodic local-cell voltage target.

Extended Summary
----------------
This module evaluates the rounded LVT.7 coefficient map for a
:class:`LocalCellPotential3D` and applies the formal physical-metric adjoint
of that rounded callable. DFT bins are selected modularly, while the integer
mode, centered cell sinc, and physical-origin phase remain unwrapped. No
sampled-grid Nyquist or producer-bandwidth gate is part of this route.

Routine Listings
----------------
:func:`apply_local_cell_potential_metric_adjoint`
    Apply the rounded callable's adjoint in the physical cell metric.
:func:`enclose_local_cell_tail`
    Enclose the complete LVT.9 Fourier tail from authenticated evidence.
:func:`realize_local_cell_galerkin_potential`
    Realize a periodic local-cell voltage field on one interaction support.

Notes
-----
The rounded coefficient and adjoint paths remain transform compatible. The
LVT.9 action is an explicit stopped host boundary because it replay-
authenticates DIRECT LVT.13 evidence before exact-rational subtraction.
"""

import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float64,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import (
    RealInterval,
    fraction_from_float,
    fraction_lower_float,
    fraction_upper_float,
    interval_add,
    point_interval,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellPotentialRealization,
    GalerkinLocalCellTailEnclosure,
    GalerkinLocalCellTailFailure,
    GalerkinProductSupport,
    GalerkinVoxelTargetRoute,
    LocalCellPotential3D,
    _create_local_cell_realization,
    _make_local_cell_tail_enclosure,
    create_local_cell_potential_3d,
)

from .acquisition import check_galerkin_acquisition_support

_LOCAL_CELL_RANK: int = 3
_RECIPROCAL_INDEX_RANK = 2
_TAIL_ARITHMETIC: str = (
    "replay-authenticated DIRECT LVT.13 binary64 rectangles; exact Fraction "
    "cell energy and rectangle modulus squares; exact outward LVT.9 "
    "subtraction; nonnegative theorem intersection; verified rational square "
    "roots; outward binary64 endpoints"
)
_TAIL_DIGEST_DOMAIN: str = "ptyrodactyl.local_cell.lvt9_tail_enclosure.v1"
_TAIL_EXACT_TARGET: str = (
    "LVT.9 complete box-L2 Fourier tail outside ordered requested I_chi"
)


def _canonical_checked_acquisition_support(
    submitted: GalerkinAcquisitionSupportResult,
) -> GalerkinAcquisitionSupportResult:
    """PRIVATE: Rebuild one acquisition result from its submitted manifest.

    Parameters
    ----------
    submitted : GalerkinAcquisitionSupportResult
        Caller-supplied checked-support carrier.

    Returns
    -------
    checked : GalerkinAcquisitionSupportResult
        Fresh eligible checker output.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the fresh result is not support eligible.

    Notes
    -----
    Result/status/evidence leaves outside ``manifest`` have no influence on
    the rebuilt artifact, so forged aggregate claims cannot enter this route
    and eager/JIT checker replay need not be bitwise equal.
    """
    canonical: GalerkinAcquisitionSupportResult = (
        check_galerkin_acquisition_support(submitted.manifest)
    )
    eligible: Bool[Array, ""] = (
        canonical.status
        == int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
    ) & canonical.support_eligible
    checked_status = eqx.error_if(
        canonical.status,
        ~eligible,
        "acquisition manifest must independently recheck SUPPORT_ELIGIBLE",
    )
    checked: GalerkinAcquisitionSupportResult = eqx.tree_at(
        lambda result: result.status,
        canonical,
        checked_status,
    )
    return checked


def _outward_add(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose one exact-point nonnegative sum from above.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left exact stored nonnegative binary64 point.
    right : Float64[Array, "..."]
        Right exact stored nonnegative binary64 point.

    Returns
    -------
    result : Float64[Array, "..."]
        FTZ-safe upper endpoint of the exact-real sum.
    """
    result: Float64[Array, "..."] = interval_add(
        point_interval(left),
        point_interval(right),
    )[1]
    return result


def _canonical_local_cell_potential(
    submitted: LocalCellPotential3D,
) -> LocalCellPotential3D:
    """PRIVATE: Rebuild and exact-compare one public source carrier.

    Parameters
    ----------
    submitted : LocalCellPotential3D
        Caller-supplied local-cell storage carrier.

    Returns
    -------
    canonical : LocalCellPotential3D
        Factory-rebuilt carrier with one checked dynamic leaf.

    Raises
    ------
    ValueError
        If any primitive field fails the source factory or static structure,
        leaf shape, or leaf dtype differs after reconstruction.
    equinox.EquinoxRuntimeError
        If cell values fail runtime validation or exact comparison.
    """
    canonical: LocalCellPotential3D = create_local_cell_potential_3d(
        submitted.cell_values,
        cell_size=submitted.cell_size,
        box_size=submitted.box_size,
        cell_center_origin=submitted.cell_center_origin,
        units=submitted.units,
        reference_value=submitted.reference_value,
        reference_semantics=submitted.reference_semantics,
        boundary=submitted.boundary,
        producer=submitted.producer,
        provenance_hash=submitted.provenance_hash,
        producer_coefficient_normalization=(
            submitted.producer_coefficient_normalization
        ),
        producer_bandwidth=submitted.producer_bandwidth,
    )
    submitted_leaves, submitted_structure = jax.tree_util.tree_flatten(
        submitted
    )
    canonical_leaves, canonical_structure = jax.tree_util.tree_flatten(
        canonical
    )
    if submitted_structure != canonical_structure:
        raise ValueError(
            "submitted LocalCellPotential3D has noncanonical static structure"
        )
    if len(submitted_leaves) != len(canonical_leaves):
        raise ValueError(
            "submitted LocalCellPotential3D has noncanonical leaves"
        )
    exact_match: Bool[Array, ""] = jnp.asarray(True)
    for submitted_leaf, canonical_leaf in zip(
        submitted_leaves,
        canonical_leaves,
        strict=True,
    ):
        submitted_array = jnp.asarray(submitted_leaf)
        canonical_array = jnp.asarray(canonical_leaf)
        if (
            submitted_array.shape != canonical_array.shape
            or submitted_array.dtype != canonical_array.dtype
        ):
            raise ValueError(
                "submitted LocalCellPotential3D has noncanonical leaf shape "
                "or dtype"
            )
        exact_match = exact_match & jnp.all(submitted_array == canonical_array)
    checked_values: Float64[Array, "nz ny nx"] = eqx.error_if(
        canonical.cell_values,
        ~exact_match,
        "LocalCellPotential3D must exactly match an independent factory "
        "reconstruction",
    )
    canonical = eqx.tree_at(
        lambda potential: potential.cell_values,
        canonical,
        checked_values,
    )
    return canonical  # noqa: RET504


def _checked_local_cell_indices(
    support: GalerkinProductSupport,
) -> Int64[Array, "p 3"]:
    """PRIVATE: Check exact integer predicates needed by the LVT.7 map.

    Parameters
    ----------
    support : GalerkinProductSupport
        Candidate finite support whose interaction order is realized.

    Returns
    -------
    checked : Int64[Array, "p 3"]
        Unique, ordinary sign-symmetric unwrapped integer modes.

    Raises
    ------
    ValueError
        If the interaction array has the wrong rank or trailing dimension.
    equinox.EquinoxRuntimeError
        If modes are empty, duplicated, unnegatable, or not exactly paired.

    Notes
    -----
    There is deliberately no comparison with the cell-array shape or producer
    bandwidth. Work-grid endpoint and no-alias predicates remain owned by the
    checked acquisition artifact.
    """
    indices: Int64[Array, "p 3"] = support.interaction_indices
    if indices.ndim != _RECIPROCAL_INDEX_RANK or indices.shape[1:] != (3,):
        raise ValueError("interaction_indices must have shape (p, 3)")
    forward_order: Int64[Array, " p"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    sorted_indices: Int64[Array, "p 3"] = indices[forward_order]
    duplicated: Bool[Array, ""] = jnp.any(
        jnp.all(sorted_indices[1:] == sorted_indices[:-1], axis=-1)
    )
    minimum_integer: int = jnp.iinfo(jnp.int64).min
    unnegatable: Bool[Array, ""] = jnp.any(indices == minimum_integer)
    negative_indices: Int64[Array, "p 3"] = -indices
    negative_order: Int64[Array, " p"] = jnp.lexsort(
        (
            negative_indices[:, 2],
            negative_indices[:, 1],
            negative_indices[:, 0],
        )
    )
    sign_symmetric: Bool[Array, ""] = jnp.all(
        sorted_indices == negative_indices[negative_order]
    )
    checked: Int64[Array, "p 3"] = eqx.error_if(
        indices,
        (indices.shape[0] == 0) | duplicated | unnegatable | (~sign_symmetric),
        "local-cell interaction support must be unique and ordinarily "
        "sign symmetric",
    )
    return checked


def _pair_positions(
    indices: Int64[Array, "p 3"],
) -> Int64[Array, " p"]:
    """PRIVATE: Return the exact additive-inverse position of every mode.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Unique ordinary sign-symmetric reciprocal indices.

    Returns
    -------
    positions : Int64[Array, " p"]
        Position of ``-indices[position]`` for every submitted position.

    Notes
    -----
    Two lexicographic sorts avoid the quadratic pair-match matrix used by the
    earlier VC-1 helper.
    """
    forward_order: Int64[Array, " p"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    negative: Int64[Array, "p 3"] = -indices
    negative_order: Int64[Array, " p"] = jnp.lexsort(
        (negative[:, 2], negative[:, 1], negative[:, 0])
    )
    positions: Int64[Array, " p"] = (
        jnp.zeros_like(forward_order).at[forward_order].set(negative_order)
    )
    return positions


def _canonical_pair_mask(
    indices: Int64[Array, "p 3"],
) -> Bool[Array, " p"]:
    """PRIVATE: Select zero and one lexicographically positive pair member.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Unique ordinary sign-symmetric reciprocal indices.

    Returns
    -------
    canonical : Bool[Array, " p"]
        Mask selecting zero and one member of every nonzero pair.
    """
    first: Int64[Array, " p"] = indices[:, 0]
    second: Int64[Array, " p"] = indices[:, 1]
    third: Int64[Array, " p"] = indices[:, 2]
    canonical: Bool[Array, " p"] = (first > 0) | (
        (first == 0) & ((second > 0) | ((second == 0) & (third >= 0)))
    )
    return canonical  # noqa: RET504


def _hermitian_projection(
    indices: Int64[Array, "p 3"],
    raw_coefficients: Complex128[Array, " p"],
) -> Complex128[Array, " p"]:
    """PRIVATE: Apply the stored ordinary-pair Hermitian projection.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Unique ordinary sign-symmetric reciprocal indices.
    raw_coefficients : Complex128[Array, " p"]
        Rounded coefficients before pair averaging.

    Returns
    -------
    coefficients : Complex128[Array, " p"]
        Exactly stored-conjugate coefficients after one pair average.

    Notes
    -----
    This floating projection does not redefine the exact LVT.7 target. Its
    rounding is included in the returned triangle error evidence.
    """
    positions: Int64[Array, " p"] = _pair_positions(indices)
    pair_average: Complex128[Array, " p"] = (
        0.5 * raw_coefficients + 0.5 * jnp.conj(raw_coefficients[positions])
    )
    zero_mode: Bool[Array, " p"] = jnp.all(indices == 0, axis=-1)
    zero_real: Complex128[Array, " p"] = jax.lax.complex(
        jnp.real(raw_coefficients),
        jnp.zeros_like(jnp.real(raw_coefficients)),
    )
    pair_average = jnp.where(
        zero_mode,
        zero_real,
        pair_average,
    )
    canonical: Bool[Array, " p"] = _canonical_pair_mask(indices)
    coefficients: Complex128[Array, " p"] = jnp.where(
        canonical,
        pair_average,
        jnp.conj(pair_average[positions]),
    )
    return coefficients


def _local_cell_shape_factors(
    indices: Int64[Array, "p 3"],
    grid_shape_xyz: Int64[Array, " 3"],
) -> Float64[Array, " p"]:
    r"""PRIVATE: Evaluate centered-cell shape factors with symbolic zeros.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Unwrapped exact integer reciprocal indices.
    grid_shape_xyz : Int64[Array, " 3"]
        Positive cell counts in physical ``(x, y, z)`` order.

    Returns
    -------
    factors : Float64[Array, " p"]
        Product of ``sinc_pi(indices / grid_shape_xyz)`` over three axes.

    Notes
    -----
    Zero modes return exactly one. Nonzero integer multiples of an axis cell
    count return exactly zero instead of a rounded evaluation of ``sin(pi q)``.
    """
    ratios: Float64[Array, "p 3"] = (
        indices.astype(jnp.float64)
        / (grid_shape_xyz.astype(jnp.float64)[None, :])
    )
    symbolic_zero: Bool[Array, "p 3"] = (indices != 0) & (
        jnp.mod(indices, grid_shape_xyz[None, :]) == 0
    )
    axis_factors: Float64[Array, "p 3"] = jnp.where(
        indices == 0,
        1.0,
        jnp.where(symbolic_zero, 0.0, jnp.sinc(ratios)),
    )
    factors: Float64[Array, " p"] = jnp.prod(axis_factors, axis=-1)
    return factors


def _physical_cell_volume(
    local_potential: LocalCellPotential3D,
) -> Float64[Array, ""]:
    """PRIVATE: Round the exact binary64-rational cell volume once.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Canonical source whose box lengths are finite binary64 values.

    Returns
    -------
    cell_volume : Float64[Array, ""]
        Positive finite cell volume after one exact-rational-to-binary64 round.

    Raises
    ------
    ValueError
        If the final metric volume overflows or underflows binary64.

    Notes
    -----
    Exact :class:`fractions.Fraction` multiplication avoids order-dependent
    intermediate overflow for strongly anisotropic boxes. Rejection at a
    binary64 zero or infinity is a rounded-action numerical admissibility
    boundary, not a failure of the exact LVT theorem.
    """
    sample_count: int = math.prod(local_potential.cell_values.shape)
    exact_volume: Fraction = Fraction(1, sample_count)
    for length in local_potential.box_size:
        exact_volume *= Fraction.from_float(length)
    try:
        rounded_volume: float = float(exact_volume)
    except OverflowError as error:
        raise ValueError(
            "physical cell volume must remain finite in binary64"
        ) from error
    if not math.isfinite(rounded_volume) or rounded_volume <= 0.0:
        raise ValueError(
            "physical cell volume must remain positive and finite in binary64"
        )
    cell_volume: Float64[Array, ""] = jnp.asarray(
        rounded_volume,
        dtype=jnp.float64,
    )
    return cell_volume


def _normalized_fft_scales(sample_count: int) -> Tuple[int, float]:
    """PRIVATE: Build fixed shape-only scales for a bounded mean DFT.

    Parameters
    ----------
    sample_count : int
        Positive number of local cells.

    Returns
    -------
    scale_exponent : int
        Power-of-two exponent defining the fixed input multiplier.
    output_scale : float
        One host-computed binary64 factor applied after the FFT.

    Raises
    ------
    ValueError
        If the sample count or derived fixed scale is not positive and finite.

    Notes
    -----
    For ``q = ceil(log2(sample_count))``, the callable is
    ``FFT(values * 2**(-q)) * (2**q / sample_count)``. Both scales depend only
    on shape, so the rounded callable retains a fixed formal adjoint.
    The power-of-two input scale provides fixed headroom and removes the
    ``max / sample_count`` upward-rounding failure. Any remaining nonfinite
    backend result is rejected; no uniform rounded-action certificate is
    claimed here.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    scale_exponent: int = (sample_count - 1).bit_length()
    output_scale: float = math.ldexp(1.0, scale_exponent) / sample_count
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("normalized FFT output scale must be finite")
    result: Tuple[int, float] = (scale_exponent, output_scale)
    return result


def _normalized_fft_adjoint_gain(
    sample_count: int,
    scale_exponent: int,
    output_scale: float,
) -> float:
    """PRIVATE: Round the fixed normalized-FFT transpose gain once.

    Parameters
    ----------
    sample_count : int
        Positive number of local cells.
    scale_exponent : int
        Forward power-of-two input-scale exponent.
    output_scale : float
        Stored forward post-FFT binary64 scale.

    Returns
    -------
    gain : float
        Binary64 rounding of ``sample_count * 2**(-q) * output_scale``.

    Raises
    ------
    ValueError
        If the derived fixed gain is not positive and finite.
    """
    exact_gain: Fraction = Fraction(
        sample_count, 1 << scale_exponent
    ) * Fraction.from_float(output_scale)
    gain: float = float(exact_gain)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("normalized FFT adjoint gain must be finite")
    return gain


def _origin_cycle_fractions(
    local_potential: LocalCellPotential3D,
) -> Float64[Array, " 3"]:
    """PRIVATE: Round canonical origin-to-box ratios once on the host.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Canonical periodic local-cell source.

    Returns
    -------
    fractions : Float64[Array, " 3"]
        Componentwise cell-center origins in box cycles.
    """
    rounded_fractions: list[float] = []
    for origin, length in zip(
        local_potential.cell_center_origin,
        local_potential.box_size,
        strict=True,
    ):
        exact_length: Fraction = Fraction.from_float(length)
        exact_remainder: Fraction = Fraction.from_float(origin) % exact_length
        rounded_fraction: float = float(exact_remainder / exact_length)
        if rounded_fraction >= 1.0:
            rounded_fraction = 0.0
        rounded_fractions.append(rounded_fraction)
    fractions_tuple: Tuple[float, float, float] = (
        rounded_fractions[0],
        rounded_fractions[1],
        rounded_fractions[2],
    )
    fractions: Float64[Array, " 3"] = jnp.asarray(
        fractions_tuple,
        dtype=jnp.float64,
    )
    return fractions


def _negative_origin_phase(
    indices: Int64[Array, "p 3"],
    origin_cycle_fractions: Float64[Array, " 3"],
    active: Bool[Array, " p"],
) -> Complex128[Array, " p"]:
    """PRIVATE: Form one safely reduced rounded negative origin phase.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Unwrapped integer reciprocal modes.
    origin_cycle_fractions : Float64[Array, " 3"]
        Canonical cell-center origins divided by their box lengths.
    active : Bool[Array, " p"]
        Whether the rounded centered-cell sinc product is nonzero. Every exact
        symbolic zero is therefore inactive.

    Returns
    -------
    phase : Complex128[Array, " p"]
        Rounded ``exp(-2 pi i m dot origin/box)`` constants.

    Notes
    -----
    The integer mode multiplies a bounded cycle fraction; this never forms
    ``mode / box_length``. Each product is reduced modulo one before summing.
    Exact symbolic sinc zeros bypass trigonometric evaluation entirely.
    """
    axis_cycles: Float64[Array, "p 3"] = jnp.remainder(
        indices.astype(jnp.float64) * origin_cycle_fractions[None, :],
        1.0,
    )
    cycles: Float64[Array, " p"] = jnp.remainder(
        jnp.sum(axis_cycles, axis=-1),
        1.0,
    )
    rounded: Complex128[Array, " p"] = jnp.exp(-2.0j * jnp.pi * cycles)
    phase: Complex128[Array, " p"] = jnp.where(
        active,
        rounded,
        jnp.ones_like(rounded),
    )
    return phase


def _local_cell_coefficients_from_full_grid(
    full_coefficients: Complex128[Array, "nz ny nx"],
    indices: Int64[Array, "p 3"],
    origin_cycle_fractions: Float64[Array, " 3"],
) -> Complex128[Array, " p"]:
    """PRIVATE: Restrict a mean DFT through the rounded LVT.7 formula.

    Parameters
    ----------
    full_coefficients : Complex128[Array, "nz ny nx"]
        Mean-normalized DFT in storage-axis order.
    indices : Int64[Array, "p 3"]
        Unwrapped exact integer reciprocal indices.
    origin_cycle_fractions : Float64[Array, " 3"]
        Canonical cell-center origins divided by box lengths.

    Returns
    -------
    coefficients : Complex128[Array, " p"]
        Rounded, exactly stored-Hermitian LVT.7 coefficients.

    Notes
    -----
    Only DFT-bin lookup is modular. Shape and origin factors use the unwrapped
    submitted integer mode, so aliases generally remain distinct.
    """
    nz: int
    ny: int
    nx: int
    nz, ny, nx = full_coefficients.shape
    grid_shape_xyz: Int64[Array, " 3"] = jnp.asarray(
        (nx, ny, nz),
        dtype=jnp.int64,
    )
    residues: Int64[Array, "p 3"] = jnp.mod(
        indices,
        grid_shape_xyz[None, :],
    )
    selected: Complex128[Array, " p"] = full_coefficients[
        residues[:, 2],
        residues[:, 1],
        residues[:, 0],
    ]
    factors: Float64[Array, " p"] = _local_cell_shape_factors(
        indices,
        grid_shape_xyz,
    )
    negative_phase: Complex128[Array, " p"] = _negative_origin_phase(
        indices,
        origin_cycle_fractions,
        factors != 0.0,
    )
    raw: Complex128[Array, " p"] = selected * factors * negative_phase
    coefficients: Complex128[Array, " p"] = _hermitian_projection(
        indices,
        raw,
    )
    return coefficients


def _coefficient_error_bounds(
    local_potential: LocalCellPotential3D,
    coefficients: Complex128[Array, " p"],
) -> Float64[Array, " p"]:
    """PRIVATE: Bound rounded coefficients by an outward triangle rule.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Exact stored real cell values.
    coefficients : Complex128[Array, " p"]
        Final rounded post-projection coefficients.

    Returns
    -------
    bounds : Float64[Array, " p"]
        Sound componentwise complex-magnitude error bounds in volts.

    Notes
    -----
    Every exact LVT.7 coefficient has magnitude at most the maximum absolute
    cell voltage because the normalized DFT average and centered-cell sinc
    both have operator norm at most one.
    """
    value_interval: RealInterval = point_interval(local_potential.cell_values)
    value_magnitude_upper: Float64[Array, "nz ny nx"] = jnp.maximum(
        jnp.abs(value_interval[0]),
        jnp.abs(value_interval[1]),
    )
    maximum_value: Float64[Array, ""] = jnp.max(value_magnitude_upper)
    real_interval: RealInterval = point_interval(jnp.real(coefficients))
    imag_interval: RealInterval = point_interval(jnp.imag(coefficients))
    real_magnitude_upper: Float64[Array, " p"] = jnp.maximum(
        jnp.abs(real_interval[0]),
        jnp.abs(real_interval[1]),
    )
    imag_magnitude_upper: Float64[Array, " p"] = jnp.maximum(
        jnp.abs(imag_interval[0]),
        jnp.abs(imag_interval[1]),
    )
    coefficient_l1: Float64[Array, " p"] = _outward_add(
        real_magnitude_upper,
        imag_magnitude_upper,
    )
    bounds: Float64[Array, " p"] = _outward_add(
        coefficient_l1,
        maximum_value,
    )
    return bounds


@jaxtyped(typechecker=beartype)
def realize_local_cell_galerkin_potential(
    local_potential: LocalCellPotential3D,
    support_eligibility: GalerkinAcquisitionSupportResult,
) -> GalerkinLocalCellPotentialRealization:
    r"""Realize a periodic local-cell voltage field on one interaction support.

    :see: :class:`~.test_local_cell.TestLocalCellRealization`

    Implementation Logic
    --------------------
    1. Independently recheck the acquisition artifact and exact box binding.
    2. Evaluate the mean DFT and gather modular bins for unwrapped modes.
    3. Apply centered-cell sinc and physical cell-center-origin phase factors.
    4. Store ordinary signed pairs Hermitianly with stopped triangle evidence.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Periodic real cell voltages with explicit local-cell semantics.
    support_eligibility : GalerkinAcquisitionSupportResult
        Submitted support artifact, independently rechecked in full.

    Returns
    -------
    realization : GalerkinLocalCellPotentialRealization
        Bound LVT-1 payload, ordered coefficients, and stopped error evidence.

    Raises
    ------
    ValueError
        If route, static boundary, or interaction structure is invalid.
    equinox.EquinoxRuntimeError
        If support evidence, exact box binding, coefficients, or errors fail.

    Notes
    -----
    Producer bandwidth is retained in target identity but never read by this
    coefficient map. Modes beyond the sampled-grid Nyquist are valid whenever
    the independent finite work support admits them.
    """
    local_potential = _canonical_local_cell_potential(local_potential)
    if (
        local_potential.target_route
        is not GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1
    ):
        raise ValueError("local_potential must use LOCAL_CELL_LVT1")
    if local_potential.boundary != "periodic":
        raise ValueError("LVT-1 local_potential boundary must be periodic")
    canonical_eligibility: GalerkinAcquisitionSupportResult = (
        _canonical_checked_acquisition_support(support_eligibility)
    )
    support: GalerkinProductSupport = canonical_eligibility.manifest.support
    indices: Int64[Array, "p 3"] = _checked_local_cell_indices(support)
    potential_box: Float64[Array, " 3"] = jnp.asarray(
        local_potential.box_size,
        dtype=jnp.float64,
    )
    potential_box_bits = jax.lax.bitcast_convert_type(
        potential_box,
        jnp.uint64,
    )
    acquisition_box_bits = jax.lax.bitcast_convert_type(
        canonical_eligibility.manifest.box_lengths,
        jnp.uint64,
    )
    indices = eqx.error_if(
        indices,
        jnp.any(acquisition_box_bits != potential_box_bits),
        "LocalCellPotential3D box lengths must exactly match acquisition "
        "support",
    )
    nz: int
    ny: int
    nx: int
    nz, ny, nx = local_potential.cell_values.shape
    sample_count: int = nx * ny * nz
    scale_exponent: int
    output_scale: float
    scale_exponent, output_scale = _normalized_fft_scales(sample_count)
    input_scale: float = math.ldexp(1.0, -scale_exponent)
    scaled_values: Float64[Array, "nz ny nx"] = (
        local_potential.cell_values * input_scale
    )
    full_coefficients: Complex128[Array, "nz ny nx"] = (
        jnp.fft.fftn(scaled_values) * output_scale
    )
    full_coefficients = eqx.error_if(
        full_coefficients,
        jnp.any(~jnp.isfinite(full_coefficients)),
        "rounded local-cell normalized DFT must remain finite",
    )
    origin_cycles: Float64[Array, " 3"] = _origin_cycle_fractions(
        local_potential
    )
    coefficients: Complex128[Array, " p"] = (
        _local_cell_coefficients_from_full_grid(
            full_coefficients,
            indices,
            origin_cycles,
        )
    )
    coefficient_errors: Float64[Array, " p"] = jax.lax.stop_gradient(
        _coefficient_error_bounds(local_potential, coefficients)
    )
    realization: GalerkinLocalCellPotentialRealization = (
        _create_local_cell_realization(
            local_potential,
            canonical_eligibility,
            coefficients,
            coefficient_errors,
        )
    )
    return realization


def _canonical_local_cell_realization(
    submitted: GalerkinLocalCellPotentialRealization,
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Rebuild the map inputs needed by the formal adjoint.

    Parameters
    ----------
    submitted : GalerkinLocalCellPotentialRealization
        Caller-supplied public storage carrier.

    Returns
    -------
    canonical : GalerkinLocalCellPotentialRealization
        Freshly recomputed LVT.7 realization from canonical source and support.

    Notes
    -----
    Public Equinox carriers are structurally forgeable, and eager and compiled
    rounded coefficients need not be bitwise identical. This adjoint depends
    only on the source and support, so it reconstructs those canonical inputs
    and ignores submitted coefficient and error leaves. Future coefficient
    consumers require independent rectangles and digest validation.
    """
    canonical: GalerkinLocalCellPotentialRealization = (
        realize_local_cell_galerkin_potential(
            submitted.local_potential,
            submitted.support_eligibility,
        )
    )
    return canonical  # noqa: RET504


@jaxtyped(typechecker=beartype)
def apply_local_cell_potential_metric_adjoint(
    realization: GalerkinLocalCellPotentialRealization,
    coefficient_cotangent: Complex[Array, "..."],
) -> Float64[Array, "nz ny nx"]:
    r"""Apply the rounded callable's adjoint in the physical cell metric.

    :see: :class:`~.test_local_cell.TestLocalCellRealization`

    Implementation Logic
    --------------------
    1. Apply the self-adjoint stored Hermitian-pair projection to the covector.
    2. Multiply by centered-cell sinc and the conjugated forward phase.
    3. Scatter-add every unwrapped mode into its modular DFT residue.
    4. Transpose the fixed FFT scales, then divide by checked rounded
       ``DeltaV``.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Fixed LVT-1 local-cell coefficient map and ordered support.
    coefficient_cotangent : Complex[Array, "..."]
        Covector under the full ordered realified Euclidean coefficient metric.

    Returns
    -------
    voxel_gradient : Float64[Array, "nz ny nx"]
        Real gradient under ``DeltaV * sum(a * b)``.

    Raises
    ------
    ValueError
        If the cotangent rank or length is invalid.
    equinox.EquinoxRuntimeError
        If a cotangent or derived metric gradient is non-finite.

    Notes
    -----
    Scatter-add is load bearing: distinct beyond-Nyquist modes can share one
    modular DFT bin. Under the checked one-rounded binary64 ``DeltaV``, the
    fixed rounded kernel defines this formal physical-metric adjoint.
    Projecting the covector matches the stored-Hermitian coefficient range.
    This routine returns no exact-target action-error enclosure and therefore
    does not certify an exact LVT.11 action.
    """
    canonical_realization: GalerkinLocalCellPotentialRealization = (
        _canonical_local_cell_realization(realization)
    )
    cotangent: Complex128[Array, " p"] = jnp.asarray(
        coefficient_cotangent,
        dtype=jnp.complex128,
    )
    if cotangent.ndim != 1:
        raise ValueError("coefficient_cotangent must be 1D")
    indices: Int64[Array, "p 3"] = (
        canonical_realization.support.interaction_indices
    )
    if cotangent.shape[0] != indices.shape[0]:
        raise ValueError(
            "coefficient_cotangent must match the interaction support"
        )
    checked_cotangent: Complex128[Array, " p"] = eqx.error_if(
        cotangent,
        jnp.any(~jnp.isfinite(cotangent)),
        "coefficient_cotangent must be finite",
    )
    projected_cotangent: Complex128[Array, " p"] = _hermitian_projection(
        indices,
        checked_cotangent,
    )
    local_potential: LocalCellPotential3D = (
        canonical_realization.local_potential
    )
    nz: int
    ny: int
    nx: int
    nz, ny, nx = local_potential.cell_values.shape
    grid_shape_xyz: Int64[Array, " 3"] = jnp.asarray(
        (nx, ny, nz),
        dtype=jnp.int64,
    )
    residues: Int64[Array, "p 3"] = jnp.mod(
        indices,
        grid_shape_xyz[None, :],
    )
    factors: Float64[Array, " p"] = _local_cell_shape_factors(
        indices,
        grid_shape_xyz,
    )
    origin_cycles: Float64[Array, " 3"] = _origin_cycle_fractions(
        local_potential
    )
    negative_phase: Complex128[Array, " p"] = _negative_origin_phase(
        indices,
        origin_cycles,
        factors != 0.0,
    )
    weights: Complex128[Array, " p"] = (
        factors * projected_cotangent * jnp.conj(negative_phase)
    )
    embedded: Complex128[Array, "nz ny nx"] = (
        jnp.zeros(local_potential.cell_values.shape, dtype=jnp.complex128)
        .at[
            residues[:, 2],
            residues[:, 1],
            residues[:, 0],
        ]
        .add(weights)
    )
    sample_count: int = nx * ny * nz
    scale_exponent: int
    output_scale: float
    scale_exponent, output_scale = _normalized_fft_scales(sample_count)
    adjoint_gain: float = _normalized_fft_adjoint_gain(
        sample_count,
        scale_exponent,
        output_scale,
    )
    euclidean_gradient: Float64[Array, "nz ny nx"] = (
        jnp.real(jnp.fft.ifftn(embedded)) * adjoint_gain
    )
    cell_volume: Float64[Array, ""] = _physical_cell_volume(local_potential)
    raw_gradient: Float64[Array, "nz ny nx"] = euclidean_gradient / cell_volume
    voxel_gradient: Float64[Array, "nz ny nx"] = eqx.error_if(
        raw_gradient,
        jnp.any(~jnp.isfinite(raw_gradient)),
        "local-cell physical-metric gradient must be finite",
    )
    return voxel_gradient


def _rational_square_interval(
    lower: Fraction,
    upper: Fraction,
) -> Tuple[Fraction, Fraction]:
    """PRIVATE: Return the exact image of one rational interval under square.

    Parameters
    ----------
    lower : Fraction
        Finite rational lower endpoint.
    upper : Fraction
        Finite rational upper endpoint.

    Returns
    -------
    squared_lower : Fraction
        Exact lower endpoint of the squared interval hull.
    squared_upper : Fraction
        Exact upper endpoint of the squared interval hull.

    Raises
    ------
    ValueError
        If the submitted endpoints cross.
    """
    if lower > upper:
        raise ValueError("rational interval endpoints must not cross")
    lower_square: Fraction = lower * lower
    upper_square: Fraction = upper * upper
    squared_lower: Fraction = (
        Fraction(0) if lower <= 0 <= upper else min(lower_square, upper_square)
    )
    squared_upper: Fraction = max(lower_square, upper_square)
    result: Tuple[Fraction, Fraction] = (squared_lower, squared_upper)
    return result


def _outward_lvt9_subtraction(
    cell_energy: Fraction,
    retained_energy_lower: Fraction,
    retained_energy_upper: Fraction,
) -> Tuple[Fraction, Fraction, bool]:
    """PRIVATE: Subtract retained energy before nonnegative intersection.

    Parameters
    ----------
    cell_energy : Fraction
        Exact ``DeltaV * sum(phi_n**2)``.
    retained_energy_lower : Fraction
        Exact rational lower bound for ``|Omega| * sum_K |c_m|**2``.
    retained_energy_upper : Fraction
        Exact rational upper bound for the same retained energy.

    Returns
    -------
    squared_tail_lower : Fraction
        Nonnegative lower endpoint after theorem intersection.
    squared_tail_upper : Fraction
        Nonnegative upper endpoint, or zero on contradiction.
    parseval_consistent : bool
        Whether the raw upper endpoint is nonnegative.

    Raises
    ------
    ValueError
        If cell energy is negative or the retained interval is invalid.

    Notes
    -----
    The raw lower endpoint is allowed to be negative because independently
    widened coefficient rectangles can overestimate retained energy. Only
    after exact subtraction is it intersected with the proved nonnegative
    range. A negative raw upper endpoint is not clipped; it fails closed.
    """
    if cell_energy < 0:
        raise ValueError("cell energy must be nonnegative")
    if (
        retained_energy_lower < 0
        or retained_energy_lower > retained_energy_upper
    ):
        raise ValueError(
            "retained-energy interval must be ordered and nonnegative"
        )
    raw_lower: Fraction = cell_energy - retained_energy_upper
    raw_upper: Fraction = cell_energy - retained_energy_lower
    parseval_consistent: bool = raw_upper >= 0
    if raw_upper < 0:
        squared_tail_lower: Fraction = Fraction(0)
        squared_tail_upper: Fraction = Fraction(0)
    else:
        squared_tail_lower = max(Fraction(0), raw_lower)
        squared_tail_upper = raw_upper
    result: Tuple[Fraction, Fraction, bool] = (
        squared_tail_lower,
        squared_tail_upper,
        parseval_consistent,
    )
    return result


def _tail_enclosure_digest(
    parent_certificate_digest: str,
    squared_lower: float,
    squared_upper: float,
    norm_lower: float,
    norm_upper: float,
    *,
    finite_enclosure: bool,
    failure: GalerkinLocalCellTailFailure,
    parent_certificate_failure: GalerkinLocalCellCertificateFailure,
) -> str:
    """PRIVATE: Bind one complete LVT.9 child evidence payload.

    Parameters
    ----------
    parent_certificate_digest : str
        Replay-authenticated DIRECT LVT.13 parent identity.
    squared_lower : float
        Stored squared-tail lower endpoint.
    squared_upper : float
        Stored squared-tail upper endpoint.
    norm_lower : float
        Stored tail-norm lower endpoint.
    norm_upper : float
        Stored tail-norm upper endpoint.
    finite_enclosure : bool
        Whether every endpoint is finite.
    failure : GalerkinLocalCellTailFailure
        Typed LVT.9 outcome.
    parent_certificate_failure : GalerkinLocalCellCertificateFailure
        Exact typed outcome propagated from the parent.

    Returns
    -------
    digest : str
        Lowercase SHA-256 identity of the complete child evidence.
    """
    digest: str = sha256(
        {
            "domain": _TAIL_DIGEST_DOMAIN,
            "target_route": GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1.value,
            "parent_certificate_digest": parent_certificate_digest,
            "exact_target": _TAIL_EXACT_TARGET,
            "arithmetic": _TAIL_ARITHMETIC,
            "finite_enclosure": stored_value_payload(
                np.asarray(finite_enclosure, dtype=np.bool_)
            ),
            "failure": failure.value,
            "parent_certificate_failure": parent_certificate_failure.value,
            "squared_tail_lower_bound": stored_value_payload(
                np.asarray(squared_lower, dtype=np.float64)
            ),
            "squared_tail_upper_bound": stored_value_payload(
                np.asarray(squared_upper, dtype=np.float64)
            ),
            "tail_l2_lower_bound": stored_value_payload(
                np.asarray(norm_lower, dtype=np.float64)
            ),
            "tail_l2_upper_bound": stored_value_payload(
                np.asarray(norm_upper, dtype=np.float64)
            ),
        }
    )
    return digest


def _make_tail_attempt(
    realization: GalerkinLocalCellPotentialRealization,
    squared_lower: float,
    squared_upper: float,
    norm_lower: float,
    norm_upper: float,
    *,
    failure: GalerkinLocalCellTailFailure,
    parent_certificate_failure: GalerkinLocalCellCertificateFailure,
) -> GalerkinLocalCellTailEnclosure:
    """PRIVATE: Digest and store one already-derived LVT.9 attempt.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Replay-authenticated DIRECT LVT.13 parent.
    squared_lower : float
        Squared-tail lower endpoint.
    squared_upper : float
        Squared-tail upper endpoint.
    norm_lower : float
        Tail-norm lower endpoint.
    norm_upper : float
        Tail-norm upper endpoint.
    failure : GalerkinLocalCellTailFailure
        Typed outcome.
    parent_certificate_failure : GalerkinLocalCellCertificateFailure
        Exact typed parent outcome.

    Returns
    -------
    enclosure : GalerkinLocalCellTailEnclosure
        Bound child evidence carrier.

    Raises
    ------
    ValueError
        If the parent has no DIRECT LVT.13 certificate.
    """
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("tail enclosure requires DIRECT LVT.13")
    finite: bool = failure is GalerkinLocalCellTailFailure.NONE
    digest: str = _tail_enclosure_digest(
        certificate.certificate_digest,
        squared_lower,
        squared_upper,
        norm_lower,
        norm_upper,
        finite_enclosure=finite,
        failure=failure,
        parent_certificate_failure=parent_certificate_failure,
    )
    enclosure: GalerkinLocalCellTailEnclosure = (
        _make_local_cell_tail_enclosure(
            realization,
            jnp.asarray(squared_lower, dtype=jnp.float64),
            jnp.asarray(squared_upper, dtype=jnp.float64),
            jnp.asarray(norm_lower, dtype=jnp.float64),
            jnp.asarray(norm_upper, dtype=jnp.float64),
            jnp.asarray(finite, dtype=jnp.bool_),
            failure=failure,
            parent_certificate_failure=parent_certificate_failure,
            exact_target=_TAIL_EXACT_TARGET,
            arithmetic=_TAIL_ARITHMETIC,
            parent_certificate_digest=certificate.certificate_digest,
            tail_enclosure_digest=digest,
        )
    )
    return enclosure


def _nonfinite_tail_attempt(
    realization: GalerkinLocalCellPotentialRealization,
    *,
    failure: GalerkinLocalCellTailFailure,
    parent_certificate_failure: GalerkinLocalCellCertificateFailure,
) -> GalerkinLocalCellTailEnclosure:
    """PRIVATE: Store one typed ``[0, +inf]`` LVT.9 noncertificate.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Replay-authenticated DIRECT LVT.13 parent.
    failure : GalerkinLocalCellTailFailure
        Typed nonfinite LVT.9 outcome.
    parent_certificate_failure : GalerkinLocalCellCertificateFailure
        Exact typed outcome propagated from the parent.

    Returns
    -------
    enclosure : GalerkinLocalCellTailEnclosure
        Bound child evidence with two explicit ``[0, +inf]`` intervals.
    """
    enclosure: GalerkinLocalCellTailEnclosure = _make_tail_attempt(
        realization,
        0.0,
        math.inf,
        0.0,
        math.inf,
        failure=failure,
        parent_certificate_failure=parent_certificate_failure,
    )
    return enclosure


@jaxtyped(typechecker=beartype)
def enclose_local_cell_tail(
    realization: GalerkinLocalCellPotentialRealization,
) -> GalerkinLocalCellTailEnclosure:
    r"""Enclose the complete LVT.9 Fourier tail from authenticated evidence.

    :see: :class:`~.test_local_cell.TestLocalCellTailEnclosure`

    Implementation Logic
    --------------------
    1. Replay-authenticate the submitted DIRECT LVT.13 realization.
    2. Form ``DeltaV * sum(phi_n**2)`` exactly from stored dyadic values.
    3. Square every authenticated exact-coefficient rectangle and sum both
       members of every retained signed pair exactly.
    4. Subtract outward, then intersect only the lower endpoint with the
       proved nonnegative range and enclose both square roots.
    5. Bind the four endpoints and typed outcome to the parent certificate.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete DIRECT LVT.13 realization. Its host certificate is replayed;
        fallback coefficient errors are never used as tail evidence.

    Returns
    -------
    enclosure : GalerkinLocalCellTailEnclosure
        Squared-tail and tail-norm intervals. An authenticated typed parent
        noncertificate propagates as two explicit ``[0, +inf]`` intervals.

    Raises
    ------
    ValueError
        If inputs are traced, parent binding or replay fails, or stored exact
        dtypes and shapes are noncanonical.
    equinox.EquinoxRuntimeError
        If independent parent reconstruction or carrier validation fails.

    Notes
    -----
    ``K`` is exactly the unique ordered interaction support bound by the
    parent certificate. Producer bandwidth, sampled Nyquist residues, work
    support, and LVT.15 operator compression do not enter this potential-tail
    identity. The diagnostic ``cell_size`` is not a geometry owner.
    """
    from .local_cell_certification import (  # noqa: PLC0415
        _authenticate_local_cell_certificate,
    )

    canonical: GalerkinLocalCellPotentialRealization = (
        _authenticate_local_cell_certificate(realization)
    )
    certificate = canonical.coefficient_certificate
    if certificate is None:
        raise ValueError("tail enclosure requires DIRECT LVT.13")
    parent_failure: GalerkinLocalCellCertificateFailure = certificate.failure
    finite_array = np.asarray(jax.device_get(certificate.finite_certificate))
    if finite_array.dtype != np.dtype(np.bool_) or finite_array.shape != ():
        raise ValueError("finite_certificate must be an exact bool scalar")
    if not bool(finite_array):
        enclosure: GalerkinLocalCellTailEnclosure = _nonfinite_tail_attempt(
            canonical,
            failure=(
                GalerkinLocalCellTailFailure.PARENT_CERTIFICATE_NOT_FINITE
            ),
            parent_certificate_failure=parent_failure,
        )
        return enclosure  # noqa: RET504
    if parent_failure is not GalerkinLocalCellCertificateFailure.NONE:
        raise ValueError("finite parent certificate must have failure NONE")

    local_potential: LocalCellPotential3D = canonical.local_potential
    cell_values = np.asarray(jax.device_get(local_potential.cell_values))
    if (
        cell_values.dtype != np.dtype(np.float64)
        or cell_values.ndim != _LOCAL_CELL_RANK
    ):
        raise ValueError(
            "local cell values must have exact float64 3D storage"
        )
    box_volume: Fraction = Fraction(1)
    for length in local_potential.box_size:
        box_volume *= fraction_from_float(length)
    cell_energy_sum: Fraction = sum(
        (
            fraction_from_float(float(value))
            * fraction_from_float(float(value))
            for value in cell_values.flat
        ),
        start=Fraction(0),
    )
    cell_energy: Fraction = box_volume * cell_energy_sum / cell_values.size

    endpoint_arrays = tuple(
        np.asarray(jax.device_get(value))
        for value in (
            certificate.exact_coefficient_real_lower_bounds,
            certificate.exact_coefficient_real_upper_bounds,
            certificate.exact_coefficient_imag_lower_bounds,
            certificate.exact_coefficient_imag_upper_bounds,
        )
    )
    expected_shape = canonical.voltage_coefficients.shape
    if any(
        value.dtype != np.dtype(np.float64)
        or value.shape != expected_shape
        or np.any(~np.isfinite(value))
        for value in endpoint_arrays
    ):
        raise ValueError(
            "finite tail evidence requires finite float64 coefficient "
            "rectangles"
        )
    real_lower, real_upper, imag_lower, imag_upper = endpoint_arrays
    retained_lower: Fraction = Fraction(0)
    retained_upper: Fraction = Fraction(0)
    for submitted in zip(
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        strict=True,
    ):
        real_square = _rational_square_interval(
            fraction_from_float(float(submitted[0])),
            fraction_from_float(float(submitted[1])),
        )
        imag_square = _rational_square_interval(
            fraction_from_float(float(submitted[2])),
            fraction_from_float(float(submitted[3])),
        )
        retained_lower += real_square[0] + imag_square[0]
        retained_upper += real_square[1] + imag_square[1]
    retained_lower *= box_volume
    retained_upper *= box_volume
    (
        squared_lower_fraction,
        squared_upper_fraction,
        parseval_consistent,
    ) = _outward_lvt9_subtraction(
        cell_energy,
        retained_lower,
        retained_upper,
    )
    if not parseval_consistent:
        enclosure: GalerkinLocalCellTailEnclosure = _nonfinite_tail_attempt(
            canonical,
            failure=GalerkinLocalCellTailFailure.PARSEVAL_CONTRADICTION,
            parent_certificate_failure=parent_failure,
        )
        return enclosure  # noqa: RET504
    norm_lower_fraction: Fraction = Fraction(0)
    if squared_lower_fraction > 0:
        norm_lower_fraction = squared_lower_fraction / sqrt_fraction_upper(
            squared_lower_fraction
        )
    norm_upper_fraction: Fraction = sqrt_fraction_upper(squared_upper_fraction)
    squared_lower = fraction_lower_float(squared_lower_fraction)
    squared_upper = fraction_upper_float(squared_upper_fraction)
    norm_lower = fraction_lower_float(norm_lower_fraction)
    norm_upper = fraction_upper_float(norm_upper_fraction)
    if not all(
        math.isfinite(value)
        for value in (squared_lower, squared_upper, norm_lower, norm_upper)
    ):
        enclosure = _nonfinite_tail_attempt(
            canonical,
            failure=GalerkinLocalCellTailFailure.ARITHMETIC_RANGE_FAILURE,
            parent_certificate_failure=parent_failure,
        )
        return enclosure  # noqa: RET504
    enclosure = _make_tail_attempt(
        canonical,
        squared_lower,
        squared_upper,
        norm_lower,
        norm_upper,
        failure=GalerkinLocalCellTailFailure.NONE,
        parent_certificate_failure=parent_failure,
    )
    return enclosure  # noqa: RET504


def _authenticate_local_cell_tail(
    enclosure: GalerkinLocalCellTailEnclosure,
) -> GalerkinLocalCellTailEnclosure:
    """PRIVATE: Replay and exact-compare one complete LVT.9 carrier.

    Parameters
    ----------
    enclosure : GalerkinLocalCellTailEnclosure
        Submitted forgeable LVT.9 evidence storage.

    Returns
    -------
    canonical : GalerkinLocalCellTailEnclosure
        Fresh host replay with exact stored-value identity.

    Raises
    ------
    ValueError
        If any parent, endpoint, typed outcome, or digest field differs.
    equinox.EquinoxRuntimeError
        If parent replay or carrier validation fails.
    """
    canonical: GalerkinLocalCellTailEnclosure = enclose_local_cell_tail(
        enclosure.realization
    )
    if stored_value_payload(canonical) != stored_value_payload(enclosure):
        raise ValueError(
            "local-cell tail enclosure does not match host replay"
        )
    return canonical


__all__: list[str] = [
    "apply_local_cell_potential_metric_adjoint",
    "enclose_local_cell_tail",
    "realize_local_cell_galerkin_potential",
]
