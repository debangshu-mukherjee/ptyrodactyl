r"""Certify stored local-cell approximants against exact LVT.7.

Extended Summary
----------------
This host-only module independently encloses the exact piecewise-constant
local-cell coefficient target on one ordered requested interaction support.
It does not replay or trust the production FFT. Instead, it interprets every
stored binary64 cell value and geometry field as an exact dyadic rational,
encloses the mean DFT, centered-cell sinc, and origin phase, and compares both
members of every stored Hermitian pair with their pre-projection targets.

Routine Listings
----------------
:func:`certify_local_cell_galerkin_potential`
    Certify an actual Hermitian approximant directly against LVT.7.

Notes
-----
The certified coefficient leaf is the actual finite Hermitian approximant
submitted by the caller. Its digest binds those bytes and their direct error;
it does not prove which backend FFT execution produced them. L3 separately
owns full product-support reconstruction, state-difference coverage, and
operator-error transfer. Direct work scales as the cell count times the
non-symbolic canonical-mode count, so this is a caller-budgeted host oracle,
not a scalable replacement for the rounded production coefficient map.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Tuple
from jax.core import Tracer
from jaxtyping import Bool, Complex128, Float64, Int64, jaxtyped
from numpy.typing import NDArray

from ptyrodactyl._canonical_digest import _sha256, _stored_value_payload
from ptyrodactyl._host_interval import (
    _complex_rectangle_multiply,
    _ComplexRectangle,
    _conjugate_rectangle,
    _fraction_from_float,
    _fraction_lower_float,
    _fraction_upper_float,
    _host_binary64_supported,
    _normalized_sinc_integer_ratio,
    _pairwise_rectangle_sum,
    _rational_turn_exponential,
    _real_interval_product,
    _RealInterval,
    _RootEnclosureError,
    _scale_complex_rectangle,
    _sqrt_fraction_upper,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellCoefficientCertificate,
    GalerkinLocalCellErrorRoute,
    GalerkinLocalCellPotentialRealization,
    GalerkinVoxelTargetRoute,
    LocalCellPotential3D,
    _create_direct_local_cell_realization,
    _create_local_cell_realization,
    _make_local_cell_certificate,
)

from .local_cell import (
    _canonical_checked_acquisition_support,
    _canonical_local_cell_potential,
)

_ARITHMETIC: str = (
    "guarded IEEE binary64 host; exact Fraction direct mean DFT; exact "
    "unwrapped integer sinc arguments with symbolic zeros; Machin rational "
    "pi; alternating rational sin/cos; binary pairwise accumulation; "
    "verified rational Euclidean square root; outward binary64 endpoints"
)
_CERTIFICATE_DOMAIN: str = "ptyrodactyl.local_cell.lvt13_certificate.v1"
_COEFFICIENT_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_cell.stored_coefficient_approximant.v1"
)
_DEFAULT_MAXIMUM_DIRECT_TERMS: int = 2_000_000
_EXACT_TARGET: str = (
    "LVT.7 exact pre-projection local-cell SC.13b coefficients"
)
_INDEX_CONVENTION: str = (
    "unwrapped integer mode; modular DFT bin only; no sampled Nyquist gate"
)
_MAXIMUM_DIRECT_TERMS: int = np.iinfo(np.int64).max
_OUTPUT_NORMALIZATION: str = (
    "SC.13b mean DFT times centered-cell sinc and physical-origin phase"
)
_REALIZATION_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_cell.coefficient_realization.v1"
)
_SOURCE_DIGEST_DOMAIN: str = "ptyrodactyl.local_cell.source.v1"
_SUPPORT_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_cell.requested_bound_support.v1"
)
_SUPPORT_RANK: int = 2
_TERM_COUNT_ROUTE: str = "lvt13-canonical-nonsymbolic-sinc-cell-products-v1"
_VOXEL_METRIC: str = "box-volume-over-cell-count weighted real L2"
_ZERO_RECTANGLE: _ComplexRectangle = (
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)


def _assert_concrete(
    realization: GalerkinLocalCellPotentialRealization,
) -> None:
    """PRIVATE: Reject traced leaves at the explicit host boundary.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Submitted realization PyTree.

    Raises
    ------
    ValueError
        If any dynamic realization leaf is a JAX tracer.
    """
    leaves = jax.tree_util.tree_leaves(realization)
    if any(isinstance(leaf, Tracer) for leaf in leaves):
        raise ValueError(
            "direct local-cell certification requires concrete host values"
        )


def _validate_static_realization_semantics(
    realization: GalerkinLocalCellPotentialRealization,
    local_potential: LocalCellPotential3D,
) -> None:
    """PRIVATE: Validate the exact LVT route and formula declarations.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Submitted public storage carrier.
    local_potential : LocalCellPotential3D
        Independently factory-rebuilt local source.

    Raises
    ------
    ValueError
        If any target, formula, ordering, normalization, or metric declaration
        is noncanonical.
    """
    if (
        realization.target_route
        is not GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1
    ):
        raise ValueError("realization must use LOCAL_CELL_LVT1")
    if realization.coefficient_formula != local_potential.coefficient_formula:
        raise ValueError("realization coefficient formula is noncanonical")
    if realization.output_coefficient_normalization != _OUTPUT_NORMALIZATION:
        raise ValueError("realization output normalization is noncanonical")
    if realization.coefficient_index_convention != _INDEX_CONVENTION:
        raise ValueError("realization index convention is noncanonical")
    if realization.voxel_metric != _VOXEL_METRIC:
        raise ValueError("realization voxel metric is noncanonical")
    if not isinstance(realization.error_route, GalerkinLocalCellErrorRoute):
        raise ValueError("realization error route is not a local-cell route")


def _canonical_source_and_support(
    realization: GalerkinLocalCellPotentialRealization,
) -> Tuple[LocalCellPotential3D, GalerkinAcquisitionSupportResult]:
    """PRIVATE: Rebuild semantics without replaying coefficients.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete submitted storage carrier.

    Returns
    -------
    local_potential : LocalCellPotential3D
        Independently factory-rebuilt local source.
    support_result : GalerkinAcquisitionSupportResult
        Fresh eligible acquisition checker output.

    Raises
    ------
    ValueError
        If route declarations or exact box binding are invalid.
    equinox.EquinoxRuntimeError
        If source or support runtime validation fails.

    Notes
    -----
    L2 rechecks the acquisition artifact and exact requested interaction
    support needed by LVT.7. It does not claim L3 full product/no-alias or
    state-difference coverage semantics.
    """
    local_potential: LocalCellPotential3D = _canonical_local_cell_potential(
        realization.local_potential
    )
    support_result: GalerkinAcquisitionSupportResult = (
        _canonical_checked_acquisition_support(realization.support_eligibility)
    )
    jax.block_until_ready((local_potential, support_result))
    _validate_static_realization_semantics(realization, local_potential)

    support_status = int(
        np.asarray(jax.device_get(support_result.status), dtype=np.int64)
    )
    support_eligible = bool(
        np.asarray(jax.device_get(support_result.support_eligible))
    )
    if (
        support_status
        != int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
        or not support_eligible
    ):
        raise ValueError("acquisition support is not SUPPORT_ELIGIBLE")
    source_box = np.asarray(local_potential.box_size, dtype=np.float64)
    support_box = np.asarray(
        jax.device_get(support_result.manifest.box_lengths)
    )
    if support_box.dtype != np.dtype(np.float64) or support_box.shape != (3,):
        raise ValueError("acquisition box must be one binary64 xyz vector")
    if not np.array_equal(
        source_box.view(np.uint64),
        support_box.view(np.uint64),
    ):
        raise ValueError(
            "LocalCellPotential3D box lengths must exactly match acquisition "
            "support"
        )
    result: Tuple[LocalCellPotential3D, GalerkinAcquisitionSupportResult] = (
        local_potential,
        support_result,
    )
    return result


def _mode_tuple(row: Int64[NDArray, " 3"]) -> Tuple[int, int, int]:
    """PRIVATE: Convert one exact host index row to Python integers.

    Parameters
    ----------
    row : Int64[NDArray, " 3"]
        Three-component signed host index.

    Returns
    -------
    first : int
        Physical x reciprocal index.
    second : int
        Physical y reciprocal index.
    third : int
        Physical z reciprocal index.
    """
    result: Tuple[int, int, int] = (
        int(row[0]),
        int(row[1]),
        int(row[2]),
    )
    return result


def _is_canonical_mode(mode: Tuple[int, int, int]) -> bool:
    """PRIVATE: Select zero and one lexicographically positive pair member.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Signed unwrapped reciprocal index.

    Returns
    -------
    result : bool
        Whether this is the canonical ordinary signed-pair member.
    """
    first, second, third = mode
    result: bool = (first > 0) or (
        first == 0 and (second > 0 or (second == 0 and third >= 0))
    )
    return result


def _checked_host_modes(
    support_result: GalerkinAcquisitionSupportResult,
) -> Tuple[
    Int64[NDArray, "p 3"],
    list[Tuple[int, int, int]],
    Dict[Tuple[int, int, int], int],
]:
    """PRIVATE: Validate exact ordinary sign symmetry on requested I-chi.

    Parameters
    ----------
    support_result : GalerkinAcquisitionSupportResult
        Fresh eligible checker output.

    Returns
    -------
    indices : Int64[NDArray, "p 3"]
        Exact ordered requested interaction indices.
    modes : list[Tuple[int, int, int]]
        Same order as exact Python integer triples.
    position_by_mode : Dict[Tuple[int, int, int], int]
        Exact additive-inverse lookup table.

    Raises
    ------
    ValueError
        If dtype, shape, uniqueness, integer range, or ordinary sign symmetry
        is invalid.
    """
    raw_indices = np.asarray(
        jax.device_get(support_result.manifest.support.interaction_indices)
    )
    if raw_indices.dtype != np.dtype(np.int64):
        raise ValueError("interaction_indices must have exact int64 dtype")
    if raw_indices.ndim != _SUPPORT_RANK or raw_indices.shape[1:] != (3,):
        raise ValueError("interaction_indices must have shape (p, 3)")
    if raw_indices.shape[0] == 0:
        raise ValueError("interaction_indices must be nonempty")
    indices: Int64[NDArray, "p 3"] = raw_indices
    modes = [_mode_tuple(row) for row in indices]
    position_by_mode: Dict[Tuple[int, int, int], int] = {
        mode: position for position, mode in enumerate(modes)
    }
    if len(position_by_mode) != len(modes):
        raise ValueError("interaction_indices must be unique")
    minimum_integer = np.iinfo(np.int64).min
    if np.any(indices == minimum_integer):
        raise ValueError("interaction_indices contain an unnegatable integer")
    for mode in modes:
        opposite = (-mode[0], -mode[1], -mode[2])
        if opposite not in position_by_mode:
            raise ValueError(
                "interaction_indices must be ordinarily sign symmetric"
            )
    result: Tuple[
        Int64[NDArray, "p 3"],
        list[Tuple[int, int, int]],
        Dict[Tuple[int, int, int], int],
    ] = (indices, modes, position_by_mode)
    return result


def _checked_host_coefficients(
    realization: GalerkinLocalCellPotentialRealization,
    modes: list[Tuple[int, int, int]],
    position_by_mode: Dict[Tuple[int, int, int], int],
) -> Complex128[NDArray, " p"]:
    """PRIVATE: Validate the actual finite ordinary-Hermitian approximant.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Submitted public coefficient storage.
    modes : list[Tuple[int, int, int]]
        Exact ordered interaction modes.
    position_by_mode : Dict[Tuple[int, int, int], int]
        Exact opposite-mode positions.

    Returns
    -------
    coefficients : Complex128[NDArray, " p"]
        Actual submitted coefficient bytes on the host.

    Raises
    ------
    ValueError
        If dtype, shape, finiteness, exact pair conjugacy, or zero-mode reality
        is invalid.

    Notes
    -----
    Passing this check does not prove provenance from a particular rounded
    FFT. The direct certificate instead binds this exact approximation point
    and its independently computed LVT.13 error.
    """
    raw_coefficients = np.asarray(
        jax.device_get(realization.voltage_coefficients)
    )
    if raw_coefficients.dtype != np.dtype(np.complex128):
        raise ValueError(
            "voltage_coefficients must have exact complex128 dtype"
        )
    if raw_coefficients.ndim != 1 or raw_coefficients.shape != (len(modes),):
        raise ValueError(
            "voltage_coefficients must match the ordered interaction support"
        )
    if not np.all(np.isfinite(raw_coefficients)):
        raise ValueError("voltage_coefficients must be finite")
    coefficients: Complex128[NDArray, " p"] = raw_coefficients
    for position, mode in enumerate(modes):
        if mode == (0, 0, 0):
            if float(np.imag(coefficients[position])) != 0.0:
                raise ValueError(
                    "the stored zero-mode coefficient must be real"
                )
            continue
        if not _is_canonical_mode(mode):
            continue
        opposite = (-mode[0], -mode[1], -mode[2])
        opposite_position = position_by_mode[opposite]
        expected = np.complex128(np.conjugate(coefficients[position]))
        if coefficients[opposite_position] != expected:
            raise ValueError(
                "voltage_coefficients must store exact Hermitian pairs"
            )
    return coefficients


def _symbolic_shape_zero(
    mode: Tuple[int, int, int],
    shape_xyz: Tuple[int, int, int],
) -> bool:
    """PRIVATE: Recognize an exact nonzero integer sinc zero.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Unwrapped integer reciprocal mode.
    shape_xyz : Tuple[int, int, int]
        Positive cell counts in physical xyz order.

    Returns
    -------
    result : bool
        Whether any axis has a nonzero integer multiple of its cell count.
    """
    result: bool = any(
        component != 0 and component % count == 0
        for component, count in zip(mode, shape_xyz, strict=True)
    )
    return result


def _direct_term_count(
    modes: list[Tuple[int, int, int]],
    shape_xyz: Tuple[int, int, int],
) -> int:
    """PRIVATE: Count terms expanded by the versioned direct algorithm.

    Parameters
    ----------
    modes : list[Tuple[int, int, int]]
        Exact ordered sign-symmetric requested modes.
    shape_xyz : Tuple[int, int, int]
        Positive cell counts in physical xyz order.

    Returns
    -------
    count : int
        Cell count times canonical representatives without a symbolic sinc
        zero.

    Notes
    -----
    ``_TERM_COUNT_ROUTE`` binds this exact definition. A representative with
    a symbolic nonzero ``q * N`` sinc zero returns an exact zero rectangle
    without constructing a phase or expanding any DFT cell term, so it adds
    zero to the work count.
    """
    cell_count = math.prod(shape_xyz)
    expanded_representatives = sum(
        _is_canonical_mode(mode) and not _symbolic_shape_zero(mode, shape_xyz)
        for mode in modes
    )
    count: int = cell_count * expanded_representatives
    return count


def _axis_phase_rectangles(
    mode: int,
    size: int,
    cache: Dict[Fraction, _ComplexRectangle],
) -> Tuple[_ComplexRectangle, ...]:
    """PRIVATE: Enclose every direct DFT phase on one cell axis.

    Parameters
    ----------
    mode : int
        Signed unwrapped integer mode.
    size : int
        Positive cell count on the axis.
    cache : Dict[Fraction, _ComplexRectangle]
        Mutable exact-turn phase cache shared across coefficients.

    Returns
    -------
    result : Tuple[_ComplexRectangle, ...]
        Ordered exact-rational phase rectangles.
    """
    values: list[_ComplexRectangle] = []
    for position in range(size):
        turn = Fraction(mode * position, size) % 1
        if turn not in cache:
            cache[turn] = _rational_turn_exponential(turn)
        values.append(cache[turn])
    result: Tuple[_ComplexRectangle, ...] = tuple(values)
    return result


def _shape_factor_rectangle(
    mode: Tuple[int, int, int],
    shape_xyz: Tuple[int, int, int],
) -> _RealInterval:
    """PRIVATE: Enclose the unwrapped separable centered-cell sinc.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Signed unwrapped integer mode.
    shape_xyz : Tuple[int, int, int]
        Positive cell counts in physical xyz order.

    Returns
    -------
    factor : _RealInterval
        Exact-rational real interval containing the LVT.5a shape factor.
    """
    factor: _RealInterval = (Fraction(1), Fraction(1))
    for component, count in zip(mode, shape_xyz, strict=True):
        axis_factor = _normalized_sinc_integer_ratio(component, count)
        factor = _real_interval_product(factor, axis_factor)
    return factor


def _exact_coefficient_rectangle(
    cell_values: Float64[NDArray, "nz ny nx"],
    mode: Tuple[int, int, int],
    origin_xyz: Tuple[float, float, float],
    box_xyz: Tuple[float, float, float],
    phase_cache: Dict[Fraction, _ComplexRectangle],
) -> _ComplexRectangle:
    """PRIVATE: Enclose one exact pre-projection LVT.7 coefficient.

    Parameters
    ----------
    cell_values : Float64[NDArray, "nz ny nx"]
        Exact stored binary64 real cell values.
    mode : Tuple[int, int, int]
        Signed unwrapped reciprocal index in physical xyz order.
    origin_xyz : Tuple[float, float, float]
        Exact stored binary64 cell-center origin.
    box_xyz : Tuple[float, float, float]
        Exact stored binary64 periodic box lengths.
    phase_cache : Dict[Fraction, _ComplexRectangle]
        Mutable exact-turn phase cache shared across coefficients.

    Returns
    -------
    result : _ComplexRectangle
        Exact-rational rectangle enclosing mean DFT times sinc times phase.

    Notes
    -----
    A symbolic sinc zero returns before any phase or DFT work. Otherwise the
    integer mode is never Nyquist-wrapped; only its exact periodic DFT phases
    repeat modulo each cell count.
    """
    nz, ny, nx = cell_values.shape
    shape_xyz: Tuple[int, int, int] = (nx, ny, nz)
    if _symbolic_shape_zero(mode, shape_xyz):
        result: _ComplexRectangle = _ZERO_RECTANGLE
        return result
    shape_factor = _shape_factor_rectangle(mode, shape_xyz)
    mode_x, mode_y, mode_z = mode
    x_phases = _axis_phase_rectangles(mode_x, nx, phase_cache)
    y_phases = _axis_phase_rectangles(mode_y, ny, phase_cache)
    z_phases = _axis_phase_rectangles(mode_z, nz, phase_cache)

    def direct_terms() -> Iterator[_ComplexRectangle]:
        """PRIVATE: Yield exact cell-weighted DFT phase rectangles.

        Yields
        ------
        term : _ComplexRectangle
            Next exact-rational interval term in storage order.
        """
        for z_position in range(nz):
            z_phase = z_phases[z_position]
            for y_position in range(ny):
                yz_phase = _complex_rectangle_multiply(
                    z_phase,
                    y_phases[y_position],
                )
                for x_position in range(nx):
                    grid_phase = _complex_rectangle_multiply(
                        yz_phase,
                        x_phases[x_position],
                    )
                    cell_value = _fraction_from_float(
                        float(
                            cell_values[
                                z_position,
                                y_position,
                                x_position,
                            ]
                        )
                    )
                    term: _ComplexRectangle = _scale_complex_rectangle(
                        grid_phase,
                        cell_value,
                    )
                    yield term

    mean_dft = _scale_complex_rectangle(
        _pairwise_rectangle_sum(direct_terms()),
        Fraction(1, cell_values.size),
    )
    shaped = _complex_rectangle_multiply(
        mean_dft,
        (
            shape_factor[0],
            shape_factor[1],
            Fraction(0),
            Fraction(0),
        ),
    )
    origin_turn = sum(
        (
            component
            * _fraction_from_float(origin)
            / _fraction_from_float(length)
            for component, origin, length in zip(
                mode,
                origin_xyz,
                box_xyz,
                strict=True,
            )
        ),
        start=Fraction(0),
    )
    reduced_origin_turn = origin_turn % 1
    if reduced_origin_turn not in phase_cache:
        phase_cache[reduced_origin_turn] = _rational_turn_exponential(
            reduced_origin_turn
        )
    result: _ComplexRectangle = _complex_rectangle_multiply(
        shaped,
        phase_cache[reduced_origin_turn],
    )
    return result


def _exact_coefficient_rectangles(
    cell_values: Float64[NDArray, "nz ny nx"],
    modes: list[Tuple[int, int, int]],
    position_by_mode: Dict[Tuple[int, int, int], int],
    origin_xyz: Tuple[float, float, float],
    box_xyz: Tuple[float, float, float],
) -> list[_ComplexRectangle]:
    """PRIVATE: Enclose requested coefficients through ordinary conjugacy.

    Parameters
    ----------
    cell_values : Float64[NDArray, "nz ny nx"]
        Exact stored binary64 cell values.
    modes : list[Tuple[int, int, int]]
        Ordered exact sign-symmetric requested modes.
    position_by_mode : Dict[Tuple[int, int, int], int]
        Exact ordinary pair positions.
    origin_xyz : Tuple[float, float, float]
        Exact stored binary64 origin.
    box_xyz : Tuple[float, float, float]
        Exact stored binary64 box lengths.

    Returns
    -------
    result : list[_ComplexRectangle]
        Exact-target rectangles in requested support order.

    Raises
    ------
    ValueError
        If any requested mode remains unpaired.
    """
    rectangles: list[_ComplexRectangle | None] = [None] * len(modes)
    phase_cache: Dict[Fraction, _ComplexRectangle] = {}
    for position, mode in enumerate(modes):
        if not _is_canonical_mode(mode):
            continue
        rectangle = _exact_coefficient_rectangle(
            cell_values,
            mode,
            origin_xyz,
            box_xyz,
            phase_cache,
        )
        rectangles[position] = rectangle
        opposite = (-mode[0], -mode[1], -mode[2])
        rectangles[position_by_mode[opposite]] = _conjugate_rectangle(
            rectangle
        )
    if any(rectangle is None for rectangle in rectangles):
        raise ValueError("requested support has an unpaired signed mode")
    result: list[_ComplexRectangle] = [
        rectangle for rectangle in rectangles if rectangle is not None
    ]
    return result


def _coefficient_euclidean_error_fraction(
    coefficient: np.complex128,
    rectangle: _ComplexRectangle,
) -> Fraction:
    """PRIVATE: Bound point-to-rectangle distance by a rational root above.

    Parameters
    ----------
    coefficient : np.complex128
        Actual stored complex binary64 approximant.
    rectangle : _ComplexRectangle
        Exact-rational target rectangle.

    Returns
    -------
    result : Fraction
        Verified rational upper bound on the farthest-corner Euclidean radius.

    Notes
    -----
    This is tighter than the legacy real-gap-plus-imaginary-gap L1 bound while
    still soundly enclosing ``abs(stored - exact_target)`` for every point in
    the rectangle.
    """
    real = _fraction_from_float(float(np.real(coefficient)))
    imaginary = _fraction_from_float(float(np.imag(coefficient)))
    real_gap = max(abs(real - rectangle[0]), abs(real - rectangle[1]))
    imaginary_gap = max(
        abs(imaginary - rectangle[2]),
        abs(imaginary - rectangle[3]),
    )
    squared_radius = real_gap * real_gap + imaginary_gap * imaginary_gap
    result: Fraction = _sqrt_fraction_upper(squared_radius)
    return result


def _local_potential_digest(local_potential: LocalCellPotential3D) -> str:
    """PRIVATE: Bind every declared source field under an LVT domain tag.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Canonical local source carrier.

    Returns
    -------
    digest : str
        Lowercase SHA-256 source identity.
    """
    digest: str = _sha256(
        {
            "domain": _SOURCE_DIGEST_DOMAIN,
            "local_potential": _stored_value_payload(local_potential),
        }
    )
    return digest


def _requested_support_digest(
    support_result: GalerkinAcquisitionSupportResult,
) -> str:
    """PRIVATE: Bind the fresh requested/bound support checker identity.

    Parameters
    ----------
    support_result : GalerkinAcquisitionSupportResult
        Fresh eligible L2 acquisition checker output.

    Returns
    -------
    digest : str
        Lowercase SHA-256 requested-support identity.

    Notes
    -----
    This binds exact ordered I-chi and its acquisition/box carrier. It is not
    an L3 claim of full product/no-alias or state-difference coverage.
    """
    digest: str = _sha256(
        {
            "domain": _SUPPORT_DIGEST_DOMAIN,
            "requested_bound_support": _stored_value_payload(support_result),
        }
    )
    return digest


def _stored_coefficients_digest(
    coefficients: Complex128[NDArray, " p"],
    indices: Int64[NDArray, "p 3"],
    coefficient_formula: str,
) -> str:
    """PRIVATE: Bind the actual approximant bytes and ordering context.

    Parameters
    ----------
    coefficients : Complex128[NDArray, " p"]
        Actual finite Hermitian stored coefficient bytes.
    indices : Int64[NDArray, "p 3"]
        Exact ordered requested interaction modes.
    coefficient_formula : str
        Exact LVT.7 target formula identifier.

    Returns
    -------
    digest : str
        Lowercase SHA-256 stored-approximant identity.
    """
    digest: str = _sha256(
        {
            "domain": _COEFFICIENT_DIGEST_DOMAIN,
            "target_route": GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1.value,
            "coefficient_formula": coefficient_formula,
            "coefficient_index_convention": _INDEX_CONVENTION,
            "ordered_interaction_indices": _stored_value_payload(indices),
            "stored_coefficients": _stored_value_payload(coefficients),
        }
    )
    return digest


def _realization_digest(
    local_potential_digest: str,
    requested_support_digest: str,
    stored_coefficients_digest: str,
    coefficient_formula: str,
) -> str:
    """PRIVATE: Build the parent L2 coefficient-realization identity.

    Parameters
    ----------
    local_potential_digest : str
        Canonical source digest.
    requested_support_digest : str
        Canonical requested-support digest.
    stored_coefficients_digest : str
        Actual approximant payload digest.
    coefficient_formula : str
        Exact LVT.7 formula identifier.

    Returns
    -------
    digest : str
        Lowercase SHA-256 realization identity.

    Notes
    -----
    This identity deliberately excludes fallback error leaves and any claim
    that eager numerical replay reproduces the submitted bytes.
    """
    digest: str = _sha256(
        {
            "domain": _REALIZATION_DIGEST_DOMAIN,
            "target_route": GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1.value,
            "local_potential_digest": local_potential_digest,
            "requested_support_digest": requested_support_digest,
            "stored_coefficients_digest": stored_coefficients_digest,
            "coefficient_formula": coefficient_formula,
            "output_normalization": _OUTPUT_NORMALIZATION,
            "index_convention": _INDEX_CONVENTION,
        }
    )
    return digest


def _certificate_digest(  # noqa: PLR0913
    realization_digest: str,
    local_potential_digest: str,
    requested_support_digest: str,
    stored_coefficients_digest: str,
    real_lower: Float64[NDArray, " p"],
    real_upper: Float64[NDArray, " p"],
    imag_lower: Float64[NDArray, " p"],
    imag_upper: Float64[NDArray, " p"],
    coefficient_errors: Float64[NDArray, " p"],
    finite_certificate: Bool[NDArray, ""],
    direct_term_count: Int64[NDArray, ""],
    maximum_direct_terms: Int64[NDArray, ""],
    failure: GalerkinLocalCellCertificateFailure,
    coefficient_formula: str,
) -> str:
    """PRIVATE: Bind the complete direct child evidence except its own hash.

    Parameters
    ----------
    realization_digest : str
        Parent L2 realization identity.
    local_potential_digest : str
        Canonical source digest.
    requested_support_digest : str
        Canonical requested-support digest.
    stored_coefficients_digest : str
        Actual approximant payload digest.
    real_lower : Float64[NDArray, " p"]
        Outward exact-target real lower endpoints.
    real_upper : Float64[NDArray, " p"]
        Outward exact-target real upper endpoints.
    imag_lower : Float64[NDArray, " p"]
        Outward exact-target imaginary lower endpoints.
    imag_upper : Float64[NDArray, " p"]
        Outward exact-target imaginary upper endpoints.
    coefficient_errors : Float64[NDArray, " p"]
        Outward direct Euclidean LVT.13 errors.
    finite_certificate : Bool[NDArray, ""]
        Exact stored finite-outcome scalar.
    direct_term_count : Int64[NDArray, ""]
        Exact stored versioned expanded direct-term count.
    maximum_direct_terms : Int64[NDArray, ""]
        Exact stored caller-declared direct work budget.
    failure : GalerkinLocalCellCertificateFailure
        Typed certificate outcome.
    coefficient_formula : str
        Exact LVT.7 formula identifier.

    Returns
    -------
    digest : str
        Lowercase SHA-256 direct-certificate identity.
    """
    digest: str = _sha256(
        {
            "domain": _CERTIFICATE_DOMAIN,
            "realization_digest": realization_digest,
            "local_potential_digest": local_potential_digest,
            "requested_support_digest": requested_support_digest,
            "stored_coefficients_digest": stored_coefficients_digest,
            "exact_target": _EXACT_TARGET,
            "arithmetic": _ARITHMETIC,
            "direct_term_count_route": _TERM_COUNT_ROUTE,
            "coefficient_formula": coefficient_formula,
            "error_route": (
                GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL.value
            ),
            "finite_certificate": _stored_value_payload(finite_certificate),
            "failure": failure.value,
            "direct_term_count": _stored_value_payload(direct_term_count),
            "maximum_direct_terms": _stored_value_payload(
                maximum_direct_terms
            ),
            "real_lower": _stored_value_payload(real_lower),
            "real_upper": _stored_value_payload(real_upper),
            "imag_lower": _stored_value_payload(imag_lower),
            "imag_upper": _stored_value_payload(imag_upper),
            "coefficient_errors": _stored_value_payload(coefficient_errors),
        }
    )
    return digest


def _bound_identity_digests(
    local_potential: LocalCellPotential3D,
    support_result: GalerkinAcquisitionSupportResult,
    coefficients: Complex128[NDArray, " p"],
    indices: Int64[NDArray, "p 3"],
) -> Tuple[str, str, str, str]:
    """PRIVATE: Build source, support, approximant, and parent identities.

    Parameters
    ----------
    local_potential : LocalCellPotential3D
        Canonical local source.
    support_result : GalerkinAcquisitionSupportResult
        Fresh requested-support checker output.
    coefficients : Complex128[NDArray, " p"]
        Actual finite Hermitian approximant.
    indices : Int64[NDArray, "p 3"]
        Exact ordered requested interaction modes.

    Returns
    -------
    local_digest : str
        Canonical source digest.
    support_digest : str
        Canonical requested-support digest.
    coefficients_digest : str
        Actual approximant payload digest.
    parent_digest : str
        Parent L2 realization digest.
    """
    local_digest: str = _local_potential_digest(local_potential)
    support_digest: str = _requested_support_digest(support_result)
    coefficients_digest: str = _stored_coefficients_digest(
        coefficients,
        indices,
        local_potential.coefficient_formula,
    )
    parent_digest: str = _realization_digest(
        local_digest,
        support_digest,
        coefficients_digest,
        local_potential.coefficient_formula,
    )
    result: Tuple[str, str, str, str] = (
        local_digest,
        support_digest,
        coefficients_digest,
        parent_digest,
    )
    return result


def _make_direct_refinement(  # noqa: PLR0913
    base: GalerkinLocalCellPotentialRealization,
    indices: Int64[NDArray, "p 3"],
    coefficients: Complex128[NDArray, " p"],
    real_lower: Float64[NDArray, " p"],
    real_upper: Float64[NDArray, " p"],
    imag_lower: Float64[NDArray, " p"],
    imag_upper: Float64[NDArray, " p"],
    coefficient_errors: Float64[NDArray, " p"],
    term_count: int,
    term_budget: int,
    failure: GalerkinLocalCellCertificateFailure,
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Create one digest-bound direct success or noncertificate.

    Parameters
    ----------
    base : GalerkinLocalCellPotentialRealization
        Canonical source/support and actual approximant.
    indices : Int64[NDArray, "p 3"]
        Exact ordered requested modes.
    coefficients : Complex128[NDArray, " p"]
        Actual approximant host bytes.
    real_lower : Float64[NDArray, " p"]
        Exact-target real lower endpoints.
    real_upper : Float64[NDArray, " p"]
        Exact-target real upper endpoints.
    imag_lower : Float64[NDArray, " p"]
        Exact-target imaginary lower endpoints.
    imag_upper : Float64[NDArray, " p"]
        Exact-target imaginary upper endpoints.
    coefficient_errors : Float64[NDArray, " p"]
        Direct Euclidean LVT.13 errors or positive infinities.
    term_count : int
        Versioned expanded direct-term count.
    term_budget : int
        Caller-declared direct work budget.
    failure : GalerkinLocalCellCertificateFailure
        Typed outcome.

    Returns
    -------
    refined : GalerkinLocalCellPotentialRealization
        Jointly checked direct route and certificate.
    """
    (
        local_digest,
        support_digest,
        coefficients_digest,
        parent_digest,
    ) = _bound_identity_digests(
        base.local_potential,
        base.support_eligibility,
        coefficients,
        indices,
    )
    finite = failure is GalerkinLocalCellCertificateFailure.NONE
    finite_array: Bool[NDArray, ""] = np.asarray(finite, dtype=np.bool_)
    term_count_array: Int64[NDArray, ""] = np.asarray(
        term_count,
        dtype=np.int64,
    )
    term_budget_array: Int64[NDArray, ""] = np.asarray(
        term_budget,
        dtype=np.int64,
    )
    child_digest = _certificate_digest(
        parent_digest,
        local_digest,
        support_digest,
        coefficients_digest,
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        coefficient_errors,
        finite_array,
        term_count_array,
        term_budget_array,
        failure,
        base.coefficient_formula,
    )
    certificate: GalerkinLocalCellCoefficientCertificate = (
        _make_local_cell_certificate(
            jnp.asarray(real_lower, dtype=jnp.float64),
            jnp.asarray(real_upper, dtype=jnp.float64),
            jnp.asarray(imag_lower, dtype=jnp.float64),
            jnp.asarray(imag_upper, dtype=jnp.float64),
            jnp.asarray(finite_array),
            jnp.asarray(term_count_array),
            jnp.asarray(term_budget_array),
            failure=failure,
            exact_target=_EXACT_TARGET,
            arithmetic=_ARITHMETIC,
            direct_term_count_route=_TERM_COUNT_ROUTE,
            coefficient_formula=base.coefficient_formula,
            local_potential_digest=local_digest,
            requested_support_digest=support_digest,
            stored_coefficients_digest=coefficients_digest,
            realization_digest=parent_digest,
            certificate_digest=child_digest,
        )
    )
    stopped_errors: Float64[jax.Array, " p"] = jax.lax.stop_gradient(
        jnp.asarray(coefficient_errors, dtype=jnp.float64)
    )
    refined: GalerkinLocalCellPotentialRealization = (
        _create_direct_local_cell_realization(
            base,
            stopped_errors,
            certificate,
        )
    )
    jax.block_until_ready(refined)
    return refined


def _failure_refinement(
    base: GalerkinLocalCellPotentialRealization,
    indices: Int64[NDArray, "p 3"],
    coefficients: Complex128[NDArray, " p"],
    term_count: int,
    term_budget: int,
    failure: GalerkinLocalCellCertificateFailure,
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Return one typed all-infinite direct noncertificate.

    Parameters
    ----------
    base : GalerkinLocalCellPotentialRealization
        Canonical source/support and actual approximant.
    indices : Int64[NDArray, "p 3"]
        Exact ordered requested modes.
    coefficients : Complex128[NDArray, " p"]
        Actual approximant bytes.
    term_count : int
        Versioned requested work count.
    term_budget : int
        Caller-declared direct work budget.
    failure : GalerkinLocalCellCertificateFailure
        Non-success typed outcome.

    Returns
    -------
    refined : GalerkinLocalCellPotentialRealization
        Direct-route carrier with infinite rectangles and errors.
    """
    coefficient_count = coefficients.shape[0]
    lower: Float64[NDArray, " p"] = np.full(
        (coefficient_count,),
        -np.inf,
        dtype=np.float64,
    )
    upper: Float64[NDArray, " p"] = np.full(
        (coefficient_count,),
        np.inf,
        dtype=np.float64,
    )
    errors: Float64[NDArray, " p"] = np.full(
        (coefficient_count,),
        np.inf,
        dtype=np.float64,
    )
    refined: GalerkinLocalCellPotentialRealization = _make_direct_refinement(
        base,
        indices,
        coefficients,
        lower,
        upper,
        lower,
        upper,
        errors,
        term_count,
        term_budget,
        failure,
    )
    return refined


def _validate_local_cell_certificate_binding(
    realization: GalerkinLocalCellPotentialRealization,
) -> None:
    """PRIVATE: Validate one concrete certificate-to-payload binding.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Submitted direct-route realization and certificate.

    Raises
    ------
    ValueError
        If source/support/static semantics, actual approximant, evidence
        structure, or any canonical digest binding differs.
    equinox.EquinoxRuntimeError
        If a rebuilt source/support or joint evidence predicate fails.

    Notes
    -----
    This checks stored-value identity, not the transcendental derivation.
    Scientific consumers call ``_authenticate_local_cell_certificate``, which
    additionally replays the independent host derivation and exact-compares
    the complete refined carrier.
    """
    _assert_concrete(realization)
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("direct local-cell evidence requires a certificate")
    finite_array = np.asarray(jax.device_get(certificate.finite_certificate))
    term_count_array = np.asarray(
        jax.device_get(certificate.direct_term_count)
    )
    term_budget_array = np.asarray(
        jax.device_get(certificate.maximum_direct_terms)
    )
    if finite_array.dtype != np.dtype(np.bool_) or finite_array.shape != ():
        raise ValueError("finite_certificate must be an exact bool scalar")
    for value, name in (
        (term_count_array, "direct_term_count"),
        (term_budget_array, "maximum_direct_terms"),
    ):
        if value.dtype != np.dtype(np.int64) or value.shape != ():
            raise ValueError(f"{name} must be an exact int64 scalar")
    if (
        realization.error_route
        is not GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
    ):
        raise ValueError("certificate requires the direct local-cell route")
    local_potential, support_result = _canonical_source_and_support(
        realization
    )
    indices, modes, position_by_mode = _checked_host_modes(support_result)
    coefficients = _checked_host_coefficients(
        realization,
        modes,
        position_by_mode,
    )
    base: GalerkinLocalCellPotentialRealization = (
        _create_local_cell_realization(
            local_potential,
            support_result,
            realization.voltage_coefficients,
            jnp.zeros_like(jnp.real(realization.voltage_coefficients)),
        )
    )
    jax.block_until_ready(base)
    structurally_checked = _create_direct_local_cell_realization(
        base,
        realization.coefficient_error_bounds,
        certificate,
    )
    jax.block_until_ready(structurally_checked)

    (
        local_digest,
        support_digest,
        coefficients_digest,
        parent_digest,
    ) = _bound_identity_digests(
        local_potential,
        support_result,
        coefficients,
        indices,
    )
    expected_static = (
        certificate.exact_target == _EXACT_TARGET
        and certificate.arithmetic == _ARITHMETIC
        and certificate.direct_term_count_route == _TERM_COUNT_ROUTE
        and certificate.coefficient_formula == base.coefficient_formula
        and certificate.local_potential_digest == local_digest
        and certificate.requested_support_digest == support_digest
        and certificate.stored_coefficients_digest == coefficients_digest
        and certificate.realization_digest == parent_digest
    )
    if not expected_static:
        raise ValueError("local-cell certificate parent binding is invalid")

    real_lower = np.asarray(
        jax.device_get(certificate.exact_coefficient_real_lower_bounds)
    )
    real_upper = np.asarray(
        jax.device_get(certificate.exact_coefficient_real_upper_bounds)
    )
    imag_lower = np.asarray(
        jax.device_get(certificate.exact_coefficient_imag_lower_bounds)
    )
    imag_upper = np.asarray(
        jax.device_get(certificate.exact_coefficient_imag_upper_bounds)
    )
    errors = np.asarray(jax.device_get(realization.coefficient_error_bounds))
    arrays = (real_lower, real_upper, imag_lower, imag_upper, errors)
    if any(value.dtype != np.dtype(np.float64) for value in arrays):
        raise ValueError("direct certificate arrays must have float64 dtype")
    if any(value.shape != coefficients.shape for value in arrays):
        raise ValueError("direct certificate arrays must match coefficients")
    expected_child = _certificate_digest(
        parent_digest,
        local_digest,
        support_digest,
        coefficients_digest,
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        errors,
        finite_array,
        term_count_array,
        term_budget_array,
        certificate.failure,
        certificate.coefficient_formula,
    )
    if certificate.certificate_digest != expected_child:
        raise ValueError("local-cell certificate digest is invalid")


def _certify_local_cell_galerkin_potential_impl(
    realization: GalerkinLocalCellPotentialRealization,
    *,
    maximum_direct_terms: int = _DEFAULT_MAXIMUM_DIRECT_TERMS,
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Derive direct evidence without recursive authentication.

    Implementation Logic
    --------------------
    1. Reject tracers, rebuild source/support semantics, and validate the
       actual complex128 finite Hermitian payload without replaying its FFT.
    2. Expand only canonical modes whose exact sinc is not symbolically zero;
       derive opposite exact rectangles by ordinary LVT.8 conjugacy.
    3. Compare both stored post-projection pair members independently using a
       verified rational Euclidean farthest-corner radius.
    4. Bind source, requested support, actual approximant bytes, exact formula,
       rectangles, errors, versioned work budget, and outcome by canonical
       digests.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete canonical LVT source/support and actual finite Hermitian
        coefficient approximant. Submitted fallback errors and certificates
        are ignored and recomputed.
    maximum_direct_terms : int
        Maximum canonical-mode--cell terms admitted by the versioned direct
        checker. Symbolic nonzero ``q * N`` sinc zeros expand no cell terms.
        Default: ``2_000_000``.

    Returns
    -------
    refined : GalerkinLocalCellPotentialRealization
        Canonical source/support and unchanged actual coefficient bytes with
        fresh direct exact-target rectangles, Euclidean LVT.13 errors, and
        canonical payload digests.

    Raises
    ------
    ValueError
        If the budget is not a positive signed-64-bit integer, an input is
        traced, source/support/static semantics fail, or actual coefficient
        dtype, shape, finiteness, exact Hermitian pairing, or zero-mode reality
        is invalid.
    equinox.EquinoxRuntimeError
        If independent source/support or joint evidence validation fails.

    Notes
    -----
    This API certifies the submitted approximation point and its error; it
    does not authenticate provenance from one particular eager or compiled
    rounded FFT execution. Host capability, budget, root-enclosure, and
    arithmetic-range failures return typed all-infinite direct
    noncertificates. Triangle fallback errors are never trusted input.
    """
    if (
        isinstance(maximum_direct_terms, bool)
        or not isinstance(maximum_direct_terms, int)
        or maximum_direct_terms <= 0
        or maximum_direct_terms > _MAXIMUM_DIRECT_TERMS
    ):
        raise ValueError(
            "maximum_direct_terms must be a positive signed-64-bit integer"
        )
    _assert_concrete(realization)
    local_potential, support_result = _canonical_source_and_support(
        realization
    )
    indices, modes, position_by_mode = _checked_host_modes(support_result)
    coefficients = _checked_host_coefficients(
        realization,
        modes,
        position_by_mode,
    )
    base: GalerkinLocalCellPotentialRealization = (
        _create_local_cell_realization(
            local_potential,
            support_result,
            realization.voltage_coefficients,
            jnp.zeros_like(jnp.real(realization.voltage_coefficients)),
        )
    )
    jax.block_until_ready(base)

    nz, ny, nx = local_potential.cell_values.shape
    shape_xyz: Tuple[int, int, int] = (nx, ny, nz)
    term_count = _direct_term_count(modes, shape_xyz)
    if term_count > _MAXIMUM_DIRECT_TERMS:
        raise ValueError(
            "exact direct_term_count must fit in signed 64-bit storage"
        )
    if not _host_binary64_supported():
        refined: GalerkinLocalCellPotentialRealization = _failure_refinement(
            base,
            indices,
            coefficients,
            term_count,
            maximum_direct_terms,
            GalerkinLocalCellCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED,
        )
        return refined  # noqa: RET504
    if term_count > maximum_direct_terms:
        refined: GalerkinLocalCellPotentialRealization = _failure_refinement(
            base,
            indices,
            coefficients,
            term_count,
            maximum_direct_terms,
            GalerkinLocalCellCertificateFailure.WORK_BUDGET_EXCEEDED,
        )
        return refined  # noqa: RET504

    cell_values = np.asarray(jax.device_get(local_potential.cell_values))
    if cell_values.dtype != np.dtype(np.float64):
        raise ValueError("LocalCellPotential3D cell_values must be float64")
    try:
        rectangles = _exact_coefficient_rectangles(
            cell_values,
            modes,
            position_by_mode,
            local_potential.cell_center_origin,
            local_potential.box_size,
        )
    except _RootEnclosureError:
        refined: GalerkinLocalCellPotentialRealization = _failure_refinement(
            base,
            indices,
            coefficients,
            term_count,
            maximum_direct_terms,
            GalerkinLocalCellCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        )
        return refined  # noqa: RET504

    error_fractions = [
        _coefficient_euclidean_error_fraction(coefficient, rectangle)
        for coefficient, rectangle in zip(
            coefficients,
            rectangles,
            strict=True,
        )
    ]
    real_lower: Float64[NDArray, " p"] = np.asarray(
        [_fraction_lower_float(value[0]) for value in rectangles],
        dtype=np.float64,
    )
    real_upper: Float64[NDArray, " p"] = np.asarray(
        [_fraction_upper_float(value[1]) for value in rectangles],
        dtype=np.float64,
    )
    imag_lower: Float64[NDArray, " p"] = np.asarray(
        [_fraction_lower_float(value[2]) for value in rectangles],
        dtype=np.float64,
    )
    imag_upper: Float64[NDArray, " p"] = np.asarray(
        [_fraction_upper_float(value[3]) for value in rectangles],
        dtype=np.float64,
    )
    coefficient_errors: Float64[NDArray, " p"] = np.asarray(
        [_fraction_upper_float(value) for value in error_fractions],
        dtype=np.float64,
    )
    endpoint_arrays = (real_lower, real_upper, imag_lower, imag_upper)
    finite_endpoints = all(
        np.all(np.isfinite(value)) for value in endpoint_arrays
    )
    finite_errors = np.all(np.isfinite(coefficient_errors))
    if not finite_endpoints or not finite_errors:
        refined: GalerkinLocalCellPotentialRealization = _failure_refinement(
            base,
            indices,
            coefficients,
            term_count,
            maximum_direct_terms,
            GalerkinLocalCellCertificateFailure.ARITHMETIC_RANGE_FAILURE,
        )
        return refined  # noqa: RET504

    refined: GalerkinLocalCellPotentialRealization = _make_direct_refinement(
        base,
        indices,
        coefficients,
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        coefficient_errors,
        term_count,
        maximum_direct_terms,
        GalerkinLocalCellCertificateFailure.NONE,
    )
    return refined  # noqa: RET504


def _authenticate_local_cell_certificate(
    realization: GalerkinLocalCellPotentialRealization,
) -> GalerkinLocalCellPotentialRealization:
    """PRIVATE: Replay and authenticate one complete direct LVT.13 carrier.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete submitted direct-route realization and certificate.

    Returns
    -------
    canonical : GalerkinLocalCellPotentialRealization
        Fresh canonical replay with exact stored-value identity to the
        submission.

    Raises
    ------
    ValueError
        If binding validation fails or replay changes any declared field,
        coefficient byte, rectangle, Euclidean error, budget, outcome, or
        digest.
    equinox.EquinoxRuntimeError
        If source/support or joint direct-evidence validation fails.

    Notes
    -----
    This re-runs the direct host LVT.7 derivation; it never replays the rounded
    FFT. A successfully authenticated typed noncertificate still has infinite
    errors and is not finite evidence for L3 or LVT.9.
    """
    _validate_local_cell_certificate_binding(realization)
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("direct local-cell evidence requires a certificate")
    term_budget_array = np.asarray(
        jax.device_get(certificate.maximum_direct_terms)
    )
    if (
        term_budget_array.dtype != np.dtype(np.int64)
        or term_budget_array.shape != ()
    ):
        raise ValueError("maximum_direct_terms must be an exact int64 scalar")
    term_budget = int(term_budget_array)
    canonical: GalerkinLocalCellPotentialRealization = (
        _certify_local_cell_galerkin_potential_impl(
            realization,
            maximum_direct_terms=term_budget,
        )
    )
    _validate_local_cell_certificate_binding(canonical)
    submitted_payload = _stored_value_payload(realization)
    canonical_payload = _stored_value_payload(canonical)
    if submitted_payload != canonical_payload:
        raise ValueError(
            "local-cell direct certificate does not exactly match host replay"
        )
    return canonical


@jaxtyped(typechecker=beartype)
def certify_local_cell_galerkin_potential(
    realization: GalerkinLocalCellPotentialRealization,
    *,
    maximum_direct_terms: int = _DEFAULT_MAXIMUM_DIRECT_TERMS,
) -> GalerkinLocalCellPotentialRealization:
    """Certify an actual Hermitian approximant directly against LVT.7.

    :see: :class:`~.test_local_cell_certification.\
TestLocalCellDirectCertification`

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete canonical LVT source/support and actual finite exact-Hermitian
        coefficient approximant. Submitted fallback errors and certificates
        are ignored and recomputed.
    maximum_direct_terms : int, optional
        Maximum canonical-mode--cell terms admitted by the versioned direct
        checker. Symbolic nonzero ``q * N`` sinc zeros expand no cell terms.
        Default: ``2_000_000``.

    Returns
    -------
    refined : GalerkinLocalCellPotentialRealization
        Canonical source/support and unchanged actual coefficient bytes with
        fresh direct exact-target rectangles, Euclidean LVT.13 errors, and
        canonical payload digests.

    Raises
    ------
    ValueError
        If budget, tracer, source/support/static semantics, or actual
        coefficient dtype, shape, finiteness, Hermitian pairing, or zero-mode
        reality is invalid.
    equinox.EquinoxRuntimeError
        If independent source/support or joint evidence validation fails.

    Notes
    -----
    This certifies the submitted approximation point and error, not provenance
    from one eager or compiled rounded FFT execution. Typed host, budget,
    root, and range failures retain infinite direct noncertificates. The
    private replay authenticator is required before later scientific use.
    This bounded host oracle is not a scalable replacement for the rounded
    production coefficient map.
    """
    refined: GalerkinLocalCellPotentialRealization = (
        _certify_local_cell_galerkin_potential_impl(
            realization,
            maximum_direct_terms=maximum_direct_terms,
        )
    )
    _validate_local_cell_certificate_binding(refined)
    return refined


__all__: list[str] = ["certify_local_cell_galerkin_potential"]
