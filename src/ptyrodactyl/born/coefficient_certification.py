r"""Certify stored VC-1 coefficients against a direct exact host target.

Extended Summary
----------------
This module refines one concrete :class:`GalerkinPotentialRealization` with
componentwise VC.17 error bounds.  It never assumes an error model for the
production FFT, origin-phase exponential, or Hermitian projection.  Instead,
it independently encloses the exact periodic-trigonometric coefficient by a
bounded-memory direct sum and measures the final stored coefficient against
that rectangle.

Routine Listings
----------------
:func:`certify_galerkin_potential_realization`
    Refine one concrete realization with direct pairwise host evidence.

Notes
-----
All host certificate arithmetic before the final outward binary64 conversion
uses exact :class:`fractions.Fraction` values.  Mathematical pi is enclosed by
Machin's formula and rational alternating arctangent series.  Rational-turn
sines and cosines are enclosed by alternating Taylor series; this module does
not call a library trigonometric function or trust the production FFT.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Tuple
from jax.core import Tracer
from jaxtyping import Float64, Int64, jaxtyped
from numpy.typing import NDArray

# Retain the complete legacy private surface as identity aliases.
from ptyrodactyl._host_interval import (  # noqa: F401
    _BINARY64_MAX_EXPONENT,
    _BINARY64_MIN_EXPONENT,
    _BINARY64_RADIX,
    _BINARY64_SIGNIFICAND_BITS,
    _HALF_TURN_QUADRANT,
    _MINIMUM_NORMAL,
    _PI_TARGET_BITS,
    _QUADRANT_COUNT,
    _TAYLOR_LOWER_LAST_INDEX,
    _TAYLOR_UPPER_LAST_INDEX,
    _THREE_QUARTER_TURN_QUADRANT,
    _atan_inverse_bounds,
    _coefficient_error_fraction,
    _complex_rectangle_add,
    _complex_rectangle_multiply,
    _ComplexRectangle,
    _conjugate_rectangle,
    _cosine_partial,
    _first_quadrant_sine_cosine,
    _floor_log2_fraction,
    _fraction_from_float,
    _fraction_lower_float,
    _fraction_upper_float,
    _host_binary64_supported,
    _negate_interval,
    _normal_floor_lower,
    _normal_floor_upper,
    _pairwise_rectangle_sum,
    _pi_bounds,
    _power_of_two_fraction,
    _rational_turn_exponential,
    _real_interval_add,
    _real_interval_product,
    _real_interval_subtract,
    _RealInterval,
    _RootEnclosureError,
    _scale_complex_rectangle,
    _sine_partial,
    _sqrt_fraction_upper,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
    GalerkinPotentialCertificateFailure,
    GalerkinPotentialCoefficientCertificate,
    GalerkinPotentialErrorRoute,
    GalerkinPotentialRealization,
    create_galerkin_potential_coefficient_certificate,
)

from .acquisition import check_galerkin_acquisition_support

_ARITHMETIC: str = (
    "guarded IEEE binary64 host; exact Fraction direct DFT; Machin rational "
    "pi; alternating rational sin/cos; binary pairwise accumulation; "
    "outward binary64 endpoints"
)
_EXACT_TARGET: str = "VC.8 periodic trigonometric interpolant"
_DEFAULT_MAXIMUM_DIRECT_TERMS: int = 2_000_000
_MAXIMUM_DIRECT_TERMS: int = np.iinfo(np.int64).max


def _axis_phase_rectangles(
    mode: int,
    size: int,
    cache: Dict[Fraction, _ComplexRectangle],
) -> Tuple[_ComplexRectangle, ...]:
    """PRIVATE: Enclose all roots for one signed mode and array axis.

    Parameters
    ----------
    mode : int
        Signed integer Fourier mode on this axis.
    size : int
        Positive integer grid extent on this axis.
    cache : Dict[Fraction, _ComplexRectangle]
        Mutable exact-turn phase cache shared across coefficients.

    Returns
    -------
    result : Tuple[_ComplexRectangle, ...]
        Ordered phase-factor rectangles for every axis position.
    """
    values: list[_ComplexRectangle] = []
    for position in range(size):
        turn = Fraction(mode * position, size) % 1
        if turn not in cache:
            cache[turn] = _rational_turn_exponential(turn)
        values.append(cache[turn])
    result: Tuple[_ComplexRectangle, ...] = tuple(values)
    return result


def _exact_coefficient_rectangle(
    volume: Float64[NDArray, "nz ny nx"],
    mode_xyz: Tuple[int, int, int],
    origin_xyz: Tuple[float, float, float],
    box_xyz: Tuple[float, float, float],
    phase_cache: Dict[Fraction, _ComplexRectangle],
) -> _ComplexRectangle:
    """PRIVATE: Enclose one exact VC.8 coefficient by direct summation.

    Parameters
    ----------
    volume : Float64[NDArray, "nz ny nx"]
        Exact stored binary64 voxel array on the host.
    mode_xyz : Tuple[int, int, int]
        Signed Fourier index in physical ``(x, y, z)`` order.
    origin_xyz : Tuple[float, float, float]
        Exact stored binary64 origin in physical axis order.
    box_xyz : Tuple[float, float, float]
        Exact stored binary64 box lengths in physical axis order.
    phase_cache : Dict[Fraction, _ComplexRectangle]
        Mutable rational-turn phase cache shared across coefficients.

    Returns
    -------
    result : _ComplexRectangle
        Exact-rational rectangle enclosing the requested coefficient.
    """
    nz, ny, nx = volume.shape
    mode_x, mode_y, mode_z = mode_xyz
    x_phases = _axis_phase_rectangles(mode_x, nx, phase_cache)
    y_phases = _axis_phase_rectangles(mode_y, ny, phase_cache)
    z_phases = _axis_phase_rectangles(mode_z, nz, phase_cache)
    origin_turn = sum(
        (
            mode * _fraction_from_float(origin) / _fraction_from_float(length)
            for mode, origin, length in zip(
                mode_xyz,
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
    origin_phase = phase_cache[reduced_origin_turn]

    def direct_terms() -> Iterator[_ComplexRectangle]:
        """PRIVATE: Yield one exact-rational interval term at a time.

        Yields
        ------
        term : _ComplexRectangle
            Next voxel-weighted phase rectangle in storage order.
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
                    voxel = _fraction_from_float(
                        float(volume[z_position, y_position, x_position])
                    )
                    term: _ComplexRectangle = _scale_complex_rectangle(
                        grid_phase,
                        voxel,
                    )
                    yield term

    accumulated = _pairwise_rectangle_sum(direct_terms())
    normalized = _scale_complex_rectangle(
        accumulated,
        Fraction(1, volume.size),
    )
    result: _ComplexRectangle = _complex_rectangle_multiply(
        origin_phase,
        normalized,
    )
    return result


def _is_canonical_mode(mode: Tuple[int, int, int]) -> bool:
    """PRIVATE: Select zero and one lexicographically positive signed mode.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Signed integer Fourier index.

    Returns
    -------
    result : bool
        Whether the mode is the canonical member of its signed pair.
    """
    first, second, third = mode
    result: bool = (first > 0) or (
        first == 0 and (second > 0 or (second == 0 and third >= 0))
    )
    return result


def _mode_tuple(row: Int64[NDArray, " 3"]) -> Tuple[int, int, int]:
    """PRIVATE: Convert one checked three-component host index row.

    Parameters
    ----------
    row : Int64[NDArray, " 3"]
        Three-component signed host index row.

    Returns
    -------
    first : int
        First signed Python integer component.
    second : int
        Second signed Python integer component.
    third : int
        Third signed Python integer component.
    """
    first: int = int(row[0])
    second: int = int(row[1])
    third: int = int(row[2])
    result: Tuple[int, int, int] = (first, second, third)
    return result


def _exact_coefficient_rectangles(
    volume: Float64[NDArray, "nz ny nx"],
    indices: Int64[NDArray, "p 3"],
    origin_xyz: Tuple[float, float, float],
    box_xyz: Tuple[float, float, float],
) -> list[_ComplexRectangle]:
    """PRIVATE: Enclose ordered support coefficients using signed symmetry.

    Parameters
    ----------
    volume : Float64[NDArray, "nz ny nx"]
        Exact stored binary64 voxel array on the host.
    indices : Int64[NDArray, "p 3"]
        Ordered sign-symmetric interaction indices.
    origin_xyz : Tuple[float, float, float]
        Exact stored binary64 origin in physical axis order.
    box_xyz : Tuple[float, float, float]
        Exact stored binary64 box lengths in physical axis order.

    Returns
    -------
    result : list[_ComplexRectangle]
        Coefficient rectangles in the submitted support order.

    Raises
    ------
    ValueError
        If a signed mode lacks its opposite partner.
    """
    ordered_modes = [_mode_tuple(row) for row in indices]
    position_by_mode = {
        mode: position for position, mode in enumerate(ordered_modes)
    }
    rectangles: list[_ComplexRectangle | None] = [None] * len(ordered_modes)
    phase_cache: Dict[Fraction, _ComplexRectangle] = {}
    for position, mode in enumerate(ordered_modes):
        if not _is_canonical_mode(mode):
            continue
        rectangle = _exact_coefficient_rectangle(
            volume,
            mode,
            origin_xyz,
            box_xyz,
            phase_cache,
        )
        rectangles[position] = rectangle
        opposite = (-mode[0], -mode[1], -mode[2])
        opposite_position = position_by_mode.get(opposite)
        if opposite_position is None:
            raise ValueError("interaction support must be sign symmetric")
        rectangles[opposite_position] = _conjugate_rectangle(rectangle)
    if any(rectangle is None for rectangle in rectangles):
        raise ValueError("interaction support has an unpaired signed mode")
    result: list[_ComplexRectangle] = [
        rectangle for rectangle in rectangles if rectangle is not None
    ]
    return result


def _voltage_operator_error_fraction(
    state_indices: Int64[NDArray, "n 3"],
    interaction_indices: Int64[NDArray, "p 3"],
    coefficient_errors: list[Fraction],
) -> Fraction:
    """PRIVATE: Derive the multiplicity-aware Frobenius/Schur minimum.

    Parameters
    ----------
    state_indices : Int64[NDArray, "n 3"]
        Ordered finite state support.
    interaction_indices : Int64[NDArray, "p 3"]
        Ordered represented interaction support.
    coefficient_errors : list[Fraction]
        Exact non-negative error bounds in interaction-support order.

    Returns
    -------
    result : Fraction
        Exact dyadic upper bound on the compressed multiplier error norm.
    """
    error_by_mode = {
        tuple(int(value) for value in mode): error
        for mode, error in zip(
            interaction_indices,
            coefficient_errors,
            strict=True,
        )
    }
    multiplicities = dict.fromkeys(error_by_mode, 0)
    state_count = state_indices.shape[0]
    row_sums = [Fraction(0) for _ in range(state_count)]
    column_sums = [Fraction(0) for _ in range(state_count)]
    for row in range(state_count):
        for column in range(state_count):
            difference = tuple(
                int(state_indices[row, axis] - state_indices[column, axis])
                for axis in range(3)
            )
            error = error_by_mode.get(difference, Fraction(0))
            row_sums[row] += error
            column_sums[column] += error
            if difference in multiplicities:
                multiplicities[difference] += 1
    frobenius_squared = sum(
        (
            multiplicities[mode] * error * error
            for mode, error in error_by_mode.items()
        ),
        start=Fraction(0),
    )
    maximum_row = max(row_sums, default=Fraction(0))
    maximum_column = max(column_sums, default=Fraction(0))
    schur_squared = maximum_row * maximum_column
    frobenius_upper = _sqrt_fraction_upper(frobenius_squared)
    schur_upper = _sqrt_fraction_upper(schur_squared)
    result: Fraction = min(frobenius_upper, schur_upper)
    return result


def _assert_concrete(realization: GalerkinPotentialRealization) -> None:
    """PRIVATE: Reject traced evidence at the explicit host boundary.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Submitted realization PyTree.

    Raises
    ------
    ValueError
        If any dynamic realization leaf is a JAX tracer.
    """
    leaves = jax.tree_util.tree_leaves(realization)
    if any(isinstance(leaf, Tracer) for leaf in leaves):
        raise ValueError(
            "direct coefficient certification requires concrete host values"
        )


def _independently_recheck_acquisition(
    submitted: GalerkinAcquisitionSupportResult,
    potential_box_size: Tuple[float, float, float],
) -> None:
    """PRIVATE: Recheck the artifact and its exact potential-box binding.

    Parameters
    ----------
    submitted : GalerkinAcquisitionSupportResult
        Submitted acquisition-support evidence.
    potential_box_size : Tuple[float, float, float]
        Exact stored binary64 potential box lengths.

    Raises
    ------
    ValueError
        If the artifact is noncanonical, ineligible, or bound to another box.
    """
    canonical = check_galerkin_acquisition_support(submitted.manifest)
    submitted_leaves, submitted_structure = jax.tree_util.tree_flatten(
        submitted
    )
    canonical_leaves, canonical_structure = jax.tree_util.tree_flatten(
        canonical
    )
    if submitted_structure != canonical_structure:
        raise ValueError("acquisition certificate has noncanonical structure")
    if len(submitted_leaves) != len(canonical_leaves):
        raise ValueError("acquisition certificate has noncanonical leaves")
    for submitted_leaf, canonical_leaf in zip(
        submitted_leaves,
        canonical_leaves,
        strict=True,
    ):
        submitted_array = np.asarray(jax.device_get(submitted_leaf))
        canonical_array = np.asarray(jax.device_get(canonical_leaf))
        if (
            submitted_array.dtype != canonical_array.dtype
            or submitted_array.shape != canonical_array.shape
            or not np.array_equal(submitted_array, canonical_array)
        ):
            raise ValueError(
                "acquisition certificate must exactly match an independent "
                "recheck"
            )
    canonical_status = int(np.asarray(jax.device_get(canonical.status)))
    canonical_eligible = bool(
        np.asarray(jax.device_get(canonical.support_eligible))
    )
    if (
        canonical_status
        != int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
        or not canonical_eligible
    ):
        raise ValueError("acquisition support is not SUPPORT_ELIGIBLE")
    acquisition_box = np.asarray(
        jax.device_get(canonical.manifest.box_lengths),
        dtype=np.float64,
    )
    potential_box = np.asarray(potential_box_size, dtype=np.float64)
    if not np.array_equal(acquisition_box, potential_box):
        raise ValueError(
            "acquisition box lengths must exactly match the potential box"
        )


def _independently_recheck_vc1_domain(
    realization: GalerkinPotentialRealization,
) -> None:
    """PRIVATE: Recheck boundary, endpoints, zero, and strict band.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Concrete realization whose VC-1 domain is checked.

    Raises
    ------
    ValueError
        If periodicity, signed endpoints, zero retention, or band fails.
    """
    potential = realization.potential
    if potential.boundary != "periodic":
        raise ValueError("VC-1 certificate requires a periodic potential")
    indices = np.asarray(
        jax.device_get(realization.support.interaction_indices),
        dtype=np.int64,
    )
    if not np.any(np.all(indices == 0, axis=1)):
        raise ValueError(
            "interaction support must retain the voltage zero mode"
        )
    nz, ny, nx = potential.volume.shape
    shape_xyz = (nx, ny, nz)
    if any(
        2 * abs(int(mode[axis])) >= size
        for mode in indices
        for axis, size in enumerate(shape_xyz)
    ):
        raise ValueError(
            "interaction support leaves the signed VC-1 grid endpoints"
        )
    box_fractions = tuple(
        _fraction_from_float(value) for value in potential.box_size
    )
    band_squared = _fraction_from_float(potential.band_limit) ** 2
    for mode in indices:
        frequency_squared = sum(
            (
                Fraction(int(mode[axis]), 1) ** 2 / (box_fractions[axis] ** 2)
                for axis in range(3)
            ),
            start=Fraction(0),
        )
        if frequency_squared >= band_squared:
            raise ValueError(
                "interaction support must stay inside the strict potential "
                "band"
            )


def _failure_certificate(
    realization: GalerkinPotentialRealization,
    term_count: int,
    term_budget: int,
    failure: GalerkinPotentialCertificateFailure,
) -> GalerkinPotentialRealization:
    """PRIVATE: Return one typed infinite direct-route noncertificate.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Concrete source realization to preserve and refine.
    term_count : int
        Requested direct coefficient and state-transfer work count.
    term_budget : int
        Positive caller-declared direct work budget.
    failure : GalerkinPotentialCertificateFailure
        Typed non-success outcome to attach.

    Returns
    -------
    refined : GalerkinPotentialRealization
        Preserved realization with infinite stopped error evidence.
    """
    coefficient_count = realization.voltage_coefficients.shape[0]
    lower = jnp.full((coefficient_count,), -jnp.inf, dtype=jnp.float64)
    upper = jnp.full((coefficient_count,), jnp.inf, dtype=jnp.float64)
    certificate = create_galerkin_potential_coefficient_certificate(
        lower,
        upper,
        lower,
        upper,
        jnp.asarray(False),
        jnp.asarray(term_count, dtype=jnp.int64),
        jnp.asarray(term_budget, dtype=jnp.int64),
        failure=failure,
        exact_target=_EXACT_TARGET,
        arithmetic=_ARITHMETIC,
    )
    refined: GalerkinPotentialRealization = dataclasses.replace(
        realization,
        coefficient_error_bounds=jax.lax.stop_gradient(
            jnp.full((coefficient_count,), jnp.inf, dtype=jnp.float64)
        ),
        voltage_operator_error_bound=jax.lax.stop_gradient(
            jnp.asarray(jnp.inf, dtype=jnp.float64)
        ),
        error_route=GalerkinPotentialErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL,
        coefficient_certificate=certificate,
    )
    return refined


@jaxtyped(typechecker=beartype)
def certify_galerkin_potential_realization(
    realization: GalerkinPotentialRealization,
    *,
    maximum_direct_terms: int = _DEFAULT_MAXIMUM_DIRECT_TERMS,
) -> GalerkinPotentialRealization:
    """Refine one concrete realization with direct pairwise host evidence.

    :see: :class:`~.test_coefficient_certification.\
TestDirectCoefficientCertification`

    Implementation Logic
    --------------------
    1. Reject tracers and independently recheck every acquisition artifact.
    2. Enclose one exact direct VC.8 sum per canonical signed mode.
    3. Derive opposite modes by exact conjugacy and compare the final stored
       production coefficients with the resulting rectangles.
    4. Preserve the source and coefficient leaves while replacing only
       stopped error evidence and its typed route.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Concrete source, checked support, and arbitrary finite stored
        production coefficients.
    maximum_direct_terms : int, optional
        Maximum number of canonical coefficient--voxel terms admitted by the
        bounded direct checker. Default: ``2_000_000``.

    Returns
    -------
    refined : GalerkinPotentialRealization
        The same potential and production coefficient leaves with direct
        exact-target rectangles and outward VC.17 bounds.

    Raises
    ------
    ValueError
        If the budget is not a positive signed-64-bit integer, an input is
        traced, or the acquisition artifact fails independent exact recheck.

    Notes
    -----
    Exceeding ``maximum_direct_terms`` returns a typed infinite
    noncertificate rather than raising or reverting to the triangle route.
    The coefficient bound uses the complex L1 distance from the stored point
    to the exact rectangle, which soundly bounds Euclidean magnitude without
    a floating square root.
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
    _independently_recheck_acquisition(
        realization.support_eligibility,
        realization.potential.box_size,
    )
    indices = np.asarray(
        jax.device_get(realization.support.interaction_indices),
        dtype=np.int64,
    )
    canonical_count = sum(
        _is_canonical_mode(_mode_tuple(row)) for row in indices
    )
    state_count = realization.support.state_indices.shape[0]
    term_count = (
        canonical_count * realization.potential.volume.size
        + state_count * state_count
    )
    if not _host_binary64_supported():
        refined: GalerkinPotentialRealization = _failure_certificate(
            realization,
            term_count,
            maximum_direct_terms,
            GalerkinPotentialCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED,
        )
        return refined
    _independently_recheck_vc1_domain(realization)
    if term_count > maximum_direct_terms:
        refined = _failure_certificate(
            realization,
            term_count,
            maximum_direct_terms,
            GalerkinPotentialCertificateFailure.WORK_BUDGET_EXCEEDED,
        )
        return refined  # noqa: RET504

    volume = np.asarray(
        jax.device_get(realization.potential.volume),
        dtype=np.float64,
    )
    coefficients = np.asarray(
        jax.device_get(realization.voltage_coefficients),
        dtype=np.complex128,
    )

    try:
        rectangles = _exact_coefficient_rectangles(
            volume,
            indices,
            realization.potential.origin,
            realization.potential.box_size,
        )
    except _RootEnclosureError:
        refined = _failure_certificate(
            realization,
            term_count,
            maximum_direct_terms,
            GalerkinPotentialCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        )
        return refined  # noqa: RET504

    error_fractions = [
        _coefficient_error_fraction(coefficient, rectangle)
        for coefficient, rectangle in zip(
            coefficients,
            rectangles,
            strict=True,
        )
    ]
    real_lower = np.asarray(
        [_fraction_lower_float(value[0]) for value in rectangles],
        dtype=np.float64,
    )
    real_upper = np.asarray(
        [_fraction_upper_float(value[1]) for value in rectangles],
        dtype=np.float64,
    )
    imag_lower = np.asarray(
        [_fraction_lower_float(value[2]) for value in rectangles],
        dtype=np.float64,
    )
    imag_upper = np.asarray(
        [_fraction_upper_float(value[3]) for value in rectangles],
        dtype=np.float64,
    )
    coefficient_errors = np.asarray(
        [_fraction_upper_float(value) for value in error_fractions],
        dtype=np.float64,
    )
    endpoint_arrays = (real_lower, real_upper, imag_lower, imag_upper)
    finite_endpoints = all(
        np.all(np.isfinite(value)) for value in endpoint_arrays
    )
    finite_errors = np.all(np.isfinite(coefficient_errors))
    if not finite_endpoints or not finite_errors:
        refined = _failure_certificate(
            realization,
            term_count,
            maximum_direct_terms,
            GalerkinPotentialCertificateFailure.ARITHMETIC_RANGE_FAILURE,
        )
        return refined  # noqa: RET504

    state_indices = np.asarray(
        jax.device_get(realization.support.state_indices),
        dtype=np.int64,
    )
    operator_error_fraction = _voltage_operator_error_fraction(
        state_indices,
        indices,
        error_fractions,
    )
    operator_error = _fraction_upper_float(operator_error_fraction)
    if not math.isfinite(operator_error):
        refined = _failure_certificate(
            realization,
            term_count,
            maximum_direct_terms,
            GalerkinPotentialCertificateFailure.ARITHMETIC_RANGE_FAILURE,
        )
        return refined  # noqa: RET504

    certificate: GalerkinPotentialCoefficientCertificate = (
        create_galerkin_potential_coefficient_certificate(
            jnp.asarray(real_lower, dtype=jnp.float64),
            jnp.asarray(real_upper, dtype=jnp.float64),
            jnp.asarray(imag_lower, dtype=jnp.float64),
            jnp.asarray(imag_upper, dtype=jnp.float64),
            jnp.asarray(True),
            jnp.asarray(term_count, dtype=jnp.int64),
            jnp.asarray(maximum_direct_terms, dtype=jnp.int64),
            failure=GalerkinPotentialCertificateFailure.NONE,
            exact_target=_EXACT_TARGET,
            arithmetic=_ARITHMETIC,
        )
    )
    refined: GalerkinPotentialRealization = dataclasses.replace(
        realization,
        coefficient_error_bounds=jax.lax.stop_gradient(
            jnp.asarray(coefficient_errors, dtype=jnp.float64)
        ),
        voltage_operator_error_bound=jax.lax.stop_gradient(
            jnp.asarray(operator_error, dtype=jnp.float64)
        ),
        error_route=GalerkinPotentialErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL,
        coefficient_certificate=certificate,
    )
    return refined


__all__: list[str] = ["certify_galerkin_potential_realization"]
