r"""Share bounded-memory direct complex interval contractions.

Extended Summary
----------------
This private module evaluates compressed multiplier actions from exact stored
binary64 coefficients without materializing a dense matrix.  In parallel it
propagates FTZ-safe outward complex rectangles through the same exact integer
coefficient lookup.  The helpers are evidence paths, not model-gradient
primitives.

Notes
-----
All real interval arithmetic is delegated to :mod:`ptyrodactyl._interval`.
The lookup first matches an endpoint-safe quotient key and then checks exact
integer equality, so a modular residue collision cannot become a physical
coefficient match.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex128, Float64, Int64

from ptyrodactyl._interval import (
    _downward_divide,
    _downward_sqrt,
    _interval_add,
    _interval_multiply,
    _interval_square,
    _interval_subtract,
    _point_interval,
    _RealInterval,
    _upward_add,
    _upward_divide,
    _upward_multiply,
    _upward_sqrt,
)
from ptyrodactyl.types import scalar_int

type _ComplexInterval = Tuple[
    Float64[Array, "..."],
    Float64[Array, "..."],
    Float64[Array, "..."],
    Float64[Array, "..."],
]
type _MultiplierEvaluation = Tuple[
    Complex128[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
]


def _complex_point_interval(
    value: Complex128[Array, "..."],
) -> _ComplexInterval:
    """PRIVATE: Embed exact stored complex binary64 values as rectangles.

    Parameters
    ----------
    value : Complex128[Array, "..."]
        Exact stored complex values in a caller-defined unit.

    Returns
    -------
    interval : _ComplexInterval
        Inclusive real-lower, real-upper, imaginary-lower, and
        imaginary-upper arrays in the input unit.
    """
    real_interval: _RealInterval = _point_interval(jnp.real(value))
    imag_interval: _RealInterval = _point_interval(jnp.imag(value))
    interval: _ComplexInterval = (
        real_interval[0],
        real_interval[1],
        imag_interval[0],
        imag_interval[1],
    )
    return interval


def _complex_interval_add(
    left: _ComplexInterval,
    right: _ComplexInterval,
) -> _ComplexInterval:
    """PRIVATE: Add two inclusive complex rectangles outwardly.

    Parameters
    ----------
    left : _ComplexInterval
        Left rectangle in a caller-defined unit.
    right : _ComplexInterval
        Right rectangle in the same unit.

    Returns
    -------
    result : _ComplexInterval
        Inclusive outward sum rectangle.
    """
    real: _RealInterval = _interval_add(
        (left[0], left[1]), (right[0], right[1])
    )
    imag: _RealInterval = _interval_add(
        (left[2], left[3]), (right[2], right[3])
    )
    result: _ComplexInterval = (real[0], real[1], imag[0], imag[1])
    return result


def _complex_interval_subtract(
    left: _ComplexInterval,
    right: _ComplexInterval,
) -> _ComplexInterval:
    """PRIVATE: Subtract two inclusive complex rectangles outwardly.

    Parameters
    ----------
    left : _ComplexInterval
        Left rectangle in a caller-defined unit.
    right : _ComplexInterval
        Right rectangle in the same unit.

    Returns
    -------
    result : _ComplexInterval
        Inclusive outward difference rectangle.
    """
    real: _RealInterval = _interval_subtract(
        (left[0], left[1]), (right[0], right[1])
    )
    imag: _RealInterval = _interval_subtract(
        (left[2], left[3]), (right[2], right[3])
    )
    result: _ComplexInterval = (real[0], real[1], imag[0], imag[1])
    return result


def _complex_interval_multiply(
    left: _ComplexInterval,
    right: _ComplexInterval,
) -> _ComplexInterval:
    """PRIVATE: Multiply two inclusive complex rectangles outwardly.

    Parameters
    ----------
    left : _ComplexInterval
        Left rectangle in a caller-defined unit.
    right : _ComplexInterval
        Right rectangle in a caller-defined unit.

    Returns
    -------
    result : _ComplexInterval
        Inclusive product rectangle in the product unit.
    """
    left_real: _RealInterval = (left[0], left[1])
    left_imag: _RealInterval = (left[2], left[3])
    right_real: _RealInterval = (right[0], right[1])
    right_imag: _RealInterval = (right[2], right[3])
    real: _RealInterval = _interval_subtract(
        _interval_multiply(left_real, right_real),
        _interval_multiply(left_imag, right_imag),
    )
    imag: _RealInterval = _interval_add(
        _interval_multiply(left_real, right_imag),
        _interval_multiply(left_imag, right_real),
    )
    result: _ComplexInterval = (real[0], real[1], imag[0], imag[1])
    return result


def _complex_interval_conjugate(
    value: _ComplexInterval,
) -> _ComplexInterval:
    """PRIVATE: Conjugate one inclusive complex rectangle exactly.

    Parameters
    ----------
    value : _ComplexInterval
        Input complex rectangle.

    Returns
    -------
    result : _ComplexInterval
        Rectangle with its imaginary endpoints sign-reversed and swapped.
    """
    result: _ComplexInterval = (value[0], value[1], -value[3], -value[2])
    return result


def _coefficient_at_difference(
    indices: Int64[Array, "p 3"],
    coefficients: Complex128[Array, " p"],
    difference: Int64[Array, " 3"],
    sorted_keys: Int64[Array, " p"],
    order: Int64[Array, " p"],
    work_shape: Tuple[int, int, int],
) -> Complex128[Array, ""]:
    """PRIVATE: Select one exact-difference coefficient or symbolic zero.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Exact multiplier indices in fixed order.
    coefficients : Complex128[Array, " p"]
        Exact stored multiplier coefficients.
    difference : Int64[Array, " 3"]
        Requested exact state-index difference.
    sorted_keys : Int64[Array, " p"]
        Sorted mixed-radix work-quotient keys.
    order : Int64[Array, " p"]
        Permutation from sorted keys to coefficient order.
    work_shape : Tuple[int, int, int]
        Endpoint-safe quotient dimensions.

    Returns
    -------
    coefficient : Complex128[Array, ""]
        Matching coefficient, or exact stored zero when absent.
    """
    moduli: Int64[Array, " 3"] = jnp.asarray(work_shape, dtype=jnp.int64)
    residue: Int64[Array, " 3"] = jnp.mod(difference, moduli)
    key: Int64[Array, ""] = (
        residue[0] * work_shape[1] + residue[1]
    ) * work_shape[2] + residue[2]
    location: Int64[Array, ""] = jnp.searchsorted(
        sorted_keys, key, side="left"
    )
    clipped: Int64[Array, ""] = jnp.clip(location, 0, indices.shape[0] - 1)
    exact_match: Bool[Array, ""] = (
        (location < indices.shape[0])
        & (sorted_keys[clipped] == key)
        & jnp.all(indices[order[clipped]] == difference)
    )
    coefficient: Complex128[Array, ""] = jnp.where(
        exact_match,
        coefficients[order[clipped]],
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )
    return coefficient


def _direct_multiplier_with_interval(
    state_indices: Int64[Array, "n 3"],
    multiplier_indices: Int64[Array, "p 3"],
    coefficients: Complex128[Array, " p"],
    field: Complex128[Array, " n"],
    work_shape: Tuple[int, int, int],
    *,
    adjoint: bool,
) -> _MultiplierEvaluation:
    """PRIVATE: Apply and enclose a compressed multiplier with bounded memory.

    Parameters
    ----------
    state_indices : Int64[Array, "n 3"]
        Exact retained state indices.
    multiplier_indices : Int64[Array, "p 3"]
        Exact multiplier support.
    coefficients : Complex128[Array, " p"]
        Exact stored multiplier coefficients.
    field : Complex128[Array, " n"]
        Exact stored input coefficients.
    work_shape : Tuple[int, int, int]
        Endpoint-safe quotient dimensions.
    adjoint : bool
        If true, evaluate the actual conjugate-transpose multiplier.

    Returns
    -------
    result : _MultiplierEvaluation
        Rounded direct action followed by inclusive real and imaginary
        interval endpoints for its exact-real stored-data target.

    Notes
    -----
    The flattened row-column loop uses ``O(n+p)`` live storage.  It never
    materializes an ``n`` by ``n`` coefficient matrix.
    """
    state_size: int = state_indices.shape[0]
    product_count: int = state_size * state_size
    moduli: Int64[Array, " 3"] = jnp.asarray(work_shape, dtype=jnp.int64)
    residues: Int64[Array, "p 3"] = jnp.mod(multiplier_indices, moduli)
    keys: Int64[Array, " p"] = (
        residues[:, 0] * work_shape[1] + residues[:, 1]
    ) * work_shape[2] + residues[:, 2]
    order: Int64[Array, " p"] = jnp.argsort(keys)
    sorted_keys: Int64[Array, " p"] = keys[order]
    work_size: int = work_shape[0] * work_shape[1] * work_shape[2]
    quotient_invalid: Bool[Array, ""] = (
        jnp.any(keys < 0)
        | jnp.any(keys >= work_size)
        | jnp.any(sorted_keys[1:] == sorted_keys[:-1])
    )
    checked_coefficients: Complex128[Array, " p"] = eqx.error_if(
        coefficients,
        quotient_invalid,
        "multiplier support must remain unique in its signed-64-bit work "
        "quotient",
    )
    zeros: Float64[Array, " n"] = jnp.zeros((state_size,), dtype=jnp.float64)
    initial: _MultiplierEvaluation = (
        jnp.zeros((state_size,), dtype=jnp.complex128),
        zeros,
        zeros,
        zeros,
        zeros,
    )

    def add_entry(
        flat_position: scalar_int,
        accumulator: _MultiplierEvaluation,
    ) -> _MultiplierEvaluation:
        """Accumulate one exact matrix entry and its outward rectangle."""
        row: scalar_int = flat_position // state_size
        column: scalar_int = flat_position % state_size
        forward_difference: Int64[Array, " 3"] = (
            state_indices[row] - state_indices[column]
        )
        requested_difference: Int64[Array, " 3"] = jnp.where(
            adjoint,
            -forward_difference,
            forward_difference,
        )
        raw_coefficient: Complex128[Array, ""] = _coefficient_at_difference(
            multiplier_indices,
            checked_coefficients,
            requested_difference,
            sorted_keys,
            order,
            work_shape,
        )
        coefficient: Complex128[Array, ""] = jnp.where(
            adjoint,
            jnp.conj(raw_coefficient),
            raw_coefficient,
        )
        product: Complex128[Array, ""] = coefficient * field[column]
        rounded: Complex128[Array, " n"] = accumulator[0].at[row].add(product)
        exact_product: _ComplexInterval = _complex_interval_multiply(
            _complex_point_interval(coefficient),
            _complex_point_interval(field[column]),
        )
        prior: _ComplexInterval = (
            accumulator[1][row],
            accumulator[2][row],
            accumulator[3][row],
            accumulator[4][row],
        )
        updated: _ComplexInterval = _complex_interval_add(prior, exact_product)
        result: _MultiplierEvaluation = (
            rounded,
            accumulator[1].at[row].set(updated[0]),
            accumulator[2].at[row].set(updated[1]),
            accumulator[3].at[row].set(updated[2]),
            accumulator[4].at[row].set(updated[3]),
        )
        return result

    result: _MultiplierEvaluation = jax.lax.fori_loop(
        0,
        product_count,
        add_entry,
        initial,
    )
    return result


def _point_to_interval_component_upper(
    point: Complex128[Array, " n"],
    interval: _ComplexInterval,
) -> Float64[Array, " n"]:
    """PRIVATE: Bound each complex point-to-rectangle distance outwardly.

    Parameters
    ----------
    point : Complex128[Array, " n"]
        Exact stored complex points in a caller-defined unit.
    interval : _ComplexInterval
        Comparison rectangles in the same unit.

    Returns
    -------
    bounds : Float64[Array, " n"]
        Componentwise Euclidean upper bounds.
    """
    real_difference: _RealInterval = _interval_subtract(
        _point_interval(jnp.real(point)),
        (interval[0], interval[1]),
    )
    imag_difference: _RealInterval = _interval_subtract(
        _point_interval(jnp.imag(point)),
        (interval[2], interval[3]),
    )
    real_radius: Float64[Array, " n"] = jnp.maximum(
        jnp.abs(real_difference[0]), jnp.abs(real_difference[1])
    )
    imag_radius: Float64[Array, " n"] = jnp.maximum(
        jnp.abs(imag_difference[0]), jnp.abs(imag_difference[1])
    )
    squared_radius: Float64[Array, " n"] = _upward_add(
        _upward_multiply(real_radius, real_radius),
        _upward_multiply(imag_radius, imag_radius),
    )
    bounds: Float64[Array, " n"] = _upward_sqrt(squared_radius)
    return bounds


def _nonnegative_vector_norm_upper(
    values: Float64[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Scale-safely bound a non-negative vector norm upward.

    Parameters
    ----------
    values : Float64[Array, " n"]
        Non-negative component bounds in one unit.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward Euclidean norm upper bound.
    """
    scale: Float64[Array, ""] = jnp.max(values)
    safe_scale: Float64[Array, ""] = jnp.where(scale > 0.0, scale, 1.0)
    ratios: Float64[Array, " n"] = _upward_divide(values, safe_scale)

    def add_square(
        index: scalar_int,
        accumulator: Float64[Array, ""],
    ) -> Float64[Array, ""]:
        """Accumulate one outward-rounded squared normalized bound."""
        square: Float64[Array, ""] = _upward_multiply(
            ratios[index], ratios[index]
        )
        updated: Float64[Array, ""] = _upward_add(accumulator, square)
        return updated

    scaled_square_sum: Float64[Array, ""] = jax.lax.fori_loop(
        0,
        values.shape[0],
        add_square,
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    scaled_norm: Float64[Array, ""] = _upward_sqrt(scaled_square_sum)
    finite_upper: Float64[Array, ""] = _upward_multiply(scale, scaled_norm)
    upper: Float64[Array, ""] = jnp.where(scale == 0.0, 0.0, finite_upper)
    return upper


def _exact_complex_norm_interval(
    vector: Complex128[Array, " n"],
) -> _RealInterval:
    """PRIVATE: Scale-safely enclose the exact norm of a stored complex vector.

    Parameters
    ----------
    vector : Complex128[Array, " n"]
        Exact stored complex vector in a caller-defined unit.

    Returns
    -------
    interval : _RealInterval
        Inclusive exact-real Euclidean norm interval.
    """
    real: Float64[Array, " n"] = jnp.real(vector)
    imag: Float64[Array, " n"] = jnp.imag(vector)
    real_points: _RealInterval = _point_interval(real)
    imag_points: _RealInterval = _point_interval(imag)
    scale: Float64[Array, ""] = jnp.maximum(
        jnp.max(jnp.maximum(jnp.abs(real_points[0]), jnp.abs(real_points[1]))),
        jnp.max(jnp.maximum(jnp.abs(imag_points[0]), jnp.abs(imag_points[1]))),
    )
    safe_scale: Float64[Array, ""] = jnp.where(scale > 0.0, scale, 1.0)
    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)

    def add_component_squares(
        index: scalar_int,
        accumulator: _RealInterval,
    ) -> _RealInterval:
        """Accumulate normalized real and imaginary component squares."""
        real_ratio: _RealInterval = (
            _downward_divide(real_points[0][index], safe_scale),
            _upward_divide(real_points[1][index], safe_scale),
        )
        imag_ratio: _RealInterval = (
            _downward_divide(imag_points[0][index], safe_scale),
            _upward_divide(imag_points[1][index], safe_scale),
        )
        component_square: _RealInterval = _interval_add(
            _interval_square(real_ratio),
            _interval_square(imag_ratio),
        )
        updated: _RealInterval = _interval_add(accumulator, component_square)
        return updated

    scaled_squares: _RealInterval = jax.lax.fori_loop(
        0,
        vector.shape[0],
        add_component_squares,
        (zero, zero),
    )
    scaled_norm: _RealInterval = (
        _downward_sqrt(jnp.maximum(scaled_squares[0], 0.0)),
        _upward_sqrt(jnp.maximum(scaled_squares[1], 0.0)),
    )
    norm_interval: _RealInterval = _interval_multiply(
        _point_interval(scale), scaled_norm
    )
    exact_zero: Bool[Array, ""] = scale == 0.0
    interval: _RealInterval = (
        jnp.where(exact_zero, 0.0, jnp.maximum(norm_interval[0], 0.0)),
        jnp.where(exact_zero, 0.0, norm_interval[1]),
    )
    return interval


__all__: list[str] = []
