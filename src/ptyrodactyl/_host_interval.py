r"""Provide reusable exact-rational host interval arithmetic.

Extended Summary
----------------
This private module owns dependency-neutral host primitives used to certify
rounded binary64 scientific calculations.  It converts stored finite
binary64 values to exact dyadic rationals, encloses mathematical pi and
rational-turn complex exponentials without trusting library trigonometry,
combines exact rational rectangles deterministically, and converts exact
endpoints back to outward binary64 values.

Notes
-----
All arithmetic before the final outward conversion uses
:class:`fractions.Fraction`.  Subnormal output endpoints are widened to zero
or the minimum normal value, so the resulting host certificates do not
depend on gradual underflow.
"""

from __future__ import annotations

import functools
import math
import sys
from collections.abc import Iterable
from fractions import Fraction

import numpy as np
from beartype.typing import Tuple

type _RealInterval = Tuple[Fraction, Fraction]
type _ComplexRectangle = Tuple[Fraction, Fraction, Fraction, Fraction]

_PI_TARGET_BITS: int = 224
_TAYLOR_UPPER_LAST_INDEX: int = 20
_TAYLOR_LOWER_LAST_INDEX: int = 21
_MINIMUM_NORMAL: float = float.fromhex("0x1.0000000000000p-1022")
_BINARY64_RADIX: int = 2
_BINARY64_SIGNIFICAND_BITS: int = 53
_BINARY64_MAX_EXPONENT: int = 1024
_BINARY64_MIN_EXPONENT: int = -1021
_QUADRANT_COUNT: int = 4
_HALF_TURN_QUADRANT: int = 2
_THREE_QUARTER_TURN_QUADRANT: int = 3


class _RootEnclosureError(ArithmeticError):
    """Identify an internal failure to enclose one rational-turn phase."""


def _host_binary64_supported() -> bool:
    """PRIVATE: Probe every host-float property used by certificates.

    Returns
    -------
    supported : bool
        Whether all required IEEE-754 binary64 host exemplars pass.

    Notes
    -----
    Exact host routes depend on conversion between Python ``float`` and
    :class:`Fraction`, normal/subnormal classification, infinities, and
    directed one-neighbor moves.  These explicit exemplars fail closed on a
    host whose float model is not the declared IEEE-754 binary64 model.
    """
    try:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        minimum_normal = float.fromhex("0x1.0000000000000p-1022")
        minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
        maximum_subnormal = float.fromhex("0x0.fffffffffffffp-1022")
        epsilon = float.fromhex("0x1.0000000000000p-52")
        previous_one = float.fromhex("0x1.fffffffffffffp-1")
        next_one = float.fromhex("0x1.0000000000001p+0")
        positive_infinity = float("inf")
        negative_infinity = -positive_infinity
        neighbor_checks = (
            math.nextafter(1.0, negative_infinity) == previous_one,
            math.nextafter(1.0, positive_infinity) == next_one,
            math.nextafter(0.0, positive_infinity) == minimum_subnormal,
            math.nextafter(0.0, negative_infinity) == -minimum_subnormal,
            math.nextafter(minimum_normal, negative_infinity)
            == maximum_subnormal,
            math.nextafter(maximum, positive_infinity) == positive_infinity,
            math.nextafter(-maximum, negative_infinity) == negative_infinity,
            math.nextafter(positive_infinity, 0.0) == maximum,
            math.nextafter(negative_infinity, 0.0) == -maximum,
        )
    except (OverflowError, ValueError):
        supported: bool = False
        return supported

    float_info = sys.float_info
    supported = (
        float_info.radix == _BINARY64_RADIX
        and float_info.mant_dig == _BINARY64_SIGNIFICAND_BITS
        and float_info.max_exp == _BINARY64_MAX_EXPONENT
        and float_info.min_exp == _BINARY64_MIN_EXPONENT
        and float_info.max == maximum
        and float_info.min == minimum_normal
        and float_info.epsilon == epsilon
        and math.isinf(positive_infinity)
        and positive_infinity > maximum
        and math.isinf(negative_infinity)
        and negative_infinity < -maximum
        and all(neighbor_checks)
    )
    return supported  # noqa: RET504


def _fraction_from_float(value: float) -> Fraction:
    """PRIVATE: Return the exact rational value of one finite binary64.

    Parameters
    ----------
    value : float
        Host binary64 value to convert exactly.

    Returns
    -------
    result : Fraction
        Exact dyadic rational represented by ``value``.

    Raises
    ------
    ValueError
        If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError("certificate inputs must be finite binary64 values")
    result: Fraction = Fraction.from_float(value)
    return result


def _atan_inverse_bounds(
    denominator: int,
    target_width: Fraction,
) -> _RealInterval:
    """PRIVATE: Enclose ``atan(1 / denominator)`` by alternating sums.

    Parameters
    ----------
    denominator : int
        Positive integer reciprocal denominator.
    target_width : Fraction
        Positive maximum rational enclosure width.

    Returns
    -------
    result : _RealInterval
        Exact lower and upper rational arctangent bounds.
    """
    total = Fraction(0)
    upper: Fraction | None = None
    index: int = 0
    while True:
        term = Fraction(
            1,
            (2 * index + 1) * denominator ** (2 * index + 1),
        )
        total = total + term if index % 2 == 0 else total - term
        if index % 2 == 0:
            upper = total
        elif upper is not None and upper - total <= target_width:
            result: _RealInterval = (total, upper)
            return result
        index += 1


@functools.lru_cache(maxsize=1)
def _pi_bounds() -> _RealInterval:
    """PRIVATE: Enclose mathematical pi using exact Machin series bounds.

    Returns
    -------
    result : _RealInterval
        Exact rational lower and upper bounds for mathematical pi.

    Raises
    ------
    _RootEnclosureError
        If the constructed bounds cross or exceed the required width.
    """
    target_width = Fraction(1, 1 << _PI_TARGET_BITS)
    atan_five_lower: Fraction
    atan_five_upper: Fraction
    atan_five_lower, atan_five_upper = _atan_inverse_bounds(
        5,
        target_width / 32,
    )
    atan_239_lower: Fraction
    atan_239_upper: Fraction
    atan_239_lower, atan_239_upper = _atan_inverse_bounds(
        239,
        target_width / 8,
    )
    lower = 16 * atan_five_lower - 4 * atan_239_upper
    upper = 16 * atan_five_upper - 4 * atan_239_lower
    if not lower < upper or upper - lower > target_width:
        raise _RootEnclosureError("Machin pi interval failed its width check")
    result: _RealInterval = (lower, upper)
    return result


def _sine_partial(value: Fraction, last_index: int) -> Fraction:
    """PRIVATE: Evaluate one exact sine Taylor partial sum.

    Parameters
    ----------
    value : Fraction
        Exact rational angle in radians.
    last_index : int
        Final Taylor recurrence index to include.

    Returns
    -------
    total : Fraction
        Exact rational partial sum.
    """
    squared = value * value
    term = value
    total: Fraction = term
    for index in range(last_index):
        term = -term * squared / ((2 * index + 2) * (2 * index + 3))
        total += term
    return total


def _cosine_partial(value: Fraction, last_index: int) -> Fraction:
    """PRIVATE: Evaluate one exact cosine Taylor partial sum.

    Parameters
    ----------
    value : Fraction
        Exact rational angle in radians.
    last_index : int
        Final Taylor recurrence index to include.

    Returns
    -------
    total : Fraction
        Exact rational partial sum.
    """
    squared = value * value
    term = Fraction(1)
    total: Fraction = term
    for index in range(last_index):
        term = -term * squared / ((2 * index + 1) * (2 * index + 2))
        total += term
    return total


def _first_quadrant_sine_cosine(
    local_turn: Fraction,
) -> Tuple[_RealInterval, _RealInterval]:
    """PRIVATE: Enclose sine and cosine for a turn in ``[0, 1/4)``.

    Parameters
    ----------
    local_turn : Fraction
        Exact rational turn reduced to the first quadrant.

    Returns
    -------
    sine : _RealInterval
        Exact-rational sine enclosure.
    cosine : _RealInterval
        Exact-rational cosine enclosure.

    Raises
    ------
    _RootEnclosureError
        If the turn leaves the first quadrant or bounds cross.
    """
    if not Fraction(0) <= local_turn < Fraction(1, 4):
        raise _RootEnclosureError("quadrant reduction left its exact domain")
    if local_turn == 0:
        sine: _RealInterval = (Fraction(0), Fraction(0))
        cosine: _RealInterval = (Fraction(1), Fraction(1))
        result: Tuple[_RealInterval, _RealInterval] = (sine, cosine)
        return result

    pi_lower: Fraction
    pi_upper: Fraction
    pi_lower, pi_upper = _pi_bounds()
    angle_lower = 2 * local_turn * pi_lower
    angle_upper = 2 * local_turn * pi_upper
    sine_lower = _sine_partial(
        angle_lower,
        _TAYLOR_LOWER_LAST_INDEX,
    )
    sine_upper = _sine_partial(
        angle_upper,
        _TAYLOR_UPPER_LAST_INDEX,
    )
    cosine_lower = _cosine_partial(
        angle_upper,
        _TAYLOR_LOWER_LAST_INDEX,
    )
    cosine_upper = _cosine_partial(
        angle_lower,
        _TAYLOR_UPPER_LAST_INDEX,
    )
    quadrant_boundary_lower = pi_lower / 2
    if angle_upper >= quadrant_boundary_lower:
        sine_upper = Fraction(1)
        cosine_lower = Fraction(0)
    if sine_lower > sine_upper or cosine_lower > cosine_upper:
        raise _RootEnclosureError("alternating trigonometric bounds crossed")
    sine = (sine_lower, sine_upper)
    cosine = (cosine_lower, cosine_upper)
    result = (sine, cosine)
    return result  # noqa: RET504


def _negate_interval(interval: _RealInterval) -> _RealInterval:
    """PRIVATE: Negate one exact real interval.

    Parameters
    ----------
    interval : _RealInterval
        Exact rational interval to negate.

    Returns
    -------
    result : _RealInterval
        Negated interval with endpoint order preserved.
    """
    lower, upper = interval
    result: _RealInterval = (-upper, -lower)
    return result


def _rational_turn_exponential(turn: Fraction) -> _ComplexRectangle:
    """PRIVATE: Enclose ``exp(-2 pi i turn)`` without library trig.

    Parameters
    ----------
    turn : Fraction
        Exact rational phase in turns.

    Returns
    -------
    result : _ComplexRectangle
        Exact-rational complex rectangle enclosing the phase factor.

    Raises
    ------
    _RootEnclosureError
        If exact quadrant reduction or trigonometric enclosure fails.
    """
    reduced = turn % 1
    scaled = _QUADRANT_COUNT * reduced
    quadrant = scaled.numerator // scaled.denominator
    local_turn = reduced - Fraction(quadrant, _QUADRANT_COUNT)
    sine: _RealInterval
    cosine: _RealInterval
    sine, cosine = _first_quadrant_sine_cosine(local_turn)

    if quadrant == 0:
        real_interval, positive_imaginary = cosine, sine
    elif quadrant == 1:
        real_interval = _negate_interval(sine)
        positive_imaginary = cosine
    elif quadrant == _HALF_TURN_QUADRANT:
        real_interval = _negate_interval(cosine)
        positive_imaginary = _negate_interval(sine)
    elif quadrant == _THREE_QUARTER_TURN_QUADRANT:
        real_interval = sine
        positive_imaginary = _negate_interval(cosine)
    else:
        raise _RootEnclosureError("exact quadrant was outside 0 through 3")

    imaginary_interval = _negate_interval(positive_imaginary)
    result: _ComplexRectangle = (
        real_interval[0],
        real_interval[1],
        imaginary_interval[0],
        imaginary_interval[1],
    )
    return result


def _real_interval_product(
    left: _RealInterval,
    right: _RealInterval,
) -> _RealInterval:
    """PRIVATE: Multiply two exact rational real intervals.

    Parameters
    ----------
    left : _RealInterval
        Left exact rational interval.
    right : _RealInterval
        Right exact rational interval.

    Returns
    -------
    result : _RealInterval
        Exact interval product hull.
    """
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    result: _RealInterval = (min(products), max(products))
    return result


def _real_interval_square(interval: _RealInterval) -> _RealInterval:
    """PRIVATE: Square one exact rational real interval.

    Parameters
    ----------
    interval : _RealInterval
        Exact rational interval to square.

    Returns
    -------
    result : _RealInterval
        Exact square-image interval, including zero when signs cross.
    """
    lower, upper = interval
    lower_square = lower * lower
    upper_square = upper * upper
    if lower <= 0 <= upper:
        result: _RealInterval = (
            Fraction(0),
            max(lower_square, upper_square),
        )
        return result
    result = (
        min(lower_square, upper_square),
        max(lower_square, upper_square),
    )
    return result  # noqa: RET504


def _real_interval_add(
    left: _RealInterval,
    right: _RealInterval,
) -> _RealInterval:
    """PRIVATE: Add two exact rational real intervals.

    Parameters
    ----------
    left : _RealInterval
        Left exact rational interval.
    right : _RealInterval
        Right exact rational interval.

    Returns
    -------
    result : _RealInterval
        Exact interval sum.
    """
    result: _RealInterval = (left[0] + right[0], left[1] + right[1])
    return result


def _real_interval_subtract(
    left: _RealInterval,
    right: _RealInterval,
) -> _RealInterval:
    """PRIVATE: Subtract two exact rational real intervals.

    Parameters
    ----------
    left : _RealInterval
        Left exact rational interval.
    right : _RealInterval
        Right exact rational interval.

    Returns
    -------
    result : _RealInterval
        Exact interval difference.
    """
    result: _RealInterval = (left[0] - right[1], left[1] - right[0])
    return result


def _normalized_sinc_integer_ratio(
    mode: int,
    count: int,
) -> _RealInterval:
    r"""PRIVATE: Enclose ``sin(pi mode/count)/(pi mode/count)``.

    Parameters
    ----------
    mode : int
        Signed unwrapped integer Fourier mode.
    count : int
        Positive integer cell count.

    Returns
    -------
    result : _RealInterval
        Exact-rational normalized-sinc enclosure.

    Raises
    ------
    ValueError
        If ``count`` is not positive.
    _RootEnclosureError
        If the rational-turn sine enclosure fails internally.

    Notes
    -----
    Zero and every nonzero integer zero are returned symbolically. For every
    other mode, the numerator is recovered from the negative imaginary part
    of ``exp(-2 pi i abs(mode)/(2 count))``; the denominator retains the
    unwrapped absolute mode.
    """
    if count <= 0:
        raise ValueError("normalized-sinc count must be positive")
    if mode == 0:
        result: _RealInterval = (Fraction(1), Fraction(1))
        return result

    magnitude = abs(mode)
    if magnitude % count == 0:
        result = (Fraction(0), Fraction(0))
        return result  # noqa: RET504

    root = _rational_turn_exponential(Fraction(magnitude, 2 * count))
    sine: _RealInterval = (-root[3], -root[2])
    pi_lower, pi_upper = _pi_bounds()
    ratio = Fraction(magnitude, count)
    denominator: _RealInterval = (
        ratio * pi_lower,
        ratio * pi_upper,
    )
    reciprocal_denominator: _RealInterval = (
        Fraction(1, 1) / denominator[1],
        Fraction(1, 1) / denominator[0],
    )
    result = _real_interval_product(sine, reciprocal_denominator)
    return result  # noqa: RET504


def _complex_rectangle_multiply(
    left: _ComplexRectangle,
    right: _ComplexRectangle,
) -> _ComplexRectangle:
    """PRIVATE: Multiply two exact rational complex rectangles.

    Parameters
    ----------
    left : _ComplexRectangle
        Left exact rational complex rectangle.
    right : _ComplexRectangle
        Right exact rational complex rectangle.

    Returns
    -------
    result : _ComplexRectangle
        Exact interval-arithmetic product rectangle.
    """
    left_real: _RealInterval = (left[0], left[1])
    left_imag: _RealInterval = (left[2], left[3])
    right_real: _RealInterval = (right[0], right[1])
    right_imag: _RealInterval = (right[2], right[3])
    real = _real_interval_subtract(
        _real_interval_product(left_real, right_real),
        _real_interval_product(left_imag, right_imag),
    )
    imag = _real_interval_add(
        _real_interval_product(left_real, right_imag),
        _real_interval_product(left_imag, right_real),
    )
    result: _ComplexRectangle = (real[0], real[1], imag[0], imag[1])
    return result


def _complex_rectangle_add(
    left: _ComplexRectangle,
    right: _ComplexRectangle,
) -> _ComplexRectangle:
    """PRIVATE: Add two exact rational complex rectangles.

    Parameters
    ----------
    left : _ComplexRectangle
        Left exact rational complex rectangle.
    right : _ComplexRectangle
        Right exact rational complex rectangle.

    Returns
    -------
    result : _ComplexRectangle
        Exact componentwise sum rectangle.
    """
    result: _ComplexRectangle = (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
    )
    return result


def _scale_complex_rectangle(
    rectangle: _ComplexRectangle,
    scalar: Fraction,
) -> _ComplexRectangle:
    """PRIVATE: Multiply one complex rectangle by an exact real scalar.

    Parameters
    ----------
    rectangle : _ComplexRectangle
        Exact rational complex rectangle.
    scalar : Fraction
        Exact rational real scale.

    Returns
    -------
    result : _ComplexRectangle
        Exact scaled complex rectangle.
    """
    real = _real_interval_product(
        (rectangle[0], rectangle[1]),
        (scalar, scalar),
    )
    imag = _real_interval_product(
        (rectangle[2], rectangle[3]),
        (scalar, scalar),
    )
    result: _ComplexRectangle = (real[0], real[1], imag[0], imag[1])
    return result


def _conjugate_rectangle(
    rectangle: _ComplexRectangle,
) -> _ComplexRectangle:
    """PRIVATE: Conjugate one exact complex rectangle.

    Parameters
    ----------
    rectangle : _ComplexRectangle
        Exact rational complex rectangle.

    Returns
    -------
    result : _ComplexRectangle
        Exact conjugate rectangle.
    """
    result: _ComplexRectangle = (
        rectangle[0],
        rectangle[1],
        -rectangle[3],
        -rectangle[2],
    )
    return result


def _pairwise_rectangle_sum(
    terms: Iterable[_ComplexRectangle],
) -> _ComplexRectangle:
    """PRIVATE: Sum rectangles through a deterministic binary reduction.

    Parameters
    ----------
    terms : Iterable[_ComplexRectangle]
        Stream of exact rational complex rectangles.

    Returns
    -------
    total : _ComplexRectangle
        Exact componentwise sum of all submitted rectangles.

    Raises
    ------
    AssertionError
        If an occupied binary-reduction slot becomes inconsistent.
    """
    zero: _ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    slots: list[_ComplexRectangle | None] = []
    for submitted in terms:
        value = submitted
        level = 0
        while level < len(slots) and slots[level] is not None:
            previous = slots[level]
            if previous is None:
                raise AssertionError("occupied pairwise slot became empty")
            value = _complex_rectangle_add(previous, value)
            slots[level] = None
            level += 1
        if level == len(slots):
            slots.append(value)
        else:
            slots[level] = value
    total: _ComplexRectangle = zero
    for value in slots:
        if value is not None:
            total = _complex_rectangle_add(total, value)
    return total


def _normal_floor_lower(value: float) -> float:
    """PRIVATE: Widen a lower subnormal endpoint without underflow.

    Parameters
    ----------
    value : float
        Directed lower binary64 endpoint.

    Returns
    -------
    result : float
        Lower endpoint widened to zero or the minimum normal if needed.
    """
    if value == 0.0 or abs(value) >= _MINIMUM_NORMAL:
        result: float = value
        return result
    result = -_MINIMUM_NORMAL if value < 0.0 else 0.0
    return result  # noqa: RET504


def _normal_floor_upper(value: float) -> float:
    """PRIVATE: Widen an upper subnormal endpoint without underflow.

    Parameters
    ----------
    value : float
        Directed upper binary64 endpoint.

    Returns
    -------
    result : float
        Upper endpoint widened to zero or the minimum normal if needed.
    """
    if value == 0.0 or abs(value) >= _MINIMUM_NORMAL:
        result: float = value
        return result
    result = 0.0 if value < 0.0 else _MINIMUM_NORMAL
    return result  # noqa: RET504


def _fraction_lower_float(value: Fraction) -> float:
    """PRIVATE: Convert one rational endpoint toward minus infinity.

    Parameters
    ----------
    value : Fraction
        Exact rational endpoint.

    Returns
    -------
    result : float
        Outward binary64 lower endpoint.
    """
    try:
        candidate = float(value)
    except OverflowError:
        result: float = sys.float_info.max if value > 0 else -math.inf
        return result
    if math.isfinite(candidate) and Fraction.from_float(candidate) > value:
        candidate = math.nextafter(candidate, -math.inf)
    result: float = _normal_floor_lower(candidate)
    return result


def _fraction_upper_float(value: Fraction) -> float:
    """PRIVATE: Convert one rational endpoint toward plus infinity.

    Parameters
    ----------
    value : Fraction
        Exact rational endpoint.

    Returns
    -------
    result : float
        Outward binary64 upper endpoint.
    """
    try:
        candidate = float(value)
    except OverflowError:
        result: float = math.inf if value > 0 else -sys.float_info.max
        return result
    if math.isfinite(candidate) and Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    result: float = _normal_floor_upper(candidate)
    return result


def _coefficient_error_fraction(
    coefficient: np.complex128,
    rectangle: _ComplexRectangle,
) -> Fraction:
    """PRIVATE: Bound one stored coefficient against an exact rectangle.

    Parameters
    ----------
    coefficient : np.complex128
        Final stored binary64 complex coefficient.
    rectangle : _ComplexRectangle
        Exact-rational target rectangle.

    Returns
    -------
    result : Fraction
        Exact complex L1 point-to-rectangle error bound.
    """
    real = _fraction_from_float(float(np.real(coefficient)))
    imag = _fraction_from_float(float(np.imag(coefficient)))
    real_gap = max(abs(real - rectangle[0]), abs(real - rectangle[1]))
    imag_gap = max(abs(imag - rectangle[2]), abs(imag - rectangle[3]))
    result: Fraction = real_gap + imag_gap
    return result


def _floor_log2_fraction(value: Fraction) -> int:
    """PRIVATE: Return exact ``floor(log2(value))`` for ``value > 0``.

    Parameters
    ----------
    value : Fraction
        Positive exact rational value.

    Returns
    -------
    result : int
        Exact floor of the base-two logarithm.

    Raises
    ------
    ValueError
        If ``value`` is not positive.
    """
    if value <= 0:
        raise ValueError("log2 input must be positive")
    numerator = value.numerator
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << (-exponent) < denominator:
        exponent -= 1
    result: int = exponent
    return result


def _power_of_two_fraction(exponent: int) -> Fraction:
    """PRIVATE: Return an exact signed-power-of-two rational scale.

    Parameters
    ----------
    exponent : int
        Signed base-two exponent.

    Returns
    -------
    result : Fraction
        Exact value ``2**exponent``.
    """
    if exponent >= 0:
        result: Fraction = Fraction(1 << exponent)
        return result
    result = Fraction(1, 1 << (-exponent))
    return result  # noqa: RET504


def _sqrt_fraction_bounds(value: Fraction) -> _RealInterval:
    """PRIVATE: Enclose one non-negative rational square root on both sides.

    Parameters
    ----------
    value : Fraction
        Non-negative exact rational radicand.

    Returns
    -------
    result : _RealInterval
        Exact dyadic lower and upper square-root bounds.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    if value < 0:
        raise ValueError("square-root radicand must be non-negative")
    if value == 0:
        result: _RealInterval = (Fraction(0), Fraction(0))
        return result
    exponent = _floor_log2_fraction(value)
    scale_exponent = exponent // 2
    scaled = value / _power_of_two_fraction(2 * scale_exponent)
    precision_bits = 128
    scaled_integer_numerator = scaled.numerator << (2 * precision_bits)
    quotient = scaled_integer_numerator // scaled.denominator
    root_floor = math.isqrt(quotient)
    exact_square = (
        root_floor * root_floor * scaled.denominator
        == scaled_integer_numerator
    )
    root_upper_integer = root_floor if exact_square else root_floor + 1
    scale = _power_of_two_fraction(scale_exponent)
    scaled_root_lower = Fraction(root_floor, 1 << precision_bits)
    scaled_root_upper = Fraction(root_upper_integer, 1 << precision_bits)
    result = (
        scaled_root_lower * scale,
        scaled_root_upper * scale,
    )
    return result  # noqa: RET504


def _sqrt_fraction_upper(value: Fraction) -> Fraction:
    """PRIVATE: Enclose one non-negative rational square root above.

    Parameters
    ----------
    value : Fraction
        Non-negative exact rational radicand.

    Returns
    -------
    result : Fraction
        Exact dyadic upper bound for the square root.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    result: Fraction = _sqrt_fraction_bounds(value)[1]
    return result
