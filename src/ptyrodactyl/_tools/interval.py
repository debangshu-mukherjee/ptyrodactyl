r"""Provide dependency-neutral FTZ-safe binary64 interval primitives.

Extended Summary
----------------
This private module embeds exact stored binary64 points in reusable real
intervals and evaluates outward arithmetic without assuming gradual
underflow.  It probes normal-range binary64 addition, multiplication,
division, square root, and ``nextafter`` behavior.  Unsupported normal
arithmetic, invalid operation preconditions, and unresolved infinite forms
fail closed to directed infinities.

Routine Listings
----------------
:func:`all_normal_arithmetic_supported`
    Combine every required normal-range arithmetic probe.
:func:`arithmetic_environment_probes`
    Probe normal primitives and gradual underflow separately.
:func:`downward_divide`
    Enclose one positive-denominator quotient from below.
:func:`downward_sqrt`
    Enclose one nonnegative square root from below.
:func:`interval_add`
    Outward-add two reusable real intervals.
:func:`interval_divide_positive`
    Outward-divide by one strictly positive real interval.
:func:`interval_multiply`
    Outward-multiply two reusable real intervals.
:func:`interval_sqrt`
    Outward-square-root one nonnegative real interval.
:func:`interval_square`
    Outward-square one reusable real interval.
:func:`interval_subtract`
    Outward-subtract two reusable real intervals.
:func:`mathematical_pi_interval`
    Enclose mathematical pi with guarded binary64 endpoints.
:func:`point_interval`
    Embed exact stored binary64 points through FTZ.2.
:func:`round_up`
    Widen a nearest binary64 point toward positive infinity.
:func:`upward_add`
    Enclose one exact-real endpoint addition from above.
:func:`upward_divide`
    Enclose one positive-denominator quotient from above.
:func:`upward_multiply`
    Enclose one exact-real endpoint product from above.
:func:`upward_sqrt`
    Enclose one nonnegative square root from above.
:obj:`RealInterval`
    Represent one traced binary64 interval by its lower and upper arrays.

Notes
-----
Every finite nonzero endpoint returned by this module is normal.  Stored
subnormal points are classified from their component bits and widened
sign-wise before DAZ-sensitive arithmetic.  Exact additive-zero,
multiplicative-zero, zero-numerator, and zero-radicand identities are
preserved; computed cancellation is deliberately widened.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jax import Array, lax
from jaxtyping import Bool, Float64

type RealInterval = Tuple[
    Float64[Array, "..."],
    Float64[Array, "..."],
]

_MINIMUM_NORMAL_HEX: str = "0x1.0000000000000p-1022"
_PI_LOWER_HEX: str = "0x1.921fb54442d18p+1"
_SIGN_MASK: int = 0x8000000000000000
_EXPONENT_MASK: int = 0x7FF0000000000000
_FRACTION_MASK: int = 0x000FFFFFFFFFFFFF
_MAGNITUDE_MASK: int = 0x7FFFFFFFFFFFFFFF


def _minimum_normal() -> Float64[Array, ""]:
    """PRIVATE: Return the positive minimum normal binary64 value.

    Returns
    -------
    tiny : Float64[Array, ""]
        Exact binary64 value ``2**-1022``.
    """
    tiny: Float64[Array, ""] = jnp.asarray(
        float.fromhex(_MINIMUM_NORMAL_HEX),
        dtype=jnp.float64,
    )
    return tiny


def _binary64_masks(
    value: Float64[Array, "..."],
) -> Tuple[
    Bool[Array, "..."],
    Bool[Array, "..."],
    Bool[Array, "..."],
    Bool[Array, "..."],
]:
    """PRIVATE: Classify binary64 values without floating comparisons.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Stored values whose IEEE-754 component bits are inspected.

    Returns
    -------
    zero : Bool[Array, "..."]
        Whether each magnitude bit pattern is zero.
    subnormal : Bool[Array, "..."]
        Whether each value is finite, nonzero, and subnormal.
    negative : Bool[Array, "..."]
        Whether each sign bit is set, including negative zero.
    finite : Bool[Array, "..."]
        Whether each exponent is not the all-ones pattern.

    Notes
    -----
    A non-binary64 runtime dtype returns false masks except for ``zero``.
    Normal-operation probes then reject that environment before arithmetic
    evidence can remain finite.
    """
    checked_value: Float64[Array, "..."] = lax.stop_gradient(value)
    if checked_value.dtype != jnp.dtype(jnp.float64):
        zero: Bool[Array, "..."] = checked_value == 0
        false_mask: Bool[Array, "..."] = jnp.zeros_like(
            checked_value,
            dtype=jnp.bool_,
        )
        subnormal: Bool[Array, "..."] = false_mask
        negative: Bool[Array, "..."] = false_mask
        finite: Bool[Array, "..."] = false_mask
        result: Tuple[
            Bool[Array, "..."],
            Bool[Array, "..."],
            Bool[Array, "..."],
            Bool[Array, "..."],
        ] = (zero, subnormal, negative, finite)
        return result

    bits = lax.bitcast_convert_type(checked_value, jnp.uint64)
    sign_mask = jnp.asarray(_SIGN_MASK, dtype=jnp.uint64)
    exponent_mask = jnp.asarray(_EXPONENT_MASK, dtype=jnp.uint64)
    fraction_mask = jnp.asarray(_FRACTION_MASK, dtype=jnp.uint64)
    magnitude_mask = jnp.asarray(_MAGNITUDE_MASK, dtype=jnp.uint64)
    magnitude = jnp.bitwise_and(bits, magnitude_mask)
    exponent = jnp.bitwise_and(bits, exponent_mask)
    fraction = jnp.bitwise_and(bits, fraction_mask)
    zero = magnitude == 0
    subnormal = (exponent == 0) & (fraction != 0)
    negative = jnp.bitwise_and(bits, sign_mask) != 0
    finite = exponent != exponent_mask
    result: Tuple[
        Bool[Array, "..."],
        Bool[Array, "..."],
        Bool[Array, "..."],
        Bool[Array, "..."],
    ] = (zero, subnormal, negative, finite)
    return result  # noqa: RET504


def _is_nonzero_subnormal(
    value: Float64[Array, "..."],
) -> Bool[Array, "..."]:
    """PRIVATE: Identify nonzero binary64 subnormals by bit pattern.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Stored values to classify before DAZ-sensitive arithmetic.

    Returns
    -------
    subnormal : Bool[Array, "..."]
        Whether each value has a zero exponent and nonzero fraction.
    """
    masks: Tuple[
        Bool[Array, "..."],
        Bool[Array, "..."],
        Bool[Array, "..."],
        Bool[Array, "..."],
    ] = _binary64_masks(value)
    subnormal: Bool[Array, "..."] = masks[1]
    return subnormal


def point_interval(value: Float64[Array, "..."]) -> RealInterval:
    """Embed exact stored binary64 points through FTZ.2.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Exact stored binary64 points in a caller-defined unit.

    Returns
    -------
    interval : RealInterval
        Exact points for finite normal values and zeros, ``[0, tiny]`` for
        positive subnormals, ``[-tiny, 0]`` for negative subnormals, and an
        unbounded interval for nonfinite inputs or an unsupported required
        arithmetic environment.
    """
    checked_value: Float64[Array, "..."] = lax.stop_gradient(value)
    _, subnormal, negative, finite = _binary64_masks(checked_value)
    tiny: Float64[Array, ""] = _minimum_normal()
    subnormal_lower: Float64[Array, "..."] = jnp.where(
        negative,
        -tiny,
        0.0,
    )
    subnormal_upper: Float64[Array, "..."] = jnp.where(
        negative,
        0.0,
        tiny,
    )
    embedded_lower: Float64[Array, "..."] = jnp.where(
        finite,
        jnp.where(subnormal, subnormal_lower, checked_value),
        -jnp.inf,
    )
    embedded_upper: Float64[Array, "..."] = jnp.where(
        finite,
        jnp.where(subnormal, subnormal_upper, checked_value),
        jnp.inf,
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    lower: Float64[Array, "..."] = jnp.where(
        normal_supported,
        embedded_lower,
        -jnp.inf,
    )
    upper: Float64[Array, "..."] = jnp.where(
        normal_supported,
        embedded_upper,
        jnp.inf,
    )
    interval: RealInterval = (
        lax.stop_gradient(lower),
        lax.stop_gradient(upper),
    )
    return interval


def arithmetic_environment_probes() -> Tuple[
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
]:
    """Probe normal primitives and gradual underflow separately.

    Returns
    -------
    probes : Tuple[Bool[Array, ""], ...]
        Addition, multiplication, division, square root, ``nextafter``, bit
        pattern, and gradual-underflow support flags in that order.

    Notes
    -----
    Every probe operand and arithmetic result crosses an optimization barrier.
    Gradual-underflow failure is diagnostic only: FTZ.2 and the guarded
    directed fallbacks remain sound.  Failure of any required normal probe
    makes the corresponding directed primitive return an infinity.
    """
    one: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(1.0, dtype=jnp.float64)
    )
    epsilon: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(float.fromhex("0x1.0000000000000p-52"), dtype=jnp.float64)
    )
    half_epsilon: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(float.fromhex("0x1.0000000000000p-53"), dtype=jnp.float64)
    )
    tie_sum: Float64[Array, ""] = lax.optimization_barrier(
        lax.optimization_barrier(one) + lax.optimization_barrier(half_epsilon)
    )
    step_sum: Float64[Array, ""] = lax.optimization_barrier(
        lax.optimization_barrier(one) + lax.optimization_barrier(epsilon)
    )
    expected_step: Float64[Array, ""] = jnp.asarray(
        float.fromhex("0x1.0000000000001p+0"),
        dtype=jnp.float64,
    )
    add_supported: Bool[Array, ""] = (
        (one.dtype == jnp.dtype(jnp.float64))
        & (tie_sum == one)
        & (step_sum == expected_step)
    )

    one_plus_epsilon: Float64[Array, ""] = lax.optimization_barrier(step_sum)
    product: Float64[Array, ""] = lax.optimization_barrier(
        lax.optimization_barrier(one_plus_epsilon)
        * lax.optimization_barrier(one_plus_epsilon)
    )
    expected_product: Float64[Array, ""] = jnp.asarray(
        float.fromhex("0x1.0000000000002p+0"),
        dtype=jnp.float64,
    )
    multiply_supported: Bool[Array, ""] = (
        one.dtype == jnp.dtype(jnp.float64)
    ) & (product == expected_product)

    ten: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(10.0, dtype=jnp.float64)
    )
    quotient: Float64[Array, ""] = lax.optimization_barrier(
        lax.optimization_barrier(one) / lax.optimization_barrier(ten)
    )
    expected_quotient: Float64[Array, ""] = jnp.asarray(
        float.fromhex("0x1.999999999999ap-4"),
        dtype=jnp.float64,
    )
    divide_supported: Bool[Array, ""] = (
        one.dtype == jnp.dtype(jnp.float64)
    ) & (quotient == expected_quotient)

    two: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(2.0, dtype=jnp.float64)
    )
    root: Float64[Array, ""] = lax.optimization_barrier(
        jnp.sqrt(lax.optimization_barrier(two))
    )
    expected_root: Float64[Array, ""] = jnp.asarray(
        float.fromhex("0x1.6a09e667f3bcdp+0"),
        dtype=jnp.float64,
    )
    sqrt_supported: Bool[Array, ""] = (one.dtype == jnp.dtype(jnp.float64)) & (
        root == expected_root
    )

    previous_one: Float64[Array, ""] = lax.optimization_barrier(
        jnp.nextafter(
            lax.optimization_barrier(one),
            jnp.asarray(-jnp.inf, dtype=jnp.float64),
        )
    )
    next_one: Float64[Array, ""] = lax.optimization_barrier(
        jnp.nextafter(
            lax.optimization_barrier(one),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
        )
    )
    expected_previous: Float64[Array, ""] = jnp.asarray(
        float.fromhex("0x1.fffffffffffffp-1"),
        dtype=jnp.float64,
    )
    maximum: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(
            float.fromhex("0x1.fffffffffffffp+1023"),
            dtype=jnp.float64,
        )
    )
    after_maximum: Float64[Array, ""] = lax.optimization_barrier(
        jnp.nextafter(
            lax.optimization_barrier(maximum),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
        )
    )
    nextafter_supported: Bool[Array, ""] = (
        (one.dtype == jnp.dtype(jnp.float64))
        & (previous_one == expected_previous)
        & (next_one == expected_step)
        & jnp.isposinf(after_maximum)
    )

    tiny: Float64[Array, ""] = lax.optimization_barrier(_minimum_normal())
    positive_subnormal: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(
            float.fromhex("0x0.0000000000001p-1022"),
            dtype=jnp.float64,
        )
    )
    negative_subnormal: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(
            float.fromhex("-0x0.0000000000001p-1022"),
            dtype=jnp.float64,
        )
    )
    one_bits = lax.bitcast_convert_type(one, jnp.uint64)
    tiny_bits = lax.bitcast_convert_type(tiny, jnp.uint64)
    positive_subnormal_bits = lax.bitcast_convert_type(
        positive_subnormal,
        jnp.uint64,
    )
    negative_subnormal_bits = lax.bitcast_convert_type(
        negative_subnormal,
        jnp.uint64,
    )
    _, positive_is_subnormal, positive_is_negative, positive_is_finite = (
        _binary64_masks(positive_subnormal)
    )
    _, negative_is_subnormal, negative_is_negative, negative_is_finite = (
        _binary64_masks(negative_subnormal)
    )
    bit_pattern_supported: Bool[Array, ""] = (
        (one.dtype == jnp.dtype(jnp.float64))
        & (one_bits == jnp.asarray(0x3FF0000000000000, dtype=jnp.uint64))
        & (tiny_bits == jnp.asarray(0x0010000000000000, dtype=jnp.uint64))
        & (positive_subnormal_bits == jnp.asarray(1, dtype=jnp.uint64))
        & (
            negative_subnormal_bits
            == jnp.asarray(0x8000000000000001, dtype=jnp.uint64)
        )
        & positive_is_subnormal
        & (~positive_is_negative)
        & positive_is_finite
        & negative_is_subnormal
        & negative_is_negative
        & negative_is_finite
    )
    half: Float64[Array, ""] = lax.optimization_barrier(
        jnp.asarray(0.5, dtype=jnp.float64)
    )
    half_normal: Float64[Array, ""] = lax.optimization_barrier(
        lax.optimization_barrier(tiny) * lax.optimization_barrier(half)
    )
    next_zero: Float64[Array, ""] = lax.optimization_barrier(
        jnp.nextafter(
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
        )
    )
    half_normal_bits = lax.bitcast_convert_type(half_normal, jnp.uint64)
    next_zero_bits = lax.bitcast_convert_type(next_zero, jnp.uint64)
    gradual_underflow_supported: Bool[Array, ""] = (
        (one.dtype == jnp.dtype(jnp.float64))
        & (
            half_normal_bits
            == jnp.asarray(0x0008000000000000, dtype=jnp.uint64)
        )
        & (next_zero_bits == jnp.asarray(1, dtype=jnp.uint64))
    )
    probes: Tuple[
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
    ] = (
        lax.stop_gradient(add_supported),
        lax.stop_gradient(multiply_supported),
        lax.stop_gradient(divide_supported),
        lax.stop_gradient(sqrt_supported),
        lax.stop_gradient(nextafter_supported),
        lax.stop_gradient(bit_pattern_supported),
        lax.stop_gradient(gradual_underflow_supported),
    )
    return probes


@jax.jit
def all_normal_arithmetic_supported() -> Bool[Array, ""]:
    """Combine every required normal-range arithmetic probe.

    Returns
    -------
    supported : Bool[Array, ""]
        Whether addition, multiplication, division, square root, and
        normal-range ``nextafter`` all passed their runtime probes.

    Notes
    -----
    Gradual underflow is intentionally excluded.  The FTZ-safe point
    embedding and normal fallbacks make that diagnostic non-load-bearing.
    """
    add: Bool[Array, ""]
    multiply: Bool[Array, ""]
    divide: Bool[Array, ""]
    sqrt: Bool[Array, ""]
    nextafter: Bool[Array, ""]
    bit_pattern: Bool[Array, ""]
    add, multiply, divide, sqrt, nextafter, bit_pattern, _ = (
        arithmetic_environment_probes()
    )
    supported: Bool[Array, ""] = lax.stop_gradient(
        add & multiply & divide & sqrt & nextafter & bit_pattern
    )
    return supported


def _round_down(value: Float64[Array, "..."]) -> Float64[Array, "..."]:
    """PRIVATE: Widen a nearest binary64 point toward negative infinity.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Binary64 point whose represented target may differ by one rounding.

    Returns
    -------
    lower : Float64[Array, "..."]
        Guarded lower endpoint, or negative infinity if unsupported.

    Notes
    -----
    Zero, a subnormal input, or a subnormal neighbor widens to ``-tiny``.
    This helper is for bracketing a rounded target, not embedding an exact
    stored point; exact points use :func:`point_interval`.
    """
    rounded_value: Float64[Array, "..."] = lax.optimization_barrier(
        lax.stop_gradient(value)
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(
            rounded_value,
            jnp.asarray(-jnp.inf, dtype=jnp.float64),
        )
    )
    zero, subnormal, _, finite = _binary64_masks(rounded_value)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    candidate: Float64[Array, "..."] = jnp.where(
        finite,
        jnp.where(
            zero | subnormal | neighbor_subnormal,
            -_minimum_normal(),
            neighbor,
        ),
        -jnp.inf,
    )
    lower: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, -jnp.inf)
    )
    return lower


def round_up(value: Float64[Array, "..."]) -> Float64[Array, "..."]:
    """Widen a nearest binary64 point toward positive infinity.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Binary64 point whose represented target may differ by one rounding.

    Returns
    -------
    upper : Float64[Array, "..."]
        Guarded upper endpoint, or positive infinity if unsupported.

    Notes
    -----
    Zero, a subnormal input, or a subnormal neighbor widens to ``tiny``.
    This helper is for bracketing a rounded target, not embedding an exact
    stored point; exact points use :func:`point_interval`.
    """
    rounded_value: Float64[Array, "..."] = lax.optimization_barrier(
        lax.stop_gradient(value)
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(
            rounded_value,
            jnp.asarray(jnp.inf, dtype=jnp.float64),
        )
    )
    zero, subnormal, _, finite = _binary64_masks(rounded_value)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    candidate: Float64[Array, "..."] = jnp.where(
        finite,
        jnp.where(
            zero | subnormal | neighbor_subnormal,
            _minimum_normal(),
            neighbor,
        ),
        jnp.inf,
    )
    upper: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, jnp.inf)
    )
    return upper


def mathematical_pi_interval() -> RealInterval:
    """Enclose mathematical pi with guarded binary64 endpoints.

    Returns
    -------
    lower : Float64[Array, ""]
        Known binary64 value below mathematical pi, or negative infinity
        when normal arithmetic is unsupported.
    upper : Float64[Array, ""]
        Immediate upward binary64 neighbor of ``lower``, or positive infinity
        when normal arithmetic is unsupported.

    Notes
    -----
    ``0x1.921fb54442d18p+1`` is strictly below mathematical pi, while its
    immediate upward binary64 neighbor is strictly above it.
    """
    pi_lower: Float64[Array, ""] = jnp.asarray(
        float.fromhex(_PI_LOWER_HEX),
        dtype=jnp.float64,
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    lower: Float64[Array, ""] = lax.stop_gradient(
        jnp.where(normal_supported, pi_lower, -jnp.inf)
    )
    upper: Float64[Array, ""] = round_up(pi_lower)
    interval: RealInterval = (lower, upper)
    return interval


def _downward_add(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose one exact-real endpoint addition from below.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left normal, zero, or infinite endpoint.
    right : Float64[Array, "..."]
        Right normal, zero, or infinite endpoint.

    Returns
    -------
    lower : Float64[Array, "..."]
        Guarded lower endpoint in the operands' common unit.

    Notes
    -----
    Addition by an exact bitwise zero re-embeds the other operand through
    FTZ.2.  A nonidentity result at zero or in the subnormal range widens to
    ``-tiny``.  Input subnormal endpoints violate the reusable-endpoint
    invariant and fail closed.
    """
    rounded_left, rounded_right = lax.optimization_barrier(
        (lax.stop_gradient(left), lax.stop_gradient(right))
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        rounded_left + rounded_right
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(-jnp.inf, dtype=jnp.float64))
    )
    left_zero, left_subnormal, _, _ = _binary64_masks(left)
    right_zero, right_subnormal, _, _ = _binary64_masks(right)
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    identity: Bool[Array, "..."] = left_zero | right_zero
    identity_value: Float64[Array, "..."] = jnp.where(
        left_zero,
        right,
        left,
    )
    identity_lower: Float64[Array, "..."] = point_interval(identity_value)[0]
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        jnp.isnan(raw),
        -jnp.inf,
        jnp.where(
            identity,
            identity_lower,
            jnp.where(
                left_subnormal | right_subnormal,
                -jnp.inf,
                jnp.where(underflow_risk, -_minimum_normal(), neighbor),
            ),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    lower: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, -jnp.inf)
    )
    return lower


def upward_add(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Enclose one exact-real endpoint addition from above.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left normal, zero, or infinite endpoint.
    right : Float64[Array, "..."]
        Right normal, zero, or infinite endpoint.

    Returns
    -------
    upper : Float64[Array, "..."]
        Guarded upper endpoint in the operands' common unit.

    Notes
    -----
    Addition by an exact bitwise zero re-embeds the other operand through
    FTZ.2.  A nonidentity result at zero or in the subnormal range widens to
    ``tiny``.  Input subnormal endpoints violate the reusable-endpoint
    invariant and fail closed.
    """
    rounded_left, rounded_right = lax.optimization_barrier(
        (lax.stop_gradient(left), lax.stop_gradient(right))
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        rounded_left + rounded_right
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(jnp.inf, dtype=jnp.float64))
    )
    left_zero, left_subnormal, _, _ = _binary64_masks(left)
    right_zero, right_subnormal, _, _ = _binary64_masks(right)
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    identity: Bool[Array, "..."] = left_zero | right_zero
    identity_value: Float64[Array, "..."] = jnp.where(
        left_zero,
        right,
        left,
    )
    identity_upper: Float64[Array, "..."] = point_interval(identity_value)[1]
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        jnp.isnan(raw),
        jnp.inf,
        jnp.where(
            identity,
            identity_upper,
            jnp.where(
                left_subnormal | right_subnormal,
                jnp.inf,
                jnp.where(underflow_risk, _minimum_normal(), neighbor),
            ),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    upper: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, jnp.inf)
    )
    return upper


def _downward_multiply(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose one exact-real endpoint product from below.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left normal, zero, or infinite endpoint.
    right : Float64[Array, "..."]
        Right normal, zero, or infinite endpoint.

    Returns
    -------
    lower : Float64[Array, "..."]
        Guarded lower endpoint in the product unit.

    Notes
    -----
    A bitwise zero factor preserves exact zero.  A nonidentity zero or
    subnormal result widens to ``-tiny``.  A subnormal input endpoint fails
    closed because it violates the reusable-endpoint invariant.
    """
    rounded_left, rounded_right = lax.optimization_barrier(
        (lax.stop_gradient(left), lax.stop_gradient(right))
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        rounded_left * rounded_right
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(-jnp.inf, dtype=jnp.float64))
    )
    left_zero, left_subnormal, _, left_finite = _binary64_masks(left)
    right_zero, right_subnormal, _, right_finite = _binary64_masks(right)
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    exact_zero: Bool[Array, "..."] = (left_zero & right_finite) | (
        right_zero & left_finite
    )
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        jnp.isnan(raw),
        -jnp.inf,
        jnp.where(
            exact_zero,
            0.0,
            jnp.where(
                left_subnormal | right_subnormal,
                -jnp.inf,
                jnp.where(underflow_risk, -_minimum_normal(), neighbor),
            ),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    lower: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(
            normal_supported,
            candidate,
            -jnp.inf,
        )
    )
    return lower


def upward_multiply(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Enclose one exact-real endpoint product from above.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left normal, zero, or infinite endpoint.
    right : Float64[Array, "..."]
        Right normal, zero, or infinite endpoint.

    Returns
    -------
    upper : Float64[Array, "..."]
        Guarded upper endpoint in the product unit.

    Notes
    -----
    A bitwise zero factor preserves exact zero.  A nonidentity zero or
    subnormal result widens to ``tiny``.  A subnormal input endpoint fails
    closed because it violates the reusable-endpoint invariant.
    """
    rounded_left, rounded_right = lax.optimization_barrier(
        (lax.stop_gradient(left), lax.stop_gradient(right))
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        rounded_left * rounded_right
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(jnp.inf, dtype=jnp.float64))
    )
    left_zero, left_subnormal, _, left_finite = _binary64_masks(left)
    right_zero, right_subnormal, _, right_finite = _binary64_masks(right)
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    exact_zero: Bool[Array, "..."] = (left_zero & right_finite) | (
        right_zero & left_finite
    )
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        jnp.isnan(raw),
        jnp.inf,
        jnp.where(
            exact_zero,
            0.0,
            jnp.where(
                left_subnormal | right_subnormal,
                jnp.inf,
                jnp.where(underflow_risk, _minimum_normal(), neighbor),
            ),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    upper: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(
            normal_supported,
            candidate,
            jnp.inf,
        )
    )
    return upper


def downward_divide(
    numerator: Float64[Array, "..."],
    denominator: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Enclose one positive-denominator quotient from below.

    Parameters
    ----------
    numerator : Float64[Array, "..."]
        Normal, zero, or infinite numerator endpoint.
    denominator : Float64[Array, "..."]
        Strictly positive normal or infinite denominator endpoint.

    Returns
    -------
    lower : Float64[Array, "..."]
        Guarded lower quotient endpoint.

    Notes
    -----
    A bitwise zero numerator preserves exact zero.  Invalid denominators,
    subnormal inputs, NaN, and unsupported normal arithmetic fail closed.
    """
    rounded_numerator, rounded_denominator = lax.optimization_barrier(
        (lax.stop_gradient(numerator), lax.stop_gradient(denominator))
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        rounded_numerator / rounded_denominator
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(-jnp.inf, dtype=jnp.float64))
    )
    numerator_zero, numerator_subnormal, _, _ = _binary64_masks(numerator)
    denominator_zero, denominator_subnormal, denominator_negative, _ = (
        _binary64_masks(denominator)
    )
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    denominator_invalid: Bool[Array, "..."] = (
        denominator_zero | denominator_negative | jnp.isnan(denominator)
    )
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        numerator_zero & (~denominator_invalid),
        0.0,
        jnp.where(
            denominator_invalid
            | numerator_subnormal
            | denominator_subnormal
            | jnp.isnan(raw),
            -jnp.inf,
            jnp.where(underflow_risk, -_minimum_normal(), neighbor),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    lower: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, -jnp.inf)
    )
    return lower


def upward_divide(
    numerator: Float64[Array, "..."],
    denominator: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """Enclose one positive-denominator quotient from above.

    Parameters
    ----------
    numerator : Float64[Array, "..."]
        Normal, zero, or infinite numerator endpoint.
    denominator : Float64[Array, "..."]
        Strictly positive normal or infinite denominator endpoint.

    Returns
    -------
    upper : Float64[Array, "..."]
        Guarded upper quotient endpoint.

    Notes
    -----
    A bitwise zero numerator preserves exact zero.  Invalid denominators,
    subnormal inputs, NaN, and unsupported normal arithmetic fail closed.
    """
    rounded_numerator, rounded_denominator = lax.optimization_barrier(
        (lax.stop_gradient(numerator), lax.stop_gradient(denominator))
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        rounded_numerator / rounded_denominator
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(jnp.inf, dtype=jnp.float64))
    )
    numerator_zero, numerator_subnormal, _, _ = _binary64_masks(numerator)
    denominator_zero, denominator_subnormal, denominator_negative, _ = (
        _binary64_masks(denominator)
    )
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    denominator_invalid: Bool[Array, "..."] = (
        denominator_zero | denominator_negative | jnp.isnan(denominator)
    )
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        numerator_zero & (~denominator_invalid),
        0.0,
        jnp.where(
            denominator_invalid
            | numerator_subnormal
            | denominator_subnormal
            | jnp.isnan(raw),
            jnp.inf,
            jnp.where(underflow_risk, _minimum_normal(), neighbor),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    upper: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, jnp.inf)
    )
    return upper


def downward_sqrt(value: Float64[Array, "..."]) -> Float64[Array, "..."]:
    """Enclose one nonnegative square root from below.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Nonnegative normal, zero, or infinite radicand endpoint.

    Returns
    -------
    lower : Float64[Array, "..."]
        Guarded lower square-root endpoint.

    Notes
    -----
    A bitwise zero radicand preserves exact zero.  Negative, subnormal, NaN,
    or unsupported inputs fail closed to negative infinity.
    """
    rounded_value: Float64[Array, "..."] = lax.optimization_barrier(
        lax.stop_gradient(value)
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.sqrt(rounded_value)
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(-jnp.inf, dtype=jnp.float64))
    )
    value_zero, value_subnormal, value_negative, _ = _binary64_masks(value)
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    invalid: Bool[Array, "..."] = (
        (value_negative & (~value_zero)) | value_subnormal | jnp.isnan(raw)
    )
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        value_zero,
        0.0,
        jnp.where(
            invalid,
            -jnp.inf,
            jnp.where(underflow_risk, 0.0, neighbor),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    lower: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, -jnp.inf)
    )
    return lower


def upward_sqrt(value: Float64[Array, "..."]) -> Float64[Array, "..."]:
    """Enclose one nonnegative square root from above.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Nonnegative normal, zero, or infinite radicand endpoint.

    Returns
    -------
    upper : Float64[Array, "..."]
        Guarded upper square-root endpoint.

    Notes
    -----
    A bitwise zero radicand preserves exact zero.  Negative, subnormal, NaN,
    or unsupported inputs fail closed to positive infinity.
    """
    rounded_value: Float64[Array, "..."] = lax.optimization_barrier(
        lax.stop_gradient(value)
    )
    raw: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.sqrt(rounded_value)
    )
    neighbor: Float64[Array, "..."] = lax.optimization_barrier(
        jnp.nextafter(raw, jnp.asarray(jnp.inf, dtype=jnp.float64))
    )
    value_zero, value_subnormal, value_negative, _ = _binary64_masks(value)
    raw_zero, raw_subnormal, _, _ = _binary64_masks(raw)
    neighbor_subnormal: Bool[Array, "..."] = _is_nonzero_subnormal(neighbor)
    invalid: Bool[Array, "..."] = (
        (value_negative & (~value_zero)) | value_subnormal | jnp.isnan(raw)
    )
    underflow_risk: Bool[Array, "..."] = (
        raw_zero | raw_subnormal | neighbor_subnormal
    )
    candidate: Float64[Array, "..."] = jnp.where(
        value_zero,
        0.0,
        jnp.where(
            invalid,
            jnp.inf,
            jnp.where(underflow_risk, _minimum_normal(), neighbor),
        ),
    )
    normal_supported: Bool[Array, ""] = all_normal_arithmetic_supported()
    upper: Float64[Array, "..."] = lax.stop_gradient(
        jnp.where(normal_supported, candidate, jnp.inf)
    )
    return upper


def interval_add(left: RealInterval, right: RealInterval) -> RealInterval:
    """Outward-add two reusable real intervals.

    Parameters
    ----------
    left : RealInterval
        Left inclusive interval in a caller-defined unit.
    right : RealInterval
        Right inclusive interval in the same unit as ``left``.

    Returns
    -------
    result : RealInterval
        Inclusive outward interval sum.
    """
    result: RealInterval = (
        _downward_add(left[0], right[0]),
        upward_add(left[1], right[1]),
    )
    return result


def interval_subtract(
    left: RealInterval,
    right: RealInterval,
) -> RealInterval:
    """Outward-subtract two reusable real intervals.

    Parameters
    ----------
    left : RealInterval
        Inclusive minuend interval in a caller-defined unit.
    right : RealInterval
        Inclusive subtrahend interval in the same unit as ``left``.

    Returns
    -------
    result : RealInterval
        Inclusive outward interval difference.
    """
    result: RealInterval = (
        _downward_add(left[0], -right[1]),
        upward_add(left[1], -right[0]),
    )
    return result


def interval_multiply(
    left: RealInterval,
    right: RealInterval,
) -> RealInterval:
    """Outward-multiply two reusable real intervals.

    Parameters
    ----------
    left : RealInterval
        Left inclusive interval in a caller-defined unit.
    right : RealInterval
        Right inclusive interval in a caller-defined unit.

    Returns
    -------
    lower : Float64[Array, "..."]
        Lower endpoint of the inclusive outward product hull.
    upper : Float64[Array, "..."]
        Upper endpoint of the inclusive outward product hull.
    """
    lower_candidates: Float64[Array, "4 ..."] = jnp.stack(
        (
            _downward_multiply(left[0], right[0]),
            _downward_multiply(left[0], right[1]),
            _downward_multiply(left[1], right[0]),
            _downward_multiply(left[1], right[1]),
        )
    )
    upper_candidates: Float64[Array, "4 ..."] = jnp.stack(
        (
            upward_multiply(left[0], right[0]),
            upward_multiply(left[0], right[1]),
            upward_multiply(left[1], right[0]),
            upward_multiply(left[1], right[1]),
        )
    )
    lower: Float64[Array, "..."] = jnp.min(lower_candidates, axis=0)
    upper: Float64[Array, "..."] = jnp.max(upper_candidates, axis=0)
    result: RealInterval = (lower, upper)
    return result


def interval_divide_positive(
    numerator: RealInterval,
    denominator: RealInterval,
) -> RealInterval:
    """Outward-divide by one strictly positive real interval.

    Parameters
    ----------
    numerator : RealInterval
        Inclusive numerator interval in a caller-defined unit.
    denominator : RealInterval
        Inclusive denominator interval required to have a positive lower
        endpoint.

    Returns
    -------
    result : RealInterval
        Inclusive quotient hull, or an unbounded interval on precondition
        failure.
    """
    lower_candidates: Float64[Array, "4 ..."] = jnp.stack(
        (
            downward_divide(numerator[0], denominator[0]),
            downward_divide(numerator[0], denominator[1]),
            downward_divide(numerator[1], denominator[0]),
            downward_divide(numerator[1], denominator[1]),
        )
    )
    upper_candidates: Float64[Array, "4 ..."] = jnp.stack(
        (
            upward_divide(numerator[0], denominator[0]),
            upward_divide(numerator[0], denominator[1]),
            upward_divide(numerator[1], denominator[0]),
            upward_divide(numerator[1], denominator[1]),
        )
    )
    denominator_valid: Bool[Array, "..."] = (
        (denominator[0] > 0.0)
        & (denominator[1] >= denominator[0])
        & (~jnp.isnan(denominator[0]))
        & (~jnp.isnan(denominator[1]))
    )
    lower: Float64[Array, "..."] = jnp.where(
        denominator_valid,
        jnp.min(lower_candidates, axis=0),
        -jnp.inf,
    )
    upper: Float64[Array, "..."] = jnp.where(
        denominator_valid,
        jnp.max(upper_candidates, axis=0),
        jnp.inf,
    )
    result: RealInterval = (
        lax.stop_gradient(lower),
        lax.stop_gradient(upper),
    )
    return result


def interval_square(value: RealInterval) -> RealInterval:
    """Outward-square one reusable real interval.

    Parameters
    ----------
    value : RealInterval
        Inclusive input interval in a caller-defined unit.

    Returns
    -------
    lower : Float64[Array, "..."]
        Lower endpoint of the inclusive squared interval.
    upper : Float64[Array, "..."]
        Upper endpoint of the inclusive squared interval.
    """
    lower_left: Float64[Array, "..."] = _downward_multiply(value[0], value[0])
    lower_right: Float64[Array, "..."] = _downward_multiply(value[1], value[1])
    upper_left: Float64[Array, "..."] = upward_multiply(value[0], value[0])
    upper_right: Float64[Array, "..."] = upward_multiply(value[1], value[1])
    crosses_zero: Bool[Array, "..."] = (value[0] <= 0.0) & (value[1] >= 0.0)
    lower: Float64[Array, "..."] = jnp.where(
        crosses_zero,
        0.0,
        jnp.minimum(lower_left, lower_right),
    )
    upper: Float64[Array, "..."] = jnp.maximum(upper_left, upper_right)
    result: RealInterval = (lower, upper)
    return result


def interval_sqrt(value: RealInterval) -> RealInterval:
    """Outward-square-root one nonnegative real interval.

    Parameters
    ----------
    value : RealInterval
        Inclusive interval required to have a nonnegative lower endpoint.

    Returns
    -------
    result : RealInterval
        Inclusive square-root interval, or an unbounded interval on
        precondition failure.
    """
    valid: Bool[Array, "..."] = (
        (value[0] >= 0.0)
        & (value[1] >= value[0])
        & (~jnp.isnan(value[0]))
        & (~jnp.isnan(value[1]))
    )
    lower: Float64[Array, "..."] = jnp.where(
        valid,
        downward_sqrt(value[0]),
        -jnp.inf,
    )
    upper: Float64[Array, "..."] = jnp.where(
        valid,
        upward_sqrt(value[1]),
        jnp.inf,
    )
    result: RealInterval = (
        lax.stop_gradient(lower),
        lax.stop_gradient(upper),
    )
    return result


__all__: list[str] = [
    "all_normal_arithmetic_supported",
    "arithmetic_environment_probes",
    "downward_divide",
    "downward_sqrt",
    "interval_add",
    "interval_divide_positive",
    "interval_multiply",
    "interval_sqrt",
    "interval_square",
    "interval_subtract",
    "mathematical_pi_interval",
    "point_interval",
    "RealInterval",
    "round_up",
    "upward_add",
    "upward_divide",
    "upward_multiply",
    "upward_sqrt",
]
