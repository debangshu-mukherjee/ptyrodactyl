"""Dependency-neutral internal floating-point range predicates."""

import jax.numpy as jnp
from beartype.typing import Tuple
from jax import Array, lax
from jaxtyping import Bool


def _real_masks(
    values: Array,
) -> Tuple[Bool[Array, "..."], Bool[Array, "..."]]:
    """PRIVATE: Return exact nonzero and subnormal masks for one real array.

    Parameters
    ----------
    values : Array
        Real values to classify from their IEEE component bits.

    Returns
    -------
    nonzero : Bool[Array, "..."]
        Exact nonzero-component mask. Unsupported floating dtypes use
        comparisons instead of component bits.
    subnormal : Bool[Array, "..."]
        Nonzero subnormal-component mask. Unsupported floating dtypes report
        no subnormal values.

    Notes
    -----
    The bit masks ignore the sign bit. Exact positive and negative zero are
    therefore both classified as zero.
    """
    if values.dtype == jnp.float64:
        bits = lax.bitcast_convert_type(values, jnp.uint64)
        exponent_mask = jnp.asarray(0x7FF0000000000000, dtype=jnp.uint64)
        fraction_mask = jnp.asarray(0x000FFFFFFFFFFFFF, dtype=jnp.uint64)
        magnitude_mask = jnp.asarray(0x7FFFFFFFFFFFFFFF, dtype=jnp.uint64)
    elif values.dtype == jnp.float32:
        bits = lax.bitcast_convert_type(values, jnp.uint32)
        exponent_mask = jnp.asarray(0x7F800000, dtype=jnp.uint32)
        fraction_mask = jnp.asarray(0x007FFFFF, dtype=jnp.uint32)
        magnitude_mask = jnp.asarray(0x7FFFFFFF, dtype=jnp.uint32)
    else:
        nonzero: Bool[Array, "..."] = values != 0
        subnormal: Bool[Array, "..."] = jnp.zeros_like(values, dtype=jnp.bool_)
        result: Tuple[Bool[Array, "..."], Bool[Array, "..."]] = (
            nonzero,
            subnormal,
        )
        return result
    nonzero: Bool[Array, "..."] = jnp.bitwise_and(bits, magnitude_mask) != 0
    subnormal: Bool[Array, "..."] = (
        jnp.bitwise_and(bits, exponent_mask) == 0
    ) & (jnp.bitwise_and(bits, fraction_mask) != 0)
    result: Tuple[Bool[Array, "..."], Bool[Array, "..."]] = (
        nonzero,
        subnormal,
    )
    return result


def _real_has_subnormal(values: Array) -> Bool[Array, ""]:
    """PRIVATE: Detect nonzero IEEE subnormal components bitwise.

    Parameters
    ----------
    values : Array
        Real array whose components are inspected.

    Returns
    -------
    result : Bool[Array, ""]
        Whether at least one component is a nonzero IEEE subnormal value.
    """
    _, subnormal = _real_masks(values)
    result: Bool[Array, ""] = jnp.any(subnormal)
    return result


def has_subnormal_components(values: Array) -> Bool[Array, ""]:
    """Return whether a real or complex array has a nonzero subnormal part."""
    real_subnormal: Bool[Array, ""] = _real_has_subnormal(jnp.real(values))
    imaginary_subnormal: Bool[Array, ""] = _real_has_subnormal(
        jnp.imag(values)
    )
    result: Bool[Array, ""] = real_subnormal | imaginary_subnormal
    return result


def has_nonzero_components(values: Array) -> Bool[Array, ""]:
    """Detect any nonzero real or imaginary IEEE component bitwise."""
    real_nonzero, _ = _real_masks(jnp.real(values))
    imaginary_nonzero, _ = _real_masks(jnp.imag(values))
    result: Bool[Array, ""] = jnp.any(real_nonzero | imaginary_nonzero)
    return result


def _real_has_lost_subtraction(
    left: Array,
    right: Array,
    difference: Array,
) -> Bool[Array, ""]:
    """PRIVATE: Detect unequal real values whose difference became zero.

    Parameters
    ----------
    left : Array
        Left operands of the subtraction.
    right : Array
        Right operands of the subtraction.
    difference : Array
        Rounded values produced by ``left - right``.

    Returns
    -------
    lost : Bool[Array, ""]
        Whether any unequal operand pair produced an exact zero difference.

    Notes
    -----
    Signed zeros compare as one zero value. Unsupported floating dtypes use
    floating-point comparisons instead of bit inspection.
    """
    if left.dtype == jnp.float64:
        left_bits = lax.bitcast_convert_type(left, jnp.uint64)
        right_bits = lax.bitcast_convert_type(right, jnp.uint64)
        difference_bits = lax.bitcast_convert_type(difference, jnp.uint64)
        magnitude_mask = jnp.asarray(0x7FFFFFFFFFFFFFFF, dtype=jnp.uint64)
    elif left.dtype == jnp.float32:
        left_bits = lax.bitcast_convert_type(left, jnp.uint32)
        right_bits = lax.bitcast_convert_type(right, jnp.uint32)
        difference_bits = lax.bitcast_convert_type(difference, jnp.uint32)
        magnitude_mask = jnp.asarray(0x7FFFFFFF, dtype=jnp.uint32)
    else:
        lost: Bool[Array, ""] = jnp.any((difference == 0) & (left != right))
        return lost
    left_magnitude = jnp.bitwise_and(left_bits, magnitude_mask)
    right_magnitude = jnp.bitwise_and(right_bits, magnitude_mask)
    difference_magnitude = jnp.bitwise_and(difference_bits, magnitude_mask)
    both_zero = (left_magnitude == 0) & (right_magnitude == 0)
    unequal = (left_bits != right_bits) & (~both_zero)
    lost: Bool[Array, ""] = jnp.any((difference_magnitude == 0) & unequal)
    return lost


def has_lost_subtraction(
    left: Array,
    right: Array,
    difference: Array,
) -> Bool[Array, ""]:
    """Detect a real or complex subtraction flushed from nonzero to zero."""
    checked_left, checked_right, checked_difference = lax.optimization_barrier(
        (left, right, difference)
    )
    lost_real: Bool[Array, ""] = _real_has_lost_subtraction(
        jnp.real(checked_left),
        jnp.real(checked_right),
        jnp.real(checked_difference),
    )
    lost_imaginary: Bool[Array, ""] = _real_has_lost_subtraction(
        jnp.imag(checked_left),
        jnp.imag(checked_right),
        jnp.imag(checked_difference),
    )
    result: Bool[Array, ""] = lax.optimization_barrier(
        lost_real | lost_imaginary
    )
    return result


def has_lost_nonzero_components(
    source: Array,
    mapped: Array,
) -> Bool[Array, ""]:
    """Detect a nonzero component mapped to zero or subnormal magnitude."""
    source_real_nonzero, _ = _real_masks(jnp.real(source))
    source_imaginary_nonzero, _ = _real_masks(jnp.imag(source))
    mapped_real_nonzero, mapped_real_subnormal = _real_masks(jnp.real(mapped))
    mapped_imaginary_nonzero, mapped_imaginary_subnormal = _real_masks(
        jnp.imag(mapped)
    )
    lost_real: Bool[Array, "..."] = source_real_nonzero & (
        (~mapped_real_nonzero) | mapped_real_subnormal
    )
    lost_imaginary: Bool[Array, "..."] = source_imaginary_nonzero & (
        (~mapped_imaginary_nonzero) | mapped_imaginary_subnormal
    )
    result: Bool[Array, ""] = jnp.any(lost_real | lost_imaginary)
    return result
