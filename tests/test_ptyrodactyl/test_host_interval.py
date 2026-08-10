"""Exact regressions for the private reusable host interval core."""

import math
from fractions import Fraction

import numpy as np
import pytest
from beartype.typing import Tuple

import ptyrodactyl._host_interval as host_interval
import ptyrodactyl.born.coefficient_certification as vc_certificate

_EXTRACTED_NAMES: Tuple[str, ...] = (
    "_RealInterval",
    "_ComplexRectangle",
    "_PI_TARGET_BITS",
    "_TAYLOR_UPPER_LAST_INDEX",
    "_TAYLOR_LOWER_LAST_INDEX",
    "_MINIMUM_NORMAL",
    "_BINARY64_RADIX",
    "_BINARY64_SIGNIFICAND_BITS",
    "_BINARY64_MAX_EXPONENT",
    "_BINARY64_MIN_EXPONENT",
    "_QUADRANT_COUNT",
    "_HALF_TURN_QUADRANT",
    "_THREE_QUARTER_TURN_QUADRANT",
    "_RootEnclosureError",
    "_host_binary64_supported",
    "_fraction_from_float",
    "_atan_inverse_bounds",
    "_pi_bounds",
    "_sine_partial",
    "_cosine_partial",
    "_first_quadrant_sine_cosine",
    "_negate_interval",
    "_rational_turn_exponential",
    "_real_interval_product",
    "_real_interval_add",
    "_real_interval_subtract",
    "_complex_rectangle_multiply",
    "_complex_rectangle_add",
    "_scale_complex_rectangle",
    "_conjugate_rectangle",
    "_pairwise_rectangle_sum",
    "_normal_floor_lower",
    "_normal_floor_upper",
    "_fraction_lower_float",
    "_fraction_upper_float",
    "_coefficient_error_fraction",
    "_floor_log2_fraction",
    "_power_of_two_fraction",
    "_sqrt_fraction_upper",
)


def test_vc_private_surface_is_identity_aliased_to_shared_owner() -> None:
    """Keep every extracted VC underscore name as the identical object."""
    for name in _EXTRACTED_NAMES:
        assert getattr(vc_certificate, name) is getattr(host_interval, name)


def test_symbolic_quadrant_roots_remain_exact() -> None:
    """Preserve exact algebraic roots without a library trig call."""
    expected = (
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(-1), Fraction(-1)),
        (Fraction(-1), Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
    )
    observed = tuple(
        host_interval._rational_turn_exponential(Fraction(turn, 4))
        for turn in range(4)
    )

    assert observed == expected


def test_rectangle_arithmetic_and_pairwise_order_remain_exact() -> None:
    """Regress exact rectangle algebra and deterministic binary reduction."""
    left = (Fraction(1), Fraction(2), Fraction(-1), Fraction(1))
    right = (Fraction(-2), Fraction(3), Fraction(4), Fraction(5))
    product = host_interval._complex_rectangle_multiply(left, right)
    terms = (
        (Fraction(1), Fraction(2), Fraction(3), Fraction(4)),
        (Fraction(-5), Fraction(-4), Fraction(6), Fraction(7)),
        (Fraction(8), Fraction(9), Fraction(-10), Fraction(-9)),
    )

    assert product == (
        Fraction(-9),
        Fraction(11),
        Fraction(1),
        Fraction(13),
    )
    assert host_interval._pairwise_rectangle_sum(terms) == (
        Fraction(4),
        Fraction(7),
        Fraction(-1),
        Fraction(2),
    )


def test_real_interval_square_preserves_sign_geometry() -> None:
    """Square positive, negative, and zero-crossing intervals exactly."""
    assert host_interval._real_interval_square((Fraction(2), Fraction(3))) == (
        Fraction(4),
        Fraction(9),
    )
    assert host_interval._real_interval_square(
        (Fraction(-3), Fraction(-2))
    ) == (Fraction(4), Fraction(9))
    assert host_interval._real_interval_square(
        (Fraction(-2), Fraction(3))
    ) == (Fraction(0), Fraction(9))
    assert host_interval._real_interval_square((Fraction(0), Fraction(0))) == (
        Fraction(0),
        Fraction(0),
    )


def test_normalized_sinc_symbolic_zeros_evenness_and_unwrapped_modes() -> None:
    """Keep sinc symbolic at zeros, even, signed, and fully unwrapped."""
    assert host_interval._normalized_sinc_integer_ratio(0, 5) == (
        Fraction(1),
        Fraction(1),
    )
    for mode in (5, -10, 5 * 10**100):
        assert host_interval._normalized_sinc_integer_ratio(mode, 5) == (
            Fraction(0),
            Fraction(0),
        )

    for mode in (1, 6, 10**80 + 1):
        positive = host_interval._normalized_sinc_integer_ratio(mode, 5)
        negative = host_interval._normalized_sinc_integer_ratio(-mode, 5)
        assert negative == positive

    base = host_interval._normalized_sinc_integer_ratio(1, 5)
    repetitions = 10**40
    huge_mode = 1 + 2 * 5 * repetitions
    huge = host_interval._normalized_sinc_integer_ratio(huge_mode, 5)
    assert base[0] > 0
    assert huge == (base[0] / huge_mode, base[1] / huge_mode)

    negative_lobe = host_interval._normalized_sinc_integer_ratio(6, 5)
    assert negative_lobe[1] < 0

    with pytest.raises(ValueError, match="count must be positive"):
        host_interval._normalized_sinc_integer_ratio(1, 0)


@pytest.mark.parametrize(
    "value",
    (
        Fraction(0),
        Fraction(2),
        Fraction(1, 3),
        Fraction(1 << 800, 3),
        Fraction(3, 1 << 800),
    ),
)
def test_rational_sqrt_bounds_contain_every_scale(value: Fraction) -> None:
    """Return verified two-sided dyadic bounds across rational scales."""
    lower, upper = host_interval._sqrt_fraction_bounds(value)

    assert 0 <= lower <= upper
    assert lower * lower <= value <= upper * upper
    assert host_interval._sqrt_fraction_upper(value) == upper

    exact = host_interval._sqrt_fraction_bounds(Fraction(9, 16))
    assert exact == (Fraction(3, 4), Fraction(3, 4))
    with pytest.raises(ValueError, match="must be non-negative"):
        host_interval._sqrt_fraction_bounds(Fraction(-1))


def test_outward_binary64_and_verified_sqrt_regressions() -> None:
    """Contain hard dyadic boundaries, underflow, and rational roots."""
    one_third = Fraction(1, 3)
    lower = host_interval._fraction_lower_float(one_third)
    upper = host_interval._fraction_upper_float(one_third)
    tiny = float.fromhex("0x1.0000000000000p-1022")
    below_subnormal = Fraction(1, 1 << 1075)

    assert Fraction.from_float(lower) <= one_third
    assert one_third <= Fraction.from_float(upper)
    assert math.nextafter(lower, math.inf) == upper
    assert host_interval._fraction_lower_float(below_subnormal) == 0.0
    assert host_interval._fraction_upper_float(below_subnormal) == tiny

    exact_root = host_interval._sqrt_fraction_upper(Fraction(9, 16))
    irrational_root_upper = host_interval._sqrt_fraction_upper(Fraction(2))
    assert exact_root == Fraction(3, 4)
    assert irrational_root_upper * irrational_root_upper >= 2
    assert irrational_root_upper * irrational_root_upper - 2 < Fraction(
        1,
        1 << 126,
    )


def test_point_to_rectangle_error_remains_exact() -> None:
    """Retain the VC checker’s exact complex-L1 error calculation."""
    rectangle = (
        Fraction(3, 4),
        Fraction(5, 4),
        Fraction(1, 4),
        Fraction(3, 4),
    )
    error = host_interval._coefficient_error_fraction(
        np.complex128(1.0 + 0.5j),
        rectangle,
    )

    assert error == Fraction(1, 2)
