"""Adversarial tests for the private FTZ-safe interval core."""

import importlib
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple

from ptyrodactyl._interval import (
    _all_normal_arithmetic_supported,
    _arithmetic_environment_probes,
    _interval_add,
    _interval_divide_positive,
    _interval_multiply,
    _interval_sqrt,
    _interval_square,
    _interval_subtract,
    _mathematical_pi_interval,
    _minimum_normal,
    _point_interval,
)


def _fraction(value: float | np.float64) -> Fraction:
    """Return the exact rational represented by one host binary64 value."""
    return Fraction.from_float(float(value))


def _point(value: float) -> Tuple[jax.Array, jax.Array]:
    """Embed one host binary64 value in the production point interval."""
    stored = jnp.asarray(value, dtype=jnp.float64)
    return _point_interval(stored)


def _assert_contains(
    interval: Tuple[jax.Array, jax.Array],
    exact: Fraction,
) -> None:
    """Assert that host interval endpoints contain one exact rational."""
    lower = float(np.asarray(interval[0]))
    upper = float(np.asarray(interval[1]))
    if not np.isneginf(lower):
        assert _fraction(lower) <= exact
    if not np.isposinf(upper):
        assert exact <= _fraction(upper)


def test_subnormal_points_embed_signwise_from_bits() -> None:
    """Embed signed DAZ points in normal-endpoint FTZ.2 intervals."""
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    tiny = float.fromhex("0x1.0000000000000p-1022")

    positive = tuple(float(np.asarray(x)) for x in _point(minimum_subnormal))
    negative = tuple(float(np.asarray(x)) for x in _point(-minimum_subnormal))

    assert positive == (0.0, tiny)
    assert negative == (-tiny, 0.0)


def test_exact_zero_identities_and_computed_cancellation_are_distinct() -> (
    None
):
    """Preserve FTZ.3 identities but widen nonidentity cancellation."""
    zero = _point(0.0)
    one = _point(1.0)
    minus_one = _point(-1.0)
    tiny = float(np.asarray(_minimum_normal()))

    assert tuple(float(x) for x in _interval_add(one, zero)) == (1.0, 1.0)
    assert tuple(float(x) for x in _interval_multiply(one, zero)) == (
        0.0,
        0.0,
    )
    assert tuple(float(x) for x in _interval_divide_positive(zero, one)) == (
        0.0,
        0.0,
    )
    assert tuple(float(x) for x in _interval_sqrt(zero)) == (0.0, 0.0)
    assert tuple(float(x) for x in _interval_add(one, minus_one)) == (
        -tiny,
        tiny,
    )
    assert tuple(float(x) for x in _interval_subtract(one, one)) == (
        -tiny,
        tiny,
    )


def test_compiled_probes_admit_ftz_but_require_normal_arithmetic() -> None:
    """Compile required probes while separating gradual underflow."""
    eager_probes = _arithmetic_environment_probes()
    probes = jax.jit(_arithmetic_environment_probes)()
    required = tuple(bool(np.asarray(value)) for value in probes[:6])
    gradual = bool(np.asarray(probes[6]))

    assert required == (True, True, True, True, True, True)
    assert bool(np.asarray(jax.jit(_all_normal_arithmetic_supported)()))
    if jax.default_backend() == "cpu":
        assert not bool(np.asarray(eager_probes[6]))

    tiny = _point(float.fromhex("0x1.0000000000000p-1022"))
    half = _point(0.5)
    lower, upper = jax.jit(_interval_multiply)(tiny, half)
    assert np.isfinite(float(np.asarray(lower)))
    assert np.isfinite(float(np.asarray(upper)))


def test_fraction_containment_adversarial_binary64_cases() -> None:
    """Contain adversarial exact rationals across every algebraic primitive."""
    tiny = float.fromhex("0x1.0000000000000p-1022")
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    next_one = float.fromhex("0x1.0000000000001p+0")

    binary_cases = (
        (_interval_add, 1.0, -1.0, _fraction(1.0) + _fraction(-1.0)),
        (
            _interval_subtract,
            next_one,
            1.0,
            _fraction(next_one) - _fraction(1.0),
        ),
        (
            _interval_multiply,
            tiny,
            0.5,
            _fraction(tiny) * _fraction(0.5),
        ),
        (
            _interval_multiply,
            minimum_subnormal,
            maximum,
            _fraction(minimum_subnormal) * _fraction(maximum),
        ),
        (
            _interval_multiply,
            maximum,
            2.0,
            _fraction(maximum) * _fraction(2.0),
        ),
        (
            _interval_divide_positive,
            tiny,
            maximum,
            _fraction(tiny) / _fraction(maximum),
        ),
        (
            _interval_divide_positive,
            maximum,
            minimum_subnormal,
            _fraction(maximum) / _fraction(minimum_subnormal),
        ),
    )
    for operation, left, right, exact in binary_cases:
        _assert_contains(operation(_point(left), _point(right)), exact)

    squared = _interval_square(_point(-minimum_subnormal))
    _assert_contains(
        squared,
        _fraction(minimum_subnormal) * _fraction(minimum_subnormal),
    )


def test_square_root_and_composed_expression_contain_exact_host_targets() -> (
    None
):
    """Check square roots and a composed interval expression independently."""
    values = (
        2.0,
        float.fromhex("0x1.0000000000000p-1022"),
        float.fromhex("0x0.0000000000001p-1022"),
        float.fromhex("0x1.fffffffffffffp+1023"),
    )
    for value in values:
        lower, upper = _interval_sqrt(_point(value))
        lower_fraction = _fraction(float(np.asarray(lower)))
        upper_fraction = _fraction(float(np.asarray(upper)))
        exact = _fraction(value)
        assert lower_fraction >= 0
        assert lower_fraction * lower_fraction <= exact
        assert exact <= upper_fraction * upper_fraction

    tiny = float.fromhex("0x1.0000000000000p-1022")
    difference = _interval_subtract(_point(1.0), _point(1.0))
    shifted = _interval_add(difference, _point(tiny))
    product = _interval_multiply(shifted, _point(0.5))
    quotient = _interval_divide_positive(product, _point(3.0))
    exact_expression = _fraction(tiny) * _fraction(1.0 / 2.0) / 3
    _assert_contains(quotient, exact_expression)


def test_pi_invalid_forms_and_unsupported_environment_fail_closed(
    monkeypatch,
) -> None:
    """Turn invalid or unsupported arithmetic into infinities."""
    pi_lower, pi_upper = _mathematical_pi_interval()
    decimal_pi = Fraction(
        314159265358979323846264338327950288419716939937510,
        10**50,
    )
    _assert_contains((pi_lower, pi_upper), decimal_pi)

    assert tuple(float(x) for x in _point_interval(jnp.asarray(jnp.nan))) == (
        -jnp.inf,
        jnp.inf,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    minus_one = jnp.asarray(-1.0, dtype=jnp.float64)
    invalid_division = _interval_divide_positive(_point(1.0), (zero, one))
    invalid_root = _interval_sqrt((minus_one, one))
    unresolved_product = _interval_multiply(
        _point(0.0),
        _point_interval(jnp.asarray(jnp.inf)),
    )
    assert np.isneginf(float(np.asarray(invalid_division[0])))
    assert np.isposinf(float(np.asarray(invalid_division[1])))
    assert np.isneginf(float(np.asarray(invalid_root[0])))
    assert np.isposinf(float(np.asarray(invalid_root[1])))
    assert np.isneginf(float(np.asarray(unresolved_product[0])))
    assert np.isposinf(float(np.asarray(unresolved_product[1])))

    interval_core = importlib.import_module("ptyrodactyl._interval")

    def unsupported_normal_arithmetic():
        return jnp.asarray(False)

    monkeypatch.setattr(
        interval_core,
        "_all_normal_arithmetic_supported",
        unsupported_normal_arithmetic,
    )
    unsupported_point = interval_core._point_interval(
        jnp.asarray(1.0, dtype=jnp.float64)
    )
    unsupported = interval_core._interval_add(_point(1.0), _point(2.0))
    assert np.isneginf(float(np.asarray(unsupported_point[0])))
    assert np.isposinf(float(np.asarray(unsupported_point[1])))
    assert np.isneginf(float(np.asarray(unsupported[0])))
    assert np.isposinf(float(np.asarray(unsupported[1])))


def test_compiled_interval_evidence_has_zero_jvp() -> None:
    """Stop tangents before every barrier and outward-neighbor operation."""
    one = jnp.asarray(1.0, dtype=jnp.float64)

    def upper(value):
        return _interval_multiply(
            _point_interval(value),
            _point_interval(one),
        )[1]

    eager = upper(one)
    compiled = jax.jit(upper)(one)
    _, tangent = jax.jvp(upper, (one,), (one,))

    assert float(np.asarray(compiled)) == float(np.asarray(eager))
    assert float(np.asarray(tangent)) == 0.0
