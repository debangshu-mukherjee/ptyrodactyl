r"""Falsification tests for exact rational elementary-function enclosures."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from beartype.typing import Tuple

import ptyrodactyl._tools.entire_interval as entire
from ptyrodactyl._tools.entire_interval import (
    EntireEnclosureError,
    EntireEnclosureFailure,
    enclose_complex_exp,
    enclose_complex_exprel,
    enclose_complex_phi2,
    enclose_real_exp,
    enclose_real_log,
    enclose_real_sin_cos,
    enclose_real_sinh_cosh,
)

_ORACLE_PRECISION: int = 130
_ORACLE_TOLERANCE: Decimal = Decimal("1e-120")
_LOG_ORACLE_PRECISION: int = 180
_PHI2_ORACLE_PRECISION: int = 190
_PHI2_ORACLE_TOLERANCE: Decimal = Decimal("1e-175")

type _DecimalComplex = Tuple[Decimal, Decimal]


def _decimal(value: Fraction) -> Decimal:
    """Convert one exact rational to a high-precision Decimal."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def _decimal_log(value: Fraction) -> Decimal:
    """Evaluate one positive rational logarithm with independent Decimal."""
    with localcontext() as context:
        context.prec = _LOG_ORACLE_PRECISION
        result: Decimal = _decimal(value).ln()
        return +result


def _decimal_multiply(
    left: _DecimalComplex, right: _DecimalComplex
) -> _DecimalComplex:
    """Multiply two high-precision decimal complex values."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _decimal_sin_cos(value: Fraction) -> Tuple[Decimal, Decimal]:
    """Evaluate sine and cosine by 120-plus-digit Decimal series."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        point = _decimal(value)
        square = point * point
        sine = point
        sine_term = point
        cosine = Decimal(1)
        cosine_term = Decimal(1)
        for index in range(1, 1000):
            sine_term *= -square / Decimal((2 * index) * (2 * index + 1))
            cosine_term *= -square / Decimal((2 * index - 1) * (2 * index))
            sine += sine_term
            cosine += cosine_term
            if (
                abs(sine_term) < _ORACLE_TOLERANCE
                and abs(cosine_term) < _ORACLE_TOLERANCE
            ):
                break
        return +sine, +cosine


def _decimal_complex_exp(real: Fraction, imag: Fraction) -> _DecimalComplex:
    """Evaluate one complex exponential with high-precision Decimal."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        magnitude = _decimal(real).exp()
        sine, cosine = _decimal_sin_cos(imag)
        return +(magnitude * cosine), +(magnitude * sine)


def _decimal_complex_exprel(real: Fraction, imag: Fraction) -> _DecimalComplex:
    """Evaluate exprel directly without near-zero cancellation."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        point = (_decimal(real), _decimal(imag))
        total: _DecimalComplex = (Decimal(1), Decimal(0))
        term: _DecimalComplex = total
        for degree in range(1, 1000):
            product = _decimal_multiply(term, point)
            denominator = Decimal(degree + 1)
            term = (product[0] / denominator, product[1] / denominator)
            total = (total[0] + term[0], total[1] + term[1])
            if (
                abs(term[0]) < _ORACLE_TOLERANCE
                and abs(term[1]) < _ORACLE_TOLERANCE
            ):
                break
        return +total[0], +total[1]


def _decimal_complex_phi2(real: Fraction, imag: Fraction) -> _DecimalComplex:
    """Evaluate phi2 directly by an independent 150-plus-digit series."""
    with localcontext() as context:
        context.prec = _PHI2_ORACLE_PRECISION
        point = (_decimal(real), _decimal(imag))
        total: _DecimalComplex = (Decimal(1) / Decimal(2), Decimal(0))
        term: _DecimalComplex = total
        for degree in range(1, 2000):
            product = _decimal_multiply(term, point)
            denominator = Decimal(degree + 2)
            term = (product[0] / denominator, product[1] / denominator)
            total = (total[0] + term[0], total[1] + term[1])
            if (
                abs(term[0]) < _PHI2_ORACLE_TOLERANCE
                and abs(term[1]) < _PHI2_ORACLE_TOLERANCE
            ):
                break
        return +total[0], +total[1]


def _assert_real_contains(
    interval: Tuple[Fraction, Fraction], value: Decimal
) -> None:
    """Require one rational interval to contain a Decimal oracle."""
    oracle = Fraction(value)
    assert interval[0] <= oracle <= interval[1]


def _assert_complex_contains(
    rectangle: Tuple[Fraction, Fraction, Fraction, Fraction],
    value: _DecimalComplex,
) -> None:
    """Require one rational rectangle to contain a Decimal oracle."""
    _assert_real_contains((rectangle[0], rectangle[1]), value[0])
    _assert_real_contains((rectangle[2], rectangle[3]), value[1])


def test_symbolic_zero_pure_real_and_deterministic_transcripts() -> None:
    """Keep exact symbolic branches and deterministic replay evidence."""
    zero = Fraction(0)
    one = Fraction(1)
    real_zero, real_work = enclose_real_exp((zero, zero))
    complex_zero, complex_work = enclose_complex_exp((zero, zero, zero, zero))
    exprel_zero, exprel_work = enclose_complex_exprel((zero, zero, zero, zero))
    sine_zero, cosine_zero, trig_work = enclose_real_sin_cos((zero, zero))
    sinh_zero, cosh_zero, hyperbolic_work = enclose_real_sinh_cosh(
        (zero, zero)
    )

    assert real_zero == (one, one)
    assert complex_zero == (one, one, zero, zero)
    assert exprel_zero == (one, one, zero, zero)
    assert sine_zero == sinh_zero == (zero, zero)
    assert cosine_zero == cosh_zero == (one, one)
    assert all(
        work.exact_work_count == 0
        for work in (
            real_work,
            complex_work,
            exprel_work,
            trig_work,
            hyperbolic_work,
        )
    )

    point = (Fraction(2, 5), Fraction(2, 5))
    first = enclose_real_exp(point, precision_bits=96)
    second = enclose_real_exp(point, precision_bits=96)
    assert first == second
    assert first[1].algorithm == "exact_fraction_real_exp_v1"
    assert first[1].exact_work_count > 0

    real_bounds, _ = enclose_real_exp(point, precision_bits=96)
    complex_bounds, pure_work = enclose_complex_exp(
        (point[0], point[1], zero, zero), precision_bits=96
    )
    assert complex_bounds == (*real_bounds, zero, zero)
    assert pure_work.root_enclosures == 0


@pytest.mark.parametrize(
    "point",
    (
        Fraction(-50),
        Fraction(-3, 2),
        Fraction(0),
        Fraction(7, 3),
        Fraction(50),
    ),
)
def test_real_exp_contains_high_precision_oracles(point: Fraction) -> None:
    """Contain 130-digit real exponential oracles across signs and scale."""
    enclosure, transcript = enclose_real_exp(
        (point, point), precision_bits=192
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        oracle = _decimal(point).exp()
    _assert_real_contains(enclosure, oracle)
    assert transcript.precision_bits == 192
    assert transcript.maximum_rational_bits == 262_144

    crossing, _ = enclose_real_exp(
        (Fraction(-2), Fraction(3)), precision_bits=192
    )
    for sample in (Fraction(-2), Fraction(0), Fraction(3)):
        with localcontext() as context:
            context.prec = _ORACLE_PRECISION
            _assert_real_contains(crossing, _decimal(sample).exp())


def test_complex_exp_and_exprel_contain_rectangular_oracles() -> None:
    """Falsify pure-imaginary, crossing-zero, and complex rectangles."""
    rectangles = (
        (
            Fraction(0),
            Fraction(0),
            Fraction(-3, 4),
            Fraction(-3, 4),
        ),
        (
            Fraction(1, 5),
            Fraction(1, 5),
            Fraction(-2, 3),
            Fraction(-2, 3),
        ),
        (
            Fraction(-1, 10),
            Fraction(1, 5),
            Fraction(-3, 10),
            Fraction(1, 10),
        ),
    )
    for rectangle in rectangles:
        exponential, exp_work = enclose_complex_exp(
            rectangle, precision_bits=192
        )
        exprel, exprel_work = enclose_complex_exprel(
            rectangle, precision_bits=192
        )
        real_samples = (
            rectangle[0],
            (rectangle[0] + rectangle[1]) / 2,
            rectangle[1],
        )
        imag_samples = (
            rectangle[2],
            (rectangle[2] + rectangle[3]) / 2,
            rectangle[3],
        )
        for real in real_samples:
            for imag in imag_samples:
                _assert_complex_contains(
                    exponential, _decimal_complex_exp(real, imag)
                )
                _assert_complex_contains(
                    exprel, _decimal_complex_exprel(real, imag)
                )
        assert exp_work.root_enclosures == 1
        assert exprel_work.root_enclosures == 1


def test_exprel_near_zero_avoids_division_cancellation() -> None:
    """Contain tiny complex phi1 values through its entire power series."""
    scale = 10**50
    real = Fraction(1, scale)
    imag = Fraction(-2, scale)
    enclosure, transcript = enclose_complex_exprel(
        (real, real, imag, imag), precision_bits=224
    )
    _assert_complex_contains(enclosure, _decimal_complex_exprel(real, imag))
    assert enclosure != (Fraction(1), Fraction(1), Fraction(0), Fraction(0))
    assert transcript.series_terms > 0


def test_phi2_zero_complex_and_near_zero_contain_190_digit_oracles() -> None:
    """Prove symbolic zero and contain independent complex phi2 oracles."""
    zero = Fraction(0)
    half = Fraction(1, 2)
    symbolic, symbolic_work = enclose_complex_phi2((zero, zero, zero, zero))
    assert symbolic == (half, half, zero, zero)
    assert symbolic_work.exact_work_count == 0
    assert symbolic_work.algorithm == "exact_fraction_complex_phi2_v1"

    points = (
        (Fraction(2, 5), Fraction(-3, 7)),
        (Fraction(1, 10**80), Fraction(-2, 10**80)),
    )
    term_counts = []
    for real, imag in points:
        enclosure, transcript = enclose_complex_phi2(
            (real, real, imag, imag), precision_bits=256
        )
        _assert_complex_contains(enclosure, _decimal_complex_phi2(real, imag))
        assert transcript.root_enclosures == 1
        term_counts.append(transcript.series_terms)
    assert term_counts[0] > 0

    rectangle = (
        Fraction(-1, 5),
        Fraction(2, 5),
        Fraction(-1, 4),
        Fraction(1, 3),
    )
    enclosure, _ = enclose_complex_phi2(rectangle, precision_bits=256)
    real_midpoint = (rectangle[0] + rectangle[1]) / Fraction(2)
    imag_midpoint = (rectangle[2] + rectangle[3]) / Fraction(2)
    for real in (rectangle[0], real_midpoint, rectangle[1]):
        for imag in (rectangle[2], imag_midpoint, rectangle[3]):
            _assert_complex_contains(
                enclosure, _decimal_complex_phi2(real, imag)
            )


def test_phi2_fails_typed_on_independent_resource_limits() -> None:
    """Bound phi2 terms, exact work, and retained rational size."""
    point = (Fraction(10), Fraction(10), Fraction(1), Fraction(1))
    with pytest.raises(EntireEnclosureError) as term_error:
        enclose_complex_phi2(point, maximum_terms=1)
    assert term_error.value.failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )

    with pytest.raises(EntireEnclosureError) as work_error:
        enclose_complex_phi2(point, maximum_work=1)
    assert work_error.value.failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )

    oversized = Fraction(1 << 30)
    with pytest.raises(EntireEnclosureError) as rational_error:
        enclose_complex_phi2(
            (oversized, oversized, Fraction(1), Fraction(1)),
            precision_bits=8,
            maximum_rational_bits=20,
        )
    assert rational_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )


def test_real_trigonometric_and_hyperbolic_routes_contain_oracles() -> None:
    """Contain real sin/cos and sinh/cosh on intervals crossing zero."""
    trig_interval = (Fraction(-1, 3), Fraction(2, 5))
    sine, cosine, trig_work = enclose_real_sin_cos(
        trig_interval, precision_bits=192
    )
    for sample in (
        trig_interval[0],
        Fraction(0),
        Fraction(1, 10),
        trig_interval[1],
    ):
        oracle_sine, oracle_cosine = _decimal_sin_cos(sample)
        _assert_real_contains(sine, oracle_sine)
        _assert_real_contains(cosine, oracle_cosine)
    assert trig_work.root_enclosures == 1

    hyperbolic_interval = (Fraction(-2), Fraction(3, 2))
    sinh, cosh, hyperbolic_work = enclose_real_sinh_cosh(
        hyperbolic_interval, precision_bits=192
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        for sample in (
            hyperbolic_interval[0],
            Fraction(0),
            hyperbolic_interval[1],
        ):
            positive = _decimal(sample).exp()
            negative = (-_decimal(sample)).exp()
            _assert_real_contains(sinh, (positive - negative) / 2)
            _assert_real_contains(cosh, (positive + negative) / 2)
    assert hyperbolic_work.range_reductions > 0


def test_exponential_composition_is_containment_not_interval_equality() -> (
    None
):
    """Enclose exp(x+y) by both direct and multiplicative interval routes."""
    left = Fraction(1, 3)
    right = Fraction(2, 5)
    left_exp, _ = enclose_real_exp((left, left), precision_bits=128)
    right_exp, _ = enclose_real_exp((right, right), precision_bits=128)
    direct, _ = enclose_real_exp(
        (left + right, left + right), precision_bits=128
    )
    product = (
        left_exp[0] * right_exp[0],
        left_exp[1] * right_exp[1],
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        oracle = _decimal(left + right).exp()
    _assert_real_contains(direct, oracle)
    _assert_real_contains(product, oracle)
    assert direct != product


def test_real_log_symbolic_identity_and_deterministic_replay() -> None:
    """Preserve exact unity, algebraic identities, and replay transcripts."""
    zero = Fraction(0)
    unity, unity_work = enclose_real_log((Fraction(1), Fraction(1)))
    assert unity == (zero, zero)
    assert unity_work.exact_work_count == 0
    assert unity_work.algorithm == "exact_fraction_real_log_atanh_pow2_v1"

    point = Fraction(7, 5)
    first = enclose_real_log((point, point), precision_bits=256)
    second = enclose_real_log((point, point), precision_bits=256)
    changed = enclose_real_log((point, point), precision_bits=257)
    assert first == second
    assert first != changed
    assert first[1].series_terms > 0

    reciprocal, _ = enclose_real_log(
        (Fraction(1, point), Fraction(1, point)), precision_bits=256
    )
    reciprocal_sum = (
        first[0][0] + reciprocal[0],
        first[0][1] + reciprocal[1],
    )
    assert reciprocal_sum[0] <= zero <= reciprocal_sum[1]

    log_two, _ = enclose_real_log(
        (Fraction(2), Fraction(2)), precision_bits=256
    )
    log_power, power_work = enclose_real_log(
        (Fraction(1 << 19), Fraction(1 << 19)), precision_bits=256
    )
    assert log_power == (19 * log_two[0], 19 * log_two[1])
    assert power_work.range_reductions == 19


@pytest.mark.parametrize(
    "point",
    (
        Fraction(1, 1 << 200),
        Fraction(1, 3),
        Fraction(1),
        Fraction(7, 5),
        Fraction(2),
        Fraction(10**100),
    ),
)
def test_real_log_contains_high_precision_oracles(point: Fraction) -> None:
    """Contain independent 180-digit logarithms across signs and scale."""
    enclosure, transcript = enclose_real_log(
        (point, point), precision_bits=320
    )
    _assert_real_contains(enclosure, _decimal_log(point))
    assert enclosure[0] <= enclosure[1]
    assert transcript.precision_bits == 320
    if point < 1:
        assert enclosure[1] < 0
    elif point > 1:
        assert enclosure[0] > 0
    else:
        assert enclosure == (Fraction(0), Fraction(0))


def test_real_log_interval_product_and_exponential_identities() -> None:
    """Check monotone intervals and independent inverse/product identities."""
    interval = (Fraction(1, 8), Fraction(17))
    enclosure, _ = enclose_real_log(interval, precision_bits=256)
    for sample in (
        interval[0],
        Fraction(1),
        Fraction(3, 2),
        Fraction(8),
        interval[1],
    ):
        _assert_real_contains(enclosure, _decimal_log(sample))

    left = Fraction(5, 7)
    right = Fraction(11, 6)
    left_log, _ = enclose_real_log((left, left), precision_bits=224)
    right_log, _ = enclose_real_log((right, right), precision_bits=224)
    product_log, _ = enclose_real_log(
        (left * right, left * right), precision_bits=224
    )
    summed = (
        left_log[0] + right_log[0],
        left_log[1] + right_log[1],
    )
    oracle = _decimal_log(left * right)
    _assert_real_contains(product_log, oracle)
    _assert_real_contains(summed, oracle)

    point = Fraction(3, 7)
    exponential, _ = enclose_real_exp((point, point), precision_bits=96)
    inverse, _ = enclose_real_log(exponential, precision_bits=128)
    assert inverse[0] <= point <= inverse[1]


def test_real_log_domain_and_independent_resource_failures() -> None:
    """Reject invalid domains and fail each bounded logarithm resource."""
    with pytest.raises(TypeError):
        enclose_real_log((Fraction(1), 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ordered"):
        enclose_real_log((Fraction(2), Fraction(1)))
    for invalid in (
        (Fraction(0), Fraction(1)),
        (Fraction(-2), Fraction(-1)),
    ):
        with pytest.raises(ValueError, match="strictly positive"):
            enclose_real_log(invalid)

    with pytest.raises(EntireEnclosureError) as term_error:
        enclose_real_log(
            (Fraction(2), Fraction(2)),
            precision_bits=64,
            maximum_terms=1,
        )
    assert term_error.value.failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )

    with pytest.raises(EntireEnclosureError) as work_error:
        enclose_real_log((Fraction(3, 2), Fraction(3, 2)), maximum_work=1)
    assert work_error.value.failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert work_error.value.exact_work_count == 2

    with pytest.raises(EntireEnclosureError) as range_error:
        enclose_real_log(
            (Fraction(1 << 20), Fraction(1 << 20)),
            maximum_range_reductions=19,
        )
    assert range_error.value.failure is (
        EntireEnclosureFailure.RANGE_REDUCTION_LIMIT
    )
    assert range_error.value.exact_work_count == 0

    oversized = Fraction(1 << 80)
    with pytest.raises(EntireEnclosureError) as input_error:
        enclose_real_log(
            (oversized, oversized),
            precision_bits=16,
            maximum_rational_bits=64,
        )
    assert input_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert input_error.value.exact_work_count == 0

    with pytest.raises(EntireEnclosureError) as intermediate_error:
        enclose_real_log(
            (Fraction(2), Fraction(2)),
            precision_bits=8,
            maximum_rational_bits=9,
        )
    assert intermediate_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert intermediate_error.value.exact_work_count > 0


def test_typed_term_work_range_root_and_rational_size_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail typed on every independently bounded exact-arithmetic resource."""
    with pytest.raises(EntireEnclosureError) as term_error:
        enclose_complex_exp(
            (
                Fraction(10),
                Fraction(10),
                Fraction(1),
                Fraction(1),
            ),
            maximum_terms=1,
        )
    assert term_error.value.failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )

    with pytest.raises(EntireEnclosureError) as work_error:
        enclose_real_exp((Fraction(1), Fraction(1)), maximum_work=1)
    assert work_error.value.failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert work_error.value.exact_work_count == 2

    with pytest.raises(EntireEnclosureError) as range_error:
        enclose_real_exp(
            (Fraction(2), Fraction(2)), maximum_range_reductions=0
        )
    assert range_error.value.failure is (
        EntireEnclosureFailure.RANGE_REDUCTION_LIMIT
    )

    def fail_root(value: Fraction) -> Fraction:
        """Force the rational modulus-root helper to fail."""
        del value
        raise ValueError("forced root failure")

    monkeypatch.setattr(entire, "sqrt_fraction_upper", fail_root)
    with pytest.raises(EntireEnclosureError) as root_error:
        enclose_complex_exp(
            (
                Fraction(0),
                Fraction(0),
                Fraction(1),
                Fraction(1),
            )
        )
    assert root_error.value.failure is (
        EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE
    )

    def oversized_root(value: Fraction) -> Fraction:
        """Force an outward root endpoint beyond the rational-size policy."""
        del value
        return Fraction(1 << 80)

    monkeypatch.setattr(entire, "sqrt_fraction_upper", oversized_root)
    with pytest.raises(EntireEnclosureError) as root_size_error:
        enclose_complex_exp(
            (
                Fraction(0),
                Fraction(0),
                Fraction(1),
                Fraction(1),
            ),
            precision_bits=16,
            maximum_rational_bits=64,
        )
    assert root_size_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )

    oversized = Fraction(1 << 80)
    with pytest.raises(EntireEnclosureError) as input_error:
        enclose_real_exp(
            (oversized, oversized),
            precision_bits=16,
            maximum_rational_bits=64,
        )
    assert input_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert input_error.value.exact_work_count == 0

    with pytest.raises(EntireEnclosureError) as intermediate_error:
        enclose_real_exp(
            (Fraction(10), Fraction(10)),
            precision_bits=12,
            maximum_rational_bits=32,
        )
    assert intermediate_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert intermediate_error.value.exact_work_count > 0
