r"""Enclose elementary functions with exact rational arithmetic.

Extended Summary
----------------
This private leaf provides deterministic, resource-bounded interval kernels
for the real and complex exponential, the entire complex phi1 and phi2
functions, the positive-real natural logarithm, and real trigonometric and
hyperbolic projections.  Every proof endpoint is a :class:`fractions.Fraction`;
floating-point libraries are never consulted.

``precision_bits`` controls only the reduced scalar seed or direct complex
series remainder target.  It does not claim a final-width bound after dyadic
squaring or interval dependency.

One exact-work unit is one issued :class:`fractions.Fraction` binary
addition, subtraction, multiplication, or division, or one call to the
rational square-root enclosure helper.  Comparisons, sign changes, constant
construction, and bit-length checks are free.  ``maximum_rational_bits``
independently bounds every input and retained rational before its next bigint
operation; ``maximum_work`` does not bound bigint size.
``maximum_terms`` bounds each individual series.
``maximum_range_reductions`` bounds each scalar exponential or logarithm,
while the transcript counters sum work across the whole invocation.

Routine Listings
----------------
:class:`EntireEnclosureError`
    Report one typed bounded entire-function enclosure failure.
:class:`EntireEnclosureFailure`
    Enumerate term, work, range, root, and rational-size failures.
:class:`EntireWorkTranscript`
    Store deterministic exact-kernel resource evidence.
:func:`enclose_complex_exp`
    Enclose the complex exponential on one rational rectangle.
:func:`enclose_complex_exprel`
    Enclose the entire complex exprel function on one rectangle.
:func:`enclose_complex_phi2`
    Enclose the entire complex second exponential phi function.
:func:`enclose_real_exp`
    Enclose the real exponential on one rational interval.
:func:`enclose_real_log`
    Enclose the real natural logarithm on one positive rational interval.
:func:`enclose_real_sin_cos`
    Enclose real sine and cosine through the complex exponential.
:func:`enclose_real_sinh_cosh`
    Enclose real hyperbolic sine and cosine through real exponentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from beartype.typing import Tuple, cast

from .host_interval import (
    ComplexRectangle,
    RationalInterval,
    sqrt_fraction_upper,
)

_DEFAULT_MAXIMUM_RANGE_REDUCTIONS: int = 4096
_DEFAULT_MAXIMUM_RATIONAL_BITS: int = 262_144
_DEFAULT_MAXIMUM_TERMS: int = 4096
_DEFAULT_MAXIMUM_WORK: int = 1_000_000
_DEFAULT_PRECISION_BITS: int = 160
_HARD_MAXIMUM_POLICY_VALUE: int = (1 << 63) - 1
_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_COMPLEX_RECTANGLE_ENDPOINT_COUNT: int = 4
_HALF: Fraction = Fraction(1, 2)
_ONE: Fraction = Fraction(1)
_REAL_INTERVAL_ENDPOINT_COUNT: int = 2
_ZERO: Fraction = Fraction(0)
_HALF_RECTANGLE: ComplexRectangle = (_HALF, _HALF, _ZERO, _ZERO)
_ZERO_RECTANGLE: ComplexRectangle = (_ZERO, _ZERO, _ZERO, _ZERO)
_ONE_RECTANGLE: ComplexRectangle = (_ONE, _ONE, _ZERO, _ZERO)


class EntireEnclosureFailure(str, Enum):
    """Enumerate term, work, range, root, and rational-size failures."""

    TERM_BUDGET_EXCEEDED = "term_budget_exceeded"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    RANGE_REDUCTION_LIMIT = "range_reduction_limit"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    RATIONAL_SIZE_LIMIT = "rational_size_limit"


class EntireEnclosureError(ArithmeticError):
    """Report one typed bounded entire-function enclosure failure."""

    failure: EntireEnclosureFailure
    exact_work_count: int
    attempted_exact_work_count: int

    def __init__(
        self,
        failure: EntireEnclosureFailure,
        exact_work_count: int,
        message: str,
        *,
        attempted_exact_work_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.exact_work_count = exact_work_count
        self.attempted_exact_work_count = (
            exact_work_count
            if attempted_exact_work_count is None
            else attempted_exact_work_count
        )


@dataclass(frozen=True)
class EntireWorkTranscript:
    """Store deterministic exact-kernel resource evidence.

    Notes
    -----
    One work unit is one issued exact rational binary arithmetic operation or
    one call to the rational square-root enclosure helper. Comparisons, sign
    changes, constant construction, and rational bit-length checks are free.
    The independent ``maximum_rational_bits`` policy bounds bigint operands;
    a work budget alone does not. ``maximum_terms`` and
    ``maximum_range_reductions`` are per-kernel limits, whereas
    ``series_terms`` and ``range_reductions`` are invocation-wide totals.
    """

    algorithm: str
    precision_bits: int
    maximum_terms: int
    maximum_work: int
    maximum_range_reductions: int
    maximum_rational_bits: int
    series_terms: int
    range_reductions: int
    root_enclosures: int
    rectangle_products: int
    reciprocal_steps: int
    exact_work_count: int


@dataclass(frozen=True)
class _EntirePolicy:
    precision_bits: int
    maximum_terms: int
    maximum_work: int
    maximum_range_reductions: int
    maximum_rational_bits: int


@dataclass
class _WorkLedger:
    algorithm: str
    policy: _EntirePolicy
    series_terms: int = 0
    range_reductions: int = 0
    root_enclosures: int = 0
    rectangle_products: int = 0
    reciprocal_steps: int = 0
    exact_work_count: int = 0

    def fail(self, failure: EntireEnclosureFailure, message: str) -> None:
        """Raise one typed failure with the exact completed-work count."""
        raise EntireEnclosureError(failure, self.exact_work_count, message)

    def charge(self, amount: int = 1) -> None:
        """Charge exact kernel work before performing the operation."""
        attempted = self.exact_work_count + amount
        if attempted > self.policy.maximum_work:
            raise EntireEnclosureError(
                EntireEnclosureFailure.WORK_BUDGET_EXCEEDED,
                self.exact_work_count,
                "entire-function exact-work budget exceeded",
                attempted_exact_work_count=attempted,
            )
        self.exact_work_count = attempted

    def retain(self, value: Fraction) -> Fraction:
        """Reject an oversized rational before its next bigint operation."""
        bits = max(
            abs(value.numerator).bit_length(),
            value.denominator.bit_length(),
        )
        if bits > self.policy.maximum_rational_bits:
            self.fail(
                EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
                "entire-function rational endpoint exceeds its bit limit",
            )
        result: Fraction = value
        return result  # noqa: RET504

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        """Add and retain two checked exact rationals."""
        self.charge()
        result: Fraction = self.retain(left + right)
        return result  # noqa: RET504

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        """Subtract and retain two checked exact rationals."""
        self.charge()
        result: Fraction = self.retain(left - right)
        return result  # noqa: RET504

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        """Multiply and retain two checked exact rationals."""
        self.charge()
        result: Fraction = self.retain(left * right)
        return result  # noqa: RET504

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        """Divide and retain two checked exact rationals."""
        self.charge()
        result: Fraction = self.retain(numerator / denominator)
        return result  # noqa: RET504

    def real_add(
        self, left: RationalInterval, right: RationalInterval
    ) -> RationalInterval:
        """Add two exact real intervals."""
        result: RationalInterval = (
            self.add(left[0], right[0]),
            self.add(left[1], right[1]),
        )
        return result  # noqa: RET504

    def real_subtract(
        self, left: RationalInterval, right: RationalInterval
    ) -> RationalInterval:
        """Subtract two exact real intervals."""
        result: RationalInterval = (
            self.subtract(left[0], right[1]),
            self.subtract(left[1], right[0]),
        )
        return result  # noqa: RET504

    def real_product(
        self, left: RationalInterval, right: RationalInterval
    ) -> RationalInterval:
        """Multiply two exact real intervals."""
        products = (
            self.multiply(left[0], right[0]),
            self.multiply(left[0], right[1]),
            self.multiply(left[1], right[0]),
            self.multiply(left[1], right[1]),
        )
        result: RationalInterval = (min(products), max(products))
        return result  # noqa: RET504

    def real_divide_positive(
        self, value: RationalInterval, denominator: Fraction
    ) -> RationalInterval:
        """Divide a real interval by one exact positive scalar."""
        result: RationalInterval = (
            self.divide(value[0], denominator),
            self.divide(value[1], denominator),
        )
        return result  # noqa: RET504

    def complex_add(
        self, left: ComplexRectangle, right: ComplexRectangle
    ) -> ComplexRectangle:
        """Add two exact complex rectangles."""
        result: ComplexRectangle = (
            self.add(left[0], right[0]),
            self.add(left[1], right[1]),
            self.add(left[2], right[2]),
            self.add(left[3], right[3]),
        )
        return result  # noqa: RET504

    def complex_product(
        self, left: ComplexRectangle, right: ComplexRectangle
    ) -> ComplexRectangle:
        """Multiply two exact complex rectangles."""
        self.rectangle_products += 1
        left_real = (left[0], left[1])
        left_imag = (left[2], left[3])
        right_real = (right[0], right[1])
        right_imag = (right[2], right[3])
        real = self.real_subtract(
            self.real_product(left_real, right_real),
            self.real_product(left_imag, right_imag),
        )
        imag = self.real_add(
            self.real_product(left_real, right_imag),
            self.real_product(left_imag, right_real),
        )
        result: ComplexRectangle = (
            real[0],
            real[1],
            imag[0],
            imag[1],
        )
        return result  # noqa: RET504

    def complex_divide_positive(
        self, value: ComplexRectangle, denominator: Fraction
    ) -> ComplexRectangle:
        """Divide a complex rectangle by one exact positive scalar."""
        result: ComplexRectangle = (
            self.divide(value[0], denominator),
            self.divide(value[1], denominator),
            self.divide(value[2], denominator),
            self.divide(value[3], denominator),
        )
        return result  # noqa: RET504

    def pad_rectangle(
        self, value: ComplexRectangle, radius: Fraction
    ) -> ComplexRectangle:
        """Pad both rectangle components by one nonnegative radius."""
        result: ComplexRectangle = (
            self.subtract(value[0], radius),
            self.add(value[1], radius),
            self.subtract(value[2], radius),
            self.add(value[3], radius),
        )
        return result  # noqa: RET504

    def root_upper(self, value: Fraction) -> Fraction:
        """Call and verify one rational square-root upper enclosure."""
        self.charge()
        self.root_enclosures += 1
        try:
            upper = sqrt_fraction_upper(value)
        except (ArithmeticError, ValueError) as error:
            raise EntireEnclosureError(
                EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
                self.exact_work_count,
                "rational modulus root enclosure failed",
            ) from error
        result: Fraction = self.retain(upper)
        square = self.multiply(result, result)
        if result < 0 or square < value:
            self.fail(
                EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
                "rational modulus root enclosure is not outward",
            )
        return result  # noqa: RET504

    def transcript(self) -> EntireWorkTranscript:
        """Freeze the deterministic resource transcript."""
        result: EntireWorkTranscript = EntireWorkTranscript(
            algorithm=self.algorithm,
            precision_bits=self.policy.precision_bits,
            maximum_terms=self.policy.maximum_terms,
            maximum_work=self.policy.maximum_work,
            maximum_range_reductions=(self.policy.maximum_range_reductions),
            maximum_rational_bits=self.policy.maximum_rational_bits,
            series_terms=self.series_terms,
            range_reductions=self.range_reductions,
            root_enclosures=self.root_enclosures,
            rectangle_products=self.rectangle_products,
            reciprocal_steps=self.reciprocal_steps,
            exact_work_count=self.exact_work_count,
        )
        return result  # noqa: RET504


def _checked_policy(
    precision_bits: int,
    maximum_terms: int,
    maximum_work: int,
    maximum_range_reductions: int,
    maximum_rational_bits: int,
) -> _EntirePolicy:
    """PRIVATE: Validate one exact entire-function resource policy.

    Parameters
    ----------
    precision_bits : int
        Positive direct-series seed remainder precision.
    maximum_terms : int
        Positive per-series maximum of retained nonconstant terms.
    maximum_work : int
        Positive maximum exact-kernel operation count.
    maximum_range_reductions : int
        Nonnegative per-scalar maximum dyadic reduction depth.
    maximum_rational_bits : int
        Positive maximum numerator or denominator bit length.

    Returns
    -------
    policy : _EntirePolicy
        Validated immutable resource policy.

    Raises
    ------
    TypeError
        If any policy input is not a Python integer.
    ValueError
        If any policy input lies outside its structural range.
    EntireEnclosureError
        If the requested seed precision already exceeds the rational limit.
    """
    values = (
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise TypeError("entire-function resource policies must be integers")
    if (
        precision_bits <= 0
        or maximum_terms <= 0
        or maximum_work <= 0
        or maximum_range_reductions < 0
        or maximum_rational_bits <= 1
    ):
        raise ValueError("entire-function resource policies are out of range")
    if any(value > _HARD_MAXIMUM_POLICY_VALUE for value in values):
        raise ValueError(
            "entire-function resource policy exceeds signed int64"
        )
    if maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS:
        raise ValueError(
            "maximum_rational_bits exceeds the implementation cap"
        )
    if precision_bits + 1 > maximum_rational_bits:
        raise EntireEnclosureError(
            EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
            0,
            "precision target exceeds the rational bit limit",
        )
    policy: _EntirePolicy = _EntirePolicy(
        precision_bits=precision_bits,
        maximum_terms=maximum_terms,
        maximum_work=maximum_work,
        maximum_range_reductions=maximum_range_reductions,
        maximum_rational_bits=maximum_rational_bits,
    )
    return policy  # noqa: RET504


def _checked_real_interval(
    value: object, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Validate and size-check one rational real interval.

    Parameters
    ----------
    value : object
        Submitted two-endpoint tuple.
    ledger : _WorkLedger
        Active resource and rational-size ledger.

    Returns
    -------
    interval : RationalInterval
        Ordered exact rational interval.

    Raises
    ------
    TypeError
        If the submission is not exactly two Fraction endpoints.
    ValueError
        If the endpoints are reversed.
    EntireEnclosureError
        If an endpoint exceeds the configured rational bit limit.
    """
    if (
        not isinstance(value, tuple)
        or len(value) != _REAL_INTERVAL_ENDPOINT_COUNT
        or any(not isinstance(endpoint, Fraction) for endpoint in value)
    ):
        raise TypeError("real interval must contain exactly two Fractions")
    submitted = cast(Tuple[Fraction, Fraction], value)
    lower = ledger.retain(submitted[0])
    upper = ledger.retain(submitted[1])
    if lower > upper:
        raise ValueError("real interval endpoints must be ordered")
    interval: RationalInterval = (lower, upper)
    return interval  # noqa: RET504


def _checked_complex_rectangle(
    value: object, ledger: _WorkLedger
) -> ComplexRectangle:
    """PRIVATE: Validate and size-check one rational complex rectangle.

    Parameters
    ----------
    value : object
        Submitted real-lower, real-upper, imaginary-lower, and
        imaginary-upper tuple.
    ledger : _WorkLedger
        Active resource and rational-size ledger.

    Returns
    -------
    rectangle : ComplexRectangle
        Ordered exact rational complex rectangle.

    Raises
    ------
    TypeError
        If the submission is not exactly four Fraction endpoints.
    ValueError
        If either component interval is reversed.
    EntireEnclosureError
        If an endpoint exceeds the configured rational bit limit.
    """
    if (
        not isinstance(value, tuple)
        or len(value) != _COMPLEX_RECTANGLE_ENDPOINT_COUNT
        or any(not isinstance(endpoint, Fraction) for endpoint in value)
    ):
        raise TypeError("complex rectangle must contain four Fractions")
    submitted = cast(ComplexRectangle, value)
    endpoints = tuple(ledger.retain(endpoint) for endpoint in submitted)
    if endpoints[0] > endpoints[1] or endpoints[2] > endpoints[3]:
        raise ValueError(
            "complex rectangle component endpoints must be ordered"
        )
    rectangle: ComplexRectangle = (
        endpoints[0],
        endpoints[1],
        endpoints[2],
        endpoints[3],
    )
    return rectangle  # noqa: RET504


def _target_remainder(ledger: _WorkLedger) -> Fraction:
    """PRIVATE: Construct the exact direct-series remainder target.

    Parameters
    ----------
    ledger : _WorkLedger
        Active validated resource ledger.

    Returns
    -------
    target : Fraction
        Exact ``2**(-precision_bits)`` remainder target.

    Raises
    ------
    EntireEnclosureError
        If the target exceeds the configured rational bit limit.
    """
    target: Fraction = ledger.retain(
        Fraction(1, 1 << ledger.policy.precision_bits)
    )
    return target  # noqa: RET504


def _positive_exp_point(
    value: Fraction, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose ``exp(value)`` for one nonnegative rational point.

    Parameters
    ----------
    value : Fraction
        Checked nonnegative exact argument.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    bounds : RationalInterval
        Exact rational lower and upper exponential bounds.

    Raises
    ------
    EntireEnclosureError
        If term, work, range-reduction, or rational-size limits fail.
    """
    if value == 0:
        bounds: RationalInterval = (_ONE, _ONE)
        return bounds  # noqa: RET504
    reduced = value
    reductions = 0
    while reduced > _HALF:
        if reductions >= ledger.policy.maximum_range_reductions:
            ledger.fail(
                EntireEnclosureFailure.RANGE_REDUCTION_LIMIT,
                "real exponential dyadic reduction limit exceeded",
            )
        reduced = ledger.divide(reduced, Fraction(2))
        reductions += 1
        ledger.range_reductions += 1

    target = _target_remainder(ledger)
    partial = _ONE
    term = _ONE
    degree = 0
    while True:
        first_omitted = ledger.divide(
            ledger.multiply(term, reduced), Fraction(degree + 1)
        )
        ratio = ledger.divide(reduced, Fraction(degree + 2))
        denominator = ledger.subtract(_ONE, ratio)
        remainder = ledger.divide(first_omitted, denominator)
        if remainder <= target:
            lower = partial
            upper = ledger.add(partial, remainder)
            break
        if degree >= ledger.policy.maximum_terms:
            ledger.fail(
                EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
                "real exponential Taylor term budget exceeded",
            )
        partial = ledger.add(partial, first_omitted)
        term = first_omitted
        degree += 1
        ledger.series_terms += 1

    for _ in range(reductions):
        lower = ledger.multiply(lower, lower)
        upper = ledger.multiply(upper, upper)
    bounds: RationalInterval = (lower, upper)
    return bounds  # noqa: RET504


def _exp_point(value: Fraction, ledger: _WorkLedger) -> RationalInterval:
    """PRIVATE: Enclose the real exponential at one checked rational point.

    Parameters
    ----------
    value : Fraction
        Checked exact real argument.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    bounds : RationalInterval
        Exact rational lower and upper exponential bounds.

    Raises
    ------
    EntireEnclosureError
        If any bounded positive-kernel or reciprocal operation fails.
    """
    if value >= 0:
        bounds: RationalInterval = _positive_exp_point(value, ledger)
        return bounds  # noqa: RET504
    positive = _positive_exp_point(-value, ledger)
    ledger.reciprocal_steps += 1
    bounds: RationalInterval = (
        ledger.divide(_ONE, positive[1]),
        ledger.divide(_ONE, positive[0]),
    )
    return bounds  # noqa: RET504


def _real_exp_interval(
    interval: RationalInterval, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose real exp monotonically on one checked interval.

    Parameters
    ----------
    interval : RationalInterval
        Checked ordered exact real interval.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    bounds : RationalInterval
        Monotone exact rational exponential enclosure.

    Raises
    ------
    EntireEnclosureError
        If either endpoint evaluation exceeds a resource limit.
    """
    lower_bounds = _exp_point(interval[0], ledger)
    if interval[0] == interval[1]:
        bounds: RationalInterval = lower_bounds
        return bounds  # noqa: RET504
    upper_bounds = _exp_point(interval[1], ledger)
    bounds: RationalInterval = (lower_bounds[0], upper_bounds[1])
    return bounds  # noqa: RET504


def _floor_log2_positive(value: Fraction) -> int:
    """PRIVATE: Compute the exact base-two exponent of a positive rational.

    Parameters
    ----------
    value : Fraction
        Positive exact rational point.

    Returns
    -------
    exponent : int
        Exact integer ``floor(log2(value))``.

    Raises
    ------
    ValueError
        If ``value`` is not strictly positive.
    """
    if value <= 0:
        raise ValueError("base-two logarithm input must be positive")
    numerator = value.numerator
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < (denominator << exponent):
            exponent -= 1
    elif (numerator << (-exponent)) < denominator:
        exponent -= 1
    return exponent


def _reduce_log_point(
    value: Fraction, ledger: _WorkLedger
) -> Tuple[Fraction, int]:
    """PRIVATE: Reduce one positive point to a unit dyadic mantissa.

    Parameters
    ----------
    value : Fraction
        Positive checked exact rational point.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    mantissa : Fraction
        Exact reduced point in ``[1, 2)``.
    exponent : int
        Exact base-two exponent.

    Raises
    ------
    EntireEnclosureError
        If range-reduction, work, or rational-size limits fail.
    """
    exponent = _floor_log2_positive(value)
    reduction_count = abs(exponent)
    if reduction_count > ledger.policy.maximum_range_reductions:
        ledger.fail(
            EntireEnclosureFailure.RANGE_REDUCTION_LIMIT,
            "real logarithm power-of-two reduction limit exceeded",
        )
    ledger.range_reductions += reduction_count
    if exponent == 0:
        mantissa = value
    else:
        scale = ledger.retain(
            Fraction(1 << exponent)
            if exponent > 0
            else Fraction(1, 1 << (-exponent))
        )
        mantissa = ledger.divide(value, scale)
    reduction: Tuple[Fraction, int] = (mantissa, exponent)
    return reduction  # noqa: RET504


def _atanh_log_mantissa(
    value: Fraction, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose log on one exact point in the unit dyadic bin.

    Parameters
    ----------
    value : Fraction
        Exact rational point in ``[1, 2]``.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    bounds : RationalInterval
        Exact rational lower and upper logarithm bounds.

    Raises
    ------
    ValueError
        If ``value`` lies outside ``[1, 2]``.
    EntireEnclosureError
        If term, work, or rational-size limits fail.
    """
    if value < _ONE or value > Fraction(2):
        raise ValueError("logarithm mantissa must lie in [1, 2]")
    if value == _ONE:
        bounds: RationalInterval = (_ZERO, _ZERO)
        return bounds  # noqa: RET504

    numerator = ledger.subtract(value, _ONE)
    denominator = ledger.add(value, _ONE)
    argument = ledger.divide(numerator, denominator)
    argument_square = ledger.multiply(argument, argument)
    geometric_denominator = ledger.subtract(_ONE, argument_square)
    target = _target_remainder(ledger)
    partial = _ZERO
    power = argument
    terms = 0
    while True:
        if terms >= ledger.policy.maximum_terms:
            ledger.fail(
                EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
                "real logarithm atanh term budget exceeded",
            )
        term = ledger.divide(power, Fraction(2 * terms + 1))
        partial = ledger.add(partial, term)
        terms += 1
        ledger.series_terms += 1
        next_power = ledger.multiply(power, argument_square)
        tail_denominator = ledger.multiply(
            Fraction(2 * terms + 1), geometric_denominator
        )
        series_tail = ledger.divide(next_power, tail_denominator)
        log_tail = ledger.multiply(Fraction(2), series_tail)
        if log_tail <= target:
            lower = ledger.multiply(Fraction(2), partial)
            upper = ledger.add(lower, log_tail)
            bounds = (lower, upper)
            return bounds  # noqa: RET504
        power = next_power


def _compose_log_reduction(
    reduction: Tuple[Fraction, int],
    log_two: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Compose one mantissa logarithm with its dyadic exponent.

    Parameters
    ----------
    reduction : Tuple[Fraction, int]
        Unit-bin mantissa and exact base-two exponent.
    log_two : RationalInterval
        Shared exact rational enclosure of ``log(2)``.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    bounds : RationalInterval
        Exact rational lower and upper point-logarithm bounds.

    Raises
    ------
    EntireEnclosureError
        If work or rational-size limits fail.
    """
    mantissa, exponent = reduction
    mantissa_log = _atanh_log_mantissa(mantissa, ledger)
    if exponent == 0:
        bounds: RationalInterval = mantissa_log
        return bounds  # noqa: RET504
    factor = Fraction(exponent)
    if exponent > 0:
        exponent_log: RationalInterval = (
            ledger.multiply(factor, log_two[0]),
            ledger.multiply(factor, log_two[1]),
        )
    else:
        exponent_log = (
            ledger.multiply(factor, log_two[1]),
            ledger.multiply(factor, log_two[0]),
        )
    bounds = ledger.real_add(mantissa_log, exponent_log)
    return bounds  # noqa: RET504


def _real_log_interval(
    interval: RationalInterval, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose real log monotonically on one positive interval.

    Parameters
    ----------
    interval : RationalInterval
        Checked ordered exact real interval.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    bounds : RationalInterval
        Monotone exact rational logarithm enclosure.

    Raises
    ------
    ValueError
        If the interval is not strictly positive.
    EntireEnclosureError
        If a term, work, range, or rational-size limit fails.
    """
    if interval[0] <= _ZERO:
        raise ValueError("real logarithm interval must be strictly positive")
    lower_reduction = _reduce_log_point(interval[0], ledger)
    if interval[0] == interval[1]:
        upper_reduction = lower_reduction
    else:
        upper_reduction = _reduce_log_point(interval[1], ledger)
    needs_log_two = lower_reduction[1] != 0 or upper_reduction[1] != 0
    log_two: RationalInterval = (
        _atanh_log_mantissa(Fraction(2), ledger)
        if needs_log_two
        else (_ZERO, _ZERO)
    )
    lower_bounds = _compose_log_reduction(lower_reduction, log_two, ledger)
    if interval[0] == interval[1]:
        bounds: RationalInterval = lower_bounds
        return bounds  # noqa: RET504
    upper_bounds = _compose_log_reduction(upper_reduction, log_two, ledger)
    bounds = (lower_bounds[0], upper_bounds[1])
    return bounds  # noqa: RET504


class _ComplexSeriesKind(str, Enum):
    """Select one direct entire complex power series."""

    EXP = "exp"
    EXPREL = "exprel"
    PHI2 = "phi2"


def _modulus_upper(
    rectangle: ComplexRectangle, ledger: _WorkLedger
) -> Fraction:
    """PRIVATE: Enclose the maximum complex modulus on one rectangle.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Checked exact rational complex rectangle.
    ledger : _WorkLedger
        Active bounded exact-work ledger.

    Returns
    -------
    modulus : Fraction
        Verified rational upper bound for every point modulus.

    Raises
    ------
    EntireEnclosureError
        If rational sizing, work, or root enclosure fails.
    """
    real_max = max(abs(rectangle[0]), abs(rectangle[1]))
    imag_max = max(abs(rectangle[2]), abs(rectangle[3]))
    radicand = ledger.add(
        ledger.multiply(real_max, real_max),
        ledger.multiply(imag_max, imag_max),
    )
    modulus: Fraction = ledger.root_upper(radicand)
    return modulus  # noqa: RET504


def _complex_series_rectangle(
    rectangle: ComplexRectangle,
    ledger: _WorkLedger,
    kind: _ComplexSeriesKind,
) -> ComplexRectangle:
    """PRIVATE: Enclose one direct entire complex power series.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Checked exact rational complex argument rectangle.
    ledger : _WorkLedger
        Active bounded exact-work ledger.
    kind : _ComplexSeriesKind
        Exponential, entire exprel, or entire phi2 coefficient recurrence.

    Returns
    -------
    enclosure : ComplexRectangle
        Exact rational direct-series rectangle with a modulus tail pad.

    Raises
    ------
    EntireEnclosureError
        If term, work, root, or rational-size limits fail.
    """
    if kind is _ComplexSeriesKind.PHI2:
        seed = _HALF_RECTANGLE
        magnitude_seed = _HALF
        coefficient_offset = 3
    else:
        seed = _ONE_RECTANGLE
        magnitude_seed = _ONE
        coefficient_offset = 1 if kind is _ComplexSeriesKind.EXP else 2
    if rectangle == _ZERO_RECTANGLE:
        enclosure: ComplexRectangle = seed
        return enclosure  # noqa: RET504
    modulus = _modulus_upper(rectangle, ledger)
    target = _target_remainder(ledger)
    partial = seed
    term = seed
    magnitude_term = magnitude_seed
    degree = 0
    while True:
        next_denominator = Fraction(degree + coefficient_offset)
        ratio_denominator = Fraction(degree + coefficient_offset + 1)
        first_omitted = ledger.divide(
            ledger.multiply(magnitude_term, modulus), next_denominator
        )
        ratio = ledger.divide(modulus, ratio_denominator)
        if ratio < _ONE:
            remainder = ledger.divide(
                first_omitted, ledger.subtract(_ONE, ratio)
            )
            if remainder <= target:
                enclosure: ComplexRectangle = ledger.pad_rectangle(
                    partial, remainder
                )
                return enclosure  # noqa: RET504
        if degree >= ledger.policy.maximum_terms:
            ledger.fail(
                EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
                f"complex {kind.value} Taylor term budget exceeded",
            )
        term = ledger.complex_divide_positive(
            ledger.complex_product(term, rectangle), next_denominator
        )
        partial = ledger.complex_add(partial, term)
        magnitude_term = first_omitted
        degree += 1
        ledger.series_terms += 1


def enclose_real_exp(
    interval: RationalInterval,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[RationalInterval, EntireWorkTranscript]:
    """Enclose the real exponential on one rational interval.

    Parameters
    ----------
    interval : RationalInterval
        Ordered exact rational real interval.
    precision_bits : int, optional
        Direct reduced-seed absolute remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant Taylor terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Per-scalar maximum dyadic halving depth; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational real exponential enclosure.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If interval ordering or resource ranges are invalid.
    EntireEnclosureError
        If an exact bounded enclosure cannot be completed.

    Notes
    -----
    ``precision_bits`` controls the reduced Taylor seed remainder, not the
    final interval width after repeated squaring.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_real_exp_v1", policy)
    checked = _checked_real_interval(interval, ledger)
    enclosure = _real_exp_interval(checked, ledger)
    transcript = ledger.transcript()
    result: Tuple[RationalInterval, EntireWorkTranscript] = (
        enclosure,
        transcript,
    )
    return result  # noqa: RET504


def enclose_real_log(
    interval: RationalInterval,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[RationalInterval, EntireWorkTranscript]:
    """Enclose the real natural logarithm on one positive rational interval.

    Parameters
    ----------
    interval : RationalInterval
        Ordered strictly positive exact rational real interval.
    precision_bits : int, optional
        Reduced atanh-series absolute tail bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Per-scalar maximum absolute dyadic exponent; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational real natural-logarithm enclosure.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If interval ordering, positivity, or resource ranges are invalid.
    EntireEnclosureError
        If an exact bounded enclosure cannot be completed.

    Notes
    -----
    Each endpoint is reduced exactly as ``x = 2**k * m`` with
    ``1 <= m < 2``.  The kernel encloses
    ``log(m) = 2 * sum(z**(2*j + 1) / (2*j + 1))`` for
    ``z = (m - 1) / (m + 1)``.  After retaining through ``j=n``, its
    rigorous positive tail is bounded by
    ``2*z**(2*n + 3) / ((2*n + 3)*(1 - z**2))``.
    ``precision_bits`` controls that reduced-series tail, not the final
    width after multiplication by a large dyadic exponent.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_real_log_atanh_pow2_v1", policy)
    checked = _checked_real_interval(interval, ledger)
    enclosure = _real_log_interval(checked, ledger)
    transcript = ledger.transcript()
    result: Tuple[RationalInterval, EntireWorkTranscript] = (
        enclosure,
        transcript,
    )
    return result  # noqa: RET504


def enclose_complex_exp(
    rectangle: ComplexRectangle,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[ComplexRectangle, EntireWorkTranscript]:
    """Enclose the complex exponential on one rational rectangle.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Ordered exact rational complex rectangle.
    precision_bits : int, optional
        Direct-series absolute remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant Taylor terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Per-scalar pure-real dyadic depth; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    enclosure : ComplexRectangle
        Exact rational complex exponential rectangle.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If rectangle ordering or resource ranges are invalid.
    EntireEnclosureError
        If an exact bounded enclosure cannot be completed.

    Notes
    -----
    The general route is a broad direct rectangle series. It makes no claim
    of sharp monotone bounds. ``precision_bits`` controls only its modulus
    tail target, not dependency width.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_complex_exp_v1", policy)
    checked = _checked_complex_rectangle(rectangle, ledger)
    if checked[2] == 0 and checked[3] == 0:
        real = _real_exp_interval((checked[0], checked[1]), ledger)
        enclosure: ComplexRectangle = (real[0], real[1], _ZERO, _ZERO)
    else:
        enclosure = _complex_series_rectangle(
            checked, ledger, _ComplexSeriesKind.EXP
        )
    transcript = ledger.transcript()
    result: Tuple[ComplexRectangle, EntireWorkTranscript] = (
        enclosure,
        transcript,
    )
    return result  # noqa: RET504


def enclose_complex_exprel(
    rectangle: ComplexRectangle,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[ComplexRectangle, EntireWorkTranscript]:
    """Enclose the entire complex exprel function on one rectangle.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Ordered exact rational complex rectangle.
    precision_bits : int, optional
        Direct-series absolute remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant Taylor terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Reserved per-scalar common-policy dyadic depth; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    enclosure : ComplexRectangle
        Exact rational enclosure of ``phi1(z) = (exp(z) - 1) / z``.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If rectangle ordering or resource ranges are invalid.
    EntireEnclosureError
        If an exact bounded enclosure cannot be completed.

    Notes
    -----
    The direct entire series ``sum(z**k / (k + 1)!)`` avoids division by the
    argument and is symbolically exact at zero. ``precision_bits`` controls
    only the direct modulus-tail target.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_complex_exprel_v1", policy)
    checked = _checked_complex_rectangle(rectangle, ledger)
    enclosure = _complex_series_rectangle(
        checked, ledger, _ComplexSeriesKind.EXPREL
    )
    transcript = ledger.transcript()
    result: Tuple[ComplexRectangle, EntireWorkTranscript] = (
        enclosure,
        transcript,
    )
    return result  # noqa: RET504


def enclose_complex_phi2(
    rectangle: ComplexRectangle,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[ComplexRectangle, EntireWorkTranscript]:
    """Enclose the entire complex second exponential phi function.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Ordered exact rational complex rectangle.
    precision_bits : int, optional
        Direct-series absolute remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant Taylor terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Reserved per-scalar common-policy dyadic depth; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    enclosure : ComplexRectangle
        Exact rational enclosure of
        ``phi2(z) = sum(z**k / (k + 2)!, k >= 0)``.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If rectangle ordering or resource ranges are invalid.
    EntireEnclosureError
        If an exact bounded enclosure cannot be completed.

    Notes
    -----
    The direct entire series is symbolically exact at zero and avoids the
    cancellation-prone quotient ``(phi1(z) - 1) / z``.  After retaining
    through degree ``n``, the modulus tail uses the first omitted term and
    the rigorous geometric ratio bound ``|z| / (n + 4)``.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_complex_phi2_v1", policy)
    checked = _checked_complex_rectangle(rectangle, ledger)
    enclosure = _complex_series_rectangle(
        checked, ledger, _ComplexSeriesKind.PHI2
    )
    transcript = ledger.transcript()
    result: Tuple[ComplexRectangle, EntireWorkTranscript] = (
        enclosure,
        transcript,
    )
    return result  # noqa: RET504


def enclose_real_sin_cos(
    interval: RationalInterval,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[RationalInterval, RationalInterval, EntireWorkTranscript]:
    """Enclose real sine and cosine through the complex exponential.

    Parameters
    ----------
    interval : RationalInterval
        Ordered exact rational real interval.
    precision_bits : int, optional
        Complex direct-series remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant Taylor terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Reserved per-scalar common-policy dyadic depth; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    sine : RationalInterval
        Exact rational sine enclosure.
    cosine : RationalInterval
        Exact rational cosine enclosure.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If interval ordering or resource ranges are invalid.
    EntireEnclosureError
        If the complex exponential route cannot be bounded.

    Notes
    -----
    The result projects a broad direct enclosure of ``exp(i*x)``. It makes
    no sharp monotonicity claim, and ``precision_bits`` controls only the
    direct complex-series remainder target.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_real_sin_cos_v1", policy)
    checked = _checked_real_interval(interval, ledger)
    if checked == (_ZERO, _ZERO):
        sine: RationalInterval = (_ZERO, _ZERO)
        cosine: RationalInterval = (_ONE, _ONE)
        transcript = ledger.transcript()
        zero_result: Tuple[
            RationalInterval,
            RationalInterval,
            EntireWorkTranscript,
        ] = (sine, cosine, transcript)
        return zero_result  # noqa: RET504
    argument: ComplexRectangle = (
        _ZERO,
        _ZERO,
        checked[0],
        checked[1],
    )
    exponential = _complex_series_rectangle(
        argument, ledger, _ComplexSeriesKind.EXP
    )
    sine = (exponential[2], exponential[3])
    cosine = (exponential[0], exponential[1])
    transcript = ledger.transcript()
    result: Tuple[
        RationalInterval,
        RationalInterval,
        EntireWorkTranscript,
    ] = (sine, cosine, transcript)
    return result  # noqa: RET504


def enclose_real_sinh_cosh(
    interval: RationalInterval,
    *,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> Tuple[RationalInterval, RationalInterval, EntireWorkTranscript]:
    """Enclose real hyperbolic sine and cosine through real exponentials.

    Parameters
    ----------
    interval : RationalInterval
        Ordered exact rational real interval.
    precision_bits : int, optional
        Reduced exponential seed remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series maximum retained nonconstant Taylor terms; default is 4096.
    maximum_work : int, optional
        Maximum charged exact-kernel operations; default is 1,000,000.
    maximum_range_reductions : int, optional
        Per-scalar maximum dyadic halving depth; default is 4096.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    sine : RationalInterval
        Exact rational hyperbolic-sine enclosure.
    cosine : RationalInterval
        Exact rational hyperbolic-cosine enclosure.
    transcript : EntireWorkTranscript
        Deterministic resource and algorithm transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If interval ordering or resource ranges are invalid.
    EntireEnclosureError
        If either real exponential route cannot be bounded.

    Notes
    -----
    ``precision_bits`` controls each reduced exponential seed remainder, not
    final interval width after squaring and dependency.
    """
    policy = _checked_policy(
        precision_bits,
        maximum_terms,
        maximum_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    ledger = _WorkLedger("exact_fraction_real_sinh_cosh_v1", policy)
    checked = _checked_real_interval(interval, ledger)
    if checked == (_ZERO, _ZERO):
        sine: RationalInterval = (_ZERO, _ZERO)
        cosine: RationalInterval = (_ONE, _ONE)
        transcript = ledger.transcript()
        zero_result: Tuple[
            RationalInterval,
            RationalInterval,
            EntireWorkTranscript,
        ] = (sine, cosine, transcript)
        return zero_result  # noqa: RET504
    positive = _real_exp_interval(checked, ledger)
    reflected: RationalInterval = (-checked[1], -checked[0])
    negative = _real_exp_interval(reflected, ledger)
    sinh_numerator = ledger.real_subtract(positive, negative)
    cosh_numerator = ledger.real_add(positive, negative)
    sine = ledger.real_divide_positive(sinh_numerator, Fraction(2))
    cosine = ledger.real_divide_positive(cosh_numerator, Fraction(2))
    transcript = ledger.transcript()
    result: Tuple[
        RationalInterval,
        RationalInterval,
        EntireWorkTranscript,
    ] = (sine, cosine, transcript)
    return result  # noqa: RET504


__all__: list[str] = [
    "enclose_complex_exp",
    "enclose_complex_exprel",
    "enclose_complex_phi2",
    "enclose_real_exp",
    "enclose_real_log",
    "enclose_real_sin_cos",
    "enclose_real_sinh_cosh",
    "EntireEnclosureError",
    "EntireEnclosureFailure",
    "EntireWorkTranscript",
]
