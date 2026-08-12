r"""Enclose censored-Poisson differentials with exact rational arithmetic.

Extended Summary
----------------
This host-only leaf encloses the first and second mean derivatives of one
censored-Poisson negative log likelihood and the scalar expected Fisher
information of the complete censored law.  An observation below the ceiling
is one ordinary Poisson symbol; equality with the ceiling denotes the whole
upper tail.  No probability floor, epsilon, floating-point evaluation, or
uncensored substitution is used.

All returned endpoints are :class:`fractions.Fraction` values.  One local
work unit is one issued exact rational binary addition, subtraction,
multiplication, or division.  Calls to the probability enclosure are
separately bounded and retain their complete
:class:`CensoredPoissonWorkTranscript` records.  The common
``maximum_work`` value limits this leaf and each nested probability call
independently; it is not an aggregate nested-work limit.

Routine Listings
----------------
:class:`CensoredPoissonDifferentialError`
    Report one typed bounded differential-enclosure failure.
:class:`CensoredPoissonDifferentialFailure`
    Enumerate disjoint local and nested enclosure failures.
:class:`CensoredPoissonDifferentialWorkTranscript`
    Store deterministic local and nested probability work evidence.
:func:`enclose_censored_poisson_fisher_information`
    Enclose the expected information of one censored Poisson law.
:func:`enclose_censored_poisson_nll_differential`
    Enclose one censored-Poisson NLL score and curvature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import NoReturn

from beartype.typing import Tuple, cast

from .entire_interval import EntireEnclosureFailure
from .host_interval import RationalInterval
from .poisson_interval import (
    CensoredPoissonEnclosureError,
    CensoredPoissonEnclosureFailure,
    CensoredPoissonWorkTranscript,
    enclose_censored_poisson_probability,
)

_DEFAULT_MAXIMUM_COUNT_CEILING: int = 4096
_DEFAULT_MAXIMUM_RATIONAL_BITS: int = 262_144
_DEFAULT_MAXIMUM_RANGE_REDUCTIONS: int = 4096
_DEFAULT_MAXIMUM_TERMS: int = 4096
_DEFAULT_MAXIMUM_WORK: int = 1_000_000
_DEFAULT_PRECISION_BITS: int = 160
_HARD_MAXIMUM_POLICY_VALUE: int = (1 << 63) - 1
_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_REAL_INTERVAL_ENDPOINT_COUNT: int = 2
_ONE: Fraction = Fraction(1)
_ZERO: Fraction = Fraction(0)


class CensoredPoissonDifferentialFailure(str, Enum):
    """Enumerate disjoint local and nested enclosure failures."""

    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    RATIONAL_SIZE_LIMIT = "rational_size_limit"
    NONPOSITIVE_MEAN_LOWER = "nonpositive_mean_lower"
    NONPOSITIVE_TAIL_LOWER = "nonpositive_tail_lower"
    POISSON_ENCLOSURE_FAILURE = "poisson_enclosure_failure"


class CensoredPoissonDifferentialError(ArithmeticError):
    """Report one typed bounded differential-enclosure failure.

    For a local preflight work failure, ``exact_work_count`` is the exact
    required count and no arithmetic has been issued.  Otherwise it is the
    completed local count.  Successful nested probability calls preceding a
    failure remain available in ``poisson_transcripts``.
    """

    failure: CensoredPoissonDifferentialFailure
    exact_work_count: int
    planned_local_work_count: int
    rational_peak_bits: int
    nested_observed_count: int | None
    nested_count_ceiling: int | None
    nested_poisson_failure: CensoredPoissonEnclosureFailure | None
    nested_poisson_exact_work_count: int | None
    nested_entire_kernel: str | None
    nested_entire_failure: EntireEnclosureFailure | None
    nested_entire_exact_work_count: int | None
    poisson_transcripts: Tuple[CensoredPoissonWorkTranscript, ...]

    def __init__(  # noqa: PLR0913
        self,
        failure: CensoredPoissonDifferentialFailure,
        exact_work_count: int,
        message: str,
        *,
        planned_local_work_count: int = 0,
        rational_peak_bits: int = 0,
        nested_observed_count: int | None = None,
        nested_count_ceiling: int | None = None,
        nested_poisson_failure: CensoredPoissonEnclosureFailure | None = None,
        nested_poisson_exact_work_count: int | None = None,
        nested_entire_kernel: str | None = None,
        nested_entire_failure: EntireEnclosureFailure | None = None,
        nested_entire_exact_work_count: int | None = None,
        poisson_transcripts: Tuple[CensoredPoissonWorkTranscript, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.exact_work_count = exact_work_count
        self.planned_local_work_count = planned_local_work_count
        self.rational_peak_bits = rational_peak_bits
        self.nested_observed_count = nested_observed_count
        self.nested_count_ceiling = nested_count_ceiling
        self.nested_poisson_failure = nested_poisson_failure
        self.nested_poisson_exact_work_count = nested_poisson_exact_work_count
        self.nested_entire_kernel = nested_entire_kernel
        self.nested_entire_failure = nested_entire_failure
        self.nested_entire_exact_work_count = nested_entire_exact_work_count
        self.poisson_transcripts = poisson_transcripts


@dataclass(frozen=True)
class CensoredPoissonDifferentialWorkTranscript:
    """Store deterministic local and nested probability work evidence.

    ``exact_work_count`` counts only this leaf's exact rational binary
    operations.  Every nested probability call has its own independently
    enforced local and exponential budgets and is stored in full.
    """

    algorithm: str
    maximum_count_ceiling: int
    maximum_work: int
    maximum_rational_bits: int
    exp_precision_bits: int
    maximum_exp_terms: int
    maximum_exp_work: int
    maximum_exp_range_reductions: int
    count_ceiling: int
    observed_count: int | None
    symbol_count: int
    probability_observed_counts: Tuple[int, ...]
    planned_local_work_count: int
    exact_work_count: int
    rational_peak_bits: int
    poisson_transcripts: Tuple[CensoredPoissonWorkTranscript, ...]


@dataclass(frozen=True)
class _DifferentialPolicy:
    """Store validated local and nested probability resource policies."""

    maximum_count_ceiling: int
    maximum_work: int
    maximum_rational_bits: int
    exp_precision_bits: int
    maximum_exp_terms: int
    maximum_exp_work: int
    maximum_exp_range_reductions: int


@dataclass
class _WorkLedger:
    """Track exact local work, rational size, and nested probability calls."""

    algorithm: str
    policy: _DifferentialPolicy
    count_ceiling: int = 0
    observed_count: int | None = None
    symbol_count: int = 0
    planned_local_work_count: int = 0
    exact_work_count: int = 0
    rational_peak_bits: int = 0
    probability_observed_counts: list[int] = field(default_factory=list)
    poisson_transcripts: list[CensoredPoissonWorkTranscript] = field(
        default_factory=list
    )

    def fail(  # noqa: PLR0913
        self,
        failure: CensoredPoissonDifferentialFailure,
        message: str,
        *,
        reported_work: int | None = None,
        nested_observed_count: int | None = None,
        nested_count_ceiling: int | None = None,
        nested_poisson_failure: (
            CensoredPoissonEnclosureFailure | None
        ) = None,
        nested_poisson_exact_work_count: int | None = None,
        nested_entire_kernel: str | None = None,
        nested_entire_failure: EntireEnclosureFailure | None = None,
        nested_entire_exact_work_count: int | None = None,
    ) -> NoReturn:
        """Raise one typed failure with all completed nested evidence."""
        exact_work = (
            self.exact_work_count if reported_work is None else reported_work
        )
        raise CensoredPoissonDifferentialError(
            failure,
            exact_work,
            message,
            planned_local_work_count=self.planned_local_work_count,
            rational_peak_bits=self.rational_peak_bits,
            nested_observed_count=nested_observed_count,
            nested_count_ceiling=nested_count_ceiling,
            nested_poisson_failure=nested_poisson_failure,
            nested_poisson_exact_work_count=(nested_poisson_exact_work_count),
            nested_entire_kernel=nested_entire_kernel,
            nested_entire_failure=nested_entire_failure,
            nested_entire_exact_work_count=nested_entire_exact_work_count,
            poisson_transcripts=tuple(self.poisson_transcripts),
        )

    def preflight(self, required_work: int) -> None:
        """Bind and check the deterministic local exact-work count."""
        self.planned_local_work_count = required_work
        if required_work > self.policy.maximum_work:
            self.fail(
                CensoredPoissonDifferentialFailure.WORK_BUDGET_EXCEEDED,
                "censored-Poisson differential local-work budget exceeded",
                reported_work=required_work,
            )

    def charge(self) -> None:
        """Charge one exact rational binary operation."""
        attempted = self.exact_work_count + 1
        if attempted > self.policy.maximum_work:
            self.fail(
                CensoredPoissonDifferentialFailure.WORK_BUDGET_EXCEEDED,
                "censored-Poisson differential local-work budget exceeded",
                reported_work=attempted,
            )
        self.exact_work_count = attempted

    def retain(self, value: Fraction) -> Fraction:
        """Check and retain one rational endpoint before further arithmetic."""
        bits = max(
            abs(value.numerator).bit_length(),
            value.denominator.bit_length(),
        )
        self.rational_peak_bits = max(self.rational_peak_bits, bits)
        if bits > self.policy.maximum_rational_bits:
            self.fail(
                CensoredPoissonDifferentialFailure.RATIONAL_SIZE_LIMIT,
                "censored-Poisson differential rational size limit exceeded",
            )
        result: Fraction = value
        return result  # noqa: RET504

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        """Add checked rational operands and retain the exact result."""
        checked_left = self.retain(left)
        checked_right = self.retain(right)
        self.charge()
        result = self.retain(checked_left + checked_right)
        return result  # noqa: RET504

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        """Subtract checked rational operands and retain the exact result."""
        checked_left = self.retain(left)
        checked_right = self.retain(right)
        self.charge()
        result = self.retain(checked_left - checked_right)
        return result  # noqa: RET504

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        """Multiply checked rational operands and retain the exact result."""
        checked_left = self.retain(left)
        checked_right = self.retain(right)
        self.charge()
        result = self.retain(checked_left * checked_right)
        return result  # noqa: RET504

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        """Divide checked rational operands and retain the exact result."""
        checked_numerator = self.retain(numerator)
        checked_denominator = self.retain(denominator)
        self.charge()
        result = self.retain(checked_numerator / checked_denominator)
        return result  # noqa: RET504

    def probability(
        self,
        mean: RationalInterval,
        observed_count: int,
        count_ceiling: int,
    ) -> RationalInterval:
        """Call one bounded probability enclosure and preserve its evidence."""
        try:
            enclosure, transcript = enclose_censored_poisson_probability(
                mean,
                observed_count,
                count_ceiling,
                maximum_count_ceiling=self.policy.maximum_count_ceiling,
                maximum_work=self.policy.maximum_work,
                maximum_rational_bits=self.policy.maximum_rational_bits,
                exp_precision_bits=self.policy.exp_precision_bits,
                maximum_exp_terms=self.policy.maximum_exp_terms,
                maximum_exp_work=self.policy.maximum_exp_work,
                maximum_exp_range_reductions=(
                    self.policy.maximum_exp_range_reductions
                ),
            )
        except CensoredPoissonEnclosureError as error:
            self.fail(
                (CensoredPoissonDifferentialFailure.POISSON_ENCLOSURE_FAILURE),
                "nested censored-Poisson probability enclosure failed",
                nested_observed_count=observed_count,
                nested_count_ceiling=count_ceiling,
                nested_poisson_failure=error.failure,
                nested_poisson_exact_work_count=error.exact_work_count,
                nested_entire_kernel=error.nested_kernel,
                nested_entire_failure=error.nested_failure,
                nested_entire_exact_work_count=(error.nested_exact_work_count),
            )
        checked: RationalInterval = (
            self.retain(enclosure[0]),
            self.retain(enclosure[1]),
        )
        self.probability_observed_counts.append(observed_count)
        self.poisson_transcripts.append(transcript)
        return checked

    def transcript(self) -> CensoredPoissonDifferentialWorkTranscript:
        """Freeze deterministic local and complete nested work evidence."""
        transcript = CensoredPoissonDifferentialWorkTranscript(
            algorithm=self.algorithm,
            maximum_count_ceiling=self.policy.maximum_count_ceiling,
            maximum_work=self.policy.maximum_work,
            maximum_rational_bits=self.policy.maximum_rational_bits,
            exp_precision_bits=self.policy.exp_precision_bits,
            maximum_exp_terms=self.policy.maximum_exp_terms,
            maximum_exp_work=self.policy.maximum_exp_work,
            maximum_exp_range_reductions=(
                self.policy.maximum_exp_range_reductions
            ),
            count_ceiling=self.count_ceiling,
            observed_count=self.observed_count,
            symbol_count=self.symbol_count,
            probability_observed_counts=tuple(
                self.probability_observed_counts
            ),
            planned_local_work_count=self.planned_local_work_count,
            exact_work_count=self.exact_work_count,
            rational_peak_bits=self.rational_peak_bits,
            poisson_transcripts=tuple(self.poisson_transcripts),
        )
        return transcript  # noqa: RET504


def _checked_policy_integer(
    value: object,
    name: str,
    *,
    allow_zero: bool,
) -> int:
    """PRIVATE: Validate one signed-int64 resource policy.

    Parameters
    ----------
    value : object
        Submitted resource value.
    name : str
        Diagnostic resource name.
    allow_zero : bool
        Whether zero is admitted.

    Returns
    -------
    checked : int
        Validated exact Python integer.

    Raises
    ------
    TypeError
        If the value is not exactly a Python integer.
    ValueError
        If the value is outside its signed-int64 structural range.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer")
    minimum = 0 if allow_zero else 1
    if value < minimum or value > _HARD_MAXIMUM_POLICY_VALUE:
        raise ValueError(f"{name} is outside its structural range")
    checked: int = value
    return checked


def _checked_policy(  # noqa: PLR0913
    *,
    maximum_count_ceiling: object,
    maximum_work: object,
    maximum_rational_bits: object,
    exp_precision_bits: object,
    maximum_exp_terms: object,
    maximum_exp_work: object,
    maximum_exp_range_reductions: object,
) -> _DifferentialPolicy:
    """PRIVATE: Validate local and nested probability resource policies.

    Parameters
    ----------
    maximum_count_ceiling : object
        Submitted nonnegative censoring-ceiling cap.
    maximum_work : object
        Submitted positive per-ledger exact-work cap.
    maximum_rational_bits : object
        Submitted positive rational endpoint bit cap.
    exp_precision_bits : object
        Submitted positive nested exponential precision.
    maximum_exp_terms : object
        Submitted positive nested exponential term cap.
    maximum_exp_work : object
        Submitted positive nested exponential work cap.
    maximum_exp_range_reductions : object
        Submitted nonnegative nested range-reduction cap.

    Returns
    -------
    policy : _DifferentialPolicy
        Validated immutable resource policy.

    Raises
    ------
    TypeError
        If a policy is not exactly a Python integer.
    ValueError
        If a policy is outside its structural or hard range.
    CensoredPoissonDifferentialError
        If the nested exponential seed exceeds the rational-size policy.
    """
    ceiling = _checked_policy_integer(
        maximum_count_ceiling,
        "maximum_count_ceiling",
        allow_zero=True,
    )
    work = _checked_policy_integer(
        maximum_work, "maximum_work", allow_zero=False
    )
    bits = _checked_policy_integer(
        maximum_rational_bits,
        "maximum_rational_bits",
        allow_zero=False,
    )
    precision = _checked_policy_integer(
        exp_precision_bits,
        "exp_precision_bits",
        allow_zero=False,
    )
    terms = _checked_policy_integer(
        maximum_exp_terms,
        "maximum_exp_terms",
        allow_zero=False,
    )
    exp_work = _checked_policy_integer(
        maximum_exp_work,
        "maximum_exp_work",
        allow_zero=False,
    )
    reductions = _checked_policy_integer(
        maximum_exp_range_reductions,
        "maximum_exp_range_reductions",
        allow_zero=True,
    )
    if bits <= 1:
        raise ValueError("maximum_rational_bits must exceed one")
    if bits > _HARD_MAXIMUM_RATIONAL_BITS:
        raise ValueError(
            "maximum_rational_bits exceeds the implementation cap"
        )
    if precision + 1 > bits:
        raise CensoredPoissonDifferentialError(
            (CensoredPoissonDifferentialFailure.POISSON_ENCLOSURE_FAILURE),
            0,
            "nested exponential precision exceeds the rational bit limit",
            nested_poisson_failure=(
                CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
            ),
            nested_poisson_exact_work_count=0,
            nested_entire_kernel="exp",
            nested_entire_failure=(EntireEnclosureFailure.RATIONAL_SIZE_LIMIT),
            nested_entire_exact_work_count=0,
        )
    policy = _DifferentialPolicy(
        maximum_count_ceiling=ceiling,
        maximum_work=work,
        maximum_rational_bits=bits,
        exp_precision_bits=precision,
        maximum_exp_terms=terms,
        maximum_exp_work=exp_work,
        maximum_exp_range_reductions=reductions,
    )
    return policy  # noqa: RET504


def _checked_problem(
    mean: object,
    observed_count: object | None,
    count_ceiling: object,
    ledger: _WorkLedger,
) -> Tuple[RationalInterval, int | None, int]:
    """PRIVATE: Validate one exact mean interval and censored symbol.

    Parameters
    ----------
    mean : object
        Submitted two-Fraction nonnegative mean interval.
    observed_count : object | None
        Submitted censored symbol, or None for expected information.
    count_ceiling : object
        Submitted nonnegative censoring ceiling.
    ledger : _WorkLedger
        Active exact local work and rational-size ledger.

    Returns
    -------
    checked_mean : RationalInterval
        Checked ordered nonnegative exact mean interval.
    observed : int | None
        Checked observed symbol, or None for expected information.
    ceiling : int
        Checked exact censoring ceiling.

    Raises
    ------
    TypeError
        If the counts or mean carrier have the wrong exact types.
    ValueError
        If counts or mean endpoints are outside their mathematical domain.
    CensoredPoissonDifferentialError
        If the ceiling or a rational endpoint exceeds its policy.
    """
    ceiling = _checked_policy_integer(
        count_ceiling, "count_ceiling", allow_zero=True
    )
    if observed_count is None:
        observed: int | None = None
    else:
        observed = _checked_policy_integer(
            observed_count, "observed_count", allow_zero=True
        )
        if observed > ceiling:
            raise ValueError("observed_count must not exceed count_ceiling")
    ledger.count_ceiling = ceiling
    ledger.observed_count = observed
    if ceiling > ledger.policy.maximum_count_ceiling:
        ledger.fail(
            CensoredPoissonDifferentialFailure.POISSON_ENCLOSURE_FAILURE,
            "censored-Poisson count ceiling exceeds its policy cap",
            nested_observed_count=observed,
            nested_count_ceiling=ceiling,
            nested_poisson_failure=(
                CensoredPoissonEnclosureFailure.COUNT_CEILING_LIMIT
            ),
            nested_poisson_exact_work_count=0,
        )
    ledger.retain(Fraction(ceiling))
    if observed is not None:
        ledger.retain(Fraction(observed))
    if (
        not isinstance(mean, tuple)
        or len(mean) != _REAL_INTERVAL_ENDPOINT_COUNT
        or any(not isinstance(endpoint, Fraction) for endpoint in mean)
    ):
        raise TypeError("mean interval must contain exactly two Fractions")
    submitted = cast(Tuple[Fraction, Fraction], mean)
    lower = ledger.retain(submitted[0])
    upper = ledger.retain(submitted[1])
    if lower > upper:
        raise ValueError("mean interval endpoints must be ordered")
    if lower < _ZERO:
        raise ValueError("mean interval must be nonnegative")
    checked_mean: RationalInterval = (lower, upper)
    problem: Tuple[RationalInterval, int | None, int] = (
        checked_mean,
        observed,
        ceiling,
    )
    return problem


def _divide_positive(
    numerator: RationalInterval,
    denominator: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Divide a nonnegative interval by a positive interval.

    Parameters
    ----------
    numerator : RationalInterval
        Ordered nonnegative numerator interval.
    denominator : RationalInterval
        Ordered interval with a strictly positive lower endpoint.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    quotient : RationalInterval
        Exact natural interval quotient.

    Raises
    ------
    ZeroDivisionError
        If the denominator lower endpoint is not positive.
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    if denominator[0] <= _ZERO:
        raise ZeroDivisionError("positive interval division requires > 0")
    quotient: RationalInterval = (
        ledger.divide(numerator[0], denominator[1]),
        ledger.divide(numerator[1], denominator[0]),
    )
    return quotient


def _one_minus(
    interval: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Subtract one ordered interval from exact one.

    Parameters
    ----------
    interval : RationalInterval
        Ordered exact rational interval.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    difference : RationalInterval
        Exact natural interval for one minus the input.

    Raises
    ------
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    difference: RationalInterval = (
        ledger.subtract(_ONE, interval[1]),
        ledger.subtract(_ONE, interval[0]),
    )
    return difference


def _square_interval(
    interval: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Enclose the square of one exact rational interval.

    Parameters
    ----------
    interval : RationalInterval
        Ordered exact rational interval.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    square : RationalInterval
        Exact natural interval square.

    Raises
    ------
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    lower_square = ledger.multiply(interval[0], interval[0])
    upper_square = ledger.multiply(interval[1], interval[1])
    if interval[0] <= _ZERO <= interval[1]:
        square: RationalInterval = (
            _ZERO,
            max(lower_square, upper_square),
        )
    else:
        square = (
            min(lower_square, upper_square),
            max(lower_square, upper_square),
        )
    return square


def _multiply_nonnegative(
    left: RationalInterval,
    right: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Multiply two ordered nonnegative rational intervals.

    Parameters
    ----------
    left : RationalInterval
        First ordered nonnegative interval.
    right : RationalInterval
        Second ordered nonnegative interval.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    product : RationalInterval
        Exact natural interval product.

    Raises
    ------
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    product: RationalInterval = (
        ledger.multiply(left[0], right[0]),
        ledger.multiply(left[1], right[1]),
    )
    return product


def _multiply_intervals(
    left: RationalInterval,
    right: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Multiply two general exact rational intervals.

    Parameters
    ----------
    left : RationalInterval
        First ordered exact interval.
    right : RationalInterval
        Second ordered exact interval.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    product : RationalInterval
        Exact natural interval product.

    Raises
    ------
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    products = (
        ledger.multiply(left[0], right[0]),
        ledger.multiply(left[0], right[1]),
        ledger.multiply(left[1], right[0]),
        ledger.multiply(left[1], right[1]),
    )
    product: RationalInterval = (min(products), max(products))
    return product


def _add_intervals(
    left: RationalInterval,
    right: RationalInterval,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Add two exact rational intervals.

    Parameters
    ----------
    left : RationalInterval
        First ordered exact interval.
    right : RationalInterval
        Second ordered exact interval.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    total : RationalInterval
        Exact natural interval sum.

    Raises
    ------
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    total: RationalInterval = (
        ledger.add(left[0], right[0]),
        ledger.add(left[1], right[1]),
    )
    return total


def _unsaturated_differential(
    mean: RationalInterval,
    observed_count: int,
    ledger: _WorkLedger,
) -> Tuple[RationalInterval, RationalInterval]:
    """PRIVATE: Enclose one positive-count unsaturated differential.

    Parameters
    ----------
    mean : RationalInterval
        Ordered interval with a strictly positive lower endpoint.
    observed_count : int
        Positive exact uncensored Poisson symbol.
    ledger : _WorkLedger
        Active exact local work ledger.

    Returns
    -------
    score : RationalInterval
        Exact rational score interval.
    curvature : RationalInterval
        Exact rational curvature interval.

    Raises
    ------
    ZeroDivisionError
        If the mean lower endpoint is not positive.
    CensoredPoissonDifferentialError
        If work or rational-size limits fail.
    """
    count = ledger.retain(Fraction(observed_count))
    ratio = _divide_positive((count, count), mean, ledger)
    score = _one_minus(ratio, ledger)
    curvature = _divide_positive(ratio, mean, ledger)
    result: Tuple[RationalInterval, RationalInterval] = (
        score,
        curvature,
    )
    return result


def _saturated_differential(
    mean: RationalInterval,
    count_ceiling: int,
    ledger: _WorkLedger,
) -> Tuple[RationalInterval, RationalInterval]:
    """PRIVATE: Enclose one positive-ceiling upper-tail differential.

    Parameters
    ----------
    mean : RationalInterval
        Ordered interval with a strictly positive lower endpoint.
    count_ceiling : int
        Positive exact censoring ceiling.
    ledger : _WorkLedger
        Active exact local and nested work ledger.

    Returns
    -------
    score : RationalInterval
        Exact rational upper-tail score interval.
    curvature : RationalInterval
        Exact rational upper-tail curvature interval.

    Raises
    ------
    ZeroDivisionError
        If a required interval denominator is not positive.
    CensoredPoissonDifferentialError
        If the tail lower, work, rational, or nested enclosure fails.
    """
    tail = ledger.probability(mean, count_ceiling, count_ceiling)
    if tail[0] <= _ZERO:
        ledger.fail(
            CensoredPoissonDifferentialFailure.NONPOSITIVE_TAIL_LOWER,
            "censored upper-tail probability lower bound is not positive",
        )
    first_tail_mass = ledger.probability(
        mean, count_ceiling - 1, count_ceiling
    )
    ratio = _divide_positive(first_tail_mass, tail, ledger)
    score: RationalInterval = (-ratio[1], -ratio[0])
    ratio_square = _square_interval(ratio, ledger)
    if count_ceiling == 1:
        linear = ratio
    else:
        predecessor = ledger.retain(Fraction(count_ceiling - 1))
        scaled_reciprocal = _divide_positive(
            (predecessor, predecessor), mean, ledger
        )
        coefficient = _one_minus(scaled_reciprocal, ledger)
        linear = _multiply_intervals(coefficient, ratio, ledger)
    curvature = _add_intervals(ratio_square, linear, ledger)
    result: Tuple[RationalInterval, RationalInterval] = (
        score,
        curvature,
    )
    return result


def enclose_censored_poisson_nll_differential(  # noqa: PLR0913
    mean: RationalInterval,
    observed_count: int,
    count_ceiling: int,
    *,
    maximum_count_ceiling: int = _DEFAULT_MAXIMUM_COUNT_CEILING,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    exp_precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_exp_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_exp_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_exp_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
) -> Tuple[
    RationalInterval,
    RationalInterval,
    CensoredPoissonDifferentialWorkTranscript,
]:
    """Enclose one censored-Poisson NLL score and curvature.

    The score is the derivative of ``-log p_z(mean)`` with respect to the
    uncensored Poisson mean.  For ``z < c``, it is
    ``1 - z / mean`` and its curvature is ``z / mean**2``.  For the
    censored symbol ``z = c >= 1``, let ``T=P(Y>=c)`` and
    ``a=P(Y=c-1)``; the score is ``-a/T`` and the curvature is
    ``(a/T)**2 + (1-(c-1)/mean)*(a/T)``.

    Parameters
    ----------
    mean : RationalInterval
        Ordered nonnegative exact interval for the uncensored Poisson mean.
    observed_count : int
        Exact Python censored symbol in ``0,...,count_ceiling``.
    count_ceiling : int
        Exact Python nonnegative censoring ceiling.
    maximum_count_ceiling : int, optional
        Maximum accepted censoring ceiling; default is 4096.
    maximum_work : int, optional
        Per-ledger local exact-work cap; default is 1,000,000.
    maximum_rational_bits : int, optional
        Shared retained rational bit cap; default is 262,144.
    exp_precision_bits : int, optional
        Nested exponential seed precision; default is 160.
    maximum_exp_terms : int, optional
        Nested exponential per-series term cap; default is 4096.
    maximum_exp_work : int, optional
        Nested exponential exact-work cap; default is 1,000,000.
    maximum_exp_range_reductions : int, optional
        Nested exponential reduction cap; default is 4096.

    Returns
    -------
    score_interval : RationalInterval
        Exact rational enclosure of the NLL first derivative.
    curvature_interval : RationalInterval
        Exact rational enclosure of the NLL second derivative.
    transcript : CensoredPoissonDifferentialWorkTranscript
        Deterministic local and complete nested work evidence.

    Raises
    ------
    TypeError
        If input or resource carriers have the wrong exact types.
    ValueError
        If counts, mean endpoints, or policies are structurally invalid.
    CensoredPoissonDifferentialError
        If the positive domain or a bounded resource enclosure fails.

    Notes
    -----
    The symbols ``c=0,z=0`` and ``c>0,z=0`` are continued
    analytically at mean zero.  Their respective score/curvature pairs are
    exactly ``(0,0)`` and ``(1,0)``.  Every other route requires a
    strictly positive mean lower endpoint.
    """
    policy = _checked_policy(
        maximum_count_ceiling=maximum_count_ceiling,
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        exp_precision_bits=exp_precision_bits,
        maximum_exp_terms=maximum_exp_terms,
        maximum_exp_work=maximum_exp_work,
        maximum_exp_range_reductions=maximum_exp_range_reductions,
    )
    ledger = _WorkLedger(
        "exact_fraction_censored_poisson_nll_differential_v1",
        policy,
    )
    checked_mean, checked_observed, checked_ceiling = _checked_problem(
        mean, observed_count, count_ceiling, ledger
    )
    observed = cast(int, checked_observed)
    ledger.symbol_count = 1
    zero_interval: RationalInterval = (_ZERO, _ZERO)
    if checked_ceiling == 0:
        ledger.preflight(0)
        score_interval = zero_interval
        curvature_interval = zero_interval
    elif observed == 0:
        ledger.preflight(0)
        score_interval = (_ONE, _ONE)
        curvature_interval = zero_interval
    else:
        if checked_mean[0] <= _ZERO:
            ledger.fail(
                CensoredPoissonDifferentialFailure.NONPOSITIVE_MEAN_LOWER,
                "positive-count differential requires positive mean lower",
            )
        saturated = observed == checked_ceiling
        planned_work = 6 if (not saturated or checked_ceiling == 1) else 14
        ledger.preflight(planned_work)
        if saturated:
            score_interval, curvature_interval = _saturated_differential(
                checked_mean, checked_ceiling, ledger
            )
        else:
            score_interval, curvature_interval = _unsaturated_differential(
                checked_mean, observed, ledger
            )
    transcript = ledger.transcript()
    result: Tuple[
        RationalInterval,
        RationalInterval,
        CensoredPoissonDifferentialWorkTranscript,
    ] = (score_interval, curvature_interval, transcript)
    return result


def enclose_censored_poisson_fisher_information(  # noqa: PLR0913
    mean: RationalInterval,
    count_ceiling: int,
    *,
    maximum_count_ceiling: int = _DEFAULT_MAXIMUM_COUNT_CEILING,
    maximum_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    exp_precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_exp_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_exp_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_exp_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
) -> Tuple[
    RationalInterval,
    CensoredPoissonDifferentialWorkTranscript,
]:
    """Enclose the expected information of one censored Poisson law.

    The returned scalar is
    ``sum_z p_z(mean) * score_z(mean)**2`` over all symbols
    ``z=0,...,c``, where ``z=c`` is the upper-tail symbol.  This is
    generally different from the uncensored Poisson value ``1/mean``.

    Parameters
    ----------
    mean : RationalInterval
        Ordered nonnegative exact interval for the uncensored Poisson mean.
    count_ceiling : int
        Exact Python nonnegative censoring ceiling.
    maximum_count_ceiling : int, optional
        Maximum accepted censoring ceiling; default is 4096.
    maximum_work : int, optional
        Per-ledger local exact-work cap; default is 1,000,000.
    maximum_rational_bits : int, optional
        Shared retained rational bit cap; default is 262,144.
    exp_precision_bits : int, optional
        Nested exponential seed precision; default is 160.
    maximum_exp_terms : int, optional
        Nested exponential per-series term cap; default is 4096.
    maximum_exp_work : int, optional
        Nested exponential exact-work cap; default is 1,000,000.
    maximum_exp_range_reductions : int, optional
        Nested exponential reduction cap; default is 4096.

    Returns
    -------
    information_interval : RationalInterval
        Exact rational enclosure of the scalar expected information.
    transcript : CensoredPoissonDifferentialWorkTranscript
        Deterministic local and complete nested work evidence.

    Raises
    ------
    TypeError
        If input or resource carriers have the wrong exact types.
    ValueError
        If the ceiling, mean endpoints, or policies are structurally invalid.
    CensoredPoissonDifferentialError
        If the positive domain or a bounded resource enclosure fails.

    Notes
    -----
    Ceiling zero is the deterministic one-symbol law and has exact
    information zero for every nonnegative mean interval.  Every positive
    ceiling requires a strictly positive mean lower endpoint.
    """
    policy = _checked_policy(
        maximum_count_ceiling=maximum_count_ceiling,
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        exp_precision_bits=exp_precision_bits,
        maximum_exp_terms=maximum_exp_terms,
        maximum_exp_work=maximum_exp_work,
        maximum_exp_range_reductions=maximum_exp_range_reductions,
    )
    ledger = _WorkLedger(
        "exact_fraction_censored_poisson_fisher_information_v1",
        policy,
    )
    checked_mean, _, checked_ceiling = _checked_problem(
        mean, None, count_ceiling, ledger
    )
    ledger.symbol_count = checked_ceiling + 1
    if checked_ceiling == 0:
        ledger.preflight(0)
        information_interval: RationalInterval = (_ZERO, _ZERO)
    else:
        if checked_mean[0] <= _ZERO:
            ledger.fail(
                CensoredPoissonDifferentialFailure.NONPOSITIVE_MEAN_LOWER,
                "positive-ceiling information requires positive mean lower",
            )
        ledger.preflight(10 * checked_ceiling - 4)
        probabilities = tuple(
            ledger.probability(checked_mean, observed_count, checked_ceiling)
            for observed_count in range(checked_ceiling + 1)
        )
        total = probabilities[0]
        for observed_count in range(1, checked_ceiling):
            count = ledger.retain(Fraction(observed_count))
            ratio = _divide_positive((count, count), checked_mean, ledger)
            score = _one_minus(ratio, ledger)
            score_square = _square_interval(score, ledger)
            contribution = _multiply_nonnegative(
                probabilities[observed_count],
                score_square,
                ledger,
            )
            total = _add_intervals(total, contribution, ledger)
        tail = probabilities[checked_ceiling]
        if tail[0] <= _ZERO:
            ledger.fail(
                CensoredPoissonDifferentialFailure.NONPOSITIVE_TAIL_LOWER,
                "censored upper-tail probability lower bound is not positive",
            )
        first_tail_mass = probabilities[checked_ceiling - 1]
        saturated_square = _square_interval(first_tail_mass, ledger)
        saturated_contribution = _divide_positive(
            saturated_square, tail, ledger
        )
        information_interval = _add_intervals(
            total, saturated_contribution, ledger
        )
    transcript = ledger.transcript()
    result: Tuple[
        RationalInterval,
        CensoredPoissonDifferentialWorkTranscript,
    ] = (information_interval, transcript)
    return result


__all__: list[str] = [
    "CensoredPoissonDifferentialError",
    "CensoredPoissonDifferentialFailure",
    "CensoredPoissonDifferentialWorkTranscript",
    "enclose_censored_poisson_fisher_information",
    "enclose_censored_poisson_nll_differential",
]
