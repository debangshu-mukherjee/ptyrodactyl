r"""Enclose censored-Poisson laws with exact rational arithmetic.

Extended Summary
----------------
This private host-only leaf encloses a censored Poisson probability, the
mean of the censored count, and (when its probability enclosure is bounded
strictly away from zero) the corresponding negative log-likelihood.  A
reported count below the ceiling denotes its ordinary Poisson mass; the
ceiling denotes the whole upper tail.  No probability floor is introduced.

All polynomial arithmetic uses :class:`fractions.Fraction`.  Exponential
and logarithm enclosures are delegated to the independently bounded exact
entire-interval kernels.  One local work unit is one issued exact rational
binary addition, subtraction, multiplication, or division.  Nested entire
work is recorded separately and is governed by its own policies.

Routine Listings
----------------
:class:`CensoredPoissonEnclosureError`
    Report one typed bounded censored-Poisson enclosure failure.
:class:`CensoredPoissonEnclosureFailure`
    Enumerate local, nested-kernel, and positivity failures.
:class:`CensoredPoissonWorkTranscript`
    Store deterministic censored-Poisson and nested-kernel work evidence.
:func:`enclose_censored_poisson_mean`
    Enclose the expectation of one censored Poisson count.
:func:`enclose_censored_poisson_nll`
    Enclose one censored Poisson negative log-likelihood.
:func:`enclose_censored_poisson_probability`
    Enclose one probability in a censored Poisson law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import NoReturn

from beartype.typing import Tuple, cast

from .entire_interval import (
    EntireEnclosureError,
    EntireEnclosureFailure,
    EntireWorkTranscript,
    enclose_real_exp,
    enclose_real_log,
)
from .host_interval import RationalInterval

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


class CensoredPoissonEnclosureFailure(str, Enum):
    """Enumerate local, nested-kernel, and positivity failures."""

    COUNT_CEILING_LIMIT = "count_ceiling_limit"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    RATIONAL_SIZE_LIMIT = "rational_size_limit"
    EXPONENTIAL_ENCLOSURE_FAILURE = "exponential_enclosure_failure"
    LOGARITHM_ENCLOSURE_FAILURE = "logarithm_enclosure_failure"
    NONPOSITIVE_PROBABILITY_LOWER = "nonpositive_probability_lower"
    ENCLOSURE_INTERSECTION_FAILURE = "enclosure_intersection_failure"


class CensoredPoissonEnclosureError(ArithmeticError):
    """Report one typed bounded censored-Poisson enclosure failure."""

    failure: CensoredPoissonEnclosureFailure
    exact_work_count: int
    attempted_exact_work_count: int
    nested_kernel: str | None
    nested_failure: EntireEnclosureFailure | None
    nested_exact_work_count: int | None
    nested_attempted_exact_work_count: int | None
    prior_exp_transcripts: Tuple[EntireWorkTranscript, ...]
    prior_log_transcripts: Tuple[EntireWorkTranscript, ...]

    def __init__(
        self,
        failure: CensoredPoissonEnclosureFailure,
        exact_work_count: int,
        message: str,
        *,
        attempted_exact_work_count: int | None = None,
        nested_kernel: str | None = None,
        nested_failure: EntireEnclosureFailure | None = None,
        nested_exact_work_count: int | None = None,
        nested_attempted_exact_work_count: int | None = None,
        prior_exp_transcripts: Tuple[EntireWorkTranscript, ...] = (),
        prior_log_transcripts: Tuple[EntireWorkTranscript, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.exact_work_count = exact_work_count
        self.attempted_exact_work_count = (
            exact_work_count
            if attempted_exact_work_count is None
            else attempted_exact_work_count
        )
        self.nested_kernel = nested_kernel
        self.nested_failure = nested_failure
        self.nested_exact_work_count = nested_exact_work_count
        self.nested_attempted_exact_work_count = (
            nested_exact_work_count
            if nested_attempted_exact_work_count is None
            else nested_attempted_exact_work_count
        )
        self.prior_exp_transcripts = prior_exp_transcripts
        self.prior_log_transcripts = prior_log_transcripts


@dataclass(frozen=True)
class CensoredPoissonWorkTranscript:
    """Store deterministic censored-Poisson and nested-kernel work evidence.

    Notes
    -----
    ``exact_work_count`` counts only local exact polynomial and interval
    arithmetic.  The complete successful exponential and logarithm kernel
    transcripts are stored separately.  ``polynomial_terms`` counts every
    retained Poisson power-series factor, including its constant term.
    """

    algorithm: str
    maximum_count_ceiling: int
    maximum_work: int
    maximum_rational_bits: int
    exp_precision_bits: int
    maximum_exp_terms: int
    maximum_exp_work: int
    maximum_exp_range_reductions: int
    log_precision_bits: int
    maximum_log_terms: int
    maximum_log_work: int
    maximum_log_range_reductions: int
    count_ceiling: int
    observed_count: int | None
    polynomial_terms: int
    endpoint_evaluations: int
    critical_point_evaluations: int
    direct_tail_lower_evaluations: int
    exact_work_count: int
    exp_transcripts: Tuple[EntireWorkTranscript, ...]
    log_transcripts: Tuple[EntireWorkTranscript, ...]


@dataclass(frozen=True)
class _EntirePolicy:
    """PRIVATE: Store one nested entire-kernel resource policy."""

    precision_bits: int
    maximum_terms: int
    maximum_work: int
    maximum_range_reductions: int


@dataclass(frozen=True)
class _PoissonPolicy:
    """PRIVATE: Store validated local and nested resource policies."""

    maximum_count_ceiling: int
    maximum_work: int
    maximum_rational_bits: int
    exp: _EntirePolicy
    log: _EntirePolicy | None


@dataclass
class _WorkLedger:
    """PRIVATE: Track exact local work and successful nested transcripts."""

    algorithm: str
    policy: _PoissonPolicy
    count_ceiling: int
    observed_count: int | None
    polynomial_terms: int = 0
    endpoint_evaluations: int = 0
    critical_point_evaluations: int = 0
    direct_tail_lower_evaluations: int = 0
    exact_work_count: int = 0
    exp_transcripts: list[EntireWorkTranscript] = field(default_factory=list)
    log_transcripts: list[EntireWorkTranscript] = field(default_factory=list)

    def fail(
        self, failure: CensoredPoissonEnclosureFailure, message: str
    ) -> None:
        """Raise one typed local failure at the completed-work count."""
        raise CensoredPoissonEnclosureError(
            failure,
            self.exact_work_count,
            message,
            prior_exp_transcripts=tuple(self.exp_transcripts),
            prior_log_transcripts=tuple(self.log_transcripts),
        )

    def charge(self, amount: int = 1) -> None:
        """Charge exact local work before issuing an operation."""
        attempted = self.exact_work_count + amount
        if attempted > self.policy.maximum_work:
            raise CensoredPoissonEnclosureError(
                CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED,
                self.exact_work_count,
                "censored-Poisson exact-work budget exceeded",
                attempted_exact_work_count=attempted,
                prior_exp_transcripts=tuple(self.exp_transcripts),
                prior_log_transcripts=tuple(self.log_transcripts),
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
                CensoredPoissonEnclosureFailure.RATIONAL_SIZE_LIMIT,
                "censored-Poisson rational endpoint exceeds its bit limit",
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

    def transcript(self) -> CensoredPoissonWorkTranscript:
        """Freeze deterministic local and nested resource evidence."""
        log_policy = self.policy.log
        result = CensoredPoissonWorkTranscript(
            algorithm=self.algorithm,
            maximum_count_ceiling=self.policy.maximum_count_ceiling,
            maximum_work=self.policy.maximum_work,
            maximum_rational_bits=self.policy.maximum_rational_bits,
            exp_precision_bits=self.policy.exp.precision_bits,
            maximum_exp_terms=self.policy.exp.maximum_terms,
            maximum_exp_work=self.policy.exp.maximum_work,
            maximum_exp_range_reductions=(
                self.policy.exp.maximum_range_reductions
            ),
            log_precision_bits=(
                0 if log_policy is None else log_policy.precision_bits
            ),
            maximum_log_terms=(
                0 if log_policy is None else log_policy.maximum_terms
            ),
            maximum_log_work=(
                0 if log_policy is None else log_policy.maximum_work
            ),
            maximum_log_range_reductions=(
                0
                if log_policy is None
                else log_policy.maximum_range_reductions
            ),
            count_ceiling=self.count_ceiling,
            observed_count=self.observed_count,
            polynomial_terms=self.polynomial_terms,
            endpoint_evaluations=self.endpoint_evaluations,
            critical_point_evaluations=self.critical_point_evaluations,
            direct_tail_lower_evaluations=self.direct_tail_lower_evaluations,
            exact_work_count=self.exact_work_count,
            exp_transcripts=tuple(self.exp_transcripts),
            log_transcripts=tuple(self.log_transcripts),
        )
        return result  # noqa: RET504


def _checked_policy_integer(
    value: object,
    name: str,
    *,
    allow_zero: bool,
) -> int:
    """PRIVATE: Validate one signed-int64 resource policy integer.

    Parameters
    ----------
    value : object
        Submitted policy value.
    name : str
        Diagnostic policy name.
    allow_zero : bool
        Whether zero is structurally valid.

    Returns
    -------
    checked : int
        Validated Python integer.

    Raises
    ------
    TypeError
        If the submission is not exactly a Python integer.
    ValueError
        If it is negative, forbidden zero, or exceeds signed int64.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer")
    minimum = 0 if allow_zero else 1
    if value < minimum or value > _HARD_MAXIMUM_POLICY_VALUE:
        raise ValueError(f"{name} is outside its structural range")
    checked: int = value
    return checked


def _checked_nested_policy(
    *,
    precision_bits: object,
    maximum_terms: object,
    maximum_work: object,
    maximum_range_reductions: object,
    maximum_rational_bits: int,
    kernel: str,
    outer_failure: CensoredPoissonEnclosureFailure,
) -> _EntirePolicy:
    """PRIVATE: Validate one nested entire-kernel policy exactly.

    Parameters
    ----------
    precision_bits : object
        Submitted positive seed precision.
    maximum_terms : object
        Submitted positive per-series term budget.
    maximum_work : object
        Submitted positive exact-work budget.
    maximum_range_reductions : object
        Submitted nonnegative range-reduction budget.
    maximum_rational_bits : int
        Validated shared rational-size limit.
    kernel : str
        Nested kernel diagnostic name.
    outer_failure : CensoredPoissonEnclosureFailure
        Typed wrapper failure for this nested kernel.

    Returns
    -------
    policy : _EntirePolicy
        Validated immutable nested policy.

    Raises
    ------
    TypeError
        If a submitted policy is not exactly a Python integer.
    ValueError
        If a submitted policy is outside its structural range.
    CensoredPoissonEnclosureError
        If the seed precision already exceeds the rational-size limit.
    """
    checked_precision = _checked_policy_integer(
        precision_bits, f"{kernel}_precision_bits", allow_zero=False
    )
    checked_terms = _checked_policy_integer(
        maximum_terms, f"maximum_{kernel}_terms", allow_zero=False
    )
    checked_work = _checked_policy_integer(
        maximum_work, f"maximum_{kernel}_work", allow_zero=False
    )
    checked_reductions = _checked_policy_integer(
        maximum_range_reductions,
        f"maximum_{kernel}_range_reductions",
        allow_zero=True,
    )
    if checked_precision + 1 > maximum_rational_bits:
        raise CensoredPoissonEnclosureError(
            outer_failure,
            0,
            f"{kernel} precision target exceeds the rational bit limit",
            nested_kernel=kernel,
            nested_failure=EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
            nested_exact_work_count=0,
            nested_attempted_exact_work_count=0,
        )
    policy = _EntirePolicy(
        precision_bits=checked_precision,
        maximum_terms=checked_terms,
        maximum_work=checked_work,
        maximum_range_reductions=checked_reductions,
    )
    return policy  # noqa: RET504


def _checked_policy(  # noqa: PLR0913
    *,
    maximum_count_ceiling: object,
    maximum_work: object,
    maximum_rational_bits: object,
    exp_precision_bits: object,
    maximum_exp_terms: object,
    maximum_exp_work: object,
    maximum_exp_range_reductions: object,
    log_precision_bits: object | None,
    maximum_log_terms: object | None,
    maximum_log_work: object | None,
    maximum_log_range_reductions: object | None,
) -> _PoissonPolicy:
    """PRIVATE: Validate one complete censored-Poisson resource policy.

    Parameters
    ----------
    maximum_count_ceiling : object
        Submitted nonnegative count-ceiling cap.
    maximum_work : object
        Submitted positive local exact-work budget.
    maximum_rational_bits : object
        Submitted positive rational-size limit.
    exp_precision_bits : object
        Submitted positive exponential precision.
    maximum_exp_terms : object
        Submitted positive exponential term budget.
    maximum_exp_work : object
        Submitted positive exponential work budget.
    maximum_exp_range_reductions : object
        Submitted nonnegative exponential reduction budget.
    log_precision_bits : object | None
        Submitted positive logarithm precision, or ``None`` when unused.
    maximum_log_terms : object | None
        Submitted positive logarithm term budget, or ``None``.
    maximum_log_work : object | None
        Submitted positive logarithm work budget, or ``None``.
    maximum_log_range_reductions : object | None
        Submitted nonnegative logarithm reduction budget, or ``None``.

    Returns
    -------
    policy : _PoissonPolicy
        Validated immutable complete resource policy.

    Raises
    ------
    TypeError
        If any submitted policy has an invalid type.
    ValueError
        If any submitted policy is outside its structural range.
    CensoredPoissonEnclosureError
        If nested precision exceeds the rational-size limit.
    """
    checked_ceiling = _checked_policy_integer(
        maximum_count_ceiling,
        "maximum_count_ceiling",
        allow_zero=True,
    )
    checked_work = _checked_policy_integer(
        maximum_work, "maximum_work", allow_zero=False
    )
    checked_rational_bits = _checked_policy_integer(
        maximum_rational_bits,
        "maximum_rational_bits",
        allow_zero=False,
    )
    if checked_rational_bits <= 1:
        raise ValueError("maximum_rational_bits must exceed one")
    if checked_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS:
        raise ValueError(
            "maximum_rational_bits exceeds the implementation cap"
        )
    exp_policy = _checked_nested_policy(
        precision_bits=exp_precision_bits,
        maximum_terms=maximum_exp_terms,
        maximum_work=maximum_exp_work,
        maximum_range_reductions=maximum_exp_range_reductions,
        maximum_rational_bits=checked_rational_bits,
        kernel="exp",
        outer_failure=(
            CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
        ),
    )
    log_values = (
        log_precision_bits,
        maximum_log_terms,
        maximum_log_work,
        maximum_log_range_reductions,
    )
    if all(value is None for value in log_values):
        log_policy: _EntirePolicy | None = None
    elif any(value is None for value in log_values):
        raise TypeError("all logarithm policies must be supplied together")
    else:
        log_policy = _checked_nested_policy(
            precision_bits=log_precision_bits,
            maximum_terms=maximum_log_terms,
            maximum_work=maximum_log_work,
            maximum_range_reductions=maximum_log_range_reductions,
            maximum_rational_bits=checked_rational_bits,
            kernel="log",
            outer_failure=(
                CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE
            ),
        )
    policy = _PoissonPolicy(
        maximum_count_ceiling=checked_ceiling,
        maximum_work=checked_work,
        maximum_rational_bits=checked_rational_bits,
        exp=exp_policy,
        log=log_policy,
    )
    return policy  # noqa: RET504


def _checked_count(value: object, name: str) -> int:
    """PRIVATE: Validate one nonnegative signed-int64 count.

    Parameters
    ----------
    value : object
        Submitted count.
    name : str
        Diagnostic count name.

    Returns
    -------
    count : int
        Validated nonnegative Python integer.

    Raises
    ------
    TypeError
        If the submission is not exactly a Python integer.
    ValueError
        If the count is negative or exceeds signed int64.
    """
    count = _checked_policy_integer(value, name, allow_zero=True)
    return count  # noqa: RET504


def _checked_mean_interval(
    value: object, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Validate one ordered nonnegative rational mean interval.

    Parameters
    ----------
    value : object
        Submitted two-endpoint tuple.
    ledger : _WorkLedger
        Active local resource and rational-size ledger.

    Returns
    -------
    interval : RationalInterval
        Ordered nonnegative exact rational mean interval.

    Raises
    ------
    TypeError
        If the submission is not exactly two Fraction endpoints.
    ValueError
        If the endpoints are reversed or the lower endpoint is negative.
    CensoredPoissonEnclosureError
        If an endpoint exceeds the rational-size limit.
    """
    if (
        not isinstance(value, tuple)
        or len(value) != _REAL_INTERVAL_ENDPOINT_COUNT
        or any(not isinstance(endpoint, Fraction) for endpoint in value)
    ):
        raise TypeError("mean interval must contain exactly two Fractions")
    submitted = cast(Tuple[Fraction, Fraction], value)
    lower = ledger.retain(submitted[0])
    upper = ledger.retain(submitted[1])
    if lower > upper:
        raise ValueError("mean interval endpoints must be ordered")
    if lower < _ZERO:
        raise ValueError("mean interval must be nonnegative")
    interval: RationalInterval = (lower, upper)
    return interval


def _raise_entire_failure(
    ledger: _WorkLedger,
    error: EntireEnclosureError,
    *,
    kernel: str,
    failure: CensoredPoissonEnclosureFailure,
) -> NoReturn:
    """PRIVATE: Raise while preserving one nested entire-kernel failure.

    Parameters
    ----------
    ledger : _WorkLedger
        Active local exact-work ledger.
    error : EntireEnclosureError
        Nested typed enclosure failure.
    kernel : str
        Nested kernel name.
    failure : CensoredPoissonEnclosureFailure
        Outer wrapper failure category.

    Raises
    ------
    CensoredPoissonEnclosureError
        Always, retaining the nested reason and attempted work.
    """
    raise CensoredPoissonEnclosureError(
        failure,
        ledger.exact_work_count,
        f"censored-Poisson {kernel} enclosure failed: {error.failure.value}",
        nested_kernel=kernel,
        nested_failure=error.failure,
        nested_exact_work_count=error.exact_work_count,
        nested_attempted_exact_work_count=(error.attempted_exact_work_count),
        prior_exp_transcripts=tuple(ledger.exp_transcripts),
        prior_log_transcripts=tuple(ledger.log_transcripts),
    ) from error


def _exp_negative_point(
    mean: Fraction, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose ``exp(-mean)`` with the bound nested policy.

    Parameters
    ----------
    mean : Fraction
        Checked nonnegative exact mean point.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational enclosure of ``exp(-mean)``.

    Raises
    ------
    CensoredPoissonEnclosureError
        If the nested exponential enclosure fails.
    """
    policy = ledger.policy.exp
    try:
        enclosure, transcript = enclose_real_exp(
            (-mean, -mean),
            precision_bits=policy.precision_bits,
            maximum_terms=policy.maximum_terms,
            maximum_work=policy.maximum_work,
            maximum_range_reductions=policy.maximum_range_reductions,
            maximum_rational_bits=ledger.policy.maximum_rational_bits,
        )
    except EntireEnclosureError as error:
        _raise_entire_failure(
            ledger,
            error,
            kernel="exp",
            failure=(
                CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
            ),
        )
    ledger.exp_transcripts.append(transcript)
    return enclosure  # noqa: RET504


def _log_interval(
    probability: RationalInterval, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose log on one strictly positive probability interval.

    Parameters
    ----------
    probability : RationalInterval
        Checked probability interval with a positive lower endpoint.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational natural-logarithm enclosure.

    Raises
    ------
    CensoredPoissonEnclosureError
        If the nested logarithm enclosure fails or has no policy.
    """
    policy = ledger.policy.log
    if policy is None:
        raise CensoredPoissonEnclosureError(
            CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE,
            ledger.exact_work_count,
            "censored-Poisson NLL has no logarithm policy",
            prior_exp_transcripts=tuple(ledger.exp_transcripts),
            prior_log_transcripts=tuple(ledger.log_transcripts),
        )
    try:
        enclosure, transcript = enclose_real_log(
            probability,
            precision_bits=policy.precision_bits,
            maximum_terms=policy.maximum_terms,
            maximum_work=policy.maximum_work,
            maximum_range_reductions=policy.maximum_range_reductions,
            maximum_rational_bits=ledger.policy.maximum_rational_bits,
        )
    except EntireEnclosureError as error:
        _raise_entire_failure(
            ledger,
            error,
            kernel="log",
            failure=(
                CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE
            ),
        )
    ledger.log_transcripts.append(transcript)
    return enclosure


def _poisson_factor(
    mean: Fraction, order: int, ledger: _WorkLedger
) -> Fraction:
    """PRIVATE: Form exactly ``mean**order / order!`` by recurrence.

    Parameters
    ----------
    mean : Fraction
        Checked nonnegative exact mean point.
    order : int
        Checked nonnegative polynomial order.
    ledger : _WorkLedger
        Active local resource ledger.

    Returns
    -------
    factor : Fraction
        Exact Poisson polynomial factor.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local work or rational-size limits fail.
    """
    factor = _ONE
    ledger.polynomial_terms += 1
    for degree in range(1, order + 1):
        denominator = ledger.retain(Fraction(degree))
        factor = ledger.divide(ledger.multiply(factor, mean), denominator)
        ledger.polynomial_terms += 1
    return factor


def _prefix_polynomials(
    mean: Fraction, count_ceiling: int, ledger: _WorkLedger
) -> Tuple[Fraction, Fraction, Fraction]:
    r"""PRIVATE: Form CDF, censored-deficit, and first-tail factors.

    Parameters
    ----------
    mean : Fraction
        Checked nonnegative exact mean point.
    count_ceiling : int
        Positive checked censoring ceiling.
    ledger : _WorkLedger
        Active local resource ledger.

    Returns
    -------
    cdf_factor : Fraction
        Exact ``sum(mean**r/r!, r=0,...,count_ceiling-1)``.
    deficit_factor : Fraction
        Exact ``sum((count_ceiling-r)*mean**r/r!, r=0,...,c-1)``.
    first_tail_factor : Fraction
        Exact ``mean**count_ceiling/count_ceiling!``.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local work or rational-size limits fail.
    """
    term = _ONE
    cdf_factor = _ONE
    deficit_factor = ledger.retain(Fraction(count_ceiling))
    ledger.polynomial_terms += 1
    for order in range(1, count_ceiling + 1):
        denominator = ledger.retain(Fraction(order))
        term = ledger.divide(ledger.multiply(term, mean), denominator)
        ledger.polynomial_terms += 1
        if order < count_ceiling:
            cdf_factor = ledger.add(cdf_factor, term)
            weight = ledger.retain(Fraction(count_ceiling - order))
            weighted = ledger.multiply(weight, term)
            deficit_factor = ledger.add(deficit_factor, weighted)
    first_tail_factor = term
    result: Tuple[Fraction, Fraction, Fraction] = (
        cdf_factor,
        deficit_factor,
        first_tail_factor,
    )
    return result


def _checked_unit_interval(
    lower: Fraction, upper: Fraction, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Intersect one proved probability enclosure with ``[0, 1]``.

    Parameters
    ----------
    lower : Fraction
        Proved lower endpoint.
    upper : Fraction
        Proved upper endpoint.
    ledger : _WorkLedger
        Active local work ledger.

    Returns
    -------
    enclosure : RationalInterval
        Nonempty intersection with the probability range.

    Raises
    ------
    CensoredPoissonEnclosureError
        If the proved enclosure has an empty probability intersection.
    """
    checked_lower = max(_ZERO, lower)
    checked_upper = min(_ONE, upper)
    if checked_lower > checked_upper:
        ledger.fail(
            CensoredPoissonEnclosureFailure.ENCLOSURE_INTERSECTION_FAILURE,
            "censored-Poisson probability intersection is empty",
        )
    enclosure: RationalInterval = (checked_lower, checked_upper)
    return enclosure


def _pmf_point(
    mean: Fraction, observed_count: int, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose one uncensored Poisson mass at a mean point.

    Parameters
    ----------
    mean : Fraction
        Checked nonnegative exact mean point.
    observed_count : int
        Checked unsaturated observed count.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational probability enclosure.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local or nested resource bounds fail.
    """
    if mean == _ZERO:
        probability = _ONE if observed_count == 0 else _ZERO
        enclosure: RationalInterval = (probability, probability)
        return enclosure  # noqa: RET504
    exponential = _exp_negative_point(mean, ledger)
    factor = _poisson_factor(mean, observed_count, ledger)
    enclosure = _checked_unit_interval(
        ledger.multiply(exponential[0], factor),
        ledger.multiply(exponential[1], factor),
        ledger,
    )
    return enclosure  # noqa: RET504


def _tail_point(
    mean: Fraction, count_ceiling: int, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose one censored upper-tail probability at a point.

    Parameters
    ----------
    mean : Fraction
        Checked nonnegative exact mean point.
    count_ceiling : int
        Checked censoring ceiling.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational upper-tail probability enclosure.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local or nested resource bounds fail.
    """
    if count_ceiling == 0:
        enclosure: RationalInterval = (_ONE, _ONE)
        return enclosure
    if mean == _ZERO:
        enclosure = (_ZERO, _ZERO)
        return enclosure  # noqa: RET504
    exponential = _exp_negative_point(mean, ledger)
    cdf_factor, _, first_tail_factor = _prefix_polynomials(
        mean, count_ceiling, ledger
    )
    cdf_lower = ledger.multiply(exponential[0], cdf_factor)
    cdf_upper = ledger.multiply(exponential[1], cdf_factor)
    complement_lower = ledger.subtract(_ONE, cdf_upper)
    complement_upper = ledger.subtract(_ONE, cdf_lower)
    direct_lower = ledger.multiply(exponential[0], first_tail_factor)
    ledger.direct_tail_lower_evaluations += 1
    enclosure = _checked_unit_interval(
        max(complement_lower, direct_lower), complement_upper, ledger
    )
    return enclosure  # noqa: RET504


def _censored_mean_point(
    mean: Fraction, count_ceiling: int, ledger: _WorkLedger
) -> RationalInterval:
    """PRIVATE: Enclose one censored-Poisson expectation at a mean point.

    Parameters
    ----------
    mean : Fraction
        Checked nonnegative exact Poisson mean point.
    count_ceiling : int
        Checked censoring ceiling.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational enclosure of ``E[min(Y, count_ceiling)]``.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local or nested resource bounds fail or intersection is empty.
    """
    if count_ceiling == 0 or mean == _ZERO:
        enclosure: RationalInterval = (_ZERO, _ZERO)
        return enclosure
    exponential = _exp_negative_point(mean, ledger)
    _, deficit_factor, _ = _prefix_polynomials(mean, count_ceiling, ledger)
    ceiling = ledger.retain(Fraction(count_ceiling))
    raw_lower = ledger.subtract(
        ceiling, ledger.multiply(exponential[1], deficit_factor)
    )
    raw_upper = ledger.subtract(
        ceiling, ledger.multiply(exponential[0], deficit_factor)
    )

    # min(Y, c) >= 1 on {Y >= 1}.  This independently proved positive
    # lower is intersected with the main deficit enclosure, never added.
    tail_complement_lower = ledger.subtract(_ONE, exponential[1])
    first_positive_mass_lower = ledger.multiply(exponential[0], mean)
    direct_lower = max(_ZERO, tail_complement_lower, first_positive_mass_lower)
    ledger.direct_tail_lower_evaluations += 1
    checked_lower = max(_ZERO, raw_lower, direct_lower)
    checked_upper = min(ceiling, raw_upper)
    if checked_lower > checked_upper:
        ledger.fail(
            CensoredPoissonEnclosureFailure.ENCLOSURE_INTERSECTION_FAILURE,
            "censored-Poisson mean intersection is empty",
        )
    enclosure = (checked_lower, checked_upper)
    return enclosure  # noqa: RET504


def _probability_interval(
    mean: RationalInterval,
    observed_count: int,
    count_ceiling: int,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Enclose a censored probability over one mean interval.

    Parameters
    ----------
    mean : RationalInterval
        Checked ordered nonnegative mean interval.
    observed_count : int
        Checked observed censored count.
    count_ceiling : int
        Checked censoring ceiling.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational censored-probability enclosure.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local or nested resource bounds fail.
    """
    if observed_count == count_ceiling:
        lower_bounds = _tail_point(mean[0], count_ceiling, ledger)
        ledger.endpoint_evaluations += 1
        if mean[0] == mean[1]:
            enclosure: RationalInterval = lower_bounds
            return enclosure
        upper_bounds = _tail_point(mean[1], count_ceiling, ledger)
        ledger.endpoint_evaluations += 1
        enclosure: RationalInterval = (
            lower_bounds[0],
            upper_bounds[1],
        )
        return enclosure

    lower_bounds = _pmf_point(mean[0], observed_count, ledger)
    ledger.endpoint_evaluations += 1
    if mean[0] == mean[1]:
        enclosure = lower_bounds
        return enclosure  # noqa: RET504
    upper_bounds = _pmf_point(mean[1], observed_count, ledger)
    ledger.endpoint_evaluations += 1
    enclosure_lower = min(lower_bounds[0], upper_bounds[0])
    enclosure_upper = max(lower_bounds[1], upper_bounds[1])
    critical = ledger.retain(Fraction(observed_count))
    if observed_count > 0 and mean[0] < critical < mean[1]:
        critical_bounds = _pmf_point(critical, observed_count, ledger)
        ledger.critical_point_evaluations += 1
        enclosure_upper = max(enclosure_upper, critical_bounds[1])
    enclosure = (enclosure_lower, enclosure_upper)
    return enclosure  # noqa: RET504


def _mean_interval(
    mean: RationalInterval,
    count_ceiling: int,
    ledger: _WorkLedger,
) -> RationalInterval:
    """PRIVATE: Enclose the monotone censored mean over a mean interval.

    Parameters
    ----------
    mean : RationalInterval
        Checked ordered nonnegative Poisson mean interval.
    count_ceiling : int
        Checked censoring ceiling.
    ledger : _WorkLedger
        Active complete resource ledger.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational censored-mean enclosure.

    Raises
    ------
    CensoredPoissonEnclosureError
        If local or nested resource bounds fail.
    """
    lower_bounds = _censored_mean_point(mean[0], count_ceiling, ledger)
    ledger.endpoint_evaluations += 1
    if mean[0] == mean[1]:
        enclosure: RationalInterval = lower_bounds
        return enclosure
    upper_bounds = _censored_mean_point(mean[1], count_ceiling, ledger)
    ledger.endpoint_evaluations += 1
    enclosure: RationalInterval = (lower_bounds[0], upper_bounds[1])
    return enclosure


def _checked_problem(
    mean: object,
    observed_count: object | None,
    count_ceiling: object,
    ledger: _WorkLedger,
) -> Tuple[RationalInterval, int | None, int]:
    """PRIVATE: Validate one mean, ceiling, and optional observation.

    Parameters
    ----------
    mean : object
        Submitted exact mean interval.
    observed_count : object | None
        Submitted censored count, or ``None`` for a mean enclosure.
    count_ceiling : object
        Submitted censoring ceiling.
    ledger : _WorkLedger
        Active local resource ledger.

    Returns
    -------
    checked_mean : RationalInterval
        Ordered nonnegative exact mean interval.
    checked_observed : int | None
        Validated censored count, when supplied.
    checked_ceiling : int
        Validated censoring ceiling.

    Raises
    ------
    TypeError
        If count types or the mean type are invalid.
    ValueError
        If counts or mean endpoints are outside their domain.
    CensoredPoissonEnclosureError
        If the ceiling or rational-size limit is exceeded.
    """
    checked_ceiling = _checked_count(count_ceiling, "count_ceiling")
    if checked_ceiling > ledger.policy.maximum_count_ceiling:
        ledger.fail(
            CensoredPoissonEnclosureFailure.COUNT_CEILING_LIMIT,
            "censored-Poisson count ceiling exceeds its policy cap",
        )
    ledger.retain(Fraction(checked_ceiling))
    if observed_count is None:
        checked_observed: int | None = None
    else:
        checked_observed = _checked_count(observed_count, "observed_count")
        if checked_observed > checked_ceiling:
            raise ValueError("observed_count must not exceed count_ceiling")
        ledger.retain(Fraction(checked_observed))
    checked_mean = _checked_mean_interval(mean, ledger)
    result: Tuple[RationalInterval, int | None, int] = (
        checked_mean,
        checked_observed,
        checked_ceiling,
    )
    return result


def enclose_censored_poisson_probability(
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
) -> Tuple[RationalInterval, CensoredPoissonWorkTranscript]:
    """Enclose one probability in a censored Poisson law.

    Parameters
    ----------
    mean : RationalInterval
        Ordered nonnegative exact interval for the uncensored Poisson mean.
    observed_count : int
        Observed censored count in ``0,...,count_ceiling``.  Equality with
        the ceiling denotes the whole upper tail.
    count_ceiling : int
        Nonnegative censoring ceiling.
    maximum_count_ceiling : int, optional
        Maximum accepted censoring ceiling; default is 4096.
    maximum_work : int, optional
        Maximum local exact polynomial operations; default is 1,000,000.
    maximum_rational_bits : int, optional
        Shared maximum retained rational bits; default is 262,144.
    exp_precision_bits : int, optional
        Nested exponential seed precision; default is 160.
    maximum_exp_terms : int, optional
        Nested exponential per-series term budget; default is 4096.
    maximum_exp_work : int, optional
        Nested exponential exact-work budget; default is 1,000,000.
    maximum_exp_range_reductions : int, optional
        Nested exponential reduction limit; default is 4096.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational enclosure of the requested censored probability.
    transcript : CensoredPoissonWorkTranscript
        Deterministic local and nested resource transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If the mean, counts, or resource ranges are invalid.
    CensoredPoissonEnclosureError
        If a bounded exact enclosure cannot be completed.
    """
    policy = _checked_policy(
        maximum_count_ceiling=maximum_count_ceiling,
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        exp_precision_bits=exp_precision_bits,
        maximum_exp_terms=maximum_exp_terms,
        maximum_exp_work=maximum_exp_work,
        maximum_exp_range_reductions=maximum_exp_range_reductions,
        log_precision_bits=None,
        maximum_log_terms=None,
        maximum_log_work=None,
        maximum_log_range_reductions=None,
    )
    ledger = _WorkLedger(
        "exact_fraction_censored_poisson_probability_v1",
        policy,
        count_ceiling=0,
        observed_count=None,
    )
    checked_mean, checked_observed, checked_ceiling = _checked_problem(
        mean, observed_count, count_ceiling, ledger
    )
    checked_observed = cast(int, checked_observed)
    ledger.count_ceiling = checked_ceiling
    ledger.observed_count = checked_observed
    enclosure = _probability_interval(
        checked_mean, checked_observed, checked_ceiling, ledger
    )
    result: Tuple[RationalInterval, CensoredPoissonWorkTranscript] = (
        enclosure,
        ledger.transcript(),
    )
    return result


def enclose_censored_poisson_mean(
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
) -> Tuple[RationalInterval, CensoredPoissonWorkTranscript]:
    """Enclose the expectation of one censored Poisson count.

    Parameters
    ----------
    mean : RationalInterval
        Ordered nonnegative exact interval for the uncensored Poisson mean.
    count_ceiling : int
        Nonnegative censoring ceiling.
    maximum_count_ceiling : int, optional
        Maximum accepted censoring ceiling; default is 4096.
    maximum_work : int, optional
        Maximum local exact polynomial operations; default is 1,000,000.
    maximum_rational_bits : int, optional
        Shared maximum retained rational bits; default is 262,144.
    exp_precision_bits : int, optional
        Nested exponential seed precision; default is 160.
    maximum_exp_terms : int, optional
        Nested exponential per-series term budget; default is 4096.
    maximum_exp_work : int, optional
        Nested exponential exact-work budget; default is 1,000,000.
    maximum_exp_range_reductions : int, optional
        Nested exponential reduction limit; default is 4096.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational enclosure of ``E[min(Y, count_ceiling)]``.
    transcript : CensoredPoissonWorkTranscript
        Deterministic local and nested resource transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If the mean, ceiling, or resource ranges are invalid.
    CensoredPoissonEnclosureError
        If a bounded exact enclosure cannot be completed.
    """
    policy = _checked_policy(
        maximum_count_ceiling=maximum_count_ceiling,
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        exp_precision_bits=exp_precision_bits,
        maximum_exp_terms=maximum_exp_terms,
        maximum_exp_work=maximum_exp_work,
        maximum_exp_range_reductions=maximum_exp_range_reductions,
        log_precision_bits=None,
        maximum_log_terms=None,
        maximum_log_work=None,
        maximum_log_range_reductions=None,
    )
    ledger = _WorkLedger(
        "exact_fraction_censored_poisson_mean_v1",
        policy,
        count_ceiling=0,
        observed_count=None,
    )
    checked_mean, _, checked_ceiling = _checked_problem(
        mean, None, count_ceiling, ledger
    )
    ledger.count_ceiling = checked_ceiling
    enclosure = _mean_interval(checked_mean, checked_ceiling, ledger)
    result: Tuple[RationalInterval, CensoredPoissonWorkTranscript] = (
        enclosure,
        ledger.transcript(),
    )
    return result


def enclose_censored_poisson_nll(  # noqa: PLR0913
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
    log_precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_log_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_log_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_log_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
) -> Tuple[RationalInterval, CensoredPoissonWorkTranscript]:
    """Enclose one censored Poisson negative log-likelihood.

    Parameters
    ----------
    mean : RationalInterval
        Ordered nonnegative exact interval for the uncensored Poisson mean.
    observed_count : int
        Observed censored count in ``0,...,count_ceiling``.
    count_ceiling : int
        Nonnegative censoring ceiling; equality with ``observed_count``
        denotes an upper-tail event.
    maximum_count_ceiling : int, optional
        Maximum accepted censoring ceiling; default is 4096.
    maximum_work : int, optional
        Maximum local exact polynomial operations; default is 1,000,000.
    maximum_rational_bits : int, optional
        Shared maximum retained rational bits; default is 262,144.
    exp_precision_bits : int, optional
        Nested exponential seed precision; default is 160.
    maximum_exp_terms : int, optional
        Nested exponential per-series term budget; default is 4096.
    maximum_exp_work : int, optional
        Nested exponential exact-work budget; default is 1,000,000.
    maximum_exp_range_reductions : int, optional
        Nested exponential reduction limit; default is 4096.
    log_precision_bits : int, optional
        Nested logarithm atanh-series precision; default is 160.
    maximum_log_terms : int, optional
        Nested logarithm per-series term budget; default is 4096.
    maximum_log_work : int, optional
        Nested logarithm exact-work budget; default is 1,000,000.
    maximum_log_range_reductions : int, optional
        Nested logarithm reduction limit; default is 4096.

    Returns
    -------
    enclosure : RationalInterval
        Exact rational NLL enclosure with a nonnegative lower endpoint.
    transcript : CensoredPoissonWorkTranscript
        Deterministic local, exponential, and logarithm work transcript.

    Raises
    ------
    TypeError
        If input or resource types are invalid.
    ValueError
        If the mean, counts, or resource ranges are invalid.
    CensoredPoissonEnclosureError
        If probability positivity or a bounded nested enclosure fails.

    Notes
    -----
    This routine never installs an epsilon floor.  NLL eligibility requires
    the computed probability lower endpoint itself to be strictly positive.
    """
    policy = _checked_policy(
        maximum_count_ceiling=maximum_count_ceiling,
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        exp_precision_bits=exp_precision_bits,
        maximum_exp_terms=maximum_exp_terms,
        maximum_exp_work=maximum_exp_work,
        maximum_exp_range_reductions=maximum_exp_range_reductions,
        log_precision_bits=log_precision_bits,
        maximum_log_terms=maximum_log_terms,
        maximum_log_work=maximum_log_work,
        maximum_log_range_reductions=maximum_log_range_reductions,
    )
    ledger = _WorkLedger(
        "exact_fraction_censored_poisson_nll_v1",
        policy,
        count_ceiling=0,
        observed_count=None,
    )
    checked_mean, checked_observed, checked_ceiling = _checked_problem(
        mean, observed_count, count_ceiling, ledger
    )
    checked_observed = cast(int, checked_observed)
    ledger.count_ceiling = checked_ceiling
    ledger.observed_count = checked_observed
    probability = _probability_interval(
        checked_mean, checked_observed, checked_ceiling, ledger
    )
    if probability[0] <= _ZERO:
        ledger.fail(
            CensoredPoissonEnclosureFailure.NONPOSITIVE_PROBABILITY_LOWER,
            "censored-Poisson probability lower bound is not positive",
        )
    logarithm = _log_interval(probability, ledger)
    lower = max(_ZERO, -logarithm[1])
    upper = -logarithm[0]
    if lower > upper:
        ledger.fail(
            CensoredPoissonEnclosureFailure.ENCLOSURE_INTERSECTION_FAILURE,
            "censored-Poisson NLL intersection is empty",
        )
    enclosure: RationalInterval = (lower, upper)
    result: Tuple[RationalInterval, CensoredPoissonWorkTranscript] = (
        enclosure,
        ledger.transcript(),
    )
    return result


__all__: list[str] = [
    "CensoredPoissonEnclosureError",
    "CensoredPoissonEnclosureFailure",
    "CensoredPoissonWorkTranscript",
    "enclose_censored_poisson_mean",
    "enclose_censored_poisson_nll",
    "enclose_censored_poisson_probability",
]
