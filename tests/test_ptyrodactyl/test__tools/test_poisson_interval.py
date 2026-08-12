r"""Falsification tests for exact censored-Poisson host enclosures."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from beartype.typing import Tuple

from ptyrodactyl._tools.entire_interval import EntireEnclosureFailure
from ptyrodactyl._tools.poisson_interval import (
    CensoredPoissonEnclosureError,
    CensoredPoissonEnclosureFailure,
    enclose_censored_poisson_mean,
    enclose_censored_poisson_nll,
    enclose_censored_poisson_probability,
)

_KERNEL_PRECISION_BITS: int = 384
_ORACLE_PRECISION: int = 220
_WIDTH_TARGET: Decimal = Decimal("1e-100")


def _decimal(value: Fraction) -> Decimal:
    """Convert one exact rational under the active Decimal context."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def _poisson_factor(mean: Decimal, order: int) -> Decimal:
    """Form one independent high-precision Poisson polynomial factor."""
    term = Decimal(1)
    for degree in range(1, order + 1):
        term *= mean / Decimal(degree)
    return term


def _probability_oracle(
    mean: Fraction, observed_count: int, count_ceiling: int
) -> Decimal:
    """Evaluate one censored probability with 190-digit Decimal."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        point = _decimal(mean)
        exponential = (-point).exp()
        if observed_count < count_ceiling:
            result = exponential * _poisson_factor(point, observed_count)
        elif count_ceiling == 0:
            result = Decimal(1)
        else:
            prefix = sum(
                (
                    _poisson_factor(point, order)
                    for order in range(count_ceiling)
                ),
                start=Decimal(0),
            )
            result = Decimal(1) - exponential * prefix
        return +result


def _mean_oracle(mean: Fraction, count_ceiling: int) -> Decimal:
    """Evaluate one censored count mean with 190-digit Decimal."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        point = _decimal(mean)
        exponential = (-point).exp()
        deficit = sum(
            (
                Decimal(count_ceiling - order) * _poisson_factor(point, order)
                for order in range(count_ceiling)
            ),
            start=Decimal(0),
        )
        result = Decimal(count_ceiling) - exponential * deficit
        return +result


def _assert_contains(
    enclosure: Tuple[Fraction, Fraction], oracle: Decimal
) -> None:
    """Require one exact rational interval to contain a Decimal oracle."""
    exact_oracle = Fraction(oracle)
    assert enclosure[0] <= exact_oracle <= enclosure[1]


def _decimal_width(enclosure: Tuple[Fraction, Fraction]) -> Decimal:
    """Return one rational enclosure width under high precision."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        return +_decimal(enclosure[1] - enclosure[0])


def test_symbolic_zero_edges_exact_replay_and_frozen_transcripts() -> None:
    """Pin lambda-zero domains, c=0 semantics, and deterministic replay."""
    zero = Fraction(0)
    one = Fraction(1)
    probability, probability_work = enclose_censored_poisson_probability(
        (zero, zero), 0, 0
    )
    mean, mean_work = enclose_censored_poisson_mean((zero, zero), 0)
    nll, nll_work = enclose_censored_poisson_nll((zero, zero), 0, 0)
    assert probability == (one, one)
    assert mean == nll == (zero, zero)
    assert probability_work.exact_work_count == mean_work.exact_work_count == 0
    assert probability_work.exp_transcripts == ()
    assert mean_work.log_precision_bits == 0
    assert len(nll_work.log_transcripts) == 1

    for observed_count in range(4):
        enclosure, _ = enclose_censored_poisson_probability(
            (zero, zero), observed_count, 3
        )
        expected = one if observed_count == 0 else zero
        assert enclosure == (expected, expected)
    assert enclose_censored_poisson_mean((zero, zero), 3)[0] == (zero, zero)

    call = ((Fraction(7, 3), Fraction(7, 3)), 2, 5)
    first = enclose_censored_poisson_probability(*call, exp_precision_bits=256)
    second = enclose_censored_poisson_probability(
        *call, exp_precision_bits=256
    )
    assert first == second
    assert first[1].algorithm == (
        "exact_fraction_censored_poisson_probability_v1"
    )
    with pytest.raises(FrozenInstanceError):
        first[1].exact_work_count = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mean", "observed_count", "count_ceiling"),
    (
        (Fraction(1, 7), 0, 5),
        (Fraction(7, 3), 2, 5),
        (Fraction(7, 3), 5, 5),
        (Fraction(25, 2), 3, 20),
        (Fraction(40), 12, 12),
    ),
)
def test_probabilities_contain_190_digit_point_oracles(
    mean: Fraction, observed_count: int, count_ceiling: int
) -> None:
    """Contain independent 190-digit PMF and upper-tail point oracles."""
    enclosure, transcript = enclose_censored_poisson_probability(
        (mean, mean),
        observed_count,
        count_ceiling,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
    )
    _assert_contains(
        enclosure, _probability_oracle(mean, observed_count, count_ceiling)
    )
    assert _decimal_width(enclosure) < _WIDTH_TARGET
    assert transcript.endpoint_evaluations == 1
    assert transcript.exp_precision_bits == _KERNEL_PRECISION_BITS


def test_normalization_critical_extremum_and_monotone_tail() -> None:
    """Enclose normalization, the PMF critical point, and tail monotonicity."""
    point = Fraction(7, 3)
    count_ceiling = 8
    probabilities = [
        enclose_censored_poisson_probability(
            (point, point),
            observed_count,
            count_ceiling,
            exp_precision_bits=384,
        )[0]
        for observed_count in range(count_ceiling + 1)
    ]
    assert sum(bounds[0] for bounds in probabilities) <= 1
    assert sum(bounds[1] for bounds in probabilities) >= 1

    interval = (Fraction(1, 2), Fraction(11, 2))
    enclosure, transcript = enclose_censored_poisson_probability(
        interval, 3, count_ceiling, exp_precision_bits=384
    )
    for sample in (interval[0], Fraction(3), interval[1]):
        _assert_contains(
            enclosure, _probability_oracle(sample, 3, count_ceiling)
        )
    assert transcript.endpoint_evaluations == 2
    assert transcript.critical_point_evaluations == 1

    small_tail = enclose_censored_poisson_probability(
        (Fraction(1), Fraction(1)), 4, 4, exp_precision_bits=384
    )[0]
    large_tail = enclose_censored_poisson_probability(
        (Fraction(3), Fraction(3)), 4, 4, exp_precision_bits=384
    )[0]
    assert small_tail[1] < large_tail[0]


def test_censored_mean_oracle_monotonicity_and_saturation_distinction() -> (
    None
):
    """Enclose the true censored mean rather than clipping its input mean."""
    point = Fraction(7, 3)
    count_ceiling = 5
    enclosure, transcript = enclose_censored_poisson_mean(
        (point, point),
        count_ceiling,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
    )
    _assert_contains(enclosure, _mean_oracle(point, count_ceiling))
    assert _decimal_width(enclosure) < _WIDTH_TARGET
    assert transcript.direct_tail_lower_evaluations == 1

    saturated, _ = enclose_censored_poisson_mean(
        (Fraction(10), Fraction(10)),
        2,
        exp_precision_bits=384,
    )
    _assert_contains(saturated, _mean_oracle(Fraction(10), 2))
    assert saturated[1] < Fraction(2)
    assert saturated != (Fraction(2), Fraction(2))

    interval, _ = enclose_censored_poisson_mean(
        (Fraction(1, 4), Fraction(4)), 5, exp_precision_bits=384
    )
    for sample in (Fraction(1, 4), Fraction(1), Fraction(4)):
        _assert_contains(interval, _mean_oracle(sample, 5))


def test_tiny_positive_means_retain_direct_tail_and_mean_lowers() -> None:
    """Avoid zero lower collapse when complement subtraction is too wide."""
    tiny = Fraction(1, 10**120)
    tail, tail_work = enclose_censored_poisson_probability((tiny, tiny), 5, 5)
    mean, mean_work = enclose_censored_poisson_mean((tiny, tiny), 5)
    assert tail[0] > 0
    assert mean[0] > 0
    assert tail_work.direct_tail_lower_evaluations == 1
    assert mean_work.direct_tail_lower_evaluations == 1

    likelihood_tiny = Fraction(1, 10**20)
    nll, nll_work = enclose_censored_poisson_nll(
        (likelihood_tiny, likelihood_tiny),
        5,
        5,
        log_precision_bits=192,
        maximum_rational_bits=1_048_576,
    )
    assert nll[0] > 0
    assert nll_work.direct_tail_lower_evaluations == 1


@pytest.mark.parametrize(
    ("mean", "observed_count", "count_ceiling"),
    (
        (Fraction(7, 3), 2, 5),
        (Fraction(7, 3), 5, 5),
        (Fraction(1, 7), 0, 5),
    ),
)
def test_nll_contains_190_digit_oracles_without_probability_floor(
    mean: Fraction, observed_count: int, count_ceiling: int
) -> None:
    """Contain independent 190-digit NLL oracles without epsilon clipping."""
    enclosure, transcript = enclose_censored_poisson_nll(
        (mean, mean),
        observed_count,
        count_ceiling,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
        log_precision_bits=_KERNEL_PRECISION_BITS,
        maximum_rational_bits=1_048_576,
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        oracle = -_probability_oracle(mean, observed_count, count_ceiling).ln()
    _assert_contains(enclosure, oracle)
    assert _decimal_width(enclosure) < _WIDTH_TARGET
    assert len(transcript.exp_transcripts) == 1
    assert len(transcript.log_transcripts) == 1


def test_nll_positivity_gate_and_tiny_positive_probability() -> None:
    """Reject an exact zero lower but accept a tiny proved positive mass."""
    with pytest.raises(CensoredPoissonEnclosureError) as zero_error:
        enclose_censored_poisson_nll((Fraction(0), Fraction(1)), 1, 5)
    assert zero_error.value.failure is (
        CensoredPoissonEnclosureFailure.NONPOSITIVE_PROBABILITY_LOWER
    )
    assert zero_error.value.nested_failure is None

    tiny = Fraction(1, 10**80)
    enclosure, _ = enclose_censored_poisson_nll(
        (tiny, tiny), 1, 5, log_precision_bits=256
    )
    assert enclosure[0] > 0


def test_domains_exact_python_ints_and_count_ceiling_policy() -> None:
    """Reject malformed means, nonexact ints, and count-domain violations."""

    class IntSubclass(int):
        """Provide one nonexact Python-int submission."""

    invalid_means = (
        [Fraction(0), Fraction(1)],
        (0, Fraction(1)),
        (Fraction(2), Fraction(1)),
        (Fraction(-1), Fraction(1)),
    )
    for invalid in invalid_means:
        error_type = (
            TypeError
            if isinstance(invalid, list) or invalid[0] == 0
            else ValueError
        )
        with pytest.raises(error_type):
            enclose_censored_poisson_mean(invalid, 2)  # type: ignore[arg-type]

    for invalid_count in (True, IntSubclass(2)):
        with pytest.raises(TypeError):
            enclose_censored_poisson_mean(
                (Fraction(1), Fraction(1)),
                invalid_count,
            )
    with pytest.raises(TypeError):
        enclose_censored_poisson_mean(
            (Fraction(1), Fraction(1)),
            2,
            maximum_work=True,
        )
    with pytest.raises(ValueError):
        enclose_censored_poisson_probability((Fraction(1), Fraction(1)), 3, 2)
    with pytest.raises(CensoredPoissonEnclosureError) as ceiling_error:
        enclose_censored_poisson_mean(
            (Fraction(1), Fraction(1)), 3, maximum_count_ceiling=2
        )
    assert ceiling_error.value.failure is (
        CensoredPoissonEnclosureFailure.COUNT_CEILING_LIMIT
    )


def test_local_and_nested_resource_failures_preserve_typed_provenance() -> (
    None
):
    """Fail each independent budget and preserve nested reason and work."""
    point = (Fraction(10), Fraction(10))
    with pytest.raises(CensoredPoissonEnclosureError) as work_error:
        enclose_censored_poisson_probability(
            (Fraction(1), Fraction(1)), 0, 2, maximum_work=1
        )
    assert work_error.value.failure is (
        CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert work_error.value.exact_work_count == 2

    with pytest.raises(CensoredPoissonEnclosureError) as exp_error:
        enclose_censored_poisson_probability(point, 0, 2, maximum_exp_terms=1)
    assert exp_error.value.failure is (
        CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
    )
    assert exp_error.value.nested_kernel == "exp"
    assert exp_error.value.nested_failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )
    assert exp_error.value.nested_exact_work_count is not None
    assert exp_error.value.nested_exact_work_count > 0

    with pytest.raises(CensoredPoissonEnclosureError) as exp_work_error:
        enclose_censored_poisson_probability(point, 0, 2, maximum_exp_work=1)
    assert exp_work_error.value.failure is (
        CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
    )
    assert exp_work_error.value.nested_failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert exp_work_error.value.nested_exact_work_count == 2

    with pytest.raises(CensoredPoissonEnclosureError) as range_error:
        enclose_censored_poisson_probability(
            (Fraction(2), Fraction(2)),
            0,
            2,
            maximum_exp_range_reductions=0,
        )
    assert range_error.value.nested_failure is (
        EntireEnclosureFailure.RANGE_REDUCTION_LIMIT
    )

    with pytest.raises(CensoredPoissonEnclosureError) as log_error:
        enclose_censored_poisson_nll(
            (Fraction(2), Fraction(2)),
            1,
            3,
            maximum_log_terms=1,
        )
    assert log_error.value.failure is (
        CensoredPoissonEnclosureFailure.LOGARITHM_ENCLOSURE_FAILURE
    )
    assert log_error.value.nested_kernel == "log"
    assert log_error.value.nested_failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )
    assert log_error.value.nested_exact_work_count is not None
    assert log_error.value.nested_exact_work_count > 0

    oversized = Fraction(1 << 30)
    with pytest.raises(CensoredPoissonEnclosureError) as rational_error:
        enclose_censored_poisson_mean(
            (oversized, oversized),
            2,
            maximum_rational_bits=20,
            exp_precision_bits=8,
        )
    assert rational_error.value.failure is (
        CensoredPoissonEnclosureFailure.RATIONAL_SIZE_LIMIT
    )

    with pytest.raises(CensoredPoissonEnclosureError) as constant_error:
        enclose_censored_poisson_mean(
            (Fraction(0), Fraction(0)),
            1 << 20,
            maximum_count_ceiling=1 << 20,
            maximum_rational_bits=8,
            exp_precision_bits=7,
        )
    assert constant_error.value.failure is (
        CensoredPoissonEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
