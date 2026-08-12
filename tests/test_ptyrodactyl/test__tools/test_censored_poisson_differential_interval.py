r"""Falsify exact censored-Poisson differential and Fisher enclosures."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from beartype.typing import Tuple

from ptyrodactyl._tools.censored_poisson_differential_interval import (
    CensoredPoissonDifferentialError,
    CensoredPoissonDifferentialFailure,
    enclose_censored_poisson_fisher_information,
    enclose_censored_poisson_nll_differential,
)
from ptyrodactyl._tools.entire_interval import EntireEnclosureFailure
from ptyrodactyl._tools.poisson_interval import (
    CensoredPoissonEnclosureFailure,
    enclose_censored_poisson_probability,
)

_KERNEL_PRECISION_BITS: int = 512
_ORACLE_PRECISION: int = 220
_WIDTH_TARGET: Decimal = Decimal("1e-140")


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
    mean: Decimal,
    observed_count: int,
    count_ceiling: int,
) -> Decimal:
    """Evaluate one censored probability with 200-plus-digit Decimal."""
    if count_ceiling == 0:
        return Decimal(1)
    exponential = (-mean).exp()
    if observed_count < count_ceiling:
        return exponential * _poisson_factor(mean, observed_count)
    prefix = sum(
        (_poisson_factor(mean, order) for order in range(count_ceiling)),
        start=Decimal(0),
    )
    return Decimal(1) - exponential * prefix


def _differential_oracle(
    mean: Decimal,
    observed_count: int,
    count_ceiling: int,
) -> Tuple[Decimal, Decimal]:
    """Evaluate one score and curvature with 200-plus-digit Decimal."""
    if count_ceiling == 0:
        return Decimal(0), Decimal(0)
    if observed_count < count_ceiling:
        return (
            Decimal(1) - Decimal(observed_count) / mean,
            Decimal(observed_count) / (mean * mean),
        )
    tail = _probability_oracle(mean, count_ceiling, count_ceiling)
    first = _probability_oracle(mean, count_ceiling - 1, count_ceiling)
    ratio = first / tail
    score = -ratio
    curvature = (
        ratio * ratio
        + (Decimal(1) - Decimal(count_ceiling - 1) / mean) * ratio
    )
    return score, curvature


def _fisher_oracle(mean: Decimal, count_ceiling: int) -> Decimal:
    """Evaluate the complete censored expected information in Decimal."""
    if count_ceiling == 0:
        return Decimal(0)
    total = Decimal(0)
    for observed_count in range(count_ceiling + 1):
        probability = _probability_oracle(mean, observed_count, count_ceiling)
        score, _ = _differential_oracle(mean, observed_count, count_ceiling)
        total += probability * score * score
    return total


def _nll_oracle(
    mean: Decimal,
    observed_count: int,
    count_ceiling: int,
) -> Decimal:
    """Evaluate one independent censored NLL in Decimal."""
    probability = _probability_oracle(mean, observed_count, count_ceiling)
    return -probability.ln()


def _assert_contains(
    enclosure: Tuple[Fraction, Fraction],
    oracle: Decimal,
) -> None:
    """Require one exact interval to contain a high-precision oracle."""
    exact_oracle = Fraction(oracle)
    assert enclosure[0] <= exact_oracle <= enclosure[1]


def _decimal_width(enclosure: Tuple[Fraction, Fraction]) -> Decimal:
    """Return one exact interval width under high Decimal precision."""
    return _decimal(enclosure[1] - enclosure[0])


def test_nll_differential_contains_220_digit_oracles_and_identities() -> None:
    """Contain interval, saturated, and finite-difference checks."""
    unsaturated = enclose_censored_poisson_nll_differential(
        (Fraction(2, 3), Fraction(5, 2)),
        2,
        5,
    )
    for sample in (Fraction(2, 3), Fraction(4, 3), Fraction(5, 2)):
        with localcontext() as context:
            context.prec = _ORACLE_PRECISION
            score, curvature = _differential_oracle(_decimal(sample), 2, 5)
        _assert_contains(unsaturated[0], score)
        _assert_contains(unsaturated[1], curvature)
    assert unsaturated[2].planned_local_work_count == 6
    assert unsaturated[2].exact_work_count == 6
    assert unsaturated[2].poisson_transcripts == ()

    point = Fraction(7, 3)
    saturated = enclose_censored_poisson_nll_differential(
        (point, point),
        4,
        4,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        decimal_point = _decimal(point)
        score, curvature = _differential_oracle(decimal_point, 4, 4)
        step = Decimal("1e-45")
        center = _nll_oracle(decimal_point, 4, 4)
        left = _nll_oracle(decimal_point - step, 4, 4)
        right = _nll_oracle(decimal_point + step, 4, 4)
        finite_score = (right - left) / (Decimal(2) * step)
        finite_curvature = (right - Decimal(2) * center + left) / (step * step)
        assert abs(finite_score - score) < Decimal("1e-85")
        assert abs(finite_curvature - curvature) < Decimal("1e-80")
    _assert_contains(saturated[0], score)
    _assert_contains(saturated[1], curvature)
    assert _decimal_width(saturated[0]) < _WIDTH_TARGET
    assert _decimal_width(saturated[1]) < _WIDTH_TARGET
    assert saturated[2].planned_local_work_count == 14
    assert saturated[2].exact_work_count == 14
    assert saturated[2].probability_observed_counts == (4, 3)
    assert tuple(
        item.observed_count for item in saturated[2].poisson_transcripts
    ) == (4, 3)

    saturated_interval = (Fraction(7, 3), Fraction(5, 2))
    saturated_wide = enclose_censored_poisson_nll_differential(
        saturated_interval,
        4,
        4,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
    )
    for sample in (
        saturated_interval[0],
        sum(saturated_interval, start=Fraction(0)) / 2,
        saturated_interval[1],
    ):
        with localcontext() as context:
            context.prec = _ORACLE_PRECISION
            score, curvature = _differential_oracle(_decimal(sample), 4, 4)
        _assert_contains(saturated_wide[0], score)
        _assert_contains(saturated_wide[1], curvature)
    assert saturated_wide[0][1] < 0
    assert saturated_wide[1][0] > 0


def test_symbolic_zero_edges_and_deterministic_frozen_replay() -> None:
    """Pin c=0, z=0, lambda=0, work, and immutable replay semantics."""
    wide_zero = (Fraction(0), Fraction(10))
    degenerate = enclose_censored_poisson_nll_differential(wide_zero, 0, 0)
    ordinary_zero = enclose_censored_poisson_nll_differential(wide_zero, 0, 5)
    information = enclose_censored_poisson_fisher_information(wide_zero, 0)
    assert degenerate[:2] == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert ordinary_zero[:2] == (
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    assert information[0] == (Fraction(0), Fraction(0))
    for transcript in (degenerate[2], ordinary_zero[2], information[1]):
        assert transcript.planned_local_work_count == 0
        assert transcript.exact_work_count == 0
        assert transcript.poisson_transcripts == ()

    call = (
        (Fraction(7, 3), Fraction(7, 3)),
        3,
        3,
    )
    first = enclose_censored_poisson_nll_differential(
        *call, exp_precision_bits=384
    )
    second = enclose_censored_poisson_nll_differential(
        *call, exp_precision_bits=384
    )
    assert first == second
    assert first[2].algorithm == (
        "exact_fraction_censored_poisson_nll_differential_v1"
    )
    with pytest.raises(FrozenInstanceError):
        first[2].exact_work_count = 0  # type: ignore[misc]


def test_fisher_contains_oracles_normalization_and_censoring_loss() -> None:
    """Enclose exact Fisher sums and distinguish them from one over lambda."""
    point = Fraction(2)
    information, transcript = enclose_censored_poisson_fisher_information(
        (point, point),
        2,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        oracle = _fisher_oracle(_decimal(point), 2)
    _assert_contains(information, oracle)
    assert _decimal_width(information) < _WIDTH_TARGET
    assert information[1] < Fraction(1, 2)
    assert transcript.planned_local_work_count == 16
    assert transcript.exact_work_count == 16
    assert transcript.probability_observed_counts == (0, 1, 2)
    assert transcript.symbol_count == 3

    interval = (Fraction(1, 2), Fraction(3))
    broad, _ = enclose_censored_poisson_fisher_information(
        interval,
        4,
        exp_precision_bits=384,
    )
    for sample in (interval[0], Fraction(3, 2), interval[1]):
        with localcontext() as context:
            context.prec = _ORACLE_PRECISION
            oracle = _fisher_oracle(_decimal(sample), 4)
        _assert_contains(broad, oracle)

    probabilities = [
        enclose_censored_poisson_probability(
            (point, point),
            observed_count,
            4,
            exp_precision_bits=_KERNEL_PRECISION_BITS,
        )[0]
        for observed_count in range(5)
    ]
    assert sum(bounds[0] for bounds in probabilities) <= 1
    assert sum(bounds[1] for bounds in probabilities) >= 1


def test_strict_domains_exact_python_counts_and_positive_mean_gate() -> None:
    """Reject malformed domains and never tolerance-classify mean zero."""

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
            enclose_censored_poisson_fisher_information(
                invalid,  # type: ignore[arg-type]
                2,
            )
    for invalid_count in (True, IntSubclass(2)):
        with pytest.raises(TypeError):
            enclose_censored_poisson_fisher_information(
                (Fraction(1), Fraction(1)),
                invalid_count,
            )
    with pytest.raises(TypeError):
        enclose_censored_poisson_fisher_information(
            (Fraction(1), Fraction(1)),
            2,
            maximum_work=True,
        )
    with pytest.raises(ValueError):
        enclose_censored_poisson_nll_differential(
            (Fraction(1), Fraction(1)), 3, 2
        )

    for call in (
        lambda: enclose_censored_poisson_nll_differential(
            (Fraction(0), Fraction(1)), 1, 4
        ),
        lambda: enclose_censored_poisson_nll_differential(
            (Fraction(0), Fraction(1)), 4, 4
        ),
        lambda: enclose_censored_poisson_fisher_information(
            (Fraction(0), Fraction(1)), 4
        ),
    ):
        with pytest.raises(CensoredPoissonDifferentialError) as error:
            call()
        assert error.value.failure is (
            CensoredPoissonDifferentialFailure.NONPOSITIVE_MEAN_LOWER
        )
        assert error.value.poisson_transcripts == ()


def test_local_and_nested_resource_failures_preserve_full_provenance() -> None:
    """Pin local, count, exponential, work, and rational failure evidence."""
    with pytest.raises(CensoredPoissonDifferentialError) as work_error:
        enclose_censored_poisson_nll_differential(
            (Fraction(1), Fraction(2)),
            1,
            3,
            maximum_work=5,
        )
    assert work_error.value.failure is (
        CensoredPoissonDifferentialFailure.WORK_BUDGET_EXCEEDED
    )
    assert work_error.value.exact_work_count == 6
    assert work_error.value.planned_local_work_count == 6
    assert work_error.value.poisson_transcripts == ()

    with pytest.raises(CensoredPoissonDifferentialError) as count_error:
        enclose_censored_poisson_fisher_information(
            (Fraction(1), Fraction(1)),
            3,
            maximum_count_ceiling=2,
        )
    assert count_error.value.failure is (
        CensoredPoissonDifferentialFailure.POISSON_ENCLOSURE_FAILURE
    )
    assert count_error.value.nested_poisson_failure is (
        CensoredPoissonEnclosureFailure.COUNT_CEILING_LIMIT
    )
    assert count_error.value.nested_count_ceiling == 3

    oversized = Fraction(1 << 30)
    with pytest.raises(CensoredPoissonDifferentialError) as input_bits:
        enclose_censored_poisson_fisher_information(
            (oversized, oversized),
            1,
            maximum_rational_bits=20,
            exp_precision_bits=8,
        )
    assert input_bits.value.failure is (
        CensoredPoissonDifferentialFailure.RATIONAL_SIZE_LIMIT
    )

    with pytest.raises(CensoredPoissonDifferentialError) as policy_bits:
        enclose_censored_poisson_fisher_information(
            (Fraction(1), Fraction(1)),
            1,
            maximum_rational_bits=20,
            exp_precision_bits=20,
        )
    assert policy_bits.value.nested_poisson_failure is (
        CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
    )
    assert policy_bits.value.nested_entire_failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert policy_bits.value.nested_entire_exact_work_count == 0

    point = (Fraction(10), Fraction(10))
    with pytest.raises(CensoredPoissonDifferentialError) as exp_error:
        enclose_censored_poisson_nll_differential(
            point,
            3,
            3,
            maximum_exp_terms=1,
        )
    assert exp_error.value.nested_poisson_failure is (
        CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE
    )
    assert exp_error.value.nested_entire_failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )
    assert exp_error.value.nested_entire_exact_work_count is not None

    with pytest.raises(CensoredPoissonDifferentialError) as nested_work:
        enclose_censored_poisson_nll_differential(
            (Fraction(1), Fraction(1)),
            20,
            20,
            maximum_work=14,
        )
    assert nested_work.value.nested_poisson_failure is (
        CensoredPoissonEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert nested_work.value.nested_poisson_exact_work_count is not None
    assert nested_work.value.nested_poisson_exact_work_count > 14

    with pytest.raises(CensoredPoissonDifferentialError) as nested_bits:
        enclose_censored_poisson_nll_differential(
            (Fraction(1), Fraction(1)),
            200,
            200,
            maximum_rational_bits=1024,
            exp_precision_bits=64,
        )
    assert nested_bits.value.nested_poisson_failure is (
        CensoredPoissonEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert nested_bits.value.nested_poisson_exact_work_count is not None
    assert nested_bits.value.nested_poisson_exact_work_count > 0

    with pytest.raises(CensoredPoissonDifferentialError) as fisher_bits:
        enclose_censored_poisson_fisher_information(
            (Fraction(1), Fraction(1)),
            50,
            maximum_rational_bits=96,
            exp_precision_bits=32,
        )
    assert fisher_bits.value.failure is (
        CensoredPoissonDifferentialFailure.POISSON_ENCLOSURE_FAILURE
    )
    assert fisher_bits.value.nested_poisson_failure is (
        CensoredPoissonEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert fisher_bits.value.poisson_transcripts
    assert fisher_bits.value.nested_observed_count == len(
        fisher_bits.value.poisson_transcripts
    )
    assert tuple(
        transcript.observed_count
        for transcript in fisher_bits.value.poisson_transcripts
    ) == tuple(range(len(fisher_bits.value.poisson_transcripts)))


def test_c_one_saturated_formula_and_fisher_work_are_exact() -> None:
    """Exercise the c=1 branch without a spurious zero-over-mean operation."""
    point = Fraction(3, 2)
    score, curvature, differential_work = (
        enclose_censored_poisson_nll_differential(
            (point, point),
            1,
            1,
            exp_precision_bits=_KERNEL_PRECISION_BITS,
        )
    )
    information, fisher_work = enclose_censored_poisson_fisher_information(
        (point, point),
        1,
        exp_precision_bits=_KERNEL_PRECISION_BITS,
    )
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        expected_score, expected_curvature = _differential_oracle(
            _decimal(point), 1, 1
        )
        expected_information = _fisher_oracle(_decimal(point), 1)
    _assert_contains(score, expected_score)
    _assert_contains(curvature, expected_curvature)
    _assert_contains(information, expected_information)
    assert differential_work.exact_work_count == 6
    assert fisher_work.exact_work_count == 6
    assert fisher_work.probability_observed_counts == (0, 1)
