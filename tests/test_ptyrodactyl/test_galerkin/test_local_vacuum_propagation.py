r"""Falsification tests for strict local-vacuum root propagators."""

from __future__ import annotations

import dataclasses
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from beartype.typing import Tuple

import ptyrodactyl._tools.entire_interval as entire
import ptyrodactyl.galerkin.local_vacuum_propagation as local
from ptyrodactyl._tools.entire_interval import (
    EntireEnclosureError,
    EntireEnclosureFailure,
)
from ptyrodactyl.galerkin.local_vacuum_propagation import (
    classify_local_vacuum_root,
    enclose_local_vacuum_propagator,
    make_local_vacuum_zero_witness,
    prepare_local_vacuum_propagator,
    prepare_local_vacuum_root_certificate,
)
from ptyrodactyl.types.local_vacuum_propagation_types import (
    GalerkinLocalVacuumPropagationError,
    GalerkinLocalVacuumPropagationFailure,
    GalerkinLocalVacuumPropagator,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRootCertificate,
    GalerkinLocalVacuumRootClass,
    GalerkinLocalVacuumZeroWitnessRoute,
)

_ORACLE_PRECISION: int = 140
_ORACLE_TOLERANCE: Decimal = Decimal("1e-130")

type _Interval = Tuple[Fraction, Fraction]
type _IntervalMatrix = Tuple[_Interval, _Interval, _Interval, _Interval]
type _OracleScalar = Decimal | Fraction
type _OracleMatrix = Tuple[
    _OracleScalar,
    _OracleScalar,
    _OracleScalar,
    _OracleScalar,
]


def _decimal(value: Fraction) -> Decimal:
    """Convert one exact rational to a high-precision Decimal."""
    result: Decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return result


def _decimal_sin_cos(value: Decimal) -> Tuple[Decimal, Decimal]:
    """Evaluate sine and cosine with an independent 130-digit series."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        squared = value * value
        sine = value
        sine_term = value
        cosine = Decimal(1)
        cosine_term = Decimal(1)
        for order in range(1, 1000):
            sine_term *= -squared / Decimal((2 * order) * (2 * order + 1))
            cosine_term *= -squared / Decimal((2 * order - 1) * (2 * order))
            sine += sine_term
            cosine += cosine_term
            if (
                abs(sine_term) < _ORACLE_TOLERANCE
                and abs(cosine_term) < _ORACLE_TOLERANCE
            ):
                break
        result: Tuple[Decimal, Decimal] = (+sine, +cosine)
        return result


def _oracle_matrix(q: Fraction, distance: Fraction) -> _OracleMatrix:
    """Evaluate one exact-sign homogeneous propagator independently."""
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        distance_decimal = _decimal(distance)
        if q > 0:
            root = _decimal(q).sqrt()
            sine, cosine = _decimal_sin_cos(root * distance_decimal)
            result: _OracleMatrix = (
                cosine,
                sine / root,
                -(root * sine),
                cosine,
            )
        elif q < 0:
            root = _decimal(-q).sqrt()
            argument = root * distance_decimal
            positive = argument.exp()
            negative = (-argument).exp()
            sine = (positive - negative) / Decimal(2)
            cosine = (positive + negative) / Decimal(2)
            result = (
                +cosine,
                +(sine / root),
                +(root * sine),
                +cosine,
            )
        else:
            result = (Fraction(1), distance, Fraction(0), Fraction(1))
        return result


def _entries(propagator: GalerkinLocalVacuumPropagator) -> _IntervalMatrix:
    """Project one row-major carrier matrix to exact endpoint pairs."""
    first, second, third, fourth = propagator.entries
    result: _IntervalMatrix = (
        (first.lower, first.upper),
        (second.lower, second.upper),
        (third.lower, third.upper),
        (fourth.lower, fourth.upper),
    )
    return result


def _assert_contains(interval: _Interval, value: Decimal | Fraction) -> None:
    """Require one exact rational interval to contain an oracle value."""
    oracle = value if isinstance(value, Fraction) else Fraction(value)
    assert interval[0] <= oracle <= interval[1]


def _assert_matrix_contains(
    intervals: _IntervalMatrix,
    oracle: _OracleMatrix,
) -> None:
    """Require all four row-major intervals to contain their oracles."""
    for interval, value in zip(intervals, oracle, strict=True):
        _assert_contains(interval, value)


def _interval_add(left: _Interval, right: _Interval) -> _Interval:
    """Add two exact rational test-oracle intervals."""
    result: _Interval = (left[0] + right[0], left[1] + right[1])
    return result


def _interval_negate(interval: _Interval) -> _Interval:
    """Negate one exact rational test-oracle interval."""
    result: _Interval = (-interval[1], -interval[0])
    return result


def _interval_multiply(left: _Interval, right: _Interval) -> _Interval:
    """Multiply two exact rational test-oracle intervals."""
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    result: _Interval = (min(products), max(products))
    return result


def _matrix_multiply(
    left: _IntervalMatrix,
    right: _IntervalMatrix,
) -> _IntervalMatrix:
    """Multiply two row-major exact interval matrices independently."""
    result: _IntervalMatrix = (
        _interval_add(
            _interval_multiply(left[0], right[0]),
            _interval_multiply(left[1], right[2]),
        ),
        _interval_add(
            _interval_multiply(left[0], right[1]),
            _interval_multiply(left[1], right[3]),
        ),
        _interval_add(
            _interval_multiply(left[2], right[0]),
            _interval_multiply(left[3], right[2]),
        ),
        _interval_add(
            _interval_multiply(left[2], right[1]),
            _interval_multiply(left[3], right[3]),
        ),
    )
    return result


def _classified(q: Fraction) -> GalerkinLocalVacuumRootCertificate:
    """Classify one exact singleton q interval."""
    certificate: GalerkinLocalVacuumRootCertificate = (
        classify_local_vacuum_root((q, q))
    )
    return certificate


def test_strict_root_classification_zero_witness_and_near_zero() -> None:
    """Pin strict signs, conditional formal zero, and unresolved near zero.

    :see: :func:`ptyrodactyl.galerkin.classify_local_vacuum_root`
    :see: :func:`ptyrodactyl.galerkin.make_local_vacuum_zero_witness`
    """
    positive = _classified(Fraction(4))
    negative = _classified(Fraction(-9))
    missing_zero = _classified(Fraction(0))
    epsilon = Fraction(1, 10**100)
    crossing = classify_local_vacuum_root((-epsilon, epsilon))
    one_sided = classify_local_vacuum_root((Fraction(0), epsilon))

    assert positive.classification is GalerkinLocalVacuumRootClass.PROPAGATING
    assert negative.classification is GalerkinLocalVacuumRootClass.EVANESCENT
    assert missing_zero.classification is (
        GalerkinLocalVacuumRootClass.UNCLASSIFIED
    )
    assert crossing.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED
    assert one_sided.classification is (
        GalerkinLocalVacuumRootClass.UNCLASSIFIED
    )
    assert missing_zero.root_interval is None

    exact_witness = make_local_vacuum_zero_witness(
        (("1", Fraction(3, 7)),),
        (("1", Fraction(3, 7)),),
        route=(GalerkinLocalVacuumZeroWitnessRoute.EXACT_RATIONAL_DIFFERENCE),
    )
    symbolic_witness = make_local_vacuum_zero_witness(
        (("k0_squared", Fraction(2, 3)),),
        (("k0_squared", Fraction(2, 3)),),
        route=(
            GalerkinLocalVacuumZeroWitnessRoute.SYMBOLIC_NORMAL_FORM_DIFFERENCE
        ),
    )
    exact_grazing = classify_local_vacuum_root(
        (Fraction(0), Fraction(0)), zero_witness=exact_witness
    )
    symbolic_grazing = classify_local_vacuum_root(
        (-epsilon, epsilon), zero_witness=symbolic_witness
    )
    assert exact_grazing.classification is GalerkinLocalVacuumRootClass.GRAZING
    assert symbolic_grazing.classification is (
        GalerkinLocalVacuumRootClass.GRAZING
    )
    assert exact_grazing.root_interval is not None
    assert exact_grazing.root_interval.lower == 0
    assert exact_grazing.root_interval.upper == 0
    assert "not physical LVT.39 evidence" in exact_witness.trust_scope

    with pytest.raises(GalerkinLocalVacuumPropagationError) as unequal:
        make_local_vacuum_zero_witness(
            (("1", Fraction(1)),),
            (("1", Fraction(2)),),
            route=(
                GalerkinLocalVacuumZeroWitnessRoute.EXACT_RATIONAL_DIFFERENCE
            ),
        )
    assert unequal.value.failure is (
        GalerkinLocalVacuumPropagationFailure.ZERO_WITNESS_INCONSISTENT
    )
    with pytest.raises(GalerkinLocalVacuumPropagationError) as inconsistent:
        classify_local_vacuum_root(
            (Fraction(1), Fraction(2)), zero_witness=exact_witness
        )
    assert inconsistent.value.failure is (
        GalerkinLocalVacuumPropagationFailure.ZERO_WITNESS_INCONSISTENT
    )


def test_nonsquare_root_bounds_are_outward_and_deterministic() -> None:
    """Contain 140-digit nonsquare roots through reciprocal lower bounds."""
    positive = _classified(Fraction(2))
    negative = _classified(Fraction(-2))
    repeated = _classified(Fraction(2))
    with localcontext() as context:
        context.prec = _ORACLE_PRECISION
        oracle = Decimal(2).sqrt()
    for certificate in (positive, negative):
        assert certificate.root_interval is not None
        _assert_contains(
            (certificate.root_interval.lower, certificate.root_interval.upper),
            oracle,
        )
        assert certificate.work_transcript.root_enclosures == 2
        assert certificate.work_transcript.divisions == 2
        assert certificate.work_transcript.multiplications == 3
        assert certificate.work_transcript.exact_work_count == 7
    assert local._exact_value_payload(positive) == local._exact_value_payload(
        repeated
    )


def test_propagating_evanescent_and_grazing_high_precision_oracles() -> None:
    """Contain independent 140-digit LVT.41--LVT.43 matrix oracles.

    :see: :func:`ptyrodactyl.galerkin.enclose_local_vacuum_propagator`
    """
    distance = Fraction(3, 7)
    propagating = enclose_local_vacuum_propagator(
        _classified(Fraction(2)), distance, precision_bits=420
    )
    evanescent = enclose_local_vacuum_propagator(
        _classified(Fraction(-9)), distance, precision_bits=224
    )
    witness = make_local_vacuum_zero_witness(
        (("q_identity", Fraction(1)),),
        (("q_identity", Fraction(1)),),
        route=(
            GalerkinLocalVacuumZeroWitnessRoute.SYMBOLIC_NORMAL_FORM_DIFFERENCE
        ),
    )
    grazing_root = classify_local_vacuum_root(
        (Fraction(0), Fraction(0)), zero_witness=witness
    )
    grazing = enclose_local_vacuum_propagator(grazing_root, distance)

    _assert_matrix_contains(
        _entries(propagating), _oracle_matrix(Fraction(2), distance)
    )
    _assert_matrix_contains(
        _entries(evanescent), _oracle_matrix(Fraction(-9), distance)
    )
    _assert_matrix_contains(
        _entries(grazing), _oracle_matrix(Fraction(0), distance)
    )
    assert propagating.entire_transcript is not None
    assert evanescent.entire_transcript is not None
    assert grazing.entire_transcript is None
    assert propagating.interval_work_transcript.exact_work_count == 12
    assert evanescent.interval_work_transcript.exact_work_count == 12
    assert grazing.interval_work_transcript.exact_work_count == 0
    largest_bits = max(
        max(
            abs(value.lower.numerator).bit_length(),
            value.lower.denominator.bit_length(),
            abs(value.upper.numerator).bit_length(),
            value.upper.denominator.bit_length(),
        )
        for value in propagating.entries
    )
    assert largest_bits > 16_000
    assert len(propagating.propagator_evidence_digest) == 64


@pytest.mark.parametrize("q", (Fraction(4), Fraction(-4)))
def test_semigroup_determinant_and_symplectic_containment(q: Fraction) -> None:
    """Contain the semigroup law and determinant-one symplectic identity."""
    root = _classified(q)
    left_distance = Fraction(1, 5)
    right_distance = Fraction(2, 7)
    combined_distance = left_distance + right_distance
    left = enclose_local_vacuum_propagator(
        root, left_distance, precision_bits=192
    )
    right = enclose_local_vacuum_propagator(
        root, right_distance, precision_bits=192
    )
    direct = enclose_local_vacuum_propagator(
        root, combined_distance, precision_bits=192
    )
    composed = _matrix_multiply(_entries(right), _entries(left))
    oracle = _oracle_matrix(q, combined_distance)
    _assert_matrix_contains(composed, oracle)
    _assert_matrix_contains(_entries(direct), oracle)

    entries = _entries(direct)
    determinant = _interval_add(
        _interval_multiply(entries[0], entries[3]),
        _interval_negate(_interval_multiply(entries[1], entries[2])),
    )
    _assert_contains(determinant, Fraction(1))
    zero: _Interval = (Fraction(0), Fraction(0))
    one: _Interval = (Fraction(1), Fraction(1))
    minus_one: _Interval = (Fraction(-1), Fraction(-1))
    symplectic = _matrix_multiply(
        _matrix_multiply(
            (entries[0], entries[2], entries[1], entries[3]),
            (zero, one, minus_one, zero),
        ),
        entries,
    )
    for interval, expected in zip(
        symplectic,
        (Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
        strict=True,
    ):
        _assert_contains(interval, expected)


def test_zero_and_tiny_distance_are_symbolic_and_anticancellation_safe() -> (
    None
):
    """Keep exact identity at zero and a nonzero tiny sine-over-root term."""
    root = _classified(Fraction(4))
    zero = enclose_local_vacuum_propagator(root, Fraction(0))
    assert _entries(zero) == (
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1)),
    )
    assert zero.entire_transcript is None
    assert zero.interval_work_transcript.exact_work_count == 0

    tiny_distance = Fraction(1, 10**50)
    tiny = enclose_local_vacuum_propagator(
        root, tiny_distance, precision_bits=224
    )
    oracle = _oracle_matrix(Fraction(4), tiny_distance)
    _assert_matrix_contains(_entries(tiny), oracle)
    assert _entries(tiny)[1] != (Fraction(0), Fraction(0))


def test_typed_helper_root_interval_work_and_bit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate every exact helper and local resource failure unchanged."""
    propagating = _classified(Fraction(4))
    evanescent = _classified(Fraction(-4))

    with pytest.raises(EntireEnclosureError) as term_error:
        enclose_local_vacuum_propagator(
            propagating, Fraction(5), maximum_terms=1
        )
    assert term_error.value.failure is (
        EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )

    with pytest.raises(EntireEnclosureError) as helper_work_error:
        enclose_local_vacuum_propagator(
            propagating, Fraction(1), maximum_entire_work=1
        )
    assert helper_work_error.value.failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )

    with pytest.raises(EntireEnclosureError) as range_error:
        enclose_local_vacuum_propagator(
            evanescent,
            Fraction(2),
            maximum_range_reductions=0,
        )
    assert range_error.value.failure is (
        EntireEnclosureFailure.RANGE_REDUCTION_LIMIT
    )

    with pytest.raises(EntireEnclosureError) as interval_work_error:
        enclose_local_vacuum_propagator(
            propagating, Fraction(1), maximum_interval_work=1
        )
    assert interval_work_error.value.failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert interval_work_error.value.exact_work_count == 2

    with pytest.raises(EntireEnclosureError) as root_work_error:
        classify_local_vacuum_root(
            (Fraction(2), Fraction(2)), maximum_root_work=1
        )
    assert root_work_error.value.failure is (
        EntireEnclosureFailure.WORK_BUDGET_EXCEEDED
    )
    assert root_work_error.value.exact_work_count == 2

    oversized = Fraction(1 << 80)
    with pytest.raises(EntireEnclosureError) as input_size_error:
        classify_local_vacuum_root(
            (oversized, oversized), maximum_rational_bits=64
        )
    assert input_size_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert input_size_error.value.exact_work_count == 0

    with pytest.raises(EntireEnclosureError) as intermediate_size_error:
        classify_local_vacuum_root(
            (Fraction(2), Fraction(2)), maximum_rational_bits=64
        )
    assert intermediate_size_error.value.failure is (
        EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert intermediate_size_error.value.exact_work_count > 0

    def fail_entire_root(value: Fraction) -> Fraction:
        """Force the entire helper's modulus-root call to fail."""
        del value
        raise ValueError("forced entire root failure")

    with monkeypatch.context() as context:
        context.setattr(entire, "sqrt_fraction_upper", fail_entire_root)
        with pytest.raises(EntireEnclosureError) as helper_root_error:
            enclose_local_vacuum_propagator(
                propagating, Fraction(1), precision_bits=96
            )
    assert helper_root_error.value.failure is (
        EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE
    )

    def invalid_local_root(value: Fraction) -> Fraction:
        """Force the local root helper to return an invalid enclosure."""
        del value
        return Fraction(-1)

    with monkeypatch.context() as context:
        context.setattr(local, "sqrt_fraction_upper", invalid_local_root)
        with pytest.raises(EntireEnclosureError) as local_root_error:
            classify_local_vacuum_root((Fraction(2), Fraction(2)))
    assert local_root_error.value.failure is (
        EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE
    )


def test_unclassified_root_cannot_propagate() -> None:
    """Reject a fully replayed unresolved root with a typed disposition."""
    root = _classified(Fraction(0))
    with pytest.raises(GalerkinLocalVacuumPropagationError) as error:
        enclose_local_vacuum_propagator(root, Fraction(0))
    assert error.value.failure is (
        GalerkinLocalVacuumPropagationFailure.ROOT_UNCLASSIFIED
    )


def test_root_and_propagator_replay_reject_self_rehashed_forgeries() -> None:
    """Reject raw-root and matrix forgeries despite recomputed public digests.

    :see: :func:`ptyrodactyl.galerkin.prepare_local_vacuum_propagator`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_vacuum_root_certificate`
    """
    root = _classified(Fraction(4))
    forged_q = dataclasses.replace(
        root.q_interval,
        lower_numerator=5,
        upper_numerator=5,
    )
    root_identity = local._root_identity_digest(forged_q, None)
    root_evidence = local._root_evidence_digest(
        forged_q,
        None,
        root.root_interval,
        root.work_transcript,
        root.classification,
        root_identity,
    )
    forged_root = dataclasses.replace(
        root,
        q_interval=forged_q,
        root_identity_digest=root_identity,
        root_evidence_digest=root_evidence,
    )
    with pytest.raises(ValueError, match="complete replay"):
        prepare_local_vacuum_root_certificate(forged_root)
    with pytest.raises(ValueError, match="complete replay"):
        enclose_local_vacuum_propagator(forged_root, Fraction(1, 3))

    propagator = enclose_local_vacuum_propagator(
        root, Fraction(1, 3), precision_bits=128
    )
    original = propagator.entries[0]
    forged_lower = original.lower - 1
    forged_entry = dataclasses.replace(
        original,
        lower_numerator=forged_lower.numerator,
        lower_denominator=forged_lower.denominator,
    )
    forged_entries = (
        forged_entry,
        propagator.entries[1],
        propagator.entries[2],
        propagator.entries[3],
    )
    policies: local._EntirePolicies = (
        propagator.precision_bits,
        propagator.maximum_terms,
        propagator.maximum_entire_work,
        propagator.maximum_range_reductions,
        propagator.interval_work_transcript.maximum_rational_bits,
    )
    forged_evidence = local._propagator_evidence_digest(
        propagator.root_certificate,
        forged_entries,
        propagator.entire_transcript,
        propagator.interval_work_transcript,
        propagator.distance,
        policies,
        propagator.propagator_formula,
        propagator.propagator_identity_digest,
    )
    forged_propagator = dataclasses.replace(
        propagator,
        entries=forged_entries,
        propagator_evidence_digest=forged_evidence,
    )
    with pytest.raises(ValueError, match="complete replay"):
        prepare_local_vacuum_propagator(forged_propagator, precision_bits=128)

    prepared = prepare_local_vacuum_propagator(propagator, precision_bits=128)
    assert local._exact_value_payload(prepared) == local._exact_value_payload(
        propagator
    )
    with pytest.raises(ValueError, match="does not match complete replay"):
        prepare_local_vacuum_root_certificate(root, maximum_root_work=63)
    with pytest.raises(ValueError, match="does not match complete replay"):
        prepare_local_vacuum_propagator(propagator, precision_bits=129)
    with pytest.raises(TypeError, match="GalerkinLocalVacuumRootCertificate"):
        enclose_local_vacuum_propagator(object(), Fraction(1))
    with pytest.raises(TypeError, match="GalerkinLocalVacuumPropagator"):
        prepare_local_vacuum_propagator(object())


__all__: list[str] = []
