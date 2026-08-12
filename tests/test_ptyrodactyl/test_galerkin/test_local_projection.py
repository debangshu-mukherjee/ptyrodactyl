r"""Tests for exact LVT.34--LVT.40 and LVT.55c--LVT.55e evidence."""

from __future__ import annotations

import functools
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.local_projection as projection
import ptyrodactyl.galerkin.local_stability as stability
from ptyrodactyl._tools import (
    ComplexRectangle,
    RootEnclosureError,
    conjugate_rectangle,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.galerkin.local_projection import (
    enclose_local_projection_defect,
    prepare_local_projection_defect_certificate,
)
from ptyrodactyl.types.local_projection_types import (
    GalerkinLocalProjectionDefectCertificate,
    GalerkinLocalProjectionDefectFailure,
    _make_local_projection_defect_certificate,
)
from ptyrodactyl.types.local_stability_types import (
    GalerkinLocalStabilityResult,
)
from ptyrodactyl.types.local_terminal_types import GalerkinLocalTerminalScope
from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalZeroSlabCertificate,
)
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_stability as stability_tests,
)
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_zero_slab as zero_tests,
)

_GRAM_PAIRS = 9
_STABILITY_PAIRS = 21
_STATE_BUDGET = np.float64(np.finfo(np.float64).max)
_FULL_SCOPE = GalerkinLocalTerminalScope.FULL_STATE_FIBERS
_SELECTED_SCOPE = GalerkinLocalTerminalScope.SELECTED_PRETERMINAL_FIBERS
_DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944"
    "592307816406286208998628034825342117067982148086513282306647"
    "093844609550582231725359408128481117450284102701938521105559"
    "644622948954930381964428810975665933446128475648233786783165"
    "271201909145648566923460348610454326648213393607260249141273"
)

type _GramEvidence = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
type _StateEvidence = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
type _PolicyEvidence = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
type _RationalReports = tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
]


def _decimal_sin_cos(angle: Decimal) -> tuple[Decimal, Decimal]:
    """Evaluate independent high-precision sine and cosine series."""
    two_pi = Decimal(2) * _DECIMAL_PI
    reduced = angle % two_pi
    if reduced > _DECIMAL_PI:
        reduced -= two_pi
    squared = reduced * reduced
    sine = reduced
    sine_term = reduced
    cosine = Decimal(1)
    cosine_term = Decimal(1)
    for order in range(1, 100):
        sine_term *= -squared / Decimal((2 * order) * (2 * order + 1))
        cosine_term *= -squared / Decimal((2 * order - 1) * (2 * order))
        sine += sine_term
        cosine += cosine_term
    return +sine, +cosine


def _decimal_fraction(value: Fraction) -> Decimal:
    """Convert one rational exactly in the active Decimal context."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def _make_stability_result(
    zero_slab: GalerkinLocalZeroSlabCertificate,
    *,
    field: jnp.ndarray | None = None,
    maximum_state_error: np.float64 = _STATE_BUDGET,
    maximum_direct_pairs: int = _STABILITY_PAIRS,
) -> GalerkinLocalStabilityResult:
    """Build a canonical L6 result after reusing one prepared L5 parent."""
    represented = zero_slab.represented_source_certificate
    solve = stability_tests._solve(represented, field=field)
    proof = stability_tests._check_prepared(
        represented,
        solve,
        maximum_state_error=maximum_state_error,
        maximum_direct_pairs=maximum_direct_pairs,
    )
    return stability._make_local_stability_result(
        represented,
        solve,
        proof,
        result_identity_digest=proof.result_identity_digest,
        result_evidence_digest=stability._result_evidence_digest(
            represented,
            solve,
            proof,
        ),
        completion_scope=stability._COMPLETION_SCOPE,
    )


@functools.lru_cache(maxsize=1)
def _parents() -> tuple[
    GalerkinLocalZeroSlabCertificate,
    GalerkinLocalStabilityResult,
]:
    """Return one shared zero-slab/L6 same-source parent pair."""
    zero_slab = zero_tests._canonical_slab()
    result = _make_stability_result(zero_slab)
    assert bool(result.proof.state_radius_eligible)
    assert bool(result.proof.operational_state_eligible)
    return zero_slab, result


@functools.lru_cache(maxsize=2)
def _certificate(
    scope: GalerkinLocalTerminalScope = _FULL_SCOPE,
) -> GalerkinLocalProjectionDefectCertificate:
    """Exercise projection arithmetic after both parents are prepared once."""
    zero_slab, result = _parents()
    return projection._certify_prepared(
        zero_slab,
        result,
        scope,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )


def _remake(
    certificate: GalerkinLocalProjectionDefectCertificate,
    *,
    state_evidence: _StateEvidence | None = None,
    gram_evidence: _GramEvidence | None = None,
    policy_evidence: _PolicyEvidence | None = None,
) -> GalerkinLocalProjectionDefectCertificate:
    """Re-enter the raw carrier factory with selected white-box evidence."""
    state = state_evidence or (
        certificate.state_to_fiber_rows,
        certificate.selected_state_mask,
        certificate.exact_free_diagonal_lower_bounds,
        certificate.exact_free_diagonal_upper_bounds,
        certificate.structural_exact_zero_state_mask,
    )
    gram = gram_evidence or (
        certificate.gram_real_lower_bounds,
        certificate.gram_real_upper_bounds,
        certificate.gram_imag_lower_bounds,
        certificate.gram_imag_upper_bounds,
    )
    fiber_evidence = (
        certificate.structural_exact_zero_fiber_mask,
        certificate.measured_defect_squared_lower_bounds,
        certificate.measured_defect_squared_upper_bounds,
        certificate.measured_defect_upper_bounds,
        certificate.operator_squared_norm_upper_bounds,
        certificate.operator_norm_upper_bounds,
        certificate.state_error_transfer_upper_bounds,
        certificate.total_defect_upper_bounds,
    )
    policies = policy_evidence or (
        certificate.state_radius_upper_bound,
        certificate.maximum_state_error,
        certificate.direct_pair_count,
        certificate.maximum_gram_pairs,
        certificate.maximum_stability_direct_pairs,
        certificate.pi_target_bits,
        certificate.sine_taylor_lower_last_index,
        certificate.sine_taylor_upper_last_index,
        certificate.sqrt_precision_bits,
    )
    eligibility = (
        certificate.host_binary64_eligible,
        certificate.normal_arithmetic_eligible,
        certificate.structural_exact_zero_eligible,
        certificate.finite_projection_bound_eligible,
        certificate.operational_budget_eligible,
    )
    return _make_local_projection_defect_certificate(
        certificate.zero_slab_certificate,
        certificate.stability_result,
        certificate.scope_transverse_indices,
        state,
        gram,
        fiber_evidence,
        policies,
        eligibility,
        certificate.failure_mask,
        terminal_axis=certificate.terminal_axis,
        projection_scope=certificate.projection_scope,
        maximum_state_error_numerator=(
            certificate.maximum_state_error_numerator
        ),
        maximum_state_error_denominator=(
            certificate.maximum_state_error_denominator
        ),
        direct_pair_count_exact=certificate.direct_pair_count_exact,
        gram_formula=certificate.gram_formula,
        measurement_formula=certificate.measurement_formula,
        operator_bound_formula=certificate.operator_bound_formula,
        state_lift_formula=certificate.state_lift_formula,
        precision_transcript=certificate.precision_transcript,
        error_scope=certificate.error_scope,
        completion_scope=certificate.completion_scope,
        target_digest=certificate.target_digest,
        parent_target_evidence_digest=(
            certificate.parent_target_evidence_digest
        ),
        source_digest=certificate.source_digest,
        parent_source_evidence_digest=(
            certificate.parent_source_evidence_digest
        ),
        parent_represented_certificate_digest=(
            certificate.parent_represented_certificate_digest
        ),
        parent_zero_slab_certificate_digest=(
            certificate.parent_zero_slab_certificate_digest
        ),
        parent_stability_result_identity_digest=(
            certificate.parent_stability_result_identity_digest
        ),
        parent_stability_result_evidence_digest=(
            certificate.parent_stability_result_evidence_digest
        ),
        state_identity_digest=certificate.state_identity_digest,
        projection_identity_digest=certificate.projection_identity_digest,
        arithmetic_environment_digest=(
            certificate.arithmetic_environment_digest
        ),
        gram_transcript_digest=certificate.gram_transcript_digest,
        certificate_digest=certificate.certificate_digest,
    )


def _rehashed_certificate(
    certificate: GalerkinLocalProjectionDefectCertificate,
) -> GalerkinLocalProjectionDefectCertificate:
    """Rehash every mutable projection field of one white-box forgery."""
    mapping = (
        np.asarray(certificate.scope_transverse_indices),
        np.asarray(certificate.state_to_fiber_rows),
        np.asarray(certificate.selected_state_mask),
    )
    free_evidence = (
        np.asarray(certificate.exact_free_diagonal_lower_bounds),
        np.asarray(certificate.exact_free_diagonal_upper_bounds),
        np.asarray(certificate.structural_exact_zero_state_mask),
    )
    gram_evidence = (
        np.asarray(certificate.gram_real_lower_bounds),
        np.asarray(certificate.gram_real_upper_bounds),
        np.asarray(certificate.gram_imag_lower_bounds),
        np.asarray(certificate.gram_imag_upper_bounds),
    )
    reports = (
        np.asarray(certificate.measured_defect_squared_lower_bounds),
        np.asarray(certificate.measured_defect_squared_upper_bounds),
        np.asarray(certificate.measured_defect_upper_bounds),
        np.asarray(certificate.operator_squared_norm_upper_bounds),
        np.asarray(certificate.operator_norm_upper_bounds),
        np.asarray(certificate.state_error_transfer_upper_bounds),
        np.asarray(certificate.total_defect_upper_bounds),
    )
    policies: tuple[object, ...] = (
        np.asarray(certificate.state_radius_upper_bound),
        np.asarray(certificate.maximum_state_error),
        np.asarray(certificate.direct_pair_count),
        np.asarray(certificate.maximum_gram_pairs),
        np.asarray(certificate.maximum_stability_direct_pairs),
        np.asarray(certificate.pi_target_bits),
        np.asarray(certificate.sine_taylor_lower_last_index),
        np.asarray(certificate.sine_taylor_upper_last_index),
        np.asarray(certificate.sqrt_precision_bits),
    )
    predicates = (
        bool(certificate.host_binary64_eligible),
        bool(certificate.normal_arithmetic_eligible),
        bool(certificate.structural_exact_zero_eligible),
        bool(certificate.finite_projection_bound_eligible),
        bool(certificate.operational_budget_eligible),
    )
    digest = projection._certificate_digest(
        certificate.zero_slab_certificate,
        certificate.stability_result,
        certificate.projection_identity_digest,
        certificate.direct_pair_count_exact,
        mapping,
        free_evidence,
        np.asarray(certificate.structural_exact_zero_fiber_mask),
        gram_evidence,
        reports,
        policies,
        predicates,
        GalerkinLocalProjectionDefectFailure(
            int(np.asarray(certificate.failure_mask))
        ),
        projection._environment_payload()[0],
        certificate.gram_transcript_digest,
    )
    return replace(certificate, certificate_digest=digest)


def _assert_gram_structure(
    certificate: GalerkinLocalProjectionDefectCertificate,
) -> None:
    """Check public block-zero and Hermitian rectangle invariants."""
    rows = np.asarray(certificate.state_to_fiber_rows)
    selected = np.asarray(certificate.selected_state_mask)
    real_lower = np.asarray(certificate.gram_real_lower_bounds)
    real_upper = np.asarray(certificate.gram_real_upper_bounds)
    imag_lower = np.asarray(certificate.gram_imag_lower_bounds)
    imag_upper = np.asarray(certificate.gram_imag_upper_bounds)
    block = (
        selected[:, None]
        & selected[None, :]
        & (rows[:, None] == rows[None, :])
    )
    for values in (real_lower, real_upper, imag_lower, imag_upper):
        assert_array_equal(values[~block], 0.0)
    assert_array_equal(real_lower, real_lower.T)
    assert_array_equal(real_upper, real_upper.T)
    assert_array_equal(imag_lower, -imag_upper.T)
    assert_array_equal(imag_upper, -imag_lower.T)
    diagonal = np.diag_indices(real_lower.shape[0])
    assert np.all(imag_lower[diagonal] <= 0.0)
    assert np.all(imag_upper[diagonal] >= 0.0)


def test_exact_gram_phase_sign_conjugacy_and_fiber_reports() -> None:
    """Verify the exact LVT.55c sign and conjugated LVT.55d quadratic."""
    huge_fraction = Fraction((1 << 20_000) + 1, (1 << 20_001) + 3)
    huge_payload = projection._fraction_payload(huge_fraction)
    assert huge_payload["numerator_hex"] == "+" + format(
        huge_fraction.numerator,
        "x",
    )
    assert huge_payload["denominator_hex"] == format(
        huge_fraction.denominator,
        "x",
    )
    huge_rectangle: ComplexRectangle = (
        huge_fraction,
        huge_fraction,
        Fraction(0),
        Fraction(0),
    )
    huge_gram = [[huge_rectangle]]
    huge_rows = np.asarray([0], dtype=np.int64)
    huge_selected = np.asarray([True], dtype=np.bool_)
    huge_digest = projection._gram_transcript_digest(
        huge_gram,
        huge_rows,
        huge_selected,
    )
    assert len(huge_digest) == 64
    assert huge_digest == projection._gram_transcript_digest(
        huge_gram,
        huge_rows,
        huge_selected,
    )
    delta = Fraction(1, 6)
    midpoint = Fraction(1)
    length = Fraction(3)
    diagonal = projection._gram_rectangle(delta, midpoint, length, 2, 2)
    forward = projection._gram_rectangle(delta, midpoint, length, -1, 1)
    reverse = projection._gram_rectangle(delta, midpoint, length, 1, -1)
    unwrapped = projection._gram_rectangle(
        delta,
        midpoint + 3 * length,
        length,
        -1,
        1,
    )
    assert diagonal == (delta, delta, Fraction(0), Fraction(0))
    conjugate = conjugate_rectangle(forward)
    assert max(reverse[0], conjugate[0]) <= min(reverse[1], conjugate[1])
    assert max(reverse[2], conjugate[2]) <= min(reverse[3], conjugate[3])
    assert unwrapped == forward
    assert forward[3] < 0
    with localcontext() as context:
        context.prec = 140
        wavevector = Decimal(2) * _DECIMAL_PI * Decimal(2) / Decimal(3)
        lower_angle = wavevector * Decimal(3) / Decimal(4)
        upper_angle = wavevector * Decimal(5) / Decimal(4)
        lower_sine, lower_cosine = _decimal_sin_cos(lower_angle)
        upper_sine, upper_cosine = _decimal_sin_cos(upper_angle)
        integral_real = (upper_sine - lower_sine) / (wavevector * Decimal(3))
        integral_imag = (lower_cosine - upper_cosine) / (
            wavevector * Decimal(3)
        )
        assert _decimal_fraction(forward[0]) <= integral_real
        assert integral_real <= _decimal_fraction(forward[1])
        assert _decimal_fraction(forward[2]) <= integral_imag
        assert integral_imag <= _decimal_fraction(forward[3])

    zero: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    one: ComplexRectangle = (
        Fraction(1),
        Fraction(1),
        Fraction(0),
        Fraction(0),
    )
    gram: list[list[ComplexRectangle]] = [[one, zero], [zero, one]]
    reports = projection._fiber_rational_reports(
        np.asarray([0, 1], dtype=np.int64),
        gram,
        [(Fraction(1), Fraction(1)), (Fraction(2), Fraction(2))],
        np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128),
        Fraction(1, 2),
    )
    measured = sqrt_fraction_upper(Fraction(105))
    assert reports == (
        Fraction(105),
        Fraction(105),
        measured,
        Fraction(4),
        Fraction(2),
        Fraction(1),
        measured + 1,
    )
    interval_reports = projection._fiber_rational_reports(
        np.asarray([0], dtype=np.int64),
        [[one]],
        [(Fraction(1), Fraction(2))],
        np.asarray([3.0 + 4.0j], dtype=np.complex128),
        Fraction(1, 2),
    )
    assert interval_reports == (
        Fraction(25),
        Fraction(100),
        Fraction(10),
        Fraction(4),
        Fraction(2),
        Fraction(1),
        Fraction(11),
    )


def test_scope_mapping_omits_whole_synthetic_fibers_and_changes_work() -> None:
    """Distinguish full and selected scopes on two complete fibers."""
    state = jnp.asarray(
        [
            (normal, transverse, 0)
            for transverse in (0, 1)
            for normal in (-1, 0, 1)
        ],
        dtype=jnp.int64,
    )
    state_host = np.asarray(state, dtype=np.int64)
    selected_transverse = np.asarray([[0, 0]], dtype=np.int64)
    full_fibers, full_rows, full_selected = projection._scope_mapping(
        state_host,
        selected_transverse,
        0,
        _FULL_SCOPE,
    )
    selected_fibers, selected_rows, selected_mask = projection._scope_mapping(
        state_host,
        selected_transverse,
        0,
        _SELECTED_SCOPE,
    )
    assert_array_equal(full_fibers, [[0, 0], [1, 0]])
    assert_array_equal(full_rows, [0, 0, 0, 1, 1, 1])
    assert_array_equal(full_selected, np.ones((6,), dtype=np.bool_))
    assert projection._direct_pair_count(full_rows, full_selected) == 18
    assert_array_equal(selected_fibers, [[0, 0]])
    assert_array_equal(selected_rows, np.zeros((6,), dtype=np.int64))
    assert_array_equal(
        selected_mask,
        [True, True, True, False, False, False],
    )
    assert projection._direct_pair_count(selected_rows, selected_mask) == 9


def test_projection_scopes_dense_oracle_and_independent_eligibility() -> None:
    """Check exact scopes, Gram enclosure, and three disjoint predicates."""
    full = _certificate(_FULL_SCOPE)
    selected = _certificate(_SELECTED_SCOPE)
    for certificate in (full, selected):
        _assert_gram_structure(certificate)
        assert int(certificate.direct_pair_count) == _GRAM_PAIRS
        assert certificate.direct_pair_count_exact == str(_GRAM_PAIRS)
        assert int(certificate.maximum_gram_pairs) == _GRAM_PAIRS
        assert int(certificate.maximum_stability_direct_pairs) == 21
        assert bool(certificate.finite_projection_bound_eligible)
        assert bool(certificate.operational_budget_eligible)
        assert not bool(certificate.structural_exact_zero_eligible)
        assert GalerkinLocalProjectionDefectFailure(
            int(certificate.failure_mask)
        ) is (
            GalerkinLocalProjectionDefectFailure.STRUCTURAL_EXACT_ZERO_UNAVAILABLE
        )
        assert "row-sum" in certificate.operator_bound_formula
        assert (
            "not an exact spectral norm" in certificate.operator_bound_formula
        )
        assert "appears once" in certificate.state_lift_formula

    assert (
        full.projection_identity_digest != selected.projection_identity_digest
    )
    assert_array_equal(
        full.scope_transverse_indices,
        selected.scope_transverse_indices,
    )
    gram = (
        np.asarray(full.gram_real_lower_bounds)
        + np.asarray(full.gram_real_upper_bounds)
    ) / 2.0 + 1.0j * (
        np.asarray(full.gram_imag_lower_bounds)
        + np.asarray(full.gram_imag_upper_bounds)
    ) / 2.0
    diagonal = (
        np.asarray(full.exact_free_diagonal_lower_bounds)
        + np.asarray(full.exact_free_diagonal_upper_bounds)
    ) / 2.0
    field = np.asarray(full.stability_result.solve_result.field)
    z = diagonal * field
    measured_squared = float(np.real(np.vdot(z, gram @ z)))
    measured_lower = np.nextafter(
        float(full.measured_defect_squared_lower_bounds[0]),
        -np.inf,
    )
    measured_upper = np.nextafter(
        float(full.measured_defect_squared_upper_bounds[0]),
        np.inf,
    )
    assert measured_lower <= measured_squared <= measured_upper
    operator = np.diag(diagonal) @ gram @ np.diag(diagonal)
    spectral_upper = np.sqrt(
        max(0.0, float(np.max(np.linalg.eigvalsh(operator))))
    )
    assert spectral_upper <= np.nextafter(
        float(full.operator_norm_upper_bounds[0]),
        np.inf,
    )


def test_structural_zero_never_uses_submitted_state_cancellation() -> None:
    """Keep exact-D singleton masks independent of numerical field zeros."""
    zero_slab, _ = _parents()
    target = zero_slab.represented_source_certificate.source.target
    size = target.state_indices.shape[0]
    zero_field = jnp.zeros((size,), dtype=jnp.complex128)
    zero_state_result = _make_stability_result(zero_slab, field=zero_field)
    cancelled = projection._certify_prepared(
        zero_slab,
        zero_state_result,
        _FULL_SCOPE,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )
    assert_allclose(cancelled.measured_defect_upper_bounds, 0.0)
    assert not bool(cancelled.structural_exact_zero_eligible)

    represented = zero_slab.represented_source_certificate
    target = represented.source.target
    zeros = jnp.zeros_like(
        target.fixed_linear_error_ledger.exact_free_diagonal_lower_bounds
    )
    ledger = replace(
        target.fixed_linear_error_ledger,
        exact_free_diagonal_lower_bounds=zeros,
        exact_free_diagonal_upper_bounds=zeros,
    )
    replaced_target = replace(target, fixed_linear_error_ledger=ledger)
    replaced_source = replace(represented.source, target=replaced_target)
    replaced_represented = replace(represented, source=replaced_source)
    replaced_zero = replace(
        zero_slab,
        represented_source_certificate=replaced_represented,
    )
    replaced_result = replace(
        zero_state_result,
        certificate=replaced_represented,
    )
    structural = projection._certify_prepared(
        replaced_zero,
        replaced_result,
        _FULL_SCOPE,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )
    assert bool(structural.structural_exact_zero_eligible)
    assert_allclose(structural.operator_norm_upper_bounds, 0.0)
    assert_allclose(structural.total_defect_upper_bounds, 0.0)


def test_state_radius_and_operational_budget_failures_remain_distinct() -> (
    None
):
    """Do not report a state-budget miss when no finite radius exists."""
    zero_slab, _ = _parents()
    tiny = np.float64(np.finfo(np.float64).tiny)
    nonoperational = _make_stability_result(
        zero_slab,
        maximum_state_error=tiny,
    )
    fallback = projection._certify_prepared(
        zero_slab,
        nonoperational,
        _FULL_SCOPE,
        tiny,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )
    failure = GalerkinLocalProjectionDefectFailure(int(fallback.failure_mask))
    assert bool(nonoperational.proof.state_radius_eligible)
    assert not bool(nonoperational.proof.operational_state_eligible)
    assert bool(fallback.finite_projection_bound_eligible)
    assert not bool(fallback.operational_budget_eligible)
    assert (
        failure
        & GalerkinLocalProjectionDefectFailure.OPERATIONAL_STATE_BUDGET_MISSED
    )
    assert (
        not failure
        & GalerkinLocalProjectionDefectFailure.STATE_RADIUS_UNAVAILABLE
    )

    unavailable = _make_stability_result(
        zero_slab,
        maximum_direct_pairs=1,
    )
    rejected = projection._certify_prepared(
        zero_slab,
        unavailable,
        _FULL_SCOPE,
        _STATE_BUDGET,
        1,
        _GRAM_PAIRS,
    )
    failure = GalerkinLocalProjectionDefectFailure(int(rejected.failure_mask))
    assert not bool(unavailable.proof.state_radius_eligible)
    assert (
        failure & GalerkinLocalProjectionDefectFailure.STATE_RADIUS_UNAVAILABLE
    )
    assert not failure & (
        GalerkinLocalProjectionDefectFailure.OPERATIONAL_STATE_BUDGET_MISSED
    )
    assert not bool(rejected.finite_projection_bound_eligible)


def test_projection_pair_policy_and_parent_evidence_are_independent() -> None:
    """Separate L6 work, projection work, and the complete source parent."""
    zero_slab, result = _parents()
    budgeted = projection._certify_prepared(
        zero_slab,
        result,
        _FULL_SCOPE,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS - 1,
    )
    failure = GalerkinLocalProjectionDefectFailure(int(budgeted.failure_mask))
    assert (
        failure
        & GalerkinLocalProjectionDefectFailure.GRAM_PAIR_BUDGET_EXCEEDED
    )
    assert int(budgeted.maximum_stability_direct_pairs) == _STABILITY_PAIRS
    assert int(budgeted.maximum_gram_pairs) == _GRAM_PAIRS - 1

    crossed_parent = replace(
        result.certificate,
        certificate_digest="a" * 64,
    )
    crossed_result = replace(result, certificate=crossed_parent)
    crossed = projection._certify_prepared(
        zero_slab,
        crossed_result,
        _FULL_SCOPE,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )
    failure = GalerkinLocalProjectionDefectFailure(int(crossed.failure_mask))
    assert failure & (
        GalerkinLocalProjectionDefectFailure.PARENT_SOURCE_EVIDENCE_MISMATCH
    )
    assert not bool(crossed.finite_projection_bound_eligible)


@pytest.mark.parametrize(
    "abnormal",
    [np.inf, np.nextafter(np.float64(0.0), np.float64(1.0))],
    ids=["overflow", "subnormal"],
)
def test_report_range_failures_return_typed_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    abnormal: np.float64,
) -> None:
    """Return typed noncertificates for overflowed or subnormal reports."""
    original = projection._reports_to_binary64

    def _abnormal_reports(
        rational_reports: list[_RationalReports],
        *,
        state_radius_available: bool,
    ) -> tuple[np.ndarray, ...]:
        reports = list(
            original(
                rational_reports,
                state_radius_available=state_radius_available,
            )
        )
        reports[0] = np.full_like(reports[0], abnormal)
        return tuple(reports)

    monkeypatch.setattr(projection, "_reports_to_binary64", _abnormal_reports)
    zero_slab, result = _parents()
    certificate = projection._certify_prepared(
        zero_slab,
        result,
        _FULL_SCOPE,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )
    failure = GalerkinLocalProjectionDefectFailure(
        int(certificate.failure_mask)
    )
    assert (
        failure & GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert np.all(np.isinf(certificate.total_defect_upper_bounds))
    assert not bool(certificate.finite_projection_bound_eligible)


def test_root_failure_and_carrier_gram_policy_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type root failures and reject malformed stored Gram/policy evidence."""
    zero_slab, result = _parents()

    def _root_failure(*args: object, **kwargs: object):
        raise RootEnclosureError("forced projection root failure")

    monkeypatch.setattr(projection, "_fiber_rational_reports", _root_failure)
    rooted = projection._certify_prepared(
        zero_slab,
        result,
        _FULL_SCOPE,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )
    failure = GalerkinLocalProjectionDefectFailure(int(rooted.failure_mask))
    assert (
        failure & GalerkinLocalProjectionDefectFailure.ROOT_ENCLOSURE_FAILURE
    )
    monkeypatch.undo()

    certificate = _certificate()
    selected = certificate.selected_state_mask.at[0].set(False)
    forged_state: _StateEvidence = (
        certificate.state_to_fiber_rows,
        selected,
        certificate.exact_free_diagonal_lower_bounds,
        certificate.exact_free_diagonal_upper_bounds,
        certificate.structural_exact_zero_state_mask,
    )
    with pytest.raises(ValueError, match="outside selected same-fiber"):
        _remake(certificate, state_evidence=forged_state)

    gram = [
        jnp.asarray(certificate.gram_real_lower_bounds),
        jnp.asarray(certificate.gram_real_upper_bounds),
        jnp.asarray(certificate.gram_imag_lower_bounds),
        jnp.asarray(certificate.gram_imag_upper_bounds),
    ]
    gram[0] = gram[0].at[0, 1].set(jnp.nextafter(gram[0][0, 1], -jnp.inf))
    forged_gram: _GramEvidence = (gram[0], gram[1], gram[2], gram[3])
    with pytest.raises(ValueError, match="conjugate-transpose symmetric"):
        _remake(certificate, gram_evidence=forged_gram)

    diagonal_gram = [
        jnp.asarray(certificate.gram_real_lower_bounds),
        jnp.asarray(certificate.gram_real_upper_bounds),
        jnp.asarray(certificate.gram_imag_lower_bounds),
        jnp.asarray(certificate.gram_imag_upper_bounds),
    ]
    tiny = jnp.asarray(np.finfo(np.float64).tiny, dtype=jnp.float64)
    diagonal_gram[2] = diagonal_gram[2].at[0, 0].set(tiny)
    diagonal_gram[3] = diagonal_gram[3].at[0, 0].set(tiny)
    forged_diagonal: _GramEvidence = (
        diagonal_gram[0],
        diagonal_gram[1],
        diagonal_gram[2],
        diagonal_gram[3],
    )
    with pytest.raises(ValueError, match="diagonal imaginary"):
        _remake(certificate, gram_evidence=forged_diagonal)

    policies = (
        jnp.nextafter(certificate.state_radius_upper_bound, jnp.inf),
        certificate.maximum_state_error,
        certificate.direct_pair_count,
        certificate.maximum_gram_pairs,
        certificate.maximum_stability_direct_pairs,
        certificate.pi_target_bits,
        certificate.sine_taylor_lower_last_index,
        certificate.sine_taylor_upper_last_index,
        certificate.sqrt_precision_bits,
    )
    with pytest.raises(ValueError, match="copied state radius"):
        _remake(certificate, policy_evidence=policies)

    state_policy = list(policies)
    state_policy[0] = certificate.state_radius_upper_bound
    state_policy[1] = jnp.nextafter(
        certificate.maximum_state_error,
        -jnp.inf,
    )
    forged_state_policy: _PolicyEvidence = (
        state_policy[0],
        state_policy[1],
        state_policy[2],
        state_policy[3],
        state_policy[4],
        state_policy[5],
        state_policy[6],
        state_policy[7],
        state_policy[8],
    )
    with pytest.raises(ValueError, match="copied state policy"):
        _remake(certificate, policy_evidence=forged_state_policy)

    work_policy = list(forged_state_policy)
    work_policy[1] = certificate.maximum_state_error
    work_policy[4] = certificate.maximum_stability_direct_pairs + 1
    forged_work_policy: _PolicyEvidence = (
        work_policy[0],
        work_policy[1],
        work_policy[2],
        work_policy[3],
        work_policy[4],
        work_policy[5],
        work_policy[6],
        work_policy[7],
        work_policy[8],
    )
    with pytest.raises(ValueError, match="copied L6 work policy"):
        _remake(certificate, policy_evidence=forged_work_policy)


def test_projection_public_replay_binds_full_source_state_and_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonically rebuild once and reject a self-rehashed public forgery.

    :see: :func:`ptyrodactyl.galerkin.enclose_local_projection_defect`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_projection_defect_certificate`
    """
    certificate = _certificate()
    canonical = prepare_local_projection_defect_certificate(
        certificate,
        maximum_state_error=_STATE_BUDGET,
        maximum_stability_direct_pairs=_STABILITY_PAIRS,
        maximum_gram_pairs=_GRAM_PAIRS,
    )
    assert stored_value_payload(canonical) == stored_value_payload(certificate)

    changed_field = certificate.stability_result.solve_result.field.at[0].set(
        certificate.stability_result.solve_result.field[0] + (0.125 - 0.25j)
    )
    alternate_result = _make_stability_result(
        certificate.zero_slab_certificate,
        field=changed_field,
    )
    alternate_identity = projection._projection_identity_digest(
        certificate.zero_slab_certificate,
        alternate_result,
        certificate.projection_scope,
        np.asarray(certificate.scope_transverse_indices),
    )
    forged = _rehashed_certificate(
        replace(
            certificate,
            stability_result=alternate_result,
            parent_stability_result_identity_digest=(
                alternate_result.result_identity_digest
            ),
            parent_stability_result_evidence_digest=(
                alternate_result.result_evidence_digest
            ),
            state_identity_digest=alternate_result.result_identity_digest,
            projection_identity_digest=alternate_identity,
        )
    )
    assert forged.certificate_digest != certificate.certificate_digest
    monkeypatch.setattr(
        projection,
        "prepare_local_zero_slab_certificate",
        lambda value: value,
    )
    monkeypatch.setattr(
        projection,
        "prepare_local_galerkin_stability_result",
        lambda value, **kwargs: value,
    )
    with pytest.raises(ValueError, match="complete replay"):
        prepare_local_projection_defect_certificate(
            forged,
            maximum_state_error=_STATE_BUDGET,
            maximum_stability_direct_pairs=_STABILITY_PAIRS,
            maximum_gram_pairs=_GRAM_PAIRS,
        )
    with pytest.raises(ValueError, match="copied state policy"):
        prepare_local_projection_defect_certificate(
            certificate,
            maximum_state_error=np.float64(1.0),
            maximum_stability_direct_pairs=_STABILITY_PAIRS,
            maximum_gram_pairs=_GRAM_PAIRS,
        )
    with pytest.raises(ValueError, match="complete replay"):
        prepare_local_projection_defect_certificate(
            certificate,
            maximum_state_error=_STATE_BUDGET,
            maximum_stability_direct_pairs=_STABILITY_PAIRS,
            maximum_gram_pairs=_GRAM_PAIRS - 1,
        )

    with pytest.raises(TypeError, match="exact float64"):
        enclose_local_projection_defect(
            certificate.zero_slab_certificate,
            certificate.stability_result,
            scope=_FULL_SCOPE,
            maximum_state_error=np.float32(1.0),
        )


__all__: list[str] = []
