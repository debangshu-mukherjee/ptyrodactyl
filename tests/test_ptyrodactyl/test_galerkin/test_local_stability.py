r"""Tests for bounded exact-dyadic local represented-source stability."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import ptyrodactyl.galerkin.local_represented_sources as represented
import ptyrodactyl.galerkin.local_stability as stability
from ptyrodactyl._tools import (
    RootEnclosureError,
    fraction_upper_float,
)
from ptyrodactyl.galerkin.local_stability import (
    check_local_represented_galerkin_absorber_floor,
    invoke_local_represented_galerkin_stability,
    prepare_local_galerkin_stability_result,
)
from ptyrodactyl.types.born_types import (
    GalerkinCertificateReason,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    create_galerkin_solve_result,
)
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalRepresentedSourceCertificate,
)
from ptyrodactyl.types.local_stability_types import (
    GalerkinLocalStabilityDisposition,
    GalerkinLocalStabilityFailure,
    GalerkinLocalStabilityProof,
)
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_represented_sources as represented_tests,
)

type _ComplexQ = tuple[Fraction, Fraction]

_certificate = represented_tests._certificate
_source = represented_tests._source


def _solve(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    *,
    field: jnp.ndarray | None = None,
    reported_residual: jnp.ndarray | None = None,
    reported_norm: float = 0.0,
) -> GalerkinSolveResult:
    """Return one valid solve carrier with deliberately untrusted reports."""
    size = certificate.source.target.state_indices.shape[0]
    if field is None:
        field = jnp.asarray(
            [0.125 - 0.25j, -0.375 + 0.5j, 0.625 + 0.75j],
            dtype=jnp.complex128,
        )[:size]
    if reported_residual is None:
        reported_residual = jnp.zeros((size,), dtype=jnp.complex128)
    return create_galerkin_solve_result(
        field=field,
        residual=reported_residual,
        residual_norm=reported_norm,
        normal_residual_norm=reported_norm + 1.0,
        recurrence_residual_norm=reported_norm + 2.0,
        iterations=3,
        operator_applications=8,
        status=GalerkinSolveStatus.CONVERGED,
        converged=True,
        method=GalerkinSolveMethod.CGLS,
        certificate_reason=(
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND
        ),
    )


def _q(value: complex | np.complex128) -> _ComplexQ:
    """Convert one stored complex128 point to exact dyadic components."""
    point = complex(value)
    return Fraction.from_float(point.real), Fraction.from_float(point.imag)


def _add(left: _ComplexQ, right: _ComplexQ) -> _ComplexQ:
    """Add exact complex rationals."""
    return left[0] + right[0], left[1] + right[1]


def _subtract(left: _ComplexQ, right: _ComplexQ) -> _ComplexQ:
    """Subtract exact complex rationals."""
    return left[0] - right[0], left[1] - right[1]


def _multiply(left: _ComplexQ, right: _ComplexQ) -> _ComplexQ:
    """Multiply exact complex rationals."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _independent_residual_squared(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    field: jnp.ndarray,
) -> Fraction:
    """Independently form exact ``b-Dx+Rx+iBx`` from stored dyadics."""
    source = certificate.source
    target = source.target
    size = target.state_indices.shape[0]
    x = tuple(_q(value) for value in np.asarray(field))
    residual = [_q(value) for value in np.asarray(source.actions.total_source)]
    diagonal = np.asarray(target.free_diagonal)
    interaction = np.asarray(target.compression.interaction_coefficients)
    interaction_positions = np.asarray(
        target.compression.state_pair_interaction_positions
    )
    cap_certificate = target.cap_floor_proof.coefficient_certificate
    absorber = cap_certificate.absorber
    absorber_coefficients = np.asarray(absorber.absorber_coefficients)
    absorber_positions = np.asarray(
        cap_certificate.state_pair_absorber_positions
    )
    cap_scale = Fraction.from_float(float(absorber.algebraic_cap_scale))
    for row in range(size):
        d = Fraction.from_float(float(diagonal[row]))
        residual[row] = _subtract(
            residual[row], (d * x[row][0], d * x[row][1])
        )
        for column in range(size):
            flat = row * size + column
            residual[row] = _add(
                residual[row],
                _multiply(
                    _q(interaction[interaction_positions[flat]]), x[column]
                ),
            )
            cap = _q(absorber_coefficients[absorber_positions[flat]])
            cap_product = _multiply(
                (cap_scale * cap[0], cap_scale * cap[1]), x[column]
            )
            residual[row] = _add(
                residual[row], (-cap_product[1], cap_product[0])
            )
    return sum(
        (real * real + imag * imag for real, imag in residual),
        start=Fraction(0),
    )


def _fraction(proof: object, prefix: str) -> Fraction:
    """Read one numerator/denominator pair from a proof."""
    return Fraction(
        getattr(proof, f"{prefix}_numerator"),
        getattr(proof, f"{prefix}_denominator"),
    )


def _check_prepared(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
    *,
    maximum_state_error: object,
    maximum_direct_pairs: int,
) -> GalerkinLocalStabilityProof:
    """Exercise local arithmetic after the shared parent is prepared once."""
    solve = stability._validate_solve_result(
        solve_result, certificate.source.target.state_indices.shape[0]
    )
    budget = stability._checked_positive_budget(
        maximum_state_error, "maximum_state_error"
    )
    pairs = stability._checked_pair_budget(maximum_direct_pairs)
    return stability._check_canonical_local_stability(
        certificate,
        solve,
        budget,
        pairs,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        1,
        np.int64(1),
        np.float32(1.0),
        jnp.asarray(1.0, dtype=jnp.float32),
        1.0 + 0.0j,
    ],
)
def test_state_budget_rejects_nonbinary64_input_dtype(invalid: object) -> None:
    """Reject bool, integer, float32, and complex budgets before conversion."""
    with pytest.raises(ValueError, match="float64|binary64"):
        stability._checked_positive_budget(invalid, "maximum_state_error")
    assert stability._checked_positive_budget(1.0, "budget") == 1.0
    assert stability._checked_positive_budget(np.float64(1.0), "budget") == 1.0
    assert (
        stability._checked_positive_budget(
            jnp.asarray(1.0, dtype=jnp.float64), "budget"
        )
        == 1.0
    )


def test_exact_work_count_is_unclamped_signed_int64_evidence() -> None:
    """Keep the exact count formula and overflow boundary explicit."""
    maximum = np.iinfo(np.int64).max
    boundary = (math.isqrt(1 + 8 * maximum) - 1) // 4
    while stability._direct_work_count(boundary + 1) <= maximum:
        boundary += 1
    assert stability._direct_work_count(boundary) <= maximum
    assert stability._direct_work_count(boundary + 1) > maximum


def test_exact_dense_residual_lift_budget_fallback_and_report_isolation() -> (
    None
):
    """Verify the exact residual and charge nonzero delta-H/eta-T once.

    :see: :func:`ptyrodactyl.galerkin.\
check_local_represented_galerkin_absorber_floor`
    """
    certificate = _certificate()
    first = _solve(certificate)
    forged_reports = _solve(
        certificate,
        field=first.field,
        reported_residual=jnp.full_like(first.field, 7.0 + 9.0j),
        reported_norm=123.0,
    )
    tiny_budget = np.float64(np.finfo(np.float64).tiny)
    proof = check_local_represented_galerkin_absorber_floor(
        certificate,
        first,
        maximum_state_error=tiny_budget,
        maximum_direct_pairs=21,
    )
    forged_proof = _check_prepared(
        certificate,
        forged_reports,
        maximum_state_error=tiny_budget,
        maximum_direct_pairs=21,
    )
    independent = _independent_residual_squared(certificate, first.field)
    assert _fraction(proof, "residual_squared") == independent
    assert _fraction(forged_proof, "residual_squared") == independent
    for prefix in (
        "exact_floor",
        "residual_squared",
        "field_norm_squared",
        "algebraic_residual_upper",
        "field_norm_upper",
        "fixed_linear_error",
        "fixed_linear_state_transfer_upper",
        "total_source_error_upper",
        "exact_target_residual_upper",
        "state_radius_upper",
    ):
        assert _fraction(proof, prefix) == _fraction(forged_proof, prefix)
    assert proof.result_identity_digest == forged_proof.result_identity_digest
    assert proof.proof_evidence_digest != forged_proof.proof_evidence_digest

    rho = _fraction(proof, "algebraic_residual_upper")
    xnorm = _fraction(proof, "field_norm_upper")
    delta_h = _fraction(proof, "fixed_linear_error")
    transfer = _fraction(proof, "fixed_linear_state_transfer_upper")
    eta_t = _fraction(proof, "total_source_error_upper")
    residual = _fraction(proof, "exact_target_residual_upper")
    floor = _fraction(proof, "exact_floor")
    radius = _fraction(proof, "state_radius_upper")
    assert delta_h > 0
    assert eta_t > 0
    assert transfer == Fraction.from_float(
        fraction_upper_float(delta_h * xnorm)
    )
    assert residual == Fraction.from_float(
        fraction_upper_float(rho + transfer + eta_t)
    )
    assert radius == Fraction.from_float(
        fraction_upper_float(residual / floor)
    )
    assert bool(proof.state_radius_eligible)
    assert not bool(proof.operational_state_eligible)
    assert proof.disposition is (
        GalerkinLocalStabilityDisposition.FINITE_STATE_RADIUS_FALLBACK
    )
    assert proof.failure is GalerkinLocalStabilityFailure.STATE_BUDGET_MISSED
    with pytest.raises(ValueError, match="complete replay"):
        invoke_local_represented_galerkin_stability(
            certificate,
            first,
            proof,
            maximum_state_error=np.float64(np.finfo(np.float64).max),
            maximum_direct_pairs=21,
        )
    pair_policy_variant = _check_prepared(
        certificate,
        first,
        maximum_state_error=tiny_budget,
        maximum_direct_pairs=22,
    )
    assert int(pair_policy_variant.maximum_direct_pairs) == 22
    assert pair_policy_variant.proof_evidence_digest != (
        proof.proof_evidence_digest
    )
    result = invoke_local_represented_galerkin_stability(
        certificate,
        first,
        proof,
        maximum_state_error=tiny_budget,
        maximum_direct_pairs=21,
    )
    assert result.certificate is not None
    assert result.solve_result is not None
    assert result.proof is not None
    assert (
        prepare_local_galerkin_stability_result(
            result,
            maximum_state_error=tiny_budget,
            maximum_direct_pairs=21,
        ).result_evidence_digest
        == result.result_evidence_digest
    )


def test_operational_pass_uses_exact_l4_floor_not_realized_floor() -> None:
    """Ignore realized-floor failure while retaining the exact L4 floor."""
    certificate = _certificate()
    floor = certificate.source.target.cap_floor_proof
    fake_floor = replace(
        floor,
        realized_floor_eligible=jnp.asarray(False, dtype=jnp.bool_),
        realized_physical_floor_lower_bound=jnp.asarray(
            0.0, dtype=jnp.float64
        ),
    )
    fake_target = replace(
        certificate.source.target, cap_floor_proof=fake_floor
    )
    fake_source = replace(certificate.source, target=fake_target)
    fake_certificate = replace(certificate, source=fake_source)
    proof = _check_prepared(
        fake_certificate,
        _solve(fake_certificate),
        maximum_state_error=np.float64(np.finfo(np.float64).max),
        maximum_direct_pairs=21,
    )
    assert bool(proof.state_radius_eligible)
    assert bool(proof.operational_state_eligible)
    assert proof.failure is GalerkinLocalStabilityFailure.NONE
    assert proof.lower_singular_bound == (
        floor.exact_target_physical_floor_lower_bound
    )
    assert "no realized floor" in proof.floor_scope


def test_exact_outward_radius_is_the_sharp_state_budget_boundary() -> None:
    """Pass at the stored outward radius and fall back one ulp below it."""
    certificate = _certificate()
    solve = _solve(certificate)
    roomy = _check_prepared(
        certificate,
        solve,
        maximum_state_error=np.float64(np.finfo(np.float64).max),
        maximum_direct_pairs=21,
    )
    boundary = np.float64(roomy.state_radius_upper_bound)
    below = np.nextafter(boundary, np.float64(0.0))
    assert boundary >= np.finfo(np.float64).tiny
    assert below >= np.finfo(np.float64).tiny

    exact = _check_prepared(
        certificate,
        solve,
        maximum_state_error=boundary,
        maximum_direct_pairs=21,
    )
    fallback = _check_prepared(
        certificate,
        solve,
        maximum_state_error=below,
        maximum_direct_pairs=21,
    )
    assert bool(exact.operational_state_eligible)
    assert exact.failure is GalerkinLocalStabilityFailure.NONE
    assert bool(fallback.state_radius_eligible)
    assert not bool(fallback.operational_state_eligible)
    assert fallback.failure is (
        GalerkinLocalStabilityFailure.STATE_BUDGET_MISSED
    )


def test_typed_source_host_work_floor_root_and_count_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return typed preflight, floor, root, and exact-count failures."""
    certificate = _certificate()
    solve = _solve(certificate)
    kwargs = {
        "maximum_state_error": np.float64(1.0),
        "maximum_direct_pairs": 21,
    }
    failed_source = represented._certify_canonical(_source(), 1)
    source_failure = _check_prepared(
        failed_source, _solve(failed_source), **kwargs
    )
    assert source_failure.failure is (
        GalerkinLocalStabilityFailure.SOURCE_NONCERTIFICATE
    )
    assert bool(source_failure.matrix_floor_eligible)
    assert source_failure.disposition is (
        GalerkinLocalStabilityDisposition.MATRIX_FLOOR_ONLY
    )
    work = _check_prepared(
        certificate,
        solve,
        maximum_state_error=np.float64(1.0),
        maximum_direct_pairs=1,
    )
    assert work.failure is (
        GalerkinLocalStabilityFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    assert bool(work.matrix_floor_eligible)
    with monkeypatch.context() as patch:
        patch.setattr(stability, "host_binary64_supported", lambda: False)
        host = _check_prepared(certificate, solve, **kwargs)
    assert host.failure is (
        GalerkinLocalStabilityFailure.HOST_ARITHMETIC_UNSUPPORTED
    )
    assert bool(host.matrix_floor_eligible)

    def fail_root(value: Fraction) -> Fraction:
        """Force verified rational root enclosure to fail."""
        del value
        raise RootEnclosureError("forced local stability root failure")

    with monkeypatch.context() as patch:
        patch.setattr(stability, "sqrt_fraction_upper", fail_root)
        root = _check_prepared(certificate, solve, **kwargs)
    assert root.failure is GalerkinLocalStabilityFailure.ROOT_ENCLOSURE_FAILURE
    assert bool(root.matrix_floor_eligible)
    with monkeypatch.context() as patch:
        patch.setattr(
            stability,
            "_direct_work_count",
            lambda size: np.iinfo(np.int64).max + size,
        )
        count = _check_prepared(certificate, solve, **kwargs)
    assert count.failure is (
        GalerkinLocalStabilityFailure.DIRECT_WORK_COUNT_OVERFLOW
    )
    assert bool(count.matrix_floor_eligible)
    assert int(count.direct_work_count) == 0
    assert int(count.direct_work_count_exact) > np.iinfo(np.int64).max

    floor = certificate.source.target.cap_floor_proof
    for changed, failure in (
        (
            replace(
                floor,
                exact_target_floor_eligible=jnp.asarray(
                    False, dtype=jnp.bool_
                ),
            ),
            GalerkinLocalStabilityFailure.EXACT_TARGET_FLOOR_UNAVAILABLE,
        ),
        (
            replace(
                floor,
                exact_target_physical_floor_lower_bound=jnp.asarray(
                    0.0, dtype=jnp.float64
                ),
            ),
            GalerkinLocalStabilityFailure.NONPOSITIVE_EXACT_TARGET_FLOOR,
        ),
    ):
        fake_target = replace(
            certificate.source.target, cap_floor_proof=changed
        )
        fake_source = replace(certificate.source, target=fake_target)
        fake_certificate = replace(certificate, source=fake_source)
        proof = _check_prepared(
            fake_certificate, _solve(fake_certificate), **kwargs
        )
        assert proof.failure is failure
        assert not bool(proof.matrix_floor_eligible)


@pytest.mark.parametrize("forced_call", [1, 2, 3, 4, 5])
def test_subnormal_transfer_residual_and_radius_are_typed_range_failures(
    monkeypatch: pytest.MonkeyPatch,
    forced_call: int,
) -> None:
    """Type-reject subnormal rho, norm, transfer, residual, and radius."""
    certificate = _certificate()
    solve = _solve(certificate)
    original = stability.fraction_upper_float
    calls = 0

    def force_one_subnormal(value: Fraction) -> float:
        """Force transfer, residual, or radius storage to be subnormal."""
        nonlocal calls
        calls += 1
        if calls == forced_call:
            return float(np.nextafter(0.0, 1.0))
        return original(value)

    monkeypatch.setattr(stability, "fraction_upper_float", force_one_subnormal)
    proof = _check_prepared(
        certificate,
        solve,
        maximum_state_error=np.float64(1.0),
        maximum_direct_pairs=21,
    )
    assert proof.failure is (
        GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert bool(proof.matrix_floor_eligible)
    assert not bool(proof.state_radius_eligible)
    assert proof.disposition is (
        GalerkinLocalStabilityDisposition.MATRIX_FLOOR_ONLY
    )


def test_matrix_floor_only_result_replays_under_identical_policies() -> None:
    """Preserve exact L4 evidence through a bounded-work result replay."""
    certificate = _certificate()
    solve = _solve(certificate)
    state_budget = np.float64(1.0)
    proof = _check_prepared(
        certificate,
        solve,
        maximum_state_error=state_budget,
        maximum_direct_pairs=1,
    )
    cap_floor_proof = certificate.source.target.cap_floor_proof
    parent_floor = cap_floor_proof.exact_target_physical_floor_lower_bound
    assert proof.failure is (
        GalerkinLocalStabilityFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    assert proof.disposition is (
        GalerkinLocalStabilityDisposition.MATRIX_FLOOR_ONLY
    )
    assert bool(proof.matrix_floor_eligible)
    assert not bool(proof.state_radius_eligible)
    assert not bool(proof.operational_state_eligible)
    assert proof.lower_singular_bound == parent_floor
    assert _fraction(proof, "exact_floor") == Fraction.from_float(
        float(parent_floor)
    )
    assert math.isinf(float(proof.exact_target_residual_upper_bound))
    assert math.isinf(float(proof.state_radius_upper_bound))

    result = invoke_local_represented_galerkin_stability(
        certificate,
        solve,
        proof,
        maximum_state_error=state_budget,
        maximum_direct_pairs=1,
    )
    replayed = prepare_local_galerkin_stability_result(
        result,
        maximum_state_error=state_budget,
        maximum_direct_pairs=1,
    )
    assert replayed.result_evidence_digest == result.result_evidence_digest


def test_full_replay_rejects_state_budget_proof_and_result_cross_pairs() -> (
    None
):
    """Reject same-schema cross-pairs even after attacker-side rehashing.

    :see: :func:`ptyrodactyl.galerkin.\
invoke_local_represented_galerkin_stability`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_galerkin_stability_result`
    """
    certificate = _certificate()
    first = _solve(certificate)
    proof = _check_prepared(
        certificate,
        first,
        maximum_state_error=np.float64(np.finfo(np.float64).max),
        maximum_direct_pairs=21,
    )
    crossed_source = replace(
        certificate.source, source_name="crossed-source-name"
    )
    with pytest.raises(ValueError, match="complete.*replay"):
        check_local_represented_galerkin_absorber_floor(
            replace(certificate, source=crossed_source),
            first,
            maximum_state_error=np.float64(1.0),
            maximum_direct_pairs=21,
        )
    crossed_target = replace(
        certificate.source.target, target_name="crossed-target-name"
    )
    with pytest.raises(ValueError, match="(?:complete|full).*replay"):
        check_local_represented_galerkin_absorber_floor(
            replace(
                certificate,
                source=replace(certificate.source, target=crossed_target),
            ),
            first,
            maximum_state_error=np.float64(1.0),
            maximum_direct_pairs=21,
        )
    other_field = first.field.at[0].set(first.field[0] + 1.0)
    crossed_solve = _solve(certificate, field=other_field)
    crossed_identity = stability._result_identity_digest(
        certificate, crossed_solve
    )
    changed_formula = replace(
        proof,
        residual_formula="forged formula",
        result_identity_digest=crossed_identity,
    )
    environment = stability._environment_payload()[0]
    forged_digest = stability._proof_evidence_digest(
        certificate, crossed_solve, changed_formula, environment
    )
    rehashed = replace(changed_formula, proof_evidence_digest=forged_digest)
    rehashed_result = stability._make_local_stability_result(
        certificate,
        crossed_solve,
        rehashed,
        result_identity_digest=crossed_identity,
        result_evidence_digest=stability._result_evidence_digest(
            certificate, crossed_solve, rehashed
        ),
        completion_scope=stability._COMPLETION_SCOPE,
    )
    with pytest.raises(ValueError, match="complete replay"):
        prepare_local_galerkin_stability_result(
            rehashed_result,
            maximum_state_error=np.float64(np.finfo(np.float64).max),
            maximum_direct_pairs=21,
        )


def test_legacy_source_noncertificate_rejects_before_parent_replay() -> None:
    """Reject the wrong source route before solve or parent inspection."""
    with pytest.raises(
        TypeError, match="GalerkinLocalRepresentedSourceCertificate"
    ):
        check_local_represented_galerkin_absorber_floor(
            object(),
            object(),
            maximum_state_error=np.float64(1.0),
        )
