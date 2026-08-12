r"""Falsification tests for local RM-S5 residual-observable arithmetic."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from decimal import Decimal, localcontext
from fractions import Fraction
from math import factorial
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import jax.numpy as jnp
import numpy as np
import pytest

import ptyrodactyl.types.local_residual_observable_types as residual_types
from ptyrodactyl._tools import (
    CensoredPoissonEnclosureError,
    CensoredPoissonEnclosureFailure,
    CensoredPoissonWorkTranscript,
    EntireEnclosureFailure,
    EntireWorkTranscript,
)
from ptyrodactyl.galerkin.detector import (
    create_local_censored_poisson_detector_input_manifest,
)
from ptyrodactyl.types.local_detector_types import (
    GalerkinLocalCensoredPoissonLikelihood,
    GalerkinLocalDetectorProductionStage,
    GalerkinLocalDetectorRationalInterval,
    _make_local_detector_rational_interval,
    _make_local_detector_real_production_trace,
)
from ptyrodactyl.types.local_residual_observable_types import (
    _LAW_SCOPE,
    _MEAN_SCOPE,
    _NLL_SCOPE,
    _NO_SCIENTIFIC_CLAIM_SCOPE,
    _RESOURCE_SCOPE,
    GalerkinLocalResidualObservableFailure,
    GalerkinLocalResidualObservableHelperCall,
    GalerkinLocalResidualObservableInputManifest,
    GalerkinLocalResidualObservableLayer,
    GalerkinLocalResidualObservableRoute,
    _derived_detector_mean_evidence,
    _make_local_residual_observable_certificate,
    _make_local_residual_observable_certificate_candidate,
    _make_local_residual_observable_input_manifest,
    _make_local_residual_observable_input_manifest_candidate,
    _state_authority_available,
    _trace_rounding_error,
    _tv_bound_evidence,
    _work_plan_counts,
)
from tests.test_ptyrodactyl.test_types import (
    test_local_detector_types as l9_type_tests,
)

type _Interval = tuple[Fraction, Fraction]


class _MeanOracle(NamedTuple):
    """Store the five-channel independent RM-S5 mean oracle."""

    production: tuple[_Interval, ...]
    exact: tuple[_Interval, ...]
    admitted: tuple[_Interval, ...]
    errors: tuple[Fraction, ...]
    fit_mask: tuple[bool, ...]
    observed: tuple[int, ...]
    ceilings: tuple[int, ...]
    full_l1: Fraction
    fitted_l1: Fraction


class _ScoreOracle(NamedTuple):
    """Store independent per-channel score bounds for the fitted lanes."""

    lipschitz: tuple[Decimal, ...]
    rounding: tuple[Decimal, ...]
    terms: tuple[Decimal, ...]
    total: Decimal
    predecessor_upper: Decimal
    tail_lower: Decimal


class _DirectOracle(NamedTuple):
    """Store the two fitted global NLL sums and their distance."""

    production_total: Decimal
    admitted_total: tuple[Decimal, Decimal]
    error: Decimal


class _ExpectedFixture(NamedTuple):
    """Store a parent-free exact-class L9 shell and its leaf oracle."""

    likelihood: GalerkinLocalCensoredPoissonLikelihood
    manifest: GalerkinLocalResidualObservableInputManifest
    mean: _MeanOracle
    production_nll_points: tuple[
        GalerkinLocalDetectorRationalInterval | None, ...
    ]
    total_nll: _Interval
    rounding_errors: tuple[Fraction, ...]
    tail_floor: Fraction


def _point(value: int | Fraction) -> _Interval:
    """Return one exact singleton interval."""
    rational = Fraction(value)
    return rational, rational


def _carrier(value: _Interval) -> GalerkinLocalDetectorRationalInterval:
    """Return one owner-created interval carrier."""
    return _make_local_detector_rational_interval(value[0], value[1])


def _raw(value: GalerkinLocalDetectorRationalInterval) -> _Interval:
    """Return one carrier's exact rational endpoints."""
    return value.lower, value.upper


def _union(left: _Interval, right: _Interval) -> _Interval:
    """Return the exact interval hull without production helpers."""
    return min(left[0], right[0]), max(left[1], right[1])


def _point_distance(point: Fraction, interval: _Interval) -> Fraction:
    """Return the maximum exact endpoint distance from one point."""
    return max(abs(point - interval[0]), abs(point - interval[1]))


def _sum_intervals(values: tuple[_Interval, ...]) -> _Interval:
    """Sum exact intervals without production helpers."""
    return (
        sum((value[0] for value in values), start=Fraction()),
        sum((value[1] for value in values), start=Fraction()),
    )


def _one_minus_exp_neg_decimal(value: Fraction) -> Decimal:
    """Evaluate the RM-S5 Poisson-TV curve with an independent oracle."""
    with localcontext() as context:
        context.prec = 100
        exact = Decimal(value.numerator) / Decimal(value.denominator)
        return Decimal(1) - (-exact).exp()


def _decimal(value: Fraction) -> Decimal:
    """Convert one exact rational to Decimal under the caller's context."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def _poisson_atom_decimal(mean: Fraction, value: int) -> Decimal:
    """Return one Poisson atom using only Decimal elementary arithmetic."""
    with localcontext() as context:
        context.prec = 100
        exact = Decimal(mean.numerator) / Decimal(mean.denominator)
        return (-exact).exp() * (exact**value) / Decimal(factorial(value))


def _poisson_tail_decimal(mean: Fraction, ceiling: int) -> Decimal:
    """Return one saturated Poisson-tail probability independently."""
    with localcontext() as context:
        context.prec = 100
        return Decimal(1) - sum(
            (_poisson_atom_decimal(mean, value) for value in range(ceiling)),
            start=Decimal(),
        )


def _unsaturated_nll_decimal(mean: Fraction, observed: int) -> Decimal:
    """Return one Poisson NLL including its state-independent constant."""
    if mean == 0 and observed == 0:
        return Decimal()
    with localcontext() as context:
        context.prec = 100
        exact = Decimal(mean.numerator) / Decimal(mean.denominator)
        return (
            exact
            - Decimal(observed) * exact.ln()
            + Decimal(factorial(observed)).ln()
        )


def _saturated_nll_decimal(mean: Fraction, ceiling: int) -> Decimal:
    """Return one saturated-symbol NLL from its independent tail."""
    with localcontext() as context:
        context.prec = 100
        return -_poisson_tail_decimal(mean, ceiling).ln()


def _score_lipschitz_decimal(
    admitted: _Interval,
    observed: int,
    ceiling: int,
) -> Decimal:
    """Return the exact RM-S5 score Lipschitz constant by symbol class."""
    with localcontext() as context:
        context.prec = 100
        if ceiling == 0:
            return Decimal()
        if observed == 0:
            return Decimal(1)
        if observed < ceiling:
            return max(
                abs(Decimal(1) - Decimal(observed) / _decimal(endpoint))
                for endpoint in admitted
            )
        predecessor_upper = max(
            _poisson_atom_decimal(endpoint, ceiling - 1)
            for endpoint in admitted
        )
        tail_lower = min(
            _poisson_tail_decimal(endpoint, ceiling) for endpoint in admitted
        )
        return predecessor_upper / tail_lower


def _mean_oracle() -> _MeanOracle:
    """Return a fixture covering c=0, three score lanes, and projection."""
    production = tuple(
        _point(value) for value in (Fraction(4), Fraction(1), 2, 1, 2)
    )
    exact = (
        (Fraction(127, 32), Fraction(129, 32)),
        (Fraction(15, 16), Fraction(17, 16)),
        (Fraction(15, 8), Fraction(17, 8)),
        (Fraction(3, 4), Fraction(5, 4)),
        (Fraction(3, 2), Fraction(5, 2)),
    )
    admitted = tuple(
        _union(point, state)
        for point, state in zip(production, exact, strict=True)
    )
    errors = tuple(
        _point_distance(point[0], hull)
        for point, hull in zip(production, admitted, strict=True)
    )
    fit_mask = (True, True, True, True, False)
    full_l1 = sum(errors, start=Fraction())
    fitted_l1 = sum(
        (
            error
            for error, fitted in zip(errors, fit_mask, strict=True)
            if fitted
        ),
        start=Fraction(),
    )
    assert errors == (
        Fraction(1, 32),
        Fraction(1, 16),
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(1, 2),
    )
    assert (full_l1, fitted_l1) == (Fraction(31, 32), Fraction(15, 32))
    return _MeanOracle(
        production=production,
        exact=exact,
        admitted=admitted,
        errors=errors,
        fit_mask=fit_mask,
        observed=(0, 0, 1, 2, 0),
        ceilings=(0, 2, 3, 2, 2),
        full_l1=full_l1,
        fitted_l1=fitted_l1,
    )


def _score_oracle(
    mean: _MeanOracle,
    *,
    rounding_errors: tuple[Fraction, ...],
) -> _ScoreOracle:
    """Return all four fitted score lanes without production helpers."""
    if len(rounding_errors) != len(mean.admitted):
        raise ValueError("rounding errors must be channel aligned")
    with localcontext() as context:
        context.prec = 100
        lipschitz = tuple(
            _score_lipschitz_decimal(hull, observed, ceiling)
            for hull, observed, ceiling in zip(
                mean.admitted, mean.observed, mean.ceilings, strict=True
            )
        )
        rounding = tuple(_decimal(value) for value in rounding_errors)
        terms = tuple(
            (constant * _decimal(error) + rho if fitted else Decimal())
            for constant, error, rho, fitted in zip(
                lipschitz,
                mean.errors,
                rounding,
                mean.fit_mask,
                strict=True,
            )
        )
        saturated_index = 3
        predecessor_upper = max(
            _poisson_atom_decimal(endpoint, mean.ceilings[saturated_index] - 1)
            for endpoint in mean.admitted[saturated_index]
        )
        tail_lower = min(
            _poisson_tail_decimal(endpoint, mean.ceilings[saturated_index])
            for endpoint in mean.admitted[saturated_index]
        )
        return _ScoreOracle(
            lipschitz=lipschitz,
            rounding=rounding,
            terms=terms,
            total=sum(terms, start=Decimal()),
            predecessor_upper=predecessor_upper,
            tail_lower=tail_lower,
        )


def _channel_nll_decimal(
    mean: Fraction,
    observed: int,
    ceiling: int,
) -> Decimal:
    """Evaluate the declared censored-Poisson symbol NLL independently."""
    if observed == ceiling:
        return (
            Decimal()
            if ceiling == 0
            else _saturated_nll_decimal(mean, ceiling)
        )
    return _unsaturated_nll_decimal(mean, observed)


def _direct_oracle(mean: _MeanOracle) -> _DirectOracle:
    """Derive the global fitted direct-NLL route from leaf means."""
    with localcontext() as context:
        context.prec = 100
        production_total = sum(
            (
                _channel_nll_decimal(point[0], observed, ceiling)
                for point, observed, ceiling, fitted in zip(
                    mean.production,
                    mean.observed,
                    mean.ceilings,
                    mean.fit_mask,
                    strict=True,
                )
                if fitted
            ),
            start=Decimal(),
        )
        admitted_channel_hulls = tuple(
            tuple(
                sorted(
                    _channel_nll_decimal(endpoint, observed, ceiling)
                    for endpoint in hull
                )
            )
            for hull, observed, ceiling, fitted in zip(
                mean.admitted,
                mean.observed,
                mean.ceilings,
                mean.fit_mask,
                strict=True,
            )
            if fitted
        )
        admitted_total = (
            sum(
                (value[0] for value in admitted_channel_hulls),
                start=Decimal(),
            ),
            sum(
                (value[1] for value in admitted_channel_hulls),
                start=Decimal(),
            ),
        )
        error = max(
            abs(production_total - admitted_total[0]),
            abs(production_total - admitted_total[1]),
        )
        return _DirectOracle(
            production_total=production_total,
            admitted_total=admitted_total,
            error=error,
        )


def _five_channel_manifest() -> GalerkinLocalResidualObservableInputManifest:
    """Return one owner-sealed, entirely parent-free five-channel manifest."""
    pixel = l9_type_tests._pixel_manifest()
    detector_manifest = create_local_censored_poisson_detector_input_manifest(
        pixel_inputs=(pixel,),
        ensemble_weight_numerators=(1,),
        ensemble_weight_denominators=(1,),
        incident_electron_count_interval=l9_type_tests._interval(1),
        incident_electron_count_point=np.asarray(1.0, dtype=np.float64),
        response_matrix=np.asarray(
            [[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64
        ),
        pre_gain_background=np.zeros(5, dtype=np.float64),
        deterministic_gain=np.ones(5, dtype=np.float64),
        electronic_offset=np.zeros(5, dtype=np.float64),
        count_ceilings=np.asarray([0, 2, 3, 2, 2], dtype=np.int64),
        fit_mask=np.asarray([True, True, True, True, False], dtype=np.bool_),
        calibration_provenance="parent-free RM-S5 arithmetic fixture",
        maximum_detector_work=10_000,
        maximum_detector_rational_bits=256,
        maximum_count_ceiling=8,
        maximum_poisson_work=10_000,
        maximum_poisson_rational_bits=256,
        exp_precision_bits=64,
        maximum_exp_terms=256,
        maximum_exp_work=10_000,
        maximum_exp_range_reductions=64,
    )
    return _make_local_residual_observable_input_manifest(
        _make_local_residual_observable_input_manifest_candidate(
            detector_input_manifest=detector_manifest,
            observed_counts=jnp.asarray([0, 0, 1, 2, 0], dtype=jnp.int64),
            maximum_detector_work=10_000,
            maximum_detector_rational_bits=256,
            log_precision_bits=64,
            maximum_log_terms=256,
            maximum_log_work=10_000,
            maximum_log_range_reductions=64,
            tv_exp_precision_bits=64,
            maximum_tv_exp_terms=256,
            maximum_tv_exp_work=10_000,
            maximum_tv_exp_range_reductions=64,
            maximum_residual_observable_work=10_000,
            maximum_residual_observable_rational_bits=1_024,
            manifest_digest="0" * 64,
        )
    )


def _expected_fixture() -> _ExpectedFixture:
    """Build one exact-class, parent-free L9 leaf shell for S5 replay."""
    mean = _mean_oracle()
    manifest = _five_channel_manifest()
    production_nll_points = (
        _carrier(_point(0)),
        _carrier(_point(1)),
        _carrier(_point(2)),
        _carrier(_point(3)),
        None,
    )
    rounding_errors = (
        Fraction(),
        Fraction(1, 1 << 10),
        Fraction(1, 1 << 9),
        Fraction(1, 1 << 8),
        Fraction(),
    )
    traces = tuple(
        _make_local_detector_real_production_trace(
            (
                _carrier(
                    (
                        point.lower - rounding_errors[channel],
                        point.upper + rounding_errors[channel],
                    )
                ),
            ),
            np.asarray(float(point.lower), dtype=np.float64),
            stage=GalerkinLocalDetectorProductionStage.CENSORED_NLL,
            quantity=f"production_nll.channel_{channel}",
            logical_shape=(),
        )
        for channel, point in enumerate(production_nll_points)
        if point is not None and channel != 0
    )
    detector = SimpleNamespace(
        response_matrix=np.zeros((5, 1), dtype=np.float64),
        production_pre_gain_mean_point_intervals=tuple(
            _carrier(value) for value in mean.production
        ),
        exact_state_pre_gain_mean_intervals=tuple(
            _carrier(value) for value in mean.exact
        ),
        fit_mask=np.asarray(mean.fit_mask, dtype=np.bool_),
        count_ceilings=np.asarray(mean.ceilings, dtype=np.int64),
        production_evidence_available=np.asarray(True, dtype=np.bool_),
        detector_eligible=np.asarray(True, dtype=np.bool_),
        likelihood_law_eligible=np.asarray(True, dtype=np.bool_),
        input_manifest_digest=(
            manifest.detector_input_manifest.manifest_digest
        ),
        certificate_digest="b" * 64,
    )
    likelihood = object.__new__(GalerkinLocalCensoredPoissonLikelihood)
    values: dict[str, object] = {
        "detector": detector,
        "observed_counts": jnp.asarray(mean.observed, dtype=jnp.int64),
        "nll_eligible": np.asarray(True, dtype=np.bool_),
        "admitted_pre_gain_mean_hull_intervals": tuple(
            _carrier(value) for value in mean.admitted
        ),
        "production_nll_point_intervals": production_nll_points,
        "total_nll_interval": _carrier((Fraction(47, 8), Fraction(25, 4))),
        "fitted_probability_positive_floor_intervals": (
            _carrier(_point(1)),
            _carrier(_point(1)),
            _carrier(_point(1)),
            _carrier(_point(Fraction(1, 8))),
            None,
        ),
        "production_traces": traces,
        "work_transcript": SimpleNamespace(
            exact_work_count=7,
            nested_parent_work_count_exact="11",
            nested_helper_work_count_exact="13",
            maximum_work=10_000,
            maximum_rational_bits=256,
        ),
        "log_precision_bits": 64,
        "maximum_log_terms": 256,
        "maximum_log_work": 10_000,
        "maximum_log_range_reductions": 64,
        "certificate_digest": "c" * 64,
    }
    for name, value in values.items():
        object.__setattr__(likelihood, name, value)
    return _ExpectedFixture(
        likelihood=likelihood,
        manifest=manifest,
        mean=mean,
        production_nll_points=production_nll_points,
        total_nll=(Fraction(47, 8), Fraction(25, 4)),
        rounding_errors=rounding_errors,
        tail_floor=Fraction(1, 8),
    )


def _successful_saturated_transcript(
    manifest: GalerkinLocalResidualObservableInputManifest,
) -> CensoredPoissonWorkTranscript:
    """Return one valid deterministic predecessor-helper transcript."""
    detector_manifest = manifest.detector_input_manifest
    prefix = EntireWorkTranscript(
        algorithm="exact_fraction_real_exp_v1",
        precision_bits=detector_manifest.exp_precision_bits,
        maximum_terms=detector_manifest.maximum_exp_terms,
        maximum_work=detector_manifest.maximum_exp_work,
        maximum_range_reductions=(
            detector_manifest.maximum_exp_range_reductions
        ),
        maximum_rational_bits=(
            detector_manifest.maximum_poisson_rational_bits
        ),
        series_terms=1,
        range_reductions=0,
        root_enclosures=0,
        rectangle_products=0,
        reciprocal_steps=0,
        exact_work_count=5,
    )
    return CensoredPoissonWorkTranscript(
        algorithm="exact_fraction_censored_poisson_probability_v1",
        maximum_count_ceiling=detector_manifest.maximum_count_ceiling,
        maximum_work=detector_manifest.maximum_poisson_work,
        maximum_rational_bits=(
            detector_manifest.maximum_poisson_rational_bits
        ),
        exp_precision_bits=detector_manifest.exp_precision_bits,
        maximum_exp_terms=detector_manifest.maximum_exp_terms,
        maximum_exp_work=detector_manifest.maximum_exp_work,
        maximum_exp_range_reductions=(
            detector_manifest.maximum_exp_range_reductions
        ),
        log_precision_bits=0,
        maximum_log_terms=0,
        maximum_log_work=0,
        maximum_log_range_reductions=0,
        count_ceiling=2,
        observed_count=1,
        polynomial_terms=2,
        endpoint_evaluations=2,
        critical_point_evaluations=0,
        direct_tail_lower_evaluations=0,
        exact_work_count=3,
        exp_transcripts=(prefix,),
        log_transcripts=(),
    )


def _expected_evidence(
    fixture: _ExpectedFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    predecessor: _Interval = (Fraction(1, 5), Fraction(1, 4)),
    poisson_error: CensoredPoissonEnclosureError | None = None,
) -> tuple[dict[str, Any], list[tuple[object, ...]]]:
    """Replay S5 through isolated exact-parent seams and capture the helper."""
    transcript = _successful_saturated_transcript(fixture.manifest)
    captured: list[tuple[object, ...]] = []

    def probability(
        interval: _Interval,
        observed: int,
        ceiling: int,
        **policies: object,
    ) -> tuple[_Interval, CensoredPoissonWorkTranscript]:
        captured.append((interval, observed, ceiling, policies))
        if poisson_error is not None:
            raise poisson_error
        return predecessor, transcript

    monkeypatch.setattr(
        residual_types,
        "_validate_local_censored_poisson_likelihood",
        lambda value: value,
    )
    monkeypatch.setattr(
        residual_types, "_state_authority_available", lambda _value: True
    )
    monkeypatch.setattr(
        residual_types, "enclose_censored_poisson_probability", probability
    )
    replay = inspect.unwrap(
        residual_types._expected_local_residual_observable_evidence
    )
    evidence = cast(
        dict[str, Any], replay(fixture.likelihood, fixture.manifest)
    )
    return evidence, captured


def _certificate_candidate(
    fixture: _ExpectedFixture,
    evidence: dict[str, Any],
):
    """Lift replayed evidence into one exact unsealed certificate shell."""
    flags = {
        name: jnp.asarray(evidence[name], dtype=jnp.bool_)
        for name in (
            "state_evidence_available",
            "mean_evidence_available",
            "law_evidence_available",
            "full_law_evidence_available",
            "fitted_law_evidence_available",
            "direct_nll_evidence_available",
            "score_nll_evidence_available",
            "selected_nll_evidence_available",
        )
    }
    derived = {
        name: value
        for name, value in evidence.items()
        if name not in (*flags, "failure_mask")
    }
    detector = fixture.likelihood.detector
    return _make_local_residual_observable_certificate_candidate(
        parent_likelihood=fixture.likelihood,
        input_manifest=fixture.manifest,
        **flags,
        failure_mask=jnp.asarray(
            int(evidence["failure_mask"]), dtype=jnp.int64
        ),
        **derived,
        full_law_scope=(
            residual_types.GalerkinLocalResidualObservableScope.FULL_CHANNEL_LAW
        ),
        fitted_law_scope=(
            residual_types.GalerkinLocalResidualObservableScope.FIXED_FIT_PROJECTION
        ),
        mean_scope=_MEAN_SCOPE,
        law_scope=_LAW_SCOPE,
        nll_scope=_NLL_SCOPE,
        resource_scope=_RESOURCE_SCOPE,
        no_scientific_claim_scope=_NO_SCIENTIFIC_CLAIM_SCOPE,
        parent_detector_certificate_digest=detector.certificate_digest,
        parent_detector_input_manifest_digest=detector.input_manifest_digest,
        parent_likelihood_certificate_digest=(
            fixture.likelihood.certificate_digest
        ),
        input_manifest_digest=fixture.manifest.manifest_digest,
        observable_identity_digest="0" * 64,
        observable_evidence_digest="0" * 64,
        certificate_digest="0" * 64,
    )


def test_mean_evidence_uses_detector_leaves_and_projects_only_after_full_sum() -> (  # noqa: E501
    None
):
    """Reconstruct H, d, and full/fitted L1 from detector leaves only."""
    oracle = _mean_oracle()
    detector = SimpleNamespace(
        response_matrix=np.zeros((5, 1), dtype=np.float64),
        production_pre_gain_mean_point_intervals=tuple(
            _carrier(value) for value in oracle.production
        ),
        exact_state_pre_gain_mean_intervals=tuple(
            _carrier(value) for value in oracle.exact
        ),
        fit_mask=np.asarray(oracle.fit_mask, dtype=np.bool_),
    )
    parent = object.__new__(GalerkinLocalCensoredPoissonLikelihood)
    object.__setattr__(parent, "detector", detector)
    evidence = _derived_detector_mean_evidence(parent)
    assert evidence is not None
    admitted, errors, full, fitted = evidence
    assert tuple(_raw(value) for value in admitted) == oracle.admitted
    assert tuple(_raw(value) for value in errors) == tuple(
        _point(value) for value in oracle.errors
    )
    assert _raw(full) == _point(Fraction(31, 32))
    assert _raw(fitted) == _point(Fraction(15, 32))
    assert full.lower != fitted.lower


def test_state_authority_traverses_l6_radius_and_every_provenance_edge() -> (
    None
):
    """Require projection, L6, radius, port, and pixel evidence per mode."""
    projection_digest = "a" * 64
    port_digest = "b" * 64
    pixel_digest = "c" * 64

    def authority(
        *,
        projection_eligible: bool = True,
        radius_eligible: bool = True,
        radius_present: bool = True,
        stored_radius: GalerkinLocalDetectorRationalInterval | None = None,
        stored_projection_digest: str = projection_digest,
        stored_port_digest: str = port_digest,
        stored_pixel_digest: str = pixel_digest,
    ) -> bool:
        projection = SimpleNamespace(
            finite_projection_bound_eligible=np.asarray(
                projection_eligible, dtype=np.bool_
            ),
            state_radius_upper_bound=np.asarray(0.375, dtype=np.float64),
            stability_result=SimpleNamespace(
                proof=SimpleNamespace(
                    state_radius_eligible=np.asarray(
                        radius_eligible, dtype=np.bool_
                    )
                )
            ),
        )
        terminal = SimpleNamespace(
            projection_certificate=projection,
            parent_projection_certificate_digest=projection_digest,
        )
        port = SimpleNamespace(
            terminal_certificate=terminal,
            certificate_digest=port_digest,
        )
        pixel = SimpleNamespace(
            positive_port=port,
            pixel_model_evidence_digest=pixel_digest,
        )
        detector = SimpleNamespace(
            pixel_forms=(pixel,),
            mode_state_radius_intervals=(
                (
                    _carrier(_point(Fraction(3, 8)))
                    if stored_radius is None
                    else stored_radius
                )
                if radius_present
                else None,
            ),
            mode_state_radius_provenance_digests=(stored_projection_digest,),
            mode_port_certificate_digests=(stored_port_digest,),
            mode_pixel_evidence_digests=(stored_pixel_digest,),
        )
        parent = SimpleNamespace(detector=detector)
        return bool(inspect.unwrap(_state_authority_available)(parent))

    assert authority()
    assert not authority(projection_eligible=False)
    assert not authority(radius_eligible=False)
    assert not authority(radius_present=False)
    assert not authority(stored_radius=_carrier(_point(Fraction(1, 2))))
    assert not authority(stored_projection_digest="d" * 64)
    assert not authority(stored_port_digest="d" * 64)
    assert not authority(stored_pixel_digest="d" * 64)


def test_tv_bounds_keep_linear_full_and_fit_lanes_when_exponential_fails() -> (
    None
):
    """Check 1-exp(-eta), projection, and linear fallback independently."""
    manifest = _five_channel_manifest()
    mean = _mean_oracle()
    full = _tv_bound_evidence(
        mean.full_l1,
        call=GalerkinLocalResidualObservableHelperCall.FULL_TV_EXPONENTIAL,
        manifest=manifest,
    )
    fitted = _tv_bound_evidence(
        mean.fitted_l1,
        call=GalerkinLocalResidualObservableHelperCall.FITTED_TV_EXPONENTIAL,
        manifest=manifest,
    )
    for eta, result in ((mean.full_l1, full), (mean.fitted_l1, fitted)):
        linear, exponential, selected, transcript, failure = result
        assert _raw(linear) == _point(min(Fraction(1), eta))
        assert exponential is not None
        expected = _one_minus_exp_neg_decimal(eta)
        lower, upper = (_decimal(value) for value in _raw(exponential))
        assert lower <= expected <= upper
        assert selected is exponential
        assert transcript is not None and failure is None

    failed_manifest = _make_local_residual_observable_input_manifest(
        dataclasses.replace(manifest, maximum_tv_exp_terms=1)
    )
    failed = _tv_bound_evidence(
        mean.fitted_l1,
        call=GalerkinLocalResidualObservableHelperCall.FITTED_TV_EXPONENTIAL,
        manifest=failed_manifest,
    )
    linear, exponential, selected, transcript, failure = failed
    assert _raw(linear) == _point(Fraction(15, 32))
    assert exponential is None and transcript is None
    assert selected is linear
    assert failure is not None
    assert failure.call is (
        GalerkinLocalResidualObservableHelperCall.FITTED_TV_EXPONENTIAL
    )


def test_score_rounding_rho_recomputes_unique_raw_nll_endpoint_gap() -> None:
    """Bind rho to one raw CENSORED_NLL trace, not its stored error alone."""
    raw = (_carrier((Fraction(1), Fraction(5, 4))),)
    point = _carrier(_point(Fraction(9, 8)))
    trace = _make_local_detector_real_production_trace(
        raw,
        np.asarray(1.125, dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.CENSORED_NLL,
        quantity="production_nll.channel_2",
        logical_shape=(),
    )
    parent = object.__new__(GalerkinLocalCensoredPoissonLikelihood)
    object.__setattr__(parent, "production_traces", (trace,))
    assert _trace_rounding_error(parent, 2, point) == Fraction(1, 8)
    assert Fraction.from_float(
        float(np.asarray(trace.point_to_raw_absolute_error_upper_bounds)[0])
    ) == Fraction(1, 8)
    object.__setattr__(parent, "production_traces", (trace, trace))
    with pytest.raises(ValueError, match="not unique"):
        _trace_rounding_error(parent, 2, point)
    object.__setattr__(parent, "production_traces", (trace,))
    with pytest.raises(ValueError, match="point binding"):
        _trace_rounding_error(parent, 2, _carrier(_point(Fraction(1))))


def test_direct_and_score_oracles_are_global_cover_all_symbols_and_select_once() -> (  # noqa: E501
    None
):
    """Freeze global direct/score sums, all score classes, and route choice."""
    mean = _mean_oracle()
    direct = _direct_oracle(mean)
    rounding = (
        Fraction(0),
        Fraction(1, 1_000),
        Fraction(1, 500),
        Fraction(1, 250),
        Fraction(1, 125),
    )
    score = _score_oracle(mean, rounding_errors=rounding)

    assert direct.admitted_total[0] < direct.production_total
    assert direct.production_total < direct.admitted_total[1]
    with localcontext() as context:
        context.prec = 100
        assert direct.error == max(
            direct.production_total - direct.admitted_total[0],
            direct.admitted_total[1] - direct.production_total,
        )
    assert score.lipschitz[0] == 0  # fitted c=0 branch
    assert score.terms[0] == 0 and score.rounding[0] == 0
    assert score.lipschitz[1] == 1  # fitted y=0<c branch
    with localcontext() as context:
        context.prec = 100
        assert score.lipschitz[2] == max(
            abs(Decimal(1) - Decimal(1) / _decimal(endpoint))
            for endpoint in mean.admitted[2]
        )
        assert score.lipschitz[3] == (
            score.predecessor_upper / score.tail_lower
        )
        assert score.total == sum(score.terms, start=Decimal())
    assert score.terms[4] == 0  # excluded nonzero-error channel
    assert direct.error < score.total
    selected = min(direct.error, score.total)
    assert selected == direct.error

    tied_score = direct.error
    assert min(direct.error, tied_score) == direct.error
    assert direct.error == tied_score


def test_work_plan_counts_all_score_symbol_classes_and_excludes_masked_channel() -> (  # noqa: E501
    None
):
    """Freeze literal r/f/z/interior/saturated counts and four work stages."""
    assert _work_plan_counts(_five_channel_manifest()) == (
        5,
        4,
        1,
        1,
        1,
        19,
        23,
        29,
        50,
    )


def test_expected_evidence_binds_global_routes_helpers_and_exact_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit the complete success DAG from independent leaves and literals."""
    fixture = _expected_fixture()
    evidence, captured = _expected_evidence(fixture, monkeypatch)
    mean = fixture.mean

    assert (
        evidence["state_evidence_available"],
        evidence["mean_evidence_available"],
        evidence["law_evidence_available"],
        evidence["full_law_evidence_available"],
        evidence["fitted_law_evidence_available"],
        evidence["direct_nll_evidence_available"],
        evidence["score_nll_evidence_available"],
        evidence["selected_nll_evidence_available"],
    ) == (True,) * 8
    assert evidence["failure_mask"] is (
        GalerkinLocalResidualObservableFailure.NONE
    )
    assert evidence["strongest_layer"] is (
        GalerkinLocalResidualObservableLayer.POINTWISE_NLL
    )
    assert evidence["selected_nll_route"] is (
        GalerkinLocalResidualObservableRoute.DIRECT_ADMITTED_HULL
    )

    assert (
        tuple(
            _raw(value)
            for value in evidence["admitted_pre_gain_mean_hull_intervals"]
        )
        == mean.admitted
    )
    assert tuple(
        _raw(value) for value in evidence["channel_mean_error_bound_intervals"]
    ) == tuple(_point(value) for value in mean.errors)
    assert _raw(evidence["full_mean_l1_error_bound_interval"]) == _point(
        Fraction(31, 32)
    )
    assert _raw(evidence["fitted_mean_l1_error_bound_interval"]) == _point(
        Fraction(15, 32)
    )
    assert _raw(evidence["full_linear_tv_bound_interval"]) == _point(
        Fraction(31, 32)
    )
    assert _raw(evidence["fitted_linear_tv_bound_interval"]) == _point(
        Fraction(15, 32)
    )
    for eta, name in (
        (mean.full_l1, "full_exponential_tv_bound_interval"),
        (mean.fitted_l1, "fitted_exponential_tv_bound_interval"),
    ):
        interval = evidence[name]
        truth = _one_minus_exp_neg_decimal(eta)
        lower, upper = (_decimal(value) for value in _raw(interval))
        assert lower <= truth <= upper

    assert _raw(evidence["production_fitted_total_nll_interval"]) == _point(6)
    assert _raw(evidence["direct_nll_error_bound_interval"]) == _point(
        Fraction(1, 4)
    )
    factors = (
        Fraction(),
        Fraction(1),
        Fraction(9, 17),
        Fraction(2),
        None,
    )
    rounding = (*fixture.rounding_errors[:4], None)
    terms = (
        Fraction(),
        Fraction(1, 16) + Fraction(1, 1 << 10),
        Fraction(9, 17) * Fraction(1, 8) + Fraction(1, 1 << 9),
        Fraction(2) * Fraction(1, 4) + Fraction(1, 1 << 8),
        None,
    )
    for name, expected in (
        ("score_lipschitz_factor_intervals", factors),
        ("score_rounding_error_intervals", rounding),
        ("score_term_error_intervals", terms),
    ):
        assert (
            tuple(
                None if value is None else value.lower
                for value in evidence[name]
            )
            == expected
        )
        assert all(
            value is None or value.lower == value.upper
            for value in evidence[name]
        )
    score_total = sum(
        (value for value in terms if value is not None), start=Fraction()
    )
    assert _raw(evidence["score_nll_error_bound_interval"]) == _point(
        score_total
    )
    assert score_total > Fraction(1, 4)
    assert _raw(evidence["selected_nll_error_bound_interval"]) == _point(
        Fraction(1, 4)
    )
    assert tuple(
        None if value is None else _raw(value)
        for value in evidence["saturated_predecessor_mass_upper_intervals"]
    ) == (None, None, None, _point(Fraction(1, 4)), None)
    assert tuple(
        None if value is None else _raw(value)
        for value in evidence["saturated_tail_probability_floor_intervals"]
    ) == (None, None, None, _point(Fraction(1, 8)), None)

    detector_manifest = fixture.manifest.detector_input_manifest
    assert captured == [
        (
            mean.admitted[3],
            1,
            2,
            {
                "maximum_count_ceiling": (
                    detector_manifest.maximum_count_ceiling
                ),
                "maximum_work": detector_manifest.maximum_poisson_work,
                "maximum_rational_bits": (
                    detector_manifest.maximum_poisson_rational_bits
                ),
                "exp_precision_bits": detector_manifest.exp_precision_bits,
                "maximum_exp_terms": detector_manifest.maximum_exp_terms,
                "maximum_exp_work": detector_manifest.maximum_exp_work,
                "maximum_exp_range_reductions": (
                    detector_manifest.maximum_exp_range_reductions
                ),
            },
        )
    ]
    saturated = evidence["saturated_probability_transcripts"][3]
    assert saturated == _successful_saturated_transcript(fixture.manifest)
    full_exp = evidence["full_tv_exp_transcript"]
    fitted_exp = evidence["fitted_tv_exp_transcript"]
    assert full_exp is not None and fitted_exp is not None
    helper_work = (
        full_exp.exact_work_count
        + fitted_exp.exact_work_count
        + saturated.exact_work_count
        + sum(
            value.exact_work_count
            for value in saturated.exp_transcripts + saturated.log_transcripts
        )
    )
    work = evidence["work_transcript"]
    assert (
        work.channel_count,
        work.fitted_channel_count,
        work.zero_observation_positive_ceiling_count,
        work.interior_observation_count,
        work.saturated_positive_ceiling_count,
    ) == (5, 4, 1, 1, 1)
    assert (
        work.mean_exact_work_count,
        work.law_exact_work_count,
        work.direct_nll_exact_work_count,
        work.score_nll_exact_work_count,
        work.exact_work_count,
    ) == (19, 23, 29, 50, 50)
    assert (
        int(work.nested_parent_work_count_exact),
        int(work.nested_helper_work_count_exact),
    ) == (31, helper_work)
    assert work.completed_successfully and not work.preflight_failed
    assert work.completed_layer is (
        GalerkinLocalResidualObservableLayer.POINTWISE_NLL
    )


def test_law_survives_outer_nll_stop_and_preflight_stops_before_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep detector-law evidence while outer NLL and preflight stop."""
    fixture = _expected_fixture()
    object.__setattr__(
        fixture.likelihood, "nll_eligible", np.asarray(False, dtype=np.bool_)
    )
    law, calls = _expected_evidence(fixture, monkeypatch)
    assert calls == []
    assert (
        law["state_evidence_available"],
        law["mean_evidence_available"],
        law["law_evidence_available"],
        law["direct_nll_evidence_available"],
        law["score_nll_evidence_available"],
        law["selected_nll_evidence_available"],
    ) == (True, True, True, False, False, False)
    assert law["failure_mask"] == (
        GalerkinLocalResidualObservableFailure.DIRECT_NLL_UNAVAILABLE
    )
    assert law["strongest_layer"] is GalerkinLocalResidualObservableLayer.LAW
    assert law["selected_nll_route"] is (
        GalerkinLocalResidualObservableRoute.UNAVAILABLE
    )
    assert law["full_selected_tv_bound_interval"] is not None
    assert law["fitted_selected_tv_bound_interval"] is not None
    assert law["production_fitted_total_nll_interval"] is None
    assert law["score_nll_error_bound_interval"] is None
    assert law["score_lipschitz_factor_intervals"] == (None,) * 5
    law_work = law["work_transcript"]
    assert (
        law_work.mean_exact_work_count,
        law_work.law_exact_work_count,
        law_work.direct_nll_exact_work_count,
        law_work.score_nll_exact_work_count,
        law_work.exact_work_count,
        law_work.attempted_exact_work_count_exact,
    ) == (19, 23, 23, 23, 23, "23")
    assert not law_work.preflight_failed
    assert law_work.completed_layer is GalerkinLocalResidualObservableLayer.LAW

    budget_fixture = _expected_fixture()
    budget_manifest = _make_local_residual_observable_input_manifest(
        dataclasses.replace(
            budget_fixture.manifest,
            maximum_residual_observable_work=49,
        )
    )
    budget_fixture = budget_fixture._replace(manifest=budget_manifest)
    budget, budget_calls = _expected_evidence(budget_fixture, monkeypatch)
    assert budget_calls == []
    assert budget["state_evidence_available"]
    assert not budget["mean_evidence_available"]
    assert budget["failure_mask"] == (
        GalerkinLocalResidualObservableFailure.EXACT_WORK_BUDGET_EXCEEDED
    )
    assert budget["strongest_layer"] is (
        GalerkinLocalResidualObservableLayer.STATE
    )
    assert budget["admitted_pre_gain_mean_hull_intervals"] is None
    assert budget["full_selected_tv_bound_interval"] is None
    assert budget["production_fitted_total_nll_interval"] is None
    assert budget["score_lipschitz_factor_intervals"] == (None,) * 5
    budget_work = budget["work_transcript"]
    assert (
        budget_work.mean_exact_work_count,
        budget_work.law_exact_work_count,
        budget_work.direct_nll_exact_work_count,
        budget_work.score_nll_exact_work_count,
        budget_work.exact_work_count,
        budget_work.rational_peak_bits,
        budget_work.attempted_exact_work_count_exact,
    ) == (0, 0, 0, 0, 0, 0, "50")
    assert budget_work.preflight_failed and not budget_work.count_overflow


def test_rational_preflight_and_score_prerequisite_stops_retain_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin zero-work bit stops and direct-retaining score prepasses."""
    over_bits = _expected_fixture()
    detector = over_bits.likelihood.detector
    detector.exact_state_pre_gain_mean_intervals = (
        _carrier(_point(Fraction(1, 1 << 1_100))),
        *detector.exact_state_pre_gain_mean_intervals[1:],
    )
    bit_stop, bit_calls = _expected_evidence(over_bits, monkeypatch)
    assert bit_calls == []
    assert bit_stop["state_evidence_available"]
    assert not bit_stop["mean_evidence_available"]
    assert bit_stop["failure_mask"] == (
        GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
    )
    bit_work = bit_stop["work_transcript"]
    assert (
        bit_work.exact_work_count,
        bit_work.rational_peak_bits,
        bit_work.attempted_exact_work_count_exact,
    ) == (0, 1_101, "0")
    assert not bit_work.preflight_failed

    interior = _expected_fixture()
    interior_manifest = _make_local_residual_observable_input_manifest(
        dataclasses.replace(
            interior.manifest,
            maximum_residual_observable_rational_bits=4_096,
        )
    )
    interior = interior._replace(manifest=interior_manifest)
    interior.likelihood.detector.exact_state_pre_gain_mean_intervals = (
        *interior.likelihood.detector.exact_state_pre_gain_mean_intervals[:2],
        _carrier((Fraction(), Fraction(17, 8))),
        *interior.likelihood.detector.exact_state_pre_gain_mean_intervals[3:],
    )
    object.__setattr__(
        interior.likelihood,
        "admitted_pre_gain_mean_hull_intervals",
        (
            *interior.likelihood.admitted_pre_gain_mean_hull_intervals[:2],
            _carrier((Fraction(), Fraction(17, 8))),
            *interior.likelihood.admitted_pre_gain_mean_hull_intervals[3:],
        ),
    )
    interior_stop, interior_calls = _expected_evidence(interior, monkeypatch)
    assert interior_calls == []
    assert interior_stop["failure_mask"] == (
        GalerkinLocalResidualObservableFailure.SCORE_MEAN_FLOOR_UNAVAILABLE
    )
    assert interior_stop["direct_nll_evidence_available"]
    assert not interior_stop["score_nll_evidence_available"]
    assert interior_stop["selected_nll_route"] is (
        GalerkinLocalResidualObservableRoute.DIRECT_ADMITTED_HULL
    )
    assert interior_stop["score_lipschitz_factor_intervals"] == (None,) * 5
    assert interior_stop["saturated_probability_transcripts"] == (None,) * 5
    assert interior_stop["saturated_probability_failures"] == (None,) * 5
    interior_work = interior_stop["work_transcript"]
    assert (
        interior_work.mean_exact_work_count,
        interior_work.law_exact_work_count,
        interior_work.direct_nll_exact_work_count,
        interior_work.score_nll_exact_work_count,
        interior_work.exact_work_count,
        interior_work.attempted_exact_work_count_exact,
    ) == (19, 23, 29, 29, 29, "50")

    tail = _expected_fixture()
    object.__setattr__(
        tail.likelihood,
        "fitted_probability_positive_floor_intervals",
        (_carrier(_point(1)),) * 3 + (None, None),
    )
    tail_stop, tail_calls = _expected_evidence(tail, monkeypatch)
    assert tail_calls == []
    assert tail_stop["failure_mask"] == (
        GalerkinLocalResidualObservableFailure.SCORE_PROBABILITY_FLOOR_UNAVAILABLE
    )
    assert tail_stop["direct_nll_evidence_available"]
    assert not tail_stop["score_nll_evidence_available"]
    assert (
        tail_stop["saturated_tail_probability_floor_intervals"] == (None,) * 5
    )
    tail_work = tail_stop["work_transcript"]
    assert (
        tail_work.exact_work_count,
        tail_work.attempted_exact_work_count_exact,
    ) == (29, "50")


def test_saturated_helper_failure_preserves_direct_and_nested_prefix_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain direct evidence and successful helper prefixes on failure."""
    fixture = _expected_fixture()
    prefix = _successful_saturated_transcript(
        fixture.manifest
    ).exp_transcripts[0]
    error = CensoredPoissonEnclosureError(
        CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE,
        3,
        "forced predecessor exponential failure",
        attempted_exact_work_count=3,
        nested_kernel="exp",
        nested_failure=EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
        nested_exact_work_count=2,
        nested_attempted_exact_work_count=2,
        prior_exp_transcripts=(prefix,),
        prior_log_transcripts=(),
    )
    stopped, calls = _expected_evidence(
        fixture, monkeypatch, poisson_error=error
    )
    assert len(calls) == 1
    assert stopped["failure_mask"] == (
        GalerkinLocalResidualObservableFailure.POISSON_ENCLOSURE_FAILURE
        | GalerkinLocalResidualObservableFailure.NESTED_HELPER_FAILURE
    )
    assert stopped["direct_nll_evidence_available"]
    assert not stopped["score_nll_evidence_available"]
    assert stopped["selected_nll_route"] is (
        GalerkinLocalResidualObservableRoute.DIRECT_ADMITTED_HULL
    )
    assert stopped["saturated_predecessor_mass_upper_intervals"] == (None,) * 5
    assert stopped["saturated_probability_transcripts"] == (None,) * 5
    failure = stopped["saturated_probability_failures"][3]
    assert failure is not None
    assert failure.call is (
        GalerkinLocalResidualObservableHelperCall.SATURATED_SCORE_MASS
    )
    assert failure.channel_index == 3
    assert failure.prior_exp_transcripts == (prefix,)
    assert failure.prior_log_transcripts == ()
    assert (
        failure.local_exact_work_count_exact,
        failure.nested_exact_work_count_exact,
        failure.nested_attempted_exact_work_count_exact,
        failure.planned_exact_work_count_exact,
        failure.attempted_exact_work_count_exact,
    ) == ("3", "2", "2", "3", "3")
    full_exp = stopped["full_tv_exp_transcript"]
    fitted_exp = stopped["fitted_tv_exp_transcript"]
    assert full_exp is not None and fitted_exp is not None
    expected_helper_work = (
        full_exp.exact_work_count
        + fitted_exp.exact_work_count
        + 3
        + prefix.exact_work_count
        + 2
    )
    work = stopped["work_transcript"]
    assert (
        work.exact_work_count,
        work.attempted_exact_work_count_exact,
        int(work.nested_helper_work_count_exact),
    ) == (29, "50", expected_helper_work)
    assert work.completed_layer is (
        GalerkinLocalResidualObservableLayer.POINTWISE_NLL
    )


def test_certificate_owner_rejects_derived_and_helper_transcript_forgeries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-compare every derived lane and authenticated helper resource."""
    fixture = _expected_fixture()
    evidence, _ = _expected_evidence(fixture, monkeypatch)
    monkeypatch.setattr(
        residual_types,
        "_certificate_digests",
        lambda _value: ("d" * 64, "e" * 64, "f" * 64),
    )
    sealed = _make_local_residual_observable_certificate(
        _certificate_candidate(fixture, evidence)
    )
    assert sealed.observable_identity_digest == "d" * 64
    assert sealed.observable_evidence_digest == "e" * 64
    assert sealed.certificate_digest == "f" * 64

    point = _carrier(_point(Fraction(99)))
    mean_errors = sealed.channel_mean_error_bound_intervals
    assert mean_errors is not None
    derived_forgeries = (
        dataclasses.replace(
            sealed,
            mean_evidence_available=jnp.asarray(False, dtype=jnp.bool_),
        ),
        dataclasses.replace(
            sealed,
            channel_mean_error_bound_intervals=(
                point,
                *mean_errors[1:],
            ),
        ),
        dataclasses.replace(sealed, direct_nll_error_bound_interval=point),
        dataclasses.replace(
            sealed,
            score_lipschitz_factor_intervals=(
                point,
                *sealed.score_lipschitz_factor_intervals[1:],
            ),
        ),
        dataclasses.replace(
            sealed,
            failure_mask=jnp.asarray(
                int(
                    GalerkinLocalResidualObservableFailure.RATIONAL_SIZE_LIMIT
                ),
                dtype=jnp.int64,
            ),
        ),
        dataclasses.replace(
            sealed,
            strongest_layer=GalerkinLocalResidualObservableLayer.LAW,
        ),
        dataclasses.replace(
            sealed,
            selected_nll_route=GalerkinLocalResidualObservableRoute.TIED,
        ),
        dataclasses.replace(
            sealed,
            work_transcript=dataclasses.replace(
                sealed.work_transcript,
                nested_helper_work_count_exact="0",
            ),
        ),
    )
    for forged in derived_forgeries:
        with pytest.raises(ValueError):
            _make_local_residual_observable_certificate(forged)

    transcript = sealed.saturated_probability_transcripts[3]
    assert transcript is not None
    transcript_forgeries = (
        dataclasses.replace(transcript, algorithm="wrong"),
        dataclasses.replace(
            transcript,
            maximum_count_ceiling=transcript.maximum_count_ceiling + 1,
        ),
        dataclasses.replace(
            transcript,
            maximum_work=transcript.maximum_work + 1,
        ),
        dataclasses.replace(
            transcript,
            maximum_rational_bits=transcript.maximum_rational_bits + 1,
        ),
        dataclasses.replace(
            transcript, exp_precision_bits=transcript.exp_precision_bits + 1
        ),
        dataclasses.replace(
            transcript, maximum_exp_terms=transcript.maximum_exp_terms + 1
        ),
        dataclasses.replace(
            transcript, maximum_exp_work=transcript.maximum_exp_work + 1
        ),
        dataclasses.replace(
            transcript,
            maximum_exp_range_reductions=(
                transcript.maximum_exp_range_reductions + 1
            ),
        ),
        dataclasses.replace(transcript, log_precision_bits=1),
        dataclasses.replace(transcript, maximum_log_terms=1),
        dataclasses.replace(transcript, maximum_log_work=1),
        dataclasses.replace(transcript, maximum_log_range_reductions=1),
        dataclasses.replace(transcript, observed_count=0),
        dataclasses.replace(transcript, count_ceiling=3),
        dataclasses.replace(
            transcript, polynomial_terms=transcript.polynomial_terms + 1
        ),
        dataclasses.replace(
            transcript,
            endpoint_evaluations=transcript.endpoint_evaluations + 1,
        ),
        dataclasses.replace(
            transcript,
            critical_point_evaluations=(
                transcript.critical_point_evaluations + 1
            ),
        ),
        dataclasses.replace(
            transcript,
            direct_tail_lower_evaluations=(
                transcript.direct_tail_lower_evaluations + 1
            ),
        ),
        dataclasses.replace(
            transcript, exact_work_count=transcript.exact_work_count + 1
        ),
        dataclasses.replace(
            transcript,
            exp_transcripts=(
                dataclasses.replace(
                    transcript.exp_transcripts[0], algorithm="wrong"
                ),
            ),
        ),
        dataclasses.replace(
            transcript,
            exp_transcripts=(
                dataclasses.replace(
                    transcript.exp_transcripts[0],
                    maximum_work=(
                        transcript.exp_transcripts[0].maximum_work + 1
                    ),
                ),
            ),
        ),
    )
    for forged_transcript in transcript_forgeries:
        forged = dataclasses.replace(
            sealed,
            saturated_probability_transcripts=(
                *sealed.saturated_probability_transcripts[:3],
                forged_transcript,
                *sealed.saturated_probability_transcripts[4:],
            ),
        )
        with pytest.raises(ValueError):
            _make_local_residual_observable_certificate(forged)


@functools.lru_cache(maxsize=1)
def _cached_parent_likelihood() -> GalerkinLocalCensoredPoissonLikelihood:
    """Return the genuine cached L9 parent only when a heavy node asks."""
    from tests.test_ptyrodactyl.test_galerkin import (  # noqa: PLC0415
        test_detector as l9_tests,
    )

    return l9_tests._public_chain().likelihood


__all__: list[str] = []
