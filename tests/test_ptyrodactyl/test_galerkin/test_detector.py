r"""Lightweight falsification tests for the local RM-S4 detector arithmetic."""

from __future__ import annotations

import functools
import inspect
from dataclasses import fields, replace
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, NamedTuple, TypeVar, cast

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

import ptyrodactyl.galerkin.detector as detector
import ptyrodactyl.types.local_detector_types as local_types
from ptyrodactyl._tools import (
    CensoredPoissonEnclosureError,
    CensoredPoissonEnclosureFailure,
    EntireEnclosureFailure,
    EntireWorkTranscript,
    enclose_censored_poisson_mean,
    enclose_censored_poisson_nll,
    enclose_censored_poisson_probability,
    sha256,
    stored_value_payload,
)
from ptyrodactyl.types.local_detector_types import (
    GalerkinLocalCensoredPoissonDetector,
    GalerkinLocalCensoredPoissonDetectorInputManifest,
    GalerkinLocalCensoredPoissonLikelihood,
    GalerkinLocalDetectorCoordinateConvention,
    GalerkinLocalDetectorFailure,
    GalerkinLocalDetectorHelperCall,
    GalerkinLocalDetectorProductionStage,
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalPassivePixelForms,
    GalerkinLocalPassivePixelInputManifest,
    GalerkinLocalPositivePortBranchDisposition,
    GalerkinLocalPositivePortCertificate,
    GalerkinLocalPositivePortRoute,
    _expected_carrier_digests,
    _make_local_censored_poisson_detector,
    _make_local_censored_poisson_detector_input_manifest,
    _make_local_censored_poisson_likelihood,
    _make_local_detector_rational_interval,
    _make_local_detector_real_production_trace,
    _make_local_detector_work_transcript,
    _make_local_passive_pixel_forms,
    _make_local_passive_pixel_forms_candidate,
    _make_local_passive_pixel_input_manifest,
    _make_local_positive_port_certificate,
)
from ptyrodactyl.types.local_vacuum_propagation_types import (
    GalerkinLocalVacuumRootClass,
)
from ptyrodactyl.types.local_vacuum_terminal_types import (
    GalerkinLocalVacuumHalfSpaceDisposition,
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
)
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_vacuum_terminal as l8_tests,
)

type _Interval = tuple[Fraction, Fraction]


class _PublicChain(NamedTuple):
    """Cache one genuine six-level scalar L9 replay chain."""

    terminal: GalerkinLocalVacuumTerminalCertificate
    port: GalerkinLocalPositivePortCertificate
    pixel_input: GalerkinLocalPassivePixelInputManifest
    pixel: GalerkinLocalPassivePixelForms
    detector_input: GalerkinLocalCensoredPoissonDetectorInputManifest
    detector_certificate: GalerkinLocalCensoredPoissonDetector
    likelihood: GalerkinLocalCensoredPoissonLikelihood


class _IndependentPixelOracle(NamedTuple):
    """Store test-owned pixel values rebuilt only from L8/input leaves."""

    current_raw: tuple[_Interval, ...]
    current: tuple[_Interval, ...]
    current_points: np.ndarray
    scale_raw: _Interval
    scale: _Interval
    scale_point: np.float64
    jacobians_raw: tuple[_Interval, ...]
    jacobians: tuple[_Interval, ...]
    jacobian_points: np.ndarray
    quadrature_raw: tuple[_Interval, ...]
    quadrature: tuple[_Interval, ...]
    aperture_raw: tuple[_Interval, ...]
    aperture: tuple[_Interval, ...]
    outward_raw: tuple[_Interval, ...]
    outward: tuple[_Interval, ...]
    outward_points: np.ndarray
    pixels_raw: tuple[tuple[_Interval, ...], ...]
    pixels: tuple[tuple[_Interval, ...], ...]
    pixel_points: np.ndarray
    margin_raw: tuple[_Interval, ...]
    margin: tuple[_Interval, ...]
    margin_points: np.ndarray
    amplitude_squared_raw: tuple[_Interval, ...]
    amplitude_squared: tuple[_Interval, ...]
    amplitude_squared_points: np.ndarray
    production_raw: tuple[_Interval, ...]
    production: tuple[_Interval, ...]
    production_points: np.ndarray
    form_norms: tuple[Fraction, ...]
    production_error: Fraction
    state_error: Fraction
    total_error: Fraction
    amplitude_norm: Fraction
    production_realization_errors: tuple[Fraction, ...]
    incremental_errors: tuple[Fraction, ...]
    combined_errors: tuple[Fraction, ...]
    exact_state: tuple[_Interval, ...]


_T = TypeVar("_T")

_ANGULAR = getattr(
    GalerkinLocalDetectorCoordinateConvention,
    "ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES",
)
_CYCLIC_ANGULAR_AMPLITUDE = getattr(
    GalerkinLocalDetectorCoordinateConvention,
    "ANGULAR_WAVENUMBER_AMPLITUDE_IN_CYCLIC_COORDINATES",
)
_CYCLIC_NATIVE_AMPLITUDE = getattr(
    GalerkinLocalDetectorCoordinateConvention,
    "NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES",
)
_SOLID_ANGLE = getattr(
    GalerkinLocalDetectorCoordinateConvention,
    "ANGULAR_WAVENUMBER_AMPLITUDE_IN_SOLID_ANGLE",
)


def _point(value: int | Fraction) -> _Interval:
    """Return one singleton exact rational interval."""
    rational = Fraction(value)
    return rational, rational


def _carrier(
    lower: int | Fraction, upper: int | Fraction | None = None
) -> GalerkinLocalDetectorRationalInterval:
    """Return one exact detector interval carrier."""
    low = Fraction(lower)
    high = low if upper is None else Fraction(upper)
    return _make_local_detector_rational_interval(low, high)


def _raw_intervals(
    values: tuple[GalerkinLocalDetectorRationalInterval, ...],
) -> tuple[_Interval, ...]:
    """Recover exact endpoints without invoking detector arithmetic."""
    return tuple((value.lower, value.upper) for value in values)


def _multiply_nonnegative(*values: _Interval) -> _Interval:
    """Multiply independently checked nonnegative intervals directly."""
    lower = Fraction(1)
    upper = Fraction(1)
    for value in values:
        assert value[0] >= 0
        lower *= value[0]
        upper *= value[1]
    return lower, upper


def _add_intervals(*values: _Interval) -> _Interval:
    """Add exact intervals directly."""
    return (
        sum((value[0] for value in values), start=Fraction(0)),
        sum((value[1] for value in values), start=Fraction(0)),
    )


def _divide_nonnegative(
    numerator: _Interval, denominator: _Interval
) -> _Interval:
    """Divide a nonnegative interval by a strictly positive interval."""
    assert numerator[0] >= 0 < denominator[0]
    return numerator[0] / denominator[1], numerator[1] / denominator[0]


def _dyadic(value: object) -> Fraction:
    """Recover the exact rational represented by one binary64 value."""
    return Fraction.from_float(float(np.float64(cast(Any, value))))


def _directed_lower(value: Fraction) -> Fraction:
    """Convert a normal-range rational toward minus infinity test-locally."""
    candidate = np.float64(float(value))
    assert np.isfinite(candidate)
    if _dyadic(candidate) > value:
        candidate = np.nextafter(candidate, np.float64(-np.inf))
    assert candidate == 0.0 or abs(candidate) >= np.finfo(np.float64).tiny
    result = _dyadic(candidate)
    assert result <= value
    return result


def _directed_upper(value: Fraction) -> Fraction:
    """Convert a normal-range rational toward plus infinity test-locally."""
    candidate = np.float64(float(value))
    assert np.isfinite(candidate)
    if _dyadic(candidate) < value:
        candidate = np.nextafter(candidate, np.float64(np.inf))
    assert candidate == 0.0 or abs(candidate) >= np.finfo(np.float64).tiny
    result = _dyadic(candidate)
    assert result >= value
    return result


def _union_hull(raw: _Interval, point: object) -> _Interval:
    """Build the test-owned directed binary64 union of raw and point."""
    exact_point = _dyadic(point)
    return (
        _directed_lower(min(raw[0], exact_point)),
        _directed_upper(max(raw[1], exact_point)),
    )


def _union_hulls(
    raw: tuple[_Interval, ...], points: np.ndarray
) -> tuple[_Interval, ...]:
    """Build directed raw/point unions without reading production traces."""
    flat = np.asarray(points, dtype=np.float64).reshape(-1)
    assert len(raw) == flat.size
    return tuple(
        _union_hull(interval, point)
        for interval, point in zip(raw, flat, strict=True)
    )


def _independent_pixel_oracle(chain: _PublicChain) -> _IndependentPixelOracle:
    """Reconstruct the scalar pixel DAG from authenticated L8/input leaves."""
    terminal = chain.terminal
    branch = terminal.branch_evidence
    manifest = chain.pixel_input
    roots = branch.root_certificates
    retained = tuple(
        root is not None
        and root.classification is GalerkinLocalVacuumRootClass.PROPAGATING
        for root in roots
    )
    assert retained == (True,)
    assert branch.half_space_dispositions != (
        GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_EXACT_ZERO,
    )
    zero = _point(0)
    current_raw = tuple(
        (
            (root.root_interval.lower, root.root_interval.upper)
            if keep and root is not None and root.root_interval is not None
            else zero
        )
        for root, keep in zip(roots, retained, strict=True)
    )
    current_points = np.where(
        np.asarray(retained, dtype=np.bool_),
        np.asarray(branch.frozen_positive_root_realizations, dtype=np.float64),
        np.float64(0.0),
    ).astype(np.float64)
    current = _union_hulls(current_raw, current_points)

    zero_slab = terminal.projection_certificate.zero_slab_certificate
    target = zero_slab.represented_source_certificate.source.target
    target_ledger = target.fixed_linear_error_ledger
    k0 = (
        _dyadic(target_ledger.exact_wavenumber_lower_bound),
        _dyadic(target_ledger.exact_wavenumber_upper_bound),
    )
    assert k0[0] > 0 and k0[0] <= k0[1]
    assert manifest.coordinate_convention is _CYCLIC_NATIVE_AMPLITUDE
    pi_lower = float.fromhex("0x1.921fb54442d18p+1")
    pi_upper = np.nextafter(np.float64(pi_lower), np.float64(np.inf))
    scale_raw = (2 * _dyadic(pi_lower), 2 * _dyadic(pi_upper))
    scale_point = np.float64(2.0) * np.float64(np.pi)
    scale = _union_hull(scale_raw, scale_point)
    jacobians_raw = tuple(_point(1) if keep else zero for keep in retained)
    jacobian_points = np.where(
        np.asarray(retained, dtype=np.bool_), 1.0, 0.0
    ).astype(np.float64)
    jacobians = _union_hulls(jacobians_raw, jacobian_points)

    quadrature_raw = _raw_intervals(manifest.quadrature_weight_intervals)
    quadrature_points = np.asarray(
        manifest.quadrature_weight_points, dtype=np.float64
    )
    quadrature = _union_hulls(quadrature_raw, quadrature_points)
    aperture_raw = _raw_intervals(manifest.aperture_efficiency_intervals)
    aperture_points = np.asarray(
        manifest.aperture_efficiency_points, dtype=np.float64
    )
    aperture = _union_hulls(aperture_raw, aperture_points)
    mapping = tuple(int(value) for value in np.asarray(manifest.node_to_pixel))
    scale_squared = _multiply_nonnegative(scale, scale)
    outward_raw = tuple(
        _multiply_nonnegative(
            current[index],
            jacobians[index],
            scale_squared,
            quadrature[index],
        )
        for index in range(len(current))
    )
    scale_squared_point = np.float64(scale_point) * np.float64(scale_point)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        outward_points = current_points * jacobian_points
        outward_points = outward_points * scale_squared_point
        outward_points = outward_points * quadrature_points
    outward_points = outward_points.astype(np.float64)
    outward = _union_hulls(outward_raw, outward_points)

    pixels_raw = tuple(
        tuple(
            (
                _multiply_nonnegative(outward_raw[index], aperture[index])
                if mapping[index] == row
                else zero
            )
            for index in range(len(current))
        )
        for row in range(manifest.pixel_count)
    )
    pixel_points = np.zeros(
        (manifest.pixel_count, len(current)), dtype=np.float64
    )
    for index, row in enumerate(mapping):
        if row >= 0:
            pixel_points[row, index] = (
                outward_points[index] * aperture_points[index]
            )
    flat_pixel_raw = tuple(value for row in pixels_raw for value in row)
    flat_pixel_hulls = _union_hulls(flat_pixel_raw, pixel_points)
    pixels = tuple(
        tuple(
            flat_pixel_hulls[row * len(current) + index]
            for index in range(len(current))
        )
        for row in range(manifest.pixel_count)
    )
    margin_raw = tuple(
        (
            _multiply_nonnegative(
                outward_raw[index],
                (
                    Fraction(1) - aperture[index][1],
                    Fraction(1) - aperture[index][0],
                ),
            )
            if mapping[index] >= 0
            else outward_raw[index]
        )
        for index in range(len(current))
    )
    margin_points = (
        outward_points - np.sum(pixel_points, axis=0, dtype=np.float64)
    ).astype(np.float64)
    margin = _union_hulls(margin_raw, margin_points)

    amplitudes = np.asarray(
        branch.frozen_defining_branch_points, dtype=np.complex128
    )[:, 0]
    active = np.where(
        np.asarray(retained, dtype=np.bool_),
        amplitudes,
        np.complex128(0.0 + 0.0j),
    ).astype(np.complex128)
    amplitude_squared_values = tuple(
        (
            _dyadic(value.real) ** 2 + _dyadic(value.imag) ** 2
            if keep
            else Fraction(0)
        )
        for value, keep in zip(active, retained, strict=True)
    )
    amplitude_squared_raw = tuple(
        _point(value) for value in amplitude_squared_values
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        amplitude_squared_points = (
            np.real(active) * np.real(active)
            + np.imag(active) * np.imag(active)
        ).astype(np.float64)
    amplitude_squared = _union_hulls(
        amplitude_squared_raw, amplitude_squared_points
    )

    forms = (outward, *pixels)
    production_raw = tuple(
        _add_intervals(
            *(
                _multiply_nonnegative(amplitude, diagonal)
                for amplitude, diagonal in zip(
                    amplitude_squared, form, strict=True
                )
            )
        )
        for form in forms
    )
    form_points = np.concatenate((outward_points[None, :], pixel_points))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        production_points = np.sum(
            form_points * amplitude_squared_points[None, :],
            axis=1,
            dtype=np.float64,
        )
    production = _union_hulls(production_raw, production_points)
    form_norms = tuple(max(value[1] for value in form) for form in forms)
    assert len(branch.production_to_submitted_amplitude_error_bounds) == 1
    production_error = abs(
        _dyadic(branch.production_to_submitted_amplitude_error_bounds[0, 0])
    )
    state_error = abs(
        _dyadic(branch.state_radius_amplitude_error_bounds[0, 0])
    )
    total_error = _dyadic(branch.exact_state_prediction_error_l2_upper_bound)
    amplitude_norm = _dyadic(branch.production_prediction_l2_norm_upper_bound)
    production_realization_errors = tuple(
        norm * production_error * (2 * amplitude_norm + production_error)
        for norm in form_norms
    )
    combined_errors = tuple(
        norm * total_error * (2 * amplitude_norm + total_error)
        for norm in form_norms
    )
    incremental_errors = tuple(
        total - produced
        for produced, total in zip(
            production_realization_errors, combined_errors, strict=True
        )
    )
    exact_state = tuple(
        (max(Fraction(0), value[0] - error), value[1] + error)
        for value, error in zip(production, combined_errors, strict=True)
    )
    return _IndependentPixelOracle(
        current_raw,
        current,
        current_points,
        scale_raw,
        scale,
        scale_point,
        jacobians_raw,
        jacobians,
        jacobian_points,
        quadrature_raw,
        quadrature,
        aperture_raw,
        aperture,
        outward_raw,
        outward,
        outward_points,
        pixels_raw,
        pixels,
        pixel_points,
        margin_raw,
        margin,
        margin_points,
        amplitude_squared_raw,
        amplitude_squared,
        amplitude_squared_points,
        production_raw,
        production,
        production_points,
        form_norms,
        production_error,
        state_error,
        total_error,
        amplitude_norm,
        production_realization_errors,
        incremental_errors,
        combined_errors,
        exact_state,
    )


def _trace_hulls(trace: object) -> tuple[_Interval, ...]:
    """Recover the stored binary64 union hull without detector helpers."""
    lower = np.asarray(getattr(trace, "certified_hull_lower_bounds"))
    upper = np.asarray(getattr(trace, "certified_hull_upper_bounds"))
    return tuple(
        (Fraction.from_float(float(lo)), Fraction.from_float(float(hi)))
        for lo, hi in zip(lower, upper, strict=True)
    )


def _assert_trace_union_is_independent(trace: object) -> None:
    """Recompute every raw/point union hull and distance exactly."""
    raw_intervals = getattr(trace, "raw_intervals")
    points = np.asarray(getattr(trace, "point"), dtype=np.float64)
    errors = np.asarray(
        getattr(trace, "point_to_raw_absolute_error_upper_bounds"),
        dtype=np.float64,
    )
    hull_lowers = np.asarray(
        getattr(trace, "certified_hull_lower_bounds"), dtype=np.float64
    )
    hull_uppers = np.asarray(
        getattr(trace, "certified_hull_upper_bounds"), dtype=np.float64
    )
    exact_points = getattr(trace, "exact_point_intervals")
    for raw, point, stored_point, error, lower, upper in zip(
        raw_intervals,
        points,
        exact_points,
        errors,
        hull_lowers,
        hull_uppers,
        strict=True,
    ):
        exact_point = Fraction.from_float(float(point))
        assert stored_point.lower == stored_point.upper == exact_point
        exact_error = max(
            abs(exact_point - raw.lower), abs(exact_point - raw.upper)
        )
        assert Fraction.from_float(float(error)) == _directed_upper(
            exact_error
        )
        assert Fraction.from_float(float(lower)) == _directed_lower(
            min(raw.lower, exact_point)
        )
        assert Fraction.from_float(float(upper)) == _directed_upper(
            max(raw.upper, exact_point)
        )
    assert len(getattr(trace, "point_bytes_digest")) == 64
    assert len(getattr(trace, "raw_interval_digest")) == 64
    assert len(getattr(trace, "trace_digest")) == 64


def _assert_trace_matches_leaf_oracle(
    trace: object,
    *,
    raw: tuple[_Interval, ...],
    points: np.ndarray,
    hulls: tuple[_Interval, ...],
    stage: GalerkinLocalDetectorProductionStage,
    quantity: str,
    logical_shape: tuple[int, ...],
) -> None:
    """Require one trace to match a wholly test-owned raw/point/hull DAG."""
    assert getattr(trace, "stage") is stage
    assert getattr(trace, "quantity") == quantity
    assert getattr(trace, "logical_shape") == logical_shape
    assert _raw_intervals(getattr(trace, "raw_intervals")) == raw
    np.testing.assert_array_equal(
        np.asarray(getattr(trace, "point"), dtype=np.float64),
        np.asarray(points, dtype=np.float64).reshape(-1),
    )
    assert _trace_hulls(trace) == hulls
    _assert_trace_union_is_independent(trace)


def _independent_l8_resource_totals(
    port: GalerkinLocalPositivePortCertificate,
) -> tuple[int, int, int]:
    """Sum authenticated nested L8 leaves without the L9 aggregator."""
    branch = port.terminal_certificate.branch_evidence
    cut = port.terminal_certificate.cut_balance
    root_work = sum(
        (
            root.work_transcript.exact_work_count
            if root is not None
            else failure_work
        )
        for root, failure_work in zip(
            branch.root_certificates,
            branch.root_failure_work_counts,
            strict=True,
        )
    )
    propagator_work = sum(
        (
            propagator.interval_work_transcript.exact_work_count
            + (
                0
                if propagator.entire_transcript is None
                else propagator.entire_transcript.exact_work_count
            )
            if propagator is not None
            else failure_work
        )
        for propagator, failure_work in zip(
            branch.propagators,
            branch.propagator_failure_work_counts,
            strict=True,
        )
    )
    direct_work = sum(
        int(value)
        for value in (
            branch.direct_work_count_exact,
            branch.direct_rational_work_count_exact,
            cut.direct_work_count_exact,
            cut.direct_rational_work_count_exact,
        )
    )
    work = (
        root_work
        + propagator_work
        + branch.entire_evidence.total_exact_work_count
        + direct_work
    )
    trace_count = len(port.production_traces)
    trace_hulls = sum(
        2 * np.asarray(trace.point).size for trace in port.production_traces
    )
    return (
        work,
        trace_count,
        trace_hulls + branch.hull_completed_endpoint_count,
    )


def _reseal_port(
    value: GalerkinLocalPositivePortCertificate,
) -> GalerkinLocalPositivePortCertificate:
    """Rehash one adversarial positive port through its type owner."""
    return _make_local_positive_port_certificate(value)


def _reseal_pixel(
    value: GalerkinLocalPassivePixelForms,
) -> GalerkinLocalPassivePixelForms:
    """Rehash one adversarial pixel through its type owner."""
    return _make_local_passive_pixel_forms(value)


def _reseal_detector(
    value: GalerkinLocalCensoredPoissonDetector,
) -> GalerkinLocalCensoredPoissonDetector:
    """Rehash one adversarial detector through its type owner."""
    return _make_local_censored_poisson_detector(value)


def _reseal_likelihood(
    value: GalerkinLocalCensoredPoissonLikelihood,
) -> GalerkinLocalCensoredPoissonLikelihood:
    """Rehash one adversarial likelihood through its type owner."""
    return _make_local_censored_poisson_likelihood(value)


def _set_fields(value: _T, **fields: object) -> _T:
    """Populate a private parent-free Equinox shell for arithmetic replay."""
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    return value


def _evidence(value: dict[str, object]) -> dict[str, Any]:
    """Expose the heterogeneous private replay mapping to static tests."""
    return cast(dict[str, Any], value)


def _empty_work(
    *, maximum_work: int = 1_000_000, maximum_rational_bits: int = 262_144
):
    """Return one authenticated empty local-work leaf."""
    return _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
        coordinate_factor_count=0,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        exact_work_count=0,
        rational_peak_bits=0,
    )


def _entire_prefix(
    *, algorithm: str, exact_work_count: int
) -> EntireWorkTranscript:
    """Return one valid successful nested-kernel prefix transcript."""
    return EntireWorkTranscript(
        algorithm=algorithm,
        precision_bits=64,
        maximum_terms=256,
        maximum_work=100_000,
        maximum_range_reductions=64,
        maximum_rational_bits=262_144,
        series_terms=1,
        range_reductions=0,
        root_enclosures=0,
        rectangle_products=0,
        reciprocal_steps=0,
        exact_work_count=exact_work_count,
    )


def _pixel_input_manifest(
    **policy_overrides: int,
) -> GalerkinLocalPassivePixelInputManifest:
    """Build the one-fiber public pixel manifest with explicit L8 policies."""
    quadrature = Fraction(1, 1 << 900)
    quadrature_point = float.fromhex("0x1p-900")
    assert Fraction.from_float(quadrature_point) == quadrature
    assert (
        np.isfinite(quadrature_point)
        and quadrature_point != 0.0
        and abs(quadrature_point) >= np.finfo(np.float64).tiny
    )
    policies = {
        "maximum_stability_direct_pairs": l8_tests._STRUCTURAL_STABILITY_PAIRS,
        "maximum_gram_pairs": l8_tests._STRUCTURAL_GRAM_PAIRS,
        "maximum_terminal_direct_pairs": l8_tests._TERMINAL_PAIRS,
        "maximum_branch_direct_terms": l8_tests._BRANCH_TERMS,
        "maximum_cut_direct_pairs": l8_tests._CUT_PAIRS,
        "maximum_root_work": l8_tests._ROOT_WORK,
        "precision_bits": l8_tests._PRECISION,
        "maximum_terms": l8_tests._ENTIRE_TERMS,
        "maximum_entire_work": l8_tests._ENTIRE_WORK,
        "maximum_range_reductions": l8_tests._RANGE_REDUCTIONS,
        "maximum_interval_work": l8_tests._INTERVAL_WORK,
        "maximum_l8_rational_bits": l8_tests._RATIONAL_BITS,
        "maximum_detector_work": 1_000_000,
        "maximum_detector_rational_bits": 262_144,
    }
    policies.update(policy_overrides)
    return detector.create_local_passive_pixel_input_manifest(
        maximum_state_error=np.asarray(
            np.finfo(np.float64).max, dtype=np.float64
        ),
        node_to_pixel=np.asarray([0], dtype=np.int64),
        quadrature_weight_intervals=(_carrier(quadrature),),
        quadrature_weight_points=np.asarray(
            [quadrature_point], dtype=np.float64
        ),
        aperture_efficiency_intervals=(_carrier(Fraction(1, 2)),),
        aperture_efficiency_points=np.asarray([0.5], dtype=np.float64),
        route=GalerkinLocalPositivePortRoute.PROJECTED_OUTWARD_PROPAGATING,
        disposition=(
            GalerkinLocalVacuumTerminalDisposition.PLANE_DEFINED_FREE_CONTINUATION
        ),
        coordinate_convention=(
            GalerkinLocalDetectorCoordinateConvention.NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES
        ),
        pixel_count=1,
        **policies,
    )


@functools.lru_cache(maxsize=1)
def _public_chain() -> _PublicChain:
    """Build L8 once and cross exactly four nested public L8 replays."""
    terminal_certificate = l8_tests._terminal(structural=True)
    port = detector.certify_local_positive_port(
        terminal_certificate,
        route=GalerkinLocalPositivePortRoute.PROJECTED_OUTWARD_PROPAGATING,
        disposition=(
            GalerkinLocalVacuumTerminalDisposition.PLANE_DEFINED_FREE_CONTINUATION
        ),
        maximum_state_error=np.asarray(
            np.finfo(np.float64).max, dtype=np.float64
        ),
        maximum_stability_direct_pairs=l8_tests._STRUCTURAL_STABILITY_PAIRS,
        maximum_gram_pairs=l8_tests._STRUCTURAL_GRAM_PAIRS,
        maximum_terminal_direct_pairs=l8_tests._TERMINAL_PAIRS,
        maximum_branch_direct_terms=l8_tests._BRANCH_TERMS,
        maximum_cut_direct_pairs=l8_tests._CUT_PAIRS,
        maximum_root_work=l8_tests._ROOT_WORK,
        precision_bits=l8_tests._PRECISION,
        maximum_terms=l8_tests._ENTIRE_TERMS,
        maximum_entire_work=l8_tests._ENTIRE_WORK,
        maximum_range_reductions=l8_tests._RANGE_REDUCTIONS,
        maximum_interval_work=l8_tests._INTERVAL_WORK,
        maximum_rational_bits=l8_tests._RATIONAL_BITS,
    )
    pixel_input = _pixel_input_manifest()
    pixel = detector.certify_local_passive_pixel_forms(
        port, input_manifest=pixel_input
    )
    detector_input = (
        detector.create_local_censored_poisson_detector_input_manifest(
            pixel_inputs=(pixel_input,),
            ensemble_weight_numerators=(1,),
            ensemble_weight_denominators=(1,),
            incident_electron_count_interval=_carrier(1),
            incident_electron_count_point=np.asarray(1.0, dtype=np.float64),
            response_matrix=np.asarray([[1.0]], dtype=np.float64),
            pre_gain_background=np.asarray([0.25], dtype=np.float64),
            deterministic_gain=np.asarray([2.0], dtype=np.float64),
            electronic_offset=np.asarray([-1.0], dtype=np.float64),
            count_ceilings=np.asarray([2], dtype=np.int64),
            fit_mask=np.asarray([True], dtype=np.bool_),
            calibration_provenance="genuine structural L9 scalar fixture",
            maximum_detector_work=1_000_000,
            maximum_detector_rational_bits=262_144,
            maximum_count_ceiling=8,
            maximum_poisson_work=100_000,
            maximum_poisson_rational_bits=262_144,
            exp_precision_bits=64,
            maximum_exp_terms=256,
            maximum_exp_work=100_000,
            maximum_exp_range_reductions=64,
        )
    )
    detector_certificate = detector.certify_local_censored_poisson_detector(
        (pixel,), input_manifest=detector_input
    )
    likelihood = detector.enclose_local_censored_poisson_likelihood(
        detector_certificate,
        detector_input_manifest=detector_input,
        observed_counts=np.asarray([1], dtype=np.int64),
        maximum_detector_work=1_000_000,
        maximum_detector_rational_bits=262_144,
        log_precision_bits=48,
        maximum_log_terms=256,
        maximum_log_work=100_000,
        maximum_log_range_reductions=64,
    )
    return _PublicChain(
        terminal_certificate,
        port,
        pixel_input,
        pixel,
        detector_input,
        detector_certificate,
        likelihood,
    )


def _synthetic_detector_candidate(
    *, maximum_rational_bits: int = 262_144
) -> GalerkinLocalCensoredPoissonDetector:
    """Build a minimal parent-free carrier shell for the detector DAG."""
    work = _empty_work(maximum_rational_bits=maximum_rational_bits)
    quadratic_trace = _make_local_detector_real_production_trace(
        (_carrier(3), _carrier(1)),
        np.asarray([3.0, 1.0], dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.MODE_PRODUCTION_QUADRATIC,
        quantity="production_form_quadratic",
        logical_shape=(2,),
    )
    modes = SimpleNamespace(
        exact_reduced_flux_lower_bound=np.asarray(4.0, dtype=np.float64),
        exact_reduced_flux_upper_bound=np.asarray(4.0, dtype=np.float64),
        output_reduced_flux=np.asarray(4.0, dtype=np.float64),
    )
    terminal = SimpleNamespace(
        projection_certificate=SimpleNamespace(
            zero_slab_certificate=SimpleNamespace(
                represented_source_certificate=SimpleNamespace(
                    source=SimpleNamespace(modes=modes)
                )
            )
        )
    )
    pixel = _set_fields(
        object.__new__(GalerkinLocalPassivePixelForms),
        failure_mask=np.asarray(0, dtype=np.int64),
        pixel_count=1,
        exact_state_pixel_flux_intervals=(_carrier(1, 2),),
        production_quadratic_intervals=(_carrier(1),),
        exact_state_outward_flux_interval=_carrier(2, 3),
        positive_port=SimpleNamespace(terminal_certificate=terminal),
        production_traces=(quadratic_trace,),
        work_transcript=work,
        pixel_form_norm_upper_intervals=(_carrier(1),),
        production_to_exact_x_amplitude_error_interval=_carrier(0),
        state_radius_amplitude_error_interval=_carrier(0),
        exact_state_amplitude_error_interval=_carrier(0),
        production_amplitude_norm_interval=_carrier(1),
        production_realization_error_upper_intervals=(_carrier(0),),
        state_radius_incremental_error_upper_intervals=(_carrier(0),),
        combined_exact_state_error_upper_intervals=(_carrier(0),),
    )
    candidate = _set_fields(
        object.__new__(GalerkinLocalCensoredPoissonDetector),
        pixel_forms=(pixel,),
        ensemble_weight_numerators=(1,),
        ensemble_weight_denominators=(1,),
        incident_electron_count=_carrier(1),
        incident_electron_count_point=np.asarray(1.0, dtype=np.float64),
        response_matrix=np.asarray([[1.0]], dtype=np.float64),
        pre_gain_background=np.asarray([0.0], dtype=np.float64),
        deterministic_gain=np.asarray([2.0], dtype=np.float64),
        electronic_offset=np.asarray([-1.0], dtype=np.float64),
        count_ceilings=np.asarray([2], dtype=np.int64),
        fit_mask=np.asarray([True], dtype=np.bool_),
        maximum_count_ceiling=8,
        maximum_poisson_work=100_000,
        maximum_poisson_rational_bits=262_144,
        exp_precision_bits=96,
        maximum_exp_terms=512,
        maximum_exp_work=100_000,
        maximum_exp_range_reductions=128,
        work_transcript=work,
    )
    return candidate


def _synthetic_pixel_fallback_candidate(
    *,
    quadrature: GalerkinLocalDetectorRationalInterval = _carrier(1),
    aperture: GalerkinLocalDetectorRationalInterval = _carrier(1),
    maximum_work: int = 1_000,
    maximum_rational_bits: int = 64,
) -> GalerkinLocalPassivePixelForms:
    """Build one isolated parent-free pixel owner-round-trip candidate."""
    port = SimpleNamespace(
        production_amplitudes=np.asarray([1.0 + 0.0j], dtype=np.complex128),
        retained_propagating_mask=np.asarray([True], dtype=np.bool_),
        positive_port_eligible=np.asarray(True, dtype=np.bool_),
        failure_mask=np.asarray(0, dtype=np.int64),
        exact_root_intervals=(None,),
        certificate_digest="a" * 64,
    )
    zero = _carrier(0)
    return _make_local_passive_pixel_forms_candidate(
        positive_port=port,
        node_to_pixel=jnp.asarray([0], dtype=jnp.int64),
        production_evidence_available=jnp.asarray(False),
        positive_forms_eligible=jnp.asarray(False),
        passive_forms_eligible=jnp.asarray(False),
        failure_mask=jnp.asarray(0, dtype=jnp.int64),
        quadrature_weights=jnp.asarray([1.0], dtype=jnp.float64),
        aperture_efficiencies=jnp.asarray([1.0], dtype=jnp.float64),
        production_traces=(),
        current_weight_intervals=(),
        amplitude_scale_interval=zero,
        coordinate_jacobian_intervals=(),
        quadrature_weight_intervals=(quadrature,),
        aperture_efficiency_intervals=(aperture,),
        outward_form_diagonal_intervals=(),
        pixel_form_diagonal_intervals=(),
        outward_minus_pixel_form_diagonal_intervals=(),
        production_outward_quadratic_interval=zero,
        outward_form_norm_upper_interval=zero,
        outward_production_realization_error_upper_interval=zero,
        outward_state_radius_incremental_error_upper_interval=zero,
        outward_combined_exact_state_error_upper_interval=zero,
        exact_state_outward_flux_interval=zero,
        production_quadratic_intervals=(),
        pixel_form_norm_upper_intervals=(),
        production_to_exact_x_amplitude_error_interval=zero,
        state_radius_amplitude_error_interval=zero,
        exact_state_amplitude_error_interval=zero,
        production_amplitude_norm_interval=zero,
        production_realization_error_upper_intervals=(),
        state_radius_incremental_error_upper_intervals=(),
        combined_exact_state_error_upper_intervals=(),
        exact_state_pixel_flux_intervals=(),
        coordinate_convention=_ANGULAR,
        pixel_count=1,
        work_transcript=_empty_work(
            maximum_work=maximum_work,
            maximum_rational_bits=maximum_rational_bits,
        ),
        coordinate_factor_scope="candidate",
        pixel_form_scope="candidate",
        lvt56_error_scope="candidate",
        passivity_margin_scope="candidate",
        no_experimental_validity_scope="candidate",
        parent_port_certificate_digest="a" * 64,
        input_manifest_digest="b" * 64,
        pixel_model_identity_digest="0" * 64,
        pixel_model_evidence_digest="0" * 64,
        certificate_digest="0" * 64,
    )


def _ledger(
    *, maximum_work: int = 1_000_000, maximum_rational_bits: int = 4096
) -> detector._DetectorLedger:
    """Return one ample private exact-arithmetic ledger."""
    return detector._checked_policy(maximum_work, maximum_rational_bits)


def test_coordinate_conventions_are_factor_equivalent() -> None:
    """Match angular and both cyclic amplitude/Jacobian ledgers exactly."""
    root = (_point(2),)
    retained = (True,)
    angular_scale, angular_j = detector._coordinate_factors(
        _ANGULAR, _point(5), root, retained, _ledger()
    )
    cyclic_scale, cyclic_j = detector._coordinate_factors(
        _CYCLIC_ANGULAR_AMPLITUDE,
        _point(5),
        root,
        retained,
        _ledger(),
    )
    native_scale, native_j = detector._coordinate_factors(
        _CYCLIC_NATIVE_AMPLITUDE,
        _point(5),
        root,
        retained,
        _ledger(),
    )
    assert angular_scale == cyclic_scale == _point(1)
    assert angular_j == native_j == (_point(1),)
    native_scale_squared = (
        native_scale[0] * native_scale[0],
        native_scale[1] * native_scale[1],
    )
    assert cyclic_j == (native_scale_squared,)

    current = detector._positive_current_weights(root, retained, _ledger())
    angular_out, angular_pixels, _ = detector._pixel_form_diagonals(
        current,
        angular_j,
        angular_scale,
        cyclic_j,
        (_point(1),),
        (0,),
        1,
        _ledger(),
    )
    cyclic_out, cyclic_pixels, _ = detector._pixel_form_diagonals(
        current,
        cyclic_j,
        cyclic_scale,
        (_point(1),),
        (_point(1),),
        (0,),
        1,
        _ledger(),
    )
    native_out, native_pixels, _ = detector._pixel_form_diagonals(
        current,
        native_j,
        native_scale,
        (_point(1),),
        (_point(1),),
        (0,),
        1,
        _ledger(),
    )
    assert angular_out == cyclic_out
    assert angular_pixels == cyclic_pixels
    angular_flux = detector._mode_pixel_fluxes(
        (_point(1),), angular_pixels, _ledger()
    )
    cyclic_flux = detector._mode_pixel_fluxes(
        (_point(1),), cyclic_pixels, _ledger()
    )
    native_flux = detector._mode_pixel_fluxes(
        (_point(1),), native_pixels, _ledger()
    )
    assert angular_flux == cyclic_flux == native_flux
    assert angular_out == cyclic_out == native_out


def test_native_cyclic_scale_squared_is_absorbed_into_every_q_quantity() -> (
    None
):
    """Put the full amplitude scale squared in Q, its norm, and LVT.56."""
    scale = _point(3)
    outward, pixels, margin = detector._pixel_form_diagonals(
        (_point(2),),
        (_point(5),),
        scale,
        (_point(7),),
        (_point(Fraction(1, 2)),),
        (0,),
        1,
        _ledger(),
    )
    assert outward == (_point(2 * 5 * 3**2 * 7),)
    assert pixels == ((_point(315),),)
    assert margin == (_point(315),)
    reports = detector._lvt56_quadratic_reports(
        (_point(4),),
        (outward, *pixels),
        _point(Fraction(1, 4)),
        _point(Fraction(1, 8)),
        _point(Fraction(1, 2)),
        _point(2),
        _ledger(),
    )
    production, form_norms, _, _, combined, _ = reports
    assert production == (_point(2520), _point(1260))
    assert form_norms == (_point(630), _point(315))
    assert combined == (
        _point(Fraction(2835, 2)),
        _point(Fraction(2835, 4)),
    )


def test_mixed_branch_classes_have_exact_zero_port_factors() -> None:
    """Retain only propagating roots without positive sentinels elsewhere."""
    roots: tuple[_Interval | None, ...] = (
        (Fraction(2), Fraction(3)),
        (Fraction(7), Fraction(8)),
        _point(0),
        None,
    )
    retained = (True, False, False, False)
    current = detector._positive_current_weights(roots, retained, _ledger())
    scale, jacobians = detector._coordinate_factors(
        _SOLID_ANGLE,
        (Fraction(5), Fraction(6)),
        roots,
        retained,
        _ledger(),
    )
    assert current == ((Fraction(2), Fraction(3)),) + (_point(0),) * 3
    assert scale == _point(1)
    assert jacobians == ((Fraction(10), Fraction(18)),) + (_point(0),) * 3
    outward, _, _ = detector._pixel_form_diagonals(
        current,
        jacobians,
        _point(1),
        (_point(1),) * 4,
        (_point(1),) * 4,
        (0, -1, -1, -1),
        1,
        _ledger(),
    )
    assert outward == ((Fraction(20), Fraction(54)),) + (_point(0),) * 3
    with pytest.raises(ValueError, match="require roots"):
        detector._positive_current_weights((None,), (True,), _ledger())
    with pytest.raises(ValueError, match="strictly positive"):
        detector._coordinate_factors(
            _SOLID_ANGLE, _point(5), (_point(0),), (True,), _ledger()
        )


def test_huge_zero_weight_root_is_retained_without_exposing_arithmetic() -> (
    None
):
    """Accept a large classified excluded root and return canonical zeros."""
    huge = Fraction(1 << 2047)
    roots = (_point(2), (huge, huge))
    retained = (True, False)
    work = _ledger(maximum_rational_bits=4096)
    assert detector._positive_current_weights(roots, retained, work) == (
        _point(2),
        _point(0),
    )
    scale, jacobians = detector._coordinate_factors(
        _SOLID_ANGLE, _point(5), roots, retained, work
    )
    assert scale == _point(1)
    assert jacobians == (_point(10), _point(0))
    assert work.rational_peak_bits == 2048


def test_port_routes_separate_projection_from_outgoing_zero_status() -> None:
    """Require exact inward zero only for the stronger radiation route."""
    expected_port_branches = inspect.unwrap(
        local_types._expected_port_branches
    )
    root = SimpleNamespace(
        classification=GalerkinLocalVacuumRootClass.PROPAGATING
    )

    def terminal_with(
        status: GalerkinLocalVacuumHalfSpaceDisposition,
    ) -> GalerkinLocalVacuumTerminalCertificate:
        return cast(
            GalerkinLocalVacuumTerminalCertificate,
            SimpleNamespace(
                vacuum_branch_eligible=np.asarray(True, dtype=np.bool_),
                branch_evidence=SimpleNamespace(
                    root_certificates=(root,),
                    half_space_dispositions=(status,),
                ),
            ),
        )

    exact_zero = expected_port_branches(
        terminal_with(
            GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_EXACT_ZERO
        ),
        GalerkinLocalPositivePortRoute.OUTGOING_RADIATION,
    )
    assert exact_zero == (
        (
            GalerkinLocalPositivePortBranchDisposition.PROPAGATING_OUTWARD_RETAINED_INWARD_EXACT_ZERO,
        ),
        (True,),
        (False,),
        GalerkinLocalDetectorFailure.NONE,
        True,
        True,
    )

    unresolved_parent = terminal_with(
        GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_UNRESOLVED
    )
    outgoing = expected_port_branches(
        unresolved_parent,
        GalerkinLocalPositivePortRoute.OUTGOING_RADIATION,
    )
    assert outgoing[0] == (
        GalerkinLocalPositivePortBranchDisposition.PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_UNRESOLVED,
    )
    assert outgoing[3] is (
        GalerkinLocalDetectorFailure.PROPAGATING_INWARD_NOT_EXACT_ZERO
    )
    assert outgoing[4:] == (True, False)

    projected = expected_port_branches(
        unresolved_parent,
        GalerkinLocalPositivePortRoute.PROJECTED_OUTWARD_PROPAGATING,
    )
    assert projected[0] == outgoing[0]
    assert projected[1:3] == ((True,), (False,))
    assert projected[3] is GalerkinLocalDetectorFailure.NONE
    assert projected[4:] == (True, False)


def test_solid_angle_uses_tilted_wide_angle_k0_kappa_squared_weight() -> None:
    """Keep root current and solid-angle Jacobian visible and exact."""
    roots = (_point(Fraction(1, 2)), _point(4))
    retained = (True, True)
    current = detector._positive_current_weights(roots, retained, _ledger())
    _, jacobians = detector._coordinate_factors(
        _SOLID_ANGLE, _point(5), roots, retained, _ledger()
    )
    assert current == roots
    assert jacobians == (_point(Fraction(5, 2)), _point(20))
    outward, _, _ = detector._pixel_form_diagonals(
        current,
        jacobians,
        _point(1),
        (_point(1), _point(1)),
        (_point(1), _point(1)),
        (0, 0),
        1,
        _ledger(),
    )
    assert outward == (_point(Fraction(5, 4)), _point(80))


def test_pixel_forms_are_positive_and_structurally_passive() -> None:
    """Build disjoint pixels and the shared-factor Q-out passivity margin."""
    outward, pixels, margin = detector._pixel_form_diagonals(
        ((Fraction(2), Fraction(3)), _point(1)),
        ((Fraction(4), Fraction(5)), _point(2)),
        _point(1),
        ((Fraction(1), Fraction(2)), _point(3)),
        ((Fraction(1, 4), Fraction(1, 2)), _point(1)),
        (0, -1),
        2,
        _ledger(),
    )
    assert outward == ((Fraction(8), Fraction(30)), _point(6))
    assert pixels == (
        ((Fraction(2), Fraction(15)), _point(0)),
        (_point(0), _point(0)),
    )
    assert margin == ((Fraction(4), Fraction(45, 2)), _point(6))
    with pytest.raises(ValueError, match=r"contained in \[0,1\]"):
        detector._pixel_form_diagonals(
            (_point(1),),
            (_point(1),),
            _point(1),
            (_point(1),),
            (_point(Fraction(5, 4)),),
            (0,),
            1,
            _ledger(),
        )


def test_lvt56_reports_every_term_and_composes_error_once() -> None:
    """Split realization and state-radius errors without duplicating E-a."""
    reports = detector._lvt56_quadratic_reports(
        (_point(9), _point(16)),
        ((_point(2), _point(1)),),
        _point(Fraction(1, 4)),
        _point(2),
        _point(Fraction(1, 2)),
        _point(5),
        _ledger(),
    )
    (
        production,
        q_norm,
        production_error,
        state_increment,
        combined_error,
        exact_state,
    ) = reports
    assert production == (_point(34),)
    assert q_norm == (_point(2),)
    assert production_error == (_point(Fraction(41, 8)),)
    assert state_increment == (_point(Fraction(43, 8)),)
    assert combined_error == (_point(Fraction(21, 2)),)
    assert (
        production_error[0][0] + state_increment[0][0] == combined_error[0][0]
    )
    assert exact_state == ((Fraction(47, 2), Fraction(89, 2)),)
    parameters = inspect.signature(
        detector._lvt56_quadratic_reports
    ).parameters
    assert "production_root_error" not in parameters
    assert "production_to_exact_x_amplitude_error" in parameters
    assert "state_radius_amplitude_error" in parameters
    assert "exact_state_amplitude_error" in parameters
    with pytest.raises(ValueError, match="must not be smaller"):
        detector._lvt56_quadratic_reports(
            (_point(1),),
            ((_point(1),),),
            _point(Fraction(1, 2)),
            _point(Fraction(1, 2)),
            _point(Fraction(1, 4)),
            _point(1),
            _ledger(),
        )


def test_outward_passivity_requires_a_uniform_nonnegative_margin() -> None:
    """Separate operator-form passivity from incident-flux passivity."""
    margins = detector._outward_passivity_margins(
        ((Fraction(50), Fraction(51)),),
        ((Fraction(40), Fraction(45)),),
        _ledger(),
    )
    assert margins == ((Fraction(5), Fraction(11)),)
    with pytest.raises(ValueError, match="not nonnegative"):
        detector._outward_passivity_margins(
            (_point(40),), (_point(41),), _ledger()
        )


def test_incoherent_modes_mix_after_quadratics_and_cj_cancels() -> None:
    """Normalize each mode before its exact population-weighted dose sum."""
    mode_fluxes = (
        (_point(2), _point(4)),
        (_point(3), _point(9)),
    )
    incident = (_point(10), _point(12))
    weights = (Fraction(1, 4), Fraction(3, 4))
    work = _ledger()
    fractions, means = detector._normalize_mix_and_dose(
        mode_fluxes, incident, weights, _point(100), work
    )
    assert fractions == (
        (_point(Fraction(1, 5)), _point(Fraction(2, 5))),
        (_point(Fraction(1, 4)), _point(Fraction(3, 4))),
    )
    assert means == (_point(Fraction(95, 4)), _point(Fraction(265, 4)))
    assert work.ensemble_product_count == 4

    carrier_factor = Fraction(7, 3)
    scaled_fluxes = tuple(
        tuple(
            (lower * carrier_factor, upper * carrier_factor)
            for lower, upper in mode
        )
        for mode in mode_fluxes
    )
    scaled_incident = tuple(
        (lower * carrier_factor, upper * carrier_factor)
        for lower, upper in incident
    )
    scaled_fractions, scaled_means = detector._normalize_mix_and_dose(
        scaled_fluxes,
        scaled_incident,
        weights,
        _point(100),
        _ledger(),
    )
    assert scaled_fractions == fractions
    assert scaled_means == means


def test_response_and_gain_preserve_the_pre_gain_likelihood_boundary() -> None:
    """Apply positive routing before likelihood and post-censor gain."""
    pre_gain = detector._apply_nonnegative_response(
        (_point(10), _point(20)),
        (
            (Fraction(1, 2), Fraction(1, 4)),
            (Fraction(1, 2), Fraction(1, 2)),
        ),
        (_point(1), _point(0)),
        _ledger(),
    )
    assert pre_gain == (_point(11), _point(15))
    digitized = detector._apply_gain_and_offset(
        pre_gain,
        (Fraction(2), Fraction(3)),
        (Fraction(-1), Fraction(4)),
        _ledger(),
    )
    assert digitized == (_point(21), _point(49))
    with pytest.raises(ValueError, match="nonnegative"):
        detector._apply_nonnegative_response(
            (_point(1),), ((Fraction(-1),),), (_point(0),), _ledger()
        )
    with pytest.raises(ValueError, match="at most one"):
        detector._apply_nonnegative_response(
            (_point(1),), ((Fraction(2),),), (_point(0),), _ledger()
        )


def test_detector_core_orders_all_seven_stages_and_censors_singleton_point() -> (  # noqa: E501
    None
):
    """Replay Q→fraction→dose→response→censor→gain exactly once."""
    candidate = _synthetic_detector_candidate()
    evidence = _evidence(
        detector._expected_local_censored_poisson_detector_core(candidate)
    )
    assert evidence["detector_eligible"] is True
    assert evidence["likelihood_law_eligible"] is True
    assert evidence["failure_mask"] == int(GalerkinLocalDetectorFailure.NONE)
    assert [trace.stage for trace in evidence["production_traces"]] == [
        GalerkinLocalDetectorProductionStage.MODE_PIXEL_FRACTION,
        GalerkinLocalDetectorProductionStage.ENSEMBLE_WEIGHT,
        GalerkinLocalDetectorProductionStage.INCIDENT_DOSE,
        GalerkinLocalDetectorProductionStage.IDEAL_ARRIVAL_MEAN,
        GalerkinLocalDetectorProductionStage.PRE_GAIN_RESPONSE_MEAN,
        GalerkinLocalDetectorProductionStage.CENSORED_COUNT_MEAN,
        GalerkinLocalDetectorProductionStage.POST_CENSOR_DIGITIZED_MEAN,
    ]
    assert [trace.quantity for trace in evidence["production_traces"]] == [
        "mode_pixel_fraction",
        "ensemble_weight",
        "incident_electron_count",
        "ideal_arrival_mean",
        "production_pre_gain_mean",
        "production_censored_count_mean",
        "production_digitized_mean",
    ]
    for trace in evidence["production_traces"]:
        assert trace.point_dtype == "float64"
        assert len(trace.point_bytes_digest) == 64
        assert len(trace.raw_interval_digest) == 64
        assert len(trace.trace_digest) == 64
        for index, point in enumerate(np.asarray(trace.point)):
            assert float(trace.certified_hull_lower_bounds[index]) <= point
            assert point <= float(trace.certified_hull_upper_bounds[index])

    ideal = evidence["ideal_arrival_mean_intervals"][0]
    exact_pre_gain = evidence["exact_state_pre_gain_mean_intervals"][0]
    production_pre_gain = evidence["production_pre_gain_mean_point_intervals"][
        0
    ]
    assert (ideal.lower, ideal.upper) == (Fraction(1, 4), Fraction(1, 2))
    assert (exact_pre_gain.lower, exact_pre_gain.upper) == (
        Fraction(1, 4),
        Fraction(1, 2),
    )
    assert (
        production_pre_gain.lower
        == production_pre_gain.upper
        == Fraction(1, 4)
    )
    production_mean_work = evidence["production_censored_mean_transcripts"][0]
    exact_mean_work = evidence["censored_mean_transcripts"][0]
    assert production_mean_work is not None and exact_mean_work is not None
    assert production_mean_work.algorithm.endswith("mean_v1")
    assert exact_mean_work.algorithm.endswith("mean_v1")
    assert evidence["work_transcript"].exact_work_count == (
        detector._planned_detector_exact_work(
            mode_count=1, pixel_count=1, channel_count=1
        )
    )
    assert evidence["work_transcript"].production_trace_count == 7
    assert evidence["expected_digitized_mean_intervals"][0].lower == (
        2 * evidence["censored_mean_intervals"][0].lower - 1
    )


def test_likelihood_core_uses_admitted_hull_positive_floor_and_fixed_mask() -> (  # noqa: E501
    None
):
    """Enclose point and admitted law separately with no epsilon floor."""
    detector_candidate = _synthetic_detector_candidate()
    detector_evidence = _evidence(
        detector._expected_local_censored_poisson_detector_core(
            detector_candidate
        )
    )
    object.__setattr__(
        detector_candidate,
        "production_pre_gain_mean_point_intervals",
        detector_evidence["production_pre_gain_mean_point_intervals"],
    )
    object.__setattr__(
        detector_candidate,
        "exact_state_pre_gain_mean_intervals",
        detector_evidence["exact_state_pre_gain_mean_intervals"],
    )
    object.__setattr__(
        detector_candidate,
        "work_transcript",
        detector_evidence["work_transcript"],
    )
    object.__setattr__(
        detector_candidate,
        "production_evidence_available",
        np.asarray(True, dtype=np.bool_),
    )
    object.__setattr__(
        detector_candidate,
        "detector_eligible",
        np.asarray(True, dtype=np.bool_),
    )
    object.__setattr__(
        detector_candidate,
        "likelihood_law_eligible",
        np.asarray(True, dtype=np.bool_),
    )
    likelihood = _set_fields(
        object.__new__(GalerkinLocalCensoredPoissonLikelihood),
        detector=detector_candidate,
        observed_counts=np.asarray([1], dtype=np.int64),
        log_precision_bits=96,
        maximum_log_terms=512,
        maximum_log_work=100_000,
        maximum_log_range_reductions=128,
        work_transcript=_empty_work(),
    )
    evidence = _evidence(
        detector._expected_local_censored_poisson_likelihood_core(likelihood)
    )
    point = detector_candidate.production_pre_gain_mean_point_intervals[0]
    exact = detector_candidate.exact_state_pre_gain_mean_intervals[0]
    admitted = evidence["admitted_pre_gain_mean_hull_intervals"][0]
    assert admitted.lower == min(point.lower, exact.lower)
    assert admitted.upper == max(point.upper, exact.upper)
    assert evidence["likelihood_evidence_available"] is True
    assert evidence["likelihood_law_eligible"] is True
    assert evidence["nll_eligible"] is True
    floor = evidence["fitted_probability_positive_floor_intervals"][0]
    assert floor is not None and floor.lower == floor.upper > 0
    assert evidence["production_probability_point_intervals"][0].lower > 0
    assert evidence["admitted_hull_probability_intervals"][0].lower > 0
    assert evidence["production_nll_point_intervals"][0] is not None
    assert evidence["admitted_hull_nll_intervals"][0] is not None
    assert evidence["total_nll_interval"] is not None
    assert [trace.stage for trace in evidence["production_traces"]] == [
        GalerkinLocalDetectorProductionStage.CENSORED_PROBABILITY,
        GalerkinLocalDetectorProductionStage.CENSORED_NLL,
    ]
    assert "no epsilon" in evidence["nll_scope"]
    assert "derivative eligibility" in evidence["no_derivative_scope"]

    object.__setattr__(
        detector_candidate, "fit_mask", np.asarray([False], dtype=np.bool_)
    )
    masked = _evidence(
        detector._expected_local_censored_poisson_likelihood_core(likelihood)
    )
    for name in (
        "admitted_pre_gain_mean_hull_intervals",
        "production_probability_point_intervals",
        "admitted_hull_probability_intervals",
    ):
        assert stored_value_payload(masked[name]) == stored_value_payload(
            evidence[name]
        )
    assert (
        masked["production_probability_transcripts"]
        == (evidence["production_probability_transcripts"])
    )
    assert (
        masked["admitted_hull_probability_transcripts"]
        == (evidence["admitted_hull_probability_transcripts"])
    )
    assert masked["production_traces"][0].trace_digest == (
        evidence["production_traces"][0].trace_digest
    )
    assert masked["fitted_probability_positive_floor_intervals"] == (None,)
    assert masked["production_nll_point_intervals"] == (None,)
    assert masked["admitted_hull_nll_intervals"] == (None,)
    assert masked["total_nll_interval"].lower == 0
    assert masked["total_nll_interval"].upper == 0


def test_censored_poisson_reports_keep_mean_nll_and_gain_disjoint() -> None:
    """Preserve transcripts and require a positive fitted probability."""
    reports = detector._censored_poisson_reports(
        (_point(1), _point(0)),
        (0, 1),
        (3, 2),
        (True, True),
        maximum_count_ceiling=8,
        maximum_poisson_work=100_000,
        maximum_rational_bits=262_144,
        exp_precision_bits=96,
        maximum_exp_terms=512,
        maximum_exp_work=100_000,
        maximum_exp_range_reductions=128,
        log_precision_bits=96,
        maximum_log_terms=512,
        maximum_log_work=100_000,
        maximum_log_range_reductions=128,
    )
    probabilities, means, nlls, probability_work, mean_work, nll_work = reports
    assert probabilities[0][0] > 0
    assert probabilities[1] == _point(0)
    assert means[0][0] < 1 and means[0][1] < 1
    assert means[1] == _point(0)
    assert nlls[0] is not None and nll_work[0] is not None
    assert nlls[1] is None and nll_work[1] is None
    assert probability_work[0].algorithm.endswith("probability_v1")
    assert mean_work[0].algorithm.endswith("mean_v1")
    assert (
        "deterministic_gain"
        not in inspect.signature(detector._censored_poisson_reports).parameters
    )


def test_private_detector_resource_failures_remain_typed() -> None:
    """Reject bool policies, exact-work overflow, bits, and nested failures."""
    with pytest.raises(TypeError, match="Python integers"):
        detector._checked_policy(True, 64)

    work_limited = detector._checked_policy(1, 64)
    assert work_limited.add(Fraction(1), Fraction(1)) == 2
    with pytest.raises(detector._DetectorArithmeticError) as work_error:
        work_limited.add(Fraction(1), Fraction(1))
    assert work_error.value.failure is (
        GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED
    )
    assert work_error.value.exact_work_count == 2

    bit_limited = detector._checked_policy(10, 2)
    with pytest.raises(detector._DetectorArithmeticError) as bit_error:
        bit_limited.add(Fraction(8), Fraction(1))
    assert (
        bit_error.value.failure
        is GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT
    )
    assert bit_error.value.exact_work_count == 0

    zero_denominator = detector._checked_policy(10, 64)
    with pytest.raises(ZeroDivisionError, match="denominator"):
        zero_denominator.divide(Fraction(1), Fraction(0))
    assert zero_denominator.exact_work_count == 0

    with pytest.raises(CensoredPoissonEnclosureError) as nested:
        detector._censored_poisson_reports(
            (_point(1),),
            (0,),
            (2,),
            (True,),
            maximum_count_ceiling=2,
            maximum_poisson_work=100,
            maximum_rational_bits=128,
            exp_precision_bits=96,
            maximum_exp_terms=1,
            maximum_exp_work=100,
            maximum_exp_range_reductions=8,
            log_precision_bits=96,
            maximum_log_terms=32,
            maximum_log_work=100,
            maximum_log_range_reductions=8,
        )
    assert nested.value.nested_failure is not None
    assert nested.value.nested_exact_work_count is not None


def test_pixel_manifest_fans_out_every_independent_l8_policy_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward every caller-owned L8 policy under its exact public name.

    :see: :func:`ptyrodactyl.galerkin.\
create_local_passive_pixel_input_manifest`
    """
    manifest = _pixel_input_manifest()
    captured: dict[str, object] = {}

    class Captured(RuntimeError):
        """Stop immediately after the nested public-call boundary."""

    def capture(_certificate: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise Captured

    monkeypatch.setattr(
        detector, "prepare_local_positive_port_certificate", capture
    )
    port_shell = object.__new__(GalerkinLocalPositivePortCertificate)
    with pytest.raises(Captured):
        detector.certify_local_passive_pixel_forms(
            port_shell, input_manifest=manifest
        )
    canonical_kwargs = {
        "route": manifest.route,
        "disposition": manifest.terminal_disposition,
        "maximum_state_error": manifest.maximum_state_error,
        "maximum_stability_direct_pairs": (
            manifest.maximum_stability_direct_pairs
        ),
        "maximum_gram_pairs": manifest.maximum_gram_pairs,
        "maximum_terminal_direct_pairs": (
            manifest.maximum_terminal_direct_pairs
        ),
        "maximum_branch_direct_terms": manifest.maximum_branch_direct_terms,
        "maximum_cut_direct_pairs": manifest.maximum_cut_direct_pairs,
        "maximum_root_work": manifest.maximum_root_work,
        "precision_bits": manifest.precision_bits,
        "maximum_terms": manifest.maximum_terms,
        "maximum_entire_work": manifest.maximum_entire_work,
        "maximum_range_reductions": manifest.maximum_range_reductions,
        "maximum_interval_work": manifest.maximum_interval_work,
        "maximum_rational_bits": manifest.maximum_l8_rational_bits,
    }
    assert captured == canonical_kwargs
    policy_names = (
        ("maximum_stability_direct_pairs", "maximum_stability_direct_pairs"),
        ("maximum_gram_pairs", "maximum_gram_pairs"),
        ("maximum_terminal_direct_pairs", "maximum_terminal_direct_pairs"),
        ("maximum_branch_direct_terms", "maximum_branch_direct_terms"),
        ("maximum_cut_direct_pairs", "maximum_cut_direct_pairs"),
        ("maximum_root_work", "maximum_root_work"),
        ("precision_bits", "precision_bits"),
        ("maximum_terms", "maximum_terms"),
        ("maximum_entire_work", "maximum_entire_work"),
        ("maximum_range_reductions", "maximum_range_reductions"),
        ("maximum_interval_work", "maximum_interval_work"),
        ("maximum_l8_rational_bits", "maximum_rational_bits"),
    )
    for policy, upstream_name in policy_names:
        unsealed = replace(manifest, **{policy: getattr(manifest, policy) + 1})
        with pytest.raises(ValueError, match="input digest disagrees"):
            detector.certify_local_passive_pixel_forms(
                port_shell, input_manifest=unsealed
            )
        sealed = _make_local_passive_pixel_input_manifest(unsealed)
        captured.clear()
        with pytest.raises(Captured):
            detector.certify_local_passive_pixel_forms(
                port_shell, input_manifest=sealed
            )
        expected = dict(canonical_kwargs)
        expected[upstream_name] = getattr(sealed, policy)
        assert captured == expected


def test_detector_preflight_rational_range_and_scientific_stops_are_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish budget, overflow, raw bits, range, and model failures."""
    budget = _synthetic_detector_candidate()
    object.__setattr__(budget, "work_transcript", _empty_work(maximum_work=1))
    budget_evidence = _evidence(
        detector._expected_local_censored_poisson_detector(budget)
    )
    budget_work = budget_evidence["work_transcript"]
    assert budget_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED
    )
    assert budget_work.preflight_failed
    assert not budget_work.count_overflow
    assert budget_work.exact_work_count == 0

    overflow = _synthetic_detector_candidate()
    with monkeypatch.context() as context:
        context.setattr(
            detector,
            "_planned_detector_exact_work",
            lambda **_kwargs: 1 << 63,
        )
        overflow_evidence = _evidence(
            detector._expected_local_censored_poisson_detector(overflow)
        )
    overflow_work = overflow_evidence["work_transcript"]
    assert overflow_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.EXACT_WORK_COUNT_OVERFLOW
    )
    assert overflow_work.preflight_failed and overflow_work.count_overflow

    bits = _synthetic_detector_candidate(maximum_rational_bits=16)
    object.__setattr__(
        bits,
        "incident_electron_count",
        _carrier(Fraction(1, 1 << 20)),
    )
    bit_evidence = _evidence(
        detector._expected_local_censored_poisson_detector(bits)
    )
    bit_work = bit_evidence["work_transcript"]
    assert bit_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT
    )
    assert bit_work.arithmetic_failure is (
        GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT
    )
    assert bit_work.rational_peak_bits > bit_work.maximum_rational_bits

    nonpositive = _synthetic_detector_candidate()
    object.__setattr__(
        nonpositive,
        "response_matrix",
        np.asarray([[-1.0]], dtype=np.float64),
    )
    scientific = _evidence(
        detector._expected_local_censored_poisson_detector(nonpositive)
    )
    assert scientific["failure_mask"] == int(
        GalerkinLocalDetectorFailure.RESPONSE_NONPOSITIVE
    )
    assert scientific["production_evidence_available"] is False

    out_of_range = _synthetic_detector_candidate()
    object.__setattr__(
        out_of_range,
        "incident_electron_count_point",
        np.asarray(np.inf, dtype=np.float64),
    )
    range_evidence = _evidence(
        detector._expected_local_censored_poisson_detector(out_of_range)
    )
    assert range_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert range_evidence["work_transcript"].arithmetic_failure is (
        GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE
    )


def test_detector_helper_partial_stop_preserves_nested_call_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind both exact and production mean failures to call and channel."""
    candidate = _synthetic_detector_candidate()
    exp_prefix = _entire_prefix(
        algorithm="exact_fraction_real_exp_v1", exact_work_count=5
    )
    log_prefix = _entire_prefix(
        algorithm="exact_fraction_real_log_atanh_pow2_v1",
        exact_work_count=7,
    )

    def fail_mean(*_args: object, **_kwargs: object) -> object:
        raise CensoredPoissonEnclosureError(
            CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE,
            3,
            "forced nested exp stop",
            attempted_exact_work_count=3,
            nested_kernel="exp",
            nested_failure=EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
            nested_exact_work_count=2,
            nested_attempted_exact_work_count=2,
            prior_exp_transcripts=(exp_prefix,),
            prior_log_transcripts=(log_prefix,),
        )

    monkeypatch.setattr(detector, "enclose_censored_poisson_mean", fail_mean)
    evidence = _evidence(
        detector._expected_local_censored_poisson_detector(candidate)
    )
    failure = GalerkinLocalDetectorFailure(evidence["failure_mask"])
    assert failure == (
        GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
        | GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
    )
    exact = evidence["censored_mean_failures"][0]
    production = evidence["production_censored_mean_failures"][0]
    assert (
        exact.call is GalerkinLocalDetectorHelperCall.EXACT_STATE_CENSORED_MEAN
    )
    assert production.call is (
        GalerkinLocalDetectorHelperCall.PRODUCTION_CENSORED_MEAN
    )
    for helper in (exact, production):
        assert helper.channel_index == 0
        assert helper.nested_kernel == "exp"
        assert (
            helper.nested_failure
            is EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
        )
        assert helper.local_exact_work_count_exact == "3"
        assert helper.nested_exact_work_count_exact == "2"
        assert helper.nested_attempted_exact_work_count_exact == "2"
        assert helper.prior_exp_transcripts == (exp_prefix,)
        assert helper.prior_log_transcripts == (log_prefix,)
        assert len(helper.failure_digest) == 64
    assert evidence["censored_mean_transcripts"] == (None,)
    assert evidence["production_censored_mean_transcripts"] == (None,)
    traces = evidence["production_traces"]
    assert [trace.stage for trace in traces] == [
        GalerkinLocalDetectorProductionStage.MODE_PIXEL_FRACTION,
        GalerkinLocalDetectorProductionStage.ENSEMBLE_WEIGHT,
        GalerkinLocalDetectorProductionStage.INCIDENT_DOSE,
        GalerkinLocalDetectorProductionStage.IDEAL_ARRIVAL_MEAN,
        GalerkinLocalDetectorProductionStage.PRE_GAIN_RESPONSE_MEAN,
    ]
    assert [trace.quantity for trace in traces] == [
        "mode_pixel_fraction",
        "ensemble_weight",
        "incident_electron_count",
        "ideal_arrival_mean",
        "production_pre_gain_mean",
    ]
    work = evidence["work_transcript"]
    assert work.arithmetic_failure is (
        GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
    )
    assert work.production_trace_count == len(traces) == 5
    assert work.hull_endpoint_count == 10
    assert work.exact_work_count == 39
    assert work.planned_exact_work_count_exact == "53"
    assert work.attempted_exact_work_count_exact == "39"
    assert int(work.nested_helper_work_count_exact) == 2 * (3 + 5 + 7 + 2)


def test_pixel_fallback_owner_round_trip_preserves_primitive_exact_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seal replayable bit, preflight, scientific, and range stop carriers."""
    monkeypatch.setattr(
        local_types,
        "_validate_local_positive_port_certificate",
        lambda value: value,
    )
    monkeypatch.setattr(
        local_types,
        "_expected_carrier_digests",
        lambda *_args, **_kwargs: ("1" * 64, "2" * 64, "3" * 64),
    )
    monkeypatch.setattr(
        detector, "_l8_parent_resource_totals", lambda _port: (0, 0, 0)
    )

    derived_tuple_names = (
        "current_weight_intervals",
        "coordinate_jacobian_intervals",
        "outward_form_diagonal_intervals",
        "pixel_form_diagonal_intervals",
        "outward_minus_pixel_form_diagonal_intervals",
        "production_quadratic_intervals",
        "pixel_form_norm_upper_intervals",
        "production_realization_error_upper_intervals",
        "state_radius_incremental_error_upper_intervals",
        "combined_exact_state_error_upper_intervals",
        "exact_state_pixel_flux_intervals",
    )

    def round_trip(
        candidate: GalerkinLocalPassivePixelForms,
    ) -> GalerkinLocalPassivePixelForms:
        evidence = _evidence(
            detector._expected_local_passive_pixel_evidence(candidate)
        )
        stopped = replace(candidate, **evidence)
        sealed = _make_local_passive_pixel_forms(stopped)
        replayed = _make_local_passive_pixel_forms(
            replace(
                sealed,
                pixel_model_identity_digest="0" * 64,
                pixel_model_evidence_digest="0" * 64,
                certificate_digest="0" * 64,
            )
        )
        assert stored_value_payload(sealed.quadrature_weight_intervals) == (
            stored_value_payload(candidate.quadrature_weight_intervals)
        )
        assert stored_value_payload(sealed.aperture_efficiency_intervals) == (
            stored_value_payload(candidate.aperture_efficiency_intervals)
        )
        for field in fields(GalerkinLocalPassivePixelForms):
            if field.name == "positive_port":
                continue
            assert stored_value_payload(getattr(replayed, field.name)) == (
                stored_value_payload(getattr(sealed, field.name))
            )
        assert len(sealed.quadrature_weight_intervals) == 1
        assert len(sealed.aperture_efficiency_intervals) == 1
        assert all(getattr(sealed, name) == () for name in derived_tuple_names)
        assert sealed.production_traces == ()
        return sealed

    denominator = (1 << 20) + 1
    oversized = _carrier(Fraction(1, denominator))
    bit_stop = round_trip(
        _synthetic_pixel_fallback_candidate(
            quadrature=oversized, maximum_rational_bits=16
        )
    )
    assert GalerkinLocalDetectorFailure(int(bit_stop.failure_mask)) is (
        GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT
    )
    assert bit_stop.work_transcript.exact_work_count == 0
    assert (
        bit_stop.work_transcript.rational_peak_bits == denominator.bit_length()
    )

    budget_stop = round_trip(
        _synthetic_pixel_fallback_candidate(maximum_work=1)
    )
    assert budget_stop.work_transcript.preflight_failed
    assert GalerkinLocalDetectorFailure(int(budget_stop.failure_mask)) is (
        GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED
    )

    parent_stop_candidate = _synthetic_pixel_fallback_candidate()
    object.__setattr__(
        parent_stop_candidate.positive_port,
        "positive_port_eligible",
        np.asarray(False, dtype=np.bool_),
    )
    parent_stop = round_trip(parent_stop_candidate)
    assert not bool(np.asarray(parent_stop.production_evidence_available))

    snapshot = {
        "coordinate_factor_count": 1,
        "pixel_product_count": 0,
        "mode_quadratic_count": 0,
        "ensemble_product_count": 0,
        "response_product_count": 0,
        "production_trace_count": 0,
        "hull_endpoint_count": 0,
        "exact_work_count": 1,
        "rational_peak_bits": 2,
    }

    def raise_failure(failure: GalerkinLocalDetectorFailure) -> object:
        raise detector._DetectorArithmeticError(
            failure,
            1,
            "forced isolated pixel stop",
            work_snapshot=snapshot,
        )

    negative_candidate = _synthetic_pixel_fallback_candidate(
        quadrature=_carrier(-1)
    )
    with monkeypatch.context() as context:
        context.setattr(
            detector,
            "_expected_local_passive_pixel_evidence_core",
            lambda _candidate: raise_failure(
                GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE
            ),
        )
        scientific_stop = round_trip(negative_candidate)
    assert scientific_stop.quadrature_weight_intervals[0].lower == -1
    assert scientific_stop.work_transcript.scientific_failure is (
        GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE
    )

    with monkeypatch.context() as context:
        context.setattr(
            detector,
            "_expected_local_passive_pixel_evidence_core",
            lambda _candidate: raise_failure(
                GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE
            ),
        )
        range_stop = round_trip(_synthetic_pixel_fallback_candidate())
        assert range_stop.work_transcript.arithmetic_failure is (
            GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE
        )


def test_parent_port_route_status_and_source_identity_are_bound() -> None:
    """Bind one genuine p=0 terminal to its projected positive port.

    :see: :func:`ptyrodactyl.galerkin.certify_local_positive_port`
    """
    chain = _public_chain()
    terminal_certificate = chain.terminal
    port = chain.port
    assert bool(terminal_certificate.vacuum_branch_eligible)
    assert port.route is (
        GalerkinLocalPositivePortRoute.PROJECTED_OUTWARD_PROPAGATING
    )
    assert bool(np.asarray(port.positive_port_eligible))
    assert not bool(np.asarray(port.outgoing_radiation_eligible))
    assert np.asarray(port.retained_propagating_mask).tolist() == [True]
    assert np.asarray(port.zero_weight_mask).tolist() == [False]
    assert port.branch_dispositions[0] in (
        GalerkinLocalPositivePortBranchDisposition.PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_PROVABLY_NONZERO,
        GalerkinLocalPositivePortBranchDisposition.PROPAGATING_OUTWARD_RETAINED_INWARD_PROJECTED_UNRESOLVED,
    )
    assert int(np.asarray(port.failure_mask)) == int(
        GalerkinLocalDetectorFailure.NONE
    )
    assert port.target_digest == terminal_certificate.target_digest
    assert port.source_digest == terminal_certificate.source_digest
    assert (
        port.state_identity_digest
        == terminal_certificate.state_identity_digest
    )
    assert port.parent_terminal_identity_digest == (
        terminal_certificate.terminal_identity_digest
    )
    assert port.parent_terminal_evidence_digest == (
        terminal_certificate.terminal_evidence_digest
    )
    assert [trace.stage for trace in port.production_traces] == [
        GalerkinLocalDetectorProductionStage.L8_ROLE_ZERO_AMPLITUDE,
        GalerkinLocalDetectorProductionStage.L8_ROLE_ZERO_AMPLITUDE,
        GalerkinLocalDetectorProductionStage.POSITIVE_PORT_AMPLITUDE,
        GalerkinLocalDetectorProductionStage.POSITIVE_PORT_AMPLITUDE,
    ]
    assert [trace.quantity for trace in port.production_traces] == [
        "l8_role_zero_amplitude.real",
        "l8_role_zero_amplitude.imag",
        "positive_port_amplitude.real",
        "positive_port_amplitude.imag",
    ]
    assert chain.detector_certificate.mode_source_digests == (
        port.source_digest,
    )
    assert chain.detector_certificate.mode_state_identity_digests == (
        port.state_identity_digest,
    )
    assert not hasattr(chain.detector_certificate, "source_certificates")


def test_parent_pixel_q_lvt56_and_trace_chain_match_independent_oracle() -> (
    None
):
    """Rebuild Q, scale squared, and every LVT.56 report from leaf evidence.

    :see: :func:`ptyrodactyl.galerkin.certify_local_passive_pixel_forms`
    """
    chain = _public_chain()
    pixel = chain.pixel
    branch = chain.terminal.branch_evidence
    oracle = _independent_pixel_oracle(chain)
    np.testing.assert_array_equal(
        np.asarray(pixel.positive_port.production_amplitudes),
        np.asarray(branch.frozen_defining_branch_points)[:, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(pixel.positive_port.production_root_realizations),
        np.asarray(branch.frozen_positive_root_realizations),
    )
    assert _raw_intervals(pixel.current_weight_intervals) == oracle.current
    assert (
        pixel.amplitude_scale_interval.lower,
        pixel.amplitude_scale_interval.upper,
    ) == oracle.scale
    assert _raw_intervals(pixel.coordinate_jacobian_intervals) == (
        oracle.jacobians
    )
    assert _raw_intervals(pixel.quadrature_weight_intervals) == (
        oracle.quadrature_raw
    )
    assert _raw_intervals(pixel.aperture_efficiency_intervals) == (
        oracle.aperture_raw
    )
    assert _raw_intervals(pixel.outward_form_diagonal_intervals) == (
        oracle.outward
    )
    assert (
        tuple(
            _raw_intervals(row) for row in pixel.pixel_form_diagonal_intervals
        )
        == oracle.pixels
    )
    assert (
        _raw_intervals(pixel.outward_minus_pixel_form_diagonal_intervals)
        == oracle.margin
    )
    assert (
        (
            pixel.production_outward_quadratic_interval.lower,
            pixel.production_outward_quadratic_interval.upper,
        ),
        *_raw_intervals(pixel.production_quadratic_intervals),
    ) == oracle.production
    assert (
        pixel.outward_form_norm_upper_interval.upper,
        *(value.upper for value in pixel.pixel_form_norm_upper_intervals),
    ) == oracle.form_norms
    assert pixel.production_to_exact_x_amplitude_error_interval.lower == (
        oracle.production_error
    )
    assert pixel.production_to_exact_x_amplitude_error_interval.upper == (
        oracle.production_error
    )
    assert (
        pixel.state_radius_amplitude_error_interval.lower == oracle.state_error
    )
    assert (
        pixel.state_radius_amplitude_error_interval.upper == oracle.state_error
    )
    assert (
        pixel.exact_state_amplitude_error_interval.lower == oracle.total_error
    )
    assert (
        pixel.exact_state_amplitude_error_interval.upper == oracle.total_error
    )
    assert (
        pixel.production_amplitude_norm_interval.lower == oracle.amplitude_norm
    )
    assert (
        pixel.production_amplitude_norm_interval.upper == oracle.amplitude_norm
    )
    assert (
        pixel.outward_production_realization_error_upper_interval.upper,
        *(
            value.upper
            for value in pixel.production_realization_error_upper_intervals
        ),
    ) == oracle.production_realization_errors
    assert (
        pixel.outward_state_radius_incremental_error_upper_interval.upper,
        *(
            value.upper
            for value in pixel.state_radius_incremental_error_upper_intervals
        ),
    ) == oracle.incremental_errors
    assert (
        pixel.outward_combined_exact_state_error_upper_interval.upper,
        *(
            value.upper
            for value in pixel.combined_exact_state_error_upper_intervals
        ),
    ) == oracle.combined_errors
    assert (
        (
            pixel.exact_state_outward_flux_interval.lower,
            pixel.exact_state_outward_flux_interval.upper,
        ),
        *_raw_intervals(pixel.exact_state_pixel_flux_intervals),
    ) == oracle.exact_state
    for produced, increment, combined in zip(
        oracle.production_realization_errors,
        oracle.incremental_errors,
        oracle.combined_errors,
        strict=True,
    ):
        assert produced + increment == combined

    expected_traces = (
        (
            oracle.current_raw,
            oracle.current_points,
            oracle.current,
            GalerkinLocalDetectorProductionStage.COORDINATE_FACTOR,
            "positive_current_weight",
            (1,),
        ),
        (
            (oracle.scale_raw,),
            np.asarray([oracle.scale_point], dtype=np.float64),
            (oracle.scale,),
            GalerkinLocalDetectorProductionStage.COORDINATE_FACTOR,
            "amplitude_scale",
            (),
        ),
        (
            oracle.jacobians_raw,
            oracle.jacobian_points,
            oracle.jacobians,
            GalerkinLocalDetectorProductionStage.COORDINATE_FACTOR,
            "coordinate_jacobian",
            (1,),
        ),
        (
            oracle.quadrature_raw,
            np.asarray(chain.pixel_input.quadrature_weight_points),
            oracle.quadrature,
            GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
            "quadrature_weight",
            (1,),
        ),
        (
            oracle.aperture_raw,
            np.asarray(chain.pixel_input.aperture_efficiency_points),
            oracle.aperture,
            GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
            "aperture_efficiency",
            (1,),
        ),
        (
            oracle.outward_raw,
            oracle.outward_points,
            oracle.outward,
            GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
            "outward_form_diagonal",
            (1,),
        ),
        (
            tuple(value for row in oracle.pixels_raw for value in row),
            oracle.pixel_points,
            tuple(value for row in oracle.pixels for value in row),
            GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
            "pixel_form_diagonal",
            (1, 1),
        ),
        (
            oracle.margin_raw,
            oracle.margin_points,
            oracle.margin,
            GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
            "outward_minus_pixels_diagonal",
            (1,),
        ),
        (
            oracle.amplitude_squared_raw,
            oracle.amplitude_squared_points,
            oracle.amplitude_squared,
            GalerkinLocalDetectorProductionStage.MODE_PRODUCTION_QUADRATIC,
            "retained_positive_port_amplitude_squared",
            (1,),
        ),
        (
            oracle.production_raw,
            oracle.production_points,
            oracle.production,
            GalerkinLocalDetectorProductionStage.MODE_PRODUCTION_QUADRATIC,
            "production_form_quadratic",
            (2,),
        ),
    )
    assert len(pixel.production_traces) == len(expected_traces) == 10
    for trace, expected in zip(
        pixel.production_traces, expected_traces, strict=True
    ):
        raw, points, hulls, stage, quantity, logical_shape = expected
        _assert_trace_matches_leaf_oracle(
            trace,
            raw=raw,
            points=points,
            hulls=hulls,
            stage=stage,
            quantity=quantity,
            logical_shape=logical_shape,
        )
    parent_work, parent_traces, parent_hulls = _independent_l8_resource_totals(
        chain.port
    )
    work = pixel.work_transcript
    assert (
        work.coordinate_factor_count,
        work.pixel_product_count,
        work.mode_quadratic_count,
        work.ensemble_product_count,
        work.response_product_count,
        work.production_trace_count,
        work.hull_endpoint_count,
        work.exact_work_count,
    ) == (1, 1, 2, 0, 0, 10, 22, 76)
    assert int(work.nested_parent_work_count_exact) == parent_work > 0
    assert work.nested_production_trace_count == parent_traces
    assert work.nested_hull_endpoint_count == parent_hulls


def test_parent_detector_pipeline_is_ordered_and_charged_exactly_once() -> (
    None
):
    """Freeze all seven stages, Cj cancellation, and pre-gain censoring.

    :see: :func:`ptyrodactyl.galerkin.\
create_local_censored_poisson_detector_input_manifest`
    :see: :func:`ptyrodactyl.galerkin.certify_local_censored_poisson_detector`
    """
    chain = _public_chain()
    certificate = chain.detector_certificate
    manifest = chain.detector_input
    pixel = _independent_pixel_oracle(chain)
    assert bool(np.asarray(certificate.production_evidence_available))
    assert bool(np.asarray(certificate.detector_eligible))
    assert bool(np.asarray(certificate.likelihood_law_eligible))
    assert certificate.mode_target_digests == (chain.port.target_digest,)
    assert certificate.mode_source_digests == (chain.port.source_digest,)
    assert certificate.mode_port_certificate_digests == (
        chain.port.certificate_digest,
    )
    np.testing.assert_array_equal(
        certificate.response_matrix, manifest.response_matrix
    )
    np.testing.assert_array_equal(
        certificate.pre_gain_background, manifest.pre_gain_background
    )
    np.testing.assert_array_equal(
        certificate.deterministic_gain, manifest.deterministic_gain
    )
    np.testing.assert_array_equal(
        certificate.electronic_offset, manifest.electronic_offset
    )
    np.testing.assert_array_equal(
        certificate.count_ceilings, manifest.count_ceilings
    )
    np.testing.assert_array_equal(certificate.fit_mask, manifest.fit_mask)

    zero_slab = chain.terminal.projection_certificate.zero_slab_certificate
    modes = zero_slab.represented_source_certificate.source.modes
    incident = (
        (
            _dyadic(modes.exact_reduced_flux_lower_bound),
            _dyadic(modes.exact_reduced_flux_upper_bound),
        ),
    )
    incident_points = np.asarray(
        [_dyadic(modes.output_reduced_flux)], dtype=object
    )
    incident_point_array = np.asarray(
        [float(incident_points[0])], dtype=np.float64
    )
    exact_mode_fluxes = (pixel.exact_state[1:],)
    production_mode_raw = (pixel.production[1:],)
    production_mode_points = pixel.production_points[1:][None, :]
    fractions = tuple(
        tuple(_divide_nonnegative(flux, mode_incident) for flux in mode)
        for mode, mode_incident in zip(
            exact_mode_fluxes, incident, strict=True
        )
    )
    weights = tuple(
        Fraction(numerator, denominator)
        for numerator, denominator in zip(
            manifest.ensemble_weight_numerators,
            manifest.ensemble_weight_denominators,
            strict=True,
        )
    )
    assert sum(weights, start=Fraction(0)) == 1
    dose_raw = (
        manifest.incident_electron_count_interval.lower,
        manifest.incident_electron_count_interval.upper,
    )
    exact_ideal = tuple(
        _multiply_nonnegative(
            dose_raw,
            _add_intervals(
                *(
                    _multiply_nonnegative(mode[pixel], _point(weight))
                    for mode, weight in zip(fractions, weights, strict=True)
                )
            ),
        )
        for pixel in range(len(fractions[0]))
    )
    passivity = tuple(
        (
            mode_incident[0] - outward[1],
            mode_incident[1] - outward[0],
        )
        for mode_incident, outward in zip(
            incident, (pixel.exact_state[0],), strict=True
        )
    )
    assert passivity[0][0] >= 0
    assert _raw_intervals(certificate.incident_reduced_flux_intervals) == (
        incident
    )
    assert (
        tuple(
            _raw_intervals(row)
            for row in certificate.mode_exact_state_pixel_flux_intervals
        )
        == exact_mode_fluxes
    )
    assert (
        tuple(
            _raw_intervals(row)
            for row in certificate.mode_production_quadratic_intervals
        )
        == production_mode_raw
    )
    assert (
        tuple(
            _raw_intervals(row)
            for row in certificate.mode_pixel_fraction_intervals
        )
        == fractions
    )
    assert (
        _raw_intervals(certificate.mode_outward_passivity_margin_intervals)
        == passivity
    )
    assert _raw_intervals(certificate.ideal_arrival_mean_intervals) == (
        exact_ideal
    )
    assert tuple(
        _raw_intervals(row)
        for row in certificate.mode_pixel_form_norm_upper_intervals
    ) == (tuple(_point(value) for value in pixel.form_norms[1:]),)
    assert _raw_intervals(
        certificate.mode_production_to_exact_x_amplitude_error_intervals
    ) == (_point(pixel.production_error),)
    assert _raw_intervals(
        certificate.mode_state_radius_amplitude_error_intervals
    ) == (_point(pixel.state_error),)
    assert _raw_intervals(
        certificate.mode_exact_state_amplitude_error_intervals
    ) == (_point(pixel.total_error),)
    assert _raw_intervals(
        certificate.mode_production_amplitude_norm_intervals
    ) == (_point(pixel.amplitude_norm),)
    production_errors = (
        certificate.mode_production_realization_error_upper_intervals
    )
    assert tuple(_raw_intervals(row) for row in production_errors) == (
        tuple(
            _point(value) for value in pixel.production_realization_errors[1:]
        ),
    )
    incremental_errors = (
        certificate.mode_state_radius_incremental_error_upper_intervals
    )
    assert tuple(_raw_intervals(row) for row in incremental_errors) == (
        tuple(_point(value) for value in pixel.incremental_errors[1:]),
    )
    assert tuple(
        _raw_intervals(row)
        for row in certificate.mode_combined_exact_state_error_upper_intervals
    ) == (tuple(_point(value) for value in pixel.combined_errors[1:]),)

    flat_production_fractions_raw = tuple(
        _divide_nonnegative(flux, mode_incident)
        for mode, mode_incident in zip(
            production_mode_raw, incident, strict=True
        )
        for flux in mode
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        production_fraction_points = (
            production_mode_points / incident_point_array[:, None]
        ).astype(np.float64)
    production_fraction_hulls = _union_hulls(
        flat_production_fractions_raw, production_fraction_points
    )
    weight_raw = tuple(_point(value) for value in weights)
    weight_points = np.asarray(
        [float(value) for value in weights], dtype=np.float64
    )
    weight_hulls = _union_hulls(weight_raw, weight_points)
    dose_points = np.asarray(
        [float(np.asarray(manifest.incident_electron_count_point))],
        dtype=np.float64,
    )
    dose_hull = _union_hull(dose_raw, dose_points[0])
    production_ideal_raw = tuple(
        _multiply_nonnegative(
            dose_hull,
            _add_intervals(
                *(
                    _multiply_nonnegative(fraction, weight)
                    for fraction, weight in zip(
                        production_fraction_hulls, weight_hulls, strict=True
                    )
                )
            ),
        )
        for _pixel in range(1)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        production_ideal_points = (
            np.float64(dose_points[0])
            * np.sum(
                weight_points[:, None] * production_fraction_points,
                axis=0,
                dtype=np.float64,
            )
        ).astype(np.float64)
    production_ideal_hulls = _union_hulls(
        production_ideal_raw, production_ideal_points
    )
    response = tuple(
        tuple(_dyadic(value) for value in row)
        for row in np.asarray(manifest.response_matrix, dtype=np.float64)
    )
    for pixel_index in range(len(exact_ideal)):
        assert (
            sum((row[pixel_index] for row in response), start=Fraction(0)) <= 1
        )
    background = tuple(
        _point(_dyadic(value))
        for value in np.asarray(manifest.pre_gain_background, dtype=np.float64)
    )
    exact_pre_gain = tuple(
        _add_intervals(
            *(
                _multiply_nonnegative(_point(coefficient), mean)
                for coefficient, mean in zip(row, ideal, strict=True)
            ),
            background[channel],
        )
        for channel, row in enumerate(response)
        for ideal in (exact_ideal,)
    )
    production_pre_gain_raw = tuple(
        _add_intervals(
            *(
                _multiply_nonnegative(_point(coefficient), mean)
                for coefficient, mean in zip(
                    row, production_ideal_hulls, strict=True
                )
            ),
            background[channel],
        )
        for channel, row in enumerate(response)
    )
    response_array = np.asarray(manifest.response_matrix, dtype=np.float64)
    background_array = np.asarray(
        manifest.pre_gain_background, dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        production_pre_gain_points = (
            response_array @ production_ideal_points + background_array
        ).astype(np.float64)
    production_pre_gain_hulls = _union_hulls(
        production_pre_gain_raw, production_pre_gain_points
    )
    production_pre_gain_singletons = tuple(
        _point(_dyadic(value)) for value in production_pre_gain_points
    )
    assert (
        _raw_intervals(certificate.exact_state_pre_gain_mean_intervals)
        == exact_pre_gain
    )
    assert (
        _raw_intervals(certificate.production_pre_gain_mean_point_intervals)
        == production_pre_gain_singletons
    )

    helper_policy = {
        "maximum_count_ceiling": manifest.maximum_count_ceiling,
        "maximum_work": manifest.maximum_poisson_work,
        "maximum_rational_bits": manifest.maximum_poisson_rational_bits,
        "exp_precision_bits": manifest.exp_precision_bits,
        "maximum_exp_terms": manifest.maximum_exp_terms,
        "maximum_exp_work": manifest.maximum_exp_work,
        "maximum_exp_range_reductions": manifest.maximum_exp_range_reductions,
    }
    ceiling = int(np.asarray(manifest.count_ceilings)[0])
    exact_censored, exact_censored_work = enclose_censored_poisson_mean(
        exact_pre_gain[0], ceiling, **helper_policy
    )
    production_censored_raw, production_censored_work = (
        enclose_censored_poisson_mean(
            production_pre_gain_singletons[0], ceiling, **helper_policy
        )
    )
    assert _raw_intervals(certificate.censored_mean_intervals) == (
        exact_censored,
    )
    assert certificate.censored_mean_transcripts == (exact_censored_work,)
    assert certificate.production_censored_mean_transcripts == (
        production_censored_work,
    )
    exact_mean_work = certificate.censored_mean_transcripts[0]
    point_mean_work = certificate.production_censored_mean_transcripts[0]
    assert exact_mean_work is not None
    assert point_mean_work is not None
    assert exact_mean_work.count_ceiling == point_mean_work.count_ceiling == 2
    assert point_mean_work.observed_count is None
    assert point_mean_work.endpoint_evaluations == 1

    production_censored_points = np.asarray(
        [
            np.float64(
                float(
                    (production_censored_raw[0] + production_censored_raw[1])
                    / 2
                )
            )
        ],
        dtype=np.float64,
    )
    production_censored_hulls = (
        _union_hull(production_censored_raw, production_censored_points[0]),
    )
    gains = tuple(
        _dyadic(value)
        for value in np.asarray(manifest.deterministic_gain, dtype=np.float64)
    )
    offsets = tuple(
        _dyadic(value)
        for value in np.asarray(manifest.electronic_offset, dtype=np.float64)
    )
    exact_digitized = tuple(
        (
            gain * exact_censored[0] + offset,
            gain * exact_censored[1] + offset,
        )
        for gain, offset in zip(gains, offsets, strict=True)
    )
    production_digitized_raw = tuple(
        (
            gain * censored[0] + offset,
            gain * censored[1] + offset,
        )
        for gain, offset, censored in zip(
            gains, offsets, production_censored_hulls, strict=True
        )
    )
    with np.errstate(over="ignore", invalid="ignore"):
        production_digitized_points = (
            np.asarray(manifest.deterministic_gain, dtype=np.float64)
            * production_censored_points
            + np.asarray(manifest.electronic_offset, dtype=np.float64)
        ).astype(np.float64)
    production_digitized_hulls = _union_hulls(
        production_digitized_raw, production_digitized_points
    )
    assert _raw_intervals(certificate.expected_digitized_mean_intervals) == (
        exact_digitized
    )

    expected_traces = (
        (
            flat_production_fractions_raw,
            production_fraction_points,
            production_fraction_hulls,
            GalerkinLocalDetectorProductionStage.MODE_PIXEL_FRACTION,
            "mode_pixel_fraction",
            (1, 1),
        ),
        (
            weight_raw,
            weight_points,
            weight_hulls,
            GalerkinLocalDetectorProductionStage.ENSEMBLE_WEIGHT,
            "ensemble_weight",
            (1,),
        ),
        (
            (dose_raw,),
            dose_points,
            (dose_hull,),
            GalerkinLocalDetectorProductionStage.INCIDENT_DOSE,
            "incident_electron_count",
            (),
        ),
        (
            production_ideal_raw,
            production_ideal_points,
            production_ideal_hulls,
            GalerkinLocalDetectorProductionStage.IDEAL_ARRIVAL_MEAN,
            "ideal_arrival_mean",
            (1,),
        ),
        (
            production_pre_gain_raw,
            production_pre_gain_points,
            production_pre_gain_hulls,
            GalerkinLocalDetectorProductionStage.PRE_GAIN_RESPONSE_MEAN,
            "production_pre_gain_mean",
            (1,),
        ),
        (
            (production_censored_raw,),
            production_censored_points,
            production_censored_hulls,
            GalerkinLocalDetectorProductionStage.CENSORED_COUNT_MEAN,
            "production_censored_count_mean",
            (1,),
        ),
        (
            production_digitized_raw,
            production_digitized_points,
            production_digitized_hulls,
            GalerkinLocalDetectorProductionStage.POST_CENSOR_DIGITIZED_MEAN,
            "production_digitized_mean",
            (1,),
        ),
    )
    assert len(certificate.production_traces) == len(expected_traces) == 7
    for trace, expected in zip(
        certificate.production_traces, expected_traces, strict=True
    ):
        raw, points, hulls, stage, quantity, logical_shape = expected
        _assert_trace_matches_leaf_oracle(
            trace,
            raw=raw,
            points=points,
            hulls=hulls,
            stage=stage,
            quantity=quantity,
            logical_shape=logical_shape,
        )
    work = certificate.work_transcript
    assert (
        work.coordinate_factor_count,
        work.pixel_product_count,
        work.mode_quadratic_count,
        work.ensemble_product_count,
        work.response_product_count,
        work.production_trace_count,
        work.hull_endpoint_count,
        work.exact_work_count,
    ) == (0, 0, 0, 2, 2, 7, 14, 53)
    parent_work, parent_traces, parent_hulls = _independent_l8_resource_totals(
        chain.port
    )
    assert int(work.nested_parent_work_count_exact) == parent_work + 76
    assert work.nested_production_trace_count == parent_traces + 10
    assert work.nested_hull_endpoint_count == parent_hulls + 22
    helper_work = sum(
        transcript.exact_work_count
        + sum(value.exact_work_count for value in transcript.exp_transcripts)
        + sum(value.exact_work_count for value in transcript.log_transcripts)
        for transcript in (exact_censored_work, production_censored_work)
    )
    assert int(work.nested_helper_work_count_exact) == helper_work

    likelihood = chain.likelihood
    assert bool(np.asarray(likelihood.likelihood_evidence_available))
    assert bool(np.asarray(likelihood.likelihood_law_eligible))
    assert bool(np.asarray(likelihood.nll_eligible))
    assert (
        GalerkinLocalDetectorFailure(int(np.asarray(likelihood.failure_mask)))
        is GalerkinLocalDetectorFailure.NONE
    )
    admitted = (
        min(exact_pre_gain[0][0], production_pre_gain_singletons[0][0]),
        max(exact_pre_gain[0][1], production_pre_gain_singletons[0][1]),
    )
    probability_common = {
        **helper_policy,
    }
    production_probability, production_probability_work = (
        enclose_censored_poisson_probability(
            production_pre_gain_singletons[0],
            1,
            ceiling,
            **probability_common,
        )
    )
    admitted_probability, admitted_probability_work = (
        enclose_censored_poisson_probability(
            admitted,
            1,
            ceiling,
            **probability_common,
        )
    )
    nll_policy = {
        "log_precision_bits": 48,
        "maximum_log_terms": 256,
        "maximum_log_work": 100_000,
        "maximum_log_range_reductions": 64,
    }
    assert (
        likelihood.log_precision_bits,
        likelihood.maximum_log_terms,
        likelihood.maximum_log_work,
        likelihood.maximum_log_range_reductions,
    ) == (48, 256, 100_000, 64)
    production_nll, production_nll_work = enclose_censored_poisson_nll(
        production_pre_gain_singletons[0],
        1,
        ceiling,
        **probability_common,
        **nll_policy,
    )
    admitted_nll, admitted_nll_work = enclose_censored_poisson_nll(
        admitted,
        1,
        ceiling,
        **probability_common,
        **nll_policy,
    )
    assert _raw_intervals(
        likelihood.admitted_pre_gain_mean_hull_intervals
    ) == (admitted,)
    assert _raw_intervals(likelihood.admitted_hull_probability_intervals) == (
        admitted_probability,
    )
    floor = likelihood.fitted_probability_positive_floor_intervals[0]
    assert floor is not None
    assert (floor.lower, floor.upper) == _point(admitted_probability[0])
    assert likelihood.production_probability_transcripts == (
        production_probability_work,
    )
    assert likelihood.admitted_hull_probability_transcripts == (
        admitted_probability_work,
    )
    assert likelihood.production_nll_transcripts == (production_nll_work,)
    assert likelihood.admitted_hull_nll_transcripts == (admitted_nll_work,)
    assert likelihood.production_probability_failures == (None,)
    assert likelihood.admitted_hull_probability_failures == (None,)
    assert likelihood.production_nll_failures == (None,)
    assert likelihood.admitted_hull_nll_failures == (None,)
    admitted_nll_stored = likelihood.admitted_hull_nll_intervals[0]
    assert admitted_nll_stored is not None
    assert (
        admitted_nll_stored.lower,
        admitted_nll_stored.upper,
    ) == admitted_nll
    total_nll = likelihood.total_nll_interval
    assert total_nll is not None
    assert (total_nll.lower, total_nll.upper) == admitted_nll

    production_probability_point = np.float64(
        float((production_probability[0] + production_probability[1]) / 2)
    )
    production_probability_hull = _union_hull(
        production_probability, production_probability_point
    )
    assert _raw_intervals(
        likelihood.production_probability_point_intervals
    ) == (_point(_dyadic(production_probability_point)),)
    production_nll_point = np.float64(
        float((production_nll[0] + production_nll[1]) / 2)
    )
    production_nll_hull = _union_hull(production_nll, production_nll_point)
    production_nll_stored = likelihood.production_nll_point_intervals[0]
    assert production_nll_stored is not None
    assert (
        production_nll_stored.lower,
        production_nll_stored.upper,
    ) == _point(_dyadic(production_nll_point))
    likelihood_trace_expected = (
        (
            (production_probability,),
            np.asarray([production_probability_point], dtype=np.float64),
            (production_probability_hull,),
            GalerkinLocalDetectorProductionStage.CENSORED_PROBABILITY,
            "production_censored_probability",
            (1,),
        ),
        (
            (production_nll,),
            np.asarray([production_nll_point], dtype=np.float64),
            (production_nll_hull,),
            GalerkinLocalDetectorProductionStage.CENSORED_NLL,
            "production_nll.channel_0",
            (),
        ),
    )
    assert (
        len(likelihood.production_traces)
        == len(likelihood_trace_expected)
        == 2
    )
    for trace, expected in zip(
        likelihood.production_traces, likelihood_trace_expected, strict=True
    ):
        raw, points, hulls, stage, quantity, logical_shape = expected
        _assert_trace_matches_leaf_oracle(
            trace,
            raw=raw,
            points=points,
            hulls=hulls,
            stage=stage,
            quantity=quantity,
            logical_shape=logical_shape,
        )
    likelihood_work = likelihood.work_transcript
    assert (
        likelihood_work.coordinate_factor_count,
        likelihood_work.pixel_product_count,
        likelihood_work.mode_quadratic_count,
        likelihood_work.ensemble_product_count,
        likelihood_work.response_product_count,
        likelihood_work.production_trace_count,
        likelihood_work.hull_endpoint_count,
        likelihood_work.exact_work_count,
    ) == (0, 0, 0, 0, 0, 2, 4, 10)
    assert int(likelihood_work.nested_parent_work_count_exact) == (
        parent_work + 76 + 53 + helper_work
    )
    assert (
        likelihood_work.nested_production_trace_count == parent_traces + 10 + 7
    )
    assert likelihood_work.nested_hull_endpoint_count == parent_hulls + 22 + 14
    likelihood_helper_work = sum(
        transcript.exact_work_count
        + sum(value.exact_work_count for value in transcript.exp_transcripts)
        + sum(value.exact_work_count for value in transcript.log_transcripts)
        for transcript in (
            production_probability_work,
            admitted_probability_work,
            production_nll_work,
            admitted_nll_work,
        )
    )
    assert int(likelihood_work.nested_helper_work_count_exact) == (
        likelihood_helper_work
    )


def test_parent_typed_input_and_scientific_fallbacks_raise_or_remain_evidence() -> (  # noqa: E501
    None
):
    """Reject malformed parents and keep physical stops in explicit masks."""
    chain = _public_chain()
    with pytest.raises(TypeCheckError):
        detector.prepare_local_positive_port_certificate(
            cast(GalerkinLocalPositivePortCertificate, object()),
            route=chain.pixel_input.route,
            disposition=chain.pixel_input.terminal_disposition,
            maximum_state_error=chain.pixel_input.maximum_state_error,
        )
    malformed_manifest = replace(
        chain.pixel_input,
        route=cast(GalerkinLocalPositivePortRoute, "not-a-route"),
    )
    with pytest.raises(TypeError, match="route"):
        _make_local_passive_pixel_input_manifest(malformed_manifest)
    with pytest.raises(ValueError, match="pixel parent digest disagrees"):
        _make_local_passive_pixel_forms(
            replace(chain.pixel, parent_port_certificate_digest="0" * 64)
        )
    with pytest.raises(TypeCheckError):
        detector.certify_local_passive_pixel_forms(
            chain.port,
            input_manifest=cast(
                GalerkinLocalPassivePixelInputManifest, object()
            ),
        )
    malformed_port = replace(
        chain.port,
        terminal_certificate=cast(
            GalerkinLocalVacuumTerminalCertificate, object()
        ),
    )
    with pytest.raises(TypeCheckError):
        detector.prepare_local_positive_port_certificate(
            malformed_port,
            route=chain.pixel_input.route,
            disposition=chain.pixel_input.terminal_disposition,
            maximum_state_error=chain.pixel_input.maximum_state_error,
        )
    malformed_pixel = replace(
        chain.pixel,
        positive_port=cast(GalerkinLocalPositivePortCertificate, object()),
    )
    with pytest.raises(TypeCheckError):
        detector.prepare_local_passive_pixel_forms(
            malformed_pixel, input_manifest=chain.pixel_input
        )
    malformed_detector_manifest = replace(
        chain.detector_input,
        pixel_inputs=(cast(GalerkinLocalPassivePixelInputManifest, object()),),
    )
    with pytest.raises(TypeCheckError):
        detector.prepare_local_censored_poisson_detector(
            chain.detector_certificate,
            input_manifest=malformed_detector_manifest,
        )
    malformed_detector = replace(
        chain.detector_certificate,
        pixel_forms=(cast(GalerkinLocalPassivePixelForms, object()),),
    )
    with pytest.raises(TypeCheckError):
        detector.prepare_local_censored_poisson_detector(
            malformed_detector, input_manifest=chain.detector_input
        )
    malformed_likelihood = replace(
        chain.likelihood,
        detector=cast(GalerkinLocalCensoredPoissonDetector, object()),
    )
    with pytest.raises(TypeCheckError):
        detector.prepare_local_censored_poisson_likelihood(
            malformed_likelihood,
            detector_input_manifest=chain.detector_input,
            observed_counts=np.asarray([1], dtype=np.int64),
        )
    assert not hasattr(GalerkinLocalDetectorFailure, "PARENT_REPLAY_FAILURE")

    bad_pixel = replace(
        chain.pixel,
        quadrature_weights=jnp.asarray([-1.0], dtype=jnp.float64),
    )
    pixel_evidence = _evidence(
        detector._expected_local_passive_pixel_evidence(bad_pixel)
    )
    assert pixel_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE
    )
    assert pixel_evidence["production_evidence_available"] is False

    bad_response = replace(
        chain.detector_certificate,
        response_matrix=jnp.asarray([[-1.0]], dtype=jnp.float64),
    )
    detector_evidence = _evidence(
        detector._expected_local_censored_poisson_detector(bad_response)
    )
    assert detector_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.RESPONSE_NONPOSITIVE
    )
    bad_count = replace(
        chain.likelihood, observed_counts=jnp.asarray([3], dtype=jnp.int64)
    )
    likelihood_evidence = _evidence(
        detector._expected_local_censored_poisson_likelihood(bad_count)
    )
    assert likelihood_evidence["failure_mask"] == int(
        GalerkinLocalDetectorFailure.COUNT_DOMAIN_INVALID
    )
    assert likelihood_evidence["likelihood_evidence_available"] is False


def test_parent_resource_helper_and_range_stops_preserve_partial_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay typed preflight, bit, binary64, and nested helper stops."""
    chain = _public_chain()
    pixel_budget = replace(
        chain.pixel,
        work_transcript=_empty_work(maximum_work=1),
    )
    preflight = _evidence(
        detector._expected_local_passive_pixel_evidence(pixel_budget)
    )
    assert preflight["failure_mask"] == int(
        GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED
    )
    assert preflight["work_transcript"].preflight_failed
    assert preflight["work_transcript"].exact_work_count == 0

    over_bits = replace(
        chain.detector_certificate,
        incident_electron_count=_carrier(Fraction(1, 1 << 32)),
        work_transcript=_empty_work(maximum_rational_bits=16),
    )
    bit_stop = _evidence(
        detector._expected_local_censored_poisson_detector(over_bits)
    )
    assert bit_stop["failure_mask"] == int(
        GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT
    )
    assert bit_stop["work_transcript"].rational_peak_bits > 16

    range_candidate = replace(
        chain.detector_certificate,
        incident_electron_count_point=jnp.asarray(np.inf, dtype=jnp.float64),
    )
    range_stop = _evidence(
        detector._expected_local_censored_poisson_detector(range_candidate)
    )
    assert range_stop["failure_mask"] == int(
        GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE
    )

    def fail_probability(*_args: object, **_kwargs: object) -> object:
        raise CensoredPoissonEnclosureError(
            CensoredPoissonEnclosureFailure.EXPONENTIAL_ENCLOSURE_FAILURE,
            3,
            "forced parent likelihood helper stop",
            attempted_exact_work_count=3,
            nested_kernel="exp",
            nested_failure=EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
            nested_exact_work_count=2,
            nested_attempted_exact_work_count=2,
        )

    monkeypatch.setattr(
        detector, "enclose_censored_poisson_probability", fail_probability
    )
    helper_stop = _evidence(
        detector._expected_local_censored_poisson_likelihood(chain.likelihood)
    )
    failure = GalerkinLocalDetectorFailure(helper_stop["failure_mask"])
    assert failure == (
        GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
        | GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
    )
    production = helper_stop["production_probability_failures"][0]
    admitted = helper_stop["admitted_hull_probability_failures"][0]
    assert (
        production.call
        is GalerkinLocalDetectorHelperCall.PRODUCTION_PROBABILITY
    )
    assert admitted.call is (
        GalerkinLocalDetectorHelperCall.ADMITTED_HULL_PROBABILITY
    )
    assert production.channel_index == admitted.channel_index == 0
    assert (
        production.nested_failure
        is EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
    )
    assert helper_stop["work_transcript"].arithmetic_failure is (
        GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
    )


def test_public_l9_chain_prepares_and_rejects_policy_and_coherent_forgeries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross replays five and six as isolated policy and binding forgeries.

    :see: :func:`ptyrodactyl.galerkin.prepare_local_positive_port_certificate`
    :see: :func:`ptyrodactyl.galerkin.prepare_local_passive_pixel_forms`
    :see: :func:`ptyrodactyl.galerkin.prepare_local_censored_poisson_detector`
    :see: :func:`ptyrodactyl.galerkin.\
enclose_local_censored_poisson_likelihood`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_censored_poisson_likelihood`
    """
    chain = _public_chain()
    altered_pixel_input = _make_local_passive_pixel_input_manifest(
        replace(
            chain.pixel_input,
            maximum_terminal_direct_pairs=(
                chain.pixel_input.maximum_terminal_direct_pairs + 1
            ),
        )
    )
    altered_detector_input = (
        _make_local_censored_poisson_detector_input_manifest(
            replace(
                chain.detector_input,
                pixel_inputs=(altered_pixel_input,),
            )
        )
    )
    assert (
        altered_pixel_input.manifest_digest
        != chain.pixel_input.manifest_digest
    )
    assert (
        altered_detector_input.manifest_digest
        != chain.detector_input.manifest_digest
    )
    assert altered_pixel_input.maximum_terminal_direct_pairs == (
        chain.pixel_input.maximum_terminal_direct_pairs + 1
    )
    with pytest.raises(ValueError, match="complete replay"):
        detector.prepare_local_censored_poisson_likelihood(
            chain.likelihood,
            detector_input_manifest=altered_detector_input,
            observed_counts=np.asarray([1], dtype=np.int64),
            maximum_detector_work=1_000_000,
            maximum_detector_rational_bits=262_144,
            log_precision_bits=48,
            maximum_log_terms=256,
            maximum_log_work=100_000,
            maximum_log_range_reductions=64,
        )

    forged_source = "a" * 64
    forged_state = "b" * 64
    forged_terminal = replace(
        chain.terminal,
        source_digest=forged_source,
        state_identity_digest=forged_state,
    )
    assert forged_terminal.terminal_identity_digest == (
        chain.terminal.terminal_identity_digest
    )
    assert forged_terminal.terminal_evidence_digest == (
        chain.terminal.terminal_evidence_digest
    )
    forged_port = _reseal_port(
        replace(
            chain.port,
            terminal_certificate=forged_terminal,
            source_digest=forged_source,
            state_identity_digest=forged_state,
        )
    )
    assert forged_port.parent_terminal_identity_digest == (
        chain.port.parent_terminal_identity_digest
    )
    assert forged_port.parent_terminal_evidence_digest == (
        chain.port.parent_terminal_evidence_digest
    )
    forged_pixel = _reseal_pixel(
        replace(
            chain.pixel,
            positive_port=forged_port,
            parent_port_certificate_digest=forged_port.certificate_digest,
        )
    )
    target_digests = (forged_port.target_digest,)
    source_digests = (forged_source,)
    state_digests = (forged_state,)
    port_digests = (forged_port.certificate_digest,)
    pixel_digests = (forged_pixel.pixel_model_evidence_digest,)
    mode_binding = sha256(
        {
            "domain": "ptyrodactyl.local_detector.mode_state_binding.v1",
            "target_digests": target_digests,
            "source_digests": source_digests,
            "state_identity_digests": state_digests,
            "state_radius_intervals": stored_value_payload(
                chain.detector_certificate.mode_state_radius_intervals
            ),
            "state_radius_provenance_digests": (
                chain.detector_certificate.mode_state_radius_provenance_digests
            ),
            "port_certificate_digests": port_digests,
            "pixel_evidence_digests": pixel_digests,
        }
    )
    forged_detector = _reseal_detector(
        replace(
            chain.detector_certificate,
            pixel_forms=(forged_pixel,),
            mode_target_digests=target_digests,
            mode_source_digests=source_digests,
            mode_state_identity_digests=state_digests,
            mode_port_certificate_digests=port_digests,
            mode_pixel_evidence_digests=pixel_digests,
            mode_state_binding_digest=mode_binding,
        )
    )
    forged = _reseal_likelihood(
        replace(
            chain.likelihood,
            detector=forged_detector,
            parent_detector_certificate_digest=forged_detector.certificate_digest,
        )
    )
    assert forged.detector.mode_source_digests == (forged_source,)
    assert forged.detector.mode_state_identity_digests == (forged_state,)
    assert forged.detector.mode_state_binding_digest == mode_binding
    assert forged.parent_detector_certificate_digest == (
        forged_detector.certificate_digest
    )
    assert forged.certificate_digest != chain.likelihood.certificate_digest
    original_prepare_port = cast(
        Any, detector.prepare_local_positive_port_certificate
    )
    traversed: list[tuple[str, str, str, str]] = []

    def capture_prepare_port(
        certificate: GalerkinLocalPositivePortCertificate,
        **kwargs: object,
    ) -> GalerkinLocalPositivePortCertificate:
        terminal = certificate.terminal_certificate
        traversed.append(
            (
                terminal.source_digest,
                terminal.state_identity_digest,
                terminal.terminal_identity_digest,
                terminal.terminal_evidence_digest,
            )
        )
        return original_prepare_port(certificate, **kwargs)

    monkeypatch.setattr(
        detector,
        "prepare_local_positive_port_certificate",
        capture_prepare_port,
    )
    with pytest.raises(ValueError, match="complete replay"):
        detector.prepare_local_censored_poisson_likelihood(
            forged,
            detector_input_manifest=chain.detector_input,
            observed_counts=np.asarray([1], dtype=np.int64),
            maximum_detector_work=1_000_000,
            maximum_detector_rational_bits=262_144,
            log_precision_bits=48,
            maximum_log_terms=256,
            maximum_log_work=100_000,
            maximum_log_range_reductions=64,
        )
    assert traversed == [
        (
            forged_source,
            forged_state,
            chain.terminal.terminal_identity_digest,
            chain.terminal.terminal_evidence_digest,
        )
    ]


__all__: list[str] = []
