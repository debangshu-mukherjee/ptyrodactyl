r"""Compose the local RM-S4 positive-port detector ladder.

Extended Summary
----------------
This leaf is the sole public composition boundary for the local RM-S4
positive port, passive ideal-pixel forms, calibrated fixed detector response,
and censored-count likelihood.  Its public producers replay each immediate
parent under independently supplied primitive manifests and resource policy;
the bounded private exact arithmetic remains available to lightweight
falsification tests.

The first route retains outward propagating amplitudes, assigns exact zero
count weight to isolated decaying evanescent and admissible constant-grazing
channels, and projects the backward propagating branch with zero weight.  A
projected backward branch is not an outgoing-radiation condition.  Physical
current, coordinate Jacobian, quadrature, and aperture factors remain
separate, and deterministic gain is downstream of the pre-gain likelihood.

Routine Listings
----------------
:func:`certify_local_censored_poisson_detector`
    Replay all pixels and certify one fixed censored-Poisson detector.
:func:`certify_local_passive_pixel_forms`
    Replay one positive port and certify its primitive passive pixels.
:func:`certify_local_positive_port`
    Replay L8 and compose one explicit projected or outgoing port.
:func:`create_local_censored_poisson_detector_input_manifest`
    Authenticate primitive detector inputs and every pixel replay input.
:func:`create_local_passive_pixel_input_manifest`
    Authenticate primitive pixel inputs and independent upstream policy.
:func:`enclose_local_censored_poisson_likelihood`
    Enclose full-channel probabilities and fit-only pre-gain NLL.
:func:`prepare_local_censored_poisson_detector`
    Replay independent detector inputs and exact-compare every field.
:func:`prepare_local_censored_poisson_likelihood`
    Replay a likelihood from independent inputs and exact-compare it.
:func:`prepare_local_passive_pixel_forms`
    Replay independently supplied pixel inputs and exact-compare storage.
:func:`prepare_local_positive_port_certificate`
    Replay an independently specified L8 port and exact-compare storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple, cast
from jax.core import Tracer

from ptyrodactyl._tools import (
    CensoredPoissonEnclosureError,
    CensoredPoissonWorkTranscript,
    enclose_censored_poisson_mean,
    enclose_censored_poisson_nll,
    enclose_censored_poisson_probability,
    fraction_upper_float,
    mathematical_pi_interval,
    sqrt_fraction_upper,
)
from ptyrodactyl.types.local_detector_types import (
    _POSITIVE_PORT_BRANCH_SCOPE,
    _POSITIVE_PORT_COMPLETION_SCOPE,
    _POSITIVE_PORT_EXACT_STATE_SCOPE,
    _POSITIVE_PORT_ROOT_AUDIT_SCOPE,
    GalerkinLocalCensoredPoissonDetector,
    GalerkinLocalCensoredPoissonDetectorInputManifest,
    GalerkinLocalCensoredPoissonLikelihood,
    GalerkinLocalDetectorCoordinateConvention,
    GalerkinLocalDetectorFailure,
    GalerkinLocalDetectorHelperCall,
    GalerkinLocalDetectorLikelihoodStage,
    GalerkinLocalDetectorProductionStage,
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalDetectorRealProductionTrace,
    GalerkinLocalDetectorWorkTranscript,
    GalerkinLocalPassivePixelForms,
    GalerkinLocalPassivePixelInputManifest,
    GalerkinLocalPositivePortCertificate,
    GalerkinLocalPositivePortRoute,
    _expected_mode_state_binding,
    _expected_port_branches,
    _make_local_censored_poisson_detector,
    _make_local_censored_poisson_detector_candidate,
    _make_local_censored_poisson_detector_input_manifest,
    _make_local_censored_poisson_detector_input_manifest_candidate,
    _make_local_censored_poisson_likelihood,
    _make_local_censored_poisson_likelihood_candidate,
    _make_local_detector_helper_failure_evidence,
    _make_local_detector_rational_interval,
    _make_local_detector_real_production_trace,
    _make_local_detector_work_transcript,
    _make_local_passive_pixel_forms,
    _make_local_passive_pixel_forms_candidate,
    _make_local_passive_pixel_input_manifest,
    _make_local_passive_pixel_input_manifest_candidate,
    _make_local_positive_port_candidate,
    _make_local_positive_port_certificate,
    _validate_local_censored_poisson_detector_input_manifest,
    _validate_local_passive_pixel_input_manifest,
)
from ptyrodactyl.types.local_vacuum_terminal_types import (
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
)

from .local_vacuum_terminal import prepare_local_vacuum_terminal_certificate

type _Interval = Tuple[Fraction, Fraction]
type _Intervals = Tuple[_Interval, ...]
type _OptionalIntervals = Tuple[_Interval | None, ...]
type _ModePixelIntervals = Tuple[_Intervals, ...]
type _PixelForms = Tuple[_Intervals, ...]
type _PoissonReports = tuple[
    _Intervals,
    _Intervals,
    tuple[_Interval | None, ...],
    tuple[CensoredPoissonWorkTranscript, ...],
    tuple[CensoredPoissonWorkTranscript, ...],
    tuple[CensoredPoissonWorkTranscript | None, ...],
]
type _LVT56Reports = tuple[
    _Intervals,
    _Intervals,
    _Intervals,
    _Intervals,
    _Intervals,
    _Intervals,
]

_DEFAULT_MAXIMUM_RATIONAL_BITS: int = 262_144
_DEFAULT_MAXIMUM_WORK: int = 1_000_000
_DEFAULT_L8_DIRECT_WORK: int = 2_000_000
_DEFAULT_L8_ROOT_WORK: int = 64
_DEFAULT_L8_PRECISION_BITS: int = 160
_DEFAULT_L8_TERMS: int = 4096
_DEFAULT_L8_RANGE_REDUCTIONS: int = 4096
_DEFAULT_MAXIMUM_COUNT_CEILING: int = 4096
_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_INTERVAL_ENDPOINT_COUNT: int = 2
_MAXIMUM_SIGNED_INT64: int = (1 << 63) - 1
_ONE: Fraction = Fraction(1)
_ZERO: Fraction = Fraction(0)


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced leaves at the detector host boundary.

    Parameters
    ----------
    value : object
        Required canonical input.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError(
            "local detector composition requires concrete host values"
        )


def _require_public_array(
    value: object,
    *,
    dtype: np.dtype[np.generic],
    name: str,
) -> np.ndarray:
    """PRIVATE: Reject public semantic-array coercion or truncation.

    Parameters
    ----------
    value : object
        Required canonical input.
    dtype : np.dtype[np.generic]
        Required canonical input.
    name : str
        Required canonical input.

    Returns
    -------
    array : np.ndarray
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    if not hasattr(value, "dtype") or not hasattr(value, "shape"):
        raise TypeError(f"{name} must be an array with dtype {dtype.name}")
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    return array


class _DetectorArithmeticError(ArithmeticError):
    """Report one typed bounded private detector-arithmetic failure."""

    failure: GalerkinLocalDetectorFailure
    exact_work_count: int
    work_snapshot: dict[str, int]
    production_traces: tuple[GalerkinLocalDetectorRealProductionTrace, ...]
    failure_context: dict[str, object]

    def __init__(
        self,
        failure: GalerkinLocalDetectorFailure,
        exact_work_count: int,
        message: str,
        *,
        work_snapshot: dict[str, int] | None = None,
        production_traces: tuple[
            GalerkinLocalDetectorRealProductionTrace, ...
        ] = (),
        failure_context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.exact_work_count = exact_work_count
        self.work_snapshot = {} if work_snapshot is None else work_snapshot
        self.production_traces = production_traces
        self.failure_context = (
            {} if failure_context is None else failure_context
        )


@dataclass
class _DetectorLedger:
    """Bound exact local detector arithmetic and retained rational size."""

    algorithm: str
    maximum_work: int
    maximum_rational_bits: int
    coordinate_factor_count: int = 0
    pixel_product_count: int = 0
    mode_quadratic_count: int = 0
    ensemble_product_count: int = 0
    response_product_count: int = 0
    production_trace_count: int = 0
    hull_endpoint_count: int = 0
    production_traces: list[GalerkinLocalDetectorRealProductionTrace] = field(
        default_factory=list
    )
    failure_context: dict[str, object] = field(default_factory=dict)
    exact_work_count: int = 0
    rational_peak_bits: int = 0

    def fail(
        self, failure: GalerkinLocalDetectorFailure, message: str
    ) -> None:
        """Raise one typed arithmetic failure at completed local work."""
        raise _DetectorArithmeticError(
            failure,
            self.exact_work_count,
            message,
            work_snapshot=self.snapshot(),
            production_traces=tuple(self.production_traces),
            failure_context=dict(self.failure_context),
        )

    def snapshot(self) -> dict[str, int]:
        """Return immutable scalar evidence for the current work prefix."""
        return {
            "coordinate_factor_count": self.coordinate_factor_count,
            "pixel_product_count": self.pixel_product_count,
            "mode_quadratic_count": self.mode_quadratic_count,
            "ensemble_product_count": self.ensemble_product_count,
            "response_product_count": self.response_product_count,
            "production_trace_count": self.production_trace_count,
            "hull_endpoint_count": self.hull_endpoint_count,
            "exact_work_count": self.exact_work_count,
            "rational_peak_bits": self.rational_peak_bits,
        }

    def charge(self) -> None:
        """Charge one exact binary rational operation before issuing it."""
        attempted = self.exact_work_count + 1
        if attempted > self.maximum_work:
            raise _DetectorArithmeticError(
                GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED,
                attempted,
                "local detector exact-work budget exceeded",
                work_snapshot=self.snapshot(),
                production_traces=tuple(self.production_traces),
                failure_context=dict(self.failure_context),
            )
        self.exact_work_count = attempted

    def retain(self, value: Fraction) -> Fraction:
        """Retain one exact rational under the independent bit policy."""
        bits = max(
            abs(value.numerator).bit_length(),
            value.denominator.bit_length(),
        )
        self.rational_peak_bits = max(self.rational_peak_bits, bits)
        if bits > self.maximum_rational_bits:
            self.fail(
                GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT,
                "local detector rational exceeds its retained-bit limit",
            )
        result: Fraction = value
        return result  # noqa: RET504

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        """Add and retain two exact rationals."""
        left = self.retain(left)
        right = self.retain(right)
        self.charge()
        result: Fraction = self.retain(left + right)
        return result  # noqa: RET504

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        """Subtract and retain two exact rationals."""
        left = self.retain(left)
        right = self.retain(right)
        self.charge()
        result: Fraction = self.retain(left - right)
        return result  # noqa: RET504

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        """Multiply and retain two exact rationals."""
        left = self.retain(left)
        right = self.retain(right)
        self.charge()
        result: Fraction = self.retain(left * right)
        return result  # noqa: RET504

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        """Divide and retain two exact rationals."""
        numerator = self.retain(numerator)
        denominator = self.retain(denominator)
        if denominator == _ZERO:
            raise ZeroDivisionError("local detector denominator is zero")
        self.charge()
        result: Fraction = self.retain(numerator / denominator)
        return result  # noqa: RET504

    def root_upper(self, value: Fraction) -> Fraction:
        """Charge and retain one deterministic exact square-root upper."""
        value = self.retain(value)
        if value < _ZERO:
            raise ValueError("local detector square-root radicand is negative")
        self.charge()
        return self.retain(sqrt_fraction_upper(value))

    def multiply_nonnegative(
        self, left: _Interval, right: _Interval
    ) -> _Interval:
        """Multiply two checked nonnegative exact intervals."""
        result: _Interval = (
            self.multiply(left[0], right[0]),
            self.multiply(left[1], right[1]),
        )
        return result  # noqa: RET504

    def add_intervals(self, left: _Interval, right: _Interval) -> _Interval:
        """Add two checked exact intervals endpointwise."""
        result: _Interval = (
            self.add(left[0], right[0]),
            self.add(left[1], right[1]),
        )
        return result  # noqa: RET504

    def divide_positive(
        self, numerator: _Interval, denominator: _Interval
    ) -> _Interval:
        """Divide a nonnegative interval by one strictly positive interval."""
        result: _Interval = (
            self.divide(numerator[0], denominator[1]),
            self.divide(numerator[1], denominator[0]),
        )
        return result  # noqa: RET504

    def transcript(self) -> GalerkinLocalDetectorWorkTranscript:
        """Freeze one bounded exact detector work transcript."""
        result = _make_local_detector_work_transcript(
            algorithm=self.algorithm,
            maximum_work=self.maximum_work,
            maximum_rational_bits=self.maximum_rational_bits,
            coordinate_factor_count=self.coordinate_factor_count,
            pixel_product_count=self.pixel_product_count,
            mode_quadratic_count=self.mode_quadratic_count,
            ensemble_product_count=self.ensemble_product_count,
            response_product_count=self.response_product_count,
            production_trace_count=self.production_trace_count,
            hull_endpoint_count=self.hull_endpoint_count,
            exact_work_count=self.exact_work_count,
            rational_peak_bits=self.rational_peak_bits,
        )
        return result  # noqa: RET504


def _checked_policy(
    maximum_work: object = _DEFAULT_MAXIMUM_WORK,
    maximum_rational_bits: object = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> _DetectorLedger:
    """PRIVATE: Validate and initialize one local detector work policy.

    Parameters
    ----------
    maximum_work : object
        Positive local exact-work ceiling; default is 1,000,000.
    maximum_rational_bits : object
        Retained numerator/denominator bit ceiling; default is 262,144.

    Returns
    -------
    ledger : _DetectorLedger
        Fresh bounded exact detector ledger.

    Raises
    ------
    TypeError
        If either policy is not exactly a Python integer.
    ValueError
        If a policy is outside its implementation range.
    """
    if type(maximum_work) is not int or type(maximum_rational_bits) is not int:
        raise TypeError("local detector policies must be Python integers")
    if (
        maximum_work <= 0
        or maximum_work > _MAXIMUM_SIGNED_INT64
        or maximum_rational_bits <= 1
        or maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
    ):
        raise ValueError("local detector policies are outside their ranges")
    ledger = _DetectorLedger(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=maximum_work,
        maximum_rational_bits=maximum_rational_bits,
    )
    return ledger  # noqa: RET504


def _checked_interval(
    value: object,
    ledger: _DetectorLedger,
    *,
    require_nonnegative: bool,
    require_positive: bool,
) -> _Interval:
    """PRIVATE: Validate one ordered exact detector interval.

    Parameters
    ----------
    value : object
        Submitted two-Fraction interval.
    ledger : _DetectorLedger
        Active local rational-size ledger.
    require_nonnegative : bool
        Whether the lower endpoint must be nonnegative.
    require_positive : bool
        Whether the lower endpoint must be strictly positive.

    Returns
    -------
    interval : _Interval
        Ordered retained exact interval.

    Raises
    ------
    TypeError
        If the submission is not exactly two Fractions.
    ValueError
        If ordering or the requested sign predicate fails.
    _DetectorArithmeticError
        If an endpoint exceeds the rational-size policy.
    """
    if (
        not isinstance(value, tuple)
        or len(value) != _INTERVAL_ENDPOINT_COUNT
        or any(not isinstance(endpoint, Fraction) for endpoint in value)
    ):
        raise TypeError("detector interval must contain exactly two Fractions")
    submitted = cast(Tuple[Fraction, Fraction], value)
    lower = ledger.retain(submitted[0])
    upper = ledger.retain(submitted[1])
    if lower > upper:
        raise ValueError("detector interval endpoints must be ordered")
    if require_nonnegative and lower < _ZERO:
        raise ValueError("detector interval must be nonnegative")
    if require_positive and lower <= _ZERO:
        raise ValueError("detector interval must be strictly positive")
    interval: _Interval = (lower, upper)
    return interval  # noqa: RET504


def _two_pi_interval(ledger: _DetectorLedger) -> _Interval:
    """PRIVATE: Convert the verified binary64 pi bracket to exact ``2*pi``.

    Parameters
    ----------
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    interval : _Interval
        Exact rational enclosure of mathematical ``2*pi``.

    Raises
    ------
    _DetectorArithmeticError
        If work or rational-size policies fail.
    """
    pi_lower, pi_upper = mathematical_pi_interval()
    lower = ledger.retain(Fraction.from_float(float(np.asarray(pi_lower))))
    upper = ledger.retain(Fraction.from_float(float(np.asarray(pi_upper))))
    two = ledger.retain(Fraction(2))
    interval: _Interval = (
        ledger.multiply(two, lower),
        ledger.multiply(two, upper),
    )
    return interval  # noqa: RET504


def _coordinate_factors(
    convention: GalerkinLocalDetectorCoordinateConvention,
    vacuum_wavenumber: _Interval,
    normal_root_intervals: _OptionalIntervals,
    retained_propagating_mask: tuple[bool, ...],
    ledger: _DetectorLedger,
) -> tuple[_Interval, _Intervals]:
    """PRIVATE: Derive amplitude scaling and coordinate Jacobians.

    Parameters
    ----------
    convention : GalerkinLocalDetectorCoordinateConvention
        Declared RM-S4 coordinate/amplitude pair.
    vacuum_wavenumber : _Interval
        Strictly positive exact ``k0`` interval.
    normal_root_intervals : _OptionalIntervals
        Exact root intervals in terminal-fiber order.  Retained propagating
        entries are strictly positive; zero-weight entries may instead be
        positive evanescent magnitudes, exact grazing zeros, or ``None``.
    retained_propagating_mask : tuple[bool, ...]
        Exact declaration of the positive outward propagating sector.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    amplitude_scale : _Interval
        Convention-defined scaling from angular amplitude ``A_kappa``.
    jacobians : _Intervals
        Per-node coordinate-Jacobian intervals before current weighting.

    Raises
    ------
    TypeError
        If ``convention`` or interval structures are invalid.
    ValueError
        If lengths disagree or a retained propagating root is not positive.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.
    """
    if not isinstance(convention, GalerkinLocalDetectorCoordinateConvention):
        raise TypeError("coordinate convention has the wrong enum type")
    k0 = _checked_interval(
        vacuum_wavenumber,
        ledger,
        require_nonnegative=True,
        require_positive=True,
    )
    if not isinstance(normal_root_intervals, tuple) or not isinstance(
        retained_propagating_mask, tuple
    ):
        raise TypeError("normal roots and retained mask must be tuples")
    if len(normal_root_intervals) == 0 or len(normal_root_intervals) != len(
        retained_propagating_mask
    ):
        raise ValueError("normal roots and retained mask must have one size")
    if any(type(value) is not bool for value in retained_propagating_mask):
        raise TypeError("retained propagating mask entries must be booleans")
    roots: _OptionalIntervals = tuple(
        None
        if root is None
        else _checked_interval(
            root,
            ledger,
            require_nonnegative=True,
            require_positive=retained,
        )
        for root, retained in zip(
            normal_root_intervals, retained_propagating_mask, strict=True
        )
    )
    if any(
        retained and root is None
        for root, retained in zip(
            roots, retained_propagating_mask, strict=True
        )
    ):
        raise ValueError("retained propagating fibers require positive roots")
    one: _Interval = (_ONE, _ONE)
    zero: _Interval = (_ZERO, _ZERO)
    if convention is (
        GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES
    ):
        amplitude_scale = one
        jacobians: _Intervals = tuple(
            one if retained else zero for retained in retained_propagating_mask
        )
    elif convention is (
        GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_CYCLIC_COORDINATES
    ):
        amplitude_scale = one
        two_pi = _two_pi_interval(ledger)
        squared = ledger.multiply_nonnegative(two_pi, two_pi)
        jacobians = tuple(
            squared if retained else zero
            for retained in retained_propagating_mask
        )
    elif convention is (
        GalerkinLocalDetectorCoordinateConvention.NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES
    ):
        amplitude_scale = _two_pi_interval(ledger)
        jacobians = tuple(
            one if retained else zero for retained in retained_propagating_mask
        )
    else:
        amplitude_scale = one
        jacobians = tuple(
            ledger.multiply_nonnegative(k0, cast(_Interval, root))
            if retained
            else zero
            for root, retained in zip(
                roots, retained_propagating_mask, strict=True
            )
        )
    ledger.coordinate_factor_count += len(roots)
    result: tuple[_Interval, _Intervals] = (amplitude_scale, jacobians)
    return result  # noqa: RET504


def _positive_current_weights(
    normal_root_intervals: _OptionalIntervals,
    retained_propagating_mask: tuple[bool, ...],
    ledger: _DetectorLedger,
) -> _Intervals:
    """PRIVATE: Form exact positive-port current weights from root evidence.

    Parameters
    ----------
    normal_root_intervals : _OptionalIntervals
        Exact classified root intervals in terminal-fiber order.
    retained_propagating_mask : tuple[bool, ...]
        Exact positive outward propagating sector.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    current_weights : _Intervals
        Replayed positive roots on retained fibers and canonical exact zero on
        every evanescent, grazing, or unclassified zero-weight fiber.

    Raises
    ------
    TypeError
        If tuple, mask, or interval structures have invalid exact types.
    ValueError
        If lengths disagree or a retained root is absent/nonpositive.
    _DetectorArithmeticError
        If a rational-size limit fails.
    """
    if not isinstance(normal_root_intervals, tuple) or not isinstance(
        retained_propagating_mask, tuple
    ):
        raise TypeError("normal roots and retained mask must be tuples")
    if len(normal_root_intervals) == 0 or len(normal_root_intervals) != len(
        retained_propagating_mask
    ):
        raise ValueError("normal roots and retained mask must have one size")
    if any(type(value) is not bool for value in retained_propagating_mask):
        raise TypeError("retained propagating mask entries must be booleans")
    zero: _Interval = (_ZERO, _ZERO)
    weights: list[_Interval] = []
    for root, retained in zip(
        normal_root_intervals, retained_propagating_mask, strict=True
    ):
        if not retained:
            if root is not None:
                _checked_interval(
                    root,
                    ledger,
                    require_nonnegative=True,
                    require_positive=False,
                )
            weights.append(zero)
            continue
        if root is None:
            raise ValueError("retained propagating fibers require roots")
        weights.append(
            _checked_interval(
                root,
                ledger,
                require_nonnegative=True,
                require_positive=True,
            )
        )
    return tuple(weights)


def _pixel_form_diagonals(  # noqa: PLR0913
    current_weights: _Intervals,
    coordinate_jacobians: _Intervals,
    amplitude_scale: _Interval,
    quadrature_weights: _Intervals,
    aperture_efficiencies: _Intervals,
    node_to_pixel: tuple[int, ...],
    pixel_count: int,
    ledger: _DetectorLedger,
) -> tuple[_Intervals, _PixelForms, _Intervals]:
    """PRIVATE: Build positive passive diagonal ideal-pixel forms.

    Parameters
    ----------
    current_weights : _Intervals
        Nonnegative outward-current weights, separate from every Jacobian.
    coordinate_jacobians : _Intervals
        Nonnegative coordinate-Jacobian weights.
    amplitude_scale : _Interval
        Strictly positive amplitude-coordinate scale. Its square is absorbed
        wholly into every outward and pixel form so production quadratics,
        Q norms, LVT.56 errors, and passivity use the identical Q.
    quadrature_weights : _Intervals
        Nonnegative detector quadrature weights.
    aperture_efficiencies : _Intervals
        Node efficiencies contained in ``[0,1]``.
    node_to_pixel : tuple[int, ...]
        Exact disjoint pixel index per node, or ``-1`` for explicit loss.
    pixel_count : int
        Positive physical ideal-pixel count.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    outward_diagonal : _Intervals
        Complete positive outward form before aperture/loss.
    pixel_diagonals : _PixelForms
        Per-pixel positive diagonal forms with structural passivity.
    outward_minus_pixels : _Intervals
        Exact nonnegative diagonal enclosure of ``Q_out - sum_p Q_p`` formed
        through the shared base and complementary aperture efficiency.

    Raises
    ------
    TypeError
        If tuple or exact-integer structures are invalid.
    ValueError
        If lengths, signs, efficiency, or pixel membership are invalid.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.
    """
    if type(pixel_count) is not int or pixel_count <= 0:
        raise TypeError("pixel_count must be a positive Python integer")
    sequences = (
        current_weights,
        coordinate_jacobians,
        quadrature_weights,
        aperture_efficiencies,
        node_to_pixel,
    )
    if any(not isinstance(sequence, tuple) for sequence in sequences):
        raise TypeError("pixel-form inputs must be tuples")
    node_count = len(current_weights)
    if node_count == 0 or any(len(value) != node_count for value in sequences):
        raise ValueError("pixel-form inputs must have one nonempty node size")
    if any(type(pixel) is not int for pixel in node_to_pixel):
        raise TypeError("node_to_pixel entries must be Python integers")
    if any(pixel < -1 or pixel >= pixel_count for pixel in node_to_pixel):
        raise ValueError("node_to_pixel entries lie outside the pixel domain")

    checked_current = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in current_weights
    )
    checked_jacobians = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in coordinate_jacobians
    )
    checked_scale = _checked_interval(
        amplitude_scale,
        ledger,
        require_nonnegative=True,
        require_positive=True,
    )
    scale_squared = ledger.multiply_nonnegative(checked_scale, checked_scale)
    checked_quadrature = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in quadrature_weights
    )
    checked_efficiencies = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in aperture_efficiencies
    )
    if any(value[1] > _ONE for value in checked_efficiencies):
        raise ValueError("aperture efficiencies must be contained in [0,1]")

    outward: list[_Interval] = []
    outward_minus_pixels: list[_Interval] = []
    pixels: list[list[_Interval]] = [
        [(_ZERO, _ZERO) for _ in range(node_count)] for _ in range(pixel_count)
    ]
    for index in range(node_count):
        base = ledger.multiply_nonnegative(
            checked_current[index], checked_jacobians[index]
        )
        base = ledger.multiply_nonnegative(base, scale_squared)
        base = ledger.multiply_nonnegative(base, checked_quadrature[index])
        outward.append(base)
        pixel = node_to_pixel[index]
        if pixel >= 0:
            pixels[pixel][index] = ledger.multiply_nonnegative(
                base, checked_efficiencies[index]
            )
            complement: _Interval = (
                ledger.subtract(_ONE, checked_efficiencies[index][1]),
                ledger.subtract(_ONE, checked_efficiencies[index][0]),
            )
            outward_minus_pixels.append(
                ledger.multiply_nonnegative(base, complement)
            )
        else:
            outward_minus_pixels.append(base)
        ledger.pixel_product_count += 1
    result: tuple[_Intervals, _PixelForms, _Intervals] = (
        tuple(outward),
        tuple(tuple(row) for row in pixels),
        tuple(outward_minus_pixels),
    )
    return result  # noqa: RET504


def _validate_ensemble_weights(
    weights: tuple[Fraction, ...], ledger: _DetectorLedger
) -> tuple[Fraction, ...]:
    """PRIVATE: Validate exact nonnegative ensemble weights summing to one.

    Parameters
    ----------
    weights : tuple[Fraction, ...]
        Candidate mutually incoherent population weights.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    checked : tuple[Fraction, ...]
        Retained exact weights with exact unit sum.

    Raises
    ------
    TypeError
        If the submission is not a nonempty tuple of Fractions.
    ValueError
        If a weight is negative or the exact sum differs from one.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.
    """
    if (
        not isinstance(weights, tuple)
        or len(weights) == 0
        or any(not isinstance(weight, Fraction) for weight in weights)
    ):
        raise TypeError("ensemble weights must be a nonempty Fraction tuple")
    checked = tuple(ledger.retain(weight) for weight in weights)
    if any(weight < _ZERO for weight in checked):
        raise ValueError("ensemble weights must be nonnegative")
    total = _ZERO
    for weight in checked:
        total = ledger.add(total, weight)
    if total != _ONE:
        raise ValueError("ensemble weights must sum exactly to one")
    return checked


def _mode_pixel_fluxes(
    amplitude_squared: _Intervals,
    pixel_diagonals: _PixelForms,
    ledger: _DetectorLedger,
) -> _Intervals:
    """PRIVATE: Evaluate one mode's quadratic diagonal pixel forms.

    Parameters
    ----------
    amplitude_squared : _Intervals
        Per-node nonnegative squared-amplitude enclosures after convention
        scaling and coherent terminal propagation.
    pixel_diagonals : _PixelForms
        Per-pixel nonnegative diagonal form intervals.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    pixel_fluxes : _Intervals
        Per-pixel reduced outward-flux enclosures for this one mode.

    Raises
    ------
    TypeError
        If interval structures are not tuples.
    ValueError
        If shapes or nonnegativity predicates fail.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.
    """
    if not isinstance(amplitude_squared, tuple) or not isinstance(
        pixel_diagonals, tuple
    ):
        raise TypeError("mode quadratic inputs must be tuples")
    node_count = len(amplitude_squared)
    if node_count == 0 or any(
        not isinstance(row, tuple) or len(row) != node_count
        for row in pixel_diagonals
    ):
        raise ValueError("mode quadratic shapes are inconsistent")
    amplitudes = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in amplitude_squared
    )
    pixel_fluxes: list[_Interval] = []
    for row in pixel_diagonals:
        total: _Interval = (_ZERO, _ZERO)
        for amplitude, weight in zip(amplitudes, row, strict=True):
            checked_weight = _checked_interval(
                weight,
                ledger,
                require_nonnegative=True,
                require_positive=False,
            )
            term = ledger.multiply_nonnegative(amplitude, checked_weight)
            total = ledger.add_intervals(total, term)
            ledger.mode_quadratic_count += 1
        pixel_fluxes.append(total)
    return tuple(pixel_fluxes)


def _lvt56_quadratic_reports(
    production_amplitude_squared: _Intervals,
    form_diagonals: _PixelForms,
    production_to_exact_x_amplitude_error: _Interval,
    state_radius_amplitude_error: _Interval,
    exact_state_amplitude_error: _Interval,
    production_amplitude_norm: _Interval,
    ledger: _DetectorLedger,
) -> _LVT56Reports:
    """PRIVATE: Enclose exact-state positive quadratics by LVT.56.

    Parameters
    ----------
    production_amplitude_squared : _Intervals
        Exact enclosures of the frozen production ``|a_hat_h|^2`` values.
    form_diagonals : _PixelForms
        Positive diagonal forms formed directly from replayed exact root,
        coordinate-Jacobian, quadrature, and aperture intervals.
    production_to_exact_x_amplitude_error : _Interval
        Nonnegative enclosure whose upper endpoint is the frozen production
        point-to-exact-submitted-state amplitude error ``E_prod``.
    state_radius_amplitude_error : _Interval
        Nonnegative enclosure whose upper endpoint is the exact-terminal-map
        transfer of the L6 state radius ``E_state``.  This is a distinct
        informational report and is not added to either LVT.56 radius.
    exact_state_amplitude_error : _Interval
        Nonnegative enclosure for total ``E_a`` from L8's exact-state
        role-zero amplitude-error route.  It must be ordered no smaller than
        ``E_prod``; it is not reconstructed from ``E_prod + E_state``.
    production_amplitude_norm : _Interval
        Nonnegative enclosure whose upper endpoint bounds ``||a_hat||``.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    result : _LVT56Reports
        Production quadratics, verified diagonal ``||Q||`` upper reports,
        production-realization ``f(E_prod)``, state-radius incremental
        ``f(E_a) - f(E_prod)``, combined ``f(E_a)``, and nonnegative final
        exact-state quadratic enclosures, in that order, where
        ``f(E) = ||Q|| E (2 ||a_hat|| + E)``.

    Raises
    ------
    TypeError
        If interval or form structures have invalid exact types.
    ValueError
        If shapes, ordering, or nonnegative predicates fail.
    _DetectorArithmeticError
        If exact work or rational-size policies fail.

    Notes
    -----
    Root-realization audit error is deliberately absent.  Coefficient
    uncertainty is enclosed by the exact replayed root intervals in ``Q``.
    The production-realization term and the remaining state-radius increment
    sum exactly to the combined LVT.56 error, so ``E_a`` is applied once.
    """
    production_error = _checked_interval(
        production_to_exact_x_amplitude_error,
        ledger,
        require_nonnegative=True,
        require_positive=False,
    )
    _checked_interval(
        state_radius_amplitude_error,
        ledger,
        require_nonnegative=True,
        require_positive=False,
    )
    total_error = _checked_interval(
        exact_state_amplitude_error,
        ledger,
        require_nonnegative=True,
        require_positive=False,
    )
    amplitude_norm = _checked_interval(
        production_amplitude_norm,
        ledger,
        require_nonnegative=True,
        require_positive=False,
    )
    if total_error[0] < production_error[1]:
        raise ValueError(
            "exact-state amplitude error must not be smaller than "
            "production error"
        )
    production = _mode_pixel_fluxes(
        production_amplitude_squared, form_diagonals, ledger
    )
    checked_forms = tuple(
        tuple(
            _checked_interval(
                value,
                ledger,
                require_nonnegative=True,
                require_positive=False,
            )
            for value in row
        )
        for row in form_diagonals
    )
    two = ledger.retain(Fraction(2))
    twice_norm = ledger.multiply(two, amplitude_norm[1])
    production_norm_plus_error = ledger.add(twice_norm, production_error[1])
    total_norm_plus_error = ledger.add(twice_norm, total_error[1])
    form_norms: list[_Interval] = []
    production_errors: list[_Interval] = []
    state_incremental_errors: list[_Interval] = []
    combined_errors: list[_Interval] = []
    exact_state: list[_Interval] = []
    for quadratic, row in zip(production, checked_forms, strict=True):
        form_norm = max((value[1] for value in row), default=_ZERO)
        q_times_production_error = ledger.multiply(
            form_norm, production_error[1]
        )
        production_error_upper = ledger.multiply(
            q_times_production_error, production_norm_plus_error
        )
        q_times_total_error = ledger.multiply(form_norm, total_error[1])
        combined_error_upper = ledger.multiply(
            q_times_total_error, total_norm_plus_error
        )
        state_incremental_error_upper = ledger.subtract(
            combined_error_upper, production_error_upper
        )
        if state_incremental_error_upper < _ZERO:
            raise ValueError(
                "state-radius incremental quadratic error is negative"
            )
        form_norms.append((form_norm, form_norm))
        production_errors.append(
            (production_error_upper, production_error_upper)
        )
        state_incremental_errors.append(
            (
                state_incremental_error_upper,
                state_incremental_error_upper,
            )
        )
        combined_errors.append((combined_error_upper, combined_error_upper))
        lower = max(_ZERO, ledger.subtract(quadratic[0], combined_error_upper))
        upper = ledger.add(quadratic[1], combined_error_upper)
        exact_state.append((lower, upper))
    result: _LVT56Reports = (
        production,
        tuple(form_norms),
        tuple(production_errors),
        tuple(state_incremental_errors),
        tuple(combined_errors),
        tuple(exact_state),
    )
    return result  # noqa: RET504


def _outward_passivity_margins(
    incident_reduced_fluxes: _Intervals,
    exact_state_outward_fluxes: _Intervals,
    ledger: _DetectorLedger,
) -> _Intervals:
    """PRIVATE: Prove state-specific outward flux does not exceed incidence.

    Parameters
    ----------
    incident_reduced_fluxes : _Intervals
        Strictly positive source-derived reduced incident flux enclosures.
    exact_state_outward_fluxes : _Intervals
        Nonnegative LVT.56-enclosed exact-state ``Q_out`` quadratics.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    margins : _Intervals
        Nonnegative enclosures of incident flux minus outward flux.

    Raises
    ------
    TypeError
        If either input is not a tuple of exact intervals.
    ValueError
        If sizes disagree or a uniformly nonnegative margin is unproved.
    _DetectorArithmeticError
        If exact work or rational-size policies fail.
    """
    if not isinstance(incident_reduced_fluxes, tuple) or not isinstance(
        exact_state_outward_fluxes, tuple
    ):
        raise TypeError("outward passivity inputs must be tuples")
    if len(incident_reduced_fluxes) == 0 or len(
        incident_reduced_fluxes
    ) != len(exact_state_outward_fluxes):
        raise ValueError("outward passivity mode counts must agree")
    margins: list[_Interval] = []
    for incident, outward in zip(
        incident_reduced_fluxes, exact_state_outward_fluxes, strict=True
    ):
        checked_incident = _checked_interval(
            incident,
            ledger,
            require_nonnegative=True,
            require_positive=True,
        )
        checked_outward = _checked_interval(
            outward,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        margin = (
            ledger.subtract(checked_incident[0], checked_outward[1]),
            ledger.subtract(checked_incident[1], checked_outward[0]),
        )
        if margin[0] < _ZERO:
            raise ValueError("outward passivity margin is not nonnegative")
        margins.append(margin)
    return tuple(margins)


def _normalize_mix_and_dose(
    mode_pixel_fluxes: _ModePixelIntervals,
    incident_reduced_fluxes: _Intervals,
    ensemble_weights: tuple[Fraction, ...],
    incident_electron_count: _Interval,
    ledger: _DetectorLedger,
) -> tuple[_ModePixelIntervals, _Intervals]:
    """PRIVATE: Normalize each mode, mix after quadratics, and apply dose.

    Parameters
    ----------
    mode_pixel_fluxes : _ModePixelIntervals
        Per-mode reduced pixel-flux intervals after each coherent quadratic.
    incident_reduced_fluxes : _Intervals
        Strictly positive per-mode source-derived reduced incident fluxes.
    ensemble_weights : tuple[Fraction, ...]
        Exact nonnegative mutually incoherent populations summing to one.
    incident_electron_count : _Interval
        Nonnegative calibrated exposure count applied exactly once.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    mode_fractions : _ModePixelIntervals
        Dimensionless per-mode pixel-fraction intervals after ``C_j`` cancels.
    ideal_arrival_means : _Intervals
        Dose-scaled incoherent ideal-arrival mean intervals.

    Raises
    ------
    TypeError
        If mode, flux, weight, or dose structures are invalid.
    ValueError
        If mode sizes, flux positivity, or weights are invalid.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.

    Notes
    -----
    The physical numerator is ``C_j`` times reduced pixel flux and the
    physical denominator is ``C_j`` times reduced incident flux.  ``C_j``
    cancels before the dimensionless ratio, and dose is then applied once.
    """
    if not isinstance(mode_pixel_fluxes, tuple) or not isinstance(
        incident_reduced_fluxes, tuple
    ):
        raise TypeError("mode flux inputs must be tuples")
    mode_count = len(mode_pixel_fluxes)
    if mode_count == 0 or len(incident_reduced_fluxes) != mode_count:
        raise ValueError("mode and incident-flux counts must agree")
    pixel_count = len(mode_pixel_fluxes[0])
    if pixel_count == 0 or any(
        not isinstance(mode, tuple) or len(mode) != pixel_count
        for mode in mode_pixel_fluxes
    ):
        raise ValueError("all modes must share one nonempty pixel count")
    weights = _validate_ensemble_weights(ensemble_weights, ledger)
    if len(weights) != mode_count:
        raise ValueError("ensemble weight count must equal mode count")
    dose = _checked_interval(
        incident_electron_count,
        ledger,
        require_nonnegative=True,
        require_positive=False,
    )
    incident = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=True,
        )
        for value in incident_reduced_fluxes
    )
    fractions: list[_Intervals] = []
    for mode, denominator in zip(mode_pixel_fluxes, incident, strict=True):
        fractions.append(
            tuple(
                ledger.divide_positive(
                    _checked_interval(
                        value,
                        ledger,
                        require_nonnegative=True,
                        require_positive=False,
                    ),
                    denominator,
                )
                for value in mode
            )
        )
    ideal: list[_Interval] = []
    for pixel in range(pixel_count):
        mixed: _Interval = (_ZERO, _ZERO)
        for mode, weight in zip(fractions, weights, strict=True):
            weighted = ledger.multiply_nonnegative(
                mode[pixel], (weight, weight)
            )
            mixed = ledger.add_intervals(mixed, weighted)
            ledger.ensemble_product_count += 1
        ideal.append(ledger.multiply_nonnegative(dose, mixed))
    result: tuple[_ModePixelIntervals, _Intervals] = (
        tuple(fractions),
        tuple(ideal),
    )
    return result  # noqa: RET504


def _apply_nonnegative_response(
    ideal_arrival_means: _Intervals,
    response_matrix: tuple[tuple[Fraction, ...], ...],
    pre_gain_background: _Intervals,
    ledger: _DetectorLedger,
) -> _Intervals:
    """PRIVATE: Apply one fixed nonnegative substochastic mean response.

    Parameters
    ----------
    ideal_arrival_means : _Intervals
        Nonnegative physical ideal-pixel arrival means.
    response_matrix : tuple[tuple[Fraction, ...], ...]
        Fixed channel-by-pixel nonnegative routing matrix.
    pre_gain_background : _Intervals
        Nonnegative pre-gain count-background intervals per channel.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    pre_gain_means : _Intervals
        Nonnegative response-plus-background channel means.

    Raises
    ------
    TypeError
        If interval or response structures have invalid exact types.
    ValueError
        If dimensions, nonnegativity, or column-substochasticity fail.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.
    """
    if not isinstance(ideal_arrival_means, tuple) or not isinstance(
        response_matrix, tuple
    ):
        raise TypeError("response inputs must be tuples")
    pixel_count = len(ideal_arrival_means)
    channel_count = len(response_matrix)
    if pixel_count == 0 or channel_count == 0:
        raise ValueError("response dimensions must be nonempty")
    if any(
        not isinstance(row, tuple)
        or len(row) != pixel_count
        or any(not isinstance(value, Fraction) for value in row)
        for row in response_matrix
    ):
        raise TypeError("response matrix must contain exact Fraction rows")
    if len(pre_gain_background) != channel_count:
        raise ValueError("background count must equal response channels")
    response = tuple(
        tuple(ledger.retain(value) for value in row) for row in response_matrix
    )
    if any(value < _ZERO for row in response for value in row):
        raise ValueError("response matrix must be nonnegative")
    for pixel in range(pixel_count):
        column_sum = _ZERO
        for channel in range(channel_count):
            column_sum = ledger.add(column_sum, response[channel][pixel])
        if column_sum > _ONE:
            raise ValueError("response matrix columns must sum to at most one")
    means = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in ideal_arrival_means
    )
    backgrounds = tuple(
        _checked_interval(
            value,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        for value in pre_gain_background
    )
    output: list[_Interval] = []
    for row, background in zip(response, backgrounds, strict=True):
        total = background
        for weight, mean in zip(row, means, strict=True):
            term = ledger.multiply_nonnegative((weight, weight), mean)
            total = ledger.add_intervals(total, term)
            ledger.response_product_count += 1
        output.append(total)
    return tuple(output)


def _apply_gain_and_offset(
    censored_means: _Intervals,
    deterministic_gain: tuple[Fraction, ...],
    electronic_offset: tuple[Fraction, ...],
    ledger: _DetectorLedger,
) -> _Intervals:
    """PRIVATE: Apply deterministic gain and offset after censoring.

    Parameters
    ----------
    censored_means : _Intervals
        Nonnegative expected pre-gain censored counts.
    deterministic_gain : tuple[Fraction, ...]
        Strictly positive deterministic per-channel gains.
    electronic_offset : tuple[Fraction, ...]
        Finite exact post-gain electronic offsets.
    ledger : _DetectorLedger
        Active bounded exact detector ledger.

    Returns
    -------
    digitized_means : _Intervals
        Post-censor deterministic-gain expected readouts.

    Raises
    ------
    TypeError
        If gain or offset structures are not exact Fraction tuples.
    ValueError
        If lengths differ or a gain is not strictly positive.
    _DetectorArithmeticError
        If exact work or rational-size limits fail.
    """
    sequences = (censored_means, deterministic_gain, electronic_offset)
    if any(not isinstance(sequence, tuple) for sequence in sequences):
        raise TypeError("gain-stage inputs must be tuples")
    channel_count = len(censored_means)
    if channel_count == 0 or any(
        len(sequence) != channel_count for sequence in sequences
    ):
        raise ValueError("gain-stage channel counts must agree")
    if any(
        not isinstance(value, Fraction)
        for sequence in (deterministic_gain, electronic_offset)
        for value in sequence
    ):
        raise TypeError("gain and offset entries must be Fractions")
    gains = tuple(ledger.retain(value) for value in deterministic_gain)
    offsets = tuple(ledger.retain(value) for value in electronic_offset)
    if any(gain <= _ZERO for gain in gains):
        raise ValueError("deterministic gain must be strictly positive")
    output: list[_Interval] = []
    for mean, gain, offset in zip(censored_means, gains, offsets, strict=True):
        checked = _checked_interval(
            mean,
            ledger,
            require_nonnegative=True,
            require_positive=False,
        )
        scaled = ledger.multiply_nonnegative(checked, (gain, gain))
        output.append(
            (
                ledger.add(scaled[0], offset),
                ledger.add(scaled[1], offset),
            )
        )
    return tuple(output)


def _censored_poisson_reports(  # noqa: PLR0913
    pre_gain_means: _Intervals,
    observed_counts: tuple[int, ...],
    count_ceilings: tuple[int, ...],
    fit_mask: tuple[bool, ...],
    *,
    maximum_count_ceiling: int,
    maximum_poisson_work: int,
    maximum_rational_bits: int,
    exp_precision_bits: int,
    maximum_exp_terms: int,
    maximum_exp_work: int,
    maximum_exp_range_reductions: int,
    log_precision_bits: int,
    maximum_log_terms: int,
    maximum_log_work: int,
    maximum_log_range_reductions: int,
) -> _PoissonReports:
    """PRIVATE: Enclose pre-gain censored probabilities, means, and NLLs.

    Parameters
    ----------
    pre_gain_means : _Intervals
        Nonnegative fixed pre-gain Poisson mean intervals.
    observed_counts : tuple[int, ...]
        Exact observed censored symbols.
    count_ceilings : tuple[int, ...]
        Exact nonnegative channel censoring ceilings.
    fit_mask : tuple[bool, ...]
        Fixed parameter- and data-independent likelihood mask.
    maximum_count_ceiling : int
        Maximum admitted channel ceiling.
    maximum_poisson_work : int
        Maximum local polynomial work per helper invocation.
    maximum_rational_bits : int
        Shared exact rational-size policy.
    exp_precision_bits : int
        Nested exponential precision policy.
    maximum_exp_terms : int
        Nested exponential term policy.
    maximum_exp_work : int
        Nested exponential work policy.
    maximum_exp_range_reductions : int
        Nested exponential reduction policy.
    log_precision_bits : int
        Nested logarithm precision policy.
    maximum_log_terms : int
        Nested logarithm term policy.
    maximum_log_work : int
        Nested logarithm work policy.
    maximum_log_range_reductions : int
        Nested logarithm reduction policy.

    Returns
    -------
    result : _PoissonReports
        Probability enclosures, censored-mean enclosures, optional fitted NLL
        enclosures, and their probability, mean, and optional NLL transcripts,
        in that order.

    Raises
    ------
    TypeError
        If channel structures or exact Python count types are invalid.
    ValueError
        If channel lengths or count domains are invalid.
    CensoredPoissonEnclosureError
        If a probability/mean helper or an eligible NLL helper fails.

    Notes
    -----
    The stochastic law consumes ``pre_gain_means``.  Gain and electronic
    offset are downstream deterministic readout operations and cannot alter
    any probability or NLL in this function.
    """
    sequences = (pre_gain_means, observed_counts, count_ceilings, fit_mask)
    if any(not isinstance(sequence, tuple) for sequence in sequences):
        raise TypeError("censored-Poisson channel inputs must be tuples")
    channel_count = len(pre_gain_means)
    if channel_count == 0 or any(
        len(sequence) != channel_count for sequence in sequences
    ):
        raise ValueError("censored-Poisson channel counts must agree")
    if any(
        type(value) is not int for value in observed_counts + count_ceilings
    ):
        raise TypeError("censored counts and ceilings must be Python integers")
    if any(type(value) is not bool for value in fit_mask):
        raise TypeError("fit mask entries must be Python booleans")

    probabilities: list[_Interval] = []
    censored_means: list[_Interval] = []
    nlls: list[_Interval | None] = []
    probability_transcripts: list[CensoredPoissonWorkTranscript] = []
    mean_transcripts: list[CensoredPoissonWorkTranscript] = []
    nll_transcripts: list[CensoredPoissonWorkTranscript | None] = []
    common = {
        "maximum_count_ceiling": maximum_count_ceiling,
        "maximum_work": maximum_poisson_work,
        "maximum_rational_bits": maximum_rational_bits,
        "exp_precision_bits": exp_precision_bits,
        "maximum_exp_terms": maximum_exp_terms,
        "maximum_exp_work": maximum_exp_work,
        "maximum_exp_range_reductions": maximum_exp_range_reductions,
    }
    for mean, observed, ceiling, fitted in zip(
        pre_gain_means,
        observed_counts,
        count_ceilings,
        fit_mask,
        strict=True,
    ):
        probability, probability_work = enclose_censored_poisson_probability(
            mean, observed, ceiling, **common
        )
        censored_mean, mean_work = enclose_censored_poisson_mean(
            mean, ceiling, **common
        )
        probabilities.append(probability)
        censored_means.append(censored_mean)
        probability_transcripts.append(probability_work)
        mean_transcripts.append(mean_work)
        if not fitted:
            nlls.append(None)
            nll_transcripts.append(None)
            continue
        if probability[0] <= _ZERO:
            nlls.append(None)
            nll_transcripts.append(None)
            continue
        nll, nll_work = enclose_censored_poisson_nll(
            mean,
            observed,
            ceiling,
            **common,
            log_precision_bits=log_precision_bits,
            maximum_log_terms=maximum_log_terms,
            maximum_log_work=maximum_log_work,
            maximum_log_range_reductions=maximum_log_range_reductions,
        )
        nlls.append(nll)
        nll_transcripts.append(nll_work)
    result: _PoissonReports = (
        tuple(probabilities),
        tuple(censored_means),
        tuple(nlls),
        tuple(probability_transcripts),
        tuple(mean_transcripts),
        tuple(nll_transcripts),
    )
    return result  # noqa: RET504


def _interval_carrier(
    value: _Interval,
) -> GalerkinLocalDetectorRationalInterval:
    """PRIVATE: Freeze one checked exact interval in its carrier form.

    Parameters
    ----------
    value : _Interval
        Ordered exact rational interval.

    Returns
    -------
    interval : GalerkinLocalDetectorRationalInterval
        Exact numerator/denominator storage carrier.
    """
    interval = _make_local_detector_rational_interval(value[0], value[1])
    return interval  # noqa: RET504


def _interval_carriers(
    values: _Intervals,
) -> tuple[GalerkinLocalDetectorRationalInterval, ...]:
    """PRIVATE: Freeze one ordered tuple of exact intervals.

    Parameters
    ----------
    values : _Intervals
        Required canonical input.

    Returns
    -------
    result : tuple[GalerkinLocalDetectorRationalInterval, ...]
        Canonical derived result.
    """
    return tuple(_interval_carrier(value) for value in values)


def _trace_hull_intervals(
    trace: GalerkinLocalDetectorRealProductionTrace,
) -> _Intervals:
    """PRIVATE: Recover exact dyadic intervals from one trace union hull.

    Parameters
    ----------
    trace : GalerkinLocalDetectorRealProductionTrace
        Required canonical input.

    Returns
    -------
    result : _Intervals
        Canonical derived result.
    """
    lower = np.asarray(trace.certified_hull_lower_bounds)
    upper = np.asarray(trace.certified_hull_upper_bounds)
    return tuple(
        (Fraction.from_float(float(lo)), Fraction.from_float(float(hi)))
        for lo, hi in zip(lower, upper, strict=True)
    )


def _exact_vector_norm_upper(
    values: np.ndarray, ledger: _DetectorLedger
) -> Fraction:
    """PRIVATE: Return a deterministic exact dyadic l2-norm upper bound.

    Parameters
    ----------
    values : np.ndarray
        Required canonical input.
    ledger : _DetectorLedger
        Required canonical input.

    Returns
    -------
    result : Fraction
        Canonical derived result.
    """
    squared = _ZERO
    for value in np.asarray(values, dtype=np.float64):
        exact = ledger.retain(Fraction.from_float(float(value)))
        squared = ledger.add(squared, ledger.multiply(exact, exact))
    rounded = fraction_upper_float(ledger.root_upper(squared))
    if not np.isfinite(rounded):
        ledger.fail(
            GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE,
            "local detector vector norm leaves finite binary64 range",
        )
    return Fraction.from_float(float(rounded))


def _make_charged_production_trace(  # noqa: PLR0913
    raw: _Intervals,
    point: np.ndarray,
    *,
    stage: GalerkinLocalDetectorProductionStage,
    quantity: str,
    logical_shape: tuple[int, ...],
    ledger: _DetectorLedger,
) -> GalerkinLocalDetectorRealProductionTrace:
    """PRIVATE: Charge the exact point-to-raw audit before tracing it.

    Parameters
    ----------
    raw : _Intervals
        Required canonical input.
    point : np.ndarray
        Required canonical input.
    stage : GalerkinLocalDetectorProductionStage
        Required canonical input.
    quantity : str
        Required canonical input.
    logical_shape : tuple[int, ...]
        Required canonical input.
    ledger : _DetectorLedger
        Required canonical input.

    Returns
    -------
    trace : GalerkinLocalDetectorRealProductionTrace
        Canonical derived result.
    """
    flat = np.asarray(point, dtype=np.float64).reshape(-1)
    magnitude = np.abs(flat)
    if bool(
        np.any(
            ~np.isfinite(flat)
            | ((magnitude != 0.0) & (magnitude < np.finfo(np.float64).tiny))
        )
    ):
        ledger.fail(
            GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE,
            "local detector production point is outside normal binary64",
        )
    for value, interval in zip(flat, raw, strict=True):
        exact = ledger.retain(Fraction.from_float(float(value)))
        ledger.retain(abs(ledger.subtract(exact, interval[0])))
        ledger.retain(abs(ledger.subtract(exact, interval[1])))
    try:
        trace = _make_local_detector_real_production_trace(
            _interval_carriers(raw),
            np.asarray(point, dtype=np.float64),
            stage=stage,
            quantity=quantity,
            logical_shape=logical_shape,
        )
    except (TypeError, ValueError) as error:
        ledger.fail(
            GalerkinLocalDetectorFailure.PRODUCTION_POINT_HULL_FAILURE,
            f"local detector production trace failed: {error}",
        )
    for interval in trace.exact_point_intervals:
        ledger.retain(interval.lower)
        ledger.retain(interval.upper)
    for values in (
        trace.point_to_raw_absolute_error_upper_bounds,
        trace.certified_hull_lower_bounds,
        trace.certified_hull_upper_bounds,
    ):
        for value in np.asarray(values):
            ledger.retain(Fraction.from_float(float(value)))
    ledger.production_trace_count += 1
    ledger.hull_endpoint_count += 2 * flat.size
    ledger.production_traces.append(trace)
    return trace


def _checked_fraction_production_point(
    value: Fraction, ledger: _DetectorLedger
) -> float:
    """PRIVATE: Convert one exact value to a normal-or-zero binary64 point.

    Parameters
    ----------
    value : Fraction
        Required canonical input.
    ledger : _DetectorLedger
        Required canonical input.

    Returns
    -------
    point : float
        Canonical derived result.
    """
    value = ledger.retain(value)
    try:
        point = float(value)
    except OverflowError:
        ledger.fail(
            GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE,
            "local detector exact production value exceeds binary64 range",
        )
    magnitude = abs(point)
    if not np.isfinite(point) or (
        magnitude != 0.0 and magnitude < np.finfo(np.float64).tiny
    ):
        ledger.fail(
            GalerkinLocalDetectorFailure.ARITHMETIC_RANGE_FAILURE,
            "local detector exact production value is not normal binary64",
        )
    return point


def _stored_interval_peak_bits(values: object) -> int:
    """PRIVATE: Inspect exact storage without constructing Fractions.

    Parameters
    ----------
    values : object
        Required canonical input.

    Returns
    -------
    result : int
        Canonical derived result.

    Raises
    ------
    TypeError
        If the canonical contract is violated.
    """
    if values is None:
        return 0
    if isinstance(values, tuple):
        return max(
            (_stored_interval_peak_bits(value) for value in values),
            default=0,
        )
    names = (
        "lower_numerator",
        "lower_denominator",
        "upper_numerator",
        "upper_denominator",
    )
    if any(not hasattr(values, name) for name in names):
        raise TypeError("local detector interval storage is not inspectable")
    stored = tuple(getattr(values, name) for name in names)
    if any(type(value) is not int for value in stored):
        raise TypeError("local detector interval storage must use Python ints")
    return max(abs(value).bit_length() for value in stored)


def _raise_raw_rational_size_failure(peak_bits: int, message: str) -> None:
    """PRIVATE: Stop before constructing one over-policy parent Fraction.

    Parameters
    ----------
    peak_bits : int
        Required canonical input.
    message : str
        Required canonical input.

    Raises
    ------
    _DetectorArithmeticError
        If the canonical contract is violated.
    """
    raise _DetectorArithmeticError(
        GalerkinLocalDetectorFailure.RATIONAL_SIZE_LIMIT,
        0,
        message,
        work_snapshot={
            "coordinate_factor_count": 0,
            "pixel_product_count": 0,
            "mode_quadratic_count": 0,
            "ensemble_product_count": 0,
            "response_product_count": 0,
            "production_trace_count": 0,
            "hull_endpoint_count": 0,
            "exact_work_count": 0,
            "rational_peak_bits": peak_bits,
        },
    )


def _charged_midpoint_production_point(
    interval: _Interval, ledger: _DetectorLedger
) -> float:
    """PRIVATE: Charge and round the exact midpoint of one interval.

    Parameters
    ----------
    interval : _Interval
        Required canonical input.
    ledger : _DetectorLedger
        Required canonical input.

    Returns
    -------
    result : float
        Canonical derived result.
    """
    midpoint = ledger.divide(ledger.add(interval[0], interval[1]), Fraction(2))
    return _checked_fraction_production_point(midpoint, ledger)


def _planned_pixel_exact_work(
    convention: GalerkinLocalDetectorCoordinateConvention,
    *,
    fiber_count: int,
    pixel_count: int,
    retained_count: int,
    mapped_count: int,
) -> int:
    """PRIVATE: Return the deterministic successful pixel-work preflight.

    Parameters
    ----------
    convention : GalerkinLocalDetectorCoordinateConvention
        Required canonical input.
    fiber_count : int
        Required canonical input.
    pixel_count : int
        Required canonical input.
    retained_count : int
        Required canonical input.
    mapped_count : int
        Required canonical input.

    Returns
    -------
    result : int
        Canonical derived result.
    """
    if convention is (
        GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES
    ):
        coordinate = 0
    elif convention is (
        GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_CYCLIC_COORDINATES
    ):
        coordinate = 4
    elif convention is (
        GalerkinLocalDetectorCoordinateConvention.NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES
    ):
        coordinate = 2
    else:
        coordinate = 2 * retained_count
    trace_audit = (
        2 * fiber_count
        + 2
        + 2 * fiber_count
        + 2 * fiber_count
        + 2 * fiber_count
        + 2 * fiber_count
        + 2 * pixel_count * fiber_count
        + 2 * fiber_count
        + 2 * fiber_count
        + 2 * (pixel_count + 1)
    )
    pixel_forms = 2 + 6 * fiber_count + 6 * mapped_count
    amplitude_square = 3 * retained_count
    component_norms = 4 * fiber_count + 2
    form_count = pixel_count + 1
    lvt56 = 4 * form_count * fiber_count + 3 + 7 * form_count
    traced_exact_state = 2 * form_count
    return (
        coordinate
        + trace_audit
        + pixel_forms
        + amplitude_square
        + component_norms
        + lvt56
        + traced_exact_state
    )


def _pixel_scope_evidence() -> dict[str, str]:
    """PRIVATE: Return fixed passive-pixel semantic scope strings.

    Returns
    -------
    result : dict[str, str]
        Canonical derived result.
    """
    return {
        "coordinate_factor_scope": (
            "current, amplitude-coordinate scale, coordinate Jacobian, "
            "quadrature, and aperture remain separately replayable"
        ),
        "pixel_form_scope": (
            "amplitude_scale squared is absorbed wholly into the same Q "
            "used by production, norm, LVT.56, and passivity"
        ),
        "lvt56_error_scope": (
            "f(E_prod), f(E_total)-f(E_prod), and f(E_total) are disjoint "
            "exact-once reports; E_state is informational"
        ),
        "passivity_margin_scope": (
            "Q_out minus the disjoint pixel sum uses the shared base and "
            "aperture complement"
        ),
        "no_experimental_validity_scope": (
            "mathematical detector eligibility makes no experimental-"
            "validity claim"
        ),
    }


def _l8_parent_resource_totals(
    port: GalerkinLocalPositivePortCertificate,
) -> tuple[int, int, int]:
    """PRIVATE: Sum authenticated L8 work, traces, and hull endpoints.

    Parameters
    ----------
    port : GalerkinLocalPositivePortCertificate
        Required canonical input.

    Returns
    -------
    work : int
        Canonical derived result.
    traces : int
        Canonical derived result.
    hulls : int
        Canonical derived result.
    """
    terminal = port.terminal_certificate
    branch = terminal.branch_evidence
    cut = terminal.cut_balance
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
    direct_work = (
        int(branch.direct_work_count_exact)
        + int(branch.direct_rational_work_count_exact)
        + int(cut.direct_work_count_exact)
        + int(cut.direct_rational_work_count_exact)
    )
    work = (
        root_work
        + propagator_work
        + branch.entire_evidence.total_exact_work_count
        + direct_work
    )
    traces = len(port.production_traces)
    trace_hulls = sum(
        2 * np.asarray(trace.point).size for trace in port.production_traces
    )
    hulls = trace_hulls + branch.hull_completed_endpoint_count
    return work, traces, hulls


def _pixel_preflight_failure_evidence(
    certificate: GalerkinLocalPassivePixelForms,
    *,
    failure: GalerkinLocalDetectorFailure,
    planned: int,
) -> dict[str, object]:
    """PRIVATE: Build a unique zero-completed pixel preflight sentinel.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.
    failure : GalerkinLocalDetectorFailure
        Required canonical input.
    planned : int
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    zero = _interval_carrier((_ZERO, _ZERO))
    inherited = GalerkinLocalDetectorFailure(
        int(np.asarray(certificate.positive_port.failure_mask))
    )
    parent_work, parent_traces, parent_hulls = _l8_parent_resource_totals(
        certificate.positive_port
    )
    work = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=certificate.work_transcript.maximum_work,
        maximum_rational_bits=(
            certificate.work_transcript.maximum_rational_bits
        ),
        coordinate_factor_count=0,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        production_trace_count=0,
        hull_endpoint_count=0,
        nested_production_trace_count=parent_traces,
        nested_hull_endpoint_count=parent_hulls,
        exact_work_count=0,
        rational_peak_bits=0,
        nested_parent_work_count_exact=str(parent_work),
        planned_exact_work_count_exact=str(planned),
        attempted_exact_work_count_exact=str(planned),
        completed_successfully=False,
        arithmetic_failure=failure,
        preflight_failed=True,
        count_overflow=planned > _MAXIMUM_SIGNED_INT64,
    )
    result: dict[str, object] = {
        "production_evidence_available": False,
        "current_weight_intervals": (),
        "amplitude_scale_interval": zero,
        "coordinate_jacobian_intervals": (),
        "quadrature_weight_intervals": (
            certificate.quadrature_weight_intervals
        ),
        "aperture_efficiency_intervals": (
            certificate.aperture_efficiency_intervals
        ),
        "outward_form_diagonal_intervals": (),
        "pixel_form_diagonal_intervals": (),
        "outward_minus_pixel_form_diagonal_intervals": (),
        "production_outward_quadratic_interval": zero,
        "outward_form_norm_upper_interval": zero,
        "outward_production_realization_error_upper_interval": zero,
        "outward_state_radius_incremental_error_upper_interval": zero,
        "outward_combined_exact_state_error_upper_interval": zero,
        "exact_state_outward_flux_interval": zero,
        "production_quadratic_intervals": (),
        "pixel_form_norm_upper_intervals": (),
        "production_to_exact_x_amplitude_error_interval": zero,
        "state_radius_amplitude_error_interval": zero,
        "exact_state_amplitude_error_interval": zero,
        "production_amplitude_norm_interval": zero,
        "production_realization_error_upper_intervals": (),
        "state_radius_incremental_error_upper_intervals": (),
        "combined_exact_state_error_upper_intervals": (),
        "exact_state_pixel_flux_intervals": (),
        "production_traces": (),
        "work_transcript": work,
        "positive_forms_eligible": False,
        "passive_forms_eligible": False,
        "failure_mask": int(inherited | failure),
    }
    result.update(_pixel_scope_evidence())
    return result


def _pixel_parent_failure_evidence(
    certificate: GalerkinLocalPassivePixelForms,
    *,
    additional_failure: GalerkinLocalDetectorFailure = (
        GalerkinLocalDetectorFailure.NONE
    ),
) -> dict[str, object]:
    """PRIVATE: Propagate an unavailable positive port without local work.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.
    additional_failure : GalerkinLocalDetectorFailure
        Optional input; the signature supplies its default.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    zero = _interval_carrier((_ZERO, _ZERO))
    inherited = int(np.asarray(certificate.positive_port.failure_mask))
    parent_work, parent_traces, parent_hulls = _l8_parent_resource_totals(
        certificate.positive_port
    )
    work = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=certificate.work_transcript.maximum_work,
        maximum_rational_bits=(
            certificate.work_transcript.maximum_rational_bits
        ),
        coordinate_factor_count=0,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        production_trace_count=0,
        hull_endpoint_count=0,
        nested_production_trace_count=parent_traces,
        nested_hull_endpoint_count=parent_hulls,
        exact_work_count=0,
        rational_peak_bits=0,
        nested_parent_work_count_exact=str(parent_work),
    )
    result: dict[str, object] = {
        "production_evidence_available": False,
        "current_weight_intervals": (),
        "amplitude_scale_interval": zero,
        "coordinate_jacobian_intervals": (),
        "quadrature_weight_intervals": (
            certificate.quadrature_weight_intervals
        ),
        "aperture_efficiency_intervals": (
            certificate.aperture_efficiency_intervals
        ),
        "outward_form_diagonal_intervals": (),
        "pixel_form_diagonal_intervals": (),
        "outward_minus_pixel_form_diagonal_intervals": (),
        "production_outward_quadratic_interval": zero,
        "outward_form_norm_upper_interval": zero,
        "outward_production_realization_error_upper_interval": zero,
        "outward_state_radius_incremental_error_upper_interval": zero,
        "outward_combined_exact_state_error_upper_interval": zero,
        "exact_state_outward_flux_interval": zero,
        "production_quadratic_intervals": (),
        "pixel_form_norm_upper_intervals": (),
        "production_to_exact_x_amplitude_error_interval": zero,
        "state_radius_amplitude_error_interval": zero,
        "exact_state_amplitude_error_interval": zero,
        "production_amplitude_norm_interval": zero,
        "production_realization_error_upper_intervals": (),
        "state_radius_incremental_error_upper_intervals": (),
        "combined_exact_state_error_upper_intervals": (),
        "exact_state_pixel_flux_intervals": (),
        "production_traces": (),
        "work_transcript": work,
        "positive_forms_eligible": False,
        "passive_forms_eligible": False,
        "failure_mask": int(
            GalerkinLocalDetectorFailure(inherited) | additional_failure
        ),
    }
    result.update(_pixel_scope_evidence())
    return result


def _pixel_float_points(
    certificate: GalerkinLocalPassivePixelForms,
    amplitude_scale_point: float,
    jacobian_points: np.ndarray,
    current_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PRIVATE: Rebuild the actual binary64 pixel-form production points.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.
    amplitude_scale_point : float
        Required canonical input.
    jacobian_points : np.ndarray
        Required canonical input.
    current_points : np.ndarray
        Required canonical input.

    Returns
    -------
    outward : np.ndarray
        Canonical derived result.
    pixels : np.ndarray
        Canonical derived result.
    margin : np.ndarray
        Canonical derived result.
    """
    quadrature = np.asarray(certificate.quadrature_weights, dtype=np.float64)
    aperture = np.asarray(certificate.aperture_efficiencies, dtype=np.float64)
    mapping = np.asarray(certificate.node_to_pixel, dtype=np.int64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale_squared = np.float64(amplitude_scale_point) * np.float64(
            amplitude_scale_point
        )
        outward = current_points.astype(np.float64) * jacobian_points.astype(
            np.float64
        )
        outward = outward * scale_squared
        outward = outward * quadrature
        pixels = np.zeros(
            (certificate.pixel_count, current_points.size), dtype=np.float64
        )
        for node, pixel in enumerate(mapping):
            if pixel >= 0:
                pixels[pixel, node] = outward[node] * aperture[node]
        margin = outward - np.sum(pixels, axis=0, dtype=np.float64)
    return outward, pixels, margin


def _expected_local_passive_pixel_evidence_core(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalPassivePixelForms,
) -> dict[str, object]:
    """PRIVATE: Recompute every derived passive-pixel field from its owner.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.

    Raises
    ------
    RuntimeError
        If the canonical contract is violated.
    """
    port = certificate.positive_port
    fiber_count = np.asarray(port.production_amplitudes).shape[0]
    transcript = certificate.work_transcript
    quadrature_input = np.asarray(certificate.quadrature_weights)
    aperture_input = np.asarray(certificate.aperture_efficiencies)
    input_failure = GalerkinLocalDetectorFailure.NONE
    if bool(np.any(quadrature_input < 0.0)):
        input_failure |= GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE
    if bool(np.any((aperture_input < 0.0) | (aperture_input > 1.0))):
        input_failure |= GalerkinLocalDetectorFailure.PIXEL_FORM_NONPASSIVE
    if not bool(np.asarray(port.positive_port_eligible)) or input_failure:
        return _pixel_parent_failure_evidence(
            certificate, additional_failure=input_failure
        )
    retained_array = np.asarray(port.retained_propagating_mask)
    mapping_array = np.asarray(certificate.node_to_pixel)
    planned = _planned_pixel_exact_work(
        certificate.coordinate_convention,
        fiber_count=fiber_count,
        pixel_count=certificate.pixel_count,
        retained_count=int(np.count_nonzero(retained_array)),
        mapped_count=int(np.count_nonzero(mapping_array >= 0)),
    )
    if planned > _MAXIMUM_SIGNED_INT64:
        return _pixel_preflight_failure_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.EXACT_WORK_COUNT_OVERFLOW,
            planned=planned,
        )
    if planned > transcript.maximum_work:
        return _pixel_preflight_failure_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED,
            planned=planned,
        )
    parent_peak_bits = _stored_interval_peak_bits(
        (
            port.exact_root_intervals,
            certificate.quadrature_weight_intervals,
            certificate.aperture_efficiency_intervals,
        )
    )
    if parent_peak_bits > transcript.maximum_rational_bits:
        _raise_raw_rational_size_failure(
            parent_peak_bits,
            "local detector parent root exceeds the pixel rational policy",
        )
    ledger = _checked_policy(
        transcript.maximum_work, transcript.maximum_rational_bits
    )
    ledger.rational_peak_bits = parent_peak_bits
    retained = tuple(
        bool(value) for value in np.asarray(port.retained_propagating_mask)
    )
    zero = (_ZERO, _ZERO)
    raw_roots: _OptionalIntervals = tuple(
        None if value is None else (value.lower, value.upper)
        for value in port.exact_root_intervals
    )
    raw_current: _Intervals = tuple(
        cast(_Interval, root) if keep else zero
        for root, keep in zip(raw_roots, retained, strict=True)
    )
    root_points = np.where(
        np.asarray(port.retained_propagating_mask),
        np.asarray(port.production_root_realizations, dtype=np.float64),
        0.0,
    ).astype(np.float64)
    current_trace = _make_charged_production_trace(
        raw_current,
        root_points,
        stage=GalerkinLocalDetectorProductionStage.COORDINATE_FACTOR,
        quantity="positive_current_weight",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    current = _trace_hull_intervals(current_trace)

    terminal = port.terminal_certificate
    projection = terminal.projection_certificate
    zero_slab = projection.zero_slab_certificate
    represented = zero_slab.represented_source_certificate
    target = represented.source.target
    target_ledger = target.fixed_linear_error_ledger
    k0 = (
        Fraction.from_float(
            float(np.asarray(target_ledger.exact_wavenumber_lower_bound))
        ),
        Fraction.from_float(
            float(np.asarray(target_ledger.exact_wavenumber_upper_bound))
        ),
    )
    amplitude_scale_raw, jacobians_raw = _coordinate_factors(
        certificate.coordinate_convention,
        k0,
        tuple(current),
        retained,
        ledger,
    )
    k0_point = np.float64(np.asarray(target.wavenumber))
    root_point = root_points
    two_pi_point = np.float64(2.0) * np.float64(np.pi)
    convention = certificate.coordinate_convention
    if convention is (
        GalerkinLocalDetectorCoordinateConvention.NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES
    ):
        amplitude_scale_point = two_pi_point
    else:
        amplitude_scale_point = np.float64(1.0)
    if convention in (
        GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_ANGULAR_COORDINATES,
        GalerkinLocalDetectorCoordinateConvention.NATIVE_CYCLIC_AMPLITUDE_IN_CYCLIC_COORDINATES,
    ):
        jacobian_points = np.where(np.asarray(retained), 1.0, 0.0).astype(
            np.float64
        )
    elif convention is (
        GalerkinLocalDetectorCoordinateConvention.ANGULAR_WAVENUMBER_AMPLITUDE_IN_CYCLIC_COORDINATES
    ):
        jacobian_points = np.where(
            np.asarray(retained),
            two_pi_point * two_pi_point,
            0.0,
        ).astype(np.float64)
    else:
        jacobian_points = np.where(
            np.asarray(retained), k0_point * root_point, 0.0
        ).astype(np.float64)
    scale_trace = _make_charged_production_trace(
        (amplitude_scale_raw,),
        np.asarray([amplitude_scale_point], dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.COORDINATE_FACTOR,
        quantity="amplitude_scale",
        logical_shape=(),
        ledger=ledger,
    )
    jacobian_trace = _make_charged_production_trace(
        jacobians_raw,
        jacobian_points,
        stage=GalerkinLocalDetectorProductionStage.COORDINATE_FACTOR,
        quantity="coordinate_jacobian",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    amplitude_scale = _trace_hull_intervals(scale_trace)[0]
    jacobians = _trace_hull_intervals(jacobian_trace)
    quadrature_raw = tuple(
        (value.lower, value.upper)
        for value in certificate.quadrature_weight_intervals
    )
    aperture_raw = tuple(
        (value.lower, value.upper)
        for value in certificate.aperture_efficiency_intervals
    )
    if any(value[0] < _ZERO for value in quadrature_raw):
        ledger.fail(
            GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE,
            "local detector exact quadrature interval is negative",
        )
    if any(value[0] < _ZERO or value[1] > _ONE for value in aperture_raw):
        ledger.fail(
            GalerkinLocalDetectorFailure.PIXEL_FORM_NONPASSIVE,
            "local detector exact aperture interval is outside [0,1]",
        )
    quadrature_trace = _make_charged_production_trace(
        quadrature_raw,
        np.asarray(certificate.quadrature_weights, dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
        quantity="quadrature_weight",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    aperture_trace = _make_charged_production_trace(
        aperture_raw,
        np.asarray(certificate.aperture_efficiencies, dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
        quantity="aperture_efficiency",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    quadrature = _trace_hull_intervals(quadrature_trace)
    aperture = _trace_hull_intervals(aperture_trace)
    mapping = tuple(
        int(value) for value in np.asarray(certificate.node_to_pixel)
    )
    outward_raw, pixels_raw, margin_raw = _pixel_form_diagonals(
        current,
        jacobians,
        amplitude_scale,
        quadrature,
        aperture,
        mapping,
        certificate.pixel_count,
        ledger,
    )
    outward_points, pixel_points, margin_points = _pixel_float_points(
        certificate,
        float(amplitude_scale_point),
        jacobian_points,
        root_points,
    )
    outward_trace = _make_charged_production_trace(
        outward_raw,
        outward_points,
        stage=GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
        quantity="outward_form_diagonal",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    pixel_trace = _make_charged_production_trace(
        tuple(value for row in pixels_raw for value in row),
        pixel_points,
        stage=GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
        quantity="pixel_form_diagonal",
        logical_shape=(certificate.pixel_count, fiber_count),
        ledger=ledger,
    )
    margin_trace = _make_charged_production_trace(
        margin_raw,
        margin_points,
        stage=GalerkinLocalDetectorProductionStage.PIXEL_FORM_DIAGONAL,
        quantity="outward_minus_pixels_diagonal",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    outward = _trace_hull_intervals(outward_trace)
    flat_pixels = _trace_hull_intervals(pixel_trace)
    pixels: _PixelForms = tuple(
        tuple(
            flat_pixels[row * fiber_count + column]
            for column in range(fiber_count)
        )
        for row in range(certificate.pixel_count)
    )
    margin = _trace_hull_intervals(margin_trace)

    amplitudes = np.asarray(port.production_amplitudes, dtype=np.complex128)
    active_amplitudes = np.where(
        np.asarray(retained), amplitudes, np.complex128(0.0 + 0.0j)
    ).astype(np.complex128)
    amplitude_squared_values: list[Fraction] = []
    for value, keep in zip(active_amplitudes, retained, strict=True):
        if not keep:
            amplitude_squared_values.append(_ZERO)
            continue
        real = ledger.retain(Fraction.from_float(float(value.real)))
        imag = ledger.retain(Fraction.from_float(float(value.imag)))
        squared = ledger.add(
            ledger.multiply(real, real), ledger.multiply(imag, imag)
        )
        amplitude_squared_values.append(squared)
    amplitude_squared_raw = tuple(
        (value, value) for value in amplitude_squared_values
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        amplitude_squared_points = (
            np.real(active_amplitudes) * np.real(active_amplitudes)
            + np.imag(active_amplitudes) * np.imag(active_amplitudes)
        ).astype(np.float64)
    amplitude_trace = _make_charged_production_trace(
        amplitude_squared_raw,
        amplitude_squared_points,
        stage=GalerkinLocalDetectorProductionStage.MODE_PRODUCTION_QUADRATIC,
        quantity="retained_positive_port_amplitude_squared",
        logical_shape=(fiber_count,),
        ledger=ledger,
    )
    amplitude_squared = _trace_hull_intervals(amplitude_trace)
    branch = terminal.branch_evidence
    production_error = _exact_vector_norm_upper(
        np.asarray(branch.production_to_submitted_amplitude_error_bounds)[
            :, 0
        ],
        ledger,
    )
    state_error = _exact_vector_norm_upper(
        np.asarray(branch.state_radius_amplitude_error_bounds)[:, 0],
        ledger,
    )
    total_error = Fraction.from_float(
        float(np.asarray(port.exact_state_prediction_error_l2_upper_bound))
    )
    amplitude_norm = Fraction.from_float(
        float(np.asarray(port.production_prediction_l2_norm_upper_bound))
    )
    forms: _PixelForms = (outward, *pixels)
    reports = _lvt56_quadratic_reports(
        amplitude_squared,
        forms,
        (production_error, production_error),
        (state_error, state_error),
        (total_error, total_error),
        (amplitude_norm, amplitude_norm),
        ledger,
    )
    production_raw = reports[0]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        form_points = np.concatenate((outward_points[None, :], pixel_points))
        production_points = np.sum(
            form_points * amplitude_squared_points[None, :],
            axis=1,
            dtype=np.float64,
        )
    quadratic_trace = _make_charged_production_trace(
        production_raw,
        production_points,
        stage=GalerkinLocalDetectorProductionStage.MODE_PRODUCTION_QUADRATIC,
        quantity="production_form_quadratic",
        logical_shape=(certificate.pixel_count + 1,),
        ledger=ledger,
    )
    production = _trace_hull_intervals(quadratic_trace)
    combined_errors = reports[4]
    exact_state: list[_Interval] = []
    for point_hull, error in zip(production, combined_errors, strict=True):
        lower = max(_ZERO, ledger.subtract(point_hull[0], error[1]))
        upper = ledger.add(point_hull[1], error[1])
        exact_state.append((lower, upper))
    traces = (
        current_trace,
        scale_trace,
        jacobian_trace,
        quadrature_trace,
        aperture_trace,
        outward_trace,
        pixel_trace,
        margin_trace,
        amplitude_trace,
        quadratic_trace,
    )
    hull_endpoint_count = sum(
        2 * np.asarray(trace.point).size for trace in traces
    )
    if (
        ledger.production_trace_count != len(traces)
        or ledger.hull_endpoint_count != hull_endpoint_count
    ):
        raise RuntimeError("local detector pixel trace ledger disagrees")
    if ledger.exact_work_count != planned:
        raise RuntimeError(
            "local detector pixel preflight and completed work disagree"
        )
    parent_work, parent_traces, parent_hulls = _l8_parent_resource_totals(port)
    work = _make_local_detector_work_transcript(
        algorithm=ledger.algorithm,
        maximum_work=ledger.maximum_work,
        maximum_rational_bits=ledger.maximum_rational_bits,
        coordinate_factor_count=ledger.coordinate_factor_count,
        pixel_product_count=ledger.pixel_product_count,
        mode_quadratic_count=ledger.mode_quadratic_count,
        ensemble_product_count=ledger.ensemble_product_count,
        response_product_count=ledger.response_product_count,
        production_trace_count=ledger.production_trace_count,
        hull_endpoint_count=ledger.hull_endpoint_count,
        nested_production_trace_count=parent_traces,
        nested_hull_endpoint_count=parent_hulls,
        exact_work_count=ledger.exact_work_count,
        rational_peak_bits=ledger.rational_peak_bits,
        nested_parent_work_count_exact=str(parent_work),
    )
    inherited_failure = int(np.asarray(port.failure_mask))
    eligible = bool(np.asarray(port.positive_port_eligible))
    return {
        "production_evidence_available": True,
        "current_weight_intervals": _interval_carriers(current),
        "amplitude_scale_interval": _interval_carrier(amplitude_scale),
        "coordinate_jacobian_intervals": _interval_carriers(jacobians),
        "quadrature_weight_intervals": _interval_carriers(quadrature_raw),
        "aperture_efficiency_intervals": _interval_carriers(aperture_raw),
        "outward_form_diagonal_intervals": _interval_carriers(outward),
        "pixel_form_diagonal_intervals": tuple(
            _interval_carriers(row) for row in pixels
        ),
        "outward_minus_pixel_form_diagonal_intervals": _interval_carriers(
            margin
        ),
        "production_outward_quadratic_interval": _interval_carrier(
            production[0]
        ),
        "outward_form_norm_upper_interval": _interval_carrier(reports[1][0]),
        "outward_production_realization_error_upper_interval": (
            _interval_carrier(reports[2][0])
        ),
        "outward_state_radius_incremental_error_upper_interval": (
            _interval_carrier(reports[3][0])
        ),
        "outward_combined_exact_state_error_upper_interval": (
            _interval_carrier(reports[4][0])
        ),
        "exact_state_outward_flux_interval": _interval_carrier(exact_state[0]),
        "production_quadratic_intervals": _interval_carriers(production[1:]),
        "pixel_form_norm_upper_intervals": _interval_carriers(reports[1][1:]),
        "production_to_exact_x_amplitude_error_interval": _interval_carrier(
            (production_error, production_error)
        ),
        "state_radius_amplitude_error_interval": _interval_carrier(
            (state_error, state_error)
        ),
        "exact_state_amplitude_error_interval": _interval_carrier(
            (total_error, total_error)
        ),
        "production_amplitude_norm_interval": _interval_carrier(
            (amplitude_norm, amplitude_norm)
        ),
        "production_realization_error_upper_intervals": _interval_carriers(
            reports[2][1:]
        ),
        "state_radius_incremental_error_upper_intervals": _interval_carriers(
            reports[3][1:]
        ),
        "combined_exact_state_error_upper_intervals": _interval_carriers(
            reports[4][1:]
        ),
        "exact_state_pixel_flux_intervals": _interval_carriers(
            tuple(exact_state[1:])
        ),
        "production_traces": traces,
        "work_transcript": work,
        "positive_forms_eligible": eligible,
        "passive_forms_eligible": eligible,
        "failure_mask": inherited_failure,
        **_pixel_scope_evidence(),
    }


def _pixel_partial_failure_evidence(
    certificate: GalerkinLocalPassivePixelForms,
    error: _DetectorArithmeticError,
    *,
    planned: int,
) -> dict[str, object]:
    """PRIVATE: Freeze one honest partially attempted pixel failure.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.
    error : _DetectorArithmeticError
        Required canonical input.
    planned : int
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    zero = _interval_carrier((_ZERO, _ZERO))
    snapshot = error.work_snapshot
    scientific = error.failure in (
        GalerkinLocalDetectorFailure.PIXEL_FORM_NONPOSITIVE,
        GalerkinLocalDetectorFailure.PIXEL_FORM_NONPASSIVE,
    )
    parent_work, parent_traces, parent_hulls = _l8_parent_resource_totals(
        certificate.positive_port
    )
    work = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=certificate.work_transcript.maximum_work,
        maximum_rational_bits=(
            certificate.work_transcript.maximum_rational_bits
        ),
        coordinate_factor_count=snapshot["coordinate_factor_count"],
        pixel_product_count=snapshot["pixel_product_count"],
        mode_quadratic_count=snapshot["mode_quadratic_count"],
        ensemble_product_count=snapshot["ensemble_product_count"],
        response_product_count=snapshot["response_product_count"],
        production_trace_count=snapshot["production_trace_count"],
        hull_endpoint_count=snapshot["hull_endpoint_count"],
        nested_production_trace_count=parent_traces,
        nested_hull_endpoint_count=parent_hulls,
        exact_work_count=snapshot["exact_work_count"],
        rational_peak_bits=snapshot["rational_peak_bits"],
        nested_parent_work_count_exact=str(parent_work),
        planned_exact_work_count_exact=str(planned),
        attempted_exact_work_count_exact=str(error.exact_work_count),
        completed_successfully=False,
        arithmetic_failure=(
            GalerkinLocalDetectorFailure.NONE if scientific else error.failure
        ),
        scientific_failure=(
            error.failure if scientific else GalerkinLocalDetectorFailure.NONE
        ),
    )
    inherited = GalerkinLocalDetectorFailure(
        int(np.asarray(certificate.positive_port.failure_mask))
    )
    result: dict[str, object] = {
        "production_evidence_available": False,
        "current_weight_intervals": (),
        "amplitude_scale_interval": zero,
        "coordinate_jacobian_intervals": (),
        "quadrature_weight_intervals": (
            certificate.quadrature_weight_intervals
        ),
        "aperture_efficiency_intervals": (
            certificate.aperture_efficiency_intervals
        ),
        "outward_form_diagonal_intervals": (),
        "pixel_form_diagonal_intervals": (),
        "outward_minus_pixel_form_diagonal_intervals": (),
        "production_outward_quadratic_interval": zero,
        "outward_form_norm_upper_interval": zero,
        "outward_production_realization_error_upper_interval": zero,
        "outward_state_radius_incremental_error_upper_interval": zero,
        "outward_combined_exact_state_error_upper_interval": zero,
        "exact_state_outward_flux_interval": zero,
        "production_quadratic_intervals": (),
        "pixel_form_norm_upper_intervals": (),
        "production_to_exact_x_amplitude_error_interval": zero,
        "state_radius_amplitude_error_interval": zero,
        "exact_state_amplitude_error_interval": zero,
        "production_amplitude_norm_interval": zero,
        "production_realization_error_upper_intervals": (),
        "state_radius_incremental_error_upper_intervals": (),
        "combined_exact_state_error_upper_intervals": (),
        "exact_state_pixel_flux_intervals": (),
        "production_traces": error.production_traces,
        "work_transcript": work,
        "positive_forms_eligible": False,
        "passive_forms_eligible": False,
        "failure_mask": int(inherited | error.failure),
    }
    result.update(_pixel_scope_evidence())
    return result


def _expected_local_passive_pixel_evidence(
    certificate: GalerkinLocalPassivePixelForms,
) -> dict[str, object]:
    """PRIVATE: Replay success and every typed passive-pixel stop.

    Parameters
    ----------
    certificate : GalerkinLocalPassivePixelForms
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    fiber_count = np.asarray(
        certificate.positive_port.production_amplitudes
    ).shape[0]
    retained = np.asarray(certificate.positive_port.retained_propagating_mask)
    mapping = np.asarray(certificate.node_to_pixel)
    planned = _planned_pixel_exact_work(
        certificate.coordinate_convention,
        fiber_count=fiber_count,
        pixel_count=certificate.pixel_count,
        retained_count=int(np.count_nonzero(retained)),
        mapped_count=int(np.count_nonzero(mapping >= 0)),
    )
    try:
        return _expected_local_passive_pixel_evidence_core(certificate)
    except _DetectorArithmeticError as error:
        return _pixel_partial_failure_evidence(
            certificate, error, planned=planned
        )


def _nested_entire_work(transcript: CensoredPoissonWorkTranscript) -> int:
    """PRIVATE: Sum local and successful nested exact helper work.

    Parameters
    ----------
    transcript : CensoredPoissonWorkTranscript
        Required canonical input.

    Returns
    -------
    result : int
        Canonical derived result.
    """
    return (
        transcript.exact_work_count
        + sum(value.exact_work_count for value in transcript.exp_transcripts)
        + sum(value.exact_work_count for value in transcript.log_transcripts)
    )


def _failed_helper_work(error: CensoredPoissonEnclosureError) -> int:
    """PRIVATE: Sum all successful-prefix and failing helper work.

    Parameters
    ----------
    error : CensoredPoissonEnclosureError
        Required canonical input.

    Returns
    -------
    result : int
        Canonical derived result.
    """
    return (
        error.exact_work_count
        + sum(value.exact_work_count for value in error.prior_exp_transcripts)
        + sum(value.exact_work_count for value in error.prior_log_transcripts)
        + (
            0
            if error.nested_exact_work_count is None
            else error.nested_exact_work_count
        )
    )


def _helper_failure_from_error(
    error: CensoredPoissonEnclosureError,
    *,
    call: GalerkinLocalDetectorHelperCall,
    channel: int,
) -> object:
    """PRIVATE: Freeze one replayed helper exception as public evidence.

    Parameters
    ----------
    error : CensoredPoissonEnclosureError
        Required canonical input.
    call : GalerkinLocalDetectorHelperCall
        Required canonical input.
    channel : int
        Required canonical input.

    Returns
    -------
    result : object
        Canonical derived result.
    """
    return _make_local_detector_helper_failure_evidence(
        call=call,
        channel_index=channel,
        failure=error.failure,
        local_exact_work_count=error.exact_work_count,
        nested_kernel=error.nested_kernel,
        nested_failure=error.nested_failure,
        nested_exact_work_count=error.nested_exact_work_count,
        nested_attempted_exact_work_count=(
            error.nested_attempted_exact_work_count
        ),
        prior_exp_transcripts=error.prior_exp_transcripts,
        prior_log_transcripts=error.prior_log_transcripts,
        planned_exact_work_count=error.attempted_exact_work_count,
        attempted_exact_work_count=error.attempted_exact_work_count,
    )


def _mix_fraction_hulls(
    mode_fractions: _ModePixelIntervals,
    weight_intervals: _Intervals,
    dose: _Interval,
    ledger: _DetectorLedger,
) -> _Intervals:
    """PRIVATE: Mix traced production fractions and apply traced dose.

    Parameters
    ----------
    mode_fractions : _ModePixelIntervals
        Required canonical input.
    weight_intervals : _Intervals
        Required canonical input.
    dose : _Interval
        Required canonical input.
    ledger : _DetectorLedger
        Required canonical input.

    Returns
    -------
    result : _Intervals
        Canonical derived result.
    """
    pixel_count = len(mode_fractions[0])
    output: list[_Interval] = []
    for pixel in range(pixel_count):
        mixed: _Interval = (_ZERO, _ZERO)
        for mode, weight in zip(mode_fractions, weight_intervals, strict=True):
            mixed = ledger.add_intervals(
                mixed, ledger.multiply_nonnegative(mode[pixel], weight)
            )
            ledger.ensemble_product_count += 1
        output.append(ledger.multiply_nonnegative(mixed, dose))
    return tuple(output)


def _detector_scope_evidence() -> dict[str, str]:
    """PRIVATE: Return fixed detector law and calibration scope strings.

    Returns
    -------
    result : dict[str, str]
        Canonical derived result.
    """
    return {
        "flux_normalization_scope": (
            "each coherent mode is divided only by its nested represented-"
            "source exact reduced incident flux"
        ),
        "ensemble_scope": (
            "Poissonized per-electron mode/configuration mixing occurs only "
            "after within-mode quadratics; a once-per-frame draw requires a "
            "mixture likelihood"
        ),
        "response_scope": (
            "fixed categorical Poisson routing plus independent Poisson "
            "background precedes censoring; column-substochasticity alone "
            "makes no general covariance or joint-law claim; the production "
            "censor consumes the canonical rounded pre-gain point singleton, "
            "not its audit hull, while exact-state censoring remains separate "
            "and the likelihood later forms their admitted union hull"
        ),
        "no_experimental_validity_scope": (
            "mathematical detector eligibility makes no experimental-"
            "validity claim"
        ),
    }


def _planned_detector_exact_work(
    *, mode_count: int, pixel_count: int, channel_count: int
) -> int:
    """PRIVATE: Return deterministic local detector work before allocation.

    Parameters
    ----------
    mode_count : int
        Required canonical input.
    pixel_count : int
        Required canonical input.
    channel_count : int
        Required canonical input.

    Returns
    -------
    result : int
        Canonical derived result.
    """
    return (
        5 * mode_count
        + 14 * mode_count * pixel_count
        + 6 * pixel_count
        + 10 * channel_count * pixel_count
        + 16 * channel_count
        + 2
    )


def _production_mode_points(
    pixels: tuple[GalerkinLocalPassivePixelForms, ...],
) -> np.ndarray:
    """PRIVATE: Extract canonical rounded per-mode pixel quadratics.

    Parameters
    ----------
    pixels : tuple[GalerkinLocalPassivePixelForms, ...]
        Required canonical input.

    Returns
    -------
    result : np.ndarray
        Canonical derived result.

    Raises
    ------
    ValueError
        If the canonical contract is violated.
    """
    rows: list[np.ndarray] = []
    for pixel in pixels:
        matches = tuple(
            trace
            for trace in pixel.production_traces
            if trace.stage
            is GalerkinLocalDetectorProductionStage.MODE_PRODUCTION_QUADRATIC
            and trace.quantity == "production_form_quadratic"
        )
        if len(matches) != 1:
            raise ValueError(
                "local detector pixel quadratic production trace is not unique"
            )
        rows.append(np.asarray(matches[0].point, dtype=np.float64)[1:])
    return np.stack(rows, axis=0)


def _detector_parent_failure_evidence(
    certificate: GalerkinLocalCensoredPoissonDetector,
    *,
    failure: GalerkinLocalDetectorFailure,
    planned: int | None = None,
) -> dict[str, object]:
    """PRIVATE: Build an unavailable detector stop with nested evidence.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.
    failure : GalerkinLocalDetectorFailure
        Required canonical input.
    planned : int | None
        Optional input; the signature supplies its default.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    nested_work = sum(
        pixel.work_transcript.exact_work_count
        + int(pixel.work_transcript.nested_parent_work_count_exact)
        + int(pixel.work_transcript.nested_helper_work_count_exact)
        for pixel in certificate.pixel_forms
    )
    nested_traces = sum(
        pixel.work_transcript.production_trace_count
        + pixel.work_transcript.nested_production_trace_count
        for pixel in certificate.pixel_forms
    )
    nested_hulls = sum(
        pixel.work_transcript.hull_endpoint_count
        + pixel.work_transcript.nested_hull_endpoint_count
        for pixel in certificate.pixel_forms
    )
    work = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=certificate.work_transcript.maximum_work,
        maximum_rational_bits=(
            certificate.work_transcript.maximum_rational_bits
        ),
        coordinate_factor_count=0,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        production_trace_count=0,
        hull_endpoint_count=0,
        nested_production_trace_count=nested_traces,
        nested_hull_endpoint_count=nested_hulls,
        exact_work_count=0,
        rational_peak_bits=0,
        nested_parent_work_count_exact=str(nested_work),
        planned_exact_work_count_exact=(
            None if planned is None else str(planned)
        ),
        attempted_exact_work_count_exact=(
            None if planned is None else str(planned)
        ),
        completed_successfully=planned is None,
        arithmetic_failure=(
            GalerkinLocalDetectorFailure.NONE if planned is None else failure
        ),
        preflight_failed=planned is not None,
        count_overflow=(
            False if planned is None else planned > _MAXIMUM_SIGNED_INT64
        ),
    )
    result: dict[str, object] = {
        "production_evidence_available": False,
        "exact_state_censored_mean_evidence_available": False,
        "production_censored_mean_evidence_available": False,
        "incident_reduced_flux_intervals": (),
        "mode_exact_state_pixel_flux_intervals": (),
        "mode_production_quadratic_intervals": (),
        "mode_pixel_form_norm_upper_intervals": (),
        "mode_production_to_exact_x_amplitude_error_intervals": (),
        "mode_state_radius_amplitude_error_intervals": (),
        "mode_exact_state_amplitude_error_intervals": (),
        "mode_production_amplitude_norm_intervals": (),
        "mode_production_realization_error_upper_intervals": (),
        "mode_state_radius_incremental_error_upper_intervals": (),
        "mode_combined_exact_state_error_upper_intervals": (),
        "mode_outward_passivity_margin_intervals": (),
        "mode_pixel_fraction_intervals": (),
        "ideal_arrival_mean_intervals": (),
        "production_pre_gain_mean_point_intervals": (),
        "exact_state_pre_gain_mean_intervals": (),
        "censored_mean_intervals": (),
        "expected_digitized_mean_intervals": (),
        "production_traces": (),
        "work_transcript": work,
        "censored_mean_transcripts": (),
        "censored_mean_failures": (),
        "production_censored_mean_transcripts": (),
        "production_censored_mean_failures": (),
        "detector_eligible": False,
        "likelihood_law_eligible": False,
        "failure_mask": int(failure),
    }
    result.update(_detector_scope_evidence())
    return result


def _detector_input_failure(
    certificate: GalerkinLocalCensoredPoissonDetector,
) -> GalerkinLocalDetectorFailure:
    """PRIVATE: Classify simultaneous physical detector-model failures.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.

    Returns
    -------
    failure : GalerkinLocalDetectorFailure
        Canonical derived result.
    """
    failure = GalerkinLocalDetectorFailure.NONE
    for pixel in certificate.pixel_forms:
        failure |= GalerkinLocalDetectorFailure(
            int(np.asarray(pixel.failure_mask))
        )
    if any(value < 0 for value in certificate.ensemble_weight_numerators):
        failure |= GalerkinLocalDetectorFailure.ENSEMBLE_WEIGHT_INVALID
    response = np.asarray(certificate.response_matrix)
    background = np.asarray(certificate.pre_gain_background)
    gain = np.asarray(certificate.deterministic_gain)
    dose_point = float(np.asarray(certificate.incident_electron_count_point))
    if bool(np.any(response < 0.0)) or bool(np.any(background < 0.0)):
        failure |= GalerkinLocalDetectorFailure.RESPONSE_NONPOSITIVE
    if bool(np.any(gain <= 0.0)):
        failure |= GalerkinLocalDetectorFailure.CALIBRATION_INVALID
    if dose_point < 0.0:
        failure |= GalerkinLocalDetectorFailure.DOSE_INVALID
    ceilings = np.asarray(certificate.count_ceilings)
    if bool(np.any(ceilings < 0)) or bool(
        np.any(ceilings > certificate.maximum_count_ceiling)
    ):
        failure |= GalerkinLocalDetectorFailure.COUNT_DOMAIN_INVALID
    return failure


def _expected_local_censored_poisson_detector_core(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalCensoredPoissonDetector,
) -> dict[str, object]:
    """PRIVATE: Recompute the complete fixed detector evidence DAG.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.

    Raises
    ------
    RuntimeError
        If the canonical contract is violated.
    """
    pixels = certificate.pixel_forms
    mode_count = len(pixels)
    pixel_count = pixels[0].pixel_count
    channel_count = np.asarray(certificate.response_matrix).shape[0]
    input_failure = _detector_input_failure(certificate)
    if input_failure:
        return _detector_parent_failure_evidence(
            certificate, failure=input_failure
        )
    planned = _planned_detector_exact_work(
        mode_count=mode_count,
        pixel_count=pixel_count,
        channel_count=channel_count,
    )
    if planned > _MAXIMUM_SIGNED_INT64:
        return _detector_parent_failure_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.EXACT_WORK_COUNT_OVERFLOW,
            planned=planned,
        )
    if planned > certificate.work_transcript.maximum_work:
        return _detector_parent_failure_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED,
            planned=planned,
        )
    raw_weight_bits = max(
        (
            max(abs(numerator).bit_length(), denominator.bit_length())
            for numerator, denominator in zip(
                certificate.ensemble_weight_numerators,
                certificate.ensemble_weight_denominators,
                strict=True,
            )
        ),
        default=0,
    )
    dose_storage = certificate.incident_electron_count
    raw_dose_bits = max(
        abs(dose_storage.lower_numerator).bit_length(),
        dose_storage.lower_denominator.bit_length(),
        abs(dose_storage.upper_numerator).bit_length(),
        dose_storage.upper_denominator.bit_length(),
    )
    raw_input_bits = max(raw_weight_bits, raw_dose_bits)
    parent_peak_bits = _stored_interval_peak_bits(
        tuple(
            (
                pixel.exact_state_pixel_flux_intervals,
                pixel.production_quadratic_intervals,
                pixel.exact_state_outward_flux_interval,
            )
            for pixel in pixels
        )
    )
    raw_input_bits = max(raw_input_bits, parent_peak_bits)
    if raw_input_bits > certificate.work_transcript.maximum_rational_bits:
        _raise_raw_rational_size_failure(
            raw_input_bits,
            "local detector parent/input exceeds the rational bit policy",
        )
    ledger = _checked_policy(
        certificate.work_transcript.maximum_work,
        certificate.work_transcript.maximum_rational_bits,
    )
    ledger.rational_peak_bits = raw_input_bits
    dose = (
        certificate.incident_electron_count.lower,
        certificate.incident_electron_count.upper,
    )
    if dose[0] < _ZERO:
        ledger.fail(
            GalerkinLocalDetectorFailure.DOSE_INVALID,
            "local detector dose interval must be nonnegative",
        )
    incident: list[_Interval] = []
    incident_points: list[float] = []
    for pixel in pixels:
        terminal = pixel.positive_port.terminal_certificate
        projection = terminal.projection_certificate
        zero_slab = projection.zero_slab_certificate
        represented = zero_slab.represented_source_certificate
        modes = represented.source.modes
        interval = (
            Fraction.from_float(
                float(np.asarray(modes.exact_reduced_flux_lower_bound))
            ),
            Fraction.from_float(
                float(np.asarray(modes.exact_reduced_flux_upper_bound))
            ),
        )
        if interval[0] <= _ZERO:
            return _detector_parent_failure_evidence(
                certificate,
                failure=GalerkinLocalDetectorFailure.INCIDENT_FLUX_NONPOSITIVE,
            )
        incident.append(interval)
        incident_points.append(float(np.asarray(modes.output_reduced_flux)))
    incident_tuple = tuple(incident)
    exact_state_mode = tuple(
        tuple(
            (value.lower, value.upper)
            for value in pixel.exact_state_pixel_flux_intervals
        )
        for pixel in pixels
    )
    production_mode = tuple(
        tuple(
            (value.lower, value.upper)
            for value in pixel.production_quadratic_intervals
        )
        for pixel in pixels
    )
    outward_exact = tuple(
        (
            pixel.exact_state_outward_flux_interval.lower,
            pixel.exact_state_outward_flux_interval.upper,
        )
        for pixel in pixels
    )
    try:
        passivity = _outward_passivity_margins(
            incident_tuple, outward_exact, ledger
        )
    except ValueError:
        ledger.fail(
            GalerkinLocalDetectorFailure.PIXEL_FORM_NONPASSIVE,
            "local detector exact-state outward passivity failed",
        )
    weights = tuple(
        Fraction(numerator, denominator)
        for numerator, denominator in zip(
            certificate.ensemble_weight_numerators,
            certificate.ensemble_weight_denominators,
            strict=True,
        )
    )
    try:
        exact_fractions, exact_ideal = _normalize_mix_and_dose(
            exact_state_mode, incident_tuple, weights, dose, ledger
        )
    except ValueError:
        ledger.fail(
            GalerkinLocalDetectorFailure.ENSEMBLE_WEIGHT_INVALID,
            "local detector exact ensemble validation failed",
        )
    production_fractions: list[_Intervals] = []
    for mode, denominator in zip(production_mode, incident_tuple, strict=True):
        production_fractions.append(
            tuple(ledger.divide_positive(value, denominator) for value in mode)
        )
    mode_points = _production_mode_points(pixels)
    incident_point_array = np.asarray(incident_points, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        fraction_points = mode_points / incident_point_array[:, None]
    fraction_trace = _make_charged_production_trace(
        tuple(value for row in production_fractions for value in row),
        fraction_points,
        stage=GalerkinLocalDetectorProductionStage.MODE_PIXEL_FRACTION,
        quantity="mode_pixel_fraction",
        logical_shape=(mode_count, pixel_count),
        ledger=ledger,
    )
    flat_fraction_hulls = _trace_hull_intervals(fraction_trace)
    fraction_hulls: _ModePixelIntervals = tuple(
        tuple(
            flat_fraction_hulls[mode * pixel_count + pixel]
            for pixel in range(pixel_count)
        )
        for mode in range(mode_count)
    )
    weight_raw = tuple((value, value) for value in weights)
    weight_points = np.asarray(
        [float(value) for value in weights], dtype=np.float64
    )
    weight_trace = _make_charged_production_trace(
        weight_raw,
        weight_points,
        stage=GalerkinLocalDetectorProductionStage.ENSEMBLE_WEIGHT,
        quantity="ensemble_weight",
        logical_shape=(mode_count,),
        ledger=ledger,
    )
    weight_hulls = _trace_hull_intervals(weight_trace)
    dose_point = np.asarray(
        [float(np.asarray(certificate.incident_electron_count_point))],
        dtype=np.float64,
    )
    dose_trace = _make_charged_production_trace(
        (dose,),
        dose_point,
        stage=GalerkinLocalDetectorProductionStage.INCIDENT_DOSE,
        quantity="incident_electron_count",
        logical_shape=(),
        ledger=ledger,
    )
    dose_hull = _trace_hull_intervals(dose_trace)[0]
    production_ideal_raw = _mix_fraction_hulls(
        fraction_hulls, weight_hulls, dose_hull, ledger
    )
    with np.errstate(over="ignore", invalid="ignore"):
        production_ideal_points = (
            np.float64(dose_point[0])
            * np.sum(
                weight_points[:, None] * fraction_points,
                axis=0,
                dtype=np.float64,
            )
        ).astype(np.float64)
    ideal_trace = _make_charged_production_trace(
        production_ideal_raw,
        production_ideal_points,
        stage=GalerkinLocalDetectorProductionStage.IDEAL_ARRIVAL_MEAN,
        quantity="ideal_arrival_mean",
        logical_shape=(pixel_count,),
        ledger=ledger,
    )
    ideal_hulls = _trace_hull_intervals(ideal_trace)
    response = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in np.asarray(certificate.response_matrix)
    )
    background = tuple(
        (
            Fraction.from_float(float(value)),
            Fraction.from_float(float(value)),
        )
        for value in np.asarray(certificate.pre_gain_background)
    )
    try:
        production_pre_gain_raw = _apply_nonnegative_response(
            ideal_hulls, response, background, ledger
        )
        exact_pre_gain = _apply_nonnegative_response(
            exact_ideal, response, background, ledger
        )
    except ValueError:
        ledger.fail(
            GalerkinLocalDetectorFailure.RESPONSE_NOT_SUBSTOCHASTIC,
            "local detector exact response validation failed",
        )
    response_array = np.asarray(certificate.response_matrix, dtype=np.float64)
    background_array = np.asarray(
        certificate.pre_gain_background, dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        production_pre_gain_points = (
            response_array @ production_ideal_points + background_array
        ).astype(np.float64)
    pre_gain_trace = _make_charged_production_trace(
        production_pre_gain_raw,
        production_pre_gain_points,
        stage=GalerkinLocalDetectorProductionStage.PRE_GAIN_RESPONSE_MEAN,
        quantity="production_pre_gain_mean",
        logical_shape=(channel_count,),
        ledger=ledger,
    )
    production_pre_gain_singletons = tuple(
        (value.lower, value.upper)
        for value in pre_gain_trace.exact_point_intervals
    )
    common = {
        "maximum_count_ceiling": certificate.maximum_count_ceiling,
        "maximum_work": certificate.maximum_poisson_work,
        "maximum_rational_bits": certificate.maximum_poisson_rational_bits,
        "exp_precision_bits": certificate.exp_precision_bits,
        "maximum_exp_terms": certificate.maximum_exp_terms,
        "maximum_exp_work": certificate.maximum_exp_work,
        "maximum_exp_range_reductions": (
            certificate.maximum_exp_range_reductions
        ),
    }
    censored: list[_Interval | None] = []
    mean_transcripts: list[CensoredPoissonWorkTranscript | None] = []
    mean_failures: list[object | None] = []
    helper_work = 0
    failure = GalerkinLocalDetectorFailure.NONE
    for channel, (mean, ceiling) in enumerate(
        zip(
            exact_pre_gain, np.asarray(certificate.count_ceilings), strict=True
        )
    ):
        try:
            value, work = enclose_censored_poisson_mean(
                mean, int(ceiling), **common
            )
        except CensoredPoissonEnclosureError as error:
            censored.append(None)
            mean_transcripts.append(None)
            mean_failures.append(
                _helper_failure_from_error(
                    error,
                    call=(
                        GalerkinLocalDetectorHelperCall.EXACT_STATE_CENSORED_MEAN
                    ),
                    channel=channel,
                )
            )
            helper_work += _failed_helper_work(error)
            failure |= GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
            if error.nested_failure is not None:
                failure |= GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
        else:
            censored.append(value)
            mean_transcripts.append(work)
            mean_failures.append(None)
            helper_work += _nested_entire_work(work)
    production_censored_raw: list[_Interval | None] = []
    production_mean_transcripts: list[
        CensoredPoissonWorkTranscript | None
    ] = []
    production_mean_failures: list[object | None] = []
    for channel, (mean, ceiling) in enumerate(
        zip(
            production_pre_gain_singletons,
            np.asarray(certificate.count_ceilings),
            strict=True,
        )
    ):
        try:
            value, helper = enclose_censored_poisson_mean(
                mean, int(ceiling), **common
            )
        except CensoredPoissonEnclosureError as error:
            production_censored_raw.append(None)
            production_mean_transcripts.append(None)
            production_mean_failures.append(
                _helper_failure_from_error(
                    error,
                    call=(
                        GalerkinLocalDetectorHelperCall.PRODUCTION_CENSORED_MEAN
                    ),
                    channel=channel,
                )
            )
            helper_work += _failed_helper_work(error)
            failure |= GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
            if error.nested_failure is not None:
                failure |= GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
        else:
            production_censored_raw.append(value)
            production_mean_transcripts.append(helper)
            production_mean_failures.append(None)
            helper_work += _nested_entire_work(helper)
    ledger.failure_context = {
        "censored_mean_transcripts": tuple(mean_transcripts),
        "censored_mean_failures": tuple(mean_failures),
        "production_censored_mean_transcripts": tuple(
            production_mean_transcripts
        ),
        "production_censored_mean_failures": tuple(production_mean_failures),
        "nested_helper_work_count_exact": str(helper_work),
        "helper_failure_mask": int(failure),
        "exact_state_censored_mean_evidence_available": not any(
            value is None for value in censored
        ),
        "production_censored_mean_evidence_available": not any(
            value is None for value in production_censored_raw
        ),
    }
    if failure:
        ledger.fail(
            GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE,
            "local detector censored-mean helper evidence is unavailable",
        )
    complete_censored = cast(_Intervals, tuple(censored))
    complete_production_censored = cast(
        _Intervals, tuple(production_censored_raw)
    )
    production_censored_points = tuple(
        _charged_midpoint_production_point(value, ledger)
        for value in complete_production_censored
    )
    censored_trace = _make_charged_production_trace(
        complete_production_censored,
        np.asarray(production_censored_points, dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.CENSORED_COUNT_MEAN,
        quantity="production_censored_count_mean",
        logical_shape=(channel_count,),
        ledger=ledger,
    )
    gains = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(certificate.deterministic_gain)
    )
    offsets = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(certificate.electronic_offset)
    )
    digitized = _apply_gain_and_offset(
        complete_censored, gains, offsets, ledger
    )
    production_digitized_raw = _apply_gain_and_offset(
        _trace_hull_intervals(censored_trace), gains, offsets, ledger
    )
    with np.errstate(over="ignore", invalid="ignore"):
        production_digitized_points = (
            np.asarray(certificate.deterministic_gain, dtype=np.float64)
            * np.asarray(production_censored_points, dtype=np.float64)
            + np.asarray(certificate.electronic_offset, dtype=np.float64)
        ).astype(np.float64)
    digitized_trace = _make_charged_production_trace(
        production_digitized_raw,
        production_digitized_points,
        stage=GalerkinLocalDetectorProductionStage.POST_CENSOR_DIGITIZED_MEAN,
        quantity="production_digitized_mean",
        logical_shape=(channel_count,),
        ledger=ledger,
    )
    traces = (
        fraction_trace,
        weight_trace,
        dose_trace,
        ideal_trace,
        pre_gain_trace,
        censored_trace,
        digitized_trace,
    )
    if ledger.exact_work_count != planned:
        raise RuntimeError(
            "local detector preflight and completed local work disagree"
        )
    if ledger.production_trace_count != len(traces):
        raise RuntimeError("local detector production trace ledger disagrees")
    nested_parent_work = sum(
        pixel.work_transcript.exact_work_count
        + int(pixel.work_transcript.nested_parent_work_count_exact)
        + int(pixel.work_transcript.nested_helper_work_count_exact)
        for pixel in pixels
    )
    nested_traces = sum(
        pixel.work_transcript.production_trace_count
        + pixel.work_transcript.nested_production_trace_count
        for pixel in pixels
    )
    nested_hulls = sum(
        pixel.work_transcript.hull_endpoint_count
        + pixel.work_transcript.nested_hull_endpoint_count
        for pixel in pixels
    )
    work = _make_local_detector_work_transcript(
        algorithm=ledger.algorithm,
        maximum_work=ledger.maximum_work,
        maximum_rational_bits=ledger.maximum_rational_bits,
        coordinate_factor_count=ledger.coordinate_factor_count,
        pixel_product_count=ledger.pixel_product_count,
        mode_quadratic_count=ledger.mode_quadratic_count,
        ensemble_product_count=ledger.ensemble_product_count,
        response_product_count=ledger.response_product_count,
        production_trace_count=ledger.production_trace_count,
        hull_endpoint_count=ledger.hull_endpoint_count,
        nested_production_trace_count=nested_traces,
        nested_hull_endpoint_count=nested_hulls,
        exact_work_count=ledger.exact_work_count,
        rational_peak_bits=ledger.rational_peak_bits,
        nested_parent_work_count_exact=str(nested_parent_work),
        nested_helper_work_count_exact=str(helper_work),
    )
    result: dict[str, object] = {
        "production_evidence_available": True,
        "exact_state_censored_mean_evidence_available": True,
        "production_censored_mean_evidence_available": True,
        "incident_reduced_flux_intervals": _interval_carriers(incident_tuple),
        "mode_exact_state_pixel_flux_intervals": tuple(
            _interval_carriers(row) for row in exact_state_mode
        ),
        "mode_production_quadratic_intervals": tuple(
            _interval_carriers(row) for row in production_mode
        ),
        "mode_pixel_form_norm_upper_intervals": tuple(
            pixel.pixel_form_norm_upper_intervals for pixel in pixels
        ),
        "mode_production_to_exact_x_amplitude_error_intervals": tuple(
            pixel.production_to_exact_x_amplitude_error_interval
            for pixel in pixels
        ),
        "mode_state_radius_amplitude_error_intervals": tuple(
            pixel.state_radius_amplitude_error_interval for pixel in pixels
        ),
        "mode_exact_state_amplitude_error_intervals": tuple(
            pixel.exact_state_amplitude_error_interval for pixel in pixels
        ),
        "mode_production_amplitude_norm_intervals": tuple(
            pixel.production_amplitude_norm_interval for pixel in pixels
        ),
        "mode_production_realization_error_upper_intervals": tuple(
            pixel.production_realization_error_upper_intervals
            for pixel in pixels
        ),
        "mode_state_radius_incremental_error_upper_intervals": tuple(
            pixel.state_radius_incremental_error_upper_intervals
            for pixel in pixels
        ),
        "mode_combined_exact_state_error_upper_intervals": tuple(
            pixel.combined_exact_state_error_upper_intervals
            for pixel in pixels
        ),
        "mode_outward_passivity_margin_intervals": _interval_carriers(
            passivity
        ),
        "mode_pixel_fraction_intervals": tuple(
            _interval_carriers(row) for row in exact_fractions
        ),
        "ideal_arrival_mean_intervals": _interval_carriers(exact_ideal),
        "production_pre_gain_mean_point_intervals": _interval_carriers(
            production_pre_gain_singletons
        ),
        "exact_state_pre_gain_mean_intervals": _interval_carriers(
            exact_pre_gain
        ),
        "censored_mean_intervals": _interval_carriers(complete_censored),
        "expected_digitized_mean_intervals": _interval_carriers(digitized),
        "production_traces": traces,
        "work_transcript": work,
        "censored_mean_transcripts": tuple(mean_transcripts),
        "censored_mean_failures": tuple(mean_failures),
        "production_censored_mean_transcripts": tuple(
            production_mean_transcripts
        ),
        "production_censored_mean_failures": tuple(production_mean_failures),
        "detector_eligible": True,
        "likelihood_law_eligible": True,
        "failure_mask": int(GalerkinLocalDetectorFailure.NONE),
    }
    result.update(_detector_scope_evidence())
    return result


def _expected_local_censored_poisson_detector(
    certificate: GalerkinLocalCensoredPoissonDetector,
) -> dict[str, object]:
    """PRIVATE: Replay the canonical detector evidence and typed stops.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonDetector
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    try:
        return _expected_local_censored_poisson_detector_core(certificate)
    except _DetectorArithmeticError as error:
        mode_count = len(certificate.pixel_forms)
        pixel_count = certificate.pixel_forms[0].pixel_count
        channel_count = np.asarray(certificate.response_matrix).shape[0]
        planned = _planned_detector_exact_work(
            mode_count=mode_count,
            pixel_count=pixel_count,
            channel_count=channel_count,
        )
        nested_parent = sum(
            pixel.work_transcript.exact_work_count
            + int(pixel.work_transcript.nested_parent_work_count_exact)
            + int(pixel.work_transcript.nested_helper_work_count_exact)
            for pixel in certificate.pixel_forms
        )
        nested_traces = sum(
            pixel.work_transcript.production_trace_count
            + pixel.work_transcript.nested_production_trace_count
            for pixel in certificate.pixel_forms
        )
        nested_hulls = sum(
            pixel.work_transcript.hull_endpoint_count
            + pixel.work_transcript.nested_hull_endpoint_count
            for pixel in certificate.pixel_forms
        )
        scientific_failures = (
            GalerkinLocalDetectorFailure.PIXEL_FORM_NONPASSIVE,
            GalerkinLocalDetectorFailure.ENSEMBLE_WEIGHT_INVALID,
            GalerkinLocalDetectorFailure.DOSE_INVALID,
            GalerkinLocalDetectorFailure.RESPONSE_NOT_SUBSTOCHASTIC,
        )
        snapshot = error.work_snapshot
        work = _make_local_detector_work_transcript(
            algorithm="exact_fraction_local_detector_v1",
            maximum_work=certificate.work_transcript.maximum_work,
            maximum_rational_bits=(
                certificate.work_transcript.maximum_rational_bits
            ),
            coordinate_factor_count=snapshot["coordinate_factor_count"],
            pixel_product_count=snapshot["pixel_product_count"],
            mode_quadratic_count=snapshot["mode_quadratic_count"],
            ensemble_product_count=snapshot["ensemble_product_count"],
            response_product_count=snapshot["response_product_count"],
            production_trace_count=snapshot["production_trace_count"],
            hull_endpoint_count=snapshot["hull_endpoint_count"],
            nested_production_trace_count=nested_traces,
            nested_hull_endpoint_count=nested_hulls,
            exact_work_count=snapshot["exact_work_count"],
            rational_peak_bits=snapshot["rational_peak_bits"],
            nested_parent_work_count_exact=str(nested_parent),
            nested_helper_work_count_exact=str(
                error.failure_context.get(
                    "nested_helper_work_count_exact", "0"
                )
            ),
            planned_exact_work_count_exact=str(planned),
            attempted_exact_work_count_exact=str(error.exact_work_count),
            completed_successfully=False,
            arithmetic_failure=(
                GalerkinLocalDetectorFailure.NONE
                if error.failure in scientific_failures
                else error.failure
            ),
            scientific_failure=(
                error.failure
                if error.failure in scientific_failures
                else GalerkinLocalDetectorFailure.NONE
            ),
        )
        prior_helper_failure = GalerkinLocalDetectorFailure(
            int(cast(int, error.failure_context.get("helper_failure_mask", 0)))
        )
        result = _detector_parent_failure_evidence(
            certificate, failure=error.failure | prior_helper_failure
        )
        result["production_traces"] = error.production_traces
        result["work_transcript"] = work
        for name in (
            "exact_state_censored_mean_evidence_available",
            "production_censored_mean_evidence_available",
            "censored_mean_transcripts",
            "censored_mean_failures",
            "production_censored_mean_transcripts",
            "production_censored_mean_failures",
        ):
            if name in error.failure_context:
                result[name] = error.failure_context[name]
        return result


def _likelihood_scope_evidence() -> dict[str, str]:
    """PRIVATE: Return fixed pre-gain likelihood semantic scopes.

    Returns
    -------
    result : dict[str, str]
        Canonical derived result.
    """
    return {
        "likelihood_scope": (
            "full-channel censored-Poisson probabilities are enclosed over "
            "C=hull(exact-state pre-gain mean interval, rounded production "
            "mean point); the fixed fit mask affects only the objective"
        ),
        "nll_scope": (
            "fit-only NLL uses exact positive probability floors on C with "
            "no epsilon; production-point and admitted-hull NLL evidence "
            "remain distinct and deterministic gain/offset are excluded"
        ),
        "no_derivative_scope": (
            "stopped RM-S4 detector evidence; derivative eligibility belongs "
            "only to an independently invoked RM-I1 chart"
        ),
    }


def _planned_likelihood_exact_work(
    *, channel_count: int, fitted_count: int
) -> int:
    """PRIVATE: Return fixed point, trace, and total-NLL exact work.

    Parameters
    ----------
    channel_count : int
        Required canonical input.
    fitted_count : int
        Required canonical input.

    Returns
    -------
    result : int
        Canonical derived result.
    """
    return 4 * channel_count + 6 * fitted_count


def _likelihood_parent_work(
    detector: GalerkinLocalCensoredPoissonDetector,
) -> tuple[int, int, int]:
    """PRIVATE: Aggregate the entire nested detector resource tree.

    Parameters
    ----------
    detector : GalerkinLocalCensoredPoissonDetector
        Required canonical input.

    Returns
    -------
    exact : int
        Canonical derived result.
    traces : int
        Canonical derived result.
    hulls : int
        Canonical derived result.
    """
    transcript = detector.work_transcript
    exact = (
        transcript.exact_work_count
        + int(transcript.nested_parent_work_count_exact)
        + int(transcript.nested_helper_work_count_exact)
    )
    traces = (
        transcript.production_trace_count
        + transcript.nested_production_trace_count
    )
    hulls = (
        transcript.hull_endpoint_count + transcript.nested_hull_endpoint_count
    )
    return exact, traces, hulls


def _likelihood_unavailable_evidence(
    certificate: GalerkinLocalCensoredPoissonLikelihood,
    *,
    failure: GalerkinLocalDetectorFailure,
    planned: int | None = None,
) -> dict[str, object]:
    """PRIVATE: Build one parent/input/preflight likelihood stop.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonLikelihood
        Required canonical input.
    failure : GalerkinLocalDetectorFailure
        Required canonical input.
    planned : int | None
        Optional input; the signature supplies its default.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    parent_work, parent_traces, parent_hulls = _likelihood_parent_work(
        certificate.detector
    )
    work = _make_local_detector_work_transcript(
        algorithm="exact_fraction_local_detector_v1",
        maximum_work=certificate.work_transcript.maximum_work,
        maximum_rational_bits=(
            certificate.work_transcript.maximum_rational_bits
        ),
        coordinate_factor_count=0,
        pixel_product_count=0,
        mode_quadratic_count=0,
        ensemble_product_count=0,
        response_product_count=0,
        production_trace_count=0,
        hull_endpoint_count=0,
        nested_production_trace_count=parent_traces,
        nested_hull_endpoint_count=parent_hulls,
        exact_work_count=0,
        rational_peak_bits=0,
        nested_parent_work_count_exact=str(parent_work),
        planned_exact_work_count_exact=(
            None if planned is None else str(planned)
        ),
        attempted_exact_work_count_exact=(
            None if planned is None else str(planned)
        ),
        completed_successfully=planned is None,
        arithmetic_failure=(
            GalerkinLocalDetectorFailure.NONE if planned is None else failure
        ),
        preflight_failed=planned is not None,
        count_overflow=(
            False if planned is None else planned > _MAXIMUM_SIGNED_INT64
        ),
    )
    result: dict[str, object] = {
        "likelihood_evidence_available": False,
        "admitted_pre_gain_mean_hull_intervals": (),
        "production_probability_point_intervals": (),
        "admitted_hull_probability_intervals": (),
        "fitted_probability_positive_floor_intervals": (),
        "production_nll_point_intervals": (),
        "admitted_hull_nll_intervals": (),
        "total_nll_interval": None,
        "production_probability_transcripts": (),
        "production_probability_failures": (),
        "admitted_hull_probability_transcripts": (),
        "admitted_hull_probability_failures": (),
        "production_nll_transcripts": (),
        "production_nll_failures": (),
        "admitted_hull_nll_transcripts": (),
        "admitted_hull_nll_failures": (),
        "production_traces": (),
        "work_transcript": work,
        "likelihood_law_eligible": False,
        "nll_eligible": False,
        "failure_mask": int(failure),
    }
    result.update(_likelihood_scope_evidence())
    return result


def _expected_local_censored_poisson_likelihood_core(  # noqa: PLR0912, PLR0915
    certificate: GalerkinLocalCensoredPoissonLikelihood,
) -> dict[str, object]:
    """PRIVATE: Recompute admitted-hull probability and NLL evidence.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonLikelihood
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.

    Raises
    ------
    RuntimeError
        If the canonical contract is violated.
    """
    detector = certificate.detector
    channel_count = np.asarray(detector.response_matrix).shape[0]
    observed = tuple(
        int(value) for value in np.asarray(certificate.observed_counts)
    )
    ceilings = tuple(
        int(value) for value in np.asarray(detector.count_ceilings)
    )
    fit_mask = tuple(bool(value) for value in np.asarray(detector.fit_mask))
    fitted_count = sum(fit_mask)
    if (
        not bool(np.asarray(detector.production_evidence_available))
        or not bool(np.asarray(detector.detector_eligible))
        or not bool(np.asarray(detector.likelihood_law_eligible))
    ):
        inherited = GalerkinLocalDetectorFailure(
            int(np.asarray(detector.failure_mask))
        )
        if inherited is GalerkinLocalDetectorFailure.NONE:
            inherited = GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
        return _likelihood_unavailable_evidence(
            certificate,
            failure=inherited,
        )
    if any(
        value < 0 or value > ceiling
        for value, ceiling in zip(observed, ceilings, strict=True)
    ):
        return _likelihood_unavailable_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.COUNT_DOMAIN_INVALID,
        )
    planned = _planned_likelihood_exact_work(
        channel_count=channel_count, fitted_count=fitted_count
    )
    if planned > _MAXIMUM_SIGNED_INT64:
        return _likelihood_unavailable_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.EXACT_WORK_COUNT_OVERFLOW,
            planned=planned,
        )
    if planned > certificate.work_transcript.maximum_work:
        return _likelihood_unavailable_evidence(
            certificate,
            failure=GalerkinLocalDetectorFailure.EXACT_WORK_BUDGET_EXCEEDED,
            planned=planned,
        )
    parent_peak_bits = _stored_interval_peak_bits(
        (
            detector.production_pre_gain_mean_point_intervals,
            detector.exact_state_pre_gain_mean_intervals,
        )
    )
    if parent_peak_bits > certificate.work_transcript.maximum_rational_bits:
        _raise_raw_rational_size_failure(
            parent_peak_bits,
            "local detector parent means exceed the likelihood rational "
            "policy",
        )
    ledger = _checked_policy(
        certificate.work_transcript.maximum_work,
        certificate.work_transcript.maximum_rational_bits,
    )
    ledger.rational_peak_bits = parent_peak_bits
    production_means = tuple(
        (value.lower, value.upper)
        for value in detector.production_pre_gain_mean_point_intervals
    )
    exact_means = tuple(
        (value.lower, value.upper)
        for value in detector.exact_state_pre_gain_mean_intervals
    )
    admitted = tuple(
        (
            min(exact[0], point[0]),
            max(exact[1], point[1]),
        )
        for point, exact in zip(production_means, exact_means, strict=True)
    )
    common = {
        "maximum_count_ceiling": detector.maximum_count_ceiling,
        "maximum_work": detector.maximum_poisson_work,
        "maximum_rational_bits": detector.maximum_poisson_rational_bits,
        "exp_precision_bits": detector.exp_precision_bits,
        "maximum_exp_terms": detector.maximum_exp_terms,
        "maximum_exp_work": detector.maximum_exp_work,
        "maximum_exp_range_reductions": (
            detector.maximum_exp_range_reductions
        ),
    }
    log = {
        "log_precision_bits": certificate.log_precision_bits,
        "maximum_log_terms": certificate.maximum_log_terms,
        "maximum_log_work": certificate.maximum_log_work,
        "maximum_log_range_reductions": (
            certificate.maximum_log_range_reductions
        ),
    }
    production_probabilities: list[_Interval | None] = []
    admitted_probabilities: list[_Interval | None] = []
    production_probability_work: list[
        CensoredPoissonWorkTranscript | None
    ] = []
    production_probability_failures: list[object | None] = []
    admitted_probability_work: list[CensoredPoissonWorkTranscript | None] = []
    admitted_probability_failures: list[object | None] = []
    helper_work = 0
    failure = GalerkinLocalDetectorFailure.NONE
    for channel, (point, hull, value, ceiling) in enumerate(
        zip(production_means, admitted, observed, ceilings, strict=True)
    ):
        try:
            probability, work = enclose_censored_poisson_probability(
                point, value, ceiling, **common
            )
        except CensoredPoissonEnclosureError as error:
            production_probabilities.append(None)
            production_probability_work.append(None)
            production_probability_failures.append(
                _helper_failure_from_error(
                    error,
                    call=GalerkinLocalDetectorHelperCall.PRODUCTION_PROBABILITY,
                    channel=channel,
                )
            )
            helper_work += _failed_helper_work(error)
            failure |= GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
            if error.nested_failure is not None:
                failure |= GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
        else:
            production_probabilities.append(probability)
            production_probability_work.append(work)
            production_probability_failures.append(None)
            helper_work += _nested_entire_work(work)
        try:
            probability, work = enclose_censored_poisson_probability(
                hull, value, ceiling, **common
            )
        except CensoredPoissonEnclosureError as error:
            admitted_probabilities.append(None)
            admitted_probability_work.append(None)
            admitted_probability_failures.append(
                _helper_failure_from_error(
                    error,
                    call=(
                        GalerkinLocalDetectorHelperCall.ADMITTED_HULL_PROBABILITY
                    ),
                    channel=channel,
                )
            )
            helper_work += _failed_helper_work(error)
            failure |= GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
            if error.nested_failure is not None:
                failure |= GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
        else:
            admitted_probabilities.append(probability)
            admitted_probability_work.append(work)
            admitted_probability_failures.append(None)
            helper_work += _nested_entire_work(work)
    ledger.failure_context = {
        "production_probability_transcripts": tuple(
            production_probability_work
        ),
        "production_probability_failures": tuple(
            production_probability_failures
        ),
        "admitted_hull_probability_transcripts": tuple(
            admitted_probability_work
        ),
        "admitted_hull_probability_failures": tuple(
            admitted_probability_failures
        ),
        "nested_helper_work_count_exact": str(helper_work),
        "helper_failure_mask": int(failure),
        "likelihood_evidence_available": False,
        "likelihood_law_eligible": False,
        "nll_eligible": False,
    }
    if failure:
        ledger.fail(
            GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE,
            "local detector probability helper evidence is unavailable",
        )
    complete_production_probabilities = cast(
        _Intervals, tuple(production_probabilities)
    )
    complete_admitted_probabilities = cast(
        _Intervals, tuple(admitted_probabilities)
    )
    production_probability_points = tuple(
        _charged_midpoint_production_point(value, ledger)
        for value in complete_production_probabilities
    )
    probability_trace = _make_charged_production_trace(
        complete_production_probabilities,
        np.asarray(production_probability_points, dtype=np.float64),
        stage=GalerkinLocalDetectorProductionStage.CENSORED_PROBABILITY,
        quantity="production_censored_probability",
        logical_shape=(channel_count,),
        ledger=ledger,
    )
    floors = tuple(
        (
            None
            if not fitted or probability[0] <= _ZERO
            else (probability[0], probability[0])
        )
        for fitted, probability in zip(
            fit_mask, complete_admitted_probabilities, strict=True
        )
    )
    probability_context: dict[str, object] = {
        "likelihood_evidence_available": True,
        "admitted_pre_gain_mean_hull_intervals": _interval_carriers(admitted),
        "production_probability_point_intervals": _interval_carriers(
            tuple(
                (value.lower, value.upper)
                for value in probability_trace.exact_point_intervals
            )
        ),
        "admitted_hull_probability_intervals": _interval_carriers(
            complete_admitted_probabilities
        ),
        "fitted_probability_positive_floor_intervals": tuple(
            None if value is None else _interval_carrier(value)
            for value in floors
        ),
        "production_probability_transcripts": tuple(
            production_probability_work
        ),
        "production_probability_failures": tuple(
            production_probability_failures
        ),
        "admitted_hull_probability_transcripts": tuple(
            admitted_probability_work
        ),
        "admitted_hull_probability_failures": tuple(
            admitted_probability_failures
        ),
        "nested_helper_work_count_exact": str(helper_work),
        "helper_failure_mask": int(GalerkinLocalDetectorFailure.NONE),
        "likelihood_law_eligible": True,
        "nll_eligible": False,
    }
    ledger.failure_context = probability_context
    if any(
        fitted and floor is None
        for fitted, floor in zip(fit_mask, floors, strict=True)
    ):
        ledger.failure_context["helper_failure_mask"] = int(
            GalerkinLocalDetectorFailure.NLL_UNAVAILABLE
        )
        ledger.fail(
            GalerkinLocalDetectorFailure.NLL_UNAVAILABLE,
            "local detector fitted probability has no exact positive floor",
        )
    production_nlls: list[_Interval | None] = []
    admitted_nlls: list[_Interval | None] = []
    production_nll_work: list[CensoredPoissonWorkTranscript | None] = []
    production_nll_failures: list[object | None] = []
    admitted_nll_work: list[CensoredPoissonWorkTranscript | None] = []
    admitted_nll_failures: list[object | None] = []
    for channel, (point, hull, value, ceiling, fitted) in enumerate(
        zip(
            production_means,
            admitted,
            observed,
            ceilings,
            fit_mask,
            strict=True,
        )
    ):
        if not fitted:
            production_nlls.append(None)
            admitted_nlls.append(None)
            production_nll_work.append(None)
            production_nll_failures.append(None)
            admitted_nll_work.append(None)
            admitted_nll_failures.append(None)
        else:
            production_value: _Interval | None = None
            admitted_value: _Interval | None = None
            production_error: object | None = None
            admitted_error: object | None = None
            try:
                production_value, production_work = (
                    enclose_censored_poisson_nll(
                        point, value, ceiling, **common, **log
                    )
                )
            except CensoredPoissonEnclosureError as error:
                production_work = None
                production_error = _helper_failure_from_error(
                    error,
                    call=GalerkinLocalDetectorHelperCall.PRODUCTION_NLL,
                    channel=channel,
                )
                helper_work += _failed_helper_work(error)
                failure |= (
                    GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
                    | GalerkinLocalDetectorFailure.NLL_UNAVAILABLE
                )
                if error.nested_failure is not None:
                    failure |= (
                        GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
                    )
            else:
                helper_work += _nested_entire_work(production_work)
            try:
                admitted_value, admitted_work = enclose_censored_poisson_nll(
                    hull, value, ceiling, **common, **log
                )
            except CensoredPoissonEnclosureError as error:
                admitted_work = None
                admitted_error = _helper_failure_from_error(
                    error,
                    call=GalerkinLocalDetectorHelperCall.ADMITTED_HULL_NLL,
                    channel=channel,
                )
                helper_work += _failed_helper_work(error)
                failure |= (
                    GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE
                    | GalerkinLocalDetectorFailure.NLL_UNAVAILABLE
                )
                if error.nested_failure is not None:
                    failure |= (
                        GalerkinLocalDetectorFailure.NESTED_HELPER_FAILURE
                    )
            else:
                helper_work += _nested_entire_work(admitted_work)
            production_nlls.append(production_value)
            admitted_nlls.append(admitted_value)
            production_nll_work.append(production_work)
            production_nll_failures.append(production_error)
            admitted_nll_work.append(admitted_work)
            admitted_nll_failures.append(admitted_error)
    ledger.failure_context = {
        **probability_context,
        "production_nll_transcripts": tuple(production_nll_work),
        "production_nll_failures": tuple(production_nll_failures),
        "admitted_hull_nll_transcripts": tuple(admitted_nll_work),
        "admitted_hull_nll_failures": tuple(admitted_nll_failures),
        "nested_helper_work_count_exact": str(helper_work),
        "helper_failure_mask": int(failure),
    }
    if failure:
        ledger.fail(
            GalerkinLocalDetectorFailure.POISSON_ENCLOSURE_FAILURE,
            "local detector NLL helper evidence is unavailable",
        )
    production_nll_points: list[_Interval | None] = []
    nll_traces: list[object] = []
    for channel, (fitted, value) in enumerate(
        zip(fit_mask, production_nlls, strict=True)
    ):
        if not fitted:
            production_nll_points.append(None)
            continue
        complete_value = cast(_Interval, value)
        point = _charged_midpoint_production_point(complete_value, ledger)
        trace = _make_charged_production_trace(
            (complete_value,),
            np.asarray([point], dtype=np.float64),
            stage=GalerkinLocalDetectorProductionStage.CENSORED_NLL,
            quantity=f"production_nll.channel_{channel}",
            logical_shape=(),
            ledger=ledger,
        )
        nll_traces.append(trace)
        production_nll_points.append(
            (
                trace.exact_point_intervals[0].lower,
                trace.exact_point_intervals[0].upper,
            )
        )
    total: _Interval = (_ZERO, _ZERO)
    for fitted, value in zip(fit_mask, admitted_nlls, strict=True):
        if not fitted:
            continue
        total = ledger.add_intervals(total, cast(_Interval, value))
    traces = (probability_trace, *nll_traces)
    if ledger.exact_work_count != planned:
        raise RuntimeError(
            "local detector likelihood preflight and completed work disagree"
        )
    parent_work, parent_traces, parent_hulls = _likelihood_parent_work(
        detector
    )
    work = _make_local_detector_work_transcript(
        algorithm=ledger.algorithm,
        maximum_work=ledger.maximum_work,
        maximum_rational_bits=ledger.maximum_rational_bits,
        coordinate_factor_count=ledger.coordinate_factor_count,
        pixel_product_count=ledger.pixel_product_count,
        mode_quadratic_count=ledger.mode_quadratic_count,
        ensemble_product_count=ledger.ensemble_product_count,
        response_product_count=ledger.response_product_count,
        production_trace_count=ledger.production_trace_count,
        hull_endpoint_count=ledger.hull_endpoint_count,
        nested_production_trace_count=parent_traces,
        nested_hull_endpoint_count=parent_hulls,
        exact_work_count=ledger.exact_work_count,
        rational_peak_bits=ledger.rational_peak_bits,
        nested_parent_work_count_exact=str(parent_work),
        nested_helper_work_count_exact=str(helper_work),
    )
    result: dict[str, object] = {
        "likelihood_evidence_available": True,
        "admitted_pre_gain_mean_hull_intervals": _interval_carriers(admitted),
        "production_probability_point_intervals": _interval_carriers(
            tuple(
                (value.lower, value.upper)
                for value in probability_trace.exact_point_intervals
            )
        ),
        "admitted_hull_probability_intervals": _interval_carriers(
            complete_admitted_probabilities
        ),
        "fitted_probability_positive_floor_intervals": tuple(
            None if value is None else _interval_carrier(value)
            for value in floors
        ),
        "production_nll_point_intervals": tuple(
            None if value is None else _interval_carrier(value)
            for value in production_nll_points
        ),
        "admitted_hull_nll_intervals": tuple(
            None if value is None else _interval_carrier(value)
            for value in admitted_nlls
        ),
        "total_nll_interval": _interval_carrier(total),
        "production_probability_transcripts": tuple(
            production_probability_work
        ),
        "production_probability_failures": tuple(
            production_probability_failures
        ),
        "admitted_hull_probability_transcripts": tuple(
            admitted_probability_work
        ),
        "admitted_hull_probability_failures": tuple(
            admitted_probability_failures
        ),
        "production_nll_transcripts": tuple(production_nll_work),
        "production_nll_failures": tuple(production_nll_failures),
        "admitted_hull_nll_transcripts": tuple(admitted_nll_work),
        "admitted_hull_nll_failures": tuple(admitted_nll_failures),
        "production_traces": traces,
        "work_transcript": work,
        "likelihood_law_eligible": True,
        "nll_eligible": True,
        "failure_mask": int(GalerkinLocalDetectorFailure.NONE),
    }
    result.update(_likelihood_scope_evidence())
    return result


def _expected_local_censored_poisson_likelihood(
    certificate: GalerkinLocalCensoredPoissonLikelihood,
) -> dict[str, object]:
    """PRIVATE: Replay likelihood success and typed stopped evidence.

    Parameters
    ----------
    certificate : GalerkinLocalCensoredPoissonLikelihood
        Required canonical input.

    Returns
    -------
    result : dict[str, object]
        Canonical derived result.
    """
    try:
        return _expected_local_censored_poisson_likelihood_core(certificate)
    except _DetectorArithmeticError as error:
        inherited = GalerkinLocalDetectorFailure(
            int(np.asarray(certificate.detector.failure_mask))
        )
        prior_failure = GalerkinLocalDetectorFailure(
            int(cast(int, error.failure_context.get("helper_failure_mask", 0)))
        )
        complete_failure = inherited | error.failure | prior_failure
        result = _likelihood_unavailable_evidence(
            certificate, failure=complete_failure
        )
        parent_work, parent_traces, parent_hulls = _likelihood_parent_work(
            certificate.detector
        )
        channel_count = np.asarray(certificate.observed_counts).shape[0]
        fitted_count = int(
            np.count_nonzero(np.asarray(certificate.detector.fit_mask))
        )
        planned = _planned_likelihood_exact_work(
            channel_count=channel_count, fitted_count=fitted_count
        )
        snapshot = error.work_snapshot
        scientific = (
            error.failure is GalerkinLocalDetectorFailure.NLL_UNAVAILABLE
        )
        work = _make_local_detector_work_transcript(
            algorithm="exact_fraction_local_detector_v1",
            maximum_work=certificate.work_transcript.maximum_work,
            maximum_rational_bits=(
                certificate.work_transcript.maximum_rational_bits
            ),
            coordinate_factor_count=snapshot["coordinate_factor_count"],
            pixel_product_count=snapshot["pixel_product_count"],
            mode_quadratic_count=snapshot["mode_quadratic_count"],
            ensemble_product_count=snapshot["ensemble_product_count"],
            response_product_count=snapshot["response_product_count"],
            production_trace_count=snapshot["production_trace_count"],
            hull_endpoint_count=snapshot["hull_endpoint_count"],
            nested_production_trace_count=parent_traces,
            nested_hull_endpoint_count=parent_hulls,
            exact_work_count=snapshot["exact_work_count"],
            rational_peak_bits=snapshot["rational_peak_bits"],
            nested_parent_work_count_exact=str(parent_work),
            nested_helper_work_count_exact=str(
                error.failure_context.get(
                    "nested_helper_work_count_exact", "0"
                )
            ),
            planned_exact_work_count_exact=str(planned),
            attempted_exact_work_count_exact=str(error.exact_work_count),
            completed_successfully=False,
            arithmetic_failure=(
                GalerkinLocalDetectorFailure.NONE
                if scientific
                else error.failure
            ),
            scientific_failure=(
                error.failure
                if scientific
                else GalerkinLocalDetectorFailure.NONE
            ),
        )
        result["production_traces"] = error.production_traces
        result["work_transcript"] = work
        for name in (
            "likelihood_evidence_available",
            "admitted_pre_gain_mean_hull_intervals",
            "production_probability_point_intervals",
            "admitted_hull_probability_intervals",
            "fitted_probability_positive_floor_intervals",
            "production_probability_transcripts",
            "production_probability_failures",
            "admitted_hull_probability_transcripts",
            "admitted_hull_probability_failures",
            "production_nll_transcripts",
            "production_nll_failures",
            "admitted_hull_nll_transcripts",
            "admitted_hull_nll_failures",
            "likelihood_law_eligible",
            "nll_eligible",
        ):
            if name in error.failure_context:
                result[name] = error.failure_context[name]
        result["failure_mask"] = int(complete_failure)
        return result


def _positive_port_from_prepared_terminal(
    terminal: GalerkinLocalVacuumTerminalCertificate,
    route: GalerkinLocalPositivePortRoute,
) -> GalerkinLocalPositivePortCertificate:
    """PRIVATE: Compose a canonical port carrier from replayed L8.

    Parameters
    ----------
    terminal : GalerkinLocalVacuumTerminalCertificate
        Required canonical input.
    route : GalerkinLocalPositivePortRoute
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalPositivePortCertificate
        Canonical derived result.
    """
    branch = terminal.branch_evidence
    expected = _expected_port_branches(terminal, route)
    amplitudes = np.asarray(branch.frozen_defining_branch_points)[:, 0]
    amplitude_errors = np.asarray(
        branch.exact_state_total_amplitude_error_bounds
    )[:, 0]
    roots = np.asarray(branch.frozen_positive_root_realizations)
    root_errors = np.asarray(branch.frozen_positive_root_error_bounds)
    exact_roots = tuple(
        (
            None
            if root is None or root.root_interval is None
            else _interval_carrier(
                (root.root_interval.lower, root.root_interval.upper)
            )
        )
        for root in branch.root_certificates
    )
    defining = branch.defining_branch_rectangles[0]
    rectangles = tuple(np.asarray(value) for value in defining)
    finite_values = (
        amplitudes.view(np.float64),
        *rectangles,
    )
    trace_available = all(
        bool(
            np.all(
                np.isfinite(value)
                & (
                    (np.abs(value) == 0.0)
                    | (np.abs(value) >= np.finfo(np.float64).tiny)
                )
            )
        )
        for value in finite_values
    )
    if trace_available:
        size = amplitudes.size
        real_raw = tuple(
            (
                Fraction.from_float(float(rectangles[0][index])),
                Fraction.from_float(float(rectangles[1][index])),
            )
            for index in range(size)
        )
        imag_raw = tuple(
            (
                Fraction.from_float(float(rectangles[2][index])),
                Fraction.from_float(float(rectangles[3][index])),
            )
            for index in range(size)
        )
        real_points = np.asarray(np.real(amplitudes), dtype=np.float64)
        imag_points = np.asarray(np.imag(amplitudes), dtype=np.float64)
        real_singletons = tuple(
            (
                Fraction.from_float(float(value)),
                Fraction.from_float(float(value)),
            )
            for value in real_points
        )
        imag_singletons = tuple(
            (
                Fraction.from_float(float(value)),
                Fraction.from_float(float(value)),
            )
            for value in imag_points
        )
        traces = (
            _make_local_detector_real_production_trace(
                _interval_carriers(real_raw),
                real_points,
                stage=GalerkinLocalDetectorProductionStage.L8_ROLE_ZERO_AMPLITUDE,
                quantity="l8_role_zero_amplitude.real",
                logical_shape=(size,),
            ),
            _make_local_detector_real_production_trace(
                _interval_carriers(imag_raw),
                imag_points,
                stage=GalerkinLocalDetectorProductionStage.L8_ROLE_ZERO_AMPLITUDE,
                quantity="l8_role_zero_amplitude.imag",
                logical_shape=(size,),
            ),
            _make_local_detector_real_production_trace(
                _interval_carriers(real_singletons),
                real_points,
                stage=GalerkinLocalDetectorProductionStage.POSITIVE_PORT_AMPLITUDE,
                quantity="positive_port_amplitude.real",
                logical_shape=(size,),
            ),
            _make_local_detector_real_production_trace(
                _interval_carriers(imag_singletons),
                imag_points,
                stage=GalerkinLocalDetectorProductionStage.POSITIVE_PORT_AMPLITUDE,
                quantity="positive_port_amplitude.imag",
                logical_shape=(size,),
            ),
        )
    else:
        traces = ()
    candidate = _make_local_positive_port_candidate(
        terminal_certificate=terminal,
        production_amplitudes=jnp.asarray(amplitudes, dtype=jnp.complex128),
        exact_state_total_amplitude_error_bounds=jnp.asarray(
            amplitude_errors, dtype=jnp.float64
        ),
        production_prediction_l2_norm_upper_bound=jnp.asarray(
            branch.production_prediction_l2_norm_upper_bound,
            dtype=jnp.float64,
        ),
        exact_state_prediction_error_l2_upper_bound=jnp.asarray(
            branch.exact_state_prediction_error_l2_upper_bound,
            dtype=jnp.float64,
        ),
        production_root_realizations=jnp.asarray(roots, dtype=jnp.float64),
        production_root_error_upper_bounds=jnp.asarray(
            root_errors, dtype=jnp.float64
        ),
        retained_propagating_mask=jnp.asarray(expected[1], dtype=jnp.bool_),
        zero_weight_mask=jnp.asarray(expected[2], dtype=jnp.bool_),
        positive_port_eligible=jnp.asarray(expected[4]),
        outgoing_radiation_eligible=jnp.asarray(expected[5]),
        failure_mask=jnp.asarray(int(expected[3]), dtype=jnp.int64),
        production_traces=traces,
        exact_root_intervals=exact_roots,
        parent_half_space_dispositions=branch.half_space_dispositions,
        branch_dispositions=expected[0],
        route=route,
        branch_role=branch.prediction_branch_role,
        branch_scope=_POSITIVE_PORT_BRANCH_SCOPE,
        exact_state_amplitude_scope=_POSITIVE_PORT_EXACT_STATE_SCOPE,
        root_realization_audit_scope=_POSITIVE_PORT_ROOT_AUDIT_SCOPE,
        completion_scope=_POSITIVE_PORT_COMPLETION_SCOPE,
        target_digest=terminal.target_digest,
        source_digest=terminal.source_digest,
        state_identity_digest=terminal.state_identity_digest,
        parent_terminal_identity_digest=terminal.terminal_identity_digest,
        parent_terminal_evidence_digest=terminal.terminal_evidence_digest,
        port_identity_digest="0" * 64,
        port_evidence_digest="0" * 64,
        certificate_digest="0" * 64,
    )
    return _make_local_positive_port_certificate(candidate)


def certify_local_positive_port(  # noqa: PLR0913
    terminal_certificate: GalerkinLocalVacuumTerminalCertificate,
    *,
    route: GalerkinLocalPositivePortRoute | str,
    disposition: GalerkinLocalVacuumTerminalDisposition | str,
    maximum_state_error: object,
    maximum_stability_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_gram_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_terminal_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_branch_direct_terms: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_cut_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_root_work: int = _DEFAULT_L8_ROOT_WORK,
    precision_bits: int = _DEFAULT_L8_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_L8_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_L8_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalPositivePortCertificate:
    """Replay L8 and compose one explicit projected or outgoing port.

    :see: :func:`~.test_detector.\
test_parent_port_route_status_and_source_identity_are_bound`
    """
    _assert_concrete((terminal_certificate, maximum_state_error))
    terminal = prepare_local_vacuum_terminal_certificate(
        terminal_certificate,
        disposition=disposition,
        maximum_state_error=maximum_state_error,
        maximum_stability_direct_pairs=maximum_stability_direct_pairs,
        maximum_gram_pairs=maximum_gram_pairs,
        maximum_terminal_direct_pairs=maximum_terminal_direct_pairs,
        maximum_branch_direct_terms=maximum_branch_direct_terms,
        maximum_cut_direct_pairs=maximum_cut_direct_pairs,
        maximum_root_work=maximum_root_work,
        precision_bits=precision_bits,
        maximum_terms=maximum_terms,
        maximum_entire_work=maximum_entire_work,
        maximum_range_reductions=maximum_range_reductions,
        maximum_interval_work=maximum_interval_work,
        maximum_rational_bits=maximum_rational_bits,
    )
    checked_route = (
        route
        if isinstance(route, GalerkinLocalPositivePortRoute)
        else GalerkinLocalPositivePortRoute(route)
    )
    return _positive_port_from_prepared_terminal(terminal, checked_route)


def prepare_local_positive_port_certificate(  # noqa: PLR0913
    certificate: GalerkinLocalPositivePortCertificate,
    *,
    route: GalerkinLocalPositivePortRoute | str,
    disposition: GalerkinLocalVacuumTerminalDisposition | str,
    maximum_state_error: object,
    maximum_stability_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_gram_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_terminal_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_branch_direct_terms: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_cut_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_root_work: int = _DEFAULT_L8_ROOT_WORK,
    precision_bits: int = _DEFAULT_L8_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_L8_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_L8_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalPositivePortCertificate:
    """Replay an independently specified L8 port and exact-compare storage.

    :see: :func:`~.test_detector.\
test_public_l9_chain_prepares_and_rejects_policy_and_coherent_forgeries`
    """
    _assert_concrete((certificate, maximum_state_error))
    if not isinstance(certificate, GalerkinLocalPositivePortCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalPositivePortCertificate"
        )
    canonical = certify_local_positive_port(
        certificate.terminal_certificate,
        route=route,
        disposition=disposition,
        maximum_state_error=maximum_state_error,
        maximum_stability_direct_pairs=maximum_stability_direct_pairs,
        maximum_gram_pairs=maximum_gram_pairs,
        maximum_terminal_direct_pairs=maximum_terminal_direct_pairs,
        maximum_branch_direct_terms=maximum_branch_direct_terms,
        maximum_cut_direct_pairs=maximum_cut_direct_pairs,
        maximum_root_work=maximum_root_work,
        precision_bits=precision_bits,
        maximum_terms=maximum_terms,
        maximum_entire_work=maximum_entire_work,
        maximum_range_reductions=maximum_range_reductions,
        maximum_interval_work=maximum_interval_work,
        maximum_rational_bits=maximum_rational_bits,
    )
    if not bool(eqx.tree_equal(canonical, certificate, typematch=True)):
        raise ValueError(
            "local positive-port certificate failed complete replay"
        )
    return canonical


def create_local_passive_pixel_input_manifest(  # noqa: PLR0913
    *,
    maximum_state_error: object,
    node_to_pixel: object,
    quadrature_weight_intervals: tuple[
        GalerkinLocalDetectorRationalInterval, ...
    ],
    quadrature_weight_points: object,
    aperture_efficiency_intervals: tuple[
        GalerkinLocalDetectorRationalInterval, ...
    ],
    aperture_efficiency_points: object,
    route: GalerkinLocalPositivePortRoute | str,
    disposition: GalerkinLocalVacuumTerminalDisposition | str,
    coordinate_convention: GalerkinLocalDetectorCoordinateConvention | str,
    pixel_count: int,
    maximum_stability_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_gram_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_terminal_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_branch_direct_terms: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_cut_direct_pairs: int = _DEFAULT_L8_DIRECT_WORK,
    maximum_root_work: int = _DEFAULT_L8_ROOT_WORK,
    precision_bits: int = _DEFAULT_L8_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_L8_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_range_reductions: int = _DEFAULT_L8_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_l8_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    maximum_detector_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_detector_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalPassivePixelInputManifest:
    """Authenticate primitive pixel inputs and independent upstream policy.

    :see: :func:`~.test_detector.\
test_pixel_manifest_fans_out_every_independent_l8_policy_without_replay`
    """
    _assert_concrete(
        (
            maximum_state_error,
            node_to_pixel,
            quadrature_weight_intervals,
            quadrature_weight_points,
            aperture_efficiency_intervals,
            aperture_efficiency_points,
        )
    )
    mapping = _require_public_array(
        node_to_pixel, dtype=np.dtype(np.int64), name="node_to_pixel"
    )
    quadrature_points = _require_public_array(
        quadrature_weight_points,
        dtype=np.dtype(np.float64),
        name="quadrature_weight_points",
    )
    aperture_points = _require_public_array(
        aperture_efficiency_points,
        dtype=np.dtype(np.float64),
        name="aperture_efficiency_points",
    )
    state_error = _require_public_array(
        maximum_state_error,
        dtype=np.dtype(np.float64),
        name="maximum_state_error",
    )
    manifest = _make_local_passive_pixel_input_manifest_candidate(
        maximum_state_error=jnp.asarray(state_error),
        node_to_pixel=jnp.asarray(mapping),
        quadrature_weight_points=jnp.asarray(quadrature_points),
        aperture_efficiency_points=jnp.asarray(aperture_points),
        route=(
            route
            if isinstance(route, GalerkinLocalPositivePortRoute)
            else GalerkinLocalPositivePortRoute(route)
        ),
        terminal_disposition=(
            disposition
            if isinstance(disposition, GalerkinLocalVacuumTerminalDisposition)
            else GalerkinLocalVacuumTerminalDisposition(disposition)
        ),
        maximum_stability_direct_pairs=maximum_stability_direct_pairs,
        maximum_gram_pairs=maximum_gram_pairs,
        maximum_terminal_direct_pairs=maximum_terminal_direct_pairs,
        maximum_branch_direct_terms=maximum_branch_direct_terms,
        maximum_cut_direct_pairs=maximum_cut_direct_pairs,
        maximum_root_work=maximum_root_work,
        precision_bits=precision_bits,
        maximum_terms=maximum_terms,
        maximum_entire_work=maximum_entire_work,
        maximum_range_reductions=maximum_range_reductions,
        maximum_interval_work=maximum_interval_work,
        maximum_l8_rational_bits=maximum_l8_rational_bits,
        coordinate_convention=(
            coordinate_convention
            if isinstance(
                coordinate_convention,
                GalerkinLocalDetectorCoordinateConvention,
            )
            else GalerkinLocalDetectorCoordinateConvention(
                coordinate_convention
            )
        ),
        quadrature_weight_intervals=quadrature_weight_intervals,
        aperture_efficiency_intervals=aperture_efficiency_intervals,
        pixel_count=pixel_count,
        maximum_detector_work=maximum_detector_work,
        maximum_detector_rational_bits=maximum_detector_rational_bits,
        manifest_digest="0" * 64,
    )
    return _make_local_passive_pixel_input_manifest(manifest)


def _empty_detector_work(
    maximum_work: int, maximum_rational_bits: int
) -> GalerkinLocalDetectorWorkTranscript:
    """PRIVATE: Build a valid zero-work candidate transcript.

    Parameters
    ----------
    maximum_work : int
        Required canonical input.
    maximum_rational_bits : int
        Required canonical input.

    Returns
    -------
    result : GalerkinLocalDetectorWorkTranscript
        Canonical derived result.
    """
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


def certify_local_passive_pixel_forms(
    positive_port: GalerkinLocalPositivePortCertificate,
    *,
    input_manifest: GalerkinLocalPassivePixelInputManifest,
) -> GalerkinLocalPassivePixelForms:
    """Replay one positive port and certify its primitive passive pixels.

    :see: :func:`~.test_detector.\
test_parent_pixel_q_lvt56_and_trace_chain_match_independent_oracle`
    """
    _assert_concrete((positive_port, input_manifest))
    manifest = _validate_local_passive_pixel_input_manifest(input_manifest)
    port = prepare_local_positive_port_certificate(
        positive_port,
        route=manifest.route,
        disposition=manifest.terminal_disposition,
        maximum_state_error=manifest.maximum_state_error,
        maximum_stability_direct_pairs=(
            manifest.maximum_stability_direct_pairs
        ),
        maximum_gram_pairs=manifest.maximum_gram_pairs,
        maximum_terminal_direct_pairs=(manifest.maximum_terminal_direct_pairs),
        maximum_branch_direct_terms=manifest.maximum_branch_direct_terms,
        maximum_cut_direct_pairs=manifest.maximum_cut_direct_pairs,
        maximum_root_work=manifest.maximum_root_work,
        precision_bits=manifest.precision_bits,
        maximum_terms=manifest.maximum_terms,
        maximum_entire_work=manifest.maximum_entire_work,
        maximum_range_reductions=manifest.maximum_range_reductions,
        maximum_interval_work=manifest.maximum_interval_work,
        maximum_rational_bits=manifest.maximum_l8_rational_bits,
    )
    zero = _interval_carrier((_ZERO, _ZERO))
    candidate = _make_local_passive_pixel_forms_candidate(
        positive_port=port,
        node_to_pixel=manifest.node_to_pixel,
        production_evidence_available=jnp.asarray(False),
        positive_forms_eligible=jnp.asarray(False),
        passive_forms_eligible=jnp.asarray(False),
        failure_mask=jnp.asarray(
            int(np.asarray(port.failure_mask)), dtype=jnp.int64
        ),
        quadrature_weights=manifest.quadrature_weight_points,
        aperture_efficiencies=manifest.aperture_efficiency_points,
        production_traces=(),
        current_weight_intervals=(),
        amplitude_scale_interval=zero,
        coordinate_jacobian_intervals=(),
        quadrature_weight_intervals=manifest.quadrature_weight_intervals,
        aperture_efficiency_intervals=(manifest.aperture_efficiency_intervals),
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
        coordinate_convention=manifest.coordinate_convention,
        pixel_count=manifest.pixel_count,
        work_transcript=_empty_detector_work(
            manifest.maximum_detector_work,
            manifest.maximum_detector_rational_bits,
        ),
        coordinate_factor_scope="candidate",
        pixel_form_scope="candidate",
        lvt56_error_scope="candidate",
        passivity_margin_scope="candidate",
        no_experimental_validity_scope="candidate",
        parent_port_certificate_digest=port.certificate_digest,
        input_manifest_digest=manifest.manifest_digest,
        pixel_model_identity_digest="0" * 64,
        pixel_model_evidence_digest="0" * 64,
        certificate_digest="0" * 64,
    )
    evidence = _expected_local_passive_pixel_evidence(candidate)
    return _make_local_passive_pixel_forms(replace(candidate, **evidence))


def prepare_local_passive_pixel_forms(
    certificate: GalerkinLocalPassivePixelForms,
    *,
    input_manifest: GalerkinLocalPassivePixelInputManifest,
) -> GalerkinLocalPassivePixelForms:
    """Replay independently supplied pixel inputs and exact-compare storage.

    :see: :func:`~.test_detector.\
test_public_l9_chain_prepares_and_rejects_policy_and_coherent_forgeries`
    """
    _assert_concrete((certificate, input_manifest))
    if not isinstance(certificate, GalerkinLocalPassivePixelForms):
        raise TypeError("certificate must be GalerkinLocalPassivePixelForms")
    canonical = certify_local_passive_pixel_forms(
        certificate.positive_port, input_manifest=input_manifest
    )
    if not bool(eqx.tree_equal(canonical, certificate, typematch=True)):
        raise ValueError(
            "local passive-pixel certificate failed complete replay"
        )
    return canonical


def create_local_censored_poisson_detector_input_manifest(  # noqa: PLR0913
    *,
    pixel_inputs: tuple[GalerkinLocalPassivePixelInputManifest, ...],
    ensemble_weight_numerators: tuple[int, ...],
    ensemble_weight_denominators: tuple[int, ...],
    incident_electron_count_interval: GalerkinLocalDetectorRationalInterval,
    incident_electron_count_point: object,
    response_matrix: object,
    pre_gain_background: object,
    deterministic_gain: object,
    electronic_offset: object,
    count_ceilings: object,
    fit_mask: object,
    calibration_provenance: str,
    maximum_detector_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_detector_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    maximum_count_ceiling: int = _DEFAULT_MAXIMUM_COUNT_CEILING,
    maximum_poisson_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_poisson_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    exp_precision_bits: int = _DEFAULT_L8_PRECISION_BITS,
    maximum_exp_terms: int = _DEFAULT_L8_TERMS,
    maximum_exp_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_exp_range_reductions: int = _DEFAULT_L8_RANGE_REDUCTIONS,
) -> GalerkinLocalCensoredPoissonDetectorInputManifest:
    """Authenticate primitive detector inputs and every pixel replay input.

    :see: :func:`~.test_detector.\
test_parent_detector_pipeline_is_ordered_and_charged_exactly_once`
    """
    _assert_concrete(
        (
            pixel_inputs,
            incident_electron_count_interval,
            incident_electron_count_point,
            response_matrix,
            pre_gain_background,
            deterministic_gain,
            electronic_offset,
            count_ceilings,
            fit_mask,
        )
    )
    response = _require_public_array(
        response_matrix, dtype=np.dtype(np.float64), name="response_matrix"
    )
    background = _require_public_array(
        pre_gain_background,
        dtype=np.dtype(np.float64),
        name="pre_gain_background",
    )
    gain = _require_public_array(
        deterministic_gain,
        dtype=np.dtype(np.float64),
        name="deterministic_gain",
    )
    offset = _require_public_array(
        electronic_offset,
        dtype=np.dtype(np.float64),
        name="electronic_offset",
    )
    ceilings = _require_public_array(
        count_ceilings, dtype=np.dtype(np.int64), name="count_ceilings"
    )
    fitted = _require_public_array(
        fit_mask, dtype=np.dtype(np.bool_), name="fit_mask"
    )
    dose_point = _require_public_array(
        incident_electron_count_point,
        dtype=np.dtype(np.float64),
        name="incident_electron_count_point",
    )
    manifest = _make_local_censored_poisson_detector_input_manifest_candidate(
        pixel_inputs=pixel_inputs,
        response_matrix=jnp.asarray(response),
        pre_gain_background=jnp.asarray(background),
        deterministic_gain=jnp.asarray(gain),
        electronic_offset=jnp.asarray(offset),
        count_ceilings=jnp.asarray(ceilings),
        fit_mask=jnp.asarray(fitted),
        incident_electron_count_point=jnp.asarray(dose_point),
        ensemble_weight_numerators=ensemble_weight_numerators,
        ensemble_weight_denominators=ensemble_weight_denominators,
        incident_electron_count_interval=incident_electron_count_interval,
        calibration_provenance=calibration_provenance,
        maximum_detector_work=maximum_detector_work,
        maximum_detector_rational_bits=maximum_detector_rational_bits,
        maximum_count_ceiling=maximum_count_ceiling,
        maximum_poisson_work=maximum_poisson_work,
        maximum_poisson_rational_bits=maximum_poisson_rational_bits,
        exp_precision_bits=exp_precision_bits,
        maximum_exp_terms=maximum_exp_terms,
        maximum_exp_work=maximum_exp_work,
        maximum_exp_range_reductions=maximum_exp_range_reductions,
        manifest_digest="0" * 64,
    )
    return _make_local_censored_poisson_detector_input_manifest(manifest)


def certify_local_censored_poisson_detector(
    pixel_forms: tuple[GalerkinLocalPassivePixelForms, ...],
    *,
    input_manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
) -> GalerkinLocalCensoredPoissonDetector:
    """Replay all pixels and certify one fixed censored-Poisson detector.

    :see: :func:`~.test_detector.\
test_parent_detector_pipeline_is_ordered_and_charged_exactly_once`
    """
    _assert_concrete((pixel_forms, input_manifest))
    manifest = _validate_local_censored_poisson_detector_input_manifest(
        input_manifest
    )
    if not isinstance(pixel_forms, tuple) or len(pixel_forms) != len(
        manifest.pixel_inputs
    ):
        raise ValueError("detector pixel forms and input manifests disagree")
    pixels = tuple(
        prepare_local_passive_pixel_forms(pixel, input_manifest=pixel_input)
        for pixel, pixel_input in zip(
            pixel_forms, manifest.pixel_inputs, strict=True
        )
    )
    binding = _expected_mode_state_binding(pixels)
    candidate = _make_local_censored_poisson_detector_candidate(
        pixel_forms=pixels,
        response_matrix=manifest.response_matrix,
        pre_gain_background=manifest.pre_gain_background,
        deterministic_gain=manifest.deterministic_gain,
        electronic_offset=manifest.electronic_offset,
        count_ceilings=manifest.count_ceilings,
        fit_mask=manifest.fit_mask,
        incident_electron_count_point=(manifest.incident_electron_count_point),
        production_evidence_available=jnp.asarray(False),
        exact_state_censored_mean_evidence_available=jnp.asarray(False),
        production_censored_mean_evidence_available=jnp.asarray(False),
        detector_eligible=jnp.asarray(False),
        likelihood_law_eligible=jnp.asarray(False),
        failure_mask=jnp.asarray(0, dtype=jnp.int64),
        production_traces=(),
        mode_target_digests=binding[0],
        mode_source_digests=binding[1],
        mode_state_identity_digests=binding[2],
        mode_state_radius_intervals=binding[3],
        mode_state_radius_provenance_digests=binding[4],
        mode_port_certificate_digests=binding[5],
        mode_pixel_evidence_digests=binding[6],
        mode_state_binding_digest=binding[7],
        ensemble_weight_numerators=manifest.ensemble_weight_numerators,
        ensemble_weight_denominators=manifest.ensemble_weight_denominators,
        incident_reduced_flux_intervals=(),
        mode_exact_state_pixel_flux_intervals=(),
        mode_production_quadratic_intervals=(),
        mode_pixel_form_norm_upper_intervals=(),
        mode_production_to_exact_x_amplitude_error_intervals=(),
        mode_state_radius_amplitude_error_intervals=(),
        mode_exact_state_amplitude_error_intervals=(),
        mode_production_amplitude_norm_intervals=(),
        mode_production_realization_error_upper_intervals=(),
        mode_state_radius_incremental_error_upper_intervals=(),
        mode_combined_exact_state_error_upper_intervals=(),
        mode_outward_passivity_margin_intervals=(),
        mode_pixel_fraction_intervals=(),
        ideal_arrival_mean_intervals=(),
        production_pre_gain_mean_point_intervals=(),
        exact_state_pre_gain_mean_intervals=(),
        censored_mean_intervals=(),
        expected_digitized_mean_intervals=(),
        incident_electron_count=(manifest.incident_electron_count_interval),
        likelihood_stage=(
            GalerkinLocalDetectorLikelihoodStage.PRE_GAIN_CENSORED_COUNTS
        ),
        work_transcript=_empty_detector_work(
            manifest.maximum_detector_work,
            manifest.maximum_detector_rational_bits,
        ),
        censored_mean_transcripts=(),
        censored_mean_failures=(),
        production_censored_mean_transcripts=(),
        production_censored_mean_failures=(),
        maximum_count_ceiling=manifest.maximum_count_ceiling,
        maximum_poisson_work=manifest.maximum_poisson_work,
        maximum_poisson_rational_bits=(manifest.maximum_poisson_rational_bits),
        exp_precision_bits=manifest.exp_precision_bits,
        maximum_exp_terms=manifest.maximum_exp_terms,
        maximum_exp_work=manifest.maximum_exp_work,
        maximum_exp_range_reductions=manifest.maximum_exp_range_reductions,
        flux_normalization_scope="candidate",
        ensemble_scope="candidate",
        response_scope="candidate",
        calibration_provenance=manifest.calibration_provenance,
        no_experimental_validity_scope="candidate",
        target_digest=binding[0][0],
        input_manifest_digest=manifest.manifest_digest,
        detector_model_identity_digest="0" * 64,
        detector_model_evidence_digest="0" * 64,
        certificate_digest="0" * 64,
    )
    evidence = _expected_local_censored_poisson_detector(candidate)
    return _make_local_censored_poisson_detector(
        replace(candidate, **evidence)
    )


def prepare_local_censored_poisson_detector(
    certificate: GalerkinLocalCensoredPoissonDetector,
    *,
    input_manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
) -> GalerkinLocalCensoredPoissonDetector:
    """Replay independent detector inputs and exact-compare every field.

    :see: :func:`~.test_detector.\
test_public_l9_chain_prepares_and_rejects_policy_and_coherent_forgeries`
    """
    _assert_concrete((certificate, input_manifest))
    if not isinstance(certificate, GalerkinLocalCensoredPoissonDetector):
        raise TypeError(
            "certificate must be GalerkinLocalCensoredPoissonDetector"
        )
    canonical = certify_local_censored_poisson_detector(
        certificate.pixel_forms, input_manifest=input_manifest
    )
    if not bool(eqx.tree_equal(canonical, certificate, typematch=True)):
        raise ValueError("local detector certificate failed complete replay")
    return canonical


def enclose_local_censored_poisson_likelihood(  # noqa: PLR0913
    detector_certificate: GalerkinLocalCensoredPoissonDetector,
    *,
    detector_input_manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
    observed_counts: object,
    maximum_detector_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_detector_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    log_precision_bits: int = _DEFAULT_L8_PRECISION_BITS,
    maximum_log_terms: int = _DEFAULT_L8_TERMS,
    maximum_log_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_log_range_reductions: int = _DEFAULT_L8_RANGE_REDUCTIONS,
) -> GalerkinLocalCensoredPoissonLikelihood:
    """Enclose full-channel probabilities and fit-only pre-gain NLL.

    :see: :func:`~.test_detector.\
test_public_l9_chain_prepares_and_rejects_policy_and_coherent_forgeries`
    """
    _assert_concrete(
        (detector_certificate, detector_input_manifest, observed_counts)
    )
    detector = prepare_local_censored_poisson_detector(
        detector_certificate, input_manifest=detector_input_manifest
    )
    observed_array = _require_public_array(
        observed_counts, dtype=np.dtype(np.int64), name="observed_counts"
    )
    observed = jnp.asarray(observed_array)
    candidate = _make_local_censored_poisson_likelihood_candidate(
        detector=detector,
        observed_counts=observed,
        likelihood_evidence_available=jnp.asarray(False),
        likelihood_law_eligible=jnp.asarray(False),
        nll_eligible=jnp.asarray(False),
        failure_mask=jnp.asarray(0, dtype=jnp.int64),
        production_traces=(),
        admitted_pre_gain_mean_hull_intervals=(),
        production_probability_point_intervals=(),
        admitted_hull_probability_intervals=(),
        fitted_probability_positive_floor_intervals=(),
        production_nll_point_intervals=(),
        admitted_hull_nll_intervals=(),
        total_nll_interval=None,
        production_probability_transcripts=(),
        production_probability_failures=(),
        admitted_hull_probability_transcripts=(),
        admitted_hull_probability_failures=(),
        production_nll_transcripts=(),
        production_nll_failures=(),
        admitted_hull_nll_transcripts=(),
        admitted_hull_nll_failures=(),
        work_transcript=_empty_detector_work(
            maximum_detector_work, maximum_detector_rational_bits
        ),
        log_precision_bits=log_precision_bits,
        maximum_log_terms=maximum_log_terms,
        maximum_log_work=maximum_log_work,
        maximum_log_range_reductions=maximum_log_range_reductions,
        likelihood_scope="candidate",
        nll_scope="candidate",
        no_derivative_scope="candidate",
        parent_detector_certificate_digest=detector.certificate_digest,
        likelihood_identity_digest="0" * 64,
        likelihood_evidence_digest="0" * 64,
        certificate_digest="0" * 64,
    )
    evidence = _expected_local_censored_poisson_likelihood(candidate)
    return _make_local_censored_poisson_likelihood(
        replace(candidate, **evidence)
    )


def prepare_local_censored_poisson_likelihood(  # noqa: PLR0913
    certificate: GalerkinLocalCensoredPoissonLikelihood,
    *,
    detector_input_manifest: GalerkinLocalCensoredPoissonDetectorInputManifest,
    observed_counts: object,
    maximum_detector_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_detector_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
    log_precision_bits: int = _DEFAULT_L8_PRECISION_BITS,
    maximum_log_terms: int = _DEFAULT_L8_TERMS,
    maximum_log_work: int = _DEFAULT_MAXIMUM_WORK,
    maximum_log_range_reductions: int = _DEFAULT_L8_RANGE_REDUCTIONS,
) -> GalerkinLocalCensoredPoissonLikelihood:
    """Replay a likelihood from independent inputs and exact-compare it.

    :see: :func:`~.test_detector.\
test_public_l9_chain_prepares_and_rejects_policy_and_coherent_forgeries`
    """
    _assert_concrete((certificate, detector_input_manifest, observed_counts))
    if not isinstance(certificate, GalerkinLocalCensoredPoissonLikelihood):
        raise TypeError(
            "certificate must be GalerkinLocalCensoredPoissonLikelihood"
        )
    canonical = enclose_local_censored_poisson_likelihood(
        certificate.detector,
        detector_input_manifest=detector_input_manifest,
        observed_counts=observed_counts,
        maximum_detector_work=maximum_detector_work,
        maximum_detector_rational_bits=maximum_detector_rational_bits,
        log_precision_bits=log_precision_bits,
        maximum_log_terms=maximum_log_terms,
        maximum_log_work=maximum_log_work,
        maximum_log_range_reductions=maximum_log_range_reductions,
    )
    if not bool(eqx.tree_equal(canonical, certificate, typematch=True)):
        raise ValueError("local detector likelihood failed complete replay")
    return canonical


__all__: list[str] = [
    "certify_local_censored_poisson_detector",
    "certify_local_passive_pixel_forms",
    "certify_local_positive_port",
    "create_local_censored_poisson_detector_input_manifest",
    "create_local_passive_pixel_input_manifest",
    "enclose_local_censored_poisson_likelihood",
    "prepare_local_censored_poisson_detector",
    "prepare_local_censored_poisson_likelihood",
    "prepare_local_passive_pixel_forms",
    "prepare_local_positive_port_certificate",
]
