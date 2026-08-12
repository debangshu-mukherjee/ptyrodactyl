r"""Compose authenticated local vacuum-terminal continuation evidence.

Extended Summary
----------------
This host-only leaf consumes one fully replayed local projection certificate
and internally rebuilds both slab-endpoint L7 current diagnostics.  It derives
physical transverse vacuum roots from the target's exact carrier ledger,
encloses homogeneous Cauchy propagators and an independent finite forced-
integral route, intersects both routes, and checks the nonsymmetrized
projection-work/current cut balance.  Concrete rounded role-zero amplitudes
and their exact-state error DAG form the LVT.56 detector bridge, without
asserting detector eligibility.

Routine Listings
----------------
:func:`certify_local_vacuum_terminal`
    Compose one scoped LVT.39--LVT.56 vacuum-terminal certificate.
:func:`prepare_local_vacuum_terminal_certificate`
    Replay every L8 parent, policy, helper route, field, and digest.

Notes
-----
Plane-defined continuation, selected native-sector continuation, and full
native-slab continuation remain distinct dispositions.  Endpoint and forced
integrals use the submitted stored state.  Projection ``E_f(x, B)`` is kept
as separate plane/native mismatch evidence and is never charged into the
defining-plane LVT.56 amplitude error.
"""

from __future__ import annotations

import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from ptyrodactyl._tools import (
    ComplexRectangle,
    EntireEnclosureError,
    EntireEnclosureFailure,
    EntireWorkTranscript,
    all_normal_arithmetic_supported,
    enclose_complex_exp,
    enclose_complex_exprel,
    enclose_complex_phi2,
    fraction_lower_float,
    fraction_upper_float,
    has_subnormal_components,
    host_binary64_supported,
    mathematical_pi_interval,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types import (
    GalerkinLocalCellTargetManifest,
    GalerkinLocalCoordinateCauchyCurrent,
    GalerkinLocalProjectionDefectCertificate,
    GalerkinLocalProjectionDefectFailure,
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalScope,
    GalerkinLocalVacuumBranchEvidence,
    GalerkinLocalVacuumCutBalance,
    GalerkinLocalVacuumHalfSpaceDisposition,
    GalerkinLocalVacuumPropagationError,
    GalerkinLocalVacuumPropagationFailure,
    GalerkinLocalVacuumPropagator,
    GalerkinLocalVacuumRootCertificate,
    GalerkinLocalVacuumRootClass,
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
    GalerkinLocalVacuumTerminalEntireEvidence,
    GalerkinLocalVacuumTerminalFailure,
    GalerkinLocalVacuumZeroWitness,
    GalerkinLocalVacuumZeroWitnessRoute,
    GalerkinTerminalSide,
    _make_local_vacuum_branch_evidence,
    _make_local_vacuum_cut_balance,
    _make_local_vacuum_terminal_certificate,
    _make_local_vacuum_terminal_entire_evidence,
    _make_prepared_local_current_operator,
    _PlaneMismatchBounds,
    _ProductionEvidence,
)

from .local_projection import prepare_local_projection_defect_certificate
from .local_terminal import (
    _certify_prepared_operator,
    _enclose_current_prepared,
    _fraction_rectangle,
)
from .local_vacuum_propagation import (
    classify_local_vacuum_root,
    enclose_local_vacuum_propagator,
    make_local_vacuum_zero_witness,
    prepare_local_vacuum_propagator,
    prepare_local_vacuum_root_certificate,
)

type _RealInterval = tuple[Fraction, Fraction]
type _RectanglePair = tuple[
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalComplexRectangles,
]
type _OptionalRectangleLists = tuple[
    list[ComplexRectangle | None], list[ComplexRectangle | None]
]
type _PropagationFailures = tuple[
    EntireEnclosureFailure | GalerkinLocalVacuumPropagationFailure | None,
    ...,
]
type _EntirePolicies = tuple[int, int, int, int, int]
type _OptionalRootIntervals = tuple[_RealInterval | None, ...]
type _OptionalPropagatorEntries = tuple[
    tuple[_RealInterval, _RealInterval, _RealInterval, _RealInterval] | None,
    ...,
]

_DEFAULT_MAXIMUM_BRANCH_DIRECT_TERMS: int = 2_000_000
_DEFAULT_MAXIMUM_CUT_DIRECT_PAIRS: int = 2_000_000
_DEFAULT_MAXIMUM_ENTIRE_WORK: int = 1_000_000
_DEFAULT_MAXIMUM_GRAM_PAIRS: int = 2_000_000
_DEFAULT_MAXIMUM_INTERVAL_WORK: int = 1_000_000
_DEFAULT_MAXIMUM_RANGE_REDUCTIONS: int = 4096
_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_DEFAULT_MAXIMUM_RATIONAL_BITS: int = 262_144
_DEFAULT_MAXIMUM_ROOT_WORK: int = 64
_DEFAULT_MAXIMUM_STABILITY_DIRECT_PAIRS: int = 2_000_000
_DEFAULT_MAXIMUM_TERMINAL_DIRECT_PAIRS: int = 2_000_000
_DEFAULT_MAXIMUM_TERMS: int = 4096
_DEFAULT_PRECISION_BITS: int = 160
_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_ZERO: Fraction = Fraction(0)
_ONE: Fraction = Fraction(1)
_HALF: Fraction = Fraction(1, 2)
_HULL_ALGORITHM: str = "outward_binary64_normal_hull_v1"
_PROPAGATOR_ENTRY_COUNT: int = 4

_DIRECT_WORK_FORMULA: str = (
    "6*n_scope + 32*f: four physical-Cauchy and two forced-integral "
    "state contributions per selected coefficient, plus thirty-two "
    "per-fiber transforms, intersections, bounds, and reductions"
)
_PHYSICAL_ROOT_FORMULA: str = (
    "q_h=[k0]^2-sum_a[k_i,a+2*pi*g_h,a/L_a]^2 from the authenticated "
    "exact wavenumber/carrier ledger; grazing requires direct singleton zero "
    "or one same-fiber singleton-zero D and oriented-normal support row; raw "
    "root evidence is preserved and one outward normal-binary64 dyadic copy "
    "is used only by downstream branch arithmetic"
)
_ROOT_REALIZATION_FORMULA: str = (
    "nongrazing r_hat=float((r_lower+r_upper)/2) nearest binary64 from the "
    "raw replayed root and e_r=up(max(|r_hat-r_lower|,|r_hat-r_upper|)); "
    "grazing uses 0 exactly"
)
_PHYSICAL_CAUCHY_FORMULA: str = (
    "(t,nu_s)=exp(i*k_i,n*xi)*(T_xi*x,N_xi^(s)*x), using exact L7 "
    "coefficient rectangles and once-hulled exact-ledger carrier-phase exp "
    "enclosures"
)
_ENDPOINT_MISMATCH_FORMULA: str = (
    "m=y_outer-E_h(ell)y_inner in outward coordinate r=s*(xi-xi_inner), "
    "using once-hulled prepared propagator entries"
)
_FORCED_MISMATCH_FORMULA: str = (
    "J(a,lambda)=ell*exp(a*ell)*phi1((i*lambda-a)*ell); propagating and "
    "evanescent sine/cosine or sinh/cosh kernels use J(+a) and J(-a), "
    "while grazing uses ell^2*phi2(i*lambda*ell) and "
    "ell*phi1(i*lambda*ell), all multiplied by -d*x/sqrt(L) and the "
    "physical inner phase; every helper output and consumed positive-root "
    "interval is outward normal-binary64 dyadically hulled exactly once"
)
_PLANE_MISMATCH_BOUND_FORMULA: str = (
    "LVT.48/LVT.52--54 kernel factors multiply submitted measured E_f and "
    "projection ||D0||B separately after one root/helper dyadic hull; their "
    "outward sum is total E_f once"
)
_AMPLITUDE_ERROR_FORMULA: str = (
    "defining E_a=direct distance from frozen role point to its exact-x "
    "once-hulled branch rectangle + ||A_exact,role||*B once; hull widening "
    "is already in that rectangle and is not added again; no projection D0 "
    "or E_f"
)
_AMPLITUDE_NORM_FORMULA: str = (
    "component norms are up(|a_hat|) and up(|a_hat|+E_a); role-zero l2 "
    "norms and error are exact Fraction square-root/outward reductions"
)
_CURRENT_DIFFERENCE_FORMULA: str = (
    "outer side-oriented current minus inner side-oriented current equals "
    "the positive-coordinate upper-cut minus lower-cut current on both sides"
)
_DEFECT_WORK_FORMULA: str = (
    "-Im sum_h,p,q conj(x_h,p)*G_h[p,q]*d_h,q*x_h,q with literal "
    "G*diag(d), never Hermitian symmetrized"
)


class _LocalArithmeticRangeError(ValueError):
    """Signal an expected finite/subnormal production-range failure."""


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced values at the exact host boundary.

    Parameters
    ----------
    value : object
        Submitted carrier or policy tree.

    Raises
    ------
    ValueError
        If any leaf is a JAX tracer.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError(
            "local vacuum-terminal certification requires concrete host values"
        )


def _checked_positive_int(value: object, name: str) -> int:
    """PRIVATE: Validate one positive signed-int64 resource policy.

    Parameters
    ----------
    value : object
        Candidate policy.
    name : str
        Public parameter name.

    Returns
    -------
    value : int
        Validated positive Python integer.

    Raises
    ------
    TypeError
        If the policy is not exactly a Python integer.
    ValueError
        If the policy is outside the positive signed-int64 range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value <= 0 or value > _MAXIMUM_SIGNED_INT64:
        raise ValueError(f"{name} must be a positive signed-int64 integer")
    return value


def _checked_disposition(
    value: GalerkinLocalVacuumTerminalDisposition | str,
) -> GalerkinLocalVacuumTerminalDisposition:
    """PRIVATE: Parse one explicit continuation disposition.

    Parameters
    ----------
    value : GalerkinLocalVacuumTerminalDisposition | str
        Candidate plane-defined or exact-native disposition.

    Returns
    -------
    disposition : GalerkinLocalVacuumTerminalDisposition
        Canonical terminal disposition.

    Raises
    ------
    ValueError
        If the string does not name an admitted disposition.
    """
    disposition = GalerkinLocalVacuumTerminalDisposition(value)
    return disposition  # noqa: RET504


def _fraction_payload(value: Fraction) -> dict[str, str]:
    """PRIVATE: Encode one arbitrary-size exact rational for hashing.

    Parameters
    ----------
    value : Fraction
        Exact reduced rational.

    Returns
    -------
    payload : dict[str, str]
        Signed-hex numerator and positive-hex denominator.
    """
    sign = "-" if value.numerator < 0 else "+"
    payload = {
        "numerator_hex": sign + format(abs(value.numerator), "x"),
        "denominator_hex": format(value.denominator, "x"),
    }
    return payload  # noqa: RET504


def _interval_payload(value: _RealInterval) -> tuple[dict[str, str], ...]:
    """PRIVATE: Encode one exact rational interval for hashing.

    Parameters
    ----------
    value : _RealInterval
        Ordered exact rational interval.

    Returns
    -------
    payload : tuple[dict[str, str], ...]
        Arbitrary-size-safe endpoint payloads.
    """
    return tuple(_fraction_payload(endpoint) for endpoint in value)


def _normal_or_zero(value: object) -> bool:
    """PRIVATE: Check finite binary64 normal components or exact zeros.

    Parameters
    ----------
    value : object
        Candidate real or complex scalar or array.

    Returns
    -------
    valid : bool
        Whether every component is finite and non-subnormal.
    """
    array = jnp.asarray(value)
    valid = bool(jnp.all(jnp.isfinite(array))) and not bool(
        has_subnormal_components(array)
    )
    return valid  # noqa: RET504


class _DirectRationalLedger:
    """Bound every source-local reduced Fraction and count exact operations."""

    def __init__(self, maximum_rational_bits: int) -> None:
        self.maximum_rational_bits = maximum_rational_bits
        self.exact_work_count = 0
        self.peak_bits = 0

    def retain(self, value: Fraction) -> Fraction:
        """Retain one reduced rational or raise the typed size failure."""
        bits = max(
            abs(value.numerator).bit_length(), value.denominator.bit_length()
        )
        self.peak_bits = max(self.peak_bits, bits)
        if bits > self.maximum_rational_bits:
            raise EntireEnclosureError(
                EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
                self.exact_work_count,
                "direct vacuum-terminal rational exceeds its bit policy",
            )
        return value

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        """Issue one checked rational addition."""
        left = self.retain(left)
        right = self.retain(right)
        self.exact_work_count += 1
        return self.retain(left + right)

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        """Issue one checked rational subtraction."""
        left = self.retain(left)
        right = self.retain(right)
        self.exact_work_count += 1
        return self.retain(left - right)

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        """Issue one checked rational multiplication."""
        left = self.retain(left)
        right = self.retain(right)
        self.exact_work_count += 1
        return self.retain(left * right)

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        """Issue one checked rational division."""
        numerator = self.retain(numerator)
        denominator = self.retain(denominator)
        if denominator == 0:
            raise ZeroDivisionError("direct rational denominator is zero")
        self.exact_work_count += 1
        return self.retain(numerator / denominator)

    def root_upper(self, value: Fraction) -> Fraction:
        """Issue and retain one verified rational square-root upper bound."""
        value = self.retain(value)
        self.exact_work_count += 1
        return self.retain(sqrt_fraction_upper(value))

    def interval_multiply(
        self, left: _RealInterval, right: _RealInterval
    ) -> _RealInterval:
        """Multiply two real intervals through checked corner products."""
        products = (
            self.multiply(left[0], right[0]),
            self.multiply(left[0], right[1]),
            self.multiply(left[1], right[0]),
            self.multiply(left[1], right[1]),
        )
        return min(products), max(products)

    def rectangle_multiply(
        self, left: ComplexRectangle, right: ComplexRectangle
    ) -> ComplexRectangle:
        """Multiply complex rectangles with checked real interval work."""
        ac = self.interval_multiply(left[:2], right[:2])
        bd = self.interval_multiply(left[2:], right[2:])
        ad = self.interval_multiply(left[:2], right[2:])
        bc = self.interval_multiply(left[2:], right[:2])
        real = _interval_subtract(ac, bd, self)
        imag = _interval_add(ad, bc, self)
        return real[0], real[1], imag[0], imag[1]

    def rectangle_scale(
        self, value: ComplexRectangle, scalar: Fraction
    ) -> ComplexRectangle:
        """Scale a complex rectangle through checked real interval work."""
        real = self.interval_multiply(value[:2], (scalar, scalar))
        imag = self.interval_multiply(value[2:], (scalar, scalar))
        return real[0], real[1], imag[0], imag[1]


class _OutwardDyadicHullLedger:
    """Record one fixed outward normal-binary64 dyadic-hull transcript."""

    def __init__(self, maximum_rational_bits: int) -> None:
        self.maximum_rational_bits = maximum_rational_bits
        self.attempted_endpoint_count = 0
        self.completed_endpoint_count = 0
        self.input_peak_bits = 0
        self.output_peak_bits = 0
        self.normal_floor_count = 0
        self.range_failure = False

    @staticmethod
    def _bits(value: Fraction) -> int:
        """PRIVATE: Return the largest reduced endpoint bit length.

        Parameters
        ----------
        value : Fraction
            Exact rational endpoint.

        Returns
        -------
        bits : int
            Largest reduced numerator or denominator bit length.
        """
        return max(
            abs(value.numerator).bit_length(), value.denominator.bit_length()
        )

    @staticmethod
    def _directed_candidate(value: Fraction, *, lower: bool) -> float | None:
        """PRIVATE: Reconstruct the pre-normal-floor directed candidate.

        Parameters
        ----------
        value : Fraction
            Exact rational endpoint.
        lower : bool
            Whether to direct the candidate toward negative infinity.

        Returns
        -------
        candidate : float | None
            Directed binary64 candidate, or none after range overflow.
        """
        try:
            candidate = float(value)
        except OverflowError:
            return None
        if math.isfinite(candidate):
            exact_candidate = Fraction.from_float(candidate)
            if lower and exact_candidate > value:
                candidate = math.nextafter(candidate, -math.inf)
            elif not lower and exact_candidate < value:
                candidate = math.nextafter(candidate, math.inf)
        return candidate

    def _endpoint(self, value: Fraction, *, lower: bool) -> Fraction | None:
        """PRIVATE: Hull one endpoint or record a typed range failure.

        Parameters
        ----------
        value : Fraction
            Exact rational endpoint.
        lower : bool
            Whether to construct the outward lower endpoint.

        Returns
        -------
        exact : Fraction | None
            Outward normal-or-zero dyadic endpoint, or none on range failure.

        Raises
        ------
        EntireEnclosureError
            If the input exceeds the rational-bit policy.
        """
        self.attempted_endpoint_count += 1
        input_bits = self._bits(value)
        self.input_peak_bits = max(self.input_peak_bits, input_bits)
        if input_bits > self.maximum_rational_bits:
            raise EntireEnclosureError(
                EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
                self.completed_endpoint_count,
                "dyadic-hull input exceeds its rational-bit policy",
            )
        candidate = self._directed_candidate(value, lower=lower)
        try:
            converted = (
                fraction_lower_float(value)
                if lower
                else fraction_upper_float(value)
            )
        except (OverflowError, ValueError):
            self.range_failure = True
            return None
        if not _normal_or_zero(np.float64(converted)):
            self.range_failure = True
            return None
        exact = Fraction.from_float(converted)
        if (lower and exact > value) or (not lower and exact < value):
            self.range_failure = True
            return None
        self.completed_endpoint_count += 1
        self.output_peak_bits = max(self.output_peak_bits, self._bits(exact))
        if value != 0 and (
            exact == 0
            or (
                candidate is not None
                and math.isfinite(candidate)
                and candidate != converted
            )
        ):
            self.normal_floor_count += 1
        return exact

    def interval(self, value: _RealInterval) -> _RealInterval | None:
        """Hull both endpoints of one ordered real interval independently."""
        lower = self._endpoint(value[0], lower=True)
        upper = self._endpoint(value[1], lower=False)
        if lower is None or upper is None:
            return None
        if lower > upper:
            self.range_failure = True
            return None
        return lower, upper

    def rectangle(self, value: ComplexRectangle) -> ComplexRectangle | None:
        """Hull all four endpoints of one complex rectangle independently."""
        real_lower = self._endpoint(value[0], lower=True)
        real_upper = self._endpoint(value[1], lower=False)
        imag_lower = self._endpoint(value[2], lower=True)
        imag_upper = self._endpoint(value[3], lower=False)
        if any(
            endpoint is None
            for endpoint in (real_lower, real_upper, imag_lower, imag_upper)
        ):
            return None
        if (
            real_lower is None
            or real_upper is None
            or imag_lower is None
            or imag_upper is None
        ):
            raise AssertionError("checked hull endpoint unexpectedly missing")
        if real_lower > real_upper or imag_lower > imag_upper:
            self.range_failure = True
            return None
        return real_lower, real_upper, imag_lower, imag_upper

    def reject_nonpositive_root_lower(self) -> None:
        """Mark a hulled strict positive root unusable for division."""
        self.range_failure = True

    def evidence_digest(self) -> str:
        """Hash the fixed algorithm and complete aggregate hull transcript."""
        return sha256(
            {
                "domain": "ptyrodactyl.local_vacuum_terminal.hull.v1",
                "algorithm": _HULL_ALGORITHM,
                "maximum_rational_bits": self.maximum_rational_bits,
                "attempted_endpoints": self.attempted_endpoint_count,
                "completed_endpoints": self.completed_endpoint_count,
                "input_peak_bits": self.input_peak_bits,
                "output_peak_bits": self.output_peak_bits,
                "normal_floor_count": self.normal_floor_count,
                "range_failure": self.range_failure,
            }
        )


def _float_sum_upper(
    left: float,
    right: float,
    ledger: _DirectRationalLedger,
) -> float:
    """PRIVATE: Form one ledger-accounted outward binary64 sum.

    Parameters
    ----------
    left : float
        First finite nonnegative binary64 upper.
    right : float
        Second finite nonnegative binary64 upper.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    upper : float
        Outward binary64 upper bound on the exact sum.

    Raises
    ------
    _LocalArithmeticRangeError
        If an input or the outward sum is not finite normal-or-zero.
    """
    if (
        left < 0.0
        or right < 0.0
        or not _normal_or_zero(np.float64(left))
        or not _normal_or_zero(np.float64(right))
    ):
        raise _LocalArithmeticRangeError(
            "outward sum inputs are outside nonnegative normal binary64"
        )
    exact = ledger.add(Fraction.from_float(left), Fraction.from_float(right))
    upper = fraction_upper_float(exact)
    if not _normal_or_zero(np.float64(upper)):
        raise _LocalArithmeticRangeError(
            "outward sum is outside normal binary64 range"
        )
    return upper


def _complex_point_norm_upper(
    value: complex,
    ledger: _DirectRationalLedger,
) -> float:
    """PRIVATE: Form one ledger-accounted frozen complex-point norm.

    Parameters
    ----------
    value : complex
        Finite stored complex128 point.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    upper : float
        Outward binary64 Euclidean norm upper.
    """
    real = ledger.retain(Fraction.from_float(float(np.real(value))))
    imag = ledger.retain(Fraction.from_float(float(np.imag(value))))
    squared = ledger.add(
        ledger.multiply(real, real), ledger.multiply(imag, imag)
    )
    return fraction_upper_float(ledger.root_upper(squared))


def _complex_vector_norm_upper(
    values: np.ndarray,
    ledger: _DirectRationalLedger,
) -> float:
    """PRIVATE: Form one ledger-accounted complex-vector l2 norm.

    Parameters
    ----------
    values : np.ndarray
        Finite stored complex128 vector.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    upper : float
        Outward binary64 l2-norm upper.
    """
    squared = _ZERO
    for value in values:
        real = ledger.retain(Fraction.from_float(float(np.real(value))))
        imag = ledger.retain(Fraction.from_float(float(np.imag(value))))
        component = ledger.add(
            ledger.multiply(real, real), ledger.multiply(imag, imag)
        )
        squared = ledger.add(squared, component)
    return fraction_upper_float(ledger.root_upper(squared))


def _real_vector_norm_upper(
    values: np.ndarray,
    ledger: _DirectRationalLedger,
) -> float:
    """PRIVATE: Form one ledger-accounted real-vector l2 norm.

    Parameters
    ----------
    values : np.ndarray
        Finite nonnegative stored binary64 component uppers.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    upper : float
        Outward binary64 l2-norm upper.
    """
    squared = _ZERO
    for value in values:
        exact = ledger.retain(Fraction.from_float(float(value)))
        squared = ledger.add(squared, ledger.multiply(exact, exact))
    return fraction_upper_float(ledger.root_upper(squared))


def _point_to_rectangle_error_upper(
    point: complex,
    rectangles: GalerkinLocalTerminalComplexRectangles,
    index: int,
    ledger: _DirectRationalLedger,
) -> float:
    """PRIVATE: Form one ledger-accounted point-to-rectangle distance.

    Parameters
    ----------
    point : complex
        Finite frozen production point.
    rectangles : GalerkinLocalTerminalComplexRectangles
        Finite exact-target branch rectangles.
    index : int
        Scoped fiber row.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    upper : float
        Outward maximum complex corner-distance upper.
    """
    point_real = ledger.retain(Fraction.from_float(float(np.real(point))))
    point_imag = ledger.retain(Fraction.from_float(float(np.imag(point))))
    real_errors = tuple(
        abs(
            ledger.subtract(
                ledger.retain(
                    Fraction.from_float(
                        float(np.asarray(rectangles[column])[index])
                    )
                ),
                point_real,
            )
        )
        for column in (0, 1)
    )
    imag_errors = tuple(
        abs(
            ledger.subtract(
                ledger.retain(
                    Fraction.from_float(
                        float(np.asarray(rectangles[column])[index])
                    )
                ),
                point_imag,
            )
        )
        for column in (2, 3)
    )
    real_error = ledger.retain(max(real_errors))
    imag_error = ledger.retain(max(imag_errors))
    squared = ledger.add(
        ledger.multiply(real_error, real_error),
        ledger.multiply(imag_error, imag_error),
    )
    return fraction_upper_float(ledger.root_upper(squared))


def _interval_add(
    left: _RealInterval,
    right: _RealInterval,
    ledger: _DirectRationalLedger,
) -> _RealInterval:
    """PRIVATE: Add two exact rational real intervals.

    Parameters
    ----------
    left : _RealInterval
        First interval.
    right : _RealInterval
        Second interval.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : _RealInterval
        Exact Minkowski sum.
    """
    return ledger.add(left[0], right[0]), ledger.add(left[1], right[1])


def _interval_subtract(
    left: _RealInterval,
    right: _RealInterval,
    ledger: _DirectRationalLedger,
) -> _RealInterval:
    """PRIVATE: Subtract two exact rational real intervals.

    Parameters
    ----------
    left : _RealInterval
        Minuend interval.
    right : _RealInterval
        Subtrahend interval.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : _RealInterval
        Exact Minkowski difference.
    """
    return ledger.subtract(left[0], right[1]), ledger.subtract(
        left[1], right[0]
    )


def _interval_square(
    value: _RealInterval, ledger: _DirectRationalLedger
) -> _RealInterval:
    """PRIVATE: Square one ordered exact rational interval.

    Parameters
    ----------
    value : _RealInterval
        Ordered exact interval.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : _RealInterval
        Exact interval hull of all squared values.
    """
    lower, upper = value
    values = (ledger.multiply(lower, lower), ledger.multiply(upper, upper))
    minimum = _ZERO if lower <= 0 <= upper else min(values)
    return minimum, max(values)


def _rectangle_add(
    left: ComplexRectangle,
    right: ComplexRectangle,
    ledger: _DirectRationalLedger,
) -> ComplexRectangle:
    """PRIVATE: Add two exact complex rectangles.

    Parameters
    ----------
    left : ComplexRectangle
        First rectangle.
    right : ComplexRectangle
        Second rectangle.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : ComplexRectangle
        Exact componentwise Minkowski sum.
    """
    return (
        ledger.add(left[0], right[0]),
        ledger.add(left[1], right[1]),
        ledger.add(left[2], right[2]),
        ledger.add(left[3], right[3]),
    )


def _rectangle_negate(value: ComplexRectangle) -> ComplexRectangle:
    """PRIVATE: Negate one exact complex rectangle.

    Parameters
    ----------
    value : ComplexRectangle
        Rectangle to negate.

    Returns
    -------
    result : ComplexRectangle
        Negated rectangle with ordered endpoints.
    """
    return -value[1], -value[0], -value[3], -value[2]


def _rectangle_subtract(
    left: ComplexRectangle,
    right: ComplexRectangle,
    ledger: _DirectRationalLedger,
) -> ComplexRectangle:
    """PRIVATE: Subtract two exact complex rectangles.

    Parameters
    ----------
    left : ComplexRectangle
        Minuend rectangle.
    right : ComplexRectangle
        Subtrahend rectangle.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : ComplexRectangle
        Exact componentwise difference hull.
    """
    return _rectangle_add(left, _rectangle_negate(right), ledger)


def _rectangle_real_interval(value: _RealInterval) -> ComplexRectangle:
    """PRIVATE: Embed one real interval as a complex rectangle.

    Parameters
    ----------
    value : _RealInterval
        Exact real interval.

    Returns
    -------
    rectangle : ComplexRectangle
        Rectangle with exact-zero imaginary component.
    """
    return value[0], value[1], _ZERO, _ZERO


def _rectangle_imag_interval(value: _RealInterval) -> ComplexRectangle:
    """PRIVATE: Embed one imaginary interval as a complex rectangle.

    Parameters
    ----------
    value : _RealInterval
        Exact imaginary interval.

    Returns
    -------
    rectangle : ComplexRectangle
        Rectangle with exact-zero real component.
    """
    return _ZERO, _ZERO, value[0], value[1]


def _rectangle_divide_positive(
    value: ComplexRectangle,
    denominator: _RealInterval,
    ledger: _DirectRationalLedger,
) -> ComplexRectangle:
    """PRIVATE: Divide a complex rectangle by a positive real interval.

    Parameters
    ----------
    value : ComplexRectangle
        Numerator rectangle.
    denominator : _RealInterval
        Strictly positive denominator interval.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : ComplexRectangle
        Exact interval-product quotient hull.

    Raises
    ------
    ZeroDivisionError
        If the denominator has no strictly positive lower endpoint.
    """
    if denominator[0] <= 0:
        raise ZeroDivisionError(
            "positive interval division requires lower > 0"
        )
    reciprocal = (
        ledger.divide(_ONE, denominator[1]),
        ledger.divide(_ONE, denominator[0]),
    )
    return ledger.rectangle_multiply(
        value, _rectangle_real_interval(reciprocal)
    )


def _point_rectangle(value: complex) -> ComplexRectangle:
    """PRIVATE: Embed one finite complex128 point as exact dyadics.

    Parameters
    ----------
    value : complex
        Stored complex binary64 point.

    Returns
    -------
    rectangle : ComplexRectangle
        Degenerate exact-rational complex rectangle.
    """
    real = Fraction.from_float(float(np.real(value)))
    imag = Fraction.from_float(float(np.imag(value)))
    return real, real, imag, imag


def _sentinel_rectangle() -> ComplexRectangle:
    """PRIVATE: Return an internal marker for unavailable exact arithmetic.

    Returns
    -------
    rectangle : ComplexRectangle
        Exact-zero placeholder converted to unbounded storage by callers.
    """
    return _ZERO, _ZERO, _ZERO, _ZERO


class _EntireRecorder:
    """Collect deterministic per-kernel exact entire-helper evidence."""

    def __init__(
        self,
        policies: _EntirePolicies,
        hull: _OutwardDyadicHullLedger,
    ) -> None:
        self.policies = policies
        self.hull = hull
        self.labels: list[str] = []
        self.transcripts: list[EntireWorkTranscript | None] = []
        self.failures: list[EntireEnclosureFailure | None] = []
        self.failure_work: list[int] = []

    def call(
        self,
        label: str,
        kind: str,
        rectangle: ComplexRectangle,
    ) -> ComplexRectangle | None:
        """Issue and record one exact entire-helper call."""
        precision, terms, work, reductions, bits = self.policies
        helper = {
            "exp": enclose_complex_exp,
            "phi1": enclose_complex_exprel,
            "phi2": enclose_complex_phi2,
        }.get(kind)
        if helper is None:
            raise ValueError("unknown exact entire-helper kind")
        self.labels.append(label)
        try:
            enclosure, transcript = helper(
                rectangle,
                precision_bits=precision,
                maximum_terms=terms,
                maximum_work=work,
                maximum_range_reductions=reductions,
                maximum_rational_bits=bits,
            )
        except EntireEnclosureError as error:
            self.transcripts.append(None)
            self.failures.append(error.failure)
            self.failure_work.append(error.exact_work_count)
            return None
        self.transcripts.append(transcript)
        self.failures.append(None)
        self.failure_work.append(0)
        return self.hull.rectangle(enclosure)

    def evidence(self) -> GalerkinLocalVacuumTerminalEntireEvidence:
        """Build the validated aggregate helper evidence carrier."""
        labels = tuple(self.labels)
        transcripts = tuple(self.transcripts)
        failures = tuple(self.failures)
        failure_work = tuple(self.failure_work)
        digest = sha256(
            {
                "domain": "ptyrodactyl.local_vacuum_terminal.entire.v1",
                "labels": stored_value_payload(labels),
                "transcripts": stored_value_payload(transcripts),
                "failures": stored_value_payload(failures),
                "failure_work": stored_value_payload(failure_work),
                "policies": stored_value_payload(self.policies),
                "helper_attempted": bool(labels),
                "helper_eligible": bool(labels)
                and not any(value is not None for value in failures),
            }
        )
        return _make_local_vacuum_terminal_entire_evidence(
            labels,
            transcripts,
            failures,
            failure_work,
            self.policies,
            jnp.asarray(bool(labels)),
            jnp.asarray(
                bool(labels)
                and not any(value is not None for value in failures)
            ),
            helper_evidence_digest=digest,
        )


def _real_interval_scale(
    value: _RealInterval,
    scalar: Fraction,
    ledger: _DirectRationalLedger,
) -> _RealInterval:
    """PRIVATE: Scale one real interval by an exact rational.

    Parameters
    ----------
    value : _RealInterval
        Ordered interval.
    scalar : Fraction
        Exact real scale.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    result : _RealInterval
        Ordered scaled interval.
    """
    products = (
        ledger.multiply(value[0], scalar),
        ledger.multiply(value[1], scalar),
    )
    return min(products), max(products)


def _rectangles_from_rational(
    values: list[ComplexRectangle],
) -> GalerkinLocalTerminalComplexRectangles:
    """PRIVATE: Convert exact rectangles to outward binary64 storage.

    Parameters
    ----------
    values : list[ComplexRectangle]
        Ordered exact rational rectangles.

    Returns
    -------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Componentwise outward float64 endpoints.
    """
    columns = list(zip(*values, strict=True))
    arrays = (
        np.asarray(
            [fraction_lower_float(value) for value in columns[0]],
            dtype=np.float64,
        ),
        np.asarray(
            [fraction_upper_float(value) for value in columns[1]],
            dtype=np.float64,
        ),
        np.asarray(
            [fraction_lower_float(value) for value in columns[2]],
            dtype=np.float64,
        ),
        np.asarray(
            [fraction_upper_float(value) for value in columns[3]],
            dtype=np.float64,
        ),
    )
    return GalerkinLocalTerminalComplexRectangles(
        *(jnp.asarray(value) for value in arrays)
    )


def _unbounded_rectangles(
    size: int,
) -> GalerkinLocalTerminalComplexRectangles:
    """PRIVATE: Build one fail-closed unbounded rectangle vector.

    Parameters
    ----------
    size : int
        Required vector length.

    Returns
    -------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Componentwise ``[-inf,+inf]`` sentinel rectangles.
    """
    lower = jnp.full((size,), -jnp.inf, dtype=jnp.float64)
    upper = jnp.full((size,), jnp.inf, dtype=jnp.float64)
    return GalerkinLocalTerminalComplexRectangles(lower, upper, lower, upper)


def _rational_from_rectangles(
    values: GalerkinLocalTerminalComplexRectangles,
) -> list[ComplexRectangle]:
    """PRIVATE: Recover exact dyadic rectangles from stored endpoints.

    Parameters
    ----------
    values : GalerkinLocalTerminalComplexRectangles
        Stored componentwise binary64 rectangles.

    Returns
    -------
    rectangles : list[ComplexRectangle]
        Exact dyadic interpretation of every component.
    """
    size = values.real_lower_bounds.shape[0]
    return [_fraction_rectangle(values, index) for index in range(size)]


def _intersect_rectangle_pairs(
    left: _RectanglePair,
    right: _RectanglePair,
    left_available: np.ndarray,
    right_available: np.ndarray,
) -> tuple[_RectanglePair, np.ndarray]:
    """PRIVATE: Intersect two per-fiber, per-role rectangle pairs.

    Parameters
    ----------
    left : _RectanglePair
        First two-role rectangle vector.
    right : _RectanglePair
        Second two-role rectangle vector.
    left_available : np.ndarray
        Per-fiber, per-role availability of the first route.
    right_available : np.ndarray
        Per-fiber, per-role availability of the second route.

    Returns
    -------
    intersections : _RectanglePair
        Exact float-endpoint intersections or unbounded empty sentinels.
    mask : np.ndarray
        Per-fiber, per-role nonempty-intersection mask.
    """
    role_values: list[GalerkinLocalTerminalComplexRectangles] = []
    columns: list[np.ndarray] = []
    for role, (left_role, right_role) in enumerate(
        zip(left, right, strict=True)
    ):
        left_arrays = tuple(np.asarray(value) for value in left_role)
        right_arrays = tuple(np.asarray(value) for value in right_role)
        real_lower = np.maximum(left_arrays[0], right_arrays[0])
        real_upper = np.minimum(left_arrays[1], right_arrays[1])
        imag_lower = np.maximum(left_arrays[2], right_arrays[2])
        imag_upper = np.minimum(left_arrays[3], right_arrays[3])
        present = (
            (real_lower <= real_upper)
            & (imag_lower <= imag_upper)
            & left_available[:, role]
            & right_available[:, role]
        )
        columns.append(present)
        role_values.append(
            GalerkinLocalTerminalComplexRectangles(
                jnp.asarray(np.where(present, real_lower, -np.inf)),
                jnp.asarray(np.where(present, real_upper, np.inf)),
                jnp.asarray(np.where(present, imag_lower, -np.inf)),
                jnp.asarray(np.where(present, imag_upper, np.inf)),
            )
        )
    return (role_values[0], role_values[1]), np.stack(columns, axis=1)


def _optional_rectangle_pair(
    values: _OptionalRectangleLists,
) -> tuple[_RectanglePair, np.ndarray, bool]:
    """PRIVATE: Store optional rectangles and their explicit availability.

    Parameters
    ----------
    values : _OptionalRectangleLists
        Exact per-role rectangles with typed unavailable entries.

    Returns
    -------
    rectangles : _RectanglePair
        Outward binary64 rectangles with unbounded unavailable sentinels.
    available : np.ndarray
        Per-fiber, per-role availability mask.
    storage_range_ok : bool
        Whether every attempted outward conversion was normal-or-zero.
    """
    first, first_ok = _optional_rectangles(values[0])
    second, second_ok = _optional_rectangles(values[1])
    rectangles = (
        first,
        second,
    )
    available = np.stack(
        [
            np.all(
                np.stack([np.isfinite(np.asarray(column)) for column in role]),
                axis=0,
            )
            for role in rectangles
        ],
        axis=1,
    )
    return rectangles, available, first_ok and second_ok


def _physical_q_intervals(
    target: GalerkinLocalCellTargetManifest,
    fibers: np.ndarray,
    direct: _DirectRationalLedger,
) -> list[_RealInterval]:
    """PRIVATE: Derive exact-ledger physical transverse-root intervals.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local target.
    fibers : np.ndarray
        Canonical scoped transverse reciprocal indices.
    direct : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    intervals : list[_RealInterval]
        Exact rational LVT.39 ``k0^2-|k_perp|^2`` enclosures.
    """
    axis = target.acquisition.terminal_axis
    transverse_axes = tuple(index for index in range(3) if index != axis)
    ledger = target.fixed_linear_error_ledger
    k0 = (
        Fraction.from_float(
            float(np.asarray(ledger.exact_wavenumber_lower_bound))
        ),
        Fraction.from_float(
            float(np.asarray(ledger.exact_wavenumber_upper_bound))
        ),
    )
    k0 = (direct.retain(k0[0]), direct.retain(k0[1]))
    k0_squared = _interval_square(k0, direct)
    pi_values = mathematical_pi_interval()
    pi_interval = (
        Fraction.from_float(float(np.asarray(pi_values[0]))),
        Fraction.from_float(float(np.asarray(pi_values[1]))),
    )
    box_lengths = np.asarray(target.box_lengths, dtype=np.float64)
    carrier_lower = np.asarray(
        ledger.exact_carrier_lower_bounds, dtype=np.float64
    )
    carrier_upper = np.asarray(
        ledger.exact_carrier_upper_bounds, dtype=np.float64
    )
    intervals: list[_RealInterval] = []
    for fiber in fibers:
        transverse_squared = (_ZERO, _ZERO)
        for component, component_axis in zip(
            fiber, transverse_axes, strict=True
        ):
            carrier = (
                Fraction.from_float(float(carrier_lower[component_axis])),
                Fraction.from_float(float(carrier_upper[component_axis])),
            )
            length = Fraction.from_float(float(box_lengths[component_axis]))
            reciprocal = direct.divide(Fraction(2 * int(component), 1), length)
            offset = direct.interval_multiply(
                (reciprocal, reciprocal), pi_interval
            )
            physical = _interval_add(carrier, offset, direct)
            transverse_squared = _interval_add(
                transverse_squared, _interval_square(physical, direct), direct
            )
        intervals.append(
            _interval_subtract(k0_squared, transverse_squared, direct)
        )
    return intervals


def _oriented_normal_intervals(
    target: GalerkinLocalCellTargetManifest,
    direct: _DirectRationalLedger,
) -> list[_RealInterval]:
    """PRIVATE: Build exact-ledger side-oriented normal frequencies.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local target.
    direct : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    intervals : list[_RealInterval]
        Exact per-state side-oriented physical normal-wavevector intervals.
    """
    axis = target.acquisition.terminal_axis
    side = (
        1
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else -1
    )
    ledger = target.fixed_linear_error_ledger
    carrier = (
        Fraction.from_float(
            float(np.asarray(ledger.exact_carrier_lower_bounds[axis]))
        ),
        Fraction.from_float(
            float(np.asarray(ledger.exact_carrier_upper_bounds[axis]))
        ),
    )
    length = direct.retain(
        Fraction.from_float(float(np.asarray(target.box_lengths[axis])))
    )
    pi_values = mathematical_pi_interval()
    pi_interval = (
        Fraction.from_float(float(np.asarray(pi_values[0]))),
        Fraction.from_float(float(np.asarray(pi_values[1]))),
    )
    indices = np.asarray(target.state_indices[:, axis], dtype=np.int64)
    intervals: list[_RealInterval] = []
    for index in indices:
        reciprocal = direct.divide(Fraction(2 * int(index)), length)
        offset = direct.interval_multiply(
            (reciprocal, reciprocal), pi_interval
        )
        physical = _interval_add(carrier, offset, direct)
        intervals.append(
            physical
            if side > 0
            else (
                direct.multiply(Fraction(-1), physical[1]),
                direct.multiply(Fraction(-1), physical[0]),
            )
        )
    return intervals


def _physical_zero_witness(
    q_interval: _RealInterval,
    state_rows: np.ndarray,
    selected: np.ndarray,
    fiber_row: int,
    free_intervals: list[_RealInterval],
    normal_intervals: list[_RealInterval],
    maximum_rational_bits: int,
) -> tuple[GalerkinLocalVacuumZeroWitness | None, int | None]:
    """PRIVATE: Reconstruct target-owned physical grazing support.

    Parameters
    ----------
    q_interval : _RealInterval
        Direct exact physical LVT.39 interval.
    state_rows : np.ndarray
        Scoped-fiber row for every retained coefficient.
    selected : np.ndarray
        Exact scoped state-membership mask.
    fiber_row : int
        Scoped fiber being classified.
    free_intervals : list[_RealInterval]
        Parent exact free-diagonal intervals.
    normal_intervals : list[_RealInterval]
        Exact-ledger oriented normal-wavevector intervals.
    maximum_rational_bits : int
        Independent formal-witness rational-size policy.

    Returns
    -------
    witness : GalerkinLocalVacuumZeroWitness | None
        Internally reconstructed formal zero witness when physically proven.
    support_row : int | None
        Supporting state row, or no row for direct singleton-zero q.
    """
    route: GalerkinLocalVacuumZeroWitnessRoute | None = None
    support_row: int | None = None
    if q_interval == (_ZERO, _ZERO):
        route = GalerkinLocalVacuumZeroWitnessRoute.EXACT_RATIONAL_DIFFERENCE
    else:
        for index in np.flatnonzero(selected & (state_rows == fiber_row)):
            row = int(index)
            if free_intervals[row] == (_ZERO, _ZERO) and normal_intervals[
                row
            ] == (_ZERO, _ZERO):
                support_row = row
                route = GalerkinLocalVacuumZeroWitnessRoute.SYMBOLIC_NORMAL_FORM_DIFFERENCE  # noqa: E501
                break
    if route is None:
        return None, None
    normal_form = (("authenticated_physical_q_zero", Fraction(1)),)
    witness = make_local_vacuum_zero_witness(
        normal_form,
        normal_form,
        route=route,
        maximum_rational_bits=maximum_rational_bits,
    )
    return witness, support_row


def _carrier_phase_rectangle(
    target: GalerkinLocalCellTargetManifest,
    coordinate: Fraction,
    label: str,
    recorder: _EntireRecorder,
    direct: _DirectRationalLedger,
) -> ComplexRectangle | None:
    """PRIVATE: Enclose one exact-ledger physical carrier phase.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local target.
    coordinate : Fraction
        Exact stored plane coordinate.
    label : str
        Deterministic helper transcript label.
    recorder : _EntireRecorder
        Per-call exact entire-helper recorder.
    direct : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    phase : ComplexRectangle | None
        Exact ``exp(i*k_i,n*xi)`` rectangle, or a typed failed-call marker.
    """
    axis = target.acquisition.terminal_axis
    ledger = target.fixed_linear_error_ledger
    carrier = (
        Fraction.from_float(
            float(np.asarray(ledger.exact_carrier_lower_bounds[axis]))
        ),
        Fraction.from_float(
            float(np.asarray(ledger.exact_carrier_upper_bounds[axis]))
        ),
    )
    exponent = _real_interval_scale(carrier, coordinate, direct)
    return recorder.call(label, "exp", _rectangle_imag_interval(exponent))


def _physical_cauchy_rectangles(
    diagnostic: GalerkinLocalCoordinateCauchyCurrent,
    phase: ComplexRectangle,
    direct: _DirectRationalLedger,
) -> tuple[list[ComplexRectangle], list[ComplexRectangle]]:
    """PRIVATE: Reconstruct exact physical Cauchy rectangles at one plane.

    Parameters
    ----------
    diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt exact-target L7 submitted-state diagnostic.
    phase : ComplexRectangle
        Exact physical carrier-phase enclosure at the bound plane.
    direct : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    trace : list[ComplexRectangle]
        Exact physical field-trace rectangles by scoped fiber.
    normal : list[ComplexRectangle]
        Exact side-oriented normal-derivative rectangles by scoped fiber.
    """
    action = diagnostic.action_enclosure
    operator = action.certificate
    field = np.asarray(action.submitted_field, dtype=np.complex128)
    rows = np.asarray(operator.state_to_fiber_rows, dtype=np.int64)
    selected = np.asarray(operator.selected_state_mask, dtype=np.bool_)
    fiber_size = operator.scope_transverse_indices.shape[0]
    coefficient_roles = (
        operator.exact_trace_coefficient_rectangles,
        operator.exact_normal_coefficient_rectangles,
    )
    outputs: list[list[ComplexRectangle]] = []
    for coefficients in coefficient_roles:
        terms: list[list[ComplexRectangle]] = [[] for _ in range(fiber_size)]
        for index, value in enumerate(field):
            if bool(selected[index]):
                product = direct.rectangle_multiply(
                    _fraction_rectangle(coefficients, index),
                    _point_rectangle(complex(value)),
                )
                terms[int(rows[index])].append(product)
        outputs.append(
            [
                direct.rectangle_multiply(
                    phase,
                    _rectangle_sum(row_terms, direct),
                )
                for row_terms in terms
            ]
        )
    return outputs[0], outputs[1]


def _rectangle_sum(
    values: list[ComplexRectangle], ledger: _DirectRationalLedger
) -> ComplexRectangle:
    """PRIVATE: Sum rectangles through the checked direct ledger.

    Parameters
    ----------
    values : list[ComplexRectangle]
        Deterministically ordered exact rectangle terms.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    total : ComplexRectangle
        Checked exact componentwise sum.
    """
    total = (_ZERO, _ZERO, _ZERO, _ZERO)
    for value in values:
        total = _rectangle_add(total, value, ledger)
    return total


def _physical_phase_point(
    target: GalerkinLocalCellTargetManifest, coordinate: float
) -> complex:
    """PRIVATE: Evaluate the frozen physical carrier phase realization.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local target.
    coordinate : float
        Exact stored binary64 plane coordinate.

    Returns
    -------
    phase : complex
        Frozen complex128 ``exp(i*k_i,n*xi)`` point.
    """
    axis = target.acquisition.terminal_axis
    carrier = float(np.asarray(target.carrier[axis]))
    return complex(np.complex128(np.exp(1.0j * carrier * coordinate)))


def _classify_physical_roots(  # noqa: PLR0913
    projection: GalerkinLocalProjectionDefectCertificate,
    q_intervals: list[_RealInterval],
    distance: Fraction,
    maximum_root_work: int,
    entire_policies: _EntirePolicies,
    maximum_interval_work: int,
    maximum_rational_bits: int,
    direct: _DirectRationalLedger,
    normal_intervals: list[_RealInterval],
) -> tuple[
    tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    tuple[GalerkinLocalVacuumPropagator | None, ...],
    tuple[int | None, ...],
    _PropagationFailures,
    tuple[int, ...],
    _PropagationFailures,
    tuple[int, ...],
]:
    """PRIVATE: Build physical roots and fully replayed propagators.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed projection parent.
    q_intervals : list[_RealInterval]
        Exact target-owned physical LVT.39 intervals.
    distance : Fraction
        Exact nonnegative slab distance.
    maximum_root_work : int
        Independent root-helper work policy.
    entire_policies : _EntirePolicies
        Precision, term, work, range, and rational-bit helper policies.
    maximum_interval_work : int
        Independent propagator post-helper work policy.
    maximum_rational_bits : int
        Independent shared root and propagator rational-size policy.
    direct : _DirectRationalLedger
        Active source-local physical-provenance arithmetic ledger.
    normal_intervals : list[_RealInterval]
        Checked exact-ledger oriented normal-wavevector intervals.

    Returns
    -------
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Strict physical roots, with typed missing entries on helper failure.
    propagators : tuple[GalerkinLocalVacuumPropagator | None, ...]
        Fully replayed propagators for every classified root.
    support_rows : tuple[int | None, ...]
        Physical grazing support rows, if the alternate route was used.
    root_failures : _PropagationFailures
        Typed root-helper failures, otherwise none.
    root_failure_work : tuple[int, ...]
        Exact completed work for each root failure, otherwise zero.
    propagator_failures : _PropagationFailures
        Typed propagator-helper failures, otherwise none or skipped.
    propagator_failure_work : tuple[int, ...]
        Exact completed work for each propagator failure, otherwise zero.
    """
    rows = np.asarray(projection.state_to_fiber_rows, dtype=np.int64)
    selected = np.asarray(projection.selected_state_mask, dtype=np.bool_)
    free_intervals = [
        (
            direct.retain(Fraction.from_float(float(lower))),
            direct.retain(Fraction.from_float(float(upper))),
        )
        for lower, upper in zip(
            np.asarray(projection.exact_free_diagonal_lower_bounds),
            np.asarray(projection.exact_free_diagonal_upper_bounds),
            strict=True,
        )
    ]
    precision, terms, entire_work, reductions, _ = entire_policies
    roots: list[GalerkinLocalVacuumRootCertificate | None] = []
    propagators: list[GalerkinLocalVacuumPropagator | None] = []
    support_rows: list[int | None] = []
    root_failures: list[
        EntireEnclosureFailure | GalerkinLocalVacuumPropagationFailure | None
    ] = []
    root_failure_work: list[int] = []
    propagator_failures: list[
        EntireEnclosureFailure | GalerkinLocalVacuumPropagationFailure | None
    ] = []
    propagator_failure_work: list[int] = []
    for fiber_row, q_interval in enumerate(q_intervals):
        try:
            witness, support_row = _physical_zero_witness(
                q_interval,
                rows,
                selected,
                fiber_row,
                free_intervals,
                normal_intervals,
                maximum_rational_bits,
            )
            root = classify_local_vacuum_root(
                q_interval,
                zero_witness=witness,
                maximum_root_work=maximum_root_work,
                maximum_rational_bits=maximum_rational_bits,
            )
        except (
            EntireEnclosureError,
            GalerkinLocalVacuumPropagationError,
        ) as error:
            roots.append(None)
            propagators.append(None)
            support_rows.append(None)
            root_failures.append(error.failure)
            root_failure_work.append(error.exact_work_count)
            propagator_failures.append(None)
            propagator_failure_work.append(0)
            continue
        root = prepare_local_vacuum_root_certificate(
            root,
            maximum_root_work=maximum_root_work,
            maximum_rational_bits=maximum_rational_bits,
        )
        support_rows.append(support_row)
        roots.append(root)
        root_failures.append(None)
        root_failure_work.append(0)
        if root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED:
            propagators.append(None)
            propagator_failures.append(None)
            propagator_failure_work.append(0)
            continue
        try:
            propagator = enclose_local_vacuum_propagator(
                root,
                distance,
                maximum_root_work=maximum_root_work,
                precision_bits=precision,
                maximum_terms=terms,
                maximum_entire_work=entire_work,
                maximum_range_reductions=reductions,
                maximum_interval_work=maximum_interval_work,
                maximum_rational_bits=maximum_rational_bits,
            )
        except (
            EntireEnclosureError,
            GalerkinLocalVacuumPropagationError,
        ) as error:
            propagators.append(None)
            propagator_failures.append(error.failure)
            propagator_failure_work.append(error.exact_work_count)
            continue
        propagator = prepare_local_vacuum_propagator(
            propagator,
            maximum_root_work=maximum_root_work,
            precision_bits=precision,
            maximum_terms=terms,
            maximum_entire_work=entire_work,
            maximum_range_reductions=reductions,
            maximum_interval_work=maximum_interval_work,
            maximum_rational_bits=maximum_rational_bits,
        )
        propagators.append(propagator)
        propagator_failures.append(None)
        propagator_failure_work.append(0)
    return (
        tuple(roots),
        tuple(propagators),
        tuple(support_rows),
        tuple(root_failures),
        tuple(root_failure_work),
        tuple(propagator_failures),
        tuple(propagator_failure_work),
    )


def _hull_branch_root_intervals(
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    hull: _OutwardDyadicHullLedger,
) -> _OptionalRootIntervals:
    """PRIVATE: Copy raw root intervals once for bounded branch arithmetic.

    Parameters
    ----------
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Fully replayed raw root evidence retained unchanged in the carrier.
    hull : _OutwardDyadicHullLedger
        Active fixed-algorithm hull transcript.

    Returns
    -------
    intervals : _OptionalRootIntervals
        Outward normal-binary64 dyadic copies, or unavailable route markers.

    Raises
    ------
    ValueError
        If classified root evidence omits its required interval.
    """
    intervals: list[_RealInterval | None] = []
    for root in roots:
        if root is None or root.classification is (
            GalerkinLocalVacuumRootClass.UNCLASSIFIED
        ):
            intervals.append(None)
            continue
        raw = root.root_interval
        if raw is None:
            raise ValueError("classified branch root lacks its raw interval")
        interval = hull.interval((raw.lower, raw.upper))
        if interval is None:
            intervals.append(None)
            continue
        if (
            root.classification is not GalerkinLocalVacuumRootClass.GRAZING
            and interval[0] <= 0
        ):
            hull.reject_nonpositive_root_lower()
            intervals.append(None)
            continue
        intervals.append(interval)
    return tuple(intervals)


def _hull_branch_propagator_entries(
    propagators: tuple[GalerkinLocalVacuumPropagator | None, ...],
    hull: _OutwardDyadicHullLedger,
) -> _OptionalPropagatorEntries:
    """PRIVATE: Copy prepared propagator entries once before direct work.

    Parameters
    ----------
    propagators : tuple[GalerkinLocalVacuumPropagator | None, ...]
        Fully replayed raw propagator evidence retained unchanged.
    hull : _OutwardDyadicHullLedger
        Active fixed-algorithm hull transcript.

    Returns
    -------
    entries : _OptionalPropagatorEntries
        Four dyadic real intervals per available propagator.

    Raises
    ------
    ValueError
        If a prepared propagator does not contain exactly four entries.
    AssertionError
        If a checked four-entry hull unexpectedly loses an entry.
    """
    results: list[
        tuple[_RealInterval, _RealInterval, _RealInterval, _RealInterval]
        | None
    ] = []
    for propagator in propagators:
        if propagator is None:
            results.append(None)
            continue
        if len(propagator.entries) != _PROPAGATOR_ENTRY_COUNT:
            raise ValueError("prepared propagator must contain four entries")
        copied = [
            hull.interval((entry.lower, entry.upper))
            for entry in propagator.entries
        ]
        if any(entry is None for entry in copied):
            results.append(None)
            continue
        first, second, third, fourth = copied
        if first is None or second is None or third is None or fourth is None:
            raise AssertionError(
                "checked propagator hull unexpectedly missing"
            )
        results.append((first, second, third, fourth))
    return tuple(results)


def _root_realization(
    root: GalerkinLocalVacuumRootCertificate,
    ledger: _DirectRationalLedger,
) -> tuple[float, float]:
    """PRIVATE: Build the canonical rounded positive root audit record.

    Parameters
    ----------
    root : GalerkinLocalVacuumRootCertificate
        Classified fully replayed physical root.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    realization : float
        Nearest binary64 midpoint realization, or exact zero at grazing.
    error : float
        Outward maximum distance to the full exact root interval.

    Raises
    ------
    _LocalArithmeticRangeError
        If the rounded root or its audit error is outside normal binary64.
    ValueError
        If the classified root lacks its interval.
    """
    interval = root.root_interval
    if interval is None:
        raise ValueError("classified root is missing its root interval")
    if root.classification is GalerkinLocalVacuumRootClass.GRAZING:
        return 0.0, 0.0
    midpoint = ledger.divide(
        ledger.add(interval.lower, interval.upper), Fraction(2)
    )
    try:
        realization = float(midpoint)
    except OverflowError as error:
        raise _LocalArithmeticRangeError(
            "root midpoint has no finite float64 realization"
        ) from error
    if (
        not math.isfinite(realization)
        or realization <= 0.0
        or not _normal_or_zero(np.float64(realization))
    ):
        raise _LocalArithmeticRangeError(
            "root midpoint has no positive normal float64"
        )
    exact = ledger.retain(Fraction.from_float(realization))
    lower_error = abs(ledger.subtract(exact, interval.lower))
    upper_error = abs(ledger.subtract(exact, interval.upper))
    error_fraction = ledger.retain(max(lower_error, upper_error))
    error = fraction_upper_float(error_fraction)
    if not _normal_or_zero(np.float64(error)):
        raise _LocalArithmeticRangeError(
            "root realization error is outside normal float64"
        )
    return realization, error


def _optional_rectangles(
    values: list[ComplexRectangle | None],
) -> tuple[GalerkinLocalTerminalComplexRectangles, bool]:
    """PRIVATE: Convert optional exact rectangles with fail-closed sentinels.

    Parameters
    ----------
    values : list[ComplexRectangle | None]
        Exact rectangles or typed unavailable entries.

    Returns
    -------
    rectangles : GalerkinLocalTerminalComplexRectangles
        Outward endpoints, with unbounded unavailable components.
    storage_range_ok : bool
        Whether every attempted outward conversion was normal-or-zero.
    """
    columns: list[list[float]] = [[], [], [], []]
    storage_range_ok = True
    for value in values:
        if value is None:
            converted = (-np.inf, np.inf, -np.inf, np.inf)
        else:
            converted = (
                fraction_lower_float(value[0]),
                fraction_upper_float(value[1]),
                fraction_lower_float(value[2]),
                fraction_upper_float(value[3]),
            )
            if not _normal_or_zero(np.asarray(converted, dtype=np.float64)):
                storage_range_ok = False
                converted = (-np.inf, np.inf, -np.inf, np.inf)
        for column, endpoint in zip(columns, converted, strict=True):
            column.append(endpoint)
    rectangles = GalerkinLocalTerminalComplexRectangles(
        *(jnp.asarray(column, dtype=jnp.float64) for column in columns)
    )
    return rectangles, storage_range_ok


def _branch_transform(
    trace: ComplexRectangle,
    normal: ComplexRectangle,
    root: GalerkinLocalVacuumRootCertificate,
    root_interval: _RealInterval | None,
    ledger: _DirectRationalLedger,
) -> tuple[ComplexRectangle, ComplexRectangle]:
    """PRIVATE: Apply the exact side-oriented branch map at one plane.

    Parameters
    ----------
    trace : ComplexRectangle
        Physical field-trace rectangle.
    normal : ComplexRectangle
        Physical side-oriented normal rectangle.
    root : GalerkinLocalVacuumRootCertificate
        Classified positive or grazing physical root.
    root_interval : _RealInterval | None
        Once-hulled branch-consumption root interval.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    primary : ComplexRectangle
        Outward, decaying, or grazing-field role.
    secondary : ComplexRectangle
        Inward, growing, or grazing-derivative role.

    Raises
    ------
    ValueError
        If a nongrazing root interval is absent.
    """
    if root.classification is GalerkinLocalVacuumRootClass.GRAZING:
        primary, secondary = trace, normal
    else:
        if root_interval is None:
            raise ValueError(
                "nongrazing branch map requires a hulled root interval"
            )
        divided = _rectangle_divide_positive(normal, root_interval, ledger)
        if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
            quotient = (
                divided[2],
                divided[3],
                -divided[1],
                -divided[0],
            )
            primary = ledger.rectangle_scale(
                _rectangle_add(trace, quotient, ledger), _HALF
            )
            secondary = ledger.rectangle_scale(
                _rectangle_subtract(trace, quotient, ledger), _HALF
            )
        else:
            primary = ledger.rectangle_scale(
                _rectangle_subtract(trace, divided, ledger), _HALF
            )
            secondary = ledger.rectangle_scale(
                _rectangle_add(trace, divided, ledger), _HALF
            )
    primary = (
        ledger.retain(primary[0]),
        ledger.retain(primary[1]),
        ledger.retain(primary[2]),
        ledger.retain(primary[3]),
    )
    secondary = (
        ledger.retain(secondary[0]),
        ledger.retain(secondary[1]),
        ledger.retain(secondary[2]),
        ledger.retain(secondary[3]),
    )
    return primary, secondary


def _endpoint_cauchy_mismatch(
    inner: tuple[list[ComplexRectangle], list[ComplexRectangle]],
    outer: tuple[list[ComplexRectangle], list[ComplexRectangle]],
    propagator_entries: _OptionalPropagatorEntries,
    ledger: _DirectRationalLedger,
) -> tuple[list[ComplexRectangle | None], list[ComplexRectangle | None]]:
    """PRIVATE: Enclose outer minus propagated-inner Cauchy mismatch.

    Parameters
    ----------
    inner : tuple[list[ComplexRectangle], list[ComplexRectangle]]
        Inner physical field and side-normal rectangles.
    outer : tuple[list[ComplexRectangle], list[ComplexRectangle]]
        Outer physical field and side-normal rectangles.
    propagator_entries : _OptionalPropagatorEntries
        Per-fiber once-hulled homogeneous propagator entries.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    field : list[ComplexRectangle | None]
        Endpoint-route field mismatch rectangles.
    normal : list[ComplexRectangle | None]
        Endpoint-route side-normal mismatch rectangles.
    """
    field: list[ComplexRectangle | None] = []
    normal: list[ComplexRectangle | None] = []
    for index, entries in enumerate(propagator_entries):
        if entries is None:
            field.append(None)
            normal.append(None)
            continue
        propagated_field = _rectangle_add(
            ledger.rectangle_multiply(
                _rectangle_real_interval(entries[0]), inner[0][index]
            ),
            ledger.rectangle_multiply(
                _rectangle_real_interval(entries[1]), inner[1][index]
            ),
            ledger,
        )
        propagated_normal = _rectangle_add(
            ledger.rectangle_multiply(
                _rectangle_real_interval(entries[2]), inner[0][index]
            ),
            ledger.rectangle_multiply(
                _rectangle_real_interval(entries[3]), inner[1][index]
            ),
            ledger,
        )
        field.append(
            _rectangle_subtract(outer[0][index], propagated_field, ledger)
        )
        normal.append(
            _rectangle_subtract(outer[1][index], propagated_normal, ledger)
        )
    return field, normal


def _j_kernel(
    a: ComplexRectangle,
    oriented_lambda: _RealInterval,
    distance: Fraction,
    label: str,
    recorder: _EntireRecorder,
    ledger: _DirectRationalLedger,
) -> ComplexRectangle | None:
    """PRIVATE: Enclose one cancellation-safe finite exponential integral.

    Parameters
    ----------
    a : ComplexRectangle
        Exact propagating or evanescent homogeneous exponent rectangle.
    oriented_lambda : _RealInterval
        Exact outward-coordinate physical forcing frequency.
    distance : Fraction
        Exact nonnegative slab length.
    label : str
        Deterministic kernel transcript label prefix.
    recorder : _EntireRecorder
        Per-call exact entire-helper recorder.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    kernel : ComplexRectangle | None
        Exact ``ell*exp(a*ell)*phi1((i*lambda-a)*ell)`` enclosure, or a
        typed helper-failure marker.
    """
    exponential_argument = ledger.rectangle_scale(a, distance)
    relative = _rectangle_subtract(
        _rectangle_imag_interval(oriented_lambda), a, ledger
    )
    phi_argument = ledger.rectangle_scale(relative, distance)
    exponential = recorder.call(f"{label}.exp", "exp", exponential_argument)
    phi1 = recorder.call(f"{label}.phi1", "phi1", phi_argument)
    if exponential is None or phi1 is None:
        return None
    product = ledger.rectangle_multiply(exponential, phi1)
    return ledger.rectangle_scale(product, distance)


def _forced_cauchy_rectangles(  # noqa: PLR0912,PLR0913,PLR0915
    projection: GalerkinLocalProjectionDefectCertificate,
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    root_intervals: _OptionalRootIntervals,
    inner_coordinate: Fraction,
    distance: Fraction,
    recorder: _EntireRecorder,
    ledger: _DirectRationalLedger,
    oriented_intervals: list[_RealInterval],
) -> tuple[list[ComplexRectangle | None], list[ComplexRectangle | None]]:
    """PRIVATE: Enclose the independent finite forced-integral route.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed parent fixing exact ``d``, state, scope, and mapping.
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Strict physical per-fiber roots.
    root_intervals : _OptionalRootIntervals
        Once-hulled branch-consumption root intervals.
    inner_coordinate : Fraction
        Exact physical inner-plane coordinate.
    distance : Fraction
        Exact outward-coordinate slab length.
    recorder : _EntireRecorder
        Per-kernel phase, exp, phi1, and phi2 helper recorder.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.
    oriented_intervals : list[_RealInterval]
        Checked exact per-state outward-coordinate forcing frequencies.

    Returns
    -------
    field : list[ComplexRectangle | None]
        Exact LVT.44a field-component forced mismatch rectangles.
    normal : list[ComplexRectangle | None]
        Exact LVT.44a side-normal forced mismatch rectangles.

    Raises
    ------
    ValueError
        If a classified nongrazing root lacks its positive interval.
    """
    target = projection.zero_slab_certificate.represented_source_certificate.source.target  # noqa: E501
    side = (
        1
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else -1
    )
    rows = np.asarray(projection.state_to_fiber_rows, dtype=np.int64)
    selected = np.asarray(projection.selected_state_mask, dtype=np.bool_)
    field_values = np.asarray(
        projection.stability_result.solve_result.field, dtype=np.complex128
    )
    free_lower = np.asarray(projection.exact_free_diagonal_lower_bounds)
    free_upper = np.asarray(projection.exact_free_diagonal_upper_bounds)
    length = ledger.retain(
        Fraction.from_float(
            float(
                np.asarray(
                    target.box_lengths[target.acquisition.terminal_axis]
                )
            )
        )
    )
    normalization = (
        ledger.divide(_ONE, ledger.root_upper(length)),
        ledger.root_upper(ledger.divide(_ONE, length)),
    )
    normalization_rectangle = _rectangle_real_interval(normalization)
    field_terms: list[list[ComplexRectangle]] = [[] for _ in range(len(roots))]
    normal_terms: list[list[ComplexRectangle]] = [
        [] for _ in range(len(roots))
    ]
    available = [
        root is not None
        and (
            root.classification is GalerkinLocalVacuumRootClass.GRAZING
            or interval is not None
        )
        for root, interval in zip(roots, root_intervals, strict=True)
    ]

    for state_index, state_value in enumerate(field_values):
        if not bool(selected[state_index]):
            continue
        fiber = int(rows[state_index])
        root = roots[fiber]
        branch_root_interval = root_intervals[fiber]
        oriented: _RealInterval = (
            ledger.retain(oriented_intervals[state_index][0]),
            ledger.retain(oriented_intervals[state_index][1]),
        )
        physical: _RealInterval = (
            oriented if side > 0 else (-oriented[1], -oriented[0])
        )
        phase_argument = _rectangle_imag_interval(
            _real_interval_scale(physical, inner_coordinate, ledger)
        )
        prefix = f"forced.state_{state_index}"
        phase = recorder.call(
            f"{prefix}.physical_inner_phase.exp", "exp", phase_argument
        )

        field_kernel: ComplexRectangle | None = None
        normal_kernel: ComplexRectangle | None = None
        if root is not None and root.classification is not (
            GalerkinLocalVacuumRootClass.UNCLASSIFIED
        ):
            root_interval = branch_root_interval
            if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
                if root_interval is None:
                    available[fiber] = False
                    continue
                plus = _j_kernel(
                    _rectangle_imag_interval(root_interval),
                    oriented,
                    distance,
                    f"{prefix}.propagating_plus",
                    recorder,
                    ledger,
                )
                minus_interval = (-root_interval[1], -root_interval[0])
                minus = _j_kernel(
                    _rectangle_imag_interval(minus_interval),
                    oriented,
                    distance,
                    f"{prefix}.propagating_minus",
                    recorder,
                    ledger,
                )
                if plus is not None and minus is not None:
                    difference = _rectangle_subtract(plus, minus, ledger)
                    divided = _rectangle_divide_positive(
                        difference,
                        (
                            ledger.multiply(Fraction(2), root_interval[0]),
                            ledger.multiply(Fraction(2), root_interval[1]),
                        ),
                        ledger,
                    )
                    field_kernel = (
                        divided[2],
                        divided[3],
                        -divided[1],
                        -divided[0],
                    )
                    normal_kernel = ledger.rectangle_scale(
                        _rectangle_add(plus, minus, ledger), _HALF
                    )
            elif root.classification is (
                GalerkinLocalVacuumRootClass.EVANESCENT
            ):
                if root_interval is None:
                    available[fiber] = False
                    continue
                plus = _j_kernel(
                    _rectangle_real_interval(root_interval),
                    oriented,
                    distance,
                    f"{prefix}.evanescent_plus",
                    recorder,
                    ledger,
                )
                minus_interval = (-root_interval[1], -root_interval[0])
                minus = _j_kernel(
                    _rectangle_real_interval(minus_interval),
                    oriented,
                    distance,
                    f"{prefix}.evanescent_minus",
                    recorder,
                    ledger,
                )
                if plus is not None and minus is not None:
                    field_kernel = _rectangle_divide_positive(
                        _rectangle_subtract(plus, minus, ledger),
                        (
                            ledger.multiply(Fraction(2), root_interval[0]),
                            ledger.multiply(Fraction(2), root_interval[1]),
                        ),
                        ledger,
                    )
                    normal_kernel = ledger.rectangle_scale(
                        _rectangle_add(plus, minus, ledger), _HALF
                    )
            else:
                argument = _rectangle_imag_interval(
                    _real_interval_scale(oriented, distance, ledger)
                )
                phi2 = recorder.call(
                    f"{prefix}.grazing_field.phi2", "phi2", argument
                )
                phi1 = recorder.call(
                    f"{prefix}.grazing_normal.phi1", "phi1", argument
                )
                if phi2 is not None:
                    distance_squared = ledger.multiply(distance, distance)
                    field_kernel = ledger.rectangle_scale(
                        phi2, distance_squared
                    )
                if phi1 is not None:
                    normal_kernel = ledger.rectangle_scale(phi1, distance)

        if (
            root is None
            or root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED
            or phase is None
            or field_kernel is None
            or normal_kernel is None
        ):
            available[fiber] = False
            continue
        free_interval = (
            ledger.retain(Fraction.from_float(float(free_lower[state_index]))),
            ledger.retain(Fraction.from_float(float(free_upper[state_index]))),
        )
        prefactor = ledger.rectangle_multiply(
            _rectangle_real_interval(free_interval),
            _point_rectangle(complex(state_value)),
        )
        prefactor = ledger.rectangle_multiply(
            prefactor, normalization_rectangle
        )
        prefactor = ledger.rectangle_multiply(prefactor, phase)
        prefactor = _rectangle_negate(prefactor)
        field_terms[fiber].append(
            ledger.rectangle_multiply(prefactor, field_kernel)
        )
        normal_terms[fiber].append(
            ledger.rectangle_multiply(prefactor, normal_kernel)
        )

    field: list[ComplexRectangle | None] = []
    normal: list[ComplexRectangle | None] = []
    for fiber in range(len(roots)):
        if not available[fiber]:
            field.append(None)
            normal.append(None)
        else:
            field.append(_rectangle_sum(field_terms[fiber], ledger))
            normal.append(_rectangle_sum(normal_terms[fiber], ledger))
    return field, normal


def _branch_mismatch_rectangles(
    cauchy: _OptionalRectangleLists,
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    root_intervals: _OptionalRootIntervals,
    ledger: _DirectRationalLedger,
) -> tuple[list[ComplexRectangle | None], list[ComplexRectangle | None]]:
    """PRIVATE: Transform optional Cauchy rectangles to branch roles.

    Parameters
    ----------
    cauchy : _OptionalRectangleLists
        Per-fiber field and side-normal rectangles.
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Strict physical roots.
    root_intervals : _OptionalRootIntervals
        Once-hulled intervals used only by branch arithmetic.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    primary : list[ComplexRectangle | None]
        Outward, decaying, or grazing-field rectangles.
    secondary : list[ComplexRectangle | None]
        Inward, growing, or grazing-derivative rectangles.
    """
    primary: list[ComplexRectangle | None] = []
    secondary: list[ComplexRectangle | None] = []
    for trace, normal, root, interval in zip(
        cauchy[0], cauchy[1], roots, root_intervals, strict=True
    ):
        if (
            trace is None
            or normal is None
            or root is None
            or root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED
            or (
                root.classification is not GalerkinLocalVacuumRootClass.GRAZING
                and interval is None
            )
        ):
            primary.append(None)
            secondary.append(None)
            continue
        first, second = _branch_transform(
            trace, normal, root, interval, ledger
        )
        primary.append(first)
        secondary.append(second)
    return primary, secondary


def _rectangle_magnitude_upper(
    value: ComplexRectangle, ledger: _DirectRationalLedger
) -> Fraction:
    """PRIVATE: Bound the magnitude of every point in a rectangle.

    Parameters
    ----------
    value : ComplexRectangle
        Exact complex rectangle.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    upper : Fraction
        Verified rational upper bound on complex magnitude.
    """
    real = ledger.retain(max(abs(value[0]), abs(value[1])))
    imag = ledger.retain(max(abs(value[2]), abs(value[3])))
    squared = ledger.add(
        ledger.multiply(real, real), ledger.multiply(imag, imag)
    )
    return ledger.root_upper(squared)


def _fiber_operator_norms(
    diagnostic: GalerkinLocalCoordinateCauchyCurrent,
    ledger: _DirectRationalLedger,
) -> tuple[list[Fraction], list[Fraction]]:
    """PRIVATE: Reconstruct exact per-fiber L7 trace and normal row norms.

    Parameters
    ----------
    diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt defining-plane L7 diagnostic.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    trace_norms : list[Fraction]
        Verified exact-rational per-fiber trace row-norm uppers.
    normal_norms : list[Fraction]
        Verified exact-rational per-fiber normal row-norm uppers.
    """
    operator = diagnostic.action_enclosure.certificate
    rows = np.asarray(operator.state_to_fiber_rows, dtype=np.int64)
    selected = np.asarray(operator.selected_state_mask, dtype=np.bool_)
    fiber_size = operator.scope_transverse_indices.shape[0]
    results: list[list[Fraction]] = []
    for rectangles in (
        operator.exact_trace_coefficient_rectangles,
        operator.exact_normal_coefficient_rectangles,
    ):
        squared = [_ZERO for _ in range(fiber_size)]
        for index in np.flatnonzero(selected):
            rectangle = _fraction_rectangle(rectangles, int(index))
            magnitude = _rectangle_magnitude_upper(rectangle, ledger)
            row = int(rows[index])
            squared[row] = ledger.add(
                squared[row], ledger.multiply(magnitude, magnitude)
            )
        results.append([ledger.root_upper(value) for value in squared])
    return results[0], results[1]


def _branch_mismatch_factors(  # noqa: PLR0912
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    root_intervals: _OptionalRootIntervals,
    distance: Fraction,
    recorder: _EntireRecorder,
    ledger: _DirectRationalLedger,
) -> list[tuple[Fraction | None, Fraction | None]]:
    """PRIVATE: Enclose LVT.48/LVT.52--LVT.54 kernel factors.

    Parameters
    ----------
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Strict physical roots.
    root_intervals : _OptionalRootIntervals
        Once-hulled branch-consumption root intervals.
    distance : Fraction
        Exact slab length.
    recorder : _EntireRecorder
        Per-kernel exact exponential helper recorder.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    factors : list[tuple[Fraction | None, Fraction | None]]
        Outward role-wise factors multiplying each ``E_f`` component.

    Raises
    ------
    ValueError
        If a classified root lacks its root interval.
    """
    factors: list[tuple[Fraction | None, Fraction | None]] = []
    square_root_distance = ledger.root_upper(distance)
    for fiber, (root, interval) in enumerate(
        zip(roots, root_intervals, strict=True)
    ):
        if root is None or root.classification is (
            GalerkinLocalVacuumRootClass.UNCLASSIFIED
        ):
            factors.append((None, None))
            continue
        if interval is None:
            factors.append((None, None))
            continue
        if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
            denominator = ledger.multiply(Fraction(2), interval[0])
            factor = ledger.divide(square_root_distance, denominator)
            factors.append((factor, factor))
        elif root.classification is GalerkinLocalVacuumRootClass.EVANESCENT:
            lower = ledger.retain(interval[0])
            upper = ledger.retain(interval[1])
            decaying_argument = ledger.multiply(
                Fraction(-2), ledger.multiply(lower, distance)
            )
            growing_argument = ledger.multiply(
                Fraction(2), ledger.multiply(upper, distance)
            )
            decaying_exp = recorder.call(
                f"bound.fiber_{fiber}.evanescent_decaying.exp",
                "exp",
                _rectangle_real_interval(
                    (decaying_argument, decaying_argument)
                ),
            )
            growing_exp = recorder.call(
                f"bound.fiber_{fiber}.evanescent_growing.exp",
                "exp",
                _rectangle_real_interval((growing_argument, growing_argument)),
            )
            decaying: Fraction | None = None
            growing: Fraction | None = None
            denominator = ledger.multiply(Fraction(2), lower)
            if decaying_exp is not None:
                numerator = ledger.subtract(_ONE, decaying_exp[0])
                inner = ledger.divide(numerator, denominator)
                decaying = ledger.divide(ledger.root_upper(inner), denominator)
            if growing_exp is not None:
                numerator = ledger.subtract(growing_exp[1], _ONE)
                inner = ledger.divide(numerator, denominator)
                growing = ledger.divide(ledger.root_upper(inner), denominator)
            factors.append((decaying, growing))
        else:
            field_squared = ledger.divide(
                ledger.multiply(ledger.multiply(distance, distance), distance),
                Fraction(3),
            )
            factors.append(
                (ledger.root_upper(field_squared), square_root_distance)
            )
    return factors


def _plane_mismatch_bounds(
    projection: GalerkinLocalProjectionDefectCertificate,
    factors: list[tuple[Fraction | None, Fraction | None]],
    ledger: _DirectRationalLedger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """PRIVATE: Keep submitted, state-transfer, and total plane bounds apart.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed parent owning measured, transfer, and total ``E_f``.
    factors : list[tuple[Fraction | None, Fraction | None]]
        Exact role-wise mismatch factors.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    submitted : np.ndarray
        Submitted-state forced mismatch upper bounds.
    transfer : np.ndarray
        Projection ``||D0||B`` mismatch transfer bounds.
    total : np.ndarray
        Exact-once outward sums of submitted and transfer bounds.
    storage_range_ok : bool
        Whether every attempted outward binary64 report was normal-or-zero.
    """
    measured = np.asarray(projection.measured_defect_upper_bounds)
    transferred = np.asarray(projection.state_error_transfer_upper_bounds)
    fiber_size = measured.shape[0]
    submitted = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    transfer = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    total = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    storage_range_ok = True
    for fiber, role_factors in enumerate(factors):
        for role, factor in enumerate(role_factors):
            if factor is None:
                continue
            if math.isfinite(float(measured[fiber])):
                submitted_fraction = ledger.multiply(
                    factor, Fraction.from_float(float(measured[fiber]))
                )
                report = fraction_upper_float(submitted_fraction)
                if _normal_or_zero(np.float64(report)):
                    submitted[fiber, role] = report
                else:
                    storage_range_ok = False
            if math.isfinite(float(transferred[fiber])):
                transfer_fraction = ledger.multiply(
                    factor, Fraction.from_float(float(transferred[fiber]))
                )
                report = fraction_upper_float(transfer_fraction)
                if _normal_or_zero(np.float64(report)):
                    transfer[fiber, role] = report
                else:
                    storage_range_ok = False
            if math.isfinite(submitted[fiber, role]) and math.isfinite(
                transfer[fiber, role]
            ):
                try:
                    report = _float_sum_upper(
                        float(submitted[fiber, role]),
                        float(transfer[fiber, role]),
                        ledger,
                    )
                except _LocalArithmeticRangeError:
                    storage_range_ok = False
                else:
                    total[fiber, role] = report
    return submitted, transfer, total, storage_range_ok


def _unavailable_production_evidence(
    fiber_size: int,
) -> tuple[np.ndarray, ...]:
    """PRIVATE: Build the canonical fail-closed LVT.56 storage sentinel.

    Parameters
    ----------
    fiber_size : int
        Scoped transverse-fiber count.

    Returns
    -------
    evidence : tuple[np.ndarray, ...]
        Zero frozen realizations and unbounded nonnegative error reports.
    """
    return (
        np.zeros((fiber_size,), dtype=np.float64),
        np.zeros((fiber_size,), dtype=np.float64),
        np.zeros((fiber_size, 2), dtype=np.complex128),
        np.zeros((fiber_size, 2), dtype=np.complex128),
        *(
            np.full((fiber_size, 2), np.inf, dtype=np.float64)
            for _ in range(5)
        ),
        np.asarray(np.inf, dtype=np.float64),
        np.asarray(np.inf, dtype=np.float64),
        np.asarray(np.inf, dtype=np.float64),
    )


def _production_storage_range_ok(production: tuple[np.ndarray, ...]) -> bool:
    """PRIVATE: Check all LVT.56 fields against their carrier range rules.

    Parameters
    ----------
    production : tuple[np.ndarray, ...]
        Candidate production evidence tuple.

    Returns
    -------
    valid : bool
        Whether frozen points are finite normal-or-zero and reports contain
        no NaN, negative value, or subnormal component.
    """
    head_ok = all(_normal_or_zero(value) for value in production[:4])
    report_ok = all(
        not bool(jnp.any(jnp.isnan(jnp.asarray(value))))
        and not bool(jnp.any(jnp.asarray(value) < 0.0))
        and not bool(has_subnormal_components(jnp.asarray(value)))
        for value in production[4:]
    )
    return head_ok and report_ok


def _production_evidence(  # noqa: PLR0912,PLR0915
    projection: GalerkinLocalProjectionDefectCertificate,
    outer: GalerkinLocalCoordinateCauchyCurrent,
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    root_intervals: _OptionalRootIntervals,
    defining_branches: _RectanglePair,
    inner_phase_point: complex,
    outer_phase_point: complex,
    ledger: _DirectRationalLedger,
) -> tuple[np.ndarray, ...]:
    """PRIVATE: Build the concrete LVT.56 production point and error DAG.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed same-state projection parent owning radius ``B``.
    outer : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt defining-plane L7 submitted-state diagnostic.
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Strict physical per-fiber roots.
    root_intervals : _OptionalRootIntervals
        Once-hulled branch-map intervals; raw roots remain audit evidence.
    defining_branches : _RectanglePair
        Exact submitted-state defining-plane branch rectangles.
    inner_phase_point : complex
        Frozen physical carrier phase at the comparison plane.
    outer_phase_point : complex
        Frozen physical carrier phase at the defining plane.
    ledger : _DirectRationalLedger
        Active source-local exact arithmetic ledger.

    Returns
    -------
    evidence : tuple[np.ndarray, ...]
        Roots, phases, branch points, component errors/norms, and role-zero
        vector reductions expected by the branch carrier factory.

    Raises
    ------
    _LocalArithmeticRangeError
        If a frozen root realization or production point leaves normal range.
    """
    fiber_size = len(roots)
    root_points = np.zeros((fiber_size,), dtype=np.float64)
    root_errors = np.zeros((fiber_size,), dtype=np.float64)
    phases = np.empty((fiber_size, 2), dtype=np.complex128)
    phases[:, 0] = np.complex128(inner_phase_point)
    phases[:, 1] = np.complex128(outer_phase_point)
    trace = outer_phase_point * np.asarray(outer.trace_coefficients)
    normal = outer_phase_point * np.asarray(
        outer.normal_derivative_coefficients
    )
    points = np.zeros((fiber_size, 2), dtype=np.complex128)
    production_errors = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    state_errors = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    total_errors = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    point_norms = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    total_norms = np.full((fiber_size, 2), np.inf, dtype=np.float64)
    trace_norms, normal_norms = _fiber_operator_norms(outer, ledger)
    state_radius_value = float(np.asarray(projection.state_radius_upper_bound))
    state_radius = (
        ledger.retain(Fraction.from_float(state_radius_value))
        if math.isfinite(state_radius_value)
        else None
    )
    defining_values = tuple(defining_branches)
    for fiber, (root, interval) in enumerate(
        zip(roots, root_intervals, strict=True)
    ):
        if root is None or root.classification is (
            GalerkinLocalVacuumRootClass.UNCLASSIFIED
        ):
            phases[fiber] = 0.0 + 0.0j
            continue
        realization, root_error = _root_realization(root, ledger)
        root_points[fiber] = realization
        root_errors[fiber] = root_error
        if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
            quotient = normal[fiber] / (1.0j * realization)
            points[fiber, 0] = 0.5 * (trace[fiber] + quotient)
            points[fiber, 1] = 0.5 * (trace[fiber] - quotient)
        elif root.classification is GalerkinLocalVacuumRootClass.EVANESCENT:
            quotient = normal[fiber] / realization
            points[fiber, 0] = 0.5 * (trace[fiber] - quotient)
            points[fiber, 1] = 0.5 * (trace[fiber] + quotient)
        else:
            points[fiber, 0] = trace[fiber]
            points[fiber, 1] = normal[fiber]
        if not _normal_or_zero(points[fiber]):
            raise _LocalArithmeticRangeError(
                "frozen production point is outside normal complex128 range"
            )

        if state_radius is None:
            state_fractions: tuple[Fraction, Fraction] | None = None
        elif root.classification is GalerkinLocalVacuumRootClass.GRAZING:
            state_fractions = (
                ledger.multiply(trace_norms[fiber], state_radius),
                ledger.multiply(normal_norms[fiber], state_radius),
            )
        else:
            if interval is None:
                state_fractions = None
                continue
            quotient_norm = ledger.divide(normal_norms[fiber], interval[0])
            map_norm = ledger.multiply(
                _HALF, ledger.add(trace_norms[fiber], quotient_norm)
            )
            shared = ledger.multiply(map_norm, state_radius)
            state_fractions = (shared, shared)
        for role in range(2):
            point = complex(points[fiber, role])
            production = _point_to_rectangle_error_upper(
                point, defining_values[role], fiber, ledger
            )
            point_norm = _complex_point_norm_upper(point, ledger)
            production_errors[fiber, role] = production
            point_norms[fiber, role] = point_norm
            if state_fractions is not None:
                state_error = fraction_upper_float(state_fractions[role])
                total = _float_sum_upper(production, state_error, ledger)
                total_norm = _float_sum_upper(point_norm, total, ledger)
                state_errors[fiber, role] = state_error
                total_errors[fiber, role] = total
                total_norms[fiber, role] = total_norm
    if bool(np.all(np.isfinite(point_norms[:, 0]))):
        production_l2 = _complex_vector_norm_upper(points[:, 0], ledger)
    else:
        production_l2 = np.inf
    if bool(np.all(np.isfinite(total_errors[:, 0]))):
        total_error_l2 = _real_vector_norm_upper(total_errors[:, 0], ledger)
        exact_state_l2 = _float_sum_upper(
            production_l2, total_error_l2, ledger
        )
    else:
        total_error_l2 = np.inf
        exact_state_l2 = np.inf
    return (
        root_points,
        root_errors,
        phases,
        points,
        production_errors,
        state_errors,
        total_errors,
        point_norms,
        total_norms,
        np.asarray(production_l2, dtype=np.float64),
        np.asarray(total_error_l2, dtype=np.float64),
        np.asarray(exact_state_l2, dtype=np.float64),
    )


def _half_space_dispositions(
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...],
    propagators: tuple[GalerkinLocalVacuumPropagator | None, ...],
    defining_secondary: GalerkinLocalTerminalComplexRectangles,
) -> tuple[GalerkinLocalVacuumHalfSpaceDisposition, ...]:
    """PRIVATE: Classify excluded branch rectangles against exact zero.

    Parameters
    ----------
    roots : tuple[GalerkinLocalVacuumRootCertificate | None, ...]
        Strict physical roots.
    propagators : tuple[GalerkinLocalVacuumPropagator | None, ...]
        Successfully enclosed homogeneous propagators.
    defining_secondary : GalerkinLocalTerminalComplexRectangles
        Inward, growing, or grazing-derivative defining rectangles.

    Returns
    -------
    dispositions : tuple[GalerkinLocalVacuumHalfSpaceDisposition, ...]
        Exact-zero, provably-nonzero, unresolved, or unclassified statuses.
    """
    values = tuple(np.asarray(value) for value in defining_secondary)
    dispositions: list[GalerkinLocalVacuumHalfSpaceDisposition] = []
    prefixes = {
        GalerkinLocalVacuumRootClass.PROPAGATING: "PROPAGATING_INWARD",
        GalerkinLocalVacuumRootClass.EVANESCENT: "EVANESCENT_GROWING",
        GalerkinLocalVacuumRootClass.GRAZING: "GRAZING_DERIVATIVE",
    }
    for fiber, (root, propagator) in enumerate(
        zip(roots, propagators, strict=True)
    ):
        if (
            root is None
            or propagator is None
            or root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED
        ):
            dispositions.append(
                GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED
            )
            continue
        endpoints = tuple(float(value[fiber]) for value in values)
        if all(value == 0.0 for value in endpoints):
            suffix = "EXACT_ZERO"
        elif (
            endpoints[0] > 0.0
            or endpoints[1] < 0.0
            or endpoints[2] > 0.0
            or endpoints[3] < 0.0
        ):
            suffix = "PROVABLY_NONZERO"
        else:
            suffix = "UNRESOLVED"
        dispositions.append(
            GalerkinLocalVacuumHalfSpaceDisposition[
                f"{prefixes[root.classification]}_{suffix}"
            ]
        )
    return tuple(dispositions)


def _cut_balance(  # noqa: PLR0912,PLR0915
    projection: GalerkinLocalProjectionDefectCertificate,
    inner: GalerkinLocalCoordinateCauchyCurrent,
    outer: GalerkinLocalCoordinateCauchyCurrent,
    maximum_direct_pairs: int,
    maximum_rational_bits: int,
) -> GalerkinLocalVacuumCutBalance:
    """PRIVATE: Build the independent nonsymmetrized current/work balance.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed projection parent owning exact ``G`` and ``d``.
    inner : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt inner-plane exact-target current diagnostic.
    outer : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt outer-plane exact-target current diagnostic.
    maximum_direct_pairs : int
        Independent cut pair-count policy.
    maximum_rational_bits : int
        Independent cut exact-rational endpoint bit policy.

    Returns
    -------
    balance : GalerkinLocalVacuumCutBalance
        Typed cut-balance evidence with separate rational transcript.

    Raises
    ------
    EntireEnclosureError
        If direct exact arithmetic reports an unexpected non-size failure.
    """
    rows = np.asarray(projection.state_to_fiber_rows, dtype=np.int64)
    selected = np.asarray(projection.selected_state_mask, dtype=np.bool_)
    pair_count = sum(
        int(np.count_nonzero(selected & (rows == row))) ** 2
        for row in range(projection.scope_transverse_indices.shape[0])
    )
    count_overflow = pair_count > _MAXIMUM_SIGNED_INT64
    host_ok = host_binary64_supported()
    environment_ok = bool(all_normal_arithmetic_supported())
    pair_preflight_ok = (
        not count_overflow and pair_count <= maximum_direct_pairs
    )
    diagnostic_ready = bool(inner.current_diagnostic_eligible) and bool(
        outer.current_diagnostic_eligible
    )
    current_ranges_ok = all(
        _normal_or_zero(value)
        for value in (
            inner.exact_reduced_current_lower_bound,
            inner.exact_reduced_current_upper_bound,
            outer.exact_reduced_current_lower_bound,
            outer.exact_reduced_current_upper_bound,
        )
    )
    gram_ranges_ok = all(
        _normal_or_zero(value)
        for value in (
            projection.exact_free_diagonal_lower_bounds,
            projection.exact_free_diagonal_upper_bounds,
            projection.stability_result.solve_result.field,
            projection.gram_real_lower_bounds,
            projection.gram_real_upper_bounds,
            projection.gram_imag_lower_bounds,
            projection.gram_imag_upper_bounds,
        )
    )
    projection_failure = GalerkinLocalProjectionDefectFailure(
        int(np.asarray(projection.failure_mask))
    )
    gram_fatal = (
        GalerkinLocalProjectionDefectFailure.PARENT_SOURCE_EVIDENCE_MISMATCH
        | GalerkinLocalProjectionDefectFailure.TERMINAL_SCOPE_INCOMPLETE
        | GalerkinLocalProjectionDefectFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalProjectionDefectFailure.GRAM_PAIR_BUDGET_EXCEEDED
        | GalerkinLocalProjectionDefectFailure.GRAM_PAIR_COUNT_OVERFLOW
        | GalerkinLocalProjectionDefectFailure.ROOT_ENCLOSURE_FAILURE
        | GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
    )
    projection_work_ready = not bool(projection_failure & gram_fatal)
    current_input_ok = (
        host_ok and environment_ok and diagnostic_ready and current_ranges_ok
    )
    work_input_ok = (
        host_ok and environment_ok and projection_work_ready and gram_ranges_ok
    )
    ledger = _DirectRationalLedger(maximum_rational_bits)
    rational_failure: EntireEnclosureFailure | None = None
    current_interval: _RealInterval | None = None
    work_interval: _RealInterval | None = None
    try:
        if current_input_ok:
            current_interval = (
                ledger.subtract(
                    Fraction.from_float(
                        float(
                            np.asarray(outer.exact_reduced_current_lower_bound)
                        )
                    ),
                    Fraction.from_float(
                        float(
                            np.asarray(inner.exact_reduced_current_upper_bound)
                        )
                    ),
                ),
                ledger.subtract(
                    Fraction.from_float(
                        float(
                            np.asarray(outer.exact_reduced_current_upper_bound)
                        )
                    ),
                    Fraction.from_float(
                        float(
                            np.asarray(inner.exact_reduced_current_lower_bound)
                        )
                    ),
                ),
            )
        if work_input_ok and pair_preflight_ok:
            field = np.asarray(
                projection.stability_result.solve_result.field,
                dtype=np.complex128,
            )
            gram_columns = tuple(
                np.asarray(value)
                for value in (
                    projection.gram_real_lower_bounds,
                    projection.gram_real_upper_bounds,
                    projection.gram_imag_lower_bounds,
                    projection.gram_imag_upper_bounds,
                )
            )
            free_lower = np.asarray(
                projection.exact_free_diagonal_lower_bounds
            )
            free_upper = np.asarray(
                projection.exact_free_diagonal_upper_bounds
            )
            terms: list[ComplexRectangle] = []
            for left in np.flatnonzero(selected):
                for right in np.flatnonzero(selected & (rows == rows[left])):
                    gram: ComplexRectangle = tuple(
                        Fraction.from_float(float(column[left, right]))
                        for column in gram_columns
                    )  # type: ignore[assignment]
                    free = (
                        Fraction.from_float(float(free_lower[right])),
                        Fraction.from_float(float(free_upper[right])),
                    )
                    left_point = _point_rectangle(complex(field[left]))
                    term = ledger.rectangle_multiply(
                        (
                            left_point[0],
                            left_point[1],
                            -left_point[3],
                            -left_point[2],
                        ),
                        gram,
                    )
                    term = ledger.rectangle_multiply(
                        term, _rectangle_real_interval(free)
                    )
                    term = ledger.rectangle_multiply(
                        term, _point_rectangle(complex(field[right]))
                    )
                    terms.append(term)
            work = _rectangle_sum(terms, ledger)
            work_interval = (-work[3], -work[2])
    except EntireEnclosureError as error:
        if error.failure is not EntireEnclosureFailure.RATIONAL_SIZE_LIMIT:
            raise
        rational_failure = error.failure

    if current_interval is None:
        current_reports = (np.float64(-np.inf), np.float64(np.inf))
    else:
        current_reports = (
            np.float64(fraction_lower_float(current_interval[0])),
            np.float64(fraction_upper_float(current_interval[1])),
        )
    if work_interval is None:
        work_reports = (np.float64(-np.inf), np.float64(np.inf))
    else:
        work_reports = (
            np.float64(fraction_lower_float(work_interval[0])),
            np.float64(fraction_upper_float(work_interval[1])),
        )
    current_output_ok = current_interval is None or all(
        _normal_or_zero(value) for value in current_reports
    )
    work_output_ok = work_interval is None or all(
        _normal_or_zero(value) for value in work_reports
    )
    if current_interval is not None and not current_output_ok:
        current_reports = (np.float64(-np.inf), np.float64(np.inf))
    if work_interval is not None and not work_output_ok:
        work_reports = (np.float64(-np.inf), np.float64(np.inf))
    current_route_available = (
        current_interval is not None and current_output_ok
    )
    work_route_available = work_interval is not None and work_output_ok
    both_routes_available = current_route_available and work_route_available
    intersection_lower = max(current_reports[0], work_reports[0])
    intersection_upper = min(current_reports[1], work_reports[1])
    overlaps = bool(
        both_routes_available and intersection_lower <= intersection_upper
    )
    disjoint = bool(
        both_routes_available and intersection_lower > intersection_upper
    )
    if not overlaps:
        intersection_lower = np.float64(-np.inf)
        intersection_upper = np.float64(np.inf)
    reports = (
        *current_reports,
        *work_reports,
        np.float64(intersection_lower),
        np.float64(intersection_upper),
    )
    normal_ok = (
        environment_ok
        and (not diagnostic_ready or current_ranges_ok)
        and (not projection_work_ready or gram_ranges_ok)
        and (current_interval is None or current_output_ok)
        and (work_interval is None or work_output_ok)
    )
    failure = GalerkinLocalVacuumTerminalFailure.NONE
    if not diagnostic_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE  # noqa: E501
    if not projection_work_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
    if disjoint:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
        )
    if not host_ok:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
    if not normal_ok:
        failure |= GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    if pair_count > maximum_direct_pairs:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        )
    if count_overflow:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        )
    if rational_failure is not None:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
        )
    fatal = (
        GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    eligible = not bool(failure & fatal)
    scope = (
        "literal full-state complete-fiber cut"
        if projection.projection_scope
        is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
        else "literal selected-preterminal complete-fiber cut"
    )
    digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.cut.v1",
            "projection_identity": projection.projection_identity_digest,
            "projection_evidence": projection.certificate_digest,
            "inner_current": inner.diagnostic_evidence_digest,
            "outer_current": outer.diagnostic_evidence_digest,
            "reports": stored_value_payload(
                tuple(np.asarray(v) for v in reports)
            ),
            "pair_count": pair_count,
            "maximum_direct_pairs": maximum_direct_pairs,
            "maximum_rational_bits": maximum_rational_bits,
            "rational_peak_bits": ledger.peak_bits,
            "rational_work": ledger.exact_work_count,
            "rational_failure": stored_value_payload(rational_failure),
            "current_available": current_route_available,
            "work_available": work_route_available,
            "routes_disjoint": disjoint,
            "failure": int(failure),
            "scope": scope,
        }
    )
    balance = _make_local_vacuum_cut_balance(
        tuple(jnp.asarray(value, dtype=jnp.float64) for value in reports),
        (
            jnp.asarray(0 if count_overflow else pair_count, dtype=jnp.int64),
            jnp.asarray(maximum_direct_pairs, dtype=jnp.int64),
        ),
        (
            jnp.asarray(host_ok),
            jnp.asarray(normal_ok),
            jnp.asarray(eligible),
        ),
        jnp.asarray(int(failure), dtype=jnp.int64),
        direct_work_count_exact=str(pair_count),
        maximum_rational_bits=maximum_rational_bits,
        direct_rational_peak_bits=ledger.peak_bits,
        direct_rational_work_count_exact=str(ledger.exact_work_count),
        direct_rational_failure=rational_failure,
        direct_work_formula="sum_h |I_scope(h)|^2 literal G*diag(d) pairs",
        current_difference_formula=_CURRENT_DIFFERENCE_FORMULA,
        defect_work_formula=_DEFECT_WORK_FORMULA,
        balance_scope=scope,
        cut_balance_digest=digest,
    )
    return balance  # noqa: RET504


def _branch_evidence(  # noqa: PLR0912,PLR0913,PLR0915
    projection: GalerkinLocalProjectionDefectCertificate,
    inner: GalerkinLocalCoordinateCauchyCurrent,
    outer: GalerkinLocalCoordinateCauchyCurrent,
    inner_coordinate: np.float64,
    outer_coordinate: np.float64,
    maximum_direct_terms: int,
    maximum_root_work: int,
    entire_policies: _EntirePolicies,
    maximum_interval_work: int,
    maximum_rational_bits: int,
) -> GalerkinLocalVacuumBranchEvidence:
    """PRIVATE: Compose roots, two mismatch routes, and LVT.56 evidence.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed projection parent.
    inner : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt inner-plane L7 diagnostic.
    outer : GalerkinLocalCoordinateCauchyCurrent
        Internally rebuilt defining outer-plane L7 diagnostic.
    inner_coordinate : np.float64
        Exact stored physical comparison-plane coordinate.
    outer_coordinate : np.float64
        Exact stored physical defining-plane coordinate.
    maximum_direct_terms : int
        Independent linear branch direct-term policy.
    maximum_root_work : int
        Independent per-fiber root-helper work policy.
    entire_policies : _EntirePolicies
        Precision, term, work, range, and rational-bit helper policies.
    maximum_interval_work : int
        Independent propagator post-helper work policy.
    maximum_rational_bits : int
        Independent source-local and helper rational-size policy.

    Returns
    -------
    evidence : GalerkinLocalVacuumBranchEvidence
        Typed branch, plane-mismatch, and production-amplitude evidence.

    Raises
    ------
    EntireEnclosureError
        If direct exact arithmetic reports an unexpected non-size failure.
    """
    target = projection.zero_slab_certificate.represented_source_certificate.source.target  # noqa: E501
    fiber_size = projection.scope_transverse_indices.shape[0]
    selected = np.asarray(projection.selected_state_mask, dtype=np.bool_)
    selected_size = int(np.count_nonzero(selected))
    direct_count = 6 * selected_size + 32 * fiber_size
    count_overflow = direct_count > _MAXIMUM_SIGNED_INT64
    budget_ok = direct_count <= maximum_direct_terms
    host_ok = host_binary64_supported()
    environment_ok = bool(all_normal_arithmetic_supported())
    computation_allowed = (
        host_ok and environment_ok and not count_overflow and budget_ok
    )
    projection_ready = bool(projection.finite_projection_bound_eligible)
    diagnostic_ready = bool(inner.current_diagnostic_eligible) and bool(
        outer.current_diagnostic_eligible
    )
    direct = _DirectRationalLedger(maximum_rational_bits)
    hull = _OutwardDyadicHullLedger(maximum_rational_bits)
    recorder = _EntireRecorder(entire_policies, hull)
    rational_failure: EntireEnclosureFailure | None = None
    roots: tuple[GalerkinLocalVacuumRootCertificate | None, ...] = (
        None,
    ) * fiber_size
    propagators: tuple[GalerkinLocalVacuumPropagator | None, ...] = (
        None,
    ) * fiber_size
    support_rows: tuple[int | None, ...] = (None,) * fiber_size
    root_failures: tuple[
        EntireEnclosureFailure | GalerkinLocalVacuumPropagationFailure | None,
        ...,
    ] = (None,) * fiber_size
    root_failure_work: tuple[int, ...] = (0,) * fiber_size
    propagator_failures: tuple[
        EntireEnclosureFailure | GalerkinLocalVacuumPropagationFailure | None,
        ...,
    ] = (None,) * fiber_size
    propagator_failure_work: tuple[int, ...] = (0,) * fiber_size
    root_intervals: _OptionalRootIntervals = (None,) * fiber_size
    propagator_entries: _OptionalPropagatorEntries = (None,) * fiber_size
    q_intervals: list[_RealInterval] = []
    oriented_normal_intervals: list[_RealInterval] = []
    none_values: list[ComplexRectangle | None] = [None] * fiber_size
    inner_rational: tuple[
        list[ComplexRectangle | None], list[ComplexRectangle | None]
    ] = (list(none_values), list(none_values))
    outer_rational: tuple[
        list[ComplexRectangle | None], list[ComplexRectangle | None]
    ] = (list(none_values), list(none_values))
    endpoint_rational: tuple[
        list[ComplexRectangle | None], list[ComplexRectangle | None]
    ] = (list(none_values), list(none_values))
    forced_rational: tuple[
        list[ComplexRectangle | None], list[ComplexRectangle | None]
    ] = (list(none_values), list(none_values))
    defining_rational: tuple[
        list[ComplexRectangle | None], list[ComplexRectangle | None]
    ] = (list(none_values), list(none_values))
    endpoint_branches_rational = defining_rational
    forced_branches_rational = defining_rational
    factors: list[tuple[Fraction | None, Fraction | None]] = [
        (None, None) for _ in range(fiber_size)
    ]
    mismatch_bounds: tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(
        np.full((fiber_size, 2), np.inf, dtype=np.float64) for _ in range(3)
    )  # type: ignore[assignment]
    mismatch_storage_range_ok = True
    production = _unavailable_production_evidence(fiber_size)
    production_range_failure = False

    inner_fraction = Fraction.from_float(float(inner_coordinate))
    outer_fraction = Fraction.from_float(float(outer_coordinate))
    distance = _ZERO
    if computation_allowed:
        try:
            inner_fraction = direct.retain(inner_fraction)
            outer_fraction = direct.retain(outer_fraction)
            distance = direct.retain(
                abs(direct.subtract(outer_fraction, inner_fraction))
            )
            q_intervals = _physical_q_intervals(
                target,
                np.asarray(projection.scope_transverse_indices),
                direct,
            )
            oriented_normal_intervals = _oriented_normal_intervals(
                target, direct
            )
            (
                roots,
                propagators,
                support_rows,
                root_failures,
                root_failure_work,
                propagator_failures,
                propagator_failure_work,
            ) = _classify_physical_roots(
                projection,
                q_intervals,
                distance,
                maximum_root_work,
                entire_policies,
                maximum_interval_work,
                maximum_rational_bits,
                direct,
                oriented_normal_intervals,
            )
            root_intervals = _hull_branch_root_intervals(roots, hull)
            propagator_entries = _hull_branch_propagator_entries(
                propagators, hull
            )
            inner_phase = _carrier_phase_rectangle(
                target,
                inner_fraction,
                "physical_cauchy.inner_carrier_phase.exp",
                recorder,
                direct,
            )
            outer_phase = _carrier_phase_rectangle(
                target,
                outer_fraction,
                "physical_cauchy.outer_carrier_phase.exp",
                recorder,
                direct,
            )
            if diagnostic_ready and inner_phase is not None:
                physical = _physical_cauchy_rectangles(
                    inner, inner_phase, direct
                )
                inner_rational = (list(physical[0]), list(physical[1]))
            if diagnostic_ready and outer_phase is not None:
                physical = _physical_cauchy_rectangles(
                    outer, outer_phase, direct
                )
                outer_rational = (list(physical[0]), list(physical[1]))
            if all(
                value is not None
                for value in (
                    *inner_rational[0],
                    *inner_rational[1],
                    *outer_rational[0],
                    *outer_rational[1],
                )
            ):
                inner_exact = (
                    [
                        value
                        for value in inner_rational[0]
                        if value is not None
                    ],
                    [
                        value
                        for value in inner_rational[1]
                        if value is not None
                    ],
                )
                outer_exact = (
                    [
                        value
                        for value in outer_rational[0]
                        if value is not None
                    ],
                    [
                        value
                        for value in outer_rational[1]
                        if value is not None
                    ],
                )
                endpoint_rational = _endpoint_cauchy_mismatch(
                    inner_exact, outer_exact, propagator_entries, direct
                )
            forced_rational = _forced_cauchy_rectangles(
                projection,
                roots,
                root_intervals,
                inner_fraction,
                distance,
                recorder,
                direct,
                oriented_normal_intervals,
            )
            defining_rational = _branch_mismatch_rectangles(
                outer_rational, roots, root_intervals, direct
            )
            endpoint_branches_rational = _branch_mismatch_rectangles(
                endpoint_rational, roots, root_intervals, direct
            )
            forced_branches_rational = _branch_mismatch_rectangles(
                forced_rational, roots, root_intervals, direct
            )
            factors = _branch_mismatch_factors(
                roots, root_intervals, distance, recorder, direct
            )
            (
                submitted_mismatch,
                transfer_mismatch,
                total_mismatch,
                mismatch_storage_range_ok,
            ) = _plane_mismatch_bounds(projection, factors, direct)
            mismatch_bounds = (
                submitted_mismatch,
                transfer_mismatch,
                total_mismatch,
            )
        except EntireEnclosureError as error:
            if error.failure is not EntireEnclosureFailure.RATIONAL_SIZE_LIMIT:
                raise
            rational_failure = error.failure

    inner_pair, _, inner_storage_ok = _optional_rectangle_pair(inner_rational)
    outer_pair, _, outer_storage_ok = _optional_rectangle_pair(outer_rational)
    endpoint_pair, endpoint_available, endpoint_storage_ok = (
        _optional_rectangle_pair(endpoint_rational)
    )
    forced_pair, forced_available, forced_storage_ok = (
        _optional_rectangle_pair(forced_rational)
    )
    certified_cauchy, cauchy_mask = _intersect_rectangle_pairs(
        endpoint_pair,
        forced_pair,
        endpoint_available,
        forced_available,
    )
    cauchy_disjoint = bool(
        np.any(endpoint_available & forced_available & ~cauchy_mask)
    )
    defining_pair, defining_available, defining_storage_range_ok = (
        _optional_rectangle_pair(defining_rational)
    )
    (
        endpoint_branches,
        endpoint_branch_available,
        endpoint_branch_storage_ok,
    ) = _optional_rectangle_pair(endpoint_branches_rational)
    (
        forced_branches,
        forced_branch_available,
        forced_branch_storage_ok,
    ) = _optional_rectangle_pair(forced_branches_rational)
    certified_branches, branch_mask = _intersect_rectangle_pairs(
        endpoint_branches,
        forced_branches,
        endpoint_branch_available,
        forced_branch_available,
    )
    branch_disjoint = bool(
        np.any(
            endpoint_branch_available & forced_branch_available & ~branch_mask
        )
    )
    defining_storage_ok = (
        defining_storage_range_ok
        and bool(np.all(defining_available))
        and all(
            _normal_or_zero(value) for role in defining_pair for value in role
        )
    )
    production_attempted = (
        computation_allowed
        and diagnostic_ready
        and rational_failure is None
        and defining_storage_ok
    )
    production_completed = False
    if production_attempted:
        try:
            inner_phase_point = _physical_phase_point(
                target, float(inner_coordinate)
            )
            outer_phase_point = _physical_phase_point(
                target, float(outer_coordinate)
            )
            production = _production_evidence(
                projection,
                outer,
                roots,
                root_intervals,
                defining_pair,
                inner_phase_point,
                outer_phase_point,
                direct,
            )
            if _production_storage_range_ok(production):
                production_completed = True
            else:
                production_range_failure = True
                production = _unavailable_production_evidence(fiber_size)
        except _LocalArithmeticRangeError:
            production_range_failure = True
        except EntireEnclosureError as error:
            if error.failure is not EntireEnclosureFailure.RATIONAL_SIZE_LIMIT:
                raise
            rational_failure = error.failure

    entire = recorder.evidence()
    failure = GalerkinLocalVacuumTerminalFailure.NONE
    if not projection_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
    if not diagnostic_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE  # noqa: E501
    roots_classified = all(
        root is not None
        and root.classification
        is not GalerkinLocalVacuumRootClass.UNCLASSIFIED
        for root in roots
    )
    if not roots_classified:
        failure |= GalerkinLocalVacuumTerminalFailure.ROOT_UNCLASSIFIED
    if any(
        reason is not None for reason in (*root_failures, *propagator_failures)
    ):
        failure |= GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
    if cauchy_disjoint:
        failure |= GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
    if branch_disjoint:
        failure |= GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
    if bool(entire.helper_attempted) and not bool(entire.helper_eligible):
        failure |= (
            GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
        )
    if not host_ok:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
    if direct_count > maximum_direct_terms:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        )
    if count_overflow:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        )
    if rational_failure is not None:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
        )

    rectangle_range_ok = all(
        (
            inner_storage_ok,
            outer_storage_ok,
            endpoint_storage_ok,
            forced_storage_ok,
            defining_storage_range_ok,
            endpoint_branch_storage_ok,
            forced_branch_storage_ok,
        )
    )
    independent_reports = [
        production[0],
        production[1],
        production[2],
        production[3],
        production[4],
        production[7],
        production[9],
    ]
    production_range_ok = not production_range_failure and (
        not production_completed
        or all(_normal_or_zero(value) for value in independent_reports)
    )
    state_radius_available = math.isfinite(
        float(np.asarray(projection.state_radius_upper_bound))
    )
    state_reports_ok = (
        not production_completed
        or not state_radius_available
        or all(
            _normal_or_zero(value)
            for value in (*production[5:7], *production[8:])
        )
    )
    measured = np.asarray(projection.measured_defect_upper_bounds)
    transferred = np.asarray(projection.state_error_transfer_upper_bounds)
    factor_available = np.asarray(
        [
            [factor is not None for factor in role_factors]
            for role_factors in factors
        ],
        dtype=np.bool_,
    )
    submitted_available = factor_available & np.isfinite(measured[:, None])
    transfer_available = factor_available & np.isfinite(transferred[:, None])
    total_available = submitted_available & transfer_available
    mismatch_range_ok = mismatch_storage_range_ok and all(
        _normal_or_zero(values[available])
        for values, available in zip(
            mismatch_bounds,
            (submitted_available, transfer_available, total_available),
            strict=True,
        )
    )
    normal_ok = (
        environment_ok
        and not hull.range_failure
        and rectangle_range_ok
        and production_range_ok
        and state_reports_ok
        and mismatch_range_ok
    )
    if not normal_ok:
        failure |= GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    fatal = (
        GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
        | GalerkinLocalVacuumTerminalFailure.ROOT_UNCLASSIFIED
        | GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
        | GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
        | GalerkinLocalVacuumTerminalFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
        | GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
        | GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    eligible = (
        not bool(failure & fatal)
        and bool(np.all(cauchy_mask))
        and bool(np.all(branch_mask))
    )
    dispositions = _half_space_dispositions(
        roots, propagators, defining_pair[1]
    )
    hull_digest = hull.evidence_digest()
    rational_peak_bits = max(direct.peak_bits, hull.input_peak_bits)
    helper_policy_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.helper_policy.v1",
            "root_work": maximum_root_work,
            "entire": stored_value_payload(entire_policies),
            "interval_work": maximum_interval_work,
            "branch_terms": maximum_direct_terms,
            "rational_bits": maximum_rational_bits,
            "hull_algorithm": _HULL_ALGORITHM,
            "hull": hull_digest,
        }
    )
    physical_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.physical_root.v1",
            "target": target.target_digest,
            "target_evidence": target.manifest_evidence_digest,
            "scope": projection.projection_scope.value,
            "fibers": stored_value_payload(
                projection.scope_transverse_indices
            ),
            "q": tuple(_interval_payload(value) for value in q_intervals),
            "support_rows": stored_value_payload(support_rows),
            "support_free": stored_value_payload(
                (
                    projection.exact_free_diagonal_lower_bounds,
                    projection.exact_free_diagonal_upper_bounds,
                )
            ),
            "support_normal": tuple(
                _interval_payload(value) for value in oriented_normal_intervals
            ),
            "root_identity": tuple(
                None if root is None else root.root_identity_digest
                for root in roots
            ),
            "root_evidence": tuple(
                None if root is None else root.root_evidence_digest
                for root in roots
            ),
            "zero_witness": tuple(
                None
                if root is None or root.zero_witness is None
                else root.zero_witness.witness_digest
                for root in roots
            ),
            "root_failures": stored_value_payload(root_failures),
            "root_failure_work": stored_value_payload(root_failure_work),
            "propagator_failures": stored_value_payload(propagator_failures),
            "propagator_failure_work": stored_value_payload(
                propagator_failure_work
            ),
        }
    )
    cauchy_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.cauchy.v1",
            "inner": stored_value_payload(inner_pair),
            "outer": stored_value_payload(outer_pair),
            "endpoint": stored_value_payload(endpoint_pair),
            "forced": stored_value_payload(forced_pair),
            "certified": stored_value_payload(certified_cauchy),
            "mask": stored_value_payload(jnp.asarray(cauchy_mask)),
            "endpoint_available": stored_value_payload(endpoint_available),
            "forced_available": stored_value_payload(forced_available),
            "disjoint": cauchy_disjoint,
            "helper": entire.helper_evidence_digest,
            "hull": hull_digest,
        }
    )
    stored_mismatch_bounds: _PlaneMismatchBounds = (
        jnp.asarray(mismatch_bounds[0]),
        jnp.asarray(mismatch_bounds[1]),
        jnp.asarray(mismatch_bounds[2]),
    )
    stored_production: _ProductionEvidence = (
        jnp.asarray(production[0]),
        jnp.asarray(production[1]),
        jnp.asarray(production[2]),
        jnp.asarray(production[3]),
        jnp.asarray(production[4]),
        jnp.asarray(production[5]),
        jnp.asarray(production[6]),
        jnp.asarray(production[7]),
        jnp.asarray(production[8]),
        jnp.asarray(production[9]),
        jnp.asarray(production[10]),
        jnp.asarray(production[11]),
    )
    branch_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.branch.v1",
            "projection": projection.certificate_digest,
            "physical_root": physical_digest,
            "root_evidence": tuple(
                None if root is None else root.root_evidence_digest
                for root in roots
            ),
            "propagators": tuple(
                None if value is None else value.propagator_evidence_digest
                for value in propagators
            ),
            "cauchy": cauchy_digest,
            "defining": stored_value_payload(defining_pair),
            "endpoint_branch": stored_value_payload(endpoint_branches),
            "forced_branch": stored_value_payload(forced_branches),
            "certified_branch": stored_value_payload(certified_branches),
            "branch_mask": stored_value_payload(jnp.asarray(branch_mask)),
            "endpoint_branch_available": stored_value_payload(
                endpoint_branch_available
            ),
            "forced_branch_available": stored_value_payload(
                forced_branch_available
            ),
            "branch_disjoint": branch_disjoint,
            "mismatch_bounds": stored_value_payload(stored_mismatch_bounds),
            "production": stored_value_payload(stored_production),
            "dispositions": stored_value_payload(dispositions),
            "direct_count": direct_count,
            "maximum_direct_terms": maximum_direct_terms,
            "maximum_root_work": maximum_root_work,
            "maximum_interval_work": maximum_interval_work,
            "maximum_rational_bits": maximum_rational_bits,
            "direct_rational_peak": rational_peak_bits,
            "direct_rational_work": direct.exact_work_count,
            "direct_rational_failure": stored_value_payload(rational_failure),
            "hull_algorithm": _HULL_ALGORITHM,
            "hull_attempted_endpoints": hull.attempted_endpoint_count,
            "hull_completed_endpoints": hull.completed_endpoint_count,
            "hull_input_peak_bits": hull.input_peak_bits,
            "hull_output_peak_bits": hull.output_peak_bits,
            "hull_normal_floor_count": hull.normal_floor_count,
            "hull_range_failure": hull.range_failure,
            "hull": hull_digest,
            "failure": int(failure),
            "helper_policy": helper_policy_digest,
        }
    )
    evidence = _make_local_vacuum_branch_evidence(
        projection,
        roots,
        propagators,
        root_failures,
        root_failure_work,
        propagator_failures,
        propagator_failure_work,
        entire,
        (
            inner_pair,
            outer_pair,
            endpoint_pair,
            forced_pair,
            certified_cauchy,
        ),
        (
            defining_pair,
            endpoint_branches,
            forced_branches,
            certified_branches,
        ),
        stored_mismatch_bounds,
        stored_production,
        (jnp.asarray(cauchy_mask), jnp.asarray(branch_mask)),
        (
            jnp.asarray(
                0 if count_overflow else direct_count, dtype=jnp.int64
            ),
            jnp.asarray(maximum_direct_terms, dtype=jnp.int64),
        ),
        (
            jnp.asarray(host_ok),
            jnp.asarray(normal_ok),
            jnp.asarray(eligible),
        ),
        jnp.asarray(int(failure), dtype=jnp.int64),
        half_space_dispositions=dispositions,
        direct_work_count_exact=str(direct_count),
        maximum_root_work=maximum_root_work,
        maximum_propagator_interval_work=maximum_interval_work,
        maximum_rational_bits=maximum_rational_bits,
        direct_rational_peak_bits=rational_peak_bits,
        direct_rational_work_count_exact=str(direct.exact_work_count),
        direct_rational_failure=rational_failure,
        hull_algorithm=_HULL_ALGORITHM,
        hull_attempted_endpoint_count=hull.attempted_endpoint_count,
        hull_completed_endpoint_count=hull.completed_endpoint_count,
        hull_input_peak_bits=hull.input_peak_bits,
        hull_output_peak_bits=hull.output_peak_bits,
        hull_normal_floor_count=hull.normal_floor_count,
        hull_range_failure=hull.range_failure,
        hull_evidence_digest=hull_digest,
        direct_work_formula=_DIRECT_WORK_FORMULA,
        physical_root_formula=_PHYSICAL_ROOT_FORMULA,
        root_realization_formula=_ROOT_REALIZATION_FORMULA,
        physical_cauchy_formula=_PHYSICAL_CAUCHY_FORMULA,
        endpoint_mismatch_formula=_ENDPOINT_MISMATCH_FORMULA,
        forced_mismatch_formula=_FORCED_MISMATCH_FORMULA,
        plane_mismatch_bound_formula=_PLANE_MISMATCH_BOUND_FORMULA,
        amplitude_error_formula=_AMPLITUDE_ERROR_FORMULA,
        amplitude_norm_formula=_AMPLITUDE_NORM_FORMULA,
        helper_policy_digest=helper_policy_digest,
        physical_root_identity_digest=physical_digest,
        cauchy_evidence_digest=cauchy_digest,
        branch_evidence_digest=branch_digest,
    )
    return evidence  # noqa: RET504


def _certify_prepared_terminal(  # noqa: PLR0912,PLR0913,PLR0915
    projection: GalerkinLocalProjectionDefectCertificate,
    disposition: GalerkinLocalVacuumTerminalDisposition,
    maximum_terminal_direct_pairs: int,
    maximum_branch_direct_terms: int,
    maximum_cut_direct_pairs: int,
    maximum_root_work: int,
    entire_policies: _EntirePolicies,
    maximum_interval_work: int,
    maximum_rational_bits: int,
) -> GalerkinLocalVacuumTerminalCertificate:
    """PRIVATE: Compose one terminal after exactly one projection replay.

    Parameters
    ----------
    projection : GalerkinLocalProjectionDefectCertificate
        Fully replayed projection parent.
    disposition : GalerkinLocalVacuumTerminalDisposition
        Explicit plane-defined or exact-native continuation claim.
    maximum_terminal_direct_pairs : int
        Shared independent L7 operator/action/current direct-work policy.
    maximum_branch_direct_terms : int
        Independent linear branch direct-term policy.
    maximum_cut_direct_pairs : int
        Independent nonsymmetrized cut pair policy.
    maximum_root_work : int
        Independent per-fiber root-helper work policy.
    entire_policies : _EntirePolicies
        Precision, term, work, range, and rational-bit helper policies.
    maximum_interval_work : int
        Independent propagator post-helper work policy.
    maximum_rational_bits : int
        Independent source-local/root/helper rational-size policy.

    Returns
    -------
    certificate : GalerkinLocalVacuumTerminalCertificate
        Canonical composed local vacuum-terminal certificate.
    """
    zero_slab = projection.zero_slab_certificate
    target = zero_slab.represented_source_certificate.source.target
    state = np.asarray(
        projection.stability_result.solve_result.field, dtype=np.complex128
    )
    side = target.acquisition.terminal_side
    lower = np.float64(np.asarray(zero_slab.slab_lower_coordinate))
    upper = np.float64(np.asarray(zero_slab.slab_upper_coordinate))
    outer_coordinate, inner_coordinate = (
        (upper, lower)
        if side is GalerkinTerminalSide.POSITIVE
        else (lower, upper)
    )
    scope = projection.projection_scope
    inner_operator = _certify_prepared_operator(
        target,
        inner_coordinate,
        scope,
        maximum_terminal_direct_pairs,
    )
    outer_operator = _certify_prepared_operator(
        target,
        outer_coordinate,
        scope,
        maximum_terminal_direct_pairs,
    )
    inner_prepared = _make_prepared_local_current_operator(inner_operator)
    outer_prepared = _make_prepared_local_current_operator(outer_operator)
    inner = _enclose_current_prepared(
        inner_prepared, state, maximum_terminal_direct_pairs
    )
    outer = _enclose_current_prepared(
        outer_prepared, state, maximum_terminal_direct_pairs
    )
    branch = _branch_evidence(
        projection,
        inner,
        outer,
        inner_coordinate,
        outer_coordinate,
        maximum_branch_direct_terms,
        maximum_root_work,
        entire_policies,
        maximum_interval_work,
        maximum_rational_bits,
    )
    cut = _cut_balance(
        projection,
        inner,
        outer,
        maximum_cut_direct_pairs,
        maximum_rational_bits,
    )
    diagnostic_ready = bool(inner.current_diagnostic_eligible) and bool(
        outer.current_diagnostic_eligible
    )
    operator_ready = bool(inner_operator.current_operator_eligible) and bool(
        outer_operator.current_operator_eligible
    )
    action_ready = bool(
        inner.action_enclosure.current_action_eligible
    ) and bool(outer.action_enclosure.current_action_eligible)
    zero_ready = bool(zero_slab.terminal_zero_slab_eligible)
    projection_ready = bool(projection.finite_projection_bound_eligible)
    failure = GalerkinLocalVacuumTerminalFailure(
        int(np.asarray(branch.failure_mask))
        | int(np.asarray(cut.failure_mask))
    )
    if not zero_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.ZERO_SLAB_NONCERTIFICATE
    if not projection_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
    if not diagnostic_ready:
        failure |= GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE  # noqa: E501
    if not operator_ready:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.CURRENT_OPERATOR_NONCERTIFICATE
        )
    if not action_ready:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.CURRENT_ACTION_NONCERTIFICATE
        )
    selected_native = disposition is (
        GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_TERMINAL_SECTOR
    )
    full_native = disposition is (
        GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_SLAB
    )
    scope_matches = (
        not selected_native
        or scope is GalerkinLocalTerminalScope.SELECTED_PRETERMINAL_FIBERS
    ) and (
        not full_native
        or scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
    )
    if not scope_matches:
        failure |= (
            GalerkinLocalVacuumTerminalFailure.DISPOSITION_SCOPE_MISMATCH
        )
    native = selected_native or full_native
    structural = bool(projection.structural_exact_zero_eligible)
    if native and not structural:
        failure |= GalerkinLocalVacuumTerminalFailure.NATIVE_STRUCTURAL_ZERO_UNAVAILABLE  # noqa: E501
    disposition_ready = scope_matches and (not native or structural)
    vacuum_ready = (
        zero_ready
        and projection_ready
        and diagnostic_ready
        and operator_ready
        and action_ready
        and bool(branch.branch_evidence_eligible)
        and bool(jnp.all(branch.cauchy_crosscheck_mask))
        and bool(jnp.all(branch.branch_crosscheck_mask))
        and bool(cut.cut_balance_eligible)
        and disposition_ready
    )
    terminal_identity = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.identity.v1",
            "projection_identity": projection.projection_identity_digest,
            "state_identity": projection.state_identity_digest,
            "scope": scope.value,
            "side": side.value,
            "axis": target.acquisition.terminal_axis,
            "outer_coordinate": stored_value_payload(
                np.asarray(outer_coordinate)
            ),
            "inner_coordinate": stored_value_payload(
                np.asarray(inner_coordinate)
            ),
            "disposition": disposition.value,
        }
    )
    terminal_evidence = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.evidence.v1",
            "identity": terminal_identity,
            "projection": projection.certificate_digest,
            "inner": inner.diagnostic_evidence_digest,
            "outer": outer.diagnostic_evidence_digest,
            "branch": branch.branch_evidence_digest,
            "cut": cut.cut_balance_digest,
            "failure": int(failure),
            "predicates": stored_value_payload(
                tuple(
                    jnp.asarray(value)
                    for value in (
                        diagnostic_ready,
                        operator_ready,
                        action_ready,
                        vacuum_ready,
                    )
                )
            ),
            "terminal_direct_pairs": maximum_terminal_direct_pairs,
            "branch_direct_terms": maximum_branch_direct_terms,
            "cut_direct_pairs": maximum_cut_direct_pairs,
            "root_work": maximum_root_work,
            "entire_policies": stored_value_payload(entire_policies),
            "interval_work": maximum_interval_work,
            "rational_bits": maximum_rational_bits,
        }
    )
    certificate = _make_local_vacuum_terminal_certificate(
        projection,
        inner,
        outer,
        branch,
        cut,
        (
            jnp.asarray(outer_coordinate, dtype=jnp.float64),
            jnp.asarray(inner_coordinate, dtype=jnp.float64),
        ),
        (
            jnp.asarray(diagnostic_ready),
            jnp.asarray(operator_ready),
            jnp.asarray(action_ready),
            jnp.asarray(vacuum_ready),
        ),
        jnp.asarray(int(failure), dtype=jnp.int64),
        terminal_axis=target.acquisition.terminal_axis,
        terminal_side=side,
        terminal_scope=scope,
        disposition=disposition,
        target_digest=target.target_digest,
        source_digest=(
            zero_slab.represented_source_certificate.source.source_digest
        ),
        state_identity_digest=projection.state_identity_digest,
        projection_identity_digest=projection.projection_identity_digest,
        parent_projection_certificate_digest=projection.certificate_digest,
        inner_terminal_evidence_digest=inner.diagnostic_evidence_digest,
        outer_terminal_evidence_digest=outer.diagnostic_evidence_digest,
        branch_evidence_digest=branch.branch_evidence_digest,
        cut_balance_digest=cut.cut_balance_digest,
        terminal_identity_digest=terminal_identity,
        terminal_evidence_digest=terminal_evidence,
    )
    return certificate  # noqa: RET504


def _checked_nonnegative_int(value: object, name: str) -> int:
    """PRIVATE: Validate one nonnegative signed-int64 resource policy.

    Parameters
    ----------
    value : object
        Candidate policy.
    name : str
        Public parameter name.

    Returns
    -------
    value : int
        Validated nonnegative Python integer.

    Raises
    ------
    TypeError
        If the policy is not exactly a Python integer.
    ValueError
        If the policy is outside the nonnegative signed-int64 range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value < 0 or value > _MAXIMUM_SIGNED_INT64:
        raise ValueError(f"{name} must be a nonnegative signed-int64 integer")
    return value


def certify_local_vacuum_terminal(  # noqa: PLR0913
    projection_certificate: GalerkinLocalProjectionDefectCertificate,
    *,
    disposition: GalerkinLocalVacuumTerminalDisposition | str,
    maximum_state_error: object,
    maximum_stability_direct_pairs: int = (
        _DEFAULT_MAXIMUM_STABILITY_DIRECT_PAIRS
    ),
    maximum_gram_pairs: int = _DEFAULT_MAXIMUM_GRAM_PAIRS,
    maximum_terminal_direct_pairs: int = (
        _DEFAULT_MAXIMUM_TERMINAL_DIRECT_PAIRS
    ),
    maximum_branch_direct_terms: int = (_DEFAULT_MAXIMUM_BRANCH_DIRECT_TERMS),
    maximum_cut_direct_pairs: int = _DEFAULT_MAXIMUM_CUT_DIRECT_PAIRS,
    maximum_root_work: int = _DEFAULT_MAXIMUM_ROOT_WORK,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_ENTIRE_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_INTERVAL_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumTerminalCertificate:
    """Compose one scoped LVT.39--LVT.56 vacuum-terminal certificate.

    The raw projection parent crosses its full public replay boundary. Both
    slab-endpoint L7 diagnostics are then rebuilt internally from the same
    authenticated target, state, scope, and exact stored coordinates. No
    caller-supplied terminal diagnostic, root, witness, or propagator is
    accepted.

    :see: :func:`~.test_local_vacuum_terminal.\
test_public_replay_rejects_digest_disposition_policy_and_tracing`

    Parameters
    ----------
    projection_certificate : GalerkinLocalProjectionDefectCertificate
        Raw local projection-defect parent to authenticate in full.
    disposition : GalerkinLocalVacuumTerminalDisposition | str
        Plane-defined, selected-native, or full-native continuation claim.
    maximum_state_error : object
        Independent finite positive exact-float64 L6 replay policy.
    maximum_stability_direct_pairs : int, optional
        Independent L6 replay work policy; defaults to 2,000,000.
    maximum_gram_pairs : int, optional
        Independent projection Gram replay policy; defaults to 2,000,000.
    maximum_terminal_direct_pairs : int, optional
        Shared independent L7 work policy; defaults to 2,000,000.
    maximum_branch_direct_terms : int, optional
        Linear branch direct-term policy; defaults to 2,000,000.
    maximum_cut_direct_pairs : int, optional
        Nonsymmetrized cut pair policy; defaults to 2,000,000.
    maximum_root_work : int, optional
        Per-fiber strict-root work policy; defaults to 64.
    precision_bits : int, optional
        Entire-helper remainder precision; defaults to 160.
    maximum_terms : int, optional
        Per-series helper term policy; defaults to 4096.
    maximum_entire_work : int, optional
        Per-call helper work policy; defaults to 1,000,000.
    maximum_range_reductions : int, optional
        Per-call helper range-reduction limit; defaults to 4096.
    maximum_interval_work : int, optional
        Per-fiber propagator interval-work policy; defaults to 1,000,000.
    maximum_rational_bits : int, optional
        Shared explicit direct/root/helper retained-bit policy; defaults to
        262,144 and is hard-capped at 1,048,576.

    Returns
    -------
    certificate : GalerkinLocalVacuumTerminalCertificate
        Canonical provisional certificate or typed noncertificate.

    Raises
    ------
    TypeError
        If the parent, disposition, or a resource policy has the wrong type.
    ValueError
        If projection replay or an independent resource policy differs.
    """
    if not isinstance(
        projection_certificate, GalerkinLocalProjectionDefectCertificate
    ):
        raise TypeError(
            "projection_certificate must be raw local projection storage"
        )
    _assert_concrete(projection_certificate)
    checked_disposition = _checked_disposition(disposition)
    stability_pairs = _checked_positive_int(
        maximum_stability_direct_pairs,
        "maximum_stability_direct_pairs",
    )
    gram_pairs = _checked_positive_int(
        maximum_gram_pairs, "maximum_gram_pairs"
    )
    terminal_pairs = _checked_positive_int(
        maximum_terminal_direct_pairs, "maximum_terminal_direct_pairs"
    )
    branch_terms = _checked_positive_int(
        maximum_branch_direct_terms, "maximum_branch_direct_terms"
    )
    cut_pairs = _checked_positive_int(
        maximum_cut_direct_pairs, "maximum_cut_direct_pairs"
    )
    root_work = _checked_positive_int(maximum_root_work, "maximum_root_work")
    precision = _checked_positive_int(precision_bits, "precision_bits")
    terms = _checked_positive_int(maximum_terms, "maximum_terms")
    entire_work = _checked_positive_int(
        maximum_entire_work, "maximum_entire_work"
    )
    reductions = _checked_nonnegative_int(
        maximum_range_reductions, "maximum_range_reductions"
    )
    interval_work = _checked_positive_int(
        maximum_interval_work, "maximum_interval_work"
    )
    rational_bits = _checked_positive_int(
        maximum_rational_bits, "maximum_rational_bits"
    )
    if rational_bits <= 1 or rational_bits > _HARD_MAXIMUM_RATIONAL_BITS:
        raise ValueError(
            "maximum_rational_bits must be greater than 1 and no larger "
            "than the hard 1,048,576-bit limit"
        )
    projection = prepare_local_projection_defect_certificate(
        projection_certificate,
        maximum_state_error=maximum_state_error,
        maximum_stability_direct_pairs=stability_pairs,
        maximum_gram_pairs=gram_pairs,
    )
    certificate = _certify_prepared_terminal(
        projection,
        checked_disposition,
        terminal_pairs,
        branch_terms,
        cut_pairs,
        root_work,
        (precision, terms, entire_work, reductions, rational_bits),
        interval_work,
        rational_bits,
    )
    return certificate  # noqa: RET504


def prepare_local_vacuum_terminal_certificate(  # noqa: PLR0913
    certificate: GalerkinLocalVacuumTerminalCertificate,
    *,
    disposition: GalerkinLocalVacuumTerminalDisposition | str,
    maximum_state_error: object,
    maximum_stability_direct_pairs: int = (
        _DEFAULT_MAXIMUM_STABILITY_DIRECT_PAIRS
    ),
    maximum_gram_pairs: int = _DEFAULT_MAXIMUM_GRAM_PAIRS,
    maximum_terminal_direct_pairs: int = (
        _DEFAULT_MAXIMUM_TERMINAL_DIRECT_PAIRS
    ),
    maximum_branch_direct_terms: int = (_DEFAULT_MAXIMUM_BRANCH_DIRECT_TERMS),
    maximum_cut_direct_pairs: int = _DEFAULT_MAXIMUM_CUT_DIRECT_PAIRS,
    maximum_root_work: int = _DEFAULT_MAXIMUM_ROOT_WORK,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_ENTIRE_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_INTERVAL_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumTerminalCertificate:
    """Replay every L8 parent, policy, helper route, field, and digest.

    :see: :func:`~.test_local_vacuum_terminal.\
test_public_replay_rejects_digest_disposition_policy_and_tracing`

    Parameters
    ----------
    certificate : GalerkinLocalVacuumTerminalCertificate
        Raw composed local vacuum-terminal storage to authenticate.
    disposition : GalerkinLocalVacuumTerminalDisposition | str
        Independent continuation disposition to replay exactly.
    maximum_state_error : object
        Independent finite positive exact-float64 L6 replay policy.
    maximum_stability_direct_pairs : int, optional
        Independent L6 replay work policy; defaults to 2,000,000.
    maximum_gram_pairs : int, optional
        Independent projection Gram replay policy; defaults to 2,000,000.
    maximum_terminal_direct_pairs : int, optional
        Shared independent L7 work policy; defaults to 2,000,000.
    maximum_branch_direct_terms : int, optional
        Linear branch direct-term policy; defaults to 2,000,000.
    maximum_cut_direct_pairs : int, optional
        Nonsymmetrized cut pair policy; defaults to 2,000,000.
    maximum_root_work : int, optional
        Per-fiber strict-root work policy; defaults to 64.
    precision_bits : int, optional
        Entire-helper remainder precision; defaults to 160.
    maximum_terms : int, optional
        Per-series helper term policy; defaults to 4096.
    maximum_entire_work : int, optional
        Per-call helper work policy; defaults to 1,000,000.
    maximum_range_reductions : int, optional
        Per-call helper range-reduction limit; defaults to 4096.
    maximum_interval_work : int, optional
        Per-fiber propagator interval-work policy; defaults to 1,000,000.
    maximum_rational_bits : int, optional
        Shared explicit direct/root/helper retained-bit policy; defaults to
        262,144 and is hard-capped at 1,048,576.

    Returns
    -------
    canonical : GalerkinLocalVacuumTerminalCertificate
        Fresh certificate reconstructed from authenticated primitive inputs.

    Raises
    ------
    TypeError
        If the submitted object or a resource policy has the wrong type.
    ValueError
        If any parent, policy, helper transcript, arithmetic route, field, or
        digest differs from complete replay.
    """
    if not isinstance(certificate, GalerkinLocalVacuumTerminalCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalVacuumTerminalCertificate"
        )
    _assert_concrete(certificate)
    canonical = certify_local_vacuum_terminal(
        certificate.projection_certificate,
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
            "local vacuum-terminal certificate failed complete replay"
        )
    return canonical


__all__: list[str] = [
    "certify_local_vacuum_terminal",
    "prepare_local_vacuum_terminal_certificate",
]
