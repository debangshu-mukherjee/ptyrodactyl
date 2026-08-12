r"""Classify local vacuum roots and enclose homogeneous propagators.

Extended Summary
----------------
This pure host leaf implements strict LVT.39 classification and exact
rational real-interval enclosures of LVT.41--LVT.43.  It uses no tolerance,
ordinary transcendental library, projection parent, forcing term, branch
amplitude, terminal disposition, or detector claim.  Every nongrazing
transcendental enclosure is delegated to the verified exact-rational entire
helper.

Routine Listings
----------------
:func:`classify_local_vacuum_root`
    Strictly classify and enclose one exact rational LVT.39 quantity.
:func:`enclose_local_vacuum_propagator`
    Enclose one homogeneous branch-specific Cauchy propagator.
:func:`make_local_vacuum_zero_witness`
    Bind equality of two canonical formal algebraic normal forms.
:func:`prepare_local_vacuum_propagator`
    Full-replay and exact-compare one homogeneous propagator.
:func:`prepare_local_vacuum_root_certificate`
    Full-replay and exact-compare one strict root certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction

from beartype.typing import Dict, Tuple

from ptyrodactyl._tools import (
    EntireEnclosureError,
    EntireEnclosureFailure,
    EntireWorkTranscript,
    RationalInterval,
    enclose_real_sin_cos,
    enclose_real_sinh_cosh,
    sha256,
    sqrt_fraction_upper,
)
from ptyrodactyl.types.local_vacuum_propagation_types import (
    _EVANESCENT_FORMULA,
    _GRAZING_FORMULA,
    _PROPAGATING_FORMULA,
    _PROPAGATOR_COMPLETION_SCOPE,
    _ROOT_CLASSIFICATION_FORMULA,
    _ROOT_COMPLETION_SCOPE,
    _ROOT_FORMULA,
    _WITNESS_FORMULA,
    _WITNESS_SCOPE,
    GalerkinLocalVacuumPropagationError,
    GalerkinLocalVacuumPropagationFailure,
    GalerkinLocalVacuumPropagator,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRootCertificate,
    GalerkinLocalVacuumRootClass,
    GalerkinLocalVacuumWorkTranscript,
    GalerkinLocalVacuumZeroWitness,
    GalerkinLocalVacuumZeroWitnessRoute,
    _make_local_vacuum_propagator,
    _make_local_vacuum_rational_interval,
    _make_local_vacuum_root_certificate,
    _make_local_vacuum_work_transcript,
    _make_local_vacuum_zero_witness,
    _validate_local_vacuum_propagator,
    _validate_local_vacuum_root_certificate,
    _validate_local_vacuum_zero_witness,
)

_DEFAULT_MAXIMUM_ENTIRE_WORK: int = 1_000_000
_DEFAULT_MAXIMUM_INTERVAL_WORK: int = 1_000_000
_DEFAULT_MAXIMUM_RANGE_REDUCTIONS: int = 4096
_DEFAULT_MAXIMUM_RATIONAL_BITS: int = 262_144
_DEFAULT_MAXIMUM_ROOT_WORK: int = 64
_DEFAULT_MAXIMUM_TERMS: int = 4096
_DEFAULT_PRECISION_BITS: int = 160
_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_MAXIMUM_SIGNED_INT64: int = (1 << 63) - 1
_PAIR_LENGTH: int = 2
_ONE: Fraction = Fraction(1)
_ZERO: Fraction = Fraction(0)
_ROOT_ALGORITHM: str = "exact_fraction_local_vacuum_root_v1"
_PROPAGATOR_ALGORITHM: str = "exact_fraction_local_vacuum_propagator_v1"
_ROOT_IDENTITY_DOMAIN: str = (
    "ptyrodactyl.local_vacuum_propagation.root_identity.v1"
)
_ROOT_EVIDENCE_DOMAIN: str = (
    "ptyrodactyl.local_vacuum_propagation.root_evidence.v1"
)
_WITNESS_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_vacuum_propagation.zero_witness.v1"
)
_PROPAGATOR_IDENTITY_DOMAIN: str = (
    "ptyrodactyl.local_vacuum_propagation.propagator_identity.v1"
)
_PROPAGATOR_EVIDENCE_DOMAIN: str = (
    "ptyrodactyl.local_vacuum_propagation.propagator_evidence.v1"
)

type _InputFormalTerm = Tuple[str, Fraction]
type _InputNormalForm = Tuple[_InputFormalTerm, ...]
type _StoredFormalTerm = Tuple[str, int, int]
type _StoredNormalForm = Tuple[_StoredFormalTerm, ...]
type _PropagatorEntries = Tuple[
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRationalInterval,
]
type _EntirePolicies = Tuple[int, int, int, int, int]


def _hex_integer(value: int) -> str:
    """PRIVATE: Encode one arbitrary-size integer without decimal conversion.

    Parameters
    ----------
    value : int
        Exact Python integer.

    Returns
    -------
    encoded : str
        Canonical sign-prefixed lowercase hexadecimal text.
    """
    sign = "-" if value < 0 else "+"
    encoded: str = f"{sign}{abs(value):x}"
    return encoded


def _exact_value_payload(value: object) -> object:  # noqa: PLR0911
    """PRIVATE: Canonically encode arbitrary-size exact local evidence.

    Parameters
    ----------
    value : object
        Supported local carrier, exact scalar, enum, or tuple.

    Returns
    -------
    payload : object
        JSON-safe exact payload using hexadecimal arbitrary-size integers.

    Raises
    ------
    TypeError
        If the value is outside this leaf's exact evidence vocabulary.
    """
    if value is None:
        payload: object = None
    elif isinstance(value, Enum):
        payload = {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _exact_value_payload(value.value),
        }
    elif isinstance(value, bool):
        payload = {"bool": value}
    elif isinstance(value, int):
        payload = {"integer_hex": _hex_integer(value)}
    elif isinstance(value, str):
        payload = {"string": value}
    elif isinstance(value, Fraction):
        payload = {
            "fraction_hex": {
                "numerator": _hex_integer(value.numerator),
                "denominator": _hex_integer(value.denominator),
            }
        }
    elif isinstance(value, tuple):
        payload = {"tuple": [_exact_value_payload(item) for item in value]}
    elif is_dataclass(value) and not isinstance(value, type):
        carrier_name = f"{type(value).__module__}.{type(value).__qualname__}"
        payload = {
            "dataclass": carrier_name,
            "fields": {
                field.name: _exact_value_payload(getattr(value, field.name))
                for field in fields(value)
            },
        }
    else:
        raise TypeError(
            "unsupported value in exact local-vacuum evidence payload"
        )
    return payload


def _checked_policy(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> int:
    """PRIVATE: Validate one bounded exact integer resource policy.

    Parameters
    ----------
    value : object
        Candidate Python integer policy.
    name : str
        Public parameter name used in diagnostics.
    allow_zero : bool
        Whether zero is admitted; default is false.

    Returns
    -------
    result : int
        Validated signed-int64 Python integer.

    Raises
    ------
    TypeError
        If the policy is not exactly a Python integer.
    ValueError
        If the policy lies outside its admitted range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    lower = 0 if allow_zero else 1
    if value < lower or value > _MAXIMUM_SIGNED_INT64:
        raise ValueError(f"{name} lies outside its signed-int64 range")
    result: int = value
    return result


def _checked_rational_bits(value: object) -> int:
    """PRIVATE: Validate one bounded retained-rational bit policy.

    Parameters
    ----------
    value : object
        Candidate positive maximum rational bit length.

    Returns
    -------
    result : int
        Validated hard-capped rational bit policy.

    Raises
    ------
    TypeError
        If the policy is not exactly a Python integer.
    ValueError
        If it exceeds the hard implementation cap.
    """
    result: int = _checked_policy(value, "maximum_rational_bits")
    if result > _HARD_MAXIMUM_RATIONAL_BITS:
        raise ValueError(
            "maximum_rational_bits exceeds the implementation cap"
        )
    return result


def _checked_entire_policies(
    precision_bits: object,
    maximum_terms: object,
    maximum_entire_work: object,
    maximum_range_reductions: object,
    maximum_rational_bits: object,
) -> _EntirePolicies:
    """PRIVATE: Validate the shared exact entire-helper policy tuple.

    Parameters
    ----------
    precision_bits : object
        Positive direct-series remainder target bits.
    maximum_terms : object
        Positive per-series retained-term budget.
    maximum_entire_work : object
        Positive helper exact-work budget.
    maximum_range_reductions : object
        Nonnegative per-scalar dyadic reduction limit.
    maximum_rational_bits : object
        Positive retained-rational bit policy.

    Returns
    -------
    policies : _EntirePolicies
        Validated precision, term, work, range, and rational-bit policies.

    Raises
    ------
    TypeError
        If any policy is not exactly a Python integer.
    ValueError
        If any policy is outside its structural range.
    EntireEnclosureError
        If the precision target exceeds the rational-size policy.
    """
    precision = _checked_policy(precision_bits, "precision_bits")
    terms = _checked_policy(maximum_terms, "maximum_terms")
    work = _checked_policy(maximum_entire_work, "maximum_entire_work")
    reductions = _checked_policy(
        maximum_range_reductions,
        "maximum_range_reductions",
        allow_zero=True,
    )
    rational_bits = _checked_rational_bits(maximum_rational_bits)
    if precision + 1 > rational_bits:
        raise EntireEnclosureError(
            EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
            0,
            "precision target exceeds the rational bit limit",
        )
    policies: _EntirePolicies = (
        precision,
        terms,
        work,
        reductions,
        rational_bits,
    )
    return policies


@dataclass
class _RationalLedger:
    """Count and size-bound exact rational interval operations."""

    algorithm: str
    maximum_work: int
    maximum_rational_bits: int
    additions: int = 0
    subtractions: int = 0
    multiplications: int = 0
    divisions: int = 0
    root_enclosures: int = 0
    exact_work_count: int = 0

    def fail(
        self,
        failure: EntireEnclosureFailure,
        message: str,
    ) -> None:
        """Raise one exact-resource failure at completed work."""
        raise EntireEnclosureError(failure, self.exact_work_count, message)

    def charge(self, operation: str) -> None:
        """Charge one numbered exact operation before evaluating it."""
        attempted = self.exact_work_count + 1
        if attempted > self.maximum_work:
            raise EntireEnclosureError(
                EntireEnclosureFailure.WORK_BUDGET_EXCEEDED,
                attempted,
                f"{self.algorithm} exact-work budget exceeded",
            )
        self.exact_work_count = attempted
        if operation == "add":
            self.additions += 1
        elif operation == "subtract":
            self.subtractions += 1
        elif operation == "multiply":
            self.multiplications += 1
        elif operation == "divide":
            self.divisions += 1
        elif operation == "root":
            self.root_enclosures += 1
        else:
            raise AssertionError("unknown exact-work operation")

    def retain(self, value: Fraction) -> Fraction:
        """Reject a non-Fraction or oversized retained endpoint."""
        if not isinstance(value, Fraction):
            self.fail(
                EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
                "exact helper returned a non-rational endpoint",
            )
        bits = max(
            abs(value.numerator).bit_length(),
            value.denominator.bit_length(),
        )
        if bits > self.maximum_rational_bits:
            self.fail(
                EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
                "local vacuum rational endpoint exceeds its bit limit",
            )
        result: Fraction = value
        return result

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        """Add and retain two exact rational operands."""
        self.charge("add")
        result: Fraction = self.retain(left + right)
        return result

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        """Subtract and retain two exact rational operands."""
        self.charge("subtract")
        result: Fraction = self.retain(left - right)
        return result

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        """Multiply and retain two exact rational operands."""
        self.charge("multiply")
        result: Fraction = self.retain(left * right)
        return result

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        """Divide and retain two exact rational operands."""
        self.charge("divide")
        result: Fraction = self.retain(numerator / denominator)
        return result

    def root_upper(self, value: Fraction) -> Fraction:
        """Call and verify one rational square-root upper enclosure."""
        self.charge("root")
        try:
            candidate = sqrt_fraction_upper(value)
        except (ArithmeticError, ValueError) as error:
            raise EntireEnclosureError(
                EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
                self.exact_work_count,
                "rational square-root upper enclosure failed",
            ) from error
        upper: Fraction = self.retain(candidate)
        square = self.multiply(upper, upper)
        if upper < 0 or square < value:
            self.fail(
                EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
                "rational square-root endpoint is not outward",
            )
        return upper

    def transcript(self) -> GalerkinLocalVacuumWorkTranscript:
        """Freeze one deterministic exact-work transcript."""
        result: GalerkinLocalVacuumWorkTranscript = (
            _make_local_vacuum_work_transcript(
                algorithm=self.algorithm,
                maximum_work=self.maximum_work,
                maximum_rational_bits=self.maximum_rational_bits,
                additions=self.additions,
                subtractions=self.subtractions,
                multiplications=self.multiplications,
                divisions=self.divisions,
                root_enclosures=self.root_enclosures,
                exact_work_count=self.exact_work_count,
            )
        )
        return result


def _checked_fraction(
    value: object,
    ledger: _RationalLedger,
    name: str,
) -> Fraction:
    """PRIVATE: Validate and size-check one exact rational input.

    Parameters
    ----------
    value : object
        Candidate exact rational.
    ledger : _RationalLedger
        Active retained-rational size policy.
    name : str
        Public input name used in diagnostics.

    Returns
    -------
    result : Fraction
        Validated exact rational input.

    Raises
    ------
    TypeError
        If the input is not exactly a Fraction.
    EntireEnclosureError
        If its numerator or denominator exceeds the bit policy.
    """
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a Fraction")
    result: Fraction = ledger.retain(value)
    return result


def _checked_interval(
    value: object,
    ledger: _RationalLedger,
) -> RationalInterval:
    """PRIVATE: Validate one ordered exact rational real interval.

    Parameters
    ----------
    value : object
        Candidate pair of exact Fraction endpoints.
    ledger : _RationalLedger
        Active retained-rational size policy.

    Returns
    -------
    interval : RationalInterval
        Ordered and size-checked exact rational interval.

    Raises
    ------
    TypeError
        If the input is not exactly two Fraction endpoints.
    ValueError
        If the submitted endpoints are reversed.
    EntireEnclosureError
        If an endpoint exceeds the rational-size policy.
    """
    if not isinstance(value, tuple) or len(value) != _PAIR_LENGTH:
        raise TypeError("q_interval must contain exactly two Fractions")
    lower_value, upper_value = value
    if not isinstance(lower_value, Fraction) or not isinstance(
        upper_value, Fraction
    ):
        raise TypeError("q_interval must contain exactly two Fractions")
    lower = ledger.retain(lower_value)
    upper = ledger.retain(upper_value)
    if lower > upper:
        raise ValueError("q_interval endpoints must be ordered")
    interval: RationalInterval = (lower, upper)
    return interval


def _checked_normal_form(
    value: object,
    maximum_rational_bits: int,
    name: str,
) -> _StoredNormalForm:
    """PRIVATE: Validate one caller-supplied canonical formal sum.

    Parameters
    ----------
    value : object
        Candidate tuple of ``(atom, Fraction)`` terms.
    maximum_rational_bits : int
        Positive coefficient bit policy.
    name : str
        Public input name used in diagnostics.

    Returns
    -------
    normal_form : _StoredNormalForm
        Canonical sorted nonzero formal terms in stored representation.

    Raises
    ------
    TypeError
        If the normal form or one term has the wrong type.
    ValueError
        If atoms are unsorted, duplicated, empty, or noncanonical.
    EntireEnclosureError
        If a rational coefficient exceeds the bit policy.
    """
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    result_terms: list[_StoredFormalTerm] = []
    atoms: list[str] = []
    for term in value:
        if not isinstance(term, tuple) or len(term) != _PAIR_LENGTH:
            raise TypeError(f"{name} terms must be (atom, Fraction) pairs")
        atom, coefficient = term
        if not isinstance(atom, str) or not isinstance(coefficient, Fraction):
            raise TypeError(f"{name} terms must be (atom, Fraction) pairs")
        if coefficient == 0:
            raise ValueError(f"{name} cannot retain zero coefficients")
        bits = max(
            abs(coefficient.numerator).bit_length(),
            coefficient.denominator.bit_length(),
        )
        if bits > maximum_rational_bits:
            raise EntireEnclosureError(
                EntireEnclosureFailure.RATIONAL_SIZE_LIMIT,
                0,
                f"{name} coefficient exceeds maximum_rational_bits",
            )
        atoms.append(atom)
        result_terms.append(
            (atom, coefficient.numerator, coefficient.denominator)
        )
    if atoms != sorted(set(atoms)):
        raise ValueError(f"{name} atoms must be sorted and unique")
    normal_form: _StoredNormalForm = tuple(result_terms)
    return normal_form


def _witness_digest(
    route: GalerkinLocalVacuumZeroWitnessRoute,
    left: _StoredNormalForm,
    right: _StoredNormalForm,
    maximum_rational_bits: int,
) -> str:
    """PRIVATE: Bind one complete formal zero-witness payload.

    Parameters
    ----------
    route : GalerkinLocalVacuumZeroWitnessRoute
        Exact-rational or symbolic formal route.
    left : _StoredNormalForm
        Canonical left formal sum.
    right : _StoredNormalForm
        Canonical right formal sum.
    maximum_rational_bits : int
        Bound coefficient bit policy.

    Returns
    -------
    digest : str
        Canonical lowercase SHA-256 witness digest.
    """
    payload: Dict[str, object] = {
        "domain": _WITNESS_DIGEST_DOMAIN,
        "route": _exact_value_payload(route),
        "left_normal_form": _exact_value_payload(left),
        "right_normal_form": _exact_value_payload(right),
        "maximum_rational_bits": maximum_rational_bits,
        "witness_formula": _WITNESS_FORMULA,
        "trust_scope": _WITNESS_SCOPE,
    }
    digest: str = sha256(payload)
    return digest


def make_local_vacuum_zero_witness(
    left_normal_form: _InputNormalForm,
    right_normal_form: _InputNormalForm,
    *,
    route: GalerkinLocalVacuumZeroWitnessRoute,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumZeroWitness:
    """Bind equality of two canonical formal algebraic normal forms.

    :see: :func:`~.test_local_vacuum_propagation.\
test_strict_root_classification_zero_witness_and_near_zero`

    Parameters
    ----------
    left_normal_form : _InputNormalForm
        Sorted unique nonzero ``(atom, Fraction)`` terms for the left side.
    right_normal_form : _InputNormalForm
        Sorted unique nonzero ``(atom, Fraction)`` terms for the right side.
    route : GalerkinLocalVacuumZeroWitnessRoute
        Exact-rational or symbolic normal-form route.
    maximum_rational_bits : int, optional
        Maximum coefficient numerator or denominator bits; default is 262,144.

    Returns
    -------
    witness : GalerkinLocalVacuumZeroWitness
        Unauthenticated formal equality witness.

    Raises
    ------
    TypeError
        If the route, forms, terms, or coefficients have wrong types.
    ValueError
        If either normal form is noncanonical for the selected route.
    EntireEnclosureError
        If a coefficient exceeds the rational-size policy.
    GalerkinLocalVacuumPropagationError
        If the two canonical normal forms are not exactly equal.

    Notes
    -----
    This witness is not physical LVT.39 evidence. A later target-owned
    composition must reconstruct both forms from authenticated physical
    primitives and may not accept a caller-created witness.
    """
    if not isinstance(route, GalerkinLocalVacuumZeroWitnessRoute):
        raise TypeError("route must be GalerkinLocalVacuumZeroWitnessRoute")
    bits = _checked_rational_bits(maximum_rational_bits)
    left = _checked_normal_form(left_normal_form, bits, "left_normal_form")
    right = _checked_normal_form(right_normal_form, bits, "right_normal_form")
    if left != right:
        raise GalerkinLocalVacuumPropagationError(
            GalerkinLocalVacuumPropagationFailure.ZERO_WITNESS_INCONSISTENT,
            0,
            "formal zero-witness normal forms are unequal",
        )
    digest = _witness_digest(route, left, right, bits)
    witness: GalerkinLocalVacuumZeroWitness = _make_local_vacuum_zero_witness(
        left,
        right,
        bits,
        route=route,
        witness_digest=digest,
    )
    return witness


def _prepare_zero_witness(
    witness: GalerkinLocalVacuumZeroWitness,
    maximum_rational_bits: int,
) -> GalerkinLocalVacuumZeroWitness:
    """PRIVATE: Full-replay one raw formal algebraic-zero witness.

    Parameters
    ----------
    witness : GalerkinLocalVacuumZeroWitness
        Raw unauthenticated formal witness.
    maximum_rational_bits : int
        Independent expected coefficient bit policy.

    Returns
    -------
    expected : GalerkinLocalVacuumZeroWitness
        Fully replayed exact formal witness.

    Raises
    ------
    TypeError
        If the witness has the wrong carrier type.
    ValueError
        If schema, policy, declarations, or digest do not replay.
    """
    checked = _validate_local_vacuum_zero_witness(witness)
    if checked.maximum_rational_bits != maximum_rational_bits:
        raise ValueError(
            "zero-witness rational-bit policy does not match replay"
        )
    digest = _witness_digest(
        checked.route,
        checked.left_normal_form,
        checked.right_normal_form,
        maximum_rational_bits,
    )
    expected: GalerkinLocalVacuumZeroWitness = _make_local_vacuum_zero_witness(
        checked.left_normal_form,
        checked.right_normal_form,
        maximum_rational_bits,
        route=checked.route,
        witness_digest=digest,
    )
    if _exact_value_payload(expected) != _exact_value_payload(checked):
        raise ValueError("zero witness does not match complete replay")
    return expected


def _positive_root_interval(
    magnitude: RationalInterval,
    ledger: _RationalLedger,
) -> RationalInterval:
    """PRIVATE: Enclose one positive interval square root on both sides.

    Parameters
    ----------
    magnitude : RationalInterval
        Strictly positive exact rational interval ``[a, b]``.
    ledger : _RationalLedger
        Active root-work and rational-size ledger.

    Returns
    -------
    root : RationalInterval
        Verified positive lower and upper square-root bounds.

    Raises
    ------
    EntireEnclosureError
        If work, rational size, reciprocal, or root enclosure fails.
    ValueError
        If the submitted magnitude is not strictly positive and ordered.
    """
    lower, upper = magnitude
    if lower <= 0 or lower > upper:
        raise ValueError("root magnitude interval must be strictly positive")
    reciprocal_lower = ledger.divide(_ONE, lower)
    reciprocal_root_upper = ledger.root_upper(reciprocal_lower)
    if reciprocal_root_upper <= 0:
        ledger.fail(
            EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
            "reciprocal square-root upper bound is nonpositive",
        )
    root_lower = ledger.divide(_ONE, reciprocal_root_upper)
    root_upper = ledger.root_upper(upper)
    lower_square = ledger.multiply(root_lower, root_lower)
    if root_lower <= 0 or root_lower > root_upper or lower_square > lower:
        ledger.fail(
            EntireEnclosureFailure.ROOT_ENCLOSURE_FAILURE,
            "reciprocal lower square-root enclosure is not outward",
        )
    root: RationalInterval = (root_lower, root_upper)
    return root


def _root_identity_digest(
    q_interval: GalerkinLocalVacuumRationalInterval,
    witness: GalerkinLocalVacuumZeroWitness | None,
) -> str:
    """PRIVATE: Bind one conditional q-interval and witness identity.

    Parameters
    ----------
    q_interval : GalerkinLocalVacuumRationalInterval
        Exact rational LVT.39 quantity enclosure.
    witness : GalerkinLocalVacuumZeroWitness | None
        Optional replayed formal algebraic-zero witness.

    Returns
    -------
    digest : str
        Canonical root-input identity digest.
    """
    payload: Dict[str, object] = {
        "domain": _ROOT_IDENTITY_DOMAIN,
        "q_interval": _exact_value_payload(q_interval),
        "zero_witness": _exact_value_payload(witness),
        "classification_formula": _ROOT_CLASSIFICATION_FORMULA,
        "witness_scope": _WITNESS_SCOPE,
    }
    digest: str = sha256(payload)
    return digest


def _root_evidence_digest(
    q_interval: GalerkinLocalVacuumRationalInterval,
    witness: GalerkinLocalVacuumZeroWitness | None,
    root_interval: GalerkinLocalVacuumRationalInterval | None,
    work: GalerkinLocalVacuumWorkTranscript,
    classification: GalerkinLocalVacuumRootClass,
    identity_digest: str,
) -> str:
    """PRIVATE: Bind every strict root-classification evidence field.

    Parameters
    ----------
    q_interval : GalerkinLocalVacuumRationalInterval
        Exact rational LVT.39 quantity enclosure.
    witness : GalerkinLocalVacuumZeroWitness | None
        Optional replayed formal zero witness.
    root_interval : GalerkinLocalVacuumRationalInterval | None
        Positive root, exact zero, or absent unresolved root.
    work : GalerkinLocalVacuumWorkTranscript
        Exact root-operation transcript and policies.
    classification : GalerkinLocalVacuumRootClass
        Strict root classification.
    identity_digest : str
        Bound conditional q-and-witness identity.

    Returns
    -------
    digest : str
        Canonical complete root-evidence digest.
    """
    payload: Dict[str, object] = {
        "domain": _ROOT_EVIDENCE_DOMAIN,
        "q_interval": _exact_value_payload(q_interval),
        "zero_witness": _exact_value_payload(witness),
        "root_interval": _exact_value_payload(root_interval),
        "work_transcript": _exact_value_payload(work),
        "classification": _exact_value_payload(classification),
        "classification_formula": _ROOT_CLASSIFICATION_FORMULA,
        "root_formula": _ROOT_FORMULA,
        "witness_scope": _WITNESS_SCOPE,
        "completion_scope": _ROOT_COMPLETION_SCOPE,
        "root_identity_digest": identity_digest,
    }
    digest: str = sha256(payload)
    return digest


def _classify_canonical_root(
    q_interval: RationalInterval,
    witness: GalerkinLocalVacuumZeroWitness | None,
    maximum_root_work: int,
    maximum_rational_bits: int,
) -> GalerkinLocalVacuumRootCertificate:
    """PRIVATE: Classify one validated q interval and enclose its root.

    Parameters
    ----------
    q_interval : RationalInterval
        Ordered size-checked exact rational LVT.39 interval.
    witness : GalerkinLocalVacuumZeroWitness | None
        Optional fully replayed formal algebraic-zero witness.
    maximum_root_work : int
        Positive call-wide exact root-operation budget.
    maximum_rational_bits : int
        Positive retained-rational bit policy.

    Returns
    -------
    certificate : GalerkinLocalVacuumRootCertificate
        Unauthenticated raw strict classification evidence.

    Raises
    ------
    GalerkinLocalVacuumPropagationError
        If a supplied zero witness conflicts with the q interval.
    EntireEnclosureError
        If root work, rational size, or enclosure fails.
    """
    ledger = _RationalLedger(
        _ROOT_ALGORITHM,
        maximum_root_work,
        maximum_rational_bits,
    )
    q_carrier = _make_local_vacuum_rational_interval(*q_interval)
    root: RationalInterval | None = None
    if q_interval[0] > 0:
        if witness is not None:
            raise GalerkinLocalVacuumPropagationError(
                GalerkinLocalVacuumPropagationFailure.ZERO_WITNESS_INCONSISTENT,
                0,
                "zero witness conflicts with strictly positive q interval",
            )
        classification = GalerkinLocalVacuumRootClass.PROPAGATING
        root = _positive_root_interval(q_interval, ledger)
    elif q_interval[1] < 0:
        if witness is not None:
            raise GalerkinLocalVacuumPropagationError(
                GalerkinLocalVacuumPropagationFailure.ZERO_WITNESS_INCONSISTENT,
                0,
                "zero witness conflicts with strictly negative q interval",
            )
        classification = GalerkinLocalVacuumRootClass.EVANESCENT
        magnitude: RationalInterval = (-q_interval[1], -q_interval[0])
        root = _positive_root_interval(magnitude, ledger)
    elif witness is not None:
        classification = GalerkinLocalVacuumRootClass.GRAZING
        root = (_ZERO, _ZERO)
    else:
        classification = GalerkinLocalVacuumRootClass.UNCLASSIFIED
    root_carrier = (
        None if root is None else _make_local_vacuum_rational_interval(*root)
    )
    work = ledger.transcript()
    identity = _root_identity_digest(q_carrier, witness)
    evidence = _root_evidence_digest(
        q_carrier,
        witness,
        root_carrier,
        work,
        classification,
        identity,
    )
    certificate: GalerkinLocalVacuumRootCertificate = (
        _make_local_vacuum_root_certificate(
            q_carrier,
            witness,
            root_carrier,
            work,
            classification=classification,
            root_identity_digest=identity,
            root_evidence_digest=evidence,
        )
    )
    return certificate


def classify_local_vacuum_root(
    q_interval: RationalInterval,
    *,
    zero_witness: GalerkinLocalVacuumZeroWitness | None = None,
    maximum_root_work: int = _DEFAULT_MAXIMUM_ROOT_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumRootCertificate:
    """Strictly classify and enclose one exact rational LVT.39 quantity.

    :see: :func:`~.test_local_vacuum_propagation.\
test_strict_root_classification_zero_witness_and_near_zero`

    Parameters
    ----------
    q_interval : RationalInterval
        Exact rational enclosure of the conditional LVT.39 quantity.
    zero_witness : GalerkinLocalVacuumZeroWitness or None, optional
        Optional formal zero witness; default is no witness.
    maximum_root_work : int, optional
        Maximum charged rational/root operations; default is 64.
    maximum_rational_bits : int, optional
        Maximum retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    certificate : GalerkinLocalVacuumRootCertificate
        Unauthenticated raw strict root-classification storage.

    Raises
    ------
    TypeError
        If inputs or policies have wrong types.
    ValueError
        If the interval, witness, or policies are structurally invalid.
    GalerkinLocalVacuumPropagationError
        If a zero witness conflicts with a strict-sign interval.
    EntireEnclosureError
        If work, rational size, or root enclosure fails.

    Notes
    -----
    Missing witness at exact ``[0, 0]`` remains ``UNCLASSIFIED``. Formal
    witness equality is not physical LVT.39 provenance.
    """
    root_work = _checked_policy(maximum_root_work, "maximum_root_work")
    rational_bits = _checked_rational_bits(maximum_rational_bits)
    ledger = _RationalLedger(
        _ROOT_ALGORITHM,
        root_work,
        rational_bits,
    )
    checked_q = _checked_interval(q_interval, ledger)
    checked_witness = (
        None
        if zero_witness is None
        else _prepare_zero_witness(zero_witness, rational_bits)
    )
    certificate: GalerkinLocalVacuumRootCertificate = _classify_canonical_root(
        checked_q,
        checked_witness,
        root_work,
        rational_bits,
    )
    return certificate


def prepare_local_vacuum_root_certificate(
    certificate: object,
    *,
    maximum_root_work: int = _DEFAULT_MAXIMUM_ROOT_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumRootCertificate:
    """Full-replay and exact-compare one strict root certificate.

    :see: :func:`~.test_local_vacuum_propagation.\
test_root_and_propagator_replay_reject_self_rehashed_forgeries`

    Parameters
    ----------
    certificate : object
        Raw unauthenticated root certificate.
    maximum_root_work : int, optional
        Independent expected root-operation budget; default is 64.
    maximum_rational_bits : int, optional
        Independent expected retained-rational bits; default is 262,144.

    Returns
    -------
    certificate : GalerkinLocalVacuumRootCertificate
        Fully replayed strict root certificate.

    Raises
    ------
    TypeError
        If the carrier or policies have wrong types.
    ValueError
        If schema, policies, classification, root, or digests do not replay.
    EntireEnclosureError
        If independently bounded root replay cannot complete.
    """
    if not isinstance(certificate, GalerkinLocalVacuumRootCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalVacuumRootCertificate"
        )
    checked = _validate_local_vacuum_root_certificate(certificate)
    root_work = _checked_policy(maximum_root_work, "maximum_root_work")
    rational_bits = _checked_rational_bits(maximum_rational_bits)
    witness = (
        None
        if checked.zero_witness is None
        else _prepare_zero_witness(checked.zero_witness, rational_bits)
    )
    expected = _classify_canonical_root(
        (checked.q_interval.lower, checked.q_interval.upper),
        witness,
        root_work,
        rational_bits,
    )
    if _exact_value_payload(expected) != _exact_value_payload(checked):
        raise ValueError("root certificate does not match complete replay")
    certificate_result: GalerkinLocalVacuumRootCertificate = expected
    return certificate_result


def _real_product(
    left: RationalInterval,
    right: RationalInterval,
    ledger: _RationalLedger,
) -> RationalInterval:
    """PRIVATE: Multiply two exact rational real intervals.

    Parameters
    ----------
    left : RationalInterval
        Checked left real interval.
    right : RationalInterval
        Checked right real interval.
    ledger : _RationalLedger
        Active exact interval-operation ledger.

    Returns
    -------
    product : RationalInterval
        Exact interval-arithmetic product.

    Raises
    ------
    EntireEnclosureError
        If work or rational-size policies fail.
    """
    products = (
        ledger.multiply(left[0], right[0]),
        ledger.multiply(left[0], right[1]),
        ledger.multiply(left[1], right[0]),
        ledger.multiply(left[1], right[1]),
    )
    product: RationalInterval = (min(products), max(products))
    return product


def _divide_by_positive(
    numerator: RationalInterval,
    denominator: RationalInterval,
    ledger: _RationalLedger,
) -> RationalInterval:
    """PRIVATE: Divide one real interval by a positive real interval.

    Parameters
    ----------
    numerator : RationalInterval
        Checked arbitrary-sign numerator interval.
    denominator : RationalInterval
        Checked strictly positive denominator interval.
    ledger : _RationalLedger
        Active exact interval-operation ledger.

    Returns
    -------
    quotient : RationalInterval
        Exact outward interval quotient.

    Raises
    ------
    ValueError
        If the denominator is not strictly positive.
    EntireEnclosureError
        If work or rational-size policies fail.
    """
    if denominator[0] <= 0:
        raise ValueError("interval denominator must be strictly positive")
    reciprocal: RationalInterval = (
        ledger.divide(_ONE, denominator[1]),
        ledger.divide(_ONE, denominator[0]),
    )
    quotient: RationalInterval = _real_product(numerator, reciprocal, ledger)
    return quotient


def _scale_nonnegative(
    interval: RationalInterval,
    scalar: Fraction,
    ledger: _RationalLedger,
) -> RationalInterval:
    """PRIVATE: Scale one real interval by a nonnegative exact scalar.

    Parameters
    ----------
    interval : RationalInterval
        Checked real interval.
    scalar : Fraction
        Checked nonnegative exact scalar.
    ledger : _RationalLedger
        Active exact interval-operation ledger.

    Returns
    -------
    scaled : RationalInterval
        Exact scaled real interval.

    Raises
    ------
    ValueError
        If the scalar is negative.
    EntireEnclosureError
        If work or rational-size policies fail.
    """
    if scalar < 0:
        raise ValueError("interval scale must be nonnegative")
    scaled: RationalInterval = (
        ledger.multiply(interval[0], scalar),
        ledger.multiply(interval[1], scalar),
    )
    return scaled


def _negate_interval(interval: RationalInterval) -> RationalInterval:
    """PRIVATE: Negate one exact rational real interval.

    Parameters
    ----------
    interval : RationalInterval
        Checked real interval.

    Returns
    -------
    negated : RationalInterval
        Exact sign-reversed interval.
    """
    negated: RationalInterval = (-interval[1], -interval[0])
    return negated


def _checked_helper_interval(
    interval: RationalInterval,
    ledger: _RationalLedger,
) -> RationalInterval:
    """PRIVATE: Size-check one exact entire-helper interval output.

    Parameters
    ----------
    interval : RationalInterval
        Exact helper-produced interval.
    ledger : _RationalLedger
        Active retained-rational size policy.

    Returns
    -------
    checked : RationalInterval
        Ordered size-checked helper interval.

    Raises
    ------
    ValueError
        If helper endpoints are reversed.
    EntireEnclosureError
        If a retained helper endpoint exceeds the bit policy.
    """
    lower = ledger.retain(interval[0])
    upper = ledger.retain(interval[1])
    if lower > upper:
        raise ValueError("entire helper returned a reversed interval")
    checked: RationalInterval = (lower, upper)
    return checked


def _identity_entries() -> _PropagatorEntries:
    """PRIVATE: Construct the exact rational 2x2 identity intervals.

    Returns
    -------
    entries : _PropagatorEntries
        Four row-major singleton identity intervals.
    """
    zero = _make_local_vacuum_rational_interval(_ZERO, _ZERO)
    one = _make_local_vacuum_rational_interval(_ONE, _ONE)
    entries: _PropagatorEntries = (one, zero, zero, one)
    return entries


def _grazing_entries(distance: Fraction) -> _PropagatorEntries:
    """PRIVATE: Construct the exact affine LVT.43 matrix intervals.

    Parameters
    ----------
    distance : Fraction
        Exact nonnegative propagation distance.

    Returns
    -------
    entries : _PropagatorEntries
        Four row-major singleton affine intervals.
    """
    zero = _make_local_vacuum_rational_interval(_ZERO, _ZERO)
    one = _make_local_vacuum_rational_interval(_ONE, _ONE)
    distance_interval = _make_local_vacuum_rational_interval(
        distance, distance
    )
    entries: _PropagatorEntries = (
        one,
        distance_interval,
        zero,
        one,
    )
    return entries


def _propagating_entries(
    root: RationalInterval,
    distance: Fraction,
    policies: _EntirePolicies,
    ledger: _RationalLedger,
) -> Tuple[_PropagatorEntries, EntireWorkTranscript]:
    """PRIVATE: Enclose the homogeneous propagating LVT.41 matrix.

    Parameters
    ----------
    root : RationalInterval
        Strictly positive normal-wavenumber interval.
    distance : Fraction
        Exact positive propagation distance.
    policies : _EntirePolicies
        Entire-helper precision, term, work, range, and bit policies.
    ledger : _RationalLedger
        Active local interval-operation ledger.

    Returns
    -------
    entries : _PropagatorEntries
        Four row-major exact rational matrix-entry intervals.
    transcript : EntireWorkTranscript
        Exact entire-helper resource transcript.

    Raises
    ------
    EntireEnclosureError
        If helper, interval-work, root, or rational-size policies fail.
    """
    argument = _scale_nonnegative(root, distance, ledger)
    sine_raw, cosine_raw, transcript = enclose_real_sin_cos(
        argument,
        precision_bits=policies[0],
        maximum_terms=policies[1],
        maximum_work=policies[2],
        maximum_range_reductions=policies[3],
        maximum_rational_bits=policies[4],
    )
    sine = _checked_helper_interval(sine_raw, ledger)
    cosine = _checked_helper_interval(cosine_raw, ledger)
    divided = _divide_by_positive(sine, root, ledger)
    root_sine = _real_product(root, sine, ledger)
    negative_root_sine = _negate_interval(root_sine)
    entries: _PropagatorEntries = (
        _make_local_vacuum_rational_interval(*cosine),
        _make_local_vacuum_rational_interval(*divided),
        _make_local_vacuum_rational_interval(*negative_root_sine),
        _make_local_vacuum_rational_interval(*cosine),
    )
    result: Tuple[_PropagatorEntries, EntireWorkTranscript] = (
        entries,
        transcript,
    )
    return result


def _evanescent_entries(
    root: RationalInterval,
    distance: Fraction,
    policies: _EntirePolicies,
    ledger: _RationalLedger,
) -> Tuple[_PropagatorEntries, EntireWorkTranscript]:
    """PRIVATE: Enclose the homogeneous evanescent LVT.42 matrix.

    Parameters
    ----------
    root : RationalInterval
        Strictly positive evanescence-rate interval.
    distance : Fraction
        Exact positive propagation distance.
    policies : _EntirePolicies
        Entire-helper precision, term, work, range, and bit policies.
    ledger : _RationalLedger
        Active local interval-operation ledger.

    Returns
    -------
    entries : _PropagatorEntries
        Four row-major exact rational matrix-entry intervals.
    transcript : EntireWorkTranscript
        Exact entire-helper resource transcript.

    Raises
    ------
    EntireEnclosureError
        If helper, interval-work, root, or rational-size policies fail.
    """
    argument = _scale_nonnegative(root, distance, ledger)
    sine_raw, cosine_raw, transcript = enclose_real_sinh_cosh(
        argument,
        precision_bits=policies[0],
        maximum_terms=policies[1],
        maximum_work=policies[2],
        maximum_range_reductions=policies[3],
        maximum_rational_bits=policies[4],
    )
    sine = _checked_helper_interval(sine_raw, ledger)
    cosine = _checked_helper_interval(cosine_raw, ledger)
    divided = _divide_by_positive(sine, root, ledger)
    root_sine = _real_product(root, sine, ledger)
    entries: _PropagatorEntries = (
        _make_local_vacuum_rational_interval(*cosine),
        _make_local_vacuum_rational_interval(*divided),
        _make_local_vacuum_rational_interval(*root_sine),
        _make_local_vacuum_rational_interval(*cosine),
    )
    result: Tuple[_PropagatorEntries, EntireWorkTranscript] = (
        entries,
        transcript,
    )
    return result


def _propagator_identity_digest(
    root: GalerkinLocalVacuumRootCertificate,
    distance: Fraction,
    formula: str,
) -> str:
    """PRIVATE: Bind one replayed root, distance, and branch formula.

    Parameters
    ----------
    root : GalerkinLocalVacuumRootCertificate
        Fully replayed strict root certificate.
    distance : Fraction
        Exact nonnegative propagation distance.
    formula : str
        Canonical branch-specific propagator formula.

    Returns
    -------
    digest : str
        Canonical propagator identity digest.
    """
    payload: Dict[str, object] = {
        "domain": _PROPAGATOR_IDENTITY_DOMAIN,
        "root_identity_digest": root.root_identity_digest,
        "root_classification": _exact_value_payload(root.classification),
        "distance": {
            "numerator": _hex_integer(distance.numerator),
            "denominator": _hex_integer(distance.denominator),
        },
        "propagator_formula": formula,
    }
    digest: str = sha256(payload)
    return digest


def _propagator_evidence_digest(
    root: GalerkinLocalVacuumRootCertificate,
    entries: _PropagatorEntries,
    entire_transcript: EntireWorkTranscript | None,
    interval_work: GalerkinLocalVacuumWorkTranscript,
    distance: Fraction,
    policies: _EntirePolicies,
    formula: str,
    identity_digest: str,
) -> str:
    """PRIVATE: Bind every homogeneous propagator evidence field.

    Parameters
    ----------
    root : GalerkinLocalVacuumRootCertificate
        Fully replayed strict root certificate.
    entries : _PropagatorEntries
        Four row-major exact rational matrix-entry intervals.
    entire_transcript : EntireWorkTranscript | None
        Exact helper evidence, absent on symbolic routes.
    interval_work : GalerkinLocalVacuumWorkTranscript
        Exact post-helper interval-operation transcript.
    distance : Fraction
        Exact nonnegative propagation distance.
    policies : _EntirePolicies
        Bound entire-helper policy tuple.
    formula : str
        Canonical branch-specific propagator formula.
    identity_digest : str
        Bound root, branch, and distance identity digest.

    Returns
    -------
    digest : str
        Canonical complete propagator-evidence digest.
    """
    payload: Dict[str, object] = {
        "domain": _PROPAGATOR_EVIDENCE_DOMAIN,
        "root_certificate": _exact_value_payload(root),
        "entries": _exact_value_payload(entries),
        "entire_transcript": _exact_value_payload(entire_transcript),
        "interval_work_transcript": _exact_value_payload(interval_work),
        "distance": {
            "numerator": _hex_integer(distance.numerator),
            "denominator": _hex_integer(distance.denominator),
        },
        "precision_bits": policies[0],
        "maximum_terms": policies[1],
        "maximum_entire_work": policies[2],
        "maximum_range_reductions": policies[3],
        "maximum_rational_bits": policies[4],
        "propagator_formula": formula,
        "trust_scope": _WITNESS_SCOPE,
        "completion_scope": _PROPAGATOR_COMPLETION_SCOPE,
        "propagator_identity_digest": identity_digest,
    }
    digest: str = sha256(payload)
    return digest


def _enclose_canonical_propagator(
    root: GalerkinLocalVacuumRootCertificate,
    distance: Fraction,
    policies: _EntirePolicies,
    maximum_interval_work: int,
) -> GalerkinLocalVacuumPropagator:
    """PRIVATE: Enclose one fully replayed classified homogeneous propagator.

    Parameters
    ----------
    root : GalerkinLocalVacuumRootCertificate
        Fully replayed strict root certificate.
    distance : Fraction
        Checked exact nonnegative propagation distance.
    policies : _EntirePolicies
        Entire-helper precision, term, work, range, and bit policies.
    maximum_interval_work : int
        Positive call-wide post-helper exact interval-operation budget.

    Returns
    -------
    propagator : GalerkinLocalVacuumPropagator
        Unauthenticated raw homogeneous propagator storage.

    Raises
    ------
    GalerkinLocalVacuumPropagationError
        If the replayed root remains unclassified.
    ValueError
        If a classified nongrazing certificate lacks its positive root.
    EntireEnclosureError
        If helper, interval work, range, root, or rational-size policy fails.
    """
    classification = root.classification
    if classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED:
        raise GalerkinLocalVacuumPropagationError(
            GalerkinLocalVacuumPropagationFailure.ROOT_UNCLASSIFIED,
            0,
            "unclassified root cannot define a vacuum propagator",
        )
    ledger = _RationalLedger(
        _PROPAGATOR_ALGORITHM,
        maximum_interval_work,
        policies[4],
    )
    checked_distance = ledger.retain(distance)
    entire_transcript: EntireWorkTranscript | None = None
    if checked_distance == 0:
        entries = _identity_entries()
    elif classification is GalerkinLocalVacuumRootClass.GRAZING:
        entries = _grazing_entries(checked_distance)
    else:
        if root.root_interval is None:
            raise ValueError("classified nongrazing root interval is absent")
        root_interval: RationalInterval = (
            ledger.retain(root.root_interval.lower),
            ledger.retain(root.root_interval.upper),
        )
        if classification is GalerkinLocalVacuumRootClass.PROPAGATING:
            entries, entire_transcript = _propagating_entries(
                root_interval,
                checked_distance,
                policies,
                ledger,
            )
        else:
            entries, entire_transcript = _evanescent_entries(
                root_interval,
                checked_distance,
                policies,
                ledger,
            )
    formula = {
        GalerkinLocalVacuumRootClass.PROPAGATING: _PROPAGATING_FORMULA,
        GalerkinLocalVacuumRootClass.EVANESCENT: _EVANESCENT_FORMULA,
        GalerkinLocalVacuumRootClass.GRAZING: _GRAZING_FORMULA,
    }[classification]
    interval_work = ledger.transcript()
    identity = _propagator_identity_digest(root, checked_distance, formula)
    evidence = _propagator_evidence_digest(
        root,
        entries,
        entire_transcript,
        interval_work,
        checked_distance,
        policies,
        formula,
        identity,
    )
    propagator: GalerkinLocalVacuumPropagator = _make_local_vacuum_propagator(
        root,
        entries,
        entire_transcript,
        interval_work,
        checked_distance,
        policies[0],
        policies[1],
        policies[2],
        policies[3],
        propagator_formula=formula,
        propagator_identity_digest=identity,
        propagator_evidence_digest=evidence,
    )
    return propagator


def enclose_local_vacuum_propagator(
    root_certificate: object,
    distance: Fraction,
    *,
    maximum_root_work: int = _DEFAULT_MAXIMUM_ROOT_WORK,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_ENTIRE_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_INTERVAL_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumPropagator:
    """Enclose one homogeneous branch-specific Cauchy propagator.

    :see: :func:`~.test_local_vacuum_propagation.\
test_propagating_evanescent_and_grazing_high_precision_oracles`

    The raw root certificate is never trusted. It is first fully replayed
    against the independently supplied root-work and rational-bit policies.

    Parameters
    ----------
    root_certificate : object
        Raw unauthenticated strict root certificate.
    distance : Fraction
        Exact nonnegative propagation distance.
    maximum_root_work : int, optional
        Independent root replay work budget; default is 64.
    precision_bits : int, optional
        Entire-helper direct-series remainder bits; default is 160.
    maximum_terms : int, optional
        Per-series entire-helper retained-term budget; default is 4096.
    maximum_entire_work : int, optional
        Call-wide entire-helper exact-work budget; default is 1,000,000.
    maximum_range_reductions : int, optional
        Per-scalar helper dyadic-reduction limit; default is 4096.
    maximum_interval_work : int, optional
        Post-helper exact interval-operation budget; default is 1,000,000.
    maximum_rational_bits : int, optional
        Shared retained numerator or denominator bits; default is 262,144.

    Returns
    -------
    propagator : GalerkinLocalVacuumPropagator
        Unauthenticated raw row-major exact rational matrix enclosure.

    Raises
    ------
    TypeError
        If the root, distance, or resource policies have wrong types.
    ValueError
        If the root replay, distance, or resource ranges are invalid.
    GalerkinLocalVacuumPropagationError
        If the fully replayed root remains unclassified.
    EntireEnclosureError
        If term, work, range, root, or rational-size policy fails.
    """
    root = prepare_local_vacuum_root_certificate(
        root_certificate,
        maximum_root_work=maximum_root_work,
        maximum_rational_bits=maximum_rational_bits,
    )
    policies = _checked_entire_policies(
        precision_bits,
        maximum_terms,
        maximum_entire_work,
        maximum_range_reductions,
        maximum_rational_bits,
    )
    interval_work = _checked_policy(
        maximum_interval_work, "maximum_interval_work"
    )
    ledger = _RationalLedger(
        _PROPAGATOR_ALGORITHM,
        interval_work,
        policies[4],
    )
    checked_distance = _checked_fraction(distance, ledger, "distance")
    if checked_distance < 0:
        raise ValueError("distance must be nonnegative")
    propagator: GalerkinLocalVacuumPropagator = _enclose_canonical_propagator(
        root,
        checked_distance,
        policies,
        interval_work,
    )
    return propagator


def prepare_local_vacuum_propagator(
    propagator: object,
    *,
    maximum_root_work: int = _DEFAULT_MAXIMUM_ROOT_WORK,
    precision_bits: int = _DEFAULT_PRECISION_BITS,
    maximum_terms: int = _DEFAULT_MAXIMUM_TERMS,
    maximum_entire_work: int = _DEFAULT_MAXIMUM_ENTIRE_WORK,
    maximum_range_reductions: int = _DEFAULT_MAXIMUM_RANGE_REDUCTIONS,
    maximum_interval_work: int = _DEFAULT_MAXIMUM_INTERVAL_WORK,
    maximum_rational_bits: int = _DEFAULT_MAXIMUM_RATIONAL_BITS,
) -> GalerkinLocalVacuumPropagator:
    """Full-replay and exact-compare one homogeneous propagator.

    :see: :func:`~.test_local_vacuum_propagation.\
test_root_and_propagator_replay_reject_self_rehashed_forgeries`

    Parameters
    ----------
    propagator : object
        Raw unauthenticated propagator storage.
    maximum_root_work : int, optional
        Independent root replay work budget; default is 64.
    precision_bits : int, optional
        Independent entire-helper remainder bits; default is 160.
    maximum_terms : int, optional
        Independent per-series retained-term budget; default is 4096.
    maximum_entire_work : int, optional
        Independent helper exact-work budget; default is 1,000,000.
    maximum_range_reductions : int, optional
        Independent per-scalar reduction limit; default is 4096.
    maximum_interval_work : int, optional
        Independent post-helper interval-work budget; default is 1,000,000.
    maximum_rational_bits : int, optional
        Independent shared rational-bit policy; default is 262,144.

    Returns
    -------
    propagator : GalerkinLocalVacuumPropagator
        Fully replayed homogeneous propagator enclosure.

    Raises
    ------
    TypeError
        If the carrier, distance, or policies have wrong types.
    ValueError
        If any nested root, policy, matrix, transcript, or digest differs.
    GalerkinLocalVacuumPropagationError
        If the independently replayed root remains unclassified.
    EntireEnclosureError
        If independently bounded replay cannot complete.
    """
    if not isinstance(propagator, GalerkinLocalVacuumPropagator):
        raise TypeError("propagator must be GalerkinLocalVacuumPropagator")
    checked = _validate_local_vacuum_propagator(propagator)
    expected = enclose_local_vacuum_propagator(
        checked.root_certificate,
        checked.distance,
        maximum_root_work=maximum_root_work,
        precision_bits=precision_bits,
        maximum_terms=maximum_terms,
        maximum_entire_work=maximum_entire_work,
        maximum_range_reductions=maximum_range_reductions,
        maximum_interval_work=maximum_interval_work,
        maximum_rational_bits=maximum_rational_bits,
    )
    if _exact_value_payload(expected) != _exact_value_payload(checked):
        raise ValueError("vacuum propagator does not match complete replay")
    propagator_result: GalerkinLocalVacuumPropagator = expected
    return propagator_result


__all__: list[str] = [
    "classify_local_vacuum_root",
    "enclose_local_vacuum_propagator",
    "make_local_vacuum_zero_witness",
    "prepare_local_vacuum_propagator",
    "prepare_local_vacuum_root_certificate",
]
