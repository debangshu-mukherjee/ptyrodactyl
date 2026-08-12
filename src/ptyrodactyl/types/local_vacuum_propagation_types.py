r"""Define exact local vacuum-root and propagator evidence carriers.

Extended Summary
----------------
These carriers preserve exact rational LVT.39 root classifications and the
four real interval entries of homogeneous LVT.41--LVT.43 Cauchy
propagators.  A formal zero witness proves only equality of its two stored
canonical algebraic normal forms.  It is not physical LVT.39 evidence until
a later target-owned composition reconstructs those forms from authenticated
physical primitives.

Routine Listings
----------------
:class:`GalerkinLocalVacuumPropagationError`
    Report a typed branch or formal-witness failure.
:class:`GalerkinLocalVacuumPropagationFailure`
    Enumerate failures outside exact entire-helper resources.
:class:`GalerkinLocalVacuumPropagator`
    Store one replayable homogeneous 2x2 Cauchy propagator enclosure.
:class:`GalerkinLocalVacuumRationalInterval`
    Store one exact rational real interval.
:class:`GalerkinLocalVacuumRootCertificate`
    Store one strict replayable LVT.39 root classification.
:class:`GalerkinLocalVacuumRootClass`
    Distinguish propagating, evanescent, grazing, and unresolved roots.
:class:`GalerkinLocalVacuumWorkTranscript`
    Store deterministic exact interval-operation work evidence.
:class:`GalerkinLocalVacuumZeroWitness`
    Store equality of two canonical formal algebraic normal forms.
:class:`GalerkinLocalVacuumZeroWitnessRoute`
    Distinguish exact-rational and symbolic normal-form equality.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from fractions import Fraction

import equinox as eqx
from beartype.typing import Tuple

from ptyrodactyl._tools import EntireWorkTranscript

_HARD_MAXIMUM_RATIONAL_BITS: int = 1_048_576
_MAXIMUM_SIGNED_INT64: int = (1 << 63) - 1
_SHA256_HEX_LENGTH: int = 64
_FORMAL_TERM_SIZE: int = 3
_PROPAGATOR_ENTRY_COUNT: int = 4
_FORMAL_ATOM = re.compile(r"(?:1|[A-Za-z][A-Za-z0-9_.:-]{0,127})\Z")
_ROOT_CLASSIFICATION_FORMULA: str = (
    "PROPAGATING iff q_lower>0; EVANESCENT iff q_upper<0; GRAZING iff "
    "q_lower<=0<=q_upper and an algebraic-zero witness replays; otherwise "
    "UNCLASSIFIED"
)
_ROOT_FORMULA: str = (
    "positive root [1/sqrt_upper(1/a), sqrt_upper(b)] for [a,b]>0; "
    "evanescent magnitude uses [-q_upper,-q_lower]"
)
_WITNESS_FORMULA: str = (
    "q := left_canonical_normal_form - right_canonical_normal_form"
)
_WITNESS_SCOPE: str = (
    "formal equality only; not physical LVT.39 evidence; a later target-owned "
    "composition must reconstruct both normal forms from authenticated "
    "physical primitives and may not accept a caller witness"
)
_ROOT_COMPLETION_SCOPE: str = (
    "strict conditional root classification and positive root interval only; "
    "no projection parent, physical LVT.39 provenance, branch amplitude, "
    "forced integral, terminal disposition, or detector claim"
)
_PROPAGATOR_COMPLETION_SCOPE: str = (
    "homogeneous LVT.41--LVT.43 2x2 Cauchy propagator enclosure conditional "
    "on a fully replayed root certificate; no projection parent, physical "
    "LVT.39 provenance, forcing, mismatch, branch amplitude, terminal "
    "disposition, or detector claim"
)
_PROPAGATING_FORMULA: str = "[[cos(k*s), sin(k*s)/k], [-k*sin(k*s), cos(k*s)]]"
_EVANESCENT_FORMULA: str = (
    "[[cosh(gamma*s), sinh(gamma*s)/gamma], "
    "[gamma*sinh(gamma*s), cosh(gamma*s)]]"
)
_GRAZING_FORMULA: str = "[[1, s], [0, 1]]"

type _FormalTerm = Tuple[str, int, int]
type _FormalNormalForm = Tuple[_FormalTerm, ...]
type _PropagatorEntries = Tuple[
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRationalInterval,
]


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise one structural carrier-invariant failure.

    Parameters
    ----------
    condition : bool
        Whether the invariant failed.
    message : str
        Diagnostic for the rejected raw carrier.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 text value.

    Parameters
    ----------
    value : str
        Candidate hexadecimal digest.

    Returns
    -------
    valid : bool
        Whether the candidate is canonical SHA-256 text.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


def _valid_policy(value: int, *, allow_zero: bool = False) -> bool:
    """PRIVATE: Check one bounded Python-integer resource policy.

    Parameters
    ----------
    value : int
        Candidate exact resource policy.
    allow_zero : bool
        Whether zero is admitted; default is false.

    Returns
    -------
    valid : bool
        Whether the policy is an admitted signed-int64 integer.
    """
    lower = 0 if allow_zero else 1
    valid: bool = (
        not isinstance(value, bool)
        and isinstance(value, int)
        and lower <= value <= _MAXIMUM_SIGNED_INT64
    )
    return valid


class GalerkinLocalVacuumRootClass(str, Enum):
    """Distinguish propagating, evanescent, grazing, and unresolved roots.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_root_enums_and_carrier_fields_are_disjoint`
    """

    PROPAGATING = "propagating"
    EVANESCENT = "evanescent"
    GRAZING = "grazing"
    UNCLASSIFIED = "unclassified"


class GalerkinLocalVacuumZeroWitnessRoute(str, Enum):
    """Distinguish exact-rational and symbolic normal-form equality.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_root_enums_and_carrier_fields_are_disjoint`
    """

    EXACT_RATIONAL_DIFFERENCE = "exact_rational_difference"
    SYMBOLIC_NORMAL_FORM_DIFFERENCE = "symbolic_normal_form_difference"


class GalerkinLocalVacuumPropagationFailure(str, Enum):
    """Enumerate failures outside exact entire-helper resources.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_root_enums_and_carrier_fields_are_disjoint`
    """

    ZERO_WITNESS_INCONSISTENT = "zero_witness_inconsistent"
    ROOT_UNCLASSIFIED = "root_unclassified"


class GalerkinLocalVacuumPropagationError(ValueError):
    """Report a typed branch or formal-witness failure.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_root_enums_and_carrier_fields_are_disjoint`
    """

    failure: GalerkinLocalVacuumPropagationFailure
    exact_work_count: int

    def __init__(
        self,
        failure: GalerkinLocalVacuumPropagationFailure,
        exact_work_count: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.exact_work_count = exact_work_count


class GalerkinLocalVacuumRationalInterval(eqx.Module):
    """Store one exact rational real interval.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_root_enums_and_carrier_fields_are_disjoint`
    """

    lower_numerator: int = eqx.field(static=True)
    lower_denominator: int = eqx.field(static=True)
    upper_numerator: int = eqx.field(static=True)
    upper_denominator: int = eqx.field(static=True)

    @property
    def lower(self) -> Fraction:
        """Return the exact lower endpoint."""
        result: Fraction = Fraction(
            self.lower_numerator, self.lower_denominator
        )
        return result

    @property
    def upper(self) -> Fraction:
        """Return the exact upper endpoint."""
        result: Fraction = Fraction(
            self.upper_numerator, self.upper_denominator
        )
        return result


class GalerkinLocalVacuumZeroWitness(eqx.Module):
    """Store equality of two canonical formal algebraic normal forms.

    The witness proves only formal equality under ``q := left - right``. It
    does not authenticate either normal form as the physical LVT.39
    expression.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_zero_witness_scope_is_formal_and_parent_free`
    """

    left_normal_form: _FormalNormalForm = eqx.field(static=True)
    right_normal_form: _FormalNormalForm = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    route: GalerkinLocalVacuumZeroWitnessRoute = eqx.field(static=True)
    witness_formula: str = eqx.field(static=True)
    trust_scope: str = eqx.field(static=True)
    witness_digest: str = eqx.field(static=True)


class GalerkinLocalVacuumWorkTranscript(eqx.Module):
    """Store deterministic exact interval-operation work evidence.

    One work unit is one issued exact rational addition, subtraction,
    multiplication, or division, or one rational square-root helper call.
    Comparisons, sign changes, constants, and bit checks are free.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_root_enums_and_carrier_fields_are_disjoint`
    """

    algorithm: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_rational_bits: int = eqx.field(static=True)
    additions: int = eqx.field(static=True)
    subtractions: int = eqx.field(static=True)
    multiplications: int = eqx.field(static=True)
    divisions: int = eqx.field(static=True)
    root_enclosures: int = eqx.field(static=True)
    exact_work_count: int = eqx.field(static=True)


class GalerkinLocalVacuumRootCertificate(eqx.Module):
    """Store one strict replayable LVT.39 root classification.

    Raw storage is unauthenticated until
    ``prepare_local_vacuum_root_certificate`` independently replays it.
    Formal grazing evidence remains conditional and parent-free.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_zero_witness_scope_is_formal_and_parent_free`
    """

    q_interval: GalerkinLocalVacuumRationalInterval
    zero_witness: GalerkinLocalVacuumZeroWitness | None
    root_interval: GalerkinLocalVacuumRationalInterval | None
    work_transcript: GalerkinLocalVacuumWorkTranscript
    classification: GalerkinLocalVacuumRootClass = eqx.field(static=True)
    classification_formula: str = eqx.field(static=True)
    root_formula: str = eqx.field(static=True)
    witness_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    root_identity_digest: str = eqx.field(static=True)
    root_evidence_digest: str = eqx.field(static=True)


class GalerkinLocalVacuumPropagator(eqx.Module):
    """Store one replayable homogeneous 2x2 Cauchy propagator enclosure.

    Entries are row-major exact rational real intervals. Raw storage is
    unauthenticated until ``prepare_local_vacuum_propagator`` independently
    replays the nested root certificate and both resource policies.

    :see: :func:`~.test_local_vacuum_propagation_types.\
test_vacuum_propagator_keeps_exact_entries_and_helper_transcript`
    """

    root_certificate: GalerkinLocalVacuumRootCertificate
    entries: _PropagatorEntries
    entire_transcript: EntireWorkTranscript | None
    interval_work_transcript: GalerkinLocalVacuumWorkTranscript
    distance_numerator: int = eqx.field(static=True)
    distance_denominator: int = eqx.field(static=True)
    precision_bits: int = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    maximum_entire_work: int = eqx.field(static=True)
    maximum_range_reductions: int = eqx.field(static=True)
    propagator_formula: str = eqx.field(static=True)
    trust_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    propagator_identity_digest: str = eqx.field(static=True)
    propagator_evidence_digest: str = eqx.field(static=True)

    @property
    def distance(self) -> Fraction:
        """Return the exact nonnegative propagation distance."""
        result: Fraction = Fraction(
            self.distance_numerator, self.distance_denominator
        )
        return result


def _validate_local_vacuum_rational_interval(
    interval: GalerkinLocalVacuumRationalInterval,
) -> GalerkinLocalVacuumRationalInterval:
    """PRIVATE: Validate one raw exact rational interval carrier.

    Parameters
    ----------
    interval : GalerkinLocalVacuumRationalInterval
        Raw interval storage.

    Returns
    -------
    validated : GalerkinLocalVacuumRationalInterval
        Structurally validated interval storage.

    Raises
    ------
    TypeError
        If the submitted value has the wrong carrier type.
    ValueError
        If endpoints are noncanonical, oversized, or reversed.
    """
    if not isinstance(interval, GalerkinLocalVacuumRationalInterval):
        raise TypeError("interval must be GalerkinLocalVacuumRationalInterval")
    values = (
        interval.lower_numerator,
        interval.lower_denominator,
        interval.upper_numerator,
        interval.upper_denominator,
    )
    _raise_if(
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ),
        "rational interval fields must be Python integers",
    )
    _raise_if(
        interval.lower_denominator <= 0 or interval.upper_denominator <= 0,
        "rational interval denominators must be positive",
    )
    _raise_if(
        math.gcd(abs(interval.lower_numerator), interval.lower_denominator)
        != 1
        or math.gcd(abs(interval.upper_numerator), interval.upper_denominator)
        != 1,
        "rational interval endpoints must be reduced",
    )
    _raise_if(
        any(
            max(abs(value).bit_length(), 1) > _HARD_MAXIMUM_RATIONAL_BITS
            for value in values
        ),
        "rational interval endpoint exceeds the hard bit limit",
    )
    _raise_if(interval.lower > interval.upper, "rational interval is reversed")
    validated: GalerkinLocalVacuumRationalInterval = interval
    return validated


def _validate_normal_form(
    normal_form: _FormalNormalForm,
    maximum_rational_bits: int,
) -> _FormalNormalForm:
    """PRIVATE: Validate one canonical sparse formal normal form.

    Parameters
    ----------
    normal_form : _FormalNormalForm
        Candidate sorted atom and reduced rational-coefficient terms.
    maximum_rational_bits : int
        Positive configured coefficient bit limit.

    Returns
    -------
    validated : _FormalNormalForm
        Structurally validated canonical normal form.

    Raises
    ------
    ValueError
        If terms, atoms, coefficients, or ordering are noncanonical.
    """
    _raise_if(
        not isinstance(normal_form, tuple), "normal form must be a tuple"
    )
    atoms: list[str] = []
    for term in normal_form:
        _raise_if(
            not isinstance(term, tuple) or len(term) != _FORMAL_TERM_SIZE,
            "formal terms must be (atom, numerator, denominator) tuples",
        )
        atom, numerator, denominator = term
        _raise_if(
            not isinstance(atom, str) or _FORMAL_ATOM.fullmatch(atom) is None,
            "formal atom is not canonical",
        )
        _raise_if(
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or numerator == 0
            or denominator <= 0,
            "formal coefficient must be one nonzero reduced rational",
        )
        _raise_if(
            math.gcd(abs(numerator), denominator) != 1,
            "formal coefficient must be reduced",
        )
        _raise_if(
            max(abs(numerator).bit_length(), denominator.bit_length())
            > maximum_rational_bits,
            "formal coefficient exceeds maximum_rational_bits",
        )
        atoms.append(atom)
    _raise_if(
        atoms != sorted(set(atoms)),
        "formal atoms must be sorted and unique",
    )
    validated: _FormalNormalForm = normal_form
    return validated


def _validate_local_vacuum_zero_witness(
    witness: GalerkinLocalVacuumZeroWitness,
) -> GalerkinLocalVacuumZeroWitness:
    """PRIVATE: Validate one raw formal algebraic-zero witness.

    Parameters
    ----------
    witness : GalerkinLocalVacuumZeroWitness
        Raw formal witness storage.

    Returns
    -------
    validated : GalerkinLocalVacuumZeroWitness
        Structurally validated witness storage.

    Raises
    ------
    TypeError
        If the value or route has the wrong type.
    ValueError
        If policy, normal forms, scope, or digest is inconsistent.
    """
    if not isinstance(witness, GalerkinLocalVacuumZeroWitness):
        raise TypeError("witness must be GalerkinLocalVacuumZeroWitness")
    if not isinstance(witness.route, GalerkinLocalVacuumZeroWitnessRoute):
        raise TypeError("witness route has the wrong enum type")
    _raise_if(
        not _valid_policy(witness.maximum_rational_bits)
        or witness.maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS,
        "witness maximum_rational_bits is invalid",
    )
    left = _validate_normal_form(
        witness.left_normal_form, witness.maximum_rational_bits
    )
    right = _validate_normal_form(
        witness.right_normal_form, witness.maximum_rational_bits
    )
    _raise_if(left != right, "formal zero-witness normal forms must be equal")
    if witness.route is (
        GalerkinLocalVacuumZeroWitnessRoute.EXACT_RATIONAL_DIFFERENCE
    ):
        _raise_if(
            len(left) != 1 or left[0][0] != "1",
            "exact-rational witness must contain only the constant atom",
        )
    else:
        _raise_if(
            not any(term[0] != "1" for term in left),
            "symbolic witness must contain a nonconstant atom",
        )
    _raise_if(
        witness.witness_formula != _WITNESS_FORMULA,
        "zero-witness formula is not canonical",
    )
    _raise_if(
        witness.trust_scope != _WITNESS_SCOPE,
        "zero-witness trust scope is not canonical",
    )
    _raise_if(
        not _valid_digest(witness.witness_digest), "invalid witness digest"
    )
    validated: GalerkinLocalVacuumZeroWitness = witness
    return validated


def _validate_local_vacuum_work_transcript(
    transcript: GalerkinLocalVacuumWorkTranscript,
) -> GalerkinLocalVacuumWorkTranscript:
    """PRIVATE: Validate one deterministic local exact-work transcript.

    Parameters
    ----------
    transcript : GalerkinLocalVacuumWorkTranscript
        Raw local-work evidence.

    Returns
    -------
    validated : GalerkinLocalVacuumWorkTranscript
        Structurally validated work evidence.

    Raises
    ------
    TypeError
        If the value has the wrong carrier type.
    ValueError
        If policies, counters, or their exact sum are inconsistent.
    """
    if not isinstance(transcript, GalerkinLocalVacuumWorkTranscript):
        raise TypeError("transcript must be GalerkinLocalVacuumWorkTranscript")
    _raise_if(
        not transcript.algorithm.strip(), "work algorithm must be nonempty"
    )
    _raise_if(
        not _valid_policy(transcript.maximum_work)
        or not _valid_policy(transcript.maximum_rational_bits)
        or transcript.maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS,
        "work transcript policies are invalid",
    )
    counters = (
        transcript.additions,
        transcript.subtractions,
        transcript.multiplications,
        transcript.divisions,
        transcript.root_enclosures,
        transcript.exact_work_count,
    )
    _raise_if(
        any(not _valid_policy(value, allow_zero=True) for value in counters),
        "work transcript counters must be nonnegative signed-int64 integers",
    )
    expected = sum(counters[:-1])
    _raise_if(
        transcript.exact_work_count != expected
        or transcript.exact_work_count > transcript.maximum_work,
        "exact work count must equal the bounded counter sum",
    )
    validated: GalerkinLocalVacuumWorkTranscript = transcript
    return validated


def _validate_entire_transcript(transcript: EntireWorkTranscript) -> None:
    """PRIVATE: Validate structural entire-helper transcript fields.

    Parameters
    ----------
    transcript : EntireWorkTranscript
        Raw nested helper evidence.

    Raises
    ------
    TypeError
        If the value has the wrong transcript type.
    ValueError
        If policies or counters are outside structural ranges.
    """
    if not isinstance(transcript, EntireWorkTranscript):
        raise TypeError("entire_transcript has the wrong type")
    positive = (
        transcript.precision_bits,
        transcript.maximum_terms,
        transcript.maximum_work,
        transcript.maximum_rational_bits,
    )
    nonnegative = (
        transcript.maximum_range_reductions,
        transcript.series_terms,
        transcript.range_reductions,
        transcript.root_enclosures,
        transcript.rectangle_products,
        transcript.reciprocal_steps,
        transcript.exact_work_count,
    )
    _raise_if(
        any(not _valid_policy(value) for value in positive)
        or any(
            not _valid_policy(value, allow_zero=True) for value in nonnegative
        ),
        "entire-helper transcript has invalid policies or counters",
    )
    _raise_if(
        transcript.maximum_rational_bits > _HARD_MAXIMUM_RATIONAL_BITS
        or transcript.exact_work_count > transcript.maximum_work,
        "entire-helper transcript exceeds a resource policy",
    )


def _validate_local_vacuum_root_certificate(
    certificate: GalerkinLocalVacuumRootCertificate,
) -> GalerkinLocalVacuumRootCertificate:
    """PRIVATE: Validate one raw strict root-classification carrier.

    Parameters
    ----------
    certificate : GalerkinLocalVacuumRootCertificate
        Raw root evidence storage.

    Returns
    -------
    validated : GalerkinLocalVacuumRootCertificate
        Structurally validated root evidence.

    Raises
    ------
    TypeError
        If nested carriers or the classification enum have wrong types.
    ValueError
        If strict classification, formulas, work, or digests disagree.
    """
    if not isinstance(certificate, GalerkinLocalVacuumRootCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalVacuumRootCertificate"
        )
    if not isinstance(
        certificate.classification, GalerkinLocalVacuumRootClass
    ):
        raise TypeError("classification has the wrong root-class enum")
    q_interval = _validate_local_vacuum_rational_interval(
        certificate.q_interval
    )
    witness = certificate.zero_witness
    if witness is not None:
        _validate_local_vacuum_zero_witness(witness)
    root = certificate.root_interval
    if root is not None:
        _validate_local_vacuum_rational_interval(root)
    work = _validate_local_vacuum_work_transcript(certificate.work_transcript)
    zero_contained = q_interval.lower <= 0 <= q_interval.upper
    if certificate.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
        _raise_if(
            q_interval.lower <= 0
            or witness is not None
            or root is None
            or root.lower <= 0,
            "propagating root must have q_lower>0 and a positive root",
        )
    elif certificate.classification is GalerkinLocalVacuumRootClass.EVANESCENT:
        _raise_if(
            q_interval.upper >= 0
            or witness is not None
            or root is None
            or root.lower <= 0,
            "evanescent root must have q_upper<0 and a positive root",
        )
    elif certificate.classification is GalerkinLocalVacuumRootClass.GRAZING:
        _raise_if(
            not zero_contained
            or witness is None
            or root is None
            or root.lower != 0
            or root.upper != 0,
            "grazing root requires a consistent witness and exact zero root",
        )
    else:
        _raise_if(
            not zero_contained or witness is not None or root is not None,
            "unclassified root must straddle zero without a witness",
        )
    expected_root_calls = (
        2
        if certificate.classification
        in (
            GalerkinLocalVacuumRootClass.PROPAGATING,
            GalerkinLocalVacuumRootClass.EVANESCENT,
        )
        else 0
    )
    _raise_if(
        work.algorithm != "exact_fraction_local_vacuum_root_v1"
        or work.root_enclosures != expected_root_calls,
        "root work transcript is inconsistent with its classification",
    )
    _raise_if(
        certificate.classification_formula != _ROOT_CLASSIFICATION_FORMULA
        or certificate.root_formula != _ROOT_FORMULA
        or certificate.witness_scope != _WITNESS_SCOPE
        or certificate.completion_scope != _ROOT_COMPLETION_SCOPE,
        "root declaration or completion scope is not canonical",
    )
    _raise_if(
        not _valid_digest(certificate.root_identity_digest)
        or not _valid_digest(certificate.root_evidence_digest),
        "root digests must be canonical SHA-256",
    )
    validated: GalerkinLocalVacuumRootCertificate = certificate
    return validated


def _validate_local_vacuum_propagator(
    propagator: GalerkinLocalVacuumPropagator,
) -> GalerkinLocalVacuumPropagator:
    """PRIVATE: Validate one raw homogeneous propagator carrier.

    Parameters
    ----------
    propagator : GalerkinLocalVacuumPropagator
        Raw propagator evidence storage.

    Returns
    -------
    validated : GalerkinLocalVacuumPropagator
        Structurally validated propagator storage.

    Raises
    ------
    TypeError
        If nested carriers or helper evidence have wrong types.
    ValueError
        If entries, distance, formulas, policies, or digests disagree.
    """
    if not isinstance(propagator, GalerkinLocalVacuumPropagator):
        raise TypeError("propagator must be GalerkinLocalVacuumPropagator")
    root = _validate_local_vacuum_root_certificate(propagator.root_certificate)
    _raise_if(
        root.classification is GalerkinLocalVacuumRootClass.UNCLASSIFIED,
        "unclassified root cannot own a propagator",
    )
    _raise_if(
        not isinstance(propagator.entries, tuple)
        or len(propagator.entries) != _PROPAGATOR_ENTRY_COUNT,
        "propagator must contain four row-major intervals",
    )
    entries = tuple(
        _validate_local_vacuum_rational_interval(value)
        for value in propagator.entries
    )
    distance_values = (
        propagator.distance_numerator,
        propagator.distance_denominator,
    )
    _raise_if(
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in distance_values
        )
        or propagator.distance_denominator <= 0
        or math.gcd(
            abs(propagator.distance_numerator),
            propagator.distance_denominator,
        )
        != 1,
        "propagator distance must be one reduced rational",
    )
    _raise_if(
        propagator.distance < 0,
        "propagator distance must be nonnegative",
    )
    work = _validate_local_vacuum_work_transcript(
        propagator.interval_work_transcript
    )
    _raise_if(
        not _valid_policy(propagator.precision_bits)
        or not _valid_policy(propagator.maximum_terms)
        or not _valid_policy(propagator.maximum_entire_work)
        or not _valid_policy(
            propagator.maximum_range_reductions, allow_zero=True
        )
        or propagator.precision_bits + 1 > work.maximum_rational_bits,
        "propagator entire-helper policies are invalid",
    )
    exact_identity = (
        entries[0].lower == entries[0].upper == 1
        and entries[1].lower == entries[1].upper == 0
        and entries[2].lower == entries[2].upper == 0
        and entries[3].lower == entries[3].upper == 1
    )
    if propagator.distance == 0:
        _raise_if(
            not exact_identity
            or propagator.entire_transcript is not None
            or work.exact_work_count != 0,
            "zero-distance propagator must be the symbolic identity",
        )
    elif root.classification is GalerkinLocalVacuumRootClass.GRAZING:
        _raise_if(
            entries[0].lower != 1
            or entries[0].upper != 1
            or entries[1].lower != propagator.distance
            or entries[1].upper != propagator.distance
            or entries[2].lower != 0
            or entries[2].upper != 0
            or entries[3].lower != 1
            or entries[3].upper != 1
            or propagator.entire_transcript is not None
            or work.exact_work_count != 0,
            "grazing propagator must be the exact affine matrix",
        )
    else:
        if propagator.entire_transcript is None:
            raise ValueError(
                "nonzero nongrazing propagator needs helper evidence"
            )
        _validate_entire_transcript(propagator.entire_transcript)
        expected_algorithm = (
            "exact_fraction_real_sin_cos_v1"
            if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING
            else "exact_fraction_real_sinh_cosh_v1"
        )
        _raise_if(
            propagator.entire_transcript.algorithm != expected_algorithm
            or propagator.entire_transcript.precision_bits
            != propagator.precision_bits
            or propagator.entire_transcript.maximum_terms
            != propagator.maximum_terms
            or propagator.entire_transcript.maximum_work
            != propagator.maximum_entire_work
            or propagator.entire_transcript.maximum_range_reductions
            != propagator.maximum_range_reductions
            or propagator.entire_transcript.maximum_rational_bits
            != work.maximum_rational_bits
            or work.exact_work_count == 0,
            "propagator helper or interval-work transcript is inconsistent",
        )
    expected_formula = {
        GalerkinLocalVacuumRootClass.PROPAGATING: _PROPAGATING_FORMULA,
        GalerkinLocalVacuumRootClass.EVANESCENT: _EVANESCENT_FORMULA,
        GalerkinLocalVacuumRootClass.GRAZING: _GRAZING_FORMULA,
    }[root.classification]
    _raise_if(
        propagator.propagator_formula != expected_formula
        or propagator.trust_scope != _WITNESS_SCOPE
        or propagator.completion_scope != _PROPAGATOR_COMPLETION_SCOPE,
        "propagator formula, trust scope, or completion scope is not "
        "canonical",
    )
    _raise_if(
        not _valid_digest(propagator.propagator_identity_digest)
        or not _valid_digest(propagator.propagator_evidence_digest),
        "propagator digests must be canonical SHA-256",
    )
    validated: GalerkinLocalVacuumPropagator = propagator
    return validated


def _make_local_vacuum_rational_interval(
    lower: Fraction,
    upper: Fraction,
) -> GalerkinLocalVacuumRationalInterval:
    """PRIVATE: Construct one validated exact rational interval.

    Parameters
    ----------
    lower : Fraction
        Exact lower endpoint.
    upper : Fraction
        Exact upper endpoint.

    Returns
    -------
    validated : GalerkinLocalVacuumRationalInterval
        Validated exact interval carrier.

    Raises
    ------
    TypeError
        If either endpoint is not a Fraction.
    ValueError
        If endpoints are reversed or exceed the hard bit cap.
    """
    if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
        raise TypeError("rational interval endpoints must be Fractions")
    interval: GalerkinLocalVacuumRationalInterval = (
        GalerkinLocalVacuumRationalInterval(
            lower_numerator=lower.numerator,
            lower_denominator=lower.denominator,
            upper_numerator=upper.numerator,
            upper_denominator=upper.denominator,
        )
    )
    validated: GalerkinLocalVacuumRationalInterval = (
        _validate_local_vacuum_rational_interval(interval)
    )
    return validated


def _make_local_vacuum_zero_witness(
    left_normal_form: _FormalNormalForm,
    right_normal_form: _FormalNormalForm,
    maximum_rational_bits: int,
    *,
    route: GalerkinLocalVacuumZeroWitnessRoute,
    witness_digest: str,
) -> GalerkinLocalVacuumZeroWitness:
    """PRIVATE: Construct one validated formal zero witness.

    Parameters
    ----------
    left_normal_form : _FormalNormalForm
        Canonical left sparse formal sum.
    right_normal_form : _FormalNormalForm
        Canonical right sparse formal sum.
    maximum_rational_bits : int
        Positive witness coefficient bit policy.
    route : GalerkinLocalVacuumZeroWitnessRoute
        Exact-rational or symbolic normal-form route.
    witness_digest : str
        Complete formal-witness evidence digest.

    Returns
    -------
    validated : GalerkinLocalVacuumZeroWitness
        Validated raw formal witness storage.

    Raises
    ------
    TypeError
        If the route has the wrong enum type.
    ValueError
        If normal forms, policy, declarations, or digest are invalid.
    """
    witness: GalerkinLocalVacuumZeroWitness = GalerkinLocalVacuumZeroWitness(
        left_normal_form=left_normal_form,
        right_normal_form=right_normal_form,
        maximum_rational_bits=maximum_rational_bits,
        route=route,
        witness_formula=_WITNESS_FORMULA,
        trust_scope=_WITNESS_SCOPE,
        witness_digest=witness_digest,
    )
    validated: GalerkinLocalVacuumZeroWitness = (
        _validate_local_vacuum_zero_witness(witness)
    )
    return validated


def _make_local_vacuum_work_transcript(
    *,
    algorithm: str,
    maximum_work: int,
    maximum_rational_bits: int,
    additions: int,
    subtractions: int,
    multiplications: int,
    divisions: int,
    root_enclosures: int,
    exact_work_count: int,
) -> GalerkinLocalVacuumWorkTranscript:
    """PRIVATE: Construct one validated local exact-work transcript.

    Parameters
    ----------
    algorithm : str
        Canonical local algorithm identifier.
    maximum_work : int
        Positive call-wide exact-operation budget.
    maximum_rational_bits : int
        Positive retained-rational bit policy.
    additions : int
        Number of charged exact additions.
    subtractions : int
        Number of charged exact subtractions.
    multiplications : int
        Number of charged exact multiplications.
    divisions : int
        Number of charged exact divisions.
    root_enclosures : int
        Number of charged rational-root helper calls.
    exact_work_count : int
        Exact sum of all charged operation counters.

    Returns
    -------
    validated : GalerkinLocalVacuumWorkTranscript
        Validated deterministic work evidence.

    Raises
    ------
    ValueError
        If policies, counters, or their exact sum are inconsistent.
    """
    transcript: GalerkinLocalVacuumWorkTranscript = (
        GalerkinLocalVacuumWorkTranscript(
            algorithm=algorithm,
            maximum_work=maximum_work,
            maximum_rational_bits=maximum_rational_bits,
            additions=additions,
            subtractions=subtractions,
            multiplications=multiplications,
            divisions=divisions,
            root_enclosures=root_enclosures,
            exact_work_count=exact_work_count,
        )
    )
    validated: GalerkinLocalVacuumWorkTranscript = (
        _validate_local_vacuum_work_transcript(transcript)
    )
    return validated


def _make_local_vacuum_root_certificate(
    q_interval: GalerkinLocalVacuumRationalInterval,
    zero_witness: GalerkinLocalVacuumZeroWitness | None,
    root_interval: GalerkinLocalVacuumRationalInterval | None,
    work_transcript: GalerkinLocalVacuumWorkTranscript,
    *,
    classification: GalerkinLocalVacuumRootClass,
    root_identity_digest: str,
    root_evidence_digest: str,
) -> GalerkinLocalVacuumRootCertificate:
    """PRIVATE: Construct one validated strict root certificate.

    Parameters
    ----------
    q_interval : GalerkinLocalVacuumRationalInterval
        Exact submitted LVT.39 quantity enclosure.
    zero_witness : GalerkinLocalVacuumZeroWitness | None
        Optional formal algebraic-zero witness.
    root_interval : GalerkinLocalVacuumRationalInterval | None
        Positive branch root, exact zero, or absent unresolved root.
    work_transcript : GalerkinLocalVacuumWorkTranscript
        Exact root-enclosure work evidence.
    classification : GalerkinLocalVacuumRootClass
        Strict branch classification.
    root_identity_digest : str
        Bound q-and-witness identity digest.
    root_evidence_digest : str
        Complete root-certificate evidence digest.

    Returns
    -------
    validated : GalerkinLocalVacuumRootCertificate
        Validated raw root-certificate storage.

    Raises
    ------
    TypeError
        If nested carriers or the classification have wrong types.
    ValueError
        If classification, formulas, work, or digests are inconsistent.
    """
    certificate: GalerkinLocalVacuumRootCertificate = (
        GalerkinLocalVacuumRootCertificate(
            q_interval=q_interval,
            zero_witness=zero_witness,
            root_interval=root_interval,
            work_transcript=work_transcript,
            classification=classification,
            classification_formula=_ROOT_CLASSIFICATION_FORMULA,
            root_formula=_ROOT_FORMULA,
            witness_scope=_WITNESS_SCOPE,
            completion_scope=_ROOT_COMPLETION_SCOPE,
            root_identity_digest=root_identity_digest,
            root_evidence_digest=root_evidence_digest,
        )
    )
    validated: GalerkinLocalVacuumRootCertificate = (
        _validate_local_vacuum_root_certificate(certificate)
    )
    return validated


def _make_local_vacuum_propagator(  # noqa: PLR0913
    root_certificate: GalerkinLocalVacuumRootCertificate,
    entries: _PropagatorEntries,
    entire_transcript: EntireWorkTranscript | None,
    interval_work_transcript: GalerkinLocalVacuumWorkTranscript,
    distance: Fraction,
    precision_bits: int,
    maximum_terms: int,
    maximum_entire_work: int,
    maximum_range_reductions: int,
    *,
    propagator_formula: str,
    propagator_identity_digest: str,
    propagator_evidence_digest: str,
) -> GalerkinLocalVacuumPropagator:
    """PRIVATE: Construct one validated homogeneous propagator carrier.

    Parameters
    ----------
    root_certificate : GalerkinLocalVacuumRootCertificate
        Fully replayed strict root certificate.
    entries : _PropagatorEntries
        Four exact row-major real interval entries.
    entire_transcript : EntireWorkTranscript | None
        Exact helper evidence, absent for symbolic identity or grazing routes.
    interval_work_transcript : GalerkinLocalVacuumWorkTranscript
        Exact post-helper interval-operation evidence.
    distance : Fraction
        Exact nonnegative propagation distance.
    precision_bits : int
        Entire-helper direct-series remainder target bits.
    maximum_terms : int
        Per-series entire-helper term budget.
    maximum_entire_work : int
        Call-wide entire-helper exact-work budget.
    maximum_range_reductions : int
        Per-scalar entire-helper dyadic-reduction limit.
    propagator_formula : str
        Branch-specific LVT.41, LVT.42, or LVT.43 formula.
    propagator_identity_digest : str
        Bound root/distance/formula identity digest.
    propagator_evidence_digest : str
        Complete propagator evidence digest.

    Returns
    -------
    validated : GalerkinLocalVacuumPropagator
        Validated raw homogeneous propagator storage.

    Raises
    ------
    TypeError
        If nested carriers, entries, transcript, or distance have wrong types.
    ValueError
        If matrix, policies, formulas, scope, or digests are inconsistent.
    """
    if not isinstance(distance, Fraction):
        raise TypeError("distance must be a Fraction")
    propagator: GalerkinLocalVacuumPropagator = GalerkinLocalVacuumPropagator(
        root_certificate=root_certificate,
        entries=entries,
        entire_transcript=entire_transcript,
        interval_work_transcript=interval_work_transcript,
        distance_numerator=distance.numerator,
        distance_denominator=distance.denominator,
        precision_bits=precision_bits,
        maximum_terms=maximum_terms,
        maximum_entire_work=maximum_entire_work,
        maximum_range_reductions=maximum_range_reductions,
        propagator_formula=propagator_formula,
        trust_scope=_WITNESS_SCOPE,
        completion_scope=_PROPAGATOR_COMPLETION_SCOPE,
        propagator_identity_digest=propagator_identity_digest,
        propagator_evidence_digest=propagator_evidence_digest,
    )
    validated: GalerkinLocalVacuumPropagator = (
        _validate_local_vacuum_propagator(propagator)
    )
    return validated


__all__: list[str] = [
    "GalerkinLocalVacuumPropagationError",
    "GalerkinLocalVacuumPropagationFailure",
    "GalerkinLocalVacuumPropagator",
    "GalerkinLocalVacuumRationalInterval",
    "GalerkinLocalVacuumRootCertificate",
    "GalerkinLocalVacuumRootClass",
    "GalerkinLocalVacuumWorkTranscript",
    "GalerkinLocalVacuumZeroWitness",
    "GalerkinLocalVacuumZeroWitnessRoute",
]
