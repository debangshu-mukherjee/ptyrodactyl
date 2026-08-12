r"""Define bounded local represented-source stability carriers.

Extended Summary
----------------
These disjoint carriers bind one fully replayed ``LOCAL_CELL_LVT1``
represented-source certificate, one submitted algebraic solve result, and one
bounded exact-dyadic residual proof.  The exact L4 physical CAP floor is used
directly; the RM-S2 fixed-linear radius is charged only in the submitted-state
residual lift.

Routine Listings
----------------
:class:`GalerkinLocalStabilityDisposition`
    Distinguish operational success, finite-radius fallback, and rejection.
:class:`GalerkinLocalStabilityFailure`
    Store one typed bounded-checker outcome.
:class:`GalerkinLocalStabilityProof`
    Store exact rational transcripts and outward binary64 reports.
:class:`GalerkinLocalStabilityResult`
    Nest the complete source certificate, solve result, and checked proof.
:class:`GalerkinLocalStabilityRoute`
    Select the exact axial-CAP-floor route.
"""

from __future__ import annotations

import math
from enum import Enum
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float64, Int64

from ptyrodactyl._tools import (
    fraction_upper_float,
    has_subnormal_components,
)

from .born_types import GalerkinSolveResult
from .local_represented_source_types import (
    GalerkinLocalRepresentedSourceCertificate,
)

_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_PROOF_FLAG_COUNT: int = 5
_PROOF_REPORT_COUNT: int = 9
_SHA256_HEX_LENGTH: int = 64
_TRANSCRIPT_PREFIXES: tuple[str, ...] = (
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
    "maximum_state_error",
)


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for one carrier invariant failure.

    Parameters
    ----------
    condition : bool
        Whether the invariant failed.
    message : str
        Failure detail for the caller.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 value.

    Parameters
    ----------
    value : str
        Candidate hexadecimal digest.

    Returns
    -------
    valid : bool
        Whether the candidate is one canonical lowercase SHA-256 digest.
    """
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _normal_or_zero(value: Float64[Array, ""]) -> bool:
    """PRIVATE: Check one finite normal-range scalar or exact zero.

    Parameters
    ----------
    value : Float64[Array, ""]
        Scalar binary64 report.

    Returns
    -------
    valid : bool
        Whether the report is finite and normal or exactly zero.
    """
    return bool(jnp.isfinite(value)) and not bool(
        has_subnormal_components(value)
    )


class GalerkinLocalStabilityRoute(str, Enum):
    """Select the exact axial-CAP-floor route.

    :see: :func:`~.test_local_stability_types.\
test_local_stability_routes_dispositions_and_failures_are_disjoint`
    """

    EXACT_AXIAL_CAP_FLOOR = "local_exact_axial_cap_floor"


class GalerkinLocalStabilityDisposition(str, Enum):
    """Distinguish operational success, finite-radius fallback, and rejection.

    :see: :func:`~.test_local_stability_types.\
test_local_stability_routes_dispositions_and_failures_are_disjoint`
    """

    OPERATIONAL_PASS = "operational_pass"  # noqa: S105
    FINITE_STATE_RADIUS_FALLBACK = "finite_state_radius_fallback"
    MATRIX_FLOOR_ONLY = "matrix_floor_only"
    REJECTED = "rejected"


class GalerkinLocalStabilityFailure(str, Enum):
    """Store one typed bounded-checker outcome.

    :see: :func:`~.test_local_stability_types.\
test_local_stability_routes_dispositions_and_failures_are_disjoint`
    """

    NONE = "none"
    STATE_BUDGET_MISSED = "state_budget_missed"
    SOURCE_NONCERTIFICATE = "source_noncertificate"
    EXACT_TARGET_FLOOR_UNAVAILABLE = "exact_target_floor_unavailable"
    NONPOSITIVE_EXACT_TARGET_FLOOR = "nonpositive_exact_target_floor"
    DIRECT_WORK_BUDGET_EXCEEDED = "direct_work_budget_exceeded"
    DIRECT_WORK_COUNT_OVERFLOW = "direct_work_count_overflow"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalStabilityProof(eqx.Module):
    r"""Store exact rational transcripts and outward binary64 reports.

    ``state_radius_eligible`` is independent of the operational budget.  It
    remains true for ``STATE_BUDGET_MISSED`` so later terminal reasoning can
    consume the finite radius without promoting the solve to an operational
    success.

    The floor and ``delta_H`` have inverse-square Angstrom units. Residuals
    have the corresponding operator-action units, while the state norm,
    radius, and ``maximum_state_error`` use the Euclidean coefficient norm.
    ``maximum_direct_pairs`` bounds only the post-parent-replay dense
    ``n + 2 n**2`` residual construction; target and represented-source
    authentication are outside that resource budget. Raw proof storage is
    unauthenticated until the invocation function reconstructs it.

    :see: :func:`~.test_local_stability_types.\
test_local_stability_carriers_bind_full_parents_and_transcripts`
    """

    lower_singular_bound: Float64[Array, ""]
    algebraic_residual_upper_bound: Float64[Array, ""]
    field_norm_upper_bound: Float64[Array, ""]
    fixed_linear_operator_error_bound: Float64[Array, ""]
    fixed_linear_state_transfer_upper_bound: Float64[Array, ""]
    total_source_error_upper_bound: Float64[Array, ""]
    exact_target_residual_upper_bound: Float64[Array, ""]
    state_radius_upper_bound: Float64[Array, ""]
    maximum_state_error: Float64[Array, ""]
    direct_work_count: Int64[Array, ""]
    maximum_direct_pairs: Int64[Array, ""]
    matrix_floor_eligible: Bool[Array, ""]
    state_radius_eligible: Bool[Array, ""]
    operational_state_eligible: Bool[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    normal_arithmetic_eligible: Bool[Array, ""]
    exact_floor_numerator: int = eqx.field(static=True)
    exact_floor_denominator: int = eqx.field(static=True)
    residual_squared_numerator: int = eqx.field(static=True)
    residual_squared_denominator: int = eqx.field(static=True)
    field_norm_squared_numerator: int = eqx.field(static=True)
    field_norm_squared_denominator: int = eqx.field(static=True)
    algebraic_residual_upper_numerator: int = eqx.field(static=True)
    algebraic_residual_upper_denominator: int = eqx.field(static=True)
    field_norm_upper_numerator: int = eqx.field(static=True)
    field_norm_upper_denominator: int = eqx.field(static=True)
    fixed_linear_error_numerator: int = eqx.field(static=True)
    fixed_linear_error_denominator: int = eqx.field(static=True)
    fixed_linear_state_transfer_upper_numerator: int = eqx.field(static=True)
    fixed_linear_state_transfer_upper_denominator: int = eqx.field(static=True)
    total_source_error_upper_numerator: int = eqx.field(static=True)
    total_source_error_upper_denominator: int = eqx.field(static=True)
    exact_target_residual_upper_numerator: int = eqx.field(static=True)
    exact_target_residual_upper_denominator: int = eqx.field(static=True)
    state_radius_upper_numerator: int = eqx.field(static=True)
    state_radius_upper_denominator: int = eqx.field(static=True)
    maximum_state_error_numerator: int = eqx.field(static=True)
    maximum_state_error_denominator: int = eqx.field(static=True)
    direct_work_count_exact: str = eqx.field(static=True)
    route: GalerkinLocalStabilityRoute = eqx.field(static=True)
    disposition: GalerkinLocalStabilityDisposition = eqx.field(static=True)
    failure: GalerkinLocalStabilityFailure = eqx.field(static=True)
    checker_id: str = eqx.field(static=True)
    residual_formula: str = eqx.field(static=True)
    residual_lift_formula: str = eqx.field(static=True)
    floor_scope: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)
    result_identity_digest: str = eqx.field(static=True)
    arithmetic_environment_digest: str = eqx.field(static=True)
    proof_evidence_digest: str = eqx.field(static=True)


class GalerkinLocalStabilityResult(eqx.Module):
    """Nest the complete source certificate, solve result, and checked proof.

    Raw result storage is not authoritative. Call
    ``prepare_local_galerkin_stability_result`` after every trust boundary;
    it replays the complete parent certificate, solve payload, proof,
    resource budget, and evidence digests against independently supplied
    state-error and direct-work policy values.

    :see: :func:`~.test_local_stability_types.\
test_local_stability_carriers_bind_full_parents_and_transcripts`
    """

    certificate: GalerkinLocalRepresentedSourceCertificate
    solve_result: GalerkinSolveResult
    proof: GalerkinLocalStabilityProof
    result_identity_digest: str = eqx.field(static=True)
    result_evidence_digest: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)

    @property
    def lower_singular_bound(self) -> Float64[Array, ""]:
        """Return the exact-target L4 floor report."""
        return self.proof.lower_singular_bound

    @property
    def residual_upper_bound(self) -> Float64[Array, ""]:
        """Return the lifted exact-target residual bound."""
        return self.proof.exact_target_residual_upper_bound

    @property
    def state_radius_upper_bound(self) -> Float64[Array, ""]:
        """Return the finite state-radius report or positive infinity."""
        return self.proof.state_radius_upper_bound

    @property
    def maximum_state_error(self) -> Float64[Array, ""]:
        """Return the independently supplied operational budget."""
        return self.proof.maximum_state_error

    @property
    def state_radius_eligible(self) -> Bool[Array, ""]:
        """Return whether a finite same-target radius was proved."""
        return self.proof.state_radius_eligible

    @property
    def matrix_floor_eligible(self) -> Bool[Array, ""]:
        """Return whether the replayed exact L4 matrix floor is valid."""
        return self.proof.matrix_floor_eligible

    @property
    def operational_state_eligible(self) -> Bool[Array, ""]:
        """Return whether the radius also meets its operational budget."""
        return self.proof.operational_state_eligible

    @property
    def route(self) -> GalerkinLocalStabilityRoute:
        """Return the checked local stability route."""
        return self.proof.route

    @property
    def disposition(self) -> GalerkinLocalStabilityDisposition:
        """Return the checked invocation disposition."""
        return self.proof.disposition

    @property
    def failure(self) -> GalerkinLocalStabilityFailure:
        """Return the typed checker outcome."""
        return self.proof.failure


def _fraction_field(
    proof: GalerkinLocalStabilityProof, prefix: str
) -> Fraction:
    """PRIVATE: Recover one exact rational transcript from a proof.

    Parameters
    ----------
    proof : GalerkinLocalStabilityProof
        Submitted proof carrier.
    prefix : str
        Common numerator and denominator field prefix.

    Returns
    -------
    fraction : Fraction
        Reduced exact rational transcript.
    """
    numerator = getattr(proof, f"{prefix}_numerator")
    denominator = getattr(proof, f"{prefix}_denominator")
    return Fraction(numerator, denominator)


def _validate_local_stability_proof(  # noqa: PLR0912,PLR0915
    proof: GalerkinLocalStabilityProof,
) -> GalerkinLocalStabilityProof:
    """PRIVATE: Enforce proof schema and arithmetic invariants.

    Parameters
    ----------
    proof : GalerkinLocalStabilityProof
        Raw proof storage to validate.

    Returns
    -------
    proof : GalerkinLocalStabilityProof
        The same proof after complete structural validation.

    Raises
    ------
    TypeError
        If the carrier or any static outcome enum has the wrong type.
    ValueError
        If scalar, transcript, digest, work, or outcome invariants fail.
    """
    if not isinstance(proof, GalerkinLocalStabilityProof):
        raise TypeError("proof must be GalerkinLocalStabilityProof")
    if not isinstance(proof.route, GalerkinLocalStabilityRoute):
        raise TypeError("route has the wrong local stability enum")
    if not isinstance(proof.disposition, GalerkinLocalStabilityDisposition):
        raise TypeError("disposition has the wrong local stability enum")
    if not isinstance(proof.failure, GalerkinLocalStabilityFailure):
        raise TypeError("failure has the wrong local stability enum")
    scalars = (
        proof.lower_singular_bound,
        proof.algebraic_residual_upper_bound,
        proof.field_norm_upper_bound,
        proof.fixed_linear_operator_error_bound,
        proof.fixed_linear_state_transfer_upper_bound,
        proof.total_source_error_upper_bound,
        proof.exact_target_residual_upper_bound,
        proof.state_radius_upper_bound,
        proof.maximum_state_error,
        proof.direct_work_count,
        proof.maximum_direct_pairs,
        proof.matrix_floor_eligible,
        proof.state_radius_eligible,
        proof.operational_state_eligible,
        proof.host_binary64_eligible,
        proof.normal_arithmetic_eligible,
    )
    _raise_if(
        any(value.shape != () for value in scalars),
        "local stability proof fields must be scalar",
    )
    _raise_if(
        proof.direct_work_count.dtype != jnp.dtype(jnp.int64)
        or proof.maximum_direct_pairs.dtype != jnp.dtype(jnp.int64),
        "local stability work evidence must use int64 scalars",
    )
    _raise_if(
        bool(proof.maximum_direct_pairs <= 0),
        "maximum_direct_pairs must be positive",
    )
    budget = float(np.asarray(proof.maximum_state_error))
    _raise_if(
        not math.isfinite(budget) or budget < float(np.finfo(np.float64).tiny),
        "maximum_state_error must be finite normal positive binary64",
    )
    for prefix in _TRANSCRIPT_PREFIXES:
        numerator = getattr(proof, f"{prefix}_numerator")
        denominator = getattr(proof, f"{prefix}_denominator")
        _raise_if(
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator < 0
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0,
            f"{prefix} transcript must be one nonnegative rational",
        )
    _raise_if(
        _fraction_field(proof, "maximum_state_error")
        != Fraction.from_float(budget),
        "maximum_state_error transcript must match its stored value",
    )
    for text, name in (
        (proof.direct_work_count_exact, "direct_work_count_exact"),
        (proof.checker_id, "checker_id"),
        (proof.residual_formula, "residual_formula"),
        (proof.residual_lift_formula, "residual_lift_formula"),
        (proof.floor_scope, "floor_scope"),
        (proof.error_scope, "error_scope"),
        (proof.coefficient_norm, "coefficient_norm"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    try:
        exact_count = int(proof.direct_work_count_exact)
    except ValueError as error:
        raise ValueError(
            "direct_work_count_exact must be decimal integer"
        ) from error
    _raise_if(exact_count < 0, "direct_work_count_exact cannot be negative")
    stored_count = int(np.asarray(proof.direct_work_count))
    pair_budget = int(np.asarray(proof.maximum_direct_pairs))
    if (
        proof.failure
        is GalerkinLocalStabilityFailure.DIRECT_WORK_COUNT_OVERFLOW
    ):
        _raise_if(
            exact_count <= _MAXIMUM_SIGNED_INT64 or stored_count != 0,
            "work-count overflow must retain an unclamped decimal transcript",
        )
    else:
        _raise_if(
            exact_count > _MAXIMUM_SIGNED_INT64 or stored_count != exact_count,
            "stored work count must equal the exact signed-int64 count",
        )
        matrix_eligible = bool(proof.matrix_floor_eligible)
        environment_supported = bool(proof.host_binary64_eligible) and bool(
            proof.normal_arithmetic_eligible
        )
        if matrix_eligible and not environment_supported:
            expected_host_failure = (
                GalerkinLocalStabilityFailure.HOST_ARITHMETIC_UNSUPPORTED
            )
            _raise_if(
                proof.failure is not expected_host_failure,
                "unsupported arithmetic requires the typed host failure",
            )
        elif matrix_eligible and stored_count > pair_budget:
            expected_budget_failure = (
                GalerkinLocalStabilityFailure.DIRECT_WORK_BUDGET_EXCEEDED
            )
            _raise_if(
                proof.failure is not expected_budget_failure,
                "excess direct work requires the typed budget failure",
            )
        elif matrix_eligible:
            _raise_if(
                proof.failure
                in (
                    GalerkinLocalStabilityFailure.HOST_ARITHMETIC_UNSUPPORTED,
                    GalerkinLocalStabilityFailure.DIRECT_WORK_BUDGET_EXCEEDED,
                ),
                "preflight failure is inconsistent with its evidence",
            )
    for digest, name in (
        (proof.target_digest, "target_digest"),
        (proof.source_digest, "source_digest"),
        (proof.certificate_digest, "certificate_digest"),
        (proof.result_identity_digest, "result_identity_digest"),
        (
            proof.arithmetic_environment_digest,
            "arithmetic_environment_digest",
        ),
        (proof.proof_evidence_digest, "proof_evidence_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")
    radius_eligible = bool(proof.state_radius_eligible)
    matrix_eligible = bool(proof.matrix_floor_eligible)
    operational = bool(proof.operational_state_eligible)
    _raise_if(
        operational and not radius_eligible, "operational implies radius"
    )
    _raise_if(
        radius_eligible and not matrix_eligible,
        "finite state radius requires a valid matrix floor",
    )
    if radius_eligible:
        _raise_if(
            proof.failure
            not in (
                GalerkinLocalStabilityFailure.NONE,
                GalerkinLocalStabilityFailure.STATE_BUDGET_MISSED,
            ),
            "finite radius has an inconsistent failure outcome",
        )
        expected_disposition = (
            GalerkinLocalStabilityDisposition.OPERATIONAL_PASS
            if operational
            else GalerkinLocalStabilityDisposition.FINITE_STATE_RADIUS_FALLBACK
        )
        _raise_if(
            proof.disposition is not expected_disposition,
            "finite radius has an inconsistent disposition",
        )
        finite_reports = (
            proof.lower_singular_bound,
            proof.algebraic_residual_upper_bound,
            proof.field_norm_upper_bound,
            proof.fixed_linear_operator_error_bound,
            proof.fixed_linear_state_transfer_upper_bound,
            proof.total_source_error_upper_bound,
            proof.exact_target_residual_upper_bound,
            proof.state_radius_upper_bound,
        )
        _raise_if(
            any(not _normal_or_zero(value) for value in finite_reports)
            or bool(proof.lower_singular_bound <= 0.0)
            or any(bool(value < 0.0) for value in finite_reports),
            "eligible local stability reports must be finite normal-or-zero",
        )
        report_pairs = (
            (proof.lower_singular_bound, "exact_floor"),
            (proof.algebraic_residual_upper_bound, "algebraic_residual_upper"),
            (proof.field_norm_upper_bound, "field_norm_upper"),
            (proof.fixed_linear_operator_error_bound, "fixed_linear_error"),
            (
                proof.fixed_linear_state_transfer_upper_bound,
                "fixed_linear_state_transfer_upper",
            ),
            (proof.total_source_error_upper_bound, "total_source_error_upper"),
            (
                proof.exact_target_residual_upper_bound,
                "exact_target_residual_upper",
            ),
            (proof.state_radius_upper_bound, "state_radius_upper"),
        )
        _raise_if(
            any(
                Fraction.from_float(float(np.asarray(value)))
                != _fraction_field(proof, prefix)
                for value, prefix in report_pairs
            ),
            "eligible reports must match their exact rational transcripts",
        )
        rho = _fraction_field(proof, "algebraic_residual_upper")
        xnorm = _fraction_field(proof, "field_norm_upper")
        delta_h = _fraction_field(proof, "fixed_linear_error")
        transfer = _fraction_field(proof, "fixed_linear_state_transfer_upper")
        eta_t = _fraction_field(proof, "total_source_error_upper")
        residual = _fraction_field(proof, "exact_target_residual_upper")
        floor = _fraction_field(proof, "exact_floor")
        radius = _fraction_field(proof, "state_radius_upper")
        _raise_if(
            rho * rho < _fraction_field(proof, "residual_squared")
            or xnorm * xnorm < _fraction_field(proof, "field_norm_squared"),
            "root reports do not dominate exact squared transcripts",
        )
        _raise_if(
            transfer < delta_h * xnorm,
            "fixed-linear state transfer is not outward",
        )
        lifted = Fraction.from_float(
            fraction_upper_float(rho + transfer + eta_t)
        )
        quotient = Fraction.from_float(fraction_upper_float(residual / floor))
        _raise_if(
            residual != lifted or radius != quotient,
            "residual lift or state-radius quotient is noncanonical",
        )
        budget_fraction = _fraction_field(proof, "maximum_state_error")
        budget_passes = radius <= budget_fraction
        _raise_if(
            operational != budget_passes
            or (proof.failure is GalerkinLocalStabilityFailure.NONE)
            != budget_passes,
            "operational state-budget outcome is inconsistent",
        )
        _raise_if(
            stored_count > pair_budget,
            "eligible proof exceeds its direct-work budget",
        )
    else:
        expected_disposition = (
            GalerkinLocalStabilityDisposition.MATRIX_FLOOR_ONLY
            if matrix_eligible
            else GalerkinLocalStabilityDisposition.REJECTED
        )
        _raise_if(
            proof.disposition is not expected_disposition or operational,
            "matrix-only/rejected proof has an inconsistent disposition",
        )
        _raise_if(
            proof.failure
            in (
                GalerkinLocalStabilityFailure.NONE,
                GalerkinLocalStabilityFailure.STATE_BUDGET_MISSED,
            ),
            "rejected proof requires a rejection failure",
        )
        floor_value = float(np.asarray(proof.lower_singular_bound))
        floor = _fraction_field(proof, "exact_floor")
        if matrix_eligible:
            _raise_if(
                not _normal_or_zero(proof.lower_singular_bound)
                or floor_value <= 0.0
                or Fraction.from_float(floor_value) != floor,
                "matrix-only proof must preserve its exact positive L4 floor",
            )
            _raise_if(
                proof.failure
                in (
                    GalerkinLocalStabilityFailure.EXACT_TARGET_FLOOR_UNAVAILABLE,
                    GalerkinLocalStabilityFailure.NONPOSITIVE_EXACT_TARGET_FLOOR,
                ),
                "matrix-only outcome cannot report an invalid floor",
            )
        else:
            _raise_if(
                floor_value != 0.0 or floor != 0,
                "matrix-rejected proof must use the zero floor sentinel",
            )
        _raise_if(
            not math.isinf(
                float(np.asarray(proof.exact_target_residual_upper_bound))
            )
            or not math.isinf(
                float(np.asarray(proof.state_radius_upper_bound))
            ),
            "non-radius proof must use fail-closed state sentinels",
        )
    return proof


def _make_local_stability_proof(  # noqa: PLR0913
    reports: tuple[Float64[Array, ""], ...],
    direct_work_count: Int64[Array, ""],
    maximum_direct_pairs: Int64[Array, ""],
    flags: tuple[Bool[Array, ""], ...],
    transcript_fields: dict[str, int],
    *,
    direct_work_count_exact: str,
    route: GalerkinLocalStabilityRoute,
    disposition: GalerkinLocalStabilityDisposition,
    failure: GalerkinLocalStabilityFailure,
    checker_id: str,
    residual_formula: str,
    residual_lift_formula: str,
    floor_scope: str,
    error_scope: str,
    coefficient_norm: str,
    target_digest: str,
    source_digest: str,
    certificate_digest: str,
    result_identity_digest: str,
    arithmetic_environment_digest: str,
    proof_evidence_digest: str,
) -> GalerkinLocalStabilityProof:
    """PRIVATE: Construct and validate one local stability proof.

    Parameters
    ----------
    reports : tuple[Float64[Array, ""], ...]
        Nine ordered floor, residual, norm, transfer, source, and radius
        reports.
    direct_work_count : Int64[Array, ""]
        Stored exact work count, or zero for typed count overflow.
    maximum_direct_pairs : Int64[Array, ""]
        Independent signed-int64 dense-work budget.
    flags : tuple[Bool[Array, ""], ...]
        Five ordered matrix, radius, operational, host, and normal-arithmetic
        predicates.
    transcript_fields : dict[str, int]
        Complete rational numerator and denominator constructor fields.
    direct_work_count_exact : str
        Unclamped exact decimal work count.
    route : GalerkinLocalStabilityRoute
        Local exact-floor route.
    disposition : GalerkinLocalStabilityDisposition
        Matrix/state operational disposition.
    failure : GalerkinLocalStabilityFailure
        Typed checker outcome.
    checker_id : str
        Canonical checker implementation identifier.
    residual_formula : str
        Exact algebraic residual declaration.
    residual_lift_formula : str
        Exact-target lift declaration.
    floor_scope : str
        Exact L4 floor inclusion and exclusion scope.
    error_scope : str
        Residual-error inclusion and exclusion scope.
    coefficient_norm : str
        Retained-state norm declaration.
    target_digest : str
        Bound local target identity digest.
    source_digest : str
        Bound represented-source identity digest.
    certificate_digest : str
        Bound represented-source certificate digest.
    result_identity_digest : str
        Target/source/submitted-field identity digest.
    arithmetic_environment_digest : str
        Bound arithmetic-probe digest.
    proof_evidence_digest : str
        Complete proof-evidence digest.

    Returns
    -------
    proof : GalerkinLocalStabilityProof
        Validated local stability proof carrier.

    Raises
    ------
    TypeError
        If an enum or carrier has the wrong route-specific type.
    ValueError
        If shapes, transcripts, work, outcomes, or reports are inconsistent.
    """
    _raise_if(
        len(reports) != _PROOF_REPORT_COUNT,
        "proof requires nine ordered reports",
    )
    _raise_if(
        len(flags) != _PROOF_FLAG_COUNT,
        "proof requires five ordered predicates",
    )
    proof = GalerkinLocalStabilityProof(
        lower_singular_bound=reports[0],
        algebraic_residual_upper_bound=reports[1],
        field_norm_upper_bound=reports[2],
        fixed_linear_operator_error_bound=reports[3],
        fixed_linear_state_transfer_upper_bound=reports[4],
        total_source_error_upper_bound=reports[5],
        exact_target_residual_upper_bound=reports[6],
        state_radius_upper_bound=reports[7],
        maximum_state_error=reports[8],
        direct_work_count=direct_work_count,
        maximum_direct_pairs=maximum_direct_pairs,
        matrix_floor_eligible=flags[0],
        state_radius_eligible=flags[1],
        operational_state_eligible=flags[2],
        host_binary64_eligible=flags[3],
        normal_arithmetic_eligible=flags[4],
        **transcript_fields,
        direct_work_count_exact=direct_work_count_exact,
        route=route,
        disposition=disposition,
        failure=failure,
        checker_id=checker_id,
        residual_formula=residual_formula,
        residual_lift_formula=residual_lift_formula,
        floor_scope=floor_scope,
        error_scope=error_scope,
        coefficient_norm=coefficient_norm,
        target_digest=target_digest,
        source_digest=source_digest,
        certificate_digest=certificate_digest,
        result_identity_digest=result_identity_digest,
        arithmetic_environment_digest=arithmetic_environment_digest,
        proof_evidence_digest=proof_evidence_digest,
    )
    return _validate_local_stability_proof(proof)


def _make_local_stability_result(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
    proof: GalerkinLocalStabilityProof,
    *,
    result_identity_digest: str,
    result_evidence_digest: str,
    completion_scope: str,
) -> GalerkinLocalStabilityResult:
    """PRIVATE: Validate and store one complete local stability result.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Complete replayed represented-source certificate.
    solve_result : GalerkinSolveResult
        Submitted algebraic solve payload.
    proof : GalerkinLocalStabilityProof
        Fully reconstructed local stability proof.
    result_identity_digest : str
        Canonical target/source/field identity digest.
    result_evidence_digest : str
        Canonical digest of the complete nested result evidence.
    completion_scope : str
        Explicit statement of the certified and excluded claims.

    Returns
    -------
    result : GalerkinLocalStabilityResult
        Validated complete local stability result.

    Raises
    ------
    TypeError
        If a nested carrier has the wrong route-specific type.
    ValueError
        If nested identities, evidence, or completion scope disagree.
    """
    if not isinstance(certificate, GalerkinLocalRepresentedSourceCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalRepresentedSourceCertificate"
        )
    if not isinstance(solve_result, GalerkinSolveResult):
        raise TypeError("solve_result must be GalerkinSolveResult")
    checked = _validate_local_stability_proof(proof)
    _raise_if(
        result_identity_digest != checked.result_identity_digest,
        "result identity must match the checked proof",
    )
    _raise_if(
        checked.target_digest != certificate.source.target.target_digest
        or checked.source_digest != certificate.source.source_digest
        or checked.certificate_digest != certificate.certificate_digest,
        "proof must bind the nested represented-source certificate",
    )
    _raise_if(
        not _valid_digest(result_evidence_digest),
        "result_evidence_digest must be SHA-256",
    )
    _raise_if(
        not completion_scope.strip(), "completion_scope must be nonempty"
    )
    return GalerkinLocalStabilityResult(
        certificate=certificate,
        solve_result=solve_result,
        proof=checked,
        result_identity_digest=result_identity_digest,
        result_evidence_digest=result_evidence_digest,
        completion_scope=completion_scope.strip(),
    )


__all__: list[str] = [
    "GalerkinLocalStabilityDisposition",
    "GalerkinLocalStabilityFailure",
    "GalerkinLocalStabilityProof",
    "GalerkinLocalStabilityResult",
    "GalerkinLocalStabilityRoute",
]
