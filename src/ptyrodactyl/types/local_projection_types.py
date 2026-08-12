r"""Define exact local projection-defect evidence carriers.

Extended Summary
----------------
These carriers bind one fully replayed exact zero slab to one fully replayed
same-source local-stability result.  They keep the structural singleton-zero
free-diagonal witness independent of finite LVT.55d--LVT.55e projection
bounds and of the caller's operational state-error policy.

Routine Listings
----------------
:class:`GalerkinLocalProjectionDefectCertificate`
    Store replayable LVT.34--LVT.40 and LVT.55c--LVT.55e evidence.
:class:`GalerkinLocalProjectionDefectFailure`
    Enumerate simultaneous structural, policy, and arithmetic outcomes.
"""

from __future__ import annotations

import math
from enum import IntFlag
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float64, Int64

from ptyrodactyl._tools import has_subnormal_components, stored_value_payload

from .local_stability_types import GalerkinLocalStabilityResult
from .local_terminal_types import GalerkinLocalTerminalScope
from .local_zero_slab_types import GalerkinLocalZeroSlabCertificate

_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_PI_TARGET_BITS: int = 224
_TRANSVERSE_DIMENSIONS: int = 2
_SINE_TAYLOR_LOWER_LAST_INDEX: int = 21
_SINE_TAYLOR_UPPER_LAST_INDEX: int = 20
_SQRT_PRECISION_BITS: int = 128
_SHA256_HEX_LENGTH: int = 64

type _StateEvidence = tuple[
    Int64[Array, " n"],
    Bool[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
    Bool[Array, " n"],
]
type _GramEvidence = tuple[
    Float64[Array, "n n"],
    Float64[Array, "n n"],
    Float64[Array, "n n"],
    Float64[Array, "n n"],
]
type _FiberEvidence = tuple[
    Bool[Array, " f"],
    Float64[Array, " f"],
    Float64[Array, " f"],
    Float64[Array, " f"],
    Float64[Array, " f"],
    Float64[Array, " f"],
    Float64[Array, " f"],
    Float64[Array, " f"],
]
type _PolicyEvidence = tuple[
    Float64[Array, ""],
    Float64[Array, ""],
    Int64[Array, ""],
    Int64[Array, ""],
    Int64[Array, ""],
    Int64[Array, ""],
    Int64[Array, ""],
    Int64[Array, ""],
    Int64[Array, ""],
]
type _EligibilityEvidence = tuple[
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
    Bool[Array, ""],
]


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise for one structural carrier failure.

    Parameters
    ----------
    condition : bool
        Whether the structural failure is present.
    message : str
        Diagnostic for the rejected carrier.

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
        Whether ``value`` is canonical lowercase SHA-256 text.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


def _normal_or_zero(values: Array) -> bool:
    """PRIVATE: Check finite normal-range array components or exact zeros.

    Parameters
    ----------
    values : Array
        Candidate real binary64 array.

    Returns
    -------
    valid : bool
        Whether every value is finite and non-subnormal.
    """
    valid: bool = bool(jnp.all(jnp.isfinite(values))) and not bool(
        has_subnormal_components(values)
    )
    return valid


class GalerkinLocalProjectionDefectFailure(IntFlag):
    """Enumerate simultaneous structural, policy, and arithmetic outcomes.

    :see: :func:`~.test_local_projection_types.\
test_local_projection_failure_bits_and_carrier_fields_are_disjoint`
    """

    NONE = 0
    ZERO_SLAB_NONCERTIFICATE = 1 << 0
    PARENT_SOURCE_EVIDENCE_MISMATCH = 1 << 1
    STATE_RADIUS_UNAVAILABLE = 1 << 2
    OPERATIONAL_STATE_BUDGET_MISSED = 1 << 3
    TERMINAL_SCOPE_INCOMPLETE = 1 << 4
    STRUCTURAL_EXACT_ZERO_UNAVAILABLE = 1 << 5
    HOST_ARITHMETIC_UNSUPPORTED = 1 << 6
    GRAM_PAIR_BUDGET_EXCEEDED = 1 << 7
    GRAM_PAIR_COUNT_OVERFLOW = 1 << 8
    ROOT_ENCLOSURE_FAILURE = 1 << 9
    ARITHMETIC_RANGE_FAILURE = 1 << 10


class GalerkinLocalProjectionDefectCertificate(eqx.Module):
    r"""Store replayable LVT.34--LVT.40 and LVT.55c--LVT.55e evidence.

    The Gram rectangles are block diagonal in canonical scoped-fiber order.
    ``structural_exact_zero_eligible`` uses only singleton ``[0, 0]`` exact
    free-diagonal intervals.  It never inspects the submitted state for a
    numerical cancellation.  ``finite_projection_bound_eligible`` requires
    a finite L6 state radius, while ``operational_budget_eligible`` also
    requires the independently replayed L6 state-error policy to pass.

    :see: :func:`~.test_local_projection_types.\
test_local_projection_failure_bits_and_carrier_fields_are_disjoint`
    """

    zero_slab_certificate: GalerkinLocalZeroSlabCertificate
    stability_result: GalerkinLocalStabilityResult
    scope_transverse_indices: Int64[Array, "f 2"]
    state_to_fiber_rows: Int64[Array, " n"]
    selected_state_mask: Bool[Array, " n"]
    exact_free_diagonal_lower_bounds: Float64[Array, " n"]
    exact_free_diagonal_upper_bounds: Float64[Array, " n"]
    structural_exact_zero_state_mask: Bool[Array, " n"]
    structural_exact_zero_fiber_mask: Bool[Array, " f"]
    gram_real_lower_bounds: Float64[Array, "n n"]
    gram_real_upper_bounds: Float64[Array, "n n"]
    gram_imag_lower_bounds: Float64[Array, "n n"]
    gram_imag_upper_bounds: Float64[Array, "n n"]
    measured_defect_squared_lower_bounds: Float64[Array, " f"]
    measured_defect_squared_upper_bounds: Float64[Array, " f"]
    measured_defect_upper_bounds: Float64[Array, " f"]
    operator_squared_norm_upper_bounds: Float64[Array, " f"]
    operator_norm_upper_bounds: Float64[Array, " f"]
    state_error_transfer_upper_bounds: Float64[Array, " f"]
    total_defect_upper_bounds: Float64[Array, " f"]
    state_radius_upper_bound: Float64[Array, ""]
    maximum_state_error: Float64[Array, ""]
    direct_pair_count: Int64[Array, ""]
    maximum_gram_pairs: Int64[Array, ""]
    maximum_stability_direct_pairs: Int64[Array, ""]
    pi_target_bits: Int64[Array, ""]
    sine_taylor_lower_last_index: Int64[Array, ""]
    sine_taylor_upper_last_index: Int64[Array, ""]
    sqrt_precision_bits: Int64[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    normal_arithmetic_eligible: Bool[Array, ""]
    structural_exact_zero_eligible: Bool[Array, ""]
    finite_projection_bound_eligible: Bool[Array, ""]
    operational_budget_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    terminal_axis: int = eqx.field(static=True)
    projection_scope: GalerkinLocalTerminalScope = eqx.field(static=True)
    maximum_state_error_numerator: int = eqx.field(static=True)
    maximum_state_error_denominator: int = eqx.field(static=True)
    direct_pair_count_exact: str = eqx.field(static=True)
    gram_formula: str = eqx.field(static=True)
    measurement_formula: str = eqx.field(static=True)
    operator_bound_formula: str = eqx.field(static=True)
    state_lift_formula: str = eqx.field(static=True)
    precision_transcript: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    parent_target_evidence_digest: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    parent_source_evidence_digest: str = eqx.field(static=True)
    parent_represented_certificate_digest: str = eqx.field(static=True)
    parent_zero_slab_certificate_digest: str = eqx.field(static=True)
    parent_stability_result_identity_digest: str = eqx.field(static=True)
    parent_stability_result_evidence_digest: str = eqx.field(static=True)
    state_identity_digest: str = eqx.field(static=True)
    projection_identity_digest: str = eqx.field(static=True)
    arithmetic_environment_digest: str = eqx.field(static=True)
    gram_transcript_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


def _make_local_projection_defect_certificate(  # noqa: PLR0912,PLR0913,PLR0915
    zero_slab_certificate: GalerkinLocalZeroSlabCertificate,
    stability_result: GalerkinLocalStabilityResult,
    scope_transverse_indices: Int64[Array, "f 2"],
    state_evidence: _StateEvidence,
    gram_evidence: _GramEvidence,
    fiber_evidence: _FiberEvidence,
    policy_evidence: _PolicyEvidence,
    eligibility_evidence: _EligibilityEvidence,
    failure_mask: Int64[Array, ""],
    *,
    terminal_axis: int,
    projection_scope: GalerkinLocalTerminalScope,
    maximum_state_error_numerator: int,
    maximum_state_error_denominator: int,
    direct_pair_count_exact: str,
    gram_formula: str,
    measurement_formula: str,
    operator_bound_formula: str,
    state_lift_formula: str,
    precision_transcript: str,
    error_scope: str,
    completion_scope: str,
    target_digest: str,
    parent_target_evidence_digest: str,
    source_digest: str,
    parent_source_evidence_digest: str,
    parent_represented_certificate_digest: str,
    parent_zero_slab_certificate_digest: str,
    parent_stability_result_identity_digest: str,
    parent_stability_result_evidence_digest: str,
    state_identity_digest: str,
    projection_identity_digest: str,
    arithmetic_environment_digest: str,
    gram_transcript_digest: str,
    certificate_digest: str,
) -> GalerkinLocalProjectionDefectCertificate:
    """PRIVATE: Validate one local projection-defect certificate.

    Parameters
    ----------
    zero_slab_certificate : GalerkinLocalZeroSlabCertificate
        Fully replayed exact zero-slab parent.
    stability_result : GalerkinLocalStabilityResult
        Fully replayed same-source state-radius parent.
    scope_transverse_indices : Int64[Array, "f 2"]
        Canonically ordered complete transverse fibers.
    state_evidence : _StateEvidence
        Fiber rows, selection, exact-D intervals, and singleton-zero mask.
    gram_evidence : _GramEvidence
        Outward binary64 real and imaginary Gram rectangles.
    fiber_evidence : _FiberEvidence
        Structural mask and seven per-fiber outward reports.
    policy_evidence : _PolicyEvidence
        State radius, state budget, pair policies, and precision values.
    eligibility_evidence : _EligibilityEvidence
        Host, arithmetic, structural, finite, and operational predicates.
    failure_mask : Int64[Array, ""]
        Simultaneous typed projection-defect outcomes.
    terminal_axis : int
        Target-owned physical xyz terminal axis.
    projection_scope : GalerkinLocalTerminalScope
        Full-state or selected-preterminal complete-fiber scope.
    maximum_state_error_numerator : int
        Exact binary64 state-policy numerator.
    maximum_state_error_denominator : int
        Exact positive binary64 state-policy denominator.
    direct_pair_count_exact : str
        Arbitrary-precision decimal block-square work transcript.
    gram_formula : str
        Exact LVT.55c construction declaration.
    measurement_formula : str
        Exact-rectangle LVT.55d declaration.
    operator_bound_formula : str
        Verified row-sum LVT.55e upper-bound declaration.
    state_lift_formula : str
        Nonduplicated measured-plus-state-radius formula.
    precision_transcript : str
        Fixed rational trigonometric and square-root precision declaration.
    error_scope : str
        Exact list of included and excluded error terms.
    completion_scope : str
        Explicit downstream exclusions.
    target_digest : str
        Bound local target identity digest.
    parent_target_evidence_digest : str
        Bound complete target evidence digest.
    source_digest : str
        Bound represented-source identity digest.
    parent_source_evidence_digest : str
        Bound represented-source evidence digest.
    parent_represented_certificate_digest : str
        Bound represented-source direct-certificate digest.
    parent_zero_slab_certificate_digest : str
        Bound exact zero-slab evidence digest.
    parent_stability_result_identity_digest : str
        Bound same-target/source/state identity digest.
    parent_stability_result_evidence_digest : str
        Bound full local-stability result digest.
    state_identity_digest : str
        Explicit target/source/submitted-state identity digest.
    projection_identity_digest : str
        Slab, scope, target, source, and state identity digest.
    arithmetic_environment_digest : str
        Bound host arithmetic-probe digest.
    gram_transcript_digest : str
        Bound exact-rational Gram transcript digest.
    certificate_digest : str
        Complete projection-defect evidence digest.

    Returns
    -------
    certificate : GalerkinLocalProjectionDefectCertificate
        Structurally validated raw carrier.

    Raises
    ------
    TypeError
        If a nested parent or scope enum has the wrong type.
    ValueError
        If shapes, dtypes, masks, policies, predicates, or digests disagree.
    """
    if not isinstance(zero_slab_certificate, GalerkinLocalZeroSlabCertificate):
        raise TypeError("zero_slab_certificate has the wrong carrier type")
    if not isinstance(stability_result, GalerkinLocalStabilityResult):
        raise TypeError("stability_result has the wrong carrier type")
    if not isinstance(projection_scope, GalerkinLocalTerminalScope):
        raise TypeError("projection_scope has the wrong terminal-scope enum")

    fibers = jnp.asarray(scope_transverse_indices)
    rows, selected, free_lower, free_upper, state_zero = (
        jnp.asarray(value) for value in state_evidence
    )
    gram = tuple(jnp.asarray(value) for value in gram_evidence)
    fiber_zero, *fiber_reports_values = (
        jnp.asarray(value) for value in fiber_evidence
    )
    fiber_reports = tuple(fiber_reports_values)
    policies = tuple(jnp.asarray(value) for value in policy_evidence)
    flags = tuple(jnp.asarray(value) for value in eligibility_evidence)
    submitted_failure = jnp.asarray(failure_mask)

    target = zero_slab_certificate.represented_source_certificate.source.target
    state_size = target.state_indices.shape[0]
    fiber_size = (
        fibers.shape[0] if fibers.ndim == _TRANSVERSE_DIMENSIONS else 0
    )
    _raise_if(
        fibers.dtype != jnp.dtype(jnp.int64)
        or fibers.ndim != _TRANSVERSE_DIMENSIONS
        or fibers.shape[1:] != (_TRANSVERSE_DIMENSIONS,)
        or fiber_size == 0,
        "projection fibers must be nonempty int64 (f, 2)",
    )
    _raise_if(
        rows.dtype != jnp.dtype(jnp.int64)
        or rows.shape != (state_size,)
        or selected.dtype != jnp.dtype(jnp.bool_)
        or selected.shape != (state_size,),
        "projection state mapping must match target I_u",
    )
    _raise_if(
        free_lower.dtype != jnp.dtype(jnp.float64)
        or free_upper.dtype != jnp.dtype(jnp.float64)
        or free_lower.shape != (state_size,)
        or free_upper.shape != (state_size,)
        or state_zero.dtype != jnp.dtype(jnp.bool_)
        or state_zero.shape != (state_size,),
        "exact free-diagonal evidence must be float64/bool on I_u",
    )
    _raise_if(
        bool(jnp.any(~jnp.isfinite(free_lower)))
        or bool(jnp.any(~jnp.isfinite(free_upper)))
        or bool(jnp.any(free_lower > free_upper)),
        "exact free-diagonal intervals must be finite and ordered",
    )
    target_ledger = target.fixed_linear_error_ledger
    _raise_if(
        not np.array_equal(
            np.asarray(free_lower),
            np.asarray(target_ledger.exact_free_diagonal_lower_bounds),
        )
        or not np.array_equal(
            np.asarray(free_upper),
            np.asarray(target_ledger.exact_free_diagonal_upper_bounds),
        ),
        "exact free-diagonal intervals must match the zero-slab target",
    )
    expected_state_zero = (free_lower == 0.0) & (free_upper == 0.0)
    _raise_if(
        bool(jnp.any(state_zero != expected_state_zero)),
        "structural exact-zero state mask must use singleton [0, 0]",
    )
    _raise_if(
        any(
            value.dtype != jnp.dtype(jnp.float64)
            or value.shape != (state_size, state_size)
            for value in gram
        ),
        "Gram rectangles must be float64 square target-state arrays",
    )
    _raise_if(
        bool(jnp.any(gram[0] > gram[1]))
        or bool(jnp.any(gram[2] > gram[3]))
        or any(bool(jnp.any(~jnp.isfinite(value))) for value in gram),
        "Gram rectangles must be finite and ordered",
    )
    selected_host = np.asarray(selected, dtype=np.bool_)
    rows_host = np.asarray(rows, dtype=np.int64)
    same_fiber_block = (
        selected_host[:, None]
        & selected_host[None, :]
        & (rows_host[:, None] == rows_host[None, :])
    )
    outside = ~same_fiber_block
    gram_host = tuple(np.asarray(value) for value in gram)
    _raise_if(
        any(np.any(value[outside] != 0.0) for value in gram_host),
        "Gram entries outside selected same-fiber blocks must be zero",
    )
    diagonal = np.diag_indices(state_size)
    _raise_if(
        bool(
            np.any(gram_host[2][diagonal] > 0.0)
            or np.any(gram_host[3][diagonal] < 0.0)
        ),
        "Gram diagonal imaginary intervals must contain zero",
    )
    _raise_if(
        not np.array_equal(gram_host[0], gram_host[0].T)
        or not np.array_equal(gram_host[1], gram_host[1].T)
        or not np.array_equal(gram_host[2], -gram_host[3].T)
        or not np.array_equal(gram_host[3], -gram_host[2].T),
        "Gram rectangles must be conjugate-transpose symmetric",
    )
    _raise_if(
        fiber_zero.dtype != jnp.dtype(jnp.bool_)
        or fiber_zero.shape != (fiber_size,)
        or any(
            value.dtype != jnp.dtype(jnp.float64)
            or value.shape != (fiber_size,)
            for value in fiber_reports
        ),
        "per-fiber masks and reports must match the scoped fibers",
    )
    _raise_if(
        any(bool(jnp.any(jnp.isnan(value))) for value in fiber_reports)
        or bool(jnp.any(fiber_reports[0] < 0.0))
        or bool(jnp.any(fiber_reports[1] < fiber_reports[0]))
        or any(bool(jnp.any(value < 0.0)) for value in fiber_reports[2:]),
        "per-fiber bounds must be ordered, nonnegative, and not NaN",
    )
    _raise_if(
        any(
            value.shape != ()
            for value in (*policies, *flags, submitted_failure)
        ),
        "projection policies, predicates, and failure must be scalar",
    )
    _raise_if(
        policies[0].dtype != jnp.dtype(jnp.float64)
        or policies[1].dtype != jnp.dtype(jnp.float64)
        or any(value.dtype != jnp.dtype(jnp.int64) for value in policies[2:])
        or any(value.dtype != jnp.dtype(jnp.bool_) for value in flags)
        or submitted_failure.dtype != jnp.dtype(jnp.int64),
        "projection policies and predicates must use exact storage dtypes",
    )
    state_radius, maximum_state_error = policies[:2]
    direct_count, gram_budget, stability_budget = policies[2:5]
    precision = tuple(int(np.asarray(value)) for value in policies[5:])
    _raise_if(
        not math.isfinite(float(np.asarray(maximum_state_error)))
        or float(np.asarray(maximum_state_error))
        < float(np.finfo(np.float64).tiny),
        "maximum_state_error must be finite positive normal float64",
    )
    _raise_if(
        int(np.asarray(gram_budget)) <= 0
        or int(np.asarray(stability_budget)) <= 0,
        "both direct-pair policies must be positive signed int64",
    )
    _raise_if(
        precision
        != (
            _PI_TARGET_BITS,
            _SINE_TAYLOR_LOWER_LAST_INDEX,
            _SINE_TAYLOR_UPPER_LAST_INDEX,
            _SQRT_PRECISION_BITS,
        ),
        "projection precision transcript is noncanonical",
    )
    _raise_if(
        not np.array_equal(
            np.asarray(state_radius),
            np.asarray(stability_result.proof.state_radius_upper_bound),
        ),
        "copied state radius must exactly match the stability proof",
    )
    _raise_if(
        not np.array_equal(
            np.asarray(maximum_state_error),
            np.asarray(stability_result.proof.maximum_state_error),
        ),
        "copied state policy must exactly match the stability proof",
    )
    _raise_if(
        int(np.asarray(stability_budget))
        != int(np.asarray(stability_result.proof.maximum_direct_pairs)),
        "copied L6 work policy must exactly match the stability proof",
    )
    _raise_if(
        isinstance(maximum_state_error_numerator, bool)
        or not isinstance(maximum_state_error_numerator, int)
        or isinstance(maximum_state_error_denominator, bool)
        or not isinstance(maximum_state_error_denominator, int)
        or maximum_state_error_denominator <= 0
        or Fraction(
            maximum_state_error_numerator,
            maximum_state_error_denominator,
        )
        != Fraction.from_float(float(np.asarray(maximum_state_error))),
        "maximum_state_error exact transcript is inconsistent",
    )
    _raise_if(
        maximum_state_error_numerator
        != stability_result.proof.maximum_state_error_numerator
        or maximum_state_error_denominator
        != stability_result.proof.maximum_state_error_denominator,
        "state-policy transcript must exactly match the stability proof",
    )
    try:
        exact_count = int(direct_pair_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "direct_pair_count_exact must be canonical decimal"
        ) from error
    _raise_if(
        str(exact_count) != direct_pair_count_exact or exact_count < 0,
        "direct_pair_count_exact must be canonical nonnegative decimal",
    )

    known_failure = 0
    for reason in GalerkinLocalProjectionDefectFailure:
        known_failure |= int(reason)
    failure_value = int(np.asarray(submitted_failure))
    _raise_if(
        failure_value < 0 or bool(failure_value & ~known_failure),
        "failure_mask contains unknown projection-defect bits",
    )
    failure = GalerkinLocalProjectionDefectFailure(failure_value)
    overflow = bool(
        failure & GalerkinLocalProjectionDefectFailure.GRAM_PAIR_COUNT_OVERFLOW
    )
    budget_exceeded = bool(
        failure
        & GalerkinLocalProjectionDefectFailure.GRAM_PAIR_BUDGET_EXCEEDED
    )
    if overflow:
        _raise_if(
            exact_count <= _MAXIMUM_SIGNED_INT64
            or int(np.asarray(direct_count)) != 0
            or budget_exceeded,
            "overflow must retain exact work and store zero int64 count",
        )
    else:
        _raise_if(
            exact_count > _MAXIMUM_SIGNED_INT64
            or int(np.asarray(direct_count)) != exact_count,
            "stored Gram-pair count must match its exact transcript",
        )
        _raise_if(
            budget_exceeded != (exact_count > int(np.asarray(gram_budget))),
            "Gram-pair budget bit disagrees with exact work",
        )

    target_state = np.asarray(target.state_indices, dtype=np.int64)
    axis = target.acquisition.terminal_axis
    transverse_axes = tuple(index for index in range(3) if index != axis)
    state_transverse = target_state[:, transverse_axes]
    expected_fibers = (
        np.unique(state_transverse, axis=0)
        if projection_scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
        else np.asarray(target.acquisition.transverse_indices, dtype=np.int64)
    )
    _raise_if(
        not np.array_equal(np.asarray(fibers), expected_fibers),
        "projection fibers disagree with the canonical scope",
    )
    lookup = {
        tuple(int(component) for component in row): index
        for index, row in enumerate(expected_fibers)
    }
    expected_rows = np.zeros((state_size,), dtype=np.int64)
    expected_selected = np.zeros((state_size,), dtype=np.bool_)
    for index, transverse in enumerate(state_transverse):
        row = lookup.get(tuple(int(component) for component in transverse))
        if row is not None:
            expected_rows[index] = row
            expected_selected[index] = True
    _raise_if(
        not np.array_equal(np.asarray(rows), expected_rows)
        or not np.array_equal(np.asarray(selected), expected_selected),
        "state-to-fiber mapping disagrees with the canonical scope",
    )
    state_zero_host = np.asarray(state_zero, dtype=np.bool_)
    expected_fiber_zero_values: list[bool] = []
    for row in range(fiber_size):
        selection = expected_selected & (expected_rows == row)
        expected_fiber_zero_values.append(
            bool(np.any(selection))
            and bool(np.all(state_zero_host[selection]))
        )
    expected_fiber_zero = np.asarray(
        expected_fiber_zero_values,
        dtype=np.bool_,
    )
    _raise_if(
        not np.array_equal(np.asarray(fiber_zero), expected_fiber_zero),
        "fiber structural-zero mask disagrees with exact-D intervals",
    )

    host_ok, arithmetic_ok, structural, finite, operational = (
        bool(np.asarray(value)) for value in flags
    )
    expected_structural = bool(np.all(expected_fiber_zero))
    structural_reason = (
        GalerkinLocalProjectionDefectFailure.STRUCTURAL_EXACT_ZERO_UNAVAILABLE
    )
    structural_failure = bool(failure & structural_reason)
    _raise_if(
        structural != expected_structural or structural_failure == structural,
        "structural exact-zero predicate disagrees with its mask/outcome",
    )
    zero_ineligible = not bool(
        zero_slab_certificate.terminal_zero_slab_eligible
    )
    source_mismatch = stored_value_payload(
        zero_slab_certificate.represented_source_certificate
    ) != stored_value_payload(stability_result.certificate)
    radius_unavailable = not bool(stability_result.proof.state_radius_eligible)
    operational_missed = bool(
        stability_result.proof.state_radius_eligible
    ) and not bool(stability_result.proof.operational_state_eligible)
    scope_incomplete = (
        projection_scope
        is GalerkinLocalTerminalScope.SELECTED_PRETERMINAL_FIBERS
        and not bool(target.support_eligibility.terminal_fiber_complete)
    )
    for present, reason, name in (
        (
            zero_ineligible,
            GalerkinLocalProjectionDefectFailure.ZERO_SLAB_NONCERTIFICATE,
            "zero-slab",
        ),
        (
            source_mismatch,
            GalerkinLocalProjectionDefectFailure.PARENT_SOURCE_EVIDENCE_MISMATCH,
            "parent-source",
        ),
        (
            radius_unavailable,
            GalerkinLocalProjectionDefectFailure.STATE_RADIUS_UNAVAILABLE,
            "state-radius",
        ),
        (
            operational_missed,
            GalerkinLocalProjectionDefectFailure.OPERATIONAL_STATE_BUDGET_MISSED,
            "operational-budget",
        ),
        (
            scope_incomplete,
            GalerkinLocalProjectionDefectFailure.TERMINAL_SCOPE_INCOMPLETE,
            "terminal-scope",
        ),
        (
            not host_ok or not arithmetic_ok,
            GalerkinLocalProjectionDefectFailure.HOST_ARITHMETIC_UNSUPPORTED,
            "host-arithmetic",
        ),
    ):
        _raise_if(
            bool(failure & reason) != present,
            f"{name} failure bit disagrees with its evidence",
        )
    fatal_reasons = (
        GalerkinLocalProjectionDefectFailure.ZERO_SLAB_NONCERTIFICATE
        | GalerkinLocalProjectionDefectFailure.PARENT_SOURCE_EVIDENCE_MISMATCH
        | GalerkinLocalProjectionDefectFailure.STATE_RADIUS_UNAVAILABLE
        | GalerkinLocalProjectionDefectFailure.TERMINAL_SCOPE_INCOMPLETE
        | GalerkinLocalProjectionDefectFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalProjectionDefectFailure.GRAM_PAIR_BUDGET_EXCEEDED
        | GalerkinLocalProjectionDefectFailure.GRAM_PAIR_COUNT_OVERFLOW
        | GalerkinLocalProjectionDefectFailure.ROOT_ENCLOSURE_FAILURE
        | GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
    )
    expected_finite = not bool(failure & fatal_reasons)
    _raise_if(
        finite != expected_finite
        or operational != (finite and not operational_missed),
        "finite/operational eligibility disagrees with typed outcomes",
    )
    if finite:
        _raise_if(
            not _normal_or_zero(state_radius)
            or any(not _normal_or_zero(value) for value in fiber_reports),
            "eligible projection reports must be finite normal-or-zero",
        )

    source = zero_slab_certificate.represented_source_certificate.source
    for value, expected, name in (
        (target_digest, target.target_digest, "target_digest"),
        (
            parent_target_evidence_digest,
            target.manifest_evidence_digest,
            "parent_target_evidence_digest",
        ),
        (source_digest, source.source_digest, "source_digest"),
        (
            parent_source_evidence_digest,
            source.source_evidence_digest,
            "parent_source_evidence_digest",
        ),
        (
            parent_represented_certificate_digest,
            zero_slab_certificate.represented_source_certificate.certificate_digest,
            "parent_represented_certificate_digest",
        ),
        (
            parent_zero_slab_certificate_digest,
            zero_slab_certificate.certificate_digest,
            "parent_zero_slab_certificate_digest",
        ),
        (
            parent_stability_result_identity_digest,
            stability_result.result_identity_digest,
            "parent_stability_result_identity_digest",
        ),
        (
            parent_stability_result_evidence_digest,
            stability_result.result_evidence_digest,
            "parent_stability_result_evidence_digest",
        ),
        (
            state_identity_digest,
            stability_result.result_identity_digest,
            "state_identity_digest",
        ),
    ):
        _raise_if(value != expected, f"{name} disagrees with its parent")
    for digest, name in (
        (projection_identity_digest, "projection_identity_digest"),
        (arithmetic_environment_digest, "arithmetic_environment_digest"),
        (gram_transcript_digest, "gram_transcript_digest"),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")
    for value, name in (
        (gram_formula, "gram_formula"),
        (measurement_formula, "measurement_formula"),
        (operator_bound_formula, "operator_bound_formula"),
        (state_lift_formula, "state_lift_formula"),
        (precision_transcript, "precision_transcript"),
        (error_scope, "error_scope"),
        (completion_scope, "completion_scope"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")
    _raise_if(
        isinstance(terminal_axis, bool)
        or not isinstance(terminal_axis, int)
        or terminal_axis != axis,
        "terminal_axis must match the zero-slab target",
    )

    certificate = GalerkinLocalProjectionDefectCertificate(
        zero_slab_certificate=zero_slab_certificate,
        stability_result=stability_result,
        scope_transverse_indices=fibers,
        state_to_fiber_rows=rows,
        selected_state_mask=selected,
        exact_free_diagonal_lower_bounds=free_lower,
        exact_free_diagonal_upper_bounds=free_upper,
        structural_exact_zero_state_mask=state_zero,
        structural_exact_zero_fiber_mask=fiber_zero,
        gram_real_lower_bounds=gram[0],
        gram_real_upper_bounds=gram[1],
        gram_imag_lower_bounds=gram[2],
        gram_imag_upper_bounds=gram[3],
        measured_defect_squared_lower_bounds=fiber_reports[0],
        measured_defect_squared_upper_bounds=fiber_reports[1],
        measured_defect_upper_bounds=fiber_reports[2],
        operator_squared_norm_upper_bounds=fiber_reports[3],
        operator_norm_upper_bounds=fiber_reports[4],
        state_error_transfer_upper_bounds=fiber_reports[5],
        total_defect_upper_bounds=fiber_reports[6],
        state_radius_upper_bound=policies[0],
        maximum_state_error=policies[1],
        direct_pair_count=policies[2],
        maximum_gram_pairs=policies[3],
        maximum_stability_direct_pairs=policies[4],
        pi_target_bits=policies[5],
        sine_taylor_lower_last_index=policies[6],
        sine_taylor_upper_last_index=policies[7],
        sqrt_precision_bits=policies[8],
        host_binary64_eligible=flags[0],
        normal_arithmetic_eligible=flags[1],
        structural_exact_zero_eligible=flags[2],
        finite_projection_bound_eligible=flags[3],
        operational_budget_eligible=flags[4],
        failure_mask=submitted_failure,
        terminal_axis=terminal_axis,
        projection_scope=projection_scope,
        maximum_state_error_numerator=maximum_state_error_numerator,
        maximum_state_error_denominator=maximum_state_error_denominator,
        direct_pair_count_exact=direct_pair_count_exact,
        gram_formula=gram_formula.strip(),
        measurement_formula=measurement_formula.strip(),
        operator_bound_formula=operator_bound_formula.strip(),
        state_lift_formula=state_lift_formula.strip(),
        precision_transcript=precision_transcript.strip(),
        error_scope=error_scope.strip(),
        completion_scope=completion_scope.strip(),
        target_digest=target_digest,
        parent_target_evidence_digest=parent_target_evidence_digest,
        source_digest=source_digest,
        parent_source_evidence_digest=parent_source_evidence_digest,
        parent_represented_certificate_digest=(
            parent_represented_certificate_digest
        ),
        parent_zero_slab_certificate_digest=parent_zero_slab_certificate_digest,
        parent_stability_result_identity_digest=(
            parent_stability_result_identity_digest
        ),
        parent_stability_result_evidence_digest=(
            parent_stability_result_evidence_digest
        ),
        state_identity_digest=state_identity_digest,
        projection_identity_digest=projection_identity_digest,
        arithmetic_environment_digest=arithmetic_environment_digest,
        gram_transcript_digest=gram_transcript_digest,
        certificate_digest=certificate_digest,
    )
    return certificate  # noqa: RET504


__all__: list[str] = [
    "GalerkinLocalProjectionDefectCertificate",
    "GalerkinLocalProjectionDefectFailure",
]
