r"""Define authenticated local coordinate-terminal current carriers.

Extended Summary
----------------
These carriers keep the uniform coordinate ``T/N/F`` operator, one frozen
per-call action enclosure, and one exact-target current diagnostic disjoint.
The operator binds either every retained transverse fiber or the explicitly
selected complete preterminal fibers of one ``LOCAL_CELL_LVT1`` target at an
exact stored binary64 coordinate.  No carrier exposes vacuum-branch or
detector eligibility.

Routine Listings
----------------
:class:`GalerkinLocalCoordinateCauchyCurrent`
    Store one exact-target submitted-state scoped current enclosure.
:class:`GalerkinLocalCurrentOperatorCertificate`
    Store uniform local coordinate ``T/N/F`` evidence.
:class:`GalerkinLocalCurrentOperatorFailure`
    Enumerate typed uniform current-operator outcomes.
:class:`GalerkinLocalTerminalActionFailure`
    Enumerate typed per-call frozen-action outcomes.
:class:`GalerkinLocalTerminalComplexRectangles`
    Store componentwise complex rectangles.
:class:`GalerkinLocalTerminalCurrentActionEnclosure`
    Store one exact-real frozen-matrix action enclosure.
:class:`GalerkinLocalTerminalCurrentFailure`
    Enumerate typed exact-target current-diagnostic outcomes.
:class:`GalerkinLocalTerminalScope`
    Select the complete transverse-fiber scope.
:class:`GalerkinPreparedLocalCurrentOperator`
    Mark a host-replayed operator for frozen transform actions.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex128, Float64, Int64

from ptyrodactyl._tools import has_subnormal_components

from .acquisition_types import GalerkinTerminalSide
from .local_cell_target_types import GalerkinLocalCellTargetManifest

_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_SHA256_HEX_LENGTH: int = 64
_TRANSVERSE_DIMENSIONS: int = 2
type _ScopeArrays = tuple[
    Int64[Array, "f 2"], Int64[Array, " n"], Bool[Array, " n"]
]
type _FrozenCoefficients = tuple[
    Complex128[Array, " n"], Complex128[Array, " n"]
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
        Candidate real or complex binary64 array.

    Returns
    -------
    valid : bool
        Whether every component is finite and non-subnormal.
    """
    valid: bool = bool(jnp.all(jnp.isfinite(values))) and not bool(
        has_subnormal_components(values)
    )
    return valid


class GalerkinLocalTerminalScope(str, Enum):
    """Select the complete transverse-fiber scope.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_enums_and_carrier_boundaries`
    """

    FULL_STATE_FIBERS = "full_state_fibers"
    SELECTED_PRETERMINAL_FIBERS = "selected_preterminal_fibers"


class GalerkinLocalCurrentOperatorFailure(str, Enum):
    """Enumerate typed uniform current-operator outcomes.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_enums_and_carrier_boundaries`
    """

    NONE = "none"
    TARGET_FIXED_LINEAR_INELIGIBLE = "target_fixed_linear_ineligible"
    TERMINAL_FIBER_INCOMPLETE = "terminal_fiber_incomplete"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    DIRECT_WORK_BUDGET_EXCEEDED = "direct_work_budget_exceeded"
    DIRECT_WORK_COUNT_OVERFLOW = "direct_work_count_overflow"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"
    CURRENT_NORMALIZATION_UNENCLOSED = "current_normalization_unenclosed"


class GalerkinLocalTerminalActionFailure(str, Enum):
    """Enumerate typed per-call frozen-action outcomes.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_enums_and_carrier_boundaries`
    """

    NONE = "none"
    OPERATOR_NONCERTIFICATE = "operator_noncertificate"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    DIRECT_WORK_BUDGET_EXCEEDED = "direct_work_budget_exceeded"
    DIRECT_WORK_COUNT_OVERFLOW = "direct_work_count_overflow"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalTerminalCurrentFailure(str, Enum):
    """Enumerate typed exact-target current-diagnostic outcomes.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_enums_and_carrier_boundaries`
    """

    NONE = "none"
    OPERATOR_NONCERTIFICATE = "operator_noncertificate"
    ACTION_NONCERTIFICATE = "action_noncertificate"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    DIRECT_WORK_BUDGET_EXCEEDED = "direct_work_budget_exceeded"
    DIRECT_WORK_COUNT_OVERFLOW = "direct_work_count_overflow"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalTerminalComplexRectangles(NamedTuple):
    """Store componentwise complex rectangles.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_enums_and_carrier_boundaries`
    """

    real_lower_bounds: Float64[Array, " n"]
    real_upper_bounds: Float64[Array, " n"]
    imag_lower_bounds: Float64[Array, " n"]
    imag_upper_bounds: Float64[Array, " n"]


type _ExactRectangles = tuple[
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalComplexRectangles,
]


class GalerkinLocalCurrentOperatorCertificate(eqx.Module):
    r"""Store uniform local coordinate ``T/N/F`` evidence.

    The frozen coefficient vectors define the actual implemented matrices.
    Their adjoints are their literal conjugate transposes.  The two exact
    target rectangles include coordinate phase, trace normalization, exact
    carrier evidence, reciprocal offset, and side orientation.  The uniform
    current error is LVT.55a5 exactly once; per-call arithmetic is absent.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_operator_carrier_rejects_forged_shapes_and_digests`
    """

    target: GalerkinLocalCellTargetManifest
    terminal_plane_coordinate: Float64[Array, ""]
    scope_transverse_indices: Int64[Array, "f 2"]
    state_to_fiber_rows: Int64[Array, " n"]
    selected_state_mask: Bool[Array, " n"]
    trace_frozen_coefficients: Complex128[Array, " n"]
    normal_frozen_coefficients: Complex128[Array, " n"]
    exact_trace_coefficient_rectangles: GalerkinLocalTerminalComplexRectangles
    exact_normal_coefficient_rectangles: GalerkinLocalTerminalComplexRectangles
    trace_coefficient_error_bounds: Float64[Array, " n"]
    normal_coefficient_error_bounds: Float64[Array, " n"]
    exact_trace_operator_norm_upper_bound: Float64[Array, ""]
    exact_normal_operator_norm_upper_bound: Float64[Array, ""]
    trace_operator_error_upper_bound: Float64[Array, ""]
    normal_operator_error_upper_bound: Float64[Array, ""]
    current_operator_error_upper_bound: Float64[Array, ""]
    number_current_scale: Float64[Array, ""]
    exact_number_current_scale_lower_bound: Float64[Array, ""]
    exact_number_current_scale_upper_bound: Float64[Array, ""]
    number_current_scale_error_upper_bound: Float64[Array, ""]
    action_work_count: Int64[Array, ""]
    current_diagnostic_work_count: Int64[Array, ""]
    maximum_direct_pairs: Int64[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    normal_arithmetic_eligible: Bool[Array, ""]
    current_operator_eligible: Bool[Array, ""]
    terminal_axis: int = eqx.field(static=True)
    terminal_side: GalerkinTerminalSide = eqx.field(static=True)
    current_scope: GalerkinLocalTerminalScope = eqx.field(static=True)
    failure: GalerkinLocalCurrentOperatorFailure = eqx.field(static=True)
    action_work_count_exact: str = eqx.field(static=True)
    current_diagnostic_work_count_exact: str = eqx.field(static=True)
    trace_formula: str = eqx.field(static=True)
    normal_formula: str = eqx.field(static=True)
    current_formula: str = eqx.field(static=True)
    fixed_linear_error_formula: str = eqx.field(static=True)
    current_normalization: str = eqx.field(static=True)
    coefficient_metrics: str = eqx.field(static=True)
    eligibility_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    parent_target_evidence_digest: str = eqx.field(static=True)
    operator_identity_digest: str = eqx.field(static=True)
    operator_evidence_digest: str = eqx.field(static=True)


class GalerkinPreparedLocalCurrentOperator(eqx.Module):
    """Mark a host-replayed operator for frozen transform actions.

    This is an explicit caller trust marker, not an unforgeable Python token.
    Scientific callers must use the value returned by
    :func:`ptyrodactyl.galerkin.prepare_local_terminal_current_operator`.
    Host scientific boundaries ignore this wrapper and replay raw certificate
    storage; raw storage never enters a JIT action directly.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_operator_carrier_rejects_forged_shapes_and_digests`
    """

    certificate: GalerkinLocalCurrentOperatorCertificate


class GalerkinLocalTerminalCurrentActionEnclosure(eqx.Module):
    """Store one exact-real frozen-matrix action enclosure.

    The error is only rounded-call to frozen dyadic matrix action.  It never
    includes or duplicates the uniform exact-target ``epsilon_F``.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_operator_carrier_rejects_forged_shapes_and_digests`
    """

    certificate: GalerkinLocalCurrentOperatorCertificate
    submitted_field: Complex128[Array, " n"]
    production_action: Complex128[Array, " n"]
    frozen_action_rectangles: GalerkinLocalTerminalComplexRectangles
    component_error_bounds: Float64[Array, " n"]
    action_error_upper_bound: Float64[Array, ""]
    direct_work_count: Int64[Array, ""]
    maximum_direct_pairs: Int64[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    current_action_eligible: Bool[Array, ""]
    failure: GalerkinLocalTerminalActionFailure = eqx.field(static=True)
    direct_work_count_exact: str = eqx.field(static=True)
    exact_action_target: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    state_identity_digest: str = eqx.field(static=True)
    action_evidence_digest: str = eqx.field(static=True)


class GalerkinLocalCoordinateCauchyCurrent(eqx.Module):
    """Store one exact-target submitted-state scoped current enclosure.

    The scalar interval is evaluated directly against exact-target ``T/N``
    coefficient rectangles.  It replaces, and is never added to, a uniform
    operator-error transfer route.

    :see: :func:`~.test_local_terminal_types.\
test_local_terminal_operator_carrier_rejects_forged_shapes_and_digests`
    """

    action_enclosure: GalerkinLocalTerminalCurrentActionEnclosure
    trace_coefficients: Complex128[Array, " f"]
    normal_derivative_coefficients: Complex128[Array, " f"]
    reduced_current: Float64[Array, ""]
    exact_reduced_current_lower_bound: Float64[Array, ""]
    exact_reduced_current_upper_bound: Float64[Array, ""]
    reduced_current_error_upper_bound: Float64[Array, ""]
    direct_work_count: Int64[Array, ""]
    maximum_direct_pairs: Int64[Array, ""]
    host_binary64_eligible: Bool[Array, ""]
    current_diagnostic_eligible: Bool[Array, ""]
    failure: GalerkinLocalTerminalCurrentFailure = eqx.field(static=True)
    direct_work_count_exact: str = eqx.field(static=True)
    current_target: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    diagnostic_evidence_digest: str = eqx.field(static=True)


def _validate_operator_certificate(  # noqa: PLR0912
    certificate: GalerkinLocalCurrentOperatorCertificate,
) -> GalerkinLocalCurrentOperatorCertificate:
    """PRIVATE: Validate one raw uniform operator carrier.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Candidate raw storage record.

    Returns
    -------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Structurally validated record.

    Raises
    ------
    TypeError
        If a nested carrier or enum has the wrong type.
    ValueError
        If shapes, evidence, work, outcome, or digests are inconsistent.
    """
    if not isinstance(certificate, GalerkinLocalCurrentOperatorCertificate):
        raise TypeError("certificate has the wrong local-terminal type")
    if not isinstance(certificate.target, GalerkinLocalCellTargetManifest):
        raise TypeError("target must be GalerkinLocalCellTargetManifest")
    if not isinstance(certificate.current_scope, GalerkinLocalTerminalScope):
        raise TypeError("current_scope has the wrong enum")
    if not isinstance(certificate.terminal_side, GalerkinTerminalSide):
        raise TypeError("terminal_side has the wrong enum")
    if not isinstance(
        certificate.failure, GalerkinLocalCurrentOperatorFailure
    ):
        raise TypeError("failure has the wrong operator enum")
    state_size = certificate.target.state_indices.shape[0]
    fiber_size = certificate.scope_transverse_indices.shape[0]
    _raise_if(
        certificate.scope_transverse_indices.ndim != _TRANSVERSE_DIMENSIONS
        or certificate.scope_transverse_indices.shape[1:]
        != (_TRANSVERSE_DIMENSIONS,)
        or fiber_size <= 0,
        "scope_transverse_indices must be nonempty (f, 2)",
    )
    for values, name in (
        (certificate.state_to_fiber_rows, "state_to_fiber_rows"),
        (certificate.selected_state_mask, "selected_state_mask"),
        (certificate.trace_frozen_coefficients, "trace_frozen_coefficients"),
        (certificate.normal_frozen_coefficients, "normal_frozen_coefficients"),
        (certificate.trace_coefficient_error_bounds, "trace errors"),
        (certificate.normal_coefficient_error_bounds, "normal errors"),
    ):
        _raise_if(values.shape != (state_size,), f"{name} must match state")
    for rectangles, name in (
        (certificate.exact_trace_coefficient_rectangles, "trace rectangles"),
        (certificate.exact_normal_coefficient_rectangles, "normal rectangles"),
    ):
        _raise_if(
            any(values.shape != (state_size,) for values in rectangles),
            f"{name} must match state",
        )
        _raise_if(
            bool(
                jnp.any(
                    rectangles.real_lower_bounds > rectangles.real_upper_bounds
                )
            )
            or bool(
                jnp.any(
                    rectangles.imag_lower_bounds > rectangles.imag_upper_bounds
                )
            ),
            f"{name} must be ordered",
        )
    _raise_if(
        bool(jnp.any(certificate.state_to_fiber_rows < 0))
        or bool(jnp.any(certificate.state_to_fiber_rows >= fiber_size)),
        "state_to_fiber_rows must index the scoped fibers safely",
    )
    scalar_fields = (
        certificate.terminal_plane_coordinate,
        certificate.exact_trace_operator_norm_upper_bound,
        certificate.exact_normal_operator_norm_upper_bound,
        certificate.trace_operator_error_upper_bound,
        certificate.normal_operator_error_upper_bound,
        certificate.current_operator_error_upper_bound,
        certificate.number_current_scale,
        certificate.exact_number_current_scale_lower_bound,
        certificate.exact_number_current_scale_upper_bound,
        certificate.number_current_scale_error_upper_bound,
        certificate.action_work_count,
        certificate.current_diagnostic_work_count,
        certificate.maximum_direct_pairs,
        certificate.host_binary64_eligible,
        certificate.normal_arithmetic_eligible,
        certificate.current_operator_eligible,
    )
    _raise_if(
        any(value.shape != () for value in scalar_fields),
        "operator reports must be scalar",
    )
    _raise_if(
        certificate.terminal_axis
        != certificate.target.acquisition.terminal_axis
        or certificate.terminal_side
        is not certificate.target.acquisition.terminal_side,
        "terminal axis and side must match the target",
    )
    _raise_if(
        certificate.action_work_count.dtype != jnp.dtype(jnp.int64)
        or certificate.current_diagnostic_work_count.dtype
        != jnp.dtype(jnp.int64)
        or certificate.maximum_direct_pairs.dtype != jnp.dtype(jnp.int64)
        or int(np.asarray(certificate.maximum_direct_pairs)) <= 0,
        "operator work evidence must use positive-policy int64 scalars",
    )
    try:
        action_count = int(certificate.action_work_count_exact)
        current_count = int(certificate.current_diagnostic_work_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "exact work counts must be decimal integers"
        ) from error
    _raise_if(
        action_count < 0 or current_count < action_count,
        "work counts are inconsistent",
    )
    if (
        certificate.failure
        is GalerkinLocalCurrentOperatorFailure.DIRECT_WORK_COUNT_OVERFLOW
    ):
        _raise_if(
            action_count <= _MAXIMUM_SIGNED_INT64
            and current_count <= _MAXIMUM_SIGNED_INT64
            or int(np.asarray(certificate.action_work_count)) != 0
            or int(np.asarray(certificate.current_diagnostic_work_count)) != 0,
            "count-overflow outcome requires one oversized exact count",
        )
    else:
        _raise_if(
            action_count > _MAXIMUM_SIGNED_INT64
            or current_count > _MAXIMUM_SIGNED_INT64
            or int(np.asarray(certificate.action_work_count)) != action_count
            or int(np.asarray(certificate.current_diagnostic_work_count))
            != current_count,
            "stored work counts must equal exact signed-int64 counts",
        )
        if (
            certificate.failure
            is GalerkinLocalCurrentOperatorFailure.DIRECT_WORK_BUDGET_EXCEEDED
        ):
            _raise_if(
                action_count
                <= int(np.asarray(certificate.maximum_direct_pairs)),
                "operator budget failure requires action work above policy",
            )
    eligible = bool(certificate.current_operator_eligible)
    _raise_if(
        eligible
        != (certificate.failure is GalerkinLocalCurrentOperatorFailure.NONE),
        "operator eligibility must agree with its typed outcome",
    )
    if eligible:
        finite_arrays = (
            certificate.trace_frozen_coefficients,
            certificate.normal_frozen_coefficients,
            *certificate.exact_trace_coefficient_rectangles,
            *certificate.exact_normal_coefficient_rectangles,
            certificate.trace_coefficient_error_bounds,
            certificate.normal_coefficient_error_bounds,
            *scalar_fields[:10],
        )
        _raise_if(
            not all(_normal_or_zero(values) for values in finite_arrays),
            "eligible operator evidence must be finite normal-or-zero",
        )
        _raise_if(
            any(
                bool(value < 0.0)
                for value in (
                    certificate.exact_trace_operator_norm_upper_bound,
                    certificate.exact_normal_operator_norm_upper_bound,
                    certificate.trace_operator_error_upper_bound,
                    certificate.normal_operator_error_upper_bound,
                    certificate.current_operator_error_upper_bound,
                    certificate.number_current_scale_error_upper_bound,
                )
            )
            or not bool(certificate.host_binary64_eligible)
            or not bool(certificate.normal_arithmetic_eligible)
            or action_count
            > int(np.asarray(certificate.maximum_direct_pairs)),
            "eligible operator evidence violates arithmetic or work policy",
        )
    for text, name in (
        (certificate.trace_formula, "trace_formula"),
        (certificate.normal_formula, "normal_formula"),
        (certificate.current_formula, "current_formula"),
        (certificate.fixed_linear_error_formula, "fixed_linear_error_formula"),
        (certificate.current_normalization, "current_normalization"),
        (certificate.coefficient_metrics, "coefficient_metrics"),
        (certificate.eligibility_scope, "eligibility_scope"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (certificate.target_digest, "target_digest"),
        (certificate.parent_target_evidence_digest, "parent target evidence"),
        (certificate.operator_identity_digest, "operator identity"),
        (certificate.operator_evidence_digest, "operator evidence"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")
    return certificate


def _make_local_current_operator_certificate(  # noqa: PLR0913
    target: GalerkinLocalCellTargetManifest,
    coordinate: Float64[Array, ""],
    scope_arrays: _ScopeArrays,
    frozen_coefficients: _FrozenCoefficients,
    exact_rectangles: _ExactRectangles,
    coefficient_errors: tuple[Float64[Array, " n"], Float64[Array, " n"]],
    reports: tuple[Float64[Array, ""], ...],
    work: tuple[Int64[Array, ""], Int64[Array, ""], Int64[Array, ""]],
    flags: tuple[Bool[Array, ""], Bool[Array, ""], Bool[Array, ""]],
    *,
    terminal_axis: int,
    terminal_side: GalerkinTerminalSide,
    current_scope: GalerkinLocalTerminalScope,
    failure: GalerkinLocalCurrentOperatorFailure,
    action_work_count_exact: str,
    current_diagnostic_work_count_exact: str,
    declarations: tuple[str, ...],
    digests: tuple[str, str, str, str],
) -> GalerkinLocalCurrentOperatorCertificate:
    """PRIVATE: Construct and validate one uniform operator certificate.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully replayed local-cell target.
    coordinate : Float64[Array, ""]
        Exact stored plane coordinate.
    scope_arrays : _ScopeArrays
        Scoped fibers, safe state rows, and selected-state mask.
    frozen_coefficients : _FrozenCoefficients
        Actual frozen trace and normal coefficients.
    exact_rectangles : _ExactRectangles
        Exact-target trace and normal coefficient rectangles.
    coefficient_errors : tuple[Float64[Array, " n"], Float64[Array, " n"]]
        Frozen-to-exact per-coefficient errors.
    reports : tuple[Float64[Array, ""], ...]
        Nine ordered norm, error, and SC.35c reports.
    work : tuple[Int64[Array, ""], Int64[Array, ""], Int64[Array, ""]]
        Action count, diagnostic count, and independent budget.
    flags : tuple[Bool[Array, ""], Bool[Array, ""], Bool[Array, ""]]
        Host, normal-arithmetic, and operator predicates.
    terminal_axis : int
        Target-owned normal axis.
    terminal_side : GalerkinTerminalSide
        Target-owned side orientation.
    current_scope : GalerkinLocalTerminalScope
        Complete transverse-fiber scope.
    failure : GalerkinLocalCurrentOperatorFailure
        Typed uniform outcome.
    action_work_count_exact : str
        Unclamped linear action-work transcript.
    current_diagnostic_work_count_exact : str
        Unclamped current-diagnostic work transcript.
    declarations : tuple[str, ...]
        Seven ordered formula and scope declarations.
    digests : tuple[str, str, str, str]
        Target, parent evidence, operator identity, and evidence digests.

    Returns
    -------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Validated raw storage record.
    """
    certificate = GalerkinLocalCurrentOperatorCertificate(
        target=target,
        terminal_plane_coordinate=coordinate,
        scope_transverse_indices=scope_arrays[0],
        state_to_fiber_rows=scope_arrays[1],
        selected_state_mask=scope_arrays[2],
        trace_frozen_coefficients=frozen_coefficients[0],
        normal_frozen_coefficients=frozen_coefficients[1],
        exact_trace_coefficient_rectangles=exact_rectangles[0],
        exact_normal_coefficient_rectangles=exact_rectangles[1],
        trace_coefficient_error_bounds=coefficient_errors[0],
        normal_coefficient_error_bounds=coefficient_errors[1],
        exact_trace_operator_norm_upper_bound=reports[0],
        exact_normal_operator_norm_upper_bound=reports[1],
        trace_operator_error_upper_bound=reports[2],
        normal_operator_error_upper_bound=reports[3],
        current_operator_error_upper_bound=reports[4],
        number_current_scale=reports[5],
        exact_number_current_scale_lower_bound=reports[6],
        exact_number_current_scale_upper_bound=reports[7],
        number_current_scale_error_upper_bound=reports[8],
        action_work_count=work[0],
        current_diagnostic_work_count=work[1],
        maximum_direct_pairs=work[2],
        host_binary64_eligible=flags[0],
        normal_arithmetic_eligible=flags[1],
        current_operator_eligible=flags[2],
        terminal_axis=terminal_axis,
        terminal_side=terminal_side,
        current_scope=current_scope,
        failure=failure,
        action_work_count_exact=action_work_count_exact,
        current_diagnostic_work_count_exact=current_diagnostic_work_count_exact,
        trace_formula=declarations[0],
        normal_formula=declarations[1],
        current_formula=declarations[2],
        fixed_linear_error_formula=declarations[3],
        current_normalization=declarations[4],
        coefficient_metrics=declarations[5],
        eligibility_scope=declarations[6],
        target_digest=digests[0],
        parent_target_evidence_digest=digests[1],
        operator_identity_digest=digests[2],
        operator_evidence_digest=digests[3],
    )
    return _validate_operator_certificate(certificate)


def _make_prepared_local_current_operator(
    certificate: GalerkinLocalCurrentOperatorCertificate,
) -> GalerkinPreparedLocalCurrentOperator:
    """PRIVATE: Wrap one canonically replayed operator certificate.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Canonically rebuilt raw certificate.

    Returns
    -------
    prepared : GalerkinPreparedLocalCurrentOperator
        Prepared-only frozen-action capability.
    """
    prepared = GalerkinPreparedLocalCurrentOperator(
        certificate=_validate_operator_certificate(certificate)
    )
    return prepared  # noqa: RET504


def _make_local_terminal_current_action_enclosure(  # noqa: PLR0913
    certificate: GalerkinLocalCurrentOperatorCertificate,
    submitted_field: Complex128[Array, " n"],
    production_action: Complex128[Array, " n"],
    rectangles: GalerkinLocalTerminalComplexRectangles,
    component_errors: Float64[Array, " n"],
    action_error: Float64[Array, ""],
    work: tuple[Int64[Array, ""], Int64[Array, ""]],
    flags: tuple[Bool[Array, ""], Bool[Array, ""]],
    *,
    failure: GalerkinLocalTerminalActionFailure,
    direct_work_count_exact: str,
    exact_action_target: str,
    error_scope: str,
    state_identity_digest: str,
    action_evidence_digest: str,
) -> GalerkinLocalTerminalCurrentActionEnclosure:
    """PRIVATE: Construct and validate one frozen-action enclosure.

    Parameters
    ----------
    certificate : GalerkinLocalCurrentOperatorCertificate
        Parent uniform operator certificate.
    submitted_field : Complex128[Array, " n"]
        Exact stored submitted state.
    production_action : Complex128[Array, " n"]
        Rounded implicit frozen action.
    rectangles : GalerkinLocalTerminalComplexRectangles
        Exact-real frozen-action component rectangles.
    component_errors : Float64[Array, " n"]
        Per-component rounded-to-frozen errors.
    action_error : Float64[Array, ""]
        Outward Euclidean action error.
    work : tuple[Int64[Array, ""], Int64[Array, ""]]
        Stored direct work count and independent budget.
    flags : tuple[Bool[Array, ""], Bool[Array, ""]]
        Host and action predicates.
    failure : GalerkinLocalTerminalActionFailure
        Typed per-call outcome.
    direct_work_count_exact : str
        Unclamped work transcript.
    exact_action_target : str
        Frozen-matrix target declaration.
    error_scope : str
        Nonduplication declaration.
    state_identity_digest : str
        Operator-and-field identity digest.
    action_evidence_digest : str
        Full per-call evidence digest.

    Returns
    -------
    enclosure : GalerkinLocalTerminalCurrentActionEnclosure
        Validated per-call action record.

    Raises
    ------
    ValueError
        If shapes, work policy, disposition, range, or digests are invalid.
    """
    enclosure = GalerkinLocalTerminalCurrentActionEnclosure(
        certificate=certificate,
        submitted_field=submitted_field,
        production_action=production_action,
        frozen_action_rectangles=rectangles,
        component_error_bounds=component_errors,
        action_error_upper_bound=action_error,
        direct_work_count=work[0],
        maximum_direct_pairs=work[1],
        host_binary64_eligible=flags[0],
        current_action_eligible=flags[1],
        failure=failure,
        direct_work_count_exact=direct_work_count_exact,
        exact_action_target=exact_action_target,
        error_scope=error_scope,
        state_identity_digest=state_identity_digest,
        action_evidence_digest=action_evidence_digest,
    )
    state_size = certificate.target.state_indices.shape[0]
    _raise_if(
        submitted_field.shape != (state_size,)
        or production_action.shape != (state_size,)
        or component_errors.shape != (state_size,)
        or any(values.shape != (state_size,) for values in rectangles),
        "action vectors and rectangles must match target state",
    )
    _raise_if(
        action_error.shape != ()
        or work[0].shape != ()
        or work[1].shape != ()
        or flags[0].shape != ()
        or flags[1].shape != (),
        "action reports must be scalar",
    )
    _raise_if(
        work[0].dtype != jnp.dtype(jnp.int64)
        or work[1].dtype != jnp.dtype(jnp.int64)
        or int(np.asarray(work[1])) <= 0,
        "action work evidence must use positive-policy int64 scalars",
    )
    try:
        exact_count = int(direct_work_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "direct_work_count_exact must be a decimal integer"
        ) from error
    _raise_if(exact_count < 0, "direct_work_count_exact cannot be negative")
    stored_count = int(np.asarray(work[0]))
    budget = int(np.asarray(work[1]))
    _raise_if(
        exact_count != int(certificate.action_work_count_exact),
        "action work transcript must match the parent operator",
    )
    if (
        failure
        is GalerkinLocalTerminalActionFailure.DIRECT_WORK_COUNT_OVERFLOW
    ):
        _raise_if(
            exact_count <= _MAXIMUM_SIGNED_INT64 or stored_count != 0,
            "action count overflow must preserve an unclamped transcript",
        )
    else:
        _raise_if(
            exact_count > _MAXIMUM_SIGNED_INT64 or stored_count != exact_count,
            "stored action work must equal its signed-int64 transcript",
        )
        if (
            bool(certificate.current_operator_eligible)
            and bool(flags[0])
            and exact_count > budget
        ):
            budget_failure = (
                GalerkinLocalTerminalActionFailure.DIRECT_WORK_BUDGET_EXCEEDED
            )
            _raise_if(
                failure is not budget_failure,
                "excess action work requires the typed budget outcome",
            )
        if (
            failure
            is GalerkinLocalTerminalActionFailure.DIRECT_WORK_BUDGET_EXCEEDED
        ):
            _raise_if(
                exact_count <= budget,
                "action budget failure requires work above the policy",
            )
    _raise_if(
        bool(flags[1]) != (failure is GalerkinLocalTerminalActionFailure.NONE),
        "action eligibility must agree with typed outcome",
    )
    if bool(flags[1]):
        _raise_if(
            not all(
                _normal_or_zero(values)
                for values in (
                    submitted_field,
                    production_action,
                    *rectangles,
                    component_errors,
                    action_error,
                )
            )
            or not bool(flags[0])
            or bool(action_error < 0.0),
            "eligible action evidence must be finite normal-or-zero",
        )
        _raise_if(
            exact_count > budget,
            "eligible action evidence cannot exceed its work policy",
        )
    for value, name in (
        (exact_action_target, "exact_action_target"),
        (error_scope, "error_scope"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")
    for digest in (state_identity_digest, action_evidence_digest):
        _raise_if(not _valid_digest(digest), "action digests must be SHA-256")
    return enclosure


def _make_local_coordinate_cauchy_current(  # noqa: PLR0913
    action_enclosure: GalerkinLocalTerminalCurrentActionEnclosure,
    trace: Complex128[Array, " f"],
    normal: Complex128[Array, " f"],
    current_reports: tuple[Float64[Array, ""], ...],
    work: tuple[Int64[Array, ""], Int64[Array, ""]],
    flags: tuple[Bool[Array, ""], Bool[Array, ""]],
    *,
    failure: GalerkinLocalTerminalCurrentFailure,
    direct_work_count_exact: str,
    current_target: str,
    error_scope: str,
    diagnostic_evidence_digest: str,
) -> GalerkinLocalCoordinateCauchyCurrent:
    """PRIVATE: Construct and validate one exact-target current diagnostic.

    Parameters
    ----------
    action_enclosure : GalerkinLocalTerminalCurrentActionEnclosure
        Authenticated per-call frozen-action evidence.
    trace : Complex128[Array, " f"]
        Rounded carrier-stripped trace.
    normal : Complex128[Array, " f"]
        Rounded side-oriented normal trace.
    current_reports : tuple[Float64[Array, ""], ...]
        Rounded current, exact lower/upper endpoints, and error.
    work : tuple[Int64[Array, ""], Int64[Array, ""]]
        Diagnostic work count and independent budget.
    flags : tuple[Bool[Array, ""], Bool[Array, ""]]
        Host and current predicates.
    failure : GalerkinLocalTerminalCurrentFailure
        Typed diagnostic outcome.
    direct_work_count_exact : str
        Unclamped diagnostic work transcript.
    current_target : str
        Exact-target scalar declaration.
    error_scope : str
        Direct-route nonduplication declaration.
    diagnostic_evidence_digest : str
        Full diagnostic evidence digest.

    Returns
    -------
    diagnostic : GalerkinLocalCoordinateCauchyCurrent
        Validated exact-target current record.

    Raises
    ------
    ValueError
        If shapes, work policy, disposition, range, or digest are invalid.
    """
    diagnostic = GalerkinLocalCoordinateCauchyCurrent(
        action_enclosure=action_enclosure,
        trace_coefficients=trace,
        normal_derivative_coefficients=normal,
        reduced_current=current_reports[0],
        exact_reduced_current_lower_bound=current_reports[1],
        exact_reduced_current_upper_bound=current_reports[2],
        reduced_current_error_upper_bound=current_reports[3],
        direct_work_count=work[0],
        maximum_direct_pairs=work[1],
        host_binary64_eligible=flags[0],
        current_diagnostic_eligible=flags[1],
        failure=failure,
        direct_work_count_exact=direct_work_count_exact,
        current_target=current_target,
        error_scope=error_scope,
        diagnostic_evidence_digest=diagnostic_evidence_digest,
    )
    fiber_size = action_enclosure.certificate.scope_transverse_indices.shape[0]
    _raise_if(
        trace.shape != (fiber_size,) or normal.shape != (fiber_size,),
        "current traces must match scoped fibers",
    )
    _raise_if(
        any(value.shape != () for value in (*current_reports, *work, *flags)),
        "current reports must be scalar",
    )
    _raise_if(
        work[0].dtype != jnp.dtype(jnp.int64)
        or work[1].dtype != jnp.dtype(jnp.int64)
        or int(np.asarray(work[1])) <= 0,
        "current work evidence must use positive-policy int64 scalars",
    )
    try:
        exact_count = int(direct_work_count_exact)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "direct_work_count_exact must be a decimal integer"
        ) from error
    _raise_if(exact_count < 0, "direct_work_count_exact cannot be negative")
    stored_count = int(np.asarray(work[0]))
    budget = int(np.asarray(work[1]))
    _raise_if(
        exact_count
        != int(
            action_enclosure.certificate.current_diagnostic_work_count_exact
        ),
        "current work transcript must match the parent operator",
    )
    if (
        failure
        is GalerkinLocalTerminalCurrentFailure.DIRECT_WORK_COUNT_OVERFLOW
    ):
        _raise_if(
            exact_count <= _MAXIMUM_SIGNED_INT64 or stored_count != 0,
            "current count overflow must preserve an unclamped transcript",
        )
    else:
        _raise_if(
            exact_count > _MAXIMUM_SIGNED_INT64 or stored_count != exact_count,
            "stored current work must equal its signed-int64 transcript",
        )
        if (
            bool(action_enclosure.current_action_eligible)
            and bool(flags[0])
            and exact_count > budget
        ):
            budget_failure = (
                GalerkinLocalTerminalCurrentFailure.DIRECT_WORK_BUDGET_EXCEEDED
            )
            _raise_if(
                failure is not budget_failure,
                "excess current work requires the typed budget outcome",
            )
        if (
            failure
            is GalerkinLocalTerminalCurrentFailure.DIRECT_WORK_BUDGET_EXCEEDED
        ):
            _raise_if(
                exact_count <= budget,
                "current budget failure requires work above the policy",
            )
    _raise_if(
        bool(current_reports[1] > current_reports[2])
        or bool(current_reports[3] < 0.0),
        "current interval and error must be ordered and nonnegative",
    )
    _raise_if(
        bool(flags[1])
        != (failure is GalerkinLocalTerminalCurrentFailure.NONE),
        "current eligibility must agree with typed outcome",
    )
    if bool(flags[1]):
        _raise_if(
            not bool(action_enclosure.current_action_eligible)
            or not bool(flags[0])
            or not all(
                _normal_or_zero(value)
                for value in (trace, normal, *current_reports)
            ),
            "eligible current evidence must be finite normal-or-zero",
        )
        _raise_if(
            exact_count > budget,
            "eligible current evidence cannot exceed its work policy",
        )
    for value, name in (
        (current_target, "current_target"),
        (error_scope, "error_scope"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")
    _raise_if(
        not _valid_digest(diagnostic_evidence_digest),
        "diagnostic evidence digest must be SHA-256",
    )
    return diagnostic


__all__: list[str] = [
    "GalerkinLocalCoordinateCauchyCurrent",
    "GalerkinLocalCurrentOperatorCertificate",
    "GalerkinLocalCurrentOperatorFailure",
    "GalerkinLocalTerminalActionFailure",
    "GalerkinLocalTerminalComplexRectangles",
    "GalerkinLocalTerminalCurrentActionEnclosure",
    "GalerkinLocalTerminalCurrentFailure",
    "GalerkinLocalTerminalScope",
    "GalerkinPreparedLocalCurrentOperator",
]
