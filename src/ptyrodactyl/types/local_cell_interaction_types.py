r"""Define disjoint LVT-1 exact-compression and interaction-core carriers.

Extended Summary
----------------
This module stores the LVT.14--LVT.18 evidence that turns one authenticated
local-cell coefficient realization into a fixed finite interaction action.
It deliberately does not store a free diagonal, absorber, source, terminal,
or solver-ready target.  Its reverse-conjugate action is the formal adjoint of
the frozen algebraic matrix; per-call floating arithmetic is outside LVT.18.

Routine Listings
----------------
:class:`GalerkinLocalCellCompressionFailure`
    Enumerate typed LVT exact-compression noncertificate outcomes.
:class:`GalerkinLocalCellExactCompression`
    Store authenticated LVT.14--LVT.18 exact-compression evidence.
:class:`GalerkinLocalCellInteractionCore`
    Store one non-solver-ready fixed LVT interaction action core.
"""

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from .born_potential_types import GalerkinProductSupport
from .local_cell_types import GalerkinLocalCellPotentialRealization

_SHA256_HEX_LENGTH: int = 64
_SUPPORT_RANK: int = 2
_SPACE_DIMENSIONS: int = 3
_WORK_RESOURCE_COUNT: int = 4

type _FloatBounds = Tuple[Float[Array, "..."], Float[Array, "..."]]
type _IntScalarQuad = Tuple[
    Int[Array, ""],
    Int[Array, ""],
    Int[Array, ""],
    Int[Array, ""],
]


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

    Parameters
    ----------
    condition : bool
        Whether the structural contract failed.
    message : str
        Rejection message for the caller.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one lowercase SHA-256 hexadecimal string.

    Parameters
    ----------
    value : str
        Candidate digest text.

    Returns
    -------
    result : bool
        Whether the text is one canonical SHA-256 digest.
    """
    result: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return result


class GalerkinLocalCellCompressionFailure(str, Enum):
    """Enumerate typed LVT exact-compression noncertificate outcomes."""

    NONE = "none"
    L2_CERTIFICATE_NOT_FINITE = "l2_certificate_not_finite"
    STATE_PAIR_BUDGET_EXCEEDED = "state_pair_budget_exceeded"
    INTERACTION_MODE_BUDGET_EXCEEDED = "interaction_mode_budget_exceeded"
    WORK_GRID_BUDGET_EXCEEDED = "work_grid_budget_exceeded"
    HOST_ARRAY_WORKING_SET_BUDGET_EXCEEDED = (
        "host_array_working_set_budget_exceeded"
    )
    DIFFERENCE_ARITHMETIC_RANGE_FAILURE = "difference_arithmetic_range_failure"
    DIFFERENCE_COVERAGE_MISSING = "difference_coverage_missing"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    SIGMA_ENCLOSURE_FAILURE = "sigma_enclosure_failure"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    INTERACTION_RANGE_FAILURE = "interaction_range_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalCellExactCompression(eqx.Module):
    """Store authenticated LVT.14--LVT.18 exact-compression evidence.

    Notes
    -----
    Interaction rectangles and errors are ordered by ``difference_indices``.
    ``difference_interaction_positions`` maps those modes into the ordered
    interaction support.  The stored interaction coefficients cover the full
    ordered interaction support used by the rounded action.
    """

    realization: GalerkinLocalCellPotentialRealization
    product_support: GalerkinProductSupport
    difference_indices: Int64[Array, "d 3"]
    difference_interaction_positions: Int64[Array, " d"]
    difference_multiplicities: Int64[Array, " d"]
    state_pair_interaction_positions: Int64[Array, " s"]
    accelerating_voltage_kv: Float64[Array, ""]
    interaction_coupling: Float64[Array, ""]
    interaction_coefficients: Complex128[Array, " p"]
    exact_coupling_lower_bound: Float64[Array, ""]
    exact_coupling_upper_bound: Float64[Array, ""]
    coupling_error_bound: Float64[Array, ""]
    exact_interaction_real_lower_bounds: Float64[Array, " d"]
    exact_interaction_real_upper_bounds: Float64[Array, " d"]
    exact_interaction_imag_lower_bounds: Float64[Array, " d"]
    exact_interaction_imag_upper_bounds: Float64[Array, " d"]
    interaction_coefficient_error_bounds: Float64[Array, " d"]
    fixed_interaction_error_bound: Float64[Array, ""]
    finite_certificate: Bool[Array, ""]
    state_pair_count: Int64[Array, ""]
    interaction_mode_count: Int64[Array, ""]
    work_grid_point_count: Int64[Array, ""]
    host_array_working_set_upper_bound: Int64[Array, ""]
    maximum_state_pairs: Int64[Array, ""]
    maximum_interaction_modes: Int64[Array, ""]
    maximum_work_grid_points: Int64[Array, ""]
    maximum_host_array_working_set_bytes: Int64[Array, ""]
    failure: GalerkinLocalCellCompressionFailure = eqx.field(static=True)
    exact_target: str = eqx.field(static=True)
    coupling_target: str = eqx.field(static=True)
    interaction_realization_route: str = eqx.field(static=True)
    difference_count_route: str = eqx.field(static=True)
    compression_claim: str = eqx.field(static=True)
    operator_error_scope: str = eqx.field(static=True)
    per_call_arithmetic_exclusion: str = eqx.field(static=True)
    host_transient_scalar_scope: str = eqx.field(static=True)
    parent_certificate_digest: str = eqx.field(static=True)
    operator_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


class GalerkinLocalCellInteractionCore(eqx.Module):
    """Store one non-solver-ready fixed LVT interaction action core."""

    compression: GalerkinLocalCellExactCompression
    action_route: str = eqx.field(static=True)
    adjoint_route: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    operator_digest: str = eqx.field(static=True)

    @property
    def support(self) -> GalerkinProductSupport:
        """Return the independently rebuilt product support."""
        support: GalerkinProductSupport = self.compression.product_support
        return support


@jaxtyped(typechecker=beartype)
def _make_local_cell_exact_compression(  # noqa: PLR0913
    realization: GalerkinLocalCellPotentialRealization,
    product_support: GalerkinProductSupport,
    difference_indices: Int[Array, "..."],
    difference_interaction_positions: Int[Array, "..."],
    difference_multiplicities: Int[Array, "..."],
    state_pair_interaction_positions: Int[Array, "..."],
    accelerating_voltage_kv: Float[Array, "..."],
    interaction_coupling: Float[Array, "..."],
    interaction_coefficients: Complex[Array, "..."],
    exact_coupling_bounds: _FloatBounds,
    coupling_error_bound: Float[Array, "..."],
    exact_interaction_real_bounds: _FloatBounds,
    exact_interaction_imag_bounds: _FloatBounds,
    interaction_coefficient_error_bounds: Float[Array, "..."],
    fixed_interaction_error_bound: Float[Array, "..."],
    finite_certificate: Bool[Array, ""],
    counts: _IntScalarQuad,
    budgets: _IntScalarQuad,
    *,
    failure: GalerkinLocalCellCompressionFailure,
    exact_target: str,
    coupling_target: str,
    interaction_realization_route: str,
    difference_count_route: str,
    compression_claim: str,
    operator_error_scope: str,
    per_call_arithmetic_exclusion: str,
    host_transient_scalar_scope: str,
    parent_certificate_digest: str,
    operator_digest: str,
    certificate_digest: str,
) -> GalerkinLocalCellExactCompression:
    """PRIVATE: Jointly validate and store one LVT compression attempt.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Authenticated finite L2 parent realization.
    product_support : GalerkinProductSupport
        Independently rebuilt product support.
    difference_indices : Int[Array, "..."]
        Lexicographically ordered exact state differences.
    difference_interaction_positions : Int[Array, "..."]
        Difference positions in ordered interaction support.
    difference_multiplicities : Int[Array, "..."]
        Exact ordered-pair multiplicities.
    state_pair_interaction_positions : Int[Array, "..."]
        Row-major pair-to-interaction lookup.
    accelerating_voltage_kv : Float[Array, "..."]
        Positive stored accelerating voltage in kilovolts.
    interaction_coupling : Float[Array, "..."]
        Canonical stored SC.4 coupling.
    interaction_coefficients : Complex[Array, "..."]
        Canonical stored interaction coefficients on ordered support.
    exact_coupling_bounds : _FloatBounds
        Outward binary64 enclosure of the exact coupling point.
    coupling_error_bound : Float[Array, "..."]
        Direct stored-coupling audit error.
    exact_interaction_real_bounds : _FloatBounds
        Outward real interaction rectangle endpoints.
    exact_interaction_imag_bounds : _FloatBounds
        Outward imaginary interaction rectangle endpoints.
    interaction_coefficient_error_bounds : Float[Array, "..."]
        Direct point-to-rectangle Euclidean errors.
    fixed_interaction_error_bound : Float[Array, "..."]
        LVT.18 fixed interaction error.
    finite_certificate : Bool[Array, ""]
        Whether the attempt is one finite certificate.
    counts : _IntScalarQuad
        Exact pair, mode, work-grid, and host-array counts.
    budgets : _IntScalarQuad
        Corresponding positive resource budgets.
    failure : GalerkinLocalCellCompressionFailure
        Typed outcome stored as static evidence.
    exact_target : str
        Nonempty exact-target declaration.
    coupling_target : str
        Nonempty coupling-target declaration.
    interaction_realization_route : str
        Nonempty canonical production route.
    difference_count_route : str
        Nonempty exact difference-counting route.
    compression_claim : str
        Nonempty finite-compression claim scope.
    operator_error_scope : str
        Nonempty fixed-operator error scope.
    per_call_arithmetic_exclusion : str
        Nonempty exclusion of per-call rounded arithmetic.
    host_transient_scalar_scope : str
        Nonempty bounded host scalar-object scope.
    parent_certificate_digest : str
        Authenticated parent certificate identity.
    operator_digest : str
        Evidence-free fixed interaction identity.
    certificate_digest : str
        Complete child certificate identity.

    Returns
    -------
    compression : GalerkinLocalCellExactCompression
        Structurally validated exact-compression storage.

    Raises
    ------
    ValueError
        If shapes, declarations, counts, outcomes, or digests are invalid.
    equinox.EquinoxRuntimeError
        If dynamic scientific fields contradict the typed outcome.
    """
    differences: Int64[Array, "d 3"] = jnp.asarray(
        difference_indices, dtype=jnp.int64
    )
    positions: Int64[Array, " d"] = jnp.asarray(
        difference_interaction_positions, dtype=jnp.int64
    )
    multiplicities: Int64[Array, " d"] = jnp.asarray(
        difference_multiplicities, dtype=jnp.int64
    )
    pair_positions: Int64[Array, " s"] = jnp.asarray(
        state_pair_interaction_positions, dtype=jnp.int64
    )
    voltage: Float64[Array, ""] = jnp.asarray(
        accelerating_voltage_kv, dtype=jnp.float64
    )
    coupling: Float64[Array, ""] = jnp.asarray(
        interaction_coupling, dtype=jnp.float64
    )
    coefficients: Complex128[Array, " p"] = jnp.asarray(
        interaction_coefficients, dtype=jnp.complex128
    )
    sigma_lower: Float64[Array, ""] = jnp.asarray(
        exact_coupling_bounds[0], dtype=jnp.float64
    )
    sigma_upper: Float64[Array, ""] = jnp.asarray(
        exact_coupling_bounds[1], dtype=jnp.float64
    )
    sigma_error: Float64[Array, ""] = jnp.asarray(
        coupling_error_bound, dtype=jnp.float64
    )
    real_lower: Float64[Array, " d"] = jnp.asarray(
        exact_interaction_real_bounds[0], dtype=jnp.float64
    )
    real_upper: Float64[Array, " d"] = jnp.asarray(
        exact_interaction_real_bounds[1], dtype=jnp.float64
    )
    imag_lower: Float64[Array, " d"] = jnp.asarray(
        exact_interaction_imag_bounds[0], dtype=jnp.float64
    )
    imag_upper: Float64[Array, " d"] = jnp.asarray(
        exact_interaction_imag_bounds[1], dtype=jnp.float64
    )
    coefficient_errors: Float64[Array, " d"] = jnp.asarray(
        interaction_coefficient_error_bounds, dtype=jnp.float64
    )
    operator_error: Float64[Array, ""] = jnp.asarray(
        fixed_interaction_error_bound, dtype=jnp.float64
    )
    finite: Bool[Array, ""] = jnp.asarray(finite_certificate, dtype=jnp.bool_)
    count_arrays: Tuple[Int64[Array, ""], ...] = tuple(
        jnp.asarray(value, dtype=jnp.int64) for value in counts
    )
    budget_arrays: Tuple[Int64[Array, ""], ...] = tuple(
        jnp.asarray(value, dtype=jnp.int64) for value in budgets
    )

    _raise_if(
        differences.ndim != _SUPPORT_RANK
        or differences.shape[1:] != (_SPACE_DIMENSIONS,),
        "difference_indices must have shape (d, 3)",
    )
    difference_shape = (differences.shape[0],)
    for value, name in (
        (positions, "difference_interaction_positions"),
        (multiplicities, "difference_multiplicities"),
        (real_lower, "exact_interaction_real_lower_bounds"),
        (real_upper, "exact_interaction_real_upper_bounds"),
        (imag_lower, "exact_interaction_imag_lower_bounds"),
        (imag_upper, "exact_interaction_imag_upper_bounds"),
        (coefficient_errors, "interaction_coefficient_error_bounds"),
    ):
        _raise_if(value.shape != difference_shape, f"{name} must match D_u")
    _raise_if(
        pair_positions.ndim != 1,
        "state_pair_interaction_positions must be 1D",
    )
    _raise_if(
        coefficients.ndim != 1
        or coefficients.shape[0]
        != product_support.interaction_indices.shape[0],
        "interaction_coefficients must match I_chi",
    )
    scalar_values = (
        voltage,
        coupling,
        sigma_lower,
        sigma_upper,
        sigma_error,
        operator_error,
        finite,
        *count_arrays,
        *budget_arrays,
    )
    _raise_if(
        any(value.shape != () for value in scalar_values),
        "compression scalar fields must be scalars",
    )
    _raise_if(
        len(count_arrays) != _WORK_RESOURCE_COUNT
        or len(budget_arrays) != _WORK_RESOURCE_COUNT,
        "counts and budgets must each contain four scalars",
    )
    for value, name in (
        (exact_target, "exact_target"),
        (coupling_target, "coupling_target"),
        (interaction_realization_route, "interaction_realization_route"),
        (difference_count_route, "difference_count_route"),
        (compression_claim, "compression_claim"),
        (operator_error_scope, "operator_error_scope"),
        (per_call_arithmetic_exclusion, "per_call_arithmetic_exclusion"),
        (host_transient_scalar_scope, "host_transient_scalar_scope"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")
    for value, name in (
        (parent_certificate_digest, "parent_certificate_digest"),
        (operator_digest, "operator_digest"),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(not _valid_digest(value), f"{name} must be a SHA-256 digest")

    failure_is_none: bool = failure is GalerkinLocalCellCompressionFailure.NONE
    endpoint_arrays = (real_lower, real_upper, imag_lower, imag_upper)
    invalid: Bool[Array, ""] = (
        jnp.any(jnp.isnan(real_lower))
        | jnp.any(jnp.isnan(real_upper))
        | jnp.any(jnp.isnan(imag_lower))
        | jnp.any(jnp.isnan(imag_upper))
        | jnp.any(real_lower > real_upper)
        | jnp.any(imag_lower > imag_upper)
        | jnp.any(jnp.isnan(coefficient_errors))
        | jnp.any(coefficient_errors < 0.0)
        | jnp.isnan(operator_error)
        | (operator_error < 0.0)
        | (finite != failure_is_none)
        | jnp.any(jnp.asarray(count_arrays) < 0)
        | jnp.any(jnp.asarray(budget_arrays) <= 0)
    )
    success_invalid: Bool[Array, ""] = finite & (
        (differences.shape[0] == 0)
        | jnp.any(positions < 0)
        | jnp.any(positions >= coefficients.shape[0])
        | jnp.any(multiplicities <= 0)
        | (pair_positions.shape[0] != count_arrays[0])
        | jnp.any(pair_positions < 0)
        | jnp.any(pair_positions >= coefficients.shape[0])
        | jnp.any(~jnp.isfinite(coefficients))
        | (~jnp.isfinite(voltage))
        | (voltage <= 0.0)
        | (~jnp.isfinite(coupling))
        | (coupling <= 0.0)
        | (~jnp.isfinite(sigma_lower))
        | (~jnp.isfinite(sigma_upper))
        | (sigma_lower <= 0.0)
        | (sigma_lower > sigma_upper)
        | (~jnp.isfinite(sigma_error))
        | jnp.any(
            jnp.stack(
                [jnp.any(~jnp.isfinite(value)) for value in endpoint_arrays]
            )
        )
        | jnp.any(~jnp.isfinite(coefficient_errors))
        | (~jnp.isfinite(operator_error))
    )
    failure_invalid: Bool[Array, ""] = (~finite) & (
        jnp.any(~jnp.isinf(coefficient_errors)) | (~jnp.isinf(operator_error))
    )
    checked_differences: Int64[Array, "d 3"] = eqx.error_if(
        differences,
        invalid | success_invalid | failure_invalid,
        "local-cell compression fields contradict their typed outcome",
    )
    compression: GalerkinLocalCellExactCompression = (
        GalerkinLocalCellExactCompression(
            realization=realization,
            product_support=product_support,
            difference_indices=checked_differences,
            difference_interaction_positions=positions,
            difference_multiplicities=multiplicities,
            state_pair_interaction_positions=pair_positions,
            accelerating_voltage_kv=voltage,
            interaction_coupling=coupling,
            interaction_coefficients=coefficients,
            exact_coupling_lower_bound=sigma_lower,
            exact_coupling_upper_bound=sigma_upper,
            coupling_error_bound=sigma_error,
            exact_interaction_real_lower_bounds=real_lower,
            exact_interaction_real_upper_bounds=real_upper,
            exact_interaction_imag_lower_bounds=imag_lower,
            exact_interaction_imag_upper_bounds=imag_upper,
            interaction_coefficient_error_bounds=coefficient_errors,
            fixed_interaction_error_bound=operator_error,
            finite_certificate=finite,
            state_pair_count=count_arrays[0],
            interaction_mode_count=count_arrays[1],
            work_grid_point_count=count_arrays[2],
            host_array_working_set_upper_bound=count_arrays[3],
            maximum_state_pairs=budget_arrays[0],
            maximum_interaction_modes=budget_arrays[1],
            maximum_work_grid_points=budget_arrays[2],
            maximum_host_array_working_set_bytes=budget_arrays[3],
            failure=failure,
            exact_target=exact_target.strip(),
            coupling_target=coupling_target.strip(),
            interaction_realization_route=(
                interaction_realization_route.strip()
            ),
            difference_count_route=difference_count_route.strip(),
            compression_claim=compression_claim.strip(),
            operator_error_scope=operator_error_scope.strip(),
            per_call_arithmetic_exclusion=(
                per_call_arithmetic_exclusion.strip()
            ),
            host_transient_scalar_scope=host_transient_scalar_scope.strip(),
            parent_certificate_digest=parent_certificate_digest,
            operator_digest=operator_digest,
            certificate_digest=certificate_digest,
        )
    )
    return compression


def _make_local_cell_interaction_core(
    compression: GalerkinLocalCellExactCompression,
    *,
    action_route: str,
    adjoint_route: str,
    completion_scope: str,
    operator_digest: str,
) -> GalerkinLocalCellInteractionCore:
    """PRIVATE: Store one accepted non-solver-ready interaction core.

    Parameters
    ----------
    compression : GalerkinLocalCellExactCompression
        Finite exact-compression evidence.
    action_route : str
        Canonical forward action declaration.
    adjoint_route : str
        Canonical formal-adjoint declaration.
    completion_scope : str
        Non-solver-ready completion boundary.
    operator_digest : str
        Evidence-free operator identity.

    Returns
    -------
    core : GalerkinLocalCellInteractionCore
        Structurally validated fixed interaction core.

    Raises
    ------
    ValueError
        If compression, declarations, or identity are invalid.
    """
    if not bool(compression.finite_certificate):
        raise ValueError("interaction core requires a finite compression")
    if compression.failure is not GalerkinLocalCellCompressionFailure.NONE:
        raise ValueError("interaction core requires failure NONE")
    for value, name in (
        (action_route, "action_route"),
        (adjoint_route, "adjoint_route"),
        (completion_scope, "completion_scope"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")
    _raise_if(
        not _valid_digest(operator_digest),
        "operator_digest must be a SHA-256 digest",
    )
    _raise_if(
        operator_digest != compression.operator_digest,
        "core operator digest must match its compression",
    )
    core: GalerkinLocalCellInteractionCore = GalerkinLocalCellInteractionCore(
        compression=compression,
        action_route=action_route.strip(),
        adjoint_route=adjoint_route.strip(),
        completion_scope=completion_scope.strip(),
        operator_digest=operator_digest,
    )
    return core


__all__: list[str] = [
    "GalerkinLocalCellCompressionFailure",
    "GalerkinLocalCellExactCompression",
    "GalerkinLocalCellInteractionCore",
]
