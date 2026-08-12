r"""Define disjoint LVT-1 axial-cell CAP evidence carriers.

Extended Summary
----------------
This leaf stores the LVT.23 axial profile and frozen LVT.24 coefficient
approximant, the independently replayable LVT.26/LVT.31 coefficient
certificate, and the verified support-only LVT.29--LVT.32 floor proof.  Exact
target-floor eligibility is deliberately independent of coefficient and
realized-matrix eligibility.  None of these carriers is solver-ready.

Routine Listings
----------------
:class:`GalerkinAxialCapCoefficientCertificate`
    Store replayable LVT.24 rectangles and the LVT.31 transfer.
:class:`GalerkinAxialCapCoefficientFailure`
    Enumerate typed LVT.24/LVT.26/LVT.31 noncertificate outcomes.
:class:`GalerkinAxialCapExactFloorFailure`
    Enumerate typed support-only exact LVT.29a proof outcomes.
:class:`GalerkinAxialCapFloorProof`
    Store independently eligible exact-target and realized CAP floors.
:class:`GalerkinAxialCapRealizedFloorFailure`
    Enumerate typed coefficient-dependent realized-floor outcomes.
:class:`GalerkinAxialCapRealizedFloorRoute`
    Select one mutually exclusive physical realized-floor route.
:class:`GalerkinAxialCellAbsorber`
    Store one axis-only LVT.23 profile and frozen LVT.24 approximant.
"""

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
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
from .local_cell_interaction_types import GalerkinLocalCellInteractionCore

_SHA256_HEX_LENGTH: int = 64
_SPACE_DIMENSIONS: int = 3
_SUPPORT_RANK: int = 2
_STRICT_HALF_WIDTH: float = 0.5


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for one structural failure.

    Parameters
    ----------
    condition : bool
        Internal value used by this helper.
    message : str
        Internal value used by this helper.

    Raises
    ------
    ValueError
        If an internal validation or arithmetic check fails.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 string.

    Parameters
    ----------
    value : str
        Internal value used by this helper.

    Returns
    -------
    valid : bool
        Internal result produced by this helper.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


def _valid_unsigned_decimal(value: str, *, positive: bool) -> bool:
    """PRIVATE: Check one canonical unsigned decimal integer string.

    Parameters
    ----------
    value : str
        Internal value used by this helper.
    positive : bool
        Internal value used by this helper.

    Returns
    -------
    _return_value : bool
        Internal result produced by this helper.
    """
    if not isinstance(value, str) or not value or not value.isdecimal():
        _return_value: bool = False
        return _return_value
    if len(value) > 1 and value.startswith("0"):
        _return_value: bool = False
        return _return_value
    integer = int(value)
    valid: bool = integer > 0 if positive else integer >= 0
    _return_value: bool = valid
    return _return_value


class GalerkinAxialCapCoefficientFailure(str, Enum):
    """Enumerate typed LVT.24/LVT.26/LVT.31 noncertificate outcomes.

    :see: :func:`~.test_absorber_types.\
test_exact_and_realized_failure_spaces_cannot_be_conflated`
    """

    NONE = "none"
    DIRECT_TERM_BUDGET_EXCEEDED = "direct_term_budget_exceeded"
    STATE_PAIR_BUDGET_EXCEEDED = "state_pair_budget_exceeded"
    DIFFERENCE_COVERAGE_MISSING = "difference_coverage_missing"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinAxialCapExactFloorFailure(str, Enum):
    """Enumerate typed support-only exact LVT.29a proof outcomes.

    :see: :func:`~.test_absorber_types.\
test_exact_and_realized_failure_spaces_cannot_be_conflated`
    """

    NONE = "none"
    GRAM_DEGREE_BUDGET_EXCEEDED = "gram_degree_budget_exceeded"
    GRAM_WORK_BUDGET_EXCEEDED = "gram_work_budget_exceeded"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    GRAM_ARITHMETIC_FAILURE = "gram_arithmetic_failure"
    GRAM_LOWER_BOUND_NONPOSITIVE = "gram_lower_bound_nonpositive"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinAxialCapRealizedFloorFailure(str, Enum):
    """Enumerate typed coefficient-dependent realized-floor outcomes.

    :see: :func:`~.test_absorber_types.\
test_exact_and_realized_failure_spaces_cannot_be_conflated`
    """

    NONE = "none"
    EXACT_TARGET_FLOOR_NOT_FINITE = "exact_target_floor_not_finite"
    COEFFICIENT_CERTIFICATE_NOT_FINITE = "coefficient_certificate_not_finite"
    REALIZED_DIMENSIONLESS_FLOOR_NONPOSITIVE = (
        "realized_dimensionless_floor_nonpositive"
    )
    REALIZED_PHYSICAL_FLOOR_NONPOSITIVE = "realized_physical_floor_nonpositive"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinAxialCapRealizedFloorRoute(str, Enum):
    """Select one mutually exclusive physical realized-floor route.

    :see: :func:`~.test_absorber_types.\
test_exact_and_realized_failure_spaces_cannot_be_conflated`
    """

    EXACT_FROZEN_SCALE_LVT32A = "exact_frozen_scale_lvt32a"
    SCALE_TRANSFER_LVT32B = "scale_transfer_lvt32b"


class GalerkinAxialCellAbsorber(eqx.Module):
    """Store one axis-only LVT.23 profile and frozen LVT.24 approximant.

    :see: :func:`~.test_absorber_types.\
test_absorber_carriers_own_only_the_l4_evidence_layers`

    Notes
    -----
    The nested L3 core binds the local-cell grid, acquisition terminal axis,
    and all four product supports. ``signed_absorber_positions`` proves the
    ordinary signed-index pairing used to validate exact numeric Hermitian
    symmetry. Signed-zero byte choices remain distinct in ``operator_digest``.
    The exact and algebraic CAP scales are physical inverse-square Angstrom
    quantities (Angstrom^-2).
    """

    interaction_core: GalerkinLocalCellInteractionCore
    layer_values: Float64[Array, " l"]
    plateau_floor: Float64[Array, ""]
    exact_cap_scale: Float64[Array, ""]
    algebraic_cap_scale: Float64[Array, ""]
    absorber_coefficients: Complex128[Array, " a"]
    signed_absorber_positions: Int64[Array, " a"]
    terminal_axis: int = eqx.field(static=True)
    plateau_start: int = eqx.field(static=True)
    plateau_count: int = eqx.field(static=True)
    zero_start: int = eqx.field(static=True)
    zero_count: int = eqx.field(static=True)
    exact_profile_target: str = eqx.field(static=True)
    coefficient_formula: str = eqx.field(static=True)
    hermitian_approximant_claim: str = eqx.field(static=True)
    scale_semantics: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    operator_digest: str = eqx.field(static=True)

    @property
    def support(self) -> GalerkinProductSupport:
        """Return the L3-owned product support."""
        support: GalerkinProductSupport = self.interaction_core.support
        return support


class GalerkinAxialCapCoefficientCertificate(eqx.Module):
    """Store replayable LVT.24 rectangles and the LVT.31 transfer.

    :see: :func:`~.test_absorber_types.\
test_absorber_carriers_own_only_the_l4_evidence_layers`
    """

    absorber: GalerkinAxialCellAbsorber
    exact_coefficient_real_lower_bounds: Float64[Array, " a"]
    exact_coefficient_real_upper_bounds: Float64[Array, " a"]
    exact_coefficient_imag_lower_bounds: Float64[Array, " a"]
    exact_coefficient_imag_upper_bounds: Float64[Array, " a"]
    coefficient_error_bounds: Float64[Array, " a"]
    difference_indices: Int64[Array, "d 3"]
    difference_absorber_positions: Int64[Array, " d"]
    difference_multiplicities: Int64[Array, " d"]
    state_pair_absorber_positions: Int64[Array, " s"]
    absorber_operator_error_bound: Float64[Array, ""]
    finite_certificate: Bool[Array, ""]
    direct_term_count: Int64[Array, ""]
    state_pair_count: Int64[Array, ""]
    maximum_direct_terms: Int64[Array, ""]
    maximum_state_pairs: Int64[Array, ""]
    failure: GalerkinAxialCapCoefficientFailure = eqx.field(static=True)
    exact_target: str = eqx.field(static=True)
    arithmetic: str = eqx.field(static=True)
    coverage_claim: str = eqx.field(static=True)
    operator_error_scope: str = eqx.field(static=True)
    per_call_arithmetic_exclusion: str = eqx.field(static=True)
    parent_operator_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


class GalerkinAxialCapFloorProof(eqx.Module):
    """Store independently eligible exact-target and realized CAP floors.

    :see: :func:`~.test_absorber_types.\
test_absorber_carriers_own_only_the_l4_evidence_layers`

    Notes
    -----
    ``exact_target_floor_eligible`` depends only on the authenticated profile,
    plateau, exact CAP scale, support, and verified Gram arithmetic. It can
    therefore remain true when the coefficient certificate or a realized
    floor fails. ``gram_subinterval_numerator`` and denominator identify the
    exact rational interval used by the proof; the float field is display
    evidence only.
    """

    coefficient_certificate: GalerkinAxialCapCoefficientCertificate
    gram_degree: Int64[Array, ""]
    gram_subinterval_width: Float64[Array, ""]
    gram_midpoint_shift_lower_bound: Float64[Array, ""]
    gram_entry_frobenius_error_upper_bound: Float64[Array, ""]
    plateau_gram_lower_bound: Float64[Array, ""]
    dimensionless_exact_floor_lower_bound: Float64[Array, ""]
    exact_target_physical_floor_lower_bound: Float64[Array, ""]
    realized_dimensionless_floor_lower_bound: Float64[Array, ""]
    scale_error_bound: Float64[Array, ""]
    physical_operator_error_upper_bound: Float64[Array, ""]
    realized_physical_floor_lower_bound: Float64[Array, ""]
    exact_target_floor_eligible: Bool[Array, ""]
    realized_floor_eligible: Bool[Array, ""]
    maximum_gram_degree: Int64[Array, ""]
    gram_precision_bits: Int64[Array, ""]
    ldl_iteration_count: Int64[Array, ""]
    gram_work_count: Int64[Array, ""]
    maximum_gram_work: Int64[Array, ""]
    exact_target_failure: GalerkinAxialCapExactFloorFailure = eqx.field(
        static=True
    )
    realized_floor_failure: GalerkinAxialCapRealizedFloorFailure = eqx.field(
        static=True
    )
    realized_floor_route: GalerkinAxialCapRealizedFloorRoute = eqx.field(
        static=True
    )
    gram_subinterval_numerator: str = eqx.field(static=True)
    gram_subinterval_denominator: str = eqx.field(static=True)
    exact_floor_target: str = eqx.field(static=True)
    gram_proof_route: str = eqx.field(static=True)
    gram_work_scope: str = eqx.field(static=True)
    realized_floor_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    parent_certificate_digest: str = eqx.field(static=True)
    gram_transcript_digest: str = eqx.field(static=True)
    proof_digest: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def _make_axial_cell_absorber(  # noqa: PLR0913,PLR0915
    interaction_core: GalerkinLocalCellInteractionCore,
    layer_values: Float[Array, "..."],
    plateau_floor: Float[Array, "..."],
    exact_cap_scale: Float[Array, "..."],
    algebraic_cap_scale: Float[Array, "..."],
    absorber_coefficients: Complex[Array, "..."],
    signed_absorber_positions: Int[Array, "..."],
    *,
    terminal_axis: int,
    plateau_start: int,
    plateau_count: int,
    zero_start: int,
    zero_count: int,
    exact_profile_target: str,
    coefficient_formula: str,
    hermitian_approximant_claim: str,
    scale_semantics: str,
    completion_scope: str,
    source_digest: str,
    operator_digest: str,
) -> GalerkinAxialCellAbsorber:
    """PRIVATE: Validate and store one axial-cell absorber realization.

    Parameters
    ----------
    interaction_core : GalerkinLocalCellInteractionCore
        Internal value used by this helper.
    layer_values : Float[Array, '...']
        Internal value used by this helper.
    plateau_floor : Float[Array, '...']
        Internal value used by this helper.
    exact_cap_scale : Float[Array, '...']
        Internal value used by this helper.
    algebraic_cap_scale : Float[Array, '...']
        Internal value used by this helper.
    absorber_coefficients : Complex[Array, '...']
        Internal value used by this helper.
    signed_absorber_positions : Int[Array, '...']
        Internal value used by this helper.
    terminal_axis : int
        Internal value used by this helper.
    plateau_start : int
        Internal value used by this helper.
    plateau_count : int
        Internal value used by this helper.
    zero_start : int
        Internal value used by this helper.
    zero_count : int
        Internal value used by this helper.
    exact_profile_target : str
        Internal value used by this helper.
    coefficient_formula : str
        Internal value used by this helper.
    hermitian_approximant_claim : str
        Internal value used by this helper.
    scale_semantics : str
        Internal value used by this helper.
    completion_scope : str
        Internal value used by this helper.
    source_digest : str
        Internal value used by this helper.
    operator_digest : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCellAbsorber
        Internal result produced by this helper.
    """
    values: Float64[Array, " l"] = jnp.asarray(layer_values, dtype=jnp.float64)
    floor: Float64[Array, ""] = jnp.asarray(plateau_floor, dtype=jnp.float64)
    exact_scale: Float64[Array, ""] = jnp.asarray(
        exact_cap_scale, dtype=jnp.float64
    )
    algebraic_scale: Float64[Array, ""] = jnp.asarray(
        algebraic_cap_scale, dtype=jnp.float64
    )
    coefficients: Complex128[Array, " a"] = jnp.asarray(
        absorber_coefficients, dtype=jnp.complex128
    )
    signed_positions: Int64[Array, " a"] = jnp.asarray(
        signed_absorber_positions, dtype=jnp.int64
    )
    _raise_if(
        values.ndim != 1 or values.shape[0] == 0,
        "layer_values must be nonempty 1D",
    )
    _raise_if(coefficients.ndim != 1, "absorber_coefficients must be 1D")
    absorber_size: int = interaction_core.support.absorber_indices.shape[0]
    _raise_if(
        coefficients.shape != (absorber_size,),
        "absorber_coefficients must match I_a",
    )
    _raise_if(
        signed_positions.shape != (absorber_size,),
        "signed_absorber_positions must match I_a",
    )
    for scalar, name in (
        (floor, "plateau_floor"),
        (exact_scale, "exact_cap_scale"),
        (algebraic_scale, "algebraic_cap_scale"),
    ):
        _raise_if(scalar.shape != (), f"{name} must be a scalar")
    _raise_if(
        isinstance(terminal_axis, bool)
        or not isinstance(terminal_axis, int)
        or terminal_axis not in range(_SPACE_DIMENSIONS),
        "terminal_axis must be 0, 1, or 2",
    )
    acquisition = (
        interaction_core.compression.realization.support_eligibility.manifest
    )
    _raise_if(
        terminal_axis != acquisition.terminal_axis,
        "terminal_axis must match the nested acquisition",
    )
    cell_values = (
        interaction_core.compression.realization.local_potential.cell_values
    )
    grid_shape_xyz = tuple(reversed(cell_values.shape))
    layer_count: int = values.shape[0]
    _raise_if(
        layer_count != grid_shape_xyz[terminal_axis],
        "layer_values must match the local-cell grid along terminal_axis",
    )
    for start, count, name in (
        (plateau_start, plateau_count, "plateau"),
        (zero_start, zero_count, "zero"),
    ):
        _raise_if(
            isinstance(start, bool)
            or not isinstance(start, int)
            or not 0 <= start < layer_count,
            f"{name}_start must be a canonical layer index",
        )
        _raise_if(
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= layer_count,
            f"{name}_count must lie in [1, layer_count]",
        )
    plateau_positions = {
        (plateau_start + offset) % layer_count
        for offset in range(plateau_count)
    }
    zero_positions = {
        (zero_start + offset) % layer_count for offset in range(zero_count)
    }
    _raise_if(
        not plateau_positions.isdisjoint(zero_positions),
        "plateau and zero blocks must be disjoint",
    )
    plateau_indices = jnp.asarray(sorted(plateau_positions), dtype=jnp.int64)
    zero_indices = jnp.asarray(sorted(zero_positions), dtype=jnp.int64)
    support_indices = interaction_core.support.absorber_indices
    safe_signed_positions = jnp.clip(signed_positions, 0, absorber_size - 1)
    invalid_signed_positions = (
        jnp.any(signed_positions < 0)
        | jnp.any(signed_positions >= absorber_size)
        | jnp.any(
            support_indices + support_indices[safe_signed_positions] != 0
        )
        | jnp.any(
            signed_positions[safe_signed_positions]
            != jnp.arange(absorber_size, dtype=jnp.int64)
        )
    )
    minimum_normal = jnp.asarray(
        jnp.finfo(jnp.float64).tiny, dtype=jnp.float64
    )
    coefficient_real = jnp.real(coefficients)
    coefficient_imaginary = jnp.imag(coefficients)
    subnormal_storage = (
        jnp.any((values != 0.0) & (jnp.abs(values) < minimum_normal))
        | (floor < minimum_normal)
        | (exact_scale < minimum_normal)
        | (algebraic_scale < minimum_normal)
        | jnp.any(
            (coefficient_real != 0.0)
            & (jnp.abs(coefficient_real) < minimum_normal)
        )
        | jnp.any(
            (coefficient_imaginary != 0.0)
            & (jnp.abs(coefficient_imaginary) < minimum_normal)
        )
    )
    invalid: Bool[Array, ""] = (
        jnp.any(~jnp.isfinite(values))
        | jnp.any(values < 0.0)
        | jnp.any(values > 1.0)
        | (~jnp.isfinite(floor))
        | (floor <= 0.0)
        | (floor > 1.0)
        | jnp.any(values[plateau_indices] < floor)
        | jnp.any(values[zero_indices] != 0.0)
        | (~jnp.isfinite(exact_scale))
        | (exact_scale <= 0.0)
        | (~jnp.isfinite(algebraic_scale))
        | (algebraic_scale <= 0.0)
        | jnp.any(~jnp.isfinite(coefficients))
        | subnormal_storage
        | invalid_signed_positions
        | jnp.any(
            coefficients[safe_signed_positions] != jnp.conj(coefficients)
        )
    )
    checked_values: Float64[Array, " l"] = eqx.error_if(
        values,
        invalid,
        "axial profile, scale, support, and Hermitian approximant must "
        "satisfy LVT.23-LVT.24",
    )
    for declaration, name in (
        (exact_profile_target, "exact_profile_target"),
        (coefficient_formula, "coefficient_formula"),
        (hermitian_approximant_claim, "hermitian_approximant_claim"),
        (scale_semantics, "scale_semantics"),
        (completion_scope, "completion_scope"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")
    for digest, name in (
        (source_digest, "source_digest"),
        (operator_digest, "operator_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    absorber = GalerkinAxialCellAbsorber(
        interaction_core=interaction_core,
        layer_values=checked_values,
        plateau_floor=floor,
        exact_cap_scale=exact_scale,
        algebraic_cap_scale=algebraic_scale,
        absorber_coefficients=coefficients,
        signed_absorber_positions=signed_positions,
        terminal_axis=terminal_axis,
        plateau_start=plateau_start,
        plateau_count=plateau_count,
        zero_start=zero_start,
        zero_count=zero_count,
        exact_profile_target=exact_profile_target.strip(),
        coefficient_formula=coefficient_formula.strip(),
        hermitian_approximant_claim=hermitian_approximant_claim.strip(),
        scale_semantics=scale_semantics.strip(),
        completion_scope=completion_scope.strip(),
        source_digest=source_digest,
        operator_digest=operator_digest,
    )
    _return_value: GalerkinAxialCellAbsorber = absorber
    return _return_value  # noqa: RET504


@jaxtyped(typechecker=beartype)
def _make_axial_cap_coefficient_certificate(  # noqa: PLR0913,PLR0915
    absorber: GalerkinAxialCellAbsorber,
    exact_coefficient_real_lower_bounds: Float[Array, "..."],
    exact_coefficient_real_upper_bounds: Float[Array, "..."],
    exact_coefficient_imag_lower_bounds: Float[Array, "..."],
    exact_coefficient_imag_upper_bounds: Float[Array, "..."],
    coefficient_error_bounds: Float[Array, "..."],
    difference_indices: Int[Array, "..."],
    difference_absorber_positions: Int[Array, "..."],
    difference_multiplicities: Int[Array, "..."],
    state_pair_absorber_positions: Int[Array, "..."],
    absorber_operator_error_bound: Float[Array, "..."],
    finite_certificate: Bool[Array, ""],
    direct_term_count: Int[Array, "..."],
    state_pair_count: Int[Array, "..."],
    maximum_direct_terms: Int[Array, "..."],
    maximum_state_pairs: Int[Array, "..."],
    *,
    failure: GalerkinAxialCapCoefficientFailure,
    exact_target: str,
    arithmetic: str,
    coverage_claim: str,
    operator_error_scope: str,
    per_call_arithmetic_exclusion: str,
    parent_operator_digest: str,
    certificate_digest: str,
) -> GalerkinAxialCapCoefficientCertificate:
    """PRIVATE: Validate and store one LVT.24/LVT.31 attempt.

    Parameters
    ----------
    absorber : GalerkinAxialCellAbsorber
        Internal value used by this helper.
    exact_coefficient_real_lower_bounds : Float[Array, '...']
        Internal value used by this helper.
    exact_coefficient_real_upper_bounds : Float[Array, '...']
        Internal value used by this helper.
    exact_coefficient_imag_lower_bounds : Float[Array, '...']
        Internal value used by this helper.
    exact_coefficient_imag_upper_bounds : Float[Array, '...']
        Internal value used by this helper.
    coefficient_error_bounds : Float[Array, '...']
        Internal value used by this helper.
    difference_indices : Int[Array, '...']
        Internal value used by this helper.
    difference_absorber_positions : Int[Array, '...']
        Internal value used by this helper.
    difference_multiplicities : Int[Array, '...']
        Internal value used by this helper.
    state_pair_absorber_positions : Int[Array, '...']
        Internal value used by this helper.
    absorber_operator_error_bound : Float[Array, '...']
        Internal value used by this helper.
    finite_certificate : Bool[Array, '']
        Internal value used by this helper.
    direct_term_count : Int[Array, '...']
        Internal value used by this helper.
    state_pair_count : Int[Array, '...']
        Internal value used by this helper.
    maximum_direct_terms : Int[Array, '...']
        Internal value used by this helper.
    maximum_state_pairs : Int[Array, '...']
        Internal value used by this helper.
    failure : GalerkinAxialCapCoefficientFailure
        Internal value used by this helper.
    exact_target : str
        Internal value used by this helper.
    arithmetic : str
        Internal value used by this helper.
    coverage_claim : str
        Internal value used by this helper.
    operator_error_scope : str
        Internal value used by this helper.
    per_call_arithmetic_exclusion : str
        Internal value used by this helper.
    parent_operator_digest : str
        Internal value used by this helper.
    certificate_digest : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapCoefficientCertificate
        Internal result produced by this helper.
    """
    real_lower = jnp.asarray(
        exact_coefficient_real_lower_bounds, dtype=jnp.float64
    )
    real_upper = jnp.asarray(
        exact_coefficient_real_upper_bounds, dtype=jnp.float64
    )
    imag_lower = jnp.asarray(
        exact_coefficient_imag_lower_bounds, dtype=jnp.float64
    )
    imag_upper = jnp.asarray(
        exact_coefficient_imag_upper_bounds, dtype=jnp.float64
    )
    errors = jnp.asarray(coefficient_error_bounds, dtype=jnp.float64)
    differences = jnp.asarray(difference_indices, dtype=jnp.int64)
    positions = jnp.asarray(difference_absorber_positions, dtype=jnp.int64)
    multiplicities = jnp.asarray(difference_multiplicities, dtype=jnp.int64)
    pair_positions = jnp.asarray(
        state_pair_absorber_positions, dtype=jnp.int64
    )
    operator_error = jnp.asarray(
        absorber_operator_error_bound, dtype=jnp.float64
    )
    finite = jnp.asarray(finite_certificate, dtype=jnp.bool_)
    term_count = jnp.asarray(direct_term_count, dtype=jnp.int64)
    pair_count = jnp.asarray(state_pair_count, dtype=jnp.int64)
    term_budget = jnp.asarray(maximum_direct_terms, dtype=jnp.int64)
    pair_budget = jnp.asarray(maximum_state_pairs, dtype=jnp.int64)
    coefficient_shape = absorber.absorber_coefficients.shape
    for value, name in (
        (real_lower, "exact_coefficient_real_lower_bounds"),
        (real_upper, "exact_coefficient_real_upper_bounds"),
        (imag_lower, "exact_coefficient_imag_lower_bounds"),
        (imag_upper, "exact_coefficient_imag_upper_bounds"),
        (errors, "coefficient_error_bounds"),
    ):
        _raise_if(
            value.ndim != 1 or value.shape != coefficient_shape,
            f"{name} must match I_a",
        )
    _raise_if(
        differences.ndim != _SUPPORT_RANK
        or differences.shape[1:] != (_SPACE_DIMENSIONS,),
        "difference_indices must have shape (d, 3)",
    )
    difference_shape = (differences.shape[0],)
    for value, name in (
        (positions, "difference_absorber_positions"),
        (multiplicities, "difference_multiplicities"),
    ):
        _raise_if(
            value.ndim != 1 or value.shape != difference_shape,
            f"{name} must match D_u",
        )
    _raise_if(
        pair_positions.ndim != 1,
        "state_pair_absorber_positions must be 1D",
    )
    for scalar, name in (
        (operator_error, "absorber_operator_error_bound"),
        (finite, "finite_certificate"),
        (term_count, "direct_term_count"),
        (pair_count, "state_pair_count"),
        (term_budget, "maximum_direct_terms"),
        (pair_budget, "maximum_state_pairs"),
    ):
        _raise_if(scalar.shape != (), f"{name} must be a scalar")
    for declaration, name in (
        (exact_target, "exact_target"),
        (arithmetic, "arithmetic"),
        (coverage_claim, "coverage_claim"),
        (operator_error_scope, "operator_error_scope"),
        (per_call_arithmetic_exclusion, "per_call_arithmetic_exclusion"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")
    for digest, name in (
        (parent_operator_digest, "parent_operator_digest"),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    success: bool = failure is GalerkinAxialCapCoefficientFailure.NONE
    invalid: Bool[Array, ""] = (
        jnp.any(jnp.isnan(real_lower))
        | jnp.any(jnp.isnan(real_upper))
        | jnp.any(jnp.isnan(imag_lower))
        | jnp.any(jnp.isnan(imag_upper))
        | jnp.any(real_lower > real_upper)
        | jnp.any(imag_lower > imag_upper)
        | jnp.any(jnp.isnan(errors))
        | jnp.any(errors < 0.0)
        | jnp.isnan(operator_error)
        | (operator_error < 0.0)
        | (finite != success)
        | (term_count < 0)
        | (pair_count < 0)
        | (term_budget <= 0)
        | (pair_budget <= 0)
    )
    success_invalid: Bool[Array, ""] = finite & (
        jnp.any(~jnp.isfinite(real_lower))
        | jnp.any(~jnp.isfinite(real_upper))
        | jnp.any(~jnp.isfinite(imag_lower))
        | jnp.any(~jnp.isfinite(imag_upper))
        | jnp.any(~jnp.isfinite(errors))
        | (~jnp.isfinite(operator_error))
        | (differences.shape[0] == 0)
        | jnp.any(positions < 0)
        | jnp.any(positions >= coefficient_shape[0])
        | jnp.any(multiplicities <= 0)
        | (
            pair_positions.shape[0]
            != absorber.support.state_indices.shape[0] ** 2
        )
        | (pair_count != pair_positions.shape[0])
        | (term_count > term_budget)
        | (pair_count > pair_budget)
        | jnp.any(pair_positions < 0)
        | jnp.any(pair_positions >= coefficient_shape[0])
    )
    failure_invalid: Bool[Array, ""] = (~finite) & (
        jnp.any(~jnp.isinf(errors)) | (~jnp.isinf(operator_error))
    )
    budget_outcome_invalid: Bool[Array, ""] = (
        (
            failure
            is GalerkinAxialCapCoefficientFailure.DIRECT_TERM_BUDGET_EXCEEDED
        )
        & (term_count <= term_budget)
    ) | (
        (
            failure
            is GalerkinAxialCapCoefficientFailure.STATE_PAIR_BUDGET_EXCEEDED
        )
        & ((term_count > term_budget) | (pair_count <= pair_budget))
    )
    checked_differences = eqx.error_if(
        differences,
        invalid | success_invalid | failure_invalid | budget_outcome_invalid,
        "axial CAP coefficient evidence contradicts its typed outcome",
    )
    certificate = GalerkinAxialCapCoefficientCertificate(
        absorber=absorber,
        exact_coefficient_real_lower_bounds=real_lower,
        exact_coefficient_real_upper_bounds=real_upper,
        exact_coefficient_imag_lower_bounds=imag_lower,
        exact_coefficient_imag_upper_bounds=imag_upper,
        coefficient_error_bounds=errors,
        difference_indices=checked_differences,
        difference_absorber_positions=positions,
        difference_multiplicities=multiplicities,
        state_pair_absorber_positions=pair_positions,
        absorber_operator_error_bound=operator_error,
        finite_certificate=finite,
        direct_term_count=term_count,
        state_pair_count=pair_count,
        maximum_direct_terms=term_budget,
        maximum_state_pairs=pair_budget,
        failure=failure,
        exact_target=exact_target.strip(),
        arithmetic=arithmetic.strip(),
        coverage_claim=coverage_claim.strip(),
        operator_error_scope=operator_error_scope.strip(),
        per_call_arithmetic_exclusion=per_call_arithmetic_exclusion.strip(),
        parent_operator_digest=parent_operator_digest,
        certificate_digest=certificate_digest,
    )
    _return_value: GalerkinAxialCapCoefficientCertificate = certificate
    return _return_value  # noqa: RET504


@jaxtyped(typechecker=beartype)
def _make_axial_cap_floor_proof(  # noqa: PLR0913,PLR0915
    coefficient_certificate: GalerkinAxialCapCoefficientCertificate,
    gram_degree: Int[Array, "..."],
    gram_subinterval_width: Float[Array, "..."],
    gram_midpoint_shift_lower_bound: Float[Array, "..."],
    gram_entry_frobenius_error_upper_bound: Float[Array, "..."],
    plateau_gram_lower_bound: Float[Array, "..."],
    dimensionless_exact_floor_lower_bound: Float[Array, "..."],
    exact_target_physical_floor_lower_bound: Float[Array, "..."],
    realized_dimensionless_floor_lower_bound: Float[Array, "..."],
    scale_error_bound: Float[Array, "..."],
    physical_operator_error_upper_bound: Float[Array, "..."],
    realized_physical_floor_lower_bound: Float[Array, "..."],
    exact_target_floor_eligible: Bool[Array, ""],
    realized_floor_eligible: Bool[Array, ""],
    maximum_gram_degree: Int[Array, "..."],
    gram_precision_bits: Int[Array, "..."],
    ldl_iteration_count: Int[Array, "..."],
    gram_work_count: Int[Array, "..."],
    maximum_gram_work: Int[Array, "..."],
    *,
    exact_target_failure: GalerkinAxialCapExactFloorFailure,
    realized_floor_failure: GalerkinAxialCapRealizedFloorFailure,
    realized_floor_route: GalerkinAxialCapRealizedFloorRoute,
    gram_subinterval_numerator: str,
    gram_subinterval_denominator: str,
    exact_floor_target: str,
    gram_proof_route: str,
    gram_work_scope: str,
    realized_floor_scope: str,
    completion_scope: str,
    parent_certificate_digest: str,
    gram_transcript_digest: str,
    proof_digest: str,
) -> GalerkinAxialCapFloorProof:
    """PRIVATE: Validate and store one LVT.29--LVT.32 floor attempt.

    Parameters
    ----------
    coefficient_certificate : GalerkinAxialCapCoefficientCertificate
        Internal value used by this helper.
    gram_degree : Int[Array, '...']
        Internal value used by this helper.
    gram_subinterval_width : Float[Array, '...']
        Internal value used by this helper.
    gram_midpoint_shift_lower_bound : Float[Array, '...']
        Internal value used by this helper.
    gram_entry_frobenius_error_upper_bound : Float[Array, '...']
        Internal value used by this helper.
    plateau_gram_lower_bound : Float[Array, '...']
        Internal value used by this helper.
    dimensionless_exact_floor_lower_bound : Float[Array, '...']
        Internal value used by this helper.
    exact_target_physical_floor_lower_bound : Float[Array, '...']
        Internal value used by this helper.
    realized_dimensionless_floor_lower_bound : Float[Array, '...']
        Internal value used by this helper.
    scale_error_bound : Float[Array, '...']
        Internal value used by this helper.
    physical_operator_error_upper_bound : Float[Array, '...']
        Internal value used by this helper.
    realized_physical_floor_lower_bound : Float[Array, '...']
        Internal value used by this helper.
    exact_target_floor_eligible : Bool[Array, '']
        Internal value used by this helper.
    realized_floor_eligible : Bool[Array, '']
        Internal value used by this helper.
    maximum_gram_degree : Int[Array, '...']
        Internal value used by this helper.
    gram_precision_bits : Int[Array, '...']
        Internal value used by this helper.
    ldl_iteration_count : Int[Array, '...']
        Internal value used by this helper.
    gram_work_count : Int[Array, '...']
        Internal value used by this helper.
    maximum_gram_work : Int[Array, '...']
        Internal value used by this helper.
    exact_target_failure : GalerkinAxialCapExactFloorFailure
        Internal value used by this helper.
    realized_floor_failure : GalerkinAxialCapRealizedFloorFailure
        Internal value used by this helper.
    realized_floor_route : GalerkinAxialCapRealizedFloorRoute
        Internal value used by this helper.
    gram_subinterval_numerator : str
        Internal value used by this helper.
    gram_subinterval_denominator : str
        Internal value used by this helper.
    exact_floor_target : str
        Internal value used by this helper.
    gram_proof_route : str
        Internal value used by this helper.
    gram_work_scope : str
        Internal value used by this helper.
    realized_floor_scope : str
        Internal value used by this helper.
    completion_scope : str
        Internal value used by this helper.
    parent_certificate_digest : str
        Internal value used by this helper.
    gram_transcript_digest : str
        Internal value used by this helper.
    proof_digest : str
        Internal value used by this helper.

    Returns
    -------
    _return_value : GalerkinAxialCapFloorProof
        Internal result produced by this helper.
    """
    degree = jnp.asarray(gram_degree, dtype=jnp.int64)
    width = jnp.asarray(gram_subinterval_width, dtype=jnp.float64)
    midpoint = jnp.asarray(gram_midpoint_shift_lower_bound, dtype=jnp.float64)
    entry_error = jnp.asarray(
        gram_entry_frobenius_error_upper_bound, dtype=jnp.float64
    )
    gram_lower = jnp.asarray(plateau_gram_lower_bound, dtype=jnp.float64)
    exact_dimensionless = jnp.asarray(
        dimensionless_exact_floor_lower_bound, dtype=jnp.float64
    )
    exact_physical = jnp.asarray(
        exact_target_physical_floor_lower_bound, dtype=jnp.float64
    )
    realized_dimensionless = jnp.asarray(
        realized_dimensionless_floor_lower_bound, dtype=jnp.float64
    )
    scale_error = jnp.asarray(scale_error_bound, dtype=jnp.float64)
    physical_error = jnp.asarray(
        physical_operator_error_upper_bound, dtype=jnp.float64
    )
    realized_physical = jnp.asarray(
        realized_physical_floor_lower_bound, dtype=jnp.float64
    )
    exact_eligible = jnp.asarray(exact_target_floor_eligible, dtype=jnp.bool_)
    realized_eligible = jnp.asarray(realized_floor_eligible, dtype=jnp.bool_)
    degree_budget = jnp.asarray(maximum_gram_degree, dtype=jnp.int64)
    precision = jnp.asarray(gram_precision_bits, dtype=jnp.int64)
    iterations = jnp.asarray(ldl_iteration_count, dtype=jnp.int64)
    work_count = jnp.asarray(gram_work_count, dtype=jnp.int64)
    work_budget = jnp.asarray(maximum_gram_work, dtype=jnp.int64)
    scalar_values = (
        degree,
        width,
        midpoint,
        entry_error,
        gram_lower,
        exact_dimensionless,
        exact_physical,
        realized_dimensionless,
        scale_error,
        physical_error,
        realized_physical,
        exact_eligible,
        realized_eligible,
        degree_budget,
        precision,
        iterations,
        work_count,
        work_budget,
    )
    _raise_if(
        any(value.shape != () for value in scalar_values),
        "floor-proof fields must be scalars",
    )
    _raise_if(
        not _valid_unsigned_decimal(gram_subinterval_numerator, positive=True)
        or not _valid_unsigned_decimal(
            gram_subinterval_denominator, positive=True
        ),
        "exact Gram subinterval must use canonical positive decimal strings",
    )
    for declaration, name in (
        (exact_floor_target, "exact_floor_target"),
        (gram_proof_route, "gram_proof_route"),
        (gram_work_scope, "gram_work_scope"),
        (realized_floor_scope, "realized_floor_scope"),
        (completion_scope, "completion_scope"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")
    for digest, name in (
        (parent_certificate_digest, "parent_certificate_digest"),
        (gram_transcript_digest, "gram_transcript_digest"),
        (proof_digest, "proof_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    exact_expected: bool = (
        exact_target_failure is GalerkinAxialCapExactFloorFailure.NONE
    )
    exact_not_finite = (
        GalerkinAxialCapRealizedFloorFailure.EXACT_TARGET_FLOOR_NOT_FINITE
    )
    coefficient_not_finite = (
        GalerkinAxialCapRealizedFloorFailure.COEFFICIENT_CERTIFICATE_NOT_FINITE
    )
    _raise_if(
        (not exact_expected)
        and realized_floor_failure is not exact_not_finite,
        "an ineligible exact floor requires the exact-target realized failure",
    )
    _raise_if(
        exact_expected and realized_floor_failure is exact_not_finite,
        "an eligible exact floor cannot use the exact-target realized failure",
    )
    realized_expected: bool = (
        exact_expected
        and realized_floor_failure is GalerkinAxialCapRealizedFloorFailure.NONE
    )
    route_is_exact_scale: bool = (
        realized_floor_route
        is GalerkinAxialCapRealizedFloorRoute.EXACT_FROZEN_SCALE_LVT32A
    )
    coefficient_is_finite = coefficient_certificate.finite_certificate
    scale_is_exact = (
        coefficient_certificate.absorber.exact_cap_scale
        == coefficient_certificate.absorber.algebraic_cap_scale
    )
    invalid: Bool[Array, ""] = (
        (degree < 0)
        | (degree_budget < 0)
        | (precision <= 0)
        | (iterations <= 0)
        | (work_count <= 0)
        | (work_budget <= 0)
        | (scale_is_exact != route_is_exact_scale)
        | (exact_eligible != exact_expected)
        | (realized_eligible != realized_expected)
        | (realized_eligible & (~coefficient_is_finite))
        | (
            exact_eligible
            & (~coefficient_is_finite)
            & (realized_floor_failure is not coefficient_not_finite)
        )
        | (
            exact_eligible
            & coefficient_is_finite
            & (realized_floor_failure is coefficient_not_finite)
        )
        | jnp.any(
            jnp.isnan(
                jnp.asarray(
                    (
                        width,
                        midpoint,
                        entry_error,
                        gram_lower,
                        exact_dimensionless,
                        exact_physical,
                        realized_dimensionless,
                        scale_error,
                        physical_error,
                        realized_physical,
                    )
                )
            )
        )
        | (entry_error < 0.0)
        | (scale_error < 0.0)
        | (physical_error < 0.0)
        | (coefficient_is_finite & (~jnp.isfinite(physical_error)))
        | ((~coefficient_is_finite) & (~jnp.isinf(physical_error)))
    )
    exact_invalid: Bool[Array, ""] = exact_eligible & (
        (degree > degree_budget)
        | (work_count > work_budget)
        | (~jnp.isfinite(width))
        | (width <= 0.0)
        | (width >= _STRICT_HALF_WIDTH)
        | (~jnp.isfinite(midpoint))
        | (~jnp.isfinite(entry_error))
        | (~jnp.isfinite(gram_lower))
        | (gram_lower <= 0.0)
        | (~jnp.isfinite(exact_dimensionless))
        | (exact_dimensionless <= 0.0)
        | (~jnp.isfinite(exact_physical))
        | (exact_physical <= 0.0)
    )
    realized_invalid: Bool[Array, ""] = realized_eligible & (
        (~jnp.isfinite(realized_dimensionless))
        | (realized_dimensionless <= 0.0)
        | (~jnp.isfinite(realized_physical))
        | (realized_physical <= 0.0)
    )
    checked_degree = eqx.error_if(
        degree,
        invalid | exact_invalid | realized_invalid,
        "axial CAP floor fields contradict their typed outcomes",
    )
    proof = GalerkinAxialCapFloorProof(
        coefficient_certificate=coefficient_certificate,
        gram_degree=checked_degree,
        gram_subinterval_width=width,
        gram_midpoint_shift_lower_bound=midpoint,
        gram_entry_frobenius_error_upper_bound=entry_error,
        plateau_gram_lower_bound=gram_lower,
        dimensionless_exact_floor_lower_bound=exact_dimensionless,
        exact_target_physical_floor_lower_bound=exact_physical,
        realized_dimensionless_floor_lower_bound=realized_dimensionless,
        scale_error_bound=scale_error,
        physical_operator_error_upper_bound=physical_error,
        realized_physical_floor_lower_bound=realized_physical,
        exact_target_floor_eligible=exact_eligible,
        realized_floor_eligible=realized_eligible,
        maximum_gram_degree=degree_budget,
        gram_precision_bits=precision,
        ldl_iteration_count=iterations,
        gram_work_count=work_count,
        maximum_gram_work=work_budget,
        exact_target_failure=exact_target_failure,
        realized_floor_failure=realized_floor_failure,
        realized_floor_route=realized_floor_route,
        gram_subinterval_numerator=gram_subinterval_numerator,
        gram_subinterval_denominator=gram_subinterval_denominator,
        exact_floor_target=exact_floor_target.strip(),
        gram_proof_route=gram_proof_route.strip(),
        gram_work_scope=gram_work_scope.strip(),
        realized_floor_scope=realized_floor_scope.strip(),
        completion_scope=completion_scope.strip(),
        parent_certificate_digest=parent_certificate_digest,
        gram_transcript_digest=gram_transcript_digest,
        proof_digest=proof_digest,
    )
    _return_value: GalerkinAxialCapFloorProof = proof
    return _return_value  # noqa: RET504


__all__: list[str] = [
    "GalerkinAxialCapCoefficientCertificate",
    "GalerkinAxialCapCoefficientFailure",
    "GalerkinAxialCapExactFloorFailure",
    "GalerkinAxialCapFloorProof",
    "GalerkinAxialCapRealizedFloorFailure",
    "GalerkinAxialCapRealizedFloorRoute",
    "GalerkinAxialCellAbsorber",
]
