r"""Define scalar Galerkin operator and solve-result carriers.

Extended Summary
----------------
This module owns the sparse data carriers for one fixed complex-linear
Galerkin operator and its algebraic iterative-solve result. The operator
stores coordinate data for the real interaction and an absorber factor. It
does not assemble a dense matrix or claim an outward numerical certificate.

Routine Listings
----------------
:class:`GalerkinCertificateReason`
    Store the reason that a Galerkin result lacks certification.
:class:`GalerkinOperator`
    Store one fixed complex-linear scalar Galerkin operator.
:class:`GalerkinSolveMethod`
    Store the selected Galerkin iterative-solve method.
:class:`GalerkinSolveResult`
    Store one algebraic scalar Galerkin solve result.
:class:`GalerkinSolveStatus`
    Store the termination status of a Galerkin solve.
:func:`create_galerkin_operator`
    Create a validated scalar Galerkin operator carrier.
:func:`create_galerkin_solve_result`
    Create a validated algebraic Galerkin solve result.

Notes
-----
The represented fixed operator is
:math:`H_{\mathrm{alg}}=D-R-i\varepsilon_{\mathrm{CAP}}G^*G`.
The factor form makes the absorber positive semidefinite. It does not prove
strict positivity, a stability bound, or an outward residual enclosure.
"""

from enum import Enum, IntEnum

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Float,
    Float64,
    Int,
    Int32,
    Int64,
    jaxtyped,
)

from ptyrodactyl._numeric import has_subnormal_components

from .custom_types import scalar_bool, scalar_float, scalar_int


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a true structural condition.

    Parameters
    ----------
    condition : bool
        Whether the structural contract is invalid.
    message : str
        Error message for the rejected contract.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _checked_finite_vector(
    values: Complex[Array, " n"] | Float[Array, " n"],
    name: str,
) -> Complex[Array, " n"] | Float[Array, " n"]:
    """PRIVATE: Attach a traced finite-value check to one numeric vector.

    Parameters
    ----------
    values : Complex[Array, " n"] | Float[Array, " n"]
        Numeric vector to validate.
    name : str
        Field name included in the runtime error.

    Returns
    -------
    checked_values : Complex[Array, " n"] | Float[Array, " n"]
        Input vector with a traced finite-value assertion.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any vector element is non-finite under compiled execution.
    """
    checked_values: Complex[Array, " n"] | Float[Array, " n"] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        f"{name} must contain only finite values",
    )
    return checked_values


def _checked_nonnegative_scalar(
    value: scalar_float,
    name: str,
) -> Float64[Array, ""]:
    """PRIVATE: Convert and validate one finite non-negative scalar.

    Parameters
    ----------
    value : scalar_float
        Scalar value to convert to binary64.
    name : str
        Field name included in validation errors.

    Returns
    -------
    checked_value : Float64[Array, ""]
        Binary64 scalar with traced finite and non-negative assertions.

    Raises
    ------
    ValueError
        If ``value`` is not scalar-shaped.
    equinox.EquinoxRuntimeError
        If ``value`` is non-finite or negative under compiled execution.
    """
    value_array: Float64[Array, ""] = jnp.asarray(value, dtype=jnp.float64)
    scalar_shape: Tuple[()] = ()
    _raise_if(value_array.shape != scalar_shape, f"{name} must be a scalar")
    checked_value: Float64[Array, ""] = eqx.error_if(
        value_array,
        (~jnp.isfinite(value_array)) | (value_array < 0.0),
        f"{name} must be finite and non-negative",
    )
    return checked_value


def _checked_nonnegative_integer(
    value: scalar_int,
    name: str,
) -> Int32[Array, ""]:
    """PRIVATE: Convert and validate one non-negative integer scalar.

    Parameters
    ----------
    value : scalar_int
        Integer scalar to convert to signed 32-bit storage.
    name : str
        Field name included in validation errors.

    Returns
    -------
    checked_value : Int32[Array, ""]
        Signed 32-bit scalar with a traced non-negative assertion.

    Raises
    ------
    ValueError
        If ``value`` is boolean or is not scalar-shaped.
    equinox.EquinoxRuntimeError
        If ``value`` is negative under compiled execution.
    """
    _raise_if(isinstance(value, bool), f"{name} must not be boolean")
    value_array: Int32[Array, ""] = jnp.asarray(value, dtype=jnp.int32)
    scalar_shape: Tuple[()] = ()
    _raise_if(value_array.shape != scalar_shape, f"{name} must be a scalar")
    checked_value: Int32[Array, ""] = eqx.error_if(
        value_array,
        value_array < 0,
        f"{name} must be non-negative",
    )
    return checked_value


class GalerkinCertificateReason(str, Enum):
    """Store the reason that a Galerkin result lacks certification.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Attributes
    ----------
    NO_OUTWARD_RESIDUAL_BOUND : str
        No independently enclosed outward residual is available.
    NO_STABILITY_BOUND : str
        No accepted lower singular-value bound is available.
    STATE_BUDGET_MISSED : str
        The residual-to-state bound exceeds the declared state budget.
    INVALID_OPERATOR_CONTRACT : str
        The operator does not satisfy the required finite-target contract.
    """

    NO_OUTWARD_RESIDUAL_BOUND = "no_outward_residual_bound"
    NO_STABILITY_BOUND = "no_stability_bound"
    STATE_BUDGET_MISSED = "state_budget_missed"
    INVALID_OPERATOR_CONTRACT = "invalid_operator_contract"


class GalerkinOperator(eqx.Module):
    r"""Store one fixed complex-linear scalar Galerkin operator.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Attributes
    ----------
    free_diagonal : Float[Array, " n"]
        Real shifted free-operator diagonal in inverse-square Angstroms.
    interaction_rows : Int[Array, " p"]
        Row indices of the unique Hermitian interaction COO entries.
    interaction_columns : Int[Array, " p"]
        Column indices of the unique Hermitian interaction COO entries.
    interaction_values : Complex[Array, " p"]
        Interaction matrix entries in inverse-square Angstroms.
    absorber_factor_rows : Int[Array, " q"]
        Row indices of the unique absorber-factor COO entries.
    absorber_factor_columns : Int[Array, " q"]
        Column indices of the unique absorber-factor COO entries.
    absorber_factor_values : Complex[Array, " q"]
        Dimensionless absorber-factor entries.
    cap_scale : Float64[Array, ""]
        Positive normal-range physical CAP scale in inverse-square Angstroms.
    absorber_factor_size : int
        Static row count of the absorber factor. This value affects tracing.

    See Also
    --------
    :func:`create_galerkin_operator`
        Create and validate a :class:`GalerkinOperator`.

    Notes
    -----
    The carrier represents
    :math:`H_{\mathrm{alg}}=D-R-i\varepsilon_{\mathrm{CAP}}G^*G`.
    Factorization guarantees only a positive-semidefinite absorber. It does
    not establish strict positivity or a numerical certificate.
    """

    free_diagonal: Float[Array, " n"]
    interaction_rows: Int[Array, " p"]
    interaction_columns: Int[Array, " p"]
    interaction_values: Complex[Array, " p"]
    absorber_factor_rows: Int[Array, " q"]
    absorber_factor_columns: Int[Array, " q"]
    absorber_factor_values: Complex[Array, " q"]
    cap_scale: Float64[Array, ""]
    absorber_factor_size: int = eqx.field(static=True)


class GalerkinSolveMethod(str, Enum):
    """Store the selected Galerkin iterative-solve method.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Attributes
    ----------
    CGLS : str
        Conjugate-gradient least-squares iteration.
    LSQR : str
        Golub--Kahan LSQR iteration.
    """

    CGLS = "cgls"
    LSQR = "lsqr"


class GalerkinSolveResult(eqx.Module):
    """Store one algebraic scalar Galerkin solve result.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Attributes
    ----------
    field : Complex[Array, " n"]
        Computed retained-state coefficient vector.
    residual : Complex[Array, " n"]
        Independently recomputed algebraic residual vector.
    residual_norm : Float64[Array, ""]
        Norm of the independently recomputed algebraic residual.
    normal_residual_norm : Float64[Array, ""]
        Norm of the normal-equation residual.
    recurrence_residual_norm : Float64[Array, ""]
        Residual norm reported by the iterative recurrence.
    iterations : Int32[Array, ""]
        Number of completed solver iterations.
    operator_applications : Int32[Array, ""]
        Number of forward and adjoint operator applications.
    status : Int32[Array, ""]
        Integer code from :class:`GalerkinSolveStatus`.
    converged : Bool[Array, ""]
        Whether ``status`` equals :attr:`GalerkinSolveStatus.CONVERGED`.
    method : GalerkinSolveMethod
        Static iterative-solve method. This value affects tracing.
    certificate_reason : GalerkinCertificateReason
        Static reason that this algebraic result is not certified.

    See Also
    --------
    :func:`create_galerkin_solve_result`
        Create and validate a :class:`GalerkinSolveResult`.

    Notes
    -----
    ``residual`` is an algebraic residual only. The carrier does not assert an
    outward residual enclosure, a state-error bound, or detector-level
    certification.
    """

    field: Complex[Array, " n"]
    residual: Complex[Array, " n"]
    residual_norm: Float64[Array, ""]
    normal_residual_norm: Float64[Array, ""]
    recurrence_residual_norm: Float64[Array, ""]
    iterations: Int32[Array, ""]
    operator_applications: Int32[Array, ""]
    status: Int32[Array, ""]
    converged: Bool[Array, ""]
    method: GalerkinSolveMethod = eqx.field(static=True)
    certificate_reason: GalerkinCertificateReason = eqx.field(static=True)


class GalerkinSolveStatus(IntEnum):
    """Store the termination status of a Galerkin solve.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Attributes
    ----------
    CONVERGED : int
        The independently recomputed residual met its target.
    MAX_ITERATIONS : int
        The iteration limit was reached before convergence.
    BREAKDOWN : int
        The recurrence encountered a numerical breakdown.
    RESIDUAL_MISMATCH : int
        The independent retained-result residual rejected an internal stop.
    """

    CONVERGED = 0
    MAX_ITERATIONS = 1
    BREAKDOWN = 2
    RESIDUAL_MISMATCH = 3


@jaxtyped(typechecker=beartype)
def create_galerkin_operator(  # noqa: PLR0915
    free_diagonal: Float[Array, "..."],
    interaction_rows: Int[Array, "..."],
    interaction_columns: Int[Array, "..."],
    interaction_values: Complex[Array, "..."],
    absorber_factor_rows: Int[Array, "..."],
    absorber_factor_columns: Int[Array, "..."],
    absorber_factor_values: Complex[Array, "..."],
    cap_scale: scalar_float,
    absorber_factor_size: int,
) -> GalerkinOperator:
    r"""Create a validated scalar Galerkin operator carrier.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Implementation Logic
    --------------------
    1. Validate vector ranks, shared COO lengths, and static dimensions.
    2. Attach traced range, uniqueness, Hermitian, and finite-value checks.
    3. Store the sparse interaction and absorber factor without densification.

    Parameters
    ----------
    free_diagonal : Float[Array, "..."]
        Real shifted free diagonal in inverse-square Angstroms. It must be a
        nonempty vector.
    interaction_rows : Int[Array, "..."]
        Interaction COO row indices. They must form a vector of length ``p``.
    interaction_columns : Int[Array, "..."]
        Interaction COO column indices. They must have length ``p``.
    interaction_values : Complex[Array, "..."]
        Unique Hermitian interaction COO values. They must have length ``p``.
    absorber_factor_rows : Int[Array, "..."]
        Absorber-factor COO row indices. They must form a nonempty vector of
        length ``q``.
    absorber_factor_columns : Int[Array, "..."]
        Absorber-factor COO column indices. They must have length ``q``.
    absorber_factor_values : Complex[Array, "..."]
        Unique absorber-factor COO values. They must have length ``q``.
    cap_scale : scalar_float
        Positive normal-range physical CAP scale in inverse-square Angstroms.
    absorber_factor_size : int
        Static positive row count of the absorber factor. Changing this value
        causes retracing.

    Returns
    -------
    operator : GalerkinOperator
        Validated fixed complex-linear operator carrier.

    Raises
    ------
    ValueError
        If an array rank, COO length, or static factor size is invalid.
    equinox.EquinoxRuntimeError
        If an index, numeric value, CAP scale, uniqueness predicate, or
        Hermitian predicate is invalid during traced execution.

    Notes
    -----
    The interaction comparison sorts forward and reverse COO keys before it
    compares conjugate values. The absorber factor is nonempty and unique but
    need not be Hermitian. Its Gram form is positive semidefinite only.
    """
    _raise_if(
        isinstance(absorber_factor_size, bool),
        "absorber_factor_size must not be boolean",
    )
    _raise_if(
        absorber_factor_size <= 0,
        "absorber_factor_size must be positive",
    )

    free_array: Float[Array, " n"] = jnp.asarray(free_diagonal)
    interaction_rows_array: Int[Array, " p"] = jnp.asarray(interaction_rows)
    interaction_columns_array: Int[Array, " p"] = jnp.asarray(
        interaction_columns
    )
    interaction_values_array: Complex[Array, " p"] = jnp.asarray(
        interaction_values
    )
    absorber_rows_array: Int[Array, " q"] = jnp.asarray(absorber_factor_rows)
    absorber_columns_array: Int[Array, " q"] = jnp.asarray(
        absorber_factor_columns
    )
    absorber_values_array: Complex[Array, " q"] = jnp.asarray(
        absorber_factor_values
    )
    cap_scale_array: Float64[Array, ""] = jnp.asarray(
        cap_scale,
        dtype=jnp.float64,
    )

    vector_rank: int = 1
    scalar_shape: Tuple[()] = ()
    _raise_if(free_array.ndim != vector_rank, "free_diagonal must be 1D")
    _raise_if(free_array.shape[0] == 0, "free_diagonal must be nonempty")
    _raise_if(
        interaction_rows_array.ndim != vector_rank,
        "interaction_rows must be 1D",
    )
    _raise_if(
        interaction_columns_array.ndim != vector_rank,
        "interaction_columns must be 1D",
    )
    _raise_if(
        interaction_values_array.ndim != vector_rank,
        "interaction_values must be 1D",
    )
    interaction_shape: Tuple[int, ...] = interaction_rows_array.shape
    _raise_if(
        interaction_columns_array.shape != interaction_shape
        or interaction_values_array.shape != interaction_shape,
        "interaction COO arrays must have matching shapes",
    )
    _raise_if(
        absorber_rows_array.ndim != vector_rank,
        "absorber_factor_rows must be 1D",
    )
    _raise_if(
        absorber_columns_array.ndim != vector_rank,
        "absorber_factor_columns must be 1D",
    )
    _raise_if(
        absorber_values_array.ndim != vector_rank,
        "absorber_factor_values must be 1D",
    )
    absorber_shape: Tuple[int, ...] = absorber_rows_array.shape
    _raise_if(
        absorber_columns_array.shape != absorber_shape
        or absorber_values_array.shape != absorber_shape,
        "absorber factor COO arrays must have matching shapes",
    )
    _raise_if(
        absorber_rows_array.shape[0] == 0,
        "absorber factor must be nonempty",
    )
    _raise_if(
        cap_scale_array.shape != scalar_shape,
        "cap_scale must be a scalar",
    )

    state_size: int = free_array.shape[0]
    interaction_key_rows: Int64[Array, " p"] = interaction_rows_array.astype(
        jnp.int64
    )
    interaction_key_columns: Int64[Array, " p"] = (
        interaction_columns_array.astype(jnp.int64)
    )
    interaction_keys: Int64[Array, " p"] = (
        interaction_key_rows * state_size + interaction_key_columns
    )
    interaction_order: Int64[Array, " p"] = jnp.argsort(interaction_keys)
    sorted_interaction_keys: Int64[Array, " p"] = interaction_keys[
        interaction_order
    ]
    sorted_interaction_values: Complex[Array, " p"] = interaction_values_array[
        interaction_order
    ]
    reverse_keys: Int64[Array, " p"] = (
        interaction_key_columns * state_size + interaction_key_rows
    )
    reverse_order: Int64[Array, " p"] = jnp.argsort(reverse_keys)
    sorted_reverse_keys: Int64[Array, " p"] = reverse_keys[reverse_order]
    sorted_reverse_values: Complex[Array, " p"] = jnp.conj(
        interaction_values_array[reverse_order]
    )
    duplicate_interaction: Bool[Array, ""] = jnp.any(
        sorted_interaction_keys[1:] == sorted_interaction_keys[:-1]
    )
    invalid_interaction_range: Bool[Array, ""] = jnp.any(
        (interaction_rows_array < 0)
        | (interaction_rows_array >= state_size)
        | (interaction_columns_array < 0)
        | (interaction_columns_array >= state_size)
    )
    nonhermitian_interaction: Bool[Array, ""] = jnp.any(
        sorted_interaction_keys != sorted_reverse_keys
    ) | jnp.any(sorted_interaction_values != sorted_reverse_values)

    absorber_key_rows: Int64[Array, " q"] = absorber_rows_array.astype(
        jnp.int64
    )
    absorber_key_columns: Int64[Array, " q"] = absorber_columns_array.astype(
        jnp.int64
    )
    absorber_keys: Int64[Array, " q"] = (
        absorber_key_rows * state_size + absorber_key_columns
    )
    sorted_absorber_keys: Int64[Array, " q"] = jnp.sort(absorber_keys)
    duplicate_absorber: Bool[Array, ""] = jnp.any(
        sorted_absorber_keys[1:] == sorted_absorber_keys[:-1]
    )
    invalid_absorber_range: Bool[Array, ""] = jnp.any(
        (absorber_rows_array < 0)
        | (absorber_rows_array >= absorber_factor_size)
        | (absorber_columns_array < 0)
        | (absorber_columns_array >= state_size)
    )
    absorber_magnitudes: Float[Array, " q"] = jnp.abs(absorber_values_array)
    scaled_absorber_powers: Float[Array, " q"] = (
        cap_scale_array * absorber_magnitudes**2
    )
    lost_absorber_power: Bool[Array, ""] = jnp.any(
        (absorber_magnitudes > 0.0)
        & (
            (~jnp.isfinite(scaled_absorber_powers))
            | (scaled_absorber_powers < jnp.finfo(jnp.float64).tiny)
        )
    )

    checked_free: Float[Array, " n"] = eqx.error_if(
        free_array,
        jnp.any(~jnp.isfinite(free_array))
        | has_subnormal_components(free_array),
        "free_diagonal must contain only finite normal-range values or zero",
    )
    checked_interaction_rows: Int[Array, " p"] = eqx.error_if(
        interaction_rows_array,
        invalid_interaction_range,
        "interaction indices must lie in the state range",
    )
    checked_interaction_columns: Int[Array, " p"] = eqx.error_if(
        interaction_columns_array,
        duplicate_interaction,
        "interaction COO coordinates must be unique",
    )
    checked_interaction_values: Complex[Array, " p"] = eqx.error_if(
        interaction_values_array,
        jnp.any(~jnp.isfinite(interaction_values_array))
        | has_subnormal_components(interaction_values_array)
        | nonhermitian_interaction,
        "interaction COO values must be finite, normal-range, and Hermitian",
    )
    checked_absorber_rows: Int[Array, " q"] = eqx.error_if(
        absorber_rows_array,
        invalid_absorber_range,
        "absorber factor indices must lie in their matrix ranges",
    )
    checked_absorber_columns: Int[Array, " q"] = eqx.error_if(
        absorber_columns_array,
        duplicate_absorber,
        "absorber factor COO coordinates must be unique",
    )
    checked_absorber_values: Complex[Array, " q"] = eqx.error_if(
        absorber_values_array,
        jnp.any(~jnp.isfinite(absorber_values_array))
        | has_subnormal_components(absorber_values_array),
        "absorber_factor_values must contain only finite normal-range values",
    )
    checked_cap_scale: Float64[Array, ""] = eqx.error_if(
        cap_scale_array,
        (~jnp.isfinite(cap_scale_array))
        | (cap_scale_array < jnp.finfo(jnp.float64).tiny)
        | lost_absorber_power,
        "cap_scale and absorber factor must preserve finite normal-range "
        "absorber products",
    )

    operator: GalerkinOperator = GalerkinOperator(
        free_diagonal=checked_free,
        interaction_rows=checked_interaction_rows,
        interaction_columns=checked_interaction_columns,
        interaction_values=checked_interaction_values,
        absorber_factor_rows=checked_absorber_rows,
        absorber_factor_columns=checked_absorber_columns,
        absorber_factor_values=checked_absorber_values,
        cap_scale=checked_cap_scale,
        absorber_factor_size=absorber_factor_size,
    )
    return operator


@jaxtyped(typechecker=beartype)
def create_galerkin_solve_result(  # noqa: PLR0913
    field: Complex[Array, "..."],
    residual: Complex[Array, "..."],
    residual_norm: scalar_float,
    normal_residual_norm: scalar_float,
    recurrence_residual_norm: scalar_float,
    iterations: scalar_int,
    operator_applications: scalar_int,
    status: scalar_int,
    converged: scalar_bool,
    method: GalerkinSolveMethod | str,
    certificate_reason: GalerkinCertificateReason | str,
) -> GalerkinSolveResult:
    """Create a validated algebraic Galerkin solve result.

    :see: :class:`~.test_born_types.TestGalerkinCarriers`

    Implementation Logic
    --------------------
    1. Validate field, residual, and scalar structures.
    2. Attach traced finite, non-negative, status, and consistency checks.
    3. Store method and noncertificate reason as static metadata.

    Parameters
    ----------
    field : Complex[Array, "..."]
        Computed nonempty retained-state coefficient vector.
    residual : Complex[Array, "..."]
        Independently recomputed algebraic residual with the field shape.
    residual_norm : scalar_float
        Finite non-negative algebraic residual norm.
    normal_residual_norm : scalar_float
        Finite non-negative normal-equation residual norm.
    recurrence_residual_norm : scalar_float
        Finite non-negative residual norm from the iterative recurrence.
    iterations : scalar_int
        Non-negative completed iteration count.
    operator_applications : scalar_int
        Non-negative forward and adjoint application count.
    status : scalar_int
        Integer value of :class:`GalerkinSolveStatus`.
    converged : scalar_bool
        Whether ``status`` is :attr:`GalerkinSolveStatus.CONVERGED`.
    method : GalerkinSolveMethod | str
        Static iterative-solve method. Changing it causes retracing.
    certificate_reason : GalerkinCertificateReason | str
        Static reason that this algebraic result is not certified. Changing it
        causes retracing.

    Returns
    -------
    result : GalerkinSolveResult
        Validated algebraic solve-result carrier.

    Raises
    ------
    ValueError
        If field, residual, or scalar structures are invalid.
    equinox.EquinoxRuntimeError
        If a value is non-finite or negative, the status is invalid, or the
        convergence flag disagrees with the status during traced execution.

    Notes
    -----
    The factory does not infer an outward enclosure from ``residual`` or its
    norm. ``certificate_reason`` records why certification is unavailable.
    """
    checked_method: GalerkinSolveMethod = GalerkinSolveMethod(method)
    checked_certificate_reason: GalerkinCertificateReason = (
        GalerkinCertificateReason(certificate_reason)
    )
    field_array: Complex[Array, " n"] = jnp.asarray(field)
    residual_array: Complex[Array, " n"] = jnp.asarray(residual)
    vector_rank: int = 1
    _raise_if(field_array.ndim != vector_rank, "field must be 1D")
    _raise_if(field_array.shape[0] == 0, "field must be nonempty")
    _raise_if(residual_array.ndim != vector_rank, "residual must be 1D")
    _raise_if(
        residual_array.shape != field_array.shape,
        "field and residual must have matching shapes",
    )

    checked_field_value: Complex[Array, " n"] | Float[Array, " n"] = (
        _checked_finite_vector(field_array, "field")
    )
    checked_residual_value: Complex[Array, " n"] | Float[Array, " n"] = (
        _checked_finite_vector(residual_array, "residual")
    )
    checked_field: Complex[Array, " n"] = checked_field_value
    checked_residual: Complex[Array, " n"] = checked_residual_value
    checked_residual_norm: Float64[Array, ""] = _checked_nonnegative_scalar(
        residual_norm,
        "residual_norm",
    )
    checked_normal_residual_norm: Float64[Array, ""] = (
        _checked_nonnegative_scalar(
            normal_residual_norm,
            "normal_residual_norm",
        )
    )
    checked_recurrence_residual_norm: Float64[Array, ""] = (
        _checked_nonnegative_scalar(
            recurrence_residual_norm,
            "recurrence_residual_norm",
        )
    )
    checked_iterations: Int32[Array, ""] = _checked_nonnegative_integer(
        iterations,
        "iterations",
    )
    checked_operator_applications: Int32[Array, ""] = (
        _checked_nonnegative_integer(
            operator_applications,
            "operator_applications",
        )
    )
    _raise_if(isinstance(status, bool), "status must not be boolean")
    status_array: Int32[Array, ""] = jnp.asarray(status, dtype=jnp.int32)
    converged_array: Bool[Array, ""] = jnp.asarray(converged, dtype=jnp.bool_)
    scalar_shape: Tuple[()] = ()
    _raise_if(status_array.shape != scalar_shape, "status must be a scalar")
    _raise_if(
        converged_array.shape != scalar_shape,
        "converged must be a scalar",
    )
    valid_status: Bool[Array, ""] = (
        (status_array == int(GalerkinSolveStatus.CONVERGED))
        | (status_array == int(GalerkinSolveStatus.MAX_ITERATIONS))
        | (status_array == int(GalerkinSolveStatus.BREAKDOWN))
        | (status_array == int(GalerkinSolveStatus.RESIDUAL_MISMATCH))
    )
    checked_status: Int32[Array, ""] = eqx.error_if(
        status_array,
        ~valid_status,
        "status must be a GalerkinSolveStatus value",
    )
    checked_converged: Bool[Array, ""] = eqx.error_if(
        converged_array,
        converged_array
        != (status_array == int(GalerkinSolveStatus.CONVERGED)),
        "converged must agree with status",
    )

    result: GalerkinSolveResult = GalerkinSolveResult(
        field=checked_field,
        residual=checked_residual,
        residual_norm=checked_residual_norm,
        normal_residual_norm=checked_normal_residual_norm,
        recurrence_residual_norm=checked_recurrence_residual_norm,
        iterations=checked_iterations,
        operator_applications=checked_operator_applications,
        status=checked_status,
        converged=checked_converged,
        method=checked_method,
        certificate_reason=checked_certificate_reason,
    )
    return result


__all__: list[str] = [
    "GalerkinCertificateReason",
    "GalerkinOperator",
    "GalerkinSolveMethod",
    "GalerkinSolveResult",
    "GalerkinSolveStatus",
    "create_galerkin_operator",
    "create_galerkin_solve_result",
]
