r"""Apply and solve a fixed-support scalar Galerkin system.

Extended Summary
----------------
This module applies either a frozen algebraic Galerkin realization or the
canonical manifested SC-1 target without assembling the dense operator. The
manifested path uses endpoint-safe Fourier products for its interaction and
absorber. The forward and adjoint actions are distinct, and CGLS and LSQR
stop on a directly recomputed original-system residual.

The returned residual is an algebraic floating-point diagnostic.  It is not
an outward residual enclosure or an RM-S6-I state certificate.  The implicit
root uses a custom VJP and a separately solved adjoint system.  Its admitted
derivative chart has fixed support and fixed sparse structure.

Routine Listings
----------------
:func:`apply_galerkin_adjoint`
    Apply the matrix-free adjoint Galerkin operator.
:func:`apply_galerkin_operator`
    Apply the matrix-free forward Galerkin operator.
:func:`cgls_solve`
    Solve a Galerkin system with CGLS and a fresh residual.
:func:`evaluate_galerkin_adjoint_residual`
    Evaluate a fresh adjoint-system algebraic residual.
:func:`evaluate_galerkin_residual`
    Evaluate a fresh forward-system algebraic residual.
:func:`implicit_galerkin_solve`
    Solve a Galerkin root with an implicit custom VJP.
:func:`lsqr_solve`
    Solve a Galerkin system with LSQR and a fresh residual.
:func:`shifted_free_diagonal`
    Construct the carrier-shifted free Galerkin diagonal.

Notes
-----
The coefficient inner product is ``Re(sum(conj(x) * y))``. Reciprocal
frequencies use cycles per Angstrom, while the carrier and wavenumber use
radians per Angstrom.  Support-changing derivatives are outside this module's
fixed reference chart.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from ptyrodactyl._numeric import (
    has_lost_subtraction,
    has_subnormal_components,
)
from ptyrodactyl.types import (
    GalerkinCertificateReason,
    GalerkinOperator,
    GalerkinPhysicalResidual,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    GalerkinTargetManifest,
    create_galerkin_solve_result,
    scalar_float,
    scalar_int,
)

from .system import (
    apply_galerkin_target,
    apply_galerkin_target_adjoint,
    evaluate_physical_galerkin_adjoint_residual,
    evaluate_physical_galerkin_residual,
)

_FREQUENCY_RANK: int = 2
_SPACE_DIMENSIONS: int = 3

type _GalerkinSystem = GalerkinOperator | GalerkinTargetManifest


class _CGLSState(NamedTuple):
    """Carry one scale-normalized CGLS iteration without a trajectory."""

    field: Complex[Array, "n"]
    residual: Complex[Array, "n"]
    adjoint_residual: Complex[Array, "n"]
    direction: Complex[Array, "n"]
    adjoint_scale: Float[Array, ""]
    direction_scale_ratio: Float[Array, ""]
    residual_norm: Float[Array, ""]
    recurrence_residual_norm: Float[Array, ""]
    iterations: Int[Array, ""]
    operator_applications: Int[Array, ""]
    converged: Bool[Array, ""]
    breakdown: Bool[Array, ""]


class _LSQRState(NamedTuple):
    """Carry one LSQR iteration without exposing solver trajectory."""

    field: Complex[Array, "n"]
    left_vector: Complex[Array, "n"]
    right_vector: Complex[Array, "n"]
    direction: Complex[Array, "n"]
    alpha: Float[Array, ""]
    beta: Float[Array, ""]
    rho_bar: Float[Array, ""]
    phi_bar: Float[Array, ""]
    recurrence_residual_norm: Float[Array, ""]
    iterations: Int[Array, ""]
    operator_applications: Int[Array, ""]
    converged: Bool[Array, ""]
    breakdown: Bool[Array, ""]


class _SolverInputs(NamedTuple):
    """Store checked Krylov inputs and the fresh-residual threshold."""

    source: Complex[Array, "n"]
    initial_field: Complex[Array, "n"]
    max_iterations: Int[Array, ""]
    stopping_threshold: Float[Array, ""]


def _complex_norm(vector: Complex[Array, "n"]) -> Float[Array, ""]:
    """Return the Euclidean norm induced by the SC-1 coefficient metric."""
    magnitudes: Float[Array, " n"] = jnp.abs(vector)
    scale: Float[Array, ""] = jnp.max(magnitudes)
    safe_scale: Float[Array, ""] = jnp.where(scale > 0.0, scale, 1.0)
    scaled_magnitudes: Float[Array, " n"] = magnitudes / safe_scale
    scaled_norm: Float[Array, ""] = jnp.sqrt(jnp.sum(scaled_magnitudes**2))
    finite_norm: Float[Array, ""] = scale * scaled_norm
    norm: Float[Array, ""] = jnp.where(
        scale == 0.0,
        0.0,
        jnp.where(jnp.isinf(scale), scale, finite_norm),
    )
    return norm


def _sparse_action(
    rows: Int[Array, "nnz"],
    columns: Int[Array, "nnz"],
    values: Complex[Array, "nnz"],
    vector: Complex[Array, "input_size"],
    output_size: int,
) -> Complex[Array, "output_size"]:
    """Apply one frozen COO map by gather and scatter-add."""
    dtype: jnp.dtype = jnp.result_type(values.dtype, vector.dtype)
    output: Complex[Array, "output_size"] = jnp.zeros(
        (output_size,), dtype=dtype
    )
    result: Complex[Array, "output_size"] = output.at[rows].add(
        values * vector[columns]
    )
    return result


def _sparse_adjoint_action(
    rows: Int[Array, "nnz"],
    columns: Int[Array, "nnz"],
    values: Complex[Array, "nnz"],
    vector: Complex[Array, "output_size"],
    input_size: int,
) -> Complex[Array, "input_size"]:
    """Apply the actual conjugate transpose of one frozen COO map."""
    dtype: jnp.dtype = jnp.result_type(values.dtype, vector.dtype)
    output: Complex[Array, "input_size"] = jnp.zeros(
        (input_size,), dtype=dtype
    )
    result: Complex[Array, "input_size"] = output.at[columns].add(
        jnp.conj(values) * vector[rows]
    )
    return result


def _interaction_action(
    operator: GalerkinOperator,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    """Apply the frozen sparse interaction realization."""
    state_size: int = operator.free_diagonal.shape[0]
    interaction: Complex[Array, "n"] = _sparse_action(
        operator.interaction_rows,
        operator.interaction_columns,
        operator.interaction_values,
        field,
        state_size,
    )
    return interaction


def _interaction_adjoint_action(
    operator: GalerkinOperator,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    """Apply the actual adjoint of the sparse interaction realization."""
    state_size: int = operator.free_diagonal.shape[0]
    interaction: Complex[Array, "n"] = _sparse_adjoint_action(
        operator.interaction_rows,
        operator.interaction_columns,
        operator.interaction_values,
        field,
        state_size,
    )
    return interaction


def _absorber_action(
    operator: GalerkinOperator,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    """Apply the positive-semidefinite absorber Gramian G*G."""
    factor_field: Complex[Array, "factor_size"] = _sparse_action(
        operator.absorber_factor_rows,
        operator.absorber_factor_columns,
        operator.absorber_factor_values,
        field,
        operator.absorber_factor_size,
    )
    state_size: int = operator.free_diagonal.shape[0]
    absorber: Complex[Array, "n"] = _sparse_adjoint_action(
        operator.absorber_factor_rows,
        operator.absorber_factor_columns,
        operator.absorber_factor_values,
        factor_field,
        state_size,
    )
    return absorber


@jaxtyped(typechecker=beartype)
def shifted_free_diagonal(
    reciprocal_frequencies: Float[Array, "n d"],
    carrier: Float[Array, "d"],
    wavenumber: scalar_float,
) -> Float[Array, "n"]:
    r"""Construct the carrier-shifted free Galerkin diagonal.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    reciprocal_frequencies : Float[Array, "n d"]
        Retained reciprocal-lattice frequencies in cycles per Angstrom.
    carrier : Float[Array, "d"]
        Real incident-carrier components in radians per Angstrom.
    wavenumber : scalar_float
        Positive vacuum angular wavenumber in radians per Angstrom.

    Returns
    -------
    free_diagonal : Float[Array, "n"]
        Values ``|carrier + 2 pi g|^2 - wavenumber^2`` in inverse square
        Angstroms.

    Raises
    ------
    ValueError
        If the static ranks or dimensions disagree.
    equinox.EquinoxRuntimeError
        If a frequency, carrier component, or wavenumber is invalid during
        traced execution.

    Notes
    -----
    Runtime validation rejects non-finite frequencies or carriers and a
    non-finite or non-positive wavenumber in eager and compiled execution.
    """
    if reciprocal_frequencies.ndim != _FREQUENCY_RANK:
        raise ValueError("reciprocal_frequencies must have rank two")
    if carrier.ndim != 1:
        raise ValueError("carrier must have rank one")
    if reciprocal_frequencies.shape[1] != _SPACE_DIMENSIONS:
        raise ValueError("reciprocal frequencies must have three dimensions")
    if reciprocal_frequencies.shape[1] != carrier.shape[0]:
        raise ValueError("carrier dimension must match reciprocal frequencies")

    wavenumber_array: Float[Array, ""] = jnp.asarray(wavenumber)
    if wavenumber_array.shape != ():
        raise ValueError("wavenumber must be a scalar")
    checked_frequencies: Float[Array, "n d"] = eqx.error_if(
        reciprocal_frequencies,
        jnp.any(~jnp.isfinite(reciprocal_frequencies)),
        "reciprocal_frequencies must be finite",
    )
    checked_carrier: Float[Array, "d"] = eqx.error_if(
        carrier,
        jnp.any(~jnp.isfinite(carrier)),
        "carrier must be finite",
    )
    checked_wavenumber: Float[Array, ""] = eqx.error_if(
        wavenumber_array,
        (~jnp.isfinite(wavenumber_array)) | (wavenumber_array <= 0.0),
        "wavenumber must be finite and positive",
    )
    shifted_frequencies: Float[Array, "n d"] = (
        checked_carrier[None, :] + 2.0 * jnp.pi * checked_frequencies
    )
    raw_free_diagonal: Float[Array, "n"] = (
        jnp.sum(shifted_frequencies**2, axis=1) - checked_wavenumber**2
    )
    free_diagonal: Float[Array, "n"] = eqx.error_if(
        raw_free_diagonal,
        jnp.any(~jnp.isfinite(raw_free_diagonal)),
        "derived free_diagonal must contain only finite values",
    )
    return free_diagonal


def _checked_action_vector(
    values: Complex[Array, "n"],
    name: str,
) -> Complex[Array, "n"]:
    """Reject non-finite and nonzero-subnormal action vectors."""
    checked: Complex[Array, "n"] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | has_subnormal_components(values),
        f"{name} must be finite and contain no nonzero subnormal components",
    )
    return checked


def _checked_residual_difference(
    source: Complex[Array, "n"],
    action: Complex[Array, "n"],
    name: str,
) -> Complex[Array, "n"]:
    """Subtract one action and reject a flushed nonzero component."""
    rounded_source, rounded_action = jax.lax.optimization_barrier(
        (source, action)
    )
    raw_residual: Complex[Array, "n"] = rounded_source - rounded_action
    checked_residual: Complex[Array, "n"] = _checked_action_vector(
        raw_residual, name
    )
    residual: Complex[Array, "n"] = eqx.error_if(
        checked_residual,
        has_subnormal_components(raw_residual)
        | has_lost_subtraction(
            rounded_source,
            rounded_action,
            raw_residual,
        ),
        f"{name} subtraction lost a nonzero component",
    )
    return residual


@jaxtyped(typechecker=beartype)
def apply_galerkin_operator(
    operator: _GalerkinSystem,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    r"""Apply the matrix-free forward Galerkin operator.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Frozen algebraic realization or canonical manifested SC-1 target.
    field : Complex[Array, "n"]
        State coefficients in the frozen support ordering.

    Returns
    -------
    applied_field : Complex[Array, "n"]
        Forward operator action in the same coefficient ordering.

    Raises
    ------
    ValueError
        If the field length differs from the operator state size.
    """
    if field.ndim != 1:
        raise ValueError("field must have rank one")
    if field.shape[0] != operator.free_diagonal.shape[0]:
        raise ValueError("field length must match the operator state size")
    checked_field: Complex[Array, "n"] = _checked_action_vector(field, "field")
    if isinstance(operator, GalerkinTargetManifest):
        raw_applied_field: Complex[Array, "n"] = apply_galerkin_target(
            operator,
            checked_field,
        )
    else:
        free_action: Complex[Array, "n"] = (
            operator.free_diagonal * checked_field
        )
        interaction_action: Complex[Array, "n"] = _interaction_action(
            operator, checked_field
        )
        absorber_action: Complex[Array, "n"] = _absorber_action(
            operator, checked_field
        )
        raw_applied_field = (
            free_action
            - interaction_action
            - 1j * operator.cap_scale * absorber_action
        )
    applied_field: Complex[Array, "n"] = _checked_action_vector(
        raw_applied_field, "applied_field"
    )
    return applied_field


@jaxtyped(typechecker=beartype)
def apply_galerkin_adjoint(
    operator: _GalerkinSystem,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    r"""Apply the matrix-free adjoint Galerkin operator.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Frozen algebraic realization or canonical manifested SC-1 target.
    field : Complex[Array, "n"]
        Adjoint-state coefficients in the frozen support ordering.

    Returns
    -------
    applied_field : Complex[Array, "n"]
        Actual conjugate-transpose action in the same ordering.

    Raises
    ------
    ValueError
        If the field length differs from the operator state size.

    Notes
    -----
    The algebraic branch derives the adjoint from its stored COO maps. The
    manifested branch delegates to its separately defined target-adjoint
    action. Neither branch calls the forward action or assumes the
    dissipative target is self-adjoint.
    """
    if field.ndim != 1:
        raise ValueError("field must have rank one")
    if field.shape[0] != operator.free_diagonal.shape[0]:
        raise ValueError("field length must match the operator state size")
    checked_field: Complex[Array, "n"] = _checked_action_vector(field, "field")
    if isinstance(operator, GalerkinTargetManifest):
        raw_applied_field: Complex[Array, "n"] = apply_galerkin_target_adjoint(
            operator,
            checked_field,
        )
    else:
        free_action: Complex[Array, "n"] = (
            operator.free_diagonal * checked_field
        )
        interaction_action: Complex[Array, "n"] = _interaction_adjoint_action(
            operator, checked_field
        )
        absorber_action: Complex[Array, "n"] = _absorber_action(
            operator, checked_field
        )
        raw_applied_field = (
            free_action
            - interaction_action
            + 1j * operator.cap_scale * absorber_action
        )
    applied_field: Complex[Array, "n"] = _checked_action_vector(
        raw_applied_field, "adjoint applied_field"
    )
    return applied_field


def _independent_forward_action(
    operator: GalerkinOperator,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    """Re-evaluate the forward action outside the Krylov recurrence."""
    state_size: int = operator.free_diagonal.shape[0]
    interaction: Complex[Array, "n"] = _sparse_action(
        operator.interaction_rows,
        operator.interaction_columns,
        operator.interaction_values,
        field,
        state_size,
    )
    factor_field: Complex[Array, "factor_size"] = _sparse_action(
        operator.absorber_factor_rows,
        operator.absorber_factor_columns,
        operator.absorber_factor_values,
        field,
        operator.absorber_factor_size,
    )
    absorber: Complex[Array, "n"] = _sparse_adjoint_action(
        operator.absorber_factor_rows,
        operator.absorber_factor_columns,
        operator.absorber_factor_values,
        factor_field,
        state_size,
    )
    action: Complex[Array, "n"] = (
        operator.free_diagonal * field
        - interaction
        - 1j * operator.cap_scale * absorber
    )
    return action


def _independent_adjoint_action(
    operator: GalerkinOperator,
    field: Complex[Array, "n"],
) -> Complex[Array, "n"]:
    """Re-evaluate the actual adjoint outside the Krylov recurrence."""
    state_size: int = operator.free_diagonal.shape[0]
    interaction: Complex[Array, "n"] = _sparse_adjoint_action(
        operator.interaction_rows,
        operator.interaction_columns,
        operator.interaction_values,
        field,
        state_size,
    )
    factor_field: Complex[Array, "factor_size"] = _sparse_action(
        operator.absorber_factor_rows,
        operator.absorber_factor_columns,
        operator.absorber_factor_values,
        field,
        operator.absorber_factor_size,
    )
    absorber: Complex[Array, "n"] = _sparse_adjoint_action(
        operator.absorber_factor_rows,
        operator.absorber_factor_columns,
        operator.absorber_factor_values,
        factor_field,
        state_size,
    )
    action: Complex[Array, "n"] = (
        operator.free_diagonal * field
        - interaction
        + 1j * operator.cap_scale * absorber
    )
    return action


@jaxtyped(typechecker=beartype)
def evaluate_galerkin_residual(
    operator: _GalerkinSystem,
    field: Complex[Array, "n"],
    source: Complex[Array, "n"],
) -> tuple[Complex[Array, "n"], Float[Array, ""]]:
    """Evaluate a fresh forward-system algebraic residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Frozen algebraic realization or canonical manifested SC-1 target.
    field : Complex[Array, "n"]
        Submitted state coefficients.
    source : Complex[Array, "n"]
        Original finite-system right-hand side.

    Returns
    -------
    residual : Complex[Array, "n"]
        Freshly evaluated original-system residual ``source - H field``.
    residual_norm : Float[Array, ""]
        Euclidean coefficient norm of ``residual``.

    Raises
    ------
    ValueError
        If the field and source shapes differ.

    Notes
    -----
    This function does not reuse a Krylov recurrence. The manifested branch
    uses direct coefficient contraction instead of the production FFT action.
    Neither floating-point diagnostic supplies an outward rounding enclosure;
    per-result stability invocation is a separate host-side check.
    """
    if field.shape != source.shape:
        raise ValueError("field and source must have the same shape")
    checked_field: Complex[Array, "n"] = _checked_action_vector(field, "field")
    checked_source: Complex[Array, "n"] = _checked_action_vector(
        source, "source"
    )
    if isinstance(operator, GalerkinTargetManifest):
        physical_residual: GalerkinPhysicalResidual = (
            evaluate_physical_galerkin_residual(
                operator, checked_field, checked_source
            )
        )
        residual: Complex[Array, "n"] = physical_residual.residual
        residual_norm: Float[Array, ""] = physical_residual.residual_norm
    else:
        action: Complex[Array, "n"] = _independent_forward_action(
            operator,
            checked_field,
        )
        residual = _checked_residual_difference(
            checked_source, action, "residual"
        )
        residual_norm = _complex_norm(residual)
    result: tuple[Complex[Array, "n"], Float[Array, ""]] = (
        residual,
        residual_norm,
    )
    return result


@jaxtyped(typechecker=beartype)
def evaluate_galerkin_adjoint_residual(
    operator: _GalerkinSystem,
    field: Complex[Array, "n"],
    source: Complex[Array, "n"],
) -> tuple[Complex[Array, "n"], Float[Array, ""]]:
    """Evaluate a fresh adjoint-system algebraic residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Frozen algebraic realization or canonical manifested SC-1 target.
    field : Complex[Array, "n"]
        Submitted adjoint-state coefficients.
    source : Complex[Array, "n"]
        Original adjoint-system right-hand side.

    Returns
    -------
    residual : Complex[Array, "n"]
        Freshly evaluated residual ``source - H* field``.
    residual_norm : Float[Array, ""]
        Euclidean coefficient norm of ``residual``.

    Raises
    ------
    ValueError
        If the field and source shapes differ.

    Notes
    -----
    This algebraic diagnostic is independent of the CGLS/LSQR recurrence but
    is not an outward adjoint-residual certificate.
    """
    if field.shape != source.shape:
        raise ValueError("field and source must have the same shape")
    checked_field: Complex[Array, "n"] = _checked_action_vector(field, "field")
    checked_source: Complex[Array, "n"] = _checked_action_vector(
        source, "source"
    )
    if isinstance(operator, GalerkinTargetManifest):
        physical_residual: GalerkinPhysicalResidual = (
            evaluate_physical_galerkin_adjoint_residual(
                operator,
                checked_field,
                checked_source,
            )
        )
        residual: Complex[Array, "n"] = physical_residual.residual
        residual_norm: Float[Array, ""] = physical_residual.residual_norm
    else:
        action: Complex[Array, "n"] = _independent_adjoint_action(
            operator,
            checked_field,
        )
        residual = _checked_residual_difference(
            checked_source, action, "adjoint residual"
        )
        residual_norm = _complex_norm(residual)
    result: tuple[Complex[Array, "n"], Float[Array, ""]] = (
        residual,
        residual_norm,
    )
    return result


def _solver_original_residual(
    operator: _GalerkinSystem,
    field: Complex[Array, "n"],
    source: Complex[Array, "n"],
    *,
    adjoint: bool,
) -> tuple[Complex[Array, "n"], Float[Array, ""]]:
    """Recompute a scalable original-system residual outside recurrence."""
    if isinstance(operator, GalerkinTargetManifest):
        if adjoint:
            action: Complex[Array, "n"] = apply_galerkin_adjoint(
                operator,
                field,
            )
        else:
            action = apply_galerkin_operator(operator, field)
        residual: Complex[Array, "n"] = _checked_residual_difference(
            source, action, "solver residual"
        )
        residual_norm: Float[Array, ""] = _complex_norm(residual)
        result: tuple[Complex[Array, "n"], Float[Array, ""]] = (
            residual,
            residual_norm,
        )
    elif adjoint:
        result = evaluate_galerkin_adjoint_residual(operator, field, source)
    else:
        result = evaluate_galerkin_residual(operator, field, source)
    return result


def _checked_solver_inputs(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    initial_field: Complex[Array, "n"] | None,
    max_iterations: scalar_int,
    relative_tolerance: scalar_float,
    absolute_tolerance: scalar_float,
) -> _SolverInputs:
    """Validate solver structure and return a residual stopping threshold."""
    if isinstance(max_iterations, bool):
        raise ValueError("max_iterations must not be boolean")
    state_size: int = operator.free_diagonal.shape[0]
    if source.ndim != 1 or source.shape[0] != state_size:
        raise ValueError("source length must match the operator state size")
    if initial_field is not None and initial_field.shape != source.shape:
        raise ValueError("initial_field and source must have the same shape")

    max_iterations_array: Int[Array, ""] = jnp.asarray(max_iterations)
    relative_array: Float[Array, ""] = jnp.asarray(relative_tolerance)
    absolute_array: Float[Array, ""] = jnp.asarray(absolute_tolerance)
    if max_iterations_array.shape != ():
        raise ValueError("max_iterations must be a scalar")
    if relative_array.shape != () or absolute_array.shape != ():
        raise ValueError("solver tolerances must be scalars")

    checked_source: Complex[Array, "n"] = eqx.error_if(
        source,
        jnp.any(~jnp.isfinite(source)) | has_subnormal_components(source),
        "source must be finite and contain no nonzero subnormal components",
    )
    checked_source = eqx.error_if(
        checked_source,
        max_iterations_array <= 0,
        "max_iterations must be positive",
    )
    checked_source = eqx.error_if(
        checked_source,
        (~jnp.isfinite(relative_array))
        | (~jnp.isfinite(absolute_array))
        | (relative_array < 0.0)
        | (absolute_array < 0.0),
        "solver tolerances must be finite and non-negative",
    )
    if initial_field is None:
        checked_initial: Complex[Array, "n"] = jnp.zeros_like(checked_source)
    else:
        checked_initial = eqx.error_if(
            initial_field,
            jnp.any(~jnp.isfinite(initial_field))
            | has_subnormal_components(initial_field),
            "initial_field must be finite and contain no nonzero subnormal "
            "components",
        )
    source_norm: Float[Array, ""] = _complex_norm(checked_source)
    raw_stopping_threshold: Float[Array, ""] = (
        absolute_array + relative_array * source_norm
    )
    stopping_threshold: Float[Array, ""] = eqx.error_if(
        raw_stopping_threshold,
        ~jnp.isfinite(raw_stopping_threshold),
        "derived solver stopping threshold must be finite",
    )
    result: _SolverInputs = _SolverInputs(
        source=checked_source,
        initial_field=checked_initial,
        max_iterations=max_iterations_array,
        stopping_threshold=stopping_threshold,
    )
    return result


def _solve_status(
    converged: Bool[Array, ""],
    breakdown: Bool[Array, ""],
    residual_mismatch: Bool[Array, ""],
) -> Int[Array, ""]:
    """Encode one dynamic Krylov termination status."""
    converged_code: int = int(GalerkinSolveStatus.CONVERGED)
    max_iterations_code: int = int(GalerkinSolveStatus.MAX_ITERATIONS)
    breakdown_code: int = int(GalerkinSolveStatus.BREAKDOWN)
    residual_mismatch_code: int = int(GalerkinSolveStatus.RESIDUAL_MISMATCH)
    status: Int[Array, ""] = jnp.where(
        converged,
        converged_code,
        jnp.where(
            breakdown,
            breakdown_code,
            jnp.where(
                residual_mismatch,
                residual_mismatch_code,
                max_iterations_code,
            ),
        ),
    )
    return status


def _finalize_cgls_result(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    state: _CGLSState,
    stopping_threshold: Float[Array, ""],
    *,
    adjoint: bool,
) -> GalerkinSolveResult:
    """Recompute final CGLS residuals and construct the result carrier."""
    if adjoint:
        residual, residual_norm = evaluate_galerkin_adjoint_residual(
            operator, state.field, source
        )
        normal_residual: Complex[Array, "n"] = apply_galerkin_operator(
            operator, residual
        )
    else:
        residual, residual_norm = evaluate_galerkin_residual(
            operator, state.field, source
        )
        normal_residual = apply_galerkin_adjoint(operator, residual)
    normal_residual_norm: Float[Array, ""] = _complex_norm(normal_residual)
    converged: Bool[Array, ""] = (residual_norm <= stopping_threshold) & (
        ~state.breakdown
    )
    residual_mismatch: Bool[Array, ""] = state.converged & (~converged)
    status: Int[Array, ""] = _solve_status(
        converged, state.breakdown, residual_mismatch
    )
    result: GalerkinSolveResult = create_galerkin_solve_result(
        field=state.field,
        residual=residual,
        residual_norm=residual_norm,
        normal_residual_norm=normal_residual_norm,
        recurrence_residual_norm=state.recurrence_residual_norm,
        iterations=state.iterations,
        operator_applications=state.operator_applications + 2,
        converged=converged,
        status=status,
        method=GalerkinSolveMethod.CGLS,
        certificate_reason=(
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND
        ),
    )
    return result


def _cgls_core(  # noqa: PLR0915
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    initial_field: Complex[Array, "n"] | None,
    max_iterations: scalar_int,
    relative_tolerance: scalar_float,
    absolute_tolerance: scalar_float,
    *,
    adjoint: bool,
) -> GalerkinSolveResult:
    """Run CGLS against either the forward or adjoint original system."""
    inputs: _SolverInputs = _checked_solver_inputs(
        operator,
        source,
        initial_field,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    initial_residual, initial_residual_norm = _solver_original_residual(
        operator,
        inputs.initial_field,
        inputs.source,
        adjoint=adjoint,
    )
    safe_initial_residual_norm: Float[Array, ""] = jnp.where(
        initial_residual_norm > 0.0,
        initial_residual_norm,
        1.0,
    )
    normalized_initial_residual: Complex[Array, "n"] = (
        initial_residual / safe_initial_residual_norm
    )
    if adjoint:
        raw_initial_adjoint_residual: Complex[Array, "n"] = (
            apply_galerkin_operator(operator, normalized_initial_residual)
        )
    else:
        raw_initial_adjoint_residual = apply_galerkin_adjoint(
            operator, normalized_initial_residual
        )
    initial_adjoint_scale: Float[Array, ""] = _complex_norm(
        raw_initial_adjoint_residual
    )
    safe_initial_adjoint_scale: Float[Array, ""] = jnp.where(
        initial_adjoint_scale > 0.0,
        initial_adjoint_scale,
        1.0,
    )
    initial_adjoint_residual: Complex[Array, "n"] = (
        raw_initial_adjoint_residual / safe_initial_adjoint_scale
    )
    initial_converged: Bool[Array, ""] = (
        initial_residual_norm <= inputs.stopping_threshold
    )
    initial_breakdown: Bool[Array, ""] = (~initial_converged) & (
        (~jnp.isfinite(initial_adjoint_scale)) | (initial_adjoint_scale <= 0.0)
    )
    initial_state: _CGLSState = _CGLSState(
        field=inputs.initial_field,
        residual=initial_residual,
        adjoint_residual=initial_adjoint_residual,
        direction=initial_adjoint_residual,
        adjoint_scale=initial_adjoint_scale,
        direction_scale_ratio=jnp.asarray(
            1.0, dtype=initial_adjoint_scale.dtype
        ),
        residual_norm=initial_residual_norm,
        recurrence_residual_norm=initial_residual_norm,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        operator_applications=jnp.asarray(2, dtype=jnp.int32),
        converged=initial_converged,
        breakdown=initial_breakdown,
    )

    def condition(state: _CGLSState) -> Bool[Array, ""]:
        """Continue while the requested solve remains active."""
        active: Bool[Array, ""] = (
            (state.iterations < inputs.max_iterations)
            & (~state.converged)
            & (~state.breakdown)
        )
        return active

    def step(state: _CGLSState) -> _CGLSState:
        """Perform one CGLS step and recompute the original residual."""
        if adjoint:
            image: Complex[Array, "n"] = apply_galerkin_adjoint(
                operator, state.direction
            )
        else:
            image = apply_galerkin_operator(operator, state.direction)
        image_norm: Float[Array, ""] = _complex_norm(image)
        valid_step: Bool[Array, ""] = (
            jnp.isfinite(image_norm)
            & jnp.all(jnp.isfinite(image))
            & (image_norm > 0.0)
            & jnp.isfinite(state.adjoint_scale)
            & (state.adjoint_scale > 0.0)
            & jnp.isfinite(state.direction_scale_ratio)
            & (state.direction_scale_ratio > 0.0)
            & jnp.isfinite(state.residual_norm)
            & (state.residual_norm > 0.0)
        )

        def fail() -> _CGLSState:
            """Freeze the last valid state and mark algorithmic breakdown."""
            failed: _CGLSState = state._replace(
                operator_applications=state.operator_applications + 1,
                breakdown=jnp.asarray(True),
            )
            return failed

        def advance() -> _CGLSState:
            """Advance using a valid least-squares denominator."""
            step_length: Float[Array, ""] = (
                (state.residual_norm / image_norm)
                * (state.adjoint_scale / image_norm)
                / state.direction_scale_ratio
            )
            candidate_field: Complex[Array, "n"] = (
                state.field + step_length * state.direction
            )
            recurrence_residual: Complex[Array, "n"] = (
                state.residual - step_length * image
            )
            recurrence_norm: Float[Array, ""] = _complex_norm(
                recurrence_residual
            )
            fresh_residual, fresh_norm = _solver_original_residual(
                operator,
                candidate_field,
                inputs.source,
                adjoint=adjoint,
            )
            safe_fresh_norm: Float[Array, ""] = jnp.where(
                fresh_norm > 0.0, fresh_norm, 1.0
            )
            normalized_fresh_residual: Complex[Array, "n"] = (
                fresh_residual / safe_fresh_norm
            )
            if adjoint:
                raw_fresh_adjoint_residual: Complex[Array, "n"] = (
                    apply_galerkin_operator(
                        operator, normalized_fresh_residual
                    )
                )
            else:
                raw_fresh_adjoint_residual = apply_galerkin_adjoint(
                    operator, normalized_fresh_residual
                )
            fresh_adjoint_scale: Float[Array, ""] = _complex_norm(
                raw_fresh_adjoint_residual
            )
            safe_fresh_adjoint_scale: Float[Array, ""] = jnp.where(
                fresh_adjoint_scale > 0.0,
                fresh_adjoint_scale,
                1.0,
            )
            fresh_adjoint_residual: Complex[Array, "n"] = (
                raw_fresh_adjoint_residual / safe_fresh_adjoint_scale
            )
            adjoint_scale_ratio: Float[Array, ""] = (
                fresh_norm / state.residual_norm
            ) * (fresh_adjoint_scale / state.adjoint_scale)
            raw_direction: Complex[Array, "n"] = (
                fresh_adjoint_residual
                + adjoint_scale_ratio
                * state.direction_scale_ratio
                * state.direction
            )
            direction_scale_ratio_new: Float[Array, ""] = _complex_norm(
                raw_direction
            )
            safe_direction_scale_ratio: Float[Array, ""] = jnp.where(
                direction_scale_ratio_new > 0.0,
                direction_scale_ratio_new,
                1.0,
            )
            direction_new: Complex[Array, "n"] = (
                raw_direction / safe_direction_scale_ratio
            )
            converged_new: Bool[Array, ""] = (
                fresh_norm <= inputs.stopping_threshold
            )
            finite_new: Bool[Array, ""] = (
                jnp.isfinite(fresh_norm)
                & jnp.isfinite(fresh_adjoint_scale)
                & jnp.isfinite(direction_scale_ratio_new)
                & jnp.isfinite(step_length)
                & jnp.all(jnp.isfinite(candidate_field))
                & jnp.all(jnp.isfinite(direction_new))
            )
            breakdown_new: Bool[Array, ""] = (~finite_new) | (
                (~converged_new)
                & (
                    (fresh_adjoint_scale <= 0.0)
                    | (direction_scale_ratio_new <= 0.0)
                )
            )
            advanced: _CGLSState = _CGLSState(
                field=candidate_field,
                residual=fresh_residual,
                adjoint_residual=fresh_adjoint_residual,
                direction=direction_new,
                adjoint_scale=fresh_adjoint_scale,
                direction_scale_ratio=direction_scale_ratio_new,
                residual_norm=fresh_norm,
                recurrence_residual_norm=recurrence_norm,
                iterations=state.iterations + 1,
                operator_applications=state.operator_applications + 3,
                converged=converged_new,
                breakdown=breakdown_new,
            )
            return advanced

        next_state: _CGLSState = jax.lax.cond(valid_step, advance, fail)
        return next_state

    final_state: _CGLSState = jax.lax.while_loop(
        condition, step, initial_state
    )
    result: GalerkinSolveResult = _finalize_cgls_result(
        operator,
        inputs.source,
        final_state,
        inputs.stopping_threshold,
        adjoint=adjoint,
    )
    return result


@jaxtyped(typechecker=beartype)
def cgls_solve(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    initial_field: Complex[Array, "n"] | None = None,
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> GalerkinSolveResult:
    """Solve a Galerkin system with CGLS and a fresh residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Frozen algebraic realization or canonical manifested SC-1 target.
    source : Complex[Array, "n"]
        Original finite-system right-hand side.
    initial_field : Complex[Array, "n"] | None
        Initial state.  ``None`` selects zero.  Default is ``None``.
    max_iterations : scalar_int
        Positive maximum number of CGLS iterations.  Default is 100.
    relative_tolerance : scalar_float
        Non-negative relative tolerance on the fresh residual norm.  Default
        is ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute tolerance on the fresh residual norm.  Default
        is zero.

    Returns
    -------
    result : GalerkinSolveResult
        Field, fresh residual diagnostics, work counts, and typed termination
        status.  The result is not an outward state certificate.

    Raises
    ------
    ValueError
        If a static input rank, shape, or scalar structure is invalid.
    equinox.EquinoxRuntimeError
        If a source, initial field, iteration limit, or tolerance is invalid
        during traced execution.

    Notes
    -----
    CGLS applies ``H`` and the actual ``H*`` without forming ``H*H``.  The
    stopping decision uses ``source - H field`` recomputed outside the
    recurrence. Residual and search directions are normalized before adjoint
    actions to avoid avoidable scaling breakdown. An unrepresentable fresh
    residual fails closed; the method does not promise convergence for every
    factory-admitted condition number.
    """
    result: GalerkinSolveResult = _cgls_core(
        operator,
        source,
        initial_field,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
        adjoint=False,
    )
    return result


def _finalize_lsqr_result(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    state: _LSQRState,
    stopping_threshold: Float[Array, ""],
) -> GalerkinSolveResult:
    """Recompute final LSQR residuals and construct the result carrier."""
    residual, residual_norm = evaluate_galerkin_residual(
        operator, state.field, source
    )
    normal_residual: Complex[Array, "n"] = apply_galerkin_adjoint(
        operator, residual
    )
    normal_residual_norm: Float[Array, ""] = _complex_norm(normal_residual)
    converged: Bool[Array, ""] = (residual_norm <= stopping_threshold) & (
        ~state.breakdown
    )
    residual_mismatch: Bool[Array, ""] = state.converged & (~converged)
    status: Int[Array, ""] = _solve_status(
        converged, state.breakdown, residual_mismatch
    )
    result: GalerkinSolveResult = create_galerkin_solve_result(
        field=state.field,
        residual=residual,
        residual_norm=residual_norm,
        normal_residual_norm=normal_residual_norm,
        recurrence_residual_norm=state.recurrence_residual_norm,
        iterations=state.iterations,
        operator_applications=state.operator_applications + 2,
        converged=converged,
        status=status,
        method=GalerkinSolveMethod.LSQR,
        certificate_reason=(
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND
        ),
    )
    return result


def _lsqr_core(  # noqa: PLR0915
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    initial_field: Complex[Array, "n"] | None,
    max_iterations: scalar_int,
    relative_tolerance: scalar_float,
    absolute_tolerance: scalar_float,
) -> GalerkinSolveResult:
    """Run LSQR with fresh original-system residual stopping."""
    inputs: _SolverInputs = _checked_solver_inputs(
        operator,
        source,
        initial_field,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    initial_residual, initial_residual_norm = _solver_original_residual(
        operator,
        inputs.initial_field,
        inputs.source,
        adjoint=False,
    )
    beta: Float[Array, ""] = initial_residual_norm
    safe_beta: Float[Array, ""] = jnp.where(beta > 0.0, beta, 1.0)
    left_vector: Complex[Array, "n"] = initial_residual / safe_beta
    adjoint_left: Complex[Array, "n"] = apply_galerkin_adjoint(
        operator, left_vector
    )
    alpha: Float[Array, ""] = _complex_norm(adjoint_left)
    safe_alpha: Float[Array, ""] = jnp.where(alpha > 0.0, alpha, 1.0)
    right_vector: Complex[Array, "n"] = adjoint_left / safe_alpha
    initial_converged: Bool[Array, ""] = beta <= inputs.stopping_threshold
    initial_state: _LSQRState = _LSQRState(
        field=inputs.initial_field,
        left_vector=left_vector,
        right_vector=right_vector,
        direction=right_vector,
        alpha=alpha,
        beta=beta,
        rho_bar=alpha,
        phi_bar=beta,
        recurrence_residual_norm=beta,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        operator_applications=jnp.asarray(2, dtype=jnp.int32),
        converged=initial_converged,
        breakdown=(~initial_converged)
        & ((~jnp.isfinite(alpha)) | (alpha <= 0.0)),
    )

    def condition(state: _LSQRState) -> Bool[Array, ""]:
        """Continue while the requested solve remains active."""
        active: Bool[Array, ""] = (
            (state.iterations < inputs.max_iterations)
            & (~state.converged)
            & (~state.breakdown)
        )
        return active

    def step(state: _LSQRState) -> _LSQRState:
        """Perform one Golub--Kahan step and recompute the residual."""
        left_raw: Complex[Array, "n"] = (
            apply_galerkin_operator(operator, state.right_vector)
            - state.alpha * state.left_vector
        )
        beta_new: Float[Array, ""] = _complex_norm(left_raw)
        safe_beta_new: Float[Array, ""] = jnp.where(
            beta_new > 0.0, beta_new, 1.0
        )
        left_new: Complex[Array, "n"] = left_raw / safe_beta_new
        right_raw: Complex[Array, "n"] = (
            apply_galerkin_adjoint(operator, left_new)
            - beta_new * state.right_vector
        )
        alpha_new: Float[Array, ""] = _complex_norm(right_raw)
        safe_alpha_new: Float[Array, ""] = jnp.where(
            alpha_new > 0.0, alpha_new, 1.0
        )
        right_new: Complex[Array, "n"] = right_raw / safe_alpha_new
        rho: Float[Array, ""] = jnp.hypot(state.rho_bar, beta_new)
        valid_step: Bool[Array, ""] = (
            jnp.isfinite(rho)
            & (rho > 0.0)
            & jnp.isfinite(alpha_new)
            & jnp.isfinite(beta_new)
            & jnp.all(jnp.isfinite(left_raw))
            & jnp.all(jnp.isfinite(right_raw))
        )

        def fail() -> _LSQRState:
            """Freeze the last valid state and mark algorithmic breakdown."""
            failed: _LSQRState = state._replace(
                operator_applications=state.operator_applications + 2,
                breakdown=jnp.asarray(True),
            )
            return failed

        def advance() -> _LSQRState:
            """Apply the next stable orthogonal transformation."""
            cosine: Float[Array, ""] = state.rho_bar / rho
            sine: Float[Array, ""] = beta_new / rho
            theta: Float[Array, ""] = sine * alpha_new
            rho_bar_new: Float[Array, ""] = -cosine * alpha_new
            phi: Float[Array, ""] = cosine * state.phi_bar
            phi_bar_new: Float[Array, ""] = sine * state.phi_bar
            field_new: Complex[Array, "n"] = (
                state.field + (phi / rho) * state.direction
            )
            direction_new: Complex[Array, "n"] = (
                right_new - (theta / rho) * state.direction
            )
            fresh_residual, fresh_norm = _solver_original_residual(
                operator,
                field_new,
                inputs.source,
                adjoint=False,
            )
            safe_fresh_norm: Float[Array, ""] = jnp.where(
                fresh_norm > 0.0, fresh_norm, 1.0
            )
            normalized_fresh_residual: Complex[Array, "n"] = (
                fresh_residual / safe_fresh_norm
            )
            fresh_normalized_adjoint: Complex[Array, "n"] = (
                apply_galerkin_adjoint(
                    operator,
                    normalized_fresh_residual,
                )
            )
            fresh_normalized_adjoint_scale: Float[Array, ""] = _complex_norm(
                fresh_normalized_adjoint
            )
            converged_new: Bool[Array, ""] = (
                fresh_norm <= inputs.stopping_threshold
            )
            finite_new: Bool[Array, ""] = (
                jnp.isfinite(fresh_norm)
                & jnp.isfinite(fresh_normalized_adjoint_scale)
                & jnp.all(jnp.isfinite(field_new))
                & jnp.all(jnp.isfinite(direction_new))
            )
            breakdown_new: Bool[Array, ""] = (~finite_new) | (
                (~converged_new)
                & (
                    (fresh_normalized_adjoint_scale <= 0.0)
                    | ((alpha_new <= 0.0) & (beta_new <= 0.0))
                )
            )
            advanced: _LSQRState = _LSQRState(
                field=field_new,
                left_vector=left_new,
                right_vector=right_new,
                direction=direction_new,
                alpha=alpha_new,
                beta=beta_new,
                rho_bar=rho_bar_new,
                phi_bar=phi_bar_new,
                recurrence_residual_norm=jnp.abs(phi_bar_new),
                iterations=state.iterations + 1,
                operator_applications=state.operator_applications + 4,
                converged=converged_new,
                breakdown=breakdown_new,
            )
            return advanced

        next_state: _LSQRState = jax.lax.cond(valid_step, advance, fail)
        return next_state

    final_state: _LSQRState = jax.lax.while_loop(
        condition, step, initial_state
    )
    result: GalerkinSolveResult = _finalize_lsqr_result(
        operator,
        inputs.source,
        final_state,
        inputs.stopping_threshold,
    )
    return result


@jaxtyped(typechecker=beartype)
def lsqr_solve(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    initial_field: Complex[Array, "n"] | None = None,
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> GalerkinSolveResult:
    """Solve a Galerkin system with LSQR and a fresh residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Frozen algebraic realization or canonical manifested SC-1 target.
    source : Complex[Array, "n"]
        Original finite-system right-hand side.
    initial_field : Complex[Array, "n"] | None
        Initial state.  ``None`` selects zero.  Default is ``None``.
    max_iterations : scalar_int
        Positive maximum number of LSQR iterations.  Default is 100.
    relative_tolerance : scalar_float
        Non-negative relative tolerance on the fresh residual norm.  Default
        is ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute tolerance on the fresh residual norm.  Default
        is zero.

    Returns
    -------
    result : GalerkinSolveResult
        Field, fresh residual diagnostics, work counts, and typed termination
        status.  The result is not an outward state certificate.

    Raises
    ------
    ValueError
        If a static input rank, shape, or scalar structure is invalid.
    equinox.EquinoxRuntimeError
        If a source, initial field, iteration limit, or tolerance is invalid
        during traced execution.

    Notes
    -----
    LSQR uses the actual ``H/H*`` pair.  Its recurrence residual is reported
    separately from the freshly evaluated original-system residual used for
    acceptance. The fresh residual is normalized before the adjoint control
    action so a uniform normal-range scaling does not change termination.
    An unrepresentable fresh residual fails closed.
    """
    result: GalerkinSolveResult = _lsqr_core(
        operator,
        source,
        initial_field,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )
    return result


def _implicit_galerkin_solve_impl(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    max_iterations: scalar_int,
    relative_tolerance: scalar_float,
    absolute_tolerance: scalar_float,
) -> Complex[Array, "n"]:
    """Evaluate the primal fixed-support root without recursive VJP use."""
    result: GalerkinSolveResult = _cgls_core(
        operator,
        source,
        None,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
        adjoint=False,
    )
    field: Complex[Array, "n"] = eqx.error_if(
        result.field,
        ~result.converged,
        "implicit Galerkin forward solve did not converge",
    )
    return field


@jax.custom_vjp
@jaxtyped(typechecker=beartype)
def implicit_galerkin_solve(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> Complex[Array, "n"]:
    """Solve a Galerkin root with an implicit custom VJP.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : GalerkinOperator | GalerkinTargetManifest
        Fixed-support differentiable algebraic or manifested realization.
    source : Complex[Array, "n"]
        Differentiable finite-system source coefficients.
    max_iterations : scalar_int
        Positive CGLS iteration limit for both primal and adjoint roots.
        Default is 100.
    relative_tolerance : scalar_float
        Non-negative relative algebraic residual tolerance.  Default is
        ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute algebraic residual tolerance.  Default is zero.

    Returns
    -------
    field : Complex[Array, "n"]
        Converged fixed-support root.

    Raises
    ------
    ValueError
        If a static input rank, shape, or scalar structure is invalid.
    equinox.EquinoxRuntimeError
        If an input is invalid or either algebraic solve fails to converge
        during traced execution.

    Notes
    -----
    The backward pass solves ``H* lambda = grad_u L`` and recomputes its
    original-system residual through CGLS.  It saves only the operator, final
    field, source-independent solver settings, and no Krylov trajectory.
    Gradients remain algebraic diagnostics until RM-I2 supplies an inexact
    gradient theorem and RM-S6-I supplies outward state evidence.
    """
    field: Complex[Array, "n"] = _implicit_galerkin_solve_impl(
        operator,
        source,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )
    return field


def _implicit_galerkin_solve_fwd(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    max_iterations: scalar_int,
    relative_tolerance: scalar_float,
    absolute_tolerance: scalar_float,
) -> tuple[
    Complex[Array, "n"],
    tuple[
        _GalerkinSystem,
        Complex[Array, "n"],
        Int[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ],
]:
    """Save the converged root and fixed data, but no Krylov trajectory."""
    field: Complex[Array, "n"] = _implicit_galerkin_solve_impl(
        operator,
        source,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )
    residual: tuple[
        _GalerkinSystem,
        Complex[Array, "n"],
        Int[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ] = (
        operator,
        field,
        jnp.asarray(max_iterations),
        jnp.asarray(relative_tolerance),
        jnp.asarray(absolute_tolerance),
    )
    result: tuple[
        Complex[Array, "n"],
        tuple[
            _GalerkinSystem,
            Complex[Array, "n"],
            Int[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
    ] = field, residual
    return result


def _implicit_galerkin_solve_bwd(
    residual: tuple[
        _GalerkinSystem,
        Complex[Array, "n"],
        Int[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ],
    output_cotangent: Complex[Array, "n"],
) -> tuple[_GalerkinSystem, Complex[Array, "n"], None, None, None]:
    """Solve the SC-1 adjoint and pull it back through operator parameters."""
    operator, field, max_iterations, relative_tolerance, absolute_tolerance = (
        residual
    )
    adjoint_source: Complex[Array, "n"] = jnp.conj(output_cotangent)
    adjoint_result: GalerkinSolveResult = _cgls_core(
        operator,
        adjoint_source,
        None,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
        adjoint=True,
    )
    adjoint_field: Complex[Array, "n"] = eqx.error_if(
        adjoint_result.field,
        ~adjoint_result.converged,
        "implicit Galerkin adjoint solve did not converge",
    )

    def fixed_state_action(
        candidate: _GalerkinSystem,
    ) -> Complex[Array, "n"]:
        """Apply one candidate operator to the saved converged state."""
        action: Complex[Array, "n"] = apply_galerkin_operator(candidate, field)
        return action

    _, operator_pullback = jax.vjp(fixed_state_action, operator)
    operator_cotangent: _GalerkinSystem = operator_pullback(
        -jnp.conj(adjoint_field)
    )[0]
    source_cotangent: Complex[Array, "n"] = jnp.conj(adjoint_field)
    cotangents: tuple[
        _GalerkinSystem,
        Complex[Array, "n"],
        None,
        None,
        None,
    ] = (
        operator_cotangent,
        source_cotangent,
        None,
        None,
        None,
    )
    return cotangents


implicit_galerkin_solve.defvjp(
    _implicit_galerkin_solve_fwd,
    _implicit_galerkin_solve_bwd,
)


__all__: list[str] = [
    "apply_galerkin_adjoint",
    "apply_galerkin_operator",
    "cgls_solve",
    "evaluate_galerkin_adjoint_residual",
    "evaluate_galerkin_residual",
    "implicit_galerkin_solve",
    "lsqr_solve",
    "shifted_free_diagonal",
]
