r"""Apply and solve a fixed-support scalar Galerkin system.

Extended Summary
----------------
This module applies a frozen algebraic Galerkin realization, a canonical
manifested SC-1 target, or a prepared ``LOCAL_CELL_LVT1`` target without
assembling the dense operator. The forward and adjoint actions are distinct,
and CGLS and LSQR stop on a directly recomputed original-system residual.

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
:func:`cgls_adjoint_solve`
    Solve an actual adjoint Galerkin system with CGLS and a fresh residual.
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

Callers must cross the local-cell trust boundary exactly once through
``prepare_local_cell_galerkin_target`` and supply its result to this engine.
Raw same-type storage is not authenticated or distinguishable here; actions
and solvers deliberately do not replay proof storage inside JAX transforms.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
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
    Int32,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import (
    has_lost_subtraction,
    has_subnormal_components,
)
from ptyrodactyl.types import (
    GalerkinCertificateReason,
    GalerkinLocalCellTargetManifest,
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

from .local_cell_system import (
    apply_local_cell_galerkin_target,
    apply_local_cell_galerkin_target_adjoint,
)
from .system import (
    apply_galerkin_target,
    apply_galerkin_target_adjoint,
    evaluate_physical_galerkin_adjoint_residual,
    evaluate_physical_galerkin_residual,
)

_FREQUENCY_RANK: int = 2
_SPACE_DIMENSIONS: int = 3

type _GalerkinSystem = (
    GalerkinOperator | GalerkinTargetManifest | GalerkinLocalCellTargetManifest
)


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
    iterations: Int32[Array, ""]
    operator_applications: Int32[Array, ""]
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
    iterations: Int32[Array, ""]
    operator_applications: Int32[Array, ""]
    converged: Bool[Array, ""]
    breakdown: Bool[Array, ""]


class _SolverInputs(NamedTuple):
    """Store checked Krylov inputs and the fresh-residual threshold."""

    source: Complex[Array, "n"]
    initial_field: Complex[Array, "n"]
    max_iterations: Int[Array, ""]
    stopping_threshold: Float[Array, ""]


def _complex_norm(vector: Complex[Array, "n"]) -> Float[Array, ""]:
    """PRIVATE: Compute the norm induced by the SC-1 coefficient metric.

    Parameters
    ----------
    vector : Complex[Array, "n"]
        Complex coefficients in one fixed state ordering.

    Returns
    -------
    norm : Float[Array, ""]
        Scale-safe Euclidean coefficient norm.

    Notes
    -----
    The SC-1 coefficient metric is the standard complex Euclidean metric.
    Scaling by the largest magnitude limits intermediate overflow and
    underflow.
    """
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
    """PRIVATE: Apply one frozen COO map by gather and scatter-add.

    Parameters
    ----------
    rows : Int[Array, "nnz"]
        COO output row for each stored entry.
    columns : Int[Array, "nnz"]
        COO input column for each stored entry.
    values : Complex[Array, "nnz"]
        COO values in the shared entry order.
    vector : Complex[Array, "input_size"]
        Input coefficient vector.
    output_size : int
        Static output-vector length.

    Returns
    -------
    result : Complex[Array, "output_size"]
        Sparse map action in output-row order.

    Notes
    -----
    Repeated rows accumulate through scatter-add. Stored zeros and duplicate
    entries therefore retain ordinary COO semantics.
    """
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
    """PRIVATE: Apply the conjugate transpose of one frozen COO map.

    Parameters
    ----------
    rows : Int[Array, "nnz"]
        COO output row for each forward-map entry.
    columns : Int[Array, "nnz"]
        COO input column for each forward-map entry.
    values : Complex[Array, "nnz"]
        COO values in the shared entry order.
    vector : Complex[Array, "output_size"]
        Vector in the forward map's output space.
    input_size : int
        Static forward-input vector length.

    Returns
    -------
    result : Complex[Array, "input_size"]
        Conjugate-transpose action in forward-input order.

    Notes
    -----
    The operation conjugates every stored value and exchanges COO row and
    column roles.
    """
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
    """PRIVATE: Apply the frozen sparse interaction realization.

    Parameters
    ----------
    operator : GalerkinOperator
        Frozen algebraic operator with an interaction COO map.
    field : Complex[Array, "n"]
        Retained-state coefficient vector.

    Returns
    -------
    interaction : Complex[Array, "n"]
        Forward sparse interaction action.

    Notes
    -----
    This helper applies only the interaction term, without its target-equation
    sign.
    """
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
    """PRIVATE: Apply the sparse interaction realization's actual adjoint.

    Parameters
    ----------
    operator : GalerkinOperator
        Frozen algebraic operator with an interaction COO map.
    field : Complex[Array, "n"]
        Retained-state coefficient vector.

    Returns
    -------
    interaction : Complex[Array, "n"]
        Conjugate-transpose sparse interaction action.

    Notes
    -----
    No Hermiticity assumption replaces the stored conjugate-transpose action.
    """
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
    """PRIVATE: Apply the positive-semidefinite absorber Gramian ``G*G``.

    Parameters
    ----------
    operator : GalerkinOperator
        Frozen operator containing one sparse absorber factor ``G``.
    field : Complex[Array, "n"]
        Retained-state coefficient vector.

    Returns
    -------
    absorber : Complex[Array, "n"]
        Positive-semidefinite Gramian action in state order.

    Notes
    -----
    The helper applies ``G`` followed by its actual conjugate transpose. It
    does not materialize the dense Gramian.
    """
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
    """PRIVATE: Reject non-finite and nonzero-subnormal action vectors.

    Parameters
    ----------
    values : Complex[Array, "n"]
        Candidate vector produced or consumed by an operator action.
    name : str
        Vector name used in the traced rejection message.

    Returns
    -------
    checked : Complex[Array, "n"]
        Vector with traced finite and normal-range checks attached.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any component is non-finite or nonzero subnormal.
    """
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
    """PRIVATE: Subtract one action and reject a flushed component.

    Parameters
    ----------
    source : Complex[Array, "n"]
        Original-system source coefficients.
    action : Complex[Array, "n"]
        Independently evaluated operator action.
    name : str
        Residual name used in traced rejection messages.

    Returns
    -------
    residual : Complex[Array, "n"]
        Checked difference ``source - action``.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the residual is invalid or subtraction loses a nonzero component.

    Notes
    -----
    An optimization barrier preserves an independently rounded subtraction
    boundary for the lost-component check.
    """
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
    operator : _GalerkinSystem
        Frozen algebraic realization, canonical manifested SC-1 target, or
        prepared LOCAL_CELL_LVT1 target.
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
    if isinstance(operator, GalerkinLocalCellTargetManifest):
        raw_applied_field: Complex128[Array, "n"] = (
            apply_local_cell_galerkin_target(operator, checked_field)
        )
    elif isinstance(operator, GalerkinTargetManifest):
        raw_applied_field = apply_galerkin_target(
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
    operator : _GalerkinSystem
        Frozen algebraic realization, canonical manifested SC-1 target, or
        prepared LOCAL_CELL_LVT1 target.
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
    if isinstance(operator, GalerkinLocalCellTargetManifest):
        raw_applied_field: Complex128[Array, "n"] = (
            apply_local_cell_galerkin_target_adjoint(
                operator,
                checked_field,
            )
        )
    elif isinstance(operator, GalerkinTargetManifest):
        raw_applied_field = apply_galerkin_target_adjoint(
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
    """PRIVATE: Re-evaluate the forward action outside Krylov recurrence.

    Parameters
    ----------
    operator : GalerkinOperator
        Frozen sparse algebraic operator.
    field : Complex[Array, "n"]
        Retained-state coefficient vector.

    Returns
    -------
    action : Complex[Array, "n"]
        Fresh forward action ``D u - R u - i epsilon_CAP G*G u``.

    Notes
    -----
    This implementation expands the sparse terms directly instead of calling
    the production forward-action wrapper.
    """
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
    """PRIVATE: Re-evaluate the actual adjoint outside Krylov recurrence.

    Parameters
    ----------
    operator : GalerkinOperator
        Frozen sparse algebraic operator.
    field : Complex[Array, "n"]
        Retained-state coefficient vector.

    Returns
    -------
    action : Complex[Array, "n"]
        Fresh adjoint action ``D u - R* u + i epsilon_CAP G*G u``.

    Notes
    -----
    The interaction uses its stored conjugate transpose. The absorber sign is
    reversed from the forward action.
    """
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
) -> Tuple[Complex[Array, "n"], Float[Array, ""]]:
    """Evaluate a fresh forward-system algebraic residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : _GalerkinSystem
        Frozen algebraic realization, canonical manifested SC-1 target, or
        prepared LOCAL_CELL_LVT1 target.
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
    This function does not reuse a Krylov recurrence. The local branch uses
    its explicit frozen L3/L4 action, the legacy manifested branch uses direct
    coefficient contraction instead of its production FFT action, and the raw
    algebraic branch uses independent COO contraction. Neither floating-point
    diagnostic supplies an outward rounding enclosure; per-result stability
    invocation is a separate host-side check.
    """
    if field.shape != source.shape:
        raise ValueError("field and source must have the same shape")
    checked_field: Complex[Array, "n"] = _checked_action_vector(field, "field")
    checked_source: Complex[Array, "n"] = _checked_action_vector(
        source, "source"
    )
    if isinstance(operator, GalerkinLocalCellTargetManifest):
        local_action: Complex128[Array, "n"] = (
            apply_local_cell_galerkin_target(operator, checked_field)
        )
        residual: Complex128[Array, "n"] = _checked_residual_difference(
            checked_source, local_action, "residual"
        )
        residual_norm: Float64[Array, ""] = _complex_norm(residual)
    elif isinstance(operator, GalerkinTargetManifest):
        physical_residual: GalerkinPhysicalResidual = (
            evaluate_physical_galerkin_residual(
                operator, checked_field, checked_source
            )
        )
        residual = physical_residual.residual
        residual_norm = physical_residual.residual_norm
    else:
        action: Complex[Array, "n"] = _independent_forward_action(
            operator,
            checked_field,
        )
        residual = _checked_residual_difference(
            checked_source, action, "residual"
        )
        residual_norm = _complex_norm(residual)
    result: Tuple[Complex[Array, "n"], Float[Array, ""]] = (
        residual,
        residual_norm,
    )
    return result


@jaxtyped(typechecker=beartype)
def evaluate_galerkin_adjoint_residual(
    operator: _GalerkinSystem,
    field: Complex[Array, "n"],
    source: Complex[Array, "n"],
) -> Tuple[Complex[Array, "n"], Float[Array, ""]]:
    """Evaluate a fresh adjoint-system algebraic residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : _GalerkinSystem
        Frozen algebraic realization, canonical manifested SC-1 target, or
        prepared LOCAL_CELL_LVT1 target.
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
    This algebraic diagnostic is independent of the CGLS/LSQR recurrence. The
    local branch uses its explicit frozen L3/L4 formal adjoint, the legacy
    manifested branch uses its physical adjoint residual, and the raw branch
    uses independent COO contraction. It is not an outward adjoint-residual
    certificate.
    """
    if field.shape != source.shape:
        raise ValueError("field and source must have the same shape")
    checked_field: Complex[Array, "n"] = _checked_action_vector(field, "field")
    checked_source: Complex[Array, "n"] = _checked_action_vector(
        source, "source"
    )
    if isinstance(operator, GalerkinLocalCellTargetManifest):
        local_action: Complex128[Array, "n"] = (
            apply_local_cell_galerkin_target_adjoint(
                operator,
                checked_field,
            )
        )
        residual: Complex128[Array, "n"] = _checked_residual_difference(
            checked_source, local_action, "adjoint residual"
        )
        residual_norm: Float64[Array, ""] = _complex_norm(residual)
    elif isinstance(operator, GalerkinTargetManifest):
        physical_residual: GalerkinPhysicalResidual = (
            evaluate_physical_galerkin_adjoint_residual(
                operator,
                checked_field,
                checked_source,
            )
        )
        residual = physical_residual.residual
        residual_norm = physical_residual.residual_norm
    else:
        action: Complex[Array, "n"] = _independent_adjoint_action(
            operator,
            checked_field,
        )
        residual = _checked_residual_difference(
            checked_source, action, "adjoint residual"
        )
        residual_norm = _complex_norm(residual)
    result: Tuple[Complex[Array, "n"], Float[Array, ""]] = (
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
) -> Tuple[Complex[Array, "n"], Float[Array, ""]]:
    """PRIVATE: Recompute a scalable residual outside solver recurrence.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed-support algebraic or manifested target.
    field : Complex[Array, "n"]
        Candidate retained-state solution.
    source : Complex[Array, "n"]
        Original-system source coefficients.
    adjoint : bool
        If true, evaluate the actual adjoint original system.

    Returns
    -------
    residual : Complex[Array, "n"]
        Fresh original-system residual.
    residual_norm : Float[Array, ""]
        Scale-safe Euclidean norm of ``residual``.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If an action or residual is non-finite, subnormal, or numerically lost.

    Notes
    -----
    Manifested targets use the production action with an independently rounded
    subtraction. Sparse algebraic targets use the independent residual APIs.
    """
    if isinstance(operator, GalerkinLocalCellTargetManifest):
        if adjoint:
            target_action: Complex128[Array, "n"] = (
                apply_local_cell_galerkin_target_adjoint(operator, field)
            )
        else:
            target_action = apply_local_cell_galerkin_target(operator, field)
        target_residual: Complex128[Array, "n"] = _checked_residual_difference(
            source, target_action, "solver residual"
        )
        target_residual_norm: Float64[Array, ""] = _complex_norm(
            target_residual
        )
        target_result: Tuple[Complex128[Array, "n"], Float64[Array, ""]] = (
            target_residual,
            target_residual_norm,
        )
        result: Tuple[Complex[Array, "n"], Float[Array, ""]] = target_result
    elif isinstance(operator, GalerkinTargetManifest):
        if adjoint:
            target_action = apply_galerkin_adjoint(
                operator,
                field,
            )
        else:
            target_action = apply_galerkin_operator(operator, field)
        target_residual = _checked_residual_difference(
            source, target_action, "solver residual"
        )
        target_residual_norm = _complex_norm(target_residual)
        target_result = (
            target_residual,
            target_residual_norm,
        )
        result = target_result
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
    """PRIVATE: Validate solver inputs and derive a residual threshold.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed-support algebraic or manifested target.
    source : Complex[Array, "n"]
        Original-system source coefficients.
    initial_field : Complex[Array, "n"] | None
        Optional initial state. ``None`` selects the zero vector.
    max_iterations : scalar_int
        Positive Krylov iteration limit.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.

    Returns
    -------
    result : _SolverInputs
        Checked source, initial state, iteration limit, and fresh-residual
        stopping threshold.

    Raises
    ------
    ValueError
        If a rank, shape, Boolean iteration limit, or scalar structure is
        invalid.
    equinox.EquinoxRuntimeError
        If a value is invalid during traced execution or the threshold is not
        finite.

    Notes
    -----
    The threshold is ``absolute_tolerance + relative_tolerance * ||source||``.
    Solver acceptance later uses a fresh original-system residual.
    """
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
) -> Int64[Array, ""]:
    """PRIVATE: Encode one dynamic Krylov termination status.

    Parameters
    ----------
    converged : Bool[Array, ""]
        Whether the fresh physical residual meets its threshold.
    breakdown : Bool[Array, ""]
        Whether the Krylov recurrence encountered a breakdown.
    residual_mismatch : Bool[Array, ""]
        Whether recurrence convergence disagrees with the fresh residual.

    Returns
    -------
    status : Int64[Array, ""]
        Integer code from :class:`GalerkinSolveStatus`.

    Notes
    -----
    Convergence has highest precedence, followed by breakdown and residual
    mismatch. The remaining outcome is maximum iterations.
    """
    converged_code: int = int(GalerkinSolveStatus.CONVERGED)
    max_iterations_code: int = int(GalerkinSolveStatus.MAX_ITERATIONS)
    breakdown_code: int = int(GalerkinSolveStatus.BREAKDOWN)
    residual_mismatch_code: int = int(GalerkinSolveStatus.RESIDUAL_MISMATCH)
    status: Int64[Array, ""] = jnp.where(
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
    """PRIVATE: Recompute final CGLS residuals and construct the result.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed target used by the completed recurrence.
    source : Complex[Array, "n"]
        Original-system source coefficients.
    state : _CGLSState
        Final CGLS recurrence state.
    stopping_threshold : Float[Array, ""]
        Fresh original-residual acceptance threshold.
    adjoint : bool
        If true, finalize a solve of the actual adjoint system.

    Returns
    -------
    result : GalerkinSolveResult
        Field, fresh residual diagnostics, counts, and termination status.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If fresh forward, adjoint, or residual evaluation is invalid.

    Notes
    -----
    Fresh residual acceptance overrides recurrence-only convergence. The
    result remains algebraic and carries no outward residual certificate.
    """
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
    status: Int64[Array, ""] = _solve_status(
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
    """PRIVATE: Run CGLS against a forward or adjoint original system.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed-support target that supplies the ``H/H*`` action pair.
    source : Complex[Array, "n"]
        Original-system source coefficients.
    initial_field : Complex[Array, "n"] | None
        Optional initial state. ``None`` selects the zero vector.
    max_iterations : scalar_int
        Positive CGLS iteration limit.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.
    adjoint : bool
        If true, solve the actual adjoint system.

    Returns
    -------
    result : GalerkinSolveResult
        Final field, fresh residual diagnostics, counts, and typed status.

    Raises
    ------
    ValueError
        If a static rank, shape, or scalar structure is invalid.
    equinox.EquinoxRuntimeError
        If a traced input, operator action, or residual is invalid.

    Notes
    -----
    CGLS applies ``H`` and the actual ``H*`` without forming ``H*H``. It
    normalizes residual directions and accepts only a fresh residual check.
    """
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
def cgls_adjoint_solve(
    operator: _GalerkinSystem,
    source: Complex[Array, "n"],
    initial_field: Complex[Array, "n"] | None = None,
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> GalerkinSolveResult:
    """Solve an actual adjoint Galerkin system with CGLS and a fresh residual.

    :see: :class:`~.test_engine.TestMatrixFreeGalerkinEngine`

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed target supplying the ``H/H*`` action pair.
    source : Complex[Array, "n"]
        Right-hand side of ``H* field = source``.
    initial_field : Complex[Array, "n"] | None
        Initial state. ``None`` selects zero. Default is ``None``.
    max_iterations : scalar_int
        Positive CGLS iteration limit. Default is 100.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance. Default is ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance. Default is zero.

    Returns
    -------
    result : GalerkinSolveResult
        Field, freshly recomputed ``source - H* field`` residual, work counts,
        and typed status. The result is not an outward state certificate.

    Raises
    ------
    ValueError
        If a static input rank, shape, or scalar structure is invalid.
    equinox.EquinoxRuntimeError
        If an input, action, or residual is invalid during traced execution.

    Notes
    -----
    This is the retained-record counterpart of the adjoint solve used by the
    implicit custom VJP. Convergence is accepted only after independently
    recomputing the residual of the actual adjoint original system.
    """
    result: GalerkinSolveResult = _cgls_core(
        operator,
        source,
        initial_field,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
        adjoint=True,
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
    operator : _GalerkinSystem
        Frozen algebraic realization, canonical manifested SC-1 target, or
        prepared LOCAL_CELL_LVT1 target.
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
    """PRIVATE: Recompute final LSQR residuals and construct the result.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed target used by the completed recurrence.
    source : Complex[Array, "n"]
        Original-system source coefficients.
    state : _LSQRState
        Final LSQR recurrence state.
    stopping_threshold : Float[Array, ""]
        Fresh original-residual acceptance threshold.

    Returns
    -------
    result : GalerkinSolveResult
        Field, fresh residual diagnostics, counts, and termination status.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If fresh forward, adjoint, or residual evaluation is invalid.

    Notes
    -----
    Fresh residual acceptance overrides recurrence-only convergence. The
    result remains algebraic and carries no outward residual certificate.
    """
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
    status: Int64[Array, ""] = _solve_status(
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
    """PRIVATE: Run LSQR with fresh original-system residual stopping.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed-support target that supplies the ``H/H*`` action pair.
    source : Complex[Array, "n"]
        Original-system source coefficients.
    initial_field : Complex[Array, "n"] | None
        Optional initial state. ``None`` selects the zero vector.
    max_iterations : scalar_int
        Positive LSQR iteration limit.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.

    Returns
    -------
    result : GalerkinSolveResult
        Final field, fresh residual diagnostics, counts, and typed status.

    Raises
    ------
    ValueError
        If a static rank, shape, or scalar structure is invalid.
    equinox.EquinoxRuntimeError
        If a traced input, operator action, or residual is invalid.

    Notes
    -----
    LSQR uses the actual ``H/H*`` pair. It reports its recurrence residual
    separately and accepts only a fresh original-system residual check.
    """
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
    operator : _GalerkinSystem
        Frozen algebraic realization, canonical manifested SC-1 target, or
        prepared LOCAL_CELL_LVT1 target.
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
    """PRIVATE: Evaluate the primal root without recursive VJP use.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed-support differentiable target.
    source : Complex[Array, "n"]
        Differentiable finite-system source coefficients.
    max_iterations : scalar_int
        Positive CGLS iteration limit.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.

    Returns
    -------
    field : Complex[Array, "n"]
        Converged fixed-support root.

    Raises
    ------
    ValueError
        If a static solver input is invalid.
    equinox.EquinoxRuntimeError
        If a traced input is invalid or the primal solve does not converge.

    Notes
    -----
    Calling the CGLS core directly prevents the custom VJP rule from invoking
    itself while evaluating the primal root.
    """
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
    operator : _GalerkinSystem
        Fixed-support differentiable algebraic, manifested SC-1, or prepared
        LOCAL_CELL_LVT1 realization.
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
) -> Tuple[
    Complex[Array, "n"],
    Tuple[
        _GalerkinSystem,
        Complex[Array, "n"],
        Int[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ],
]:
    """PRIVATE: Save the converged root and fixed data without trajectory.

    Parameters
    ----------
    operator : _GalerkinSystem
        Fixed-support differentiable target.
    source : Complex[Array, "n"]
        Differentiable finite-system source coefficients.
    max_iterations : scalar_int
        Positive CGLS iteration limit.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.

    Returns
    -------
    field : Complex[Array, "n"]
        Converged fixed-support root.
    residual : Tuple[_GalerkinSystem, Complex[Array, "n"], Int[Array, ""], Float[Array, ""], Float[Array, ""]]
        Custom-VJP residual containing the operator, root, and solver controls.

    Raises
    ------
    ValueError
        If a static solver input is invalid.
    equinox.EquinoxRuntimeError
        If a traced input is invalid or the primal solve does not converge.

    Notes
    -----
    The custom-VJP residual deliberately excludes every Krylov trajectory
    vector.
    """  # noqa: E501
    field: Complex[Array, "n"] = _implicit_galerkin_solve_impl(
        operator,
        source,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )
    residual: Tuple[
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
    result: Tuple[
        Complex[Array, "n"],
        Tuple[
            _GalerkinSystem,
            Complex[Array, "n"],
            Int[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
    ] = field, residual
    return result


def _implicit_galerkin_solve_bwd(
    residual: Tuple[
        _GalerkinSystem,
        Complex[Array, "n"],
        Int[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ],
    output_cotangent: Complex[Array, "n"],
) -> Tuple[_GalerkinSystem, Complex[Array, "n"], None, None, None]:
    """PRIVATE: Solve the SC-1 adjoint and pull back operator parameters.

    Parameters
    ----------
    residual : Tuple[_GalerkinSystem, Complex[Array, "n"], Int[Array, ""], Float[Array, ""], Float[Array, ""]]
        Saved operator, primal root, and solver controls from the forward rule.
    output_cotangent : Complex[Array, "n"]
        Cotangent of the converged field under JAX's complex convention.

    Returns
    -------
    operator_cotangent : _GalerkinSystem
        Pulled-back cotangent for differentiable operator leaves.
    source_cotangent : Complex[Array, "n"]
        Source cotangent from the converged adjoint root.
    max_iterations_cotangent : None
        No cotangent for the discrete iteration limit.
    relative_tolerance_cotangent : None
        No cotangent for the solver relative tolerance.
    absolute_tolerance_cotangent : None
        No cotangent for the solver absolute tolerance.

    Raises
    ------
    ValueError
        If a static adjoint-solver input is invalid.
    equinox.EquinoxRuntimeError
        If the adjoint solve is invalid or does not converge.

    Notes
    -----
    The rule solves ``H* lambda = grad_u L`` and differentiates the forward
    action at the saved primal state. It does not differentiate solver
    controls or a Krylov trajectory.
    """  # noqa: E501
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
    cotangents: Tuple[
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
    "cgls_adjoint_solve",
    "cgls_solve",
    "evaluate_galerkin_adjoint_residual",
    "evaluate_galerkin_residual",
    "implicit_galerkin_solve",
    "lsqr_solve",
    "shifted_free_diagonal",
]
