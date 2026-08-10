r"""Enclose rounded scalar Galerkin actions and independent residuals.

Extended Summary
----------------
This module implements the per-state numerical layer of RM-S2.  A production
FFT action is compared with an independently recomputed bounded-memory direct
coefficient contraction.  The same direct contraction is enclosed against
exact-real arithmetic on the stored binary64 ``H_alg`` data by componentwise
outward intervals.  A separate residual path freshly recomputes the direct
action, forms ``S_alg - d_direct``, and encloses its subtraction and norm.

Routine Listings
----------------
:func:`enclose_galerkin_residual`
    Enclose an independently formed same-``H_alg`` residual.
:func:`enclose_galerkin_target_action`
    Enclose one rounded production action at a submitted state.

Notes
-----
The action result is a witness only for its submitted state.  It is not a
uniform rounded-callable bound, a secant bound, or a fixed perturbation
matrix.  The residual result implements S2.64 rather than S2.91 and therefore
does not charge the production FFT action error.  Fixed-linear ``delta_H``,
source error, model discrepancy, and solver recurrence error remain separate.
"""

from __future__ import annotations

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
    Float64,
    jaxtyped,
)

from ptyrodactyl._interval import (
    _all_normal_arithmetic_supported,
    _arithmetic_environment_probes,
    _interval_subtract,
    _point_interval,
    _RealInterval,
    _upward_add,
    _upward_divide,
)
from ptyrodactyl._numeric import has_subnormal_components
from ptyrodactyl.types import (
    GalerkinActionDirection,
    GalerkinActionErrorRoute,
    GalerkinResidualErrorEnclosure,
    GalerkinTargetActionEnclosure,
    GalerkinTargetManifest,
    create_galerkin_residual_error_enclosure,
    create_galerkin_target_action_enclosure,
)

from ._direct_interval import (
    _complex_interval_add,
    _complex_interval_multiply,
    _complex_interval_subtract,
    _complex_point_interval,
    _ComplexInterval,
    _direct_multiplier_with_interval,
    _exact_complex_norm_interval,
    _nonnegative_vector_norm_upper,
    _point_to_interval_component_upper,
)
from .system import apply_galerkin_target, apply_galerkin_target_adjoint

_ROUTE: GalerkinActionErrorRoute = (
    GalerkinActionErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE
)
_EXACT_ACTION_TARGET: str = (
    "exact-real H_alg action on exact stored binary64 D, interaction, "
    "absorber, CAP scale, and submitted field"
)
_RESIDUAL_TARGET: str = (
    "RM-S2 S2.64 independent residual S_alg-H_alg*x; direct coefficient "
    "action freshly recomputed outside production FFT and solver recurrence"
)
_COEFFICIENT_NORM: str = "SC.12/SC.13 Euclidean complex coefficient norm"
_ACTION_ERROR_SCOPE: str = (
    "submitted-state production-call arithmetic only; excludes fixed-linear "
    "delta_H, source, secant, solver, detector, band, and model errors"
)
_RESIDUAL_ERROR_SCOPE: str = (
    "independent direct-action, subtraction, and norm arithmetic only; "
    "excludes production delta_fl, fixed-linear delta_H, source, solver, "
    "detector, band, and model errors"
)
_ARITHMETIC_ENVIRONMENT: str = (
    "assumed IEEE-754 binary64 round-to-nearest-even model for normal "
    "results and normal-range nextafter, guarded by runtime exemplars but "
    "not claimed as a proof of all backend semantics; FTZ/DAZ is admitted "
    "because binary64 bit-pattern classification is probed, exact subnormal "
    "points are widened sign-wise, and every nonidentity zero or subnormal "
    "result is widened to normal +/-tiny before reuse; every outward "
    "primitive has an XLA optimization barrier"
)


def _action_arithmetic_environment_probes() -> Tuple[
    Bool[Array, ""],
    Bool[Array, ""],
]:
    """PRIVATE: Return the common arithmetic-environment probe results.

    Returns
    -------
    normal_supported : Bool[Array, ""]
        Whether the required normal binary64 arithmetic exemplars pass.
    gradual_underflow_supported : Bool[Array, ""]
        Whether the diagnostic gradual-underflow exemplar passes.
    """
    probes: Tuple[
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
    ] = _arithmetic_environment_probes()
    gradual_underflow_supported: Bool[Array, ""] = probes[-1]
    normal_supported: Bool[Array, ""] = _all_normal_arithmetic_supported()
    result: Tuple[Bool[Array, ""], Bool[Array, ""]] = (
        normal_supported,
        gradual_underflow_supported,
    )
    return result


def _direct_target_with_interval(
    target: GalerkinTargetManifest,
    field: Complex128[Array, " n"],
    *,
    adjoint: bool,
) -> Tuple[Complex128[Array, " n"], _ComplexInterval]:
    """PRIVATE: Directly apply and interval-enclose exact ``H_alg``.

    Implementation Logic
    --------------------
    1. Apply the interaction and absorber by independent direct contractions.
    2. Compose the rounded free, interaction, and signed CAP actions.
    3. Repeat that composition with outward complex rectangles.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target whose inverse-square-Angstrom arrays define ``H_alg``.
    field : Complex128[Array, " n"]
        Exact stored binary state in a caller-defined field unit.
    adjoint : bool
        Dimensionless selector. If true, evaluate the actual adjoint.

    Returns
    -------
    direct : Complex128[Array, " n"]
        Rounded direct action in field units per square Angstrom.
    exact_interval : _ComplexInterval
        Exact-action rectangles in field units per square Angstrom.

    Notes
    -----
    The forward composition is ``D x - R x - i epsilon A x``. The actual
    adjoint uses ``D* x - R* x + i epsilon A* x``. Each starred multiplier is
    evaluated by its conjugate-transpose coefficient lookup. The interval path
    encloses exact-real arithmetic on the stored binary64 target and field.
    """
    interaction = _direct_multiplier_with_interval(
        target.support.state_indices,
        target.support.interaction_indices,
        target.interaction_coefficients,
        field,
        target.support.work_shape,
        adjoint=adjoint,
    )
    absorber = _direct_multiplier_with_interval(
        target.support.state_indices,
        target.support.absorber_indices,
        target.absorber_coefficients,
        field,
        target.support.work_shape,
        adjoint=adjoint,
    )
    cap_sign: complex = 1j if adjoint else -1j
    rounded_cap_action: Complex128[Array, " n"] = (
        jnp.asarray(cap_sign, dtype=jnp.complex128)
        * target.cap_scale
        * absorber[0]
    )
    direct: Complex128[Array, " n"] = (
        target.free_diagonal * field - interaction[0] + rounded_cap_action
    )

    free_interval: _ComplexInterval = _complex_interval_multiply(
        _complex_point_interval(target.free_diagonal.astype(jnp.complex128)),
        _complex_point_interval(field),
    )
    interaction_interval: _ComplexInterval = interaction[1:]
    absorber_interval: _ComplexInterval = absorber[1:]
    cap_interval: _ComplexInterval = _complex_interval_multiply(
        _complex_point_interval(jnp.asarray(cap_sign, dtype=jnp.complex128)),
        _complex_interval_multiply(
            _complex_point_interval(target.cap_scale.astype(jnp.complex128)),
            absorber_interval,
        ),
    )
    exact_interval: _ComplexInterval = _complex_interval_add(
        _complex_interval_subtract(free_interval, interaction_interval),
        cap_interval,
    )
    result: Tuple[Complex128[Array, " n"], _ComplexInterval] = (
        direct,
        exact_interval,
    )
    return result


def _scale_safe_complex_norm(
    vector: Complex128[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Compute one scale-safe rounded Euclidean norm diagnostic.

    Implementation Logic
    --------------------
    1. Scale component magnitudes by their maximum without dividing by zero.
    2. Compute the rounded norm of the scaled vector.
    3. Rescale the diagnostic and preserve the exact all-zero result.

    Parameters
    ----------
    vector : Complex128[Array, " n"]
        Stored complex vector in a caller-defined unit.

    Returns
    -------
    norm : Float64[Array, ""]
        Rounded scale-safe Euclidean norm in the vector unit.

    Notes
    -----
    For ``s = max(abs(vector))``, the nonzero branch evaluates
    ``s * sqrt(sum((abs(vector) / s)**2))``. The all-zero branch returns exact
    zero. This diagnostic is not outward rounded; a separate interval helper
    encloses its error. The maximum and zero branch are not smooth boundaries.
    """
    magnitudes: Float64[Array, " n"] = jnp.abs(vector)
    scale: Float64[Array, ""] = jnp.max(magnitudes)
    safe_scale: Float64[Array, ""] = jnp.where(scale > 0.0, scale, 1.0)
    scaled_norm: Float64[Array, ""] = jnp.sqrt(
        jnp.sum((magnitudes / safe_scale) ** 2)
    )
    norm: Float64[Array, ""] = jnp.where(
        scale == 0.0,
        0.0,
        scale * scaled_norm,
    )
    return norm


def _scalar_point_to_interval_upper(
    point: Float64[Array, ""],
    interval: _RealInterval,
) -> Float64[Array, ""]:
    """PRIVATE: Bound one exact point's distance from an interval value.

    Parameters
    ----------
    point : Float64[Array, ""]
        Exact stored binary64 point in a caller-defined unit.
    interval : _RealInterval
        Inclusive comparison interval in the same unit as ``point``.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward absolute-distance upper bound in the shared input unit.

    Notes
    -----
    The result is the larger absolute endpoint of ``point - interval``. The
    maximum boundary makes this evidence helper unsuitable for model gradients.
    """
    difference: _RealInterval = _interval_subtract(
        _point_interval(point),
        interval,
    )
    upper: Float64[Array, ""] = jnp.maximum(
        jnp.abs(difference[0]),
        jnp.abs(difference[1]),
    )
    return upper


def _checked_state_vector(
    target: GalerkinTargetManifest,
    values: Complex[Array, "..."],
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one target-sized finite normal-range vector.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Target defining the dimensionless retained state size.
    values : Complex[Array, "..."]
        Candidate vector in its caller-defined field or source unit.
    name : str
        Unit-free vector name used in errors.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Checked binary64 vector in the input physical unit.

    Raises
    ------
    ValueError
        If vector rank or length is invalid.
    equinox.EquinoxRuntimeError
        If a value is non-finite or nonzero subnormal.

    Notes
    -----
    This host/traced boundary canonicalizes storage to complex128, then checks
    rank and state size on the host and value validity under JAX execution.
    """
    array: Complex128[Array, " n"] = jnp.asarray(values, dtype=jnp.complex128)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if array.shape[0] != target.support.state_indices.shape[0]:
        raise ValueError(f"{name} must match the state support")
    checked: Complex128[Array, " n"] = eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | has_subnormal_components(array),
        f"{name} must be finite and contain no nonzero subnormal components",
    )
    return checked


def _direction(adjoint: bool) -> GalerkinActionDirection:
    """PRIVATE: Convert a static adjoint flag to the public direction enum.

    Parameters
    ----------
    adjoint : bool
        Dimensionless static direction selector for the evidence carrier.

    Returns
    -------
    direction : GalerkinActionDirection
        Unit-free static public direction.
    """
    direction: GalerkinActionDirection = (
        GalerkinActionDirection.ADJOINT
        if adjoint
        else GalerkinActionDirection.FORWARD
    )
    return direction


@jaxtyped(typechecker=beartype)
def enclose_galerkin_target_action(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    *,
    adjoint: bool = False,
) -> GalerkinTargetActionEnclosure:
    r"""Enclose one rounded production action at a submitted state.

    :see: :class:`~.test_action_enclosures.TestActionEnclosures`

    Implementation Logic
    --------------------
    1. Evaluate the production FFT forward or actual-adjoint action.
    2. Independently contract exact coefficient differences with bounded
       working memory.
    3. Interval-evaluate exact-real ``H_alg x`` from the stored binary data.
    4. Bound production-to-direct and direct-to-exact distances separately.
    5. Return an absolute per-call bound and a submitted-state ratio in the
       S2.89 form, without claiming the uniform S2.89 hypothesis.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical SC-1 target whose stored arrays define ``H_alg``.
    field : Complex[Array, "..."]
        Submitted retained-state coefficients.
    adjoint : bool
        If true, enclose the production actual-adjoint action. Default is
        false.

    Returns
    -------
    enclosure : GalerkinTargetActionEnclosure
        Outward per-state action evidence.

    Raises
    ------
    ValueError
        If field structure is invalid.
    equinox.EquinoxRuntimeError
        If the field or production action violates the target runtime
        contract.

    Notes
    -----
    The returned relative ratio is evidence only for the submitted input. It
    does not establish the uniform S2.89 hypothesis or the S2.94 secant
    condition.
    For the exact zero state it is zero only when the complete action error is
    also exactly zero; otherwise it is positive infinity.
    """
    checked_field: Complex128[Array, " n"] = _checked_state_vector(
        target, field, "field"
    )
    evidence_target: GalerkinTargetManifest = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        target,
    )
    evidence_field: Complex128[Array, " n"] = jax.lax.stop_gradient(
        checked_field
    )
    if adjoint:
        production: Complex128[Array, " n"] = apply_galerkin_target_adjoint(
            evidence_target, evidence_field
        )
    else:
        production = apply_galerkin_target(evidence_target, evidence_field)
    direct, algebraic_interval = _direct_target_with_interval(
        evidence_target,
        evidence_field,
        adjoint=adjoint,
    )
    production_direct_components: Float64[Array, " n"] = (
        _point_to_interval_component_upper(
            production,
            _complex_point_interval(direct),
        )
    )
    direct_algebraic_components: Float64[Array, " n"] = (
        _point_to_interval_component_upper(direct, algebraic_interval)
    )
    production_direct_bound: Float64[Array, ""] = (
        _nonnegative_vector_norm_upper(production_direct_components)
    )
    direct_algebraic_bound: Float64[Array, ""] = (
        _nonnegative_vector_norm_upper(direct_algebraic_components)
    )
    action_error_bound: Float64[Array, ""] = _upward_add(
        production_direct_bound,
        direct_algebraic_bound,
    )
    field_norm: _RealInterval = _exact_complex_norm_interval(evidence_field)
    environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    environment_supported, gradual_underflow_supported = (
        _action_arithmetic_environment_probes()
    )
    algebraic_interval = (
        jnp.where(environment_supported, algebraic_interval[0], -jnp.inf),
        jnp.where(environment_supported, algebraic_interval[1], jnp.inf),
        jnp.where(environment_supported, algebraic_interval[2], -jnp.inf),
        jnp.where(environment_supported, algebraic_interval[3], jnp.inf),
    )
    production_direct_components = jnp.where(
        environment_supported,
        production_direct_components,
        jnp.inf,
    )
    direct_algebraic_components = jnp.where(
        environment_supported,
        direct_algebraic_components,
        jnp.inf,
    )
    production_direct_bound = jnp.where(
        environment_supported,
        production_direct_bound,
        jnp.inf,
    )
    direct_algebraic_bound = jnp.where(
        environment_supported,
        direct_algebraic_bound,
        jnp.inf,
    )
    action_error_bound = jnp.where(
        environment_supported,
        action_error_bound,
        jnp.inf,
    )
    field_norm = (
        jnp.where(environment_supported, field_norm[0], 0.0),
        jnp.where(environment_supported, field_norm[1], jnp.inf),
    )
    relative_finite: Float64[Array, ""] = _upward_divide(
        action_error_bound,
        jnp.where(field_norm[0] > 0.0, field_norm[0], 1.0),
    )
    relative_bound: Float64[Array, ""] = jnp.where(
        field_norm[0] > 0.0,
        relative_finite,
        jnp.where(action_error_bound == 0.0, 0.0, jnp.inf),
    )
    stopped = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        (
            production,
            direct,
            *algebraic_interval,
            production_direct_components,
            direct_algebraic_components,
            production_direct_bound,
            direct_algebraic_bound,
            action_error_bound,
            field_norm[0],
            field_norm[1],
            relative_bound,
            environment_supported,
            gradual_underflow_supported,
        ),
    )
    enclosure: GalerkinTargetActionEnclosure = (
        create_galerkin_target_action_enclosure(
            target=evidence_target,
            submitted_field=evidence_field,
            production_action=stopped[0],
            independent_direct_action=stopped[1],
            algebraic_action_real_lower_bounds=stopped[2],
            algebraic_action_real_upper_bounds=stopped[3],
            algebraic_action_imag_lower_bounds=stopped[4],
            algebraic_action_imag_upper_bounds=stopped[5],
            production_direct_component_error_bounds=stopped[6],
            direct_algebraic_component_error_bounds=stopped[7],
            production_direct_error_bound=stopped[8],
            direct_algebraic_action_error_bound=stopped[9],
            action_error_bound=stopped[10],
            field_norm_lower_bound=stopped[11],
            field_norm_upper_bound=stopped[12],
            per_state_relative_action_error_bound=stopped[13],
            arithmetic_environment_supported=stopped[14],
            gradual_underflow_supported=stopped[15],
            direction=_direction(adjoint),
            route=_ROUTE,
            exact_action_target=_EXACT_ACTION_TARGET,
            coefficient_norm=_COEFFICIENT_NORM,
            error_scope=_ACTION_ERROR_SCOPE,
            arithmetic_environment=_ARITHMETIC_ENVIRONMENT,
        )
    )
    return enclosure


@jaxtyped(typechecker=beartype)
def enclose_galerkin_residual(
    target: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    source: Complex[Array, "..."],
    *,
    adjoint: bool = False,
) -> GalerkinResidualErrorEnclosure:
    r"""Enclose an independently formed same-``H_alg`` residual.

    :see: :class:`~.test_action_enclosures.TestActionEnclosures`

    Implementation Logic
    --------------------
    1. Freshly recompute the bounded-memory direct coefficient action and its
       exact-real ``H_alg x`` interval.
    2. Form ``rhat = fl(S_alg - d_direct)`` outside any solver recurrence.
    3. Enclose direct-action and subtraction errors as distinct terms.
    4. Independently enclose the exact norm of the stored ``rhat`` and the
       error of its rounded scale-safe norm diagnostic.
    5. Return the S2.64 same-``H_alg`` residual-norm upper bound.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target whose stored arrays define ``H_alg``.
    field : Complex[Array, "..."]
        Submitted retained-state coefficients.
    source : Complex[Array, "..."]
        Exact stored binary algebraic source ``S_alg``.
    adjoint : bool
        If true, enclose ``S_alg - H_alg* x``. Default is false.

    Returns
    -------
    enclosure : GalerkinResidualErrorEnclosure
        Independent S2.64 residual evidence.

    Raises
    ------
    ValueError
        If field or source structure is invalid.
    equinox.EquinoxRuntimeError
        If a field or source value is non-finite or nonzero subnormal.

    Notes
    -----
    This function never calls the production FFT action and never reuses a
    Krylov, normal-equation, transformed, or preconditioned residual.  Its
    evaluator error already contains the direct-action interval error, so a
    production ``delta_fl`` must not be added.  Fixed-linear ``delta_H`` and
    source transfer are composed later exactly once.
    """
    checked_field: Complex128[Array, " n"] = _checked_state_vector(
        target, field, "field"
    )
    checked_source: Complex128[Array, " n"] = _checked_state_vector(
        target, source, "source"
    )
    evidence_target: GalerkinTargetManifest = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        target,
    )
    evidence_field: Complex128[Array, " n"] = jax.lax.stop_gradient(
        checked_field
    )
    evidence_source: Complex128[Array, " n"] = jax.lax.stop_gradient(
        checked_source
    )
    direct, algebraic_interval = _direct_target_with_interval(
        evidence_target,
        evidence_field,
        adjoint=adjoint,
    )
    direct_algebraic_components: Float64[Array, " n"] = (
        _point_to_interval_component_upper(direct, algebraic_interval)
    )
    direct_algebraic_bound: Float64[Array, ""] = (
        _nonnegative_vector_norm_upper(direct_algebraic_components)
    )

    rounded_source, rounded_direct = jax.lax.optimization_barrier(
        (evidence_source, direct)
    )
    formed_residual: Complex128[Array, " n"] = rounded_source - rounded_direct
    exact_source_minus_direct: _ComplexInterval = _complex_interval_subtract(
        _complex_point_interval(evidence_source),
        _complex_point_interval(direct),
    )
    subtraction_components: Float64[Array, " n"] = (
        _point_to_interval_component_upper(
            formed_residual,
            exact_source_minus_direct,
        )
    )
    subtraction_bound: Float64[Array, ""] = _nonnegative_vector_norm_upper(
        subtraction_components
    )
    evaluator_bound: Float64[Array, ""] = _upward_add(
        direct_algebraic_bound,
        subtraction_bound,
    )

    formed_norm: Float64[Array, ""] = _scale_safe_complex_norm(formed_residual)
    exact_formed_norm: _RealInterval = _exact_complex_norm_interval(
        formed_residual
    )
    norm_error: Float64[Array, ""] = _scalar_point_to_interval_upper(
        formed_norm,
        exact_formed_norm,
    )
    algebraic_residual_upper: Float64[Array, ""] = _upward_add(
        exact_formed_norm[1],
        evaluator_bound,
    )
    field_norm: _RealInterval = _exact_complex_norm_interval(evidence_field)
    environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    environment_supported, gradual_underflow_supported = (
        _action_arithmetic_environment_probes()
    )
    direct_algebraic_components = jnp.where(
        environment_supported,
        direct_algebraic_components,
        jnp.inf,
    )
    subtraction_components = jnp.where(
        environment_supported,
        subtraction_components,
        jnp.inf,
    )
    direct_algebraic_bound = jnp.where(
        environment_supported,
        direct_algebraic_bound,
        jnp.inf,
    )
    subtraction_bound = jnp.where(
        environment_supported,
        subtraction_bound,
        jnp.inf,
    )
    evaluator_bound = jnp.where(
        environment_supported,
        evaluator_bound,
        jnp.inf,
    )
    exact_formed_norm = (
        jnp.where(environment_supported, exact_formed_norm[0], 0.0),
        jnp.where(environment_supported, exact_formed_norm[1], jnp.inf),
    )
    norm_error = jnp.where(environment_supported, norm_error, jnp.inf)
    algebraic_residual_upper = jnp.where(
        environment_supported,
        algebraic_residual_upper,
        jnp.inf,
    )
    field_norm = (
        jnp.where(environment_supported, field_norm[0], 0.0),
        jnp.where(environment_supported, field_norm[1], jnp.inf),
    )
    stopped = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        (
            direct,
            formed_residual,
            direct_algebraic_components,
            subtraction_components,
            direct_algebraic_bound,
            subtraction_bound,
            evaluator_bound,
            formed_norm,
            exact_formed_norm[0],
            exact_formed_norm[1],
            norm_error,
            algebraic_residual_upper,
            field_norm[0],
            field_norm[1],
            environment_supported,
            gradual_underflow_supported,
        ),
    )
    enclosure: GalerkinResidualErrorEnclosure = (
        create_galerkin_residual_error_enclosure(
            target=evidence_target,
            submitted_field=evidence_field,
            algebraic_source=evidence_source,
            independent_direct_action=stopped[0],
            formed_residual=stopped[1],
            direct_algebraic_component_error_bounds=stopped[2],
            subtraction_component_error_bounds=stopped[3],
            direct_algebraic_action_error_bound=stopped[4],
            subtraction_error_bound=stopped[5],
            residual_evaluator_error_bound=stopped[6],
            formed_residual_norm=stopped[7],
            formed_residual_norm_lower_bound=stopped[8],
            formed_residual_norm_upper_bound=stopped[9],
            norm_evaluation_error_bound=stopped[10],
            algebraic_residual_norm_upper_bound=stopped[11],
            field_norm_lower_bound=stopped[12],
            field_norm_upper_bound=stopped[13],
            arithmetic_environment_supported=stopped[14],
            gradual_underflow_supported=stopped[15],
            direction=_direction(adjoint),
            route=_ROUTE,
            residual_target=_RESIDUAL_TARGET,
            coefficient_norm=_COEFFICIENT_NORM,
            error_scope=_RESIDUAL_ERROR_SCOPE,
            arithmetic_environment=_ARITHMETIC_ENVIRONMENT,
        )
    )
    return enclosure


__all__: list[str] = [
    "enclose_galerkin_residual",
    "enclose_galerkin_target_action",
]
