r"""Define per-call RM-S2 action and residual error evidence.

Extended Summary
----------------
This module owns typed evidence for one submitted binary64 state.  It keeps
the rounded production action, an independent direct coefficient action, and
the exact-real frozen algebraic target separate.  The residual carrier uses
the independent direct action and therefore implements the RM-S2 S2.64
route, not the S2.91 production-callable residual route.

Routine Listings
----------------
:class:`GalerkinActionDirection`
    Store the finite operator direction being enclosed.
:class:`GalerkinActionErrorRoute`
    Store the admitted per-call action enclosure route.
:class:`GalerkinResidualErrorEnclosure`
    Store one independent same-``H_alg`` residual enclosure.
:class:`GalerkinTargetActionEnclosure`
    Store one per-state RM-S2 production-action enclosure.
:func:`create_galerkin_residual_error_enclosure`
    Create a structurally validated independent residual enclosure.
:func:`create_galerkin_target_action_enclosure`
    Create a structurally validated per-state action enclosure.

Notes
-----
Positive infinity is a typed noncertificate.  These carriers deliberately do
not contain the fixed-linear target error ``delta_H``, source-model error, a
uniform action bound over all states, or an RM-S2 secant bound.
"""

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
    jaxtyped,
)

from ptyrodactyl._tools import has_subnormal_components

from .galerkin_types import GalerkinTargetManifest


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

    Parameters
    ----------
    condition : bool
        Dimensionless structural failure predicate.
    message : str
        Unit-free exception text used when the predicate is true.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


class GalerkinActionDirection(str, Enum):
    """Store the finite operator direction being enclosed.

    :see: :class:`~.test_action_error_types.TestActionErrorTypes`

    Attributes
    ----------
    FORWARD : str
        Enclose ``H_alg x`` and the production forward action.
    ADJOINT : str
        Enclose ``H_alg* x`` and the production actual-adjoint action.
    """

    FORWARD = "forward"
    ADJOINT = "adjoint"


class GalerkinActionErrorRoute(str, Enum):
    """Store the admitted per-call action enclosure route.

    :see: :class:`~.test_action_error_types.TestActionErrorTypes`

    Attributes
    ----------
    FTZ_SAFE_DIRECT_INTERVAL_BRIDGE : str
        Compare the production FFT action with an independent bounded-memory
        direct contraction, then bridge that result to exact-real ``H_alg x``
        with intervals that never propagate nonzero subnormal endpoints.
    """

    FTZ_SAFE_DIRECT_INTERVAL_BRIDGE = "rm_s2_ftz_safe_direct_interval_bridge"


class GalerkinTargetActionEnclosure(eqx.Module):
    """Store one per-state RM-S2 production-action enclosure.

    :see: :class:`~.test_action_error_types.TestActionErrorTypes`

    Attributes
    ----------
    target : GalerkinTargetManifest
        Exact manifested identity of the frozen algebraic target.
    submitted_field : Complex128[Array, " n"]
        Exact stored binary input for this per-state witness.
    production_action : Complex128[Array, " n"]
        Rounded production FFT action ``F_H(x)``.
    independent_direct_action : Complex128[Array, " n"]
        Separately evaluated bounded-memory direct coefficient action.
    algebraic_action_real_lower_bounds : Float64[Array, " n"]
        Lower endpoints enclosing the real components of exact ``H_alg x``.
    algebraic_action_real_upper_bounds : Float64[Array, " n"]
        Upper endpoints enclosing the real components of exact ``H_alg x``.
    algebraic_action_imag_lower_bounds : Float64[Array, " n"]
        Lower endpoints enclosing the imaginary components of exact
        ``H_alg x``.
    algebraic_action_imag_upper_bounds : Float64[Array, " n"]
        Upper endpoints enclosing the imaginary components of exact
        ``H_alg x``.
    production_direct_component_error_bounds : Float64[Array, " n"]
        Componentwise bounds between the two stored rounded actions.
    direct_algebraic_component_error_bounds : Float64[Array, " n"]
        Componentwise direct-action errors relative to exact ``H_alg x``.
    production_direct_error_bound : Float64[Array, ""]
        Euclidean bound between the production and direct stored actions.
    direct_algebraic_action_error_bound : Float64[Array, ""]
        Euclidean bound between the direct action and exact ``H_alg x``.
    action_error_bound : Float64[Array, ""]
        Per-call absolute bound on ``||F_H(x) - H_alg x||``.
    field_norm_lower_bound : Float64[Array, ""]
        Lower endpoint enclosing the exact norm of the stored input.
    field_norm_upper_bound : Float64[Array, ""]
        Upper endpoint enclosing the exact norm of the stored input.
    per_state_relative_action_error_bound : Float64[Array, ""]
        Submitted-state ratio in the S2.89 form. It does not establish the
        uniform S2.89 hypothesis or a secant bound.
    arithmetic_environment_supported : Bool[Array, ""]
        Whether runtime probes observed binary64 round-to-nearest-even and
        the required normal-range ``nextafter`` behavior.
    gradual_underflow_supported : Bool[Array, ""]
        Whether runtime probes observed gradual underflow. The FTZ-safe route
        remains valid when this is false.
    finite_certificate : Bool[Array, ""]
        Whether every submitted-state action bound is finite under the stated
        arithmetic model. It does not certify the uniform S2.89 hypothesis.
    direction : GalerkinActionDirection
        Static forward/adjoint direction. This value affects tracing.
    route : GalerkinActionErrorRoute
        Static enclosure route. This value affects tracing.
    exact_action_target : str
        Static exact-real target declaration. This value affects tracing.
    coefficient_norm : str
        Static norm declaration. This value affects tracing.
    error_scope : str
        Static exclusions from this evidence. This value affects tracing.
    arithmetic_environment : str
        Static arithmetic assumptions and runtime-probe declaration. This
        value affects tracing.

    Notes
    -----
    ``per_state_relative_action_error_bound`` is valid only for the submitted
    state.  No singular value, linear perturbation matrix, injectivity claim,
    or S2.94 secant conclusion attaches to the rounded callable.
    """

    target: GalerkinTargetManifest
    submitted_field: Complex128[Array, " n"]
    production_action: Complex128[Array, " n"]
    independent_direct_action: Complex128[Array, " n"]
    algebraic_action_real_lower_bounds: Float64[Array, " n"]
    algebraic_action_real_upper_bounds: Float64[Array, " n"]
    algebraic_action_imag_lower_bounds: Float64[Array, " n"]
    algebraic_action_imag_upper_bounds: Float64[Array, " n"]
    production_direct_component_error_bounds: Float64[Array, " n"]
    direct_algebraic_component_error_bounds: Float64[Array, " n"]
    production_direct_error_bound: Float64[Array, ""]
    direct_algebraic_action_error_bound: Float64[Array, ""]
    action_error_bound: Float64[Array, ""]
    field_norm_lower_bound: Float64[Array, ""]
    field_norm_upper_bound: Float64[Array, ""]
    per_state_relative_action_error_bound: Float64[Array, ""]
    arithmetic_environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    finite_certificate: Bool[Array, ""]
    direction: GalerkinActionDirection = eqx.field(static=True)
    route: GalerkinActionErrorRoute = eqx.field(static=True)
    exact_action_target: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    arithmetic_environment: str = eqx.field(static=True)


class GalerkinResidualErrorEnclosure(eqx.Module):
    """Store one independent same-``H_alg`` residual enclosure.

    :see: :class:`~.test_action_error_types.TestActionErrorTypes`

    Attributes
    ----------
    target : GalerkinTargetManifest
        Exact manifested identity of the frozen algebraic target.
    submitted_field : Complex128[Array, " n"]
        Exact stored binary submitted state.
    algebraic_source : Complex128[Array, " n"]
        Exact stored binary right-hand side ``S_alg``.
    independent_direct_action : Complex128[Array, " n"]
        Fresh bounded-memory direct action used only by this residual call.
    formed_residual : Complex128[Array, " n"]
        Stored subtraction ``fl(S_alg - d_direct)``.
    direct_algebraic_component_error_bounds : Float64[Array, " n"]
        Componentwise errors between the direct action and exact
        ``H_alg x``.
    subtraction_component_error_bounds : Float64[Array, " n"]
        Componentwise errors in forming ``S_alg - d_direct``.
    direct_algebraic_action_error_bound : Float64[Array, ""]
        Euclidean direct-action arithmetic bound.
    subtraction_error_bound : Float64[Array, ""]
        Euclidean residual-subtraction error bound.
    residual_evaluator_error_bound : Float64[Array, ""]
        S2.64 bound on the vector error relative to ``S_alg - H_alg x``.
    formed_residual_norm : Float64[Array, ""]
        Scale-safe rounded diagnostic norm of ``formed_residual``.
    formed_residual_norm_lower_bound : Float64[Array, ""]
        Lower endpoint for the exact norm of the stored residual vector.
    formed_residual_norm_upper_bound : Float64[Array, ""]
        Upper endpoint for the exact norm of the stored residual vector.
    norm_evaluation_error_bound : Float64[Array, ""]
        Error of the rounded diagnostic norm relative to that exact norm.
    algebraic_residual_norm_upper_bound : Float64[Array, ""]
        S2.64 upper bound on ``||S_alg - H_alg x||``.
    field_norm_lower_bound : Float64[Array, ""]
        Lower endpoint for the exact norm of the submitted stored field.
    field_norm_upper_bound : Float64[Array, ""]
        Upper endpoint for the exact norm of the submitted stored field.
    arithmetic_environment_supported : Bool[Array, ""]
        Whether runtime probes observed the required normal binary64
        environment.
    gradual_underflow_supported : Bool[Array, ""]
        Whether runtime probes observed gradual underflow. This is diagnostic
        for the FTZ-safe route.
    finite_certificate : Bool[Array, ""]
        Whether every bound for this submitted-state residual is finite under
        the stated arithmetic model.
    direction : GalerkinActionDirection
        Static forward/adjoint direction. This value affects tracing.
    route : GalerkinActionErrorRoute
        Static direct interval route. This value affects tracing.
    residual_target : str
        Static same-target declaration. This value affects tracing.
    coefficient_norm : str
        Static norm declaration. This value affects tracing.
    error_scope : str
        Static exclusions from this evidence. This value affects tracing.
    arithmetic_environment : str
        Static arithmetic assumptions and runtime-probe declaration. This
        value affects tracing.

    Notes
    -----
    This is the independent S2.64 route.  It is not S2.91 because the formed
    residual does not subtract the production FFT action.  Consequently the
    production ``delta_fl`` is not charged here; direct-action and subtraction
    errors already cover the same exact ``H_alg`` target.
    """

    target: GalerkinTargetManifest
    submitted_field: Complex128[Array, " n"]
    algebraic_source: Complex128[Array, " n"]
    independent_direct_action: Complex128[Array, " n"]
    formed_residual: Complex128[Array, " n"]
    direct_algebraic_component_error_bounds: Float64[Array, " n"]
    subtraction_component_error_bounds: Float64[Array, " n"]
    direct_algebraic_action_error_bound: Float64[Array, ""]
    subtraction_error_bound: Float64[Array, ""]
    residual_evaluator_error_bound: Float64[Array, ""]
    formed_residual_norm: Float64[Array, ""]
    formed_residual_norm_lower_bound: Float64[Array, ""]
    formed_residual_norm_upper_bound: Float64[Array, ""]
    norm_evaluation_error_bound: Float64[Array, ""]
    algebraic_residual_norm_upper_bound: Float64[Array, ""]
    field_norm_lower_bound: Float64[Array, ""]
    field_norm_upper_bound: Float64[Array, ""]
    arithmetic_environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    finite_certificate: Bool[Array, ""]
    direction: GalerkinActionDirection = eqx.field(static=True)
    route: GalerkinActionErrorRoute = eqx.field(static=True)
    residual_target: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    arithmetic_environment: str = eqx.field(static=True)


def _checked_vector(
    values: Complex[Array, "..."],
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Convert and structurally validate one complex vector.

    Parameters
    ----------
    values : Complex[Array, "..."]
        Candidate vector in the physical units of the owning evidence field.
    name : str
        Unit-free field name used in a structural error.

    Returns
    -------
    result : Complex128[Array, " n"]
        Canonical complex binary64 vector in the input field's units.

    Raises
    ------
    ValueError
        If the converted vector is not one-dimensional or is empty.

    Notes
    -----
    The binary64 conversion can round values. It never converts physical
    units.
    """
    result: Complex128[Array, " n"] = jnp.asarray(values, dtype=jnp.complex128)
    _raise_if(result.ndim != 1, f"{name} must be 1D")
    _raise_if(result.shape[0] == 0, f"{name} must be nonempty")
    return result


def _checked_float_vector(
    values: Float[Array, "..."],
    size: int,
    name: str,
) -> Float64[Array, " n"]:
    """PRIVATE: Convert and structurally validate one real evidence vector.

    Parameters
    ----------
    values : Float[Array, "..."]
        Candidate evidence vector in the owning field's physical units.
    size : int
        Required dimensionless vector length.
    name : str
        Unit-free field name used in a structural error.

    Returns
    -------
    result : Float64[Array, " n"]
        Canonical real binary64 vector in the input field's units.

    Raises
    ------
    ValueError
        If the converted vector does not have shape ``(size,)``.

    Notes
    -----
    The binary64 conversion can round values. It never converts physical
    units.
    """
    result: Float64[Array, " n"] = jnp.asarray(values, dtype=jnp.float64)
    _raise_if(result.shape != (size,), f"{name} must have shape ({size},)")
    return result


def _checked_scalar(
    value: Float[Array, ""],
    name: str,
) -> Float64[Array, ""]:
    """PRIVATE: Convert and structurally validate one real scalar.

    Parameters
    ----------
    value : Float[Array, ""]
        Candidate evidence scalar in the owning field's physical units.
    name : str
        Unit-free field name used in a structural error.

    Returns
    -------
    result : Float64[Array, ""]
        Canonical real binary64 scalar in the input field's units.

    Raises
    ------
    ValueError
        If the converted value is not a scalar.

    Notes
    -----
    The binary64 conversion can round values. It never converts physical
    units.
    """
    result: Float64[Array, ""] = jnp.asarray(value, dtype=jnp.float64)
    _raise_if(result.shape != (), f"{name} must be a scalar")
    return result


@jaxtyped(typechecker=beartype)
def create_galerkin_target_action_enclosure(  # noqa: PLR0913
    target: GalerkinTargetManifest,
    submitted_field: Complex[Array, "..."],
    production_action: Complex[Array, "..."],
    independent_direct_action: Complex[Array, "..."],
    algebraic_action_real_lower_bounds: Float[Array, "..."],
    algebraic_action_real_upper_bounds: Float[Array, "..."],
    algebraic_action_imag_lower_bounds: Float[Array, "..."],
    algebraic_action_imag_upper_bounds: Float[Array, "..."],
    production_direct_component_error_bounds: Float[Array, "..."],
    direct_algebraic_component_error_bounds: Float[Array, "..."],
    production_direct_error_bound: Float[Array, ""],
    direct_algebraic_action_error_bound: Float[Array, ""],
    action_error_bound: Float[Array, ""],
    field_norm_lower_bound: Float[Array, ""],
    field_norm_upper_bound: Float[Array, ""],
    per_state_relative_action_error_bound: Float[Array, ""],
    arithmetic_environment_supported: Bool[Array, ""],
    gradual_underflow_supported: Bool[Array, ""],
    *,
    direction: GalerkinActionDirection | str,
    route: GalerkinActionErrorRoute | str,
    exact_action_target: str,
    coefficient_norm: str,
    error_scope: str,
    arithmetic_environment: str,
) -> GalerkinTargetActionEnclosure:
    """Create a structurally validated per-state action enclosure.

    :see: :class:`~.test_action_error_types.TestActionErrorTypes`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Frozen target bound to this result.
    submitted_field : Complex[Array, "..."]
        Exact stored binary submitted state.
    production_action : Complex[Array, "..."]
        Rounded production action.
    independent_direct_action : Complex[Array, "..."]
        Independently recomputed direct action.
    algebraic_action_real_lower_bounds : Float[Array, "..."]
        Exact-action real lower endpoints.
    algebraic_action_real_upper_bounds : Float[Array, "..."]
        Exact-action real upper endpoints.
    algebraic_action_imag_lower_bounds : Float[Array, "..."]
        Exact-action imaginary lower endpoints.
    algebraic_action_imag_upper_bounds : Float[Array, "..."]
        Exact-action imaginary upper endpoints.
    production_direct_component_error_bounds : Float[Array, "..."]
        Componentwise production/direct errors.
    direct_algebraic_component_error_bounds : Float[Array, "..."]
        Componentwise direct/exact-algebraic errors.
    production_direct_error_bound : Float[Array, ""]
        Euclidean production/direct error.
    direct_algebraic_action_error_bound : Float[Array, ""]
        Euclidean direct/exact-algebraic error.
    action_error_bound : Float[Array, ""]
        Complete per-call absolute action error.
    field_norm_lower_bound : Float[Array, ""]
        Exact input-norm lower endpoint.
    field_norm_upper_bound : Float[Array, ""]
        Exact input-norm upper endpoint.
    per_state_relative_action_error_bound : Float[Array, ""]
        Per-state relative action error.
    arithmetic_environment_supported : Bool[Array, ""]
        Result of the runtime binary64 arithmetic-environment probe.
    gradual_underflow_supported : Bool[Array, ""]
        Result of the runtime gradual-underflow probe.
    direction : GalerkinActionDirection | str
        Forward or adjoint target.
    route : GalerkinActionErrorRoute | str
        Enclosure route.
    exact_action_target : str
        Nonempty target declaration.
    coefficient_norm : str
        Nonempty norm declaration.
    error_scope : str
        Nonempty error-scope declaration.
    arithmetic_environment : str
        Nonempty arithmetic-environment declaration.

    Returns
    -------
    enclosure : GalerkinTargetActionEnclosure
        Validated submitted-state action evidence. Infinity remains a typed
        noncertificate; finiteness does not establish uniform S2.89.

    Raises
    ------
    ValueError
        If static structure or metadata is invalid.
    equinox.EquinoxRuntimeError
        If an interval is reversed or any bound is NaN or negative.
    """
    submitted = _checked_vector(submitted_field, "submitted_field")
    production = _checked_vector(production_action, "production_action")
    direct = _checked_vector(
        independent_direct_action, "independent_direct_action"
    )
    _raise_if(
        direct.shape != production.shape,
        "action vectors must have matching shapes",
    )
    size: int = production.shape[0]
    _raise_if(
        submitted.shape != production.shape,
        "submitted field and action vectors must have matching shapes",
    )
    _raise_if(
        target.support.state_indices.shape[0] != size,
        "target support must match the submitted action size",
    )
    real_lower = _checked_float_vector(
        algebraic_action_real_lower_bounds,
        size,
        "algebraic_action_real_lower_bounds",
    )
    real_upper = _checked_float_vector(
        algebraic_action_real_upper_bounds,
        size,
        "algebraic_action_real_upper_bounds",
    )
    imag_lower = _checked_float_vector(
        algebraic_action_imag_lower_bounds,
        size,
        "algebraic_action_imag_lower_bounds",
    )
    imag_upper = _checked_float_vector(
        algebraic_action_imag_upper_bounds,
        size,
        "algebraic_action_imag_upper_bounds",
    )
    production_direct_components = _checked_float_vector(
        production_direct_component_error_bounds,
        size,
        "production_direct_component_error_bounds",
    )
    direct_algebraic_components = _checked_float_vector(
        direct_algebraic_component_error_bounds,
        size,
        "direct_algebraic_component_error_bounds",
    )
    scalar_names = (
        "production_direct_error_bound",
        "direct_algebraic_action_error_bound",
        "action_error_bound",
        "field_norm_lower_bound",
        "field_norm_upper_bound",
        "per_state_relative_action_error_bound",
    )
    scalars = tuple(
        _checked_scalar(value, name)
        for value, name in zip(
            (
                production_direct_error_bound,
                direct_algebraic_action_error_bound,
                action_error_bound,
                field_norm_lower_bound,
                field_norm_upper_bound,
                per_state_relative_action_error_bound,
            ),
            scalar_names,
            strict=True,
        )
    )
    (
        production_direct_bound,
        direct_algebraic_bound,
        complete_bound,
        field_lower,
        field_upper,
        relative_bound,
    ) = scalars
    for metadata, name in (
        (exact_action_target, "exact_action_target"),
        (coefficient_norm, "coefficient_norm"),
        (error_scope, "error_scope"),
        (arithmetic_environment, "arithmetic_environment"),
    ):
        _raise_if(not metadata.strip(), f"{name} must be nonempty")
    environment_supported: Bool[Array, ""] = jnp.asarray(
        arithmetic_environment_supported,
        dtype=jnp.bool_,
    )
    _raise_if(
        environment_supported.shape != (),
        "arithmetic_environment_supported must be a scalar",
    )
    gradual_supported: Bool[Array, ""] = jnp.asarray(
        gradual_underflow_supported,
        dtype=jnp.bool_,
    )
    _raise_if(
        gradual_supported.shape != (),
        "gradual_underflow_supported must be a scalar",
    )

    invalid: Bool[Array, ""] = (
        jnp.any(~jnp.isfinite(submitted))
        | has_subnormal_components(submitted)
        | jnp.any(~jnp.isfinite(production))
        | jnp.any(~jnp.isfinite(direct))
        | jnp.any(jnp.isnan(real_lower) | jnp.isnan(real_upper))
        | jnp.any(jnp.isnan(imag_lower) | jnp.isnan(imag_upper))
        | jnp.any(real_lower > real_upper)
        | jnp.any(imag_lower > imag_upper)
        | jnp.any(
            jnp.isnan(production_direct_components)
            | (production_direct_components < 0.0)
        )
        | jnp.any(
            jnp.isnan(direct_algebraic_components)
            | (direct_algebraic_components < 0.0)
        )
    )
    for value in scalars:
        invalid = invalid | jnp.isnan(value) | (value < 0.0)
    invalid = invalid | (field_lower > field_upper)
    checked_production = eqx.error_if(
        production,
        invalid,
        "action enclosure must contain ordered intervals and non-negative "
        "non-NaN bounds",
    )
    finite_certificate: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(real_lower))
        & jnp.all(jnp.isfinite(real_upper))
        & jnp.all(jnp.isfinite(imag_lower))
        & jnp.all(jnp.isfinite(imag_upper))
        & jnp.all(jnp.isfinite(production_direct_components))
        & jnp.all(jnp.isfinite(direct_algebraic_components))
        & jnp.all(jnp.isfinite(jnp.stack(scalars)))
        & environment_supported
    )
    enclosure: GalerkinTargetActionEnclosure = GalerkinTargetActionEnclosure(
        target=target,
        submitted_field=submitted,
        production_action=checked_production,
        independent_direct_action=direct,
        algebraic_action_real_lower_bounds=real_lower,
        algebraic_action_real_upper_bounds=real_upper,
        algebraic_action_imag_lower_bounds=imag_lower,
        algebraic_action_imag_upper_bounds=imag_upper,
        production_direct_component_error_bounds=(
            production_direct_components
        ),
        direct_algebraic_component_error_bounds=(direct_algebraic_components),
        production_direct_error_bound=production_direct_bound,
        direct_algebraic_action_error_bound=direct_algebraic_bound,
        action_error_bound=complete_bound,
        field_norm_lower_bound=field_lower,
        field_norm_upper_bound=field_upper,
        per_state_relative_action_error_bound=relative_bound,
        arithmetic_environment_supported=environment_supported,
        gradual_underflow_supported=gradual_supported,
        finite_certificate=finite_certificate,
        direction=GalerkinActionDirection(direction),
        route=GalerkinActionErrorRoute(route),
        exact_action_target=exact_action_target,
        coefficient_norm=coefficient_norm,
        error_scope=error_scope,
        arithmetic_environment=arithmetic_environment,
    )
    return enclosure


@jaxtyped(typechecker=beartype)
def create_galerkin_residual_error_enclosure(  # noqa: PLR0913
    target: GalerkinTargetManifest,
    submitted_field: Complex[Array, "..."],
    algebraic_source: Complex[Array, "..."],
    independent_direct_action: Complex[Array, "..."],
    formed_residual: Complex[Array, "..."],
    direct_algebraic_component_error_bounds: Float[Array, "..."],
    subtraction_component_error_bounds: Float[Array, "..."],
    direct_algebraic_action_error_bound: Float[Array, ""],
    subtraction_error_bound: Float[Array, ""],
    residual_evaluator_error_bound: Float[Array, ""],
    formed_residual_norm: Float[Array, ""],
    formed_residual_norm_lower_bound: Float[Array, ""],
    formed_residual_norm_upper_bound: Float[Array, ""],
    norm_evaluation_error_bound: Float[Array, ""],
    algebraic_residual_norm_upper_bound: Float[Array, ""],
    field_norm_lower_bound: Float[Array, ""],
    field_norm_upper_bound: Float[Array, ""],
    arithmetic_environment_supported: Bool[Array, ""],
    gradual_underflow_supported: Bool[Array, ""],
    *,
    direction: GalerkinActionDirection | str,
    route: GalerkinActionErrorRoute | str,
    residual_target: str,
    coefficient_norm: str,
    error_scope: str,
    arithmetic_environment: str,
) -> GalerkinResidualErrorEnclosure:
    """Create a structurally validated independent residual enclosure.

    :see: :class:`~.test_action_error_types.TestActionErrorTypes`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Frozen target bound to this result.
    submitted_field : Complex[Array, "..."]
        Exact stored binary submitted state.
    algebraic_source : Complex[Array, "..."]
        Exact stored binary algebraic right-hand side.
    independent_direct_action : Complex[Array, "..."]
        Fresh direct coefficient action.
    formed_residual : Complex[Array, "..."]
        Stored subtraction ``fl(S_alg - d_direct)``.
    direct_algebraic_component_error_bounds : Float[Array, "..."]
        Direct-action component errors.
    subtraction_component_error_bounds : Float[Array, "..."]
        Residual-subtraction component errors.
    direct_algebraic_action_error_bound : Float[Array, ""]
        Euclidean direct-action error.
    subtraction_error_bound : Float[Array, ""]
        Euclidean subtraction error.
    residual_evaluator_error_bound : Float[Array, ""]
        Complete S2.64 residual-vector error.
    formed_residual_norm : Float[Array, ""]
        Rounded scale-safe residual norm.
    formed_residual_norm_lower_bound : Float[Array, ""]
        Exact stored-residual norm lower endpoint.
    formed_residual_norm_upper_bound : Float[Array, ""]
        Exact stored-residual norm upper endpoint.
    norm_evaluation_error_bound : Float[Array, ""]
        Rounded norm-evaluation error.
    algebraic_residual_norm_upper_bound : Float[Array, ""]
        S2.64 same-``H_alg`` residual norm bound.
    field_norm_lower_bound : Float[Array, ""]
        Exact stored-field norm lower endpoint.
    field_norm_upper_bound : Float[Array, ""]
        Exact stored-field norm upper endpoint.
    arithmetic_environment_supported : Bool[Array, ""]
        Result of the runtime binary64 arithmetic-environment probe.
    gradual_underflow_supported : Bool[Array, ""]
        Result of the runtime gradual-underflow probe.
    direction : GalerkinActionDirection | str
        Forward or actual-adjoint residual.
    route : GalerkinActionErrorRoute | str
        Enclosure route.
    residual_target : str
        Nonempty same-target declaration.
    coefficient_norm : str
        Nonempty norm declaration.
    error_scope : str
        Nonempty error-scope declaration.
    arithmetic_environment : str
        Nonempty arithmetic-environment declaration.

    Returns
    -------
    enclosure : GalerkinResidualErrorEnclosure
        Validated independent residual evidence.

    Raises
    ------
    ValueError
        If static structure or metadata is invalid.
    equinox.EquinoxRuntimeError
        If a bound is NaN or negative.
    """
    submitted = _checked_vector(submitted_field, "submitted_field")
    algebraic_rhs = _checked_vector(algebraic_source, "algebraic_source")
    direct = _checked_vector(
        independent_direct_action, "independent_direct_action"
    )
    residual = _checked_vector(formed_residual, "formed_residual")
    _raise_if(
        residual.shape != direct.shape,
        "action and residual vectors must have matching shapes",
    )
    size: int = direct.shape[0]
    _raise_if(
        submitted.shape != direct.shape or algebraic_rhs.shape != direct.shape,
        "field, source, action, and residual vectors must have matching "
        "shapes",
    )
    _raise_if(
        target.support.state_indices.shape[0] != size,
        "target support must match the submitted residual size",
    )
    direct_components = _checked_float_vector(
        direct_algebraic_component_error_bounds,
        size,
        "direct_algebraic_component_error_bounds",
    )
    subtraction_components = _checked_float_vector(
        subtraction_component_error_bounds,
        size,
        "subtraction_component_error_bounds",
    )
    scalar_names = (
        "direct_algebraic_action_error_bound",
        "subtraction_error_bound",
        "residual_evaluator_error_bound",
        "formed_residual_norm",
        "formed_residual_norm_lower_bound",
        "formed_residual_norm_upper_bound",
        "norm_evaluation_error_bound",
        "algebraic_residual_norm_upper_bound",
        "field_norm_lower_bound",
        "field_norm_upper_bound",
    )
    scalars = tuple(
        _checked_scalar(value, name)
        for value, name in zip(
            (
                direct_algebraic_action_error_bound,
                subtraction_error_bound,
                residual_evaluator_error_bound,
                formed_residual_norm,
                formed_residual_norm_lower_bound,
                formed_residual_norm_upper_bound,
                norm_evaluation_error_bound,
                algebraic_residual_norm_upper_bound,
                field_norm_lower_bound,
                field_norm_upper_bound,
            ),
            scalar_names,
            strict=True,
        )
    )
    (
        direct_bound,
        subtraction_bound,
        evaluator_bound,
        formed_norm,
        norm_lower,
        norm_upper,
        norm_error,
        residual_upper,
        field_lower,
        field_upper,
    ) = scalars
    for metadata, name in (
        (residual_target, "residual_target"),
        (coefficient_norm, "coefficient_norm"),
        (error_scope, "error_scope"),
        (arithmetic_environment, "arithmetic_environment"),
    ):
        _raise_if(not metadata.strip(), f"{name} must be nonempty")
    environment_supported: Bool[Array, ""] = jnp.asarray(
        arithmetic_environment_supported,
        dtype=jnp.bool_,
    )
    _raise_if(
        environment_supported.shape != (),
        "arithmetic_environment_supported must be a scalar",
    )
    gradual_supported: Bool[Array, ""] = jnp.asarray(
        gradual_underflow_supported,
        dtype=jnp.bool_,
    )
    _raise_if(
        gradual_supported.shape != (),
        "gradual_underflow_supported must be a scalar",
    )

    invalid: Bool[Array, ""] = (
        jnp.any(~jnp.isfinite(submitted))
        | has_subnormal_components(submitted)
        | jnp.any(~jnp.isfinite(algebraic_rhs))
        | has_subnormal_components(algebraic_rhs)
        | jnp.any(~jnp.isfinite(direct))
        | jnp.any(~jnp.isfinite(residual))
        | jnp.any(jnp.isnan(direct_components) | (direct_components < 0.0))
        | jnp.any(
            jnp.isnan(subtraction_components) | (subtraction_components < 0.0)
        )
    )
    for value in scalars:
        invalid = invalid | jnp.isnan(value) | (value < 0.0)
    invalid = invalid | (norm_lower > norm_upper) | (field_lower > field_upper)
    checked_residual = eqx.error_if(
        residual,
        invalid,
        "residual enclosure must contain ordered non-negative non-NaN bounds",
    )
    finite_certificate: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(direct_components))
        & jnp.all(jnp.isfinite(subtraction_components))
        & jnp.all(jnp.isfinite(jnp.stack(scalars)))
        & environment_supported
    )
    enclosure: GalerkinResidualErrorEnclosure = GalerkinResidualErrorEnclosure(
        target=target,
        submitted_field=submitted,
        algebraic_source=algebraic_rhs,
        independent_direct_action=direct,
        formed_residual=checked_residual,
        direct_algebraic_component_error_bounds=direct_components,
        subtraction_component_error_bounds=subtraction_components,
        direct_algebraic_action_error_bound=direct_bound,
        subtraction_error_bound=subtraction_bound,
        residual_evaluator_error_bound=evaluator_bound,
        formed_residual_norm=formed_norm,
        formed_residual_norm_lower_bound=norm_lower,
        formed_residual_norm_upper_bound=norm_upper,
        norm_evaluation_error_bound=norm_error,
        algebraic_residual_norm_upper_bound=residual_upper,
        field_norm_lower_bound=field_lower,
        field_norm_upper_bound=field_upper,
        arithmetic_environment_supported=environment_supported,
        gradual_underflow_supported=gradual_supported,
        finite_certificate=finite_certificate,
        direction=GalerkinActionDirection(direction),
        route=GalerkinActionErrorRoute(route),
        residual_target=residual_target,
        coefficient_norm=coefficient_norm,
        error_scope=error_scope,
        arithmetic_environment=arithmetic_environment,
    )
    return enclosure


__all__: list[str] = [
    "GalerkinActionDirection",
    "GalerkinActionErrorRoute",
    "GalerkinResidualErrorEnclosure",
    "GalerkinTargetActionEnclosure",
    "create_galerkin_residual_error_enclosure",
    "create_galerkin_target_action_enclosure",
]
