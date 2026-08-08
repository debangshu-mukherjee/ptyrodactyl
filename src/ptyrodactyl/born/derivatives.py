r"""Differentiate one fixed-support scalar Galerkin root.

Extended Summary
----------------
This module exposes a bounded derivative seam for the carrier, interaction
coefficients, and finite source. The support, coefficient ordering, absorber,
box, wavenumber, and solver controls stay fixed. The JVP solves the implicit
tangent equation. The VJP uses the converged root's custom adjoint.

Routine Listings
----------------
:func:`galerkin_state_jvp`
    Evaluate a fixed-support Galerkin state and parameter JVP.
:func:`galerkin_state_vjp`
    Evaluate a fixed-support Galerkin state and parameter VJP.

Notes
-----
The fixed chart uses the stored support ordering and the real metric
``Re(sum(conj(x) * y))``. A support, work quotient, or endpoint change is a
different discrete model and is not a derivative accepted by this module.
The exposed array directions assign no optimizer-facing parameter metric.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from ptyrodactyl.types import (
    GalerkinSolveResult,
    GalerkinTargetManifest,
    scalar_float,
    scalar_int,
)

from .engine import (
    apply_galerkin_operator,
    cgls_solve,
    implicit_galerkin_solve,
    shifted_free_diagonal,
)


def _replace_differentiable_target_leaves(
    target: GalerkinTargetManifest,
    carrier: Float[Array, "d"],
    interaction_coefficients: Complex[Array, "p"],
) -> GalerkinTargetManifest:
    """Replace only the target leaves admitted by the bounded chart."""
    reciprocal_frequencies: Float[Array, "n d"] = (
        target.support.state_indices / target.box_lengths[None, :]
    )
    free_diagonal: Float[Array, "n"] = shifted_free_diagonal(
        reciprocal_frequencies,
        carrier,
        target.wavenumber,
    )
    voltage_coefficients: Complex[Array, "p"] = (
        interaction_coefficients / target.interaction_coupling
    )
    candidate: GalerkinTargetManifest = eqx.tree_at(
        lambda value: (
            value.carrier,
            value.free_diagonal,
            value.voltage_coefficients,
            value.interaction_coefficients,
        ),
        target,
        (
            carrier,
            free_diagonal,
            voltage_coefficients,
            interaction_coefficients,
        ),
    )
    return candidate


def _validate_derivative_shapes(
    target: GalerkinTargetManifest,
    source: Complex[Array, "n"],
    carrier_tangent: Float[Array, "d"],
    interaction_coefficients_tangent: Complex[Array, "p"],
) -> None:
    """Reject derivative arrays outside the fixed target chart."""
    if source.shape != target.free_diagonal.shape:
        raise ValueError("source must match the target state shape")
    if carrier_tangent.shape != target.carrier.shape:
        raise ValueError("carrier_tangent must match the carrier shape")
    if (
        interaction_coefficients_tangent.shape
        != target.interaction_coefficients.shape
    ):
        raise ValueError(
            "interaction_coefficients_tangent must match the fixed support"
        )


def _checked_interaction_tangent(
    target: GalerkinTargetManifest,
    tangent: Complex[Array, "p"],
) -> Complex[Array, "p"]:
    """Require a finite tangent in the real-interaction coefficient space."""
    indices: Int[Array, "p d"] = target.support.interaction_indices
    inverse_indices: Int[Array, "p d"] = -indices
    forward_order: Int[Array, " p"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    inverse_order: Int[Array, " p"] = jnp.lexsort(
        (
            inverse_indices[:, 2],
            inverse_indices[:, 1],
            inverse_indices[:, 0],
        )
    )
    nonhermitian: Bool[Array, ""] = jnp.any(
        indices[forward_order] != inverse_indices[inverse_order]
    ) | jnp.any(tangent[forward_order] != jnp.conj(tangent[inverse_order]))
    checked_tangent: Complex[Array, "p"] = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent)) | nonhermitian,
        "interaction_coefficients_tangent must be finite and Hermitian",
    )
    return checked_tangent


def _checked_carrier_tangent(
    target: GalerkinTargetManifest,
    tangent: Float[Array, "d"],
) -> Float[Array, "d"]:
    """Require a finite tangent to the fixed-energy carrier sphere."""
    tangent_norm: Float[Array, ""] = jnp.linalg.norm(tangent)
    tangent_tolerance: Float[Array, ""] = (
        128.0
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(1.0, target.wavenumber * tangent_norm)
    )
    checked_tangent: Float[Array, "d"] = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent))
        | (jnp.abs(jnp.vdot(target.carrier, tangent)) > tangent_tolerance),
        "carrier_tangent must be finite and tangent to the on-shell sphere",
    )
    return checked_tangent


@jaxtyped(typechecker=beartype)
def galerkin_state_jvp(
    target: GalerkinTargetManifest,
    source: Complex[Array, "n"],
    carrier_tangent: Float[Array, "d"],
    interaction_coefficients_tangent: Complex[Array, "p"],
    source_tangent: Complex[Array, "n"],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> tuple[Complex[Array, "n"], Complex[Array, "n"]]:
    """Evaluate a fixed-support Galerkin state and parameter JVP.

    :see: :class:`~.test_derivatives.TestGalerkinDerivatives`

    Implementation Logic
    --------------------
    1. Solve the finite target with the production custom-VJP primal.
    2. Differentiate its action at the converged state within the fixed chart.
    3. Solve ``H du = dS - dH u`` with the production CGLS path.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Fixed-support target. Its support, absorber, box, and metadata stay
        constant.
    source : Complex[Array, "n"]
        Finite source in the fixed coefficient ordering.
    carrier_tangent : Float[Array, "d"]
        Directional derivative tangent to the fixed-energy carrier sphere, in
        radians per Angstrom per declared parameter unit.
    interaction_coefficients_tangent : Complex[Array, "p"]
        Hermitian directional derivative of the SC.13b interaction
        coefficients, in inverse-square Angstroms per parameter unit.
    source_tangent : Complex[Array, "n"]
        Directional derivative of the finite source.
    max_iterations : scalar_int
        Positive CGLS iteration limit for both roots. Default is 100.
    relative_tolerance : scalar_float
        Non-negative relative algebraic residual tolerance. Default is
        ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute algebraic residual tolerance. Default is zero.

    Returns
    -------
    field : Complex[Array, "n"]
        Converged finite Galerkin state.
    field_tangent : Complex[Array, "n"]
        Implicit directional state derivative in the fixed chart.

    Raises
    ------
    ValueError
        If a derivative array leaves the fixed target shape.
    equinox.EquinoxRuntimeError
        If a tangent is non-finite, off-shell, or non-Hermitian, or a solve
        does not converge during traced execution.

    Notes
    -----
    This function varies the carrier and recomputes its shifted free
    diagonal. An interaction-coefficient direction is the equivalent
    fixed-voltage potential direction ``dphi = dchi / sigma_H``; both
    manifested leaves remain coupled. The function does not differentiate
    support selection, the absorber, detector parameters, geometry,
    coherence, or solver controls. Its algebraic tangent is not an
    RM-I2-certified inexact gradient. The source leaf is finite
    right-hand-side plumbing, not a full source-coordinate chart.
    """
    _validate_derivative_shapes(
        target,
        source,
        carrier_tangent,
        interaction_coefficients_tangent,
    )
    if source_tangent.shape != source.shape:
        raise ValueError("source_tangent must match the source shape")
    checked_carrier_tangent: Float[Array, "d"] = _checked_carrier_tangent(
        target,
        carrier_tangent,
    )
    checked_interaction_tangent: Complex[Array, "p"] = (
        _checked_interaction_tangent(
            target,
            interaction_coefficients_tangent,
        )
    )
    checked_source_tangent: Complex[Array, "n"] = eqx.error_if(
        source_tangent,
        jnp.any(~jnp.isfinite(source_tangent)),
        "source_tangent must be finite",
    )
    field: Complex[Array, "n"] = implicit_galerkin_solve(
        target,
        source,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    def fixed_state_action(
        carrier: Float[Array, "d"],
        interaction_coefficients: Complex[Array, "p"],
    ) -> Complex[Array, "n"]:
        """Apply varied admitted leaves to the fixed converged state."""
        candidate: GalerkinTargetManifest = (
            _replace_differentiable_target_leaves(
                target,
                carrier,
                interaction_coefficients,
            )
        )
        action: Complex[Array, "n"] = apply_galerkin_operator(
            candidate,
            field,
        )
        return action

    _, raw_action_tangent = jax.jvp(
        fixed_state_action,
        (target.carrier, target.interaction_coefficients),
        (checked_carrier_tangent, checked_interaction_tangent),
    )
    action_tangent: Complex[Array, "n"] = raw_action_tangent
    tangent_source: Complex[Array, "n"] = (
        checked_source_tangent - action_tangent
    )
    tangent_result: GalerkinSolveResult = cgls_solve(
        target,
        tangent_source,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    field_tangent: Complex[Array, "n"] = eqx.error_if(
        tangent_result.field,
        ~tangent_result.converged,
        "Galerkin tangent solve did not converge",
    )
    result: tuple[Complex[Array, "n"], Complex[Array, "n"]] = (
        field,
        field_tangent,
    )
    return result


@jaxtyped(typechecker=beartype)
def galerkin_state_vjp(
    target: GalerkinTargetManifest,
    source: Complex[Array, "n"],
    output_cotangent: Complex[Array, "n"],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> tuple[
    Complex[Array, "n"],
    Float[Array, "d"],
    Complex[Array, "p"],
    Complex[Array, "n"],
]:
    """Evaluate a fixed-support Galerkin state and parameter VJP.

    :see: :class:`~.test_derivatives.TestGalerkinDerivatives`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Fixed-support target whose absorber, box, and metadata stay constant.
    source : Complex[Array, "n"]
        Finite source in the fixed coefficient ordering.
    output_cotangent : Complex[Array, "n"]
        State cotangent under JAX's real-linear complex convention.
    max_iterations : scalar_int
        Positive CGLS iteration limit for primal and adjoint roots. Default is
        100.
    relative_tolerance : scalar_float
        Non-negative relative algebraic residual tolerance. Default is
        ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute algebraic residual tolerance. Default is zero.

    Returns
    -------
    field : Complex[Array, "n"]
        Converged finite Galerkin state.
    carrier_cotangent : Float[Array, "d"]
        Cotangent for the real carrier in radians per Angstrom.
    interaction_coefficients_cotangent : Complex[Array, "p"]
        Ambient cotangent for the fixed interaction-coefficient ordering.
    source_cotangent : Complex[Array, "n"]
        Cotangent for the finite source.

    Raises
    ------
    ValueError
        If the source or output cotangent leaves the fixed state shape.
    equinox.EquinoxRuntimeError
        If the output cotangent is non-finite or a solve does not converge
        during traced execution.

    Notes
    -----
    The carrier and interaction cotangents are ambient. Physical on-shell and
    Hermitian parameterizations must pull them back through their declared
    real charts. This function activates no support, absorber, detector,
    geometry, coherence, nuisance, or inverse parameter block.
    """
    zero_carrier_tangent: Float[Array, "d"] = jnp.zeros_like(target.carrier)
    zero_interaction_tangent: Complex[Array, "p"] = jnp.zeros_like(
        target.interaction_coefficients
    )
    _validate_derivative_shapes(
        target,
        source,
        zero_carrier_tangent,
        zero_interaction_tangent,
    )
    if output_cotangent.shape != source.shape:
        raise ValueError("output_cotangent must match the source shape")
    checked_output_cotangent: Complex[Array, "n"] = eqx.error_if(
        output_cotangent,
        jnp.any(~jnp.isfinite(output_cotangent)),
        "output_cotangent must be finite",
    )

    def dynamic_root(
        carrier: Float[Array, "d"],
        interaction_coefficients: Complex[Array, "p"],
        candidate_source: Complex[Array, "n"],
    ) -> Complex[Array, "n"]:
        """Solve after varying only the admitted dynamic leaves."""
        candidate: GalerkinTargetManifest = (
            _replace_differentiable_target_leaves(
                target,
                carrier,
                interaction_coefficients,
            )
        )
        candidate_field: Complex[Array, "n"] = implicit_galerkin_solve(
            candidate,
            candidate_source,
            max_iterations,
            relative_tolerance,
            absolute_tolerance,
        )
        return candidate_field

    raw_field, pullback = jax.vjp(
        dynamic_root,
        target.carrier,
        target.interaction_coefficients,
        source,
    )
    field: Complex[Array, "n"] = raw_field
    raw_cotangents = pullback(checked_output_cotangent)
    carrier_cotangent: Float[Array, "d"] = raw_cotangents[0]
    interaction_coefficients_cotangent: Complex[Array, "p"] = raw_cotangents[1]
    source_cotangent: Complex[Array, "n"] = raw_cotangents[2]
    result: tuple[
        Complex[Array, "n"],
        Float[Array, "d"],
        Complex[Array, "p"],
        Complex[Array, "n"],
    ] = (
        field,
        carrier_cotangent,
        interaction_coefficients_cotangent,
        source_cotangent,
    )
    return result


__all__: list[str] = [
    "galerkin_state_jvp",
    "galerkin_state_vjp",
]
