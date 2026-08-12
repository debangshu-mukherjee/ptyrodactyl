r"""Differentiate the rounded selected-sector terminal number current.

Extended Summary
----------------
This module composes the physical fixed-support Galerkin state derivative
with the bounded coordinate-terminal current operator.  Its only observable
is the rounded acquisition-selected ``K_d``-fiber number current at ``xi=0``,

``j_hat(u, k) = C_j_hat Re(<u, F_hat(k) u>)``.

The chart keeps the support, box, terminal axis and side, selected fibers,
absorber, accelerating voltage, geometry, and evidence fixed.  The carrier
varies only on its fixed-radius sphere.  The scalar data metric is Euclidean;
the reduced current has units of inverse square Angstroms, ``C_j`` has units
of square Angstroms per second, and the returned number current has units of
inverse seconds.

Routine Listings
----------------
:func:`galerkin_terminal_number_current_jvp`
    Evaluate the rounded selected-sector terminal number-current JVP.
:func:`galerkin_terminal_number_current_vjp`
    Evaluate the rounded selected-sector terminal number-current VJP.

Notes
-----
The public operator certificate is storage rather than proof by possession.
Both entry points therefore replay its canonical host-side producer before
using it.  Like the owning terminal action-enclosure boundary, these wrappers
are deliberately not JIT entry points.  The authenticated inner chart is a
plain differentiable JAX calculation.

This chart supplies neither a full-plane identity nor a vacuum branch,
outgoing extraction, detector response, likelihood, or inexact-gradient
certificate.  In particular, it does not compose the per-call frozen-action
error, the uniform frozen-to-exact operator error, state error, or the
``C_j`` normalization interval into an exact-current or derivative
enclosure.  Certificate authentication qualifies the fixed operator route;
it does not certify the returned rounded current, JVP, or VJP as exact-target
quantities.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from ptyrodactyl.types import (
    GalerkinCurrentOperatorCertificate,
    GalerkinTargetManifest,
    GalerkinTerminalSide,
    scalar_float,
    scalar_int,
)

from .derivatives import galerkin_state_jvp, galerkin_state_vjp
from .engine import implicit_galerkin_solve
from .terminal import (
    apply_galerkin_terminal_current,
    apply_galerkin_terminal_trace,
    apply_galerkin_terminal_trace_adjoint,
    certify_galerkin_terminal_current_operator,
)


def _authenticated_operator_certificate(
    certificate: GalerkinCurrentOperatorCertificate,
) -> GalerkinCurrentOperatorCertificate:
    """PRIVATE: Canonically replay one public current-operator carrier.

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Public carrier whose complete payload must match canonical replay.

    Returns
    -------
    authenticated : GalerkinCurrentOperatorCertificate
        Canonically reproduced eligible operator certificate.
    """
    canonical: GalerkinCurrentOperatorCertificate = (
        certify_galerkin_terminal_current_operator(certificate.diagnostic)
    )
    same_payload = jnp.asarray(
        eqx.tree_equal(certificate, canonical, typematch=True)
    )
    checked_scale: Float64[Array, ""] = eqx.error_if(
        canonical.number_current_scale,
        ~same_payload,
        "terminal-current derivative certificate failed canonical replay",
    )
    checked_scale = eqx.error_if(
        checked_scale,
        ~canonical.current_operator_eligible,
        "terminal-current derivative requires an eligible current operator",
    )
    authenticated: GalerkinCurrentOperatorCertificate = eqx.tree_at(
        lambda record: record.number_current_scale,
        canonical,
        checked_scale,
    )
    return authenticated


def _local_reduced_current(
    certificate: GalerkinCurrentOperatorCertificate,
    field: Complex128[Array, " n"],
    carrier: Float64[Array, " d"],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the anchored rounded ``Re(<u,F(k)u>)`` chart.

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Authenticated fixed-geometry operator certificate.
    field : Complex128[Array, " n"]
        Canonical retained-state field.
    carrier : Float64[Array, " d"]
        On-shell carrier at which to evaluate the anchored chart.

    Returns
    -------
    reduced_current : Float64[Array, ""]
        Rounded selected-sector reduced current in inverse square Angstroms.

    Notes
    -----
    For fixed trace geometry, changing only the carrier gives

    ``F(k) = F(k0) + s (k_n-k0_n) T* T``.

    This identity retains every same-fiber cross term and makes the direct
    carrier dependence explicit without mutating the canonical target.
    """
    target: GalerkinTargetManifest = certificate.diagnostic.target
    axis: int = target.acquisition.terminal_axis
    side_sign: float = (
        1.0
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else -1.0
    )
    base_carrier: Float64[Array, " d"] = jax.lax.stop_gradient(target.carrier)
    base_action: Complex128[Array, " n"] = apply_galerkin_terminal_current(
        target, field
    )
    trace: Complex128[Array, " t"] = apply_galerkin_terminal_trace(
        target, field
    )
    trace_gram_action: Complex128[Array, " n"] = (
        apply_galerkin_terminal_trace_adjoint(target, trace)
    )
    carrier_delta: Float64[Array, ""] = carrier[axis] - base_carrier[axis]
    action: Complex128[Array, " n"] = base_action + (
        side_sign * carrier_delta * trace_gram_action
    )
    reduced_current: Float64[Array, ""] = jnp.real(jnp.vdot(field, action))
    return reduced_current


def _local_number_current(
    certificate: GalerkinCurrentOperatorCertificate,
    field: Complex128[Array, " n"],
    carrier: Float64[Array, " d"],
) -> Float64[Array, ""]:
    """PRIVATE: Scale the rounded reduced current by stored ``C_j``.

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Authenticated certificate containing the fixed rounded scale.
    field : Complex128[Array, " n"]
        Canonical retained-state field.
    carrier : Float64[Array, " d"]
        On-shell carrier at which to evaluate the anchored chart.

    Returns
    -------
    number_current : Float64[Array, ""]
        Rounded selected-sector number current in inverse seconds.
    """
    scale: Float64[Array, ""] = jax.lax.stop_gradient(
        certificate.number_current_scale
    )
    number_current: Float64[Array, ""] = scale * _local_reduced_current(
        certificate, field, carrier
    )
    return number_current


def _project_carrier_cotangent(
    target: GalerkinTargetManifest,
    ambient_cotangent: Float64[Array, " d"],
) -> Float64[Array, " d"]:
    """PRIVATE: Select the Euclidean Riesz representative on the sphere.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Fixed target supplying the carrier sphere.
    ambient_cotangent : Float64[Array, " d"]
        Finite ambient Euclidean carrier cotangent.

    Returns
    -------
    checked : Float64[Array, " d"]
        Finite carrier cotangent projected onto the tangent plane.
    """
    carrier: Float64[Array, " d"] = target.carrier
    radial_weight: Float64[Array, ""] = jnp.vdot(
        carrier, ambient_cotangent
    ) / jnp.vdot(carrier, carrier)
    projected: Float64[Array, " d"] = (
        ambient_cotangent - radial_weight * carrier
    )
    checked: Float64[Array, " d"] = eqx.error_if(
        projected,
        jnp.any(~jnp.isfinite(projected)),
        "terminal-current carrier cotangent must be finite",
    )
    return checked


def _checked_data_cotangent(
    value: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Check one scalar Euclidean number-current cotangent.

    Parameters
    ----------
    value : Float64[Array, ""]
        Scalar Euclidean number-current cotangent.

    Returns
    -------
    checked : Float64[Array, ""]
        Finite scalar data cotangent.
    """
    checked: Float64[Array, ""] = eqx.error_if(
        value,
        ~jnp.isfinite(value),
        "number_current_cotangent must be finite",
    )
    return checked


@jaxtyped(typechecker=beartype)
def galerkin_terminal_number_current_jvp(
    certificate: GalerkinCurrentOperatorCertificate,
    source: Complex128[Array, "n"],
    potential_volume_tangent: Float64[Array, "nz ny nx"],
    carrier_tangent: Float64[Array, "d"],
    source_tangent: Complex128[Array, "n"],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> Tuple[
    Complex128[Array, "n"],
    Float64[Array, ""],
    Float64[Array, ""],
]:
    """Evaluate the rounded selected-sector terminal number-current JVP.

    :see: :class:`~.test_terminal_derivatives.\
TestTerminalNumberCurrentDerivatives`

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Canonical eligible selected-``K_d`` operator certificate.  Its
        complete payload is host-replayed before use.
    source : Complex128[Array, "n"]
        Finite source in the canonical retained-state ordering.
    potential_volume_tangent : Float64[Array, "nz ny nx"]
        Real voxel-potential tangent in volts per parameter unit.
    carrier_tangent : Float64[Array, "d"]
        On-shell carrier tangent in radians per Angstrom per parameter unit.
    source_tangent : Complex128[Array, "n"]
        Finite-source tangent under the realified complex metric.
    max_iterations : scalar_int
        Positive CGLS iteration limit for state and tangent roots.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.

    Returns
    -------
    field : Complex128[Array, "n"]
        Converged canonical finite state used to evaluate the returned
        rounded current.
    number_current : Float64[Array, ""]
        Rounded, physically normalized selected-sector current in inverse
        seconds.
    number_current_tangent : Float64[Array, ""]
        Rounded implemented-model directional derivative in inverse seconds
        per parameter unit.

    Raises
    ------
    ValueError
        If an array leaves the fixed chart or canonical replay fails.
    equinox.EquinoxRuntimeError
        If the certificate is forged or ineligible, a tangent is invalid,
        or a solve fails.

    Notes
    -----
    Writing ``j_hat=C_j_hat Re(<u,F_hat(k)u>)`` for the anchored rounded
    callable, the evaluated differential is

    ``dj_hat=C_j_hat Re(<du,F_hat u>+<u,F_hat du>+<u,dF_hat[dk]u>)``.

    The last term is the direct terminal dependence and is not part of the
    implicit state JVP.  ``C_j_hat`` and all geometry/evidence leaves are
    fixed.  No exact-current or derivative enclosure is returned.
    """
    authenticated: GalerkinCurrentOperatorCertificate = (
        _authenticated_operator_certificate(certificate)
    )
    target: GalerkinTargetManifest = authenticated.diagnostic.target
    field: Complex128[Array, "n"]
    field_tangent: Complex128[Array, "n"]
    field, field_tangent = galerkin_state_jvp(
        target,
        source,
        potential_volume_tangent,
        carrier_tangent,
        source_tangent,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    def current_chart(
        candidate_field: Complex128[Array, "n"],
        candidate_carrier: Float64[Array, "d"],
    ) -> Float64[Array, ""]:
        """Evaluate the authenticated inner terminal chart."""
        candidate_current: Float64[Array, ""] = _local_number_current(
            authenticated, candidate_field, candidate_carrier
        )
        return candidate_current

    number_current: Float64[Array, ""]
    number_current_tangent: Float64[Array, ""]
    number_current, number_current_tangent = jax.jvp(
        current_chart,
        (field, target.carrier),
        (field_tangent, carrier_tangent),
    )
    number_current = eqx.error_if(
        number_current,
        ~jnp.isfinite(number_current) | ~jnp.isfinite(number_current_tangent),
        "terminal number-current JVP must be finite",
    )
    result: Tuple[
        Complex128[Array, "n"],
        Float64[Array, ""],
        Float64[Array, ""],
    ] = (field, number_current, number_current_tangent)
    return result


@jaxtyped(typechecker=beartype)
def galerkin_terminal_number_current_vjp(
    certificate: GalerkinCurrentOperatorCertificate,
    source: Complex128[Array, "n"],
    number_current_cotangent: Float64[Array, ""],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> Tuple[
    Complex128[Array, "n"],
    Float64[Array, ""],
    Float64[Array, "nz ny nx"],
    Float64[Array, "d"],
    Complex128[Array, "n"],
]:
    """Evaluate the rounded selected-sector terminal number-current VJP.

    :see: :class:`~.test_terminal_derivatives.\
TestTerminalNumberCurrentDerivatives`

    Parameters
    ----------
    certificate : GalerkinCurrentOperatorCertificate
        Canonical eligible selected-``K_d`` operator certificate.  Its
        complete payload is host-replayed before use.
    source : Complex128[Array, "n"]
        Finite source in the canonical retained-state ordering.
    number_current_cotangent : Float64[Array, ""]
        Euclidean scalar-data cotangent for the number current.
    max_iterations : scalar_int
        Positive CGLS iteration limit for primal and adjoint roots.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance.

    Returns
    -------
    field : Complex128[Array, "n"]
        Converged canonical finite state used to evaluate the returned
        rounded current.
    number_current : Float64[Array, ""]
        Rounded, physically normalized selected-sector current in inverse
        seconds.
    potential_volume_metric_cotangent : Float64[Array, "nz ny nx"]
        Voxel-potential Riesz gradient under
        ``DeltaV * sum(g_volume * delta_volume)``.
    carrier_tangent_cotangent : Float64[Array, "d"]
        Euclidean tangent-plane carrier Riesz representative.  It includes
        both the implicit-state dependence and the direct ``F(k)`` term.
    source_cotangent : Complex128[Array, "n"]
        Finite-source cotangent paired as
        ``Re(sum(g_source * delta_source))``.

    Raises
    ------
    ValueError
        If an array leaves the fixed chart or canonical replay fails.
    equinox.EquinoxRuntimeError
        If the certificate is forged or ineligible, a value is nonfinite,
        or a primal/adjoint solve fails.

    Notes
    -----
    The inner realified pullback supplies the state covector in the declared
    complex convention expected by :func:`galerkin_state_vjp`.  The latter
    returns the implicit volume/carrier/source blocks.  Because that public
    state-VJP seam accepts no prepared field, this composition must replay
    the same canonical primal after the prerequisite root used to construct
    the terminal state covector.  The two rounded fields are required to be
    exactly elementwise identical, and the first field remains bound to the
    returned current.  This is duplicate primal work, not differentiation
    through a solver trajectory.

    The implicit carrier block is combined once with the independently
    pulled-back direct terminal carrier block, and the sum is projected at
    the public return boundary onto the fixed-radius carrier tangent plane.
    This implemented-model VJP carries no exact-current, per-call derivative,
    or inexact-gradient enclosure.
    """
    authenticated: GalerkinCurrentOperatorCertificate = (
        _authenticated_operator_certificate(certificate)
    )
    target: GalerkinTargetManifest = authenticated.diagnostic.target
    checked_data_cotangent: Float64[Array, ""] = _checked_data_cotangent(
        number_current_cotangent
    )
    field: Complex128[Array, "n"] = implicit_galerkin_solve(
        target,
        source,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    def current_chart(
        candidate_field: Complex128[Array, "n"],
        candidate_carrier: Float64[Array, "d"],
    ) -> Float64[Array, ""]:
        """Evaluate the authenticated inner terminal chart."""
        candidate_current: Float64[Array, ""] = _local_number_current(
            authenticated, candidate_field, candidate_carrier
        )
        return candidate_current

    number_current: Float64[Array, ""]
    number_current, pullback = jax.vjp(
        current_chart,
        field,
        target.carrier,
    )
    state_cotangent: Complex128[Array, "n"]
    ambient_direct_carrier_cotangent: Float64[Array, "d"]
    state_cotangent, ambient_direct_carrier_cotangent = pullback(
        checked_data_cotangent
    )
    replayed_field: Complex128[Array, "n"]
    potential_volume_metric_cotangent: Float64[Array, "nz ny nx"]
    implicit_carrier_cotangent: Float64[Array, "d"]
    source_cotangent: Complex128[Array, "n"]
    (
        replayed_field,
        potential_volume_metric_cotangent,
        implicit_carrier_cotangent,
        source_cotangent,
    ) = galerkin_state_vjp(
        target,
        source,
        state_cotangent,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )
    field = eqx.error_if(
        field,
        ~jnp.array_equal(field, replayed_field),
        "terminal-current VJP primal replay must be elementwise identical",
    )
    carrier_tangent_cotangent: Float64[Array, "d"] = (
        _project_carrier_cotangent(
            target,
            implicit_carrier_cotangent + ambient_direct_carrier_cotangent,
        )
    )
    number_current = eqx.error_if(
        number_current,
        ~jnp.isfinite(number_current),
        "terminal number current must be finite",
    )
    result: Tuple[
        Complex128[Array, "n"],
        Float64[Array, ""],
        Float64[Array, "nz ny nx"],
        Float64[Array, "d"],
        Complex128[Array, "n"],
    ] = (
        field,
        number_current,
        potential_volume_metric_cotangent,
        carrier_tangent_cotangent,
        source_cotangent,
    )
    return result


__all__: list[str] = [
    "galerkin_terminal_number_current_jvp",
    "galerkin_terminal_number_current_vjp",
]
