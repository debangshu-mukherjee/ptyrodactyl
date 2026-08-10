r"""Differentiate one canonical fixed-support scalar Galerkin root.

Extended Summary
----------------
This module exposes the production RM-I1 seam for a real voxel potential, an
on-shell carrier tangent, and a finite-source tangent.  A canonical
:class:`~ptyrodactyl.types.GalerkinTargetManifest` supplies the immutable
support, evidence, geometry, absorber, coupling, and base algebraic root.  A
private local action recomputes the VC-1 coefficient map, interaction, and
shifted free diagonal without constructing or modifying a manifest.

Routine Listings
----------------
:func:`galerkin_state_jvp`
    Evaluate the physical fixed-support Galerkin state JVP.
:func:`galerkin_state_vjp`
    Evaluate the physical fixed-support Galerkin state VJP.

Notes
-----
The state and source use JAX's real-linear complex convention.  The returned
voxel cotangent instead uses the physical cell-volume-weighted real metric.
Support, work quotient, box, origin, voltage, absorber, and every evidence
leaf remain fixed.  A support or evidence change is a different model, not a
derivative admitted here.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from ptyrodactyl.types import (
    GalerkinSolveResult,
    GalerkinTargetManifest,
    scalar_float,
    scalar_int,
)

from .engine import cgls_solve, implicit_galerkin_solve, shifted_free_diagonal
from .potential import apply_absorber_action, apply_interaction_product
from .realization import _vc1_voltage_coefficients_from_full_grid


def _vc1_voltage_coefficients(
    target: GalerkinTargetManifest,
    volume: Float64[Array, "nz ny nx"],
) -> Complex128[Array, " p"]:
    """PRIVATE: Evaluate VC-1 on the target's fixed interaction support.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical base target supplying fixed support and geometry.
    volume : Float64[Array, "nz ny nx"]
        Candidate real voxel values in volts.

    Returns
    -------
    coefficients : Complex128[Array, " p"]
        Mean-normalized, origin-shifted, stored-Hermitian voltage
        coefficients in the canonical interaction ordering.
    """
    sample_count: int = volume.size
    full_coefficients: Complex128[Array, "nz ny nx"] = (
        jnp.fft.fftn(volume) / sample_count
    )
    box_size: Float64[Array, " 3"] = jnp.asarray(
        target.potential.box_size,
        dtype=jnp.float64,
    )
    origin: Float64[Array, " 3"] = jnp.asarray(
        target.potential.origin,
        dtype=jnp.float64,
    )
    coefficients: Complex128[Array, " p"] = (
        _vc1_voltage_coefficients_from_full_grid(
            full_coefficients,
            target.support.interaction_indices,
            box_size,
            origin,
        )
    )
    return coefficients


def _fixed_support_local_action(
    target: GalerkinTargetManifest,
    volume: Float64[Array, "nz ny nx"],
    carrier: Float64[Array, "d"],
    field: Complex128[Array, "n"],
) -> Complex128[Array, "n"]:
    """PRIVATE: Apply the physical local chart without manifest mutation.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical base target.  Its evidence and discrete structure are
        immutable chart data.
    volume : Float64[Array, "nz ny nx"]
        Candidate voxel values in volts.
    carrier : Float64[Array, "d"]
        Candidate carrier in radians per Angstrom.
    field : Complex128[Array, "n"]
        Fixed state at which to evaluate the action.

    Returns
    -------
    action : Complex128[Array, "n"]
        Local SC-1 action with recomputed VC-1, interaction, and free terms.

    Notes
    -----
    The chart is anchored at the stored binary64 realization:
    ``C(v) = C0 + VC1(v) - VC1(v0)`` and
    ``D(k) = D0 + Draw(k) - Draw(k0)``.  Therefore its origin is exactly the
    manifested algebraic target even when an eager and compiled FFT choose
    adjacent rounded values, while its derivative is exactly the derivative
    of the full VC-1 and shifted-free maps.  No certificate or manifest leaf
    is replaced.
    """
    base_volume: Float64[Array, "nz ny nx"] = jax.lax.stop_gradient(
        target.potential.volume
    )
    base_carrier: Float64[Array, "d"] = jax.lax.stop_gradient(target.carrier)
    raw_voltage: Complex128[Array, " p"] = _vc1_voltage_coefficients(
        target,
        volume,
    )
    raw_base_voltage: Complex128[Array, " p"] = _vc1_voltage_coefficients(
        target,
        base_volume,
    )
    voltage_delta: Complex128[Array, " p"] = raw_voltage - raw_base_voltage
    interaction_coefficients: Complex128[Array, " p"] = (
        target.interaction_coefficients
        + target.interaction_coupling * voltage_delta
    )

    reciprocal_frequencies: Float64[Array, "n d"] = (
        target.support.state_indices / target.box_lengths[None, :]
    )
    raw_free_diagonal: Float64[Array, "n"] = shifted_free_diagonal(
        reciprocal_frequencies,
        carrier,
        target.wavenumber,
    )
    raw_base_free_diagonal: Float64[Array, "n"] = shifted_free_diagonal(
        reciprocal_frequencies,
        base_carrier,
        target.wavenumber,
    )
    free_diagonal: Float64[Array, "n"] = target.free_diagonal + (
        raw_free_diagonal - raw_base_free_diagonal
    )
    interaction: Complex128[Array, "n"] = apply_interaction_product(
        target.support,
        interaction_coefficients,
        field,
    )
    absorber: Complex128[Array, "n"] = apply_absorber_action(
        target.support,
        target.absorber_coefficients,
        field,
    )
    action: Complex128[Array, "n"] = (
        free_diagonal * field
        - interaction
        - 1.0j * target.cap_scale * absorber
    )
    return action


def _validate_jvp_shapes(
    target: GalerkinTargetManifest,
    source: Complex128[Array, "n"],
    volume_tangent: Float64[Array, "nz ny nx"],
    carrier_tangent: Float64[Array, "d"],
    source_tangent: Complex128[Array, "n"],
) -> None:
    """PRIVATE: Reject arrays outside the canonical fixed chart.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target supplying every fixed chart shape.
    source : Complex128[Array, "n"]
        Candidate finite source.
    volume_tangent : Float64[Array, "nz ny nx"]
        Candidate real voxel-volume tangent.
    carrier_tangent : Float64[Array, "d"]
        Candidate carrier tangent.
    source_tangent : Complex128[Array, "n"]
        Candidate finite-source tangent.

    Raises
    ------
    ValueError
        If a candidate array differs from its canonical chart shape.
    """
    if source.shape != target.free_diagonal.shape:
        raise ValueError("source must match the target state shape")
    if volume_tangent.shape != target.potential.volume.shape:
        raise ValueError(
            "potential_volume_tangent must match Potential3D.volume"
        )
    if carrier_tangent.shape != target.carrier.shape:
        raise ValueError("carrier_tangent must match the carrier shape")
    if source_tangent.shape != source.shape:
        raise ValueError("source_tangent must match the source shape")


def _checked_carrier_tangent(
    target: GalerkinTargetManifest,
    tangent: Float64[Array, "d"],
) -> Float64[Array, "d"]:
    """PRIVATE: Require a finite tangent to the on-shell carrier sphere.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target supplying the carrier and wavenumber scale.
    tangent : Float64[Array, "d"]
        Candidate carrier tangent in radians per Angstrom.

    Returns
    -------
    checked_tangent : Float64[Array, "d"]
        Tangent with traced finiteness and sphere-orthogonality checks.
    """
    tangent_norm: Float64[Array, ""] = jnp.linalg.norm(tangent)
    tangent_tolerance: Float64[Array, ""] = (
        128.0
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(1.0, target.wavenumber * tangent_norm)
    )
    checked_tangent: Float64[Array, "d"] = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent))
        | (jnp.abs(jnp.vdot(target.carrier, tangent)) > tangent_tolerance),
        "carrier_tangent must be finite and tangent to the on-shell sphere",
    )
    return checked_tangent


def _checked_finite(
    values: Float64[Array, "..."] | Complex128[Array, "..."],
    message: str,
) -> Float64[Array, "..."] | Complex128[Array, "..."]:
    """PRIVATE: Attach one traced finite-value check.

    Parameters
    ----------
    values : Float64[Array, "..."] | Complex128[Array, "..."]
        Candidate real or complex binary64 values.
    message : str
        Error text used when a non-finite component is found.

    Returns
    -------
    checked : Float64[Array, "..."] | Complex128[Array, "..."]
        Values with the traced finite predicate attached.
    """
    checked: Float64[Array, "..."] | Complex128[Array, "..."] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        message,
    )
    return checked


def _physical_voxel_cotangent(
    target: GalerkinTargetManifest,
    raw_euclidean_cotangent: Float64[Array, "nz ny nx"],
) -> Float64[Array, "nz ny nx"]:
    """PRIVATE: Convert a JAX raw cotangent to the physical voxel metric.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target supplying exact box volume and voxel count.
    raw_euclidean_cotangent : Float64[Array, "nz ny nx"]
        Raw JAX Euclidean cotangent for the real volume array.

    Returns
    -------
    physical_cotangent : Float64[Array, "nz ny nx"]
        Riesz gradient under the cell-volume-weighted real voxel metric.

    Notes
    -----
    The physical metric is ``DeltaV * sum(a * b)``.  Consequently the raw JAX
    Euclidean cotangent is ``DeltaV`` times the returned Riesz gradient.
    """
    box_volume: Float64[Array, ""] = jnp.asarray(
        math.prod(target.potential.box_size),
        dtype=jnp.float64,
    )
    voxel_volume: Float64[Array, ""] = (
        box_volume / target.potential.volume.size
    )
    physical: Float64[Array, "nz ny nx"] = (
        raw_euclidean_cotangent / voxel_volume
    )
    physical_cotangent: Float64[Array, "nz ny nx"] = eqx.error_if(
        physical,
        jnp.any(~jnp.isfinite(physical)),
        "physical voxel-metric cotangent must be finite",
    )
    return physical_cotangent


def _on_shell_carrier_cotangent(
    target: GalerkinTargetManifest,
    ambient_cotangent: Float64[Array, "d"],
) -> Float64[Array, "d"]:
    """PRIVATE: Select the Euclidean Riesz representative on the sphere.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical target supplying the nonzero on-shell carrier.
    ambient_cotangent : Float64[Array, "d"]
        Ambient Euclidean carrier cotangent from JAX.

    Returns
    -------
    carrier_cotangent : Float64[Array, "d"]
        Orthogonal tangent-plane representative with identical admitted
        tangent pairings.
    """
    carrier: Float64[Array, "d"] = target.carrier
    radial_weight: Float64[Array, ""] = jnp.vdot(
        carrier, ambient_cotangent
    ) / jnp.vdot(carrier, carrier)
    tangent_cotangent: Float64[Array, "d"] = (
        ambient_cotangent - radial_weight * carrier
    )
    carrier_cotangent: Float64[Array, "d"] = eqx.error_if(
        tangent_cotangent,
        jnp.any(~jnp.isfinite(tangent_cotangent)),
        "on-shell carrier cotangent must be finite",
    )
    return carrier_cotangent


@jaxtyped(typechecker=beartype)
def galerkin_state_jvp(
    target: GalerkinTargetManifest,
    source: Complex128[Array, "n"],
    potential_volume_tangent: Float64[Array, "nz ny nx"],
    carrier_tangent: Float64[Array, "d"],
    source_tangent: Complex128[Array, "n"],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> Tuple[Complex128[Array, "n"], Complex128[Array, "n"]]:
    """Evaluate the physical fixed-support Galerkin state JVP.

    :see: :class:`~.test_derivatives.TestGalerkinDerivatives`

    Implementation Logic
    --------------------
    1. Solve the immutable canonical base target.
    2. Differentiate a local action that recomputes VC-1, interaction, and the
       free diagonal from the real voxel and on-shell carrier chart.
    3. Solve ``H du = dS - dH u`` on the unchanged base target.

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical fixed-support base target and immutable evidence.
    source : Complex128[Array, "n"]
        Finite source in the target state ordering.
    potential_volume_tangent : Float64[Array, "nz ny nx"]
        Real tangent to the bound :class:`Potential3D.volume`, in volts per
        declared parameter unit.
    carrier_tangent : Float64[Array, "d"]
        Tangent to the fixed-energy carrier sphere, in radians per Angstrom
        per declared parameter unit.
    source_tangent : Complex128[Array, "n"]
        Directional derivative of the finite source.
    max_iterations : scalar_int
        Positive CGLS iteration limit for both roots. Default is 100.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance. Default is ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance. Default is zero.

    Returns
    -------
    field : Complex128[Array, "n"]
        Converged finite Galerkin state.
    field_tangent : Complex128[Array, "n"]
        Implicit directional derivative in the physical local chart.

    Raises
    ------
    ValueError
        If an array leaves the fixed chart shape.
    equinox.EquinoxRuntimeError
        If a tangent is non-finite or off-shell, or either solve fails.

    Notes
    -----
    The function differentiates no support selection, evidence, absorber,
    geometry, accelerating voltage, detector, or solver control.  Its
    algebraic tangent remains subject to the separate RM-I2 inexact-gradient
    contract.
    """
    _validate_jvp_shapes(
        target,
        source,
        potential_volume_tangent,
        carrier_tangent,
        source_tangent,
    )
    checked_volume_tangent = _checked_finite(
        potential_volume_tangent,
        "potential_volume_tangent must be finite",
    )
    checked_carrier_tangent: Float64[Array, "d"] = _checked_carrier_tangent(
        target,
        carrier_tangent,
    )
    checked_source_tangent = _checked_finite(
        source_tangent,
        "source_tangent must be finite",
    )
    field: Complex128[Array, "n"] = implicit_galerkin_solve(
        target,
        source,
        max_iterations,
        relative_tolerance,
        absolute_tolerance,
    )

    def fixed_state_action(
        volume: Float64[Array, "nz ny nx"],
        carrier: Float64[Array, "d"],
    ) -> Complex128[Array, "n"]:
        """Apply the local parameter chart to the converged base state."""
        action: Complex128[Array, "n"] = _fixed_support_local_action(
            target,
            volume,
            carrier,
            field,
        )
        return action

    _, action_tangent = jax.jvp(
        fixed_state_action,
        (target.potential.volume, target.carrier),
        (checked_volume_tangent, checked_carrier_tangent),
    )
    tangent_source: Complex128[Array, "n"] = (
        checked_source_tangent - action_tangent
    )
    tangent_result: GalerkinSolveResult = cgls_solve(
        target,
        tangent_source,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    field_tangent: Complex128[Array, "n"] = eqx.error_if(
        tangent_result.field,
        ~tangent_result.converged,
        "Galerkin tangent solve did not converge",
    )
    result: Tuple[Complex128[Array, "n"], Complex128[Array, "n"]] = (
        field,
        field_tangent,
    )
    return result


@jaxtyped(typechecker=beartype)
def galerkin_state_vjp(
    target: GalerkinTargetManifest,
    source: Complex128[Array, "n"],
    output_cotangent: Complex128[Array, "n"],
    max_iterations: scalar_int = 100,
    relative_tolerance: scalar_float = 1e-10,
    absolute_tolerance: scalar_float = 0.0,
) -> Tuple[
    Complex128[Array, "n"],
    Float64[Array, "nz ny nx"],
    Float64[Array, "d"],
    Complex128[Array, "n"],
]:
    """Evaluate the physical fixed-support Galerkin state VJP.

    :see: :class:`~.test_derivatives.TestGalerkinDerivatives`

    Parameters
    ----------
    target : GalerkinTargetManifest
        Canonical fixed-support base target and immutable evidence.
    source : Complex128[Array, "n"]
        Finite source in the target state ordering.
    output_cotangent : Complex128[Array, "n"]
        State cotangent under JAX's real-linear complex convention.
    max_iterations : scalar_int
        Positive CGLS iteration limit for primal and adjoint roots. Default is
        100.
    relative_tolerance : scalar_float
        Non-negative relative residual tolerance. Default is ``1e-10``.
    absolute_tolerance : scalar_float
        Non-negative absolute residual tolerance. Default is zero.

    Returns
    -------
    field : Complex128[Array, "n"]
        Converged finite Galerkin state.
    potential_volume_metric_cotangent : Float64[Array, "nz ny nx"]
        Real Riesz gradient under the physical metric
        ``DeltaV * sum(g * dv)``.  It is not JAX's raw Euclidean array
        cotangent.
    carrier_tangent_cotangent : Float64[Array, "d"]
        Tangent-plane Euclidean Riesz representative on the carrier sphere.
    source_cotangent : Complex128[Array, "n"]
        Cotangent for the finite source under JAX's complex convention.

    Raises
    ------
    ValueError
        If the source or output cotangent leaves the fixed state shape.
    equinox.EquinoxRuntimeError
        If a cotangent is non-finite or a primal/adjoint solve fails.

    Notes
    -----
    JAX first produces a raw Euclidean voxel cotangent ``g_raw``.  This
    routine returns ``g_physical = g_raw / DeltaV`` so that
    ``DeltaV * sum(g_physical * dv)`` equals the state-cotangent pairing.  The
    carrier result is projected onto the tangent plane; it has the same
    pairing as the ambient pullback for every admitted on-shell tangent.
    """
    if source.shape != target.free_diagonal.shape:
        raise ValueError("source must match the target state shape")
    if output_cotangent.shape != source.shape:
        raise ValueError("output_cotangent must match the source shape")
    checked_output_cotangent = _checked_finite(
        output_cotangent,
        "output_cotangent must be finite",
    )

    def source_root(
        candidate_source: Complex128[Array, "n"],
    ) -> Complex128[Array, "n"]:
        """Solve the immutable base target for one varied finite source."""
        candidate_field: Complex128[Array, "n"] = implicit_galerkin_solve(
            target,
            candidate_source,
            max_iterations,
            relative_tolerance,
            absolute_tolerance,
        )
        return candidate_field

    field, source_pullback = jax.vjp(source_root, source)
    source_cotangent: Complex128[Array, "n"] = source_pullback(
        checked_output_cotangent
    )[0]

    def fixed_state_action(
        volume: Float64[Array, "nz ny nx"],
        carrier: Float64[Array, "d"],
    ) -> Complex128[Array, "n"]:
        """Apply the physical local chart to the converged base state."""
        action: Complex128[Array, "n"] = _fixed_support_local_action(
            target,
            volume,
            carrier,
            field,
        )
        return action

    _, action_pullback = jax.vjp(
        fixed_state_action,
        target.potential.volume,
        target.carrier,
    )
    raw_euclidean_volume_cotangent: Float64[Array, "nz ny nx"]
    ambient_carrier_cotangent: Float64[Array, "d"]
    raw_euclidean_volume_cotangent, ambient_carrier_cotangent = (
        action_pullback(-source_cotangent)
    )
    potential_volume_metric_cotangent: Float64[Array, "nz ny nx"] = (
        _physical_voxel_cotangent(
            target,
            raw_euclidean_volume_cotangent,
        )
    )
    carrier_tangent_cotangent: Float64[Array, "d"] = (
        _on_shell_carrier_cotangent(
            target,
            ambient_carrier_cotangent,
        )
    )
    result: Tuple[
        Complex128[Array, "n"],
        Float64[Array, "nz ny nx"],
        Float64[Array, "d"],
        Complex128[Array, "n"],
    ] = (
        field,
        potential_volume_metric_cotangent,
        carrier_tangent_cotangent,
        source_cotangent,
    )
    return result


__all__: list[str] = [
    "galerkin_state_jvp",
    "galerkin_state_vjp",
]
