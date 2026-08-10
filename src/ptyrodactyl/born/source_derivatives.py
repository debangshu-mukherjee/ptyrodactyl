r"""Differentiate represented finite-source construction coordinates.

Extended Summary
----------------
This module supplies the fixed-stratum RM-I1 source-construction seam for a
canonical :class:`~ptyrodactyl.types.GalerkinRepresentedSource`.  It varies
the stored active aperture coefficients, requested reduced flux, transverse
scan position, aberration phases, and source-plane coordinate while keeping
the target, active modes, source route, and every evidence carrier immutable.
The local chart is anchored at the stored algebraic total right-hand side and
uses the actual rounded production free-plus-CAP callable.  Its executable
tangent is ``D S_hat = A_hat_H0_alg(delta v)``; it is neither exact-real
``H0_alg delta v`` arithmetic nor an exact-normalized-target source claim.
Both public routines return the stored canonical ``total_source`` as their
first leaf.  That exact return binds every derivative result to the submitted
represented source rather than substituting an unanchored chart recomputation.

Routine Listings
----------------
:func:`represented_total_source_jvp`
    Evaluate the fixed-stratum represented total-source JVP.
:func:`represented_total_source_vjp`
    Evaluate the fixed-stratum represented total-source VJP.

Notes
-----
The aperture coordinates use the realified complex Euclidean metric, scan
and source-plane coordinates are stored in Angstroms, aberration phases are
stored in radians, and requested flux uses the dimensionless logarithmic
coordinate ``log(tau / tau_0)``.  The complete redundant differential family
is retained deliberately: common real aperture scale, aperture phase versus
aberration phase, scan phase versus aberration phase, and source-plane phase
versus aberration phase.  These null directions are an RM-I3 quotient issue,
not an RM-I1 differentiation ambiguity.

The first returned leaf is always ``source.actions.total_source`` exactly.
The derivative leaf concerns only the anchored local chart and therefore
cannot refresh, replace, or strengthen any RM-S2/RM-S3 evidence carrier.
The chart accepts a structurally valid represented source and preserves,
rather than grants, ``rm_s3_eligible``.  A production physical-source claim
must compose the independent RM-S3 eligibility gate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex128, Float64, jaxtyped

from ptyrodactyl.types import GalerkinRepresentedSource

from .potential import apply_absorber_action

_SPACE_DIMENSIONS: int = 3


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural chart failure.

    Parameters
    ----------
    condition : bool
        Structural failure predicate.
    message : str
        Exception message used when the predicate is true.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _checked_active_complex_tangent(
    source: GalerkinRepresentedSource,
    tangent: Complex128[Array, " n"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one fixed-active-set complex aperture tangent.

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical source supplying the fixed active aperture mask.
    tangent : Complex128[Array, " n"]
        Candidate complex aperture-coefficient tangent.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite tangent that is exactly zero outside the active set.
    """
    inactive_nonzero = (~source.modes.active_mask) & (
        (jnp.real(tangent) != 0.0) | (jnp.imag(tangent) != 0.0)
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent)) | jnp.any(inactive_nonzero),
        "aperture_weights_tangent must be finite and exactly zero outside "
        "the fixed active aperture set",
    )
    return checked


def _checked_active_phase_tangent(
    source: GalerkinRepresentedSource,
    tangent: Float64[Array, " n"],
) -> Float64[Array, " n"]:
    """PRIVATE: Validate one fixed-active-set aberration-phase tangent.

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical source supplying the fixed active aperture mask.
    tangent : Float64[Array, " n"]
        Candidate aberration-phase tangent in radians.

    Returns
    -------
    checked : Float64[Array, " n"]
        Finite phase tangent that is zero outside the active set.
    """
    checked: Float64[Array, " n"] = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent))
        | jnp.any((~source.modes.active_mask) & (tangent != 0.0)),
        "aberration_phases_tangent must be finite and exactly zero outside "
        "the fixed active aperture set",
    )
    return checked


def _checked_scan_tangent(
    source: GalerkinRepresentedSource,
    tangent: Float64[Array, " d"],
) -> Float64[Array, " d"]:
    """PRIVATE: Validate one transverse physical scan tangent.

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical source supplying the static source-normal axis.
    tangent : Float64[Array, " d"]
        Candidate scan-position tangent in Angstroms.

    Returns
    -------
    checked : Float64[Array, " d"]
        Finite scan tangent with exactly zero normal component.
    """
    normal_axis: int = int(source.modes.normal_axis)
    checked: Float64[Array, " d"] = eqx.error_if(
        tangent,
        jnp.any(~jnp.isfinite(tangent)) | (tangent[normal_axis] != 0.0),
        "scan_position_tangent must be finite and exactly transverse",
    )
    return checked


def _checked_scalar_tangent(
    tangent: Float64[Array, ""],
    message: str,
) -> Float64[Array, ""]:
    """PRIVATE: Attach one finite scalar-tangent check.

    Parameters
    ----------
    tangent : Float64[Array, ""]
        Candidate real scalar tangent.
    message : str
        Error text used when the value is non-finite.

    Returns
    -------
    checked : Float64[Array, ""]
        Finite scalar tangent.
    """
    checked: Float64[Array, ""] = eqx.error_if(
        tangent,
        ~jnp.isfinite(tangent),
        message,
    )
    return checked


def _unanchored_incident_field(
    source: GalerkinRepresentedSource,
    aperture_weights: Complex128[Array, " n"],
    log_target_reduced_flux: Float64[Array, ""],
    scan_position: Float64[Array, " d"],
    aberration_phases: Float64[Array, " n"],
    source_plane_coordinate: Float64[Array, ""],
) -> Complex128[Array, " n"]:
    r"""PRIVATE: Recompute the normalized represented incident field.

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical source supplying fixed wavevectors, target, and source axis.
    aperture_weights : Complex128[Array, " n"]
        Candidate full-state aperture coefficients on the fixed active set.
    log_target_reduced_flux : Float64[Array, ""]
        Dimensionless ``log(tau / tau_0)`` flux coordinate.
    scan_position : Float64[Array, " d"]
        Candidate transverse scan position in Angstroms.
    aberration_phases : Float64[Array, " n"]
        Candidate per-state aberration phases in radians.
    source_plane_coordinate : Float64[Array, ""]
        Candidate source-plane coordinate in Angstroms.

    Returns
    -------
    incident : Complex128[Array, " n"]
        Explicitly phased, common-flux-normalized incident coefficients.

    Notes
    -----
    The map evaluates

    ``F = sum(kappa_n * |a|**2) / L_n`` and
    ``v = sqrt(tau_0 * exp(eta) / F) * a * exp(i theta)``.

    Its aperture derivative therefore contains the full normalization term
    ``dF = 2 Re sum(kappa_n * conj(a) * da) / L_n``.
    """
    normal_axis: int = int(source.modes.normal_axis)
    wavevectors: Float64[Array, "n d"] = source.modes.physical_wavevectors
    normal_components: Float64[Array, " n"] = wavevectors[:, normal_axis]
    normal_box_length: Float64[Array, ""] = source.manifest.box_lengths[
        normal_axis
    ]
    squared_moduli: Float64[Array, " n"] = (
        jnp.real(aperture_weights) ** 2 + jnp.imag(aperture_weights) ** 2
    )
    aperture_flux: Float64[Array, ""] = (
        jnp.sum(normal_components * squared_moduli) / normal_box_length
    )
    target_flux: Float64[Array, ""] = (
        source.modes.target_reduced_flux * jnp.exp(log_target_reduced_flux)
    )
    phase: Float64[Array, " n"] = (
        -jnp.sum(wavevectors * scan_position[None, :], axis=-1)
        - normal_components * source_plane_coordinate
        + aberration_phases
    )
    normalization: Float64[Array, ""] = jnp.sqrt(target_flux / aperture_flux)
    raw_incident: Complex128[Array, " n"] = (
        aperture_weights * jnp.exp(1j * phase) * normalization
    )
    incident: Complex128[Array, " n"] = raw_incident
    return incident


def _fixed_total_source_chart(
    source: GalerkinRepresentedSource,
    aperture_weights: Complex128[Array, " n"],
    log_target_reduced_flux: Float64[Array, ""],
    scan_position: Float64[Array, " d"],
    aberration_phases: Float64[Array, " n"],
    source_plane_coordinate: Float64[Array, ""],
) -> Complex128[Array, " n"]:
    """PRIVATE: Evaluate the anchored rounded total-source chart.

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical source whose target, evidence, and additional source stay
        fixed.
    aperture_weights : Complex128[Array, " n"]
        Candidate active aperture coefficients.
    log_target_reduced_flux : Float64[Array, ""]
        Dimensionless logarithmic requested-flux coordinate.
    scan_position : Float64[Array, " d"]
        Candidate transverse scan position in Angstroms.
    aberration_phases : Float64[Array, " n"]
        Candidate active aberration phases in radians.
    source_plane_coordinate : Float64[Array, ""]
        Candidate source-plane coordinate in Angstroms.

    Returns
    -------
    total_source : Complex128[Array, " n"]
        Anchored value ``S0 + A_hat_H0_alg(v - v0)`` evaluated by the rounded
        production free-plus-CAP callable.

    Notes
    -----
    The stored binary64 source is the exact chart origin.  Only the incident
    field delta is acted on by the fixed stored free diagonal and rounded CAP
    callable.  The additional source and every evidence leaf remain
    unchanged.  The exact-real ``H0_alg`` action is an independent
    conformance oracle, not the executable arithmetic or an exact normalized
    carrier target owned by RM-S2 and RM-I2.
    """
    candidate_incident: Complex128[Array, " n"] = _unanchored_incident_field(
        source,
        aperture_weights,
        log_target_reduced_flux,
        scan_position,
        aberration_phases,
        source_plane_coordinate,
    )
    base_incident: Complex128[Array, " n"] = _unanchored_incident_field(
        source,
        source.modes.aperture_weights,
        jnp.asarray(0.0, dtype=jnp.float64),
        source.modes.scan_position,
        source.modes.aberration_phases,
        source.modes.source_plane_coordinate,
    )
    incident_delta: Complex128[Array, " n"] = (
        candidate_incident - base_incident
    )
    free_delta: Complex128[Array, " n"] = (
        source.manifest.free_diagonal * incident_delta
    )
    cap_delta: Complex128[Array, " n"] = (
        source.manifest.cap_scale
        * apply_absorber_action(
            source.manifest.support,
            source.manifest.absorber_coefficients,
            incident_delta,
        )
    )
    varied_total: Complex128[Array, " n"] = (
        source.actions.total_source + free_delta - 1j * cap_delta
    )
    total_source: Complex128[Array, " n"] = varied_total
    return total_source


@jaxtyped(typechecker=beartype)
def represented_total_source_jvp(
    source: GalerkinRepresentedSource,
    aperture_weights_tangent: Complex128[Array, " n"],
    log_target_reduced_flux_tangent: Float64[Array, ""],
    scan_position_tangent: Float64[Array, " d"],
    aberration_phases_tangent: Float64[Array, " n"],
    source_plane_coordinate_tangent: Float64[Array, ""],
) -> Tuple[Complex128[Array, " n"], Complex128[Array, " n"]]:
    """Evaluate the fixed-stratum represented total-source JVP.

    :see: :class:`~.test_source_derivatives.TestRepresentedSourceDerivatives`

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical plane or focused represented source and fixed algebraic
        target.
    aperture_weights_tangent : Complex128[Array, " n"]
        Active complex aperture tangent in stored SC.13 coefficient units,
        under the realified Euclidean metric.
    log_target_reduced_flux_tangent : Float64[Array, ""]
        Tangent to dimensionless ``log(tau / tau_0)``.
    scan_position_tangent : Float64[Array, " d"]
        Transverse scan tangent in Angstroms; the static normal component must
        be exactly zero.
    aberration_phases_tangent : Float64[Array, " n"]
        Active per-mode phase tangent in radians.
    source_plane_coordinate_tangent : Float64[Array, ""]
        Source-plane-coordinate tangent in Angstroms.

    Returns
    -------
    total_source : Complex128[Array, " n"]
        Stored canonical algebraic total right-hand side.
    total_source_tangent : Complex128[Array, " n"]
        Directional derivative of the rounded anchored source callable,
        ``A_hat_H0_alg(delta v)``.

    Raises
    ------
    ValueError
        If a tangent leaves the fixed source shape.
    equinox.EquinoxRuntimeError
        If a tangent is non-finite or changes a fixed active/transverse
        stratum.

    Notes
    -----
    The aperture normalization derivative is included.  A common real
    aperture rescaling is therefore null.  Aperture phase, aberration phase,
    scan phase, and source-plane phase retain their exact represented gauge
    relations for later RM-I3 quotienting.

    ``total_source`` is the stored submitted-source vector, not a newly
    rounded evaluation of the local chart at its origin.  This return binding
    leaves the additional source and every evidence carrier immutable.
    ``total_source_tangent`` carries no per-call action enclosure and must not
    be relabeled as exact-real ``H0_alg delta v`` arithmetic.
    """
    state_shape: Tuple[int, ...] = source.modes.aperture_weights.shape
    _raise_if(
        aperture_weights_tangent.shape != state_shape,
        "aperture_weights_tangent must match the represented state shape",
    )
    _raise_if(
        aberration_phases_tangent.shape != state_shape,
        "aberration_phases_tangent must match the represented state shape",
    )
    _raise_if(
        scan_position_tangent.shape != (_SPACE_DIMENSIONS,),
        "scan_position_tangent must have shape (3,)",
    )
    _raise_if(
        log_target_reduced_flux_tangent.shape != (),
        "log_target_reduced_flux_tangent must be scalar",
    )
    _raise_if(
        source_plane_coordinate_tangent.shape != (),
        "source_plane_coordinate_tangent must be scalar",
    )
    checked_aperture: Complex128[Array, " n"] = (
        _checked_active_complex_tangent(source, aperture_weights_tangent)
    )
    checked_log_flux: Float64[Array, ""] = _checked_scalar_tangent(
        log_target_reduced_flux_tangent,
        "log_target_reduced_flux_tangent must be finite",
    )
    checked_scan: Float64[Array, " d"] = _checked_scan_tangent(
        source,
        scan_position_tangent,
    )
    checked_aberrations: Float64[Array, " n"] = _checked_active_phase_tangent(
        source, aberration_phases_tangent
    )
    checked_plane: Float64[Array, ""] = _checked_scalar_tangent(
        source_plane_coordinate_tangent,
        "source_plane_coordinate_tangent must be finite",
    )

    def local_total_source(
        aperture_weights: Complex128[Array, " n"],
        log_target_reduced_flux: Float64[Array, ""],
        scan_position: Float64[Array, " d"],
        aberration_phases: Float64[Array, " n"],
        source_plane_coordinate: Float64[Array, ""],
    ) -> Complex128[Array, " n"]:
        """Evaluate the differentiable source chart on fixed evidence."""
        candidate: Complex128[Array, " n"] = _fixed_total_source_chart(
            source,
            aperture_weights,
            log_target_reduced_flux,
            scan_position,
            aberration_phases,
            source_plane_coordinate,
        )
        return candidate

    _, differentiated = jax.jvp(
        local_total_source,
        (
            source.modes.aperture_weights,
            jnp.asarray(0.0, dtype=jnp.float64),
            source.modes.scan_position,
            source.modes.aberration_phases,
            source.modes.source_plane_coordinate,
        ),
        (
            checked_aperture,
            checked_log_flux,
            checked_scan,
            checked_aberrations,
            checked_plane,
        ),
    )
    checked_tangent: Complex128[Array, " n"] = eqx.error_if(
        differentiated,
        jnp.any(~jnp.isfinite(differentiated)),
        "represented total-source tangent must be finite",
    )
    total_source: Complex128[Array, " n"] = source.actions.total_source
    total_source_tangent: Complex128[Array, " n"] = checked_tangent
    result: Tuple[Complex128[Array, " n"], Complex128[Array, " n"]] = (
        total_source,
        total_source_tangent,
    )
    return result


@jaxtyped(typechecker=beartype)
def represented_total_source_vjp(
    source: GalerkinRepresentedSource,
    total_source_cotangent: Complex128[Array, " n"],
) -> Tuple[
    Complex128[Array, " n"],
    Complex128[Array, " n"],
    Float64[Array, ""],
    Float64[Array, " d"],
    Float64[Array, " n"],
    Float64[Array, ""],
]:
    """Evaluate the fixed-stratum represented total-source VJP.

    :see: :class:`~.test_source_derivatives.TestRepresentedSourceDerivatives`

    Parameters
    ----------
    source : GalerkinRepresentedSource
        Canonical plane or focused represented source and fixed algebraic
        target.
    total_source_cotangent : Complex128[Array, " n"]
        Total-source cotangent under the declared real pairing
        ``Re(vdot(q, dS))``.

    Returns
    -------
    total_source : Complex128[Array, " n"]
        Stored canonical algebraic total right-hand side.
    aperture_weights_cotangent : Complex128[Array, " n"]
        Realified-complex Euclidean Riesz representative, paired as
        ``Re(vdot(g, da))``, and zero outside the fixed active set.
    log_target_reduced_flux_cotangent : Float64[Array, ""]
        Euclidean cotangent for dimensionless ``log(tau / tau_0)``.
    scan_position_cotangent : Float64[Array, " d"]
        Euclidean coordinate cotangent for scan values stored in Angstroms,
        with zero normal component.
    aberration_phases_cotangent : Float64[Array, " n"]
        Euclidean coordinate cotangent for active phases stored in radians.
    source_plane_coordinate_cotangent : Float64[Array, ""]
        Euclidean coordinate cotangent for the plane value in Angstroms.

    Raises
    ------
    ValueError
        If the output cotangent leaves the fixed source shape.
    equinox.EquinoxRuntimeError
        If the output cotangent or a returned block is non-finite.

    Notes
    -----
    JAX's raw complex pullback is bilinear.  The pullback therefore receives
    the conjugated output cotangent, and the aperture result is conjugated so
    that both boundaries use the declared real pairing
    ``Re(vdot(g, da))``.  The returned null pairings expose the full redundant
    represented-phase family; they do not close RM-I3.

    ``total_source`` is the exact stored submitted-source vector.  The
    pullback differentiates the anchored rounded free-plus-CAP callable only;
    it makes no exact-real ``H0_alg``, exact-normalized-target, per-call action
    enclosure, or refreshed-evidence claim.
    """
    _raise_if(
        total_source_cotangent.shape != source.actions.total_source.shape,
        "total_source_cotangent must match the represented state shape",
    )
    checked_output: Complex128[Array, " n"] = eqx.error_if(
        total_source_cotangent,
        jnp.any(~jnp.isfinite(total_source_cotangent)),
        "total_source_cotangent must be finite",
    )

    def local_total_source(
        aperture_weights: Complex128[Array, " n"],
        log_target_reduced_flux: Float64[Array, ""],
        scan_position: Float64[Array, " d"],
        aberration_phases: Float64[Array, " n"],
        source_plane_coordinate: Float64[Array, ""],
    ) -> Complex128[Array, " n"]:
        """Evaluate the differentiable source chart on fixed evidence."""
        candidate: Complex128[Array, " n"] = _fixed_total_source_chart(
            source,
            aperture_weights,
            log_target_reduced_flux,
            scan_position,
            aberration_phases,
            source_plane_coordinate,
        )
        return candidate

    _, pullback = jax.vjp(
        local_total_source,
        source.modes.aperture_weights,
        jnp.asarray(0.0, dtype=jnp.float64),
        source.modes.scan_position,
        source.modes.aberration_phases,
        source.modes.source_plane_coordinate,
    )
    raw_aperture, log_flux, scan, aberrations, plane = pullback(
        jnp.conj(checked_output)
    )
    aperture_riesz: Complex128[Array, " n"] = jnp.where(
        source.modes.active_mask,
        jnp.conj(raw_aperture),
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )
    scan_tangent: Float64[Array, " d"] = scan.at[
        int(source.modes.normal_axis)
    ].set(0.0)
    phase_tangent: Float64[Array, " n"] = jnp.where(
        source.modes.active_mask,
        aberrations,
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    all_outputs = (
        aperture_riesz,
        log_flux,
        scan_tangent,
        phase_tangent,
        plane,
    )
    finite: jax.Array = jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in all_outputs))
    )
    checked_aperture: Complex128[Array, " n"] = eqx.error_if(
        aperture_riesz,
        ~finite,
        "represented total-source cotangents must be finite",
    )
    total_source: Complex128[Array, " n"] = source.actions.total_source
    aperture_weights_cotangent: Complex128[Array, " n"] = checked_aperture
    log_target_reduced_flux_cotangent: Float64[Array, ""] = log_flux
    scan_position_cotangent: Float64[Array, " d"] = scan_tangent
    aberration_phases_cotangent: Float64[Array, " n"] = phase_tangent
    source_plane_coordinate_cotangent: Float64[Array, ""] = plane
    result: Tuple[
        Complex128[Array, " n"],
        Complex128[Array, " n"],
        Float64[Array, ""],
        Float64[Array, " d"],
        Float64[Array, " n"],
        Float64[Array, ""],
    ] = (
        total_source,
        aperture_weights_cotangent,
        log_target_reduced_flux_cotangent,
        scan_position_cotangent,
        aberration_phases_cotangent,
        source_plane_coordinate_cotangent,
    )
    return result


__all__: list[str] = [
    "represented_total_source_jvp",
    "represented_total_source_vjp",
]
