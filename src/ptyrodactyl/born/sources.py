r"""Construct represented forward sources for scalar Galerkin solves.

Extended Summary
----------------
This module implements the narrow production RM-S3 branch in which every
nonzero incident coefficient is already a represented propagating shell mode
on one positive coordinate-aligned source plane. It constructs either one
plane mode or a coherent finite focused superposition, applies one explicit
physical-wavevector phase convention, normalizes the pre-window reduced flux,
and forms the unique matched finite source ``H_0v = Dv - iBv``.

Routine Listings
----------------
:func:`build_represented_focused_galerkin_source`
    Build a coherent stored-shell finite focused source.
:func:`build_represented_plane_galerkin_source`
    Build one stored-shell represented forward plane mode.

Notes
-----
This module deliberately rejects generic projected angular spectra,
off-shell coefficients, grazing or backward active modes, and multiple normal
branches in one transverse reciprocal fiber. Its exact analytic comparison
target is the resulting finite periodic shell superposition itself. The
stored phase and normalization define that exact finite incident vector.
Shared FTZ-safe intervals enclose its free, CAP, interaction, matched, total,
and scattered source actions. Narrow RM-S3 eligibility additionally requires
the bound acquisition's ``SUPPORT_ELIGIBLE`` status, exact declared incident
modes, symbolic exact-shell intervals, positive exact-carrier flux, and finite
source-action bounds. It excludes continuum, window, box-enlargement,
CAP-removal, current, and detector claims.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int64,
    jaxtyped,
)

from ptyrodactyl._interval import (
    _all_normal_arithmetic_supported,
    _arithmetic_environment_probes,
    _interval_add,
    _interval_divide_positive,
    _interval_multiply,
    _interval_square,
    _interval_subtract,
    _mathematical_pi_interval,
    _point_interval,
    _RealInterval,
    _upward_add,
    _upward_multiply,
)
from ptyrodactyl._numeric import (
    has_lost_nonzero_components,
    has_subnormal_components,
)
from ptyrodactyl.types import (
    GalerkinRepresentedSource,
    GalerkinRepresentedSourceKind,
    GalerkinSourceActions,
    GalerkinSourceAxis,
    GalerkinSourceErrorEnclosure,
    GalerkinSourceErrorRoute,
    GalerkinSourcePhaseConvention,
    GalerkinSourceRepresentationRoute,
    GalerkinStoredShellRoute,
    GalerkinTargetManifest,
    create_galerkin_source_actions,
    create_galerkin_source_error_enclosure,
    create_galerkin_source_ledger,
    create_galerkin_source_modes,
    create_represented_galerkin_source,
    scalar_float,
    scalar_int,
)

from ._direct_interval import (
    _complex_interval_add,
    _complex_interval_multiply,
    _complex_point_interval,
    _ComplexInterval,
    _direct_multiplier_with_interval,
    _exact_complex_norm_interval,
    _nonnegative_vector_norm_upper,
    _point_to_interval_component_upper,
)
from .potential import apply_absorber_action, apply_interaction_product

_SPACE_DIMENSIONS: int = 3
_FLUX_ROUNDOFF_FACTOR: float = 4096.0
_TRANSVERSE_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural source failure.

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


def _active_mask(
    coefficients: Complex128[Array, " n"],
) -> Bool[Array, " n"]:
    """PRIVATE: Mark exactly nonzero complex source coefficients.

    Parameters
    ----------
    coefficients : Complex128[Array, " n"]
        Complex source coefficients.

    Returns
    -------
    active : Bool[Array, " n"]
        Mask that is true when either stored component is nonzero.
    """
    active: Bool[Array, " n"] = (jnp.real(coefficients) != 0.0) | (
        jnp.imag(coefficients) != 0.0
    )
    return active


def _physical_wavevectors(
    manifest: GalerkinTargetManifest,
) -> Float64[Array, "n 3"]:
    """PRIVATE: Reconstruct physical angular wavevectors from a target.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target support, box lengths, and carrier.

    Returns
    -------
    wavevectors : Float64[Array, "n 3"]
        Physical angular wavevectors in radians per Angstrom.
    """
    reciprocal: Float64[Array, "n 3"] = (
        manifest.support.state_indices / manifest.box_lengths[None, :]
    )
    wavevectors: Float64[Array, "n 3"] = (
        manifest.carrier[None, :] + 2.0 * jnp.pi * reciprocal
    )
    return wavevectors


def _reduced_flux(
    coefficients: Complex128[Array, " n"],
    normal_components: Float64[Array, " n"],
    normal_box_length: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the SC.13 reduced-flux formula.

    Parameters
    ----------
    coefficients : Complex128[Array, " n"]
        Orthonormal-box source coefficients.
    normal_components : Float64[Array, " n"]
        Normal angular-wavevector components in radians per Angstrom.
    normal_box_length : Float64[Array, ""]
        Box length along the normal axis, in Angstroms.

    Returns
    -------
    flux : Float64[Array, ""]
        Reduced flux in the declared SC.13 normalization.

    Notes
    -----
    This helper evaluates
    ``sum_j normal_components[j] * |coefficients[j]|**2 / normal_box_length``
    for orthonormal-box coefficients. It keeps the signed normal component;
    upstream checks reject active grazing and backward modes. The caller
    applies ``sqrt(target_flux / input_flux)`` normalization. This direct
    binary64 evaluation is not an outward interval certificate.
    """
    squared_moduli: Float64[Array, " n"] = (
        jnp.real(coefficients) ** 2 + jnp.imag(coefficients) ** 2
    )
    flux: Float64[Array, ""] = (
        jnp.sum(normal_components * squared_moduli) / normal_box_length
    )
    return flux


def _flux_consistency_bound(
    left: Float64[Array, ""],
    right: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Bound binary64 disagreement between two flux values.

    Parameters
    ----------
    left : Float64[Array, ""]
        First reduced-flux value.
    right : Float64[Array, ""]
        Second reduced-flux value.

    Returns
    -------
    bound : Float64[Array, ""]
        Scale-relative diagnostic tolerance.

    Notes
    -----
    This rounded diagnostic is not an outward certificate.
    """
    scale: Float64[Array, ""] = jnp.maximum(
        jnp.finfo(jnp.float64).tiny,
        jnp.maximum(jnp.abs(left), jnp.abs(right)),
    )
    raw: Float64[Array, ""] = (
        _FLUX_ROUNDOFF_FACTOR
        * jnp.finfo(jnp.float64).eps
        * lax.stop_gradient(scale)
    )
    bound: Float64[Array, ""] = raw
    return bound


def _checked_aperture_weights(
    manifest: GalerkinTargetManifest,
    aperture_weights: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """PRIVATE: Convert and validate full-state aperture coefficients.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target that fixes the state support length.
    aperture_weights : Complex[Array, "..."]
        Candidate aperture coefficient vector.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite, normal-range, nonzero aperture coefficients.

    Raises
    ------
    ValueError
        If the candidate is not one-dimensional or has the wrong length.
    """
    weights: Complex128[Array, " n"] = jnp.asarray(
        aperture_weights,
        dtype=jnp.complex128,
    )
    _raise_if(weights.ndim != 1, "aperture_weights must be 1D")
    _raise_if(
        weights.shape[0] != manifest.support.state_indices.shape[0],
        "aperture_weights must match the target state support",
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights))
        | has_subnormal_components(weights)
        | ~jnp.any(_active_mask(weights)),
        "aperture_weights must be finite, normal-range, and nonzero",
    )
    return checked


def _checked_additional_source(
    manifest: GalerkinTargetManifest,
    additional_source: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """PRIVATE: Convert and validate an optional additional source.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target that fixes the state support length.
    additional_source : Complex[Array, "..."]
        Candidate additional source coefficients.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite, normal-range additional source coefficients.

    Raises
    ------
    ValueError
        If the candidate is not one-dimensional or has the wrong length.
    """
    additional: Complex128[Array, " n"] = jnp.asarray(
        additional_source,
        dtype=jnp.complex128,
    )
    _raise_if(additional.ndim != 1, "additional_source must be 1D")
    _raise_if(
        additional.shape[0] != manifest.support.state_indices.shape[0],
        "additional_source must match the target state support",
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        additional,
        jnp.any(~jnp.isfinite(additional))
        | has_subnormal_components(additional),
        "additional_source must be finite and contain no subnormal components",
    )
    return checked


def _checked_phase_geometry(
    manifest: GalerkinTargetManifest,
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    source_plane_coordinate: scalar_float,
    normal_axis: GalerkinSourceAxis,
) -> Tuple[
    Float64[Array, " 3"],
    Float64[Array, " n"],
    Float64[Array, ""],
]:
    """PRIVATE: Convert and validate explicit source phase geometry.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target that fixes the state support length.
    scan_position : Float[Array, "..."]
        Three-component transverse scan position in Angstroms.
    aberration_phases : Float[Array, "..."]
        Per-state aberration phases in radians.
    source_plane_coordinate : scalar_float
        Source-plane coordinate along the normal axis, in Angstroms.
    normal_axis : GalerkinSourceAxis
        Coordinate axis normal to the source plane.

    Returns
    -------
    checked_scan : Float64[Array, " 3"]
        Finite transverse scan position.
    aberrations : Float64[Array, " n"]
        Finite aberration phases in radians.
    coordinate : Float64[Array, ""]
        Finite source-plane coordinate in Angstroms.

    Raises
    ------
    ValueError
        If any candidate has the wrong static shape.
    """
    scan: Float64[Array, " 3"] = jnp.asarray(
        scan_position,
        dtype=jnp.float64,
    )
    aberrations: Float64[Array, " n"] = jnp.asarray(
        aberration_phases,
        dtype=jnp.float64,
    )
    coordinate: Float64[Array, ""] = jnp.asarray(
        source_plane_coordinate,
        dtype=jnp.float64,
    )
    size: int = manifest.support.state_indices.shape[0]
    _raise_if(scan.shape != (_SPACE_DIMENSIONS,), "scan_position must be (3,)")
    _raise_if(
        aberrations.shape != (size,),
        "aberration_phases must match the target state support",
    )
    _raise_if(coordinate.shape != (), "source_plane_coordinate must be scalar")
    checked_scan: Float64[Array, " 3"] = eqx.error_if(
        scan,
        jnp.any(~jnp.isfinite(scan))
        | (scan[int(normal_axis)] != 0.0)
        | jnp.any(~jnp.isfinite(aberrations))
        | (~jnp.isfinite(coordinate)),
        "phase geometry must be finite and scan_position must be transverse",
    )
    result: Tuple[
        Float64[Array, " 3"],
        Float64[Array, " n"],
        Float64[Array, ""],
    ] = (checked_scan, aberrations, coordinate)
    return result


def _apply_explicit_phases(
    weights: Complex128[Array, " n"],
    physical_wavevectors: Float64[Array, "n 3"],
    scan_position: Float64[Array, " 3"],
    aberration_phases: Float64[Array, " n"],
    source_plane_coordinate: Float64[Array, ""],
    normal_axis: GalerkinSourceAxis,
) -> Complex128[Array, " n"]:
    r"""PRIVATE: Apply explicit source-plane and aberration phases.

    Parameters
    ----------
    weights : Complex128[Array, " n"]
        Validated aperture coefficients.
    physical_wavevectors : Float64[Array, "n 3"]
        Physical angular wavevectors in radians per Angstrom.
    scan_position : Float64[Array, " 3"]
        Transverse scan position in Angstroms.
    aberration_phases : Float64[Array, " n"]
        Per-state aberration phases in radians.
    source_plane_coordinate : Float64[Array, ""]
        Source-plane coordinate in Angstroms.
    normal_axis : GalerkinSourceAxis
        Coordinate axis normal to the source plane.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Phased coefficients with every nonzero input component preserved.

    Notes
    -----
    The applied phase is
    ``-kappa_perp.scan - kappa_n xi_s + chi_g``.
    """
    normal: int = int(normal_axis)
    normal_components: Float64[Array, " n"] = physical_wavevectors[:, normal]
    scan_phase: Float64[Array, " n"] = jnp.sum(
        physical_wavevectors * scan_position[None, :],
        axis=-1,
    )
    total_phase: Float64[Array, " n"] = (
        -scan_phase
        - normal_components * source_plane_coordinate
        + aberration_phases
    )
    phase_factors: Complex128[Array, " n"] = jnp.exp(1j * total_phase)
    phased: Complex128[Array, " n"] = weights * phase_factors
    checked: Complex128[Array, " n"] = eqx.error_if(
        phased,
        jnp.any(~jnp.isfinite(phased))
        | has_subnormal_components(phased)
        | has_lost_nonzero_components(weights, phased),
        "phase application must preserve every finite nonzero coefficient",
    )
    return checked


def _check_shell_and_branches(
    manifest: GalerkinTargetManifest,
    weights: Complex128[Array, " n"],
    physical_wavevectors: Float64[Array, "n 3"],
    normal_axis: GalerkinSourceAxis,
    stored_shell_route: GalerkinStoredShellRoute,
    shell_defect_tolerance: scalar_float,
) -> Tuple[
    Float64[Array, ""],
    Bool[Array, " n"],
    Bool[Array, " n"],
    Bool[Array, " n"],
    Bool[Array, " n"],
]:
    """PRIVATE: Validate represented forward-shell branch predicates.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target that supplies represented shell defects.
    weights : Complex128[Array, " n"]
        Validated aperture coefficients.
    physical_wavevectors : Float64[Array, "n 3"]
        Physical angular wavevectors in radians per Angstrom.
    normal_axis : GalerkinSourceAxis
        Coordinate axis normal to the source plane.
    stored_shell_route : GalerkinStoredShellRoute
        Requested shell-evidence route.
    shell_defect_tolerance : scalar_float
        Required represented shell tolerance.

    Returns
    -------
    checked_tolerance : Float64[Array, ""]
        Validated exact-zero shell tolerance.
    active : Bool[Array, " n"]
        Mask of active aperture modes.
    forward : Bool[Array, " n"]
        Mask of positive normal branches.
    grazing : Bool[Array, " n"]
        Mask of zero normal branches.
    backward : Bool[Array, " n"]
        Mask of negative normal branches.

    Raises
    ------
    ValueError
        If the route is unsupported or the tolerance is not scalar.
    """
    _raise_if(
        stored_shell_route
        is not GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL,
        "only exact target-diagonal shell evidence is implemented",
    )
    tolerance: Float64[Array, ""] = jnp.asarray(
        shell_defect_tolerance,
        dtype=jnp.float64,
    )
    _raise_if(tolerance.shape != (), "shell_defect_tolerance must be scalar")
    active: Bool[Array, " n"] = _active_mask(weights)
    normal_components: Float64[Array, " n"] = physical_wavevectors[
        :, int(normal_axis)
    ]
    forward: Bool[Array, " n"] = normal_components > 0.0
    grazing: Bool[Array, " n"] = normal_components == 0.0
    backward: Bool[Array, " n"] = normal_components < 0.0
    recomputed_defects: Float64[Array, " n"] = (
        jnp.sum(physical_wavevectors**2, axis=-1) - manifest.wavenumber**2
    )
    target_mismatch: Bool[Array, ""] = jnp.any(
        recomputed_defects != manifest.free_diagonal
    )
    invalid_tolerance: Bool[Array, ""] = (~jnp.isfinite(tolerance)) | (
        tolerance != 0.0
    )
    checked_tolerance: Float64[Array, ""] = eqx.error_if(
        tolerance,
        invalid_tolerance
        | target_mismatch
        | jnp.any(active & ~forward)
        | jnp.any(active & (manifest.free_diagonal != 0.0)),
        "active modes must be forward with exactly zero represented target "
        "free diagonal",
    )
    result: Tuple[
        Float64[Array, ""],
        Bool[Array, " n"],
        Bool[Array, " n"],
        Bool[Array, " n"],
        Bool[Array, " n"],
    ] = (checked_tolerance, active, forward, grazing, backward)
    return result


def _check_unique_active_fibers(
    manifest: GalerkinTargetManifest,
    weights: Complex128[Array, " n"],
    normal_axis: GalerkinSourceAxis,
) -> Complex128[Array, " n"]:
    """PRIVATE: Require one active branch per transverse reciprocal index.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target with ordered state indices.
    weights : Complex128[Array, " n"]
        Validated aperture coefficients.
    normal_axis : GalerkinSourceAxis
        Axis removed when comparing transverse fibers.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Coefficients after the unique-active-fiber check.
    """
    normal: int = int(normal_axis)
    transverse_axes: Tuple[int, int] = _TRANSVERSE_AXES[normal]
    transverse: Int64[Array, "n 2"] = manifest.support.state_indices[
        :, transverse_axes
    ]
    active: Bool[Array, " n"] = _active_mask(weights)
    size: int = weights.shape[0]
    same_fiber: Bool[Array, "n n"] = jnp.all(
        transverse[:, None, :] == transverse[None, :, :],
        axis=-1,
    )
    distinct: Bool[Array, "n n"] = ~jnp.eye(size, dtype=jnp.bool_)
    duplicate: Bool[Array, ""] = jnp.any(
        same_fiber & distinct & active[:, None] & active[None, :]
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        weights,
        duplicate,
        "active source modes must have one normal branch per transverse "
        "harmonic",
    )
    return checked


def _exact_normal_wavevector_interval(
    manifest: GalerkinTargetManifest,
    normal_axis: GalerkinSourceAxis,
) -> _RealInterval:
    """PRIVATE: Enclose exact-target normal wavevectors on every state row.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target carrying RM-S2 exact-carrier intervals and exact box lengths.
    normal_axis : GalerkinSourceAxis
        Coordinate normal used by the represented source.

    Returns
    -------
    interval : _RealInterval
        Inclusive exact-target normal angular-wavevector endpoints.
    """
    normal: int = int(normal_axis)
    indices: Float64[Array, " n"] = manifest.support.state_indices[
        :, normal
    ].astype(jnp.float64)
    reciprocal: _RealInterval = _interval_divide_positive(
        _point_interval(indices),
        _point_interval(manifest.box_lengths[normal]),
    )
    two_pi: _RealInterval = _interval_multiply(
        _point_interval(jnp.asarray(2.0, dtype=jnp.float64)),
        _mathematical_pi_interval(),
    )
    offset: _RealInterval = _interval_multiply(reciprocal, two_pi)
    ledger = manifest.fixed_linear_error_ledger
    carrier: _RealInterval = (
        jnp.broadcast_to(
            ledger.exact_carrier_lower_bounds[normal], indices.shape
        ),
        jnp.broadcast_to(
            ledger.exact_carrier_upper_bounds[normal], indices.shape
        ),
    )
    interval: _RealInterval = _interval_add(carrier, offset)
    return interval


def _exact_reduced_flux_interval(
    incident_field: Complex128[Array, " n"],
    exact_normal_wavevectors: _RealInterval,
    normal_box_length: Float64[Array, ""],
    target_reduced_flux: Float64[Array, ""],
) -> Tuple[_RealInterval, Float64[Array, ""]]:
    """PRIVATE: Enclose exact-carrier reduced flux and target discrepancy.

    Parameters
    ----------
    incident_field : Complex128[Array, " n"]
        Exact stored finite incident coefficients.
    exact_normal_wavevectors : _RealInterval
        Exact-target normal angular-wavevector intervals.
    normal_box_length : Float64[Array, ""]
        Exact stored positive box length along the source normal.
    target_reduced_flux : Float64[Array, ""]
        Exact stored requested reduced-flux target.

    Returns
    -------
    flux_interval : _RealInterval
        Inclusive exact-carrier reduced-flux endpoints.
    discrepancy_upper : Float64[Array, ""]
        Outward target-to-exact-flux absolute discrepancy upper bound.
    """
    real_points: _RealInterval = _point_interval(jnp.real(incident_field))
    imag_points: _RealInterval = _point_interval(jnp.imag(incident_field))
    squared_moduli: _RealInterval = _interval_add(
        _interval_square(real_points),
        _interval_square(imag_points),
    )
    contributions: _RealInterval = _interval_multiply(
        exact_normal_wavevectors, squared_moduli
    )
    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)

    def add_contribution(
        index: scalar_int,
        accumulator: _RealInterval,
    ) -> _RealInterval:
        """PRIVATE: Accumulate one exact modal-flux interval."""
        contribution: _RealInterval = (
            contributions[0][index],
            contributions[1][index],
        )
        updated: _RealInterval = _interval_add(accumulator, contribution)
        return updated

    summed: _RealInterval = lax.fori_loop(
        0,
        incident_field.shape[0],
        add_contribution,
        (zero, zero),
    )
    flux_interval: _RealInterval = _interval_divide_positive(
        summed, _point_interval(normal_box_length)
    )
    discrepancy: _RealInterval = _interval_subtract(
        _point_interval(target_reduced_flux), flux_interval
    )
    discrepancy_upper: Float64[Array, ""] = jnp.maximum(
        jnp.abs(discrepancy[0]), jnp.abs(discrepancy[1])
    )
    result: Tuple[_RealInterval, Float64[Array, ""]] = (
        flux_interval,
        discrepancy_upper,
    )
    return result


def _enclose_source_actions(
    manifest: GalerkinTargetManifest,
    incident_field: Complex128[Array, " n"],
    actions: GalerkinSourceActions,
) -> GalerkinSourceErrorEnclosure:
    """PRIVATE: Build full algebraic and exact-target RM-S3 source bounds.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical finite target that owns the algebraic operator and bounds.
    incident_field : Complex128[Array, " n"]
        Exact stored incident field in the manifested state ordering.
    actions : GalerkinSourceActions
        Rounded free, absorber, interaction, and source action carrier.

    Returns
    -------
    enclosure : GalerkinSourceErrorEnclosure
        Outward algebraic-action and exact-target source-error evidence.
    """
    absorber = _direct_multiplier_with_interval(
        manifest.support.state_indices,
        manifest.support.absorber_indices,
        manifest.absorber_coefficients,
        incident_field,
        manifest.support.work_shape,
        adjoint=False,
    )
    interaction = _direct_multiplier_with_interval(
        manifest.support.state_indices,
        manifest.support.interaction_indices,
        manifest.interaction_coefficients,
        incident_field,
        manifest.support.work_shape,
        adjoint=False,
    )
    direct_cap: Complex128[Array, " n"] = manifest.cap_scale * absorber[0]
    direct_interaction: Complex128[Array, " n"] = interaction[0]
    free_interval: _ComplexInterval = _complex_interval_multiply(
        _complex_point_interval(manifest.free_diagonal.astype(jnp.complex128)),
        _complex_point_interval(incident_field),
    )
    cap_interval: _ComplexInterval = _complex_interval_multiply(
        _complex_point_interval(manifest.cap_scale.astype(jnp.complex128)),
        absorber[1:],
    )
    interaction_interval: _ComplexInterval = interaction[1:]
    matched_interval: _ComplexInterval = _complex_interval_add(
        free_interval,
        _complex_interval_multiply(
            _complex_point_interval(jnp.asarray(-1j, dtype=jnp.complex128)),
            cap_interval,
        ),
    )
    total_interval: _ComplexInterval = _complex_interval_add(
        matched_interval,
        _complex_point_interval(actions.additional_source),
    )
    scattered_interval: _ComplexInterval = _complex_interval_add(
        interaction_interval,
        _complex_point_interval(actions.additional_source),
    )

    def action_error(
        stored: Complex128[Array, " n"],
        interval: _ComplexInterval,
    ) -> Float64[Array, ""]:
        """PRIVATE: Reduce point-to-rectangle component errors in L2."""
        components: Float64[Array, " n"] = _point_to_interval_component_upper(
            stored, interval
        )
        upper: Float64[Array, ""] = _nonnegative_vector_norm_upper(components)
        return upper

    free_error: Float64[Array, ""] = action_error(
        actions.free_action, free_interval
    )
    cap_error: Float64[Array, ""] = action_error(
        actions.cap_action, cap_interval
    )
    interaction_error: Float64[Array, ""] = action_error(
        actions.interaction_action, interaction_interval
    )
    matched_error: Float64[Array, ""] = action_error(
        actions.incident_source, matched_interval
    )
    total_error: Float64[Array, ""] = action_error(
        actions.total_source, total_interval
    )
    scattered_error: Float64[Array, ""] = action_error(
        actions.scattered_source, scattered_interval
    )
    field_norm: _RealInterval = _exact_complex_norm_interval(incident_field)
    zeros: Float64[Array, " n"] = jnp.zeros(
        incident_field.shape, dtype=jnp.float64
    )
    field_magnitude_upper: Float64[Array, " n"] = (
        _point_to_interval_component_upper(
            incident_field,
            (zeros, zeros, zeros, zeros),
        )
    )
    ledger = manifest.fixed_linear_error_ledger
    free_transfer_components: Float64[Array, " n"] = _upward_multiply(
        ledger.free_diagonal_error_bounds, field_magnitude_upper
    )
    free_transfer: Float64[Array, ""] = _nonnegative_vector_norm_upper(
        free_transfer_components
    )
    cap_transfer: Float64[Array, ""] = _upward_multiply(
        ledger.cap_operator_error_bound, field_norm[1]
    )
    interaction_transfer: Float64[Array, ""] = _upward_multiply(
        ledger.interaction_operator_error_bound, field_norm[1]
    )
    exact_matched_error: Float64[Array, ""] = _upward_add(
        matched_error, _upward_add(free_transfer, cap_transfer)
    )
    exact_total_error: Float64[Array, ""] = _upward_add(
        total_error, _upward_add(free_transfer, cap_transfer)
    )
    exact_scattered_error: Float64[Array, ""] = _upward_add(
        scattered_error, interaction_transfer
    )
    *_, gradual_underflow_supported = _arithmetic_environment_probes()
    environment_supported: Bool[Array, ""] = _all_normal_arithmetic_supported()
    certified_bounds = tuple(
        jnp.where(environment_supported, value, jnp.inf)
        for value in (
            free_error,
            cap_error,
            matched_error,
            interaction_error,
            total_error,
            scattered_error,
            field_norm[1],
            free_transfer,
            cap_transfer,
            interaction_transfer,
            exact_matched_error,
            exact_total_error,
            exact_scattered_error,
        )
    )
    stopped = jax.tree.map(
        lax.stop_gradient,
        (
            *certified_bounds[:6],
            direct_cap,
            direct_interaction,
            *certified_bounds[6:],
            environment_supported,
            gradual_underflow_supported,
        ),
    )
    enclosure: GalerkinSourceErrorEnclosure = (
        create_galerkin_source_error_enclosure(
            free_action_error_upper_bound=stopped[0],
            cap_action_error_upper_bound=stopped[1],
            matched_source_error_upper_bound=stopped[2],
            interaction_action_error_upper_bound=stopped[3],
            total_source_error_upper_bound=stopped[4],
            scattered_source_error_upper_bound=stopped[5],
            independent_direct_cap_action=stopped[6],
            independent_direct_interaction_action=stopped[7],
            incident_field_norm_upper_bound=stopped[8],
            free_target_transfer_error_upper_bound=stopped[9],
            cap_target_transfer_error_upper_bound=stopped[10],
            interaction_target_transfer_error_upper_bound=stopped[11],
            exact_target_matched_source_error_upper_bound=stopped[12],
            exact_target_total_source_error_upper_bound=stopped[13],
            exact_target_scattered_source_error_upper_bound=stopped[14],
            arithmetic_environment_supported=stopped[15],
            gradual_underflow_supported=stopped[16],
            route=GalerkinSourceErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE,
        )
    )
    return enclosure


def _build_represented_galerkin_source(  # noqa: PLR0913
    manifest: GalerkinTargetManifest,
    aperture_weights: Complex[Array, "..."],
    target_reduced_flux: scalar_float,
    additional_source: Complex[Array, "..."] | None,
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    source_plane_coordinate: scalar_float,
    shell_defect_tolerance: scalar_float,
    *,
    normal_axis: GalerkinSourceAxis,
    phase_convention: GalerkinSourcePhaseConvention,
    stored_shell_route: GalerkinStoredShellRoute,
    kind: GalerkinRepresentedSourceKind,
) -> GalerkinRepresentedSource:
    """PRIVATE: Build one represented stored-shell finite source.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical finite target that owns the support and free diagonal.
    aperture_weights : Complex[Array, "..."]
        Full-state aperture coefficients before phases and normalization.
    target_reduced_flux : scalar_float
        Positive target reduced flux in the SC.13 normalization.
    additional_source : Complex[Array, "..."] | None
        Optional additive source coefficients.
    scan_position : Float[Array, "..."]
        Three-component transverse scan position in Angstroms.
    aberration_phases : Float[Array, "..."]
        Per-state aberration phases in radians.
    source_plane_coordinate : scalar_float
        Source-plane coordinate along the normal axis, in Angstroms.
    shell_defect_tolerance : scalar_float
        Required represented shell tolerance; this route requires zero.
    normal_axis : GalerkinSourceAxis
        Coordinate axis normal to the source plane.
    phase_convention : GalerkinSourcePhaseConvention
        Static convention for explicit source phases.
    stored_shell_route : GalerkinStoredShellRoute
        Static represented-shell evidence route.
    kind : GalerkinRepresentedSourceKind
        Static source-kind classification.

    Returns
    -------
    source : GalerkinRepresentedSource
        Validated represented source, actions, ledgers, and diagnostics.

    Raises
    ------
    ValueError
        If a structural input, route, or static shape is invalid.

    Notes
    -----
    The returned finite source carries every gate for the narrow exact-periodic
    RM-S3 branch. Its eligibility does not extend to an external continuum,
    window, enlarged box, removed CAP, current, or detector claim.
    """
    weights: Complex128[Array, " n"] = _checked_aperture_weights(
        manifest,
        aperture_weights,
    )
    weights = _check_unique_active_fibers(manifest, weights, normal_axis)
    scan, aberrations, plane_coordinate = _checked_phase_geometry(
        manifest,
        scan_position,
        aberration_phases,
        source_plane_coordinate,
        normal_axis,
    )
    physical_wavevectors: Float64[Array, "n 3"] = _physical_wavevectors(
        manifest
    )
    tolerance, active, forward, grazing, backward = _check_shell_and_branches(
        manifest,
        weights,
        physical_wavevectors,
        normal_axis,
        stored_shell_route,
        shell_defect_tolerance,
    )
    phased: Complex128[Array, " n"] = _apply_explicit_phases(
        weights,
        physical_wavevectors,
        scan,
        aberrations,
        plane_coordinate,
        normal_axis,
    )
    target_flux: Float64[Array, ""] = jnp.asarray(
        target_reduced_flux,
        dtype=jnp.float64,
    )
    _raise_if(target_flux.shape != (), "target_reduced_flux must be scalar")
    normal_components: Float64[Array, " n"] = physical_wavevectors[
        :, int(normal_axis)
    ]
    normal_box_length: Float64[Array, ""] = manifest.box_lengths[
        int(normal_axis)
    ]
    aperture_flux: Float64[Array, ""] = _reduced_flux(
        weights,
        normal_components,
        normal_box_length,
    )
    input_flux: Float64[Array, ""] = _reduced_flux(
        phased,
        normal_components,
        normal_box_length,
    )
    normalization: Float64[Array, ""] = jnp.sqrt(target_flux / input_flux)
    incident: Complex128[Array, " n"] = phased * normalization
    output_flux: Float64[Array, ""] = _reduced_flux(
        incident,
        normal_components,
        normal_box_length,
    )
    phase_flux_bound: Float64[Array, ""] = _flux_consistency_bound(
        aperture_flux,
        input_flux,
    )
    output_flux_bound: Float64[Array, ""] = _flux_consistency_bound(
        target_flux,
        output_flux,
    )
    checked_incident: Complex128[Array, " n"] = eqx.error_if(
        incident,
        (~jnp.isfinite(target_flux))
        | (target_flux <= 0.0)
        | (~jnp.isfinite(aperture_flux))
        | (aperture_flux <= 0.0)
        | (~jnp.isfinite(input_flux))
        | (input_flux <= 0.0)
        | (~jnp.isfinite(normalization))
        | (normalization <= 0.0)
        | jnp.any(~jnp.isfinite(incident))
        | has_subnormal_components(incident)
        | has_lost_nonzero_components(phased, incident)
        | (jnp.abs(aperture_flux - input_flux) > phase_flux_bound)
        | (jnp.abs(output_flux - target_flux) > output_flux_bound),
        "common pre-window reduced-flux normalization must be finite, "
        "positive, phase invariant, and preserve every active coefficient",
    )
    exact_normal_wavevectors: _RealInterval = (
        _exact_normal_wavevector_interval(manifest, normal_axis)
    )
    exact_flux: _RealInterval
    exact_flux_discrepancy: Float64[Array, ""]
    exact_flux, exact_flux_discrepancy = _exact_reduced_flux_interval(
        lax.stop_gradient(checked_incident),
        exact_normal_wavevectors,
        normal_box_length,
        target_flux,
    )
    exact_source_evidence = jax.tree.map(
        lax.stop_gradient,
        (
            exact_normal_wavevectors,
            exact_flux,
            exact_flux_discrepancy,
        ),
    )

    free_action: Complex128[Array, " n"] = (
        manifest.free_diagonal * checked_incident
    )
    dimensionless_cap_action: Complex128[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        checked_incident,
    )
    cap_action: Complex128[Array, " n"] = (
        manifest.cap_scale * dimensionless_cap_action
    )
    incident_source: Complex128[Array, " n"] = free_action - 1j * cap_action
    interaction_action: Complex128[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        checked_incident,
    )
    if additional_source is None:
        additional: Complex128[Array, " n"] = jnp.zeros_like(checked_incident)
    else:
        additional = _checked_additional_source(manifest, additional_source)
    total_source: Complex128[Array, " n"] = incident_source + additional
    scattered_source: Complex128[Array, " n"] = interaction_action + additional
    actions: GalerkinSourceActions = create_galerkin_source_actions(
        free_action=free_action,
        cap_action=cap_action,
        interaction_action=interaction_action,
        incident_source=incident_source,
        additional_source=additional,
        total_source=total_source,
        scattered_source=scattered_source,
    )
    modes = create_galerkin_source_modes(
        aperture_weights=weights,
        phased_coefficients=phased,
        incident_field=checked_incident,
        physical_wavevectors=physical_wavevectors,
        shell_defects=manifest.free_diagonal,
        exact_free_diagonal_lower_bounds=(
            manifest.fixed_linear_error_ledger.exact_free_diagonal_lower_bounds
        ),
        exact_free_diagonal_upper_bounds=(
            manifest.fixed_linear_error_ledger.exact_free_diagonal_upper_bounds
        ),
        exact_normal_wavevector_lower_bounds=exact_source_evidence[0][0],
        exact_normal_wavevector_upper_bounds=exact_source_evidence[0][1],
        active_mask=active,
        forward_mask=forward,
        grazing_mask=grazing,
        backward_mask=backward,
        scan_position=scan,
        aberration_phases=aberrations,
        source_plane_coordinate=plane_coordinate,
        shell_defect_tolerance=tolerance,
        aperture_reduced_flux=aperture_flux,
        input_reduced_flux=input_flux,
        target_reduced_flux=target_flux,
        output_reduced_flux=output_flux,
        flux_normalization=normalization,
        exact_reduced_flux_lower_bound=exact_source_evidence[1][0],
        exact_reduced_flux_upper_bound=exact_source_evidence[1][1],
        target_reduced_flux_discrepancy_upper_bound=(exact_source_evidence[2]),
        normal_axis=normal_axis,
        phase_convention=phase_convention,
        stored_shell_route=stored_shell_route,
    )
    representation_ledger = create_galerkin_source_ledger(
        box_error_upper_bound=0.0,
        carrier_error_upper_bound=0.0,
        window_error_upper_bound=0.0,
        preband_error_upper_bound=0.0,
        band_error_upper_bound=0.0,
        algebraic_error_upper_bound=0.0,
        route=GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET,
    )
    error_enclosure: GalerkinSourceErrorEnclosure = _enclose_source_actions(
        manifest,
        lax.stop_gradient(checked_incident),
        actions,
    )
    source: GalerkinRepresentedSource = create_represented_galerkin_source(
        manifest=manifest,
        modes=modes,
        actions=actions,
        representation_ledger=representation_ledger,
        error_enclosure=error_enclosure,
        kind=kind,
    )
    return source


@jaxtyped(typechecker=beartype)
def build_represented_plane_galerkin_source(  # noqa: PLR0913
    manifest: GalerkinTargetManifest,
    state_position: int,
    aperture_weight: complex | Complex[Array, ""],
    target_reduced_flux: scalar_float,
    *,
    normal_axis: GalerkinSourceAxis,
    phase_convention: GalerkinSourcePhaseConvention,
    stored_shell_route: GalerkinStoredShellRoute,
    shell_defect_tolerance: scalar_float,
    source_plane_coordinate: scalar_float,
    scan_position: Float[Array, "..."],
    aberration_phase: scalar_float,
    additional_source: Complex[Array, "..."] | None = None,
) -> GalerkinRepresentedSource:
    """Build one stored-shell represented forward plane mode.

    :see: :class:`~.test_sources.TestRepresentedSources`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical scalar target supplying the state support and physical CAP.
    state_position : int
        Static position of the one active state coefficient.
    aperture_weight : complex | Complex[Array, ""]
        Nonzero complex SC.13 coefficient before phases and normalization.
    target_reduced_flux : scalar_float
        Positive target reduced flux.
    normal_axis : GalerkinSourceAxis
        Positive coordinate-aligned propagation normal.
    phase_convention : GalerkinSourcePhaseConvention
        Explicit coefficient-phase convention.
    stored_shell_route : GalerkinStoredShellRoute
        Exact stored-target free-diagonal evidence route.
    shell_defect_tolerance : scalar_float
        Exact binary64 zero; nonzero tolerances are not admitted by this slice.
    source_plane_coordinate : scalar_float
        Physical reference-plane coordinate in Angstroms.
    scan_position : Float[Array, "..."]
        Three-component physical transverse scan vector in Angstroms.
    aberration_phase : scalar_float
        Phase in radians applied to the active plane coefficient.
    additional_source : Complex[Array, "..."] | None
        Separately declared finite source. ``None`` selects zero. Default is
        ``None``.

    Returns
    -------
    source : GalerkinRepresentedSource
        Structured represented plane-source evidence and eligibility gates.

    Raises
    ------
    ValueError
        If the static position or an array shape is invalid.
    equinox.EquinoxRuntimeError
        If the mode is off shell, grazing, backward, duplicated, non-finite,
        or cannot be positive-flux normalized.

    Notes
    -----
    The matched source includes the physical CAP action even when ``Dv`` is
    exactly zero. ``state_position`` refers to the bound target ordering; it
    is not a reciprocal index and therefore cannot silently select another
    carrier fiber.
    """
    _raise_if(
        isinstance(state_position, bool), "state_position cannot be bool"
    )
    size: int = manifest.support.state_indices.shape[0]
    _raise_if(
        state_position < 0 or state_position >= size,
        "state_position must index the target state support",
    )
    weight: Complex128[Array, ""] = jnp.asarray(
        aperture_weight,
        dtype=jnp.complex128,
    )
    aberration: Float64[Array, ""] = jnp.asarray(
        aberration_phase,
        dtype=jnp.float64,
    )
    _raise_if(weight.shape != (), "aperture_weight must be scalar")
    _raise_if(aberration.shape != (), "aberration_phase must be scalar")
    weights: Complex128[Array, " n"] = (
        jnp.zeros(
            (size,),
            dtype=jnp.complex128,
        )
        .at[state_position]
        .set(weight)
    )
    aberrations: Float64[Array, " n"] = (
        jnp.zeros(
            (size,),
            dtype=jnp.float64,
        )
        .at[state_position]
        .set(aberration)
    )
    source: GalerkinRepresentedSource = _build_represented_galerkin_source(
        manifest=manifest,
        aperture_weights=weights,
        target_reduced_flux=target_reduced_flux,
        additional_source=additional_source,
        scan_position=scan_position,
        aberration_phases=aberrations,
        source_plane_coordinate=source_plane_coordinate,
        shell_defect_tolerance=shell_defect_tolerance,
        normal_axis=normal_axis,
        phase_convention=phase_convention,
        stored_shell_route=stored_shell_route,
        kind=GalerkinRepresentedSourceKind.PLANE_MODE,
    )
    return source


@jaxtyped(typechecker=beartype)
def build_represented_focused_galerkin_source(  # noqa: PLR0913
    manifest: GalerkinTargetManifest,
    aperture_weights: Complex[Array, "..."],
    target_reduced_flux: scalar_float,
    *,
    normal_axis: GalerkinSourceAxis,
    phase_convention: GalerkinSourcePhaseConvention,
    stored_shell_route: GalerkinStoredShellRoute,
    shell_defect_tolerance: scalar_float,
    source_plane_coordinate: scalar_float,
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    additional_source: Complex[Array, "..."] | None = None,
) -> GalerkinRepresentedSource:
    """Build a coherent stored-shell finite focused source.

    :see: :class:`~.test_sources.TestRepresentedSources`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical scalar target supplying the state support and physical CAP.
    aperture_weights : Complex[Array, "..."]
        Full-state SC.13 coefficient vector before phase factors. At least two
        entries must be nonzero.
    target_reduced_flux : scalar_float
        Positive target reduced flux for the coherent superposition.
    normal_axis : GalerkinSourceAxis
        Positive coordinate-aligned propagation normal.
    phase_convention : GalerkinSourcePhaseConvention
        Explicit coefficient-phase convention.
    stored_shell_route : GalerkinStoredShellRoute
        Exact stored-target free-diagonal evidence route.
    shell_defect_tolerance : scalar_float
        Exact binary64 zero; nonzero tolerances are not admitted by this slice.
    source_plane_coordinate : scalar_float
        Physical reference-plane coordinate in Angstroms.
    scan_position : Float[Array, "..."]
        Three-component physical transverse scan vector in Angstroms.
    aberration_phases : Float[Array, "..."]
        Explicit per-state aberration phases in radians.
    additional_source : Complex[Array, "..."] | None
        Separately declared finite source. ``None`` selects zero. Default is
        ``None``.

    Returns
    -------
    source : GalerkinRepresentedSource
        Structured coherent represented-source evidence and eligibility gates.

    Raises
    ------
    ValueError
        If an input array has the wrong rank or length.
    equinox.EquinoxRuntimeError
        If a nonzero mode is off shell, grazing, backward, duplicates a
        transverse fiber, or cannot be positive-flux normalized.

    Notes
    -----
    A single common factor normalizes the coherent superposition. Independent
    per-coefficient normalization would alter the supplied aperture shape and
    is intentionally unavailable.
    """
    source: GalerkinRepresentedSource = _build_represented_galerkin_source(
        manifest=manifest,
        aperture_weights=aperture_weights,
        target_reduced_flux=target_reduced_flux,
        additional_source=additional_source,
        scan_position=scan_position,
        aberration_phases=aberration_phases,
        source_plane_coordinate=source_plane_coordinate,
        shell_defect_tolerance=shell_defect_tolerance,
        normal_axis=normal_axis,
        phase_convention=phase_convention,
        stored_shell_route=stored_shell_route,
        kind=GalerkinRepresentedSourceKind.COHERENT_FOCUSED,
    )
    return source


__all__: list[str] = [
    "build_represented_focused_galerkin_source",
    "build_represented_plane_galerkin_source",
]
