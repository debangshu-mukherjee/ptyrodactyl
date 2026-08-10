r"""Define represented-source carriers for scalar Galerkin solves.

Extended Summary
----------------
This module owns the narrow RM-S3 carrier vocabulary for coordinate-aligned
sources made only from represented forward shell modes. It separates source
geometry and flux data, finite operator actions, the six-stage representation
ledger, and numerical error enclosures. The carriers do not relabel a generic
projected angular spectrum as an exact finite source.

Routine Listings
----------------
:class:`GalerkinRepresentedSource`
    Bind one represented stored-shell source to a scalar target.
:class:`GalerkinRepresentedSourceKind`
    Store the represented stored-shell source-construction kind.
:class:`GalerkinSourceActions`
    Store mandatory free, CAP, interaction, and source vectors.
:class:`GalerkinSourceAxis`
    Store the positive coordinate-aligned propagation normal.
:class:`GalerkinSourceErrorEnclosure`
    Store numerical error enclosures for the finite source actions.
:class:`GalerkinSourceErrorRoute`
    Store the source/action numerical-error route.
:class:`GalerkinSourceModes`
    Store represented modes, phase geometry, masks, and reduced fluxes.
:class:`GalerkinSourcePhaseConvention`
    Store the explicit coefficient-phase convention.
:class:`GalerkinSourceRepresentationLedger`
    Store all six RM-S3 staged representation-error terms.
:class:`GalerkinSourceRepresentationRoute`
    Store the narrow represented-target ledger route.
:class:`GalerkinStoredShellRoute`
    Store the represented-target shell-evidence route.
:func:`create_galerkin_source_actions`
    Create validated finite source-action vectors.
:func:`create_galerkin_source_error_enclosure`
    Create honest source/action numerical-error enclosures.
:func:`create_galerkin_source_ledger`
    Create the complete six-stage RM-S3 representation ledger.
:func:`create_galerkin_source_modes`
    Create validated exact represented-mode and flux data.
:func:`create_represented_galerkin_source`
    Bind source evidence to one manifested scalar target.

Notes
-----
The finite source is ``H_0 v = Dv - iBv`` with the physical CAP included.
The represented-target ledger means that the intended incident field is the
exact stored finite periodic coefficient vector on the same box. It says
nothing about a continuum probe, an open-boundary field, or a projected
angular spectrum. Production eligibility is earned only when that vector is
declared by the bound acquisition, is on the exact RM-S2 shell, carries an
outward exact-carrier flux enclosure, and has finite matched, total, and
scattered source-action enclosures.
"""

from enum import Enum, IntEnum

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
    jaxtyped,
)

from ptyrodactyl._interval import _interval_subtract, _point_interval
from ptyrodactyl._numeric import has_subnormal_components

from .acquisition_types import (
    GalerkinAcquisitionSupportStatus,
    GalerkinDirectionDisposition,
    GalerkinTerminalSide,
)
from .custom_types import scalar_float
from .galerkin_types import GalerkinTargetManifest

_MIN_FOCUSED_MODES: int = 2
_TRANSVERSE_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))
_ELIGIBILITY_SCOPE: str = (
    "RM-S3 exact stored periodic finite-source branch only; excludes "
    "continuum, window, pre-band, box-enlargement, open-boundary, "
    "CAP-removal, "
    "current, detector, calibration, solver, and model-discrepancy claims"
)
_EXACT_ACTION_TARGET: str = (
    "exact-real D_alg/B_alg/R_alg actions on the exact stored binary64 "
    "incident vector and exact stored additional source"
)
_SOURCE_ERROR_SCOPE: str = (
    "rounded source actions plus source-specific RM-S2 D/B/R target transfer; "
    "excludes full delta_H, continuum, window, box, current, detector, "
    "solver, "
    "and model errors"
)
_ARITHMETIC_ENVIRONMENT: str = (
    "shared FTZ-safe binary64 outward arithmetic with runtime "
    "normal-operation "
    "probes; gradual underflow is diagnostic rather than load-bearing"
)


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


def _checked_bound(
    value: scalar_float,
    name: str,
    *,
    finite: bool,
) -> Float64[Array, ""]:
    """PRIVATE: Convert and validate a non-negative scalar bound.

    Parameters
    ----------
    value : scalar_float
        Candidate scalar bound.
    name : str
        Field name used in validation messages.
    finite : bool
        Require a finite value when true; otherwise permit positive infinity.

    Returns
    -------
    checked : Float64[Array, ""]
        Validated non-negative binary64 bound.

    Raises
    ------
    ValueError
        If the candidate is not scalar.
    """
    value_array: Float64[Array, ""] = jnp.asarray(value, dtype=jnp.float64)
    _raise_if(value_array.shape != (), f"{name} must be a scalar")
    invalid: Bool[Array, ""] = jnp.isnan(value_array) | (value_array < 0.0)
    if finite:
        invalid = invalid | ~jnp.isfinite(value_array)
    checked: Float64[Array, ""] = eqx.error_if(
        value_array,
        invalid,
        f"{name} must be non-negative and "
        + ("finite" if finite else "not NaN"),
    )
    return checked


def _checked_vector(
    values: Complex[Array, "..."],
    size: int,
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Convert and validate a complex source vector.

    Parameters
    ----------
    values : Complex[Array, "..."]
        Candidate complex source coefficients.
    size : int
        Required vector length.
    name : str
        Field name used in validation messages.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite, normal-range complex128 source vector.

    Raises
    ------
    ValueError
        If the candidate is not one-dimensional or has the wrong length.
    """
    array: Complex128[Array, " n"] = jnp.asarray(
        values,
        dtype=jnp.complex128,
    )
    _raise_if(array.ndim != 1, f"{name} must be 1D")
    _raise_if(array.shape[0] != size, f"{name} must have length {size}")
    checked: Complex128[Array, " n"] = eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | has_subnormal_components(array),
        f"{name} must be finite and contain no nonzero subnormal components",
    )
    return checked


class GalerkinSourceAxis(IntEnum):
    """Store the positive coordinate-aligned propagation normal.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    X : int
        Positive first-coordinate normal.
    Y : int
        Positive second-coordinate normal.
    Z : int
        Positive third-coordinate normal.
    """

    X = 0
    Y = 1
    Z = 2


class GalerkinRepresentedSourceKind(str, Enum):
    """Store the represented stored-shell source-construction kind.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    PLANE_MODE : str
        Exactly one nonzero represented forward shell coefficient.
    COHERENT_FOCUSED : str
        At least two coherent represented forward shell coefficients.
    """

    PLANE_MODE = "represented_plane_mode"
    COHERENT_FOCUSED = "represented_coherent_focused"


class GalerkinSourcePhaseConvention(str, Enum):
    r"""Store the explicit coefficient-phase convention.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    PHYSICAL_WAVEVECTOR : str
        Apply ``exp[-i kappa_perp.scan - i kappa_n xi_s + i chi_g]``.
    """

    PHYSICAL_WAVEVECTOR = "physical_kappa_scan_source_plus_aberration"


class GalerkinStoredShellRoute(str, Enum):
    """Store the represented-target shell-evidence route.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    EXACT_STORED_DIAGONAL : str
        Every active stored target free defect is exactly binary64 zero.

    Notes
    -----
    This certifies only the stored target-diagonal predicate. It is not an
    RM-S1 finite-support/core eligibility artifact, full RM-S1 detector
    eligibility, or full RM-S3 production eligibility.
    """

    EXACT_STORED_DIAGONAL = "exact_stored_free_diagonal"


class GalerkinSourceRepresentationRoute(str, Enum):
    """Store the narrow represented-target ledger route.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    EXACT_PERIODIC_FINITE_TARGET : str
        The analytic target is the represented periodic finite field itself.
    """

    EXACT_PERIODIC_FINITE_TARGET = "exact_periodic_finite_target"


class GalerkinSourceErrorRoute(str, Enum):
    """Store the source/action numerical-error route.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    FTZ_SAFE_DIRECT_INTERVAL_BRIDGE : str
        Bound production source actions through an independent direct
        coefficient contraction and shared FTZ-safe outward intervals.
    NONCERTIFIED_INFINITY : str
        No finite outward binary64 action enclosure has been established.
    """

    FTZ_SAFE_DIRECT_INTERVAL_BRIDGE = "rm_s3_ftz_safe_direct_interval_bridge"
    NONCERTIFIED_INFINITY = "typed_noncertificate_infinity"


class GalerkinSourceRepresentationLedger(eqx.Module):
    """Store all six RM-S3 staged representation-error terms.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    box_error_upper_bound : Float64[Array, ""]
        Finite-box surrogate error ``delta_box``.
    carrier_error_upper_bound : Float64[Array, ""]
        Carrier-model error ``delta_car``.
    window_error_upper_bound : Float64[Array, ""]
        Window error ``delta_win``.
    preband_error_upper_bound : Float64[Array, ""]
        Pre-band periodization or geometry error ``delta_Pi``.
    band_error_upper_bound : Float64[Array, ""]
        Exact state-band truncation error ``delta_band``.
    algebraic_error_upper_bound : Float64[Array, ""]
        Finite coefficient-realization error ``delta_alg``.
    route : GalerkinSourceRepresentationRoute
        Static definition of the analytic comparison target.
    """

    box_error_upper_bound: Float64[Array, ""]
    carrier_error_upper_bound: Float64[Array, ""]
    window_error_upper_bound: Float64[Array, ""]
    preband_error_upper_bound: Float64[Array, ""]
    band_error_upper_bound: Float64[Array, ""]
    algebraic_error_upper_bound: Float64[Array, ""]
    route: GalerkinSourceRepresentationRoute = eqx.field(static=True)


class GalerkinSourceErrorEnclosure(eqx.Module):
    """Store numerical error enclosures for the finite source actions.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    free_action_error_upper_bound : Float64[Array, ""]
        Outward norm bound for numerical error in the stored ``Dv``.
    cap_action_error_upper_bound : Float64[Array, ""]
        Outward norm bound for numerical error in the stored physical ``Bv``.
    matched_source_error_upper_bound : Float64[Array, ""]
        Outward algebraic-action error in stored ``Dv - iBv``.
    interaction_action_error_upper_bound : Float64[Array, ""]
        Outward algebraic-action error in stored ``Rv``.
    total_source_error_upper_bound : Float64[Array, ""]
        Outward algebraic formation error in ``Dv-iBv+S_add``.
    scattered_source_error_upper_bound : Float64[Array, ""]
        Outward algebraic formation error in ``Rv+S_add``.
    independent_direct_cap_action : Complex128[Array, " n"]
        Independently accumulated rounded ``B_alg v``.
    independent_direct_interaction_action : Complex128[Array, " n"]
        Independently accumulated rounded ``R_alg v``.
    incident_field_norm_upper_bound : Float64[Array, ""]
        Outward exact-real norm upper bound for stored ``v``.
    free_target_transfer_error_upper_bound : Float64[Array, ""]
        Source-specific ``||(D_alg-D)v||`` bound.
    cap_target_transfer_error_upper_bound : Float64[Array, ""]
        Source-specific ``||(B_alg-B)v||`` bound.
    interaction_target_transfer_error_upper_bound : Float64[Array, ""]
        Source-specific ``||(R_alg-R)v||`` bound.
    exact_target_matched_source_error_upper_bound : Float64[Array, ""]
        Complete error from stored matched source to exact ``H_0v``.
    exact_target_total_source_error_upper_bound : Float64[Array, ""]
        Complete error from stored total RHS to exact ``H_0v+S_add``.
    exact_target_scattered_source_error_upper_bound : Float64[Array, ""]
        Complete error from stored scattered RHS to exact ``Rv+S_add``.
    arithmetic_environment_supported : Bool[Array, ""]
        Whether all load-bearing normal binary64 probes passed.
    gradual_underflow_supported : Bool[Array, ""]
        Diagnostic gradual-underflow probe result.
    finite_certificate : Bool[Array, ""]
        Whether every final source bound is finite under the stated route.
    route : GalerkinSourceErrorRoute
        Static numerical-enclosure route.
    exact_action_target : str
        Static exact-real algebraic target declaration.
    error_scope : str
        Static declaration of included and excluded error terms.
    arithmetic_environment : str
        Static arithmetic assumptions and runtime-probe declaration.
    """

    free_action_error_upper_bound: Float64[Array, ""]
    cap_action_error_upper_bound: Float64[Array, ""]
    matched_source_error_upper_bound: Float64[Array, ""]
    interaction_action_error_upper_bound: Float64[Array, ""]
    total_source_error_upper_bound: Float64[Array, ""]
    scattered_source_error_upper_bound: Float64[Array, ""]
    independent_direct_cap_action: Complex128[Array, " n"]
    independent_direct_interaction_action: Complex128[Array, " n"]
    incident_field_norm_upper_bound: Float64[Array, ""]
    free_target_transfer_error_upper_bound: Float64[Array, ""]
    cap_target_transfer_error_upper_bound: Float64[Array, ""]
    interaction_target_transfer_error_upper_bound: Float64[Array, ""]
    exact_target_matched_source_error_upper_bound: Float64[Array, ""]
    exact_target_total_source_error_upper_bound: Float64[Array, ""]
    exact_target_scattered_source_error_upper_bound: Float64[Array, ""]
    arithmetic_environment_supported: Bool[Array, ""]
    gradual_underflow_supported: Bool[Array, ""]
    finite_certificate: Bool[Array, ""]
    route: GalerkinSourceErrorRoute = eqx.field(static=True)
    exact_action_target: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    arithmetic_environment: str = eqx.field(static=True)


class GalerkinSourceModes(eqx.Module):
    """Store represented modes, phase geometry, masks, and reduced fluxes.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    aperture_weights : Complex128[Array, " n"]
        Supplied SC.13 orthonormal state coefficients before phase factors.
    phased_coefficients : Complex128[Array, " n"]
        Coefficients after scan, source-plane, and aberration phases.
    incident_field : Complex128[Array, " n"]
        Common-flux-normalized incident coefficients.
    physical_wavevectors : Float64[Array, "n 3"]
        Physical angular wavevectors ``k_i + 2 pi g``.
    shell_defects : Float64[Array, " n"]
        Stored target free-shell defects for all state modes.
    exact_free_diagonal_lower_bounds : Float64[Array, " n"]
        RM-S2 lower endpoints for exact SC-1 free defects.
    exact_free_diagonal_upper_bounds : Float64[Array, " n"]
        RM-S2 upper endpoints for exact SC-1 free defects.
    exact_normal_wavevector_lower_bounds : Float64[Array, " n"]
        Outward exact-target normal angular-wavevector lower endpoints.
    exact_normal_wavevector_upper_bounds : Float64[Array, " n"]
        Outward exact-target normal angular-wavevector upper endpoints.
    active_mask : Bool[Array, " n"]
        Modes having nonzero supplied aperture weights.
    forward_mask : Bool[Array, " n"]
        Modes with positive physical normal angular wavenumber.
    grazing_mask : Bool[Array, " n"]
        Modes with zero physical normal angular wavenumber.
    backward_mask : Bool[Array, " n"]
        Modes with negative physical normal angular wavenumber.
    scan_position : Float64[Array, " 3"]
        Physical transverse scan position in Angstroms.
    aberration_phases : Float64[Array, " n"]
        Explicit per-mode aberration phases in radians.
    source_plane_coordinate : Float64[Array, ""]
        Physical source/reference-plane coordinate in Angstroms.
    shell_defect_tolerance : Float64[Array, ""]
        Exact binary64 zero for this first represented-shell slice.
    aperture_reduced_flux : Float64[Array, ""]
        Reduced normal flux before phase factors.
    input_reduced_flux : Float64[Array, ""]
        Reduced normal flux after phases and before common normalization.
    target_reduced_flux : Float64[Array, ""]
        Declared positive target reduced flux.
    output_reduced_flux : Float64[Array, ""]
        Recomputed reduced flux after common normalization.
    flux_normalization : Float64[Array, ""]
        One common positive coefficient normalization factor.
    exact_reduced_flux_lower_bound : Float64[Array, ""]
        Outward lower endpoint for exact-carrier reduced flux of stored ``v``.
    exact_reduced_flux_upper_bound : Float64[Array, ""]
        Outward upper endpoint for exact-carrier reduced flux of stored ``v``.
    target_reduced_flux_discrepancy_upper_bound : Float64[Array, ""]
        Outward error between exact-carrier flux and requested target flux.
    normal_axis : GalerkinSourceAxis
        Static positive coordinate-aligned propagation normal.
    phase_convention : GalerkinSourcePhaseConvention
        Static coefficient-phase convention.
    stored_shell_route : GalerkinStoredShellRoute
        Static represented-shell acceptance route.
    """

    aperture_weights: Complex128[Array, " n"]
    phased_coefficients: Complex128[Array, " n"]
    incident_field: Complex128[Array, " n"]
    physical_wavevectors: Float64[Array, "n 3"]
    shell_defects: Float64[Array, " n"]
    exact_free_diagonal_lower_bounds: Float64[Array, " n"]
    exact_free_diagonal_upper_bounds: Float64[Array, " n"]
    exact_normal_wavevector_lower_bounds: Float64[Array, " n"]
    exact_normal_wavevector_upper_bounds: Float64[Array, " n"]
    active_mask: Bool[Array, " n"]
    forward_mask: Bool[Array, " n"]
    grazing_mask: Bool[Array, " n"]
    backward_mask: Bool[Array, " n"]
    scan_position: Float64[Array, " 3"]
    aberration_phases: Float64[Array, " n"]
    source_plane_coordinate: Float64[Array, ""]
    shell_defect_tolerance: Float64[Array, ""]
    aperture_reduced_flux: Float64[Array, ""]
    input_reduced_flux: Float64[Array, ""]
    target_reduced_flux: Float64[Array, ""]
    output_reduced_flux: Float64[Array, ""]
    flux_normalization: Float64[Array, ""]
    exact_reduced_flux_lower_bound: Float64[Array, ""]
    exact_reduced_flux_upper_bound: Float64[Array, ""]
    target_reduced_flux_discrepancy_upper_bound: Float64[Array, ""]
    normal_axis: GalerkinSourceAxis = eqx.field(static=True)
    phase_convention: GalerkinSourcePhaseConvention = eqx.field(static=True)
    stored_shell_route: GalerkinStoredShellRoute = eqx.field(static=True)


class GalerkinSourceActions(eqx.Module):
    """Store mandatory free, CAP, interaction, and source vectors.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    free_action : Complex128[Array, " n"]
        Shifted finite free action ``Dv``.
    cap_action : Complex128[Array, " n"]
        Physical compressed CAP action ``Bv`` including its scale.
    interaction_action : Complex128[Array, " n"]
        Represented specimen interaction action ``Rv``.
    incident_source : Complex128[Array, " n"]
        Mandatory matched source ``Dv - iBv``.
    additional_source : Complex128[Array, " n"]
        Separately declared finite source beyond the matched injection.
    total_source : Complex128[Array, " n"]
        Total-field right-hand side ``H_0v + S_add``.
    scattered_source : Complex128[Array, " n"]
        Equivalent scattered right-hand side ``Rv + S_add``.
    """

    free_action: Complex128[Array, " n"]
    cap_action: Complex128[Array, " n"]
    interaction_action: Complex128[Array, " n"]
    incident_source: Complex128[Array, " n"]
    additional_source: Complex128[Array, " n"]
    total_source: Complex128[Array, " n"]
    scattered_source: Complex128[Array, " n"]


class GalerkinRepresentedSource(eqx.Module):
    """Bind one represented stored-shell source to a scalar target.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Attributes
    ----------
    manifest : GalerkinTargetManifest
        Canonical scalar target supplying the state space, carrier, and CAP.
    modes : GalerkinSourceModes
        Represented shell modes, source geometry, masks, and flux data.
    actions : GalerkinSourceActions
        Mandatory finite operator actions and matched-source decomposition.
    representation_ledger : GalerkinSourceRepresentationLedger
        Complete six-term RM-S3 representation ledger.
    error_enclosure : GalerkinSourceErrorEnclosure
        Honest numerical action/source enclosures.
    kind : GalerkinRepresentedSourceKind
        Static plane-mode or coherent-focused construction kind.
    support_eligible : Bool[Array, ""]
        Whether the bound acquisition retains SUPPORT_ELIGIBLE status.
    declared_incident_eligible : Bool[Array, ""]
        Whether every active coefficient is one declared exact incident mode.
    exact_shell_eligible : Bool[Array, ""]
        Whether every active mode has symbolic exact RM-S2 free defect zero.
    exact_flux_eligible : Bool[Array, ""]
        Whether exact-carrier flux is finite, strictly forward, and bounded.
    action_enclosures_eligible : Bool[Array, ""]
        Whether all matched, total, and scattered action bounds are finite.
    rm_s3_eligible : Bool[Array, ""]
        Conjunction of every narrow exact-periodic RM-S3 production gate.
    eligibility_scope : str
        Static boundary of the eligibility claim.
    """

    manifest: GalerkinTargetManifest
    modes: GalerkinSourceModes
    actions: GalerkinSourceActions
    representation_ledger: GalerkinSourceRepresentationLedger
    error_enclosure: GalerkinSourceErrorEnclosure
    support_eligible: Bool[Array, ""]
    declared_incident_eligible: Bool[Array, ""]
    exact_shell_eligible: Bool[Array, ""]
    exact_flux_eligible: Bool[Array, ""]
    action_enclosures_eligible: Bool[Array, ""]
    rm_s3_eligible: Bool[Array, ""]
    kind: GalerkinRepresentedSourceKind = eqx.field(static=True)
    eligibility_scope: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_source_ledger(
    box_error_upper_bound: scalar_float,
    carrier_error_upper_bound: scalar_float,
    window_error_upper_bound: scalar_float,
    preband_error_upper_bound: scalar_float,
    band_error_upper_bound: scalar_float,
    algebraic_error_upper_bound: scalar_float,
    *,
    route: GalerkinSourceRepresentationRoute,
) -> GalerkinSourceRepresentationLedger:
    """Create the complete six-stage RM-S3 representation ledger.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Parameters
    ----------
    box_error_upper_bound : scalar_float
        Non-negative ``delta_box`` bound.
    carrier_error_upper_bound : scalar_float
        Non-negative ``delta_car`` bound.
    window_error_upper_bound : scalar_float
        Non-negative ``delta_win`` bound.
    preband_error_upper_bound : scalar_float
        Non-negative ``delta_Pi`` bound.
    band_error_upper_bound : scalar_float
        Non-negative ``delta_band`` bound.
    algebraic_error_upper_bound : scalar_float
        Non-negative ``delta_alg`` bound; infinity is a noncertificate.
    route : GalerkinSourceRepresentationRoute
        Static analytic-target definition.

    Returns
    -------
    ledger : GalerkinSourceRepresentationLedger
        Validated six-term representation ledger.

    Raises
    ------
    ValueError
        If an input is not scalar.
    equinox.EquinoxRuntimeError
        If a bound is negative or NaN, or an exact stage is not zero.

    Notes
    -----
    For the exact periodic finite-target route, the exact comparison field is
    the stored coefficient vector itself, so all six representation stages
    are exact identities. Rounded source-action error is stored separately.
    """
    bounds: Tuple[Float64[Array, ""], ...] = tuple(
        _checked_bound(value, name, finite=False)
        for value, name in (
            (box_error_upper_bound, "box_error_upper_bound"),
            (carrier_error_upper_bound, "carrier_error_upper_bound"),
            (window_error_upper_bound, "window_error_upper_bound"),
            (preband_error_upper_bound, "preband_error_upper_bound"),
            (band_error_upper_bound, "band_error_upper_bound"),
            (algebraic_error_upper_bound, "algebraic_error_upper_bound"),
        )
    )
    if route is GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET:
        exact_stages: Float64[Array, " 6"] = jnp.stack(bounds)
        bounds = (
            eqx.error_if(
                bounds[0],
                jnp.any(exact_stages != 0.0),
                "exact periodic finite-target stages must be exactly zero",
            ),
            *bounds[1:],
        )
    ledger: GalerkinSourceRepresentationLedger = (
        GalerkinSourceRepresentationLedger(
            box_error_upper_bound=bounds[0],
            carrier_error_upper_bound=bounds[1],
            window_error_upper_bound=bounds[2],
            preband_error_upper_bound=bounds[3],
            band_error_upper_bound=bounds[4],
            algebraic_error_upper_bound=bounds[5],
            route=route,
        )
    )
    return ledger


@jaxtyped(typechecker=beartype)
def create_galerkin_source_error_enclosure(  # noqa: PLR0913
    free_action_error_upper_bound: scalar_float,
    cap_action_error_upper_bound: scalar_float,
    matched_source_error_upper_bound: scalar_float,
    interaction_action_error_upper_bound: scalar_float,
    total_source_error_upper_bound: scalar_float,
    scattered_source_error_upper_bound: scalar_float,
    independent_direct_cap_action: Complex[Array, "..."],
    independent_direct_interaction_action: Complex[Array, "..."],
    incident_field_norm_upper_bound: scalar_float,
    free_target_transfer_error_upper_bound: scalar_float,
    cap_target_transfer_error_upper_bound: scalar_float,
    interaction_target_transfer_error_upper_bound: scalar_float,
    exact_target_matched_source_error_upper_bound: scalar_float,
    exact_target_total_source_error_upper_bound: scalar_float,
    exact_target_scattered_source_error_upper_bound: scalar_float,
    arithmetic_environment_supported: Bool[Array, ""],
    gradual_underflow_supported: Bool[Array, ""],
    *,
    route: GalerkinSourceErrorRoute,
) -> GalerkinSourceErrorEnclosure:
    """Create honest source/action numerical-error enclosures.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Parameters
    ----------
    free_action_error_upper_bound : scalar_float
        Non-negative numerical ``Dv`` error enclosure.
    cap_action_error_upper_bound : scalar_float
        Non-negative numerical ``Bv`` error enclosure.
    matched_source_error_upper_bound : scalar_float
        Non-negative algebraic matched-source error enclosure.
    interaction_action_error_upper_bound : scalar_float
        Non-negative algebraic ``Rv`` error enclosure.
    total_source_error_upper_bound : scalar_float
        Non-negative algebraic total-source formation error.
    scattered_source_error_upper_bound : scalar_float
        Non-negative algebraic scattered-source formation error.
    independent_direct_cap_action : Complex[Array, "..."]
        Independently accumulated rounded ``B_alg v``.
    independent_direct_interaction_action : Complex[Array, "..."]
        Independently accumulated rounded ``R_alg v``.
    incident_field_norm_upper_bound : scalar_float
        Outward exact-real norm upper bound for stored ``v``.
    free_target_transfer_error_upper_bound : scalar_float
        Source-specific exact-vs-algebraic free-action transfer.
    cap_target_transfer_error_upper_bound : scalar_float
        Source-specific exact-vs-algebraic CAP-action transfer.
    interaction_target_transfer_error_upper_bound : scalar_float
        Source-specific exact-vs-algebraic interaction-action transfer.
    exact_target_matched_source_error_upper_bound : scalar_float
        Complete matched-source error relative to exact SC-1.
    exact_target_total_source_error_upper_bound : scalar_float
        Complete total-source error relative to exact SC-1.
    exact_target_scattered_source_error_upper_bound : scalar_float
        Complete scattered-source error relative to exact SC-1.
    arithmetic_environment_supported : Bool[Array, ""]
        Whether every load-bearing normal arithmetic probe passed.
    gradual_underflow_supported : Bool[Array, ""]
        Diagnostic gradual-underflow probe result.
    route : GalerkinSourceErrorRoute
        Static numerical-enclosure route.

    Returns
    -------
    enclosure : GalerkinSourceErrorEnclosure
        Validated action and source error evidence.

    Raises
    ------
    ValueError
        If an input is not scalar.
    equinox.EquinoxRuntimeError
        If a bound is negative or NaN, vectors mismatch, exact-target bounds
        fail to dominate their algebraic terms, or a noncertificate is finite.
    """
    direct_cap = jnp.asarray(
        independent_direct_cap_action, dtype=jnp.complex128
    )
    direct_interaction = jnp.asarray(
        independent_direct_interaction_action, dtype=jnp.complex128
    )
    _raise_if(direct_cap.ndim != 1, "independent direct actions must be 1D")
    _raise_if(
        direct_interaction.shape != direct_cap.shape,
        "independent direct actions must have matching nonempty shapes",
    )
    _raise_if(direct_cap.shape[0] == 0, "independent direct actions are empty")
    bounds: Tuple[Float64[Array, ""], ...] = tuple(
        _checked_bound(value, name, finite=False)
        for value, name in (
            (
                free_action_error_upper_bound,
                "free_action_error_upper_bound",
            ),
            (cap_action_error_upper_bound, "cap_action_error_upper_bound"),
            (
                matched_source_error_upper_bound,
                "matched_source_error_upper_bound",
            ),
            (
                interaction_action_error_upper_bound,
                "interaction_action_error_upper_bound",
            ),
            (total_source_error_upper_bound, "total_source_error_upper_bound"),
            (
                scattered_source_error_upper_bound,
                "scattered_source_error_upper_bound",
            ),
            (
                incident_field_norm_upper_bound,
                "incident_field_norm_upper_bound",
            ),
            (
                free_target_transfer_error_upper_bound,
                "free_target_transfer_error_upper_bound",
            ),
            (
                cap_target_transfer_error_upper_bound,
                "cap_target_transfer_error_upper_bound",
            ),
            (
                interaction_target_transfer_error_upper_bound,
                "interaction_target_transfer_error_upper_bound",
            ),
            (
                exact_target_matched_source_error_upper_bound,
                "exact_target_matched_source_error_upper_bound",
            ),
            (
                exact_target_total_source_error_upper_bound,
                "exact_target_total_source_error_upper_bound",
            ),
            (
                exact_target_scattered_source_error_upper_bound,
                "exact_target_scattered_source_error_upper_bound",
            ),
        )
    )
    environment = jnp.asarray(
        arithmetic_environment_supported, dtype=jnp.bool_
    )
    gradual = jnp.asarray(gradual_underflow_supported, dtype=jnp.bool_)
    _raise_if(
        environment.shape != (),
        "arithmetic environment flag must be scalar",
    )
    _raise_if(
        gradual.shape != (),
        "gradual underflow flag must be scalar",
    )
    checked_direct_cap = eqx.error_if(
        direct_cap,
        jnp.any(~jnp.isfinite(direct_cap))
        | jnp.any(~jnp.isfinite(direct_interaction)),
        "independent direct source actions must be finite",
    )
    checked_direct_interaction = eqx.error_if(
        direct_interaction,
        jnp.any(~jnp.isfinite(direct_cap))
        | jnp.any(~jnp.isfinite(direct_interaction)),
        "independent direct source actions must be finite",
    )
    if route is GalerkinSourceErrorRoute.NONCERTIFIED_INFINITY:
        final_errors: Float64[Array, " 6"] = jnp.stack(
            (bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        )
        bounds = (
            eqx.error_if(
                bounds[0],
                jnp.any(~jnp.isinf(final_errors)),
                "the noncertified source-error route requires infinity",
            ),
            *bounds[1:],
        )
    dominance_invalid: Bool[Array, ""] = (
        (bounds[10] < bounds[2])
        | (bounds[11] < bounds[4])
        | (bounds[12] < bounds[5])
    )
    checked_direct_cap = eqx.error_if(
        checked_direct_cap,
        dominance_invalid,
        "exact-target source errors must dominate algebraic formation errors",
    )
    finite_certificate: Bool[Array, ""] = environment & jnp.all(
        jnp.isfinite(jnp.stack(bounds))
    )
    enclosure: GalerkinSourceErrorEnclosure = GalerkinSourceErrorEnclosure(
        free_action_error_upper_bound=bounds[0],
        cap_action_error_upper_bound=bounds[1],
        matched_source_error_upper_bound=bounds[2],
        interaction_action_error_upper_bound=bounds[3],
        total_source_error_upper_bound=bounds[4],
        scattered_source_error_upper_bound=bounds[5],
        independent_direct_cap_action=checked_direct_cap,
        independent_direct_interaction_action=checked_direct_interaction,
        incident_field_norm_upper_bound=bounds[6],
        free_target_transfer_error_upper_bound=bounds[7],
        cap_target_transfer_error_upper_bound=bounds[8],
        interaction_target_transfer_error_upper_bound=bounds[9],
        exact_target_matched_source_error_upper_bound=bounds[10],
        exact_target_total_source_error_upper_bound=bounds[11],
        exact_target_scattered_source_error_upper_bound=bounds[12],
        arithmetic_environment_supported=environment,
        gradual_underflow_supported=gradual,
        finite_certificate=finite_certificate,
        route=route,
        exact_action_target=_EXACT_ACTION_TARGET,
        error_scope=_SOURCE_ERROR_SCOPE,
        arithmetic_environment=_ARITHMETIC_ENVIRONMENT,
    )
    return enclosure


@jaxtyped(typechecker=beartype)
def create_galerkin_source_modes(  # noqa: PLR0913
    aperture_weights: Complex[Array, "..."],
    phased_coefficients: Complex[Array, "..."],
    incident_field: Complex[Array, "..."],
    physical_wavevectors: Float[Array, "..."],
    shell_defects: Float[Array, "..."],
    exact_free_diagonal_lower_bounds: Float[Array, "..."],
    exact_free_diagonal_upper_bounds: Float[Array, "..."],
    exact_normal_wavevector_lower_bounds: Float[Array, "..."],
    exact_normal_wavevector_upper_bounds: Float[Array, "..."],
    active_mask: Bool[Array, "..."],
    forward_mask: Bool[Array, "..."],
    grazing_mask: Bool[Array, "..."],
    backward_mask: Bool[Array, "..."],
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    source_plane_coordinate: scalar_float,
    shell_defect_tolerance: scalar_float,
    aperture_reduced_flux: scalar_float,
    input_reduced_flux: scalar_float,
    target_reduced_flux: scalar_float,
    output_reduced_flux: scalar_float,
    flux_normalization: scalar_float,
    exact_reduced_flux_lower_bound: scalar_float,
    exact_reduced_flux_upper_bound: scalar_float,
    target_reduced_flux_discrepancy_upper_bound: scalar_float,
    *,
    normal_axis: GalerkinSourceAxis,
    phase_convention: GalerkinSourcePhaseConvention,
    stored_shell_route: GalerkinStoredShellRoute,
) -> GalerkinSourceModes:
    """Create validated exact represented-mode and flux data.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Parameters
    ----------
    aperture_weights : Complex[Array, "..."]
        Supplied pre-phase SC.13 state coefficients.
    phased_coefficients : Complex[Array, "..."]
        Phase-applied coefficients before flux normalization.
    incident_field : Complex[Array, "..."]
        Common-normalized source coefficients.
    physical_wavevectors : Float[Array, "..."]
        Physical wavevectors with shape ``(n, 3)``.
    shell_defects : Float[Array, "..."]
        Target free-shell defects with shape ``(n,)``.
    exact_free_diagonal_lower_bounds : Float[Array, "..."]
        RM-S2 exact free-defect lower endpoints with shape ``(n,)``.
    exact_free_diagonal_upper_bounds : Float[Array, "..."]
        RM-S2 exact free-defect upper endpoints with shape ``(n,)``.
    exact_normal_wavevector_lower_bounds : Float[Array, "..."]
        Exact-target normal-wavevector lower endpoints with shape ``(n,)``.
    exact_normal_wavevector_upper_bounds : Float[Array, "..."]
        Exact-target normal-wavevector upper endpoints with shape ``(n,)``.
    active_mask : Bool[Array, "..."]
        Nonzero aperture-mode mask.
    forward_mask : Bool[Array, "..."]
        Positive-normal-wavenumber mask.
    grazing_mask : Bool[Array, "..."]
        Zero-normal-wavenumber mask.
    backward_mask : Bool[Array, "..."]
        Negative-normal-wavenumber mask.
    scan_position : Float[Array, "..."]
        Full physical scan vector; its normal component must be zero.
    aberration_phases : Float[Array, "..."]
        Per-state aberration phases in radians.
    source_plane_coordinate : scalar_float
        Physical source/reference-plane coordinate.
    shell_defect_tolerance : scalar_float
        Exact binary64 zero for this first represented-shell slice.
    aperture_reduced_flux : scalar_float
        Reduced flux before phases.
    input_reduced_flux : scalar_float
        Reduced flux after phases and before normalization.
    target_reduced_flux : scalar_float
        Declared positive output reduced flux.
    output_reduced_flux : scalar_float
        Recomputed normalized reduced flux.
    flux_normalization : scalar_float
        Common positive normalization factor.
    exact_reduced_flux_lower_bound : scalar_float
        Exact-carrier reduced-flux lower endpoint.
    exact_reduced_flux_upper_bound : scalar_float
        Exact-carrier reduced-flux upper endpoint.
    target_reduced_flux_discrepancy_upper_bound : scalar_float
        Outward requested-to-exact-carrier flux discrepancy.
    normal_axis : GalerkinSourceAxis
        Static positive coordinate-axis normal.
    phase_convention : GalerkinSourcePhaseConvention
        Static explicit phase convention.
    stored_shell_route : GalerkinStoredShellRoute
        Static stored-target shell-evidence route.

    Returns
    -------
    modes : GalerkinSourceModes
        Validated modes, geometry, masks, and flux values.

    Raises
    ------
    ValueError
        If array ranks or sizes are inconsistent.
    equinox.EquinoxRuntimeError
        If values, masks, shell membership, or fluxes are invalid.
    """
    aperture: Complex128[Array, " n"] = jnp.asarray(
        aperture_weights,
        dtype=jnp.complex128,
    )
    phased: Complex128[Array, " n"] = jnp.asarray(
        phased_coefficients,
        dtype=jnp.complex128,
    )
    incident: Complex128[Array, " n"] = jnp.asarray(
        incident_field,
        dtype=jnp.complex128,
    )
    wavevectors: Float64[Array, "n 3"] = jnp.asarray(
        physical_wavevectors,
        dtype=jnp.float64,
    )
    defects: Float64[Array, " n"] = jnp.asarray(
        shell_defects,
        dtype=jnp.float64,
    )
    exact_free_lower = jnp.asarray(
        exact_free_diagonal_lower_bounds, dtype=jnp.float64
    )
    exact_free_upper = jnp.asarray(
        exact_free_diagonal_upper_bounds, dtype=jnp.float64
    )
    exact_normal_lower = jnp.asarray(
        exact_normal_wavevector_lower_bounds, dtype=jnp.float64
    )
    exact_normal_upper = jnp.asarray(
        exact_normal_wavevector_upper_bounds, dtype=jnp.float64
    )
    scan: Float64[Array, " 3"] = jnp.asarray(
        scan_position,
        dtype=jnp.float64,
    )
    aberrations: Float64[Array, " n"] = jnp.asarray(
        aberration_phases,
        dtype=jnp.float64,
    )
    active: Bool[Array, " n"] = jnp.asarray(active_mask, dtype=jnp.bool_)
    forward: Bool[Array, " n"] = jnp.asarray(forward_mask, dtype=jnp.bool_)
    grazing: Bool[Array, " n"] = jnp.asarray(grazing_mask, dtype=jnp.bool_)
    backward: Bool[Array, " n"] = jnp.asarray(
        backward_mask,
        dtype=jnp.bool_,
    )
    size: int = aperture.shape[0] if aperture.ndim == 1 else -1
    for values, name in (
        (aperture, "aperture_weights"),
        (phased, "phased_coefficients"),
        (incident, "incident_field"),
        (defects, "shell_defects"),
        (exact_free_lower, "exact_free_diagonal_lower_bounds"),
        (exact_free_upper, "exact_free_diagonal_upper_bounds"),
        (exact_normal_lower, "exact_normal_wavevector_lower_bounds"),
        (exact_normal_upper, "exact_normal_wavevector_upper_bounds"),
        (aberrations, "aberration_phases"),
        (active, "active_mask"),
        (forward, "forward_mask"),
        (grazing, "grazing_mask"),
        (backward, "backward_mask"),
    ):
        _raise_if(values.ndim != 1, f"{name} must be 1D")
        _raise_if(values.shape[0] != size, f"{name} must have length {size}")
    _raise_if(size <= 0, "source mode arrays must be nonempty")
    _raise_if(
        wavevectors.shape != (size, 3),
        "physical_wavevectors must have shape (n, 3)",
    )
    _raise_if(scan.shape != (3,), "scan_position must have shape (3,)")

    coordinate: Float64[Array, ""] = jnp.asarray(
        source_plane_coordinate,
        dtype=jnp.float64,
    )
    tolerance: Float64[Array, ""] = _checked_bound(
        shell_defect_tolerance,
        "shell_defect_tolerance",
        finite=True,
    )
    aperture_flux: Float64[Array, ""] = _checked_bound(
        aperture_reduced_flux,
        "aperture_reduced_flux",
        finite=True,
    )
    input_flux: Float64[Array, ""] = _checked_bound(
        input_reduced_flux,
        "input_reduced_flux",
        finite=True,
    )
    target_flux: Float64[Array, ""] = _checked_bound(
        target_reduced_flux,
        "target_reduced_flux",
        finite=True,
    )
    output_flux: Float64[Array, ""] = _checked_bound(
        output_reduced_flux,
        "output_reduced_flux",
        finite=True,
    )
    normalization: Float64[Array, ""] = _checked_bound(
        flux_normalization,
        "flux_normalization",
        finite=True,
    )
    exact_flux_lower: Float64[Array, ""] = jnp.asarray(
        exact_reduced_flux_lower_bound, dtype=jnp.float64
    )
    exact_flux_upper: Float64[Array, ""] = jnp.asarray(
        exact_reduced_flux_upper_bound, dtype=jnp.float64
    )
    flux_discrepancy: Float64[Array, ""] = _checked_bound(
        target_reduced_flux_discrepancy_upper_bound,
        "target_reduced_flux_discrepancy_upper_bound",
        finite=False,
    )
    _raise_if(exact_flux_lower.shape != (), "exact flux lower must be scalar")
    _raise_if(exact_flux_upper.shape != (), "exact flux upper must be scalar")
    _raise_if(coordinate.shape != (), "source_plane_coordinate must be scalar")
    target_flux_difference = _interval_subtract(
        _point_interval(target_flux),
        (exact_flux_lower, exact_flux_upper),
    )
    required_flux_discrepancy: Float64[Array, ""] = jnp.maximum(
        jnp.abs(target_flux_difference[0]),
        jnp.abs(target_flux_difference[1]),
    )

    nonzero_aperture: Bool[Array, " n"] = (jnp.real(aperture) != 0.0) | (
        jnp.imag(aperture) != 0.0
    )
    normal_components: Float64[Array, " n"] = wavevectors[:, int(normal_axis)]
    exact_forward: Bool[Array, " n"] = normal_components > 0.0
    exact_grazing: Bool[Array, " n"] = normal_components == 0.0
    exact_backward: Bool[Array, " n"] = normal_components < 0.0
    masks_partition: Bool[Array, ""] = jnp.all(
        forward.astype(jnp.int8)
        + grazing.astype(jnp.int8)
        + backward.astype(jnp.int8)
        == 1
    )
    shell_bad: Bool[Array, ""] = (tolerance != 0.0) | jnp.any(
        active & (defects != 0.0)
    )

    checked_aperture: Complex128[Array, " n"] = eqx.error_if(
        aperture,
        jnp.any(~jnp.isfinite(aperture))
        | jnp.any(~jnp.isfinite(phased))
        | jnp.any(~jnp.isfinite(incident))
        | has_subnormal_components(aperture)
        | has_subnormal_components(phased)
        | has_subnormal_components(incident)
        | jnp.any(~jnp.isfinite(wavevectors))
        | jnp.any(~jnp.isfinite(defects))
        | jnp.any(jnp.isnan(exact_free_lower))
        | jnp.any(jnp.isnan(exact_free_upper))
        | jnp.any(jnp.isnan(exact_normal_lower))
        | jnp.any(jnp.isnan(exact_normal_upper))
        | jnp.any(exact_free_lower > exact_free_upper)
        | jnp.any(exact_normal_lower > exact_normal_upper)
        | jnp.isnan(exact_flux_lower)
        | jnp.isnan(exact_flux_upper)
        | (exact_flux_lower > exact_flux_upper)
        | (flux_discrepancy < required_flux_discrepancy)
        | jnp.any(~jnp.isfinite(scan))
        | jnp.any(~jnp.isfinite(aberrations))
        | (~jnp.isfinite(coordinate))
        | (scan[int(normal_axis)] != 0.0)
        | jnp.any(active != nonzero_aperture)
        | (~masks_partition)
        | jnp.any(forward != exact_forward)
        | jnp.any(grazing != exact_grazing)
        | jnp.any(backward != exact_backward)
        | jnp.any(active & ~forward)
        | shell_bad
        | (aperture_flux <= 0.0)
        | (input_flux <= 0.0)
        | (target_flux <= 0.0)
        | (output_flux <= 0.0)
        | (normalization <= 0.0),
        "represented source modes must be finite, forward, on shell, "
        "consistently "
        "masked, transversely scanned, and positive-flux normalized",
    )
    modes: GalerkinSourceModes = GalerkinSourceModes(
        aperture_weights=checked_aperture,
        phased_coefficients=phased,
        incident_field=incident,
        physical_wavevectors=wavevectors,
        shell_defects=defects,
        exact_free_diagonal_lower_bounds=exact_free_lower,
        exact_free_diagonal_upper_bounds=exact_free_upper,
        exact_normal_wavevector_lower_bounds=exact_normal_lower,
        exact_normal_wavevector_upper_bounds=exact_normal_upper,
        active_mask=active,
        forward_mask=forward,
        grazing_mask=grazing,
        backward_mask=backward,
        scan_position=scan,
        aberration_phases=aberrations,
        source_plane_coordinate=coordinate,
        shell_defect_tolerance=tolerance,
        aperture_reduced_flux=aperture_flux,
        input_reduced_flux=input_flux,
        target_reduced_flux=target_flux,
        output_reduced_flux=output_flux,
        flux_normalization=normalization,
        exact_reduced_flux_lower_bound=exact_flux_lower,
        exact_reduced_flux_upper_bound=exact_flux_upper,
        target_reduced_flux_discrepancy_upper_bound=flux_discrepancy,
        normal_axis=normal_axis,
        phase_convention=phase_convention,
        stored_shell_route=stored_shell_route,
    )
    return modes


@jaxtyped(typechecker=beartype)
def create_galerkin_source_actions(
    free_action: Complex[Array, "..."],
    cap_action: Complex[Array, "..."],
    interaction_action: Complex[Array, "..."],
    incident_source: Complex[Array, "..."],
    additional_source: Complex[Array, "..."],
    total_source: Complex[Array, "..."],
    scattered_source: Complex[Array, "..."],
) -> GalerkinSourceActions:
    """Create validated finite source-action vectors.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Parameters
    ----------
    free_action : Complex[Array, "..."]
        Shifted finite free action ``Dv``.
    cap_action : Complex[Array, "..."]
        Physical CAP action ``Bv``.
    interaction_action : Complex[Array, "..."]
        Represented interaction action ``Rv``.
    incident_source : Complex[Array, "..."]
        Matched incident source ``Dv - iBv``.
    additional_source : Complex[Array, "..."]
        Separately declared additional source.
    total_source : Complex[Array, "..."]
        Complete total-field source.
    scattered_source : Complex[Array, "..."]
        Equivalent scattered-field source.

    Returns
    -------
    actions : GalerkinSourceActions
        Validated finite source vectors.

    Raises
    ------
    ValueError
        If vectors are empty or have inconsistent ranks or lengths.
    equinox.EquinoxRuntimeError
        If a vector is non-finite or contains a nonzero subnormal component.
    """
    raw_values = (
        free_action,
        cap_action,
        interaction_action,
        incident_source,
        additional_source,
        total_source,
        scattered_source,
    )
    first: Complex128[Array, " n"] = jnp.asarray(
        raw_values[0],
        dtype=jnp.complex128,
    )
    _raise_if(first.ndim != 1, "source action vectors must be 1D")
    _raise_if(first.shape[0] == 0, "source action vectors must be nonempty")
    size: int = first.shape[0]
    names = (
        "free_action",
        "cap_action",
        "interaction_action",
        "incident_source",
        "additional_source",
        "total_source",
        "scattered_source",
    )
    values: Tuple[Complex128[Array, " n"], ...] = tuple(
        _checked_vector(value, size, name)
        for value, name in zip(raw_values, names, strict=True)
    )
    actions: GalerkinSourceActions = GalerkinSourceActions(
        free_action=values[0],
        cap_action=values[1],
        interaction_action=values[2],
        incident_source=values[3],
        additional_source=values[4],
        total_source=values[5],
        scattered_source=values[6],
    )
    return actions


@jaxtyped(typechecker=beartype)
def create_represented_galerkin_source(
    manifest: GalerkinTargetManifest,
    modes: GalerkinSourceModes,
    actions: GalerkinSourceActions,
    representation_ledger: GalerkinSourceRepresentationLedger,
    error_enclosure: GalerkinSourceErrorEnclosure,
    *,
    kind: GalerkinRepresentedSourceKind,
) -> GalerkinRepresentedSource:
    """Bind source evidence to one manifested scalar target.

    :see: :class:`~.test_source_types.TestGalerkinSourceVocabulary`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical scalar target and state support.
    modes : GalerkinSourceModes
        Represented mode, mask, geometry, and flux carrier.
    actions : GalerkinSourceActions
        Finite source-action carrier.
    representation_ledger : GalerkinSourceRepresentationLedger
        Complete six-term representation ledger.
    error_enclosure : GalerkinSourceErrorEnclosure
        Numerical action and source error enclosures.
    kind : GalerkinRepresentedSourceKind
        Static plane or coherent-focused construction kind.

    Returns
    -------
    source : GalerkinRepresentedSource
        Bound represented-source carrier with explicit noneligibility state.

    Raises
    ------
    ValueError
        If mode or action lengths do not match the target state support.
    equinox.EquinoxRuntimeError
        If the active-mode count or transverse-fiber predicate fails.
    """
    size: int = manifest.support.state_indices.shape[0]
    _raise_if(
        modes.incident_field.shape != (size,),
        "source modes must match the target state support",
    )
    _raise_if(
        actions.incident_source.shape != (size,),
        "source actions must match the target state support",
    )
    _raise_if(
        error_enclosure.independent_direct_cap_action.shape != (size,),
        "source-error actions must match the target state support",
    )
    _raise_if(
        error_enclosure.independent_direct_interaction_action.shape != (size,),
        "source-error actions must match the target state support",
    )
    normal_axis: int = int(modes.normal_axis)
    transverse_indices: Tuple[int, int] = _TRANSVERSE_AXES[normal_axis]
    transverse: Array = manifest.support.state_indices[:, transverse_indices]
    same_fiber: Bool[Array, "n n"] = jnp.all(
        transverse[:, None, :] == transverse[None, :, :],
        axis=-1,
    )
    distinct_pairs: Bool[Array, "n n"] = ~jnp.eye(size, dtype=jnp.bool_)
    active_pairs: Bool[Array, "n n"] = (
        modes.active_mask[:, None] & modes.active_mask[None, :]
    )
    duplicate_fiber: Bool[Array, ""] = jnp.any(
        same_fiber & distinct_pairs & active_pairs
    )
    active_count = jnp.sum(modes.active_mask.astype(jnp.int32))
    if kind is GalerkinRepresentedSourceKind.PLANE_MODE:
        invalid_count: Bool[Array, ""] = active_count != 1
    else:
        invalid_count = active_count < _MIN_FOCUSED_MODES
    checked_modes_field: Complex128[Array, " n"] = eqx.error_if(
        modes.incident_field,
        invalid_count | duplicate_fiber,
        "source kind and active modes must have the required count and one "
        "normal branch per transverse harmonic",
    )
    checked_modes: GalerkinSourceModes = eqx.tree_at(
        lambda value: value.incident_field,
        modes,
        checked_modes_field,
    )
    acquisition = manifest.support_eligibility
    support_eligible: Bool[Array, ""] = (
        acquisition.status
        == int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
    ) & acquisition.support_eligible
    incident_matches: Bool[Array, "n i"] = jnp.all(
        manifest.support.state_indices[:, None, :]
        == manifest.acquisition.incident_indices[None, :, :],
        axis=-1,
    )
    exact_dispositions: Bool[Array, " i"] = (
        manifest.acquisition.incident_direction_dispositions
        == int(GalerkinDirectionDisposition.EXACT_COEFFICIENT)
    )
    declared_exact: Bool[Array, " n"] = jnp.any(
        incident_matches & exact_dispositions[None, :], axis=1
    )
    declared_incident_eligible: Bool[Array, ""] = jnp.all(
        (~modes.active_mask) | declared_exact
    )
    exact_shell_eligible: Bool[Array, ""] = jnp.all(
        (~modes.active_mask)
        | (
            (modes.exact_free_diagonal_lower_bounds == 0.0)
            & (modes.exact_free_diagonal_upper_bounds == 0.0)
        )
    )
    orientation_eligible: bool = (
        normal_axis == manifest.acquisition.terminal_axis
        and manifest.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
    )
    exact_flux_eligible: Bool[Array, ""] = (
        jnp.asarray(orientation_eligible)
        & jnp.all(
            (~modes.active_mask)
            | (modes.exact_normal_wavevector_lower_bounds > 0.0)
        )
        & jnp.isfinite(modes.exact_reduced_flux_lower_bound)
        & jnp.isfinite(modes.exact_reduced_flux_upper_bound)
        & jnp.isfinite(modes.target_reduced_flux_discrepancy_upper_bound)
        & (modes.exact_reduced_flux_lower_bound > 0.0)
    )
    action_identities: Bool[Array, ""] = (
        jnp.all(
            actions.incident_source
            == actions.free_action - 1j * actions.cap_action
        )
        & jnp.all(
            actions.total_source
            == actions.incident_source + actions.additional_source
        )
        & jnp.all(
            actions.scattered_source
            == actions.interaction_action + actions.additional_source
        )
    )
    action_enclosures_eligible: Bool[Array, ""] = (
        error_enclosure.finite_certificate & action_identities
    )
    representation_exact: Bool[Array, ""] = jnp.all(
        jnp.stack(
            (
                representation_ledger.box_error_upper_bound,
                representation_ledger.carrier_error_upper_bound,
                representation_ledger.window_error_upper_bound,
                representation_ledger.preband_error_upper_bound,
                representation_ledger.band_error_upper_bound,
                representation_ledger.algebraic_error_upper_bound,
            )
        )
        == 0.0
    ) & jnp.asarray(
        representation_ledger.route
        is GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET
    )
    rm_s3_eligible: Bool[Array, ""] = (
        support_eligible
        & declared_incident_eligible
        & exact_shell_eligible
        & exact_flux_eligible
        & action_enclosures_eligible
        & representation_exact
    )
    stopped_eligibility = jax.tree.map(
        jax.lax.stop_gradient,
        (
            support_eligible,
            declared_incident_eligible,
            exact_shell_eligible,
            exact_flux_eligible,
            action_enclosures_eligible,
            rm_s3_eligible,
        ),
    )
    source: GalerkinRepresentedSource = GalerkinRepresentedSource(
        manifest=manifest,
        modes=checked_modes,
        actions=actions,
        representation_ledger=representation_ledger,
        error_enclosure=error_enclosure,
        support_eligible=stopped_eligibility[0],
        declared_incident_eligible=stopped_eligibility[1],
        exact_shell_eligible=stopped_eligibility[2],
        exact_flux_eligible=stopped_eligibility[3],
        action_enclosures_eligible=stopped_eligibility[4],
        rm_s3_eligible=stopped_eligibility[5],
        kind=kind,
        eligibility_scope=_ELIGIBILITY_SCOPE,
    )
    return source


__all__: list[str] = [
    "GalerkinRepresentedSource",
    "GalerkinRepresentedSourceKind",
    "GalerkinSourceActions",
    "GalerkinSourceAxis",
    "GalerkinSourceErrorEnclosure",
    "GalerkinSourceErrorRoute",
    "GalerkinSourceModes",
    "GalerkinSourcePhaseConvention",
    "GalerkinSourceRepresentationLedger",
    "GalerkinSourceRepresentationRoute",
    "GalerkinStoredShellRoute",
    "create_galerkin_source_actions",
    "create_galerkin_source_error_enclosure",
    "create_galerkin_source_ledger",
    "create_galerkin_source_modes",
    "create_represented_galerkin_source",
]
