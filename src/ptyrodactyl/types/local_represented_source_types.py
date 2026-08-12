r"""Define disjoint represented-source carriers for ``LOCAL_CELL_LVT1``.

Extended Summary
----------------
The carriers in this leaf bind one finite represented incident field to one
fully replayed local-cell target and one manifested LVT.20 additional-source
certificate.  They keep the vacuum-matched, total, and scattered algebraic
branches distinct and store direct exact-target rectangles for every
``D/B/R/S/M/T/C`` component without importing a legacy source carrier.

Routine Listings
----------------
:class:`GalerkinLocalComplexRectangles`
    Store componentwise outward complex rectangles on ordered ``I_u``.
:class:`GalerkinLocalRepresentedSource`
    Bind incident modes, algebraic actions, and both authenticated parents.
:class:`GalerkinLocalRepresentedSourceActions`
    Store the seven frozen ``D/B/R/S/M/T/C`` algebraic vectors.
:class:`GalerkinLocalRepresentedSourceCertificate`
    Store direct exact-target action rectangles and disjoint error bounds.
:class:`GalerkinLocalRepresentedSourceFailure`
    Store one typed represented-source or direct-certificate outcome.
:class:`GalerkinLocalRepresentedSourceKind`
    Select one plane or coherent-focused represented incident construction.
:class:`GalerkinLocalRepresentedSourceModes`
    Store phase, exact-shell, branch, and reduced-flux evidence.
:class:`GalerkinLocalSourceAxis`
    Select the positive coordinate-aligned source normal.
:class:`GalerkinLocalSourcePhaseConvention`
    Select the sole admitted physical-wavevector phase convention.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex128,
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import (
    RootEnclosureError,
    fraction_from_float,
    fraction_upper_float,
    has_subnormal_components,
    sqrt_fraction_upper,
    stored_value_payload,
)

from .local_cell_target_types import GalerkinLocalCellTargetManifest
from .local_source_types import GalerkinLocalAdditionalSourceCertificate

_SHA256_HEX_LENGTH: int = 64
_ACTION_COUNT: int = 7


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise for one structural represented-source failure.

    Parameters
    ----------
    condition : bool
        Whether the structural failure is present.
    message : str
        Error message for the failed invariant.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 text value.

    Parameters
    ----------
    value : str
        Candidate digest text.

    Returns
    -------
    valid : bool
        Whether the value is one canonical lowercase SHA-256 digest.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


class GalerkinLocalRepresentedSourceKind(str, Enum):
    """Select one plane or coherent-focused represented incident construction.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_routes_axes_phases_and_failures_are_disjoint`
    """

    PLANE_MODE = "local_represented_plane_mode"
    COHERENT_FOCUSED = "local_represented_coherent_focused"


class GalerkinLocalSourceAxis(IntEnum):
    """Select the positive coordinate-aligned source normal.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_routes_axes_phases_and_failures_are_disjoint`
    """

    X = 0
    Y = 1
    Z = 2


class GalerkinLocalSourcePhaseConvention(str, Enum):
    """Select the sole admitted physical-wavevector phase convention.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_routes_axes_phases_and_failures_are_disjoint`
    """

    PHYSICAL_WAVEVECTOR = "local_physical_kappa_scan_source_plus_aberration"


class GalerkinLocalRepresentedSourceFailure(str, Enum):
    """Store one typed represented-source or direct-certificate outcome.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_routes_axes_phases_and_failures_are_disjoint`
    """

    NONE = "none"
    ADDITIONAL_SOURCE_NONCERTIFICATE = "additional_source_noncertificate"
    TERMINAL_ORIENTATION_UNSUPPORTED = "terminal_orientation_unsupported"
    UNDECLARED_INCIDENT_MODE = "undeclared_incident_mode"
    NONEXACT_INCIDENT_DISPOSITION = "nonexact_incident_disposition"
    EXACT_SHELL_FAILURE = "exact_shell_failure"
    NONFORWARD_OR_GRAZING = "nonforward_or_grazing"
    DUPLICATE_TRANSVERSE_FIBER = "duplicate_transverse_fiber"
    NONPOSITIVE_EXACT_FLUX = "nonpositive_exact_flux"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    DIRECT_WORK_BUDGET_EXCEEDED = "direct_work_budget_exceeded"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalComplexRectangles(NamedTuple):
    """Store componentwise outward complex rectangles on ordered ``I_u``.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_carriers_own_direct_dbrsmtc_evidence_only`
    """

    real_lower_bounds: Float64[Array, " n"]
    real_upper_bounds: Float64[Array, " n"]
    imag_lower_bounds: Float64[Array, " n"]
    imag_upper_bounds: Float64[Array, " n"]


class GalerkinLocalRepresentedSourceModes(NamedTuple):
    """Store phase, exact-shell, branch, and reduced-flux evidence.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_carriers_own_direct_dbrsmtc_evidence_only`
    """

    aperture_weights: Complex128[Array, " n"]
    phased_coefficients: Complex128[Array, " n"]
    incident_field: Complex128[Array, " n"]
    algebraic_physical_wavevectors: Float64[Array, "n 3"]
    exact_normal_wavevector_lower_bounds: Float64[Array, " n"]
    exact_normal_wavevector_upper_bounds: Float64[Array, " n"]
    active_mask: Bool[Array, " n"]
    forward_mask: Bool[Array, " n"]
    grazing_mask: Bool[Array, " n"]
    backward_mask: Bool[Array, " n"]
    declared_incident_mask: Bool[Array, " n"]
    exact_incident_disposition_mask: Bool[Array, " n"]
    exact_shell_mask: Bool[Array, " n"]
    exact_forward_mask: Bool[Array, " n"]
    scan_position: Float64[Array, " 3"]
    aberration_phases: Float64[Array, " n"]
    source_plane_coordinate: Float64[Array, ""]
    aperture_reduced_flux: Float64[Array, ""]
    input_reduced_flux: Float64[Array, ""]
    target_reduced_flux: Float64[Array, ""]
    output_reduced_flux: Float64[Array, ""]
    flux_normalization: Float64[Array, ""]
    exact_reduced_flux_lower_bound: Float64[Array, ""]
    exact_reduced_flux_upper_bound: Float64[Array, ""]
    target_reduced_flux_discrepancy_upper_bound: Float64[Array, ""]


class GalerkinLocalRepresentedSourceActions(NamedTuple):
    """Store the seven frozen ``D/B/R/S/M/T/C`` algebraic vectors.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_carriers_own_direct_dbrsmtc_evidence_only`
    """

    free_action: Complex128[Array, " n"]
    physical_cap_action: Complex128[Array, " n"]
    interaction_action: Complex128[Array, " n"]
    additional_source: Complex128[Array, " n"]
    vacuum_matched_source: Complex128[Array, " n"]
    total_source: Complex128[Array, " n"]
    scattered_source: Complex128[Array, " n"]


class GalerkinLocalRepresentedSource(eqx.Module):
    r"""Bind incident modes, algebraic actions, and both authenticated parents.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_carriers_own_direct_dbrsmtc_evidence_only`

    ``total_source`` is the projected LVT.20 lift
    ``D_alg v - i B_alg v + S_add,alg``.  ``vacuum_matched_source`` and
    ``scattered_source`` remain separate so a later residual checker can use
    the correct exact-target source enclosure once and only once.
    """

    target: GalerkinLocalCellTargetManifest
    additional_source_certificate: GalerkinLocalAdditionalSourceCertificate
    modes: GalerkinLocalRepresentedSourceModes
    actions: GalerkinLocalRepresentedSourceActions
    incident_eligible: Bool[Array, ""]
    kind: GalerkinLocalRepresentedSourceKind = eqx.field(static=True)
    normal_axis: GalerkinLocalSourceAxis = eqx.field(static=True)
    phase_convention: GalerkinLocalSourcePhaseConvention = eqx.field(
        static=True
    )
    failure: GalerkinLocalRepresentedSourceFailure = eqx.field(static=True)
    local_source_lift_formula: str = eqx.field(static=True)
    projected_lift_formula: str = eqx.field(static=True)
    vacuum_matched_formula: str = eqx.field(static=True)
    total_source_formula: str = eqx.field(static=True)
    scattered_source_formula: str = eqx.field(static=True)
    eligibility_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    additional_source_digest: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    source_evidence_digest: str = eqx.field(static=True)
    source_name: str = eqx.field(static=True)


class GalerkinLocalRepresentedSourceCertificate(eqx.Module):
    r"""Store direct exact-target action rectangles and disjoint error bounds.

    :see: :func:`~.test_local_represented_source_types.\
test_local_represented_carriers_own_direct_dbrsmtc_evidence_only`

    The matched, total, and scattered bounds are direct point-to-full-
    rectangle reductions.  The component bounds are audit evidence and are
    never summed into a branch bound or combined with the parent target's
    ``delta_D``, ``delta_R``, ``delta_B``, or ``delta_H``.

    ``finite_certificate`` is public forgeable storage.  It becomes
    authoritative only after complete reconstruction by
    ``prepare_local_represented_source_certificate``.
    """

    source: GalerkinLocalRepresentedSource
    free_rectangles: GalerkinLocalComplexRectangles
    physical_cap_rectangles: GalerkinLocalComplexRectangles
    interaction_rectangles: GalerkinLocalComplexRectangles
    additional_source_rectangles: GalerkinLocalComplexRectangles
    vacuum_matched_rectangles: GalerkinLocalComplexRectangles
    total_source_rectangles: GalerkinLocalComplexRectangles
    scattered_source_rectangles: GalerkinLocalComplexRectangles
    free_component_error_bounds: Float64[Array, " n"]
    physical_cap_component_error_bounds: Float64[Array, " n"]
    interaction_component_error_bounds: Float64[Array, " n"]
    additional_source_component_error_bounds: Float64[Array, " n"]
    vacuum_matched_component_error_bounds: Float64[Array, " n"]
    total_source_component_error_bounds: Float64[Array, " n"]
    scattered_source_component_error_bounds: Float64[Array, " n"]
    free_action_error_upper_bound: Float64[Array, ""]
    physical_cap_action_error_upper_bound: Float64[Array, ""]
    interaction_action_error_upper_bound: Float64[Array, ""]
    additional_source_error_upper_bound: Float64[Array, ""]
    vacuum_matched_source_error_upper_bound: Float64[Array, ""]
    total_source_error_upper_bound: Float64[Array, ""]
    scattered_source_error_upper_bound: Float64[Array, ""]
    incident_field_norm_upper_bound: Float64[Array, ""]
    finite_certificate: Bool[Array, ""]
    direct_pair_count: Int64[Array, ""]
    maximum_direct_pairs: Int64[Array, ""]
    failure: GalerkinLocalRepresentedSourceFailure = eqx.field(static=True)
    exact_target: str = eqx.field(static=True)
    arithmetic: str = eqx.field(static=True)
    direct_pair_count_route: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    parent_source_evidence_digest: str = eqx.field(static=True)
    parent_additional_certificate_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


def _rectangles_are_finite(
    rectangles: GalerkinLocalComplexRectangles,
) -> Bool[Array, ""]:
    """PRIVATE: Check finite ordered endpoints for one rectangle group.

    Parameters
    ----------
    rectangles : GalerkinLocalComplexRectangles
        Candidate componentwise complex rectangles.

    Returns
    -------
    valid : Bool[Array, ""]
        Whether endpoints are finite, ordered, and normal-or-zero.
    """
    arrays: Tuple[Float64[Array, " n"], ...] = tuple(rectangles)
    valid: Bool[Array, ""] = (
        jnp.all(
            jnp.asarray(
                [
                    jnp.all(jnp.isfinite(values))
                    & ~has_subnormal_components(values)
                    for values in arrays
                ]
            )
        )
        & jnp.all(rectangles.real_lower_bounds <= rectangles.real_upper_bounds)
        & jnp.all(rectangles.imag_lower_bounds <= rectangles.imag_upper_bounds)
    )
    return valid


def _host_nonnegative_norm_upper(
    values: Float64[Array, " n"],
) -> float:
    """PRIVATE: Bound one stored non-negative vector norm exactly on host.

    Parameters
    ----------
    values : Float64[Array, " n"]
        Finite non-negative binary64 component bounds.

    Returns
    -------
    upper : float
        Outward binary64 Euclidean norm upper bound.

    Raises
    ------
    RootEnclosureError
        If the verified rational square-root enclosure fails.
    """
    host = np.asarray(jax.device_get(values), dtype=np.float64)
    squared = sum(
        (fraction_from_float(float(value)) ** 2 for value in host),
        start=fraction_from_float(0.0),
    )
    upper: float = fraction_upper_float(sqrt_fraction_upper(squared))
    return upper


def _host_complex_norm_upper(
    values: Complex128[Array, " n"],
) -> float:
    """PRIVATE: Bound one exact stored complex-vector norm on host.

    Parameters
    ----------
    values : Complex128[Array, " n"]
        Finite exact stored binary64 complex vector.

    Returns
    -------
    upper : float
        Outward binary64 Euclidean norm upper bound.

    Raises
    ------
    RootEnclosureError
        If the verified rational square-root enclosure fails.
    """
    host = np.asarray(jax.device_get(values), dtype=np.complex128)
    squared = sum(
        (
            fraction_from_float(float(value.real)) ** 2
            + fraction_from_float(float(value.imag)) ** 2
            for value in host
        ),
        start=fraction_from_float(0.0),
    )
    upper: float = fraction_upper_float(sqrt_fraction_upper(squared))
    return upper


@jaxtyped(typechecker=beartype)
def _make_local_represented_source(  # noqa: PLR0913, PLR0915
    target: GalerkinLocalCellTargetManifest,
    additional_source_certificate: GalerkinLocalAdditionalSourceCertificate,
    modes: GalerkinLocalRepresentedSourceModes,
    actions: GalerkinLocalRepresentedSourceActions,
    incident_eligible: Bool[Array, ""],
    *,
    kind: GalerkinLocalRepresentedSourceKind,
    normal_axis: GalerkinLocalSourceAxis,
    phase_convention: GalerkinLocalSourcePhaseConvention,
    failure: GalerkinLocalRepresentedSourceFailure,
    local_source_lift_formula: str,
    projected_lift_formula: str,
    vacuum_matched_formula: str,
    total_source_formula: str,
    scattered_source_formula: str,
    eligibility_scope: str,
    target_digest: str,
    additional_source_digest: str,
    source_digest: str,
    source_evidence_digest: str,
    source_name: str,
) -> GalerkinLocalRepresentedSource:
    """PRIVATE: Validate and store one represented local source.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.
    additional_source_certificate : GalerkinLocalAdditionalSourceCertificate
        Fully prepared LVT.20 additional-source result.
    modes : GalerkinLocalRepresentedSourceModes
        Incident phase, shell, branch, and flux evidence.
    actions : GalerkinLocalRepresentedSourceActions
        Frozen algebraic ``D/B/R/S/M/T/C`` vectors.
    incident_eligible : Bool[Array, ""]
        Whether every incident and parent evidence gate passed.
    kind : GalerkinLocalRepresentedSourceKind
        Plane or coherent-focused construction kind.
    normal_axis : GalerkinLocalSourceAxis
        Positive coordinate-aligned source normal.
    phase_convention : GalerkinLocalSourcePhaseConvention
        Explicit physical-wavevector phase convention.
    failure : GalerkinLocalRepresentedSourceFailure
        Typed incident-source outcome.
    local_source_lift_formula : str
        Exact pointwise LVT.20 source formula.
    projected_lift_formula : str
        Exact projection identity for the total source.
    vacuum_matched_formula : str
        Frozen vacuum-matched algebraic formula.
    total_source_formula : str
        Frozen total algebraic formula.
    scattered_source_formula : str
        Frozen scattered algebraic formula.
    eligibility_scope : str
        Explicit incident-source eligibility boundary.
    target_digest : str
        Bound target operator digest.
    additional_source_digest : str
        Bound exact additional-source identity digest.
    source_digest : str
        Represented-source identity digest.
    source_evidence_digest : str
        Full represented-source evidence digest.
    source_name : str
        Canonically stripped source name.

    Returns
    -------
    source : GalerkinLocalRepresentedSource
        Validated represented local source.

    Raises
    ------
    TypeError
        If a carrier or enum has the wrong type.
    """
    if not isinstance(target, GalerkinLocalCellTargetManifest):
        raise TypeError("target must be GalerkinLocalCellTargetManifest")
    if not isinstance(
        additional_source_certificate,
        GalerkinLocalAdditionalSourceCertificate,
    ):
        raise TypeError(
            "additional_source_certificate must be "
            "GalerkinLocalAdditionalSourceCertificate"
        )
    for value, expected, name in (
        (kind, GalerkinLocalRepresentedSourceKind, "kind"),
        (normal_axis, GalerkinLocalSourceAxis, "normal_axis"),
        (
            phase_convention,
            GalerkinLocalSourcePhaseConvention,
            "phase_convention",
        ),
        (failure, GalerkinLocalRepresentedSourceFailure, "failure"),
    ):
        if not isinstance(value, expected):
            raise TypeError(f"{name} has the wrong represented-source enum")
    size = target.state_indices.shape[0]
    mode_vectors = (
        modes.aperture_weights,
        modes.phased_coefficients,
        modes.incident_field,
        modes.exact_normal_wavevector_lower_bounds,
        modes.exact_normal_wavevector_upper_bounds,
        modes.active_mask,
        modes.forward_mask,
        modes.grazing_mask,
        modes.backward_mask,
        modes.declared_incident_mask,
        modes.exact_incident_disposition_mask,
        modes.exact_shell_mask,
        modes.exact_forward_mask,
        modes.aberration_phases,
    )
    _raise_if(
        any(values.shape != (size,) for values in mode_vectors),
        "represented mode evidence must match target I_u",
    )
    _raise_if(
        modes.algebraic_physical_wavevectors.shape != (size, 3)
        or modes.scan_position.shape != (3,),
        "represented wavevector and scan evidence has the wrong shape",
    )
    scalar_modes = (
        modes.source_plane_coordinate,
        modes.aperture_reduced_flux,
        modes.input_reduced_flux,
        modes.target_reduced_flux,
        modes.output_reduced_flux,
        modes.flux_normalization,
        modes.exact_reduced_flux_lower_bound,
        modes.exact_reduced_flux_upper_bound,
        modes.target_reduced_flux_discrepancy_upper_bound,
    )
    _raise_if(
        any(value.shape != () for value in scalar_modes),
        "represented flux and plane evidence must be scalar",
    )
    action_vectors: Tuple[Complex128[Array, " n"], ...] = tuple(actions)
    _raise_if(
        any(values.shape != (size,) for values in action_vectors),
        "represented actions must match target I_u",
    )
    eligible = jnp.asarray(incident_eligible, dtype=jnp.bool_)
    _raise_if(eligible.shape != (), "incident_eligible must be scalar")
    _raise_if(
        bool(eligible)
        != (failure is GalerkinLocalRepresentedSourceFailure.NONE),
        "incident eligibility and failure outcome are inconsistent",
    )
    expected_active = (jnp.real(modes.aperture_weights) != 0.0) | (
        jnp.imag(modes.aperture_weights) != 0.0
    )
    phased_active = (jnp.real(modes.phased_coefficients) != 0.0) | (
        jnp.imag(modes.phased_coefficients) != 0.0
    )
    incident_active = (jnp.real(modes.incident_field) != 0.0) | (
        jnp.imag(modes.incident_field) != 0.0
    )
    normal_components = modes.algebraic_physical_wavevectors[
        :, int(normal_axis)
    ]
    invalid_masks = (
        jnp.any(modes.active_mask != expected_active)
        | jnp.any(phased_active != expected_active)
        | jnp.any(incident_active != expected_active)
        | jnp.any(modes.forward_mask != (normal_components > 0.0))
        | jnp.any(modes.grazing_mask != (normal_components == 0.0))
        | jnp.any(modes.backward_mask != (normal_components < 0.0))
        | jnp.any(
            modes.exact_forward_mask
            != (modes.exact_normal_wavevector_lower_bounds > 0.0)
        )
    )
    invalid_intervals = (
        jnp.any(
            modes.exact_normal_wavevector_lower_bounds
            > modes.exact_normal_wavevector_upper_bounds
        )
        | (
            modes.exact_reduced_flux_lower_bound
            > modes.exact_reduced_flux_upper_bound
        )
        | (modes.target_reduced_flux_discrepancy_upper_bound < 0.0)
    )
    expected_additional = (
        additional_source_certificate.source.algebraic_additional_source
    )
    invalid_actions = (
        jnp.any(actions.additional_source != expected_additional)
        | jnp.any(
            actions.vacuum_matched_source
            != actions.free_action - 1j * actions.physical_cap_action
        )
        | jnp.any(
            actions.total_source
            != actions.vacuum_matched_source + actions.additional_source
        )
        | jnp.any(
            actions.scattered_source
            != actions.interaction_action + actions.additional_source
        )
    )
    all_arrays = (
        *mode_vectors[:5],
        modes.algebraic_physical_wavevectors,
        modes.scan_position,
        modes.aberration_phases,
        *scalar_modes,
        *action_vectors,
    )
    invalid_range = any(
        bool(jnp.any(~jnp.isfinite(values)))
        or bool(has_subnormal_components(values))
        for values in all_arrays
    )
    for text, name in (
        (local_source_lift_formula, "local_source_lift_formula"),
        (projected_lift_formula, "projected_lift_formula"),
        (vacuum_matched_formula, "vacuum_matched_formula"),
        (total_source_formula, "total_source_formula"),
        (scattered_source_formula, "scattered_source_formula"),
        (eligibility_scope, "eligibility_scope"),
        (source_name, "source_name"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (target_digest, "target_digest"),
        (additional_source_digest, "additional_source_digest"),
        (source_digest, "source_digest"),
        (source_evidence_digest, "source_evidence_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")
    _raise_if(target_digest != target.target_digest, "target digest mismatch")
    _raise_if(
        int(normal_axis) != target.acquisition.terminal_axis,
        "normal axis must be derived from the target terminal axis",
    )
    _raise_if(
        stored_value_payload(additional_source_certificate.source.target)
        != stored_value_payload(target),
        "additional-source certificate must bind the identical target",
    )
    _raise_if(
        additional_source_digest
        != additional_source_certificate.source.source_digest,
        "additional-source digest mismatch",
    )
    checked_incident: Complex128[Array, " n"] = eqx.error_if(
        modes.incident_field,
        invalid_masks | invalid_intervals | invalid_actions | invalid_range,
        "represented source contains invalid modes or algebraic actions",
    )
    checked_modes = modes._replace(incident_field=checked_incident)
    source: GalerkinLocalRepresentedSource = GalerkinLocalRepresentedSource(
        target=target,
        additional_source_certificate=additional_source_certificate,
        modes=checked_modes,
        actions=actions,
        incident_eligible=eligible,
        kind=kind,
        normal_axis=normal_axis,
        phase_convention=phase_convention,
        failure=failure,
        local_source_lift_formula=local_source_lift_formula.strip(),
        projected_lift_formula=projected_lift_formula.strip(),
        vacuum_matched_formula=vacuum_matched_formula.strip(),
        total_source_formula=total_source_formula.strip(),
        scattered_source_formula=scattered_source_formula.strip(),
        eligibility_scope=eligibility_scope.strip(),
        target_digest=target_digest,
        additional_source_digest=additional_source_digest,
        source_digest=source_digest,
        source_evidence_digest=source_evidence_digest,
        source_name=source_name.strip(),
    )
    return source


@jaxtyped(typechecker=beartype)
def _make_local_represented_source_certificate(  # noqa: PLR0913, PLR0915
    source: GalerkinLocalRepresentedSource,
    rectangles: Tuple[GalerkinLocalComplexRectangles, ...],
    component_error_bounds: Tuple[Float[Array, "..."], ...],
    action_error_upper_bounds: Tuple[Float[Array, ""], ...],
    incident_field_norm_upper_bound: Float[Array, ""],
    finite_certificate: Bool[Array, ""],
    direct_pair_count: Int[Array, ""],
    maximum_direct_pairs: Int[Array, ""],
    *,
    failure: GalerkinLocalRepresentedSourceFailure,
    exact_target: str,
    arithmetic: str,
    direct_pair_count_route: str,
    error_scope: str,
    coefficient_norm: str,
    parent_source_evidence_digest: str,
    parent_additional_certificate_digest: str,
    certificate_digest: str,
) -> GalerkinLocalRepresentedSourceCertificate:
    """PRIVATE: Validate one direct represented-source certificate outcome.

    Parameters
    ----------
    source : GalerkinLocalRepresentedSource
        Canonical represented local source.
    rectangles : Tuple[GalerkinLocalComplexRectangles, ...]
        Ordered exact ``D/B/R/S/M/T/C`` rectangle groups.
    component_error_bounds : Tuple[Float[Array, "..."], ...]
        Ordered per-component ``D/B/R/S/M/T/C`` error arrays.
    action_error_upper_bounds : Tuple[Float[Array, ""], ...]
        Ordered outward ``D/B/R/S/M/T/C`` Euclidean bounds.
    incident_field_norm_upper_bound : Float[Array, ""]
        Outward exact-real norm upper bound for stored ``v``.
    finite_certificate : Bool[Array, ""]
        Whether this outcome is a finite success certificate.
    direct_pair_count : Int[Array, ""]
        Direct ``D/B/R`` product count.
    maximum_direct_pairs : Int[Array, ""]
        Certified direct-work budget.
    failure : GalerkinLocalRepresentedSourceFailure
        Typed success or noncertificate outcome.
    exact_target : str
        Declared exact ``D/B/R/S/M/T/C`` target.
    arithmetic : str
        Declared direct interval arithmetic.
    direct_pair_count_route : str
        Declared work-count convention.
    error_scope : str
        Explicit inclusion and exclusion scope.
    coefficient_norm : str
        Retained coefficient norm.
    parent_source_evidence_digest : str
        Bound represented-source evidence digest.
    parent_additional_certificate_digest : str
        Bound LVT.20 certificate digest.
    certificate_digest : str
        Complete direct-certificate evidence digest.

    Returns
    -------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Validated finite certificate or typed noncertificate.

    Raises
    ------
    TypeError
        If ``source`` or ``failure`` has the wrong carrier type.
    ValueError
        If verified finite-certificate norm enclosure fails.
    """
    if not isinstance(source, GalerkinLocalRepresentedSource):
        raise TypeError("source must be GalerkinLocalRepresentedSource")
    if not isinstance(failure, GalerkinLocalRepresentedSourceFailure):
        raise TypeError("failure has the wrong represented-source enum")
    _raise_if(
        len(rectangles) != _ACTION_COUNT
        or len(component_error_bounds) != _ACTION_COUNT
        or len(action_error_upper_bounds) != _ACTION_COUNT,
        "certificate requires ordered D/B/R/S/M/T/C evidence",
    )
    size = source.target.state_indices.shape[0]
    checked_rectangles: Tuple[GalerkinLocalComplexRectangles, ...] = tuple(
        GalerkinLocalComplexRectangles(
            *(jnp.asarray(values, dtype=jnp.float64) for values in rectangle)
        )
        for rectangle in rectangles
    )
    _raise_if(
        any(
            any(values.shape != (size,) for values in rectangle)
            for rectangle in checked_rectangles
        ),
        "all exact rectangles must match target I_u",
    )
    errors: Tuple[Float64[Array, " n"], ...] = tuple(
        jnp.asarray(values, dtype=jnp.float64)
        for values in component_error_bounds
    )
    bounds: Tuple[Float64[Array, ""], ...] = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in action_error_upper_bounds
    )
    field_norm = jnp.asarray(
        incident_field_norm_upper_bound, dtype=jnp.float64
    )
    finite = jnp.asarray(finite_certificate, dtype=jnp.bool_)
    pair_count = jnp.asarray(direct_pair_count, dtype=jnp.int64)
    pair_budget = jnp.asarray(maximum_direct_pairs, dtype=jnp.int64)
    _raise_if(
        any(values.shape != (size,) for values in errors),
        "component errors must match target I_u",
    )
    _raise_if(
        any(
            value.shape != ()
            for value in (
                *bounds,
                field_norm,
                finite,
                pair_count,
                pair_budget,
            )
        ),
        "certificate bounds and work evidence must be scalar",
    )
    success = failure is GalerkinLocalRepresentedSourceFailure.NONE
    _raise_if(
        bool(finite) != success,
        "finite certificate and failure outcome are inconsistent",
    )
    _raise_if(
        bool(pair_count < 0) or bool(pair_budget <= 0),
        "work evidence is invalid",
    )
    for text, name in (
        (exact_target, "exact_target"),
        (arithmetic, "arithmetic"),
        (direct_pair_count_route, "direct_pair_count_route"),
        (error_scope, "error_scope"),
        (coefficient_norm, "coefficient_norm"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (parent_source_evidence_digest, "parent_source_evidence_digest"),
        (
            parent_additional_certificate_digest,
            "parent_additional_certificate_digest",
        ),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")
    _raise_if(
        parent_source_evidence_digest != source.source_evidence_digest,
        "certificate must bind represented-source evidence",
    )
    _raise_if(
        parent_additional_certificate_digest
        != source.additional_source_certificate.certificate_digest,
        "certificate must bind the LVT.20 certificate",
    )
    additional = source.additional_source_certificate
    s_rectangle = checked_rectangles[3]
    nested_rectangle = (
        additional.exact_source_real_lower_bounds,
        additional.exact_source_real_upper_bounds,
        additional.exact_source_imag_lower_bounds,
        additional.exact_source_imag_upper_bounds,
    )
    nested_source_mismatch = (
        any(
            bool(jnp.any(left != right))
            for left, right in zip(s_rectangle, nested_rectangle, strict=True)
        )
        or bool(jnp.any(errors[3] != additional.component_error_bounds))
        or bool(bounds[3] != additional.additional_source_error_upper_bound)
    )
    _raise_if(
        nested_source_mismatch,
        "additional-source evidence must be copied exactly once",
    )
    if success:
        invalid_rectangles = any(
            not bool(_rectangles_are_finite(value))
            for value in checked_rectangles
        )
        invalid_errors = any(
            bool(jnp.any(~jnp.isfinite(values)))
            or bool(jnp.any(values < 0.0))
            or bool(has_subnormal_components(values))
            for values in errors
        )
        invalid_scalar_bounds = any(
            not bool(jnp.isfinite(value))
            or bool(value < 0.0)
            or bool(has_subnormal_components(value))
            for value in (*bounds, field_norm)
        )
        norm_dominance_failure = False
        if not (invalid_rectangles or invalid_errors or invalid_scalar_bounds):
            try:
                required_bounds = tuple(
                    _host_nonnegative_norm_upper(values) for values in errors
                )
                required_field_norm = _host_complex_norm_upper(
                    source.modes.incident_field
                )
            except RootEnclosureError as error:
                raise ValueError(
                    "finite certificate norm enclosure must succeed"
                ) from error
            norm_dominance_failure = any(
                float(np.asarray(jax.device_get(bound))) < required
                for bound, required in zip(
                    bounds, required_bounds, strict=True
                )
            ) or (
                float(np.asarray(jax.device_get(field_norm)))
                < required_field_norm
            )
        invalid_outcome = (
            not bool(source.incident_eligible)
            or not bool(additional.finite_certificate)
            or invalid_rectangles
            or invalid_errors
            or invalid_scalar_bounds
            or bool(pair_count > pair_budget)
            or norm_dominance_failure
        )
    else:
        non_source_positions = (0, 1, 2, 4, 5, 6)
        invalid_outcome = (
            any(
                not (
                    bool(
                        jnp.all(
                            jnp.isneginf(
                                checked_rectangles[position].real_lower_bounds
                            )
                        )
                    )
                    and bool(
                        jnp.all(
                            jnp.isposinf(
                                checked_rectangles[position].real_upper_bounds
                            )
                        )
                    )
                    and bool(
                        jnp.all(
                            jnp.isneginf(
                                checked_rectangles[position].imag_lower_bounds
                            )
                        )
                    )
                    and bool(
                        jnp.all(
                            jnp.isposinf(
                                checked_rectangles[position].imag_upper_bounds
                            )
                        )
                    )
                )
                for position in non_source_positions
            )
            or any(
                not bool(jnp.all(jnp.isposinf(errors[position])))
                for position in non_source_positions
            )
            or any(
                not bool(jnp.isposinf(bounds[position]))
                for position in non_source_positions
            )
            or not bool(jnp.isposinf(field_norm))
        )
        work_failure = (
            GalerkinLocalRepresentedSourceFailure.DIRECT_WORK_BUDGET_EXCEEDED
        )
        if failure is work_failure:
            invalid_outcome = invalid_outcome or bool(
                pair_count <= pair_budget
            )
    checked_free_lower: Float64[Array, " n"] = eqx.error_if(
        checked_rectangles[0].real_lower_bounds,
        invalid_outcome,
        "represented-source certificate outcome is inconsistent",
    )
    checked_free = checked_rectangles[0]._replace(
        real_lower_bounds=checked_free_lower
    )
    certificate: GalerkinLocalRepresentedSourceCertificate = (
        GalerkinLocalRepresentedSourceCertificate(
            source=source,
            free_rectangles=checked_free,
            physical_cap_rectangles=checked_rectangles[1],
            interaction_rectangles=checked_rectangles[2],
            additional_source_rectangles=checked_rectangles[3],
            vacuum_matched_rectangles=checked_rectangles[4],
            total_source_rectangles=checked_rectangles[5],
            scattered_source_rectangles=checked_rectangles[6],
            free_component_error_bounds=errors[0],
            physical_cap_component_error_bounds=errors[1],
            interaction_component_error_bounds=errors[2],
            additional_source_component_error_bounds=errors[3],
            vacuum_matched_component_error_bounds=errors[4],
            total_source_component_error_bounds=errors[5],
            scattered_source_component_error_bounds=errors[6],
            free_action_error_upper_bound=bounds[0],
            physical_cap_action_error_upper_bound=bounds[1],
            interaction_action_error_upper_bound=bounds[2],
            additional_source_error_upper_bound=bounds[3],
            vacuum_matched_source_error_upper_bound=bounds[4],
            total_source_error_upper_bound=bounds[5],
            scattered_source_error_upper_bound=bounds[6],
            incident_field_norm_upper_bound=field_norm,
            finite_certificate=finite,
            direct_pair_count=pair_count,
            maximum_direct_pairs=pair_budget,
            failure=failure,
            exact_target=exact_target.strip(),
            arithmetic=arithmetic.strip(),
            direct_pair_count_route=direct_pair_count_route.strip(),
            error_scope=error_scope.strip(),
            coefficient_norm=coefficient_norm.strip(),
            parent_source_evidence_digest=parent_source_evidence_digest,
            parent_additional_certificate_digest=(
                parent_additional_certificate_digest
            ),
            certificate_digest=certificate_digest,
        )
    )
    return certificate


__all__: list[str] = [
    "GalerkinLocalComplexRectangles",
    "GalerkinLocalRepresentedSource",
    "GalerkinLocalRepresentedSourceActions",
    "GalerkinLocalRepresentedSourceCertificate",
    "GalerkinLocalRepresentedSourceFailure",
    "GalerkinLocalRepresentedSourceKind",
    "GalerkinLocalRepresentedSourceModes",
    "GalerkinLocalSourceAxis",
    "GalerkinLocalSourcePhaseConvention",
]
