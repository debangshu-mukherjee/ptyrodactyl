r"""Shared exact-carrier and shifted-free geometry evidence.

Extended Summary
----------------
This private orchestration leaf is route-neutral: it depends only on an
eligible acquisition, an exact stored accelerating voltage, and one ordered
state support.  It neither knows nor charges an interaction or absorber.

Routine Listings
----------------
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float64, Int64

from ptyrodactyl._tools import sha256, stored_value_payload
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportResult,
    _derive_algebraic_wavenumber,
    _exact_target_direction_error_bounds,
    _exact_target_full_offset_max,
)

from .enclosures import (
    _algebraic_free_diagonal,
    _exact_free_diagonal_interval,
    _exact_kinematic_intervals,
    _point_interval_distance_upper,
)

_EXACT_GEOMETRY_TARGET: str = (
    "SC.2 exact stored-input k0; SC.8 k0 times exact-real normalization of "
    "the nonzero stored carrier seed; exact on-shell SC.23/LVT-1 diagonal; "
    "all integer indices entering free/acquisition arithmetic satisfy the "
    "canonical acquisition safe bound"
)
_ALGEBRAIC_GEOMETRY_REALIZATION: str = (
    "canonical voltage-derived binary64 k0, stored binary64 carrier/box, "
    "and frozen SC.22 shifted-free diagonal"
)
_FREE_GEOMETRY_DIGEST_DOMAIN: str = (
    "ptyrodactyl.galerkin.route_neutral_free_geometry.v1"
)
_INDEX_BINARY64_RULE: str = (
    "every integer index consumed by free/exact-carrier arithmetic satisfies "
    "|n| <= min(INT64_MAX // 4, 2**52), preserving exact signed-int64 "
    "sums/differences and exact binary64 conversion"
)
_INDEX_SAFE_LIMIT: int = min(jnp.iinfo(jnp.int64).max // 4, 1 << 52)


class FreeGeometryEnclosure(NamedTuple):
    """Carry exact-carrier and frozen-free evidence between owning layers."""

    algebraic_free_diagonal: Float64[Array, " n"]
    exact_wavenumber_lower_bound: Float64[Array, ""]
    exact_wavenumber_upper_bound: Float64[Array, ""]
    wavenumber_error_bound: Float64[Array, ""]
    exact_carrier_lower_bounds: Float64[Array, " 3"]
    exact_carrier_upper_bounds: Float64[Array, " 3"]
    carrier_component_error_bounds: Float64[Array, " 3"]
    exact_free_diagonal_lower_bounds: Float64[Array, " n"]
    exact_free_diagonal_upper_bounds: Float64[Array, " n"]
    free_diagonal_error_bounds: Float64[Array, " n"]
    free_operator_error_bound: Float64[Array, ""]
    exact_geometry_target: str
    algebraic_geometry_realization: str
    free_geometry_digest: str


class ExactCarrierAcquisitionTransfer(NamedTuple):
    """Carry the six exact-target acquisition-transfer quantities."""

    incident_full_offset_max: Float64[Array, ""]
    outgoing_full_offset_max: Float64[Array, ""]
    incident_shell_defect_bounds: Float64[Array, " i"]
    outgoing_shell_defect_bounds: Float64[Array, " o"]
    incident_projection_error_bounds: Float64[Array, " i"]
    outgoing_projection_error_bounds: Float64[Array, " o"]


def _require_exact_binary64_indices(
    indices: Array,
    name: str,
) -> None:
    """PRIVATE: Re-enforce the canonical acquisition safe-index bound.

    Parameters
    ----------
    indices : Array
        Integer index array consumed by exact-carrier arithmetic.
    name : str
        Evidence-field name used in a failure message.

    Raises
    ------
    ValueError
        If any index exceeds the canonical safe bound.
    """
    if bool(
        jnp.any((indices < -_INDEX_SAFE_LIMIT) | (indices > _INDEX_SAFE_LIMIT))
    ):
        raise ValueError(
            f"{name} must satisfy the canonical acquisition safe-index "
            "bound for exact int64 and binary64 carrier/free geometry"
        )


def enclose_free_geometry(
    state_indices: Int64[Array, "n 3"],
    acquisition: GalerkinAcquisitionSupportResult,
    accelerating_voltage_kv: Float64[Array, ""],
) -> FreeGeometryEnclosure:
    """Build the frozen free diagonal and its exact-target enclosure."""
    if not bool(acquisition.support_eligible):
        raise ValueError(
            "free geometry requires eligible acquisition evidence"
        )
    manifest = acquisition.manifest
    _require_exact_binary64_indices(state_indices, "state_indices")
    canonical_wavenumber = _derive_algebraic_wavenumber(
        accelerating_voltage_kv
    )
    if bool(jnp.any(manifest.wavenumber != canonical_wavenumber)):
        raise ValueError(
            "acquisition wavenumber must equal canonical voltage-derived k0"
        )
    exact_wavenumber, exact_carrier, _ = _exact_kinematic_intervals(
        accelerating_voltage_kv,
        manifest.carrier,
    )
    exact_free = _exact_free_diagonal_interval(
        state_indices,
        manifest.box_lengths,
        exact_carrier,
    )
    diagonal = _algebraic_free_diagonal(
        state_indices,
        manifest.box_lengths,
        manifest.carrier,
        manifest.wavenumber,
    )
    k_error = _point_interval_distance_upper(
        manifest.wavenumber,
        exact_wavenumber,
    )
    carrier_errors = _point_interval_distance_upper(
        manifest.carrier,
        exact_carrier,
    )
    free_errors = _point_interval_distance_upper(diagonal, exact_free)
    delta_d = jnp.max(free_errors)
    arrays = (
        diagonal,
        exact_wavenumber[0],
        exact_wavenumber[1],
        k_error,
        exact_carrier[0],
        exact_carrier[1],
        carrier_errors,
        exact_free[0],
        exact_free[1],
        free_errors,
        delta_d,
    )
    if any(bool(jnp.any(~jnp.isfinite(value))) for value in arrays):
        raise ValueError("free-geometry enclosure left finite binary64 range")
    digest = sha256(
        {
            "domain": _FREE_GEOMETRY_DIGEST_DOMAIN,
            "state_indices": stored_value_payload(state_indices),
            "index_binary64_rule": _INDEX_BINARY64_RULE,
            "box_lengths": stored_value_payload(manifest.box_lengths),
            "accelerating_voltage_kv": stored_value_payload(
                accelerating_voltage_kv
            ),
            "algebraic_carrier": stored_value_payload(manifest.carrier),
            "algebraic_wavenumber": stored_value_payload(manifest.wavenumber),
            "algebraic_free_diagonal": stored_value_payload(diagonal),
            "exact_wavenumber": stored_value_payload(exact_wavenumber),
            "exact_carrier": stored_value_payload(exact_carrier),
            "exact_free_diagonal": stored_value_payload(exact_free),
            "wavenumber_error": stored_value_payload(k_error),
            "carrier_errors": stored_value_payload(carrier_errors),
            "free_errors": stored_value_payload(free_errors),
            "delta_D": stored_value_payload(delta_d),
            "exact_target": _EXACT_GEOMETRY_TARGET,
            "algebraic_route": _ALGEBRAIC_GEOMETRY_REALIZATION,
        }
    )
    enclosure: FreeGeometryEnclosure = FreeGeometryEnclosure(
        algebraic_free_diagonal=diagonal,
        exact_wavenumber_lower_bound=exact_wavenumber[0],
        exact_wavenumber_upper_bound=exact_wavenumber[1],
        wavenumber_error_bound=k_error,
        exact_carrier_lower_bounds=exact_carrier[0],
        exact_carrier_upper_bounds=exact_carrier[1],
        carrier_component_error_bounds=carrier_errors,
        exact_free_diagonal_lower_bounds=exact_free[0],
        exact_free_diagonal_upper_bounds=exact_free[1],
        free_diagonal_error_bounds=free_errors,
        free_operator_error_bound=delta_d,
        exact_geometry_target=_EXACT_GEOMETRY_TARGET,
        algebraic_geometry_realization=_ALGEBRAIC_GEOMETRY_REALIZATION,
        free_geometry_digest=digest,
    )
    return enclosure


def transfer_exact_carrier_acquisition(
    acquisition: GalerkinAcquisitionSupportResult,
    geometry: FreeGeometryEnclosure,
) -> ExactCarrierAcquisitionTransfer:
    """Transfer every represented/omitted acquisition gate to exact SC.8."""
    if not bool(acquisition.support_eligible):
        raise ValueError(
            "exact-carrier transfer requires eligible acquisition evidence"
        )
    manifest = acquisition.manifest
    support = manifest.support
    for indices, name in (
        (support.state_indices, "state_indices"),
        (support.interaction_indices, "interaction_indices"),
        (support.absorber_indices, "absorber_indices"),
        (support.work_indices, "work_indices"),
        (manifest.incident_indices, "incident_indices"),
        (manifest.elastic_outgoing_indices, "elastic_outgoing_indices"),
        (manifest.preterminal_indices, "preterminal_indices"),
        (manifest.transverse_indices, "transverse_indices"),
        (
            manifest.deliberately_omitted_indices,
            "deliberately_omitted_indices",
        ),
    ):
        _require_exact_binary64_indices(indices, name)
    incident_shell: Float64[Array, " i"]
    incident_projection: Float64[Array, " i"]
    incident_shell, incident_projection = _exact_target_direction_error_bounds(
        acquisition.incident_shell_defect_upper_bounds,
        acquisition.incident_projection_error_upper_bounds,
        manifest.incident_direction_dispositions,
        geometry.carrier_component_error_bounds,
        geometry.wavenumber_error_bound,
        manifest.wavenumber,
    )
    outgoing_shell: Float64[Array, " o"]
    outgoing_projection: Float64[Array, " o"]
    outgoing_shell, outgoing_projection = _exact_target_direction_error_bounds(
        acquisition.outgoing_shell_defect_upper_bounds,
        acquisition.outgoing_projection_error_upper_bounds,
        manifest.outgoing_direction_dispositions,
        geometry.carrier_component_error_bounds,
        geometry.wavenumber_error_bound,
        manifest.wavenumber,
    )
    evidence_valid = (
        jnp.all(incident_shell <= manifest.incident_on_shell_defect_bounds)
        & jnp.all(outgoing_shell <= manifest.outgoing_on_shell_defect_bounds)
        & jnp.all(
            incident_projection <= manifest.incident_projection_error_bounds
        )
        & jnp.all(
            outgoing_projection <= manifest.outgoing_projection_error_bounds
        )
    )
    normal_error = geometry.carrier_component_error_bounds[
        manifest.terminal_axis
    ]
    represented_valid = (
        jnp.all(
            (~acquisition.state_forward_mask)
            | (acquisition.state_oriented_normal_interval_lower > normal_error)
        )
        & jnp.all(
            (~acquisition.state_backward_mask)
            | (
                acquisition.state_oriented_normal_interval_upper
                < -normal_error
            )
        )
        & jnp.all(
            (~acquisition.state_grazing_mask)
            | (
                (acquisition.state_oriented_normal_interval_lower == 0.0)
                & (acquisition.state_oriented_normal_interval_upper == 0.0)
                & (normal_error == 0.0)
            )
        )
        & ~jnp.any(acquisition.state_ambiguous_mask)
    )
    omitted_valid = (
        jnp.all(
            (~acquisition.omitted_forward_mask)
            | (
                acquisition.omitted_oriented_normal_interval_lower
                > normal_error
            )
        )
        & jnp.all(
            (~acquisition.omitted_backward_mask)
            | (
                acquisition.omitted_oriented_normal_interval_upper
                < -normal_error
            )
        )
        & jnp.all(
            (~acquisition.omitted_grazing_mask)
            | (
                (acquisition.omitted_oriented_normal_interval_lower == 0.0)
                & (acquisition.omitted_oriented_normal_interval_upper == 0.0)
                & (normal_error == 0.0)
            )
        )
        & ~jnp.any(acquisition.omitted_ambiguous_mask)
    )
    if not bool(evidence_valid):
        raise ValueError(
            "exact-carrier shell or projection evidence exceeds a "
            "submitted ceiling"
        )
    if not bool(represented_valid):
        raise ValueError(
            "represented sector classification is not preserved by exact "
            "carrier normalization"
        )
    if not bool(omitted_valid):
        raise ValueError(
            "omitted sector classification is not preserved by exact "
            "carrier normalization"
        )
    incident_full = _exact_target_full_offset_max(
        acquisition.incident_full_offset_max,
        geometry.carrier_component_error_bounds,
        manifest.incident_direction_dispositions,
    )
    outgoing_full = _exact_target_full_offset_max(
        acquisition.outgoing_full_offset_max,
        geometry.carrier_component_error_bounds,
        manifest.outgoing_direction_dispositions,
    )
    transfer: ExactCarrierAcquisitionTransfer = (
        ExactCarrierAcquisitionTransfer(
            incident_full_offset_max=incident_full,
            outgoing_full_offset_max=outgoing_full,
            incident_shell_defect_bounds=incident_shell,
            outgoing_shell_defect_bounds=outgoing_shell,
            incident_projection_error_bounds=incident_projection,
            outgoing_projection_error_bounds=outgoing_projection,
        )
    )
    return transfer


__all__: list[str] = []
