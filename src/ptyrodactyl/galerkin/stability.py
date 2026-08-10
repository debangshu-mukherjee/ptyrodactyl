r"""Check and invoke an exact SC-1 Route-A stability certificate.

Extended Summary
----------------
This host-side module implements a trusted Route-A checker for one retained
scalar Galerkin result. It reconstructs the manifested algebraic matrix and
stored finite right-hand side from exact dyadic binary64 values. The legacy
route treats that stored right-hand side as exact. A separate represented-
source route rebuilds narrow RM-S3 evidence and includes its source error. It
proves a
direct exact-target floor for the shared analytic cosine-shell absorber with
exact rational arithmetic. The independently reconstructed ``H_alg``
residual is lifted to the exact SC-1 target with RM-S2 ``delta_H ||x||``;
the stored source is the exact right-hand side for this invocation. At small
dimension the checker also computes the exact-rational Gershgorin absorber
floor and selects the stronger absorber route. Each payload binds one target,
source, submitted state, and independently supplied state budget.

Routine Listings
----------------
:func:`check_galerkin_absorber_floor`
    Build the bounded exact-stored-RHS algebraic-oracle proof.
:func:`check_represented_galerkin_absorber_floor`
    Build a Route-A proof with an eligible rebuilt RM-S3 source.
:func:`invoke_galerkin_stability`
    Recheck and apply the exact-stored-RHS algebraic-oracle route.
:func:`invoke_represented_galerkin_stability`
    Recheck and apply an eligible represented-source stability proof.

Notes
-----
The analytic absorber-floor acquisition scales with the state-index list and
uses no transcendental-library result. Algebraic residual reconstruction is
still dense and quadratic, so this is not a scalable interval eigensolver or
a universal stability theorem. Its exact arithmetic treats each stored
binary64 component as its exact dyadic value. The legacy route excludes
analytic RM-S3 source conformity. The represented route includes the RM-S3
exact-target total-source enclosure exactly once. Per-call residual-formation
errors remain outside both routes. The SHA-256 values are provenance
checksums; checker
reconstruction establishes target identity. Invoke the checker separately
for every retained result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Tuple
from jaxtyping import Complex128, Float64, Int64, Num
from numpy.typing import NDArray

from ptyrodactyl._canonical_digest import (
    _array_payload,
    _host_array,
    _sha256,
    _stored_value_payload,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportStatus,
    GalerkinPotentialErrorRoute,
    GalerkinProductSupport,
    GalerkinRepresentedSource,
    GalerkinRepresentedSourceKind,
    GalerkinSolveResult,
    GalerkinSource,
    GalerkinStabilityDisposition,
    GalerkinStabilityFailure,
    GalerkinStabilityProof,
    GalerkinStabilityResult,
    GalerkinStabilityRoute,
    GalerkinTargetManifest,
    Potential3D,
    create_galerkin_acquisition_manifest,
    create_galerkin_product_support,
    create_galerkin_solve_result,
    create_galerkin_source,
    create_galerkin_stability_proof,
    create_galerkin_stability_result,
    create_potential_3d,
    scalar_float,
)

from .acquisition import check_galerkin_acquisition_support
from .sources import (
    build_represented_focused_galerkin_source,
    build_represented_plane_galerkin_source,
)
from .system import (
    create_galerkin_target,
    create_host_checked_galerkin_target,
)

_CHECKER_ID: str = "ptyrodactyl.exact_dyadic_route_a.source_lift.v4"
_LEGACY_RHS_TARGET: str = "stored_total_source_exact_binary_components"
_LEGACY_RESIDUAL_SCOPE: str = (
    "independent_exact_H_alg_residual_plus_RM-S2_delta_H_field_lift; "
    "excludes analytic_RM-S3_source_conformity and per-call action/residual "
    "formation errors"
)
_LEGACY_SOURCE_ERROR_ROUTE: str = "legacy_stored_binary_rhs_exact"
_LEGACY_SOURCE_ERROR_SCOPE: str = (
    "stored binary total source defines the exact invocation RHS; delta_S=0; "
    "no analytic RM-S3 source-conformity claim"
)
# Backward-compatible private labels retained for existing exact-RHS evidence.
_RHS_TARGET: str = _LEGACY_RHS_TARGET
_RESIDUAL_SCOPE: str = _LEGACY_RESIDUAL_SCOPE
_REPRESENTED_RHS_TARGET: str = (
    "RM-S3_exact_periodic_total_source_H0v_plus_stored_additional_source"
)
_REPRESENTED_RESIDUAL_SCOPE: str = (
    "independent_exact_H_alg_residual_plus_RM-S2_delta_H_field_lift_plus_"
    "RM-S3_exact_target_total_source_delta_S_lift; excludes per-call "
    "residual-formation, solver, continuum, window, box, current, and "
    "detector errors"
)
_MAX_GERSHGORIN_DIMENSION: int = 32
_SPACE_DIMENSIONS: int = 3
_SUPPORT_RANK: int = 2
_SINGLETON_DENOMINATOR_INDEX: int = 2
_MAX_BINARY64: float = float.fromhex("0x1.fffffffffffffp+1023")
_INVALID_TARGET_DIGEST: str = "0" * 64
_INVALID_RESULT_DIGEST: str = "f" * 64
type _ComplexFraction = Tuple[Fraction, Fraction]
type _BoundSource = GalerkinSource | GalerkinRepresentedSource


@dataclass(frozen=True)
class _SourceLift:
    """PRIVATE: Freeze one source route's exact residual-lift metadata."""

    error_upper: Fraction
    finite: bool
    rhs_target: str
    residual_scope: str
    error_route: str
    error_scope: str


_LEGACY_SOURCE_LIFT: _SourceLift = _SourceLift(
    error_upper=Fraction(0),
    finite=True,
    rhs_target=_LEGACY_RHS_TARGET,
    residual_scope=_LEGACY_RESIDUAL_SCOPE,
    error_route=_LEGACY_SOURCE_ERROR_ROUTE,
    error_scope=_LEGACY_SOURCE_ERROR_SCOPE,
)


def _complex_add(
    left: _ComplexFraction,
    right: _ComplexFraction,
) -> _ComplexFraction:
    """PRIVATE: Add two exact complex rationals.

    Parameters
    ----------
    left : _ComplexFraction
        Left real-imaginary rational pair.
    right : _ComplexFraction
        Right real-imaginary rational pair.

    Returns
    -------
    result : _ComplexFraction
        Componentwise exact sum.
    """
    result: _ComplexFraction = (left[0] + right[0], left[1] + right[1])
    return result


def _complex_subtract(
    left: _ComplexFraction,
    right: _ComplexFraction,
) -> _ComplexFraction:
    """PRIVATE: Subtract two exact complex rationals.

    Parameters
    ----------
    left : _ComplexFraction
        Left real-imaginary rational pair.
    right : _ComplexFraction
        Right real-imaginary rational pair.

    Returns
    -------
    result : _ComplexFraction
        Componentwise exact difference ``left - right``.
    """
    result: _ComplexFraction = (left[0] - right[0], left[1] - right[1])
    return result


def _complex_multiply(
    left: _ComplexFraction,
    right: _ComplexFraction,
) -> _ComplexFraction:
    """PRIVATE: Multiply two exact complex rationals.

    Parameters
    ----------
    left : _ComplexFraction
        Left real-imaginary rational pair.
    right : _ComplexFraction
        Right real-imaginary rational pair.

    Returns
    -------
    result : _ComplexFraction
        Exact complex product in real-imaginary pair form.
    """
    result: _ComplexFraction = (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )
    return result


def _complex_conjugate(value: _ComplexFraction) -> _ComplexFraction:
    """PRIVATE: Conjugate one exact complex rational.

    Parameters
    ----------
    value : _ComplexFraction
        Real-imaginary rational pair.

    Returns
    -------
    result : _ComplexFraction
        Pair with its imaginary component negated exactly.
    """
    result: _ComplexFraction = (value[0], -value[1])
    return result


def _fraction_from_float(value: float) -> Fraction:
    """PRIVATE: Convert one finite binary float to its exact dyadic rational.

    Parameters
    ----------
    value : float
        Finite binary floating-point value.

    Returns
    -------
    result : Fraction
        Exact rational representation of the stored binary value.

    Notes
    -----
    This conversion does not recover a pre-rounding real value.
    """
    result: Fraction = Fraction.from_float(float(value))
    return result


def _complex_fraction(value: complex) -> _ComplexFraction:
    """PRIVATE: Convert one binary complex value to exact dyadic components.

    Parameters
    ----------
    value : complex
        Finite binary complex value.

    Returns
    -------
    result : _ComplexFraction
        Exact dyadic real and imaginary components.

    Notes
    -----
    Each stored component is interpreted as its exact binary floating value.
    """
    result: _ComplexFraction = (
        _fraction_from_float(float(value.real)),
        _fraction_from_float(float(value.imag)),
    )
    return result


def _target_payload(manifest: GalerkinTargetManifest) -> Dict[str, object]:
    """PRIVATE: Bind every manifested-target field in an exact payload.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to serialize.

    Returns
    -------
    payload : Dict[str, object]
        Metadata and exact stored-array representations for the target.

    Raises
    ------
    TypeError
        If recursive serialization does not produce a canonical mapping.

    Notes
    -----
    This payload recursively includes every declared field of the target,
    voxel potential, acquisition submission, acquisition checker result,
    VC-1 realization, and RM-S2 fixed-linear ledger. Read-only properties are
    intentionally absent because their owning stored values already appear
    exactly once in the nested tree.
    """
    stored = _stored_value_payload(manifest)
    if not isinstance(stored, dict):
        raise TypeError("target payload must be a canonical mapping")
    payload: Dict[str, object] = cast(Dict[str, object], stored)
    return payload


def _target_digest(manifest: GalerkinTargetManifest) -> str:
    """PRIVATE: Compute the canonical manifested-target checksum.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target whose complete payload is bound.

    Returns
    -------
    digest : str
        SHA-256 checksum of the canonical target payload.
    """
    payload: Dict[str, object] = _target_payload(manifest)
    digest: str = _sha256(payload)
    return digest


def _source_payload(source: _BoundSource) -> Dict[str, object]:
    """PRIVATE: Bind every declared source field in an exact payload.

    Parameters
    ----------
    source : _BoundSource
        Canonical legacy or represented source carrier to serialize.

    Returns
    -------
    payload : Dict[str, object]
        Complete recursive carrier payload, including its concrete type.

    Raises
    ------
    TypeError
        If recursive source serialization does not produce a mapping.

    Notes
    -----
    The represented payload includes its nested target, modes, actions,
    representation ledger, error enclosure, and eligibility gates. This is
    intentionally redundant with ``target_digest``: exact target binding is
    checked independently and the redundancy prevents a source proof from
    being replayed under a different nested manifest.
    """
    stored = _stored_value_payload(source)
    if not isinstance(stored, dict):
        raise TypeError("source payload must be a canonical mapping")
    payload: Dict[str, object] = cast(Dict[str, object], stored)
    return payload


def _solve_result_payload(
    solve_result: GalerkinSolveResult,
) -> Dict[str, object]:
    """PRIVATE: Bind every algebraic solve-result field in an exact payload.

    Parameters
    ----------
    solve_result : GalerkinSolveResult
        Submitted solve-result carrier to serialize.

    Returns
    -------
    payload : Dict[str, object]
        Exact arrays and enumerated provenance for the submitted result.
    """
    payload: Dict[str, object] = {
        "field": _array_payload(solve_result.field),
        "residual": _array_payload(solve_result.residual),
        "residual_norm": _array_payload(solve_result.residual_norm),
        "normal_residual_norm": _array_payload(
            solve_result.normal_residual_norm
        ),
        "recurrence_residual_norm": _array_payload(
            solve_result.recurrence_residual_norm
        ),
        "iterations": _array_payload(solve_result.iterations),
        "operator_applications": _array_payload(
            solve_result.operator_applications
        ),
        "status": _array_payload(solve_result.status),
        "converged": _array_payload(solve_result.converged),
        "method": solve_result.method.value,
        "certificate_reason": solve_result.certificate_reason.value,
    }
    return payload


def _result_digest(
    target_digest: str,
    source: _BoundSource,
    solve_result: GalerkinSolveResult,
) -> str:
    """PRIVATE: Compute the source- and submitted-state result checksum.

    Parameters
    ----------
    target_digest : str
        Canonical checksum of the bound target.
    source : _BoundSource
        Bound legacy or represented source carrier.
    solve_result : GalerkinSolveResult
        Bound submitted field and solver provenance.

    Returns
    -------
    digest : str
        SHA-256 checksum binding target, source, and submitted result.
    """
    payload: Dict[str, object] = {
        "target_digest": target_digest,
        "source": _source_payload(source),
        "solve_result": _solve_result_payload(solve_result),
    }
    digest: str = _sha256(payload)
    return digest


def _rebuild_target_primitives(
    manifest: GalerkinTargetManifest,
) -> Tuple[Potential3D, GalerkinAcquisitionManifest]:
    """PRIVATE: Recreate the voxel and acquisition submissions from leaves.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Candidate nested target whose primitive submissions are recreated.

    Returns
    -------
    potential : Potential3D
        Factory-reconstructed :class:`~ptyrodactyl.types.Potential3D`.
    acquisition_submission : GalerkinAcquisitionManifest
        Factory-reconstructed acquisition manifest.

    Notes
    -----
    These are the only trusted submission seams. Every acquisition-checker,
    VC-1, RM-S2, and final target field is derived again downstream.
    """
    candidate_potential: Potential3D = manifest.realization.potential
    potential: Potential3D = create_potential_3d(
        volume=candidate_potential.volume,
        voxel_size=candidate_potential.voxel_size,
        box_size=candidate_potential.box_size,
        origin=candidate_potential.origin,
        units=candidate_potential.units,
        reference_value=candidate_potential.reference_value,
        reference_semantics=candidate_potential.reference_semantics,
        boundary=candidate_potential.boundary,
        producer=candidate_potential.producer,
        provenance_hash=candidate_potential.provenance_hash,
        coefficient_normalization=(
            candidate_potential.coefficient_normalization
        ),
        band_limit=candidate_potential.band_limit,
    )
    candidate_acquisition: GalerkinAcquisitionManifest = (
        manifest.realization.support_eligibility.manifest
    )
    candidate_support: GalerkinProductSupport = candidate_acquisition.support
    support: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=candidate_support.state_indices,
        interaction_indices=candidate_support.interaction_indices,
        absorber_indices=candidate_support.absorber_indices,
        work_indices=candidate_support.work_indices,
        work_shape=candidate_support.work_shape,
    )
    acquisition_submission: GalerkinAcquisitionManifest = (
        create_galerkin_acquisition_manifest(
            support,
            candidate_acquisition.incident_indices,
            candidate_acquisition.elastic_outgoing_indices,
            candidate_acquisition.preterminal_indices,
            candidate_acquisition.transverse_indices,
            candidate_acquisition.deliberately_omitted_indices,
            incident_physical_wavevectors=(
                candidate_acquisition.incident_physical_wavevectors
            ),
            outgoing_physical_wavevectors=(
                candidate_acquisition.outgoing_physical_wavevectors
            ),
            incident_direction_dispositions=(
                candidate_acquisition.incident_direction_dispositions
            ),
            outgoing_direction_dispositions=(
                candidate_acquisition.outgoing_direction_dispositions
            ),
            incident_on_shell_defect_bounds=(
                candidate_acquisition.incident_on_shell_defect_bounds
            ),
            outgoing_on_shell_defect_bounds=(
                candidate_acquisition.outgoing_on_shell_defect_bounds
            ),
            incident_projection_error_bounds=(
                candidate_acquisition.incident_projection_error_bounds
            ),
            outgoing_projection_error_bounds=(
                candidate_acquisition.outgoing_projection_error_bounds
            ),
            carrier=candidate_acquisition.carrier,
            box_lengths=candidate_acquisition.box_lengths,
            wavenumber=candidate_acquisition.wavenumber,
            carrier_on_shell_defect_bound=(
                candidate_acquisition.carrier_on_shell_defect_bound
            ),
            on_shell_defect_tolerance=(
                candidate_acquisition.on_shell_defect_tolerance
            ),
            terminal_axis=candidate_acquisition.terminal_axis,
            terminal_side=candidate_acquisition.terminal_side,
            carrier_id=candidate_acquisition.carrier_id,
            carrier_ownership=candidate_acquisition.carrier_ownership,
            carrier_overlap_disposition=(
                candidate_acquisition.carrier_overlap_disposition
            ),
            carrier_target_route=candidate_acquisition.carrier_target_route,
            endpoint_convention=candidate_acquisition.endpoint_convention,
            backward_disposition=candidate_acquisition.backward_disposition,
            backward_exclusion_basis=(
                candidate_acquisition.backward_exclusion_basis
            ),
            claims_backscatter=candidate_acquisition.claims_backscatter,
        )
    )
    result: Tuple[Potential3D, GalerkinAcquisitionManifest] = (
        potential,
        acquisition_submission,
    )
    return result


def _manifest_is_canonical(manifest: GalerkinTargetManifest) -> bool:
    """PRIVATE: Recheck and rebuild every derived nested target branch.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Candidate SC-1 target manifest.

    Returns
    -------
    canonical_match : bool
        True only when factory reconstruction matches every payload field.

    Notes
    -----
    Canonicality starts only from a reconstructed ``Potential3D`` and the
    reconstructed acquisition submission. The checker must reproduce the
    complete stored acquisition result and return ``SUPPORT_ELIGIBLE`` before
    the public production builder independently reruns VC-1 and RM-S2. Any
    reconstruction or synchronization error fails closed as ``False``.
    """
    try:
        potential, acquisition_submission = _rebuild_target_primitives(
            manifest
        )
        checked_acquisition = check_galerkin_acquisition_support(
            acquisition_submission,
        )
        jax.block_until_ready(checked_acquisition)
        candidate_eligibility = manifest.realization.support_eligibility
        eligibility_matches: bool = _stored_value_payload(
            checked_acquisition
        ) == _stored_value_payload(candidate_eligibility)
        status: int = int(_host_array(checked_acquisition.status))
        eligible: bool = bool(
            _host_array(checked_acquisition.support_eligible)
        )
        if (
            not eligibility_matches
            or status != int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
            or not eligible
        ):
            canonical_match: bool = False
            return canonical_match
        certificate = manifest.realization.coefficient_certificate
        if (
            manifest.realization.error_route
            is GalerkinPotentialErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
        ):
            if certificate is None:
                canonical_match = False
                return canonical_match  # noqa: RET504
            maximum_direct_terms: int = int(
                _host_array(certificate.maximum_direct_terms)
            )
            canonical = create_host_checked_galerkin_target(
                potential,
                checked_acquisition,
                accelerating_voltage_kv=manifest.accelerating_voltage_kv,
                cap_scale=manifest.cap_scale,
                target_name=manifest.target_name,
                maximum_direct_terms=maximum_direct_terms,
            )
        else:
            if certificate is not None:
                canonical_match = False
                return canonical_match  # noqa: RET504
            canonical = create_galerkin_target(
                potential,
                checked_acquisition,
                accelerating_voltage_kv=manifest.accelerating_voltage_kv,
                cap_scale=manifest.cap_scale,
                target_name=manifest.target_name,
            )
        jax.block_until_ready(canonical)
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        canonical_match: bool = False
        return canonical_match
    canonical_match: bool = _target_payload(canonical) == _target_payload(
        manifest
    )
    return canonical_match


def _source_is_canonical(source: GalerkinSource) -> bool:
    """PRIVATE: Rebuild and compare every factory-owned source field.

    Parameters
    ----------
    source : GalerkinSource
        Candidate finite-source carrier.

    Returns
    -------
    canonical_match : bool
        True only when factory reconstruction matches every payload field.

    Notes
    -----
    Any reconstruction or synchronization error fails closed as ``False``.
    """
    try:
        canonical = create_galerkin_source(
            incident_field=source.incident_field,
            incident_source=source.incident_source,
            additional_source=source.additional_source,
            total_source=source.total_source,
            scattered_source=source.scattered_source,
            branch=source.branch,
        )
        jax.block_until_ready(canonical)
        canonical_match: bool = _source_payload(canonical) == _source_payload(
            source
        )
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        canonical_match = False
    return canonical_match


def _represented_source_is_canonical(
    manifest: GalerkinTargetManifest,
    source: GalerkinRepresentedSource,
) -> bool:
    """PRIVATE: Rebuild a represented source from stored primitive inputs.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Candidate stability target that must exactly own the source.
    source : GalerkinRepresentedSource
        Candidate represented plane or focused source.

    Returns
    -------
    canonical_match : bool
        True only when the nested target is identical and the appropriate
        public source builder reproduces every declared source field.

    Notes
    -----
    The rebuild trusts only the stored pre-phase aperture, requested flux,
    scan/source-plane geometry, aberration input, optional additional source,
    and static construction choices. All modes, actions, intervals, ledgers,
    and eligibility gates are recomputed. For a plane source, the active
    position is derived from the aperture itself rather than its stored mask.
    """
    try:
        if _target_payload(source.manifest) != _target_payload(manifest):
            canonical_match: bool = False
            return canonical_match
        aperture: Complex128[NDArray, " n"] = _host_array(
            source.modes.aperture_weights
        )
        active_positions = np.flatnonzero(
            (np.real(aperture) != 0.0) | (np.imag(aperture) != 0.0)
        )
        if source.kind is GalerkinRepresentedSourceKind.PLANE_MODE:
            if active_positions.size != 1:
                canonical_match = False
                return canonical_match  # noqa: RET504
            position: int = int(active_positions[0])
            canonical = build_represented_plane_galerkin_source(
                manifest=manifest,
                state_position=position,
                aperture_weight=source.modes.aperture_weights[position],
                target_reduced_flux=source.modes.target_reduced_flux,
                normal_axis=source.modes.normal_axis,
                phase_convention=source.modes.phase_convention,
                stored_shell_route=source.modes.stored_shell_route,
                shell_defect_tolerance=source.modes.shell_defect_tolerance,
                source_plane_coordinate=(source.modes.source_plane_coordinate),
                scan_position=source.modes.scan_position,
                aberration_phase=source.modes.aberration_phases[position],
                additional_source=source.actions.additional_source,
            )
        elif source.kind is GalerkinRepresentedSourceKind.COHERENT_FOCUSED:
            canonical = build_represented_focused_galerkin_source(
                manifest=manifest,
                aperture_weights=source.modes.aperture_weights,
                target_reduced_flux=source.modes.target_reduced_flux,
                normal_axis=source.modes.normal_axis,
                phase_convention=source.modes.phase_convention,
                stored_shell_route=source.modes.stored_shell_route,
                shell_defect_tolerance=source.modes.shell_defect_tolerance,
                source_plane_coordinate=(source.modes.source_plane_coordinate),
                scan_position=source.modes.scan_position,
                aberration_phases=source.modes.aberration_phases,
                additional_source=source.actions.additional_source,
            )
        else:
            canonical_match = False
            return canonical_match  # noqa: RET504
        jax.block_until_ready(canonical)
        canonical_match = _source_payload(canonical) == _source_payload(source)
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        canonical_match = False
    return canonical_match


def _solve_result_is_canonical(
    solve_result: GalerkinSolveResult,
) -> bool:
    """PRIVATE: Rebuild and compare every factory-owned solve-result field.

    Parameters
    ----------
    solve_result : GalerkinSolveResult
        Candidate manifested solve-result submission.

    Returns
    -------
    canonical_match : bool
        True only for the required dtypes and exact factory reconstruction.

    Notes
    -----
    Manifested certification requires Complex128 fields, Float64 metrics, and
    Int32 counters and status. Polymorphic algebraic results fail closed.
    """
    try:
        canonical_dtypes: bool = (
            solve_result.field.dtype == jnp.complex128
            and solve_result.residual.dtype == jnp.complex128
            and solve_result.residual_norm.dtype == jnp.float64
            and solve_result.normal_residual_norm.dtype == jnp.float64
            and solve_result.recurrence_residual_norm.dtype == jnp.float64
            and solve_result.iterations.dtype == jnp.int32
            and solve_result.operator_applications.dtype == jnp.int32
            and solve_result.status.dtype == jnp.int32
            and solve_result.converged.dtype == jnp.bool_
        )
        if not canonical_dtypes:
            canonical_match: bool = False
            return canonical_match
        canonical = create_galerkin_solve_result(
            field=solve_result.field,
            residual=solve_result.residual,
            residual_norm=solve_result.residual_norm,
            normal_residual_norm=solve_result.normal_residual_norm,
            recurrence_residual_norm=solve_result.recurrence_residual_norm,
            iterations=solve_result.iterations,
            operator_applications=solve_result.operator_applications,
            status=solve_result.status,
            converged=solve_result.converged,
            method=solve_result.method,
            certificate_reason=solve_result.certificate_reason,
        )
        jax.block_until_ready(canonical)
        canonical_match: bool = _solve_result_payload(
            canonical
        ) == _solve_result_payload(solve_result)
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        canonical_match = False
    return canonical_match


def _coefficient_map(
    indices: Int64[jax.Array, "... 3"],
    coefficients: Complex128[jax.Array, "..."],
) -> Dict[Tuple[int, int, int], _ComplexFraction]:
    """PRIVATE: Map reciprocal indices to exact dyadic coefficients.

    Parameters
    ----------
    indices : Int64[jax.Array, "... 3"]
        Reciprocal indices in ``(x, y, z)`` order.
    coefficients : Complex128[jax.Array, "..."]
        Binary64-complex coefficients in matching order.

    Returns
    -------
    mapping : Dict[Tuple[int, int, int], _ComplexFraction]
        Exact coefficient keyed by its reciprocal-index tuple.

    Notes
    -----
    Each binary64 component becomes its exact dyadic rational value.
    """
    index_array: Int64[NDArray, "... 3"] = _host_array(indices)
    coefficient_array: Complex128[NDArray, "..."] = _host_array(coefficients)
    mapping: Dict[Tuple[int, int, int], _ComplexFraction] = {}
    for index, coefficient in zip(index_array, coefficient_array, strict=True):
        key = (int(index[0]), int(index[1]), int(index[2]))
        mapping[key] = _complex_fraction(complex(coefficient))
    return mapping


def _matrix_from_coefficients(
    state_indices: Int64[NDArray, "n 3"],
    mapping: Dict[Tuple[int, int, int], _ComplexFraction],
) -> list[list[_ComplexFraction]]:
    """PRIVATE: Assemble one exact compressed multiplier matrix.

    Parameters
    ----------
    state_indices : Int64[NDArray, "n 3"]
        Retained reciprocal indices in ``(x, y, z)`` order.
    mapping : Dict[Tuple[int, int, int], _ComplexFraction]
        Exact multiplier coefficient map.

    Returns
    -------
    matrix : list[list[_ComplexFraction]]
        Dense exact principal compression in retained-state order.

    Notes
    -----
    Matrix entry ``(i, j)`` uses the coefficient at ``k_i - k_j`` and exact
    zero when the difference is absent.
    """
    matrix: list[list[_ComplexFraction]] = []
    zero: _ComplexFraction = (Fraction(0), Fraction(0))
    for row in state_indices:
        matrix_row: list[_ComplexFraction] = []
        for column in state_indices:
            difference = (
                int(row[0]) - int(column[0]),
                int(row[1]) - int(column[1]),
                int(row[2]) - int(column[2]),
            )
            matrix_row.append(mapping.get(difference, zero))
        matrix.append(matrix_row)
    return matrix


def _is_hermitian(matrix: list[list[_ComplexFraction]]) -> bool:
    """PRIVATE: Determine whether an exact rational matrix is Hermitian.

    Parameters
    ----------
    matrix : list[list[_ComplexFraction]]
        Square dense matrix of exact complex rationals.

    Returns
    -------
    result : bool
        True when every entry equals the conjugate transposed entry.

    Notes
    -----
    The comparison is exact and applies no floating tolerance.
    """
    size: int = len(matrix)
    result: bool = all(
        matrix[row][column] == _complex_conjugate(matrix[column][row])
        for row in range(size)
        for column in range(size)
    )
    return result


def _absorber_floor(
    absorber: list[list[_ComplexFraction]],
) -> Fraction:
    """PRIVATE: Compute a rational Gershgorin absorber lower bound.

    Parameters
    ----------
    absorber : list[list[_ComplexFraction]]
        Nonempty Hermitian absorber compression in exact rational form.

    Returns
    -------
    floor : Fraction
        Exact rational lower bound for the smallest absorber eigenvalue.

    Notes
    -----
    The off-diagonal radius uses ``abs(real) + abs(imag)``, an exact rational
    upper bound on complex magnitude. A non-real diagonal fails closed at
    zero.
    """
    row_bounds: list[Fraction] = []
    for row, values in enumerate(absorber):
        diagonal: _ComplexFraction = values[row]
        if diagonal[1] != 0:
            floor: Fraction = Fraction(0)
            return floor
        off_diagonal_upper: Fraction = sum(
            (
                abs(value[0]) + abs(value[1])
                for column, value in enumerate(values)
                if column != row
            ),
            start=Fraction(0),
        )
        row_bounds.append(diagonal[0] - off_diagonal_upper)
    floor: Fraction = min(row_bounds)
    return floor


def _rational_cosine_shell_box_floor(
    state_indices: Int64[NDArray, "n 3"],
) -> Fraction:
    r"""PRIVATE: Prove a rational floor for the analytic cosine shell.

    Parameters
    ----------
    state_indices : Int64[NDArray, "n 3"]
        Nonempty retained reciprocal indices in ``(x, y, z)`` order.

    Returns
    -------
    floor : Fraction
        Positive rational lower bound for the shell compression.

    Raises
    ------
    ValueError
        If ``state_indices`` does not have shape ``(n, 3)`` or is empty.

    Notes
    -----
    Let ``N_j`` be the inclusive integer span of the state support on axis
    ``j`` and put ``m_j = N_j + 1``. The sharp floor on the enclosing
    rectangular support is

    ``1 - product_j cos(pi / (2 m_j))**2``.

    No transcendental value enters this checker. For ``m = 2`` it uses the
    exact identity ``sin(pi / 4)**2 = 1 / 2``. For ``m >= 3``, concavity of
    sine on ``[0, pi / 6]`` proves
    ``sin(pi / (2 m))**2 >= 9 / (4 m**2)``. Replacing each squared cosine by
    the resulting rational upper bound and multiplying gives a rigorous
    lower bound. A principal-compression argument transfers the rectangular
    result to every nonempty finite support inside that box.
    """
    if state_indices.ndim != _SUPPORT_RANK or state_indices.shape[1:] != (
        _SPACE_DIMENSIONS,
    ):
        raise ValueError("state_indices must have shape (n, 3)")
    if state_indices.shape[0] == 0:
        raise ValueError("state_indices must be nonempty")

    cosine_product_upper: Fraction = Fraction(1)
    for axis in range(3):
        coordinates: list[int] = [int(index[axis]) for index in state_indices]
        span: int = max(coordinates) - min(coordinates) + 1
        denominator_index: int = span + 1
        sine_squared_lower: Fraction = (
            Fraction(1, 2)
            if denominator_index == _SINGLETON_DENOMINATOR_INDEX
            else Fraction(9, 4 * denominator_index**2)
        )
        cosine_product_upper *= Fraction(1) - sine_squared_lower
    floor: Fraction = Fraction(1) - cosine_product_upper
    return floor


def _target_matrices(
    manifest: GalerkinTargetManifest,
) -> Tuple[
    list[list[_ComplexFraction]],
    list[list[_ComplexFraction]],
    list[list[_ComplexFraction]],
]:
    """PRIVATE: Reconstruct exact interaction, absorber, and target matrices.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical target whose stored binary64 coefficients are reconstructed.

    Returns
    -------
    interaction : list[list[_ComplexFraction]]
        Exact interaction compression.
    absorber : list[list[_ComplexFraction]]
        Exact absorber compression.
    target : list[list[_ComplexFraction]]
        Exact SC-1 target ``D - R - i epsilon_CAP A``.

    Notes
    -----
    Every stored binary64 component is interpreted as its exact dyadic value.
    The reconstruction performs no floating arithmetic.
    """
    state_indices: Int64[NDArray, "n 3"] = _host_array(
        manifest.support.state_indices
    )
    interaction = _matrix_from_coefficients(
        state_indices,
        _coefficient_map(
            manifest.support.interaction_indices,
            manifest.interaction_coefficients,
        ),
    )
    absorber = _matrix_from_coefficients(
        state_indices,
        _coefficient_map(
            manifest.support.absorber_indices,
            manifest.absorber_coefficients,
        ),
    )
    diagonal_array: Float64[NDArray, " n"] = _host_array(
        manifest.free_diagonal
    )
    cap: Fraction = _fraction_from_float(
        float(_host_array(manifest.cap_scale))
    )
    target: list[list[_ComplexFraction]] = []
    for row in range(len(interaction)):
        target_row: list[_ComplexFraction] = []
        for column in range(len(interaction)):
            diagonal: _ComplexFraction = (
                _fraction_from_float(float(diagonal_array[row]))
                if row == column
                else Fraction(0),
                Fraction(0),
            )
            real_part: _ComplexFraction = _complex_subtract(
                diagonal, interaction[row][column]
            )
            cap_part: _ComplexFraction = (
                cap * absorber[row][column][1],
                -cap * absorber[row][column][0],
            )
            target_row.append(_complex_add(real_part, cap_part))
        target.append(target_row)
    matrices: Tuple[
        list[list[_ComplexFraction]],
        list[list[_ComplexFraction]],
        list[list[_ComplexFraction]],
    ] = (interaction, absorber, target)
    return matrices


def _exact_vector(
    value: Complex128[jax.Array, " n"],
) -> list[_ComplexFraction]:
    """PRIVATE: Convert one complex binary vector to dyadic components.

    Parameters
    ----------
    value : Complex128[jax.Array, " n"]
        Binary64-complex vector to reconstruct.

    Returns
    -------
    vector : list[_ComplexFraction]
        Exact dyadic components in the original coefficient order.
    """
    array: Complex128[NDArray, " n"] = _host_array(value)
    vector: list[_ComplexFraction] = [
        _complex_fraction(complex(entry)) for entry in array
    ]
    return vector


def _matrix_action(
    matrix: list[list[_ComplexFraction]],
    vector: list[_ComplexFraction],
) -> list[_ComplexFraction]:
    """PRIVATE: Apply one exact complex-rational dense matrix.

    Parameters
    ----------
    matrix : list[list[_ComplexFraction]]
        Dense exact matrix in row-major nested-list form.
    vector : list[_ComplexFraction]
        Exact input vector in column order.

    Returns
    -------
    result : list[_ComplexFraction]
        Exact matrix-vector action in row order.
    """
    zero: _ComplexFraction = (Fraction(0), Fraction(0))
    result: list[_ComplexFraction] = []
    for row in matrix:
        accumulated: _ComplexFraction = zero
        for value, entry in zip(row, vector, strict=True):
            accumulated = _complex_add(
                accumulated, _complex_multiply(value, entry)
            )
        result.append(accumulated)
    return result


def _residual_squared(
    target: list[list[_ComplexFraction]],
    total_source: Complex128[jax.Array, " n"],
    solve_result: GalerkinSolveResult,
) -> Fraction:
    """PRIVATE: Recompute the same-target residual squared exactly.

    Parameters
    ----------
    target : list[list[_ComplexFraction]]
        Exact target matrix reconstructed from the manifest.
    total_source : Complex128[jax.Array, " n"]
        Stored total right-hand side for the selected source route.
    solve_result : GalerkinSolveResult
        Bound submitted field; its reported residual is ignored.

    Returns
    -------
    squared : Fraction
        Exact squared Euclidean norm of ``source - target @ field``.

    Notes
    -----
    The calculation uses only exact rational additions and multiplications.
    """
    right_hand_side = _exact_vector(total_source)
    field = _exact_vector(solve_result.field)
    applied = _matrix_action(target, field)
    residual = [
        _complex_subtract(rhs, image)
        for rhs, image in zip(right_hand_side, applied, strict=True)
    ]
    squared: Fraction = sum(
        (value[0] ** 2 + value[1] ** 2 for value in residual),
        start=Fraction(0),
    )
    return squared


def _vector_norm_squared(
    value: Complex128[jax.Array, " n"],
) -> Fraction:
    """PRIVATE: Reconstruct one stored vector's squared norm exactly.

    Parameters
    ----------
    value : Complex128[jax.Array, " n"]
        Stored binary64-complex coefficient vector.

    Returns
    -------
    squared : Fraction
        Exact rational sum of squared stored real and imaginary components.
    """
    vector: list[_ComplexFraction] = _exact_vector(value)
    squared: Fraction = sum(
        (entry[0] ** 2 + entry[1] ** 2 for entry in vector),
        start=Fraction(0),
    )
    return squared


def _proof(  # noqa: PLR0913
    target_digest: str,
    result_digest: str,
    floor: Fraction,
    residual_squared: Fraction,
    budget: Fraction,
    failure: GalerkinStabilityFailure,
    route: GalerkinStabilityRoute = GalerkinStabilityRoute.ABSORBER_FLOOR,
    *,
    algebraic_floor: Fraction | None = None,
    transferred_floor: Fraction = Fraction(0),
    transferred_floor_finite: bool = False,
    field_norm_squared: Fraction = Fraction(0),
    exact_target_residual_upper: Fraction = Fraction(0),
    exact_target_residual_finite: bool = False,
    source_lift: _SourceLift = _LEGACY_SOURCE_LIFT,
) -> GalerkinStabilityProof:
    """PRIVATE: Construct one canonical checker proof payload.

    Parameters
    ----------
    target_digest : str
        Canonical bound-target checksum.
    result_digest : str
        Canonical bound-submission checksum.
    floor : Fraction
        Exact selected direct exact-target lower singular-value bound.
    residual_squared : Fraction
        Exact squared original-system residual.
    budget : Fraction
        Exact preregistered state-error budget.
    failure : GalerkinStabilityFailure
        Typed checker failure, or ``NONE``.
    route : GalerkinStabilityRoute
        Exact proof route. Default is ``ABSORBER_FLOOR``.
    algebraic_floor : Fraction | None
        Exact ``H_alg`` Route-A floor, or ``floor`` when omitted.
    transferred_floor : Fraction
        Signed perturbative margin ``algebraic_floor - delta_H``. Default:
        zero.
    transferred_floor_finite : bool
        Whether ``transferred_floor`` is finite and meaningful. Default:
        false.
    field_norm_squared : Fraction
        Exact squared norm of the submitted stored field. Default: zero.
    exact_target_residual_upper : Fraction
        Rational representation of the final directed-up exact-target
        residual enclosure. Default: zero.
    exact_target_residual_finite : bool
        Whether the residual lift is finite. Default: false.
    source_lift : _SourceLift
        Exact ``delta_S`` bound plus route and scope metadata. Default: the
        legacy exact-stored-RHS route with ``delta_S = 0``.

    Returns
    -------
    proof : GalerkinStabilityProof
        Canonical exact-integer proof carrier.

    Notes
    -----
    A negative floor is stored as zero so invalid bounds fail closed.
    """
    stored_algebraic_floor: Fraction = (
        floor if algebraic_floor is None else algebraic_floor
    )
    proof: GalerkinStabilityProof = create_galerkin_stability_proof(
        target_digest=target_digest,
        result_digest=result_digest,
        algebraic_floor_numerator=max(stored_algebraic_floor.numerator, 0),
        algebraic_floor_denominator=stored_algebraic_floor.denominator,
        transferred_floor_numerator=(
            transferred_floor.numerator if transferred_floor_finite else 0
        ),
        transferred_floor_denominator=transferred_floor.denominator,
        transferred_floor_finite=transferred_floor_finite,
        floor_numerator=max(floor.numerator, 0),
        floor_denominator=floor.denominator,
        residual_squared_numerator=residual_squared.numerator,
        residual_squared_denominator=residual_squared.denominator,
        field_norm_squared_numerator=field_norm_squared.numerator,
        field_norm_squared_denominator=field_norm_squared.denominator,
        exact_target_residual_upper_numerator=(
            exact_target_residual_upper.numerator
            if exact_target_residual_finite
            else 0
        ),
        exact_target_residual_upper_denominator=(
            exact_target_residual_upper.denominator
        ),
        exact_target_residual_finite=exact_target_residual_finite,
        source_error_upper_numerator=(
            source_lift.error_upper.numerator if source_lift.finite else 0
        ),
        source_error_upper_denominator=source_lift.error_upper.denominator,
        source_error_finite=source_lift.finite,
        state_budget_numerator=budget.numerator,
        state_budget_denominator=budget.denominator,
        route=route,
        failure=failure,
        checker_id=_CHECKER_ID,
        rhs_target=source_lift.rhs_target,
        residual_scope=source_lift.residual_scope,
        source_error_route=source_lift.error_route,
        source_error_scope=source_lift.error_scope,
    )
    return proof


def _check_galerkin_absorber_floor(  # noqa: PLR0911, PLR0912, PLR0915
    manifest: GalerkinTargetManifest,
    source: _BoundSource,
    solve_result: GalerkinSolveResult,
    state_budget: scalar_float,
    *,
    source_lift: _SourceLift,
    represented_source: bool,
) -> GalerkinStabilityProof:
    r"""PRIVATE: Build one legacy or represented-source Route-A proof.

    Implementation Logic
    --------------------
    1. Bind the target, stored source, submitted state, and state budget.
    2. Reconstruct ``R``, ``A``, ``H``, and the residual with Fractions.
    3. Prove ``A >= mu I`` with the rational cosine-box floor and, for at
       most 32 modes, select the stronger exact Gershgorin floor.
    4. Apply Route A directly to the exact target, then lift the independent
       ``H_alg`` residual by the outward RM-S2 term ``delta_H ||x||``.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to reconstruct.
    source : _BoundSource
        Bound source carrier for the selected invocation route.
    solve_result : GalerkinSolveResult
        Submitted manifested binary64 solver result. Its field and residual
        must be Complex128, its metrics Float64, and its counters and status
        Int32. The generic algebraic carrier may preserve other dtypes, but
        those results are outside this checker. Its reported residual is not
        trusted.
    state_budget : scalar_float
        Positive normal-range preregistered state-error budget.
    source_lift : _SourceLift
        Frozen source error, target, route, and scope metadata.
    represented_source : bool
        Require full RM-S3 source reconstruction and eligibility when true.

    Returns
    -------
    proof : GalerkinStabilityProof
        Exact per-result checker payload, including a typed failure.

    Raises
    ------
    ValueError
        If the state budget is non-scalar, non-finite, or below the smallest
        normal binary64 value.

    Notes
    -----
    Canonical reconstruction proves that the exact free and interaction
    operators are Hermitian and that the exact target uses the identical
    manifested dyadic cosine CAP. Route A therefore applies directly to the
    exact target; its floor is derived again, not copied from ``H_alg``. The
    separately bound perturbative margin ``s_alg - delta_H`` is diagnostic
    and may be weaker or negative. Ordinary floating eigenvalues,
    transcendental-library values, and producer Boolean assertions are not
    proof inputs. The analytic box floor has no dimension cap, while the
    supplementary dense Gershgorin route is limited to 32 retained
    coefficients. Algebraic residual reconstruction remains dense and
    quadratic. The payload is tied to this target, source, state, and budget;
    another result requires another invocation. The legacy route treats its
    stored total source as exact. The represented route adds its certified
    exact-target total-source error once. Per-call residual-formation errors
    remain separate.
    """
    budget_array: Num[NDArray, ""] = _host_array(jnp.asarray(state_budget))
    if budget_array.shape != ():
        raise ValueError("state_budget must be a scalar")
    budget_float: float = float(budget_array)
    if not math.isfinite(budget_float) or budget_float < float(
        np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "state_budget must be finite and at least the smallest normal "
            "float64"
        )
    budget: Fraction = _fraction_from_float(budget_float)
    try:
        target_digest: str = _target_digest(manifest)
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        proof: GalerkinStabilityProof = _proof(
            _INVALID_TARGET_DIGEST,
            _INVALID_RESULT_DIGEST,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT,
            source_lift=source_lift,
        )
        return proof
    try:
        result_digest: str = _result_digest(
            target_digest, source, solve_result
        )
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        proof = _proof(
            target_digest,
            _INVALID_RESULT_DIGEST,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT,
            source_lift=source_lift,
        )
        return proof  # noqa: RET504
    try:
        dimension: int = manifest.support.state_indices.shape[0]
    except (AttributeError, IndexError, TypeError):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT,
            source_lift=source_lift,
        )
        return proof  # noqa: RET504
    if not _manifest_is_canonical(manifest):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT,
            source_lift=source_lift,
        )
        return proof  # noqa: RET504

    if represented_source:
        source_is_canonical: bool = isinstance(
            source, GalerkinRepresentedSource
        ) and _represented_source_is_canonical(manifest, source)
    else:
        source_is_canonical = isinstance(
            source, GalerkinSource
        ) and _source_is_canonical(source)
    if not source_is_canonical or not _solve_result_is_canonical(solve_result):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT,
            source_lift=source_lift,
        )
        return proof  # noqa: RET504

    if represented_source:
        if not source_lift.finite:
            proof = _proof(
                target_digest,
                result_digest,
                Fraction(0),
                Fraction(0),
                budget,
                GalerkinStabilityFailure.NO_FINITE_EXACT_TARGET_SOURCE_ERROR_BOUND,
                source_lift=source_lift,
            )
            return proof  # noqa: RET504
        represented = cast(GalerkinRepresentedSource, source)
        if not bool(_host_array(represented.rm_s3_eligible)):
            proof = _proof(
                target_digest,
                result_digest,
                Fraction(0),
                Fraction(0),
                budget,
                GalerkinStabilityFailure.SOURCE_NOT_RM_S3_ELIGIBLE,
                source_lift=source_lift,
            )
            return proof  # noqa: RET504

    try:
        state_indices: Int64[NDArray, "n 3"] = _host_array(
            manifest.support.state_indices
        )
        box_floor: Fraction = _rational_cosine_shell_box_floor(state_indices)
        interaction, absorber, target = _target_matrices(manifest)
        total_source: Complex128[jax.Array, " n"] = (
            source.actions.total_source
            if isinstance(source, GalerkinRepresentedSource)
            else source.total_source
        )
        residual_squared: Fraction = _residual_squared(
            target, total_source, solve_result
        )
        field_norm_squared: Fraction = _vector_norm_squared(solve_result.field)
        delta_h_float: float = float(
            _host_array(
                manifest.fixed_linear_error_ledger.fixed_linear_operator_error_bound
            )
        )
        exact_target_residual_upper: Fraction
        exact_target_residual_finite: bool
        (
            exact_target_residual_upper,
            exact_target_residual_finite,
        ) = _lift_exact_target_residual_up(
            residual_squared,
            field_norm_squared,
            delta_h_float,
            source_lift.error_upper,
            source_lift.finite,
        )
    except (ArithmeticError, IndexError, TypeError, ValueError):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT,
            source_lift=source_lift,
        )
        return proof  # noqa: RET504
    if not _is_hermitian(interaction) or not _is_hermitian(absorber):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT,
            source_lift=source_lift,
        )
        return proof  # noqa: RET504
    if dimension <= _MAX_GERSHGORIN_DIMENSION:
        gershgorin_floor: Fraction = _absorber_floor(absorber)
        if gershgorin_floor >= box_floor:
            absorber_floor: Fraction = gershgorin_floor
            route: GalerkinStabilityRoute = (
                GalerkinStabilityRoute.ABSORBER_FLOOR_GERSHGORIN
            )
        else:
            absorber_floor = box_floor
            route = GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX
    else:
        absorber_floor = box_floor
        route = GalerkinStabilityRoute.ABSORBER_FLOOR_COSINE_BOX
    cap: Fraction = _fraction_from_float(
        float(_host_array(manifest.cap_scale))
    )
    algebraic_floor: Fraction = cap * absorber_floor
    exact_floor: Fraction = algebraic_floor
    transferred_floor_finite: bool = math.isfinite(delta_h_float)
    transferred_floor: Fraction = (
        algebraic_floor - _fraction_from_float(delta_h_float)
        if transferred_floor_finite
        else Fraction(0)
    )
    failure: GalerkinStabilityFailure = (
        GalerkinStabilityFailure.NONE
        if cap > 0 and absorber_floor > 0
        else GalerkinStabilityFailure.NO_POSITIVE_ABSORBER_FLOOR
    )
    proof = _proof(
        target_digest,
        result_digest,
        exact_floor,
        residual_squared,
        budget,
        failure,
        route,
        algebraic_floor=algebraic_floor,
        transferred_floor=transferred_floor,
        transferred_floor_finite=transferred_floor_finite,
        field_norm_squared=field_norm_squared,
        exact_target_residual_upper=exact_target_residual_upper,
        exact_target_residual_finite=exact_target_residual_finite,
        source_lift=source_lift,
    )
    return proof  # noqa: RET504


@beartype
def check_galerkin_absorber_floor(
    manifest: GalerkinTargetManifest,
    source: GalerkinSource,
    solve_result: GalerkinSolveResult,
    state_budget: scalar_float,
) -> GalerkinStabilityProof:
    """Build the bounded exact-stored-RHS algebraic-oracle proof.

    :see: :class:`~.test_stability.TestGalerkinStabilityInvocation`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to reconstruct.
    source : GalerkinSource
        Legacy finite source whose stored total vector defines the exact RHS.
    solve_result : GalerkinSolveResult
        Submitted binary64 solver result; reported residuals are not trusted.
    state_budget : scalar_float
        Positive normal-range preregistered state-error budget.

    Returns
    -------
    proof : GalerkinStabilityProof
        Exact algebraic-oracle proof payload or typed rejection.

    Raises
    ------
    ValueError
        If the state budget is structurally invalid.

    Notes
    -----
    This bounded compatibility surface is an algebraic oracle, not a
    production source path. It deliberately defines the stored binary64 total
    source as its exact right-hand side, so ``delta_S = 0`` and makes no
    analytic RM-S3 source-conformity claim. Production callers must use
    :func:`check_represented_galerkin_absorber_floor` for a rebuilt eligible
    represented source and explicit exact-target source lift. This route can
    retire after solver fixtures and remaining consumers migrate from
    :class:`GalerkinSource` to :class:`GalerkinRepresentedSource`.
    """
    proof: GalerkinStabilityProof = _check_galerkin_absorber_floor(
        manifest,
        source,
        solve_result,
        state_budget,
        source_lift=_LEGACY_SOURCE_LIFT,
        represented_source=False,
    )
    return proof


@beartype
def check_represented_galerkin_absorber_floor(
    manifest: GalerkinTargetManifest,
    source: GalerkinRepresentedSource,
    solve_result: GalerkinSolveResult,
    state_budget: scalar_float,
) -> GalerkinStabilityProof:
    r"""Build a Route-A proof with an eligible rebuilt RM-S3 source.

    :see: :class:`~.test_stability.TestRepresentedSourceStabilityInvocation`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to reconstruct.
    source : GalerkinRepresentedSource
        Represented plane or focused source to rebuild and bind.
    solve_result : GalerkinSolveResult
        Submitted binary64 solver result; reported residuals are not trusted.
    state_budget : scalar_float
        Positive normal-range preregistered state-error budget.

    Returns
    -------
    proof : GalerkinStabilityProof
        Exact represented-source proof payload or typed rejection.

    Raises
    ------
    ValueError
        If the state budget is structurally invalid.

    Notes
    -----
    The checker binds ``source.manifest`` exactly to ``manifest``, rebuilds
    the plane or focused source from its primitive stored inputs, compares the
    complete payload, requires ``rm_s3_eligible``, and certifies

    ``rho_exact <= rho_alg + delta_H ||x|| + delta_S_total``.

    Here ``rho_alg`` uses ``source.actions.total_source`` and
    ``delta_S_total`` is the source's exact-target total-source enclosure.
    The source term is added once and is distinct from the operator action on
    the submitted state.
    """
    try:
        source_error_float: float = float(
            _host_array(
                source.error_enclosure.exact_target_total_source_error_upper_bound
            )
        )
        source_error_finite: bool = (
            math.isfinite(source_error_float) and source_error_float >= 0.0
        )
        source_error_upper: Fraction = (
            _fraction_from_float(source_error_float)
            if source_error_finite
            else Fraction(0)
        )
        error_route: str = source.error_enclosure.route.value
        error_scope: str = source.error_enclosure.error_scope
    except (ArithmeticError, AttributeError, TypeError, ValueError):
        source_error_finite = False
        source_error_upper = Fraction(0)
        error_route = "invalid_represented_source_error_route"
        error_scope = "invalid represented source-error payload"
    source_lift = _SourceLift(
        error_upper=source_error_upper,
        finite=source_error_finite,
        rhs_target=_REPRESENTED_RHS_TARGET,
        residual_scope=_REPRESENTED_RESIDUAL_SCOPE,
        error_route=error_route,
        error_scope=error_scope,
    )
    proof: GalerkinStabilityProof = _check_galerkin_absorber_floor(
        manifest,
        source,
        solve_result,
        state_budget,
        source_lift=source_lift,
        represented_source=True,
    )
    return proof


def _proof_payload(proof: GalerkinStabilityProof) -> Tuple[object, ...]:
    """PRIVATE: Order every proof field for canonical comparison.

    Parameters
    ----------
    proof : GalerkinStabilityProof
        Exact checker proof carrier.

    Returns
    -------
    payload : Tuple[object, ...]
        Every proof field in factory declaration order.

    Notes
    -----
    Invocation uses this tuple to reject any proof-field mutation.
    """
    payload: Tuple[object, ...] = (
        proof.target_digest,
        proof.result_digest,
        proof.algebraic_floor_numerator,
        proof.algebraic_floor_denominator,
        proof.transferred_floor_numerator,
        proof.transferred_floor_denominator,
        proof.transferred_floor_finite,
        proof.floor_numerator,
        proof.floor_denominator,
        proof.residual_squared_numerator,
        proof.residual_squared_denominator,
        proof.field_norm_squared_numerator,
        proof.field_norm_squared_denominator,
        proof.exact_target_residual_upper_numerator,
        proof.exact_target_residual_upper_denominator,
        proof.exact_target_residual_finite,
        proof.source_error_upper_numerator,
        proof.source_error_upper_denominator,
        proof.source_error_finite,
        proof.state_budget_numerator,
        proof.state_budget_denominator,
        proof.route,
        proof.failure,
        proof.checker_id,
        proof.rhs_target,
        proof.residual_scope,
        proof.source_error_route,
        proof.source_error_scope,
    )
    return payload


def _positive_fraction_to_float_down(value: Fraction) -> float:
    """PRIVATE: Convert a positive rational downward to a binary float.

    Parameters
    ----------
    value : Fraction
        Positive exact rational value.

    Returns
    -------
    candidate : float
        Greatest reached binary64 value that does not exceed ``value``.

    Notes
    -----
    Overflow saturates at the largest finite binary64 value. ``nextafter``
    corrects any round-to-nearest overshoot.
    """
    try:
        candidate: float = float(value)
    except OverflowError:
        candidate = _MAX_BINARY64
    if math.isinf(candidate):
        candidate = _MAX_BINARY64
    while Fraction.from_float(candidate) > value:
        candidate = math.nextafter(candidate, -math.inf)
    return candidate


def _sqrt_fraction_to_float_up(value: Fraction) -> float:
    """PRIVATE: Convert a rational square root upward to a binary float.

    Parameters
    ----------
    value : Fraction
        Non-negative exact rational radicand.

    Returns
    -------
    candidate : float
        Binary64 enclosure greater than or equal to ``sqrt(value)``.

    Notes
    -----
    An 80-digit decimal estimate seeds the conversion. Exact rational squaring
    and ``nextafter`` then enforce the outward direction.
    """
    if value == 0:
        candidate: float = 0.0
        return candidate
    with localcontext() as context:
        context.prec = 80
        decimal_value: Decimal = Decimal(value.numerator) / Decimal(
            value.denominator
        )
        candidate: float = float(decimal_value.sqrt())
    if math.isinf(candidate):
        return candidate
    if candidate == 0.0:
        candidate = math.nextafter(0.0, math.inf)
    while Fraction.from_float(candidate) ** 2 < value:
        candidate = math.nextafter(candidate, math.inf)
        if math.isinf(candidate):
            return candidate
    return candidate


def _nonnegative_fraction_to_float_up(value: Fraction) -> float:
    """PRIVATE: Convert a non-negative rational upward to binary64.

    Parameters
    ----------
    value : Fraction
        Non-negative exact rational value.

    Returns
    -------
    candidate : float
        Least reached binary64 value not smaller than ``value``, or positive
        infinity when no finite binary64 enclosure exists.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    if value < 0:
        raise ValueError("value must be non-negative")
    try:
        candidate: float = float(value)
    except OverflowError:
        candidate = math.inf
    if math.isinf(candidate):
        return candidate
    while Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


def _lift_exact_target_residual_up(
    algebraic_residual_squared: Fraction,
    field_norm_squared: Fraction,
    fixed_linear_error_bound: float,
    source_error_upper_bound: Fraction = Fraction(0),
    source_error_finite: bool = True,
) -> Tuple[Fraction, bool]:
    r"""PRIVATE: Lift an exact ``H_alg`` residual to the exact target.

    Parameters
    ----------
    algebraic_residual_squared : Fraction
        Exact squared residual for the independently reconstructed stored
        ``H_alg`` action and stored right-hand side.
    field_norm_squared : Fraction
        Exact squared norm of the submitted stored field.
    fixed_linear_error_bound : float
        Manifested outward RM-S2 ``delta_H`` bound.
    source_error_upper_bound : Fraction
        Exact rational value of the stored outward ``delta_S`` bound.
        Default: zero.
    source_error_finite : bool
        Whether the source-error bound is finite. Default: true.

    Returns
    -------
    residual_upper : Fraction
        Exact rational representation of the final directed-up binary64
        residual enclosure, or zero when no finite enclosure exists.
    finite : bool
        Whether every directed-up square root, product, and sum is finite.

    Notes
    -----
    The legacy route supplies ``delta_S = 0``. The represented-source route
    supplies its RM-S3 exact-target total-source enclosure, giving

    ``rho_exact <= rho_alg + delta_H ||x|| + delta_S``.

    Each square root, product, and sum is rounded upward separately, then its
    stored binary64 value is converted back to an exact rational. Analytic
    The source term is added exactly once. Per-call residual-formation errors
    remain outside this proof scope.
    """
    finite: bool = (
        not (
            math.isnan(fixed_linear_error_bound)
            or fixed_linear_error_bound < 0.0
        )
        and source_error_finite
        and source_error_upper_bound >= 0
    )
    residual_upper: Fraction = Fraction(0)
    algebraic_residual_up: float = math.inf
    field_norm_up: float = math.inf
    transfer_up: float = math.inf
    sum_up: float = math.inf
    if finite:
        algebraic_residual_up = _sqrt_fraction_to_float_up(
            algebraic_residual_squared
        )
        field_norm_up = _sqrt_fraction_to_float_up(field_norm_squared)
        finite = math.isfinite(algebraic_residual_up) and math.isfinite(
            field_norm_up
        )
    if finite:
        if field_norm_squared == 0:
            transfer_up = 0.0
        elif math.isfinite(fixed_linear_error_bound):
            transfer_exact: Fraction = _fraction_from_float(
                fixed_linear_error_bound
            ) * _fraction_from_float(field_norm_up)
            transfer_up = _nonnegative_fraction_to_float_up(transfer_exact)
            finite = math.isfinite(transfer_up)
        else:
            finite = False
    if finite:
        sum_exact: Fraction = (
            _fraction_from_float(algebraic_residual_up)
            + _fraction_from_float(transfer_up)
            + source_error_upper_bound
        )
        sum_up = _nonnegative_fraction_to_float_up(sum_exact)
        finite = math.isfinite(sum_up)
    if finite:
        residual_upper = _fraction_from_float(sum_up)
    result: Tuple[Fraction, bool] = (residual_upper, finite)
    return result


def _rejected_result(
    expected: GalerkinStabilityProof,
    failure: GalerkinStabilityFailure,
) -> GalerkinStabilityResult:
    """PRIVATE: Construct one fail-closed rejected invocation.

    Parameters
    ----------
    expected : GalerkinStabilityProof
        Recomputed proof that supplies bound provenance and budget.
    failure : GalerkinStabilityFailure
        Typed reason for rejection.

    Returns
    -------
    result : GalerkinStabilityResult
        Rejected result with zero lower bound and infinite upper bounds.

    Notes
    -----
    Rejection preserves target, result, route, checker, and budget provenance.
    """
    budget: float = float(
        Fraction(
            expected.state_budget_numerator,
            expected.state_budget_denominator,
        )
    )
    result: GalerkinStabilityResult = create_galerkin_stability_result(
        lower_singular_bound=0.0,
        residual_upper_bound=math.inf,
        state_error_upper_bound=math.inf,
        state_budget=budget,
        route=expected.route,
        disposition=GalerkinStabilityDisposition.REJECTED,
        failure=failure,
        target_digest=expected.target_digest,
        result_digest=expected.result_digest,
        checker_id=_CHECKER_ID,
    )
    return result


def _invoke_galerkin_stability(  # noqa: PLR0911
    manifest: GalerkinTargetManifest,
    source: _BoundSource,
    solve_result: GalerkinSolveResult,
    proof: GalerkinStabilityProof,
    state_budget: scalar_float,
    *,
    represented_source: bool,
) -> GalerkinStabilityResult:
    r"""PRIVATE: Recheck one legacy or represented-source proof.

    Implementation Logic
    --------------------
    1. Reconstruct the proof from the bound target, source, state, and budget.
    2. Reject every checksum, arithmetic, checker, or proof mutation.
    3. Compare the directed-up exact-target residual enclosure against the
       exact state-budget inequality.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to reconstruct.
    source : _BoundSource
        Bound source carrier for the selected route.
    solve_result : GalerkinSolveResult
        Bound manifested binary64 state and solver provenance. Its field and
        residual must be Complex128, its metrics Float64, and its counters and
        status Int32; polymorphic algebraic carrier instances are outside this
        invocation.
    proof : GalerkinStabilityProof
        Checker payload to reconstruct and validate.
    state_budget : scalar_float
        Independently supplied preregistered positive normal-range state-error
        budget.
    represented_source : bool
        Rebuild and require the represented RM-S3 route when true.

    Returns
    -------
    result : GalerkinStabilityResult
        Operational pass, typed fallback, or fail-closed rejection.

    Raises
    ------
    ValueError
        If the independent state budget is non-scalar, non-finite, or below
        the smallest normal binary64 value.

    Notes
    -----
    The caller supplies the preregistered budget independently of the proof
    payload. The state-budget decision uses
    ``rho_exact_upper <= state_budget * s_exact`` in exact rational
    arithmetic, where ``rho_exact_upper`` is the directed-up lift of the
    independently reconstructed ``H_alg`` residual. Floating values in the
    result are outward reporting bounds. The invocation applies only to this
    retained result and is not reusable.
    """
    budget_array: Num[NDArray, ""] = _host_array(jnp.asarray(state_budget))
    if budget_array.shape != ():
        raise ValueError("state_budget must be a scalar")
    budget_float: float = float(budget_array)
    if not math.isfinite(budget_float) or budget_float < float(
        np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "state_budget must be finite and at least the smallest normal "
            "float64"
        )
    budget: Fraction = _fraction_from_float(budget_float)
    if represented_source:
        expected: GalerkinStabilityProof = (
            check_represented_galerkin_absorber_floor(
                manifest,
                cast(GalerkinRepresentedSource, source),
                solve_result,
                budget_float,
            )
        )
    else:
        expected = check_galerkin_absorber_floor(
            manifest,
            cast(GalerkinSource, source),
            solve_result,
            budget_float,
        )
    if _proof_payload(proof) != _proof_payload(expected):
        result: GalerkinStabilityResult = _rejected_result(
            expected, GalerkinStabilityFailure.PROOF_RECORD_MISMATCH
        )
        return result
    if expected.failure is not GalerkinStabilityFailure.NONE:
        result = _rejected_result(expected, expected.failure)
        return result  # noqa: RET504

    floor: Fraction = Fraction(
        expected.floor_numerator, expected.floor_denominator
    )
    if not expected.exact_target_residual_finite:
        residual_failure: GalerkinStabilityFailure = (
            GalerkinStabilityFailure.ARITHMETIC_RANGE_FAILURE
            if expected.transferred_floor_finite
            else GalerkinStabilityFailure.NO_FINITE_EXACT_TARGET_RESIDUAL_BOUND
        )
        result = _rejected_result(
            expected,
            residual_failure,
        )
        return result  # noqa: RET504
    exact_target_residual_upper: Fraction = Fraction(
        expected.exact_target_residual_upper_numerator,
        expected.exact_target_residual_upper_denominator,
    )
    budget_passes: bool = exact_target_residual_upper <= budget * floor
    max_binary64: Fraction = Fraction.from_float(_MAX_BINARY64)
    if floor > max_binary64:
        result = _rejected_result(
            expected, GalerkinStabilityFailure.ARITHMETIC_RANGE_FAILURE
        )
        return result  # noqa: RET504
    lower_bound: float = _positive_fraction_to_float_down(floor)
    if lower_bound <= 0.0:
        result = _rejected_result(
            expected, GalerkinStabilityFailure.ARITHMETIC_RANGE_FAILURE
        )
        return result  # noqa: RET504
    residual_upper: float = _nonnegative_fraction_to_float_up(
        exact_target_residual_upper
    )
    state_upper: float = _nonnegative_fraction_to_float_up(
        exact_target_residual_upper / floor
    )
    if not math.isfinite(residual_upper) or not math.isfinite(state_upper):
        result = _rejected_result(
            expected, GalerkinStabilityFailure.ARITHMETIC_RANGE_FAILURE
        )
        return result  # noqa: RET504
    disposition: GalerkinStabilityDisposition = (
        GalerkinStabilityDisposition.OPERATIONAL_PASS
        if budget_passes
        else GalerkinStabilityDisposition.TYPED_FALLBACK
    )
    failure: GalerkinStabilityFailure = (
        GalerkinStabilityFailure.NONE
        if budget_passes
        else GalerkinStabilityFailure.STATE_BUDGET_MISSED
    )
    result = create_galerkin_stability_result(
        lower_singular_bound=lower_bound,
        residual_upper_bound=residual_upper,
        state_error_upper_bound=state_upper,
        state_budget=float(budget),
        route=expected.route,
        disposition=disposition,
        failure=failure,
        target_digest=expected.target_digest,
        result_digest=expected.result_digest,
        checker_id=_CHECKER_ID,
    )
    return result  # noqa: RET504


@beartype
def invoke_galerkin_stability(
    manifest: GalerkinTargetManifest,
    source: GalerkinSource,
    solve_result: GalerkinSolveResult,
    proof: GalerkinStabilityProof,
    state_budget: scalar_float,
) -> GalerkinStabilityResult:
    """Recheck and apply the exact-stored-RHS algebraic-oracle route.

    :see: :class:`~.test_stability.TestGalerkinStabilityInvocation`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical target bound by the proof.
    source : GalerkinSource
        Legacy finite source bound by the proof.
    solve_result : GalerkinSolveResult
        Submitted binary64 state and solver provenance.
    proof : GalerkinStabilityProof
        Exact-stored-RHS checker payload to reconstruct.
    state_budget : scalar_float
        Independently supplied positive normal-range state budget.

    Returns
    -------
    result : GalerkinStabilityResult
        Operational pass, typed fallback, or fail-closed rejection.

    Raises
    ------
    ValueError
        If the state budget is structurally invalid.

    Notes
    -----
    This bounded algebraic oracle has ``delta_S = 0`` by definition and is
    not the production RM-S3 source route.
    """
    result: GalerkinStabilityResult = _invoke_galerkin_stability(
        manifest,
        source,
        solve_result,
        proof,
        state_budget,
        represented_source=False,
    )
    return result


@beartype
def invoke_represented_galerkin_stability(
    manifest: GalerkinTargetManifest,
    source: GalerkinRepresentedSource,
    solve_result: GalerkinSolveResult,
    proof: GalerkinStabilityProof,
    state_budget: scalar_float,
) -> GalerkinStabilityResult:
    """Recheck and apply an eligible represented-source stability proof.

    :see: :class:`~.test_stability.TestRepresentedSourceStabilityInvocation`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical target bound by the proof.
    source : GalerkinRepresentedSource
        Rebuilt eligible RM-S3 source bound by the proof.
    solve_result : GalerkinSolveResult
        Submitted binary64 state and solver provenance.
    proof : GalerkinStabilityProof
        Represented-source checker payload to reconstruct.
    state_budget : scalar_float
        Independently supplied positive normal-range state budget.

    Returns
    -------
    result : GalerkinStabilityResult
        Operational pass, typed fallback, or fail-closed rejection.

    Raises
    ------
    ValueError
        If the state budget is structurally invalid.
    """
    result: GalerkinStabilityResult = _invoke_galerkin_stability(
        manifest,
        source,
        solve_result,
        proof,
        state_budget,
        represented_source=True,
    )
    return result


__all__: list[str] = [
    "check_galerkin_absorber_floor",
    "check_represented_galerkin_absorber_floor",
    "invoke_galerkin_stability",
    "invoke_represented_galerkin_stability",
]
