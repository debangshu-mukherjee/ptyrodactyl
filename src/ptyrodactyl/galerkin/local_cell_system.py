r"""Compose and apply the solver-ready ``LOCAL_CELL_LVT1`` target.

Extended Summary
----------------
Host composition and preparation replay every L2--L4 proof field.  The
resulting forward and formal-adjoint actions contain no host proof work and
may be closed over in JAX transforms.

Routine Listings
----------------
:func:`apply_local_cell_galerkin_target`
    Apply frozen ``H_alg = D_alg - R_alg - i B_alg``.
:func:`apply_local_cell_galerkin_target_adjoint`
    Apply the explicit formal adjoint of the same frozen target.
:func:`compose_local_cell_galerkin_target`
    Replay L2--L4 and compose one solver-ready local-cell target.
:func:`prepare_local_cell_galerkin_target`
    Full-reconstruct and exact-compare a submitted local-cell target.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Complex128, jaxtyped

from ptyrodactyl._tools import (
    has_subnormal_components,
    sha256,
    stored_value_payload,
    upward_add,
)
from ptyrodactyl.types import (
    GalerkinAxialCapFloorProof,
    GalerkinLocalCellErrorRoute,
    GalerkinLocalCellFixedLinearErrorLedger,
    GalerkinLocalCellTargetManifest,
    GalerkinTargetManifest,
    GalerkinVoxelTargetRoute,
    _make_local_cell_fixed_linear_error_ledger,
    _make_local_cell_target_manifest,
)

from .absorber import (
    apply_axial_physical_cap,
    apply_axial_physical_cap_adjoint,
    prepare_axial_cap_floor,
)
from .free_geometry import (
    FreeGeometryEnclosure,
    enclose_free_geometry,
    transfer_exact_carrier_acquisition,
)
from .local_cell_interaction import (
    apply_local_cell_interaction,
    apply_local_cell_interaction_adjoint,
)

_ACTION_FORMULA: str = "H_alg u = D_alg u - R_alg u - i B_alg u"
_ADJOINT_FORMULA: str = "H_alg^* v = D_alg v - R_alg^* v + i B_alg^* v"
_COEFFICIENT_NORM: str = "SC.12/SC.13 Euclidean complex coefficient norm"
_ERROR_SCOPE: str = (
    "fixed_linear_H_alg_minus_exact_LOCAL_CELL_LVT1_H_only; total is "
    "delta_D+delta_R+delta_B exactly once; excludes LVT9 tail, delta_A and "
    "delta_epsilon as separate addends, coupling re-audit, Gram/floor, "
    "source, per-call, residual, solver, terminal, and model errors"
)
_INTERACTION_ERROR_PROVENANCE: str = (
    "copied once from replay-authenticated LVT.18 fixed interaction error"
)
_ABSORBER_ERROR_PROVENANCE: str = (
    "delta_A and delta_epsilon retained for audit; delta_B copied once from "
    "replay-authenticated LVT.32 physical fixed-operator transfer"
)
_LEDGER_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_cell.solver_target.fixed_linear_ledger.v1"
)
_EVIDENCE_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_cell.solver_target.manifest_evidence.v1"
)
_TARGET_DIGEST_DOMAIN: str = "ptyrodactyl.local_cell.solver_target.operator.v1"
_SC1_CONTRACT_VERSION: str = "SC-1 corrected scalar Helmholtz v1"
_LVT1_CONTRACT_VERSION: str = "LVT-1 local-cell finite target v1"
_TARGET_FORMULA: str = (
    "exact H = D(k0*,k_i*,box) - P M_(sigma_H*phi_cell) P "
    "- i epsilon_CAP P M_a P"
)


def _target_digest(
    proof: GalerkinAxialCapFloorProof,
    geometry: FreeGeometryEnclosure,
) -> str:
    """PRIVATE: Digest operator primitives without proof context.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Fully replayed L4 proof supplying frozen operator primitives.
    geometry : FreeGeometryEnclosure
        Route-neutral exact and algebraic free geometry.

    Returns
    -------
    target_digest : str
        Operator-only target identity digest.
    """
    certificate = proof.coefficient_certificate
    absorber = certificate.absorber
    core = absorber.interaction_core
    compression = core.compression
    realization = compression.realization
    potential = realization.local_potential
    support = core.support
    target_digest: str = sha256(
        {
            "domain": _TARGET_DIGEST_DOMAIN,
            "target_route": GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1.value,
            "sc1_contract_version": _SC1_CONTRACT_VERSION,
            "lvt1_contract_version": _LVT1_CONTRACT_VERSION,
            "target_formula": _TARGET_FORMULA,
            "action_formula": _ACTION_FORMULA,
            "adjoint_formula": _ADJOINT_FORMULA,
            "cell_values": stored_value_payload(potential.cell_values),
            "cell_size": stored_value_payload(potential.cell_size),
            "box_size": stored_value_payload(potential.box_size),
            "cell_center_origin": stored_value_payload(
                potential.cell_center_origin
            ),
            "units": potential.units,
            "reference_value": stored_value_payload(potential.reference_value),
            "reference_semantics": potential.reference_semantics,
            "boundary": potential.boundary,
            "cell_value_semantics": potential.cell_value_semantics,
            "cell_support_convention": potential.cell_support_convention,
            "coefficient_formula": potential.coefficient_formula,
            "state_indices": stored_value_payload(support.state_indices),
            "interaction_indices": stored_value_payload(
                support.interaction_indices
            ),
            "absorber_indices": stored_value_payload(support.absorber_indices),
            "work_indices": stored_value_payload(support.work_indices),
            "work_shape": stored_value_payload(support.work_shape),
            "interaction_pair_map": stored_value_payload(
                compression.state_pair_interaction_positions
            ),
            "absorber_pair_map": stored_value_payload(
                certificate.state_pair_absorber_positions
            ),
            "accelerating_voltage_kv": stored_value_payload(
                compression.accelerating_voltage_kv
            ),
            "interaction_coupling": stored_value_payload(
                compression.interaction_coupling
            ),
            "interaction_coefficients": stored_value_payload(
                compression.interaction_coefficients
            ),
            "algebraic_carrier": stored_value_payload(
                realization.support_eligibility.manifest.carrier
            ),
            "algebraic_wavenumber": stored_value_payload(
                realization.support_eligibility.manifest.wavenumber
            ),
            "algebraic_free_diagonal": stored_value_payload(
                geometry.algebraic_free_diagonal
            ),
            "exact_carrier_normalization": geometry.exact_geometry_target,
            "axial_layer_values": stored_value_payload(absorber.layer_values),
            "terminal_axis_as_operator_axis": absorber.terminal_axis,
            "exact_cap_scale": stored_value_payload(absorber.exact_cap_scale),
            "algebraic_cap_scale": stored_value_payload(
                absorber.algebraic_cap_scale
            ),
            "absorber_coefficients": stored_value_payload(
                absorber.absorber_coefficients
            ),
            "absorber_coefficient_formula": absorber.coefficient_formula,
        }
    )
    return target_digest


def _ledger(
    proof: GalerkinAxialCapFloorProof,
    geometry: FreeGeometryEnclosure,
) -> GalerkinLocalCellFixedLinearErrorLedger:
    """PRIVATE: Compose the three disjoint fixed-matrix errors.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Fully replayed L4 proof supplying interaction and CAP errors.
    geometry : FreeGeometryEnclosure
        Route-neutral free-operator enclosure.

    Returns
    -------
    ledger : GalerkinLocalCellFixedLinearErrorLedger
        Fixed-linear ledger with ``delta_H`` charged exactly once.
    """
    certificate = proof.coefficient_certificate
    absorber = certificate.absorber
    compression = absorber.interaction_core.compression
    delta_d = geometry.free_operator_error_bound
    delta_r = compression.fixed_interaction_error_bound
    delta_a = certificate.absorber_operator_error_bound
    delta_epsilon = proof.scale_error_bound
    delta_b = proof.physical_operator_error_upper_bound
    delta_h = upward_add(upward_add(delta_d, delta_r), delta_b)
    finite = (
        jnp.isfinite(delta_d)
        & jnp.isfinite(delta_r)
        & jnp.isfinite(delta_b)
        & jnp.isfinite(delta_h)
    )
    parent_operator_digest = absorber.operator_digest
    payload = {
        "domain": _LEDGER_DIGEST_DOMAIN,
        "algebraic_free_diagonal": stored_value_payload(
            geometry.algebraic_free_diagonal
        ),
        "exact_wavenumber_lower_bound": stored_value_payload(
            geometry.exact_wavenumber_lower_bound
        ),
        "exact_wavenumber_upper_bound": stored_value_payload(
            geometry.exact_wavenumber_upper_bound
        ),
        "wavenumber_error_bound": stored_value_payload(
            geometry.wavenumber_error_bound
        ),
        "exact_carrier_lower_bounds": stored_value_payload(
            geometry.exact_carrier_lower_bounds
        ),
        "exact_carrier_upper_bounds": stored_value_payload(
            geometry.exact_carrier_upper_bounds
        ),
        "carrier_component_error_bounds": stored_value_payload(
            geometry.carrier_component_error_bounds
        ),
        "exact_free_diagonal_lower_bounds": stored_value_payload(
            geometry.exact_free_diagonal_lower_bounds
        ),
        "exact_free_diagonal_upper_bounds": stored_value_payload(
            geometry.exact_free_diagonal_upper_bounds
        ),
        "free_diagonal_error_bounds": stored_value_payload(
            geometry.free_diagonal_error_bounds
        ),
        "free_geometry_digest": geometry.free_geometry_digest,
        "parent_operator_digest": parent_operator_digest,
        "delta_D": stored_value_payload(delta_d),
        "delta_R": stored_value_payload(delta_r),
        "delta_A_audit_only": stored_value_payload(delta_a),
        "delta_epsilon_audit_only": stored_value_payload(delta_epsilon),
        "delta_B": stored_value_payload(delta_b),
        "delta_H": stored_value_payload(delta_h),
        "finite_certificate": stored_value_payload(finite),
        "exact_geometry_target": geometry.exact_geometry_target,
        "algebraic_geometry_realization": (
            geometry.algebraic_geometry_realization
        ),
        "error_scope": _ERROR_SCOPE,
        "coefficient_norm": _COEFFICIENT_NORM,
        "interaction_provenance": _INTERACTION_ERROR_PROVENANCE,
        "absorber_provenance": _ABSORBER_ERROR_PROVENANCE,
    }
    ledger_digest = sha256(payload)
    ledger: GalerkinLocalCellFixedLinearErrorLedger = (
        _make_local_cell_fixed_linear_error_ledger(
            algebraic_free_diagonal=geometry.algebraic_free_diagonal,
            exact_wavenumber_lower_bound=(
                geometry.exact_wavenumber_lower_bound
            ),
            exact_wavenumber_upper_bound=(
                geometry.exact_wavenumber_upper_bound
            ),
            wavenumber_error_bound=geometry.wavenumber_error_bound,
            exact_carrier_lower_bounds=geometry.exact_carrier_lower_bounds,
            exact_carrier_upper_bounds=geometry.exact_carrier_upper_bounds,
            carrier_component_error_bounds=(
                geometry.carrier_component_error_bounds
            ),
            exact_free_diagonal_lower_bounds=(
                geometry.exact_free_diagonal_lower_bounds
            ),
            exact_free_diagonal_upper_bounds=(
                geometry.exact_free_diagonal_upper_bounds
            ),
            free_diagonal_error_bounds=geometry.free_diagonal_error_bounds,
            free_operator_error_bound=delta_d,
            interaction_operator_error_bound=delta_r,
            absorber_operator_error_bound=delta_a,
            cap_scale_error_bound=delta_epsilon,
            cap_operator_error_bound=delta_b,
            fixed_linear_operator_error_bound=delta_h,
            finite_certificate=finite,
            exact_geometry_target=geometry.exact_geometry_target,
            algebraic_geometry_realization=(
                geometry.algebraic_geometry_realization
            ),
            interaction_error_provenance=_INTERACTION_ERROR_PROVENANCE,
            absorber_error_provenance=_ABSORBER_ERROR_PROVENANCE,
            error_scope=_ERROR_SCOPE,
            coefficient_norm=_COEFFICIENT_NORM,
            free_geometry_digest=geometry.free_geometry_digest,
            parent_operator_digest=parent_operator_digest,
            ledger_digest=ledger_digest,
        )
    )
    return ledger


def _manifest_evidence_digest(
    proof: GalerkinAxialCapFloorProof,
    geometry: FreeGeometryEnclosure,
    transfer: object,
    ledger: GalerkinLocalCellFixedLinearErrorLedger,
    target_digest: str,
    target_name: str,
) -> str:
    """PRIVATE: Digest proof context separately from operator identity.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Fully replayed L2--L4 proof.
    geometry : FreeGeometryEnclosure
        Route-neutral free-geometry evidence.
    transfer : object
        Exact-carrier acquisition transfer evidence.
    ledger : GalerkinLocalCellFixedLinearErrorLedger
        Disjoint fixed-linear error ledger.
    target_digest : str
        Operator-only target digest.
    target_name : str
        Canonically stripped target name.

    Returns
    -------
    manifest_evidence_digest : str
        Digest binding all proof and acquisition evidence.
    """
    manifest_evidence_digest: str = sha256(
        {
            "domain": _EVIDENCE_DIGEST_DOMAIN,
            "target_digest": target_digest,
            "target_name": target_name.strip(),
            "full_l2_l3_l4_proof": stored_value_payload(proof),
            "free_geometry": stored_value_payload(geometry),
            "acquisition_transfer": stored_value_payload(transfer),
            "fixed_linear_ledger": stored_value_payload(ledger),
        }
    )
    return manifest_evidence_digest


def _compose_prepared(
    proof: GalerkinAxialCapFloorProof,
    target_name: str,
) -> GalerkinLocalCellTargetManifest:
    """PRIVATE: Compose from one replay-authenticated L4 proof.

    Parameters
    ----------
    proof : GalerkinAxialCapFloorProof
        Fully replayed L4 proof.
    target_name : str
        Canonically stripped target name.

    Returns
    -------
    manifest : GalerkinLocalCellTargetManifest
        Solver-ready local-cell target manifest.

    Raises
    ------
    ValueError
        If the proof route or finite fixed-operator evidence is inadmissible.
    """
    certificate = proof.coefficient_certificate
    absorber = certificate.absorber
    compression = absorber.interaction_core.compression
    realization = compression.realization
    if (
        realization.error_route
        is not GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
        or realization.coefficient_certificate is None
    ):
        raise ValueError("LOCAL_CELL_LVT1 composition requires DIRECT LVT.13")
    if not bool(realization.coefficient_certificate.finite_certificate):
        raise ValueError("DIRECT LVT.13 coefficient evidence must be finite")
    if not bool(compression.finite_certificate):
        raise ValueError("LVT.18 interaction evidence must be finite")
    if not bool(certificate.finite_certificate):
        raise ValueError("LVT.31 CAP coefficient evidence must be finite")
    acquisition = realization.support_eligibility
    geometry = enclose_free_geometry(
        absorber.interaction_core.support.state_indices,
        acquisition,
        compression.accelerating_voltage_kv,
    )
    transfer = transfer_exact_carrier_acquisition(acquisition, geometry)
    ledger = _ledger(proof, geometry)
    if not bool(ledger.finite_certificate):
        raise ValueError("local-cell fixed-linear ledger must be finite")
    target_digest = _target_digest(proof, geometry)
    evidence_digest = _manifest_evidence_digest(
        proof,
        geometry,
        transfer,
        ledger,
        target_digest,
        target_name,
    )
    manifest: GalerkinLocalCellTargetManifest = (
        _make_local_cell_target_manifest(
            proof,
            ledger,
            transfer.incident_full_offset_max,
            transfer.outgoing_full_offset_max,
            transfer.incident_shell_defect_bounds,
            transfer.outgoing_shell_defect_bounds,
            transfer.incident_projection_error_bounds,
            transfer.outgoing_projection_error_bounds,
            target_route=GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1,
            sc1_contract_version=_SC1_CONTRACT_VERSION,
            lvt1_contract_version=_LVT1_CONTRACT_VERSION,
            target_formula=_TARGET_FORMULA,
            action_formula=_ACTION_FORMULA,
            adjoint_formula=_ADJOINT_FORMULA,
            target_digest=target_digest,
            manifest_evidence_digest=evidence_digest,
            target_name=target_name,
        )
    )
    return manifest


def compose_local_cell_galerkin_target(
    cap_floor_proof: GalerkinAxialCapFloorProof,
    *,
    target_name: str,
) -> GalerkinLocalCellTargetManifest:
    """Replay L2--L4 and compose one solver-ready local-cell target.

    :see: :func:`~.test_local_cell_system.\
test_delta_total_once_and_floor_failure_remains_solver_ready`

    Returns
    -------
    manifest : GalerkinLocalCellTargetManifest
        Fully replayed solver-ready local-cell target.
    """
    if not isinstance(cap_floor_proof, GalerkinAxialCapFloorProof):
        raise TypeError("cap_floor_proof must be GalerkinAxialCapFloorProof")
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("target_name must be nonempty")
    prepared = prepare_axial_cap_floor(cap_floor_proof)
    manifest: GalerkinLocalCellTargetManifest = _compose_prepared(
        prepared, target_name
    )
    return manifest


def prepare_local_cell_galerkin_target(
    manifest: GalerkinLocalCellTargetManifest | GalerkinTargetManifest,
) -> GalerkinLocalCellTargetManifest:
    """Full-reconstruct and exact-compare a submitted local-cell target.

    :see: :func:`~.test_local_cell_system.\
test_prepare_rejects_nested_and_manifest_forgery_and_legacy`

    The legacy type appears in the input union only so this trust boundary can
    issue an explicit route-specific rejection.  A legacy target is never
    accepted as LOCAL_CELL_LVT1 scientific evidence.

    Returns
    -------
    prepared_manifest : GalerkinLocalCellTargetManifest
        Exact-reconstructed and authenticated target.
    """
    if not isinstance(manifest, GalerkinLocalCellTargetManifest):
        raise TypeError(
            "manifest must be GalerkinLocalCellTargetManifest; legacy targets "
            "cannot enter LOCAL_CELL_LVT1 preparation"
        )
    prepared_proof = prepare_axial_cap_floor(manifest.cap_floor_proof)
    prepared_manifest: GalerkinLocalCellTargetManifest = _compose_prepared(
        prepared_proof, manifest.target_name
    )
    if stored_value_payload(prepared_manifest) != stored_value_payload(
        manifest
    ):
        raise ValueError(
            "local-cell target does not match full operator/evidence replay"
        )
    return prepared_manifest


def _checked_field(
    manifest: GalerkinLocalCellTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one field without repeating host proof work.

    Parameters
    ----------
    manifest : GalerkinLocalCellTargetManifest
        Prepared solver-ready local-cell target.
    field : Complex[Array, '...']
        Submitted coefficient field.

    Returns
    -------
    checked_field : Complex128[Array, ' n']
        Finite normal-range field on the retained support.

    Raises
    ------
    ValueError
        If target semantics or the field shape are noncanonical.
    """
    if (
        manifest.target_route is not GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1
        or manifest.sc1_contract_version != _SC1_CONTRACT_VERSION
        or manifest.lvt1_contract_version != _LVT1_CONTRACT_VERSION
        or manifest.target_formula != _TARGET_FORMULA
        or manifest.action_formula != _ACTION_FORMULA
        or manifest.adjoint_formula != _ADJOINT_FORMULA
    ):
        raise ValueError("local-cell target action semantics are noncanonical")
    values: Complex128[Array, " n"] = jnp.asarray(field, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("field must be 1D")
    if values.shape != manifest.free_diagonal.shape:
        raise ValueError("field must match the retained state support")
    checked_field: Complex128[Array, " n"] = eqx.error_if(
        values,
        (~manifest.fixed_linear_error_ledger.finite_certificate)
        | jnp.any(~jnp.isfinite(values))
        | has_subnormal_components(values),
        "local-cell target and field must be finite and normal-range",
    )
    return checked_field


def _checked_output(
    value: Complex128[Array, " n"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Fail closed outside finite normal output range.

    Parameters
    ----------
    value : Complex128[Array, ' n']
        Composed frozen target action.

    Returns
    -------
    checked_output : Complex128[Array, ' n']
        Finite normal-range action result.
    """
    checked_output: Complex128[Array, " n"] = eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)) | has_subnormal_components(value),
        "local-cell target action left finite normal binary64 range",
    )
    return checked_output


@jaxtyped(typechecker=beartype)
def apply_local_cell_galerkin_target(
    manifest: GalerkinLocalCellTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply frozen ``H_alg = D_alg - R_alg - i B_alg``.

    :see: :func:`~.test_local_cell_system.\
test_dense_forward_adjoint_dot_jit_and_vjp`

    Raw public carrier storage is not authenticated.  A target crossing any
    storage or trust boundary must first be returned by
    :func:`prepare_local_cell_galerkin_target`; transform callers then close
    over that prepared value.

    Returns
    -------
    applied_field : Complex128[Array, ' n']
        Frozen forward action on the retained support.
    """
    checked = _checked_field(manifest, field)
    interaction = apply_local_cell_interaction(
        manifest.interaction_core, checked
    )
    absorber = apply_axial_physical_cap(manifest.cap_floor_proof, checked)
    applied = manifest.free_diagonal * checked - interaction - 1j * absorber
    applied_field: Complex128[Array, " n"] = _checked_output(applied)
    return applied_field


@jaxtyped(typechecker=beartype)
def apply_local_cell_galerkin_target_adjoint(
    manifest: GalerkinLocalCellTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the explicit formal adjoint of the same frozen target.

    :see: :func:`~.test_local_cell_system.\
test_dense_forward_adjoint_dot_jit_and_vjp`

    Raw public carrier storage is not authenticated.  A target crossing any
    storage or trust boundary must first be returned by
    :func:`prepare_local_cell_galerkin_target`; transform callers then close
    over that prepared value.

    Returns
    -------
    applied_field : Complex128[Array, ' n']
        Frozen formal-adjoint action on the retained support.
    """
    checked = _checked_field(manifest, field)
    interaction = apply_local_cell_interaction_adjoint(
        manifest.interaction_core, checked
    )
    absorber = apply_axial_physical_cap_adjoint(
        manifest.cap_floor_proof, checked
    )
    applied = manifest.free_diagonal * checked - interaction + 1j * absorber
    applied_field: Complex128[Array, " n"] = _checked_output(applied)
    return applied_field


__all__: list[str] = [
    "apply_local_cell_galerkin_target",
    "apply_local_cell_galerkin_target_adjoint",
    "compose_local_cell_galerkin_target",
    "prepare_local_cell_galerkin_target",
]
