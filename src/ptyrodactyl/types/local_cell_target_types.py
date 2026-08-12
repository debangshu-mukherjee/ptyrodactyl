r"""Define the solver-ready LOCAL_CELL_LVT1 finite-target carriers.

Extended Summary
----------------
This leaf joins, without relabelling, one replay-authenticated L3 local-cell
interaction and one replay-authenticated L4 axial physical CAP.  The fixed
linear ledger contains only the three disjoint matrix errors
``delta_D``, ``delta_R``, and ``delta_B`` and their single outward total.
Source, tail, Gram, per-call, solver, and terminal evidence are intentionally
absent.

Routine Listings
----------------
:class:`GalerkinLocalCellFixedLinearErrorLedger`
    Store the disjoint fixed-linear LVT-1 error composition.
:class:`GalerkinLocalCellTargetManifest`
    Store one completed solver-ready ``LOCAL_CELL_LVT1`` target.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex128,
    Float,
    Float64,
    jaxtyped,
)

from ptyrodactyl._tools import upward_add

from .absorber_types import GalerkinAxialCapFloorProof
from .acquisition_types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportResult,
)
from .born_potential_types import GalerkinProductSupport
from .local_cell_interaction_types import (
    GalerkinLocalCellExactCompression,
    GalerkinLocalCellInteractionCore,
)
from .local_cell_types import (
    GalerkinLocalCellPotentialRealization,
    GalerkinVoxelTargetRoute,
    LocalCellPotential3D,
)

_SHA256_HEX_LENGTH: int = 64


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for one structural carrier failure.

    Parameters
    ----------
    condition : bool
        Internal value used by this helper.
    message : str
        Internal value used by this helper.

    Raises
    ------
    ValueError
        If the structural carrier condition is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 string.

    Parameters
    ----------
    value : str
        Internal value used by this helper.

    Returns
    -------
    valid : bool
        Whether the value is canonical lowercase SHA-256 text.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


class GalerkinLocalCellFixedLinearErrorLedger(eqx.Module):
    r"""Store the disjoint fixed-linear LVT-1 error composition.

    :see: :func:`~.test_local_cell_target_types.\
test_local_cell_target_carriers_keep_fixed_linear_scope_disjoint`

    ``fixed_linear_operator_error_bound`` is exactly one outward evaluation
    of ``delta_D + delta_R + delta_B``.  ``delta_A`` and the CAP-scale error
    are retained as audit components of the already-composed L4
    ``delta_B``; they are never added independently to the total.
    """

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
    interaction_operator_error_bound: Float64[Array, ""]
    absorber_operator_error_bound: Float64[Array, ""]
    cap_scale_error_bound: Float64[Array, ""]
    cap_operator_error_bound: Float64[Array, ""]
    fixed_linear_operator_error_bound: Float64[Array, ""]
    finite_certificate: Bool[Array, ""]
    exact_geometry_target: str = eqx.field(static=True)
    algebraic_geometry_realization: str = eqx.field(static=True)
    interaction_error_provenance: str = eqx.field(static=True)
    absorber_error_provenance: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    free_geometry_digest: str = eqx.field(static=True)
    parent_operator_digest: str = eqx.field(static=True)
    ledger_digest: str = eqx.field(static=True)


class GalerkinLocalCellTargetManifest(eqx.Module):
    r"""Store one completed solver-ready ``LOCAL_CELL_LVT1`` target.

    :see: :func:`~.test_local_cell_target_types.\
test_local_cell_target_carriers_keep_fixed_linear_scope_disjoint`

    The target is the fixed binary64 matrix
    ``H_alg = D_alg - R_alg - i B_alg``.  The nested floor proof is carried
    for later Route-A use, but target construction requires only its finite
    coefficient evidence; exact or realized positive-floor eligibility is
    not an operator-construction predicate.
    """

    cap_floor_proof: GalerkinAxialCapFloorProof
    fixed_linear_error_ledger: GalerkinLocalCellFixedLinearErrorLedger
    exact_target_incident_full_offset_max: Float64[Array, ""]
    exact_target_outgoing_full_offset_max: Float64[Array, ""]
    exact_target_incident_shell_defect_bounds: Float64[Array, " i"]
    exact_target_outgoing_shell_defect_bounds: Float64[Array, " o"]
    exact_target_incident_projection_error_bounds: Float64[Array, " i"]
    exact_target_outgoing_projection_error_bounds: Float64[Array, " o"]
    target_route: GalerkinVoxelTargetRoute = eqx.field(static=True)
    sc1_contract_version: str = eqx.field(static=True)
    lvt1_contract_version: str = eqx.field(static=True)
    target_formula: str = eqx.field(static=True)
    action_formula: str = eqx.field(static=True)
    adjoint_formula: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    manifest_evidence_digest: str = eqx.field(static=True)
    target_name: str = eqx.field(static=True)

    @property
    def interaction_core(self) -> GalerkinLocalCellInteractionCore:
        """Return the nested L3 interaction core."""
        certificate = self.cap_floor_proof.coefficient_certificate
        interaction_core: GalerkinLocalCellInteractionCore = (
            certificate.absorber.interaction_core
        )
        return interaction_core

    @property
    def compression(self) -> GalerkinLocalCellExactCompression:
        """Return the nested exact-compression certificate."""
        compression: GalerkinLocalCellExactCompression = (
            self.interaction_core.compression
        )
        return compression

    @property
    def realization(self) -> GalerkinLocalCellPotentialRealization:
        """Return the nested local-cell potential realization."""
        realization: GalerkinLocalCellPotentialRealization = (
            self.compression.realization
        )
        return realization

    @property
    def support_eligibility(self) -> GalerkinAcquisitionSupportResult:
        """Return the checked acquisition result."""
        support_eligibility: GalerkinAcquisitionSupportResult = (
            self.realization.support_eligibility
        )
        return support_eligibility

    @property
    def acquisition(self) -> GalerkinAcquisitionManifest:
        """Return the checked acquisition manifest."""
        acquisition: GalerkinAcquisitionManifest = (
            self.support_eligibility.manifest
        )
        return acquisition

    @property
    def support(self) -> GalerkinProductSupport:
        """Return the exact ordered product support."""
        support: GalerkinProductSupport = self.interaction_core.support
        return support

    @property
    def local_potential(self) -> LocalCellPotential3D:
        """Return the disjoint local-cell potential; no ``potential`` alias."""
        local_potential: LocalCellPotential3D = (
            self.realization.local_potential
        )
        return local_potential

    @property
    def state_indices(self) -> Array:
        """Return ordered ``I_u``."""
        state_indices: Array = self.support.state_indices
        return state_indices

    @property
    def interaction_indices(self) -> Array:
        """Return ordered ``I_chi``."""
        interaction_indices: Array = self.support.interaction_indices
        return interaction_indices

    @property
    def absorber_indices(self) -> Array:
        """Return ordered ``I_a``."""
        absorber_indices: Array = self.support.absorber_indices
        return absorber_indices

    @property
    def work_indices(self) -> Array:
        """Return ordered ``I_w``."""
        work_indices: Array = self.support.work_indices
        return work_indices

    @property
    def work_shape(self) -> Tuple[int, int, int]:
        """Return the fixed product work-grid shape."""
        work_shape: Tuple[int, int, int] = self.support.work_shape
        return work_shape

    @property
    def interaction_coefficients(self) -> Complex128[Array, " p"]:
        """Return the frozen L3 interaction coefficients."""
        interaction_coefficients: Complex128[Array, " p"] = (
            self.compression.interaction_coefficients
        )
        return interaction_coefficients

    @property
    def absorber_coefficients(self) -> Complex128[Array, " a"]:
        """Return the frozen L4 dimensionless absorber coefficients."""
        absorber = self.cap_floor_proof.coefficient_certificate.absorber
        absorber_coefficients: Complex128[Array, " a"] = (
            absorber.absorber_coefficients
        )
        return absorber_coefficients

    @property
    def accelerating_voltage_kv(self) -> Float64[Array, ""]:
        """Return exact stored accelerating voltage in kilovolts."""
        accelerating_voltage_kv: Float64[Array, ""] = (
            self.compression.accelerating_voltage_kv
        )
        return accelerating_voltage_kv

    @property
    def interaction_coupling(self) -> Float64[Array, ""]:
        """Return the canonical frozen SC.4 coupling."""
        interaction_coupling: Float64[Array, ""] = (
            self.compression.interaction_coupling
        )
        return interaction_coupling

    @property
    def carrier(self) -> Float64[Array, " 3"]:
        """Return the algebraic carrier direction seed."""
        carrier: Float64[Array, " 3"] = self.acquisition.carrier
        return carrier

    @property
    def wavenumber(self) -> Float64[Array, ""]:
        """Return the algebraic acquisition wavenumber."""
        wavenumber: Float64[Array, ""] = self.acquisition.wavenumber
        return wavenumber

    @property
    def box_lengths(self) -> Float64[Array, " 3"]:
        """Return exact target box lengths."""
        box_lengths: Float64[Array, " 3"] = self.acquisition.box_lengths
        return box_lengths

    @property
    def free_diagonal(self) -> Float64[Array, " n"]:
        """Return the sole frozen free diagonal certified by this target."""
        free_diagonal: Float64[Array, " n"] = (
            self.fixed_linear_error_ledger.algebraic_free_diagonal
        )
        return free_diagonal

    @property
    def exact_cap_scale(self) -> Float64[Array, ""]:
        """Return exact LVT physical CAP scale."""
        absorber = self.cap_floor_proof.coefficient_certificate.absorber
        exact_cap_scale: Float64[Array, ""] = absorber.exact_cap_scale
        return exact_cap_scale

    @property
    def algebraic_cap_scale(self) -> Float64[Array, ""]:
        """Return frozen LVT physical CAP scale."""
        absorber = self.cap_floor_proof.coefficient_certificate.absorber
        algebraic_cap_scale: Float64[Array, ""] = absorber.algebraic_cap_scale
        return algebraic_cap_scale

    @property
    def preterminal_indices(self) -> Array:
        """Return the checked preterminal support."""
        preterminal_indices: Array = self.acquisition.preterminal_indices
        return preterminal_indices

    @property
    def incident_full_offset_max(self) -> Float64[Array, ""]:
        """Return exact-target incident full-offset evidence."""
        incident_full_offset_max: Float64[Array, ""] = (
            self.exact_target_incident_full_offset_max
        )
        return incident_full_offset_max

    @property
    def outgoing_full_offset_max(self) -> Float64[Array, ""]:
        """Return exact-target outgoing full-offset evidence."""
        outgoing_full_offset_max: Float64[Array, ""] = (
            self.exact_target_outgoing_full_offset_max
        )
        return outgoing_full_offset_max


@jaxtyped(typechecker=beartype)
def _make_local_cell_fixed_linear_error_ledger(  # noqa: PLR0913
    algebraic_free_diagonal: Float[Array, "..."],
    exact_wavenumber_lower_bound: Float[Array, ""],
    exact_wavenumber_upper_bound: Float[Array, ""],
    wavenumber_error_bound: Float[Array, ""],
    exact_carrier_lower_bounds: Float[Array, "..."],
    exact_carrier_upper_bounds: Float[Array, "..."],
    carrier_component_error_bounds: Float[Array, "..."],
    exact_free_diagonal_lower_bounds: Float[Array, "..."],
    exact_free_diagonal_upper_bounds: Float[Array, "..."],
    free_diagonal_error_bounds: Float[Array, "..."],
    free_operator_error_bound: Float[Array, ""],
    interaction_operator_error_bound: Float[Array, ""],
    absorber_operator_error_bound: Float[Array, ""],
    cap_scale_error_bound: Float[Array, ""],
    cap_operator_error_bound: Float[Array, ""],
    fixed_linear_operator_error_bound: Float[Array, ""],
    finite_certificate: Bool[Array, ""],
    *,
    exact_geometry_target: str,
    algebraic_geometry_realization: str,
    interaction_error_provenance: str,
    absorber_error_provenance: str,
    error_scope: str,
    coefficient_norm: str,
    free_geometry_digest: str,
    parent_operator_digest: str,
    ledger_digest: str,
) -> GalerkinLocalCellFixedLinearErrorLedger:
    """PRIVATE: Validate and store one local-cell fixed-linear ledger.

    Parameters
    ----------
    algebraic_free_diagonal : Float[Array, '...']
        Frozen shifted-free diagonal.
    exact_wavenumber_lower_bound : Float[Array, '']
        Lower endpoint of the exact wavenumber enclosure.
    exact_wavenumber_upper_bound : Float[Array, '']
        Upper endpoint of the exact wavenumber enclosure.
    wavenumber_error_bound : Float[Array, '']
        Stored-to-exact wavenumber error bound.
    exact_carrier_lower_bounds : Float[Array, '...']
        Componentwise lower endpoints of the exact carrier enclosure.
    exact_carrier_upper_bounds : Float[Array, '...']
        Componentwise upper endpoints of the exact carrier enclosure.
    carrier_component_error_bounds : Float[Array, '...']
        Stored-to-exact carrier component error bounds.
    exact_free_diagonal_lower_bounds : Float[Array, '...']
        Componentwise lower endpoints of the exact free diagonal.
    exact_free_diagonal_upper_bounds : Float[Array, '...']
        Componentwise upper endpoints of the exact free diagonal.
    free_diagonal_error_bounds : Float[Array, '...']
        Stored-to-exact free-diagonal error bounds.
    free_operator_error_bound : Float[Array, '']
        Fixed free-operator error bound.
    interaction_operator_error_bound : Float[Array, '']
        Fixed interaction-operator error bound.
    absorber_operator_error_bound : Float[Array, '']
        Dimensionless absorber-operator audit bound.
    cap_scale_error_bound : Float[Array, '']
        Physical CAP-scale audit bound.
    cap_operator_error_bound : Float[Array, '']
        Fixed physical CAP-operator error bound.
    fixed_linear_operator_error_bound : Float[Array, '']
        Outward fixed-linear total.
    finite_certificate : Bool[Array, '']
        Whether every charged fixed-linear component is finite.
    exact_geometry_target : str
        Exact free-geometry semantics.
    algebraic_geometry_realization : str
        Frozen algebraic free-geometry semantics.
    interaction_error_provenance : str
        Provenance of the copied interaction error.
    absorber_error_provenance : str
        Provenance of the copied physical CAP error.
    error_scope : str
        Explicit inclusion and exclusion scope.
    coefficient_norm : str
        Coefficient norm used by the operator bounds.
    free_geometry_digest : str
        Route-neutral free-geometry digest.
    parent_operator_digest : str
        Nested L4 operator digest.
    ledger_digest : str
        Digest of every ledger semantic and evidence leaf.

    Returns
    -------
    ledger : GalerkinLocalCellFixedLinearErrorLedger
        Validated fixed-linear ledger.
    """
    diagonal = jnp.asarray(algebraic_free_diagonal, dtype=jnp.float64)
    k_lower = jnp.asarray(exact_wavenumber_lower_bound, dtype=jnp.float64)
    k_upper = jnp.asarray(exact_wavenumber_upper_bound, dtype=jnp.float64)
    k_error = jnp.asarray(wavenumber_error_bound, dtype=jnp.float64)
    carrier_lower = jnp.asarray(exact_carrier_lower_bounds, dtype=jnp.float64)
    carrier_upper = jnp.asarray(exact_carrier_upper_bounds, dtype=jnp.float64)
    carrier_errors = jnp.asarray(
        carrier_component_error_bounds, dtype=jnp.float64
    )
    free_lower = jnp.asarray(
        exact_free_diagonal_lower_bounds, dtype=jnp.float64
    )
    free_upper = jnp.asarray(
        exact_free_diagonal_upper_bounds, dtype=jnp.float64
    )
    free_errors = jnp.asarray(free_diagonal_error_bounds, dtype=jnp.float64)
    scalars = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            free_operator_error_bound,
            interaction_operator_error_bound,
            absorber_operator_error_bound,
            cap_scale_error_bound,
            cap_operator_error_bound,
            fixed_linear_operator_error_bound,
        )
    )
    finite = jnp.asarray(finite_certificate, dtype=jnp.bool_)
    _raise_if(
        diagonal.ndim != 1 or diagonal.shape[0] == 0,
        "algebraic_free_diagonal must be nonempty 1D",
    )
    _raise_if(
        free_lower.shape != diagonal.shape
        or free_upper.shape != diagonal.shape
        or free_errors.shape != diagonal.shape,
        "free-diagonal evidence must match the state support",
    )
    _raise_if(
        carrier_lower.shape != (3,)
        or carrier_upper.shape != (3,)
        or carrier_errors.shape != (3,),
        "carrier evidence must have shape (3,)",
    )
    _raise_if(
        any(
            value.shape != ()
            for value in (k_lower, k_upper, k_error, finite, *scalars)
        ),
        "fixed-linear scalar evidence must be scalar",
    )
    for text, name in (
        (exact_geometry_target, "exact_geometry_target"),
        (algebraic_geometry_realization, "algebraic_geometry_realization"),
        (interaction_error_provenance, "interaction_error_provenance"),
        (absorber_error_provenance, "absorber_error_provenance"),
        (error_scope, "error_scope"),
        (coefficient_norm, "coefficient_norm"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (free_geometry_digest, "free_geometry_digest"),
        (parent_operator_digest, "parent_operator_digest"),
        (ledger_digest, "ledger_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    delta_d, delta_r, delta_a, delta_eps, delta_b, delta_h = scalars
    invalid = (
        jnp.any(~jnp.isfinite(diagonal))
        | jnp.any(~jnp.isfinite(carrier_lower))
        | jnp.any(~jnp.isfinite(carrier_upper))
        | jnp.any(carrier_lower > carrier_upper)
        | jnp.any(~jnp.isfinite(carrier_errors))
        | jnp.any(carrier_errors < 0.0)
        | jnp.any(~jnp.isfinite(free_lower))
        | jnp.any(~jnp.isfinite(free_upper))
        | jnp.any(free_lower > free_upper)
        | jnp.any(~jnp.isfinite(free_errors))
        | jnp.any(free_errors < 0.0)
        | (~jnp.isfinite(k_lower))
        | (~jnp.isfinite(k_upper))
        | (k_lower <= 0.0)
        | (k_lower > k_upper)
        | (~jnp.isfinite(k_error))
        | (k_error < 0.0)
        | jnp.any(jnp.asarray(scalars) < 0.0)
        | jnp.any(jnp.isnan(jnp.asarray(scalars)))
        | (
            finite
            != jnp.all(
                jnp.isfinite(jnp.asarray((delta_d, delta_r, delta_b, delta_h)))
            )
        )
        | (delta_d != jnp.max(free_errors))
        | (delta_h != upward_add(upward_add(delta_d, delta_r), delta_b))
    )
    checked_diagonal = eqx.error_if(
        diagonal,
        invalid,
        "local-cell fixed-linear ledger contains invalid evidence",
    )
    ledger: GalerkinLocalCellFixedLinearErrorLedger = (
        GalerkinLocalCellFixedLinearErrorLedger(
            algebraic_free_diagonal=checked_diagonal,
            exact_wavenumber_lower_bound=k_lower,
            exact_wavenumber_upper_bound=k_upper,
            wavenumber_error_bound=k_error,
            exact_carrier_lower_bounds=carrier_lower,
            exact_carrier_upper_bounds=carrier_upper,
            carrier_component_error_bounds=carrier_errors,
            exact_free_diagonal_lower_bounds=free_lower,
            exact_free_diagonal_upper_bounds=free_upper,
            free_diagonal_error_bounds=free_errors,
            free_operator_error_bound=delta_d,
            interaction_operator_error_bound=delta_r,
            absorber_operator_error_bound=delta_a,
            cap_scale_error_bound=delta_eps,
            cap_operator_error_bound=delta_b,
            fixed_linear_operator_error_bound=delta_h,
            finite_certificate=finite,
            exact_geometry_target=exact_geometry_target.strip(),
            algebraic_geometry_realization=(
                algebraic_geometry_realization.strip()
            ),
            interaction_error_provenance=(
                interaction_error_provenance.strip()
            ),
            absorber_error_provenance=absorber_error_provenance.strip(),
            error_scope=error_scope.strip(),
            coefficient_norm=coefficient_norm.strip(),
            free_geometry_digest=free_geometry_digest,
            parent_operator_digest=parent_operator_digest,
            ledger_digest=ledger_digest,
        )
    )
    return ledger


@jaxtyped(typechecker=beartype)
def _make_local_cell_target_manifest(  # noqa: PLR0913
    cap_floor_proof: GalerkinAxialCapFloorProof,
    fixed_linear_error_ledger: GalerkinLocalCellFixedLinearErrorLedger,
    exact_target_incident_full_offset_max: Float[Array, ""],
    exact_target_outgoing_full_offset_max: Float[Array, ""],
    exact_target_incident_shell_defect_bounds: Float[Array, "..."],
    exact_target_outgoing_shell_defect_bounds: Float[Array, "..."],
    exact_target_incident_projection_error_bounds: Float[Array, "..."],
    exact_target_outgoing_projection_error_bounds: Float[Array, "..."],
    *,
    target_route: GalerkinVoxelTargetRoute,
    sc1_contract_version: str,
    lvt1_contract_version: str,
    target_formula: str,
    action_formula: str,
    adjoint_formula: str,
    target_digest: str,
    manifest_evidence_digest: str,
    target_name: str,
) -> GalerkinLocalCellTargetManifest:
    """PRIVATE: Validate and store one completed local-cell target.

    Parameters
    ----------
    cap_floor_proof : GalerkinAxialCapFloorProof
        Fully replayed L4 CAP proof.
    fixed_linear_error_ledger : GalerkinLocalCellFixedLinearErrorLedger
        Disjoint fixed-linear error ledger.
    exact_target_incident_full_offset_max : Float[Array, '']
        Exact-carrier incident full-offset maximum.
    exact_target_outgoing_full_offset_max : Float[Array, '']
        Exact-carrier outgoing full-offset maximum.
    exact_target_incident_shell_defect_bounds : Float[Array, '...']
        Exact-carrier incident shell-defect bounds.
    exact_target_outgoing_shell_defect_bounds : Float[Array, '...']
        Exact-carrier outgoing shell-defect bounds.
    exact_target_incident_projection_error_bounds : Float[Array, '...']
        Exact-carrier incident projection-error bounds.
    exact_target_outgoing_projection_error_bounds : Float[Array, '...']
        Exact-carrier outgoing projection-error bounds.
    target_route : GalerkinVoxelTargetRoute
        Static solver-target route.
    sc1_contract_version : str
        Scalar-contract version.
    lvt1_contract_version : str
        Local-voxel-target contract version.
    target_formula : str
        Exact target formula.
    action_formula : str
        Frozen forward-action formula.
    adjoint_formula : str
        Frozen formal-adjoint formula.
    target_digest : str
        Operator-only identity digest.
    manifest_evidence_digest : str
        Full proof and acquisition-evidence digest.
    target_name : str
        Canonically stripped target name.

    Returns
    -------
    manifest : GalerkinLocalCellTargetManifest
        Validated solver-ready local-cell target.
    """
    _raise_if(
        target_route is not GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1,
        "target_route must be LOCAL_CELL_LVT1",
    )
    _raise_if(not target_name.strip(), "target_name must be nonempty")
    for text, name in (
        (sc1_contract_version, "sc1_contract_version"),
        (lvt1_contract_version, "lvt1_contract_version"),
        (target_formula, "target_formula"),
        (action_formula, "action_formula"),
        (adjoint_formula, "adjoint_formula"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (target_digest, "target_digest"),
        (manifest_evidence_digest, "manifest_evidence_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    certificate = cap_floor_proof.coefficient_certificate
    core = certificate.absorber.interaction_core
    acquisition = core.compression.realization.support_eligibility.manifest
    incident_shape = acquisition.incident_indices.shape[:1]
    outgoing_shape = acquisition.elastic_outgoing_indices.shape[:1]
    incident_full = jnp.asarray(
        exact_target_incident_full_offset_max, dtype=jnp.float64
    )
    outgoing_full = jnp.asarray(
        exact_target_outgoing_full_offset_max, dtype=jnp.float64
    )
    incident_shell = jnp.asarray(
        exact_target_incident_shell_defect_bounds, dtype=jnp.float64
    )
    outgoing_shell = jnp.asarray(
        exact_target_outgoing_shell_defect_bounds, dtype=jnp.float64
    )
    incident_projection = jnp.asarray(
        exact_target_incident_projection_error_bounds, dtype=jnp.float64
    )
    outgoing_projection = jnp.asarray(
        exact_target_outgoing_projection_error_bounds, dtype=jnp.float64
    )
    _raise_if(
        incident_full.shape != () or outgoing_full.shape != (),
        "full-offset maxima must be scalars",
    )
    _raise_if(
        incident_shell.shape != incident_shape
        or incident_projection.shape != incident_shape,
        "incident exact-target evidence must match incident rows",
    )
    _raise_if(
        outgoing_shell.shape != outgoing_shape
        or outgoing_projection.shape != outgoing_shape,
        "outgoing exact-target evidence must match outgoing rows",
    )
    state_count = core.support.state_indices.shape[0]
    _raise_if(
        fixed_linear_error_ledger.algebraic_free_diagonal.shape
        != (state_count,),
        "free diagonal must match I_u",
    )
    invalid = jnp.any(
        ~jnp.isfinite(
            jnp.concatenate(
                (
                    jnp.atleast_1d(incident_full),
                    jnp.atleast_1d(outgoing_full),
                    incident_shell,
                    outgoing_shell,
                    incident_projection,
                    outgoing_projection,
                )
            )
        )
    ) | jnp.any(
        jnp.concatenate(
            (
                jnp.atleast_1d(incident_full),
                jnp.atleast_1d(outgoing_full),
                incident_shell,
                outgoing_shell,
                incident_projection,
                outgoing_projection,
            )
        )
        < 0.0
    )
    checked_incident_full = eqx.error_if(
        incident_full,
        invalid,
        "exact-target acquisition evidence must be finite and non-negative",
    )
    manifest: GalerkinLocalCellTargetManifest = (
        GalerkinLocalCellTargetManifest(
            cap_floor_proof=cap_floor_proof,
            fixed_linear_error_ledger=fixed_linear_error_ledger,
            exact_target_incident_full_offset_max=checked_incident_full,
            exact_target_outgoing_full_offset_max=outgoing_full,
            exact_target_incident_shell_defect_bounds=incident_shell,
            exact_target_outgoing_shell_defect_bounds=outgoing_shell,
            exact_target_incident_projection_error_bounds=incident_projection,
            exact_target_outgoing_projection_error_bounds=outgoing_projection,
            target_route=target_route,
            sc1_contract_version=sc1_contract_version.strip(),
            lvt1_contract_version=lvt1_contract_version.strip(),
            target_formula=target_formula.strip(),
            action_formula=action_formula.strip(),
            adjoint_formula=adjoint_formula.strip(),
            target_digest=target_digest,
            manifest_evidence_digest=manifest_evidence_digest,
            target_name=target_name.strip(),
        )
    )
    return manifest


__all__: list[str] = [
    "GalerkinLocalCellFixedLinearErrorLedger",
    "GalerkinLocalCellTargetManifest",
]
