r"""Apply the manifested SC-1 target and construct its finite source.

Extended Summary
----------------
This module composes endpoint-safe coefficient products into the production
SC-1 target and actual adjoint. It also constructs a rounded finite realization
of the RM-S3 matched-source formula. Physical residuals use a separate
bounded-memory coefficient
lookup instead of the production FFT action or a Krylov recurrence.

Routine Listings
----------------
:func:`apply_galerkin_target`
    Apply the manifested matrix-free SC-1 target.
:func:`apply_galerkin_target_adjoint`
    Apply the actual adjoint of the manifested SC-1 target.
:func:`create_galerkin_target`
    Build one production SC-1 target from a shared voxel potential.
:func:`create_host_checked_galerkin_target`
    Build a target after a concrete-host coefficient-certificate attempt.
:func:`create_matched_galerkin_source`
    Construct a rounded finite matched-source realization.
:func:`evaluate_physical_galerkin_adjoint_residual`
    Recompute an adjoint-system residual by direct coefficient lookup.
:func:`evaluate_physical_galerkin_residual`
    Recompute a forward-system residual by direct coefficient lookup.

Notes
-----
The differentiable target constructor retains a conservative triangle
coefficient bound.  The separate concrete-host constructor refines that same
VC-1 realization before RM-S2 construction and preserves typed certificate
failure as infinite evidence.
"""

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
    Float64,
    Int32,
    Int64,
    jaxtyped,
)

from ptyrodactyl._numeric import (
    has_lost_subtraction,
    has_nonzero_components,
    has_subnormal_components,
)
from ptyrodactyl._physics import coupled_interaction_value
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    M_E,
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportResult,
    GalerkinFixedLinearErrorLedger,
    GalerkinPhysicalResidual,
    GalerkinPotentialRealization,
    GalerkinSource,
    GalerkinSourceBranch,
    GalerkinTargetManifest,
    Potential3D,
    _create_galerkin_target_manifest,
    create_galerkin_physical_residual,
    create_galerkin_source,
    scalar_float,
    scalar_int,
)

from .coefficient_certification import certify_galerkin_potential_realization
from .enclosures import build_galerkin_fixed_linear_error_ledger
from .potential import apply_absorber_action, apply_interaction_product
from .realization import realize_galerkin_potential


def _create_target_from_realization(
    realization: GalerkinPotentialRealization,
    voltage: Float64[Array, ""],
    cap: Float64[Array, ""],
    target_name: str,
) -> GalerkinTargetManifest:
    """PRIVATE: Bind one checked VC-1 realization into its RM-S2 target.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Realized voxel target with either triangle or direct host evidence.
    voltage : Float64[Array, ""]
        Accelerating voltage in kilovolts.
    cap : Float64[Array, ""]
        CAP scale in inverse-square Angstroms.
    target_name : str
        Canonical finite-target name.

    Returns
    -------
    target : GalerkinTargetManifest
        Nested target whose RM-S2 ledger consumes the realization's exact
        coefficient-error route.
    """
    acquisition: GalerkinAcquisitionManifest = (
        realization.support_eligibility.manifest
    )
    potential_box: Float64[Array, " 3"] = jnp.asarray(
        realization.potential.box_size,
        dtype=jnp.float64,
    )
    checked_voltage: Float64[Array, ""] = eqx.error_if(
        voltage,
        jnp.any(acquisition.box_lengths != potential_box),
        "Potential3D box lengths must exactly match acquisition support",
    )
    interaction_coupling: Float64[Array, ""]
    interaction_coefficients: Complex128[Array, " p"]
    interaction_coupling, interaction_coefficients = coupled_interaction_value(
        realization.voltage_coefficients,
        checked_voltage,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    fixed_linear_error_ledger: GalerkinFixedLinearErrorLedger = (
        build_galerkin_fixed_linear_error_ledger(
            state_indices=realization.support.state_indices,
            interaction_indices=realization.support.interaction_indices,
            voltage_coefficients=realization.voltage_coefficients,
            voltage_coefficient_error_bounds=(
                realization.coefficient_error_bounds
            ),
            interaction_coupling=interaction_coupling,
            interaction_coefficients=interaction_coefficients,
            accelerating_voltage_kv=checked_voltage,
            carrier=acquisition.carrier,
            box_lengths=acquisition.box_lengths,
            wavenumber=acquisition.wavenumber,
            cap_scale=cap,
        )
    )
    target: GalerkinTargetManifest = _create_galerkin_target_manifest(
        realization=realization,
        fixed_linear_error_ledger=fixed_linear_error_ledger,
        accelerating_voltage_kv=checked_voltage,
        cap_scale=cap,
        target_name=target_name,
    )
    return target


@jaxtyped(typechecker=beartype)
def create_galerkin_target(
    potential: Potential3D,
    support_eligibility: GalerkinAcquisitionSupportResult,
    *,
    accelerating_voltage_kv: scalar_float,
    cap_scale: scalar_float,
    target_name: str,
) -> GalerkinTargetManifest:
    """Build one production SC-1 target from a shared voxel potential.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    potential : Potential3D
        Periodic scalar voltage samples and complete VC-1 metadata.
    support_eligibility : GalerkinAcquisitionSupportResult
        Complete checker output. The result is independently recomputed and
        must exactly match in every manifest, evidence, and status field.
    accelerating_voltage_kv : scalar_float
        Positive accelerating voltage in kilovolts.
    cap_scale : scalar_float
        Positive physical CAP scale in inverse-square Angstroms.
    target_name : str
        Nonempty canonical name for this finite target.

    Returns
    -------
    target : GalerkinTargetManifest
        Nested Potential3D-to-RM-S2 target with no raw coefficient seam.

    Raises
    ------
    ValueError
        If a scalar shape, static name, or exact box binding is invalid.
    equinox.EquinoxRuntimeError
        If eligibility is forged/ineligible, nominal wavenumber differs from
        the canonical voltage route, or any realization predicate fails.

    Notes
    -----
    This is the transformation-compatible orchestration route. It rechecks
    the support, performs VC-1 realization with the conservative triangle
    evidence, derives the canonical interaction, constructs RM-S2, and only
    then invokes the owning type factory. Use
    :func:`create_host_checked_galerkin_target` when a useful direct
    coefficient certificate is required before RM-S2 construction.
    """
    voltage: Float64[Array, ""] = jnp.asarray(
        accelerating_voltage_kv,
        dtype=jnp.float64,
    )
    cap: Float64[Array, ""] = jnp.asarray(cap_scale, dtype=jnp.float64)
    if voltage.shape != ():
        raise ValueError("accelerating_voltage_kv must be a scalar")
    if cap.shape != ():
        raise ValueError("cap_scale must be a scalar")

    realization: GalerkinPotentialRealization = realize_galerkin_potential(
        potential,
        support_eligibility,
    )
    target: GalerkinTargetManifest = _create_target_from_realization(
        realization,
        voltage,
        cap,
        target_name,
    )
    return target


@jaxtyped(typechecker=beartype)
def create_host_checked_galerkin_target(
    potential: Potential3D,
    support_eligibility: GalerkinAcquisitionSupportResult,
    *,
    accelerating_voltage_kv: scalar_float,
    cap_scale: scalar_float,
    target_name: str,
    maximum_direct_terms: int = 2_000_000,
) -> GalerkinTargetManifest:
    """Build a target after a concrete-host coefficient-certificate attempt.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    potential : Potential3D
        Periodic scalar voltage samples and complete VC-1 metadata.
    support_eligibility : GalerkinAcquisitionSupportResult
        Independently rechecked finite acquisition-support result.
    accelerating_voltage_kv : scalar_float
        Positive accelerating voltage in kilovolts.
    cap_scale : scalar_float
        Positive physical CAP scale in inverse-square Angstroms.
    target_name : str
        Nonempty canonical name for this finite target.
    maximum_direct_terms : int, optional
        Host direct-certificate work budget. Default: ``2_000_000``.

    Returns
    -------
    target : GalerkinTargetManifest
        Target containing the direct certificate attempt.  A typed failed
        attempt is preserved as infinite coefficient and RM-S2 evidence.

    Raises
    ------
    ValueError
        If scalar shapes, the work budget, static metadata, or exact box
        binding are invalid, or if any input is traced.
    equinox.EquinoxRuntimeError
        If support eligibility or another realization predicate fails.

    Notes
    -----
    This constructor is deliberately not JIT-compatible.  It computes the
    ordinary differentiable realization first, refines only its stopped error
    evidence on the concrete host, and then constructs RM-S2.  The production
    coefficient leaves and the exact VC-1 target are unchanged.
    """
    voltage: Float64[Array, ""] = jnp.asarray(
        accelerating_voltage_kv,
        dtype=jnp.float64,
    )
    cap: Float64[Array, ""] = jnp.asarray(cap_scale, dtype=jnp.float64)
    if voltage.shape != ():
        raise ValueError("accelerating_voltage_kv must be a scalar")
    if cap.shape != ():
        raise ValueError("cap_scale must be a scalar")

    realization: GalerkinPotentialRealization = realize_galerkin_potential(
        potential,
        support_eligibility,
    )
    jax.block_until_ready(realization)
    checked_realization: GalerkinPotentialRealization = (
        certify_galerkin_potential_realization(
            realization,
            maximum_direct_terms=maximum_direct_terms,
        )
    )
    target: GalerkinTargetManifest = _create_target_from_realization(
        checked_realization,
        voltage,
        cap,
        target_name,
    )
    return target


def _checked_field(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one retained-state coefficient vector.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Target whose state support defines the required vector length.
    field : Complex[Array, "..."]
        Candidate retained-state coefficients.
    name : str
        Vector name used in rejection messages.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Canonical binary64-complex vector with traced value checks attached.

    Raises
    ------
    ValueError
        If the candidate is not one-dimensional or has the wrong length.
    equinox.EquinoxRuntimeError
        If the vector contains a non-finite or nonzero subnormal component.

    Notes
    -----
    Input values are converted to binary64-complex before value validation.
    """
    field_array: Complex128[Array, " n"] = jnp.asarray(
        field, dtype=jnp.complex128
    )
    if field_array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if field_array.shape[0] != manifest.support.state_indices.shape[0]:
        raise ValueError(f"{name} must match the state support")
    checked: Complex128[Array, " n"] = eqx.error_if(
        field_array,
        jnp.any(~jnp.isfinite(field_array))
        | has_subnormal_components(field_array),
        f"{name} must be finite and contain no nonzero subnormal components",
    )
    return checked


def _direct_multiplier_action(
    state_indices: Int64[Array, "n 3"],
    multiplier_indices: Int64[Array, "p 3"],
    coefficients: Complex128[Array, " p"],
    field: Complex128[Array, " n"],
    work_shape: Tuple[int, int, int],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply direct coefficient lookup with bounded working memory.

    Parameters
    ----------
    state_indices : Int64[Array, "n 3"]
        Retained reciprocal indices in ``(x, y, z)`` order.
    multiplier_indices : Int64[Array, "p 3"]
        Multiplier reciprocal indices in ``(x, y, z)`` order.
    coefficients : Complex128[Array, " p"]
        Multiplier coefficients in fixed support order.
    field : Complex128[Array, " n"]
        Retained-state coefficient vector.
    work_shape : Tuple[int, int, int]
        Endpoint-safe periodic quotient shape in ``(x, y, z)`` order.

    Returns
    -------
    applied : Complex128[Array, " n"]
        Direct compressed multiplier action in state order.

    Notes
    -----
    The loop stores only the output vector and sorted multiplier keys. It does
    not manifest the dense ``n`` by ``n`` matrix.
    """
    moduli: Int64[Array, " 3"] = jnp.asarray(work_shape, dtype=jnp.int64)
    multiplier_residues: Int64[Array, "p 3"] = jnp.mod(
        multiplier_indices, moduli
    )
    multiplier_keys: Int64[Array, " p"] = (
        multiplier_residues[:, 0] * work_shape[1] + multiplier_residues[:, 1]
    ) * work_shape[2] + multiplier_residues[:, 2]
    order: Int64[Array, " p"] = jnp.argsort(multiplier_keys)
    sorted_keys: Int64[Array, " p"] = multiplier_keys[order]
    state_size: int = state_indices.shape[0]
    product_count: int = state_size * state_size

    def add_entry(
        flat_position: scalar_int,
        accumulator: Complex128[Array, " n"],
    ) -> Complex128[Array, " n"]:
        """Accumulate one direct matrix entry without storing the matrix."""
        row: scalar_int = flat_position // state_size
        column: scalar_int = flat_position % state_size
        difference: Int64[Array, " 3"] = (
            state_indices[row] - state_indices[column]
        )
        residue: Int64[Array, " 3"] = jnp.mod(difference, moduli)
        key: Int64[Array, ""] = (
            residue[0] * work_shape[1] + residue[1]
        ) * work_shape[2] + residue[2]
        location: Int32[Array, ""] = jnp.searchsorted(
            sorted_keys, key, side="left"
        )
        clipped: Int32[Array, ""] = jnp.clip(
            location, 0, multiplier_indices.shape[0] - 1
        )
        exact_match: Bool[Array, ""] = (
            (location < multiplier_indices.shape[0])
            & (sorted_keys[clipped] == key)
            & jnp.all(multiplier_indices[order[clipped]] == difference)
        )
        coefficient: Complex128[Array, ""] = jnp.where(
            exact_match,
            coefficients[order[clipped]],
            jnp.asarray(0.0 + 0.0j, dtype=coefficients.dtype),
        )
        updated: Complex128[Array, " n"] = accumulator.at[row].add(
            coefficient * field[column]
        )
        return updated

    initial: Complex128[Array, " n"] = jnp.zeros(
        (state_size,), dtype=jnp.result_type(coefficients, field)
    )
    applied: Complex128[Array, " n"] = jax.lax.fori_loop(
        0,
        product_count,
        add_entry,
        initial,
    )
    return applied


def _direct_target_action(
    manifest: GalerkinTargetManifest,
    field: Complex128[Array, " n"],
    *,
    adjoint: bool,
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the manifested coefficient matrix by a direct path.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target and independent supports.
    field : Complex128[Array, " n"]
        Retained-state coefficient vector.
    adjoint : bool
        If true, apply the actual adjoint by reversing the absorber sign.

    Returns
    -------
    applied : Complex128[Array, " n"]
        Direct forward or adjoint target action.

    Notes
    -----
    Hermitian interaction and absorber multipliers leave only the imaginary
    absorber sign to distinguish the forward and adjoint targets.
    """
    interaction: Complex128[Array, " n"] = _direct_multiplier_action(
        manifest.support.state_indices,
        manifest.support.interaction_indices,
        manifest.interaction_coefficients,
        field,
        manifest.support.work_shape,
    )
    absorber: Complex128[Array, " n"] = _direct_multiplier_action(
        manifest.support.state_indices,
        manifest.support.absorber_indices,
        manifest.absorber_coefficients,
        field,
        manifest.support.work_shape,
    )
    cap_sign: complex = 1j if adjoint else -1j
    applied: Complex128[Array, " n"] = (
        manifest.free_diagonal * field
        - interaction
        + cap_sign * manifest.cap_scale * absorber
    )
    return applied


def _complex_norm(vector: Complex128[Array, " n"]) -> Float64[Array, ""]:
    """PRIVATE: Compute a scale-safe Euclidean complex-vector norm.

    Parameters
    ----------
    vector : Complex128[Array, " n"]
        Complex vector whose coefficient norm is required.

    Returns
    -------
    norm : Float64[Array, ""]
        Binary64 Euclidean norm of ``vector``.

    Notes
    -----
    Scaling by the largest magnitude avoids intermediate square overflow and
    underflow. The exact zero vector remains zero.
    """
    magnitudes: Float64[Array, " n"] = jnp.abs(vector)
    scale: Float64[Array, ""] = jnp.max(magnitudes)
    safe_scale: Float64[Array, ""] = jnp.where(scale > 0.0, scale, 1.0)
    scaled_norm: Float64[Array, ""] = jnp.sqrt(
        jnp.sum((magnitudes / safe_scale) ** 2)
    )
    finite_norm: Float64[Array, ""] = scale * scaled_norm
    norm: Float64[Array, ""] = jnp.where(
        scale == 0.0,
        0.0,
        jnp.where(jnp.isinf(scale), scale, finite_norm),
    )
    return norm


def _checked_target_action(
    field: Complex128[Array, " n"],
    applied: Complex128[Array, " n"],
    name: str,
) -> Complex128[Array, " n"]:
    """PRIVATE: Reject invalid or globally lost injective target actions.

    Parameters
    ----------
    field : Complex128[Array, " n"]
        Input retained-state coefficients.
    applied : Complex128[Array, " n"]
        Candidate target action.
    name : str
        Action name used in the traced rejection message.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Action with traced finite, normal-range, and retention checks.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the action is non-finite, subnormal, or erases a nonzero input.

    Notes
    -----
    The global retention check is a numerical fail-closed guard, not a proof
    of target injectivity.
    """
    lost_nonzero_action: Bool[Array, ""] = has_nonzero_components(field) & (
        ~has_nonzero_components(applied)
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        applied,
        jnp.any(~jnp.isfinite(applied))
        | has_subnormal_components(applied)
        | lost_nonzero_action,
        f"{name} must be finite, normal-range, and retain a nonzero input",
    )
    return checked


@jaxtyped(typechecker=beartype)
def apply_galerkin_target(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    r"""Apply the manifested matrix-free SC-1 target.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target and independent Fourier supports.
    field : Complex[Array, "..."]
        Retained-state coefficient vector.

    Returns
    -------
    applied_field : Complex128[Array, " n"]
        Canonical binary64-complex action
        ``D u - R u - i epsilon_CAP A u``.

    Raises
    ------
    ValueError
        If the field rank or size is invalid.
    equinox.EquinoxRuntimeError
        If the field contains a non-finite value.
    """
    checked_field: Complex128[Array, " n"] = _checked_field(
        manifest, field, "field"
    )
    interaction: Complex128[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        checked_field,
    )
    absorber: Complex128[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        checked_field,
    )
    raw_applied_field: Complex128[Array, " n"] = (
        manifest.free_diagonal * checked_field
        - interaction
        - 1j * manifest.cap_scale * absorber
    )
    applied_field: Complex128[Array, " n"] = _checked_target_action(
        checked_field,
        raw_applied_field,
        "target action",
    )
    return applied_field


@jaxtyped(typechecker=beartype)
def apply_galerkin_target_adjoint(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    r"""Apply the actual adjoint of the manifested SC-1 target.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target with Hermitian interaction and absorber data.
    field : Complex[Array, "..."]
        Retained-state adjoint coefficient vector.

    Returns
    -------
    applied_field : Complex128[Array, " n"]
        Canonical binary64-complex adjoint action
        ``D u - R u + i epsilon_CAP A u``.

    Raises
    ------
    ValueError
        If the field rank or size is invalid.
    equinox.EquinoxRuntimeError
        If the field contains a non-finite value.
    """
    checked_field: Complex128[Array, " n"] = _checked_field(
        manifest, field, "field"
    )
    interaction: Complex128[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        checked_field,
    )
    absorber: Complex128[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        checked_field,
    )
    raw_applied_field: Complex128[Array, " n"] = (
        manifest.free_diagonal * checked_field
        - interaction
        + 1j * manifest.cap_scale * absorber
    )
    applied_field: Complex128[Array, " n"] = _checked_target_action(
        checked_field,
        raw_applied_field,
        "adjoint target action",
    )
    return applied_field


@jaxtyped(typechecker=beartype)
def create_matched_galerkin_source(
    manifest: GalerkinTargetManifest,
    incident_field: Complex[Array, "..."],
    additional_source: Complex[Array, "..."] | None = None,
) -> GalerkinSource:
    r"""Construct a rounded finite matched-source realization.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target that supplies ``D``, ``R``, and ``B``.
    incident_field : Complex[Array, "..."]
        Declared finite incident-vector coefficients.
    additional_source : Complex[Array, "..."] | None
        Separately declared finite source. ``None`` selects zero. Default is
        ``None``.

    Returns
    -------
    source : GalerkinSource
        Matched incident, total, and equivalent scattered sources.

    Raises
    ------
    ValueError
        If an input vector rank or size is invalid.
    equinox.EquinoxRuntimeError
        If an input vector contains a non-finite value.

    Notes
    -----
    This evaluates the finite RM-S3 formula
    ``S_inc = (D - i epsilon_CAP A) v_inc`` with the production FFT action.
    The scattered source is ``R v_inc + S_add``. Binary64 construction drift
    relative to the independent direct action is diagnostic only: this carrier
    has no outward source/action-error enclosure and does not establish full
    RM-S3 implementation eligibility.
    """
    incident: Complex128[Array, " n"] = _checked_field(
        manifest, incident_field, "incident_field"
    )
    if additional_source is None:
        additional: Complex128[Array, " n"] = jnp.zeros_like(incident)
    else:
        additional = _checked_field(
            manifest, additional_source, "additional_source"
        )
    absorber: Complex128[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        incident,
    )
    raw_incident_source: Complex128[Array, " n"] = (
        manifest.free_diagonal * incident - 1j * manifest.cap_scale * absorber
    )
    incident_source: Complex128[Array, " n"] = _checked_target_action(
        incident,
        raw_incident_source,
        "matched incident action",
    )
    rounded_incident_source, rounded_additional = jax.lax.optimization_barrier(
        (incident_source, additional)
    )
    raw_total_source: Complex128[Array, " n"] = (
        rounded_incident_source + rounded_additional
    )
    total_source: Complex128[Array, " n"] = eqx.error_if(
        raw_total_source,
        has_subnormal_components(raw_total_source)
        | has_lost_subtraction(
            rounded_incident_source,
            -rounded_additional,
            raw_total_source,
        ),
        "matched total-source addition lost a nonzero component",
    )
    interaction: Complex128[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        incident,
    )
    rounded_interaction, rounded_scattered_additional = (
        jax.lax.optimization_barrier((interaction, additional))
    )
    raw_scattered_source: Complex128[Array, " n"] = (
        rounded_interaction + rounded_scattered_additional
    )
    scattered_source: Complex128[Array, " n"] = eqx.error_if(
        raw_scattered_source,
        has_subnormal_components(raw_scattered_source)
        | has_lost_subtraction(
            rounded_interaction,
            -rounded_scattered_additional,
            raw_scattered_source,
        ),
        "matched scattered-source addition lost a nonzero component",
    )
    source: GalerkinSource = create_galerkin_source(
        incident_field=incident,
        incident_source=incident_source,
        additional_source=additional,
        total_source=total_source,
        scattered_source=scattered_source,
        branch=GalerkinSourceBranch.FINITE_MATCHED,
    )
    return source


def _physical_residual(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    source: Complex[Array, "..."],
    *,
    adjoint: bool,
) -> GalerkinPhysicalResidual:
    """PRIVATE: Recompute one direct original-system residual.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical target used for independent direct coefficient lookup.
    field : Complex[Array, "..."]
        Candidate retained-state solution coefficients.
    source : Complex[Array, "..."]
        Original-system source coefficients.
    adjoint : bool
        If true, recompute the residual for the actual adjoint target.

    Returns
    -------
    physical_residual : GalerkinPhysicalResidual
        Independently recomputed residual vector and scale-safe norm.

    Raises
    ------
    ValueError
        If the field or source rank or length is invalid.
    equinox.EquinoxRuntimeError
        If validation or residual subtraction detects an invalid value.

    Notes
    -----
    This path does not reuse the matrix-free Krylov action or its recurrence.
    It therefore supplies an independent physical residual.
    """
    checked_field: Complex128[Array, " n"] = _checked_field(
        manifest, field, "field"
    )
    checked_source: Complex128[Array, " n"] = _checked_field(
        manifest, source, "source"
    )
    applied: Complex128[Array, " n"] = _direct_target_action(
        manifest, checked_field, adjoint=adjoint
    )
    checked_applied: Complex128[Array, " n"] = _checked_target_action(
        checked_field,
        applied,
        "direct target action",
    )
    rounded_source, rounded_applied = jax.lax.optimization_barrier(
        (checked_source, checked_applied)
    )
    raw_residual: Complex128[Array, " n"] = rounded_source - rounded_applied
    residual: Complex128[Array, " n"] = eqx.error_if(
        raw_residual,
        has_subnormal_components(raw_residual)
        | has_lost_subtraction(
            rounded_source,
            rounded_applied,
            raw_residual,
        ),
        "physical residual subtraction lost a nonzero component",
    )
    residual_norm: Float64[Array, ""] = _complex_norm(residual)
    physical_residual: GalerkinPhysicalResidual = (
        create_galerkin_physical_residual(residual, residual_norm)
    )
    return physical_residual


@jaxtyped(typechecker=beartype)
def evaluate_physical_galerkin_residual(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    source: Complex[Array, "..."],
) -> GalerkinPhysicalResidual:
    """Recompute a forward-system residual by direct coefficient lookup.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonically manifested SC-1 target.
    field : Complex[Array, "..."]
        Submitted retained-state coefficient vector.
    source : Complex[Array, "..."]
        Original finite-system right-hand side.

    Returns
    -------
    physical_residual : GalerkinPhysicalResidual
        Direct residual ``source - H field`` and its Euclidean norm.

    Raises
    ------
    ValueError
        If an input vector rank or size is invalid.
    equinox.EquinoxRuntimeError
        If an input vector contains a non-finite value.

    Notes
    -----
    This evaluator scans exact coefficient differences with bounded working
    memory. It does not call the production FFT action, a Krylov recurrence,
    or a normal equation. The stability checker separately encloses this same
    named target with exact binary-rational arithmetic.
    """
    physical_residual: GalerkinPhysicalResidual = _physical_residual(
        manifest, field, source, adjoint=False
    )
    return physical_residual


@jaxtyped(typechecker=beartype)
def evaluate_physical_galerkin_adjoint_residual(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    source: Complex[Array, "..."],
) -> GalerkinPhysicalResidual:
    """Recompute an adjoint-system residual by direct coefficient lookup.

    :see: :class:`~.test_system.TestScalarGalerkinSystem`

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonically manifested SC-1 target.
    field : Complex[Array, "..."]
        Submitted adjoint-state coefficient vector.
    source : Complex[Array, "..."]
        Original adjoint-system right-hand side.

    Returns
    -------
    physical_residual : GalerkinPhysicalResidual
        Direct residual ``source - H* field`` and its Euclidean norm.

    Raises
    ------
    ValueError
        If an input vector rank or size is invalid.
    equinox.EquinoxRuntimeError
        If an input vector contains a non-finite value.
    """
    physical_residual: GalerkinPhysicalResidual = _physical_residual(
        manifest, field, source, adjoint=True
    )
    return physical_residual


__all__: list[str] = [
    "apply_galerkin_target",
    "apply_galerkin_target_adjoint",
    "create_galerkin_target",
    "create_host_checked_galerkin_target",
    "create_matched_galerkin_source",
    "evaluate_physical_galerkin_adjoint_residual",
    "evaluate_physical_galerkin_residual",
]
