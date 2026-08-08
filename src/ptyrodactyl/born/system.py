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
:func:`create_matched_galerkin_source`
    Construct a rounded finite matched-source realization.
:func:`evaluate_physical_galerkin_adjoint_residual`
    Recompute an adjoint-system residual by direct coefficient lookup.
:func:`evaluate_physical_galerkin_residual`
    Recompute a forward-system residual by direct coefficient lookup.

Notes
-----
The finite source branch realizes a declared coefficient formula in binary64.
It carries no outward source/action enclosure and does not establish full
RM-S3 eligibility, analytic angular-spectrum, window, projection,
reduced-flux, or detector conformance.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from ptyrodactyl._numeric import (
    has_lost_subtraction,
    has_nonzero_components,
    has_subnormal_components,
)
from ptyrodactyl.types import (
    GalerkinPhysicalResidual,
    GalerkinSource,
    GalerkinSourceBranch,
    GalerkinTargetManifest,
    create_galerkin_physical_residual,
    create_galerkin_source,
    scalar_int,
)

from .potential import apply_absorber_action, apply_interaction_product


def _checked_field(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
    name: str,
) -> Complex[Array, " n"]:
    """Validate one retained-state coefficient vector."""
    field_array: Complex[Array, " n"] = jnp.asarray(
        field, dtype=jnp.complex128
    )
    if field_array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if field_array.shape[0] != manifest.support.state_indices.shape[0]:
        raise ValueError(f"{name} must match the state support")
    checked: Complex[Array, " n"] = eqx.error_if(
        field_array,
        jnp.any(~jnp.isfinite(field_array))
        | has_subnormal_components(field_array),
        f"{name} must be finite and contain no nonzero subnormal components",
    )
    return checked


def _direct_multiplier_action(
    state_indices: Int[Array, "n 3"],
    multiplier_indices: Int[Array, "p 3"],
    coefficients: Complex[Array, " p"],
    field: Complex[Array, " n"],
    work_shape: tuple[int, int, int],
) -> Complex[Array, " n"]:
    """Apply direct coefficient lookup with bounded working memory."""
    moduli: Int[Array, " 3"] = jnp.asarray(work_shape, dtype=jnp.int64)
    multiplier_residues: Int[Array, "p 3"] = jnp.mod(
        multiplier_indices, moduli
    )
    multiplier_keys: Int[Array, " p"] = (
        multiplier_residues[:, 0] * work_shape[1] + multiplier_residues[:, 1]
    ) * work_shape[2] + multiplier_residues[:, 2]
    order: Int[Array, " p"] = jnp.argsort(multiplier_keys)
    sorted_keys: Int[Array, " p"] = multiplier_keys[order]
    state_size: int = state_indices.shape[0]
    product_count: int = state_size * state_size

    def add_entry(
        flat_position: scalar_int,
        accumulator: Complex[Array, " n"],
    ) -> Complex[Array, " n"]:
        """Accumulate one direct matrix entry without storing the matrix."""
        row: scalar_int = flat_position // state_size
        column: scalar_int = flat_position % state_size
        difference: Int[Array, " 3"] = (
            state_indices[row] - state_indices[column]
        )
        residue: Int[Array, " 3"] = jnp.mod(difference, moduli)
        key: Int[Array, ""] = (
            residue[0] * work_shape[1] + residue[1]
        ) * work_shape[2] + residue[2]
        location: Int[Array, ""] = jnp.searchsorted(
            sorted_keys, key, side="left"
        )
        clipped: Int[Array, ""] = jnp.clip(
            location, 0, multiplier_indices.shape[0] - 1
        )
        exact_match: Bool[Array, ""] = (
            (location < multiplier_indices.shape[0])
            & (sorted_keys[clipped] == key)
            & jnp.all(multiplier_indices[order[clipped]] == difference)
        )
        coefficient: Complex[Array, ""] = jnp.where(
            exact_match,
            coefficients[order[clipped]],
            jnp.asarray(0.0 + 0.0j, dtype=coefficients.dtype),
        )
        updated: Complex[Array, " n"] = accumulator.at[row].add(
            coefficient * field[column]
        )
        return updated

    initial: Complex[Array, " n"] = jnp.zeros(
        (state_size,), dtype=jnp.result_type(coefficients, field)
    )
    applied: Complex[Array, " n"] = jax.lax.fori_loop(
        0,
        product_count,
        add_entry,
        initial,
    )
    return applied


def _direct_target_action(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, " n"],
    *,
    adjoint: bool,
) -> Complex[Array, " n"]:
    """Apply the exact manifested coefficient matrix by a direct path."""
    interaction: Complex[Array, " n"] = _direct_multiplier_action(
        manifest.support.state_indices,
        manifest.support.interaction_indices,
        manifest.interaction_coefficients,
        field,
        manifest.support.work_shape,
    )
    absorber: Complex[Array, " n"] = _direct_multiplier_action(
        manifest.support.state_indices,
        manifest.support.absorber_indices,
        manifest.absorber_coefficients,
        field,
        manifest.support.work_shape,
    )
    cap_sign: complex = 1j if adjoint else -1j
    applied: Complex[Array, " n"] = (
        manifest.free_diagonal * field
        - interaction
        + cap_sign * manifest.cap_scale * absorber
    )
    return applied


def _complex_norm(vector: Complex[Array, " n"]) -> Float[Array, ""]:
    """Return a scale-safe Euclidean norm of a complex vector."""
    magnitudes: Float[Array, " n"] = jnp.abs(vector)
    scale: Float[Array, ""] = jnp.max(magnitudes)
    safe_scale: Float[Array, ""] = jnp.where(scale > 0.0, scale, 1.0)
    scaled_norm: Float[Array, ""] = jnp.sqrt(
        jnp.sum((magnitudes / safe_scale) ** 2)
    )
    finite_norm: Float[Array, ""] = scale * scaled_norm
    norm: Float[Array, ""] = jnp.where(
        scale == 0.0,
        0.0,
        jnp.where(jnp.isinf(scale), scale, finite_norm),
    )
    return norm


def _checked_target_action(
    field: Complex[Array, " n"],
    applied: Complex[Array, " n"],
    name: str,
) -> Complex[Array, " n"]:
    """Reject non-finite, subnormal, or globally lost injective actions."""
    lost_nonzero_action: Bool[Array, ""] = has_nonzero_components(field) & (
        ~has_nonzero_components(applied)
    )
    checked: Complex[Array, " n"] = eqx.error_if(
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
) -> Complex[Array, " n"]:
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
    applied_field : Complex[Array, " n"]
        Matrix-free action ``D u - R u - i epsilon_CAP A u``.

    Raises
    ------
    ValueError
        If the field rank or size is invalid.
    equinox.EquinoxRuntimeError
        If the field contains a non-finite value.
    """
    checked_field: Complex[Array, " n"] = _checked_field(
        manifest, field, "field"
    )
    interaction: Complex[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        checked_field,
    )
    absorber: Complex[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        checked_field,
    )
    raw_applied_field: Complex[Array, " n"] = (
        manifest.free_diagonal * checked_field
        - interaction
        - 1j * manifest.cap_scale * absorber
    )
    applied_field: Complex[Array, " n"] = _checked_target_action(
        checked_field,
        raw_applied_field,
        "target action",
    )
    return applied_field


@jaxtyped(typechecker=beartype)
def apply_galerkin_target_adjoint(
    manifest: GalerkinTargetManifest,
    field: Complex[Array, "..."],
) -> Complex[Array, " n"]:
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
    applied_field : Complex[Array, " n"]
        Adjoint action ``D u - R u + i epsilon_CAP A u``.

    Raises
    ------
    ValueError
        If the field rank or size is invalid.
    equinox.EquinoxRuntimeError
        If the field contains a non-finite value.
    """
    checked_field: Complex[Array, " n"] = _checked_field(
        manifest, field, "field"
    )
    interaction: Complex[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        checked_field,
    )
    absorber: Complex[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        checked_field,
    )
    raw_applied_field: Complex[Array, " n"] = (
        manifest.free_diagonal * checked_field
        - interaction
        + 1j * manifest.cap_scale * absorber
    )
    applied_field: Complex[Array, " n"] = _checked_target_action(
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
    incident: Complex[Array, " n"] = _checked_field(
        manifest, incident_field, "incident_field"
    )
    if additional_source is None:
        additional: Complex[Array, " n"] = jnp.zeros_like(incident)
    else:
        additional = _checked_field(
            manifest, additional_source, "additional_source"
        )
    absorber: Complex[Array, " n"] = apply_absorber_action(
        manifest.support,
        manifest.absorber_coefficients,
        incident,
    )
    raw_incident_source: Complex[Array, " n"] = (
        manifest.free_diagonal * incident - 1j * manifest.cap_scale * absorber
    )
    incident_source: Complex[Array, " n"] = _checked_target_action(
        incident,
        raw_incident_source,
        "matched incident action",
    )
    rounded_incident_source, rounded_additional = jax.lax.optimization_barrier(
        (incident_source, additional)
    )
    raw_total_source: Complex[Array, " n"] = (
        rounded_incident_source + rounded_additional
    )
    total_source: Complex[Array, " n"] = eqx.error_if(
        raw_total_source,
        has_subnormal_components(raw_total_source)
        | has_lost_subtraction(
            rounded_incident_source,
            -rounded_additional,
            raw_total_source,
        ),
        "matched total-source addition lost a nonzero component",
    )
    interaction: Complex[Array, " n"] = apply_interaction_product(
        manifest.support,
        manifest.interaction_coefficients,
        incident,
    )
    rounded_interaction, rounded_scattered_additional = (
        jax.lax.optimization_barrier((interaction, additional))
    )
    raw_scattered_source: Complex[Array, " n"] = (
        rounded_interaction + rounded_scattered_additional
    )
    scattered_source: Complex[Array, " n"] = eqx.error_if(
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
    """Recompute one direct original-system residual."""
    checked_field: Complex[Array, " n"] = _checked_field(
        manifest, field, "field"
    )
    checked_source: Complex[Array, " n"] = _checked_field(
        manifest, source, "source"
    )
    applied: Complex[Array, " n"] = _direct_target_action(
        manifest, checked_field, adjoint=adjoint
    )
    checked_applied: Complex[Array, " n"] = _checked_target_action(
        checked_field,
        applied,
        "direct target action",
    )
    rounded_source, rounded_applied = jax.lax.optimization_barrier(
        (checked_source, checked_applied)
    )
    raw_residual: Complex[Array, " n"] = rounded_source - rounded_applied
    residual: Complex[Array, " n"] = eqx.error_if(
        raw_residual,
        has_subnormal_components(raw_residual)
        | has_lost_subtraction(
            rounded_source,
            rounded_applied,
            raw_residual,
        ),
        "physical residual subtraction lost a nonzero component",
    )
    residual_norm: Float[Array, ""] = _complex_norm(residual)
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
    "create_matched_galerkin_source",
    "evaluate_physical_galerkin_adjoint_residual",
    "evaluate_physical_galerkin_residual",
]
