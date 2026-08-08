r"""Define production scalar Galerkin manifests and evidence carriers.

Extended Summary
----------------
This module owns the SC-1 target, matched-source, physical-residual, and
per-result stability carriers. The target manifest binds the independent
Fourier supports and physical coefficients. The target factory derives one
bounded analytic cosine-shell absorber and records its formula. It does not
accept an opaque algebraic operator that a caller can relabel as the
production target.

Routine Listings
----------------
:class:`GalerkinPhysicalResidual`
    Store one independently recomputed physical residual.
:class:`GalerkinSource`
    Store one finite matched-source realization.
:class:`GalerkinSourceBranch`
    Store the admitted finite source-construction branch.
:class:`GalerkinStabilityDisposition`
    Store one per-result stability invocation disposition.
:class:`GalerkinStabilityFailure`
    Store one fail-closed stability invocation reason.
:class:`GalerkinStabilityProof`
    Store one checker-produced exact Route-A proof payload.
:class:`GalerkinStabilityResult`
    Store one per-result operational stability invocation.
:class:`GalerkinStabilityRoute`
    Store the checked singular-value certificate route.
:class:`GalerkinTargetManifest`
    Store one canonical SC-1 finite target manifest.
:func:`create_galerkin_physical_residual`
    Create a validated physical-residual carrier.
:func:`create_galerkin_source`
    Create a validated finite matched-source carrier.
:func:`create_galerkin_stability_proof`
    Create a structurally validated exact stability proof payload.
:func:`create_galerkin_stability_result`
    Create a validated per-result stability invocation.
:func:`create_galerkin_target_manifest`
    Create a canonical SC-1 target from physical coefficient data.

Notes
-----
The target coefficients use the SC.13b multiplier convention. The source
carrier stores a rounded finite realization of the RM-S3 matched-injection
formula. It carries no outward source/action enclosure and does not establish
full RM-S3 implementation eligibility, analytic angular-spectrum, window, or
reduced-flux conformance.
"""

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Complex, Float, Int, jaxtyped

from ptyrodactyl._numeric import (
    has_lost_nonzero_components,
    has_subnormal_components,
)
from ptyrodactyl._physics import coupled_interaction_value

from .born_potential_types import (
    GalerkinProductSupport,
    _cosine_shell_coefficients,
    _has_complete_cosine_shell_support,
)
from .constants import C_LIGHT, E_CHARGE, H_PLANCK, M_E
from .custom_types import scalar_float

_SPACE_DIMENSIONS: int = 3
_SUPPORT_RANK: int = 2
_CONTRACT_VERSION: str = "SC-1"
_COEFFICIENT_NORMALIZATION: str = "SC.13b"
_PRECISION: str = (
    "float64/complex128; voltage-derived coupling and interaction use "
    "canonical 50-mantissa-bit rounding"
)
_ABSORBER_PROFILE: str = "analytic_cosine_shell_v1"
_ABSORBER_PROVENANCE: str = (
    "a(x)=1-product_j cos(pi*x_j/L_j)^2; exact SC.13b coefficients"
)
_INTERACTION_PROVENANCE: str = (
    "chi=sigma_H(accelerating_voltage_kv)*phi; phi in volts; "
    "sigma_H=8*pi^2*m_rel*e/h^2*1e-20; coupling and chi canonically "
    "rounded to 50 mantissa bits"
)
_MIN_CAP_SCALE: float = 64.0 * float(jnp.finfo(jnp.float64).tiny)


def _raise_if(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a structural condition is true."""
    if condition:
        raise ValueError(message)


def _checked_coefficients(
    indices: Int[Array, "p 3"],
    coefficients: Complex[Array, " p"],
    name: str,
) -> Complex[Array, " p"]:
    """Attach finite and exact Hermitian-symmetry checks."""
    inverse_indices: Int[Array, "p 3"] = -indices
    forward_order: Int[Array, " p"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    inverse_order: Int[Array, " p"] = jnp.lexsort(
        (
            inverse_indices[:, 2],
            inverse_indices[:, 1],
            inverse_indices[:, 0],
        )
    )
    nonhermitian: Bool[Array, ""] = jnp.any(
        indices[forward_order] != inverse_indices[inverse_order]
    ) | jnp.any(
        coefficients[forward_order] != jnp.conj(coefficients[inverse_order])
    )
    checked: Complex[Array, " p"] = eqx.error_if(
        coefficients,
        jnp.any(~jnp.isfinite(coefficients)) | nonhermitian,
        f"{name} must be finite and exactly Hermitian",
    )
    return checked


def _derive_interaction_coefficients(
    voltage_kv: Float[Array, ""],
    voltage_coefficients: Complex[Array, " p"],
) -> tuple[Float[Array, ""], Complex[Array, " p"]]:
    """Derive the canonical coupling and finite SC-1 interaction."""
    raw_coupling, raw_interaction = coupled_interaction_value(
        voltage_coefficients,
        voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    coupling: Float[Array, ""] = eqx.error_if(
        raw_coupling,
        (~jnp.isfinite(raw_coupling)) | (raw_coupling <= 0.0),
        "voltage-derived interaction coupling must be finite and positive",
    )
    interaction: Complex[Array, " p"] = eqx.error_if(
        raw_interaction,
        jnp.any(~jnp.isfinite(raw_interaction))
        | (~jnp.isfinite(voltage_kv))
        | (voltage_kv <= 0.0)
        | has_subnormal_components(voltage_coefficients)
        | has_lost_nonzero_components(voltage_coefficients, raw_interaction),
        "voltage-derived interaction coefficients must be finite and "
        "preserve every nonzero normal voltage component",
    )
    result: tuple[Float[Array, ""], Complex[Array, " p"]] = (
        coupling,
        interaction,
    )
    return result


class GalerkinSourceBranch(str, Enum):
    """Store the admitted finite source-construction branch.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    FINITE_MATCHED : str
        Rounded finite realization of ``S_inc = H_0 v_inc``.
    """

    FINITE_MATCHED = "finite_matched"


class GalerkinStabilityDisposition(str, Enum):
    """Store one per-result stability invocation disposition.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    OPERATIONAL_PASS : str
        The checked state bound meets its preregistered budget.
    TYPED_FALLBACK : str
        The matrix proof is valid, but the state budget is missed.
    REJECTED : str
        A target, proof, or checker predicate failed closed.
    """

    OPERATIONAL_PASS = "operational_pass"  # noqa: S105
    TYPED_FALLBACK = "typed_fallback"
    REJECTED = "rejected"


class GalerkinStabilityFailure(str, Enum):
    """Store one fail-closed stability invocation reason.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    NONE : str
        No invocation predicate failed.
    CHECKER_DIMENSION_LIMIT : str
        The bounded exact checker does not admit the target dimension.
    ARITHMETIC_RANGE_FAILURE : str
        Outward binary64 reporting would overflow, underflow, or be infinite.
    INVALID_OPERATOR_CONTRACT : str
        The target is not an exact Hermitian-real/positive-CAP instance.
    INVALID_SUBMISSION_CONTRACT : str
        The bound source or submitted field is structurally invalid.
    NO_POSITIVE_ABSORBER_FLOOR : str
        Exact Gershgorin arithmetic did not prove a positive absorber floor.
    PROOF_RECORD_MISMATCH : str
        The submitted proof differs from independent checker reconstruction.
    STATE_BUDGET_MISSED : str
        The exact residual-to-stability comparison misses the state budget.
    """

    NONE = "none"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"
    CHECKER_DIMENSION_LIMIT = "checker_dimension_limit"
    INVALID_OPERATOR_CONTRACT = "invalid_operator_contract"
    INVALID_SUBMISSION_CONTRACT = "invalid_submission_contract"
    NO_POSITIVE_ABSORBER_FLOOR = "no_positive_absorber_floor"
    PROOF_RECORD_MISMATCH = "proof_record_mismatch"
    STATE_BUDGET_MISSED = "state_budget_missed"


class GalerkinStabilityRoute(str, Enum):
    """Store the checked singular-value certificate route.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    ABSORBER_FLOOR : str
        Exact-dyadic Route A using a Gershgorin absorber floor.
    """

    ABSORBER_FLOOR = "absorber_floor"


class GalerkinTargetManifest(eqx.Module):
    """Store one canonical SC-1 finite target manifest.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    support : GalerkinProductSupport
        Independent state, interaction, absorber, and work supports.
    preterminal_indices : Int[Array, "m 3"]
        Exact integer reciprocal indices in the state-side preterminal.
    voltage_coefficients : Complex[Array, " p"]
        Bound SC.13b electrostatic-potential coefficients in volts.
    interaction_coefficients : Complex[Array, " p"]
        Voltage-derived SC.13b interaction coefficients in inverse-square
        Angstroms.
    interaction_coupling : Float[Array, ""]
        Voltage-derived Helmholtz coupling in inverse-square Angstroms per
        volt.
    absorber_coefficients : Complex[Array, " q"]
        Factory-derived exact SC.13b coefficients of the dimensionless
        analytic cosine-shell profile.
    free_diagonal : Float[Array, " n"]
        Carrier-shifted SC.22 diagonal in inverse-square Angstroms.
    carrier : Float[Array, " 3"]
        Real incident carrier in radians per Angstrom.
    box_lengths : Float[Array, " 3"]
        Physical box lengths in Angstroms, ordered by coordinate axis.
    wavenumber : Float[Array, ""]
        Positive vacuum angular wavenumber in radians per Angstrom.
    accelerating_voltage_kv : Float[Array, ""]
        Positive accelerating voltage in kilovolts.
    cap_scale : Float[Array, ""]
        Positive normal-range physical CAP scale in inverse-square Angstroms.
    target_name : str
        Static nonempty canonical target name. This value affects tracing.
    contract_version : str
        Static normative scalar contract version. This value affects tracing.
    coefficient_normalization : str
        Static multiplier-coefficient normalization identifier. This value
        affects tracing.
    precision : str
        Static stored arithmetic precision identifier. This value affects
        tracing.
    absorber_profile : str
        Static analytic bounded-profile identifier. This value affects
        tracing.
    absorber_coefficient_provenance : str
        Static coefficient formula and normalization provenance. This value
        affects tracing.
    interaction_coefficient_provenance : str
        Static voltage-to-interaction formula and unit provenance. This value
        affects tracing.

    Notes
    -----
    The profile is ``1 - product_j cos(pi x_j / L_j)^2`` on the centered
    periodic box. It is dimensionless, lies in ``[0, 1]``, and has nonzero
    modes only in ``{-1, 0, 1}^3``.

    See Also
    --------
    :func:`create_galerkin_target_manifest`
        Construct and validate this target from physical coefficients.
    """

    support: GalerkinProductSupport
    preterminal_indices: Int[Array, "m 3"]
    voltage_coefficients: Complex[Array, " p"]
    interaction_coefficients: Complex[Array, " p"]
    interaction_coupling: Float[Array, ""]
    absorber_coefficients: Complex[Array, " q"]
    free_diagonal: Float[Array, " n"]
    carrier: Float[Array, " 3"]
    box_lengths: Float[Array, " 3"]
    wavenumber: Float[Array, ""]
    accelerating_voltage_kv: Float[Array, ""]
    cap_scale: Float[Array, ""]
    target_name: str = eqx.field(static=True)
    contract_version: str = eqx.field(static=True)
    coefficient_normalization: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    absorber_profile: str = eqx.field(static=True)
    absorber_coefficient_provenance: str = eqx.field(static=True)
    interaction_coefficient_provenance: str = eqx.field(static=True)


class GalerkinSource(eqx.Module):
    """Store one finite matched-source realization.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    incident_field : Complex[Array, " n"]
        Declared finite incident-vector coefficients.
    incident_source : Complex[Array, " n"]
        Rounded finite matched source ``(D - i B) incident_field``.
    additional_source : Complex[Array, " n"]
        Separately declared finite source beyond matched injection.
    total_source : Complex[Array, " n"]
        Complete total-field right-hand side.
    scattered_source : Complex[Array, " n"]
        Equivalent scattered-field right-hand side.
    branch : GalerkinSourceBranch
        Static finite source branch. This value affects tracing.

    See Also
    --------
    :func:`create_galerkin_source`
        Validate one finite matched-source decomposition.
    """

    incident_field: Complex[Array, " n"]
    incident_source: Complex[Array, " n"]
    additional_source: Complex[Array, " n"]
    total_source: Complex[Array, " n"]
    scattered_source: Complex[Array, " n"]
    branch: GalerkinSourceBranch = eqx.field(static=True)


class GalerkinPhysicalResidual(eqx.Module):
    """Store one independently recomputed physical residual.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    residual : Complex[Array, " n"]
        Original-system residual from the direct coefficient action.
    residual_norm : Float[Array, ""]
        Euclidean norm of the independently recomputed residual.

    See Also
    --------
    :func:`create_galerkin_physical_residual`
        Validate a physical residual and its norm.
    """

    residual: Complex[Array, " n"]
    residual_norm: Float[Array, ""]


class GalerkinStabilityProof(eqx.Module):
    """Store one checker-produced exact Route-A proof payload.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    target_digest : str
        Static SHA-256 checksum of the canonical target manifest.
    result_digest : str
        Static SHA-256 checksum of the bound source and submitted result.
    floor_numerator : int
        Static exact lower-bound numerator.
    floor_denominator : int
        Static positive exact lower-bound denominator.
    residual_squared_numerator : int
        Static exact squared-residual numerator.
    residual_squared_denominator : int
        Static positive exact squared-residual denominator.
    state_budget_numerator : int
        Static exact preregistered state-budget numerator.
    state_budget_denominator : int
        Static positive preregistered state-budget denominator.
    route : GalerkinStabilityRoute
        Static checked proof route.
    failure : GalerkinStabilityFailure
        Static checker failure, or ``NONE``.
    checker_id : str
        Static trusted checker implementation identifier.

    See Also
    --------
    :func:`create_galerkin_stability_proof`
        Validate one exact proof payload.
    """

    target_digest: str = eqx.field(static=True)
    result_digest: str = eqx.field(static=True)
    floor_numerator: int = eqx.field(static=True)
    floor_denominator: int = eqx.field(static=True)
    residual_squared_numerator: int = eqx.field(static=True)
    residual_squared_denominator: int = eqx.field(static=True)
    state_budget_numerator: int = eqx.field(static=True)
    state_budget_denominator: int = eqx.field(static=True)
    route: GalerkinStabilityRoute = eqx.field(static=True)
    failure: GalerkinStabilityFailure = eqx.field(static=True)
    checker_id: str = eqx.field(static=True)


class GalerkinStabilityResult(eqx.Module):
    """Store one per-result operational stability invocation.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    lower_singular_bound : Float[Array, ""]
        Downward-rounded lower singular-value bound.
    residual_upper_bound : Float[Array, ""]
        Upward-rounded same-target residual-norm enclosure.
    state_error_upper_bound : Float[Array, ""]
        Upward-rounded residual-to-stability state bound.
    state_budget : Float[Array, ""]
        Preregistered state-error budget.
    route : GalerkinStabilityRoute
        Static checked proof route.
    disposition : GalerkinStabilityDisposition
        Static per-result invocation disposition.
    failure : GalerkinStabilityFailure
        Static fail-closed reason, or ``NONE``.
    target_digest : str
        Static canonical target checksum.
    result_digest : str
        Static bound-result checksum.
    checker_id : str
        Static trusted checker implementation identifier.

    See Also
    --------
    :func:`create_galerkin_stability_result`
        Validate one per-result stability invocation.
    """

    lower_singular_bound: Float[Array, ""]
    residual_upper_bound: Float[Array, ""]
    state_error_upper_bound: Float[Array, ""]
    state_budget: Float[Array, ""]
    route: GalerkinStabilityRoute = eqx.field(static=True)
    disposition: GalerkinStabilityDisposition = eqx.field(static=True)
    failure: GalerkinStabilityFailure = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    result_digest: str = eqx.field(static=True)
    checker_id: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_target_manifest(
    support: GalerkinProductSupport,
    preterminal_indices: Int[Array, "..."],
    voltage_coefficients: Complex[Array, "..."],
    box_lengths: Float[Array, "..."],
    carrier: Float[Array, "..."],
    accelerating_voltage_kv: scalar_float,
    cap_scale: scalar_float,
    target_name: str,
) -> GalerkinTargetManifest:
    r"""Create a canonical SC-1 target from physical coefficient data.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Implementation Logic
    --------------------
    1. Bind the four product supports, preterminal, and voltage coefficients.
    2. Derive the interaction and analytic absorber with their provenance.
    3. Validate the on-shell carrier, box, voltage, and physical CAP scale.
    4. Derive and store the shifted SC.22 diagonal from the bound metadata.

    Parameters
    ----------
    support : GalerkinProductSupport
        Validated state, interaction, absorber, and work supports.
    preterminal_indices : Int[Array, "..."]
        Exact preterminal reciprocal indices with shape ``(m, 3)``.
    voltage_coefficients : Complex[Array, "..."]
        SC.13b electrostatic-potential coefficients in volts. The factory
        derives ``chi = sigma_H(accelerating_voltage_kv) * phi``.
    box_lengths : Float[Array, "..."]
        Three positive box lengths in Angstroms.
    carrier : Float[Array, "..."]
        Three real carrier components in radians per Angstrom.
    accelerating_voltage_kv : scalar_float
        Positive accelerating voltage in kilovolts.
    cap_scale : scalar_float
        Positive normal-range physical CAP scale in inverse-square Angstroms.
    target_name : str
        Nonempty canonical name for this finite target.

    Returns
    -------
    manifest : GalerkinTargetManifest
        Validated target with the derived absorber and shifted free diagonal.

    Raises
    ------
    ValueError
        If a static name, rank, or shape is invalid.
    equinox.EquinoxRuntimeError
        If coefficients, supports, units, or the on-shell condition fail.

    Notes
    -----
    The factory accepts neither pre-coupled interaction coefficients nor
    arbitrary absorber coefficients. It derives the voltage-dependent
    interaction and ``a(x) = 1 - product_j cos(pi x_j / L_j)^2``, requiring
    all 27 absorber modes. It also does not accept a prior
    ``GalerkinOperator``. These restrictions prevent arbitrary algebraic data
    from being relabeled as the SC-1 target. The CAP floor preserves the
    nonzero analytic coefficient products in normal-range arithmetic; it is
    not a condition-number or Krylov-convergence guarantee.
    """
    _raise_if(not target_name.strip(), "target_name must be nonempty")
    preterminal_array: Int[Array, "m 3"] = jnp.asarray(
        preterminal_indices, dtype=jnp.int64
    )
    voltage_coefficients_array: Complex[Array, " p"] = jnp.asarray(
        voltage_coefficients, dtype=jnp.complex128
    )
    absorber_array: Complex[Array, " q"] = _cosine_shell_coefficients(
        support.absorber_indices
    )
    box_array: Float[Array, " 3"] = jnp.asarray(box_lengths, dtype=jnp.float64)
    carrier_array: Float[Array, " 3"] = jnp.asarray(carrier, dtype=jnp.float64)
    voltage_array: Float[Array, ""] = jnp.asarray(
        accelerating_voltage_kv, dtype=jnp.float64
    )
    cap_array: Float[Array, ""] = jnp.asarray(cap_scale, dtype=jnp.float64)

    _raise_if(
        preterminal_array.ndim != _SUPPORT_RANK
        or preterminal_array.shape[1:] != (_SPACE_DIMENSIONS,),
        "preterminal_indices must have shape (m, 3)",
    )
    _raise_if(
        preterminal_array.shape[0] == 0,
        "preterminal_indices must be nonempty",
    )
    _raise_if(
        voltage_coefficients_array.ndim != 1,
        "voltage_coefficients must be 1D",
    )
    _raise_if(
        voltage_coefficients_array.shape[0]
        != support.interaction_indices.shape[0],
        "voltage_coefficients must match the interaction support",
    )
    _raise_if(
        box_array.shape != (_SPACE_DIMENSIONS,),
        "box_lengths must have shape (3,)",
    )
    _raise_if(
        carrier_array.shape != (_SPACE_DIMENSIONS,),
        "carrier must have shape (3,)",
    )
    for values, name in (
        (voltage_array, "accelerating_voltage_kv"),
        (cap_array, "cap_scale"),
    ):
        _raise_if(values.shape != (), f"{name} must be a scalar")

    work_moduli: Int[Array, " 3"] = jnp.asarray(
        support.work_shape, dtype=jnp.int64
    )
    terminal_residues: Int[Array, "m 3"] = jnp.mod(
        preterminal_array, work_moduli
    )
    state_residues: Int[Array, "n 3"] = jnp.mod(
        support.state_indices, work_moduli
    )
    terminal_keys: Int[Array, " m"] = (
        terminal_residues[:, 0] * support.work_shape[1]
        + terminal_residues[:, 1]
    ) * support.work_shape[2] + terminal_residues[:, 2]
    state_keys: Int[Array, " n"] = (
        state_residues[:, 0] * support.work_shape[1] + state_residues[:, 1]
    ) * support.work_shape[2] + state_residues[:, 2]
    state_order: Int[Array, " n"] = jnp.argsort(state_keys)
    sorted_state_keys: Int[Array, " n"] = state_keys[state_order]
    terminal_locations: Int[Array, " m"] = jnp.searchsorted(
        sorted_state_keys, terminal_keys, side="left"
    )
    clipped_locations: Int[Array, " m"] = jnp.clip(
        terminal_locations, 0, support.state_indices.shape[0] - 1
    )
    terminal_matches: Bool[Array, " m"] = (
        terminal_locations < support.state_indices.shape[0]
    ) & (sorted_state_keys[clipped_locations] == terminal_keys)
    exact_terminal_matches: Bool[Array, " m"] = jnp.all(
        support.state_indices[state_order[clipped_locations]]
        == preterminal_array,
        axis=-1,
    )
    sorted_terminal_keys: Int[Array, " m"] = jnp.sort(terminal_keys)
    terminal_duplicates: Bool[Array, ""] = jnp.any(
        sorted_terminal_keys[1:] == sorted_terminal_keys[:-1]
    )
    checked_preterminal: Int[Array, "m 3"] = eqx.error_if(
        preterminal_array,
        ~jnp.all(terminal_matches & exact_terminal_matches)
        | terminal_duplicates,
        "preterminal support must be unique and contained in state support",
    )
    checked_voltage_coefficients: Complex[Array, " p"] = _checked_coefficients(
        support.interaction_indices,
        voltage_coefficients_array,
        "voltage_coefficients",
    )
    checked_absorber: Complex[Array, " q"] = _checked_coefficients(
        support.absorber_indices, absorber_array, "absorber_coefficients"
    )
    checked_absorber = eqx.error_if(
        checked_absorber,
        ~_has_complete_cosine_shell_support(
            support.absorber_indices, support.work_shape
        ),
        "absorber support must contain all cosine-shell profile modes",
    )
    checked_box: Float[Array, " 3"] = eqx.error_if(
        box_array,
        jnp.any(~jnp.isfinite(box_array)) | jnp.any(box_array <= 0.0),
        "box_lengths must be finite and positive",
    )
    checked_carrier: Float[Array, " 3"] = eqx.error_if(
        carrier_array,
        jnp.any(~jnp.isfinite(carrier_array)),
        "carrier must be finite",
    )
    checked_voltage: Float[Array, ""] = eqx.error_if(
        voltage_array,
        (~jnp.isfinite(voltage_array)) | (voltage_array <= 0.0),
        "accelerating_voltage_kv must be finite and positive",
    )
    interaction_coupling, checked_interaction = (
        _derive_interaction_coefficients(
            voltage_array,
            voltage_coefficients_array,
        )
    )
    checked_cap: Float[Array, ""] = eqx.error_if(
        cap_array,
        (~jnp.isfinite(cap_array)) | (cap_array < _MIN_CAP_SCALE),
        "cap_scale must be finite and preserve every nonzero analytic "
        "absorber coefficient in normal-range arithmetic",
    )
    energy_joule: Float[Array, ""] = (
        checked_voltage * 1000.0 * jnp.asarray(E_CHARGE)
    )
    wavelength_metre: Float[Array, ""] = jnp.sqrt(
        (jnp.asarray(H_PLANCK) * jnp.asarray(C_LIGHT)) ** 2
        / (
            energy_joule
            * (
                2.0 * jnp.asarray(M_E) * jnp.asarray(C_LIGHT) ** 2
                + energy_joule
            )
        )
    )
    wavelength_angstrom: Float[Array, ""] = 1.0e10 * wavelength_metre
    wavenumber: Float[Array, ""] = 2.0 * jnp.pi / wavelength_angstrom
    checked_wavenumber: Float[Array, ""] = eqx.error_if(
        wavenumber,
        (~jnp.isfinite(wavenumber)) | (wavenumber <= 0.0),
        "voltage-derived wavenumber must be finite and positive",
    )
    shell_tolerance: Float[Array, ""] = (
        64.0
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(1.0, checked_wavenumber)
    )
    checked_carrier = eqx.error_if(
        checked_carrier,
        jnp.abs(jnp.linalg.norm(checked_carrier) - checked_wavenumber)
        > shell_tolerance,
        "carrier must satisfy the voltage-derived on-shell condition",
    )

    reciprocal_frequencies: Float[Array, "n 3"] = (
        support.state_indices / checked_box[None, :]
    )
    physical_wavevectors: Float[Array, "n 3"] = (
        checked_carrier[None, :] + 2.0 * jnp.pi * reciprocal_frequencies
    )
    derived_free_diagonal: Float[Array, " n"] = (
        jnp.sum(physical_wavevectors**2, axis=1) - checked_wavenumber**2
    )
    free_diagonal: Float[Array, " n"] = eqx.error_if(
        derived_free_diagonal,
        jnp.any(~jnp.isfinite(derived_free_diagonal)),
        "derived free_diagonal must contain only finite values",
    )
    manifest: GalerkinTargetManifest = GalerkinTargetManifest(
        support=support,
        preterminal_indices=checked_preterminal,
        voltage_coefficients=checked_voltage_coefficients,
        interaction_coefficients=checked_interaction,
        interaction_coupling=interaction_coupling,
        absorber_coefficients=checked_absorber,
        free_diagonal=free_diagonal,
        carrier=checked_carrier,
        box_lengths=checked_box,
        wavenumber=checked_wavenumber,
        accelerating_voltage_kv=checked_voltage,
        cap_scale=checked_cap,
        target_name=target_name,
        contract_version=_CONTRACT_VERSION,
        coefficient_normalization=_COEFFICIENT_NORMALIZATION,
        precision=_PRECISION,
        absorber_profile=_ABSORBER_PROFILE,
        absorber_coefficient_provenance=_ABSORBER_PROVENANCE,
        interaction_coefficient_provenance=_INTERACTION_PROVENANCE,
    )
    return manifest


@jaxtyped(typechecker=beartype)
def create_galerkin_source(
    incident_field: Complex[Array, "..."],
    incident_source: Complex[Array, "..."],
    additional_source: Complex[Array, "..."],
    total_source: Complex[Array, "..."],
    scattered_source: Complex[Array, "..."],
    branch: GalerkinSourceBranch | str,
) -> GalerkinSource:
    """Create a validated finite matched-source carrier.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Parameters
    ----------
    incident_field : Complex[Array, "..."]
        Nonempty finite incident-vector coefficients.
    incident_source : Complex[Array, "..."]
        Matched finite incident source.
    additional_source : Complex[Array, "..."]
        Separately declared additional source.
    total_source : Complex[Array, "..."]
        Complete total-field right-hand side.
    scattered_source : Complex[Array, "..."]
        Equivalent scattered-field right-hand side.
    branch : GalerkinSourceBranch | str
        Static admitted source branch.

    Returns
    -------
    source : GalerkinSource
        Validated source decomposition.

    Raises
    ------
    ValueError
        If vectors are empty or have inconsistent shapes.
    equinox.EquinoxRuntimeError
        If a source vector contains a non-finite value.

    Notes
    -----
    This factory validates storage only. Semantic RM-S3 conformance comes
    exclusively from :func:`ptyrodactyl.born.create_matched_galerkin_source`.
    """
    checked_branch: GalerkinSourceBranch = GalerkinSourceBranch(branch)
    arrays = tuple(
        jnp.asarray(value, dtype=jnp.complex128)
        for value in (
            incident_field,
            incident_source,
            additional_source,
            total_source,
            scattered_source,
        )
    )
    reference_shape: tuple[int, ...] = arrays[0].shape
    _raise_if(len(reference_shape) != 1, "source vectors must be 1D")
    _raise_if(reference_shape[0] == 0, "source vectors must be nonempty")
    _raise_if(
        any(values.shape != reference_shape for values in arrays[1:]),
        "source vectors must have matching shapes",
    )
    checked_arrays = tuple(
        eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)) | has_subnormal_components(values),
            "source vectors must be finite and contain no nonzero "
            "subnormal components",
        )
        for values in arrays
    )
    source: GalerkinSource = GalerkinSource(
        incident_field=checked_arrays[0],
        incident_source=checked_arrays[1],
        additional_source=checked_arrays[2],
        total_source=checked_arrays[3],
        scattered_source=checked_arrays[4],
        branch=checked_branch,
    )
    return source


@jaxtyped(typechecker=beartype)
def create_galerkin_physical_residual(
    residual: Complex[Array, "..."],
    residual_norm: scalar_float,
) -> GalerkinPhysicalResidual:
    """Create a validated physical-residual carrier.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Parameters
    ----------
    residual : Complex[Array, "..."]
        Nonempty independently recomputed residual vector.
    residual_norm : scalar_float
        Finite non-negative Euclidean residual norm.

    Returns
    -------
    physical_residual : GalerkinPhysicalResidual
        Validated residual vector and norm.

    Raises
    ------
    ValueError
        If the residual or norm has invalid structure.
    equinox.EquinoxRuntimeError
        If the residual or norm is non-finite or negative.
    """
    residual_array: Complex[Array, " n"] = jnp.asarray(
        residual, dtype=jnp.complex128
    )
    norm_array: Float[Array, ""] = jnp.asarray(
        residual_norm, dtype=jnp.float64
    )
    _raise_if(residual_array.ndim != 1, "residual must be 1D")
    _raise_if(residual_array.shape[0] == 0, "residual must be nonempty")
    _raise_if(norm_array.shape != (), "residual_norm must be a scalar")
    checked_residual: Complex[Array, " n"] = eqx.error_if(
        residual_array,
        jnp.any(~jnp.isfinite(residual_array))
        | has_subnormal_components(residual_array),
        "residual must be finite and contain no nonzero subnormal components",
    )
    checked_norm: Float[Array, ""] = eqx.error_if(
        norm_array,
        (~jnp.isfinite(norm_array)) | (norm_array < 0.0),
        "residual_norm must be finite and non-negative",
    )
    physical_residual: GalerkinPhysicalResidual = GalerkinPhysicalResidual(
        residual=checked_residual,
        residual_norm=checked_norm,
    )
    return physical_residual


@beartype
def create_galerkin_stability_proof(  # noqa: PLR0913
    target_digest: str,
    result_digest: str,
    floor_numerator: int,
    floor_denominator: int,
    residual_squared_numerator: int,
    residual_squared_denominator: int,
    state_budget_numerator: int,
    state_budget_denominator: int,
    route: GalerkinStabilityRoute | str,
    failure: GalerkinStabilityFailure | str,
    checker_id: str,
) -> GalerkinStabilityProof:
    """Create a structurally validated exact stability proof payload.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Parameters
    ----------
    target_digest : str
        Nonempty canonical target checksum.
    result_digest : str
        Nonempty canonical bound-result checksum.
    floor_numerator : int
        Non-negative exact floor numerator.
    floor_denominator : int
        Positive exact floor denominator.
    residual_squared_numerator : int
        Non-negative exact squared-residual numerator.
    residual_squared_denominator : int
        Positive exact squared-residual denominator.
    state_budget_numerator : int
        Positive exact state-budget numerator.
    state_budget_denominator : int
        Positive exact state-budget denominator.
    route : GalerkinStabilityRoute | str
        Static checked route.
    failure : GalerkinStabilityFailure | str
        Static checker failure or ``NONE``.
    checker_id : str
        Nonempty trusted checker identifier.

    Returns
    -------
    proof : GalerkinStabilityProof
        Structurally valid exact proof payload.

    Raises
    ------
    ValueError
        If a digest, integer, denominator, or checker identifier is invalid.

    Notes
    -----
    Structural construction is not checker acceptance. The stability
    invocation independently reconstructs and compares this payload.
    """
    checked_route: GalerkinStabilityRoute = GalerkinStabilityRoute(route)
    checked_failure: GalerkinStabilityFailure = GalerkinStabilityFailure(
        failure
    )
    _raise_if(not target_digest, "target_digest must be nonempty")
    _raise_if(not result_digest, "result_digest must be nonempty")
    _raise_if(not checker_id, "checker_id must be nonempty")
    for value, name in (
        (floor_numerator, "floor_numerator"),
        (residual_squared_numerator, "residual_squared_numerator"),
    ):
        _raise_if(
            isinstance(value, bool) or value < 0,
            f"{name} must be non-negative",
        )
    for value, name in (
        (floor_denominator, "floor_denominator"),
        (residual_squared_denominator, "residual_squared_denominator"),
        (state_budget_numerator, "state_budget_numerator"),
        (state_budget_denominator, "state_budget_denominator"),
    ):
        _raise_if(
            isinstance(value, bool) or value <= 0, f"{name} must be positive"
        )
    proof: GalerkinStabilityProof = GalerkinStabilityProof(
        target_digest=target_digest,
        result_digest=result_digest,
        floor_numerator=floor_numerator,
        floor_denominator=floor_denominator,
        residual_squared_numerator=residual_squared_numerator,
        residual_squared_denominator=residual_squared_denominator,
        state_budget_numerator=state_budget_numerator,
        state_budget_denominator=state_budget_denominator,
        route=checked_route,
        failure=checked_failure,
        checker_id=checker_id,
    )
    return proof


@jaxtyped(typechecker=beartype)
def create_galerkin_stability_result(
    lower_singular_bound: scalar_float,
    residual_upper_bound: scalar_float,
    state_error_upper_bound: scalar_float,
    state_budget: scalar_float,
    route: GalerkinStabilityRoute | str,
    disposition: GalerkinStabilityDisposition | str,
    failure: GalerkinStabilityFailure | str,
    target_digest: str,
    result_digest: str,
    checker_id: str,
) -> GalerkinStabilityResult:
    """Create a validated per-result stability invocation.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Parameters
    ----------
    lower_singular_bound : scalar_float
        Non-negative downward-rounded singular-value floor.
    residual_upper_bound : scalar_float
        Non-negative upward-rounded residual enclosure.
    state_error_upper_bound : scalar_float
        Non-negative state-error enclosure. Infinity marks rejection.
    state_budget : scalar_float
        Positive normal-range preregistered state-error budget.
    route : GalerkinStabilityRoute | str
        Static checked route.
    disposition : GalerkinStabilityDisposition | str
        Static invocation disposition.
    failure : GalerkinStabilityFailure | str
        Static failure reason or ``NONE``.
    target_digest : str
        Nonempty target checksum.
    result_digest : str
        Nonempty bound-result checksum.
    checker_id : str
        Nonempty checker identifier.

    Returns
    -------
    result : GalerkinStabilityResult
        Validated per-result invocation carrier.

    Raises
    ------
    ValueError
        If a scalar or string has invalid structure.
    equinox.EquinoxRuntimeError
        If a bound is NaN, negative, or the budget is not positive.
    """
    checked_route: GalerkinStabilityRoute = GalerkinStabilityRoute(route)
    checked_disposition: GalerkinStabilityDisposition = (
        GalerkinStabilityDisposition(disposition)
    )
    checked_failure: GalerkinStabilityFailure = GalerkinStabilityFailure(
        failure
    )
    if checked_disposition is GalerkinStabilityDisposition.OPERATIONAL_PASS:
        _raise_if(
            checked_failure is not GalerkinStabilityFailure.NONE,
            "an operational pass must have failure NONE",
        )
    elif checked_disposition is GalerkinStabilityDisposition.TYPED_FALLBACK:
        _raise_if(
            checked_failure
            is not GalerkinStabilityFailure.STATE_BUDGET_MISSED,
            "a typed fallback must record STATE_BUDGET_MISSED",
        )
    else:
        _raise_if(
            checked_failure is GalerkinStabilityFailure.NONE,
            "a rejected result must record a failure",
        )
    values = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            lower_singular_bound,
            residual_upper_bound,
            state_error_upper_bound,
            state_budget,
        )
    )
    _raise_if(
        any(value.shape != () for value in values),
        "stability bounds must be scalars",
    )
    _raise_if(not target_digest, "target_digest must be nonempty")
    _raise_if(not result_digest, "result_digest must be nonempty")
    _raise_if(not checker_id, "checker_id must be nonempty")
    lower, residual_upper, state_upper, budget = values
    lower = eqx.error_if(
        lower,
        jnp.isnan(lower) | (lower < 0.0),
        "lower_singular_bound must be non-negative",
    )
    residual_upper = eqx.error_if(
        residual_upper,
        jnp.isnan(residual_upper) | (residual_upper < 0.0),
        "residual_upper_bound must be non-negative",
    )
    state_upper = eqx.error_if(
        state_upper,
        jnp.isnan(state_upper) | (state_upper < 0.0),
        "state_error_upper_bound must be non-negative",
    )
    budget = eqx.error_if(
        budget,
        (~jnp.isfinite(budget)) | (budget < jnp.finfo(jnp.float64).tiny),
        "state_budget must be finite and at least the smallest normal float64",
    )
    if checked_disposition is GalerkinStabilityDisposition.REJECTED:
        lower = eqx.error_if(
            lower,
            lower != 0.0,
            "a rejected result must report a zero singular bound",
        )
        residual_upper = eqx.error_if(
            residual_upper,
            ~jnp.isposinf(residual_upper),
            "a rejected result must report an infinite residual bound",
        )
        state_upper = eqx.error_if(
            state_upper,
            ~jnp.isposinf(state_upper),
            "a rejected result must report an infinite state bound",
        )
    else:
        lower = eqx.error_if(
            lower,
            (~jnp.isfinite(lower)) | (lower <= 0.0),
            "an accepted matrix proof needs a finite positive singular bound",
        )
        residual_upper = eqx.error_if(
            residual_upper,
            ~jnp.isfinite(residual_upper),
            "an accepted matrix proof needs a finite residual bound",
        )
        state_upper = eqx.error_if(
            state_upper,
            ~jnp.isfinite(state_upper),
            "an accepted matrix proof needs a finite state bound",
        )
        if (
            checked_disposition
            is GalerkinStabilityDisposition.OPERATIONAL_PASS
        ):
            state_upper = eqx.error_if(
                state_upper,
                state_upper > budget,
                "an operational pass must meet the state budget",
            )
        else:
            state_upper = eqx.error_if(
                state_upper,
                state_upper <= budget,
                "a typed fallback must miss the state budget",
            )
    result: GalerkinStabilityResult = GalerkinStabilityResult(
        lower_singular_bound=lower,
        residual_upper_bound=residual_upper,
        state_error_upper_bound=state_upper,
        state_budget=budget,
        route=checked_route,
        disposition=checked_disposition,
        failure=checked_failure,
        target_digest=target_digest,
        result_digest=result_digest,
        checker_id=checker_id,
    )
    return result


__all__: list[str] = [
    "GalerkinPhysicalResidual",
    "GalerkinSource",
    "GalerkinSourceBranch",
    "GalerkinStabilityDisposition",
    "GalerkinStabilityFailure",
    "GalerkinStabilityProof",
    "GalerkinStabilityResult",
    "GalerkinStabilityRoute",
    "GalerkinTargetManifest",
    "create_galerkin_physical_residual",
    "create_galerkin_source",
    "create_galerkin_stability_proof",
    "create_galerkin_stability_result",
    "create_galerkin_target_manifest",
]
