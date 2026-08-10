r"""Define production scalar Galerkin manifests and evidence carriers.

Extended Summary
----------------
This module owns the SC-1 target, matched-source, physical-residual, and
per-result stability carriers. The target manifest nests the checked
acquisition/VC-1 realization and the RM-S2 fixed-linear ledger, then stores
only the remaining derived interaction, absorber, and physical scalar leaves.
The target factory does not accept raw support or coefficient data that a
caller can relabel as the production target.

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

from ptyrodactyl._tools import (
    coupled_interaction_value,
    has_lost_nonzero_components,
    has_subnormal_components,
    interval_add,
    interval_divide_positive,
    interval_multiply,
    point_interval,
)

from .acquisition_types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
    GalerkinDirectionDisposition,
)
from .born_potential_types import (
    GalerkinProductSupport,
    _cosine_shell_coefficients,
    _has_complete_cosine_shell_support,
)
from .constants import C_LIGHT, E_CHARGE, H_PLANCK, M_E
from .custom_types import scalar_float
from .potential_types import Potential3D
from .realization_error_types import (
    GalerkinFixedLinearAbsorberRoute,
    GalerkinFixedLinearErrorLedger,
)
from .realization_types import GalerkinPotentialRealization

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
_TWO_PI_LOWER: float = 2.0 * float.fromhex("0x1.921fb54442d18p+1")


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a true structural condition.

    Parameters
    ----------
    condition : bool
        Whether the structural contract is invalid.
    message : str
        Error message for the rejected contract.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _checked_coefficients(
    indices: Int64[Array, "p 3"],
    coefficients: Complex128[Array, " p"],
    name: str,
) -> Complex128[Array, " p"]:
    """PRIVATE: Attach finite and exact Hermitian-symmetry checks.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Sign-symmetric exact support for ``coefficients``.
    coefficients : Complex128[Array, " p"]
        Complex coefficient vector to validate.
    name : str
        Field name included in the runtime error.

    Returns
    -------
    checked : Complex128[Array, " p"]
        Coefficients with traced finite and Hermitian assertions.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the coefficients are non-finite or are not exactly Hermitian
        under compiled execution.
    """
    inverse_indices: Int64[Array, "p 3"] = -indices
    forward_order: Int64[Array, " p"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    inverse_order: Int64[Array, " p"] = jnp.lexsort(
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
    checked: Complex128[Array, " p"] = eqx.error_if(
        coefficients,
        jnp.any(~jnp.isfinite(coefficients)) | nonhermitian,
        f"{name} must be finite and exactly Hermitian",
    )
    return checked


def _derive_interaction_coefficients(
    voltage_kv: Float64[Array, ""],
    voltage_coefficients: Complex128[Array, " p"],
) -> Tuple[Float64[Array, ""], Complex128[Array, " p"]]:
    """PRIVATE: Derive the canonical coupling and SC-1 interaction.

    Parameters
    ----------
    voltage_kv : Float64[Array, ""]
        Electron accelerating voltage in kilovolts.
    voltage_coefficients : Complex128[Array, " p"]
        SC.13b electrostatic-potential coefficients in volts.

    Returns
    -------
    coupling : Float64[Array, ""]
        Canonically rounded interaction coupling.
    interaction : Complex128[Array, " p"]
        Finite SC-1 interaction coefficients in inverse-square Angstroms.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the coupling or interaction is invalid, or the conversion loses a
        nonzero normal voltage component under compiled execution.

    Notes
    -----
    The conversion uses the fixed physical constants and rounding contract
    implemented by
    :func:`ptyrodactyl._tools.coupled_interaction_value`.
    """
    raw_coupling, raw_interaction = coupled_interaction_value(
        voltage_coefficients,
        voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    coupling: Float64[Array, ""] = eqx.error_if(
        raw_coupling,
        (~jnp.isfinite(raw_coupling)) | (raw_coupling <= 0.0),
        "voltage-derived interaction coupling must be finite and positive",
    )
    interaction: Complex128[Array, " p"] = eqx.error_if(
        raw_interaction,
        jnp.any(~jnp.isfinite(raw_interaction))
        | (~jnp.isfinite(voltage_kv))
        | (voltage_kv <= 0.0)
        | has_subnormal_components(voltage_coefficients)
        | has_lost_nonzero_components(voltage_coefficients, raw_interaction),
        "voltage-derived interaction coefficients must be finite and "
        "preserve every nonzero normal voltage component",
    )
    result: Tuple[Float64[Array, ""], Complex128[Array, " p"]] = (
        coupling,
        interaction,
    )
    return result


def _derive_algebraic_wavenumber(
    voltage_kv: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Derive the canonical stored binary64 vacuum wavenumber.

    Parameters
    ----------
    voltage_kv : Float64[Array, ""]
        Positive accelerating voltage in kilovolts.

    Returns
    -------
    wavenumber : Float64[Array, ""]
        Canonical Planck-form angular wavenumber in inverse Angstroms.

    Notes
    -----
    This is the frozen algebraic geometry route enclosed by RM-S2. Exact
    SC.2 is defined separately with ``HBAR`` in that enclosure.
    """
    energy_joule: Float64[Array, ""] = (
        voltage_kv * 1000.0 * jnp.asarray(E_CHARGE)
    )
    wavelength_metre: Float64[Array, ""] = jnp.sqrt(
        (jnp.asarray(H_PLANCK) * jnp.asarray(C_LIGHT)) ** 2
        / (
            energy_joule
            * (
                2.0 * jnp.asarray(M_E) * jnp.asarray(C_LIGHT) ** 2
                + energy_joule
            )
        )
    )
    wavelength_angstrom: Float64[Array, ""] = 1.0e10 * wavelength_metre
    wavenumber: Float64[Array, ""] = 2.0 * jnp.pi / wavelength_angstrom
    return wavenumber


def _outward_nonnegative_add(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Add non-negative evidence with an FTZ-safe upper endpoint.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Non-negative left addend in the evidence quantity's physical units.
    right : Float64[Array, "..."]
        Non-negative right addend in the same physical units as ``left``.

    Returns
    -------
    result : Float64[Array, "..."]
        Outward sum in the inputs' shared physical units.

    Notes
    -----
    Exact stored inputs first enter the common FTZ-safe point embedding.
    The interval upper endpoint preserves a proved zero identity, widens any
    nonidentity underflow to a normal endpoint, and fails closed when the
    required normal binary64 arithmetic probes do not pass.
    """
    result: Float64[Array, "..."] = interval_add(
        point_interval(left),
        point_interval(right),
    )[1]
    return result


def _outward_nonnegative_multiply(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Multiply evidence with an FTZ-safe upper endpoint.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Non-negative left factor in its declared physical units.
    right : Float64[Array, "..."]
        Non-negative right factor in its declared physical units.

    Returns
    -------
    result : Float64[Array, "..."]
        Outward product in the product units of ``left`` and ``right``.

    Notes
    -----
    Exact stored inputs first enter the common FTZ-safe point embedding.
    The interval upper endpoint preserves a proved zero factor, widens any
    nonidentity underflow to a normal endpoint, and fails closed when the
    required normal binary64 arithmetic probes do not pass.
    """
    result: Float64[Array, "..."] = interval_multiply(
        point_interval(left),
        point_interval(right),
    )[1]
    return result


def _carrier_l1_error_upper(
    carrier_component_error_bounds: Float64[Array, " 3"],
) -> Float64[Array, ""]:
    """PRIVATE: Bound exact-to-algebraic carrier error in Euclidean norm.

    Parameters
    ----------
    carrier_component_error_bounds : Float64[Array, " 3"]
        Outward errors in radians per Angstrom, in Cartesian xyz order.

    Returns
    -------
    carrier_l1 : Float64[Array, ""]
        Outward L1 carrier-error bound in radians per Angstrom.

    Notes
    -----
    The L1 norm bounds the Euclidean norm. Successive outward additions keep
    the componentwise enclosure conservative in binary64 arithmetic.
    """
    carrier_l1: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    for axis in range(_SPACE_DIMENSIONS):
        carrier_l1 = _outward_nonnegative_add(
            carrier_l1,
            carrier_component_error_bounds[axis],
        )
    return carrier_l1


def _exact_target_full_offset_max(
    nominal_maximum: Float64[Array, ""],
    carrier_component_error_bounds: Float64[Array, " 3"],
    direction_dispositions: Int32[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Inflate projected full offsets for exact carrier scaling.

    Implementation Logic
    --------------------
    1. Bound the angular carrier correction by its outward L1 error.
    2. Convert that correction to cyclic units with a lower bound for two pi.
    3. Inflate the maximum only when the direction set contains a projection.

    Parameters
    ----------
    nominal_maximum : Float64[Array, ""]
        Outward nominal full cyclic-offset maximum in inverse Angstroms.
    carrier_component_error_bounds : Float64[Array, " 3"]
        Outward errors in radians per Angstrom, in Cartesian xyz order.
    direction_dispositions : Int32[Array, " n"]
        Dimensionless exact-or-projected codes for the direction rows.

    Returns
    -------
    corrected : Float64[Array, ""]
        Exact-target full cyclic-offset maximum in inverse Angstroms.

    Notes
    -----
    The cyclic correction is the outward L1 carrier error divided by a
    binary64 lower bound for two pi. The final addition is also rounded
    upward. Exact-only direction sets retain the submitted nominal maximum.
    """
    carrier_l1: Float64[Array, ""] = _carrier_l1_error_upper(
        carrier_component_error_bounds
    )
    correction_lower: Float64[Array, ""]
    correction: Float64[Array, ""]
    correction_lower, correction = interval_divide_positive(
        point_interval(carrier_l1),
        point_interval(
            jnp.asarray(_TWO_PI_LOWER, dtype=jnp.float64),
        ),
    )
    del correction_lower
    projected: Bool[Array, ""] = jnp.any(
        direction_dispositions == int(GalerkinDirectionDisposition.PROJECTED)
    )
    corrected: Float64[Array, ""] = jnp.where(
        projected,
        _outward_nonnegative_add(nominal_maximum, correction),
        nominal_maximum,
    )
    return corrected


def _exact_target_direction_error_bounds(
    nominal_shell_bounds: Float64[Array, " n"],
    nominal_projection_bounds: Float64[Array, " n"],
    direction_dispositions: Int32[Array, " n"],
    carrier_component_error_bounds: Float64[Array, " 3"],
    wavenumber_error_bound: Float64[Array, ""],
    algebraic_wavenumber: Float64[Array, ""],
) -> Tuple[Float64[Array, " n"], Float64[Array, " n"]]:
    """PRIVATE: Transfer nominal projected-direction evidence to SC-1.

    Implementation Logic
    --------------------
    1. Enclose carrier error by the outward componentwise L1 sum.
    2. Inflate shell defects by the wavenumber-square correction.
    3. Inflate projection errors by the carrier correction.
    4. Preserve symbolic zero for exact direction rows.

    Parameters
    ----------
    nominal_shell_bounds : Float64[Array, " n"]
        Outward nominal squared-shell defects in inverse-square Angstroms.
    nominal_projection_bounds : Float64[Array, " n"]
        Outward nominal projection errors in radians per Angstrom.
    direction_dispositions : Int32[Array, " n"]
        Dimensionless exact-or-projected codes for the direction rows.
    carrier_component_error_bounds : Float64[Array, " 3"]
        Outward errors in radians per Angstrom, in Cartesian xyz order.
    wavenumber_error_bound : Float64[Array, ""]
        Outward vacuum-wavenumber error in radians per Angstrom.
    algebraic_wavenumber : Float64[Array, ""]
        Positive stored vacuum wavenumber in radians per Angstrom.

    Returns
    -------
    exact_shell : Float64[Array, " n"]
        Exact-target squared-shell bounds in inverse-square Angstroms.
    exact_projection : Float64[Array, " n"]
        Exact-target projection-error bounds in radians per Angstrom.

    Notes
    -----
    For projected rows, the squared-shell correction is
    ``delta_k * (2 * k_alg + delta_k)`` and the projection correction is the
    outward L1 carrier error. Exact rows remain symbolic zero because their
    canonical binary64 round trip was checked separately.
    """
    carrier_l1: Float64[Array, ""] = _carrier_l1_error_upper(
        carrier_component_error_bounds
    )
    twice_wavenumber: Float64[Array, ""] = _outward_nonnegative_multiply(
        jnp.asarray(2.0, dtype=jnp.float64),
        algebraic_wavenumber,
    )
    wavenumber_sum_upper: Float64[Array, ""] = _outward_nonnegative_add(
        twice_wavenumber,
        wavenumber_error_bound,
    )
    squared_shell_correction: Float64[Array, ""] = (
        _outward_nonnegative_multiply(
            wavenumber_error_bound,
            wavenumber_sum_upper,
        )
    )
    projected_shell: Float64[Array, " n"] = _outward_nonnegative_add(
        nominal_shell_bounds,
        squared_shell_correction,
    )
    projected_projection: Float64[Array, " n"] = _outward_nonnegative_add(
        nominal_projection_bounds,
        carrier_l1,
    )
    projected: Bool[Array, " n"] = direction_dispositions == int(
        GalerkinDirectionDisposition.PROJECTED
    )
    exact_shell: Float64[Array, " n"] = jnp.where(
        projected,
        projected_shell,
        0.0,
    )
    exact_projection: Float64[Array, " n"] = jnp.where(
        projected,
        projected_projection,
        0.0,
    )
    result: Tuple[Float64[Array, " n"], Float64[Array, " n"]] = (
        exact_shell,
        exact_projection,
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
        Exact arithmetic did not prove a positive absorber floor.
    NO_FINITE_EXACT_TARGET_RESIDUAL_BOUND : str
        RM-S2 supplied no finite ``delta_H`` for a nonzero submitted field.
    NO_FINITE_EXACT_TARGET_SOURCE_ERROR_BOUND : str
        RM-S3 supplied no finite exact-target total-source error bound.
    SOURCE_NOT_RM_S3_ELIGIBLE : str
        The represented source did not earn the narrow RM-S3 eligibility gate.
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
    NO_FINITE_EXACT_TARGET_RESIDUAL_BOUND = (
        "no_finite_exact_target_residual_bound"
    )
    NO_FINITE_EXACT_TARGET_SOURCE_ERROR_BOUND = (
        "no_finite_exact_target_source_error_bound"
    )
    PROOF_RECORD_MISMATCH = "proof_record_mismatch"
    SOURCE_NOT_RM_S3_ELIGIBLE = "source_not_rm_s3_eligible"
    STATE_BUDGET_MISSED = "state_budget_missed"


class GalerkinStabilityRoute(str, Enum):
    """Store the checked singular-value certificate route.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    ABSORBER_FLOOR : str
        Generic legacy Route-A absorber floor.
    ABSORBER_FLOOR_GERSHGORIN : str
        Exact-dyadic Route A whose selected floor is Gershgorin.
    ABSORBER_FLOOR_COSINE_BOX : str
        Exact-rational Route A using the analytic cosine-shell box floor.
    """

    ABSORBER_FLOOR = "absorber_floor"
    ABSORBER_FLOOR_GERSHGORIN = "absorber_floor_gershgorin"
    ABSORBER_FLOOR_COSINE_BOX = "absorber_floor_cosine_box"


class GalerkinTargetManifest(eqx.Module):
    """Store one canonical SC-1 finite target manifest.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    realization : GalerkinPotentialRealization
        Bound ``Potential3D`` and independently checked VC-1 realization.
    fixed_linear_error_ledger : GalerkinFixedLinearErrorLedger
        RM-S2 enclosure whose algebraic diagonal defines this target.
    interaction_coefficients : Complex128[Array, " p"]
        Voltage-derived SC.13b interaction coefficients in inverse-square
        Angstroms.
    interaction_coupling : Float64[Array, ""]
        Voltage-derived Helmholtz coupling in inverse-square Angstroms per
        volt.
    absorber_coefficients : Complex128[Array, " q"]
        Factory-derived exact SC.13b coefficients of the dimensionless
        analytic cosine-shell profile.
    exact_target_incident_full_offset_max : Float64[Array, ""]
        Outward S1.16 full incident cyclic-offset maximum after exact carrier
        normalization, with inflation applied only to projected directions.
    exact_target_outgoing_full_offset_max : Float64[Array, ""]
        Outward S1.16 full outgoing cyclic-offset maximum after exact carrier
        normalization, with inflation applied only to projected directions.
    exact_target_incident_shell_defect_bounds : Float64[Array, " i"]
        Outward projected-direction shell defects relative to exact SC.2;
        exact coefficient rows remain symbolic zero.
    exact_target_outgoing_shell_defect_bounds : Float64[Array, " o"]
        Outward outgoing shell defects relative to exact SC.2.
    exact_target_incident_projection_error_bounds : Float64[Array, " i"]
        Outward requested-to-exact-coefficient incident discrepancies.
    exact_target_outgoing_projection_error_bounds : Float64[Array, " o"]
        Outward requested-to-exact-coefficient outgoing discrepancies.
    accelerating_voltage_kv : Float64[Array, ""]
        Positive accelerating voltage in kilovolts.
    cap_scale : Float64[Array, ""]
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
    :func:`ptyrodactyl.galerkin.create_galerkin_target`
        Construct this target through the differentiable production route.
    :func:`ptyrodactyl.galerkin.create_host_checked_galerkin_target`
        Construct this target after a direct host-certificate attempt.
    """

    realization: GalerkinPotentialRealization
    fixed_linear_error_ledger: GalerkinFixedLinearErrorLedger
    interaction_coefficients: Complex128[Array, " p"]
    interaction_coupling: Float64[Array, ""]
    absorber_coefficients: Complex128[Array, " q"]
    exact_target_incident_full_offset_max: Float64[Array, ""]
    exact_target_outgoing_full_offset_max: Float64[Array, ""]
    exact_target_incident_shell_defect_bounds: Float64[Array, " i"]
    exact_target_outgoing_shell_defect_bounds: Float64[Array, " o"]
    exact_target_incident_projection_error_bounds: Float64[Array, " i"]
    exact_target_outgoing_projection_error_bounds: Float64[Array, " o"]
    accelerating_voltage_kv: Float64[Array, ""]
    cap_scale: Float64[Array, ""]
    target_name: str = eqx.field(static=True)
    contract_version: str = eqx.field(static=True)
    coefficient_normalization: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    absorber_profile: str = eqx.field(static=True)
    absorber_coefficient_provenance: str = eqx.field(static=True)
    interaction_coefficient_provenance: str = eqx.field(static=True)

    @property
    def support(self) -> GalerkinProductSupport:
        """Return the independently checked product support."""
        support: GalerkinProductSupport = self.realization.support
        return support

    @property
    def support_eligibility(self) -> GalerkinAcquisitionSupportResult:
        """Return the complete checked acquisition-support result."""
        eligibility: GalerkinAcquisitionSupportResult = (
            self.realization.support_eligibility
        )
        return eligibility

    @property
    def acquisition(self) -> GalerkinAcquisitionManifest:
        """Return the checked acquisition manifest without copying it."""
        acquisition: GalerkinAcquisitionManifest = (
            self.realization.support_eligibility.manifest
        )
        return acquisition

    @property
    def preterminal_indices(self) -> Int64[Array, "m 3"]:
        """Return the checked state-side preterminal without copying it."""
        indices: Int64[Array, "m 3"] = self.acquisition.preterminal_indices
        return indices

    @property
    def voltage_coefficients(self) -> Complex128[Array, " p"]:
        """Return the VC-1 voltage coefficients without copying them."""
        coefficients: Complex128[Array, " p"] = (
            self.realization.voltage_coefficients
        )
        return coefficients

    @property
    def free_diagonal(self) -> Float64[Array, " n"]:
        """Return the sole RM-S2-certified algebraic free diagonal."""
        diagonal: Float64[Array, " n"] = (
            self.fixed_linear_error_ledger.algebraic_free_diagonal
        )
        return diagonal

    @property
    def carrier(self) -> Float64[Array, " 3"]:
        """Return the checked algebraic carrier-direction seed."""
        carrier: Float64[Array, " 3"] = self.acquisition.carrier
        return carrier

    @property
    def box_lengths(self) -> Float64[Array, " 3"]:
        """Return the checked acquisition box lengths."""
        box_lengths: Float64[Array, " 3"] = self.acquisition.box_lengths
        return box_lengths

    @property
    def wavenumber(self) -> Float64[Array, ""]:
        """Return the checked nominal algebraic wavenumber."""
        wavenumber: Float64[Array, ""] = self.acquisition.wavenumber
        return wavenumber

    @property
    def potential(self) -> Potential3D:
        """Return the bound voxel potential."""
        potential: Potential3D = self.realization.potential
        return potential

    @property
    def incident_full_offset_max(self) -> Float64[Array, ""]:
        """Return exact-target, not nominal-carrier, incident evidence."""
        maximum: Float64[Array, ""] = (
            self.exact_target_incident_full_offset_max
        )
        return maximum

    @property
    def outgoing_full_offset_max(self) -> Float64[Array, ""]:
        """Return exact-target, not nominal-carrier, outgoing evidence."""
        maximum: Float64[Array, ""] = (
            self.exact_target_outgoing_full_offset_max
        )
        return maximum

    @property
    def incident_transverse_offset_max(self) -> Float64[Array, ""]:
        """Return the carrier-normalization-invariant incident maximum."""
        maximum: Float64[Array, ""] = (
            self.support_eligibility.incident_transverse_offset_max
        )
        return maximum

    @property
    def outgoing_transverse_offset_max(self) -> Float64[Array, ""]:
        """Return the carrier-normalization-invariant outgoing maximum."""
        maximum: Float64[Array, ""] = (
            self.support_eligibility.outgoing_transverse_offset_max
        )
        return maximum

    @property
    def transfer_transverse_max(self) -> Float64[Array, ""]:
        """Return the carrier-cancelling transverse transfer maximum."""
        maximum: Float64[Array, ""] = (
            self.support_eligibility.transfer_transverse_max
        )
        return maximum

    @property
    def transfer_full_max(self) -> Float64[Array, ""]:
        """Return the carrier-cancelling full transfer maximum."""
        maximum: Float64[Array, ""] = (
            self.support_eligibility.transfer_full_max
        )
        return maximum


class GalerkinSource(eqx.Module):
    """Store one finite matched-source realization.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    incident_field : Complex128[Array, " n"]
        Declared finite incident-vector coefficients.
    incident_source : Complex128[Array, " n"]
        Rounded finite matched source ``(D - i B) incident_field``.
    additional_source : Complex128[Array, " n"]
        Separately declared finite source beyond matched injection.
    total_source : Complex128[Array, " n"]
        Complete total-field right-hand side.
    scattered_source : Complex128[Array, " n"]
        Equivalent scattered-field right-hand side.
    branch : GalerkinSourceBranch
        Static finite source branch. This value affects tracing.

    See Also
    --------
    :func:`create_galerkin_source`
        Validate one finite matched-source decomposition.
    """

    incident_field: Complex128[Array, " n"]
    incident_source: Complex128[Array, " n"]
    additional_source: Complex128[Array, " n"]
    total_source: Complex128[Array, " n"]
    scattered_source: Complex128[Array, " n"]
    branch: GalerkinSourceBranch = eqx.field(static=True)


class GalerkinPhysicalResidual(eqx.Module):
    """Store one independently recomputed physical residual.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    residual : Complex128[Array, " n"]
        Original-system residual from the direct coefficient action.
    residual_norm : Float64[Array, ""]
        Euclidean norm of the independently recomputed residual.

    See Also
    --------
    :func:`create_galerkin_physical_residual`
        Validate a physical residual and its norm.
    """

    residual: Complex128[Array, " n"]
    residual_norm: Float64[Array, ""]


class GalerkinStabilityProof(eqx.Module):
    """Store one checker-produced exact Route-A proof payload.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    target_digest : str
        Static SHA-256 checksum of the canonical target manifest.
    result_digest : str
        Static SHA-256 checksum of the bound source and submitted result.
    algebraic_floor_numerator : int
        Static exact numerator of the ``H_alg`` Route-A floor.
    algebraic_floor_denominator : int
        Static positive denominator of the ``H_alg`` Route-A floor.
    transferred_floor_numerator : int
        Static signed numerator of ``s_alg - delta_H`` when finite, else zero.
    transferred_floor_denominator : int
        Static positive denominator of the perturbative transfer margin.
    transferred_floor_finite : bool
        Whether the manifested ``delta_H`` gives a finite transfer margin.
    floor_numerator : int
        Static exact selected direct exact-target lower-bound numerator.
    floor_denominator : int
        Static positive exact selected lower-bound denominator.
    residual_squared_numerator : int
        Static exact independently reconstructed ``H_alg`` residual-square
        numerator.
    residual_squared_denominator : int
        Static positive ``H_alg`` residual-square denominator.
    field_norm_squared_numerator : int
        Static exact submitted-field squared-norm numerator.
    field_norm_squared_denominator : int
        Static positive submitted-field squared-norm denominator.
    exact_target_residual_upper_numerator : int
        Static numerator of the directed-up exact-target residual bound.
    exact_target_residual_upper_denominator : int
        Static positive exact-target residual-bound denominator.
    exact_target_residual_finite : bool
        Whether the exact-target residual lift has a finite binary64 bound.
    source_error_upper_numerator : int
        Static numerator of the bound exact-target source error ``delta_S``.
    source_error_upper_denominator : int
        Static positive denominator of the bound source error.
    source_error_finite : bool
        Whether ``delta_S`` is finite for this invocation.
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
    rhs_target : str
        Static declaration of the exact right-hand-side target.
    residual_scope : str
        Static residual lift and exclusion declaration.
    source_error_route : str
        Static source-error certification route or legacy exact-RHS route.
    source_error_scope : str
        Static declaration of the source-error comparison and exclusions.

    See Also
    --------
    :func:`create_galerkin_stability_proof`
        Validate one exact proof payload.

    Notes
    -----
    The selected floor is the direct exact-target Route-A floor. Canonical
    reconstruction proves an exact Hermitian free/interaction part and the
    identical exact dyadic cosine CAP; the algebraic number is therefore
    independently rederived for the exact target rather than copied. The
    separately stored ``H_alg - delta_H`` margin records the perturbative
    route even when it is weaker or noncertifying.
    """

    target_digest: str = eqx.field(static=True)
    result_digest: str = eqx.field(static=True)
    algebraic_floor_numerator: int = eqx.field(static=True)
    algebraic_floor_denominator: int = eqx.field(static=True)
    transferred_floor_numerator: int = eqx.field(static=True)
    transferred_floor_denominator: int = eqx.field(static=True)
    transferred_floor_finite: bool = eqx.field(static=True)
    floor_numerator: int = eqx.field(static=True)
    floor_denominator: int = eqx.field(static=True)
    residual_squared_numerator: int = eqx.field(static=True)
    residual_squared_denominator: int = eqx.field(static=True)
    field_norm_squared_numerator: int = eqx.field(static=True)
    field_norm_squared_denominator: int = eqx.field(static=True)
    exact_target_residual_upper_numerator: int = eqx.field(static=True)
    exact_target_residual_upper_denominator: int = eqx.field(static=True)
    exact_target_residual_finite: bool = eqx.field(static=True)
    source_error_upper_numerator: int = eqx.field(static=True)
    source_error_upper_denominator: int = eqx.field(static=True)
    source_error_finite: bool = eqx.field(static=True)
    state_budget_numerator: int = eqx.field(static=True)
    state_budget_denominator: int = eqx.field(static=True)
    route: GalerkinStabilityRoute = eqx.field(static=True)
    failure: GalerkinStabilityFailure = eqx.field(static=True)
    checker_id: str = eqx.field(static=True)
    rhs_target: str = eqx.field(static=True)
    residual_scope: str = eqx.field(static=True)
    source_error_route: str = eqx.field(static=True)
    source_error_scope: str = eqx.field(static=True)


class GalerkinStabilityResult(eqx.Module):
    """Store one per-result operational stability invocation.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Attributes
    ----------
    lower_singular_bound : Float64[Array, ""]
        Downward-rounded lower singular-value bound.
    residual_upper_bound : Float64[Array, ""]
        Upward-rounded exact-target residual-norm enclosure for the stored
        exact right-hand side.
    state_error_upper_bound : Float64[Array, ""]
        Upward-rounded exact-target residual-to-stability state bound.
    state_budget : Float64[Array, ""]
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

    lower_singular_bound: Float64[Array, ""]
    residual_upper_bound: Float64[Array, ""]
    state_error_upper_bound: Float64[Array, ""]
    state_budget: Float64[Array, ""]
    route: GalerkinStabilityRoute = eqx.field(static=True)
    disposition: GalerkinStabilityDisposition = eqx.field(static=True)
    failure: GalerkinStabilityFailure = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    result_digest: str = eqx.field(static=True)
    checker_id: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def _create_galerkin_target_manifest(
    realization: GalerkinPotentialRealization,
    fixed_linear_error_ledger: GalerkinFixedLinearErrorLedger,
    accelerating_voltage_kv: scalar_float,
    cap_scale: scalar_float,
    target_name: str,
) -> GalerkinTargetManifest:
    r"""PRIVATE: Store one target from system-constructed nested evidence.

    Implementation Logic
    --------------------
    1. Bind one checked VC-1 realization and its RM-S2 ledger.
    2. Derive the interaction and analytic absorber with their provenance.
    3. Validate exact box and canonical algebraic-wavenumber consistency.
    4. Require a correctly shaped fixed-linear ledger carrier and route.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Checked acquisition-bound VC-1 potential realization.
    fixed_linear_error_ledger : GalerkinFixedLinearErrorLedger
        RM-S2 ledger built from this realization and the physical scalars.
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
    The raw support/coefficient/carrier/box/wavenumber construction path no
    longer exists. Those values are exposed as read-only properties of the
    nested checked artifacts. The helper accepts neither pre-coupled
    interaction coefficients nor arbitrary absorber coefficients. It is
    deliberately private: only :mod:`ptyrodactyl.galerkin.system` supplies the
    realization and ledger, and certifying consumers rebuild that public
    constructor before trusting the target.
    """
    _raise_if(not target_name.strip(), "target_name must be nonempty")
    support: GalerkinProductSupport = realization.support
    voltage_coefficients: Complex128[Array, " p"] = (
        realization.voltage_coefficients
    )
    absorber_array: Complex128[Array, " q"] = _cosine_shell_coefficients(
        support.absorber_indices
    )
    voltage_array: Float64[Array, ""] = jnp.asarray(
        accelerating_voltage_kv, dtype=jnp.float64
    )
    cap_array: Float64[Array, ""] = jnp.asarray(cap_scale, dtype=jnp.float64)

    _raise_if(
        voltage_coefficients.ndim != 1,
        "voltage_coefficients must be 1D",
    )
    _raise_if(
        voltage_coefficients.shape[0] != support.interaction_indices.shape[0],
        "voltage_coefficients must match the interaction support",
    )
    _raise_if(
        fixed_linear_error_ledger.algebraic_free_diagonal.shape
        != (support.state_indices.shape[0],),
        "fixed-linear free diagonal must match the state support",
    )
    _raise_if(
        fixed_linear_error_ledger.interaction_coefficient_error_bounds.shape
        != voltage_coefficients.shape,
        "fixed-linear interaction errors must match voltage coefficients",
    )
    _raise_if(
        fixed_linear_error_ledger.difference_multiplicities.shape
        != voltage_coefficients.shape,
        "fixed-linear multiplicities must match voltage coefficients",
    )
    _raise_if(
        fixed_linear_error_ledger.interaction_row_error_bounds.shape
        != (support.state_indices.shape[0],)
        or fixed_linear_error_ledger.interaction_column_error_bounds.shape
        != (support.state_indices.shape[0],),
        "fixed-linear row and column errors must match the state support",
    )
    expected_absorber_route: GalerkinFixedLinearAbsorberRoute = (
        GalerkinFixedLinearAbsorberRoute.ANALYTIC_COSINE_SHELL_EXACT_DYADIC
    )
    _raise_if(
        fixed_linear_error_ledger.absorber_route
        is not expected_absorber_route,
        "fixed-linear ledger must use the exact analytic cosine-shell route",
    )
    for values, name in (
        (voltage_array, "accelerating_voltage_kv"),
        (cap_array, "cap_scale"),
    ):
        _raise_if(values.shape != (), f"{name} must be a scalar")

    checked_voltage_coefficients: Complex128[Array, " p"] = (
        _checked_coefficients(
            support.interaction_indices,
            voltage_coefficients,
            "voltage_coefficients",
        )
    )
    checked_absorber: Complex128[Array, " q"] = _checked_coefficients(
        support.absorber_indices, absorber_array, "absorber_coefficients"
    )
    checked_absorber = eqx.error_if(
        checked_absorber,
        ~_has_complete_cosine_shell_support(
            support.absorber_indices, support.work_shape
        ),
        "absorber support must contain all cosine-shell profile modes",
    )
    checked_voltage: Float64[Array, ""] = eqx.error_if(
        voltage_array,
        (~jnp.isfinite(voltage_array)) | (voltage_array <= 0.0),
        "accelerating_voltage_kv must be finite and positive",
    )
    acquisition = realization.support_eligibility
    acquisition_manifest = acquisition.manifest
    potential_box: Float64[Array, " 3"] = jnp.asarray(
        realization.potential.box_size,
        dtype=jnp.float64,
    )
    canonical_wavenumber: Float64[Array, ""] = _derive_algebraic_wavenumber(
        checked_voltage
    )
    nested_contract_invalid: Bool[Array, ""] = (
        (
            acquisition.status
            != int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
        )
        | (~acquisition.support_eligible)
        | jnp.any(acquisition_manifest.box_lengths != potential_box)
        | (acquisition_manifest.wavenumber != canonical_wavenumber)
    )
    checked_voltage_coefficients = eqx.error_if(
        checked_voltage_coefficients,
        nested_contract_invalid,
        "nested realization must bind eligible support, exact Potential3D "
        "box lengths, and the canonical voltage-derived wavenumber",
    )
    interaction_coupling, checked_interaction = (
        _derive_interaction_coefficients(
            checked_voltage,
            checked_voltage_coefficients,
        )
    )
    checked_cap: Float64[Array, ""] = eqx.error_if(
        cap_array,
        (~jnp.isfinite(cap_array)) | (cap_array < _MIN_CAP_SCALE),
        "cap_scale must be finite and preserve every nonzero analytic "
        "absorber coefficient in normal-range arithmetic",
    )
    ledger_contract_invalid: Bool[Array, ""] = (
        (fixed_linear_error_ledger.absorber_operator_error_bound != 0.0)
        | (fixed_linear_error_ledger.cap_scale_error_bound != 0.0)
        | (fixed_linear_error_ledger.cap_operator_error_bound != 0.0)
    )
    normal_error: Float64[Array, ""] = (
        fixed_linear_error_ledger.carrier_component_error_bounds[
            acquisition_manifest.terminal_axis
        ]
    )
    exact_incident_shell: Float64[Array, " i"]
    exact_incident_projection: Float64[Array, " i"]
    exact_incident_shell, exact_incident_projection = (
        _exact_target_direction_error_bounds(
            acquisition.incident_shell_defect_upper_bounds,
            acquisition.incident_projection_error_upper_bounds,
            acquisition_manifest.incident_direction_dispositions,
            fixed_linear_error_ledger.carrier_component_error_bounds,
            fixed_linear_error_ledger.wavenumber_error_bound,
            acquisition_manifest.wavenumber,
        )
    )
    exact_outgoing_shell: Float64[Array, " o"]
    exact_outgoing_projection: Float64[Array, " o"]
    exact_outgoing_shell, exact_outgoing_projection = (
        _exact_target_direction_error_bounds(
            acquisition.outgoing_shell_defect_upper_bounds,
            acquisition.outgoing_projection_error_upper_bounds,
            acquisition_manifest.outgoing_direction_dispositions,
            fixed_linear_error_ledger.carrier_component_error_bounds,
            fixed_linear_error_ledger.wavenumber_error_bound,
            acquisition_manifest.wavenumber,
        )
    )
    exact_direction_evidence_valid: Bool[Array, ""] = (
        jnp.all(
            exact_incident_shell
            <= acquisition_manifest.incident_on_shell_defect_bounds
        )
        & jnp.all(
            exact_outgoing_shell
            <= acquisition_manifest.outgoing_on_shell_defect_bounds
        )
        & jnp.all(
            exact_incident_projection
            <= acquisition_manifest.incident_projection_error_bounds
        )
        & jnp.all(
            exact_outgoing_projection
            <= acquisition_manifest.outgoing_projection_error_bounds
        )
    )
    state_sector_preserved: Bool[Array, ""] = jnp.all(
        (~acquisition.state_forward_mask)
        | (acquisition.state_oriented_normal_interval_lower > normal_error)
    ) & jnp.all(
        (~acquisition.state_backward_mask)
        | (acquisition.state_oriented_normal_interval_upper < -normal_error)
    )
    state_grazing_preserved: Bool[Array, ""] = jnp.all(
        (~acquisition.state_grazing_mask)
        | (
            (acquisition.state_oriented_normal_interval_lower == 0.0)
            & (acquisition.state_oriented_normal_interval_upper == 0.0)
            & (normal_error == 0.0)
        )
    )
    omitted_sector_preserved: Bool[Array, ""] = jnp.all(
        (~acquisition.omitted_forward_mask)
        | (acquisition.omitted_oriented_normal_interval_lower > normal_error)
    ) & jnp.all(
        (~acquisition.omitted_backward_mask)
        | (acquisition.omitted_oriented_normal_interval_upper < -normal_error)
    )
    omitted_grazing_preserved: Bool[Array, ""] = jnp.all(
        (~acquisition.omitted_grazing_mask)
        | (
            (acquisition.omitted_oriented_normal_interval_lower == 0.0)
            & (acquisition.omitted_oriented_normal_interval_upper == 0.0)
            & (normal_error == 0.0)
        )
    )
    sector_preserved: Bool[Array, ""] = (
        state_sector_preserved
        & state_grazing_preserved
        & omitted_sector_preserved
        & omitted_grazing_preserved
    )
    ledger_contract_invalid = ledger_contract_invalid | (
        (~jnp.isfinite(normal_error))
        | (~sector_preserved)
        | (~exact_direction_evidence_valid)
    )
    checked_interaction = eqx.error_if(
        checked_interaction,
        ledger_contract_invalid,
        "fixed_linear_error_ledger must use the exact analytic absorber/CAP "
        "route and preserve every acquisition sector, shell ceiling, and "
        "projection ceiling under exact carrier normalization",
    )
    exact_incident_full_max: Float64[Array, ""] = (
        _exact_target_full_offset_max(
            acquisition.incident_full_offset_max,
            fixed_linear_error_ledger.carrier_component_error_bounds,
            acquisition_manifest.incident_direction_dispositions,
        )
    )
    exact_outgoing_full_max: Float64[Array, ""] = (
        _exact_target_full_offset_max(
            acquisition.outgoing_full_offset_max,
            fixed_linear_error_ledger.carrier_component_error_bounds,
            acquisition_manifest.outgoing_direction_dispositions,
        )
    )
    manifest: GalerkinTargetManifest = GalerkinTargetManifest(
        realization=realization,
        fixed_linear_error_ledger=fixed_linear_error_ledger,
        interaction_coefficients=checked_interaction,
        interaction_coupling=interaction_coupling,
        absorber_coefficients=checked_absorber,
        exact_target_incident_full_offset_max=exact_incident_full_max,
        exact_target_outgoing_full_offset_max=exact_outgoing_full_max,
        exact_target_incident_shell_defect_bounds=exact_incident_shell,
        exact_target_outgoing_shell_defect_bounds=exact_outgoing_shell,
        exact_target_incident_projection_error_bounds=(
            exact_incident_projection
        ),
        exact_target_outgoing_projection_error_bounds=(
            exact_outgoing_projection
        ),
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
    exclusively from :func:`ptyrodactyl.galerkin.\
create_matched_galerkin_source`.
    """
    checked_branch: GalerkinSourceBranch = GalerkinSourceBranch(branch)
    arrays: Tuple[Complex128[Array, "..."], ...] = tuple(
        jnp.asarray(value, dtype=jnp.complex128)
        for value in (
            incident_field,
            incident_source,
            additional_source,
            total_source,
            scattered_source,
        )
    )
    reference_shape: Tuple[int, ...] = arrays[0].shape
    _raise_if(len(reference_shape) != 1, "source vectors must be 1D")
    _raise_if(reference_shape[0] == 0, "source vectors must be nonempty")
    _raise_if(
        any(values.shape != reference_shape for values in arrays[1:]),
        "source vectors must have matching shapes",
    )
    checked_arrays: Tuple[Complex128[Array, " n"], ...] = tuple(
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
    residual_array: Complex128[Array, " n"] = jnp.asarray(
        residual, dtype=jnp.complex128
    )
    norm_array: Float64[Array, ""] = jnp.asarray(
        residual_norm, dtype=jnp.float64
    )
    _raise_if(residual_array.ndim != 1, "residual must be 1D")
    _raise_if(residual_array.shape[0] == 0, "residual must be nonempty")
    _raise_if(norm_array.shape != (), "residual_norm must be a scalar")
    checked_residual: Complex128[Array, " n"] = eqx.error_if(
        residual_array,
        jnp.any(~jnp.isfinite(residual_array))
        | has_subnormal_components(residual_array),
        "residual must be finite and contain no nonzero subnormal components",
    )
    checked_norm: Float64[Array, ""] = eqx.error_if(
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
    algebraic_floor_numerator: int,
    algebraic_floor_denominator: int,
    transferred_floor_numerator: int,
    transferred_floor_denominator: int,
    transferred_floor_finite: bool,
    floor_numerator: int,
    floor_denominator: int,
    residual_squared_numerator: int,
    residual_squared_denominator: int,
    field_norm_squared_numerator: int,
    field_norm_squared_denominator: int,
    exact_target_residual_upper_numerator: int,
    exact_target_residual_upper_denominator: int,
    exact_target_residual_finite: bool,
    source_error_upper_numerator: int,
    source_error_upper_denominator: int,
    source_error_finite: bool,
    state_budget_numerator: int,
    state_budget_denominator: int,
    route: GalerkinStabilityRoute | str,
    failure: GalerkinStabilityFailure | str,
    checker_id: str,
    rhs_target: str,
    residual_scope: str,
    source_error_route: str,
    source_error_scope: str,
) -> GalerkinStabilityProof:
    """Create a structurally validated exact stability proof payload.

    :see: :class:`~.test_galerkin_types.TestGalerkinProductionCarriers`

    Parameters
    ----------
    target_digest : str
        Nonempty canonical target checksum.
    result_digest : str
        Nonempty canonical bound-result checksum.
    algebraic_floor_numerator : int
        Non-negative exact ``H_alg`` Route-A floor numerator.
    algebraic_floor_denominator : int
        Positive exact ``H_alg`` Route-A floor denominator.
    transferred_floor_numerator : int
        Signed exact perturbative transfer-margin numerator when finite.
    transferred_floor_denominator : int
        Positive perturbative transfer-margin denominator.
    transferred_floor_finite : bool
        Whether the perturbative transfer margin is finite.
    floor_numerator : int
        Non-negative exact floor numerator.
    floor_denominator : int
        Positive exact floor denominator.
    residual_squared_numerator : int
        Non-negative exact squared-residual numerator.
    residual_squared_denominator : int
        Positive exact ``H_alg`` squared-residual denominator.
    field_norm_squared_numerator : int
        Non-negative exact submitted-field squared-norm numerator.
    field_norm_squared_denominator : int
        Positive exact submitted-field squared-norm denominator.
    exact_target_residual_upper_numerator : int
        Non-negative directed-up exact-target residual-bound numerator.
    exact_target_residual_upper_denominator : int
        Positive exact-target residual-bound denominator.
    exact_target_residual_finite : bool
        Whether the exact-target residual lift is finite.
    source_error_upper_numerator : int
        Non-negative exact-target source-error numerator.
    source_error_upper_denominator : int
        Positive exact-target source-error denominator.
    source_error_finite : bool
        Whether the source-error bound is finite.
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
    rhs_target : str
        Nonempty exact right-hand-side target declaration.
    residual_scope : str
        Nonempty residual-lift and exclusion declaration.
    source_error_route : str
        Nonempty source-error certification or exact-RHS route.
    source_error_scope : str
        Nonempty source-error comparison and exclusion declaration.

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
    _raise_if(not rhs_target, "rhs_target must be nonempty")
    _raise_if(not residual_scope, "residual_scope must be nonempty")
    _raise_if(not source_error_route, "source_error_route must be nonempty")
    _raise_if(not source_error_scope, "source_error_scope must be nonempty")
    _raise_if(
        not isinstance(transferred_floor_finite, bool),
        "transferred_floor_finite must be boolean",
    )
    _raise_if(
        not isinstance(exact_target_residual_finite, bool),
        "exact_target_residual_finite must be boolean",
    )
    _raise_if(
        not isinstance(source_error_finite, bool),
        "source_error_finite must be boolean",
    )
    for value, name in (
        (algebraic_floor_numerator, "algebraic_floor_numerator"),
        (floor_numerator, "floor_numerator"),
        (residual_squared_numerator, "residual_squared_numerator"),
        (field_norm_squared_numerator, "field_norm_squared_numerator"),
        (
            exact_target_residual_upper_numerator,
            "exact_target_residual_upper_numerator",
        ),
        (source_error_upper_numerator, "source_error_upper_numerator"),
    ):
        _raise_if(
            isinstance(value, bool) or value < 0,
            f"{name} must be non-negative",
        )
    for value, name in (
        (algebraic_floor_denominator, "algebraic_floor_denominator"),
        (transferred_floor_denominator, "transferred_floor_denominator"),
        (floor_denominator, "floor_denominator"),
        (residual_squared_denominator, "residual_squared_denominator"),
        (field_norm_squared_denominator, "field_norm_squared_denominator"),
        (
            exact_target_residual_upper_denominator,
            "exact_target_residual_upper_denominator",
        ),
        (source_error_upper_denominator, "source_error_upper_denominator"),
        (state_budget_numerator, "state_budget_numerator"),
        (state_budget_denominator, "state_budget_denominator"),
    ):
        _raise_if(
            isinstance(value, bool) or value <= 0, f"{name} must be positive"
        )
    _raise_if(
        isinstance(transferred_floor_numerator, bool),
        "transferred_floor_numerator must be an integer",
    )
    _raise_if(
        not transferred_floor_finite and transferred_floor_numerator != 0,
        "a non-finite transfer margin must use numerator zero",
    )
    _raise_if(
        not exact_target_residual_finite
        and exact_target_residual_upper_numerator != 0,
        "a non-finite exact-target residual must use numerator zero",
    )
    _raise_if(
        not source_error_finite and source_error_upper_numerator != 0,
        "a non-finite source error must use numerator zero",
    )
    proof: GalerkinStabilityProof = GalerkinStabilityProof(
        target_digest=target_digest,
        result_digest=result_digest,
        algebraic_floor_numerator=algebraic_floor_numerator,
        algebraic_floor_denominator=algebraic_floor_denominator,
        transferred_floor_numerator=transferred_floor_numerator,
        transferred_floor_denominator=transferred_floor_denominator,
        transferred_floor_finite=transferred_floor_finite,
        floor_numerator=floor_numerator,
        floor_denominator=floor_denominator,
        residual_squared_numerator=residual_squared_numerator,
        residual_squared_denominator=residual_squared_denominator,
        field_norm_squared_numerator=field_norm_squared_numerator,
        field_norm_squared_denominator=field_norm_squared_denominator,
        exact_target_residual_upper_numerator=(
            exact_target_residual_upper_numerator
        ),
        exact_target_residual_upper_denominator=(
            exact_target_residual_upper_denominator
        ),
        exact_target_residual_finite=exact_target_residual_finite,
        source_error_upper_numerator=source_error_upper_numerator,
        source_error_upper_denominator=source_error_upper_denominator,
        source_error_finite=source_error_finite,
        state_budget_numerator=state_budget_numerator,
        state_budget_denominator=state_budget_denominator,
        route=checked_route,
        failure=checked_failure,
        checker_id=checker_id,
        rhs_target=rhs_target,
        residual_scope=residual_scope,
        source_error_route=source_error_route,
        source_error_scope=source_error_scope,
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
    values: Tuple[Float64[Array, ""], ...] = tuple(
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
]
