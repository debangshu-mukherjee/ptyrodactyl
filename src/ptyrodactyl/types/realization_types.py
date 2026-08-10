r"""Define evidence for one voxel-to-Galerkin potential realization.

Extended Summary
----------------
This module owns the carrier that binds a canonical :class:`Potential3D` to
one independently checked Galerkin acquisition-support artifact. The carrier
keeps coefficient rounding separate from potential-band truncation and binds
the exact source volume, support evidence, and metadata.

Routine Listings
----------------
:class:`GalerkinPotentialCertificateFailure`
    Store the outcome of one host coefficient-certificate attempt.
:class:`GalerkinPotentialCoefficientCertificate`
    Store exact-coefficient rectangles produced by one host checker.
:class:`GalerkinPotentialErrorRoute`
    Store the outward coefficient-error route.
:class:`GalerkinPotentialRealization`
    Store one VC-1 voxel-to-coefficient realization.
:class:`GalerkinPotentialRealizationMethod`
    Store the exact finite-target realization method.
:func:`create_galerkin_potential_coefficient_certificate`
    Create a structurally validated host coefficient certificate.
:func:`create_galerkin_potential_realization`
    Create a validated voxel-to-Galerkin realization carrier.

Notes
-----
The governing mathematical contract is VC-1 in
``ptyrodactyl-plans/physics/scalar_voxel_realization.md``. A finite error
bound can be sound but too large for a useful RM-S2 perturbation margin.
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
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from .acquisition_types import (
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
)
from .born_potential_types import GalerkinProductSupport
from .local_cell_types import GalerkinVoxelTargetRoute
from .potential_types import Potential3D


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

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


class GalerkinPotentialErrorRoute(str, Enum):
    """Store the outward coefficient-error route.

    :see: :class:`~.test_realization_types.TestGalerkinPotentialRealization`

    Attributes
    ----------
    DIRECT_PAIRWISE_HOST_INTERVAL : str
        Exact-rational direct DFT with pairwise interval accumulation.
    TRIANGLE_FALLBACK : str
        Backend-independent triangle bound for one rounded FFT result.
    """

    DIRECT_PAIRWISE_HOST_INTERVAL = "vc1_direct_pairwise_host_interval"
    TRIANGLE_FALLBACK = "vc1_triangle_fallback"


class GalerkinPotentialCertificateFailure(str, Enum):
    """Store the outcome of one host coefficient-certificate attempt.

    :see: :class:`GalerkinPotentialCoefficientCertificate`
    :see: :class:`~.test_realization_types.\
TestCoefficientCertificateTypes`

    Attributes
    ----------
    NONE : str
        The stored exact-coefficient rectangles are finite certificates.
    HOST_ARITHMETIC_UNSUPPORTED : str
        The host failed a required binary64 arithmetic capability probe.
    WORK_BUDGET_EXCEEDED : str
        The declared direct-term budget was insufficient.
    ROOT_ENCLOSURE_FAILURE : str
        Certified rational-turn phase construction failed closed.
    ARITHMETIC_RANGE_FAILURE : str
        An exact host endpoint could not be represented outward.
    """

    NONE = "none"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinPotentialCoefficientCertificate(eqx.Module):
    """Store exact-coefficient rectangles produced by one host checker.

    :see: :func:`create_galerkin_potential_coefficient_certificate`
    :see: :class:`~.test_realization_types.\
TestCoefficientCertificateTypes`

    Attributes
    ----------
    exact_coefficient_real_lower_bounds : Float64[Array, " p"]
        Lower real endpoints for the exact VC-1 coefficients.
    exact_coefficient_real_upper_bounds : Float64[Array, " p"]
        Upper real endpoints for the exact VC-1 coefficients.
    exact_coefficient_imag_lower_bounds : Float64[Array, " p"]
        Lower imaginary endpoints for the exact VC-1 coefficients.
    exact_coefficient_imag_upper_bounds : Float64[Array, " p"]
        Upper imaginary endpoints for the exact VC-1 coefficients.
    finite_certificate : Bool[Array, ""]
        Whether every stored rectangle is finite.
    direct_term_count : Int64[Array, ""]
        Number of voxel--coefficient terms and state-difference pairs
        requested by the checker.
    maximum_direct_terms : Int64[Array, ""]
        Caller-declared maximum direct-term budget.
    failure : GalerkinPotentialCertificateFailure
        Static typed failure outcome. This value affects tracing.
    exact_target : str
        Static exact-target declaration. This value affects tracing.
    arithmetic : str
        Static host arithmetic declaration. This value affects tracing.

    Notes
    -----
    This carrier is evidence nested inside one
    :class:`GalerkinPotentialRealization`; it does not duplicate the source
    potential, ordered support, or rounded production coefficients.
    """

    exact_coefficient_real_lower_bounds: Float64[Array, " p"]
    exact_coefficient_real_upper_bounds: Float64[Array, " p"]
    exact_coefficient_imag_lower_bounds: Float64[Array, " p"]
    exact_coefficient_imag_upper_bounds: Float64[Array, " p"]
    finite_certificate: Bool[Array, ""]
    direct_term_count: Int64[Array, ""]
    maximum_direct_terms: Int64[Array, ""]
    failure: GalerkinPotentialCertificateFailure = eqx.field(static=True)
    exact_target: str = eqx.field(static=True)
    arithmetic: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_potential_coefficient_certificate(  # noqa: PLR0913
    exact_coefficient_real_lower_bounds: Float[Array, "..."],
    exact_coefficient_real_upper_bounds: Float[Array, "..."],
    exact_coefficient_imag_lower_bounds: Float[Array, "..."],
    exact_coefficient_imag_upper_bounds: Float[Array, "..."],
    finite_certificate: Bool[Array, ""],
    direct_term_count: Int[Array, ""],
    maximum_direct_terms: Int[Array, ""],
    *,
    failure: GalerkinPotentialCertificateFailure,
    exact_target: str,
    arithmetic: str,
) -> GalerkinPotentialCoefficientCertificate:
    """Create a structurally validated host coefficient certificate.

    :see: :class:`~.test_realization_types.\
TestCoefficientCertificateTypes`

    Parameters
    ----------
    exact_coefficient_real_lower_bounds : Float[Array, "..."]
        Submitted real lower endpoints.
    exact_coefficient_real_upper_bounds : Float[Array, "..."]
        Submitted real upper endpoints.
    exact_coefficient_imag_lower_bounds : Float[Array, "..."]
        Submitted imaginary lower endpoints.
    exact_coefficient_imag_upper_bounds : Float[Array, "..."]
        Submitted imaginary upper endpoints.
    finite_certificate : Bool[Array, ""]
        Whether every endpoint is finite.
    direct_term_count : Int[Array, ""]
        Requested direct coefficient and matrix-transfer work count.
    maximum_direct_terms : Int[Array, ""]
        Declared positive work budget.
    failure : GalerkinPotentialCertificateFailure
        Typed certificate outcome.
    exact_target : str
        Nonempty exact-target declaration.
    arithmetic : str
        Nonempty host arithmetic declaration.

    Returns
    -------
    certificate : GalerkinPotentialCoefficientCertificate
        Validated exact-coefficient rectangles and outcome metadata.

    Raises
    ------
    ValueError
        If ranks, shapes, scalar fields, or static declarations are invalid.
    equinox.EquinoxRuntimeError
        If endpoints are NaN, ordered incorrectly, or contradict the typed
        finite/failure outcome.
    """
    real_lower: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_real_lower_bounds,
        dtype=jnp.float64,
    )
    real_upper: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_real_upper_bounds,
        dtype=jnp.float64,
    )
    imag_lower: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_imag_lower_bounds,
        dtype=jnp.float64,
    )
    imag_upper: Float64[Array, " p"] = jnp.asarray(
        exact_coefficient_imag_upper_bounds,
        dtype=jnp.float64,
    )
    finite: Bool[Array, ""] = jnp.asarray(
        finite_certificate,
        dtype=jnp.bool_,
    )
    term_count: Int64[Array, ""] = jnp.asarray(
        direct_term_count,
        dtype=jnp.int64,
    )
    term_budget: Int64[Array, ""] = jnp.asarray(
        maximum_direct_terms,
        dtype=jnp.int64,
    )
    endpoint_arrays: Tuple[Float64[Array, " p"], ...] = (
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
    )
    _raise_if(
        any(array.ndim != 1 for array in endpoint_arrays),
        "exact coefficient endpoints must be 1D",
    )
    _raise_if(
        any(array.shape != real_lower.shape for array in endpoint_arrays),
        "exact coefficient endpoint arrays must share one shape",
    )
    for value, name in (
        (finite, "finite_certificate"),
        (term_count, "direct_term_count"),
        (term_budget, "maximum_direct_terms"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")
    _raise_if(not exact_target.strip(), "exact_target must be nonempty")
    _raise_if(not arithmetic.strip(), "arithmetic must be nonempty")

    invalid_endpoints: Bool[Array, ""] = (
        jnp.any(jnp.isnan(real_lower))
        | jnp.any(jnp.isnan(real_upper))
        | jnp.any(jnp.isnan(imag_lower))
        | jnp.any(jnp.isnan(imag_upper))
        | jnp.any(real_lower > real_upper)
        | jnp.any(imag_lower > imag_upper)
    )
    all_endpoints_finite: Bool[Array, ""] = (
        jnp.all(jnp.isfinite(real_lower))
        & jnp.all(jnp.isfinite(real_upper))
        & jnp.all(jnp.isfinite(imag_lower))
        & jnp.all(jnp.isfinite(imag_upper))
    )
    failure_is_none: bool = failure is GalerkinPotentialCertificateFailure.NONE
    contradiction: Bool[Array, ""] = (
        (finite != all_endpoints_finite)
        | (finite != failure_is_none)
        | (term_count < 0)
        | (term_budget <= 0)
        | (finite & (term_count > term_budget))
    )
    checked_real_lower: Float64[Array, " p"] = eqx.error_if(
        real_lower,
        invalid_endpoints | contradiction,
        "coefficient certificate endpoints or outcome are inconsistent",
    )
    certificate: GalerkinPotentialCoefficientCertificate = (
        GalerkinPotentialCoefficientCertificate(
            exact_coefficient_real_lower_bounds=checked_real_lower,
            exact_coefficient_real_upper_bounds=real_upper,
            exact_coefficient_imag_lower_bounds=imag_lower,
            exact_coefficient_imag_upper_bounds=imag_upper,
            finite_certificate=finite,
            direct_term_count=term_count,
            maximum_direct_terms=term_budget,
            failure=failure,
            exact_target=exact_target.strip(),
            arithmetic=arithmetic.strip(),
        )
    )
    return certificate


class GalerkinPotentialRealizationMethod(str, Enum):
    """Store the exact finite-target realization method.

    :see: :class:`~.test_realization_types.TestGalerkinPotentialRealization`

    Attributes
    ----------
    PERIODIC_TRIGONOMETRIC : str
        VC-1 periodic trigonometric interpolant of the stored voxels.
    """

    PERIODIC_TRIGONOMETRIC = "vc1_periodic_trigonometric"


class GalerkinPotentialRealization(eqx.Module):
    """Store one VC-1 voxel-to-coefficient realization.

    :see: :class:`~.test_realization_types.TestGalerkinPotentialRealization`

    Attributes
    ----------
    potential : Potential3D
        Exact source voxel values and their static physical metadata.
    support_eligibility : GalerkinAcquisitionSupportResult
        Independently checked finite acquisition-support artifact.
    voltage_coefficients : Complex128[Array, " p"]
        Rounded SC.13b voltage coefficients on the interaction support.
    coefficient_error_bounds : Float64[Array, " p"]
        Outward componentwise bounds relative to the exact VC-1 DFT.
    voltage_operator_error_bound : Float64[Array, ""]
        Outward spectral-norm bound for the compressed voltage multiplier.
    omitted_voltage_l2_diagnostic : Float64[Array, ""]
        Floating Parseval diagnostic for modes outside the interaction band,
        in volt Angstrom to the power three-halves.
    omitted_voltage_l2_upper_bound : Float64[Array, ""]
        Sound upper bound for the same omitted-field norm. Infinity is an
        explicit noncertificate.
    coefficient_certificate : GalerkinPotentialCoefficientCertificate or None
        Optional stopped host evidence for the exact VC-1 coefficient
        rectangles. ``None`` means that no direct certificate was attempted.
    method : GalerkinPotentialRealizationMethod
        Static exact finite-target method. This value affects tracing.
    target_route : GalerkinVoxelTargetRoute
        Hardcoded trigonometric VC-1 route identity. This value affects
        tracing and canonical target identity.
    error_route : GalerkinPotentialErrorRoute
        Static coefficient-error route. This value affects tracing.
    output_coefficient_normalization : str
        Static output normalization identifier. This value affects tracing.
    endpoint_convention : str
        Static signed DFT endpoint convention. This value affects tracing.
    voxel_metric : str
        Static real voxel parameter metric. This value affects tracing.

    See Also
    --------
    :func:`create_galerkin_potential_realization`
        Validate one realization and its evidence arrays.
    """

    potential: Potential3D
    support_eligibility: GalerkinAcquisitionSupportResult
    voltage_coefficients: Complex128[Array, " p"]
    coefficient_error_bounds: Float64[Array, " p"]
    voltage_operator_error_bound: Float64[Array, ""]
    omitted_voltage_l2_diagnostic: Float64[Array, ""]
    omitted_voltage_l2_upper_bound: Float64[Array, ""]
    target_route: GalerkinVoxelTargetRoute = eqx.field(static=True)
    method: GalerkinPotentialRealizationMethod = eqx.field(static=True)
    error_route: GalerkinPotentialErrorRoute = eqx.field(static=True)
    output_coefficient_normalization: str = eqx.field(static=True)
    endpoint_convention: str = eqx.field(static=True)
    voxel_metric: str = eqx.field(static=True)
    coefficient_certificate: GalerkinPotentialCoefficientCertificate | None = (
        None
    )

    @property
    def support(self) -> GalerkinProductSupport:
        """Return the checked finite product support without duplicating it."""
        support: GalerkinProductSupport = (
            self.support_eligibility.manifest.support
        )
        return support


@jaxtyped(typechecker=beartype)
def create_galerkin_potential_realization(  # noqa: PLR0913
    potential: Potential3D,
    support_eligibility: GalerkinAcquisitionSupportResult,
    voltage_coefficients: Complex[Array, "..."],
    coefficient_error_bounds: Float[Array, "..."],
    voltage_operator_error_bound: Float[Array, ""],
    omitted_voltage_l2_diagnostic: Float[Array, ""],
    omitted_voltage_l2_upper_bound: Float[Array, ""],
    *,
    method: GalerkinPotentialRealizationMethod,
    error_route: GalerkinPotentialErrorRoute,
    output_coefficient_normalization: str,
    endpoint_convention: str,
    voxel_metric: str,
) -> GalerkinPotentialRealization:
    """Create a validated voxel-to-Galerkin realization carrier.

    :see: :class:`~.test_realization_types.TestGalerkinPotentialRealization`

    Parameters
    ----------
    potential : Potential3D
        Exact source voxel values and static physical metadata.
    support_eligibility : GalerkinAcquisitionSupportResult
        Checked acquisition-support artifact bound to the coefficient order.
    voltage_coefficients : Complex[Array, "..."]
        Rounded voltage coefficients in SC.13b normalization.
    coefficient_error_bounds : Float[Array, "..."]
        Non-negative outward componentwise coefficient-error bounds.
    voltage_operator_error_bound : Float[Array, ""]
        Non-negative outward compressed-multiplier error bound.
    omitted_voltage_l2_diagnostic : Float[Array, ""]
        Non-negative floating truncation diagnostic.
    omitted_voltage_l2_upper_bound : Float[Array, ""]
        Non-negative sound truncation upper bound.
    method : GalerkinPotentialRealizationMethod
        Static exact finite-target method.
    error_route : GalerkinPotentialErrorRoute
        Static outward-error route.
    output_coefficient_normalization : str
        Static nonempty output normalization identifier.
    endpoint_convention : str
        Static nonempty signed endpoint identifier.
    voxel_metric : str
        Static nonempty real voxel metric identifier.

    Returns
    -------
    realization : GalerkinPotentialRealization
        Validated source, coefficients, and distinct error evidence.

    Raises
    ------
    ValueError
        If an array rank, size, scalar shape, or static identifier is invalid.
    equinox.EquinoxRuntimeError
        If a coefficient is non-finite or an error value is NaN or negative.

    Notes
    -----
    Infinite bounds are retained as typed noncertificates. They are not
    converted to finite evidence or rejected as though the target were
    structurally invalid.
    """
    coefficient_array: Complex128[Array, " p"] = jnp.asarray(
        voltage_coefficients,
        dtype=jnp.complex128,
    )
    coefficient_error_array: Float64[Array, " p"] = jnp.asarray(
        coefficient_error_bounds,
        dtype=jnp.float64,
    )
    operator_error_array: Float64[Array, ""] = jnp.asarray(
        voltage_operator_error_bound,
        dtype=jnp.float64,
    )
    omitted_diagnostic_array: Float64[Array, ""] = jnp.asarray(
        omitted_voltage_l2_diagnostic,
        dtype=jnp.float64,
    )
    omitted_upper_array: Float64[Array, ""] = jnp.asarray(
        omitted_voltage_l2_upper_bound,
        dtype=jnp.float64,
    )

    support: GalerkinProductSupport = support_eligibility.manifest.support
    expected_size: int = support.interaction_indices.shape[0]
    _raise_if(coefficient_array.ndim != 1, "voltage_coefficients must be 1D")
    _raise_if(
        coefficient_array.shape[0] != expected_size,
        "voltage_coefficients must match the interaction support",
    )
    _raise_if(
        coefficient_error_array.shape != coefficient_array.shape,
        "coefficient_error_bounds must match voltage_coefficients",
    )
    for value, name in (
        (operator_error_array, "voltage_operator_error_bound"),
        (omitted_diagnostic_array, "omitted_voltage_l2_diagnostic"),
        (omitted_upper_array, "omitted_voltage_l2_upper_bound"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")
    for value, name in (
        (output_coefficient_normalization, "output_coefficient_normalization"),
        (endpoint_convention, "endpoint_convention"),
        (voxel_metric, "voxel_metric"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")

    eligible_status: Bool[Array, ""] = support_eligibility.status == int(
        GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE
    )
    checked_eligibility_anchor: Float64[Array, ""] = eqx.error_if(
        operator_error_array,
        (~eligible_status) | (~support_eligibility.support_eligible),
        "support_eligibility must report SUPPORT_ELIGIBLE",
    )

    checked_coefficients: Complex128[Array, " p"] = eqx.error_if(
        coefficient_array,
        jnp.any(~jnp.isfinite(coefficient_array)),
        "voltage_coefficients must be finite",
    )
    checked_coefficient_errors: Float64[Array, " p"] = eqx.error_if(
        coefficient_error_array,
        jnp.any(jnp.isnan(coefficient_error_array))
        | jnp.any(coefficient_error_array < 0.0),
        "coefficient_error_bounds must be non-negative and not NaN",
    )
    checked_operator_error: Float64[Array, ""] = eqx.error_if(
        checked_eligibility_anchor,
        jnp.isnan(operator_error_array) | (operator_error_array < 0.0),
        "voltage_operator_error_bound must be non-negative and not NaN",
    )
    checked_omitted_diagnostic: Float64[Array, ""] = eqx.error_if(
        omitted_diagnostic_array,
        (~jnp.isfinite(omitted_diagnostic_array))
        | (omitted_diagnostic_array < 0.0),
        "omitted_voltage_l2_diagnostic must be finite and non-negative",
    )
    checked_omitted_upper: Float64[Array, ""] = eqx.error_if(
        omitted_upper_array,
        jnp.isnan(omitted_upper_array) | (omitted_upper_array < 0.0),
        "omitted_voltage_l2_upper_bound must be non-negative and not NaN",
    )

    realization: GalerkinPotentialRealization = GalerkinPotentialRealization(
        potential=potential,
        support_eligibility=support_eligibility,
        voltage_coefficients=checked_coefficients,
        coefficient_error_bounds=checked_coefficient_errors,
        voltage_operator_error_bound=checked_operator_error,
        omitted_voltage_l2_diagnostic=checked_omitted_diagnostic,
        omitted_voltage_l2_upper_bound=checked_omitted_upper,
        target_route=GalerkinVoxelTargetRoute.TRIGONOMETRIC_VC1,
        method=method,
        error_route=error_route,
        output_coefficient_normalization=output_coefficient_normalization.strip(),
        endpoint_convention=endpoint_convention.strip(),
        voxel_metric=voxel_metric.strip(),
        coefficient_certificate=None,
    )
    return realization


__all__: list[str] = [
    "GalerkinPotentialCertificateFailure",
    "GalerkinPotentialCoefficientCertificate",
    "GalerkinPotentialErrorRoute",
    "GalerkinPotentialRealization",
    "GalerkinPotentialRealizationMethod",
    "create_galerkin_potential_coefficient_certificate",
    "create_galerkin_potential_realization",
]
