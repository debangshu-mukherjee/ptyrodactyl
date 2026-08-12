r"""Certify bounded local represented-source solve stability.

Extended Summary
----------------
This leaf authenticates one local represented-source certificate, ignores all
solver-reported residual values, and reconstructs the stored algebraic
residual with exact dyadic host arithmetic.  It lifts that residual exactly
once by ``delta_H ||x||`` and the represented total-source error, then divides
by the independently certified exact-target L4 physical CAP floor.

Routine Listings
----------------
:func:`check_local_represented_galerkin_absorber_floor`
    Build one bounded exact-dyadic proof for a submitted state.
:func:`invoke_local_represented_galerkin_stability`
    Recheck the proof and return a fully nested stability result.
:func:`prepare_local_galerkin_stability_result`
    Full-replay and exact-compare one local stability result.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from numpy.typing import NDArray

from ptyrodactyl._tools import (
    RootEnclosureError,
    all_normal_arithmetic_supported,
    arithmetic_environment_probes,
    array_payload,
    fraction_from_float,
    fraction_upper_float,
    has_subnormal_components,
    host_array,
    host_binary64_supported,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types.born_types import (
    GalerkinCertificateReason,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
)
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalRepresentedSourceCertificate,
)
from ptyrodactyl.types.local_stability_types import (
    GalerkinLocalStabilityDisposition,
    GalerkinLocalStabilityFailure,
    GalerkinLocalStabilityProof,
    GalerkinLocalStabilityResult,
    GalerkinLocalStabilityRoute,
    _make_local_stability_proof,
    _make_local_stability_result,
)

from .local_represented_sources import (
    prepare_local_represented_source_certificate,
)

type _ComplexFraction = tuple[Fraction, Fraction]

_CHECKER_ID: str = "ptyrodactyl.local_stability.exact_dyadic_dense.v1"
_COEFFICIENT_NORM: str = (
    "Euclidean l2 norm of ordered I_u complex state vectors and state-error "
    "radii"
)
_COMPLETION_SCOPE: str = (
    "one exact L4 same-target matrix floor, plus a retained-state radius when "
    "the represented source and bounded residual route succeed; operational "
    "eligibility is conditional on independent caller-supplied state/direct-"
    "work policy; no terminal, current, detector, continuum, CAP-removal, or "
    "solver-convergence claim"
)
_DEFAULT_MAXIMUM_DIRECT_PAIRS: int = 2_000_000
_ERROR_SCOPE: str = (
    "rho_alg from exact dyadic stored D_alg/R_alg/B_alg/b_alg/x; adds only "
    "delta_H*||x|| and eta_T exactly once; excludes solver-reported residual, "
    "per-call rounding, individual delta_D/delta_R/delta_B, component source "
    "bounds, duplicate LVT.20 error, incident/scattered bounds, slab, "
    "terminal, current, and detector errors; floor and delta_H use "
    "inverse-square Angstrom units, residuals use corresponding action "
    "units, and the radius uses the coefficient l2 norm; maximum_direct_pairs "
    "bounds only the post-parent-replay n+2*n**2 dense residual and excludes "
    "target/represented-source authentication"
)
_FLOOR_SCOPE: str = (
    "exact_target_physical_floor_lower_bound from the replayed L4 proof used "
    "directly in inverse-square Angstrom units; no realized floor and no "
    "subtraction of delta_H"
)
_IDENTITY_DIGEST_DOMAIN: str = "ptyrodactyl.local_stability.identity.v1"
_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_PROOF_DIGEST_DOMAIN: str = "ptyrodactyl.local_stability.proof_evidence.v1"
_RESIDUAL_FORMULA: str = (
    "r_alg=b_alg-(D_alg-R_alg-i*B_alg)x; exact dyadic dense contraction"
)
_RESIDUAL_LIFT_FORMULA: str = (
    "R=up(rho_alg+up(delta_H*||x||)+eta_T); "
    "state_radius=up(R/exact_L4_physical_floor)"
)
_RESULT_DIGEST_DOMAIN: str = "ptyrodactyl.local_stability.result_evidence.v1"
_ROUTE: GalerkinLocalStabilityRoute = (
    GalerkinLocalStabilityRoute.EXACT_AXIAL_CAP_FLOOR
)


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for one host-boundary failure.

    Parameters
    ----------
    condition : bool
        Whether the checked boundary condition failed.
    message : str
        Failure detail for the caller.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced leaves at this exact host checker boundary.

    Parameters
    ----------
    value : object
        Submitted carrier or value tree.

    Raises
    ------
    ValueError
        If any submitted leaf is a JAX tracer.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError(
            "local stability replay requires concrete host values"
        )


def _checked_positive_budget(value: object, name: str) -> float:
    """PRIVATE: Return one finite positive normal binary64 budget.

    Parameters
    ----------
    value : object
        Candidate Python float or exact NumPy/JAX float64 scalar.
    name : str
        Public parameter name used in diagnostics.

    Returns
    -------
    result : float
        Concrete finite positive normal binary64 policy value.

    Raises
    ------
    ValueError
        If the value is traced, nonscalar, non-float64, or outside range.
    """
    if isinstance(value, bool | int | complex):
        raise ValueError(f"{name} must be an exact binary64 scalar")
    try:
        candidate = np.asarray(jax.device_get(value))
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a concrete scalar") from error
    _raise_if(candidate.shape != (), f"{name} must be a scalar")
    _raise_if(
        candidate.dtype != np.dtype(np.float64),
        f"{name} must be an exact float64 scalar",
    )
    try:
        result = float(candidate)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be binary64-convertible") from error
    _raise_if(
        not math.isfinite(result) or result < float(np.finfo(np.float64).tiny),
        f"{name} must be finite and at least the smallest normal float64",
    )
    return result


def _finite_normal_or_zero_float(value: float) -> bool:
    """PRIVATE: Check one host float for finite normal-or-zero storage.

    Parameters
    ----------
    value : float
        Host binary64 report candidate.

    Returns
    -------
    valid : bool
        Whether the value is finite and normal or exactly zero.
    """
    return math.isfinite(value) and (
        value == 0.0 or abs(value) >= float(np.finfo(np.float64).tiny)
    )


def _checked_pair_budget(value: object) -> int:
    """PRIVATE: Return one positive signed-int64 direct-work budget.

    Parameters
    ----------
    value : object
        Candidate direct-work policy value.

    Returns
    -------
    value : int
        Positive Python integer representable by signed int64.

    Raises
    ------
    ValueError
        If the value is not an integer in the admitted range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("maximum_direct_pairs must be an integer")
    _raise_if(
        value <= 0 or value > _MAXIMUM_SIGNED_INT64,
        "maximum_direct_pairs must be a positive signed-int64 integer",
    )
    return value


def _validate_solve_result(
    solve_result: object, target_size: int
) -> GalerkinSolveResult:
    """PRIVATE: Validate one concrete algebraic solve carrier schema.

    Reported residual values are checked only as carrier data and are never
    read by the mathematical residual reconstruction.

    Parameters
    ----------
    solve_result : object
        Raw submitted algebraic solve carrier.
    target_size : int
        Expected retained-state vector length.

    Returns
    -------
    solve_result : GalerkinSolveResult
        The same solve carrier after complete schema validation.

    Raises
    ------
    TypeError
        If the submitted object is not a Galerkin solve carrier.
    ValueError
        If array, scalar, enum, convergence, or finiteness checks fail.
    """
    if not isinstance(solve_result, GalerkinSolveResult):
        raise TypeError("solve_result must be GalerkinSolveResult")
    _assert_concrete(solve_result)
    field = host_array(solve_result.field)
    residual = host_array(solve_result.residual)
    _raise_if(
        field.dtype != np.dtype(np.complex128)
        or field.shape != (target_size,),
        "solve_result.field must be complex128 and match target I_u",
    )
    _raise_if(
        residual.dtype != np.dtype(np.complex128)
        or residual.shape != (target_size,),
        "solve_result.residual must be complex128 and match target I_u",
    )
    _raise_if(
        not bool(np.all(np.isfinite(field)))
        or bool(has_subnormal_components(solve_result.field)),
        "solve_result.field must be finite normal-or-zero complex128",
    )
    _raise_if(
        not bool(np.all(np.isfinite(residual))),
        "solve_result.residual carrier data must be finite",
    )
    for metric, name in (
        (solve_result.residual_norm, "residual_norm"),
        (solve_result.normal_residual_norm, "normal_residual_norm"),
        (solve_result.recurrence_residual_norm, "recurrence_residual_norm"),
    ):
        host = host_array(metric)
        _raise_if(
            host.dtype != np.dtype(np.float64)
            or host.shape != ()
            or not math.isfinite(float(host))
            or float(host) < 0.0,
            f"solve_result.{name} must be finite nonnegative float64 scalar",
        )
    for counter, name in (
        (solve_result.iterations, "iterations"),
        (solve_result.operator_applications, "operator_applications"),
        (solve_result.status, "status"),
    ):
        host = host_array(counter)
        _raise_if(
            host.dtype != np.dtype(np.int32) or host.shape != (),
            f"solve_result.{name} must be int32 scalar",
        )
    converged = host_array(solve_result.converged)
    _raise_if(
        converged.dtype != np.dtype(np.bool_) or converged.shape != (),
        "solve_result.converged must be bool scalar",
    )
    _raise_if(
        int(host_array(solve_result.iterations)) < 0
        or int(host_array(solve_result.operator_applications)) < 0,
        "solve_result counters cannot be negative",
    )
    try:
        status = GalerkinSolveStatus(int(host_array(solve_result.status)))
    except ValueError as error:
        raise ValueError("solve_result.status is not canonical") from error
    _raise_if(
        bool(converged) != (status is GalerkinSolveStatus.CONVERGED),
        "solve_result convergence flag disagrees with status",
    )
    _raise_if(
        not isinstance(solve_result.method, GalerkinSolveMethod)
        or not isinstance(
            solve_result.certificate_reason, GalerkinCertificateReason
        ),
        "solve_result static method/reason schema is invalid",
    )
    return solve_result


def _environment_payload() -> tuple[dict[str, object], bool, bool]:
    """PRIVATE: Record every relevant host and device arithmetic probe.

    Returns
    -------
    payload : dict[str, object]
        Canonical named arithmetic-probe values.
    host_supported : bool
        Whether host exact-binary64 assumptions hold.
    normal_supported : bool
        Whether all required normal-range arithmetic probes pass.
    """
    probes = arithmetic_environment_probes()
    probe_values = tuple(bool(host_array(value)) for value in probes)
    host_supported = host_binary64_supported()
    normal_supported = bool(host_array(all_normal_arithmetic_supported()))
    payload: dict[str, object] = {
        "host_binary64_supported": host_supported,
        "normal_arithmetic_supported": normal_supported,
        "addition_supported": probe_values[0],
        "multiplication_supported": probe_values[1],
        "division_supported": probe_values[2],
        "square_root_supported": probe_values[3],
        "nextafter_supported": probe_values[4],
        "bit_pattern_supported": probe_values[5],
        "gradual_underflow_supported_diagnostic_only": probe_values[6],
    }
    return payload, host_supported, normal_supported


def _direct_work_count(size: int) -> int:
    """PRIVATE: Return the exact ``n + 2 n**2`` dense residual work count.

    Parameters
    ----------
    size : int
        Retained-state dimension ``n``.

    Returns
    -------
    count : int
        Unclamped exact dense residual work count.

    Raises
    ------
    ValueError
        If ``size`` is negative.
    """
    _raise_if(size < 0, "target-state count cannot be negative")
    return size + 2 * size * size


def _complex_fraction(value: np.complex128) -> _ComplexFraction:
    """PRIVATE: Convert one finite complex128 point to exact dyadic parts.

    Parameters
    ----------
    value : np.complex128
        Stored complex binary64 point.

    Returns
    -------
    parts : _ComplexFraction
        Exact real and imaginary rational components.
    """
    point = complex(value)
    return (
        fraction_from_float(float(point.real)),
        fraction_from_float(float(point.imag)),
    )


def _complex_add(
    left: _ComplexFraction, right: _ComplexFraction
) -> _ComplexFraction:
    """PRIVATE: Add two exact complex rationals.

    Parameters
    ----------
    left : _ComplexFraction
        Left exact complex value.
    right : _ComplexFraction
        Right exact complex value.

    Returns
    -------
    total : _ComplexFraction
        Exact componentwise sum.
    """
    return left[0] + right[0], left[1] + right[1]


def _complex_subtract(
    left: _ComplexFraction, right: _ComplexFraction
) -> _ComplexFraction:
    """PRIVATE: Subtract two exact complex rationals.

    Parameters
    ----------
    left : _ComplexFraction
        Left exact complex value.
    right : _ComplexFraction
        Right exact complex value.

    Returns
    -------
    difference : _ComplexFraction
        Exact componentwise difference.
    """
    return left[0] - right[0], left[1] - right[1]


def _complex_multiply(
    left: _ComplexFraction, right: _ComplexFraction
) -> _ComplexFraction:
    """PRIVATE: Multiply two exact complex rationals.

    Parameters
    ----------
    left : _ComplexFraction
        Left exact complex value.
    right : _ComplexFraction
        Right exact complex value.

    Returns
    -------
    product : _ComplexFraction
        Exact complex product.
    """
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _multiply_i(value: _ComplexFraction) -> _ComplexFraction:
    """PRIVATE: Multiply one exact complex rational by exact ``i``.

    Parameters
    ----------
    value : _ComplexFraction
        Exact complex value.

    Returns
    -------
    product : _ComplexFraction
        Exact value after multiplication by ``i``.
    """
    return -value[1], value[0]


def _exact_dense_residual_squared(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
) -> tuple[Fraction, Fraction]:
    r"""PRIVATE: Reconstruct exact ``||b-(D-R-iB)x||^2`` and ``||x||^2``.

    Every input component is interpreted as the exact dyadic represented by
    its stored binary64 bits.  No rounded action or reported solve residual is
    consulted.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Fully replayed local represented-source certificate.
    solve_result : GalerkinSolveResult
        Validated submitted state carrier.

    Returns
    -------
    residual_squared : Fraction
        Exact squared Euclidean norm of the dense algebraic residual.
    field_norm_squared : Fraction
        Exact squared Euclidean norm of the submitted coefficient vector.
    """
    source = certificate.source
    target = source.target
    size = target.state_indices.shape[0]
    field = cast(NDArray[np.complex128], host_array(solve_result.field))
    rhs = cast(NDArray[np.complex128], host_array(source.actions.total_source))
    diagonal = cast(NDArray[np.float64], host_array(target.free_diagonal))
    compression = target.compression
    interaction_coefficients = cast(
        NDArray[np.complex128],
        host_array(compression.interaction_coefficients),
    )
    interaction_positions = cast(
        NDArray[np.int64],
        host_array(compression.state_pair_interaction_positions),
    )
    cap_certificate = target.cap_floor_proof.coefficient_certificate
    absorber = cap_certificate.absorber
    absorber_coefficients = cast(
        NDArray[np.complex128], host_array(absorber.absorber_coefficients)
    )
    absorber_positions = cast(
        NDArray[np.int64],
        host_array(cap_certificate.state_pair_absorber_positions),
    )
    cap_scale = fraction_from_float(
        float(host_array(absorber.algebraic_cap_scale))
    )
    field_exact = tuple(_complex_fraction(value) for value in field)
    residual_exact = [_complex_fraction(value) for value in rhs]
    for row in range(size):
        free_product = (
            fraction_from_float(float(diagonal[row])) * field_exact[row][0],
            fraction_from_float(float(diagonal[row])) * field_exact[row][1],
        )
        residual_exact[row] = _complex_subtract(
            residual_exact[row], free_product
        )
        for column in range(size):
            flat = row * size + column
            interaction = _complex_fraction(
                interaction_coefficients[interaction_positions[flat]]
            )
            interaction_product = _complex_multiply(
                interaction, field_exact[column]
            )
            residual_exact[row] = _complex_add(
                residual_exact[row], interaction_product
            )
            absorber_point = _complex_fraction(
                absorber_coefficients[absorber_positions[flat]]
            )
            physical_absorber = (
                cap_scale * absorber_point[0],
                cap_scale * absorber_point[1],
            )
            absorber_product = _complex_multiply(
                physical_absorber, field_exact[column]
            )
            residual_exact[row] = _complex_add(
                residual_exact[row], _multiply_i(absorber_product)
            )
    residual_squared = sum(
        (real * real + imag * imag for real, imag in residual_exact),
        start=Fraction(0),
    )
    field_norm_squared = sum(
        (real * real + imag * imag for real, imag in field_exact),
        start=Fraction(0),
    )
    return residual_squared, field_norm_squared


def _result_identity_digest(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
) -> str:
    """PRIVATE: Bind target, source identity, and submitted field bytes.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Fully replayed represented-source certificate.
    solve_result : GalerkinSolveResult
        Validated submitted state carrier.

    Returns
    -------
    digest : str
        Canonical lowercase SHA-256 identity digest.
    """
    source = certificate.source
    payload: dict[str, object] = {
        "domain": _IDENTITY_DIGEST_DOMAIN,
        "target_digest": source.target.target_digest,
        "source_digest": source.source_digest,
        "field": array_payload(solve_result.field),
    }
    return sha256(payload)


def _fraction_fields(prefix: str, value: Fraction) -> dict[str, int]:
    """PRIVATE: Return constructor fields for one reduced rational value.

    Parameters
    ----------
    prefix : str
        Common constructor-field prefix.
    value : Fraction
        Reduced nonnegative exact rational value.

    Returns
    -------
    fields : dict[str, int]
        Numerator and denominator constructor fields.
    """
    return {
        f"{prefix}_numerator": value.numerator,
        f"{prefix}_denominator": value.denominator,
    }


def _proof_evidence_digest(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
    proof: GalerkinLocalStabilityProof,
    environment_payload: dict[str, object],
) -> str:
    """PRIVATE: Bind complete parents, transcripts, budgets, and outcome.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Complete replayed represented-source certificate.
    solve_result : GalerkinSolveResult
        Full submitted algebraic solve payload.
    proof : GalerkinLocalStabilityProof
        Proof payload whose digest field is excluded from self-hashing.
    environment_payload : dict[str, object]
        Complete named arithmetic-probe mapping.

    Returns
    -------
    digest : str
        Canonical lowercase SHA-256 proof-evidence digest.

    Raises
    ------
    TypeError
        If canonical proof storage does not serialize as a mapping.
    """
    proof_payload = stored_value_payload(proof)
    if not isinstance(proof_payload, dict):
        raise TypeError("proof payload must be a canonical mapping")
    proof_mapping = cast(dict[str, object], proof_payload)
    fields = cast(dict[str, object], proof_mapping["fields"])
    fields = dict(fields)
    fields.pop("proof_evidence_digest")
    payload: dict[str, object] = {
        "domain": _PROOF_DIGEST_DOMAIN,
        "target": stored_value_payload(certificate.source.target),
        "source": stored_value_payload(certificate.source),
        "certificate": stored_value_payload(certificate),
        "l4_floor_proof": stored_value_payload(
            certificate.source.target.cap_floor_proof
        ),
        "solve_result": stored_value_payload(solve_result),
        "proof_fields": fields,
        "arithmetic_environment": environment_payload,
    }
    return sha256(payload)


def _base_proof(  # noqa: PLR0913
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
    maximum_state_error: float,
    maximum_direct_pairs: int,
    exact_work_count: int,
    environment_payload: dict[str, object],
    host_supported: bool,
    normal_supported: bool,
    *,
    failure: GalerkinLocalStabilityFailure,
    matrix_floor: Fraction | None = None,
    finite_values: dict[str, Fraction] | None = None,
) -> GalerkinLocalStabilityProof:
    """PRIVATE: Construct, hash, and validate one proof outcome.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Complete replayed represented-source certificate.
    solve_result : GalerkinSolveResult
        Validated submitted algebraic solve payload.
    maximum_state_error : float
        Independent coefficient-l2 state policy budget.
    maximum_direct_pairs : int
        Independent signed-int64 dense-work policy budget.
    exact_work_count : int
        Unclamped exact ``n + 2 n**2`` work count.
    environment_payload : dict[str, object]
        Complete named arithmetic-probe mapping.
    host_supported : bool
        Whether exact host binary64 assumptions hold.
    normal_supported : bool
        Whether required normal-range device arithmetic is supported.
    failure : GalerkinLocalStabilityFailure
        Typed outcome selected by the caller.
    matrix_floor : Fraction | None
        Positive authenticated exact L4 floor, or ``None`` if omitted because
        the matrix floor itself failed.
    finite_values : dict[str, Fraction] | None
        Complete finite state transcripts, or ``None`` if omitted for a
        matrix-only or rejected proof.

    Returns
    -------
    proof : GalerkinLocalStabilityProof
        Owner-factory-validated proof with canonical evidence digest.
    """
    source = certificate.source
    identity = _result_identity_digest(certificate, solve_result)
    overflow = exact_work_count > _MAXIMUM_SIGNED_INT64
    stored_work_count = 0 if overflow else exact_work_count
    budget_fraction = Fraction.from_float(maximum_state_error)
    eligible = finite_values is not None
    matrix_eligible = matrix_floor is not None or eligible
    if finite_values is not None:
        values = finite_values
        state_radius = values["state_radius_upper"]
        operational = state_radius <= budget_fraction
        failure = (
            GalerkinLocalStabilityFailure.NONE
            if operational
            else GalerkinLocalStabilityFailure.STATE_BUDGET_MISSED
        )
        disposition = (
            GalerkinLocalStabilityDisposition.OPERATIONAL_PASS
            if operational
            else GalerkinLocalStabilityDisposition.FINITE_STATE_RADIUS_FALLBACK
        )
        scalar_values = {
            "lower_singular_bound": values["exact_floor"],
            "algebraic_residual_upper_bound": values[
                "algebraic_residual_upper"
            ],
            "field_norm_upper_bound": values["field_norm_upper"],
            "fixed_linear_operator_error_bound": values["fixed_linear_error"],
            "fixed_linear_state_transfer_upper_bound": values[
                "fixed_linear_state_transfer_upper"
            ],
            "total_source_error_upper_bound": values[
                "total_source_error_upper"
            ],
            "exact_target_residual_upper_bound": values[
                "exact_target_residual_upper"
            ],
            "state_radius_upper_bound": state_radius,
        }
    else:
        operational = False
        disposition = (
            GalerkinLocalStabilityDisposition.MATRIX_FLOOR_ONLY
            if matrix_eligible
            else GalerkinLocalStabilityDisposition.REJECTED
        )
        stored_floor = (
            matrix_floor if matrix_floor is not None else Fraction(0)
        )
        scalar_values = {
            "lower_singular_bound": stored_floor,
            "algebraic_residual_upper_bound": None,
            "field_norm_upper_bound": None,
            "fixed_linear_operator_error_bound": None,
            "fixed_linear_state_transfer_upper_bound": None,
            "total_source_error_upper_bound": None,
            "exact_target_residual_upper_bound": None,
            "state_radius_upper_bound": None,
        }
        values = {
            "exact_floor": stored_floor,
            "residual_squared": Fraction(0),
            "field_norm_squared": Fraction(0),
            "algebraic_residual_upper": Fraction(0),
            "field_norm_upper": Fraction(0),
            "fixed_linear_error": Fraction(0),
            "fixed_linear_state_transfer_upper": Fraction(0),
            "total_source_error_upper": Fraction(0),
            "exact_target_residual_upper": Fraction(0),
            "state_radius_upper": Fraction(0),
        }
    transcript_fields: dict[str, int] = {}
    for prefix in (
        "exact_floor",
        "residual_squared",
        "field_norm_squared",
        "algebraic_residual_upper",
        "field_norm_upper",
        "fixed_linear_error",
        "fixed_linear_state_transfer_upper",
        "total_source_error_upper",
        "exact_target_residual_upper",
        "state_radius_upper",
    ):
        transcript_fields.update(_fraction_fields(prefix, values[prefix]))
    transcript_fields.update(
        _fraction_fields("maximum_state_error", budget_fraction)
    )
    environment_digest = sha256(
        {
            "domain": "ptyrodactyl.local_stability.environment.v1",
            "environment": environment_payload,
        }
    )

    def report(value: Fraction | None) -> jax.Array:
        """Convert one exact report or rejection sentinel to float64."""
        output = math.inf if value is None else float(value)
        return jnp.asarray(output, dtype=jnp.float64)

    reports = (
        report(scalar_values["lower_singular_bound"]),
        report(scalar_values["algebraic_residual_upper_bound"]),
        report(scalar_values["field_norm_upper_bound"]),
        report(scalar_values["fixed_linear_operator_error_bound"]),
        report(scalar_values["fixed_linear_state_transfer_upper_bound"]),
        report(scalar_values["total_source_error_upper_bound"]),
        report(scalar_values["exact_target_residual_upper_bound"]),
        report(scalar_values["state_radius_upper_bound"]),
        jnp.asarray(maximum_state_error, dtype=jnp.float64),
    )
    work_count = jnp.asarray(stored_work_count, dtype=jnp.int64)
    work_budget = jnp.asarray(maximum_direct_pairs, dtype=jnp.int64)
    flags = (
        jnp.asarray(matrix_eligible, dtype=jnp.bool_),
        jnp.asarray(eligible, dtype=jnp.bool_),
        jnp.asarray(operational, dtype=jnp.bool_),
        jnp.asarray(host_supported, dtype=jnp.bool_),
        jnp.asarray(normal_supported, dtype=jnp.bool_),
    )

    def make_proof(evidence_digest: str) -> GalerkinLocalStabilityProof:
        """Build the proof through its route-owning private factory."""
        return _make_local_stability_proof(
            reports,
            work_count,
            work_budget,
            flags,
            transcript_fields,
            direct_work_count_exact=str(exact_work_count),
            route=_ROUTE,
            disposition=disposition,
            failure=failure,
            checker_id=_CHECKER_ID,
            residual_formula=_RESIDUAL_FORMULA,
            residual_lift_formula=_RESIDUAL_LIFT_FORMULA,
            floor_scope=_FLOOR_SCOPE,
            error_scope=_ERROR_SCOPE,
            coefficient_norm=_COEFFICIENT_NORM,
            target_digest=source.target.target_digest,
            source_digest=source.source_digest,
            certificate_digest=certificate.certificate_digest,
            result_identity_digest=identity,
            arithmetic_environment_digest=environment_digest,
            proof_evidence_digest=evidence_digest,
        )

    proof = make_proof("0" * 64)
    digest = _proof_evidence_digest(
        certificate, solve_result, proof, environment_payload
    )
    return make_proof(digest)


def _finite_fraction(value: jax.Array, name: str) -> tuple[float, Fraction]:
    """PRIVATE: Read one nonnegative finite normal-or-zero float64 scalar.

    Parameters
    ----------
    value : jax.Array
        Submitted scalar array.
    name : str
        Evidence-field name used in diagnostics.

    Returns
    -------
    scalar : float
        Concrete validated binary64 value.
    exact : Fraction
        Exact dyadic rational represented by the stored binary64 bits.
    """
    host = host_array(value)
    _raise_if(
        host.dtype != np.dtype(np.float64) or host.shape != (),
        f"{name} must be a float64 scalar",
    )
    scalar = float(host)
    _raise_if(
        not math.isfinite(scalar)
        or scalar < 0.0
        or bool(has_subnormal_components(value)),
        f"{name} must be finite nonnegative normal-or-zero binary64",
    )
    return scalar, fraction_from_float(scalar)


def _exact_matrix_floor_outcome(
    certificate: GalerkinLocalRepresentedSourceCertificate,
) -> tuple[Fraction | None, GalerkinLocalStabilityFailure | None]:
    """PRIVATE: Read the independently authenticated exact L4 floor.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Complete replayed represented-source certificate.

    Returns
    -------
    floor : Fraction | None
        Positive exact dyadic L4 physical floor when matrix-eligible.
    failure : GalerkinLocalStabilityFailure | None
        Typed floor failure, or ``None`` when the floor is valid.
    """
    floor_proof = certificate.source.target.cap_floor_proof
    if not bool(floor_proof.exact_target_floor_eligible):
        return (
            None,
            GalerkinLocalStabilityFailure.EXACT_TARGET_FLOOR_UNAVAILABLE,
        )
    try:
        floor_float, floor = _finite_fraction(
            floor_proof.exact_target_physical_floor_lower_bound,
            "exact_target_physical_floor_lower_bound",
        )
    except ValueError:
        return (
            None,
            GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE,
        )
    if floor_float <= 0.0:
        return (
            None,
            GalerkinLocalStabilityFailure.NONPOSITIVE_EXACT_TARGET_FLOOR,
        )
    return floor, None


def _check_canonical_local_stability(  # noqa: PLR0911,PLR0912
    canonical: GalerkinLocalRepresentedSourceCertificate,
    solve: GalerkinSolveResult,
    state_budget: float,
    pair_budget: int,
) -> GalerkinLocalStabilityProof:
    """PRIVATE: Check one canonical parent, solve, and policy tuple.

    Parameters
    ----------
    canonical : GalerkinLocalRepresentedSourceCertificate
        Fully replayed local represented-source certificate.
    solve : GalerkinSolveResult
        Fully validated submitted algebraic solve carrier.
    state_budget : float
        Independently validated coefficient-l2 state policy budget.
    pair_budget : int
        Independently validated dense-work policy budget.

    Returns
    -------
    proof : GalerkinLocalStabilityProof
        Reconstructed raw local stability proof storage.

    Raises
    ------
    AssertionError
        If an internally inconsistent exact-floor outcome is encountered.
    """
    target_size = canonical.source.target.state_indices.shape[0]
    environment, host_supported, normal_supported = _environment_payload()
    exact_work_count = _direct_work_count(target_size)
    floor, floor_failure = _exact_matrix_floor_outcome(canonical)
    if exact_work_count > _MAXIMUM_SIGNED_INT64:
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=(GalerkinLocalStabilityFailure.DIRECT_WORK_COUNT_OVERFLOW),
            matrix_floor=floor,
        )
    if floor_failure is not None:
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=floor_failure,
        )
    if floor is None:
        raise AssertionError("positive exact L4 floor outcome lost its value")
    if not (host_supported and normal_supported):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=(
                GalerkinLocalStabilityFailure.HOST_ARITHMETIC_UNSUPPORTED
            ),
            matrix_floor=floor,
        )
    if exact_work_count > pair_budget:
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=(
                GalerkinLocalStabilityFailure.DIRECT_WORK_BUDGET_EXCEEDED
            ),
            matrix_floor=floor,
        )
    if not bool(canonical.finite_certificate):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.SOURCE_NONCERTIFICATE,
            matrix_floor=floor,
        )
    try:
        _, delta_h = _finite_fraction(
            canonical.source.target.fixed_linear_error_ledger.fixed_linear_operator_error_bound,
            "fixed_linear_operator_error_bound",
        )
        _, eta_t = _finite_fraction(
            canonical.total_source_error_upper_bound,
            "total_source_error_upper_bound",
        )
        residual_squared, field_squared = _exact_dense_residual_squared(
            canonical, solve
        )
        rho_root = sqrt_fraction_upper(residual_squared)
        field_root = sqrt_fraction_upper(field_squared)
    except RootEnclosureError:
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.ROOT_ENCLOSURE_FAILURE,
            matrix_floor=floor,
        )
    except (ArithmeticError, IndexError, TypeError, ValueError):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE,
            matrix_floor=floor,
        )
    rho_float = fraction_upper_float(rho_root)
    field_norm_float = fraction_upper_float(field_root)
    range_values = (rho_float, field_norm_float)
    if any(not _finite_normal_or_zero_float(value) for value in range_values):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE,
            matrix_floor=floor,
        )
    rho = Fraction.from_float(rho_float)
    field_norm = Fraction.from_float(field_norm_float)
    transfer_float = fraction_upper_float(delta_h * field_norm)
    if not _finite_normal_or_zero_float(transfer_float):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE,
            matrix_floor=floor,
        )
    transfer = Fraction.from_float(transfer_float)
    residual_float = fraction_upper_float(rho + transfer + eta_t)
    if not _finite_normal_or_zero_float(residual_float):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE,
            matrix_floor=floor,
        )
    exact_target_residual = Fraction.from_float(residual_float)
    radius_float = fraction_upper_float(exact_target_residual / floor)
    if not _finite_normal_or_zero_float(radius_float):
        return _base_proof(
            canonical,
            solve,
            state_budget,
            pair_budget,
            exact_work_count,
            environment,
            host_supported,
            normal_supported,
            failure=GalerkinLocalStabilityFailure.ARITHMETIC_RANGE_FAILURE,
            matrix_floor=floor,
        )
    radius = Fraction.from_float(radius_float)
    values = {
        "exact_floor": floor,
        "residual_squared": residual_squared,
        "field_norm_squared": field_squared,
        "algebraic_residual_upper": rho,
        "field_norm_upper": field_norm,
        "fixed_linear_error": delta_h,
        "fixed_linear_state_transfer_upper": transfer,
        "total_source_error_upper": eta_t,
        "exact_target_residual_upper": exact_target_residual,
        "state_radius_upper": radius,
    }
    return _base_proof(
        canonical,
        solve,
        state_budget,
        pair_budget,
        exact_work_count,
        environment,
        host_supported,
        normal_supported,
        failure=GalerkinLocalStabilityFailure.NONE,
        finite_values=values,
    )


def check_local_represented_galerkin_absorber_floor(
    certificate: object,
    solve_result: object,
    *,
    maximum_state_error: object,
    maximum_direct_pairs: int = _DEFAULT_MAXIMUM_DIRECT_PAIRS,
) -> GalerkinLocalStabilityProof:
    r"""Build one bounded exact-dyadic proof for a submitted state.

    :see: :func:`~.test_local_stability.\
test_exact_dense_residual_lift_budget_fallback_and_report_isolation`

    ``solve_result.residual`` and every reported residual norm are ignored by
    the mathematical calculation.  They remain bound in evidence only.

    Parameters
    ----------
    certificate : object
        A raw local represented-source certificate. Complete parent replay
        occurs before the dense residual budget starts.
    solve_result : object
        Submitted complex128 state and provenance carrier. Its field is used;
        its reported residual and metrics are not mathematical inputs.
    maximum_state_error : object
        Positive normal float64 budget in the coefficient l2 state norm.
    maximum_direct_pairs : int, optional
        Bound only for the post-parent-replay ``n + 2 n**2`` exact-dyadic
        dense residual work. It excludes target and source authentication.

    Returns
    -------
    proof : GalerkinLocalStabilityProof
        Unauthenticated raw proof storage. Pass it to
        :func:`invoke_local_represented_galerkin_stability` before use.

    Notes
    -----
    The L4 floor and ``delta_H`` have inverse-square Angstrom units; the
    residual has the corresponding action units. Dividing them yields the
    Euclidean coefficient-norm state radius.
    """
    canonical = prepare_local_represented_source_certificate(certificate)
    solve = _validate_solve_result(
        solve_result, canonical.source.target.state_indices.shape[0]
    )
    state_budget = _checked_positive_budget(
        maximum_state_error, "maximum_state_error"
    )
    pair_budget = _checked_pair_budget(maximum_direct_pairs)
    return _check_canonical_local_stability(
        canonical,
        solve,
        state_budget,
        pair_budget,
    )


def _result_evidence_digest(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    solve_result: GalerkinSolveResult,
    proof: GalerkinLocalStabilityProof,
) -> str:
    """PRIVATE: Bind every nested result field and completion scope.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Complete replayed represented-source certificate.
    solve_result : GalerkinSolveResult
        Full submitted algebraic solve payload.
    proof : GalerkinLocalStabilityProof
        Fully reconstructed and authenticated proof.

    Returns
    -------
    digest : str
        Canonical lowercase SHA-256 result-evidence digest.
    """
    payload: dict[str, object] = {
        "domain": _RESULT_DIGEST_DOMAIN,
        "certificate": stored_value_payload(certificate),
        "solve_result": stored_value_payload(solve_result),
        "proof": stored_value_payload(proof),
        "result_identity_digest": proof.result_identity_digest,
        "completion_scope": _COMPLETION_SCOPE,
    }
    return sha256(payload)


def invoke_local_represented_galerkin_stability(
    certificate: object,
    solve_result: object,
    proof: object,
    *,
    maximum_state_error: object,
    maximum_direct_pairs: int = _DEFAULT_MAXIMUM_DIRECT_PAIRS,
) -> GalerkinLocalStabilityResult:
    """Recheck the proof and return a fully nested stability result.

    :see: :func:`~.test_local_stability.\
test_full_replay_rejects_state_budget_proof_and_result_cross_pairs`

    The submitted proof is unauthenticated until this function independently
    repeats parent replay, bounded dense arithmetic, and exact comparison.
    Both budgets are independent caller policy inputs; neither is read from
    the untrusted proof. The state and radius use the coefficient l2 norm;
    the floor and ``delta_H`` use inverse-square Angstrom units, and the
    residual uses the corresponding action units. ``maximum_direct_pairs``
    bounds only post-parent-replay dense ``n + 2 n**2`` construction, not
    target or represented-source authentication.
    """
    if not isinstance(proof, GalerkinLocalStabilityProof):
        raise TypeError("proof must be GalerkinLocalStabilityProof")
    canonical = prepare_local_represented_source_certificate(certificate)
    solve = _validate_solve_result(
        solve_result, canonical.source.target.state_indices.shape[0]
    )
    _assert_concrete(proof)
    budget = _checked_positive_budget(
        maximum_state_error, "maximum_state_error"
    )
    maximum_pairs = _checked_pair_budget(maximum_direct_pairs)
    expected = _check_canonical_local_stability(
        canonical,
        solve,
        budget,
        maximum_pairs,
    )
    if stored_value_payload(expected) != stored_value_payload(proof):
        raise ValueError(
            "local stability proof does not match complete replay"
        )
    digest = _result_evidence_digest(canonical, solve, expected)
    return _make_local_stability_result(
        canonical,
        solve,
        expected,
        result_identity_digest=expected.result_identity_digest,
        result_evidence_digest=digest,
        completion_scope=_COMPLETION_SCOPE,
    )


def prepare_local_galerkin_stability_result(
    result: object,
    *,
    maximum_state_error: object,
    maximum_direct_pairs: int = _DEFAULT_MAXIMUM_DIRECT_PAIRS,
) -> GalerkinLocalStabilityResult:
    """Full-replay and exact-compare one local stability result.

    :see: :func:`~.test_local_stability.\
test_full_replay_rejects_state_budget_proof_and_result_cross_pairs`

    Raw result storage is unauthenticated until this boundary replays the
    complete certificate, solve payload, exact residual, and proof against
    the independently supplied coefficient-l2 state budget and direct-work
    budget. The direct-work budget covers only the dense ``n + 2 n**2``
    reconstruction after parent authentication. Floor and ``delta_H`` use
    inverse-square Angstrom units; the residual uses action units.
    """
    if not isinstance(result, GalerkinLocalStabilityResult):
        raise TypeError("result must be GalerkinLocalStabilityResult")
    _assert_concrete(result)
    canonical = invoke_local_represented_galerkin_stability(
        result.certificate,
        result.solve_result,
        result.proof,
        maximum_state_error=maximum_state_error,
        maximum_direct_pairs=maximum_direct_pairs,
    )
    if stored_value_payload(canonical) != stored_value_payload(result):
        raise ValueError(
            "local stability result does not match complete replay"
        )
    return canonical


__all__: list[str] = [
    "check_local_represented_galerkin_absorber_floor",
    "invoke_local_represented_galerkin_stability",
    "prepare_local_galerkin_stability_result",
]
