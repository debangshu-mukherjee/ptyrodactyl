r"""Check and invoke a bounded exact SC-1 stability certificate.

Extended Summary
----------------
This host-side module implements a bounded trusted Route-A checker for one
retained scalar Galerkin result. It reconstructs the manifested matrix and
the stored finite right-hand side from exact dyadic binary64 values. It
proves an absorber floor with rational Gershgorin bounds and encloses the
same-target residual without floating arithmetic. Each payload binds one
target, source, submitted state, and independently supplied state budget.

Routine Listings
----------------
:func:`check_galerkin_absorber_floor`
    Build an exact bounded Route-A proof for one submitted result.
:func:`invoke_galerkin_stability`
    Recheck and apply one per-result bound as pass, fallback, or rejection.

Notes
-----
This is a bounded dense checker, not a scalable interval eigensolver or a
universal stability theorem. It fails closed above its admitted dimension.
Its exact arithmetic treats each stored binary64 component as its exact
dyadic value. The SHA-256 values are provenance checksums; checker
reconstruction establishes target identity. Invoke the checker separately
for every retained result.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Num, Shaped
from numpy.typing import NDArray

from ptyrodactyl.types import (
    GalerkinSolveResult,
    GalerkinSource,
    GalerkinStabilityDisposition,
    GalerkinStabilityFailure,
    GalerkinStabilityProof,
    GalerkinStabilityResult,
    GalerkinStabilityRoute,
    GalerkinTargetManifest,
    create_galerkin_product_support,
    create_galerkin_solve_result,
    create_galerkin_source,
    create_galerkin_stability_proof,
    create_galerkin_stability_result,
    create_galerkin_target_manifest,
    scalar_float,
)

_CHECKER_ID: str = "ptyrodactyl.exact_dyadic_route_a.v1"
_MAX_EXACT_DIMENSION: int = 32
_MAX_BINARY64: float = float.fromhex("0x1.fffffffffffffp+1023")
_INVALID_TARGET_DIGEST: str = "0" * 64
_INVALID_RESULT_DIGEST: str = "f" * 64
type _ComplexFraction = tuple[Fraction, Fraction]


def _complex_add(
    left: _ComplexFraction,
    right: _ComplexFraction,
) -> _ComplexFraction:
    """Add two exact complex rationals."""
    result: _ComplexFraction = (left[0] + right[0], left[1] + right[1])
    return result


def _complex_subtract(
    left: _ComplexFraction,
    right: _ComplexFraction,
) -> _ComplexFraction:
    """Subtract two exact complex rationals."""
    result: _ComplexFraction = (left[0] - right[0], left[1] - right[1])
    return result


def _complex_multiply(
    left: _ComplexFraction,
    right: _ComplexFraction,
) -> _ComplexFraction:
    """Multiply two exact complex rationals."""
    result: _ComplexFraction = (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )
    return result


def _complex_conjugate(value: _ComplexFraction) -> _ComplexFraction:
    """Conjugate one exact complex rational."""
    result: _ComplexFraction = (value[0], -value[1])
    return result


def _fraction_from_float(value: float) -> Fraction:
    """Convert one finite binary float to its exact dyadic rational."""
    result: Fraction = Fraction.from_float(float(value))
    return result


def _complex_fraction(value: complex) -> _ComplexFraction:
    """Convert one finite binary complex value to exact dyadic components."""
    result: _ComplexFraction = (
        _fraction_from_float(float(value.real)),
        _fraction_from_float(float(value.imag)),
    )
    return result


def _host_array(value: jax.Array) -> Shaped[NDArray, "..."]:
    """Transfer one JAX array to a read-only host NumPy value."""
    array: Shaped[NDArray, "..."] = np.asarray(jax.device_get(value))
    return array


def _array_payload(value: jax.Array) -> dict[str, object]:
    """Return a canonical dtype-, shape-, and byte-bound array payload."""
    array: Shaped[NDArray, "..."] = _host_array(value)
    contiguous: Shaped[NDArray, "..."] = np.ascontiguousarray(array)
    payload: dict[str, object] = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes": contiguous.tobytes(order="C").hex(),
    }
    return payload


def _sha256(payload: dict[str, object]) -> str:
    """Hash one canonical JSON payload as a provenance checksum."""
    encoded: bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest: str = hashlib.sha256(encoded).hexdigest()
    return digest


def _target_payload(manifest: GalerkinTargetManifest) -> dict[str, object]:
    """Return every manifested-target field as an exact payload."""
    support = manifest.support
    payload: dict[str, object] = {
        "target_name": manifest.target_name,
        "contract_version": manifest.contract_version,
        "coefficient_normalization": manifest.coefficient_normalization,
        "precision": manifest.precision,
        "absorber_profile": manifest.absorber_profile,
        "absorber_coefficient_provenance": (
            manifest.absorber_coefficient_provenance
        ),
        "interaction_coefficient_provenance": (
            manifest.interaction_coefficient_provenance
        ),
        "work_shape": list(support.work_shape),
        "state_indices": _array_payload(support.state_indices),
        "interaction_indices": _array_payload(support.interaction_indices),
        "absorber_indices": _array_payload(support.absorber_indices),
        "work_indices": _array_payload(support.work_indices),
        "preterminal_indices": _array_payload(manifest.preterminal_indices),
        "voltage_coefficients": _array_payload(manifest.voltage_coefficients),
        "interaction_coefficients": _array_payload(
            manifest.interaction_coefficients
        ),
        "interaction_coupling": _array_payload(manifest.interaction_coupling),
        "absorber_coefficients": _array_payload(
            manifest.absorber_coefficients
        ),
        "free_diagonal": _array_payload(manifest.free_diagonal),
        "carrier": _array_payload(manifest.carrier),
        "box_lengths": _array_payload(manifest.box_lengths),
        "wavenumber": _array_payload(manifest.wavenumber),
        "accelerating_voltage_kv": _array_payload(
            manifest.accelerating_voltage_kv
        ),
        "cap_scale": _array_payload(manifest.cap_scale),
    }
    return payload


def _target_digest(manifest: GalerkinTargetManifest) -> str:
    """Compute the canonical manifested-target checksum."""
    payload: dict[str, object] = _target_payload(manifest)
    digest: str = _sha256(payload)
    return digest


def _source_payload(source: GalerkinSource) -> dict[str, object]:
    """Return every finite-source carrier field as an exact payload."""
    payload: dict[str, object] = {
        "branch": source.branch.value,
        "incident_field": _array_payload(source.incident_field),
        "incident_source": _array_payload(source.incident_source),
        "additional_source": _array_payload(source.additional_source),
        "total_source": _array_payload(source.total_source),
        "scattered_source": _array_payload(source.scattered_source),
    }
    return payload


def _solve_result_payload(
    solve_result: GalerkinSolveResult,
) -> dict[str, object]:
    """Return every algebraic solve-result field as an exact payload."""
    payload: dict[str, object] = {
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
    source: GalerkinSource,
    solve_result: GalerkinSolveResult,
) -> str:
    """Compute the source- and submitted-state-bound result checksum."""
    payload: dict[str, object] = {
        "target_digest": target_digest,
        "source": _source_payload(source),
        "solve_result": _solve_result_payload(solve_result),
    }
    digest: str = _sha256(payload)
    return digest


def _manifest_is_canonical(manifest: GalerkinTargetManifest) -> bool:
    """Rebuild and compare every factory-owned SC-1 manifest field."""
    try:
        support = create_galerkin_product_support(
            state_indices=manifest.support.state_indices,
            interaction_indices=manifest.support.interaction_indices,
            absorber_indices=manifest.support.absorber_indices,
            work_indices=manifest.support.work_indices,
            work_shape=manifest.support.work_shape,
        )
        canonical = create_galerkin_target_manifest(
            support=support,
            preterminal_indices=manifest.preterminal_indices,
            voltage_coefficients=manifest.voltage_coefficients,
            box_lengths=manifest.box_lengths,
            carrier=manifest.carrier,
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
    """Rebuild and compare every factory-owned source-carrier field."""
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


def _solve_result_is_canonical(
    solve_result: GalerkinSolveResult,
) -> bool:
    """Rebuild and compare every factory-owned solve-result field."""
    try:
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
    indices: jax.Array,
    coefficients: jax.Array,
) -> dict[tuple[int, int, int], _ComplexFraction]:
    """Map exact reciprocal indices to exact dyadic coefficients."""
    index_array: Num[NDArray, "... 3"] = _host_array(indices)
    coefficient_array: Num[NDArray, "..."] = _host_array(coefficients)
    mapping: dict[tuple[int, int, int], _ComplexFraction] = {}
    for index, coefficient in zip(index_array, coefficient_array, strict=True):
        key = (int(index[0]), int(index[1]), int(index[2]))
        mapping[key] = _complex_fraction(complex(coefficient))
    return mapping


def _matrix_from_coefficients(
    state_indices: Num[NDArray, "n 3"],
    mapping: dict[tuple[int, int, int], _ComplexFraction],
) -> list[list[_ComplexFraction]]:
    """Assemble one exact compressed multiplier matrix."""
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
    """Return whether one exact complex-rational matrix is Hermitian."""
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
    """Return a rational Gershgorin lower bound for one Hermitian absorber."""
    row_bounds: list[Fraction] = []
    for row, values in enumerate(absorber):
        diagonal: _ComplexFraction = values[row]
        if diagonal[1] != 0:
            zero_floor: Fraction = Fraction(0)
            return zero_floor
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


def _target_matrices(
    manifest: GalerkinTargetManifest,
) -> tuple[
    list[list[_ComplexFraction]],
    list[list[_ComplexFraction]],
    list[list[_ComplexFraction]],
]:
    """Reconstruct exact interaction, absorber, and target matrices."""
    state_indices: Num[NDArray, "n 3"] = _host_array(
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
    diagonal_array: Num[NDArray, " n"] = _host_array(manifest.free_diagonal)
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
    matrices: tuple[
        list[list[_ComplexFraction]],
        list[list[_ComplexFraction]],
        list[list[_ComplexFraction]],
    ] = (interaction, absorber, target)
    return matrices


def _exact_vector(value: jax.Array) -> list[_ComplexFraction]:
    """Convert one complex binary vector to exact dyadic components."""
    array: Num[NDArray, " n"] = _host_array(value)
    vector: list[_ComplexFraction] = [
        _complex_fraction(complex(entry)) for entry in array
    ]
    return vector


def _matrix_action(
    matrix: list[list[_ComplexFraction]],
    vector: list[_ComplexFraction],
) -> list[_ComplexFraction]:
    """Apply one exact complex-rational dense matrix."""
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
    source: GalerkinSource,
    solve_result: GalerkinSolveResult,
) -> Fraction:
    """Recompute the same-target residual squared in exact arithmetic."""
    right_hand_side = _exact_vector(source.total_source)
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


def _proof(
    target_digest: str,
    result_digest: str,
    floor: Fraction,
    residual_squared: Fraction,
    budget: Fraction,
    failure: GalerkinStabilityFailure,
) -> GalerkinStabilityProof:
    """Construct one canonical checker proof payload."""
    proof: GalerkinStabilityProof = create_galerkin_stability_proof(
        target_digest=target_digest,
        result_digest=result_digest,
        floor_numerator=max(floor.numerator, 0),
        floor_denominator=floor.denominator,
        residual_squared_numerator=residual_squared.numerator,
        residual_squared_denominator=residual_squared.denominator,
        state_budget_numerator=budget.numerator,
        state_budget_denominator=budget.denominator,
        route=GalerkinStabilityRoute.ABSORBER_FLOOR,
        failure=failure,
        checker_id=_CHECKER_ID,
    )
    return proof


@beartype
def check_galerkin_absorber_floor(  # noqa: PLR0911
    manifest: GalerkinTargetManifest,
    source: GalerkinSource,
    solve_result: GalerkinSolveResult,
    state_budget: scalar_float,
) -> GalerkinStabilityProof:
    r"""Build an exact bounded Route-A proof for one submitted result.

    :see: :class:`~.test_stability.TestGalerkinStabilityInvocation`

    Implementation Logic
    --------------------
    1. Bind the target, stored source, submitted state, and state budget.
    2. Reconstruct ``R``, ``A``, ``H``, and the residual with Fractions.
    3. Prove ``A >= mu I`` by rational Gershgorin lower bounds.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to reconstruct.
    source : GalerkinSource
        Source carrier whose stored total right-hand side is certified.
    solve_result : GalerkinSolveResult
        Submitted solver result. Its reported residual is not trusted.
    state_budget : scalar_float
        Positive normal-range preregistered state-error budget.

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
    The checker is bounded to 32 retained coefficients. Ordinary floating
    eigenvalues and producer Boolean assertions are not proof inputs. The
    payload is tied to this target, source, state, and budget; another result
    requires another invocation. RM-S3 source eligibility is tested
    separately and is not a premise of the exact residual enclosure for the
    stored right-hand side.
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
        )
        return proof  # noqa: RET504
    if dimension > _MAX_EXACT_DIMENSION:
        proof: GalerkinStabilityProof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.CHECKER_DIMENSION_LIMIT,
        )
        return proof

    if not _manifest_is_canonical(manifest):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_OPERATOR_CONTRACT,
        )
        return proof  # noqa: RET504

    if not _source_is_canonical(source) or not _solve_result_is_canonical(
        solve_result
    ):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT,
        )
        return proof  # noqa: RET504

    try:
        interaction, absorber, target = _target_matrices(manifest)
        residual_squared: Fraction = _residual_squared(
            target, source, solve_result
        )
    except (ArithmeticError, IndexError, TypeError, ValueError):
        proof = _proof(
            target_digest,
            result_digest,
            Fraction(0),
            Fraction(0),
            budget,
            GalerkinStabilityFailure.INVALID_SUBMISSION_CONTRACT,
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
        )
        return proof  # noqa: RET504
    absorber_floor: Fraction = _absorber_floor(absorber)
    cap: Fraction = _fraction_from_float(
        float(_host_array(manifest.cap_scale))
    )
    floor: Fraction = cap * absorber_floor
    failure: GalerkinStabilityFailure = (
        GalerkinStabilityFailure.NONE
        if cap > 0 and absorber_floor > 0
        else GalerkinStabilityFailure.NO_POSITIVE_ABSORBER_FLOOR
    )
    proof = _proof(
        target_digest,
        result_digest,
        floor,
        residual_squared,
        budget,
        failure,
    )
    return proof  # noqa: RET504


def _proof_payload(proof: GalerkinStabilityProof) -> tuple[object, ...]:
    """Return every proof field in canonical comparison order."""
    payload: tuple[object, ...] = (
        proof.target_digest,
        proof.result_digest,
        proof.floor_numerator,
        proof.floor_denominator,
        proof.residual_squared_numerator,
        proof.residual_squared_denominator,
        proof.state_budget_numerator,
        proof.state_budget_denominator,
        proof.route,
        proof.failure,
        proof.checker_id,
    )
    return payload


def _positive_fraction_to_float_down(value: Fraction) -> float:
    """Convert a positive rational to a downward-rounded binary float."""
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
    """Convert a non-negative rational square root upward to binary float."""
    if value == 0:
        root: float = 0.0
        return root
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


def _state_bound_to_float_up(
    residual_squared: Fraction,
    floor: Fraction,
) -> float:
    """Enclose ``sqrt(residual_squared) / floor`` upward."""
    if residual_squared == 0:
        state_bound: float = 0.0
        return state_bound
    squared_state_bound: Fraction = residual_squared / (floor**2)
    state_bound: float = _sqrt_fraction_to_float_up(squared_state_bound)
    return state_bound


def _rejected_result(
    expected: GalerkinStabilityProof,
    failure: GalerkinStabilityFailure,
) -> GalerkinStabilityResult:
    """Construct one fail-closed rejected invocation."""
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
        route=GalerkinStabilityRoute.ABSORBER_FLOOR,
        disposition=GalerkinStabilityDisposition.REJECTED,
        failure=failure,
        target_digest=expected.target_digest,
        result_digest=expected.result_digest,
        checker_id=_CHECKER_ID,
    )
    return result


@beartype
def invoke_galerkin_stability(
    manifest: GalerkinTargetManifest,
    source: GalerkinSource,
    solve_result: GalerkinSolveResult,
    proof: GalerkinStabilityProof,
    state_budget: scalar_float,
) -> GalerkinStabilityResult:
    r"""Recheck and apply one per-result bound as pass, fallback, or rejection.

    :see: :class:`~.test_stability.TestGalerkinStabilityInvocation`

    Implementation Logic
    --------------------
    1. Reconstruct the proof from the bound target, source, state, and budget.
    2. Reject every checksum, arithmetic, checker, or proof mutation.
    3. Compare the exact residual against the exact state-budget inequality.

    Parameters
    ----------
    manifest : GalerkinTargetManifest
        Canonical SC-1 target to reconstruct.
    source : GalerkinSource
        Bound finite matched source.
    solve_result : GalerkinSolveResult
        Bound submitted state and solver provenance.
    proof : GalerkinStabilityProof
        Checker payload to reconstruct and validate.
    state_budget : scalar_float
        Independently supplied preregistered positive normal-range state-error
        budget.

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
    ``residual_squared <= (state_budget * lower_bound)^2`` in exact rational
    arithmetic. Floating values in the result are outward reporting bounds.
    The invocation applies only to this retained result and is not reusable.
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
    expected: GalerkinStabilityProof = check_galerkin_absorber_floor(
        manifest,
        source,
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
    residual_squared: Fraction = Fraction(
        expected.residual_squared_numerator,
        expected.residual_squared_denominator,
    )
    budget_passes: bool = residual_squared <= (budget * floor) ** 2
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
    residual_upper: float = _sqrt_fraction_to_float_up(residual_squared)
    state_upper: float = _state_bound_to_float_up(residual_squared, floor)
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
        route=GalerkinStabilityRoute.ABSORBER_FLOOR,
        disposition=disposition,
        failure=failure,
        target_digest=expected.target_digest,
        result_digest=expected.result_digest,
        checker_id=_CHECKER_ID,
    )
    return result  # noqa: RET504


__all__: list[str] = [
    "check_galerkin_absorber_floor",
    "invoke_galerkin_stability",
]
