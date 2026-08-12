#!/usr/bin/env python3
"""Bounded pure-stdlib RM-I3 quotient and likelihood-metric oracle.

The oracle uses tiny real matrices to exercise the Moore--Penrose nuisance
quotient, range tests, shared-view stacking, rank jumps, clustered projectors,
and exact censored-Poisson Fisher weights.  Its Jacobi eigensolver is local,
deterministic, and bounded; no project package or numerical library is used.

Passing is falsification evidence only.  It proves neither RM-I3 nor any
production Jacobian, likelihood metric, rank threshold, range certificate,
or perturbation bound.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = SCRIPT_DIR / "rm_i3_quotient_cases.json"

MAX_ABS_INPUT = 1.0e6
MAX_CASES = 32
MAX_DIMENSION = 8
MAX_GRID_POINTS = 257
MAX_ID_LENGTH = 128
MAX_JACOBI_ITERATIONS = 20_000
MAX_POISSON_TERMS = 10_000
RANK_RELATIVE_TOLERANCE = 1.0e-11

Matrix = list[list[float]]
Vector = list[float]


class OracleDataError(ValueError):
    """Raised when a bounded fixture is malformed or outside oracle scope."""


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleDataError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject nonstandard JSON NaN and infinity constants."""
    raise OracleDataError(f"non-finite JSON constant {value!r} is forbidden")


def _finite_real(value: Any, context: str) -> float:
    """Parse one bounded finite JSON real, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OracleDataError(f"{context} must be a finite real")
    parsed = float(value)
    if not math.isfinite(parsed) or abs(parsed) > MAX_ABS_INPUT:
        raise OracleDataError(
            f"{context} must be finite with magnitude <= {MAX_ABS_INPUT}"
        )
    return parsed


def _positive(value: Any, context: str) -> float:
    """Parse one bounded strictly positive real."""
    parsed = _finite_real(value, context)
    if parsed <= 0.0:
        raise OracleDataError(f"{context} must be strictly positive")
    return parsed


def _exact_int(value: Any, context: str, lower: int, upper: int) -> int:
    """Parse one exact bounded Python integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleDataError(f"{context} must be an integer")
    if value < lower or value > upper:
        raise OracleDataError(f"{context} must lie in {lower}..{upper}")
    return value


def _case_id(case: dict[str, Any]) -> str:
    """Parse one bounded case identifier."""
    value = case.get("id")
    if not isinstance(value, str) or not value or len(value) > MAX_ID_LENGTH:
        raise OracleDataError(f"case id must contain 1..{MAX_ID_LENGTH} characters")
    return value


def _group(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return one nonempty bounded case group."""
    value = payload.get(name)
    if not isinstance(value, list) or not value or len(value) > MAX_CASES:
        raise OracleDataError(f"{name} must contain 1..{MAX_CASES} cases")
    if any(not isinstance(item, dict) for item in value):
        raise OracleDataError(f"every {name} entry must be an object")
    return value


def _matrix(value: Any, context: str, rows: int | None = None) -> Matrix:
    """Parse one nonempty bounded rectangular real matrix."""
    if not isinstance(value, list) or not value or len(value) > MAX_DIMENSION:
        raise OracleDataError(f"{context} must have 1..{MAX_DIMENSION} rows")
    if rows is not None and len(value) != rows:
        raise OracleDataError(f"{context} must have {rows} rows")
    if any(not isinstance(row, list) for row in value):
        raise OracleDataError(f"{context} rows must be arrays")
    columns = len(value[0])
    if columns > MAX_DIMENSION or any(len(row) != columns for row in value):
        raise OracleDataError(f"{context} must be rectangular and bounded")
    return [
        [_finite_real(item, f"{context}[{i}][{j}]") for j, item in enumerate(row)]
        for i, row in enumerate(value)
    ]


def _vector(value: Any, size: int, context: str) -> Vector:
    """Parse one fixed-size real vector."""
    if not isinstance(value, list) or len(value) != size:
        raise OracleDataError(f"{context} must contain exactly {size} entries")
    return [_finite_real(item, f"{context}[{i}]") for i, item in enumerate(value)]


def _zeros(rows: int, columns: int) -> Matrix:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def _identity(size: int) -> Matrix:
    result = _zeros(size, size)
    for index in range(size):
        result[index][index] = 1.0
    return result


def _transpose(matrix: Matrix) -> Matrix:
    columns = len(matrix[0])
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(columns)]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right:
        raise OracleDataError("matrix multiplication requires nonempty factors")
    inner = len(right)
    columns = len(right[0])
    if len(left[0]) != inner or any(len(row) != columns for row in right):
        raise OracleDataError("matrix multiplication dimensions disagree")
    return [
        [sum(left[i][k] * right[k][j] for k in range(inner)) for j in range(columns)]
        for i in range(len(left))
    ]


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    if len(matrix[0]) != len(vector):
        raise OracleDataError("matrix-vector dimensions disagree")
    return [sum(item * value for item, value in zip(row, vector)) for row in matrix]


def _subtract(left: Matrix, right: Matrix) -> Matrix:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise OracleDataError("matrix subtraction dimensions disagree")
    return [
        [left[i][j] - right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def _add(left: Matrix, right: Matrix) -> Matrix:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise OracleDataError("matrix addition dimensions disagree")
    return [
        [left[i][j] + right[i][j] for j in range(len(left[0]))]
        for i in range(len(left))
    ]


def _frobenius(matrix: Matrix) -> float:
    return math.sqrt(sum(item * item for row in matrix for item in row))


def _vector_norm(vector: Vector) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _outer(vector: Vector) -> Matrix:
    return [[left * right for right in vector] for left in vector]


def _jacobi_eigh(matrix: Matrix) -> tuple[Vector, Matrix]:
    """Diagonalize one tiny real symmetric matrix by bounded Jacobi rotations."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise OracleDataError("eigendecomposition requires a square matrix")
    scale = max(1.0, max(abs(item) for row in matrix for item in row))
    symmetry = max(
        (abs(matrix[i][j] - matrix[j][i]) for i in range(size) for j in range(size)),
        default=0.0,
    )
    if symmetry > 1.0e-12 * scale:
        raise OracleDataError("eigendecomposition matrix is not symmetric")
    work = [row[:] for row in matrix]
    vectors = _identity(size)
    for _ in range(MAX_JACOBI_ITERATIONS):
        largest = 0.0
        pivot = (0, 0)
        for i in range(size):
            for j in range(i + 1, size):
                candidate = abs(work[i][j])
                if candidate > largest:
                    largest = candidate
                    pivot = (i, j)
        if largest <= 1.0e-14 * scale:
            break
        p, q = pivot
        off = work[p][q]
        tau = (work[q][q] - work[p][p]) / (2.0 * off)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        app = work[p][p]
        aqq = work[q][q]
        work[p][p] = app - tangent * off
        work[q][q] = aqq + tangent * off
        work[p][q] = 0.0
        work[q][p] = 0.0
        for k in range(size):
            if k in (p, q):
                continue
            akp = work[k][p]
            akq = work[k][q]
            work[k][p] = cosine * akp - sine * akq
            work[p][k] = work[k][p]
            work[k][q] = sine * akp + cosine * akq
            work[q][k] = work[k][q]
        for k in range(size):
            vkp = vectors[k][p]
            vkq = vectors[k][q]
            vectors[k][p] = cosine * vkp - sine * vkq
            vectors[k][q] = sine * vkp + cosine * vkq
    else:
        raise ArithmeticError("bounded Jacobi eigensolver did not converge")
    order = sorted(range(size), key=lambda index: work[index][index], reverse=True)
    values = [work[index][index] for index in order]
    sorted_vectors = [[vectors[row][index] for index in order] for row in range(size)]
    return values, sorted_vectors


def _spectral_cutoff(values: Vector) -> float:
    return RANK_RELATIVE_TOLERANCE * max(1.0, max((abs(value) for value in values), default=0.0))


def _range_projector(matrix: Matrix) -> tuple[Matrix, int, Vector]:
    """Return the orthogonal projector onto the column range."""
    rows = len(matrix)
    columns = len(matrix[0])
    if columns == 0:
        return _zeros(rows, rows), 0, [0.0] * rows
    gram = _matmul(matrix, _transpose(matrix))
    values, vectors = _jacobi_eigh(gram)
    cutoff = _spectral_cutoff(values)
    if min(values) < -10.0 * cutoff:
        raise ArithmeticError("column Gram matrix is not positive semidefinite")
    active = [index for index, value in enumerate(values) if value > cutoff]
    projector = _zeros(rows, rows)
    for index in active:
        column = [vectors[row][index] for row in range(rows)]
        projector = _add(projector, _outer(column))
    return projector, len(active), values


def _symmetric_pseudoinverse(matrix: Matrix) -> tuple[Matrix, int]:
    values, vectors = _jacobi_eigh(matrix)
    cutoff = _spectral_cutoff(values)
    if min(values) < -10.0 * cutoff:
        raise ArithmeticError("information matrix is not positive semidefinite")
    inverse = _zeros(len(matrix), len(matrix))
    rank = 0
    for index, value in enumerate(values):
        if value <= cutoff:
            continue
        rank += 1
        column = [vectors[row][index] for row in range(len(matrix))]
        contribution = _outer(column)
        inverse = _add(
            inverse,
            [[item / value for item in row] for row in contribution],
        )
    return inverse, rank


def _effective_information(target: Matrix, nuisance: Matrix) -> tuple[Matrix, Matrix, int]:
    projector, rank, _ = _range_projector(nuisance)
    complement = _subtract(_identity(len(target)), projector)
    information = _matmul(_transpose(target), _matmul(complement, target))
    return information, projector, rank


def _schur_information(target: Matrix, nuisance: Matrix) -> Matrix:
    ata = _matmul(_transpose(target), target)
    if len(nuisance[0]) == 0:
        return ata
    atb = _matmul(_transpose(target), nuisance)
    btb = _matmul(_transpose(nuisance), nuisance)
    inverse, _ = _symmetric_pseudoinverse(btb)
    correction = _matmul(atb, _matmul(inverse, _transpose(atb)))
    return _subtract(ata, correction)


def _close(left: Matrix, right: Matrix, tolerance: float) -> bool:
    return _frobenius(_subtract(left, right)) <= tolerance * max(1.0, _frobenius(right))


def _quotient_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    target = _matrix(case.get("A"), f"{case_id}.A")
    nuisance = _matrix(case.get("B"), f"{case_id}.B", rows=len(target))
    tolerance = _positive(case.get("tolerance"), f"{case_id}.tolerance")
    kind = case.get("kind")
    allowed = {"b_zero", "range_contained", "orthogonal", "rank_deficient", "hidden_ridge"}
    if not isinstance(kind, str) or kind not in allowed:
        raise OracleDataError(f"{case_id}.kind is invalid")
    information, projector, rank = _effective_information(target, nuisance)
    schur = _schur_information(target, nuisance)
    ata = _matmul(_transpose(target), target)
    projector_defect = _frobenius(_subtract(_matmul(projector, projector), projector))
    schur_defect = _frobenius(_subtract(information, schur))
    values, _ = _jacobi_eigh(information)
    checks = [projector_defect <= tolerance, schur_defect <= 20.0 * tolerance, min(values) >= -20.0 * tolerance]
    extra: dict[str, Any] = {}
    if kind == "b_zero":
        checks.extend([rank == 0, _close(information, ata, tolerance)])
    elif kind == "range_contained":
        checks.append(_frobenius(information) <= 20.0 * tolerance)
    elif kind == "orthogonal":
        cross = _matmul(_transpose(target), nuisance)
        checks.extend([_frobenius(cross) <= tolerance, _close(information, ata, tolerance)])
    elif kind == "rank_deficient":
        expected = _exact_int(case.get("expected_rank"), f"{case_id}.expected_rank", 0, MAX_DIMENSION)
        checks.extend([rank == expected, expected < len(nuisance[0])])
    else:
        ridge = _positive(case.get("ridge"), f"{case_id}.ridge")
        minimum = _positive(
            case.get("minimum_spurious_information"),
            f"{case_id}.minimum_spurious_information",
        )
        btb = _matmul(_transpose(nuisance), nuisance)
        regularized = [row[:] for row in btb]
        for index in range(len(regularized)):
            regularized[index][index] += ridge
        inverse, _ = _symmetric_pseudoinverse(regularized)
        atb = _matmul(_transpose(target), nuisance)
        ridge_information = _subtract(ata, _matmul(atb, _matmul(inverse, _transpose(atb))))
        ridge_norm = _frobenius(ridge_information)
        checks.extend([_frobenius(information) <= tolerance, ridge_norm >= minimum])
        extra = {"ridge_information": ridge_information, "spurious_information_norm": ridge_norm}
    return {
        "id": case_id,
        "kind": kind,
        "nuisance_rank": rank,
        "effective_information": information,
        "projector_idempotence_defect": projector_defect,
        "schur_equivalence_defect": schur_defect,
        **extra,
        "passed": all(checks),
    }


def _stack_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    views = case.get("views")
    if not isinstance(views, list) or not views or len(views) > MAX_DIMENSION:
        raise OracleDataError(f"{case_id}.views must be a bounded nonempty array")
    targets: list[Matrix] = []
    nuisances: list[Matrix] = []
    viewwise: Matrix | None = None
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise OracleDataError(f"{case_id}.views[{index}] must be an object")
        target = _matrix(view.get("A"), f"{case_id}.views[{index}].A")
        nuisance = _matrix(view.get("B"), f"{case_id}.views[{index}].B", rows=len(target))
        targets.extend(target)
        nuisances.extend(nuisance)
        local, _, _ = _effective_information(target, nuisance)
        viewwise = local if viewwise is None else _add(viewwise, local)
    joint, _, _ = _effective_information(targets, nuisances)
    expected_joint = _matrix(case.get("expected_joint_information"), f"{case_id}.expected_joint")
    expected_viewwise = _matrix(
        case.get("expected_sum_of_viewwise_information"), f"{case_id}.expected_viewwise"
    )
    tolerance = _positive(case.get("tolerance"), f"{case_id}.tolerance")
    complete_viewwise = viewwise if viewwise is not None else [[0.0]]
    difference = _frobenius(_subtract(joint, complete_viewwise))
    passed = (
        _close(joint, expected_joint, tolerance)
        and _close(complete_viewwise, expected_viewwise, tolerance)
        and difference > tolerance
    )
    return {
        "id": case_id,
        "joint_information": joint,
        "sum_of_viewwise_quotients": complete_viewwise,
        "ordering_difference": difference,
        "passed": passed,
    }


def _functional_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    information = _matrix(case.get("information"), f"{case_id}.information")
    if len(information) != len(information[0]):
        raise OracleDataError(f"{case_id}.information must be square")
    functional = _vector(case.get("functional"), len(information), f"{case_id}.functional")
    tolerance = _positive(case.get("tolerance"), f"{case_id}.tolerance")
    expected = case.get("expected_supported")
    if not isinstance(expected, bool):
        raise OracleDataError(f"{case_id}.expected_supported must be boolean")
    projector, _, _ = _range_projector(information)
    residual = [a - b for a, b in zip(functional, _matvec(projector, functional))]
    residual_norm = _vector_norm(residual)
    supported = residual_norm <= tolerance * max(1.0, _vector_norm(functional))
    inverse, _ = _symmetric_pseudoinverse(information)
    denominator = sum(a * b for a, b in zip(functional, _matvec(inverse, functional)))
    strength = 1.0 / denominator if supported and denominator > tolerance else None
    checks = [supported is expected]
    if expected:
        expected_strength = _positive(case.get("expected_strength"), f"{case_id}.expected_strength")
        checks.append(strength is not None and abs(strength - expected_strength) <= tolerance)
    else:
        checks.append(strength is None)
    return {
        "id": case_id,
        "supported": supported,
        "range_residual_norm": residual_norm,
        "pseudoinverse_quadratic": denominator,
        "reported_strength": strength,
        "passed": all(checks),
    }


def _rank_jump_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    target = _matrix(case.get("A"), f"{case_id}.A")
    reference = _matrix(case.get("B_reference"), f"{case_id}.B_reference", rows=len(target))
    perturbed = _matrix(case.get("B_perturbed"), f"{case_id}.B_perturbed", rows=len(target))
    reference_information, reference_projector, reference_rank = _effective_information(target, reference)
    perturbed_information, perturbed_projector, perturbed_rank = _effective_information(target, perturbed)
    expected_reference = _exact_int(
        case.get("expected_reference_rank"), f"{case_id}.expected_reference_rank", 0, MAX_DIMENSION
    )
    expected_perturbed = _exact_int(
        case.get("expected_perturbed_rank"), f"{case_id}.expected_perturbed_rank", 0, MAX_DIMENSION
    )
    minimum_jump = _positive(
        case.get("minimum_effective_information_jump"),
        f"{case_id}.minimum_effective_information_jump",
    )
    information_jump = _frobenius(_subtract(reference_information, perturbed_information))
    projector_jump = _frobenius(_subtract(reference_projector, perturbed_projector))
    return {
        "id": case_id,
        "reference_rank": reference_rank,
        "perturbed_rank": perturbed_rank,
        "information_jump": information_jump,
        "projector_jump": projector_jump,
        "passed": (
            reference_rank == expected_reference
            and perturbed_rank == expected_perturbed
            and information_jump >= minimum_jump
        ),
    }


def _cluster_projector(vectors: Matrix, cluster_size: int) -> Matrix:
    size = len(vectors)
    result = _zeros(size, size)
    for index in range(size - cluster_size, size):
        column = [vectors[row][index] for row in range(size)]
        result = _add(result, _outer(column))
    return result


def _cluster_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    reference = _matrix(case.get("reference_information"), f"{case_id}.reference")
    perturbed = _matrix(case.get("perturbed_information"), f"{case_id}.perturbed", rows=len(reference))
    if len(reference) != len(reference[0]) or len(perturbed[0]) != len(reference):
        raise OracleDataError(f"{case_id} information matrices must be equal square matrices")
    cluster_size = _exact_int(
        case.get("cluster_size"), f"{case_id}.cluster_size", 2, len(reference)
    )
    _, reference_vectors = _jacobi_eigh(reference)
    _, perturbed_vectors = _jacobi_eigh(perturbed)
    reference_projector = _cluster_projector(reference_vectors, cluster_size)
    perturbed_projector = _cluster_projector(perturbed_vectors, cluster_size)
    matrix_change = _frobenius(_subtract(reference, perturbed))
    projector_change = _frobenius(_subtract(reference_projector, perturbed_projector))
    individual_changes = []
    for index in range(len(reference) - cluster_size, len(reference)):
        left = [reference_vectors[row][index] for row in range(len(reference))]
        right = [perturbed_vectors[row][index] for row in range(len(reference))]
        plus = _vector_norm([a + b for a, b in zip(left, right)])
        minus = _vector_norm([a - b for a, b in zip(left, right)])
        individual_changes.append(min(plus, minus))
    maximum_matrix = _positive(
        case.get("maximum_matrix_perturbation"), f"{case_id}.maximum_matrix_perturbation"
    )
    maximum_projector = _positive(
        case.get("maximum_projector_distance"), f"{case_id}.maximum_projector_distance"
    )
    minimum_vector = _positive(
        case.get("minimum_individual_vector_change"), f"{case_id}.minimum_individual_vector_change"
    )
    return {
        "id": case_id,
        "matrix_perturbation": matrix_change,
        "cluster_projector_distance": projector_change,
        "individual_vector_changes": individual_changes,
        "passed": (
            matrix_change <= maximum_matrix
            and projector_change <= maximum_projector
            and max(individual_changes) >= minimum_vector
        ),
    }


def _poisson_probabilities(mean: float, ceiling: int) -> tuple[Vector, float]:
    """Return ordinary masses below a positive ceiling and its upper tail."""
    masses: Vector = []
    term = math.exp(-mean)
    for count in range(ceiling):
        if count:
            term *= mean / count
        masses.append(term)
    tail_term = masses[-1] * mean / ceiling
    tail = 0.0
    for offset in range(MAX_POISSON_TERMS):
        tail += tail_term
        next_term = tail_term * mean / (ceiling + offset + 1)
        if next_term <= 2.0e-16 * max(tail, 1.0e-300):
            break
        tail_term = next_term
    else:
        raise ArithmeticError("bounded Poisson tail series did not converge")
    if not 0.0 < tail <= 1.0 + 1.0e-12:
        raise ArithmeticError("Poisson tail is not a positive probability")
    return masses, min(tail, 1.0)


def _censored_fisher(mean: float, ceiling: int | None) -> dict[str, float | None]:
    if ceiling is None:
        return {
            "fisher": 1.0 / mean,
            "expected_score": 0.0,
            "expected_nll_curvature": 1.0 / mean,
            "readout_mean": mean,
            "readout_variance": mean,
            "inverse_variance": 1.0 / mean,
            "mean_summary_information": 1.0 / mean,
        }
    if ceiling == 0:
        return {
            "fisher": 0.0,
            "expected_score": 0.0,
            "expected_nll_curvature": 0.0,
            "readout_mean": 0.0,
            "readout_variance": 0.0,
            "inverse_variance": None,
            "mean_summary_information": None,
        }
    masses, tail = _poisson_probabilities(mean, ceiling)
    tail_score = masses[-1] / tail
    scores = [count / mean - 1.0 for count in range(ceiling)] + [tail_score]
    probabilities = masses + [tail]
    fisher = sum(probability * score * score for probability, score in zip(probabilities, scores))
    expected_score = sum(probability * score for probability, score in zip(probabilities, scores))
    tail_curvature = tail_score * tail_score + (1.0 - (ceiling - 1) / mean) * tail_score
    curvatures = [count / (mean * mean) for count in range(ceiling)] + [tail_curvature]
    expected_curvature = sum(
        probability * curvature for probability, curvature in zip(probabilities, curvatures)
    )
    values = list(range(ceiling)) + [ceiling]
    readout_mean = sum(probability * value for probability, value in zip(probabilities, values))
    variance = sum(
        probability * (value - readout_mean) ** 2
        for probability, value in zip(probabilities, values)
    )
    sensitivity = sum(masses)
    return {
        "fisher": fisher,
        "expected_score": expected_score,
        "expected_nll_curvature": expected_curvature,
        "readout_mean": readout_mean,
        "readout_variance": variance,
        "inverse_variance": 1.0 / variance,
        "mean_summary_information": sensitivity * sensitivity / variance,
    }


def _ceiling(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _exact_int(value, context, 0, 64)


def _fisher_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    mean = _positive(case.get("mean"), f"{case_id}.mean")
    ceiling = _ceiling(case.get("count_ceiling"), f"{case_id}.count_ceiling")
    expected = _finite_real(case.get("expected_fisher"), f"{case_id}.expected_fisher")
    tolerance = _positive(case.get("tolerance"), f"{case_id}.tolerance")
    report = _censored_fisher(mean, ceiling)
    fisher = float(report["fisher"])
    checks = [abs(fisher - expected) <= tolerance * max(1.0, abs(expected))]
    checks.append(abs(float(report["expected_score"])) <= 20.0 * tolerance)
    checks.append(
        abs(float(report["expected_nll_curvature"]) - fisher)
        <= 50.0 * tolerance * max(1.0, fisher)
    )
    if ceiling is not None and ceiling >= 1:
        checks.extend([fisher > 0.0, fisher < 1.0 / mean])
    if case.get("require_inverse_variance_difference") is True:
        inverse_variance = float(report["inverse_variance"])
        checks.append(abs(inverse_variance - fisher) > 100.0 * tolerance)
    if case.get("require_mean_summary_strict_loss") is True:
        summary = float(report["mean_summary_information"])
        checks.append(summary < fisher - tolerance)
    return {"id": case_id, "count_ceiling": ceiling, **report, "passed": all(checks)}


def _fisher_interval_report(case: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(case)
    interval = case.get("mean_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise OracleDataError(f"{case_id}.mean_interval must contain two values")
    lower = _positive(interval[0], f"{case_id}.mean_interval[0]")
    upper = _positive(interval[1], f"{case_id}.mean_interval[1]")
    if lower > upper:
        raise OracleDataError(f"{case_id}.mean_interval must be ordered")
    points = _exact_int(case.get("grid_points"), f"{case_id}.grid_points", 3, MAX_GRID_POINTS)
    ceiling = _ceiling(case.get("count_ceiling"), f"{case_id}.count_ceiling")
    expected_positive = case.get("expect_strictly_positive")
    if not isinstance(expected_positive, bool):
        raise OracleDataError(f"{case_id}.expect_strictly_positive must be boolean")
    values = [
        float(_censored_fisher(lower + (upper - lower) * index / (points - 1), ceiling)["fisher"])
        for index in range(points)
    ]
    finite = all(math.isfinite(value) for value in values)
    positive = min(values) > 0.0
    if ceiling is not None and ceiling >= 1:
        below_complete = all(
            value < 1.0 / (lower + (upper - lower) * index / (points - 1))
            for index, value in enumerate(values)
        )
    else:
        below_complete = True
    passed = finite and positive is expected_positive and below_complete
    if ceiling == 0:
        passed = passed and max(abs(value) for value in values) == 0.0
    return {
        "id": case_id,
        "count_ceiling": ceiling,
        "sampled_minimum": min(values),
        "sampled_maximum": max(values),
        "all_finite": finite,
        "passed": passed,
    }


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise OracleDataError("fixture root must use schema_version 1")
    return payload


def run(path: Path) -> dict[str, Any]:
    """Evaluate every bounded fixture group."""
    payload = _load(path)
    reports = {
        "quotient_reports": [_quotient_report(case) for case in _group(payload, "quotient_cases")],
        "stack_reports": [_stack_report(case) for case in _group(payload, "stack_cases")],
        "functional_reports": [
            _functional_report(case) for case in _group(payload, "functional_cases")
        ],
        "rank_jump_reports": [
            _rank_jump_report(case) for case in _group(payload, "rank_jump_cases")
        ],
        "cluster_reports": [_cluster_report(case) for case in _group(payload, "cluster_cases")],
        "censored_fisher_reports": [
            _fisher_report(case) for case in _group(payload, "censored_fisher_cases")
        ],
        "fisher_interval_reports": [
            _fisher_interval_report(case) for case in _group(payload, "fisher_interval_cases")
        ],
    }
    all_reports = [item for group in reports.values() for item in group]
    return {
        "schema_version": 1,
        "fixture": str(path),
        "scope": (
            "bounded pure-stdlib falsification evidence; not proof or production "
            "likelihood/Jacobian/rank/range conformance"
        ),
        **reports,
        "all_passed": all(bool(report["passed"]) for report in all_reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="?", type=Path, default=DEFAULT_CASES)
    arguments = parser.parse_args()
    try:
        report = run(arguments.cases)
    except (ArithmeticError, OSError, OracleDataError, json.JSONDecodeError) as error:
        print(json.dumps({"all_passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

