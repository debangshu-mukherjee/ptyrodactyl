"""Tests for :mod:`ptyrodactyl.galerkin.engine`.

Extended Summary
----------------
Builds independent dense SC-1 matrices on fixed three- and four-mode
supports. The fixtures use full Hermitian interactions and positive absorber
Gramians, including a four-mode factor that does not commute with the
interaction.
"""

from collections.abc import Callable
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import Complex, Float
from numpy.typing import NDArray

from ptyrodactyl.galerkin import (
    apply_galerkin_adjoint,
    apply_galerkin_operator,
    cgls_solve,
    evaluate_galerkin_adjoint_residual,
    evaluate_galerkin_residual,
    implicit_galerkin_solve,
    lsqr_solve,
    shifted_free_diagonal,
)
from ptyrodactyl.types import (
    GalerkinOperator,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    create_galerkin_operator,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)

_RECIPROCAL_FREQUENCIES: Float[NDArray, "4 3"] = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.12, 0.0, 0.0],
        [0.0, -0.09, 0.04],
        [-0.08, 0.06, -0.03],
    ],
    dtype=np.float64,
)
_BASE_CARRIER: Float[NDArray, "3"] = np.array(
    [0.35, -0.25, 2.25],
    dtype=np.float64,
)
_WAVENUMBER = float(np.linalg.norm(_BASE_CARRIER))
_INTERACTION: Complex[NDArray, "4 4"] = np.array(
    [
        [0.45, 0.08 + 0.05j, -0.03 + 0.02j, 0.04j],
        [0.08 - 0.05j, 0.35, 0.06 - 0.03j, 0.02 + 0.01j],
        [-0.03 - 0.02j, 0.06 + 0.03j, 0.25, -0.07 + 0.02j],
        [-0.04j, 0.02 - 0.01j, -0.07 - 0.02j, 0.30],
    ],
    dtype=np.complex128,
)
_ABSORBER_FACTOR: Complex[NDArray, "4 4"] = np.array(
    [
        [0.65, 0.12 + 0.08j, -0.04j, 0.03],
        [0.02 - 0.03j, 0.72, -0.09 + 0.02j, 0.04j],
        [0.11, 0.03 - 0.06j, 0.58, 0.13 + 0.04j],
        [-0.02j, 0.08, 0.05 + 0.03j, 0.61],
    ],
    dtype=np.complex128,
)
_SOURCE: Complex[NDArray, "4"] = np.array(
    [1.0 + 0.2j, -0.3 + 0.4j, 0.25 - 0.15j, 0.1 + 0.05j],
    dtype=np.complex128,
)
_TERMINAL: Complex[NDArray, "4"] = np.array(
    [0.2 - 0.1j, -0.05 + 0.3j, 0.15 + 0.08j, -0.2 + 0.04j],
    dtype=np.complex128,
)
_TARGET = np.complex128(0.13 - 0.07j)
_CAP_SCALE = 0.27


class _ImplicitAdjointCase(NamedTuple):
    """Store one locally copied fixed-support RM-I1 oracle case."""

    case_id: str
    indices: Tuple[int, ...]
    length: float
    wavenumber: float
    interaction_coefficients: Dict[int, complex]
    absorber_coefficients: Dict[int, complex]
    source: Tuple[complex, ...]
    terminal: Tuple[complex, ...]
    target: complex
    parameters: Tuple[float, float, float, float, float]
    tolerance: float


_IMPLICIT_ADJOINT_CASES = (
    _ImplicitAdjointCase(
        case_id="three_mode_carrier_source_detector",
        indices=(-1, 0, 1),
        length=5.0,
        wavenumber=2.4,
        interaction_coefficients={
            0: 0.7 + 0.0j,
            1: 0.08 + 0.03j,
            -1: 0.08 - 0.03j,
        },
        absorber_coefficients={
            0: 1.0 + 0.0j,
            1: 0.12 - 0.02j,
            -1: 0.12 + 0.02j,
        },
        source=(1.0 + 0.2j, 0.3 - 0.1j, -0.15 + 0.25j),
        terminal=(0.6 + 0.1j, -0.2 + 0.4j, 0.3 - 0.25j),
        target=0.12 - 0.08j,
        parameters=(2.1, 0.9, 0.35, 1.2, 0.8),
        tolerance=3e-7,
    ),
    _ImplicitAdjointCase(
        case_id="four_mode_nonsymmetric_carrier_frame",
        indices=(-2, -1, 0, 1),
        length=7.0,
        wavenumber=1.9,
        interaction_coefficients={
            0: 0.45 + 0.0j,
            1: -0.04 + 0.06j,
            -1: -0.04 - 0.06j,
            2: 0.02 - 0.01j,
            -2: 0.02 + 0.01j,
        },
        absorber_coefficients={
            0: 0.8 + 0.0j,
            1: 0.05 + 0.01j,
            -1: 0.05 - 0.01j,
        },
        source=(0.2 - 0.1j, 0.8 + 0.0j, -0.3 + 0.2j, 0.1 + 0.05j),
        terminal=(0.1 + 0.3j, 0.4 - 0.2j, -0.15 + 0.1j, 0.2 + 0.25j),
        target=-0.04 + 0.15j,
        parameters=(1.55, 1.1, 0.22, 0.75, 1.25),
        tolerance=4e-7,
    ),
)

_INTERACTION_ROWS, _INTERACTION_COLUMNS = np.indices(_INTERACTION.shape)
_INTERACTION_ROWS = _INTERACTION_ROWS.reshape(-1).astype(np.int32)
_INTERACTION_COLUMNS = _INTERACTION_COLUMNS.reshape(-1).astype(np.int32)
_ABSORBER_ROWS, _ABSORBER_COLUMNS = np.indices(_ABSORBER_FACTOR.shape)
_ABSORBER_ROWS = _ABSORBER_ROWS.reshape(-1).astype(np.int32)
_ABSORBER_COLUMNS = _ABSORBER_COLUMNS.reshape(-1).astype(np.int32)


def _dense_operator(
    carrier: Float[NDArray, "3"] = _BASE_CARRIER,
    interaction_scale: float = 1.0,
    cap_scale: float = _CAP_SCALE,
) -> Complex[NDArray, "4 4"]:
    """Assemble the tiny dense target without production action helpers."""
    shifted = carrier[None, :] + 2.0 * np.pi * _RECIPROCAL_FREQUENCIES
    free_diagonal = np.sum(shifted**2, axis=-1) - _WAVENUMBER**2
    absorber = _ABSORBER_FACTOR.conj().T @ _ABSORBER_FACTOR
    dense_operator = (
        np.diag(free_diagonal)
        - interaction_scale * _INTERACTION
        - 1j * cap_scale * absorber
    )
    return dense_operator


def _create_operator(
    carrier: Float[NDArray, "3"] | jax.Array = _BASE_CARRIER,
    interaction_scale: float | jax.Array = 1.0,
    cap_scale: float | jax.Array = _CAP_SCALE,
) -> GalerkinOperator:
    """Construct the production carrier from the independent COO fixture."""
    free_diagonal = shifted_free_diagonal(
        jnp.asarray(_RECIPROCAL_FREQUENCIES),
        jnp.asarray(carrier),
        jnp.asarray(_WAVENUMBER),
    )
    operator = create_galerkin_operator(
        free_diagonal=free_diagonal,
        interaction_rows=jnp.asarray(_INTERACTION_ROWS),
        interaction_columns=jnp.asarray(_INTERACTION_COLUMNS),
        interaction_values=(
            jnp.asarray(interaction_scale)
            * jnp.asarray(_INTERACTION).reshape(-1)
        ),
        absorber_factor_rows=jnp.asarray(_ABSORBER_ROWS),
        absorber_factor_columns=jnp.asarray(_ABSORBER_COLUMNS),
        absorber_factor_values=jnp.asarray(_ABSORBER_FACTOR).reshape(-1),
        cap_scale=jnp.asarray(cap_scale),
        absorber_factor_size=_ABSORBER_FACTOR.shape[0],
    )
    return operator


def _operator_and_source(
    parameters: Float[jax.Array, "4"],
) -> Tuple[GalerkinOperator, Complex[jax.Array, "4"]]:
    """Map the four state-equation parameters to one fixed support."""
    carrier = jnp.asarray(_BASE_CARRIER).at[0].set(parameters[0])
    operator = _create_operator(
        carrier=carrier,
        interaction_scale=parameters[1],
        cap_scale=parameters[2],
    )
    source = parameters[3] * jnp.asarray(_SOURCE)
    mapped = (operator, source)
    return mapped


def _implicit_loss(
    parameters: Float[jax.Array, "5"],
    *,
    max_iterations: int = 64,
) -> Float[jax.Array, ""]:
    """Evaluate a real detector loss through the implicit root."""
    operator, source = _operator_and_source(parameters[:4])
    field = implicit_galerkin_solve(
        operator,
        source,
        max_iterations=max_iterations,
        relative_tolerance=1e-13,
        absolute_tolerance=1e-14,
    )
    amplitude = jnp.vdot(jnp.asarray(_TERMINAL), field)
    detector_residual = parameters[4] * amplitude - jnp.asarray(_TARGET)
    loss = 0.5 * jnp.real(detector_residual.conj() * detector_residual)
    return loss


def _reference_coefficient_matrix(
    case: _ImplicitAdjointCase,
    coefficients: Dict[int, complex],
) -> Complex[NDArray, "n n"]:
    """Assemble one dense multiplier from copied Fourier coefficients."""
    matrix = np.array(
        [
            [coefficients.get(row - column, 0.0j) for column in case.indices]
            for row in case.indices
        ],
        dtype=np.complex128,
    )
    return matrix


def _reference_operator_and_source(
    case: _ImplicitAdjointCase,
    parameters: Float[jax.Array, "5"],
) -> Tuple[GalerkinOperator, Complex[jax.Array, " n"]]:
    """Map one copied RM-I1 parameter chart into the production carrier."""
    state_size = len(case.indices)
    reciprocal_frequencies = np.zeros((state_size, 3), dtype=np.float64)
    reciprocal_frequencies[:, 0] = np.asarray(case.indices) / case.length
    carrier = jnp.stack((parameters[0], jnp.asarray(0.0), jnp.asarray(0.0)))
    free_diagonal = shifted_free_diagonal(
        jnp.asarray(reciprocal_frequencies),
        carrier,
        jnp.asarray(case.wavenumber),
    )
    interaction = _reference_coefficient_matrix(
        case,
        case.interaction_coefficients,
    )
    absorber = _reference_coefficient_matrix(
        case,
        case.absorber_coefficients,
    )
    absorber_factor = np.linalg.cholesky(absorber).conj().T
    rows, columns = np.indices((state_size, state_size))
    operator = create_galerkin_operator(
        free_diagonal=free_diagonal,
        interaction_rows=jnp.asarray(rows.reshape(-1), dtype=jnp.int32),
        interaction_columns=jnp.asarray(
            columns.reshape(-1),
            dtype=jnp.int32,
        ),
        interaction_values=(
            parameters[1] * jnp.asarray(interaction).reshape(-1)
        ),
        absorber_factor_rows=jnp.asarray(
            rows.reshape(-1),
            dtype=jnp.int32,
        ),
        absorber_factor_columns=jnp.asarray(
            columns.reshape(-1),
            dtype=jnp.int32,
        ),
        absorber_factor_values=jnp.asarray(absorber_factor).reshape(-1),
        cap_scale=parameters[2],
        absorber_factor_size=state_size,
    )
    source = parameters[3] * jnp.asarray(case.source)
    result = (operator, source)
    return result


def _reference_implicit_loss(
    case: _ImplicitAdjointCase,
    parameters: Float[jax.Array, "5"],
) -> Float[jax.Array, ""]:
    """Evaluate one copied RM-I1 loss through the production custom VJP."""
    operator, source = _reference_operator_and_source(case, parameters)
    field = implicit_galerkin_solve(
        operator,
        source,
        max_iterations=64,
        relative_tolerance=1e-13,
        absolute_tolerance=1e-14,
    )
    terminal_value = jnp.sum(jnp.asarray(case.terminal) * field)
    detector_residual = parameters[4] * terminal_value - jnp.asarray(
        case.target
    )
    loss = 0.5 * jnp.real(detector_residual.conj() * detector_residual)
    return loss


def _reference_dense_loss(
    case: _ImplicitAdjointCase,
    parameters: Float[NDArray, "5"],
) -> float:
    """Evaluate one copied RM-I1 loss through an independent dense solve."""
    indices = np.asarray(case.indices, dtype=np.float64)
    free_diagonal = (
        parameters[0] + 2.0 * np.pi * indices / case.length
    ) ** 2 - case.wavenumber**2
    interaction = _reference_coefficient_matrix(
        case,
        case.interaction_coefficients,
    )
    absorber = _reference_coefficient_matrix(
        case,
        case.absorber_coefficients,
    )
    dense_operator = (
        np.diag(free_diagonal)
        - parameters[1] * interaction
        - 1j * parameters[2] * absorber
    )
    source = parameters[3] * np.asarray(case.source)
    field = np.linalg.solve(dense_operator, source)
    terminal_value = np.sum(np.asarray(case.terminal) * field)
    detector_residual = parameters[4] * terminal_value - case.target
    loss = float(
        0.5 * np.real(detector_residual.conjugate() * detector_residual)
    )
    return loss


def _reference_central_gradient(
    case: _ImplicitAdjointCase,
    parameters: Float[NDArray, "5"],
    step: float,
) -> Float[NDArray, "5"]:
    """Differentiate one copied RM-I1 dense loss in every real block."""
    gradient = np.zeros_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            _reference_dense_loss(case, plus)
            - _reference_dense_loss(case, minus)
        ) / (2.0 * step)
    return gradient


def _realify_vector(
    values: Complex[NDArray, "n"],
) -> Float[NDArray, "two_n"]:
    """Map a complex vector into the declared real coordinate chart."""
    realified = np.concatenate((values.real, values.imag))
    return realified


def _realify_matrix(
    matrix: Complex[NDArray, "n n"],
) -> Float[NDArray, "two_n two_n"]:
    """Map a complex-linear matrix into the declared real chart."""
    realified = np.block(
        [
            [matrix.real, -matrix.imag],
            [matrix.imag, matrix.real],
        ]
    )
    return realified


def _jaxpr_array_shapes(value: object) -> Tuple[Tuple[int, ...], ...]:
    """Collect array shapes from a nested custom-VJP JAXPR."""
    shapes: list[Tuple[int, ...]] = []
    visited: set[int] = set()

    def visit(candidate: object) -> None:
        """Visit JAXPR containers, equations, variables, and parameters."""
        if id(candidate) in visited:
            return
        visited.add(id(candidate))
        nested_jaxpr = getattr(candidate, "jaxpr", None)
        if nested_jaxpr is not None and not hasattr(candidate, "eqns"):
            visit(nested_jaxpr)
            return
        if all(
            hasattr(candidate, attribute)
            for attribute in (
                "constvars",
                "invars",
                "outvars",
                "eqns",
            )
        ):
            constvars = getattr(candidate, "constvars", ())
            invars = getattr(candidate, "invars", ())
            outvars = getattr(candidate, "outvars", ())
            for variable in (
                *constvars,
                *invars,
                *outvars,
            ):
                visit(variable)
            for equation in getattr(candidate, "eqns", ()):
                equation_invars = getattr(equation, "invars", ())
                equation_outvars = getattr(equation, "outvars", ())
                for variable in (*equation_invars, *equation_outvars):
                    visit(variable)
                visit(getattr(equation, "params", {}))
            return
        if isinstance(candidate, dict):
            for nested in candidate.values():
                visit(nested)
            return
        if isinstance(candidate, (tuple, list)):
            for nested in candidate:
                visit(nested)
            return
        array_value = getattr(candidate, "aval", None)
        shape = getattr(array_value, "shape", None)
        if shape is not None:
            shapes.append(tuple(int(size) for size in shape))

    visit(value)
    collected_shapes = tuple(sorted(shapes))
    return collected_shapes


class TestMatrixFreeGalerkinEngine:
    """Verify the fixed-support matrix-free Galerkin engine.

    :see: :func:`ptyrodactyl.galerkin.shifted_free_diagonal`
    :see: :func:`ptyrodactyl.galerkin.apply_galerkin_operator`
    :see: :func:`ptyrodactyl.galerkin.apply_galerkin_adjoint`
    :see: :func:`ptyrodactyl.galerkin.evaluate_galerkin_residual`
    :see: :func:`ptyrodactyl.galerkin.evaluate_galerkin_adjoint_residual`
    :see: :func:`ptyrodactyl.galerkin.cgls_solve`
    :see: :func:`ptyrodactyl.galerkin.lsqr_solve`
    :see: :func:`ptyrodactyl.galerkin.implicit_galerkin_solve`
    """

    @pytest.mark.parametrize(
        "carrier",
        [
            np.array([0.0, 0.0, _WAVENUMBER]),
            _BASE_CARRIER,
        ],
        ids=["on_axis", "tilted"],
    )
    def test_shifted_free_diagonal_matches_closed_form(
        self,
        carrier: Float[NDArray, "3"],
    ) -> None:
        """Match the shifted free diagonal for on-axis and tilted carriers."""
        reciprocal_frequencies = jnp.asarray(_RECIPROCAL_FREQUENCIES)
        carrier_array = jnp.asarray(carrier)
        wavenumber = jnp.asarray(_WAVENUMBER)
        expected = (
            np.sum(
                (carrier[None, :] + 2.0 * np.pi * _RECIPROCAL_FREQUENCIES)
                ** 2,
                axis=-1,
            )
            - _WAVENUMBER**2
        )

        eager = shifted_free_diagonal(
            reciprocal_frequencies,
            carrier_array,
            wavenumber,
        )
        compiled = jax.jit(shifted_free_diagonal)(
            reciprocal_frequencies,
            carrier_array,
            wavenumber,
        )

        np.testing.assert_allclose(eager, expected, rtol=1e-13, atol=1e-13)
        chex.assert_trees_all_close(
            compiled,
            eager,
            rtol=2e-15,
            atol=1e-15,
        )

    @pytest.mark.parametrize("dimensions", [2, 4], ids=["two", "four"])
    def test_shifted_free_diagonal_rejects_non_three_dimensional_support(
        self,
        dimensions: int,
    ) -> None:
        """Reject retained frequencies outside three spatial dimensions."""
        reciprocal_frequencies = jnp.zeros((3, dimensions))
        carrier = jnp.zeros(dimensions)

        with pytest.raises(
            ValueError,
            match="reciprocal frequencies must have three dimensions",
        ):
            shifted_free_diagonal(
                reciprocal_frequencies,
                carrier,
                jnp.asarray(2.0),
            )
        with pytest.raises(
            ValueError,
            match="reciprocal frequencies must have three dimensions",
        ):
            jax.jit(shifted_free_diagonal)(
                reciprocal_frequencies,
                carrier,
                jnp.asarray(2.0),
            )

    def test_shifted_free_diagonal_rejects_derived_overflow(self) -> None:
        """Reject a finite frequency whose squared diagonal overflows."""
        frequencies = jnp.array([[1.0e308, 0.0, 0.0]])
        carrier = jnp.zeros(3)
        for build in (shifted_free_diagonal, jax.jit(shifted_free_diagonal)):
            with pytest.raises(_RUNTIME_ERRORS, match="free_diagonal"):
                jax.block_until_ready(build(frequencies, carrier, 1.0))

    def test_operator_and_adjoint_match_dense_nonnormal_matrix(self) -> None:
        """Match dense H/H* actions and their complex dot identity."""
        operator = _create_operator()
        dense_operator = _dense_operator()
        field = jnp.asarray([0.3 - 0.1j, -0.2 + 0.5j, 0.4 + 0.2j, -0.1 - 0.3j])
        probe = jnp.asarray(
            [-0.15 + 0.2j, 0.35 - 0.1j, 0.05 + 0.4j, 0.2 - 0.25j]
        )

        action = apply_galerkin_operator(operator, field)
        adjoint_action = apply_galerkin_adjoint(operator, probe)
        compiled_action = jax.jit(apply_galerkin_operator)(operator, field)
        compiled_adjoint = jax.jit(apply_galerkin_adjoint)(operator, probe)

        assert (
            np.linalg.norm(
                dense_operator @ dense_operator.conj().T
                - dense_operator.conj().T @ dense_operator
            )
            > 1e-2
        )
        np.testing.assert_allclose(
            action,
            dense_operator @ np.asarray(field),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            adjoint_action,
            dense_operator.conj().T @ np.asarray(probe),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.vdot(np.asarray(probe), np.asarray(action)),
            np.vdot(np.asarray(adjoint_action), np.asarray(field)),
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            compiled_action,
            action,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            compiled_adjoint,
            adjoint_action,
            rtol=0.0,
            atol=0.0,
        )

    def test_actions_and_cgls_vectorize_over_complex_sources(self) -> None:
        """Match vmap H/H* and CGLS fields to explicit two-source stacks."""
        operator = _create_operator()
        sources = jnp.stack(
            (
                jnp.asarray(_SOURCE),
                jnp.asarray(
                    [
                        -0.2 + 0.1j,
                        0.35 - 0.15j,
                        0.05 + 0.25j,
                        -0.1 - 0.3j,
                    ]
                ),
            )
        )
        vmapped_actions = jax.vmap(
            lambda source: apply_galerkin_operator(operator, source)
        )(sources)
        vmapped_adjoint_actions = jax.vmap(
            lambda source: apply_galerkin_adjoint(operator, source)
        )(sources)
        explicit_actions = jnp.stack(
            tuple(
                apply_galerkin_operator(operator, source) for source in sources
            )
        )
        explicit_adjoint_actions = jnp.stack(
            tuple(
                apply_galerkin_adjoint(operator, source) for source in sources
            )
        )

        def solve_field(
            source: Complex[jax.Array, "4"],
        ) -> Complex[jax.Array, "4"]:
            """Return one converged CGLS field for batching."""
            result = cgls_solve(
                operator,
                source,
                max_iterations=64,
                relative_tolerance=1e-12,
                absolute_tolerance=1e-14,
            )
            return result.field

        vmapped_fields = jax.vmap(solve_field)(sources)
        explicit_fields = jnp.stack(
            tuple(solve_field(source) for source in sources)
        )
        expected_fields = np.stack(
            tuple(
                np.linalg.solve(_dense_operator(), np.asarray(source))
                for source in sources
            )
        )

        chex.assert_trees_all_close(
            vmapped_actions,
            explicit_actions,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            vmapped_adjoint_actions,
            explicit_adjoint_actions,
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            vmapped_fields,
            explicit_fields,
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            vmapped_fields,
            expected_fields,
            rtol=2e-10,
            atol=2e-11,
        )

    @pytest.mark.parametrize(
        ("solver", "method"),
        [
            (cgls_solve, GalerkinSolveMethod.CGLS),
            (lsqr_solve, GalerkinSolveMethod.LSQR),
        ],
        ids=["cgls", "lsqr"],
    )
    def test_krylov_solvers_match_independent_dense_solution(
        self,
        solver: Callable[..., GalerkinSolveResult],
        method: GalerkinSolveMethod,
    ) -> None:
        """Match dense solves for a nonnormal system in eager and JIT modes."""
        operator = _create_operator()
        source = jnp.asarray(_SOURCE)
        expected = np.linalg.solve(_dense_operator(), _SOURCE)

        def solve(
            candidate: GalerkinOperator,
            right_hand_side: Complex[jax.Array, "4"],
        ) -> GalerkinSolveResult:
            """Run one fixed-budget matrix-free solve."""
            result = solver(
                candidate,
                right_hand_side,
                max_iterations=64,
                relative_tolerance=1e-12,
                absolute_tolerance=1e-14,
            )
            return result

        eager = solve(operator, source)
        compiled = jax.jit(solve)(operator, source)

        assert eager.status == GalerkinSolveStatus.CONVERGED
        assert bool(eager.converged)
        assert eager.method is method
        np.testing.assert_allclose(
            eager.field,
            expected,
            rtol=2e-10,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            compiled.field,
            eager.field,
            rtol=1e-12,
            atol=1e-12,
        )
        assert compiled.status == eager.status
        assert bool(compiled.converged)
        assert compiled.method is method
        for result in (eager, compiled):
            assert result.field.dtype == jnp.complex128
            assert result.residual.dtype == jnp.complex128
            assert result.residual_norm.dtype == jnp.float64
            assert result.normal_residual_norm.dtype == jnp.float64
            assert result.recurrence_residual_norm.dtype == jnp.float64
            assert result.iterations.dtype == jnp.int32
            assert result.operator_applications.dtype == jnp.int32
            assert result.status.dtype == jnp.int32

    @pytest.mark.parametrize(
        "solver",
        [cgls_solve, lsqr_solve],
        ids=["cgls", "lsqr"],
    )
    def test_krylov_solvers_are_invariant_to_uniform_normal_range_scaling(
        self,
        solver: Callable[..., GalerkinSolveResult],
    ) -> None:
        """Retain the dense solution when both target and RHS scale down."""
        scale = jnp.asarray(1.0e-200, dtype=jnp.float64)
        base = _create_operator()
        operator = create_galerkin_operator(
            free_diagonal=scale * base.free_diagonal,
            interaction_rows=base.interaction_rows,
            interaction_columns=base.interaction_columns,
            interaction_values=scale * base.interaction_values,
            absorber_factor_rows=base.absorber_factor_rows,
            absorber_factor_columns=base.absorber_factor_columns,
            absorber_factor_values=base.absorber_factor_values,
            cap_scale=scale * base.cap_scale,
            absorber_factor_size=base.absorber_factor_size,
        )
        source = scale * jnp.asarray(_SOURCE)
        expected = np.linalg.solve(_dense_operator(), _SOURCE)

        def solve(candidate_source: jax.Array) -> GalerkinSolveResult:
            """Solve the uniformly scaled fixed operator."""
            result = solver(
                operator,
                candidate_source,
                max_iterations=64,
                relative_tolerance=1.0e-12,
                absolute_tolerance=0.0,
            )
            return result

        eager = solve(source)
        compiled = jax.jit(solve)(source)

        for result in (eager, compiled):
            assert bool(result.converged)
            assert result.status == GalerkinSolveStatus.CONVERGED
            np.testing.assert_allclose(
                result.field,
                expected,
                rtol=3.0e-10,
                atol=3.0e-11,
            )

    @pytest.mark.parametrize(
        "solver",
        [cgls_solve, lsqr_solve],
        ids=["cgls", "lsqr"],
    )
    def test_krylov_solvers_accept_zero_source_and_exact_initial_field(
        self,
        solver: Callable[..., GalerkinSolveResult],
    ) -> None:
        """Stop at iteration zero for zero and exactly satisfied residuals."""
        operator = _create_operator()
        zero_source = jnp.zeros(4, dtype=jnp.complex128)
        zero_result = solver(operator, zero_source, max_iterations=8)
        initial_field = jnp.asarray(
            [0.2 + 0.1j, -0.3j, 0.15 - 0.05j, -0.1 + 0.2j]
        )
        exact_source = apply_galerkin_operator(operator, initial_field)
        exact_result = solver(
            operator,
            exact_source,
            initial_field=initial_field,
            max_iterations=8,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )

        assert zero_result.status == GalerkinSolveStatus.CONVERGED
        assert bool(zero_result.converged)
        assert int(zero_result.iterations) == 0
        chex.assert_trees_all_close(
            zero_result.field,
            zero_source,
            rtol=0.0,
            atol=0.0,
        )
        assert exact_result.status == GalerkinSolveStatus.CONVERGED
        assert bool(exact_result.converged)
        assert int(exact_result.iterations) == 0
        chex.assert_trees_all_close(
            exact_result.field,
            initial_field,
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "solver",
        [cgls_solve, lsqr_solve],
        ids=["cgls", "lsqr"],
    )
    def test_krylov_solvers_report_iteration_limit(
        self,
        solver: Callable[..., GalerkinSolveResult],
    ) -> None:
        """Return a typed iteration limit after an insufficient single step."""
        result = solver(
            _create_operator(),
            jnp.asarray(_SOURCE),
            max_iterations=1,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )

        assert result.status == GalerkinSolveStatus.MAX_ITERATIONS
        assert not bool(result.converged)
        assert int(result.iterations) == 1
        assert jnp.all(jnp.isfinite(result.field))
        assert jnp.all(jnp.isfinite(result.residual))

    @pytest.mark.parametrize(
        "solver",
        [cgls_solve, lsqr_solve],
        ids=["cgls", "lsqr"],
    )
    def test_krylov_solvers_fail_closed_on_zero_operator(
        self,
        solver: Callable[..., GalerkinSolveResult],
    ) -> None:
        """Report breakdown instead of regularizing a singular zero action."""
        zero_operator = create_galerkin_operator(
            free_diagonal=jnp.zeros(4),
            interaction_rows=jnp.array([0], dtype=jnp.int32),
            interaction_columns=jnp.array([0], dtype=jnp.int32),
            interaction_values=jnp.array([0.0j]),
            absorber_factor_rows=jnp.array([0], dtype=jnp.int32),
            absorber_factor_columns=jnp.array([0], dtype=jnp.int32),
            absorber_factor_values=jnp.array([0.0j]),
            cap_scale=jnp.asarray(1.0),
            absorber_factor_size=1,
        )
        source = jnp.ones(4, dtype=jnp.complex128)
        result = solver(zero_operator, source, max_iterations=4)

        assert result.status == GalerkinSolveStatus.BREAKDOWN
        assert not bool(result.converged)
        chex.assert_trees_all_close(
            result.residual,
            source,
            rtol=0.0,
            atol=0.0,
        )
        assert jnp.all(jnp.isfinite(result.field))

    def test_lsqr_fails_closed_on_normal_residual_exhaustion(self) -> None:
        """Fail on an exhausted normal residual and count every action."""
        rank_deficient_operator = create_galerkin_operator(
            free_diagonal=jnp.asarray([1.0, 0.0]),
            interaction_rows=jnp.array([0], dtype=jnp.int32),
            interaction_columns=jnp.array([0], dtype=jnp.int32),
            interaction_values=jnp.array([0.0j]),
            absorber_factor_rows=jnp.array([0], dtype=jnp.int32),
            absorber_factor_columns=jnp.array([0], dtype=jnp.int32),
            absorber_factor_values=jnp.array([0.0j]),
            cap_scale=jnp.asarray(1.0),
            absorber_factor_size=1,
        )
        source = jnp.asarray([0.6 + 0.0j, 0.8 + 0.0j])
        result = lsqr_solve(
            rank_deficient_operator,
            source,
            max_iterations=3,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )

        assert result.status == GalerkinSolveStatus.BREAKDOWN
        assert not bool(result.converged)
        assert result.method is GalerkinSolveMethod.LSQR
        assert int(result.iterations) == 1
        assert int(result.operator_applications) == 8
        assert float(result.normal_residual_norm) == 0.0
        chex.assert_trees_all_close(
            result.field,
            jnp.asarray([0.6 + 0.0j, 0.0 + 0.0j]),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            result.residual,
            jnp.asarray([0.0 + 0.0j, 0.8 + 0.0j]),
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        ("keyword", "invalid_value", "message"),
        [
            ("max_iterations", 0, "max_iterations must be positive"),
            (
                "relative_tolerance",
                -1.0,
                "solver tolerances must be finite and non-negative",
            ),
        ],
        ids=["iteration_limit", "negative_tolerance"],
    )
    def test_solver_rejects_invalid_controls_eager_and_traced(
        self,
        keyword: str,
        invalid_value: float,
        message: str,
    ) -> None:
        """Reject nonpositive iteration limits and negative tolerances."""
        operator = _create_operator()
        source = jnp.asarray(_SOURCE)

        def solve_with_control(control: Float[jax.Array, ""]):
            """Apply one dynamic invalid solver control."""
            arguments = {keyword: control}
            result = cgls_solve(operator, source, **arguments)
            return result.field

        with pytest.raises(_RUNTIME_ERRORS, match=message):
            eager = solve_with_control(jnp.asarray(invalid_value))
            jax.block_until_ready(eager)
        with pytest.raises(_RUNTIME_ERRORS, match=message):
            compiled = jax.jit(solve_with_control)(jnp.asarray(invalid_value))
            jax.block_until_ready(compiled)

    @pytest.mark.parametrize(
        "solver",
        [cgls_solve, lsqr_solve],
        ids=["cgls", "lsqr"],
    )
    def test_solver_rejects_overflowing_stopping_threshold(
        self,
        solver: Callable[..., GalerkinSolveResult],
    ) -> None:
        """Reject a finite tolerance whose derived threshold overflows."""
        operator = _create_operator()
        source = jnp.asarray([2.0 + 0.0j, 0.0, 0.0, 0.0])

        def solve(
            relative_tolerance: Float[jax.Array, ""],
        ) -> Complex[jax.Array, "4"]:
            """Solve with one dynamic tolerance near binary64 overflow."""
            result: GalerkinSolveResult = solver(
                operator,
                source,
                relative_tolerance=relative_tolerance,
            )
            return result.field

        invalid = jnp.asarray(1.0e308, dtype=jnp.float64)
        with pytest.raises(_RUNTIME_ERRORS, match="threshold must be finite"):
            eager = solve(invalid)
            jax.block_until_ready(eager)
        with pytest.raises(_RUNTIME_ERRORS, match="threshold must be finite"):
            compiled = jax.jit(solve)(invalid)
            jax.block_until_ready(compiled)

    def test_shifted_diagonal_rejects_nonfinite_carrier_eager_and_jit(
        self,
    ) -> None:
        """Reject a nonfinite carrier before a shifted diagonal is consumed."""
        reciprocal_frequencies = jnp.asarray(_RECIPROCAL_FREQUENCIES)
        invalid_carrier = jnp.asarray([jnp.inf, 0.0, _WAVENUMBER])

        def build(carrier: Float[jax.Array, "3"]) -> Float[jax.Array, "4"]:
            """Build one shifted diagonal from a dynamic carrier."""
            diagonal = shifted_free_diagonal(
                reciprocal_frequencies,
                carrier,
                jnp.asarray(_WAVENUMBER),
            )
            return diagonal

        with pytest.raises(_RUNTIME_ERRORS, match="carrier must be finite"):
            eager = build(invalid_carrier)
            jax.block_until_ready(eager)
        with pytest.raises(_RUNTIME_ERRORS, match="carrier must be finite"):
            compiled = jax.jit(build)(invalid_carrier)
            jax.block_until_ready(compiled)

    @pytest.mark.parametrize(
        "solver",
        [cgls_solve, lsqr_solve],
        ids=["cgls", "lsqr"],
    )
    def test_reported_residual_is_recomputed_from_original_system(
        self,
        solver: Callable[..., GalerkinSolveResult],
    ) -> None:
        """Match reported physical and normal residuals to dense H."""
        dense_operator = _dense_operator()
        result = solver(
            _create_operator(),
            jnp.asarray(_SOURCE),
            max_iterations=2,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
        expected_residual = _SOURCE - dense_operator @ np.asarray(result.field)
        expected_normal_residual = dense_operator.conj().T @ expected_residual

        np.testing.assert_allclose(
            result.residual,
            expected_residual,
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            result.residual_norm,
            np.linalg.norm(expected_residual),
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            result.normal_residual_norm,
            np.linalg.norm(expected_normal_residual),
            rtol=2e-12,
            atol=2e-12,
        )
        assert jnp.isfinite(result.recurrence_residual_norm)
        assert result.recurrence_residual_norm >= 0.0

    def test_residual_evaluators_match_independent_dense_actions(self) -> None:
        """Recompute forward and adjoint residuals outside recurrences."""
        operator = _create_operator()
        dense_operator = _dense_operator()
        field = np.linalg.solve(dense_operator, _SOURCE)
        state_gradient = np.array(
            [0.3 + 0.1j, -0.2 + 0.05j, 0.1 - 0.4j, 0.25 + 0.2j]
        )
        adjoint_field = np.linalg.solve(
            dense_operator.conj().T,
            state_gradient,
        )

        residual, residual_norm = evaluate_galerkin_residual(
            operator,
            jnp.asarray(field),
            jnp.asarray(_SOURCE),
        )
        adjoint_residual, adjoint_residual_norm = (
            evaluate_galerkin_adjoint_residual(
                operator,
                jnp.asarray(adjoint_field),
                jnp.asarray(state_gradient),
            )
        )
        compiled_adjoint_residual, compiled_adjoint_residual_norm = jax.jit(
            evaluate_galerkin_adjoint_residual
        )(
            operator,
            jnp.asarray(adjoint_field),
            jnp.asarray(state_gradient),
        )

        np.testing.assert_allclose(
            residual,
            _SOURCE - dense_operator @ field,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            adjoint_residual,
            state_gradient - dense_operator.conj().T @ adjoint_field,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.linalg.norm(np.asarray(adjoint_residual)) < 1e-12
        np.testing.assert_allclose(
            residual_norm,
            np.linalg.norm(_SOURCE - dense_operator @ field),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            adjoint_residual_norm,
            np.linalg.norm(
                state_gradient - dense_operator.conj().T @ adjoint_field
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            compiled_adjoint_residual,
            adjoint_residual,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            compiled_adjoint_residual_norm,
            adjoint_residual_norm,
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "case",
        _IMPLICIT_ADJOINT_CASES,
        ids=lambda case: case.case_id,
    )
    def test_implicit_vjp_matches_five_block_dense_differences(
        self,
        case: _ImplicitAdjointCase,
    ) -> None:
        """Match both copied RM-I1 cases over a centered-step sweep."""
        parameters = np.asarray(case.parameters, dtype=np.float64)
        parameter_array = jnp.asarray(parameters)

        def loss(
            candidate_parameters: Float[jax.Array, "5"],
        ) -> Float[jax.Array, ""]:
            """Evaluate one fixed copied implicit-adjoint case."""
            result = _reference_implicit_loss(case, candidate_parameters)
            return result

        eager_gradient = jax.grad(loss)(parameter_array)
        compiled_gradient = jax.jit(jax.grad(loss))(parameter_array)
        steps = (2e-3, 1e-3, 5e-4, 2.5e-4)
        finite_gradients = np.stack(
            [
                _reference_central_gradient(case, parameters, step)
                for step in steps
            ]
        )
        errors = np.max(
            np.abs(finite_gradients - np.asarray(eager_gradient)[None, :]),
            axis=1,
        )

        chex.assert_trees_all_close(
            compiled_gradient,
            eager_gradient,
            rtol=2e-11,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            eager_gradient,
            finite_gradients[-1],
            rtol=10.0 * case.tolerance,
            atol=case.tolerance,
        )
        assert np.all(np.abs(np.asarray(eager_gradient)) > 1e-6)
        np.testing.assert_allclose(
            errors[1:] / errors[:-1],
            0.25,
            rtol=2e-2,
            atol=1e-4,
        )

    def test_implicit_vjp_matches_real_and_imaginary_source_directions(
        self,
    ) -> None:
        """Match complex-source contractions to two centered differences."""
        operator = _create_operator()
        source = jnp.asarray(_SOURCE)
        gain = 0.85

        def source_loss(
            candidate_source: Complex[jax.Array, "4"],
        ) -> Float[jax.Array, ""]:
            """Evaluate a real detector loss through the implicit root."""
            field = implicit_galerkin_solve(
                operator,
                candidate_source,
                max_iterations=64,
                relative_tolerance=1e-13,
                absolute_tolerance=1e-14,
            )
            amplitude = jnp.vdot(jnp.asarray(_TERMINAL), field)
            residual = gain * amplitude - jnp.asarray(_TARGET)
            loss = 0.5 * jnp.real(residual.conj() * residual)
            return loss

        def dense_source_loss(
            candidate_source: Complex[NDArray, "4"],
        ) -> float:
            """Evaluate the same source-dependent loss by a dense solve."""
            field = np.linalg.solve(_dense_operator(), candidate_source)
            amplitude = np.vdot(_TERMINAL, field)
            residual = gain * amplitude - _TARGET
            loss = float(0.5 * np.real(residual.conjugate() * residual))
            return loss

        source_gradient = jax.grad(source_loss)(source)
        compiled_gradient = jax.jit(jax.grad(source_loss))(source)
        coordinate = 1
        real_direction = np.zeros(4, dtype=np.complex128)
        real_direction[coordinate] = 1.0
        imaginary_direction = np.zeros(4, dtype=np.complex128)
        imaginary_direction[coordinate] = 1.0j
        step = 2e-5

        for direction in (real_direction, imaginary_direction):
            finite_difference = (
                dense_source_loss(_SOURCE + step * direction)
                - dense_source_loss(_SOURCE - step * direction)
            ) / (2.0 * step)
            jax_directional = float(
                jnp.real(jnp.sum(source_gradient * jnp.asarray(direction)))
            )
            np.testing.assert_allclose(
                jax_directional,
                finite_difference,
                rtol=2e-7,
                atol=2e-9,
            )
            assert abs(jax_directional) > 1e-5

        chex.assert_trees_all_close(
            compiled_gradient,
            source_gradient,
            rtol=2e-11,
            atol=2e-12,
        )

    def test_explicit_tangent_solve_matches_dense_realified_direction(
        self,
    ) -> None:
        """Solve the JVP tangent identity in the fixed real reference chart."""
        parameters = jnp.asarray([0.35, 1.0, _CAP_SCALE, 1.0])
        direction = jnp.asarray([0.11, -0.07, 0.05, 0.09])
        operator, source = _operator_and_source(parameters)
        field = implicit_galerkin_solve(
            operator,
            source,
            max_iterations=64,
            relative_tolerance=1e-13,
            absolute_tolerance=1e-14,
        )

        def action_and_source(
            candidate_parameters: Float[jax.Array, "4"],
        ) -> Tuple[Complex[jax.Array, "4"], Complex[jax.Array, "4"]]:
            """Apply the varying operator to the fixed converged state."""
            candidate_operator, candidate_source = _operator_and_source(
                candidate_parameters
            )
            operated = apply_galerkin_operator(candidate_operator, field)
            pair = (operated, candidate_source)
            return pair

        (_, _), (operator_tangent, source_tangent) = jax.jvp(
            action_and_source,
            (parameters,),
            (direction,),
        )
        tangent_source = source_tangent - operator_tangent
        tangent_result = cgls_solve(
            operator,
            tangent_source,
            max_iterations=64,
            relative_tolerance=1e-13,
            absolute_tolerance=1e-14,
        )
        dense_operator = _dense_operator()
        dense_realified = _realify_matrix(dense_operator)
        tangent_source_realified = _realify_vector(np.asarray(tangent_source))
        expected_realified = np.linalg.solve(
            dense_realified,
            tangent_source_realified,
        )
        expected = expected_realified[:4] + 1j * expected_realified[4:]

        assert tangent_result.status == GalerkinSolveStatus.CONVERGED
        np.testing.assert_allclose(
            tangent_result.field,
            expected,
            rtol=3e-10,
            atol=3e-11,
        )
        assert np.linalg.norm(expected) > 1e-5

    def test_adjoint_residual_uses_realified_loss_gradient(self) -> None:
        """Solve H* lambda for a real loss and recompute its residual."""
        dense_operator = _dense_operator()
        field = np.linalg.solve(dense_operator, _SOURCE)
        gain = 0.85

        def realified_loss(
            realified_field: Float[jax.Array, "8"],
        ) -> Float[jax.Array, ""]:
            """Evaluate the detector loss in explicit real coordinates."""
            complex_field = realified_field[:4] + 1j * realified_field[4:]
            amplitude = jnp.vdot(jnp.asarray(_TERMINAL), complex_field)
            residual = gain * amplitude - jnp.asarray(_TARGET)
            loss = 0.5 * jnp.real(residual.conj() * residual)
            return loss

        state_gradient_real = jax.grad(realified_loss)(
            jnp.asarray(_realify_vector(field))
        )
        state_gradient = np.asarray(state_gradient_real[:4]) + 1j * np.asarray(
            state_gradient_real[4:]
        )
        adjoint_field = np.linalg.solve(
            dense_operator.conj().T,
            state_gradient,
        )
        residual, residual_norm = evaluate_galerkin_adjoint_residual(
            _create_operator(),
            jnp.asarray(adjoint_field),
            jnp.asarray(state_gradient),
        )

        np.testing.assert_allclose(
            residual,
            state_gradient - dense_operator.conj().T @ adjoint_field,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.linalg.norm(np.asarray(residual)) < 1e-12
        assert residual_norm < 1e-12

    def test_custom_vjp_saved_shapes_do_not_scale_with_iteration_budget(
        self,
    ) -> None:
        """Keep custom-VJP saved array shapes independent of Krylov budget."""
        parameters = jnp.asarray([0.35, 1.0, _CAP_SCALE, 1.0, 0.85])

        def short_loss(
            candidate_parameters: Float[jax.Array, "5"],
        ) -> Float[jax.Array, ""]:
            """Evaluate the implicit loss with an eight-step budget."""
            loss = _implicit_loss(candidate_parameters, max_iterations=8)
            return loss

        def long_loss(
            candidate_parameters: Float[jax.Array, "5"],
        ) -> Float[jax.Array, ""]:
            """Evaluate the implicit loss with a 32-step budget."""
            loss = _implicit_loss(candidate_parameters, max_iterations=32)
            return loss

        short_jaxpr = jax.make_jaxpr(jax.grad(short_loss))(parameters)
        long_jaxpr = jax.make_jaxpr(jax.grad(long_loss))(parameters)
        short_shapes = _jaxpr_array_shapes(short_jaxpr)
        long_shapes = _jaxpr_array_shapes(long_jaxpr)

        assert short_shapes == long_shapes
        assert all(
            8 not in shape and 32 not in shape for shape in short_shapes
        )

    def test_algebraic_actions_and_residuals_fail_closed_on_range(
        self,
    ) -> None:
        """Reject NaN, overflow, and subnormal algebraic boundary inputs."""
        operator = _create_operator()
        overflow_operator = create_galerkin_operator(
            free_diagonal=jnp.full(4, 1.0e10),
            interaction_rows=operator.interaction_rows,
            interaction_columns=operator.interaction_columns,
            interaction_values=operator.interaction_values,
            absorber_factor_rows=operator.absorber_factor_rows,
            absorber_factor_columns=operator.absorber_factor_columns,
            absorber_factor_values=operator.absorber_factor_values,
            cap_scale=operator.cap_scale,
            absorber_factor_size=operator.absorber_factor_size,
        )
        zeros = jnp.zeros(4, dtype=jnp.complex128)
        nan_field = zeros.at[0].set(jnp.nan + 0.0j)
        huge_field = jnp.full(4, 1.0e308 + 0.0j, dtype=jnp.complex128)
        subnormal_source = zeros.at[0].set(1.0e-308 + 0.0j)

        for action in (apply_galerkin_operator, apply_galerkin_adjoint):
            for call in (action, jax.jit(action)):
                with pytest.raises(_RUNTIME_ERRORS, match="must be finite"):
                    jax.block_until_ready(call(operator, nan_field))
                with pytest.raises(_RUNTIME_ERRORS, match="must be finite"):
                    jax.block_until_ready(call(overflow_operator, huge_field))
        for residual in (
            evaluate_galerkin_residual,
            evaluate_galerkin_adjoint_residual,
        ):
            for call in (residual, jax.jit(residual)):
                with pytest.raises(_RUNTIME_ERRORS, match="subnormal"):
                    jax.block_until_ready(
                        call(operator, zeros, subnormal_source)
                    )

    def test_algebraic_adjacent_normal_residual_cancellation_fails_closed(
        self,
    ) -> None:
        """Reject a flushed nonzero residual in both algebraic directions."""
        operator = create_galerkin_operator(
            free_diagonal=jnp.ones(1),
            interaction_rows=jnp.zeros(1, dtype=jnp.int32),
            interaction_columns=jnp.zeros(1, dtype=jnp.int32),
            interaction_values=jnp.zeros(1, dtype=jnp.complex128),
            absorber_factor_rows=jnp.zeros(1, dtype=jnp.int32),
            absorber_factor_columns=jnp.zeros(1, dtype=jnp.int32),
            absorber_factor_values=jnp.ones(1, dtype=jnp.complex128),
            cap_scale=1.0,
            absorber_factor_size=1,
        )
        field = jnp.asarray(
            [8.0 * np.finfo(np.float64).tiny + 0.0j],
            dtype=jnp.complex128,
        )
        pairs = (
            (apply_galerkin_operator, evaluate_galerkin_residual),
            (apply_galerkin_adjoint, evaluate_galerkin_adjoint_residual),
        )
        for action, residual in pairs:
            applied = np.asarray(action(operator, field)).copy()
            source_host = applied.copy()
            source_host[0] = complex(
                np.nextafter(source_host[0].real, np.inf),
                source_host[0].imag,
            )
            source = jnp.asarray(source_host)

            assert source_host[0] - applied[0] != 0.0j

            def residual_dynamic(
                candidate_field: jax.Array,
                candidate_source: jax.Array,
            ):
                """Evaluate one algebraic adjacent-normal residual."""
                result = residual(
                    operator,
                    candidate_field,
                    candidate_source,
                )
                return result

            calls = (
                lambda: residual_dynamic(field, source),
                lambda: jax.jit(residual_dynamic)(field, source),
                jax.jit(lambda: residual_dynamic(field, source)),
            )
            for call in calls:
                with pytest.raises(
                    _RUNTIME_ERRORS, match="subtraction lost|subnormal"
                ):
                    jax.block_until_ready(call())
