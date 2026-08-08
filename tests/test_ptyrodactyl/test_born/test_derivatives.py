"""Tests for :mod:`ptyrodactyl.born.derivatives`.

Extended Summary
----------------
The tests fix one three-mode SC-1 support. They assemble the dense target,
realified Jacobian, and centered differences directly from copied
coefficients without using production actions as an oracle. The base
right-hand side traverses the exact finite matched-source injection.
"""

from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Complex, Float, Int
from numpy.typing import NDArray

from ptyrodactyl.born import (
    create_matched_galerkin_source,
    galerkin_state_jvp,
    galerkin_state_vjp,
)
from ptyrodactyl.tools import helmholtz_coupling
from ptyrodactyl.types import (
    GalerkinSource,
    GalerkinTargetManifest,
    create_galerkin_product_support,
    create_galerkin_target_manifest,
)

_STATE_INDICES: Int[NDArray, "3 3"] = np.array(
    [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
    dtype=np.int32,
)
_INTERACTION_INDICES: Int[NDArray, "3 3"] = np.array(
    [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
    dtype=np.int32,
)
_ABSORBER_INDICES: Int[NDArray, "29 3"] = np.array(
    [
        [first, second, third]
        for first in range(-1, 2)
        for second in range(-1, 2)
        for third in range(-1, 2)
    ]
    + [[-2, 0, 0], [2, 0, 0]],
    dtype=np.int32,
)
_WORK_INDICES: Int[NDArray, "63 3"] = np.array(
    [
        [first, second, third]
        for first in range(-3, 4)
        for second in range(-1, 2)
        for third in range(-1, 2)
    ],
    dtype=np.int32,
)
_BOX_LENGTHS: Float[NDArray, "3"] = np.array(
    [5.0, 6.0, 7.0],
    dtype=np.float64,
)
_CARRIER: Float[NDArray, "3"] = np.array(
    [0.31, -0.18, 250.5320619523448],
    dtype=np.float64,
)
_VOLTAGE_KV = 200.0
_WAVENUMBER = 250.53231840641544
_CAP_SCALE = 0.23
_INTERACTION: Complex[NDArray, "3"] = np.array(
    [0.045 - 0.025j, 0.22 + 0.0j, 0.045 + 0.025j],
    dtype=np.complex128,
)
_ABSORBER_AXIS_COEFFICIENTS = np.where(
    _ABSORBER_INDICES == 0,
    0.5,
    np.where(np.abs(_ABSORBER_INDICES) == 1, 0.25, 0.0),
)
_ABSORBER_INTERIOR = np.prod(_ABSORBER_AXIS_COEFFICIENTS, axis=-1)
_ABSORBER: Complex[NDArray, "29"] = np.where(
    np.all(_ABSORBER_INDICES == 0, axis=-1),
    1.0 - _ABSORBER_INTERIOR,
    -_ABSORBER_INTERIOR,
).astype(np.complex128)
_SOURCE: Complex[NDArray, "3"] = np.array(
    [0.8 + 0.2j, -0.25 + 0.35j, 0.12 - 0.18j],
    dtype=np.complex128,
)
_OUTPUT_COTANGENT: Complex[NDArray, "3"] = np.array(
    [0.2 + 0.3j, -0.1 + 0.05j, 0.15 - 0.2j],
    dtype=np.complex128,
)
_SOLVER_ARGUMENTS = {
    "max_iterations": 48,
    "relative_tolerance": 1e-13,
    "absolute_tolerance": 1e-14,
}


class _Direction(NamedTuple):
    """Store one real coordinate direction in the fixed target chart."""

    name: str
    carrier: Float[NDArray, "3"]
    interaction: Complex[NDArray, "3"]
    source: Complex[NDArray, "3"]


def _directions() -> tuple[_Direction, ...]:
    """Build carrier, Hermitian-interaction, and complex-source directions."""
    zero_carrier = np.zeros(3, dtype=np.float64)
    zero_interaction = np.zeros(3, dtype=np.complex128)
    zero_source = np.zeros(3, dtype=np.complex128)
    directions: list[_Direction] = [
        _Direction(
            "carrier_x",
            np.array([1.0, 0.0, -_CARRIER[0] / _CARRIER[2]]),
            zero_interaction.copy(),
            zero_source.copy(),
        ),
        _Direction(
            "interaction_zero_real",
            zero_carrier.copy(),
            np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            zero_source.copy(),
        ),
        _Direction(
            "interaction_pair_real",
            zero_carrier.copy(),
            np.array([1.0, 0.0, 1.0], dtype=np.complex128),
            zero_source.copy(),
        ),
        _Direction(
            "interaction_pair_imaginary",
            zero_carrier.copy(),
            np.array([-1.0j, 0.0, 1.0j], dtype=np.complex128),
            zero_source.copy(),
        ),
    ]
    for index in range(_SOURCE.size):
        real_source = zero_source.copy()
        real_source[index] = 1.0
        imaginary_source = zero_source.copy()
        imaginary_source[index] = 1.0j
        directions.extend(
            [
                _Direction(
                    f"source_{index}_real",
                    zero_carrier.copy(),
                    zero_interaction.copy(),
                    real_source,
                ),
                _Direction(
                    f"source_{index}_imaginary",
                    zero_carrier.copy(),
                    zero_interaction.copy(),
                    imaginary_source,
                ),
            ]
        )
    result: tuple[_Direction, ...] = tuple(directions)
    return result


_DIRECTIONS = _directions()


def _create_target(
    carrier: Float[NDArray, "3"] | Float[jax.Array, "3"] = _CARRIER,
    interaction: Complex[NDArray, "3"]
    | Complex[jax.Array, "3"] = _INTERACTION,
) -> GalerkinTargetManifest:
    """Create the production target from the fixed independent fixture."""
    support = create_galerkin_product_support(
        state_indices=jnp.asarray(_STATE_INDICES),
        interaction_indices=jnp.asarray(_INTERACTION_INDICES),
        absorber_indices=jnp.asarray(_ABSORBER_INDICES),
        work_indices=jnp.asarray(_WORK_INDICES),
        work_shape=(7, 3, 3),
    )
    target = create_galerkin_target_manifest(
        support=support,
        preterminal_indices=jnp.asarray(_STATE_INDICES),
        voltage_coefficients=(
            jnp.asarray(interaction)
            / helmholtz_coupling(jnp.asarray(_VOLTAGE_KV))
        ),
        carrier=jnp.asarray(carrier),
        box_lengths=jnp.asarray(_BOX_LENGTHS),
        accelerating_voltage_kv=jnp.asarray(_VOLTAGE_KV),
        cap_scale=jnp.asarray(_CAP_SCALE),
        target_name="fixed-three-mode-derivative-target",
    )
    return target


def _coefficient_matrix(
    support_indices: Int[NDArray, "p 3"],
    coefficients: Complex[NDArray, "p"],
) -> Complex[NDArray, "3 3"]:
    """Assemble a multiplier matrix directly from coefficient differences."""
    coefficient_map = {
        tuple(int(component) for component in index): coefficient
        for index, coefficient in zip(
            support_indices,
            coefficients,
            strict=True,
        )
    }
    matrix = np.array(
        [
            [
                coefficient_map.get(
                    tuple(int(value) for value in row - column),
                    0.0j,
                )
                for column in _STATE_INDICES
            ]
            for row in _STATE_INDICES
        ],
        dtype=np.complex128,
    )
    return matrix


def _dense_operator(
    carrier: Float[NDArray, "3"],
    interaction: Complex[NDArray, "3"],
) -> Complex[NDArray, "3 3"]:
    """Assemble the tiny SC-1 target without production action helpers."""
    reciprocal_frequencies = _STATE_INDICES / _BOX_LENGTHS[None, :]
    shifted = carrier[None, :] + 2.0 * np.pi * reciprocal_frequencies
    free_diagonal = np.sum(shifted**2, axis=-1) - _WAVENUMBER**2
    interaction_matrix = _coefficient_matrix(
        _INTERACTION_INDICES,
        interaction,
    )
    absorber_matrix = _coefficient_matrix(
        _ABSORBER_INDICES,
        _ABSORBER,
    )
    operator = (
        np.diag(free_diagonal)
        - interaction_matrix
        - 1j * _CAP_SCALE * absorber_matrix
    )
    return operator


def _create_conforming_source(
    target: GalerkinTargetManifest,
) -> GalerkinSource:
    """Represent the copied right-hand side by exact matched injection."""
    reciprocal_frequencies = _STATE_INDICES / _BOX_LENGTHS[None, :]
    shifted = _CARRIER[None, :] + 2.0 * np.pi * reciprocal_frequencies
    free_diagonal = np.sum(shifted**2, axis=-1) - _WAVENUMBER**2
    absorber_matrix = _coefficient_matrix(
        _ABSORBER_INDICES,
        _ABSORBER,
    )
    free_target = np.diag(free_diagonal) - 1j * _CAP_SCALE * absorber_matrix
    incident_field = np.linalg.solve(free_target, _SOURCE)
    source = create_matched_galerkin_source(
        target,
        jnp.asarray(incident_field),
    )
    return source


def _directional_dense_tangent(
    direction: _Direction,
    field: Complex[NDArray, "3"],
    operator: Complex[NDArray, "3 3"],
) -> Complex[NDArray, "3"]:
    """Solve one independently assembled implicit tangent equation."""
    reciprocal_frequencies = _STATE_INDICES / _BOX_LENGTHS[None, :]
    shifted = _CARRIER[None, :] + 2.0 * np.pi * reciprocal_frequencies
    diagonal_tangent = 2.0 * np.sum(
        shifted * direction.carrier[None, :],
        axis=-1,
    )
    interaction_tangent = _coefficient_matrix(
        _INTERACTION_INDICES,
        direction.interaction,
    )
    operator_tangent = np.diag(diagonal_tangent) - interaction_tangent
    tangent_source = direction.source - operator_tangent @ field
    field_tangent = np.linalg.solve(operator, tangent_source)
    return field_tangent


def _realify_vector(
    values: Complex[NDArray, "3"],
) -> Float[NDArray, "6"]:
    """Map one complex state to the frozen block-ordered real chart."""
    realified = np.concatenate((values.real, values.imag))
    return realified


def _dense_realified_jacobian() -> Float[NDArray, "6 parameters"]:
    """Assemble every tangent column without production differentiation."""
    operator = _dense_operator(_CARRIER, _INTERACTION)
    field = np.linalg.solve(operator, _SOURCE)
    columns = [
        _realify_vector(_directional_dense_tangent(direction, field, operator))
        for direction in _DIRECTIONS
    ]
    jacobian = np.column_stack(columns)
    return jacobian


def _parameterized_dense_field(
    parameters: Float[NDArray, "parameters"],
) -> Complex[NDArray, "3"]:
    """Evaluate an independent dense solve in the declared real chart."""
    carrier = _CARRIER.copy()
    carrier[0] += parameters[0]
    carrier[2] = np.sqrt(_WAVENUMBER**2 - carrier[0] ** 2 - carrier[1] ** 2)
    interaction = _INTERACTION.copy()
    source = _SOURCE.copy()
    for parameter, direction in zip(
        parameters[1:],
        _DIRECTIONS[1:],
        strict=True,
    ):
        interaction += parameter * direction.interaction
        source += parameter * direction.source
    field = np.linalg.solve(_dense_operator(carrier, interaction), source)
    return field


def _mixed_direction() -> Float[NDArray, "parameters"]:
    """Return one nonzero direction spanning every admitted leaf family."""
    direction = np.array(
        [0.08, -0.04, 0.03, -0.05, 0.02, -0.01, 0.04, 0.03, -0.02, 0.01],
        dtype=np.float64,
    )
    return direction


class TestGalerkinDerivatives:
    """Verify the fixed-support production derivative harness.

    :see: :func:`ptyrodactyl.born.galerkin_state_jvp`
    :see: :func:`ptyrodactyl.born.galerkin_state_vjp`
    """

    def test_jvp_matches_dense_realified_jacobian_eager_and_jit(self) -> None:
        """Match all carrier, interaction, and complex-source columns."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        carrier_tangents = jnp.asarray(
            np.stack([direction.carrier for direction in _DIRECTIONS])
        )
        interaction_tangents = jnp.asarray(
            np.stack([direction.interaction for direction in _DIRECTIONS])
        )
        source_tangents = jnp.asarray(
            np.stack([direction.source for direction in _DIRECTIONS])
        )

        def all_tangents(
            candidate_carrier_tangents: Float[jax.Array, "parameters 3"],
            candidate_interaction_tangents: Complex[jax.Array, "parameters 3"],
            candidate_source_tangents: Complex[jax.Array, "parameters 3"],
        ) -> tuple[
            Complex[jax.Array, "parameters 3"],
            Complex[jax.Array, "parameters 3"],
        ]:
            """Vectorize the public JVP over real coordinate directions."""
            fields, field_tangents = jax.vmap(
                lambda carrier_tangent, interaction_tangent, source_tangent: (
                    galerkin_state_jvp(
                        target,
                        source,
                        carrier_tangent,
                        interaction_tangent,
                        source_tangent,
                        **_SOLVER_ARGUMENTS,
                    )
                )
            )(
                candidate_carrier_tangents,
                candidate_interaction_tangents,
                candidate_source_tangents,
            )
            return fields, field_tangents

        eager_fields, eager_tangents = all_tangents(
            carrier_tangents,
            interaction_tangents,
            source_tangents,
        )
        compiled_fields, compiled_tangents = jax.jit(all_tangents)(
            carrier_tangents,
            interaction_tangents,
            source_tangents,
        )
        dense_operator = _dense_operator(_CARRIER, _INTERACTION)
        dense_field = np.linalg.solve(dense_operator, _SOURCE)
        dense_jacobian = _dense_realified_jacobian()
        production_jacobian = np.column_stack(
            [
                _realify_vector(np.asarray(tangent))
                for tangent in eager_tangents
            ]
        )

        np.testing.assert_allclose(
            eager_fields,
            np.broadcast_to(dense_field, eager_fields.shape),
            rtol=3e-10,
            atol=3e-11,
        )
        np.testing.assert_allclose(
            production_jacobian,
            dense_jacobian,
            rtol=4e-9,
            atol=4e-10,
        )
        chex.assert_trees_all_close(
            compiled_fields,
            eager_fields,
            rtol=2e-11,
            atol=2e-12,
        )
        chex.assert_trees_all_close(
            compiled_tangents,
            eager_tangents,
            rtol=2e-10,
            atol=2e-11,
        )
        column_norms = np.linalg.norm(production_jacobian, axis=0)
        assert np.all(column_norms > 1e-5)

    def test_jvp_matches_independent_centered_step_sweep(self) -> None:
        """Show second-order centered convergence for one mixed direction."""
        direction_weights = _mixed_direction()
        carrier_tangent = sum(
            (
                weight * direction.carrier
                for weight, direction in zip(
                    direction_weights,
                    _DIRECTIONS,
                    strict=True,
                )
            ),
            np.zeros(3, dtype=np.float64),
        )
        interaction_tangent = sum(
            (
                weight * direction.interaction
                for weight, direction in zip(
                    direction_weights,
                    _DIRECTIONS,
                    strict=True,
                )
            ),
            np.zeros(3, dtype=np.complex128),
        )
        source_tangent = sum(
            (
                weight * direction.source
                for weight, direction in zip(
                    direction_weights,
                    _DIRECTIONS,
                    strict=True,
                )
            ),
            np.zeros(3, dtype=np.complex128),
        )
        target = _create_target()
        source = _create_conforming_source(target).total_source
        _, production_tangent = galerkin_state_jvp(
            target,
            source,
            jnp.asarray(carrier_tangent),
            jnp.asarray(interaction_tangent),
            jnp.asarray(source_tangent),
            **_SOLVER_ARGUMENTS,
        )
        steps = (2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3)
        differences = np.stack(
            [
                (
                    _parameterized_dense_field(step * direction_weights)
                    - _parameterized_dense_field(-step * direction_weights)
                )
                / (2.0 * step)
                for step in steps
            ]
        )
        errors = np.linalg.norm(
            differences - np.asarray(production_tangent)[None, :],
            axis=1,
        )

        np.testing.assert_allclose(
            production_tangent,
            differences[-1],
            rtol=2e-7,
            atol=2e-9,
        )
        np.testing.assert_allclose(
            errors[1:] / errors[:-1],
            0.25,
            rtol=8e-2,
            atol=2e-3,
        )

    def test_vjp_matches_dense_transpose_and_centered_differences(
        self,
    ) -> None:
        """Match the custom adjoint in eager, JIT, dense, and FD paths.

        The three leading error ratios select the observed second-order
        pre-roundoff region. Smaller steps must continue reducing the error.
        """
        target = _create_target()
        source = _create_conforming_source(target).total_source
        output_cotangent = jnp.asarray(_OUTPUT_COTANGENT)

        def pullback(
            candidate_output_cotangent: Complex[jax.Array, "3"],
        ) -> tuple[
            Complex[jax.Array, "3"],
            Float[jax.Array, "3"],
            Complex[jax.Array, "3"],
            Complex[jax.Array, "3"],
        ]:
            """Evaluate the public VJP for one state cotangent."""
            result = galerkin_state_vjp(
                target,
                source,
                candidate_output_cotangent,
                **_SOLVER_ARGUMENTS,
            )
            return result

        eager = pullback(output_cotangent)
        compiled = jax.jit(pullback)(output_cotangent)
        field, carrier_cotangent, interaction_cotangent, source_cotangent = (
            eager
        )
        dense_field = np.linalg.solve(
            _dense_operator(_CARRIER, _INTERACTION),
            _SOURCE,
        )
        dense_jacobian = _dense_realified_jacobian()
        output_real_covector = np.concatenate(
            (_OUTPUT_COTANGENT.real, -_OUTPUT_COTANGENT.imag)
        )
        dense_pullback = dense_jacobian.T @ output_real_covector
        production_pullback = np.array(
            [
                np.sum(np.asarray(carrier_cotangent) * direction.carrier)
                + np.real(
                    np.sum(
                        np.asarray(interaction_cotangent)
                        * direction.interaction
                    )
                )
                + np.real(
                    np.sum(np.asarray(source_cotangent) * direction.source)
                )
                for direction in _DIRECTIONS
            ]
        )

        np.testing.assert_allclose(
            field,
            dense_field,
            rtol=3e-10,
            atol=3e-11,
        )
        np.testing.assert_allclose(
            production_pullback,
            dense_pullback,
            rtol=5e-9,
            atol=5e-10,
        )
        chex.assert_trees_all_close(
            compiled,
            eager,
            rtol=2e-10,
            atol=2e-11,
        )

        def dense_loss(parameters: Float[NDArray, "parameters"]) -> float:
            """Contract an independent dense state with the JAX cotangent."""
            candidate_field = _parameterized_dense_field(parameters)
            loss = float(np.real(np.sum(_OUTPUT_COTANGENT * candidate_field)))
            return loss

        steps = (2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5)
        zero = np.zeros(len(_DIRECTIONS), dtype=np.float64)
        finite_gradients = []
        for step in steps:
            gradient = np.zeros(len(_DIRECTIONS), dtype=np.float64)
            for index in range(len(_DIRECTIONS)):
                offset = zero.copy()
                offset[index] = step
                gradient[index] = (
                    dense_loss(offset) - dense_loss(-offset)
                ) / (2.0 * step)
            finite_gradients.append(gradient)
        finite_gradient_array = np.stack(finite_gradients)
        errors = np.max(
            np.abs(finite_gradient_array - production_pullback[None, :]),
            axis=1,
        )

        np.testing.assert_allclose(
            production_pullback,
            finite_gradient_array[-1],
            rtol=5e-7,
            atol=3e-9,
        )
        np.testing.assert_allclose(
            errors[1:4] / errors[:3],
            0.25,
            rtol=2e-2,
            atol=1e-3,
        )
        assert np.all(np.diff(errors) < 0.0)

    def test_jvp_rejects_directions_outside_the_fixed_chart(self) -> None:
        """Reject shape, off-shell, and non-Hermitian directions."""
        target = _create_target()
        source = _create_conforming_source(target).total_source

        with pytest.raises(ValueError, match="carrier_tangent"):
            galerkin_state_jvp(
                target,
                source,
                jnp.zeros(2),
                jnp.zeros(3, dtype=jnp.complex128),
                jnp.zeros(3, dtype=jnp.complex128),
            )

        with pytest.raises(
            (
                eqx.EquinoxRuntimeError,
                jax.errors.JaxRuntimeError,
                ValueError,
            ),
            match="tangent to the on-shell sphere",
        ):
            radial_result = galerkin_state_jvp(
                target,
                source,
                target.carrier,
                jnp.zeros(3, dtype=jnp.complex128),
                jnp.zeros(3, dtype=jnp.complex128),
            )
            jax.block_until_ready(radial_result[1])

        with pytest.raises(
            (
                eqx.EquinoxRuntimeError,
                jax.errors.JaxRuntimeError,
                ValueError,
            ),
            match="must be finite and Hermitian",
        ):
            result = galerkin_state_jvp(
                target,
                source,
                jnp.zeros(3),
                jnp.asarray([0.0j, 0.0j, 1.0j]),
                jnp.zeros(3, dtype=jnp.complex128),
            )
            jax.block_until_ready(result[1])
