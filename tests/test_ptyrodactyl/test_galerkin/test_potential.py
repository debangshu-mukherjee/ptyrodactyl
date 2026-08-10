"""Tests for :mod:`ptyrodactyl.galerkin.potential`.

Extended Summary
----------------
These tests compare endpoint-safe interaction and positive absorber actions
with independently assembled dense convolutions on a tiny fixed support.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from numpy.testing import assert_allclose

from ptyrodactyl.galerkin import (
    apply_absorber_action,
    apply_interaction_product,
    build_absorber_factor,
    build_cosine_shell_absorber_coefficients,
    build_interaction_coefficients,
)
from ptyrodactyl.types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _line_indices(values: Tuple[int, ...]) -> jax.Array:
    """Place one-dimensional exact indices on the final work-grid axis."""
    indices: jax.Array = jnp.asarray(
        [[0, 0, value] for value in values],
        dtype=jnp.int32,
    )
    return indices


def _support() -> GalerkinProductSupport:
    """Return one odd-grid support that distinguishes both endpoints."""
    support: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=_line_indices((-1, 0, 1)),
        interaction_indices=_line_indices((-1, 0, 1)),
        absorber_indices=_line_indices((-2, -1, 0, 1, 2)),
        work_indices=_line_indices((-3, -2, -1, 0, 1, 2, 3)),
        work_shape=(1, 1, 7),
    )
    return support


def _mixed_axis_support() -> GalerkinProductSupport:
    """Return an asymmetric state with a complete analytic absorber band."""
    state = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
    interaction = jnp.array(
        [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, -1], [0, 1, 1]]
    )
    absorber = jnp.array(
        [
            [first, second, third]
            for first in range(-1, 2)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ]
    )
    work = jnp.array(
        [
            [first, second, third]
            for first in range(-1, 3)
            for second in range(-1, 3)
            for third in range(-1, 3)
        ]
    )
    support = create_galerkin_product_support(
        state,
        interaction,
        absorber,
        work,
        (4, 4, 4),
    )
    return support


def _voltage_coefficients(
    amplitude: jax.Array | float = 0.2,
    imaginary: jax.Array | float = 0.1,
) -> jax.Array:
    """Return exact Hermitian SC.13b voltage coefficients in volts."""
    real_value: jax.Array = jnp.asarray(amplitude, dtype=jnp.float64)
    imaginary_value: jax.Array = jnp.asarray(imaginary, dtype=jnp.float64)
    coefficients: jax.Array = jnp.asarray(
        [
            real_value + 1j * imaginary_value,
            1.5 + 0.0j,
            real_value - 1j * imaginary_value,
        ],
        dtype=jnp.complex128,
    )
    return coefficients


def _absorber_coefficients(
    dc: jax.Array | float = 0.6,
    pair_real: jax.Array | float = 0.1,
    pair_imaginary: jax.Array | float = 0.02,
) -> jax.Array:
    """Return exact Hermitian coefficients of a positive compression."""
    center: jax.Array = jnp.asarray(dc, dtype=jnp.float64)
    real_value: jax.Array = jnp.asarray(pair_real, dtype=jnp.float64)
    imaginary_value: jax.Array = jnp.asarray(pair_imaginary, dtype=jnp.float64)
    coefficients: jax.Array = jnp.asarray(
        [
            0.03 + 0.0j,
            real_value + 1j * imaginary_value,
            center + 0.0j,
            real_value - 1j * imaginary_value,
            0.03 + 0.0j,
        ],
        dtype=jnp.complex128,
    )
    return coefficients


def _dense_convolution(
    state_indices: jax.Array,
    multiplier_indices: jax.Array,
    coefficients: jax.Array,
) -> np.ndarray:
    """Assemble an independent dense Toeplitz convolution by Python lookup."""
    state: np.ndarray = np.asarray(state_indices)
    multiplier: np.ndarray = np.asarray(multiplier_indices)
    values: np.ndarray = np.asarray(coefficients)
    lookup: Dict[Tuple[int, int, int], complex] = {
        (int(index[0]), int(index[1]), int(index[2])): complex(value)
        for index, value in zip(multiplier, values, strict=True)
    }
    matrix: np.ndarray = np.zeros(
        (state.shape[0], state.shape[0]),
        dtype=np.complex128,
    )
    for row, output_index in enumerate(state):
        for column, input_index in enumerate(state):
            delta: np.ndarray = output_index - input_index
            difference: Tuple[int, int, int] = (
                int(delta[0]),
                int(delta[1]),
                int(delta[2]),
            )
            matrix[row, column] = lookup.get(difference, 0.0j)
    return matrix


class TestScalarPotentialProducts:
    """Verify the SC-1 interaction and exact positive absorber products.

    :see: :func:`ptyrodactyl.galerkin.apply_absorber_action`
    :see: :func:`ptyrodactyl.galerkin.apply_interaction_product`
    :see: :func:`ptyrodactyl.galerkin.build_absorber_factor`
    :see: :func:`ptyrodactyl.galerkin.build_cosine_shell_absorber_coefficients`
    :see: :func:`ptyrodactyl.galerkin.build_interaction_coefficients`
    """

    def test_canonical_builders_and_actions_return_exact_widths(self) -> None:
        """Expose binary64 outputs while retaining a polymorphic diagnostic."""
        support = _support()
        voltage = _voltage_coefficients().astype(jnp.complex64)
        absorber = _absorber_coefficients().astype(jnp.complex64)
        field = jnp.asarray(
            [0.2 + 0.1j, -0.3 + 0.05j, 0.4 - 0.2j],
            dtype=jnp.complex64,
        )

        interaction = build_interaction_coefficients(support, voltage, 100.0)
        interaction_action = apply_interaction_product(
            support,
            voltage,
            field,
        )
        absorber_action = apply_absorber_action(support, absorber, field)
        shell = build_cosine_shell_absorber_coefficients(_mixed_axis_support())
        diagnostic_factor = build_absorber_factor(support, absorber)

        assert interaction.dtype == jnp.complex128
        assert interaction_action.dtype == jnp.complex128
        assert absorber_action.dtype == jnp.complex128
        assert shell.dtype == jnp.complex128
        assert diagnostic_factor.dtype == jnp.complex64

    def test_voltage_builder_has_sc1_units_and_positive_sign(self) -> None:
        """Match the 100 kV coupling and preserve positive voltage sign."""
        coefficients: jax.Array = build_interaction_coefficients(
            _support(),
            _voltage_coefficients(),
            100.0,
        )

        assert_allclose(
            np.asarray(coefficients / _voltage_coefficients()),
            np.full((3,), 0.31383),
            rtol=1.0e-4,
            atol=0.0,
        )
        assert float(jnp.real(coefficients[1])) > 0.0

    def test_interaction_builder_is_jit_and_vmap_clean(self) -> None:
        """Compile and batch the volts-to-interaction coefficient map."""
        support: GalerkinProductSupport = _support()
        voltage_batch: jax.Array = jnp.stack(
            [_voltage_coefficients(0.2), _voltage_coefficients(0.3)]
        )
        accelerating_voltages: jax.Array = jnp.asarray([100.0, 200.0])
        batched_builder = jax.jit(
            jax.vmap(
                build_interaction_coefficients,
                in_axes=(None, 0, 0),
            )
        )

        batched: jax.Array = batched_builder(
            support,
            voltage_batch,
            accelerating_voltages,
        )
        expected: jax.Array = jnp.stack(
            [
                build_interaction_coefficients(
                    support,
                    coefficients,
                    voltage,
                )
                for coefficients, voltage in zip(
                    voltage_batch,
                    accelerating_voltages,
                    strict=True,
                )
            ]
        )
        chex.assert_trees_all_close(batched, expected, rtol=1e-13)

    def test_interaction_matches_hand_computed_synthetic_field(self) -> None:
        """Match one basis-column convolution in inverse-square Angstroms."""
        support: GalerkinProductSupport = _support()
        coefficients: jax.Array = build_interaction_coefficients(
            support,
            _voltage_coefficients(),
            200.0,
        )
        center_mode: jax.Array = jnp.asarray(
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            dtype=jnp.complex128,
        )

        interaction: jax.Array = apply_interaction_product(
            support,
            coefficients,
            center_mode,
        )

        chex.assert_trees_all_close(interaction, coefficients, rtol=1e-13)

    def test_interaction_equals_independent_dense_convolution(self) -> None:
        """Match every endpoint-safe product against exact integer lookup."""
        support: GalerkinProductSupport = _support()
        coefficients: jax.Array = build_interaction_coefficients(
            support,
            _voltage_coefficients(),
            120.0,
        )
        field: jax.Array = jnp.asarray(
            [0.4 + 0.2j, -0.3 + 0.1j, 0.7 - 0.5j],
            dtype=jnp.complex128,
        )
        dense: np.ndarray = _dense_convolution(
            support.state_indices,
            support.interaction_indices,
            coefficients,
        )

        actual: jax.Array = apply_interaction_product(
            support,
            coefficients,
            field,
        )

        assert_allclose(
            np.asarray(actual),
            dense @ np.asarray(field),
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    def test_interaction_is_jit_and_vmap_clean(self) -> None:
        """Compile and batch the product without shape changes."""
        support: GalerkinProductSupport = _support()
        coefficients: jax.Array = build_interaction_coefficients(
            support,
            _voltage_coefficients(),
            100.0,
        )
        fields: jax.Array = jnp.asarray(
            [
                [0.4 + 0.2j, -0.3 + 0.1j, 0.7 - 0.5j],
                [-0.2 + 0.6j, 0.8 - 0.4j, 0.1 + 0.3j],
            ],
            dtype=jnp.complex128,
        )
        compiled = jax.jit(apply_interaction_product)
        batched = jax.jit(
            jax.vmap(
                apply_interaction_product,
                in_axes=(None, None, 0),
            )
        )

        expected: jax.Array = jnp.stack(
            [
                apply_interaction_product(support, coefficients, field)
                for field in fields
            ]
        )
        chex.assert_trees_all_close(
            compiled(support, coefficients, fields[0]),
            expected[0],
            rtol=1e-13,
        )
        chex.assert_trees_all_close(
            batched(support, coefficients, fields),
            expected,
            rtol=1e-13,
        )

    def test_product_indices_are_canonical_int64_before_flattening(
        self,
    ) -> None:
        """Avoid narrow-dtype overflow on a valid 80,000-point work grid."""
        half_period = jnp.array([199, 199, 1], dtype=jnp.int16)
        zero = jnp.zeros(3, dtype=jnp.int16)
        state = jnp.stack((zero, half_period))
        multiplier = jnp.stack((-half_period, zero, half_period))
        work = jnp.stack((-half_period, zero, half_period, 2 * half_period))
        support = create_galerkin_product_support(
            state_indices=state,
            interaction_indices=multiplier,
            absorber_indices=multiplier,
            work_indices=work,
            work_shape=(200, 200, 2),
        )
        coefficients = jnp.array(
            [0.2 + 0.0j, 1.0 + 0.0j, 0.2 + 0.0j],
            dtype=jnp.complex128,
        )
        field = jnp.array([0.4 + 0.1j, -0.2 + 0.3j])
        dense = _dense_convolution(
            support.state_indices,
            support.interaction_indices,
            coefficients,
        )

        actual = apply_interaction_product(support, coefficients, field)

        assert support.state_indices.dtype == jnp.int64
        assert_allclose(actual, dense @ np.asarray(field), atol=2.0e-14)

    @pytest.mark.parametrize("component", ["real", "imaginary"])
    def test_interaction_gradient_matches_finite_difference(
        self,
        component: str,
    ) -> None:
        """Match each paired coefficient direction at step 1e-5."""
        support: GalerkinProductSupport = _support()
        field: jax.Array = jnp.asarray(
            [0.4 + 0.2j, -0.3 + 0.1j, 0.7 - 0.5j],
            dtype=jnp.complex128,
        )

        def loss(value: jax.Array) -> jax.Array:
            voltage_coefficients: jax.Array
            if component == "real":
                voltage_coefficients = _voltage_coefficients(value, 0.1)
            else:
                voltage_coefficients = _voltage_coefficients(0.2, value)
            coefficients: jax.Array = build_interaction_coefficients(
                support,
                voltage_coefficients,
                100.0,
            )
            action: jax.Array = apply_interaction_product(
                support,
                coefficients,
                field,
            )
            objective: jax.Array = jnp.real(jnp.vdot(action, action))
            return objective

        parameter: jax.Array = jnp.asarray(0.2, dtype=jnp.float64)
        step: float = 1.0e-5
        automatic: jax.Array = jax.grad(loss)(parameter)
        finite_difference: jax.Array = (
            loss(parameter + step) - loss(parameter - step)
        ) / (2.0 * step)

        assert float(jnp.abs(automatic)) > 1.0e-5
        assert_allclose(
            np.asarray(automatic),
            np.asarray(finite_difference),
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    def test_absorber_factor_matches_exact_dense_convolution(self) -> None:
        """Match G*G to direct SC.13b compression and prove positivity."""
        support: GalerkinProductSupport = _support()
        coefficients: jax.Array = _absorber_coefficients()
        field: jax.Array = jnp.asarray(
            [0.2 - 0.4j, 0.6 + 0.1j, -0.3 + 0.5j],
            dtype=jnp.complex128,
        )
        dense: np.ndarray = _dense_convolution(
            support.state_indices,
            support.absorber_indices,
            coefficients,
        )

        factor: jax.Array = build_absorber_factor(support, coefficients)
        actual: jax.Array = apply_absorber_action(
            support,
            coefficients,
            field,
        )

        assert_allclose(
            np.asarray(jnp.conj(factor.T) @ factor),
            dense,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        assert_allclose(
            np.asarray(actual),
            dense @ np.asarray(field),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        assert float(jnp.real(jnp.vdot(field, actual))) > 0.0

    def test_analytic_shell_profile_and_mixed_axis_action_match_dense(
        self,
    ) -> None:
        """Pin the bounded shell formula on an asymmetric mixed-axis state."""
        support = _mixed_axis_support()
        coefficients = build_cosine_shell_absorber_coefficients(support)
        field = jnp.array(
            [0.2 - 0.4j, 0.6 + 0.1j, -0.3 + 0.5j],
            dtype=jnp.complex128,
        )
        dense = _dense_convolution(
            support.state_indices,
            support.absorber_indices,
            coefficients,
        )
        actual = apply_absorber_action(support, coefficients, field)
        normalized_points = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.17, -0.23, 0.31]]
        )
        indices = np.asarray(support.absorber_indices)
        synthesized = np.array(
            [
                np.sum(
                    np.asarray(coefficients)
                    * np.exp(2j * np.pi * (indices @ point))
                )
                for point in normalized_points
            ]
        )
        expected_profile = 1.0 - np.prod(
            np.cos(np.pi * normalized_points) ** 2, axis=-1
        )

        assert_allclose(actual, dense @ np.asarray(field), atol=2.0e-14)
        assert_allclose(synthesized.imag, 0.0, atol=2.0e-15)
        assert_allclose(synthesized.real, expected_profile, atol=2.0e-15)
        assert_allclose(synthesized.real[:2], [0.0, 1.0], atol=2.0e-15)

    def test_dense_absorber_validation_has_an_explicit_size_bound(
        self,
    ) -> None:
        """Reject dense validation beyond its documented 32-mode wall."""
        support = eqx.tree_at(
            lambda value: value.state_indices,
            _support(),
            jnp.zeros((33, 3), dtype=jnp.int64),
        )
        with pytest.raises(ValueError, match="limited to 32"):
            build_absorber_factor(support, _absorber_coefficients())

    def test_absorber_builder_and_action_are_jit_vmap_clean(self) -> None:
        """Compile coefficient factorization and batch the positive action."""
        support: GalerkinProductSupport = _support()
        coefficients: jax.Array = _absorber_coefficients()
        fields: jax.Array = jnp.asarray(
            [
                [0.2 - 0.4j, 0.6 + 0.1j, -0.3 + 0.5j],
                [0.1 + 0.2j, -0.4 + 0.3j, 0.7 - 0.2j],
            ],
            dtype=jnp.complex128,
        )
        coefficient_batch: jax.Array = jnp.stack(
            [coefficients, _absorber_coefficients(0.65)]
        )
        factors: jax.Array = jax.jit(
            jax.vmap(build_absorber_factor, in_axes=(None, 0))
        )(
            support,
            coefficient_batch,
        )
        assert factors.shape == (2, 3, 3)
        batched: jax.Array = jax.jit(
            jax.vmap(
                apply_absorber_action,
                in_axes=(None, None, 0),
            )
        )(support, coefficients, fields)
        expected: jax.Array = jnp.stack(
            [
                apply_absorber_action(support, coefficients, field)
                for field in fields
            ]
        )

        chex.assert_trees_all_close(batched, expected, rtol=1e-13)

    def test_absorber_gradient_matches_finite_difference(self) -> None:
        """Match the DC-profile derivative at step 1e-5 within 2e-8."""
        support: GalerkinProductSupport = _support()
        field: jax.Array = jnp.asarray(
            [0.2 - 0.4j, 0.6 + 0.1j, -0.3 + 0.5j],
            dtype=jnp.complex128,
        )

        def product_loss(dc: jax.Array) -> jax.Array:
            coefficients: jax.Array = _absorber_coefficients(dc)
            action: jax.Array = apply_absorber_action(
                support,
                coefficients,
                field,
            )
            objective: jax.Array = jnp.real(jnp.vdot(field, action))
            return objective

        def factor_loss(dc: jax.Array) -> jax.Array:
            factor: jax.Array = build_absorber_factor(
                support,
                _absorber_coefficients(dc),
            )
            factor_field: jax.Array = factor @ field
            objective: jax.Array = jnp.real(
                jnp.vdot(factor_field, factor_field)
            )
            return objective

        dc: jax.Array = jnp.asarray(0.6, dtype=jnp.float64)
        step: float = 1.0e-5
        automatic_product: jax.Array = jax.grad(product_loss)(dc)
        automatic_factor: jax.Array = jax.grad(factor_loss)(dc)
        finite_difference: jax.Array = (
            product_loss(dc + step) - product_loss(dc - step)
        ) / (2.0 * step)

        assert float(jnp.abs(automatic_product)) > 1.0e-5
        assert_allclose(
            np.asarray(automatic_product),
            np.asarray(finite_difference),
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        assert_allclose(
            np.asarray(automatic_factor),
            np.asarray(finite_difference),
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    @pytest.mark.parametrize("component", ["real", "imaginary"])
    def test_absorber_pair_gradient_matches_finite_difference(
        self,
        component: str,
    ) -> None:
        """Differentiate both real and imaginary Hermitian pair directions."""
        support: GalerkinProductSupport = _support()
        field: jax.Array = jnp.asarray(
            [0.2 - 0.4j, 0.6 + 0.1j, -0.3 + 0.5j],
            dtype=jnp.complex128,
        )

        def coefficients(value: jax.Array) -> jax.Array:
            """Vary one Hermitian off-diagonal coefficient component."""
            if component == "real":
                values: jax.Array = _absorber_coefficients(
                    pair_real=value,
                    pair_imaginary=0.02,
                )
            else:
                values = _absorber_coefficients(
                    pair_real=0.1,
                    pair_imaginary=value,
                )
            return values

        def product_loss(value: jax.Array) -> jax.Array:
            """Return the real absorber quadratic form."""
            action: jax.Array = apply_absorber_action(
                support,
                coefficients(value),
                field,
            )
            objective: jax.Array = jnp.real(jnp.vdot(field, action))
            return objective

        def factor_loss(value: jax.Array) -> jax.Array:
            """Return the same form through the bounded dense factor."""
            factor: jax.Array = build_absorber_factor(
                support,
                coefficients(value),
            )
            factor_field: jax.Array = factor @ field
            objective: jax.Array = jnp.real(
                jnp.vdot(factor_field, factor_field)
            )
            return objective

        parameter: jax.Array = jnp.asarray(
            0.1 if component == "real" else 0.02,
            dtype=jnp.float64,
        )
        step: float = 1.0e-5
        finite_difference: jax.Array = (
            product_loss(parameter + step) - product_loss(parameter - step)
        ) / (2.0 * step)
        automatic_product: jax.Array = jax.grad(product_loss)(parameter)
        automatic_factor: jax.Array = jax.grad(factor_loss)(parameter)

        assert float(jnp.abs(automatic_product)) > 1.0e-5
        assert_allclose(
            np.asarray(automatic_product),
            np.asarray(finite_difference),
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        assert_allclose(
            np.asarray(automatic_factor),
            np.asarray(finite_difference),
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    @pytest.mark.parametrize(
        ("builder", "coefficients", "message"),
        [
            (
                build_interaction_coefficients,
                jnp.asarray([0.2 + 0.1j, 1.5 + 0.0j, 0.4 - 0.1j]),
                "Hermitian",
            ),
            (
                build_absorber_factor,
                jnp.asarray([0.05, 0.15, 0.0, 0.15, 0.05]) + 0.0j,
                "positive definite",
            ),
        ],
    )
    def test_builders_reject_nonphysical_coefficients(
        self,
        builder: object,
        coefficients: jax.Array,
        message: str,
    ) -> None:
        """Reject a complex interaction or indefinite absorber compression."""
        with pytest.raises(_RUNTIME_ERRORS, match=message):
            if builder is build_interaction_coefficients:
                result: jax.Array = build_interaction_coefficients(
                    _support(),
                    coefficients,
                    100.0,
                )
            else:
                result = build_absorber_factor(_support(), coefficients)
            jax.block_until_ready(result)

    def test_interaction_builder_rejects_nonpositive_voltage(self) -> None:
        """Reject zero accelerating voltage before computing the coupling."""
        compiled = jax.jit(build_interaction_coefficients)
        with pytest.raises(_RUNTIME_ERRORS, match="finite and positive"):
            result: jax.Array = compiled(
                _support(),
                _voltage_coefficients(),
                jnp.asarray(0.0),
            )
            jax.block_until_ready(result)

    def test_interaction_builder_rejects_derived_range_loss(self) -> None:
        """Reject overflow and nonzero voltage coefficients lost to FTZ."""
        support = _support()
        huge = jnp.full(3, 1.0e10 + 0.0j, dtype=jnp.complex128)
        smallest_normal = jnp.full(
            3,
            jnp.finfo(jnp.float64).tiny + 0.0j,
            dtype=jnp.complex128,
        )
        for build in (
            build_interaction_coefficients,
            jax.jit(build_interaction_coefficients),
        ):
            with pytest.raises(_RUNTIME_ERRORS, match="derived interaction"):
                jax.block_until_ready(build(support, huge, 1.0e308))
            with pytest.raises(
                _RUNTIME_ERRORS,
                match="preserve every nonzero normal voltage component",
            ):
                jax.block_until_ready(build(support, smallest_normal, 200.0))

    def test_product_actions_reject_nonfinite_derived_outputs(self) -> None:
        """Reject overflow at both public endpoint-safe product boundaries."""
        support = _support()
        field = jnp.full(3, 1.0e200 + 0.0j, dtype=jnp.complex128)
        interaction = jnp.full(3, 1.0e200 + 0.0j, dtype=jnp.complex128)
        absorber = jnp.full(5, 1.0e200 + 0.0j, dtype=jnp.complex128)
        cases = (
            (apply_interaction_product, interaction),
            (apply_absorber_action, absorber),
        )
        for action, coefficients in cases:
            for call in (action, jax.jit(action)):
                with pytest.raises(
                    _RUNTIME_ERRORS, match="action must be finite"
                ):
                    jax.block_until_ready(call(support, coefficients, field))

    def test_product_actions_reject_incompatible_shapes(self) -> None:
        """Reject coefficient, factor, and state arrays with wrong lengths."""
        with pytest.raises(ValueError, match="interaction support"):
            apply_interaction_product(
                _support(),
                jnp.ones((2,), dtype=jnp.complex128),
                jnp.ones((3,), dtype=jnp.complex128),
            )
        with pytest.raises(ValueError, match="absorber support"):
            apply_absorber_action(
                _support(),
                jnp.ones((4,), dtype=jnp.complex128),
                jnp.ones((3,), dtype=jnp.complex128),
            )
