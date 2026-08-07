"""Tests for :mod:`ptyrodactyl.multislice.reduce`.

Extended Summary
----------------
These tests pin the late distribution reducer for hand-computed coherent
and incoherent cases, trivial identity behavior, gradients, JIT static
branching, multi-axis cursor semantics, and single-axis agreement. The bound
kernels here are intentionally toy closures, not multislice kernels, so the
tests assert that the reducer imposes no kernel-family-specific contract.

:see: :func:`ptyrodactyl.multislice.apply_distribution`
:see: :func:`ptyrodactyl.multislice.apply_distributions`
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.multislice import apply_distribution, apply_distributions
from ptyrodactyl.types import (
    TRIVIAL,
    ReductionMode,
    create_distribution,
)


def _scalar_sample_field(sample):
    """Return a deterministic 2x2 complex field from one scalar sample."""
    x = sample[0]
    row0 = jnp.stack((x + 1j * (x + 1.0), 2.0 * x - 0.5j))
    row1 = jnp.stack((x - 1.0j, -x + 0.25j))
    field = jnp.stack((row0, row1)).astype(jnp.complex128)
    return field


def _parameterized_field(sample, parameter):
    """Return a deterministic 2x2 complex field with one scalar parameter."""
    x = sample[0]
    y = sample[1]
    row0 = jnp.stack((parameter * x + 1j * (y + 1.0), x - 1j * parameter))
    row1 = jnp.stack((parameter + 0.5j * y, x + parameter * y + 1.0j))
    field = jnp.stack((row0, row1)).astype(jnp.complex128)
    return field


def _cursor_field(cursor):
    """Return a deterministic 2x2 complex field from a three-value cursor."""
    c = cursor[0]
    u = cursor[1]
    v = cursor[2]
    row0 = jnp.stack((c + 1j * u, v - 1j * c))
    row1 = jnp.stack((u * v + 0.25j, c + u + v + 0.5j))
    field = jnp.stack((row0, row1)).astype(jnp.complex128)
    return field


def test_apply_distribution_coherent_matches_hand_computed_sum_then_square():
    """Coherent reduction sums weighted amplitudes before squaring."""
    distribution = create_distribution(
        samples=jnp.asarray([[1.0], [2.0]], dtype=jnp.float64),
        weights=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        reduction=ReductionMode.COHERENT,
    )
    a0 = jnp.asarray(
        [[1.0 + 2.0j, 2.0 - 0.5j], [1.0 - 1.0j, -1.0 + 0.25j]],
        dtype=jnp.complex128,
    )
    a1 = jnp.asarray(
        [[2.0 + 3.0j, 4.0 - 0.5j], [2.0 - 1.0j, -2.0 + 0.25j]],
        dtype=jnp.complex128,
    )
    expected = jnp.abs(0.25 * a0 + 0.75 * a1) ** 2

    result = apply_distribution(distribution, _scalar_sample_field)

    chex.assert_trees_all_close(result, expected, rtol=1e-12, atol=1e-12)


def test_apply_distribution_incoherent_matches_hand_computed_square_then_sum():
    """Incoherent reduction squares each amplitude before weighted summing."""
    distribution = create_distribution(
        samples=jnp.asarray([[1.0], [2.0]], dtype=jnp.float64),
        weights=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        reduction=ReductionMode.INCOHERENT,
    )
    a0 = jnp.asarray(
        [[1.0 + 2.0j, 2.0 - 0.5j], [1.0 - 1.0j, -1.0 + 0.25j]],
        dtype=jnp.complex128,
    )
    a1 = jnp.asarray(
        [[2.0 + 3.0j, 4.0 - 0.5j], [2.0 - 1.0j, -2.0 + 0.25j]],
        dtype=jnp.complex128,
    )
    expected = 0.25 * jnp.abs(a0) ** 2 + 0.75 * jnp.abs(a1) ** 2

    result = apply_distribution(distribution, _scalar_sample_field)

    chex.assert_trees_all_close(result, expected, rtol=1e-12, atol=1e-12)


def test_apply_distribution_trivial_axis_is_identity_modulus_square():
    """A one-sample unit-weight axis reduces exactly to ``|bound(row)|^2``."""
    expected = jnp.abs(_scalar_sample_field(TRIVIAL.samples[0])) ** 2

    result = apply_distribution(TRIVIAL, _scalar_sample_field)

    chex.assert_trees_all_close(result, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "mode",
    [ReductionMode.COHERENT, ReductionMode.INCOHERENT],
)
def test_apply_distribution_gradients_are_finite_for_weights_and_bound_param(
    mode,
):
    """Gradients are finite through weights and bound-closure parameters."""
    samples = jnp.asarray(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.7]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float64)

    def objective_weights(weight_values):
        distribution = create_distribution(
            samples=samples,
            weights=weight_values,
            reduction=mode,
            axis_id="grad",
        )

        def bound(sample):
            return _parameterized_field(sample, jnp.asarray(0.8))

        value = jnp.sum(apply_distribution(distribution, bound))
        return value

    def objective_parameter(parameter):
        distribution = create_distribution(
            samples=samples,
            weights=weights,
            reduction=mode,
            axis_id="grad",
        )

        def bound(sample):
            return _parameterized_field(sample, parameter)

        value = jnp.sum(apply_distribution(distribution, bound))
        return value

    weight_gradient = jax.grad(objective_weights)(weights)
    parameter_gradient = jax.grad(objective_parameter)(
        jnp.asarray(0.8, dtype=jnp.float64),
    )

    chex.assert_shape(weight_gradient, (3,))
    chex.assert_tree_all_finite(weight_gradient)
    chex.assert_tree_all_finite(parameter_gradient)


@pytest.mark.parametrize(
    "mode",
    [ReductionMode.COHERENT, ReductionMode.INCOHERENT],
)
def test_apply_distribution_static_branch_jit_reuses_sample_value_trace(mode):
    """JIT compiles once per static mode when only sample values change."""
    weights = jnp.asarray([0.4, 0.6], dtype=jnp.float64)
    trace_count = {"count": 0}

    @jax.jit
    def reduce_samples(samples):
        trace_count["count"] += 1
        distribution = create_distribution(
            samples=samples,
            weights=weights,
            reduction=mode,
            axis_id="jit",
        )
        intensity = apply_distribution(distribution, _scalar_sample_field)
        return intensity

    first = reduce_samples(jnp.asarray([[1.0], [2.0]], dtype=jnp.float64))
    second = reduce_samples(jnp.asarray([[3.0], [4.0]], dtype=jnp.float64))
    first.block_until_ready()
    second.block_until_ready()

    assert trace_count["count"] == 1
    chex.assert_shape(first, (2, 2))
    chex.assert_shape(second, (2, 2))


def test_apply_distributions_mixed_two_by_two_matches_hand_computed_formula():
    """One coherent axis and one incoherent axis follow the nested formula."""
    coherent = create_distribution(
        samples=jnp.asarray([[1.0], [2.0]], dtype=jnp.float64),
        weights=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        reduction=ReductionMode.COHERENT,
        axis_id="coherent",
    )
    incoherent = create_distribution(
        samples=jnp.asarray([[10.0, -1.0], [20.0, 0.5]], dtype=jnp.float64),
        weights=jnp.asarray([0.4, 0.6], dtype=jnp.float64),
        reduction=ReductionMode.INCOHERENT,
        axis_id="incoherent",
    )
    a_c0_i0 = jnp.asarray(
        [[1.0 + 10.0j, -1.0 - 1.0j], [-10.0 + 0.25j, 10.0 + 0.5j]],
        dtype=jnp.complex128,
    )
    a_c1_i0 = jnp.asarray(
        [[2.0 + 10.0j, -1.0 - 2.0j], [-10.0 + 0.25j, 11.0 + 0.5j]],
        dtype=jnp.complex128,
    )
    a_c0_i1 = jnp.asarray(
        [[1.0 + 20.0j, 0.5 - 1.0j], [10.0 + 0.25j, 21.5 + 0.5j]],
        dtype=jnp.complex128,
    )
    a_c1_i1 = jnp.asarray(
        [[2.0 + 20.0j, 0.5 - 2.0j], [10.0 + 0.25j, 22.5 + 0.5j]],
        dtype=jnp.complex128,
    )
    expected = (
        0.4 * jnp.abs(0.25 * a_c0_i0 + 0.75 * a_c1_i0) ** 2
        + 0.6 * jnp.abs(0.25 * a_c0_i1 + 0.75 * a_c1_i1) ** 2
    )

    result = apply_distributions((coherent, incoherent), _cursor_field)

    chex.assert_trees_all_close(result, expected, rtol=1e-12, atol=1e-12)


def test_apply_distributions_cursor_concatenates_rows_in_tuple_order():
    """The bound cursor is the concatenation of axis rows in tuple order."""
    first_axis = create_distribution(
        samples=jnp.asarray([[2.0]], dtype=jnp.float64),
        weights=jnp.asarray([1.0], dtype=jnp.float64),
        reduction=ReductionMode.COHERENT,
        axis_id="first",
    )
    second_axis = create_distribution(
        samples=jnp.asarray([[3.0, 5.0]], dtype=jnp.float64),
        weights=jnp.asarray([1.0], dtype=jnp.float64),
        reduction=ReductionMode.COHERENT,
        axis_id="second",
    )

    def echo_cursor(cursor):
        row0 = jnp.stack((cursor[0] + 1j * cursor[1], cursor[2] + 0.0j))
        row1 = jnp.stack((cursor[0] + 1j * cursor[2], cursor[1] + 0.0j))
        field = jnp.stack((row0, row1)).astype(jnp.complex128)
        return field

    expected_field = jnp.asarray(
        [[2.0 + 3.0j, 5.0 + 0.0j], [2.0 + 5.0j, 3.0 + 0.0j]],
        dtype=jnp.complex128,
    )
    expected = jnp.abs(expected_field) ** 2

    result = apply_distributions((first_axis, second_axis), echo_cursor)

    chex.assert_trees_all_close(result, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "mode",
    [ReductionMode.COHERENT, ReductionMode.INCOHERENT],
)
def test_apply_distributions_single_axis_agrees_with_apply_distribution(mode):
    """A one-axis tuple agrees with the single-axis reducer for both modes."""
    distribution = create_distribution(
        samples=jnp.asarray([[-1.0], [0.5], [2.0]], dtype=jnp.float64),
        weights=jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float64),
        reduction=mode,
        axis_id="single",
    )

    single = apply_distribution(distribution, _scalar_sample_field)
    nested = apply_distributions((distribution,), _scalar_sample_field)

    chex.assert_trees_all_close(nested, single, rtol=1e-12, atol=1e-12)
