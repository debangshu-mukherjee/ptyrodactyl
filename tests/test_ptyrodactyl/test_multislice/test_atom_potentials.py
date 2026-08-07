"""Tests for atom_potentials module.

:see: :func:`ptyrodactyl.multislice.single_atom_potential`
"""
# ruff: noqa: E402

import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.multislice.atom_potentials import (
    _slice_atoms,
    bessel_kv,
    crystal_potential_slices,
    single_atom_potential,
)
from ptyrodactyl.types import (
    CrystalData,
    PotentialSlices,
    create_crystal_data,
)


@pytest.mark.parametrize(
    ("atomic_number", "expected_sum", "expected_maximum"),
    [
        (6, 3304.630500664369, 599.0660669230273),
        (14, 7467.50633667953, 1309.8209272295387),
        (79, 16835.066119859865, 6296.83419348897),
    ],
)
def test_explicit_kirkland_matches_captured_element_reference(
    atomic_number: int,
    expected_sum: float,
    expected_maximum: float,
) -> None:
    """Explicit Kirkland preserves the captured C/Si/Au numerical path."""
    potential = single_atom_potential(
        atomic_number,
        0.2,
        (16, 16),
        supersampling=2,
        parameterization="kirkland",
    )

    np.testing.assert_allclose(
        np.asarray(jnp.sum(potential)),
        expected_sum,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.max(potential)),
        expected_maximum,
        rtol=2e-15,
    )


@pytest.mark.parametrize("parameterization", ["lobato", "kirkland"])
def test_single_atom_explicit_zero_center_preserves_centered_kernel(
    parameterization: str,
) -> None:
    """An explicit origin retains the established centered-kernel result."""
    omitted_center = single_atom_potential(
        14,
        0.2,
        (16, 16),
        supersampling=2,
        parameterization=parameterization,
    )
    explicit_zero = single_atom_potential(
        14,
        0.2,
        (16, 16),
        center_coords=jnp.zeros(2, dtype=jnp.float64),
        supersampling=2,
        parameterization=parameterization,
    )

    np.testing.assert_array_equal(
        np.asarray(explicit_zero),
        np.asarray(omitted_center),
    )


@pytest.mark.parametrize("parameterization", ["lobato", "kirkland"])
def test_single_atom_center_translation_is_material_and_differentiable(
    parameterization: str,
) -> None:
    """Both projected models retain their public sub-pixel position seam."""
    center = jnp.array([0.13, -0.17], dtype=jnp.float64)
    zero_center = jnp.zeros(2, dtype=jnp.float64)
    axis = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float64)
    y_weights, x_weights = jnp.meshgrid(axis, axis, indexing="ij")
    weights = (
        0.37 * x_weights - 0.61 * y_weights + 0.23 * x_weights * y_weights
    )

    def loss(center_coords: jax.Array) -> jax.Array:
        potential = single_atom_potential(
            14,
            0.2,
            (16, 16),
            center_coords=center_coords,
            supersampling=2,
            parameterization=parameterization,
        )
        return jnp.mean(weights * potential)

    centered = single_atom_potential(
        14,
        0.2,
        (16, 16),
        center_coords=zero_center,
        supersampling=2,
        parameterization=parameterization,
    )
    shifted = single_atom_potential(
        14,
        0.2,
        (16, 16),
        center_coords=center,
        supersampling=2,
        parameterization=parameterization,
    )
    relative_change = jnp.linalg.norm(shifted - centered) / jnp.linalg.norm(
        centered
    )
    gradient = jax.grad(loss)(center)
    direction = jnp.array([0.6, -0.8], dtype=jnp.float64)
    step = 1e-5
    finite_difference = (
        loss(center + step * direction) - loss(center - step * direction)
    ) / (2.0 * step)

    assert relative_change > 1e-3
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 1e-6
    np.testing.assert_allclose(
        np.asarray(jnp.vdot(gradient, direction)),
        np.asarray(finite_difference),
        rtol=2e-5,
        atol=1e-7,
    )


class TestBesselKv(chex.TestCase):
    """Test suite for the bessel_kv function.

    :see: :func:`ptyrodactyl.multislice.bessel_kv`
    """

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    @parameterized.parameters(
        (0.1, 2.4270690, 1e-5),
        (0.2, 1.7527038, 1e-5),
        (0.5, 0.9244190, 1e-5),
        (1.0, 0.4210244, 1e-5),
        (1.5, 0.2138056, 1e-2),
        (2.0, 0.1138938, 5e-3),
        (3.0, 0.0347395, 1e-3),
        (5.0, 0.0036911, 1e-3),
        (10.0, 0.0000778, 1e-4),
    )
    def test_bessel_k0_accuracy(self, x, expected, tol) -> None:
        """Test K_0(x) against known values."""
        x_array = jnp.asarray(x, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_computed = self.variant(bessel_kv)(v_scalar, x_array)

        self.assertAlmostEqual(
            k0_computed,
            expected,
            delta=tol,
            msg=f"K_0({x}) = {k0_computed:.8f}, expected {expected:.8f}",
        )

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    @parameterized.parameters(
        (0.01,),
        (0.1,),
        (0.5,),
        (1.0,),
        (2.0,),
        (5.0,),
        (10.0,),
    )
    def test_bessel_k0_derivative(self, x) -> None:
        """Test that K_0'(x) = -K_1(x) property holds numerically."""
        x_array = jnp.asarray(x, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)

        grad_fn = jax.grad(lambda y: self.variant(bessel_kv)(v_scalar, y))
        dk0_dx = grad_fn(x_array)

        assert jnp.isfinite(dk0_dx), f"K_0'({x}) must be finite"
        assert dk0_dx < 0.0, f"K_0'({x}) should be negative"
        k1_x = self.variant(bessel_kv)(jnp.asarray(1.0), x_array)
        assert jnp.allclose(dk0_dx, -k1_x, rtol=1e-5, atol=1e-7)

        if abs(x - 1.0) > 0.01:
            eps = 1e-4
            k0_x = self.variant(bessel_kv)(v_scalar, x_array)
            k0_x_plus = self.variant(bessel_kv)(v_scalar, x_array + eps)
            assert k0_x_plus < k0_x, f"K_0 should be decreasing at x={x}"

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_bessel_k0_vectorization(self) -> None:
        """Test that bessel_kv properly handles vector inputs."""
        x_1d = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_1d = self.variant(bessel_kv)(v_scalar, x_1d)
        assert k0_1d.shape == x_1d.shape

        x_2d = jnp.array(
            [[0.1, 0.5], [1.0, 2.0], [3.0, 5.0]], dtype=jnp.float64
        )
        k0_2d = self.variant(bessel_kv)(v_scalar, x_2d)
        assert k0_2d.shape == x_2d.shape

        x_3d = jnp.ones((2, 3, 4), dtype=jnp.float64)
        k0_3d = self.variant(bessel_kv)(v_scalar, x_3d)
        assert k0_3d.shape == x_3d.shape

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_bessel_k0_small_x_behavior(self) -> None:
        """Test K_0(x) behavior for small x: K_0(x) ~ -log(x/2) - gamma."""
        gamma_euler = 0.5772156649015329
        x_small = jnp.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)

        k0_values = self.variant(bessel_kv)(v_scalar, x_small)
        expected_approx = -jnp.log(x_small / 2.0) - gamma_euler

        relative_errors = jnp.abs(
            (k0_values - expected_approx) / expected_approx
        )
        assert jnp.all(relative_errors < 0.01), (
            "Small x approximation failed: "
            f"max error = {jnp.max(relative_errors):.6f}"
        )

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_bessel_k0_large_x_behavior(self) -> None:
        """Test K_0(x) asymptotic behavior for large x."""
        x_large = jnp.array([10.0, 20.0, 50.0, 100.0], dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)

        k0_values = self.variant(bessel_kv)(v_scalar, x_large)
        expected_asymptotic = jnp.sqrt(jnp.pi / (2 * x_large)) * jnp.exp(
            -x_large
        )

        relative_errors = jnp.abs(
            (k0_values - expected_asymptotic) / expected_asymptotic
        )
        assert jnp.all(relative_errors < 0.1), (
            "Large x asymptotic failed: "
            f"max error = {jnp.max(relative_errors):.6f}"
        )

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_bessel_k0_positivity(self) -> None:
        """Test that K_0(x) is always positive for x > 0."""
        x_test = jnp.logspace(-3, 2, 50, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_values = self.variant(bessel_kv)(v_scalar, x_test)

        assert jnp.all(k0_values > 0), "K_0(x) must be positive for all x > 0"

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_bessel_k0_monotonicity(self) -> None:
        """Test that K_0(x) is strictly decreasing."""
        x_test = jnp.linspace(0.1, 10.0, 50, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_values = self.variant(bessel_kv)(v_scalar, x_test)

        differences = jnp.diff(k0_values)
        assert jnp.all(differences < 0), "K_0(x) must be strictly decreasing"

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    @parameterized.parameters(
        (jnp.float32,),
        (
            (jnp.float64,)
            if getattr(jax.config, "jax_enable_x64", False)
            else (jnp.float32,)
        ),
    )
    def test_bessel_k0_dtype_consistency(self, dtype) -> None:
        """Test that output dtype matches input dtype."""
        x = jnp.array([0.5, 1.0, 2.0], dtype=dtype)
        v_scalar = jnp.asarray(0.0, dtype=dtype)
        k0_values = self.variant(bessel_kv)(v_scalar, x)

        assert k0_values.dtype == dtype

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    @parameterized.parameters(
        (0.5, 0.5, 1.0750476, 1e-5),
        (0.5, 1.0, 0.4610685, 1e-5),
        (0.5, 2.0, 0.1199377, 1e-5),
        (0.5, 5.0, 0.0037766, 1e-5),
        (1.0, 0.5, 1.6564411, 1e-5),
        (1.0, 1.0, 0.6019072, 1e-5),
        (1.0, 2.0, 0.1398659, 1e-5),
        (1.0, 5.0, 0.0040446, 1e-5),
        (2.0, 1.0, 1.6248389, 1e-5),
        (2.0, 2.0, 0.2537598, 1e-5),
        (2.0, 5.0, 0.0053089, 1e-5),
    )
    def test_bessel_kv_general_order(self, v, x, expected, tol) -> None:
        """Test K_v(x) for general orders against known values."""
        x_array = jnp.asarray(x, dtype=jnp.float64)
        v_scalar = jnp.asarray(v, dtype=jnp.float64)
        kv_computed = self.variant(bessel_kv)(v_scalar, x_array)

        self.assertAlmostEqual(
            kv_computed,
            expected,
            delta=tol,
            msg=f"K_{v}({x}) = {kv_computed:.8f}, expected {expected:.8f}",
        )

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_bessel_k0_continuity_at_transition(self) -> None:
        """Test continuity at the transition point x=1.0."""
        x_before = 0.999
        x_at = 1.0
        x_after = 1.001

        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_before = self.variant(bessel_kv)(
            v_scalar, jnp.asarray(x_before, dtype=jnp.float64)
        )
        k0_at = self.variant(bessel_kv)(
            v_scalar, jnp.asarray(x_at, dtype=jnp.float64)
        )
        k0_after = self.variant(bessel_kv)(
            v_scalar, jnp.asarray(x_after, dtype=jnp.float64)
        )

        self.assertAlmostEqual(k0_before, k0_at, delta=2e-2)
        self.assertAlmostEqual(k0_at, k0_after, delta=2e-2)

        jump_before = abs(k0_at - k0_before)
        jump_after = abs(k0_after - k0_at)
        assert jump_before < 0.02
        assert jump_after < 0.02


class TestSliceAtoms(chex.TestCase):
    """Test suite for the _slice_atoms function."""

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_simple_slicing(self) -> None:
        """Test basic slicing with evenly spaced atoms."""
        # Create 3 atoms at z = 0, 1, 2 Angstroms
        coords = jnp.array(
            [
                [1.0, 2.0, 0.0],  # Atom at z=0
                [3.0, 4.0, 1.0],  # Atom at z=1
                [5.0, 6.0, 2.0],  # Atom at z=2
            ]
        )
        atom_numbers = jnp.array([6, 8, 14])  # C, O, Si
        slice_thickness = 1.0

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Check shape
        assert result.shape == (3, 4)

        # Check that atoms are sorted by slice number
        slice_nums = result[:, 2]
        chex.assert_trees_all_equal(slice_nums, jnp.array([0.0, 1.0, 2.0]))

        # Check x, y coordinates are preserved
        chex.assert_trees_all_close(result[:, 0], jnp.array([1.0, 3.0, 5.0]))
        chex.assert_trees_all_close(result[:, 1], jnp.array([2.0, 4.0, 6.0]))

        # Check atom numbers are preserved
        chex.assert_trees_all_equal(result[:, 3], jnp.array([6.0, 8.0, 14.0]))

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_multiple_atoms_per_slice(self) -> None:
        """Test slicing when multiple atoms fall in the same slice."""
        coords = jnp.array(
            [
                [1.0, 1.0, 0.1],  # Slice 0
                [2.0, 2.0, 0.2],  # Slice 0
                [3.0, 3.0, 1.5],  # Slice 1
                [4.0, 4.0, 0.3],  # Slice 0
                [5.0, 5.0, 1.8],  # Slice 1
            ]
        )
        atom_numbers = jnp.array([1, 2, 3, 4, 5])
        slice_thickness = 1.0

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Check that atoms are grouped by slice
        slice_nums = result[:, 2]
        expected_slices = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
        chex.assert_trees_all_equal(slice_nums, expected_slices)

        # Verify atoms within same slice maintain their relative order
        slice_0_atoms = result[result[:, 2] == 0]
        chex.assert_trees_all_equal(
            slice_0_atoms[:, 3], jnp.array([1.0, 2.0, 4.0])
        )

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_non_uniform_z_distribution(self) -> None:
        """Test slicing with non-uniformly distributed z coordinates."""
        coords = jnp.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.1],
                [3.0, 3.0, 2.5],
                [4.0, 4.0, 2.6],
                [5.0, 5.0, 5.0],
            ]
        )
        atom_numbers = jnp.array([79, 79, 79, 79, 79])  # All gold atoms
        slice_thickness = 1.0

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Expected slices: 0, 0, 2, 2, 5
        expected_slices = jnp.array([0.0, 0.0, 2.0, 2.0, 5.0])
        chex.assert_trees_all_equal(result[:, 2], expected_slices)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    @parameterized.parameters(
        (0.5,),
        (1.0,),
        (2.0,),
        (0.1,),
        (10.0,),
    )
    def test_different_slice_thicknesses(self, slice_thickness) -> None:
        """Test slicing with various slice thicknesses."""
        # Create atoms from z=0 to z=10 with 0.5 spacing
        n_atoms = 21
        z_coords = jnp.linspace(0, 10, n_atoms)
        coords = jnp.column_stack(
            [
                jnp.ones(n_atoms),  # x coordinates
                jnp.ones(n_atoms) * 2,  # y coordinates
                z_coords,  # z coordinates
            ]
        )
        atom_numbers = jnp.ones(n_atoms, dtype=jnp.int32) * 6  # All carbon

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Check that slice numbers are monotonically increasing
        slice_nums = result[:, 2]
        assert jnp.all(jnp.diff(slice_nums) >= 0)

        # Check that max slice number is correct
        expected_max_slice = jnp.floor(10.0 / slice_thickness)
        actual_max_slice = jnp.max(slice_nums)
        chex.assert_trees_all_equal(actual_max_slice, expected_max_slice)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_single_atom(self) -> None:
        """Test slicing with a single atom."""
        coords = jnp.array([[1.5, 2.5, 3.5]])
        atom_numbers = jnp.array([26])  # Iron
        slice_thickness = 1.0

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        assert result.shape == (1, 4)
        chex.assert_trees_all_equal(result, jnp.array([[1.5, 2.5, 0.0, 26.0]]))

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_atoms_at_slice_boundaries(self) -> None:
        """Test atoms exactly at slice boundaries."""
        coords = jnp.array(
            [
                [1.0, 1.0, 0.0],  # Exactly at z=0
                [2.0, 2.0, 1.0],  # Exactly at boundary between slice 0 and 1
                [3.0, 3.0, 2.0],  # Exactly at boundary between slice 1 and 2
                [4.0, 4.0, 0.999],  # Just before boundary
                [5.0, 5.0, 1.001],  # Just after boundary
            ]
        )
        atom_numbers = jnp.array([1, 2, 3, 4, 5])
        slice_thickness = 1.0

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Atoms at boundaries should be assigned to lower slice
        expected_slices = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0])
        chex.assert_trees_all_equal(result[:, 2], expected_slices)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_negative_z_coordinates(self) -> None:
        """Test slicing with negative z coordinates."""
        coords = jnp.array(
            [
                [1.0, 1.0, -2.0],
                [2.0, 2.0, -1.0],
                [3.0, 3.0, 0.0],
                [4.0, 4.0, 1.0],
                [5.0, 5.0, 2.0],
            ]
        )
        atom_numbers = jnp.array([6, 7, 8, 9, 10])
        slice_thickness = 1.0

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Slices should start from 0 regardless of actual z values
        expected_slices = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        chex.assert_trees_all_equal(result[:, 2], expected_slices)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_dtype_consistency(self) -> None:
        """Test that output dtype is always float32."""
        # Test with float32 input
        coords_f32 = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32
        )
        atom_numbers = jnp.array([1, 2], dtype=jnp.int32)
        slice_thickness_f32 = jnp.array(1.0, dtype=jnp.float32)

        result_f32 = self.variant(_slice_atoms)(
            coords_f32, atom_numbers, slice_thickness_f32
        )
        assert result_f32.dtype == jnp.float32

        # Test with float64 input if x64 is enabled
        if getattr(jax.config, "jax_enable_x64", False):
            coords_f64 = jnp.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64
            )
            slice_thickness_f64 = jnp.array(1.0, dtype=jnp.float64)

            result_f64 = self.variant(_slice_atoms)(
                coords_f64, atom_numbers, slice_thickness_f64
            )
            # Note: When x64 is enabled, JAX preserves float64 precision
            assert result_f64.dtype == jnp.float64

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_scalar_numeric_slice_thickness(self) -> None:
        """Test that slice_thickness works with different scalar types."""
        coords = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, 2.5]])
        atom_numbers = jnp.array([1, 2])

        # Test with float
        result_float = self.variant(_slice_atoms)(coords, atom_numbers, 1.0)

        # Test with int
        result_int = self.variant(_slice_atoms)(coords, atom_numbers, 1)

        # Test with JAX scalar
        result_jax = self.variant(_slice_atoms)(
            coords, atom_numbers, jnp.array(1.0)
        )

        # All should give same result
        chex.assert_trees_all_equal(result_float, result_int)
        chex.assert_trees_all_equal(result_float, result_jax)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_large_atom_count(self) -> None:
        """Test slicing with a large number of atoms."""
        n_atoms = 1000
        # Random positions
        key = jax.random.PRNGKey(42)
        x_key, y_key, z_key = jax.random.split(key, 3)

        coords = jnp.column_stack(
            [
                jax.random.uniform(x_key, (n_atoms,), minval=0, maxval=10),
                jax.random.uniform(y_key, (n_atoms,), minval=0, maxval=10),
                jax.random.uniform(z_key, (n_atoms,), minval=0, maxval=5),
            ]
        )
        atom_numbers = jax.random.randint(
            jax.random.PRNGKey(43), (n_atoms,), minval=1, maxval=100
        )
        slice_thickness = 0.5

        result = self.variant(_slice_atoms)(
            coords, atom_numbers, slice_thickness
        )

        # Check shape
        assert result.shape == (n_atoms, 4)

        # Check that slices are sorted
        slice_nums = result[:, 2]
        assert jnp.all(jnp.diff(slice_nums) >= 0)

        # Check that all atoms are accounted for
        unique_atom_indices = jnp.unique(result[:, 3])
        assert len(unique_atom_indices) == len(jnp.unique(atom_numbers))


class TestCrystalPotentialSlices(chex.TestCase):
    """Test suite for the crystal_potential_slices function.

    :see: :func:`ptyrodactyl.multislice.crystal_potential_slices`
    """

    def _run_potential(
        self,
        crystal_data: CrystalData,
        pixel_size: float,
        slice_thickness: float = 1.0,
        *,
        padding: float = 4.0,
        repeats: tuple[int, int, int] = (1, 1, 1),
        grid_shape: tuple[int, int, int],
        parameterization: str = "kirkland",
    ) -> PotentialSlices:
        """Run a variant with the explicit static shape required under JIT."""
        repeats_array = jnp.asarray(repeats, dtype=jnp.int32)
        traced_variants = {
            chex.ChexVariantType.WITH_JIT,
            chex.ChexVariantType.WITH_PMAP,
        }
        variant_type = getattr(self.variant, "type", None)
        if variant_type in traced_variants:

            def fixed_shape_potential(data: CrystalData) -> PotentialSlices:
                return crystal_potential_slices(
                    data,
                    pixel_size,
                    slice_thickness,
                    repeats=repeats_array,
                    padding=padding,
                    grid_shape=grid_shape,
                    parameterization=parameterization,
                )

            return self.variant(fixed_shape_potential)(crystal_data)

        return self.variant(crystal_potential_slices)(
            crystal_data,
            pixel_size,
            slice_thickness,
            repeats=repeats_array,
            padding=padding,
            parameterization=parameterization,
        )

    def test_default_parameterization_is_lobato(self) -> None:
        """Assert every projected entry point defaults to Lobato."""
        default = single_atom_potential(
            atom_no=14,
            pixel_size=0.2,
            grid_shape=(16, 16),
            supersampling=2,
        )
        explicit_lobato = single_atom_potential(
            atom_no=14,
            pixel_size=0.2,
            grid_shape=(16, 16),
            supersampling=2,
            parameterization="lobato",
        )
        chex.assert_trees_all_equal(default, explicit_lobato)

        crystal = create_crystal_data(
            positions=jnp.array(
                [[0.5, 0.5, 0.2], [1.0, 1.0, 0.7]],
                dtype=jnp.float64,
            ),
            atomic_numbers=jnp.array([6, 14], dtype=jnp.int32),
            lattice=jnp.eye(3, dtype=jnp.float64) * 2.0,
        )
        default_slices = crystal_potential_slices(
            crystal,
            0.25,
            padding=0.25,
            supersampling=1,
            grid_shape=(4, 4, 1),
        )
        explicit_lobato_slices = crystal_potential_slices(
            crystal,
            0.25,
            padding=0.25,
            supersampling=1,
            grid_shape=(4, 4, 1),
            parameterization="lobato",
        )
        chex.assert_trees_all_equal(
            default_slices,
            explicit_lobato_slices,
        )

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_single_atom_centered(self) -> None:
        """Test potential generation for a single atom at the center."""

        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([1])  # Hydrogen
        xyz_data = create_crystal_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=None,
            stress=None,
            energy=None,
            properties=None,
            comment=None,
        )

        pixel_size = 0.1  # Angstroms
        slice_thickness = 1.0
        padding = 2.0

        result = self._run_potential(
            xyz_data,
            pixel_size,
            slice_thickness,
            padding=padding,
            grid_shape=(41, 41, 1),
        )

        # Check that result has correct structure
        chex.assert_type(result.slices, jnp.float64)
        assert len(result.slices.shape) == 3  # H x W x S
        assert result.slice_thickness == slice_thickness
        assert result.calib == pixel_size

        # Check that potential is centered and symmetric
        h, w, s = result.slices.shape
        center_h, center_w = h // 2, w // 2
        potential_slice = result.slices[:, :, 0]

        # Maximum should be at center
        max_idx = jnp.unravel_index(
            jnp.argmax(potential_slice), potential_slice.shape
        )
        self.assertAlmostEqual(max_idx[0], center_h, delta=1)
        self.assertAlmostEqual(max_idx[1], center_w, delta=1)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_multiple_atoms_different_slices(self) -> None:
        """Test potential generation for atoms in different slices."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # First slice
                [2.0, 2.0, 1.5],  # Second slice
                [1.0, 1.0, 3.0],  # Third slice
            ]
        )
        atomic_numbers = jnp.array([1, 6, 14])  # H, C, Si
        xyz_data = create_crystal_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=None,
            stress=None,
            energy=None,
            properties=None,
            comment=None,
        )

        pixel_size = 0.2
        slice_thickness = 1.0

        result = self._run_potential(
            xyz_data,
            pixel_size,
            slice_thickness,
            grid_shape=(50, 50, 4),
        )

        # Should have 4 slices (z from 0 to 3)
        assert result.slices.shape[2] == 4

        # The atoms at z = 0, 1.5, and 3 occupy slices 0, 1, and 3.
        slice_sums = jnp.sum(result.slices, axis=(0, 1))
        assert jnp.all(slice_sums[jnp.array([0, 1, 3])] > 0.0)
        assert slice_sums[2] == 0.0

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_same_atoms_same_slice(self) -> None:
        """Test multiple atoms of same type in same slice."""
        positions = jnp.array(
            [
                [1.0, 1.0, 0.5],
                [3.0, 1.0, 0.5],
                [2.0, 3.0, 0.5],
            ]
        )
        atomic_numbers = jnp.array([6, 6, 6])  # All carbon
        xyz_data = create_crystal_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=None,
            stress=None,
            energy=None,
            properties=None,
            comment=None,
        )

        pixel_size = 0.1
        slice_thickness = 1.0

        result = self._run_potential(
            xyz_data,
            pixel_size,
            slice_thickness,
            grid_shape=(100, 100, 1),
        )

        # Should have 1 slice
        assert result.slices.shape[2] == 1

        # Check that potential is sum of contributions
        potential = result.slices[:, :, 0]
        assert jnp.sum(potential) > 0.0

        # Find peaks corresponding to atom positions
        # Due to FFT shifting, peaks should be at atom positions
        h, w = potential.shape
        x_min = jnp.min(positions[:, 0]) - 4.0  # Default padding
        y_min = jnp.min(positions[:, 1]) - 4.0

        # Check that we have local maxima near atom positions
        for pos in positions:
            pixel_x = int((pos[0] - x_min) / pixel_size)
            pixel_y = int((pos[1] - y_min) / pixel_size)
            if 1 < pixel_x < w - 1 and 1 < pixel_y < h - 1:
                local_value = potential[pixel_y, pixel_x]
                neighbors = [
                    potential[pixel_y - 1, pixel_x],
                    potential[pixel_y + 1, pixel_x],
                    potential[pixel_y, pixel_x - 1],
                    potential[pixel_y, pixel_x + 1],
                ]
                # Value at atom position should be local maximum
                assert all(local_value >= n for n in neighbors)

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_padding_removal(self) -> None:
        """Test that padding is correctly removed from final result."""
        positions = jnp.array([[2.0, 2.0, 0.0]])
        atomic_numbers = jnp.array([1])
        xyz_data = create_crystal_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=None,
            stress=None,
            energy=None,
            properties=None,
            comment=None,
        )

        pixel_size = 0.5
        padding = 3.0

        result = self._run_potential(
            xyz_data,
            pixel_size,
            padding=padding,
            grid_shape=(13, 13, 1),
        )

        # Expected size after padding removal
        x_range = 0.0  # Single atom has no x range
        y_range = 0.0  # Single atom has no y range
        int(jnp.ceil(x_range / pixel_size))
        int(jnp.ceil(y_range / pixel_size))

        # Size should be at least 1x1
        assert result.slices.shape[0] >= 1
        assert result.slices.shape[1] >= 1

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_different_atomic_numbers(self) -> None:
        """Test that different atomic species produce different potentials."""
        # Test with single atoms of different types at same position
        pixel_size = 0.1
        slice_thickness = 1.0

        potentials = []
        for z in [1, 6, 14, 79]:  # H, C, Si, Au
            positions = jnp.array([[0.0, 0.0, 0.0]])
            atomic_numbers = jnp.array([z])
            xyz_data = create_crystal_data(
                positions=positions,
                atomic_numbers=atomic_numbers,
                lattice=None,
                stress=None,
                energy=None,
                properties=None,
                comment=None,
            )

            result = self._run_potential(
                xyz_data,
                pixel_size,
                slice_thickness,
                grid_shape=(81, 81, 1),
            )
            potentials.append(jnp.max(result.slices[:, :, 0]))

        # Heavier atoms should have stronger potentials
        for i in range(len(potentials) - 1):
            assert potentials[i] < potentials[i + 1]

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_vmap_compatibility(self) -> None:
        """Test that function works with vmap over multiple structures."""

        n_structures = 3
        positions_batch = jnp.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0]],
                [[2.0, 2.0, 0.0]],
            ]
        )
        atomic_numbers_batch = jnp.array([[1], [6], [14]])

        pixel_size = 0.2

        def process_single(pos, atoms):
            xyz_data = create_crystal_data(
                positions=pos,
                atomic_numbers=atoms,
                lattice=None,
                stress=None,
                energy=None,
                properties=None,
                comment=None,
            )
            return self._run_potential(
                xyz_data,
                pixel_size,
                grid_shape=(41, 41, 1),
            )

        # Process each structure
        results = []
        for i in range(n_structures):
            results.append(
                process_single(positions_batch[i], atomic_numbers_batch[i])
            )

        for result in results:
            assert result is not None
            assert jnp.sum(result.slices) > 0.0

    @chex.variants(
        with_jit=True, without_jit=True, with_device=True, with_pmap=True
    )
    def test_repeats_with_default_lattice(self) -> None:
        """Test that repeats work with default identity lattice."""
        positions = jnp.array([[0.5, 0.5, 0.0]])
        atomic_numbers = jnp.array([6])  # Carbon

        # Not providing lattice - will get identity matrix
        xyz_data = create_crystal_data(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=None,  # Will default to identity
            stress=None,
            energy=None,
            properties=None,
            comment=None,
        )

        pixel_size = 0.1
        repeats = (2, 2, 1)

        # Should work fine with default identity lattice
        result = self._run_potential(
            xyz_data,
            pixel_size,
            repeats=repeats,
            grid_shape=(90, 90, 1),
        )

        # With identity lattice and repeats [2,2,1], atoms will be at:
        # (0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)
        assert result is not None
        assert jnp.sum(result.slices) > 0.0

        # Check we have one slice (all atoms at z=0)
        assert result.slices.shape[2] == 1

        expected_data = create_crystal_data(
            positions=jnp.array(
                [
                    [0.5, 0.5, 0.0],
                    [0.5, 1.5, 0.0],
                    [1.5, 0.5, 0.0],
                    [1.5, 1.5, 0.0],
                ]
            ),
            atomic_numbers=jnp.full(4, 6),
        )
        expected = self._run_potential(
            expected_data,
            pixel_size,
            grid_shape=(90, 90, 1),
        )
        chex.assert_trees_all_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
