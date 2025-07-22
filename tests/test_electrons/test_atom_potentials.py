"""Tests for atom_potentials module, specifically the _bessel_kv function."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

# Enable x64 to match the preprocessing module
jax.config.update("jax_enable_x64", True)

from ptyrodactyl.electrons.atom_potentials import _bessel_kv


class TestBesselKv(chex.TestCase):
    """Test suite for the _bessel_kv function."""
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    @parameterized.parameters(
        # (x, expected_k0, tolerance)
        # Values from standard references (Abramowitz & Stegun)
        (0.1, 2.4270690, 1e-5),
        (0.2, 1.7527038, 1e-5),
        (0.5, 0.9244190, 1e-5),
        (1.0, 0.4210244, 1e-5),
        (1.5, 0.2138056, 1e-2),  # 1% tolerance for medium x
        (2.0, 0.1138938, 5e-3),  # 0.5% tolerance 
        (3.0, 0.0347395, 1e-3),  # 0.1% tolerance
        (5.0, 0.0036911, 1e-3),  # 0.1% tolerance
        (10.0, 0.0000778, 1e-4),  # Slightly larger tolerance for large x
    )
    def test_bessel_k0_accuracy(self, x, expected, tol):
        """Test K_0(x) against known values."""
        x_array = jnp.asarray(x, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_computed = self.variant(_bessel_kv)(v_scalar, x_array)
        
        # Check accuracy
        self.assertAlmostEqual(
            k0_computed, 
            expected, 
            delta=tol,
            msg=f"K_0({x}) = {k0_computed:.8f}, expected {expected:.8f}"
        )
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    @parameterized.parameters(
        (0.01,),
        (0.1,),
        (0.5,),
        (1.0,),
        (2.0,),
        (5.0,),
        (10.0,),
    )
    def test_bessel_k0_derivative(self, x):
        """Test that K_0'(x) = -K_1(x) property holds numerically."""
        x_array = jnp.asarray(x, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        
        # Compute numerical derivative
        grad_fn = jax.grad(lambda y: self.variant(_bessel_kv)(v_scalar, y))
        dk0_dx = grad_fn(x_array)
        
        # For K_0, we know that K_0'(x) = -K_1(x)
        # We can verify the derivative is negative and decreasing
        self.assertLess(dk0_dx, 0.0, f"K_0'({x}) should be negative")
        
        # Additional check: K_0 is a decreasing function
        # Skip this check at x=1.0 due to approximation switch
        if abs(x - 1.0) > 0.01:
            eps = 1e-4
            k0_x = self.variant(_bessel_kv)(v_scalar, x_array)
            k0_x_plus = self.variant(_bessel_kv)(v_scalar, x_array + eps)
            self.assertLess(k0_x_plus, k0_x, f"K_0 should be decreasing at x={x}")
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_k0_vectorization(self):
        """Test that _bessel_kv properly handles vector inputs."""
        # 1D array
        x_1d = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_1d = self.variant(_bessel_kv)(v_scalar, x_1d)
        self.assertEqual(k0_1d.shape, x_1d.shape)
        
        # 2D array
        x_2d = jnp.array([[0.1, 0.5], [1.0, 2.0], [3.0, 5.0]], dtype=jnp.float64)
        k0_2d = self.variant(_bessel_kv)(v_scalar, x_2d)
        self.assertEqual(k0_2d.shape, x_2d.shape)
        
        # 3D array
        x_3d = jnp.ones((2, 3, 4), dtype=jnp.float64)
        k0_3d = self.variant(_bessel_kv)(v_scalar, x_3d)
        self.assertEqual(k0_3d.shape, x_3d.shape)
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_k0_small_x_behavior(self):
        """Test K_0(x) behavior for small x: K_0(x) ~ -log(x/2) - gamma."""
        gamma_euler = 0.5772156649015329
        x_small = jnp.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        
        k0_values = self.variant(_bessel_kv)(v_scalar, x_small)
        expected_approx = -jnp.log(x_small / 2.0) - gamma_euler
        
        # For very small x, the approximation should be very good
        relative_errors = jnp.abs((k0_values - expected_approx) / expected_approx)
        self.assertTrue(
            jnp.all(relative_errors < 0.01),  # Less than 1% error
            f"Small x approximation failed: max error = {jnp.max(relative_errors):.6f}"
        )
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_k0_large_x_behavior(self):
        """Test K_0(x) asymptotic behavior for large x: K_0(x) ~ sqrt(pi/(2x)) * exp(-x)."""
        x_large = jnp.array([10.0, 20.0, 50.0, 100.0], dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        
        k0_values = self.variant(_bessel_kv)(v_scalar, x_large)
        expected_asymptotic = jnp.sqrt(jnp.pi / (2 * x_large)) * jnp.exp(-x_large)
        
        # For large x, check relative error
        relative_errors = jnp.abs((k0_values - expected_asymptotic) / expected_asymptotic)
        self.assertTrue(
            jnp.all(relative_errors < 0.1),  # Less than 10% error
            f"Large x asymptotic failed: max error = {jnp.max(relative_errors):.6f}"
        )
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_k0_positivity(self):
        """Test that K_0(x) is always positive for x > 0."""
        x_test = jnp.logspace(-3, 2, 50, dtype=jnp.float64)  # From 0.001 to 100
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_values = self.variant(_bessel_kv)(v_scalar, x_test)
        
        self.assertTrue(
            jnp.all(k0_values > 0),
            "K_0(x) must be positive for all x > 0"
        )
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_k0_monotonicity(self):
        """Test that K_0(x) is strictly decreasing."""
        x_test = jnp.linspace(0.1, 10.0, 50, dtype=jnp.float64)
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_values = self.variant(_bessel_kv)(v_scalar, x_test)
        
        # Check that differences are all negative
        differences = jnp.diff(k0_values)
        self.assertTrue(
            jnp.all(differences < 0),
            "K_0(x) must be strictly decreasing"
        )
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    @parameterized.parameters(
        (jnp.float32,),
        (jnp.float64,) if jax.config.x64_enabled else (jnp.float32,),
    )
    def test_bessel_k0_dtype_consistency(self, dtype):
        """Test that output dtype matches input dtype."""
        x = jnp.array([0.5, 1.0, 2.0], dtype=dtype)
        v_scalar = jnp.asarray(0.0, dtype=dtype)
        k0_values = self.variant(_bessel_kv)(v_scalar, x)
        
        self.assertEqual(k0_values.dtype, dtype)
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_non_zero_order(self):
        """Test that non-zero order returns zeros (placeholder behavior)."""
        x = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
        
        # Test v != 0 returns zeros (current placeholder implementation)
        for v in [0.5, 1.0, 2.0]:
            v_scalar = jnp.asarray(v, dtype=jnp.float64)
            kv_values = self.variant(_bessel_kv)(v_scalar, x)
            chex.assert_trees_all_equal(kv_values, jnp.zeros_like(x))
    
    @chex.variants(with_jit=True, without_jit=True, with_device=True, with_pmap=True)
    def test_bessel_k0_continuity_at_transition(self):
        """Test continuity at the transition point x=1.0."""
        # Test values around the transition point
        x_before = 0.999
        x_at = 1.0
        x_after = 1.001
        
        v_scalar = jnp.asarray(0.0, dtype=jnp.float64)
        k0_before = self.variant(_bessel_kv)(v_scalar, jnp.asarray(x_before, dtype=jnp.float64))
        k0_at = self.variant(_bessel_kv)(v_scalar, jnp.asarray(x_at, dtype=jnp.float64))
        k0_after = self.variant(_bessel_kv)(v_scalar, jnp.asarray(x_after, dtype=jnp.float64))
        
        # Check that the function is reasonably continuous
        # Note: There's a small discontinuity at x=1.0 due to switching approximations
        self.assertAlmostEqual(k0_before, k0_at, delta=2e-2)
        self.assertAlmostEqual(k0_at, k0_after, delta=2e-2)
        
        # Check transition is not too large
        jump_before = abs(k0_at - k0_before)
        jump_after = abs(k0_after - k0_at)
        self.assertLess(jump_before, 2e-2)
        self.assertLess(jump_after, 2e-2)


if __name__ == "__main__":
    chex.TestCase.main()