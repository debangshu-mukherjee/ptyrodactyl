import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float

from ptyrodactyl.photons.helper import (
    add_phase_screen,
    create_spatial_grid,
    field_intensity,
    normalize_field
)

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestAddPhaseScreen(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (40, 40), "offset": 0.0},
        {"shape": (60, 20), "offset": 0.1},
        {"shape": (30, 90), "offset": 45678},
    )
    def test_add_phase_screen_values(self, shape: Tuple[int, int], offset: float):
        """Check that add_phase_screen produces expected values."""
        # Create a random complex field
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field: Complex[Array, "H W"] = (field_real + (1j * field_imag)).astype(
            jnp.complex128
        )

        # Create a phase screen with a constant offset (for reproducibility)
        phase: Float[Array, "H W"] = jnp.ones(shape, dtype=jnp.float64) * offset

        # Apply the function (variant_fn will jit, vmap, or pmap if needed)
        var_add_phase_screen = self.variant(add_phase_screen)
        result = var_add_phase_screen(field, phase)

        # Compute expected result: field * exp(i * phase)
        expected = field * jnp.exp(1j * phase)

        # Check shapes
        chex.assert_shape(result, shape)

        # Check numerical correctness
        chex.assert_trees_all_close(result, expected, atol=1e-6, rtol=1e-6)

    @chex.all_variants()
    @parameterized.parameters(
        {"field_shape": (40, 40), "phase_shape": (60, 40)},
        {"field_shape": (60, 20), "phase_shape": (20, 60)},
    )
    def test_shape_mismatch(
        self, field_shape: Tuple[int, int], phase_shape: Tuple[int, int]
    ):
        """Check that shape mismatch raises an error."""
        key = jax.random.PRNGKey(123)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, field_shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, field_shape, dtype=jnp.float64)
        field: Complex[Array, "H W"] = (field_real + (1j * field_imag)).astype(
            jnp.complex128
        )

        phase: Float[Array, "H W"] = jax.random.normal(
            key, phase_shape, dtype=jnp.float64
        )

        var_add_phase_screen = self.variant(add_phase_screen)
        with pytest.raises(ValueError):
            # We expect the function (or its internal checks) to fail
            _ = var_add_phase_screen(field, phase)

    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (40, 40)},
        {"shape": (60, 20)},
        {"shape": (30, 90)},
    )
    def test_zero_phase(self, shape: Tuple[int, int]):
        """Check that a zero phase does not change the field."""
        key = jax.random.PRNGKey(999)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field: Complex[Array, "H W"] = (field_real + 1j * field_imag).astype(
            jnp.complex128
        )

        phase: Float[Array, "H W"] = jnp.zeros(shape, dtype=jnp.float64)

        var_add_phase_screen = self.variant(add_phase_screen)
        result = var_add_phase_screen(field, phase)

        # If phase is zero, result should be identical to field
        chex.assert_trees_all_close(result, field, atol=1e-6, rtol=1e-6)


class TestCreateSpatialGrid(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"diameter": 0.001, "num_points": 32},
        {"diameter": 0.01, "num_points": 64},
        {"diameter": 0.1, "num_points": 128},
    )
    def test_grid_shape(self, diameter: float, num_points: int):
        """Test that create_spatial_grid returns correct shape."""
        var_create_spatial_grid = self.variant(create_spatial_grid)
        xx, yy = var_create_spatial_grid(
            diameter=jnp.array(diameter),
            num_points=jnp.array(num_points),
        )
        
        expected_shape = (num_points, num_points)
        chex.assert_shape(xx, expected_shape)
        chex.assert_shape(yy, expected_shape)
    
    @chex.all_variants()
    def test_grid_values(self):
        """Test that the grid has correct values."""
        diameter = 0.001  # 1mm
        num_points = 32
        
        var_create_spatial_grid = self.variant(create_spatial_grid)
        xx, yy = var_create_spatial_grid(
            diameter=jnp.array(diameter),
            num_points=jnp.array(num_points),
        )
        
        # Check that the grid spans from -diameter/2 to diameter/2
        assert jnp.isclose(jnp.min(xx), -diameter/2)
        assert jnp.isclose(jnp.max(xx), diameter/2)
        assert jnp.isclose(jnp.min(yy), -diameter/2)
        assert jnp.isclose(jnp.max(yy), diameter/2)


class TestNormalizeField(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (32, 32)},
        {"shape": (64, 64)},
        {"shape": (128, 128)},
    )
    def test_normalization(self, shape: Tuple[int, int]):
        """Test that normalized field has unit power."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        var_normalize_field = self.variant(normalize_field)
        normalized = var_normalize_field(field)
        
        # Check that power is 1.0
        power = jnp.sum(jnp.abs(normalized) ** 2)
        assert jnp.isclose(power, 1.0, atol=1e-6)
        
        # Check shape
        chex.assert_shape(normalized, shape)


class TestFieldIntensity(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (32, 32)},
        {"shape": (64, 64)},
        {"shape": (128, 128)},
    )
    def test_intensity_calculation(self, shape: Tuple[int, int]):
        """Test that field_intensity returns |field|^2."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        var_field_intensity = self.variant(field_intensity)
        intensity = var_field_intensity(field)
        
        # Check that intensity is |field|^2
        expected = jnp.abs(field) ** 2
        chex.assert_trees_all_close(intensity, expected, atol=1e-6)
        
        # Check shape
        chex.assert_shape(intensity, shape)


if __name__ == "__main__":
    pytest.main([__file__])