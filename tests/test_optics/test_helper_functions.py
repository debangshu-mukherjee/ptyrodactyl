from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from ptyrodactyl.optics import add_phase_screen

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    pytest.main([__file__])

class test_add_phase_screen(chex.TestCase):
    @chex.all_variants()
    @parameterized.named_parameters(
        ("no_offset", (40, 40), 0.0),
        ("small_offset", (60, 20), 0.1),
        ("large_offset", (30, 90), 45678),
    )
    def test_add_phase_screen_values(self, shape, offset):
        """Check that add_phase_screen produces expected values."""
        # Create a random complex field
        key = jax.random.PRNGKey(42)
        field_real = jax.random.normal(key, shape)
        field_imag = jax.random.normal(key, shape)
        field = field_real + (1j * field_imag)

        # Create a phase screen with a constant offset (for reproducibility)
        phase = jnp.ones(shape) * offset

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
    @parameterized.named_parameters(
        ("field_3x4_phase_4x4", (3, 4), (4, 4)),
        ("field_4x5_phase_4x4", (4, 5), (4, 4)),
    )
    def test_shape_mismatch(self, field_shape, phase_shape):
        """Check that shape mismatch raises an error."""
        key = jax.random.PRNGKey(123)
        field_real = jax.random.normal(key, field_shape)
        field_imag = jax.random.normal(key, field_shape)
        field = field_real + (1j * field_imag)

        phase = jax.random.normal(key, phase_shape)

        var_add_phase_screen = self.variant(add_phase_screen)
        with pytest.raises(ValueError):
            # We expect the function (or its internal checks) to fail
            _ = var_add_phase_screen(field, phase)

    @chex.all_variants()
    @parameterized.parameters(
        ((20, 20)),
        ((30, 30)),
        ((40, 60)),
    )
    def test_zero_phase(self, shape):
        """Check that a zero phase does not change the field."""
        key = jax.random.PRNGKey(999)
        field_real = jax.random.normal(key, shape)
        field_imag = jax.random.normal(key, shape)
        field = field_real + 1j * field_imag

        phase = jnp.zeros(shape)

        var_add_phase_screen = self.variant(add_phase_screen)
        result = var_add_phase_screen(field, phase)

        # If phase is zero, result should be identical to field
        chex.assert_trees_all_close(result, field, atol=1e-6, rtol=1e-6)
