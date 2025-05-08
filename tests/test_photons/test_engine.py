import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestEngineModule(chex.TestCase):
    def test_imports(self):
        """Test that the engine module can be imported."""
        import ptyrodactyl.photons.engine
        
        # The engine module is currently a placeholder
        # This test just verifies it can be imported without errors


if __name__ == "__main__":
    pytest.main([__file__])