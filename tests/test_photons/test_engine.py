import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


class TestEngineModule(chex.TestCase):
    def test_imports(self):
        """Test that the engine module can be imported."""
        import ptyrodactyl.photons.engine


if __name__ == "__main__":
    pytest.main([__file__])
