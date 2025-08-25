import chex
import jax
import pytest

jax.config.update("jax_enable_x64", True)


class TestEngineModule(chex.TestCase):
    def test_imports(self) -> None:
        """Test that the engine module can be imported."""


if __name__ == "__main__":
    pytest.main([__file__])
