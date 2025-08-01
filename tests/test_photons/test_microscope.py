import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float

from ptyrodactyl.photons.lenses import double_convex_lens
from ptyrodactyl.photons.microscope import lens_propagation
from ptyrodactyl.photons.photon_types import (LensParams, OpticalWavefront,
                                              make_optical_wavefront)

jax.config.update("jax_enable_x64", True)


class TestLensPropagation(chex.TestCase):
    @chex.all_variants(without_device=False)
    @parameterized.parameters(
        {"shape": (64, 64), "wavelength": 500e-9, "focal_length": 0.05},
        {"shape": (128, 128), "wavelength": 600e-9, "focal_length": 0.1},
    )
    def test_propagation_shape(
        self, shape: Tuple[int, int], wavelength: float, focal_length: float
    ):
        """Test that the lens propagation preserves shape."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag

        dx = 5e-6

        incoming = make_optical_wavefront(
            field=field,
            wavelength=jnp.array(wavelength, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )

        lens = double_convex_lens(
            focal_length=jnp.array(focal_length, dtype=jnp.float64),
            diameter=jnp.array(0.01, dtype=jnp.float64),
            n=jnp.array(1.5, dtype=jnp.float64),
            center_thickness=jnp.array(0.002, dtype=jnp.float64),
        )

        var_lens_propagation = self.variant(lens_propagation)
        propagated = var_lens_propagation(incoming=incoming, lens=lens)

        chex.assert_shape(propagated.field, shape)

        chex.assert_trees_all_close(
            propagated.z_position, incoming.z_position, atol=1e-10
        )

    @chex.all_variants(without_device=False)
    def test_phase_modulation(self):
        """Test that the lens adds a phase modulation."""
        shape = (64, 64)
        wavelength = 500e-9
        focal_length = 0.05
        dx = 5e-6

        field = jnp.ones(shape, dtype=jnp.complex128)

        incoming = make_optical_wavefront(
            field=field,
            wavelength=jnp.array(wavelength, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )

        lens = double_convex_lens(
            focal_length=jnp.array(focal_length, dtype=jnp.float64),
            diameter=jnp.array(0.01, dtype=jnp.float64),
            n=jnp.array(1.5, dtype=jnp.float64),
            center_thickness=jnp.array(0.002, dtype=jnp.float64),
        )

        var_lens_propagation = self.variant(lens_propagation)
        propagated = var_lens_propagation(incoming=incoming, lens=lens)

        center_idx = (shape[0] // 2, shape[1] // 2)
        assert jnp.abs(propagated.field[center_idx]) > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
