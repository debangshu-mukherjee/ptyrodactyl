"""Tests for helper_functions.py."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import ptyrodactyl.optics as pto

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    pytest.main([__file__])


class TestCreateSpatialGrid(chex.TestCase):
    """Test cases for create_spatial_grid function."""

    @chex.all_variants
    @parameterized.parameters(
        {"diameter": 1.0, "num_points": 32},
        {"diameter": 0.1, "num_points": 64},
        {"diameter": 0.01, "num_points": 128},
    )
    def test_grid_shape_and_values(self, diameter: float, num_points: int):
        """Test grid creation with different parameters."""
        # Create the grid
        X, Y = self.variant(pto.create_spatial_grid)(
            jnp.array(diameter), jnp.array(num_points)
        )

        # Check shapes
        chex.assert_shape(X, (num_points, num_points))
        chex.assert_shape(Y, (num_points, num_points))

        # Check grid ranges
        chex.assert_trees_all_close(X[0, -1], jnp.array(diameter / 2))
        chex.assert_trees_all_close(X[0, 0], jnp.array(-diameter / 2))
        chex.assert_trees_all_close(Y[-1, 0], jnp.array(diameter / 2))
        chex.assert_trees_all_close(Y[0, 0], jnp.array(-diameter / 2))


class TestAngularSpectrumProp(chex.TestCase):
    """Test cases for angular_spectrum_prop function."""

    @chex.all_variants
    @parameterized.parameters(
        {
            "shape": (64, 64),
            "z": 0.1,
            "dx": 10e-6,
            "wavelength": 632.8e-9,
        },
        {
            "shape": (128, 128),
            "z": 0.05,
            "dx": 5e-6,
            "wavelength": 532e-9,
        },
    )
    def test_propagation_conservation(
        self, shape: Tuple[int, int], z: float, dx: float, wavelength: float
    ):
        """Test energy conservation during propagation."""
        # Create a Gaussian beam
        x = jnp.linspace(-shape[0] // 2, shape[0] // 2, shape[0]) * dx
        y = jnp.linspace(-shape[1] // 2, shape[1] // 2, shape[1]) * dx
        X, Y = jnp.meshgrid(x, y)
        w0 = 20 * dx
        field = jnp.exp(-(X**2 + Y**2) / w0**2)

        # Propagate the field
        propagated = self.variant(pto.angular_spectrum_prop)(
            field, jnp.array(z), jnp.array(dx), jnp.array(wavelength)
        )

        # Check energy conservation
        initial_power = jnp.sum(jnp.abs(field) ** 2)
        final_power = jnp.sum(jnp.abs(propagated) ** 2)
        chex.assert_trees_all_close(initial_power, final_power, rtol=1e-10)


class TestFresnelProp(chex.TestCase):
    """Test cases for fresnel_prop function."""

    @chex.all_variants
    @parameterized.parameters(
        {
            "shape": (64, 64),
            "z": 0.1,
            "dx": 10e-6,
            "wavelength": 632.8e-9,
        },
        {
            "shape": (128, 128),
            "z": 0.05,
            "dx": 5e-6,
            "wavelength": 532e-9,
        },
    )
    def test_fresnel_propagation(
        self, shape: Tuple[int, int], z: float, dx: float, wavelength: float
    ):
        """Test Fresnel propagation properties."""
        # Create a Gaussian beam
        x = jnp.linspace(-shape[0] // 2, shape[0] // 2, shape[0]) * dx
        y = jnp.linspace(-shape[1] // 2, shape[1] // 2, shape[1]) * dx
        X, Y = jnp.meshgrid(x, y)
        w0 = 20 * dx
        field = jnp.exp(-(X**2 + Y**2) / w0**2)

        # Propagate forward and backward
        forward = self.variant(pto.fresnel_prop)(
            field, jnp.array(z), jnp.array(dx), jnp.array(wavelength)
        )
        backward = self.variant(pto.fresnel_prop)(
            forward, jnp.array(-z), jnp.array(dx), jnp.array(wavelength)
        )

        # Check reversibility
        chex.assert_trees_all_close(field, backward, rtol=1e-10)


class TestFraunhoferProp(chex.TestCase):
    """Test cases for fraunhofer_prop function."""

    @chex.all_variants
    @parameterized.parameters(
        {
            "shape": (64, 64),
            "z": 1.0,
            "dx": 10e-6,
            "wavelength": 632.8e-9,
        },
        {
            "shape": (128, 128),
            "z": 0.5,
            "dx": 5e-6,
            "wavelength": 532e-9,
        },
    )
    def test_fraunhofer_propagation(
        self, shape: Tuple[int, int], z: float, dx: float, wavelength: float
    ):
        """Test Fraunhofer propagation properties."""
        # Create a rectangular aperture
        field = jnp.ones(shape)
        field = field * (jnp.abs(jnp.arange(-shape[0]//2, shape[0]//2)[:, None]) < shape[0]//4)
        field = field * (jnp.abs(jnp.arange(-shape[1]//2, shape[1]//2)[None, :]) < shape[1]//4)

        # Propagate the field
        propagated = self.variant(pto.fraunhofer_prop)(
            field, jnp.array(z), jnp.array(dx), jnp.array(wavelength)
        )

        # Check the propagated field is symmetric
        chex.assert_trees_all_close(
            jnp.abs(propagated), jnp.abs(jnp.flip(propagated)), rtol=1e-10
        )


class TestFieldIntensity(chex.TestCase):
    """Test cases for field_intensity function."""

    @chex.all_variants
    @parameterized.parameters(
        {"shape": (32, 32)},
        {"shape": (64, 64)},
    )
    def test_intensity_calculation(self, shape: Tuple[int, int]):
        """Test intensity calculation properties."""
        # Create a complex field
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, shape)
        imag = jax.random.normal(key2, shape)
        field = real + 1j * imag

        # Calculate intensity
        intensity = self.variant(pto.field_intensity)(field)

        # Check properties
        chex.assert_shape(intensity, shape)
        chex.assert_trees_all_greater(intensity, jnp.zeros_like(intensity))
        chex.assert_trees_all_close(intensity, jnp.abs(field) ** 2)


class TestNormalizeField(chex.TestCase):
    """Test cases for normalize_field function."""

    @chex.all_variants
    @parameterized.parameters(
        {"shape": (32, 32)},
        {"shape": (64, 64)},
    )
    def test_normalization(self, shape: Tuple[int, int]):
        """Test field normalization properties."""
        # Create a random complex field
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, shape)
        imag = jax.random.normal(key2, shape)
        field = real + 1j * imag

        # Normalize the field
        normalized = self.variant(pto.normalize_field)(field)

        # Check total power is 1
        total_power = jnp.sum(jnp.abs(normalized) ** 2)
        chex.assert_trees_all_close(total_power, jnp.array(1.0), rtol=1e-10)


class TestAddPhaseScreen(chex.TestCase):
    """Test cases for add_phase_screen function."""

    @chex.all_variants
    @parameterized.parameters(
        {"shape": (32, 32)},
        {"shape": (64, 64)},
    )
    def test_phase_screen_application(self, shape: Tuple[int, int]):
        """Test phase screen application properties."""
        # Create a field and phase screen
        key = jax.random.PRNGKey(0)
        field = jnp.ones(shape)
        phase = jax.random.uniform(key, shape, minval=-jnp.pi, maxval=jnp.pi)

        # Apply phase screen
        result = self.variant(pto.add_phase_screen)(field, phase)

        # Check amplitude preservation
        chex.assert_trees_all_close(jnp.abs(result), jnp.abs(field), rtol=1e-10)

        # Check phase application
        chex.assert_trees_all_close(jnp.angle(result), phase, rtol=1e-10)