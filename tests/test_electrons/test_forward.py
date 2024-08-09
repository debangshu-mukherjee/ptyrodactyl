from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float, Int, Shaped

jax.config.update("jax_enable_x64", True)

# Import your functions here
from ptyrodactyl.electrons import (fourier_calib, transmission_func,
                                   wavelength_ang)

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)

if __name__ == "__main__":
    pytest.main([__file__])


class test_wavelength_ang(chex.TestCase):

    @chex.all_variants
    @parameterized.parameters(
        {"test_kV": 200, "expected_wavelength": 0.02508},
        {"test_kV": 1000, "expected_wavelength": 0.008719185412913083},
        {"test_kV": 0.001, "expected_wavelength": 12.2642524552},
        {"test_kV": 300, "expected_wavelength": 0.0196874863882},
    )
    def test_voltage_values(self, test_kV, expected_wavelength):
        var_wavelength_ang = self.variant(wavelength_ang)
        # voltage_kV = 200.0
        # expected_wavelength = 0.02508  # Expected value based on known physics
        result = var_wavelength_ang(test_kV)
        assert jnp.isclose(
            result, expected_wavelength, atol=1e-6
        ), f"Expected {expected_wavelength}, but got {result}"

    # Check for precision and rounding errors
    @chex.all_variants
    def test_precision_and_rounding_errors(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltage_kV = 150.0
        expected_wavelength = 0.02957  # Expected value based on known physics
        result = var_wavelength_ang(voltage_kV)
        assert jnp.isclose(
            result, expected_wavelength, atol=1e-5
        ), f"Expected {expected_wavelength}, but got {result}"

    # Ensure function returns a Float Array
    @chex.all_variants
    def test_returns_float(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltage_kV = 200.0
        result = var_wavelength_ang(voltage_kV)
        assert isinstance(
            result, Float[Array, "*"]
        ), "Expected the function to return a float"

    # Test whether array inputs work
    @chex.all_variants
    def test_array_input(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltages = jnp.array([100, 200, 300, 400], dtype=jnp.float64)
        results = var_wavelength_ang(voltages)
        expected = jnp.array([0.03701436, 0.02507934, 0.01968749, 0.01643943])
        assert jnp.allclose(results, expected, atol=1e-5)


class test_transmission_func(chex.TestCase):

    @chex.all_variants
    @parameterized.parameters(
        {"voltage_kV": 200, "shape": (64, 64)},
        {"voltage_kV": 300, "shape": (128, 128)},
    )
    def test_basic_functionality(self, voltage_kV, shape):
        var_transmission_func = self.variant(transmission_func)
        pot_slice = jnp.pi * jnp.ones(shape, dtype=jnp.float64)
        result = var_transmission_func(pot_slice, voltage_kV)

        chex.assert_shape(result, shape)
        chex.assert_type(result, jnp.complex128)
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    @chex.all_variants
    @parameterized.parameters(
        {"shape_y": 64, "shape_x": 64},
        {"shape_y": 128, "shape_x": 64},
        {"shape_y": 64, "shape_x": 128},
        {"shape_y": 128, "shape_x": 128},
    )
    def test_output_magnitude(self, shape_y, shape_x):
        pot_slice = jnp.pi * jnp.ones((shape_y, shape_x), dtype=jnp.float64)
        var_transmission_func = self.variant(transmission_func)
        result = var_transmission_func(pot_slice, 200)
        magnitude = jnp.abs(result)

        chex.assert_trees_all_close(magnitude, jnp.ones_like(magnitude), atol=1e-6)

    @chex.all_variants
    @parameterized.parameters(
        {"shape_y": 64, "shape_x": 64},
        {"shape_y": 128, "shape_x": 64},
        {"shape_y": 64, "shape_x": 128},
        {"shape_y": 128, "shape_x": 128},
    )
    def test_voltage_dependence(self, shape_y, shape_x):
        pot_slice = jnp.pi * jnp.ones((shape_y, shape_x), dtype=jnp.float64)
        var_transmission_func = self.variant(transmission_func)
        result1 = var_transmission_func(pot_slice, 100)
        result2 = var_transmission_func(pot_slice, 300)

        assert not jnp.allclose(
            result1, result2
        ), "Results should differ for different voltages"

    @chex.all_variants
    def test_potential_dependence(self):
        var_transmission_func = self.variant(transmission_func)
        pot_slice1 = jnp.ones((64, 64), dtype=jnp.float64)
        pot_slice2 = jnp.ones((64, 64), dtype=jnp.float64) * 2

        result1 = var_transmission_func(pot_slice1, 200)
        result2 = var_transmission_func(pot_slice2, 200)

        assert not jnp.allclose(
            result1, result2
        ), "Results should differ for different potentials"

    @chex.all_variants
    @parameterized.parameters(
        {"shape_y": 64, "shape_x": 64},
        {"shape_y": 128, "shape_x": 64},
        {"shape_y": 64, "shape_x": 128},
        {"shape_y": 128, "shape_x": 128},
    )
    def test_differentiable(self, shape_y, shape_x):
        pot_slice = jnp.pi * jnp.ones((shape_y, shape_x), dtype=jnp.float64)
        var_transmission_func = self.variant(transmission_func)

        def loss(voltage):
            return jnp.sum(jnp.abs(var_transmission_func(pot_slice, voltage)))

        grad_fn = jax.grad(loss)
        grad = grad_fn(200.0)

        assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"

    @chex.all_variants
    @parameterized.parameters(
        {"shape_y": 64, "shape_x": 64},
        {"shape_y": 128, "shape_x": 64},
        {"shape_y": 64, "shape_x": 128},
        {"shape_y": 128, "shape_x": 128},
    )
    def test_array_voltage_input(self, shape_y, shape_x):
        pot_slice = jnp.pi * jnp.ones((shape_y, shape_x), dtype=jnp.float64)
        var_transmission_func = self.variant(transmission_func)
        voltages = jnp.array([100, 200, 300], dtype=jnp.float64)

        results = jax.vmap(lambda v: var_transmission_func(pot_slice, v))(voltages)

        chex.assert_shape(results, (3, *pot_slice.shape))
        chex.assert_type(results, jnp.complex128)

    @chex.all_variants
    @parameterized.parameters(
        {"voltage_kV": 200, "shape": (64, 64)},
        {"voltage_kV": 300, "shape": (128, 128)},
    )
    def test_consistency_with_wavelength(self, voltage_kV, shape):
        pot_slice = jnp.pi * jnp.ones(shape, dtype=jnp.float64)
        var_transmission_func = self.variant(transmission_func)
        var_wavelength_ang = self.variant(wavelength_ang)

        voltage_kV = 200.0
        wavelength = var_wavelength_ang(voltage_kV)
        trans = var_transmission_func(pot_slice, voltage_kV)

        # Check if the wavelength is consistent with the transmission function
        # This is a simplified check and may need adjustment based on the exact relationship
        assert jnp.isclose(jnp.angle(trans).max(), 2 * jnp.pi / wavelength, rtol=1e-2)


class test_fourier_calib(chex.TestCase):
    @chex.all_variants
    @parameterized.parameters(
        {"calib_r": 0.5, "beam_size": (100, 100)},
        {"calib_r": 0.25, "beam_size": (200, 200)},
        {"calib_r": 1.5, "beam_size": (400, 400)},
        {"calib_r": 7.5, "beam_size": (1000, 1000)},
    )
    def test_basic_functionality(self, beam_size, calib_r):
        var_fourier_calib = self.variant(fourier_calib)
        result = var_fourier_calib(
            jnp.float64(calib_r), jnp.array(beam_size, dtype=float)
        )
        assert jnp.all(jnp.isfinite(result)), "Result contains non-finite values"

    @chex.all_variants
    @parameterized.parameters(
        {"calib_r": 0.5, "beam_size": (100, 100)},
        {"calib_r": 0.25, "beam_size": (200, 200)},
        {"calib_r": 1.5, "beam_size": (400, 800)},
        {"calib_r": 7.5, "beam_size": (1000, 5000)},
        {"calib_r": 7.5, "beam_size": (1000, 1000)},
    )
    def calib_1_value(self, beam_size, calib_r):
        var_fourier_calib = self.variant(fourier_calib)
        result = var_fourier_calib(
            jnp.float64(calib_r), jnp.array(beam_size, dtype=float)
        )
        if beam_size[0] == beam_size[1]:
            chex.assert_equal(result[0], result[1])
        else:
            assert not jnp.allclose(
                result[0], result[1]
            ), "Results should differ for different beam size dimensions"

    @chex.all_variants
    @parameterized.parameters(
        {"calib_r": (0.5, 0.5), "beam_size": (100, 100)},
        {"calib_r": (0.25, 0.75), "beam_size": (200, 200)},
        {"calib_r": (1.5, 1.6), "beam_size": (400, 400)},
        {"calib_r": (7.5, 7.5), "beam_size": (1000, 1000)},
    )
    def calib_2_value(self, beam_size, calib_r):
        var_fourier_calib = self.variant(fourier_calib)
        calib = jnp.array(calib_r, dtype=jnp.float64)
        result = var_fourier_calib(calib, jnp.array(beam_size))
        if calib[0] == calib[1]:
            chex.assert_equal(result[0], result[1])
        else:
            assert not jnp.allclose(
                result[0], result[1]
            ), "Results should differ for different calibrations"
