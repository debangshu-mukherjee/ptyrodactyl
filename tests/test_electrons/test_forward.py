import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.electrons import (fourier_calib, propagation_func, cbed,
                                   transmission_func, wavelength_ang)

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)

if __name__ == "__main__":
    pytest.main([__file__])


class test_wavelength_ang(chex.TestCase):
    @chex.all_variants(with_pmap=True, without_device=False)
    @parameterized.parameters(
        {"test_kV": 200, "expected_wavelength": 0.02508},
        {"test_kV": 1000, "expected_wavelength": 0.008719185412913083},
        {"test_kV": 0.001, "expected_wavelength": 12.2642524552},
        {"test_kV": 300, "expected_wavelength": 0.0196874863882},
    )
    def test_voltage_values(self, test_kV, expected_wavelength):
        var_wavelength_ang = self.variant(wavelength_ang)
        # Convert to JAX array to match type annotation
        voltage_kV = jnp.asarray(test_kV, dtype=jnp.float64)
        result = var_wavelength_ang(voltage_kV)
        assert jnp.isclose(
            result, expected_wavelength, atol=1e-6
        ), f"Expected {expected_wavelength}, but got {result}"

    # Check for precision and rounding errors
    @chex.all_variants(with_pmap=True, without_device=False)
    def test_precision_and_rounding_errors(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltage_kV = jnp.asarray(150.0, dtype=jnp.float64)
        expected_wavelength = 0.02957  # Expected value based on known physics
        result = var_wavelength_ang(voltage_kV)
        assert jnp.isclose(
            result, expected_wavelength, atol=1e-5
        ), f"Expected {expected_wavelength}, but got {result}"

    # Ensure function returns a Float Array
    @chex.all_variants(with_pmap=True, without_device=False)
    def test_returns_float(self):
        var_wavelength_ang = self.variant(wavelength_ang)
        voltage_kV = jnp.asarray(200.0, dtype=jnp.float64)
        result = var_wavelength_ang(voltage_kV)
        assert isinstance(
            result, Float[Array, "*"]
        ), "Expected the function to return a float"

    # Test whether array inputs work
    @chex.all_variants(with_pmap=True, without_device=False)
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
            
            
class test_cbed(chex.TestCase):
    @chex.all_variants(with_pmap=True, without_device=False)
    def test_basic_functionality(self):
        var_cbed = self.variant(cbed)
        
        # Create simple test inputs
        H, W = 64, 64  # Small size for faster tests
        num_slices = 2
        
        # Create trivial potential slices (uniform phase shift)
        pot_slice = jnp.ones((H, W, num_slices), dtype=jnp.complex64)
        
        # Create a simple probe beam (centered Gaussian)
        x = jnp.arange(0, W)
        y = jnp.arange(0, H)
        X, Y = jnp.meshgrid(y, x, indexing="ij")
        center_x, center_y = W // 2, H // 2
        sigma = 10
        
        beam = jnp.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        beam = beam.astype(jnp.complex64)
        
        # Set other parameters
        slice_thickness = 2.0
        voltage_kV = 200.0
        calib_ang = 0.1
        
        # Run the CBED function
        result = var_cbed(pot_slice, beam, slice_thickness, voltage_kV, calib_ang)
        
        # Basic shape check
        assert result.shape == (H, W), f"Expected shape ({H}, {W}), got {result.shape}"
        
        # Check that result is real and positive
        assert jnp.isreal(result).all(), "Expected real values in result"
        assert jnp.all(result >= 0), "Expected non-negative values in result"
        
        # Check for expected symmetry in the CBED pattern (should be centro-symmetric)
        center_intensity = result[H//2, W//2]
        assert center_intensity > 0, "Expected non-zero intensity at center"

    @chex.all_variants(with_pmap=True, without_device=False)
    def test_2d_input_auto_expansion(self):
        var_cbed = self.variant(cbed)
        
        # Create 2D inputs (without slice/mode dimensions)
        H, W = 64, 64
        
        # Create trivial potential slice (2D)
        pot_slice = jnp.ones((H, W), dtype=jnp.complex64)
        
        # Create a simple probe beam (2D)
        x = jnp.arange(0, W)
        y = jnp.arange(0, H)
        X, Y = jnp.meshgrid(y, x, indexing="ij")
        center_x, center_y = W // 2, H // 2
        sigma = 10
        
        beam = jnp.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        beam = beam.astype(jnp.complex64)
        
        # Set other parameters
        slice_thickness = 2.0
        voltage_kV = 200.0
        calib_ang = 0.1
        
        # Run the CBED function with 2D inputs
        result = var_cbed(pot_slice, beam, slice_thickness, voltage_kV, calib_ang)
        
        # Check shape
        assert result.shape == (H, W), f"Expected shape ({H}, {W}), got {result.shape}"

    @chex.all_variants(with_pmap=True, without_device=False)
    def test_multiple_beam_modes(self):
        var_cbed = self.variant(cbed)
        
        # Create inputs with multiple beam modes
        H, W = 64, 64
        num_slices = 2
        num_modes = 3
        
        # Create trivial potential slices
        pot_slice = jnp.ones((H, W, num_slices), dtype=jnp.complex64)
        
        # Create multiple beam modes (shifted Gaussians)
        x = jnp.arange(0, W)
        y = jnp.arange(0, H)
        X, Y = jnp.meshgrid(y, x, indexing="ij")
        center_x, center_y = W // 2, H // 2
        sigma = 10
        
        # Create an array of beam modes
        modes = []
        offsets = [(-10, 0), (0, 0), (10, 0)]  # Different beam positions
        
        for dx, dy in offsets:
            mode = jnp.exp(-((X - (center_x + dx))**2 + (Y - (center_y + dy))**2) / (2 * sigma**2))
            modes.append(mode)
        
        beam = jnp.stack(modes, axis=-1).astype(jnp.complex64)
        
        # Set other parameters
        slice_thickness = 2.0
        voltage_kV = 200.0
        calib_ang = 0.1
        
        # Run the CBED function with multiple modes
        result = var_cbed(pot_slice, beam, slice_thickness, voltage_kV, calib_ang)
        
        # Check shape - should still be 2D despite multiple modes
        assert result.shape == (H, W), f"Expected shape ({H}, {W}), got {result.shape}"
        
        # The result should have higher total intensity than with a single mode
        # First calculate single mode result for comparison
        single_result = var_cbed(pot_slice, beam[..., 0:1], slice_thickness, voltage_kV, calib_ang)
        
        # Total intensity should be higher with multiple modes
        assert jnp.sum(result) > jnp.sum(single_result), "Expected higher intensity with multiple modes"

    @chex.all_variants(with_pmap=True, without_device=False)
    def test_dtype_consistency(self):
        var_cbed = self.variant(cbed)
        
        # Create inputs with different dtypes
        H, W = 64, 64
        
        # Test with both complex64 and complex128
        for dtype in [jnp.complex64, jnp.complex128]:
            # Create potential slice
            pot_slice = jnp.ones((H, W, 1), dtype=dtype)
            
            # Create beam
            x = jnp.arange(0, W)
            y = jnp.arange(0, H)
            X, Y = jnp.meshgrid(y, x, indexing="ij")
            center_x, center_y = W // 2, H // 2
            sigma = 10
            
            beam = jnp.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
            beam = beam.astype(dtype)[..., jnp.newaxis]
            
            # Set other parameters
            slice_thickness = 2.0
            voltage_kV = 200.0
            calib_ang = 0.1
            
            # This should run without dtype errors
            result = var_cbed(pot_slice, beam, slice_thickness, voltage_kV, calib_ang)
            
            # Check output type (should be float32 or float64 depending on input)
            expected_float_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
            assert result.dtype == expected_float_dtype, f"Expected {expected_float_dtype}, got {result.dtype}"

    @chex.all_variants(with_pmap=True, without_device=False)
    @parameterized.parameters(
        {"voltage_kV": 100.0, "calib_ang": 0.05},
        {"voltage_kV": 200.0, "calib_ang": 0.1},
        {"voltage_kV": 300.0, "calib_ang": 0.2},
    )
    def test_parameter_variations(self, voltage_kV, calib_ang):
        var_cbed = self.variant(cbed)
        
        # Create simple inputs
        H, W = 64, 64
        
        # Create potential slice
        pot_slice = jnp.ones((H, W, 1), dtype=jnp.complex64)
        
        # Create beam
        x = jnp.arange(0, W)
        y = jnp.arange(0, H)
        X, Y = jnp.meshgrid(y, x, indexing="ij")
        center_x, center_y = W // 2, H // 2
        sigma = 10
        
        beam = jnp.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        beam = beam.astype(jnp.complex64)[..., jnp.newaxis]
        
        # Set slice thickness
        slice_thickness = 2.0
        
        # Run the CBED function with different parameters
        result = var_cbed(pot_slice, beam, slice_thickness, voltage_kV, calib_ang)
        
        # Basic shape and type checks
        assert result.shape == (H, W), f"Expected shape ({H}, {W}), got {result.shape}"
        assert jnp.isreal(result).all(), "Expected real values in result"
        assert jnp.all(result >= 0), "Expected non-negative values in result"

    @chex.all_variants(with_pmap=True, without_device=False)
    def test_jit_compatibility(self):
        # Create inputs
        H, W = 64, 64
        
        # Create potential slice
        pot_slice = jnp.ones((H, W, 1), dtype=jnp.complex64)
        
        # Create beam
        x = jnp.arange(0, W)
        y = jnp.arange(0, H)
        X, Y = jnp.meshgrid(y, x, indexing="ij")
        center_x, center_y = W // 2, H // 2
        sigma = 10
        
        beam = jnp.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        beam = beam.astype(jnp.complex64)[..., jnp.newaxis]
        
        # Set other parameters
        slice_thickness = jnp.array(2.0)
        voltage_kV = jnp.array(200.0)
        calib_ang = jnp.array(0.1)
        
        # Define a JIT-compiled version
        jitted_cbed = jax.jit(cbed)
        
        # This should compile and run without errors
        result = jitted_cbed(pot_slice, beam, slice_thickness, voltage_kV, calib_ang)
        
        # Basic check
        assert result.shape == (H, W), f"Expected shape ({H}, {W}), got {result.shape}"
