import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float

from ptyrodactyl.photons.optics import (
    angular_spectrum_prop,
    fresnel_prop,
    fraunhofer_prop,
    circular_aperture,
    zoom_wavefront,
)
from ptyrodactyl.photons.photon_types import OpticalWavefront

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestAngularSpectrumProp(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (64, 64), "wavelength": 500e-9, "z_move": 0.01},
        {"shape": (128, 128), "wavelength": 600e-9, "z_move": 0.05},
    )
    def test_propagation_shape(self, shape: Tuple[int, int], wavelength: float, z_move: float):
        """Test that the propagated field has the correct shape."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        dx = 5e-6  # 5 microns
        
        incoming = OpticalWavefront(
            field=field,
            wavelength=jnp.array(wavelength, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )
        
        var_angular_spectrum_prop = self.variant(angular_spectrum_prop)
        propagated = var_angular_spectrum_prop(
            incoming=incoming, 
            z_move=jnp.array(z_move, dtype=jnp.float64)
        )
        
        # Check shapes
        chex.assert_shape(propagated.field, shape)
        
        # Check z_position updated
        chex.assert_trees_all_close(
            propagated.z_position, 
            jnp.array(z_move, dtype=jnp.float64),
            atol=1e-10
        )

    @chex.all_variants()
    def test_zero_propagation(self):
        """Test that propagating by zero distance returns the original field."""
        shape = (64, 64)
        wavelength = 500e-9
        dx = 5e-6
        
        key = jax.random.PRNGKey(123)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        incoming = OpticalWavefront(
            field=field,
            wavelength=jnp.array(wavelength, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )
        
        var_angular_spectrum_prop = self.variant(angular_spectrum_prop)
        propagated = var_angular_spectrum_prop(
            incoming=incoming, 
            z_move=jnp.array(0.0, dtype=jnp.float64)
        )
        
        # Field should remain almost identical
        chex.assert_trees_all_close(propagated.field, incoming.field, atol=1e-6)


class TestFresnelProp(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (64, 64), "wavelength": 500e-9, "z_move": 0.01},
        {"shape": (128, 128), "wavelength": 600e-9, "z_move": 0.05},
    )
    def test_propagation_shape(self, shape: Tuple[int, int], wavelength: float, z_move: float):
        """Test that the propagated field has the correct shape."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        dx = 5e-6  # 5 microns
        
        incoming = OpticalWavefront(
            field=field,
            wavelength=jnp.array(wavelength, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )
        
        var_fresnel_prop = self.variant(fresnel_prop)
        propagated = var_fresnel_prop(
            incoming=incoming, 
            z_move=jnp.array(z_move, dtype=jnp.float64)
        )
        
        # Check shapes
        chex.assert_shape(propagated.field, shape)
        
        # Check z_position updated
        chex.assert_trees_all_close(
            propagated.z_position, 
            jnp.array(z_move, dtype=jnp.float64),
            atol=1e-10
        )


class TestFraunhoferProp(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (64, 64), "wavelength": 500e-9, "z_move": 0.1},
        {"shape": (128, 128), "wavelength": 600e-9, "z_move": 0.2},
    )
    def test_propagation_shape(self, shape: Tuple[int, int], wavelength: float, z_move: float):
        """Test that the propagated field has the correct shape."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        dx = 5e-6  # 5 microns
        
        incoming = OpticalWavefront(
            field=field,
            wavelength=jnp.array(wavelength, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )
        
        var_fraunhofer_prop = self.variant(fraunhofer_prop)
        propagated = var_fraunhofer_prop(
            incoming=incoming, 
            z_move=jnp.array(z_move, dtype=jnp.float64)
        )
        
        # Check shapes
        chex.assert_shape(propagated.field, shape)
        
        # Check z_position updated
        chex.assert_trees_all_close(
            propagated.z_position, 
            jnp.array(z_move, dtype=jnp.float64),
            atol=1e-10
        )


class TestCircularAperture(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (64, 64), "diameter": 20e-6, "center": None},
        {"shape": (128, 128), "diameter": 40e-6, "center": jnp.array([10e-6, -5e-6])},
    )
    def test_aperture_shape(self, shape: Tuple[int, int], diameter: float, center):
        """Test that the aperture application preserves shape."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        dx = 1e-6  # 1 micron
        
        incoming = OpticalWavefront(
            field=field,
            wavelength=jnp.array(500e-9, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )
        
        var_circular_aperture = self.variant(circular_aperture)
        apertured = var_circular_aperture(
            incoming=incoming, 
            diameter=jnp.array(diameter, dtype=jnp.float64),
            center=center
        )
        
        # Check shapes
        chex.assert_shape(apertured.field, shape)
        
        # Check that z_position is preserved
        chex.assert_trees_all_close(
            apertured.z_position, 
            incoming.z_position,
            atol=1e-10
        )


class TestZoomWavefront(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (64, 64), "zoom_factor": 2.0},
        {"shape": (128, 128), "zoom_factor": 0.5},
    )
    def test_zoom_shape(self, shape: Tuple[int, int], zoom_factor: float):
        """Test that zooming preserves the wavefront shape."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        field_real = jax.random.normal(key1, shape, dtype=jnp.float64)
        field_imag = jax.random.normal(key2, shape, dtype=jnp.float64)
        field = field_real + 1j * field_imag
        
        dx = 1e-6  # 1 micron
        
        wavefront = OpticalWavefront(
            field=field,
            wavelength=jnp.array(500e-9, dtype=jnp.float64),
            dx=jnp.array(dx, dtype=jnp.float64),
            z_position=jnp.array(0.0, dtype=jnp.float64),
        )
        
        var_zoom_wavefront = self.variant(zoom_wavefront)
        zoomed = var_zoom_wavefront(
            wavefront=wavefront, 
            zoom_factor=jnp.array(zoom_factor, dtype=jnp.float64)
        )
        
        # Check shapes
        chex.assert_shape(zoomed.field, shape)
        
        # Check that dx is updated
        expected_dx = dx / zoom_factor
        chex.assert_trees_all_close(
            zoomed.dx, 
            jnp.array(expected_dx, dtype=jnp.float64),
            atol=1e-10
        )


if __name__ == "__main__":
    pytest.main([__file__])