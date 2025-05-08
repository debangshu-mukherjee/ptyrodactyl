import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import Array, Complex, Float, Integer

from ptyrodactyl.photons.photon_types import (
    LensParams, 
    GridParams, 
    OpticalWavefront,
    make_lens_params,
    make_grid_params,
    make_optical_wavefront
)

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestMakeLensParams(chex.TestCase):
    """Test the make_lens_params factory function."""
    
    def test_make_lens_params_with_valid_types(self):
        """Test that make_lens_params works with valid types."""
        # Create a valid LensParams instance
        lens_params = make_lens_params(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )
        
        # Check that the returned value is a LensParams instance
        assert isinstance(lens_params, LensParams)
        
        # Check that all attributes are set correctly
        assert lens_params.focal_length == jnp.array(0.1)
        assert lens_params.diameter == jnp.array(0.05)
        assert lens_params.n == jnp.array(1.5)
        assert lens_params.center_thickness == jnp.array(0.01)
        assert lens_params.R1 == jnp.array(0.2)
        assert lens_params.R2 == jnp.array(0.2)
    
    def test_make_lens_params_with_invalid_types(self):
        """Test that make_lens_params raises an error with invalid types."""
        # Try with non-float/non-array type for focal_length
        with pytest.raises(Exception):
            make_lens_params(
                focal_length=1,  # should be jnp.array(1.0)
                diameter=jnp.array(0.05),
                n=jnp.array(1.5),
                center_thickness=jnp.array(0.01),
                R1=jnp.array(0.2),
                R2=jnp.array(0.2),
            )
            
        # Try with non-float/non-array type for diameter
        with pytest.raises(Exception):
            make_lens_params(
                focal_length=jnp.array(0.1),
                diameter=0.05,  # should be jnp.array(0.05)
                n=jnp.array(1.5),
                center_thickness=jnp.array(0.01),
                R1=jnp.array(0.2),
                R2=jnp.array(0.2),
            )


class TestMakeGridParams(chex.TestCase):
    """Test the make_grid_params factory function."""
    
    def test_make_grid_params_with_valid_types(self):
        """Test that make_grid_params works with valid types."""
        # Set up test values
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1], dtype=jnp.float64), jnp.arange(shape[0], dtype=jnp.float64))
        phase_profile = jnp.zeros(shape)
        transmission = jnp.ones(shape)
        
        # Create a valid GridParams instance
        grid_params = make_grid_params(
            X=X,
            Y=Y,
            phase_profile=phase_profile,
            transmission=transmission
        )
        
        # Check that the returned value is a GridParams instance
        assert isinstance(grid_params, GridParams)
        
        # Check that all attributes are set correctly
        chex.assert_trees_all_close(grid_params.X, X)
        chex.assert_trees_all_close(grid_params.Y, Y)
        chex.assert_trees_all_close(grid_params.phase_profile, phase_profile)
        chex.assert_trees_all_close(grid_params.transmission, transmission)
        
        # Check shapes of all attributes
        chex.assert_shape(grid_params.X, shape)
        chex.assert_shape(grid_params.Y, shape)
        chex.assert_shape(grid_params.phase_profile, shape)
        chex.assert_shape(grid_params.transmission, shape)
    
    def test_make_grid_params_with_invalid_types(self):
        """Test that make_grid_params raises an error with invalid types."""
        # Set up test values
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[0]))
        phase_profile = jnp.zeros(shape)
        transmission = jnp.ones(shape)
        
        # Try with wrong array shape for X
        with pytest.raises(Exception):
            make_grid_params(
                X=jnp.arange(10),  # Should be 2D array
                Y=Y,
                phase_profile=phase_profile,
                transmission=transmission
            )
        
        # Try with wrong array shape for phase_profile
        with pytest.raises(Exception):
            make_grid_params(
                X=X,
                Y=Y,
                phase_profile=jnp.zeros((16, 16)),  # Shape mismatch
                transmission=transmission
            )
        
        # Try with wrong dtype for transmission (should be float)
        with pytest.raises(Exception):
            make_grid_params(
                X=X,
                Y=Y,
                phase_profile=phase_profile,
                transmission=jnp.ones(shape, dtype=jnp.complex128)  # Should be float
            )


class TestMakeOpticalWavefront(chex.TestCase):
    """Test the make_optical_wavefront factory function."""
    
    def test_make_optical_wavefront_with_valid_types(self):
        """Test that make_optical_wavefront works with valid types."""
        # Set up test values
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)
        z_position = jnp.array(0.0)
        
        # Create a valid OpticalWavefront instance
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=wavelength,
            dx=dx,
            z_position=z_position
        )
        
        # Check that the returned value is an OpticalWavefront instance
        assert isinstance(wavefront, OpticalWavefront)
        
        # Check that all attributes are set correctly
        chex.assert_trees_all_close(wavefront.field, field)
        assert wavefront.wavelength == wavelength
        assert wavefront.dx == dx
        assert wavefront.z_position == z_position
        
        # Check shapes and dtypes
        chex.assert_shape(wavefront.field, shape)
        assert wavefront.field.dtype == jnp.complex128
    
    def test_make_optical_wavefront_with_invalid_types(self):
        """Test that make_optical_wavefront raises an error with invalid types."""
        # Set up test values
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)
        z_position = jnp.array(0.0)
        
        # Try with non-complex field
        with pytest.raises(Exception):
            make_optical_wavefront(
                field=jnp.ones(shape),  # Should be complex array
                wavelength=wavelength,
                dx=dx,
                z_position=z_position
            )
        
        # Try with non-array wavelength
        with pytest.raises(Exception):
            make_optical_wavefront(
                field=field,
                wavelength=500e-9,  # Should be jnp.array
                dx=dx,
                z_position=z_position
            )
        
        # Try with non-array dx
        with pytest.raises(Exception):
            make_optical_wavefront(
                field=field,
                wavelength=wavelength,
                dx=1e-6,  # Should be jnp.array
                z_position=z_position
            )
        
        # Try with non-array z_position
        with pytest.raises(Exception):
            make_optical_wavefront(
                field=field,
                wavelength=wavelength,
                dx=dx,
                z_position=0.0  # Should be jnp.array
            )


if __name__ == "__main__":
    pytest.main([__file__])