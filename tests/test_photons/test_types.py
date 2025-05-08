import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float

from ptyrodactyl.photons.photon_types import GridParams, LensParams, OpticalWavefront

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestLensParams(chex.TestCase):
    def test_lens_params_structure(self):
        """Test the structure of LensParams."""
        params = LensParams(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )
        
        # Check attributes
        assert hasattr(params, 'focal_length')
        assert hasattr(params, 'diameter')
        assert hasattr(params, 'n')
        assert hasattr(params, 'center_thickness')
        assert hasattr(params, 'R1')
        assert hasattr(params, 'R2')
    
    def test_lens_params_pytree_compatibility(self):
        """Test that LensParams is compatible with JAX transformations."""
        params = LensParams(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )
        
        # Test jit compatibility
        @jax.jit
        def fn(p):
            return p.focal_length * 2.0
        
        result = fn(params)
        assert result == 0.2
        
        # Test vmap compatibility
        array_params = LensParams(
            focal_length=jnp.array([0.1, 0.2]),
            diameter=jnp.array([0.05, 0.06]),
            n=jnp.array([1.5, 1.6]),
            center_thickness=jnp.array([0.01, 0.02]),
            R1=jnp.array([0.2, 0.3]),
            R2=jnp.array([0.2, 0.3]),
        )
        
        @jax.vmap
        def batch_fn(p):
            return p
        
        # This would fail if LensParams wasn't properly registered as a PyTree
        _ = batch_fn(array_params)


class TestGridParams(chex.TestCase):
    def test_grid_params_structure(self):
        """Test the structure of GridParams."""
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[0]))
        phase = jnp.zeros(shape)
        transmission = jnp.ones(shape)
        
        params = GridParams(
            X=X,
            Y=Y,
            phase_profile=phase,
            transmission=transmission
        )
        
        # Check attributes
        assert hasattr(params, 'X')
        assert hasattr(params, 'Y')
        assert hasattr(params, 'phase_profile')
        assert hasattr(params, 'transmission')
        
        # Check shapes
        chex.assert_shape(params.X, shape)
        chex.assert_shape(params.Y, shape)
        chex.assert_shape(params.phase_profile, shape)
        chex.assert_shape(params.transmission, shape)
    
    def test_grid_params_pytree_compatibility(self):
        """Test that GridParams is compatible with JAX transformations."""
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[0]))
        phase = jnp.zeros(shape)
        transmission = jnp.ones(shape)
        
        params = GridParams(
            X=X,
            Y=Y,
            phase_profile=phase,
            transmission=transmission
        )
        
        # Test jit compatibility
        @jax.jit
        def fn(p):
            return jnp.sum(p.X + p.Y)
        
        _ = fn(params)


class TestOpticalWavefront(chex.TestCase):
    def test_wavefront_structure(self):
        """Test the structure of OpticalWavefront."""
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)
        
        wavefront = OpticalWavefront(
            field=field,
            wavelength=jnp.array(500e-9),
            dx=jnp.array(1e-6),
            z_position=jnp.array(0.0)
        )
        
        # Check attributes
        assert hasattr(wavefront, 'field')
        assert hasattr(wavefront, 'wavelength')
        assert hasattr(wavefront, 'dx')
        assert hasattr(wavefront, 'z_position')
        
        # Check shapes
        chex.assert_shape(wavefront.field, shape)
    
    def test_wavefront_pytree_compatibility(self):
        """Test that OpticalWavefront is compatible with JAX transformations."""
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)
        
        wavefront = OpticalWavefront(
            field=field,
            wavelength=jnp.array(500e-9),
            dx=jnp.array(1e-6),
            z_position=jnp.array(0.0)
        )
        
        # Test jit compatibility
        @jax.jit
        def fn(w):
            return jnp.sum(jnp.abs(w.field))
        
        result = fn(wavefront)
        assert result == 32 * 32
        
        # Test that the wavefront is compatible with JAX transformations
        @jax.jit
        def propagate(w, z):
            return OpticalWavefront(
                field=w.field,
                wavelength=w.wavelength,
                dx=w.dx,
                z_position=w.z_position + z
            )
        
        new_wavefront = propagate(wavefront, jnp.array(0.1))
        assert new_wavefront.z_position == 0.1


if __name__ == "__main__":
    pytest.main([__file__])