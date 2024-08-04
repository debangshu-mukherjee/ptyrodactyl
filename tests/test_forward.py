import pytest
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Complex, Shaped, Int

# Import your functions here
from pterodactyl.electrons import FourierCoords, FourierCalib, make_probe, aberration, wavelength_ang

# Set a random seed for reproducibility
jax.random.PRNGKey(0)

def test_FourierCoords():
    calibration = 0.1
    sizebeam = jnp.array([128, 128])
    dL, L1 = FourierCoords(calibration, sizebeam)
    
    assert isinstance(dL, float)
    assert isinstance(L1, jax.Array)
    assert L1.shape == (128, 128)
    assert jnp.isclose(dL, 1 / (128 * 0.1), atol=1e-6)
    assert jnp.allclose(L1[64, 64], 0, atol=1e-6)  # Center should be close to 0

def test_FourierCalib():
    calibration = 0.1
    sizebeam = jnp.array([128, 256])
    result = FourierCalib(calibration, sizebeam)
    
    assert isinstance(result, jax.Array)
    assert result.shape == (2,)
    assert jnp.isclose(result[0], 1 / (128 * 0.1), atol=1e-6)
    assert jnp.isclose(result[1], 1 / (256 * 0.1), atol=1e-6)

def test_make_probe():
    aperture = 30.0  # mrad
    voltage = 300.0  # kV
    image_size = jnp.array([128, 128])
    calibration_pm = 10.0  # pm
    
    probe = make_probe(aperture, voltage, image_size, calibration_pm)
    
    assert isinstance(probe, jax.Array)
    assert probe.shape == (128, 128)
    assert probe.dtype == jnp.complex64
    assert jnp.allclose(jnp.sum(jnp.abs(probe)**2), 1.0, atol=1e-6)  # Check normalization

def test_aberration():
    fourier_coord = jnp.ones((128, 128))
    wavelength = 1.97e-12  # for 300 kV
    defocus = 50.0  # nm
    c3 = 1.0  # mm
    c5 = 1.0  # mm
    
    result = aberration(fourier_coord, wavelength, defocus, c3, c5)
    
    assert isinstance(result, jax.Array)
    assert result.shape == (128, 128)
    assert jnp.all(result != 0)  # Should have non-zero values

def test_wavelength_ang():
    voltage_kV = 300.0
    wavelength = wavelength_ang(voltage_kV)
    
    assert isinstance(wavelength, float)
    assert jnp.isclose(wavelength, 1.97e-2, atol=1e-4)  # Expected wavelength for 300 kV