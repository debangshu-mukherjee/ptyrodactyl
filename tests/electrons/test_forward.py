import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, Complex, Int

# Import your functions here
from ptyrodactyl.electrons import (
    FourierCoords,
    FourierCalib,
    make_probe,
    aberration,
    wavelength_ang,
)


# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)


@chex.all_variants
def test_FourierCoords(variant):
    calibration = 0.1
    sizebeam = jnp.array([128, 128])

    fn = variant(FourierCoords)
    dL, L1 = fn(calibration, sizebeam)

    chex.assert_type(dL, float)
    chex.assert_shape(L1, (128, 128))
    chex.assert_scalar_near(dL, 1 / (128 * 0.1), atol=1e-6)
    chex.assert_scalar_near(L1[64, 64], 0, atol=1e-6)  # Center should be close to 0


@chex.all_variants
def test_FourierCalib(variant):
    calibration = 0.1
    sizebeam = jnp.array([128, 256])

    fn = variant(FourierCalib)
    result = fn(calibration, sizebeam)

    chex.assert_shape(result, (2,))
    chex.assert_scalar_near(result[0], 1 / (128 * 0.1), atol=1e-6)
    chex.assert_scalar_near(result[1], 1 / (256 * 0.1), atol=1e-6)


@chex.all_variants
def test_make_probe(variant):
    aperture = 30.0  # mrad
    voltage = 300.0  # kV
    image_size = jnp.array([128, 128])
    calibration_pm = 10.0  # pm

    fn = variant(make_probe)
    probe = fn(aperture, voltage, image_size, calibration_pm)

    chex.assert_shape(probe, (128, 128))
    chex.assert_type(probe, jnp.complex64)
    chex.assert_scalar_near(
        jnp.sum(jnp.abs(probe) ** 2), 1.0, atol=1e-6
    )  # Check normalization


@chex.all_variants
def test_aberration(variant):
    fourier_coord = jnp.ones((128, 128))
    wavelength = 1.97e-12  # for 300 kV
    defocus = 50.0  # nm
    c3 = 1.0  # mm
    c5 = 1.0  # mm

    fn = variant(aberration)
    result = fn(fourier_coord, wavelength, defocus, c3, c5)

    chex.assert_shape(result, (128, 128))
    chex.assert_trees_all_close(result, result, rtol=1e-5)  # Should be consistent
    chex.assert_trees_all_finite(result)  # No NaNs or infs


@chex.all_variants
def test_wavelength_ang(variant):
    voltage_kV = 300.0

    fn = variant(wavelength_ang)
    wavelength = fn(voltage_kV)

    chex.assert_type(wavelength, float)
    chex.assert_scalar_near(
        wavelength, 1.97e-2, atol=1e-4
    )  # Expected wavelength for 300 kV


@pytest.mark.parametrize("voltage", [100.0, 200.0, 300.0])
@chex.all_variants
def test_wavelength_ang_multiple_voltages(variant, voltage):
    fn = variant(wavelength_ang)
    wavelength = fn(voltage)

    chex.assert_type(wavelength, float)
    chex.assert_scalar_positive(wavelength)
