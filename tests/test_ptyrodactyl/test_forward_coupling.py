"""Test Lobato potential signs, units, and Born forward coupling."""

import jax.numpy as jnp

from ptyrodactyl.born import (
    convergence_parameter,
    green_function_fourier,
    wavenumber_background,
)
from ptyrodactyl.multislice.potential_volume import crystal_potential_volume
from ptyrodactyl.types import (
    create_crystal_data,
    helmholtz_coupling,
    relativistic_wavelength_ang,
)


def test_lobato_potential_has_finite_nonzero_born_response() -> None:
    """A positive Lobato voltage field yields a finite Born response."""
    shape = (6, 6, 6)
    spacing = 0.5
    crystal = create_crystal_data(
        jnp.array([[1.5, 1.5, 1.5]], dtype=jnp.float64),
        jnp.array([6], dtype=jnp.int32),
        lattice=jnp.eye(3, dtype=jnp.float64) * 3.0,
    )
    potential = crystal_potential_volume(
        crystal,
        spacing,
        shape,
        band_limit=0.75,
    )

    sigma_h = helmholtz_coupling(100.0)
    chi_real = sigma_h * potential.volume
    wavelength = relativistic_wavelength_ang(100.0)
    vacuum_k_squared = jnp.square(2.0 * jnp.pi / wavelength)
    local_k_squared = vacuum_k_squared + chi_real
    background_k_squared = wavenumber_background(local_k_squared)
    scattering = local_k_squared - background_k_squared
    epsilon = convergence_parameter(
        scattering.astype(jnp.complex128),
        safety_factor=1.05,
    )
    green = green_function_fourier(
        shape,
        spacing,
        background_k_squared,
        epsilon,
    )
    incident = jnp.ones(shape, dtype=jnp.complex128)
    first_born_field = jnp.fft.ifftn(green * jnp.fft.fftn(chi_real * incident))

    center = tuple(axis // 2 for axis in shape)
    assert potential.units == "V"
    assert potential.reference_value == 0.0
    assert jnp.mean(potential.volume) > 0.0
    assert potential.volume[center] > 0.0
    assert sigma_h > 0.0
    assert chi_real[center] > 0.0
    assert local_k_squared[center] > vacuum_k_squared
    assert epsilon > jnp.max(jnp.abs(scattering))
    assert jnp.all(jnp.isfinite(first_born_field))
    assert jnp.linalg.norm(first_born_field) > 0.0
