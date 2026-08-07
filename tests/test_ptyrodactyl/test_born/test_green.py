"""Tests for :mod:`ptyrodactyl.born.green`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/born/green.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.born.green import (
    convergence_parameter,
    green_function_fourier,
    reciprocal_coords,
    wavenumber_background,
)


def test_green_helpers_tiny_grid_smoke() -> None:
    """Exercise Green's-function helpers on a tiny grid.

    :see: :func:`ptyrodactyl.born.reciprocal_coords`
    :see: :func:`ptyrodactyl.born.wavenumber_background`
    """
    k_squared = jnp.ones((2, 2, 2))
    k0_squared = wavenumber_background(k_squared)
    scattering_potential = k_squared.astype(jnp.complex128) - k0_squared
    epsilon = convergence_parameter(scattering_potential)
    px, py, pz = reciprocal_coords((2, 2, 2), 1.0)
    g0_tilde = green_function_fourier((2, 2, 2), 1.0, k0_squared, epsilon)

    assert px.shape == (2, 2, 2)
    assert py.shape == (2, 2, 2)
    assert pz.shape == (2, 2, 2)
    assert g0_tilde.shape == (2, 2, 2)
    assert jnp.all(jnp.isfinite(g0_tilde))


def test_convergence_parameter_rejects_unit_safety_factor() -> None:
    """Safety factor must be strictly above the convergence bound.

    :see: :func:`ptyrodactyl.born.convergence_parameter`
    """
    with pytest.raises(
        ValueError,
        match="safety_factor must be greater than 1.0",
    ):
        convergence_parameter(
            jnp.ones((2, 2, 2), dtype=jnp.complex128),
            safety_factor=1.0,
        )


def test_green_function_fourier_rejects_nonfinite_k0_squared() -> None:
    """Non-finite scalar Green inputs raise through eqx.error_if.

    :see: :func:`ptyrodactyl.born.green_function_fourier`
    """
    with pytest.raises(Exception, match="k0_squared must be finite"):
        g0_tilde = green_function_fourier(
            (2, 2, 2),
            1.0,
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(1.0, dtype=jnp.float64),
        )
        jax.block_until_ready(g0_tilde)
