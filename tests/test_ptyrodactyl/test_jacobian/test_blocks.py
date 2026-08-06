"""Tests for :mod:`ptyrodactyl.jacobian.blocks`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/blocks.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.jacobian.blocks import (
    alternating_block_solve,
    block_gauss_newton_step,
    block_jacobian_operator,
)
from ptyrodactyl.types import PtychoParams, create_ptycho_params


def _tiny_ptycho_params() -> PtychoParams:
    """Build a minimal valid ptychography parameter tree."""
    params: PtychoParams = create_ptycho_params(
        exit_wave=jnp.ones((2, 2), dtype=jnp.complex128),
        zernike_coeffs=jnp.zeros((1,), dtype=jnp.float64),
        aperture_mrad=jnp.array(20.0, dtype=jnp.float64),
        aperture_softness=jnp.array(0.5, dtype=jnp.float64),
        rotation_rad=jnp.array(0.0, dtype=jnp.float64),
        center_offset=jnp.zeros((2,), dtype=jnp.float64),
        ellipticity=jnp.zeros((2,), dtype=jnp.float64),
        position_offsets=jnp.zeros((1, 2), dtype=jnp.float64),
        mode_weights=jnp.ones((1,), dtype=jnp.float64),
        mode_phases=jnp.zeros((1, 2, 2), dtype=jnp.float64),
    )
    return params


def _tiny_forward(params: PtychoParams):
    """Map tiny params to the block-operator output contract."""
    return jnp.real(params.exit_wave.wave)[None, :, :]


def _complex_linear_forward(params: PtychoParams):
    """Observe independent real and imaginary exit-wave directions."""
    wave = params.exit_wave.wave
    return (jnp.real(wave) + 2.0 * jnp.imag(wave))[None, :, :]


def test_block_jacobian_operator_rejects_invalid_block_name() -> None:
    """Invalid block labels raise at operator construction."""
    with pytest.raises(ValueError, match="not a valid OptimizableBlock"):
        block_jacobian_operator(
            _tiny_forward,
            _tiny_ptycho_params(),
            "bogus",
        )


def test_block_gauss_newton_step_updates_complex_exit_wave() -> None:
    """A block GN step reduces residuals along both complex components."""
    params = eqx.tree_at(
        lambda p: p.exit_wave.wave,
        _tiny_ptycho_params(),
        jnp.array(
            [[1.0 + 1.0j, -0.5 + 2.0j], [2.0 - 1.0j, -1.0 - 0.5j]],
            dtype=jnp.complex128,
        ),
    )
    data = jnp.zeros((1, 2, 2), dtype=jnp.float64)

    updated = block_gauss_newton_step(
        _complex_linear_forward,
        params,
        data,
        ["exit_wave"],
        cg_max_iterations=4,
        cg_tolerance=1e-12,
    )

    initial_norm = jnp.linalg.norm(_complex_linear_forward(params) - data)
    updated_norm = jnp.linalg.norm(_complex_linear_forward(updated) - data)
    assert updated_norm < initial_norm
    assert jnp.allclose(updated_norm, 0.0, atol=1e-10)
    assert not jnp.allclose(updated.exit_wave.wave, params.exit_wave.wave)


def test_alternating_block_solve_complex_exit_wave_jit() -> None:
    """The complex exit-wave block remains a valid JIT/scan carry."""
    params = eqx.tree_at(
        lambda p: p.exit_wave.wave,
        _tiny_ptycho_params(),
        jnp.full((2, 2), 1.0 + 1.0j, dtype=jnp.complex128),
    )
    data = jnp.zeros((1, 2, 2), dtype=jnp.float64)

    @jax.jit
    def solve(initial_params, observations):
        return alternating_block_solve(
            _complex_linear_forward,
            initial_params,
            observations,
            [["exit_wave"]],
            num_outer_iterations=1,
            cg_max_iterations=4,
            cg_tolerance=1e-12,
        )

    updated, residual_history = solve(params, data)

    assert residual_history.shape == (1,)
    assert jnp.all(jnp.isfinite(updated.exit_wave.wave))
    assert jnp.allclose(residual_history, 0.0, atol=1e-10)
