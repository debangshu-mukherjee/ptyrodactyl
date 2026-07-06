"""Tests for :mod:`ptyrodactyl.jacobian.blocks`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/jacobian/blocks.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import jax.numpy as jnp
import pytest

from ptyrodactyl.jacobian.blocks import block_jacobian_operator
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


def test_block_jacobian_operator_rejects_invalid_block_name() -> None:
    """Invalid block labels raise at operator construction."""
    with pytest.raises(ValueError, match="not a valid OptimizableBlock"):
        block_jacobian_operator(
            _tiny_forward,
            _tiny_ptycho_params(),
            "bogus",
        )
