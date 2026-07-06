"""Tests for :mod:`ptyrodactyl.types.electron_types`.

Extended Summary
----------------
Placeholder mirroring ``src/ptyrodactyl/types/electron_types.py`` per the
tests-mirror-src layout; coverage to be added with the module's
plan-driven rework.
"""

import os
import subprocess
import sys


def test_eqx_on_error_off_disables_runtime_error() -> None:
    """EQX_ON_ERROR=off disables data-dependent factory runtime errors."""
    script = """
import jax
import jax.numpy as jnp
from ptyrodactyl.types import create_calibrated_array

array = create_calibrated_array(
    jnp.ones((1, 1), dtype=jnp.float64),
    jnp.array(-1.0, dtype=jnp.float64),
    jnp.array(1.0, dtype=jnp.float64),
    True,
)
jax.block_until_ready(array.calib_y)
"""
    off_env = os.environ.copy()
    off_env["EQX_ON_ERROR"] = "off"
    off_result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=off_env,
        text=True,
    )

    raise_env = os.environ.copy()
    raise_env.pop("EQX_ON_ERROR", None)
    raise_result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=raise_env,
        text=True,
    )

    assert off_result.returncode == 0, off_result.stderr
    assert raise_result.returncode != 0
    assert "calib_y must be positive" in raise_result.stderr
