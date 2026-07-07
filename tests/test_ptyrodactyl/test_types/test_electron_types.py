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

import chex
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.types import combine_axis_updates, create_axis_update


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
        check=False,
        env=off_env,
        text=True,
    )

    raise_env = os.environ.copy()
    raise_env.pop("EQX_ON_ERROR", None)
    raise_result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=raise_env,
        text=True,
    )

    assert off_result.returncode == 0, off_result.stderr
    assert raise_result.returncode != 0
    assert "calib_y must be positive" in raise_result.stderr


def test_create_axis_update_zero_default_is_noop() -> None:
    """AxisUpdate defaults are the additive identity."""
    update = create_axis_update()

    chex.assert_trees_all_close(
        update.position_delta_ang,
        jnp.zeros((2,), dtype=jnp.float64),
        rtol=0.0,
        atol=0.0,
    )
    chex.assert_trees_all_close(
        update.energy_delta_ev,
        jnp.asarray(0.0, dtype=jnp.float64),
        rtol=0.0,
        atol=0.0,
    )
    chex.assert_trees_all_close(
        update.tilt_delta_mrad,
        jnp.zeros((2,), dtype=jnp.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_create_axis_update_rejects_nonfinite_values() -> None:
    """AxisUpdate validates finite dynamic leaves."""
    with pytest.raises(Exception, match="position_delta_ang must be finite"):
        update = create_axis_update(
            position_delta_ang=jnp.asarray([jnp.inf, 0.0], dtype=jnp.float64),
        )
        jax.block_until_ready(update.position_delta_ang)


def test_combine_axis_updates_sums_all_deltas() -> None:
    """Combiner sums each additive delta field."""
    first = create_axis_update(
        position_delta_ang=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        energy_delta_ev=jnp.asarray(3.0, dtype=jnp.float64),
        tilt_delta_mrad=jnp.asarray([4.0, 5.0], dtype=jnp.float64),
    )
    second = create_axis_update(
        position_delta_ang=jnp.asarray([-0.25, 0.5], dtype=jnp.float64),
        energy_delta_ev=jnp.asarray(-1.0, dtype=jnp.float64),
        tilt_delta_mrad=jnp.asarray([0.75, -2.0], dtype=jnp.float64),
    )

    combined = combine_axis_updates((first, second))

    chex.assert_trees_all_close(
        combined.position_delta_ang,
        jnp.asarray([0.75, 2.5], dtype=jnp.float64),
        rtol=0.0,
        atol=0.0,
    )
    chex.assert_trees_all_close(
        combined.energy_delta_ev,
        jnp.asarray(2.0, dtype=jnp.float64),
        rtol=0.0,
        atol=0.0,
    )
    chex.assert_trees_all_close(
        combined.tilt_delta_mrad,
        jnp.asarray([4.75, 3.0], dtype=jnp.float64),
        rtol=0.0,
        atol=0.0,
    )
