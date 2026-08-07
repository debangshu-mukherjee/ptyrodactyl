"""Tests for :mod:`ptyrodactyl.tools.loss_functions`.

:see: :func:`ptyrodactyl.tools.create_loss_function`
"""

import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.tools import create_loss_function
from ptyrodactyl.types import LossType


def _identity_forward(params):
    """Return input parameters unchanged."""
    return params


def test_create_loss_function_rejects_invalid_loss_type() -> None:
    """Invalid loss labels raise at the factory boundary."""
    with pytest.raises(ValueError, match="not a valid LossType"):
        create_loss_function(
            _identity_forward,
            jnp.zeros((2,), dtype=jnp.float64),
            loss_type="bogus",
        )


def test_create_loss_function_accepts_enum_and_string_loss_type() -> None:
    """Enum and string loss selections produce identical values."""
    experimental_data = jnp.array([1.0, -2.0], dtype=jnp.float64)
    params = jnp.array([3.0, 2.0], dtype=jnp.float64)

    string_loss = create_loss_function(
        _identity_forward,
        experimental_data,
        loss_type="mse",
    )
    enum_loss = create_loss_function(
        _identity_forward,
        experimental_data,
        loss_type=LossType.MSE,
    )

    string_value = string_loss(params)
    enum_value = enum_loss(params)
    jax.block_until_ready((string_value, enum_value))

    assert string_value == enum_value
