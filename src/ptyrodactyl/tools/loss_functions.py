from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PyTree


def create_loss_function(
    forward_function: Callable[..., Array],
    experimental_data: Array,
    loss_type: str = "mae",
) -> Callable[..., Float[Array, ""]]:
    """
    Create a JIT-compatible loss function for comparing model output with experimental data.

    This function returns a new function that computes the loss between the output
    of a forward model and experimental data. The returned function is JIT-compatible
    and can be used with various optimization algorithms.

    Args:
    - forward_function (Callable[..., Array]):
        The forward model function (e.g., stem_4d).
    - experimental_data (Array):
        The experimental data to compare against.
    - loss_type (Literal["mae", "mse", "rmse"]):
        The type of loss to use. Options are "mae" (Mean Absolute Error),
        "mse" (Mean Squared Error), or "rmse" (Root Mean Squared Error).
        Default is "mae".

    Loss Functions:
    - `mae_loss`:
        Mean Absolute Error loss function.
    - `mse_loss`:
        Mean Squared Error loss function.
    - `rmse_loss`:
        Root Mean Squared Error loss function.

    Returns:
    - loss_fn (Callable[[PyTree, ...], Float[Array, ""]]):
        A JIT-compatible function that computes the loss given the model parameters
        and any additional arguments required by the forward function.
    """

    def mae_loss(diff):
        return jnp.mean(jnp.abs(diff))

    def mse_loss(diff):
        return jnp.mean(jnp.square(diff))

    def rmse_loss(diff):
        return jnp.sqrt(jnp.mean(jnp.square(diff)))

    loss_functions = {"mae": mae_loss, "mse": mse_loss, "rmse": rmse_loss}

    selected_loss_fn = loss_functions[loss_type]

    @jax.jit
    def loss_fn(params: PyTree, *args: Any) -> Float[Array, ""]:
        # Compute the forward model
        model_output = forward_function(params, *args)

        # Compute the difference
        diff = model_output - experimental_data

        # Compute the loss
        return selected_loss_fn(diff)

    return loss_fn
