"""Loss function implementations for ptychography optimization.

Extended Summary
----------------
Provides loss functions for comparing model outputs with
experimental data in ptychography applications. All functions
are JAX-compatible and support automatic differentiation for
optimization.

Routine Listings
----------------
:func:`create_loss_function`
    Factory that creates a JIT-compiled loss function
    supporting MAE, MSE, and RMSE loss types.

Notes
-----
All loss functions are designed to work with JAX transformations
including ``jit``, ``grad``, and ``vmap``. The
:func:`create_loss_function` factory returns a JIT-compiled
function that can be used with various optimization algorithms.
"""

import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array, Float, Num, PyTree


def create_loss_function(
    forward_function: Callable[..., Num[Array, " ..."]],
    experimental_data: Num[Array, " ..."],
    loss_type: str = "mae",
) -> Callable[..., Float[Array, ""]]:
    """Create a JIT-compiled loss function for ptychography.

    Extended Summary
    ----------------
    Returns a new function that computes the loss between the
    output of a forward model and experimental data. The
    returned function is JIT-compiled and can be used directly
    with gradient-based optimizers.

    Implementation Logic
    --------------------
    1. **Define internal loss functions** --
       ``mae_loss``, ``mse_loss``, and ``rmse_loss`` each take
       a difference array and return a scalar.
    2. **Select loss function** --
       Look up *loss_type* in the function dictionary.
    3. **Build JIT-compiled closure** --
       Create ``loss_fn`` that runs the forward model, computes
       the residual, and applies the selected loss.

    Parameters
    ----------
    forward_function : Callable[..., Array]
        The forward model function (e.g., a STEM simulation).
    experimental_data : Array
        The experimental data to compare against.
    loss_type : str, optional
        Loss variant to use. One of ``"mae"`` (Mean Absolute
        Error), ``"mse"`` (Mean Squared Error), or ``"rmse"``
        (Root Mean Squared Error). Default is ``"mae"``.

    Returns
    -------
    loss_fn : Callable[[PyTree, ...], Float[Array, ""]]
        A JIT-compiled function that computes the scalar loss
        given model parameters and any additional arguments
        required by the forward function.

    See Also
    --------
    :func:`~ptyrodactyl.tools.optimizers.wirtinger_grad`
        Wirtinger gradient for complex-valued optimisation.
    """

    def mae_loss(diff: Num[Array, " ..."]) -> Float[Array, " "]:
        """Compute mean absolute error from residuals.

        Parameters
        ----------
        diff : Array
            Residual array (model - experiment).

        Returns
        -------
        loss : Float[Array, ""]
            Scalar MAE value.
        """
        return jnp.mean(jnp.abs(diff))

    def mse_loss(diff: Num[Array, " ..."]) -> Float[Array, " "]:
        """Compute mean squared error from residuals.

        Parameters
        ----------
        diff : Array
            Residual array (model - experiment).

        Returns
        -------
        loss : Float[Array, ""]
            Scalar MSE value.
        """
        return jnp.mean(jnp.square(diff))

    def rmse_loss(diff: Num[Array, " ..."]) -> Float[Array, " "]:
        """Compute root mean squared error from residuals.

        Parameters
        ----------
        diff : Array
            Residual array (model - experiment).

        Returns
        -------
        loss : Float[Array, ""]
            Scalar RMSE value.
        """
        return jnp.sqrt(jnp.mean(jnp.square(diff)))

    loss_functions = {"mae": mae_loss, "mse": mse_loss, "rmse": rmse_loss}

    selected_loss_fn = loss_functions[loss_type]

    @jax.jit
    def loss_fn(params: PyTree, *args: Any) -> Float[Array, ""]:
        """Evaluate forward model and return scalar loss.

        Parameters
        ----------
        params : PyTree
            Model parameters passed to the forward function.
        *args : Any
            Additional positional arguments for the forward
            function.

        Returns
        -------
        loss : Float[Array, ""]
            Scalar loss value.
        """
        model_output = forward_function(params, *args)
        diff = model_output - experimental_data
        return selected_loss_fn(diff)

    return loss_fn


__all__: list[str] = [
    "create_loss_function",
]
