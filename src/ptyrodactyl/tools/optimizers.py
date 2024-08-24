from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def wirtinger_grad(
    f: Callable[[Complex[Array, "..."], Any], Float[Array, "..."]]
) -> Callable[[Complex[Array, "..."], Any], Complex[Array, "..."]]:
    """
    Compute the Wirtinger gradient of a complex-valued function.

    This function returns a new function that computes the Wirtinger gradient
    of the input function f with respect to its first argument.

    Args:
    - f (Callable[[Complex[Array, "..."], Any], Float[Array, "..."]]):
        A complex-valued function to differentiate.

    Returns:
    - grad_f (Callable[[Complex[Array, "..."], Any], Complex[Array, "..."]]):
        A function that computes the Wirtinger gradient of f.
    """

    def grad_f(z: Complex[Array, "..."], *args: Any) -> Complex[Array, "..."]:
        r = jnp.real(z)
        i = jnp.imag(z)

        def f_real(
            r: Float[Array, "..."], i: Float[Array, "..."], *args: Any
        ) -> Float[Array, "..."]:
            return jnp.real(f(r + 1j * i, *args))

        def f_imag(
            r: Float[Array, "..."], i: Float[Array, "..."], *args: Any
        ) -> Float[Array, "..."]:
            return jnp.imag(f(r + 1j * i, *args))

        gr = jax.grad(f_real, argnums=0)(r, i, *args)
        gi = jax.grad(f_imag, argnums=1)(r, i, *args)
        return gr + 1j * gi

    return grad_f


def complex_adam(
    params: Complex[Array, "..."],
    grads: Complex[Array, "..."],
    state: Tuple[Complex[Array, "..."], Complex[Array, "..."], int],
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[
    Complex[Array, "..."], Tuple[Complex[Array, "..."], Complex[Array, "..."], int]
]:
    """
    Complex-valued Adam optimizer based on Wirtinger derivatives.

    This function performs one step of the Adam optimization algorithm
    for complex-valued parameters.

    Args:
    - params (Complex[Array, "..."]):
        Current complex-valued parameters.
    - grads (Complex[Array, "..."]):
        Complex-valued gradients.
    - state (Tuple[Complex[Array, "..."], Complex[Array, "..."], int]):
        Optimizer state (first moment, second moment, timestep).
    - learning_rate (float):
        Learning rate (default: 0.001).
    - beta1 (float):
        Exponential decay rate for first moment estimates (default: 0.9).
    - beta2 (float):
        Exponential decay rate for second moment estimates (default: 0.999).
    - eps (float):
        Small value to avoid division by zero (default: 1e-8).

    Returns:
    - new_params (Complex[Array, "..."]):
        Updated parameters.
    - new_state (Tuple[Complex[Array, "..."], Complex[Array, "..."], int]):
        Updated optimizer state.
    """
    m, v, t = state
    t += 1
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * jnp.abs(grads) ** 2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    update = learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    new_params = params - update
    return new_params, (m, v, t)


def complex_adagrad(
    params: Complex[Array, "..."],
    grads: Complex[Array, "..."],
    state: Complex[Array, "..."],
    learning_rate: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, "..."], Complex[Array, "..."]]:
    """
    Complex-valued Adagrad optimizer based on Wirtinger derivatives.

    This function performs one step of the Adagrad optimization algorithm
    for complex-valued parameters.

    Args:
    - params (Complex[Array, "..."]):
        Current complex-valued parameters.
    - grads (Complex[Array, "..."]):
        Complex-valued gradients.
    - state (Complex[Array, "..."]):
        Optimizer state (accumulated squared gradients).
    - learning_rate (float):
        Learning rate (default: 0.01).
    - eps (float):
        Small value to avoid division by zero (default: 1e-8).

    Returns:
    - new_params (Complex[Array, "..."]): Updated parameters.
    - new_state (Complex[Array, "..."]): Updated optimizer state.
    """
    accumulated_grads = state

    # Update accumulated squared gradients
    new_accumulated_grads = accumulated_grads + jnp.abs(grads) ** 2

    # Compute update
    update = learning_rate * grads / (jnp.sqrt(new_accumulated_grads) + eps)

    # Update parameters
    new_params = params - update

    return new_params, new_accumulated_grads


def complex_rmsprop(
    params: Complex[Array, "..."],
    grads: Complex[Array, "..."],
    state: Complex[Array, "..."],
    learning_rate: float = 0.001,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, "..."], Complex[Array, "..."]]:
    """
    Complex-valued RMSprop optimizer based on Wirtinger derivatives.

    This function performs one step of the RMSprop optimization algorithm
    for complex-valued parameters.

    Args:
    - params (Complex[Array, "..."]):
        Current complex-valued parameters.
    - grads (Complex[Array, "..."]):
        Complex-valued gradients.
    - state (Complex[Array, "..."]):
        Optimizer state (moving average of squared gradients).
    - learning_rate (float):
        Learning rate (default: 0.001).
    - decay_rate (float):
        Decay rate for moving average (default: 0.9).
    - eps (float):
        Small value to avoid division by zero (default: 1e-8).

    Returns:
    - new_params (Complex[Array, "..."]): Updated parameters.
    - new_state (Complex[Array, "..."]): Updated optimizer state.
    """
    moving_avg_squared_grads = state

    # Update moving average of squared gradients
    new_moving_avg_squared_grads = (
        decay_rate * moving_avg_squared_grads + (1 - decay_rate) * jnp.abs(grads) ** 2
    )

    # Compute update
    update = learning_rate * grads / (jnp.sqrt(new_moving_avg_squared_grads) + eps)

    # Update parameters
    new_params = params - update

    return new_params, new_moving_avg_squared_grads
