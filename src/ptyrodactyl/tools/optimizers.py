import jax
import jax.numpy as jnp
from beartype.typing import (Any, Callable, NamedTuple, Optional, Sequence,
                             Tuple, Union)
from jaxtyping import Array, Complex, Float


class LRSchedulerState(NamedTuple):
    """State maintained by learning rate schedulers."""

    step: int
    learning_rate: float
    initial_lr: float


SchedulerFn = Callable[[LRSchedulerState], tuple[float, LRSchedulerState]]


def create_cosine_scheduler(
    total_steps: int,
    final_lr_factor: float = 0.01,
) -> SchedulerFn:
    """
    Creates a cosine learning rate scheduler.

    Args:
        total_steps: Total number of optimization steps
        final_lr_factor: Final learning rate as a fraction of initial
    """

    @jax.jit
    def scheduler_fn(state: LRSchedulerState) -> tuple[float, LRSchedulerState]:
        progress = jnp.minimum(state.step / total_steps, 1.0)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        lr = state.initial_lr * (final_lr_factor + (1 - final_lr_factor) * cosine_decay)
        new_state = LRSchedulerState(
            step=state.step + 1, learning_rate=lr, initial_lr=state.initial_lr
        )
        return lr, new_state

    return scheduler_fn


def create_step_scheduler(step_size: int, gamma: float = 0.1) -> SchedulerFn:
    """
    Creates a step decay scheduler that reduces learning rate by gamma every step_size steps.

    Args:
        step_size: Number of steps between learning rate drops
        gamma: Multiplicative factor for learning rate decay
    """

    @jax.jit
    def scheduler_fn(state: LRSchedulerState) -> tuple[float, LRSchedulerState]:
        num_drops = state.step // step_size
        lr = state.initial_lr * (gamma**num_drops)
        new_state = LRSchedulerState(
            step=state.step + 1, learning_rate=lr, initial_lr=state.initial_lr
        )
        return lr, new_state

    return scheduler_fn


def create_warmup_cosine_scheduler(
    total_steps: int,
    warmup_steps: int,
    final_lr_factor: float = 0.01,
) -> SchedulerFn:
    """
    Creates a scheduler with linear warmup followed by cosine decay.

    Args:
        total_steps: Total number of optimization steps
        warmup_steps: Number of warmup steps
        final_lr_factor: Final learning rate as a fraction of initial
    """

    @jax.jit
    def scheduler_fn(state: LRSchedulerState) -> tuple[float, LRSchedulerState]:
        # Linear warmup
        warmup_progress = jnp.minimum(state.step / warmup_steps, 1.0)
        warmup_lr = state.initial_lr * warmup_progress

        # Cosine decay after warmup
        remaining_steps = total_steps - warmup_steps
        decay_progress = jnp.maximum(0.0, state.step - warmup_steps) / remaining_steps
        decay_progress = jnp.minimum(decay_progress, 1.0)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
        decay_lr = state.initial_lr * (
            final_lr_factor + (1 - final_lr_factor) * cosine_decay
        )

        # Choose between warmup and decay
        lr = jnp.where(state.step < warmup_steps, warmup_lr, decay_lr)

        new_state = LRSchedulerState(
            step=state.step + 1, learning_rate=lr, initial_lr=state.initial_lr
        )
        return lr, new_state

    return scheduler_fn


def init_scheduler_state(initial_lr: float) -> LRSchedulerState:
    """Initialize scheduler state with given learning rate."""
    return LRSchedulerState(step=0, learning_rate=initial_lr, initial_lr=initial_lr)


class OptimizerState(NamedTuple):
    m: Array  # First moment estimate
    v: Array  # Second moment estimate
    step: Array  # Step count


class Optimizer(NamedTuple):
    init: Callable
    update: Callable


def wirtinger_grad(
    func2diff: Callable[..., Float[Array, "..."]],
    argnums: Optional[Union[int, Sequence[int]]] = 0,
) -> Callable[..., Union[Complex[Array, "..."], Tuple[Complex[Array, "..."], ...]]]:
    """
    Description
    -----------
    Compute the Wirtinger gradient of a complex-valued function.
    This function returns a new function that computes the Wirtinger gradient
    of the input function f with respect to the specified argument(s).
    This is based on the formula for Wirtinger derivative:

    ∂f/∂z = ½(∂f/∂x - i∂f/∂y)

    Parameters
    ----------
    - `func2diff` (Callable[..., Float[Array, "..."]]):
        A complex-valued function to differentiate.
    - `argnums` (Union[int, Sequence[int]]):
        Specifies which argument(s) to compute the gradient with respect to.
        Can be an int or a sequence of ints. Default is 0.

    Returns
    -------
    - grad_f (Callable[..., Union[Complex[Array, "..."],
              Tuple[Complex[Array, "..."], ...]]]):
        A function that computes the Wirtinger gradient of f with respect to
        the specified argument(s).
    """

    def grad_f(
        *args: Any,
    ) -> Union[Complex[Array, "..."], Tuple[Complex[Array, "..."], ...]]:
        def split_complex(args):
            return tuple(
                jnp.real(arg) if jnp.iscomplexobj(arg) else arg for arg in args
            ) + tuple(
                jnp.imag(arg) if jnp.iscomplexobj(arg) else jnp.zeros_like(arg)
                for arg in args
            )

        def combine_complex(r, i):
            return tuple(
                rr + 1j * ii if jnp.iscomplexobj(arg) else rr
                for rr, ii, arg in zip(r, i, args)
            )

        split_args = split_complex(args)
        n = len(args)

        def f_real(*split_args):
            return jnp.real(func2diff(*combine_complex(split_args[:n], split_args[n:])))

        def f_imag(*split_args):
            return jnp.imag(func2diff(*combine_complex(split_args[:n], split_args[n:])))

        gr = jax.grad(f_real, argnums=argnums)(*split_args)
        gi = jax.grad(f_imag, argnums=argnums)(*split_args)

        if isinstance(argnums, int):
            return 0.5 * (gr - 1j * gi)
        else:
            return tuple(0.5 * (grr - 1j * gii) for grr, gii in zip(gr, gi))

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


def init_adam(shape: tuple) -> OptimizerState:
    return OptimizerState(jnp.zeros(shape), jnp.zeros(shape), jnp.array(0))


def init_adagrad(shape: tuple) -> OptimizerState:
    return OptimizerState(jnp.zeros(shape), jnp.zeros(shape), jnp.array(0))


def init_rmsprop(shape: tuple) -> OptimizerState:
    return OptimizerState(None, jnp.zeros(shape), jnp.array(0))


def adam_update(
    params: Complex[Array, "..."],
    grads: Complex[Array, "..."],
    state: OptimizerState,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Complex[Array, "..."], OptimizerState]:
    m, v, step = state
    step += 1
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * jnp.abs(grads) ** 2
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    update = learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    new_params = params - update
    return new_params, OptimizerState(m, v, step)


def adagrad_update(
    params: Complex[Array, "..."],
    grads: Complex[Array, "..."],
    state: OptimizerState,
    learning_rate: float = 0.01,
    eps: float = 1e-8,
) -> tuple[Complex[Array, "..."], OptimizerState]:
    _, v, step = state
    step += 1
    v += jnp.abs(grads) ** 2
    update = learning_rate * grads / (jnp.sqrt(v) + eps)
    new_params = params - update
    return new_params, OptimizerState(None, v, step)


def rmsprop_update(
    params: Complex[Array, "..."],
    grads: Complex[Array, "..."],
    state: OptimizerState,
    learning_rate: float = 0.001,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
) -> tuple[Complex[Array, "..."], OptimizerState]:
    _, v, step = state
    step += 1
    v = decay_rate * v + (1 - decay_rate) * jnp.abs(grads) ** 2
    update = learning_rate * grads / (jnp.sqrt(v) + eps)
    new_params = params - update
    return new_params, OptimizerState(None, v, step)
