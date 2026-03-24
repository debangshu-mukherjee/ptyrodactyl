r"""Complex-valued optimizers with Wirtinger derivatives.

Extended Summary
----------------
Implements complex-valued optimization algorithms including
Adam, Adagrad, and RMSprop using Wirtinger calculus. Also
provides learning rate schedulers for training optimization.
All functions are JAX-compatible and support automatic
differentiation.

The Wirtinger derivative is defined as:

.. math::

    \frac{\partial f}{\partial z}
    = \frac{1}{2}\!\left(
        \frac{\partial f}{\partial x}
        - i\,\frac{\partial f}{\partial y}
    \right)

Routine Listings
----------------
:class:`LRSchedulerState`
    State maintained by learning rate schedulers.
:class:`OptimizerState`
    State maintained by optimizers (moments, step count).
:class:`Optimizer`
    Optimizer configuration with init and update functions.
:func:`create_cosine_scheduler`
    Cosine annealing learning rate scheduler.
:func:`create_step_scheduler`
    Step decay learning rate scheduler.
:func:`create_warmup_cosine_scheduler`
    Linear warmup followed by cosine decay scheduler.
:func:`init_scheduler_state`
    Initialise scheduler state with a given learning rate.
:func:`wirtinger_grad`
    Compute Wirtinger gradient of a complex-valued function.
:func:`complex_adam`
    One step of complex-valued Adam.
:func:`complex_adagrad`
    One step of complex-valued Adagrad.
:func:`complex_rmsprop`
    One step of complex-valued RMSprop.
:func:`init_adam`
    Initialise Adam optimizer state.
:func:`init_adagrad`
    Initialise Adagrad optimizer state.
:func:`init_rmsprop`
    Initialise RMSprop optimizer state.
:func:`adam_update`
    Adam parameter update step.
:func:`adagrad_update`
    Adagrad parameter update step.
:func:`rmsprop_update`
    RMSprop parameter update step.

Notes
-----
All optimizers use Wirtinger calculus for proper handling of
complex-valued parameters. All functions are designed to work
with JAX transformations including ``jit``, ``grad``, and
``vmap``.
"""

import jax
import jax.numpy as jnp
from beartype.typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from jaxtyping import Array, Complex, Float, Int, Num


class LRSchedulerState(NamedTuple):
    """State maintained by learning rate schedulers.

    Attributes
    ----------
    step : int
        Current optimisation step.
    learning_rate : float
        Current learning rate value.
    initial_lr : float
        Initial learning rate value.
    """

    step: int | Int[Array, " "]
    learning_rate: float | Float[Array, " "]
    initial_lr: float | Float[Array, " "]


SchedulerFn = Callable[
    [LRSchedulerState],
    tuple[float | Float[Array, " "], LRSchedulerState],
]


def create_cosine_scheduler(
    total_steps: int,
    final_lr_factor: float = 0.01,
) -> SchedulerFn:
    r"""Create a cosine annealing learning rate scheduler.

    Extended Summary
    ----------------
    Smoothly decreases the learning rate from the initial value
    to ``initial_lr * final_lr_factor`` over *total_steps*
    using a cosine curve:

    .. math::

        \eta_t = \eta_0 \bigl(
            \alpha + (1 - \alpha)\,
            \tfrac{1}{2}(1 + \cos(\pi\, p))
        \bigr)

    where :math:`p = \min(t / T,\; 1)` and
    :math:`\alpha` = *final_lr_factor*.

    Implementation Logic
    --------------------
    1. **Compute progress** --
       ``progress = min(step / total_steps, 1.0)``.
    2. **Cosine decay factor** --
       ``0.5 * (1 + cos(pi * progress))``.
    3. **Interpolate learning rate** --
       Linear interpolation between *final_lr_factor* and 1.
    4. **Update state** --
       Increment step and store new learning rate.

    Parameters
    ----------
    total_steps : int
        Total number of optimisation steps.
    final_lr_factor : float, optional
        Final learning rate as a fraction of the initial
        learning rate. Default is ``0.01``.

    Returns
    -------
    scheduler_fn : SchedulerFn
        A JIT-compiled function mapping
        :class:`LRSchedulerState` to ``(lr, new_state)``.
    """

    @jax.jit
    def scheduler_fn(
        state: LRSchedulerState,
    ) -> tuple[float | Float[Array, " "], LRSchedulerState]:
        """Apply cosine annealing to the learning rate.

        Parameters
        ----------
        state : LRSchedulerState
            Current scheduler state.

        Returns
        -------
        lr : float
            Updated learning rate.
        new_state : LRSchedulerState
            State with incremented step.
        """
        progress = jnp.minimum(state.step / total_steps, 1.0)
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        lr = state.initial_lr * (
            final_lr_factor + (1 - final_lr_factor) * cosine_decay
        )
        new_state = LRSchedulerState(
            step=state.step + 1, learning_rate=lr, initial_lr=state.initial_lr
        )
        return lr, new_state

    return scheduler_fn


def create_step_scheduler(step_size: int, gamma: float = 0.1) -> SchedulerFn:
    r"""Create a step decay learning rate scheduler.

    Extended Summary
    ----------------
    Reduces the learning rate by a multiplicative factor
    *gamma* every *step_size* steps:

    .. math::

        \eta_t = \eta_0 \,\gamma^{\lfloor t / S \rfloor}

    where :math:`S` = *step_size*.

    Implementation Logic
    --------------------
    1. **Count drops** --
       ``num_drops = step // step_size``.
    2. **Compute learning rate** --
       ``lr = initial_lr * gamma ** num_drops``.
    3. **Update state** --
       Increment step and store new learning rate.

    Parameters
    ----------
    step_size : int
        Number of steps between learning rate drops.
    gamma : float, optional
        Multiplicative decay factor. Default is ``0.1``.

    Returns
    -------
    scheduler_fn : SchedulerFn
        A JIT-compiled function mapping
        :class:`LRSchedulerState` to ``(lr, new_state)``.
    """

    @jax.jit
    def scheduler_fn(
        state: LRSchedulerState,
    ) -> tuple[float | Float[Array, " "], LRSchedulerState]:
        """Apply step decay to the learning rate.

        Parameters
        ----------
        state : LRSchedulerState
            Current scheduler state.

        Returns
        -------
        lr : float
            Updated learning rate.
        new_state : LRSchedulerState
            State with incremented step.
        """
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
    r"""Create a warmup-then-cosine-decay scheduler.

    Extended Summary
    ----------------
    Combines a linear warmup phase with cosine annealing.
    During warmup the learning rate increases linearly from
    zero to *initial_lr*; afterwards it follows a cosine
    decay to ``initial_lr * final_lr_factor``.

    .. math::

        \eta_t =
        \begin{cases}
            \eta_0 \, t / W & t < W \\
            \eta_0 \bigl(\alpha + (1-\alpha)\,
            \tfrac{1}{2}(1+\cos(\pi\,p))\bigr) & t \ge W
        \end{cases}

    where :math:`W` = *warmup_steps*,
    :math:`p = (t - W)/(T - W)`, and
    :math:`\alpha` = *final_lr_factor*.

    Implementation Logic
    --------------------
    1. **Linear warmup** --
       ``warmup_lr = initial_lr * min(step / warmup_steps, 1)``.
    2. **Cosine decay** --
       Compute decay progress and cosine factor after warmup.
    3. **Select phase** --
       Use ``jnp.where`` to pick warmup or decay LR.
    4. **Update state** --
       Increment step and store new learning rate.

    Parameters
    ----------
    total_steps : int
        Total number of optimisation steps.
    warmup_steps : int
        Number of linear warmup steps.
    final_lr_factor : float, optional
        Final learning rate as a fraction of the initial
        learning rate. Default is ``0.01``.

    Returns
    -------
    scheduler_fn : SchedulerFn
        A JIT-compiled function mapping
        :class:`LRSchedulerState` to ``(lr, new_state)``.
    """

    @jax.jit
    def scheduler_fn(
        state: LRSchedulerState,
    ) -> tuple[float | Float[Array, " "], LRSchedulerState]:
        """Apply warmup then cosine decay to the learning rate.

        Parameters
        ----------
        state : LRSchedulerState
            Current scheduler state.

        Returns
        -------
        lr : float
            Updated learning rate.
        new_state : LRSchedulerState
            State with incremented step.
        """
        # Linear warmup
        warmup_progress = jnp.minimum(state.step / warmup_steps, 1.0)
        warmup_lr = state.initial_lr * warmup_progress

        # Cosine decay after warmup
        remaining_steps = total_steps - warmup_steps
        decay_progress = (
            jnp.maximum(0.0, state.step - warmup_steps) / remaining_steps
        )
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
    """Initialise scheduler state with a given learning rate.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate value.

    Returns
    -------
    state : LRSchedulerState
        Scheduler state with ``step=0`` and
        ``learning_rate=initial_lr``.
    """
    return LRSchedulerState(
        step=0, learning_rate=initial_lr, initial_lr=initial_lr
    )


class OptimizerState(NamedTuple):
    """State maintained by optimizers.

    Attributes
    ----------
    m : Array
        First moment estimate (mean of gradients).
    v : Array
        Second moment estimate (mean of squared gradients).
    step : Array
        Scalar step counter.
    """

    m: Num[Array, " ..."]  # First moment estimate
    v: Num[Array, " ..."]  # Second moment estimate
    step: int | Int[Array, " "]  # Step count


class Optimizer(NamedTuple):
    """Optimizer configuration pairing init and update callables.

    Attributes
    ----------
    init : Callable
        Function to initialise :class:`OptimizerState`.
    update : Callable
        Function to update parameters given gradients and
        state.
    """

    init: Callable
    update: Callable


def wirtinger_grad(
    func2diff: Callable[..., Float[Array, " ..."]],
    argnums: Optional[Union[int, Sequence[int]]] = 0,
) -> Callable[
    ..., Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]
]:
    r"""Compute the Wirtinger gradient of a real-valued function.

    Extended Summary
    ----------------
    Returns a new function that computes the Wirtinger gradient
    of *func2diff* with respect to the argument(s) specified by
    *argnums*. The Wirtinger derivative is:

    .. math::

        \frac{\partial f}{\partial z}
        = \frac{1}{2}\!\left(
            \frac{\partial f}{\partial x}
            - i\,\frac{\partial f}{\partial y}
        \right)

    Implementation Logic
    --------------------
    1. **Split complex arguments** --
       Separate every complex argument into its real and
       imaginary parts, doubling the argument count.
    2. **Differentiate real and imaginary parts** --
       Use ``jax.grad`` on the real part and the imaginary
       part of the function output separately.
    3. **Recombine** --
       Form the Wirtinger gradient as
       ``0.5 * (grad_real - 1j * grad_imag)``.

    Parameters
    ----------
    func2diff : Callable[..., Float[Array, " ..."]]
        A function returning a real scalar to differentiate.
    argnums : Union[int, Sequence[int]], optional
        Which positional argument(s) to differentiate with
        respect to. Default is ``0``.

    Returns
    -------
    grad_f : Callable[..., Union[Complex[Array, " ..."], \
Tuple[Complex[Array, " ..."], ...]]]
        A function that returns the Wirtinger gradient(s).

    See Also
    --------
    :func:`complex_adam`
        Adam optimizer using Wirtinger gradients.
    :func:`complex_adagrad`
        Adagrad optimizer using Wirtinger gradients.
    :func:`complex_rmsprop`
        RMSprop optimizer using Wirtinger gradients.
    """

    def grad_f(
        *args: Any,
    ) -> Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]:
        """Evaluate the Wirtinger gradient at *args*.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to *func2diff*.

        Returns
        -------
        wirt_grad : Union[Complex[Array, " ..."], \
Tuple[Complex[Array, " ..."], ...]]
            Wirtinger gradient(s) for the selected arguments.
        """

        def split_complex(args: tuple) -> tuple:
            """Split complex args into real and imaginary parts.

            Parameters
            ----------
            args : tuple
                Original positional arguments.

            Returns
            -------
            split : tuple
                Real parts followed by imaginary parts.
            """
            return tuple(
                jnp.real(arg) if jnp.iscomplexobj(arg) else arg for arg in args
            ) + tuple(
                jnp.imag(arg) if jnp.iscomplexobj(arg) else jnp.zeros_like(arg)
                for arg in args
            )

        def combine_complex(r: tuple, i: tuple) -> tuple:
            """Recombine real and imaginary tuples.

            Parameters
            ----------
            r : tuple
                Real parts of each argument.
            i : tuple
                Imaginary parts of each argument.

            Returns
            -------
            combined : tuple
                Complex (or real) arguments.
            """
            return tuple(
                rr + 1j * ii if jnp.iscomplexobj(arg) else rr
                for rr, ii, arg in zip(r, i, args, strict=False)
            )

        split_args = split_complex(args)
        n = len(args)

        def f_real(*split_args: Num[Array, " ..."]) -> Float[Array, " ..."]:
            """Return the real part of the function output.

            Parameters
            ----------
            *split_args : Array
                Split real/imaginary arguments.

            Returns
            -------
            real_val : Float[Array, " ..."]
                Real part of ``func2diff`` output.
            """
            return jnp.real(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )

        def f_imag(*split_args: Num[Array, " ..."]) -> Float[Array, " ..."]:
            """Return the imaginary part of the function output.

            Parameters
            ----------
            *split_args : Array
                Split real/imaginary arguments.

            Returns
            -------
            imag_val : Float[Array, " ..."]
                Imaginary part of ``func2diff`` output.
            """
            return jnp.imag(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )

        gr = jax.grad(f_real, argnums=argnums)(*split_args)
        gi = jax.grad(f_imag, argnums=argnums)(*split_args)

        if isinstance(argnums, int):
            return 0.5 * (gr - 1j * gi)
        return tuple(
            0.5 * (grr - 1j * gii) for grr, gii in zip(gr, gi, strict=False)
        )

    return grad_f


def complex_adam(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: Tuple[Complex[Array, " ..."], Complex[Array, " ..."], int | Array],
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[
    Complex[Array, " ..."],
    Tuple[Complex[Array, " ..."], Complex[Array, " ..."], int | Array],
]:
    r"""Perform one step of complex-valued Adam.

    Extended Summary
    ----------------
    Applies the Adam update rule to complex-valued parameters
    using Wirtinger calculus. The bias-corrected update is:

    .. math::

        z_{t+1} = z_t
        - \frac{\eta\,\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}

    Implementation Logic
    --------------------
    1. **Increment timestep** -- ``t += 1``.
    2. **Update first moment** --
       ``m = beta1 * m + (1 - beta1) * grads``.
    3. **Update second moment** --
       ``v = beta2 * v + (1 - beta2) * |grads|^2``.
    4. **Bias-correct** --
       ``m_hat = m / (1 - beta1^t)``,
       ``v_hat = v / (1 - beta2^t)``.
    5. **Apply update** --
       ``new_params = params - lr * m_hat / (sqrt(v_hat) + eps)``.

    Parameters
    ----------
    params : Complex[Array, " ..."]
        Current complex-valued parameters.
    grads : Complex[Array, " ..."]
        Wirtinger gradients.
    state : Tuple[Complex[Array, " ..."], Complex[Array, " ..."], int]
        Optimizer state ``(m, v, t)``.
    learning_rate : float, optional
        Step size. Default is ``0.001``.
    beta1 : float, optional
        Exponential decay rate for the first moment.
        Default is ``0.9``.
    beta2 : float, optional
        Exponential decay rate for the second moment.
        Default is ``0.999``.
    eps : float, optional
        Small constant for numerical stability.
        Default is ``1e-8``.

    Returns
    -------
    new_params : Complex[Array, " ..."]
        Updated complex-valued parameters.
    new_state : Tuple[Complex[Array, " ..."], \
Complex[Array, " ..."], int]
        Updated optimizer state ``(m, v, t)``.

    See Also
    --------
    :func:`adam_update`
        Convenience wrapper using :class:`OptimizerState`.
    :func:`wirtinger_grad`
        Compute Wirtinger gradients.
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
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: Complex[Array, " ..."],
    learning_rate: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], Complex[Array, " ..."]]:
    r"""Perform one step of complex-valued Adagrad.

    Extended Summary
    ----------------
    Applies the Adagrad update rule to complex-valued parameters
    using Wirtinger calculus:

    .. math::

        z_{t+1} = z_t
        - \frac{\eta}{\sqrt{G_t} + \varepsilon}\,g_t

    where :math:`G_t = G_{t-1} + |g_t|^2`.

    Implementation Logic
    --------------------
    1. **Accumulate squared gradients** --
       ``G = G + |grads|^2``.
    2. **Adaptive learning rate** --
       ``lr_adaptive = lr / (sqrt(G) + eps)``.
    3. **Apply update** --
       ``new_params = params - lr_adaptive * grads``.

    Parameters
    ----------
    params : Complex[Array, " ..."]
        Current complex-valued parameters.
    grads : Complex[Array, " ..."]
        Wirtinger gradients.
    state : Complex[Array, " ..."]
        Accumulated squared gradients.
    learning_rate : float, optional
        Step size. Default is ``0.01``.
    eps : float, optional
        Small constant for numerical stability.
        Default is ``1e-8``.

    Returns
    -------
    new_params : Complex[Array, " ..."]
        Updated complex-valued parameters.
    new_state : Complex[Array, " ..."]
        Updated accumulated squared gradients.

    See Also
    --------
    :func:`adagrad_update`
        Convenience wrapper using :class:`OptimizerState`.
    :func:`wirtinger_grad`
        Compute Wirtinger gradients.
    """
    accumulated_grads = state

    # Update accumulated squared gradients
    new_accumulated_grads = accumulated_grads + jnp.abs(grads) ** 2

    # Compute adaptive learning rate
    adaptive_lr = learning_rate / (jnp.sqrt(new_accumulated_grads) + eps)

    # Update parameters
    new_params = params - adaptive_lr * grads

    return new_params, new_accumulated_grads


def complex_rmsprop(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: Complex[Array, " ..."],
    learning_rate: float = 0.001,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], Complex[Array, " ..."]]:
    r"""Perform one step of complex-valued RMSprop.

    Extended Summary
    ----------------
    Applies the RMSprop update rule to complex-valued parameters
    using Wirtinger calculus:

    .. math::

        v_t = \rho\,v_{t-1} + (1 - \rho)\,|g_t|^2

        z_{t+1} = z_t
        - \frac{\eta}{\sqrt{v_t} + \varepsilon}\,g_t

    Implementation Logic
    --------------------
    1. **Update moving average** --
       ``v = rho * v + (1 - rho) * |grads|^2``.
    2. **Adaptive learning rate** --
       ``lr_adaptive = lr / (sqrt(v) + eps)``.
    3. **Apply update** --
       ``new_params = params - lr_adaptive * grads``.

    Parameters
    ----------
    params : Complex[Array, " ..."]
        Current complex-valued parameters.
    grads : Complex[Array, " ..."]
        Wirtinger gradients.
    state : Complex[Array, " ..."]
        Moving average of squared gradients.
    learning_rate : float, optional
        Step size. Default is ``0.001``.
    decay_rate : float, optional
        Decay rate for the moving average.
        Default is ``0.9``.
    eps : float, optional
        Small constant for numerical stability.
        Default is ``1e-8``.

    Returns
    -------
    new_params : Complex[Array, " ..."]
        Updated complex-valued parameters.
    new_state : Complex[Array, " ..."]
        Updated moving average of squared gradients.

    See Also
    --------
    :func:`rmsprop_update`
        Convenience wrapper using :class:`OptimizerState`.
    :func:`wirtinger_grad`
        Compute Wirtinger gradients.
    """
    moving_avg = state

    # Update moving average of squared gradients
    new_moving_avg = (
        decay_rate * moving_avg + (1 - decay_rate) * jnp.abs(grads) ** 2
    )

    # Compute adaptive learning rate
    adaptive_lr = learning_rate / (jnp.sqrt(new_moving_avg) + eps)

    # Update parameters
    new_params = params - adaptive_lr * grads

    return new_params, new_moving_avg


def init_adam(shape: tuple) -> OptimizerState:
    """Initialise Adam optimizer state.

    Parameters
    ----------
    shape : tuple
        Shape of the parameters to be optimised.

    Returns
    -------
    state : OptimizerState
        State with zero first and second moments and
        ``step=0``.
    """
    return OptimizerState(
        m=jnp.zeros(shape), v=jnp.zeros(shape), step=jnp.array(0)
    )


def init_adagrad(shape: tuple) -> OptimizerState:
    """Initialise Adagrad optimizer state.

    Parameters
    ----------
    shape : tuple
        Shape of the parameters to be optimised.

    Returns
    -------
    state : OptimizerState
        State with zero accumulated gradients and ``step=0``.
    """
    return OptimizerState(
        m=jnp.zeros(shape), v=jnp.zeros(shape), step=jnp.array(0)
    )


def init_rmsprop(shape: tuple) -> OptimizerState:
    """Initialise RMSprop optimizer state.

    Parameters
    ----------
    shape : tuple
        Shape of the parameters to be optimised.

    Returns
    -------
    state : OptimizerState
        State with zero moving average and ``step=0``.
    """
    return OptimizerState(
        m=jnp.zeros(shape), v=jnp.zeros(shape), step=jnp.array(0)
    )


def adam_update(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: OptimizerState,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Complex[Array, " ..."], OptimizerState]:
    """Update parameters using Adam with Wirtinger derivatives.

    Implementation Logic
    --------------------
    1. **Unpack state** -- Extract ``m``, ``v``, ``step``.
    2. **Delegate** -- Call :func:`complex_adam`.
    3. **Repack state** -- Wrap results in
       :class:`OptimizerState`.

    Parameters
    ----------
    params : Complex[Array, " ..."]
        Current complex-valued parameters.
    grads : Complex[Array, " ..."]
        Wirtinger gradients.
    state : OptimizerState
        Current optimizer state.
    learning_rate : float, optional
        Step size. Default is ``0.001``.
    beta1 : float, optional
        First moment decay rate. Default is ``0.9``.
    beta2 : float, optional
        Second moment decay rate. Default is ``0.999``.
    eps : float, optional
        Numerical stability constant. Default is ``1e-8``.

    Returns
    -------
    new_params : Complex[Array, " ..."]
        Updated parameters.
    new_state : OptimizerState
        Updated optimizer state.

    See Also
    --------
    :func:`complex_adam`
        Low-level Adam implementation.
    """
    m, v, step = state
    new_params, (new_m, new_v, new_step) = complex_adam(
        params, grads, (m, v, step), learning_rate, beta1, beta2, eps
    )
    return new_params, OptimizerState(m=new_m, v=new_v, step=new_step)


def adagrad_update(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: OptimizerState,
    learning_rate: float = 0.01,
    eps: float = 1e-8,
) -> tuple[Complex[Array, " ..."], OptimizerState]:
    """Update parameters using Adagrad with Wirtinger derivatives.

    Implementation Logic
    --------------------
    1. **Unpack state** -- Extract ``m``, ``v``, ``step``.
    2. **Delegate** -- Call :func:`complex_adagrad` with ``v``
       as accumulated gradients.
    3. **Repack state** -- Wrap results in
       :class:`OptimizerState`.

    Parameters
    ----------
    params : Complex[Array, " ..."]
        Current complex-valued parameters.
    grads : Complex[Array, " ..."]
        Wirtinger gradients.
    state : OptimizerState
        Current optimizer state.
    learning_rate : float, optional
        Step size. Default is ``0.01``.
    eps : float, optional
        Numerical stability constant. Default is ``1e-8``.

    Returns
    -------
    new_params : Complex[Array, " ..."]
        Updated parameters.
    new_state : OptimizerState
        Updated optimizer state.

    See Also
    --------
    :func:`complex_adagrad`
        Low-level Adagrad implementation.
    """
    m, v, step = state
    new_params, new_v = complex_adagrad(params, grads, v, learning_rate, eps)
    return new_params, OptimizerState(m=m, v=new_v, step=step + 1)


def rmsprop_update(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: OptimizerState,
    learning_rate: float = 0.001,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
) -> tuple[Complex[Array, " ..."], OptimizerState]:
    """Update parameters using RMSprop with Wirtinger derivatives.

    Implementation Logic
    --------------------
    1. **Unpack state** -- Extract ``m``, ``v``, ``step``.
    2. **Delegate** -- Call :func:`complex_rmsprop` with ``v``
       as moving average.
    3. **Repack state** -- Wrap results in
       :class:`OptimizerState`.

    Parameters
    ----------
    params : Complex[Array, " ..."]
        Current complex-valued parameters.
    grads : Complex[Array, " ..."]
        Wirtinger gradients.
    state : OptimizerState
        Current optimizer state.
    learning_rate : float, optional
        Step size. Default is ``0.001``.
    decay_rate : float, optional
        Decay rate for the moving average.
        Default is ``0.9``.
    eps : float, optional
        Numerical stability constant. Default is ``1e-8``.

    Returns
    -------
    new_params : Complex[Array, " ..."]
        Updated parameters.
    new_state : OptimizerState
        Updated optimizer state.

    See Also
    --------
    :func:`complex_rmsprop`
        Low-level RMSprop implementation.
    """
    m, v, step = state
    new_params, new_v = complex_rmsprop(
        params, grads, v, learning_rate, decay_rate, eps
    )
    return new_params, OptimizerState(m=m, v=new_v, step=step + 1)


__all__: list[str] = [
    # Classes
    "LRSchedulerState",
    "Optimizer",
    "OptimizerState",
    # Functions
    "adagrad_update",
    "adam_update",
    "complex_adagrad",
    "complex_adam",
    "complex_rmsprop",
    "create_cosine_scheduler",
    "create_step_scheduler",
    "create_warmup_cosine_scheduler",
    "init_adagrad",
    "init_adam",
    "init_rmsprop",
    "init_scheduler_state",
    "rmsprop_update",
    "wirtinger_grad",
]
