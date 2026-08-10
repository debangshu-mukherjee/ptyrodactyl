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
:class:`Optimizer`
    Optimizer configuration pairing init and update callables.
:class:`OptimizerState`
    State maintained by optimizers.
:func:`adagrad_update`
    Update parameters using Adagrad with Wirtinger derivatives.
:func:`adam_update`
    Update parameters using Adam with Wirtinger derivatives.
:func:`complex_adagrad`
    Perform one step of complex-valued Adagrad.
:func:`complex_adam`
    Perform one step of complex-valued Adam.
:func:`complex_rmsprop`
    Perform one step of complex-valued RMSprop.
:func:`create_cosine_scheduler`
    Create a cosine annealing learning rate scheduler.
:func:`create_step_scheduler`
    Create a step decay learning rate scheduler.
:func:`create_warmup_cosine_scheduler`
    Create a warmup-then-cosine-decay scheduler.
:func:`init_adagrad`
    Initialise Adagrad optimizer state.
:func:`init_adam`
    Initialise Adam optimizer state.
:func:`init_rmsprop`
    Initialise RMSprop optimizer state.
:func:`init_scheduler_state`
    Initialise scheduler state with a given learning rate.
:func:`rmsprop_update`
    Update parameters using RMSprop with Wirtinger derivatives.
:func:`wirtinger_grad`
    Compute the Wirtinger gradient of a real-valued function.

Notes
-----
All optimizers use Wirtinger calculus for proper handling of
complex-valued parameters. All functions are designed to work
with JAX transformations including ``jit``, ``grad``, and
``vmap``.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped


class LRSchedulerState(NamedTuple):
    """State maintained by learning rate schedulers.

    :see: :mod:`~.test_optimizers`

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
    Tuple[float | Float[Array, " "], LRSchedulerState],
]


@jaxtyped(typechecker=beartype)
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

    :see: :mod:`~.test_optimizers`

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
    scheduler : SchedulerFn
        A JIT-compiled function mapping
        :class:`LRSchedulerState` to ``(lr, new_state)``.
    """

    @jax.jit
    def scheduler_fn(
        state: LRSchedulerState,
    ) -> Tuple[float | Float[Array, " "], LRSchedulerState]:
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
        result: Tuple[float | Float[Array, " "], LRSchedulerState] = (
            lr,
            new_state,
        )
        return result

    scheduler: SchedulerFn = scheduler_fn
    return scheduler


@jaxtyped(typechecker=beartype)
def create_step_scheduler(step_size: int, gamma: float = 0.1) -> SchedulerFn:
    r"""Create a step decay learning rate scheduler.

    Extended Summary
    ----------------
    Reduces the learning rate by a multiplicative factor
    *gamma* every *step_size* steps:

    .. math::

        \eta_t = \eta_0 \,\gamma^{\lfloor t / S \rfloor}

    where :math:`S` = *step_size*.

    :see: :mod:`~.test_optimizers`

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
    scheduler : SchedulerFn
        A JIT-compiled function mapping
        :class:`LRSchedulerState` to ``(lr, new_state)``.
    """

    @jax.jit
    def scheduler_fn(
        state: LRSchedulerState,
    ) -> Tuple[float | Float[Array, " "], LRSchedulerState]:
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
        result: Tuple[float | Float[Array, " "], LRSchedulerState] = (
            lr,
            new_state,
        )
        return result

    scheduler: SchedulerFn = scheduler_fn
    return scheduler


@jaxtyped(typechecker=beartype)
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

    :see: :mod:`~.test_optimizers`

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
    scheduler : SchedulerFn
        A JIT-compiled function mapping
        :class:`LRSchedulerState` to ``(lr, new_state)``.
    """

    @jax.jit
    def scheduler_fn(
        state: LRSchedulerState,
    ) -> Tuple[float | Float[Array, " "], LRSchedulerState]:
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
        result: Tuple[float | Float[Array, " "], LRSchedulerState] = (
            lr,
            new_state,
        )
        return result

    scheduler: SchedulerFn = scheduler_fn
    return scheduler


@jaxtyped(typechecker=beartype)
def init_scheduler_state(
    initial_lr: float | Float[Array, " "],
) -> LRSchedulerState:
    """Initialise scheduler state with a given learning rate.

    :see: :mod:`~.test_optimizers`

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
    state: LRSchedulerState = LRSchedulerState(
        step=0, learning_rate=initial_lr, initial_lr=initial_lr
    )
    return state


class OptimizerState(NamedTuple):
    """State maintained by optimizers.

    :see: :mod:`~.test_optimizers`

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

    :see: :mod:`~.test_optimizers`

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


@jaxtyped(typechecker=beartype)
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

    :see: :func:`~.test_wirtinger_grad_matches_known_complex_quadratic`

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
    gradient_function : Callable
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
        wirt_grad : Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]
            Wirtinger gradient(s) for the selected arguments.
        """  # noqa: E501

        def split_complex(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
            """Split complex args into real and imaginary parts.

            Parameters
            ----------
            args : Tuple[Any, ...]
                Original positional arguments.

            Returns
            -------
            split : Tuple[Any, ...]
                Real parts followed by imaginary parts.
            """
            split = tuple(
                jnp.real(arg) if jnp.iscomplexobj(arg) else arg for arg in args
            ) + tuple(
                jnp.imag(arg) if jnp.iscomplexobj(arg) else jnp.zeros_like(arg)
                for arg in args
            )
            result: Tuple[Any, ...] = split
            return result

        def combine_complex(
            r: Tuple[Any, ...], i: Tuple[Any, ...]
        ) -> Tuple[Any, ...]:
            """Recombine real and imaginary tuples.

            Parameters
            ----------
            r : Tuple[Any, ...]
                Real parts of each argument.
            i : Tuple[Any, ...]
                Imaginary parts of each argument.

            Returns
            -------
            combined : Tuple[Any, ...]
                Complex (or real) arguments.
            """
            combined = tuple(
                rr + 1j * ii if jnp.iscomplexobj(arg) else rr
                for rr, ii, arg in zip(r, i, args, strict=False)
            )
            result: Tuple[Any, ...] = combined
            return result

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
            real_val = jnp.real(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )
            result: Float[Array, " ..."] = real_val
            return result

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
            imag_val = jnp.imag(
                func2diff(*combine_complex(split_args[:n], split_args[n:]))
            )
            result: Float[Array, " ..."] = imag_val
            return result

        gr = jax.grad(f_real, argnums=argnums)(*split_args)
        gi = jax.grad(f_imag, argnums=argnums)(*split_args)

        if isinstance(argnums, int):
            wirt_grad = 0.5 * (gr - 1j * gi)
            result: Union[
                Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]
            ] = wirt_grad
            return result
        wirt_grad = tuple(
            0.5 * (grr - 1j * gii) for grr, gii in zip(gr, gi, strict=False)
        )
        result: Union[
            Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]
        ] = wirt_grad
        return result

    gradient_function: Callable[
        ..., Union[Complex[Array, " ..."], Tuple[Complex[Array, " ..."], ...]]
    ] = grad_f
    return gradient_function


@jaxtyped(typechecker=beartype)
def complex_adam(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: Tuple[
        Num[Array, " ..."], Num[Array, " ..."], int | Int[Array, " "]
    ],
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[
    Complex[Array, " ..."],
    Tuple[Num[Array, " ..."], Num[Array, " ..."], int | Int[Array, " "]],
]:
    r"""Perform one step of complex-valued Adam.

    Extended Summary
    ----------------
    Applies the Adam update rule to complex-valued parameters
    using Wirtinger calculus. The bias-corrected update is:

    .. math::

        z_{t+1} = z_t
        - \frac{\eta\,\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}

    :see: :func:`~.test_complex_adam_two_step_matches_historical_oracle`

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
    state : Tuple[Num[Array, " ..."], Num[Array, " ..."], int]
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
    new_state : Tuple[Num[Array, " ..."], Num[Array, " ..."], int]
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
    new_state = (m, v, t)
    optimizer_step: Tuple[
        Complex[Array, " ..."],
        Tuple[Num[Array, " ..."], Num[Array, " ..."], int | Int[Array, " "]],
    ] = (new_params, new_state)
    return optimizer_step


@jaxtyped(typechecker=beartype)
def complex_adagrad(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: Num[Array, " ..."],
    learning_rate: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], Num[Array, " ..."]]:
    r"""Perform one step of complex-valued Adagrad.

    Extended Summary
    ----------------
    Applies the Adagrad update rule to complex-valued parameters
    using Wirtinger calculus:

    .. math::

        z_{t+1} = z_t
        - \frac{\eta}{\sqrt{G_t} + \varepsilon}\,g_t

    where :math:`G_t = G_{t-1} + |g_t|^2`.

    :see: :mod:`~.test_optimizers`

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
    state : Num[Array, " ..."]
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
    new_state : Num[Array, " ..."]
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

    new_state = new_accumulated_grads
    optimizer_step: Tuple[Complex[Array, " ..."], Num[Array, " ..."]] = (
        new_params,
        new_state,
    )
    return optimizer_step


@jaxtyped(typechecker=beartype)
def complex_rmsprop(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: Num[Array, " ..."],
    learning_rate: float = 0.001,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], Num[Array, " ..."]]:
    r"""Perform one step of complex-valued RMSprop.

    Extended Summary
    ----------------
    Applies the RMSprop update rule to complex-valued parameters
    using Wirtinger calculus:

    .. math::

        v_t = \rho\,v_{t-1} + (1 - \rho)\,|g_t|^2

        z_{t+1} = z_t
        - \frac{\eta}{\sqrt{v_t} + \varepsilon}\,g_t

    :see: :mod:`~.test_optimizers`

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
    state : Num[Array, " ..."]
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
    new_state : Num[Array, " ..."]
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

    new_state = new_moving_avg
    optimizer_step: Tuple[Complex[Array, " ..."], Num[Array, " ..."]] = (
        new_params,
        new_state,
    )
    return optimizer_step


def _init_optimizer_state(shape: Tuple[int, ...]) -> OptimizerState:
    """PRIVATE: Initialise optimizer state with zero moments and step.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of both optimizer moment arrays.

    Returns
    -------
    result : OptimizerState
        Optimizer state with zero moments and a zero step count.
    """
    state = OptimizerState(
        m=jnp.zeros(shape), v=jnp.zeros(shape), step=jnp.array(0)
    )
    result: OptimizerState = state
    return result


@jaxtyped(typechecker=beartype)
def init_adam(shape: Tuple[int, ...]) -> OptimizerState:
    """Initialise Adam optimizer state.

    :see: :mod:`~.test_optimizers`

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the parameters to be optimised.

    Returns
    -------
    state : OptimizerState
        State with zero first and second moments and
        ``step=0``.
    """
    state: OptimizerState = _init_optimizer_state(shape)
    return state


@jaxtyped(typechecker=beartype)
def init_adagrad(shape: Tuple[int, ...]) -> OptimizerState:
    """Initialise Adagrad optimizer state.

    :see: :mod:`~.test_optimizers`

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the parameters to be optimised.

    Returns
    -------
    state : OptimizerState
        State with zero accumulated gradients and ``step=0``.
    """
    state: OptimizerState = _init_optimizer_state(shape)
    return state


@jaxtyped(typechecker=beartype)
def init_rmsprop(shape: Tuple[int, ...]) -> OptimizerState:
    """Initialise RMSprop optimizer state.

    :see: :mod:`~.test_optimizers`

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the parameters to be optimised.

    Returns
    -------
    state : OptimizerState
        State with zero moving average and ``step=0``.
    """
    state: OptimizerState = _init_optimizer_state(shape)
    return state


@jaxtyped(typechecker=beartype)
def adam_update(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: OptimizerState,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], OptimizerState]:
    """Update parameters using Adam with Wirtinger derivatives.

    :see: :mod:`~.test_optimizers`

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
    new_state = OptimizerState(m=new_m, v=new_v, step=new_step)
    optimizer_step: Tuple[Complex[Array, " ..."], OptimizerState] = (
        new_params,
        new_state,
    )
    return optimizer_step


@jaxtyped(typechecker=beartype)
def adagrad_update(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: OptimizerState,
    learning_rate: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], OptimizerState]:
    """Update parameters using Adagrad with Wirtinger derivatives.

    :see: :mod:`~.test_optimizers`

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
    new_state = OptimizerState(m=m, v=new_v, step=step + 1)
    optimizer_step: Tuple[Complex[Array, " ..."], OptimizerState] = (
        new_params,
        new_state,
    )
    return optimizer_step


@jaxtyped(typechecker=beartype)
def rmsprop_update(
    params: Complex[Array, " ..."],
    grads: Complex[Array, " ..."],
    state: OptimizerState,
    learning_rate: float = 0.001,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
) -> Tuple[Complex[Array, " ..."], OptimizerState]:
    """Update parameters using RMSprop with Wirtinger derivatives.

    :see: :mod:`~.test_optimizers`

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
    new_state = OptimizerState(m=m, v=new_v, step=step + 1)
    optimizer_step: Tuple[Complex[Array, " ..."], OptimizerState] = (
        new_params,
        new_state,
    )
    return optimizer_step


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
