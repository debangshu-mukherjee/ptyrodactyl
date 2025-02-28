import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Dict, Optional, Tuple, TypeAlias, Union
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

import ptyrodactyl.electrons as pte
import ptyrodactyl.tools as ptt

jax.config.update("jax_enable_x64", True)

scalar_numeric: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Int[Array, ""]]

OPTIMIZERS: Dict[str, ptt.Optimizer] = {
    "adam": ptt.Optimizer(ptt.init_adam, ptt.adam_update),
    "adagrad": ptt.Optimizer(ptt.init_adagrad, ptt.adagrad_update),
    "rmsprop": ptt.Optimizer(ptt.init_rmsprop, ptt.rmsprop_update),
}


def get_optimizer(optimizer_name: str) -> ptt.Optimizer:
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return OPTIMIZERS[optimizer_name]


@jaxtyped(typechecker=typechecker)
def single_slice_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: Complex[Array, "H W"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: scalar_numeric,
    voltage_kV: scalar_numeric,
    calib_ang: scalar_float,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[scalar_float] = 0.001,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    Complex[Array, "H W"],
    Complex[Array, "H W"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    """
    Description
    -----------
    Create and run an optimization routine for 4D-STEM reconstruction.

    Parameters
    ----------
    - `experimental_4dstem` (Float[Array, "P H W"]):
        Experimental 4D-STEM data.
    - `initial_pot_slice` (Complex[Array, "H W"]):
        Initial guess for potential slice.
    - `initial_beam` (Complex[Array, "H W"]):
        Initial guess for electron beam.
    - `pos_list` (Float[Array, "P 2"]):
        List of probe positions.
    - `slice_thickness` (scalar_numeric):
        Thickness of each slice.
    - `voltage_kV` (scalar_numeric):
        Accelerating voltage.
    - `calib_ang` (scalar_float):
        Calibration in angstroms.
    - `save_every` (scalar_int):
        Save every nth iteration.
        Optional, default is 10.
    - `num_iterations` (scalar_int):
        Number of optimization iterations.
        Optional, default is 1000.
    - `learning_rate` (scalar_float):
        Learning rate for optimization.
        Optional, default is 0.001.
    - `loss_type` (str):
        Type of loss function to use.
        Optional, default is "mse".
    - `optimizer_name` (str):
        Name of optimizer to use.
        Optional, default is "adam".

    Returns
    -------
    - `pot_slice` (Complex[Array, "H W"]):
        Optimized potential slice.
    - `beam` (Complex[Array, "H W"]):
        Optimized electron beam.
    - `intermediate_potslice` (Complex[Array, "H W S"]):
        Intermediate potential slices.
    - `intermediate_beam` (Complex[Array, "H W S"]):
        Intermediate electron beams.
    """

    def forward_fn(pot_slice, beam):
        return pte.stem_4D(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )

    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    @jax.jit
    def loss_and_grad(
        pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
    ) -> Tuple[Float[Array, ""], Dict[str, Complex[Array, "H W"]]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1))(pot_slice, beam)
        return loss, {"pot_slice": grads[0], "beam": grads[1]}

    optimizer: ptt.Optimizer = get_optimizer(optimizer_name)
    pot_slice_state = optimizer.init(initial_pot_slice.shape)
    beam_state = optimizer.init(initial_beam.shape)

    pot_slice = initial_pot_slice
    beam = initial_beam

    @jax.jit
    def update_step(pot_slice, beam, pot_slice_state, beam_state):
        loss, grads = loss_and_grad(pot_slice, beam)
        pot_slice, pot_slice_state = optimizer.update(
            pot_slice, grads["pot_slice"], pot_slice_state, learning_rate
        )
        beam, beam_state = optimizer.update(
            beam, grads["beam"], beam_state, learning_rate
        )
        return pot_slice, beam, pot_slice_state, beam_state, loss

    intermediate_potslice = jnp.zeros(
        shape=(
            initial_pot_slice.shape[0],
            initial_pot_slice.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_pot_slice.dtype,
    )
    intermediate_beam = jnp.zeros(
        shape=(
            initial_beam.shape[0],
            initial_beam.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_beam.dtype,
    )

    for ii in range(num_iterations):
        pot_slice, beam, pot_slice_state, beam_state, loss = update_step(
            pot_slice, beam, pot_slice_state, beam_state
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: scalar_int = jnp.floor(ii / save_every)
            intermediate_potslice.at[:, :, saver].set(pot_slice)
            intermediate_beam.at[:, :, saver].set(beam)

    return pot_slice, beam, intermediate_potslice, intermediate_beam


@jaxtyped(typechecker=typechecker)
def single_slice_poscorrected(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: Complex[Array, "H W"],
    initial_pos_list: Float[Array, "P 2"],
    slice_thickness: scalar_numeric,
    voltage_kV: scalar_numeric,
    calib_ang: scalar_float,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[scalar_float] = 0.001,
    pos_learning_rate: Optional[scalar_float] = 0.01,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    Complex[Array, "H W"],
    Complex[Array, "H W"],
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    """
    Description
    -----------
    Create and run an optimization routine for 4D-STEM reconstruction with position correction.

    Parameters
    ----------
    - `experimental_4dstem` (Float[Array, "P H W"]):
        Experimental 4D-STEM data.
    - `initial_pot_slice` (Complex[Array, "H W"]):
        Initial guess for potential slice.
    - `initial_beam` (Complex[Array, "H W"]):
        Initial guess for electron beam.
    - `initial_pos_list` (Float[Array, "P 2"]):
        Initial list of probe positions.
    - `slice_thickness` (scalar_numeric):
        Thickness of each slice.
    - `voltage_kV` (scalar_numeric):
        Accelerating voltage.
    - `calib_ang` (scalar_float):
        Calibration in angstroms.
    - `save_every` (scalar_int):
        Save every nth iteration.
        Optional, default is 10.
    - `num_iterations` (scalar_int):
        Number of optimization iterations.
        Optional, default is 1000.
    - `learning_rate` (scalar_float):
        Learning rate for potential slice and beam optimization.
        Optional, default is 0.001.
    - `pos_learning_rate` (scalar_float):
        Learning rate for position optimization.
        Optional, default is 0.01.
    - `loss_type` (str):
        Type of loss function to use.
        Optional, default is "mse".
    - `optimizer_name` (str):
        Name of optimizer to use.
        Optional, default is "adam".

    Returns
    -------
    - `pot_slice` (Complex[Array, "H W"]):
        Optimized potential slice.
    - `beam` (Complex[Array, "H W"]):
        Optimized electron beam.
    - `pos_list` (Float[Array, "P 2"]):
        Optimized list of probe positions.
    - `intermediate_potslice` (Complex[Array, "H W S"]):
        Intermediate potential slices.
    - `intermediate_beam` (Complex[Array, "H W S"]):
        Intermediate electron beams.
    """

    def forward_fn(pot_slice, beam, pos_list):
        return pte.stem_4D(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )

    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    @jax.jit
    def loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, ""], Dict[str, Array]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        return loss, {"pot_slice": grads[0], "beam": grads[1], "pos_list": grads[2]}

    optimizer = get_optimizer(optimizer_name)
    pot_slice_state = optimizer.init(initial_pot_slice.shape)
    beam_state = optimizer.init(initial_beam.shape)
    pos_state = optimizer.init(initial_pos_list.shape)

    @jax.jit
    def update_step(pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state):
        loss, grads = loss_and_grad(pot_slice, beam, pos_list)
        pot_slice, pot_slice_state = optimizer.update(
            pot_slice, grads["pot_slice"], pot_slice_state, learning_rate
        )
        beam, beam_state = optimizer.update(
            beam, grads["beam"], beam_state, learning_rate
        )
        pos_list, pos_state = optimizer.update(
            pos_list, grads["pos_list"], pos_state, pos_learning_rate
        )
        return pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss

    pot_slice = initial_pot_slice
    beam = initial_beam
    pos_list = initial_pos_list

    intermediate_potslice = jnp.zeros(
        shape=(
            initial_pot_slice.shape[0],
            initial_pot_slice.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_pot_slice.dtype,
    )
    intermediate_beam = jnp.zeros(
        shape=(
            initial_beam.shape[0],
            initial_beam.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_beam.dtype,
    )

    for ii in range(num_iterations):
        pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss = (
            update_step(
                pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
            )
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: scalar_int = jnp.floor(ii / save_every)
            intermediate_potslice.at[:, :, saver].set(pot_slice)
            intermediate_beam.at[:, :, saver].set(beam)

    return pot_slice, beam, pos_list, intermediate_potslice, intermediate_beam


@jaxtyped(typechecker=typechecker)
def multi_slice_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slices: Complex[Array, "H W S"],
    initial_beam: Complex[Array, "H W"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, ""],
    voltage_kV: Float[Array, ""],
    calib_ang: Float[Array, ""],
    num_iterations: int = 1000,
    learning_rate: float = 0.001,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
    scheduler_fn: Optional[ptt.SchedulerFn] = None,
) -> Tuple[Complex[Array, "H W S"], Complex[Array, "H W"]]:
    """
    Multi-slice ptychographic reconstruction.

    Args:
        experimental_4dstem: Experimental 4D-STEM data
        initial_pot_slices: Initial guess for potential slices
        initial_beam: Initial guess for electron beam
        pos_list: List of probe positions
        slice_thickness: Thickness of each slice
        voltage_kV: Accelerating voltage
        calib_ang: Calibration in angstroms
        num_iterations: Number of optimization iterations
        learning_rate: Initial learning rate
        loss_type: Type of loss function
        optimizer_name: Name of optimizer to use
        scheduler_fn: Optional learning rate scheduler

    Returns:
        Tuple of optimized potential slices and beam
    """

    # Create the forward function for multiple slices
    def forward_fn(pot_slices: Complex[Array, "H W S"], beam: Complex[Array, "H W"]):
        return pte.stem_4d_multi(
            pot_slices,
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )

    # Create the loss function
    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    # Get loss and gradients
    @jax.jit
    def loss_and_grad(
        pot_slices: Complex[Array, "H W S"], beam: Complex[Array, "H W"]
    ) -> Tuple[Float[Array, ""], Dict[str, Array]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1))(pot_slices, beam)
        return loss, {"pot_slices": grads[0], "beam": grads[1]}

    # Initialize optimizer
    optimizer = get_optimizer(optimizer_name)
    pot_slices_state = optimizer.init(initial_pot_slices.shape)
    beam_state = optimizer.init(initial_beam.shape)

    # Initialize scheduler if provided
    if scheduler_fn is not None:
        scheduler_state = ptt.init_scheduler_state(learning_rate)

    # Initialize variables
    pot_slices = initial_pot_slices
    beam = initial_beam
    current_lr = learning_rate

    @jax.jit
    def update_step(pot_slices, beam, pot_slices_state, beam_state, lr):
        loss, grads = loss_and_grad(pot_slices, beam)
        pot_slices, pot_slices_state = optimizer.update(
            pot_slices, grads["pot_slices"], pot_slices_state, lr
        )
        beam, beam_state = optimizer.update(beam, grads["beam"], beam_state, lr)
        return pot_slices, beam, pot_slices_state, beam_state, loss

    for i in range(num_iterations):
        # Update learning rate if scheduler is provided
        if scheduler_fn is not None:
            current_lr, scheduler_state = scheduler_fn(scheduler_state)

        # Perform optimization step
        pot_slices, beam, pot_slices_state, beam_state, loss = update_step(
            pot_slices, beam, pot_slices_state, beam_state, current_lr
        )

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}, LR: {current_lr}")

    return pot_slices, beam


def multi_mode_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slices: Complex[Array, "H W S"],
    initial_probe_state: pte.ProbeModes,
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, ""],
    voltage_kV: Float[Array, ""],
    calib_ang: Float[Array, ""],
    num_iterations: int = 1000,
    learning_rate: float = 0.001,
    weight_learning_rate: float = 0.0001,  # Separate LR for weights
    loss_type: str = "mse",
    optimizer_name: str = "adam",
    scheduler_fn: Optional[ptt.SchedulerFn] = None,
) -> Tuple[Complex[Array, "H W S"], pte.ProbeState]:
    """
    Multi-mode ptychographic reconstruction.

    Args:
        experimental_4dstem: Experimental 4D-STEM data
        initial_pot_slices: Initial guess for potential slices
        initial_probe_state: Initial probe modes and weights
        pos_list: List of probe positions
        slice_thickness: Thickness of each slice
        voltage_kV: Accelerating voltage
        calib_ang: Calibration in angstroms
        num_iterations: Number of optimization iterations
        learning_rate: Initial learning rate
        weight_learning_rate: Learning rate for mode weights
        loss_type: Type of loss function
        optimizer_name: Name of optimizer to use
        scheduler_fn: Optional learning rate scheduler

    Returns:
        Tuple of optimized potential slices and probe state
    """

    def forward_fn(
        pot_slices: Complex[Array, "H W S"],
        probe_modes: Complex[Array, "H W M"],
        mode_weights: Float[Array, "M"],
    ):
        # Calculate pattern for each mode
        patterns = pte.stem_4d_multi(
            pot_slices,
            probe_modes,
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )

        # Weight patterns by mode occupations
        weighted_sum = jnp.sum(
            patterns[..., None] * mode_weights[None, None, None, :], axis=-1
        )
        return weighted_sum

    # Create loss function
    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    @jax.jit
    def loss_and_grad(
        pot_slices: Complex[Array, "H W S"],
        probe_state: pte.ProbeState,
    ) -> Tuple[Float[Array, ""], dict]:
        loss, grads = jax.value_and_grad(
            lambda p, m, w: loss_func(p, m, w), argnums=(0, 1, 2)
        )(pot_slices, probe_state.modes, probe_state.weights)

        return loss, {"pot_slices": grads[0], "modes": grads[1], "weights": grads[2]}

    # Initialize optimizers
    optimizer = get_optimizer(optimizer_name)
    pot_slices_state = optimizer.init(initial_pot_slices.shape)
    modes_state = optimizer.init(initial_probe_state.modes.shape)
    weights_state = optimizer.init(initial_probe_state.weights.shape)

    if scheduler_fn is not None:
        scheduler_state = ptt.init_scheduler_state(learning_rate)

    pot_slices = initial_pot_slices
    probe_state = initial_probe_state
    current_lr = learning_rate

    @jax.jit
    def update_step(pot_slices, probe_state, opt_states, lr, weight_lr):
        pot_slices_state, modes_state, weights_state = opt_states
        loss, grads = loss_and_grad(pot_slices, probe_state)

        # Update potential slices and modes
        pot_slices, pot_slices_state = optimizer.update(
            pot_slices, grads["pot_slices"], pot_slices_state, lr
        )
        modes, modes_state = optimizer.update(
            probe_state.modes, grads["modes"], modes_state, lr
        )

        # Update weights with separate learning rate
        weights, weights_state = optimizer.update(
            probe_state.weights, grads["weights"], weights_state, weight_lr
        )

        # Normalize weights
        weights = jnp.abs(weights)  # Ensure positive
        weights = weights / jnp.sum(weights)  # Normalize

        new_probe_state = pte.ProbeState(modes=modes, weights=weights)
        new_opt_states = (pot_slices_state, modes_state, weights_state)

        return pot_slices, new_probe_state, new_opt_states, loss

    for i in range(num_iterations):
        if scheduler_fn is not None:
            current_lr, scheduler_state = scheduler_fn(scheduler_state)

        opt_states = (pot_slices_state, modes_state, weights_state)
        pot_slices, probe_state, opt_states, loss = update_step(
            pot_slices, probe_state, opt_states, current_lr, weight_learning_rate
        )
        pot_slices_state, modes_state, weights_state = opt_states

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}, LR: {current_lr}")
            print(f"Mode weights: {probe_state.weights}")

    return pot_slices, probe_state
