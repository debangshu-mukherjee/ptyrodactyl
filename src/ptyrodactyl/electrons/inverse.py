from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

import ptyrodactyl.electrons as pte
import ptyrodactyl.tools as ptt

OPTIMIZERS: Dict[str, ptt.Optimizer] = {
    "adam": ptt.Optimizer(ptt.init_adam, ptt.adam_update),
    "adagrad": ptt.Optimizer(ptt.init_adagrad, ptt.adagrad_update),
    "rmsprop": ptt.Optimizer(ptt.init_rmsprop, ptt.rmsprop_update),
}


def get_optimizer(optimizer_name: str) -> ptt.Optimizer:
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return OPTIMIZERS[optimizer_name]


def single_slice_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: Complex[Array, "H W"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
    devices: jax.Array,
    num_iterations: int = 1000,
    learning_rate: float = 0.001,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
) -> Tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
    """
    Create and run an optimization routine for 4D-STEM reconstruction.

    Args:
    - experimental_4dstem (Float[Array, "P H W"]):
        Experimental 4D-STEM data.
    - initial_pot_slice (Complex[Array, "H W"]):
        Initial guess for potential slice.
    - initial_beam (Complex[Array, "H W"]):
        Initial guess for electron beam.
    - pos_list (Float[Array, "P 2"]):
        List of probe positions.
    - slice_thickness (Float[Array, "*"]):
        Thickness of each slice.
    - voltage_kV (Float[Array, "*"]):
        Accelerating voltage.
    - calib_ang (Float[Array, "*"]):
        Calibration in angstroms.
    - devices (jax.Array):
        Array of devices for sharding.
    - num_iterations (int):
        Number of optimization iterations.
    - learning_rate (float):
        Learning rate for optimization.
    - loss_type (str):
        Type of loss function to use.

    Returns:
    - Tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
        Optimized potential slice and beam.
    """

    # Create the forward function
    def forward_fn(pot_slice, beam):
        return pte.stem_4d(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
            devices,
        )

    # Create the loss function
    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    # Create a function that returns both loss and gradients
    @jax.jit
    def loss_and_grad(
        pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
    ) -> Tuple[Float[Array, ""], Dict[str, Complex[Array, "H W"]]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1))(pot_slice, beam)
        return loss, {"pot_slice": grads[0], "beam": grads[1]}

    optimizer = get_optimizer(optimizer_name)
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

    for i in range(num_iterations):
        pot_slice, beam, pot_slice_state, beam_state, loss = update_step(
            pot_slice, beam, pot_slice_state, beam_state
        )

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    return pot_slice, beam


def single_slice_poscorrected(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: Complex[Array, "H W"],
    initial_pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
    devices: jax.Array,
    num_iterations: int = 1000,
    learning_rate: float = 0.001,
    pos_learning_rate: float = 0.1,  # Separate learning rate for positions
    loss_type: str = "mse",
    optimizer_name: str = "adam",
) -> Tuple[Complex[Array, "H W"], Complex[Array, "H W"], Float[Array, "P 2"]]:
    """
    Create and run an optimization routine for 4D-STEM reconstruction with position correction.

    Args:
    - `experimental_4dstem` (Float[Array, "P H W"]):
        Experimental 4D-STEM data.
    - `initial_pot_slice` (Complex[Array, "H W"]):
        Initial guess for potential slice.
    - `initial_beam` (Complex[Array, "H W"]):
        Initial guess for electron beam.
    - `initial_pos_list` (Float[Array, "P 2"]):
        Initial list of probe positions.
    - `slice_thickness` (Float[Array, "*"]):
        Thickness of each slice.
    - `voltage_kV` (Float[Array, "*"]):
        Accelerating voltage.
    - `calib_ang` (Float[Array, "*"]):
        Calibration in angstroms.
    - `devices` (jax.Array):
        Array of devices for sharding.
    - `num_iterations` (int):
        Number of optimization iterations.
    - `learning_rate` (float):
        Learning rate for potential slice and beam optimization.
    - `pos_learning_rate` (float):
        Learning rate for position optimization.
    - `loss_type` (str):
        Type of loss function to use.

    Returns:
    - Tuple[Complex[Array, "H W"], Complex[Array, "H W"], Float[Array, "P 2"]]:
        Optimized potential slice, beam, and corrected positions.
    """

    # Create the forward function
    def forward_fn(pot_slice, beam, pos_list):
        return pte.stem_4d(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
            devices,
        )

    # Create the loss function
    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    # Create a function that returns both loss and gradients
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

    # ... [rest of the function remains the same, just update the optimizer calls] ...

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

    for i in range(num_iterations):
        pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss = (
            update_step(
                pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
            )
        )

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    return pot_slice, beam, pos_list
