from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

import ptyrodactyl.electrons as pte
import ptyrodactyl.tools as ptt

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

    # Initialize optimizer states
    pot_slice_state = (
        jnp.zeros_like(initial_pot_slice),
        jnp.zeros_like(initial_pot_slice),
        jnp.array(0),
    )
    beam_state = (jnp.zeros_like(initial_beam), jnp.zeros_like(initial_beam), jnp.array(0))

    # Optimization loop
    pot_slice = initial_pot_slice
    beam = initial_beam

    @jax.jit
    def update_step(pot_slice, beam, pot_slice_state, beam_state):
        loss, grads = loss_and_grad(pot_slice, beam)
        pot_slice, pot_slice_state = ptt.complex_adam(
            pot_slice, grads["pot_slice"], pot_slice_state, learning_rate
        )
        beam, beam_state = ptt.complex_adam(
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