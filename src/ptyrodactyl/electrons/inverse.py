from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jaxtyping import Complex, Float, Int, Shaped

import ptyrodactyl.electrons as pte


def loss_fn(
    params,
    beam,
    pos_list,
    slice_thickness,
    voltage_kV,
    calib_ang,
    devices,
    experimental_4dstem,
):
    """
    Compute the loss between the calculated 4D-STEM pattern and the experimental data.

    Args:
    - params: The parameters to optimize (e.g., potential slices)
    - beam, pos_list, slice_thickness, voltage_kV, calib_ang, devices: Same as in stem_4d
    - experimental_4dstem: The experimental 4D-STEM data to match

    Returns:
    - loss: The computed loss
    """
    # Unpack parameters if necessary
    pot_slice = params  # Assuming params directly represents the potential slices

    # Calculate 4D-STEM pattern
    calc_4dstem = pte.stem_4d(
        pot_slice, beam, pos_list, slice_thickness, voltage_kV, calib_ang, devices
    )

    # Compute loss
    loss = jnp.sum(jnp.abs(calc_4dstem - experimental_4dstem))

    return loss


def optimize_potential(
    initial_params,
    beam,
    pos_list,
    slice_thickness,
    voltage_kV,
    calib_ang,
    devices,
    experimental_4dstem,
    num_iterations=1000,
):
    """
    Optimize the potential slices to match the experimental 4D-STEM data.

    Args:
    - initial_params: Initial guess for the potential slices
    - Other args: Same as in loss_fn
    - num_iterations: Number of optimization iterations

    Returns:
    - optimized_params: The optimized potential slices
    """
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(initial_params)

    # Optimization loop
    params = initial_params
    for i in range(num_iterations):
        loss, grads = grad_fn(
            params,
            beam,
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
            devices,
            experimental_4dstem,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    return params
