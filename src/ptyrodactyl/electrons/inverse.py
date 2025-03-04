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
    initial_potential: pte.CalibratedArray,
    initial_beam: pte.CalibratedArray,
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
    pte.CalibratedArray,
    pte.CalibratedArray,
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    """
    Description
    -----------
    Single Slice Ptychography where the electrostatic potential
    slice and the beam guess are of the same size.

    Parameters
    ----------
    - `experimental_4dstem` (Float[Array, "P H W"]):
        Experimental 4D-STEM data.
    - `initial_potential` (pte.CalibratedArray):
        Initial guess for potential slice.
    - `initial_beam` (pte.CalibratedArray):
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
    - `pot_slice` (pte.CalibratedArray):
        Optimized potential slice.
    - `beam` (pte.CalibratedArray):
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
    pot_slice_state = optimizer.init(initial_potential.data_array.shape)
    beam_state = optimizer.init(initial_beam.data_array.shape)

    pot_slice: Complex[Array, "H W"] = initial_potential.data_array
    beam: Complex[Array, "H W"]
    if initial_beam.real_space:
        beam = initial_beam.data_array
    else:
        beam = jnp.fft.ifft2(initial_beam.data_array)

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
            pot_slice.shape[0],
            pot_slice.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=pot_slice.dtype,
    )
    intermediate_beam = jnp.zeros(
        shape=(
            beam.shape[0],
            beam.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=beam.dtype,
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

    final_potential = pte.CalibratedArray(
        data_array=pot_slice,
        calib_y=initial_potential.calib_y,
        calib_x=initial_potential.calib_x,
        real_space=True,
    )
    final_beam = pte.CalibratedArray(
        data_array=beam,
        calib_y=initial_beam.calib_y,
        calib_x=initial_beam.calib_x,
        real_space=True,
    )

    return (final_potential, final_beam, intermediate_potslice, intermediate_beam)


@jaxtyped(typechecker=typechecker)
def single_slice_poscorrected(
    experimental_4dstem: Float[Array, "P H W"],
    initial_potential: pte.CalibratedArray,
    initial_beam: pte.CalibratedArray,
    initial_pos_list: Float[Array, "P 2"],
    slice_thickness: scalar_numeric,
    voltage_kV: scalar_numeric,
    calib_ang: scalar_float,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[Union[scalar_float, Float[Array, "2"]]] = 0.01,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    pte.CalibratedArray,
    pte.CalibratedArray,
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
    Float[Array, "P 2 S"],
]:
    """
    Description
    -----------
    Create and run an optimization routine for 4D-STEM reconstruction with position correction.

    Parameters
    ----------
    - `experimental_4dstem` (Float[Array, "P H W"]):
        Experimental 4D-STEM data.
    - `initial_pot_slice` (pte.CalibratedArray):
        Initial guess for potential slice.
    - `initial_beam` (pte.CalibratedArray):
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
    - `learning_rate` (Optional[Union[scalar_float, Float[Array, "2"]]]):
        Learning rate for potential slice and beam optimization.
        If the learning rate is a scalar, it is used for both
        potential slice and position optimization. If it is an array,
        the first element is used for potential slice and beam optimization,
        and the second element is used for position optimization.
        Optional, default is 0.01.
    - `loss_type` (str):
        Type of loss function to use.
        Optional, default is "mse".
    - `optimizer_name` (str):
        Name of optimizer to use.
        Optional, default is "adam".

    Returns
    -------
    - `final_potential` (pte.CalibratedArray):
        Optimized potential slice.
    - `final_beam` (pte.CalibratedArray):
        Optimized electron beam.
    - `pos_guess` (Float[Array, "P 2"]):
        Optimized list of probe positions.
    - `intermediate_potslices` (Complex[Array, "H W S"]):
        Intermediate potential slices.
    - `intermediate_beams` (Complex[Array, "H W S"]):
        Intermediate electron beams.
    - `intermediate_positions` (Float[Array, "P 2 S"]):
        Intermediate probe positions.
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
    pot_slice_state = optimizer.init(initial_potential.data_array.shape)
    beam_state = optimizer.init(initial_beam.data_array.shape)
    pos_state = optimizer.init(initial_pos_list.shape)

    learning_rate = jnp.array(learning_rate)

    if len(learning_rate) == 1:
        learning_rate = jnp.array([learning_rate, learning_rate])

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
            pos_list, grads["pos_list"], pos_state, learning_rate[1]
        )
        return pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss

    pot_guess = initial_potential.data_array
    beam_guess = initial_beam.data_array
    pos_guess = initial_pos_list

    intermediate_potslices = jnp.zeros(
        shape=(
            pot_guess.shape[0],
            pot_guess.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=pot_guess.dtype,
    )
    intermediate_beams = jnp.zeros(
        shape=(
            beam_guess.shape[0],
            beam_guess.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_beam.dtype,
    )
    intermediate_positions = jnp.zeros(
        shape=(
            pos_guess.shape[0],
            pos_guess.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=pos_guess.dtype,
    )

    for ii in range(num_iterations):
        (
            pot_guess,
            beam_guess,
            pos_guess,
            pot_slice_state,
            beam_state,
            pos_state,
            loss,
        ) = update_step(
            pot_guess, beam_guess, pos_guess, pot_slice_state, beam_state, pos_state
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: scalar_int = jnp.floor(ii / save_every)
            intermediate_potslices.at[:, :, saver].set(pot_guess)
            intermediate_beams.at[:, :, saver].set(beam_guess)
            intermediate_positions.at[:, :, saver].set(pos_guess)

    final_potential = pte.CalibratedArray(
        data_array=pot_guess,
        calib_y=initial_potential.calib_y,
        calib_x=initial_potential.calib_x,
        real_space=True,
    )
    final_beam = pte.CalibratedArray(
        data_array=beam_guess,
        calib_y=initial_beam.calib_y,
        calib_x=initial_beam.calib_x,
        real_space=True,
    )
    return (
        final_potential,
        final_beam,
        pos_guess,
        intermediate_potslices,
        intermediate_beams,
        intermediate_positions,
    )


@jaxtyped(typechecker=typechecker)
def single_slice_multi_modal(
    experimental_4dstem: Float[Array, "H W"],
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: pte.ProbeModes,
    initial_pos_list: Float[Array, "P 2"],
    slice_thickness: scalar_numeric,
    voltage_kV: scalar_numeric,
    calib_ang: scalar_float,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[Union[scalar_float, Float[Array, "2"]]] = 0.01,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    Complex[Array, "H W"],
    pte.ProbeModes,
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
    - `learning_rate` (Optional[Union[scalar_float, Float[Array, "2"]]]):
        Learning rate for potential slice and beam optimization.
        If the learning rate is a scalar, it is used for both
        potential slice and position optimization. If it is an array,
        the first element is used for potential slice and beam optimization,
        and the second element is used for position optimization.
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
    - `beam` (pte.ProbeModes):
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
            pos_list, grads["pos_list"], pos_state, learning_rate[1]
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
def multi_slice_multi_modal(
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
