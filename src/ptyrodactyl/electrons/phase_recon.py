"""
Module: ptyrodactyl.electrons.phase_recon
------------------------------------------
Inverse reconstruction algorithms for electron ptychography.

This module contains functions for reconstructing sample potentials from
experimental ptychography data using various optimization algorithms.
All functions support both single-slice and multi-slice reconstructions,
with options for position correction and multi-modal probe handling.

Functions
---------
- `single_slice_ptychography`:
    Performs single-slice ptychography reconstruction
- `single_slice_poscorrected`:
    Performs single-slice reconstruction with position correction
single_slice_multi_modal
    Performs single-slice reconstruction with multi-modal probe
multi_slice_multi_modal
    Performs multi-slice reconstruction with multi-modal probe

Notes
-----
All reconstruction functions use JAX-compatible optimizers and support
automatic differentiation. The functions are designed to work with
experimental data and can handle various noise levels and experimental
conditions. Input data should be properly preprocessed and validated
using the factory functions from electron_types module.

Internal functions (not exported):
- _get_optimizer: Returns an optimizer instance for the specified optimization algorithm
"""

import jax
import jax.numpy as jnp
from beartype.typing import Any, Dict, Optional, Tuple, Union
from jaxtyping import Array, Complex, Float, Int

import ptyrodactyl.tools as ptt
from ptyrodactyl._decorators import beartype as typechecker
from ptyrodactyl._decorators import jaxtyped

from .electron_types import (
    STEM4D,
    CalibratedArray,
    ProbeModes,
    make_calibrated_array,
    scalar_float,
    scalar_int,
    scalar_numeric,
)
from .simulations import stem_4d

jax.config.update("jax_enable_x64", True)

OPTIMIZERS: Dict[str, ptt.Optimizer] = {
    "adam": ptt.Optimizer(ptt.init_adam, ptt.adam_update),
    "adagrad": ptt.Optimizer(ptt.init_adagrad, ptt.adagrad_update),
    "rmsprop": ptt.Optimizer(ptt.init_rmsprop, ptt.rmsprop_update),
}


@typechecker
def _get_optimizer(optimizer_name: str) -> ptt.Optimizer:
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return OPTIMIZERS[optimizer_name]


@jaxtyped(typechecker=typechecker)
def single_slice_ptychography(
    experimental_data: STEM4D,
    initial_potential: CalibratedArray,
    initial_beam: CalibratedArray,
    slice_thickness: scalar_numeric,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[scalar_float] = 0.001,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    CalibratedArray,
    CalibratedArray,
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    """Single Slice Ptychography where the electrostatic potential slice and the beam guess are of the same size.

    Parameters
    ----------
    experimental_data : STEM4D
        Experimental 4D-STEM data PyTree containing diffraction patterns,
        scan positions, and calibration information.
    initial_potential : CalibratedArray
        Initial guess for potential slice.
    initial_beam : CalibratedArray
        Initial guess for electron beam.
    slice_thickness : scalar_numeric
        Thickness of each slice.
    save_every : scalar_int, optional
        Save every nth iteration. Default is 10.
    num_iterations : scalar_int, optional
        Number of optimization iterations. Default is 1000.
    learning_rate : scalar_float, optional
        Learning rate for optimization. Default is 0.001.
    loss_type : str, optional
        Type of loss function to use. Default is "mse".
    optimizer_name : str, optional
        Name of optimizer to use. Default is "adam".

    Returns
    -------
    tuple of (CalibratedArray, CalibratedArray, Complex[Array, "H W S"], Complex[Array, "H W S"])
        - pot_slice : CalibratedArray
            Optimized potential slice.
        - beam : CalibratedArray
            Optimized electron beam.
        - intermediate_potslice : Complex[Array, "H W S"]
            Intermediate potential slices.
        - intermediate_beam : Complex[Array, "H W S"]
            Intermediate electron beams.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    pos_list: Float[Array, "P 2"] = experimental_data.scan_positions
    voltage_kV: Float[Array, " "] = experimental_data.voltage_kV
    calib_ang: Float[Array, " "] = experimental_data.real_space_calib

    def _forward_fn(
        pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
    ) -> Float[Array, "P H W"]:
        stem4d_result = stem_4d(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )
        return stem4d_result.data

    loss_func: Any = ptt.create_loss_function(
        _forward_fn, experimental_4dstem, loss_type
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
    ) -> Tuple[Float[Array, " "], Dict[str, Complex[Array, "H W"]]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1))(pot_slice, beam)
        return loss, {"pot_slice": grads[0], "beam": grads[1]}

    optimizer: ptt.Optimizer = _get_optimizer(optimizer_name)
    pot_slice_state: Any = optimizer.init(initial_potential.data_array.shape)
    beam_state: Any = optimizer.init(initial_beam.data_array.shape)

    pot_slice: Complex[Array, "H W"] = initial_potential.data_array
    beam: Complex[Array, "H W"]
    if initial_beam.real_space:
        beam = initial_beam.data_array
    else:
        beam = jnp.fft.ifft2(initial_beam.data_array)

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pot_slice_state: Any,
        beam_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"], Complex[Array, "H W"], Any, Any, Float[Array, " "]
    ]:
        loss: Float[Array, " "]
        grads: Dict[str, Complex[Array, "H W"]]
        loss, grads = _loss_and_grad(pot_slice, beam)
        pot_slice, pot_slice_state = optimizer.update(
            pot_slice, grads["pot_slice"], pot_slice_state, learning_rate
        )
        beam, beam_state = optimizer.update(
            beam, grads["beam"], beam_state, learning_rate
        )
        return pot_slice, beam, pot_slice_state, beam_state, loss

    intermediate_potslice: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            pot_slice.shape[0],
            pot_slice.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=pot_slice.dtype,
    )
    intermediate_beam: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            beam.shape[0],
            beam.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=beam.dtype,
    )

    for ii in range(num_iterations):
        loss: Float[Array, " "]
        pot_slice, beam, pot_slice_state, beam_state, loss = _update_step(
            pot_slice, beam, pot_slice_state, beam_state
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: Int[Array, ""] = jnp.floor(ii / save_every).astype(jnp.int32)
            intermediate_potslice = intermediate_potslice.at[:, :, saver].set(pot_slice)
            intermediate_beam = intermediate_beam.at[:, :, saver].set(beam)

    final_potential: CalibratedArray = make_calibrated_array(
        data_array=pot_slice,
        calib_y=initial_potential.calib_y,
        calib_x=initial_potential.calib_x,
        real_space=True,
    )
    final_beam: CalibratedArray = make_calibrated_array(
        data_array=beam,
        calib_y=initial_beam.calib_y,
        calib_x=initial_beam.calib_x,
        real_space=True,
    )

    return (final_potential, final_beam, intermediate_potslice, intermediate_beam)


@jaxtyped(typechecker=typechecker)
def single_slice_poscorrected(
    experimental_data: STEM4D,
    initial_potential: CalibratedArray,
    initial_beam: CalibratedArray,
    slice_thickness: scalar_numeric,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[Union[scalar_float, Float[Array, "2"]]] = 0.01,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    CalibratedArray,
    CalibratedArray,
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
    Float[Array, "P 2 S"],
]:
    """Create and run an optimization routine for 4D-STEM reconstruction with position correction.

    Parameters
    ----------
    experimental_data : STEM4D
        Experimental 4D-STEM data PyTree containing diffraction patterns,
        scan positions, and calibration information.
    initial_potential : CalibratedArray
        Initial guess for potential slice.
    initial_beam : CalibratedArray
        Initial guess for electron beam.
    slice_thickness : scalar_numeric
        Thickness of each slice.
    save_every : scalar_int, optional
        Save every nth iteration. Default is 10.
    num_iterations : scalar_int, optional
        Number of optimization iterations. Default is 1000.
    learning_rate : scalar_float or Float[Array, "2"], optional
        Learning rate for potential slice and beam optimization.
        If the learning rate is a scalar, it is used for both
        potential slice and position optimization. If it is an array,
        the first element is used for potential slice and beam optimization,
        and the second element is used for position optimization.
        Default is 0.01.
    loss_type : str, optional
        Type of loss function to use. Default is "mse".
    optimizer_name : str, optional
        Name of optimizer to use. Default is "adam".

    Returns
    -------
    tuple of (CalibratedArray, CalibratedArray, Float[Array, "P 2"], Complex[Array, "H W S"], Complex[Array, "H W S"], Float[Array, "P 2 S"])
        - final_potential : CalibratedArray
            Optimized potential slice.
        - final_beam : CalibratedArray
            Optimized electron beam.
        - pos_guess : Float[Array, "P 2"]
            Optimized list of probe positions.
        - intermediate_potslices : Complex[Array, "H W S"]
            Intermediate potential slices.
        - intermediate_beams : Complex[Array, "H W S"]
            Intermediate electron beams.
        - intermediate_positions : Float[Array, "P 2 S"]
            Intermediate probe positions.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    voltage_kV: Float[Array, " "] = experimental_data.voltage_kV
    calib_ang: Float[Array, " "] = experimental_data.real_space_calib
    initial_pos_list: Float[Array, "P 2"] = experimental_data.scan_positions

    def _forward_fn(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Float[Array, "P H W"]:
        stem4d_result = stem_4d(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )
        return stem4d_result.data

    loss_func: Any = ptt.create_loss_function(
        _forward_fn, experimental_4dstem, loss_type
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Array]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        return loss, {"pot_slice": grads[0], "beam": grads[1], "pos_list": grads[2]}

    optimizer: ptt.Optimizer = _get_optimizer(optimizer_name)
    pot_slice_state: Any = optimizer.init(initial_potential.data_array.shape)
    beam_state: Any = optimizer.init(initial_beam.data_array.shape)
    pos_state: Any = optimizer.init(initial_pos_list.shape)

    learning_rate: Float[Array, ...] = jnp.array(learning_rate)

    if len(learning_rate) == 1:
        learning_rate = jnp.array([learning_rate, learning_rate])

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
        pot_slice_state: Any,
        beam_state: Any,
        pos_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        Complex[Array, "H W"],
        Float[Array, "P 2"],
        Any,
        Any,
        Any,
        Float[Array, " "],
    ]:
        loss: Float[Array, " "]
        grads: Dict[str, Array]
        loss, grads = _loss_and_grad(pot_slice, beam, pos_list)
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

    pot_guess: Complex[Array, "H W"] = initial_potential.data_array
    beam_guess: Complex[Array, "H W"] = initial_beam.data_array
    pos_guess: Float[Array, "P 2"] = initial_pos_list

    intermediate_potslices: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            pot_guess.shape[0],
            pot_guess.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=pot_guess.dtype,
    )
    intermediate_beams: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            beam_guess.shape[0],
            beam_guess.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_beam.dtype,
    )
    intermediate_positions: Float[Array, "P 2 S"] = jnp.zeros(
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
        ) = _update_step(
            pot_guess, beam_guess, pos_guess, pot_slice_state, beam_state, pos_state
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: Int[Array, ""] = jnp.floor(ii / save_every).astype(jnp.int32)
            intermediate_potslices = intermediate_potslices.at[:, :, saver].set(
                pot_guess
            )
            intermediate_beams = intermediate_beams.at[:, :, saver].set(beam_guess)
            intermediate_positions = intermediate_positions.at[:, :, saver].set(
                pos_guess
            )

    final_potential: CalibratedArray = make_calibrated_array(
        data_array=pot_guess,
        calib_y=initial_potential.calib_y,
        calib_x=initial_potential.calib_x,
        real_space=True,
    )
    final_beam: CalibratedArray = make_calibrated_array(
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
    experimental_data: STEM4D,
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: ProbeModes,
    slice_thickness: scalar_numeric,
    save_every: Optional[scalar_int] = 10,
    num_iterations: Optional[scalar_int] = 1000,
    learning_rate: Optional[Union[scalar_float, Float[Array, "2"]]] = 0.01,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
) -> Tuple[
    Complex[Array, "H W"],
    ProbeModes,
    Float[Array, "P 2"],
    Complex[Array, "H W S"],
    Complex[Array, "H W S"],
]:
    """Create and run an optimization routine for 4D-STEM reconstruction with multi-modal probe.

    Parameters
    ----------
    experimental_data : STEM4D
        Experimental 4D-STEM data PyTree containing diffraction patterns,
        scan positions, and calibration information.
    initial_pot_slice : Complex[Array, "H W"]
        Initial guess for potential slice.
    initial_beam : ProbeModes
        Initial guess for electron beam with multiple probe modes.
    slice_thickness : scalar_numeric
        Thickness of each slice.
    save_every : scalar_int, optional
        Save every nth iteration. Default is 10.
    num_iterations : scalar_int, optional
        Number of optimization iterations. Default is 1000.
    learning_rate : scalar_float or Float[Array, "2"], optional
        Learning rate for potential slice and beam optimization.
        If the learning rate is a scalar, it is used for both
        potential slice and position optimization. If it is an array,
        the first element is used for potential slice and beam optimization,
        and the second element is used for position optimization.
        Default is 0.01.
    loss_type : str, optional
        Type of loss function to use. Default is "mse".
    optimizer_name : str, optional
        Name of optimizer to use. Default is "adam".

    Returns
    -------
    tuple of (Complex[Array, "H W"], ProbeModes, Float[Array, "P 2"], Complex[Array, "H W S"], Complex[Array, "H W S"])
        - pot_slice : Complex[Array, "H W"]
            Optimized potential slice.
        - beam : ProbeModes
            Optimized electron beam.
        - pos_list : Float[Array, "P 2"]
            Optimized list of probe positions.
        - intermediate_potslice : Complex[Array, "H W S"]
            Intermediate potential slices.
        - intermediate_beam : Complex[Array, "H W S"]
            Intermediate electron beams.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    voltage_kV: Float[Array, " "] = experimental_data.voltage_kV
    calib_ang: Float[Array, " "] = experimental_data.real_space_calib
    initial_pos_list: Float[Array, "P 2"] = experimental_data.scan_positions

    def _forward_fn(
        pot_slice: Complex[Array, "H W"],
        beam: ProbeModes,
        pos_list: Float[Array, "P 2"],
    ) -> Float[Array, "P H W"]:
        stem4d_result = stem_4d(
            pot_slice[None, ...],
            beam,
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )
        return stem4d_result.data

    loss_func: Any = ptt.create_loss_function(
        _forward_fn, experimental_4dstem, loss_type
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: ProbeModes,
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Any]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        return loss, {"pot_slice": grads[0], "beam": grads[1], "pos_list": grads[2]}

    optimizer: ptt.Optimizer = _get_optimizer(optimizer_name)
    pot_slice_state: Any = optimizer.init(initial_pot_slice.shape)
    beam_state: Any = optimizer.init(initial_beam.modes.shape)
    pos_state: Any = optimizer.init(initial_pos_list.shape)

    learning_rate: Float[Array, ...] = jnp.array(learning_rate)
    if len(learning_rate.shape) == 0:
        learning_rate = jnp.array([learning_rate, learning_rate])

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: ProbeModes,
        pos_list: Float[Array, "P 2"],
        pot_slice_state: Any,
        beam_state: Any,
        pos_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        ProbeModes,
        Float[Array, "P 2"],
        Any,
        Any,
        Any,
        Float[Array, " "],
    ]:
        loss: Float[Array, " "]
        grads: Dict[str, Any]
        loss, grads = _loss_and_grad(pot_slice, beam, pos_list)
        pot_slice, pot_slice_state = optimizer.update(
            pot_slice, grads["pot_slice"], pot_slice_state, learning_rate[0]
        )
        beam_modes: Complex[Array, "H W M"]
        beam_modes, beam_state = optimizer.update(
            beam.modes, grads["beam"].modes, beam_state, learning_rate[0]
        )
        beam = ProbeModes(modes=beam_modes, weights=beam.weights, calib=beam.calib)
        pos_list, pos_state = optimizer.update(
            pos_list, grads["pos_list"], pos_state, learning_rate[1]
        )
        return pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss

    pot_slice: Complex[Array, "H W"] = initial_pot_slice
    beam: ProbeModes = initial_beam
    pos_list: Float[Array, "P 2"] = initial_pos_list

    intermediate_potslice: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            initial_pot_slice.shape[0],
            initial_pot_slice.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_pot_slice.dtype,
    )
    intermediate_beam: Complex[Array, "H W M S"] = jnp.zeros(
        shape=(
            initial_beam.modes.shape[0],
            initial_beam.modes.shape[1],
            initial_beam.modes.shape[2],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_beam.modes.dtype,
    )

    for ii in range(num_iterations):
        loss: Float[Array, " "]
        pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss = (
            _update_step(
                pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
            )
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: Int[Array, ""] = jnp.floor(ii / save_every).astype(jnp.int32)
            intermediate_potslice = intermediate_potslice.at[:, :, saver].set(pot_slice)
            intermediate_beam = intermediate_beam.at[:, :, :, saver].set(beam.modes)

    return pot_slice, beam, pos_list, intermediate_potslice, intermediate_beam


@jaxtyped(typechecker=typechecker)
def multi_slice_multi_modal(
    experimental_data: STEM4D,
    initial_pot_slice: Complex[Array, "H W"],
    initial_beam: Complex[Array, "H W"],
    slice_thickness: scalar_numeric,
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
    """Create and run an optimization routine for multi-slice 4D-STEM reconstruction with position correction.

    Parameters
    ----------
    experimental_data : STEM4D
        Experimental 4D-STEM data PyTree containing diffraction patterns,
        scan positions, and calibration information.
    initial_pot_slice : Complex[Array, "H W"]
        Initial guess for potential slice.
    initial_beam : Complex[Array, "H W"]
        Initial guess for electron beam.
    slice_thickness : scalar_numeric
        Thickness of each slice.
    save_every : scalar_int, optional
        Save every nth iteration. Default is 10.
    num_iterations : scalar_int, optional
        Number of optimization iterations. Default is 1000.
    learning_rate : scalar_float, optional
        Learning rate for potential slice and beam optimization. Default is 0.001.
    pos_learning_rate : scalar_float, optional
        Learning rate for position optimization. Default is 0.01.
    loss_type : str, optional
        Type of loss function to use. Default is "mse".
    optimizer_name : str, optional
        Name of optimizer to use. Default is "adam".

    Returns
    -------
    tuple of (Complex[Array, "H W"], Complex[Array, "H W"], Float[Array, "P 2"], Complex[Array, "H W S"], Complex[Array, "H W S"])
        - pot_slice : Complex[Array, "H W"]
            Optimized potential slice.
        - beam : Complex[Array, "H W"]
            Optimized electron beam.
        - pos_list : Float[Array, "P 2"]
            Optimized list of probe positions.
        - intermediate_potslice : Complex[Array, "H W S"]
            Intermediate potential slices.
        - intermediate_beam : Complex[Array, "H W S"]
            Intermediate electron beams.
    """
    experimental_4dstem: Float[Array, "P H W"] = experimental_data.data
    voltage_kV: Float[Array, " "] = experimental_data.voltage_kV
    calib_ang: Float[Array, " "] = experimental_data.real_space_calib
    initial_pos_list: Float[Array, "P 2"] = experimental_data.scan_positions

    def _forward_fn(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Float[Array, "P H W"]:
        stem4d_result = stem_4d(
            pot_slice[None, ...],
            beam[None, ...],
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )
        return stem4d_result.data

    loss_func: Any = ptt.create_loss_function(
        _forward_fn, experimental_4dstem, loss_type
    )

    @jax.jit
    def _loss_and_grad(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
    ) -> Tuple[Float[Array, " "], Dict[str, Array]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slice, beam, pos_list
        )
        return loss, {"pot_slice": grads[0], "beam": grads[1], "pos_list": grads[2]}

    optimizer: ptt.Optimizer = _get_optimizer(optimizer_name)
    pot_slice_state: Any = optimizer.init(initial_pot_slice.shape)
    beam_state: Any = optimizer.init(initial_beam.shape)
    pos_state: Any = optimizer.init(initial_pos_list.shape)

    @jax.jit
    def _update_step(
        pot_slice: Complex[Array, "H W"],
        beam: Complex[Array, "H W"],
        pos_list: Float[Array, "P 2"],
        pot_slice_state: Any,
        beam_state: Any,
        pos_state: Any,
    ) -> Tuple[
        Complex[Array, "H W"],
        Complex[Array, "H W"],
        Float[Array, "P 2"],
        Any,
        Any,
        Any,
        Float[Array, " "],
    ]:
        loss: Float[Array, " "]
        grads: Dict[str, Array]
        loss, grads = _loss_and_grad(pot_slice, beam, pos_list)
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

    pot_slice: Complex[Array, "H W"] = initial_pot_slice
    beam: Complex[Array, "H W"] = initial_beam
    pos_list: Float[Array, "P 2"] = initial_pos_list

    intermediate_potslice: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            initial_pot_slice.shape[0],
            initial_pot_slice.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_pot_slice.dtype,
    )
    intermediate_beam: Complex[Array, "H W S"] = jnp.zeros(
        shape=(
            initial_beam.shape[0],
            initial_beam.shape[1],
            jnp.floor(num_iterations / save_every),
        ),
        dtype=initial_beam.dtype,
    )

    for ii in range(num_iterations):
        loss: Float[Array, " "]
        pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state, loss = (
            _update_step(
                pot_slice, beam, pos_list, pot_slice_state, beam_state, pos_state
            )
        )

        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            saver: Int[Array, ""] = jnp.floor(ii / save_every).astype(jnp.int32)
            intermediate_potslice = intermediate_potslice.at[:, :, saver].set(pot_slice)
            intermediate_beam = intermediate_beam.at[:, :, saver].set(beam)

    return pot_slice, beam, pos_list, intermediate_potslice, intermediate_beam
