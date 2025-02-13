import jax
import jax.numpy as jnp
from beartype.typing import Callable, Dict, NamedTuple, Optional, Tuple
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


def multi_slice_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slices: Complex[Array, "H W S"],  # S is number of slices
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


def initialize_probe_modes(
    base_probe: Complex[Array, "H W"], num_modes: int, random_seed: int = 42
) -> pte.ProbeState:
    """
    Initialize multiple probe modes from a base probe.

    Args:
        base_probe: Base probe function
        num_modes: Number of modes to generate
        random_seed: Random seed for initialization

    Returns:
        ProbeState with initialized modes and weights
    """
    key = jax.random.PRNGKey(random_seed)

    # Initialize modes with small random perturbations of base probe
    perturbations = (
        jax.random.normal(
            key,
            shape=(base_probe.shape[0], base_probe.shape[1], num_modes),
            dtype=base_probe.dtype,
        )
        * 0.1
    )

    modes = jnp.tile(base_probe[..., None], (1, 1, num_modes)) + perturbations

    # Initialize weights with decreasing values
    weights = jnp.exp(-jnp.arange(num_modes, dtype=jnp.float32))
    weights = weights / jnp.sum(weights)  # Normalize

    return pte.ProbeState(modes=modes, weights=weights)


def multi_mode_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slices: Complex[Array, "H W S"],
    initial_probe_state: pte.ProbeState,
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


def initialize_mixed_states(
    base_state: Complex[Array, "H W"],
    num_states: int,
    energy_spread: float = 0.5,  # eV
    random_seed: int = 42,
) -> pte.MixedQuantumStates:
    """
    Initialize mixed states for partial temporal coherence.

    Args:
        base_state: Base state (e.g., probe)
        num_states: Number of states in mixture
        energy_spread: Energy spread in eV (FWHM)
        random_seed: Random seed for initialization

    Returns:
        MixedState with initialized states and probabilities
    """
    key = jax.random.PRNGKey(random_seed)

    # Generate energy offsets with Gaussian distribution
    sigma = energy_spread / (2.355)  # Convert FWHM to sigma
    energies = jax.random.normal(key, shape=(num_states,)) * sigma

    # Create states with phase variations
    phase_factors = jnp.exp(1j * energies[:, None, None])
    states = base_state[None, ...] * phase_factors

    # Calculate probabilities (Gaussian distribution)
    probabilities = jnp.exp(-(energies**2) / (2 * sigma**2))
    probabilities = probabilities / jnp.sum(probabilities)

    return pte.MixedQuantumStates(
        states=states.transpose(1, 2, 0), probabilities=probabilities
    )


def multi_mode_ptychography(
    experimental_4dstem: Float[Array, "P H W"],
    initial_pot_slices: Complex[Array, "H W S"],
    initial_modes: Complex[Array, "H W M"],  # M different probe modes
    mode_weights: Float[Array, "M"],  # Weights for each mode
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, ""],
    voltage_kV: Float[Array, ""],
    calib_ang: Float[Array, ""],
    num_iterations: int = 1000,
    learning_rate: float = 0.001,
    loss_type: str = "mse",
    optimizer_name: str = "adam",
    scheduler_fn: Optional[ptt.SchedulerFn] = None,
) -> Tuple[Complex[Array, "H W S"], Complex[Array, "H W M"], Float[Array, "M"]]:
    """
    Multi-mode ptychographic reconstruction with optional mixed state.

    Args:
        experimental_4dstem: Experimental 4D-STEM data
        initial_pot_slices: Initial guess for potential slices
        initial_modes: Initial guess for probe modes
        mode_weights: Initial weights for each mode
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
        Tuple of optimized potential slices, probe modes, and mode weights
    """
    # Normalize mode weights
    mode_weights = mode_weights / jnp.sum(mode_weights)

    def forward_fn(
        pot_slices: Complex[Array, "H W S"],
        modes: Complex[Array, "H W M"],
        weights: Float[Array, "M"],
    ) -> Float[Array, "P H W"]:
        return pte.stem_4d_mixed_state(
            pot_slices,
            modes,
            weights,
            pos_list,
            slice_thickness,
            voltage_kV,
            calib_ang,
        )

    # Create loss function
    loss_func = ptt.create_loss_function(forward_fn, experimental_4dstem, loss_type)

    @jax.jit
    def loss_and_grad(
        pot_slices: Complex[Array, "H W S"],
        modes: Complex[Array, "H W M"],
        weights: Float[Array, "M"],
    ) -> Tuple[Float[Array, ""], Dict[str, Array]]:
        loss, grads = jax.value_and_grad(loss_func, argnums=(0, 1, 2))(
            pot_slices, modes, weights
        )
        return loss, {"pot_slices": grads[0], "modes": grads[1], "weights": grads[2]}

    # Initialize optimizers
    optimizer = get_optimizer(optimizer_name)
    pot_slices_state = optimizer.init(initial_pot_slices.shape)
    modes_state = optimizer.init(initial_modes.shape)
    weights_state = optimizer.init(mode_weights.shape)

    if scheduler_fn is not None:
        scheduler_state = ptt.init_scheduler_state(learning_rate)

    # Initialize variables
    pot_slices = initial_pot_slices
    modes = initial_modes
    weights = mode_weights
    current_lr = learning_rate

    @jax.jit
    def update_step(
        pot_slices, modes, weights, pot_slices_state, modes_state, weights_state, lr
    ):
        loss, grads = loss_and_grad(pot_slices, modes, weights)

        # Update potential slices and modes
        pot_slices, pot_slices_state = optimizer.update(
            pot_slices, grads["pot_slices"], pot_slices_state, lr
        )
        modes, modes_state = optimizer.update(modes, grads["modes"], modes_state, lr)

        # Update weights and normalize
        weights, weights_state = optimizer.update(
            weights, grads["weights"], weights_state, lr
        )
        weights = weights / jnp.sum(weights)  # Ensure normalization

        return (
            pot_slices,
            modes,
            weights,
            pot_slices_state,
            modes_state,
            weights_state,
            loss,
        )

    for i in range(num_iterations):
        # Update learning rate if scheduler is provided
        if scheduler_fn is not None:
            current_lr, scheduler_state = scheduler_fn(scheduler_state)

        # Perform optimization step
        (
            pot_slices,
            modes,
            weights,
            pot_slices_state,
            modes_state,
            weights_state,
            loss,
        ) = update_step(
            pot_slices,
            modes,
            weights,
            pot_slices_state,
            modes_state,
            weights_state,
            current_lr,
        )

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}, LR: {current_lr}")

    return pot_slices, modes, weights
