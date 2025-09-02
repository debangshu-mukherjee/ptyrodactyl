"""
Module: ptyrodactyl.photons.invertor
------------------------------------
Codes for optical propagation through lenses and optical elements.

Functions
---------
- `get_optimizer`:
    Returns an optimizer object based on the specified name
- `simple_microscope_ptychography`:
    Performs ptychography reconstruction using a simple microscope model
"""

import jax
import jax.numpy as jnp
from beartype.typing import Dict, Optional, Tuple
from jaxtyping import Array, Complex, Float

import ptyrodactyl.tools as ptt
from ptyrodactyl._decorators import beartype, jaxtyped

from .microscope import simple_microscope
from .photon_types import (MicroscopeData, OpticalWavefront, SampleFunction,
                           make_optical_wavefront, make_sample_function,
                           scalar_float, scalar_integer)

jax.config.update("jax_enable_x64", True)

OPTIMIZERS: Dict[str, ptt.Optimizer] = {
    "adam": ptt.Optimizer(ptt.init_adam, ptt.adam_update),
    "adagrad": ptt.Optimizer(ptt.init_adagrad, ptt.adagrad_update),
    "rmsprop": ptt.Optimizer(ptt.init_rmsprop, ptt.rmsprop_update),
}


def get_optimizer(optimizer_name: str) -> ptt.Optimizer:
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return OPTIMIZERS[optimizer_name]


@jaxtyped(typechecker=beartype)
def simple_microscope_ptychography(
    experimental_data: MicroscopeData,
    guess_sample: SampleFunction,
    guess_lightwave: OpticalWavefront,
    zoom_factor: scalar_float,
    aperture_diameter: scalar_float,
    travel_distance: scalar_float,
    camera_pixel_size: scalar_float,
    aperture_center: Optional[Float[Array, " 2"]] = None,
    learning_rate: Optional[scalar_float] = 0.01,
    num_iterations: Optional[scalar_integer] = 1000,
    save_every: Optional[scalar_integer] = 10,
    loss_type: Optional[str] = "mse",
    optimizer_name: Optional[str] = "adam",
    zoom_factor_bounds: Optional[Tuple[scalar_float, scalar_float]] = None,
    aperture_diameter_bounds: Optional[Tuple[scalar_float, scalar_float]] = None,
    travel_distance_bounds: Optional[Tuple[scalar_float, scalar_float]] = None,
    aperture_center_bounds: Optional[
        Tuple[Float[Array, " 2"], Float[Array, " 2"]]
    ] = None,
) -> Tuple[
    Tuple[
        SampleFunction,  # final_sample
        OpticalWavefront,  # final_lightwave
        scalar_float,  # final_zoom_factor
        scalar_float,  # final_aperture_diameter
        Optional[Float[Array, " 2"]],  # final_aperture_center
        scalar_float,  # final_travel_distance
    ],
    Tuple[
        Complex[Array, "H W S"],  # intermediate_samples
        Complex[Array, "H W S"],  # intermediate_lightwaves
        Float[Array, " S"],  # intermediate_zoom_factors
        Float[Array, " S"],  # intermediate_aperture_diameters
        Float[Array, " 2 S"],  # intermediate_aperture_centers
        Float[Array, " S"],  # intermediate_travel_distances
    ],
]:
    """Solve the optical ptychography inverse problem where experimental diffraction patterns are used to reconstruct a sample, lightwave, and optical system parameters.

    Parameters
    ----------
    experimental_data : MicroscopeData
        The experimental diffraction patterns collected at different positions.
    guess_sample : SampleFunction
        Initial guess for the sample properties.
    guess_lightwave : OpticalWavefront
        Initial guess for the lightwave.
    zoom_factor : scalar_float
        Initial guess for the optical zoom factor.
    aperture_diameter : scalar_float
        Initial guess for the aperture diameter in meters.
    travel_distance : scalar_float
        Initial guess for the light propagation distance in meters.
    camera_pixel_size : scalar_float
        The pixel size of the camera in meters (fixed parameter).
    aperture_center : Float[Array, " 2"], optional
        Initial guess for the center of the aperture.
    learning_rate : scalar_float, optional
        Learning rate for optimization. Default is 0.01.
    num_iterations : scalar_integer, optional
        Number of optimization iterations. Default is 1000.
    save_every : scalar_integer, optional
        Save intermediate results every n iterations. Default is 10.
    loss_type : str, optional
        Type of loss function to use. Default is "mse".
    optimizer_name : str, optional
        Name of the optimizer to use. Default is "adam".
    zoom_factor_bounds : Tuple[scalar_float, scalar_float], optional
        Lower and upper bounds for zoom factor optimization.
    aperture_diameter_bounds : Tuple[scalar_float, scalar_float], optional
        Lower and upper bounds for aperture diameter optimization.
    travel_distance_bounds : Tuple[scalar_float, scalar_float], optional
        Lower and upper bounds for travel distance optimization.
    aperture_center_bounds : Tuple[Float[Array, " 2"], Float[Array, " 2"]], optional
        Lower and upper bounds for aperture center optimization.

    Returns
    -------
    Tuple[Tuple[...], Tuple[...]]
        Tuple containing:
        - Final results tuple:
            - final_sample : SampleFunction
                Optimized sample properties.
            - final_lightwave : OpticalWavefront
                Optimized lightwave.
            - final_zoom_factor : scalar_float
                Optimized zoom factor.
            - final_aperture_diameter : scalar_float
                Optimized aperture diameter.
            - final_aperture_center : Float[Array, " 2"] or None
                Optimized aperture center.
            - final_travel_distance : scalar_float
                Optimized travel distance.
        - Intermediate results tuple:
            - intermediate_samples : Complex[Array, "H W S"]
                Intermediate samples during optimization.
            - intermediate_lightwaves : Complex[Array, "H W S"]
                Intermediate lightwaves during optimization.
            - intermediate_zoom_factors : Float[Array, " S"]
                Intermediate zoom factors during optimization.
            - intermediate_aperture_diameters : Float[Array, " S"]
                Intermediate aperture diameters during optimization.
            - intermediate_aperture_centers : Float[Array, " 2 S"]
                Intermediate aperture centers during optimization.
            - intermediate_travel_distances : Float[Array, " S"]
                Intermediate travel distances during optimization.
    """

    # Define bound enforcement functions
    def enforce_bounds(param, param_bounds):
        if param_bounds is None:
            return param
        lower, upper = param_bounds
        return jnp.clip(param, lower, upper)

    def enforce_bounds_2d(param, param_bounds):
        if param_bounds is None:
            return param
        lower, upper = param_bounds
        return jnp.clip(param, lower, upper)

    # Define the forward model function for the loss calculation
    def forward_fn(
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
    ):
        # Reconstruct PyTree objects from arrays
        sample = make_sample_function(sample=sample_field, dx=guess_sample.dx)

        lightwave = make_optical_wavefront(
            field=lightwave_field,
            wavelength=guess_lightwave.wavelength,
            dx=guess_lightwave.dx,
            z_position=guess_lightwave.z_position,
        )

        # Generate the microscope data using the forward model
        simulated_data = simple_microscope(
            sample=sample,
            positions=experimental_data.positions,
            lightwave=lightwave,
            zoom_factor=zoom_factor,
            aperture_diameter=aperture_diameter,
            travel_distance=travel_distance,
            camera_pixel_size=camera_pixel_size,
            aperture_center=aperture_center,
        )

        return simulated_data.image_data

    # Create loss function using the tools module
    loss_func = ptt.create_loss_function(
        forward_fn, experimental_data.image_data, loss_type
    )

    # Define function to compute loss and gradients
    @jax.jit
    def loss_and_grad(
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
    ):
        def loss_wrapped(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        ):
            # Enforce bounds before calculating loss
            bounded_zoom_factor = enforce_bounds(zoom_factor, zoom_factor_bounds)
            bounded_aperture_diameter = enforce_bounds(
                aperture_diameter, aperture_diameter_bounds
            )
            bounded_travel_distance = enforce_bounds(
                travel_distance, travel_distance_bounds
            )
            bounded_aperture_center = enforce_bounds_2d(
                aperture_center, aperture_center_bounds
            )

            return loss_func(
                sample_field,
                lightwave_field,
                bounded_zoom_factor,
                bounded_aperture_diameter,
                bounded_travel_distance,
                bounded_aperture_center,
            )

        loss, grads = jax.value_and_grad(loss_wrapped, argnums=(0, 1, 2, 3, 4, 5))(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        )

        return loss, {
            "sample": grads[0],
            "lightwave": grads[1],
            "zoom_factor": grads[2],
            "aperture_diameter": grads[3],
            "travel_distance": grads[4],
            "aperture_center": grads[5],
        }

    # Get the selected optimizer
    optimizer = get_optimizer(optimizer_name)

    # Initialize optimizer states
    sample_state = optimizer.init(guess_sample.sample.shape)
    lightwave_state = optimizer.init(guess_lightwave.field.shape)
    zoom_factor_state = optimizer.init(())  # Scalar param
    aperture_diameter_state = optimizer.init(())  # Scalar param
    travel_distance_state = optimizer.init(())  # Scalar param
    aperture_center_state = optimizer.init((2,) if aperture_center is not None else ())

    # Initialize parameters
    sample_field = guess_sample.sample
    lightwave_field = guess_lightwave.field
    current_zoom_factor = zoom_factor
    current_aperture_diameter = aperture_diameter
    current_travel_distance = travel_distance
    current_aperture_center = (
        jnp.zeros(2) if aperture_center is None else aperture_center
    )

    # Set up intermediate result storage
    num_saves = jnp.floor(num_iterations / save_every).astype(int)

    intermediate_samples = jnp.zeros(
        (sample_field.shape[0], sample_field.shape[1], num_saves),
        dtype=sample_field.dtype,
    )

    intermediate_lightwaves = jnp.zeros(
        (lightwave_field.shape[0], lightwave_field.shape[1], num_saves),
        dtype=lightwave_field.dtype,
    )

    intermediate_zoom_factors = jnp.zeros(num_saves, dtype=jnp.float64)
    intermediate_aperture_diameters = jnp.zeros(num_saves, dtype=jnp.float64)
    intermediate_travel_distances = jnp.zeros(num_saves, dtype=jnp.float64)
    intermediate_aperture_centers = jnp.zeros((2, num_saves), dtype=jnp.float64)

    @jax.jit
    def update_step(
        sample_field,
        lightwave_field,
        zoom_factor,
        aperture_diameter,
        travel_distance,
        aperture_center,
        sample_state,
        lightwave_state,
        zoom_factor_state,
        aperture_diameter_state,
        travel_distance_state,
        aperture_center_state,
    ):
        loss, grads = loss_and_grad(
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
        )

        # Update sample
        sample_field, sample_state = optimizer.update(
            sample_field, grads["sample"], sample_state, learning_rate
        )

        # Update lightwave
        lightwave_field, lightwave_state = optimizer.update(
            lightwave_field, grads["lightwave"], lightwave_state, learning_rate
        )

        # Update zoom factor
        zoom_factor, zoom_factor_state = optimizer.update(
            zoom_factor, grads["zoom_factor"], zoom_factor_state, learning_rate
        )
        zoom_factor = enforce_bounds(zoom_factor, zoom_factor_bounds)

        # Update aperture diameter
        aperture_diameter, aperture_diameter_state = optimizer.update(
            aperture_diameter,
            grads["aperture_diameter"],
            aperture_diameter_state,
            learning_rate,
        )
        aperture_diameter = enforce_bounds(aperture_diameter, aperture_diameter_bounds)

        # Update travel distance
        travel_distance, travel_distance_state = optimizer.update(
            travel_distance,
            grads["travel_distance"],
            travel_distance_state,
            learning_rate,
        )
        travel_distance = enforce_bounds(travel_distance, travel_distance_bounds)

        # Update aperture center
        aperture_center, aperture_center_state = optimizer.update(
            aperture_center,
            grads["aperture_center"],
            aperture_center_state,
            learning_rate,
        )
        aperture_center = enforce_bounds_2d(aperture_center, aperture_center_bounds)

        return (
            sample_field,
            lightwave_field,
            zoom_factor,
            aperture_diameter,
            travel_distance,
            aperture_center,
            sample_state,
            lightwave_state,
            zoom_factor_state,
            aperture_diameter_state,
            travel_distance_state,
            aperture_center_state,
            loss,
        )

    # Run optimization loop
    for ii in range(num_iterations):
        (
            sample_field,
            lightwave_field,
            current_zoom_factor,
            current_aperture_diameter,
            current_travel_distance,
            current_aperture_center,
            sample_state,
            lightwave_state,
            zoom_factor_state,
            aperture_diameter_state,
            travel_distance_state,
            aperture_center_state,
            loss,
        ) = update_step(
            sample_field,
            lightwave_field,
            current_zoom_factor,
            current_aperture_diameter,
            current_travel_distance,
            current_aperture_center,
            sample_state,
            lightwave_state,
            zoom_factor_state,
            aperture_diameter_state,
            travel_distance_state,
            aperture_center_state,
        )

        # Save intermediate results
        if ii % save_every == 0:
            print(f"Iteration {ii}, Loss: {loss}")
            save_idx = ii // save_every
            if save_idx < num_saves:
                intermediate_samples = intermediate_samples.at[:, :, save_idx].set(
                    sample_field
                )
                intermediate_lightwaves = intermediate_lightwaves.at[
                    :, :, save_idx
                ].set(lightwave_field)
                intermediate_zoom_factors = intermediate_zoom_factors.at[save_idx].set(
                    current_zoom_factor
                )
                intermediate_aperture_diameters = intermediate_aperture_diameters.at[
                    save_idx
                ].set(current_aperture_diameter)
                intermediate_travel_distances = intermediate_travel_distances.at[
                    save_idx
                ].set(current_travel_distance)
                intermediate_aperture_centers = intermediate_aperture_centers.at[
                    :, save_idx
                ].set(current_aperture_center)

    # Create final objects
    final_sample = make_sample_function(sample=sample_field, dx=guess_sample.dx)

    final_lightwave = make_optical_wavefront(
        field=lightwave_field,
        wavelength=guess_lightwave.wavelength,
        dx=guess_lightwave.dx,
        z_position=guess_lightwave.z_position,
    )

    # Create final values tuple
    final_values = (
        final_sample,
        final_lightwave,
        current_zoom_factor,
        current_aperture_diameter,
        current_aperture_center,
        current_travel_distance,
    )

    # Create intermediate values tuple
    intermediate_values = (
        intermediate_samples,
        intermediate_lightwaves,
        intermediate_zoom_factors,
        intermediate_aperture_diameters,
        intermediate_aperture_centers,
        intermediate_travel_distances,
    )

    # Return both tuples as a single tuple of tuples
    return (final_values, intermediate_values)
