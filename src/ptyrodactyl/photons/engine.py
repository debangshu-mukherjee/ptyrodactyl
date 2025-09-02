"""
Module: ptyrodactyl.photons.engine
----------------------------------
Engine for optical ptychography.

Functions
---------
- `epie_optical`:
    Main ePIE reconstruction algorithm for optical ptychography
- `single_pie_iteration`:
    Single iteration of the ePIE algorithm
- `single_pie_vmap`:
    Parallel processing over positions using vmap for faster but approximate PIE.
- `single_pie_sequential`:
    Sequential processing over positions for refinement.

Internal Functions
------------------
- `_update_object_wavefront`:
    Updates object wavefront using rPIE algorithm
- `_update_surface_pattern`:
    Updates surface pattern using modified PIE
- `_apply_coherent_transfer_function`:
    Applies coherent transfer function in Fourier domain
- `_apply_position_shift`:
    Applies position shift using phase multiplication
- `_compute_sensor_intensity`:
    Computes sensor plane intensity with pixel response
- `_create_frequency_grids`:
    Creates frequency grids for Fourier transforms
"""

import jax
import jax.numpy as jnp
from beartype.typing import Callable, Optional, Tuple
from jaxtyping import Array, Complex, Float

from ptyrodactyl._decorators import beartype, jaxtyped

from .lens_optics import angular_spectrum_prop
from .photon_types import (MicroscopeData, OpticalWavefront, SampleFunction,
                           make_optical_wavefront, make_sample_function,
                           scalar_float, scalar_integer)

jax.config.update("jax_enable_x64", True)


@jax.jit
@jaxtyped(typechecker=beartype)
def epie_optical(
    microscope_data: MicroscopeData,
    initial_object: OpticalWavefront,
    initial_surface: SampleFunction,
    pixel_mask: Float[Array, "H W"],
    propagation_distance_1: scalar_float,
    propagation_distance_2: scalar_float,
    magnification: scalar_integer,
    vmap_iterations: Optional[scalar_integer] = 0,
    alpha_object: Optional[scalar_float] = 0.1,
    gamma_object: Optional[scalar_float] = 0.5,
    alpha_surface: Optional[scalar_float] = 0.1,
    gamma_surface: Optional[scalar_float] = 0.5,
    num_loops: Optional[scalar_integer] = 10,
) -> tuple[OpticalWavefront, SampleFunction]:
    """Extended Ptychographic Iterative Engine for optical ptychography reconstruction.

    Parameters
    ----------
    microscope_data : MicroscopeData
        Measured intensity data with positions.
    initial_object : OpticalWavefront
        Initial guess for object wavefront.
    initial_surface : SampleFunction
        Initial guess for surface pattern.
    pixel_mask : Float[Array, "H W"]
        Pixel response mask for modeling sensor characteristics.
    propagation_distance_1 : scalar_float
        Distance from object to diffuser plane in meters.
    propagation_distance_2 : scalar_float
        Distance from diffuser to sensor plane in meters.
    magnification : scalar_integer
        Magnification factor for downsampling.
    vmap_iterations : scalar_integer, optional
        Number of initial iterations to run in vmap mode for rapid convergence.
        If 0, use sequential mode for all iterations.
        If > 0, use vmap for first N iterations, then switch to sequential.
        Default is 0.
    alpha_object : scalar_float, optional
        Object update mixing parameter. Default is 0.1.
    gamma_object : scalar_float, optional
        Object update step size. Default is 0.5.
    alpha_surface : scalar_float, optional
        Surface update mixing parameter. Default is 0.1.
    gamma_surface : scalar_float, optional
        Surface update step size. Default is 0.5.
    num_loops : scalar_integer, optional
        Number of iteration loops. Default is 10.

    Returns
    -------
    tuple of (OpticalWavefront, SampleFunction)
        - recovered_object : OpticalWavefront
            Reconstructed object wavefront.
        - recovered_surface : SampleFunction
            Reconstructed surface pattern.

    Notes
    -----
    Algorithm:
    - Compute image data
    - Compute positions
    - Compute frequency grids
    - Compute object recovery propagation field
    - Compute surface pattern
    - Define loop body
    - Apply fori_loop over loops
    - Compute final object field
    - Compute final object wavefront
    - Compute final surface pattern
    - Return final object and surface
    """
    image_data: Float[Array, "P H W"] = microscope_data.image_data
    positions: Float[Array, "P 2"] = microscope_data.positions

    frequency_x_grid: Float[Array, "H W"]
    frequency_y_grid: Float[Array, "H W"]
    frequency_x_grid, frequency_y_grid = _create_frequency_grids(
        initial_object.field, initial_object.dx
    )
    object_recovery_prop_ft: Complex[Array, "H W"] = angular_spectrum_prop(
        initial_object, propagation_distance_1
    ).field
    surface_pattern: Complex[Array, "H W"] = initial_surface.sample

    def loop_body(
        loop_idx: scalar_integer,
        state: tuple[Complex[Array, "H W"], Complex[Array, "H W"]],
    ) -> tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
        object_prop_ft: Complex[Array, "H W"]
        surface_pattern_current: Complex[Array, "H W"]
        object_prop_ft, surface_pattern_current = state
        use_vmap: bool = loop_idx < vmap_iterations
        position_processor: Callable = jax.lax.cond(
            use_vmap, lambda: single_pie_vmap, lambda: single_pie_sequential
        )
        updated_state: tuple[Complex[Array, "H W"], Complex[Array, "H W"]] = (
            position_processor(
                object_prop_ft,
                surface_pattern_current,
                image_data,
                positions,
                frequency_x_grid,
                frequency_y_grid,
                pixel_mask,
                propagation_distance_2,
                magnification,
                alpha_object,
                gamma_object,
                alpha_surface,
                gamma_surface,
                initial_object.wavelength,
                initial_object.dx,
            )
        )
        return updated_state

    final_object_ft: Complex[Array, "H W"]
    final_surface: Complex[Array, "H W"]
    final_object_ft, final_surface = jax.lax.fori_loop(
        0, num_loops, loop_body, (object_recovery_prop_ft, surface_pattern)
    )
    final_object_field: Complex[Array, "H W"] = angular_spectrum_prop(
        make_optical_wavefront(
            final_object_ft,
            initial_object.wavelength,
            initial_object.dx,
            initial_object.z_position + propagation_distance_1,
        ),
        -propagation_distance_1,
    ).field
    recovered_object: OpticalWavefront = make_optical_wavefront(
        final_object_field,
        initial_object.wavelength,
        initial_object.dx,
        initial_object.z_position,
    )
    recovered_surface: SampleFunction = make_sample_function(
        final_surface, initial_surface.dx
    )
    return recovered_object, recovered_surface


@jaxtyped(typechecker=beartype)
def single_pie_iteration(
    object_prop_ft: Complex[Array, "H W"],
    surface_pattern: Complex[Array, "H W"],
    measurement: Float[Array, "H W"],
    position: Float[Array, " 2"],
    frequency_x_grid: Float[Array, "H W"],
    frequency_y_grid: Float[Array, "H W"],
    pixel_mask: Float[Array, "H W"],
    propagation_distance_2: scalar_float,
    magnification: scalar_integer,
    alpha_object: scalar_float,
    gamma_object: scalar_float,
    alpha_surface: scalar_float,
    gamma_surface: scalar_float,
    wavelength: scalar_float,
    dx: scalar_float,
) -> tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
    """Single iteration of the extended PIE algorithm.

    Parameters
    ----------
    object_prop_ft : Complex[Array, "H W"]
        Object wavefront at diffuser plane in Fourier domain.
    surface_pattern : Complex[Array, "H W"]
        Surface pattern function.
    measurement : Float[Array, "H W"]
        Measured intensity at current position.
    position : Float[Array, " 2"]
        Current scanning position [x, y].
    frequency_x_grid : Float[Array, "H W"]
        Frequency grid in x direction.
    frequency_y_grid : Float[Array, "H W"]
        Frequency grid in y direction.
    pixel_mask : Float[Array, "H W"]
        Pixel response mask for sensor modeling.
    propagation_distance_2 : scalar_float
        Distance from diffuser to sensor.
    magnification : scalar_integer
        Downsampling magnification factor.
    alpha_object : scalar_float
        Object update mixing parameter.
    gamma_object : scalar_float
        Object update step size.
    alpha_surface : scalar_float
        Surface update mixing parameter.
    gamma_surface : scalar_float
        Surface update step size.
    wavelength : scalar_float
        Wavelength of light in meters.
    dx : scalar_float
        Pixel spacing in meters.

    Returns
    -------
    tuple of (Complex[Array, "H W"], Complex[Array, "H W"])
        - updated_object_ft : Complex[Array, "H W"]
            Updated object wavefront in Fourier domain.
        - updated_surface : Complex[Array, "H W"]
            Updated surface pattern.

    Notes
    -----
    Algorithm:
    - Compute object shifted
    - Compute surface plane
    - Compute surface propagation kernel
    - Compute sensor plane
    - Compute sensor intensity
    - Compute ratio map
    - Compute ratio map upsampled
    - Compute sensor plane new
    - Compute sensor plane new in Fourier domain
    - Compute CTF conjugate
    - Compute CTF maximum squared
    - Compute surface propagation kernel
    - Compute updated surface pattern
    - Compute updated object wavefront
    - Compute updated object wavefront in Fourier domain
    - Return updated object and surface
    """
    object_shifted: Complex[Array, "H W"] = _apply_position_shift(
        object_prop_ft, position, frequency_x_grid, frequency_y_grid
    )
    surface_plane: Complex[Array, "H W"] = object_shifted * surface_pattern
    surface_prop_ft: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(surface_plane)
    ) * _get_propagation_kernel(
        surface_plane.shape, propagation_distance_2, wavelength, dx
    )
    sensor_plane_ft: Complex[Array, "H W"] = _apply_coherent_transfer_function(
        surface_prop_ft
    )
    sensor_plane: Complex[Array, "H W"] = jnp.fft.ifft2(
        jnp.fft.ifftshift(sensor_plane_ft)
    )
    sensor_intensity: Float[Array, "H W"] = _compute_sensor_intensity(
        sensor_plane, pixel_mask, magnification
    )
    ratio_map: Float[Array, "H W"] = jnp.sqrt(measurement) / (
        jnp.sqrt(sensor_intensity) + 1e-10
    )
    ratio_map_upsampled: Float[Array, "H W"] = jnp.repeat(
        jnp.repeat(ratio_map, magnification, axis=0), magnification, axis=1
    )
    sensor_plane_new: Complex[Array, "H W"] = ratio_map_upsampled * sensor_plane
    sensor_plane_new_ft: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(sensor_plane_new)
    )
    ctf_conj: Complex[Array, "H W"] = jnp.conj(_get_ctf())
    ctf_max_squared: Float[Array, ""] = jnp.max(jnp.abs(_get_ctf()) ** 2)
    surface_prop_ft_updated: Complex[Array, "H W"] = surface_prop_ft + (
        ctf_conj * (sensor_plane_new_ft - sensor_plane_ft) / ctf_max_squared
    )
    surface_plane_new: Complex[Array, "H W"] = jnp.fft.ifft2(
        jnp.fft.ifftshift(
            surface_prop_ft_updated
            * _get_propagation_kernel(
                surface_prop_ft_updated.shape, -propagation_distance_2, wavelength, dx
            )
        )
    )
    updated_surface: Complex[Array, "H W"] = _update_surface_pattern(
        surface_pattern,
        object_shifted,
        surface_plane,
        surface_plane_new,
        alpha_surface,
        gamma_surface,
    )
    object_shifted_updated: Complex[Array, "H W"] = _update_object_wavefront(
        object_shifted,
        surface_pattern,
        surface_plane,
        surface_plane_new,
        alpha_object,
        gamma_object,
    )
    updated_object_ft: Complex[Array, "H W"] = _apply_position_shift(
        jnp.fft.fftshift(jnp.fft.fft2(object_shifted_updated)),
        -position,
        frequency_x_grid,
        frequency_y_grid,
    )
    return updated_object_ft, updated_surface


@jaxtyped(typechecker=beartype)
def single_pie_sequential(
    object_prop_ft: Complex[Array, "H W"],
    surface_pattern: Complex[Array, "H W"],
    image_data: Float[Array, "P H W"],
    positions: Float[Array, "P 2"],
    frequency_x_grid: Float[Array, "H W"],
    frequency_y_grid: Float[Array, "H W"],
    pixel_mask: Float[Array, "H W"],
    propagation_distance_2: scalar_float,
    magnification: scalar_integer,
    alpha_object: scalar_float,
    gamma_object: scalar_float,
    alpha_surface: scalar_float,
    gamma_surface: scalar_float,
    wavelength: scalar_float,
    dx: scalar_float,
) -> tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
    """Sequential processing over positions using fori_loop for proper PIE convergence.

    Parameters
    ----------
    object_prop_ft : Complex[Array, "H W"]
        Current object wavefront in Fourier domain.
    surface_pattern : Complex[Array, "H W"]
        Current surface pattern.
    image_data : Float[Array, "P H W"]
        Measurement data for all positions.
    positions : Float[Array, "P 2"]
        Position coordinates for all measurements.
    frequency_x_grid : Float[Array, "H W"]
        Frequency grid in x direction.
    frequency_y_grid : Float[Array, "H W"]
        Frequency grid in y direction.
    pixel_mask : Float[Array, "H W"]
        Pixel response mask for sensor modeling.
    propagation_distance_2 : scalar_float
        Distance from diffuser to sensor.
    magnification : scalar_integer
        Downsampling magnification factor.
    alpha_object : scalar_float
        Object update mixing parameter.
    gamma_object : scalar_float
        Object update step size.
    alpha_surface : scalar_float
        Surface update mixing parameter.
    gamma_surface : scalar_float
        Surface update step size.
    wavelength : scalar_float
        Wavelength of light in meters.
    dx : scalar_float
        Pixel spacing in meters.

    Returns
    -------
    tuple of (Complex[Array, "H W"], Complex[Array, "H W"])
        Updated object and surface state after sequential processing.

    Notes
    -----
    Algorithm:
    - Compute number of positions
    - Define position body
    - Apply fori_loop over positions
    - Return updated state
    """
    num_positions: scalar_integer = image_data.shape[0]

    def position_body(
        pos_idx: scalar_integer,
        state: tuple[Complex[Array, "H W"], Complex[Array, "H W"]],
    ) -> tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
        return single_pie_iteration(
            state[0],
            state[1],
            image_data[pos_idx],
            positions[pos_idx],
            frequency_x_grid,
            frequency_y_grid,
            pixel_mask,
            propagation_distance_2,
            magnification,
            alpha_object,
            gamma_object,
            alpha_surface,
            gamma_surface,
            wavelength,
            dx,
        )

    updated_state: tuple[Complex[Array, "H W"], Complex[Array, "H W"]] = (
        jax.lax.fori_loop(
            0, num_positions, position_body, (object_prop_ft, surface_pattern)
        )
    )
    return updated_state


@jaxtyped(typechecker=beartype)
def single_pie_vmap(
    object_prop_ft: Complex[Array, "H W"],
    surface_pattern: Complex[Array, "H W"],
    image_data: Float[Array, "P H W"],
    positions: Float[Array, "P 2"],
    frequency_x_grid: Float[Array, "H W"],
    frequency_y_grid: Float[Array, "H W"],
    pixel_mask: Float[Array, "H W"],
    propagation_distance_2: scalar_float,
    magnification: scalar_integer,
    alpha_object: scalar_float,
    gamma_object: scalar_float,
    alpha_surface: scalar_float,
    gamma_surface: scalar_float,
    wavelength: scalar_float,
    dx: scalar_float,
) -> tuple[Complex[Array, "H W"], Complex[Array, "H W"]]:
    """Parallel processing over positions using vmap for faster but approximate PIE.

    All positions use the same initial state, then updates are averaged.

    Parameters
    ----------
    object_prop_ft : Complex[Array, "H W"]
        Current object wavefront in Fourier domain.
    surface_pattern : Complex[Array, "H W"]
        Current surface pattern.
    image_data : Float[Array, "P H W"]
        Measurement data for all positions.
    positions : Float[Array, "P 2"]
        Position coordinates for all measurements.
    frequency_x_grid : Float[Array, "H W"]
        Frequency grid in x direction.
    frequency_y_grid : Float[Array, "H W"]
        Frequency grid in y direction.
    pixel_mask : Float[Array, "H W"]
        Pixel response mask for sensor modeling.
    propagation_distance_2 : scalar_float
        Distance from diffuser to sensor.
    magnification : scalar_integer
        Downsampling magnification factor.
    alpha_object : scalar_float
        Object update mixing parameter.
    gamma_object : scalar_float
        Object update step size.
    alpha_surface : scalar_float
        Surface update mixing parameter.
    gamma_surface : scalar_float
        Surface update step size.
    wavelength : scalar_float
        Wavelength of light in meters.
    dx : scalar_float
        Pixel spacing in meters.

    Returns
    -------
    tuple of (Complex[Array, "H W"], Complex[Array, "H W"])
        Updated object and surface state after parallel processing and averaging.

    Notes
    -----
    Algorithm:
    - Apply vmap over all positions using same initial state
    - Compute average of all object updates
    - Compute average of all surface updates
    - Return averaged states
    """
    vmapped_iteration = jax.vmap(
        single_pie_iteration,
        in_axes=(
            None,
            None,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        out_axes=0,
    )

    batch_object_updates: Complex[Array, "P H W"]
    batch_surface_updates: Complex[Array, "P H W"]
    batch_object_updates, batch_surface_updates = vmapped_iteration(
        object_prop_ft,
        surface_pattern,
        image_data,
        positions,
        frequency_x_grid,
        frequency_y_grid,
        pixel_mask,
        propagation_distance_2,
        magnification,
        alpha_object,
        gamma_object,
        alpha_surface,
        gamma_surface,
        wavelength,
        dx,
    )

    averaged_object: Complex[Array, "H W"] = jnp.mean(batch_object_updates, axis=0)
    averaged_surface: Complex[Array, "H W"] = jnp.mean(batch_surface_updates, axis=0)
    return averaged_object, averaged_surface


@jaxtyped(typechecker=beartype)
def _update_object_wavefront(
    object_shift: Complex[Array, "H W"],
    surface_pattern: Complex[Array, "H W"],
    surface_plane: Complex[Array, "H W"],
    surface_plane_new: Complex[Array, "H W"],
    alpha_object: scalar_float,
    gamma_object: scalar_float,
) -> Complex[Array, "H W"]:
    """Update object wavefront using rPIE algorithm.

    Parameters
    ----------
    object_shift : Complex[Array, "H W"]
        Current shifted object wavefront.
    surface_pattern : Complex[Array, "H W"]
        Surface pattern function.
    surface_plane : Complex[Array, "H W"]
        Current exit wave at surface.
    surface_plane_new : Complex[Array, "H W"]
        Updated exit wave at surface.
    alpha_object : scalar_float
        Mixing parameter for denominator.
    gamma_object : scalar_float
        Update step size.

    Returns
    -------
    Complex[Array, "H W"]
        Updated object wavefront.

    Notes
    -----
    Algorithm:
    - Compute surface conjugate
    - Compute difference between current and updated surface plane
    - Compute surface absolute squared
    - Compute surface maximum squared
    - Compute denominator
    - Compute update term
    - Compute updated object wavefront
    """
    surface_conj: Complex[Array, "H W"] = jnp.conj(surface_pattern)
    difference: Complex[Array, "H W"] = surface_plane_new - surface_plane
    surface_abs_squared: Float[Array, "H W"] = jnp.abs(surface_pattern) ** 2
    surface_max_squared: Float[Array, ""] = jnp.max(surface_abs_squared)
    denominator: Float[Array, "H W"] = (
        alpha_object * surface_max_squared + (1 - alpha_object) * surface_abs_squared
    )
    update_term: Complex[Array, "H W"] = (
        gamma_object * surface_conj * difference / (denominator + 1e-10)
    )
    updated_object: Complex[Array, "H W"] = object_shift + update_term
    return updated_object


@jaxtyped(typechecker=beartype)
def _update_surface_pattern(
    surface_pattern: Complex[Array, "H W"],
    object_shift: Complex[Array, "H W"],
    surface_plane: Complex[Array, "H W"],
    surface_plane_new: Complex[Array, "H W"],
    alpha_surface: scalar_float,
    gamma_surface: scalar_float,
) -> Complex[Array, "H W"]:
    """Update surface pattern using modified PIE algorithm.

    Parameters
    ----------
    surface_pattern : Complex[Array, "H W"]
        Current surface pattern.
    object_shift : Complex[Array, "H W"]
        Shifted object wavefront.
    surface_plane : Complex[Array, "H W"]
        Current exit wave at surface.
    surface_plane_new : Complex[Array, "H W"]
        Updated exit wave at surface.
    alpha_surface : scalar_float
        Mixing parameter for denominator.
    gamma_surface : scalar_float
        Update step size.

    Returns
    -------
    Complex[Array, "H W"]
        Updated surface pattern.

    Notes
    -----
    Algorithm:
    - Compute object conjugate
    - Compute difference between current and updated surface plane
    - Compute object absolute squared
    - Compute object maximum squared
    - Compute denominator
    - Compute update term
    - Compute updated surface pattern
    """
    object_conj: Complex[Array, "H W"] = jnp.conj(object_shift)
    difference: Complex[Array, "H W"] = surface_plane_new - surface_plane
    object_abs_squared: Float[Array, "H W"] = jnp.abs(object_shift) ** 2
    object_max_squared: Float[Array, ""] = jnp.max(object_abs_squared)
    denominator: Float[Array, "H W"] = (
        alpha_surface * object_max_squared + (1 - alpha_surface) * object_abs_squared
    )
    update_term: Complex[Array, "H W"] = (
        gamma_surface * object_conj * difference / (denominator + 1e-10)
    )
    updated_surface: Complex[Array, "H W"] = surface_pattern + update_term
    return updated_surface


@jaxtyped(typechecker=beartype)
def _apply_coherent_transfer_function(
    field_ft: Complex[Array, "H W"],
) -> Complex[Array, "H W"]:
    """Apply coherent transfer function in Fourier domain.

    Parameters
    ----------
    field_ft : Complex[Array, "H W"]
        Field in Fourier domain.

    Returns
    -------
    Complex[Array, "H W"]
        Field after CTF application.

    Notes
    -----
    Algorithm:
    - Compute CTF
    - Apply CTF to field
    - Return result
    """
    ctf: Complex[Array, "H W"] = _get_ctf()
    result: Complex[Array, "H W"] = ctf * field_ft
    return result


@jaxtyped(typechecker=beartype)
def _apply_position_shift(
    field_ft: Complex[Array, "H W"],
    position: Float[Array, " 2"],
    frequency_x_grid: Float[Array, "H W"],
    frequency_y_grid: Float[Array, "H W"],
) -> Complex[Array, "H W"]:
    """Apply position shift using phase multiplication in Fourier domain.

    Implements MATLAB: Hs = exp(-1j*2*pi.*(FX.*xlocation(tt)/imSize+FY.*ylocation(tt)/imSize))

    Parameters
    ----------
    field_ft : Complex[Array, "H W"]
        Field in Fourier domain.
    position : Float[Array, " 2"]
        Position shift [x, y].
    frequency_x_grid : Float[Array, "H W"]
        Frequency grid in x direction.
    frequency_y_grid : Float[Array, "H W"]
        Frequency grid in y direction.

    Returns
    -------
    Complex[Array, "H W"]
        Position-shifted field in real space.

    Notes
    -----
    Algorithm:
    - Compute image size
    - Compute phase factor
    - Compute position-shifted field in Fourier domain
    - Compute position-shifted field in real space
    - Return position-shifted field
    """
    image_size: scalar_integer = field_ft.shape[0]
    phase_factor: Complex[Array, "H W"] = jnp.exp(
        -1j
        * 2
        * jnp.pi
        * (
            frequency_x_grid * position[0] / image_size
            + frequency_y_grid * position[1] / image_size
        )
    )
    shifted_field_ft: Complex[Array, "H W"] = field_ft * phase_factor
    shifted_field: Complex[Array, "H W"] = jnp.fft.ifft2(
        jnp.fft.ifftshift(shifted_field_ft)
    )
    return shifted_field


@jaxtyped(typechecker=beartype)
def _get_propagation_kernel(
    field_shape: Tuple[int, int],
    distance: scalar_float,
    wavelength: scalar_float,
    dx: scalar_float,
) -> Complex[Array, "H W"]:
    """Return propagation kernel H_d for free space propagation.

    Parameters
    ----------
    field_shape : Tuple[int, int]
        Shape of the field (height, width).
    distance : scalar_float
        Propagation distance in meters.
    wavelength : scalar_float
        Wavelength in meters.
    dx : scalar_float
        Pixel spacing in meters.

    Returns
    -------
    Complex[Array, "H W"]
        Propagation kernel for angular spectrum method.

    Notes
    -----
    Algorithm:
    - Compute height and width of field
    - Compute frequency grid in x direction
    - Compute frequency grid in y direction
    - Compute frequency grid in x and y directions
    - Compute frequency squared
    - Compute k_0
    - Compute k_z
    - Compute propagation kernel
    - Return propagation kernel
    """
    height: int
    width: int
    height, width = field_shape
    frequency_x: Float[Array, " W"] = jnp.fft.fftfreq(width, dx)
    frequency_y: Float[Array, " H"] = jnp.fft.fftfreq(height, dx)
    frequency_x_grid: Float[Array, "H W"]
    frequency_y_grid: Float[Array, "H W"]
    frequency_x_grid, frequency_y_grid = jnp.meshgrid(frequency_x, frequency_y)
    frequency_squared: Float[Array, "H W"] = frequency_x_grid**2 + frequency_y_grid**2
    k_0: scalar_float = 2 * jnp.pi / wavelength
    k_z: Complex[Array, "H W"] = jnp.sqrt(
        k_0**2 - (2 * jnp.pi) ** 2 * frequency_squared + 0j
    )
    kernel: Complex[Array, "H W"] = jnp.exp(1j * k_z * distance)
    return kernel


@jaxtyped(typechecker=beartype)
def _compute_sensor_intensity(
    sensor_field: Complex[Array, "H W"],
    pixel_mask: Float[Array, "H W"],
    magnification: scalar_integer,
) -> Float[Array, "H W"]:
    """Compute sensor plane intensity with pixel response mask and downsampling.

    Implements the MATLAB: conv2(pixelMask.*abs(sensorPlane).^2, ones(mag,mag))

    Parameters
    ----------
    sensor_field : Complex[Array, "H W"]
        Complex field at sensor plane.
    pixel_mask : Float[Array, "H W"]
        Pixel response mask modeling sensor characteristics.
    magnification : scalar_integer
        Downsampling magnification factor.

    Returns
    -------
    Float[Array, "H W"]
        Downsampled intensity pattern.

    Notes
    -----
    Algorithm:
    - Compute intensity
    - Apply pixel mask
    - Compute kernel
    - Convolve intensity with kernel
    - Downsample intensity
    """
    intensity: Float[Array, "H W"] = jnp.abs(sensor_field) ** 2
    masked_intensity: Float[Array, "H W"] = pixel_mask * intensity
    kernel: Float[Array, "mag mag"] = jnp.ones((magnification, magnification))
    convolved_intensity: Float[Array, "H W"] = jax.scipy.signal.convolve2d(
        masked_intensity, kernel, mode="same"
    )
    downsampled_intensity: Float[Array, "H_new W_new"] = convolved_intensity[
        magnification - 1 :: magnification, magnification - 1 :: magnification
    ]
    return downsampled_intensity


@jaxtyped(typechecker=beartype)
def _create_frequency_grids(
    field: Complex[Array, "H W"], dx: scalar_float
) -> Tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """Create frequency grids for Fourier transforms.

    Parameters
    ----------
    field : Complex[Array, "H W"]
        Input field to determine grid size.
    dx : scalar_float
        Spatial sampling interval.

    Returns
    -------
    tuple of (Float[Array, "H W"], Float[Array, "H W"])
        - frequency_x_grid : Float[Array, "H W"]
            Frequency grid in x direction.
        - frequency_y_grid : Float[Array, "H W"]
            Frequency grid in y direction.

    Notes
    -----
    Algorithm:
    - Compute height and width of field
    - Compute frequency grid in x direction
    - Compute frequency grid in y direction
    - Compute frequency grid in x and y directions
    - Return frequency grids
    """
    height: scalar_integer = field.shape[0]
    width: scalar_integer = field.shape[1]
    frequency_x: Float[Array, " W"] = jnp.fft.fftfreq(width, dx)
    frequency_y: Float[Array, " H"] = jnp.fft.fftfreq(height, dx)
    frequency_x_grid: Float[Array, "H W"]
    frequency_y_grid: Float[Array, "H W"]
    frequency_x_grid, frequency_y_grid = jnp.meshgrid(frequency_x, frequency_y)
    return frequency_x_grid, frequency_y_grid


@jaxtyped(typechecker=beartype)
def _get_ctf(
    field_shape: Optional[Tuple[int, int]] = (256, 256)
) -> Complex[Array, "H W"]:
    """Return a placeholder coherent transfer function.

    Parameters
    ----------
    field_shape : Tuple[int, int], optional
        Shape of the field (height, width). Default is (256, 256).

    Returns
    -------
    Complex[Array, "H W"]
        Coherent transfer function.
    """
    ctf: Complex[Array, "H W"] = jnp.ones(field_shape, dtype=complex)
    return ctf
