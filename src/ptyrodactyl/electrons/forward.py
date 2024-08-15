from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
# from typeguard import typechecked as typechecker
from jax import Array, lax
from jaxtyping import Complex, Float, Int, Shaped, jaxtyped

import ptyrodactyl.electrons as pte

jax.config.update("jax_enable_x64", True)


def transmission_func(
    pot_slice: Float[Array, "*"], voltage_kV: int | float | Float[Array, "*"]
) -> Complex[Array, "*"]:
    """
    Calculates the complex transmission function from
    a single potential slice at a given electron accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Args:
    - `pot_slice`, Float[Array, "*"]:
        potential slice in Kirkland units
    - `voltage_kV`, int | float | Float[Array, "*"]:
        microscope operating voltage in kilo
        electronVolts

    Returns:
    - `trans` Complex[Array, "*"]:
        The transmission function of a single
        crystal slice

    Flow:
    - Calculate the electron energy in electronVolts
    - Calculate the wavelength in angstroms
    - Calculate the Einstein energy
    - Calculate the sigma value, which is the constant for the phase shift
    - Calculate the transmission function as a complex exponential
    """

    voltage: Float[Array, "*"] = jnp.multiply(
        jnp.float64(voltage_kV), jnp.float64(1000)
    )

    m_e: Float[Array, "*"] = jnp.float64(9.109383e-31)  # mass of an electron
    e_e: Float[Array, "*"] = jnp.float64(1.602177e-19)  # charge of an electron
    c: Float[Array, "*"] = jnp.float64(299792458.0)  # speed of light

    eV = jnp.multiply(e_e, voltage)
    lambda_angstrom: Float[Array, "*"] = pte.wavelength_ang(
        voltage_kV
    )  # wavelength in angstroms
    einstein_energy = jnp.multiply(m_e, jnp.square(c))  # Einstein energy
    sigma: Float[Array, "*"] = (
        (2 * jnp.pi / (lambda_angstrom * voltage)) * (einstein_energy + eV)
    ) / ((2 * einstein_energy) + eV)
    trans: Complex[Array, "*"] = jnp.exp(1j * sigma * pot_slice)
    return trans


def propagation_func(
    imsize_y: int,
    imsize_x: int,
    thickness_ang: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
) -> Complex[Array, "H W"]:
    """
    Calculates the complex propagation function that results
    in the phase shift of the exit wave when it travels from
    one slice to the next in the multislice algorithm

    Args:
    - `imsize_y`, (int):
        Size of the image of the propagator in y axis
    - `imsize_x`, (int):
        Size of the image of the propagator in x axis
    -  `thickness_ang`, (Float[Array, "*"])
        Distance between the slices in angstroms
    - `voltage_kV`, (Float[Array, "*"])
        Accelerating voltage in kilovolts
    - `calib_ang`, (Float[Array, "*"])
        Calibration or pixel size in angstroms

    Returns:
    - `prop` Complex[Array, "H W"]:
        The propagation function of the same size given by imsize
    """
    # Generate frequency arrays directly using fftfreq
    qy: Float[Array, "H"] = jnp.fft.fftfreq(imsize_y, d=calib_ang)
    qx: Float[Array, "W"] = jnp.fft.fftfreq(imsize_x, d=calib_ang)

    # Create 2D meshgrid of frequencies
    Lya, Lxa = jnp.meshgrid(qy, qx, indexing="ij")

    # Calculate squared sum of frequencies
    L_sq: Float[Array, "H W"] = jnp.square(Lxa) + jnp.square(Lya)

    # Calculate wavelength
    lambda_angstrom: float = pte.wavelength_ang(voltage_kV)

    # Compute the propagation function
    prop: Complex[Array, "H W"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * L_sq
    )
    return prop


def fourier_coords(calibration: float, image_size: Int[Array, "2"]) -> NamedTuple:
    """
    Return the Fourier coordinates

    Args:
    - `calibration`, float:
        The pixel size in angstroms in real space
    - `image_size`, Int[Array, "2"]:
        The size of the beam in pixels

    Returns:
    - A NamedTuple with the following fields:
        - `array`, Any[Array, "* *"]:
            The array values
        - `calib_y`, float:
            Calibration along the first axis
        - `calib_x`, float:
            Calibration along the second axis
    """
    real_fov_y: float = image_size[0] * calibration  # real space field of view in y
    real_fov_x: float = image_size[1] * calibration  # real space field of view in x
    inverse_arr_y: Float[Array, "H"] = (
        jnp.arange((-image_size[0] / 2), ((image_size[0] / 2)), 1)
    ) / real_fov_y  # inverse space array y
    inverse_arr_x: Float[Array, "W"] = (
        jnp.arange((-image_size[1] / 2), ((image_size[1] / 2)), 1)
    ) / real_fov_x  # inverse space array x
    shifter_y: float = image_size[0] // 2
    shifter_x: float = image_size[1] // 2
    inverse_shifted_y: Float[Array, "H"] = jnp.roll(
        inverse_arr_y, shifter_y
    )  # shifted inverse space array y
    inverse_shifted_x: Float[Array, "W"] = jnp.roll(
        inverse_arr_x, shifter_x
    )  # shifted inverse space array y
    inverse_xx: Float[Array, "H W"]
    inverse_yy: Float[Array, "H W"]
    inverse_xx, inverse_yy = jnp.meshgrid(inverse_shifted_x, inverse_shifted_y)
    inv_squared = jnp.multiply(inverse_yy, inverse_yy) + jnp.multiply(
        inverse_xx, inverse_xx
    )
    inverse_array: Float[Array, "H W"] = inv_squared**0.5
    calib_inverse_y: float = inverse_arr_y[1] - inverse_arr_y[0]
    calib_inverse_x: float = inverse_arr_x[1] - inverse_arr_x[0]
    calibrated_array = NamedTuple(
        "array_with_calibrations",
        [("array", Any[Array, "* *"]), ("calib_y", float), ("calib_x", float)],
    )
    return calibrated_array(inverse_array, calib_inverse_y, calib_inverse_x)


def fourier_calib(
    real_space_calib: float | Float[Array, "*"],
    sizebeam: Int[Array, "2"],
) -> Float[Array, "2"]:
    """
    Generate the Fourier calibration for the beam

    Args:
    - `real_space_calib`, float | Float[Array, "*"]:
        The pixel size in angstroms in real space
    - `sizebeam`, Int[Array, "2"]:
        The size of the beam in pixels

    Returns:
    - `inverse_space_calib`, Float[Array, "2"]:
        The Fourier calibration in angstroms
    """
    field_of_view: Float[Array, "*"] = jnp.multiply(
        jnp.float64(sizebeam), real_space_calib
    )
    inverse_space_calib = 1 / field_of_view
    return inverse_space_calib


@jax.jit
def make_probe(
    aperture: float,
    voltage: float,
    image_size: Int[Array, "2"],
    calibration_pm: float,
    defocus: float = 0,
    c3: float = 0,
    c5: float = 0,
) -> Complex[Array, "H W"]:
    """
    This calculates an electron probe based on the
    size and the estimated Fourier co-ordinates with
    the option of adding spherical aberration in the
    form of defocus, C3 and C5
    """
    aperture = aperture / 1000
    wavelength = wavelength_ang(voltage)
    LMax = aperture / wavelength
    image_y, image_x = image_size
    x_FOV = image_x * 0.01 * calibration_pm
    y_FOV = image_y * 0.01 * calibration_pm
    qx = (jnp.arange((-image_x / 2), (image_x / 2), 1)) / x_FOV
    x_shifter = image_x // 2
    qy = (jnp.arange((-image_y / 2), (image_y / 2), 1)) / y_FOV
    y_shifter = image_y // 2
    Lx = jnp.roll(qx, x_shifter)
    Ly = jnp.roll(qy, y_shifter)
    Lya, Lxa = jnp.meshgrid(Lx, Ly)
    L2 = jnp.multiply(Lxa, Lxa) + jnp.multiply(Lya, Lya)
    inverse_real_matrix = L2**0.5
    Adist = jnp.asarray(inverse_real_matrix <= LMax, dtype=jnp.complex64)
    chi_probe = aberration(inverse_real_matrix, wavelength, defocus, c3, c5)
    Adist *= jnp.exp(-1j * chi_probe)
    probe_real_space = jnp.fft.ifftshift(jnp.fft.ifft2(Adist))
    return probe_real_space


@jax.jit
def aberration(
    fourier_coord: Float[Array, "H W"],
    wavelength_ang: float,
    defocus: float = 0,
    c3: float = 0,
    c5: float = 0,
) -> Float[Array, "H W"]:
    p_matrix = wavelength_ang * fourier_coord
    chi = (
        ((defocus * jnp.power(p_matrix, 2)) / 2)
        + ((c3 * (1e7) * jnp.power(p_matrix, 4)) / 4)
        + ((c5 * (1e7) * jnp.power(p_matrix, 6)) / 6)
    )
    chi_probe = (2 * jnp.pi * chi) / wavelength_ang
    return chi_probe


@jaxtyped(typechecker=typechecker)
def wavelength_ang(voltage_kV: int | float | Float[Array, "*"]) -> Float[Array, "*"]:
    """
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Args:
    - `voltage_kV`, int | float | Float[Array, "*"]:
        The microscope accelerating voltage in kilo
        electronVolts

    Returns:
    - `in_angstroms`, Float[Array, "*"]:
        The electron wavelength in angstroms

    Flow:
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: Float[Array, "*"] = jnp.float64(9.109383e-31)  # mass of an electron
    e: Float[Array, "*"] = jnp.float64(1.602177e-19)  # charge of an electron
    c: Float[Array, "*"] = jnp.float64(299792458.0)  # speed of light
    h: Float[Array, "*"] = jnp.float64(6.62607e-34)  # Planck's constant

    voltage: Float[Array, "*"] = jnp.multiply(
        jnp.float64(voltage_kV), jnp.float64(1000)
    )
    eV = jnp.multiply(e, voltage)
    numerator: Float[Array, "*"] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, "*"] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, "*"] = jnp.sqrt(
        numerator / denominator
    )  # in meters
    in_angstroms: Float[Array, "*"] = 1e10 * wavelength_meters  # in angstroms
    return in_angstroms


# @jaxtyped(typechecker=typechecker)
def cbed_single_slice_single_beam(
    pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W"]
) -> Float[Array, "H W"]:
    """
    Simplest form of the CBED calculation

    Args:
    - `pot_slice`, Complex[Array, "H W"]:
        The potential slice
    - `beam`, Complex[Array, "H W"]:
        The electron beam

    Returns:
    - `cbed`, Float[Array, "H W"]:
        The calculated CBED pattern
    """
    real_space_convolve: Complex[Array, "H W"] = jnp.multiply(pot_slice, beam)
    fourier_space: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(real_space_convolve)
    )
    cbed: Float[Array, "H W"] = jnp.square(jnp.abs(fourier_space))
    return cbed


# @jaxtyped(typechecker=typechecker)
def cbed_single_slice_multi_beam(
    pot_slice: Complex[Array, "H W"], beam: Complex[Array, "H W M"]
) -> Float[Array, "H W"]:
    """
    Calculates the CBED pattern for a single slice when the
    beam is multimodal. After the calculation, we do a non-coherent
    sum, where the intensities are summed together, rather than the
    complex values being summed.

    Args:
    - `pot_slice`, Complex[Array, "H W"]:
        The potential slice
    - `beam`, Complex[Array, "H W M"]:
        The electron beam, where the last dimension is the number of modes

    Returns:
    - `cbed`, Float[Array, "H W"]:
        The calculated CBED pattern
    """

    def process_single_mode(beam_mode: Complex[Array, "H W"]) -> Complex[Array, "H W"]:
        real_space_mode: Complex[Array, "H W"] = jnp.multiply(pot_slice, beam_mode)
        return jnp.fft.fftshift(jnp.fft.fft2(real_space_mode))

    # Vectorize the process_single_mode function over the last axis of beam
    vectorized_process: Complex[Array, "H W *"] = jax.vmap(
        process_single_mode, in_axes=-1, out_axes=-1
    )

    # Apply the vectorized function to all modes at once
    fourier_space_modes: Complex[Array, "H W M"] = vectorized_process(beam)

    # Sum the intensities of all modes
    cbed: Float[Array, "H W"] = jnp.sum(
        jnp.square(jnp.abs(fourier_space_modes)), axis=-1
    )

    return cbed


# @jaxtyped(typechecker=typechecker)
def cbed_multi_slice_single_beam(
    pot_slice: Complex[Array, "H W S"],
    beam: Complex[Array, "H W"],
    slice_thickness: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
) -> Float[Array, "H W"]:
    """
    Multi-slice form of the CBED calculation,
    where the potential slice is in the form of
    a stack of slices, with the last dimension
    being the number of slices

    Args:
    - `pot_slice`, (Complex[Array, "H W S"]):
        The potential slice
    - `beam`, (Complex[Array, "H W"]):
        The electron beam
    - `slice_thickness`, (Float[Array, "*"]):
        The thickness of the slices in angstroms
    - `voltage_kV`, (Float[Array, "*"]):
        The accelerating voltage in kilovolts
    - `calib_ang`, (Float[Array, "*"]):
        The calibration in angstroms

    Returns:
    - `cbed`, (Float[Array, "H W"]):
        The calculated CBED pattern
    """
    size_y, size_x = beam.shape
    propagation_func_jit = jax.jit(pte.propagation_func, static_argnums=(0, 1))
    slice_transmission: Complex[Array, "H W"] = propagation_func_jit(
        size_y, size_x, slice_thickness, voltage_kV, calib_ang
    )

    def body_fun(carry, x):
        real_space_convolve, beam = carry
        this_slice = jnp.multiply(x, beam)
        propagated_slice = jnp.fft.ifft2(
            jnp.multiply(jnp.fft.fft2(this_slice), slice_transmission)
        )
        new_real_space_convolve = jnp.multiply(real_space_convolve, propagated_slice)
        return (new_real_space_convolve, beam), None

    initial_carry = (jnp.ones_like(beam, dtype=jnp.complex64), beam)
    (real_space_convolve, _), _ = jax.lax.scan(
        body_fun, initial_carry, pot_slice.transpose(2, 0, 1)
    )

    fourier_space: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(real_space_convolve)
    )
    cbed: Float[Array, "H W"] = jnp.square(jnp.abs(fourier_space))
    return cbed

def cbed_multi_slice_single_beam_slice(
    pot_slice: Complex[Array, "H W S"],
    beam: Complex[Array, "H W"],
    transmission_slice: Complex[Array, "H W"],
) -> Float[Array, "H W"]:
    """
    Multi-slice form of the CBED calculation,
    where the potential slice is in the form of
    a stack of slices, with the last dimension
    being the number of slices

    Args:
    - `pot_slice`, (Complex[Array, "H W S"]):
        The potential slice
    - `beam`, (Complex[Array, "H W"]):
        The electron beam
    - `transmission_slice` (Complex[Array, "H W"]):
        The transmission function for a single slice. You do't need to
        recalculate it then if you are doing the same CBED calculations 
        many times.

    Returns:
    - `cbed`, (Float[Array, "H W"]):
        The calculated CBED pattern
    """

    def body_fun(carry, x):
        real_space_convolve, beam = carry
        this_slice = jnp.multiply(x, beam)
        propagated_slice = jnp.fft.ifft2(
            jnp.multiply(jnp.fft.fft2(this_slice), transmission_slice)
        )
        new_real_space_convolve = jnp.multiply(real_space_convolve, propagated_slice)
        return (new_real_space_convolve, beam), None

    initial_carry = (jnp.ones_like(beam, dtype=jnp.complex64), beam)
    (real_space_convolve, _), _ = jax.lax.scan(
        body_fun, initial_carry, pot_slice.transpose(2, 0, 1)
    )

    fourier_space: Complex[Array, "H W"] = jnp.fft.fftshift(
        jnp.fft.fft2(real_space_convolve)
    )
    cbed: Float[Array, "H W"] = jnp.square(jnp.abs(fourier_space))
    return cbed


# @jaxtyped(typechecker=typechecker)
def cbed_multi_slice_multi_beam(
    pot_slice: Complex[Array, "H W S"],
    beam: Complex[Array, "H W M"],
    slice_thickness: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
) -> Float[Array, "H W"]:
    """
    Calculates the CBED pattern for multiple slices and multiple beam modes.

    This function computes the Convergent Beam Electron Diffraction (CBED) pattern
    by propagating multiple beam modes through multiple potential slices.

    Args:
    - `pot_slice` (Complex[Array, "H W S"]):
        The potential slices. H and W are height and width, S is the number of slices.
    - `beam` (Complex[Array, "H W M"]):
        The electron beam modes. M is the number of modes.
    - `slice_thickness` (Float[Array, "*"]):
        The thickness of each slice in angstroms.
    - `voltage_kV` (Float[Array, "*"]):
        The accelerating voltage in kilovolts.
    - `calib_ang` (Float[Array, "*"]):
        The calibration in angstroms.

    Returns:
    -  `cbed` (Float[Array, "H W"]):
        The calculated CBED pattern.
    """

    # Calculate the transmission function for a single slice
    slice_transmission: Complex[Array, "H W"] = pte.propagation_func(
        beam.shape[:2], slice_thickness, voltage_kV, calib_ang
    )

    def process_slice(carry, pot_slice_i):
        """
        Process a single potential slice for all beam modes.

        This function is used within lax.scan to iterate over slices.

        Args:
        - `carry`: A tuple containing the current convolution state and the beam.
        - `pot_slice_i`: A single potential slice.

        Returns:
        - A tuple containing the updated convolution state and None (scan accumulator).
        """
        convolve, _ = carry
        # Multiply the potential slice with all beam modes
        this_slice = jnp.multiply(pot_slice_i, beam)
        # Propagate the slice through free space
        propagated_slice = jnp.fft.ifft2(
            jnp.multiply(jnp.fft.fft2(this_slice), slice_transmission[:, :, None])
        )
        # Update the convolution state
        new_convolve = jnp.multiply(convolve, propagated_slice)
        return (new_convolve, beam), None

    def process_beam_mode(beam_mode):
        """
        Process all slices for a single beam mode.

        This function uses lax.scan to iterate over all slices for one beam mode.

        Args:
        - `beam_mode`: A single beam mode.

        Returns:
        - The Fourier transform of the final convolution state.
        """
        # Initialize the convolution state
        initial_carry = (jnp.ones_like(beam_mode, dtype=jnp.complex64), beam_mode)
        # Scan over all slices
        (real_space_convolve, _), _ = lax.scan(
            process_slice, initial_carry, pot_slice.transpose(2, 0, 1)
        )
        # Compute and return the Fourier transform
        return jnp.fft.fftshift(jnp.fft.fft2(real_space_convolve))

    # Use vmap to process all beam modes in parallel
    fourier_space_mode = jax.vmap(process_beam_mode, in_axes=-1, out_axes=-1)(beam)

    # Compute the intensity for each mode
    cbed_mode: Float[Array, "H W M"] = jnp.square(jnp.abs(fourier_space_mode))

    # Sum the intensities across all modes
    cbed: Float[Array, "H W"] = jnp.sum(cbed_mode, axis=-1)

    return cbed

def cbed_multi_slice_multi_beam_slice(
    pot_slice: Complex[Array, "H W S"],
    beam: Complex[Array, "H W M"],
    transmission_slice: Complex[Array, "H W"],
) -> Float[Array, "H W"]:
    """
    Calculates the CBED pattern for multiple slices and multiple beam modes.

    This function computes the Convergent Beam Electron Diffraction (CBED) pattern
    by propagating multiple beam modes through multiple potential slices.

    Args:
    - `pot_slice` (Complex[Array, "H W S"]):
        The potential slices. H and W are height and width, S is the number of slices.
    - `beam` (Complex[Array, "H W M"]):
        The electron beam modes. M is the number of modes.
    - `transmission_slice` (Complex[Array, "*"]):
        The transmission function for a single slice. You do't need to
        recalculate it then if you are doing the same CBED calculations 
        many times.

    Returns:
    -  `cbed` (Float[Array, "H W"]):
        The calculated CBED pattern.
    """

    def process_slice(carry, pot_slice_i):
        """
        Process a single potential slice for all beam modes.

        This function is used within lax.scan to iterate over slices.

        Args:
        - `carry`: A tuple containing the current convolution state and the beam.
        - `pot_slice_i`: A single potential slice.

        Returns:
        - A tuple containing the updated convolution state and None (scan accumulator).
        """
        convolve, _ = carry
        # Multiply the potential slice with all beam modes
        this_slice = jnp.multiply(pot_slice_i, beam)
        # Propagate the slice through free space
        propagated_slice = jnp.fft.ifft2(
            jnp.multiply(jnp.fft.fft2(this_slice), transmission_slice[:, :, None])
        )
        # Update the convolution state
        new_convolve = jnp.multiply(convolve, propagated_slice)
        return (new_convolve, beam), None

    def process_beam_mode(beam_mode):
        """
        Process all slices for a single beam mode.

        This function uses lax.scan to iterate over all slices for one beam mode.

        Args:
        - `beam_mode`: A single beam mode.

        Returns:
        - The Fourier transform of the final convolution state.
        """
        # Initialize the convolution state
        initial_carry = (jnp.ones_like(beam_mode, dtype=jnp.complex64), beam_mode)
        # Scan over all slices
        (real_space_convolve, _), _ = lax.scan(
            process_slice, initial_carry, pot_slice.transpose(2, 0, 1)
        )
        # Compute and return the Fourier transform
        return jnp.fft.fftshift(jnp.fft.fft2(real_space_convolve))

    # Use vmap to process all beam modes in parallel
    fourier_space_mode = jax.vmap(process_beam_mode, in_axes=-1, out_axes=-1)(beam)

    # Compute the intensity for each mode
    cbed_mode: Float[Array, "H W M"] = jnp.square(jnp.abs(fourier_space_mode))

    # Sum the intensities across all modes
    cbed: Float[Array, "H W"] = jnp.sum(cbed_mode, axis=-1)

    return cbed
