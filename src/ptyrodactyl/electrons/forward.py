from functools import partial
from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
# from typeguard import typechecked as typechecker
from jax import Array, lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Complex, Float, Int, jaxtyped

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
        jnp.arange((-image_size[0] / 2), (image_size[0] / 2), 1)
    ) / real_fov_y  # inverse space array y
    inverse_arr_x: Float[Array, "W"] = (
        jnp.arange((-image_size[1] / 2), (image_size[1] / 2), 1)
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
    Adist = jnp.asarray(inverse_real_matrix <= LMax, dtype=jnp.complex128)
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


def cbed(
    pot_slice: Complex[Array, "H W *S"],
    beam: Complex[Array, "H W *M"],
    slice_thickness: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
) -> Float[Array, "H W"]:
    """
    Calculates the CBED pattern for single/multiple slices
    and single/multiple beam modes. This function computes
    the Convergent Beam Electron Diffraction (CBED) pattern
    by propagating one or more beam modes through one or
    more potential slices.

    Args:
    - `pot_slice` (Complex[Array, "H W *S"]):
        The potential slice(s). H and W are height and width,
        S is the number of slices (optional).
    - `beam` (Complex[Array, "H W *M"]):
        The electron beam mode(s).
        M is the number of modes (optional).
    - `slice_thickness` (Float[Array, "*"]):
        The thickness of each slice in angstroms.
    - `voltage_kV` (Float[Array, "*"]):
        The accelerating voltage in kilovolts.
    - `calib_ang` (Float[Array, "*"]):
        The calibration in angstroms.

    Returns:
    -  `cbed_pattern` (Float[Array, "H W"]):
        The calculated CBED pattern.
    """
    # Ensure 3D arrays even for single slice/mode
    pot_slice: Complex[Array, "H W *S"] = jnp.atleast_3d(pot_slice)
    beam: Complex[Array, "H W *M"] = jnp.atleast_3d(beam)

    # Calculate the transmission function for a single slice
    slice_transmission: Complex[Array, "H W"] = pte.propagation_func(
        beam.shape[0], beam.shape[1], slice_thickness, voltage_kV, calib_ang
    )

    def process_slice(
        carry: Tuple[Complex[Array, "H W *M"], Complex[Array, "H W *M"]], pot_slice_i
    ):
        convolve, _ = carry
        this_slice = jnp.multiply(pot_slice_i[:, :, None], beam)
        propagated_slice = jnp.fft.ifft2(
            jnp.multiply(jnp.fft.fft2(this_slice), slice_transmission[:, :, None])
        )
        new_convolve = jnp.multiply(convolve, propagated_slice)
        return (new_convolve, beam), None

    # Initialize the convolution state
    initial_carry = (jnp.ones_like(beam, dtype=jnp.complex128), beam)

    # Scan over all slices
    (real_space_convolve, _), _ = lax.scan(
        process_slice, initial_carry, pot_slice.transpose(2, 0, 1)
    )

    # Compute the Fourier transform
    fourier_space_modes = jnp.fft.fftshift(
        jnp.fft.fft2(real_space_convolve), axes=(0, 1)
    )

    # Compute the intensity for each mode
    cbed_mode: Float[Array, "H W M"] = jnp.square(jnp.abs(fourier_space_modes))

    # Sum the intensities across all modes
    cbed_pattern: Float[Array, "H W"] = jnp.sum(cbed_mode, axis=-1)

    return cbed_pattern


def cbed_no_slice(
    pot_slice: Complex[Array, "H W *S"],
    beam: Complex[Array, "H W *M"],
    slice_transmission: Complex[Array, "*"],
) -> Float[Array, "H W"]:
    """
    Calculates the CBED pattern for single/multiple slices
    and single/multiple beam modes.

    This function computes the Convergent Beam Electron
    Diffraction (CBED) pattern by propagating one or more
    beam modes through one or more potential slices.
    This version takes in a pre-calculated transmission
    function for going from one slice to the next, which is
    useful for calculating the CBED pattern multiple times
    where the transmission function remains the same.,
    example is 4D-STEM.

    Args:
    - `pot_slice` (Complex[Array, "H W *S"]):
        The potential slice(s). H and W are height and width,
        S is the number of slices (optional).
    - `beam` (Complex[Array, "H W *M"]):
        The electron beam mode(s).
        M is the number of modes (optional).
    - `slice_transmission` (Complex[Array, "*"]):
        The pre-calculated transmission function
        for going from one slice to the next.

    Returns:
    -  `cbed_pattern` (Float[Array, "H W"]):
        The calculated CBED pattern.
    """
    # Ensure 3D arrays even for single slice/mode
    pot_slice: Complex[Array, "H W *S"] = jnp.atleast_3d(pot_slice)
    beam: Complex[Array, "H W *M"] = jnp.atleast_3d(beam)

    def process_slice(
        carry: Tuple[Complex[Array, "H W *M"], Complex[Array, "H W *M"]], pot_slice_i
    ):
        convolve, _ = carry
        this_slice = jnp.multiply(pot_slice_i[:, :, None], beam)
        propagated_slice = jnp.fft.ifft2(
            jnp.multiply(jnp.fft.fft2(this_slice), slice_transmission[:, :, None])
        )
        new_convolve = jnp.multiply(convolve, propagated_slice)
        return (new_convolve, beam), None

    # Initialize the convolution state
    initial_carry = (jnp.ones_like(beam, dtype=jnp.complex128), beam)

    # Scan over all slices
    (real_space_convolve, _), _ = lax.scan(
        process_slice, initial_carry, pot_slice.transpose(2, 0, 1)
    )

    # Compute the Fourier transform
    fourier_space_modes = jnp.fft.fftshift(
        jnp.fft.fft2(real_space_convolve), axes=(0, 1)
    )

    # Compute the intensity for each mode
    cbed_mode: Float[Array, "H W M"] = jnp.square(jnp.abs(fourier_space_modes))

    # Sum the intensities across all modes
    cbed_pattern: Float[Array, "H W"] = jnp.sum(cbed_mode, axis=-1)

    return cbed_pattern


def shift_beam_fourier(
    beam: Complex[Array, "H W M"],
    pos: Float[Array, "... 2"],
    calib_ang: Float[Array, "*"],
) -> Complex[Array, "... H W M"]:
    """
    Shifts the beam to new position(s) using Fourier shifting.

    Args:
    - beam (Complex[Array, "H W M"]):
        The electron beam modes.
    - pos (Float[Array, "... 2"]):
        The (y, x) position(s) to shift to in pixels.
        Can be a single position [2] or multiple [..., 2].
    - calib_ang (Float[Array, "*"]):
        The calibration in angstroms.

    Returns:
    - shifted_beams (Complex[Array, "... H W M"]):
        The shifted beam(s) for all position(s) and mode(s).
    """
    H, W, M = beam.shape

    # Ensure pos is at least 2D, even for a single position
    pos = jnp.atleast_2d(pos)

    # Convert positions from real space to Fourier space
    fy = pos[..., 0] / (H * calib_ang)
    fx = pos[..., 1] / (W * calib_ang)

    # Create phase ramps in Fourier space for all positions
    y, x = jnp.meshgrid(jnp.fft.fftfreq(H), jnp.fft.fftfreq(W), indexing="ij")
    phase_ramps = jnp.exp(
        -2j * jnp.pi * (fy[..., None, None] * y + fx[..., None, None] * x)
    )

    # Apply shifts to each mode for all positions
    fft_beam = jnp.fft.fft2(beam, axes=(0, 1))
    shifted_fft_beams = fft_beam * phase_ramps[..., None]
    shifted_beams = jnp.fft.ifft2(shifted_fft_beams, axes=(-3, -2))

    return shifted_beams


@partial(jax.jit, static_argnames=["devices"])
def stem_4d(
    pot_slice: Complex[Array, "H W S"],
    beam: Complex[Array, "H W M"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, "*"],
    voltage_kV: Float[Array, "*"],
    calib_ang: Float[Array, "*"],
    devices: jax.Array,
) -> Float[Array, "P H W"]:
    """
    Calculates the 4D-STEM pattern for multiple probe positions with sharding.

    Args:
    - pot_slice (Complex[Array, "H W S"]):
        The potential slices.
    - beam (Complex[Array, "H W M"]):
        The electron beam modes.
    - pos_list (Float[Array, "... 2"]):
        List of (y, x) probe positions in pixels.
    - slice_thickness (Float[Array, "*"]):
        The thickness of each slice in angstroms.
    - voltage_kV (Float[Array, "*"]):
        The accelerating voltage in kilovolts.
    - calib_ang (Float[Array, "*"]):
        The calibration in angstroms.
    - devices (jax.Array):
        Array of devices to use for sharding.

    Returns:
    - stem_pattern (Float[Array, "... H W"]):
        The calculated 4D-STEM pattern.
    """
    # Set up device mesh
    mesh = Mesh(devices, ("devices",))

    # Define sharding strategy
    pos_sharding = P("devices", None)  # Shard along the position dimension
    beam_sharding = P(None, None, None)  # Don't shard the beam
    pot_slice_sharding = P(None, None, None)  # Don't shard the potential slices
    shifted_beams_sharding = P(
        "devices", None, None, None
    )  # Shard shifted beams along position dimension

    # Shard inputs
    with mesh:
        pos_list = jax.device_put(pos_list, pos_sharding)
        beam = jax.device_put(beam, beam_sharding)
        pot_slice = jax.device_put(pot_slice, pot_slice_sharding)

    # Calculate the transmission function once
    slice_transmission: Complex[Array, "H W"] = pte.propagation_func(
        beam.shape[0], beam.shape[1], slice_thickness, voltage_kV, calib_ang
    )

    # Shift the beam to all positions and shard the result
    with mesh:
        shifted_beams = jax.device_put(
            shift_beam_fourier(beam, pos_list, calib_ang), shifted_beams_sharding
        )

    # Calculate CBED patterns for all positions
    def calc_cbed(electron_beam):
        return cbed_no_slice(pot_slice, electron_beam, slice_transmission)

    stem_pattern = jax.vmap(calc_cbed, in_axes=0, out_axes=0)(shifted_beams)

    return stem_pattern
