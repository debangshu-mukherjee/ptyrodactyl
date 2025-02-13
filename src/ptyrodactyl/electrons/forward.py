from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import NamedTuple, Optional, Tuple, Union
from jax import lax
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

import ptyrodactyl.electrons as pte

jax.config.update("jax_enable_x64", True)


def transmission_func(
    pot_slice: Float[Array, "#a #b"], voltage_kV: Num[Array, ""]
) -> Complex[Array, ""]:
    """
    Description
    -----------
    Calculates the complex transmission function from
    a single potential slice at a given electron accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Parameters
    ----------
    - `pot_slice` (Float[Array, "#a #b"]):
        potential slice in Kirkland units
    - `voltage_kV` (scalar_number):
        microscope operating voltage in kilo
        electronVolts

    Returns
    -------
    - `trans` (Complex[Array, "#a #b"]):
        The transmission function of a single
        crystal slice

    Flow
    ----
    - Calculate the electron energy in electronVolts
    - Calculate the wavelength in angstroms
    - Calculate the Einstein energy
    - Calculate the sigma value, which is the constant for the phase shift
    - Calculate the transmission function as a complex exponential
    """

    voltage: Float[Array, ""] = jnp.multiply(voltage_kV, 1000.0)

    m_e: Float[Array, ""] = 9.109383e-31  # mass of an electron
    e_e: Float[Array, ""] = 1.602177e-19  # charge of an electron
    c: Float[Array, ""] = 299792458.0  # speed of light

    eV: Float[Array, ""] = jnp.multiply(e_e, voltage)
    lambda_angstrom: Float[Array, ""] = pte.wavelength_ang(
        voltage_kV
    )  # wavelength in angstroms
    einstein_energy = jnp.multiply(m_e, jnp.square(c))  # Einstein energy
    sigma: Float[Array, ""] = (
        (2 * jnp.pi / (lambda_angstrom * voltage)) * (einstein_energy + eV)
    ) / ((2 * einstein_energy) + eV)
    trans: Complex[Array, "#a #b"] = jnp.exp(1j * sigma * pot_slice)
    return trans


def propagation_func(
    imsize_y: int,
    imsize_x: int,
    thickness_ang: Num[Array, ""],
    voltage_kV: Num[Array, ""],
    calib_ang: Float[Array, ""],
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Calculates the complex propagation function that results
    in the phase shift of the exit wave when it travels from
    one slice to the next in the multislice algorithm

    Parameters
    ----------
    - `imsize_y`, (int):
        Size of the image of the propagator in y axis
    - `imsize_x`, (int):
        Size of the image of the propagator in x axis
    -  `thickness_ang`, (scalar_number):
        Distance between the slices in angstroms
    - `voltage_kV`, (scalar_number):
        Accelerating voltage in kilovolts
    - `calib_ang`, (scalar_number):
        Calibration or pixel size in angstroms

    Returns
    -------
    - `prop` (Complex[Array, "H W"]):
        The propagation function of the same size given by imsize

    Flow
    ----
    - Generate frequency arrays directly using fftfreq
    - Create 2D meshgrid of frequencies
    - Calculate squared sum of frequencies
    - Calculate wavelength
    - Compute the propagation function
    """

    # Generate frequency arrays directly using fftfreq
    qy: Num[Array, "H"] = jnp.fft.fftfreq(imsize_y, d=calib_ang)
    qx: Num[Array, "W"] = jnp.fft.fftfreq(imsize_x, d=calib_ang)

    # Create 2D meshgrid of frequencies
    Lya: Num[Array, "H W"]
    Lxa: Num[Array, "H W"]
    Lya, Lxa = jnp.meshgrid(qy, qx, indexing="ij")

    # Calculate squared sum of frequencies
    L_sq: Num[Array, "H W"] = jnp.square(Lxa) + jnp.square(Lya)

    # Calculate wavelength
    lambda_angstrom: Float[Array, ""] = pte.wavelength_ang(voltage_kV)

    # Compute the propagation function
    prop: Complex[Array, "H W"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * L_sq
    )
    return prop


def fourier_coords(calibration: float, image_size: Int[Array, "2"]) -> NamedTuple:
    """
    Description
    -----------
    Return the Fourier coordinates

    Parameters
    ----------
    - `calibration` (float):
        The pixel size in angstroms in real space
    - `image_size`, (Int[Array, "2"]):
        The size of the beam in pixels

    Returns
    -------
    - A NamedTuple with the following fields:
        - `array` (Any[Array, "* *"]):
            The array values
        - `calib_y` (float):
            Calibration along the first axis
        - `calib_x` (float):
            Calibration along the second axis

    Flow
    ----
    - Calculate the real space field of view in y and x
    - Generate the inverse space array y and x
    - Shift the inverse space array y and x
    - Create meshgrid of shifted inverse space arrays
    - Calculate the inverse array
    - Calculate the calibration in y and x
    - Return the calibrated array
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
        [("array", Num[Array, "* *"]), ("calib_y", float), ("calib_x", float)],
    )
    return calibrated_array(inverse_array, calib_inverse_y, calib_inverse_x)


def fourier_calib(
    real_space_calib: float | Float[Array, "*"],
    sizebeam: Int[Array, "2"],
) -> Float[Array, "2"]:
    """
    Description
    -----------
    Generate the Fourier calibration for the beam

    Parameters
    ----------
    - `real_space_calib` (float | Float[Array, "*"]):
        The pixel size in angstroms in real space
    - `sizebeam` (Int[Array, "2"]):
        The size of the beam in pixels

    Returns
    -------
    - `inverse_space_calib` (Float[Array, "2"]):
        The Fourier calibration in angstroms

    Flow
    ----
    - Calculate the field of view in real space
    - Calculate the inverse space calibration
    """
    field_of_view: Float[Array, "*"] = jnp.multiply(
        jnp.float64(sizebeam), real_space_calib
    )
    inverse_space_calib = 1 / field_of_view
    return inverse_space_calib


@jax.jit
def make_probe(
    aperture: Num[Array, ""],
    voltage: Union[float, int],
    image_size: Int[Array, "2"],
    calibration_pm: float,
    defocus: float = 0,
    c3: float = 0,
    c5: float = 0,
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    This calculates an electron probe based on the
    size and the estimated Fourier co-ordinates with
    the option of adding spherical aberration in the
    form of defocus, C3 and C5

    Parameters
    ----------
    - `aperture` (Union[float, int]):
        The aperture size in milliradians
    - `voltage` (Union[float, int]):
        The microscope accelerating voltage in kilo
        electronVolts
    - `image_size`, (Int[Array, "2"]):
        The size of the beam in pixels
    - `calibration_pm` (float):
        The calibration in picometers
    - `defocus` (float):
        The defocus value in angstroms
    - `c3` (float):
        The C3 value in angstroms
    - `c5` (float):
        The C5 value in angstroms

    Returns
    -------
    - `probe_real_space` (Complex[Array, "H W"]):
        The calculated electron probe in real space

    Flow
    ----
    - Convert the aperture to radians
    - Calculate the wavelength in angstroms
    - Calculate the maximum L value
    - Calculate the field of view in x and y
    - Generate the inverse space array y and x
    - Shift the inverse space array y and x
    - Create meshgrid of shifted inverse space arrays
    - Calculate the inverse array
    - Calculate the calibration in y and x
    - Calculate the probe in real space
    """
    aperture = aperture / 1000
    wavelength = pte.wavelength_ang(voltage)
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
    chi_probe = pte.aberration(inverse_real_matrix, wavelength, defocus, c3, c5)
    Adist *= jnp.exp(-1j * chi_probe)
    probe_real_space = jnp.fft.ifftshift(jnp.fft.ifft2(Adist))
    return probe_real_space


@jax.jit
def aberration(
    fourier_coord: Float[Array, "H W"],
    lambda_angstrom: Num[Array, ""],
    defocus: Optional[Float[Array, ""]] = jnp.asarray(0.0),
    c3: Optional[Float[Array, ""]] = jnp.asarray(0.0),
    c5: Optional[Float[Array, ""]] = jnp.asarray(0.0),
) -> Float[Array, "H W"]:
    """
    Description
    -----------
    This calculates the aberration function for the
    electron probe based on the Fourier co-ordinates

    Parameters
    ----------
    - `fourier_coord` (Float[Array, "H W"]):
        The Fourier co-ordinates
    - `lambda_angstrom` (Num[Array, ""]):
        The wavelength in angstroms
    - `defocus` (Float[Array, ""]):
        The defocus value in angstroms.
        Default is 0.0
    - `c3` (Float[Array, ""]):
        The C3 value in angstroms.
        Default is 0.0
    - `c5` (Float[Array, ""]):
        The C5 value in angstroms.
        Default is 0.0

    Returns
    -------
    - `chi_probe` (Float[Array, "H W"]):
        The calculated aberration function

    Flow
    ----
    - Calculate the phase shift
    - Calculate the chi value
    - Calculate the chi probe value
    """
    p_matrix = lambda_angstrom * fourier_coord
    chi: Float[Array, "H W"] = (
        ((defocus * jnp.power(p_matrix, 2)) / 2)
        + ((c3 * (1e7) * jnp.power(p_matrix, 4)) / 4)
        + ((c5 * (1e7) * jnp.power(p_matrix, 6)) / 6)
    )
    chi_probe: Float[Array, "H W"] = (2 * jnp.pi * chi) / lambda_angstrom
    return chi_probe


@jaxtyped(typechecker=typechecker)
def wavelength_ang(voltage_kV: Num[Array, "#a"]) -> Float[Array, "#a"]:
    """
    Description
    -----------
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Parameters
    ----------
    - `voltage_kV` (num_type | Float[Array, "#a"]):
        The microscope accelerating voltage in kilo
        electronVolts

    Returns
    -------
    - `in_angstroms (Float[Array, "*"]):
        The electron wavelength in angstroms

    Flow
    ----
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: float = 9.109383e-31  # mass of an electron
    e: float = 1.602177e-19  # charge of an electron
    c: float = 299792458.0  # speed of light
    h: float = 6.62607e-34  # Planck's constant

    eV: Float[Array, "#a"] = (
        jnp.float64(voltage_kV) * jnp.float64(1000.0) * jnp.float64(e)
    )
    numerator: Float[Array, ""] = jnp.float64(h * c)
    denominator: Float[Array, "#a"] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, "#a"] = jnp.sqrt(
        numerator / denominator
    )  # in meters
    lambda_angstroms: Float[Array, "#a"] = 1e10 * wavelength_meters  # in angstroms
    return lambda_angstroms


def cbed(
    pot_slice: Complex[Array, "H W #S"],
    beam: Complex[Array, "H W #M"],
    slice_thickness: Float[Array, ""],
    voltage_kV: Float[Array, "#v"],
    calib_ang: Float[Array, ""],
) -> Float[Array, "H W"]:
    """
    Description
    -----------
    Calculates the CBED pattern for single/multiple slices
    and single/multiple beam modes. This function computes
    the Convergent Beam Electron Diffraction (CBED) pattern
    by propagating one or more beam modes through one or
    more potential slices.

    Parameters
    ----------
    - `pot_slice` (Complex[Array, "H W #S"]):
        The potential slice(s). H and W are height and width,
        S is the number of slices (optional).
    - `beam` (Complex[Array, "H W #M"]):
        The electron beam mode(s).
        M is the number of modes (optional).
    - `slice_thickness` (Float[Array, ""]):
        The thickness of each slice in angstroms.
    - `voltage_kV` (Float[Array, "#v"]):
        The accelerating voltage(s) in kilovolts.
    - `calib_ang` (Float[Array, ""]):
        The calibration in angstroms.

    Returns
    -------
    -  `cbed_pattern` (Float[Array, "H W"]):
        The calculated CBED pattern.

    Flow
    ----
    - Ensure 3D arrays even for single slice/mode
    - Calculate the transmission function for a single slice
    - Initialize the convolution state
    - Scan over all slices
    - Compute the Fourier transform
    - Compute the intensity for each mode
    - Sum the intensities across all modes.
    """
    # Ensure 3D arrays even for single slice/mode
    pot_slice: Complex[Array, "H W #S"] = jnp.atleast_3d(pot_slice)
    beam: Complex[Array, "H W #M"] = jnp.atleast_3d(beam)

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
    Description
    -----------
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

    Parameters
    ----------
    - `pot_slice` (Complex[Array, "H W *S"]):
        The potential slice(s). H and W are height and width,
        S is the number of slices (optional).
    - `beam` (Complex[Array, "H W *M"]):
        The electron beam mode(s).
        M is the number of modes (optional).
    - `slice_transmission` (Complex[Array, "*"]):
        The pre-calculated transmission function
        for going from one slice to the next.

    Returns
    -------
    -  `cbed_pattern` (Float[Array, "H W"]):
        The calculated CBED pattern.

    Flow
    ----
    - Ensure 3D arrays even for single slice/mode
    - Initialize the convolution state
    - Scan over all slices
    - Compute the Fourier transform
    - Compute the intensity for each mode
    - Sum the intensities across all modes
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
    Description
    -----------
    Shifts the beam to new position(s) using Fourier shifting.

    Parameters
    ----------
    - beam (Complex[Array, "H W M"]):
        The electron beam modes.
    - pos (Float[Array, "... 2"]):
        The (y, x) position(s) to shift to in pixels.
        Can be a single position [2] or multiple [..., 2].
    - calib_ang (Float[Array, "*"]):
        The calibration in angstroms.

    Returns
    -------
    - shifted_beams (Complex[Array, "... H W M"]):
        The shifted beam(s) for all position(s) and mode(s).

    Flow
    ----
    - Convert positions from real space to Fourier space
    - Create phase ramps in Fourier space for all positions
    - Apply shifts to each mode for all positions
    """
    H, W, _ = beam.shape

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


@partial(jax.jit, static_argnames=["slice_thickness", "voltage_kV", "calib_ang"])
def stem_4d(
    pot_slice: Complex[Array, "H W S"],
    beam: Complex[Array, "H W M"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: float,
    voltage_kV: float,
    calib_ang: float,
) -> Float[Array, "P H W"]:
    """
    Description
    -----------
    Calculates the 4D-STEM pattern for multiple probe positions with sharding.

    Parameters
    ----------
    - `pot_slice` (Complex[Array, "H W S"]):
        The potential slices.
    - `beam` (Complex[Array, "H W M"]):
        The electron beam modes.
    - `pos_list` (Float[Array, "P 2"]):
        List of (y, x) probe positions in pixels.
    - `slice_thickness` (float):
        The thickness of each slice in angstroms.
    - `voltage_kV` (float):
        The accelerating voltage in kilovolts.
    - `calib_ang` (float):
        The calibration in angstroms.
    - `mesh` (Mesh):
        The device mesh for sharding.

    Returns
    -------
    - stem_pattern (Float[Array, "P H W"]): The calculated 4D-STEM pattern.

    Flow
    ----
    - Calculate the transmission function once
    - Shift the beam to all positions
    - Calculate CBED patterns for all positions
    """
    # Calculate the transmission function once
    slice_transmission: Complex[Array, "H W"] = pte.propagation_func(
        beam.shape[0], beam.shape[1], slice_thickness, voltage_kV, calib_ang
    )

    # Shift the beam to all positions
    shifted_beams = pte.shift_beam_fourier(beam, pos_list, calib_ang)

    # Calculate CBED patterns for all positions
    def calc_cbed(electron_beam):
        return cbed_no_slice(pot_slice, electron_beam, slice_transmission)

    stem_pattern = jax.vmap(calc_cbed, in_axes=0, out_axes=0)(shifted_beams)

    return stem_pattern


def stem_4d_multi(
    pot_slices: Complex[Array, "H W S"],
    beam: Complex[Array, "H W M"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, ""],
    voltage_kV: Float[Array, ""],
    calib_ang: Float[Array, ""],
) -> Float[Array, "P H W"]:
    """
    Calculates 4D-STEM pattern for multiple slices.

    This function propagates the beam through multiple slices
    before calculating the final diffraction pattern.
    """
    # Calculate transmission function once (same for all slices)
    slice_transmission = pte.propagation_func(
        beam.shape[0], beam.shape[1], slice_thickness, voltage_kV, calib_ang
    )

    # Shift beam to all positions
    shifted_beams = pte.shift_beam_fourier(beam, pos_list, calib_ang)

    # Calculate patterns for all positions
    def calc_multi_slice_cbed(electron_beam):
        # Start with initial beam
        wave = electron_beam

        # Propagate through all slices
        for i in range(pot_slices.shape[-1]):
            # Apply transmission function for current slice
            wave = wave * pot_slices[..., i : i + 1]

            # Propagate to next slice
            if i < pot_slices.shape[-1] - 1:  # Don't propagate after last slice
                wave = jnp.fft.ifft2(jnp.fft.fft2(wave) * slice_transmission[..., None])

        # Calculate final diffraction pattern
        fourier = jnp.fft.fftshift(jnp.fft.fft2(wave), axes=(0, 1))
        intensity = jnp.sum(jnp.abs(fourier) ** 2, axis=-1)
        return intensity

    # Apply to all probe positions
    stem_pattern = jax.vmap(calc_multi_slice_cbed, in_axes=0, out_axes=0)(shifted_beams)

    return stem_pattern


def stem_4d_mixed_state(
    pot_slices: Complex[Array, "H W S"],
    modes: Complex[Array, "H W M"],
    mode_weights: Float[Array, "M"],
    pos_list: Float[Array, "P 2"],
    slice_thickness: Float[Array, ""],
    voltage_kV: Float[Array, ""],
    calib_ang: Float[Array, ""],
) -> Float[Array, "P H W"]:
    """
    Calculates 4D-STEM pattern for multiple modes and slices with mixed state.
    """
    # Calculate transmission function
    slice_transmission = pte.propagation_func(
        modes.shape[0], modes.shape[1], slice_thickness, voltage_kV, calib_ang
    )

    # Shift all modes to all positions
    shifted_modes = pte.shift_beam_fourier(modes, pos_list, calib_ang)

    def calc_mixed_state_cbed(shifted_mode_set):
        # Process each mode
        def process_mode(mode):
            wave = mode
            # Propagate through slices
            for i in range(pot_slices.shape[-1]):
                wave = wave * pot_slices[..., i]
                if i < pot_slices.shape[-1] - 1:
                    wave = jnp.fft.ifft2(jnp.fft.fft2(wave) * slice_transmission)
            return wave

        # Apply to all modes
        waves = jax.vmap(process_mode)(shifted_mode_set)

        # Calculate diffraction patterns for all modes
        fourier = jnp.fft.fftshift(jnp.fft.fft2(waves), axes=(0, 1))
        mode_intensities = jnp.abs(fourier) ** 2

        # Weight and sum the mode intensities
        weighted_sum = jnp.sum(mode_intensities * mode_weights[:, None, None], axis=0)
        return weighted_sum

    # Apply to all probe positions
    stem_pattern = jax.vmap(calc_mixed_state_cbed)(shifted_modes)

    return stem_pattern


def initialize_random_modes(
    shape: Tuple[int, int], num_modes: int, dtype=jnp.complex128
) -> Complex[Array, "H W M"]:
    """Initialize random orthogonal modes."""
    key = jax.random.PRNGKey(0)
    modes = jax.random.normal(key, (shape[0], shape[1], num_modes), dtype=dtype)

    # Orthogonalize modes using QR decomposition
    modes_flat = modes.reshape(-1, num_modes)
    q, r = jnp.linalg.qr(modes_flat)
    modes = q.reshape(shape[0], shape[1], num_modes)

    return modes


def normalize_mode_weights(weights: Float[Array, "M"]) -> Float[Array, "M"]:
    """Normalize mode weights to sum to 1."""
    return weights / jnp.sum(weights)
