from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Complex, Float, Int, Shaped, jaxtyped
from typeguard import typechecked as typechecker

import ptyrodactyl.electrons as pte

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=typechecker)
def transmission_func(
    pot_slice: Float[Array, "H W"], voltage_kV: int | float | Float[Array, "*"]
) -> Complex[Array, "H W"]:
    """
    Calculates the complex transmission function from
    a single potential slice at a given electron accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Args:
    - `pot_slice`, Float[Array, "H W"]:
        potential slice in Kirkland units
    - `voltage_kV`, int | float | Float[Array, "*"]:
        microscope operating voltage in kilo
        electronVolts

    Returns:
    - `trans` Complex[Array, "H W"]:
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
    trans: Complex[Array, "H W"] = jnp.exp(1j * sigma * pot_slice)
    return trans


def propagation_func(
    imsize: Shaped[Array, "2"],
    thickness_ang: float,
    voltage_kV: float,
    calib_ang: float,
) -> Complex[Array, "H W"]:
    """
    Calculates the complex propagation function that results
    in the phase shift of the exit wave when it travels from
    one slice to the next in the multislice algorithm

    Args:
    - `imsize`, Shaped[Array, "2"]:
        Size of the image of the propagator
    -  `thickness_ang`, float
        Distance between the slices in angstroms
    - `voltage_kV`, float
        Accelerating voltage in kilovolts
    - `calib_ang`, float
        Calibration or pixel size in angstroms

    Returns:
    - `prop_shift` Complex[Array, "H W"]:
        This is of the same size given by imsize

    Flow:

    """
    FOV_y: float = imsize[0] * calib_ang
    FOV_x: float = imsize[1] * calib_ang
    qy: Float[Array, "H"] = (jnp.arange((-imsize[0] / 2), ((imsize[0] / 2)), 1)) / FOV_y
    qx: Float[Array, "W"] = (jnp.arange((-imsize[1] / 2), ((imsize[1] / 2)), 1)) / FOV_x
    shifter_y: int = imsize[0] // 2
    shifter_x: int = imsize[1] // 2
    Ly: Float[Array, "H"] = jnp.roll(qy, shifter_y)
    Lx: Float[Array, "W"] = jnp.roll(qx, shifter_x)
    Lya, Lxa = jnp.meshgrid(Lx, Ly)
    L_sq: Float[Array, "H W"] = jnp.multiply(Lxa, Lxa) + jnp.multiply(Lya, Lya)
    lambda_angstrom: float = wavelength_ang(voltage_kV)
    prop: Complex[Array, "H W"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * L_sq
    )
    prop_shift: Complex[Array, "H W"] = jnp.fft.fftshift(
        prop
    )  # FFT shift the propagator
    return prop_shift


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


def FourierCalib(calibration: float, sizebeam: Int[Array, "2"]) -> Float[Array, "2"]:
    FOV_y = sizebeam[0] * calibration
    FOV_x = sizebeam[1] * calibration
    qy = (jnp.arange((-sizebeam[0] / 2), ((sizebeam[0] / 2)), 1)) / FOV_y
    qx = (jnp.arange((-sizebeam[1] / 2), ((sizebeam[1] / 2)), 1)) / FOV_x
    shifter_y = sizebeam[0] // 2
    shifter_x = sizebeam[1] // 2
    Ly = jnp.roll(qy, shifter_y)
    Lx = jnp.roll(qx, shifter_x)
    dL_y = Ly[1] - Ly[0]
    dL_x = Lx[1] - Lx[0]
    return jnp.array([dL_y, dL_x])


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
