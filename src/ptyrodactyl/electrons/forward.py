import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Complex, Float, Int, Shaped


def transmission_func(
    pot_slice: Float[Array, "H W"], 
    voltage_kV: float
) -> Complex[Array, "H W"]:
    """
    Calculates the complex transmission function from
    a single potential slice at a given electron accelerating
    voltage

    Args:
    - `pot_slice`, Float[Array, "H W"]:
        potential slice in Kirkland units
    - `voltage_kV`, float:
        microscope operating voltage in kilo
        electronVolts

    Returns:
    - `trans` Complex[Array, "H W"]:
        The transmission function of a single
        crystal slice

    Flow:
    - Calculate the electron wavelength in angstroms
    - Calculate the phase shift of the electron wave
    - Calculate the transmission function
    """
    m_e: float = 9.109383e-31  # electron mass
    e_e: float = 1.602177e-19  # electron charge
    c: float = 299792458  # speed of light
    h: float = 6.62607e-34  # planck's constant
    numerator: float = (h**2) * (c**2)
    denominator: float = (e_e * voltage_kV * 1000) * (
        (2 * m_e * (c**2)) + (e_e * voltage_kV * 1000)
    )
    wavelength_ang: float = 1e10 * jnp.sqrt(
        numerator / denominator
    )  # wavelength in angstroms
    sigma: float = (
        (2 * jnp.pi / (wavelength_ang * voltage_kV * 1000))
        * ((m_e * c * c) + (e_e * voltage_kV * 1000))
    ) / ((2 * m_e * c * c) + (e_e * voltage_kV * 1000))
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


def FourierCoords(
    calibration: float, sizebeam: Int[Array, "2"]
) -> tuple[float, Float[Array, "H W"]]:
    FOV = sizebeam[0] * calibration
    qx = (jnp.arange((-sizebeam[0] / 2), ((sizebeam[0] / 2)), 1)) / FOV
    shifter = sizebeam[0] // 2
    Lx = jnp.roll(qx, shifter)
    Lya, Lxa = jnp.meshgrid(Lx, Lx)
    L2 = jnp.multiply(Lxa, Lxa) + jnp.multiply(Lya, Lya)
    L1: Float[Array, "H W"] = L2**0.5
    dL = Lx[1] - Lx[0]
    return dL, L1


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

def wavelength_ang(voltage_kV: float) -> float:
    """
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage
    
    Args:
    - `voltage_kV`, float:
        The microscope accelerating voltage in kilo
        electronVolts
    
    Returns:
    - `in_angstroms`, float:
        The electron wavelength in angstroms
        
    Flow:
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: float = 9.109383e-31  # mass of an electron
    e: float = 1.602177e-19  # charge of an electron
    c: float = 299792458.0  # speed of light
    h: float = 6.62607e-34  # Planck's constant

    voltage: float = voltage_kV * 1000
    numerator: float = (h**2) * (c**2)
    denominator: float = (e * voltage) * ((2 * m * (c**2)) + (e * voltage))
    wavelength_meters: float = jnp.sqrt(numerator / denominator)  # in meters
    in_angstroms: float = 1e10 * wavelength_meters  # in angstroms
    return in_angstroms
