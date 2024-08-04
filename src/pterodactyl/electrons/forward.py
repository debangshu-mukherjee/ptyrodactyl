import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Complex, Shape


def transmission_func(
    pot_slice: Float[Array, "H W"], voltage_kV: float
) -> Complex[Array, "H W"]:
    """
    Calculates the complex transmission function from
    a single potential slice at a given electron accelerating
    voltage

    Parameters
    ----------
    pot_slice:  Float[Array, "H W"]
                potential slice in Kirkland units
    voltage_kV: float
                microscope operating voltage in kilo
                electronVolts

    Returns
    -------
    trans: Complex[Array, "H W"]
           The transmission function of a single
           crystal slice

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    JAX adaptation and type hinting: Assistant
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
    return trans.astype(jnp.complex64)


def propagation_func(
    imsize: Shape[Array, "2"], thickness_ang: float, voltage_kV: float, calib_ang: float
) -> Complex[Array, "H W"]:
    """
    Calculates the complex propagation function that results
    in the phase shift of the exit wave when it travels from
    one slice to the next in the multislice algorithm

    Parameters
    ----------
    imsize:        Shape[Array, "2"]
                   Size of the image of the propagator
    thickness_ang: float
                   Distance between the slices in angstroms
    voltage_kV:    float
                   Accelerating voltage in kilovolts
    calib_ang:     float
                   Calibration or pixel size in angstroms

    Returns
    -------
    prop_shift:  Complex[Array, "H W"]
                 This is of the same size given by imsize

    :Authors:
    Debangshu Mukherjee <mukherjeed@ornl.gov>
    JAX adaptation and type hinting: Assistant
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
    L2: Float[Array, "H W"] = jnp.multiply(Lxa, Lxa) + jnp.multiply(Lya, Lya)
    wavelength_ang: float = wavelength_ang(voltage_kV)
    prop: Complex[Array, "H W"] = jnp.exp(
        (-1j) * jnp.pi * wavelength_ang * thickness_ang * L2
    )
    prop_shift: Complex[Array, "H W"] = jnp.fft.fftshift(
        prop
    )  # FFT shift the propagator
    return prop_shift.astype(jnp.complex64)


@jax.jit
def wavelength_ang(voltage_kV: float) -> float:
    """
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage
    """
    m: float = 9.109383e-31  # mass of an electron
    e: float = 1.602177e-19  # charge of an electron
    c: float = 299792458.0  # speed of light
    h: float = 6.62607e-34  # Planck's constant

    voltage: float = voltage_kV * 1000
    numerator: float = (h**2) * (c**2)
    denominator: float = (e * voltage) * ((2 * m * (c**2)) + (e * voltage))
    return 1e10 * jnp.sqrt(numerator / denominator)  # in angstroms
