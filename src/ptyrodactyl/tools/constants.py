r"""Physical constants and derived quantities for electron microscopy.

Extended Summary
----------------
Provides derived functions for relativistic electron optics. The
physical constants live in :mod:`ptyrodactyl.types` and are
imported here for computation. Functions are JIT-compatible and
support automatic differentiation.

Routine Listings
----------------
:func:`helmholtz_coupling`
    Helmholtz potential coupling sigma_H in 1/(V·Angstrom^2).
:func:`phase_interaction_parameter`
    Phase interaction parameter sigma in rad/(V·Angstrom).
:func:`relativistic_mass`
    Relativistic electron mass in kg.
:func:`relativistic_wavelength_ang`
    Relativistic electron wavelength in Angstroms.

Notes
-----
Constants are canonical 0-dimensional weakly typed JAX arrays in
:mod:`ptyrodactyl.types`; the formulas cast them to ``jnp.float64``
inside the jitted functions to preserve the previous numerical
behavior.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from ptyrodactyl._physics import helmholtz_coupling_value
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    M_E,
    scalar_num,
)


@jax.jit
@jaxtyped(typechecker=beartype)
def relativistic_wavelength_ang(
    voltage_kv: scalar_num,
) -> Float[Array, " "]:
    r"""Relativistic electron wavelength in Angstroms.

    Extended Summary
    ----------------
    Uses the relativistic de Broglie relation:

    .. math::

        \lambda = \frac{hc}{\sqrt{eV\,(2\,m_e c^2 + eV)}}

    where :math:`V` is the accelerating voltage and :math:`m_e`
    is the electron rest mass.

    :see: :mod:`~.test_constants`

    Implementation Logic
    --------------------
    1. **Convert voltage** --
       Multiply kV by 1000 and by the elementary charge to
       obtain energy in Joules.
    2. **Relativistic formula** --
       Compute wavelength in metres from the de Broglie
       relation with relativistic kinetic energy.
    3. **Convert to Angstroms** --
       Multiply by :math:`10^{10}`.

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    lambda_ang : Float[Array, " "]
        Electron wavelength in Angstroms.

    See Also
    --------
    :func:`relativistic_mass` :
        Relativistic electron mass at the same voltage.
    :func:`phase_interaction_parameter` :
        Phase interaction parameter derived from the same voltage.

    Notes
    -----
    Uses CODATA 2018 constants.
    """
    h: Float[Array, " "] = jnp.float64(H_PLANCK)
    m: Float[Array, " "] = jnp.float64(M_E)
    e: Float[Array, " "] = jnp.float64(E_CHARGE)
    c: Float[Array, " "] = jnp.float64(C_LIGHT)

    ev: Float[Array, " "] = jnp.float64(voltage_kv) * jnp.float64(1000.0) * e
    numerator: Float[Array, " "] = jnp.square(h) * jnp.square(c)
    denominator: Float[Array, " "] = ev * (2.0 * m * jnp.square(c) + ev)
    wavelength_m: Float[Array, " "] = jnp.sqrt(numerator / denominator)
    lambda_ang: Float[Array, " "] = jnp.float64(1e10) * wavelength_m
    return lambda_ang


@jax.jit
@jaxtyped(typechecker=beartype)
def phase_interaction_parameter(
    voltage_kv: scalar_num,
) -> Float[Array, " "]:
    r"""Phase interaction parameter sigma in rad/(V·Angstrom).

    Extended Summary
    ----------------
    The phase interaction parameter relates the projected
    electrostatic potential (in V·Angstrom) to the phase shift of
    the electron wave, :math:`\Delta\phi = \sigma \int V\,dz`:

    .. math::

        \sigma = \frac{2\pi\,m\,e\,\lambda}{h^2}

    where :math:`m` is the relativistic electron mass and
    :math:`\lambda` is the relativistic wavelength, both
    evaluated at the given accelerating voltage. Note the Planck
    constant :math:`h` (not :math:`\hbar`): with :math:`\hbar` the
    result is inflated by :math:`(2\pi)^2 \approx 39.48`.

    Reference values: :math:`\sigma(100\,\mathrm{kV}) =
    0.92440\times 10^{-3}`, :math:`\sigma(300\,\mathrm{kV}) =
    0.65262\times 10^{-3}` rad/(V·Angstrom).

    :see: :class:`~.test_constants.TestPhaseInteractionParameter`

    Implementation Logic
    --------------------
    1. **Compute relativistic mass** --
       Call :func:`relativistic_mass` for :math:`m`.
    2. **Compute wavelength** --
       Call :func:`relativistic_wavelength_ang` for
       :math:`\lambda` in Angstroms, convert to metres.
    3. **Evaluate sigma** --
       Apply the formula and convert to rad/(V·Angstrom).

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    sigma : Float[Array, " "]
        Phase interaction parameter in rad/(V·Angstrom).

    See Also
    --------
    :func:`helmholtz_coupling` :
        Volumetric coupling; equals :math:`2 k_0 \sigma` with
        :math:`k_0 = 2\pi/\lambda`.
    :func:`relativistic_wavelength_ang` :
        Wavelength used in computation.
    :func:`relativistic_mass` :
        Relativistic mass used in computation.
    """
    m_rel: Float[Array, " "] = relativistic_mass(voltage_kv)
    lam_m: Float[Array, " "] = relativistic_wavelength_ang(
        voltage_kv
    ) * jnp.float64(1e-10)
    h: Float[Array, " "] = jnp.float64(H_PLANCK)
    e: Float[Array, " "] = jnp.float64(E_CHARGE)

    sigma_si: Float[Array, " "] = (
        2.0 * jnp.pi * m_rel * e * lam_m / jnp.square(h)
    )
    # Convert from rad/(V·m) to rad/(V·Å): multiply by 1e-10
    sigma: Float[Array, " "] = sigma_si * jnp.float64(1e-10)
    return sigma


@jaxtyped(typechecker=beartype)
def helmholtz_coupling(
    voltage_kv: scalar_num,
) -> Float[Array, " "]:
    r"""Helmholtz potential coupling sigma_H in 1/(V·Angstrom^2).

    Extended Summary
    ----------------
    The coupling that converts an electrostatic potential
    :math:`\phi` (in volts) into the scattering potential of the
    fixed-energy Helmholtz equation,
    :math:`\left(\nabla^2 + k_0^2 + \sigma_H\,\phi\right)\psi = 0`:

    .. math::

        \sigma_H = \frac{2\,m\,e}{\hbar^2}
                 = \frac{8\pi^2\,m_0 e}{h^2}
                   \left(1 + \frac{eU_0}{m_0 c^2}\right)

    where :math:`m` is the relativistic electron mass. The
    implementation uses the exact (2019 SI) Planck constant
    :math:`h` rather than the rounded stored :math:`\hbar`, so the
    identity :math:`\sigma_H = 2 k_0 \sigma` holds to machine
    precision against :func:`phase_interaction_parameter`. The
    wavelength cancels, so :math:`\sigma_H` is linear in the
    accelerating voltage :math:`U_0`. It relates to the phase
    interaction parameter by :math:`\sigma_H = 2 k_0 \sigma` with
    :math:`k_0 = 2\pi/\lambda`. This is the coupling consumed by
    the convergent Born series forward model.

    Reference values: :math:`\sigma_H(100\,\mathrm{kV}) = 0.31383`,
    :math:`\sigma_H(300\,\mathrm{kV}) = 0.41656`
    1/(V·Angstrom^2).

    :see: :class:`~.test_constants.TestHelmholtzCoupling`

    Implementation Logic
    --------------------
    1. **Compute relativistic mass** --
       Call :func:`relativistic_mass` for :math:`m`.
    2. **Evaluate sigma_H** --
       Apply :math:`8\pi^2 m e/h^2` in SI (1/(V·m^2)) and convert
       to 1/(V·Angstrom^2) with the factor :math:`10^{-20}`.

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    sigma_h : Float[Array, " "]
        Helmholtz potential coupling in 1/(V·Angstrom^2).

    See Also
    --------
    :func:`phase_interaction_parameter` :
        Projected-potential phase coupling; equals
        :math:`\sigma_H/(2 k_0)`.
    :func:`relativistic_mass` :
        Relativistic mass used in computation.
    """
    sigma_h: Float[Array, " "] = helmholtz_coupling_value(
        voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    return sigma_h


@jax.jit
@jaxtyped(typechecker=beartype)
def relativistic_mass(
    voltage_kv: scalar_num,
) -> Float[Array, " "]:
    r"""Relativistic electron mass in kg.

    Extended Summary
    ----------------
    Computes the relativistic mass of an electron accelerated
    through voltage :math:`V`:

    .. math::

        m = m_e\left(1 + \frac{eV}{m_e c^2}\right)

    :see: :mod:`~.test_constants`

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    m_rel : Float[Array, " "]
        Relativistic electron mass in kg.

    See Also
    --------
    :func:`relativistic_wavelength_ang` :
        Wavelength at the same voltage.
    """
    m: Float[Array, " "] = jnp.float64(M_E)
    e: Float[Array, " "] = jnp.float64(E_CHARGE)
    c: Float[Array, " "] = jnp.float64(C_LIGHT)

    ev: Float[Array, " "] = jnp.float64(voltage_kv) * jnp.float64(1000.0) * e
    m_rel: Float[Array, " "] = m * (1.0 + ev / (m * jnp.square(c)))
    return m_rel


__all__: list[str] = [
    "helmholtz_coupling",
    "phase_interaction_parameter",
    "relativistic_mass",
    "relativistic_wavelength_ang",
]
