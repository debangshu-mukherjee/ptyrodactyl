r"""Physical constants and derived quantities for electron microscopy.

Extended Summary
----------------
Provides fundamental physical constants as JAX float64 scalars
and derived functions for relativistic electron optics. All
constants follow CODATA 2018 recommended values. Functions are
JIT-compatible and support automatic differentiation.

Routine Listings
----------------
:data:`HBAR`
    Reduced Planck constant in J·s.
:data:`H_PLANCK`
    Planck constant in J·s.
:data:`M_E`
    Electron rest mass in kg.
:data:`E_CHARGE`
    Elementary charge in C.
:data:`C_LIGHT`
    Speed of light in m/s.
:data:`A_BOHR`
    Bohr radius in Angstroms.
:data:`M0C2_EV`
    Electron rest energy in eV.
:func:`relativistic_wavelength_ang`
    Relativistic electron wavelength in Angstroms.
:func:`interaction_parameter`
    Interaction parameter sigma in 1/(V·Angstrom).
:func:`relativistic_mass`
    Relativistic electron mass in kg.

Notes
-----
Constants are stored as Python floats and cast to
``jnp.float64`` inside functions. This avoids triggering
JAX tracing at module import time while preserving full
float64 precision in computation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import Array
from jaxtyping import Float, jaxtyped

from .electron_types import ScalarNumeric

HBAR: float = 1.054571817e-34
"""Reduced Planck constant in J·s."""

H_PLANCK: float = 6.62607015e-34
"""Planck constant in J·s."""

M_E: float = 9.1093837015e-31
"""Electron rest mass in kg."""

E_CHARGE: float = 1.602176634e-19
"""Elementary charge in C."""

C_LIGHT: float = 2.99792458e8
"""Speed of light in m/s."""

A_BOHR: float = 0.529177210903
"""Bohr radius in Angstroms."""

M0C2_EV: float = 510998.95
"""Electron rest energy in eV."""


@jaxtyped(typechecker=beartype)
@jax.jit
def relativistic_wavelength_ang(
    voltage_kv: ScalarNumeric,
) -> Float[Array, " "]:
    r"""Relativistic electron wavelength in Angstroms.

    Extended Summary
    ----------------
    Uses the relativistic de Broglie relation:

    .. math::

        \lambda = \frac{hc}{\sqrt{eV\,(2\,m_e c^2 + eV)}}

    where :math:`V` is the accelerating voltage and :math:`m_e`
    is the electron rest mass.

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
    voltage_kv : ScalarNumeric
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    lambda_ang : Float[Array, " "]
        Electron wavelength in Angstroms.

    See Also
    --------
    :func:`relativistic_mass` :
        Relativistic electron mass at the same voltage.
    :func:`interaction_parameter` :
        Interaction parameter derived from the same voltage.

    Notes
    -----
    Matches :func:`ptyrodactyl.simul.simulations.wavelength_ang`
    to machine precision. Uses CODATA 2018 constants.
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


@jaxtyped(typechecker=beartype)
@jax.jit
def interaction_parameter(
    voltage_kv: ScalarNumeric,
) -> Float[Array, " "]:
    r"""Interaction parameter sigma in 1/(V·Angstrom).

    Extended Summary
    ----------------
    The interaction parameter relates the electrostatic
    potential to the phase shift of the electron wave:

    .. math::

        \sigma = \frac{2\pi\,m\,e\,\lambda}{\hbar^2}

    where :math:`m` is the relativistic electron mass and
    :math:`\lambda` is the relativistic wavelength, both
    evaluated at the given accelerating voltage.

    Implementation Logic
    --------------------
    1. **Compute relativistic mass** --
       Call :func:`relativistic_mass` for :math:`m`.
    2. **Compute wavelength** --
       Call :func:`relativistic_wavelength_ang` for
       :math:`\lambda` in Angstroms, convert to metres.
    3. **Evaluate sigma** --
       Apply the formula and convert to 1/(V·Angstrom).

    Parameters
    ----------
    voltage_kv : ScalarNumeric
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    sigma : Float[Array, " "]
        Interaction parameter in 1/(V·Angstrom).

    See Also
    --------
    :func:`relativistic_wavelength_ang` :
        Wavelength used in computation.
    :func:`relativistic_mass` :
        Relativistic mass used in computation.
    """
    m_rel: Float[Array, " "] = relativistic_mass(voltage_kv)
    lam_m: Float[Array, " "] = relativistic_wavelength_ang(
        voltage_kv
    ) * jnp.float64(1e-10)
    hbar: Float[Array, " "] = jnp.float64(HBAR)
    e: Float[Array, " "] = jnp.float64(E_CHARGE)

    sigma_si: Float[Array, " "] = (
        2.0 * jnp.pi * m_rel * e * lam_m / jnp.square(hbar)
    )
    # Convert from 1/(V·m) to 1/(V·Å): multiply by 1e-10
    sigma: Float[Array, " "] = sigma_si * jnp.float64(1e-10)
    return sigma


@jaxtyped(typechecker=beartype)
@jax.jit
def relativistic_mass(
    voltage_kv: ScalarNumeric,
) -> Float[Array, " "]:
    r"""Relativistic electron mass in kg.

    Extended Summary
    ----------------
    Computes the relativistic mass of an electron accelerated
    through voltage :math:`V`:

    .. math::

        m = m_e\left(1 + \frac{eV}{m_e c^2}\right)

    Parameters
    ----------
    voltage_kv : ScalarNumeric
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
    "A_BOHR",
    "C_LIGHT",
    "E_CHARGE",
    "H_PLANCK",
    "HBAR",
    "M0C2_EV",
    "M_E",
    "interaction_parameter",
    "relativistic_mass",
    "relativistic_wavelength_ang",
]
