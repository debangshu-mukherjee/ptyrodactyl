r"""Physical constants and derived quantities for electron microscopy.

Extended Summary
----------------
Provides derived functions for relativistic electron optics. The
physical constants live in :mod:`ptyrodactyl.types` and are
imported here for computation. Functions are JIT-compatible and
support automatic differentiation.

Routine Listings
----------------
:func:`interaction_parameter`
    Interaction parameter sigma in 1/(V·Angstrom).
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

from ptyrodactyl.types import (
    C_LIGHT as _C_LIGHT,
)
from ptyrodactyl.types import (
    E_CHARGE as _E_CHARGE,
)
from ptyrodactyl.types import (
    H_PLANCK as _H_PLANCK,
)
from ptyrodactyl.types import (
    HBAR as _HBAR,
)
from ptyrodactyl.types import (
    M_E as _M_E,
)
from ptyrodactyl.types import (
    scalar_num,
)


@jaxtyped(typechecker=beartype)
@jax.jit
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
    :func:`interaction_parameter` :
        Interaction parameter derived from the same voltage.

    Notes
    -----
    Uses CODATA 2018 constants.
    """
    h: Float[Array, " "] = jnp.float64(_H_PLANCK)
    m: Float[Array, " "] = jnp.float64(_M_E)
    e: Float[Array, " "] = jnp.float64(_E_CHARGE)
    c: Float[Array, " "] = jnp.float64(_C_LIGHT)

    ev: Float[Array, " "] = jnp.float64(voltage_kv) * jnp.float64(1000.0) * e
    numerator: Float[Array, " "] = jnp.square(h) * jnp.square(c)
    denominator: Float[Array, " "] = ev * (2.0 * m * jnp.square(c) + ev)
    wavelength_m: Float[Array, " "] = jnp.sqrt(numerator / denominator)
    lambda_ang: Float[Array, " "] = jnp.float64(1e10) * wavelength_m
    return lambda_ang


@jaxtyped(typechecker=beartype)
@jax.jit
def interaction_parameter(
    voltage_kv: scalar_num,
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
    voltage_kv : scalar_num
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
    hbar: Float[Array, " "] = jnp.float64(_HBAR)
    e: Float[Array, " "] = jnp.float64(_E_CHARGE)

    sigma_si: Float[Array, " "] = (
        2.0 * jnp.pi * m_rel * e * lam_m / jnp.square(hbar)
    )
    # Convert from 1/(V·m) to 1/(V·Å): multiply by 1e-10
    sigma: Float[Array, " "] = sigma_si * jnp.float64(1e-10)
    return sigma


@jaxtyped(typechecker=beartype)
@jax.jit
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
    m: Float[Array, " "] = jnp.float64(_M_E)
    e: Float[Array, " "] = jnp.float64(_E_CHARGE)
    c: Float[Array, " "] = jnp.float64(_C_LIGHT)

    ev: Float[Array, " "] = jnp.float64(voltage_kv) * jnp.float64(1000.0) * e
    m_rel: Float[Array, " "] = m * (1.0 + ev / (m * jnp.square(c)))
    return m_rel


__all__: list[str] = [
    "interaction_parameter",
    "relativistic_mass",
    "relativistic_wavelength_ang",
]
