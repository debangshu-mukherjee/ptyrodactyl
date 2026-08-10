"""Physical constants and derived electron-optics quantities.

Extended Summary
----------------
This module defines the shared physical constants used by
ptyrodactyl's electron microscopy calculations. Constants are
materialized as **0-dimensional, weak-typed JAX arrays** at import
time: they are created from Python float literals (never with an
explicit ``dtype=``), so JAX preserves weak typing and the constants
promote exactly like Python scalars (``HBAR * x`` leaves a
``float32``/``complex64`` array's dtype untouched). The package
``__init__`` enables ``jax_enable_x64`` before any submodule import,
so these constants always materialize as ``float64``; the import-time
guard below turns any future ordering regression into a loud
``ImportError`` instead of silently truncated ``float32`` physics.

Because they are arrays, these constants are **not hashable** and must
never be used as JIT static arguments. The derived functions implement
relativistic electron-optics formulas using these canonical constants;
they are JIT-compatible and support automatic differentiation.

Routine Listings
----------------
:func:`helmholtz_coupling`
    Compute the Helmholtz potential coupling in 1/(V·Angstrom²).
:func:`phase_interaction_parameter`
    Compute the phase interaction parameter in rad/(V·Angstrom).
:func:`relativistic_mass`
    Compute the relativistic electron mass in kg.
:func:`relativistic_wavelength_ang`
    Compute the relativistic electron wavelength in Angstroms.
:obj:`A_BOHR`
    Bohr radius in Angstroms.
:obj:`C_LIGHT`
    Speed of light in m/s.
:obj:`E_CHARGE`
    Elementary charge in C.
:obj:`H_PLANCK`
    Planck constant in J·s.
:obj:`HBAR`
    Reduced Planck constant in J·s.
:obj:`M0C2_EV`
    Electron rest energy in eV.
:obj:`M_E`
    Electron rest mass in kg.
:obj:`MOTT_BETHE_VOLT_ANGSTROM_SQ`
    Mott-Bethe constant h²/(2π m₀ e) in V·Å².

"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Final
from jax import Array
from jaxtyping import Float64, jaxtyped

from ptyrodactyl._physics import helmholtz_coupling_value

from .custom_types import scalar_float, scalar_num

HBAR: Final[scalar_float] = jnp.asarray(1.054571817e-34)
"""Reduced Planck constant in J·s.

:see: :class:`~.test_constants.TestConstantsContract`
"""

H_PLANCK: Final[scalar_float] = jnp.asarray(6.62607015e-34)
"""Planck constant in J·s.

:see: :mod:`~.test_constants`
"""

M_E: Final[scalar_float] = jnp.asarray(9.1093837015e-31)
"""Electron rest mass in kg.

:see: :mod:`~.test_constants`
"""

E_CHARGE: Final[scalar_float] = jnp.asarray(1.602176634e-19)
"""Elementary charge in C.

:see: :mod:`~.test_constants`
"""

C_LIGHT: Final[scalar_float] = jnp.asarray(2.99792458e8)
"""Speed of light in m/s.

:see: :mod:`~.test_constants`
"""

A_BOHR: Final[scalar_float] = jnp.asarray(0.529177210903)
"""Bohr radius in Angstroms.

:see: :class:`~.test_constants.TestConstantsContract`
"""

M0C2_EV: Final[scalar_float] = jnp.asarray(510998.95)
"""Electron rest energy in eV.

:see: :mod:`~.test_constants`
"""

MOTT_BETHE_VOLT_ANGSTROM_SQ: Final[scalar_float] = jnp.asarray(47.87801)
"""Mott-Bethe constant h²/(2π m₀ e) in V·Å² (potential Fourier convention).

:see: :mod:`~.test_constants`
"""

if HBAR.dtype != jnp.float64:
    raise ImportError(
        "ptyrodactyl.types.constants materialized under float32: "
        "jax_enable_x64 must be set before this module is imported "
        "(ptyrodactyl/__init__.py owns that ordering — see CONTRIBUTING)."
    )


@jaxtyped(typechecker=beartype)
def helmholtz_coupling(
    voltage_kv: scalar_num,
) -> Float64[Array, " "]:
    r"""Compute the Helmholtz potential coupling in 1/(V·Angstrom²).

    Extended Summary
    ----------------
    The coupling converts an electrostatic potential :math:`\phi` in volts
    into the scattering potential of the fixed-energy Helmholtz equation:

    .. math::

        \sigma_H = \frac{2 m e}{\hbar^2}
                 = \frac{8\pi^2 m_0 e}{h^2}
                   \left(1 + \frac{eU_0}{m_0 c^2}\right).

    The implementation uses the exact (2019 SI) Planck constant :math:`h`,
    so :math:`\sigma_H = 2 k_0 \sigma` holds to machine precision against
    :func:`phase_interaction_parameter`.

    :see: :class:`~.test_constants.TestHelmholtzCoupling`

    Implementation Logic
    --------------------
    1. **Evaluate the coupling** -- Pass the accelerating voltage and
       canonical constants to the shared fixed-precision scalar primitive.

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    sigma_h : Float64[Array, " "]
        Helmholtz potential coupling in 1/(V·Angstrom²).

    Notes
    -----
    The wavelength cancels, so the coupling is affine in the accelerating
    voltage. Reference values are 0.31383 at 100 kV and 0.41656 at 300 kV,
    in 1/(V·Angstrom²).

    See Also
    --------
    :func:`phase_interaction_parameter` :
        Compute the corresponding projected-potential phase coupling.
    :func:`relativistic_mass` :
        Compute the relativistic mass used by the coupling.
    """
    sigma_h: Float64[Array, " "] = helmholtz_coupling_value(
        voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    return sigma_h


@jax.jit
@jaxtyped(typechecker=beartype)
def phase_interaction_parameter(
    voltage_kv: scalar_num,
) -> Float64[Array, " "]:
    r"""Compute the phase interaction parameter in rad/(V·Angstrom).

    Extended Summary
    ----------------
    Relate projected electrostatic potential to electron-wave phase through
    :math:`\Delta\phi = \sigma \int V\,dz`, where

    .. math::

        \sigma = \frac{2\pi m e \lambda}{h^2}.

    Here :math:`m` and :math:`\lambda` are evaluated at the accelerating
    voltage. The formula uses :math:`h`, not :math:`\hbar`; substituting
    :math:`\hbar` inflates the result by :math:`(2\pi)^2`.

    :see: :class:`~.test_constants.TestPhaseInteractionParameter`

    Implementation Logic
    --------------------
    1. **Compute relativistic quantities** -- Evaluate the mass in kilograms
       and the wavelength in Angstroms, then convert wavelength to metres.
    2. **Evaluate and convert** -- Apply the SI formula and convert the result
       from rad/(V·m) to rad/(V·Angstrom).

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    sigma : Float64[Array, " "]
        Phase interaction parameter in rad/(V·Angstrom).

    Notes
    -----
    Reference values at 100 and 300 kV are respectively
    :math:`0.92440\times10^{-3}` and :math:`0.65262\times10^{-3}`
    rad/(V·Angstrom).

    See Also
    --------
    :func:`helmholtz_coupling` :
        Compute the volumetric coupling :math:`2k_0\sigma`.
    :func:`relativistic_mass` :
        Compute the mass used in the phase parameter.
    :func:`relativistic_wavelength_ang` :
        Compute the wavelength used in the phase parameter.
    """
    m_rel: Float64[Array, " "] = relativistic_mass(voltage_kv)
    lam_m: Float64[Array, " "] = relativistic_wavelength_ang(
        voltage_kv
    ) * jnp.float64(1e-10)
    h: Float64[Array, " "] = jnp.float64(H_PLANCK)
    e: Float64[Array, " "] = jnp.float64(E_CHARGE)

    sigma_si: Float64[Array, " "] = (
        2.0 * jnp.pi * m_rel * e * lam_m / jnp.square(h)
    )
    sigma: Float64[Array, " "] = sigma_si * jnp.float64(1e-10)
    return sigma


@jax.jit
@jaxtyped(typechecker=beartype)
def relativistic_mass(
    voltage_kv: scalar_num,
) -> Float64[Array, " "]:
    r"""Compute the relativistic electron mass in kg.

    Extended Summary
    ----------------
    Evaluate the relativistic mass of an electron accelerated through
    voltage :math:`V`:

    .. math::

        m = m_e\left(1 + \frac{eV}{m_e c^2}\right).

    :see: :class:`~.test_constants.TestRelativisticMass`

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    m_rel : Float64[Array, " "]
        Relativistic electron mass in kg.

    See Also
    --------
    :func:`relativistic_wavelength_ang` :
        Compute the wavelength at the same accelerating voltage.
    """
    m: Float64[Array, " "] = jnp.float64(M_E)
    e: Float64[Array, " "] = jnp.float64(E_CHARGE)
    c: Float64[Array, " "] = jnp.float64(C_LIGHT)

    ev: Float64[Array, " "] = jnp.float64(voltage_kv) * jnp.float64(1000.0) * e
    m_rel: Float64[Array, " "] = m * (1.0 + ev / (m * jnp.square(c)))
    return m_rel


@jax.jit
@jaxtyped(typechecker=beartype)
def relativistic_wavelength_ang(
    voltage_kv: scalar_num,
) -> Float64[Array, " "]:
    r"""Compute the relativistic electron wavelength in Angstroms.

    Extended Summary
    ----------------
    Use the relativistic de Broglie relation

    .. math::

        \lambda = \frac{hc}{\sqrt{eV\,(2m_e c^2 + eV)}},

    where :math:`V` is the accelerating voltage.

    :see: :class:`~.test_constants.TestRelativisticWavelength`

    Implementation Logic
    --------------------
    1. **Convert voltage** -- Convert kilovolts to electron kinetic energy
       in joules using the elementary charge.
    2. **Evaluate and convert** -- Evaluate the de Broglie wavelength in
       metres and convert it to Angstroms.

    Parameters
    ----------
    voltage_kv : scalar_num
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    lambda_ang : Float64[Array, " "]
        Electron wavelength in Angstroms.

    Notes
    -----
    Uses the exact CODATA 2018 values stored in this module.

    See Also
    --------
    :func:`phase_interaction_parameter` :
        Compute the phase interaction parameter at the same voltage.
    :func:`relativistic_mass` :
        Compute the relativistic mass at the same voltage.
    """
    h: Float64[Array, " "] = jnp.float64(H_PLANCK)
    m: Float64[Array, " "] = jnp.float64(M_E)
    e: Float64[Array, " "] = jnp.float64(E_CHARGE)
    c: Float64[Array, " "] = jnp.float64(C_LIGHT)

    ev: Float64[Array, " "] = jnp.float64(voltage_kv) * jnp.float64(1000.0) * e
    numerator: Float64[Array, " "] = jnp.square(h) * jnp.square(c)
    denominator: Float64[Array, " "] = ev * (2.0 * m * jnp.square(c) + ev)
    wavelength_m: Float64[Array, " "] = jnp.sqrt(numerator / denominator)
    lambda_ang: Float64[Array, " "] = jnp.float64(1e10) * wavelength_m
    return lambda_ang


__all__: list[str] = [
    "A_BOHR",
    "C_LIGHT",
    "E_CHARGE",
    "H_PLANCK",
    "HBAR",
    "M0C2_EV",
    "MOTT_BETHE_VOLT_ANGSTROM_SQ",
    "M_E",
    "helmholtz_coupling",
    "phase_interaction_parameter",
    "relativistic_mass",
    "relativistic_wavelength_ang",
]
