"""Provide dependency-neutral scalar physics primitives.

Extended Summary
----------------
This private leaf evaluates the canonical relativistic Helmholtz coupling
and applies its declared precision reduction to voltage coefficients.

Routine Listings
----------------
:func:`coupled_interaction_value`
    Evaluate and canonically round the coupling and coupled coefficients.
:func:`helmholtz_coupling_value`
    Evaluate the 50-mantissa-bit canonical Helmholtz coupling.
"""

import jax.numpy as jnp
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, Complex, Complex128, Float, Float64


def helmholtz_coupling_value(
    voltage_kv: Float[Array, ""] | float,
    electron_mass: Float[Array, ""] | float,
    elementary_charge: Float[Array, ""] | float,
    light_speed: Float[Array, ""] | float,
    planck_constant: Float[Array, ""] | float,
) -> Float64[Array, ""]:
    """Evaluate the 50-mantissa-bit canonical Helmholtz coupling.

    Parameters
    ----------
    voltage_kv : Float[Array, ""] | float
        Accelerating voltage in kilovolts.
    electron_mass : Float[Array, ""] | float
        Electron rest mass in kilograms.
    elementary_charge : Float[Array, ""] | float
        Positive elementary charge in coulombs.
    light_speed : Float[Array, ""] | float
        Speed of light in metres per second.
    planck_constant : Float[Array, ""] | float
        Planck constant in joule seconds.

    Returns
    -------
    coupling : Float64[Array, ""]
        Canonically rounded Helmholtz coupling in inverse square angstroms
        per volt.
    """
    mass: Float64[Array, ""] = jnp.float64(electron_mass)
    charge: Float64[Array, ""] = jnp.float64(elementary_charge)
    speed: Float64[Array, ""] = jnp.float64(light_speed)
    planck: Float64[Array, ""] = jnp.float64(planck_constant)
    voltage: Float64[Array, ""] = jnp.float64(voltage_kv)
    energy: Float64[Array, ""] = voltage * jnp.float64(1000.0) * charge
    relativistic_mass: Float64[Array, ""] = mass * (
        1.0 + energy / (mass * jnp.square(speed))
    )
    coupling_si: Float64[Array, ""] = (
        8.0
        * jnp.square(jnp.pi)
        * relativistic_mass
        * charge
        / jnp.square(planck)
    )
    raw_coupling: Float64[Array, ""] = coupling_si * jnp.float64(1.0e-20)
    rounded_coupling: Float64[Array, ""] = lax.reduce_precision(
        raw_coupling,
        exponent_bits=11,
        mantissa_bits=50,
    )
    coupling: Float64[Array, ""] = lax.optimization_barrier(rounded_coupling)
    return coupling


def coupled_interaction_value(
    voltage_coefficients: Complex[Array, " p"],
    voltage_kv: Float[Array, ""] | float,
    electron_mass: Float[Array, ""] | float,
    elementary_charge: Float[Array, ""] | float,
    light_speed: Float[Array, ""] | float,
    planck_constant: Float[Array, ""] | float,
) -> Tuple[Float64[Array, ""], Complex128[Array, " p"]]:
    """Evaluate and canonically round the coupling and coupled coefficients.

    Parameters
    ----------
    voltage_coefficients : Complex[Array, " p"]
        Voltage coefficients in volts.
    voltage_kv : Float[Array, ""] | float
        Accelerating voltage in kilovolts.
    electron_mass : Float[Array, ""] | float
        Electron rest mass in kilograms.
    elementary_charge : Float[Array, ""] | float
        Positive elementary charge in coulombs.
    light_speed : Float[Array, ""] | float
        Speed of light in metres per second.
    planck_constant : Float[Array, ""] | float
        Planck constant in joule seconds.

    Returns
    -------
    coupling : Float64[Array, ""]
        Canonically rounded Helmholtz coupling in inverse square angstroms
        per volt.
    interaction : Complex128[Array, " p"]
        Canonically rounded coupled coefficients in inverse square angstroms.
    """
    coupling: Float64[Array, ""] = helmholtz_coupling_value(
        voltage_kv,
        electron_mass,
        elementary_charge,
        light_speed,
        planck_constant,
    )
    unrounded_interaction: Complex128[Array, " p"] = (
        coupling * voltage_coefficients
    )
    rounded_real: Float64[Array, " p"] = lax.reduce_precision(
        jnp.real(unrounded_interaction),
        exponent_bits=11,
        mantissa_bits=50,
    )
    rounded_imaginary: Float64[Array, " p"] = lax.reduce_precision(
        jnp.imag(unrounded_interaction),
        exponent_bits=11,
        mantissa_bits=50,
    )
    raw_interaction: Complex128[Array, " p"] = (
        rounded_real + 1j * rounded_imaginary
    )
    interaction: Complex128[Array, " p"] = lax.optimization_barrier(
        raw_interaction
    )
    result: Tuple[Float64[Array, ""], Complex128[Array, " p"]] = (
        coupling,
        interaction,
    )
    return result


__all__: list[str] = [
    "coupled_interaction_value",
    "helmholtz_coupling_value",
]
