"""Dependency-neutral internal scalar physics primitives."""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Complex, Float


def helmholtz_coupling_value(
    voltage_kv: Float[Array, ""] | float,
    electron_mass: Float[Array, ""] | float,
    elementary_charge: Float[Array, ""] | float,
    light_speed: Float[Array, ""] | float,
    planck_constant: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Evaluate the 50-mantissa-bit canonical Helmholtz coupling."""
    mass: Float[Array, ""] = jnp.float64(electron_mass)
    charge: Float[Array, ""] = jnp.float64(elementary_charge)
    speed: Float[Array, ""] = jnp.float64(light_speed)
    planck: Float[Array, ""] = jnp.float64(planck_constant)
    voltage: Float[Array, ""] = jnp.float64(voltage_kv)
    energy: Float[Array, ""] = voltage * jnp.float64(1000.0) * charge
    relativistic_mass: Float[Array, ""] = mass * (
        1.0 + energy / (mass * jnp.square(speed))
    )
    coupling_si: Float[Array, ""] = (
        8.0
        * jnp.square(jnp.pi)
        * relativistic_mass
        * charge
        / jnp.square(planck)
    )
    raw_coupling: Float[Array, ""] = coupling_si * jnp.float64(1.0e-20)
    rounded_coupling: Float[Array, ""] = lax.reduce_precision(
        raw_coupling,
        exponent_bits=11,
        mantissa_bits=50,
    )
    coupling: Float[Array, ""] = lax.optimization_barrier(rounded_coupling)
    return coupling


def coupled_interaction_value(
    voltage_coefficients: Complex[Array, " p"],
    voltage_kv: Float[Array, ""] | float,
    electron_mass: Float[Array, ""] | float,
    elementary_charge: Float[Array, ""] | float,
    light_speed: Float[Array, ""] | float,
    planck_constant: Float[Array, ""] | float,
) -> tuple[Float[Array, ""], Complex[Array, " p"]]:
    """Evaluate and canonically round the coupling and coupled coefficients."""
    coupling: Float[Array, ""] = helmholtz_coupling_value(
        voltage_kv,
        electron_mass,
        elementary_charge,
        light_speed,
        planck_constant,
    )
    unrounded_interaction: Complex[Array, " p"] = (
        coupling * voltage_coefficients
    )
    rounded_real: Float[Array, " p"] = lax.reduce_precision(
        jnp.real(unrounded_interaction),
        exponent_bits=11,
        mantissa_bits=50,
    )
    rounded_imaginary: Float[Array, " p"] = lax.reduce_precision(
        jnp.imag(unrounded_interaction),
        exponent_bits=11,
        mantissa_bits=50,
    )
    raw_interaction: Complex[Array, " p"] = (
        rounded_real + 1j * rounded_imaginary
    )
    interaction: Complex[Array, " p"] = lax.optimization_barrier(
        raw_interaction
    )
    result: tuple[Float[Array, ""], Complex[Array, " p"]] = (
        coupling,
        interaction,
    )
    return result
