"""Define validated atomic form-factor parameter carriers.

Extended Summary
----------------
This module provides the canonical Equinox PyTree carriers for one element's
Lobato--Van Dyck or Kirkland electron-scattering coefficients. Factory
functions separate Python-visible structural checks from traced numerical
checks so the resulting carriers remain compatible with JAX transformations.

Routine Listings
----------------
:class:`KirklandParameters`
    Kirkland Lorentzian and Gaussian amplitude/scale pairs.
:class:`LobatoParameters`
    Lobato--Van Dyck amplitude/scale pairs.
:func:`create_kirkland_parameters`
    Create validated Kirkland coefficients for one element.
:func:`create_lobato_parameters`
    Create validated Lobato--Van Dyck coefficients for one element.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

_KIRKLAND_TERM_COUNT: int = 3
_LOBATO_TERM_COUNT: int = 5


class LobatoParameters(eqx.Module):
    """Store Lobato--Van Dyck coefficients for one element.

    Attributes
    ----------
    amplitudes : Float[Array, " 5"]
        Five amplitude coefficients :math:`a_i`.
    scales : Float[Array, " 5"]
        Five strictly positive width coefficients :math:`b_i` in square
        Angstroms.
    """

    amplitudes: Float[Array, " 5"]
    scales: Float[Array, " 5"]


class KirklandParameters(eqx.Module):
    """Store Kirkland coefficients for one element.

    Attributes
    ----------
    lorentzian_amplitudes : Float[Array, " 3"]
        Three Lorentzian amplitude coefficients.
    lorentzian_scales : Float[Array, " 3"]
        Three strictly positive Lorentzian scale coefficients.
    gaussian_amplitudes : Float[Array, " 3"]
        Three Gaussian amplitude coefficients.
    gaussian_scales : Float[Array, " 3"]
        Three strictly positive Gaussian scale coefficients.
    """

    lorentzian_amplitudes: Float[Array, " 3"]
    lorentzian_scales: Float[Array, " 3"]
    gaussian_amplitudes: Float[Array, " 3"]
    gaussian_scales: Float[Array, " 3"]


def _coerce_vector(
    values: Float[Array, "..."],
    *,
    name: str,
    length: int,
) -> Float[Array, " length"]:
    """Convert one coefficient vector and reject a wrong static shape."""
    array: Float[Array, " length"] = jnp.asarray(values, dtype=jnp.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return array


def _checked_finite(
    values: Float[Array, " length"],
    *,
    name: str,
) -> Float[Array, " length"]:
    """Apply a traced finiteness check to one coefficient vector."""
    checked: Float[Array, " length"] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        f"{name} must contain only finite values",
    )
    return checked


def _checked_scales(
    values: Float[Array, " length"],
    *,
    name: str,
) -> Float[Array, " length"]:
    """Apply traced finite and positive checks to physical scale values."""
    checked: Float[Array, " length"] = _checked_finite(values, name=name)
    return eqx.error_if(
        checked,
        jnp.any(checked <= 0),
        f"{name} must be strictly positive",
    )


@jaxtyped(typechecker=beartype)
def create_lobato_parameters(
    amplitudes: Float[Array, "..."],
    scales: Float[Array, "..."],
) -> LobatoParameters:
    """Create validated Lobato--Van Dyck coefficients for one element.

    Parameters
    ----------
    amplitudes : Float[Array, "..."]
        Five Lobato amplitude coefficients.
    scales : Float[Array, "..."]
        Five Lobato scale coefficients in square Angstroms.

    Returns
    -------
    lobato_parameters : LobatoParameters
        Float64 Lobato coefficient carrier.

    Raises
    ------
    ValueError
        If either input does not have shape ``(5,)``.

    Notes
    -----
    Static shapes are checked in Python. Finiteness and strictly positive
    scales are checked with :func:`equinox.error_if`, including under JIT.
    """
    amplitudes_arr: Float[Array, " 5"] = _coerce_vector(
        amplitudes,
        name="amplitudes",
        length=_LOBATO_TERM_COUNT,
    )
    scales_arr: Float[Array, " 5"] = _coerce_vector(
        scales,
        name="scales",
        length=_LOBATO_TERM_COUNT,
    )
    checked_amplitudes: Float[Array, " 5"] = _checked_finite(
        amplitudes_arr,
        name="amplitudes",
    )
    checked_scales: Float[Array, " 5"] = _checked_scales(
        scales_arr,
        name="scales",
    )
    parameters: LobatoParameters = LobatoParameters(
        amplitudes=checked_amplitudes,
        scales=checked_scales,
    )
    return parameters


@jaxtyped(typechecker=beartype)
def create_kirkland_parameters(
    lorentzian_amplitudes: Float[Array, "..."],
    lorentzian_scales: Float[Array, "..."],
    gaussian_amplitudes: Float[Array, "..."],
    gaussian_scales: Float[Array, "..."],
) -> KirklandParameters:
    """Create validated Kirkland coefficients for one element.

    Parameters
    ----------
    lorentzian_amplitudes : Float[Array, "..."]
        Three Lorentzian amplitude coefficients.
    lorentzian_scales : Float[Array, "..."]
        Three Lorentzian scale coefficients.
    gaussian_amplitudes : Float[Array, "..."]
        Three Gaussian amplitude coefficients.
    gaussian_scales : Float[Array, "..."]
        Three Gaussian scale coefficients.

    Returns
    -------
    kirkland_parameters : KirklandParameters
        Float64 Kirkland coefficient carrier.

    Raises
    ------
    ValueError
        If any input does not have shape ``(3,)``.

    Notes
    -----
    Static shapes are checked in Python. Finiteness and strictly positive
    scales are checked with :func:`equinox.error_if`, including under JIT.
    """
    lorentzian_amplitudes_arr: Float[Array, " 3"] = _coerce_vector(
        lorentzian_amplitudes,
        name="lorentzian_amplitudes",
        length=_KIRKLAND_TERM_COUNT,
    )
    lorentzian_scales_arr: Float[Array, " 3"] = _coerce_vector(
        lorentzian_scales,
        name="lorentzian_scales",
        length=_KIRKLAND_TERM_COUNT,
    )
    gaussian_amplitudes_arr: Float[Array, " 3"] = _coerce_vector(
        gaussian_amplitudes,
        name="gaussian_amplitudes",
        length=_KIRKLAND_TERM_COUNT,
    )
    gaussian_scales_arr: Float[Array, " 3"] = _coerce_vector(
        gaussian_scales,
        name="gaussian_scales",
        length=_KIRKLAND_TERM_COUNT,
    )

    parameters: KirklandParameters = KirklandParameters(
        lorentzian_amplitudes=_checked_finite(
            lorentzian_amplitudes_arr,
            name="lorentzian_amplitudes",
        ),
        lorentzian_scales=_checked_scales(
            lorentzian_scales_arr,
            name="lorentzian_scales",
        ),
        gaussian_amplitudes=_checked_finite(
            gaussian_amplitudes_arr,
            name="gaussian_amplitudes",
        ),
        gaussian_scales=_checked_scales(
            gaussian_scales_arr,
            name="gaussian_scales",
        ),
    )
    return parameters


__all__: list[str] = [
    "KirklandParameters",
    "LobatoParameters",
    "create_kirkland_parameters",
    "create_lobato_parameters",
]
