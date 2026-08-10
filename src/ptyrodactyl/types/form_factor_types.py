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
    Store Kirkland coefficients for one element.
:class:`LobatoParameters`
    Store Lobato--Van Dyck coefficients for one element.
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

    :see: :func:`~.test_factories_create_float64_equinox_pytrees_under_jit`

    Attributes
    ----------
    amplitudes : Float[Array, " 5"]
        Five amplitude coefficients :math:`a_i`.
    scales : Float[Array, " 5"]
        Five strictly positive width coefficients :math:`b_i` in square
        Angstroms.

    See Also
    --------
    :func:`create_lobato_parameters`
        Create and validate :class:`LobatoParameters`.
    """

    amplitudes: Float[Array, " 5"]
    scales: Float[Array, " 5"]


class KirklandParameters(eqx.Module):
    """Store Kirkland coefficients for one element.

    :see: :func:`~.test_factories_create_float64_equinox_pytrees_under_jit`

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

    See Also
    --------
    :func:`create_kirkland_parameters`
        Create and validate :class:`KirklandParameters`.
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
    """PRIVATE: Convert one coefficient vector and validate its shape.

    Parameters
    ----------
    values : Float[Array, "..."]
        Floating coefficient values to convert to binary64.
    name : str
        Field name included in the validation error.
    length : int
        Required static vector length.

    Returns
    -------
    array : Float[Array, " length"]
        Binary64 coefficient vector with the requested length.

    Raises
    ------
    ValueError
        If the converted array does not have shape ``(length,)``.
    """
    array: Float[Array, " length"] = jnp.asarray(values, dtype=jnp.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return array


def _checked_finite(
    values: Float[Array, " length"],
    *,
    name: str,
) -> Float[Array, " length"]:
    """PRIVATE: Apply a traced finite-value check to one vector.

    Parameters
    ----------
    values : Float[Array, " length"]
        Coefficient vector to validate.
    name : str
        Field name included in the runtime error.

    Returns
    -------
    checked : Float[Array, " length"]
        Input vector with a traced finite-value assertion.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any value is non-finite under compiled execution.
    """
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
    """PRIVATE: Apply traced finite and positive checks to scale values.

    Parameters
    ----------
    values : Float[Array, " length"]
        Physical scale values to validate.
    name : str
        Field name included in the runtime error.

    Returns
    -------
    result : Float[Array, " length"]
        Scale vector with traced finite and strictly positive assertions.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any value is non-finite or non-positive under compiled execution.
    """
    checked: Float[Array, " length"] = _checked_finite(values, name=name)
    result: Float[Array, " length"] = eqx.error_if(
        checked, jnp.any(checked <= 0), f"{name} must be strictly positive"
    )
    return result


@jaxtyped(typechecker=beartype)
def create_lobato_parameters(
    amplitudes: Float[Array, "..."],
    scales: Float[Array, "..."],
) -> LobatoParameters:
    """Create validated Lobato--Van Dyck coefficients for one element.

    :see: :mod:`~.test_form_factor_types`

    Parameters
    ----------
    amplitudes : Float[Array, "..."]
        Five Lobato amplitude coefficients.
    scales : Float[Array, "..."]
        Five Lobato scale coefficients in square Angstroms.

    Returns
    -------
    parameters : LobatoParameters
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

    :see: :mod:`~.test_form_factor_types`

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
    parameters : KirklandParameters
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
