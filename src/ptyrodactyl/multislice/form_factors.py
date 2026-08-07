"""Atomic form factors and projected independent-atom potentials.

Extended Summary
----------------
This module implements the Lobato--Van Dyck and Kirkland electron-scattering
parameterizations. Primitive functions consume validated coefficient carriers;
atomic-number dispatchers load the bundled coefficients and default to Lobato.

Routine Listings
----------------
:func:`atomic_form_factor`
    Evaluate an atomic form factor, using Lobato by default.
:func:`kirkland_form_factor`
    Evaluate the Kirkland reciprocal-space form factor.
:func:`kirkland_projected_potential`
    Evaluate the Kirkland projected real-space potential.
:func:`lobato_bandlimited_peak`
    Evaluate the on-nucleus peak of a band-limited Lobato potential.
:func:`lobato_form_factor`
    Evaluate the Lobato--Van Dyck reciprocal-space form factor.
:func:`lobato_projected_potential`
    Evaluate the Lobato--Van Dyck projected real-space potential.
:func:`projected_atom_potential`
    Evaluate an atomic projected potential, using Lobato by default.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, jaxtyped

from ptyrodactyl.inout.form_factor_data import (
    kirkland_potentials,
    lobato_potentials,
)
from ptyrodactyl.types import (
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    scalar_float,
    scalar_int,
)
from ptyrodactyl.types.form_factor_types import (
    KirklandParameters,
    LobatoParameters,
    create_kirkland_parameters,
    create_lobato_parameters,
)

_TWO_PI: scalar_float = 2.0 * jnp.pi
_LOBATO_AMPLITUDE_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8)
_LOBATO_SCALE_INDICES: tuple[int, ...] = (1, 3, 5, 7, 9)
_KIRKLAND_AMPLITUDE_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8, 10)
_KIRKLAND_SCALE_INDICES: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
_MAX_ATOMIC_NUMBER: int = 103


def _validate_parameterization(parameterization: str) -> None:
    """Reject an unsupported independent-atom parameterization."""
    if parameterization not in {"lobato", "kirkland"}:
        raise ValueError("parameterization must be 'lobato' or 'kirkland'")


def _checked_radius(r: Float[Array, "..."]) -> Float[Array, "..."]:
    """Reject negative or non-finite radial coordinates under tracing."""
    checked: Float[Array, "..."] = eqx.error_if(
        r,
        jnp.any(~jnp.isfinite(r)) | jnp.any(r < 0.0),
        "r must contain finite nonnegative radial distances",
    )
    return checked


def _checked_atom_index(atom_no: scalar_int) -> Int[Array, ""]:
    """Validate an atomic number and return its zero-based table index."""
    if isinstance(atom_no, bool):
        raise ValueError("atom_no must be an integer between 1 and 103")
    if isinstance(atom_no, int) and not 1 <= atom_no <= _MAX_ATOMIC_NUMBER:
        raise ValueError("atom_no must be between 1 and 103")
    atomic_number: Int[Array, ""] = jnp.asarray(atom_no, dtype=jnp.int32)
    checked_atomic_number: Int[Array, ""] = eqx.error_if(
        atomic_number,
        (atomic_number < 1) | (atomic_number > _MAX_ATOMIC_NUMBER),
        "atom_no must be between 1 and 103",
    )
    return checked_atomic_number - 1


def _load_lobato_parameters(atom_no: scalar_int) -> LobatoParameters:
    """Load one element's Lobato coefficients into the validated carrier."""
    atom_index: Int[Array, ""] = _checked_atom_index(atom_no)
    row: Float[Array, " 10"] = lobato_potentials()[atom_index]
    amplitudes: Float[Array, " 5"] = row[
        jnp.asarray(_LOBATO_AMPLITUDE_INDICES, dtype=jnp.int32)
    ]
    scales: Float[Array, " 5"] = row[
        jnp.asarray(_LOBATO_SCALE_INDICES, dtype=jnp.int32)
    ]
    return create_lobato_parameters(amplitudes=amplitudes, scales=scales)


def _load_kirkland_parameters(atom_no: scalar_int) -> KirklandParameters:
    """Load one element's Kirkland coefficients into the validated carrier."""
    atom_index: Int[Array, ""] = _checked_atom_index(atom_no)
    row: Float[Array, " 12"] = kirkland_potentials()[atom_index]
    amplitudes: Float[Array, " 6"] = row[
        jnp.asarray(_KIRKLAND_AMPLITUDE_INDICES, dtype=jnp.int32)
    ]
    scales: Float[Array, " 6"] = row[
        jnp.asarray(_KIRKLAND_SCALE_INDICES, dtype=jnp.int32)
    ]
    return create_kirkland_parameters(
        lorentzian_amplitudes=amplitudes[:3],
        lorentzian_scales=scales[:3],
        gaussian_amplitudes=amplitudes[3:],
        gaussian_scales=scales[3:],
    )


@jaxtyped(typechecker=beartype)
def lobato_form_factor(
    params: LobatoParameters,
    q: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Evaluate the Lobato--Van Dyck electron form factor.

    Parameters
    ----------
    params : LobatoParameters
        Five validated Lobato amplitude and scale pairs.
    q : Float[Array, "..."]
        Angular scattering-vector magnitude in radians per Angstrom.

    Returns
    -------
    form_factor : Float[Array, "..."]
        Electron form factor in Angstroms, with the same shape as ``q``.

    Notes
    -----
    With :math:`g=q/(2\pi)`, the parameterization is

    .. math::

        f(g) = \sum_i a_i\frac{2+b_i g^2}{(1+b_i g^2)^2}.
    """
    g_squared: Float[Array, "..."] = jnp.square(q / _TWO_PI)
    scaled_squared: Float[Array, "... 5"] = (
        g_squared[..., jnp.newaxis] * params.scales
    )
    terms: Float[Array, "... 5"] = (
        params.amplitudes
        * (2.0 + scaled_squared)
        / jnp.square(1.0 + scaled_squared)
    )
    form_factor: Float[Array, "..."] = jnp.sum(terms, axis=-1)
    return form_factor


@jaxtyped(typechecker=beartype)
def kirkland_form_factor(
    params: KirklandParameters,
    q: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Evaluate the Kirkland electron form factor.

    Parameters
    ----------
    params : KirklandParameters
        Validated Kirkland Lorentzian and Gaussian coefficients.
    q : Float[Array, "..."]
        Angular scattering-vector magnitude in radians per Angstrom.

    Returns
    -------
    form_factor : Float[Array, "..."]
        Electron form factor with the same shape as ``q``.

    Notes
    -----
    With :math:`g=q/(2\pi)`, the parameterization is

    .. math::

        f(g) = \sum_i \frac{a_i}{g^2+b_i}
             + \sum_i c_i\exp(-d_i g^2).
    """
    g_squared: Float[Array, "... 1"] = jnp.square(q / _TWO_PI)[
        ..., jnp.newaxis
    ]
    lorentzian_terms: Float[Array, "... 3"] = params.lorentzian_amplitudes / (
        g_squared + params.lorentzian_scales
    )
    gaussian_terms: Float[Array, "... 3"] = (
        params.gaussian_amplitudes
        * jnp.exp(-params.gaussian_scales * g_squared)
    )
    form_factor: Float[Array, "..."] = jnp.sum(
        lorentzian_terms, axis=-1
    ) + jnp.sum(gaussian_terms, axis=-1)
    return form_factor


def _bessel_kv(
    order: float, argument: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Evaluate the local differentiable Bessel-K implementation lazily."""
    # A lazy import avoids a module cycle: atom_potentials consumes the public
    # dispatchers below, while its established Bessel implementation remains
    # the single numerical source used by both parameterizations.
    from .atom_potentials import bessel_kv  # noqa: PLC0415

    return bessel_kv(order, argument)


@jaxtyped(typechecker=beartype)
def lobato_projected_potential(
    params: LobatoParameters,
    r: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Evaluate the Lobato--Van Dyck projected electrostatic potential.

    Parameters
    ----------
    params : LobatoParameters
        Five validated Lobato amplitude and scale pairs.
    r : Float[Array, "..."]
        Radial distance from the atom in Angstroms.

    Returns
    -------
    potential : Float[Array, "..."]
        Positive projected potential in volt-Angstroms.

    Notes
    -----
    This is Eq. 16 of Lobato and Van Dyck (2014), with the positive
    electrostatic-potential convention and :math:`C_{MB}=47.87801`
    V Angstrom squared.
    """
    checked_r: Float[Array, "..."] = _checked_radius(r)
    r_safe: Float[Array, "..."] = jnp.maximum(checked_r, 1e-10)
    expanded_r: Float[Array, "... 1"] = r_safe[..., jnp.newaxis]
    sqrt_scales: Float[Array, " 5"] = jnp.sqrt(params.scales)
    arguments: Float[Array, "... 5"] = _TWO_PI * expanded_r / sqrt_scales
    k0_values: Float[Array, "... 5"] = _bessel_kv(0.0, arguments)
    k1_values: Float[Array, "... 5"] = _bessel_kv(1.0, arguments)
    k0_terms: Float[Array, "... 5"] = (
        params.amplitudes / params.scales * k0_values
    )
    k1_terms: Float[Array, "... 5"] = (
        params.amplitudes
        * jnp.pi
        * expanded_r
        / jnp.power(params.scales, 1.5)
        * k1_values
    )
    potential: Float[Array, "..."] = (
        _TWO_PI
        * MOTT_BETHE_VOLT_ANGSTROM_SQ
        * jnp.sum(k0_terms + k1_terms, axis=-1)
    )
    return potential


@jaxtyped(typechecker=beartype)
def kirkland_projected_potential(
    params: KirklandParameters,
    r: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Evaluate the Kirkland projected electrostatic potential.

    Parameters
    ----------
    params : KirklandParameters
        Validated Kirkland Lorentzian and Gaussian coefficients.
    r : Float[Array, "..."]
        Radial distance from the atom in Angstroms.

    Returns
    -------
    potential : Float[Array, "..."]
        Projected potential in the established ptyrodactyl units.

    Notes
    -----
    The legacy ``0.5292 * 14.4`` normalization is intentionally retained so
    selecting Kirkland reproduces pre-Plan-06 projected potentials exactly.
    """
    checked_r: Float[Array, "..."] = _checked_radius(r)
    r_safe: Float[Array, "..."] = jnp.maximum(checked_r, 1e-10)
    expanded_r: Float[Array, "... 1"] = r_safe[..., jnp.newaxis]
    arguments: Float[Array, "... 3"] = (
        _TWO_PI * expanded_r * jnp.sqrt(params.lorentzian_scales)
    )
    lorentzian_terms: Float[Array, "... 3"] = (
        params.lorentzian_amplitudes * _bessel_kv(0.0, arguments)
    )
    gaussian_terms: Float[Array, "... 3"] = (
        params.gaussian_amplitudes
        / params.gaussian_scales
        * jnp.exp(
            -(jnp.pi**2) * jnp.square(expanded_r) / params.gaussian_scales
        )
    )
    a0_legacy: scalar_float = jnp.asarray(0.5292)
    ek_legacy: scalar_float = jnp.asarray(14.4)
    lorentzian_prefactor: scalar_float = (
        4.0 * jnp.pi**2 * a0_legacy * ek_legacy
    )
    gaussian_prefactor: scalar_float = 2.0 * jnp.pi**2 * a0_legacy * ek_legacy
    # Preserve the historical three-term addition order as part of the
    # same-parameterization numerical contract.
    lorentzian_sum: Float[Array, "..."] = (
        lorentzian_terms[..., 0]
        + lorentzian_terms[..., 1]
        + lorentzian_terms[..., 2]
    )
    gaussian_sum: Float[Array, "..."] = (
        gaussian_terms[..., 0]
        + gaussian_terms[..., 1]
        + gaussian_terms[..., 2]
    )
    potential: Float[Array, "..."] = (
        lorentzian_prefactor * lorentzian_sum
        + gaussian_prefactor * gaussian_sum
    )
    return potential


@jaxtyped(typechecker=beartype)
def atomic_form_factor(
    atom_no: scalar_int,
    q: Float[Array, "..."],
    *,
    parameterization: str = "lobato",
) -> Float[Array, "..."]:
    """Evaluate an atomic form factor with Lobato as the default.

    Parameters
    ----------
    atom_no : scalar_int
        Atomic number from 1 through 103.
    q : Float[Array, "..."]
        Angular scattering-vector magnitude in radians per Angstrom.
    parameterization : str, optional
        ``"lobato"`` (default) or explicit ``"kirkland"``.

    Returns
    -------
    form_factor : Float[Array, "..."]
        Electron form factor with the same shape as ``q``.

    Raises
    ------
    ValueError
        If ``parameterization`` is unsupported.
    """
    _validate_parameterization(parameterization)
    if parameterization == "lobato":
        return lobato_form_factor(_load_lobato_parameters(atom_no), q)
    return kirkland_form_factor(_load_kirkland_parameters(atom_no), q)


@jaxtyped(typechecker=beartype)
def projected_atom_potential(
    atom_no: scalar_int,
    r: Float[Array, "..."],
    *,
    parameterization: str = "lobato",
) -> Float[Array, "..."]:
    """Evaluate an atomic projected potential with Lobato as the default.

    Parameters
    ----------
    atom_no : scalar_int
        Atomic number from 1 through 103.
    r : Float[Array, "..."]
        Radial distance from the atom in Angstroms.
    parameterization : str, optional
        ``"lobato"`` (default) or explicit ``"kirkland"``.

    Returns
    -------
    potential : Float[Array, "..."]
        Projected electrostatic potential with the same shape as ``r``.

    Raises
    ------
    ValueError
        If ``parameterization`` is unsupported.
    """
    _validate_parameterization(parameterization)
    if parameterization == "lobato":
        return lobato_projected_potential(_load_lobato_parameters(atom_no), r)
    return kirkland_projected_potential(_load_kirkland_parameters(atom_no), r)


@jaxtyped(typechecker=beartype)
def lobato_bandlimited_peak(
    atom_no: scalar_int,
    g_max: scalar_float,
) -> Float[Array, ""]:
    r"""Evaluate the on-nucleus peak of a band-limited Lobato potential.

    Parameters
    ----------
    atom_no : scalar_int
        Atomic number from 1 through 103.
    g_max : scalar_float
        Positive crystallographic band limit in cycles per Angstrom.

    Returns
    -------
    peak_potential : Float[Array, ""]
        Positive band-limited electrostatic-potential peak in volts.

    Raises
    ------
    ValueError
        If a Python ``g_max`` is nonpositive or non-finite. Traced invalid
        values are rejected by a runtime check.

    Notes
    -----
    Implements Eqs. E.51--E.52 of the scalar-CBS convention:

    .. math::

        \phi_{BL}(0;G)=4\pi C_{MB}\sum_i\frac{a_i}{b_i^{3/2}}
        \left[X_i-\frac{X_i}{2(1+X_i^2)}-\frac{\arctan X_i}{2}\right],
        \quad X_i=G\sqrt{b_i}.
    """
    if isinstance(g_max, int | float) and (
        not math.isfinite(g_max) or g_max <= 0
    ):
        raise ValueError("g_max must be positive and finite")
    g_max_array: Float[Array, ""] = jnp.asarray(g_max, dtype=jnp.float64)
    checked_g_max: Float[Array, ""] = eqx.error_if(
        g_max_array,
        (~jnp.isfinite(g_max_array)) | (g_max_array <= 0),
        "g_max must be positive and finite",
    )
    params: LobatoParameters = _load_lobato_parameters(atom_no)
    scaled_limit: Float[Array, " 5"] = checked_g_max * jnp.sqrt(params.scales)
    antiderivative: Float[Array, " 5"] = (
        scaled_limit
        - scaled_limit / (2.0 * (1.0 + jnp.square(scaled_limit)))
        - 0.5 * jnp.arctan(scaled_limit)
    )
    radial_integral: Float[Array, ""] = jnp.sum(
        params.amplitudes * jnp.power(params.scales, -1.5) * antiderivative
    )
    peak_potential: Float[Array, ""] = (
        4.0 * jnp.pi * MOTT_BETHE_VOLT_ANGSTROM_SQ * radial_integral
    )
    return peak_potential


__all__: list[str] = [
    "atomic_form_factor",
    "kirkland_form_factor",
    "kirkland_projected_potential",
    "lobato_bandlimited_peak",
    "lobato_form_factor",
    "lobato_projected_potential",
    "projected_atom_potential",
]
