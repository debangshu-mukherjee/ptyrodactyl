"""Distribution-axis reducers for public detector intensities.

Extended Summary
----------------
Layer-0 = amplitude kernels: complex in, complex out, no reduction over any
distribution axis. Layer-1 = an integrator that builds Distribution axes,
binds them to one Layer-0 kernel, and calls apply_distribution(s) for the
single late reduction. The only `|·|²` on the public detector path lives in
this module.

Routine Listings
----------------
:func:`apply_distribution`
    Reduce one weighted distribution axis to detector intensity.
:func:`apply_distributions`
    Reduce multiple weighted distribution axes to detector intensity.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Complex, Float, jaxtyped

from ptyrodactyl.types import Distribution, ReductionMode


@jaxtyped(typechecker=beartype)
def apply_distribution(
    distribution: Distribution,
    bound_amplitude_fn: Callable[
        [Float[Array, "D"]],
        Complex[Array, "H W"],
    ],
) -> Float[Array, "H W"]:
    """Apply the late detector reduction for one distribution axis.

    Parameters
    ----------
    distribution : Distribution
        Weighted sample axis to reduce. ``reduction`` is static, so the
        coherent/incoherent branch is selected with Python control flow.
    bound_amplitude_fn : Callable[[Float[Array, "D"]], Complex[Array, "H W"]]
        Bound Layer-0 kernel that maps one sample row to one complex field.

    Returns
    -------
    intensity : Float[Array, "H W"]
        Detector intensity after the single late modulus-squared reduction.

    Notes
    -----
    Coherent axes sum weighted amplitudes before taking ``|.|^2``.
    Incoherent axes take ``|.|^2`` per sample before the weighted sum.
    The reducer is intentionally not JIT-decorated; callers choose the
    transformation boundary.
    """
    amplitudes: Complex[Array, "N H W"] = jax.vmap(bound_amplitude_fn)(
        distribution.samples,
    )
    if distribution.reduction is ReductionMode.COHERENT:
        weights_complexsafe: Complex[Array, "N"] = distribution.weights.astype(
            amplitudes.dtype,
        )
        field: Complex[Array, "H W"] = jnp.einsum(
            "n,nhw->hw",
            weights_complexsafe,
            amplitudes,
        )
        intensity: Float[Array, "H W"] = jnp.abs(field) ** 2
    elif distribution.reduction is ReductionMode.INCOHERENT:
        intensity = jnp.einsum(
            "n,nhw->hw",
            distribution.weights,
            jnp.abs(amplitudes) ** 2,
        )
    else:
        raise ValueError("unknown distribution reduction mode")
    return intensity


@jaxtyped(typechecker=beartype)
def apply_distributions(
    distributions: tuple[Distribution, ...],
    bound_amplitude_fn: Callable[
        [Float[Array, "D"]],
        Complex[Array, "H W"],
    ],
) -> Float[Array, "H W"]:
    """Apply the late detector reduction for multiple distribution axes.

    The kernel sample fed to `bound_amplitude_fn` is the **concatenation of one
    row from each axis, in tuple order** — so for axes with dims (D1, D2, ...)
    the bound fn receives `Float[Array, " D1+D2+..."]`.

    Parameters
    ----------
    distributions : tuple[Distribution, ...]
        Weighted sample axes to reduce. Axis order defines the cursor
        concatenation order passed to ``bound_amplitude_fn``.
    bound_amplitude_fn : Callable[[Float[Array, "D"]], Complex[Array, "H W"]]
        Bound Layer-0 kernel that maps one concatenated cursor row to one
        complex field.

    Returns
    -------
    intensity : Float[Array, "H W"]
        Detector intensity after coherent axes have been reduced inside the
        modulus and incoherent axes outside it.

    Notes
    -----
    Implements
    ``sum_i prod(w_i) * |sum_c prod(w_c) * A(concat rows)|^2``, where
    incoherent products live outside the modulus and coherent products live
    inside it. The reducer is intentionally not JIT-decorated; callers choose
    the transformation boundary.

    Raises
    ------
    ValueError
        If no distribution axes are supplied.
    """
    if len(distributions) == 0:
        raise ValueError("distributions must contain at least one axis")

    sample_counts: tuple[int, ...] = tuple(
        distribution.samples.shape[0] for distribution in distributions
    )
    sample_indices: tuple[Array, ...] = tuple(
        jnp.arange(sample_count) for sample_count in sample_counts
    )
    index_grids: list[Array] = jnp.meshgrid(
        *sample_indices,
        indexing="ij",
    )
    flat_samples_by_axis: list[Float[Array, "P D"]] = [
        distribution.samples[index_grid.reshape(-1)]
        for distribution, index_grid in zip(
            distributions,
            index_grids,
            strict=True,
        )
    ]
    cartesian_samples: Float[Array, "P D"] = jnp.concatenate(
        flat_samples_by_axis,
        axis=1,
    )
    flat_amplitudes: Complex[Array, "P H W"] = jax.vmap(bound_amplitude_fn)(
        cartesian_samples,
    )
    amplitudes: Complex[Array, "... H W"] = flat_amplitudes.reshape(
        (*sample_counts, *flat_amplitudes.shape[-2:]),
    )

    axis_labels: list[int] = list(range(len(distributions)))
    coherent_axes: list[int] = [
        axis
        for axis, distribution in enumerate(distributions)
        if distribution.reduction is ReductionMode.COHERENT
    ]
    field: Array = amplitudes
    for original_axis in sorted(coherent_axes, reverse=True):
        current_axis: int = axis_labels.index(original_axis)
        weights_complexsafe = distributions[original_axis].weights.astype(
            field.dtype,
        )
        field = _weighted_sum_axis(field, weights_complexsafe, current_axis)
        axis_labels.pop(current_axis)

    intensity = jnp.abs(field) ** 2
    incoherent_axes: list[int] = [
        axis
        for axis, distribution in enumerate(distributions)
        if distribution.reduction is ReductionMode.INCOHERENT
    ]
    for original_axis in sorted(incoherent_axes, reverse=True):
        current_axis = axis_labels.index(original_axis)
        intensity = _weighted_sum_axis(
            intensity,
            distributions[original_axis].weights,
            current_axis,
        )
        axis_labels.pop(current_axis)

    reduced_intensity: Float[Array, "H W"] = intensity
    return reduced_intensity


def _weighted_sum_axis(
    values: Array,
    weights: Array,
    axis: int,
) -> Array:
    """Return a weighted sum over one sample axis."""
    weight_shape: tuple[int, ...] = (
        (1,) * axis + (weights.shape[0],) + (1,) * (values.ndim - axis - 1)
    )
    weighted_values: Array = values * weights.reshape(weight_shape)
    summed_values: Array = jnp.sum(weighted_values, axis=axis)
    return summed_values
