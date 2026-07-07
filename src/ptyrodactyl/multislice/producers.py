"""Distribution producers for CBED ensemble axes.

Extended Summary
----------------
This module turns physical CBED averages into explicit
:class:`~ptyrodactyl.types.Distribution` axes. Producers own quadrature
construction for the ensemble binder in
:func:`ptyrodactyl.multislice.bind_cbed_axes`.

Source-size broadening maps onto the same incoherent position-delta axis as
``position_jitter_to_distribution``. A separate source-size carrier would
duplicate the position-update field without changing the kernel contract, and
Plan 03 W3d authorizes folding that physical average into position jitter.

Routine Listings
----------------
:func:`coherence_to_distribution`
    Build the incoherent chromatic/angular coherence distribution.
:func:`position_jitter_to_distribution`
    Build the incoherent two-dimensional position-jitter distribution.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ptyrodactyl.multislice._ensemble_axes import (
    _AXIS_COHERENCE,
    _AXIS_POSITION_JITTER,
)
from ptyrodactyl.types import (
    Distribution,
    ReductionMode,
    create_distribution,
    scalar_float,
    scalar_int,
)


@jaxtyped(typechecker=beartype)
def coherence_to_distribution(
    energy_spread_ev: scalar_float,
    angular_divergence_mrad: scalar_float,
    n_quad: scalar_int,
) -> Distribution:
    """Build an incoherent chromatic/angular coherence distribution.

    Parameters
    ----------
    energy_spread_ev : scalar_float
        Gaussian standard deviation for energy offsets in electronvolts.
    angular_divergence_mrad : scalar_float
        Gaussian standard deviation for independent x/y beam tilts in mrad.
    n_quad : scalar_int
        Static number of Gauss-Hermite nodes per axis.

    Returns
    -------
    distribution : Distribution
        Incoherent distribution with ``axis_id="coherence"`` and samples
        ``[dE_ev, dtheta_x_mrad, dtheta_y_mrad]``. The binder folds these
        columns into an ``AxisUpdate`` energy delta and tilt delta.

    Notes
    -----
    Physicists' Gauss-Hermite nodes are scaled by ``sqrt(2)`` so
    ``width * nodes`` samples a normal distribution with standard deviation
    ``width``; weights are normalized by ``sqrt(pi)``.
    
    :see: bind_cbed_axes, position_jitter_to_distribution.
    """
    nodes: Float[Array, " Q"]
    weights: Float[Array, " Q"]
    nodes, weights = _normal_gauss_hermite_rule(n_quad)
    energy_nodes: Float[Array, " Q"] = _collapse_width(
        energy_spread_ev,
        nodes,
        "energy_spread_ev",
    )
    tilt_nodes: Float[Array, " Q"] = _collapse_width(
        angular_divergence_mrad,
        nodes,
        "angular_divergence_mrad",
    )
    energy_grid: Float[Array, " Q Q Q"]
    tilt_x_grid: Float[Array, " Q Q Q"]
    tilt_y_grid: Float[Array, " Q Q Q"]
    energy_grid, tilt_x_grid, tilt_y_grid = jnp.meshgrid(
        energy_nodes,
        tilt_nodes,
        tilt_nodes,
        indexing="ij",
    )
    weight_energy: Float[Array, " Q Q Q"]
    weight_tilt_x: Float[Array, " Q Q Q"]
    weight_tilt_y: Float[Array, " Q Q Q"]
    weight_energy, weight_tilt_x, weight_tilt_y = jnp.meshgrid(
        weights,
        weights,
        weights,
        indexing="ij",
    )
    samples: Float[Array, " N 3"] = jnp.stack(
        (
            energy_grid.reshape(-1),
            tilt_x_grid.reshape(-1),
            tilt_y_grid.reshape(-1),
        ),
        axis=1,
    )
    sample_weights: Float[Array, " N"] = (
        weight_energy * weight_tilt_x * weight_tilt_y
    ).reshape(-1)
    distribution: Distribution = create_distribution(
        samples=samples,
        weights=sample_weights,
        reduction=ReductionMode.INCOHERENT,
        axis_id=_AXIS_COHERENCE,
    )
    return distribution


@jaxtyped(typechecker=beartype)
def position_jitter_to_distribution(
    sigma_ang: scalar_float,
    n_quad: scalar_int,
) -> Distribution:
    """Build an incoherent two-dimensional position-jitter distribution.

    Parameters
    ----------
    sigma_ang : scalar_float
        Gaussian standard deviation for independent y/x position deltas in
        Angstroms.
    n_quad : scalar_int
        Static number of Gauss-Hermite nodes per axis.

    Returns
    -------
    distribution : Distribution
        Incoherent distribution with ``axis_id="position_jitter"`` and
        samples ``[dy_ang, dx_ang]``. The binder folds these columns into
        ``AxisUpdate.position_delta_ang``.

    Notes
    -----
    Source-size broadening uses this same producer contract. The
    zero-width path collapses all quadrature samples to zero with a finite
    derivative at exactly zero width.
    
    :see: bind_cbed_axes, coherence_to_distribution.
    """
    nodes: Float[Array, " Q"]
    weights: Float[Array, " Q"]
    nodes, weights = _normal_gauss_hermite_rule(n_quad)
    delta_nodes: Float[Array, " Q"] = _collapse_width(
        sigma_ang,
        nodes,
        "sigma_ang",
    )
    delta_y_grid: Float[Array, " Q Q"]
    delta_x_grid: Float[Array, " Q Q"]
    delta_y_grid, delta_x_grid = jnp.meshgrid(
        delta_nodes,
        delta_nodes,
        indexing="ij",
    )
    weight_y_grid: Float[Array, " Q Q"]
    weight_x_grid: Float[Array, " Q Q"]
    weight_y_grid, weight_x_grid = jnp.meshgrid(
        weights,
        weights,
        indexing="ij",
    )
    samples: Float[Array, " N 2"] = jnp.stack(
        (delta_y_grid.reshape(-1), delta_x_grid.reshape(-1)),
        axis=1,
    )
    sample_weights: Float[Array, " N"] = (
        weight_y_grid * weight_x_grid
    ).reshape(-1)
    distribution: Distribution = create_distribution(
        samples=samples,
        weights=sample_weights,
        reduction=ReductionMode.INCOHERENT,
        axis_id=_AXIS_POSITION_JITTER,
    )
    return distribution


@jaxtyped(typechecker=beartype)
def _collapse_width(
    width: scalar_float,
    nodes: Float[Array, " Q"],
    name: str,
) -> Float[Array, " Q"]:
    """Scale GH nodes while collapsing nonpositive widths to zero safely."""
    width_arr: Float[Array, " "] = jnp.asarray(width, dtype=jnp.float64)
    if width_arr.shape != ():
        raise ValueError(f"{name} must be a scalar")
    checked_width: Float[Array, " "] = eqx.error_if(
        width_arr,
        ~jnp.isfinite(width_arr),
        f"{name} must be finite",
    )
    checked_width = eqx.error_if(
        checked_width,
        checked_width < 0.0,
        f"{name} must be non-negative",
    )
    scaled_nodes: Float[Array, " Q"] = jnp.where(
        checked_width > 0.0,
        checked_width * nodes,
        jnp.zeros_like(nodes),
    )
    return scaled_nodes


def _normal_gauss_hermite_rule(
    n_quad: scalar_int,
) -> tuple[Float[Array, " Q"], Float[Array, " Q"]]:
    """Return standard-normal GH nodes and normalized weights."""
    n_quad_int: int = int(n_quad)
    if n_quad_int <= 0:
        raise ValueError("n_quad must be positive")
    nodes_np: np.ndarray
    weights_np: np.ndarray
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_quad_int)
    nodes: Float[Array, " Q"] = jnp.asarray(
        np.sqrt(2.0) * nodes_np,
        dtype=jnp.float64,
    )
    weights: Float[Array, " Q"] = jnp.asarray(
        weights_np / np.sqrt(np.pi),
        dtype=jnp.float64,
    )
    return nodes, weights


__all__: list[str] = [
    "coherence_to_distribution",
    "position_jitter_to_distribution",
]
