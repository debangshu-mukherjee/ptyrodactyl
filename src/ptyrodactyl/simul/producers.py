"""Distribution producers and binders for CBED ensemble axes.

Extended Summary
----------------
This module turns physical CBED averages into explicit
:class:`~ptyrodactyl.types.Distribution` axes. Producers own quadrature
construction; the binder owns the cursor walk from concatenated distribution
samples into one additive :class:`~ptyrodactyl.types.AxisUpdate`.

Source-size broadening maps onto the same incoherent position-delta axis as
``position_jitter_to_distribution``. A separate source-size carrier would
duplicate the position-update field without changing the kernel contract, and
Plan 03 W3d authorizes folding that physical average into position jitter.

Routine Listings
----------------
:func:`bind_cbed_axes`
    Bind distribution cursor rows to the single-mode CBED amplitude kernel.
:func:`coherence_to_distribution`
    Build the incoherent chromatic/angular coherence distribution.
:func:`position_jitter_to_distribution`
    Build the incoherent two-dimensional position-jitter distribution.
"""

import numpy as np
import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Complex, Float, jaxtyped

from ptyrodactyl.tools import relativistic_wavelength_ang
from ptyrodactyl.types import (
    AxisUpdate,
    Distribution,
    PotentialSlices,
    ProbeModes,
    ReductionMode,
    combine_axis_updates,
    create_axis_update,
    create_distribution,
    scalar_float,
    scalar_int,
    scalar_num,
)

from .simulations import cbed_amplitude, shift_beam_fourier

_AXIS_COHERENCE: str = "coherence"
_AXIS_POSITION_JITTER: str = "position_jitter"
_AXIS_PROBE_MODES: str = "probe_modes"
_COHERENCE_DIM: int = 3
_POSITION_DIM: int = 2
_PROBE_MODE_DIM: int = 1


@jaxtyped(typechecker=beartype)
def bind_cbed_axes(
    pot_slices: PotentialSlices,
    probe_modes: ProbeModes,
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
    axes: tuple[Distribution, ...],
    column_maps: tuple[str, ...] = (),
) -> Callable[[Float[Array, " D"]], Complex[Array, " H W"]]:
    """Return a CBED amplitude closure bound to distribution-axis columns.

    Parameters
    ----------
    pot_slices : PotentialSlices
        Potential slices passed unchanged to :func:`cbed_amplitude`.
    probe_modes : ProbeModes
        Probe modes. If a ``"probe_modes"`` axis is present, it must be the
        first axis and its single sample column is the mode index. Otherwise
        ``probe_modes`` must contain exactly one mode.
    voltage_kv : scalar_num
        Nominal accelerating voltage in kilovolts.
    calib_ang : scalar_float
        Real-space pixel size used for position shifts and tilt ramps.
    axes : tuple[Distribution, ...]
        Distribution axes whose sample rows are concatenated by
        :func:`~ptyrodactyl.simul.reduce.apply_distributions`.
    column_maps : tuple[str, ...], optional
        Static column-map names. An empty tuple derives them from each
        distribution ``axis_id``. Supported names are ``"probe_modes"``,
        ``"position_jitter"``, and ``"coherence"``.

    Returns
    -------
    bound_amplitude_fn : Callable[[Float[Array, " D"]], Complex[Array, " H W"]]
        Closure that maps one concatenated cursor row to one complex CBED
        detector field.

    Raises
    ------
    ValueError
        If any axis id/map is unknown, dimensions do not match the static
        binder contract, or a multimode probe lacks a probe-mode axis.

    Notes
    -----
    The column contracts are:

    - ``"probe_modes"``: ``[mode_index]``; selects one retained probe mode.
    - ``"position_jitter"``: ``[dy_ang, dx_ang]``; folds to
      ``AxisUpdate.position_delta_ang``.
    - ``"coherence"``: ``[dE_ev, dtheta_x_mrad, dtheta_y_mrad]``; folds to
      ``AxisUpdate.energy_delta_ev`` and ``AxisUpdate.tilt_delta_mrad``.
    """
    axis_maps: tuple[str, ...] = _resolve_column_maps(axes, column_maps)
    _validate_axis_maps(axes, axis_maps, probe_modes)
    expected_dim: int = sum(axis.samples.shape[1] for axis in axes)
    nominal_voltage_kv: Float[Array, " "] = jnp.asarray(
        voltage_kv,
        dtype=jnp.float64,
    )

    def bound_amplitude_fn(
        sample: Float[Array, " D"],
    ) -> Complex[Array, " H W"]:
        """Map one concatenated sample row to a single-mode CBED field."""
        if sample.shape[0] != expected_dim:
            raise ValueError("sample width does not match bound axes")

        update: AxisUpdate
        mode_idx: Array
        update, mode_idx = _axis_update_from_sample(sample, axes, axis_maps)
        updated_voltage_kv: Float[Array, " "] = (
            nominal_voltage_kv + update.energy_delta_ev / 1000.0
        )
        selected_modes: Complex[Array, " H W 1"] = jnp.take(
            probe_modes.modes,
            mode_idx.astype(jnp.int32),
            axis=2,
        )[..., jnp.newaxis]
        shifted_modes_all: Complex[Array, " 1 H W 1"] = shift_beam_fourier(
            selected_modes,
            update.position_delta_ang,
            calib_ang,
        )
        shifted_modes: Complex[Array, " H W 1"] = shifted_modes_all[0]
        tilted_modes: Complex[Array, " H W 1"] = _apply_tilt_phase_ramp(
            shifted_modes,
            update.tilt_delta_mrad,
            updated_voltage_kv,
            calib_ang,
        )
        bound_probe: ProbeModes = ProbeModes(
            modes=tilted_modes,
            weights=jnp.ones((1,), dtype=jnp.float64),
            calib=probe_modes.calib,
        )
        amplitudes: Complex[Array, " H W 1"] = cbed_amplitude(
            pot_slices,
            bound_probe,
            updated_voltage_kv,
        )
        amplitude: Complex[Array, " H W"] = amplitudes[..., 0]
        return amplitude

    return bound_amplitude_fn


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
def _apply_tilt_phase_ramp(
    modes: Complex[Array, " H W M"],
    tilt_delta_mrad: Float[Array, " 2"],
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
) -> Complex[Array, " H W M"]:
    r"""Apply a small-angle incident-tilt phase ramp to real-space modes.

    The convention is ``exp(i k dot r)`` with
    ``k = 2 pi theta / lambda``. Tilt columns are stored as
    ``[dtheta_x_mrad, dtheta_y_mrad]`` and converted to radians.
    """
    height: int = modes.shape[0]
    width: int = modes.shape[1]
    y_coords: Float[Array, " H"] = (
        jnp.arange(height, dtype=jnp.float64) - (height - 1) / 2.0
    ) * calib_ang
    x_coords: Float[Array, " W"] = (
        jnp.arange(width, dtype=jnp.float64) - (width - 1) / 2.0
    ) * calib_ang
    yy: Float[Array, " H W"]
    xx: Float[Array, " H W"]
    yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    theta_x_rad: Float[Array, " "] = tilt_delta_mrad[0] / 1000.0
    theta_y_rad: Float[Array, " "] = tilt_delta_mrad[1] / 1000.0
    wavelength_ang: Float[Array, " "] = relativistic_wavelength_ang(
        voltage_kv,
    )
    phase: Float[Array, " H W"] = (2.0 * jnp.pi / wavelength_ang) * (
        theta_x_rad * xx + theta_y_rad * yy
    )
    ramp: Complex[Array, " H W"] = jnp.exp(1j * phase).astype(modes.dtype)
    tilted_modes: Complex[Array, " H W M"] = modes * ramp[..., jnp.newaxis]
    return tilted_modes


@jaxtyped(typechecker=beartype)
def _axis_update_from_sample(
    sample: Float[Array, " D"],
    axes: tuple[Distribution, ...],
    axis_maps: tuple[str, ...],
) -> tuple[AxisUpdate, Array]:
    """Fold one concatenated cursor row into an AxisUpdate and mode index."""
    cursor: int = 0
    updates: list[AxisUpdate] = []
    mode_idx: Array = jnp.asarray(0, dtype=jnp.int32)
    for axis, axis_map in zip(axes, axis_maps, strict=True):
        axis_dim: int = axis.samples.shape[1]
        columns: Float[Array, " K"] = sample[cursor : cursor + axis_dim]
        if axis_map == _AXIS_PROBE_MODES:
            mode_idx = columns[0].astype(jnp.int32)
            axis_update: AxisUpdate = create_axis_update()
        elif axis_map == _AXIS_POSITION_JITTER:
            axis_update = create_axis_update(position_delta_ang=columns)
        elif axis_map == _AXIS_COHERENCE:
            axis_update = create_axis_update(
                energy_delta_ev=columns[0],
                tilt_delta_mrad=columns[1:3],
            )
        else:
            raise ValueError(f"unknown axis_id {axis_map!r}")
        updates.append(axis_update)
        cursor += axis_dim

    update: AxisUpdate = combine_axis_updates(tuple(updates))
    return update, mode_idx


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


def _expected_axis_dim(axis_map: str) -> int:
    """Return the static sample-column count for one supported axis map."""
    if axis_map == _AXIS_PROBE_MODES:
        axis_dim: int = _PROBE_MODE_DIM
    elif axis_map == _AXIS_POSITION_JITTER:
        axis_dim = _POSITION_DIM
    elif axis_map == _AXIS_COHERENCE:
        axis_dim = _COHERENCE_DIM
    else:
        raise ValueError(f"unknown axis_id {axis_map!r}")
    return axis_dim


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


def _resolve_column_maps(
    axes: tuple[Distribution, ...],
    column_maps: tuple[str, ...],
) -> tuple[str, ...]:
    """Return explicit static column-map names for all axes."""
    if len(column_maps) == 0:
        resolved_maps: tuple[str, ...] = tuple(
            _require_axis_id(axis) for axis in axes
        )
    elif len(column_maps) == len(axes):
        resolved_maps = column_maps
    else:
        raise ValueError("column_maps must be empty or match axes length")
    return resolved_maps


def _require_axis_id(axis: Distribution) -> str:
    """Return a distribution axis_id or raise for missing metadata."""
    if axis.axis_id is None:
        raise ValueError("distribution axis_id is required")
    axis_id: str = axis.axis_id
    return axis_id


def _validate_axis_maps(
    axes: tuple[Distribution, ...],
    axis_maps: tuple[str, ...],
    probe_modes: ProbeModes,
) -> None:
    """Validate static axis-map dimensions and probe-mode composition."""
    for axis_index, (axis, axis_map) in enumerate(
        zip(axes, axis_maps, strict=True),
    ):
        expected_dim: int = _expected_axis_dim(axis_map)
        actual_dim: int = axis.samples.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"{axis_map!r} axis must have sample width {expected_dim}",
            )
        if axis_map == _AXIS_PROBE_MODES and axis_index != 0:
            raise ValueError("probe_modes axis must be first")

    mode_count: int = probe_modes.modes.shape[2]
    if _AXIS_PROBE_MODES not in axis_maps and mode_count != 1:
        raise ValueError("multimode probes require a first probe_modes axis")


__all__: list[str] = [
    "bind_cbed_axes",
    "coherence_to_distribution",
    "position_jitter_to_distribution",
]
