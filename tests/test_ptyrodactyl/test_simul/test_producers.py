"""Tests for :mod:`ptyrodactyl.simul.producers`."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ptyrodactyl.simul import (
    apply_distributions,
    bind_cbed_axes,
    cbed_image,
    coherence_to_distribution,
    make_probe,
    position_jitter_to_distribution,
    probe_modes_to_distribution,
)
from ptyrodactyl.simul.producers import _axis_update_from_sample
from ptyrodactyl.types import (
    ReductionMode,
    create_detector_config,
    create_distribution,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
)

_CALIB_ANG = 0.5
_VOLTAGE_KV = 80.0


def _microscope():
    return create_microscope_config(
        voltage_kv=_VOLTAGE_KV,
        aperture_mrad=25.0,
        probe_shape=(8, 8),
    )


def _detector():
    return create_detector_config(
        real_space_calib_ang=_CALIB_ANG,
        probe_calibration_pm=_CALIB_ANG * 100.0,
    )


def _tiny_potential():
    y_coords = (jnp.arange(8, dtype=jnp.float64) - 3.5) * _CALIB_ANG
    x_coords = (jnp.arange(8, dtype=jnp.float64) - 3.5) * _CALIB_ANG
    yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    centered = 1800.0 * jnp.exp(-((yy + 0.2) ** 2 + (xx - 0.1) ** 2) / 0.18)
    offset = 1200.0 * jnp.exp(-((yy - 0.4) ** 2 + (xx + 0.3) ** 2) / 0.28)
    slices = jnp.stack((centered, offset), axis=2)
    pot_slices = create_potential_slices(slices, 1.0, _CALIB_ANG)
    return pot_slices


def _probe_modes(mode_count=1, scale=1.0):
    base = make_probe(_microscope(), _detector())
    first_mode = scale * base
    if mode_count == 1:
        modes = first_mode[..., jnp.newaxis]
        weights = jnp.ones((1,), dtype=jnp.float64)
    else:
        second_mode = jnp.roll(base, shift=1, axis=0)
        modes = jnp.stack((first_mode, second_mode), axis=2)
        weights = jnp.asarray([0.35, 0.65], dtype=jnp.float64)
    probe_modes = create_probe_modes(modes, weights, _CALIB_ANG)
    return probe_modes


def _cbed_loss_for_axes(axes, beam):
    pot_slices = _tiny_potential()
    bound = bind_cbed_axes(
        pot_slices,
        beam,
        _microscope(),
        _detector(),
        axes,
        (),
    )
    intensity = apply_distributions(axes, bound)
    loss = jnp.sum(intensity)
    return loss


def test_zero_width_axes_are_noop_against_plain_cbed_image():
    pot_slices = _tiny_potential()
    beam = _probe_modes(mode_count=2)
    mode_axis = probe_modes_to_distribution(beam)
    jitter = position_jitter_to_distribution(0.0, 1)
    coherence = coherence_to_distribution(0.0, 0.0, 1)
    axes = (mode_axis, jitter, coherence)
    bound = bind_cbed_axes(
        pot_slices,
        beam,
        _microscope(),
        _detector(),
        axes,
        (),
    )
    expected = cbed_image(pot_slices, beam, _microscope()).data_array

    actual = apply_distributions(axes, bound)

    chex.assert_trees_all_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_cursor_walk_two_by_two_columns_arrive_in_tuple_order():
    position_axis = create_distribution(
        samples=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64),
        weights=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        reduction=ReductionMode.INCOHERENT,
        axis_id="position_jitter",
    )
    coherence_axis = create_distribution(
        samples=jnp.asarray(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            dtype=jnp.float64,
        ),
        weights=jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        reduction=ReductionMode.INCOHERENT,
        axis_id="coherence",
    )
    axes = (position_axis, coherence_axis)

    def echo_bound(cursor):
        update, _ = _axis_update_from_sample(
            cursor,
            axes,
            ("position_jitter", "coherence"),
        )
        row0 = jnp.stack(
            (
                update.position_delta_ang[0]
                + 1j * update.position_delta_ang[1],
                update.energy_delta_ev + 1j * update.tilt_delta_mrad[0],
            ),
        )
        row1 = jnp.stack((update.tilt_delta_mrad[1] + 0.0j, 0.0 + 0.0j))
        field = jnp.stack((row0, row1)).astype(jnp.complex128)
        return field

    expected_field = jnp.asarray(
        [[3.0 + 4.0j, 10.0 + 20.0j], [30.0 + 0.0j, 0.0 + 0.0j]],
        dtype=jnp.complex128,
    )
    expected = jnp.abs(expected_field) ** 2

    result = apply_distributions(axes, echo_bound)

    chex.assert_trees_all_close(result, expected, rtol=0.0, atol=0.0)


def test_jitter_sigma_zero_and_near_zero_gradients_are_finite():
    beam = _probe_modes()

    def objective(sigma_ang):
        jitter = position_jitter_to_distribution(sigma_ang, 2)
        value = _cbed_loss_for_axes((jitter,), beam)
        return value

    zero_grad = jax.grad(objective)(jnp.asarray(0.0, dtype=jnp.float64))
    near_zero_grad = jax.grad(objective)(
        jnp.asarray(1.0e-30, dtype=jnp.float64),
    )

    assert np.isfinite(np.asarray(zero_grad))
    assert np.isfinite(np.asarray(near_zero_grad))


def test_coherence_zero_and_near_zero_gradients_are_finite():
    beam = _probe_modes()

    def energy_objective(energy_spread_ev):
        coherence = coherence_to_distribution(energy_spread_ev, 0.0, 2)
        value = _cbed_loss_for_axes((coherence,), beam)
        return value

    def divergence_objective(angular_divergence_mrad):
        coherence = coherence_to_distribution(0.0, angular_divergence_mrad, 2)
        value = _cbed_loss_for_axes((coherence,), beam)
        return value

    energy_zero_grad = jax.grad(energy_objective)(
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    energy_near_zero_grad = jax.grad(energy_objective)(
        jnp.asarray(1.0e-30, dtype=jnp.float64),
    )
    divergence_zero_grad = jax.grad(divergence_objective)(
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    divergence_near_zero_grad = jax.grad(divergence_objective)(
        jnp.asarray(1.0e-30, dtype=jnp.float64),
    )

    assert np.isfinite(np.asarray(energy_zero_grad))
    assert np.isfinite(np.asarray(energy_near_zero_grad))
    assert np.isfinite(np.asarray(divergence_zero_grad))
    assert np.isfinite(np.asarray(divergence_near_zero_grad))


def test_real_tiny_cbed_bind_has_finite_end_to_end_gradients():
    def jitter_objective(sigma_ang):
        beam = _probe_modes()
        jitter = position_jitter_to_distribution(sigma_ang, 2)
        value = _cbed_loss_for_axes((jitter,), beam)
        return value

    def energy_objective(energy_spread_ev):
        beam = _probe_modes()
        coherence = coherence_to_distribution(energy_spread_ev, 0.02, 2)
        value = _cbed_loss_for_axes((coherence,), beam)
        return value

    def probe_objective(scale):
        beam = _probe_modes(scale=scale)
        jitter = position_jitter_to_distribution(0.05, 2)
        value = _cbed_loss_for_axes((jitter,), beam)
        return value

    jitter_grad = jax.grad(jitter_objective)(
        jnp.asarray(0.05, dtype=jnp.float64),
    )
    energy_grad = jax.grad(energy_objective)(
        jnp.asarray(0.1, dtype=jnp.float64),
    )
    probe_grad = jax.grad(probe_objective)(
        jnp.asarray(1.0, dtype=jnp.float64),
    )

    assert np.isfinite(np.asarray(jitter_grad))
    assert np.isfinite(np.asarray(energy_grad))
    assert np.isfinite(np.asarray(probe_grad))


def test_nonzero_jitter_reduces_focused_probe_peak_intensity():
    beam = _probe_modes()
    zero_jitter = position_jitter_to_distribution(0.0, 1)
    nonzero_jitter = position_jitter_to_distribution(0.35, 3)

    zero_intensity = _cbed_loss_image((zero_jitter,), beam)
    nonzero_intensity = _cbed_loss_image((nonzero_jitter,), beam)

    assert jnp.max(nonzero_intensity) < jnp.max(zero_intensity)


def _cbed_loss_image(axes, beam):
    pot_slices = _tiny_potential()
    bound = bind_cbed_axes(
        pot_slices,
        beam,
        _microscope(),
        _detector(),
        axes,
        (),
    )
    intensity = apply_distributions(axes, bound)
    return intensity
