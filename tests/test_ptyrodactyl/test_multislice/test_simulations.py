"""Tests for :mod:`ptyrodactyl.multislice.simulations`."""
# ruff: noqa: E402, I001

import inspect
import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.multislice import (
    apply_distribution,
    apply_distributions,
    bind_cbed_axes,
    checked_cbed_image,
    checked_make_probe,
    checked_stem4d_sharded,
    checked_stem_4d,
    coherence_to_distribution,
    position_jitter_to_distribution,
)
from ptyrodactyl.multislice.parallelized import stem4d_sharded
from ptyrodactyl.multislice.simulations import (
    aberration,
    annular_detector,
    cbed_amplitude,
    cbed_image,
    decompose_beam_to_modes,
    make_probe,
    probe_modes_to_distribution,
    stem_4d,
)
from ptyrodactyl.types import (
    ReductionMode,
    create_calibrated_array,
    create_detector_config,
    create_ensemble_axes,
    create_distribution,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
    create_stem4d,
)


def test_public_integrator_signatures_have_at_most_six_parameters() -> None:
    """Carrierized public integrators stay below the IM7 signature cap."""
    public_integrators = (
        aberration,
        annular_detector,
        cbed_amplitude,
        cbed_image,
        checked_cbed_image,
        checked_make_probe,
        checked_stem4d_sharded,
        checked_stem_4d,
        make_probe,
        stem4d_sharded,
        stem_4d,
    )

    for integrator in public_integrators:
        parameters = inspect.signature(integrator).parameters
        assert len(parameters) <= 6, integrator


def test_annular_detector_static_scan_shape_and_jit() -> None:
    """Annular detector reshapes with a static raster and JIT-compiles."""
    positions = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    data = jnp.arange(6 * 4 * 4, dtype=jnp.float64).reshape(6, 4, 4)
    stem4d = create_stem4d(data, 0.5, 0.25, positions, 80.0)
    collection_angles = jnp.array([0.0, 1000.0], dtype=jnp.float64)
    expected = np.array(
        [[120.0, 376.0, 632.0], [888.0, 1144.0, 1400.0]],
        dtype=np.float64,
    )

    detector = create_detector_config(
        real_space_calib_ang=0.5,
        collection_inner_mrad=collection_angles[0],
        collection_outer_mrad=collection_angles[1],
        scan_shape=(2, 3),
    )

    result = annular_detector(stem4d, detector)
    jitted = jax.jit(annular_detector)(stem4d, detector)

    assert np.array_equal(np.asarray(result.data_array), expected)
    assert np.array_equal(np.asarray(jitted.data_array), expected)
    assert np.array_equal(
        np.asarray(result.data_array), np.asarray(jitted.data_array)
    )


def test_decompose_beam_to_modes_requires_key() -> None:
    """Beam decomposition no longer has an implicit PRNG key."""
    beam = create_calibrated_array(
        (jnp.arange(16, dtype=jnp.float64).reshape(4, 4) + 1j).astype(
            jnp.complex128
        ),
        0.5,
        0.5,
        True,
    )

    with pytest.raises(TypeError):
        decompose_beam_to_modes(beam, 3)


def test_decompose_beam_to_modes_fixed_key_reproducible() -> None:
    """A fixed key gives reproducible unscaled modes and weights."""
    beam = create_calibrated_array(
        (jnp.arange(16, dtype=jnp.float64).reshape(4, 4) + 1j).astype(
            jnp.complex128
        ),
        0.5,
        0.5,
        True,
    )

    first = decompose_beam_to_modes(
        beam, 3, jax.random.PRNGKey(0), first_mode_weight=0.6
    )
    second = decompose_beam_to_modes(
        beam, 3, jax.random.PRNGKey(0), first_mode_weight=0.6
    )

    assert np.array_equal(np.asarray(first.modes), np.asarray(second.modes))
    assert np.array_equal(
        np.asarray(first.weights), np.asarray(second.weights)
    )
    assert np.array_equal(np.asarray(first.calib), np.asarray(second.calib))
    assert np.array_equal(
        np.asarray(first.modes[0, 0, 0]),
        np.asarray(-0.34562459928563927 + 0.25753846176370626j),
    )
    assert np.array_equal(
        np.asarray(first.weights),
        np.asarray([0.6, 0.2, 0.2], dtype=np.float64),
    )

    weighted_intensity = jnp.einsum(
        "m,hwm->hw",
        first.weights,
        jnp.abs(first.modes) ** 2,
    )
    old_prescaled_modes = first.modes * jnp.sqrt(first.weights).reshape(
        1, 1, -1
    )
    old_prescaled_intensity = jnp.sum(
        jnp.abs(old_prescaled_modes) ** 2,
        axis=-1,
    )
    np.testing.assert_allclose(
        np.asarray(weighted_intensity),
        np.asarray(old_prescaled_intensity),
        rtol=1.0e-15,
        atol=1.0e-15,
    )


def test_decompose_beam_to_single_mode_jits_with_static_mode_count() -> None:
    """The one-mode boundary avoids traced Python maximum operations."""
    beam = create_calibrated_array(
        jnp.ones((4, 4), dtype=jnp.complex128),
        0.5,
        0.5,
        True,
    )
    jitted_decompose = jax.jit(
        decompose_beam_to_modes,
        static_argnames=("num_modes",),
    )

    result = jitted_decompose(
        beam,
        num_modes=1,
        key=jax.random.PRNGKey(3),
    )

    assert result.modes.shape == (4, 4, 1)
    assert np.array_equal(
        np.asarray(result.weights),
        np.ones((1,), dtype=np.float64),
    )


def test_probe_modes_to_distribution_uses_explicit_weights() -> None:
    """Probe mode distributions carry samples and weights explicitly."""
    modes = jnp.ones((2, 2, 3), dtype=jnp.complex128)
    weights = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64)
    probe = create_probe_modes(modes, weights, 0.5)

    distribution = probe_modes_to_distribution(probe)

    np.testing.assert_array_equal(
        np.asarray(distribution.samples),
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(distribution.weights),
        np.asarray(weights),
        rtol=0.0,
        atol=0.0,
    )
    assert distribution.reduction is ReductionMode.INCOHERENT
    assert distribution.axis_id == "probe_modes"


def test_cbed_image_ensemble_axes_match_explicit_composition() -> None:
    """A jitter axis inside MicroscopeConfig matches explicit composition."""
    pot_slices = create_potential_slices(
        jnp.zeros((8, 8, 1), dtype=jnp.float64),
        1.0,
        0.5,
    )
    detector = create_detector_config(
        real_space_calib_ang=0.5,
        probe_calibration_pm=50.0,
    )
    probe = make_probe(
        create_microscope_config(80.0, 25.0, probe_shape=(8, 8)),
        detector,
    )
    beam = create_probe_modes(
        probe[..., jnp.newaxis],
        jnp.ones((1,), dtype=jnp.float64),
        0.5,
    )
    jitter = position_jitter_to_distribution(0.15, 2)
    ensemble = create_ensemble_axes(position_jitter=jitter)
    microscope = create_microscope_config(
        80.0,
        25.0,
        ensemble=ensemble,
        probe_shape=(8, 8),
    )
    axes = (jitter,)
    bound = bind_cbed_axes(
        pot_slices=pot_slices,
        probe_modes=beam,
        microscope=microscope,
        detector=detector,
        axes=axes,
    )
    explicit = apply_distributions(axes, bound)
    carried = cbed_image(pot_slices, beam, microscope).data_array

    assert np.array_equal(np.asarray(carried), np.asarray(explicit))


def test_coherent_probe_distribution_has_finite_weight_grad() -> None:
    """A coherent toy mode distribution remains differentiable in weights."""
    samples = jnp.arange(2, dtype=jnp.float64)[:, jnp.newaxis]
    amplitudes = jnp.stack(
        [
            jnp.array([[1.0 + 0.5j, 0.25 - 0.75j]], dtype=jnp.complex128),
            jnp.array([[0.5 - 0.25j, -1.0 + 0.125j]], dtype=jnp.complex128),
        ],
        axis=-1,
    )

    def loss(weights):
        distribution = create_distribution(
            samples=samples,
            weights=weights,
            reduction=ReductionMode.COHERENT,
            axis_id="probe_modes",
        )

        def bound(sample):
            mode_idx = sample[0].astype(jnp.int32)
            return amplitudes[..., mode_idx]

        return jnp.sum(apply_distribution(distribution, bound))

    grad_value = jax.grad(loss)(jnp.array([0.25, 0.75], dtype=jnp.float64))

    assert np.all(np.isfinite(np.asarray(grad_value)))


def test_forward_carrier_dynamic_scalars_have_finite_gradients() -> None:
    """Physical carrier leaves remain dynamic under an end-to-end gradient."""
    pot_slices = create_potential_slices(
        jnp.zeros((8, 8, 1), dtype=jnp.float64),
        1.0,
        0.5,
    )
    detector = create_detector_config(
        real_space_calib_ang=0.5,
        probe_calibration_pm=50.0,
    )

    def objective(params):
        voltage = params[0]
        aperture = params[1]
        defocus = params[2]
        jitter_sigma = params[3]
        energy_width = params[4]
        angular_width = params[5]
        first_weight = params[6]
        probe_microscope = create_microscope_config(
            voltage_kv=voltage,
            aperture_mrad=aperture,
            defocus_ang=defocus,
            probe_shape=(8, 8),
        )
        probe = make_probe(probe_microscope, detector)
        modes = jnp.stack((probe, jnp.roll(probe, shift=1, axis=0)), axis=2)
        weights = jnp.stack((first_weight, 1.0 - first_weight))
        beam = create_probe_modes(modes, weights, 0.5)
        ensemble = create_ensemble_axes(
            position_jitter=position_jitter_to_distribution(jitter_sigma, 2),
            coherence=coherence_to_distribution(
                energy_width,
                angular_width,
                2,
            ),
        )
        microscope = create_microscope_config(
            voltage_kv=voltage,
            aperture_mrad=aperture,
            defocus_ang=defocus,
            ensemble=ensemble,
            probe_shape=(8, 8),
        )
        return jnp.sum(cbed_image(pot_slices, beam, microscope).data_array)

    params = jnp.array(
        [80.0, 25.0, 1.5, 0.05, 0.1, 0.02, 0.6],
        dtype=jnp.float64,
    )
    grad_value = jax.grad(objective)(params)

    assert grad_value.shape == params.shape
    assert np.all(np.isfinite(np.asarray(grad_value)))


def test_cbed_amplitude_phase_gradient_survives_intensity_seam() -> None:
    """Amplitude keeps phase information that the intensity path removes."""
    pot_slices = create_potential_slices(
        jnp.zeros((4, 4, 1), dtype=jnp.float64),
        1.0,
        0.5,
    )
    base = (
        jnp.arange(1, 17, dtype=jnp.float64).reshape(4, 4)
        + 0.125j
    ).astype(jnp.complex128)

    def probe_with_phase(phase):
        modes = (base * jnp.exp(1j * phase))[..., jnp.newaxis]
        return create_probe_modes(
            modes,
            jnp.ones((1,), dtype=jnp.float64),
            0.5,
        )

    def amplitude_loss(phase):
        amplitudes = cbed_amplitude(
            pot_slices,
            probe_with_phase(phase),
            create_microscope_config(80.0, 1.0),
        )
        return jnp.real(jnp.sum(amplitudes))

    def intensity_loss(phase):
        image = cbed_image(
            pot_slices,
            probe_with_phase(phase),
            create_microscope_config(80.0, 1.0),
        )
        return jnp.sum(image.data_array)

    phase = jnp.asarray(0.37, dtype=jnp.float64)
    amplitude_grad = jax.grad(amplitude_loss)(phase)
    intensity_grad = jax.grad(intensity_loss)(phase)

    assert np.isfinite(np.asarray(amplitude_grad))
    assert np.isfinite(np.asarray(intensity_grad))
    assert not np.allclose(
        np.asarray(amplitude_grad),
        np.asarray(intensity_grad),
        rtol=0.0,
        atol=1.0e-8,
    )
