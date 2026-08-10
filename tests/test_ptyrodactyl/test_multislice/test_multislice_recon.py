"""Tests for :mod:`ptyrodactyl.multislice.multislice_recon`.

:see: :obj:`ptyrodactyl.multislice.OPTIMIZERS`
"""
# ruff: noqa: E402, I001

from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import ptyrodactyl.multislice.multislice_recon as multislice_recon
from ptyrodactyl.multislice.multislice_recon import (
    multi_slice_multi_modal,
    single_slice_multi_modal,
    single_slice_poscorrected,
    single_slice_ptychography,
)
from ptyrodactyl.types import (
    DetectorConfig,
    MicroscopeConfig,
    PotentialSlices,
    ProbeModes,
    create_calibrated_array,
    create_probe_modes,
    create_stem4d,
)


def _tiny_recon_inputs():
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 5)
    shape = (4, 4)
    positions = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    experimental = create_stem4d(
        jnp.ones((2, *shape), dtype=jnp.float64),
        1.0,
        0.25,
        positions,
        80.0,
    )
    potential_array = (
        jax.random.normal(keys[0], shape, dtype=jnp.float64) * 0.01
    ).astype(jnp.complex128)
    potential = create_calibrated_array(potential_array, 1.0, 1.0, True)
    beam_array = (
        jax.random.normal(keys[1], shape, dtype=jnp.float64)
        + 1j * jax.random.normal(keys[2], shape, dtype=jnp.float64)
    ).astype(jnp.complex128)
    beam = create_calibrated_array(beam_array, 1.0, 1.0, True)
    mode_array = jnp.stack([beam_array, beam_array * (0.5 + 0.1j)], axis=-1)
    modes = create_probe_modes(
        modes=mode_array,
        weights=jnp.array([0.7, 0.3], dtype=jnp.float64),
        calib=jnp.array(1.0, dtype=jnp.float64),
    )
    return experimental, potential, beam, potential_array, beam_array, modes


def _assert_scalar_equal(actual, expected) -> None:
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    ("optimizer_name", "loss_type"),
    (("adam", "mae"), ("adagrad", "mse"), ("rmsprop", "rmse")),
)
def test_public_reconstruction_descends_complex_beam_loss(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_name: str,
    loss_type: str,
) -> None:
    """All advertised Optax/loss paths descend an imaginary-beam loss."""
    shape = (2, 2)
    experimental = create_stem4d(
        jnp.zeros((1, *shape), dtype=jnp.float64),
        1.0,
        0.25,
        jnp.zeros((1, 2), dtype=jnp.float64),
        80.0,
    )
    potential = create_calibrated_array(
        jnp.zeros(shape, dtype=jnp.float32),
        1.0,
        1.0,
        True,
    )
    beam = create_calibrated_array(
        1j * jnp.ones(shape, dtype=jnp.complex64),
        1.0,
        1.0,
        True,
    )

    def _imaginary_beam_stem4d(
        potential_slices: PotentialSlices,
        probe_modes: ProbeModes,
        microscope: MicroscopeConfig,
        detector: DetectorConfig,
    ) -> SimpleNamespace:
        """Expose the beam's imaginary part as detector data."""
        del potential_slices, microscope, detector
        simulated = jnp.imag(probe_modes.modes[..., 0])[jnp.newaxis, ...]
        return SimpleNamespace(data=simulated)

    monkeypatch.setattr(multislice_recon, "stem_4d", _imaginary_beam_stem4d)

    output = single_slice_ptychography(
        experimental,
        potential,
        beam,
        1.0,
        save_every=1,
        num_iterations=1,
        learning_rate=0.001,
        loss_type=loss_type,
        optimizer_name=optimizer_name,
    )

    initial_loss = jnp.mean(jnp.imag(beam.data_array) ** 2)
    final_loss = jnp.mean(jnp.imag(output[1].data_array) ** 2)
    assert output[0].data_array.dtype == jnp.complex64
    assert output[1].data_array.dtype == jnp.complex64
    assert final_loss < initial_loss


def test_public_reconstruction_promotes_real_reciprocal_beam() -> None:
    """Promote a real beam before selecting the inverse-Fourier branch."""
    experimental, potential, beam, *_ = _tiny_recon_inputs()
    real_reciprocal_beam = create_calibrated_array(
        jnp.real(beam.data_array),
        beam.calib_y,
        beam.calib_x,
        False,
    )

    output = single_slice_ptychography(
        experimental,
        potential,
        real_reciprocal_beam,
        1.0,
        save_every=1,
        num_iterations=1,
        learning_rate=0.001,
    )

    assert output[1].data_array.dtype == jnp.complex128


@pytest.mark.parametrize(
    ("keyword", "selection"),
    (("optimizer_name", "bogus"), ("loss_type", "bogus")),
)
def test_public_reconstruction_rejects_unknown_selection(
    keyword: str,
    selection: str,
) -> None:
    """Unknown optimizer and loss selections fail at setup."""
    experimental, potential, beam, *_ = _tiny_recon_inputs()

    with pytest.raises(ValueError):
        single_slice_ptychography(
            experimental,
            potential,
            beam,
            1.0,
            save_every=1,
            num_iterations=1,
            **{keyword: selection},
        )


def test_single_slice_ptychography_runs_and_regresses() -> None:
    """The base reconstructor runs and returns fixed-seed values.

    :see: :func:`ptyrodactyl.multislice.single_slice_ptychography`
    """
    experimental, potential, beam, *_ = _tiny_recon_inputs()

    output = single_slice_ptychography(
        experimental,
        potential,
        beam,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[2].shape == (4, 4, 2)
    assert output[3].shape == (4, 4, 2)
    _assert_scalar_equal(output[0].data_array[0, 0], 0.02866302842954916 + 0j)
    _assert_scalar_equal(
        output[1].data_array[0, 0],
        -1.4996974690426799 - 1.4814616332191217j,
    )
    _assert_scalar_equal(output[2][0, 0, 0], 0.027663112105862656 + 0j)


def test_single_slice_poscorrected_regresses() -> None:
    """Position-corrected single-slice reconstruction is fixed-seed stable.

    :see: :func:`ptyrodactyl.multislice.single_slice_poscorrected`
    """
    experimental, potential, beam, *_ = _tiny_recon_inputs()

    output = single_slice_poscorrected(
        experimental,
        potential,
        beam,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[2].shape == (2, 2)
    assert output[5].shape == (2, 2, 2)
    _assert_scalar_equal(output[0].data_array[0, 0], 0.02866303668627853 + 0j)
    _assert_scalar_equal(
        output[1].data_array[0, 0],
        -1.4996974690390232 - 1.4814616332263657j,
    )
    _assert_scalar_equal(output[2][0, 0], 0.002001176434824531)


def test_single_slice_multi_modal_regresses() -> None:
    """Multi-modal single-slice reconstruction is fixed-seed stable.

    :see: :func:`ptyrodactyl.multislice.single_slice_multi_modal`
    """
    experimental, _, _, potential_array, _, modes = _tiny_recon_inputs()

    output = single_slice_multi_modal(
        experimental,
        potential_array,
        modes,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[3].shape == (4, 4, 2)
    assert output[4].shape == (4, 4, 2, 2)
    # The explicit ProbeModes weights ([0.7, 0.3] here) determine this pinned
    # result. The obsolete CBED kernel silently ignored nonuniform weights.
    _assert_scalar_equal(output[0][0, 0], 0.0286630285066058 + 0j)
    _assert_scalar_equal(
        output[1].modes[0, 0, 0],
        -1.4996968965393802 - 1.481462648068354j,
    )
    _assert_scalar_equal(output[2][0, 0], 0.0020011719351587594)


def test_multi_slice_multi_modal_regresses() -> None:
    """Multi-slice multi-modal reconstruction is fixed-seed stable.

    :see: :func:`ptyrodactyl.multislice.multi_slice_multi_modal`
    """
    experimental, _, _, potential_array, beam_array, _ = _tiny_recon_inputs()

    output = multi_slice_multi_modal(
        experimental,
        potential_array,
        beam_array,
        1.0,
        save_every=1,
        num_iterations=2,
        learning_rate=0.001,
    )

    assert output[3].shape == (4, 4, 2)
    assert output[4].shape == (4, 4, 2)
    _assert_scalar_equal(output[0][0, 0], 0.028663098749764054 + 0j)
    _assert_scalar_equal(
        output[1][0, 0],
        -1.4996974690074887 - 1.4814616332889121j,
    )
    _assert_scalar_equal(output[2][0, 0], 0.02000915293273863)
