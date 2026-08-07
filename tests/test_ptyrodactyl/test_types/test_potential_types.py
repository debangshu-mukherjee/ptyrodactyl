"""Tests for :mod:`ptyrodactyl.types.potential_types`."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from ptyrodactyl.types import Potential3D, create_potential_3d

_PROVENANCE = "a" * 64
_NORMALIZATION = "continuous Fourier coefficients; JAX inverse DFT"


def _create(volume: jax.Array) -> Potential3D:
    """Create the standard small carrier used by these tests."""
    return create_potential_3d(
        volume,
        voxel_size=(0.5, 0.25, 0.2),
        box_size=(2.0, 0.75, 0.4),
        origin=(-1.0, -0.25, 0.1),
        producer="test producer",
        provenance_hash=_PROVENANCE,
        coefficient_normalization=_NORMALIZATION,
        band_limit=0.75,
    )


def test_potential_3d_preserves_voltage_geometry_and_one_dynamic_leaf() -> (
    None
):
    """The carrier makes units, axes, reference, and PyTree seam explicit."""
    volume = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4)
    potential = _create(volume)

    assert potential.volume.shape == (2, 3, 4)
    assert potential.volume.dtype == jnp.float64
    assert potential.voxel_size == (0.5, 0.25, 0.2)
    assert potential.box_size == (2.0, 0.75, 0.4)
    assert potential.origin == (-1.0, -0.25, 0.1)
    assert potential.units == "V"
    assert potential.reference_value == 0.0
    assert "vacuum zero" in potential.reference_semantics
    assert potential.provenance_hash == _PROVENANCE
    leaves = jax.tree_util.tree_leaves(potential)
    assert len(leaves) == 1
    assert leaves[0] is potential.volume


def test_potential_3d_factory_is_jittable_and_volume_differentiable() -> None:
    """Validation leaves the voltage samples differentiable end to end."""

    @jax.jit
    def loss(volume: jax.Array) -> jax.Array:
        potential = _create(volume)
        return jnp.sum(jnp.square(potential.volume))

    volume = jnp.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
    gradient = jax.grad(loss)(volume)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, 2.0 * volume)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"units": "eV"}, "units must be exactly 'V'"),
        ({"reference_semantics": "unspecified"}, "physical reference"),
        ({"reference_semantics": "unknown reference"}, "physical reference"),
        ({"reference_semantics": "unknown-reference"}, "physical reference"),
        ({"reference_semantics": "not specified"}, "physical reference"),
        ({"reference_semantics": "to-be-determined"}, "physical reference"),
        ({"reference_semantics": "TBD"}, "physical reference"),
        ({"provenance_hash": "not-a-digest"}, "SHA-256"),
        ({"band_limit": 3.0}, "Nyquist"),
        ({"box_size": (2.1, 0.75, 0.4)}, "box_size must equal"),
    ],
)
def test_potential_3d_rejects_ambiguous_metadata(
    override: dict[str, object],
    message: str,
) -> None:
    """Quantitative fields cannot carry guessed units or discretization."""
    arguments: dict[str, object] = {
        "volume": jnp.zeros((2, 3, 4)),
        "voxel_size": (0.5, 0.25, 0.2),
        "box_size": (2.0, 0.75, 0.4),
        "origin": (0.0, 0.0, 0.0),
        "producer": "test producer",
        "provenance_hash": _PROVENANCE,
        "coefficient_normalization": _NORMALIZATION,
        "band_limit": 0.75,
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        create_potential_3d(**arguments)


def test_potential_3d_rejects_nonfinite_voltage_samples() -> None:
    """The traced factory check rejects a non-finite physical field."""
    volume = jnp.zeros((2, 3, 4)).at[0, 0, 0].set(jnp.nan)
    with pytest.raises(Exception, match="volume contains non-finite values"):
        potential = _create(volume)
        jax.block_until_ready(potential.volume)


@pytest.mark.parametrize(
    "override",
    [
        {"voxel_size": (True, 0.25, 0.2)},
        {"origin": (False, -0.25, 0.1)},
        {"reference_value": True},
        {"band_limit": True},
    ],
)
def test_potential_3d_rejects_boolean_metadata(
    override: dict[str, object],
) -> None:
    """Boolean metadata cannot masquerade as zero or one physical units."""
    arguments: dict[str, object] = {
        "volume": jnp.zeros((2, 3, 4)),
        "voxel_size": (0.5, 0.25, 0.2),
        "box_size": (2.0, 0.75, 0.4),
        "origin": (-1.0, -0.25, 0.1),
        "producer": "test producer",
        "provenance_hash": _PROVENANCE,
        "coefficient_normalization": _NORMALIZATION,
        "band_limit": 0.75,
    }
    arguments.update(override)

    with pytest.raises((ValueError, TypeCheckError)):
        create_potential_3d(**arguments)
