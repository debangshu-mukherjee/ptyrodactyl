"""Test lossless HDF5 potential ingest and emission.

:see: :class:`ptyrodactyl.inout.HDF5SchemaError`
:see: :func:`ptyrodactyl.inout.load_from_h5`
"""

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import EquinoxRuntimeError

from ptyrodactyl.inout import HDF5SchemaError, load_from_h5, save_to_h5
from ptyrodactyl.types import PotentialSlices, create_potential_slices


def _sample_potential() -> PotentialSlices:
    """Return a small, nontrivial canonical scalar-potential carrier."""
    return create_potential_slices(
        slices=jnp.array(
            [
                [[-3.25, 0.0], [1.5, 8.125], [2.0**-40, -(2.0**40)]],
                [[7.75, -1.25], [3.0, 4.5], [-6.0, 9.0]],
            ],
            dtype=jnp.float64,
        ),
        slice_thickness=jnp.asarray(1.75, dtype=jnp.float64),
        calib=jnp.asarray(0.125, dtype=jnp.float64),
    )


def _save_sample(path: Path) -> PotentialSlices:
    """Save and return the canonical test potential."""
    potential = _sample_potential()
    save_to_h5(potential, path)
    return potential


def test_scalar_potential_round_trip_is_bit_exact_and_versioned(
    tmp_path: Path,
) -> None:
    """The scalar-potential carrier uses the canonical lossless schema."""
    path = tmp_path / "potential.h5"
    expected = _save_sample(path)

    loaded = load_from_h5(path)

    assert isinstance(loaded, PotentialSlices)
    for field_name in ("slices", "slice_thickness", "calib"):
        expected_value = np.asarray(getattr(expected, field_name))
        loaded_value = np.asarray(getattr(loaded, field_name))
        assert loaded_value.dtype == expected_value.dtype == np.float64
        np.testing.assert_array_equal(loaded_value, expected_value)

    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["schema_version"]) == 1
        assert handle.attrs["_node_kind"] == "pytree"
        assert handle.attrs["_pytree_type"] == "PotentialSlices"
        assert set(handle) == {"calib", "slice_thickness", "slices"}
        for field_name in handle:
            assert isinstance(handle[field_name], h5py.Dataset)


def test_large_potential_uses_lossless_gzip_compression(
    tmp_path: Path,
) -> None:
    """Large scalar-potential arrays are chunked and gzip-compressed.

    :see: :func:`ptyrodactyl.inout.save_to_h5`
    """
    path = tmp_path / "large-potential.h5"
    values = jnp.arange(256 * 256 * 2, dtype=jnp.float64).reshape(256, 256, 2)
    expected = create_potential_slices(values, 2.0, 0.25)

    save_to_h5(expected, path)

    with h5py.File(path, "r") as handle:
        slices = handle["slices"]
        assert isinstance(slices, h5py.Dataset)
        assert slices.compression == "gzip"
        assert slices.compression_opts == 4
        assert slices.chunks is not None
        assert slices.dtype == np.dtype(np.float64)
        assert handle["slice_thickness"].compression is None
        assert handle["calib"].compression is None

    loaded = load_from_h5(path)
    np.testing.assert_array_equal(
        np.asarray(loaded.slices), np.asarray(expected.slices)
    )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("schema_version", "not-an-integer", "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("_pytree_type", "UnknownPotential", "Unknown.*type"),
        ("_node_kind", "unknown", "Unknown.*kind"),
    ],
)
def test_unknown_or_future_schema_is_rejected(
    tmp_path: Path,
    attribute: str,
    value: object,
    message: str,
) -> None:
    """Unknown schema versions, carrier types, and node kinds fail closed."""
    path = tmp_path / "mutated-schema.h5"
    _save_sample(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs[attribute] = value

    with pytest.raises(HDF5SchemaError, match=message):
        load_from_h5(path)


def test_missing_schema_version_is_rejected(tmp_path: Path) -> None:
    """An HDF5 file without the canonical version marker is malformed."""
    path = tmp_path / "missing-version.h5"
    _save_sample(path)
    with h5py.File(path, "r+") as handle:
        del handle.attrs["schema_version"]

    with pytest.raises(HDF5SchemaError, match="schema_version"):
        load_from_h5(path)


def test_reload_runs_potential_factory_validation(tmp_path: Path) -> None:
    """Loading reconstructs through create_potential_slices validation."""
    path = tmp_path / "invalid-potential.h5"
    _save_sample(path)
    with h5py.File(path, "r+") as handle:
        handle["slice_thickness"][...] = -1.0

    with pytest.raises(EquinoxRuntimeError, match="slice_thickness"):
        load_from_h5(path)


def test_loaded_potential_preserves_gradient_behavior(tmp_path: Path) -> None:
    """Gradients through loaded dynamic leaves match the original carrier."""
    path = tmp_path / "gradient-potential.h5"
    original = _save_sample(path)
    loaded = load_from_h5(path)

    def objective(
        slices: jax.Array,
        slice_thickness: jax.Array,
        calib: jax.Array,
    ) -> jax.Array:
        return slice_thickness * jnp.sum(jnp.sin(slices * calib))

    gradient = jax.grad(objective, argnums=(0, 1, 2))
    expected_gradient = gradient(
        original.slices,
        original.slice_thickness,
        original.calib,
    )
    loaded_gradient = gradient(
        loaded.slices,
        loaded.slice_thickness,
        loaded.calib,
    )

    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        loaded_gradient,
        expected_gradient,
    )
