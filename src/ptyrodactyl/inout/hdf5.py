"""Read and write the schema-v1 scalar-potential HDF5 archive.

Extended Summary
----------------
This module stores a validated
:class:`~ptyrodactyl.types.PotentialSlices` carrier in a versioned HDF5 schema
and reconstructs it through its canonical factory. The internal node codec
supports nested containers and static JSON metadata so later carrier
registrations can reuse the same format without unsafe dynamic imports.

Routine Listings
----------------
:class:`HDF5SchemaError`
    Report an incompatible or malformed ptyrodactyl HDF5 archive.
:func:`load_from_h5`
    Load one validated scalar-potential carrier from an HDF5 archive.
:func:`save_to_h5`
    Save one scalar-potential carrier to a versioned HDF5 archive.

Notes
-----
Only ``PotentialSlices`` is registered in schema version 1. Other carrier
types remain unsupported until the schema explicitly registers them.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from jaxtyping import Shaped
from numpy.typing import NDArray

from ..types import PotentialSlices, create_potential_slices

_ATTR_DICT_KEYS: str = "_dict_keys_json"
_ATTR_JSON_VALUE: str = "_json_value"
_ATTR_LENGTH: str = "_num_items"
_ATTR_NODE_KIND: str = "_node_kind"
_ATTR_PYTREE_TYPE: str = "_pytree_type"
_ATTR_SCHEMA_VERSION: str = "schema_version"
_ATTR_STATIC_METADATA: str = "_static_metadata_json"
_COMPRESSION_THRESHOLD_BYTES: int = 1 << 20
_JSON_COMPLEX: str = "complex"
_JSON_DICT: str = "dict"
_JSON_KIND: str = "__ptyrodactyl_json_kind__"
_JSON_TUPLE: str = "tuple"
_KIND_DICT: str = "dict"
_KIND_JSON: str = "json"
_KIND_LIST: str = "list"
_KIND_NONE: str = "none"
_KIND_PYTREE: str = "pytree"
_KIND_TUPLE: str = "tuple"
_SCHEMA_VERSION: int = 1


class HDF5SchemaError(ValueError):
    """Report an incompatible or malformed ptyrodactyl HDF5 archive.

    :see: :mod:`~.test_hdf5`
    """


@dataclass(frozen=True)
class _CarrierMeta:
    """Describe one explicitly supported carrier and its safe factory."""

    cls: type[Any]
    factory: Callable[..., Any]
    dynamic_fields: tuple[str, ...]
    static_fields: tuple[str, ...] = ()


_CARRIER_REGISTRY: dict[str, _CarrierMeta] = {
    "PotentialSlices": _CarrierMeta(
        cls=PotentialSlices,
        factory=create_potential_slices,
        dynamic_fields=("slices", "slice_thickness", "calib"),
    ),
}
_CARRIER_REGISTRY_BY_CLASS: dict[type[Any], _CarrierMeta] = {
    metadata.cls: metadata for metadata in _CARRIER_REGISTRY.values()
}


def _encode_json_value(value: Any) -> Any:  # noqa: PLR0911
    """Convert supported static metadata to an exact JSON representation."""
    if value is None or isinstance(value, bool | int | float | str):
        result: Any = value
        return result

    if isinstance(value, complex):
        result: Any = {
            _JSON_KIND: _JSON_COMPLEX,
            "real": value.real,
            "imag": value.imag,
        }
        return result

    if isinstance(value, np.generic):
        result: Any = _encode_json_value(value.item())
        return result

    if hasattr(value, "shape"):
        scalar_array: Shaped[NDArray, "..."] = np.asarray(value)
        if scalar_array.ndim == 0:
            result: Any = _encode_json_value(scalar_array.item())
            return result

    if isinstance(value, list):
        result: Any = [_encode_json_value(item) for item in value]
        return result

    if isinstance(value, tuple):
        result: Any = {
            _JSON_KIND: _JSON_TUPLE,
            "items": [_encode_json_value(item) for item in value],
        }
        return result

    if isinstance(value, dict):
        result: Any = {
            _JSON_KIND: _JSON_DICT,
            "items": [
                [_encode_json_value(key), _encode_json_value(item)]
                for key, item in value.items()
            ],
        }
        return result

    message: str = f"Unsupported static metadata type: {type(value).__name__}"
    raise TypeError(message)


def _decode_json_value(value: Any) -> Any:
    """Reconstruct a value produced by :func:`_encode_json_value`."""
    if isinstance(value, list):
        result: Any = [_decode_json_value(item) for item in value]
        return result

    if not isinstance(value, dict):
        result: Any = value
        return result

    marker: Any = value.get(_JSON_KIND)
    if marker == _JSON_COMPLEX:
        result: Any = complex(value["real"], value["imag"])
        return result
    if marker == _JSON_TUPLE:
        result: Any = tuple(
            _decode_json_value(item) for item in value["items"]
        )
        return result
    if marker == _JSON_DICT:
        result: Any = {
            _decode_json_value(key): _decode_json_value(item)
            for key, item in value["items"]
        }
        return result

    message: str = "Malformed tagged JSON value in HDF5 archive"
    raise HDF5SchemaError(message)


def _attribute_text(node: Any, name: str) -> str:
    """Return one required HDF5 attribute as text."""
    if name not in node.attrs:
        raise HDF5SchemaError(f"Missing required HDF5 attribute: {name}")
    value: Any = node.attrs[name]
    if isinstance(value, bytes):
        result: str = value.decode("utf-8")
        return result
    if isinstance(value, str):
        return value
    raise HDF5SchemaError(f"HDF5 attribute {name!r} must be text")


def _write_carrier_node(node: Any, carrier: Any) -> None:
    """Write one registered carrier into an existing HDF5 group."""
    metadata: _CarrierMeta | None = _CARRIER_REGISTRY_BY_CLASS.get(
        type(carrier)
    )
    if metadata is None:
        message: str = (
            "Unsupported carrier type for HDF5 schema version 1: "
            f"{type(carrier).__name__}"
        )
        raise TypeError(message)

    node.attrs[_ATTR_NODE_KIND] = _KIND_PYTREE
    node.attrs[_ATTR_PYTREE_TYPE] = type(carrier).__name__
    static_metadata: dict[str, Any] = {
        field_name: getattr(carrier, field_name)
        for field_name in metadata.static_fields
    }
    node.attrs[_ATTR_STATIC_METADATA] = json.dumps(
        _encode_json_value(static_metadata)
    )
    for field_name in metadata.dynamic_fields:
        _write_value(node, field_name, getattr(carrier, field_name))


def _write_value(parent: Any, name: str, value: Any) -> None:
    """Write one recursively supported value below an HDF5 group."""
    if type(value) in _CARRIER_REGISTRY_BY_CLASS:
        group: Any = parent.create_group(name)
        _write_carrier_node(group, value)
        return

    if value is None:
        group = parent.create_group(name)
        group.attrs[_ATTR_NODE_KIND] = _KIND_NONE
        return

    if isinstance(value, dict):
        group = parent.create_group(name)
        group.attrs[_ATTR_NODE_KIND] = _KIND_DICT
        group.attrs[_ATTR_LENGTH] = len(value)
        group.attrs[_ATTR_DICT_KEYS] = json.dumps(
            [_encode_json_value(key) for key in value]
        )
        for index, item in enumerate(value.values()):
            _write_value(group, f"item_{index}", item)
        return

    if isinstance(value, list | tuple):
        group = parent.create_group(name)
        group.attrs[_ATTR_NODE_KIND] = (
            _KIND_LIST if isinstance(value, list) else _KIND_TUPLE
        )
        group.attrs[_ATTR_LENGTH] = len(value)
        for index, item in enumerate(value):
            _write_value(group, f"item_{index}", item)
        return

    if isinstance(value, bool | int | float | complex | str):
        group = parent.create_group(name)
        group.attrs[_ATTR_NODE_KIND] = _KIND_JSON
        group.attrs[_ATTR_JSON_VALUE] = json.dumps(_encode_json_value(value))
        return

    array_value: Shaped[NDArray, "..."] = np.asarray(value)
    if array_value.dtype.kind in {"O", "S", "U", "V"}:
        message = (
            f"Unsupported array dtype for HDF5 storage: {array_value.dtype}"
        )
        raise TypeError(message)

    dataset_options: dict[str, Any] = {}
    if (
        array_value.ndim > 0
        and array_value.nbytes >= _COMPRESSION_THRESHOLD_BYTES
    ):
        dataset_options = {
            "compression": "gzip",
            "compression_opts": 4,
            "shuffle": True,
        }
    parent.create_dataset(name, data=array_value, **dataset_options)


def _read_carrier_node(node: Any) -> Any:
    """Read one registered carrier from an existing HDF5 group."""
    node_kind: str = _attribute_text(node, _ATTR_NODE_KIND)
    if node_kind != _KIND_PYTREE:
        raise HDF5SchemaError(f"Unknown HDF5 node kind: {node_kind}")

    type_name: str = _attribute_text(node, _ATTR_PYTREE_TYPE)
    metadata: _CarrierMeta | None = _CARRIER_REGISTRY.get(type_name)
    if metadata is None:
        raise HDF5SchemaError(f"Unknown PyTree type: {type_name}")

    expected_children: set[str] = set(metadata.dynamic_fields)
    actual_children: set[str] = set(node.keys())
    missing_children: set[str] = expected_children - actual_children
    if missing_children:
        missing: str = ", ".join(sorted(missing_children))
        raise HDF5SchemaError(f"Missing required carrier field(s): {missing}")
    unexpected_children: set[str] = actual_children - expected_children
    if unexpected_children:
        unexpected: str = ", ".join(sorted(unexpected_children))
        raise HDF5SchemaError(f"Unexpected carrier field(s): {unexpected}")

    static_json: str = _attribute_text(node, _ATTR_STATIC_METADATA)
    try:
        decoded_static: Any = _decode_json_value(json.loads(static_json))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise HDF5SchemaError(
            "Malformed static metadata in HDF5 archive"
        ) from error
    if not isinstance(decoded_static, dict):
        raise HDF5SchemaError("Static carrier metadata must be a mapping")
    if set(decoded_static) != set(metadata.static_fields):
        raise HDF5SchemaError("Static carrier metadata fields do not match")

    fields: dict[str, Any] = {
        field_name: _read_value(node[field_name])
        for field_name in metadata.dynamic_fields
    }
    fields.update(decoded_static)
    result: Any = metadata.factory(**fields)
    return result


def _read_length(node: Any) -> int:
    """Read and validate a container length attribute."""
    if _ATTR_LENGTH not in node.attrs:
        raise HDF5SchemaError(
            f"Missing required HDF5 attribute: {_ATTR_LENGTH}"
        )
    raw_length: Any = node.attrs[_ATTR_LENGTH]
    if isinstance(raw_length, bool | np.bool_) or not isinstance(
        raw_length, int | np.integer
    ):
        raise HDF5SchemaError("HDF5 container length must be an integer")
    length: int = int(raw_length)
    if length < 0:
        raise HDF5SchemaError("HDF5 container length cannot be negative")
    return length


def _read_value(node: Any) -> Any:  # noqa: PLR0911, PLR0912
    """Read one recursively supported value from an HDF5 node."""
    if isinstance(node, h5py.Dataset):
        result: Any = jnp.asarray(node[()])
        return result

    node_kind: str = _attribute_text(node, _ATTR_NODE_KIND)
    if node_kind == _KIND_PYTREE:
        result: Any = _read_carrier_node(node)
        return result
    if node_kind == _KIND_NONE:
        result: Any = None
        return result
    if node_kind == _KIND_JSON:
        encoded_value: str = _attribute_text(node, _ATTR_JSON_VALUE)
        try:
            result: Any = _decode_json_value(json.loads(encoded_value))
            return result
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise HDF5SchemaError(
                "Malformed JSON value in HDF5 archive"
            ) from error
    if node_kind in {_KIND_LIST, _KIND_TUPLE}:
        length: int = _read_length(node)
        items: list[Any] = []
        for index in range(length):
            item_name: str = f"item_{index}"
            if item_name not in node:
                raise HDF5SchemaError(
                    f"Missing HDF5 container item: {item_name}"
                )
            items.append(_read_value(node[item_name]))
        result: Any = items if node_kind == _KIND_LIST else tuple(items)
        return result
    if node_kind == _KIND_DICT:
        length = _read_length(node)
        encoded_keys: str = _attribute_text(node, _ATTR_DICT_KEYS)
        try:
            raw_keys: Any = json.loads(encoded_keys)
            keys: list[Any] = [_decode_json_value(item) for item in raw_keys]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise HDF5SchemaError(
                "Malformed dictionary keys in HDF5 archive"
            ) from error
        if len(keys) != length:
            raise HDF5SchemaError("HDF5 dictionary length does not match")
        values: list[Any] = []
        for index in range(length):
            item_name = f"item_{index}"
            if item_name not in node:
                raise HDF5SchemaError(
                    f"Missing HDF5 container item: {item_name}"
                )
            values.append(_read_value(node[item_name]))
        result: Any = dict(zip(keys, values, strict=True))
        return result

    raise HDF5SchemaError(f"Unknown HDF5 node kind: {node_kind}")


def _validate_schema_version(handle: Any) -> None:
    """Validate the root schema-version attribute."""
    if _ATTR_SCHEMA_VERSION not in handle.attrs:
        raise HDF5SchemaError(
            f"Missing required HDF5 attribute: {_ATTR_SCHEMA_VERSION}"
        )
    raw_version: Any = handle.attrs[_ATTR_SCHEMA_VERSION]
    if isinstance(raw_version, bool | np.bool_) or not isinstance(
        raw_version, int | np.integer
    ):
        raise HDF5SchemaError("HDF5 schema_version must be an integer")
    version: int = int(raw_version)
    if version != _SCHEMA_VERSION:
        raise HDF5SchemaError(
            "Unsupported HDF5 schema_version "
            f"{version}; supported version is {_SCHEMA_VERSION}"
        )


def save_to_h5(carrier: PotentialSlices, path: str | Path) -> None:
    """Save one scalar-potential carrier to a versioned HDF5 archive.

    :see: :func:`~.test_large_potential_uses_lossless_gzip_compression`

    Parameters
    ----------
    carrier : PotentialSlices
        Canonical scalar-potential slice carrier to archive.
    path : str | Path
        Destination HDF5 file. An existing file is replaced.

    Raises
    ------
    TypeError
        If ``carrier`` is not registered in schema version 1 or contains an
        unsupported value.
    OSError
        If the destination cannot be created or written.
    """
    if type(carrier) not in _CARRIER_REGISTRY_BY_CLASS:
        message: str = (
            "Unsupported carrier type for HDF5 schema version 1: "
            f"{type(carrier).__name__}"
        )
        raise TypeError(message)

    file_path: Path = Path(path)
    with h5py.File(file_path, "w") as handle:
        handle.attrs[_ATTR_SCHEMA_VERSION] = _SCHEMA_VERSION
        _write_carrier_node(handle, carrier)


def load_from_h5(path: str | Path) -> PotentialSlices:
    """Load one validated scalar-potential carrier from an HDF5 archive.

    :see: :mod:`~.test_hdf5`

    Parameters
    ----------
    path : str | Path
        Source HDF5 archive.

    Returns
    -------
    carrier : PotentialSlices
        Scalar-potential carrier reconstructed through
        :func:`~ptyrodactyl.types.create_potential_slices`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    HDF5SchemaError
        If the file is not valid HDF5 or has a missing, malformed, or
        unsupported ptyrodactyl schema.
    ValueError
        If stored carrier values fail canonical carrier validation.
    """
    file_path: Path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    try:
        with h5py.File(file_path, "r") as handle:
            _validate_schema_version(handle)
            carrier: Any = _read_carrier_node(handle)
    except HDF5SchemaError:
        raise
    except OSError as error:
        raise HDF5SchemaError(
            f"Failed to open HDF5 archive {file_path}: {error}"
        ) from error

    if not isinstance(carrier, PotentialSlices):
        raise HDF5SchemaError("Archive did not contain PotentialSlices")
    return carrier


__all__: list[str] = [
    "HDF5SchemaError",
    "load_from_h5",
    "save_to_h5",
]
