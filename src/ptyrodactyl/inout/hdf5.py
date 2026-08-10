"""Read and write the schema-v1 scalar-potential HDF5 archive.

Extended Summary
----------------
This module stores validated scalar-potential carriers in a versioned HDF5
schema and reconstructs each carrier through its canonical factory. The
internal node codec supports nested containers and static JSON metadata so
later carrier registrations can reuse the same format without unsafe dynamic
imports.

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
``Potential3D`` and ``PotentialSlices`` are registered in schema version 1.
Other carrier types remain unsupported until the schema explicitly registers
them.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from beartype.typing import Any, Dict, Tuple
from jaxtyping import Shaped
from numpy.typing import NDArray

from ptyrodactyl.types import (
    Potential3D,
    PotentialSlices,
    create_potential_3d,
    create_potential_slices,
)

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
    dynamic_fields: Tuple[str, ...]
    static_fields: Tuple[str, ...] = ()


_CARRIER_REGISTRY: Dict[str, _CarrierMeta] = {
    "Potential3D": _CarrierMeta(
        cls=Potential3D,
        factory=create_potential_3d,
        dynamic_fields=("volume",),
        static_fields=(
            "voxel_size",
            "box_size",
            "origin",
            "units",
            "reference_value",
            "reference_semantics",
            "boundary",
            "producer",
            "provenance_hash",
            "coefficient_normalization",
            "band_limit",
        ),
    ),
    "PotentialSlices": _CarrierMeta(
        cls=PotentialSlices,
        factory=create_potential_slices,
        dynamic_fields=("slices", "slice_thickness", "calib"),
    ),
}
_CARRIER_REGISTRY_BY_CLASS: Dict[type[Any], _CarrierMeta] = {
    metadata.cls: metadata for metadata in _CARRIER_REGISTRY.values()
}


def _encode_json_value(value: Any) -> Any:  # noqa: PLR0911
    """PRIVATE: Convert static metadata to an exact JSON representation.

    Parameters
    ----------
    value : Any
        Supported scalar, container, mapping, or scalar array value.

    Returns
    -------
    result : Any
        JSON-encodable value that preserves tuples, mappings, and complex
        scalars through explicit tags.

    Raises
    ------
    TypeError
        If ``value`` has no registered static-metadata representation.

    Notes
    -----
    Mapping keys use the same tagged representation as mapping values.
    """
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
    """PRIVATE: Reconstruct one tagged static-metadata value.

    Parameters
    ----------
    value : Any
        JSON-decoded value produced by :func:`_encode_json_value`.

    Returns
    -------
    result : Any
        Reconstructed scalar, tuple, list, or mapping value.

    Raises
    ------
    HDF5SchemaError
        If a tagged mapping uses an unknown metadata representation.
    """
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
    """PRIVATE: Return one required HDF5 attribute as text.

    Parameters
    ----------
    node : Any
        HDF5 node that owns the required attribute.
    name : str
        Required attribute name.

    Returns
    -------
    result : str
        UTF-8 attribute text.

    Raises
    ------
    HDF5SchemaError
        If the attribute is missing or is not stored as text.
    """
    if name not in node.attrs:
        raise HDF5SchemaError(f"Missing required HDF5 attribute: {name}")
    value: Any = node.attrs[name]
    if isinstance(value, bytes):
        result: str = value.decode("utf-8")
        return result
    if isinstance(value, str):
        result: str = value
        return result
    raise HDF5SchemaError(f"HDF5 attribute {name!r} must be text")


def _write_carrier_node(node: Any, carrier: Any) -> None:
    """PRIVATE: Write one registered carrier into an HDF5 group.

    Parameters
    ----------
    node : Any
        Existing HDF5 group that receives the carrier fields and metadata.
    carrier : Any
        Carrier instance registered in schema version 1.

    Raises
    ------
    TypeError
        If the carrier type is not registered or its static metadata contains
        an unsupported value.
    """
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
    static_metadata: Dict[str, Any] = {
        field_name: getattr(carrier, field_name)
        for field_name in metadata.static_fields
    }
    node.attrs[_ATTR_STATIC_METADATA] = json.dumps(
        _encode_json_value(static_metadata)
    )
    for field_name in metadata.dynamic_fields:
        _write_value(node, field_name, getattr(carrier, field_name))


def _write_value(parent: Any, name: str, value: Any) -> None:
    """PRIVATE: Write one recursively supported value below an HDF5 group.

    Parameters
    ----------
    parent : Any
        HDF5 group that receives the new child node.
    name : str
        Name of the child group or dataset.
    value : Any
        Registered carrier, supported container, scalar, or numeric array.

    Raises
    ------
    TypeError
        If ``value`` contains unsupported metadata or has a non-numeric array
        dtype.

    Notes
    -----
    Numeric arrays at least one mebibyte use lossless gzip compression and
    byte shuffling.
    """
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

    dataset_options: Dict[str, Any] = {}
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
    """PRIVATE: Read one registered carrier from an HDF5 group.

    Parameters
    ----------
    node : Any
        HDF5 group containing one encoded carrier.

    Returns
    -------
    result : Any
        Carrier reconstructed through its registered canonical factory.

    Raises
    ------
    HDF5SchemaError
        If the node kind, carrier type, fields, or static metadata is invalid.
    """
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

    fields: Dict[str, Any] = {
        field_name: _read_value(node[field_name])
        for field_name in metadata.dynamic_fields
    }
    fields.update(decoded_static)
    result: Any = metadata.factory(**fields)
    return result


def _read_length(node: Any) -> int:
    """PRIVATE: Read and validate one container length attribute.

    Parameters
    ----------
    node : Any
        HDF5 container node with a stored item count.

    Returns
    -------
    length : int
        Validated non-negative container length.

    Raises
    ------
    HDF5SchemaError
        If the length is missing, non-integral, boolean, or negative.
    """
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
    """PRIVATE: Read one recursively supported value from an HDF5 node.

    Parameters
    ----------
    node : Any
        HDF5 dataset or tagged group to decode.

    Returns
    -------
    result : Any
        Reconstructed JAX array, carrier, container, scalar, or ``None``.

    Raises
    ------
    HDF5SchemaError
        If the node kind or encoded container payload is malformed.
    """
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
    """PRIVATE: Validate the root schema-version attribute.

    Parameters
    ----------
    handle : Any
        Open HDF5 file handle whose root metadata is validated.

    Raises
    ------
    HDF5SchemaError
        If ``schema_version`` is missing, non-integral, boolean, or
        unsupported.
    """
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


def save_to_h5(
    carrier: Potential3D | PotentialSlices,
    path: str | Path,
) -> None:
    """Save one scalar-potential carrier to a versioned HDF5 archive.

    :see: :func:`~.test_large_potential_uses_lossless_gzip_compression`

    Parameters
    ----------
    carrier : Potential3D | PotentialSlices
        Canonical scalar-potential carrier to archive.
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


def load_from_h5(path: str | Path) -> Potential3D | PotentialSlices:
    """Load one validated scalar-potential carrier from an HDF5 archive.

    :see: :mod:`~.test_hdf5`

    Parameters
    ----------
    path : str | Path
        Source HDF5 archive.

    Returns
    -------
    result : Potential3D | PotentialSlices
        Scalar-potential carrier reconstructed through its canonical factory.

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

    if not isinstance(carrier, Potential3D | PotentialSlices):
        raise HDF5SchemaError(
            "Archive did not contain a registered scalar-potential carrier"
        )
    result: Potential3D | PotentialSlices = carrier
    return result


__all__: list[str] = [
    "HDF5SchemaError",
    "load_from_h5",
    "save_to_h5",
]
