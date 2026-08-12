r"""Provide canonical stored-value payloads and provenance digests.

Extended Summary
----------------
This private host-side module owns the dependency-neutral serialization used
to bind scientific carriers to exact provenance checksums. Arrays are bound
by dtype, shape, and C-order bytes; dataclass fields are walked in declaration
order without evaluating convenience properties; and static binary floats
use hexadecimal notation. Canonical JSON encoding then fixes the SHA-256
digest independently of mapping insertion order.

Routine Listings
----------------
:func:`array_payload`
    Build a canonical dtype-, shape-, and byte-bound payload.
:func:`host_array`
    Transfer one JAX array to a read-only host NumPy value.
:func:`sha256`
    Hash one canonical JSON payload as a provenance checksum.
:func:`stored_value_payload`
    Serialize one declared carrier value without properties.

Notes
-----
These checksums establish stored-value identity only. Scientific checker
modules remain responsible for reconstructing and validating carrier
semantics before trusting any digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict
from jaxtyping import Shaped
from numpy.typing import NDArray

# Python permits a process-wide decimal integer conversion limit as low as
# 640 digits.  Every integer with at most 2,048 bits has at most 617 decimal
# digits, so the historical decimal payload remains safe below this fixed,
# version-independent boundary.  Larger integers use power-of-two conversion,
# which is exempt from that interpreter limit.
_MAXIMUM_DECIMAL_INTEGER_BITS: int = 2048


def host_array(value: jax.Array) -> Shaped[NDArray, "..."]:
    """Transfer one JAX array to a read-only host NumPy value.

    Parameters
    ----------
    value : jax.Array
        Device array to transfer.

    Returns
    -------
    array : Shaped[NDArray, "..."]
        Host NumPy view or copy with the same dtype, shape, and values.

    Notes
    -----
    Device transfer establishes a host-side checker boundary.
    """
    array: Shaped[NDArray, "..."] = np.asarray(jax.device_get(value))
    return array


def array_payload(value: jax.Array) -> Dict[str, object]:
    """Build a canonical dtype-, shape-, and byte-bound payload.

    Parameters
    ----------
    value : jax.Array
        Array whose exact stored representation is bound.

    Returns
    -------
    payload : Dict[str, object]
        Canonical dtype string, shape list, and C-order byte encoding.

    Notes
    -----
    Contiguous C-order bytes make layout-independent array provenance stable.
    """
    array: Shaped[NDArray, "..."] = host_array(value)
    contiguous: Shaped[NDArray, "..."] = np.ascontiguousarray(array)
    payload: Dict[str, object] = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes": contiguous.tobytes(order="C").hex(),
    }
    return payload


def stored_value_payload(value: object) -> object:  # noqa: PLR0911
    """Serialize one declared carrier value without properties.

    Parameters
    ----------
    value : object
        Stored array, Equinox/dataclass module, enum, tuple, or primitive.

    Returns
    -------
    payload : object
        Canonically tagged JSON-serializable representation.

    Raises
    ------
    TypeError
        If a declared field has no admitted canonical representation.

    Notes
    -----
    Walking :func:`dataclasses.fields` binds every declared dynamic and static
    field while deliberately excluding read-only convenience properties.
    Exact array bytes include dtype and shape. Static floating-point metadata
    uses hexadecimal notation so signed zero and every binary value remain
    distinct. Large static integers use a signed hexadecimal tag so digest
    construction is independent of Python's process-wide decimal conversion
    limit; ordinary small-integer payloads retain their historical tag.
    """
    if isinstance(value, jax.Array | np.ndarray | np.generic):
        payload: object = {"array": array_payload(jnp.asarray(value))}
        return payload
    if isinstance(value, Enum):
        payload: object = {
            "enum_type": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "value": stored_value_payload(value.value),
        }
        return payload
    if is_dataclass(value) and not isinstance(value, type):
        payload: object = {
            "module_type": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "fields": {
                field.name: stored_value_payload(getattr(value, field.name))
                for field in fields(value)
            },
        }
        return payload
    if isinstance(value, tuple):
        payload: object = {
            "tuple": [stored_value_payload(entry) for entry in value]
        }
        return payload
    if isinstance(value, bool):
        payload: object = {"bool": value}
        return payload
    if isinstance(value, int):
        if value.bit_length() <= _MAXIMUM_DECIMAL_INTEGER_BITS:
            payload = {"int": str(value)}
        else:
            sign = "-" if value < 0 else ""
            payload = {"int_hex": f"{sign}{abs(value):x}"}
        return payload
    if isinstance(value, float):
        payload: object = {"float_hex": value.hex()}
        return payload
    if isinstance(value, str):
        payload: object = {"str": value}
        return payload
    if value is None:
        payload: object = {"none": True}
        return payload
    raise TypeError(
        "unsupported declared provenance field type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def sha256(payload: Dict[str, object]) -> str:
    """Hash one canonical JSON payload as a provenance checksum.

    Parameters
    ----------
    payload : Dict[str, object]
        JSON-serializable canonical provenance payload.

    Returns
    -------
    digest : str
        Lowercase hexadecimal SHA-256 checksum.

    Notes
    -----
    Sorted keys, compact separators, and ASCII encoding fix serialization.
    """
    encoded: bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest: str = hashlib.sha256(encoded).hexdigest()
    return digest


__all__: list[str] = [
    "array_payload",
    "host_array",
    "sha256",
    "stored_value_payload",
]
