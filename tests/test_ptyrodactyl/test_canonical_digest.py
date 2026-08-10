"""Regression tests for the private canonical provenance digest owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np
from beartype.typing import Dict

import ptyrodactyl._canonical_digest as digest_module
from ptyrodactyl._canonical_digest import (
    _array_payload,
    _sha256,
    _stored_value_payload,
)
from ptyrodactyl.born import stability as stability_module


@dataclass(frozen=True)
class _OrderedStaticCarrier:
    """Expose declared-field order and exact static-float encodings."""

    zeta: float
    alpha: float

    @property
    def duplicate(self) -> float:
        """Return an undeclared property that must not enter the payload."""
        return self.zeta


def test_extracted_owner_preserves_stability_private_aliases() -> None:
    """Keep stability's established underscore names as identity aliases."""
    assert stability_module._host_array is digest_module._host_array
    assert stability_module._array_payload is digest_module._array_payload
    assert (
        stability_module._stored_value_payload
        is digest_module._stored_value_payload
    )
    assert stability_module._sha256 is digest_module._sha256


def test_exact_payload_bytes_and_digest_are_frozen() -> None:
    """Freeze the pre-extraction array bytes, tags, JSON, and digest."""
    array = jnp.asarray(
        [
            [1.0, -0.0],
            [float.fromhex("0x1.0000000000001p+0"), -2.5],
        ],
        dtype=jnp.float64,
    )
    payload = {
        "array": _array_payload(array),
        "static": _stored_value_payload((True, 7, -0.0, "LVT1", None)),
    }

    assert payload == {
        "array": {
            "dtype": "<f8",
            "shape": [2, 2],
            "bytes": (
                "000000000000f03f0000000000000080"
                "010000000000f03f00000000000004c0"
            ),
        },
        "static": {
            "tuple": [
                {"bool": True},
                {"int": "7"},
                {"float_hex": "-0x0.0p+0"},
                {"str": "LVT1"},
                {"none": True},
            ]
        },
    }
    assert _sha256(payload) == (
        "32fda6bfda7eca3676c1435fe3861f8b79394f7fa26d34888c7e57391329451e"
    )


def test_mapping_order_is_canonical_but_sequence_order_is_bound() -> None:
    """Sort mapping keys for hashing while retaining declared/tuple order."""
    forward: Dict[str, object] = {
        "z": {"int": "2"},
        "a": {"int": "1"},
    }
    reverse: Dict[str, object] = {
        "a": {"int": "1"},
        "z": {"int": "2"},
    }

    assert _sha256(forward) == _sha256(reverse)
    assert _sha256(forward) == (
        "dde11edb36cc2848fe438feacadae6c30fa34a5eadc97921a4af9ca428ea8a9e"
    )
    assert _stored_value_payload(("first", "second")) != (
        _stored_value_payload(("second", "first"))
    )

    stored = _stored_value_payload(_OrderedStaticCarrier(1.0, 2.0))
    assert isinstance(stored, dict)
    stored_mapping = cast(Dict[str, object], stored)
    fields = stored_mapping["fields"]
    assert isinstance(fields, dict)
    field_mapping = cast(Dict[str, object], fields)
    assert tuple(field_mapping) == ("zeta", "alpha")
    assert "duplicate" not in field_mapping


def test_array_dtype_and_shape_are_digest_identity() -> None:
    """Distinguish equal numeric values stored with different array schemas."""
    float32 = _array_payload(jnp.asarray([1.0, -2.0], dtype=jnp.float32))
    float64 = _array_payload(jnp.asarray([1.0, -2.0], dtype=jnp.float64))
    reshaped = _array_payload(jnp.asarray([[1.0, -2.0]], dtype=jnp.float64))

    assert float32["dtype"] == "<f4"
    assert float64["dtype"] == "<f8"
    assert float32["bytes"] != float64["bytes"]
    assert _sha256({"value": float32}) != _sha256({"value": float64})
    assert float64["bytes"] == reshaped["bytes"]
    assert float64["shape"] != reshaped["shape"]
    assert _sha256({"value": float64}) != _sha256({"value": reshaped})


def test_static_binary_floats_use_exact_hexadecimal_tags() -> None:
    """Keep signed zero and adjacent binary64 metadata values distinct."""
    positive_zero = _stored_value_payload(0.0)
    negative_zero = _stored_value_payload(-0.0)
    one = _stored_value_payload(1.0)
    next_one = _stored_value_payload(float.fromhex("0x1.0000000000001p+0"))

    assert positive_zero == {"float_hex": "0x0.0p+0"}
    assert negative_zero == {"float_hex": "-0x0.0p+0"}
    assert one == {"float_hex": "0x1.0000000000000p+0"}
    assert next_one == {"float_hex": "0x1.0000000000001p+0"}
    assert positive_zero != negative_zero
    assert one != next_one

    numpy_scalar = _stored_value_payload(np.float64(-0.0))
    assert isinstance(numpy_scalar, dict)
    assert "array" in numpy_scalar
    assert numpy_scalar != negative_zero
