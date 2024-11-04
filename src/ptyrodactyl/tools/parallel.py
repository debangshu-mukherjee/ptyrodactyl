from typing import Sequence, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array


def shard_array(
    input_array: Array,
    shard_axes: Union[int, Sequence[int]],
    devices: Sequence[jax.Device] = None,
) -> Array:
    """
    Shards an array across specified axes and devices.

    Args:
    - `input_array` (Array):
        The input input_array to be sharded.
    - `shard_axes` (Union[int, Sequence[int]]):
        The axis or axes to shard along.
        Use -1 or sequence of -1s to not shard along any axis.
    - `devices` (Sequence[jax.Device], optional):
        The devices to shard across.
        If None, uses all available devices.

    Returns:
    - `sharded_array` (Array):
        The sharded array.
    """
    if devices is None:
        devices = jax.devices()

    # Ensure shard_axes is a sequence
    if isinstance(shard_axes, int):
        shard_axes = [shard_axes]

    # Create a mesh
    mesh = Mesh(devices, ("devices",))

    # Create PartitionSpec
    pspec = [None] * input_array.ndim
    for ax in shard_axes:
        if ax != -1 and ax < input_array.ndim:
            pspec[ax] = "devices"
    pspec = PartitionSpec(*pspec)

    # Create NamedSharding
    sharding = NamedSharding(mesh, pspec)

    # Shard the input_array
    with mesh:
        sharded_array = jax.device_put(input_array, sharding)

    return sharded_array
