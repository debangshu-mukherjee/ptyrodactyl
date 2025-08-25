"""Parallel processing utilities for distributed computing in ptychography.

This module provides utilities for sharding arrays across multiple devices
for parallel processing and distributed computing in ptychography workflows.
All functions are JAX-compatible and support automatic differentiation.

Functions
---------
shard_array
    Shards an array across specified axes and devices for parallel processing

Notes
-----
This module is designed for distributed computing scenarios where large
arrays need to be processed across multiple devices. The sharding utilities
work with JAX's device mesh system and can be used with various JAX
transformations including jit, grad, and vmap.
"""

from collections.abc import Sequence

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array


def shard_array(
    input_array: Array,
    shard_axes: int | Sequence[int],
    devices: Sequence[jax.Device] = None,
) -> Array:
    """Shards an array across specified axes and devices.

    This function distributes an array across multiple devices for parallel
    processing. It creates a mesh of devices and applies appropriate
    partitioning based on the specified axes.

    Parameters
    ----------
    input_array : Array
        The input array to be sharded
    shard_axes : int | Sequence[int]
        The axis or axes to shard along.
        Use -1 or sequence of -1s to not shard along any axis
    devices : Sequence[jax.Device], optional
        The devices to shard across.
        If None, uses all available devices

    Returns
    -------
    Array
        The sharded array distributed across the specified devices

    Notes
    -----
    Algorithm:
    - Get all available devices if none specified
    - Ensure shard_axes is a sequence (convert single int to list)
    - Create a mesh with the specified devices
    - Create PartitionSpec with None for non-sharded axes and "devices" for sharded axes
    - Create NamedSharding with the mesh and partition spec
    - Place the array on devices using the sharding configuration
    - Return the sharded array
    """
    if devices is None:
        devices = jax.devices()
    if isinstance(shard_axes, int):
        shard_axes = [shard_axes]
    mesh = Mesh(devices, ("devices",))
    pspec = [None] * input_array.ndim
    for ax in shard_axes:
        if ax != -1 and ax < input_array.ndim:
            pspec[ax] = "devices"
    pspec = PartitionSpec(*pspec)
    sharding = NamedSharding(mesh, pspec)
    with mesh:
        return jax.device_put(input_array, sharding)
