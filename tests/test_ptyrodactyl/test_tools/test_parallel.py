"""Test :mod:`ptyrodactyl.tools.parallel`.

:see: :func:`ptyrodactyl.tools.shard_array`
"""

import ptyrodactyl.tools as tools
from ptyrodactyl.tools import parallel


def test_shard_array_resolves_through_public_package() -> None:
    """Prove the sharding helper has one canonical public package export.

    :see: :func:`ptyrodactyl.tools.shard_array`
    """
    assert tools.shard_array is parallel.shard_array
