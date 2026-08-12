"""Tests for route-neutral exact-carrier and shifted-free geometry evidence."""

from ptyrodactyl.galerkin.free_geometry import _INDEX_SAFE_LIMIT


def test_free_geometry_reenforces_the_acquisition_safe_limit() -> None:
    """Freeze the exact int64/binary64 acquisition arithmetic boundary."""
    assert _INDEX_SAFE_LIMIT == 1 << 52
