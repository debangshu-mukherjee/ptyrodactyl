"""Tests for plotting and visualization helpers."""
# ruff: noqa: E402

import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.plots import (
    contrast_stretch,
    create_phosphor_colormap,
)


class TestContrastStretch(chex.TestCase):
    """Test suite for percentile contrast stretching."""

    def test_contrast_stretch_2d_static_rank(self) -> None:
        """2D input returns a 2D image with fixed values."""
        image = jnp.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=jnp.float64
        )
        expected = np.array(
            [[0.0, 0.125, 0.375], [0.625, 0.875, 1.0]],
            dtype=np.float64,
        )

        result = contrast_stretch(image, 10.0, 90.0)

        assert result.shape == image.shape
        assert np.array_equal(np.asarray(result), expected)

    def test_contrast_stretch_3d_static_rank(self) -> None:
        """3D input returns a 3D stack with fixed values per image."""
        image = jnp.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=jnp.float64
        )
        stack = jnp.stack([image, image + 10.0], axis=0)
        expected_image = np.array(
            [[0.0, 0.125, 0.375], [0.625, 0.875, 1.0]],
            dtype=np.float64,
        )
        expected = np.stack([expected_image, expected_image], axis=0)

        result = contrast_stretch(stack, 10.0, 90.0)

        assert result.shape == stack.shape
        assert np.array_equal(np.asarray(result), expected)


class TestCreatePhosphorColormap(chex.TestCase):
    """Test suite for phosphor colormap construction."""

    def test_create_phosphor_colormap_returns_finite_rgba(self) -> None:
        """Default colormap returns finite 256-sample RGBA values."""
        cmap = create_phosphor_colormap()
        samples = cmap(np.linspace(0.0, 1.0, cmap.N))

        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.name == "phosphor"
        assert cmap.N == 256
        assert samples.shape == (256, 4)
        assert np.isfinite(samples).all()


if __name__ == "__main__":
    unittest.main()
