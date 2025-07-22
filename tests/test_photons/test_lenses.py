import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from ptyrodactyl.photons.lenses import (LensParams, double_concave_lens,
                                        double_convex_lens, lens_focal_length,
                                        lens_thickness_profile, meniscus_lens,
                                        plano_concave_lens, plano_convex_lens)

jax.config.update("jax_enable_x64", True)


class TestLensParams(chex.TestCase):
    def test_lens_params_creation(self):
        """Test creation of LensParams with valid values."""
        params = LensParams(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )

        assert isinstance(params.focal_length, jnp.ndarray)
        assert isinstance(params.diameter, jnp.ndarray)
        assert isinstance(params.n, jnp.ndarray)
        assert isinstance(params.center_thickness, jnp.ndarray)
        assert isinstance(params.R1, jnp.ndarray)
        assert isinstance(params.R2, jnp.ndarray)

    def test_lens_params_tree_flatten(self):
        """Test PyTree flattening of LensParams."""
        params = LensParams(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )

        children, aux_data = params.tree_flatten()

        assert len(children) == 6
        assert aux_data is None


class TestLensThicknessProfile(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"shape": (40, 40)},
        {"shape": (60, 20)},
        {"shape": (30, 90)},
    )
    def test_thickness_profile_shape(self, shape: Tuple[int, int]):
        """Test that thickness profile returns correct shape."""
        x = jnp.asarray(jnp.linspace(-0.1, 0.1, shape[1]))
        y = jnp.asarray(jnp.linspace(-0.1, 0.1, shape[0]))
        X, Y = jnp.meshgrid(x, y)
        r = jnp.asarray(jnp.sqrt(X**2 + Y**2))

        var_lens_thickness = self.variant(lens_thickness_profile)
        thickness = var_lens_thickness(
            r=r,
            R1=jnp.asarray(0.2),
            R2=jnp.asarray(0.2),
            center_thickness=jnp.asarray(0.01),
            diameter=jnp.asarray(0.1),
        )

        chex.assert_shape(thickness, shape)

    @chex.all_variants()
    def test_thickness_profile_center(self):
        """Test that thickness at center matches center_thickness."""
        shape = (40, 40)
        x = jnp.asarray(jnp.linspace(-0.1, 0.1, shape[1]))
        y = jnp.asarray(jnp.linspace(-0.1, 0.1, shape[0]))
        X, Y = jnp.meshgrid(x, y)
        r = jnp.asarray(jnp.sqrt(X**2 + Y**2))

        center_thickness = jnp.asarray(0.01)
        var_lens_thickness = self.variant(lens_thickness_profile)
        thickness = var_lens_thickness(
            r=r,
            R1=jnp.asarray(0.2),
            R2=jnp.asarray(0.2),
            center_thickness=center_thickness,
            diameter=jnp.asarray(0.1),
        )

        center_idx = (shape[0] // 2, shape[1] // 2)
        chex.assert_trees_all_close(thickness[center_idx], center_thickness, atol=1e-6)


class TestLensFocalLength(chex.TestCase):
    @chex.all_variants()
    @parameterized.parameters(
        {"n": 1.5, "R1": 0.2, "R2": 0.2, "expected": 0.2},  # Symmetric lens
        {"n": 1.5, "R1": 0.1, "R2": 0.3, "expected": 0.15},  # Asymmetric lens
        {"n": 2.0, "R1": 0.2, "R2": 0.2, "expected": 0.1},  # Different n
    )
    def test_focal_length_calculation(self, n, R1, R2, expected):
        """Test focal length calculation for various lens parameters."""
        var_lens_focal_length = self.variant(lens_focal_length)
        f = var_lens_focal_length(
            n=jnp.asarray(n),
            R1=jnp.asarray(R1),
            R2=jnp.asarray(R2),
        )

        chex.assert_trees_all_close(f, jnp.array(expected), atol=1e-6)


class TestLensCreation(chex.TestCase):
    @chex.all_variants()
    def test_double_convex_lens(self):
        """Test creation of double convex lens."""
        params = double_convex_lens(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
        )

        assert jnp.all(params.R1 > 0)
        assert jnp.all(params.R2 > 0)

        calculated_f = lens_focal_length(params.n, params.R1, params.R2)
        chex.assert_trees_all_close(calculated_f, params.focal_length, atol=1e-6)

    @chex.all_variants()
    def test_double_concave_lens(self):
        """Test creation of double concave lens."""
        params = double_concave_lens(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
        )

        assert jnp.all(params.R1 < 0)
        assert jnp.all(params.R2 < 0)

        calculated_f = lens_focal_length(params.n, params.R1, params.R2)
        chex.assert_trees_all_close(calculated_f, params.focal_length, atol=1e-6)

    @chex.all_variants()
    @parameterized.parameters(
        {"convex_first": True},
        {"convex_first": False},
    )
    def test_plano_convex_lens(self, convex_first):
        """Test creation of plano-convex lens."""
        var_plano_convex_lens = self.variant(plano_convex_lens)
        params = var_plano_convex_lens(
            focal_length=jnp.asarray(0.1),
            diameter=jnp.asarray(0.05),
            n=jnp.asarray(1.5),
            center_thickness=jnp.asarray(0.01),
            convex_first=jnp.asarray(convex_first),
        )

        if convex_first:
            assert jnp.isfinite(params.R1)
            assert not jnp.isfinite(params.R2)
        else:
            assert not jnp.isfinite(params.R1)
            assert jnp.isfinite(params.R2)

    @chex.all_variants()
    def test_meniscus_lens(self):
        """Test creation of meniscus lens."""
        params = meniscus_lens(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R_ratio=jnp.array(1.2),
            convex_first=jnp.array(True),
        )

        assert jnp.all(params.R1 > 0)
        assert jnp.all(params.R2 < 0)

        calculated_f = lens_focal_length(params.n, params.R1, params.R2)
        chex.assert_trees_all_close(calculated_f, params.focal_length, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
