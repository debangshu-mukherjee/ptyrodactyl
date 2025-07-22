import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from beartype.roar import BeartypeCallHintParamViolation
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, Integer, Num

from ptyrodactyl.photons.photon_types import (
    LensParams,
    GridParams,
    OpticalWavefront,
    MicroscopeData,
    Diffractogram,
    make_lens_params,
    make_grid_params,
    make_optical_wavefront,
    make_microscope_data,
    make_diffractogram,
)

jax.config.update("jax_enable_x64", True)


class TestLensParams(chex.TestCase):
    """Test the LensParams class structure and PyTree compatibility."""

    def test_lens_params_structure(self):
        """Test the structure of LensParams."""
        params = LensParams(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )

        assert hasattr(params, "focal_length")
        assert hasattr(params, "diameter")
        assert hasattr(params, "n")
        assert hasattr(params, "center_thickness")
        assert hasattr(params, "R1")
        assert hasattr(params, "R2")

    def test_lens_params_pytree_compatibility(self):
        """Test that LensParams is compatible with JAX transformations."""
        params = LensParams(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )

        @jax.jit
        def fn(p):
            return p.focal_length * 2.0

        result = fn(params)
        assert result == 0.2

        array_params = LensParams(
            focal_length=jnp.array([0.1, 0.2]),
            diameter=jnp.array([0.05, 0.06]),
            n=jnp.array([1.5, 1.6]),
            center_thickness=jnp.array([0.01, 0.02]),
            R1=jnp.array([0.2, 0.3]),
            R2=jnp.array([0.2, 0.3]),
        )

        @jax.vmap
        def batch_fn(p):
            return p

        _ = batch_fn(array_params)


class TestMakeLensParams(chex.TestCase):
    """Test the make_lens_params factory function."""

    def test_make_lens_params_with_valid_types(self):
        """Test that make_lens_params works with valid types."""
        lens_params = make_lens_params(
            focal_length=jnp.array(0.1),
            diameter=jnp.array(0.05),
            n=jnp.array(1.5),
            center_thickness=jnp.array(0.01),
            R1=jnp.array(0.2),
            R2=jnp.array(0.2),
        )

        assert isinstance(lens_params, LensParams)

        assert lens_params.focal_length == jnp.array(0.1)
        assert lens_params.diameter == jnp.array(0.05)
        assert lens_params.n == jnp.array(1.5)
        assert lens_params.center_thickness == jnp.array(0.01)
        assert lens_params.R1 == jnp.array(0.2)
        assert lens_params.R2 == jnp.array(0.2)

    def test_make_lens_params_with_invalid_types(self):
        """Test that make_lens_params raises an error with invalid types."""
        with pytest.raises(Exception):
            make_lens_params(
                focal_length=1,
                diameter=jnp.array(0.05),
                n=jnp.array(1.5),
                center_thickness=jnp.array(0.01),
                R1=jnp.array(0.2),
                R2=jnp.array(0.2),
            )

        with pytest.raises(Exception):
            make_lens_params(
                focal_length=jnp.array(0.1),
                diameter=0.05,
                n=jnp.array(1.5),
                center_thickness=jnp.array(0.01),
                R1=jnp.array(0.2),
                R2=jnp.array(0.2),
            )


class TestGridParams(chex.TestCase):
    """Test the GridParams class structure and PyTree compatibility."""

    def test_grid_params_structure(self):
        """Test the structure of GridParams."""
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[0]))
        phase = jnp.zeros(shape)
        transmission = jnp.ones(shape)

        params = GridParams(X=X, Y=Y, phase_profile=phase, transmission=transmission)

        assert hasattr(params, "X")
        assert hasattr(params, "Y")
        assert hasattr(params, "phase_profile")
        assert hasattr(params, "transmission")

        chex.assert_shape(params.X, shape)
        chex.assert_shape(params.Y, shape)
        chex.assert_shape(params.phase_profile, shape)
        chex.assert_shape(params.transmission, shape)

    def test_grid_params_pytree_compatibility(self):
        """Test that GridParams is compatible with JAX transformations."""
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[0]))
        phase = jnp.zeros(shape)
        transmission = jnp.ones(shape)

        params = GridParams(X=X, Y=Y, phase_profile=phase, transmission=transmission)

        @jax.jit
        def fn(p):
            return jnp.sum(p.X + p.Y)

        _ = fn(params)


class TestMakeGridParams(chex.TestCase):
    """Test the make_grid_params factory function."""

    def test_make_grid_params_with_valid_types(self):
        """Test that make_grid_params works with valid types."""
        shape = (32, 32)
        X, Y = jnp.meshgrid(
            jnp.arange(shape[1], dtype=jnp.float64),
            jnp.arange(shape[0], dtype=jnp.float64),
        )
        phase_profile = jnp.zeros(shape)
        transmission = jnp.ones(shape)

        grid_params = make_grid_params(
            X=X, Y=Y, phase_profile=phase_profile, transmission=transmission
        )

        assert isinstance(grid_params, GridParams)

        chex.assert_trees_all_close(grid_params.X, X)
        chex.assert_trees_all_close(grid_params.Y, Y)
        chex.assert_trees_all_close(grid_params.phase_profile, phase_profile)
        chex.assert_trees_all_close(grid_params.transmission, transmission)

        chex.assert_shape(grid_params.X, shape)
        chex.assert_shape(grid_params.Y, shape)
        chex.assert_shape(grid_params.phase_profile, shape)
        chex.assert_shape(grid_params.transmission, shape)

    def test_make_grid_params_with_invalid_types(self):
        """Test that make_grid_params raises an error with invalid types."""
        shape = (32, 32)
        X, Y = jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[0]))
        phase_profile = jnp.zeros(shape)
        transmission = jnp.ones(shape)

        with pytest.raises(Exception):
            make_grid_params(
                X=jnp.arange(10),
                Y=Y,
                phase_profile=phase_profile,
                transmission=transmission,
            )

        with pytest.raises(Exception):
            make_grid_params(
                X=X,
                Y=Y,
                phase_profile=jnp.zeros((16, 16)),
                transmission=transmission,
            )

        with pytest.raises(Exception):
            make_grid_params(
                X=X,
                Y=Y,
                phase_profile=phase_profile,
                transmission=jnp.ones(shape, dtype=jnp.complex128),
            )


class TestOpticalWavefront(chex.TestCase):
    """Test the OpticalWavefront class structure and PyTree compatibility."""

    def test_wavefront_structure(self):
        """Test the structure of OpticalWavefront."""
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)

        wavefront = OpticalWavefront(
            field=field,
            wavelength=jnp.array(500e-9),
            dx=jnp.array(1e-6),
            z_position=jnp.array(0.0),
        )

        assert hasattr(wavefront, "field")
        assert hasattr(wavefront, "wavelength")
        assert hasattr(wavefront, "dx")
        assert hasattr(wavefront, "z_position")

        chex.assert_shape(wavefront.field, shape)

    def test_wavefront_pytree_compatibility(self):
        """Test that OpticalWavefront is compatible with JAX transformations."""
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)

        wavefront = OpticalWavefront(
            field=field,
            wavelength=jnp.array(500e-9),
            dx=jnp.array(1e-6),
            z_position=jnp.array(0.0),
        )

        @jax.jit
        def fn(w):
            return jnp.sum(jnp.abs(w.field))

        result = fn(wavefront)
        assert result == 32 * 32

        @jax.jit
        def propagate(w, z):
            return OpticalWavefront(
                field=w.field,
                wavelength=w.wavelength,
                dx=w.dx,
                z_position=w.z_position + z,
            )

        new_wavefront = propagate(wavefront, jnp.array(0.1))
        assert new_wavefront.z_position == 0.1


class TestMakeOpticalWavefront(chex.TestCase):
    """Test the make_optical_wavefront factory function."""

    def test_make_optical_wavefront_with_valid_types(self):
        """Test that make_optical_wavefront works with valid types."""
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)
        z_position = jnp.array(0.0)

        wavefront = make_optical_wavefront(
            field=field, wavelength=wavelength, dx=dx, z_position=z_position
        )

        assert isinstance(wavefront, OpticalWavefront)

        chex.assert_trees_all_close(wavefront.field, field)
        assert wavefront.wavelength == wavelength
        assert wavefront.dx == dx
        assert wavefront.z_position == z_position

        chex.assert_shape(wavefront.field, shape)
        assert wavefront.field.dtype == jnp.complex128

    def test_make_optical_wavefront_with_invalid_types(self):
        """Test that make_optical_wavefront raises an error with invalid types."""
        shape = (32, 32)
        field = jnp.ones(shape, dtype=jnp.complex128)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)
        z_position = jnp.array(0.0)

        with pytest.raises(Exception):
            make_optical_wavefront(
                field=jnp.ones(shape),
                wavelength=wavelength,
                dx=dx,
                z_position=z_position,
            )

        with pytest.raises(Exception):
            make_optical_wavefront(
                field=field,
                wavelength=500e-9,
                dx=dx,
                z_position=z_position,
            )

        with pytest.raises(Exception):
            make_optical_wavefront(
                field=field,
                wavelength=wavelength,
                dx=1e-6,
                z_position=z_position,
            )

        with pytest.raises(Exception):
            make_optical_wavefront(
                field=field,
                wavelength=wavelength,
                dx=dx,
                z_position=0.0,
            )


class TestMicroscopeData(chex.TestCase):
    """Test the MicroscopeData class structure and PyTree compatibility."""

    def test_microscope_data_structure(self):
        """Test the structure of MicroscopeData."""
        shape_3d = (5, 32, 32)
        image_data = jnp.ones(shape_3d, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        microscope_data = MicroscopeData(
            image_data=image_data, wavelength=wavelength, dx=dx
        )

        assert hasattr(microscope_data, "image_data")
        assert hasattr(microscope_data, "wavelength")
        assert hasattr(microscope_data, "dx")

        chex.assert_shape(microscope_data.image_data, shape_3d)

    def test_microscope_data_pytree_compatibility(self):
        """Test that MicroscopeData is compatible with JAX transformations."""
        shape_3d = (5, 32, 32)
        image_data = jnp.ones(shape_3d, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        microscope_data = MicroscopeData(
            image_data=image_data, wavelength=wavelength, dx=dx
        )

        @jax.jit
        def fn(m):
            return jnp.sum(m.image_data)

        result = fn(microscope_data)
        assert result == 5 * 32 * 32


class TestMakeMicroscopeData(chex.TestCase):
    """Test the make_microscope_data factory function."""

    def test_make_microscope_data_3d_with_valid_types(self):
        """Test that make_microscope_data works with valid 3D types."""
        shape_3d = (5, 32, 32)
        image_data_3d = jnp.ones(shape_3d, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        microscope_data = make_microscope_data(
            image_data=image_data_3d, wavelength=wavelength, dx=dx
        )

        assert isinstance(microscope_data, MicroscopeData)

        chex.assert_trees_all_close(microscope_data.image_data, image_data_3d)
        assert microscope_data.wavelength == wavelength
        assert microscope_data.dx == dx

        chex.assert_shape(microscope_data.image_data, shape_3d)
        assert microscope_data.image_data.dtype == jnp.float64

    def test_make_microscope_data_4d_with_valid_types(self):
        """Test that make_microscope_data works with valid 4D types."""
        shape_4d = (3, 3, 32, 32)
        image_data_4d = jnp.ones(shape_4d, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        microscope_data = make_microscope_data(
            image_data=image_data_4d, wavelength=wavelength, dx=dx
        )

        assert isinstance(microscope_data, MicroscopeData)

        chex.assert_trees_all_close(microscope_data.image_data, image_data_4d)
        assert microscope_data.wavelength == wavelength
        assert microscope_data.dx == dx

        chex.assert_shape(microscope_data.image_data, shape_4d)
        assert microscope_data.image_data.dtype == jnp.float64

    def test_make_microscope_data_with_invalid_types(self):
        """Test that make_microscope_data raises an error with invalid types."""
        shape_3d = (5, 32, 32)
        image_data_3d = jnp.ones(shape_3d, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        with pytest.raises(Exception):
            make_microscope_data(
                image_data=jnp.ones(
                    shape_3d, dtype=jnp.complex128
                ),
                wavelength=wavelength,
                dx=dx,
            )

        with pytest.raises(Exception):
            make_microscope_data(
                image_data=jnp.ones((32, 32), dtype=jnp.float64),
                wavelength=wavelength,
                dx=dx,
            )

        with pytest.raises(Exception):
            make_microscope_data(
                image_data=image_data_3d,
                wavelength=500e-9,
                dx=dx,
            )

        with pytest.raises(Exception):
            make_microscope_data(
                image_data=image_data_3d,
                wavelength=wavelength,
                dx=1e-6,
            )


class TestDiffractogram(chex.TestCase):
    """Test the Diffractogram class structure and PyTree compatibility."""

    def test_diffractogram_structure(self):
        """Test the structure of Diffractogram."""
        shape = (32, 32)
        image = jnp.ones(shape, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        diffractogram = Diffractogram(image=image, wavelength=wavelength, dx=dx)

        assert hasattr(diffractogram, "image")
        assert hasattr(diffractogram, "wavelength")
        assert hasattr(diffractogram, "dx")

        chex.assert_shape(diffractogram.image, shape)

    def test_diffractogram_pytree_compatibility(self):
        """Test that Diffractogram is compatible with JAX transformations."""
        shape = (32, 32)
        image = jnp.ones(shape, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        diffractogram = Diffractogram(image=image, wavelength=wavelength, dx=dx)

        @jax.jit
        def fn(d):
            return jnp.sum(d.image)

        result = fn(diffractogram)
        assert result == 32 * 32


class TestMakeDiffractogram(chex.TestCase):
    """Test the make_diffractogram factory function."""

    def test_make_diffractogram_with_valid_types(self):
        """Test that make_diffractogram works with valid types."""
        shape = (32, 32)
        image = jnp.ones(shape, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        diffractogram = make_diffractogram(image=image, wavelength=wavelength, dx=dx)

        assert isinstance(diffractogram, Diffractogram)

        chex.assert_trees_all_close(diffractogram.image, image)
        assert diffractogram.wavelength == wavelength
        assert diffractogram.dx == dx

        chex.assert_shape(diffractogram.image, shape)
        assert diffractogram.image.dtype == jnp.float64

    def test_make_diffractogram_with_invalid_types(self):
        """Test that make_diffractogram raises an error with invalid types."""
        shape = (32, 32)
        image = jnp.ones(shape, dtype=jnp.float64)
        wavelength = jnp.array(500e-9)
        dx = jnp.array(1e-6)

        with pytest.raises(Exception):
            make_diffractogram(
                image=jnp.ones(shape, dtype=jnp.complex128),
                wavelength=wavelength,
                dx=dx,
            )

        with pytest.raises(Exception):
            make_diffractogram(
                image=jnp.ones((3, 32, 32), dtype=jnp.float64),
                wavelength=wavelength,
                dx=dx,
            )

        with pytest.raises(Exception):
            make_diffractogram(
                image=image, wavelength=500e-9, dx=dx
            )

        with pytest.raises(Exception):
            make_diffractogram(
                image=image, wavelength=wavelength, dx=1e-6
            )


if __name__ == "__main__":
    pytest.main([__file__])
