from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Complex, Float, Shaped

# Import your functions here
from ptyrodactyl.electrons import (propagation_func, 
                                   transmission_func,
                                   wavelength_ang)

# Set a random seed for reproducibility
key = jax.random.PRNGKey(0)


@chex.all_variants
class TestWavelengthAng:

    @pytest.fixture
    def voltage_kV(self):
        return 200.0  # Using a single voltage value for all tests

    def test_basic_functionality(self, variant, voltage_kV: float) -> None:
        """Test basic functionality of wavelength_ang."""
        fn = variant(wavelength_ang)
        result = fn(voltage_kV)

        chex.assert_type(result, float)
        chex.assert_scalar_positive(result)

        # Expected value (approximate)
        expected_value = 0.0251
        chex.assert_scalar_near(result, expected_value, atol=1e-4)

    def test_array_input(self, variant) -> None:
        """Test wavelength_ang with array input."""
        voltages: Float[jax.Array, "3"] = jnp.array([100.0, 200.0, 300.0])
        fn = jax.vmap(variant(wavelength_ang))
        results = fn(voltages)

        chex.assert_shape(results, (3,))
        chex.assert_type(results, jnp.float32)
        chex.assert_trees_all_close(
            results, jnp.array([0.0370, 0.0251, 0.0197]), atol=1e-4
        )

    def test_jit_compilation(self, variant, voltage_kV: float) -> None:
        """Test that wavelength_ang can be JIT-compiled."""
        jitted_fn = jax.jit(variant(wavelength_ang))
        result = jitted_fn(voltage_kV)

        assert isinstance(result, float)
        chex.assert_scalar_positive(result)

    def test_grad(self, variant, voltage_kV: float) -> None:
        """Test that wavelength_ang is differentiable."""
        grad_fn = jax.grad(variant(wavelength_ang))
        grad_value = grad_fn(voltage_kV)

        assert isinstance(grad_value, float)
        assert grad_value < 0  # Wavelength decreases as voltage increases

    def test_invalid_voltage(self, variant) -> None:
        """Test wavelength_ang behavior with invalid voltage inputs."""
        invalid_voltages = [-100.0, 0.0, float("nan"), float("inf")]
        for invalid_voltage in invalid_voltages:
            with pytest.raises((ValueError, FloatingPointError)):
                variant(wavelength_ang)(invalid_voltage)

    def test_precision(self, variant) -> None:
        """Test the precision of wavelength_ang calculations."""
        result = variant(wavelength_ang)(200.0)
        expected = 0.0251

        assert abs(result - expected) < 1e-4


@chex.all_variants
class TestTransmissionFunc:

    @pytest.fixture
    def voltage_kV(self):
        return 200.0  # Using a single voltage value for all tests

    @pytest.fixture
    def pot_slice(self) -> Float[Array, "H W"]:
        return jnp.ones((64, 64))  # Example potential slice

    def test_basic_functionality(
        self, variant, pot_slice: Float[Array, "H W"], voltage_kV: float
    ) -> None:
        """Test basic functionality of transmission_func."""
        fn = variant(transmission_func)
        result = fn(pot_slice, voltage_kV)

        chex.assert_type(result, jnp.complex64)
        chex.assert_shape(result, pot_slice.shape)
        chex.assert_trees_all_finite(result)

    def test_output_range(
        self, variant, pot_slice: Float[Array, "H W"], voltage_kV: float
    ) -> None:
        """Test if the output has magnitude 1 everywhere."""
        fn = variant(transmission_func)
        result = fn(pot_slice, voltage_kV)

        magnitude = jnp.abs(result)
        chex.assert_trees_all_close(magnitude, jnp.ones_like(magnitude), atol=1e-6)

    def test_jit_compilation(
        self, variant, pot_slice: Float[Array, "H W"], voltage_kV: float
    ) -> None:
        """Test that transmission_func can be JIT-compiled."""
        jitted_fn = jax.jit(variant(transmission_func))
        result = jitted_fn(pot_slice, voltage_kV)

        assert isinstance(result, jax.Array)
        chex.assert_type(result, jnp.complex64)
        chex.assert_shape(result, pot_slice.shape)

    def test_grad(
        self, variant, pot_slice: Float[Array, "H W"], voltage_kV: float
    ) -> None:
        """Test that transmission_func is differentiable."""
        grad_fn = jax.grad(
            lambda x, v: jnp.sum(jnp.real(variant(transmission_func)(x, v)))
        )
        grad_value = grad_fn(pot_slice, voltage_kV)

        assert isinstance(grad_value, jax.Array)
        chex.assert_shape(grad_value, pot_slice.shape)
        chex.assert_trees_all_finite(grad_value)

    def test_invalid_voltage(self, variant, pot_slice: Float[Array, "H W"]) -> None:
        """Test transmission_func behavior with invalid voltage inputs."""
        invalid_voltages = [-100.0, 0.0, float("nan"), float("inf")]
        for invalid_voltage in invalid_voltages:
            with pytest.raises((ValueError, FloatingPointError)):
                variant(transmission_func)(pot_slice, invalid_voltage)

    def test_different_input_shapes(self, variant) -> None:
        """Test transmission_func with different input shapes."""
        fn = variant(transmission_func)
        shapes = [(32, 32), (64, 64), (128, 128)]
        voltage_kV = 200.0

        for shape in shapes:
            pot_slice = jnp.ones(shape)
            result = fn(pot_slice, voltage_kV)
            chex.assert_shape(result, shape)


@chex.all_variants
class TestPropagationFunc:

    @pytest.fixture
    def imsize(self) -> Shaped[Array, "2"]:
        return jnp.array([64, 64])

    @pytest.fixture
    def thickness_ang(self) -> float:
        return 10.0

    @pytest.fixture
    def voltage_kV(self) -> float:
        return 200.0

    @pytest.fixture
    def calib_ang(self) -> float:
        return 0.1

    def test_basic_functionality(
        self, variant, imsize, thickness_ang, voltage_kV, calib_ang
    ):
        """Test basic functionality of propagation_func."""
        fn = variant(propagation_func)
        result = fn(imsize, thickness_ang, voltage_kV, calib_ang)

        chex.assert_type(result, jnp.complex64)
        chex.assert_shape(result, tuple(imsize))
        chex.assert_trees_all_finite(result)

    def test_output_properties(
        self, variant, imsize, thickness_ang, voltage_kV, calib_ang
    ):
        """Test specific properties of the output."""
        fn = variant(propagation_func)
        result = fn(imsize, thickness_ang, voltage_kV, calib_ang)

        # Check that the magnitude is 1 everywhere
        magnitude = jnp.abs(result)
        chex.assert_trees_all_close(magnitude, jnp.ones_like(magnitude), atol=1e-6)

        # Check that the phase is symmetric
        phase = jnp.angle(result)
        chex.assert_trees_all_close(phase, jnp.flip(phase), atol=1e-6)

    def test_jit_compilation(
        self, variant, imsize, thickness_ang, voltage_kV, calib_ang
    ):
        """Test that propagation_func can be JIT-compiled."""
        jitted_fn = jax.jit(variant(propagation_func))
        result = jitted_fn(imsize, thickness_ang, voltage_kV, calib_ang)

        chex.assert_type(result, jnp.complex64)
        chex.assert_shape(result, tuple(imsize))

    def test_grad(self, variant, imsize, thickness_ang, voltage_kV, calib_ang):
        """Test that propagation_func is differentiable."""

        def loss_fn(thickness):
            return jnp.sum(
                jnp.abs(
                    variant(propagation_func)(imsize, thickness, voltage_kV, calib_ang)
                )
            )

        grad_fn = jax.grad(loss_fn)
        grad_value = grad_fn(thickness_ang)

        chex.assert_type(grad_value, float)
        chex.assert_scalar_non_zero(grad_value)

    def test_different_input_shapes(
        self, variant, thickness_ang, voltage_kV, calib_ang
    ):
        """Test propagation_func with different input shapes."""
        fn = variant(propagation_func)
        shapes = [(32, 32), (64, 64), (128, 128)]

        for shape in shapes:
            imsize = jnp.array(shape)
            result = fn(imsize, thickness_ang, voltage_kV, calib_ang)
            chex.assert_shape(result, shape)

    def test_wavelength_dependency(self, variant, imsize, thickness_ang, calib_ang):
        """Test that the output changes with different voltages (wavelengths)."""
        fn = variant(propagation_func)
        result1 = fn(imsize, thickness_ang, 100.0, calib_ang)
        result2 = fn(imsize, thickness_ang, 300.0, calib_ang)

        assert not jnp.allclose(result1, result2)

    def test_thickness_dependency(self, variant, imsize, voltage_kV, calib_ang):
        """Test that the output changes with different thicknesses."""
        fn = variant(propagation_func)
        result1 = fn(imsize, 10.0, voltage_kV, calib_ang)
        result2 = fn(imsize, 20.0, voltage_kV, calib_ang)

        assert not jnp.allclose(result1, result2)

    @pytest.mark.parametrize("invalid_input", [-1.0, 0.0, float("nan"), float("inf")])
    def test_invalid_inputs(self, variant, imsize, invalid_input):
        """Test propagation_func behavior with invalid inputs."""
        fn = variant(propagation_func)
        with pytest.raises((ValueError, FloatingPointError)):
            fn(imsize, invalid_input, invalid_input, invalid_input)
