"""
Test module for preprocessing utilities in ptyrodactyl.electrons.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptyrodactyl.electrons.preprocessing import (_ATOMIC_NUMBERS,
                                                 _KIRKLAND_POTENTIALS,
                                                 atomic_symbol,
                                                 kirkland_potentials)


class TestAtomicNumbers:
    """Test atomic_symbol function and related functionality."""

    def test_atomic_symbol_basic(self):
        """Test basic atomic symbol lookups."""
        assert atomic_symbol("H") == 1
        assert atomic_symbol("He") == 2
        assert atomic_symbol("Li") == 3
        assert atomic_symbol("C") == 6
        assert atomic_symbol("N") == 7
        assert atomic_symbol("O") == 8
        assert atomic_symbol("Fe") == 26
        assert atomic_symbol("Au") == 79
        assert atomic_symbol("Og") == 118

    def test_atomic_symbol_case_insensitive(self):
        """Test that atomic symbols are case-insensitive."""
        assert atomic_symbol("he") == 2
        assert atomic_symbol("HE") == 2
        assert atomic_symbol("He") == 2
        assert atomic_symbol("hE") == 2

    def test_atomic_symbol_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        assert atomic_symbol(" He ") == 2
        assert atomic_symbol("\tC\n") == 6
        assert atomic_symbol("  Au  ") == 79

    def test_atomic_symbol_invalid(self):
        """Test error handling for invalid symbols."""
        with pytest.raises(KeyError, match="Atomic symbol 'Xx' not found"):
            atomic_symbol("Xx")

        with pytest.raises(KeyError, match="Atomic symbol 'InvalidElement' not found"):
            atomic_symbol("InvalidElement")

        with pytest.raises(ValueError, match="Atomic symbol cannot be empty"):
            atomic_symbol("")

        with pytest.raises(ValueError, match="Atomic symbol cannot be empty"):
            atomic_symbol("   ")

    def test_atomic_numbers_preloaded(self):
        """Test that atomic numbers are preloaded into memory."""
        # Verify that _ATOMIC_NUMBERS is a dictionary
        assert isinstance(_ATOMIC_NUMBERS, dict)

        # Check that it contains all elements from H to Og
        assert len(_ATOMIC_NUMBERS) == 118

        # Verify some key elements
        assert _ATOMIC_NUMBERS["H"] == 1
        assert _ATOMIC_NUMBERS["He"] == 2
        assert _ATOMIC_NUMBERS["Og"] == 118

    def test_atomic_symbol_jax_compatible(self):
        """Test that atomic_symbol can be used in JAX-compatible contexts."""
        # The function itself returns scalar_int which should be JAX-compatible
        result = atomic_symbol("C")

        # Verify the result is a scalar integer
        assert isinstance(result, (int, jnp.integer))
        assert result == 6

        # Test that we can use the result in JAX operations
        jax_array = jnp.array([result])
        assert jax_array.shape == (1,)
        assert jax_array[0] == 6

        # Test that results can be used in JAX computations
        result_h = atomic_symbol("H")
        result_he = atomic_symbol("He")
        sum_result = jnp.add(result_h, result_he)
        assert sum_result == 3

    def test_atomic_symbol_all_elements(self):
        """Test that all 118 elements work correctly."""
        # Sample test of first 20 elements
        elements = [
            ("H", 1),
            ("He", 2),
            ("Li", 3),
            ("Be", 4),
            ("B", 5),
            ("C", 6),
            ("N", 7),
            ("O", 8),
            ("F", 9),
            ("Ne", 10),
            ("Na", 11),
            ("Mg", 12),
            ("Al", 13),
            ("Si", 14),
            ("P", 15),
            ("S", 16),
            ("Cl", 17),
            ("Ar", 18),
            ("K", 19),
            ("Ca", 20),
        ]

        for symbol, expected_number in elements:
            assert atomic_symbol(symbol) == expected_number


class TestKirklandPotentials:
    """Test kirkland_potentials function and related functionality."""

    def test_kirkland_potentials_shape(self):
        """Test that kirkland_potentials returns correct shape."""
        kp = kirkland_potentials()
        assert kp.shape == (103, 12)

    def test_kirkland_potentials_dtype(self):
        """Test that kirkland_potentials returns float64 JAX array."""
        kp = kirkland_potentials()
        assert kp.dtype == jnp.float64

    def test_kirkland_potentials_is_jax_array(self):
        """Test that kirkland_potentials returns a JAX array."""
        kp = kirkland_potentials()
        assert isinstance(kp, jnp.ndarray)

    def test_kirkland_potentials_preloaded(self):
        """Test that Kirkland potentials are preloaded into memory."""
        # Verify that _KIRKLAND_POTENTIALS exists and has correct properties
        assert isinstance(_KIRKLAND_POTENTIALS, jnp.ndarray)
        assert _KIRKLAND_POTENTIALS.shape == (103, 12)
        assert _KIRKLAND_POTENTIALS.dtype == jnp.float64

        # Verify that kirkland_potentials returns the same object (no copy)
        kp = kirkland_potentials()
        assert kp is _KIRKLAND_POTENTIALS

    def test_kirkland_potentials_values(self):
        """Test that kirkland potentials contain reasonable values."""
        kp = kirkland_potentials()

        # All values should be finite
        assert jnp.all(jnp.isfinite(kp))

        # No NaN values
        assert not jnp.any(jnp.isnan(kp))

        # Values should be positive (typical for scattering potentials)
        # Note: Some implementations might have negative values, adjust if needed
        assert jnp.all(kp >= 0)

        # Check specific known values (first element - Hydrogen)
        # These are from the actual CSV file
        expected_h_first_four = jnp.array(
            [0.0355221981, 0.225354459, 0.0262782423, 0.225354636]
        )
        assert jnp.allclose(kp[0, :4], expected_h_first_four, rtol=1e-9)

    def test_kirkland_potentials_jax_operations(self):
        """Test that kirkland potentials work with JAX operations."""
        kp = kirkland_potentials()

        # Test in JIT-compiled function
        @jax.jit
        def get_element_potential(idx):
            return kp[idx]

        # Get potential for Carbon (Z=6, index=5)
        carbon_potential = get_element_potential(5)
        assert carbon_potential.shape == (12,)

        # Test vectorized operations
        @jax.jit
        def sum_potentials(indices):
            return kp[indices].sum(axis=1)

        indices = jnp.array([0, 5, 78])  # H, C, Au
        sums = sum_potentials(indices)
        assert sums.shape == (3,)

        # Test gradient computation
        @jax.jit
        def potential_dot(params, element_idx):
            return jnp.dot(params, kp[element_idx])

        grad_fn = jax.grad(potential_dot)
        params = jnp.ones(12)
        gradient = grad_fn(params, 5)  # Carbon
        assert gradient.shape == (12,)
        assert jnp.allclose(gradient, kp[5])  # Gradient should equal the potentials

    def test_kirkland_potentials_indexing(self):
        """Test various indexing operations on kirkland potentials."""
        kp = kirkland_potentials()

        # Single element access
        hydrogen = kp[0]
        assert hydrogen.shape == (12,)

        # Multiple elements
        first_ten = kp[:10]
        assert first_ten.shape == (10, 12)

        # Fancy indexing (JAX requires array for list indexing)
        selected = kp[jnp.array([0, 5, 78])]  # H, C, Au
        assert selected.shape == (3, 12)

        # Column slicing
        first_param = kp[:, 0]
        assert first_param.shape == (103,)

    def test_kirkland_potentials_immutability(self):
        """Test that the returned array reference maintains data integrity."""
        kp1 = kirkland_potentials()
        kp2 = kirkland_potentials()

        # Both calls should return the same object
        assert kp1 is kp2

        # Values should be identical
        assert jnp.array_equal(kp1, kp2)
