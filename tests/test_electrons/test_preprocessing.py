"""
Test module for preprocessing utilities in ptyrodactyl.electrons.
"""

import tempfile
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.electrons.preprocessing import (
    _ATOMIC_NUMBERS,
    _KIRKLAND_POTENTIALS,
    _parse_xyz_metadata,
    atomic_symbol,
    kirkland_potentials,
    parse_xyz,
)


class TestAtomicSymbol(chex.TestCase):
    """Test atomic_symbol function and related functionality."""

    def test_atomic_symbol_basic(self) -> None:
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

    def test_atomic_symbol_case_insensitive(self) -> None:
        """Test that atomic symbols are case-insensitive."""
        assert atomic_symbol("he") == 2
        assert atomic_symbol("HE") == 2
        assert atomic_symbol("He") == 2
        assert atomic_symbol("hE") == 2

    def test_atomic_symbol_whitespace_handling(self) -> None:
        """Test that whitespace is properly handled."""
        assert atomic_symbol(" He ") == 2
        assert atomic_symbol("\tC\n") == 6
        assert atomic_symbol("  Au  ") == 79

    def test_atomic_symbol_invalid(self) -> None:
        """Test error handling for invalid symbols."""
        with pytest.raises(KeyError, match="Atomic symbol 'Xx' not found"):
            atomic_symbol("Xx")

        with pytest.raises(KeyError, match="Atomic symbol 'InvalidElement' not found"):
            atomic_symbol("InvalidElement")

        with pytest.raises(ValueError, match="Atomic symbol cannot be empty"):
            atomic_symbol("")

        with pytest.raises(ValueError, match="Atomic symbol cannot be empty"):
            atomic_symbol("   ")

    def test_atomic_numbers_preloaded(self) -> None:
        """Test that atomic numbers are preloaded into memory."""
        # Verify that _ATOMIC_NUMBERS is a dictionary
        assert isinstance(_ATOMIC_NUMBERS, dict)

        # Check that it contains all elements from H to Og
        assert len(_ATOMIC_NUMBERS) == 118

        # Verify some key elements
        assert _ATOMIC_NUMBERS["H"] == 1
        assert _ATOMIC_NUMBERS["He"] == 2
        assert _ATOMIC_NUMBERS["Og"] == 118

    @chex.variants(with_jit=True, without_jit=True)
    def test_atomic_symbol_jax_compatible(self) -> None:
        """Test that atomic_symbol can be used in JAX-compatible contexts."""
        # The function itself returns scalar_int which should be JAX-compatible
        result = atomic_symbol("C")

        # Verify the result is a scalar integer
        assert isinstance(result, int | jnp.integer)
        assert result == 6

        # Test that we can use the result in JAX operations
        def create_array(value):
            return jnp.array([value])

        jax_array = self.variant(create_array)(result)
        assert jax_array.shape == (1,)
        assert jax_array[0] == 6

        # Test that results can be used in JAX computations
        def add_atomic_numbers():
            result_h = atomic_symbol("H")
            result_he = atomic_symbol("He")
            return jnp.add(result_h, result_he)

        sum_result = self.variant(add_atomic_numbers)()
        assert sum_result == 3

    def test_atomic_symbol_all_elements(self) -> None:
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


class TestKirklandPotentials(chex.TestCase):
    """Test kirkland_potentials function and related functionality."""

    def test_kirkland_potentials_shape(self) -> None:
        """Test that kirkland_potentials returns correct shape."""
        kp = kirkland_potentials()
        assert kp.shape == (103, 12)

    def test_kirkland_potentials_dtype(self) -> None:
        """Test that kirkland_potentials returns float64 JAX array."""
        kp = kirkland_potentials()
        assert kp.dtype == jnp.float64

    def test_kirkland_potentials_is_jax_array(self) -> None:
        """Test that kirkland_potentials returns a JAX array."""
        kp = kirkland_potentials()
        assert isinstance(kp, jnp.ndarray)

    def test_kirkland_potentials_preloaded(self) -> None:
        """Test that Kirkland potentials are preloaded into memory."""
        # Verify that _KIRKLAND_POTENTIALS exists and has correct properties
        assert isinstance(_KIRKLAND_POTENTIALS, jnp.ndarray)
        assert _KIRKLAND_POTENTIALS.shape == (103, 12)
        assert _KIRKLAND_POTENTIALS.dtype == jnp.float64

        # Verify that kirkland_potentials returns the same object (no copy)
        kp = kirkland_potentials()
        assert kp is _KIRKLAND_POTENTIALS

    def test_kirkland_potentials_values(self) -> None:
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
        expected_h_first_four = jnp.array([0.0355221981, 0.225354459, 0.0262782423, 0.225354636])
        assert jnp.allclose(kp[0, :4], expected_h_first_four, rtol=1e-9)

    @chex.variants(with_jit=True, without_jit=True)
    def test_kirkland_potentials_jax_operations(self) -> None:
        """Test that kirkland potentials work with JAX operations."""
        kp = kirkland_potentials()

        # Test in JIT-compiled function
        def get_element_potential(idx):
            return kp[idx]

        # Get potential for Carbon (Z=6, index=5)
        carbon_potential = self.variant(get_element_potential)(5)
        assert carbon_potential.shape == (12,)

        # Test vectorized operations
        def sum_potentials(indices):
            return kp[indices].sum(axis=1)

        indices = jnp.array([0, 5, 78])  # H, C, Au
        sums = self.variant(sum_potentials)(indices)
        assert sums.shape == (3,)

        # Test gradient computation
        def potential_dot(params, element_idx):
            return jnp.dot(params, kp[element_idx])

        grad_fn = jax.grad(potential_dot)
        params = jnp.ones(12)
        gradient = self.variant(grad_fn)(params, 5)  # Carbon
        assert gradient.shape == (12,)
        assert jnp.allclose(gradient, kp[5])  # Gradient should equal the potentials

    @chex.variants(with_jit=True, without_jit=True)
    def test_kirkland_potentials_indexing(self) -> None:
        """Test various indexing operations on kirkland potentials."""
        kp = kirkland_potentials()

        # Test indexing operations through JAX transformations
        def get_single_element():
            return kp[0]

        hydrogen = self.variant(get_single_element)()
        assert hydrogen.shape == (12,)

        def get_multiple_elements():
            return kp[:10]

        first_ten = self.variant(get_multiple_elements)()
        assert first_ten.shape == (10, 12)

        # Fancy indexing (JAX requires array for list indexing)
        def get_selected_elements():
            return kp[jnp.array([0, 5, 78])]  # H, C, Au

        selected = self.variant(get_selected_elements)()
        assert selected.shape == (3, 12)

        # Column slicing
        def get_first_param():
            return kp[:, 0]

        first_param = self.variant(get_first_param)()
        assert first_param.shape == (103,)

    def test_kirkland_potentials_immutability(self) -> None:
        """Test that the returned array reference maintains data integrity."""
        kp1 = kirkland_potentials()
        kp2 = kirkland_potentials()

        # Both calls should return the same object
        assert kp1 is kp2

        # Values should be identical
        assert jnp.array_equal(kp1, kp2)


class TestParseXYZ(chex.TestCase):
    """Test parse_xyz function and XYZ file parsing functionality."""

    def create_temp_xyz_file(self, content: str) -> Path:
        """Helper to create temporary XYZ files for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def test_parse_xyz_basic(self) -> None:
        """Test parsing a basic XYZ file with just atoms."""
        xyz_content = """3
Water molecule
O   0.0000   0.0000   0.0000
H   0.7570   0.5860   0.0000
H  -0.7570   0.5860   0.0000
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check positions shape and values
            assert result.positions.shape == (3, 3)
            assert jnp.allclose(result.positions[0], jnp.array([0.0, 0.0, 0.0]))
            assert jnp.allclose(result.positions[1], jnp.array([0.7570, 0.5860, 0.0]))
            assert jnp.allclose(result.positions[2], jnp.array([-0.7570, 0.5860, 0.0]))

            # Check atomic numbers
            assert result.atomic_numbers.shape == (3,)
            assert result.atomic_numbers[0] == 8  # Oxygen
            assert result.atomic_numbers[1] == 1  # Hydrogen
            assert result.atomic_numbers[2] == 1  # Hydrogen

            # Check comment
            assert result.comment == "Water molecule"

            # Check no optional fields
            assert result.lattice is None
            assert result.stress is None
            assert result.energy is None
            assert result.properties is None
        finally:
            xyz_path.unlink()

    def test_parse_xyz_with_lattice(self) -> None:
        """Test parsing XYZ file with lattice information."""
        xyz_content = """2
        Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0"
        C   0.0   0.0   0.0
        C   2.5   2.5   2.5
        """
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check lattice
            assert result.lattice is not None
            assert result.lattice.shape == (3, 3)
            expected_lattice = jnp.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
            assert jnp.allclose(result.lattice, expected_lattice)

            # Check atoms
            assert result.positions.shape == (2, 3)
            assert jnp.all(result.atomic_numbers == 6)  # Both carbons
        finally:
            xyz_path.unlink()

    def test_parse_xyz_with_stress(self) -> None:
        """Test parsing XYZ file with stress tensor."""
        xyz_content = """1
stress="1.0 0.5 0.3 0.5 2.0 0.1 0.3 0.1 1.5"
Fe   0.0   0.0   0.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check stress tensor
            assert result.stress is not None
            assert result.stress.shape == (3, 3)
            expected_stress = jnp.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.1], [0.3, 0.1, 1.5]])
            assert jnp.allclose(result.stress, expected_stress)

            # Check atom
            assert result.atomic_numbers[0] == 26  # Iron
        finally:
            xyz_path.unlink()

    def test_parse_xyz_with_energy(self) -> None:
        """Test parsing XYZ file with energy."""
        xyz_content = """2
energy=-125.5
H   0.0   0.0   0.0
H   0.74  0.0   0.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check energy
            assert result.energy is not None
            assert jnp.isclose(result.energy, -125.5)

            # Check it's a JAX scalar
            assert result.energy.shape == ()
            assert result.energy.dtype == jnp.float64
        finally:
            xyz_path.unlink()

    def test_parse_xyz_with_properties(self) -> None:
        """Test parsing XYZ file with properties metadata."""
        xyz_content = """2
Properties=species:S:1:pos:R:3:force:R:3
C   0.0   0.0   0.0
N   1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check properties
            assert result.properties is not None
            assert len(result.properties) == 3

            assert result.properties[0]["name"] == "species"
            assert result.properties[0]["type"] == "S"
            assert result.properties[0]["count"] == 1

            assert result.properties[1]["name"] == "pos"
            assert result.properties[1]["type"] == "R"
            assert result.properties[1]["count"] == 3

            assert result.properties[2]["name"] == "force"
            assert result.properties[2]["type"] == "R"
            assert result.properties[2]["count"] == 3
        finally:
            xyz_path.unlink()

    def test_parse_xyz_all_features(self) -> None:
        """Test parsing XYZ file with all optional features."""
        xyz_content = """3
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" stress="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" energy=-100.5 Properties=species:S:1:pos:R:3
C   0.0   0.0   0.0
Si  5.0   5.0   5.0
Ge  2.5   2.5   2.5
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check all components exist
            assert result.lattice is not None
            assert result.stress is not None
            assert result.energy is not None
            assert result.properties is not None

            # Check atoms
            assert result.atomic_numbers[0] == 6  # Carbon
            assert result.atomic_numbers[1] == 14  # Silicon
            assert result.atomic_numbers[2] == 32  # Germanium
        finally:
            xyz_path.unlink()

    def test_parse_xyz_extended_xyz_format(self) -> None:
        """Test parsing extended XYZ format with atom indices."""
        xyz_content = """2
Comment line
1 C   0.0   0.0   0.0
2 N   1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Should handle 5-column format
            assert result.positions.shape == (2, 3)
            assert result.atomic_numbers[0] == 6  # Carbon
            assert result.atomic_numbers[1] == 7  # Nitrogen
        finally:
            xyz_path.unlink()

    def test_parse_xyz_extra_columns(self) -> None:
        """Test parsing XYZ with extra columns (ignored)."""
        xyz_content = """2
Extended format with extra data
C   0.0   0.0   0.0   0.1   0.2   0.3
N   1.0   1.0   1.0   -0.1  -0.2  -0.3
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Should only read first 4 columns
            assert result.positions.shape == (2, 3)
            assert jnp.allclose(result.positions[0], jnp.array([0.0, 0.0, 0.0]))
            assert jnp.allclose(result.positions[1], jnp.array([1.0, 1.0, 1.0]))
        finally:
            xyz_path.unlink()

    def test_parse_xyz_scientific_notation(self) -> None:
        """Test parsing XYZ with scientific notation coordinates."""
        xyz_content = """2
Scientific notation test
H   1.0e-10   2.5e-9   -3.0e-8
He  -1.5e+2   3.7e+1    4.2e+0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            assert jnp.allclose(result.positions[0], jnp.array([1.0e-10, 2.5e-9, -3.0e-8]))
            assert jnp.allclose(result.positions[1], jnp.array([-1.5e2, 3.7e1, 4.2e0]))
        finally:
            xyz_path.unlink()

    def test_parse_xyz_case_sensitive_elements(self) -> None:
        """Test that element symbols are case-sensitive."""
        xyz_content = """3
Mixed case elements
c   0.0   0.0   0.0
CO  1.0   0.0   0.0
co  2.0   0.0   0.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # All should be parsed as Carbon (case-insensitive)
            assert result.atomic_numbers[0] == 6  # c -> C
            assert result.atomic_numbers[1] == 27  # CO -> Co (Cobalt)
            assert result.atomic_numbers[2] == 27  # co -> Co (Cobalt)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_empty_file(self) -> None:
        """Test error handling for empty XYZ file."""
        xyz_content = ""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(ValueError, match="Invalid XYZ file: fewer than 2 lines"):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_invalid_atom_count(self) -> None:
        """Test error handling for invalid atom count."""
        xyz_content = """not_a_number
Comment line
C   0.0   0.0   0.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(ValueError, match="First line must be the number of atoms"):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_too_few_atoms(self) -> None:
        """Test error handling when file has fewer atoms than declared."""
        xyz_content = """5
Declared 5 atoms but only have 2
C   0.0   0.0   0.0
N   1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(ValueError, match="Expected 5 atoms, found only 2"):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_invalid_format(self) -> None:
        """Test error handling for lines with wrong number of columns."""
        xyz_content = """2
Invalid format
C   0.0   0.0
N   1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(ValueError, match="Line 3 has unexpected format"):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_invalid_element(self) -> None:
        """Test error handling for invalid element symbols."""
        xyz_content = """2
Invalid element
C    0.0   0.0   0.0
Xx   1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(KeyError, match="Atomic symbol 'Xx' not found"):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_with_atomic_numbers(self) -> None:
        """Test parsing XYZ file with atomic numbers instead of symbols."""
        xyz_content = """3
Water molecule with atomic numbers
8    0.0000   0.0000   0.0000
1    0.7570   0.5860   0.0000
1   -0.7570   0.5860   0.0000
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check positions shape and values
            assert result.positions.shape == (3, 3)
            assert jnp.allclose(result.positions[0], jnp.array([0.0, 0.0, 0.0]))
            assert jnp.allclose(result.positions[1], jnp.array([0.7570, 0.5860, 0.0]))
            assert jnp.allclose(result.positions[2], jnp.array([-0.7570, 0.5860, 0.0]))

            # Check atomic numbers
            assert result.atomic_numbers.shape == (3,)
            assert result.atomic_numbers[0] == 8  # Oxygen
            assert result.atomic_numbers[1] == 1  # Hydrogen
            assert result.atomic_numbers[2] == 1  # Hydrogen

            # Check comment
            assert result.comment == "Water molecule with atomic numbers"
        finally:
            xyz_path.unlink()

    def test_parse_xyz_mixed_symbols_and_numbers(self) -> None:
        """Test parsing XYZ file with mixed atomic symbols and numbers."""
        xyz_content = """4
Mixed format test
C    0.0   0.0   0.0
14   1.0   1.0   1.0
Fe   2.0   2.0   2.0
79   3.0   3.0   3.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check atomic numbers
            assert result.atomic_numbers.shape == (4,)
            assert result.atomic_numbers[0] == 6  # C -> Carbon
            assert result.atomic_numbers[1] == 14  # 14 -> Silicon
            assert result.atomic_numbers[2] == 26  # Fe -> Iron
            assert result.atomic_numbers[3] == 79  # 79 -> Gold
        finally:
            xyz_path.unlink()

    def test_parse_xyz_large_atomic_numbers(self) -> None:
        """Test parsing XYZ file with large atomic numbers."""
        xyz_content = """3
Heavy elements
92   0.0   0.0   0.0
103  1.0   1.0   1.0
118  2.0   2.0   2.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check atomic numbers
            assert result.atomic_numbers[0] == 92  # Uranium
            assert result.atomic_numbers[1] == 103  # Lawrencium
            assert result.atomic_numbers[2] == 118  # Oganesson
        finally:
            xyz_path.unlink()

    def test_parse_xyz_invalid_atomic_number(self) -> None:
        """Test error handling for invalid atomic numbers."""
        xyz_content = """2
Invalid atomic number
1    0.0   0.0   0.0
999  1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            # Should parse successfully - no validation on atomic number range
            result = parse_xyz(xyz_path)
            assert result.atomic_numbers[0] == 1
            assert result.atomic_numbers[1] == 999
        finally:
            xyz_path.unlink()

    def test_parse_xyz_zero_atomic_number(self) -> None:
        """Test parsing XYZ file with zero as atomic number."""
        xyz_content = """2
Zero test
0    0.0   0.0   0.0
1    1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)
            assert result.atomic_numbers[0] == 0
            assert result.atomic_numbers[1] == 1
        finally:
            xyz_path.unlink()

    def test_parse_xyz_negative_atomic_number(self) -> None:
        """Test parsing XYZ file with negative atomic number."""
        xyz_content = """2
Negative test
-1   0.0   0.0   0.0
1    1.0   1.0   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(ValueError, match="atomic_numbers must be non-negative"):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_invalid_coordinates(self) -> None:
        """Test error handling for non-numeric coordinates."""
        xyz_content = """2
Invalid coordinates
C   0.0   0.0   0.0
N   1.0   NaN   1.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            with pytest.raises(ValueError):
                parse_xyz(xyz_path)
        finally:
            xyz_path.unlink()

    def test_parse_xyz_path_types(self) -> None:
        """Test that parse_xyz accepts both str and Path objects."""
        xyz_content = """1
Path type test
H   0.0   0.0   0.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            # Test with Path object
            result_path = parse_xyz(xyz_path)
            assert result_path.atomic_numbers[0] == 1

            # Test with string
            result_str = parse_xyz(str(xyz_path))
            assert result_str.atomic_numbers[0] == 1

            # Results should be equivalent
            assert jnp.array_equal(result_path.positions, result_str.positions)
            assert jnp.array_equal(result_path.atomic_numbers, result_str.atomic_numbers)
        finally:
            xyz_path.unlink()

    @chex.variants(with_jit=True, without_jit=True)
    def test_parse_xyz_jax_compatible(self) -> None:
        """Test that parsed XYZ data works with JAX transformations."""
        xyz_content = """3
JAX compatibility test
C   0.0   0.0   0.0
N   1.0   0.0   0.0
O   0.0   1.0   0.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Test that we can use results in JAX operations
            def compute_center_of_mass(xyz_data):
                # Simple unweighted center
                return jnp.mean(xyz_data.positions, axis=0)

            com = self.variant(compute_center_of_mass)(result)
            expected_com = jnp.array([1 / 3, 1 / 3, 0.0])
            assert jnp.allclose(com, expected_com)

            # Test vectorized operations on atomic numbers
            def count_heavy_atoms(xyz_data):
                return jnp.sum(xyz_data.atomic_numbers > 1)

            heavy_count = self.variant(count_heavy_atoms)(result)
            assert heavy_count == 3  # All atoms have Z > 1
        finally:
            xyz_path.unlink()

    def test_parse_xyz_dtypes(self) -> None:
        """Test that parsed data has correct dtypes."""
        xyz_content = """2
Dtype test
C   1.0   2.0   3.0
N   4.0   5.0   6.0
"""
        xyz_path = self.create_temp_xyz_file(xyz_content)

        try:
            result = parse_xyz(xyz_path)

            # Check dtypes
            assert result.positions.dtype == jnp.float64
            assert result.atomic_numbers.dtype == jnp.int32

            # Check optional fields when present
            xyz_with_energy = """1
energy=-100.0
C   0.0   0.0   0.0
"""
            xyz_path2 = self.create_temp_xyz_file(xyz_with_energy)
            try:
                result2 = parse_xyz(xyz_path2)
                assert result2.energy.dtype == jnp.float64
            finally:
                xyz_path2.unlink()
        finally:
            xyz_path.unlink()

    def test_parse_xyz_metadata_regex(self) -> None:
        """Test the internal metadata parsing function."""
        # Test lattice parsing
        metadata = _parse_xyz_metadata('Lattice="1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0"')
        assert "lattice" in metadata
        assert metadata["lattice"].shape == (3, 3)
        assert jnp.allclose(metadata["lattice"], jnp.diag(jnp.array([1.0, 2.0, 3.0])))

        # Test stress parsing
        metadata = _parse_xyz_metadata('stress="1.0 0.5 0.3 0.5 2.0 0.1 0.3 0.1 1.5"')
        assert "stress" in metadata
        assert metadata["stress"].shape == (3, 3)

        # Test energy parsing with different formats
        metadata = _parse_xyz_metadata("energy=-125.5")
        assert metadata["energy"] == -125.5

        metadata = _parse_xyz_metadata("energy=1.23e-4")
        assert jnp.isclose(metadata["energy"], 1.23e-4)

        metadata = _parse_xyz_metadata("energy=+3.14E+2")
        assert jnp.isclose(metadata["energy"], 3.14e2)

        # Test properties parsing
        metadata = _parse_xyz_metadata("Properties=species:S:1:pos:R:3")
        assert "properties" in metadata
        assert len(metadata["properties"]) == 2

    def test_parse_xyz_metadata_errors(self) -> None:
        """Test error handling in metadata parsing."""
        # Invalid lattice dimensions
        with pytest.raises(ValueError, match="Lattice must contain 9 values"):
            _parse_xyz_metadata('Lattice="1.0 2.0 3.0"')

        # Invalid stress dimensions
        with pytest.raises(ValueError, match="Stress tensor must contain 9 values"):
            _parse_xyz_metadata('stress="1.0 2.0"')
