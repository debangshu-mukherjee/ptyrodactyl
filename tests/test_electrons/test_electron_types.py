"""Tests for electron_types module - PyTrees and factory functions."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.electrons.electron_types import (
    CalibratedArray,
    CrystalStructure,
    PotentialSlices,
    ProbeModes,
    XYZData,
    make_calibrated_array,
    make_crystal_structure,
    make_potential_slices,
    make_probe_modes,
    make_xyz_data,
)


class TestCalibratedArray(chex.TestCase):
    """Test suite for CalibratedArray PyTree and factory function."""
    
    def setUp(self):
        """Set up test data."""
        self.h, self.w = 64, 64
        self.int_data = jnp.ones((self.h, self.w), dtype=jnp.int32)
        self.float_data = jnp.ones((self.h, self.w), dtype=jnp.float64)
        self.complex_data = jnp.ones((self.h, self.w), dtype=jnp.complex128)
        self.calib_y = 0.1
        self.calib_x = 0.2
        self.real_space = True
    
    def test_factory_function_valid_int(self):
        """Test factory function with valid integer data."""
        arr = make_calibrated_array(
            self.int_data, self.calib_y, self.calib_x, self.real_space
        )
        self.assertIsInstance(arr, CalibratedArray)
        self.assertEqual(arr.data_array.shape, (self.h, self.w))
        self.assertEqual(arr.data_array.dtype, jnp.int32)
        self.assertAlmostEqual(float(arr.calib_y), 0.1)
        self.assertAlmostEqual(float(arr.calib_x), 0.2)
        self.assertTrue(arr.real_space)
    
    def test_factory_function_valid_float(self):
        """Test factory function with valid float data."""
        arr = make_calibrated_array(
            self.float_data, self.calib_y, self.calib_x, self.real_space
        )
        self.assertIsInstance(arr, CalibratedArray)
        self.assertEqual(arr.data_array.dtype, jnp.float64)
    
    def test_factory_function_valid_complex(self):
        """Test factory function with valid complex data."""
        arr = make_calibrated_array(
            self.complex_data, self.calib_y, self.calib_x, self.real_space
        )
        self.assertIsInstance(arr, CalibratedArray)
        self.assertEqual(arr.data_array.dtype, jnp.complex128)
    
    def test_pytree_flatten_unflatten(self):
        """Test PyTree flatten and unflatten operations."""
        arr = make_calibrated_array(
            self.float_data, self.calib_y, self.calib_x, self.real_space
        )
        
        leaves, treedef = jax.tree_util.tree_flatten(arr)
        self.assertEqual(len(leaves), 4)
        
        arr_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        chex.assert_trees_all_equal(arr, arr_reconstructed)
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self):
        """Test JAX transformations on CalibratedArray."""
        def process_array(arr):
            return CalibratedArray(
                data_array=arr.data_array * 2,
                calib_y=arr.calib_y,
                calib_x=arr.calib_x,
                real_space=arr.real_space,
            )
        
        arr = make_calibrated_array(
            self.float_data, self.calib_y, self.calib_x, self.real_space
        )
        
        processed = self.variant(process_array)(arr)
        self.assertTrue(jnp.allclose(processed.data_array, self.float_data * 2))
    
    def test_invalid_inputs(self):
        """Test factory function with invalid inputs."""
        with self.assertRaises(Exception):
            make_calibrated_array(
                jnp.ones(10), self.calib_y, self.calib_x, self.real_space
            )
    
    def test_jax_compliant_adjustments(self):
        """Test JAX-compliant adjustments for invalid values."""
        arr = make_calibrated_array(
            self.float_data, -0.1, -0.2, self.real_space
        )
        self.assertTrue(arr.calib_y > 0)
        self.assertTrue(arr.calib_x > 0)
        
        arr = make_calibrated_array(
            self.float_data, jnp.nan, self.calib_x, self.real_space
        )
        self.assertTrue(jnp.isnan(arr.calib_y))


class TestProbeModes(chex.TestCase):
    """Test suite for ProbeModes PyTree and factory function."""
    
    def setUp(self):
        """Set up test data."""
        self.h, self.w, self.m = 64, 64, 3
        self.modes = jnp.ones((self.h, self.w, self.m), dtype=jnp.complex128)
        self.weights = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64)
        self.calib = 0.1
    
    def test_factory_function_valid(self):
        """Test factory function with valid data."""
        probe = make_probe_modes(self.modes, self.weights, self.calib)
        self.assertIsInstance(probe, ProbeModes)
        self.assertEqual(probe.modes.shape, (self.h, self.w, self.m))
        self.assertEqual(probe.modes.dtype, jnp.complex128)
        self.assertEqual(probe.weights.shape, (self.m,))
        self.assertAlmostEqual(float(probe.calib), 0.1)
    
    def test_pytree_flatten_unflatten(self):
        """Test PyTree flatten and unflatten operations."""
        probe = make_probe_modes(self.modes, self.weights, self.calib)
        
        leaves, treedef = jax.tree_util.tree_flatten(probe)
        self.assertEqual(len(leaves), 3)
        
        probe_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        chex.assert_trees_all_equal(probe, probe_reconstructed)
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self):
        """Test JAX transformations on ProbeModes."""
        def normalize_weights(probe):
            return ProbeModes(
                modes=probe.modes,
                weights=probe.weights / jnp.sum(probe.weights),
                calib=probe.calib,
            )
        
        probe = make_probe_modes(self.modes, self.weights, self.calib)
        normalized = self.variant(normalize_weights)(probe)
        self.assertAlmostEqual(jnp.sum(normalized.weights), 1.0, places=6)
    
    def test_invalid_inputs(self):
        """Test factory function with invalid inputs."""
        with self.assertRaises(Exception):
            make_probe_modes(jnp.ones((10, 10)), self.weights, self.calib)
        
        with self.assertRaises(Exception):
            make_probe_modes(self.modes, jnp.array([0.5, 0.5]), self.calib)
    
    def test_jax_compliant_adjustments(self):
        """Test JAX-compliant adjustments for invalid values."""
        probe = make_probe_modes(self.modes, jnp.array([-0.5, -0.3, -0.2]), self.calib)
        self.assertTrue(jnp.all(probe.weights >= 0))
        self.assertAlmostEqual(jnp.sum(probe.weights), 1.0, places=6)
        
        probe = make_probe_modes(self.modes, jnp.zeros(self.m), self.calib)
        self.assertTrue(jnp.all(probe.weights >= 0))
        self.assertAlmostEqual(jnp.sum(probe.weights), 1.0, places=6)
        
        probe = make_probe_modes(self.modes, self.weights, -0.1)
        self.assertTrue(probe.calib > 0)


class TestPotentialSlices(chex.TestCase):
    """Test suite for PotentialSlices PyTree and factory function."""
    
    def setUp(self):
        """Set up test data."""
        self.h, self.w, self.s = 64, 64, 10
        self.slices = jnp.ones((self.h, self.w, self.s), dtype=jnp.complex128)
        self.slice_thickness = 2.0
        self.calib = 0.1
    
    def test_factory_function_valid(self):
        """Test factory function with valid data."""
        pot = make_potential_slices(self.slices, self.slice_thickness, self.calib)
        self.assertIsInstance(pot, PotentialSlices)
        self.assertEqual(pot.slices.shape, (self.h, self.w, self.s))
        self.assertEqual(pot.slices.dtype, jnp.complex128)
        self.assertAlmostEqual(float(pot.slice_thickness), 2.0)
        self.assertAlmostEqual(float(pot.calib), 0.1)
    
    def test_pytree_flatten_unflatten(self):
        """Test PyTree flatten and unflatten operations."""
        pot = make_potential_slices(self.slices, self.slice_thickness, self.calib)
        
        leaves, treedef = jax.tree_util.tree_flatten(pot)
        self.assertEqual(len(leaves), 3)
        
        pot_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        chex.assert_trees_all_equal(pot, pot_reconstructed)
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self):
        """Test JAX transformations on PotentialSlices."""
        def scale_slices(pot, scale):
            return PotentialSlices(
                slices=pot.slices * scale,
                slice_thickness=pot.slice_thickness,
                calib=pot.calib,
            )
        
        pot = make_potential_slices(self.slices, self.slice_thickness, self.calib)
        scaled = self.variant(lambda p: scale_slices(p, 2.0))(pot)
        self.assertTrue(jnp.allclose(scaled.slices, self.slices * 2.0))
    
    def test_invalid_inputs(self):
        """Test factory function with invalid inputs."""
        with self.assertRaises(Exception):
            make_potential_slices(jnp.ones((10, 10)), self.slice_thickness, self.calib)
    
    def test_jax_compliant_adjustments(self):
        """Test JAX-compliant adjustments for invalid values."""
        pot = make_potential_slices(self.slices, -1.0, self.calib)
        self.assertTrue(pot.slice_thickness > 0)
        
        pot = make_potential_slices(self.slices, self.slice_thickness, -0.1)
        self.assertTrue(pot.calib > 0)


class TestCrystalStructure(chex.TestCase):
    """Test suite for CrystalStructure PyTree and factory function."""
    
    def setUp(self):
        """Set up test data."""
        self.n_atoms = 4
        self.frac_positions = jnp.array([
            [0.0, 0.0, 0.0, 6],
            [0.25, 0.25, 0.25, 6],
            [0.5, 0.5, 0.0, 6],
            [0.75, 0.75, 0.25, 6],
        ], dtype=jnp.float64)
        self.cart_positions = self.frac_positions.copy()
        self.cart_positions = self.cart_positions.at[:, :3].multiply(5.0)
        self.cell_lengths = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float64)
        self.cell_angles = jnp.array([90.0, 90.0, 90.0], dtype=jnp.float64)
    
    def test_factory_function_valid(self):
        """Test factory function with valid data."""
        crystal = make_crystal_structure(
            self.frac_positions, self.cart_positions, 
            self.cell_lengths, self.cell_angles
        )
        self.assertIsInstance(crystal, CrystalStructure)
        self.assertEqual(crystal.frac_positions.shape, (self.n_atoms, 4))
        self.assertEqual(crystal.cart_positions.shape, (self.n_atoms, 4))
        self.assertEqual(crystal.cell_lengths.shape, (3,))
        self.assertEqual(crystal.cell_angles.shape, (3,))
    
    def test_pytree_flatten_unflatten(self):
        """Test PyTree flatten and unflatten operations."""
        crystal = make_crystal_structure(
            self.frac_positions, self.cart_positions,
            self.cell_lengths, self.cell_angles
        )
        
        leaves, treedef = jax.tree_util.tree_flatten(crystal)
        self.assertEqual(len(leaves), 4)
        
        crystal_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        chex.assert_trees_all_equal(crystal, crystal_reconstructed)
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self):
        """Test JAX transformations on CrystalStructure."""
        def translate_crystal(crystal, translation):
            return CrystalStructure(
                frac_positions=crystal.frac_positions,
                cart_positions=crystal.cart_positions.at[:, :3].add(translation),
                cell_lengths=crystal.cell_lengths,
                cell_angles=crystal.cell_angles,
            )
        
        crystal = make_crystal_structure(
            self.frac_positions, self.cart_positions,
            self.cell_lengths, self.cell_angles
        )
        translation = jnp.array([1.0, 0.0, 0.0])
        translated = self.variant(lambda c: translate_crystal(c, translation))(crystal)
        
        expected_cart = self.cart_positions.copy()
        expected_cart = expected_cart.at[:, 0].add(1.0)
        self.assertTrue(jnp.allclose(translated.cart_positions, expected_cart))
    
    def test_invalid_inputs(self):
        """Test factory function with invalid inputs."""
        with self.assertRaises(Exception):
            make_crystal_structure(
                jnp.ones((4, 3)), self.cart_positions,
                self.cell_lengths, self.cell_angles
            )
    
    def test_jax_compliant_adjustments(self):
        """Test JAX-compliant adjustments for invalid values."""
        crystal = make_crystal_structure(
            self.frac_positions, self.cart_positions,
            jnp.array([-5.0, -5.0, -5.0]), self.cell_angles
        )
        self.assertTrue(jnp.all(crystal.cell_lengths > 0))
        
        crystal = make_crystal_structure(
            self.frac_positions, self.cart_positions,
            self.cell_lengths, jnp.array([-10.0, 90.0, 190.0])
        )
        self.assertTrue(jnp.all(crystal.cell_angles > 0))
        self.assertTrue(jnp.all(crystal.cell_angles < 180))


class TestXYZData(chex.TestCase):
    """Test suite for XYZData PyTree and factory function."""
    
    def setUp(self):
        """Set up test data."""
        self.n_atoms = 3
        self.positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=jnp.float64)
        self.atomic_numbers = jnp.array([1, 1, 8], dtype=jnp.int32)
        self.lattice = jnp.eye(3, dtype=jnp.float64) * 10.0
        self.stress = jnp.eye(3, dtype=jnp.float64) * 0.1
        self.energy = -76.4
        self.properties = [{"species": "H"}, {"species": "H"}, {"species": "O"}]
        self.comment = "Water molecule"
    
    def test_factory_function_minimal(self):
        """Test factory function with minimal required data."""
        xyz = make_xyz_data(self.positions, self.atomic_numbers)
        self.assertIsInstance(xyz, XYZData)
        self.assertEqual(xyz.positions.shape, (self.n_atoms, 3))
        self.assertEqual(xyz.atomic_numbers.shape, (self.n_atoms,))
        self.assertIsNone(xyz.lattice)
        self.assertIsNone(xyz.stress)
        self.assertIsNone(xyz.energy)
        self.assertIsNone(xyz.properties)
        self.assertIsNone(xyz.comment)
    
    def test_factory_function_full(self):
        """Test factory function with all optional data."""
        xyz = make_xyz_data(
            self.positions, self.atomic_numbers,
            lattice=self.lattice, stress=self.stress,
            energy=self.energy, properties=self.properties,
            comment=self.comment
        )
        self.assertIsInstance(xyz, XYZData)
        self.assertEqual(xyz.lattice.shape, (3, 3))
        self.assertEqual(xyz.stress.shape, (3, 3))
        self.assertAlmostEqual(float(xyz.energy), -76.4)
        self.assertEqual(xyz.properties, self.properties)
        self.assertEqual(xyz.comment, self.comment)
    
    def test_pytree_flatten_unflatten(self):
        """Test PyTree flatten and unflatten operations."""
        xyz = make_xyz_data(
            self.positions, self.atomic_numbers,
            lattice=self.lattice, stress=self.stress,
            energy=self.energy, properties=self.properties,
            comment=self.comment
        )
        
        leaves, treedef = jax.tree_util.tree_flatten(xyz)
        self.assertEqual(len(leaves), 5)
        
        xyz_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        chex.assert_trees_all_equal(xyz, xyz_reconstructed)
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self):
        """Test JAX transformations on XYZData."""
        def center_positions(xyz):
            center = jnp.mean(xyz.positions, axis=0)
            return XYZData(
                positions=xyz.positions - center,
                atomic_numbers=xyz.atomic_numbers,
                lattice=xyz.lattice,
                stress=xyz.stress,
                energy=xyz.energy,
                properties=xyz.properties,
                comment=xyz.comment,
            )
        
        xyz = make_xyz_data(
            self.positions, self.atomic_numbers,
            lattice=self.lattice, stress=self.stress,
            energy=self.energy
        )
        centered = self.variant(center_positions)(xyz)
        
        com = jnp.mean(centered.positions, axis=0)
        self.assertTrue(jnp.allclose(com, 0.0, atol=1e-10))
    
    def test_invalid_inputs(self):
        """Test factory function with invalid inputs."""
        with self.assertRaises(Exception):
            make_xyz_data(jnp.ones((3, 2)), self.atomic_numbers)
        
        with self.assertRaises(Exception):
            make_xyz_data(self.positions, jnp.array([1, 1]))
        
        with self.assertRaises(ValueError):
            pos_with_nan = self.positions.at[0, 0].set(jnp.nan)
            make_xyz_data(pos_with_nan, self.atomic_numbers)
        
        with self.assertRaises(ValueError):
            make_xyz_data(self.positions, jnp.array([-1, 1, 8]))
        
        with self.assertRaises(Exception):
            make_xyz_data(
                self.positions, self.atomic_numbers,
                lattice=jnp.ones((2, 3))
            )


class TestJAXCompliance(chex.TestCase):
    """Test JAX compliance of factory functions."""
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_factory_functions_jittable(self):
        """Test that factory functions can be JIT-compiled."""
        jitted_calibrated = self.variant(make_calibrated_array)
        arr = jitted_calibrated(jnp.ones((5, 5)), 0.1, 0.2, True)
        self.assertEqual(arr.data_array.shape, (5, 5))
        
        jitted_probe = self.variant(make_probe_modes)
        probe = jitted_probe(
            jnp.ones((5, 5, 2), dtype=jnp.complex128),
            jnp.array([0.5, 0.5]),
            0.1
        )
        self.assertEqual(probe.modes.shape, (5, 5, 2))
        
        jitted_potential = self.variant(make_potential_slices)
        pot = jitted_potential(
            jnp.ones((5, 5, 3), dtype=jnp.complex128),
            1.0,
            0.1
        )
        self.assertEqual(pot.slices.shape, (5, 5, 3))
        
        jitted_crystal = self.variant(make_crystal_structure)
        crystal = jitted_crystal(
            jnp.ones((4, 4)),
            jnp.ones((4, 4)),
            jnp.array([5.0, 5.0, 5.0]),
            jnp.array([90.0, 90.0, 90.0])
        )
        self.assertEqual(crystal.frac_positions.shape, (4, 4))
    
    def test_factory_functions_vmappable(self):
        """Test that factory functions can be vmapped over batch dimensions."""
        batch_calib_y = jnp.array([0.1, 0.2, 0.3])
        batch_calib_x = jnp.array([0.15, 0.25, 0.35])
        
        vmapped_fn = jax.vmap(
            lambda cy, cx: make_calibrated_array(jnp.ones((5, 5)), cy, cx, True)
        )
        
        arrays = vmapped_fn(batch_calib_y, batch_calib_x)
        self.assertEqual(arrays.calib_y.shape, (3,))
        self.assertEqual(arrays.calib_x.shape, (3,))
        self.assertTrue(jnp.allclose(arrays.calib_y, batch_calib_y))


class TestPyTreeOperations(chex.TestCase):
    """Test PyTree operations across different data structures."""
    
    def test_tree_map_on_calibrated_array(self):
        """Test jax.tree_map on CalibratedArray."""
        arr1 = make_calibrated_array(
            jnp.ones((10, 10)), 0.1, 0.2, True
        )
        arr2 = make_calibrated_array(
            jnp.ones((10, 10)) * 2, 0.1, 0.2, True
        )
        
        def add_arrays(a1, a2):
            if isinstance(a1, jnp.ndarray) and isinstance(a2, jnp.ndarray):
                return a1 + a2
            return a1
        
        result = jax.tree_map(add_arrays, arr1, arr2)
        self.assertTrue(jnp.allclose(result.data_array, 3.0))
    
    def test_tree_leaves_on_probe_modes(self):
        """Test jax.tree_leaves on ProbeModes."""
        probe = make_probe_modes(
            jnp.ones((10, 10, 2), dtype=jnp.complex128),
            jnp.array([0.6, 0.4]),
            0.1
        )
        
        leaves = jax.tree_leaves(probe)
        self.assertEqual(len(leaves), 3)
        self.assertEqual(leaves[0].shape, (10, 10, 2))
        self.assertEqual(leaves[1].shape, (2,))
        self.assertEqual(leaves[2].shape, ())
    
    @parameterized.parameters(
        (CalibratedArray, make_calibrated_array, 
         (jnp.ones((5, 5)), 0.1, 0.2, True)),
        (ProbeModes, make_probe_modes,
         (jnp.ones((5, 5, 2), dtype=jnp.complex128), jnp.array([0.5, 0.5]), 0.1)),
        (PotentialSlices, make_potential_slices,
         (jnp.ones((5, 5, 3), dtype=jnp.complex128), 1.0, 0.1)),
    )
    def test_pytree_registration(self, cls, factory_fn, args):
        """Test that PyTree registration works correctly."""
        instance = factory_fn(*args)
        
        self.assertTrue(jax.tree_util.tree_map(lambda x: x, instance) is not None)
        
        leaves, treedef = jax.tree_util.tree_flatten(instance)
        self.assertIsInstance(leaves, list)
        self.assertTrue(all(isinstance(leaf, jnp.ndarray) for leaf in leaves))


if __name__ == "__main__":
    chex.TestCase.main()