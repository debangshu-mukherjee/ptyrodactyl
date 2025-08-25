"""Tests for geometry module - rotation matrices and geometric transformations."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

from ptyrodactyl.electrons.geometry import (
    reciprocal_lattice,
    rotate_structure,
    rotmatrix_axis,
    rotmatrix_vectors,
)


class TestRotmatrixVectors(chex.TestCase):
    """Test suite for rotmatrix_vectors function."""

    def setUp(self) -> None:
        """Set up test vectors."""
        self.v1_x = jnp.array([1.0, 0.0, 0.0])
        self.v1_y = jnp.array([0.0, 1.0, 0.0])
        self.v1_z = jnp.array([0.0, 0.0, 1.0])
        self.v1_arbitrary = jnp.array([1.0, 1.0, 1.0])
        self.v1_normalized = self.v1_arbitrary / jnp.linalg.norm(self.v1_arbitrary)

    def test_identity_rotation(self) -> None:
        """Test rotation of vector to itself yields identity matrix."""
        R = rotmatrix_vectors(self.v1_x, self.v1_x)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-10)

        R = rotmatrix_vectors(self.v1_arbitrary, self.v1_arbitrary)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-10)

    def test_opposite_vectors(self) -> None:
        """Test rotation of vector to its opposite (180 degree rotation)."""
        R = rotmatrix_vectors(self.v1_x, -self.v1_x)
        rotated = R @ self.v1_x
        assert jnp.allclose(rotated, -self.v1_x, atol=1e-10)

        R = rotmatrix_vectors(self.v1_y, -self.v1_y)
        rotated = R @ self.v1_y
        assert jnp.allclose(rotated, -self.v1_y, atol=1e-10)

    def test_orthogonal_rotations(self) -> None:
        """Test rotations between orthogonal unit vectors."""
        R = rotmatrix_vectors(self.v1_x, self.v1_y)
        rotated = R @ self.v1_x
        expected = self.v1_y / jnp.linalg.norm(self.v1_y)
        assert jnp.allclose(rotated, expected, atol=1e-10)

        R = rotmatrix_vectors(self.v1_y, self.v1_z)
        rotated = R @ self.v1_y
        expected = self.v1_z / jnp.linalg.norm(self.v1_z)
        assert jnp.allclose(rotated, expected, atol=1e-10)

    def test_arbitrary_vectors(self) -> None:
        """Test rotation between arbitrary vectors."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, -1.0, 2.0])

        R = rotmatrix_vectors(v1, v2)
        rotated = R @ (v1 / jnp.linalg.norm(v1))
        expected = v2 / jnp.linalg.norm(v2)
        assert jnp.allclose(rotated, expected, atol=1e-08)

    def test_rotation_matrix_properties(self) -> None:
        """Test that rotation matrices are proper (det=1) and orthogonal."""
        test_cases = [
            (self.v1_x, self.v1_y),
            (self.v1_arbitrary, self.v1_z),
            (jnp.array([1.0, 2.0, 3.0]), jnp.array([-1.0, 0.5, 2.0])),
        ]

        for v1, v2 in test_cases:
            R = rotmatrix_vectors(v1, v2)

            self.assertAlmostEqual(jnp.linalg.det(R), 1.0, places=10)

            RTR = R.T @ R
            assert jnp.allclose(RTR, jnp.eye(3), atol=1e-10)

    def test_normalization_invariance(self) -> None:
        """Test that pre-normalized vectors give same result."""
        v1 = jnp.array([2.0, 3.0, 4.0])
        v2 = jnp.array([1.0, -2.0, 1.0])

        R1 = rotmatrix_vectors(v1, v2)
        R2 = rotmatrix_vectors(v1 / jnp.linalg.norm(v1), v2 / jnp.linalg.norm(v2))

        assert jnp.allclose(R1, R2, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self) -> None:
        """Test JAX transformations on rotmatrix_vectors."""

        def compute_rotation(v1, v2):
            return rotmatrix_vectors(v1, v2)

        v1 = jnp.array([1.0, 0.5, 0.2])
        v2 = jnp.array([0.0, 1.0, 0.0])

        R = self.variant(compute_rotation)(v1, v2)
        assert R.shape == (3, 3)
        self.assertAlmostEqual(jnp.linalg.det(R), 1.0, places=10)

    def test_special_cases_numerically_stable(self) -> None:
        """Test numerical stability for nearly parallel/antiparallel vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2_nearly_parallel = jnp.array([1.0, 1e-9, 0.0])
        v2_nearly_antiparallel = jnp.array([-1.0, 1e-9, 0.0])

        R1 = rotmatrix_vectors(v1, v2_nearly_parallel)
        assert not jnp.any(jnp.isnan(R1))
        self.assertAlmostEqual(jnp.linalg.det(R1), 1.0, places=8)

        R2 = rotmatrix_vectors(v1, v2_nearly_antiparallel)
        assert not jnp.any(jnp.isnan(R2))
        self.assertAlmostEqual(jnp.linalg.det(R2), 1.0, places=8)


class TestRotmatrixAxis(chex.TestCase):
    """Test suite for rotmatrix_axis function."""

    def setUp(self) -> None:
        """Set up test axes and angles."""
        self.axis_x = jnp.array([1.0, 0.0, 0.0])
        self.axis_y = jnp.array([0.0, 1.0, 0.0])
        self.axis_z = jnp.array([0.0, 0.0, 1.0])
        self.axis_arbitrary = jnp.array([1.0, 1.0, 1.0])

    def test_zero_rotation(self) -> None:
        """Test that zero angle gives identity matrix."""
        for axis in [self.axis_x, self.axis_y, self.axis_z, self.axis_arbitrary]:
            R = rotmatrix_axis(axis, 0.0)
            assert jnp.allclose(R, jnp.eye(3), atol=1e-10)

    def test_90_degree_rotations(self) -> None:
        """Test 90 degree rotations around coordinate axes."""
        R = rotmatrix_axis(self.axis_z, jnp.pi / 2)
        test_vec = jnp.array([1.0, 0.0, 0.0])
        expected = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(R @ test_vec, expected, atol=1e-10)

        R = rotmatrix_axis(self.axis_x, jnp.pi / 2)
        test_vec = jnp.array([0.0, 1.0, 0.0])
        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(R @ test_vec, expected, atol=1e-10)

    def test_180_degree_rotation(self) -> None:
        """Test 180 degree rotation."""
        R = rotmatrix_axis(self.axis_z, jnp.pi)
        test_vec = jnp.array([1.0, 0.0, 0.0])
        expected = jnp.array([-1.0, 0.0, 0.0])
        assert jnp.allclose(R @ test_vec, expected, atol=1e-10)

    def test_arbitrary_axis_rotation(self) -> None:
        """Test rotation around arbitrary axis."""
        axis = jnp.array([1.0, 1.0, 0.0])
        angle = jnp.pi / 3
        R = rotmatrix_axis(axis, angle)

        self.assertAlmostEqual(jnp.linalg.det(R), 1.0, places=10)

        RTR = R.T @ R
        assert jnp.allclose(RTR, jnp.eye(3), atol=1e-10)

    def test_rodrigues_formula_verification(self) -> None:
        """Verify the Rodrigues rotation formula implementation."""
        axis = jnp.array([0.0, 0.0, 1.0])
        theta = jnp.pi / 4

        R = rotmatrix_axis(axis, theta)

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        expected = jnp.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

        assert jnp.allclose(R, expected, atol=1e-10)

    def test_axis_normalization(self) -> None:
        """Test that unnormalized axes are handled correctly."""
        axis1 = jnp.array([2.0, 0.0, 0.0])
        axis2 = jnp.array([1.0, 0.0, 0.0])
        angle = jnp.pi / 6

        R1 = rotmatrix_axis(axis1, angle)
        R2 = rotmatrix_axis(axis2, angle)

        assert jnp.allclose(R1, R2, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self) -> None:
        """Test JAX transformations on rotmatrix_axis."""

        def compute_rotation(axis, angle):
            return rotmatrix_axis(axis, angle)

        axis = jnp.array([1.0, 1.0, 1.0])
        angle = jnp.pi / 4

        R = self.variant(compute_rotation)(axis, angle)
        assert R.shape == (3, 3)
        self.assertAlmostEqual(jnp.linalg.det(R), 1.0, places=10)

    @parameterized.parameters(
        (jnp.array([1.0, 0.0, 0.0]), 0.0),
        (jnp.array([0.0, 1.0, 0.0]), jnp.pi / 2),
        (jnp.array([0.0, 0.0, 1.0]), jnp.pi),
        (jnp.array([1.0, 1.0, 1.0]), jnp.pi / 3),
        (jnp.array([1.0, 2.0, 3.0]), 2 * jnp.pi / 3),
    )
    def test_rotation_angle_preservation(self, axis, angle) -> None:
        """Test that rotation preserves the angle between vectors."""
        R = rotmatrix_axis(axis, angle)

        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 1.0, 0.0])

        v1_rot = R @ v1
        v2_rot = R @ v2

        cos_angle_before = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
        cos_angle_after = jnp.dot(v1_rot, v2_rot) / (
            jnp.linalg.norm(v1_rot) * jnp.linalg.norm(v2_rot)
        )

        self.assertAlmostEqual(cos_angle_before, cos_angle_after, places=10)


class TestRotateStructure(chex.TestCase):
    """Test suite for rotate_structure function."""

    def setUp(self) -> None:
        """Set up test structure data."""
        self.coords = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.0],
                [3.0, 0.0, 1.0, 0.0],
                [4.0, 0.0, 0.0, 1.0],
            ]
        )
        self.cell = jnp.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        self.R_identity = jnp.eye(3)
        self.R_90z = rotmatrix_axis(jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2)

    def test_identity_rotation(self) -> None:
        """Test that identity rotation preserves structure."""
        rotated_coords, rotated_cell = rotate_structure(self.coords, self.cell, self.R_identity)

        assert jnp.allclose(rotated_coords, self.coords)
        assert jnp.allclose(rotated_cell, self.cell)

    def test_90_degree_rotation(self) -> None:
        """Test 90 degree rotation around z-axis."""
        rotated_coords, rotated_cell = rotate_structure(self.coords, self.cell, self.R_90z)

        self.assertAlmostEqual(rotated_coords[1, 1], 0.0, places=10)
        self.assertAlmostEqual(rotated_coords[1, 2], 1.0, places=10)

        self.assertAlmostEqual(rotated_coords[2, 1], -1.0, places=10)
        self.assertAlmostEqual(rotated_coords[2, 2], 0.0, places=10)

        expected_cell = jnp.array(
            [
                [0.0, 10.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        assert jnp.allclose(rotated_cell, expected_cell, atol=1e-10)

    def test_atom_ids_preserved(self) -> None:
        """Test that atom IDs are preserved during rotation."""
        R = rotmatrix_axis(jnp.array([1.0, 1.0, 1.0]), jnp.pi / 3)
        rotated_coords, _ = rotate_structure(self.coords, self.cell, R)

        assert jnp.allclose(rotated_coords[:, 0], self.coords[:, 0])

    def test_with_inplane_rotation(self) -> None:
        """Test combined rotation with additional in-plane rotation."""
        theta = jnp.pi / 4
        rotated_coords, rotated_cell = rotate_structure(
            self.coords, self.cell, self.R_identity, theta
        )

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        expected_x = self.coords[1, 1] * cos_theta - self.coords[1, 2] * sin_theta
        expected_y = self.coords[1, 1] * sin_theta + self.coords[1, 2] * cos_theta

        self.assertAlmostEqual(rotated_coords[1, 1], expected_x, places=10)
        self.assertAlmostEqual(rotated_coords[1, 2], expected_y, places=10)

    def test_structure_consistency(self) -> None:
        """Test that rotated structure maintains relative positions."""
        R = rotmatrix_axis(jnp.array([1.0, 2.0, 3.0]), jnp.pi / 5)
        rotated_coords, rotated_cell = rotate_structure(self.coords, self.cell, R)

        original_distances = []
        rotated_distances = []

        for i in range(len(self.coords)):
            for j in range(i + 1, len(self.coords)):
                orig_dist = jnp.linalg.norm(self.coords[i, 1:] - self.coords[j, 1:])
                rot_dist = jnp.linalg.norm(rotated_coords[i, 1:] - rotated_coords[j, 1:])
                original_distances.append(orig_dist)
                rotated_distances.append(rot_dist)

        assert jnp.allclose(jnp.array(original_distances), jnp.array(rotated_distances), atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self) -> None:
        """Test JAX transformations on rotate_structure."""

        def apply_rotation(coords, cell, R, theta):
            return rotate_structure(coords, cell, R, theta)

        R = rotmatrix_axis(jnp.array([0.0, 1.0, 0.0]), jnp.pi / 6)
        rotated_coords, rotated_cell = self.variant(apply_rotation)(self.coords, self.cell, R, 0.0)

        assert rotated_coords.shape == self.coords.shape
        assert rotated_cell.shape == self.cell.shape

    def test_combined_rotations(self) -> None:
        """Test that sequential rotations compose correctly."""
        R1 = rotmatrix_axis(jnp.array([1.0, 0.0, 0.0]), jnp.pi / 4)
        R2 = rotmatrix_axis(jnp.array([0.0, 1.0, 0.0]), jnp.pi / 3)

        coords1, cell1 = rotate_structure(self.coords, self.cell, R1)
        coords2, cell2 = rotate_structure(coords1, cell1, R2)

        R_combined = R2 @ R1
        coords_direct, cell_direct = rotate_structure(self.coords, self.cell, R_combined)

        assert jnp.allclose(coords2, coords_direct, atol=1e-10)
        assert jnp.allclose(cell2, cell_direct, atol=1e-10)


class TestReciprocalLattice(chex.TestCase):
    """Test suite for reciprocal_lattice function."""

    def setUp(self) -> None:
        """Set up test lattice structures."""
        self.cubic_cell = jnp.array(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )

        self.orthorhombic_cell = jnp.array(
            [
                [4.0, 0.0, 0.0],
                [0.0, 6.0, 0.0],
                [0.0, 0.0, 8.0],
            ]
        )

        self.hexagonal_cell = jnp.array(
            [
                [3.0, 0.0, 0.0],
                [-1.5, 2.598076211, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )

    def test_cubic_reciprocal(self) -> None:
        """Test reciprocal lattice for cubic system."""
        recip = reciprocal_lattice(self.cubic_cell)

        expected = jnp.array(
            [
                [2 * jnp.pi / 5, 0.0, 0.0],
                [0.0, 2 * jnp.pi / 5, 0.0],
                [0.0, 0.0, 2 * jnp.pi / 5],
            ]
        )

        assert jnp.allclose(recip, expected, atol=1e-10)

    def test_orthorhombic_reciprocal(self) -> None:
        """Test reciprocal lattice for orthorhombic system."""
        recip = reciprocal_lattice(self.orthorhombic_cell)

        expected = jnp.array(
            [
                [2 * jnp.pi / 4, 0.0, 0.0],
                [0.0, 2 * jnp.pi / 6, 0.0],
                [0.0, 0.0, 2 * jnp.pi / 8],
            ]
        )

        assert jnp.allclose(recip, expected, atol=1e-10)

    def test_reciprocal_properties(self) -> None:
        """Test fundamental properties of reciprocal lattice."""
        cells = [self.cubic_cell, self.orthorhombic_cell, self.hexagonal_cell]

        for cell in cells:
            recip = reciprocal_lattice(cell)

            product = cell @ recip.T
            expected = 2 * jnp.pi * jnp.eye(3)
            assert jnp.allclose(product, expected, atol=1e-10)

    def test_double_reciprocal(self) -> None:
        """Test that double reciprocal gives back the original cell."""
        cells = [self.cubic_cell, self.orthorhombic_cell, self.hexagonal_cell]

        for cell in cells:
            recip1 = reciprocal_lattice(cell)
            recip2 = reciprocal_lattice(recip1)

            # The double reciprocal should give back the original cell
            assert jnp.allclose(recip2, cell, atol=1e-10)

    def test_volume_relationship(self) -> None:
        """Test volume relationship between real and reciprocal lattices."""
        cells = [self.cubic_cell, self.orthorhombic_cell, self.hexagonal_cell]

        for cell in cells:
            V_real = jnp.linalg.det(cell)
            recip = reciprocal_lattice(cell)
            V_recip = jnp.linalg.det(recip)

            expected_V_recip = (2 * jnp.pi) ** 3 / V_real
            self.assertAlmostEqual(V_recip, expected_V_recip, places=10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_jax_transformations(self) -> None:
        """Test JAX transformations on reciprocal_lattice."""

        def compute_reciprocal(cell):
            return reciprocal_lattice(cell)

        recip = self.variant(compute_reciprocal)(self.cubic_cell)
        assert recip.shape == (3, 3)

        product = self.cubic_cell @ recip.T
        expected = 2 * jnp.pi * jnp.eye(3)
        assert jnp.allclose(product, expected, atol=1e-10)

    def test_triclinic_lattice(self) -> None:
        """Test reciprocal lattice for general triclinic system."""
        triclinic_cell = jnp.array(
            [
                [5.0, 0.5, 0.3],
                [0.2, 6.0, 0.4],
                [0.1, 0.3, 7.0],
            ]
        )

        recip = reciprocal_lattice(triclinic_cell)

        product = triclinic_cell @ recip.T
        expected = 2 * jnp.pi * jnp.eye(3)
        assert jnp.allclose(product, expected, atol=1e-10)

    @parameterized.parameters(
        (0.1,),
        (1.0,),
        (10.0,),
        (100.0,),
    )
    def test_scaling_invariance(self, scale) -> None:
        """Test that scaling real lattice inversely scales reciprocal lattice."""
        scaled_cell = self.cubic_cell * scale
        recip_scaled = reciprocal_lattice(scaled_cell)

        recip_original = reciprocal_lattice(self.cubic_cell)
        expected = recip_original / scale

        assert jnp.allclose(recip_scaled, expected, atol=1e-10)


class TestGeometryIntegration(chex.TestCase):
    """Integration tests combining multiple geometry functions."""

    def test_rotation_matrices_compatibility(self) -> None:
        """Test that rotmatrix_vectors and rotmatrix_axis produce compatible results."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 1.0, 0.0])

        R_vectors = rotmatrix_vectors(v1, v2)

        axis = jnp.cross(v1, v2)
        angle = jnp.arccos(jnp.dot(v1, v2))
        R_axis = rotmatrix_axis(axis, angle)

        assert jnp.allclose(R_vectors, R_axis, atol=1e-10)

    def test_structure_rotation_preservation(self) -> None:
        """Test that rotations preserve crystal structure properties."""
        coords = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.5, 0.5, 0.5],
            ]
        )
        cell = jnp.array(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )

        R = rotmatrix_axis(jnp.array([1.0, 1.0, 1.0]), jnp.pi / 3)
        rotated_coords, rotated_cell = rotate_structure(coords, cell, R)

        original_volume = jnp.linalg.det(cell)
        rotated_volume = jnp.linalg.det(rotated_cell)
        self.assertAlmostEqual(original_volume, rotated_volume, places=10)

        original_recip = reciprocal_lattice(cell)
        rotated_recip = reciprocal_lattice(rotated_cell)
        expected_rotated_recip = original_recip @ R.T
        assert jnp.allclose(rotated_recip, expected_rotated_recip, atol=1e-10)

    def test_full_workflow(self) -> None:
        """Test complete workflow: create structure, rotate, compute reciprocal."""
        cell = jnp.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        )
        coords = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.5, 0.5, 0.0],
                [3.0, 0.5, 0.0, 0.5],
                [4.0, 0.0, 0.5, 0.5],
            ]
        )

        R = rotmatrix_vectors(jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 1.0]))

        rotated_coords, rotated_cell = rotate_structure(coords, cell, R, jnp.pi / 6)

        recip_original = reciprocal_lattice(cell)
        recip_rotated = reciprocal_lattice(rotated_cell)

        assert rotated_coords.shape == coords.shape
        assert rotated_cell.shape == cell.shape
        assert recip_rotated.shape == recip_original.shape

        self.assertAlmostEqual(jnp.linalg.det(rotated_cell), jnp.linalg.det(cell), places=10)


if __name__ == "__main__":
    chex.TestCase.main()
