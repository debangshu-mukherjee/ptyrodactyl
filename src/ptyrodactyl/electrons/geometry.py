"""
Module: ptyrodactyl.electrons.geometry
--------------------------------------
Geometric transformations and operations for crystal structures.

Functions
---------
- `rotmatrix_vectors`:
    Compute a rotation matrix that rotates one vector to align with another
- `rotmatrix_axis`:
    Generate a rotation matrix for rotation around an arbitrary axis
- `rotate_structure`:
    Apply rotation transformations to crystal structures
- `reciprocal_lattice`:
    Compute reciprocal lattice vectors from real-space unit cell
"""

import jax
import jax.numpy as jnp
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Float, Real

from ptyrodactyl._decorators import beartype, jaxtyped

from .electron_types import scalar_float, scalar_numeric


@jaxtyped(typechecker=beartype)
def rotmatrix_vectors(
    v1: Real[Array, " 3"], v2: Real[Array, " 3"]
) -> Float[Array, "3 3"]:
    """Compute a proper rotation matrix that rotates vector v1 to align with vector v2.

    Parameters
    ----------
    v1 : Real[Array, " 3"]
        Initial 3D vector to be rotated
    v2 : Real[Array, " 3"]
        Target 3D vector that v1 should be rotated to align with

    Returns
    -------
    Float[Array, "3 3"]
        3x3 rotation matrix such that rotation_matrix @ v1 is parallel to v2

    Notes
    -----
    Uses the Rodrigues rotation formula. Handles special cases where vectors are
    parallel or anti-parallel.

    Algorithm:
    ---------
    - Normalize input vectors:
        - Divide v1 and v2 by their respective norms to get unit vectors
        - This ensures the rotation is purely rotational without scaling
    - Calculate rotation parameters:
        - Compute cross product: cross = v1 * v2 (gives rotation axis direction)
        - Compute dot product: dot = v1 · v2 (gives cosine of rotation angle)
        - Calculate sin(θ) as the norm of the cross product
    - Handle special cases:
        - Check if vectors are nearly parallel (sin_theta < 1e-8)
        - Check if vectors are nearly opposite (dot < -0.9999)
    - Define fallback functions for special cases:
        - fallback_parallel(): Returns identity matrix when vectors are already aligned
        - fallback_opposite(): Handles 180° rotation case
            - Choose an orthogonal axis to v1 (prefer x-axis unless v1 is nearly along x)
            - Compute rotation axis as cross product of v1 and the orthogonal vector
            - Use double application of skew-symmetric matrix for 180° rotation
    - Compute general rotation matrix using Rodrigues formula:
        - Normalize cross product to get unit rotation axis
        - Construct skew-symmetric matrix K from rotation axis components
        - Apply Rodrigues formula: rotation_matrix = I + sin(θ)K + (1-cos(θ))K²
    - Use conditional logic to select appropriate computation:
        - If vectors are parallel, check if they're opposite or same direction
        - Return appropriate rotation matrix based on the case
    """
    v1: Float[Array, " 3"] = v1 / jnp.linalg.norm(v1)
    v2: Float[Array, " 3"] = v2 / jnp.linalg.norm(v2)
    cross: Float[Array, " 3"] = jnp.cross(v1, v2)
    dot: Float[Array, " "] = jnp.dot(v1, v2)
    sin_theta: Float[Array, " "] = jnp.linalg.norm(cross)

    def _fallback_parallel() -> Float[Array, "3 3"]:
        rotation_matrix_parallel: Float[Array, "3 3"] = jnp.eye(3)
        return rotation_matrix_parallel

    def _fallback_opposite() -> Float[Array, "3 3"]:
        """
        Description
        -----------
        Compute a rotation matrix for a 180° rotation around an arbitrary axis.
        This handles the case where the vectors are nearly opposite.
        """
        magic_number: scalar_float = 0.9
        ortho: Float[Array, " 3"] = jnp.where(
            jnp.abs(v1[0]) < magic_number,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
        )
        axis: Float[Array, " 3"] = jnp.cross(v1, ortho)
        axis: Float[Array, " 3"] = axis / jnp.linalg.norm(axis)
        kk: Float[Array, "3 3"] = jnp.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        rotation_matrix_opposite: Float[Array, "3 3"] = jnp.eye(3) + 2 * kk @ kk
        return rotation_matrix_opposite

    def _compute() -> Float[Array, "3 3"]:
        axis: Float[Array, " 3"] = cross / sin_theta
        kk: Float[Array, "3 3"] = jnp.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        rotation_matrix_general: Float[Array, "3 3"] = (
            jnp.eye(3) + sin_theta * kk + (1 - dot) * (kk @ kk)
        )
        return rotation_matrix_general

    close_to_zero: scalar_float = 1e-8
    almost_parallel: Bool[Array, " "] = sin_theta < close_to_zero
    close_to_one: scalar_float = 0.999999
    almost_opposite: Bool[Array, " "] = dot < -close_to_one
    rotation_matrix: Float[Array, "3 3"] = jax.lax.cond(
        almost_parallel,
        lambda: jax.lax.cond(almost_opposite, _fallback_opposite, _fallback_parallel),
        _compute,
    )
    return rotation_matrix


@jaxtyped(typechecker=beartype)
def rotmatrix_axis(
    axis: Real[Array, " 3"], theta: scalar_numeric
) -> Float[Array, "3 3"]:
    """Generate a 3D rotation matrix for rotation around an arbitrary axis by a specified angle.

    Parameters
    ----------
    axis : Real[Array, " 3"]
        3D vector defining the axis of rotation (will be normalized)
    theta : scalar_numeric
        Rotation angle in radians (positive for counter-clockwise rotation
        when looking along the axis)

    Returns
    -------
    Float[Array, "3 3"]
        3x3 rotation matrix that rotates vectors by theta radians around the axis

    Notes
    -----
    Uses the Rodrigues rotation formula. This creates a right-handed rotation
    when looking along the axis direction.

    Algorithm:
    ---------
    - Normalize the rotation axis:
        - Divide axis vector by its norm to ensure unit length
        - This guarantees the rotation matrix is orthogonal
    - Calculate trigonometric values:
        - Compute cos(theta) for diagonal and off-diagonal terms
        - Compute sin(theta) for antisymmetric components
    - Extract axis components:
        - Unpack normalized axis into components (ux, uy, uz)
        - These will be used to construct the rotation matrix
    - Build rotation matrix using Rodrigues formula:
        - The formula is: rotation_matrix = I*cos(θ) + (1-cos(θ))*n⊗n + sin(θ)*[n]×
        - Where n is the unit axis vector and [n]× is the skew-symmetric matrix
    - Matrix construction details:
        - Diagonal terms: cos(θ) + n_i² * (1 - cos(θ))
        - Off-diagonal symmetric part: n_i * n_j * (1 - cos(θ))
        - Off-diagonal antisymmetric part: ±n_k * sin(θ) (follows right-hand rule)
    - Explicit matrix elements:
        - rotation_matrix[0,0] = cos(θ) + ux² * (1 - cos(θ))
        - rotation_matrix[0,1] = ux * uy * (1 - cos(θ)) - uz * sin(θ)
        - rotation_matrix[0,2] = ux * uz * (1 - cos(θ)) + uy * sin(θ)
        - And similarly for other rows following the pattern
    - Return the constructed 3x3 rotation matrix
    """
    axis: Float[Array, " 3"] = axis / jnp.linalg.norm(axis)
    cos_theta: Float[Array, " "] = jnp.cos(theta)
    sin_theta: Float[Array, " "] = jnp.sin(theta)
    ux: Float[Array, " "]
    uy: Float[Array, " "]
    uz: Float[Array, " "]
    ux, uy, uz = axis
    rot_matrix: Float[Array, "3 3"] = jnp.array(
        [
            [
                cos_theta + ux**2 * (1 - cos_theta),
                ux * uy * (1 - cos_theta) - uz * sin_theta,
                ux * uz * (1 - cos_theta) + uy * sin_theta,
            ],
            [
                uy * ux * (1 - cos_theta) + uz * sin_theta,
                cos_theta + uy**2 * (1 - cos_theta),
                uy * uz * (1 - cos_theta) - ux * sin_theta,
            ],
            [
                uz * ux * (1 - cos_theta) - uy * sin_theta,
                uz * uy * (1 - cos_theta) + ux * sin_theta,
                cos_theta + uz**2 * (1 - cos_theta),
            ],
        ]
    )
    return rot_matrix


@jaxtyped(typechecker=beartype)
def rotate_structure(
    coords: Real[Array, " N 4"],
    cell: Real[Array, "3 3"],
    rotation_matrix: Real[Array, "3 3"],
    theta: Optional[scalar_numeric] = 0,
) -> Tuple[Float[Array, " N 4"], Float[Array, "3 3"]]:
    """Apply rotation transformations to a crystal structure.

    Parameters
    ----------
    coords : Real[Array, " N 4"]
        Atomic coordinates array where each row contains [atom_id, x, y, z].
        First column is the atom identifier, remaining columns are 3D positions
    cell : Real[Array, "3 3"]
        Unit cell matrix where rows represent the three lattice vectors a, b, c
    rotation_matrix : Real[Array, "3 3"]
        Primary rotation matrix to apply to the structure
    theta : scalar_numeric, optional
        Additional rotation angle in radians for in-plane (z-axis) rotation.
        Default is 0 (no additional rotation)

    Returns
    -------
    rotated_coords : Float[Array, " N 4"]
        Rotated atomic coordinates maintaining the same format as input
    rotated_cell : Float[Array, "3 3"]
        Rotated unit cell matrix

    Notes
    -----
    Applies rotation transformations to both atomic coordinates and unit cell vectors.
    Supports an optional additional in-plane rotation around the z-axis after the primary rotation.

    Algorithm:
    ---------
    - Extract atomic positions:
        - Separate atom IDs (first column) from position vectors (columns 1-3)
        - This reserves atom type information during rotation
    - Apply primary rotation to coordinates:
        - Multiply position vectors by transpose of rotation matrix: coords @ rotation_matrix.T
        - This rotates all atomic positions according to the given rotation
    - Reconstruct coordinate array:
        - Concatenate atom IDs with rotated positions
        - Maintains original array structure [atom_id, x', y', z']
    - Rotate unit cell:
        - Apply same rotation to lattice vectors: cell @ rotation_matrix.T
        - This ensures the crystal structure remains consistent
    - Handle optional in-plane rotation (if theta ≠ 0):
        - Create rotation matrix for z-axis rotation using rotmatrix_axis
        - Apply this secondary rotation to already-rotated coordinates
        - Extract positions, rotate, and reconstruct array as before
    - Return transformed structure:
        - Both atomic coordinates and unit cell are rotated consistently
        - Crystal symmetry and relative positions are preserved
    """
    rotated_coords: Real[Array, " N 3"] = coords[:, 1:4] @ rotation_matrix.T
    rotated_coords_with_ids: Float[Array, " N 4"] = jnp.hstack(
        (coords[:, 0:1], rotated_coords)
    )
    rotated_cell: Real[Array, "3 3"] = cell @ rotation_matrix.T

    def _apply_inplane_rotation() -> Float[Array, " N 4"]:
        in_plane_rotation: Float[Array, "3 3"] = rotmatrix_axis(
            jnp.array([0.0, 0.0, 1.0]), theta
        )
        rotated_coords_in_plane: Float[Array, " N 3"] = (
            rotated_coords_with_ids[:, 1:4] @ in_plane_rotation.T
        )
        return jnp.hstack((rotated_coords_with_ids[:, 0:1], rotated_coords_in_plane))

    def _no_inplane_rotation() -> Float[Array, " N 4"]:
        return rotated_coords_with_ids

    rotated_coords_final: Float[Array, " N 4"] = jax.lax.cond(
        theta != 0, _apply_inplane_rotation, _no_inplane_rotation
    )
    return (rotated_coords_final, rotated_cell)


@jaxtyped(typechecker=beartype)
def reciprocal_lattice(cell: Real[Array, "3 3"]) -> Float[Array, "3 3"]:
    """Compute the reciprocal lattice vectors from a real-space unit cell matrix.

    Parameters
    ----------
    cell : Real[Array, "3 3"]
        Real-space unit cell matrix where rows are lattice vectors a1, a2, a3

    Returns
    -------
    Float[Array, "3 3"]
        Reciprocal lattice matrix where rows are reciprocal vectors b1, b2, b3

    Notes
    -----
    The reciprocal lattice is fundamental for crystallography and diffraction calculations.

    Algorithm:
    ---------
    - Extract lattice vectors:
        - Unpack rows of cell matrix as individual lattice vectors a1, a2, a3
        - These represent the fundamental periodicity of the crystal
    - Calculate unit cell volume:
        - Compute scalar triple product: V = a1 · (a2 × a3)
        - This gives the volume of the parallelepiped formed by lattice vectors
    - Compute reciprocal lattice vectors:
        - b1 = 2π * (a2 × a3) / V
        - b2 = 2π * (a3 × a1) / V
        - b3 = 2π * (a1 × a2) / V
        - Each reciprocal vector is perpendicular to two real-space vectors
    - Assemble reciprocal lattice matrix:
        - Stack reciprocal vectors as rows to form 3x3 matrix
        - The resulting matrix satisfies: cell @ reciprocal_cell.T = 2π * I
    - Return the reciprocal lattice matrix for use in:
        - Fourier transforms between real and reciprocal space
        - Diffraction pattern calculations
        - Brillouin zone constructions
    """
    a1: Float[Array, " 3"]
    a2: Float[Array, " 3"]
    a3: Float[Array, " 3"]
    a1, a2, a3 = cell
    vv: Float[Array, ""] = jnp.dot(a1, jnp.cross(a2, a3))
    b1: Float[Array, " 3"] = 2 * jnp.pi * jnp.cross(a2, a3) / vv
    b2: Float[Array, " 3"] = 2 * jnp.pi * jnp.cross(a3, a1) / vv
    b3: Float[Array, " 3"] = 2 * jnp.pi * jnp.cross(a1, a2) / vv
    return jnp.stack([b1, b2, b3])
