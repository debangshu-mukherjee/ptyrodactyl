"""Geometric transformations and operations for crystal structures.

Extended Summary
----------------
Provides rotation matrices and lattice operations for manipulating
crystal structures in electron microscopy simulations.

Routine Listings
----------------
:func:`rotmatrix_vectors`
    Compute a rotation matrix that rotates one vector to align
    with another.
:func:`rotmatrix_axis`
    Generate a rotation matrix for rotation around an arbitrary
    axis.
:func:`rotate_structure`
    Apply rotation transformations to crystal structures.
:func:`reciprocal_lattice`
    Compute reciprocal lattice vectors from real-space unit
    cell.
:func:`tilt_crystal`
    Tilt :class:`~ptyrodactyl.tools.CrystalData` by alpha and
    beta angles (TEM stage-like tilts).

Notes
-----
All functions use the Rodrigues rotation formula and are
JAX-compatible for automatic differentiation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float, Real, jaxtyped

from ptyrodactyl.tools import (
    CrystalData,
    ScalarFloat,
    ScalarNumeric,
    make_crystal_data,
)


@jaxtyped(typechecker=beartype)
@jax.jit
def rotmatrix_vectors(
    v1: Real[Array, " 3"], v2: Real[Array, " 3"]
) -> Float[Array, "3 3"]:
    r"""Compute a rotation matrix that rotates v1 to align with v2.

    Extended Summary
    ----------------
    Uses the Rodrigues rotation formula to compute a proper 3x3
    rotation matrix. Handles special cases where vectors are
    parallel or anti-parallel.

    .. math::

        R = I + \sin\theta\, K + (1 - \cos\theta)\, K^2

    where :math:`K` is the skew-symmetric matrix of the unit
    rotation axis.

    Implementation Logic
    --------------------
    1. **Normalize input vectors** --
       Divide v1 and v2 by their norms to get unit vectors.
    2. **Compute rotation parameters** --
       Cross product gives rotation axis direction, dot product
       gives cosine of the rotation angle, and the norm of the
       cross product gives sine of the angle.
    3. **Handle parallel vectors** --
       If ``sin_theta < 1e-8``, vectors are nearly parallel.
       Return identity if same direction, or a 180-degree
       rotation matrix if opposite.
    4. **Apply Rodrigues formula** --
       Construct skew-symmetric matrix K from the unit rotation
       axis and compute
       ``R = I + sin(theta) * K + (1 - cos(theta)) * K @ K``.

    Parameters
    ----------
    v1 : Real[Array, " 3"]
        Initial 3D vector to be rotated.
    v2 : Real[Array, " 3"]
        Target 3D vector that v1 should be rotated to align
        with.

    Returns
    -------
    rotation_matrix : Float[Array, "3 3"]
        3x3 rotation matrix such that
        ``rotation_matrix @ v1`` is parallel to v2.

    Notes
    -----
    Fully JIT-compilable. Uses ``jax.lax.cond`` for the
    parallel/anti-parallel branching logic.
    """
    v1: Float[Array, " 3"] = v1 / jnp.linalg.norm(v1)
    v2: Float[Array, " 3"] = v2 / jnp.linalg.norm(v2)
    cross: Float[Array, " 3"] = jnp.cross(v1, v2)
    dot: Float[Array, " "] = jnp.dot(v1, v2)
    sin_theta: Float[Array, " "] = jnp.linalg.norm(cross)

    def _fallback_parallel() -> Float[Array, "3 3"]:
        """Return identity when vectors are already parallel.

        Returns
        -------
        rotation_matrix_parallel : Float[Array, "3 3"]
            3x3 identity matrix.
        """
        rotation_matrix_parallel: Float[Array, "3 3"] = jnp.eye(3)
        return rotation_matrix_parallel

    def _fallback_opposite() -> Float[Array, "3 3"]:
        """Compute 180-degree rotation for anti-parallel vectors.

        Returns
        -------
        rotation_matrix_opposite : Float[Array, "3 3"]
            Rotation matrix for 180-degree rotation around an
            axis orthogonal to v1.
        """
        magic_number: ScalarFloat = 0.9
        ortho: Float[Array, " 3"] = jnp.where(
            jnp.abs(v1[0]) < magic_number,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
        )
        axis: Float[Array, " 3"] = jnp.cross(v1, ortho)
        axis: Float[Array, " 3"] = axis / jnp.linalg.norm(axis)
        kk: Float[Array, "3 3"] = jnp.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        rotation_matrix_opposite: Float[Array, "3 3"] = (
            jnp.eye(3) + 2 * kk @ kk
        )
        return rotation_matrix_opposite

    def _compute() -> Float[Array, "3 3"]:
        """Compute rotation via Rodrigues formula.

        Returns
        -------
        rotation_matrix_general : Float[Array, "3 3"]
            Rotation matrix for the general (non-degenerate)
            case.
        """
        axis: Float[Array, " 3"] = cross / sin_theta
        kk: Float[Array, "3 3"] = jnp.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        rotation_matrix_general: Float[Array, "3 3"] = (
            jnp.eye(3) + sin_theta * kk + (1 - dot) * (kk @ kk)
        )
        return rotation_matrix_general

    close_to_zero: ScalarFloat = 1e-8
    almost_parallel: Bool[Array, " "] = sin_theta < close_to_zero
    close_to_one: ScalarFloat = 0.999999
    almost_opposite: Bool[Array, " "] = dot < -close_to_one
    rotation_matrix: Float[Array, "3 3"] = jax.lax.cond(
        almost_parallel,
        lambda: jax.lax.cond(
            almost_opposite, _fallback_opposite, _fallback_parallel
        ),
        _compute,
    )
    return rotation_matrix


@jaxtyped(typechecker=beartype)
@jax.jit
def rotmatrix_axis(
    axis: Real[Array, " 3"], theta: ScalarNumeric
) -> Float[Array, "3 3"]:
    r"""Generate a rotation matrix around an arbitrary axis.

    Extended Summary
    ----------------
    Uses the Rodrigues rotation formula to produce a right-handed
    rotation when looking along the axis direction.

    .. math::

        R = I\cos\theta
            + (1 - \cos\theta)\,\hat{n}\otimes\hat{n}
            + \sin\theta\,[\hat{n}]_\times

    Implementation Logic
    --------------------
    1. **Normalize rotation axis** --
       Divide the axis vector by its norm.
    2. **Compute trigonometric values** --
       ``cos(theta)`` and ``sin(theta)``.
    3. **Build rotation matrix** --
       Assemble the 3x3 matrix from diagonal terms
       ``cos(theta) + n_i^2 * (1 - cos(theta))``,
       symmetric off-diagonal terms, and antisymmetric
       terms proportional to ``sin(theta)``.

    Parameters
    ----------
    axis : Real[Array, " 3"]
        3D vector defining the axis of rotation (will be
        normalized).
    theta : ScalarNumeric
        Rotation angle in radians. Positive for
        counter-clockwise rotation when looking along the
        axis.

    Returns
    -------
    rot_matrix : Float[Array, "3 3"]
        3x3 rotation matrix that rotates vectors by *theta*
        radians around the axis.

    Notes
    -----
    Fully JIT-compilable. The resulting matrix is orthogonal
    with determinant +1.
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
@jax.jit
def rotate_structure(
    coords: Real[Array, " N 4"],
    cell: Real[Array, "3 3"],
    rotation_matrix: Real[Array, "3 3"],
    theta: ScalarNumeric = 0,
) -> Tuple[Float[Array, " N 4"], Float[Array, "3 3"]]:
    """Apply rotation to a crystal structure.

    Extended Summary
    ----------------
    Rotates both atomic coordinates and unit cell vectors by the
    given rotation matrix. Optionally applies an additional
    in-plane rotation around the z-axis.

    Implementation Logic
    --------------------
    1. **Extract positions** --
       Separate atom IDs (column 0) from xyz positions
       (columns 1-3).
    2. **Apply primary rotation** --
       ``coords[:, 1:4] @ rotation_matrix.T``.
    3. **Rotate unit cell** --
       ``cell @ rotation_matrix.T``.
    4. **Apply optional in-plane rotation** --
       If ``theta != 0``, build a z-axis rotation via
       :func:`rotmatrix_axis` and apply it to the already
       rotated coordinates.

    Parameters
    ----------
    coords : Real[Array, " N 4"]
        Atomic coordinates where each row is
        ``[atom_id, x, y, z]``. Positions in Angstroms.
    cell : Real[Array, "3 3"]
        Unit cell matrix where rows are lattice vectors
        a, b, c in Angstroms.
    rotation_matrix : Real[Array, "3 3"]
        Primary rotation matrix to apply.
    theta : ScalarNumeric, optional
        Additional in-plane (z-axis) rotation angle in
        radians. Default is 0.

    Returns
    -------
    rotated_coords_final : Float[Array, " N 4"]
        Rotated coordinates in ``[atom_id, x', y', z']``
        format.
    rotated_cell : Float[Array, "3 3"]
        Rotated unit cell matrix in Angstroms.

    See Also
    --------
    :func:`rotmatrix_axis` : Build rotation matrix from axis
        and angle.
    :func:`rotmatrix_vectors` : Build rotation matrix from two
        vectors.
    """
    rotated_coords: Real[Array, " N 3"] = coords[:, 1:4] @ rotation_matrix.T
    rotated_coords_with_ids: Float[Array, " N 4"] = jnp.hstack(
        (coords[:, 0:1], rotated_coords)
    )
    rotated_cell: Real[Array, "3 3"] = cell @ rotation_matrix.T

    def _apply_inplane_rotation() -> Float[Array, " N 4"]:
        """Apply in-plane z-axis rotation to coordinates.

        Returns
        -------
        Float[Array, " N 4"]
            Coordinates after secondary z-axis rotation.
        """
        in_plane_rotation: Float[Array, "3 3"] = rotmatrix_axis(
            jnp.array([0.0, 0.0, 1.0]), theta
        )
        rotated_coords_in_plane: Float[Array, " N 3"] = (
            rotated_coords_with_ids[:, 1:4] @ in_plane_rotation.T
        )
        return jnp.hstack(
            (rotated_coords_with_ids[:, 0:1], rotated_coords_in_plane)
        )

    def _no_inplane_rotation() -> Float[Array, " N 4"]:
        """Return coordinates unchanged (no-op branch).

        Returns
        -------
        Float[Array, " N 4"]
            Unmodified rotated coordinates.
        """
        return rotated_coords_with_ids

    rotated_coords_final: Float[Array, " N 4"] = jax.lax.cond(
        theta != 0, _apply_inplane_rotation, _no_inplane_rotation
    )
    return (rotated_coords_final, rotated_cell)


@jaxtyped(typechecker=beartype)
def reciprocal_lattice(cell: Real[Array, "3 3"]) -> Float[Array, "3 3"]:
    r"""Compute reciprocal lattice vectors from a real-space cell.

    Extended Summary
    ----------------
    Computes the reciprocal lattice matrix satisfying
    ``cell @ reciprocal.T = 2 pi I``. Fundamental for
    diffraction pattern and Brillouin zone calculations.

    .. math::

        \mathbf{b}_i = \frac{2\pi\,
        (\mathbf{a}_j \times \mathbf{a}_k)}{
        \mathbf{a}_1 \cdot
        (\mathbf{a}_2 \times \mathbf{a}_3)}

    Implementation Logic
    --------------------
    1. **Extract lattice vectors** --
       Unpack rows as a1, a2, a3.
    2. **Compute cell volume** --
       Scalar triple product ``V = a1 . (a2 x a3)``.
    3. **Compute reciprocal vectors** --
       ``bi = 2 pi (aj x ak) / V`` for cyclic permutations.
    4. **Stack into matrix** --
       Return ``[b1, b2, b3]`` as rows.

    Parameters
    ----------
    cell : Real[Array, "3 3"]
        Real-space unit cell matrix where rows are lattice
        vectors a1, a2, a3 in Angstroms.

    Returns
    -------
    reciprocal_cell : Float[Array, "3 3"]
        Reciprocal lattice matrix where rows are reciprocal
        vectors b1, b2, b3 in inverse Angstroms.
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


@jaxtyped(typechecker=beartype)
def tilt_crystal(
    crystal_data: CrystalData,
    alpha_rad: ScalarNumeric,
    beta_rad: ScalarNumeric,
) -> CrystalData:
    r"""Tilt :class:`~ptyrodactyl.tools.CrystalData` by alpha and beta.

    Extended Summary
    ----------------
    Applies a combined rotation mimicking a TEM double-tilt
    holder: first tilt around the x-axis by *alpha*, then
    around the y-axis by *beta*.

    .. math::

        R_{\text{total}} = R_y(\beta)\, R_x(\alpha)

    The function is fully differentiable with respect to both
    angles, enabling gradient-based orientation optimization.

    Implementation Logic
    --------------------
    1. **Build individual rotation matrices** --
       :func:`rotmatrix_axis` for x-axis (alpha) and y-axis
       (beta).
    2. **Combine rotations** --
       ``R_total = R_y @ R_x``.
    3. **Rotate positions** --
       ``positions @ R_total.T``.
    4. **Rotate lattice** --
       ``lattice @ R_total.T`` if lattice is present.
    5. **Return new CrystalData** --
       Via :func:`~ptyrodactyl.tools.make_crystal_data` with
       all other fields preserved.

    Parameters
    ----------
    crystal_data : CrystalData
        Input crystal structure data.
    alpha_rad : ScalarNumeric
        Tilt angle around the x-axis in radians. Positive
        alpha tilts +z toward +y.
    beta_rad : ScalarNumeric
        Tilt angle around the y-axis in radians. Positive
        beta tilts +z toward -x.

    Returns
    -------
    tilted_crystal : CrystalData
        New :class:`~ptyrodactyl.tools.CrystalData` with
        rotated positions and lattice. All other fields
        (atomic_numbers, energy, etc.) are preserved.

    See Also
    --------
    :func:`rotmatrix_axis` : Rotation matrix from axis and
        angle.
    :func:`rotate_structure` : Lower-level rotation for
        coords + cell arrays.
    """
    x_axis: Float[Array, " 3"] = jnp.array([1.0, 0.0, 0.0])
    y_axis: Float[Array, " 3"] = jnp.array([0.0, 1.0, 0.0])

    r_x: Float[Array, "3 3"] = rotmatrix_axis(x_axis, alpha_rad)
    r_y: Float[Array, "3 3"] = rotmatrix_axis(y_axis, beta_rad)

    r_total: Float[Array, "3 3"] = r_y @ r_x

    rotated_positions: Float[Array, "N 3"] = crystal_data.positions @ r_total.T

    rotated_lattice: Float[Array, "3 3"] | None = None
    if crystal_data.lattice is not None:
        rotated_lattice = crystal_data.lattice @ r_total.T

    tilted_crystal: CrystalData = make_crystal_data(
        positions=rotated_positions,
        atomic_numbers=crystal_data.atomic_numbers,
        lattice=rotated_lattice,
        stress=crystal_data.stress,
        energy=crystal_data.energy,
        properties=crystal_data.properties,
        comment=crystal_data.comment,
    )
    return tilted_crystal
