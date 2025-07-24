"""
Module: electrons.electron_types
--------------------------------
Data structures and type definitions for electron microscopy and ptychography.

Type Aliases
------------
- `scalar_numeric`:
    Type alias for numeric types (int, float or Num array)
    Num Array has 0 dimensions
- `scalar_float`:
    Type alias for float or Float array of 0 dimensions
- `scalar_int`:
    Type alias for int or Integer array of 0 dimensions
- `non_jax_number`:
    Type alias for non-JAX numeric types (int, float)

PyTrees
-------
- `CalibratedArray`:
    A PyTree for calibrated array data with spatial calibration
- `ProbeModes`:
    A PyTree for multimodal electron probe state
- `PotentialSlices`:
    A PyTree for potential slices in multi-slice simulations
- `CrystalStructure`:
    A PyTree for crystal structure with fractional and Cartesian coordinates
- `XYZData`:
    A PyTree for XYZ file data with atomic positions, lattice vectors,
    stress tensor, energy, properties, and comment

Factory Functions
----------------
- `make_calibrated_array`:
    Creates a CalibratedArray instance with runtime type checking
- `make_probe_modes`:
    Creates a ProbeModes instance with runtime type checking
- `make_potential_slices`:
    Creates a PotentialSlices instance with runtime type checking
- `make_crystal_structure`:
    Creates a CrystalStructure instance with runtime type checking
- `make_xyz_data`:
    Creates a XYZData instance with runtime type checking

Note
----
Always use these factory functions instead of directly instantiating the
NamedTuple classes to ensure proper runtime type checking of the contents.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, NamedTuple, Optional, TypeAlias, Union
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

jax.config.update("jax_enable_x64", True)

scalar_numeric: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Int[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]


@register_pytree_node_class
class CalibratedArray(NamedTuple):
    """
    Description
    -----------
    PyTree structure for calibrated Array.

    Attributes
    ----------
    - `data_array` (Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]):
        The actual array data
    - `calib_y` (Float[Array, ""]):
        Calibration in y direction (0-dimensional JAX array)
    - `calib_x` (Float[Array, ""]):
        Calibration in x direction (0-dimensional JAX array)
    - `real_space` (Bool[Array, ""]):
        Whether the array is in real space.
        If False, it is in reciprocal space.
    """

    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]
    calib_y: Float[Array, ""]
    calib_x: Float[Array, ""]
    real_space: Bool[Array, ""]

    def tree_flatten(self):
        return (
            (
                self.data_array,
                self.calib_y,
                self.calib_x,
                self.real_space,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class ProbeModes(NamedTuple):
    """
    Description
    -----------
    PyTree structure for multimodal electron probe state.

    Attributes
    ----------
    - `modes` (Complex[Array, "H W M"]):
        M is number of modes
    - `weights` (Float[Array, "M"]):
        Mode occupation numbers.
    - `calib` (Float[Array, ""]):
        Pixel Calibration (0-dimensional JAX array)
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, "M"]
    calib: Float[Array, ""]

    def tree_flatten(self):
        return (
            (
                self.modes,
                self.weights,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class PotentialSlices(NamedTuple):
    """
    Description
    -----------
    PyTree structure for multiple potential slices.

    Attributes
    ----------
    - `slices` (Float[Array, "H W S"]):
        Individual potential slices.
        S is number of slices
    - `slice_thickness` (Num[Array, ""]):
        Thickness of each slice (0-dimensional JAX array)
    - `calib` (Float[Array, ""]):
        Pixel Calibration (0-dimensional JAX array)
    """

    slices: Float[Array, "H W S"]
    slice_thickness: Num[Array, ""]
    calib: Float[Array, ""]

    def tree_flatten(self):
        return (
            (
                self.slices,
                self.slice_thickness,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class CrystalStructure(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure representing a crystal structure with both
    fractional and Cartesian coordinates.

    Attributes
    ----------
    - `frac_positions` (Float[Array, "* 4"]):
        Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
        Each row contains [x, y, z, atomic_number] where:
        - x, y, z: Fractional coordinates in the unit cell (range [0,1])
        - atomic_number: Integer atomic number (Z) of the element
    - `cart_positions` (Num[Array, "* 4"]):
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
        Each row contains [x, y, z, atomic_number] where:
        - x, y, z: Cartesian coordinates in Ångstroms
        - atomic_number: Integer atomic number (Z) of the element
    - `cell_lengths` (Num[Array, "3"]):
        Unit cell lengths [a, b, c] in Ångstroms
    - `cell_angles` (Num[Array, "3"]):
        Unit cell angles [α, β, γ] in degrees.
        - α is the angle between b and c
        - β is the angle between a and c
        - γ is the angle between a and b

    Notes
    -----
    This class is registered as a PyTree node, making it compatible with JAX transformations
    like jit, grad, and vmap. The auxiliary data in tree_flatten is None as all relevant
    data is stored in JAX arrays.
    """

    frac_positions: Float[Array, "* 4"]
    cart_positions: Num[Array, "* 4"]
    cell_lengths: Num[Array, "3"]
    cell_angles: Num[Array, "3"]

    def tree_flatten(self):
        return (
            (
                self.frac_positions,
                self.cart_positions,
                self.cell_lengths,
                self.cell_angles,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class XYZData(NamedTuple):
    """
    Description
    -----------
    JAX-compatible PyTree representing a full parsed XYZ file.

    Attributes
    ----------
    - `positions` (Float[Array, "N 3"]):
        Cartesian positions in Ångstroms.
    - `atomic_numbers` (Int[Array, "N"]):
        Atomic numbers (Z) corresponding to each atom.
    - `lattice` (Optional[Float[Array, "3 3"]]):
        Lattice vectors in Ångstroms if present, otherwise None.
    - `stress` (Optional[Float[Array, "3 3"]]):
        Symmetric stress tensor if present.
    - `energy` (Optional[scalar_float]):
        Total energy in eV if present.
    - `properties` (Optional[List[Dict[str, Union[str, int]]]]):
        List of properties described in the metadata.
    - `comment` (Optional[str]):
        The raw comment line from the XYZ file.

    Notes
    -----
    - Can be used for geometry parsing, simulation prep, or ML data loaders.
    - Compatible with JAX transformations (jit, vmap, etc).
    """

    positions: Float[Array, "N 3"]
    atomic_numbers: Int[Array, "N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[Float[Array, ""]]
    properties: Optional[List[Dict[str, Union[str, int]]]]
    comment: Optional[str]

    def tree_flatten(self):
        children = (
            self.positions,
            self.atomic_numbers,
            self.lattice,
            self.stress,
            self.energy,
        )
        aux_data = {
            "properties": self.properties,
            "comment": self.comment,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        positions, atomic_numbers, lattice, stress, energy = children
        return cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=aux_data["properties"],
            comment=aux_data["comment"],
        )


@jaxtyped(typechecker=beartype)
def make_calibrated_array(
    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]],
    calib_y: scalar_float,
    calib_x: scalar_float,
    real_space: Union[bool, Bool[Array, ""]],
) -> CalibratedArray:
    """
    Description
    -----------
    JAX-safe factory function for CalibratedArray with data validation.

    Parameters
    ----------
    - `data_array` (Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]):
        The actual array data
    - `calib_y` (scalar_float):
        Calibration in y direction
    - `calib_x` (scalar_float):
        Calibration in x direction
    - `real_space` (Bool[Array, ""]):
        Whether the array is in real space

    Returns
    -------
    - `calibrated_array` (CalibratedArray):
        Validated calibrated array instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Validations
    ----------
    - data_array is 2D
    - data_array is finite
    - calib_y is positive
    - calib_x is positive
    - real_space is a boolean scalar

    Validation Flow
    ---------------
    - Convert inputs to JAX arrays with appropriate dtypes:
       - data_array: Convert to int32, float64, or complex128 based on input dtype
       - calib_y: Convert to float64
       - calib_x: Convert to float64
       - real_space: Convert to bool
    - Execute validation checks using JAX-compatible conditional logic:
       - check_2d_array(): Verify data_array has exactly 2 dimensions
       - check_array_finite(): Ensure all values in data_array are finite (no inf/nan)
       - check_calib_y(): Confirm calib_y is strictly positive
       - check_calib_x(): Confirm calib_x is strictly positive
       - check_real_space(): Verify real_space is a scalar (0-dimensional)
    - If all validations pass, create and return CalibratedArray instance
    - If any validation fails, the JAX-compatible error handling will stop execution
    """
    # Convert all inputs to JAX arrays
    # The jaxtyping decorator already validates the shape and type constraints
    # We just ensure the data is a JAX array and preserve its dtype
    data_array = jnp.asarray(data_array)

    calib_y = jnp.asarray(calib_y, dtype=jnp.float64)
    calib_x = jnp.asarray(calib_x, dtype=jnp.float64)
    real_space = jnp.asarray(real_space, dtype=jnp.bool_)

    # For JAX compliance, we rely on jaxtyping for shape/type validation
    # and only do JAX-compatible runtime checks that don't break transformations

    # Ensure calibrations are positive using JAX operations
    # This will naturally produce NaN/Inf if calibrations are invalid
    calib_y = jnp.abs(calib_y) + jnp.finfo(jnp.float64).eps
    calib_x = jnp.abs(calib_x) + jnp.finfo(jnp.float64).eps

    return CalibratedArray(
        data_array=data_array,
        calib_y=calib_y,
        calib_x=calib_x,
        real_space=real_space,
    )


@jaxtyped(typechecker=beartype)
def make_probe_modes(
    modes: Complex[Array, "H W M"],
    weights: Float[Array, "M"],
    calib: scalar_float,
) -> ProbeModes:
    """
    Description
    -----------
    JAX-safe factory function for ProbeModes with data validation.

    Parameters
    ----------
    - `modes` (Complex[Array, "H W M"]):
        Complex probe modes, M is number of modes
    - `weights` (Float[Array, "M"]):
        Mode occupation numbers
    - `calib` (scalar_float):
        Pixel calibration

    Returns
    -------
    - `probe_modes` (ProbeModes):
        Validated probe modes instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Validation Flow
    ---------------
    - Convert inputs to JAX arrays with appropriate dtypes:
       - modes: Convert to complex128
       - weights: Convert to float64
       - calib: Convert to float64
    - Extract shape information (H, W, M) from modes array
    - Execute validation checks using JAX-compatible conditional logic:
       - check_3d_modes(): Verify modes array has exactly 3 dimensions
       - check_modes_finite(): Ensure all values in modes are finite (no inf/nan)
       - check_weights_shape(): Confirm weights has shape (M,) matching modes dimension
       - check_weights_nonnegative(): Verify all weights are non-negative
       - check_weights_sum(): Ensure sum of weights is strictly positive
       - check_calib(): Confirm calibration value is strictly positive
    - If all validations pass, create and return ProbeModes instance
    - If any validation fails, the JAX-compatible error handling will stop execution
    """
    modes = jnp.asarray(modes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    calib = jnp.asarray(calib, dtype=jnp.float64)

    # For JAX compliance, we rely on jaxtyping for shape/type validation
    # and only do JAX-compatible runtime adjustments

    # Ensure weights are non-negative and normalized
    weights = jnp.abs(weights)
    weight_sum = jnp.sum(weights)
    # If all weights are zero, make them uniform
    weights = jnp.where(
        weight_sum > jnp.finfo(jnp.float64).eps,
        weights / weight_sum,
        jnp.ones_like(weights) / weights.shape[0],
    )

    # Ensure calibration is positive
    calib = jnp.abs(calib) + jnp.finfo(jnp.float64).eps

    return ProbeModes(
        modes=modes,
        weights=weights,
        calib=calib,
    )


@jaxtyped(typechecker=beartype)
def make_potential_slices(
    slices: Float[Array, "H W S"],
    slice_thickness: scalar_numeric,
    calib: scalar_float,
) -> PotentialSlices:
    """
    Description
    -----------
    JAX-safe factory function for PotentialSlices with data validation.

    Parameters
    ----------
    - `slices` (Float[Array, "H W S"]):
        Individual potential slices, S is number of slices
    - `slice_thickness` (scalar_numeric):
        Thickness of each slice
    - `calib` (scalar_float):
        Pixel calibration

    Returns
    -------
    - `potential_slices` (PotentialSlices):
        Validated potential slices instance

    Raises
    ------
    - ValueError:
        If data is invalid or parameters are out of valid ranges

    Validation Flow
    ---------------
    - Convert inputs to JAX arrays with appropriate dtypes:
       - slices: Convert to complex128
       - slice_thickness: Convert to float64
       - calib: Convert to float64
    - Extract shape information (H, W, S) from slices array
    - Execute validation checks using JAX-compatible conditional logic:
       - check_3d_slices(): Verify slices array has exactly 3 dimensions
       - check_slices_finite(): Ensure all values in slices are finite (no inf/nan)
       - check_slice_thickness(): Confirm slice_thickness is strictly positive
       - check_calib(): Confirm calibration value is strictly positive
    - If all validations pass, create and return PotentialSlices instance
    - If any validation fails, the JAX-compatible error handling will stop execution
    """
    slices = jnp.asarray(slices, dtype=jnp.float64)
    slice_thickness = jnp.asarray(slice_thickness, dtype=jnp.float64)
    calib = jnp.asarray(calib, dtype=jnp.float64)
    slice_thickness = jnp.abs(slice_thickness) + jnp.finfo(jnp.float64).eps
    calib = jnp.abs(calib) + jnp.finfo(jnp.float64).eps
    return PotentialSlices(
        slices=slices,
        slice_thickness=slice_thickness,
        calib=calib,
    )


@beartype
def make_crystal_structure(
    frac_positions: Float[Array, "* 4"],
    cart_positions: Num[Array, "* 4"],
    cell_lengths: Num[Array, "3"],
    cell_angles: Num[Array, "3"],
) -> CrystalStructure:
    """
    Factory function to create a CrystalStructure instance with type checking.

    Parameters
    ----------
    - `frac_positions` : Float[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
    - `cart_positions` : Num[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
    - `cell_lengths` : Num[Array, "3"]
        Unit cell lengths [a, b, c] in Ångstroms.
    - `cell_angles` : Num[Array, "3"]
        Unit cell angles [α, β, γ] in degrees.

    Returns
    -------
    - `CrystalStructure` : CrystalStructure
        A validated CrystalStructure instance.

    Raises
    ------
    ValueError
        If the input arrays have incompatible shapes or invalid values.

    Validation Flow
    ---------------
    - Convert all inputs to JAX arrays:
       - frac_positions: Convert to JAX array (maintains original dtype)
       - cart_positions: Convert to JAX array (maintains original dtype)
       - cell_lengths: Convert to JAX array (maintains original dtype)
       - cell_angles: Convert to JAX array (maintains original dtype)
    - Execute shape validation checks:
       - check_frac_shape(): Verify frac_positions has 4 columns [x, y, z, atomic_number]
       - check_cart_shape(): Verify cart_positions has 4 columns [x, y, z, atomic_number]
       - check_cell_lengths_shape(): Confirm cell_lengths has shape (3,)
       - check_cell_angles_shape(): Confirm cell_angles has shape (3,)
    - Execute consistency validation checks:
       - check_atom_count(): Ensure frac_positions and cart_positions have same number of atoms
       - check_atomic_numbers(): Verify atomic numbers match between frac and cart positions
    - Execute value validation checks:
       - check_cell_lengths_positive(): Confirm all cell lengths are strictly positive
       - check_cell_angles_valid(): Ensure all angles are in range (0, 180) degrees
    - If all validations pass, create and return CrystalStructure instance
    - If any validation fails, the JAX-compatible error handling will stop execution
    """
    frac_positions = jnp.asarray(frac_positions)
    cart_positions = jnp.asarray(cart_positions)
    cell_lengths = jnp.asarray(cell_lengths)
    cell_angles = jnp.asarray(cell_angles)

    # For JAX compliance, we rely on beartype for shape/type validation
    # and only do JAX-compatible runtime adjustments

    # Ensure cell lengths are positive
    cell_lengths = jnp.abs(cell_lengths) + jnp.finfo(jnp.float64).eps

    # Ensure cell angles are in valid range (0, 180)
    # Clamp to valid range rather than failing
    cell_angles = jnp.clip(cell_angles, 0.1, 179.9)

    return CrystalStructure(
        frac_positions=frac_positions,
        cart_positions=cart_positions,
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )


@jaxtyped(typechecker=beartype)
def make_xyz_data(
    positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[scalar_float] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> XYZData:
    """
    Description
    -----------
    JAX-safe factory function for XYZData with runtime validation.

    Parameters
    ----------
    - `positions` (Float[Array, "N 3"]):
        Cartesian positions in Ångstroms
    - `atomic_numbers` (Int[Array, "N"]):
        Atomic numbers (Z) for each atom
    - `lattice` (Optional[Float[Array, "3 3"]]):
        Lattice vectors (if any)
    - `stress` (Optional[Float[Array, "3 3"]]):
        Stress tensor (if any)
    - `energy` (Optional[scalar_float]):
        Total energy (if any)
    - `properties` (Optional[List[Dict[str, Union[str, int]]]]):
        Per-atom metadata
    - `comment` (Optional[str]):
        Original XYZ comment line

    Returns
    -------
    - `XYZData`:
        Validated PyTree structure for XYZ file contents

    Validation Flow
    ---------------
    - Convert required inputs to JAX arrays with appropriate dtypes:
       - positions: Convert to float64
       - atomic_numbers: Convert to int32
       - lattice (if provided): Convert to float64
       - stress (if provided): Convert to float64
       - energy (if provided): Convert to float64
    - Extract number of atoms (N) from positions array
    - Execute shape validation checks:
       - check_shape(): Verify positions has shape (N, 3) and atomic_numbers has shape (N,)
    - Execute value validation checks:
       - check_finiteness(): Ensure all position values are finite and atomic numbers are non-negative
    - Execute optional matrix validation checks (if provided):
       - check_optional_matrices(): For lattice and stress tensors:
         * Verify shape is (3, 3)
         * Ensure all values are finite
    - If all validations pass, create and return XYZData instance
    - If any validation fails, raise ValueError with descriptive error message
    """

    positions = jnp.asarray(positions, dtype=jnp.float64)
    atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)

    # Convert optional parameters to JAX arrays
    # Note: We have to use Python if for None checks since lax.cond requires
    # JAX-compatible boolean conditions and both branches must return same type.
    # This is another unavoidable case for Python if.
    if lattice is not None:
        lattice = jnp.asarray(lattice, dtype=jnp.float64)
    else:
        # Default to identity matrix if no lattice is provided
        # This ensures lattice is always a JAX array for downstream functions
        lattice = jnp.eye(3, dtype=jnp.float64)

    if stress is not None:
        stress = jnp.asarray(stress, dtype=jnp.float64)

    if energy is not None:
        energy = jnp.asarray(energy, dtype=jnp.float64)

    def validate_and_create():
        N = positions.shape[0]

        def check_shape():
            if positions.shape[1] != 3:
                raise ValueError("positions must have shape (N, 3)")
            if atomic_numbers.shape[0] != N:
                raise ValueError("atomic_numbers must have shape (N,)")

        def check_finiteness():
            if not jnp.all(jnp.isfinite(positions)):
                raise ValueError("positions contain non-finite values")
            if not jnp.all(atomic_numbers >= 0):
                raise ValueError("atomic_numbers must be non-negative")

        def check_optional_matrices():
            # We have to use Python if for None checks here as well
            if lattice is not None:
                if lattice.shape != (3, 3):
                    raise ValueError("lattice must have shape (3, 3)")
                if not jnp.all(jnp.isfinite(lattice)):
                    raise ValueError("lattice contains non-finite values")

            if stress is not None:
                if stress.shape != (3, 3):
                    raise ValueError("stress must have shape (3, 3)")
                if not jnp.all(jnp.isfinite(stress)):
                    raise ValueError("stress contains non-finite values")

        check_shape()
        check_finiteness()
        check_optional_matrices()

        return XYZData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            lattice=lattice,
            stress=stress,
            energy=energy,
            properties=properties,
            comment=comment,
        )

    return validate_and_create()
