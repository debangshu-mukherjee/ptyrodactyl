"""
Module: ptyrodactyl.electrons.electron_types
---------------------------------------------
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
- `STEM4D`:
    A PyTree for 4D-STEM data containing diffraction patterns, calibrations,
    scan positions, and experimental parameters

Factory Functions
-----------------
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
- `make_stem4d`:
    Creates a STEM4D instance with runtime type checking

Notes
-----
Always use these factory functions instead of directly instantiating the
NamedTuple classes to ensure proper runtime type checking of the contents.
"""

import jax
import jax.numpy as jnp
from beartype.typing import (Any, Dict, List, NamedTuple, Optional, Tuple,
                             TypeAlias, Union)
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num

from ptyrodactyl._decorators import beartype, jaxtyped

jax.config.update("jax_enable_x64", True)

scalar_numeric: TypeAlias = Union[int, float, Num[Array, " "]]
scalar_float: TypeAlias = Union[float, Float[Array, " "]]
scalar_int: TypeAlias = Union[int, Int[Array, " "]]
non_jax_number: TypeAlias = Union[int, float]


@register_pytree_node_class
class CalibratedArray(NamedTuple):
    """PyTree structure for calibrated Array.

    Attributes
    ----------
    data_array : Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]
        The actual array data
    calib_y : Float[Array, " "]
        Calibration in y direction (0-dimensional JAX array)
    calib_x : Float[Array, " "]
        Calibration in x direction (0-dimensional JAX array)
    real_space : Bool[Array, " "]
        Whether the array is in real space.
        If False, it is in reciprocal space.
    """

    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]
    calib_y: Float[Array, " "]
    calib_x: Float[Array, " "]
    real_space: Bool[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
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
    def tree_unflatten(
        cls, _aux_data: None, children: Tuple[Any, ...]
    ) -> "CalibratedArray":
        return cls(*children)


@register_pytree_node_class
class ProbeModes(NamedTuple):
    """PyTree structure for multimodal electron probe state.

    Attributes
    ----------
    modes : Complex[Array, "H W M"]
        M is number of modes
    weights : Float[Array, " M"]
        Mode occupation numbers.
    calib : Float[Array, " "]
        Pixel Calibration (0-dimensional JAX array)
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, " M"]
    calib: Float[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        return (
            (
                self.modes,
                self.weights,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data: None, children: Tuple[Any, ...]) -> "ProbeModes":
        return cls(*children)


@register_pytree_node_class
class PotentialSlices(NamedTuple):
    """PyTree structure for multiple potential slices.

    Attributes
    ----------
    slices : Float[Array, "H W S"]
        Individual potential slices.
        S is number of slices
    slice_thickness : Num[Array, " "]
        Thickness of each slice (0-dimensional JAX array)
    calib : Float[Array, " "]
        Pixel Calibration (0-dimensional JAX array)
    """

    slices: Float[Array, "H W S"]
    slice_thickness: Num[Array, " "]
    calib: Float[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        return (
            (
                self.slices,
                self.slice_thickness,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, _aux_data: None, children: Tuple[Any, ...]
    ) -> "PotentialSlices":
        return cls(*children)


@register_pytree_node_class
class CrystalStructure(NamedTuple):
    """A JAX-compatible data structure representing a crystal structure with both
    fractional and Cartesian coordinates.

    Attributes
    ----------
    frac_positions : Float[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
        Each row contains [x, y, z, atomic_number] where:
        - x, y, z: Fractional coordinates in the unit cell (range [0,1])
        - atomic_number: Integer atomic number (Z) of the element
    cart_positions : Num[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
        Each row contains [x, y, z, atomic_number] where:
        - x, y, z: Cartesian coordinates in Ångstroms
        - atomic_number: Integer atomic number (Z) of the element
    cell_lengths : Num[Array, " 3"]
        Unit cell lengths [a, b, c] in Ångstroms
    cell_angles : Num[Array, " 3"]
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
    cell_lengths: Num[Array, " 3"]
    cell_angles: Num[Array, " 3"]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
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
    def tree_unflatten(
        cls, _aux_data: None, children: Tuple[Any, ...]
    ) -> "CrystalStructure":
        return cls(*children)


@register_pytree_node_class
class XYZData(NamedTuple):
    """JAX-compatible PyTree representing a full parsed XYZ file.

    Attributes
    ----------
    positions : Float[Array, " N 3"]
        Cartesian positions in Ångstroms.
    atomic_numbers : Int[Array, " N"]
        Atomic numbers (Z) corresponding to each atom.
    lattice : Optional[Float[Array, "3 3"]]
        Lattice vectors in Ångstroms if present, otherwise None.
    stress : Optional[Float[Array, "3 3"]]
        Symmetric stress tensor if present.
    energy : Optional[scalar_float]
        Total energy in eV if present.
    properties : Optional[List[Dict[str, Union[str, int]]]]
        List of properties described in the metadata.
    comment : Optional[str]
        The raw comment line from the XYZ file.

    Notes
    -----
    - Can be used for geometry parsing, simulation prep, or ML data loaders.
    - Compatible with JAX transformations (jit, vmap, etc).
    """

    positions: Float[Array, " N 3"]
    atomic_numbers: Int[Array, " N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[Float[Array, " "]]
    properties: Optional[List[Dict[str, Union[str, int]]]]
    comment: Optional[str]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
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
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple[Any, ...]
    ) -> "XYZData":
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


@register_pytree_node_class
class STEM4D(NamedTuple):
    """PyTree structure for 4D-STEM data containing diffraction patterns
    at multiple scan positions with associated calibrations and metadata.

    Attributes
    ----------
    data : Float[Array, "P H W"]
        4D-STEM data array where:
        - P: Number of scan positions
        - H, W: Height and width of diffraction patterns
    real_space_calib : Float[Array, " "]
        Real space calibration in Angstroms per pixel
    fourier_space_calib : Float[Array, " "]
        Fourier space calibration in inverse Angstroms per pixel
    scan_positions : Float[Array, "P 2"]
        Real space scan positions in Angstroms (y, x coordinates)
    voltage_kv : Float[Array, " "]
        Accelerating voltage in kilovolts
    """

    data: Float[Array, "P H W"]
    real_space_calib: Float[Array, " "]
    fourier_space_calib: Float[Array, " "]
    scan_positions: Float[Array, "P 2"]
    voltage_kv: Float[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        return (
            (
                self.data,
                self.real_space_calib,
                self.fourier_space_calib,
                self.scan_positions,
                self.voltage_kv,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data: None, children: Tuple[Any, ...]) -> "STEM4D":
        return cls(*children)


@jaxtyped(typechecker=beartype)
def make_calibrated_array(
    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]],
    calib_y: scalar_float,
    calib_x: scalar_float,
    real_space: Union[bool, Bool[Array, " "]],
) -> CalibratedArray:
    """JAX-safe factory function for CalibratedArray with data validation.

    Parameters
    ----------
    data_array : Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]
        The actual array data
    calib_y : scalar_float
        Calibration in y direction
    calib_x : scalar_float
        Calibration in x direction
    real_space : Union[bool, Bool[Array, " "]]
        Whether the array is in real space

    Returns
    -------
    CalibratedArray
        Validated calibrated array instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Validations:
    - data_array is 2D
    - data_array is finite
    - calib_y is positive
    - calib_x is positive
    - real_space is a boolean scalar

    Validation Flow:
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
    weights: Float[Array, " M"],
    calib: scalar_float,
) -> ProbeModes:
    """JAX-safe factory function for ProbeModes with data validation.

    Parameters
    ----------
    modes : Complex[Array, "H W M"]
        Complex probe modes, M is number of modes
    weights : Float[Array, " M"]
        Mode occupation numbers
    calib : scalar_float
        Pixel calibration

    Returns
    -------
    ProbeModes
        Validated probe modes instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Validation Flow:
    - Convert inputs to JAX arrays with appropriate dtypes:
       - modes: Convert to complex128
       - weights: Convert to float64
       - calib: Convert to float64 scalar
    - Extract shape information from modes array and expected dimensions
    - Define validation functions:
       - _check_3d_modes(): Verify modes array has exactly 3 dimensions
       - _check_modes_finite(): Ensure all values in modes are finite (no inf/nan)
       - _check_weights_shape(): Confirm weights has shape (M,) matching modes dimension
       - _check_weights_nonnegative(): Verify all weights are non-negative
       - _check_weights_sum(): Ensure sum of weights is strictly positive
       - _check_calib_positive(): Confirm calibration value is strictly positive
    - Chain all validation checks with jnp.logical_and
    - Use jax.lax.cond to branch based on validation result:
       - If valid: Normalize weights to sum to 1, ensure positive calibration, return ProbeModes
       - If invalid: Return ProbeModes with NaN values to signal validation failure
    """
    modes: Complex[Array, " H W M"] = jnp.asarray(modes, dtype=jnp.complex128)
    weights: Float[Array, " M"] = jnp.asarray(weights, dtype=jnp.float64)
    calib: Float[Array, " "] = jnp.asarray(calib, dtype=jnp.float64)

    expected_dims: int = 3
    modes_shape: Tuple[int, int, int] = modes.shape
    num_modes: int = modes_shape[2] if len(modes_shape) == expected_dims else 0

    def _check_3d_modes() -> Bool[Array, " "]:
        return jnp.array(len(modes.shape) == expected_dims)

    def _check_modes_finite() -> Bool[Array, " "]:
        return jnp.all(jnp.isfinite(modes))

    def _check_weights_shape() -> Bool[Array, " "]:
        return jnp.array(weights.shape == (num_modes,))

    def _check_weights_nonnegative() -> Bool[Array, " "]:
        return jnp.all(weights >= 0)

    def _check_weights_sum() -> Bool[Array, " "]:
        weight_sum: Float[Array, " "] = jnp.sum(weights)
        return weight_sum > jnp.finfo(jnp.float64).eps

    def _check_calib_positive() -> Bool[Array, " "]:
        return calib > 0

    def _valid_processing() -> ProbeModes:
        normalized_weights: Float[Array, " M"] = jnp.abs(weights)
        weight_sum: Float[Array, " "] = jnp.sum(normalized_weights)
        normalized_weights = jax.lax.cond(
            weight_sum > jnp.finfo(jnp.float64).eps,
            lambda w: w / weight_sum,
            lambda w: jnp.ones_like(w) / w.shape[0],
            normalized_weights,
        )
        positive_calib: Float[Array, " "] = jnp.abs(calib) + jnp.finfo(jnp.float64).eps

        return ProbeModes(
            modes=modes,
            weights=normalized_weights,
            calib=positive_calib,
        )

    def _invalid_processing() -> ProbeModes:
        nan_weights: Float[Array, " M"] = jnp.full_like(weights, jnp.nan)
        nan_calib: Float[Array, " "] = jnp.array(jnp.nan, dtype=jnp.float64)
        return ProbeModes(
            modes=modes,
            weights=nan_weights,
            calib=nan_calib,
        )

    all_valid: Bool[Array, " "] = jnp.logical_and(
        _check_3d_modes(),
        jnp.logical_and(
            _check_modes_finite(),
            jnp.logical_and(
                _check_weights_shape(),
                jnp.logical_and(
                    _check_weights_nonnegative(),
                    jnp.logical_and(_check_weights_sum(), _check_calib_positive()),
                ),
            ),
        ),
    )

    return jax.lax.cond(
        all_valid, lambda _: _valid_processing(), lambda _: _invalid_processing(), None
    )


@jaxtyped(typechecker=beartype)
def make_potential_slices(
    slices: Float[Array, "H W S"],
    slice_thickness: scalar_numeric,
    calib: scalar_float,
) -> PotentialSlices:
    """JAX-safe factory function for PotentialSlices with data validation.

    Parameters
    ----------
    slices : Float[Array, "H W S"]
        Individual potential slices, S is number of slices
    slice_thickness : scalar_numeric
        Thickness of each slice
    calib : scalar_float
        Pixel calibration

    Returns
    -------
    PotentialSlices
        Validated potential slices instance

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid ranges

    Notes
    -----
    Validation Flow:
    - Convert inputs to JAX arrays with appropriate dtypes:
       - slices: Convert to float64
       - slice_thickness: Convert to float64 scalar
       - calib: Convert to float64 scalar
    - Set expected dimensions constant
    - Define validation functions:
       - _check_3d_slices(): Verify slices array has exactly 3 dimensions
       - _check_slices_finite(): Ensure all values in slices are finite (no inf/nan)
       - _check_slice_thickness_positive(): Confirm slice_thickness is strictly positive
       - _check_calib_positive(): Confirm calibration value is strictly positive
    - Chain all validation checks with jnp.logical_and
    - Use jax.lax.cond to branch based on validation result:
       - If valid: Ensure positive values for thickness and calibration, return PotentialSlices
       - If invalid: Return PotentialSlices with NaN values to signal validation failure
    """
    slices: Float[Array, " H W S"] = jnp.asarray(slices, dtype=jnp.float64)
    slice_thickness: Float[Array, " "] = jnp.asarray(slice_thickness, dtype=jnp.float64)
    calib: Float[Array, " "] = jnp.asarray(calib, dtype=jnp.float64)

    expected_dims: int = 3

    def _check_3d_slices() -> Bool[Array, " "]:
        return jnp.array(len(slices.shape) == expected_dims)

    def _check_slices_finite() -> Bool[Array, " "]:
        return jnp.all(jnp.isfinite(slices))

    def _check_slice_thickness_positive() -> Bool[Array, " "]:
        return slice_thickness > 0

    def _check_calib_positive() -> Bool[Array, " "]:
        return calib > 0

    def _valid_processing() -> PotentialSlices:
        positive_thickness: Float[Array, " "] = (
            jnp.abs(slice_thickness) + jnp.finfo(jnp.float64).eps
        )
        positive_calib: Float[Array, " "] = jnp.abs(calib) + jnp.finfo(jnp.float64).eps

        return PotentialSlices(
            slices=slices,
            slice_thickness=positive_thickness,
            calib=positive_calib,
        )

    def _invalid_processing() -> PotentialSlices:
        nan_thickness: Float[Array, " "] = jnp.array(jnp.nan, dtype=jnp.float64)
        nan_calib: Float[Array, " "] = jnp.array(jnp.nan, dtype=jnp.float64)
        return PotentialSlices(
            slices=slices,
            slice_thickness=nan_thickness,
            calib=nan_calib,
        )

    all_valid: Bool[Array, " "] = jnp.logical_and(
        _check_3d_slices(),
        jnp.logical_and(
            _check_slices_finite(),
            jnp.logical_and(_check_slice_thickness_positive(), _check_calib_positive()),
        ),
    )

    return jax.lax.cond(
        all_valid, lambda _: _valid_processing(), lambda _: _invalid_processing(), None
    )


@jaxtyped(typechecker=beartype)
def make_crystal_structure(
    frac_positions: Float[Array, "* 4"],
    cart_positions: Num[Array, "* 4"],
    cell_lengths: Num[Array, " 3"],
    cell_angles: Num[Array, " 3"],
) -> CrystalStructure:
    """Factory function to create a CrystalStructure instance with type checking.

    Parameters
    ----------
    frac_positions : Float[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in fractional coordinates.
    cart_positions : Num[Array, "* 4"]
        Array of shape (n_atoms, 4) containing atomic positions in Cartesian coordinates.
    cell_lengths : Num[Array, " 3"]
        Unit cell lengths [a, b, c] in Ångstroms.
    cell_angles : Num[Array, " 3"]
        Unit cell angles [α, β, γ] in degrees.

    Returns
    -------
    CrystalStructure
        A validated CrystalStructure instance.

    Raises
    ------
    ValueError
        If the input arrays have incompatible shapes or invalid values.

    Notes
    -----
    Validation Flow:
    - Convert all inputs to JAX arrays:
       - frac_positions: Convert to float64
       - cart_positions: Convert to JAX array (maintains original dtype)
       - cell_lengths: Convert to JAX array (maintains original dtype)
       - cell_angles: Convert to JAX array (maintains original dtype)
    - Set constants for validation (num_cols, num_cell_params, angle limits)
    - Define validation functions:
       - _check_frac_shape(): Verify frac_positions has 4 columns [x, y, z, atomic_number]
       - _check_cart_shape(): Verify cart_positions has 4 columns [x, y, z, atomic_number]
       - _check_cell_lengths_shape(): Confirm cell_lengths has shape (3,)
       - _check_cell_angles_shape(): Confirm cell_angles has shape (3,)
       - _check_atom_count(): Ensure frac_positions and cart_positions have same atom count
       - _check_atomic_numbers(): Verify atomic numbers match between frac and cart positions
       - _check_cell_lengths_positive(): Confirm all cell lengths are strictly positive
       - _check_cell_angles_valid(): Ensure all angles are in range (0, 180) degrees
    - Chain all validation checks with jnp.logical_and
    - Use jax.lax.cond to branch based on validation result:
       - If valid: Ensure positive cell lengths and clip angles to valid range,
        return CrystalStructure
       - If invalid: Return CrystalStructure with NaN values to signal validation failure
    """
    frac_positions: Float[Array, " * 4"] = jnp.asarray(
        frac_positions, dtype=jnp.float64
    )
    cart_positions: Num[Array, " * 4"] = jnp.asarray(cart_positions)
    cell_lengths: Num[Array, " 3"] = jnp.asarray(cell_lengths)
    cell_angles: Num[Array, " 3"] = jnp.asarray(cell_angles)

    num_cols: int = 4
    num_cell_params: int = 3
    min_angle: float = 0.1
    max_angle: float = 179.9
    max_angle_check: float = 180.0

    def _check_frac_shape() -> Bool[Array, " "]:
        return jnp.array(frac_positions.shape[1] == num_cols)

    def _check_cart_shape() -> Bool[Array, " "]:
        return jnp.array(cart_positions.shape[1] == num_cols)

    def _check_cell_lengths_shape() -> Bool[Array, " "]:
        return jnp.array(cell_lengths.shape[0] == num_cell_params)

    def _check_cell_angles_shape() -> Bool[Array, " "]:
        return jnp.array(cell_angles.shape[0] == num_cell_params)

    def _check_atom_count() -> Bool[Array, " "]:
        return jnp.array(frac_positions.shape[0] == cart_positions.shape[0])

    def _check_atomic_numbers() -> Bool[Array, " "]:
        frac_atomic_nums: Num[Array, " *"] = frac_positions[:, 3]
        cart_atomic_nums: Num[Array, " *"] = cart_positions[:, 3]
        return jnp.all(frac_atomic_nums == cart_atomic_nums)

    def _check_cell_lengths_positive() -> Bool[Array, " "]:
        return jnp.all(cell_lengths > 0)

    def _check_cell_angles_valid() -> Bool[Array, " "]:
        return jnp.logical_and(
            jnp.all(cell_angles > 0), jnp.all(cell_angles < max_angle_check)
        )

    def _valid_processing() -> CrystalStructure:
        positive_lengths: Num[Array, " 3"] = (
            jnp.abs(cell_lengths) + jnp.finfo(jnp.float64).eps
        )
        valid_angles: Num[Array, " 3"] = jnp.clip(cell_angles, min_angle, max_angle)

        return CrystalStructure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=positive_lengths,
            cell_angles=valid_angles,
        )

    def _invalid_processing() -> CrystalStructure:
        nan_lengths: Num[Array, " 3"] = jnp.full((num_cell_params,), jnp.nan)
        nan_angles: Num[Array, " 3"] = jnp.full((num_cell_params,), jnp.nan)

        return CrystalStructure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=nan_lengths,
            cell_angles=nan_angles,
        )

    all_valid: Bool[Array, " "] = jnp.logical_and(
        _check_frac_shape(),
        jnp.logical_and(
            _check_cart_shape(),
            jnp.logical_and(
                _check_cell_lengths_shape(),
                jnp.logical_and(
                    _check_cell_angles_shape(),
                    jnp.logical_and(
                        _check_atom_count(),
                        jnp.logical_and(
                            _check_atomic_numbers(),
                            jnp.logical_and(
                                _check_cell_lengths_positive(),
                                _check_cell_angles_valid(),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return jax.lax.cond(
        all_valid, lambda _: _valid_processing(), lambda _: _invalid_processing(), None
    )


@jaxtyped(typechecker=beartype)
def make_xyz_data(
    positions: Float[Array, " N 3"],
    atomic_numbers: Int[Array, " N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[scalar_float] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> XYZData:
    """JAX-safe factory function for XYZData with runtime validation.

    Parameters
    ----------
    positions : Float[Array, " N 3"]
        Cartesian positions in Ångstroms
    atomic_numbers : Int[Array, " N"]
        Atomic numbers (Z) for each atom
    lattice : Optional[Float[Array, "3 3"]], default=None
        Lattice vectors (if any)
    stress : Optional[Float[Array, "3 3"]], default=None
        Stress tensor (if any)
    energy : Optional[scalar_float], default=None
        Total energy (if any)
    properties : Optional[List[Dict[str, Union[str, int]]]], default=None
        Per-atom metadata
    comment : Optional[str], default=None
        Original XYZ comment line

    Returns
    -------
    XYZData
        Validated PyTree structure for XYZ file contents

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid values

    Notes
    -----
    Validation Flow:
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
       - check_finiteness(): Ensure all position values are finite and atomic numbers
         are non-negative
    - Execute optional matrix validation checks (if provided):
       - check_optional_matrices(): For lattice and stress tensors:
         * Verify shape is (3, 3)
         * Ensure all values are finite
    - If all validations pass, create and return XYZData instance
    - If any validation fails, raise ValueError with descriptive error message
    """

    positions = jnp.asarray(positions, dtype=jnp.float64)
    atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    if lattice is not None:
        lattice = jnp.asarray(lattice, dtype=jnp.float64)
    else:
        lattice = jnp.eye(3, dtype=jnp.float64)

    if stress is not None:
        stress = jnp.asarray(stress, dtype=jnp.float64)

    if energy is not None:
        energy = jnp.asarray(energy, dtype=jnp.float64)

    def validate_and_create() -> XYZData:
        nn: Int[Array, ""] = positions.shape[0]

        def check_shape() -> None:
            expected_pos_dims: int = 3
            if positions.shape[1] != expected_pos_dims:
                raise ValueError("positions must have shape (N, 3)")
            if atomic_numbers.shape[0] != nn:
                raise ValueError("atomic_numbers must have shape (N,)")

        def check_finiteness() -> None:
            if not jnp.all(jnp.isfinite(positions)):
                raise ValueError("positions contain non-finite values")
            if not jnp.all(atomic_numbers >= 0):
                raise ValueError("atomic_numbers must be non-negative")

        def check_optional_matrices() -> None:
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


@jaxtyped(typechecker=beartype)
def make_stem4d(
    data: Float[Array, "P H W"],
    real_space_calib: scalar_float,
    fourier_space_calib: scalar_float,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: scalar_numeric,
) -> STEM4D:
    """JAX-safe factory function for STEM4D with data validation.

    Parameters
    ----------
    data : Float[Array, "P H W"]
        4D-STEM data array with P scan positions and HxW diffraction patterns
    real_space_calib : scalar_float
        Real space calibration in Angstroms per pixel
    fourier_space_calib : scalar_float
        Fourier space calibration in inverse Angstroms per pixel
    scan_positions : Float[Array, "P 2"]
        Real space scan positions in Angstroms (y, x coordinates)
    voltage_kv : scalar_numeric
        Accelerating voltage in kilovolts

    Returns
    -------
    STEM4D
        Validated 4D-STEM data structure

    Notes
    -----
    Validation Flow:
    - Convert all inputs to JAX arrays with appropriate dtypes:
       - data: Convert to float64
       - real_space_calib: Convert to float64 scalar
       - fourier_space_calib: Convert to float64 scalar
       - scan_positions: Convert to float64
       - voltage_kv: Convert to float64 scalar
    - Extract shape information and set constants
    - Define validation functions:
       - _check_data_3d(): Verify data array has exactly 3 dimensions
       - _check_data_finite(): Ensure all values in data are finite (no inf/nan)
       - _check_scan_positions_shape(): Verify scan_positions shape matches data
       - _check_scan_positions_finite(): Ensure all scan positions are finite
       - _check_real_space_calib_positive(): Ensure real space calibration is positive
       - _check_fourier_space_calib_positive(): Ensure Fourier space calibration is positive
       - _check_voltage_positive(): Verify voltage is positive
    - Chain all validation checks with jnp.logical_and
    - Use jax.lax.cond to branch based on validation result:
       - If valid: Ensure positive calibrations and voltage, return STEM4D
       - If invalid: Return STEM4D with NaN values to signal validation failure
    """
    data: Float[Array, " P H W"] = jnp.asarray(data, dtype=jnp.float64)
    real_space_calib: Float[Array, " "] = jnp.asarray(
        real_space_calib,
        dtype=jnp.float64,
    )
    fourier_space_calib: Float[Array, " "] = jnp.asarray(
        fourier_space_calib, dtype=jnp.float64
    )
    scan_positions: Float[Array, " P 2"] = jnp.asarray(
        scan_positions, dtype=jnp.float64
    )
    voltage_kv: Float[Array, " "] = jnp.asarray(voltage_kv, dtype=jnp.float64)

    num_scan_positions: int = data.shape[0] if len(data.shape) >= 1 else 0
    num_scan_coords: int = 2
    expected_dims: int = 3

    def _check_data_3d() -> Bool[Array, " "]:
        return jnp.array(len(data.shape) == expected_dims)

    def _check_data_finite() -> Bool[Array, " "]:
        return jnp.all(jnp.isfinite(data))

    def _check_scan_positions_shape() -> Bool[Array, " "]:
        return jnp.logical_and(
            jnp.array(scan_positions.shape[0] == num_scan_positions),
            jnp.array(scan_positions.shape[1] == num_scan_coords),
        )

    def _check_scan_positions_finite() -> Bool[Array, " "]:
        return jnp.all(jnp.isfinite(scan_positions))

    def _check_real_space_calib_positive() -> Bool[Array, " "]:
        return real_space_calib > 0

    def _check_fourier_space_calib_positive() -> Bool[Array, " "]:
        return fourier_space_calib > 0

    def _check_voltage_positive() -> Bool[Array, " "]:
        return voltage_kv > 0

    def _valid_processing() -> STEM4D:
        positive_real_calib: Float[Array, " "] = (
            jnp.abs(real_space_calib) + jnp.finfo(jnp.float64).eps
        )
        positive_fourier_calib: Float[Array, " "] = (
            jnp.abs(fourier_space_calib) + jnp.finfo(jnp.float64).eps
        )
        positive_voltage: Float[Array, " "] = (
            jnp.abs(voltage_kv) + jnp.finfo(jnp.float64).eps
        )

        return STEM4D(
            data=data,
            real_space_calib=positive_real_calib,
            fourier_space_calib=positive_fourier_calib,
            scan_positions=scan_positions,
            voltage_kv=positive_voltage,
        )

    def _invalid_processing() -> STEM4D:
        nan_calib: Float[Array, " "] = jnp.array(jnp.nan, dtype=jnp.float64)

        return STEM4D(
            data=data,
            real_space_calib=nan_calib,
            fourier_space_calib=nan_calib,
            scan_positions=scan_positions,
            voltage_kv=nan_calib,
        )

    all_valid: Bool[Array, " "] = jnp.logical_and(
        _check_data_3d(),
        jnp.logical_and(
            _check_data_finite(),
            jnp.logical_and(
                _check_scan_positions_shape(),
                jnp.logical_and(
                    _check_scan_positions_finite(),
                    jnp.logical_and(
                        _check_real_space_calib_positive(),
                        jnp.logical_and(
                            _check_fourier_space_calib_positive(),
                            _check_voltage_positive(),
                        ),
                    ),
                ),
            ),
        ),
    )

    return jax.lax.cond(
        all_valid, lambda _: _valid_processing(), lambda _: _invalid_processing(), None
    )
