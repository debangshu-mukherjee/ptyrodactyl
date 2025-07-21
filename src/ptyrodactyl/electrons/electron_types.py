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

Classes
-------
- `CalibratedArray`:
    A named tuple for calibrated array data with spatial calibration
- `ProbeModes`:
    A named tuple for multimodal electron probe state
- `PotentialSlices`:
    A named tuple for potential slices in multi-slice simulations
- `CrystalStructure`:
    A named tuple for crystal structure with fractional and Cartesian coordinates
- `XYZData`:
    A named tuple for XYZ file data with atomic positions, lattice vectors,
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
    - `calib_y` (scalar_float):
        Calibration in y direction
    - `calib_x` (scalar_float):
        Calibration in x direction
    - `real_space` (Bool[Array, ""]):
        Whether the array is in real space.
        If False, it is in reciprocal space.
    """

    data_array: Union[Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]]
    calib_y: scalar_float
    calib_x: scalar_float
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
    - `calib` (scalar_float):
        Pixel Calibration
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, "M"]
    calib: scalar_float

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
    PyTree structure for multimodal electron probe state.

    Attributes
    ----------
    - `slices` (Complex[Array, "H W S"]):
        Individual potential slices.
        S is number of slices
    - `slice_thickness` (scalar_numeric):
        Mode occupation numbers
    - `calib` (scalar_float):
        Pixel Calibration
    """

    slices: Complex[Array, "H W S"]
    slice_thickness: scalar_numeric
    calib: scalar_float

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
    - `energy` (Optional[float]):
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
    energy: Optional[float]
    properties: Optional[List[Dict[str, Union[str, int]]]]
    comment: Optional[str]

    def tree_flatten(self):
        children = (
            self.positions,
            self.atomic_numbers,
            self.lattice,
            self.stress,
            self.energy,
            self.properties,
            self.comment,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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
    """
    # Convert data_array to appropriate dtype based on input type
    if jnp.issubdtype(data_array.dtype, jnp.integer):
        data_array = jnp.asarray(data_array, dtype=jnp.int32)
    elif jnp.issubdtype(data_array.dtype, jnp.floating):
        data_array = jnp.asarray(data_array, dtype=jnp.float64)
    elif jnp.issubdtype(data_array.dtype, jnp.complexfloating):
        data_array = jnp.asarray(data_array, dtype=jnp.complex128)
    else:
        data_array = jnp.asarray(data_array)

    calib_y = jnp.asarray(calib_y, dtype=jnp.float64)
    calib_x = jnp.asarray(calib_x, dtype=jnp.float64)
    real_space = jnp.asarray(real_space, dtype=jnp.bool_)

    def validate_and_create():
        def check_2d_array():
            return lax.cond(
                data_array.ndim == 2,
                lambda: data_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: data_array, lambda: data_array)
                ),
            )

        def check_array_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(data_array)),
                lambda: data_array,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: data_array, lambda: data_array)
                ),
            )

        def check_calib_y():
            return lax.cond(
                calib_y > 0,
                lambda: calib_y,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib_y, lambda: calib_y)
                ),
            )

        def check_calib_x():
            return lax.cond(
                calib_x > 0,
                lambda: calib_x,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib_x, lambda: calib_x)
                ),
            )

        def check_real_space():
            return lax.cond(
                real_space.ndim == 0,
                lambda: real_space,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: real_space, lambda: real_space)
                ),
            )

        check_2d_array()
        check_array_finite()
        check_calib_y()
        check_calib_x()
        check_real_space()

        return CalibratedArray(
            data_array=data_array,
            calib_y=calib_y,
            calib_x=calib_x,
            real_space=real_space,
        )

    return validate_and_create()


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

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate modes array:
        - Check it's 3D with shape (H, W, M)
        - Ensure all values are finite
    - Validate weights array:
        - Check it's 1D with length M
        - Ensure all values are non-negative
        - Ensure sum is positive
    - Validate calibration:
        - Check calib is positive
    - Create and return ProbeModes instance
    """
    modes = jnp.asarray(modes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    calib = jnp.asarray(calib, dtype=jnp.float64)

    def validate_and_create():
        H, W, M = modes.shape

        def check_3d_modes():
            return lax.cond(
                modes.ndim == 3,
                lambda: modes,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: modes, lambda: modes)
                ),
            )

        def check_modes_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(modes)),
                lambda: modes,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: modes, lambda: modes)
                ),
            )

        def check_weights_shape():
            return lax.cond(
                weights.shape == (M,),
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_weights_nonnegative():
            return lax.cond(
                jnp.all(weights >= 0),
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_weights_sum():
            return lax.cond(
                jnp.sum(weights) > 0,
                lambda: weights,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: weights, lambda: weights)
                ),
            )

        def check_calib():
            return lax.cond(
                calib > 0,
                lambda: calib,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib, lambda: calib)
                ),
            )

        check_3d_modes()
        check_modes_finite()
        check_weights_shape()
        check_weights_nonnegative()
        check_weights_sum()
        check_calib()

        return ProbeModes(
            modes=modes,
            weights=weights,
            calib=calib,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_potential_slices(
    slices: Complex[Array, "H W S"],
    slice_thickness: scalar_numeric,
    calib: scalar_float,
) -> PotentialSlices:
    """
    Description
    -----------
    JAX-safe factory function for PotentialSlices with data validation.

    Parameters
    ----------
    - `slices` (Complex[Array, "H W S"]):
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

    Flow
    ----
    - Convert inputs to JAX arrays
    - Validate slices array:
        - Check it's 3D with shape (H, W, S)
        - Ensure all values are finite
    - Validate slice_thickness:
        - Check it's positive
    - Validate calibration:
        - Check calib is positive
    - Create and return PotentialSlices instance
    """
    slices = jnp.asarray(slices, dtype=jnp.complex128)
    slice_thickness = jnp.asarray(slice_thickness, dtype=jnp.float64)
    calib = jnp.asarray(calib, dtype=jnp.float64)

    def validate_and_create():
        H, W, S = slices.shape

        def check_3d_slices():
            return lax.cond(
                slices.ndim == 3,
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_slices_finite():
            return lax.cond(
                jnp.all(jnp.isfinite(slices)),
                lambda: slices,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slices, lambda: slices)
                ),
            )

        def check_slice_thickness():
            return lax.cond(
                slice_thickness > 0,
                lambda: slice_thickness,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: slice_thickness, lambda: slice_thickness)
                ),
            )

        def check_calib():
            return lax.cond(
                calib > 0,
                lambda: calib,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: calib, lambda: calib)
                ),
            )

        check_3d_slices()
        check_slices_finite()
        check_slice_thickness()
        check_calib()

        return PotentialSlices(
            slices=slices,
            slice_thickness=slice_thickness,
            calib=calib,
        )

    return validate_and_create()


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

    Flow
    ----
    - Convert all inputs to JAX arrays using jnp.asarray
    - Validate shape of frac_positions is (n_atoms, 4)
    - Validate shape of cart_positions is (n_atoms, 4)
    - Validate shape of cell_lengths is (3,)
    - Validate shape of cell_angles is (3,)
    - Verify number of atoms matches between frac and cart positions
    - Verify atomic numbers match between frac and cart positions
    - Ensure cell lengths are positive
    - Ensure cell angles are between 0 and 180 degrees
    - Create and return CrystalStructure instance with validated data
    """
    frac_positions = jnp.asarray(frac_positions)
    cart_positions = jnp.asarray(cart_positions)
    cell_lengths = jnp.asarray(cell_lengths)
    cell_angles = jnp.asarray(cell_angles)

    def validate_and_create():
        def check_frac_shape():
            return lax.cond(
                frac_positions.shape[1] == 4,
                lambda: frac_positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: frac_positions, lambda: frac_positions)
                ),
            )

        def check_cart_shape():
            return lax.cond(
                cart_positions.shape[1] == 4,
                lambda: cart_positions,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cart_positions, lambda: cart_positions)
                ),
            )

        def check_cell_lengths_shape():
            return lax.cond(
                cell_lengths.shape == (3,),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def check_cell_angles_shape():
            return lax.cond(
                cell_angles.shape == (3,),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        def check_atom_count():
            return lax.cond(
                frac_positions.shape[0] == cart_positions.shape[0],
                lambda: (frac_positions, cart_positions),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (frac_positions, cart_positions),
                        lambda: (frac_positions, cart_positions),
                    )
                ),
            )

        def check_atomic_numbers():
            return lax.cond(
                jnp.all(frac_positions[:, 3] == cart_positions[:, 3]),
                lambda: (frac_positions, cart_positions),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: (frac_positions, cart_positions),
                        lambda: (frac_positions, cart_positions),
                    )
                ),
            )

        def check_cell_lengths_positive():
            return lax.cond(
                jnp.all(cell_lengths > 0),
                lambda: cell_lengths,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_lengths, lambda: cell_lengths)
                ),
            )

        def check_cell_angles_valid():
            return lax.cond(
                jnp.all(jnp.logical_and(cell_angles > 0, cell_angles < 180)),
                lambda: cell_angles,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: cell_angles, lambda: cell_angles)
                ),
            )

        check_frac_shape()
        check_cart_shape()
        check_cell_lengths_shape()
        check_cell_angles_shape()
        check_atom_count()
        check_atomic_numbers()
        check_cell_lengths_positive()
        check_cell_angles_valid()
        return CrystalStructure(
            frac_positions=frac_positions,
            cart_positions=cart_positions,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

    return validate_and_create()


@jaxtyped(typechecker=beartype)
def make_xyz_data(
    positions: Float[Array, "N 3"],
    atomic_numbers: Int[Array, "N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[float] = None,
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
    - `energy` (Optional[float]):
        Total energy (if any)
    - `properties` (Optional[List[Dict[str, Union[str, int]]]]):
        Per-atom metadata
    - `comment` (Optional[str]):
        Original XYZ comment line

    Returns
    -------
    - `XYZData`:
        Validated PyTree structure for XYZ file contents
    """

    positions = jnp.asarray(positions, dtype=jnp.float64)
    atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)

    if lattice is not None:
        lattice = jnp.asarray(lattice, dtype=jnp.float64)

    if stress is not None:
        stress = jnp.asarray(stress, dtype=jnp.float64)

    def validate_and_create():
        N = positions.shape[0]

        def check_shape():
            lax.cond(
                positions.shape[1] == 3,
                lambda: True,
                lambda: lax.stop_gradient(
                    (_ for _ in ()).throw(
                        ValueError("positions must have shape (N, 3)")
                    )
                ),
            )
            lax.cond(
                atomic_numbers.shape[0] == N,
                lambda: True,
                lambda: lax.stop_gradient(
                    (_ for _ in ()).throw(
                        ValueError("atomic_numbers must have shape (N,)")
                    )
                ),
            )

        def check_finiteness():
            lax.cond(
                jnp.all(jnp.isfinite(positions)),
                lambda: True,
                lambda: lax.stop_gradient(
                    (_ for _ in ()).throw(
                        ValueError("positions contain non-finite values")
                    )
                ),
            )
            lax.cond(
                jnp.all(atomic_numbers >= 0),
                lambda: True,
                lambda: lax.stop_gradient(
                    (_ for _ in ()).throw(
                        ValueError("atomic_numbers must be non-negative")
                    )
                ),
            )

        def check_optional_matrices():
            if lattice is not None:
                lax.cond(
                    lattice.shape == (3, 3),
                    lambda: True,
                    lambda: lax.stop_gradient(
                        (_ for _ in ()).throw(
                            ValueError("lattice must have shape (3, 3)")
                        )
                    ),
                )
                lax.cond(
                    jnp.all(jnp.isfinite(lattice)),
                    lambda: True,
                    lambda: lax.stop_gradient(
                        (_ for _ in ()).throw(
                            ValueError("lattice contains non-finite values")
                        )
                    ),
                )
            if stress is not None:
                lax.cond(
                    stress.shape == (3, 3),
                    lambda: True,
                    lambda: lax.stop_gradient(
                        (_ for _ in ()).throw(
                            ValueError("stress must have shape (3, 3)")
                        )
                    ),
                )
                lax.cond(
                    jnp.all(jnp.isfinite(stress)),
                    lambda: True,
                    lambda: lax.stop_gradient(
                        (_ for _ in ()).throw(
                            ValueError("stress contains non-finite values")
                        )
                    ),
                )

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
