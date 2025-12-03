"""Data structures and type definitions for electron microscopy.

Extended Summary
----------------
Provides JAX-compatible PyTree structures for electron ptychography data
including calibrated arrays, probe modes, potential slices, and 4D-STEM
datasets. All structures support JAX transformations.

Routine Listings
----------------
ScalarNumeric : TypeAlias
    Numeric types (int, float, or 0-dimensional Num array).
ScalarFloat : TypeAlias
    Float or 0-dimensional Float array.
ScalarInt : TypeAlias
    Int or 0-dimensional Int array.
NonJaxNumber : TypeAlias
    Non-JAX numeric types (int, float).
CalibratedArray : PyTree
    Calibrated array data with spatial calibration.
ProbeModes : PyTree
    Multimodal electron probe state.
PotentialSlices : PyTree
    Potential slices for multi-slice simulations.
CrystalStructure : PyTree
    Crystal structure with fractional and Cartesian coordinates.
XYZData : PyTree
    XYZ file data with atomic positions, lattice vectors, and metadata.
STEM4D : PyTree
    4D-STEM data with diffraction patterns, calibrations, and parameters.
"""

import jax
from beartype.typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Complex, Float, Int, Num

jax.config.update("jax_enable_x64", True)

ScalarNumeric: TypeAlias = Union[int, float, Num[Array, " "]]
ScalarFloat: TypeAlias = Union[float, Float[Array, " "]]
ScalarInt: TypeAlias = Union[int, Int[Array, " "]]
NonJaxNumber: TypeAlias = Union[int, float]


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

    data_array: Union[
        Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]
    ]
    calib_y: Float[Array, " "]
    calib_x: Float[Array, " "]
    real_space: Bool[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        """Flatten CalibratedArray for JAX pytree serialization."""
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
        """Reconstruct CalibratedArray from flattened pytree data."""
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
        """Flatten ProbeModes for JAX pytree serialization."""
        return (
            (
                self.modes,
                self.weights,
                self.calib,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, _aux_data: None, children: Tuple[Any, ...]
    ) -> "ProbeModes":
        """Reconstruct ProbeModes from flattened pytree data."""
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
        """Flatten PotentialSlices for JAX pytree serialization."""
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
        """Reconstruct PotentialSlices from flattened pytree data."""
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
        """Flatten CrystalStructure for JAX pytree serialization."""
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
        """Reconstruct CrystalStructure from flattened pytree data."""
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
    energy : Optional[ScalarFloat]
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
        """Flatten XYZData for JAX pytree serialization."""
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
        """Reconstruct XYZData from flattened pytree data."""
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
        """Flatten STEM4D for JAX pytree serialization."""
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
    def tree_unflatten(
        cls, _aux_data: None, children: Tuple[Any, ...]
    ) -> "STEM4D":
        """Reconstruct STEM4D from flattened pytree data."""
        return cls(*children)
