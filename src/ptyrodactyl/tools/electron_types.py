"""Data structures and type definitions for electron microscopy.

Extended Summary
----------------
Provides JAX-compatible PyTree structures for electron ptychography
data including calibrated arrays, probe modes, potential slices, and
4D-STEM datasets. All structures are registered as JAX PyTree nodes
and support JAX transformations such as ``jit``, ``grad``, and
``vmap``.

Routine Listings
----------------
:data:`ScalarNumeric`
    Numeric types (int, float, or 0-dimensional Num array).
:data:`ScalarFloat`
    Float or 0-dimensional Float array.
:data:`ScalarInt`
    Int or 0-dimensional Int array.
:data:`NonJaxNumber`
    Non-JAX numeric types (int, float).
:class:`CalibratedArray`
    Calibrated array data with spatial calibration.
:class:`ProbeModes`
    Multimodal electron probe state.
:class:`PotentialSlices`
    Potential slices for multi-slice simulations.
:class:`CrystalStructure`
    Crystal structure with fractional and Cartesian coordinates.
:class:`CrystalData`
    Crystal data with atomic positions, lattice vectors,
    and metadata.
:class:`STEM4D`
    4D-STEM data with diffraction patterns, calibrations,
    and parameters.
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
    """PyTree structure for a calibrated array.

    Extended Summary
    ----------------
    Stores a 2-D data array together with per-axis spatial
    calibrations and a flag indicating whether the data lives in
    real space or reciprocal space. Registered as a JAX PyTree
    node so it can be passed through ``jit``, ``grad``, and
    ``vmap``. All fields are children (traced by JAX); there is
    no auxiliary data.

    Attributes
    ----------
    data_array : Union[Int[Array, "H W"], Float[Array, "H W"], \
Complex[Array, "H W"]]
        The actual array data.
    calib_y : Float[Array, " "]
        Calibration in the y direction, in Angstroms per pixel.
    calib_x : Float[Array, " "]
        Calibration in the x direction, in Angstroms per pixel.
    real_space : Bool[Array, " "]
        Whether the array is in real space. If ``False``, the
        array is in reciprocal space.

    Notes
    -----
    Because every field is a JAX array, the entire structure is
    differentiable and JIT-compatible.

    See Also
    --------
    :func:`~ptyrodactyl.tools.factory.make_calibrated_array`
        Factory function with runtime validation.
    """

    data_array: Union[
        Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]
    ]
    calib_y: Float[Array, " "]
    calib_x: Float[Array, " "]
    real_space: Bool[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        """Flatten CalibratedArray for JAX pytree serialization.

        Returns
        -------
        children : Tuple[Any, ...]
            All fields (``data_array``, ``calib_y``, ``calib_x``,
            ``real_space``) as traced children.
        aux_data : None
            No auxiliary (static) data.
        """
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
        """Reconstruct CalibratedArray from flattened pytree.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : Tuple[Any, ...]
            Flattened child arrays.

        Returns
        -------
        calibrated_array : CalibratedArray
            Reconstructed instance.
        """
        return cls(*children)


@register_pytree_node_class
class ProbeModes(NamedTuple):
    """PyTree structure for multimodal electron probe state.

    Extended Summary
    ----------------
    Holds a set of complex-valued probe modes with associated
    occupation weights and a pixel calibration. Registered as a
    JAX PyTree node; all fields are children (traced by JAX) and
    there is no auxiliary data.

    Attributes
    ----------
    modes : Complex[Array, "H W M"]
        Complex probe modes where M is the number of modes.
    weights : Float[Array, " M"]
        Mode occupation numbers, normalised to sum to one.
    calib : Float[Array, " "]
        Pixel calibration in Angstroms per pixel.

    Notes
    -----
    Weights are normalised during construction via
    :func:`~ptyrodactyl.tools.factory.make_probe_modes`.

    See Also
    --------
    :func:`~ptyrodactyl.tools.factory.make_probe_modes`
        Factory function with runtime validation.
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, " M"]
    calib: Float[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        """Flatten ProbeModes for JAX pytree serialization.

        Returns
        -------
        children : Tuple[Any, ...]
            All fields (``modes``, ``weights``, ``calib``) as
            traced children.
        aux_data : None
            No auxiliary (static) data.
        """
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
        """Reconstruct ProbeModes from flattened pytree.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : Tuple[Any, ...]
            Flattened child arrays.

        Returns
        -------
        probe_modes : ProbeModes
            Reconstructed instance.
        """
        return cls(*children)


@register_pytree_node_class
class PotentialSlices(NamedTuple):
    """PyTree structure for multiple potential slices.

    Extended Summary
    ----------------
    Stores a stack of 2-D electrostatic potential slices used in
    multi-slice electron scattering simulations. Registered as a
    JAX PyTree node; all fields are children (traced by JAX) and
    there is no auxiliary data.

    Attributes
    ----------
    slices : Float[Array, "H W S"]
        Individual potential slices where S is the number of
        slices.
    slice_thickness : Num[Array, " "]
        Thickness of each slice in Angstroms.
    calib : Float[Array, " "]
        Pixel calibration in Angstroms per pixel.

    Notes
    -----
    Slice thickness and calibration are enforced to be positive
    by the factory function.

    See Also
    --------
    :func:`~ptyrodactyl.tools.factory.make_potential_slices`
        Factory function with runtime validation.
    """

    slices: Float[Array, "H W S"]
    slice_thickness: Num[Array, " "]
    calib: Float[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        """Flatten PotentialSlices for JAX pytree serialization.

        Returns
        -------
        children : Tuple[Any, ...]
            All fields (``slices``, ``slice_thickness``,
            ``calib``) as traced children.
        aux_data : None
            No auxiliary (static) data.
        """
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
        """Reconstruct PotentialSlices from flattened pytree.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : Tuple[Any, ...]
            Flattened child arrays.

        Returns
        -------
        potential_slices : PotentialSlices
            Reconstructed instance.
        """
        return cls(*children)


@register_pytree_node_class
class CrystalStructure(NamedTuple):
    """Crystal structure with fractional and Cartesian coordinates.

    Extended Summary
    ----------------
    A JAX-compatible data structure representing a crystal
    structure that stores both fractional and Cartesian atomic
    coordinates together with unit-cell parameters. Registered
    as a JAX PyTree node; all fields are children (traced by
    JAX) and there is no auxiliary data.

    Attributes
    ----------
    frac_positions : Float[Array, "* 4"]
        Array of shape ``(n_atoms, 4)`` containing atomic
        positions in fractional coordinates. Each row is
        ``[x, y, z, atomic_number]`` where x, y, z are in the
        range [0, 1].
    cart_positions : Num[Array, "* 4"]
        Array of shape ``(n_atoms, 4)`` containing atomic
        positions in Cartesian coordinates. Each row is
        ``[x, y, z, atomic_number]`` where x, y, z are in
        Angstroms.
    cell_lengths : Num[Array, " 3"]
        Unit cell lengths ``[a, b, c]`` in Angstroms.
    cell_angles : Num[Array, " 3"]
        Unit cell angles ``[alpha, beta, gamma]`` in degrees.

    Notes
    -----
    All relevant data is stored in JAX arrays, so ``tree_flatten``
    returns ``None`` as auxiliary data. This makes the structure
    fully compatible with ``jit``, ``grad``, and ``vmap``.

    See Also
    --------
    :func:`~ptyrodactyl.tools.factory.make_crystal_structure`
        Factory function with runtime validation.
    """

    frac_positions: Float[Array, "* 4"]
    cart_positions: Num[Array, "* 4"]
    cell_lengths: Num[Array, " 3"]
    cell_angles: Num[Array, " 3"]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        """Flatten CrystalStructure for JAX pytree serialization.

        Returns
        -------
        children : Tuple[Any, ...]
            All fields (``frac_positions``, ``cart_positions``,
            ``cell_lengths``, ``cell_angles``) as traced children.
        aux_data : None
            No auxiliary (static) data.
        """
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
        """Reconstruct CrystalStructure from flattened pytree.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : Tuple[Any, ...]
            Flattened child arrays.

        Returns
        -------
        crystal_structure : CrystalStructure
            Reconstructed instance.
        """
        return cls(*children)


@register_pytree_node_class
class CrystalData(NamedTuple):
    """JAX-compatible PyTree representing crystal structure data.

    Extended Summary
    ----------------
    Supports data from XYZ files, VASP POSCAR/CONTCAR files, and
    other crystal structure formats. Registered as a JAX PyTree
    node. JAX-traced children are the numeric arrays
    (``positions``, ``atomic_numbers``, ``lattice``, ``stress``,
    ``energy``); the ``properties`` and ``comment`` fields are
    stored as auxiliary (static) data because they are not JAX
    arrays.

    Attributes
    ----------
    positions : Float[Array, " N 3"]
        Cartesian positions in Angstroms.
    atomic_numbers : Int[Array, " N"]
        Atomic numbers (Z) corresponding to each atom.
    lattice : Optional[Float[Array, "3 3"]]
        Lattice vectors in Angstroms, or ``None``.
    stress : Optional[Float[Array, "3 3"]]
        Symmetric stress tensor, or ``None``.
    energy : Optional[ScalarFloat]
        Total energy in eV, or ``None``.
    properties : Optional[List[Dict[str, Union[str, int]]]]
        List of per-atom property dictionaries from metadata.
    comment : Optional[str]
        The raw comment line from the file.

    Notes
    -----
    ``properties`` and ``comment`` are Python objects and cannot
    be traced by JAX. They are carried as auxiliary data in
    ``tree_flatten`` / ``tree_unflatten``.

    See Also
    --------
    :func:`~ptyrodactyl.tools.factory.make_crystal_data`
        Factory function with runtime validation.
    """

    positions: Float[Array, " N 3"]
    atomic_numbers: Int[Array, " N"]
    lattice: Optional[Float[Array, "3 3"]]
    stress: Optional[Float[Array, "3 3"]]
    energy: Optional[Float[Array, " "]]
    properties: Optional[List[Dict[str, Union[str, int]]]]
    comment: Optional[str]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], dict[str, Any]]:
        """Flatten CrystalData for JAX pytree serialization.

        Children (traced by JAX) are ``positions``,
        ``atomic_numbers``, ``lattice``, ``stress``, and
        ``energy``. Auxiliary (static) data are ``properties``
        and ``comment``.

        Returns
        -------
        children : Tuple[Any, ...]
            Numeric array fields traced by JAX.
        aux_data : Dict[str, Any]
            Non-array fields (``properties``, ``comment``).
        """
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
    ) -> "CrystalData":
        """Reconstruct CrystalData from flattened pytree.

        Parameters
        ----------
        aux_data : Dict[str, Any]
            Auxiliary data containing ``properties`` and
            ``comment``.
        children : Tuple[Any, ...]
            Numeric array children.

        Returns
        -------
        crystal_data : CrystalData
            Reconstructed instance.
        """
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
    """PyTree structure for 4D-STEM data.

    Extended Summary
    ----------------
    Contains diffraction patterns at multiple scan positions with
    associated real-space and Fourier-space calibrations and the
    accelerating voltage. Registered as a JAX PyTree node; all
    fields are children (traced by JAX) and there is no auxiliary
    data.

    Attributes
    ----------
    data : Float[Array, "P H W"]
        4D-STEM data array where P is the number of scan
        positions and H, W are the diffraction pattern
        dimensions.
    real_space_calib : Float[Array, " "]
        Real space calibration in Angstroms per pixel.
    fourier_space_calib : Float[Array, " "]
        Fourier space calibration in inverse Angstroms per
        pixel.
    scan_positions : Float[Array, "P 2"]
        Real space scan positions in Angstroms as (y, x)
        coordinates.
    voltage_kv : Float[Array, " "]
        Accelerating voltage in kilovolts.

    Notes
    -----
    All fields are JAX arrays, so the entire structure is
    differentiable and JIT-compatible.

    See Also
    --------
    :func:`~ptyrodactyl.tools.factory.make_stem4d`
        Factory function with runtime validation.
    """

    data: Float[Array, "P H W"]
    real_space_calib: Float[Array, " "]
    fourier_space_calib: Float[Array, " "]
    scan_positions: Float[Array, "P 2"]
    voltage_kv: Float[Array, " "]

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        """Flatten STEM4D for JAX pytree serialization.

        Returns
        -------
        children : Tuple[Any, ...]
            All fields (``data``, ``real_space_calib``,
            ``fourier_space_calib``, ``scan_positions``,
            ``voltage_kv``) as traced children.
        aux_data : None
            No auxiliary (static) data.
        """
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
        """Reconstruct STEM4D from flattened pytree.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : Tuple[Any, ...]
            Flattened child arrays.

        Returns
        -------
        stem4d : STEM4D
            Reconstructed instance.
        """
        return cls(*children)
