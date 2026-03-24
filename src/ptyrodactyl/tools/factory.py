"""Factory functions for validating data before PyTree loading.

Extended Summary
----------------
Provides JAX-safe functional data validation before loading data
into PyTrees. It is recommended to use these functions to access
PyTrees rather than instantiating them directly.

Routine Listings
----------------
:func:`make_calibrated_array`
    Create a :class:`CalibratedArray` with runtime type
    checking.
:func:`make_probe_modes`
    Create a :class:`ProbeModes` with runtime type checking.
:func:`make_potential_slices`
    Create a :class:`PotentialSlices` with runtime type
    checking.
:func:`make_crystal_structure`
    Create a :class:`CrystalStructure` with runtime type
    checking.
:func:`make_crystal_data`
    Create a :class:`CrystalData` with runtime type checking.
:func:`make_stem4d`
    Create a :class:`STEM4D` with runtime type checking.

Notes
-----
Always use factory functions instead of directly instantiating
NamedTuple classes to ensure proper runtime type checking of the
contents.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

from .electron_types import (
    STEM4D,
    CalibratedArray,
    CrystalData,
    CrystalStructure,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarNumeric,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
@jax.jit
def make_calibrated_array(
    data_array: Union[
        Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]
    ],
    calib_y: ScalarFloat,
    calib_x: ScalarFloat,
    real_space: Union[bool, Bool[Array, " "]],
) -> CalibratedArray:
    """Create a validated :class:`CalibratedArray`.

    Extended Summary
    ----------------
    JIT-compiled factory that casts inputs to JAX arrays,
    ensures calibrations are strictly positive, and returns
    a fully validated :class:`CalibratedArray` PyTree.

    Implementation Logic
    --------------------
    1. **Cast inputs** --
       Convert *data_array*, calibrations, and *real_space*
       to JAX arrays with appropriate dtypes.
    2. **Enforce positive calibrations** --
       Apply ``abs() + eps`` to guarantee positivity.
    3. **Construct PyTree** --
       Return a :class:`CalibratedArray`.

    Parameters
    ----------
    data_array : Union[Int[Array, "H W"], Float[Array, "H W"], \
Complex[Array, "H W"]]
        The 2-D array data.
    calib_y : ScalarFloat
        Calibration in the y direction, in Angstroms per pixel.
    calib_x : ScalarFloat
        Calibration in the x direction, in Angstroms per pixel.
    real_space : Union[bool, Bool[Array, " "]]
        Whether the array is in real space.

    Returns
    -------
    calibrated_array : CalibratedArray
        Validated calibrated array instance.

    Raises
    ------
    ValueError
        If data is invalid or parameters are out of valid
        ranges (caught by jaxtyping/beartype).

    See Also
    --------
    :class:`~ptyrodactyl.tools.electron_types.CalibratedArray`
        Target PyTree class.
    """
    data_arr: Union[
        Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]
    ] = jnp.asarray(data_array)

    calib_y_arr: Float[Array, " "] = jnp.asarray(calib_y, dtype=jnp.float64)
    calib_x_arr: Float[Array, " "] = jnp.asarray(calib_x, dtype=jnp.float64)
    real_space_arr: Bool[Array, " "] = jnp.asarray(real_space, dtype=jnp.bool_)

    # Ensure calibrations are positive using JAX operations
    calib_y_pos: Float[Array, " "] = (
        jnp.abs(calib_y_arr) + jnp.finfo(jnp.float64).eps
    )
    calib_x_pos: Float[Array, " "] = (
        jnp.abs(calib_x_arr) + jnp.finfo(jnp.float64).eps
    )

    return CalibratedArray(
        data_array=data_arr,
        calib_y=calib_y_pos,
        calib_x=calib_x_pos,
        real_space=real_space_arr,
    )


@jaxtyped(typechecker=beartype)
@jax.jit
def make_probe_modes(
    modes: Complex[Array, "H W M"],
    weights: Float[Array, " M"],
    calib: ScalarFloat,
) -> ProbeModes:
    """Create a validated :class:`ProbeModes`.

    Extended Summary
    ----------------
    JIT-compiled factory that validates mode shapes,
    finiteness, weight non-negativity, and calibration
    positivity. Normalises weights to sum to one.

    Implementation Logic
    --------------------
    1. **Cast inputs** --
       Convert modes, weights, and calibration to float64 /
       complex128 JAX arrays.
    2. **Run validation checks** --
       Verify 3-D modes, finiteness, matching weights shape,
       non-negative weights, positive weight sum, and
       positive calibration.
    3. **Branch on validity** --
       If all checks pass, normalise weights and enforce
       positive calibration; otherwise fill scalar fields
       with NaN.

    Parameters
    ----------
    modes : Complex[Array, "H W M"]
        Complex probe modes where M is the number of modes.
    weights : Float[Array, " M"]
        Mode occupation numbers.
    calib : ScalarFloat
        Pixel calibration in Angstroms per pixel.

    Returns
    -------
    probe_modes : ProbeModes
        Validated probe modes instance with normalised
        weights.

    Raises
    ------
    ValueError
        If shapes or values are invalid (caught by
        jaxtyping/beartype).

    See Also
    --------
    :class:`~ptyrodactyl.tools.electron_types.ProbeModes`
        Target PyTree class.
    """
    modes_arr: Complex[Array, " H W M"] = jnp.asarray(
        modes, dtype=jnp.complex128
    )
    weights_arr: Float[Array, " M"] = jnp.asarray(weights, dtype=jnp.float64)
    calib_arr: Float[Array, " "] = jnp.asarray(calib, dtype=jnp.float64)

    expected_dims: int = 3
    modes_shape: Tuple[int, ...] = modes_arr.shape
    num_modes: int = modes_shape[2] if len(modes_shape) == expected_dims else 0

    def _check_3d_modes() -> Bool[Array, " "]:
        """Check that modes array has exactly 3 dimensions.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if modes are 3-D.
        """
        is_3d: bool = len(modes_arr.shape) == expected_dims
        result: Bool[Array, " "] = jnp.array(is_3d)
        return result

    def _check_modes_finite() -> Bool[Array, " "]:
        """Check that all mode values are finite.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all values are finite.
        """
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(modes_arr))
        return result

    def _check_weights_shape() -> Bool[Array, " "]:
        """Check that weights length matches number of modes.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if shapes match.
        """
        result: Bool[Array, " "] = jnp.array(weights_arr.shape == (num_modes,))
        return result

    def _check_weights_nonnegative() -> Bool[Array, " "]:
        """Check that all weights are non-negative.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all weights >= 0.
        """
        result: Bool[Array, " "] = jnp.all(weights_arr >= 0)
        return result

    def _check_weights_sum() -> Bool[Array, " "]:
        """Check that weights sum to a positive value.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if the sum is positive.
        """
        weight_sum: Float[Array, " "] = jnp.sum(weights_arr)
        result: Bool[Array, " "] = weight_sum > jnp.finfo(jnp.float64).eps
        return result

    def _check_calib_positive() -> Bool[Array, " "]:
        """Check that calibration is positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if calibration > 0.
        """
        result: Bool[Array, " "] = calib_arr > 0
        return result

    def _valid_processing() -> ProbeModes:
        """Create ProbeModes with normalised weights.

        Returns
        -------
        probe_modes : ProbeModes
            Validated instance.
        """
        abs_weights: Float[Array, " M"] = jnp.abs(weights_arr)
        weight_sum: Float[Array, " "] = jnp.sum(abs_weights)
        normalized_weights: Float[Array, " M"] = jax.lax.cond(
            weight_sum > jnp.finfo(jnp.float64).eps,
            lambda w: w / weight_sum,
            lambda w: jnp.ones_like(w) / w.shape[0],
            abs_weights,
        )
        positive_calib: Float[Array, " "] = (
            jnp.abs(calib_arr) + jnp.finfo(jnp.float64).eps
        )

        return ProbeModes(
            modes=modes_arr,
            weights=normalized_weights,
            calib=positive_calib,
        )

    def _invalid_processing() -> ProbeModes:
        """Create ProbeModes with NaN values for invalid input.

        Returns
        -------
        probe_modes : ProbeModes
            Instance with NaN weights and calibration.
        """
        nan_weights: Float[Array, " M"] = jnp.full_like(weights_arr, jnp.nan)
        nan_calib: Float[Array, " "] = jnp.array(jnp.nan, dtype=jnp.float64)
        return ProbeModes(
            modes=modes_arr,
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
                    jnp.logical_and(
                        _check_weights_sum(), _check_calib_positive()
                    ),
                ),
            ),
        ),
    )

    result: ProbeModes = jax.lax.cond(
        all_valid,
        lambda _: _valid_processing(),
        lambda _: _invalid_processing(),
        None,
    )
    return result


@jaxtyped(typechecker=beartype)
@jax.jit
def make_potential_slices(
    slices: Float[Array, "H W S"],
    slice_thickness: ScalarNumeric,
    calib: ScalarFloat,
) -> PotentialSlices:
    """Create a validated :class:`PotentialSlices`.

    Extended Summary
    ----------------
    JIT-compiled factory that validates slice dimensionality,
    finiteness, and enforces positive thickness and calibration.

    Implementation Logic
    --------------------
    1. **Cast inputs** --
       Convert slices, thickness, and calibration to float64.
    2. **Run validation checks** --
       Verify 3-D slices, finiteness, positive thickness,
       and positive calibration.
    3. **Branch on validity** --
       If valid, enforce positivity via ``abs() + eps``;
       otherwise fill scalars with NaN.

    Parameters
    ----------
    slices : Float[Array, "H W S"]
        Potential slices where S is the number of slices.
    slice_thickness : ScalarNumeric
        Thickness of each slice in Angstroms.
    calib : ScalarFloat
        Pixel calibration in Angstroms per pixel.

    Returns
    -------
    potential_slices : PotentialSlices
        Validated potential slices instance.

    Raises
    ------
    ValueError
        If shapes or values are invalid (caught by
        jaxtyping/beartype).

    See Also
    --------
    :class:`~ptyrodactyl.tools.electron_types.PotentialSlices`
        Target PyTree class.
    """
    slices_arr: Float[Array, " H W S"] = jnp.asarray(slices, dtype=jnp.float64)
    thickness_arr: Float[Array, " "] = jnp.asarray(
        slice_thickness, dtype=jnp.float64
    )
    calib_arr: Float[Array, " "] = jnp.asarray(calib, dtype=jnp.float64)

    expected_dims: int = 3

    def _check_3d_slices() -> Bool[Array, " "]:
        """Check that slices array has exactly 3 dimensions.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if slices are 3-D.
        """
        is_3d: bool = len(slices_arr.shape) == expected_dims
        result: Bool[Array, " "] = jnp.array(is_3d)
        return result

    def _check_slices_finite() -> Bool[Array, " "]:
        """Check that all slice values are finite.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all values are finite.
        """
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(slices_arr))
        return result

    def _check_slice_thickness_positive() -> Bool[Array, " "]:
        """Check that slice thickness is positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if thickness > 0.
        """
        result: Bool[Array, " "] = thickness_arr > 0
        return result

    def _check_calib_positive() -> Bool[Array, " "]:
        """Check that calibration is positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if calibration > 0.
        """
        result: Bool[Array, " "] = calib_arr > 0
        return result

    def _valid_processing() -> PotentialSlices:
        """Create PotentialSlices with positive values.

        Returns
        -------
        potential_slices : PotentialSlices
            Validated instance.
        """
        positive_thickness: Float[Array, " "] = (
            jnp.abs(thickness_arr) + jnp.finfo(jnp.float64).eps
        )
        positive_calib: Float[Array, " "] = (
            jnp.abs(calib_arr) + jnp.finfo(jnp.float64).eps
        )

        return PotentialSlices(
            slices=slices_arr,
            slice_thickness=positive_thickness,
            calib=positive_calib,
        )

    def _invalid_processing() -> PotentialSlices:
        """Create PotentialSlices with NaN for invalid input.

        Returns
        -------
        potential_slices : PotentialSlices
            Instance with NaN thickness and calibration.
        """
        nan_val: float = jnp.nan
        dtype: type = jnp.float64
        nan_thickness: Float[Array, " "] = jnp.array(nan_val, dtype=dtype)
        nan_calib: Float[Array, " "] = jnp.array(nan_val, dtype=dtype)
        return PotentialSlices(
            slices=slices_arr,
            slice_thickness=nan_thickness,
            calib=nan_calib,
        )

    all_valid: Bool[Array, " "] = jnp.logical_and(
        _check_3d_slices(),
        jnp.logical_and(
            _check_slices_finite(),
            jnp.logical_and(
                _check_slice_thickness_positive(), _check_calib_positive()
            ),
        ),
    )

    result: PotentialSlices = jax.lax.cond(
        all_valid,
        lambda _: _valid_processing(),
        lambda _: _invalid_processing(),
        None,
    )
    return result


@jaxtyped(typechecker=beartype)
@jax.jit
def make_crystal_structure(
    frac_positions: Float[Array, "* 4"],
    cart_positions: Num[Array, "* 4"],
    cell_lengths: Num[Array, " 3"],
    cell_angles: Num[Array, " 3"],
) -> CrystalStructure:
    """Create a validated :class:`CrystalStructure`.

    Extended Summary
    ----------------
    JIT-compiled factory that validates shape consistency,
    atom-count matching, atomic-number agreement, cell-length
    positivity, and angle validity (0--180 degrees).

    Implementation Logic
    --------------------
    1. **Cast inputs** --
       Convert arrays to JAX arrays (float64 for fractional
       positions).
    2. **Run validation checks** --
       Verify column counts, matching atom counts, matching
       atomic numbers, positive cell lengths, and valid
       angles.
    3. **Branch on validity** --
       If valid, clamp angles and enforce positive lengths;
       otherwise fill cell parameters with NaN.

    Parameters
    ----------
    frac_positions : Float[Array, "* 4"]
        Fractional coordinates ``[x, y, z, Z]`` per atom.
    cart_positions : Num[Array, "* 4"]
        Cartesian coordinates ``[x, y, z, Z]`` per atom, in
        Angstroms.
    cell_lengths : Num[Array, " 3"]
        Unit cell lengths ``[a, b, c]`` in Angstroms.
    cell_angles : Num[Array, " 3"]
        Unit cell angles ``[alpha, beta, gamma]`` in degrees.

    Returns
    -------
    crystal_structure : CrystalStructure
        Validated crystal structure instance.

    Raises
    ------
    ValueError
        If shapes or values are invalid (caught by
        jaxtyping/beartype).

    See Also
    --------
    :class:`~ptyrodactyl.tools.electron_types.CrystalStructure`
        Target PyTree class.
    """
    frac_arr: Float[Array, " * 4"] = jnp.asarray(
        frac_positions, dtype=jnp.float64
    )
    cart_arr: Num[Array, " * 4"] = jnp.asarray(cart_positions)
    lengths_arr: Num[Array, " 3"] = jnp.asarray(cell_lengths)
    angles_arr: Num[Array, " 3"] = jnp.asarray(cell_angles)

    num_cols: int = 4
    num_cell_params: int = 3
    min_angle: float = 0.1
    max_angle: float = 179.9
    max_angle_check: float = 180.0

    def _check_frac_shape() -> Bool[Array, " "]:
        """Check fractional positions have 4 columns.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if shape is correct.
        """
        result: Bool[Array, " "] = jnp.array(frac_arr.shape[1] == num_cols)
        return result

    def _check_cart_shape() -> Bool[Array, " "]:
        """Check Cartesian positions have 4 columns.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if shape is correct.
        """
        result: Bool[Array, " "] = jnp.array(cart_arr.shape[1] == num_cols)
        return result

    def _check_cell_lengths_shape() -> Bool[Array, " "]:
        """Check cell lengths array has 3 elements.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if shape is correct.
        """
        valid_shape: bool = lengths_arr.shape[0] == num_cell_params
        result: Bool[Array, " "] = jnp.array(valid_shape)
        return result

    def _check_cell_angles_shape() -> Bool[Array, " "]:
        """Check cell angles array has 3 elements.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if shape is correct.
        """
        valid_shape: bool = angles_arr.shape[0] == num_cell_params
        result: Bool[Array, " "] = jnp.array(valid_shape)
        return result

    def _check_atom_count() -> Bool[Array, " "]:
        """Check atom counts match between arrays.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if counts match.
        """
        result: Bool[Array, " "] = jnp.array(
            frac_arr.shape[0] == cart_arr.shape[0]
        )
        return result

    def _check_atomic_numbers() -> Bool[Array, " "]:
        """Check atomic numbers match between arrays.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all atomic numbers match.
        """
        frac_atomic_nums: Num[Array, " *"] = frac_arr[:, 3]
        cart_atomic_nums: Num[Array, " *"] = cart_arr[:, 3]
        nums_match: Bool[Array, " *"] = frac_atomic_nums == cart_atomic_nums
        result: Bool[Array, " "] = jnp.all(nums_match)
        return result

    def _check_cell_lengths_positive() -> Bool[Array, " "]:
        """Check all cell lengths are positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all lengths > 0.
        """
        result: Bool[Array, " "] = jnp.all(lengths_arr > 0)
        return result

    def _check_cell_angles_valid() -> Bool[Array, " "]:
        """Check cell angles are in (0, 180) degrees.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all angles are valid.
        """
        result: Bool[Array, " "] = jnp.logical_and(
            jnp.all(angles_arr > 0), jnp.all(angles_arr < max_angle_check)
        )
        return result

    def _valid_processing() -> CrystalStructure:
        """Create CrystalStructure with clamped values.

        Returns
        -------
        crystal_structure : CrystalStructure
            Validated instance.
        """
        positive_lengths: Num[Array, " 3"] = (
            jnp.abs(lengths_arr) + jnp.finfo(jnp.float64).eps
        )
        valid_angles: Num[Array, " 3"] = jnp.clip(
            angles_arr, min_angle, max_angle
        )

        return CrystalStructure(
            frac_positions=frac_arr,
            cart_positions=cart_arr,
            cell_lengths=positive_lengths,
            cell_angles=valid_angles,
        )

    def _invalid_processing() -> CrystalStructure:
        """Create CrystalStructure with NaN cell params.

        Returns
        -------
        crystal_structure : CrystalStructure
            Instance with NaN lengths and angles.
        """
        nan_lengths: Num[Array, " 3"] = jnp.full((num_cell_params,), jnp.nan)
        nan_angles: Num[Array, " 3"] = jnp.full((num_cell_params,), jnp.nan)

        return CrystalStructure(
            frac_positions=frac_arr,
            cart_positions=cart_arr,
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

    result: CrystalStructure = jax.lax.cond(
        all_valid,
        lambda _: _valid_processing(),
        lambda _: _invalid_processing(),
        None,
    )
    return result


@jaxtyped(typechecker=beartype)
def make_crystal_data(
    positions: Float[Array, " N 3"],
    atomic_numbers: Int[Array, " N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[ScalarFloat] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> CrystalData:
    """Create a validated :class:`CrystalData`.

    Extended Summary
    ----------------
    Eagerly validates shapes and finiteness of the provided
    arrays. Cannot be JIT-compiled because of ``Optional``
    parameters and Python control flow with exceptions.

    Implementation Logic
    --------------------
    1. **Cast inputs** --
       Convert positions and atomic numbers to JAX arrays.
       Default lattice to identity if ``None``.
    2. **Validate shapes** --
       Check positions have 3 columns and atomic numbers
       match the atom count.
    3. **Validate finiteness** --
       Ensure positions are finite and atomic numbers are
       non-negative.
    4. **Validate optional matrices** --
       Check lattice and stress shapes and finiteness when
       provided.
    5. **Construct PyTree** --
       Return a :class:`CrystalData` instance.

    Parameters
    ----------
    positions : Float[Array, " N 3"]
        Cartesian positions in Angstroms.
    atomic_numbers : Int[Array, " N"]
        Atomic numbers (Z) for each atom.
    lattice : Optional[Float[Array, "3 3"]], optional
        Lattice vectors in Angstroms. Defaults to the 3x3
        identity matrix.
    stress : Optional[Float[Array, "3 3"]], optional
        Stress tensor, or ``None``.
    energy : Optional[ScalarFloat], optional
        Total energy in eV, or ``None``.
    properties : Optional[List[Dict[str, Union[str, int]]]], \
optional
        Per-atom metadata dictionaries.
    comment : Optional[str], optional
        Original comment line from the file.

    Returns
    -------
    crystal_data : CrystalData
        Validated PyTree structure for crystal data.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or contain
        non-finite values.

    See Also
    --------
    :class:`~ptyrodactyl.tools.electron_types.CrystalData`
        Target PyTree class.

    Notes
    -----
    Cannot be JIT compiled due to ``Optional`` parameters and
    Python control flow with exceptions.
    """
    positions_arr: Float[Array, " N 3"] = jnp.asarray(
        positions, dtype=jnp.float64
    )
    atomic_numbers_arr: Int[Array, " N"] = jnp.asarray(
        atomic_numbers, dtype=jnp.int32
    )

    lattice_arr: Optional[Float[Array, "3 3"]]
    if lattice is not None:
        lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)
    else:
        lattice_arr = jnp.eye(3, dtype=jnp.float64)

    stress_arr: Optional[Float[Array, "3 3"]] = None
    if stress is not None:
        stress_arr = jnp.asarray(stress, dtype=jnp.float64)

    energy_arr: Optional[Float[Array, " "]] = None
    if energy is not None:
        energy_arr = jnp.asarray(energy, dtype=jnp.float64)

    def validate_and_create() -> CrystalData:
        """Validate inputs and construct CrystalData.

        Returns
        -------
        crystal_data : CrystalData
            Validated instance.

        Raises
        ------
        ValueError
            On shape mismatch or non-finite values.
        """
        num_atoms: int = positions_arr.shape[0]
        expected_pos_dims: int = 3

        def check_shape() -> None:
            """Validate positions and atomic_numbers shapes.

            Raises
            ------
            ValueError
                If shapes are incorrect.
            """
            if positions_arr.shape[1] != expected_pos_dims:
                raise ValueError("positions must have shape (N, 3)")
            if atomic_numbers_arr.shape[0] != num_atoms:
                raise ValueError("atomic_numbers must have shape (N,)")

        def check_finiteness() -> None:
            """Validate arrays contain finite non-negative values.

            Raises
            ------
            ValueError
                If positions are non-finite or atomic numbers
                are negative.
            """
            if not jnp.all(jnp.isfinite(positions_arr)):
                raise ValueError("positions contain non-finite values")
            if not jnp.all(atomic_numbers_arr >= 0):
                raise ValueError("atomic_numbers must be non-negative")

        def check_optional_matrices() -> None:
            """Validate optional lattice and stress matrices.

            Raises
            ------
            ValueError
                If matrices have wrong shape or non-finite
                values.
            """
            lattice_shape: Tuple[int, int] = (3, 3)
            if lattice_arr is not None:
                if lattice_arr.shape != lattice_shape:
                    raise ValueError("lattice must have shape (3, 3)")
                if not jnp.all(jnp.isfinite(lattice_arr)):
                    raise ValueError("lattice contains non-finite values")

            if stress_arr is not None:
                if stress_arr.shape != lattice_shape:
                    raise ValueError("stress must have shape (3, 3)")
                if not jnp.all(jnp.isfinite(stress_arr)):
                    raise ValueError("stress contains non-finite values")

        check_shape()
        check_finiteness()
        check_optional_matrices()

        return CrystalData(
            positions=positions_arr,
            atomic_numbers=atomic_numbers_arr,
            lattice=lattice_arr,
            stress=stress_arr,
            energy=energy_arr,
            properties=properties,
            comment=comment,
        )

    crystal_data: CrystalData = validate_and_create()
    return crystal_data


@jaxtyped(typechecker=beartype)
@jax.jit
def make_stem4d(
    data: Float[Array, "P H W"],
    real_space_calib: ScalarFloat,
    fourier_space_calib: ScalarFloat,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
) -> STEM4D:
    """Create a validated :class:`STEM4D`.

    Extended Summary
    ----------------
    JIT-compiled factory that validates data dimensionality,
    finiteness, scan-position consistency, and enforces
    positive calibrations and voltage.

    Implementation Logic
    --------------------
    1. **Cast inputs** --
       Convert all inputs to float64 JAX arrays.
    2. **Run validation checks** --
       Verify 3-D data, finiteness, scan-position shape and
       finiteness, and positivity of calibrations and
       voltage.
    3. **Branch on validity** --
       If valid, enforce positivity via ``abs() + eps``;
       otherwise fill calibrations and voltage with NaN.

    Parameters
    ----------
    data : Float[Array, "P H W"]
        4D-STEM data with P scan positions and H x W
        diffraction patterns.
    real_space_calib : ScalarFloat
        Real space calibration in Angstroms per pixel.
    fourier_space_calib : ScalarFloat
        Fourier space calibration in inverse Angstroms per
        pixel.
    scan_positions : Float[Array, "P 2"]
        Scan positions in Angstroms as (y, x) coordinates.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.

    Returns
    -------
    stem4d : STEM4D
        Validated 4D-STEM data structure.

    See Also
    --------
    :class:`~ptyrodactyl.tools.electron_types.STEM4D`
        Target PyTree class.
    """
    data_arr: Float[Array, " P H W"] = jnp.asarray(data, dtype=jnp.float64)
    real_calib_arr: Float[Array, " "] = jnp.asarray(
        real_space_calib,
        dtype=jnp.float64,
    )
    fourier_calib_arr: Float[Array, " "] = jnp.asarray(
        fourier_space_calib, dtype=jnp.float64
    )
    scan_pos_arr: Float[Array, " P 2"] = jnp.asarray(
        scan_positions, dtype=jnp.float64
    )
    voltage_arr: Float[Array, " "] = jnp.asarray(voltage_kv, dtype=jnp.float64)

    has_shape: bool = len(data_arr.shape) >= 1
    num_scan_positions: int = data_arr.shape[0] if has_shape else 0
    num_scan_coords: int = 2
    expected_dims: int = 3

    def _check_data_3d() -> Bool[Array, " "]:
        """Check that data array has exactly 3 dimensions.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if data is 3-D.
        """
        is_3d: bool = len(data_arr.shape) == expected_dims
        result: Bool[Array, " "] = jnp.array(is_3d)
        return result

    def _check_data_finite() -> Bool[Array, " "]:
        """Check that all data values are finite.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all values are finite.
        """
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(data_arr))
        return result

    def _check_scan_positions_shape() -> Bool[Array, " "]:
        """Check scan positions match data and have 2 coords.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if shape is ``(P, 2)``.
        """
        result: Bool[Array, " "] = jnp.logical_and(
            jnp.array(scan_pos_arr.shape[0] == num_scan_positions),
            jnp.array(scan_pos_arr.shape[1] == num_scan_coords),
        )
        return result

    def _check_scan_positions_finite() -> Bool[Array, " "]:
        """Check that all scan position values are finite.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if all positions are finite.
        """
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(scan_pos_arr))
        return result

    def _check_real_space_calib_positive() -> Bool[Array, " "]:
        """Check that real space calibration is positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if calibration > 0.
        """
        result: Bool[Array, " "] = real_calib_arr > 0
        return result

    def _check_fourier_space_calib_positive() -> Bool[Array, " "]:
        """Check that Fourier space calibration is positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if calibration > 0.
        """
        result: Bool[Array, " "] = fourier_calib_arr > 0
        return result

    def _check_voltage_positive() -> Bool[Array, " "]:
        """Check that accelerating voltage is positive.

        Returns
        -------
        is_valid : Bool[Array, " "]
            ``True`` if voltage > 0.
        """
        result: Bool[Array, " "] = voltage_arr > 0
        return result

    def _valid_processing() -> STEM4D:
        """Create STEM4D with positive calibration values.

        Returns
        -------
        stem4d : STEM4D
            Validated instance.
        """
        positive_real_calib: Float[Array, " "] = (
            jnp.abs(real_calib_arr) + jnp.finfo(jnp.float64).eps
        )
        positive_fourier_calib: Float[Array, " "] = (
            jnp.abs(fourier_calib_arr) + jnp.finfo(jnp.float64).eps
        )
        positive_voltage: Float[Array, " "] = (
            jnp.abs(voltage_arr) + jnp.finfo(jnp.float64).eps
        )

        return STEM4D(
            data=data_arr,
            real_space_calib=positive_real_calib,
            fourier_space_calib=positive_fourier_calib,
            scan_positions=scan_pos_arr,
            voltage_kv=positive_voltage,
        )

    def _invalid_processing() -> STEM4D:
        """Create STEM4D with NaN values for invalid input.

        Returns
        -------
        stem4d : STEM4D
            Instance with NaN calibrations and voltage.
        """
        nan_calib: Float[Array, " "] = jnp.array(jnp.nan, dtype=jnp.float64)

        return STEM4D(
            data=data_arr,
            real_space_calib=nan_calib,
            fourier_space_calib=nan_calib,
            scan_positions=scan_pos_arr,
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

    result: STEM4D = jax.lax.cond(
        all_valid,
        lambda _: _valid_processing(),
        lambda _: _invalid_processing(),
        None,
    )
    return result


__all__: list[str] = [
    "make_calibrated_array",
    "make_crystal_data",
    "make_crystal_structure",
    "make_potential_slices",
    "make_probe_modes",
    "make_stem4d",
]
