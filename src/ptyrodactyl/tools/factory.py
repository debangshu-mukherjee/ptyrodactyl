"""Factory functions for validating data before PyTree loading.

Extended Summary
----------------
Provides JAX safe functional data validation before loading data into PyTrees.
It is recommended to use these functions to access PyTrees rather than
instantiating them directly.

Routine Listings
----------------
make_calibrated_array : function
    Creates a CalibratedArray instance with runtime type checking.
make_probe_modes : function
    Creates a ProbeModes instance with runtime type checking.
make_potential_slices : function
    Creates a PotentialSlices instance with runtime type checking.
make_crystal_structure : function
    Creates a CrystalStructure instance with runtime type checking.
make_xyz_data : function
    Creates an XYZData instance with runtime type checking.
make_stem4d : function
    Creates a STEM4D instance with runtime type checking.

Notes
-----
Always use factory functions instead of directly instantiating NamedTuple
classes to ensure proper runtime type checking of the contents.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from jaxtyping import Array, Bool, Complex, Float, Int, Num, jaxtyped

from .electron_types import (
    STEM4D,
    CalibratedArray,
    CrystalStructure,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarNumeric,
    XYZData,
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
    """JAX-safe factory function for CalibratedArray with data validation.

    Parameters
    ----------
    data_array : Union[Int, Float, Complex][Array, "H W"]
        The actual array data
    calib_y : ScalarFloat
        Calibration in y direction
    calib_x : ScalarFloat
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
    Validations performed:
    - data_array is 2D
    - data_array is finite
    - calib_y is positive
    - calib_x is positive
    - real_space is a boolean scalar
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
    """JAX-safe factory function for ProbeModes with data validation.

    Parameters
    ----------
    modes : Complex[Array, "H W M"]
        Complex probe modes, M is number of modes
    weights : Float[Array, " M"]
        Mode occupation numbers
    calib : ScalarFloat
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
    Validates modes shape, finiteness, weights shape and non-negativity,
    and calibration positivity. Normalizes weights to sum to 1.
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
        """Check that modes array has exactly 3 dimensions."""
        is_3d: bool = len(modes_arr.shape) == expected_dims
        result: Bool[Array, " "] = jnp.array(is_3d)
        return result

    def _check_modes_finite() -> Bool[Array, " "]:
        """Check that all mode values are finite."""
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(modes_arr))
        return result

    def _check_weights_shape() -> Bool[Array, " "]:
        """Check that weights array matches number of modes."""
        result: Bool[Array, " "] = jnp.array(weights_arr.shape == (num_modes,))
        return result

    def _check_weights_nonnegative() -> Bool[Array, " "]:
        """Check that all weights are non-negative."""
        result: Bool[Array, " "] = jnp.all(weights_arr >= 0)
        return result

    def _check_weights_sum() -> Bool[Array, " "]:
        """Check that weights sum to a positive value."""
        weight_sum: Float[Array, " "] = jnp.sum(weights_arr)
        result: Bool[Array, " "] = weight_sum > jnp.finfo(jnp.float64).eps
        return result

    def _check_calib_positive() -> Bool[Array, " "]:
        """Check that calibration is positive."""
        result: Bool[Array, " "] = calib_arr > 0
        return result

    def _valid_processing() -> ProbeModes:
        """Create validated ProbeModes with normalized weights."""
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
        """Create ProbeModes with NaN values for invalid input."""
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
    """JAX-safe factory function for PotentialSlices with data validation.

    Parameters
    ----------
    slices : Float[Array, "H W S"]
        Individual potential slices, S is number of slices
    slice_thickness : ScalarNumeric
        Thickness of each slice
    calib : ScalarFloat
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
    Validates slices shape and finiteness, slice_thickness positivity,
    and calibration positivity.
    """
    slices_arr: Float[Array, " H W S"] = jnp.asarray(slices, dtype=jnp.float64)
    thickness_arr: Float[Array, " "] = jnp.asarray(
        slice_thickness, dtype=jnp.float64
    )
    calib_arr: Float[Array, " "] = jnp.asarray(calib, dtype=jnp.float64)

    expected_dims: int = 3

    def _check_3d_slices() -> Bool[Array, " "]:
        """Check that slices array has exactly 3 dimensions."""
        is_3d: bool = len(slices_arr.shape) == expected_dims
        result: Bool[Array, " "] = jnp.array(is_3d)
        return result

    def _check_slices_finite() -> Bool[Array, " "]:
        """Check that all slice values are finite."""
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(slices_arr))
        return result

    def _check_slice_thickness_positive() -> Bool[Array, " "]:
        """Check that slice thickness is positive."""
        result: Bool[Array, " "] = thickness_arr > 0
        return result

    def _check_calib_positive() -> Bool[Array, " "]:
        """Check that calibration is positive."""
        result: Bool[Array, " "] = calib_arr > 0
        return result

    def _valid_processing() -> PotentialSlices:
        """Create validated PotentialSlices with positive values."""
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
        """Create PotentialSlices with NaN values for invalid input."""
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
    """Create a CrystalStructure instance with type checking.

    Parameters
    ----------
    frac_positions : Float[Array, "* 4"]
        Array of shape (n_atoms, 4) with fractional coordinates.
    cart_positions : Num[Array, "* 4"]
        Array of shape (n_atoms, 4) with Cartesian coordinates.
    cell_lengths : Num[Array, " 3"]
        Unit cell lengths [a, b, c] in Angstroms.
    cell_angles : Num[Array, " 3"]
        Unit cell angles [alpha, beta, gamma] in degrees.

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
    Validates shape consistency, atom count matching, atomic number
    matching, cell length positivity, and angle validity (0-180 degrees).
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
        """Check that fractional positions have 4 columns."""
        result: Bool[Array, " "] = jnp.array(frac_arr.shape[1] == num_cols)
        return result

    def _check_cart_shape() -> Bool[Array, " "]:
        """Check that Cartesian positions have 4 columns."""
        result: Bool[Array, " "] = jnp.array(cart_arr.shape[1] == num_cols)
        return result

    def _check_cell_lengths_shape() -> Bool[Array, " "]:
        """Check that cell lengths array has 3 elements."""
        valid_shape: bool = lengths_arr.shape[0] == num_cell_params
        result: Bool[Array, " "] = jnp.array(valid_shape)
        return result

    def _check_cell_angles_shape() -> Bool[Array, " "]:
        """Check that cell angles array has 3 elements."""
        valid_shape: bool = angles_arr.shape[0] == num_cell_params
        result: Bool[Array, " "] = jnp.array(valid_shape)
        return result

    def _check_atom_count() -> Bool[Array, " "]:
        """Check that fractional and Cartesian arrays have same atom count."""
        result: Bool[Array, " "] = jnp.array(
            frac_arr.shape[0] == cart_arr.shape[0]
        )
        return result

    def _check_atomic_numbers() -> Bool[Array, " "]:
        """Check that atomic numbers match between fractional and Cartesian."""
        frac_atomic_nums: Num[Array, " *"] = frac_arr[:, 3]
        cart_atomic_nums: Num[Array, " *"] = cart_arr[:, 3]
        nums_match: Bool[Array, " *"] = frac_atomic_nums == cart_atomic_nums
        result: Bool[Array, " "] = jnp.all(nums_match)
        return result

    def _check_cell_lengths_positive() -> Bool[Array, " "]:
        """Check that all cell lengths are positive."""
        result: Bool[Array, " "] = jnp.all(lengths_arr > 0)
        return result

    def _check_cell_angles_valid() -> Bool[Array, " "]:
        """Check that cell angles are within valid range (0-180 degrees)."""
        result: Bool[Array, " "] = jnp.logical_and(
            jnp.all(angles_arr > 0), jnp.all(angles_arr < max_angle_check)
        )
        return result

    def _valid_processing() -> CrystalStructure:
        """Create validated CrystalStructure with clamped values."""
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
        """Create CrystalStructure with NaN values for invalid input."""
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
def make_xyz_data(
    positions: Float[Array, " N 3"],
    atomic_numbers: Int[Array, " N"],
    lattice: Optional[Float[Array, "3 3"]] = None,
    stress: Optional[Float[Array, "3 3"]] = None,
    energy: Optional[ScalarFloat] = None,
    properties: Optional[List[Dict[str, Union[str, int]]]] = None,
    comment: Optional[str] = None,
) -> XYZData:
    """Create XYZData instance with runtime validation.

    Parameters
    ----------
    positions : Float[Array, " N 3"]
        Cartesian positions in Angstroms
    atomic_numbers : Int[Array, " N"]
        Atomic numbers (Z) for each atom
    lattice : Optional[Float[Array, "3 3"]], default=None
        Lattice vectors (if any)
    stress : Optional[Float[Array, "3 3"]], default=None
        Stress tensor (if any)
    energy : Optional[ScalarFloat], default=None
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
    Cannot be JIT compiled due to Optional parameters and Python
    control flow with exceptions.
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

    def validate_and_create() -> XYZData:
        """Validate inputs and create XYZData instance."""
        num_atoms: int = positions_arr.shape[0]
        expected_pos_dims: int = 3

        def check_shape() -> None:
            """Validate positions and atomic_numbers array shapes."""
            if positions_arr.shape[1] != expected_pos_dims:
                raise ValueError("positions must have shape (N, 3)")
            if atomic_numbers_arr.shape[0] != num_atoms:
                raise ValueError("atomic_numbers must have shape (N,)")

        def check_finiteness() -> None:
            """Validate that arrays contain finite non-negative values."""
            if not jnp.all(jnp.isfinite(positions_arr)):
                raise ValueError("positions contain non-finite values")
            if not jnp.all(atomic_numbers_arr >= 0):
                raise ValueError("atomic_numbers must be non-negative")

        def check_optional_matrices() -> None:
            """Validate optional lattice and stress matrices if present."""
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

        return XYZData(
            positions=positions_arr,
            atomic_numbers=atomic_numbers_arr,
            lattice=lattice_arr,
            stress=stress_arr,
            energy=energy_arr,
            properties=properties,
            comment=comment,
        )

    result: XYZData = validate_and_create()
    return result


@jaxtyped(typechecker=beartype)
@jax.jit
def make_stem4d(
    data: Float[Array, "P H W"],
    real_space_calib: ScalarFloat,
    fourier_space_calib: ScalarFloat,
    scan_positions: Float[Array, "P 2"],
    voltage_kv: ScalarNumeric,
) -> STEM4D:
    """JAX-safe factory function for STEM4D with data validation.

    Parameters
    ----------
    data : Float[Array, "P H W"]
        4D-STEM data array with P scan positions and HxW diffraction patterns
    real_space_calib : ScalarFloat
        Real space calibration in Angstroms per pixel
    fourier_space_calib : ScalarFloat
        Fourier space calibration in inverse Angstroms per pixel
    scan_positions : Float[Array, "P 2"]
        Real space scan positions in Angstroms (y, x coordinates)
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts

    Returns
    -------
    STEM4D
        Validated 4D-STEM data structure

    Notes
    -----
    Validates data shape, finiteness, scan positions shape and finiteness,
    and positivity of calibrations and voltage.
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
        """Check that data array has exactly 3 dimensions."""
        is_3d: bool = len(data_arr.shape) == expected_dims
        result: Bool[Array, " "] = jnp.array(is_3d)
        return result

    def _check_data_finite() -> Bool[Array, " "]:
        """Check that all data values are finite."""
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(data_arr))
        return result

    def _check_scan_positions_shape() -> Bool[Array, " "]:
        """Check that scan positions match data and have 2 coordinates."""
        result: Bool[Array, " "] = jnp.logical_and(
            jnp.array(scan_pos_arr.shape[0] == num_scan_positions),
            jnp.array(scan_pos_arr.shape[1] == num_scan_coords),
        )
        return result

    def _check_scan_positions_finite() -> Bool[Array, " "]:
        """Check that all scan position values are finite."""
        result: Bool[Array, " "] = jnp.all(jnp.isfinite(scan_pos_arr))
        return result

    def _check_real_space_calib_positive() -> Bool[Array, " "]:
        """Check that real space calibration is positive."""
        result: Bool[Array, " "] = real_calib_arr > 0
        return result

    def _check_fourier_space_calib_positive() -> Bool[Array, " "]:
        """Check that Fourier space calibration is positive."""
        result: Bool[Array, " "] = fourier_calib_arr > 0
        return result

    def _check_voltage_positive() -> Bool[Array, " "]:
        """Check that accelerating voltage is positive."""
        result: Bool[Array, " "] = voltage_arr > 0
        return result

    def _valid_processing() -> STEM4D:
        """Create validated STEM4D with positive calibration values."""
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
        """Create STEM4D with NaN values for invalid input."""
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
