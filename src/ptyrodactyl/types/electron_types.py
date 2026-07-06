"""Define electron microscopy carriers and factories.

Extended Summary
----------------
This module defines the canonical Equinox PyTree carriers for
electron-microscopy arrays, probe modes, multislice potentials, and
4D-STEM datasets. Factory functions coerce inputs to JAX arrays and
validate structural properties separately from traced data-dependent
properties.

Routine Listings
----------------
:class:`CalibratedArray`
    Calibrated array data with spatial calibration.
:class:`ProbeModes`
    Multimodal electron probe state.
:class:`PotentialSlices`
    Potential slices for multi-slice simulations.
:class:`STEM4D`
    4D-STEM data with diffraction patterns, calibrations, and parameters.
:func:`create_calibrated_array`
    Create a CalibratedArray with runtime validation.
:func:`create_probe_modes`
    Create a ProbeModes with runtime validation.
:func:`create_potential_slices`
    Create a PotentialSlices with runtime validation.
:func:`create_stem4d`
    Create a STEM4D with runtime validation.

Notes
-----
All carriers are Equinox modules. Array fields are dynamic JAX leaves, and
the module classes rely on Equinox's automatic PyTree registration.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Union
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from .custom_types import scalar_bool, scalar_float, scalar_num


class CalibratedArray(eqx.Module):
    """Store calibrated array data with spatial calibration.

    :see: :class:`~.test_electron_types.TestCalibratedArray`

    Attributes
    ----------
    data_array : Union[Int[Array, "H W"], Float[Array, "H W"], \
Complex[Array, "H W"]]
        Two-dimensional image, diffraction pattern, or complex field.
    calib_y : scalar_float
        Calibration along the y axis.
    calib_x : scalar_float
        Calibration along the x axis.
    real_space : scalar_bool
        Whether the array is calibrated in real space.
    """

    data_array: Union[
        Int[Array, "H W"], Float[Array, "H W"], Complex[Array, "H W"]
    ]
    calib_y: scalar_float
    calib_x: scalar_float
    real_space: scalar_bool


class ProbeModes(eqx.Module):
    """Store multimodal electron probe data.

    :see: :class:`~.test_electron_types.TestProbeModes`

    Attributes
    ----------
    modes : Complex[Array, "H W M"]
        Complex probe modes, with mode index along the final axis.
    weights : Float[Array, " M"]
        Non-negative mode occupation weights normalized to sum to one.
    calib : scalar_float
        Pixel calibration in Angstroms per pixel.
    """

    modes: Complex[Array, "H W M"]
    weights: Float[Array, " M"]
    calib: scalar_float


class PotentialSlices(eqx.Module):
    """Store potential slices for multislice simulations.

    :see: :class:`~.test_electron_types.TestPotentialSlices`

    Attributes
    ----------
    slices : Float[Array, "H W S"]
        Potential slice stack, with slice index along the final axis.
    slice_thickness : scalar_float
        Thickness of each slice in Angstroms.
    calib : scalar_float
        Pixel calibration in Angstroms per pixel.
    """

    slices: Float[Array, "H W S"]
    slice_thickness: scalar_float
    calib: scalar_float


class STEM4D(eqx.Module):
    """Store 4D-STEM diffraction data.

    :see: :class:`~.test_electron_types.TestSTEM4D`

    Attributes
    ----------
    data : Float[Array, "P H W"]
        Diffraction patterns for P scan positions.
    real_space_calib : scalar_float
        Real-space scan calibration in Angstroms per pixel.
    fourier_space_calib : scalar_float
        Fourier-space detector calibration in inverse Angstroms per pixel.
    scan_positions : Float[Array, "P 2"]
        Real-space scan positions as y/x coordinates in Angstroms.
    voltage_kv : scalar_float
        Accelerating voltage in kilovolts.
    """

    data: Float[Array, "P H W"]
    real_space_calib: scalar_float
    fourier_space_calib: scalar_float
    scan_positions: Float[Array, "P 2"]
    voltage_kv: scalar_float


@jaxtyped(typechecker=beartype)
def create_calibrated_array(
    data_array: Union[
        Int[Array, "..."], Float[Array, "..."], Complex[Array, "..."]
    ],
    calib_y: scalar_float,
    calib_x: scalar_float,
    real_space: scalar_bool,
) -> CalibratedArray:
    """Create a CalibratedArray with runtime validation.

    :see: :class:`~.test_electron_types.TestCalibratedArray`

    Parameters
    ----------
    data_array : Union[Int[Array, "..."], Float[Array, "..."], \
Complex[Array, "..."]]
        Two-dimensional array data.
    calib_y : scalar_float
        Calibration along the y axis.
    calib_x : scalar_float
        Calibration along the x axis.
    real_space : scalar_bool
        Whether the array lives in real space.

    Returns
    -------
    calibrated_array : CalibratedArray
        Validated calibrated array.

    Raises
    ------
    ValueError
        If the array rank or scalar structures are invalid.

    Notes
    -----
    1. Convert inputs to JAX arrays.
    2. Validate array rank and scalar shapes.
    3. Require positive calibrations with traced error checks.
    4. Create and return a CalibratedArray.
    """
    data_array_arr = jnp.asarray(data_array)
    calib_y_arr: Float[Array, ""] = jnp.asarray(
        calib_y, dtype=jnp.float64
    )
    calib_x_arr: Float[Array, ""] = jnp.asarray(
        calib_x, dtype=jnp.float64
    )
    real_space_arr = jnp.asarray(real_space, dtype=jnp.bool_)

    expected_rank: int = 2
    scalar_shape: tuple[()] = ()
    if data_array_arr.ndim != expected_rank:
        raise ValueError("data_array must be 2D")
    if calib_y_arr.shape != scalar_shape:
        raise ValueError("calib_y must be a scalar")
    if calib_x_arr.shape != scalar_shape:
        raise ValueError("calib_x must be a scalar")
    if real_space_arr.shape != scalar_shape:
        raise ValueError("real_space must be a scalar")

    checked_calib_y: Float[Array, ""] = eqx.error_if(
        calib_y_arr,
        calib_y_arr <= 0,
        "calib_y must be positive",
    )
    checked_calib_x: Float[Array, ""] = eqx.error_if(
        calib_x_arr,
        calib_x_arr <= 0,
        "calib_x must be positive",
    )
    calibrated_array: CalibratedArray = CalibratedArray(
        data_array=data_array_arr,
        calib_y=checked_calib_y,
        calib_x=checked_calib_x,
        real_space=real_space_arr,
    )
    return calibrated_array


@jaxtyped(typechecker=beartype)
def create_probe_modes(
    modes: Complex[Array, "..."],
    weights: Float[Array, "..."],
    calib: scalar_float,
) -> ProbeModes:
    """Create a ProbeModes with runtime validation.

    :see: :class:`~.test_electron_types.TestProbeModes`

    Parameters
    ----------
    modes : Complex[Array, "..."]
        Complex probe modes. Must have shape ``(H, W, M)``.
    weights : Float[Array, "..."]
        Non-negative mode occupation weights. Must have shape ``(M,)``.
    calib : scalar_float
        Pixel calibration in Angstroms per pixel.

    Returns
    -------
    probe_modes : ProbeModes
        Validated probe modes with normalized weights.

    Raises
    ------
    ValueError
        If ranks, shapes, or scalar structures are invalid.

    Notes
    -----
    1. Convert inputs to complex128 and float64 JAX arrays.
    2. Validate the mode rank and weight shape.
    3. Require finite modes, non-negative weights, positive weight sum, and
       positive calibration with traced error checks.
    4. Normalize the weights and create a ProbeModes.
    """
    modes_arr: Complex[Array, "H W M"] = jnp.asarray(
        modes, dtype=jnp.complex128
    )
    weights_arr: Float[Array, " M"] = jnp.asarray(
        weights, dtype=jnp.float64
    )
    calib_arr: Float[Array, ""] = jnp.asarray(calib, dtype=jnp.float64)

    expected_rank: int = 3
    weights_rank: int = 1
    scalar_shape: tuple[()] = ()
    if modes_arr.ndim != expected_rank:
        raise ValueError("modes must be 3D")
    if weights_arr.ndim != weights_rank:
        raise ValueError("weights must be 1D")
    num_modes: int = modes_arr.shape[2]
    if weights_arr.shape != (num_modes,):
        raise ValueError("weights must have shape (M,)")
    if calib_arr.shape != scalar_shape:
        raise ValueError("calib must be a scalar")

    checked_modes: Complex[Array, "H W M"] = eqx.error_if(
        modes_arr,
        jnp.any(~jnp.isfinite(modes_arr)),
        "modes contain non-finite values",
    )
    checked_weights: Float[Array, " M"] = eqx.error_if(
        weights_arr,
        jnp.any(~jnp.isfinite(weights_arr)),
        "weights contain non-finite values",
    )
    checked_weights = eqx.error_if(
        checked_weights,
        jnp.any(checked_weights < 0),
        "weights must be non-negative",
    )
    weight_sum: Float[Array, ""] = jnp.sum(checked_weights)
    checked_weights = eqx.error_if(
        checked_weights,
        weight_sum <= jnp.finfo(jnp.float64).eps,
        "weights must sum to a positive value",
    )
    checked_calib: Float[Array, ""] = eqx.error_if(
        calib_arr,
        calib_arr <= 0,
        "calib must be positive",
    )
    normalized_weights: Float[Array, " M"] = checked_weights / weight_sum
    probe_modes: ProbeModes = ProbeModes(
        modes=checked_modes,
        weights=normalized_weights,
        calib=checked_calib,
    )
    return probe_modes


@jaxtyped(typechecker=beartype)
def create_potential_slices(
    slices: Float[Array, "..."],
    slice_thickness: scalar_num,
    calib: scalar_float,
) -> PotentialSlices:
    """Create a PotentialSlices with runtime validation.

    :see: :class:`~.test_electron_types.TestPotentialSlices`

    Parameters
    ----------
    slices : Float[Array, "..."]
        Potential slice stack. Must have shape ``(H, W, S)``.
    slice_thickness : scalar_num
        Thickness of each slice in Angstroms.
    calib : scalar_float
        Pixel calibration in Angstroms per pixel.

    Returns
    -------
    potential_slices : PotentialSlices
        Validated potential slices.

    Raises
    ------
    ValueError
        If ranks, shapes, or scalar structures are invalid.

    Notes
    -----
    1. Convert inputs to float64 JAX arrays.
    2. Validate the slice rank and scalar shapes.
    3. Require finite slices, positive thickness, and positive calibration
       with traced error checks.
    4. Create and return a PotentialSlices.
    """
    slices_arr: Float[Array, "H W S"] = jnp.asarray(
        slices, dtype=jnp.float64
    )
    thickness_arr: Float[Array, ""] = jnp.asarray(
        slice_thickness, dtype=jnp.float64
    )
    calib_arr: Float[Array, ""] = jnp.asarray(calib, dtype=jnp.float64)

    expected_rank: int = 3
    scalar_shape: tuple[()] = ()
    if slices_arr.ndim != expected_rank:
        raise ValueError("slices must be 3D")
    if thickness_arr.shape != scalar_shape:
        raise ValueError("slice_thickness must be a scalar")
    if calib_arr.shape != scalar_shape:
        raise ValueError("calib must be a scalar")

    checked_slices: Float[Array, "H W S"] = eqx.error_if(
        slices_arr,
        jnp.any(~jnp.isfinite(slices_arr)),
        "slices contain non-finite values",
    )
    checked_thickness: Float[Array, ""] = eqx.error_if(
        thickness_arr,
        thickness_arr <= 0,
        "slice_thickness must be positive",
    )
    checked_calib: Float[Array, ""] = eqx.error_if(
        calib_arr,
        calib_arr <= 0,
        "calib must be positive",
    )
    potential_slices: PotentialSlices = PotentialSlices(
        slices=checked_slices,
        slice_thickness=checked_thickness,
        calib=checked_calib,
    )
    return potential_slices


@jaxtyped(typechecker=beartype)
def create_stem4d(
    data: Float[Array, "..."],
    real_space_calib: scalar_float,
    fourier_space_calib: scalar_float,
    scan_positions: Float[Array, "..."],
    voltage_kv: scalar_num,
) -> STEM4D:
    """Create a STEM4D with runtime validation.

    :see: :class:`~.test_electron_types.TestSTEM4D`

    Parameters
    ----------
    data : Float[Array, "..."]
        Diffraction patterns. Must have shape ``(P, H, W)``.
    real_space_calib : scalar_float
        Real-space scan calibration in Angstroms per pixel.
    fourier_space_calib : scalar_float
        Fourier-space detector calibration in inverse Angstroms per pixel.
    scan_positions : Float[Array, "..."]
        Scan positions in Angstroms. Must have shape ``(P, 2)``.
    voltage_kv : scalar_num
        Accelerating voltage in kilovolts.

    Returns
    -------
    stem4d : STEM4D
        Validated 4D-STEM dataset.

    Raises
    ------
    ValueError
        If ranks, shapes, or scalar structures are invalid.

    Notes
    -----
    1. Convert inputs to float64 JAX arrays.
    2. Validate data rank, scan-position shape, and scalar shapes.
    3. Require finite data and scan positions, positive calibrations, and
       positive voltage with traced error checks.
    4. Create and return a STEM4D.
    """
    data_arr: Float[Array, "P H W"] = jnp.asarray(data, dtype=jnp.float64)
    real_calib_arr: Float[Array, ""] = jnp.asarray(
        real_space_calib,
        dtype=jnp.float64,
    )
    fourier_calib_arr: Float[Array, ""] = jnp.asarray(
        fourier_space_calib,
        dtype=jnp.float64,
    )
    scan_positions_arr: Float[Array, "P 2"] = jnp.asarray(
        scan_positions,
        dtype=jnp.float64,
    )
    voltage_arr: Float[Array, ""] = jnp.asarray(
        voltage_kv, dtype=jnp.float64
    )

    expected_data_rank: int = 3
    expected_scan_rank: int = 2
    num_scan_coords: int = 2
    scalar_shape: tuple[()] = ()
    if data_arr.ndim != expected_data_rank:
        raise ValueError("data must be 3D")
    if scan_positions_arr.ndim != expected_scan_rank:
        raise ValueError("scan_positions must be 2D")
    if scan_positions_arr.shape != (data_arr.shape[0], num_scan_coords):
        raise ValueError("scan_positions must have shape (P, 2)")
    if real_calib_arr.shape != scalar_shape:
        raise ValueError("real_space_calib must be a scalar")
    if fourier_calib_arr.shape != scalar_shape:
        raise ValueError("fourier_space_calib must be a scalar")
    if voltage_arr.shape != scalar_shape:
        raise ValueError("voltage_kv must be a scalar")

    checked_data: Float[Array, "P H W"] = eqx.error_if(
        data_arr,
        jnp.any(~jnp.isfinite(data_arr)),
        "data contain non-finite values",
    )
    checked_scan_positions: Float[Array, "P 2"] = eqx.error_if(
        scan_positions_arr,
        jnp.any(~jnp.isfinite(scan_positions_arr)),
        "scan_positions contain non-finite values",
    )
    checked_real_calib: Float[Array, ""] = eqx.error_if(
        real_calib_arr,
        real_calib_arr <= 0,
        "real_space_calib must be positive",
    )
    checked_fourier_calib: Float[Array, ""] = eqx.error_if(
        fourier_calib_arr,
        fourier_calib_arr <= 0,
        "fourier_space_calib must be positive",
    )
    checked_voltage: Float[Array, ""] = eqx.error_if(
        voltage_arr,
        voltage_arr <= 0,
        "voltage_kv must be positive",
    )
    stem4d: STEM4D = STEM4D(
        data=checked_data,
        real_space_calib=checked_real_calib,
        fourier_space_calib=checked_fourier_calib,
        scan_positions=checked_scan_positions,
        voltage_kv=checked_voltage,
    )
    return stem4d


__all__: list[str] = [
    "CalibratedArray",
    "PotentialSlices",
    "ProbeModes",
    "STEM4D",
    "create_calibrated_array",
    "create_potential_slices",
    "create_probe_modes",
    "create_stem4d",
]
