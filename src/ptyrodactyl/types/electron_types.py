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
:class:`AxisUpdate`
    Additive distribution-axis deltas for one kernel evaluation.
:class:`AtomicSliceData`
    On-the-fly atomic slice inputs for sharded multislice.
:class:`CalibratedArray`
    Calibrated array data with spatial calibration.
:class:`DetectorConfig`
    Detector, scan-position, and calibration configuration.
:class:`EnsembleAxes`
    Optional ensemble distributions carried inside microscope config.
:class:`MicroscopeConfig`
    Microscope voltage and probe-aberration configuration.
:class:`ProbeModes`
    Multimodal electron probe state.
:class:`PotentialSlices`
    Potential slices for multi-slice simulations.
:class:`STEM4D`
    4D-STEM data with diffraction patterns, calibrations, and parameters.
:func:`combine_axis_updates`
    Sum multiple AxisUpdate carriers.
:func:`create_axis_update`
    Create an AxisUpdate with runtime validation.
:func:`create_atomic_slice_data`
    Create AtomicSliceData with runtime validation.
:func:`create_calibrated_array`
    Create a CalibratedArray with runtime validation.
:func:`create_detector_config`
    Create DetectorConfig with runtime validation.
:func:`create_ensemble_axes`
    Create EnsembleAxes with runtime validation.
:func:`create_microscope_config`
    Create MicroscopeConfig with runtime validation.
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
from beartype.typing import Optional, Tuple, Union
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from .custom_types import scalar_bool, scalar_float, scalar_num
from .distributions import Distribution

_CUBE_RANK: int = 3
_MATRIX_RANK: int = 2
_VECTOR_RANK: int = 1
_XY_COORDS: int = 2
_XYZ_COORDS: int = 3


class EnsembleAxes(eqx.Module):
    """Store optional ensemble distributions for forward simulators.

    The optional distribution fields are dynamic PyTree leaves when present.
    Their ``None``/present structure is handled by the PyTree definition and
    is deliberately not marked static here.

    Attributes
    ----------
    probe_modes : Optional[Distribution]
        Optional explicit probe-mode distribution.
    position_jitter : Optional[Distribution]
        Optional incoherent position-jitter distribution.
    coherence : Optional[Distribution]
        Optional chromatic/angular coherence distribution.
    """

    probe_modes: Optional[Distribution] = None
    position_jitter: Optional[Distribution] = None
    coherence: Optional[Distribution] = None


class MicroscopeConfig(eqx.Module):
    """Store microscope voltage and probe-aberration configuration.

    Physical scalar fields are dynamic leaves so gradients can flow through
    voltage, aperture, defocus, and aberration coefficients. Probe array
    shape is static because it controls JAX array construction.

    Attributes
    ----------
    voltage_kv : Float[Array, " "]
        Accelerating voltage in kilovolts.
    aperture_mrad : Float[Array, " "]
        Probe aperture semi-angle in milliradians.
    defocus_ang : Float[Array, " "]
        Defocus in Angstroms.
    c3_ang : Float[Array, " "]
        Third-order spherical aberration in Angstroms.
    c5_ang : Float[Array, " "]
        Fifth-order spherical aberration in Angstroms.
    ensemble : EnsembleAxes
        Optional ensemble distributions applied by public integrators.
    probe_shape : Optional[tuple[int, int]]
        Static probe grid shape ``(H, W)``.
    """

    voltage_kv: Float[Array, " "]
    aperture_mrad: Float[Array, " "]
    defocus_ang: Float[Array, " "]
    c3_ang: Float[Array, " "]
    c5_ang: Float[Array, " "]
    ensemble: EnsembleAxes
    probe_shape: Optional[Tuple[int, int]] = eqx.field(
        default=None,
        static=True,
    )


class DetectorConfig(eqx.Module):
    """Store detector, scan, and calibration configuration.

    Calibration and collection-angle scalars are dynamic leaves. Raster shape
    is static because it controls reshaping/JIT structure. Scan-position
    arrays are optional dynamic leaves: pixel positions feed ``stem_4d`` and
    Angstrom positions feed ``stem4d_sharded``.

    Attributes
    ----------
    real_space_calib_ang : Float[Array, " "]
        Real-space pixel calibration in Angstroms.
    probe_calibration_pm : Float[Array, " "]
        Probe-construction pixel calibration in picometers.
    collection_inner_mrad : Float[Array, " "]
        Annular detector inner collection angle in milliradians.
    collection_outer_mrad : Float[Array, " "]
        Annular detector outer collection angle in milliradians.
    scan_positions_px : Optional[Float[Array, "P 2"]]
        Scan positions in pixels for ``stem_4d``.
    scan_positions_ang : Optional[Float[Array, "P 2"]]
        Scan positions in Angstroms for ``stem4d_sharded``.
    scan_shape : Optional[tuple[int, int]]
        Static raster shape ``(ny, nx)`` for detector reshaping.
    """

    real_space_calib_ang: Float[Array, " "]
    probe_calibration_pm: Float[Array, " "]
    collection_inner_mrad: Float[Array, " "]
    collection_outer_mrad: Float[Array, " "]
    scan_positions_px: Optional[Float[Array, "P 2"]] = None
    scan_positions_ang: Optional[Float[Array, "P 2"]] = None
    scan_shape: Optional[Tuple[int, int]] = eqx.field(
        default=None,
        static=True,
    )


class AtomicSliceData(eqx.Module):
    """Store on-the-fly atomic slice inputs for sharded multislice.

    This carrier covers the genuine sample-side gap in ``stem4d_sharded``:
    the precomputed atomic potential kernels and z bounds are not represented
    by ``CrystalData`` or ``PotentialSlices``.

    Attributes
    ----------
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in Angstroms, columns ``(x, y, z)``.
    atom_types : Int[Array, " N"]
        Zero-based atom type indices into ``atom_potentials``.
    slice_z_bounds : Float[Array, "S 2"]
        Z boundaries per slice, columns ``(z_min, z_max)``.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each atom type.
    atom_mask : Optional[Float[Array, " N"]]
        Optional atom inclusion mask.
    """

    atom_coords: Float[Array, "N 3"]
    atom_types: Int[Array, " N"]
    slice_z_bounds: Float[Array, "S 2"]
    atom_potentials: Float[Array, "T H W"]
    atom_mask: Optional[Float[Array, " N"]] = None


class AxisUpdate(eqx.Module):
    """Store additive distribution-axis deltas for one kernel evaluation.

    This is the single shared axis-update carrier for the Plan 03 W3 binder
    idiom. Distribution producers fold sample columns into this record, the
    binder combines records additively, and kernel-specific code applies the
    resulting perturbation without widening existing kernel signatures.
    Override-style fields should be added only when a consumer demands them.

    Attributes
    ----------
    energy_delta_ev : Float[Array, " "]
        Beam-energy offset in electronvolts.
    position_delta_ang : Float[Array, " 2"]
        Scan-position shift in Angstroms as ``(y, x)``.
    tilt_delta_mrad : Float[Array, " 2"]
        Beam-tilt offset in milliradians as ``(x, y)``.
    """

    energy_delta_ev: Float[Array, " "] = eqx.field(
        default_factory=lambda: jnp.asarray(0.0, dtype=jnp.float64)
    )
    position_delta_ang: Float[Array, " 2"] = eqx.field(
        default_factory=lambda: jnp.zeros((2,), dtype=jnp.float64)
    )
    tilt_delta_mrad: Float[Array, " 2"] = eqx.field(
        default_factory=lambda: jnp.zeros((2,), dtype=jnp.float64)
    )


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
def combine_axis_updates(updates: tuple[AxisUpdate, ...]) -> AxisUpdate:
    """Sum a tuple of additive axis-update carriers.

    Parameters
    ----------
    updates : tuple[AxisUpdate, ...]
        Axis updates to combine. An empty tuple returns the zero-effect
        update.

    Returns
    -------
    combined_update : AxisUpdate
        Additive sum of all supplied deltas.
    """
    if len(updates) == 0:
        combined_update: AxisUpdate = create_axis_update()
        return combined_update

    position_delta_ang: Float[Array, " 2"] = jnp.sum(
        jnp.stack(tuple(update.position_delta_ang for update in updates)),
        axis=0,
    )
    energy_delta_ev: Float[Array, " "] = jnp.sum(
        jnp.stack(tuple(update.energy_delta_ev for update in updates)),
    )
    tilt_delta_mrad: Float[Array, " 2"] = jnp.sum(
        jnp.stack(tuple(update.tilt_delta_mrad for update in updates)),
        axis=0,
    )
    combined_update: AxisUpdate = create_axis_update(
        position_delta_ang=position_delta_ang,
        energy_delta_ev=energy_delta_ev,
        tilt_delta_mrad=tilt_delta_mrad,
    )
    return combined_update


@jaxtyped(typechecker=beartype)
def create_axis_update(
    position_delta_ang: Optional[Float[Array, " 2"]] = None,
    energy_delta_ev: Optional[Float[Array, " "]] = None,
    tilt_delta_mrad: Optional[Float[Array, " 2"]] = None,
) -> AxisUpdate:
    """Create an AxisUpdate with structural and runtime validation.

    Parameters
    ----------
    position_delta_ang : Optional[Float[Array, " 2"]], optional
        Scan-position shift in Angstroms. Default: ``[0, 0]``.
    energy_delta_ev : Optional[Float[Array, " "]], optional
        Beam-energy offset in electronvolts. Default: ``0``.
    tilt_delta_mrad : Optional[Float[Array, " 2"]], optional
        Beam-tilt offset in milliradians. Default: ``[0, 0]``.

    Returns
    -------
    axis_update : AxisUpdate
        Validated additive update carrier.

    Raises
    ------
    ValueError
        If any supplied value has the wrong static shape.

    Notes
    -----
    1. Replace missing fields with zero-effect values.
    2. Validate static shapes before tracing.
    3. Validate finiteness with ``eqx.error_if``.
    """
    position_arr: Float[Array, " 2"] = _axis_update_vector(
        position_delta_ang,
        "position_delta_ang",
    )
    energy_arr: Float[Array, " "] = _axis_update_scalar(
        energy_delta_ev,
        "energy_delta_ev",
    )
    tilt_arr: Float[Array, " 2"] = _axis_update_vector(
        tilt_delta_mrad,
        "tilt_delta_mrad",
    )

    checked_position: Float[Array, " 2"] = eqx.error_if(
        position_arr,
        jnp.any(~jnp.isfinite(position_arr)),
        "position_delta_ang must be finite",
    )
    checked_energy: Float[Array, " "] = eqx.error_if(
        energy_arr,
        ~jnp.isfinite(energy_arr),
        "energy_delta_ev must be finite",
    )
    checked_tilt: Float[Array, " 2"] = eqx.error_if(
        tilt_arr,
        jnp.any(~jnp.isfinite(tilt_arr)),
        "tilt_delta_mrad must be finite",
    )
    axis_update: AxisUpdate = AxisUpdate(
        position_delta_ang=checked_position,
        energy_delta_ev=checked_energy,
        tilt_delta_mrad=checked_tilt,
    )
    return axis_update


@jaxtyped(typechecker=beartype)
def create_ensemble_axes(
    probe_modes: Optional[Distribution] = None,
    position_jitter: Optional[Distribution] = None,
    coherence: Optional[Distribution] = None,
) -> EnsembleAxes:
    """Create EnsembleAxes with structural validation.

    Parameters
    ----------
    probe_modes : Optional[Distribution], optional
        Probe-mode distribution with one sample column.
    position_jitter : Optional[Distribution], optional
        Position-jitter distribution with two sample columns.
    coherence : Optional[Distribution], optional
        Chromatic/angular coherence distribution with three sample columns.

    Returns
    -------
    ensemble : EnsembleAxes
        Validated optional ensemble-axis carrier.

    Raises
    ------
    ValueError
        If a present distribution has the wrong sample rank, width, or
        incompatible static axis id.
    """
    _validate_distribution_axis(probe_modes, "probe_modes", 1)
    _validate_distribution_axis(position_jitter, "position_jitter", 2)
    _validate_distribution_axis(coherence, "coherence", 3)
    ensemble: EnsembleAxes = EnsembleAxes(
        probe_modes=probe_modes,
        position_jitter=position_jitter,
        coherence=coherence,
    )
    return ensemble


@jaxtyped(typechecker=beartype)
def create_microscope_config(
    voltage_kv: scalar_num,
    aperture_mrad: scalar_num,
    defocus_ang: scalar_num = 0.0,
    c3_ang: scalar_num = 0.0,
    c5_ang: scalar_num = 0.0,
    ensemble: Optional[EnsembleAxes] = None,
    probe_shape: Optional[Tuple[int, int] | Int[Array, " 2"]] = None,
) -> MicroscopeConfig:
    """Create a MicroscopeConfig with structural and runtime validation.

    Physical scalars remain dynamic leaves. ``probe_shape`` is stored as a
    static Python tuple because it controls array construction.
    """
    voltage_arr: Float[Array, ""] = _scalar_float_array(
        voltage_kv,
        "voltage_kv",
    )
    aperture_arr: Float[Array, ""] = _scalar_float_array(
        aperture_mrad,
        "aperture_mrad",
    )
    defocus_arr: Float[Array, ""] = _scalar_float_array(
        defocus_ang,
        "defocus_ang",
    )
    c3_arr: Float[Array, ""] = _scalar_float_array(c3_ang, "c3_ang")
    c5_arr: Float[Array, ""] = _scalar_float_array(c5_ang, "c5_ang")
    probe_shape_tuple: Optional[Tuple[int, int]] = _optional_shape_tuple(
        probe_shape,
        "probe_shape",
    )
    ensemble_axes: EnsembleAxes = (
        create_ensemble_axes() if ensemble is None else ensemble
    )

    checked_voltage: Float[Array, ""] = _checked_positive_scalar(
        voltage_arr,
        "voltage_kv",
    )
    checked_aperture: Float[Array, ""] = _checked_positive_scalar(
        aperture_arr,
        "aperture_mrad",
    )
    checked_defocus: Float[Array, ""] = _checked_finite_scalar(
        defocus_arr,
        "defocus_ang",
    )
    checked_c3: Float[Array, ""] = _checked_finite_scalar(c3_arr, "c3_ang")
    checked_c5: Float[Array, ""] = _checked_finite_scalar(c5_arr, "c5_ang")
    microscope: MicroscopeConfig = MicroscopeConfig(
        voltage_kv=checked_voltage,
        aperture_mrad=checked_aperture,
        defocus_ang=checked_defocus,
        c3_ang=checked_c3,
        c5_ang=checked_c5,
        ensemble=ensemble_axes,
        probe_shape=probe_shape_tuple,
    )
    return microscope


@jaxtyped(typechecker=beartype)
def create_detector_config(
    real_space_calib_ang: scalar_float,
    probe_calibration_pm: Optional[scalar_float] = None,
    collection_inner_mrad: scalar_float = 0.0,
    collection_outer_mrad: scalar_float = 0.0,
    scan_positions_px: Optional[Float[Array, "..."]] = None,
    scan_positions_ang: Optional[Float[Array, "..."]] = None,
    scan_shape: Optional[Tuple[int, int] | Int[Array, " 2"]] = None,
) -> DetectorConfig:
    """Create a DetectorConfig with structural and runtime validation."""
    real_calib_arr: Float[Array, ""] = _scalar_float_array(
        real_space_calib_ang,
        "real_space_calib_ang",
    )
    probe_calib_arr: Float[Array, ""] = (
        real_calib_arr * 100.0
        if probe_calibration_pm is None
        else _scalar_float_array(probe_calibration_pm, "probe_calibration_pm")
    )
    inner_arr: Float[Array, ""] = _scalar_float_array(
        collection_inner_mrad,
        "collection_inner_mrad",
    )
    outer_arr: Float[Array, ""] = _scalar_float_array(
        collection_outer_mrad,
        "collection_outer_mrad",
    )
    positions_px_arr: Optional[Float[Array, "P 2"]] = _optional_positions(
        scan_positions_px,
        "scan_positions_px",
    )
    positions_ang_arr: Optional[Float[Array, "P 2"]] = _optional_positions(
        scan_positions_ang,
        "scan_positions_ang",
    )
    scan_shape_tuple: Optional[Tuple[int, int]] = _optional_shape_tuple(
        scan_shape,
        "scan_shape",
    )

    checked_real_calib: Float[Array, ""] = _checked_positive_scalar(
        real_calib_arr,
        "real_space_calib_ang",
    )
    checked_probe_calib: Float[Array, ""] = _checked_positive_scalar(
        probe_calib_arr,
        "probe_calibration_pm",
    )
    checked_inner: Float[Array, ""] = _checked_nonnegative_scalar(
        inner_arr,
        "collection_inner_mrad",
    )
    checked_outer: Float[Array, ""] = _checked_nonnegative_scalar(
        outer_arr,
        "collection_outer_mrad",
    )
    checked_outer = eqx.error_if(
        checked_outer,
        checked_outer < checked_inner,
        "collection_outer_mrad must be >= collection_inner_mrad",
    )
    checked_positions_px: Optional[Float[Array, "P 2"]] = (
        None
        if positions_px_arr is None
        else eqx.error_if(
            positions_px_arr,
            jnp.any(~jnp.isfinite(positions_px_arr)),
            "scan_positions_px contain non-finite values",
        )
    )
    checked_positions_ang: Optional[Float[Array, "P 2"]] = (
        None
        if positions_ang_arr is None
        else eqx.error_if(
            positions_ang_arr,
            jnp.any(~jnp.isfinite(positions_ang_arr)),
            "scan_positions_ang contain non-finite values",
        )
    )

    detector: DetectorConfig = DetectorConfig(
        real_space_calib_ang=checked_real_calib,
        probe_calibration_pm=checked_probe_calib,
        collection_inner_mrad=checked_inner,
        collection_outer_mrad=checked_outer,
        scan_positions_px=checked_positions_px,
        scan_positions_ang=checked_positions_ang,
        scan_shape=scan_shape_tuple,
    )
    return detector


@jaxtyped(typechecker=beartype)
def create_atomic_slice_data(
    atom_coords: Float[Array, "..."],
    atom_types: Int[Array, "..."],
    slice_z_bounds: Float[Array, "..."],
    atom_potentials: Float[Array, "..."],
    atom_mask: Optional[Float[Array, "..."]] = None,
) -> AtomicSliceData:
    """Create AtomicSliceData with structural and runtime validation."""
    coords_arr: Float[Array, "N 3"] = jnp.asarray(
        atom_coords,
        dtype=jnp.float64,
    )
    types_arr: Int[Array, " N"] = jnp.asarray(atom_types, dtype=jnp.int32)
    bounds_arr: Float[Array, "S 2"] = jnp.asarray(
        slice_z_bounds,
        dtype=jnp.float64,
    )
    potentials_arr: Float[Array, "T H W"] = jnp.asarray(
        atom_potentials,
        dtype=jnp.float64,
    )
    mask_arr: Optional[Float[Array, " N"]] = (
        None
        if atom_mask is None
        else jnp.asarray(atom_mask, dtype=jnp.float64)
    )

    if coords_arr.ndim != _MATRIX_RANK or coords_arr.shape[1] != _XYZ_COORDS:
        raise ValueError("atom_coords must have shape (N, 3)")
    if types_arr.ndim != _VECTOR_RANK or types_arr.shape != (
        coords_arr.shape[0],
    ):
        raise ValueError("atom_types must have shape (N,)")
    if bounds_arr.ndim != _MATRIX_RANK or bounds_arr.shape[1] != _XY_COORDS:
        raise ValueError("slice_z_bounds must have shape (S, 2)")
    if potentials_arr.ndim != _CUBE_RANK:
        raise ValueError("atom_potentials must have shape (T, H, W)")
    if mask_arr is not None and mask_arr.shape != (coords_arr.shape[0],):
        raise ValueError("atom_mask must have shape (N,)")

    checked_coords: Float[Array, "N 3"] = eqx.error_if(
        coords_arr,
        jnp.any(~jnp.isfinite(coords_arr)),
        "atom_coords contain non-finite values",
    )
    checked_types: Int[Array, " N"] = eqx.error_if(
        types_arr,
        jnp.any(types_arr < 0),
        "atom_types must be non-negative",
    )
    checked_types = eqx.error_if(
        checked_types,
        jnp.any(checked_types >= potentials_arr.shape[0]),
        "atom_types must index atom_potentials",
    )
    checked_bounds: Float[Array, "S 2"] = eqx.error_if(
        bounds_arr,
        jnp.any(~jnp.isfinite(bounds_arr)),
        "slice_z_bounds contain non-finite values",
    )
    checked_bounds = eqx.error_if(
        checked_bounds,
        jnp.any(checked_bounds[:, 1] - checked_bounds[:, 0] <= 0),
        "slice_z_bounds thicknesses must be positive",
    )
    checked_potentials: Float[Array, "T H W"] = eqx.error_if(
        potentials_arr,
        jnp.any(~jnp.isfinite(potentials_arr)),
        "atom_potentials contain non-finite values",
    )
    checked_mask: Optional[Float[Array, " N"]] = (
        None
        if mask_arr is None
        else eqx.error_if(
            mask_arr,
            jnp.any(~jnp.isfinite(mask_arr)),
            "atom_mask contains non-finite values",
        )
    )

    atomic_slice_data: AtomicSliceData = AtomicSliceData(
        atom_coords=checked_coords,
        atom_types=checked_types,
        slice_z_bounds=checked_bounds,
        atom_potentials=checked_potentials,
        atom_mask=checked_mask,
    )
    return atomic_slice_data


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
    3. Require finite data and finite, positive calibrations with traced
       error checks.
    4. Create and return a CalibratedArray.
    """
    data_array_arr = jnp.asarray(data_array)
    calib_y_arr: Float[Array, ""] = jnp.asarray(calib_y, dtype=jnp.float64)
    calib_x_arr: Float[Array, ""] = jnp.asarray(calib_x, dtype=jnp.float64)
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

    checked_data_array = eqx.error_if(
        data_array_arr,
        jnp.any(~jnp.isfinite(data_array_arr)),
        "data_array contains non-finite values",
    )
    checked_calib_y: Float[Array, ""] = _checked_positive_scalar(
        calib_y_arr,
        "calib_y",
    )
    checked_calib_x: Float[Array, ""] = _checked_positive_scalar(
        calib_x_arr,
        "calib_x",
    )
    calibrated_array: CalibratedArray = CalibratedArray(
        data_array=checked_data_array,
        calib_y=checked_calib_y,
        calib_x=checked_calib_x,
        real_space=real_space_arr,
    )
    return calibrated_array


def _validate_distribution_axis(
    axis: Optional[Distribution],
    expected_axis_id: str,
    expected_dim: int,
) -> None:
    """Validate static structure for an optional distribution axis."""
    if axis is None:
        return
    if axis.samples.ndim != _MATRIX_RANK:
        raise ValueError(f"{expected_axis_id}.samples must be 2D")
    if axis.samples.shape[1] != expected_dim:
        raise ValueError(
            f"{expected_axis_id}.samples must have width {expected_dim}"
        )
    if axis.weights.ndim != _VECTOR_RANK or axis.weights.shape != (
        axis.samples.shape[0],
    ):
        raise ValueError(f"{expected_axis_id}.weights must have shape (N,)")
    if axis.axis_id not in (None, expected_axis_id):
        raise ValueError(
            f"{expected_axis_id}.axis_id must be {expected_axis_id!r}"
        )


def _scalar_float_array(value: scalar_num, name: str) -> Float[Array, ""]:
    """Return a float64 scalar array with static shape validation."""
    value_arr: Float[Array, ""] = jnp.asarray(value, dtype=jnp.float64)
    if value_arr.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value_arr


def _checked_finite_scalar(
    value: Float[Array, ""],
    name: str,
) -> Float[Array, ""]:
    """Return a scalar with a traced finiteness check."""
    return eqx.error_if(
        value,
        ~jnp.isfinite(value),
        f"{name} must be finite",
    )


def _checked_positive_scalar(
    value: Float[Array, ""],
    name: str,
) -> Float[Array, ""]:
    """Return a scalar with traced finite and positive checks."""
    checked_value: Float[Array, ""] = _checked_finite_scalar(value, name)
    return eqx.error_if(
        checked_value,
        checked_value <= 0,
        f"{name} must be positive",
    )


def _checked_nonnegative_scalar(
    value: Float[Array, ""],
    name: str,
) -> Float[Array, ""]:
    """Return a scalar with traced finite and non-negative checks."""
    checked_value: Float[Array, ""] = _checked_finite_scalar(value, name)
    return eqx.error_if(
        checked_value,
        checked_value < 0,
        f"{name} must be non-negative",
    )


def _optional_positions(
    positions: Optional[Float[Array, "..."]],
    name: str,
) -> Optional[Float[Array, "P 2"]]:
    """Return optional scan positions with static shape validation."""
    if positions is None:
        return None
    positions_arr: Float[Array, "P 2"] = jnp.asarray(
        positions,
        dtype=jnp.float64,
    )
    if (
        positions_arr.ndim != _MATRIX_RANK
        or positions_arr.shape[1] != _XY_COORDS
    ):
        raise ValueError(f"{name} must have shape (P, 2)")
    return positions_arr


def _optional_shape_tuple(
    shape: Optional[Tuple[int, int] | Int[Array, " 2"]],
    name: str,
) -> Optional[Tuple[int, int]]:
    """Return optional static ``(H, W)`` shape as a Python tuple."""
    if shape is None:
        return None
    if isinstance(shape, tuple | list):
        if len(shape) != _XY_COORDS:
            raise ValueError(f"{name} must have shape (2,)")
        shape_tuple: Tuple[int, int] = (int(shape[0]), int(shape[1]))
        if shape_tuple[0] <= 0 or shape_tuple[1] <= 0:
            raise ValueError(f"{name} entries must be positive")
        return shape_tuple
    shape_arr: Int[Array, " 2"] = jnp.asarray(shape)
    if shape_arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    shape_tuple = (int(shape_arr[0]), int(shape_arr[1]))
    if shape_tuple[0] <= 0 or shape_tuple[1] <= 0:
        raise ValueError(f"{name} entries must be positive")
    return shape_tuple


def _axis_update_scalar(
    value: Optional[Float[Array, " "]],
    name: str,
) -> Float[Array, " "]:
    """Return a scalar axis-update field with static shape validation."""
    scalar: Float[Array, " "] = (
        jnp.asarray(0.0, dtype=jnp.float64)
        if value is None
        else jnp.asarray(value, dtype=jnp.float64)
    )
    if scalar.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return scalar


def _axis_update_vector(
    value: Optional[Float[Array, " 2"]],
    name: str,
) -> Float[Array, " 2"]:
    """Return a two-vector axis-update field with static shape validation."""
    vector: Float[Array, " 2"] = (
        jnp.zeros((2,), dtype=jnp.float64)
        if value is None
        else jnp.asarray(value, dtype=jnp.float64)
    )
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    return vector


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
    weights_arr: Float[Array, " M"] = jnp.asarray(weights, dtype=jnp.float64)
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
    checked_calib: Float[Array, ""] = _checked_positive_scalar(
        calib_arr,
        "calib",
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
    slices_arr: Float[Array, "H W S"] = jnp.asarray(slices, dtype=jnp.float64)
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
    checked_thickness: Float[Array, ""] = _checked_positive_scalar(
        thickness_arr,
        "slice_thickness",
    )
    checked_calib: Float[Array, ""] = _checked_positive_scalar(
        calib_arr,
        "calib",
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
    voltage_arr: Float[Array, ""] = jnp.asarray(voltage_kv, dtype=jnp.float64)

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
    checked_real_calib: Float[Array, ""] = _checked_positive_scalar(
        real_calib_arr,
        "real_space_calib",
    )
    checked_fourier_calib: Float[Array, ""] = _checked_positive_scalar(
        fourier_calib_arr,
        "fourier_space_calib",
    )
    checked_voltage: Float[Array, ""] = _checked_positive_scalar(
        voltage_arr,
        "voltage_kv",
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
    "AxisUpdate",
    "AtomicSliceData",
    "CalibratedArray",
    "DetectorConfig",
    "EnsembleAxes",
    "MicroscopeConfig",
    "PotentialSlices",
    "ProbeModes",
    "STEM4D",
    "combine_axis_updates",
    "create_axis_update",
    "create_atomic_slice_data",
    "create_calibrated_array",
    "create_detector_config",
    "create_ensemble_axes",
    "create_microscope_config",
    "create_potential_slices",
    "create_probe_modes",
    "create_stem4d",
]
