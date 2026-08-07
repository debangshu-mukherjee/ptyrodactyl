"""JIT-safe validating wrappers for simulation kernels.

Extended Summary
----------------
This module provides opt-in wrappers around the bare simulator
kernels. The wrappers perform traced, JIT-compatible input
validation with :func:`equinox.error_if`, then tail-call the
corresponding bare kernel without changing its behavior on valid
inputs.

Routine Listings
----------------
:func:`checked_cbed_image`
    Validate CBED inputs and run the bare CBED intensity kernel.
:func:`checked_make_probe`
    Validate probe-construction inputs and run the bare probe kernel.
:func:`checked_stem4d_sharded`
    Validate sharded 4D-STEM inputs and run the bare sharded kernel.
:func:`checked_stem_4d`
    Validate 4D-STEM inputs and run the bare 4D-STEM kernel.

Notes
-----
The wrappers are not JIT-compiled themselves. Callers choose the
transformation context, and the runtime checks remain compatible with
``jax.jit``, ``jax.grad``, and ``jax.vmap``.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jax.sharding import Mesh
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from ptyrodactyl.multislice.simulations import cbed_image, make_probe, stem_4d
from ptyrodactyl.types import (
    STEM4D,
    AtomicSliceData,
    CalibratedArray,
    DetectorConfig,
    MicroscopeConfig,
    PotentialSlices,
    ProbeModes,
    create_atomic_slice_data,
    scalar_num,
)

from .parallelized import stem4d_sharded

_VECTOR_RANK = 1
_MATRIX_RANK = 2
_CUBE_RANK = 3
_XY_COORDS = 2
_XYZ_COORDS = 3
_SINGLE_MODE = 1


def _shape(value: Array) -> tuple[int, ...]:
    result: tuple[int, ...] = tuple(jnp.shape(value))
    return result


def _raise_if(condition: bool, message: str) -> None:
    if condition:
        raise ValueError(message)


def _checked_positive_scalar(
    value: scalar_num,
    name: str,
) -> Num[Array, ""]:
    value_arr: Num[Array, ""] = jnp.asarray(value)
    _raise_if(value_arr.shape != (), f"{name} must be a scalar")

    checked_value: Num[Array, ""] = eqx.error_if(
        value_arr,
        ~jnp.isfinite(value_arr),
        f"{name} must be finite",
    )
    result: Num[Array, ""] = eqx.error_if(
        checked_value,
        checked_value <= 0,
        f"{name} must be positive",
    )
    return result


def _checked_finite_scalar(
    value: scalar_num,
    name: str,
) -> Num[Array, ""]:
    value_arr: Num[Array, ""] = jnp.asarray(value)
    _raise_if(value_arr.shape != (), f"{name} must be a scalar")
    result: Num[Array, ""] = eqx.error_if(
        value_arr,
        ~jnp.isfinite(value_arr),
        f"{name} must be finite",
    )
    return result


def _checked_nonnegative_scalar(
    value: scalar_num,
    name: str,
) -> Num[Array, ""]:
    checked_value: Num[Array, ""] = _checked_finite_scalar(value, name)
    result: Num[Array, ""] = eqx.error_if(
        checked_value,
        checked_value < 0,
        f"{name} must be non-negative",
    )
    return result


def _checked_microscope(microscope: MicroscopeConfig) -> MicroscopeConfig:
    """Return a microscope carrier with traced scalar checks attached."""
    checked_values = (
        _checked_positive_scalar(microscope.voltage_kv, "voltage_kv"),
        _checked_positive_scalar(microscope.aperture_mrad, "aperture_mrad"),
        _checked_finite_scalar(microscope.defocus_ang, "defocus_ang"),
        _checked_finite_scalar(microscope.c3_ang, "c3_ang"),
        _checked_finite_scalar(microscope.c5_ang, "c5_ang"),
    )
    result: MicroscopeConfig = eqx.tree_at(
        lambda config: (
            config.voltage_kv,
            config.aperture_mrad,
            config.defocus_ang,
            config.c3_ang,
            config.c5_ang,
        ),
        microscope,
        checked_values,
    )
    return result


def _checked_detector(detector: DetectorConfig) -> DetectorConfig:
    """Return a detector carrier with traced scalar checks attached."""
    checked_inner: Num[Array, ""] = _checked_nonnegative_scalar(
        detector.collection_inner_mrad,
        "collection_inner_mrad",
    )
    checked_outer: Num[Array, ""] = _checked_nonnegative_scalar(
        detector.collection_outer_mrad,
        "collection_outer_mrad",
    )
    checked_outer = eqx.error_if(
        checked_outer,
        checked_outer < checked_inner,
        "collection_outer_mrad must be >= collection_inner_mrad",
    )
    checked_detector: DetectorConfig = eqx.tree_at(
        lambda config: (
            config.real_space_calib_ang,
            config.probe_calibration_pm,
            config.collection_inner_mrad,
            config.collection_outer_mrad,
        ),
        detector,
        (
            _checked_positive_scalar(
                detector.real_space_calib_ang,
                "real_space_calib_ang",
            ),
            _checked_positive_scalar(
                detector.probe_calibration_pm,
                "probe_calibration_pm",
            ),
            checked_inner,
            checked_outer,
        ),
    )
    if checked_detector.scan_positions_px is not None:
        checked_positions_px: Float[Array, "P 2"] = eqx.error_if(
            checked_detector.scan_positions_px,
            jnp.any(~jnp.isfinite(checked_detector.scan_positions_px)),
            "scan_positions_px contain non-finite values",
        )
        checked_detector = eqx.tree_at(
            lambda config: config.scan_positions_px,
            checked_detector,
            checked_positions_px,
        )
    if checked_detector.scan_positions_ang is not None:
        checked_positions_ang: Float[Array, "P 2"] = eqx.error_if(
            checked_detector.scan_positions_ang,
            jnp.any(~jnp.isfinite(checked_detector.scan_positions_ang)),
            "scan_positions_ang contain non-finite values",
        )
        checked_detector = eqx.tree_at(
            lambda config: config.scan_positions_ang,
            checked_detector,
            checked_positions_ang,
        )
    return checked_detector


def _potential_grid_shape(
    pot_slices: PotentialSlices,
    name: str,
) -> tuple[int, int]:
    slices_shape: tuple[int, ...] = _shape(pot_slices.slices)
    _raise_if(
        len(slices_shape) not in (_MATRIX_RANK, _CUBE_RANK),
        f"{name}.slices must be 2D or 3D",
    )
    result: tuple[int, int] = slices_shape[0], slices_shape[1]
    return result


def _beam_grid_shape(beam: ProbeModes, name: str) -> tuple[int, int]:
    modes_shape: tuple[int, ...] = _shape(beam.modes)
    weights_shape: tuple[int, ...] = _shape(beam.weights)
    _raise_if(
        len(modes_shape) not in (_MATRIX_RANK, _CUBE_RANK),
        f"{name}.modes must be 2D or 3D",
    )
    _raise_if(
        len(weights_shape) != _VECTOR_RANK,
        f"{name}.weights must be 1D",
    )

    num_modes: int = (
        _SINGLE_MODE if len(modes_shape) == _MATRIX_RANK else modes_shape[2]
    )
    _raise_if(
        weights_shape != (num_modes,),
        f"{name}.weights must have shape (M,)",
    )
    result: tuple[int, int] = modes_shape[0], modes_shape[1]
    return result


def _validate_cbed_structure(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    pot_name: str,
) -> None:
    pot_grid_shape: tuple[int, int] = _potential_grid_shape(
        pot_slices,
        pot_name,
    )
    beam_grid_shape: tuple[int, int] = _beam_grid_shape(beam, "beam")
    _raise_if(
        beam_grid_shape != pot_grid_shape,
        f"beam.modes spatial shape must match {pot_name}.slices",
    )


def _checked_potential_slices(
    pot_slices: PotentialSlices,
    name: str,
) -> PotentialSlices:
    checked_pot_slices: PotentialSlices = eqx.error_if(
        pot_slices,
        jnp.any(~jnp.isfinite(jnp.asarray(pot_slices.slices))),
        f"{name}.slices contain non-finite values",
    )
    checked_pot_slices = eqx.error_if(
        checked_pot_slices,
        ~jnp.isfinite(jnp.asarray(checked_pot_slices.slice_thickness)),
        f"{name}.slice_thickness must be finite",
    )
    checked_pot_slices = eqx.error_if(
        checked_pot_slices,
        jnp.asarray(checked_pot_slices.slice_thickness) <= 0,
        f"{name}.slice_thickness must be positive",
    )
    checked_pot_slices = eqx.error_if(
        checked_pot_slices,
        ~jnp.isfinite(jnp.asarray(checked_pot_slices.calib)),
        f"{name}.calib must be finite",
    )
    result: PotentialSlices = eqx.error_if(
        checked_pot_slices,
        jnp.asarray(checked_pot_slices.calib) <= 0,
        f"{name}.calib must be positive",
    )
    return result


def _checked_beam(beam: ProbeModes) -> ProbeModes:
    checked_beam: ProbeModes = eqx.error_if(
        beam,
        jnp.any(~jnp.isfinite(jnp.asarray(beam.modes))),
        "beam.modes contain non-finite values",
    )
    checked_beam = eqx.error_if(
        checked_beam,
        jnp.any(~jnp.isfinite(jnp.asarray(checked_beam.weights))),
        "beam.weights contain non-finite values",
    )
    checked_beam = eqx.error_if(
        checked_beam,
        ~jnp.isfinite(jnp.asarray(checked_beam.calib)),
        "beam.calib must be finite",
    )
    result: ProbeModes = eqx.error_if(
        checked_beam,
        jnp.asarray(checked_beam.calib) <= 0,
        "beam.calib must be positive",
    )
    return result


def _validate_positions_structure(
    positions: Array,
    name: str,
) -> None:
    positions_shape: tuple[int, ...] = _shape(positions)
    _raise_if(len(positions_shape) != _MATRIX_RANK, f"{name} must be 2D")
    _raise_if(
        positions_shape[1] != _XY_COORDS,
        f"{name} must have shape (P, 2)",
    )


def _checked_positions_in_pixels(
    positions: Num[Array, "#P 2"],
    grid_shape: tuple[int, int],
) -> Num[Array, "#P 2"]:
    checked_positions: Num[Array, "#P 2"] = eqx.error_if(
        positions,
        jnp.any(~jnp.isfinite(positions)),
        "positions contain non-finite values",
    )
    height: int
    width: int
    height, width = grid_shape
    result: Num[Array, "#P 2"] = eqx.error_if(
        checked_positions,
        jnp.any(
            (checked_positions[:, 0] < 0)
            | (checked_positions[:, 1] < 0)
            | (checked_positions[:, 0] >= height)
            | (checked_positions[:, 1] >= width)
        ),
        "positions must be within pot_slice grid bounds",
    )
    return result


def _validate_sharded_structure(
    probe_modes: Complex[Array, "H W M"],
    scan_positions_ang: Float[Array, "P 2"],
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
) -> None:
    probe_shape: tuple[int, ...] = _shape(probe_modes)
    scan_shape: tuple[int, ...] = _shape(scan_positions_ang)
    coords_shape: tuple[int, ...] = _shape(atom_coords)
    types_shape: tuple[int, ...] = _shape(atom_types)
    bounds_shape: tuple[int, ...] = _shape(slice_z_bounds)
    potentials_shape: tuple[int, ...] = _shape(atom_potentials)

    _raise_if(len(probe_shape) != _CUBE_RANK, "probe_modes must be 3D")
    _raise_if(
        len(scan_shape) != _MATRIX_RANK,
        "scan_positions_ang must be 2D",
    )
    _raise_if(
        scan_shape[1] != _XY_COORDS,
        "scan_positions_ang must have shape (P, 2)",
    )
    _raise_if(len(coords_shape) != _MATRIX_RANK, "atom_coords must be 2D")
    _raise_if(
        coords_shape[1] != _XYZ_COORDS,
        "atom_coords must have shape (N, 3)",
    )
    _raise_if(len(types_shape) != _VECTOR_RANK, "atom_types must be 1D")
    _raise_if(
        types_shape != (coords_shape[0],),
        "atom_types must have shape (N,)",
    )
    _raise_if(
        len(bounds_shape) != _MATRIX_RANK,
        "slice_z_bounds must be 2D",
    )
    _raise_if(
        bounds_shape[1] != _XY_COORDS,
        "slice_z_bounds must have shape (S, 2)",
    )
    _raise_if(
        len(potentials_shape) != _CUBE_RANK,
        "atom_potentials must be 3D",
    )
    _raise_if(
        probe_shape[:2] != potentials_shape[1:],
        "probe_modes spatial shape must match atom_potentials",
    )


def _checked_sharded_arrays(
    probe_modes: Complex[Array, "H W M"],
    scan_positions_ang: Float[Array, "P 2"],
    atom_coords: Float[Array, "N 3"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    calib_ang: Num[Array, ""],
) -> tuple[
    Complex[Array, "H W M"],
    Float[Array, "P 2"],
    Float[Array, "N 3"],
    Float[Array, "S 2"],
    Float[Array, "T H W"],
]:
    checked_probe_modes: Complex[Array, "H W M"] = eqx.error_if(
        probe_modes,
        jnp.any(~jnp.isfinite(probe_modes)),
        "probe_modes contain non-finite values",
    )
    checked_scan_positions: Float[Array, "P 2"] = eqx.error_if(
        scan_positions_ang,
        jnp.any(~jnp.isfinite(scan_positions_ang)),
        "scan_positions_ang contain non-finite values",
    )
    checked_atom_coords: Float[Array, "N 3"] = eqx.error_if(
        atom_coords,
        jnp.any(~jnp.isfinite(atom_coords)),
        "atom_coords contain non-finite values",
    )
    checked_slice_z_bounds: Float[Array, "S 2"] = eqx.error_if(
        slice_z_bounds,
        jnp.any(~jnp.isfinite(slice_z_bounds)),
        "slice_z_bounds contain non-finite values",
    )
    checked_slice_z_bounds = eqx.error_if(
        checked_slice_z_bounds,
        jnp.any(
            checked_slice_z_bounds[:, 1] - checked_slice_z_bounds[:, 0] <= 0
        ),
        "slice_z_bounds thicknesses must be positive",
    )
    checked_atom_potentials: Float[Array, "T H W"] = eqx.error_if(
        atom_potentials,
        jnp.any(~jnp.isfinite(atom_potentials)),
        "atom_potentials contain non-finite values",
    )

    height: int = checked_probe_modes.shape[0]
    width: int = checked_probe_modes.shape[1]
    checked_scan_positions = eqx.error_if(
        checked_scan_positions,
        jnp.any(
            (checked_scan_positions[:, 0] < 0)
            | (checked_scan_positions[:, 1] < 0)
            | (checked_scan_positions[:, 0] >= height * calib_ang)
            | (checked_scan_positions[:, 1] >= width * calib_ang)
        ),
        "scan_positions_ang must be within atom_potentials grid bounds",
    )
    result: tuple[
        Complex[Array, "H W M"],
        Float[Array, "P 2"],
        Float[Array, "N 3"],
        Float[Array, "S 2"],
        Float[Array, "T H W"],
    ] = (
        checked_probe_modes,
        checked_scan_positions,
        checked_atom_coords,
        checked_slice_z_bounds,
        checked_atom_potentials,
    )
    return result


@jaxtyped(typechecker=beartype)
def checked_make_probe(
    microscope: MicroscopeConfig,
    detector: DetectorConfig,
) -> Complex[Array, " h w"]:
    """Validate probe-construction inputs and run the bare probe kernel.

    :see: :func:`~.test_checked_make_probe_transparent_jit_grad_and_raises`

    Parameters
    ----------
    microscope : MicroscopeConfig
        Microscope voltage, aperture, aberrations, and static probe shape.
    detector : DetectorConfig
        Detector calibration carrying the probe pixel size.

    Returns
    -------
    probe_real_space : Complex[Array, " h w"]
        Electron probe wavefunction in real space.

    Raises
    ------
    EquinoxRuntimeError
        If traced numeric values are non-finite or non-positive where
        positivity is required.
    ValueError
        If ``image_size`` does not have shape ``(2,)``.

    :see: make_probe, checked_cbed_image.
    """
    _raise_if(
        microscope.probe_shape is None,
        "microscope.probe_shape is required",
    )
    checked_microscope: MicroscopeConfig = _checked_microscope(microscope)
    checked_detector: DetectorConfig = _checked_detector(detector)

    probe_real_space: Complex[Array, " h w"] = make_probe.__wrapped__(
        microscope=checked_microscope,
        detector=checked_detector,
    )
    return probe_real_space


@jaxtyped(typechecker=beartype)
def checked_cbed_image(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    microscope: MicroscopeConfig,
) -> CalibratedArray:
    """Validate CBED inputs and run the bare CBED intensity kernel.

    :see: :mod:`~.test_checked`

    Parameters
    ----------
    pot_slices : PotentialSlices
        Potential slices for multislice propagation.
    beam : ProbeModes
        Electron beam modes.
    microscope : MicroscopeConfig
        Microscope voltage and optional ensemble axes.

    Returns
    -------
    cbed_pytree : CalibratedArray
        CBED intensity pattern with Fourier-space calibrations.

    Raises
    ------
    EquinoxRuntimeError
        If traced numeric values are non-finite or non-positive where
        positivity is required.
    ValueError
        If static ranks or spatial grid shapes are incompatible.

    :see: cbed_image, checked_make_probe.
    """
    _validate_cbed_structure(pot_slices, beam, "pot_slices")
    checked_pot_slices: PotentialSlices = _checked_potential_slices(
        pot_slices,
        "pot_slices",
    )
    checked_beam: ProbeModes = _checked_beam(beam)
    checked_microscope: MicroscopeConfig = _checked_microscope(microscope)

    cbed_pytree: CalibratedArray = cbed_image(
        pot_slices=checked_pot_slices,
        beam=checked_beam,
        microscope=checked_microscope,
    )
    return cbed_pytree


@jaxtyped(typechecker=beartype)
def checked_stem_4d(
    pot_slice: PotentialSlices,
    beam: ProbeModes,
    microscope: MicroscopeConfig,
    detector: DetectorConfig,
) -> STEM4D:
    """Validate 4D-STEM inputs and run the bare 4D-STEM kernel.

    :see: :func:`~.test_checked_stem_4d_transparent_jit_grad_and_raises`

    Parameters
    ----------
    pot_slice : PotentialSlices
        Potential slices for the sample.
    beam : ProbeModes
        Electron beam modes.
    microscope : MicroscopeConfig
        Microscope voltage and optional ensemble axes.
    detector : DetectorConfig
        Scan positions in pixels and real-space calibration.

    Returns
    -------
    stem4d_data : STEM4D
        Complete 4D-STEM dataset.

    Raises
    ------
    EquinoxRuntimeError
        If traced numeric values are non-finite, non-positive where
        positivity is required, or scan positions leave the grid.
    ValueError
        If static ranks or spatial grid shapes are incompatible.

    :see: stem_4d, checked_cbed_image.
    """
    _validate_cbed_structure(pot_slice, beam, "pot_slice")
    scan_positions_px = detector.scan_positions_px
    if scan_positions_px is None:
        raise ValueError("detector.scan_positions_px is required")
    _validate_positions_structure(scan_positions_px, "positions")

    checked_pot_slice: PotentialSlices = _checked_potential_slices(
        pot_slice,
        "pot_slice",
    )
    checked_beam: ProbeModes = _checked_beam(beam)
    checked_positions: Num[Array, "#P 2"] = _checked_positions_in_pixels(
        scan_positions_px,
        _potential_grid_shape(pot_slice, "pot_slice"),
    )
    checked_microscope: MicroscopeConfig = _checked_microscope(microscope)
    checked_detector: DetectorConfig = _checked_detector(detector)
    checked_detector = eqx.tree_at(
        lambda config: config.scan_positions_px,
        checked_detector,
        checked_positions,
    )

    stem4d_data: STEM4D = stem_4d(
        pot_slice=checked_pot_slice,
        beam=checked_beam,
        microscope=checked_microscope,
        detector=checked_detector,
    )
    return stem4d_data


@jaxtyped(typechecker=beartype)
def checked_stem4d_sharded(
    probe_modes: ProbeModes,
    sample: AtomicSliceData,
    microscope: MicroscopeConfig,
    detector: DetectorConfig,
    mesh: Optional[Mesh] = None,
) -> STEM4D:
    """Validate sharded 4D-STEM inputs and run the bare sharded kernel.

    :see: :func:`~.test_checked_stem4d_sharded_transparent_jit_grad_and_raises`

    Parameters
    ----------
    probe_modes : ProbeModes
        Base electron probe modes.
    sample : AtomicSliceData
        Atom coordinates, type indices, z bounds, and potential kernels.
    microscope : MicroscopeConfig
        Microscope voltage and ensemble configuration.
    detector : DetectorConfig
        Scan positions in Angstroms and real-space calibration.
    mesh : Optional[Mesh]
        JAX device mesh for distributed execution. Default is ``None``.

    Returns
    -------
    stem4d_data_sharded : STEM4D
        Complete 4D-STEM dataset.

    Raises
    ------
    EquinoxRuntimeError
        If traced numeric values are non-finite, non-positive where
        positivity is required, or scan positions leave the grid.
    ValueError
        If static ranks or spatial grid shapes are incompatible.

    :see: stem4d_sharded, checked_stem_4d.
    """
    scan_positions_ang = detector.scan_positions_ang
    if scan_positions_ang is None:
        raise ValueError("detector.scan_positions_ang is required")
    _validate_sharded_structure(
        probe_modes.modes,
        scan_positions_ang,
        sample.atom_coords,
        sample.atom_types,
        sample.slice_z_bounds,
        sample.atom_potentials,
    )
    checked_microscope: MicroscopeConfig = _checked_microscope(microscope)
    checked_detector: DetectorConfig = _checked_detector(detector)
    checked_probe_modes: ProbeModes = _checked_beam(probe_modes)
    checked_probe_mode_array: Complex[Array, "H W M"]
    checked_scan_positions_ang: Float[Array, "P 2"]
    checked_atom_coords: Float[Array, "N 3"]
    checked_slice_z_bounds: Float[Array, "S 2"]
    checked_atom_potentials: Float[Array, "T H W"]
    (
        checked_probe_mode_array,
        checked_scan_positions_ang,
        checked_atom_coords,
        checked_slice_z_bounds,
        checked_atom_potentials,
    ) = _checked_sharded_arrays(
        probe_modes.modes,
        scan_positions_ang,
        sample.atom_coords,
        sample.slice_z_bounds,
        sample.atom_potentials,
        checked_detector.real_space_calib_ang,
    )
    checked_probe_modes = eqx.tree_at(
        lambda probe: probe.modes,
        checked_probe_modes,
        checked_probe_mode_array,
    )
    checked_sample: AtomicSliceData = create_atomic_slice_data(
        atom_coords=checked_atom_coords,
        atom_types=sample.atom_types,
        slice_z_bounds=checked_slice_z_bounds,
        atom_potentials=checked_atom_potentials,
        atom_mask=sample.atom_mask,
    )
    checked_detector = eqx.tree_at(
        lambda config: config.scan_positions_ang,
        checked_detector,
        checked_scan_positions_ang,
    )

    stem4d_data_sharded: STEM4D = stem4d_sharded(
        probe_modes=checked_probe_modes,
        sample=checked_sample,
        microscope=checked_microscope,
        detector=checked_detector,
        mesh=mesh,
    )
    return stem4d_data_sharded


__all__: list[str] = [
    "checked_cbed_image",
    "checked_make_probe",
    "checked_stem4d_sharded",
    "checked_stem_4d",
]
