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
:func:`checked_cbed`
    Validate CBED inputs and run the bare CBED kernel.
:func:`checked_make_probe`
    Validate probe-construction inputs and run the bare probe
    kernel.
:func:`checked_stem4d_sharded`
    Validate sharded 4D-STEM inputs and run the bare sharded
    4D-STEM kernel.
:func:`checked_stem_4d`
    Validate 4D-STEM inputs and run the bare 4D-STEM kernel.

Notes
-----
The wrappers are not JIT-compiled themselves. Callers choose the
transformation context, and the runtime checks remain compatible with
``jax.jit``, ``jax.grad``, and ``jax.vmap``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jax.sharding import Mesh
from jaxtyping import Array, Complex, Float, Int, Num, jaxtyped

from ptyrodactyl.types import (
    STEM4D,
    CalibratedArray,
    PotentialSlices,
    ProbeModes,
    scalar_float,
    scalar_num,
)

from .parallelized import stem4d_sharded
from .simulations import cbed, make_probe, stem_4d

_VECTOR_RANK = 1
_MATRIX_RANK = 2
_CUBE_RANK = 3
_XY_COORDS = 2
_XYZ_COORDS = 3
_SINGLE_MODE = 1


def _shape(value: object) -> tuple[int, ...]:
    return tuple(jnp.shape(value))


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
    return eqx.error_if(
        checked_value,
        checked_value <= 0,
        f"{name} must be positive",
    )


def _potential_grid_shape(
    pot_slices: PotentialSlices,
    name: str,
) -> tuple[int, int]:
    slices_shape: tuple[int, ...] = _shape(pot_slices.slices)
    _raise_if(
        len(slices_shape) not in (_MATRIX_RANK, _CUBE_RANK),
        f"{name}.slices must be 2D or 3D",
    )
    return slices_shape[0], slices_shape[1]


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
    return modes_shape[0], modes_shape[1]


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
    return eqx.error_if(
        checked_pot_slices,
        jnp.asarray(checked_pot_slices.calib) <= 0,
        f"{name}.calib must be positive",
    )


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
    return eqx.error_if(
        checked_beam,
        jnp.asarray(checked_beam.calib) <= 0,
        "beam.calib must be positive",
    )


def _validate_positions_structure(
    positions: object,
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
    return eqx.error_if(
        checked_positions,
        jnp.any(
            (checked_positions[:, 0] < 0)
            | (checked_positions[:, 1] < 0)
            | (checked_positions[:, 0] >= height)
            | (checked_positions[:, 1] >= width)
        ),
        "positions must be within pot_slice grid bounds",
    )


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
    return (
        checked_probe_modes,
        checked_scan_positions,
        checked_atom_coords,
        checked_slice_z_bounds,
        checked_atom_potentials,
    )


@jaxtyped(typechecker=beartype)
def checked_make_probe(
    aperture: scalar_num,
    voltage: scalar_num,
    image_size: Union[Tuple[int, int], Int[Array, " 2"]],
    calibration_pm: scalar_float,
    defocus: scalar_num = 0.0,
    c3: scalar_num = 0.0,
    c5: scalar_num = 0.0,
) -> Complex[Array, " h w"]:
    """Validate probe-construction inputs and run the bare probe kernel.

    Parameters
    ----------
    aperture : scalar_num
        Aperture semi-angle in milliradians.
    voltage : scalar_num
        Accelerating voltage in kiloelectronvolts.
    image_size : Tuple[int, int] | Int[Array, " 2"]
        Grid size in pixels ``(H, W)``.
    calibration_pm : scalar_float
        Real-space pixel size in picometers.
    defocus : scalar_num, optional
        Defocus in Angstroms. Default is 0.
    c3 : scalar_num, optional
        Third-order spherical aberration in Angstroms.
        Default is 0.
    c5 : scalar_num, optional
        Fifth-order spherical aberration in Angstroms.
        Default is 0.

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
    """
    image_size_arr: Int[Array, " 2"] = jnp.asarray(image_size)
    _raise_if(
        image_size_arr.shape != (_XY_COORDS,),
        "image_size must have shape (2,)",
    )

    with jax.ensure_compile_time_eval():
        static_image_size: tuple[int, int] = (
            int(image_size_arr[0]),
            int(image_size_arr[1]),
        )

    checked_aperture: Num[Array, ""] = _checked_positive_scalar(
        aperture,
        "aperture",
    )
    checked_voltage: Num[Array, ""] = _checked_positive_scalar(
        voltage,
        "voltage",
    )
    checked_image_size: Int[Array, " 2"] = eqx.error_if(
        image_size_arr,
        jnp.any(~jnp.isfinite(image_size_arr)),
        "image_size entries must be finite",
    )
    checked_image_size = eqx.error_if(
        checked_image_size,
        jnp.any(checked_image_size <= 0),
        "image_size entries must be positive",
    )
    checked_aperture = checked_aperture + (
        jnp.zeros_like(checked_aperture) * jnp.sum(checked_image_size)
    )
    checked_calibration_pm: Num[Array, ""] = _checked_positive_scalar(
        calibration_pm,
        "calibration_pm",
    )

    # The public type wrapper rejects the static tuple needed by arange in JIT.
    probe_real_space: Complex[Array, " h w"] = make_probe.__wrapped__(
        aperture=checked_aperture,
        voltage=checked_voltage,
        image_size=static_image_size,
        calibration_pm=checked_calibration_pm,
        defocus=defocus,
        c3=c3,
        c5=c5,
    )
    return probe_real_space


@jaxtyped(typechecker=beartype)
def checked_cbed(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    voltage_kv: scalar_num,
) -> CalibratedArray:
    """Validate CBED inputs and run the bare CBED kernel.

    Parameters
    ----------
    pot_slices : PotentialSlices
        Potential slices for multislice propagation.
    beam : ProbeModes
        Electron beam modes.
    voltage_kv : scalar_num
        Accelerating voltage in kilovolts.

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
    """
    _validate_cbed_structure(pot_slices, beam, "pot_slices")
    checked_pot_slices: PotentialSlices = _checked_potential_slices(
        pot_slices,
        "pot_slices",
    )
    checked_beam: ProbeModes = _checked_beam(beam)
    checked_voltage: Num[Array, ""] = _checked_positive_scalar(
        voltage_kv,
        "voltage_kv",
    )

    cbed_pytree: CalibratedArray = cbed(
        pot_slices=checked_pot_slices,
        beam=checked_beam,
        voltage_kv=checked_voltage,
    )
    return cbed_pytree


@jaxtyped(typechecker=beartype)
def checked_stem_4d(
    pot_slice: PotentialSlices,
    beam: ProbeModes,
    positions: Num[Array, "#P 2"],
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
) -> STEM4D:
    """Validate 4D-STEM inputs and run the bare 4D-STEM kernel.

    Parameters
    ----------
    pot_slice : PotentialSlices
        Potential slices for the sample.
    beam : ProbeModes
        Electron beam modes.
    positions : Num[Array, "#P 2"]
        Scan positions ``(y, x)`` in pixels.
    voltage_kv : scalar_num
        Accelerating voltage in kilovolts.
    calib_ang : scalar_float
        Pixel size in Angstroms.

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
    """
    _validate_cbed_structure(pot_slice, beam, "pot_slice")
    _validate_positions_structure(positions, "positions")

    checked_pot_slice: PotentialSlices = _checked_potential_slices(
        pot_slice,
        "pot_slice",
    )
    checked_beam: ProbeModes = _checked_beam(beam)
    checked_positions: Num[Array, "#P 2"] = _checked_positions_in_pixels(
        positions,
        _potential_grid_shape(pot_slice, "pot_slice"),
    )
    checked_voltage: Num[Array, ""] = _checked_positive_scalar(
        voltage_kv,
        "voltage_kv",
    )
    checked_calib_ang: Num[Array, ""] = _checked_positive_scalar(
        calib_ang,
        "calib_ang",
    )

    stem4d_data: STEM4D = stem_4d(
        pot_slice=checked_pot_slice,
        beam=checked_beam,
        positions=checked_positions,
        voltage_kv=checked_voltage,
        calib_ang=checked_calib_ang,
    )
    return stem4d_data


@jaxtyped(typechecker=beartype)
def checked_stem4d_sharded(
    probe_modes: Complex[Array, "H W M"],
    scan_positions_ang: Float[Array, "P 2"],
    atom_coords: Float[Array, "N 3"],
    atom_types: Int[Array, " N"],
    slice_z_bounds: Float[Array, "S 2"],
    atom_potentials: Float[Array, "T H W"],
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
    mesh: Optional[Mesh] = None,
) -> STEM4D:
    """Validate sharded 4D-STEM inputs and run the bare sharded kernel.

    Parameters
    ----------
    probe_modes : Complex[Array, "H W M"]
        Base electron probe modes.
    scan_positions_ang : Float[Array, "P 2"]
        Scan positions in Angstroms, columns ``(y, x)``.
    atom_coords : Float[Array, "N 3"]
        Atom coordinates in Angstroms, columns ``(x, y, z)``.
    atom_types : Int[Array, " N"]
        Atom type indices.
    slice_z_bounds : Float[Array, "S 2"]
        Z boundaries per slice, columns ``(z_min, z_max)``.
    atom_potentials : Float[Array, "T H W"]
        Precomputed 2D atomic potentials for each atom type.
    voltage_kv : scalar_num
        Accelerating voltage in kilovolts.
    calib_ang : scalar_float
        Real-space pixel size in Angstroms.
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
    """
    _validate_sharded_structure(
        probe_modes,
        scan_positions_ang,
        atom_coords,
        atom_types,
        slice_z_bounds,
        atom_potentials,
    )
    checked_voltage: Num[Array, ""] = _checked_positive_scalar(
        voltage_kv,
        "voltage_kv",
    )
    checked_calib_ang: Num[Array, ""] = _checked_positive_scalar(
        calib_ang,
        "calib_ang",
    )
    checked_probe_modes: Complex[Array, "H W M"]
    checked_scan_positions_ang: Float[Array, "P 2"]
    checked_atom_coords: Float[Array, "N 3"]
    checked_slice_z_bounds: Float[Array, "S 2"]
    checked_atom_potentials: Float[Array, "T H W"]
    (
        checked_probe_modes,
        checked_scan_positions_ang,
        checked_atom_coords,
        checked_slice_z_bounds,
        checked_atom_potentials,
    ) = _checked_sharded_arrays(
        probe_modes,
        scan_positions_ang,
        atom_coords,
        slice_z_bounds,
        atom_potentials,
        checked_calib_ang,
    )

    stem4d_data_sharded: STEM4D = stem4d_sharded(
        probe_modes=checked_probe_modes,
        scan_positions_ang=checked_scan_positions_ang,
        atom_coords=checked_atom_coords,
        atom_types=atom_types,
        slice_z_bounds=checked_slice_z_bounds,
        atom_potentials=checked_atom_potentials,
        voltage_kv=checked_voltage,
        calib_ang=checked_calib_ang,
        mesh=mesh,
    )
    return stem4d_data_sharded


__all__: list[str] = [
    "checked_cbed",
    "checked_make_probe",
    "checked_stem4d_sharded",
    "checked_stem_4d",
]
