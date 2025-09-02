"""
Module: ptyrodactyl.electrons.simulations
------------------------------------------
Forward simulation functions for electron microscopy and ptychography.

This module contains functions for simulating electron beam propagation,
creating probes, calculating aberrations, and generating CBED patterns
and 4D-STEM data. All functions are JAX-compatible and support automatic
differentiation.

Functions
---------
- `transmission_func`:
    Calculates transmission function for a given potential
- `propagation_func`:
    Propagates electron wave through free space
- `fourier_coords`:
    Generates Fourier space coordinates for diffraction calculations
- `fourier_calib`:
    Calculates Fourier space calibration from real space parameters
- `make_probe`:
    Creates electron probe with specified parameters and aberrations
- `aberration`:
    Applies aberration phase to electron wave
- `wavelength_ang`:
    Calculates electron wavelength from accelerating voltage
- `cbed`:
    Simulates convergent beam electron diffraction patterns
- `shift_beam_fourier`:
    Shifts electron beam in Fourier space for scanning
- `stem_4d`:
    Generates 4D-STEM data with multiple probe positions
- `stem_4d_sharded`:
    Sharded version using JAX's automatic sharding API
- `stem_4d_parallel`:
    Parallel version with explicit device control using shard_map
- `decompose_beam_to_modes`:
    Decomposes electron beam into orthogonal modes
- `annular_detector`:
    Simulates annular detector for STEM imaging from 4D data

Notes
-----
All functions are designed to work with JAX transformations including
jit, grad, and vmap. Input arrays should be properly typed and validated
using the factory functions from electron_types module.
"""

import jax
import jax.numpy as jnp
from beartype.typing import Any, Optional, Tuple, Union
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import (Array, Bool, Complex, Complex128, Float, Int, Num,
                       PRNGKeyArray)

from ptyrodactyl._decorators import beartype as typechecker
from ptyrodactyl._decorators import jaxtyped

from .electron_types import (STEM4D, CalibratedArray, PotentialSlices,
                             ProbeModes, make_calibrated_array,
                             make_probe_modes, make_stem4d, scalar_float,
                             scalar_int, scalar_numeric)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=typechecker)
def transmission_func(
    pot_slice: Float[Array, " a b"], voltage_kv: scalar_numeric
) -> Complex[Array, " a b"]:
    """Calculate the complex transmission function from a single potential slice.

    Parameters
    ----------
    pot_slice : Float[Array, " a b"]
        Potential slice in Kirkland units.
    voltage_kv : scalar_numeric
        Microscope operating voltage in kiloelectronvolts.

    Returns
    -------
    Complex[Array, " a b"]
        The transmission function of a single crystal slice.

    Notes
    -----
    Algorithm:
    - Calculate the electron energy in electronVolts
    - Calculate the wavelength in angstroms
    - Calculate the Einstein energy
    - Calculate the sigma value, which is the constant for the phase shift
    - Calculate the transmission function as a complex exponential
    """

    voltage: Float[Array, " "] = jnp.multiply(voltage_kv, jnp.asarray(1000.0))

    m_e: Float[Array, " "] = jnp.asarray(9.109383e-31)
    e_e: Float[Array, " "] = jnp.asarray(1.602177e-19)
    c: Float[Array, " "] = jnp.asarray(299792458.0)

    ev: Float[Array, " "] = jnp.multiply(e_e, voltage)
    lambda_angstrom: Float[Array, " "] = wavelength_ang(voltage_kv)
    einstein_energy: Float[Array, " "] = jnp.multiply(m_e, jnp.square(c))
    sigma: Float[Array, " "] = (
        (2 * jnp.pi / (lambda_angstrom * voltage)) * (einstein_energy + ev)
    ) / ((2 * einstein_energy) + ev)
    trans: Complex[Array, " a b"] = jnp.exp(1j * sigma * pot_slice)
    return trans


@jaxtyped(typechecker=typechecker)
def propagation_func(
    imsize_y: scalar_int,
    imsize_x: scalar_int,
    thickness_ang: scalar_numeric,
    voltage_kv: scalar_numeric,
    calib_ang: scalar_float,
) -> Complex[Array, " h w"]:
    """Calculate the complex propagation function for multislice algorithm.

    Parameters
    ----------
    imsize_y : scalar_int
        Size of the image of the propagator in y axis.
    imsize_x : scalar_int
        Size of the image of the propagator in x axis.
    thickness_ang : scalar_numeric
        Distance between the slices in angstroms.
    voltage_kv : scalar_numeric
        Accelerating voltage in kilovolts.
    calib_ang : scalar_float
        Calibration or pixel size in angstroms.

    Returns
    -------
    Complex[Array, " h w"]
        The propagation function of the same size given by imsize.

    Notes
    -----
    Algorithm:
    - Generate frequency arrays directly using fftfreq
    - Create 2D meshgrid of frequencies
    - Calculate squared sum of frequencies
    - Calculate wavelength
    - Compute the propagation function
    """
    qy: Num[Array, " h"] = jnp.fft.fftfreq(imsize_y, d=calib_ang)
    qx: Num[Array, " w"] = jnp.fft.fftfreq(imsize_x, d=calib_ang)
    lya: Num[Array, " h w"]
    lxa: Num[Array, " h w"]
    lya, lxa = jnp.meshgrid(qy, qx, indexing="ij")
    l_sq: Num[Array, " h w"] = jnp.square(lxa) + jnp.square(lya)
    lambda_angstrom: Float[Array, " "] = wavelength_ang(voltage_kv)
    prop: Complex[Array, " h w"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * l_sq
    )
    return prop


@jaxtyped(typechecker=typechecker)
def fourier_coords(
    calibration: scalar_float | Float[Array, " 2"], image_size: Int[Array, " 2"]
) -> CalibratedArray:
    """Return the Fourier coordinates for diffraction calculations.

    Parameters
    ----------
    calibration : scalar_float or Float[Array, " 2"]
        The pixel size in angstroms in real space.
    image_size : Int[Array, " 2"]
        The size of the beam in pixels.

    Returns
    -------
    CalibratedArray
        The calibrated inverse array with the following attributes:
        - data_array : Float[Array, " H W"]
            The inverse array data
        - calib_y : Float[Array, " "]
            Inverse calibration in y direction
        - calib_x : Float[Array, " "]
            Inverse calibration in x direction
        - real_space : bool
            False, indicating reciprocal space coordinates

    Notes
    -----
    Algorithm:
    - Calculate the real space field of view in y and x
    - Generate the inverse space array y and x
    - Shift the inverse space array y and x
    - Create meshgrid of shifted inverse space arrays
    - Calculate the inverse array
    - Calculate the calibration in y and x
    - Return the calibrated array
    """
    real_fov: Float[Array, " 2"] = jnp.multiply(image_size, calibration)
    inverse_arr_y: Float[Array, " h"] = (
        jnp.arange((-image_size[0] / 2), (image_size[0] / 2), 1)
    ) / real_fov[0]
    inverse_arr_x: Float[Array, " w"] = (
        jnp.arange((-image_size[1] / 2), (image_size[1] / 2), 1)
    ) / real_fov[1]
    shifter_y: Float[Array, " "] = image_size[0] // 2
    shifter_x: Float[Array, " "] = image_size[1] // 2
    inverse_shifted_y: Float[Array, " h"] = jnp.roll(inverse_arr_y, shifter_y)
    inverse_shifted_x: Float[Array, " w"] = jnp.roll(inverse_arr_x, shifter_x)
    inverse_xx: Float[Array, " h w"]
    inverse_yy: Float[Array, " h w"]
    inverse_xx, inverse_yy = jnp.meshgrid(inverse_shifted_x, inverse_shifted_y)
    inv_squared: Float[Array, " h w"] = jnp.multiply(
        inverse_yy, inverse_yy
    ) + jnp.multiply(inverse_xx, inverse_xx)
    inverse_array: Float[Array, " h w"] = inv_squared**0.5
    calib_inverse_y: Float[Array, " "] = inverse_arr_y[1] - inverse_arr_y[0]
    calib_inverse_x: Float[Array, " "] = inverse_arr_x[1] - inverse_arr_x[0]
    inverse_space: Bool[Array, ""] = False
    calibrated_inverse_array: CalibratedArray = make_calibrated_array(
        inverse_array, calib_inverse_y, calib_inverse_x, inverse_space
    )
    return calibrated_inverse_array


@jaxtyped(typechecker=typechecker)
def fourier_calib(
    real_space_calib: Float[Array, " "] | Float[Array, " 2"],
    sizebeam: Int[Array, " 2"],
) -> Float[Array, " 2"]:
    """Generate the Fourier calibration for the beam.

    Parameters
    ----------
    real_space_calib : Float[Array, " "] or Float[Array, " 2"]
        The pixel size in angstroms in real space.
    sizebeam : Int[Array, " 2"]
        The size of the beam in pixels.

    Returns
    -------
    Float[Array, " 2"]
        The Fourier calibration in inverse angstroms.

    Notes
    -----
    Algorithm:
    - Calculate the field of view in real space
    - Calculate the inverse space calibration
    """
    field_of_view: Float[Array, " "] = jnp.multiply(
        jnp.float64(sizebeam), real_space_calib
    )
    inverse_space_calib: Float[Array, " 2"] = 1 / field_of_view
    return inverse_space_calib


@jaxtyped(typechecker=typechecker)
def make_probe(
    aperture: scalar_numeric,
    voltage: scalar_numeric,
    image_size: Int[Array, " 2"],
    calibration_pm: scalar_float,
    defocus: Optional[scalar_numeric] = 0.0,
    c3: Optional[scalar_numeric] = 0.0,
    c5: Optional[scalar_numeric] = 0.0,
) -> Complex[Array, " h w"]:
    """Calculate an electron probe with spherical aberrations.

    Parameters
    ----------
    aperture : scalar_numeric
        The aperture size in milliradians.
    voltage : scalar_numeric
        The microscope accelerating voltage in kiloelectronvolts.
    image_size : Int[Array, " 2"]
        The size of the beam in pixels.
    calibration_pm : scalar_float
        The calibration in picometers.
    defocus : scalar_numeric, optional
        The defocus value in angstroms. Default is 0.
    c3 : scalar_numeric, optional
        The C3 value in angstroms. Default is 0.
    c5 : scalar_numeric, optional
        The C5 value in angstroms. Default is 0.

    Returns
    -------
    Complex[Array, " h w"]
        The calculated electron probe in real space.

    Notes
    -----
    Algorithm:
    - Convert the aperture to radians
    - Calculate the wavelength in angstroms
    - Calculate the maximum L value
    - Calculate the field of view in x and y
    - Generate the inverse space array y and x
    - Shift the inverse space array y and x
    - Create meshgrid of shifted inverse space arrays
    - Calculate the inverse array
    - Calculate the calibration in y and x
    - Calculate the probe in real space
    """
    aperture: Float[Array, " "] = jnp.asarray(aperture / 1000.0)
    wavelength: Float[Array, " "] = wavelength_ang(voltage)
    l_max: Float[Array, " "] = aperture / wavelength
    image_y: scalar_int
    image_x: scalar_int
    image_y, image_x = image_size
    x_fov: Float[Array, " "] = image_x * 0.01 * calibration_pm
    y_fov: Float[Array, " "] = image_y * 0.01 * calibration_pm
    qx: Float[Array, " w"] = (jnp.arange((-image_x / 2), (image_x / 2), 1)) / x_fov
    x_shifter: scalar_int = image_x // 2
    qy: Float[Array, " h"] = (jnp.arange((-image_y / 2), (image_y / 2), 1)) / y_fov
    y_shifter: scalar_int = image_y // 2
    lx: Float[Array, " w"] = jnp.roll(qx, x_shifter)
    ly: Float[Array, " h"] = jnp.roll(qy, y_shifter)
    lya: Float[Array, " h w"]
    lxa: Float[Array, " h w"]
    lya, lxa = jnp.meshgrid(lx, ly)
    l2: Float[Array, " H W"] = jnp.multiply(lxa, lxa) + jnp.multiply(lya, lya)
    inverse_real_matrix: Float[Array, " h w"] = l2**0.5
    a_dist: Complex[Array, " h w"] = jnp.asarray(
        inverse_real_matrix <= l_max, dtype=jnp.complex128
    )
    chi_probe: Float[Array, " h w"] = aberration(
        inverse_real_matrix, wavelength, defocus, c3, c5
    )
    a_dist *= jnp.exp(-1j * chi_probe)
    probe_real_space: Complex[Array, " h w"] = jnp.fft.ifftshift(jnp.fft.ifft2(a_dist))
    return probe_real_space


@jaxtyped(typechecker=typechecker)
def aberration(
    fourier_coord: Float[Array, " H W"],
    lambda_angstrom: scalar_float,
    defocus: Optional[scalar_float] = 0.0,
    c3: Optional[scalar_float] = 0.0,
    c5: Optional[scalar_float] = 0.0,
) -> Float[Array, " H W"]:
    """Calculate the aberration function for the electron probe based on the Fourier coordinates.

    Parameters
    ----------
    fourier_coord : Float[Array, " H W"]
        The Fourier coordinates.
    lambda_angstrom : scalar_float
        The wavelength in angstroms.
    defocus : scalar_float, optional
        The defocus value in angstroms. Default is 0.0.
    c3 : scalar_float, optional
        The C3 value in angstroms. Default is 0.0.
    c5 : scalar_float, optional
        The C5 value in angstroms. Default is 0.0.

    Returns
    -------
    Float[Array, " H W"]
        The calculated aberration function.

    Notes
    -----
    Algorithm:
    - Calculate the phase shift
    - Calculate the chi value
    - Calculate the chi probe value
    """
    p_matrix: Float[Array, " H W"] = lambda_angstrom * fourier_coord
    chi: Float[Array, " H W"] = (
        ((defocus * jnp.power(p_matrix, 2)) / 2)
        + ((c3 * (1e7) * jnp.power(p_matrix, 4)) / 4)
        + ((c5 * (1e7) * jnp.power(p_matrix, 6)) / 6)
    )
    chi_probe: Float[Array, " H W"] = (2 * jnp.pi * chi) / lambda_angstrom
    return chi_probe


@jaxtyped(typechecker=typechecker)
def wavelength_ang(voltage_kv: scalar_numeric) -> Float[Array, " "]:
    """Calculate the relativistic electron wavelength in angstroms.

    Parameters
    ----------
    voltage_kv : scalar_numeric
        The microscope accelerating voltage in kiloelectronvolts. Can be a scalar or array.

    Returns
    -------
    Float[Array, " "]
        The electron wavelength in angstroms with same shape as input.

    Notes
    -----
    Algorithm:
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms

    Because this is JAX - you assume that the input is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check for them - not the function itself.
    """
    m: Float[Array, " "] = jnp.asarray(9.109383e-31)
    e: Float[Array, " "] = jnp.asarray(1.602177e-19)
    c: Float[Array, " "] = jnp.asarray(299792458.0)
    h: Float[Array, " "] = jnp.asarray(6.62607e-34)

    ev: Float[Array, " "] = (
        jnp.float64(voltage_kv) * jnp.float64(1000.0) * jnp.float64(e)
    )
    numerator: Float[Array, " "] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, " "] = jnp.multiply(ev, ((2 * m * jnp.square(c)) + ev))
    wavelength_meters: Float[Array, " "] = jnp.sqrt(numerator / denominator)
    lambda_angstroms: Float[Array, " "] = jnp.asarray(1e10) * wavelength_meters
    return lambda_angstroms


@jaxtyped(typechecker=typechecker)
def cbed(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    voltage_kv: scalar_numeric,
) -> CalibratedArray:
    """Calculate the CBED pattern for single/multiple slices and single/multiple beam modes.

    This function computes the Convergent Beam Electron Diffraction (CBED) pattern
    by propagating one or more beam modes through one or more potential slices.

    Parameters
    ----------
    pot_slices : PotentialSlices
        The potential slice(s) with the following attributes:
        - slices : Float[Array, " H W S"]
            Individual potential slices in Kirkland units. S is number of slices
        - slice_thickness : scalar_numeric
            Thickness of each slice in angstroms
        - calib : scalar_float
            Pixel calibration
    beam : ProbeModes
        The electron beam with the following attributes:
        - modes : Complex[Array, " H W *M"]
            M is number of modes
        - weights : Float[Array, " M"]
            Mode occupation numbers
        - calib : scalar_float
            Pixel calibration
    voltage_kv : scalar_numeric
        The accelerating voltage in kilovolts.

    Returns
    -------
    CalibratedArray
        The calculated CBED pattern with the following attributes:
        - data_array : Float[Array, " H W"]
            The calculated CBED pattern.
        - calib_y : scalar_float
            The calibration in y direction.
        - calib_x : scalar_float
            The calibration in x direction.
        - real_space : bool
            False, indicating reciprocal space data.

    Notes
    -----
    Algorithm:
    - Ensure 3D arrays even for single slice/mode
    - Calculate the transmission function for a single slice
    - Initialize the convolution state
    - Scan over all slices
    - Compute the Fourier transform
    - Compute the intensity for each mode
    - Sum the intensities across all modes.
    """
    calib_ang: Float[Array, ""] = jnp.amin(jnp.array([pot_slices.calib, beam.calib]))
    dtype: jnp.dtype = beam.modes.dtype
    pot_slice: Float[Array, " H W S"] = jnp.atleast_3d(pot_slices.slices)
    beam_modes: Complex[Array, " H W M"] = jnp.atleast_3d(beam.modes)
    num_slices: int = pot_slice.shape[-1]
    slice_transmission: Complex[Array, " H W"] = propagation_func(
        beam_modes.shape[0],
        beam_modes.shape[1],
        pot_slices.slice_thickness,
        voltage_kv,
        calib_ang,
    ).astype(dtype)
    init_wave: Complex[Array, " H W M"] = jnp.copy(beam_modes)

    def _scan_fn(
        carry: Complex[Array, " H W M"], slice_idx: scalar_int
    ) -> Tuple[Complex[Array, " H W M"], None]:
        wave: Complex[Array, " H W M"] = carry
        pot_single_slice: Float[Array, " H W 1"] = lax.dynamic_slice_in_dim(
            pot_slice, slice_idx, 1, axis=2
        )
        pot_single_slice: Float[Array, " H W"] = jnp.squeeze(pot_single_slice, axis=2)
        trans_slice: Complex[Array, " H W"] = transmission_func(
            pot_single_slice, voltage_kv
        )
        wave = wave * trans_slice[..., jnp.newaxis]

        def _propagate(w: Complex[Array, " H W M"]) -> Complex[Array, " H W M"]:
            w_k: Complex[Array, " H W M"] = jnp.fft.fft2(w, axes=(0, 1))
            w_k = w_k * slice_transmission[..., jnp.newaxis]
            return jnp.fft.ifft2(w_k, axes=(0, 1)).astype(dtype)

        is_last_slice: Bool[Array, ""] = slice_idx == num_slices - 1
        wave = lax.cond(is_last_slice, lambda w: w, _propagate, wave)
        return wave, None

    final_wave: Complex[Array, " H W M"]
    final_wave, _ = lax.scan(_scan_fn, init_wave, jnp.arange(num_slices))
    fourier_space_pattern: Complex[Array, " H W M"] = jnp.fft.fftshift(
        jnp.fft.fft2(final_wave, axes=(0, 1)), axes=(0, 1)
    )
    intensity_per_mode: Float[Array, " H W M"] = jnp.square(
        jnp.abs(fourier_space_pattern)
    )
    cbed_pattern: Float[Array, " H W"] = jnp.sum(intensity_per_mode, axis=-1)
    real_space_fov: Float[Array, " "] = jnp.multiply(beam_modes.shape[0], calib_ang)
    inverse_space_calib: Float[Array, " "] = 1 / real_space_fov
    cbed_pytree: CalibratedArray = make_calibrated_array(
        cbed_pattern, inverse_space_calib, inverse_space_calib, False
    )
    return cbed_pytree


@jaxtyped(typechecker=typechecker)
def shift_beam_fourier(
    beam: Union[Float[Array, " hh ww *mm"], Complex[Array, " hh ww *mm"]],
    pos: Float[Array, " #pp 2"],
    calib_ang: scalar_float,
) -> Complex128[Array, "#pp hh ww #mm"]:
    """Shift the beam to new position(s) using Fourier shifting.

    Parameters
    ----------
    beam : Float[Array, " hh ww *mm"] or Complex[Array, " hh ww *mm"]
        The electron beam modes.
    pos : Float[Array, " #P 2"]
        The (y, x) position(s) to shift to in pixels.
        Can be a single position [2] or multiple [P, 2].
    calib_ang : scalar_float
        The calibration in angstroms.

    Returns
    -------
    Complex128[Array, "#P H W #M"]
        The shifted beam(s) for all position(s) and mode(s).

    Notes
    -----
    Algorithm:
    - Convert positions from real space to Fourier space
    - Create phase ramps in Fourier space for all positions
    - Apply shifts to each mode for all positions
    """
    our_beam: Complex128[Array, "H W #M"] = jnp.atleast_3d(beam.astype(jnp.complex128))
    hh: int
    ww: int
    hh, ww = our_beam.shape[0], our_beam.shape[1]
    pos: Float[Array, "#pp 2"] = jnp.atleast_2d(pos)
    num_positions: int = pos.shape[0]
    qy: Float[Array, " hh"] = jnp.fft.fftfreq(hh, d=calib_ang)
    qx: Float[Array, " ww"] = jnp.fft.fftfreq(ww, d=calib_ang)
    qya: Float[Array, " hh ww"]
    qxa: Float[Array, " hh ww"]
    qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")
    beam_k: Complex128[Array, " hh ww #mm"] = jnp.fft.fft2(our_beam, axes=(0, 1))

    def _apply_shift(position_idx: int) -> Complex128[Array, " hh ww #mm"]:
        y_shift: scalar_numeric
        x_shift: scalar_numeric
        y_shift, x_shift = pos[position_idx, 0], pos[position_idx, 1]
        phase: Float[Array, " hh ww"] = (
            -2.0 * jnp.pi * ((qya * y_shift) + (qxa * x_shift))
        )
        phase_shift: Complex[Array, " hh ww"] = jnp.exp(1j * phase)
        phase_shift_expanded: Complex128[Array, " hh ww 1"] = phase_shift[
            ..., jnp.newaxis
        ]
        shifted_beam_k: Complex128[Array, " hh ww #mm"] = beam_k * phase_shift_expanded
        shifted_beam: Complex128[Array, " hh ww #mm"] = jnp.fft.ifft2(
            shifted_beam_k, axes=(0, 1)
        )
        return shifted_beam

    all_shifted_beams: Complex128[Array, " #pp hh ww #mm"] = jax.vmap(_apply_shift)(
        jnp.arange(num_positions)
    )
    return all_shifted_beams


@jaxtyped(typechecker=typechecker)
def stem_4d(
    pot_slice: PotentialSlices,
    beam: ProbeModes,
    positions: Num[Array, "#P 2"],
    voltage_kv: scalar_numeric,
    calib_ang: scalar_float,
) -> STEM4D:
    """Simulate CBED patterns for multiple beam positions by shifting the beam and running CBED simulations.

    Parameters
    ----------
    pot_slice : PotentialSlices
        The potential slice(s).
    beam : ProbeModes
        The electron beam mode(s).
    positions : Num[Array, "#P 2"]
        The (y, x) positions to shift the beam to.
        With P being the number of positions.
    voltage_kv : scalar_numeric
        The accelerating voltage in kilovolts.
    calib_ang : scalar_float
        The calibration in angstroms.

    Returns
    -------
    STEM4D
        Complete 4D-STEM dataset containing:
        - Diffraction patterns for each scan position
        - Real and Fourier space calibrations
        - Scan positions in Angstroms
        - Accelerating voltage

    Notes
    -----
    Algorithm:
    1. Shift beam to all specified positions
    2. For each position, run CBED simulation
    3. Return STEM4D PyTree with all data and calibrations
    """
    shifted_beams: Complex[Array, " P H W #M"] = shift_beam_fourier(
        beam.modes, positions, calib_ang
    )

    def _process_single_position(pos_idx: scalar_int) -> Float[Array, " H W"]:
        current_beam: Complex[Array, " H W #M"] = jnp.take(
            shifted_beams, pos_idx, axis=0
        )
        current_probe_modes: ProbeModes = ProbeModes(
            modes=current_beam,
            weights=beam.weights,
            calib=beam.calib,
        )
        cbed_result: CalibratedArray = cbed(
            pot_slices=pot_slice, beam=current_probe_modes, voltage_kv=voltage_kv
        )
        return cbed_result.data_array

    cbed_patterns: Float[Array, " P H W"] = jax.vmap(_process_single_position)(
        jnp.arange(positions.shape[0])
    )
    first_beam_modes: ProbeModes = ProbeModes(
        modes=shifted_beams[0],
        weights=beam.weights,
        calib=beam.calib,
    )
    first_cbed: CalibratedArray = cbed(
        pot_slices=pot_slice, beam=first_beam_modes, voltage_kV=voltage_kv
    )
    fourier_calib: Float[Array, " "] = first_cbed.calib_y
    scan_positions_ang: Float[Array, " P 2"] = positions * calib_ang
    stem4d_data: STEM4D = make_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kV=voltage_kv,
    )
    return stem4d_data


@jaxtyped(typechecker=typechecker)
def decompose_beam_to_modes(
    beam: CalibratedArray,
    num_modes: scalar_int,
    first_mode_weight: scalar_float = 0.6,
) -> ProbeModes:
    """Decompose a single electron beam into multiple orthogonal modes while preserving the total intensity.

    Parameters
    ----------
    beam : CalibratedArray
        The electron beam to decompose.
    num_modes : scalar_int
        The number of modes to decompose into.
    first_mode_weight : scalar_float, optional
        The weight of the first mode. Default is 0.6.
        The remaining weight is divided equally among the other modes.
        Must be below 1.0.

    Returns
    -------
    ProbeModes
        The decomposed probe modes with the following attributes:
        - modes : Complex[Array, " H W M"]
            The orthogonal modes.
        - weights : Float[Array, " M"]
            The mode occupation numbers.
        - calib : scalar_float
            The pixel calibration.

    Notes
    -----
    Algorithm:
    - Flatten the 2D beam into a vector
    - Create a random complex matrix
    - Use QR decomposition to create orthogonal modes
    - Scale the modes to preserve total intensity
    - Reshape back to original spatial dimensions
    """
    hh: int
    ww: int
    hh, ww = beam.data_array.shape
    tp: int = hh * ww
    beam_flat: Complex[Array, " tp"] = beam.data_array.reshape(-1)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    key1: PRNGKeyArray
    key2: PRNGKeyArray
    key1, key2 = jax.random.split(key)
    random_real: Float[Array, " tp mm"] = jax.random.normal(
        key1, (tp, num_modes), dtype=jnp.float64
    )
    random_imag: Float[Array, " tp mm"] = jax.random.normal(
        key2, (tp, num_modes), dtype=jnp.float64
    )
    random_matrix: Complex[Array, " tp mm"] = random_real + (1j * random_imag)
    qq: Complex[Array, " tp mm"]
    qq, _ = jnp.linalg.qr(random_matrix, mode="reduced")
    original_intensity: Float[Array, " tp"] = jnp.square(jnp.abs(beam_flat))
    weights: Float[Array, " mm"] = jnp.zeros(num_modes, dtype=jnp.float64)
    weights = weights.at[0].set(first_mode_weight)
    remaining_weight: scalar_float = (1.0 - first_mode_weight) / max(1, num_modes - 1)
    weights = weights.at[1:].set(remaining_weight)
    sqrt_weights: Float[Array, " mm"] = jnp.sqrt(weights)
    sqrt_intensity: Float[Array, " tp 1"] = jnp.sqrt(original_intensity).reshape(-1, 1)
    weighted_modes: Complex[Array, " tp mm"] = qq * sqrt_intensity * sqrt_weights
    multimodal_beam: Complex[Array, " hh ww mm"] = weighted_modes.reshape(
        hh, ww, num_modes
    )
    probe_modes: ProbeModes = make_probe_modes(
        modes=multimodal_beam, weights=weights, calib=beam.calib_y
    )
    return probe_modes


@jaxtyped(typechecker=typechecker)
def stem_4d_sharded(
    pot_slice: PotentialSlices,
    beam: ProbeModes,
    positions: Num[Array, "#P 2"],
    voltage_kv: scalar_numeric,
    calib_ang: scalar_float,
) -> STEM4D:
    """Sharded version of stem_4d that distributes scan positions across available devices.

    This function uses JAX's sharding API to distribute the computation
    across multiple GPUs/TPUs. It is fully compatible with JIT compilation
    and automatic differentiation.

    Parameters
    ----------
    pot_slice : PotentialSlices
        The potential slice(s).
    beam : ProbeModes
        The electron beam mode(s).
    positions : Num[Array, "#P 2"]
        The (y, x) positions to shift the beam to.
        With P being the number of positions.
    voltage_kv : scalar_numeric
        The accelerating voltage in kilovolts.
    calib_ang : scalar_float
        The calibration in angstroms.

    Returns
    -------
    STEM4D
        Complete 4D-STEM dataset containing:
        - Diffraction patterns for each scan position
        - Real and Fourier space calibrations
        - Scan positions in Angstroms
        - Accelerating voltage

    Notes
    -----
    - Uses JAX sharding for automatic distribution across devices
    - Fully compatible with JIT compilation and automatic differentiation
    - The positions array is sharded along the first axis
    """
    shifted_beams: Complex[Array, " P H W #M"] = shift_beam_fourier(
        beam.modes, positions, calib_ang
    )
    devices: Any = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh: Mesh = Mesh(devices, axis_names=("positions",))
    shifted_beams_sharding: NamedSharding = NamedSharding(
        mesh, P("positions", None, None, None)
    )
    shifted_beams_sharded: Complex[Array, " P H W #M"] = jax.device_put(
        shifted_beams, shifted_beams_sharding
    )

    def _process_single_position(
        shifted_beam: Complex[Array, " H W #M"],
    ) -> Float[Array, " H W"]:
        current_probe_modes: ProbeModes = ProbeModes(
            modes=shifted_beam,
            weights=beam.weights,
            calib=beam.calib,
        )
        cbed_result: CalibratedArray = cbed(
            pot_slices=pot_slice, beam=current_probe_modes, voltage_kV=voltage_kv
        )
        return cbed_result.data_array

    cbed_patterns: Float[Array, " P H W"] = jax.vmap(_process_single_position)(
        shifted_beams_sharded
    )
    first_beam_modes: ProbeModes = ProbeModes(
        modes=shifted_beams[0],
        weights=beam.weights,
        calib=beam.calib,
    )
    first_cbed: CalibratedArray = cbed(
        pot_slices=pot_slice, beam=first_beam_modes, voltage_kV=voltage_kv
    )
    fourier_calib: Float[Array, " "] = first_cbed.calib_y
    scan_positions_ang: Float[Array, " P 2"] = positions * calib_ang
    stem4d_data: STEM4D = make_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kv=voltage_kv,
    )

    return stem4d_data


@jaxtyped(typechecker=typechecker)
def stem_4d_parallel(
    pot_slice: PotentialSlices,
    beam: ProbeModes,
    positions: Num[Array, "#P 2"],
    voltage_kv: scalar_numeric,
    calib_ang: scalar_float,
    n_devices: Optional[int] = None,
) -> STEM4D:
    """Parallel version of stem_4d using explicit device parallelism.

    This function provides more control over device usage and is suitable
    for cases where automatic sharding may not be optimal. It uses shard_map
    for explicit control over data distribution.

    Parameters
    ----------
    pot_slice : PotentialSlices
        The potential slice(s).
    beam : ProbeModes
        The electron beam mode(s).
    positions : Num[Array, "#P 2"]
        The (y, x) positions to shift the beam to.
    voltage_kv : scalar_numeric
        The accelerating voltage in kilovolts.
    calib_ang : scalar_float
        The calibration in angstroms.
    n_devices : int, optional
        Number of devices to use. If None, uses all available devices.

    Returns
    -------
    STEM4D
        Complete 4D-STEM dataset.

    Notes
    -----
    - Provides explicit control over device parallelism
    - Compatible with JIT compilation and automatic differentiation
    - Uses shard_map for fine-grained control
    """
    if n_devices is None:
        n_devices = jax.device_count()

    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(devices, axis_names=("devices",))

    shifted_beams: Complex[Array, " P H W #M"] = shift_beam_fourier(
        beam.modes, positions, calib_ang
    )

    def _compute_cbed_batch(
        shifted_beams_batch: Complex[Array, " batch H W #M"],
    ) -> Float[Array, "batch H W"]:
        def _process_single(
            shifted_beam: Complex[Array, " H W #M"],
        ) -> Float[Array, " H W"]:
            current_probe_modes: ProbeModes = ProbeModes(
                modes=shifted_beam,
                weights=beam.weights,
                calib=beam.calib,
            )
            cbed_result: CalibratedArray = cbed(
                pot_slices=pot_slice, beam=current_probe_modes, voltage_kV=voltage_kv
            )
            return cbed_result.data_array

        return jax.vmap(_process_single)(shifted_beams_batch)

    with mesh:
        cbed_patterns = shard_map(
            _compute_cbed_batch,
            mesh=mesh,
            in_specs=P("devices", None, None, None),
            out_specs=P("devices", None, None),
            check_rep=False,
        )(shifted_beams)

    first_beam_modes: ProbeModes = ProbeModes(
        modes=shifted_beams[0],
        weights=beam.weights,
        calib=beam.calib,
    )
    first_cbed: CalibratedArray = cbed(
        pot_slices=pot_slice, beam=first_beam_modes, voltage_kv=voltage_kv
    )
    fourier_calib: Float[Array, " "] = first_cbed.calib_y

    scan_positions_ang: Float[Array, " P 2"] = positions * calib_ang

    stem4d_data: STEM4D = make_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kv=voltage_kv,
    )

    return stem4d_data


@jaxtyped(typechecker=typechecker)
def annular_detector(
    stem4d_data: STEM4D,
    collection_angles: Float[Array, " 2"],
) -> CalibratedArray:
    """Simulate an annular detector that integrates the CBED signal between inner and outer collection angles to generate a STEM image.

    Parameters
    ----------
    stem4d_data : STEM4D
        The 4D-STEM data containing diffraction patterns with the following attributes:
        - data : Float[Array, " P H W"]
            4D-STEM data array where P is number of scan positions
        - fourier_space_calib : Float[Array, " "]
            Fourier space calibration in inverse Angstroms per pixel
        - voltage_kv : Float[Array, " "]
            Accelerating voltage in kilovolts
        - real_space_calib : Float[Array, " "]
            Real space calibration in Angstroms per pixel
        - scan_positions : Float[Array, " P 2"]
            Real space scan positions in Angstroms
    collection_angles : Float[Array, " 2"]
        Inner and outer collection angles in milliradians [inner_mrad, outer_mrad].

    Returns
    -------
    CalibratedArray
        STEM image generated by annular detector integration with the following attributes:
        - data_array : Float[Array, " Ny Nx"]
            The integrated STEM image
        - calib_y : Float[Array, " "]
            Real space calibration in y direction
        - calib_x : Float[Array, " "]
            Real space calibration in x direction
        - real_space : bool
            True, indicating real space image

    Notes
    -----
    Algorithm:
    - Calculate wavelength from accelerating voltage
    - Convert collection angles from mrad to inverse Angstroms
    - Create Fourier space coordinate grid for diffraction patterns
    - Create annular mask based on collection angles
    - Apply mask and integrate each diffraction pattern
    - Reshape to 2D STEM image based on scan positions
    - Return as calibrated array with real space calibrations
    """
    wavelength: Float[Array, " "] = wavelength_ang(stem4d_data.voltage_kv)
    inner_angle_rad: Float[Array, " "] = collection_angles[0] / 1000.0
    outer_angle_rad: Float[Array, " "] = collection_angles[1] / 1000.0
    inner_k: Float[Array, " "] = inner_angle_rad / wavelength
    outer_k: Float[Array, " "] = outer_angle_rad / wavelength

    hh: int
    ww: int
    _, hh, ww = stem4d_data.data.shape

    qy: Float[Array, " hh"] = jnp.arange(hh) - hh // 2
    qx: Float[Array, " ww"] = jnp.arange(ww) - ww // 2
    qya: Float[Array, " hh ww"]
    qxa: Float[Array, " hh ww"]
    qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")
    q_radius: Float[Array, " hh ww"] = (
        jnp.sqrt(qya**2 + qxa**2) * stem4d_data.fourier_space_calib
    )

    annular_mask: Bool[Array, " hh ww"] = (q_radius >= inner_k) & (q_radius <= outer_k)

    def _integrate_pattern(pattern: Float[Array, " hh ww"]) -> Float[Array, " "]:
        return jnp.sum(pattern * annular_mask)

    integrated_intensities: Float[Array, " pp"] = jax.vmap(_integrate_pattern)(
        stem4d_data.data
    )

    y_positions: Float[Array, " pp"] = stem4d_data.scan_positions[:, 0]
    x_positions: Float[Array, " pp"] = stem4d_data.scan_positions[:, 1]

    y_unique: Float[Array, " ny"] = jnp.unique(y_positions)
    x_unique: Float[Array, " nx"] = jnp.unique(x_positions)
    ny: int = y_unique.shape[0]
    nx: int = x_unique.shape[0]

    stem_image_2d: Float[Array, " ny nx"] = integrated_intensities.reshape(ny, nx)

    stem_image: CalibratedArray = make_calibrated_array(
        data_array=stem_image_2d,
        calib_y=stem4d_data.real_space_calib,
        calib_x=stem4d_data.real_space_calib,
        real_space=True,
    )

    return stem_image
