import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, TypeAlias, Union
from jax import lax
from jaxtyping import (Array, Bool, Complex, Complex128, Float, Int, Num,
                       PRNGKeyArray, PyTree, jaxtyped)

from .types import CalibratedArray, PotentialSlices, ProbeModes

jax.config.update("jax_enable_x64", True)


scalar_numeric: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Int[Array, ""]]


@jaxtyped(typechecker=typechecker)
def transmission_func(
    pot_slice: Float[Array, "a b"], voltage_kV: scalar_numeric
) -> Complex[Array, ""]:
    """
    Description
    -----------
    Calculates the complex transmission function from
    a single potential slice at a given electron accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Parameters
    ----------
    - `pot_slice` (Float[Array, "a b"]):
        potential slice in Kirkland units
    - `voltage_kV` (scalar_numeric):
        microscope operating voltage in kilo
        electronVolts

    Returns
    -------
    - `trans` (Complex[Array, "a b"]):
        The transmission function of a single
        crystal slice

    Flow
    ----
    - Calculate the electron energy in electronVolts
    - Calculate the wavelength in angstroms
    - Calculate the Einstein energy
    - Calculate the sigma value, which is the constant for the phase shift
    - Calculate the transmission function as a complex exponential
    """

    voltage: Float[Array, ""] = jnp.multiply(voltage_kV, jnp.asarray(1000.0))

    m_e: Float[Array, ""] = jnp.asarray(9.109383e-31)
    e_e: Float[Array, ""] = jnp.asarray(1.602177e-19)
    c: Float[Array, ""] = jnp.asarray(299792458.0)

    eV: Float[Array, ""] = jnp.multiply(e_e, voltage)
    lambda_angstrom: Float[Array, ""] = wavelength_ang(voltage_kV)
    einstein_energy: Float[Array, ""] = jnp.multiply(m_e, jnp.square(c))
    sigma: Float[Array, ""] = (
        (2 * jnp.pi / (lambda_angstrom * voltage)) * (einstein_energy + eV)
    ) / ((2 * einstein_energy) + eV)
    trans: Complex[Array, "a b"] = jnp.exp(1j * sigma * pot_slice)
    return trans


@jaxtyped(typechecker=typechecker)
def propagation_func(
    imsize_y: scalar_int,
    imsize_x: scalar_int,
    thickness_ang: scalar_numeric,
    voltage_kV: scalar_numeric,
    calib_ang: scalar_float,
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    Calculates the complex propagation function that results
    in the phase shift of the exit wave when it travels from
    one slice to the next in the multislice algorithm

    Parameters
    ----------
    - `imsize_y`, (scalar_int):
        Size of the image of the propagator in y axis
    - `imsize_x`, (scalar_int):
        Size of the image of the propagator in x axis
    -  `thickness_ang`, (scalar_numeric):
        Distance between the slices in angstroms
    - `voltage_kV`, (scalar_numeric):
        Accelerating voltage in kilovolts
    - `calib_ang`, (scalar_float):
        Calibration or pixel size in angstroms

    Returns
    -------
    - `prop` (Complex[Array, "H W"]):
        The propagation function of the same size given by imsize

    Flow
    ----
    - Generate frequency arrays directly using fftfreq
    - Create 2D meshgrid of frequencies
    - Calculate squared sum of frequencies
    - Calculate wavelength
    - Compute the propagation function
    """
    qy: Num[Array, "H"] = jnp.fft.fftfreq(imsize_y, d=calib_ang)
    qx: Num[Array, "W"] = jnp.fft.fftfreq(imsize_x, d=calib_ang)
    Lya: Num[Array, "H W"]
    Lxa: Num[Array, "H W"]
    Lya, Lxa = jnp.meshgrid(qy, qx, indexing="ij")
    L_sq: Num[Array, "H W"] = jnp.square(Lxa) + jnp.square(Lya)
    lambda_angstrom: Float[Array, ""] = wavelength_ang(voltage_kV)
    prop: Complex[Array, "H W"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * L_sq
    )
    return prop


@jaxtyped(typechecker=typechecker)
def fourier_coords(
    calibration: scalar_float | Float[Array, "2"], image_size: Int[Array, "2"]
) -> CalibratedArray:
    """
    Description
    -----------
    Return the Fourier coordinates

    Parameters
    ----------
    - `calibration` (scalar_float | Float[Array, "2"]):
        The pixel size in angstroms in real space
    - `image_size`, (Int[Array, "2"]):
        The size of the beam in pixels

    Returns
    -------
    - PyTree with the following fields:
        - `data_array` (Num[Array, "H W"]):
            The inverse array data
        - `calib_y` (Float[Array, ""]):
            Invsere calibration in y direction
        - `calib_x` (Float[Array, ""]):
            Invserse calibration in x direction
        - `real_space` (Bool[Array, ""]):
            False. The array is in Fourier space

    Flow
    ----
    - Calculate the real space field of view in y and x
    - Generate the inverse space array y and x
    - Shift the inverse space array y and x
    - Create meshgrid of shifted inverse space arrays
    - Calculate the inverse array
    - Calculate the calibration in y and x
    - Return the calibrated array
    """
    real_fov: Float[Array, "2"] = jnp.multiply(image_size, calibration)
    inverse_arr_y: Float[Array, "H"] = (
        jnp.arange((-image_size[0] / 2), (image_size[0] / 2), 1)
    ) / real_fov[0]
    inverse_arr_x: Float[Array, "W"] = (
        jnp.arange((-image_size[1] / 2), (image_size[1] / 2), 1)
    ) / real_fov[1]
    shifter_y: Float[Array, ""] = image_size[0] // 2
    shifter_x: Float[Array, ""] = image_size[1] // 2
    inverse_shifted_y: Float[Array, "H"] = jnp.roll(inverse_arr_y, shifter_y)
    inverse_shifted_x: Float[Array, "W"] = jnp.roll(inverse_arr_x, shifter_x)
    inverse_xx: Float[Array, "H W"]
    inverse_yy: Float[Array, "H W"]
    inverse_xx, inverse_yy = jnp.meshgrid(inverse_shifted_x, inverse_shifted_y)
    inv_squared: Float[Array, "H W"] = jnp.multiply(
        inverse_yy, inverse_yy
    ) + jnp.multiply(inverse_xx, inverse_xx)
    inverse_array: Float[Array, "H W"] = inv_squared**0.5
    calib_inverse_y: Float[Array, ""] = inverse_arr_y[1] - inverse_arr_y[0]
    calib_inverse_x: Float[Array, ""] = inverse_arr_x[1] - inverse_arr_x[0]
    inverse_space: Bool[Array, ""] = False
    calibrated_inverse_array = CalibratedArray(
        inverse_array, calib_inverse_y, calib_inverse_x, inverse_space
    )
    return calibrated_inverse_array


@jaxtyped(typechecker=typechecker)
def fourier_calib(
    real_space_calib: Float[Array, ""] | Float[Array, "2"],
    sizebeam: Int[Array, "2"],
) -> Float[Array, "2"]:
    """
    Description
    -----------
    Generate the Fourier calibration for the beam

    Parameters
    ----------
    - `real_space_calib` (Float[Array, ""] | Float[Array, "2"]):
        The pixel size in angstroms in real space
    - `sizebeam` (Int[Array, "2"]):
        The size of the beam in pixels

    Returns
    -------
    - `inverse_space_calib` (Float[Array, "2"]):
        The Fourier calibration in angstroms

    Flow
    ----
    - Calculate the field of view in real space
    - Calculate the inverse space calibration
    """
    field_of_view: Float[Array, ""] = jnp.multiply(
        jnp.float64(sizebeam), real_space_calib
    )
    inverse_space_calib = 1 / field_of_view
    return inverse_space_calib


@jaxtyped(typechecker=typechecker)
def make_probe(
    aperture: scalar_numeric,
    voltage: scalar_numeric,
    image_size: Int[Array, "2"],
    calibration_pm: scalar_float,
    defocus: Optional[scalar_numeric] = 0,
    c3: Optional[scalar_numeric] = 0,
    c5: Optional[scalar_numeric] = 0,
) -> Complex[Array, "H W"]:
    """
    Description
    -----------
    This calculates an electron probe based on the
    size and the estimated Fourier co-ordinates with
    the option of adding spherical aberration in the
    form of defocus, C3 and C5

    Parameters
    ----------
    - `aperture` (scalar_numeric):
        The aperture size in milliradians
    - `voltage` (scalar_numeric):
        The microscope accelerating voltage in kilo
        electronVolts
    - `image_size`, (Int[Array, "2"]):
        The size of the beam in pixels
    - `calibration_pm` (scalar_float):
        The calibration in picometers
    - `defocus` (Optional[scalar_numeric]):
        The defocus value in angstroms.
        Optional, default is 0.
    - `c3` (Optiona[scalar_numeric]):
        The C3 value in angstroms.
        Optional, default is 0.
    - `c5` (Optional[scalar_numeric]):
        The C5 value in angstroms.
        Optional, default is 0.

    Returns
    -------
    - `probe_real_space` (Complex[Array, "H W"]):
        The calculated electron probe in real space

    Flow
    ----
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
    aperture: Float[Array, ""] = jnp.asarray(aperture / 1000.0)
    wavelength: Float[Array, ""] = wavelength_ang(voltage)
    LMax = aperture / wavelength
    image_y, image_x = image_size
    x_FOV = image_x * 0.01 * calibration_pm
    y_FOV = image_y * 0.01 * calibration_pm
    qx = (jnp.arange((-image_x / 2), (image_x / 2), 1)) / x_FOV
    x_shifter = image_x // 2
    qy = (jnp.arange((-image_y / 2), (image_y / 2), 1)) / y_FOV
    y_shifter = image_y // 2
    Lx = jnp.roll(qx, x_shifter)
    Ly = jnp.roll(qy, y_shifter)
    Lya, Lxa = jnp.meshgrid(Lx, Ly)
    L2 = jnp.multiply(Lxa, Lxa) + jnp.multiply(Lya, Lya)
    inverse_real_matrix = L2**0.5
    Adist = jnp.asarray(inverse_real_matrix <= LMax, dtype=jnp.complex128)
    chi_probe = aberration(inverse_real_matrix, wavelength, defocus, c3, c5)
    Adist *= jnp.exp(-1j * chi_probe)
    probe_real_space = jnp.fft.ifftshift(jnp.fft.ifft2(Adist))
    return probe_real_space


@jaxtyped(typechecker=typechecker)
def aberration(
    fourier_coord: Float[Array, "H W"],
    lambda_angstrom: scalar_float,
    defocus: Optional[scalar_float] = 0.0,
    c3: Optional[scalar_float] = 0.0,
    c5: Optional[scalar_float] = 0.0,
) -> Float[Array, "H W"]:
    """
    Description
    -----------
    This calculates the aberration function for the
    electron probe based on the Fourier co-ordinates

    Parameters
    ----------
    - `fourier_coord` (Float[Array, "H W"]):
        The Fourier co-ordinates
    - `lambda_angstrom` (scalar_float):
        The wavelength in angstroms
    - `defocus` (Optional[scalar_float]):
        The defocus value in angstroms.
        Optional, default is 0.0
    - `c3` (Optional[scalar_float]):
        The C3 value in angstroms.
        Optional, default is 0.0
    - `c5` (Optional[scalar_float]):
        The C5 value in angstroms.
        Optional, default is 0.0

    Returns
    -------
    - `chi_probe` (Float[Array, "H W"]):
        The calculated aberration function

    Flow
    ----
    - Calculate the phase shift
    - Calculate the chi value
    - Calculate the chi probe value
    """
    p_matrix: Float[Array, "H W"] = lambda_angstrom * fourier_coord
    chi: Float[Array, "H W"] = (
        ((defocus * jnp.power(p_matrix, 2)) / 2)
        + ((c3 * (1e7) * jnp.power(p_matrix, 4)) / 4)
        + ((c5 * (1e7) * jnp.power(p_matrix, 6)) / 6)
    )
    chi_probe: Float[Array, "H W"] = (2 * jnp.pi * chi) / lambda_angstrom
    return chi_probe


@jaxtyped(typechecker=typechecker)
def wavelength_ang(voltage_kV: scalar_numeric) -> Float[Array, ""]:
    """
    Description
    -----------
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage.

    Because this is JAX - you assume that the input
    is clean, and you don't need to check for negative
    or NaN values. Your preprocessing steps should check
    for them - not the function itself.

    Parameters
    ----------
    - `voltage_kV` (scalar_numeric):
        The microscope accelerating voltage in kilo
        electronVolts. Can be a scalar or array.

    Returns
    -------
    - `in_angstroms` (Float[Array, ""]):
        The electron wavelength in angstroms with same shape as input

    Flow
    ----
    - Calculate the electron wavelength in meters
    - Convert the wavelength to angstroms
    """
    m: Float[Array, ""] = jnp.asarray(9.109383e-31)
    e: Float[Array, ""] = jnp.asarray(1.602177e-19)
    c: Float[Array, ""] = jnp.asarray(299792458.0)
    h: Float[Array, ""] = jnp.asarray(6.62607e-34)

    eV: Float[Array, ""] = (
        jnp.float64(voltage_kV) * jnp.float64(1000.0) * jnp.float64(e)
    )
    numerator: Float[Array, ""] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, ""] = jnp.multiply(eV, ((2 * m * jnp.square(c)) + eV))
    wavelength_meters: Float[Array, ""] = jnp.sqrt(numerator / denominator)
    lambda_angstroms: Float[Array, ""] = jnp.asarray(1e10) * wavelength_meters
    return lambda_angstroms


@jaxtyped(typechecker=typechecker)
def cbed(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    voltage_kV: scalar_numeric,
) -> Float[Array, "H W"]:
    """
    Description
    -----------
    Calculates the CBED pattern for single/multiple slices
    and single/multiple beam modes. This function computes
    the Convergent Beam Electron Diffraction (CBED) pattern
    by propagating one or more beam modes through one or
    more potential slices.

    Parameters
    ----------
    - `pot_slices` (PotentialSlices):,
        The potential slice(s). It has the following attributes:
        - `slices` (Complex[Array, "H W S"]):
            Individual potential slices.
            S is number of slices
        - `slice_thickness` (scalar_numeric):
            Mode occupation numbers
        - `calib` (scalar_float):
            Pixel Calibration
    - `beam` (ProbeModes):
        - `modes` (Complex[Array, "H W *M"]):
            M is number of modes
        - `weights` (Float[Array, "M"]):
            Mode occupation numbers
        - `calib` (scalar_float):
            Pixel Calibration
    - `voltage_kV` (scalar_numeric):
        The accelerating voltage in kilovolts.

    Returns
    -------
    -  `cbed_pattern` (Float[Array, "H W"]):
        The calculated CBED pattern.

    Flow
    ----
    - Ensure 3D arrays even for single slice/mode
    - Calculate the transmission function for a single slice
    - Initialize the convolution state
    - Scan over all slices
    - Compute the Fourier transform
    - Compute the intensity for each mode
    - Sum the intensities across all modes.
    """
    calib_ang = jnp.amin([pot_slices.calib, beam.calib])
    dtype = beam.dtype
    pot_slice = jnp.atleast_3d(pot_slice)
    beam = jnp.atleast_3d(beam)
    num_slices = pot_slice.shape[-1]
    slice_transmission = propagation_func(
        beam.shape[0], beam.shape[1], pot_slices.slice_thickness, voltage_kV, calib_ang
    ).astype(dtype)
    init_wave = jnp.copy(beam)

    def scan_fn(carry, slice_idx):
        wave = carry
        trans_slice = lax.dynamic_slice_in_dim(pot_slice, slice_idx, 1, axis=2)
        trans_slice = jnp.squeeze(trans_slice, axis=2)
        wave = wave * trans_slice[..., jnp.newaxis]

        def propagate(w):
            w_k = jnp.fft.fft2(w, axes=(0, 1))
            w_k = w_k * slice_transmission[..., jnp.newaxis]
            return jnp.fft.ifft2(w_k, axes=(0, 1)).astype(dtype)

        is_last_slice = slice_idx == num_slices - 1
        wave = lax.cond(is_last_slice, lambda w: w, propagate, wave)
        return wave, None

    final_wave, _ = lax.scan(scan_fn, init_wave, jnp.arange(num_slices))
    fourier_space_pattern = jnp.fft.fftshift(
        jnp.fft.fft2(final_wave, axes=(0, 1)), axes=(0, 1)
    )
    intensity_per_mode = jnp.square(jnp.abs(fourier_space_pattern))
    cbed_pattern = jnp.sum(intensity_per_mode, axis=-1)
    real_space_fov = jnp.multiply(beam.shape[0], calib_ang)
    inverse_space_calib = 1 / real_space_fov
    cbed_pytree: PyTree = CalibratedArray(
        cbed_pattern, inverse_space_calib, inverse_space_calib, False
    )
    return cbed_pytree


@jaxtyped(typechecker=typechecker)
def shift_beam_fourier(
    beam: Union[Float[Array, "H W *M"], Complex[Array, "H W *M"]],
    pos: Float[Array, "#P 2"],
    calib_ang: scalar_float,
) -> Complex128[Array, "#P H W #M"]:
    """
    Description
    -----------
    Shifts the beam to new position(s) using Fourier shifting.

    Parameters
    ----------
    - beam (Union[Float[Array, "H W *M"], Complex[Array, "H W *M"]]):
        The electron beam modes.
    - pos (Float[Array, "#P 2"]):
        The (y, x) position(s) to shift to in pixels.
        Can be a single position [2] or multiple [P, 2].
    - calib_ang (scalar_float):
        The calibration in angstroms.

    Returns
    -------
    - shifted_beams (Complex128[Array, "#P H W #M"]):
        The shifted beam(s) for all position(s) and mode(s).

    Flow
    ----
    - Convert positions from real space to Fourier space
    - Create phase ramps in Fourier space for all positions
    - Apply shifts to each mode for all positions
    """
    our_beam: Complex128[Array, "H W #M"] = jnp.atleast_3d(beam.astype(jnp.complex128))
    H: int
    W: int
    H, W = our_beam.shape[0], our_beam.shape[1]
    pos = jnp.atleast_2d(pos)
    num_positions: int = pos.shape[0]
    qy: Float[Array, "H"] = jnp.fft.fftfreq(H, d=calib_ang)
    qx: Float[Array, "W"] = jnp.fft.fftfreq(W, d=calib_ang)
    qya: Float[Array, "H W"]
    qxa: Float[Array, "H W"]
    qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")
    beam_k: Complex128[Array, "H W #M"] = jnp.fft.fft2(our_beam, axes=(0, 1))

    def apply_shift(position_idx: int) -> Complex128[Array, "H W #M"]:
        y_shift: scalar_numeric
        x_shift: scalar_numeric
        y_shift, x_shift = pos[position_idx, 0], pos[position_idx, 1]
        phase: Float[Array, "H W"] = -2.0 * jnp.pi * ((qya * y_shift) + (qxa * x_shift))
        phase_shift: Complex[Array, "H W"] = jnp.exp(1j * phase)
        phase_shift_expanded: Complex128[Array, "H W 1"] = phase_shift[..., jnp.newaxis]
        shifted_beam_k: Complex128[Array, "H W #M"] = beam_k * phase_shift_expanded
        shifted_beam: Complex128[Array, "H W #M"] = jnp.fft.ifft2(
            shifted_beam_k, axes=(0, 1)
        )
        return shifted_beam

    all_shifted_beams: Complex128[Array, "#P H W #M"] = jax.vmap(apply_shift)(
        jnp.arange(num_positions)
    )
    return all_shifted_beams


@jaxtyped(typechecker=typechecker)
def stem_4D(
    pot_slice: Complex[Array, "H W #S"],
    beam: Complex[Array, "H W #M"],
    positions: Num[Array, "#P 2"],
    slice_thickness: scalar_float,
    voltage_kV: scalar_numeric,
    calib_ang: scalar_float,
) -> Float[Array, "#P H W"]:
    """
    Description
    -----------
    Simulates CBED patterns for multiple beam positions by:
    1. Shifting the beam to each specified position
    2. Running CBED simulation for each shifted beam

    Parameters
    ----------
    - `pot_slice` (Complex[Array, "H W #S"]):
        The potential slice(s). H and W are height and width,
        S is the number of slices (optional).
    - `beam` (Complex[Array, "H W #M"]):
        The electron beam mode(s).
        M is the number of modes (optional).
    - `positions` (Float[Array, "P 2"]):
        The (y, x) positions to shift the beam to.
        With P being the number of positions.
    - `slice_thickness` (scalar_float):
        The thickness of each slice in angstroms.
    - `voltage_kV` (scalar_numeric):
        The accelerating voltage in kilovolts.
    - `calib_ang` (scalar_float):
        The calibration in angstroms.

    Returns
    -------
    -  `cbed_patterns` (Float[Array, "P H W"]):
        The calculated CBED patterns for each position.

    Flow
    ----
    - Shift beam to all specified positions
    - For each position, run CBED simulation
    - Return array of all CBED patterns
    """
    shifted_beams: Complex[Array, "P H W #M"] = shift_beam_fourier(
        beam, positions, calib_ang
    )

    def process_single_position(pos_idx: int) -> Float[Array, "H W"]:
        current_beam: Complex[Array, "H W #M"] = jnp.take(
            shifted_beams, pos_idx, axis=0
        )
        cbed_pattern: Float[Array, "H W"] = cbed(
            pot_slice=pot_slice,
            beam=current_beam,
            slice_thickness=slice_thickness,
            voltage_kV=voltage_kV,
            calib_ang=calib_ang,
        )
        return cbed_pattern

    cbed_patterns: Float[Array, "P H W"] = jax.vmap(process_single_position)(
        jnp.arange(positions.shape[0])
    )
    return cbed_patterns


@jaxtyped(typechecker=typechecker)
def decompose_beam_to_modes(
    beam: CalibratedArray,
    num_modes: scalar_int,
    first_mode_weight: Optional[scalar_float] = 0.6,
) -> ProbeModes:
    """
    Description
    -----------
    Decomposes a single electron beam into multiple orthogonal modes
    while preserving the total intensity.

    Parameters
    ----------
    - `beam` (CalibratedArray):
    - `num_modes` (scalar_int):
        The number of modes to decompose into.
    - `first_mode_weight` (Optional[scalar_float]):
        The weight of the first mode. Default is 0.6.
        The remaining weight is divided equally among the other modes.
        Must be below 1.0.

    Returns
    -------
    - A PyTree with the following fields:
        - `modes` (Complex[Array, "H W M"]):
            The orthogonal modes.
        - `weights` (Float[Array, "M"]):
            The mode occupation numbers.
        - `calib` (scalar_float):
            The pixel calibration.

    Flow
    ----
    - Flatten the 2D beam into a vector
    - Create a random complex matrix
    - Use QR decomposition to create orthogonal modes
    - Scale the modes to preserve total intensity
    - Reshape back to original spatial dimensions
    """
    H: int
    W: int
    H, W = beam.shape
    TP: int = H * W
    beam_flat: Complex[Array, "TP"] = beam.reshape(-1)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    key1: PRNGKeyArray
    key2: PRNGKeyArray
    key1, key2 = jax.random.split(key)
    random_real: Float[Array, "TP M"] = jax.random.normal(
        key1, (TP, num_modes), dtype=jnp.float64
    )
    random_imag: Float[Array, "TP M"] = jax.random.normal(
        key2, (TP, num_modes), dtype=jnp.float64
    )
    random_matrix: Complex[Array, "TP M"] = random_real + (1j * random_imag)
    Q: Complex[Array, "TP M"]
    Q, _ = jnp.linalg.qr(random_matrix, mode="reduced")
    original_intensity: Float[Array, "TP"] = jnp.square(jnp.abs(beam_flat))
    weights: Float[Array, "M"] = jnp.zeros(num_modes, dtype=jnp.float64)
    weights = weights.at[0].set(first_mode_weight)
    remaining_weight: scalar_float = (1.0 - first_mode_weight) / max(1, num_modes - 1)
    weights = weights.at[1:].set(remaining_weight)
    sqrt_weights: Float[Array, "M"] = jnp.sqrt(weights)
    sqrt_intensity: Float[Array, "TP 1"] = jnp.sqrt(original_intensity).reshape(-1, 1)
    weighted_modes: Complex[Array, "TP M"] = Q * sqrt_intensity * sqrt_weights
    multimodal_beam: Complex[Array, "H W M"] = weighted_modes.reshape(H, W, num_modes)
    probe_modes: PyTree = ProbeModes(modes=multimodal_beam, weights=weights)
    return probe_modes
