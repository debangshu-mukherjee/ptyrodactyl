"""Forward simulation functions for electron microscopy.

Extended Summary
----------------
Functions for simulating electron beam propagation, creating
probes, calculating aberrations, and generating CBED patterns
and 4D-STEM data. All functions are JAX-compatible and support
automatic differentiation.

Routine Listings
----------------
:func:`transmission_func`
    Calculate transmission function for a potential slice.
:func:`propagation_func`
    Compute Fresnel propagation function.
:func:`fourier_coords`
    Generate Fourier space coordinate arrays.
:func:`fourier_calib`
    Calculate Fourier space calibration from real space.
:func:`make_probe`
    Create electron probe with specified aberrations.
:func:`aberration`
    Calculate aberration phase for the electron probe.
:func:`wavelength_ang`
    Calculate relativistic electron wavelength.
:func:`cbed`
    Simulate convergent beam electron diffraction patterns.
:func:`shift_beam_fourier`
    Shift electron beam in Fourier space for scanning.
:func:`stem_4d`
    Generate 4D-STEM data with multiple probe positions.
:func:`decompose_beam_to_modes`
    Decompose electron beam into orthogonal modes.
:func:`annular_detector`
    Simulate annular detector for STEM imaging.

Notes
-----
All functions are designed to work with JAX transformations
including ``jit``, ``grad``, and ``vmap``. Input arrays should
be properly typed and validated using the factory functions
from :mod:`ptyrodactyl.tools`.
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Union
from jax import lax
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Int,
    Num,
    PRNGKeyArray,
    jaxtyped,
)

from ptyrodactyl.tools import (
    STEM4D,
    CalibratedArray,
    PotentialSlices,
    ProbeModes,
    ScalarFloat,
    ScalarInt,
    ScalarNumeric,
    make_calibrated_array,
    make_probe_modes,
    make_stem4d,
)

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
@jax.jit
def transmission_func(
    pot_slice: Float[Array, " a b"], voltage_kv: ScalarNumeric
) -> Complex[Array, " a b"]:
    r"""Calculate the complex transmission function of a potential slice.

    Extended Summary
    ----------------
    Computes the interaction constant :math:`\sigma` and
    returns the phase object:

    .. math::

        T(\mathbf{r}) = \exp\bigl(i\,\sigma\,V(\mathbf{r})\bigr)

    Implementation Logic
    --------------------
    1. **Compute interaction constant** --
       :math:`\sigma = \frac{2\pi}{\lambda V}
       \frac{m_e c^2 + eV}{2 m_e c^2 + eV}`.
    2. **Apply complex exponential** --
       ``exp(1j * sigma * pot_slice)``.

    Parameters
    ----------
    pot_slice : Float[Array, " a b"]
        Projected potential slice in Kirkland units.
    voltage_kv : ScalarNumeric
        Microscope operating voltage in kiloelectronvolts.

    Returns
    -------
    trans : Complex[Array, " a b"]
        Complex transmission function for the slice.
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


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=["imsize_y", "imsize_x"])
def propagation_func(
    imsize_y: ScalarInt,
    imsize_x: ScalarInt,
    thickness_ang: ScalarNumeric,
    voltage_kv: ScalarNumeric,
    calib_ang: ScalarFloat,
) -> Complex[Array, " h w"]:
    r"""Compute the Fresnel propagation function for multislice.

    Extended Summary
    ----------------
    Computes the free-space propagator in Fourier space:

    .. math::

        P(\mathbf{q}) = \exp\bigl(
        -i\pi\lambda\Delta z\,|\mathbf{q}|^2\bigr)

    Implementation Logic
    --------------------
    1. **Build frequency grids** --
       ``jnp.fft.fftfreq`` for both axes.
    2. **Compute q-squared** --
       ``qx^2 + qy^2``.
    3. **Evaluate propagator** --
       ``exp(-i pi lambda dz q^2)``.

    Parameters
    ----------
    imsize_y : ScalarInt
        Grid size in pixels along the y-axis.
    imsize_x : ScalarInt
        Grid size in pixels along the x-axis.
    thickness_ang : ScalarNumeric
        Slice thickness (propagation distance) in Angstroms.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    calib_ang : ScalarFloat
        Pixel size in Angstroms.

    Returns
    -------
    prop : Complex[Array, " h w"]
        Fresnel propagation function in Fourier space.
    """
    qy: Num[Array, " h"] = jnp.fft.fftfreq(int(imsize_y), d=calib_ang)
    qx: Num[Array, " w"] = jnp.fft.fftfreq(int(imsize_x), d=calib_ang)
    lya: Num[Array, " h w"]
    lxa: Num[Array, " h w"]
    lya, lxa = jnp.meshgrid(qy, qx, indexing="ij")
    l_sq: Num[Array, " h w"] = jnp.square(lxa) + jnp.square(lya)
    lambda_angstrom: Float[Array, " "] = wavelength_ang(voltage_kv)
    prop: Complex[Array, " h w"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * l_sq
    )
    return prop


@jaxtyped(typechecker=beartype)
def fourier_coords(
    calibration: ScalarFloat | Float[Array, " 2"],
    image_size: Int[Array, " 2"],
) -> CalibratedArray:
    """Generate Fourier space coordinate arrays.

    Extended Summary
    ----------------
    Builds a 2D array of radial Fourier-space frequencies
    (in inverse Angstroms) suitable for diffraction
    calculations, returned as a
    :class:`~ptyrodactyl.tools.CalibratedArray`.

    Implementation Logic
    --------------------
    1. **Compute field of view** --
       ``image_size * calibration``.
    2. **Build frequency axes** --
       Centered arrays divided by field of view, then
       ``fftshift``-ed via ``jnp.roll``.
    3. **Radial frequency grid** --
       ``sqrt(qx^2 + qy^2)`` on the meshgrid.

    Parameters
    ----------
    calibration : ScalarFloat or Float[Array, " 2"]
        Pixel size in Angstroms in real space.
    image_size : Int[Array, " 2"]
        Grid size in pixels ``(H, W)``.

    Returns
    -------
    calibrated_inverse_array : CalibratedArray
        Radial Fourier-space frequencies with calibrations
        in inverse Angstroms. ``real_space`` is ``False``.
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
    inverse_space: Bool[Array, ""] = jnp.array(False)
    calibrated_inverse_array: CalibratedArray = make_calibrated_array(
        inverse_array, calib_inverse_y, calib_inverse_x, inverse_space
    )
    return calibrated_inverse_array


@jaxtyped(typechecker=beartype)
def fourier_calib(
    real_space_calib: Float[Array, " "] | Float[Array, " 2"],
    sizebeam: Int[Array, " 2"],
) -> Float[Array, " 2"]:
    """Compute Fourier-space calibration from real-space parameters.

    Implementation Logic
    --------------------
    1. **Compute field of view** --
       ``sizebeam * real_space_calib`` in Angstroms.
    2. **Invert** --
       ``1 / field_of_view`` gives inverse Angstroms per
       pixel.

    Parameters
    ----------
    real_space_calib : Float[Array, " "] or Float[Array, " 2"]
        Pixel size in Angstroms in real space.
    sizebeam : Int[Array, " 2"]
        Grid size in pixels ``(H, W)``.

    Returns
    -------
    inverse_space_calib : Float[Array, " 2"]
        Fourier calibration in inverse Angstroms per pixel.
    """
    field_of_view: Float[Array, " "] = jnp.multiply(
        jnp.float64(sizebeam), real_space_calib
    )
    inverse_space_calib: Float[Array, " 2"] = 1 / field_of_view
    return inverse_space_calib


@jaxtyped(typechecker=beartype)
def make_probe(
    aperture: ScalarNumeric,
    voltage: ScalarNumeric,
    image_size: Int[Array, " 2"],
    calibration_pm: ScalarFloat,
    defocus: ScalarNumeric = 0.0,
    c3: ScalarNumeric = 0.0,
    c5: ScalarNumeric = 0.0,
) -> Complex[Array, " h w"]:
    """Create an electron probe with spherical aberrations.

    Extended Summary
    ----------------
    Builds a probe wavefunction in Fourier space by applying
    an aperture mask and aberration phase, then inverse-FFTs
    to real space.

    Implementation Logic
    --------------------
    1. **Convert aperture** --
       From milliradians to radians, compute max spatial
       frequency ``l_max = aperture / wavelength``.
    2. **Build Fourier grid** --
       Frequency arrays from pixel size and image dimensions.
    3. **Apply aperture and aberrations** --
       Binary mask at ``l_max``, multiply by
       ``exp(-i * chi)`` from :func:`aberration`.
    4. **Inverse FFT** --
       ``ifftshift(ifft2(...))`` to obtain the real-space
       probe.

    Parameters
    ----------
    aperture : ScalarNumeric
        Aperture semi-angle in milliradians.
    voltage : ScalarNumeric
        Accelerating voltage in kiloelectronvolts.
    image_size : Int[Array, " 2"]
        Grid size in pixels ``(H, W)``.
    calibration_pm : ScalarFloat
        Real-space pixel size in picometers.
    defocus : ScalarNumeric, optional
        Defocus in Angstroms. Default is 0.
    c3 : ScalarNumeric, optional
        Third-order spherical aberration in Angstroms.
        Default is 0.
    c5 : ScalarNumeric, optional
        Fifth-order spherical aberration in Angstroms.
        Default is 0.

    Returns
    -------
    probe_real_space : Complex[Array, " h w"]
        Electron probe wavefunction in real space.

    See Also
    --------
    :func:`aberration` : Compute the aberration phase.
    """
    aperture: Float[Array, " "] = jnp.asarray(aperture / 1000.0)
    wavelength: Float[Array, " "] = wavelength_ang(voltage)
    l_max: Float[Array, " "] = aperture / wavelength
    image_y: ScalarInt
    image_x: ScalarInt
    image_y, image_x = image_size
    x_fov: Float[Array, " "] = image_x * 0.01 * calibration_pm
    y_fov: Float[Array, " "] = image_y * 0.01 * calibration_pm
    qx: Float[Array, " w"] = (
        jnp.arange((-image_x / 2), (image_x / 2), 1)
    ) / x_fov
    x_shifter: ScalarInt = image_x // 2
    qy: Float[Array, " h"] = (
        jnp.arange((-image_y / 2), (image_y / 2), 1)
    ) / y_fov
    y_shifter: ScalarInt = image_y // 2
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
    probe_real_space: Complex[Array, " h w"] = jnp.fft.ifftshift(
        jnp.fft.ifft2(a_dist)
    )
    return probe_real_space


@jaxtyped(typechecker=beartype)
@jax.jit
def aberration(
    fourier_coord: Float[Array, " H W"],
    lambda_angstrom: ScalarFloat,
    defocus: ScalarFloat = 0.0,
    c3: ScalarFloat = 0.0,
    c5: ScalarFloat = 0.0,
) -> Float[Array, " H W"]:
    r"""Calculate the aberration phase for the electron probe.

    Extended Summary
    ----------------
    Evaluates the aberration function:

    .. math::

        \chi(\mathbf{q}) = \frac{2\pi}{\lambda}\left(
        \frac{C_1\,\theta^2}{2}
        + \frac{C_3\,\theta^4}{4}
        + \frac{C_5\,\theta^6}{6}\right)

    where :math:`\theta = \lambda\,|\mathbf{q}|`.

    Implementation Logic
    --------------------
    1. **Compute scattering angle** --
       ``p = lambda * fourier_coord``.
    2. **Evaluate polynomial** --
       Sum defocus, C3, and C5 terms.
    3. **Scale by 2 pi / lambda** --
       Converts from path-length to phase.

    Parameters
    ----------
    fourier_coord : Float[Array, " H W"]
        Radial Fourier-space frequency in inverse Angstroms.
    lambda_angstrom : ScalarFloat
        Electron wavelength in Angstroms.
    defocus : ScalarFloat, optional
        Defocus (C1) in Angstroms. Default is 0.0.
    c3 : ScalarFloat, optional
        Third-order spherical aberration in Angstroms.
        Default is 0.0.
    c5 : ScalarFloat, optional
        Fifth-order spherical aberration in Angstroms.
        Default is 0.0.

    Returns
    -------
    chi_probe : Float[Array, " H W"]
        Aberration phase in radians.
    """
    p_matrix: Float[Array, " H W"] = lambda_angstrom * fourier_coord
    chi: Float[Array, " H W"] = (
        ((defocus * jnp.power(p_matrix, 2)) / 2)
        + ((c3 * (1e7) * jnp.power(p_matrix, 4)) / 4)
        + ((c5 * (1e7) * jnp.power(p_matrix, 6)) / 6)
    )
    chi_probe: Float[Array, " H W"] = (2 * jnp.pi * chi) / lambda_angstrom
    return chi_probe


@jaxtyped(typechecker=beartype)
@jax.jit
def wavelength_ang(voltage_kv: ScalarNumeric) -> Float[Array, " "]:
    r"""Calculate the relativistic electron wavelength.

    Extended Summary
    ----------------
    Uses the relativistic de Broglie relation:

    .. math::

        \lambda = \frac{hc}{\sqrt{eV\,(2 m_e c^2 + eV)}}

    Implementation Logic
    --------------------
    1. **Convert voltage** --
       kV to eV then to Joules.
    2. **Relativistic formula** --
       Compute wavelength in metres.
    3. **Convert to Angstroms** --
       Multiply by :math:`10^{10}`.

    Parameters
    ----------
    voltage_kv : ScalarNumeric
        Accelerating voltage in kiloelectronvolts.

    Returns
    -------
    lambda_angstroms : Float[Array, " "]
        Electron wavelength in Angstroms.

    Notes
    -----
    Assumes clean input (no negative or NaN values).
    Validation should happen in preprocessing.
    """
    m: Float[Array, " "] = jnp.asarray(9.109383e-31)
    e: Float[Array, " "] = jnp.asarray(1.602177e-19)
    c: Float[Array, " "] = jnp.asarray(299792458.0)
    h: Float[Array, " "] = jnp.asarray(6.62607e-34)

    ev: Float[Array, " "] = (
        jnp.float64(voltage_kv) * jnp.float64(1000.0) * jnp.float64(e)
    )
    numerator: Float[Array, " "] = jnp.multiply(jnp.square(h), jnp.square(c))
    denominator: Float[Array, " "] = jnp.multiply(
        ev, ((2 * m * jnp.square(c)) + ev)
    )
    wavelength_meters: Float[Array, " "] = jnp.sqrt(numerator / denominator)
    lambda_angstroms: Float[Array, " "] = jnp.asarray(1e10) * wavelength_meters
    return lambda_angstroms


@jaxtyped(typechecker=beartype)
@jax.jit
def cbed(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    voltage_kv: ScalarNumeric,
) -> CalibratedArray:
    """Simulate a CBED pattern via the multislice algorithm.

    Extended Summary
    ----------------
    Propagates one or more beam modes through one or more
    potential slices to produce a Convergent Beam Electron
    Diffraction (CBED) intensity pattern.

    Implementation Logic
    --------------------
    1. **Ensure 3D arrays** --
       Promote single-slice / single-mode inputs.
    2. **Build propagator** --
       :func:`propagation_func` from slice thickness.
    3. **Scan over slices** --
       ``lax.scan``: transmit, then propagate (skip
       propagation on the last slice).
    4. **Compute intensity** --
       FFT to Fourier space, square modulus, sum over modes.

    Parameters
    ----------
    pot_slices : PotentialSlices
        Potential slices. ``slices`` has shape ``(H, W, S)``
        in Kirkland units; ``slice_thickness`` in Angstroms;
        ``calib`` pixel size in Angstroms.
    beam : ProbeModes
        Electron beam. ``modes`` has shape ``(H, W, M)``;
        ``weights`` shape ``(M,)``; ``calib`` in Angstroms.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.

    Returns
    -------
    cbed_pytree : CalibratedArray
        CBED intensity pattern as a
        :class:`~ptyrodactyl.tools.CalibratedArray` with
        Fourier-space calibrations. ``real_space`` is
        ``False``.
    """
    calib_ang: Float[Array, ""] = jnp.amin(
        jnp.array([pot_slices.calib, beam.calib])
    )
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
        carry: Complex[Array, " H W M"], slice_idx: ScalarInt
    ) -> Tuple[Complex[Array, " H W M"], None]:
        """Propagate wave through one potential slice.

        Parameters
        ----------
        carry : Complex[Array, " H W M"]
            Current wave state.
        slice_idx : ScalarInt
            Index of the current slice.

        Returns
        -------
        wave : Complex[Array, " H W M"]
            Updated wave.
        None
            No stacked output.
        """
        wave: Complex[Array, " H W M"] = carry
        pot_single_slice: Float[Array, " H W 1"] = lax.dynamic_slice_in_dim(
            pot_slice, slice_idx, 1, axis=2
        )
        pot_single_slice: Float[Array, " H W"] = jnp.squeeze(
            pot_single_slice, axis=2
        )
        trans_slice: Complex[Array, " H W"] = transmission_func(
            pot_single_slice, voltage_kv
        )
        wave = wave * trans_slice[..., jnp.newaxis]

        def _propagate(
            w: Complex[Array, " H W M"],
        ) -> Complex[Array, " H W M"]:
            """Apply Fresnel propagation in Fourier space.

            Parameters
            ----------
            w : Complex[Array, " H W M"]
                Wave in real space.

            Returns
            -------
            Complex[Array, " H W M"]
                Wave after propagation.
            """
            w_k: Complex[Array, " H W M"] = jnp.fft.fft2(w, axes=(0, 1))
            w_k = w_k * slice_transmission[..., jnp.newaxis]
            return jnp.fft.ifft2(w_k, axes=(0, 1)).astype(dtype)

        is_last_slice: Bool[Array, ""] = jnp.array(slice_idx == num_slices - 1)
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
    real_space_fov: Float[Array, " "] = jnp.multiply(
        beam_modes.shape[0], calib_ang
    )
    inverse_space_calib: Float[Array, " "] = 1 / real_space_fov
    cbed_pytree: CalibratedArray = make_calibrated_array(
        cbed_pattern, inverse_space_calib, inverse_space_calib, False
    )
    return cbed_pytree


@jaxtyped(typechecker=beartype)
@jax.jit
def shift_beam_fourier(
    beam: Union[Float[Array, " hh ww *mm"], Complex[Array, " hh ww *mm"]],
    pos: Float[Array, " #pp 2"],
    calib_ang: ScalarFloat,
) -> Complex128[Array, "#pp hh ww #mm"]:
    """Shift beam to new position(s) via Fourier phase ramp.

    Implementation Logic
    --------------------
    1. **FFT the beam** --
       All modes to Fourier space.
    2. **Per-position phase ramp** --
       ``exp(-2 pi i (qy * dy + qx * dx))`` applied to
       the Fourier-space beam.
    3. **Inverse FFT** --
       Back to real space for each position.

    Parameters
    ----------
    beam : Float[Array, " hh ww *mm"] or Complex[Array, " hh ww *mm"]
        Electron beam modes.
    pos : Float[Array, " #P 2"]
        Shift position(s) ``(y, x)`` in Angstroms. Can be a
        single ``[2]`` or multiple ``[P, 2]``.
    calib_ang : ScalarFloat
        Pixel size in Angstroms.

    Returns
    -------
    all_shifted_beams : Complex128[Array, "#P H W #M"]
        Shifted beam(s) for all positions and modes.
    """
    our_beam: Complex128[Array, "H W #M"] = jnp.atleast_3d(
        beam.astype(jnp.complex128)
    )
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
    beam_k: Complex128[Array, " hh ww #mm"] = jnp.fft.fft2(
        our_beam, axes=(0, 1)
    )

    def _apply_shift(position_idx: int) -> Complex128[Array, " hh ww #mm"]:
        """Apply Fourier phase ramp shift for one position.

        Parameters
        ----------
        position_idx : int
            Index into the positions array.

        Returns
        -------
        shifted_beam : Complex128[Array, " hh ww #mm"]
            Beam shifted to the requested position.
        """
        y_shift: ScalarNumeric
        x_shift: ScalarNumeric
        y_shift, x_shift = pos[position_idx, 0], pos[position_idx, 1]
        phase: Float[Array, " hh ww"] = (
            -2.0 * jnp.pi * ((qya * y_shift) + (qxa * x_shift))
        )
        phase_shift: Complex[Array, " hh ww"] = jnp.exp(1j * phase)
        phase_shift_expanded: Complex128[Array, " hh ww 1"] = phase_shift[
            ..., jnp.newaxis
        ]
        shifted_beam_k: Complex128[Array, " hh ww #mm"] = (
            beam_k * phase_shift_expanded
        )
        shifted_beam: Complex128[Array, " hh ww #mm"] = jnp.fft.ifft2(
            shifted_beam_k, axes=(0, 1)
        )
        return shifted_beam

    all_shifted_beams: Complex128[Array, " #pp hh ww #mm"] = jax.vmap(
        _apply_shift
    )(jnp.arange(num_positions))
    return all_shifted_beams


@jaxtyped(typechecker=beartype)
@jax.jit
def stem_4d(
    pot_slice: PotentialSlices,
    beam: ProbeModes,
    positions: Num[Array, "#P 2"],
    voltage_kv: ScalarNumeric,
    calib_ang: ScalarFloat,
) -> STEM4D:
    """Generate 4D-STEM data at multiple probe positions.

    Extended Summary
    ----------------
    Shifts the beam to each scan position and runs
    :func:`cbed` for each, collecting diffraction patterns
    into a :class:`~ptyrodactyl.tools.STEM4D` dataset.

    Implementation Logic
    --------------------
    1. **Shift beam** --
       :func:`shift_beam_fourier` to all positions at once.
    2. **CBED per position** --
       ``jax.vmap`` over positions.
    3. **Build STEM4D** --
       Combine patterns with calibrations and scan
       positions.

    Parameters
    ----------
    pot_slice : PotentialSlices
        Potential slices for the sample.
    beam : ProbeModes
        Electron beam modes.
    positions : Num[Array, "#P 2"]
        Scan positions ``(y, x)`` in pixels. P is the number
        of positions.
    voltage_kv : ScalarNumeric
        Accelerating voltage in kilovolts.
    calib_ang : ScalarFloat
        Pixel size in Angstroms.

    Returns
    -------
    stem4d_data : STEM4D
        Complete 4D-STEM dataset with diffraction patterns,
        calibrations, scan positions, and voltage.

    See Also
    --------
    :func:`cbed` : Single-position CBED simulation.
    :func:`shift_beam_fourier` : Fourier-space beam shifting.
    """
    shifted_beams: Complex[Array, " P H W #M"] = shift_beam_fourier(
        beam.modes, positions, calib_ang
    )

    def _process_single_position(pos_idx: ScalarInt) -> Float[Array, " H W"]:
        """Compute CBED pattern for a single beam position.

        Parameters
        ----------
        pos_idx : ScalarInt
            Index into the shifted beams array.

        Returns
        -------
        Float[Array, " H W"]
            CBED intensity pattern at this position.
        """
        current_beam: Complex[Array, " H W #M"] = jnp.take(
            shifted_beams, pos_idx, axis=0
        )
        current_probe_modes: ProbeModes = ProbeModes(
            modes=current_beam,
            weights=beam.weights,
            calib=beam.calib,
        )
        cbed_result: CalibratedArray = cbed(
            pot_slices=pot_slice,
            beam=current_probe_modes,
            voltage_kv=voltage_kv,
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


@jaxtyped(typechecker=beartype)
def decompose_beam_to_modes(
    beam: CalibratedArray,
    num_modes: ScalarInt,
    first_mode_weight: ScalarFloat = 0.6,
) -> ProbeModes:
    """Decompose an electron beam into orthogonal modes.

    Extended Summary
    ----------------
    Creates *num_modes* spatially orthogonal modes that
    together preserve the total intensity of the input beam.
    Useful for modelling partial spatial coherence.

    Implementation Logic
    --------------------
    1. **Flatten beam** --
       Reshape to 1D vector of length ``H * W``.
    2. **Random orthogonal basis** --
       QR decomposition of a random complex matrix gives
       orthonormal columns.
    3. **Weight and scale** --
       First mode gets ``first_mode_weight``; remaining
       weight is split equally. Each mode is scaled by
       ``sqrt(weight) * sqrt(original_intensity)``.
    4. **Reshape** --
       Back to ``(H, W, M)`` spatial dimensions.

    Parameters
    ----------
    beam : CalibratedArray
        Electron beam to decompose.
    num_modes : ScalarInt
        Number of modes to generate.
    first_mode_weight : ScalarFloat, optional
        Weight of the first (dominant) mode. Default is 0.6.
        Must be below 1.0.

    Returns
    -------
    probe_modes : ProbeModes
        Decomposed probe with ``modes`` shape ``(H, W, M)``,
        ``weights`` shape ``(M,)``, and ``calib`` in
        Angstroms.
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
        key1, (tp, int(num_modes)), dtype=jnp.float64
    )
    random_imag: Float[Array, " tp mm"] = jax.random.normal(
        key2, (tp, int(num_modes)), dtype=jnp.float64
    )
    random_matrix: Complex[Array, " tp mm"] = random_real + (1j * random_imag)
    qq: Complex[Array, " tp mm"]
    qq, _ = jnp.linalg.qr(random_matrix, mode="reduced")
    original_intensity: Float[Array, " tp"] = jnp.square(jnp.abs(beam_flat))
    weights: Float[Array, " mm"] = jnp.zeros(num_modes, dtype=jnp.float64)
    weights = weights.at[0].set(first_mode_weight)
    remaining_weight: ScalarFloat = (1.0 - first_mode_weight) / max(
        1, num_modes - 1
    )
    weights = weights.at[1:].set(remaining_weight)
    sqrt_weights: Float[Array, " mm"] = jnp.sqrt(weights)
    sqrt_intensity: Float[Array, " tp 1"] = jnp.sqrt(
        original_intensity
    ).reshape(-1, 1)
    weighted_modes: Complex[Array, " tp mm"] = (
        qq * sqrt_intensity * sqrt_weights
    )
    multimodal_beam: Complex[Array, " hh ww mm"] = weighted_modes.reshape(
        hh, ww, num_modes
    )
    probe_modes: ProbeModes = make_probe_modes(
        modes=multimodal_beam, weights=weights, calib=beam.calib_y
    )
    return probe_modes


@jaxtyped(typechecker=beartype)
def annular_detector(
    stem4d_data: STEM4D,
    collection_angles: Float[Array, " 2"],
) -> CalibratedArray:
    """Integrate 4D-STEM data with an annular detector.

    Extended Summary
    ----------------
    Creates a virtual annular detector between inner and outer
    collection angles, integrates each diffraction pattern
    within the annulus, and reshapes to a 2D STEM image.

    Implementation Logic
    --------------------
    1. **Convert angles** --
       mrad to inverse Angstroms via the electron wavelength.
    2. **Build annular mask** --
       Boolean mask on the Fourier-space coordinate grid.
    3. **Integrate** --
       ``vmap`` over patterns, sum within the mask.
    4. **Reshape** --
       Map 1D scan positions to a 2D image grid.

    Parameters
    ----------
    stem4d_data : STEM4D
        4D-STEM dataset. ``data`` shape ``(P, H, W)``,
        ``fourier_space_calib`` in inverse Angstroms per
        pixel, ``voltage_kv`` in kilovolts.
    collection_angles : Float[Array, " 2"]
        Inner and outer collection angles in milliradians,
        ``[inner, outer]``.

    Returns
    -------
    stem_image : CalibratedArray
        Real-space STEM image with ``real_space = True``
        and calibrations in Angstroms per pixel.
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

    annular_mask: Bool[Array, " hh ww"] = (q_radius >= inner_k) & (
        q_radius <= outer_k
    )

    def _integrate_pattern(
        pattern: Float[Array, " hh ww"],
    ) -> Float[Array, " "]:
        """Sum intensity within the annular mask.

        Parameters
        ----------
        pattern : Float[Array, " hh ww"]
            Single diffraction pattern.

        Returns
        -------
        Float[Array, " "]
            Integrated intensity.
        """
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

    stem_image_2d: Float[Array, " ny nx"] = integrated_intensities.reshape(
        ny, nx
    )

    stem_image: CalibratedArray = make_calibrated_array(
        data_array=stem_image_2d,
        calib_y=stem4d_data.real_space_calib,
        calib_x=stem4d_data.real_space_calib,
        real_space=True,
    )

    return stem_image
