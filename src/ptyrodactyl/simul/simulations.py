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
:func:`cbed_amplitude`
    Simulate complex CBED detector amplitudes.
:func:`cbed_image`
    Simulate convergent beam electron diffraction intensity patterns.
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
from :mod:`ptyrodactyl.types`.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Tuple, Union
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

from ptyrodactyl.tools import relativistic_wavelength_ang
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    M_E,
    STEM4D,
    CalibratedArray,
    DetectorConfig,
    Distribution,
    EnsembleAxes,
    MicroscopeConfig,
    PotentialSlices,
    ProbeModes,
    ReductionMode,
    create_calibrated_array,
    create_detector_config,
    create_distribution,
    create_probe_modes,
    create_stem4d,
    scalar_float,
    scalar_int,
    scalar_num,
)

from .reduce import apply_distribution, apply_distributions


@jaxtyped(typechecker=beartype)
@jax.jit
def transmission_func(
    pot_slice: Float[Array, " a b"], voltage_kv: scalar_num
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
    voltage_kv : scalar_num
        Microscope operating voltage in kiloelectronvolts.

    Returns
    -------
    trans : Complex[Array, " a b"]
        Complex transmission function for the slice.
    """
    voltage: Float[Array, " "] = jnp.multiply(voltage_kv, jnp.asarray(1000.0))
    e_e: Float[Array, " "] = jnp.float64(E_CHARGE)
    c: Float[Array, " "] = jnp.float64(C_LIGHT)
    ev: Float[Array, " "] = jnp.multiply(e_e, voltage)
    lambda_angstrom: Float[Array, " "] = (
        relativistic_wavelength_ang(voltage_kv)
    )
    einstein_energy: Float[Array, " "] = jnp.multiply(
        jnp.float64(M_E), jnp.square(c)
    )
    sigma: Float[Array, " "] = (
        (2 * jnp.pi / (lambda_angstrom * voltage)) * (einstein_energy + ev)
    ) / ((2 * einstein_energy) + ev)
    trans: Complex[Array, " a b"] = jnp.exp(1j * sigma * pot_slice)
    return trans


@jaxtyped(typechecker=beartype)
@jax.jit(static_argnames=["imsize_y", "imsize_x"])
def propagation_func(
    imsize_y: scalar_int,
    imsize_x: scalar_int,
    thickness_ang: scalar_num,
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
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
    imsize_y : scalar_int
        Grid size in pixels along the y-axis.
    imsize_x : scalar_int
        Grid size in pixels along the x-axis.
    thickness_ang : scalar_num
        Slice thickness (propagation distance) in Angstroms.
    voltage_kv : scalar_num
        Accelerating voltage in kilovolts.
    calib_ang : scalar_float
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
    lambda_angstrom: Float[Array, " "] = (
        relativistic_wavelength_ang(voltage_kv)
    )
    prop: Complex[Array, " h w"] = jnp.exp(
        (-1j) * jnp.pi * lambda_angstrom * thickness_ang * l_sq
    )
    return prop


@jaxtyped(typechecker=beartype)
def fourier_coords(
    calibration: scalar_float | Float[Array, " 2"],
    image_size: Int[Array, " 2"],
) -> CalibratedArray:
    """Generate Fourier space coordinate arrays.

    Extended Summary
    ----------------
    Builds a 2D array of radial Fourier-space frequencies
    (in inverse Angstroms) suitable for diffraction
    calculations, returned as a
    :class:`~ptyrodactyl.types.CalibratedArray`.

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
    calibration : scalar_float or Float[Array, " 2"]
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
    calibrated_inverse_array: CalibratedArray = create_calibrated_array(
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
    microscope: MicroscopeConfig,
    detector: DetectorConfig,
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
    microscope : MicroscopeConfig
        Microscope voltage, aperture, aberration coefficients, and static
        probe shape.
    detector : DetectorConfig
        Detector calibration carrying the probe pixel size in picometers.

    Returns
    -------
    probe_real_space : Complex[Array, " h w"]
        Electron probe wavefunction in real space.

    See Also
    --------
    :func:`aberration` : Compute the aberration phase.
    """
    if microscope.probe_shape is None:
        raise ValueError("microscope.probe_shape is required")

    aperture: Float[Array, " "] = jnp.asarray(
        microscope.aperture_mrad / 1000.0
    )
    wavelength: Float[Array, " "] = relativistic_wavelength_ang(
        microscope.voltage_kv
    )
    l_max: Float[Array, " "] = aperture / wavelength
    image_y: scalar_int
    image_x: scalar_int
    image_y, image_x = microscope.probe_shape
    x_fov: Float[Array, " "] = image_x * 0.01 * detector.probe_calibration_pm
    y_fov: Float[Array, " "] = image_y * 0.01 * detector.probe_calibration_pm
    qx: Float[Array, " w"] = (
        jnp.arange((-image_x / 2), (image_x / 2), 1)
    ) / x_fov
    x_shifter: scalar_int = image_x // 2
    qy: Float[Array, " h"] = (
        jnp.arange((-image_y / 2), (image_y / 2), 1)
    ) / y_fov
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
        inverse_real_matrix,
        wavelength,
        microscope.defocus_ang,
        microscope.c3_ang,
        microscope.c5_ang,
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
    lambda_angstrom: scalar_float,
    defocus: scalar_float = 0.0,
    c3: scalar_float = 0.0,
    c5: scalar_float = 0.0,
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
    lambda_angstrom : scalar_float
        Electron wavelength in Angstroms.
    defocus : scalar_float, optional
        Defocus (C1) in Angstroms. Default is 0.0.
    c3 : scalar_float, optional
        Third-order spherical aberration in Angstroms.
        Default is 0.0.
    c5 : scalar_float, optional
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


def _cbed_amplitude_from_slice_provider(
    beam_modes: Complex[Array, " H W M"],
    num_slices: int,
    slice_thickness: scalar_num,
    voltage_kv: scalar_num,
    calib_ang: scalar_float,
    slice_provider: Callable[[scalar_int], Float[Array, " H W"]],
) -> Complex[Array, " H W M"]:
    """Return detector amplitudes from one provider-backed multislice scan."""
    dtype: jnp.dtype = beam_modes.dtype
    propagator: Complex[Array, " H W"] = propagation_func(
        beam_modes.shape[0],
        beam_modes.shape[1],
        slice_thickness,
        voltage_kv,
        calib_ang,
    ).astype(dtype)
    init_wave: Complex[Array, " H W M"] = jnp.copy(beam_modes)

    def _scan_fn(
        carry: Complex[Array, " H W M"], slice_idx: scalar_int
    ) -> Tuple[Complex[Array, " H W M"], None]:
        """Propagate wave through one potential slice."""
        wave: Complex[Array, " H W M"] = carry
        pot_single_slice: Float[Array, " H W"] = slice_provider(slice_idx)
        trans_slice: Complex[Array, " H W"] = transmission_func(
            pot_single_slice, voltage_kv
        )
        wave = wave * trans_slice[..., jnp.newaxis]

        def _propagate(
            w: Complex[Array, " H W M"],
        ) -> Complex[Array, " H W M"]:
            """Apply Fresnel propagation in Fourier space."""
            w_k: Complex[Array, " H W M"] = jnp.fft.fft2(w, axes=(0, 1))
            w_k = w_k * propagator[..., jnp.newaxis]
            return jnp.fft.ifft2(w_k, axes=(0, 1)).astype(dtype)

        is_last_slice: Bool[Array, ""] = jnp.array(slice_idx == num_slices - 1)
        wave = lax.cond(is_last_slice, lambda w: w, _propagate, wave)
        return wave, None

    final_wave: Complex[Array, " H W M"]
    final_wave, _ = lax.scan(_scan_fn, init_wave, jnp.arange(num_slices))
    detector_amplitude: Complex[Array, " H W M"] = jnp.fft.fftshift(
        jnp.fft.fft2(final_wave, axes=(0, 1)), axes=(0, 1)
    )
    return detector_amplitude


@jaxtyped(typechecker=beartype)
@jax.jit
def cbed_amplitude(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    microscope: MicroscopeConfig,
) -> Complex[Array, " H W M"]:
    """Return complex CBED detector fields for each probe mode.

    Return the complex field so Layer 1 can choose coherent or incoherent
    reduction explicitly. The multislice scan transmits and propagates all
    modes together, Fourier-transforms the exit waves, and leaves the final
    mode axis intact.

    Parameters
    ----------
    pot_slices : PotentialSlices
        Potential slices. ``slices`` has shape ``(H, W, S)``
        in Kirkland units; ``slice_thickness`` in Angstroms;
        ``calib`` pixel size in Angstroms.
    beam : ProbeModes
        Electron beam. ``modes`` has shape ``(H, W, M)``;
        ``weights`` shape ``(M,)``; ``calib`` in Angstroms.
    microscope : MicroscopeConfig
        Microscope voltage and ensemble configuration.

    Returns
    -------
    detector_amplitude : Complex[Array, " H W M"]
        Complex detector-plane amplitudes with the probe-mode axis retained.
    """
    calib_ang: Float[Array, ""] = jnp.amin(
        jnp.array([pot_slices.calib, beam.calib])
    )
    pot_slice: Float[Array, " H W S"] = jnp.atleast_3d(pot_slices.slices)
    beam_modes: Complex[Array, " H W M"] = jnp.atleast_3d(beam.modes)
    num_slices: int = pot_slice.shape[-1]

    def _slice_at(slice_idx: scalar_int) -> Float[Array, " H W"]:
        pot_single_slice: Float[Array, " H W 1"] = lax.dynamic_slice_in_dim(
            pot_slice, slice_idx, 1, axis=2
        )
        squeezed_slice: Float[Array, " H W"] = jnp.squeeze(
            pot_single_slice, axis=2
        )
        return squeezed_slice

    detector_amplitude: Complex[Array, " H W M"] = (
        _cbed_amplitude_from_slice_provider(
            beam_modes,
            num_slices,
            pot_slices.slice_thickness,
            microscope.voltage_kv,
            calib_ang,
            _slice_at,
        )
    )
    return detector_amplitude


@jaxtyped(typechecker=beartype)
def probe_modes_to_distribution(probe: ProbeModes) -> Distribution:
    """Return the explicit incoherent distribution for probe modes."""
    mode_count: int = jnp.atleast_3d(probe.modes).shape[-1]
    samples: Float[Array, " M 1"] = jnp.arange(
        mode_count,
        dtype=jnp.float64,
    )[:, jnp.newaxis]
    distribution: Distribution = create_distribution(
        samples=samples,
        weights=probe.weights,
        reduction=ReductionMode.INCOHERENT,
        axis_id="probe_modes",
    )
    return distribution


def _has_extra_ensemble_axes(ensemble: EnsembleAxes) -> bool:
    """Return whether cbed_image should use the generalized axis binder."""
    return (
        ensemble.probe_modes is not None
        or ensemble.position_jitter is not None
        or ensemble.coherence is not None
    )


def _ensemble_axes_for_cbed(
    ensemble: EnsembleAxes,
    beam: ProbeModes,
) -> tuple[Distribution, ...]:
    """Return ordered CBED axes with probe modes first when needed."""
    axes: list[Distribution] = []
    mode_count: int = jnp.atleast_3d(beam.modes).shape[-1]
    if ensemble.probe_modes is not None:
        axes.append(ensemble.probe_modes)
    elif mode_count > 1:
        axes.append(probe_modes_to_distribution(beam))
    if ensemble.position_jitter is not None:
        axes.append(ensemble.position_jitter)
    if ensemble.coherence is not None:
        axes.append(ensemble.coherence)
    if len(axes) == 0:
        axes.append(probe_modes_to_distribution(beam))
    return tuple(axes)


@jaxtyped(typechecker=beartype)
@jax.jit
def cbed_image(
    pot_slices: PotentialSlices,
    beam: ProbeModes,
    microscope: MicroscopeConfig,
) -> CalibratedArray:
    """Simulate a CBED intensity image via explicit mode reduction.

    Computes :func:`cbed_amplitude` exactly once, then binds the retained mode
    axis into :func:`apply_distribution`: ``samples = arange(M)[:, None]`` and
    ``weights = probe.weights``. The bound closure indexes
    ``amps[..., sample[0].astype(int)]`` so the Phase-1 reducer performs the
    only public detector ``|.|^2`` and incoherent mode sum.

    Parameters
    ----------
    pot_slices : PotentialSlices
        Potential slices for multislice propagation.
    beam : ProbeModes
        Probe modes and explicit incoherent mode weights.
    microscope : MicroscopeConfig
        Microscope voltage and optional ensemble axes.

    Returns
    -------
    cbed_pytree : CalibratedArray
        CBED intensity pattern with Fourier-space calibrations.
    """
    if _has_extra_ensemble_axes(microscope.ensemble):
        calib_ang: Float[Array, ""] = jnp.amin(
            jnp.array([pot_slices.calib, beam.calib])
        )
        detector: DetectorConfig = create_detector_config(
            real_space_calib_ang=calib_ang,
            probe_calibration_pm=calib_ang * 100.0,
        )
        axes: tuple[Distribution, ...] = _ensemble_axes_for_cbed(
            microscope.ensemble,
            beam,
        )
        from .producers import bind_cbed_axes  # noqa: PLC0415

        bound_amplitude = bind_cbed_axes(
            pot_slices=pot_slices,
            probe_modes=beam,
            microscope=microscope,
            detector=detector,
            axes=axes,
        )
        cbed_pattern: Float[Array, " H W"] = apply_distributions(
            axes,
            bound_amplitude,
        )
        real_space_fov: Float[Array, " "] = jnp.multiply(
            beam.modes.shape[0], calib_ang
        )
        inverse_space_calib: Float[Array, " "] = 1 / real_space_fov
        cbed_pytree: CalibratedArray = create_calibrated_array(
            cbed_pattern, inverse_space_calib, inverse_space_calib, False
        )
        return cbed_pytree

    amplitudes: Complex[Array, " H W M"] = cbed_amplitude(
        pot_slices=pot_slices,
        beam=beam,
        microscope=microscope,
    )
    distribution: Distribution = probe_modes_to_distribution(beam)

    def _mode_amplitude(sample: Float[Array, " D"]) -> Complex[Array, " H W"]:
        mode_idx: Int[Array, ""] = sample[0].astype(jnp.int32)
        amplitude: Complex[Array, " H W"] = amplitudes[..., mode_idx]
        return amplitude

    cbed_pattern: Float[Array, " H W"] = apply_distribution(
        distribution,
        _mode_amplitude,
    )
    calib_ang: Float[Array, ""] = jnp.amin(
        jnp.array([pot_slices.calib, beam.calib])
    )
    real_space_fov: Float[Array, " "] = jnp.multiply(
        amplitudes.shape[0], calib_ang
    )
    inverse_space_calib: Float[Array, " "] = 1 / real_space_fov
    cbed_pytree: CalibratedArray = create_calibrated_array(
        cbed_pattern, inverse_space_calib, inverse_space_calib, False
    )
    return cbed_pytree


@jaxtyped(typechecker=beartype)
@jax.jit
def shift_beam_fourier(
    beam: Union[Float[Array, " hh ww *mm"], Complex[Array, " hh ww *mm"]],
    pos: Float[Array, " #pp 2"],
    calib_ang: scalar_float,
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
    calib_ang : scalar_float
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

    def _apply_shift(
        position_idx: scalar_int,
    ) -> Complex128[Array, " hh ww #mm"]:
        """Apply Fourier phase ramp shift for one position.

        Parameters
        ----------
        position_idx : scalar_int
            Index into the positions array.

        Returns
        -------
        shifted_beam : Complex128[Array, " hh ww #mm"]
            Beam shifted to the requested position.
        """
        y_shift: scalar_num
        x_shift: scalar_num
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
    microscope: MicroscopeConfig,
    detector: DetectorConfig,
) -> STEM4D:
    """Generate 4D-STEM data at multiple probe positions.

    Extended Summary
    ----------------
    Shifts the beam to each scan position and runs
    :func:`cbed_image` for each, collecting diffraction patterns
    into a :class:`~ptyrodactyl.types.STEM4D` dataset.

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
    microscope : MicroscopeConfig
        Microscope voltage and optional ensemble axes.
    detector : DetectorConfig
        Scan positions and real-space calibration.

    Returns
    -------
    stem4d_data : STEM4D
        Complete 4D-STEM dataset with diffraction patterns,
        calibrations, scan positions, and voltage.

    See Also
    --------
    :func:`cbed_image` : Single-position CBED intensity simulation.
    :func:`shift_beam_fourier` : Fourier-space beam shifting.
    """
    if detector.scan_positions_px is None:
        raise ValueError("detector.scan_positions_px is required")

    positions: Float[Array, "#P 2"] = detector.scan_positions_px
    calib_ang: Float[Array, ""] = detector.real_space_calib_ang
    shifted_beams: Complex[Array, " P H W #M"] = shift_beam_fourier(
        beam.modes, positions, calib_ang
    )

    def _process_single_position(pos_idx: scalar_int) -> Float[Array, " H W"]:
        """Compute CBED pattern for a single beam position.

        Parameters
        ----------
        pos_idx : scalar_int
            Index into the shifted beams array.

        Returns
        -------
        Float[Array, " H W"]
            CBED intensity pattern at this position.
        """
        current_beam: Complex[Array, " H W #M"] = jnp.take(
            shifted_beams, pos_idx, axis=0
        )
        current_probe_modes: ProbeModes = eqx.tree_at(
            lambda probe: probe.modes,
            beam,
            current_beam,
        )
        cbed_result: CalibratedArray = cbed_image(
            pot_slices=pot_slice,
            beam=current_probe_modes,
            microscope=microscope,
        )
        return cbed_result.data_array

    cbed_patterns: Float[Array, " P H W"] = jax.vmap(_process_single_position)(
        jnp.arange(positions.shape[0])
    )
    detector_calib_ang: Float[Array, ""] = jnp.amin(
        jnp.array([pot_slice.calib, beam.calib])
    )
    real_space_fov: Float[Array, " "] = jnp.multiply(
        shifted_beams.shape[1], detector_calib_ang
    )
    fourier_calib: Float[Array, " "] = 1 / real_space_fov
    scan_positions_ang: Float[Array, " P 2"] = positions * calib_ang
    stem4d_data: STEM4D = create_stem4d(
        data=cbed_patterns,
        real_space_calib=calib_ang,
        fourier_space_calib=fourier_calib,
        scan_positions=scan_positions_ang,
        voltage_kv=microscope.voltage_kv,
    )
    return stem4d_data


@jaxtyped(typechecker=beartype)
def decompose_beam_to_modes(
    beam: CalibratedArray,
    num_modes: scalar_int,
    key: PRNGKeyArray,
    first_mode_weight: scalar_float = 0.6,
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
    3. **Scale modes** --
       First mode gets ``first_mode_weight``; remaining
       weight is split equally. Each orthonormal mode is scaled by
       ``sqrt(original_intensity)``; ``ProbeModes.weights`` is the only
       carrier of the incoherent mixture weights.
    4. **Reshape** --
       Back to ``(H, W, M)`` spatial dimensions.

    Parameters
    ----------
    beam : CalibratedArray
        Electron beam to decompose.
    num_modes : scalar_int
        Number of modes to generate.
    key : PRNGKeyArray
        Random key used to generate the orthogonal modal basis.
    first_mode_weight : scalar_float, optional
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
    mode_count: int = int(num_modes)
    beam_flat: Complex[Array, " tp"] = beam.data_array.reshape(-1)
    key1: PRNGKeyArray
    key2: PRNGKeyArray
    key1, key2 = jax.random.split(key)
    random_real: Float[Array, " tp mm"] = jax.random.normal(
        key1, (tp, mode_count), dtype=jnp.float64
    )
    random_imag: Float[Array, " tp mm"] = jax.random.normal(
        key2, (tp, mode_count), dtype=jnp.float64
    )
    random_matrix: Complex[Array, " tp mm"] = random_real + (1j * random_imag)
    qq: Complex[Array, " tp mm"]
    qq, _ = jnp.linalg.qr(random_matrix, mode="reduced")
    original_intensity: Float[Array, " tp"] = jnp.square(jnp.abs(beam_flat))
    weights: Float[Array, " mm"] = jnp.zeros(mode_count, dtype=jnp.float64)
    weights = weights.at[0].set(first_mode_weight)
    remaining_weight: scalar_float = (1.0 - first_mode_weight) / max(
        1, mode_count - 1
    )
    weights = weights.at[1:].set(remaining_weight)
    sqrt_intensity: Float[Array, " tp 1"] = jnp.sqrt(
        original_intensity
    ).reshape(-1, 1)
    weighted_modes: Complex[Array, " tp mm"] = qq * sqrt_intensity
    multimodal_beam: Complex[Array, " hh ww mm"] = weighted_modes.reshape(
        hh, ww, mode_count
    )
    probe_modes: ProbeModes = create_probe_modes(
        modes=multimodal_beam, weights=weights, calib=beam.calib_y
    )
    return probe_modes


@jaxtyped(typechecker=beartype)
def annular_detector(
    stem4d_data: STEM4D,
    detector: DetectorConfig,
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
    detector : DetectorConfig
        Annular collection angles and static raster shape.

    Returns
    -------
    stem_image : CalibratedArray
        Real-space STEM image with ``real_space = True``
        and calibrations in Angstroms per pixel.
    """
    if detector.scan_shape is None:
        raise ValueError("detector.scan_shape is required")

    wavelength: Float[Array, " "] = relativistic_wavelength_ang(
        stem4d_data.voltage_kv
    )
    inner_angle_rad: Float[Array, " "] = (
        detector.collection_inner_mrad / 1000.0
    )
    outer_angle_rad: Float[Array, " "] = (
        detector.collection_outer_mrad / 1000.0
    )
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

    ny: int
    nx: int
    ny, nx = detector.scan_shape

    stem_image_2d: Float[Array, " ny nx"] = integrated_intensities.reshape(
        ny, nx
    )

    stem_image: CalibratedArray = create_calibrated_array(
        data_array=stem_image_2d,
        calib_y=stem4d_data.real_space_calib,
        calib_x=stem4d_data.real_space_calib,
        real_space=True,
    )

    return stem_image


__all__: list[str] = [
    "aberration",
    "annular_detector",
    "cbed_amplitude",
    "cbed_image",
    "decompose_beam_to_modes",
    "fourier_calib",
    "fourier_coords",
    "make_probe",
    "probe_modes_to_distribution",
    "propagation_func",
    "shift_beam_fourier",
    "stem_4d",
    "transmission_func",
]
