"""
Module: ptyrodactyl.photons.elements
------------------------------------
Common optical elements beyond lenses and basic apertures.

Functions
---------
- `prism_phase_ramp`:
    Applies a linear phase ramp to simulate beam deviation/dispersion.
- `apply_beam_splitter`:
    Splits a field into transmitted and reflected arms with given (t, r).
- `mirror_reflect`:
    Applies mirror reflection(s): coordinate flip(s), optional conjugation, π phase.
- `phase_grating_sine`:
    Sinusoidal phase grating.
- `amplitude_grating_binary`:
    Binary amplitude grating with duty cycle.
- `phase_grating_blazed`:
    Blazed (sawtooth) phase grating.
- `apply_phase_mask`:
    Applies an arbitrary phase mask (SLM / phase screen).
- `apply_phase_mask_fn`:
    Builds a phase mask from a callable f(X, Y) and applies it.
- `polarizer_jones`:
    Linear polarizer at angle theta (Jones matrix) for 2-component fields.
- `waveplate_jones`:
    Waveplate (retarder) with retardance delta and fast axis angle theta.
- `nd_filter`:
    Neutral density filter with optical density (OD) or direct transmittance.
- `quarter_waveplate`:
    Quarter-waveplate with fast axis angle theta.
- `half_waveplate`:
    Half-waveplate with fast axis angle theta.
- `phase_grating_blazed_elliptical`:
    Elliptical blazed phase grating with period_x, period_y, theta, depth, and two_dim.

Internal utilities
------------------
- `_xy_grids`:
    Builds centered (x, y) grids.
- `_rotate_coords`:
    Rotates coordinates by an angle theta.
"""

import jax
import jax.numpy as jnp
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Float

from ptyrodactyl._decorators import beartype, jaxtyped

from .helper import add_phase_screen
from .photon_types import OpticalWavefront, make_optical_wavefront, scalar_float

jax.config.update("jax_enable_x64", True)


def _xy_grids(
    nx: int, ny: int, dx: float
) -> Tuple[Float[Array, " H W"], Float[Array, " H W"]]:
    x: Float[Array, " W"] = jnp.arange(-nx // 2, nx // 2) * dx
    y: Float[Array, " H"] = jnp.arange(-ny // 2, ny // 2) * dx
    X: Float[Array, "H W"]
    Y: Float[Array, "H W"]
    X, Y = jnp.meshgrid(x, y)
    return X, Y


def _rotate_coords(X: Array, Y: Array, theta: float) -> Tuple[Array, Array]:
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    U = ct * X + st * Y
    V = -st * X + ct * Y
    return U, V


@jaxtyped(typechecker=beartype)
def prism_phase_ramp(
    incoming: OpticalWavefront,
    deflect_x: Optional[scalar_float] = 0.0,
    deflect_y: Optional[scalar_float] = 0.0,
    use_small_angle: bool = True,
) -> OpticalWavefront:
    """
    Apply a linear phase ramp to simulate a prism-induced beam deviation.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input scalar wavefront
    deflect_x : scalar_float, optional
        Deflection along +x. If `use_small_angle` is True, interpreted as angle (rad).
        Otherwise interpreted as spatial frequency kx [rad/m], by default 0.0
    deflect_y : scalar_float, optional
        Deflection along +y (angle or ky), by default 0.0
    use_small_angle : bool, optional
        If True, convert small angles to kx, ky via k*sin(angle) ~ k*angle, by default True

    Returns
    -------
    OpticalWavefront
        Wavefront with added linear phase

    Notes
    -----
    - Build X, Y grids (m)
    - Compute kx, ky from deflections
    - Phase = kx*X + ky*Y; multiply by exp(i*phase)
    """
    ny, nx = incoming.field.shape[:2]
    X, Y = _xy_grids(nx, ny, float(incoming.dx))
    k = (2.0 * jnp.pi) / incoming.wavelength

    if use_small_angle:
        kx = k * deflect_x
        ky = k * deflect_y
    else:
        kx = deflect_x
        ky = deflect_y

    phase = kx * X + ky * Y
    field_out = add_phase_screen(incoming.field, phase)

    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def apply_beam_splitter(
    incoming: OpticalWavefront,
    t2: Optional[scalar_float] = 0.5,
    r2: Optional[scalar_float] = 0.5,
    normalize: Optional[bool] = True,
) -> Tuple[OpticalWavefront, OpticalWavefront]:
    """
    Split an input field into transmitted and reflected components.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront (scalar field)
    t2 : scalar_float, optional
        Complex transmission amplitude, by default jnp.sqrt(0.5)
    r2 : scalar_float, optional
        Complex reflection amplitude, by default 1j * jnp.sqrt(0.5) for 50/50 convention
    normalize : bool, optional
        If True, scale (t, r) so that |t|^2 + |r|^2 = 1, by default True

    Returns
    -------
    wf_T : OpticalWavefront
        Transmitted arm (t * field)
    wf_R : OpticalWavefront
        Reflected arm (r * field)

    Notes
    -----
    - Optionally renormalize (t, r)
    - Multiply field by t and r
    - Return two wavefronts sharing same metadata
    """
    t = jnp.sqrt(t2)
    r = 1j * jnp.sqrt(r2)
    t_val = jnp.asarray(t, dtype=incoming.field.dtype)
    r_val = jnp.asarray(r, dtype=incoming.field.dtype)
    if normalize:
        power = jnp.abs(t_val) ** 2 + jnp.abs(r_val) ** 2
        # Avoid division by zero
        t_val = t_val / jnp.sqrt(jnp.maximum(power, 1e-20))
        r_val = r_val / jnp.sqrt(jnp.maximum(power, 1e-20))

    wf_t = make_optical_wavefront(
        field=incoming.field * t_val,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    wf_r = make_optical_wavefront(
        field=incoming.field * r_val,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
    return (wf_t, wf_r)


@jaxtyped(typechecker=beartype)
def mirror_reflect(
    incoming: OpticalWavefront,
    flip_x: bool = True,
    flip_y: bool = False,
    add_pi_phase: bool = True,
    conjugate: bool = True,
) -> OpticalWavefront:
    """
    Mirror reflection: coordinate flips with optional π-phase and conjugation.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input wavefront
    flip_x : bool, optional
        Flip along x-axis (columns), by default True
    flip_y : bool, optional
        Flip along y-axis (rows), by default False
    add_pi_phase : bool, optional
        Multiply by exp(i*pi) = -1 to simulate phase inversion on reflection, by default True
    conjugate : bool, optional
        Conjugate the complex field (useful when reversing propagation direction), by default True

    Returns
    -------
    OpticalWavefront
        Reflected wavefront

    Notes
    -----
    - Flip axes as requested (jnp.flip)
    - Optional complex conjugation
    - Optional -1 factor for π phase
    """
    field = incoming.field
    if flip_x:
        field = jnp.flip(field, axis=-1)
    if flip_y:
        field = jnp.flip(field, axis=-2)
    if conjugate:
        field = jnp.conjugate(field)
    if add_pi_phase:
        field = -field

    return make_optical_wavefront(
        field=field,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def phase_grating_sine(
    incoming: OpticalWavefront,
    period: scalar_float,
    depth: scalar_float,
    theta: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Sinusoidal phase grating: phase = depth * sin(2π * u / period),
    where u is the coordinate along the grating direction.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field
    period : scalar_float
        Grating period in meters
    depth : scalar_float
        Phase modulation depth in radians
    theta : scalar_float, optional
        Grating orientation (radians, CCW from x), by default 0.0

    Returns
    -------
    OpticalWavefront
        Field after phase modulation
    """
    ny, nx = incoming.field.shape[:2]
    X, Y = _xy_grids(nx, ny, float(incoming.dx))
    U, _ = _rotate_coords(X, Y, theta)
    phase = depth * jnp.sin(2.0 * jnp.pi * U / period)
    field_out = add_phase_screen(incoming.field, phase)
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def amplitude_grating_binary(
    incoming: OpticalWavefront,
    period: scalar_float,
    duty_cycle: scalar_float = 0.5,
    theta: scalar_float = 0.0,
    trans_high: scalar_float = 1.0,
    trans_low: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Binary amplitude grating with given duty cycle.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field
    period : scalar_float
        Period in meters
    duty_cycle : scalar_float, optional
        Fraction of period in 'high' state (0..1), by default 0.5
    theta : scalar_float, optional
        Orientation (radians), by default 0.0
    trans_high : scalar_float, optional
        Amplitude transmittance for 'high' bars, by default 1.0
    trans_low : scalar_float, optional
        Amplitude transmittance for 'low' bars, by default 0.0

    Returns
    -------
    OpticalWavefront
        Field after amplitude modulation

    Notes
    -----
    - Compute u along grating direction
    - Map u modulo period → binary mask via duty cycle
    - Apply amplitude levels to field
    """
    ny, nx = incoming.field.shape[:2]
    X, Y = _xy_grids(nx, ny, float(incoming.dx))
    U, _ = _rotate_coords(X, Y, theta)

    duty = jnp.clip(duty_cycle, 0.0, 1.0)
    frac = (U / period) - jnp.floor(U / period)
    mask_high: Bool[Array, " H W"] = frac < duty
    tmap = jnp.where(mask_high, trans_high, trans_low).astype(incoming.field.real.dtype)
    field_out = incoming.field * tmap

    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def phase_grating_blazed(
    incoming: OpticalWavefront,
    period: scalar_float,
    depth: scalar_float,
    theta: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Blazed (sawtooth) phase grating with peak-to-peak depth (radians).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field
    period : scalar_float
        Grating period in meters
    depth : scalar_float
        Phase depth over one period in radians
    theta : scalar_float, optional
        Orientation (radians), by default 0.0

    Returns
    -------
    OpticalWavefront
        Field after blazed phase modulation

    Notes
    -----
    - Compute fractional coordinate within each period
    - Sawtooth phase in [0, depth) → shift to mean-zero if desired (kept at [0, depth))
    - Apply phase with exp(i*phase)
    """
    ny, nx = incoming.field.shape[:2]
    X, Y = _xy_grids(nx, ny, float(incoming.dx))
    U, _ = _rotate_coords(X, Y, theta)

    frac = (U / period) - jnp.floor(U / period)  # [0,1)
    phase = depth * frac
    field_out = add_phase_screen(incoming.field, phase)

    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def apply_phase_mask(
    incoming: OpticalWavefront,
    phase_map: Float[Array, " H W"],
) -> OpticalWavefront:
    """
    Apply an arbitrary phase mask (e.g., SLM, turbulence screen):
    field_out = field_in * exp(i * phase_map).

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field
    phase_map : Float[Array, " H W"]
        Phase in radians, same spatial shape as field

    Returns
    -------
    OpticalWavefront
        Field with added phase
    """
    field_out = add_phase_screen(incoming.field, phase_map)
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def apply_phase_mask_fn(
    incoming: OpticalWavefront,
    phase_fn,  # Callable[[X, Y], Array["H W"]]
) -> OpticalWavefront:
    """
    Build and apply a phase mask from a callable `phase_fn(X, Y)`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field
    phase_fn : callable
        Function producing a phase map (radians) given centered grids X, Y (meters)

    Returns
    -------
    OpticalWavefront
        Field with added phase
    """
    ny, nx = incoming.field.shape[:2]
    X, Y = _xy_grids(nx, ny, float(incoming.dx))
    phase_map = phase_fn(X, Y)
    field_out = add_phase_screen(incoming.field, phase_map)
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def polarizer_jones(
    incoming: OpticalWavefront,
    theta: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Linear polarizer at angle `theta` (radians, CCW from x-axis) applied to a
    2-component Jones field (Ex, Ey) stored in the last dimension.

    Parameters
    ----------
    incoming : OpticalWavefront
        Field shape must be Complex[H, W, 2]
    theta : scalar_float, optional
        Transmission axis angle (radians), by default 0.0

    Returns
    -------
    OpticalWavefront
        Polarized field with same shape

    Notes
    -----
    - Jones matrix: P = R(-θ) @ [[1, 0],[0, 0]] @ R(θ)
    - Apply P to [Ex, Ey] at each pixel
    """
    field = incoming.field
    assert (
        field.ndim == 3 and field.shape[-1] == 2
    ), "polarizer_jones expects field[...,2]"
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    # Projection onto axis θ: [ct, st]^T [ct, st]
    Ex, Ey = field[..., 0], field[..., 1]
    E_par = Ex * ct + Ey * st
    Ex_out = E_par * ct
    Ey_out = E_par * st
    field_out = jnp.stack([Ex_out, Ey_out], axis=-1)
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def waveplate_jones(
    incoming: OpticalWavefront,
    delta: scalar_float,
    theta: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Waveplate/retarder with retardance `delta` (radians) and fast-axis angle `theta`.

    Special cases: quarter-wave (delta=π/2), half-wave (delta=π).

    Parameters
    ----------
    incoming : OpticalWavefront
        Field shape must be Complex[H, W, 2]
    delta : scalar_float
        Phase delay between fast and slow axes in radians
    theta : scalar_float, optional
        Fast-axis angle (radians, CCW from x), by default 0.0

    Returns
    -------
    OpticalWavefront
        Retarded field with same shape

    Notes
    -----
    - Jones matrix: J = R(-θ) @ diag(1, e^{iδ}) @ R(θ)
    - Apply J to [Ex, Ey] per pixel
    """
    field = incoming.field
    assert (
        field.ndim == 3 and field.shape[-1] == 2
    ), "waveplate_jones expects field[...,2]"

    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    e = jnp.exp(1j * delta)

    # Expanded multiplication of J with [Ex, Ey]
    # J = [[ct^2 + e*st^2, (1 - e)*ct*st],
    #      [(1 - e)*ct*st, st^2 + e*ct^2]]
    Ex, Ey = field[..., 0], field[..., 1]
    a = ct * ct + e * st * st
    b = (1.0 - e) * ct * st
    c = b
    d = st * st + e * ct * ct

    Ex_out = a * Ex + b * Ey
    Ey_out = c * Ex + d * Ey
    field_out = jnp.stack([Ex_out, Ey_out], axis=-1)

    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def nd_filter(
    incoming: OpticalWavefront,
    optical_density: Optional[scalar_float] = None,
    transmittance: Optional[scalar_float] = None,
) -> OpticalWavefront:
    """
    Neutral density (ND) filter as a uniform amplitude attenuator.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input field
    optical_density : Optional[scalar_float], optional
        OD; intensity transmittance T = 10^(-OD). If given, overrides `transmittance`
    transmittance : Optional[scalar_float], optional
        Intensity transmittance T in [0, 1]. Used if `optical_density` is None

    Returns
    -------
    OpticalWavefront
        Attenuated wavefront

    Notes
    -----
    - Determine intensity T from OD or provided T
    - Amplitude factor a = sqrt(T)
    - Multiply field by a and return
    """
    if optical_density is not None:
        T = jnp.power(10.0, -jnp.asarray(optical_density))
    else:
        T = jnp.clip(
            jnp.asarray(transmittance if transmittance is not None else 1.0), 0.0, 1.0
        )

    a = jnp.sqrt(T).astype(incoming.field.real.dtype)
    field_out = incoming.field * a

    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )


@jaxtyped(typechecker=beartype)
def quarter_waveplate(
    incoming: OpticalWavefront,
    theta: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Convenience wrapper for a quarter-wave plate (δ = π/2) with fast-axis angle `theta`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Vector field Complex[H, W, 2] (Jones: Ex, Ey)
    theta : scalar_float, optional
        Fast-axis angle in radians (CCW from x), by default 0.0

    Returns
    -------
    OpticalWavefront
        Retarded field after quarter-wave plate

    Notes
    -----
    Call `waveplate_jones` with delta = π/2.
    """
    return waveplate_jones(incoming, delta=jnp.pi / 2.0, theta=theta)


@jaxtyped(typechecker=beartype)
def half_waveplate(
    incoming: OpticalWavefront,
    theta: scalar_float = 0.0,
) -> OpticalWavefront:
    """
    Convenience wrapper for a half-wave plate (δ = π) with fast-axis angle `theta`.

    Parameters
    ----------
    incoming : OpticalWavefront
        Vector field Complex[H, W, 2] (Jones: Ex, Ey)
    theta : scalar_float, optional
        Fast-axis angle in radians (CCW from x), by default 0.0

    Returns
    -------
    OpticalWavefront
        Retarded field after half-wave plate

    Notes
    -----
    Call `waveplate_jones` with delta = π.
    """
    return waveplate_jones(incoming, delta=jnp.pi, theta=theta)


@jaxtyped(typechecker=beartype)
def phase_grating_blazed_elliptical(
    incoming: OpticalWavefront,
    period_x: scalar_float,
    period_y: scalar_float,
    theta: scalar_float = 0.0,
    depth: scalar_float = 2.0 * jnp.pi,
    two_dim: bool = False,
) -> OpticalWavefront:
    """
    Orientation-aware elliptical blazed grating.

    Supports anisotropic periods along rotated axes (x', y') and optional 2D blaze.

    Parameters
    ----------
    incoming : OpticalWavefront
        Input scalar wavefront
    period_x : scalar_float
        Blaze period along x' in meters (after rotation by `theta`)
    period_y : scalar_float
        Blaze period along y' in meters (after rotation by `theta`)
    theta : scalar_float, optional
        Grating orientation angle in radians (CCW from x), by default 0.0
    depth : scalar_float, optional
        Peak-to-peak phase depth in radians, by default 2π
    two_dim : bool, optional
        If False (default), apply a 1D blaze along x' only.
        If True, create a 2D blazed lattice using both x' and y'

    Returns
    -------
    OpticalWavefront
        Field after applying the elliptical blazed phase

    Notes
    -----
    - Build centered grids X, Y (meters) and rotate → (x', y')
    - Compute fractional coordinates fu = frac(x'/period_x), fv = frac(y'/period_y)
    - If `two_dim`:
      phase = depth * frac(fu + fv); else phase = depth * fu
    - Multiply by exp(i * phase) and return
    """
    ny, nx = incoming.field.shape[:2]
    X, Y = _xy_grids(nx, ny, float(incoming.dx))
    U, V = _rotate_coords(X, Y, theta)

    # Avoid division by zero periods
    eps = 1e-30
    px = jnp.where(jnp.abs(period_x) < eps, eps, period_x)
    py = jnp.where(jnp.abs(period_y) < eps, eps, period_y)

    fu = (U / px) - jnp.floor(U / px)  # in [0,1)
    fv = (V / py) - jnp.floor(V / py)  # in [0,1)

    if two_dim:
        phase = depth * ((fu + fv) - jnp.floor(fu + fv))  # frac(fu+fv)
    else:
        phase = depth * fu

    field_out = add_phase_screen(incoming.field, phase)
    return make_optical_wavefront(
        field=field_out,
        wavelength=incoming.wavelength,
        dx=incoming.dx,
        z_position=incoming.z_position,
    )
