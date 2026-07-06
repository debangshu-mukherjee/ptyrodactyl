"""Define jacobian parameter and solver-state carriers.

Extended Summary
----------------
This module defines the canonical Equinox PyTree carriers for
ptychographic jacobian parameter blocks, Fisher-state accumulation, and
second-order solver state. The carriers mirror the legacy jacobian
NamedTuple field names and field order while making each field a dynamic
Equinox leaf.

Routine Listings
----------------
:class:`ExitWaveParams`
    Complex exit wave array.
:class:`AberrationParams`
    Zernike coefficients and soft aperture cutoff.
:class:`GeometryParams`
    Rotation angle, centre offset, ellipticity.
:class:`PositionParams`
    Per-scan-point position corrections.
:class:`ProbeModeParams`
    Probe mode weights and shapes.
:class:`PtychoParams`
    Combined parameter container for all blocks.
:class:`FisherState`
    State container for iterative Fisher computation.
:class:`CGState`
    State container for conjugate gradient iteration.
:class:`GNState`
    State container for Gauss-Newton iteration.
:class:`LMState`
    State container for Levenberg-Marquardt iteration.
:class:`LanczosState`
    State container for Lanczos tridiagonalisation.
:func:`create_ptycho_params`
    Construct combined PtychoParams from components.

Notes
-----
All fields are dynamic Equinox leaves. Iteration counters remain traced
rank-0 integer arrays so these states can be carried through
``jax.lax.scan``.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, Int, PyTree, jaxtyped

from .custom_types import scalar_float


def _raise_if(condition: bool, message: str) -> None:
    """Raise ValueError when a structural condition is true."""
    if condition:
        raise ValueError(message)


class ExitWaveParams(eqx.Module):
    """Store complex exit-wave parameters.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    wave : Complex[Array, "h w"]
        Complex-valued exit wave in real space.
    """

    wave: Complex[Array, "h w"]


class AberrationParams(eqx.Module):
    """Store probe aberration parameters.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    zernike_coeffs : Float[Array, "num_zernike"]
        Coefficients for Zernike polynomial expansion.
    aperture_mrad : Float[Array, ""]
        Soft aperture cutoff in milliradians.
    aperture_softness : Float[Array, ""]
        Softness parameter for aperture roll-off, dimensionless.
    """

    zernike_coeffs: Float[Array, "num_zernike"]
    aperture_mrad: Float[Array, ""]
    aperture_softness: Float[Array, ""]


class GeometryParams(eqx.Module):
    """Store geometric calibration parameters.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    rotation_rad : Float[Array, ""]
        Rotation angle around the optic axis in radians.
    center_offset : Float[Array, "2"]
        Offset of pattern centre (cx, cy) in pixels.
    ellipticity : Float[Array, "2"]
        Elliptical distortion parameters (e1, e2), dimensionless.
    """

    rotation_rad: Float[Array, ""]
    center_offset: Float[Array, "2"]
    ellipticity: Float[Array, "2"]


class PositionParams(eqx.Module):
    """Store scan position error parameters.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    position_offsets : Float[Array, "num_positions 2"]
        Per-scan-point position corrections (dx, dy) in Angstroms.
    """

    position_offsets: Float[Array, "num_positions 2"]


class ProbeModeParams(eqx.Module):
    """Store probe mode parameters for partial coherence.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    mode_weights : Float[Array, "num_modes"]
        Relative weights for each probe mode, dimensionless.
    mode_phases : Float[Array, "num_modes h w"]
        Phase perturbations for each mode relative to the base probe,
        in radians.
    """

    mode_weights: Float[Array, "num_modes"]
    mode_phases: Float[Array, "num_modes h w"]


class PtychoParams(eqx.Module):
    """Store all ptychographic parameter blocks.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    exit_wave : ExitWaveParams
        Exit wave parameters.
    aberrations : AberrationParams
        Probe aberration parameters.
    geometry : GeometryParams
        Geometric calibration parameters.
    positions : PositionParams
        Scan position error parameters.
    probe_modes : ProbeModeParams
        Probe mode parameters.
    """

    exit_wave: ExitWaveParams
    aberrations: AberrationParams
    geometry: GeometryParams
    positions: PositionParams
    probe_modes: ProbeModeParams


class FisherState(eqx.Module):
    """Store state for iterative Fisher computation.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    fisher_matrix : Float[Array, "n n"]
        Current Fisher information matrix estimate.
    iteration : Int[Array, ""]
        Current iteration index.
    """

    fisher_matrix: Float[Array, "n n"]
    iteration: Int[Array, ""]


class CGState(eqx.Module):
    """Store conjugate gradient iteration state.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    x : PyTree
        Current solution estimate.
    r : PyTree
        Current residual b - A x.
    p : PyTree
        Current search direction.
    r_dot_r : Float[Array, ""]
        Squared residual norm <r, r>.
    iteration : Int[Array, ""]
        Current iteration index.
    """

    x: PyTree
    r: PyTree
    p: PyTree
    r_dot_r: Float[Array, ""]
    iteration: Int[Array, ""]


class GNState(eqx.Module):
    """Store Gauss-Newton iteration state.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    params : PyTree
        Current parameter estimate.
    residual_norm : Float[Array, ""]
        L2 norm of the current residual.
    iteration : Int[Array, ""]
        Current iteration index.
    """

    params: PyTree
    residual_norm: Float[Array, ""]
    iteration: Int[Array, ""]


class LMState(eqx.Module):
    r"""Store Levenberg-Marquardt iteration state.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    params : PyTree
        Current parameter estimate.
    residual_norm : Float[Array, ""]
        L2 norm of the current residual.
    damping : Float[Array, ""]
        Current damping parameter :math:`\lambda`.
    iteration : Int[Array, ""]
        Current iteration index.
    """

    params: PyTree
    residual_norm: Float[Array, ""]
    damping: Float[Array, ""]
    iteration: Int[Array, ""]


class LanczosState(eqx.Module):
    """Store Lanczos tridiagonalisation state.

    :see: :class:`~.test_jacobian_types.TestJacobianCarriers`

    Attributes
    ----------
    v_prev : Float[Array, "n"]
        Previous Lanczos vector.
    v_curr : Float[Array, "n"]
        Current Lanczos vector.
    alpha : Float[Array, "k"]
        Diagonal elements accumulated so far.
    beta : Float[Array, "k"]
        Off-diagonal elements accumulated so far.
    iteration : Int[Array, ""]
        Current iteration index.
    """

    v_prev: Float[Array, "n"]
    v_curr: Float[Array, "n"]
    alpha: Float[Array, "k"]
    beta: Float[Array, "k"]
    iteration: Int[Array, ""]


@jaxtyped(typechecker=beartype)
def create_ptycho_params(
    exit_wave: Complex[Array, "..."],
    zernike_coeffs: Float[Array, "..."],
    aperture_mrad: scalar_float,
    aperture_softness: scalar_float,
    rotation_rad: scalar_float,
    center_offset: Float[Array, "..."],
    ellipticity: Float[Array, "..."],
    position_offsets: Float[Array, "..."],
    mode_weights: Float[Array, "..."],
    mode_phases: Float[Array, "..."],
) -> PtychoParams:
    """Construct combined PtychoParams from components.

    :see: :class:`~.test_jacobian_types.TestCreatePtychoParams`

    Parameters
    ----------
    exit_wave : Complex[Array, "..."]
        Complex exit wave array. Must have shape ``(h, w)``.
    zernike_coeffs : Float[Array, "..."]
        Zernike polynomial coefficients. Must have shape ``(num_zernike,)``.
    aperture_mrad : scalar_float
        Soft aperture cutoff in milliradians.
    aperture_softness : scalar_float
        Aperture roll-off softness, dimensionless.
    rotation_rad : scalar_float
        Rotation angle in radians.
    center_offset : Float[Array, "..."]
        Pattern centre offset in pixels. Must have shape ``(2,)``.
    ellipticity : Float[Array, "..."]
        Elliptical distortion parameters. Must have shape ``(2,)``.
    position_offsets : Float[Array, "..."]
        Per-position corrections in Angstroms. Must have shape ``(P, 2)``.
    mode_weights : Float[Array, "..."]
        Probe mode weights, dimensionless. Must have shape ``(M,)``.
    mode_phases : Float[Array, "..."]
        Probe mode phase perturbations in radians. Must have shape
        ``(M, h, w)``.

    Returns
    -------
    params : PtychoParams
        Validated combined parameter container.

    Raises
    ------
    ValueError
        If ranks, shapes, or scalar structures are invalid.

    Notes
    -----
    1. Convert inputs to JAX arrays without changing array dtypes.
    2. Validate static ranks and shape compatibility with ``ValueError``.
    3. Require finite values, positive aperture scalars, and normalized
       non-negative mode weights with traced ``eqx.error_if`` checks.
    4. Create and return a ``PtychoParams`` with dynamic Equinox leaves.
    """
    exit_wave_arr: Complex[Array, "h w"] = jnp.asarray(exit_wave)
    zernike_coeffs_arr: Float[Array, "num_zernike"] = jnp.asarray(
        zernike_coeffs
    )
    aperture_mrad_arr: Float[Array, ""] = jnp.asarray(aperture_mrad)
    aperture_softness_arr: Float[Array, ""] = jnp.asarray(aperture_softness)
    rotation_rad_arr: Float[Array, ""] = jnp.asarray(rotation_rad)
    center_offset_arr: Float[Array, "2"] = jnp.asarray(center_offset)
    ellipticity_arr: Float[Array, "2"] = jnp.asarray(ellipticity)
    position_offsets_arr: Float[Array, "num_positions 2"] = jnp.asarray(
        position_offsets
    )
    mode_weights_arr: Float[Array, "num_modes"] = jnp.asarray(mode_weights)
    mode_phases_arr: Float[Array, "num_modes h w"] = jnp.asarray(
        mode_phases
    )

    image_rank: int = 2
    vector_rank: int = 1
    mode_phase_rank: int = 3
    num_xy: int = 2
    scalar_shape: tuple[()] = ()
    _raise_if(exit_wave_arr.ndim != image_rank, "exit_wave must be 2D")
    _raise_if(
        zernike_coeffs_arr.ndim != vector_rank,
        "zernike_coeffs must be 1D",
    )
    _raise_if(
        aperture_mrad_arr.shape != scalar_shape,
        "aperture_mrad must be a scalar",
    )
    _raise_if(
        aperture_softness_arr.shape != scalar_shape,
        "aperture_softness must be a scalar",
    )
    _raise_if(
        rotation_rad_arr.shape != scalar_shape,
        "rotation_rad must be a scalar",
    )
    _raise_if(
        center_offset_arr.shape != (num_xy,),
        "center_offset must have shape (2,)",
    )
    _raise_if(
        ellipticity_arr.shape != (num_xy,),
        "ellipticity must have shape (2,)",
    )
    _raise_if(
        position_offsets_arr.ndim != image_rank,
        "position_offsets must be 2D",
    )
    _raise_if(
        position_offsets_arr.shape[1] != num_xy,
        "position_offsets must have shape (P, 2)",
    )
    _raise_if(mode_weights_arr.ndim != vector_rank, "mode_weights must be 1D")
    _raise_if(
        mode_phases_arr.ndim != mode_phase_rank,
        "mode_phases must be 3D",
    )
    _raise_if(
        mode_phases_arr.shape[0] != mode_weights_arr.shape[0],
        "mode_phases must have shape (M, h, w)",
    )
    _raise_if(
        mode_phases_arr.shape[1:] != exit_wave_arr.shape,
        "mode_phases spatial shape must match exit_wave",
    )

    checked_exit_wave: Complex[Array, "h w"] = eqx.error_if(
        exit_wave_arr,
        jnp.any(~jnp.isfinite(exit_wave_arr)),
        "exit_wave contains non-finite values",
    )
    checked_zernike_coeffs: Float[Array, "num_zernike"] = eqx.error_if(
        zernike_coeffs_arr,
        jnp.any(~jnp.isfinite(zernike_coeffs_arr)),
        "zernike_coeffs contain non-finite values",
    )
    checked_aperture_mrad: Float[Array, ""] = eqx.error_if(
        aperture_mrad_arr,
        ~jnp.isfinite(aperture_mrad_arr),
        "aperture_mrad must be finite",
    )
    checked_aperture_mrad = eqx.error_if(
        checked_aperture_mrad,
        checked_aperture_mrad <= 0,
        "aperture_mrad must be positive",
    )
    checked_aperture_softness: Float[Array, ""] = eqx.error_if(
        aperture_softness_arr,
        ~jnp.isfinite(aperture_softness_arr),
        "aperture_softness must be finite",
    )
    checked_aperture_softness = eqx.error_if(
        checked_aperture_softness,
        checked_aperture_softness <= 0,
        "aperture_softness must be positive",
    )
    checked_rotation_rad: Float[Array, ""] = eqx.error_if(
        rotation_rad_arr,
        ~jnp.isfinite(rotation_rad_arr),
        "rotation_rad must be finite",
    )
    checked_center_offset: Float[Array, "2"] = eqx.error_if(
        center_offset_arr,
        jnp.any(~jnp.isfinite(center_offset_arr)),
        "center_offset contains non-finite values",
    )
    checked_ellipticity: Float[Array, "2"] = eqx.error_if(
        ellipticity_arr,
        jnp.any(~jnp.isfinite(ellipticity_arr)),
        "ellipticity contains non-finite values",
    )
    checked_position_offsets: Float[Array, "num_positions 2"] = eqx.error_if(
        position_offsets_arr,
        jnp.any(~jnp.isfinite(position_offsets_arr)),
        "position_offsets contain non-finite values",
    )
    checked_mode_weights: Float[Array, "num_modes"] = eqx.error_if(
        mode_weights_arr,
        jnp.any(~jnp.isfinite(mode_weights_arr)),
        "mode_weights contain non-finite values",
    )
    checked_mode_weights = eqx.error_if(
        checked_mode_weights,
        jnp.any(checked_mode_weights < 0),
        "mode_weights must be non-negative",
    )
    weight_sum: Float[Array, ""] = jnp.sum(checked_mode_weights)
    checked_mode_weights = eqx.error_if(
        checked_mode_weights,
        ~jnp.isclose(weight_sum, jnp.array(1.0, dtype=weight_sum.dtype)),
        "mode_weights must sum to one",
    )
    checked_mode_phases: Float[Array, "num_modes h w"] = eqx.error_if(
        mode_phases_arr,
        jnp.any(~jnp.isfinite(mode_phases_arr)),
        "mode_phases contain non-finite values",
    )

    exit_wave_params: ExitWaveParams = ExitWaveParams(wave=checked_exit_wave)
    aberration_params: AberrationParams = AberrationParams(
        zernike_coeffs=checked_zernike_coeffs,
        aperture_mrad=checked_aperture_mrad,
        aperture_softness=checked_aperture_softness,
    )
    geometry_params: GeometryParams = GeometryParams(
        rotation_rad=checked_rotation_rad,
        center_offset=checked_center_offset,
        ellipticity=checked_ellipticity,
    )
    position_params: PositionParams = PositionParams(
        position_offsets=checked_position_offsets,
    )
    probe_mode_params: ProbeModeParams = ProbeModeParams(
        mode_weights=checked_mode_weights,
        mode_phases=checked_mode_phases,
    )
    params: PtychoParams = PtychoParams(
        exit_wave=exit_wave_params,
        aberrations=aberration_params,
        geometry=geometry_params,
        positions=position_params,
        probe_modes=probe_mode_params,
    )
    return params


__all__: list[str] = [
    "AberrationParams",
    "CGState",
    "ExitWaveParams",
    "FisherState",
    "GNState",
    "GeometryParams",
    "LMState",
    "LanczosState",
    "PositionParams",
    "ProbeModeParams",
    "PtychoParams",
    "create_ptycho_params",
]
