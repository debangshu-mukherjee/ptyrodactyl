"""
Module: ptyrodactyl.jacobian.blocks
-----------------------------------

Block-structured parameter management for ptychographic reconstruction.

Ptychography involves multiple parameter groups with different physical
meanings and observability characteristics. This module provides PyTree
structures for organizing parameters into blocks and computing block-wise
Jacobians for Schur complement marginalization and targeted updates.

Parameter Blocks
----------------
1. Exit wave: Complex-valued specimen exit wave ψ(x,y)
2. Aberrations: Zernike polynomial coefficients + soft aperture
3. Geometry: Rotation, centering, ellipticity calibration
4. Positions: Per-scan-point position errors (dx, dy)
5. Probe modes: Modal decomposition for partial coherence

PyTrees
-------
- `ExitWaveParams`:
    Complex exit wave array
- `AberrationParams`:
    Zernike coefficients and soft aperture cutoff
- `GeometryParams`:
    Rotation angle, center offset, ellipticity
- `PositionParams`:
    Per-scan-point position corrections
- `ProbeModeParams`:
    Probe mode weights and shapes
- `PtychoParams`:
    Combined parameter container for all blocks

Functions
---------
- `make_ptycho_params`:
    Construct combined parameter PyTree from components
- `split_params`:
    Extract individual blocks from combined params
- `block_jacobian`:
    Compute Jacobian for a single parameter block
- `block_jtj`:
    Compute JᵀJ for a single parameter block
- `full_jacobian_blocks`:
    Compute all block Jacobians simultaneously
- `marginalize_nuisances`:
    Schur complement to get effective Fisher for exit wave
- `block_gauss_newton_step`:
    GN step updating only specified blocks
"""

from typing import Callable, Tuple, NamedTuple, List, Optional
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Float, Complex, Int, Array, PyTree


class ExitWaveParams(NamedTuple):
    """
    Exit wave parameters.
    
    Attributes
    ----------
    - `wave` (Complex[Array, "h w"]):
        Complex-valued exit wave in real space.
    """
    wave: Complex[Array, "h w"]


class AberrationParams(NamedTuple):
    """
    Probe aberration parameters.
    
    Attributes
    ----------
    - `zernike_coeffs` (Float[Array, "num_zernike"]):
        Coefficients for Zernike polynomial expansion.
        Ordering follows Noll convention.
    - `aperture_mrad` (Float[Array, ""]):
        Soft aperture cutoff in milliradians.
    - `aperture_softness` (Float[Array, ""]):
        Softness parameter for aperture roll-off.
    """
    zernike_coeffs: Float[Array, "num_zernike"]
    aperture_mrad: Float[Array, ""]
    aperture_softness: Float[Array, ""]


class GeometryParams(NamedTuple):
    """
    Geometric calibration parameters.
    
    Attributes
    ----------
    - `rotation_rad` (Float[Array, ""]):
        Rotation angle around optic axis in radians.
    - `center_offset` (Float[Array, "2"]):
        Offset of pattern center (cx, cy) in pixels.
    - `ellipticity` (Float[Array, "2"]):
        Elliptical distortion parameters (e1, e2).
    """
    rotation_rad: Float[Array, ""]
    center_offset: Float[Array, "2"]
    ellipticity: Float[Array, "2"]


class PositionParams(NamedTuple):
    """
    Scan position error parameters.
    
    Attributes
    ----------
    - `position_offsets` (Float[Array, "num_positions 2"]):
        Per-scan-point position corrections (dx, dy).
    """
    position_offsets: Float[Array, "num_positions 2"]


class ProbeModeParams(NamedTuple):
    """
    Probe mode parameters for partial coherence.
    
    Attributes
    ----------
    - `mode_weights` (Float[Array, "num_modes"]):
        Relative weights for each probe mode (sum to 1).
    - `mode_phases` (Float[Array, "num_modes h w"]):
        Phase perturbations for each mode relative to base probe.
    """
    mode_weights: Float[Array, "num_modes"]
    mode_phases: Float[Array, "num_modes h w"]


class PtychoParams(NamedTuple):
    """
    Combined parameter container for all ptychography blocks.
    
    Attributes
    ----------
    - `exit_wave` (ExitWaveParams):
        Exit wave parameters.
    - `aberrations` (AberrationParams):
        Probe aberration parameters.
    - `geometry` (GeometryParams):
        Geometric calibration parameters.
    - `positions` (PositionParams):
        Scan position error parameters.
    - `probe_modes` (ProbeModeParams):
        Probe mode parameters.
    """
    exit_wave: ExitWaveParams
    aberrations: AberrationParams
    geometry: GeometryParams
    positions: PositionParams
    probe_modes: ProbeModeParams


def make_ptycho_params(
    exit_wave: Complex[Array, "h w"],
    zernike_coeffs: Float[Array, "num_zernike"],
    aperture_mrad: float,
    aperture_softness: float,
    rotation_rad: float,
    center_offset: Float[Array, "2"],
    ellipticity: Float[Array, "2"],
    position_offsets: Float[Array, "num_positions 2"],
    mode_weights: Float[Array, "num_modes"],
    mode_phases: Float[Array, "num_modes h w"],
) -> PtychoParams:
    """
    Description
    -----------
    Construct combined PtychoParams from individual components.

    Parameters
    ----------
    - `exit_wave` (Complex[Array, "h w"]):
        Complex exit wave array.
    - `zernike_coeffs` (Float[Array, "num_zernike"]):
        Zernike polynomial coefficients.
    - `aperture_mrad` (float):
        Soft aperture cutoff in milliradians.
    - `aperture_softness` (float):
        Aperture roll-off softness.
    - `rotation_rad` (float):
        Rotation angle in radians.
    - `center_offset` (Float[Array, "2"]):
        Pattern center offset.
    - `ellipticity` (Float[Array, "2"]):
        Elliptical distortion.
    - `position_offsets` (Float[Array, "num_positions 2"]):
        Per-position corrections.
    - `mode_weights` (Float[Array, "num_modes"]):
        Probe mode weights.
    - `mode_phases` (Float[Array, "num_modes h w"]):
        Probe mode phase perturbations.

    Returns
    -------
    - `params` (PtychoParams):
        Combined parameter container.
    """
    exit_wave_params: ExitWaveParams = ExitWaveParams(wave=exit_wave)
    
    aberration_params: AberrationParams = AberrationParams(
        zernike_coeffs=zernike_coeffs,
        aperture_mrad=jnp.array(aperture_mrad),
        aperture_softness=jnp.array(aperture_softness),
    )
    
    geometry_params: GeometryParams = GeometryParams(
        rotation_rad=jnp.array(rotation_rad),
        center_offset=center_offset,
        ellipticity=ellipticity,
    )
    
    position_params: PositionParams = PositionParams(
        position_offsets=position_offsets,
    )
    
    probe_mode_params: ProbeModeParams = ProbeModeParams(
        mode_weights=mode_weights,
        mode_phases=mode_phases,
    )
    
    params: PtychoParams = PtychoParams(
        exit_wave=exit_wave_params,
        aberrations=aberration_params,
        geometry=geometry_params,
        positions=position_params,
        probe_modes=probe_mode_params,
    )
    
    return params


def split_params(
    params: PtychoParams,
) -> Tuple[ExitWaveParams, AberrationParams, GeometryParams, PositionParams, ProbeModeParams]:
    """
    Description
    -----------
    Extract individual parameter blocks from combined params.

    Parameters
    ----------
    - `params` (PtychoParams):
        Combined parameter container.

    Returns
    -------
    - `exit_wave` (ExitWaveParams):
        Exit wave parameters.
    - `aberrations` (AberrationParams):
        Aberration parameters.
    - `geometry` (GeometryParams):
        Geometry parameters.
    - `positions` (PositionParams):
        Position parameters.
    - `probe_modes` (ProbeModeParams):
        Probe mode parameters.
    """
    return (
        params.exit_wave,
        params.aberrations,
        params.geometry,
        params.positions,
        params.probe_modes,
    )


def _tree_zeros_like(
    tree: PyTree,
) -> PyTree:
    """
    Description
    -----------
    Create PyTree of zeros matching input structure.

    Parameters
    ----------
    - `tree` (PyTree):
        Template PyTree.

    Returns
    -------
    - `zeros` (PyTree):
        PyTree of zeros with same structure.
    """
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def block_jacobian_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name: str,
) -> Callable[[PyTree], Float[Array, "num_pos det_h det_w"]]:
    """
    Description
    -----------
    Construct JVP operator for a single parameter block.

    Computes J_block @ v where J_block = ∂forward/∂block, treating
    all other blocks as fixed.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model mapping params to 4D-STEM datacube.
    - `params` (PtychoParams):
        Current parameter values.
    - `block_name` (str):
        Name of block: 'exit_wave', 'aberrations', 'geometry', 'positions', 'probe_modes'.

    Returns
    -------
    - `jvp_fn` (Callable[[PyTree], Float[Array, "num_pos det_h det_w"]]):
        Function computing J_block @ v for tangent vectors v in block space.

    Flow
    ----
    1. Create tangent PyTree with zeros everywhere except target block
    2. Compute JVP of full forward with this sparse tangent
    3. Return output tangent
    """
    def jvp_fn(
        block_tangent: PyTree,
    ) -> Float[Array, "num_pos det_h det_w"]:
        zero_exit: ExitWaveParams = _tree_zeros_like(params.exit_wave)
        zero_aberr: AberrationParams = _tree_zeros_like(params.aberrations)
        zero_geom: GeometryParams = _tree_zeros_like(params.geometry)
        zero_pos: PositionParams = _tree_zeros_like(params.positions)
        zero_modes: ProbeModeParams = _tree_zeros_like(params.probe_modes)
        
        tangent_exit: ExitWaveParams = zero_exit
        tangent_aberr: AberrationParams = zero_aberr
        tangent_geom: GeometryParams = zero_geom
        tangent_pos: PositionParams = zero_pos
        tangent_modes: ProbeModeParams = zero_modes
        
        if block_name == 'exit_wave':
            tangent_exit = block_tangent
        elif block_name == 'aberrations':
            tangent_aberr = block_tangent
        elif block_name == 'geometry':
            tangent_geom = block_tangent
        elif block_name == 'positions':
            tangent_pos = block_tangent
        elif block_name == 'probe_modes':
            tangent_modes = block_tangent
        
        full_tangent: PtychoParams = PtychoParams(
            exit_wave=tangent_exit,
            aberrations=tangent_aberr,
            geometry=tangent_geom,
            positions=tangent_pos,
            probe_modes=tangent_modes,
        )
        
        _, output_tangent = jax.jvp(forward_fn, (params,), (full_tangent,))
        return output_tangent
    
    return jvp_fn


def block_vjp_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name: str,
) -> Callable[[Float[Array, "num_pos det_h det_w"]], PyTree]:
    """
    Description
    -----------
    Construct VJP operator for a single parameter block.

    Computes J_blockᵀ @ u where J_block = ∂forward/∂block.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model mapping params to 4D-STEM datacube.
    - `params` (PtychoParams):
        Current parameter values.
    - `block_name` (str):
        Name of block: 'exit_wave', 'aberrations', 'geometry', 'positions', 'probe_modes'.

    Returns
    -------
    - `vjp_fn` (Callable[[Float[Array, "num_pos det_h det_w"]], PyTree]):
        Function computing J_blockᵀ @ u for cotangent vectors u.

    Flow
    ----
    1. Compute VJP of full forward
    2. Extract and return only the gradient for target block
    """
    _, full_vjp_fn = jax.vjp(forward_fn, params)
    
    def vjp_fn(
        cotangent: Float[Array, "num_pos det_h det_w"],
    ) -> PyTree:
        full_grad: Tuple[PtychoParams] = full_vjp_fn(cotangent)
        params_grad: PtychoParams = full_grad[0]
        
        if block_name == 'exit_wave':
            return params_grad.exit_wave
        elif block_name == 'aberrations':
            return params_grad.aberrations
        elif block_name == 'geometry':
            return params_grad.geometry
        elif block_name == 'positions':
            return params_grad.positions
        elif block_name == 'probe_modes':
            return params_grad.probe_modes
    
    return vjp_fn


def block_jtj_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name: str,
) -> Callable[[PyTree], PyTree]:
    """
    Description
    -----------
    Construct JᵀJ operator for a single parameter block.

    Computes J_blockᵀ J_block @ v for the specified block.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model mapping params to 4D-STEM datacube.
    - `params` (PtychoParams):
        Current parameter values.
    - `block_name` (str):
        Name of block.

    Returns
    -------
    - `jtj_fn` (Callable[[PyTree], PyTree]):
        Function computing J_blockᵀ J_block @ v.
    """
    jvp_fn: Callable = block_jacobian_operator(forward_fn, params, block_name)
    vjp_fn: Callable = block_vjp_operator(forward_fn, params, block_name)
    
    def jtj_fn(
        vector: PyTree,
    ) -> PyTree:
        forward_result: Float[Array, "num_pos det_h det_w"] = jvp_fn(vector)
        backward_result: PyTree = vjp_fn(forward_result)
        return backward_result
    
    return jtj_fn


def cross_block_jtj_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name_row: str,
    block_name_col: str,
) -> Callable[[PyTree], PyTree]:
    """
    Description
    -----------
    Construct cross-block JᵀJ operator.

    Computes J_row^T J_col @ v, the off-diagonal block of the full JᵀJ.
    Needed for Schur complement computation.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model.
    - `params` (PtychoParams):
        Current parameter values.
    - `block_name_row` (str):
        Row block name (for VJP).
    - `block_name_col` (str):
        Column block name (for JVP).

    Returns
    -------
    - `cross_jtj_fn` (Callable[[PyTree], PyTree]):
        Function computing J_rowᵀ J_col @ v.
    """
    jvp_fn: Callable = block_jacobian_operator(forward_fn, params, block_name_col)
    vjp_fn: Callable = block_vjp_operator(forward_fn, params, block_name_row)
    
    def cross_jtj_fn(
        vector: PyTree,
    ) -> PyTree:
        forward_result: Float[Array, "num_pos det_h det_w"] = jvp_fn(vector)
        backward_result: PyTree = vjp_fn(forward_result)
        return backward_result
    
    return cross_jtj_fn


def compute_block_gradient(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    residual: Float[Array, "num_pos det_h det_w"],
    block_name: str,
) -> PyTree:
    """
    Description
    -----------
    Compute gradient Jᵀr for a single parameter block.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model.
    - `params` (PtychoParams):
        Current parameter values.
    - `residual` (Float[Array, "num_pos det_h det_w"]):
        Current residual (forward - data).
    - `block_name` (str):
        Name of block.

    Returns
    -------
    - `gradient` (PyTree):
        Gradient Jᵀr for the specified block.
    """
    vjp_fn: Callable = block_vjp_operator(forward_fn, params, block_name)
    gradient: PyTree = vjp_fn(residual)
    return gradient


def block_gauss_newton_step(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    data: Float[Array, "num_pos det_h det_w"],
    block_names: List[str],
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> PtychoParams:
    """
    Description
    -----------
    Perform Gauss-Newton step updating only specified blocks.

    Solves the normal equations for the selected blocks while
    holding other blocks fixed.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model.
    - `params` (PtychoParams):
        Current parameter values.
    - `data` (Float[Array, "num_pos det_h det_w"]):
        Observed 4D-STEM data.
    - `block_names` (List[str]):
        List of blocks to update.
    - `cg_max_iterations` (int):
        Max CG iterations. Default 50.
    - `cg_tolerance` (float):
        CG tolerance. Default 1e-6.

    Returns
    -------
    - `new_params` (PtychoParams):
        Updated parameters.

    Flow
    ----
    1. Compute residual
    2. For each block: compute gradient and JᵀJ operator
    3. Solve block-wise normal equations via CG
    4. Update specified blocks, keep others fixed
    5. Return new params
    """
    from ptyrodactyl.jacobian.solvers import conjugate_gradient
    
    prediction: Float[Array, "num_pos det_h det_w"] = forward_fn(params)
    residual: Float[Array, "num_pos det_h det_w"] = prediction - data
    
    new_exit_wave: ExitWaveParams = params.exit_wave
    new_aberrations: AberrationParams = params.aberrations
    new_geometry: GeometryParams = params.geometry
    new_positions: PositionParams = params.positions
    new_probe_modes: ProbeModeParams = params.probe_modes
    
    for block_name in block_names:
        gradient: PyTree = compute_block_gradient(forward_fn, params, residual, block_name)
        jtj_fn: Callable = block_jtj_operator(forward_fn, params, block_name)
        
        if block_name == 'exit_wave':
            x0: PyTree = _tree_zeros_like(params.exit_wave)
            step, _ = conjugate_gradient(jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance)
            new_wave: Complex[Array, "h w"] = params.exit_wave.wave - step.wave
            new_exit_wave = ExitWaveParams(wave=new_wave)
            
        elif block_name == 'aberrations':
            x0 = _tree_zeros_like(params.aberrations)
            step, _ = conjugate_gradient(jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance)
            new_aberrations = AberrationParams(
                zernike_coeffs=params.aberrations.zernike_coeffs - step.zernike_coeffs,
                aperture_mrad=params.aberrations.aperture_mrad - step.aperture_mrad,
                aperture_softness=params.aberrations.aperture_softness - step.aperture_softness,
            )
            
        elif block_name == 'geometry':
            x0 = _tree_zeros_like(params.geometry)
            step, _ = conjugate_gradient(jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance)
            new_geometry = GeometryParams(
                rotation_rad=params.geometry.rotation_rad - step.rotation_rad,
                center_offset=params.geometry.center_offset - step.center_offset,
                ellipticity=params.geometry.ellipticity - step.ellipticity,
            )
            
        elif block_name == 'positions':
            x0 = _tree_zeros_like(params.positions)
            step, _ = conjugate_gradient(jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance)
            new_positions = PositionParams(
                position_offsets=params.positions.position_offsets - step.position_offsets,
            )
            
        elif block_name == 'probe_modes':
            x0 = _tree_zeros_like(params.probe_modes)
            step, _ = conjugate_gradient(jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance)
            new_probe_modes = ProbeModeParams(
                mode_weights=params.probe_modes.mode_weights - step.mode_weights,
                mode_phases=params.probe_modes.mode_phases - step.mode_phases,
            )
    
    new_params: PtychoParams = PtychoParams(
        exit_wave=new_exit_wave,
        aberrations=new_aberrations,
        geometry=new_geometry,
        positions=new_positions,
        probe_modes=new_probe_modes,
    )
    
    return new_params


def alternating_block_solve(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params_init: PtychoParams,
    data: Float[Array, "num_pos det_h det_w"],
    block_schedule: List[List[str]],
    num_outer_iterations: int = 10,
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> Tuple[PtychoParams, Float[Array, "num_outer"]]:
    """
    Description
    -----------
    Solve via alternating block updates following a schedule.

    Parameters
    ----------
    - `forward_fn` (Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]):
        Forward model.
    - `params_init` (PtychoParams):
        Initial parameters.
    - `data` (Float[Array, "num_pos det_h det_w"]):
        Observed 4D-STEM data.
    - `block_schedule` (List[List[str]]):
        Schedule of blocks to update each inner iteration.
        E.g., [['exit_wave'], ['positions', 'aberrations']] alternates.
    - `num_outer_iterations` (int):
        Number of full passes through schedule. Default 10.
    - `cg_max_iterations` (int):
        Max CG iterations per block. Default 50.
    - `cg_tolerance` (float):
        CG tolerance. Default 1e-6.

    Returns
    -------
    - `final_params` (PtychoParams):
        Optimized parameters.
    - `residual_history` (Float[Array, "num_outer"]):
        Residual norm at each outer iteration.

    Flow
    ----
    1. For each outer iteration:
       a. For each block group in schedule:
          - Perform block GN step
       b. Record residual norm
    2. Return final params and history
    """
    def compute_residual_norm(
        params: PtychoParams,
    ) -> Float[Array, ""]:
        prediction: Float[Array, "num_pos det_h det_w"] = forward_fn(params)
        residual: Float[Array, "num_pos det_h det_w"] = prediction - data
        return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2))
    
    def outer_iteration(
        carry: Tuple[PtychoParams, int],
        _: None,
    ) -> Tuple[Tuple[PtychoParams, int], Float[Array, ""]]:
        params, iteration = carry
        
        def inner_step(
            params_inner: PtychoParams,
            block_group: List[str],
        ) -> PtychoParams:
            return block_gauss_newton_step(
                forward_fn, params_inner, data, block_group,
                cg_max_iterations, cg_tolerance
            )
        
        updated_params: PtychoParams = params
        for block_group in block_schedule:
            updated_params = inner_step(updated_params, block_group)
        
        residual_norm: Float[Array, ""] = compute_residual_norm(updated_params)
        
        return (updated_params, iteration + 1), residual_norm
    
    initial_carry: Tuple[PtychoParams, int] = (params_init, 0)
    (final_params, _), residual_history = lax.scan(
        outer_iteration, initial_carry, None, length=num_outer_iterations
    )
    
    return final_params, residual_history