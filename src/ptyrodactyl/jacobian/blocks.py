r"""Block-structured parameter management for ptychography.

Extended Summary
----------------
Ptychography involves multiple parameter groups with different
physical meanings and observability characteristics.  This module
computes block-wise Jacobians for Schur complement marginalisation
and targeted updates.

The five parameter blocks are:

1. **Exit wave** -- Complex-valued specimen exit wave
   :math:`\psi(x, y)`.
2. **Aberrations** -- Zernike polynomial coefficients and soft
   aperture.
3. **Geometry** -- Rotation, centering, ellipticity calibration.
4. **Positions** -- Per-scan-point position errors (dx, dy).
5. **Probe modes** -- Modal decomposition for partial coherence.

Routine Listings
----------------
:func:`alternating_block_solve`
    Solve via alternating block updates following a schedule.
:func:`block_gauss_newton_step`
    Gauss-Newton step updating only specified blocks.
:func:`block_jacobian_operator`
    JVP operator for a single parameter block.
:func:`block_jtj_operator`
    J^T J operator for a single parameter block.
:func:`block_vjp_operator`
    VJP operator for a single parameter block.
:func:`compute_block_gradient`
    Gradient J^T r for a single parameter block.
:func:`cross_block_jtj_operator`
    Cross-block J^T J operator for Schur complements.
:func:`split_params`
    Extract individual blocks from combined params.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Complex, Float, PyTree

from ptyrodactyl.jacobian.solvers import conjugate_gradient
from ptyrodactyl.types import (
    AberrationParams,
    ExitWaveParams,
    GeometryParams,
    PositionParams,
    ProbeModeParams,
    PtychoParams,
)


def split_params(
    params: PtychoParams,
) -> tuple[
    ExitWaveParams,
    AberrationParams,
    GeometryParams,
    PositionParams,
    ProbeModeParams,
]:
    """Extract individual parameter blocks from combined params.

    Parameters
    ----------
    params : PtychoParams
        Combined parameter container.

    Returns
    -------
    exit_wave : ExitWaveParams
        Exit wave parameters.
    aberrations : AberrationParams
        Aberration parameters.
    geometry : GeometryParams
        Geometry parameters.
    positions : PositionParams
        Position parameters.
    probe_modes : ProbeModeParams
        Probe mode parameters.

    See Also
    --------
    :func:`ptyrodactyl.types.create_ptycho_params` : Inverse operation.
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
    """Create a PyTree of zeros matching the input structure.

    Parameters
    ----------
    tree : PyTree
        Template PyTree whose leaf shapes and dtypes are
        copied.

    Returns
    -------
    zeros : PyTree
        PyTree of zeros with the same structure as *tree*.
    """
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def block_jacobian_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name: str,
) -> Callable[[PyTree], Float[Array, "num_pos det_h det_w"]]:
    r"""Construct a JVP operator for a single parameter block.

    Extended Summary
    ----------------
    Computes :math:`J_{\text{block}} \, v` where
    :math:`J_{\text{block}} = \partial f / \partial \theta_b`,
    treating all other blocks as fixed.

    Implementation Logic
    --------------------
    1. **Build sparse tangent** --
       Create a full-parameter tangent PyTree with zeros
       everywhere except the target block.
    2. **Evaluate JVP** --
       Call :func:`jax.jvp` with the sparse tangent.
    3. **Return output tangent** --
       The measurement-space tangent is returned directly.

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model mapping params to 4D-STEM datacube.
    params : PtychoParams
        Current parameter values.
    block_name : str
        Name of the block: ``'exit_wave'``,
        ``'aberrations'``, ``'geometry'``, ``'positions'``,
        or ``'probe_modes'``.

    Returns
    -------
    jvp_fn : Callable[[PyTree], Float[Array, "num_pos det_h det_w"]]
        Function computing J_block @ v for tangent vectors
        v in the block subspace.

    See Also
    --------
    :func:`block_vjp_operator` : Adjoint operator.
    """

    def jvp_fn(
        block_tangent: PyTree,
    ) -> Float[Array, "num_pos det_h det_w"]:
        """Compute J_block @ block_tangent."""
        full_tangent: PtychoParams = _tree_zeros_like(params)

        if block_name == "exit_wave":
            full_tangent = eqx.tree_at(
                lambda p: p.exit_wave,
                full_tangent,
                block_tangent,
            )
        elif block_name == "aberrations":
            full_tangent = eqx.tree_at(
                lambda p: p.aberrations,
                full_tangent,
                block_tangent,
            )
        elif block_name == "geometry":
            full_tangent = eqx.tree_at(
                lambda p: p.geometry,
                full_tangent,
                block_tangent,
            )
        elif block_name == "positions":
            full_tangent = eqx.tree_at(
                lambda p: p.positions,
                full_tangent,
                block_tangent,
            )
        elif block_name == "probe_modes":
            full_tangent = eqx.tree_at(
                lambda p: p.probe_modes,
                full_tangent,
                block_tangent,
            )

        _, output_tangent = jax.jvp(forward_fn, (params,), (full_tangent,))
        return output_tangent

    return jvp_fn


def block_vjp_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name: str,
) -> Callable[[Float[Array, "num_pos det_h det_w"]], PyTree]:
    r"""Construct a VJP operator for a single parameter block.

    Extended Summary
    ----------------
    Computes :math:`J_{\text{block}}^\top u` where
    :math:`J_{\text{block}} = \partial f / \partial \theta_b`.

    Implementation Logic
    --------------------
    1. **Compute full VJP** --
       Evaluate :func:`jax.vjp` over all parameters.
    2. **Extract target block** --
       Return only the gradient for *block_name*.

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model mapping params to 4D-STEM datacube.
    params : PtychoParams
        Current parameter values.
    block_name : str
        Name of the block: ``'exit_wave'``,
        ``'aberrations'``, ``'geometry'``, ``'positions'``,
        or ``'probe_modes'``.

    Returns
    -------
    vjp_fn : Callable[[Float[Array, "num_pos det_h det_w"]], PyTree]
        Function computing J_block^T @ u for cotangent u.

    See Also
    --------
    :func:`block_jacobian_operator` : Forward operator.
    """
    _, full_vjp_fn = jax.vjp(forward_fn, params)

    def vjp_fn(
        cotangent: Float[Array, "num_pos det_h det_w"],
    ) -> PyTree:
        """Compute J_block^T @ cotangent."""
        full_grad: tuple[PtychoParams] = full_vjp_fn(cotangent)
        params_grad: PtychoParams = full_grad[0]

        if block_name == "exit_wave":
            return params_grad.exit_wave
        if block_name == "aberrations":
            return params_grad.aberrations
        if block_name == "geometry":
            return params_grad.geometry
        if block_name == "positions":
            return params_grad.positions
        if block_name == "probe_modes":
            return params_grad.probe_modes

        msg = f"Unknown block_name: {block_name}"
        raise ValueError(msg)

    return vjp_fn


def block_jtj_operator(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    block_name: str,
) -> Callable[[PyTree], PyTree]:
    r"""Construct a J^T J operator for a single parameter block.

    Extended Summary
    ----------------
    Computes :math:`J_b^\top J_b \, v` for the specified block
    *b* by composing the block JVP and VJP operators.

    Implementation Logic
    --------------------
    1. **Forward multiply** --
       Apply the block JVP to get J_b v.
    2. **Backward multiply** --
       Apply the block VJP to get J_b^T (J_b v).

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model mapping params to 4D-STEM datacube.
    params : PtychoParams
        Current parameter values.
    block_name : str
        Name of the block.

    Returns
    -------
    jtj_fn : Callable[[PyTree], PyTree]
        Function computing J_block^T J_block @ v.

    See Also
    --------
    :func:`block_jacobian_operator` : Block JVP.
    :func:`block_vjp_operator` : Block VJP.
    """
    jvp_fn: Callable = block_jacobian_operator(forward_fn, params, block_name)
    vjp_fn: Callable = block_vjp_operator(forward_fn, params, block_name)

    def jtj_fn(
        vector: PyTree,
    ) -> PyTree:
        """Compute J_block^T J_block @ vector."""
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
    r"""Construct a cross-block J^T J operator.

    Extended Summary
    ----------------
    Computes :math:`J_{\text{row}}^\top J_{\text{col}} \, v`,
    the off-diagonal block of the full J^T J.  Needed for Schur
    complement computation.

    Implementation Logic
    --------------------
    1. **Forward via column block** --
       Apply the JVP for *block_name_col*.
    2. **Backward via row block** --
       Apply the VJP for *block_name_row*.

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model.
    params : PtychoParams
        Current parameter values.
    block_name_row : str
        Row block name (for VJP).
    block_name_col : str
        Column block name (for JVP).

    Returns
    -------
    cross_jtj_fn : Callable[[PyTree], PyTree]
        Function computing J_row^T J_col @ v.

    See Also
    --------
    :func:`block_jtj_operator` : Diagonal-block variant.
    """
    jvp_fn: Callable = block_jacobian_operator(
        forward_fn, params, block_name_col
    )
    vjp_fn: Callable = block_vjp_operator(forward_fn, params, block_name_row)

    def cross_jtj_fn(
        vector: PyTree,
    ) -> PyTree:
        """Compute J_row^T J_col @ vector."""
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
    r"""Compute the gradient J^T r for a single parameter block.

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model.
    params : PtychoParams
        Current parameter values.
    residual : Float[Array, "num_pos det_h det_w"]
        Current residual (forward - data).
    block_name : str
        Name of the block.

    Returns
    -------
    gradient : PyTree
        Gradient :math:`J_b^\top r` for the specified block.

    See Also
    --------
    :func:`block_vjp_operator` : Underlying VJP.
    """
    vjp_fn: Callable = block_vjp_operator(forward_fn, params, block_name)
    gradient: PyTree = vjp_fn(residual)
    return gradient


def block_gauss_newton_step(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params: PtychoParams,
    data: Float[Array, "num_pos det_h det_w"],
    block_names: list[str],
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> PtychoParams:
    """Perform a Gauss-Newton step updating only specified blocks.

    Extended Summary
    ----------------
    Solves the normal equations for the selected blocks while
    holding other blocks fixed.

    Implementation Logic
    --------------------
    1. **Compute residual** --
       residual = forward(params) - data.
    2. **Per-block solve** --
       For each block, compute the gradient and J^T J operator,
       then solve via CG.
    3. **Update selected blocks** --
       Subtract the CG solution from the current block values.
    4. **Preserve fixed blocks** --
       Non-selected blocks are returned unchanged.
    5. **Reassemble** --
       Return the updated :class:`PtychoParams`.

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model.
    params : PtychoParams
        Current parameter values.
    data : Float[Array, "num_pos det_h det_w"]
        Observed 4D-STEM data.
    block_names : List[str]
        List of blocks to update.
    cg_max_iterations : int
        Maximum CG iterations.  Default 50.
    cg_tolerance : float
        CG convergence tolerance.  Default 1e-6.

    Returns
    -------
    new_params : PtychoParams
        Updated parameters.

    See Also
    --------
    :func:`alternating_block_solve` : Full iterative solver.
    """
    prediction: Float[Array, "num_pos det_h det_w"] = forward_fn(params)
    residual: Float[Array, "num_pos det_h det_w"] = prediction - data

    new_params: PtychoParams = params

    for block_name in block_names:
        gradient: PyTree = compute_block_gradient(
            forward_fn, params, residual, block_name
        )
        jtj_fn: Callable = block_jtj_operator(forward_fn, params, block_name)

        if block_name == "exit_wave":
            x0: PyTree = _tree_zeros_like(params.exit_wave)
            step, _ = conjugate_gradient(
                jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance
            )
            new_wave: Complex[Array, "h w"] = params.exit_wave.wave - step.wave
            new_params = eqx.tree_at(
                lambda p: p.exit_wave.wave,
                new_params,
                new_wave,
            )

        elif block_name == "aberrations":
            x0 = _tree_zeros_like(params.aberrations)
            step, _ = conjugate_gradient(
                jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance
            )
            new_aberrations: tuple[
                Float[Array, "num_zernike"],
                Float[Array, ""],
                Float[Array, ""],
            ] = (
                params.aberrations.zernike_coeffs - step.zernike_coeffs,
                params.aberrations.aperture_mrad - step.aperture_mrad,
                params.aberrations.aperture_softness - step.aperture_softness,
            )
            new_params = eqx.tree_at(
                lambda p: (
                    p.aberrations.zernike_coeffs,
                    p.aberrations.aperture_mrad,
                    p.aberrations.aperture_softness,
                ),
                new_params,
                new_aberrations,
            )

        elif block_name == "geometry":
            x0 = _tree_zeros_like(params.geometry)
            step, _ = conjugate_gradient(
                jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance
            )
            new_geometry: tuple[
                Float[Array, ""],
                Float[Array, "2"],
                Float[Array, "2"],
            ] = (
                params.geometry.rotation_rad - step.rotation_rad,
                params.geometry.center_offset - step.center_offset,
                params.geometry.ellipticity - step.ellipticity,
            )
            new_params = eqx.tree_at(
                lambda p: (
                    p.geometry.rotation_rad,
                    p.geometry.center_offset,
                    p.geometry.ellipticity,
                ),
                new_params,
                new_geometry,
            )

        elif block_name == "positions":
            x0 = _tree_zeros_like(params.positions)
            step, _ = conjugate_gradient(
                jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance
            )
            new_position_offsets: Float[Array, "num_positions 2"] = (
                params.positions.position_offsets - step.position_offsets
            )
            new_params = eqx.tree_at(
                lambda p: p.positions.position_offsets,
                new_params,
                new_position_offsets,
            )

        elif block_name == "probe_modes":
            x0 = _tree_zeros_like(params.probe_modes)
            step, _ = conjugate_gradient(
                jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance
            )
            new_probe_modes: tuple[
                Float[Array, "num_modes"],
                Float[Array, "num_modes h w"],
            ] = (
                params.probe_modes.mode_weights - step.mode_weights,
                params.probe_modes.mode_phases - step.mode_phases,
            )
            new_params = eqx.tree_at(
                lambda p: (
                    p.probe_modes.mode_weights,
                    p.probe_modes.mode_phases,
                ),
                new_params,
                new_probe_modes,
            )

    return new_params


def alternating_block_solve(
    forward_fn: Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]],
    params_init: PtychoParams,
    data: Float[Array, "num_pos det_h det_w"],
    block_schedule: list[list[str]],
    num_outer_iterations: int = 10,
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> tuple[PtychoParams, Float[Array, "num_outer"]]:
    """Solve via alternating block updates following a schedule.

    Extended Summary
    ----------------
    Each outer iteration cycles through *block_schedule*,
    applying a block Gauss-Newton step for each group.  The
    outer loop is executed via :func:`jax.lax.scan` so the
    entire solve is JIT-compatible.

    Implementation Logic
    --------------------
    1. **Outer loop** --
       For each outer iteration, cycle through the full
       *block_schedule*.
    2. **Inner loop** --
       For each block group in the schedule, call
       :func:`block_gauss_newton_step`.
    3. **Record residual** --
       Compute and store the residual norm after each outer
       iteration.
    4. **Return** --
       Final params and the residual history array.

    Parameters
    ----------
    forward_fn : Callable[[PtychoParams], Float[Array, "num_pos det_h det_w"]]
        Forward model.
    params_init : PtychoParams
        Initial parameters.
    data : Float[Array, "num_pos det_h det_w"]
        Observed 4D-STEM data.
    block_schedule : List[List[str]]
        Schedule of blocks to update each inner iteration.
        E.g. ``[['exit_wave'], ['positions', 'aberrations']]``
        alternates between the two groups.
    num_outer_iterations : int
        Number of full passes through the schedule.
        Default 10.
    cg_max_iterations : int
        Maximum CG iterations per block.  Default 50.
    cg_tolerance : float
        CG convergence tolerance.  Default 1e-6.

    Returns
    -------
    final_params : PtychoParams
        Optimised parameters.
    residual_history : Float[Array, "num_outer"]
        Residual norm at each outer iteration.

    See Also
    --------
    :func:`block_gauss_newton_step` : Single-step primitive.
    """

    def compute_residual_norm(
        params: PtychoParams,
    ) -> Float[Array, ""]:
        """Compute L2 residual norm for *params*."""
        prediction: Float[Array, "num_pos det_h det_w"] = forward_fn(params)
        residual: Float[Array, "num_pos det_h det_w"] = prediction - data
        return jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2))

    def outer_iteration(
        carry: tuple[PtychoParams, int],
        _: None,
    ) -> tuple[tuple[PtychoParams, int], Float[Array, ""]]:
        """Execute one full pass through block_schedule."""
        params, iteration = carry

        def inner_step(
            params_inner: PtychoParams,
            block_group: list[str],
        ) -> PtychoParams:
            """Apply block GN step for one group."""
            return block_gauss_newton_step(
                forward_fn,
                params_inner,
                data,
                block_group,
                cg_max_iterations,
                cg_tolerance,
            )

        updated_params: PtychoParams = params
        for block_group in block_schedule:
            updated_params = inner_step(updated_params, block_group)

        residual_norm: Float[Array, ""] = compute_residual_norm(updated_params)

        return (updated_params, iteration + 1), residual_norm

    initial_carry: tuple[PtychoParams, int] = (params_init, 0)
    (final_params, _), residual_history = lax.scan(
        outer_iteration, initial_carry, None, length=num_outer_iterations
    )

    return final_params, residual_history


__all__: list[str] = [
    "alternating_block_solve",
    "block_gauss_newton_step",
    "block_jacobian_operator",
    "block_jtj_operator",
    "block_vjp_operator",
    "compute_block_gradient",
    "cross_block_jtj_operator",
    "split_params",
]
