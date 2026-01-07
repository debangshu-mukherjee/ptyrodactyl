"""
Module: ptyrodactyl.jacobian.solvers
------------------------------------

Second-order solvers and spectral analysis for nonlinear least-squares.

This module provides Gauss-Newton, Levenberg-Marquardt, and Krylov
subspace methods that expose the Jacobian structure essential for
understanding gauge freedom and observability in inverse problems.

Functions
---------
- `conjugate_gradient`:
    Matrix-free CG solver for symmetric positive semi-definite systems
- `gauss_newton_step`:
    Single Gauss-Newton update step
- `gauss_newton_solve`:
    Full Gauss-Newton iteration to convergence
- `levenberg_marquardt_step`:
    Single LM update step with adaptive damping
- `levenberg_marquardt_solve`:
    Full LM iteration to convergence
- `lanczos_tridiagonal`:
    Lanczos algorithm for tridiagonalizing symmetric operators
- `singular_spectrum`:
    Estimate singular values of Jacobian via Lanczos on JᵀJ
- `effective_nullspace_dimension`:
    Count dimensions below noise threshold
"""

from typing import Callable, Tuple, NamedTuple
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Float, Int, Array, PyTree

from ptyrodactyl.jacobian.operators import jtj_operator, vjp_operator


class CGState(NamedTuple):
    """State for conjugate gradient iteration."""
    x: PyTree
    r: PyTree
    p: PyTree
    r_dot_r: Float[Array, ""]
    iteration: Int[Array, ""]


class GNState(NamedTuple):
    """State for Gauss-Newton iteration."""
    params: PyTree
    residual_norm: Float[Array, ""]
    iteration: Int[Array, ""]


class LMState(NamedTuple):
    """State for Levenberg-Marquardt iteration."""
    params: PyTree
    residual_norm: Float[Array, ""]
    damping: Float[Array, ""]
    iteration: Int[Array, ""]


class LanczosState(NamedTuple):
    """State for Lanczos tridiagonalization."""
    v_prev: Float[Array, "n"]
    v_curr: Float[Array, "n"]
    alpha: Float[Array, "k"]
    beta: Float[Array, "k"]
    iteration: Int[Array, ""]


def _tree_dot(
    tree_a: PyTree,
    tree_b: PyTree,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute inner product between two PyTrees with matching structure.

    Parameters
    ----------
    - `tree_a` (PyTree):
        First PyTree operand.
    - `tree_b` (PyTree):
        Second PyTree operand with same structure as tree_a.

    Returns
    -------
    - `result` (Float[Array, ""]):
        Sum of element-wise products across all leaves.
    """
    leaves_a, _ = jax.tree_util.tree_flatten(tree_a)
    leaves_b, _ = jax.tree_util.tree_flatten(tree_b)
    products: list = [jnp.sum(a * b) for a, b in zip(leaves_a, leaves_b)]
    result: Float[Array, ""] = jnp.sum(jnp.array(products))
    return result


def _tree_add(
    tree_a: PyTree,
    tree_b: PyTree,
) -> PyTree:
    """
    Description
    -----------
    Element-wise addition of two PyTrees with matching structure.

    Parameters
    ----------
    - `tree_a` (PyTree):
        First PyTree operand.
    - `tree_b` (PyTree):
        Second PyTree operand with same structure as tree_a.

    Returns
    -------
    - `result` (PyTree):
        PyTree with element-wise sum of leaves.
    """
    result: PyTree = jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)
    return result


def _tree_scalar_mul(
    scalar: Float[Array, ""],
    tree: PyTree,
) -> PyTree:
    """
    Description
    -----------
    Multiply all leaves of a PyTree by a scalar.

    Parameters
    ----------
    - `scalar` (Float[Array, ""]):
        Scalar multiplier.
    - `tree` (PyTree):
        PyTree to scale.

    Returns
    -------
    - `result` (PyTree):
        Scaled PyTree.
    """
    result: PyTree = jax.tree_util.tree_map(lambda x: scalar * x, tree)
    return result


def _tree_sub(
    tree_a: PyTree,
    tree_b: PyTree,
) -> PyTree:
    """
    Description
    -----------
    Element-wise subtraction of two PyTrees with matching structure.

    Parameters
    ----------
    - `tree_a` (PyTree):
        First PyTree operand.
    - `tree_b` (PyTree):
        Second PyTree operand to subtract from tree_a.

    Returns
    -------
    - `result` (PyTree):
        PyTree with element-wise difference of leaves.
    """
    result: PyTree = jax.tree_util.tree_map(lambda a, b: a - b, tree_a, tree_b)
    return result


def _tree_zeros_like(
    tree: PyTree,
) -> PyTree:
    """
    Description
    -----------
    Create a PyTree of zeros with same structure as input.

    Parameters
    ----------
    - `tree` (PyTree):
        Template PyTree.

    Returns
    -------
    - `result` (PyTree):
        PyTree of zeros with matching structure.
    """
    result: PyTree = jax.tree_util.tree_map(jnp.zeros_like, tree)
    return result


def _tree_norm(
    tree: PyTree,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute L2 norm of a PyTree.

    Parameters
    ----------
    - `tree` (PyTree):
        Input PyTree.

    Returns
    -------
    - `result` (Float[Array, ""]):
        Square root of sum of squared elements across all leaves.
    """
    dot_product: Float[Array, ""] = _tree_dot(tree, tree)
    result: Float[Array, ""] = jnp.sqrt(dot_product)
    return result


def conjugate_gradient(
    linear_operator: Callable[[PyTree], PyTree],
    rhs: PyTree,
    x0: PyTree,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[PyTree, Int[Array, ""]]:
    """
    Description
    -----------
    Solve the linear system A @ x = b using conjugate gradient method.

    The operator A must be symmetric positive semi-definite. This is
    satisfied by JᵀJ operators arising from linearized least-squares.

    Parameters
    ----------
    - `linear_operator` (Callable[[PyTree], PyTree]):
        Function computing A @ v for any vector v. Must be symmetric PSD.
    - `rhs` (PyTree):
        Right-hand side vector b.
    - `x0` (PyTree):
        Initial guess for solution.
    - `max_iterations` (int):
        Maximum number of CG iterations. Default 100.
    - `tolerance` (float):
        Convergence tolerance on residual norm. Default 1e-6.

    Returns
    -------
    - `solution` (PyTree):
        Approximate solution x satisfying A @ x ≈ b.
    - `iterations` (Int[Array, ""]):
        Number of iterations performed.

    Flow
    ----
    1. Initialize residual r = b - A @ x0
    2. Initialize search direction p = r
    3. For each iteration:
       a. Compute A @ p
       b. Compute step size α = (r·r) / (p·Ap)
       c. Update solution x = x + α*p
       d. Update residual r = r - α*Ap
       e. Compute β = (r_new·r_new) / (r_old·r_old)
       f. Update search direction p = r + β*p
    4. Terminate when ||r|| < tolerance or max_iterations reached
    """
    initial_residual: PyTree = _tree_sub(rhs, linear_operator(x0))
    initial_r_dot_r: Float[Array, ""] = _tree_dot(initial_residual, initial_residual)

    initial_state: CGState = CGState(
        x=x0,
        r=initial_residual,
        p=initial_residual,
        r_dot_r=initial_r_dot_r,
        iteration=jnp.array(0),
    )

    tolerance_squared: Float[Array, ""] = jnp.array(tolerance ** 2)

    def cg_step(
        state: CGState,
        _: None,
    ) -> Tuple[CGState, None]:
        a_times_p: PyTree = linear_operator(state.p)
        p_dot_ap: Float[Array, ""] = _tree_dot(state.p, a_times_p)
        alpha: Float[Array, ""] = state.r_dot_r / (p_dot_ap + 1e-12)
        x_new: PyTree = _tree_add(state.x, _tree_scalar_mul(alpha, state.p))
        r_new: PyTree = _tree_sub(state.r, _tree_scalar_mul(alpha, a_times_p))
        r_dot_r_new: Float[Array, ""] = _tree_dot(r_new, r_new)
        beta: Float[Array, ""] = r_dot_r_new / (state.r_dot_r + 1e-12)
        p_new: PyTree = _tree_add(r_new, _tree_scalar_mul(beta, state.p))
        new_iteration: Int[Array, ""] = state.iteration + 1

        converged: Bool[Array, ""] = r_dot_r_new < tolerance_squared
        x_out: PyTree = lax.cond(converged, lambda: state.x, lambda: x_new)
        r_out: PyTree = lax.cond(converged, lambda: state.r, lambda: r_new)
        p_out: PyTree = lax.cond(converged, lambda: state.p, lambda: p_new)
        r_dot_r_out: Float[Array, ""] = lax.cond(
            converged, lambda: state.r_dot_r, lambda: r_dot_r_new
        )

        new_state: CGState = CGState(
            x=x_out,
            r=r_out,
            p=p_out,
            r_dot_r=r_dot_r_out,
            iteration=new_iteration,
        )
        return new_state, None

    final_state, _ = lax.scan(cg_step, initial_state, None, length=max_iterations)
    return final_state.x, final_state.iteration


def gauss_newton_step(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    data: Float[Array, "..."],
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> Tuple[PyTree, Float[Array, ""]]:
    """
    Description
    -----------
    Compute a single Gauss-Newton update step for nonlinear least-squares.

    Solves the linearized normal equations JᵀJ δ = Jᵀr using CG, where
    J is the Jacobian at current params and r is the residual.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Current parameter estimate.
    - `data` (Float[Array, "..."]):
        Observed measurements.
    - `cg_max_iterations` (int):
        Maximum CG iterations for inner solve. Default 50.
    - `cg_tolerance` (float):
        CG convergence tolerance. Default 1e-6.

    Returns
    -------
    - `new_params` (PyTree):
        Updated parameter estimate after one GN step.
    - `residual_norm` (Float[Array, ""]):
        Norm of residual at new_params.

    Flow
    ----
    1. Compute residual r = f(θ) - y
    2. Compute gradient Jᵀr via VJP
    3. Construct JᵀJ operator
    4. Solve JᵀJ δ = Jᵀr via CG
    5. Update θ_new = θ - δ
    6. Return new params and residual norm
    """
    prediction: Float[Array, "..."] = forward_fn(params)
    residual: Float[Array, "..."] = prediction - data
    residual_norm_current: Float[Array, ""] = jnp.sqrt(jnp.sum(residual ** 2))

    vjp_fn: Callable = vjp_operator(forward_fn, params)
    gradient: PyTree = vjp_fn(residual)

    jtj_fn: Callable = jtj_operator(forward_fn, params)

    x0: PyTree = _tree_zeros_like(params)
    step, _ = conjugate_gradient(jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance)

    new_params: PyTree = _tree_sub(params, step)
    new_prediction: Float[Array, "..."] = forward_fn(new_params)
    new_residual: Float[Array, "..."] = new_prediction - data
    residual_norm_new: Float[Array, ""] = jnp.sqrt(jnp.sum(new_residual ** 2))

    return new_params, residual_norm_new


def gauss_newton_solve(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params_init: PyTree,
    data: Float[Array, "..."],
    max_iterations: int = 20,
    tolerance: float = 1e-8,
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> Tuple[PyTree, GNState]:
    """
    Description
    -----------
    Solve nonlinear least-squares via iterated Gauss-Newton.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params_init` (PyTree):
        Initial parameter guess.
    - `data` (Float[Array, "..."]):
        Observed measurements.
    - `max_iterations` (int):
        Maximum number of GN iterations. Default 20.
    - `tolerance` (float):
        Convergence tolerance on residual norm. Default 1e-8.
    - `cg_max_iterations` (int):
        Maximum CG iterations per GN step. Default 50.
    - `cg_tolerance` (float):
        CG convergence tolerance. Default 1e-6.

    Returns
    -------
    - `params_final` (PyTree):
        Optimized parameters.
    - `final_state` (GNState):
        Final optimization state including residual norm and iteration count.

    Flow
    ----
    1. Initialize state with params_init
    2. For each iteration:
       a. Compute GN step
       b. Check convergence
       c. Update state
    3. Return final parameters and state
    """
    initial_prediction: Float[Array, "..."] = forward_fn(params_init)
    initial_residual: Float[Array, "..."] = initial_prediction - data
    initial_norm: Float[Array, ""] = jnp.sqrt(jnp.sum(initial_residual ** 2))

    initial_state: GNState = GNState(
        params=params_init,
        residual_norm=initial_norm,
        iteration=jnp.array(0),
    )

    def gn_iteration(
        state: GNState,
        _: None,
    ) -> Tuple[GNState, None]:
        new_params, new_norm = gauss_newton_step(
            forward_fn, state.params, data, cg_max_iterations, cg_tolerance
        )
        converged: Bool[Array, ""] = new_norm < tolerance
        params_out: PyTree = lax.cond(
            converged, lambda: state.params, lambda: new_params
        )
        norm_out: Float[Array, ""] = lax.cond(
            converged, lambda: state.residual_norm, lambda: new_norm
        )
        new_state: GNState = GNState(
            params=params_out,
            residual_norm=norm_out,
            iteration=state.iteration + 1,
        )
        return new_state, None

    final_state, _ = lax.scan(gn_iteration, initial_state, None, length=max_iterations)
    return final_state.params, final_state


def levenberg_marquardt_step(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    data: Float[Array, "..."],
    damping: Float[Array, ""],
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> Tuple[PyTree, Float[Array, ""], Float[Array, ""]]:
    """
    Description
    -----------
    Compute a single Levenberg-Marquardt update step.

    LM interpolates between Gauss-Newton (small damping) and gradient
    descent (large damping). Solves (JᵀJ + λI) δ = Jᵀr.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Current parameter estimate.
    - `data` (Float[Array, "..."]):
        Observed measurements.
    - `damping` (Float[Array, ""]):
        Damping parameter λ. Larger values → more regularized step.
    - `cg_max_iterations` (int):
        Maximum CG iterations for inner solve. Default 50.
    - `cg_tolerance` (float):
        CG convergence tolerance. Default 1e-6.

    Returns
    -------
    - `new_params` (PyTree):
        Updated parameter estimate.
    - `new_residual_norm` (Float[Array, ""]):
        Residual norm at new_params.
    - `new_damping` (Float[Array, ""]):
        Adapted damping for next iteration.

    Flow
    ----
    1. Compute residual and gradient
    2. Construct damped normal equations operator (JᵀJ + λI)
    3. Solve for step via CG
    4. Evaluate gain ratio ρ = actual_reduction / predicted_reduction
    5. Accept step if ρ > 0, adjust damping based on ρ
    6. Return new params and adapted damping
    """
    prediction: Float[Array, "..."] = forward_fn(params)
    residual: Float[Array, "..."] = prediction - data
    residual_norm_current: Float[Array, ""] = jnp.sum(residual ** 2)

    vjp_fn: Callable = vjp_operator(forward_fn, params)
    gradient: PyTree = vjp_fn(residual)
    jtj_fn: Callable = jtj_operator(forward_fn, params)

    def damped_operator(v: PyTree) -> PyTree:
        jtj_v: PyTree = jtj_fn(v)
        damped_term: PyTree = _tree_scalar_mul(damping, v)
        return _tree_add(jtj_v, damped_term)

    x0: PyTree = _tree_zeros_like(params)
    step, _ = conjugate_gradient(
        damped_operator, gradient, x0, cg_max_iterations, cg_tolerance
    )

    new_params: PyTree = _tree_sub(params, step)
    new_prediction: Float[Array, "..."] = forward_fn(new_params)
    new_residual: Float[Array, "..."] = new_prediction - data
    new_residual_norm: Float[Array, ""] = jnp.sum(new_residual ** 2)

    actual_reduction: Float[Array, ""] = residual_norm_current - new_residual_norm
    jtj_step: PyTree = jtj_fn(step)
    predicted_reduction: Float[Array, ""] = (
        _tree_dot(step, gradient) - 0.5 * _tree_dot(step, jtj_step)
    )
    gain_ratio: Float[Array, ""] = actual_reduction / (predicted_reduction + 1e-12)

    damping_decrease: Float[Array, ""] = damping / 3.0
    damping_increase: Float[Array, ""] = damping * 2.0
    new_damping: Float[Array, ""] = lax.cond(
        gain_ratio > 0.75,
        lambda: damping_decrease,
        lambda: lax.cond(
            gain_ratio < 0.25,
            lambda: damping_increase,
            lambda: damping,
        ),
    )

    params_out: PyTree = lax.cond(
        gain_ratio > 0.0,
        lambda: new_params,
        lambda: params,
    )
    norm_out: Float[Array, ""] = lax.cond(
        gain_ratio > 0.0,
        lambda: jnp.sqrt(new_residual_norm),
        lambda: jnp.sqrt(residual_norm_current),
    )

    return params_out, norm_out, new_damping


def levenberg_marquardt_solve(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params_init: PyTree,
    data: Float[Array, "..."],
    max_iterations: int = 50,
    tolerance: float = 1e-8,
    damping_init: float = 1e-3,
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> Tuple[PyTree, LMState]:
    """
    Description
    -----------
    Solve nonlinear least-squares via Levenberg-Marquardt.

    LM is more robust than pure Gauss-Newton for ill-conditioned problems
    or when far from the solution. Adaptively adjusts damping.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params_init` (PyTree):
        Initial parameter guess.
    - `data` (Float[Array, "..."]):
        Observed measurements.
    - `max_iterations` (int):
        Maximum number of LM iterations. Default 50.
    - `tolerance` (float):
        Convergence tolerance on residual norm. Default 1e-8.
    - `damping_init` (float):
        Initial damping parameter. Default 1e-3.
    - `cg_max_iterations` (int):
        Maximum CG iterations per LM step. Default 50.
    - `cg_tolerance` (float):
        CG convergence tolerance. Default 1e-6.

    Returns
    -------
    - `params_final` (PyTree):
        Optimized parameters.
    - `final_state` (LMState):
        Final optimization state.

    Flow
    ----
    1. Initialize state with params_init and damping_init
    2. For each iteration: compute LM step with adaptive damping
    3. Check convergence, update state
    4. Return final parameters and state
    """
    initial_prediction: Float[Array, "..."] = forward_fn(params_init)
    initial_residual: Float[Array, "..."] = initial_prediction - data
    initial_norm: Float[Array, ""] = jnp.sqrt(jnp.sum(initial_residual ** 2))

    initial_state: LMState = LMState(
        params=params_init,
        residual_norm=initial_norm,
        damping=jnp.array(damping_init),
        iteration=jnp.array(0),
    )

    def lm_iteration(
        state: LMState,
        _: None,
    ) -> Tuple[LMState, None]:
        new_params, new_norm, new_damping = levenberg_marquardt_step(
            forward_fn, state.params, data, state.damping,
            cg_max_iterations, cg_tolerance
        )
        converged: Bool[Array, ""] = new_norm < tolerance
        params_out: PyTree = lax.cond(
            converged, lambda: state.params, lambda: new_params
        )
        norm_out: Float[Array, ""] = lax.cond(
            converged, lambda: state.residual_norm, lambda: new_norm
        )
        damping_out: Float[Array, ""] = lax.cond(
            converged, lambda: state.damping, lambda: new_damping
        )
        new_state: LMState = LMState(
            params=params_out,
            residual_norm=norm_out,
            damping=damping_out,
            iteration=state.iteration + 1,
        )
        return new_state, None

    final_state, _ = lax.scan(lm_iteration, initial_state, None, length=max_iterations)
    return final_state.params, final_state


def lanczos_tridiagonal(
    linear_operator: Callable[[Float[Array, "n"]], Float[Array, "n"]],
    initial_vector: Float[Array, "n"],
    num_iterations: int,
) -> Tuple[Float[Array, "k"], Float[Array, "k-1"]]:
    """
    Description
    -----------
    Compute the Lanczos tridiagonalization of a symmetric operator.

    The Lanczos algorithm builds an orthonormal basis for the Krylov
    subspace and produces a tridiagonal matrix whose eigenvalues
    approximate the extremal eigenvalues of the original operator.

    Parameters
    ----------
    - `linear_operator` (Callable[[Float[Array, "n"]], Float[Array, "n"]]):
        Symmetric linear operator A. Must satisfy <Ax, y> = <x, Ay>.
    - `initial_vector` (Float[Array, "n"]):
        Starting vector for Krylov subspace. Should be random/non-sparse.
    - `num_iterations` (int):
        Number of Lanczos iterations. Determines size of tridiagonal matrix.

    Returns
    -------
    - `alpha` (Float[Array, "k"]):
        Diagonal elements of tridiagonal matrix.
    - `beta` (Float[Array, "k-1"]):
        Off-diagonal elements of tridiagonal matrix.

    Flow
    ----
    1. Normalize initial vector
    2. For each iteration:
       a. Apply operator: w = A @ v_curr
       b. Compute α = <w, v_curr>
       c. Orthogonalize: w = w - α*v_curr - β*v_prev
       d. Compute β = ||w||
       e. Normalize: v_next = w / β
    3. Return diagonal (α) and off-diagonal (β) elements
    """
    n: int = initial_vector.shape[0]
    k: int = num_iterations

    v0_norm: Float[Array, ""] = jnp.linalg.norm(initial_vector)
    v0_normalized: Float[Array, "n"] = initial_vector / v0_norm

    initial_state: LanczosState = LanczosState(
        v_prev=jnp.zeros(n),
        v_curr=v0_normalized,
        alpha=jnp.zeros(k),
        beta=jnp.zeros(k),
        iteration=jnp.array(0),
    )

    def lanczos_step(
        state: LanczosState,
        _: None,
    ) -> Tuple[LanczosState, None]:
        w: Float[Array, "n"] = linear_operator(state.v_curr)
        alpha_i: Float[Array, ""] = jnp.dot(w, state.v_curr)

        beta_prev: Float[Array, ""] = lax.cond(
            state.iteration > 0,
            lambda: state.beta[state.iteration - 1],
            lambda: jnp.array(0.0),
        )
        w_orth: Float[Array, "n"] = (
            w - alpha_i * state.v_curr - beta_prev * state.v_prev
        )

        beta_i: Float[Array, ""] = jnp.linalg.norm(w_orth)
        v_next: Float[Array, "n"] = w_orth / (beta_i + 1e-12)

        new_alpha: Float[Array, "k"] = state.alpha.at[state.iteration].set(alpha_i)
        new_beta: Float[Array, "k"] = state.beta.at[state.iteration].set(beta_i)

        new_state: LanczosState = LanczosState(
            v_prev=state.v_curr,
            v_curr=v_next,
            alpha=new_alpha,
            beta=new_beta,
            iteration=state.iteration + 1,
        )
        return new_state, None

    final_state, _ = lax.scan(lanczos_step, initial_state, None, length=k)

    alpha_out: Float[Array, "k"] = final_state.alpha
    beta_out: Float[Array, "k-1"] = final_state.beta[:-1]

    return alpha_out, beta_out


def singular_spectrum(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    num_singular_values: int = 50,
    num_lanczos_iterations: int = 100,
    random_seed: int = 42,
) -> Float[Array, "k"]:
    """
    Description
    -----------
    Estimate the singular spectrum of the Jacobian via Lanczos on JᵀJ.

    The singular values σ_i of J equal the square roots of the eigenvalues
    of JᵀJ. The Lanczos algorithm approximates extremal eigenvalues well,
    giving accurate estimates of the largest and smallest singular values.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `num_singular_values` (int):
        Number of singular values to estimate. Default 50.
    - `num_lanczos_iterations` (int):
        Lanczos iterations. Should be ≥ num_singular_values. Default 100.
    - `random_seed` (int):
        Seed for random starting vector. Default 42.

    Returns
    -------
    - `singular_values` (Float[Array, "k"]):
        Estimated singular values in descending order.

    Flow
    ----
    1. Flatten params to vector form for Lanczos
    2. Construct JᵀJ operator on flattened space
    3. Run Lanczos to get tridiagonal matrix
    4. Compute eigenvalues of tridiagonal matrix
    5. Take square roots, sort descending, return top k
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]

    jtj_pytree_fn: Callable = jtj_operator(forward_fn, params)

    def jtj_flat_fn(
        v_flat: Float[Array, "n"],
    ) -> Float[Array, "n"]:
        v_pytree: PyTree = unflatten_fn(v_flat)
        result_pytree: PyTree = jtj_pytree_fn(v_pytree)
        result_flat: Float[Array, "n"] = jax.flatten_util.ravel_pytree(result_pytree)[0]
        return result_flat

    key: jax.random.PRNGKey = jax.random.PRNGKey(random_seed)
    initial_vector: Float[Array, "n"] = jax.random.normal(key, (n,))

    alpha, beta = lanczos_tridiagonal(jtj_flat_fn, initial_vector, num_lanczos_iterations)

    tridiag_size: int = num_lanczos_iterations
    tridiag_matrix: Float[Array, "k k"] = (
        jnp.diag(alpha) +
        jnp.diag(beta, k=1) +
        jnp.diag(beta, k=-1)
    )

    eigenvalues: Float[Array, "k"] = jnp.linalg.eigvalsh(tridiag_matrix)
    eigenvalues_positive: Float[Array, "k"] = jnp.maximum(eigenvalues, 0.0)
    singular_values_all: Float[Array, "k"] = jnp.sqrt(eigenvalues_positive)
    singular_values_sorted: Float[Array, "k"] = jnp.sort(singular_values_all)[::-1]
    singular_values_out: Float[Array, "k"] = singular_values_sorted[:num_singular_values]

    return singular_values_out


def effective_nullspace_dimension(
    singular_values: Float[Array, "k"],
    noise_floor: float,
) -> Int[Array, ""]:
    """
    Description
    -----------
    Count the number of singular values below the noise floor.

    Directions with σ_i < noise_floor are effectively unobservable:
    they contribute more noise amplification than signal recovery.
    This count gives the effective dimension of the gauge subspace
    under finite SNR.

    Parameters
    ----------
    - `singular_values` (Float[Array, "k"]):
        Singular values of the Jacobian (or estimates thereof).
    - `noise_floor` (float):
        Noise threshold η. Directions with σ < η are effectively null.

    Returns
    -------
    - `nullspace_dim` (Int[Array, ""]):
        Number of singular values below noise_floor.

    Flow
    ----
    1. Count singular values < noise_floor
    2. Return count as effective nullspace dimension
    """
    below_threshold: Float[Array, "k"] = (singular_values < noise_floor).astype(jnp.int32)
    nullspace_dim: Int[Array, ""] = jnp.sum(below_threshold)
    return nullspace_dim