r"""Second-order solvers and spectral analysis for least-squares.

Extended Summary
----------------
Provides Gauss-Newton, Levenberg-Marquardt, and Krylov subspace
methods that expose the Jacobian structure essential for
understanding gauge freedom and observability in inverse
problems.  All solvers operate on arbitrary JAX PyTree parameter
structures and are fully JIT-compatible via :func:`jax.lax.scan`
and :func:`jax.lax.fori_loop`.

Routine Listings
----------------
:func:`conjugate_gradient`
    Matrix-free CG solver for symmetric PSD systems.
:func:`effective_nullspace_dimension`
    Count dimensions below noise threshold.
:func:`gauss_newton_solve`
    Full Gauss-Newton iteration to convergence.
:func:`gauss_newton_step`
    Single Gauss-Newton update step.
:func:`lanczos_tridiagonal`
    Lanczos algorithm for tridiagonalising symmetric
    operators.
:func:`levenberg_marquardt_solve`
    Full LM iteration to convergence.
:func:`levenberg_marquardt_step`
    Single LM update step with adaptive damping.
:func:`singular_spectrum`
    Estimate singular values of Jacobian via Lanczos on
    J^T J.
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jax import lax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree, jaxtyped

from ptyrodactyl.jacobian._treemath import (
    _tree_add,
    _tree_dot,
    _tree_scalar_mul,
    _tree_sub,
    _tree_zeros_like,
)
from ptyrodactyl.jacobian.operators import jtj_operator, vjp_operator
from ptyrodactyl.types import CGState, GNState, LanczosState, LMState


@jaxtyped(typechecker=beartype)
def conjugate_gradient(
    linear_operator: Callable[[PyTree], PyTree],
    rhs: PyTree,
    x0: PyTree,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[PyTree, Int[Array, ""]]:
    r"""Solve A x = b via conjugate gradient.

    Extended Summary
    ----------------
    The operator *A* must be symmetric positive semi-definite.
    This is satisfied by :math:`J^\top J` operators arising
    from linearised least-squares.  The iteration is executed
    via :func:`jax.lax.scan` for JIT compatibility.

    Implementation Logic
    --------------------
    1. **Initialise** --
       r = b - A x0, p = r.
    2. **CG loop** --
       For each iteration:

       a. Compute A p.
       b. Step size :math:`\alpha = \langle r, r \rangle
          / \langle p, A p \rangle`.
       c. Update x = x + alpha * p.
       d. Update r = r - alpha * A p.
       e. Compute :math:`\beta = \langle r_{new}, r_{new}
          \rangle / \langle r_{old}, r_{old} \rangle`.
       f. Update p = r + beta * p.
    3. **Convergence check** --
       Freeze state once ||r|| < *tolerance*.

    Parameters
    ----------
    linear_operator : Callable[[PyTree], PyTree]
        Function computing A @ v for any vector v.  Must be
        symmetric PSD.
    rhs : PyTree
        Right-hand side vector b.
    x0 : PyTree
        Initial guess for the solution.
    max_iterations : int
        Maximum number of CG iterations.  Default 100.
    tolerance : float
        Convergence tolerance on residual norm.
        Default 1e-6.

    Returns
    -------
    solution : PyTree
        Approximate solution x satisfying A x ~ b.
    iterations : Int[Array, ""]
        Number of iterations performed.
    """
    initial_residual: PyTree = _tree_sub(rhs, linear_operator(x0))
    initial_r_dot_r: Float[Array, ""] = _tree_dot(
        initial_residual, initial_residual
    )

    initial_state: CGState = CGState(
        x=x0,
        r=initial_residual,
        p=initial_residual,
        r_dot_r=initial_r_dot_r,
        iteration=jnp.array(0),
    )

    tolerance_squared: Float[Array, ""] = jnp.array(tolerance**2)

    def cg_step(
        state: CGState,
        _: None,
    ) -> tuple[CGState, None]:
        """Execute one CG iteration."""
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

    final_state, _ = lax.scan(
        cg_step, initial_state, None, length=max_iterations
    )
    solution: PyTree = final_state.x
    iterations: Int[Array, ""] = final_state.iteration
    return solution, iterations


@jaxtyped(typechecker=beartype)
def gauss_newton_step(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    data: Float[Array, "..."],
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> tuple[PyTree, Float[Array, ""]]:
    r"""Compute a single Gauss-Newton update step.

    Extended Summary
    ----------------
    Solves the linearised normal equations
    :math:`J^\top J \, \delta = J^\top r` using CG, where *J*
    is the Jacobian at current params and *r* is the residual.

    Implementation Logic
    --------------------
    1. **Compute residual** --
       r = f(theta) - y.
    2. **Compute gradient** --
       g = J^T r via VJP.
    3. **Construct J^T J** --
       Build the normal equations operator.
    4. **Solve for step** --
       delta = CG(J^T J, g).
    5. **Update** --
       theta_new = theta - delta.
    6. **Evaluate** --
       Compute residual norm at theta_new.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to predictions.
    params : PyTree
        Current parameter estimate.
    data : Float[Array, "..."]
        Observed measurements.
    cg_max_iterations : int
        Maximum CG iterations for inner solve.  Default 50.
    cg_tolerance : float
        CG convergence tolerance.  Default 1e-6.

    Returns
    -------
    new_params : PyTree
        Updated parameter estimate after one GN step.
    residual_norm : Float[Array, ""]
        L2 norm of residual at *new_params*.

    See Also
    --------
    :func:`gauss_newton_solve` : Iterative wrapper.
    :func:`conjugate_gradient` : Inner linear solver.
    """
    prediction: Float[Array, "..."] = forward_fn(params)
    residual: Float[Array, "..."] = prediction - data
    vjp_fn: Callable = vjp_operator(forward_fn, params)
    gradient: PyTree = vjp_fn(residual)

    jtj_fn: Callable = jtj_operator(forward_fn, params)

    x0: PyTree = _tree_zeros_like(params)
    step, _ = conjugate_gradient(
        jtj_fn, gradient, x0, cg_max_iterations, cg_tolerance
    )

    new_params: PyTree = _tree_sub(params, step)
    new_prediction: Float[Array, "..."] = forward_fn(new_params)
    new_residual: Float[Array, "..."] = new_prediction - data
    residual_norm: Float[Array, ""] = jnp.sqrt(jnp.sum(new_residual**2))

    return new_params, residual_norm


@jaxtyped(typechecker=beartype)
def gauss_newton_solve(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params_init: PyTree,
    data: Float[Array, "..."],
    max_iterations: int = 20,
    tolerance: float = 1e-8,
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> tuple[PyTree, GNState]:
    """Solve nonlinear least-squares via iterated Gauss-Newton.

    Implementation Logic
    --------------------
    1. **Initialise** --
       Evaluate residual norm at *params_init*.
    2. **Iterate** --
       For each iteration, compute a GN step and check
       convergence.  Freeze state once the residual norm
       drops below *tolerance*.
    3. **Return** --
       Final parameters and the :class:`GNState`.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to predictions.
    params_init : PyTree
        Initial parameter guess.
    data : Float[Array, "..."]
        Observed measurements.
    max_iterations : int
        Maximum number of GN iterations.  Default 20.
    tolerance : float
        Convergence tolerance on residual norm.
        Default 1e-8.
    cg_max_iterations : int
        Maximum CG iterations per GN step.  Default 50.
    cg_tolerance : float
        CG convergence tolerance.  Default 1e-6.

    Returns
    -------
    params_final : PyTree
        Optimised parameters.
    final_state : GNState
        Final optimisation state including residual norm
        and iteration count.

    See Also
    --------
    :func:`gauss_newton_step` : Single-step primitive.
    :func:`levenberg_marquardt_solve` : Damped variant.
    """
    initial_prediction: Float[Array, "..."] = forward_fn(params_init)
    initial_residual: Float[Array, "..."] = initial_prediction - data
    initial_norm: Float[Array, ""] = jnp.sqrt(jnp.sum(initial_residual**2))

    initial_state: GNState = GNState(
        params=params_init,
        residual_norm=initial_norm,
        iteration=jnp.array(0),
    )

    def gn_iteration(
        state: GNState,
        _: None,
    ) -> tuple[GNState, None]:
        """Execute one GN iteration with convergence check."""
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

    final_state, _ = lax.scan(
        gn_iteration, initial_state, None, length=max_iterations
    )
    params_final: PyTree = final_state.params
    return params_final, final_state


@jaxtyped(typechecker=beartype)
def levenberg_marquardt_step(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    data: Float[Array, "..."],
    damping: Float[Array, ""],
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> tuple[PyTree, Float[Array, ""], Float[Array, ""]]:
    r"""Compute a single Levenberg-Marquardt update step.

    Extended Summary
    ----------------
    LM interpolates between Gauss-Newton (small damping) and
    gradient descent (large damping).  Solves
    :math:`(J^\top J + \lambda I)\,\delta = J^\top r`.

    Implementation Logic
    --------------------
    1. **Compute residual and gradient** --
       r = f(theta) - y, g = J^T r.
    2. **Build damped operator** --
       A = J^T J + lambda * I.
    3. **Solve for step** --
       delta = CG(A, g).
    4. **Gain ratio** --
       rho = actual_reduction / predicted_reduction.
    5. **Accept or reject** --
       Accept step if rho > 0.
    6. **Adapt damping** --
       Decrease if rho > 0.75, increase if rho < 0.25.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to predictions.
    params : PyTree
        Current parameter estimate.
    data : Float[Array, "..."]
        Observed measurements.
    damping : Float[Array, ""]
        Damping parameter :math:`\lambda`.  Larger values
        produce more regularised steps.
    cg_max_iterations : int
        Maximum CG iterations for inner solve.  Default 50.
    cg_tolerance : float
        CG convergence tolerance.  Default 1e-6.

    Returns
    -------
    new_params : PyTree
        Updated parameter estimate.
    new_residual_norm : Float[Array, ""]
        Residual norm at *new_params*.
    new_damping : Float[Array, ""]
        Adapted damping for next iteration.

    See Also
    --------
    :func:`levenberg_marquardt_solve` : Iterative wrapper.
    """
    prediction: Float[Array, "..."] = forward_fn(params)
    residual: Float[Array, "..."] = prediction - data
    residual_norm_current: Float[Array, ""] = jnp.sum(residual**2)

    vjp_fn: Callable = vjp_operator(forward_fn, params)
    gradient: PyTree = vjp_fn(residual)
    jtj_fn: Callable = jtj_operator(forward_fn, params)

    def damped_operator(v: PyTree) -> PyTree:
        """Apply (J^T J + lambda I) to v."""
        jtj_v: PyTree = jtj_fn(v)
        damped_term: PyTree = _tree_scalar_mul(damping, v)
        result: PyTree = _tree_add(jtj_v, damped_term)
        return result

    x0: PyTree = _tree_zeros_like(params)
    step, _ = conjugate_gradient(
        damped_operator, gradient, x0, cg_max_iterations, cg_tolerance
    )

    candidate_params: PyTree = _tree_sub(params, step)
    new_prediction: Float[Array, "..."] = forward_fn(candidate_params)
    new_residual: Float[Array, "..."] = new_prediction - data
    new_residual_sum_squares: Float[Array, ""] = jnp.sum(new_residual**2)

    actual_reduction: Float[Array, ""] = (
        residual_norm_current - new_residual_sum_squares
    )
    jtj_step: PyTree = jtj_fn(step)
    predicted_reduction: Float[Array, ""] = _tree_dot(
        step, gradient
    ) - 0.5 * _tree_dot(step, jtj_step)
    gain_ratio: Float[Array, ""] = actual_reduction / (
        predicted_reduction + 1e-12
    )

    gain_upper: float = 0.75
    gain_lower: float = 0.25
    damping_decrease: Float[Array, ""] = damping / 3.0
    damping_increase: Float[Array, ""] = damping * 2.0
    new_damping: Float[Array, ""] = lax.cond(
        gain_ratio > gain_upper,
        lambda: damping_decrease,
        lambda: lax.cond(
            gain_ratio < gain_lower,
            lambda: damping_increase,
            lambda: damping,
        ),
    )

    new_params: PyTree = lax.cond(
        gain_ratio > 0.0,
        lambda: candidate_params,
        lambda: params,
    )
    new_residual_norm: Float[Array, ""] = lax.cond(
        gain_ratio > 0.0,
        lambda: jnp.sqrt(new_residual_sum_squares),
        lambda: jnp.sqrt(residual_norm_current),
    )

    return new_params, new_residual_norm, new_damping


@jaxtyped(typechecker=beartype)
def levenberg_marquardt_solve(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params_init: PyTree,
    data: Float[Array, "..."],
    max_iterations: int = 50,
    tolerance: float = 1e-8,
    damping_init: float = 1e-3,
    cg_max_iterations: int = 50,
    cg_tolerance: float = 1e-6,
) -> tuple[PyTree, LMState]:
    """Solve nonlinear least-squares via Levenberg-Marquardt.

    Extended Summary
    ----------------
    LM is more robust than pure Gauss-Newton for ill-conditioned
    problems or when far from the solution.  Adaptively adjusts
    the damping parameter between iterations.

    Implementation Logic
    --------------------
    1. **Initialise** --
       Evaluate residual norm and set initial damping.
    2. **Iterate** --
       For each iteration, compute an LM step with adaptive
       damping.  Freeze state once converged.
    3. **Return** --
       Final parameters and the :class:`LMState`.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to predictions.
    params_init : PyTree
        Initial parameter guess.
    data : Float[Array, "..."]
        Observed measurements.
    max_iterations : int
        Maximum number of LM iterations.  Default 50.
    tolerance : float
        Convergence tolerance on residual norm.
        Default 1e-8.
    damping_init : float
        Initial damping parameter.  Default 1e-3.
    cg_max_iterations : int
        Maximum CG iterations per LM step.  Default 50.
    cg_tolerance : float
        CG convergence tolerance.  Default 1e-6.

    Returns
    -------
    params_final : PyTree
        Optimised parameters.
    final_state : LMState
        Final optimisation state.

    See Also
    --------
    :func:`levenberg_marquardt_step` : Single-step primitive.
    :func:`gauss_newton_solve` : Undamped variant.
    """
    initial_prediction: Float[Array, "..."] = forward_fn(params_init)
    initial_residual: Float[Array, "..."] = initial_prediction - data
    initial_norm: Float[Array, ""] = jnp.sqrt(jnp.sum(initial_residual**2))

    initial_state: LMState = LMState(
        params=params_init,
        residual_norm=initial_norm,
        damping=jnp.array(damping_init),
        iteration=jnp.array(0),
    )

    def lm_iteration(
        state: LMState,
        _: None,
    ) -> tuple[LMState, None]:
        """Execute one LM iteration with convergence check."""
        new_params, new_norm, new_damping = levenberg_marquardt_step(
            forward_fn,
            state.params,
            data,
            state.damping,
            cg_max_iterations,
            cg_tolerance,
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

    final_state, _ = lax.scan(
        lm_iteration, initial_state, None, length=max_iterations
    )
    params_final: PyTree = final_state.params
    return params_final, final_state


@jaxtyped(typechecker=beartype)
def lanczos_tridiagonal(
    linear_operator: Callable[[Float[Array, "n"]], Float[Array, "n"]],
    initial_vector: Float[Array, "n"],
    num_iterations: int,
) -> tuple[Float[Array, "k"], Float[Array, "k-1"]]:
    """Compute the Lanczos tridiagonalisation of a symmetric operator.

    Extended Summary
    ----------------
    The Lanczos algorithm builds an orthonormal basis for the
    Krylov subspace and produces a tridiagonal matrix whose
    eigenvalues approximate the extremal eigenvalues of the
    original operator.

    Implementation Logic
    --------------------
    1. **Normalise starting vector** --
       v_0 = initial_vector / ||initial_vector||.
    2. **Lanczos loop** --
       For each iteration:

       a. Apply operator: w = A v_curr.
       b. Compute diagonal: alpha = <w, v_curr>.
       c. Orthogonalise: w = w - alpha v_curr - beta v_prev.
       d. Compute off-diagonal: beta = ||w||.
       e. Normalise: v_next = w / beta.
    3. **Return** --
       Diagonal (alpha) and off-diagonal (beta) arrays.

    Parameters
    ----------
    linear_operator : Callable[[Float[Array, "n"]], Float[Array, "n"]]
        Symmetric linear operator A.  Must satisfy
        <A x, y> = <x, A y>.
    initial_vector : Float[Array, "n"]
        Starting vector for Krylov subspace.  Should be
        random or non-sparse.
    num_iterations : int
        Number of Lanczos iterations.  Determines size of
        tridiagonal matrix.

    Returns
    -------
    alpha : Float[Array, "k"]
        Diagonal elements of the tridiagonal matrix.
    beta : Float[Array, "k-1"]
        Off-diagonal elements of the tridiagonal matrix.
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
    ) -> tuple[LanczosState, None]:
        """Execute one Lanczos iteration."""
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

        new_alpha: Float[Array, "k"] = state.alpha.at[state.iteration].set(
            alpha_i
        )
        new_beta: Float[Array, "k"] = state.beta.at[state.iteration].set(
            beta_i
        )

        new_state: LanczosState = LanczosState(
            v_prev=state.v_curr,
            v_curr=v_next,
            alpha=new_alpha,
            beta=new_beta,
            iteration=state.iteration + 1,
        )
        return new_state, None

    final_state, _ = lax.scan(lanczos_step, initial_state, None, length=k)

    alpha: Float[Array, "k"] = final_state.alpha
    beta: Float[Array, "k-1"] = final_state.beta[:-1]

    return alpha, beta


@jaxtyped(typechecker=beartype)
def singular_spectrum(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    num_singular_values: int = 50,
    num_lanczos_iterations: int = 100,
    random_seed: int = 42,
) -> Float[Array, "k"]:
    r"""Estimate the singular spectrum of the Jacobian.

    Extended Summary
    ----------------
    The singular values :math:`\sigma_i` of *J* equal the
    square roots of the eigenvalues of :math:`J^\top J`.  The
    Lanczos algorithm approximates extremal eigenvalues well,
    giving accurate estimates of the largest and smallest
    singular values.

    Implementation Logic
    --------------------
    1. **Flatten parameters** --
       Ravel to vector form.
    2. **Construct J^T J** --
       Build a flattened-space operator.
    3. **Run Lanczos** --
       Build the tridiagonal matrix.
    4. **Eigendecompose** --
       Compute eigenvalues of the tridiagonal matrix.
    5. **Square-root and sort** --
       Convert to singular values, sort descending,
       return top *num_singular_values*.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to predictions.
    params : PyTree
        Point at which to evaluate the Jacobian.
    num_singular_values : int
        Number of singular values to estimate.  Default 50.
    num_lanczos_iterations : int
        Lanczos iterations.  Should be >=
        *num_singular_values*.  Default 100.
    random_seed : int
        Seed for random starting vector.  Default 42.

    Returns
    -------
    singular_values : Float[Array, "k"]
        Estimated singular values in descending order.

    See Also
    --------
    :func:`lanczos_tridiagonal` : Core Lanczos routine.
    :func:`effective_nullspace_dimension` : Threshold
        counting.
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]

    jtj_pytree_fn: Callable = jtj_operator(forward_fn, params)

    def jtj_flat_fn(
        v_flat: Float[Array, "n"],
    ) -> Float[Array, "n"]:
        """Apply J^T J in flattened space."""
        v_pytree: PyTree = unflatten_fn(v_flat)
        result_pytree: PyTree = jtj_pytree_fn(v_pytree)
        result_flat: Float[Array, "n"] = jax.flatten_util.ravel_pytree(
            result_pytree
        )[0]
        return result_flat

    key: PRNGKeyArray = jax.random.PRNGKey(random_seed)
    initial_vector: Float[Array, "n"] = jax.random.normal(key, (n,))

    alpha, beta = lanczos_tridiagonal(
        jtj_flat_fn, initial_vector, num_lanczos_iterations
    )

    tridiag_matrix: Float[Array, "k k"] = (
        jnp.diag(alpha) + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
    )

    eigenvalues: Float[Array, "k"] = jnp.linalg.eigvalsh(tridiag_matrix)
    eigenvalues_positive: Float[Array, "k"] = jnp.maximum(eigenvalues, 0.0)
    singular_values_all: Float[Array, "k"] = jnp.sqrt(eigenvalues_positive)
    singular_values_sorted: Float[Array, "k"] = jnp.sort(singular_values_all)[
        ::-1
    ]
    singular_values: Float[Array, "k"] = singular_values_sorted[
        :num_singular_values
    ]

    return singular_values


@jaxtyped(typechecker=beartype)
def effective_nullspace_dimension(
    singular_values: Float[Array, "k"],
    noise_floor: float,
) -> Int[Array, ""]:
    r"""Count singular values below the noise floor.

    Extended Summary
    ----------------
    Directions with :math:`\sigma_i < \eta` are effectively
    unobservable: they contribute more noise amplification than
    signal recovery.  This count gives the effective dimension
    of the gauge subspace under finite SNR.

    Implementation Logic
    --------------------
    1. **Threshold** --
       Compare each singular value against *noise_floor*.
    2. **Count** --
       Sum the number below threshold.

    Parameters
    ----------
    singular_values : Float[Array, "k"]
        Singular values of the Jacobian (or estimates).
    noise_floor : float
        Noise threshold :math:`\eta`.  Directions with
        :math:`\sigma < \eta` are effectively null.

    Returns
    -------
    nullspace_dim : Int[Array, ""]
        Number of singular values below *noise_floor*.

    See Also
    --------
    :func:`singular_spectrum` : Spectrum estimation.
    """
    below_threshold: Float[Array, "k"] = (
        singular_values < noise_floor
    ).astype(jnp.int32)
    nullspace_dim: Int[Array, ""] = jnp.sum(below_threshold)
    return nullspace_dim


__all__: list[str] = [
    "conjugate_gradient",
    "effective_nullspace_dimension",
    "gauss_newton_solve",
    "gauss_newton_step",
    "lanczos_tridiagonal",
    "levenberg_marquardt_solve",
    "levenberg_marquardt_step",
    "singular_spectrum",
]
