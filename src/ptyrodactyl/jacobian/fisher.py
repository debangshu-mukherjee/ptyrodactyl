"""
Module: ptyrodactyl.jacobian.fisher
-----------------------------------

Fisher information for ptychographic experiment design.

The Fisher information matrix F = JᵀWJ quantifies how much information
measurements provide about parameters. Its eigenspectrum reveals which
parameter combinations are well-constrained versus poorly constrained.
This module provides tools for computing, analyzing, and optimizing
Fisher information across experimental conditions.

Functions
---------
- `fisher_information`:
    Compute Fisher information matrix at a parameter point
- `fisher_information_operator`:
    Matrix-free Fisher information operator for large problems
- `fisher_diagonal`:
    Fast diagonal approximation of Fisher information
- `schur_complement`:
    Marginalize nuisance parameters via Schur complement
- `effective_fisher`:
    Fisher information after marginalizing nuisances
- `fisher_eigenspectrum`:
    Eigenvalues of Fisher matrix via Lanczos
- `a_optimality`:
    A-optimality criterion: trace(F⁻¹)
- `d_optimality`:
    D-optimality criterion: det(F)
- `e_optimality`:
    E-optimality criterion: λ_min(F)
- `stack_fisher`:
    Combine Fisher matrices from multiple experimental conditions
- `optimal_weights`:
    Compute optimal weights for stacking experiments
"""

from typing import Callable, Tuple, NamedTuple
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Float, Int, Array, PyTree

from ptyrodactyl.jacobian.operators import jtj_operator, jvp_operator, vjp_operator


class FisherState(NamedTuple):
    """State for iterative Fisher computation."""
    fisher_matrix: Float[Array, "n n"]
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


def fisher_information(
    forward_fn: Callable[[PyTree], Float[Array, "m"]],
    params: PyTree,
    noise_variance: Float[Array, ""] = jnp.array(1.0),
) -> Float[Array, "n n"]:
    """
    Description
    -----------
    Compute the Fisher information matrix F = (1/σ²) JᵀJ.

    For Gaussian noise with variance σ², the Fisher information equals
    the inverse noise variance times the Gramian of the Jacobian. This
    matrix encodes the information content of measurements about parameters.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "m"]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate Fisher information.
    - `noise_variance` (Float[Array, ""]):
        Measurement noise variance σ². Default 1.0.

    Returns
    -------
    - `fisher_matrix` (Float[Array, "n n"]):
        Fisher information matrix of shape (n_params, n_params).

    Flow
    ----
    1. Flatten params to get dimension n
    2. Compute full Jacobian via vmap over basis vectors
    3. Form JᵀJ and scale by 1/σ²
    4. Return Fisher matrix
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]

    def jacobian_column(
        index: int,
    ) -> Float[Array, "m"]:
        basis_vector: Float[Array, "n"] = jnp.zeros(n).at[index].set(1.0)
        tangent_pytree: PyTree = unflatten_fn(basis_vector)
        _, jvp_result = jax.jvp(forward_fn, (params,), (tangent_pytree,))
        return jvp_result

    jacobian_matrix: Float[Array, "m n"] = jax.vmap(jacobian_column)(jnp.arange(n)).T
    jtj: Float[Array, "n n"] = jnp.dot(jacobian_matrix.T, jacobian_matrix)
    fisher_matrix: Float[Array, "n n"] = jtj / noise_variance

    return fisher_matrix


def fisher_information_operator(
    forward_fn: Callable[[PyTree], Float[Array, "m"]],
    params: PyTree,
    noise_variance: Float[Array, ""] = jnp.array(1.0),
) -> Callable[[PyTree], PyTree]:
    """
    Description
    -----------
    Construct matrix-free Fisher information operator F @ v = (1/σ²) JᵀJ @ v.

    For large parameter spaces where forming the full matrix is infeasible,
    this returns an operator that applies F to vectors without materialization.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "m"]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate Fisher information.
    - `noise_variance` (Float[Array, ""]):
        Measurement noise variance σ². Default 1.0.

    Returns
    -------
    - `fisher_op` (Callable[[PyTree], PyTree]):
        Operator that computes F @ v for any vector v.

    Flow
    ----
    1. Construct JᵀJ operator
    2. Wrap with 1/σ² scaling
    3. Return composed operator
    """
    jtj_fn: Callable = jtj_operator(forward_fn, params)
    inv_variance: Float[Array, ""] = 1.0 / noise_variance

    def fisher_op(
        vector: PyTree,
    ) -> PyTree:
        jtj_v: PyTree = jtj_fn(vector)
        result: PyTree = jax.tree_util.tree_map(lambda x: inv_variance * x, jtj_v)
        return result

    return fisher_op


def fisher_diagonal(
    forward_fn: Callable[[PyTree], Float[Array, "m"]],
    params: PyTree,
    noise_variance: Float[Array, ""] = jnp.array(1.0),
    num_hutchinson_samples: int = 30,
    random_seed: int = 42,
) -> PyTree:
    """
    Description
    -----------
    Estimate diagonal of Fisher information via Hutchinson's estimator.

    The diagonal F_ii indicates the information about each parameter
    individually. This is faster than computing the full matrix.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "m"]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate Fisher information.
    - `noise_variance` (Float[Array, ""]):
        Measurement noise variance σ². Default 1.0.
    - `num_hutchinson_samples` (int):
        Number of random vectors for Hutchinson estimator. Default 30.
    - `random_seed` (int):
        Seed for random vectors. Default 42.

    Returns
    -------
    - `fisher_diag` (PyTree):
        Diagonal of Fisher matrix as PyTree matching params structure.

    Flow
    ----
    1. Generate random Rademacher vectors
    2. For each vector z: compute z ⊙ (JᵀJ @ z)
    3. Average over samples to estimate diagonal
    4. Scale by 1/σ²
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]

    jtj_fn: Callable = jtj_operator(forward_fn, params)

    def jtj_flat_fn(
        v_flat: Float[Array, "n"],
    ) -> Float[Array, "n"]:
        v_pytree: PyTree = unflatten_fn(v_flat)
        result_pytree: PyTree = jtj_fn(v_pytree)
        result_flat: Float[Array, "n"] = jax.flatten_util.ravel_pytree(result_pytree)[0]
        return result_flat

    key: jax.random.PRNGKey = jax.random.PRNGKey(random_seed)

    def hutchinson_sample(
        carry: Float[Array, "n"],
        key_i: jax.random.PRNGKey,
    ) -> Tuple[Float[Array, "n"], None]:
        z: Float[Array, "n"] = jax.random.rademacher(key_i, (n,)).astype(jnp.float32)
        jtj_z: Float[Array, "n"] = jtj_flat_fn(z)
        sample: Float[Array, "n"] = z * jtj_z
        new_carry: Float[Array, "n"] = carry + sample
        return new_carry, None

    keys: jax.random.PRNGKey = jax.random.split(key, num_hutchinson_samples)
    diagonal_sum, _ = lax.scan(hutchinson_sample, jnp.zeros(n), keys)
    diagonal_mean: Float[Array, "n"] = diagonal_sum / num_hutchinson_samples
    fisher_diag_flat: Float[Array, "n"] = diagonal_mean / noise_variance
    fisher_diag: PyTree = unflatten_fn(fisher_diag_flat)

    return fisher_diag


def schur_complement(
    full_matrix: Float[Array, "n n"],
    num_params_of_interest: int,
) -> Float[Array, "p p"]:
    """
    Description
    -----------
    Compute Schur complement to marginalize nuisance parameters.

    Given a block matrix [[A, B], [C, D]] where A corresponds to parameters
    of interest and D to nuisance parameters, the Schur complement
    A - B D⁻¹ C gives the effective information about parameters of
    interest after marginalizing out nuisances.

    Parameters
    ----------
    - `full_matrix` (Float[Array, "n n"]):
        Full Fisher information matrix.
    - `num_params_of_interest` (int):
        Number of parameters of interest (first p parameters).
        Remaining (n - p) are treated as nuisances.

    Returns
    -------
    - `schur` (Float[Array, "p p"]):
        Schur complement matrix for parameters of interest.

    Flow
    ----
    1. Extract blocks A, B, C, D from full matrix
    2. Compute D⁻¹ (with regularization for stability)
    3. Compute Schur complement A - B D⁻¹ C
    4. Return marginalized information matrix
    """
    p: int = num_params_of_interest
    n: int = full_matrix.shape[0]

    a_block: Float[Array, "p p"] = full_matrix[:p, :p]
    b_block: Float[Array, "p n-p"] = full_matrix[:p, p:]
    c_block: Float[Array, "n-p p"] = full_matrix[p:, :p]
    d_block: Float[Array, "n-p n-p"] = full_matrix[p:, p:]

    d_reg: Float[Array, "n-p n-p"] = d_block + 1e-8 * jnp.eye(n - p)
    d_inv: Float[Array, "n-p n-p"] = jnp.linalg.inv(d_reg)

    correction: Float[Array, "p p"] = jnp.dot(b_block, jnp.dot(d_inv, c_block))
    schur: Float[Array, "p p"] = a_block - correction

    return schur


def effective_fisher(
    forward_fn: Callable[[PyTree], Float[Array, "m"]],
    params_interest: PyTree,
    params_nuisance: PyTree,
    noise_variance: Float[Array, ""] = jnp.array(1.0),
) -> Float[Array, "p p"]:
    """
    Description
    -----------
    Compute Fisher information for parameters of interest, marginalizing nuisances.

    This implements F_eff = F_gg - F_gn F_nn⁻¹ F_ng where g denotes parameters
    of interest and n denotes nuisance parameters (probe errors, position
    errors, drift, etc.).

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "m"]]):
        Forward model taking (params_interest, params_nuisance) tuple.
    - `params_interest` (PyTree):
        Parameters of interest (e.g., object potential).
    - `params_nuisance` (PyTree):
        Nuisance parameters (e.g., probe, positions).
    - `noise_variance` (Float[Array, ""]):
        Measurement noise variance σ². Default 1.0.

    Returns
    -------
    - `fisher_eff` (Float[Array, "p p"]):
        Effective Fisher information for parameters of interest.

    Flow
    ----
    1. Combine params into single PyTree
    2. Compute full Fisher matrix
    3. Apply Schur complement to marginalize nuisances
    4. Return effective information
    """
    flat_interest, unflatten_interest = jax.flatten_util.ravel_pytree(params_interest)
    flat_nuisance, unflatten_nuisance = jax.flatten_util.ravel_pytree(params_nuisance)
    p: int = flat_interest.shape[0]
    q: int = flat_nuisance.shape[0]
    n: int = p + q

    def combined_forward(
        combined_params: Float[Array, "n"],
    ) -> Float[Array, "m"]:
        interest_part: Float[Array, "p"] = combined_params[:p]
        nuisance_part: Float[Array, "q"] = combined_params[p:]
        params_i: PyTree = unflatten_interest(interest_part)
        params_n: PyTree = unflatten_nuisance(nuisance_part)
        return forward_fn((params_i, params_n))

    combined_params: Float[Array, "n"] = jnp.concatenate([flat_interest, flat_nuisance])

    def jacobian_column(
        index: int,
    ) -> Float[Array, "m"]:
        basis_vector: Float[Array, "n"] = jnp.zeros(n).at[index].set(1.0)
        _, jvp_result = jax.jvp(combined_forward, (combined_params,), (basis_vector,))
        return jvp_result

    jacobian_matrix: Float[Array, "m n"] = jax.vmap(jacobian_column)(jnp.arange(n)).T
    jtj: Float[Array, "n n"] = jnp.dot(jacobian_matrix.T, jacobian_matrix)
    full_fisher: Float[Array, "n n"] = jtj / noise_variance

    fisher_eff: Float[Array, "p p"] = schur_complement(full_fisher, p)

    return fisher_eff


def fisher_eigenspectrum(
    forward_fn: Callable[[PyTree], Float[Array, "m"]],
    params: PyTree,
    noise_variance: Float[Array, ""] = jnp.array(1.0),
    num_eigenvalues: int = 50,
    num_lanczos_iterations: int = 100,
    random_seed: int = 42,
) -> Float[Array, "k"]:
    """
    Description
    -----------
    Estimate eigenspectrum of Fisher information via Lanczos.

    The eigenvalues indicate how much information measurements provide
    about different parameter combinations. Small eigenvalues correspond
    to poorly constrained directions.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "m"]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate Fisher information.
    - `noise_variance` (Float[Array, ""]):
        Measurement noise variance σ². Default 1.0.
    - `num_eigenvalues` (int):
        Number of eigenvalues to estimate. Default 50.
    - `num_lanczos_iterations` (int):
        Lanczos iterations. Default 100.
    - `random_seed` (int):
        Seed for random starting vector. Default 42.

    Returns
    -------
    - `eigenvalues` (Float[Array, "k"]):
        Estimated eigenvalues in descending order.

    Flow
    ----
    1. Construct matrix-free Fisher operator
    2. Run Lanczos to build tridiagonal matrix
    3. Compute eigenvalues of tridiagonal matrix
    4. Sort descending and return top k
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]
    k: int = num_lanczos_iterations

    fisher_op: Callable = fisher_information_operator(forward_fn, params, noise_variance)

    def fisher_flat_fn(
        v_flat: Float[Array, "n"],
    ) -> Float[Array, "n"]:
        v_pytree: PyTree = unflatten_fn(v_flat)
        result_pytree: PyTree = fisher_op(v_pytree)
        result_flat: Float[Array, "n"] = jax.flatten_util.ravel_pytree(result_pytree)[0]
        return result_flat

    key: jax.random.PRNGKey = jax.random.PRNGKey(random_seed)
    v0: Float[Array, "n"] = jax.random.normal(key, (n,))
    v0_normalized: Float[Array, "n"] = v0 / jnp.linalg.norm(v0)

    alpha: Float[Array, "k"] = jnp.zeros(k)
    beta: Float[Array, "k"] = jnp.zeros(k)
    v_prev: Float[Array, "n"] = jnp.zeros(n)
    v_curr: Float[Array, "n"] = v0_normalized

    def lanczos_step(
        iteration: int,
        carry: Tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "k"], Float[Array, "k"]],
    ) -> Tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "k"], Float[Array, "k"]]:
        v_p, v_c, alphas, betas = carry
        w: Float[Array, "n"] = fisher_flat_fn(v_c)
        alpha_i: Float[Array, ""] = jnp.dot(w, v_c)

        beta_prev: Float[Array, ""] = lax.cond(
            iteration > 0,
            lambda: betas[iteration - 1],
            lambda: jnp.array(0.0),
        )

        w_orth: Float[Array, "n"] = w - alpha_i * v_c - beta_prev * v_p
        beta_i: Float[Array, ""] = jnp.linalg.norm(w_orth)
        v_next: Float[Array, "n"] = w_orth / (beta_i + 1e-12)

        alphas_new: Float[Array, "k"] = alphas.at[iteration].set(alpha_i)
        betas_new: Float[Array, "k"] = betas.at[iteration].set(beta_i)

        return (v_c, v_next, alphas_new, betas_new)

    _, _, alpha, beta = lax.fori_loop(0, k, lanczos_step, (v_prev, v_curr, alpha, beta))

    tridiag_matrix: Float[Array, "k k"] = (
        jnp.diag(alpha) +
        jnp.diag(beta[:-1], k=1) +
        jnp.diag(beta[:-1], k=-1)
    )

    eigenvalues_all: Float[Array, "k"] = jnp.linalg.eigvalsh(tridiag_matrix)
    eigenvalues_sorted: Float[Array, "k"] = jnp.sort(eigenvalues_all)[::-1]
    eigenvalues_out: Float[Array, "k"] = eigenvalues_sorted[:num_eigenvalues]

    return eigenvalues_out


def a_optimality(
    fisher_matrix: Float[Array, "n n"],
    regularization: float = 1e-8,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute A-optimality criterion: trace(F⁻¹).

    A-optimality minimizes the average variance of parameter estimates.
    Lower values indicate better experimental design.

    Parameters
    ----------
    - `fisher_matrix` (Float[Array, "n n"]):
        Fisher information matrix.
    - `regularization` (float):
        Small value added to diagonal for numerical stability. Default 1e-8.

    Returns
    -------
    - `criterion` (Float[Array, ""]):
        A-optimality value (trace of inverse Fisher).

    Flow
    ----
    1. Add regularization to Fisher matrix
    2. Compute inverse
    3. Return trace
    """
    n: int = fisher_matrix.shape[0]
    fisher_reg: Float[Array, "n n"] = fisher_matrix + regularization * jnp.eye(n)
    fisher_inv: Float[Array, "n n"] = jnp.linalg.inv(fisher_reg)
    criterion: Float[Array, ""] = jnp.trace(fisher_inv)
    return criterion


def d_optimality(
    fisher_matrix: Float[Array, "n n"],
    regularization: float = 1e-8,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute D-optimality criterion: log det(F).

    D-optimality maximizes the determinant of Fisher information,
    minimizing the volume of the confidence ellipsoid. Higher values
    indicate better experimental design.

    Parameters
    ----------
    - `fisher_matrix` (Float[Array, "n n"]):
        Fisher information matrix.
    - `regularization` (float):
        Small value added to diagonal for numerical stability. Default 1e-8.

    Returns
    -------
    - `criterion` (Float[Array, ""]):
        D-optimality value (log determinant of Fisher).

    Flow
    ----
    1. Add regularization to Fisher matrix
    2. Compute log determinant via eigenvalues
    3. Return log det
    """
    n: int = fisher_matrix.shape[0]
    fisher_reg: Float[Array, "n n"] = fisher_matrix + regularization * jnp.eye(n)
    eigenvalues: Float[Array, "n"] = jnp.linalg.eigvalsh(fisher_reg)
    log_det: Float[Array, ""] = jnp.sum(jnp.log(jnp.maximum(eigenvalues, 1e-12)))
    return log_det


def e_optimality(
    fisher_matrix: Float[Array, "n n"],
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute E-optimality criterion: λ_min(F).

    E-optimality maximizes the minimum eigenvalue of Fisher information,
    ensuring no parameter direction is poorly constrained. Higher values
    indicate better experimental design.

    Parameters
    ----------
    - `fisher_matrix` (Float[Array, "n n"]):
        Fisher information matrix.

    Returns
    -------
    - `criterion` (Float[Array, ""]):
        E-optimality value (minimum eigenvalue).

    Flow
    ----
    1. Compute eigenvalues
    2. Return minimum eigenvalue
    """
    eigenvalues: Float[Array, "n"] = jnp.linalg.eigvalsh(fisher_matrix)
    criterion: Float[Array, ""] = jnp.min(eigenvalues)
    return criterion


def stack_fisher(
    fisher_matrices: Float[Array, "k n n"],
    weights: Float[Array, "k"],
) -> Float[Array, "n n"]:
    """
    Description
    -----------
    Combine Fisher matrices from multiple experimental conditions.

    When measurements are independent, Fisher information is additive.
    Weighted stacking allows optimizing the allocation of measurement
    effort across conditions.

    Parameters
    ----------
    - `fisher_matrices` (Float[Array, "k n n"]):
        Fisher matrices from k experimental conditions.
    - `weights` (Float[Array, "k"]):
        Non-negative weights for each condition (e.g., exposure times).

    Returns
    -------
    - `combined_fisher` (Float[Array, "n n"]):
        Weighted sum of Fisher matrices.

    Flow
    ----
    1. Multiply each matrix by its weight
    2. Sum over conditions
    3. Return combined Fisher matrix
    """
    weighted_matrices: Float[Array, "k n n"] = fisher_matrices * weights[:, None, None]
    combined_fisher: Float[Array, "n n"] = jnp.sum(weighted_matrices, axis=0)
    return combined_fisher


def optimal_weights_e_criterion(
    fisher_matrices: Float[Array, "k n n"],
    num_iterations: int = 100,
    learning_rate: float = 0.1,
) -> Float[Array, "k"]:
    """
    Description
    -----------
    Find optimal weights for stacking experiments under E-optimality.

    Solves: max_{w} λ_min(Σ_i w_i F_i) subject to Σ_i w_i = 1, w_i ≥ 0.

    This determines how to allocate measurement effort across experimental
    conditions to maximize the worst-case information.

    Parameters
    ----------
    - `fisher_matrices` (Float[Array, "k n n"]):
        Fisher matrices from k experimental conditions.
    - `num_iterations` (int):
        Number of optimization iterations. Default 100.
    - `learning_rate` (float):
        Step size for projected gradient ascent. Default 0.1.

    Returns
    -------
    - `optimal_weights` (Float[Array, "k"]):
        Optimal weights summing to 1.

    Flow
    ----
    1. Initialize uniform weights
    2. For each iteration:
       a. Compute combined Fisher
       b. Find minimum eigenvector
       c. Compute gradient of λ_min w.r.t. weights
       d. Take gradient step and project to simplex
    3. Return optimal weights
    """
    k: int = fisher_matrices.shape[0]
    n: int = fisher_matrices.shape[1]

    initial_weights: Float[Array, "k"] = jnp.ones(k) / k

    def project_simplex(
        weights: Float[Array, "k"],
    ) -> Float[Array, "k"]:
        weights_clipped: Float[Array, "k"] = jnp.maximum(weights, 0.0)
        weights_sum: Float[Array, ""] = jnp.sum(weights_clipped)
        weights_normalized: Float[Array, "k"] = weights_clipped / (weights_sum + 1e-12)
        return weights_normalized

    def optimization_step(
        iteration: int,
        weights: Float[Array, "k"],
    ) -> Float[Array, "k"]:
        combined_fisher: Float[Array, "n n"] = stack_fisher(fisher_matrices, weights)
        eigenvalues, eigenvectors = jnp.linalg.eigh(combined_fisher)
        min_idx: Int[Array, ""] = jnp.argmin(eigenvalues)
        min_eigenvector: Float[Array, "n"] = eigenvectors[:, min_idx]

        def gradient_component(
            fisher_i: Float[Array, "n n"],
        ) -> Float[Array, ""]:
            return jnp.dot(min_eigenvector, jnp.dot(fisher_i, min_eigenvector))

        gradient: Float[Array, "k"] = jax.vmap(gradient_component)(fisher_matrices)
        weights_updated: Float[Array, "k"] = weights + learning_rate * gradient
        weights_projected: Float[Array, "k"] = project_simplex(weights_updated)
        return weights_projected

    optimal_weights: Float[Array, "k"] = lax.fori_loop(
        0, num_iterations, optimization_step, initial_weights
    )

    return optimal_weights


def condition_number(
    fisher_matrix: Float[Array, "n n"],
    regularization: float = 1e-12,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute condition number of Fisher information matrix.

    The condition number κ = λ_max / λ_min indicates how ill-posed the
    inverse problem is. Large condition numbers mean some directions
    are much harder to recover than others.

    Parameters
    ----------
    - `fisher_matrix` (Float[Array, "n n"]):
        Fisher information matrix.
    - `regularization` (float):
        Floor for minimum eigenvalue. Default 1e-12.

    Returns
    -------
    - `kappa` (Float[Array, ""]):
        Condition number.

    Flow
    ----
    1. Compute eigenvalues
    2. Find max and min eigenvalues
    3. Return ratio
    """
    eigenvalues: Float[Array, "n"] = jnp.linalg.eigvalsh(fisher_matrix)
    lambda_max: Float[Array, ""] = jnp.max(eigenvalues)
    lambda_min: Float[Array, ""] = jnp.maximum(jnp.min(eigenvalues), regularization)
    kappa: Float[Array, ""] = lambda_max / lambda_min
    return kappa


def information_gain(
    fisher_before: Float[Array, "n n"],
    fisher_after: Float[Array, "n n"],
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute information gain from adding measurements.

    Uses log det ratio: log det(F_after) - log det(F_before).
    Positive values indicate the new measurements added information.

    Parameters
    ----------
    - `fisher_before` (Float[Array, "n n"]):
        Fisher information before new measurements.
    - `fisher_after` (Float[Array, "n n"]):
        Fisher information after new measurements.

    Returns
    -------
    - `gain` (Float[Array, ""]):
        Information gain in nats.

    Flow
    ----
    1. Compute log det of both matrices
    2. Return difference
    """
    log_det_before: Float[Array, ""] = d_optimality(fisher_before)
    log_det_after: Float[Array, ""] = d_optimality(fisher_after)
    gain: Float[Array, ""] = log_det_after - log_det_before
    return gain