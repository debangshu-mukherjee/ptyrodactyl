"""
Module: ptyrodactyl.jacobian.gauge
----------------------------------

Gauge structure analysis for ptychographic inverse problems.

The nullspace of the Jacobian defines the gauge subspace: directions
in parameter space that produce no change in measurements. This module
provides tools to identify, project onto, and quotient out gauge freedom.

Functions
---------
- `nullspace_vectors_lanczos`:
    Estimate nullspace basis vectors via shifted inverse Lanczos
- `project_to_nullspace`:
    Project parameter perturbation onto gauge subspace
- `project_to_observable`:
    Project parameter perturbation onto observable subspace
- `decompose_gauge_observable`:
    Split vector into gauge and observable components
- `effective_rank`:
    Count observable dimensions above noise threshold
- `gauge_invariant_norm`:
    Compute norm in quotient space (modulo gauge)
- `random_gauge_direction`:
    Sample a random direction from the gauge subspace
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Float, Int, Array, PyTree

from ptyrodactyl.jacobian.operators import jtj_operator


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


def nullspace_vectors_lanczos(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    num_vectors: int = 10,
    num_lanczos_iterations: int = 100,
    threshold: float = 1e-6,
    random_seed: int = 42,
) -> Tuple[Float[Array, "num_vectors n"], Float[Array, "num_vectors"]]:
    """
    Description
    -----------
    Estimate basis vectors for the nullspace of the Jacobian.

    Uses Lanczos iteration on JᵀJ to find eigenvectors corresponding
    to the smallest eigenvalues. Eigenvectors with eigenvalue below
    threshold are considered nullspace (gauge) directions.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `num_vectors` (int):
        Maximum number of nullspace vectors to return. Default 10.
    - `num_lanczos_iterations` (int):
        Number of Lanczos iterations. Default 100.
    - `threshold` (float):
        Eigenvalue threshold for nullspace membership. Default 1e-6.
    - `random_seed` (int):
        Seed for random starting vector. Default 42.

    Returns
    -------
    - `nullspace_basis` (Float[Array, "num_vectors n"]):
        Orthonormal basis vectors for approximate nullspace.
    - `eigenvalues` (Float[Array, "num_vectors"]):
        Corresponding eigenvalues (squared singular values).

    Flow
    ----
    1. Flatten params to vector representation
    2. Construct JᵀJ operator on flattened space
    3. Run Lanczos to build tridiagonal matrix
    4. Compute eigenpairs of tridiagonal matrix
    5. Reconstruct Ritz vectors for smallest eigenvalues
    6. Filter by threshold, return nullspace basis
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]
    k: int = num_lanczos_iterations

    jtj_pytree_fn: Callable = jtj_operator(forward_fn, params)

    def jtj_flat_fn(
        v_flat: Float[Array, "n"],
    ) -> Float[Array, "n"]:
        v_pytree: PyTree = unflatten_fn(v_flat)
        result_pytree: PyTree = jtj_pytree_fn(v_pytree)
        result_flat: Float[Array, "n"] = jax.flatten_util.ravel_pytree(result_pytree)[0]
        return result_flat

    key: jax.random.PRNGKey = jax.random.PRNGKey(random_seed)
    v0: Float[Array, "n"] = jax.random.normal(key, (n,))
    v0_norm: Float[Array, ""] = jnp.linalg.norm(v0)
    v0_normalized: Float[Array, "n"] = v0 / v0_norm

    lanczos_vectors: Float[Array, "k n"] = jnp.zeros((k, n))
    lanczos_vectors = lanczos_vectors.at[0].set(v0_normalized)
    alpha: Float[Array, "k"] = jnp.zeros(k)
    beta: Float[Array, "k"] = jnp.zeros(k)

    def lanczos_body(
        iteration: int,
        carry: Tuple[Float[Array, "k n"], Float[Array, "k"], Float[Array, "k"]],
    ) -> Tuple[Float[Array, "k n"], Float[Array, "k"], Float[Array, "k"]]:
        vectors, alphas, betas = carry
        v_curr: Float[Array, "n"] = vectors[iteration]
        w: Float[Array, "n"] = jtj_flat_fn(v_curr)
        alpha_i: Float[Array, ""] = jnp.dot(w, v_curr)
        alphas_new: Float[Array, "k"] = alphas.at[iteration].set(alpha_i)

        v_prev: Float[Array, "n"] = lax.cond(
            iteration > 0,
            lambda: vectors[iteration - 1],
            lambda: jnp.zeros(n),
        )
        beta_prev: Float[Array, ""] = lax.cond(
            iteration > 0,
            lambda: betas[iteration - 1],
            lambda: jnp.array(0.0),
        )

        w_orth: Float[Array, "n"] = w - alpha_i * v_curr - beta_prev * v_prev
        beta_i: Float[Array, ""] = jnp.linalg.norm(w_orth)
        v_next: Float[Array, "n"] = w_orth / (beta_i + 1e-12)

        betas_new: Float[Array, "k"] = betas.at[iteration].set(beta_i)
        vectors_new: Float[Array, "k n"] = lax.cond(
            iteration < k - 1,
            lambda: vectors.at[iteration + 1].set(v_next),
            lambda: vectors,
        )

        return (vectors_new, alphas_new, betas_new)

    lanczos_vectors, alpha, beta = lax.fori_loop(
        0, k, lanczos_body, (lanczos_vectors, alpha, beta)
    )

    tridiag_matrix: Float[Array, "k k"] = (
        jnp.diag(alpha) +
        jnp.diag(beta[:-1], k=1) +
        jnp.diag(beta[:-1], k=-1)
    )

    eigenvalues_tri, eigenvectors_tri = jnp.linalg.eigh(tridiag_matrix)

    ritz_vectors: Float[Array, "k n"] = jnp.dot(eigenvectors_tri.T, lanczos_vectors)

    sorted_indices: Int[Array, "k"] = jnp.argsort(eigenvalues_tri)
    eigenvalues_sorted: Float[Array, "k"] = eigenvalues_tri[sorted_indices]
    ritz_vectors_sorted: Float[Array, "k n"] = ritz_vectors[sorted_indices]

    eigenvalues_out: Float[Array, "num_vectors"] = eigenvalues_sorted[:num_vectors]
    nullspace_basis: Float[Array, "num_vectors n"] = ritz_vectors_sorted[:num_vectors]

    norms: Float[Array, "num_vectors"] = jnp.linalg.norm(nullspace_basis, axis=1, keepdims=True)
    nullspace_basis_normalized: Float[Array, "num_vectors n"] = nullspace_basis / (norms + 1e-12)

    return nullspace_basis_normalized, eigenvalues_out


def project_to_nullspace(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    perturbation: PyTree,
    num_nullspace_vectors: int = 20,
    threshold: float = 1e-6,
    random_seed: int = 42,
) -> PyTree:
    """
    Description
    -----------
    Project a parameter perturbation onto the gauge (nullspace) subspace.

    The gauge component of a perturbation is the part that produces no
    measurable change. This is the projection onto the nullspace of J.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `perturbation` (PyTree):
        Perturbation vector to project.
    - `num_nullspace_vectors` (int):
        Number of nullspace basis vectors to use. Default 20.
    - `threshold` (float):
        Eigenvalue threshold for nullspace. Default 1e-6.
    - `random_seed` (int):
        Seed for Lanczos initialization. Default 42.

    Returns
    -------
    - `gauge_component` (PyTree):
        Projection of perturbation onto nullspace (gauge subspace).

    Flow
    ----
    1. Compute nullspace basis via Lanczos
    2. Filter basis vectors by eigenvalue threshold
    3. Project perturbation onto each basis vector
    4. Sum projections to get gauge component
    """
    nullspace_basis, eigenvalues = nullspace_vectors_lanczos(
        forward_fn, params, num_nullspace_vectors,
        num_lanczos_iterations=100, threshold=threshold, random_seed=random_seed
    )

    flat_perturbation, unflatten_fn = jax.flatten_util.ravel_pytree(perturbation)
    n: int = flat_perturbation.shape[0]
    num_vecs: int = nullspace_basis.shape[0]

    is_nullspace: Float[Array, "num_vectors"] = (eigenvalues < threshold).astype(jnp.float32)

    coefficients: Float[Array, "num_vectors"] = jnp.dot(nullspace_basis, flat_perturbation)
    masked_coefficients: Float[Array, "num_vectors"] = coefficients * is_nullspace

    gauge_component_flat: Float[Array, "n"] = jnp.dot(masked_coefficients, nullspace_basis)
    gauge_component: PyTree = unflatten_fn(gauge_component_flat)

    return gauge_component


def project_to_observable(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    perturbation: PyTree,
    num_nullspace_vectors: int = 20,
    threshold: float = 1e-6,
    random_seed: int = 42,
) -> PyTree:
    """
    Description
    -----------
    Project a parameter perturbation onto the observable (column) subspace.

    The observable component is the part that produces measurable change.
    This is the orthogonal complement of the nullspace projection.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `perturbation` (PyTree):
        Perturbation vector to project.
    - `num_nullspace_vectors` (int):
        Number of nullspace basis vectors to use. Default 20.
    - `threshold` (float):
        Eigenvalue threshold for nullspace. Default 1e-6.
    - `random_seed` (int):
        Seed for Lanczos initialization. Default 42.

    Returns
    -------
    - `observable_component` (PyTree):
        Projection of perturbation onto observable subspace.

    Flow
    ----
    1. Compute gauge (nullspace) component
    2. Subtract from original perturbation
    3. Return observable component
    """
    gauge_component: PyTree = project_to_nullspace(
        forward_fn, params, perturbation,
        num_nullspace_vectors, threshold, random_seed
    )
    observable_component: PyTree = _tree_sub(perturbation, gauge_component)
    return observable_component


def decompose_gauge_observable(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    perturbation: PyTree,
    num_nullspace_vectors: int = 20,
    threshold: float = 1e-6,
    random_seed: int = 42,
) -> Tuple[PyTree, PyTree]:
    """
    Description
    -----------
    Decompose a perturbation into gauge and observable components.

    Any perturbation δθ can be uniquely decomposed as:
        δθ = δθ_gauge + δθ_observable
    where δθ_gauge ∈ null(J) and δθ_observable ∈ col(Jᵀ).

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `perturbation` (PyTree):
        Perturbation vector to decompose.
    - `num_nullspace_vectors` (int):
        Number of nullspace basis vectors. Default 20.
    - `threshold` (float):
        Eigenvalue threshold for nullspace. Default 1e-6.
    - `random_seed` (int):
        Seed for Lanczos initialization. Default 42.

    Returns
    -------
    - `gauge_component` (PyTree):
        Projection onto nullspace (unobservable).
    - `observable_component` (PyTree):
        Projection onto column space (observable).

    Flow
    ----
    1. Compute gauge component via nullspace projection
    2. Compute observable component as remainder
    3. Return both components
    """
    gauge_component: PyTree = project_to_nullspace(
        forward_fn, params, perturbation,
        num_nullspace_vectors, threshold, random_seed
    )
    observable_component: PyTree = _tree_sub(perturbation, gauge_component)
    return gauge_component, observable_component


def effective_rank(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    noise_floor: float,
    num_lanczos_iterations: int = 100,
    random_seed: int = 42,
) -> Int[Array, ""]:
    """
    Description
    -----------
    Count the number of observable dimensions above the noise floor.

    The effective rank is the number of singular values σ_i > noise_floor.
    This is the dimension of the subspace that can be reliably recovered
    from data at the given SNR.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `noise_floor` (float):
        Noise threshold η. Singular values below this are unobservable.
    - `num_lanczos_iterations` (int):
        Lanczos iterations for spectrum estimation. Default 100.
    - `random_seed` (int):
        Seed for random initialization. Default 42.

    Returns
    -------
    - `rank` (Int[Array, ""]):
        Number of singular values above noise_floor.

    Flow
    ----
    1. Estimate singular spectrum via Lanczos
    2. Count values above noise_floor
    3. Return count as effective rank
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
    v0: Float[Array, "n"] = jax.random.normal(key, (n,))
    v0_norm: Float[Array, ""] = jnp.linalg.norm(v0)
    v0_normalized: Float[Array, "n"] = v0 / v0_norm

    k: int = num_lanczos_iterations
    alpha: Float[Array, "k"] = jnp.zeros(k)
    beta: Float[Array, "k"] = jnp.zeros(k)
    v_prev: Float[Array, "n"] = jnp.zeros(n)
    v_curr: Float[Array, "n"] = v0_normalized

    def lanczos_step(
        iteration: int,
        carry: Tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "k"], Float[Array, "k"]],
    ) -> Tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "k"], Float[Array, "k"]]:
        v_p, v_c, alphas, betas = carry
        w: Float[Array, "n"] = jtj_flat_fn(v_c)
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

    _, _, alpha, beta = lax.fori_loop(
        0, k, lanczos_step, (v_prev, v_curr, alpha, beta)
    )

    tridiag_matrix: Float[Array, "k k"] = (
        jnp.diag(alpha) +
        jnp.diag(beta[:-1], k=1) +
        jnp.diag(beta[:-1], k=-1)
    )

    eigenvalues: Float[Array, "k"] = jnp.linalg.eigvalsh(tridiag_matrix)
    eigenvalues_positive: Float[Array, "k"] = jnp.maximum(eigenvalues, 0.0)
    singular_values: Float[Array, "k"] = jnp.sqrt(eigenvalues_positive)

    threshold_squared: Float[Array, ""] = jnp.array(noise_floor)
    above_threshold: Float[Array, "k"] = (singular_values > threshold_squared).astype(jnp.int32)
    rank: Int[Array, ""] = jnp.sum(above_threshold)

    return rank


def gauge_invariant_norm(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    perturbation: PyTree,
    num_nullspace_vectors: int = 20,
    threshold: float = 1e-6,
    random_seed: int = 42,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute the norm of a perturbation in the quotient space modulo gauge.

    This is the norm of the observable component only. Two perturbations
    that differ by a gauge direction have the same gauge-invariant norm.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `perturbation` (PyTree):
        Perturbation vector.
    - `num_nullspace_vectors` (int):
        Number of nullspace basis vectors. Default 20.
    - `threshold` (float):
        Eigenvalue threshold for nullspace. Default 1e-6.
    - `random_seed` (int):
        Seed for Lanczos initialization. Default 42.

    Returns
    -------
    - `norm` (Float[Array, ""]):
        L2 norm of the observable component.

    Flow
    ----
    1. Project perturbation onto observable subspace
    2. Compute and return L2 norm
    """
    observable_component: PyTree = project_to_observable(
        forward_fn, params, perturbation,
        num_nullspace_vectors, threshold, random_seed
    )
    norm_squared: Float[Array, ""] = _tree_dot(observable_component, observable_component)
    norm: Float[Array, ""] = jnp.sqrt(norm_squared)
    return norm


def random_gauge_direction(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    num_nullspace_vectors: int = 20,
    threshold: float = 1e-6,
    random_seed: int = 42,
    direction_seed: int = 123,
) -> PyTree:
    """
    Description
    -----------
    Sample a random direction from the gauge (nullspace) subspace.

    Useful for exploring the gauge orbit: params + α * gauge_direction
    produces measurements identical to params for any α.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params` (PyTree):
        Point at which to evaluate the Jacobian.
    - `num_nullspace_vectors` (int):
        Number of nullspace basis vectors. Default 20.
    - `threshold` (float):
        Eigenvalue threshold for nullspace. Default 1e-6.
    - `random_seed` (int):
        Seed for Lanczos initialization. Default 42.
    - `direction_seed` (int):
        Seed for random direction coefficients. Default 123.

    Returns
    -------
    - `gauge_direction` (PyTree):
        Unit-norm vector in the gauge subspace.

    Flow
    ----
    1. Compute nullspace basis
    2. Generate random coefficients
    3. Form linear combination of basis vectors
    4. Normalize to unit length
    """
    nullspace_basis, eigenvalues = nullspace_vectors_lanczos(
        forward_fn, params, num_nullspace_vectors,
        num_lanczos_iterations=100, threshold=threshold, random_seed=random_seed
    )

    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    n: int = flat_params.shape[0]
    num_vecs: int = nullspace_basis.shape[0]

    is_nullspace: Float[Array, "num_vectors"] = (eigenvalues < threshold).astype(jnp.float32)

    key: jax.random.PRNGKey = jax.random.PRNGKey(direction_seed)
    random_coeffs: Float[Array, "num_vectors"] = jax.random.normal(key, (num_vecs,))
    masked_coeffs: Float[Array, "num_vectors"] = random_coeffs * is_nullspace

    direction_flat: Float[Array, "n"] = jnp.dot(masked_coeffs, nullspace_basis)
    direction_norm: Float[Array, ""] = jnp.linalg.norm(direction_flat)
    direction_normalized: Float[Array, "n"] = direction_flat / (direction_norm + 1e-12)

    gauge_direction: PyTree = unflatten_fn(direction_normalized)
    return gauge_direction


def gauge_orbit_distance(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params_a: PyTree,
    params_b: PyTree,
    num_nullspace_vectors: int = 20,
    threshold: float = 1e-6,
    random_seed: int = 42,
) -> Float[Array, ""]:
    """
    Description
    -----------
    Compute distance between two points in quotient space modulo gauge.

    Two points on the same gauge orbit have distance zero. This measures
    the physically meaningful difference between parameter configurations.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to predictions.
    - `params_a` (PyTree):
        First parameter configuration.
    - `params_b` (PyTree):
        Second parameter configuration.
    - `num_nullspace_vectors` (int):
        Number of nullspace basis vectors. Default 20.
    - `threshold` (float):
        Eigenvalue threshold for nullspace. Default 1e-6.
    - `random_seed` (int):
        Seed for Lanczos initialization. Default 42.

    Returns
    -------
    - `distance` (Float[Array, ""]):
        Distance in quotient space.

    Flow
    ----
    1. Compute difference params_b - params_a
    2. Project onto observable subspace
    3. Return norm of observable component
    """
    difference: PyTree = _tree_sub(params_b, params_a)
    midpoint: PyTree = jax.tree_util.tree_map(
        lambda a, b: 0.5 * (a + b), params_a, params_b
    )
    distance: Float[Array, ""] = gauge_invariant_norm(
        forward_fn, midpoint, difference,
        num_nullspace_vectors, threshold, random_seed
    )
    return distance