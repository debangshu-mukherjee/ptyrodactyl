"""
Module: ptyrodactyl.jacobian.operators
--------------------------------------

Jacobian operator primitives for matrix-free linear algebra.

These functions wrap JAX's autodiff to provide building blocks for
second-order optimization and spectral analysis without ever forming
the Jacobian matrix explicitly.

Functions
---------
- `jvp_operator`:
    Jacobian-vector product J @ v
- `vjp_operator`:
    Vector-Jacobian product Jᵀ @ u
- `jtj_operator`:
    Normal equations operator JᵀJ @ v
- `hvp_gauss_newton`:
    Gauss-Newton Hessian-vector product for least-squares
"""

from typing import Callable, Tuple, Any
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree


def jvp_operator(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
) -> Callable[[PyTree], Float[Array, "..."]]:
    """
    Description
    -----------
    Construct a function that computes the Jacobian-vector product J @ v
    where J = ∂forward_fn/∂params evaluated at the given parameters.

    The returned operator maps tangent vectors in parameter space to
    tangent vectors in measurement space. This is forward-mode autodiff.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to measurements.
    - `params` (PyTree):
        Point in parameter space at which to linearize.

    Returns
    -------
    - `jvp_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Function that computes J @ v for any tangent vector v.

    Flow
    ----
    1. Capture params and forward_fn in closure
    2. Return function that calls jax.jvp with given tangent
    3. Extract and return only the tangent output (not primals)
    """
    def jvp_fn(
        tangent_vector: PyTree,
    ) -> Float[Array, "..."]:
        _, output_tangent = jax.jvp(forward_fn, (params,), (tangent_vector,))
        return output_tangent

    return jvp_fn


def vjp_operator(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
) -> Callable[[Float[Array, "..."]], PyTree]:
    """
    Description
    -----------
    Construct a function that computes the vector-Jacobian product Jᵀ @ u
    where J = ∂forward_fn/∂params evaluated at the given parameters.

    The returned operator maps cotangent vectors in measurement space to
    cotangent vectors in parameter space. This is reverse-mode autodiff.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to measurements.
    - `params` (PyTree):
        Point in parameter space at which to linearize.

    Returns
    -------
    - `vjp_fn` (Callable[[Float[Array, "..."]], PyTree]):
        Function that computes Jᵀ @ u for any cotangent vector u.

    Flow
    ----
    1. Evaluate forward_fn and capture VJP function via jax.vjp
    2. Return function that applies captured VJP to cotangent
    3. Extract first element since jax.vjp returns tuple
    """
    _, vjp_fn_raw = jax.vjp(forward_fn, params)

    def vjp_fn(
        cotangent_vector: Float[Array, "..."],
    ) -> PyTree:
        result_tuple: Tuple[PyTree, ...] = vjp_fn_raw(cotangent_vector)
        return result_tuple[0]

    return vjp_fn


def jtj_operator(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
) -> Callable[[PyTree], PyTree]:
    """
    Description
    -----------
    Construct the normal equations operator JᵀJ for the linearized forward model.

    This operator maps parameter-space vectors to parameter-space vectors
    via v ↦ Jᵀ(J @ v). It is symmetric positive semi-definite. Its nullspace
    is the gauge subspace. Its eigenvalues are the squared singular values of J.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to measurements.
    - `params` (PyTree):
        Point in parameter space at which to linearize.

    Returns
    -------
    - `jtj_fn` (Callable[[PyTree], PyTree]):
        Function that computes JᵀJ @ v for any vector v.

    Flow
    ----
    1. Construct JVP operator for forward pass
    2. Construct VJP operator for backward pass
    3. Return composition: v → Jᵀ(J @ v)
    """
    _, vjp_fn_raw = jax.vjp(forward_fn, params)

    def jtj_fn(
        vector: PyTree,
    ) -> PyTree:
        _, forward_tangent = jax.jvp(forward_fn, (params,), (vector,))
        backward_result: Tuple[PyTree, ...] = vjp_fn_raw(forward_tangent)
        return backward_result[0]

    return jtj_fn


def hvp_gauss_newton(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    residual: Float[Array, "..."],
) -> Callable[[PyTree], PyTree]:
    """
    Description
    -----------
    Construct the Gauss-Newton Hessian-vector product operator.

    For least-squares problems min ½||f(θ) - y||², the Gauss-Newton
    approximation to the Hessian is JᵀJ, ignoring the residual-Hessian
    term. This is exact at the solution when residuals are zero.

    This function returns JᵀJ as an operator, identical to jtj_operator.
    The residual argument is included for API consistency with full
    Newton methods where it would be used.

    Parameters
    ----------
    - `forward_fn` (Callable[[PyTree], Float[Array, "..."]]):
        Forward model mapping parameters to measurements.
    - `params` (PyTree):
        Current parameter estimate.
    - `residual` (Float[Array, "..."]):
        Current residual f(params) - data. Unused in GN approximation.

    Returns
    -------
    - `hvp_fn` (Callable[[PyTree], PyTree]):
        Function that computes the Gauss-Newton HVP for any vector v.

    Flow
    ----
    1. Ignore residual (GN drops the residual-Hessian term)
    2. Return JᵀJ operator
    """
    return jtj_operator(forward_fn, params)