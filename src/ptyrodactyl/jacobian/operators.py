r"""Jacobian operator primitives for matrix-free linear algebra.

Extended Summary
----------------
Wraps JAX autodiff to provide building blocks for second-order
optimisation and spectral analysis without ever forming the
Jacobian matrix explicitly.  All operators work on arbitrary
PyTree parameter structures and are fully JIT-compatible.

Routine Listings
----------------
:func:`jvp_operator`
    Jacobian-vector product J @ v.
:func:`vjp_operator`
    Vector-Jacobian product J^T @ u.
:func:`jtj_operator`
    Normal equations operator J^T J @ v.
:func:`hvp_gauss_newton`
    Gauss-Newton Hessian-vector product for least-squares.
"""

from typing import Callable, Tuple, Any
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree


def jvp_operator(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
) -> Callable[[PyTree], Float[Array, "..."]]:
    r"""Construct a Jacobian-vector product operator J @ v.

    Extended Summary
    ----------------
    Builds a closure that computes
    :math:`J \, v = \partial f / \partial \theta \; v`
    evaluated at *params* using forward-mode autodiff.  The
    returned operator maps tangent vectors in parameter space to
    tangent vectors in measurement space.

    Implementation Logic
    --------------------
    1. **Capture linearisation point** --
       Close over *params* and *forward_fn*.
    2. **Evaluate JVP** --
       Call :func:`jax.jvp` with the supplied tangent vector.
    3. **Return tangent output** --
       Discard the primal output and return only the tangent.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to measurements.
    params : PyTree
        Point in parameter space at which to linearise.

    Returns
    -------
    jvp_fn : Callable[[PyTree], Float[Array, "..."]]
        Function that computes J @ v for any tangent vector v.
    """
    def jvp_fn(
        tangent_vector: PyTree,
    ) -> Float[Array, "..."]:
        """Compute J @ tangent_vector via forward-mode AD."""
        _, output_tangent = jax.jvp(forward_fn, (params,), (tangent_vector,))
        return output_tangent

    return jvp_fn


def vjp_operator(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
) -> Callable[[Float[Array, "..."]], PyTree]:
    r"""Construct a vector-Jacobian product operator J^T @ u.

    Extended Summary
    ----------------
    Builds a closure that computes
    :math:`J^\top u = (\partial f / \partial \theta)^\top u`
    evaluated at *params* using reverse-mode autodiff.  The
    returned operator maps cotangent vectors in measurement space
    to cotangent vectors in parameter space.

    Implementation Logic
    --------------------
    1. **Evaluate forward and capture VJP** --
       Call :func:`jax.vjp` to obtain the pullback function.
    2. **Apply pullback** --
       The returned closure applies the pullback to any
       cotangent vector.
    3. **Unwrap tuple** --
       Extract first element since :func:`jax.vjp` returns a
       tuple of parameter gradients.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to measurements.
    params : PyTree
        Point in parameter space at which to linearise.

    Returns
    -------
    vjp_fn : Callable[[Float[Array, "..."]], PyTree]
        Function that computes J^T @ u for any cotangent u.
    """
    _, vjp_fn_raw = jax.vjp(forward_fn, params)

    def vjp_fn(
        cotangent_vector: Float[Array, "..."],
    ) -> PyTree:
        """Compute J^T @ cotangent_vector via reverse-mode AD."""
        result_tuple: Tuple[PyTree, ...] = vjp_fn_raw(cotangent_vector)
        return result_tuple[0]

    return vjp_fn


def jtj_operator(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
) -> Callable[[PyTree], PyTree]:
    r"""Construct the normal equations operator J^T J.

    Extended Summary
    ----------------
    Builds a closure that computes
    :math:`J^\top J \, v` via a forward-mode JVP followed by a
    reverse-mode VJP, without materialising J.  The operator is
    symmetric positive semi-definite.  Its nullspace is the gauge
    subspace.  Its eigenvalues are the squared singular values
    of J.

    Implementation Logic
    --------------------
    1. **Compute VJP closure** --
       Pre-evaluate :func:`jax.vjp` to capture the pullback.
    2. **Forward pass** --
       Compute J @ v via :func:`jax.jvp`.
    3. **Backward pass** --
       Apply the cached pullback to get J^T (J v).

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to measurements.
    params : PyTree
        Point in parameter space at which to linearise.

    Returns
    -------
    jtj_fn : Callable[[PyTree], PyTree]
        Function that computes J^T J @ v for any vector v.
    """
    _, vjp_fn_raw = jax.vjp(forward_fn, params)

    def jtj_fn(
        vector: PyTree,
    ) -> PyTree:
        """Compute J^T J @ vector via JVP then VJP."""
        _, forward_tangent = jax.jvp(forward_fn, (params,), (vector,))
        backward_result: Tuple[PyTree, ...] = vjp_fn_raw(forward_tangent)
        return backward_result[0]

    return jtj_fn


def hvp_gauss_newton(
    forward_fn: Callable[[PyTree], Float[Array, "..."]],
    params: PyTree,
    residual: Float[Array, "..."],
) -> Callable[[PyTree], PyTree]:
    r"""Construct the Gauss-Newton Hessian-vector product operator.

    Extended Summary
    ----------------
    For least-squares problems
    :math:`\min \tfrac{1}{2}\|f(\theta) - y\|^2`, the
    Gauss-Newton approximation to the Hessian is
    :math:`J^\top J`, dropping the residual-Hessian term.  This
    approximation is exact at the solution when residuals vanish.

    The *residual* argument is accepted for API consistency with
    full Newton methods but is unused in the GN approximation.

    Implementation Logic
    --------------------
    1. **Drop residual-Hessian term** --
       The GN approximation ignores the residual argument.
    2. **Delegate to J^T J** --
       Return :func:`jtj_operator` evaluated at *params*.

    Parameters
    ----------
    forward_fn : Callable[[PyTree], Float[Array, "..."]]
        Forward model mapping parameters to measurements.
    params : PyTree
        Current parameter estimate.
    residual : Float[Array, "..."]
        Current residual f(params) - data.  Unused in the
        Gauss-Newton approximation.

    Returns
    -------
    hvp_fn : Callable[[PyTree], PyTree]
        Function that computes the GN Hessian-vector product
        for any vector v.

    See Also
    --------
    :func:`jtj_operator` : The underlying operator.
    """
    return jtj_operator(forward_fn, params)
