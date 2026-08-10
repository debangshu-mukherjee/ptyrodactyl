"""Define reconstruction problem and result carriers.

Extended Summary
----------------
This module defines skeleton Equinox PyTree carriers for inverse
reconstruction problems, solver results, and uncertainty summaries. The
factories validate callable/static metadata, array structure, and traced
finite-value constraints, but intentionally do not implement solver logic.

Routine Listings
----------------
:class:`LaplaceUncertainty`
    Store a Laplace-approximation uncertainty summary.
:class:`PosteriorSamples`
    Store posterior samples and diagnostics.
:class:`ReconProblem`
    Store a reconstruction inverse-problem contract.
:class:`ReconResult`
    Store a reconstruction solver result.
:func:`create_laplace_uncertainty`
    Create a LaplaceUncertainty with runtime validation.
:func:`create_posterior_samples`
    Create PosteriorSamples with runtime validation.
:func:`create_recon_problem`
    Create a ReconProblem with runtime validation.
:func:`create_recon_result`
    Create a ReconResult with runtime validation.

Notes
-----
All callable hooks and string labels are static Equinox fields. Numeric
state stays dynamic so reconstruction workflows can pass these carriers
through JAX transformations once solver behavior is wired in later plans.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, Num, PyTree, jaxtyped

from .custom_types import scalar_bool, scalar_int, scalar_num


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a true structural condition.

    Parameters
    ----------
    condition : bool
        Whether the structural contract is invalid.
    message : str
        Error message for the rejected contract.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _check_finite_array(
    array: Num[Array, "..."],
    name: str,
) -> Num[Array, "..."]:
    """PRIVATE: Attach a traced finite-value check to one numeric array.

    Parameters
    ----------
    array : Num[Array, "..."]
        Numeric array to validate.
    name : str
        Field name included in the runtime error.

    Returns
    -------
    checked_array : Num[Array, "..."]
        Input array with a traced finite-value assertion.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any array element is non-finite under compiled execution.
    """
    checked_array: Num[Array, "..."] = eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)),
        f"{name} contains non-finite values",
    )
    return checked_array


def _coerce_array_tree(tree: PyTree) -> PyTree:
    """PRIVATE: Convert every dynamic tree leaf to a JAX array.

    Parameters
    ----------
    tree : PyTree
        PyTree containing array-compatible leaves.

    Returns
    -------
    array_tree : PyTree
        Matching PyTree with each leaf converted by :func:`jax.numpy.asarray`.
    """
    array_tree: PyTree = jax.tree_util.tree_map(jnp.asarray, tree)
    return array_tree


def _check_finite_leaf(leaf: Any, name: str) -> Any:
    """PRIVATE: Attach a traced finite-value check to one PyTree leaf.

    Parameters
    ----------
    leaf : Any
        Numeric PyTree leaf to validate.
    name : str
        Field name included in the runtime error.

    Returns
    -------
    checked_leaf : Any
        Input leaf with a traced finite-value assertion.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any leaf element is non-finite under compiled execution.
    """
    checked_leaf: Any = eqx.error_if(
        leaf,
        jnp.any(~jnp.isfinite(leaf)),
        f"{name} contains non-finite values",
    )
    return checked_leaf


def _check_finite_tree(tree: PyTree, name: str) -> PyTree:
    """PRIVATE: Attach traced finite-value checks to every PyTree leaf.

    Parameters
    ----------
    tree : PyTree
        Numeric PyTree to validate.
    name : str
        Field name included in runtime errors.

    Returns
    -------
    checked_tree : PyTree
        Matching PyTree with a traced finite-value assertion on each leaf.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If any leaf element is non-finite under compiled execution.
    """
    checked_tree: PyTree = jax.tree_util.tree_map(
        lambda leaf: _check_finite_leaf(leaf, name),
        tree,
    )
    return checked_tree


class ReconProblem(eqx.Module):
    """Store a reconstruction inverse-problem contract.

    :see: :class:`~.test_recon_types.TestReconCarriers`

    Attributes
    ----------
    forward : Callable[..., Any]
        Static forward-model callable.
    measured : Num[Array, "..."]
        Dynamic measured data array.
    transform : Optional[Callable[..., Any]]
        Optional static parameter transform. ``None`` denotes identity.
    residual_fn : Optional[Callable[..., Any]]
        Optional static residual hook.
    loss_fn : Optional[Callable[..., Any]]
        Optional static scalar-loss hook.
    forward_family : str
        Static forward-family label, for example ``"born"``.
    terminal : str
        Static measurement-terminal label, for example ``"diffraction"``.

    See Also
    --------
    :func:`create_recon_problem`
        Create and validate a :class:`ReconProblem`.
    """

    forward: Callable[..., Any] = eqx.field(static=True)
    measured: Num[Array, "..."]
    transform: Optional[Callable[..., Any]] = eqx.field(
        static=True,
        default=None,
    )
    residual_fn: Optional[Callable[..., Any]] = eqx.field(
        static=True,
        default=None,
    )
    loss_fn: Optional[Callable[..., Any]] = eqx.field(
        static=True,
        default=None,
    )
    forward_family: str = eqx.field(static=True, default="")
    terminal: str = eqx.field(static=True, default="")


class ReconResult(eqx.Module):
    """Store a reconstruction solver result.

    :see: :class:`~.test_recon_types.TestReconCarriers`

    Attributes
    ----------
    params : PyTree
        Dynamic optimized parameter PyTree.
    latent_params : PyTree
        Dynamic latent-parameter PyTree after any transform.
    simulated : Num[Array, "..."]
        Dynamic simulated terminal data.
    residual : Num[Array, "..."]
        Dynamic residual data with the same shape as ``simulated``.
    loss : Float[Array, ""]
        Dynamic scalar loss value.
    iterations : Int[Array, ""]
        Dynamic scalar iteration count.
    converged : Bool[Array, ""]
        Dynamic scalar convergence flag.
    solver_status : str
        Static solver-status label.

    See Also
    --------
    :func:`create_recon_result`
        Create and validate a :class:`ReconResult`.
    """

    params: PyTree
    latent_params: PyTree
    simulated: Num[Array, "..."]
    residual: Num[Array, "..."]
    loss: Float[Array, ""]
    iterations: Int[Array, ""]
    converged: Bool[Array, ""]
    solver_status: str = eqx.field(static=True)


class LaplaceUncertainty(eqx.Module):
    """Store a Laplace-approximation uncertainty summary.

    :see: :class:`~.test_recon_types.TestReconCarriers`

    Attributes
    ----------
    fisher_information : Float[Array, "n n"]
        Fisher information matrix.
    covariance : Float[Array, "n n"]
        Covariance matrix matching the Fisher matrix shape.
    standard_deviation : Float[Array, " n"]
        Per-parameter standard deviation vector.
    correlation : Float[Array, "n n"]
        Correlation matrix matching the covariance shape.

    See Also
    --------
    :func:`create_laplace_uncertainty`
        Create and validate a :class:`LaplaceUncertainty`.
    """

    fisher_information: Float[Array, "n n"]
    covariance: Float[Array, "n n"]
    standard_deviation: Float[Array, " n"]
    correlation: Float[Array, "n n"]


class PosteriorSamples(eqx.Module):
    """Store posterior samples and diagnostics.

    :see: :class:`~.test_recon_types.TestReconCarriers`

    Attributes
    ----------
    samples : Float[Array, "draws n"]
        Posterior sample matrix with one row per draw.
    mean : Float[Array, " n"]
        Per-parameter posterior mean.
    covariance : Float[Array, "n n"]
        Per-parameter posterior covariance.
    rhat : Float[Array, " n"]
        Per-parameter split-R-hat diagnostic.
    ess : Float[Array, " n"]
        Per-parameter effective sample size.
    converged : Bool[Array, ""]
        Dynamic scalar convergence flag.

    See Also
    --------
    :func:`create_posterior_samples`
        Create and validate :class:`PosteriorSamples`.
    """

    samples: Float[Array, "draws n"]
    mean: Float[Array, " n"]
    covariance: Float[Array, "n n"]
    rhat: Float[Array, " n"]
    ess: Float[Array, " n"]
    converged: Bool[Array, ""]


@jaxtyped(typechecker=beartype)
def create_recon_problem(
    forward: Any,
    measured: Num[Array, "..."],
    transform: Any = None,
    residual_fn: Any = None,
    loss_fn: Any = None,
    forward_family: str = "",
    terminal: str = "",
) -> ReconProblem:
    """Create a ReconProblem with runtime validation.

    :see: :class:`~.test_recon_types.TestCreateReconFactories`

    Parameters
    ----------
    forward : Any
        Static forward-model callable.
    measured : Num[Array, "..."]
        Measured terminal data, coerced to a JAX array.
    transform : Any, optional
        Optional static parameter transform. ``None`` means identity.
    residual_fn : Any, optional
        Optional static residual hook.
    loss_fn : Any, optional
        Optional static scalar-loss hook.
    forward_family : str, optional
        Static forward-family label.
    terminal : str, optional
        Static measurement-terminal label.

    Returns
    -------
    problem : ReconProblem
        Validated reconstruction problem carrier.

    Raises
    ------
    ValueError
        If callable hooks or static labels are structurally invalid.

    Notes
    -----
    1. Validate static callable hooks and string labels.
    2. Coerce measured data to a JAX array.
    3. Attach a traced finite-value check to measured data.
    4. Create and return the skeleton carrier without running ``forward``.
    """
    if not callable(forward):
        raise ValueError("forward must be callable")
    if transform is not None and not callable(transform):
        raise ValueError("transform must be callable or None")
    if residual_fn is not None and not callable(residual_fn):
        raise ValueError("residual_fn must be callable or None")
    if loss_fn is not None and not callable(loss_fn):
        raise ValueError("loss_fn must be callable or None")
    _raise_if(
        not isinstance(forward_family, str),
        "forward_family must be a string",
    )
    _raise_if(not isinstance(terminal, str), "terminal must be a string")

    measured_arr: Num[Array, "..."] = jnp.asarray(measured)
    checked_measured: Num[Array, "..."] = _check_finite_array(
        measured_arr,
        "measured",
    )
    problem: ReconProblem = ReconProblem(
        forward=forward,
        measured=checked_measured,
        transform=transform,
        residual_fn=residual_fn,
        loss_fn=loss_fn,
        forward_family=forward_family,
        terminal=terminal,
    )
    return problem


@jaxtyped(typechecker=beartype)
def create_recon_result(
    params: PyTree,
    latent_params: PyTree,
    simulated: Num[Array, "..."],
    residual: Num[Array, "..."],
    loss: scalar_num,
    iterations: scalar_int,
    converged: scalar_bool,
    solver_status: str = "",
) -> ReconResult:
    """Create a ReconResult with runtime validation.

    :see: :class:`~.test_recon_types.TestCreateReconFactories`

    Parameters
    ----------
    params : PyTree
        Optimized parameter PyTree.
    latent_params : PyTree
        Latent parameter PyTree after any transform.
    simulated : Num[Array, "..."]
        Simulated terminal data.
    residual : Num[Array, "..."]
        Residual data with the same shape as ``simulated``.
    loss : scalar_num
        Scalar loss value.
    iterations : scalar_int
        Scalar iteration count.
    converged : scalar_bool
        Scalar convergence flag.
    solver_status : str, optional
        Static solver-status label.

    Returns
    -------
    result : ReconResult
        Validated reconstruction result carrier.

    Raises
    ------
    ValueError
        If shapes or static labels are structurally invalid.

    Notes
    -----
    1. Coerce dynamic fields to arrays or array PyTrees.
    2. Validate scalar fields and simulated/residual shape consistency.
    3. Attach traced checks for finite numeric values and non-negative
       iterations.
    4. Create and return the skeleton carrier.
    """
    _raise_if(
        not isinstance(solver_status, str),
        "solver_status must be a string",
    )
    params_arr: PyTree = _coerce_array_tree(params)
    latent_params_arr: PyTree = _coerce_array_tree(latent_params)
    simulated_arr: Num[Array, "..."] = jnp.asarray(simulated)
    residual_arr: Num[Array, "..."] = jnp.asarray(residual)
    loss_arr: Float[Array, ""] = jnp.asarray(loss, dtype=jnp.float64)
    iterations_arr: Int[Array, ""] = jnp.asarray(iterations, dtype=jnp.int32)
    converged_arr: Bool[Array, ""] = jnp.asarray(converged, dtype=jnp.bool_)

    scalar_shape: Tuple[()] = ()
    _raise_if(
        simulated_arr.shape != residual_arr.shape,
        "simulated and residual must have matching shapes",
    )
    _raise_if(loss_arr.shape != scalar_shape, "loss must be a scalar")
    _raise_if(
        iterations_arr.shape != scalar_shape,
        "iterations must be a scalar",
    )
    _raise_if(
        converged_arr.shape != scalar_shape,
        "converged must be a scalar",
    )

    checked_params: PyTree = _check_finite_tree(params_arr, "params")
    checked_latent_params: PyTree = _check_finite_tree(
        latent_params_arr,
        "latent_params",
    )
    checked_simulated: Num[Array, "..."] = _check_finite_array(
        simulated_arr,
        "simulated",
    )
    checked_residual: Num[Array, "..."] = _check_finite_array(
        residual_arr,
        "residual",
    )
    checked_loss: Float[Array, ""] = eqx.error_if(
        loss_arr,
        ~jnp.isfinite(loss_arr),
        "loss must be finite",
    )
    checked_iterations: Int[Array, ""] = eqx.error_if(
        iterations_arr,
        iterations_arr < 0,
        "iterations must be non-negative",
    )
    result: ReconResult = ReconResult(
        params=checked_params,
        latent_params=checked_latent_params,
        simulated=checked_simulated,
        residual=checked_residual,
        loss=checked_loss,
        iterations=checked_iterations,
        converged=converged_arr,
        solver_status=solver_status,
    )
    return result


@jaxtyped(typechecker=beartype)
def create_laplace_uncertainty(
    fisher_information: Float[Array, "..."],
    covariance: Float[Array, "..."],
    standard_deviation: Float[Array, "..."],
    correlation: Float[Array, "..."],
) -> LaplaceUncertainty:
    """Create a LaplaceUncertainty with runtime validation.

    :see: :class:`~.test_recon_types.TestCreateReconFactories`

    Parameters
    ----------
    fisher_information : Float[Array, "..."]
        Fisher information matrix.
    covariance : Float[Array, "..."]
        Covariance matrix matching the Fisher shape.
    standard_deviation : Float[Array, "..."]
        Per-parameter standard deviation vector.
    correlation : Float[Array, "..."]
        Correlation matrix matching the covariance shape.

    Returns
    -------
    uncertainty : LaplaceUncertainty
        Validated Laplace uncertainty carrier.

    Raises
    ------
    ValueError
        If matrix ranks, square shapes, or vector lengths are invalid.

    Notes
    -----
    1. Coerce all fields to ``float64`` JAX arrays.
    2. Validate square matrix structure and matching parameter dimensions.
    3. Attach traced finite-value checks.
    4. Create and return the skeleton carrier.
    """
    fisher_arr: Float[Array, "n n"] = jnp.asarray(
        fisher_information,
        dtype=jnp.float64,
    )
    covariance_arr: Float[Array, "n n"] = jnp.asarray(
        covariance,
        dtype=jnp.float64,
    )
    std_arr: Float[Array, " n"] = jnp.asarray(
        standard_deviation,
        dtype=jnp.float64,
    )
    correlation_arr: Float[Array, "n n"] = jnp.asarray(
        correlation,
        dtype=jnp.float64,
    )

    matrix_rank: int = 2
    vector_rank: int = 1
    _raise_if(
        fisher_arr.ndim != matrix_rank,
        "fisher_information must be 2D",
    )
    _raise_if(covariance_arr.ndim != matrix_rank, "covariance must be 2D")
    _raise_if(correlation_arr.ndim != matrix_rank, "correlation must be 2D")
    _raise_if(
        fisher_arr.shape[0] != fisher_arr.shape[1],
        "fisher_information must be square",
    )
    _raise_if(
        covariance_arr.shape[0] != covariance_arr.shape[1],
        "covariance must be square",
    )
    _raise_if(
        correlation_arr.shape[0] != correlation_arr.shape[1],
        "correlation must be square",
    )
    _raise_if(
        covariance_arr.shape != fisher_arr.shape,
        "covariance must match fisher_information shape",
    )
    _raise_if(
        correlation_arr.shape != covariance_arr.shape,
        "correlation must match covariance shape",
    )
    _raise_if(
        std_arr.ndim != vector_rank,
        "standard_deviation must be 1D",
    )
    _raise_if(
        std_arr.shape[0] != covariance_arr.shape[0],
        "standard_deviation must match covariance diagonal length",
    )

    checked_fisher: Float[Array, "n n"] = _check_finite_array(
        fisher_arr,
        "fisher_information",
    )
    checked_covariance: Float[Array, "n n"] = _check_finite_array(
        covariance_arr,
        "covariance",
    )
    checked_std: Float[Array, " n"] = _check_finite_array(
        std_arr,
        "standard_deviation",
    )
    checked_correlation: Float[Array, "n n"] = _check_finite_array(
        correlation_arr,
        "correlation",
    )
    uncertainty: LaplaceUncertainty = LaplaceUncertainty(
        fisher_information=checked_fisher,
        covariance=checked_covariance,
        standard_deviation=checked_std,
        correlation=checked_correlation,
    )
    return uncertainty


@jaxtyped(typechecker=beartype)
def create_posterior_samples(
    samples: Float[Array, "..."],
    mean: Float[Array, "..."],
    covariance: Float[Array, "..."],
    rhat: Float[Array, "..."],
    ess: Float[Array, "..."],
    converged: scalar_bool,
) -> PosteriorSamples:
    """Create PosteriorSamples with runtime validation.

    :see: :class:`~.test_recon_types.TestCreateReconFactories`

    Parameters
    ----------
    samples : Float[Array, "..."]
        Posterior sample matrix with shape ``(draws, n)``.
    mean : Float[Array, "..."]
        Per-parameter posterior mean with shape ``(n,)``.
    covariance : Float[Array, "..."]
        Posterior covariance with shape ``(n, n)``.
    rhat : Float[Array, "..."]
        Split-R-hat diagnostics with shape ``(n,)``.
    ess : Float[Array, "..."]
        Effective sample sizes with shape ``(n,)``.
    converged : scalar_bool
        Scalar convergence flag.

    Returns
    -------
    posterior : PosteriorSamples
        Validated posterior sample carrier.

    Raises
    ------
    ValueError
        If sample, vector, matrix, or scalar structures are invalid.

    Notes
    -----
    1. Coerce all numeric diagnostics to ``float64`` JAX arrays.
    2. Validate the common parameter dimension across all fields.
    3. Attach traced finite-value checks.
    4. Create and return the skeleton carrier.
    """
    samples_arr: Float[Array, "draws n"] = jnp.asarray(
        samples,
        dtype=jnp.float64,
    )
    mean_arr: Float[Array, " n"] = jnp.asarray(mean, dtype=jnp.float64)
    covariance_arr: Float[Array, "n n"] = jnp.asarray(
        covariance,
        dtype=jnp.float64,
    )
    rhat_arr: Float[Array, " n"] = jnp.asarray(rhat, dtype=jnp.float64)
    ess_arr: Float[Array, " n"] = jnp.asarray(ess, dtype=jnp.float64)
    converged_arr: Bool[Array, ""] = jnp.asarray(converged, dtype=jnp.bool_)

    matrix_rank: int = 2
    vector_rank: int = 1
    scalar_shape: Tuple[()] = ()
    _raise_if(samples_arr.ndim != matrix_rank, "samples must be 2D")
    _raise_if(mean_arr.ndim != vector_rank, "mean must be 1D")
    _raise_if(covariance_arr.ndim != matrix_rank, "covariance must be 2D")
    _raise_if(
        covariance_arr.shape[0] != covariance_arr.shape[1],
        "covariance must be square",
    )
    parameter_count: int = samples_arr.shape[1]
    _raise_if(
        mean_arr.shape != (parameter_count,),
        "mean must have shape (n,)",
    )
    _raise_if(
        covariance_arr.shape != (parameter_count, parameter_count),
        "covariance must have shape (n, n)",
    )
    _raise_if(
        rhat_arr.shape != (parameter_count,),
        "rhat must have shape (n,)",
    )
    _raise_if(ess_arr.shape != (parameter_count,), "ess must have shape (n,)")
    _raise_if(
        converged_arr.shape != scalar_shape,
        "converged must be a scalar",
    )

    checked_samples: Float[Array, "draws n"] = _check_finite_array(
        samples_arr,
        "samples",
    )
    checked_mean: Float[Array, " n"] = _check_finite_array(mean_arr, "mean")
    checked_covariance: Float[Array, "n n"] = _check_finite_array(
        covariance_arr,
        "covariance",
    )
    checked_rhat: Float[Array, " n"] = _check_finite_array(rhat_arr, "rhat")
    checked_ess: Float[Array, " n"] = _check_finite_array(ess_arr, "ess")
    posterior: PosteriorSamples = PosteriorSamples(
        samples=checked_samples,
        mean=checked_mean,
        covariance=checked_covariance,
        rhat=checked_rhat,
        ess=checked_ess,
        converged=converged_arr,
    )
    return posterior


__all__: list[str] = [
    "LaplaceUncertainty",
    "PosteriorSamples",
    "ReconProblem",
    "ReconResult",
    "create_laplace_uncertainty",
    "create_posterior_samples",
    "create_recon_problem",
    "create_recon_result",
]
