"""Jacobian-based analysis and solvers for ptychographic problems.

Extended Summary
----------------
Provides tools for characterizing observable subspaces, computing
singular spectra, and solving nonlinear least-squares problems using
second-order methods that expose gauge structure.

The submodules are organized as follows:

- :mod:`blocks`
    Block-structured parameter management for ptychography.
- :mod:`fisher`
    Fisher information for ptychographic experiment design.
- :mod:`gauge`
    Gauge structure analysis for ptychographic inverse problems.
- :mod:`operators`
    Jacobian operator primitives for matrix-free linear algebra.
- :mod:`solvers`
    Second-order solvers and spectral analysis for least-squares.

Routine Listings
----------------
:func:`a_optimality`
    Compute A-optimality criterion: trace(F^{-1}).
:func:`alternating_block_solve`
    Solve via alternating block updates following a schedule.
:func:`block_gauss_newton_step`
    Perform a Gauss-Newton step updating only specified blocks.
:func:`block_jacobian_operator`
    Construct a JVP operator for a single parameter block.
:func:`block_jtj_operator`
    Construct a J^T J operator for a single parameter block.
:func:`block_vjp_operator`
    Construct a VJP operator for a single parameter block.
:func:`compute_block_gradient`
    Compute the gradient J^T r for a single parameter block.
:func:`condition_number`
    Compute condition number of Fisher information matrix.
:func:`conjugate_gradient`
    Solve A x = b via conjugate gradient.
:func:`cross_block_jtj_operator`
    Construct a cross-block J^T J operator.
:func:`d_optimality`
    Compute D-optimality criterion: log det(F).
:func:`decompose_gauge_observable`
    Decompose a perturbation into gauge and observable parts.
:func:`e_optimality`
    Compute E-optimality criterion: lambda_min(F).
:func:`effective_fisher`
    Compute Fisher information after marginalising nuisances.
:func:`effective_nullspace_dimension`
    Count singular values below the noise floor.
:func:`effective_rank`
    Count observable dimensions above the noise floor.
:func:`fisher_diagonal`
    Estimate diagonal of Fisher information via Hutchinson.
:func:`fisher_eigenspectrum`
    Estimate eigenspectrum of Fisher information via Lanczos.
:func:`fisher_information`
    Compute the Fisher information matrix.
:func:`fisher_information_operator`
    Construct a matrix-free Fisher information operator.
:func:`gauge_invariant_norm`
    Compute the norm in quotient space modulo gauge.
:func:`gauge_orbit_distance`
    Compute distance between two points modulo gauge.
:func:`gauss_newton_solve`
    Solve nonlinear least-squares via iterated Gauss-Newton.
:func:`gauss_newton_step`
    Compute a single Gauss-Newton update step.
:func:`hvp_gauss_newton`
    Construct the Gauss-Newton Hessian-vector product operator.
:func:`information_gain`
    Compute information gain from adding measurements.
:func:`jtj_operator`
    Construct the normal equations operator J^T J.
:func:`jvp_operator`
    Construct a Jacobian-vector product operator J @ v.
:func:`lanczos_tridiagonal`
    Compute the Lanczos tridiagonalisation of a symmetric operator.
:func:`levenberg_marquardt_solve`
    Solve nonlinear least-squares via Levenberg-Marquardt.
:func:`levenberg_marquardt_step`
    Compute a single Levenberg-Marquardt update step.
:func:`nullspace_vectors_lanczos`
    Estimate basis vectors for the Jacobian nullspace.
:func:`optimal_weights_e_criterion`
    Find optimal weights under E-optimality.
:func:`project_to_nullspace`
    Project a perturbation onto the gauge (nullspace) subspace.
:func:`project_to_observable`
    Project a perturbation onto the observable subspace.
:func:`random_gauge_direction`
    Sample a random direction from the gauge subspace.
:func:`schur_complement`
    Marginalise nuisance parameters via Schur complement.
:func:`singular_spectrum`
    Estimate the singular spectrum of the Jacobian.
:func:`split_params`
    Extract individual parameter blocks from combined params.
:func:`stack_fisher`
    Combine Fisher matrices from multiple conditions.
:func:`vjp_operator`
    Construct a vector-Jacobian product operator J^T @ u.

"""

from .blocks import (
    alternating_block_solve,
    block_gauss_newton_step,
    block_jacobian_operator,
    block_jtj_operator,
    block_vjp_operator,
    compute_block_gradient,
    cross_block_jtj_operator,
    split_params,
)
from .fisher import (
    a_optimality,
    condition_number,
    d_optimality,
    e_optimality,
    effective_fisher,
    fisher_diagonal,
    fisher_eigenspectrum,
    fisher_information,
    fisher_information_operator,
    information_gain,
    optimal_weights_e_criterion,
    schur_complement,
    stack_fisher,
)
from .gauge import (
    decompose_gauge_observable,
    effective_rank,
    gauge_invariant_norm,
    gauge_orbit_distance,
    nullspace_vectors_lanczos,
    project_to_nullspace,
    project_to_observable,
    random_gauge_direction,
)
from .operators import (
    hvp_gauss_newton,
    jtj_operator,
    jvp_operator,
    vjp_operator,
)
from .solvers import (
    conjugate_gradient,
    effective_nullspace_dimension,
    gauss_newton_solve,
    gauss_newton_step,
    lanczos_tridiagonal,
    levenberg_marquardt_solve,
    levenberg_marquardt_step,
    singular_spectrum,
)

__all__: list[str] = [
    "a_optimality",
    "alternating_block_solve",
    "block_gauss_newton_step",
    "block_jacobian_operator",
    "block_jtj_operator",
    "block_vjp_operator",
    "compute_block_gradient",
    "condition_number",
    "conjugate_gradient",
    "cross_block_jtj_operator",
    "d_optimality",
    "decompose_gauge_observable",
    "e_optimality",
    "effective_fisher",
    "effective_nullspace_dimension",
    "effective_rank",
    "fisher_diagonal",
    "fisher_eigenspectrum",
    "fisher_information",
    "fisher_information_operator",
    "gauge_invariant_norm",
    "gauge_orbit_distance",
    "gauss_newton_solve",
    "gauss_newton_step",
    "hvp_gauss_newton",
    "information_gain",
    "jtj_operator",
    "jvp_operator",
    "lanczos_tridiagonal",
    "levenberg_marquardt_solve",
    "levenberg_marquardt_step",
    "nullspace_vectors_lanczos",
    "optimal_weights_e_criterion",
    "project_to_nullspace",
    "project_to_observable",
    "random_gauge_direction",
    "schur_complement",
    "singular_spectrum",
    "split_params",
    "stack_fisher",
    "vjp_operator",
]
