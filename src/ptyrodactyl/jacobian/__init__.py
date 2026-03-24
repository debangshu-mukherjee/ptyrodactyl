"""Jacobian-based analysis and solvers for ptychographic problems.

Extended Summary
----------------
Provides tools for characterizing the observable subspace,
computing singular spectra, and solving nonlinear least-squares
problems using second-order methods that expose gauge structure.

The submodules are organised by concern:

- :mod:`operators` -- JVP, VJP, and Hessian-vector product
  primitives.
- :mod:`solvers` -- Gauss-Newton, Levenberg-Marquardt, Lanczos,
  and Krylov methods.
- :mod:`gauge` -- Nullspace analysis and gauge orbit
  characterisation.
- :mod:`fisher` -- Fisher information, Schur complements, and
  experiment design.
- :mod:`blocks` -- Block-based Gauss-Newton solver for
  ptychography.

Routine Listings
----------------
:class:`AberrationParams`
    Probe aberration parameters (NamedTuple).
:class:`ExitWaveParams`
    Exit wave parameters (NamedTuple).
:class:`GeometryParams`
    Geometric calibration parameters (NamedTuple).
:class:`PositionParams`
    Scan position error parameters (NamedTuple).
:class:`ProbeModeParams`
    Probe mode parameters for partial coherence (NamedTuple).
:class:`PtychoParams`
    Combined parameter container for all blocks (NamedTuple).
:func:`a_optimality`
    A-optimality criterion: trace(F^{-1}).
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
:func:`condition_number`
    Condition number of Fisher information matrix.
:func:`conjugate_gradient`
    Matrix-free CG solver for symmetric PSD systems.
:func:`cross_block_jtj_operator`
    Cross-block J^T J operator for Schur complements.
:func:`d_optimality`
    D-optimality criterion: log det(F).
:func:`decompose_gauge_observable`
    Split vector into gauge and observable components.
:func:`e_optimality`
    E-optimality criterion: lambda_min(F).
:func:`effective_fisher`
    Fisher information after marginalising nuisances.
:func:`effective_nullspace_dimension`
    Count dimensions below noise threshold.
:func:`effective_rank`
    Count observable dimensions above noise threshold.
:func:`fisher_diagonal`
    Fast diagonal approximation of Fisher information.
:func:`fisher_eigenspectrum`
    Eigenvalues of Fisher matrix via Lanczos.
:func:`fisher_information`
    Compute Fisher information matrix at a parameter point.
:func:`fisher_information_operator`
    Matrix-free Fisher information operator.
:func:`gauge_invariant_norm`
    Norm in quotient space (modulo gauge).
:func:`gauge_orbit_distance`
    Distance between two points modulo gauge.
:func:`gauss_newton_solve`
    Full Gauss-Newton iteration to convergence.
:func:`gauss_newton_step`
    Single Gauss-Newton update step.
:func:`hvp_gauss_newton`
    Gauss-Newton Hessian-vector product operator.
:func:`information_gain`
    Information gain from adding measurements.
:func:`jtj_operator`
    Normal equations operator J^T J @ v.
:func:`jvp_operator`
    Jacobian-vector product J @ v.
:func:`lanczos_tridiagonal`
    Lanczos algorithm for tridiagonalising operators.
:func:`levenberg_marquardt_solve`
    Full Levenberg-Marquardt iteration to convergence.
:func:`levenberg_marquardt_step`
    Single LM update step with adaptive damping.
:func:`make_ptycho_params`
    Construct combined PtychoParams from components.
:func:`nullspace_vectors_lanczos`
    Estimate nullspace basis via shifted inverse Lanczos.
:func:`optimal_weights_e_criterion`
    Optimal weights for stacking under E-optimality.
:func:`project_to_nullspace`
    Project perturbation onto gauge subspace.
:func:`project_to_observable`
    Project perturbation onto observable subspace.
:func:`random_gauge_direction`
    Sample a random direction from the gauge subspace.
:func:`schur_complement`
    Marginalise nuisance parameters via Schur complement.
:func:`singular_spectrum`
    Estimate singular values of Jacobian via Lanczos.
:func:`split_params`
    Extract individual blocks from combined params.
:func:`stack_fisher`
    Combine Fisher matrices from multiple conditions.
:func:`vjp_operator`
    Vector-Jacobian product J^T @ u.
"""

from .blocks import (
    ExitWaveParams,
    AberrationParams,
    GeometryParams,
    PositionParams,
    ProbeModeParams,
    PtychoParams,
    make_ptycho_params,
    split_params,
    block_jacobian_operator,
    block_vjp_operator,
    block_jtj_operator,
    cross_block_jtj_operator,
    compute_block_gradient,
    block_gauss_newton_step,
    alternating_block_solve,
)

from .fisher import (
    fisher_information,
    fisher_information_operator,
    fisher_diagonal,
    schur_complement,
    effective_fisher,
    fisher_eigenspectrum,
    a_optimality,
    d_optimality,
    e_optimality,
    stack_fisher,
    optimal_weights_e_criterion,
    condition_number,
    information_gain,
)

from .gauge import (
    nullspace_vectors_lanczos,
    project_to_nullspace,
    project_to_observable,
    decompose_gauge_observable,
    effective_rank,
    gauge_invariant_norm,
    random_gauge_direction,
    gauge_orbit_distance,
)

from .operators import (
    hvp_gauss_newton,
    jtj_operator,
    jvp_operator,
    vjp_operator,
)

from .solvers import (
    conjugate_gradient,
    gauss_newton_step,
    gauss_newton_solve,
    levenberg_marquardt_step,
    levenberg_marquardt_solve,
    lanczos_tridiagonal,
    singular_spectrum,
    effective_nullspace_dimension,
)

__all__ = [
    "jvp_operator",
    "vjp_operator",
    "jtj_operator",
    "hvp_gauss_newton",
    "conjugate_gradient",
    "gauss_newton_step",
    "gauss_newton_solve",
    "levenberg_marquardt_step",
    "levenberg_marquardt_solve",
    "lanczos_tridiagonal",
    "singular_spectrum",
    "effective_nullspace_dimension",
    "nullspace_vectors_lanczos",
    "project_to_nullspace",
    "project_to_observable",
    "decompose_gauge_observable",
    "effective_rank",
    "gauge_invariant_norm",
    "random_gauge_direction",
    "gauge_orbit_distance",
    "fisher_information",
    "fisher_information_operator",
    "fisher_diagonal",
    "schur_complement",
    "effective_fisher",
    "fisher_eigenspectrum",
    "a_optimality",
    "d_optimality",
    "e_optimality",
    "stack_fisher",
    "optimal_weights_e_criterion",
    "condition_number",
    "information_gain",
    "ExitWaveParams",
    "AberrationParams",
    "GeometryParams",
    "PositionParams",
    "ProbeModeParams",
    "PtychoParams",
    "make_ptycho_params",
    "split_params",
    "block_jacobian_operator",
    "block_vjp_operator",
    "block_jtj_operator",
    "cross_block_jtj_operator",
    "compute_block_gradient",
    "block_gauss_newton_step",
    "alternating_block_solve",
]
