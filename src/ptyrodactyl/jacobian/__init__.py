"""
Jacobian-based analysis and solvers for ptychographic inverse problems.

This module provides tools for characterizing the observable subspace,
computing singular spectra, and solving nonlinear least-squares problems
using second-order methods that expose gauge structure.

Submodules
----------
- `operators`:
    JVP, VJP, and Hessian-vector product primitives
- `solvers`:
    Gauss-Newton, Levenberg-Marquardt, Lanczos, and Krylov methods
- `gauge`:
    Nullspace analysis and gauge orbit characterization
- `fisher`:
    Fisher information, Schur complements, and experiment design
- `blocks`:
    Block-based Gauss-Newton solver for ptychography
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