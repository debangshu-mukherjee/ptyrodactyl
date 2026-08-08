"""Scalar Galerkin scattering and Green-function utilities.

Extended Summary
----------------
The submodules are organized as follows:

- :mod:`derivatives`
    Fixed-support implicit Galerkin derivative harness.
- :mod:`engine`
    Fixed-support scalar Galerkin actions and iterative solves.
- :mod:`green`
    Fourier-space Green's function for the homogeneous Helmholtz equation.
- :mod:`potential`
    Fixed-support SC-1 interaction and absorber products.
- :mod:`stability`
    Bounded exact per-result stability checker and invocation.
- :mod:`system`
    Manifested SC-1 target actions, matched source, and physical residuals.

Routine Listings
----------------
:func:`apply_absorber_action`
    Apply the endpoint-safe fixed-support absorber product.
:func:`apply_galerkin_adjoint`
    Apply the matrix-free adjoint Galerkin operator.
:func:`apply_galerkin_operator`
    Apply the matrix-free forward Galerkin operator.
:func:`apply_galerkin_target`
    Apply the manifested matrix-free SC-1 target.
:func:`apply_galerkin_target_adjoint`
    Apply the actual adjoint of the manifested SC-1 target.
:func:`apply_interaction_product`
    Apply the endpoint-safe fixed-support interaction product.
:func:`build_absorber_factor`
    Build a bounded dense diagnostic factor for one absorber compression.
:func:`build_cosine_shell_absorber_coefficients`
    Build analytic coefficients of the bounded periodic shell absorber.
:func:`build_interaction_coefficients`
    Build SC-1 interaction coefficients from voltage coefficients.
:func:`cgls_solve`
    Solve a Galerkin system with CGLS and a fresh residual.
:func:`check_galerkin_absorber_floor`
    Build an exact bounded Route-A proof for one submitted result.
:func:`convergence_parameter`
    Compute the convergence parameter from the scattering potential.
:func:`create_matched_galerkin_source`
    Construct a rounded finite matched-source realization.
:func:`evaluate_galerkin_adjoint_residual`
    Evaluate a fresh adjoint-system algebraic residual.
:func:`evaluate_galerkin_residual`
    Evaluate a fresh forward-system algebraic residual.
:func:`evaluate_physical_galerkin_adjoint_residual`
    Recompute an adjoint-system residual by direct coefficient lookup.
:func:`evaluate_physical_galerkin_residual`
    Recompute a forward-system residual by direct coefficient lookup.
:func:`galerkin_state_jvp`
    Evaluate a fixed-support Galerkin state and parameter JVP.
:func:`galerkin_state_vjp`
    Evaluate a fixed-support Galerkin state and parameter VJP.
:func:`green_function_fourier`
    Construct the Fourier-space Green's function.
:func:`implicit_galerkin_solve`
    Solve a Galerkin root with an implicit custom VJP.
:func:`invoke_galerkin_stability`
    Recheck and apply one per-result bound as pass, fallback, or rejection.
:func:`lsqr_solve`
    Solve a Galerkin system with LSQR and a fresh residual.
:func:`reciprocal_coords`
    Construct 3-D reciprocal-space coordinate arrays.
:func:`shifted_free_diagonal`
    Construct the carrier-shifted free Galerkin diagonal.
:func:`wavenumber_background`
    Compute the optimal background wavenumber squared.

"""

from .derivatives import galerkin_state_jvp, galerkin_state_vjp
from .engine import (
    apply_galerkin_adjoint,
    apply_galerkin_operator,
    cgls_solve,
    evaluate_galerkin_adjoint_residual,
    evaluate_galerkin_residual,
    implicit_galerkin_solve,
    lsqr_solve,
    shifted_free_diagonal,
)
from .green import (
    convergence_parameter,
    green_function_fourier,
    reciprocal_coords,
    wavenumber_background,
)
from .potential import (
    apply_absorber_action,
    apply_interaction_product,
    build_absorber_factor,
    build_cosine_shell_absorber_coefficients,
    build_interaction_coefficients,
)
from .stability import (
    check_galerkin_absorber_floor,
    invoke_galerkin_stability,
)
from .system import (
    apply_galerkin_target,
    apply_galerkin_target_adjoint,
    create_matched_galerkin_source,
    evaluate_physical_galerkin_adjoint_residual,
    evaluate_physical_galerkin_residual,
)

__all__ = [
    "apply_absorber_action",
    "apply_galerkin_adjoint",
    "apply_galerkin_operator",
    "apply_galerkin_target",
    "apply_galerkin_target_adjoint",
    "apply_interaction_product",
    "build_absorber_factor",
    "build_cosine_shell_absorber_coefficients",
    "build_interaction_coefficients",
    "cgls_solve",
    "check_galerkin_absorber_floor",
    "convergence_parameter",
    "create_matched_galerkin_source",
    "evaluate_galerkin_adjoint_residual",
    "evaluate_galerkin_residual",
    "evaluate_physical_galerkin_adjoint_residual",
    "evaluate_physical_galerkin_residual",
    "galerkin_state_jvp",
    "galerkin_state_vjp",
    "green_function_fourier",
    "implicit_galerkin_solve",
    "invoke_galerkin_stability",
    "lsqr_solve",
    "reciprocal_coords",
    "shifted_free_diagonal",
    "wavenumber_background",
]
