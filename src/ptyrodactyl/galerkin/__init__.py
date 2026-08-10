"""Globally coupled scalar Fourier-Galerkin Helmholtz scattering.

Extended Summary
----------------
The submodules are organized as follows:

- :mod:`acquisition`
    Checked finite scalar acquisition-support eligibility.
- :mod:`action_enclosures`
    Per-state rounded action and independent residual enclosures.
- :mod:`coefficient_certification`
    Concrete-host exact voxel-coefficient certification.
- :mod:`derivatives`
    Fixed-support implicit Galerkin derivative harness.
- :mod:`enclosures`
    Fixed-linear RM-S2 realization-error enclosures.
- :mod:`engine`
    Fixed-support scalar Galerkin actions and iterative solves.
- :mod:`local_cell`
    Rounded LVT-1 local-cell coefficients and formal metric adjoint.
- :mod:`local_cell_certification`
    Direct host LVT.7 coefficient certification and replay.
- :mod:`local_cell_interaction`
    Exact local-cell compression and fixed interaction action.
- :mod:`potential`
    Fixed-support SC-1 interaction and absorber products.
- :mod:`realization`
    VC-1 voxel-to-Galerkin potential realization and metric adjoint.
- :mod:`source_derivatives`
    Fixed-stratum represented total-source derivatives.
- :mod:`sources`
    Represented coherent source construction and evidence.
- :mod:`stability`
    Bounded exact per-result stability checker and invocation.
- :mod:`system`
    Manifested SC-1 target actions, matched source, and physical residuals.
- :mod:`terminal`
    Matrix-free coordinate-terminal current diagnostics and enclosures.

Routine Listings
----------------
:func:`apply_absorber_action`
    Apply the endpoint-safe fixed-support absorber product.
:func:`apply_galerkin_adjoint`
    Apply the matrix-free adjoint Galerkin operator.
:func:`apply_galerkin_operator`
    Apply the matrix-free forward Galerkin operator.
:func:`apply_galerkin_potential_metric_adjoint`
    Apply the VC-1 coefficient map adjoint in the physical voxel metric.
:func:`apply_galerkin_target`
    Apply the manifested matrix-free SC-1 target.
:func:`apply_galerkin_target_adjoint`
    Apply the actual adjoint of the manifested SC-1 target.
:func:`apply_galerkin_terminal_current`
    Apply the Hermitian selected-fiber current action.
:func:`apply_galerkin_terminal_normal_derivative`
    Apply the selected oriented normal-derivative trace.
:func:`apply_galerkin_terminal_normal_derivative_adjoint`
    Apply the selected coefficient-metric normal-trace adjoint.
:func:`apply_galerkin_terminal_trace`
    Apply the selected coordinate field trace.
:func:`apply_galerkin_terminal_trace_adjoint`
    Apply the selected coefficient-metric field-trace adjoint.
:func:`apply_interaction_product`
    Apply the endpoint-safe fixed-support interaction product.
:func:`apply_local_cell_interaction`
    Apply the frozen rounded LVT.16 interaction action.
:func:`apply_local_cell_interaction_adjoint`
    Apply the formal matrix adjoint of the frozen rounded interaction.
:func:`apply_local_cell_potential_metric_adjoint`
    Apply the rounded callable's adjoint in the physical cell metric.
:func:`build_absorber_factor`
    Build a bounded dense diagnostic factor for one absorber compression.
:func:`build_cosine_shell_absorber_coefficients`
    Build analytic coefficients of the bounded periodic shell absorber.
:func:`build_galerkin_fixed_linear_error_ledger`
    Build a fixed-linear error ledger from manifested physical inputs.
:func:`build_interaction_coefficients`
    Build SC-1 interaction coefficients from voltage coefficients.
:func:`build_represented_focused_galerkin_source`
    Build a coherent stored-shell finite focused source.
:func:`build_represented_plane_galerkin_source`
    Build one stored-shell represented forward plane mode.
:func:`certify_galerkin_potential_realization`
    Refine one concrete realization with direct pairwise host evidence.
:func:`certify_local_cell_exact_compression`
    Certify exact local-cell compression and its fixed interaction error.
:func:`certify_local_cell_galerkin_potential`
    Certify an actual Hermitian approximant directly against LVT.7.
:func:`cgls_solve`
    Solve a Galerkin system with CGLS and a fresh residual.
:func:`check_galerkin_absorber_floor`
    Build the bounded exact-stored-RHS algebraic-oracle proof.
:func:`check_galerkin_acquisition_support`
    Check one bounded acquisition manifest against RM-S1.
:func:`check_represented_galerkin_absorber_floor`
    Build a Route-A proof with an eligible rebuilt RM-S3 source.
:func:`create_galerkin_target`
    Build one production SC-1 target from a shared voxel potential.
:func:`create_host_checked_galerkin_target`
    Build a target after a concrete-host coefficient-certificate attempt.
:func:`create_local_cell_interaction_core`
    Create a non-solver-ready core from finite replayed LVT evidence.
:func:`create_matched_galerkin_source`
    Construct a rounded finite matched-source realization.
:func:`enclose_galerkin_residual`
    Enclose an independently formed same-``H_alg`` residual.
:func:`enclose_galerkin_target_action`
    Enclose one rounded production action at a submitted state.
:func:`enclose_galerkin_terminal_current`
    Enclose one submitted-state exact selected-fiber current.
:func:`evaluate_galerkin_adjoint_residual`
    Evaluate a fresh adjoint-system algebraic residual.
:func:`evaluate_galerkin_residual`
    Evaluate a fresh forward-system algebraic residual.
:func:`evaluate_galerkin_terminal_current`
    Evaluate the rounded selected-fiber current quadratic form.
:func:`evaluate_physical_galerkin_adjoint_residual`
    Recompute an adjoint-system residual by direct coefficient lookup.
:func:`evaluate_physical_galerkin_residual`
    Recompute a forward-system residual by direct coefficient lookup.
:func:`galerkin_state_jvp`
    Evaluate the physical fixed-support Galerkin state JVP.
:func:`galerkin_state_vjp`
    Evaluate the physical fixed-support Galerkin state VJP.
:func:`implicit_galerkin_solve`
    Solve a Galerkin root with an implicit custom VJP.
:func:`invoke_galerkin_stability`
    Recheck and apply the exact-stored-RHS algebraic-oracle route.
:func:`invoke_represented_galerkin_stability`
    Recheck and apply an eligible represented-source stability proof.
:func:`lsqr_solve`
    Solve a Galerkin system with LSQR and a fresh residual.
:func:`prepare_local_cell_interaction_core`
    Replay-authenticate stored core data before transform-compatible use.
:func:`realize_galerkin_potential`
    Realize a periodic voxel potential on one interaction support.
:func:`realize_local_cell_galerkin_potential`
    Realize a periodic local-cell voltage field on one interaction support.
:func:`represented_total_source_jvp`
    Evaluate the fixed-stratum represented total-source JVP.
:func:`represented_total_source_vjp`
    Evaluate the fixed-stratum represented total-source VJP.
:func:`shifted_free_diagonal`
    Construct the carrier-shifted free Galerkin diagonal.

"""

from .acquisition import check_galerkin_acquisition_support
from .action_enclosures import (
    enclose_galerkin_residual,
    enclose_galerkin_target_action,
)
from .coefficient_certification import certify_galerkin_potential_realization
from .derivatives import galerkin_state_jvp, galerkin_state_vjp
from .enclosures import build_galerkin_fixed_linear_error_ledger
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
from .local_cell import (
    apply_local_cell_potential_metric_adjoint,
    realize_local_cell_galerkin_potential,
)
from .local_cell_certification import certify_local_cell_galerkin_potential
from .local_cell_interaction import (
    apply_local_cell_interaction,
    apply_local_cell_interaction_adjoint,
    certify_local_cell_exact_compression,
    create_local_cell_interaction_core,
    prepare_local_cell_interaction_core,
)
from .potential import (
    apply_absorber_action,
    apply_interaction_product,
    build_absorber_factor,
    build_cosine_shell_absorber_coefficients,
    build_interaction_coefficients,
)
from .realization import (
    apply_galerkin_potential_metric_adjoint,
    realize_galerkin_potential,
)
from .source_derivatives import (
    represented_total_source_jvp,
    represented_total_source_vjp,
)
from .sources import (
    build_represented_focused_galerkin_source,
    build_represented_plane_galerkin_source,
)
from .stability import (
    check_galerkin_absorber_floor,
    check_represented_galerkin_absorber_floor,
    invoke_galerkin_stability,
    invoke_represented_galerkin_stability,
)
from .system import (
    apply_galerkin_target,
    apply_galerkin_target_adjoint,
    create_galerkin_target,
    create_host_checked_galerkin_target,
    create_matched_galerkin_source,
    evaluate_physical_galerkin_adjoint_residual,
    evaluate_physical_galerkin_residual,
)
from .terminal import (
    apply_galerkin_terminal_current,
    apply_galerkin_terminal_normal_derivative,
    apply_galerkin_terminal_normal_derivative_adjoint,
    apply_galerkin_terminal_trace,
    apply_galerkin_terminal_trace_adjoint,
    enclose_galerkin_terminal_current,
    evaluate_galerkin_terminal_current,
)

__all__ = [
    "apply_absorber_action",
    "apply_galerkin_adjoint",
    "apply_galerkin_operator",
    "apply_galerkin_potential_metric_adjoint",
    "apply_galerkin_target",
    "apply_galerkin_target_adjoint",
    "apply_galerkin_terminal_current",
    "apply_galerkin_terminal_normal_derivative",
    "apply_galerkin_terminal_normal_derivative_adjoint",
    "apply_galerkin_terminal_trace",
    "apply_galerkin_terminal_trace_adjoint",
    "apply_interaction_product",
    "apply_local_cell_interaction",
    "apply_local_cell_interaction_adjoint",
    "apply_local_cell_potential_metric_adjoint",
    "build_absorber_factor",
    "build_cosine_shell_absorber_coefficients",
    "build_galerkin_fixed_linear_error_ledger",
    "build_interaction_coefficients",
    "build_represented_focused_galerkin_source",
    "build_represented_plane_galerkin_source",
    "cgls_solve",
    "check_galerkin_absorber_floor",
    "check_galerkin_acquisition_support",
    "check_represented_galerkin_absorber_floor",
    "certify_galerkin_potential_realization",
    "certify_local_cell_exact_compression",
    "certify_local_cell_galerkin_potential",
    "create_galerkin_target",
    "create_host_checked_galerkin_target",
    "create_local_cell_interaction_core",
    "create_matched_galerkin_source",
    "enclose_galerkin_residual",
    "enclose_galerkin_target_action",
    "enclose_galerkin_terminal_current",
    "evaluate_galerkin_adjoint_residual",
    "evaluate_galerkin_residual",
    "evaluate_galerkin_terminal_current",
    "evaluate_physical_galerkin_adjoint_residual",
    "evaluate_physical_galerkin_residual",
    "galerkin_state_jvp",
    "galerkin_state_vjp",
    "implicit_galerkin_solve",
    "invoke_galerkin_stability",
    "invoke_represented_galerkin_stability",
    "lsqr_solve",
    "prepare_local_cell_interaction_core",
    "realize_galerkin_potential",
    "realize_local_cell_galerkin_potential",
    "represented_total_source_jvp",
    "represented_total_source_vjp",
    "shifted_free_diagonal",
]
