"""Globally coupled scalar Fourier-Galerkin Helmholtz scattering.

Extended Summary
----------------
The submodules are organized as follows:

- :mod:`absorber`
    Axial local-cell CAP realization, certification, floor proof, and action.
- :mod:`acquisition`
    Checked finite scalar acquisition-support eligibility.
- :mod:`action_enclosures`
    Per-state rounded action and independent residual enclosures.
- :mod:`coefficient_certification`
    Concrete-host exact voxel-coefficient certification.
- :mod:`derivatives`
    Fixed-support implicit Galerkin derivative harness.
- :mod:`detector`
    Local positive-port, passive-pixel, detector, and likelihood composition.
- :mod:`enclosures`
    Fixed-linear RM-S2 realization-error enclosures.
- :mod:`engine`
    Fixed-support scalar Galerkin actions and iterative solves.
- :mod:`free_geometry`
    Route-neutral exact-carrier and shifted-free geometry evidence.
- :mod:`local_cell`
    Rounded LVT-1 local-cell coefficients and formal metric adjoint.
- :mod:`local_cell_certification`
    Direct host LVT.7 coefficient certification and replay.
- :mod:`local_cell_interaction`
    Exact local-cell compression and fixed interaction action.
- :mod:`local_cell_system`
    Solver-ready LOCAL_CELL_LVT1 target composition and actions.
- :mod:`local_projection`
    Exact local projection-defect Gram, measurement, and state lift.
- :mod:`local_represented_sources`
    Exact-shell represented local sources and direct action evidence.
- :mod:`local_sources`
    Disjoint LVT.20 local additional-source realization and certification.
- :mod:`local_stability`
    Bounded exact-dyadic local represented-source stability certification.
- :mod:`local_terminal`
    Authenticated local coordinate-terminal current operators and evidence.
- :mod:`local_vacuum_propagation`
    Strict local vacuum roots and homogeneous Cauchy propagators.
- :mod:`local_vacuum_terminal`
    Composed local vacuum-terminal continuation evidence.
- :mod:`local_zero_slab`
    Exact LVT.21--LVT.22 local zero-slab certification and replay.
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
- :mod:`terminal_derivatives`
    Rounded selected-sector terminal number-current derivatives.

Routine Listings
----------------
:func:`apply_absorber_action`
    Apply the endpoint-safe fixed-support absorber product.
:func:`apply_axial_physical_cap`
    Apply the frozen physical algebraic CAP ``B_alg``.
:func:`apply_axial_physical_cap_adjoint`
    Apply the formal matrix adjoint of frozen physical ``B_alg``.
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
:func:`apply_local_cell_additional_source_map`
    Apply rounded LVT.20b--LVT.20c without Hermitian projection.
:func:`apply_local_cell_additional_source_metric_adjoint`
    Apply the frozen rounded linear factors' formal metric adjoint.
:func:`apply_local_cell_galerkin_target`
    Apply frozen ``H_alg = D_alg - R_alg - i B_alg``.
:func:`apply_local_cell_galerkin_target_adjoint`
    Apply the explicit formal adjoint of the same frozen target.
:func:`apply_local_cell_interaction`
    Apply the frozen rounded LVT.16 interaction action.
:func:`apply_local_cell_interaction_adjoint`
    Apply the formal matrix adjoint of the frozen rounded interaction.
:func:`apply_local_cell_potential_metric_adjoint`
    Apply the rounded callable's adjoint in the physical cell metric.
:func:`apply_local_terminal_current`
    Apply the implicit actual frozen Hermitian current matrix.
:func:`apply_local_terminal_normal_derivative`
    Apply the side-oriented frozen physical normal trace.
:func:`apply_local_terminal_normal_derivative_adjoint`
    Apply the actual conjugate transpose of the frozen normal trace.
:func:`apply_local_terminal_trace`
    Apply the frozen carrier-stripped trace at the bound coordinate.
:func:`apply_local_terminal_trace_adjoint`
    Apply the actual conjugate transpose of the frozen trace.
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
:func:`certify_axial_cap_floor`
    Certify exact LVT.29a and, independently, realized LVT.32 floors.
:func:`certify_axial_cell_absorber`
    Certify a finite Hermitian approximant against exact LVT.24.
:func:`certify_galerkin_potential_realization`
    Refine one concrete realization with direct pairwise host evidence.
:func:`certify_galerkin_terminal_current_operator`
    Certify the uniform frozen selected-sector current operator.
:func:`certify_local_additional_source`
    Full-replay the parent/map and directly certify exact LVT.20c.
:func:`certify_local_cell_exact_compression`
    Certify exact local-cell compression and its fixed interaction error.
:func:`certify_local_cell_galerkin_potential`
    Certify an actual Hermitian approximant directly against LVT.7.
:func:`certify_local_censored_poisson_detector`
    Replay all pixels and certify one fixed censored-Poisson detector.
:func:`certify_local_passive_pixel_forms`
    Replay one positive port and certify its primitive passive pixels.
:func:`certify_local_positive_port`
    Replay L8 and compose one explicit projected or outgoing port.
:func:`certify_local_represented_source`
    Directly enclose exact ``D/B/R/S/M/T/C`` source actions.
:func:`certify_local_terminal_current_operator`
    Replay a local target and certify one scoped coordinate operator.
:func:`certify_local_vacuum_terminal`
    Compose one scoped LVT.39--LVT.56 vacuum-terminal certificate.
:func:`certify_local_zero_slab`
    Replay represented-source evidence and certify exact LVT.21--LVT.22.
:func:`cgls_adjoint_solve`
    Solve an actual adjoint Galerkin system with CGLS and a fresh residual.
:func:`cgls_solve`
    Solve a Galerkin system with CGLS and a fresh residual.
:func:`check_galerkin_absorber_floor`
    Build the bounded exact-stored-RHS algebraic-oracle proof.
:func:`check_galerkin_acquisition_support`
    Check one bounded acquisition manifest against RM-S1.
:func:`check_local_represented_galerkin_absorber_floor`
    Build one bounded exact-dyadic proof for a submitted state.
:func:`check_represented_galerkin_absorber_floor`
    Build a Route-A proof with an eligible rebuilt RM-S3 source.
:func:`classify_local_vacuum_root`
    Strictly classify and enclose one exact rational LVT.39 quantity.
:func:`compose_local_cell_galerkin_target`
    Replay L2--L4 and compose one solver-ready local-cell target.
:func:`compose_local_represented_focused_source`
    Compose one coherent exact-shell focused finite source.
:func:`compose_local_represented_plane_source`
    Compose one exact-shell represented plane mode.
:func:`create_galerkin_target`
    Build one production SC-1 target from a shared voxel potential.
:func:`create_host_checked_galerkin_target`
    Build a target after a concrete-host coefficient-certificate attempt.
:func:`create_local_cell_interaction_core`
    Create a non-solver-ready core from finite replayed LVT evidence.
:func:`create_local_censored_poisson_detector_input_manifest`
    Authenticate primitive detector inputs and every pixel replay input.
:func:`create_local_passive_pixel_input_manifest`
    Authenticate primitive pixel inputs and independent upstream policy.
:func:`create_matched_galerkin_source`
    Construct a rounded finite matched-source realization.
:func:`enclose_galerkin_residual`
    Enclose an independently formed same-``H_alg`` residual.
:func:`enclose_galerkin_target_action`
    Enclose one rounded production action at a submitted state.
:func:`enclose_galerkin_terminal_current`
    Enclose one submitted-state exact selected-fiber current.
:func:`enclose_galerkin_terminal_current_action`
    Enclose one frozen current action after certificate authentication.
:func:`enclose_local_cell_tail`
    Enclose the complete LVT.9 Fourier tail from authenticated evidence.
:func:`enclose_local_censored_poisson_likelihood`
    Enclose full-channel probabilities and fit-only pre-gain NLL.
:func:`enclose_local_projection_defect`
    Enclose scoped LVT.34--LVT.40 and LVT.55c--LVT.55e evidence.
:func:`enclose_local_terminal_current`
    Replay the operator and enclose one direct exact-target current.
:func:`enclose_local_terminal_current_action`
    Replay the operator and enclose one frozen current action.
:func:`enclose_local_vacuum_propagator`
    Enclose one homogeneous branch-specific Cauchy propagator.
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
:func:`galerkin_terminal_number_current_jvp`
    Evaluate the rounded selected-sector terminal number-current JVP.
:func:`galerkin_terminal_number_current_vjp`
    Evaluate the rounded selected-sector terminal number-current VJP.
:func:`implicit_galerkin_solve`
    Solve a Galerkin root with an implicit custom VJP.
:func:`invoke_galerkin_stability`
    Recheck and apply the exact-stored-RHS algebraic-oracle route.
:func:`invoke_local_represented_galerkin_stability`
    Recheck the proof and return a fully nested stability result.
:func:`invoke_represented_galerkin_stability`
    Recheck and apply an eligible represented-source stability proof.
:func:`lsqr_solve`
    Solve a Galerkin system with LSQR and a fresh residual.
:func:`make_local_vacuum_zero_witness`
    Bind equality of two canonical formal algebraic normal forms.
:func:`prepare_axial_cap_floor`
    Replay all nested public evidence before transform-compatible use.
:func:`prepare_galerkin_terminal_current_diagnostic`
    Host-authenticate one provisional coordinate-current diagnostic.
:func:`prepare_local_additional_source_certificate`
    Full-replay target, map, q rectangles, budget, and all digests.
:func:`prepare_local_cell_galerkin_target`
    Full-reconstruct and exact-compare a submitted local-cell target.
:func:`prepare_local_cell_interaction_core`
    Replay-authenticate stored core data before transform-compatible use.
:func:`prepare_local_censored_poisson_detector`
    Replay independent detector inputs and exact-compare every field.
:func:`prepare_local_censored_poisson_likelihood`
    Replay a likelihood from independent inputs and exact-compare it.
:func:`prepare_local_galerkin_stability_result`
    Full-replay and exact-compare one local stability result.
:func:`prepare_local_passive_pixel_forms`
    Replay independently supplied pixel inputs and exact-compare storage.
:func:`prepare_local_positive_port_certificate`
    Replay an independently specified L8 port and exact-compare storage.
:func:`prepare_local_projection_defect_certificate`
    Replay both parents, policies, exact Gram arithmetic, and all digests.
:func:`prepare_local_represented_source`
    Full-reconstruct and exact-compare a represented source.
:func:`prepare_local_represented_source_certificate`
    Full-reconstruct source, rectangles, budget, and certificate digests.
:func:`prepare_local_terminal_current`
    Replay complete operator, action, exact current, policy, and evidence.
:func:`prepare_local_terminal_current_action`
    Replay complete operator, field, frozen action, policy, and evidence.
:func:`prepare_local_terminal_current_operator`
    Replay raw operator storage and return the prepared JIT capability.
:func:`prepare_local_vacuum_propagator`
    Full-replay and exact-compare one homogeneous propagator.
:func:`prepare_local_vacuum_root_certificate`
    Full-replay and exact-compare one strict root certificate.
:func:`prepare_local_vacuum_terminal_certificate`
    Replay every L8 parent, policy, helper route, field, and digest.
:func:`prepare_local_zero_slab_certificate`
    Replay every nested carrier, exact predicate, transcript, and digest.
:func:`realize_axial_cell_absorber`
    Realize one canonical Hermitian LVT.24 coefficient approximant.
:func:`realize_galerkin_potential`
    Realize a periodic voxel potential on one interaction support.
:func:`realize_local_cell_additional_source`
    Full-replay the target and realize complex LVT.20a--LVT.20c.
:func:`realize_local_cell_galerkin_potential`
    Realize a periodic local-cell voltage field on one interaction support.
:func:`realize_zero_local_additional_source`
    Full-replay the target and build the empty-carrier ZERO route.
:func:`represented_total_source_jvp`
    Evaluate the fixed-stratum represented total-source JVP.
:func:`represented_total_source_vjp`
    Evaluate the fixed-stratum represented total-source VJP.
:func:`shifted_free_diagonal`
    Construct the carrier-shifted free Galerkin diagonal.

"""

from .absorber import (
    apply_axial_physical_cap,
    apply_axial_physical_cap_adjoint,
    certify_axial_cap_floor,
    certify_axial_cell_absorber,
    prepare_axial_cap_floor,
    realize_axial_cell_absorber,
)
from .acquisition import check_galerkin_acquisition_support
from .action_enclosures import (
    enclose_galerkin_residual,
    enclose_galerkin_target_action,
)
from .coefficient_certification import certify_galerkin_potential_realization
from .derivatives import galerkin_state_jvp, galerkin_state_vjp
from .detector import (
    certify_local_censored_poisson_detector,
    certify_local_passive_pixel_forms,
    certify_local_positive_port,
    create_local_censored_poisson_detector_input_manifest,
    create_local_passive_pixel_input_manifest,
    enclose_local_censored_poisson_likelihood,
    prepare_local_censored_poisson_detector,
    prepare_local_censored_poisson_likelihood,
    prepare_local_passive_pixel_forms,
    prepare_local_positive_port_certificate,
)
from .enclosures import build_galerkin_fixed_linear_error_ledger
from .engine import (
    apply_galerkin_adjoint,
    apply_galerkin_operator,
    cgls_adjoint_solve,
    cgls_solve,
    evaluate_galerkin_adjoint_residual,
    evaluate_galerkin_residual,
    implicit_galerkin_solve,
    lsqr_solve,
    shifted_free_diagonal,
)
from .local_cell import (
    apply_local_cell_potential_metric_adjoint,
    enclose_local_cell_tail,
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
from .local_cell_system import (
    apply_local_cell_galerkin_target,
    apply_local_cell_galerkin_target_adjoint,
    compose_local_cell_galerkin_target,
    prepare_local_cell_galerkin_target,
)
from .local_projection import (
    enclose_local_projection_defect,
    prepare_local_projection_defect_certificate,
)
from .local_represented_sources import (
    certify_local_represented_source,
    compose_local_represented_focused_source,
    compose_local_represented_plane_source,
    prepare_local_represented_source,
    prepare_local_represented_source_certificate,
)
from .local_sources import (
    apply_local_cell_additional_source_map,
    apply_local_cell_additional_source_metric_adjoint,
    certify_local_additional_source,
    prepare_local_additional_source_certificate,
    realize_local_cell_additional_source,
    realize_zero_local_additional_source,
)
from .local_stability import (
    check_local_represented_galerkin_absorber_floor,
    invoke_local_represented_galerkin_stability,
    prepare_local_galerkin_stability_result,
)
from .local_terminal import (
    apply_local_terminal_current,
    apply_local_terminal_normal_derivative,
    apply_local_terminal_normal_derivative_adjoint,
    apply_local_terminal_trace,
    apply_local_terminal_trace_adjoint,
    certify_local_terminal_current_operator,
    enclose_local_terminal_current,
    enclose_local_terminal_current_action,
    prepare_local_terminal_current,
    prepare_local_terminal_current_action,
    prepare_local_terminal_current_operator,
)
from .local_vacuum_propagation import (
    classify_local_vacuum_root,
    enclose_local_vacuum_propagator,
    make_local_vacuum_zero_witness,
    prepare_local_vacuum_propagator,
    prepare_local_vacuum_root_certificate,
)
from .local_vacuum_terminal import (
    certify_local_vacuum_terminal,
    prepare_local_vacuum_terminal_certificate,
)
from .local_zero_slab import (
    certify_local_zero_slab,
    prepare_local_zero_slab_certificate,
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
    certify_galerkin_terminal_current_operator,
    enclose_galerkin_terminal_current,
    enclose_galerkin_terminal_current_action,
    evaluate_galerkin_terminal_current,
    prepare_galerkin_terminal_current_diagnostic,
)
from .terminal_derivatives import (
    galerkin_terminal_number_current_jvp,
    galerkin_terminal_number_current_vjp,
)

__all__ = [
    "apply_absorber_action",
    "apply_axial_physical_cap",
    "apply_axial_physical_cap_adjoint",
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
    "apply_local_cell_additional_source_map",
    "apply_local_cell_additional_source_metric_adjoint",
    "apply_local_cell_galerkin_target",
    "apply_local_cell_galerkin_target_adjoint",
    "apply_local_cell_interaction",
    "apply_local_cell_interaction_adjoint",
    "apply_local_cell_potential_metric_adjoint",
    "apply_local_terminal_current",
    "apply_local_terminal_normal_derivative",
    "apply_local_terminal_normal_derivative_adjoint",
    "apply_local_terminal_trace",
    "apply_local_terminal_trace_adjoint",
    "build_absorber_factor",
    "build_cosine_shell_absorber_coefficients",
    "build_galerkin_fixed_linear_error_ledger",
    "build_interaction_coefficients",
    "build_represented_focused_galerkin_source",
    "build_represented_plane_galerkin_source",
    "cgls_adjoint_solve",
    "cgls_solve",
    "check_galerkin_absorber_floor",
    "check_galerkin_acquisition_support",
    "check_local_represented_galerkin_absorber_floor",
    "check_represented_galerkin_absorber_floor",
    "classify_local_vacuum_root",
    "certify_galerkin_potential_realization",
    "certify_axial_cap_floor",
    "certify_axial_cell_absorber",
    "certify_galerkin_terminal_current_operator",
    "certify_local_additional_source",
    "certify_local_censored_poisson_detector",
    "certify_local_cell_exact_compression",
    "certify_local_cell_galerkin_potential",
    "certify_local_passive_pixel_forms",
    "certify_local_positive_port",
    "certify_local_represented_source",
    "certify_local_terminal_current_operator",
    "certify_local_vacuum_terminal",
    "certify_local_zero_slab",
    "compose_local_cell_galerkin_target",
    "compose_local_represented_focused_source",
    "compose_local_represented_plane_source",
    "create_galerkin_target",
    "create_host_checked_galerkin_target",
    "create_local_cell_interaction_core",
    "create_local_censored_poisson_detector_input_manifest",
    "create_local_passive_pixel_input_manifest",
    "create_matched_galerkin_source",
    "enclose_galerkin_residual",
    "enclose_galerkin_target_action",
    "enclose_galerkin_terminal_current",
    "enclose_galerkin_terminal_current_action",
    "enclose_local_cell_tail",
    "enclose_local_censored_poisson_likelihood",
    "enclose_local_projection_defect",
    "enclose_local_terminal_current",
    "enclose_local_terminal_current_action",
    "enclose_local_vacuum_propagator",
    "evaluate_galerkin_adjoint_residual",
    "evaluate_galerkin_residual",
    "evaluate_galerkin_terminal_current",
    "evaluate_physical_galerkin_adjoint_residual",
    "evaluate_physical_galerkin_residual",
    "galerkin_state_jvp",
    "galerkin_state_vjp",
    "galerkin_terminal_number_current_jvp",
    "galerkin_terminal_number_current_vjp",
    "implicit_galerkin_solve",
    "invoke_galerkin_stability",
    "invoke_local_represented_galerkin_stability",
    "invoke_represented_galerkin_stability",
    "lsqr_solve",
    "make_local_vacuum_zero_witness",
    "prepare_galerkin_terminal_current_diagnostic",
    "prepare_axial_cap_floor",
    "prepare_local_additional_source_certificate",
    "prepare_local_censored_poisson_detector",
    "prepare_local_censored_poisson_likelihood",
    "prepare_local_cell_galerkin_target",
    "prepare_local_cell_interaction_core",
    "prepare_local_galerkin_stability_result",
    "prepare_local_projection_defect_certificate",
    "prepare_local_passive_pixel_forms",
    "prepare_local_positive_port_certificate",
    "prepare_local_represented_source",
    "prepare_local_represented_source_certificate",
    "prepare_local_terminal_current",
    "prepare_local_terminal_current_action",
    "prepare_local_terminal_current_operator",
    "prepare_local_vacuum_propagator",
    "prepare_local_vacuum_root_certificate",
    "prepare_local_vacuum_terminal_certificate",
    "prepare_local_zero_slab_certificate",
    "realize_galerkin_potential",
    "realize_axial_cell_absorber",
    "realize_local_cell_additional_source",
    "realize_local_cell_galerkin_potential",
    "realize_zero_local_additional_source",
    "represented_total_source_jvp",
    "represented_total_source_vjp",
    "shifted_free_diagonal",
]
