"""Canonical public type and constant exports.

Extended Summary
----------------
This package is the canonical import surface for shared type
aliases and physical constants used throughout ptyrodactyl.

The submodules are organized as follows:

- :mod:`absorber_types`
    Axial local-cell CAP profile, coefficient, and floor evidence.
- :mod:`acquisition_types`
    Checked scalar Galerkin acquisition-support carriers.
- :mod:`action_error_types`
    Per-state Galerkin action and residual error evidence.
- :mod:`born_potential_types`
    Fixed Fourier supports for scalar Galerkin products.
- :mod:`born_types`
    Scalar Galerkin operator and solve-result carriers.
- :mod:`constants`
    Physical constants and derived electron-optics quantities.
- :mod:`crystal_types`
    Crystal carriers and factories.
- :mod:`custom_types`
    Custom type aliases for scalar and image data.
- :mod:`distributions`
    Probability distribution types for ptyrodactyl ensembles.
- :mod:`electron_types`
    Electron microscopy carriers and factories.
- :mod:`form_factor_types`
    Atomic form-factor coefficient carriers and factories.
- :mod:`galerkin_types`
    Production scalar Galerkin manifests and evidence carriers.
- :mod:`jacobian_types`
    Jacobian parameter and solver-state carriers.
- :mod:`local_cell_interaction_types`
    Exact local-cell compression and fixed interaction-core carriers.
- :mod:`local_cell_target_types`
    Solver-ready local-cell target and disjoint fixed-linear ledger carriers.
- :mod:`local_cell_types`
    Disjoint periodic local-cell potential and realization carriers.
- :mod:`local_detector_types`
    Local positive-port, passive-pixel, detector, and likelihood evidence.
- :mod:`local_projection_types`
    Exact local projection-defect Gram, measurement, and state-lift carriers.
- :mod:`local_represented_source_types`
    Disjoint represented local-source and direct action-evidence carriers.
- :mod:`local_source_types`
    Disjoint LVT.20 local additional-source carriers and direct evidence.
- :mod:`local_stability_types`
    Bounded local represented-source stability proof and result carriers.
- :mod:`local_terminal_types`
    Authenticated local coordinate-terminal current carriers.
- :mod:`local_vacuum_propagation_types`
    Exact local vacuum-root and homogeneous propagator evidence carriers.
- :mod:`local_vacuum_terminal_types`
    Composed local vacuum-terminal continuation evidence carriers.
- :mod:`local_zero_slab_types`
    Exact LVT.21--LVT.22 local zero-slab evidence carriers.
- :mod:`potential_types`
    Three-dimensional scalar-potential carrier and factory.
- :mod:`realization_error_types`
    Fixed-linear scalar Galerkin realization-error evidence.
- :mod:`realization_types`
    Voxel-to-Galerkin potential realization and error carriers.
- :mod:`recon_types`
    Reconstruction problem and result carriers.
- :mod:`source_types`
    Represented-source carriers for scalar Galerkin solves.
- :mod:`terminal_types`
    Bounded coordinate-terminal current diagnostic carriers.

Routine Listings
----------------
:class:`AberrationParams`
    Store probe aberration parameters.
:class:`AtomicSliceData`
    Store on-the-fly atomic slice inputs for sharded multislice.
:class:`AxisUpdate`
    Store additive distribution-axis deltas for one kernel evaluation.
:class:`CalibratedArray`
    Store calibrated array data with spatial calibration.
:class:`CGState`
    Store conjugate gradient iteration state.
:class:`CrystalData`
    Store parsed crystal data and static metadata.
:class:`CrystalStructure`
    Store fractional and Cartesian crystal coordinates.
:class:`DetectorConfig`
    Store detector, scan, and calibration configuration.
:class:`Distribution`
    Store a generic weighted distribution over latent samples.
:class:`EnsembleAxes`
    Store optional ensemble distributions for forward simulators.
:class:`ExitWaveParams`
    Store complex exit-wave parameters.
:class:`FisherState`
    Store state for iterative Fisher computation.
:class:`GalerkinAcquisitionManifest`
    Store one submitted finite acquisition-support manifest.
:class:`GalerkinAcquisitionSupportFailure`
    Enumerate fail-closed RM-S1 predicate bits.
:class:`GalerkinAcquisitionSupportResult`
    Store one checked RM-S1 acquisition-support artifact.
:class:`GalerkinAcquisitionSupportStatus`
    Enumerate structural, support-ineligible, and support-eligible outcomes.
:class:`GalerkinActionDirection`
    Store the finite operator direction being enclosed.
:class:`GalerkinActionErrorRoute`
    Store the admitted per-call action enclosure route.
:class:`GalerkinAxialCapCoefficientCertificate`
    Store replayable LVT.24 rectangles and the LVT.31 transfer.
:class:`GalerkinAxialCapCoefficientFailure`
    Enumerate typed LVT.24/LVT.26/LVT.31 noncertificate outcomes.
:class:`GalerkinAxialCapExactFloorFailure`
    Enumerate typed support-only exact LVT.29a proof outcomes.
:class:`GalerkinAxialCapFloorProof`
    Store independently eligible exact-target and realized CAP floors.
:class:`GalerkinAxialCapRealizedFloorFailure`
    Enumerate typed coefficient-dependent realized-floor outcomes.
:class:`GalerkinAxialCapRealizedFloorRoute`
    Select one mutually exclusive physical realized-floor route.
:class:`GalerkinAxialCellAbsorber`
    Store one axis-only LVT.23 profile and frozen LVT.24 approximant.
:class:`GalerkinBackwardDisposition`
    Store the declared treatment of the backward sector.
:class:`GalerkinCarrierOverlapDisposition`
    Store the explicit single-carrier overlap disposition.
:class:`GalerkinCarrierOwnership`
    Store the admitted carrier-block ownership policy.
:class:`GalerkinCarrierTargetRoute`
    Store the exact target-side carrier normalization route.
:class:`GalerkinCertificateReason`
    Store the reason that a Galerkin result lacks certification.
:class:`GalerkinCoordinateCauchyCurrent`
    Store one bounded selected-sector coordinate-current diagnostic.
:class:`GalerkinCurrentOperatorCertificate`
    Store uniform selected-sector LVT.55a operator evidence.
:class:`GalerkinCurrentOperatorFailure`
    Enumerate fail-closed uniform current-operator predicate bits.
:class:`GalerkinDetectorFailure`
    Store the unavailable detector-contract reasons.
:class:`GalerkinDirectionDisposition`
    Store whether a requested direction is exact or projected.
:class:`GalerkinEndpointConvention`
    Store the signed integer endpoint realization.
:class:`GalerkinFixedLinearAbsorberRoute`
    Store the admitted exact absorber and CAP realization route.
:class:`GalerkinFixedLinearErrorLedger`
    Store componentwise RM-S2 S2.27--S2.43 evidence.
:class:`GalerkinLocalAdditionalSource`
    Store one rounded ZERO or complex LOCAL_CELL LVT.20a--LVT.20c map.
:class:`GalerkinLocalAdditionalSourceCertificate`
    Store direct rectangles and the LVT.20e source-norm transfer.
:class:`GalerkinLocalAdditionalSourceCertificateFailure`
    Store one typed direct LVT.20c certificate outcome.
:class:`GalerkinLocalAdditionalSourceRoute`
    Select one exact LVT.20a additional-source carrier.
:class:`GalerkinLocalCellCertificateFailure`
    Store the outcome of one direct local-cell certificate attempt.
:class:`GalerkinLocalCellCoefficientCertificate`
    Store independently enclosed exact local-cell coefficients.
:class:`GalerkinLocalCellCompressionFailure`
    Enumerate typed LVT exact-compression noncertificate outcomes.
:class:`GalerkinLocalCellErrorRoute`
    Store the outward local-cell coefficient-error route.
:class:`GalerkinLocalCellExactCompression`
    Store authenticated LVT.14--LVT.18 exact-compression evidence.
:class:`GalerkinLocalCellFixedLinearErrorLedger`
    Store the disjoint fixed-linear LVT-1 error composition.
:class:`GalerkinLocalCellInteractionCore`
    Store one non-solver-ready fixed LVT interaction action core.
:class:`GalerkinLocalCellPotentialRealization`
    Store one LVT-1 local-cell coefficient realization.
:class:`GalerkinLocalCellTailEnclosure`
    Store one authenticated LVT.9 full Fourier-tail enclosure.
:class:`GalerkinLocalCellTailFailure`
    Store the outcome of one LVT.9 enclosure attempt.
:class:`GalerkinLocalCellTargetManifest`
    Store one completed solver-ready ``LOCAL_CELL_LVT1`` target.
:class:`GalerkinLocalCensoredPoissonDetector`
    Store one fixed nonnegative pre-gain censored-count detector map.
:class:`GalerkinLocalCensoredPoissonDetectorInputManifest`
    Bind primitive detector inputs and nested pixel manifests.
:class:`GalerkinLocalCensoredPoissonLikelihood`
    Store one pre-gain censored-Poisson likelihood enclosure.
:class:`GalerkinLocalComplexRectangles`
    Store componentwise outward complex rectangles on ordered ``I_u``.
:class:`GalerkinLocalCoordinateCauchyCurrent`
    Store one exact-target submitted-state scoped current enclosure.
:class:`GalerkinLocalCurrentOperatorCertificate`
    Store uniform local coordinate ``T/N/F`` evidence.
:class:`GalerkinLocalCurrentOperatorFailure`
    Enumerate typed uniform current-operator outcomes.
:class:`GalerkinLocalDetectorCoordinateConvention`
    Select one RM-S4 coordinate and amplitude normalization pair.
:class:`GalerkinLocalDetectorFailure`
    Enumerate simultaneous local detector noncertificate outcomes.
:class:`GalerkinLocalDetectorHelperCall`
    Name one channel-specific censored-Poisson helper invocation.
:class:`GalerkinLocalDetectorHelperFailureEvidence`
    Store one replayable channel-specific helper failure.
:class:`GalerkinLocalDetectorLikelihoodStage`
    Fix the stochastic law before deterministic gain and offset.
:class:`GalerkinLocalDetectorProductionStage`
    Name every stopped-evidence production-point stage.
:class:`GalerkinLocalDetectorRationalInterval`
    Store one exact rational detector interval.
:class:`GalerkinLocalDetectorRealProductionTrace`
    Bind one rounded real production point to its exact raw enclosure.
:class:`GalerkinLocalDetectorWorkTranscript`
    Store bounded exact detector-composition work evidence.
:class:`GalerkinLocalPassivePixelForms`
    Store one mode's positive and passive diagonal ideal-pixel forms.
:class:`GalerkinLocalPassivePixelInputManifest`
    Bind independent upstream replay policy and primitive pixel inputs.
:class:`GalerkinLocalPositivePortBranchDisposition`
    Record each classified fiber's retained and excluded branch treatment.
:class:`GalerkinLocalPositivePortCertificate`
    Store one projected outward positive-port certificate.
:class:`GalerkinLocalPositivePortRoute`
    Select one explicit positive-port branch-disposition route.
:class:`GalerkinLocalProjectionDefectCertificate`
    Store replayable LVT.34--LVT.40 and LVT.55c--LVT.55e evidence.
:class:`GalerkinLocalProjectionDefectFailure`
    Enumerate simultaneous structural, policy, and arithmetic outcomes.
:class:`GalerkinLocalRepresentedSource`
    Bind incident modes, algebraic actions, and both authenticated parents.
:class:`GalerkinLocalRepresentedSourceActions`
    Store the seven frozen ``D/B/R/S/M/T/C`` algebraic vectors.
:class:`GalerkinLocalRepresentedSourceCertificate`
    Store direct exact-target action rectangles and disjoint error bounds.
:class:`GalerkinLocalRepresentedSourceFailure`
    Store one typed represented-source or direct-certificate outcome.
:class:`GalerkinLocalRepresentedSourceKind`
    Select one plane or coherent-focused represented incident construction.
:class:`GalerkinLocalRepresentedSourceModes`
    Store phase, exact-shell, branch, and reduced-flux evidence.
:class:`GalerkinLocalSourceAxis`
    Select the positive coordinate-aligned source normal.
:class:`GalerkinLocalSourcePhaseConvention`
    Select the sole admitted physical-wavevector phase convention.
:class:`GalerkinLocalStabilityDisposition`
    Distinguish operational success, finite-radius fallback, and rejection.
:class:`GalerkinLocalStabilityFailure`
    Store one typed bounded-checker outcome.
:class:`GalerkinLocalStabilityProof`
    Store exact rational transcripts and outward binary64 reports.
:class:`GalerkinLocalStabilityResult`
    Nest the complete source certificate, solve result, and checked proof.
:class:`GalerkinLocalStabilityRoute`
    Select the exact axial-CAP-floor route.
:class:`GalerkinLocalTerminalActionFailure`
    Enumerate typed per-call frozen-action outcomes.
:class:`GalerkinLocalTerminalComplexRectangles`
    Store componentwise complex rectangles.
:class:`GalerkinLocalTerminalCurrentActionEnclosure`
    Store one exact-real frozen-matrix action enclosure.
:class:`GalerkinLocalTerminalCurrentFailure`
    Enumerate typed exact-target current-diagnostic outcomes.
:class:`GalerkinLocalTerminalScope`
    Select the complete transverse-fiber scope.
:class:`GalerkinLocalVacuumBranchEvidence`
    Store exact submitted-state Cauchy and branch mismatch evidence.
:class:`GalerkinLocalVacuumCutBalance`
    Store the independent reduced-current and defect-work cross-check.
:class:`GalerkinLocalVacuumHalfSpaceDisposition`
    Classify exact-zero, nonzero, or unresolved excluded branch content.
:class:`GalerkinLocalVacuumPropagationError`
    Report a typed branch or formal-witness failure.
:class:`GalerkinLocalVacuumPropagationFailure`
    Enumerate failures outside exact entire-helper resources.
:class:`GalerkinLocalVacuumPropagator`
    Store one replayable homogeneous 2x2 Cauchy propagator enclosure.
:class:`GalerkinLocalVacuumRationalInterval`
    Store one exact rational real interval.
:class:`GalerkinLocalVacuumReference`
    Select the sole canonical local vacuum-reference declaration.
:class:`GalerkinLocalVacuumRootCertificate`
    Store one strict replayable LVT.39 root classification.
:class:`GalerkinLocalVacuumRootClass`
    Distinguish propagating, evanescent, grazing, and unresolved roots.
:class:`GalerkinLocalVacuumTerminalCertificate`
    Store one composed LVT.39--LVT.56 local vacuum terminal certificate.
:class:`GalerkinLocalVacuumTerminalDisposition`
    Select one honest plane-defined or exact-native continuation claim.
:class:`GalerkinLocalVacuumTerminalEntireEvidence`
    Store exact phase and forced-integral helper resource evidence.
:class:`GalerkinLocalVacuumTerminalFailure`
    Enumerate simultaneous local vacuum-terminal noncertificate outcomes.
:class:`GalerkinLocalVacuumWorkTranscript`
    Store deterministic exact interval-operation work evidence.
:class:`GalerkinLocalVacuumZeroWitness`
    Store equality of two canonical formal algebraic normal forms.
:class:`GalerkinLocalVacuumZeroWitnessRoute`
    Distinguish exact-rational and symbolic normal-form equality.
:class:`GalerkinLocalZeroSlabCertificate`
    Store replayable LVT.21--LVT.22 geometry and predicate evidence.
:class:`GalerkinLocalZeroSlabFailure`
    Enumerate simultaneous fail-closed zero-slab predicate reasons.
:class:`GalerkinOperator`
    Store one fixed complex-linear scalar Galerkin operator.
:class:`GalerkinPhysicalResidual`
    Store one independently recomputed physical residual.
:class:`GalerkinPotentialCertificateFailure`
    Store the outcome of one host coefficient-certificate attempt.
:class:`GalerkinPotentialCoefficientCertificate`
    Store exact-coefficient rectangles produced by one host checker.
:class:`GalerkinPotentialErrorRoute`
    Store the outward coefficient-error route.
:class:`GalerkinPotentialRealization`
    Store one VC-1 voxel-to-coefficient realization.
:class:`GalerkinPotentialRealizationMethod`
    Store the exact finite-target realization method.
:class:`GalerkinPreparedLocalCurrentOperator`
    Mark a host-replayed operator for frozen transform actions.
:class:`GalerkinProductSupport`
    Store independent supports for fixed scalar Galerkin products.
:class:`GalerkinRepresentedSource`
    Bind one represented stored-shell source to a scalar target.
:class:`GalerkinRepresentedSourceKind`
    Store the represented stored-shell source-construction kind.
:class:`GalerkinResidualErrorEnclosure`
    Store one independent same-``H_alg`` residual enclosure.
:class:`GalerkinSolveMethod`
    Store the selected Galerkin iterative-solve method.
:class:`GalerkinSolveResult`
    Store one algebraic scalar Galerkin solve result.
:class:`GalerkinSolveStatus`
    Store the termination status of a Galerkin solve.
:class:`GalerkinSource`
    Store one finite matched-source realization.
:class:`GalerkinSourceActions`
    Store mandatory free, CAP, interaction, and source vectors.
:class:`GalerkinSourceAxis`
    Store the positive coordinate-aligned propagation normal.
:class:`GalerkinSourceBranch`
    Store the admitted finite source-construction branch.
:class:`GalerkinSourceErrorEnclosure`
    Store numerical error enclosures for the finite source actions.
:class:`GalerkinSourceErrorRoute`
    Store the source/action numerical-error route.
:class:`GalerkinSourceModes`
    Store represented modes, phase geometry, masks, and reduced fluxes.
:class:`GalerkinSourcePhaseConvention`
    Store the explicit coefficient-phase convention.
:class:`GalerkinSourceRepresentationLedger`
    Store all six RM-S3 staged representation-error terms.
:class:`GalerkinSourceRepresentationRoute`
    Store the narrow represented-target ledger route.
:class:`GalerkinStabilityDisposition`
    Store one per-result stability invocation disposition.
:class:`GalerkinStabilityFailure`
    Store one fail-closed stability invocation reason.
:class:`GalerkinStabilityProof`
    Store one checker-produced exact Route-A proof payload.
:class:`GalerkinStabilityResult`
    Store one per-result operational stability invocation.
:class:`GalerkinStabilityRoute`
    Store the checked singular-value certificate route.
:class:`GalerkinStoredShellRoute`
    Store the represented-target shell-evidence route.
:class:`GalerkinTargetActionEnclosure`
    Store one per-state RM-S2 production-action enclosure.
:class:`GalerkinTargetManifest`
    Store one canonical SC-1 finite target manifest.
:class:`GalerkinTerminalCurrentActionEnclosure`
    Store one per-call frozen-current action enclosure.
:class:`GalerkinTerminalCurrentActionFailure`
    Enumerate fail-closed per-call current-action predicate bits.
:class:`GalerkinTerminalCurrentFailure`
    Enumerate fail-closed coordinate-current predicate bits.
:class:`GalerkinTerminalCurrentRoute`
    Store the coordinate-current enclosure route.
:class:`GalerkinTerminalCurrentScope`
    Store the factory-owned selected-fiber current scope.
:class:`GalerkinTerminalSide`
    Store the oriented coordinate face used by the terminal.
:class:`GalerkinVacuumBranchFailure`
    Store the unavailable physical vacuum-branch reason.
:class:`GalerkinVoxelTargetRoute`
    Store the disjoint voxel finite-target interpretation.
:class:`GeometryParams`
    Store geometric calibration parameters.
:class:`GNState`
    Store Gauss-Newton iteration state.
:class:`KirklandParameters`
    Store Kirkland coefficients for one element.
:class:`LanczosState`
    Store Lanczos tridiagonalisation state.
:class:`LaplaceUncertainty`
    Store a Laplace-approximation uncertainty summary.
:class:`LMState`
    Store Levenberg-Marquardt iteration state.
:class:`LobatoParameters`
    Store Lobato--Van Dyck coefficients for one element.
:class:`LocalCellPotential3D`
    Store real voltages constant on periodic rectangular cells.
:class:`LossType`
    Store static loss-function selection.
:class:`MicroscopeConfig`
    Store microscope voltage and probe-aberration configuration.
:class:`OptimizableBlock`
    Store static optimizable ptychography block names.
:class:`PositionParams`
    Store scan position error parameters.
:class:`PosteriorSamples`
    Store posterior samples and diagnostics.
:class:`Potential3D`
    Store a band-limited scalar electrostatic potential in volts.
:class:`PotentialSlices`
    Store potential slices for multislice simulations.
:class:`ProbeModeParams`
    Store probe mode parameters for partial coherence.
:class:`ProbeModes`
    Store multimodal electron probe data.
:class:`PtychoParams`
    Store all ptychographic parameter blocks.
:class:`ReconProblem`
    Store a reconstruction inverse-problem contract.
:class:`ReconResult`
    Store a reconstruction solver result.
:class:`ReductionMode`
    Store static ensemble reduction mode.
:class:`STEM4D`
    Store 4D-STEM diffraction data.
:func:`combine_axis_updates`
    Sum a tuple of additive axis-update carriers.
:func:`create_atomic_slice_data`
    Create AtomicSliceData with structural and runtime validation.
:func:`create_axis_update`
    Create an AxisUpdate with structural and runtime validation.
:func:`create_calibrated_array`
    Create a CalibratedArray with runtime validation.
:func:`create_cg_state`
    Create a validated conjugate gradient iteration state.
:func:`create_crystal_data`
    Create a CrystalData with runtime validation.
:func:`create_crystal_structure`
    Create a CrystalStructure with runtime validation.
:func:`create_detector_config`
    Create a DetectorConfig with structural and runtime validation.
:func:`create_distribution`
    Create a Distribution with validated probability weights.
:func:`create_ensemble_axes`
    Create EnsembleAxes with structural validation.
:func:`create_fisher_state`
    Create a validated Fisher computation state.
:func:`create_galerkin_acquisition_manifest`
    Create one structurally shaped acquisition submission.
:func:`create_galerkin_coordinate_cauchy_current`
    Create a validated bounded selected-sector current diagnostic.
:func:`create_galerkin_current_operator_certificate`
    Create validated uniform selected-sector current-operator evidence.
:func:`create_galerkin_fixed_linear_error_ledger`
    Create a structurally validated fixed-linear error ledger.
:func:`create_galerkin_operator`
    Create a validated scalar Galerkin operator carrier.
:func:`create_galerkin_physical_residual`
    Create a validated physical-residual carrier.
:func:`create_galerkin_potential_coefficient_certificate`
    Create a structurally validated host coefficient certificate.
:func:`create_galerkin_potential_realization`
    Create a validated voxel-to-Galerkin realization carrier.
:func:`create_galerkin_product_support`
    Create validated supports for fixed scalar Galerkin products.
:func:`create_galerkin_residual_error_enclosure`
    Create a structurally validated independent residual enclosure.
:func:`create_galerkin_solve_result`
    Create a validated algebraic Galerkin solve result.
:func:`create_galerkin_source`
    Create a validated finite matched-source carrier.
:func:`create_galerkin_source_actions`
    Create validated finite source-action vectors.
:func:`create_galerkin_source_error_enclosure`
    Create honest source/action numerical-error enclosures.
:func:`create_galerkin_source_ledger`
    Create the complete six-stage RM-S3 representation ledger.
:func:`create_galerkin_source_modes`
    Create validated exact represented-mode and flux data.
:func:`create_galerkin_stability_proof`
    Create a structurally validated exact stability proof payload.
:func:`create_galerkin_stability_result`
    Create a validated per-result stability invocation.
:func:`create_galerkin_target_action_enclosure`
    Create a structurally validated per-state action enclosure.
:func:`create_galerkin_terminal_current_action_enclosure`
    Create one validated per-call frozen-current action enclosure.
:func:`create_gn_state`
    Create a validated Gauss-Newton iteration state.
:func:`create_kirkland_parameters`
    Create validated Kirkland coefficients for one element.
:func:`create_lanczos_state`
    Create a validated Lanczos tridiagonalisation state.
:func:`create_laplace_uncertainty`
    Create a LaplaceUncertainty with runtime validation.
:func:`create_lm_state`
    Create a validated Levenberg-Marquardt iteration state.
:func:`create_lobato_parameters`
    Create validated Lobato--Van Dyck coefficients for one element.
:func:`create_local_cell_potential_3d`
    Create a validated periodic local-cell voltage field.
:func:`create_microscope_config`
    Create a MicroscopeConfig with structural and runtime validation.
:func:`create_posterior_samples`
    Create PosteriorSamples with runtime validation.
:func:`create_potential_3d`
    Create a validated three-dimensional electrostatic potential.
:func:`create_potential_slices`
    Create a PotentialSlices with runtime validation.
:func:`create_probe_modes`
    Create a ProbeModes with runtime validation.
:func:`create_ptycho_params`
    Construct combined PtychoParams from components.
:func:`create_recon_problem`
    Create a ReconProblem with runtime validation.
:func:`create_recon_result`
    Create a ReconResult with runtime validation.
:func:`create_represented_galerkin_source`
    Bind source evidence to one manifested scalar target.
:func:`create_stem4d`
    Create a STEM4D with runtime validation.
:func:`create_trivial_distribution`
    Create the one-sample identity distribution.
:func:`helmholtz_coupling`
    Compute the Helmholtz potential coupling in 1/(V·Angstrom²).
:func:`phase_interaction_parameter`
    Compute the phase interaction parameter in rad/(V·Angstrom).
:func:`relativistic_mass`
    Compute the relativistic electron mass in kg.
:func:`relativistic_wavelength_ang`
    Compute the relativistic electron wavelength in Angstroms.
:obj:`A_BOHR`
    Bohr radius in Angstroms.
:obj:`C_LIGHT`
    Speed of light in m/s.
:obj:`E_CHARGE`
    Elementary charge in C.
:obj:`float_jax_image`
    Type alias for 2D JAX float array (H, W).
:obj:`float_np_image`
    Type alias for 2D numpy float array (H, W).
:obj:`H_PLANCK`
    Planck constant in J·s.
:obj:`HBAR`
    Reduced Planck constant in J·s.
:obj:`int_jax_image`
    Type alias for 2D JAX integer array (H, W).
:obj:`int_np_image`
    Type alias for 2D numpy integer array (H, W).
:obj:`M0C2_EV`
    Electron rest energy in eV.
:obj:`M_E`
    Electron rest mass in kg.
:obj:`MOTT_BETHE_VOLT_ANGSTROM_SQ`
    Mott-Bethe constant h²/(2π m₀ e) in V·Å².
:obj:`non_jax_number`
    Union type for non-JAX numeric values (int or float).
:obj:`scalar_bool`
    Union type for scalar boolean values (bool or JAX scalar array).
:obj:`scalar_float`
    Union type for scalar float values (float or JAX scalar array).
:obj:`scalar_int`
    Union type for scalar integer values (int or JAX scalar array).
:obj:`scalar_num`
    Union type for scalar numeric values (int, float, or JAX scalar array).
:obj:`TRIVIAL`
    Short alias for ``TRIVIAL_DISTRIBUTION``.
:obj:`TRIVIAL_DISTRIBUTION`
    Identity one-sample distribution.

"""

from .absorber_types import (
    GalerkinAxialCapCoefficientCertificate,
    GalerkinAxialCapCoefficientFailure,
    GalerkinAxialCapExactFloorFailure,
    GalerkinAxialCapFloorProof,
    GalerkinAxialCapRealizedFloorFailure,
    GalerkinAxialCapRealizedFloorRoute,
    GalerkinAxialCellAbsorber,
)
from .absorber_types import (
    _make_axial_cap_coefficient_certificate as _make_axial_cap_coefficient_certificate,  # noqa: E501
)
from .absorber_types import (
    _make_axial_cap_floor_proof as _make_axial_cap_floor_proof,
)
from .absorber_types import (
    _make_axial_cell_absorber as _make_axial_cell_absorber,
)
from .acquisition_types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportFailure,
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
    GalerkinBackwardDisposition,
    GalerkinCarrierOverlapDisposition,
    GalerkinCarrierOwnership,
    GalerkinCarrierTargetRoute,
    GalerkinDirectionDisposition,
    GalerkinEndpointConvention,
    GalerkinTerminalSide,
    create_galerkin_acquisition_manifest,
)
from .action_error_types import (
    GalerkinActionDirection,
    GalerkinActionErrorRoute,
    GalerkinResidualErrorEnclosure,
    GalerkinTargetActionEnclosure,
    create_galerkin_residual_error_enclosure,
    create_galerkin_target_action_enclosure,
)
from .born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from .born_types import (
    GalerkinCertificateReason,
    GalerkinOperator,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    create_galerkin_operator,
    create_galerkin_solve_result,
)
from .constants import (
    A_BOHR,
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M0C2_EV,
    M_E,
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    helmholtz_coupling,
    phase_interaction_parameter,
    relativistic_mass,
    relativistic_wavelength_ang,
)
from .crystal_types import (
    CrystalData,
    CrystalStructure,
    create_crystal_data,
    create_crystal_structure,
)
from .custom_types import (
    LossType,
    float_jax_image,
    float_np_image,
    int_jax_image,
    int_np_image,
    non_jax_number,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_num,
)
from .distributions import (
    TRIVIAL,
    TRIVIAL_DISTRIBUTION,
    Distribution,
    ReductionMode,
    create_distribution,
    create_trivial_distribution,
)
from .electron_types import (
    STEM4D,
    AtomicSliceData,
    AxisUpdate,
    CalibratedArray,
    DetectorConfig,
    EnsembleAxes,
    MicroscopeConfig,
    PotentialSlices,
    ProbeModes,
    combine_axis_updates,
    create_atomic_slice_data,
    create_axis_update,
    create_calibrated_array,
    create_detector_config,
    create_ensemble_axes,
    create_microscope_config,
    create_potential_slices,
    create_probe_modes,
    create_stem4d,
)
from .form_factor_types import (
    KirklandParameters,
    LobatoParameters,
    create_kirkland_parameters,
    create_lobato_parameters,
)
from .galerkin_types import (
    GalerkinPhysicalResidual,
    GalerkinSource,
    GalerkinSourceBranch,
    GalerkinStabilityDisposition,
    GalerkinStabilityFailure,
    GalerkinStabilityProof,
    GalerkinStabilityResult,
    GalerkinStabilityRoute,
    GalerkinTargetManifest,
    create_galerkin_physical_residual,
    create_galerkin_source,
    create_galerkin_stability_proof,
    create_galerkin_stability_result,
)
from .galerkin_types import (
    _create_galerkin_target_manifest as _create_galerkin_target_manifest,
)
from .galerkin_types import (
    _derive_algebraic_wavenumber as _derive_algebraic_wavenumber,
)
from .galerkin_types import (
    _exact_target_direction_error_bounds as _exact_target_direction_error_bounds,  # noqa: E501
)
from .galerkin_types import (
    _exact_target_full_offset_max as _exact_target_full_offset_max,
)
from .jacobian_types import (
    AberrationParams,
    CGState,
    ExitWaveParams,
    FisherState,
    GeometryParams,
    GNState,
    LanczosState,
    LMState,
    OptimizableBlock,
    PositionParams,
    ProbeModeParams,
    PtychoParams,
    create_cg_state,
    create_fisher_state,
    create_gn_state,
    create_lanczos_state,
    create_lm_state,
    create_ptycho_params,
)
from .local_cell_interaction_types import (
    GalerkinLocalCellCompressionFailure,
    GalerkinLocalCellExactCompression,
    GalerkinLocalCellInteractionCore,
)
from .local_cell_interaction_types import (
    _make_local_cell_exact_compression as _make_local_cell_exact_compression,
)
from .local_cell_interaction_types import (
    _make_local_cell_interaction_core as _make_local_cell_interaction_core,
)
from .local_cell_target_types import (
    GalerkinLocalCellFixedLinearErrorLedger,
    GalerkinLocalCellTargetManifest,
)
from .local_cell_target_types import (
    _make_local_cell_fixed_linear_error_ledger as _make_local_cell_fixed_linear_error_ledger,  # noqa: E501
)
from .local_cell_target_types import (
    _make_local_cell_target_manifest as _make_local_cell_target_manifest,
)
from .local_cell_types import (
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellCoefficientCertificate,
    GalerkinLocalCellErrorRoute,
    GalerkinLocalCellPotentialRealization,
    GalerkinLocalCellTailEnclosure,
    GalerkinLocalCellTailFailure,
    GalerkinVoxelTargetRoute,
    LocalCellPotential3D,
    create_local_cell_potential_3d,
)
from .local_cell_types import (
    _create_direct_local_cell_realization as _create_direct_local,
)
from .local_cell_types import (
    _create_local_cell_realization as _create_local_cell_realization,
)
from .local_cell_types import (
    _make_local_cell_certificate as _make_local_cell_certificate,
)
from .local_cell_types import (
    _make_local_cell_tail_enclosure as _make_local_cell_tail_enclosure,
)
from .local_detector_types import (
    GalerkinLocalCensoredPoissonDetector,
    GalerkinLocalCensoredPoissonDetectorInputManifest,
    GalerkinLocalCensoredPoissonLikelihood,
    GalerkinLocalDetectorCoordinateConvention,
    GalerkinLocalDetectorFailure,
    GalerkinLocalDetectorHelperCall,
    GalerkinLocalDetectorHelperFailureEvidence,
    GalerkinLocalDetectorLikelihoodStage,
    GalerkinLocalDetectorProductionStage,
    GalerkinLocalDetectorRationalInterval,
    GalerkinLocalDetectorRealProductionTrace,
    GalerkinLocalDetectorWorkTranscript,
    GalerkinLocalPassivePixelForms,
    GalerkinLocalPassivePixelInputManifest,
    GalerkinLocalPositivePortBranchDisposition,
    GalerkinLocalPositivePortCertificate,
    GalerkinLocalPositivePortRoute,
)
from .local_projection_types import (
    GalerkinLocalProjectionDefectCertificate,
    GalerkinLocalProjectionDefectFailure,
)
from .local_represented_source_types import (
    GalerkinLocalComplexRectangles,
    GalerkinLocalRepresentedSource,
    GalerkinLocalRepresentedSourceActions,
    GalerkinLocalRepresentedSourceCertificate,
    GalerkinLocalRepresentedSourceFailure,
    GalerkinLocalRepresentedSourceKind,
    GalerkinLocalRepresentedSourceModes,
    GalerkinLocalSourceAxis,
    GalerkinLocalSourcePhaseConvention,
)
from .local_source_types import (
    GalerkinLocalAdditionalSource,
    GalerkinLocalAdditionalSourceCertificate,
    GalerkinLocalAdditionalSourceCertificateFailure,
    GalerkinLocalAdditionalSourceRoute,
)
from .local_source_types import (
    _make_local_additional_source as _make_local_additional_source,
)
from .local_source_types import (
    _make_local_additional_source_certificate as _make_local_additional_source_certificate,  # noqa: E501
)
from .local_stability_types import (
    GalerkinLocalStabilityDisposition,
    GalerkinLocalStabilityFailure,
    GalerkinLocalStabilityProof,
    GalerkinLocalStabilityResult,
    GalerkinLocalStabilityRoute,
)
from .local_terminal_types import (
    GalerkinLocalCoordinateCauchyCurrent,
    GalerkinLocalCurrentOperatorCertificate,
    GalerkinLocalCurrentOperatorFailure,
    GalerkinLocalTerminalActionFailure,
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalCurrentActionEnclosure,
    GalerkinLocalTerminalCurrentFailure,
    GalerkinLocalTerminalScope,
    GalerkinPreparedLocalCurrentOperator,
)
from .local_terminal_types import (
    _make_prepared_local_current_operator as _make_prepared_local_current_operator,  # noqa: E501
)
from .local_vacuum_propagation_types import (
    GalerkinLocalVacuumPropagationError,
    GalerkinLocalVacuumPropagationFailure,
    GalerkinLocalVacuumPropagator,
    GalerkinLocalVacuumRationalInterval,
    GalerkinLocalVacuumRootCertificate,
    GalerkinLocalVacuumRootClass,
    GalerkinLocalVacuumWorkTranscript,
    GalerkinLocalVacuumZeroWitness,
    GalerkinLocalVacuumZeroWitnessRoute,
)
from .local_vacuum_terminal_types import (
    GalerkinLocalVacuumBranchEvidence,
    GalerkinLocalVacuumCutBalance,
    GalerkinLocalVacuumHalfSpaceDisposition,
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
    GalerkinLocalVacuumTerminalEntireEvidence,
    GalerkinLocalVacuumTerminalFailure,
)
from .local_vacuum_terminal_types import (
    _make_local_vacuum_branch_evidence as _make_local_vacuum_branch_evidence,
)
from .local_vacuum_terminal_types import (
    _make_local_vacuum_cut_balance as _make_local_vacuum_cut_balance,
)
from .local_vacuum_terminal_types import (
    _make_local_vacuum_terminal_certificate as _make_local_vacuum_terminal_certificate,  # noqa: E501
)
from .local_vacuum_terminal_types import (
    _make_local_vacuum_terminal_entire_evidence as _make_local_vacuum_terminal_entire_evidence,  # noqa: E501
)
from .local_vacuum_terminal_types import (
    _PlaneMismatchBounds as _PlaneMismatchBounds,
)
from .local_vacuum_terminal_types import (
    _ProductionEvidence as _ProductionEvidence,
)
from .local_zero_slab_types import (
    GalerkinLocalVacuumReference,
    GalerkinLocalZeroSlabCertificate,
    GalerkinLocalZeroSlabFailure,
)
from .potential_types import Potential3D, create_potential_3d
from .realization_error_types import (
    GalerkinFixedLinearAbsorberRoute,
    GalerkinFixedLinearErrorLedger,
    create_galerkin_fixed_linear_error_ledger,
)
from .realization_types import (
    GalerkinPotentialCertificateFailure,
    GalerkinPotentialCoefficientCertificate,
    GalerkinPotentialErrorRoute,
    GalerkinPotentialRealization,
    GalerkinPotentialRealizationMethod,
    create_galerkin_potential_coefficient_certificate,
    create_galerkin_potential_realization,
)
from .recon_types import (
    LaplaceUncertainty,
    PosteriorSamples,
    ReconProblem,
    ReconResult,
    create_laplace_uncertainty,
    create_posterior_samples,
    create_recon_problem,
    create_recon_result,
)
from .source_types import (
    GalerkinRepresentedSource,
    GalerkinRepresentedSourceKind,
    GalerkinSourceActions,
    GalerkinSourceAxis,
    GalerkinSourceErrorEnclosure,
    GalerkinSourceErrorRoute,
    GalerkinSourceModes,
    GalerkinSourcePhaseConvention,
    GalerkinSourceRepresentationLedger,
    GalerkinSourceRepresentationRoute,
    GalerkinStoredShellRoute,
    create_galerkin_source_actions,
    create_galerkin_source_error_enclosure,
    create_galerkin_source_ledger,
    create_galerkin_source_modes,
    create_represented_galerkin_source,
)
from .terminal_types import (
    GalerkinCoordinateCauchyCurrent,
    GalerkinCurrentOperatorCertificate,
    GalerkinCurrentOperatorFailure,
    GalerkinDetectorFailure,
    GalerkinTerminalCurrentActionEnclosure,
    GalerkinTerminalCurrentActionFailure,
    GalerkinTerminalCurrentFailure,
    GalerkinTerminalCurrentRoute,
    GalerkinTerminalCurrentScope,
    GalerkinVacuumBranchFailure,
    create_galerkin_coordinate_cauchy_current,
    create_galerkin_current_operator_certificate,
    create_galerkin_terminal_current_action_enclosure,
)

_create_direct_local_cell_realization = _create_direct_local
del _create_direct_local

__all__: list[str] = [
    "A_BOHR",
    "AberrationParams",
    "AxisUpdate",
    "AtomicSliceData",
    "CGState",
    "C_LIGHT",
    "CalibratedArray",
    "CrystalData",
    "CrystalStructure",
    "Distribution",
    "DetectorConfig",
    "E_CHARGE",
    "EnsembleAxes",
    "ExitWaveParams",
    "FisherState",
    "GalerkinAcquisitionManifest",
    "GalerkinAcquisitionSupportFailure",
    "GalerkinAcquisitionSupportResult",
    "GalerkinAcquisitionSupportStatus",
    "GalerkinActionDirection",
    "GalerkinActionErrorRoute",
    "GalerkinAxialCapCoefficientCertificate",
    "GalerkinAxialCapCoefficientFailure",
    "GalerkinAxialCapExactFloorFailure",
    "GalerkinAxialCapFloorProof",
    "GalerkinAxialCapRealizedFloorFailure",
    "GalerkinAxialCapRealizedFloorRoute",
    "GalerkinAxialCellAbsorber",
    "GalerkinBackwardDisposition",
    "GalerkinCarrierOverlapDisposition",
    "GalerkinCarrierOwnership",
    "GalerkinCarrierTargetRoute",
    "GalerkinCertificateReason",
    "GalerkinCoordinateCauchyCurrent",
    "GalerkinCurrentOperatorCertificate",
    "GalerkinCurrentOperatorFailure",
    "GalerkinDetectorFailure",
    "GalerkinDirectionDisposition",
    "GalerkinEndpointConvention",
    "GalerkinFixedLinearAbsorberRoute",
    "GalerkinFixedLinearErrorLedger",
    "GalerkinLocalAdditionalSource",
    "GalerkinLocalAdditionalSourceCertificate",
    "GalerkinLocalAdditionalSourceCertificateFailure",
    "GalerkinLocalAdditionalSourceRoute",
    "GalerkinLocalCellCertificateFailure",
    "GalerkinLocalCellCoefficientCertificate",
    "GalerkinLocalCellCompressionFailure",
    "GalerkinLocalCellErrorRoute",
    "GalerkinLocalCellExactCompression",
    "GalerkinLocalCellFixedLinearErrorLedger",
    "GalerkinLocalCellInteractionCore",
    "GalerkinLocalCellPotentialRealization",
    "GalerkinLocalCellTailEnclosure",
    "GalerkinLocalCellTailFailure",
    "GalerkinLocalCellTargetManifest",
    "GalerkinLocalCensoredPoissonDetector",
    "GalerkinLocalCensoredPoissonDetectorInputManifest",
    "GalerkinLocalCensoredPoissonLikelihood",
    "GalerkinLocalComplexRectangles",
    "GalerkinLocalCoordinateCauchyCurrent",
    "GalerkinLocalCurrentOperatorCertificate",
    "GalerkinLocalCurrentOperatorFailure",
    "GalerkinLocalDetectorCoordinateConvention",
    "GalerkinLocalDetectorFailure",
    "GalerkinLocalDetectorHelperCall",
    "GalerkinLocalDetectorHelperFailureEvidence",
    "GalerkinLocalDetectorLikelihoodStage",
    "GalerkinLocalDetectorProductionStage",
    "GalerkinLocalDetectorRationalInterval",
    "GalerkinLocalDetectorRealProductionTrace",
    "GalerkinLocalDetectorWorkTranscript",
    "GalerkinLocalPassivePixelForms",
    "GalerkinLocalPassivePixelInputManifest",
    "GalerkinLocalPositivePortBranchDisposition",
    "GalerkinLocalPositivePortCertificate",
    "GalerkinLocalPositivePortRoute",
    "GalerkinLocalProjectionDefectCertificate",
    "GalerkinLocalProjectionDefectFailure",
    "GalerkinLocalRepresentedSource",
    "GalerkinLocalRepresentedSourceActions",
    "GalerkinLocalRepresentedSourceCertificate",
    "GalerkinLocalRepresentedSourceFailure",
    "GalerkinLocalRepresentedSourceKind",
    "GalerkinLocalRepresentedSourceModes",
    "GalerkinLocalSourceAxis",
    "GalerkinLocalSourcePhaseConvention",
    "GalerkinLocalStabilityDisposition",
    "GalerkinLocalStabilityFailure",
    "GalerkinLocalStabilityProof",
    "GalerkinLocalStabilityResult",
    "GalerkinLocalStabilityRoute",
    "GalerkinLocalTerminalActionFailure",
    "GalerkinLocalTerminalComplexRectangles",
    "GalerkinLocalTerminalCurrentActionEnclosure",
    "GalerkinLocalTerminalCurrentFailure",
    "GalerkinLocalTerminalScope",
    "GalerkinLocalVacuumBranchEvidence",
    "GalerkinLocalVacuumCutBalance",
    "GalerkinLocalVacuumHalfSpaceDisposition",
    "GalerkinLocalVacuumPropagationError",
    "GalerkinLocalVacuumPropagationFailure",
    "GalerkinLocalVacuumPropagator",
    "GalerkinLocalVacuumRationalInterval",
    "GalerkinLocalVacuumReference",
    "GalerkinLocalVacuumRootCertificate",
    "GalerkinLocalVacuumRootClass",
    "GalerkinLocalVacuumTerminalCertificate",
    "GalerkinLocalVacuumTerminalDisposition",
    "GalerkinLocalVacuumTerminalEntireEvidence",
    "GalerkinLocalVacuumTerminalFailure",
    "GalerkinLocalVacuumWorkTranscript",
    "GalerkinLocalVacuumZeroWitness",
    "GalerkinLocalVacuumZeroWitnessRoute",
    "GalerkinLocalZeroSlabCertificate",
    "GalerkinLocalZeroSlabFailure",
    "GalerkinOperator",
    "GalerkinPhysicalResidual",
    "GalerkinPotentialCertificateFailure",
    "GalerkinPotentialCoefficientCertificate",
    "GalerkinPotentialErrorRoute",
    "GalerkinPotentialRealization",
    "GalerkinPotentialRealizationMethod",
    "GalerkinPreparedLocalCurrentOperator",
    "GalerkinProductSupport",
    "GalerkinRepresentedSource",
    "GalerkinRepresentedSourceKind",
    "GalerkinResidualErrorEnclosure",
    "GalerkinSolveMethod",
    "GalerkinSolveResult",
    "GalerkinSolveStatus",
    "GalerkinSource",
    "GalerkinSourceActions",
    "GalerkinSourceAxis",
    "GalerkinSourceBranch",
    "GalerkinSourceErrorEnclosure",
    "GalerkinSourceErrorRoute",
    "GalerkinSourceModes",
    "GalerkinSourcePhaseConvention",
    "GalerkinSourceRepresentationLedger",
    "GalerkinSourceRepresentationRoute",
    "GalerkinStabilityDisposition",
    "GalerkinStabilityFailure",
    "GalerkinStabilityProof",
    "GalerkinStabilityResult",
    "GalerkinStabilityRoute",
    "GalerkinStoredShellRoute",
    "GalerkinTargetActionEnclosure",
    "GalerkinTargetManifest",
    "GalerkinTerminalCurrentActionEnclosure",
    "GalerkinTerminalCurrentActionFailure",
    "GalerkinTerminalCurrentFailure",
    "GalerkinTerminalCurrentRoute",
    "GalerkinTerminalCurrentScope",
    "GalerkinTerminalSide",
    "GalerkinVacuumBranchFailure",
    "GalerkinVoxelTargetRoute",
    "GNState",
    "GeometryParams",
    "HBAR",
    "H_PLANCK",
    "KirklandParameters",
    "LMState",
    "LanczosState",
    "LaplaceUncertainty",
    "LobatoParameters",
    "LocalCellPotential3D",
    "LossType",
    "M0C2_EV",
    "MOTT_BETHE_VOLT_ANGSTROM_SQ",
    "M_E",
    "MicroscopeConfig",
    "OptimizableBlock",
    "PositionParams",
    "PosteriorSamples",
    "Potential3D",
    "PotentialSlices",
    "ProbeModeParams",
    "ProbeModes",
    "PtychoParams",
    "ReconProblem",
    "ReconResult",
    "ReductionMode",
    "STEM4D",
    "TRIVIAL",
    "TRIVIAL_DISTRIBUTION",
    "combine_axis_updates",
    "create_axis_update",
    "create_atomic_slice_data",
    "create_calibrated_array",
    "create_cg_state",
    "create_crystal_data",
    "create_crystal_structure",
    "create_distribution",
    "create_detector_config",
    "create_ensemble_axes",
    "create_fisher_state",
    "create_galerkin_acquisition_manifest",
    "create_galerkin_coordinate_cauchy_current",
    "create_galerkin_current_operator_certificate",
    "create_galerkin_fixed_linear_error_ledger",
    "create_galerkin_operator",
    "create_galerkin_physical_residual",
    "create_galerkin_potential_coefficient_certificate",
    "create_galerkin_potential_realization",
    "create_galerkin_product_support",
    "create_galerkin_residual_error_enclosure",
    "create_galerkin_solve_result",
    "create_galerkin_source",
    "create_galerkin_source_actions",
    "create_galerkin_source_error_enclosure",
    "create_galerkin_source_ledger",
    "create_galerkin_source_modes",
    "create_galerkin_stability_proof",
    "create_galerkin_stability_result",
    "create_galerkin_target_action_enclosure",
    "create_galerkin_terminal_current_action_enclosure",
    "create_gn_state",
    "create_kirkland_parameters",
    "create_lanczos_state",
    "create_laplace_uncertainty",
    "create_lm_state",
    "create_lobato_parameters",
    "create_local_cell_potential_3d",
    "create_microscope_config",
    "create_posterior_samples",
    "create_potential_3d",
    "create_potential_slices",
    "create_probe_modes",
    "create_ptycho_params",
    "create_recon_problem",
    "create_recon_result",
    "create_represented_galerkin_source",
    "create_stem4d",
    "create_trivial_distribution",
    "helmholtz_coupling",
    "phase_interaction_parameter",
    "relativistic_mass",
    "relativistic_wavelength_ang",
    "float_jax_image",
    "float_np_image",
    "int_jax_image",
    "int_np_image",
    "non_jax_number",
    "scalar_bool",
    "scalar_float",
    "scalar_int",
    "scalar_num",
]
