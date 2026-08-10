r"""Define one checked scalar acquisition-support artifact.

Extended Summary
----------------
This module owns the production support-core carrier for one RM-S1
invocation. The manifest binds the illumination and requested elastic-output
indices, all five SC-1 supports, one fiber-complete coordinate terminal,
endpoint realization, carrier ownership and target-normalization route, and
backward-sector disposition. The result keeps structural validity distinct
from support ineligibility.

Routine Listings
----------------
:class:`GalerkinAcquisitionManifest`
    Store one submitted finite acquisition-support manifest.
:class:`GalerkinAcquisitionSupportFailure`
    Enumerate fail-closed RM-S1 predicate bits.
:class:`GalerkinAcquisitionSupportResult`
    Store one checked RM-S1 acquisition-support artifact.
:class:`GalerkinAcquisitionSupportStatus`
    Enumerate structural, support-ineligible, and support-eligible outcomes.
:class:`GalerkinBackwardDisposition`
    Store the declared treatment of the backward sector.
:class:`GalerkinCarrierOverlapDisposition`
    Store the explicit single-carrier overlap disposition.
:class:`GalerkinCarrierOwnership`
    Store the admitted carrier-block ownership policy.
:class:`GalerkinCarrierTargetRoute`
    Store the exact target-side carrier normalization route.
:class:`GalerkinDirectionDisposition`
    Store whether a requested direction is exact or projected.
:class:`GalerkinEndpointConvention`
    Store the signed integer endpoint realization.
:class:`GalerkinTerminalSide`
    Store the oriented coordinate face used by the terminal.
:func:`create_galerkin_acquisition_manifest`
    Create one structurally shaped acquisition submission.

Notes
-----
The manifest is deliberately a submission rather than an eligibility proof.
Duplicate indices and endpoint conflicts can be represented so that the
checker returns ``STRUCTURALLY_INVALID``. Unsafe integer arithmetic, invalid
array ranks, and invalid static identifiers are rejected at construction.
"""

import math
from enum import Enum, IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float64,
    Int,
    Int32,
    Int64,
    jaxtyped,
)

from .born_potential_types import GalerkinProductSupport

_SPACE_DIMENSIONS: int = 3
_TRANSVERSE_DIMENSIONS: int = 2
_SUPPORT_RANK: int = 2
_CONTRACT_VERSION: str = "SC-1/RM-S1"
_DIRECT_TRANSFER_RULE: str = "K_d-Q_in subset K_chi"
_TERMINAL_KIND: str = "coordinate_aligned_fiber_complete"


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural acquisition failure.

    Parameters
    ----------
    condition : bool
        Structural failure predicate.
    message : str
        Exception message used when the predicate is true.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


class GalerkinAcquisitionSupportFailure(IntFlag):
    """Enumerate fail-closed RM-S1 predicate bits.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    NONE : int
        No predicate failed.
    CHECK_CAPACITY_EXCEEDED : int
        A binary set predicate exceeds the bounded checker ceiling.
    DUPLICATE_INDEX : int
        At least one submitted finite set repeats an index.
    ENDPOINT_CONFLICT : int
        An index is outside the declared signed half-open work quotient.
    INCIDENT_OUTSIDE_STATE : int
        ``Q_in`` is not a subset of ``K_u``.
    OUTGOING_OUTSIDE_PRETERMINAL : int
        ``Q_out`` is not a subset of ``K_d``.
    PRETERMINAL_OUTSIDE_STATE : int
        ``K_d`` is not a subset of ``K_u``.
    DIRECT_TRANSFER_MISSING : int
        ``K_d-Q_in`` is not a subset of ``K_chi``.
    ABSORBER_DIFFERENCE_MISSING : int
        ``K_u-K_u`` is not a subset of ``K_a``.
    WORK_PRODUCT_MISSING : int
        ``K_u+K_chi`` is not a subset of ``K_w``.
    INTERACTION_NOT_SIGN_SYMMETRIC : int
        ``K_chi`` is not sign-symmetric.
    ABSORBER_NOT_SIGN_SYMMETRIC : int
        ``K_a`` is not sign-symmetric.
    TERMINAL_FIBER_MISMATCH : int
        ``K_d`` differs from the selected complete state fibers.
    BACKWARD_DISPOSITION_INVALID : int
        The explicit backward declaration contradicts its masks or claim.
    SECTOR_MASK_INVALID : int
        Backward and grazing masks are not disjoint subsets of ``K_u``.
    CARRIER_CONTRACT_INVALID : int
        Nominal carrier diagnostics or single-owner metadata failed closed.
    DIRECTION_EVIDENCE_INVALID : int
        Physical direction, shell, or projection evidence failed closed.
    SECTOR_CLASSIFICATION_AMBIGUOUS : int
        Outward normal-component arithmetic could not prove one sign.
    OMITTED_MASK_INVALID : int
        A deliberately omitted mode is duplicated or represented in ``K_u``.
    """

    NONE = 0
    CHECK_CAPACITY_EXCEEDED = 1 << 0
    DUPLICATE_INDEX = 1 << 1
    ENDPOINT_CONFLICT = 1 << 2
    INCIDENT_OUTSIDE_STATE = 1 << 3
    OUTGOING_OUTSIDE_PRETERMINAL = 1 << 4
    PRETERMINAL_OUTSIDE_STATE = 1 << 5
    DIRECT_TRANSFER_MISSING = 1 << 6
    ABSORBER_DIFFERENCE_MISSING = 1 << 7
    WORK_PRODUCT_MISSING = 1 << 8
    INTERACTION_NOT_SIGN_SYMMETRIC = 1 << 9
    ABSORBER_NOT_SIGN_SYMMETRIC = 1 << 10
    TERMINAL_FIBER_MISMATCH = 1 << 11
    BACKWARD_DISPOSITION_INVALID = 1 << 12
    SECTOR_MASK_INVALID = 1 << 13
    CARRIER_CONTRACT_INVALID = 1 << 14
    DIRECTION_EVIDENCE_INVALID = 1 << 15
    SECTOR_CLASSIFICATION_AMBIGUOUS = 1 << 16
    OMITTED_MASK_INVALID = 1 << 17


class GalerkinAcquisitionSupportStatus(IntEnum):
    """Enumerate structural, support-ineligible, and support-eligible outcomes.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    STRUCTURALLY_INVALID : int
        Duplicate or endpoint-conflicted data cannot denote the manifest.
    SUPPORT_INELIGIBLE : int
        The submission is structural but misses a support-core predicate.
    SUPPORT_ELIGIBLE : int
        Every implemented RM-S1 support-core predicate passes.
    """

    STRUCTURALLY_INVALID = 0
    SUPPORT_INELIGIBLE = 1
    SUPPORT_ELIGIBLE = 2


class GalerkinBackwardDisposition(str, Enum):
    """Store the declared treatment of the backward sector.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    EXCLUDED : str
        No backward coefficient is represented and no backscatter is claimed.
    REPRESENTED : str
        A nonempty exact backward-index mask is represented in ``K_u``.
    """

    EXCLUDED = "excluded"
    REPRESENTED = "represented"


class GalerkinCarrierOwnership(str, Enum):
    """Store the admitted carrier-block ownership policy.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    INDEPENDENT_SINGLE_CARRIER : str
        One carrier owns every coefficient in this independent instance.
    """

    INDEPENDENT_SINGLE_CARRIER = "independent_single_carrier"


class GalerkinCarrierOverlapDisposition(str, Enum):
    """Store the explicit single-carrier overlap disposition.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    NO_OTHER_CARRIER_BLOCKS : str
        This independent instance contains no second carrier block to overlap.
    """

    NO_OTHER_CARRIER_BLOCKS = "no_other_carrier_blocks"


class GalerkinCarrierTargetRoute(str, Enum):
    """Store the exact target-side carrier normalization route.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    NORMALIZE_FROM_ACCELERATING_VOLTAGE : str
        The target uses exact ``k0(U0) * carrier / ||carrier||`` and RM-S2
        encloses the discrepancy from the stored binary64 seed and diagonal.
    """

    NORMALIZE_FROM_ACCELERATING_VOLTAGE = "normalize_from_accelerating_voltage"


class GalerkinDirectionDisposition(IntEnum):
    """Store whether a requested direction is exact or projected.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    EXACT_COEFFICIENT : int
        The physical wavevector is the zero carrier coefficient. Nonzero
        elastic coefficients require a later exact symbolic shell witness and
        are conservatively projected by this support-core artifact.
    PROJECTED : int
        The coefficient represents a separately bounded physical direction.
    """

    EXACT_COEFFICIENT = 0
    PROJECTED = 1


class GalerkinEndpointConvention(str, Enum):
    """Store the signed integer endpoint realization.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    SIGNED_HALF_OPEN : str
        Axis ``N`` uses indices ``[-N//2, (N-1)//2]``.
    """

    SIGNED_HALF_OPEN = "signed_half_open"


class GalerkinTerminalSide(str, Enum):
    """Store the oriented coordinate face used by the terminal.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    NEGATIVE : str
        The terminal is on the negative coordinate face.
    POSITIVE : str
        The terminal is on the positive coordinate face.
    """

    NEGATIVE = "negative"
    POSITIVE = "positive"


class GalerkinAcquisitionManifest(eqx.Module):
    """Store one submitted finite acquisition-support manifest.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    support : GalerkinProductSupport
        The bound ``K_u``, ``K_chi``, ``K_a``, and ``K_w`` supports.
    incident_indices : Int64[Array, "i 3"]
        Exact integer illumination offsets ``Q_in``.
    elastic_outgoing_indices : Int64[Array, "o 3"]
        Exact requested elastic outgoing offsets ``Q_out``.
    preterminal_indices : Int64[Array, "m 3"]
        Active three-dimensional preterminal support ``K_d``.
    transverse_indices : Int64[Array, "t 2"]
        Selected transverse harmonics defining complete ``K_d`` fibers.
    deliberately_omitted_indices : Int64[Array, "v 3"]
        Finite requested/audited modes deliberately omitted from ``K_u``.
    incident_physical_wavevectors : Float64[Array, "i 3"]
        Requested nominal physical incident angular-wavevector metadata.
    outgoing_physical_wavevectors : Float64[Array, "o 3"]
        Requested nominal physical outgoing angular-wavevector metadata.
    incident_direction_dispositions : Int32[Array, "i"]
        Exact-coefficient or projected incident-direction codes.
    outgoing_direction_dispositions : Int32[Array, "o"]
        Exact-coefficient or projected outgoing-direction codes.
    incident_on_shell_defect_bounds : Float64[Array, "i"]
        Submitted outward incident shell-defect bounds.
    outgoing_on_shell_defect_bounds : Float64[Array, "o"]
        Submitted outward outgoing shell-defect bounds.
    incident_projection_error_bounds : Float64[Array, "i"]
        Submitted outward incident coefficient-projection bounds.
    outgoing_projection_error_bounds : Float64[Array, "o"]
        Submitted outward outgoing coefficient-projection bounds.
    carrier : Float64[Array, "3"]
        Nonzero real algebraic carrier-direction seed in radians per unit
        length. It is not itself an exact SC-1 on-shell certificate.
    box_lengths : Float64[Array, "3"]
        Positive physical box lengths in coordinate-axis order.
    wavenumber : Float64[Array, ""]
        Positive nominal binary64 vacuum angular wavenumber.
    carrier_on_shell_defect_bound : Float64[Array, ""]
        Submitted outward nominal-seed shell-defect diagnostic bound.
    on_shell_defect_tolerance : Float64[Array, ""]
        Declared maximum admitted outward squared-shell defect.
    terminal_axis : int
        Static normal coordinate axis. This value affects tracing.
    terminal_side : GalerkinTerminalSide
        Static oriented terminal face. This value affects tracing.
    carrier_id : str
        Static nonempty canonical owner of this carrier frame.
    carrier_ownership : GalerkinCarrierOwnership
        Static carrier-block ownership policy. This value affects tracing.
    carrier_overlap_disposition : GalerkinCarrierOverlapDisposition
        Static single-carrier overlap disposition. This value affects tracing.
    carrier_target_route : GalerkinCarrierTargetRoute
        Static exact target-side normalization route. This value affects
        tracing.
    endpoint_convention : GalerkinEndpointConvention
        Static signed endpoint convention. This value affects tracing.
    backward_disposition : GalerkinBackwardDisposition
        Static backward-sector treatment. This value affects tracing.
    backward_exclusion_basis : str
        Static nonempty basis when the backward sector is excluded.
    claims_backscatter : bool
        Static declaration of whether this artifact claims backscatter.
    contract_version : str
        Static governing support-contract version. This value affects tracing.
    direct_transfer_rule : str
        Static exact represented-transfer rule. This value affects tracing.
    terminal_kind : str
        Static terminal-construction identifier. This value affects tracing.

    Notes
    -----
    One instance owns exactly one independent carrier frame. Equivalent
    multicarrier quotient assembly and cross-block edges are not silently
    represented by this carrier.
    """

    support: GalerkinProductSupport
    incident_indices: Int64[Array, "i 3"]
    elastic_outgoing_indices: Int64[Array, "o 3"]
    preterminal_indices: Int64[Array, "m 3"]
    transverse_indices: Int64[Array, "t 2"]
    deliberately_omitted_indices: Int64[Array, "v 3"]
    incident_physical_wavevectors: Float64[Array, "i 3"]
    outgoing_physical_wavevectors: Float64[Array, "o 3"]
    incident_direction_dispositions: Int32[Array, " i"]
    outgoing_direction_dispositions: Int32[Array, " o"]
    incident_on_shell_defect_bounds: Float64[Array, " i"]
    outgoing_on_shell_defect_bounds: Float64[Array, " o"]
    incident_projection_error_bounds: Float64[Array, " i"]
    outgoing_projection_error_bounds: Float64[Array, " o"]
    carrier: Float64[Array, " 3"]
    box_lengths: Float64[Array, " 3"]
    wavenumber: Float64[Array, ""]
    carrier_on_shell_defect_bound: Float64[Array, ""]
    on_shell_defect_tolerance: Float64[Array, ""]
    terminal_axis: int = eqx.field(static=True)
    terminal_side: GalerkinTerminalSide = eqx.field(static=True)
    carrier_id: str = eqx.field(static=True)
    carrier_ownership: GalerkinCarrierOwnership = eqx.field(static=True)
    carrier_overlap_disposition: GalerkinCarrierOverlapDisposition = eqx.field(
        static=True
    )
    carrier_target_route: GalerkinCarrierTargetRoute = eqx.field(static=True)
    endpoint_convention: GalerkinEndpointConvention = eqx.field(static=True)
    backward_disposition: GalerkinBackwardDisposition = eqx.field(static=True)
    backward_exclusion_basis: str = eqx.field(static=True)
    claims_backscatter: bool = eqx.field(static=True)
    contract_version: str = eqx.field(static=True)
    direct_transfer_rule: str = eqx.field(static=True)
    terminal_kind: str = eqx.field(static=True)


class GalerkinAcquisitionSupportResult(eqx.Module):
    """Store one checked RM-S1 acquisition-support artifact.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Attributes
    ----------
    manifest : GalerkinAcquisitionManifest
        Exact submitted acquisition data checked by this artifact.
    status : Int32[Array, ""]
        Numeric :class:`GalerkinAcquisitionSupportStatus` value.
    failure_mask : Int64[Array, ""]
        Bitwise :class:`GalerkinAcquisitionSupportFailure` payload.
    structural_valid : Bool[Array, ""]
        Whether uniqueness and endpoint-realization predicates pass.
    support_eligible : Bool[Array, ""]
        Whether every implemented RM-S1 support-core predicate passes.
    check_capacity_admitted : Bool[Array, ""]
        Whether every binary predicate stayed within the checker ceiling.
    incident_in_state : Bool[Array, ""]
        Exact ``Q_in subset K_u`` result.
    outgoing_in_preterminal : Bool[Array, ""]
        Exact ``Q_out subset K_d`` result.
    preterminal_in_state : Bool[Array, ""]
        Exact ``K_d subset K_u`` result.
    direct_transfers_represented : Bool[Array, ""]
        Exact bounded ``K_d-Q_in subset K_chi`` result.
    absorber_differences_represented : Bool[Array, ""]
        Exact bounded ``K_u-K_u subset K_a`` result.
    work_products_represented : Bool[Array, ""]
        Exact bounded ``K_u+K_chi subset K_w`` result.
    interaction_sign_symmetric : Bool[Array, ""]
        Whether ``K_chi`` contains every additive inverse.
    absorber_sign_symmetric : Bool[Array, ""]
        Whether ``K_a`` contains every additive inverse.
    terminal_fiber_complete : Bool[Array, ""]
        Whether ``K_d`` equals the selected complete ``K_u`` fibers.
    backward_disposition_valid : Bool[Array, ""]
        Whether masks, exclusion basis, and claim agree with disposition.
    sector_masks_valid : Bool[Array, ""]
        Whether computed represented and omitted sector masks are complete.
    carrier_contract_valid : Bool[Array, ""]
        Whether nominal carrier diagnostics, ownership, overlap, and the exact
        target-normalization route pass.
    direction_evidence_valid : Bool[Array, ""]
        Whether physical shell and exact/projected evidence passes.
    sector_classification_complete : Bool[Array, ""]
        Whether every normal sign is proved without an ambiguous interval.
    omitted_mask_valid : Bool[Array, ""]
        Whether omitted modes are unique and disjoint from ``K_u``.
    state_forward_mask : Bool[Array, "n"]
        Complete proved-forward mask on ``K_u``.
    state_grazing_mask : Bool[Array, "n"]
        Complete exactly-zero normal-component mask on ``K_u``.
    state_backward_mask : Bool[Array, "n"]
        Complete proved-backward mask on ``K_u``.
    state_ambiguous_mask : Bool[Array, "n"]
        Modes whose outward normal interval contains but is not exactly zero.
    state_oriented_normal_interval_lower : Float64[Array, "n"]
        Outward lower endpoints of nominal oriented state normal components.
    state_oriented_normal_interval_upper : Float64[Array, "n"]
        Outward upper endpoints of nominal oriented state normal components.
    omitted_forward_mask : Bool[Array, "v"]
        Proved-forward mask on deliberately omitted modes.
    omitted_grazing_mask : Bool[Array, "v"]
        Exactly-grazing mask on deliberately omitted modes.
    omitted_backward_mask : Bool[Array, "v"]
        Proved-backward mask on deliberately omitted modes.
    omitted_ambiguous_mask : Bool[Array, "v"]
        Ambiguous-sign mask on deliberately omitted modes.
    omitted_oriented_normal_interval_lower : Float64[Array, "v"]
        Outward lower endpoints of nominal oriented omitted normal components.
    omitted_oriented_normal_interval_upper : Float64[Array, "v"]
        Outward upper endpoints of nominal oriented omitted normal components.
    carrier_shell_defect_upper_bound : Float64[Array, ""]
        Independently recomputed outward carrier squared-shell defect.
    incident_shell_defect_upper_bounds : Float64[Array, "i"]
        Independently recomputed incident squared-shell defects.
    outgoing_shell_defect_upper_bounds : Float64[Array, "o"]
        Independently recomputed outgoing squared-shell defects.
    incident_projection_error_upper_bounds : Float64[Array, "i"]
        Independently recomputed projected-row discrepancies; exact rows are
        symbolic zero after their canonical binary64 round-trip check.
    outgoing_projection_error_upper_bounds : Float64[Array, "o"]
        Independently recomputed projected-row discrepancies; exact rows are
        symbolic zero after their canonical binary64 round-trip check.
    incident_transverse_offset_max : Float64[Array, ""]
        Outward maximum incident transverse cyclic offset.
    incident_full_offset_max : Float64[Array, ""]
        Outward maximum incident full cyclic offset.
    outgoing_transverse_offset_max : Float64[Array, ""]
        Outward maximum outgoing transverse cyclic offset.
    outgoing_full_offset_max : Float64[Array, ""]
        Outward maximum outgoing full cyclic offset.
    transfer_transverse_max : Float64[Array, ""]
        Outward maximum incident-to-outgoing transverse cyclic transfer.
    transfer_full_max : Float64[Array, ""]
        Outward maximum incident-to-outgoing full cyclic transfer.
    direct_transfer_pair_count : Int64[Array, ""]
        Number ``|K_d| |Q_in|`` of directed transfer pairs.
    represented_direct_transfer_pair_count : Int64[Array, ""]
        Number of those directed pairs whose difference lies in ``K_chi``.
    max_binary_pair_checks : int
        Static per-predicate checker ceiling. This value affects tracing.

    Notes
    -----
    ``SUPPORT_ELIGIBLE`` proves the finite support, geometry, sector, and
    ownership core of RM-S1. Exact SC-1 carrier normalization is deferred to
    the bound target route ``k0(U0) * carrier / ||carrier||``; RM-S2 must
    enclose the discrepancy from this stored binary64 seed and its nominal
    free diagonal. Without a separate exact symbolic shell witness, this
    artifact labels only the zero carrier coefficient ``EXACT_COEFFICIENT``;
    every nonzero elastic coefficient is ``PROJECTED``. A raw-pixel map and
    RM-S4 invocation are not attached here, so this is not full production
    detector eligibility. It also does not prove repeated-scattering,
    boundary, or continuum accuracy.
    """

    manifest: GalerkinAcquisitionManifest
    status: Int32[Array, ""]
    failure_mask: Int64[Array, ""]
    structural_valid: Bool[Array, ""]
    support_eligible: Bool[Array, ""]
    check_capacity_admitted: Bool[Array, ""]
    incident_in_state: Bool[Array, ""]
    outgoing_in_preterminal: Bool[Array, ""]
    preterminal_in_state: Bool[Array, ""]
    direct_transfers_represented: Bool[Array, ""]
    absorber_differences_represented: Bool[Array, ""]
    work_products_represented: Bool[Array, ""]
    interaction_sign_symmetric: Bool[Array, ""]
    absorber_sign_symmetric: Bool[Array, ""]
    terminal_fiber_complete: Bool[Array, ""]
    backward_disposition_valid: Bool[Array, ""]
    sector_masks_valid: Bool[Array, ""]
    carrier_contract_valid: Bool[Array, ""]
    direction_evidence_valid: Bool[Array, ""]
    sector_classification_complete: Bool[Array, ""]
    omitted_mask_valid: Bool[Array, ""]
    state_forward_mask: Bool[Array, " n"]
    state_grazing_mask: Bool[Array, " n"]
    state_backward_mask: Bool[Array, " n"]
    state_ambiguous_mask: Bool[Array, " n"]
    state_oriented_normal_interval_lower: Float64[Array, " n"]
    state_oriented_normal_interval_upper: Float64[Array, " n"]
    omitted_forward_mask: Bool[Array, " v"]
    omitted_grazing_mask: Bool[Array, " v"]
    omitted_backward_mask: Bool[Array, " v"]
    omitted_ambiguous_mask: Bool[Array, " v"]
    omitted_oriented_normal_interval_lower: Float64[Array, " v"]
    omitted_oriented_normal_interval_upper: Float64[Array, " v"]
    carrier_shell_defect_upper_bound: Float64[Array, ""]
    incident_shell_defect_upper_bounds: Float64[Array, " i"]
    outgoing_shell_defect_upper_bounds: Float64[Array, " o"]
    incident_projection_error_upper_bounds: Float64[Array, " i"]
    outgoing_projection_error_upper_bounds: Float64[Array, " o"]
    incident_transverse_offset_max: Float64[Array, ""]
    incident_full_offset_max: Float64[Array, ""]
    outgoing_transverse_offset_max: Float64[Array, ""]
    outgoing_full_offset_max: Float64[Array, ""]
    transfer_transverse_max: Float64[Array, ""]
    transfer_full_max: Float64[Array, ""]
    direct_transfer_pair_count: Int64[Array, ""]
    represented_direct_transfer_pair_count: Int64[Array, ""]
    max_binary_pair_checks: int = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_acquisition_manifest(  # noqa: PLR0913
    support: GalerkinProductSupport,
    incident_indices: Int[Array, "..."],
    elastic_outgoing_indices: Int[Array, "..."],
    preterminal_indices: Int[Array, "..."],
    transverse_indices: Int[Array, "..."],
    deliberately_omitted_indices: Int[Array, "..."],
    *,
    incident_physical_wavevectors: Float[Array, "..."],
    outgoing_physical_wavevectors: Float[Array, "..."],
    incident_direction_dispositions: Int[Array, "..."],
    outgoing_direction_dispositions: Int[Array, "..."],
    incident_on_shell_defect_bounds: Float[Array, "..."],
    outgoing_on_shell_defect_bounds: Float[Array, "..."],
    incident_projection_error_bounds: Float[Array, "..."],
    outgoing_projection_error_bounds: Float[Array, "..."],
    carrier: Float[Array, "..."],
    box_lengths: Float[Array, "..."],
    wavenumber: Float[Array, ""],
    carrier_on_shell_defect_bound: Float[Array, ""],
    on_shell_defect_tolerance: Float[Array, ""],
    terminal_axis: int,
    terminal_side: GalerkinTerminalSide | str,
    carrier_id: str,
    carrier_ownership: GalerkinCarrierOwnership | str,
    carrier_overlap_disposition: GalerkinCarrierOverlapDisposition | str,
    carrier_target_route: GalerkinCarrierTargetRoute | str,
    endpoint_convention: GalerkinEndpointConvention | str,
    backward_disposition: GalerkinBackwardDisposition | str,
    backward_exclusion_basis: str,
    claims_backscatter: bool,
) -> GalerkinAcquisitionManifest:
    """Create one structurally shaped acquisition submission.

    :see: :class:`~.test_acquisition_types.TestGalerkinAcquisitionTypes`

    Parameters
    ----------
    support : GalerkinProductSupport
        Bound state, interaction, absorber, and product-work supports.
    incident_indices : Int[Array, "..."]
        Nonempty exact integer illumination offsets ``Q_in``.
    elastic_outgoing_indices : Int[Array, "..."]
        Nonempty exact requested elastic outgoing offsets ``Q_out``.
    preterminal_indices : Int[Array, "..."]
        Nonempty active state-side preterminal support ``K_d``.
    transverse_indices : Int[Array, "..."]
        Nonempty selected transverse harmonics with shape ``(t, 2)``.
    deliberately_omitted_indices : Int[Array, "..."]
        Finite audited modes deliberately omitted from ``K_u``.
    incident_physical_wavevectors : Float[Array, "..."]
        Requested nominal incident angular wavevectors with shape ``(i, 3)``.
    outgoing_physical_wavevectors : Float[Array, "..."]
        Requested nominal outgoing angular wavevectors with shape ``(o, 3)``.
    incident_direction_dispositions : Int[Array, "..."]
        Exact/projected codes matching the incident rows.
    outgoing_direction_dispositions : Int[Array, "..."]
        Exact/projected codes matching the outgoing rows.
    incident_on_shell_defect_bounds : Float[Array, "..."]
        Submitted outward incident squared-shell bounds.
    outgoing_on_shell_defect_bounds : Float[Array, "..."]
        Submitted outward outgoing squared-shell bounds.
    incident_projection_error_bounds : Float[Array, "..."]
        Submitted outward incident coefficient-discrepancy bounds.
    outgoing_projection_error_bounds : Float[Array, "..."]
        Submitted outward outgoing coefficient-discrepancy bounds.
    carrier : Float[Array, "..."]
        Nonzero real algebraic carrier-direction seed with shape ``(3,)``.
    box_lengths : Float[Array, "..."]
        Positive physical box lengths with shape ``(3,)``.
    wavenumber : Float[Array, ""]
        Positive nominal binary64 vacuum angular wavenumber.
    carrier_on_shell_defect_bound : Float[Array, ""]
        Submitted outward nominal-seed squared-shell diagnostic bound.
    on_shell_defect_tolerance : Float[Array, ""]
        Maximum admitted outward squared-shell defect.
    terminal_axis : int
        Normal coordinate axis in ``{0, 1, 2}``.
    terminal_side : GalerkinTerminalSide | str
        Oriented coordinate face used by the terminal.
    carrier_id : str
        Nonempty canonical owner of this carrier frame.
    carrier_ownership : GalerkinCarrierOwnership | str
        Admitted carrier-block ownership policy.
    carrier_overlap_disposition : GalerkinCarrierOverlapDisposition | str
        Explicit no-other-block overlap disposition.
    carrier_target_route : GalerkinCarrierTargetRoute | str
        Exact target-side route that normalizes the seed from voltage.
    endpoint_convention : GalerkinEndpointConvention | str
        Signed work-quotient endpoint realization.
    backward_disposition : GalerkinBackwardDisposition | str
        Explicit represented or excluded backward-sector declaration.
    backward_exclusion_basis : str
        Claim basis used when the backward sector is excluded.
    claims_backscatter : bool
        Whether this artifact makes a backscatter claim.

    Returns
    -------
    manifest : GalerkinAcquisitionManifest
        Canonical exact-width submission ready for the support checker.

    Raises
    ------
    ValueError
        If ranks, dimensions, static identifiers, or work shape are invalid.
    equinox.EquinoxRuntimeError
        If an exact index could overflow required sum/difference arithmetic.

    Notes
    -----
    Duplicate and endpoint-conflicted arrays remain representable here. The
    support checker classifies those submissions as structurally invalid.
    """
    integer_arrays: Tuple[Int64[Array, "..."], ...] = tuple(
        jnp.asarray(value, dtype=jnp.int64)
        for value in (
            incident_indices,
            elastic_outgoing_indices,
            preterminal_indices,
            transverse_indices,
            deliberately_omitted_indices,
        )
    )
    (
        incident_array,
        outgoing_array,
        preterminal_array,
        transverse_array,
        omitted_array,
    ) = integer_arrays

    float_arrays: Tuple[Float64[Array, "..."], ...] = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            incident_physical_wavevectors,
            outgoing_physical_wavevectors,
            incident_on_shell_defect_bounds,
            outgoing_on_shell_defect_bounds,
            incident_projection_error_bounds,
            outgoing_projection_error_bounds,
            carrier,
            box_lengths,
            wavenumber,
            carrier_on_shell_defect_bound,
            on_shell_defect_tolerance,
        )
    )
    (
        incident_wavevectors,
        outgoing_wavevectors,
        incident_shell_bounds,
        outgoing_shell_bounds,
        incident_projection_bounds,
        outgoing_projection_bounds,
        carrier_array,
        box_array,
        wavenumber_array,
        carrier_shell_bound,
        shell_tolerance,
    ) = float_arrays
    incident_dispositions: Int32[Array, " i"] = jnp.asarray(
        incident_direction_dispositions, dtype=jnp.int32
    )
    outgoing_dispositions: Int32[Array, " o"] = jnp.asarray(
        outgoing_direction_dispositions, dtype=jnp.int32
    )

    for values, name in (
        (incident_array, "incident_indices"),
        (outgoing_array, "elastic_outgoing_indices"),
        (preterminal_array, "preterminal_indices"),
        (omitted_array, "deliberately_omitted_indices"),
    ):
        _raise_if(
            values.ndim != _SUPPORT_RANK
            or values.shape[1:] != (_SPACE_DIMENSIONS,),
            f"{name} must have shape (n, 3)",
        )
    _raise_if(
        transverse_array.ndim != _SUPPORT_RANK
        or transverse_array.shape[1:] != (_TRANSVERSE_DIMENSIONS,),
        "transverse_indices must have shape (n, 2)",
    )
    for values, name in (
        (incident_array, "incident_indices"),
        (outgoing_array, "elastic_outgoing_indices"),
        (preterminal_array, "preterminal_indices"),
        (transverse_array, "transverse_indices"),
    ):
        _raise_if(values.shape[0] == 0, f"{name} must be nonempty")
    for values, count, name in (
        (
            incident_wavevectors,
            incident_array.shape[0],
            "incident_physical_wavevectors",
        ),
        (
            outgoing_wavevectors,
            outgoing_array.shape[0],
            "outgoing_physical_wavevectors",
        ),
    ):
        _raise_if(
            values.shape != (count, _SPACE_DIMENSIONS),
            f"{name} must match its coefficient set with shape (n, 3)",
        )
    for values, count, name in (
        (
            incident_dispositions,
            incident_array.shape[0],
            "incident_direction_dispositions",
        ),
        (
            outgoing_dispositions,
            outgoing_array.shape[0],
            "outgoing_direction_dispositions",
        ),
        (
            incident_shell_bounds,
            incident_array.shape[0],
            "incident_on_shell_defect_bounds",
        ),
        (
            outgoing_shell_bounds,
            outgoing_array.shape[0],
            "outgoing_on_shell_defect_bounds",
        ),
        (
            incident_projection_bounds,
            incident_array.shape[0],
            "incident_projection_error_bounds",
        ),
        (
            outgoing_projection_bounds,
            outgoing_array.shape[0],
            "outgoing_projection_error_bounds",
        ),
    ):
        _raise_if(values.shape != (count,), f"{name} must have shape (n,)")
    _raise_if(carrier_array.shape != (3,), "carrier must have shape (3,)")
    _raise_if(box_array.shape != (3,), "box_lengths must have shape (3,)")
    for values, name in (
        (wavenumber_array, "wavenumber"),
        (carrier_shell_bound, "carrier_on_shell_defect_bound"),
        (shell_tolerance, "on_shell_defect_tolerance"),
    ):
        _raise_if(values.shape != (), f"{name} must be a scalar")

    _raise_if(
        isinstance(terminal_axis, bool)
        or not 0 <= terminal_axis < _SPACE_DIMENSIONS,
        "terminal_axis must be one of 0, 1, or 2",
    )
    _raise_if(not carrier_id.strip(), "carrier_id must be nonempty")
    _raise_if(
        not isinstance(claims_backscatter, bool),
        "claims_backscatter must be boolean",
    )
    _raise_if(
        len(support.work_shape) != _SPACE_DIMENSIONS
        or any(
            isinstance(size, bool) or size <= 0 for size in support.work_shape
        ),
        "support work_shape must contain three positive integers",
    )
    _raise_if(
        math.prod(support.work_shape) > jnp.iinfo(jnp.int64).max,
        "support work_shape product must fit in signed 64-bit indices",
    )

    safe_limit: int = min(jnp.iinfo(jnp.int64).max // 4, 1 << 52)
    unsafe: Bool[Array, ""] = jnp.asarray(False)
    for values in (
        support.state_indices,
        support.interaction_indices,
        support.absorber_indices,
        support.work_indices,
        incident_array,
        outgoing_array,
        preterminal_array,
        transverse_array,
        omitted_array,
    ):
        unsafe = unsafe | jnp.any(
            (values < -safe_limit) | (values > safe_limit)
        )
    checked_incident: Int64[Array, "i 3"] = eqx.error_if(
        incident_array,
        unsafe,
        "acquisition indices must permit exact int64 sums and differences",
    )
    invalid_float_data: Bool[Array, ""] = (
        jnp.any(~jnp.isfinite(incident_wavevectors))
        | jnp.any(~jnp.isfinite(outgoing_wavevectors))
        | jnp.any(~jnp.isfinite(carrier_array))
        | ~jnp.any(carrier_array != 0.0)
        | jnp.any(~jnp.isfinite(box_array))
        | jnp.any(box_array <= 0.0)
        | (~jnp.isfinite(wavenumber_array))
        | (wavenumber_array <= 0.0)
    )
    checked_carrier: Float64[Array, " 3"] = eqx.error_if(
        carrier_array,
        invalid_float_data,
        "carrier must be nonzero and physical data must be finite",
    )
    invalid_bounds: Bool[Array, ""] = jnp.asarray(False)
    for values in (
        incident_shell_bounds,
        outgoing_shell_bounds,
        incident_projection_bounds,
        outgoing_projection_bounds,
        carrier_shell_bound,
        shell_tolerance,
    ):
        invalid_bounds = invalid_bounds | jnp.any(
            ~jnp.isfinite(values) | (values < 0.0)
        )
    checked_carrier_shell_bound: Float64[Array, ""] = eqx.error_if(
        carrier_shell_bound,
        invalid_bounds,
        "shell and projection bounds must be finite and non-negative",
    )

    manifest: GalerkinAcquisitionManifest = GalerkinAcquisitionManifest(
        support=support,
        incident_indices=checked_incident,
        elastic_outgoing_indices=outgoing_array,
        preterminal_indices=preterminal_array,
        transverse_indices=transverse_array,
        deliberately_omitted_indices=omitted_array,
        incident_physical_wavevectors=incident_wavevectors,
        outgoing_physical_wavevectors=outgoing_wavevectors,
        incident_direction_dispositions=incident_dispositions,
        outgoing_direction_dispositions=outgoing_dispositions,
        incident_on_shell_defect_bounds=incident_shell_bounds,
        outgoing_on_shell_defect_bounds=outgoing_shell_bounds,
        incident_projection_error_bounds=incident_projection_bounds,
        outgoing_projection_error_bounds=outgoing_projection_bounds,
        carrier=checked_carrier,
        box_lengths=box_array,
        wavenumber=wavenumber_array,
        carrier_on_shell_defect_bound=checked_carrier_shell_bound,
        on_shell_defect_tolerance=shell_tolerance,
        terminal_axis=terminal_axis,
        terminal_side=GalerkinTerminalSide(terminal_side),
        carrier_id=carrier_id,
        carrier_ownership=GalerkinCarrierOwnership(carrier_ownership),
        carrier_overlap_disposition=GalerkinCarrierOverlapDisposition(
            carrier_overlap_disposition
        ),
        carrier_target_route=GalerkinCarrierTargetRoute(carrier_target_route),
        endpoint_convention=GalerkinEndpointConvention(endpoint_convention),
        backward_disposition=GalerkinBackwardDisposition(backward_disposition),
        backward_exclusion_basis=backward_exclusion_basis,
        claims_backscatter=claims_backscatter,
        contract_version=_CONTRACT_VERSION,
        direct_transfer_rule=_DIRECT_TRANSFER_RULE,
        terminal_kind=_TERMINAL_KIND,
    )
    return manifest


def _failure_component(
    predicate: Bool[Array, ""],
    failure: GalerkinAcquisitionSupportFailure,
) -> Int64[Array, ""]:
    """PRIVATE: Encode one failed positive predicate as a failure bit.

    Parameters
    ----------
    predicate : Bool[Array, ""]
        Positive eligibility predicate.
    failure : GalerkinAcquisitionSupportFailure
        Failure bit associated with a false predicate.

    Returns
    -------
    component : Int64[Array, ""]
        Zero when the predicate holds, otherwise the exact failure bit.
    """
    component: Int64[Array, ""] = jnp.where(
        predicate,
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(int(failure), dtype=jnp.int64),
    )
    return component


@jaxtyped(typechecker=beartype)
def _create_galerkin_acquisition_support_result(  # noqa: PLR0913
    manifest: GalerkinAcquisitionManifest,
    unique: Bool[Array, ""],
    endpoint_valid: Bool[Array, ""],
    check_capacity_admitted: Bool[Array, ""],
    incident_in_state: Bool[Array, ""],
    outgoing_in_preterminal: Bool[Array, ""],
    preterminal_in_state: Bool[Array, ""],
    direct_transfers_represented: Bool[Array, ""],
    absorber_differences_represented: Bool[Array, ""],
    work_products_represented: Bool[Array, ""],
    interaction_sign_symmetric: Bool[Array, ""],
    absorber_sign_symmetric: Bool[Array, ""],
    terminal_fiber_complete: Bool[Array, ""],
    backward_disposition_valid: Bool[Array, ""],
    sector_masks_valid: Bool[Array, ""],
    carrier_contract_valid: Bool[Array, ""],
    direction_evidence_valid: Bool[Array, ""],
    sector_classification_complete: Bool[Array, ""],
    omitted_mask_valid: Bool[Array, ""],
    state_forward_mask: Bool[Array, " n"],
    state_grazing_mask: Bool[Array, " n"],
    state_backward_mask: Bool[Array, " n"],
    state_ambiguous_mask: Bool[Array, " n"],
    state_oriented_normal_interval_lower: Float[Array, " n"],
    state_oriented_normal_interval_upper: Float[Array, " n"],
    omitted_forward_mask: Bool[Array, " v"],
    omitted_grazing_mask: Bool[Array, " v"],
    omitted_backward_mask: Bool[Array, " v"],
    omitted_ambiguous_mask: Bool[Array, " v"],
    omitted_oriented_normal_interval_lower: Float[Array, " v"],
    omitted_oriented_normal_interval_upper: Float[Array, " v"],
    carrier_shell_defect_upper_bound: Float[Array, ""],
    incident_shell_defect_upper_bounds: Float[Array, " i"],
    outgoing_shell_defect_upper_bounds: Float[Array, " o"],
    incident_projection_error_upper_bounds: Float[Array, " i"],
    outgoing_projection_error_upper_bounds: Float[Array, " o"],
    incident_transverse_offset_max: Float[Array, ""],
    incident_full_offset_max: Float[Array, ""],
    outgoing_transverse_offset_max: Float[Array, ""],
    outgoing_full_offset_max: Float[Array, ""],
    transfer_transverse_max: Float[Array, ""],
    transfer_full_max: Float[Array, ""],
    direct_transfer_pair_count: Int[Array, ""],
    represented_direct_transfer_pair_count: Int[Array, ""],
    *,
    max_binary_pair_checks: int,
) -> GalerkinAcquisitionSupportResult:
    """PRIVATE: Derive one coherent acquisition support result.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Canonical acquisition submission bound to this result.
    unique : Bool[Array, ""]
        Exact predicate for uniqueness of every submitted index set.
    endpoint_valid : Bool[Array, ""]
        Exact signed-endpoint predicate.
    check_capacity_admitted : Bool[Array, ""]
        Whether every bounded pair check stayed within its ceiling.
    incident_in_state : Bool[Array, ""]
        Whether all incident indices belong to the state support.
    outgoing_in_preterminal : Bool[Array, ""]
        Whether all outgoing indices belong to the preterminal support.
    preterminal_in_state : Bool[Array, ""]
        Whether the preterminal support is contained in the state support.
    direct_transfers_represented : Bool[Array, ""]
        Whether every direct incident-to-outgoing transfer is represented.
    absorber_differences_represented : Bool[Array, ""]
        Whether every state-support difference is represented by the absorber.
    work_products_represented : Bool[Array, ""]
        Whether required interaction and absorber products fit the work grid.
    interaction_sign_symmetric : Bool[Array, ""]
        Exact sign-symmetry predicate for interaction indices.
    absorber_sign_symmetric : Bool[Array, ""]
        Exact sign-symmetry predicate for absorber indices.
    terminal_fiber_complete : Bool[Array, ""]
        Whether selected terminal fibers are complete.
    backward_disposition_valid : Bool[Array, ""]
        Whether backward-mode declarations match the stated acquisition role.
    sector_masks_valid : Bool[Array, ""]
        Whether submitted and recomputed sector masks agree.
    carrier_contract_valid : Bool[Array, ""]
        Whether carrier identity and ownership declarations are coherent.
    direction_evidence_valid : Bool[Array, ""]
        Whether physical direction evidence satisfies its declared bounds.
    sector_classification_complete : Bool[Array, ""]
        Whether every retained mode has an unambiguous sector.
    omitted_mask_valid : Bool[Array, ""]
        Whether deliberately omitted directions match their stored masks.
    state_forward_mask : Bool[Array, " n"]
        Recomputed forward mask on the state support.
    state_grazing_mask : Bool[Array, " n"]
        Recomputed grazing mask on the state support.
    state_backward_mask : Bool[Array, " n"]
        Recomputed backward mask on the state support.
    state_ambiguous_mask : Bool[Array, " n"]
        Recomputed ambiguous mask on the state support.
    state_oriented_normal_interval_lower : Float[Array, " n"]
        Outward lower oriented state-normal endpoints in radians per Angstrom.
    state_oriented_normal_interval_upper : Float[Array, " n"]
        Outward upper oriented state-normal endpoints in radians per Angstrom.
    omitted_forward_mask : Bool[Array, " v"]
        Recomputed forward mask for deliberately omitted directions.
    omitted_grazing_mask : Bool[Array, " v"]
        Recomputed grazing mask for deliberately omitted directions.
    omitted_backward_mask : Bool[Array, " v"]
        Recomputed backward mask for deliberately omitted directions.
    omitted_ambiguous_mask : Bool[Array, " v"]
        Recomputed ambiguous mask for deliberately omitted directions.
    omitted_oriented_normal_interval_lower : Float[Array, " v"]
        Outward lower omitted-direction normal endpoints in radians per
        Angstrom.
    omitted_oriented_normal_interval_upper : Float[Array, " v"]
        Outward upper omitted-direction normal endpoints in radians per
        Angstrom.
    carrier_shell_defect_upper_bound : Float[Array, ""]
        Outward carrier shell-defect upper bound in inverse square Angstroms.
    incident_shell_defect_upper_bounds : Float[Array, " i"]
        Outward incident shell-defect bounds in inverse square Angstroms.
    outgoing_shell_defect_upper_bounds : Float[Array, " o"]
        Outward outgoing shell-defect bounds in inverse square Angstroms.
    incident_projection_error_upper_bounds : Float[Array, " i"]
        Outward incident projection-error bounds in radians per Angstrom.
    outgoing_projection_error_upper_bounds : Float[Array, " o"]
        Outward outgoing projection-error bounds in radians per Angstrom.
    incident_transverse_offset_max : Float[Array, ""]
        Maximum incident transverse cyclic offset in inverse Angstroms.
    incident_full_offset_max : Float[Array, ""]
        Maximum incident full cyclic offset in inverse Angstroms.
    outgoing_transverse_offset_max : Float[Array, ""]
        Maximum outgoing transverse cyclic offset in inverse Angstroms.
    outgoing_full_offset_max : Float[Array, ""]
        Maximum outgoing full cyclic offset in inverse Angstroms.
    transfer_transverse_max : Float[Array, ""]
        Maximum transverse incident-to-outgoing transfer in inverse Angstroms.
    transfer_full_max : Float[Array, ""]
        Maximum full incident-to-outgoing transfer in inverse Angstroms.
    direct_transfer_pair_count : Int[Array, ""]
        Total direct-transfer pair count.
    represented_direct_transfer_pair_count : Int[Array, ""]
        Number of represented direct-transfer pairs.
    max_binary_pair_checks : int
        Positive static pair-check ceiling. Changing it retraces compiled
        callers.

    Returns
    -------
    result : GalerkinAcquisitionSupportResult
        Status, failure mask, predicates, masks, and recomputed bounds.

    Raises
    ------
    ValueError
        If the static pair-check ceiling is not a positive integer.
    """
    _raise_if(
        isinstance(max_binary_pair_checks, bool)
        or max_binary_pair_checks <= 0,
        "max_binary_pair_checks must be a positive integer",
    )
    structural_valid: Bool[Array, ""] = unique & endpoint_valid
    support_eligible: Bool[Array, ""] = (
        structural_valid
        & check_capacity_admitted
        & incident_in_state
        & outgoing_in_preterminal
        & preterminal_in_state
        & direct_transfers_represented
        & absorber_differences_represented
        & work_products_represented
        & interaction_sign_symmetric
        & absorber_sign_symmetric
        & terminal_fiber_complete
        & backward_disposition_valid
        & sector_masks_valid
        & carrier_contract_valid
        & direction_evidence_valid
        & sector_classification_complete
        & omitted_mask_valid
    )
    status: Int32[Array, ""] = jnp.where(
        ~structural_valid,
        jnp.asarray(
            GalerkinAcquisitionSupportStatus.STRUCTURALLY_INVALID,
            dtype=jnp.int32,
        ),
        jnp.where(
            support_eligible,
            jnp.asarray(
                GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE,
                dtype=jnp.int32,
            ),
            jnp.asarray(
                GalerkinAcquisitionSupportStatus.SUPPORT_INELIGIBLE,
                dtype=jnp.int32,
            ),
        ),
    )
    failure_mask: Int64[Array, ""] = jnp.asarray(0, dtype=jnp.int64)
    for predicate, failure in (
        (
            check_capacity_admitted,
            GalerkinAcquisitionSupportFailure.CHECK_CAPACITY_EXCEEDED,
        ),
        (unique, GalerkinAcquisitionSupportFailure.DUPLICATE_INDEX),
        (
            endpoint_valid,
            GalerkinAcquisitionSupportFailure.ENDPOINT_CONFLICT,
        ),
        (
            incident_in_state,
            GalerkinAcquisitionSupportFailure.INCIDENT_OUTSIDE_STATE,
        ),
        (
            outgoing_in_preterminal,
            GalerkinAcquisitionSupportFailure.OUTGOING_OUTSIDE_PRETERMINAL,
        ),
        (
            preterminal_in_state,
            GalerkinAcquisitionSupportFailure.PRETERMINAL_OUTSIDE_STATE,
        ),
        (
            direct_transfers_represented,
            GalerkinAcquisitionSupportFailure.DIRECT_TRANSFER_MISSING,
        ),
        (
            absorber_differences_represented,
            GalerkinAcquisitionSupportFailure.ABSORBER_DIFFERENCE_MISSING,
        ),
        (
            work_products_represented,
            GalerkinAcquisitionSupportFailure.WORK_PRODUCT_MISSING,
        ),
        (
            interaction_sign_symmetric,
            GalerkinAcquisitionSupportFailure.INTERACTION_NOT_SIGN_SYMMETRIC,
        ),
        (
            absorber_sign_symmetric,
            GalerkinAcquisitionSupportFailure.ABSORBER_NOT_SIGN_SYMMETRIC,
        ),
        (
            terminal_fiber_complete,
            GalerkinAcquisitionSupportFailure.TERMINAL_FIBER_MISMATCH,
        ),
        (
            backward_disposition_valid,
            GalerkinAcquisitionSupportFailure.BACKWARD_DISPOSITION_INVALID,
        ),
        (
            sector_masks_valid,
            GalerkinAcquisitionSupportFailure.SECTOR_MASK_INVALID,
        ),
        (
            carrier_contract_valid,
            GalerkinAcquisitionSupportFailure.CARRIER_CONTRACT_INVALID,
        ),
        (
            direction_evidence_valid,
            GalerkinAcquisitionSupportFailure.DIRECTION_EVIDENCE_INVALID,
        ),
        (
            sector_classification_complete,
            GalerkinAcquisitionSupportFailure.SECTOR_CLASSIFICATION_AMBIGUOUS,
        ),
        (
            omitted_mask_valid,
            GalerkinAcquisitionSupportFailure.OMITTED_MASK_INVALID,
        ),
    ):
        failure_mask = jnp.bitwise_or(
            failure_mask, _failure_component(predicate, failure)
        )

    pair_count: Int64[Array, ""] = jnp.asarray(
        direct_transfer_pair_count, dtype=jnp.int64
    )
    represented_count: Int64[Array, ""] = jnp.asarray(
        represented_direct_transfer_pair_count, dtype=jnp.int64
    )
    checked_pair_count: Int64[Array, ""] = eqx.error_if(
        pair_count,
        (pair_count < 0)
        | (represented_count < 0)
        | (represented_count > pair_count),
        "represented transfer count must lie within the pair count",
    )
    state_normal_lower: Float64[Array, " n"] = jnp.asarray(
        state_oriented_normal_interval_lower, dtype=jnp.float64
    )
    state_normal_upper: Float64[Array, " n"] = jnp.asarray(
        state_oriented_normal_interval_upper, dtype=jnp.float64
    )
    omitted_normal_lower: Float64[Array, " v"] = jnp.asarray(
        omitted_oriented_normal_interval_lower, dtype=jnp.float64
    )
    omitted_normal_upper: Float64[Array, " v"] = jnp.asarray(
        omitted_oriented_normal_interval_upper, dtype=jnp.float64
    )
    _raise_if(
        state_normal_lower.shape != manifest.support.state_indices.shape[:1]
        or state_normal_upper.shape
        != manifest.support.state_indices.shape[:1],
        "state normal intervals must match the state support",
    )
    _raise_if(
        omitted_normal_lower.shape
        != manifest.deliberately_omitted_indices.shape[:1]
        or omitted_normal_upper.shape
        != manifest.deliberately_omitted_indices.shape[:1],
        "omitted normal intervals must match the omitted support",
    )
    invalid_normal_intervals: Bool[Array, ""] = (
        jnp.any(~jnp.isfinite(state_normal_lower))
        | jnp.any(~jnp.isfinite(state_normal_upper))
        | jnp.any(state_normal_lower > state_normal_upper)
        | jnp.any(~jnp.isfinite(omitted_normal_lower))
        | jnp.any(~jnp.isfinite(omitted_normal_upper))
        | jnp.any(omitted_normal_lower > omitted_normal_upper)
    )
    checked_state_normal_lower: Float64[Array, " n"] = eqx.error_if(
        state_normal_lower,
        invalid_normal_intervals,
        "oriented normal intervals must be finite and ordered",
    )
    float_values: Tuple[Float64[Array, "..."], ...] = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            carrier_shell_defect_upper_bound,
            incident_shell_defect_upper_bounds,
            outgoing_shell_defect_upper_bounds,
            incident_projection_error_upper_bounds,
            outgoing_projection_error_upper_bounds,
            incident_transverse_offset_max,
            incident_full_offset_max,
            outgoing_transverse_offset_max,
            outgoing_full_offset_max,
            transfer_transverse_max,
            transfer_full_max,
        )
    )
    invalid_float: Bool[Array, ""] = jnp.asarray(False)
    for value in float_values:
        invalid_float = invalid_float | jnp.any(
            ~jnp.isfinite(value) | (value < 0.0)
        )
    checked_carrier_shell: Float64[Array, ""] = eqx.error_if(
        float_values[0],
        invalid_float,
        "recomputed geometry bounds must be finite and non-negative",
    )

    result: GalerkinAcquisitionSupportResult = (
        GalerkinAcquisitionSupportResult(
            manifest=manifest,
            status=status,
            failure_mask=failure_mask,
            structural_valid=structural_valid,
            support_eligible=support_eligible,
            check_capacity_admitted=check_capacity_admitted,
            incident_in_state=incident_in_state,
            outgoing_in_preterminal=outgoing_in_preterminal,
            preterminal_in_state=preterminal_in_state,
            direct_transfers_represented=direct_transfers_represented,
            absorber_differences_represented=(
                absorber_differences_represented
            ),
            work_products_represented=work_products_represented,
            interaction_sign_symmetric=interaction_sign_symmetric,
            absorber_sign_symmetric=absorber_sign_symmetric,
            terminal_fiber_complete=terminal_fiber_complete,
            backward_disposition_valid=backward_disposition_valid,
            sector_masks_valid=sector_masks_valid,
            carrier_contract_valid=carrier_contract_valid,
            direction_evidence_valid=direction_evidence_valid,
            sector_classification_complete=sector_classification_complete,
            omitted_mask_valid=omitted_mask_valid,
            state_forward_mask=state_forward_mask,
            state_grazing_mask=state_grazing_mask,
            state_backward_mask=state_backward_mask,
            state_ambiguous_mask=state_ambiguous_mask,
            state_oriented_normal_interval_lower=checked_state_normal_lower,
            state_oriented_normal_interval_upper=state_normal_upper,
            omitted_forward_mask=omitted_forward_mask,
            omitted_grazing_mask=omitted_grazing_mask,
            omitted_backward_mask=omitted_backward_mask,
            omitted_ambiguous_mask=omitted_ambiguous_mask,
            omitted_oriented_normal_interval_lower=omitted_normal_lower,
            omitted_oriented_normal_interval_upper=omitted_normal_upper,
            carrier_shell_defect_upper_bound=checked_carrier_shell,
            incident_shell_defect_upper_bounds=float_values[1],
            outgoing_shell_defect_upper_bounds=float_values[2],
            incident_projection_error_upper_bounds=float_values[3],
            outgoing_projection_error_upper_bounds=float_values[4],
            incident_transverse_offset_max=float_values[5],
            incident_full_offset_max=float_values[6],
            outgoing_transverse_offset_max=float_values[7],
            outgoing_full_offset_max=float_values[8],
            transfer_transverse_max=float_values[9],
            transfer_full_max=float_values[10],
            direct_transfer_pair_count=checked_pair_count,
            represented_direct_transfer_pair_count=represented_count,
            max_binary_pair_checks=max_binary_pair_checks,
        )
    )
    return result


__all__: list[str] = [
    "GalerkinAcquisitionManifest",
    "GalerkinAcquisitionSupportFailure",
    "GalerkinAcquisitionSupportResult",
    "GalerkinAcquisitionSupportStatus",
    "GalerkinBackwardDisposition",
    "GalerkinCarrierOverlapDisposition",
    "GalerkinCarrierOwnership",
    "GalerkinCarrierTargetRoute",
    "GalerkinDirectionDisposition",
    "GalerkinEndpointConvention",
    "GalerkinTerminalSide",
    "create_galerkin_acquisition_manifest",
]
