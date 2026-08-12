r"""Certify one exact local-cell vacuum/source-free terminal slab.

Extended Summary
----------------
The host-only route derives a canonical unwrapped cell-layer lift from two
exact stored binary64 Cauchy-plane coordinates.  It proves the LVT.22
potential, CAP, and source predicates from local carriers separately.  It
never infers spatial absence from a rounded coefficient-space cancellation.

Routine Listings
----------------
:func:`certify_local_zero_slab`
    Replay represented-source evidence and certify exact LVT.21--LVT.22.
:func:`prepare_local_zero_slab_certificate`
    Replay every nested carrier, exact predicate, transcript, and digest.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jax.core import Tracer
from jaxtyping import Array, Bool, Float, Int64, jaxtyped
from numpy.typing import NDArray

from ptyrodactyl._tools import sha256, stored_value_payload
from ptyrodactyl.galerkin.local_represented_sources import (
    prepare_local_represented_source_certificate,
)
from ptyrodactyl.types.acquisition_types import GalerkinDirectionDisposition
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalRepresentedSourceCertificate,
)
from ptyrodactyl.types.local_source_types import (
    GalerkinLocalAdditionalSourceRoute,
)
from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalVacuumReference,
    GalerkinLocalZeroSlabCertificate,
    GalerkinLocalZeroSlabFailure,
    _make_local_zero_slab_certificate,
)

_CERTIFICATE_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_zero_slab.lvt21_lvt22_evidence.v1"
)
_COMPLETION_SCOPE: str = (
    "exact local open-slab absence only; excludes solved-state projection "
    "defect, physical Cauchy-trace periodicity, branch extraction, current, "
    "detector response, solver error, and per-call arithmetic"
)
_EXACT_TARGET: str = (
    "LVT.21 lifted full-transverse open slab with exact LVT.22 potential, "
    "LVT.22a physical CAP, and LVT.22b represented local source all zero"
)
_GEOMETRY_CONVENTION: str = (
    "exact Fraction arithmetic on target binary64 box/origin; centered "
    "half-open layers; canonical guarded consecutive unwrapped lift; xyz "
    "physical axes and zyx cell storage"
)
_NO_CANCELLATION_SCOPE: str = (
    "first-production sufficient-factor route only: D_i v, b_square, and "
    "s_add,square vanish separately; no coefficient, point-sample, tolerance, "
    "or local cancellation oracle"
)
_SLAB_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_zero_slab.lvt21_lvt22_identity.v1"
)
_SOURCE_ZERO_ROUTE: str = (
    "active incident rows are declared exact-coefficient singleton-zero "
    "free modes; exact CAP profile and ZERO-or-LOCAL_CELL q vanish on every "
    "selected full transverse layer"
)
_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_MINIMUM_SIGNED_INT64: int = np.iinfo(np.int64).min

type _HostLayerBoolVector = Bool[NDArray, " l"]
type _HostStateBoolVector = Bool[NDArray, " n"]
type _HostIntVector = Int64[NDArray, " l"]
type _IncidentEvidence = Tuple[
    _HostStateBoolVector,
    _HostStateBoolVector,
    _HostStateBoolVector,
    _HostStateBoolVector,
    bool,
]
type _LayerMasks = Tuple[
    _HostLayerBoolVector,
    _HostLayerBoolVector,
    _HostLayerBoolVector,
]


class _LayerLift(NamedTuple):
    """Carry one exact canonical guarded cell-layer lift."""

    start: int
    stop: int
    periodic_indices: _HostIntVector
    union_lower: Fraction
    union_upper: Fraction
    cap_zero_block_lift: int
    cap_zero_block_contains_layers: bool


class _EligibilityPredicates(NamedTuple):
    """Carry independent exact-spatial, projection, and final predicates."""

    cap_zero_block_contains_layers: bool
    incident_active_mask_consistent: bool
    vacuum_reference_eligible: bool
    potential_zero_eligible: bool
    cap_zero_eligible: bool
    incident_free_zero_eligible: bool
    additional_source_zero_eligible: bool
    exact_spatial_source_zero_eligible: bool
    exact_zero_slab_eligible: bool
    projection_match_eligible: bool
    terminal_zero_slab_eligible: bool


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced leaves at the host certification boundary.

    Parameters
    ----------
    value : object
        PyTree whose leaves must be concrete host-readable values.

    Raises
    ------
    ValueError
        If any PyTree leaf is a JAX tracer.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError(
            "zero-slab certification requires concrete host values"
        )


def _host_binary64_coordinate(value: object, name: str) -> np.float64:
    """PRIVATE: Require one finite normal-or-zero exact binary64 coordinate.

    Parameters
    ----------
    value : object
        Candidate scalar coordinate.
    name : str
        Field name used in a structural failure message.

    Returns
    -------
    coordinate : np.float64
        Exact submitted binary64 coordinate.

    Raises
    ------
    ValueError
        If dtype, shape, finiteness, or normal-range requirements fail.
    """
    array = np.asarray(jax.device_get(value))
    if (
        array.dtype != np.dtype(np.float64)
        or array.shape != ()
        or not np.isfinite(array)
        or (
            float(array) != 0.0
            and abs(float(array)) < np.finfo(np.float64).tiny
        )
    ):
        raise ValueError(
            f"{name} must be one finite normal-or-zero exact float64 scalar"
        )
    coordinate: np.float64 = np.float64(array)
    return coordinate


def _fraction_fields(value: Fraction) -> Tuple[str, str]:
    """PRIVATE: Return canonical exact numerator and denominator strings.

    Parameters
    ----------
    value : Fraction
        Exact rational value.

    Returns
    -------
    numerator : str
        Canonical signed numerator text.
    denominator : str
        Canonical positive denominator text.
    """
    fields: Tuple[str, str] = (str(value.numerator), str(value.denominator))
    return fields


def _floor_fraction(value: Fraction) -> int:
    """PRIVATE: Return the mathematical floor of one exact rational.

    Parameters
    ----------
    value : Fraction
        Exact rational value.

    Returns
    -------
    floor : int
        Mathematical floor as an arbitrary-precision integer.
    """
    floor: int = value.numerator // value.denominator
    return floor


def _derive_layer_lift(
    slab_lower: Fraction,
    slab_upper: Fraction,
    origin: Fraction,
    box_length: Fraction,
    layer_count: int,
    cap_zero_start: int,
    cap_zero_count: int,
) -> _LayerLift:
    """PRIVATE: Derive the canonical guarded unwrapped layer union.

    Parameters
    ----------
    slab_lower : Fraction
        Exact inner Cauchy-plane coordinate.
    slab_upper : Fraction
        Exact outer Cauchy-plane coordinate on the same unwrapped lift.
    origin : Fraction
        Exact center coordinate of periodic layer zero.
    box_length : Fraction
        Exact positive periodic axis length.
    layer_count : int
        Positive periodic cell-layer count.
    cap_zero_start : int
        Canonical periodic first L4 exact-zero layer.
    cap_zero_count : int
        Positive consecutive L4 exact-zero layer count.

    Returns
    -------
    lift : _LayerLift
        Canonical guarded layer lift and CAP-block containment evidence.

    Raises
    ------
    ValueError
        If the open slab, periodic geometry, or derived integer range is
        structurally invalid.
    AssertionError
        If the exact guarded-layer construction loses strict containment.
    """
    if box_length <= 0 or layer_count <= 0:
        raise ValueError("box length and layer count must be positive")
    width = slab_upper - slab_lower
    if width <= 0 or width > box_length:
        raise ValueError(
            "slab must have exact width in (0, terminal box length]"
        )
    if (
        cap_zero_start < 0
        or cap_zero_start >= layer_count
        or cap_zero_count <= 0
        or cap_zero_count > layer_count
    ):
        raise ValueError("CAP zero block is not canonical for this layer grid")
    delta = box_length / layer_count
    first_face = origin - delta / 2
    lower_ratio = (slab_lower - first_face) / delta
    upper_ratio = (slab_upper - first_face) / delta
    start = _floor_fraction(lower_ratio)
    if lower_ratio.denominator == 1:
        start -= 1
    stop = _floor_fraction(upper_ratio) + 1
    selected_count = stop - start
    if selected_count <= 0 or selected_count > layer_count:
        raise ValueError(
            "guarded Cauchy planes must fit within one periodic layer lift"
        )
    if (
        start < _MINIMUM_SIGNED_INT64
        or start > _MAXIMUM_SIGNED_INT64
        or stop < _MINIMUM_SIGNED_INT64
        or stop > _MAXIMUM_SIGNED_INT64
    ):
        raise ValueError("unwrapped layer transcript must fit signed int64")
    union_lower = first_face + start * delta
    union_upper = first_face + stop * delta
    if not union_lower < slab_lower < slab_upper < union_upper:
        raise AssertionError(
            "guarded exact layer derivation lost strict containment"
        )
    cap_lift = _floor_fraction(Fraction(start - cap_zero_start, layer_count))
    lifted_cap_start = cap_zero_start + cap_lift * layer_count
    block_contains = (
        lifted_cap_start <= start and stop <= lifted_cap_start + cap_zero_count
    )
    periodic_indices = np.asarray(
        [(start + offset) % layer_count for offset in range(selected_count)],
        dtype=np.int64,
    )
    lift: _LayerLift = _LayerLift(
        start=start,
        stop=stop,
        periodic_indices=periodic_indices,
        union_lower=union_lower,
        union_upper=union_upper,
        cap_zero_block_lift=cap_lift,
        cap_zero_block_contains_layers=block_contains,
    )
    return lift


def _layer_zero_mask(
    values: NDArray[np.generic],
    physical_axis: int,
    periodic_indices: _HostIntVector,
) -> _HostLayerBoolVector:
    """PRIVATE: Check exact zero across each full transverse cell layer.

    Parameters
    ----------
    values : NDArray[np.generic]
        Real or complex cell values in zyx storage order.
    physical_axis : int
        Selected physical xyz axis.
    periodic_indices : _HostIntVector
        Ordered periodic layer indices.

    Returns
    -------
    zero_mask : _HostLayerBoolVector
        Per-layer componentwise exact-zero predicates.
    """
    storage_axis = 2 - physical_axis
    zero_mask: _HostLayerBoolVector = np.asarray(
        [
            np.all(np.take(values, int(index), axis=storage_axis) == 0.0)
            for index in periodic_indices
        ],
        dtype=np.bool_,
    )
    return zero_mask


def _incident_predicates(
    certificate: GalerkinLocalRepresentedSourceCertificate,
) -> _IncidentEvidence:
    """PRIVATE: Recompute declared exact-shell predicates for stored ``v``.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Prepared represented-source direct certificate.

    Returns
    -------
    predicates : _IncidentEvidence
        Active, declared, exact-disposition, exact-shell, and stored-support
        consistency evidence.
    """
    source = certificate.source
    target = source.target
    incident_field = np.asarray(jax.device_get(source.modes.incident_field))
    active = np.asarray(
        (np.real(incident_field) != 0.0) | (np.imag(incident_field) != 0.0),
        dtype=np.bool_,
    )
    stored_active = np.asarray(jax.device_get(source.modes.active_mask))
    active_consistent = bool(np.array_equal(active, stored_active))
    state_indices = np.asarray(jax.device_get(target.state_indices))
    declared_indices = np.asarray(
        jax.device_get(target.acquisition.incident_indices)
    )
    matches = np.all(
        state_indices[:, None, :] == declared_indices[None, :, :], axis=-1
    )
    declared = np.asarray(np.any(matches, axis=1), dtype=np.bool_)
    dispositions = np.asarray(
        jax.device_get(target.acquisition.incident_direction_dispositions)
    )
    exact_rows = dispositions == int(
        GalerkinDirectionDisposition.EXACT_COEFFICIENT
    )
    exact_disposition = np.asarray(
        np.any(matches & exact_rows[None, :], axis=1), dtype=np.bool_
    )
    ledger = target.fixed_linear_error_ledger
    free_lower = np.asarray(
        jax.device_get(ledger.exact_free_diagonal_lower_bounds)
    )
    free_upper = np.asarray(
        jax.device_get(ledger.exact_free_diagonal_upper_bounds)
    )
    exact_shell = np.asarray(
        (free_lower == 0.0) & (free_upper == 0.0), dtype=np.bool_
    )
    predicates: _IncidentEvidence = (
        active,
        declared,
        exact_disposition,
        exact_shell,
        active_consistent,
    )
    return predicates


def _vacuum_reference_eligible(
    reference_value: float,
    reference_semantics: str,
) -> bool:
    """PRIVATE: Check the sole exact local vacuum-reference declaration.

    Parameters
    ----------
    reference_value : float
        Exact stored additive potential reference in volts.
    reference_semantics : str
        Stored physical-reference declaration.

    Returns
    -------
    eligible : bool
        Whether signed-zero storage names the exact SC.2/SC.8 vacuum.
    """
    eligible: bool = (
        reference_value == 0.0
        and reference_semantics
        == GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value
    )
    return eligible


def _failure_mask(  # noqa: PLR0913
    *,
    vacuum_reference: bool,
    potential_zero: bool,
    cap_block_contains: bool,
    cap_layers_zero: bool,
    additional_zero: bool,
    active_consistent: bool,
    active: _HostStateBoolVector,
    declared: _HostStateBoolVector,
    exact_disposition: _HostStateBoolVector,
    exact_shell: _HostStateBoolVector,
    projection_match: bool,
) -> GalerkinLocalZeroSlabFailure:
    """PRIVATE: Compose all independent zero-slab failure bits.

    Parameters
    ----------
    vacuum_reference : bool
        Whether stored zero has the canonical vacuum meaning.
    potential_zero : bool
        Whether all selected potential layers are exactly zero.
    cap_block_contains : bool
        Whether one L4 zero-block lift contains the selected layer union.
    cap_layers_zero : bool
        Whether all selected exact CAP profile layers are zero.
    additional_zero : bool
        Whether the local additional source is zero on selected layers.
    active_consistent : bool
        Whether stored and incident-field-derived active masks agree.
    active : _HostStateBoolVector
        Exact active incident mask.
    declared : _HostStateBoolVector
        Declared incident membership mask.
    exact_disposition : _HostStateBoolVector
        Exact-coefficient disposition mask.
    exact_shell : _HostStateBoolVector
        Singleton-zero exact free-diagonal mask.
    projection_match : bool
        Whether the full represented-source certificate is finite.

    Returns
    -------
    failure : GalerkinLocalZeroSlabFailure
        Bitwise fail-closed outcome.
    """
    failure = GalerkinLocalZeroSlabFailure.NONE
    if not vacuum_reference:
        failure |= GalerkinLocalZeroSlabFailure.VACUUM_REFERENCE_UNDECLARED
    if not potential_zero:
        failure |= GalerkinLocalZeroSlabFailure.POTENTIAL_NONZERO
    if not cap_block_contains:
        failure |= GalerkinLocalZeroSlabFailure.CAP_ZERO_BLOCK_MISMATCH
    if not cap_layers_zero:
        failure |= GalerkinLocalZeroSlabFailure.CAP_NONZERO
    if not additional_zero:
        failure |= GalerkinLocalZeroSlabFailure.ADDITIONAL_SOURCE_NONZERO
    if not active_consistent:
        failure |= GalerkinLocalZeroSlabFailure.INCIDENT_ACTIVE_MASK_MISMATCH
    if np.any(active & ~declared):
        failure |= GalerkinLocalZeroSlabFailure.UNDECLARED_INCIDENT_MODE
    if np.any(active & ~exact_disposition):
        failure |= GalerkinLocalZeroSlabFailure.NONEXACT_INCIDENT_DISPOSITION
    if np.any(active & ~exact_shell):
        failure |= GalerkinLocalZeroSlabFailure.ACTIVE_INCIDENT_OFF_SHELL
    if not projection_match:
        failure |= (
            GalerkinLocalZeroSlabFailure.REPRESENTED_SOURCE_NONCERTIFICATE
        )
    return failure


def _eligibility_predicates(  # noqa: PLR0913
    *,
    cap_block_contains: bool,
    active_consistent: bool,
    vacuum_reference: bool,
    potential_zero: bool,
    cap_layers_zero: bool,
    incident_free_zero: bool,
    additional_zero: bool,
    projection_match: bool,
) -> _EligibilityPredicates:
    """PRIVATE: Compose exact-spatial and projection facts independently.

    Parameters
    ----------
    cap_block_contains : bool
        Whether one exact L4 zero-block lift contains all selected layers.
    active_consistent : bool
        Whether stored and incident-field-derived active masks agree.
    vacuum_reference : bool
        Whether stored zero has the canonical vacuum meaning and value.
    potential_zero : bool
        Whether the exact potential is zero on every selected layer.
    cap_layers_zero : bool
        Whether the exact CAP profile is zero on every selected layer.
    incident_free_zero : bool
        Whether every active incident mode is declared and exactly on shell.
    additional_zero : bool
        Whether the exact local additional source is zero on selected layers.
    projection_match : bool
        Whether the direct represented-source certificate is finite.

    Returns
    -------
    predicates : _EligibilityPredicates
        Independent exact-spatial, projection, and conjunction predicates.
    """
    cap_zero = cap_block_contains and cap_layers_zero
    checked_incident_free_zero = active_consistent and incident_free_zero
    spatial_source_zero = (
        cap_zero and checked_incident_free_zero and additional_zero
    )
    exact_zero_slab = (
        vacuum_reference and potential_zero and spatial_source_zero
    )
    terminal_zero_slab = exact_zero_slab and projection_match
    predicates: _EligibilityPredicates = _EligibilityPredicates(
        cap_zero_block_contains_layers=cap_block_contains,
        incident_active_mask_consistent=active_consistent,
        vacuum_reference_eligible=vacuum_reference,
        potential_zero_eligible=potential_zero,
        cap_zero_eligible=cap_zero,
        incident_free_zero_eligible=checked_incident_free_zero,
        additional_source_zero_eligible=additional_zero,
        exact_spatial_source_zero_eligible=spatial_source_zero,
        exact_zero_slab_eligible=exact_zero_slab,
        projection_match_eligible=projection_match,
        terminal_zero_slab_eligible=terminal_zero_slab,
    )
    return predicates


def _prepare_represented_source_certificate(
    certificate: GalerkinLocalRepresentedSourceCertificate,
) -> GalerkinLocalRepresentedSourceCertificate:
    """PRIVATE: Call the frozen represented-source replay seam.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Public represented-source certificate to replay.

    Returns
    -------
    prepared : GalerkinLocalRepresentedSourceCertificate
        Fresh canonical represented-source certificate.
    """
    prepared: GalerkinLocalRepresentedSourceCertificate = (
        prepare_local_represented_source_certificate(certificate)
    )
    return prepared


def _slab_digest(
    certificate: GalerkinLocalRepresentedSourceCertificate,
    lower: np.float64,
    upper: np.float64,
    lift: _LayerLift,
    exact_fields: Tuple[Tuple[str, str], ...],
    terminal_axis: int,
) -> str:
    """PRIVATE: Digest exact slab identity apart from proof evidence.

    Parameters
    ----------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Prepared represented-source certificate.
    lower : np.float64
        Exact stored inner Cauchy-plane coordinate.
    upper : np.float64
        Exact stored outer Cauchy-plane coordinate.
    lift : _LayerLift
        Canonical exact guarded layer lift.
    exact_fields : Tuple[Tuple[str, str], ...]
        Exact lower, upper, width, and union-face rational transcripts.
    terminal_axis : int
        Target-owned physical terminal axis.

    Returns
    -------
    slab_digest : str
        Operator/source/slab identity digest.
    """
    source = certificate.source
    slab_digest: str = sha256(
        {
            "domain": _SLAB_DIGEST_DOMAIN,
            "target_digest": source.target.target_digest,
            "represented_source_digest": source.source_digest,
            "additional_source_digest": source.additional_source_digest,
            "terminal_axis": terminal_axis,
            "slab_lower_coordinate": stored_value_payload(lower),
            "slab_upper_coordinate": stored_value_payload(upper),
            "unwrapped_layer_start": str(lift.start),
            "unwrapped_layer_stop": str(lift.stop),
            "cap_zero_block_lift": str(lift.cap_zero_block_lift),
            "periodic_layer_indices": stored_value_payload(
                lift.periodic_indices
            ),
            "exact_rational_transcript": exact_fields,
            "vacuum_reference": (
                GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value
            ),
            "exact_target": _EXACT_TARGET,
            "geometry_convention": _GEOMETRY_CONVENTION,
            "source_zero_route": _SOURCE_ZERO_ROUTE,
            "no_cancellation_scope": _NO_CANCELLATION_SCOPE,
        }
    )
    return slab_digest


def _certificate_digest(
    represented: GalerkinLocalRepresentedSourceCertificate,
    slab_digest: str,
    layer_masks: _LayerMasks,
    incident_evidence: _IncidentEvidence,
    predicates: Tuple[bool, ...],
    failure: GalerkinLocalZeroSlabFailure,
) -> str:
    """PRIVATE: Digest all replayed zero-slab evidence.

    Parameters
    ----------
    represented : GalerkinLocalRepresentedSourceCertificate
        Prepared represented-source direct certificate.
    slab_digest : str
        Exact slab identity digest.
    layer_masks : _LayerMasks
        Potential, CAP, and additional-source zero masks.
    incident_evidence : _IncidentEvidence
        Active, declared, disposition, shell, and support-consistency evidence.
    predicates : Tuple[bool, ...]
        Ordered scalar exact/projection/final predicates.
    failure : GalerkinLocalZeroSlabFailure
        Bitwise fail-closed outcome.

    Returns
    -------
    certificate_digest : str
        Full zero-slab evidence digest.
    """
    source = represented.source
    certificate_digest: str = sha256(
        {
            "domain": _CERTIFICATE_DIGEST_DOMAIN,
            "slab_digest": slab_digest,
            "target_digest": source.target.target_digest,
            "parent_target_evidence_digest": (
                source.target.manifest_evidence_digest
            ),
            "represented_source_digest": source.source_digest,
            "parent_source_evidence_digest": source.source_evidence_digest,
            "parent_represented_certificate_digest": (
                represented.certificate_digest
            ),
            "full_prepared_represented_certificate": stored_value_payload(
                represented
            ),
            "layer_zero_masks": stored_value_payload(layer_masks),
            "incident_evidence": stored_value_payload(incident_evidence),
            "predicates": stored_value_payload(predicates),
            "failure_mask": int(failure),
            "completion_scope": _COMPLETION_SCOPE,
        }
    )
    return certificate_digest


def _certify_prepared(  # noqa: PLR0915
    represented: GalerkinLocalRepresentedSourceCertificate,
    slab_lower_coordinate: object,
    slab_upper_coordinate: object,
) -> GalerkinLocalZeroSlabCertificate:
    """PRIVATE: Certify one slab from a replayed represented source.

    Parameters
    ----------
    represented : GalerkinLocalRepresentedSourceCertificate
        Fully replayed represented-source direct certificate.
    slab_lower_coordinate : object
        Exact float64 inner Cauchy-plane coordinate.
    slab_upper_coordinate : object
        Exact float64 outer Cauchy-plane coordinate.

    Returns
    -------
    certificate : GalerkinLocalZeroSlabCertificate
        Canonical zero-slab certificate or typed noncertificate.

    Raises
    ------
    ValueError
        If coordinate or canonical lifted geometry is structurally invalid.
    """
    lower = _host_binary64_coordinate(
        slab_lower_coordinate, "slab_lower_coordinate"
    )
    upper = _host_binary64_coordinate(
        slab_upper_coordinate, "slab_upper_coordinate"
    )
    exact_lower = Fraction.from_float(float(lower))
    exact_upper = Fraction.from_float(float(upper))
    exact_width = exact_upper - exact_lower
    source = represented.source
    target = source.target
    potential = target.local_potential
    absorber = target.cap_floor_proof.coefficient_certificate.absorber
    terminal_axis = target.acquisition.terminal_axis
    if terminal_axis != absorber.terminal_axis:
        raise ValueError("target and absorber terminal axes disagree")
    grid_shape_xyz = tuple(reversed(potential.cell_values.shape))
    layer_count = grid_shape_xyz[terminal_axis]
    box_length_value = _host_binary64_coordinate(
        potential.box_size[terminal_axis],
        "terminal box length",
    )
    origin_value = _host_binary64_coordinate(
        potential.cell_center_origin[terminal_axis],
        "terminal cell-center origin",
    )
    box_length = Fraction.from_float(float(box_length_value))
    origin = Fraction.from_float(float(origin_value))
    lift = _derive_layer_lift(
        exact_lower,
        exact_upper,
        origin,
        box_length,
        layer_count,
        absorber.zero_start,
        absorber.zero_count,
    )
    try:
        width_display = np.float64(float(exact_width))
    except OverflowError as error:
        raise ValueError(
            "exact slab width must fit finite float64 display"
        ) from error
    if (
        not np.isfinite(width_display)
        or width_display < np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "exact slab width must have finite normal float64 display"
        )

    potential_values = np.asarray(jax.device_get(potential.cell_values))
    potential_zero = _layer_zero_mask(
        potential_values, terminal_axis, lift.periodic_indices
    )
    cap_values = np.asarray(jax.device_get(absorber.layer_values))
    cap_zero = np.asarray(
        cap_values[lift.periodic_indices] == 0.0, dtype=np.bool_
    )
    additional = source.additional_source_certificate.source
    if additional.route is GalerkinLocalAdditionalSourceRoute.ZERO:
        additional_zero = np.ones_like(potential_zero, dtype=np.bool_)
    elif additional.route is GalerkinLocalAdditionalSourceRoute.LOCAL_CELL:
        additional_values = np.asarray(
            jax.device_get(additional.source_cell_values)
        )
        additional_zero = _layer_zero_mask(
            additional_values, terminal_axis, lift.periodic_indices
        )
    else:
        raise ValueError("local additional-source route is noncanonical")

    incident_evidence = _incident_predicates(represented)
    active, declared, exact_disposition, exact_shell, active_consistent = (
        incident_evidence
    )
    vacuum_reference = _vacuum_reference_eligible(
        potential.reference_value,
        potential.reference_semantics,
    )
    potential_eligible = bool(np.all(potential_zero))
    cap_layers_eligible = bool(np.all(cap_zero))
    additional_eligible = bool(np.all(additional_zero))
    incident_eligible = active_consistent and bool(
        np.all((~active) | (declared & exact_disposition & exact_shell))
    )
    projection_match = bool(represented.finite_certificate)
    eligibility = _eligibility_predicates(
        cap_block_contains=lift.cap_zero_block_contains_layers,
        active_consistent=active_consistent,
        vacuum_reference=vacuum_reference,
        potential_zero=potential_eligible,
        cap_layers_zero=cap_layers_eligible,
        incident_free_zero=incident_eligible,
        additional_zero=additional_eligible,
        projection_match=projection_match,
    )
    failure = _failure_mask(
        vacuum_reference=vacuum_reference,
        potential_zero=potential_eligible,
        cap_block_contains=lift.cap_zero_block_contains_layers,
        cap_layers_zero=cap_layers_eligible,
        additional_zero=additional_eligible,
        active_consistent=active_consistent,
        active=active,
        declared=declared,
        exact_disposition=exact_disposition,
        exact_shell=exact_shell,
        projection_match=projection_match,
    )
    exact_fields: Tuple[Tuple[str, str], ...] = (
        _fraction_fields(exact_lower),
        _fraction_fields(exact_upper),
        _fraction_fields(exact_width),
        _fraction_fields(lift.union_lower),
        _fraction_fields(lift.union_upper),
    )
    slab_digest = _slab_digest(
        represented,
        lower,
        upper,
        lift,
        exact_fields,
        terminal_axis,
    )
    predicates: Tuple[bool, ...] = tuple(eligibility)
    layer_masks: _LayerMasks = (
        potential_zero,
        cap_zero,
        additional_zero,
    )
    certificate_digest = _certificate_digest(
        represented,
        slab_digest,
        layer_masks,
        incident_evidence,
        predicates,
        failure,
    )
    stopped = jax.tree.map(
        jax.lax.stop_gradient,
        (
            jnp.asarray(lower),
            jnp.asarray(upper),
            jnp.asarray(width_display),
            jnp.asarray(lift.periodic_indices),
            *(jnp.asarray(value) for value in layer_masks),
            *(jnp.asarray(value) for value in incident_evidence[:4]),
            *(jnp.asarray(value) for value in predicates),
            jnp.asarray(int(failure), dtype=jnp.int64),
        ),
    )
    certificate: GalerkinLocalZeroSlabCertificate
    certificate = _make_local_zero_slab_certificate(
        represented,
        stopped[0],
        stopped[1],
        stopped[2],
        stopped[3],
        stopped[4],
        stopped[5],
        stopped[6],
        stopped[7],
        stopped[8],
        stopped[9],
        stopped[10],
        stopped[11],
        stopped[12],
        stopped[13],
        stopped[14],
        stopped[15],
        stopped[16],
        stopped[17],
        stopped[18],
        stopped[19],
        stopped[20],
        stopped[21],
        stopped[22],
        terminal_axis=terminal_axis,
        unwrapped_layer_start=str(lift.start),
        unwrapped_layer_stop=str(lift.stop),
        cap_zero_block_lift=str(lift.cap_zero_block_lift),
        slab_lower_numerator=exact_fields[0][0],
        slab_lower_denominator=exact_fields[0][1],
        slab_upper_numerator=exact_fields[1][0],
        slab_upper_denominator=exact_fields[1][1],
        slab_width_numerator=exact_fields[2][0],
        slab_width_denominator=exact_fields[2][1],
        layer_union_lower_numerator=exact_fields[3][0],
        layer_union_lower_denominator=exact_fields[3][1],
        layer_union_upper_numerator=exact_fields[4][0],
        layer_union_upper_denominator=exact_fields[4][1],
        vacuum_reference=GalerkinLocalVacuumReference.VACUUM_K0_CARRIER,
        exact_target=_EXACT_TARGET,
        geometry_convention=_GEOMETRY_CONVENTION,
        source_zero_route=_SOURCE_ZERO_ROUTE,
        no_cancellation_scope=_NO_CANCELLATION_SCOPE,
        completion_scope=_COMPLETION_SCOPE,
        target_digest=target.target_digest,
        parent_target_evidence_digest=target.manifest_evidence_digest,
        represented_source_digest=source.source_digest,
        parent_source_evidence_digest=source.source_evidence_digest,
        parent_represented_certificate_digest=represented.certificate_digest,
        slab_digest=slab_digest,
        certificate_digest=certificate_digest,
    )
    return certificate


@jaxtyped(typechecker=beartype)
def certify_local_zero_slab(
    represented_source_certificate: GalerkinLocalRepresentedSourceCertificate,
    *,
    slab_lower_coordinate: float | Float[Array, ""],
    slab_upper_coordinate: float | Float[Array, ""],
) -> GalerkinLocalZeroSlabCertificate:
    """Replay represented-source evidence and certify exact LVT.21--LVT.22.

    :see: :func:`~.test_local_zero_slab.\
test_exact_guarded_layer_lift_handles_wrapping_and_faces`

    Parameters
    ----------
    represented_source_certificate : GalerkinLocalRepresentedSourceCertificate
        Public full represented-source direct certificate to authenticate.
    slab_lower_coordinate : float | Float[Array, ""]
        Exact binary64 inner Cauchy-plane coordinate in Angstroms.
    slab_upper_coordinate : float | Float[Array, ""]
        Exact binary64 outer coordinate on the chosen unwrapped lift.

    Returns
    -------
    certificate : GalerkinLocalZeroSlabCertificate
        Exact spatial/projection predicates and typed failure mask.

    Raises
    ------
    TypeError
        If the represented-source carrier has the wrong type.
    ValueError
        If parent replay or exact lifted geometry is structurally invalid.
    """
    if not isinstance(
        represented_source_certificate,
        GalerkinLocalRepresentedSourceCertificate,
    ):
        raise TypeError(
            "represented_source_certificate must be "
            "GalerkinLocalRepresentedSourceCertificate"
        )
    _assert_concrete(represented_source_certificate)
    prepared = _prepare_represented_source_certificate(
        represented_source_certificate
    )
    certificate: GalerkinLocalZeroSlabCertificate = _certify_prepared(
        prepared,
        slab_lower_coordinate,
        slab_upper_coordinate,
    )
    return certificate


def prepare_local_zero_slab_certificate(
    certificate: GalerkinLocalZeroSlabCertificate,
) -> GalerkinLocalZeroSlabCertificate:
    """Replay every nested carrier, exact predicate, transcript, and digest.

    :see: :func:`~.test_local_zero_slab.\
test_zero_slab_public_boundary_consumes_only_represented_certificate`

    Parameters
    ----------
    certificate : GalerkinLocalZeroSlabCertificate
        Public zero-slab certificate to authenticate in full.

    Returns
    -------
    canonical : GalerkinLocalZeroSlabCertificate
        Fresh certificate reconstructed from authenticated primitive inputs.

    Raises
    ------
    TypeError
        If ``certificate`` has the wrong carrier type.
    ValueError
        If complete parent/predicate/digest replay differs from submission.
    """
    if not isinstance(certificate, GalerkinLocalZeroSlabCertificate):
        raise TypeError("certificate must be GalerkinLocalZeroSlabCertificate")
    _assert_concrete(certificate)
    prepared_parent = _prepare_represented_source_certificate(
        certificate.represented_source_certificate
    )
    canonical: GalerkinLocalZeroSlabCertificate = _certify_prepared(
        prepared_parent,
        certificate.slab_lower_coordinate,
        certificate.slab_upper_coordinate,
    )
    if stored_value_payload(canonical) != stored_value_payload(certificate):
        raise ValueError(
            "zero-slab certificate does not match complete host replay"
        )
    return canonical


__all__: list[str] = [
    "certify_local_zero_slab",
    "prepare_local_zero_slab_certificate",
]
