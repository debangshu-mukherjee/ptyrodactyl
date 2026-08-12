r"""Define exact LVT.21--LVT.22 local zero-slab evidence carriers.

Extended Summary
----------------
This leaf stores one lifted open slab together with separately replayable
vacuum-reference, local-potential, exact-CAP, represented-incident, and local
additional-source predicates.  Exact spatial absence remains independent of
the represented-source direct certificate; terminal readiness requires both.

Routine Listings
----------------
:class:`GalerkinLocalVacuumReference`
    Select the sole canonical local vacuum-reference declaration.
:class:`GalerkinLocalZeroSlabCertificate`
    Store replayable LVT.21--LVT.22 geometry and predicate evidence.
:class:`GalerkinLocalZeroSlabFailure`
    Enumerate simultaneous fail-closed zero-slab predicate reasons.
"""

from __future__ import annotations

from enum import Enum, IntFlag
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Float64, Int, Int64, jaxtyped

from .local_represented_source_types import (
    GalerkinLocalRepresentedSourceCertificate,
)

_SHA256_HEX_LENGTH: int = 64


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise for one structural zero-slab carrier failure.

    Parameters
    ----------
    condition : bool
        Whether the structural failure is present.
    message : str
        Error message for the failed invariant.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _valid_digest(value: str) -> bool:
    """PRIVATE: Check one canonical lowercase SHA-256 text value.

    Parameters
    ----------
    value : str
        Candidate digest text.

    Returns
    -------
    valid : bool
        Whether the value is one canonical lowercase SHA-256 digest.
    """
    valid: bool = (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )
    return valid


def _signed_decimal(value: str, name: str) -> int:
    """PRIVATE: Parse one canonical signed decimal integer.

    Parameters
    ----------
    value : str
        Candidate canonical integer text.
    name : str
        Field name used in a structural failure message.

    Returns
    -------
    integer : int
        Parsed arbitrary-precision integer.

    Raises
    ------
    ValueError
        If the text is not the canonical decimal representation.
    """
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a canonical signed decimal"
        ) from error
    _raise_if(
        str(integer) != value, f"{name} must be a canonical signed decimal"
    )
    return integer


def _positive_decimal(value: str, name: str) -> int:
    """PRIVATE: Parse one canonical positive decimal integer.

    Parameters
    ----------
    value : str
        Candidate canonical positive integer text.
    name : str
        Field name used in a structural failure message.

    Returns
    -------
    integer : int
        Parsed positive arbitrary-precision integer.

    Raises
    ------
    ValueError
        If the text is not canonical or is nonpositive.
    """
    integer = _signed_decimal(value, name)
    _raise_if(integer <= 0, f"{name} must be positive")
    return integer


class GalerkinLocalVacuumReference(str, Enum):
    """Select the sole canonical local vacuum-reference declaration.

    :see: :func:`~.test_local_zero_slab_types.\
test_zero_slab_failure_bits_and_reference_are_disjoint`
    """

    VACUUM_K0_CARRIER = (
        "stored zero is the vacuum value used by exact SC.2 k0 and SC.8 k_i"
    )


class GalerkinLocalZeroSlabFailure(IntFlag):
    """Enumerate simultaneous fail-closed zero-slab predicate reasons.

    :see: :func:`~.test_local_zero_slab_types.\
test_zero_slab_failure_bits_and_reference_are_disjoint`
    """

    NONE = 0
    VACUUM_REFERENCE_UNDECLARED = 1 << 0
    POTENTIAL_NONZERO = 1 << 1
    CAP_ZERO_BLOCK_MISMATCH = 1 << 2
    CAP_NONZERO = 1 << 3
    ADDITIONAL_SOURCE_NONZERO = 1 << 4
    INCIDENT_ACTIVE_MASK_MISMATCH = 1 << 5
    UNDECLARED_INCIDENT_MODE = 1 << 6
    NONEXACT_INCIDENT_DISPOSITION = 1 << 7
    ACTIVE_INCIDENT_OFF_SHELL = 1 << 8
    REPRESENTED_SOURCE_NONCERTIFICATE = 1 << 9


class GalerkinLocalZeroSlabCertificate(eqx.Module):
    r"""Store replayable LVT.21--LVT.22 geometry and predicate evidence.

    :see: :func:`~.test_local_zero_slab_types.\
test_zero_slab_carrier_separates_exact_and_projection_predicates`

    The coordinate and layer-union rational strings are authoritative exact
    transcripts. ``slab_width`` is display evidence only.  Signed-zero input
    bytes remain digest-distinct although every exact zero predicate is
    signed-zero tolerant.
    """

    represented_source_certificate: GalerkinLocalRepresentedSourceCertificate
    slab_lower_coordinate: Float64[Array, ""]
    slab_upper_coordinate: Float64[Array, ""]
    slab_width: Float64[Array, ""]
    periodic_layer_indices: Int64[Array, " l"]
    potential_layer_zero_mask: Bool[Array, " l"]
    cap_layer_zero_mask: Bool[Array, " l"]
    additional_source_layer_zero_mask: Bool[Array, " l"]
    incident_active_mask: Bool[Array, " n"]
    incident_declared_mask: Bool[Array, " n"]
    incident_exact_disposition_mask: Bool[Array, " n"]
    incident_exact_shell_mask: Bool[Array, " n"]
    cap_zero_block_contains_layers: Bool[Array, ""]
    incident_active_mask_consistent: Bool[Array, ""]
    vacuum_reference_eligible: Bool[Array, ""]
    potential_zero_eligible: Bool[Array, ""]
    cap_zero_eligible: Bool[Array, ""]
    incident_free_zero_eligible: Bool[Array, ""]
    additional_source_zero_eligible: Bool[Array, ""]
    exact_spatial_source_zero_eligible: Bool[Array, ""]
    exact_zero_slab_eligible: Bool[Array, ""]
    projection_match_eligible: Bool[Array, ""]
    terminal_zero_slab_eligible: Bool[Array, ""]
    failure_mask: Int64[Array, ""]
    terminal_axis: int = eqx.field(static=True)
    unwrapped_layer_start: str = eqx.field(static=True)
    unwrapped_layer_stop: str = eqx.field(static=True)
    cap_zero_block_lift: str = eqx.field(static=True)
    slab_lower_numerator: str = eqx.field(static=True)
    slab_lower_denominator: str = eqx.field(static=True)
    slab_upper_numerator: str = eqx.field(static=True)
    slab_upper_denominator: str = eqx.field(static=True)
    slab_width_numerator: str = eqx.field(static=True)
    slab_width_denominator: str = eqx.field(static=True)
    layer_union_lower_numerator: str = eqx.field(static=True)
    layer_union_lower_denominator: str = eqx.field(static=True)
    layer_union_upper_numerator: str = eqx.field(static=True)
    layer_union_upper_denominator: str = eqx.field(static=True)
    vacuum_reference: GalerkinLocalVacuumReference = eqx.field(static=True)
    exact_target: str = eqx.field(static=True)
    geometry_convention: str = eqx.field(static=True)
    source_zero_route: str = eqx.field(static=True)
    no_cancellation_scope: str = eqx.field(static=True)
    completion_scope: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    parent_target_evidence_digest: str = eqx.field(static=True)
    represented_source_digest: str = eqx.field(static=True)
    parent_source_evidence_digest: str = eqx.field(static=True)
    parent_represented_certificate_digest: str = eqx.field(static=True)
    slab_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def _make_local_zero_slab_certificate(  # noqa: PLR0912, PLR0913, PLR0915
    represented_source_certificate: GalerkinLocalRepresentedSourceCertificate,
    slab_lower_coordinate: Float[Array, ""],
    slab_upper_coordinate: Float[Array, ""],
    slab_width: Float[Array, ""],
    periodic_layer_indices: Int[Array, " l"],
    potential_layer_zero_mask: Bool[Array, " l"],
    cap_layer_zero_mask: Bool[Array, " l"],
    additional_source_layer_zero_mask: Bool[Array, " l"],
    incident_active_mask: Bool[Array, " n"],
    incident_declared_mask: Bool[Array, " n"],
    incident_exact_disposition_mask: Bool[Array, " n"],
    incident_exact_shell_mask: Bool[Array, " n"],
    cap_zero_block_contains_layers: Bool[Array, ""],
    incident_active_mask_consistent: Bool[Array, ""],
    vacuum_reference_eligible: Bool[Array, ""],
    potential_zero_eligible: Bool[Array, ""],
    cap_zero_eligible: Bool[Array, ""],
    incident_free_zero_eligible: Bool[Array, ""],
    additional_source_zero_eligible: Bool[Array, ""],
    exact_spatial_source_zero_eligible: Bool[Array, ""],
    exact_zero_slab_eligible: Bool[Array, ""],
    projection_match_eligible: Bool[Array, ""],
    terminal_zero_slab_eligible: Bool[Array, ""],
    failure_mask: Int[Array, ""],
    *,
    terminal_axis: int,
    unwrapped_layer_start: str,
    unwrapped_layer_stop: str,
    cap_zero_block_lift: str,
    slab_lower_numerator: str,
    slab_lower_denominator: str,
    slab_upper_numerator: str,
    slab_upper_denominator: str,
    slab_width_numerator: str,
    slab_width_denominator: str,
    layer_union_lower_numerator: str,
    layer_union_lower_denominator: str,
    layer_union_upper_numerator: str,
    layer_union_upper_denominator: str,
    vacuum_reference: GalerkinLocalVacuumReference,
    exact_target: str,
    geometry_convention: str,
    source_zero_route: str,
    no_cancellation_scope: str,
    completion_scope: str,
    target_digest: str,
    parent_target_evidence_digest: str,
    represented_source_digest: str,
    parent_source_evidence_digest: str,
    parent_represented_certificate_digest: str,
    slab_digest: str,
    certificate_digest: str,
) -> GalerkinLocalZeroSlabCertificate:
    """PRIVATE: Validate one exact zero-slab certificate carrier.

    Parameters
    ----------
    represented_source_certificate : GalerkinLocalRepresentedSourceCertificate
        Fully replayed represented-source direct certificate.
    slab_lower_coordinate : Float[Array, ""]
        Exact stored binary64 inner Cauchy-plane coordinate.
    slab_upper_coordinate : Float[Array, ""]
        Exact stored binary64 outer Cauchy-plane coordinate.
    slab_width : Float[Array, ""]
        Binary64 display value for the exact rational slab width.
    periodic_layer_indices : Int[Array, " l"]
        Ordered periodic layer indices covering and guarding both planes.
    potential_layer_zero_mask : Bool[Array, " l"]
        Per-layer exact full-transverse potential-zero predicates.
    cap_layer_zero_mask : Bool[Array, " l"]
        Per-layer exact axial-profile-zero predicates.
    additional_source_layer_zero_mask : Bool[Array, " l"]
        Per-layer exact local additional-source-zero predicates.
    incident_active_mask : Bool[Array, " n"]
        Exact nonzero incident-field coefficient mask.
    incident_declared_mask : Bool[Array, " n"]
        Replayed declared-incident membership mask.
    incident_exact_disposition_mask : Bool[Array, " n"]
        Replayed exact-coefficient disposition mask.
    incident_exact_shell_mask : Bool[Array, " n"]
        Singleton-zero exact free-diagonal mask.
    cap_zero_block_contains_layers : Bool[Array, ""]
        Whether one authenticated CAP zero-block lift contains the layer union.
    incident_active_mask_consistent : Bool[Array, ""]
        Whether stored and incident-field-derived active masks agree.
    vacuum_reference_eligible : Bool[Array, ""]
        Whether stored zero has the canonical vacuum meaning and value.
    potential_zero_eligible : Bool[Array, ""]
        Whether the exact local potential is zero on every selected layer.
    cap_zero_eligible : Bool[Array, ""]
        Whether the exact physical CAP is zero on every selected layer.
    incident_free_zero_eligible : Bool[Array, ""]
        Whether every active incident mode is declared and exactly on shell.
    additional_source_zero_eligible : Bool[Array, ""]
        Whether the local additional-source lift is zero on the slab.
    exact_spatial_source_zero_eligible : Bool[Array, ""]
        Conjunction of the three no-cancellation source-zero facts.
    exact_zero_slab_eligible : Bool[Array, ""]
        Whether every exact LVT.21--LVT.22 spatial predicate holds.
    projection_match_eligible : Bool[Array, ""]
        Whether the represented-source direct certificate is finite.
    terminal_zero_slab_eligible : Bool[Array, ""]
        Conjunction of exact spatial absence and finite projection matching.
    failure_mask : Int[Array, ""]
        Bitwise ``GalerkinLocalZeroSlabFailure`` outcome.
    terminal_axis : int
        Target-owned physical xyz terminal axis.
    unwrapped_layer_start : str
        Canonical exact first unwrapped layer index.
    unwrapped_layer_stop : str
        Canonical exact exclusive unwrapped layer stop.
    cap_zero_block_lift : str
        Canonical exact periodic lift of the L4 zero block.
    slab_lower_numerator : str
        Exact inner-coordinate numerator.
    slab_lower_denominator : str
        Exact positive inner-coordinate denominator.
    slab_upper_numerator : str
        Exact outer-coordinate numerator.
    slab_upper_denominator : str
        Exact positive outer-coordinate denominator.
    slab_width_numerator : str
        Exact positive slab-width numerator.
    slab_width_denominator : str
        Exact positive slab-width denominator.
    layer_union_lower_numerator : str
        Exact selected layer-union lower-face numerator.
    layer_union_lower_denominator : str
        Exact selected layer-union lower-face denominator.
    layer_union_upper_numerator : str
        Exact selected layer-union upper-face numerator.
    layer_union_upper_denominator : str
        Exact selected layer-union upper-face denominator.
    vacuum_reference : GalerkinLocalVacuumReference
        Sole admitted vacuum-reference vocabulary.
    exact_target : str
        Exact LVT.21--LVT.22 target declaration.
    geometry_convention : str
        Exact lifted centered-cell geometry declaration.
    source_zero_route : str
        Separate-factor source-zero route declaration.
    no_cancellation_scope : str
        Explicit absence of a local cancellation oracle.
    completion_scope : str
        Explicit downstream exclusions.
    target_digest : str
        Bound target operator identity digest.
    parent_target_evidence_digest : str
        Bound full target evidence digest.
    represented_source_digest : str
        Bound represented-source identity digest.
    parent_source_evidence_digest : str
        Bound represented-source evidence digest.
    parent_represented_certificate_digest : str
        Bound represented-source direct-certificate digest.
    slab_digest : str
        Exact slab identity digest.
    certificate_digest : str
        Full zero-slab evidence digest.

    Returns
    -------
    certificate : GalerkinLocalZeroSlabCertificate
        Validated zero-slab certificate carrier.

    Raises
    ------
    TypeError
        If a nested carrier or enum has the wrong type.
    ValueError
        If geometry, predicate, transcript, or digest invariants disagree.
    """
    if not isinstance(
        represented_source_certificate,
        GalerkinLocalRepresentedSourceCertificate,
    ):
        raise TypeError(
            "represented_source_certificate must be "
            "GalerkinLocalRepresentedSourceCertificate"
        )
    if not isinstance(vacuum_reference, GalerkinLocalVacuumReference):
        raise TypeError("vacuum_reference has the wrong zero-slab enum")

    source = represented_source_certificate.source
    target = source.target
    absorber = target.cap_floor_proof.coefficient_certificate.absorber
    lower = jnp.asarray(slab_lower_coordinate)
    upper = jnp.asarray(slab_upper_coordinate)
    width_display = jnp.asarray(slab_width)
    indices = jnp.asarray(periodic_layer_indices)
    layer_masks = tuple(
        jnp.asarray(value, dtype=jnp.bool_)
        for value in (
            potential_layer_zero_mask,
            cap_layer_zero_mask,
            additional_source_layer_zero_mask,
        )
    )
    incident_masks = tuple(
        jnp.asarray(value, dtype=jnp.bool_)
        for value in (
            incident_active_mask,
            incident_declared_mask,
            incident_exact_disposition_mask,
            incident_exact_shell_mask,
        )
    )
    scalar_predicates = tuple(
        jnp.asarray(value, dtype=jnp.bool_)
        for value in (
            cap_zero_block_contains_layers,
            incident_active_mask_consistent,
            vacuum_reference_eligible,
            potential_zero_eligible,
            cap_zero_eligible,
            incident_free_zero_eligible,
            additional_source_zero_eligible,
            exact_spatial_source_zero_eligible,
            exact_zero_slab_eligible,
            projection_match_eligible,
            terminal_zero_slab_eligible,
        )
    )
    submitted_failure = jnp.asarray(failure_mask)
    _raise_if(
        lower.dtype != jnp.dtype(jnp.float64)
        or upper.dtype != jnp.dtype(jnp.float64)
        or width_display.dtype != jnp.dtype(jnp.float64),
        "slab coordinates and display width must have exact float64 dtype",
    )
    _raise_if(
        any(value.shape != () for value in (lower, upper, width_display)),
        "slab coordinates and display width must be scalar",
    )
    _raise_if(
        indices.dtype != jnp.dtype(jnp.int64), "layer indices must be int64"
    )
    _raise_if(
        indices.ndim != 1 or indices.shape[0] == 0,
        "layer indices must be nonempty 1D",
    )
    layer_size = indices.shape[0]
    _raise_if(
        any(value.shape != (layer_size,) for value in layer_masks),
        "layer zero masks must match selected layers",
    )
    state_size = target.state_indices.shape[0]
    _raise_if(
        any(value.shape != (state_size,) for value in incident_masks),
        "incident predicate masks must match target I_u",
    )
    _raise_if(
        any(value.shape != () for value in scalar_predicates)
        or submitted_failure.shape != (),
        "zero-slab predicates and failure mask must be scalar",
    )
    _raise_if(
        submitted_failure.dtype != jnp.dtype(jnp.int64),
        "failure_mask must have exact int64 dtype",
    )
    coordinate_values = jnp.stack((lower, upper, width_display))
    _raise_if(
        bool(jnp.any(~jnp.isfinite(coordinate_values)))
        or bool(
            jnp.any(
                (coordinate_values != 0.0)
                & (jnp.abs(coordinate_values) < jnp.finfo(jnp.float64).tiny)
            )
        )
        or bool(width_display < jnp.finfo(jnp.float64).tiny),
        "slab coordinates and width must be finite normal-or-zero values",
    )

    start = _signed_decimal(unwrapped_layer_start, "unwrapped_layer_start")
    stop = _signed_decimal(unwrapped_layer_stop, "unwrapped_layer_stop")
    cap_lift = _signed_decimal(cap_zero_block_lift, "cap_zero_block_lift")
    rational_values = []
    for numerator, denominator, name in (
        (slab_lower_numerator, slab_lower_denominator, "slab_lower"),
        (slab_upper_numerator, slab_upper_denominator, "slab_upper"),
        (slab_width_numerator, slab_width_denominator, "slab_width"),
        (
            layer_union_lower_numerator,
            layer_union_lower_denominator,
            "layer_union_lower",
        ),
        (
            layer_union_upper_numerator,
            layer_union_upper_denominator,
            "layer_union_upper",
        ),
    ):
        rational_values.append(
            (
                _signed_decimal(numerator, f"{name}_numerator"),
                _positive_decimal(denominator, f"{name}_denominator"),
            )
        )
    exact_lower = Fraction(*rational_values[0])
    exact_upper = Fraction(*rational_values[1])
    exact_width = Fraction(*rational_values[2])
    union_lower = Fraction(*rational_values[3])
    union_upper = Fraction(*rational_values[4])
    _raise_if(
        exact_lower != Fraction.from_float(float(lower))
        or exact_upper != Fraction.from_float(float(upper)),
        "exact slab coordinate transcript does not match stored binary64",
    )
    _raise_if(
        not union_lower < exact_lower < exact_upper < union_upper,
        "Cauchy planes must lie strictly inside the selected layer union",
    )
    _raise_if(
        exact_width != exact_upper - exact_lower or exact_width <= 0,
        "exact slab width transcript is inconsistent",
    )
    _raise_if(
        float(width_display) != float(exact_width),
        "slab width display does not match the exact rational transcript",
    )
    _raise_if(stop - start != layer_size, "layer transcript length mismatch")
    grid_shape_xyz = tuple(reversed(target.local_potential.cell_values.shape))
    _raise_if(
        isinstance(terminal_axis, bool)
        or not isinstance(terminal_axis, int)
        or terminal_axis not in range(3),
        "terminal_axis must be 0, 1, or 2",
    )
    _raise_if(
        terminal_axis != target.acquisition.terminal_axis
        or terminal_axis != absorber.terminal_axis,
        "zero-slab axis must match target and absorber terminal axes",
    )
    axis_size = grid_shape_xyz[terminal_axis]
    _raise_if(
        layer_size > axis_size,
        "selected layer union may not repeat a periodic layer",
    )
    expected_indices = jnp.asarray(
        [(start + offset) % axis_size for offset in range(layer_size)],
        dtype=jnp.int64,
    )
    _raise_if(
        bool(jnp.any(indices != expected_indices)),
        "periodic layer indices do not match the unwrapped transcript",
    )
    box_length_array = jnp.asarray(
        target.local_potential.box_size[terminal_axis]
    )
    origin_array = jnp.asarray(
        target.local_potential.cell_center_origin[terminal_axis]
    )
    _raise_if(
        box_length_array.dtype != jnp.dtype(jnp.float64)
        or origin_array.dtype != jnp.dtype(jnp.float64)
        or box_length_array.shape != ()
        or origin_array.shape != ()
        or not bool(jnp.isfinite(box_length_array))
        or not bool(jnp.isfinite(origin_array))
        or not bool(box_length_array > 0.0),
        "nested terminal geometry must use finite exact float64 scalars",
    )
    exact_box_length = Fraction.from_float(float(box_length_array))
    exact_origin = Fraction.from_float(float(origin_array))
    exact_delta = exact_box_length / axis_size
    exact_first_face = exact_origin - exact_delta / 2
    _raise_if(
        union_lower != exact_first_face + start * exact_delta
        or union_upper != exact_first_face + stop * exact_delta,
        "layer-union transcript disagrees with nested exact cell geometry",
    )
    expected_cap_lift = (start - absorber.zero_start) // axis_size
    _raise_if(
        cap_lift != expected_cap_lift,
        "CAP zero-block lift is not the canonical integer lift",
    )
    zero_start = absorber.zero_start + cap_lift * axis_size
    expected_block_contains = (
        zero_start <= start and stop <= zero_start + absorber.zero_count
    )

    active, declared, exact_disposition, exact_shell = incident_masks
    expected_active_consistent = bool(
        jnp.all(active == source.modes.active_mask)
    )
    expected_vacuum = (
        target.local_potential.reference_value == 0.0
        and target.local_potential.reference_semantics
        == vacuum_reference.value
    )
    expected_potential = bool(jnp.all(layer_masks[0]))
    expected_cap = expected_block_contains and bool(jnp.all(layer_masks[1]))
    expected_additional = bool(jnp.all(layer_masks[2]))
    expected_incident = expected_active_consistent and bool(
        jnp.all((~active) | (declared & exact_disposition & exact_shell))
    )
    expected_spatial_source = (
        expected_cap and expected_additional and expected_incident
    )
    expected_exact_slab = (
        expected_vacuum and expected_potential and expected_spatial_source
    )
    expected_projection = bool(
        represented_source_certificate.finite_certificate
    )
    expected_terminal = expected_exact_slab and expected_projection
    expected_scalars = (
        expected_block_contains,
        expected_active_consistent,
        expected_vacuum,
        expected_potential,
        expected_cap,
        expected_incident,
        expected_additional,
        expected_spatial_source,
        expected_exact_slab,
        expected_projection,
        expected_terminal,
    )
    _raise_if(
        any(
            bool(value) != expected
            for value, expected in zip(
                scalar_predicates, expected_scalars, strict=True
            )
        ),
        "stored zero-slab predicates disagree with their exact masks",
    )

    expected_failure = GalerkinLocalZeroSlabFailure.NONE
    if not expected_vacuum:
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.VACUUM_REFERENCE_UNDECLARED
        )
    if not expected_potential:
        expected_failure |= GalerkinLocalZeroSlabFailure.POTENTIAL_NONZERO
    if not expected_block_contains:
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.CAP_ZERO_BLOCK_MISMATCH
        )
    if not bool(jnp.all(layer_masks[1])):
        expected_failure |= GalerkinLocalZeroSlabFailure.CAP_NONZERO
    if not expected_additional:
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.ADDITIONAL_SOURCE_NONZERO
        )
    if not expected_active_consistent:
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.INCIDENT_ACTIVE_MASK_MISMATCH
        )
    if bool(jnp.any(active & ~declared)):
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.UNDECLARED_INCIDENT_MODE
        )
    if bool(jnp.any(active & ~exact_disposition)):
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.NONEXACT_INCIDENT_DISPOSITION
        )
    if bool(jnp.any(active & ~exact_shell)):
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.ACTIVE_INCIDENT_OFF_SHELL
        )
    if not expected_projection:
        expected_failure |= (
            GalerkinLocalZeroSlabFailure.REPRESENTED_SOURCE_NONCERTIFICATE
        )
    _raise_if(
        int(submitted_failure) != int(expected_failure),
        "failure_mask disagrees with exact zero-slab predicates",
    )

    for declaration, name in (
        (exact_target, "exact_target"),
        (geometry_convention, "geometry_convention"),
        (source_zero_route, "source_zero_route"),
        (no_cancellation_scope, "no_cancellation_scope"),
        (completion_scope, "completion_scope"),
    ):
        _raise_if(not declaration.strip(), f"{name} must be nonempty")
    for digest, name in (
        (target_digest, "target_digest"),
        (parent_target_evidence_digest, "parent_target_evidence_digest"),
        (represented_source_digest, "represented_source_digest"),
        (parent_source_evidence_digest, "parent_source_evidence_digest"),
        (
            parent_represented_certificate_digest,
            "parent_represented_certificate_digest",
        ),
        (slab_digest, "slab_digest"),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(not _valid_digest(digest), f"{name} must be SHA-256")
    _raise_if(target_digest != target.target_digest, "target digest mismatch")
    _raise_if(
        parent_target_evidence_digest != target.manifest_evidence_digest,
        "parent target evidence digest mismatch",
    )
    _raise_if(
        represented_source_digest != source.source_digest,
        "represented source digest mismatch",
    )
    _raise_if(
        parent_source_evidence_digest != source.source_evidence_digest,
        "parent source evidence digest mismatch",
    )
    _raise_if(
        parent_represented_certificate_digest
        != represented_source_certificate.certificate_digest,
        "parent represented certificate digest mismatch",
    )

    certificate = GalerkinLocalZeroSlabCertificate(
        represented_source_certificate=represented_source_certificate,
        slab_lower_coordinate=lower,
        slab_upper_coordinate=upper,
        slab_width=width_display,
        periodic_layer_indices=indices,
        potential_layer_zero_mask=layer_masks[0],
        cap_layer_zero_mask=layer_masks[1],
        additional_source_layer_zero_mask=layer_masks[2],
        incident_active_mask=active,
        incident_declared_mask=declared,
        incident_exact_disposition_mask=exact_disposition,
        incident_exact_shell_mask=exact_shell,
        cap_zero_block_contains_layers=scalar_predicates[0],
        incident_active_mask_consistent=scalar_predicates[1],
        vacuum_reference_eligible=scalar_predicates[2],
        potential_zero_eligible=scalar_predicates[3],
        cap_zero_eligible=scalar_predicates[4],
        incident_free_zero_eligible=scalar_predicates[5],
        additional_source_zero_eligible=scalar_predicates[6],
        exact_spatial_source_zero_eligible=scalar_predicates[7],
        exact_zero_slab_eligible=scalar_predicates[8],
        projection_match_eligible=scalar_predicates[9],
        terminal_zero_slab_eligible=scalar_predicates[10],
        failure_mask=submitted_failure,
        terminal_axis=terminal_axis,
        unwrapped_layer_start=unwrapped_layer_start,
        unwrapped_layer_stop=unwrapped_layer_stop,
        cap_zero_block_lift=cap_zero_block_lift,
        slab_lower_numerator=slab_lower_numerator,
        slab_lower_denominator=slab_lower_denominator,
        slab_upper_numerator=slab_upper_numerator,
        slab_upper_denominator=slab_upper_denominator,
        slab_width_numerator=slab_width_numerator,
        slab_width_denominator=slab_width_denominator,
        layer_union_lower_numerator=layer_union_lower_numerator,
        layer_union_lower_denominator=layer_union_lower_denominator,
        layer_union_upper_numerator=layer_union_upper_numerator,
        layer_union_upper_denominator=layer_union_upper_denominator,
        vacuum_reference=vacuum_reference,
        exact_target=exact_target.strip(),
        geometry_convention=geometry_convention.strip(),
        source_zero_route=source_zero_route.strip(),
        no_cancellation_scope=no_cancellation_scope.strip(),
        completion_scope=completion_scope.strip(),
        target_digest=target_digest,
        parent_target_evidence_digest=parent_target_evidence_digest,
        represented_source_digest=represented_source_digest,
        parent_source_evidence_digest=parent_source_evidence_digest,
        parent_represented_certificate_digest=parent_represented_certificate_digest,
        slab_digest=slab_digest,
        certificate_digest=certificate_digest,
    )
    return certificate  # noqa: RET504


__all__: list[str] = [
    "GalerkinLocalVacuumReference",
    "GalerkinLocalZeroSlabCertificate",
    "GalerkinLocalZeroSlabFailure",
]
