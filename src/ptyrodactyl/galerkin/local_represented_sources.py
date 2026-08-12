r"""Compose and directly certify represented ``LOCAL_CELL_LVT1`` sources.

Extended Summary
----------------
This leaf builds plane and coherent-focused exact finite-periodic incident
fields on a prepared local-cell target.  It consumes only a manifested
LVT.20 additional source, forms the frozen ``D/B/R/S/M/T/C`` vectors, and
directly encloses their exact-target counterparts.  No projected acquisition
row is promoted to an exact shell mode.

Routine Listings
----------------
:func:`certify_local_represented_source`
    Directly enclose exact ``D/B/R/S/M/T/C`` source actions.
:func:`compose_local_represented_focused_source`
    Compose one coherent exact-shell focused finite source.
:func:`compose_local_represented_plane_source`
    Compose one exact-shell represented plane mode.
:func:`prepare_local_represented_source`
    Full-reconstruct and exact-compare a represented source.
:func:`prepare_local_represented_source_certificate`
    Full-reconstruct source, rectangles, budget, and certificate digests.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jax.core import Tracer
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import (
    RealInterval,
    RootEnclosureError,
    all_normal_arithmetic_supported,
    fraction_from_float,
    fraction_upper_float,
    has_lost_nonzero_components,
    has_subnormal_components,
    host_binary64_supported,
    interval_add,
    interval_divide_positive,
    interval_multiply,
    interval_square,
    interval_subtract,
    mathematical_pi_interval,
    point_interval,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types import (
    GalerkinDirectionDisposition,
    GalerkinTerminalSide,
    scalar_float,
    scalar_int,
)
from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellTargetManifest,
)
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalComplexRectangles,
    GalerkinLocalRepresentedSource,
    GalerkinLocalRepresentedSourceActions,
    GalerkinLocalRepresentedSourceCertificate,
    GalerkinLocalRepresentedSourceFailure,
    GalerkinLocalRepresentedSourceKind,
    GalerkinLocalRepresentedSourceModes,
    GalerkinLocalSourceAxis,
    GalerkinLocalSourcePhaseConvention,
    _make_local_represented_source,
    _make_local_represented_source_certificate,
)
from ptyrodactyl.types.local_source_types import (
    GalerkinLocalAdditionalSourceCertificate,
)

from ._direct_interval import (
    _complex_interval_add,
    _complex_interval_multiply,
    _complex_point_interval,
    _ComplexInterval,
)
from .absorber import apply_axial_physical_cap
from .local_cell_interaction import apply_local_cell_interaction
from .local_cell_system import prepare_local_cell_galerkin_target
from .local_sources import prepare_local_additional_source_certificate

_ARITHMETIC: str = (
    "FTZ-safe outward binary64 intervals over exact stored v; target-owned "
    "exact D, L3 R, L4 exact-cap-scale times exact absorber rectangles, and "
    "the prepared direct LVT.20c source rectangles; bounded pairwise loops"
)
_CERTIFICATE_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_represented_source.direct_certificate.v1"
)
_COEFFICIENT_NORM: str = (
    "Euclidean l2 norm of ordered I_u complex coefficient vectors"
)
_DEFAULT_MAXIMUM_DIRECT_PAIRS: int = 2_000_000
_DIRECT_PAIR_COUNT_ROUTE: str = "n-free-plus-n-squared-B-plus-n-squared-R-v1"
_ELIGIBILITY_SCOPE: str = (
    "exact finite periodic LOCAL_CELL_LVT1 represented incident source only; "
    "requires prepared LVT.20 lift, declared EXACT_COEFFICIENT incident rows, "
    "independent exact [0,0] D witnesses, positive exact normal branches, "
    "positive exact-carrier flux, and one active branch per transverse fiber; "
    "excludes projected rows, continuum/window, slab, current, and detector"
)
_ERROR_SCOPE: str = (
    "direct complete D/B/R/S/M/T/C point-to-exact-rectangle errors; S uses "
    "the nested LVT.20 rectangles exactly once; branch bounds are direct and "
    "exclude parent delta_D/delta_R/delta_B/delta_H, duplicate LVT.20e norm, "
    "solver-state residual, stability, slab, terminal, and detector errors"
)
_EXACT_TARGET: str = (
    "exact LOCAL_CELL_LVT1 actions on exact stored finite incident v: "
    "D_exact v, B_exact v, R_exact v, S_add_exact, "
    "M_exact=D_exact v-i B_exact v, T_exact=M_exact+S_add_exact, "
    "C_exact=R_exact v+S_add_exact"
)
_INCIDENT_CONSTRUCTION: str = (
    "v=phase(aperture; algebraic physical kappa, transverse scan, source "
    "plane, aberration)*sqrt(stored target flux/stored phased SC.13 flux); "
    "exact represented target is the resulting stored complex128 v; v1"
)
_LOCAL_SOURCE_LIFT_FORMULA: str = (
    "s_square = D_i v - i b_square v + s_add_square"
)
_MAXIMUM_DIRECT_PAIRS: int = np.iinfo(np.int64).max
_MINIMUM_FOCUSED_MODES: int = 2
_PROJECTED_LIFT_FORMULA: str = (
    "P_Ku s_square = D_exact v - i B_exact v + S_add_exact"
)
_SCATTERED_SOURCE_FORMULA: str = "C_alg = R_alg v + S_add_alg"
_SOURCE_DIGEST_DOMAIN: str = "ptyrodactyl.local_represented_source.identity.v1"
_SOURCE_EVIDENCE_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_represented_source.evidence.v1"
)
_SPACE_DIMENSIONS: int = 3
_TOTAL_SOURCE_FORMULA: str = "T_alg = D_alg v - i B_alg v + S_add_alg"
_TRANSVERSE_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))
_VACUUM_MATCHED_FORMULA: str = "M_alg = D_alg v - i B_alg v"


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise for one structural represented-source failure.

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


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced leaves at one explicit host replay boundary.

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
            "represented-source replay requires concrete host values"
        )


def _active_mask(
    coefficients: Complex128[Array, " n"],
) -> Bool[Array, " n"]:
    """PRIVATE: Mark exactly nonzero complex aperture coefficients.

    Parameters
    ----------
    coefficients : Complex128[Array, " n"]
        Complex aperture coefficients.

    Returns
    -------
    active : Bool[Array, " n"]
        Mask true when either stored component is nonzero.
    """
    active: Bool[Array, " n"] = (jnp.real(coefficients) != 0.0) | (
        jnp.imag(coefficients) != 0.0
    )
    return active


def _checked_weights(
    target_size: int,
    aperture_weights: Complex[Array, "..."],
    kind: GalerkinLocalRepresentedSourceKind,
) -> Complex128[Array, " n"]:
    """PRIVATE: Check aperture shape, range, and source-kind cardinality.

    Parameters
    ----------
    target_size : int
        Retained target-state count.
    aperture_weights : Complex[Array, "..."]
        Candidate full-state aperture coefficients.
    kind : GalerkinLocalRepresentedSourceKind
        Plane or coherent-focused source kind.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite normal-range aperture coefficients.

    Raises
    ------
    ValueError
        If rank, length, or active cardinality is invalid.
    """
    weights: Complex128[Array, " n"] = jnp.asarray(
        aperture_weights, dtype=jnp.complex128
    )
    _raise_if(weights.ndim != 1, "aperture_weights must be 1D")
    _raise_if(
        weights.shape != (target_size,),
        "aperture_weights must match target I_u",
    )
    active_count = int(jnp.sum(_active_mask(weights)))
    if kind is GalerkinLocalRepresentedSourceKind.PLANE_MODE:
        _raise_if(active_count != 1, "plane source requires one active mode")
    else:
        _raise_if(
            active_count < _MINIMUM_FOCUSED_MODES,
            "focused source requires at least two modes",
        )
    checked: Complex128[Array, " n"] = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | has_subnormal_components(weights),
        "aperture weights must be finite and normal-or-zero",
    )
    return checked


def _checked_phase_geometry(
    target_size: int,
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    source_plane_coordinate: scalar_float,
    normal_axis: GalerkinLocalSourceAxis,
) -> Tuple[Float64[Array, " 3"], Float64[Array, " n"], Float64[Array, ""]]:
    """PRIVATE: Validate the explicit transverse phase geometry.

    Parameters
    ----------
    target_size : int
        Retained target-state count.
    scan_position : Float[Array, "..."]
        Candidate three-component transverse scan position.
    aberration_phases : Float[Array, "..."]
        Candidate per-state aberration phases.
    source_plane_coordinate : scalar_float
        Candidate source-plane coordinate.
    normal_axis : GalerkinLocalSourceAxis
        Positive coordinate-aligned source normal.

    Returns
    -------
    checked_scan : Float64[Array, " 3"]
        Finite exactly transverse scan position.
    aberrations : Float64[Array, " n"]
        Finite per-state aberration phases.
    coordinate : Float64[Array, ""]
        Finite source-plane coordinate.

    Raises
    ------
    ValueError
        If a candidate has the wrong static shape.
    """
    scan: Float64[Array, " 3"] = jnp.asarray(scan_position, dtype=jnp.float64)
    aberrations: Float64[Array, " n"] = jnp.asarray(
        aberration_phases, dtype=jnp.float64
    )
    coordinate: Float64[Array, ""] = jnp.asarray(
        source_plane_coordinate, dtype=jnp.float64
    )
    _raise_if(scan.shape != (_SPACE_DIMENSIONS,), "scan_position must be (3,)")
    _raise_if(
        aberrations.shape != (target_size,),
        "aberration_phases must match target I_u",
    )
    _raise_if(coordinate.shape != (), "source_plane_coordinate must be scalar")
    checked_scan: Float64[Array, " 3"] = eqx.error_if(
        scan,
        jnp.any(~jnp.isfinite(scan))
        | (scan[int(normal_axis)] != 0.0)
        | jnp.any(~jnp.isfinite(aberrations))
        | (~jnp.isfinite(coordinate)),
        "phase geometry must be finite and exactly transverse",
    )
    result: Tuple[
        Float64[Array, " 3"], Float64[Array, " n"], Float64[Array, ""]
    ] = (checked_scan, aberrations, coordinate)
    return result


def _physical_wavevectors(
    target: GalerkinLocalCellTargetManifest,
) -> Float64[Array, "n 3"]:
    """PRIVATE: Reconstruct algebraic physical angular wavevectors.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target exposing state indices, box lengths, and carrier.

    Returns
    -------
    wavevectors : Float64[Array, "n 3"]
        Algebraic physical angular wavevectors in radians per Angstrom.
    """
    reciprocal = target.state_indices / target.box_lengths[None, :]
    wavevectors: Float64[Array, "n 3"] = (
        target.carrier[None, :] + 2.0 * jnp.pi * reciprocal
    )
    return wavevectors


def _target_normal_axis(
    target: GalerkinLocalCellTargetManifest,
) -> GalerkinLocalSourceAxis:
    """PRIVATE: Derive the sole represented-source axis from the target.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared target owning the terminal/acquisition axis.

    Returns
    -------
    normal_axis : GalerkinLocalSourceAxis
        Coordinate-aligned target terminal axis.

    Raises
    ------
    ValueError
        If target storage does not encode one supported coordinate axis.
    """
    try:
        normal_axis: GalerkinLocalSourceAxis = GalerkinLocalSourceAxis(
            target.acquisition.terminal_axis
        )
    except ValueError as error:
        raise ValueError("target terminal axis is unsupported") from error
    return normal_axis


def _apply_phases(
    weights: Complex128[Array, " n"],
    wavevectors: Float64[Array, "n 3"],
    scan: Float64[Array, " 3"],
    aberrations: Float64[Array, " n"],
    coordinate: Float64[Array, ""],
    normal_axis: GalerkinLocalSourceAxis,
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the physical-wavevector source phase convention.

    Parameters
    ----------
    weights : Complex128[Array, " n"]
        Checked aperture coefficients.
    wavevectors : Float64[Array, "n 3"]
        Algebraic physical wavevectors.
    scan : Float64[Array, " 3"]
        Exactly transverse scan position.
    aberrations : Float64[Array, " n"]
        Per-state aberration phases.
    coordinate : Float64[Array, ""]
        Source-plane coordinate.
    normal_axis : GalerkinLocalSourceAxis
        Positive coordinate-aligned source normal.

    Returns
    -------
    phased : Complex128[Array, " n"]
        Coefficients after the explicit source phase.
    """
    normal = int(normal_axis)
    scan_phase = jnp.sum(wavevectors * scan[None, :], axis=-1)
    total_phase = (
        -scan_phase - wavevectors[:, normal] * coordinate + aberrations
    )
    result = weights * jnp.exp(1j * total_phase)
    phased: Complex128[Array, " n"] = eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result))
        | has_subnormal_components(result)
        | has_lost_nonzero_components(weights, result),
        "phase application must preserve each nonzero coefficient",
    )
    return phased


def _reduced_flux(
    coefficients: Complex128[Array, " n"],
    normal_components: Float64[Array, " n"],
    normal_box_length: Float64[Array, ""],
) -> Float64[Array, ""]:
    """PRIVATE: Evaluate the rounded SC.13 reduced-flux formula.

    Parameters
    ----------
    coefficients : Complex128[Array, " n"]
        Orthonormal-box coefficients.
    normal_components : Float64[Array, " n"]
        Algebraic normal angular-wavevector components.
    normal_box_length : Float64[Array, ""]
        Exact stored box length along the normal.

    Returns
    -------
    flux : Float64[Array, ""]
        Rounded signed reduced flux.
    """
    squared_moduli = jnp.real(coefficients) ** 2 + jnp.imag(coefficients) ** 2
    flux: Float64[Array, ""] = (
        jnp.sum(normal_components * squared_moduli) / normal_box_length
    )
    return flux


def _exact_normal_wavevectors(
    target: GalerkinLocalCellTargetManifest,
    normal_axis: GalerkinLocalSourceAxis,
) -> RealInterval:
    """PRIVATE: Enclose exact-target normal wavevectors on all state rows.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target exposing exact carrier and state geometry.
    normal_axis : GalerkinLocalSourceAxis
        Coordinate normal used by the represented source.

    Returns
    -------
    interval : RealInterval
        Inclusive exact-target normal-wavevector endpoints.
    """
    normal = int(normal_axis)
    indices = target.state_indices[:, normal].astype(jnp.float64)
    reciprocal = interval_divide_positive(
        point_interval(indices), point_interval(target.box_lengths[normal])
    )
    two_pi = interval_multiply(
        point_interval(jnp.asarray(2.0, dtype=jnp.float64)),
        mathematical_pi_interval(),
    )
    offset = interval_multiply(reciprocal, two_pi)
    ledger = target.fixed_linear_error_ledger
    carrier: RealInterval = (
        jnp.broadcast_to(
            ledger.exact_carrier_lower_bounds[normal], indices.shape
        ),
        jnp.broadcast_to(
            ledger.exact_carrier_upper_bounds[normal], indices.shape
        ),
    )
    interval: RealInterval = interval_add(carrier, offset)
    return interval


def _exact_flux_interval(
    incident: Complex128[Array, " n"],
    normal_wavevectors: RealInterval,
    normal_box_length: Float64[Array, ""],
    target_flux: Float64[Array, ""],
) -> Tuple[RealInterval, Float64[Array, ""]]:
    """PRIVATE: Enclose exact-carrier flux and target discrepancy.

    Parameters
    ----------
    incident : Complex128[Array, " n"]
        Exact stored finite incident coefficients.
    normal_wavevectors : RealInterval
        Exact-target normal-wavevector intervals.
    normal_box_length : Float64[Array, ""]
        Exact stored positive normal box length.
    target_flux : Float64[Array, ""]
        Exact stored requested reduced flux.

    Returns
    -------
    flux_interval : RealInterval
        Inclusive exact-carrier reduced-flux endpoints.
    discrepancy_upper : Float64[Array, ""]
        Outward target-to-exact-flux discrepancy.
    """
    real_points = point_interval(jnp.real(incident))
    imag_points = point_interval(jnp.imag(incident))
    squared_moduli = interval_add(
        interval_square(real_points), interval_square(imag_points)
    )
    contributions = interval_multiply(normal_wavevectors, squared_moduli)
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def add_contribution(
        index: scalar_int, accumulator: RealInterval
    ) -> RealInterval:
        """Accumulate one exact modal-flux interval."""
        contribution: RealInterval = (
            contributions[0][index],
            contributions[1][index],
        )
        updated: RealInterval = interval_add(accumulator, contribution)
        return updated

    summed = lax.fori_loop(
        0, incident.shape[0], add_contribution, (zero, zero)
    )
    flux_interval: RealInterval = interval_divide_positive(
        summed, point_interval(normal_box_length)
    )
    discrepancy = interval_subtract(point_interval(target_flux), flux_interval)
    discrepancy_upper: Float64[Array, ""] = jnp.maximum(
        jnp.abs(discrepancy[0]), jnp.abs(discrepancy[1])
    )
    result: Tuple[RealInterval, Float64[Array, ""]] = (
        flux_interval,
        discrepancy_upper,
    )
    return result


def _incident_row_masks(
    target: GalerkinLocalCellTargetManifest,
) -> Tuple[Bool[Array, " n"], Bool[Array, " n"]]:
    """PRIVATE: Match state rows to declared and exact incident rows.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target exposing state and acquisition rows.

    Returns
    -------
    declared : Bool[Array, " n"]
        Whether each state row is declared incident.
    exact : Bool[Array, " n"]
        Whether each row has an independent EXACT_COEFFICIENT declaration.
    """
    acquisition = target.acquisition
    matches = jnp.all(
        target.state_indices[:, None, :]
        == acquisition.incident_indices[None, :, :],
        axis=-1,
    )
    exact_dispositions = acquisition.incident_direction_dispositions == int(
        GalerkinDirectionDisposition.EXACT_COEFFICIENT
    )
    declared: Bool[Array, " n"] = jnp.any(matches, axis=1)
    exact: Bool[Array, " n"] = jnp.any(
        matches & exact_dispositions[None, :], axis=1
    )
    result: Tuple[Bool[Array, " n"], Bool[Array, " n"]] = (declared, exact)
    return result


def _has_duplicate_fiber(
    target: GalerkinLocalCellTargetManifest,
    active: Bool[Array, " n"],
    normal_axis: GalerkinLocalSourceAxis,
) -> Bool[Array, ""]:
    """PRIVATE: Detect two active normal branches on one transverse fiber.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target exposing ordered state indices.
    active : Bool[Array, " n"]
        Exactly active aperture mask.
    normal_axis : GalerkinLocalSourceAxis
        Axis removed from the transverse-fiber key.

    Returns
    -------
    duplicate : Bool[Array, ""]
        Whether two distinct active rows share one transverse key.
    """
    axes = _TRANSVERSE_AXES[int(normal_axis)]
    transverse = target.state_indices[:, axes]
    same = jnp.all(transverse[:, None, :] == transverse[None, :, :], axis=-1)
    distinct = ~jnp.eye(active.shape[0], dtype=jnp.bool_)
    duplicate: Bool[Array, ""] = jnp.any(
        same & distinct & active[:, None] & active[None, :]
    )
    return duplicate


def _incident_failure(  # noqa: PLR0913
    target: GalerkinLocalCellTargetManifest,
    additional: GalerkinLocalAdditionalSourceCertificate,
    active: Bool[Array, " n"],
    forward: Bool[Array, " n"],
    declared: Bool[Array, " n"],
    exact_disposition: Bool[Array, " n"],
    exact_shell: Bool[Array, " n"],
    exact_forward: Bool[Array, " n"],
    duplicate_fiber: Bool[Array, ""],
    exact_flux: RealInterval,
    normal_axis: GalerkinLocalSourceAxis,
) -> GalerkinLocalRepresentedSourceFailure:
    """PRIVATE: Select the first fail-closed represented-source outcome.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.
    additional : GalerkinLocalAdditionalSourceCertificate
        Fully prepared LVT.20 certificate.
    active : Bool[Array, " n"]
        Exactly active aperture mask.
    forward : Bool[Array, " n"]
        Algebraic positive-normal mask.
    declared : Bool[Array, " n"]
        Declared incident-row mask.
    exact_disposition : Bool[Array, " n"]
        Independent exact incident-disposition mask.
    exact_shell : Bool[Array, " n"]
        Independent exact symbolic ``[0,0]`` D mask.
    exact_forward : Bool[Array, " n"]
        Exact positive normal-wavevector mask.
    duplicate_fiber : Bool[Array, ""]
        Duplicate active transverse-fiber predicate.
    exact_flux : RealInterval
        Exact-carrier reduced-flux interval.
    normal_axis : GalerkinLocalSourceAxis
        Requested positive source normal.

    Returns
    -------
    failure : GalerkinLocalRepresentedSourceFailure
        First typed failed gate, or ``NONE``.
    """
    failure: GalerkinLocalRepresentedSourceFailure = (
        GalerkinLocalRepresentedSourceFailure.NONE
    )
    failures = GalerkinLocalRepresentedSourceFailure
    acquisition = target.acquisition
    if not bool(additional.finite_certificate):
        failure = failures.ADDITIONAL_SOURCE_NONCERTIFICATE
    elif not host_binary64_supported() or not bool(
        all_normal_arithmetic_supported()
    ):
        failure = failures.HOST_ARITHMETIC_UNSUPPORTED
    elif not (
        int(normal_axis) == acquisition.terminal_axis
        and acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
    ):
        failure = failures.TERMINAL_ORIENTATION_UNSUPPORTED
    elif bool(jnp.any(active & ~declared)):
        failure = (
            GalerkinLocalRepresentedSourceFailure.UNDECLARED_INCIDENT_MODE
        )
    elif bool(jnp.any(active & ~exact_disposition)):
        failure = (
            GalerkinLocalRepresentedSourceFailure.NONEXACT_INCIDENT_DISPOSITION
        )
    elif bool(jnp.any(active & ~exact_shell)):
        failure = GalerkinLocalRepresentedSourceFailure.EXACT_SHELL_FAILURE
    elif bool(jnp.any(active & (~forward | ~exact_forward))):
        failure = GalerkinLocalRepresentedSourceFailure.NONFORWARD_OR_GRAZING
    elif bool(duplicate_fiber):
        failure = (
            GalerkinLocalRepresentedSourceFailure.DUPLICATE_TRANSVERSE_FIBER
        )
    elif (
        not bool(jnp.isfinite(exact_flux[0]))
        or not bool(jnp.isfinite(exact_flux[1]))
        or not bool(exact_flux[0] > 0.0)
    ):
        failure = GalerkinLocalRepresentedSourceFailure.NONPOSITIVE_EXACT_FLUX
    return failure


def _source_digest(  # noqa: PLR0913
    target: GalerkinLocalCellTargetManifest,
    additional: GalerkinLocalAdditionalSourceCertificate,
    aperture_weights: Complex128[Array, " n"],
    target_flux: Float64[Array, ""],
    scan: Float64[Array, " 3"],
    aberrations: Float64[Array, " n"],
    coordinate: Float64[Array, ""],
    incident: Complex128[Array, " n"],
    kind: GalerkinLocalRepresentedSourceKind,
    normal_axis: GalerkinLocalSourceAxis,
    phase_convention: GalerkinLocalSourcePhaseConvention,
) -> str:
    """PRIVATE: Digest represented-source identity without proof budgets.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target exposing its operator digest.
    additional : GalerkinLocalAdditionalSourceCertificate
        Prepared additional source exposing its identity digest.
    aperture_weights : Complex128[Array, " n"]
        Exact stored aperture primitives.
    target_flux : Float64[Array, ""]
        Exact stored requested reduced flux.
    scan : Float64[Array, " 3"]
        Exact stored transverse scan position.
    aberrations : Float64[Array, " n"]
        Exact stored aberration phases.
    coordinate : Float64[Array, ""]
        Exact stored source-plane coordinate.
    incident : Complex128[Array, " n"]
        Actual stored normalized incident vector serving as exact ``v``.
    kind : GalerkinLocalRepresentedSourceKind
        Plane or coherent-focused kind.
    normal_axis : GalerkinLocalSourceAxis
        Positive source normal.
    phase_convention : GalerkinLocalSourcePhaseConvention
        Explicit coefficient phase convention.

    Returns
    -------
    source_digest : str
        Operator/source-identity digest.
    """
    source_digest: str = sha256(
        {
            "domain": _SOURCE_DIGEST_DOMAIN,
            "target_digest": target.target_digest,
            "additional_source_digest": additional.source.source_digest,
            "additional_source_route": additional.source.route.value,
            "aperture_weights": stored_value_payload(aperture_weights),
            "target_reduced_flux": stored_value_payload(target_flux),
            "scan_position": stored_value_payload(scan),
            "aberration_phases": stored_value_payload(aberrations),
            "source_plane_coordinate": stored_value_payload(coordinate),
            "incident_field": stored_value_payload(incident),
            "incident_construction": _INCIDENT_CONSTRUCTION,
            "kind": kind.value,
            "normal_axis": int(normal_axis),
            "phase_convention": phase_convention.value,
            "local_source_lift_formula": _LOCAL_SOURCE_LIFT_FORMULA,
            "projected_lift_formula": _PROJECTED_LIFT_FORMULA,
            "vacuum_matched_formula": _VACUUM_MATCHED_FORMULA,
            "total_source_formula": _TOTAL_SOURCE_FORMULA,
            "scattered_source_formula": _SCATTERED_SOURCE_FORMULA,
        }
    )
    return source_digest


def _source_evidence_digest(
    target: GalerkinLocalCellTargetManifest,
    additional: GalerkinLocalAdditionalSourceCertificate,
    modes: GalerkinLocalRepresentedSourceModes,
    actions: GalerkinLocalRepresentedSourceActions,
    failure: GalerkinLocalRepresentedSourceFailure,
    source_digest: str,
    source_name: str,
) -> str:
    """PRIVATE: Bind full parents, realization, gates, and source name.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared target.
    additional : GalerkinLocalAdditionalSourceCertificate
        Fully prepared LVT.20 result.
    modes : GalerkinLocalRepresentedSourceModes
        Complete rounded incident evidence.
    actions : GalerkinLocalRepresentedSourceActions
        Complete rounded source actions.
    failure : GalerkinLocalRepresentedSourceFailure
        Typed incident outcome.
    source_digest : str
        Bound source-identity digest.
    source_name : str
        Canonically stripped source name.

    Returns
    -------
    evidence_digest : str
        Full represented-source evidence digest.
    """
    evidence_digest: str = sha256(
        {
            "domain": _SOURCE_EVIDENCE_DIGEST_DOMAIN,
            "source_digest": source_digest,
            "source_name": source_name.strip(),
            "full_prepared_target": stored_value_payload(target),
            "full_prepared_additional_certificate": stored_value_payload(
                additional
            ),
            "modes": stored_value_payload(modes),
            "actions": stored_value_payload(actions),
            "failure": failure.value,
            "eligibility_scope": _ELIGIBILITY_SCOPE,
        }
    )
    return evidence_digest


def _compose_prepared(  # noqa: PLR0913, PLR0915
    target: GalerkinLocalCellTargetManifest,
    additional: GalerkinLocalAdditionalSourceCertificate,
    aperture_weights: Complex[Array, "..."],
    target_reduced_flux: scalar_float,
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    source_plane_coordinate: scalar_float,
    *,
    kind: GalerkinLocalRepresentedSourceKind,
    phase_convention: GalerkinLocalSourcePhaseConvention,
    source_name: str,
) -> GalerkinLocalRepresentedSource:
    """PRIVATE: Compose one source from two replay-authenticated parents.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.
    additional : GalerkinLocalAdditionalSourceCertificate
        Fully prepared LVT.20 result bound to the same target.
    aperture_weights : Complex[Array, "..."]
        Full-state aperture coefficients before phase and normalization.
    target_reduced_flux : scalar_float
        Positive requested reduced flux.
    scan_position : Float[Array, "..."]
        Three-component transverse scan position.
    aberration_phases : Float[Array, "..."]
        Per-state aberration phases.
    source_plane_coordinate : scalar_float
        Physical source-plane coordinate.
    kind : GalerkinLocalRepresentedSourceKind
        Plane or coherent-focused kind.
    phase_convention : GalerkinLocalSourcePhaseConvention
        Explicit source phase convention.
    source_name : str
        Canonically stripped source name.

    Returns
    -------
    source : GalerkinLocalRepresentedSource
        Canonical represented local source.

    Raises
    ------
    ValueError
        If parents, inputs, or frozen arithmetic are structurally invalid.
    """
    _raise_if(not source_name.strip(), "source_name must be nonempty")
    _raise_if(
        phase_convention
        is not GalerkinLocalSourcePhaseConvention.PHYSICAL_WAVEVECTOR,
        "only the physical-wavevector phase convention is admitted",
    )
    _raise_if(
        stored_value_payload(additional.source.target)
        != stored_value_payload(target),
        "additional-source certificate must bind the identical target",
    )
    normal_axis = _target_normal_axis(target)
    size = target.state_indices.shape[0]
    weights = _checked_weights(size, aperture_weights, kind)
    scan, aberrations, coordinate = _checked_phase_geometry(
        size,
        scan_position,
        aberration_phases,
        source_plane_coordinate,
        normal_axis,
    )
    wavevectors = _physical_wavevectors(target)
    phased = _apply_phases(
        weights, wavevectors, scan, aberrations, coordinate, normal_axis
    )
    normal = int(normal_axis)
    normal_components = wavevectors[:, normal]
    normal_length = target.box_lengths[normal]
    target_flux: Float64[Array, ""] = jnp.asarray(
        target_reduced_flux, dtype=jnp.float64
    )
    _raise_if(target_flux.shape != (), "target_reduced_flux must be scalar")
    _raise_if(
        not bool(jnp.isfinite(target_flux)) or not bool(target_flux > 0.0),
        "target_reduced_flux must be finite and positive",
    )
    aperture_flux = _reduced_flux(weights, normal_components, normal_length)
    input_flux = _reduced_flux(phased, normal_components, normal_length)
    safe_input = jnp.where(input_flux > 0.0, input_flux, 1.0)
    normalization = jnp.sqrt(target_flux / safe_input)
    incident_result = phased * normalization
    incident: Complex128[Array, " n"] = eqx.error_if(
        incident_result,
        (~jnp.isfinite(aperture_flux))
        | (~jnp.isfinite(input_flux))
        | (~jnp.isfinite(normalization))
        | (normalization <= 0.0)
        | jnp.any(~jnp.isfinite(incident_result))
        | has_subnormal_components(incident_result)
        | has_lost_nonzero_components(phased, incident_result),
        "one common positive reduced-flux normalization must remain finite",
    )
    output_flux = _reduced_flux(incident, normal_components, normal_length)
    exact_normal = _exact_normal_wavevectors(target, normal_axis)
    exact_flux, flux_discrepancy = _exact_flux_interval(
        incident, exact_normal, normal_length, target_flux
    )
    active = _active_mask(weights)
    forward = normal_components > 0.0
    grazing = normal_components == 0.0
    backward = normal_components < 0.0
    declared, exact_disposition = _incident_row_masks(target)
    ledger = target.fixed_linear_error_ledger
    exact_shell = (ledger.exact_free_diagonal_lower_bounds == 0.0) & (
        ledger.exact_free_diagonal_upper_bounds == 0.0
    )
    exact_forward = exact_normal[0] > 0.0
    duplicate = _has_duplicate_fiber(target, active, normal_axis)
    failure = _incident_failure(
        target,
        additional,
        active,
        forward,
        declared,
        exact_disposition,
        exact_shell,
        exact_forward,
        duplicate,
        exact_flux,
        normal_axis,
    )
    modes = GalerkinLocalRepresentedSourceModes(
        aperture_weights=weights,
        phased_coefficients=phased,
        incident_field=incident,
        algebraic_physical_wavevectors=wavevectors,
        exact_normal_wavevector_lower_bounds=exact_normal[0],
        exact_normal_wavevector_upper_bounds=exact_normal[1],
        active_mask=active,
        forward_mask=forward,
        grazing_mask=grazing,
        backward_mask=backward,
        declared_incident_mask=declared,
        exact_incident_disposition_mask=exact_disposition,
        exact_shell_mask=exact_shell,
        exact_forward_mask=exact_forward,
        scan_position=scan,
        aberration_phases=aberrations,
        source_plane_coordinate=coordinate,
        aperture_reduced_flux=aperture_flux,
        input_reduced_flux=input_flux,
        target_reduced_flux=target_flux,
        output_reduced_flux=output_flux,
        flux_normalization=normalization,
        exact_reduced_flux_lower_bound=exact_flux[0],
        exact_reduced_flux_upper_bound=exact_flux[1],
        target_reduced_flux_discrepancy_upper_bound=flux_discrepancy,
    )
    free_action = target.free_diagonal * incident
    cap_action = apply_axial_physical_cap(target.cap_floor_proof, incident)
    interaction_action = apply_local_cell_interaction(
        target.interaction_core, incident
    )
    additional_action = additional.source.algebraic_additional_source
    vacuum_matched = free_action - 1j * cap_action
    total = vacuum_matched + additional_action
    scattered = interaction_action + additional_action
    actions = GalerkinLocalRepresentedSourceActions(
        free_action=free_action,
        physical_cap_action=cap_action,
        interaction_action=interaction_action,
        additional_source=additional_action,
        vacuum_matched_source=vacuum_matched,
        total_source=total,
        scattered_source=scattered,
    )
    source_digest = _source_digest(
        target,
        additional,
        weights,
        target_flux,
        scan,
        aberrations,
        coordinate,
        incident,
        kind,
        normal_axis,
        phase_convention,
    )
    evidence_digest = _source_evidence_digest(
        target,
        additional,
        modes,
        actions,
        failure,
        source_digest,
        source_name,
    )
    source: GalerkinLocalRepresentedSource = _make_local_represented_source(
        target,
        additional,
        modes,
        actions,
        jnp.asarray(
            failure is GalerkinLocalRepresentedSourceFailure.NONE,
            dtype=jnp.bool_,
        ),
        kind=kind,
        normal_axis=normal_axis,
        phase_convention=phase_convention,
        failure=failure,
        local_source_lift_formula=_LOCAL_SOURCE_LIFT_FORMULA,
        projected_lift_formula=_PROJECTED_LIFT_FORMULA,
        vacuum_matched_formula=_VACUUM_MATCHED_FORMULA,
        total_source_formula=_TOTAL_SOURCE_FORMULA,
        scattered_source_formula=_SCATTERED_SOURCE_FORMULA,
        eligibility_scope=_ELIGIBILITY_SCOPE,
        target_digest=target.target_digest,
        additional_source_digest=additional.source.source_digest,
        source_digest=source_digest,
        source_evidence_digest=evidence_digest,
        source_name=source_name,
    )
    return source


def _prepare_parents(
    target: object,
    additional: object,
) -> Tuple[
    GalerkinLocalCellTargetManifest,
    GalerkinLocalAdditionalSourceCertificate,
]:
    """PRIVATE: Full-prepare and same-target compare both parent carriers.

    Parameters
    ----------
    target : object
        Public target carrier to authenticate.
    additional : object
        Public LVT.20 certificate to authenticate.

    Returns
    -------
    prepared_target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.
    prepared_additional : GalerkinLocalAdditionalSourceCertificate
        Fully prepared same-target LVT.20 result.

    Raises
    ------
    TypeError
        If either public carrier has the wrong route-specific type.
    ValueError
        If replay or same-target comparison fails.
    """
    if not isinstance(target, GalerkinLocalCellTargetManifest):
        raise TypeError("target must be GalerkinLocalCellTargetManifest")
    if not isinstance(additional, GalerkinLocalAdditionalSourceCertificate):
        raise TypeError(
            "additional_source_certificate must be "
            "GalerkinLocalAdditionalSourceCertificate"
        )
    prepared_target = prepare_local_cell_galerkin_target(target)
    prepared_additional = prepare_local_additional_source_certificate(
        additional
    )
    if stored_value_payload(prepared_additional.source.target) != (
        stored_value_payload(prepared_target)
    ):
        raise ValueError(
            "represented source requires identical target LVT.20 evidence"
        )
    result: Tuple[
        GalerkinLocalCellTargetManifest,
        GalerkinLocalAdditionalSourceCertificate,
    ] = (
        prepared_target,
        prepared_additional,
    )
    return result


@jaxtyped(typechecker=beartype)
def compose_local_represented_plane_source(  # noqa: PLR0913
    target: object,
    additional_source_certificate: object,
    state_position: int,
    aperture_weight: complex | Complex[Array, ""],
    target_reduced_flux: scalar_float,
    *,
    phase_convention: GalerkinLocalSourcePhaseConvention,
    source_plane_coordinate: scalar_float,
    scan_position: Float[Array, "..."],
    aberration_phase: scalar_float,
    source_name: str,
) -> GalerkinLocalRepresentedSource:
    """Compose one exact-shell represented plane mode.

    :see: :func:`~.test_local_represented_sources.\
test_direct_complex_source_rectangles_errors_norms_and_actions`

    Parameters
    ----------
    target : object
        Public local-cell target to authenticate in full.
    additional_source_certificate : object
        Public LVT.20 certificate to authenticate in full.
    state_position : int
        Static position of the one active target state.
    aperture_weight : complex or Complex[Array, ""]
        Nonzero complex aperture coefficient before phase and normalization.
    target_reduced_flux : scalar_float
        Positive requested reduced flux.
    phase_convention : GalerkinLocalSourcePhaseConvention
        Explicit physical-wavevector phase convention.
    source_plane_coordinate : scalar_float
        Physical source-plane coordinate.
    scan_position : Float[Array, "..."]
        Three-component transverse scan position.
    aberration_phase : scalar_float
        Active plane-mode aberration phase.
    source_name : str
        Nonempty source name excluded from source identity.

    Returns
    -------
    source : GalerkinLocalRepresentedSource
        Canonical represented plane source.

    Raises
    ------
    ValueError
        If an input or replay invariant fails.
    """
    prepared_target, prepared_additional = _prepare_parents(
        target, additional_source_certificate
    )
    if isinstance(state_position, bool):
        raise ValueError("state_position cannot be bool")
    size = prepared_target.state_indices.shape[0]
    _raise_if(
        state_position < 0 or state_position >= size,
        "state_position must index target I_u",
    )
    weight: Complex128[Array, ""] = jnp.asarray(
        aperture_weight, dtype=jnp.complex128
    )
    aberration: Float64[Array, ""] = jnp.asarray(
        aberration_phase, dtype=jnp.float64
    )
    _raise_if(weight.shape != (), "aperture_weight must be scalar")
    _raise_if(aberration.shape != (), "aberration_phase must be scalar")
    weights = (
        jnp.zeros((size,), dtype=jnp.complex128).at[state_position].set(weight)
    )
    aberrations = (
        jnp.zeros((size,), dtype=jnp.float64)
        .at[state_position]
        .set(aberration)
    )
    source: GalerkinLocalRepresentedSource = _compose_prepared(
        prepared_target,
        prepared_additional,
        weights,
        target_reduced_flux,
        scan_position,
        aberrations,
        source_plane_coordinate,
        kind=GalerkinLocalRepresentedSourceKind.PLANE_MODE,
        phase_convention=phase_convention,
        source_name=source_name,
    )
    return source


@jaxtyped(typechecker=beartype)
def compose_local_represented_focused_source(  # noqa: PLR0913
    target: object,
    additional_source_certificate: object,
    aperture_weights: Complex[Array, "..."],
    target_reduced_flux: scalar_float,
    *,
    phase_convention: GalerkinLocalSourcePhaseConvention,
    source_plane_coordinate: scalar_float,
    scan_position: Float[Array, "..."],
    aberration_phases: Float[Array, "..."],
    source_name: str,
) -> GalerkinLocalRepresentedSource:
    """Compose one coherent exact-shell focused finite source.

    :see: :func:`~.test_local_represented_sources.\
test_incident_gates_parent_coherence_and_derived_axis`

    Parameters
    ----------
    target : object
        Public local-cell target to authenticate in full.
    additional_source_certificate : object
        Public LVT.20 certificate to authenticate in full.
    aperture_weights : Complex[Array, "..."]
        Full-state coherent aperture coefficients before phase.
    target_reduced_flux : scalar_float
        Positive requested reduced flux.
    phase_convention : GalerkinLocalSourcePhaseConvention
        Explicit physical-wavevector phase convention.
    source_plane_coordinate : scalar_float
        Physical source-plane coordinate.
    scan_position : Float[Array, "..."]
        Three-component transverse scan position.
    aberration_phases : Float[Array, "..."]
        Full-state aberration phases.
    source_name : str
        Nonempty source name excluded from source identity.

    Returns
    -------
    source : GalerkinLocalRepresentedSource
        Canonical coherent-focused source or typed ineligible evidence.

    Raises
    ------
    ValueError
        If an input or replay invariant fails.
    """
    prepared_target, prepared_additional = _prepare_parents(
        target, additional_source_certificate
    )
    source: GalerkinLocalRepresentedSource = _compose_prepared(
        prepared_target,
        prepared_additional,
        aperture_weights,
        target_reduced_flux,
        scan_position,
        aberration_phases,
        source_plane_coordinate,
        kind=GalerkinLocalRepresentedSourceKind.COHERENT_FOCUSED,
        phase_convention=phase_convention,
        source_name=source_name,
    )
    return source


def prepare_local_represented_source(
    source: object,
) -> GalerkinLocalRepresentedSource:
    """Full-reconstruct and exact-compare a represented source.

    :see: :func:`~.test_local_represented_sources.\
test_incident_v_action_static_and_certificate_rehash_forgery`

    Parameters
    ----------
    source : object
        Public represented-source carrier to authenticate.

    Returns
    -------
    canonical : GalerkinLocalRepresentedSource
        Fresh canonical represented source.

    Raises
    ------
    TypeError
        If the carrier is not the disjoint local represented-source type.
    ValueError
        If parent or complete source replay differs from submission.
    """
    if not isinstance(source, GalerkinLocalRepresentedSource):
        raise TypeError("source must be GalerkinLocalRepresentedSource")
    _assert_concrete(source)
    prepared_target, prepared_additional = _prepare_parents(
        source.target, source.additional_source_certificate
    )
    canonical: GalerkinLocalRepresentedSource = _compose_prepared(
        prepared_target,
        prepared_additional,
        source.modes.aperture_weights,
        source.modes.target_reduced_flux,
        source.modes.scan_position,
        source.modes.aberration_phases,
        source.modes.source_plane_coordinate,
        kind=source.kind,
        phase_convention=source.phase_convention,
        source_name=source.source_name,
    )
    if stored_value_payload(canonical) != stored_value_payload(source):
        raise ValueError(
            "represented source does not match complete parent/source replay"
        )
    return canonical


def _as_interval(
    rectangles: GalerkinLocalComplexRectangles,
) -> _ComplexInterval:
    """PRIVATE: Expose one public rectangle carrier to interval arithmetic.

    Parameters
    ----------
    rectangles : GalerkinLocalComplexRectangles
        Public componentwise complex rectangles.

    Returns
    -------
    interval : _ComplexInterval
        Equivalent internal interval tuple.
    """
    interval: _ComplexInterval = tuple(rectangles)
    return interval


def _as_rectangles(
    interval: _ComplexInterval,
) -> GalerkinLocalComplexRectangles:
    """PRIVATE: Store one internal interval as a public rectangle carrier.

    Parameters
    ----------
    interval : _ComplexInterval
        Internal componentwise complex interval.

    Returns
    -------
    rectangles : GalerkinLocalComplexRectangles
        Equivalent public rectangle carrier.
    """
    rectangles: GalerkinLocalComplexRectangles = (
        GalerkinLocalComplexRectangles(*interval)
    )
    return rectangles  # noqa: RET504


def _verified_component_error_bounds(
    point: Complex128[Array, " n"],
    rectangles: GalerkinLocalComplexRectangles,
) -> Float64[Array, " n"]:
    """PRIVATE: Verify each point-to-rectangle Euclidean radius on host.

    Parameters
    ----------
    point : Complex128[Array, " n"]
        Exact stored algebraic action point.
    rectangles : GalerkinLocalComplexRectangles
        Exact-target comparison rectangles.

    Returns
    -------
    bounds : Float64[Array, " n"]
        Outward componentwise Euclidean error bounds.

    Raises
    ------
    RootEnclosureError
        If any rational squared radius cannot be square-root enclosed.
    """
    point_host = np.asarray(jax.device_get(point), dtype=np.complex128)
    endpoint_hosts = tuple(
        np.asarray(jax.device_get(values), dtype=np.float64)
        for values in rectangles
    )
    host_bounds = []
    for position, value in enumerate(point_host):
        point_real = fraction_from_float(float(value.real))
        point_imag = fraction_from_float(float(value.imag))
        real_radius = max(
            abs(
                point_real
                - fraction_from_float(float(endpoint_hosts[0][position]))
            ),
            abs(
                point_real
                - fraction_from_float(float(endpoint_hosts[1][position]))
            ),
        )
        imag_radius = max(
            abs(
                point_imag
                - fraction_from_float(float(endpoint_hosts[2][position]))
            ),
            abs(
                point_imag
                - fraction_from_float(float(endpoint_hosts[3][position]))
            ),
        )
        host_bounds.append(
            fraction_upper_float(
                sqrt_fraction_upper(real_radius**2 + imag_radius**2)
            )
        )
    bounds: Float64[Array, " n"] = jnp.asarray(
        np.asarray(host_bounds, dtype=np.float64), dtype=jnp.float64
    )
    return bounds


def _verified_nonnegative_norm_upper(
    values: Float64[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Verify one stored non-negative vector norm on host.

    Parameters
    ----------
    values : Float64[Array, " n"]
        Exact stored non-negative component bounds.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward Euclidean norm upper bound.

    Raises
    ------
    RootEnclosureError
        If the rational squared norm cannot be square-root enclosed.
    """
    host = np.asarray(jax.device_get(values), dtype=np.float64)
    squared = sum(
        (fraction_from_float(float(value)) ** 2 for value in host),
        start=fraction_from_float(0.0),
    )
    upper: Float64[Array, ""] = jnp.asarray(
        fraction_upper_float(sqrt_fraction_upper(squared)),
        dtype=jnp.float64,
    )
    return upper


def _verified_complex_norm_upper(
    values: Complex128[Array, " n"],
) -> Float64[Array, ""]:
    """PRIVATE: Verify one exact stored complex-vector norm on host.

    Parameters
    ----------
    values : Complex128[Array, " n"]
        Exact stored complex vector.

    Returns
    -------
    upper : Float64[Array, ""]
        Outward Euclidean norm upper bound.

    Raises
    ------
    RootEnclosureError
        If the rational squared norm cannot be square-root enclosed.
    """
    host = np.asarray(jax.device_get(values), dtype=np.complex128)
    squared = sum(
        (
            fraction_from_float(float(value.real)) ** 2
            + fraction_from_float(float(value.imag)) ** 2
            for value in host
        ),
        start=fraction_from_float(0.0),
    )
    upper: Float64[Array, ""] = jnp.asarray(
        fraction_upper_float(sqrt_fraction_upper(squared)),
        dtype=jnp.float64,
    )
    return upper


def _contract_rectangles(
    pair_positions: Int64[Array, " s"],
    coefficient_rectangles: GalerkinLocalComplexRectangles,
    field: Complex128[Array, " n"],
) -> GalerkinLocalComplexRectangles:
    """PRIVATE: Contract exact coefficient rectangles through one pair map.

    Parameters
    ----------
    pair_positions : Int64[Array, " s"]
        Flattened row-major coefficient positions for every state pair.
    coefficient_rectangles : GalerkinLocalComplexRectangles
        Exact coefficient rectangles in support order.
    field : Complex128[Array, " n"]
        Exact stored incident coefficients.

    Returns
    -------
    rectangles : GalerkinLocalComplexRectangles
        Outward exact action rectangles.
    """
    size = field.shape[0]
    zero = jnp.zeros((size,), dtype=jnp.float64)
    initial: _ComplexInterval = (zero, zero, zero, zero)

    def add_pair(
        flat_position: scalar_int,
        accumulator: _ComplexInterval,
    ) -> _ComplexInterval:
        """Accumulate one exact coefficient-field rectangle product."""
        row = flat_position // size
        column = flat_position % size
        position = pair_positions[flat_position]
        coefficient: _ComplexInterval = (
            coefficient_rectangles.real_lower_bounds[position],
            coefficient_rectangles.real_upper_bounds[position],
            coefficient_rectangles.imag_lower_bounds[position],
            coefficient_rectangles.imag_upper_bounds[position],
        )
        product = _complex_interval_multiply(
            coefficient, _complex_point_interval(field[column])
        )
        prior: _ComplexInterval = (
            accumulator[0][row],
            accumulator[1][row],
            accumulator[2][row],
            accumulator[3][row],
        )
        updated = _complex_interval_add(prior, product)
        result: _ComplexInterval = (
            accumulator[0].at[row].set(updated[0]),
            accumulator[1].at[row].set(updated[1]),
            accumulator[2].at[row].set(updated[2]),
            accumulator[3].at[row].set(updated[3]),
        )
        return result

    interval = lax.fori_loop(0, size * size, add_pair, initial)
    rectangles: GalerkinLocalComplexRectangles = _as_rectangles(interval)
    return rectangles  # noqa: RET504


def _exact_action_rectangles(
    source: GalerkinLocalRepresentedSource,
) -> Tuple[GalerkinLocalComplexRectangles, ...]:
    """PRIVATE: Build direct exact ``D/B/R/S/M/T/C`` rectangles.

    Parameters
    ----------
    source : GalerkinLocalRepresentedSource
        Canonical represented source and exact-parent evidence.

    Returns
    -------
    rectangles : Tuple[GalerkinLocalComplexRectangles, ...]
        Ordered exact ``D/B/R/S/M/T/C`` action rectangles.
    """
    target = source.target
    incident = source.modes.incident_field
    size = incident.shape[0]
    zeros = jnp.zeros((size,), dtype=jnp.float64)
    ledger = target.fixed_linear_error_ledger
    diagonal = GalerkinLocalComplexRectangles(
        ledger.exact_free_diagonal_lower_bounds,
        ledger.exact_free_diagonal_upper_bounds,
        zeros,
        zeros,
    )
    free = _as_rectangles(
        _complex_interval_multiply(
            _as_interval(diagonal), _complex_point_interval(incident)
        )
    )
    cap_certificate = target.cap_floor_proof.coefficient_certificate
    cap_coefficients = GalerkinLocalComplexRectangles(
        cap_certificate.exact_coefficient_real_lower_bounds,
        cap_certificate.exact_coefficient_real_upper_bounds,
        cap_certificate.exact_coefficient_imag_lower_bounds,
        cap_certificate.exact_coefficient_imag_upper_bounds,
    )
    dimensionless_cap = _contract_rectangles(
        cap_certificate.state_pair_absorber_positions,
        cap_coefficients,
        incident,
    )
    exact_scale = jnp.asarray(
        target.exact_cap_scale + 0.0j, dtype=jnp.complex128
    )
    cap = _as_rectangles(
        _complex_interval_multiply(
            _complex_point_interval(exact_scale),
            _as_interval(dimensionless_cap),
        )
    )
    compression = target.compression
    support_size = compression.interaction_coefficients.shape[0]
    support_zeros = jnp.zeros((support_size,), dtype=jnp.float64)
    positions = compression.difference_interaction_positions
    interaction_coefficients = GalerkinLocalComplexRectangles(
        support_zeros.at[positions].set(
            compression.exact_interaction_real_lower_bounds
        ),
        support_zeros.at[positions].set(
            compression.exact_interaction_real_upper_bounds
        ),
        support_zeros.at[positions].set(
            compression.exact_interaction_imag_lower_bounds
        ),
        support_zeros.at[positions].set(
            compression.exact_interaction_imag_upper_bounds
        ),
    )
    interaction = _contract_rectangles(
        compression.state_pair_interaction_positions,
        interaction_coefficients,
        incident,
    )
    additional_certificate = source.additional_source_certificate
    additional = GalerkinLocalComplexRectangles(
        additional_certificate.exact_source_real_lower_bounds,
        additional_certificate.exact_source_real_upper_bounds,
        additional_certificate.exact_source_imag_lower_bounds,
        additional_certificate.exact_source_imag_upper_bounds,
    )
    minus_i = _complex_point_interval(jnp.asarray(-1j, dtype=jnp.complex128))
    matched = _as_rectangles(
        _complex_interval_add(
            _as_interval(free),
            _complex_interval_multiply(minus_i, _as_interval(cap)),
        )
    )
    total = _as_rectangles(
        _complex_interval_add(_as_interval(matched), _as_interval(additional))
    )
    scattered = _as_rectangles(
        _complex_interval_add(
            _as_interval(interaction), _as_interval(additional)
        )
    )
    rectangles: Tuple[GalerkinLocalComplexRectangles, ...] = (
        free,
        cap,
        interaction,
        additional,
        matched,
        total,
        scattered,
    )
    return rectangles


def _certificate_digest(
    source: GalerkinLocalRepresentedSource,
    rectangles: Tuple[GalerkinLocalComplexRectangles, ...],
    errors: Tuple[Float64[Array, " n"], ...],
    bounds: Tuple[Float64[Array, ""], ...],
    field_norm: Float64[Array, ""],
    pair_count: Int64[Array, ""],
    pair_budget: Int64[Array, ""],
    failure: GalerkinLocalRepresentedSourceFailure,
) -> str:
    """PRIVATE: Bind every direct rectangle, error, budget, and parent.

    Parameters
    ----------
    source : GalerkinLocalRepresentedSource
        Canonical represented source.
    rectangles : Tuple[GalerkinLocalComplexRectangles, ...]
        Ordered exact ``D/B/R/S/M/T/C`` rectangles.
    errors : Tuple[Float64[Array, " n"], ...]
        Ordered component error arrays.
    bounds : Tuple[Float64[Array, ""], ...]
        Ordered action/source norm bounds.
    field_norm : Float64[Array, ""]
        Incident-field norm upper bound.
    pair_count : Int64[Array, ""]
        Direct work count.
    pair_budget : Int64[Array, ""]
        Certified work budget.
    failure : GalerkinLocalRepresentedSourceFailure
        Typed certificate outcome.

    Returns
    -------
    certificate_digest : str
        Complete direct-certificate evidence digest.
    """
    certificate_digest: str = sha256(
        {
            "domain": _CERTIFICATE_DIGEST_DOMAIN,
            "full_prepared_source": stored_value_payload(source),
            "source_digest": source.source_digest,
            "source_evidence_digest": source.source_evidence_digest,
            "additional_certificate_digest": (
                source.additional_source_certificate.certificate_digest
            ),
            "rectangles_D_B_R_S_M_T_C": stored_value_payload(rectangles),
            "component_errors_D_B_R_S_M_T_C": stored_value_payload(errors),
            "norm_bounds_D_B_R_S_M_T_C": stored_value_payload(bounds),
            "incident_field_norm_upper": stored_value_payload(field_norm),
            "direct_pair_count": stored_value_payload(pair_count),
            "maximum_direct_pairs": stored_value_payload(pair_budget),
            "failure": failure.value,
            "exact_target": _EXACT_TARGET,
            "arithmetic": _ARITHMETIC,
            "direct_pair_count_route": _DIRECT_PAIR_COUNT_ROUTE,
            "error_scope": _ERROR_SCOPE,
            "coefficient_norm": _COEFFICIENT_NORM,
        }
    )
    return certificate_digest


def _infinite_rectangles(size: int) -> GalerkinLocalComplexRectangles:
    """PRIVATE: Build one all-infinite typed noncertificate rectangle group.

    Parameters
    ----------
    size : int
        Retained target-state count.

    Returns
    -------
    rectangles : GalerkinLocalComplexRectangles
        All-infinite rectangle group.
    """
    lower = jnp.full((size,), -jnp.inf, dtype=jnp.float64)
    upper = jnp.full((size,), jnp.inf, dtype=jnp.float64)
    rectangles: GalerkinLocalComplexRectangles = (
        GalerkinLocalComplexRectangles(lower, upper, lower, upper)
    )
    return rectangles  # noqa: RET504


def _failure_certificate(
    source: GalerkinLocalRepresentedSource,
    pair_count: int,
    pair_budget: int,
    failure: GalerkinLocalRepresentedSourceFailure,
) -> GalerkinLocalRepresentedSourceCertificate:
    """PRIVATE: Create one typed direct noncertificate with copied S evidence.

    Parameters
    ----------
    source : GalerkinLocalRepresentedSource
        Canonical represented source.
    pair_count : int
        Direct work count associated with the outcome.
    pair_budget : int
        Certified direct-work budget.
    failure : GalerkinLocalRepresentedSourceFailure
        Typed non-success outcome.

    Returns
    -------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Digest-bound typed noncertificate.
    """
    size = source.target.state_indices.shape[0]
    infinite = _infinite_rectangles(size)
    additional = source.additional_source_certificate
    source_rectangles = GalerkinLocalComplexRectangles(
        additional.exact_source_real_lower_bounds,
        additional.exact_source_real_upper_bounds,
        additional.exact_source_imag_lower_bounds,
        additional.exact_source_imag_upper_bounds,
    )
    rectangles: Tuple[GalerkinLocalComplexRectangles, ...] = (
        infinite,
        infinite,
        infinite,
        source_rectangles,
        infinite,
        infinite,
        infinite,
    )
    infinite_errors = jnp.full((size,), jnp.inf, dtype=jnp.float64)
    errors: Tuple[Float64[Array, " n"], ...] = (
        infinite_errors,
        infinite_errors,
        infinite_errors,
        additional.component_error_bounds,
        infinite_errors,
        infinite_errors,
        infinite_errors,
    )
    infinity = jnp.asarray(jnp.inf, dtype=jnp.float64)
    bounds: Tuple[Float64[Array, ""], ...] = (
        infinity,
        infinity,
        infinity,
        additional.additional_source_error_upper_bound,
        infinity,
        infinity,
        infinity,
    )
    count = jnp.asarray(pair_count, dtype=jnp.int64)
    budget = jnp.asarray(pair_budget, dtype=jnp.int64)
    digest = _certificate_digest(
        source, rectangles, errors, bounds, infinity, count, budget, failure
    )
    certificate: GalerkinLocalRepresentedSourceCertificate = (
        _make_local_represented_source_certificate(
            source,
            rectangles,
            errors,
            bounds,
            infinity,
            jnp.asarray(False, dtype=jnp.bool_),
            count,
            budget,
            failure=failure,
            exact_target=_EXACT_TARGET,
            arithmetic=_ARITHMETIC,
            direct_pair_count_route=_DIRECT_PAIR_COUNT_ROUTE,
            error_scope=_ERROR_SCOPE,
            coefficient_norm=_COEFFICIENT_NORM,
            parent_source_evidence_digest=source.source_evidence_digest,
            parent_additional_certificate_digest=additional.certificate_digest,
            certificate_digest=digest,
        )
    )
    return certificate  # noqa: RET504


def _direct_pair_count(size: int) -> int:
    """PRIVATE: Compute the exact schema-representable direct work count.

    Parameters
    ----------
    size : int
        Non-negative retained target-state count.

    Returns
    -------
    pair_count : int
        Exact ``n + 2 n**2`` direct action count.

    Raises
    ------
    ValueError
        If ``size`` is negative or the exact count exceeds signed int64.
    """
    _raise_if(size < 0, "target-state count cannot be negative")
    pair_count: int = size + 2 * size * size
    _raise_if(
        pair_count > _MAXIMUM_DIRECT_PAIRS,
        "exact direct pair count must fit signed int64 storage",
    )
    return pair_count


def _certify_canonical(  # noqa: PLR0911
    source: GalerkinLocalRepresentedSource,
    maximum_direct_pairs: int,
) -> GalerkinLocalRepresentedSourceCertificate:
    """PRIVATE: Directly certify one canonical represented source.

    Parameters
    ----------
    source : GalerkinLocalRepresentedSource
        Canonical represented source.
    maximum_direct_pairs : int
        Positive signed-64-bit direct-work budget.

    Returns
    -------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Finite direct certificate or typed noncertificate.
    """
    size = source.target.state_indices.shape[0]
    exact_count = _direct_pair_count(size)
    if exact_count > maximum_direct_pairs:
        certificate: GalerkinLocalRepresentedSourceCertificate
        certificate = _failure_certificate(
            source,
            exact_count,
            maximum_direct_pairs,
            GalerkinLocalRepresentedSourceFailure.DIRECT_WORK_BUDGET_EXCEEDED,
        )
        return certificate  # noqa: RET504
    if source.failure is not GalerkinLocalRepresentedSourceFailure.NONE:
        certificate = _failure_certificate(
            source, exact_count, maximum_direct_pairs, source.failure
        )
        return certificate  # noqa: RET504
    if not host_binary64_supported() or not bool(
        all_normal_arithmetic_supported()
    ):
        certificate = _failure_certificate(
            source,
            exact_count,
            maximum_direct_pairs,
            GalerkinLocalRepresentedSourceFailure.HOST_ARITHMETIC_UNSUPPORTED,
        )
        return certificate  # noqa: RET504
    rectangles: Tuple[GalerkinLocalComplexRectangles, ...] = (
        _exact_action_rectangles(source)
    )
    rectangle_arrays = tuple(
        values for rectangle in rectangles for values in rectangle
    )
    if any(
        not bool(jnp.all(jnp.isfinite(values)))
        or bool(has_subnormal_components(values))
        for values in rectangle_arrays
    ):
        certificate = _failure_certificate(
            source,
            exact_count,
            maximum_direct_pairs,
            GalerkinLocalRepresentedSourceFailure.ARITHMETIC_RANGE_FAILURE,
        )
        return certificate  # noqa: RET504
    action_points: Tuple[Complex128[Array, " n"], ...] = tuple(source.actions)
    direct_positions = (0, 1, 2, 4, 5, 6)
    try:
        verified_errors = {
            position: _verified_component_error_bounds(
                action_points[position], rectangles[position]
            )
            for position in direct_positions
        }
        additional = source.additional_source_certificate
        errors: Tuple[Float64[Array, " n"], ...] = (
            verified_errors[0],
            verified_errors[1],
            verified_errors[2],
            additional.component_error_bounds,
            verified_errors[4],
            verified_errors[5],
            verified_errors[6],
        )
        verified_bounds = {
            position: _verified_nonnegative_norm_upper(errors[position])
            for position in direct_positions
        }
        bounds: Tuple[Float64[Array, ""], ...] = (
            verified_bounds[0],
            verified_bounds[1],
            verified_bounds[2],
            additional.additional_source_error_upper_bound,
            verified_bounds[4],
            verified_bounds[5],
            verified_bounds[6],
        )
        field_norm: Float64[Array, ""] = _verified_complex_norm_upper(
            source.modes.incident_field
        )
    except RootEnclosureError:
        certificate = _failure_certificate(
            source,
            exact_count,
            maximum_direct_pairs,
            GalerkinLocalRepresentedSourceFailure.ROOT_ENCLOSURE_FAILURE,
        )
        return certificate  # noqa: RET504
    arrays = (
        *rectangle_arrays,
        *errors,
        *bounds,
        field_norm,
    )
    if any(
        not bool(jnp.all(jnp.isfinite(value)))
        or bool(has_subnormal_components(value))
        for value in arrays
    ):
        certificate = _failure_certificate(
            source,
            exact_count,
            maximum_direct_pairs,
            GalerkinLocalRepresentedSourceFailure.ARITHMETIC_RANGE_FAILURE,
        )
        return certificate  # noqa: RET504
    stopped = jax.tree.map(
        lax.stop_gradient,
        (rectangles, errors, bounds, field_norm),
    )
    count = jnp.asarray(exact_count, dtype=jnp.int64)
    budget = jnp.asarray(maximum_direct_pairs, dtype=jnp.int64)
    digest = _certificate_digest(
        source,
        stopped[0],
        stopped[1],
        stopped[2],
        stopped[3],
        count,
        budget,
        GalerkinLocalRepresentedSourceFailure.NONE,
    )
    certificate = _make_local_represented_source_certificate(
        source,
        stopped[0],
        stopped[1],
        stopped[2],
        stopped[3],
        jnp.asarray(True, dtype=jnp.bool_),
        count,
        budget,
        failure=GalerkinLocalRepresentedSourceFailure.NONE,
        exact_target=_EXACT_TARGET,
        arithmetic=_ARITHMETIC,
        direct_pair_count_route=_DIRECT_PAIR_COUNT_ROUTE,
        error_scope=_ERROR_SCOPE,
        coefficient_norm=_COEFFICIENT_NORM,
        parent_source_evidence_digest=source.source_evidence_digest,
        parent_additional_certificate_digest=(
            source.additional_source_certificate.certificate_digest
        ),
        certificate_digest=digest,
    )
    return certificate  # noqa: RET504


def certify_local_represented_source(
    source: object,
    *,
    maximum_direct_pairs: int = _DEFAULT_MAXIMUM_DIRECT_PAIRS,
) -> GalerkinLocalRepresentedSourceCertificate:
    """Directly enclose exact ``D/B/R/S/M/T/C`` source actions.

    :see: :func:`~.test_local_represented_sources.\
test_direct_complex_source_rectangles_errors_norms_and_actions`

    Parameters
    ----------
    source : object
        Public represented-source carrier to authenticate.
    maximum_direct_pairs : int, optional
        Positive signed-64-bit work budget. By default, 2,000,000.

    Returns
    -------
    certificate : GalerkinLocalRepresentedSourceCertificate
        Finite direct certificate or typed noncertificate.

    Raises
    ------
    ValueError
        If the work budget is invalid or complete replay detects forgery.
    """
    if (
        isinstance(maximum_direct_pairs, bool)
        or not isinstance(maximum_direct_pairs, int)
        or maximum_direct_pairs <= 0
        or maximum_direct_pairs > _MAXIMUM_DIRECT_PAIRS
    ):
        raise ValueError(
            "maximum_direct_pairs must be a positive signed-64-bit integer"
        )
    canonical = prepare_local_represented_source(source)
    certificate: GalerkinLocalRepresentedSourceCertificate = (
        _certify_canonical(canonical, maximum_direct_pairs)
    )
    return certificate


def prepare_local_represented_source_certificate(
    certificate: object,
) -> GalerkinLocalRepresentedSourceCertificate:
    """Full-reconstruct source, rectangles, budget, and certificate digests.

    :see: :func:`~.test_local_represented_sources.\
test_incident_v_action_static_and_certificate_rehash_forgery`

    Parameters
    ----------
    certificate : object
        Public represented-source certificate to authenticate.

    Returns
    -------
    canonical : GalerkinLocalRepresentedSourceCertificate
        Fresh canonical direct certificate.

    Raises
    ------
    TypeError
        If the carrier has the wrong route-specific type.
    ValueError
        If scalar storage or complete replay differs from submission.
    """
    if not isinstance(certificate, GalerkinLocalRepresentedSourceCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalRepresentedSourceCertificate"
        )
    _assert_concrete(certificate)
    budget_array = np.asarray(jax.device_get(certificate.maximum_direct_pairs))
    if budget_array.dtype != np.dtype(np.int64) or budget_array.shape != ():
        raise ValueError("maximum_direct_pairs must be exact int64 scalar")
    canonical_source = prepare_local_represented_source(certificate.source)
    canonical: GalerkinLocalRepresentedSourceCertificate = _certify_canonical(
        canonical_source, int(budget_array)
    )
    if stored_value_payload(canonical) != stored_value_payload(certificate):
        raise ValueError(
            "represented-source certificate does not match complete replay"
        )
    return canonical


__all__: list[str] = [
    "certify_local_represented_source",
    "compose_local_represented_focused_source",
    "compose_local_represented_plane_source",
    "prepare_local_represented_source",
    "prepare_local_represented_source_certificate",
]
