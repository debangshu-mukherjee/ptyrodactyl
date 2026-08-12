r"""Enclose the exact local Galerkin projection defect.

Extended Summary
----------------
This host-only leaf replays one exact local zero slab and one same-source
local-stability result.  It builds the exact-rational LVT.55c slab Gram
rectangles, encloses the submitted-state LVT.55d defect, and transfers the
L6 state radius through a verified LVT.55e row-sum operator-norm upper bound
exactly once.  Structural singleton-zero evidence remains independent.

Routine Listings
----------------
:func:`enclose_local_projection_defect`
    Enclose scoped LVT.34--LVT.40 and LVT.55c--LVT.55e evidence.
:func:`prepare_local_projection_defect_certificate`
    Replay both parents, policies, exact Gram arithmetic, and all digests.
"""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from numpy.typing import NDArray

from ptyrodactyl._tools import (
    ComplexRectangle,
    RootEnclosureError,
    all_normal_arithmetic_supported,
    arithmetic_environment_probes,
    complex_rectangle_multiply,
    conjugate_rectangle,
    fraction_lower_float,
    fraction_upper_float,
    has_subnormal_components,
    host_array,
    host_binary64_supported,
    normalized_sinc_integer_ratio,
    pairwise_rectangle_sum,
    rational_turn_exponential,
    scale_complex_rectangle,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types.local_projection_types import (
    GalerkinLocalProjectionDefectCertificate,
    GalerkinLocalProjectionDefectFailure,
    _make_local_projection_defect_certificate,
)
from ptyrodactyl.types.local_stability_types import (
    GalerkinLocalStabilityResult,
)
from ptyrodactyl.types.local_terminal_types import GalerkinLocalTerminalScope
from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalZeroSlabCertificate,
)

from .local_stability import prepare_local_galerkin_stability_result
from .local_zero_slab import prepare_local_zero_slab_certificate

type _GramMatrix = list[list[ComplexRectangle]]
type _ScopeMapping = tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
]
type _FreeEvidence = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]
type _FiberRationalReports = tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
]

_CERTIFICATE_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_projection.lvt34_lvt40_lvt55_evidence.v1"
)
_COMPLETION_SCOPE: str = (
    "exact local projection-defect Gram, submitted-state measurement, and "
    "same-state-radius transfer only; excludes Cauchy propagators, branch "
    "classification, current, native-vacuum disposition, detector response, "
    "continuum error, and per-call terminal arithmetic"
)
_DEFAULT_MAXIMUM_GRAM_PAIRS: int = 2_000_000
_DEFAULT_MAXIMUM_STABILITY_DIRECT_PAIRS: int = 2_000_000
_ERROR_SCOPE: str = (
    "per fiber E_f,h <= measured_h + operator_h*B exactly once; measured_h "
    "is the exact-rectangle z*G*z report for submitted stored x; operator_h "
    "is a verified max-absolute-row-sum upper for diag(d)Gdiag(d), square-"
    "rooted to bound ||D0,h||; B is the replayed exact-target L6 state "
    "radius; excludes delta_H/source/residual recharging, numerical "
    "cancellation upgrades, terminal realization, and detector error"
)
_GRAM_FORMULA: str = (
    "G[p,q]=delta*sinc_pi((q-p)*delta)*exp(+2*pi*i*(q-p)*zeta/L); "
    "exact Fraction delta/zeta/L, normalized_sinc_integer_ratio, and "
    "rational_turn_exponential(-turn)"
)
_GRAM_TRANSCRIPT_DOMAIN: str = (
    "ptyrodactyl.local_projection.lvt55c_exact_gram.signed_hex_magnitude.v1"
)
_IDENTITY_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_projection.lvt34_lvt40_lvt55_identity.v1"
)
_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_MEASUREMENT_FORMULA: str = (
    "z[p]=d[p]*x[p] with exact dyadic x and exact-D intervals; "
    "measured_h^2=conj(z)^T G z by deterministic rational rectangles"
)
_STATE_TRANSFER_REPORT_INDEX: int = 5
_OPERATOR_BOUND_FORMULA: str = (
    "||D0,h||^2 <= max_p sum_q |(diag(d)Gdiag(d))[p,q]|; verified "
    "Hermitian absolute-row-sum upper, not an exact spectral norm"
)
_PI_TARGET_BITS: int = 224
_PRECISION_TRANSCRIPT: str = (
    "host_interval exact Fraction route: 224-bit Machin pi target, "
    "alternating sine/cosine last indices lower=21 upper=20, and "
    "128-bit dyadic rational square-root enclosure"
)
_SINE_TAYLOR_LOWER_LAST_INDEX: int = 21
_SINE_TAYLOR_UPPER_LAST_INDEX: int = 20
_SQRT_PRECISION_BITS: int = 128
_STATE_LIFT_FORMULA: str = (
    "E_f,h=up(measured_h+up(operator_h*B)); the L6 coefficient-l2 state "
    "radius B appears once and no separate state-error term is added"
)
_TRANSVERSE_AXES: tuple[tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced leaves at this exact host boundary.

    Parameters
    ----------
    value : object
        Submitted carrier, scope, or policy tree.

    Raises
    ------
    ValueError
        If any submitted leaf is a JAX tracer.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError(
            "local projection certification requires concrete host values"
        )


def _checked_positive_binary64(value: object, name: str) -> np.float64:
    """PRIVATE: Validate one finite positive normal exact-float64 policy.

    Parameters
    ----------
    value : object
        Candidate Python, NumPy, or JAX scalar.
    name : str
        Public policy name used in diagnostics.

    Returns
    -------
    checked : np.float64
        Concrete exact binary64 policy value.

    Raises
    ------
    TypeError
        If the candidate is boolean, integral, complex, or not float64.
    ValueError
        If the candidate is traced, nonscalar, nonfinite, or subnormal.
    """
    if isinstance(value, bool | int | complex):
        raise TypeError(f"{name} must be an exact float64 scalar")
    _assert_concrete(value)
    array = np.asarray(jax.device_get(value))
    if array.shape != () or array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be an exact float64 scalar")
    checked = np.float64(array)
    if not np.isfinite(checked) or checked < np.finfo(np.float64).tiny:
        raise ValueError(f"{name} must be finite positive normal float64")
    return checked


def _checked_pair_policy(value: object, name: str) -> int:
    """PRIVATE: Validate one positive signed-int64 pair policy.

    Parameters
    ----------
    value : object
        Candidate Python or NumPy integer.
    name : str
        Public policy name used in diagnostics.

    Returns
    -------
    checked : int
        Positive Python integer representable by signed int64.

    Raises
    ------
    TypeError
        If the candidate is boolean or not an integer.
    ValueError
        If the integer is outside positive signed-int64 range.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    checked = int(value)
    if checked <= 0 or checked > _MAXIMUM_SIGNED_INT64:
        raise ValueError(f"{name} must be positive signed int64")
    return checked


def _checked_scope(value: object) -> GalerkinLocalTerminalScope:
    """PRIVATE: Validate one source-stable complete-fiber scope.

    Parameters
    ----------
    value : object
        Candidate terminal-scope enum or exact enum value.

    Returns
    -------
    scope : GalerkinLocalTerminalScope
        Canonical full-state or selected-preterminal scope.

    Raises
    ------
    ValueError
        If ``value`` does not name one admitted scope.
    """
    return GalerkinLocalTerminalScope(value)


def _environment_payload() -> tuple[dict[str, object], bool, bool]:
    """PRIVATE: Record every relevant host arithmetic probe.

    Returns
    -------
    payload : dict[str, object]
        Canonical named arithmetic-probe values.
    host_supported : bool
        Whether exact binary64 host assumptions hold.
    normal_supported : bool
        Whether required normal-range directed primitives pass.
    """
    probes = arithmetic_environment_probes()
    values = tuple(bool(host_array(value)) for value in probes)
    host_supported = host_binary64_supported()
    normal_supported = bool(host_array(all_normal_arithmetic_supported()))
    payload: dict[str, object] = {
        "host_binary64_supported": host_supported,
        "normal_arithmetic_supported": normal_supported,
        "addition_supported": values[0],
        "multiplication_supported": values[1],
        "division_supported": values[2],
        "square_root_supported": values[3],
        "nextafter_supported": values[4],
        "bit_pattern_supported": values[5],
        "gradual_underflow_supported_diagnostic_only": values[6],
    }
    return payload, host_supported, normal_supported


def _finite_normal_or_zero(values: NDArray[np.float64]) -> bool:
    """PRIVATE: Check finite binary64 normal-range components or zeros.

    Parameters
    ----------
    values : NDArray[np.float64]
        Candidate binary64 scalar or array report.

    Returns
    -------
    valid : bool
        Whether every component is finite and normal or exactly zero.
    """
    array = np.asarray(values, dtype=np.float64)
    valid = bool(np.all(np.isfinite(array))) and bool(
        np.all((array == 0.0) | (np.abs(array) >= np.finfo(np.float64).tiny))
    )
    return valid  # noqa: RET504


def _scope_mapping(
    state_indices: NDArray[np.int64],
    transverse_indices: NDArray[np.int64],
    terminal_axis: int,
    scope: GalerkinLocalTerminalScope,
) -> _ScopeMapping:
    """PRIVATE: Derive canonical fibers, rows, and selected-state mask.

    Parameters
    ----------
    state_indices : NDArray[np.int64]
        Ordered retained-state integer indices.
    transverse_indices : NDArray[np.int64]
        Ordered selected complete transverse fibers.
    terminal_axis : int
        Physical xyz terminal axis.
    scope : GalerkinLocalTerminalScope
        Full-state or selected-preterminal complete-fiber scope.

    Returns
    -------
    mapping : _ScopeMapping
        Canonical transverse fibers, safe state-to-fiber rows, and exact
        scoped state-membership mask.
    """
    transverse_axes = _TRANSVERSE_AXES[terminal_axis]
    state = np.asarray(state_indices, dtype=np.int64)
    state_transverse = state[:, transverse_axes]
    fibers = (
        np.unique(state_transverse, axis=0)
        if scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
        else np.asarray(
            transverse_indices,
            dtype=np.int64,
        )
    )
    lookup = {
        tuple(int(component) for component in row): index
        for index, row in enumerate(fibers)
    }
    rows = np.zeros((state.shape[0],), dtype=np.int64)
    selected = np.zeros((state.shape[0],), dtype=np.bool_)
    for index, transverse in enumerate(state_transverse):
        row = lookup.get(tuple(int(component) for component in transverse))
        if row is not None:
            rows[index] = row
            selected[index] = True
    return fibers, rows, selected


def _direct_pair_count(
    rows: NDArray[np.int64],
    selected: NDArray[np.bool_],
) -> int:
    """PRIVATE: Count the sum of scoped fiber block squares exactly.

    Parameters
    ----------
    rows : NDArray[np.int64]
        Scoped-fiber row for every retained state.
    selected : NDArray[np.bool_]
        Exact scoped state-membership mask.

    Returns
    -------
    count : int
        Arbitrary-precision ``sum_h |I_u(h)|**2`` pair count.
    """
    return sum(
        int(np.count_nonzero(selected & (rows == row))) ** 2
        for row in np.unique(rows[selected])
    )


def _fraction_payload(value: Fraction) -> dict[str, str]:
    """PRIVATE: Serialize one reduced exact rational for hashing.

    Parameters
    ----------
    value : Fraction
        Exact reduced rational value.

    Returns
    -------
    payload : dict[str, str]
        Canonical signed-hex numerator and positive-hex denominator strings.
    """
    numerator_sign = "-" if value.numerator < 0 else "+"
    payload: dict[str, str] = {
        "numerator_hex": numerator_sign + format(abs(value.numerator), "x"),
        "denominator_hex": format(value.denominator, "x"),
    }
    return payload


def _rectangle_payload(rectangle: ComplexRectangle) -> list[dict[str, str]]:
    """PRIVATE: Serialize one exact complex rectangle for hashing.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Exact real-lower/upper and imaginary-lower/upper bounds.

    Returns
    -------
    payload : list[dict[str, str]]
        Four ordered exact-rational endpoint payloads.
    """
    return [_fraction_payload(value) for value in rectangle]


def _gram_rectangle(
    delta: Fraction,
    midpoint: Fraction,
    length: Fraction,
    p: int,
    q: int,
) -> ComplexRectangle:
    """PRIVATE: Construct one exact-rational LVT.55c Gram rectangle.

    Parameters
    ----------
    delta : Fraction
        Exact slab-width to terminal-length ratio.
    midpoint : Fraction
        Exact unwrapped slab midpoint coordinate.
    length : Fraction
        Exact positive terminal box length.
    p : int
        Left normal reciprocal integer.
    q : int
        Right normal reciprocal integer.

    Returns
    -------
    rectangle : ComplexRectangle
        Exact-rational enclosure of the LVT.55c entry.

    Raises
    ------
    RootEnclosureError
        If rational sinc or phase enclosure fails.
    """
    difference = q - p
    sinc_argument = Fraction(difference) * delta
    sinc = normalized_sinc_integer_ratio(
        sinc_argument.numerator,
        sinc_argument.denominator,
    )
    sinc_rectangle: ComplexRectangle = (
        sinc[0],
        sinc[1],
        Fraction(0),
        Fraction(0),
    )
    turn = Fraction(difference) * midpoint / length
    phase = rational_turn_exponential(-turn)
    return scale_complex_rectangle(
        complex_rectangle_multiply(sinc_rectangle, phase),
        delta,
    )


def _point_rectangle(value: np.complex128) -> ComplexRectangle:
    """PRIVATE: Embed one exact stored complex128 point as a rectangle.

    Parameters
    ----------
    value : np.complex128
        Exact stored complex state coefficient.

    Returns
    -------
    rectangle : ComplexRectangle
        Degenerate exact dyadic complex rectangle.
    """
    real = Fraction.from_float(float(np.real(value)))
    imag = Fraction.from_float(float(np.imag(value)))
    rectangle: ComplexRectangle = (real, real, imag, imag)
    return rectangle


def _real_rectangle(lower: Fraction, upper: Fraction) -> ComplexRectangle:
    """PRIVATE: Embed one exact real interval as a complex rectangle.

    Parameters
    ----------
    lower : Fraction
        Exact real lower endpoint.
    upper : Fraction
        Exact real upper endpoint.

    Returns
    -------
    rectangle : ComplexRectangle
        Real interval with an exact zero imaginary component.
    """
    rectangle: ComplexRectangle = (
        lower,
        upper,
        Fraction(0),
        Fraction(0),
    )
    return rectangle


def _magnitude_squared_upper(rectangle: ComplexRectangle) -> Fraction:
    """PRIVATE: Bound one complex rectangle's squared magnitude above.

    Parameters
    ----------
    rectangle : ComplexRectangle
        Exact rational complex rectangle.

    Returns
    -------
    upper : Fraction
        Nonnegative exact rational upper bound on squared magnitude.
    """
    real = max(abs(rectangle[0]), abs(rectangle[1]))
    imag = max(abs(rectangle[2]), abs(rectangle[3]))
    return real * real + imag * imag


def _fiber_rational_reports(
    indices: NDArray[np.int64],
    gram: _GramMatrix,
    free_intervals: list[tuple[Fraction, Fraction]],
    field: NDArray[np.complex128],
    state_radius: Fraction | None,
) -> _FiberRationalReports:
    """PRIVATE: Enclose one fiber's measurement, norm, and L6 lift.

    Parameters
    ----------
    indices : NDArray[np.int64]
        Ordered retained-state rows in one transverse fiber.
    gram : _GramMatrix
        Full block-diagonal exact rational Gram rectangles.
    free_intervals : list[tuple[Fraction, Fraction]]
        Exact free-diagonal intervals in retained-state order.
    field : NDArray[np.complex128]
        Exact stored submitted-state coefficients.
    state_radius : Fraction | None
        Replayed exact L6 state-radius upper, or ``None`` if unavailable.

    Returns
    -------
    reports : _FiberRationalReports
        Measured-squared lower/upper, measured upper, operator-squared upper,
        operator upper, state transfer upper, and total defect upper.

    Raises
    ------
    RootEnclosureError
        If the final Hermitian quadratic rectangle excludes reality or
        nonnegativity.
    """
    z_rectangles: dict[int, ComplexRectangle] = {}
    d_rectangles: dict[int, ComplexRectangle] = {}
    for index_value in indices:
        index = int(index_value)
        d_lower, d_upper = free_intervals[index]
        d_rectangle = _real_rectangle(d_lower, d_upper)
        d_rectangles[index] = d_rectangle
        z_rectangles[index] = complex_rectangle_multiply(
            d_rectangle,
            _point_rectangle(field[index]),
        )
    quadratic_terms = (
        complex_rectangle_multiply(
            complex_rectangle_multiply(
                conjugate_rectangle(z_rectangles[int(p)]),
                gram[int(p)][int(q)],
            ),
            z_rectangles[int(q)],
        )
        for p in indices
        for q in indices
    )
    quadratic = pairwise_rectangle_sum(quadratic_terms)
    if not quadratic[2] <= 0 <= quadratic[3]:
        raise RootEnclosureError(
            "LVT.55d quadratic imaginary interval excluded exact zero"
        )
    if quadratic[1] < 0:
        raise RootEnclosureError(
            "LVT.55d quadratic real upper excluded nonnegativity"
        )
    measured_squared_lower = max(Fraction(0), quadratic[0])
    measured_squared_upper = quadratic[1]
    measured_upper = sqrt_fraction_upper(measured_squared_upper)

    row_sums: list[Fraction] = []
    for p_value in indices:
        p = int(p_value)
        row_sum = Fraction(0)
        for q_value in indices:
            q = int(q_value)
            operator_entry = complex_rectangle_multiply(
                complex_rectangle_multiply(
                    d_rectangles[p],
                    gram[p][q],
                ),
                d_rectangles[q],
            )
            row_sum += sqrt_fraction_upper(
                _magnitude_squared_upper(operator_entry)
            )
        row_sums.append(row_sum)
    operator_squared_upper = max(row_sums, default=Fraction(0))
    operator_upper = sqrt_fraction_upper(operator_squared_upper)
    if state_radius is None:
        state_transfer_upper = Fraction(0)
        total_upper = Fraction(0)
    else:
        state_transfer_upper = operator_upper * state_radius
        total_upper = measured_upper + state_transfer_upper
    reports: _FiberRationalReports = (
        measured_squared_lower,
        measured_squared_upper,
        measured_upper,
        operator_squared_upper,
        operator_upper,
        state_transfer_upper,
        total_upper,
    )
    return reports


def _reports_to_binary64(
    reports: list[_FiberRationalReports],
    *,
    state_radius_available: bool,
) -> tuple[NDArray[np.float64], ...]:
    """PRIVATE: Convert exact per-fiber reports to outward binary64 arrays.

    Parameters
    ----------
    reports : list[_FiberRationalReports]
        Exact rational reports in canonical fiber order.
    state_radius_available : bool
        Whether transfer and total reports have an authenticated L6 radius.

    Returns
    -------
    converted : tuple[NDArray[np.float64], ...]
        Seven outward binary64 report arrays.
    """
    columns = list(zip(*reports, strict=True))
    converted_values: list[NDArray[np.float64]] = []
    for column_index, column in enumerate(columns):
        if column_index == 0:
            values = [fraction_lower_float(value) for value in column]
        elif (
            column_index >= _STATE_TRANSFER_REPORT_INDEX
            and not state_radius_available
        ):
            values = [math.inf for _ in column]
        else:
            values = [fraction_upper_float(value) for value in column]
        converted_values.append(np.asarray(values, dtype=np.float64))
    return tuple(converted_values)


def _fallback_evidence(
    state_size: int,
    fiber_size: int,
) -> tuple[_GramMatrix, tuple[NDArray[np.float64], ...]]:
    """PRIVATE: Build deterministic zero-Gram and infinite-report fallback.

    Parameters
    ----------
    state_size : int
        Retained target-state count.
    fiber_size : int
        Scoped transverse-fiber count.

    Returns
    -------
    gram : _GramMatrix
        Exact all-zero square Gram placeholder.
    reports : tuple[NDArray[np.float64], ...]
        Seven positive-infinity report arrays.
    """
    zero: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    gram = [[zero for _ in range(state_size)] for _ in range(state_size)]
    reports = tuple(
        np.full((fiber_size,), np.inf, dtype=np.float64) for _ in range(7)
    )
    return gram, reports


def _fallback_reports(
    fiber_size: int,
) -> tuple[NDArray[np.float64], ...]:
    """PRIVATE: Build deterministic unavailable per-fiber reports.

    Parameters
    ----------
    fiber_size : int
        Scoped transverse-fiber count.

    Returns
    -------
    reports : tuple[NDArray[np.float64], ...]
        Seven positive-infinity report arrays.
    """
    return tuple(
        np.full((fiber_size,), np.inf, dtype=np.float64) for _ in range(7)
    )


def _gram_binary64(gram: _GramMatrix) -> tuple[NDArray[np.float64], ...]:
    """PRIVATE: Convert exact Gram rectangles to outward binary64 matrices.

    Parameters
    ----------
    gram : _GramMatrix
        Full block-diagonal exact rational Gram rectangles.

    Returns
    -------
    converted : tuple[NDArray[np.float64], ...]
        Real-lower/upper and imaginary-lower/upper matrices.
    """
    size = len(gram)
    arrays = [np.zeros((size, size), dtype=np.float64) for _ in range(4)]
    for row in range(size):
        for column in range(size):
            rectangle = gram[row][column]
            arrays[0][row, column] = fraction_lower_float(rectangle[0])
            arrays[1][row, column] = fraction_upper_float(rectangle[1])
            arrays[2][row, column] = fraction_lower_float(rectangle[2])
            arrays[3][row, column] = fraction_upper_float(rectangle[3])
    return tuple(arrays)


def _gram_transcript_digest(
    gram: _GramMatrix,
    rows: NDArray[np.int64],
    selected: NDArray[np.bool_],
) -> str:
    """PRIVATE: Digest every exact rational scoped Gram entry.

    Parameters
    ----------
    gram : _GramMatrix
        Full block-diagonal exact rational Gram rectangles.
    rows : NDArray[np.int64]
        Scoped-fiber row for every retained state.
    selected : NDArray[np.bool_]
        Exact scoped state-membership mask.

    Returns
    -------
    digest : str
        Exact-rational Gram transcript digest.
    """
    entries: list[dict[str, object]] = []
    state_size = len(gram)
    for p in range(state_size):
        for q in range(state_size):
            if selected[p] and selected[q] and rows[p] == rows[q]:
                entries.append(
                    {
                        "p": p,
                        "q": q,
                        "fiber_row": int(rows[p]),
                        "rectangle": _rectangle_payload(gram[p][q]),
                    }
                )
    return sha256(
        {
            "domain": _GRAM_TRANSCRIPT_DOMAIN,
            "integer_encoding": "signed_hex_magnitude.v1",
            "entries": entries,
            "precision": _PRECISION_TRANSCRIPT,
        }
    )


def _projection_identity_digest(
    zero_slab: GalerkinLocalZeroSlabCertificate,
    stability: GalerkinLocalStabilityResult,
    scope: GalerkinLocalTerminalScope,
    fibers: NDArray[np.int64],
) -> str:
    """PRIVATE: Bind slab, scope, target, source, and submitted state.

    Parameters
    ----------
    zero_slab : GalerkinLocalZeroSlabCertificate
        Fully replayed exact zero-slab parent.
    stability : GalerkinLocalStabilityResult
        Fully replayed same-source stability parent.
    scope : GalerkinLocalTerminalScope
        Complete transverse-fiber scope.
    fibers : NDArray[np.int64]
        Canonical scoped transverse indices.

    Returns
    -------
    digest : str
        Canonical projection identity digest.
    """
    source = zero_slab.represented_source_certificate.source
    return sha256(
        {
            "domain": _IDENTITY_DIGEST_DOMAIN,
            "target_digest": source.target.target_digest,
            "source_digest": source.source_digest,
            "slab_digest": zero_slab.slab_digest,
            "state_identity_digest": stability.result_identity_digest,
            "scope": scope.value,
            "fibers": stored_value_payload(fibers),
        }
    )


def _certificate_digest(  # noqa: PLR0913
    zero_slab: GalerkinLocalZeroSlabCertificate,
    stability: GalerkinLocalStabilityResult,
    projection_identity_digest: str,
    direct_pair_count_exact: str,
    mapping: _ScopeMapping,
    free_evidence: _FreeEvidence,
    fiber_zero: NDArray[np.bool_],
    gram_evidence: tuple[NDArray[np.float64], ...],
    fiber_reports: tuple[NDArray[np.float64], ...],
    policies: tuple[object, ...],
    predicates: tuple[bool, ...],
    failure: GalerkinLocalProjectionDefectFailure,
    environment_payload: dict[str, object],
    gram_transcript_digest: str,
) -> str:
    """PRIVATE: Digest complete parents, policies, and projection evidence.

    Parameters
    ----------
    zero_slab : GalerkinLocalZeroSlabCertificate
        Fully replayed exact zero-slab parent.
    stability : GalerkinLocalStabilityResult
        Fully replayed same-source stability parent.
    projection_identity_digest : str
        Slab/scope/target/source/state identity digest.
    direct_pair_count_exact : str
        Arbitrary-precision decimal projection-work transcript.
    mapping : _ScopeMapping
        Fibers, state rows, and selected-state mask.
    free_evidence : _FreeEvidence
        Exact-D lower/upper intervals and singleton-zero mask.
    fiber_zero : NDArray[np.bool_]
        Per-fiber structural exact-zero mask.
    gram_evidence : tuple[NDArray[np.float64], ...]
        Four outward binary64 Gram matrices.
    fiber_reports : tuple[NDArray[np.float64], ...]
        Seven per-fiber measurement/norm/lift reports.
    policies : tuple[object, ...]
        State radius, three independent policies, and work/precision values.
    predicates : tuple[bool, ...]
        Host, arithmetic, structural, finite, and operational predicates.
    failure : GalerkinLocalProjectionDefectFailure
        Simultaneous typed outcome.
    environment_payload : dict[str, object]
        Complete named arithmetic-probe values.
    gram_transcript_digest : str
        Exact-rational Gram transcript digest.

    Returns
    -------
    digest : str
        Complete projection-defect evidence digest.
    """
    return sha256(
        {
            "domain": _CERTIFICATE_DIGEST_DOMAIN,
            "zero_slab": stored_value_payload(zero_slab),
            "stability": stored_value_payload(stability),
            "projection_identity_digest": projection_identity_digest,
            "direct_pair_count_exact": direct_pair_count_exact,
            "mapping": stored_value_payload(mapping),
            "free_evidence": stored_value_payload(free_evidence),
            "fiber_zero": stored_value_payload(fiber_zero),
            "gram_evidence": stored_value_payload(gram_evidence),
            "fiber_reports": stored_value_payload(fiber_reports),
            "policies": stored_value_payload(policies),
            "predicates": stored_value_payload(predicates),
            "failure_mask": int(failure),
            "environment": environment_payload,
            "arithmetic_environment_digest": sha256(environment_payload),
            "gram_transcript_digest": gram_transcript_digest,
            "gram_formula": _GRAM_FORMULA,
            "measurement_formula": _MEASUREMENT_FORMULA,
            "operator_bound_formula": _OPERATOR_BOUND_FORMULA,
            "state_lift_formula": _STATE_LIFT_FORMULA,
            "precision": _PRECISION_TRANSCRIPT,
            "error_scope": _ERROR_SCOPE,
            "completion_scope": _COMPLETION_SCOPE,
        }
    )


def _certify_prepared(  # noqa: PLR0912,PLR0915
    zero_slab: GalerkinLocalZeroSlabCertificate,
    stability: GalerkinLocalStabilityResult,
    scope: GalerkinLocalTerminalScope,
    maximum_state_error: np.float64,
    maximum_stability_direct_pairs: int,
    maximum_gram_pairs: int,
) -> GalerkinLocalProjectionDefectCertificate:
    """PRIVATE: Certify projection evidence from two replayed parents.

    Parameters
    ----------
    zero_slab : GalerkinLocalZeroSlabCertificate
        Fully replayed exact zero-slab parent or typed noncertificate.
    stability : GalerkinLocalStabilityResult
        Fully replayed local-stability parent.
    scope : GalerkinLocalTerminalScope
        Complete transverse-fiber scope.
    maximum_state_error : np.float64
        Independently supplied exact-float64 L6 operational policy.
    maximum_stability_direct_pairs : int
        Independently supplied signed-int64 L6 replay-work policy.
    maximum_gram_pairs : int
        Independently supplied signed-int64 projection block-pair policy.

    Returns
    -------
    certificate : GalerkinLocalProjectionDefectCertificate
        Canonical projection-defect certificate or typed noncertificate.
    """
    represented = zero_slab.represented_source_certificate
    source = represented.source
    target = source.target
    axis = target.acquisition.terminal_axis
    fibers, rows, selected = _scope_mapping(
        np.asarray(jax.device_get(target.state_indices), dtype=np.int64),
        np.asarray(
            jax.device_get(target.acquisition.transverse_indices),
            dtype=np.int64,
        ),
        axis,
        scope,
    )
    state_size = target.state_indices.shape[0]
    fiber_size = fibers.shape[0]
    free_lower = np.asarray(
        jax.device_get(
            target.fixed_linear_error_ledger.exact_free_diagonal_lower_bounds
        ),
        dtype=np.float64,
    )
    free_upper = np.asarray(
        jax.device_get(
            target.fixed_linear_error_ledger.exact_free_diagonal_upper_bounds
        ),
        dtype=np.float64,
    )
    state_zero = np.asarray(
        (free_lower == 0.0) & (free_upper == 0.0),
        dtype=np.bool_,
    )
    fiber_zero = np.asarray(
        [
            bool(np.any(selected & (rows == row)))
            and bool(np.all(state_zero[selected & (rows == row)]))
            for row in range(fiber_size)
        ],
        dtype=np.bool_,
    )
    structural = bool(np.all(fiber_zero))
    exact_count = _direct_pair_count(rows, selected)
    stored_count = exact_count if exact_count <= _MAXIMUM_SIGNED_INT64 else 0
    environment, host_supported, normal_supported = _environment_payload()
    same_parent = stored_value_payload(represented) == stored_value_payload(
        stability.certificate
    )
    state_radius_eligible = bool(stability.proof.state_radius_eligible)
    state_radius_report = np.asarray(
        stability.proof.state_radius_upper_bound,
        dtype=np.float64,
    )
    try:
        state_radius_transcript = Fraction(
            stability.proof.state_radius_upper_numerator,
            stability.proof.state_radius_upper_denominator,
        )
        state_radius_transcript_matches = (
            state_radius_transcript
            == Fraction.from_float(float(state_radius_report))
        )
    except (OverflowError, ValueError, ZeroDivisionError):
        state_radius_transcript_matches = False
    state_radius_range_failure = state_radius_eligible and (
        not _finite_normal_or_zero(state_radius_report)
        or not state_radius_transcript_matches
    )
    scope_complete = (
        scope is GalerkinLocalTerminalScope.FULL_STATE_FIBERS
        or bool(target.support_eligibility.terminal_fiber_complete)
    )

    failure = GalerkinLocalProjectionDefectFailure.NONE
    parent_mismatch_reason = (
        GalerkinLocalProjectionDefectFailure.PARENT_SOURCE_EVIDENCE_MISMATCH
    )
    operational_reason = (
        GalerkinLocalProjectionDefectFailure.OPERATIONAL_STATE_BUDGET_MISSED
    )
    structural_reason = (
        GalerkinLocalProjectionDefectFailure.STRUCTURAL_EXACT_ZERO_UNAVAILABLE
    )
    if not bool(zero_slab.terminal_zero_slab_eligible):
        failure |= (
            GalerkinLocalProjectionDefectFailure.ZERO_SLAB_NONCERTIFICATE
        )
    if not same_parent:
        failure |= parent_mismatch_reason
    if not state_radius_eligible:
        failure |= (
            GalerkinLocalProjectionDefectFailure.STATE_RADIUS_UNAVAILABLE
        )
    if bool(stability.proof.state_radius_eligible) and not bool(
        stability.proof.operational_state_eligible
    ):
        failure |= operational_reason
    if not scope_complete:
        failure |= (
            GalerkinLocalProjectionDefectFailure.TERMINAL_SCOPE_INCOMPLETE
        )
    if not structural:
        failure |= structural_reason
    if not host_supported or not normal_supported:
        failure |= (
            GalerkinLocalProjectionDefectFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
    if exact_count > _MAXIMUM_SIGNED_INT64:
        failure |= (
            GalerkinLocalProjectionDefectFailure.GRAM_PAIR_COUNT_OVERFLOW
        )
    elif exact_count > maximum_gram_pairs:
        failure |= (
            GalerkinLocalProjectionDefectFailure.GRAM_PAIR_BUDGET_EXCEEDED
        )
    if state_radius_range_failure:
        failure |= (
            GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
        )

    fatal_preflight = (
        not same_parent
        or not scope_complete
        or not host_supported
        or not normal_supported
        or state_radius_range_failure
        or exact_count > maximum_gram_pairs
        or exact_count > _MAXIMUM_SIGNED_INT64
    )
    gram, reports = _fallback_evidence(state_size, fiber_size)
    arithmetic_completed = False
    if not fatal_preflight:
        try:
            length = Fraction.from_float(
                float(np.asarray(target.local_potential.box_size[axis]))
            )
            lower = Fraction(
                int(zero_slab.slab_lower_numerator),
                int(zero_slab.slab_lower_denominator),
            )
            upper = Fraction(
                int(zero_slab.slab_upper_numerator),
                int(zero_slab.slab_upper_denominator),
            )
            delta = (upper - lower) / length
            midpoint = (lower + upper) / 2
            normal_indices = np.asarray(
                jax.device_get(target.state_indices[:, axis]),
                dtype=np.int64,
            )
            zero: ComplexRectangle = (
                Fraction(0),
                Fraction(0),
                Fraction(0),
                Fraction(0),
            )
            gram = [
                [zero for _ in range(state_size)] for _ in range(state_size)
            ]
            rational_reports: list[_FiberRationalReports] = []
            field = np.asarray(
                jax.device_get(stability.solve_result.field),
                dtype=np.complex128,
            )
            free_intervals = [
                (
                    Fraction.from_float(float(free_lower[index])),
                    Fraction.from_float(float(free_upper[index])),
                )
                for index in range(state_size)
            ]
            state_radius = (
                Fraction(
                    stability.proof.state_radius_upper_numerator,
                    stability.proof.state_radius_upper_denominator,
                )
                if bool(stability.proof.state_radius_eligible)
                else None
            )
            for fiber_row in range(fiber_size):
                indices = np.flatnonzero(
                    selected & (rows == fiber_row)
                ).astype(np.int64)
                for p_offset, p_value in enumerate(indices):
                    p_index = int(p_value)
                    for q_value in indices[p_offset:]:
                        q_index = int(q_value)
                        rectangle = _gram_rectangle(
                            delta,
                            midpoint,
                            length,
                            int(normal_indices[p_index]),
                            int(normal_indices[q_index]),
                        )
                        gram[p_index][q_index] = rectangle
                        gram[q_index][p_index] = conjugate_rectangle(rectangle)
                rational_reports.append(
                    _fiber_rational_reports(
                        indices,
                        gram,
                        free_intervals,
                        field,
                        state_radius,
                    )
                )
            reports = _reports_to_binary64(
                rational_reports,
                state_radius_available=state_radius is not None,
            )
            arithmetic_completed = True
        except (AssertionError, RootEnclosureError):
            failure |= (
                GalerkinLocalProjectionDefectFailure.ROOT_ENCLOSURE_FAILURE
            )
            gram, reports = _fallback_evidence(state_size, fiber_size)
        except (OverflowError, ValueError, ZeroDivisionError):
            failure |= (
                GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
            )
            gram, reports = _fallback_evidence(state_size, fiber_size)

    gram_binary64 = _gram_binary64(gram)
    if any(
        not np.all(np.isfinite(value))
        or bool(has_subnormal_components(jnp.asarray(value)))
        for value in gram_binary64
    ):
        failure |= (
            GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
        )
        gram, reports = _fallback_evidence(state_size, fiber_size)
        gram_binary64 = _gram_binary64(gram)
        arithmetic_completed = False
    if arithmetic_completed:
        finite_report_count = (
            len(reports)
            if state_radius_eligible
            else _STATE_TRANSFER_REPORT_INDEX
        )
        if any(
            not _finite_normal_or_zero(value)
            for value in reports[:finite_report_count]
        ):
            failure |= (
                GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
            )
            reports = _fallback_reports(fiber_size)
    gram_digest = _gram_transcript_digest(gram, rows, selected)
    fatal_reasons = (
        GalerkinLocalProjectionDefectFailure.ZERO_SLAB_NONCERTIFICATE
        | GalerkinLocalProjectionDefectFailure.PARENT_SOURCE_EVIDENCE_MISMATCH
        | GalerkinLocalProjectionDefectFailure.STATE_RADIUS_UNAVAILABLE
        | GalerkinLocalProjectionDefectFailure.TERMINAL_SCOPE_INCOMPLETE
        | GalerkinLocalProjectionDefectFailure.HOST_ARITHMETIC_UNSUPPORTED
        | GalerkinLocalProjectionDefectFailure.GRAM_PAIR_BUDGET_EXCEEDED
        | GalerkinLocalProjectionDefectFailure.GRAM_PAIR_COUNT_OVERFLOW
        | GalerkinLocalProjectionDefectFailure.ROOT_ENCLOSURE_FAILURE
        | GalerkinLocalProjectionDefectFailure.ARITHMETIC_RANGE_FAILURE
    )
    finite = not bool(failure & fatal_reasons)
    operational = finite and bool(stability.proof.operational_state_eligible)
    predicates = (
        host_supported,
        normal_supported,
        structural,
        finite,
        operational,
    )
    identity = _projection_identity_digest(
        zero_slab,
        stability,
        scope,
        fibers,
    )
    mapping = (fibers, rows, selected)
    free_evidence = (free_lower, free_upper, state_zero)
    policies: tuple[object, ...] = (
        state_radius_report,
        np.asarray(maximum_state_error, dtype=np.float64),
        np.asarray(stored_count, dtype=np.int64),
        np.asarray(maximum_gram_pairs, dtype=np.int64),
        np.asarray(maximum_stability_direct_pairs, dtype=np.int64),
        np.asarray(_PI_TARGET_BITS, dtype=np.int64),
        np.asarray(_SINE_TAYLOR_LOWER_LAST_INDEX, dtype=np.int64),
        np.asarray(_SINE_TAYLOR_UPPER_LAST_INDEX, dtype=np.int64),
        np.asarray(_SQRT_PRECISION_BITS, dtype=np.int64),
    )
    certificate_digest = _certificate_digest(
        zero_slab,
        stability,
        identity,
        str(exact_count),
        mapping,
        free_evidence,
        fiber_zero,
        gram_binary64,
        reports,
        policies,
        predicates,
        failure,
        environment,
        gram_digest,
    )
    stopped = jax.tree.map(
        jax.lax.stop_gradient,
        (
            *(jnp.asarray(value) for value in mapping),
            *(jnp.asarray(value) for value in free_evidence),
            jnp.asarray(fiber_zero),
            *(jnp.asarray(value) for value in gram_binary64),
            *(jnp.asarray(value) for value in reports),
            *(jnp.asarray(value) for value in policies),
            *(jnp.asarray(value) for value in predicates),
            jnp.asarray(int(failure), dtype=jnp.int64),
        ),
    )
    state_evidence = (
        stopped[1],
        stopped[2],
        stopped[3],
        stopped[4],
        stopped[5],
    )
    gram_evidence = (stopped[7], stopped[8], stopped[9], stopped[10])
    fiber_evidence = (
        stopped[6],
        stopped[11],
        stopped[12],
        stopped[13],
        stopped[14],
        stopped[15],
        stopped[16],
        stopped[17],
    )
    policy_evidence = tuple(stopped[index] for index in range(18, 27))
    eligibility_evidence = tuple(stopped[index] for index in range(27, 32))
    maximum_state_error_fraction = Fraction.from_float(
        float(maximum_state_error)
    )
    certificate = _make_local_projection_defect_certificate(
        zero_slab,
        stability,
        stopped[0],
        state_evidence,
        gram_evidence,
        fiber_evidence,
        policy_evidence,
        eligibility_evidence,
        stopped[32],
        terminal_axis=axis,
        projection_scope=scope,
        maximum_state_error_numerator=(maximum_state_error_fraction.numerator),
        maximum_state_error_denominator=(
            maximum_state_error_fraction.denominator
        ),
        direct_pair_count_exact=str(exact_count),
        gram_formula=_GRAM_FORMULA,
        measurement_formula=_MEASUREMENT_FORMULA,
        operator_bound_formula=_OPERATOR_BOUND_FORMULA,
        state_lift_formula=_STATE_LIFT_FORMULA,
        precision_transcript=_PRECISION_TRANSCRIPT,
        error_scope=_ERROR_SCOPE,
        completion_scope=_COMPLETION_SCOPE,
        target_digest=target.target_digest,
        parent_target_evidence_digest=target.manifest_evidence_digest,
        source_digest=source.source_digest,
        parent_source_evidence_digest=source.source_evidence_digest,
        parent_represented_certificate_digest=represented.certificate_digest,
        parent_zero_slab_certificate_digest=zero_slab.certificate_digest,
        parent_stability_result_identity_digest=(
            stability.result_identity_digest
        ),
        parent_stability_result_evidence_digest=(
            stability.result_evidence_digest
        ),
        state_identity_digest=stability.result_identity_digest,
        projection_identity_digest=identity,
        arithmetic_environment_digest=sha256(environment),
        gram_transcript_digest=gram_digest,
        certificate_digest=certificate_digest,
    )
    return certificate  # noqa: RET504


def enclose_local_projection_defect(
    zero_slab_certificate: GalerkinLocalZeroSlabCertificate,
    stability_result: GalerkinLocalStabilityResult,
    *,
    scope: GalerkinLocalTerminalScope | str,
    maximum_state_error: object,
    maximum_stability_direct_pairs: int = (
        _DEFAULT_MAXIMUM_STABILITY_DIRECT_PAIRS
    ),
    maximum_gram_pairs: int = _DEFAULT_MAXIMUM_GRAM_PAIRS,
) -> GalerkinLocalProjectionDefectCertificate:
    """Enclose scoped LVT.34--LVT.40 and LVT.55c--LVT.55e evidence.

    :see: :func:`~.test_local_projection.\
test_projection_public_replay_binds_full_source_state_and_policies`

    Both parents cross their complete public replay boundaries.  The state
    budget and L6 direct-work policy are supplied independently for stability
    replay; the separate Gram-pair policy bounds only this leaf's
    ``sum_h |I_u(h)|**2`` exact-rational construction.

    Parameters
    ----------
    zero_slab_certificate : GalerkinLocalZeroSlabCertificate
        Public exact zero-slab certificate to replay.
    stability_result : GalerkinLocalStabilityResult
        Public same-source local-stability result to replay.
    scope : GalerkinLocalTerminalScope | str
        Full-state or selected-preterminal complete transverse fibers.
    maximum_state_error : object
        Independent finite positive normal exact-float64 L6 state policy.
    maximum_stability_direct_pairs : int, optional
        Independent signed-int64 L6 replay-work policy; defaults to 2000000.
    maximum_gram_pairs : int, optional
        Signed-int64 projection Gram-pair policy; defaults to 2000000.

    Returns
    -------
    certificate : GalerkinLocalProjectionDefectCertificate
        Canonical structural and finite projection-defect evidence.

    Raises
    ------
    TypeError
        If either parent or a policy has the wrong carrier/storage type.
    ValueError
        If parent replay, scope, policy, or exact carrier structure fails.
    """
    if not isinstance(zero_slab_certificate, GalerkinLocalZeroSlabCertificate):
        raise TypeError("zero_slab_certificate has the wrong carrier type")
    if not isinstance(stability_result, GalerkinLocalStabilityResult):
        raise TypeError("stability_result has the wrong carrier type")
    _assert_concrete((zero_slab_certificate, stability_result))
    checked_scope = _checked_scope(scope)
    state_budget = _checked_positive_binary64(
        maximum_state_error,
        "maximum_state_error",
    )
    stability_pairs = _checked_pair_policy(
        maximum_stability_direct_pairs,
        "maximum_stability_direct_pairs",
    )
    gram_pairs = _checked_pair_policy(
        maximum_gram_pairs,
        "maximum_gram_pairs",
    )
    zero_slab = prepare_local_zero_slab_certificate(zero_slab_certificate)
    stability = prepare_local_galerkin_stability_result(
        stability_result,
        maximum_state_error=state_budget,
        maximum_direct_pairs=stability_pairs,
    )
    certificate = _certify_prepared(
        zero_slab,
        stability,
        checked_scope,
        state_budget,
        stability_pairs,
        gram_pairs,
    )
    return certificate  # noqa: RET504


def prepare_local_projection_defect_certificate(
    certificate: GalerkinLocalProjectionDefectCertificate,
    *,
    maximum_state_error: object,
    maximum_stability_direct_pairs: int = (
        _DEFAULT_MAXIMUM_STABILITY_DIRECT_PAIRS
    ),
    maximum_gram_pairs: int = _DEFAULT_MAXIMUM_GRAM_PAIRS,
) -> GalerkinLocalProjectionDefectCertificate:
    """Replay both parents, policies, exact Gram arithmetic, and all digests.

    :see: :func:`~.test_local_projection.\
test_projection_public_replay_binds_full_source_state_and_policies`

    Parameters
    ----------
    certificate : GalerkinLocalProjectionDefectCertificate
        Public projection-defect certificate to authenticate in full.
    maximum_state_error : object
        Independent finite positive normal exact-float64 L6 state policy.
    maximum_stability_direct_pairs : int, optional
        Independent signed-int64 L6 replay-work policy; defaults to 2000000.
    maximum_gram_pairs : int, optional
        Signed-int64 projection Gram-pair policy; defaults to 2000000.

    Returns
    -------
    canonical : GalerkinLocalProjectionDefectCertificate
        Fresh certificate reconstructed from authenticated primitive inputs.

    Raises
    ------
    TypeError
        If the submitted object or a policy has the wrong type.
    ValueError
        If complete parent, policy, arithmetic, or digest replay differs.
    """
    if not isinstance(certificate, GalerkinLocalProjectionDefectCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalProjectionDefectCertificate"
        )
    _assert_concrete(certificate)
    canonical = enclose_local_projection_defect(
        certificate.zero_slab_certificate,
        certificate.stability_result,
        scope=certificate.projection_scope,
        maximum_state_error=maximum_state_error,
        maximum_stability_direct_pairs=maximum_stability_direct_pairs,
        maximum_gram_pairs=maximum_gram_pairs,
    )
    if stored_value_payload(canonical) != stored_value_payload(certificate):
        raise ValueError(
            "projection-defect certificate does not match complete replay"
        )
    return canonical


__all__: list[str] = [
    "enclose_local_projection_defect",
    "prepare_local_projection_defect_certificate",
]
