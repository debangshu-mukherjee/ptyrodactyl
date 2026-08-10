r"""Certify and apply the LVT-1 fixed local-cell interaction core.

Extended Summary
----------------
This host boundary authenticates direct LVT.13 evidence, independently
rebuilds the full product support, proves exact LVT.15 state-difference
coverage, transfers exact SC.4 coupling to interaction rectangles, and
computes the LVT.18 fixed-operator enclosure.  A separate accepted core
exposes the rounded interaction action and its formal reverse-conjugate matrix
adjoint.  No free diagonal, absorber, source, terminal, or solver target is
constructed here.

Routine Listings
----------------
:func:`apply_local_cell_interaction`
    Apply the frozen rounded LVT.16 interaction action.
:func:`apply_local_cell_interaction_adjoint`
    Apply the formal matrix adjoint of the frozen rounded interaction.
:func:`certify_local_cell_exact_compression`
    Certify exact local-cell compression and its fixed interaction error.
:func:`create_local_cell_interaction_core`
    Create a non-solver-ready core from finite replayed LVT evidence.
:func:`prepare_local_cell_interaction_core`
    Replay-authenticate stored core data before transform-compatible use.
"""

from __future__ import annotations

import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Dict, Tuple
from jax.core import Tracer
from jaxtyping import (
    Array,
    Complex,
    Complex128,
    Float64,
    Int64,
    Shaped,
    jaxtyped,
)
from numpy.typing import NDArray

from ptyrodactyl._canonical_digest import _sha256, _stored_value_payload
from ptyrodactyl._host_interval import (
    _fraction_from_float,
    _fraction_lower_float,
    _fraction_upper_float,
    _host_binary64_supported,
    _real_interval_product,
    _RootEnclosureError,
    _sqrt_fraction_upper,
)
from ptyrodactyl._numeric import has_subnormal_components
from ptyrodactyl._physics import coupled_interaction_value
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M_E,
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellCompressionFailure,
    GalerkinLocalCellErrorRoute,
    GalerkinLocalCellExactCompression,
    GalerkinLocalCellInteractionCore,
    GalerkinLocalCellPotentialRealization,
    GalerkinProductSupport,
    _make_local_cell_exact_compression,
    _make_local_cell_interaction_core,
    create_galerkin_product_support,
)

from .local_cell_certification import _authenticate_local_cell_certificate

_ACTION_ROUTE: str = "LVT.16 direct ordered-Du rounded interaction v1"
_ADJOINT_ROUTE: str = (
    "formal reverse-conjugate adjoint of frozen LVT.16 algebraic matrix v1"
)
_CERTIFICATE_DOMAIN: str = "ptyrodactyl.local_cell.lvt14_lvt18_certificate.v1"
_COMPLETION_SCOPE: str = (
    "interaction_core_only; no free diagonal, CAP, source, terminal, or H"
)
_COMPRESSION_CLAIM: str = (
    "LVT.15 exact difference coverage; LVT.16 exact local compression; "
    "LVT.18 fixed interaction error"
)
_COUPLING_TARGET: str = (
    "SC.4 positive rational point from exact stored binary64 U0/M_E/"
    "E_CHARGE/C_LIGHT/HBAR and mathematical 10^-20; stored bounds are only "
    "its outward binary64 enclosure"
)
_DEFAULT_MAXIMUM_HOST_ARRAY_WORKING_SET_BYTES: int = 2_000_000_000
_DEFAULT_MAXIMUM_INTERACTION_MODES: int = 2_000_000
_DEFAULT_MAXIMUM_STATE_PAIRS: int = 20_000_000
_DEFAULT_MAXIMUM_WORK_GRID_POINTS: int = 200_000_000
_DIFFERENCE_COUNT_ROUTE: str = (
    "streamed Python-int ordered Iu x Iu differences; lexicographic Du; "
    "exact Ichi positions and multiplicities v1"
)
_EXACT_TARGET: str = "LVT.14-LVT.18 finite local-cell interaction on fixed Iu"
_INTERACTION_REALIZATION_ROUTE: str = (
    "existing coupled_interaction_value H_PLANCK/pi route with canonical "
    "50-mantissa-bit sigma and componentwise chi rounding"
)
_HOST_TRANSIENT_SCALAR_SCOPE: str = (
    "post-replay L3 construction only: O(1) exact-rational scalar objects "
    "per processed mode; nested L2 authentication, product-support factory "
    "replay, Python runtime, allocator, device, and total-process RSS "
    "overhead excluded"
)
_MAXIMUM_SIGNED_INT64: int = np.iinfo(np.int64).max
_MINIMUM_SIGNED_INT64: int = np.iinfo(np.int64).min
_OPERATOR_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_cell.lvt16_interaction_operator.v1"
)
_OPERATOR_ERROR_SCOPE: str = (
    "fixed rounded R minus exact LVT.16 interaction only"
)
_PER_CALL_ARITHMETIC_EXCLUSION: str = (
    "excludes every per-call multiply, accumulation, transform, and output "
    "rounding error"
)
_RESOURCE_COUNT: int = 4
_SPACE_DIMENSIONS: int = 3
_SUPPORT_RANK: int = 2
_ZERO_DIGEST: str = "0" * 64

type _HostComplex = Complex128[NDArray, " p"]
type _HostDifferenceFloat = Float64[NDArray, " d"]
type _HostIndices = Int64[NDArray, "n 3"]
type _HostInteractionIndices = Int64[NDArray, "p 3"]
type _HostStateIndices = Int64[NDArray, "n 3"]
type _DifferenceEvidence = Tuple[
    Int64[NDArray, "d 3"],
    Int64[NDArray, " d"],
    Int64[NDArray, " d"],
    Int64[NDArray, " s"],
    GalerkinLocalCellCompressionFailure,
]


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject JAX tracers at the explicit host boundary.

    Parameters
    ----------
    value : object
        Submitted PyTree or scalar.

    Raises
    ------
    ValueError
        If any dynamic leaf is a tracer.
    """
    if any(
        isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value)
    ):
        raise ValueError("local-cell exact compression requires host values")


def _checked_budget(value: int, name: str) -> int:
    """PRIVATE: Validate one positive signed-64-bit host budget.

    Parameters
    ----------
    value : int
        Submitted budget.
    name : str
        Field name used in rejection messages.

    Returns
    -------
    checked : int
        Validated positive Python integer.

    Raises
    ------
    ValueError
        If the budget is boolean, nonintegral, nonpositive, or too large.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > _MAXIMUM_SIGNED_INT64
    ):
        raise ValueError(f"{name} must be a positive signed-64-bit integer")
    checked: int = value
    return checked


def _host_exact_array(
    value: jax.Array,
    dtype: np.dtype,
    name: str,
) -> Shaped[NDArray, "..."]:
    """PRIVATE: Transfer an array while preserving exact dtype and shape.

    Parameters
    ----------
    value : jax.Array
        Concrete array leaf.
    dtype : np.dtype
        Required NumPy dtype.
    name : str
        Field name used in rejection messages.

    Returns
    -------
    array : Shaped[NDArray, "..."]
        Concrete host array with the required exact dtype.

    Raises
    ------
    ValueError
        If the stored dtype differs.
    """
    array: Shaped[NDArray, "..."] = np.asarray(jax.device_get(value))
    if array.dtype != dtype:
        raise ValueError(f"{name} must have exact {dtype.name} dtype")
    return array


def _host_exact_scalar(
    value: jax.Array,
    dtype: np.dtype,
    name: str,
) -> bool | float | int:
    """PRIVATE: Transfer one exact-dtype scalar.

    Parameters
    ----------
    value : jax.Array
        Concrete scalar leaf.
    dtype : np.dtype
        Required NumPy dtype.
    name : str
        Field name used in rejection messages.

    Returns
    -------
    scalar : bool | float | int
        Exact stored scalar converted to its corresponding Python type.

    Raises
    ------
    ValueError
        If dtype or shape differs.
    """
    array = _host_exact_array(value, dtype, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    scalar: bool | float | int = array.item()
    return scalar


def _checked_support_matrix(value: jax.Array, name: str) -> _HostIndices:
    """PRIVATE: Validate one exact int64 reciprocal-index matrix.

    Parameters
    ----------
    value : jax.Array
        Submitted support array.
    name : str
        Field name used in rejection messages.

    Returns
    -------
    indices : _HostIndices
        Exact nonempty host matrix with shape ``(n, 3)``.

    Raises
    ------
    ValueError
        If dtype, rank, shape, or size differs.
    """
    raw = _host_exact_array(value, np.dtype(np.int64), name)
    if raw.ndim != _SUPPORT_RANK or raw.shape[1:] != (_SPACE_DIMENSIONS,):
        raise ValueError(f"{name} must have shape (n, 3)")
    if raw.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty")
    indices: _HostIndices = raw
    return indices


def _rebuild_product_support(
    realization: GalerkinLocalCellPotentialRealization,
) -> GalerkinProductSupport:
    """PRIVATE: Rebuild all RM-S2 product predicates from primitive arrays.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Fully replay-authenticated L2 realization.

    Returns
    -------
    rebuilt : GalerkinProductSupport
        Fresh product support with injectivity, inclusion, and no-alias checks.

    Raises
    ------
    ValueError
        If primitive storage or its exact replay differs.
    equinox.EquinoxRuntimeError
        If a full product predicate fails.
    """
    submitted = realization.support_eligibility.manifest.support
    state = _checked_support_matrix(submitted.state_indices, "state_indices")
    interaction = _checked_support_matrix(
        submitted.interaction_indices, "interaction_indices"
    )
    absorber = _checked_support_matrix(
        submitted.absorber_indices, "absorber_indices"
    )
    work = _checked_support_matrix(submitted.work_indices, "work_indices")
    work_shape = submitted.work_shape
    if (
        not isinstance(work_shape, tuple)
        or len(work_shape) != _SPACE_DIMENSIONS
        or any(
            isinstance(size, bool) or not isinstance(size, int)
            for size in work_shape
        )
    ):
        raise ValueError("work_shape must be one exact integer xyz tuple")
    rebuilt: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=jnp.asarray(state, dtype=jnp.int64),
        interaction_indices=jnp.asarray(interaction, dtype=jnp.int64),
        absorber_indices=jnp.asarray(absorber, dtype=jnp.int64),
        work_indices=jnp.asarray(work, dtype=jnp.int64),
        work_shape=work_shape,
    )
    jax.block_until_ready(rebuilt)
    if _stored_value_payload(rebuilt) != _stored_value_payload(submitted):
        raise ValueError("product support does not match full factory replay")
    return rebuilt


def _resource_counts(
    support: GalerkinProductSupport,
) -> Tuple[int, int, int, int]:
    """PRIVATE: Compute exact preflight work counts without pair allocation.

    Parameters
    ----------
    support : GalerkinProductSupport
        Fresh full product support.

    Returns
    -------
    state_pairs : int
        Exact ordered ``Iu x Iu`` pair count.
    interaction_modes : int
        Exact ordered ``Ichi`` size.
    work_grid_points : int
        Exact product of the static work shape.
    host_array_working_set_upper_bound : int
        Conservative post-replay L3 explicit host-array working-set bound
        before pair allocation.

    Notes
    -----
    The bound includes every explicit support and pair-map array plus 256
    bytes per interaction mode for quotient lookup, sort scratch, difference
    evidence, coefficient inputs, and rectangle/error output arrays.  Exact
    rational objects are processed one mode at a time and are a separately
    declared bounded transient scalar scope.  Python runtime, allocator,
    device, total-process RSS, nested L2 authentication, and product-support
    factory replay are outside this count.
    """
    state_count = support.state_indices.shape[0]
    interaction_modes = support.interaction_indices.shape[0]
    state_pairs = state_count * state_count
    work_grid_points = math.prod(support.work_shape)
    support_rows = (
        support.state_indices.shape[0]
        + support.interaction_indices.shape[0]
        + support.absorber_indices.shape[0]
        + support.work_indices.shape[0]
    )
    support_bytes = support_rows * _SPACE_DIMENSIONS * 8
    pair_map_bytes = state_pairs * 8
    interaction_working_bytes = interaction_modes * 256
    fixed_working_margin = 1 << 20
    host_array_working_set_upper_bound = (
        support_bytes
        + pair_map_bytes
        + interaction_working_bytes
        + fixed_working_margin
    )
    counts: Tuple[int, int, int, int] = (
        state_pairs,
        interaction_modes,
        work_grid_points,
        host_array_working_set_upper_bound,
    )
    return counts


def _first_resource_failure(
    counts: Tuple[int, int, int, int],
    budgets: Tuple[int, int, int, int],
) -> GalerkinLocalCellCompressionFailure:
    """PRIVATE: Return the first deterministic preflight budget failure.

    Parameters
    ----------
    counts : Tuple[int, int, int, int]
        Pair, mode, work-grid, and host-byte counts.
    budgets : Tuple[int, int, int, int]
        Corresponding caller-declared maxima.

    Returns
    -------
    result : GalerkinLocalCellCompressionFailure
        First exceeded resource or ``NONE``.
    """
    failures = (
        GalerkinLocalCellCompressionFailure.STATE_PAIR_BUDGET_EXCEEDED,
        GalerkinLocalCellCompressionFailure.INTERACTION_MODE_BUDGET_EXCEEDED,
        GalerkinLocalCellCompressionFailure.WORK_GRID_BUDGET_EXCEEDED,
        GalerkinLocalCellCompressionFailure.HOST_ARRAY_WORKING_SET_BUDGET_EXCEEDED,
    )
    for count, budget, failure in zip(counts, budgets, failures, strict=True):
        if count > budget:
            result: GalerkinLocalCellCompressionFailure = failure
            return result
    result: GalerkinLocalCellCompressionFailure = (
        GalerkinLocalCellCompressionFailure.NONE
    )
    return result


def _stream_difference_evidence(
    state_indices: _HostStateIndices,
    interaction_indices: _HostInteractionIndices,
    work_shape: Tuple[int, int, int],
) -> _DifferenceEvidence:
    """PRIVATE: Stream exact state differences and ordered support lookups.

    Parameters
    ----------
    state_indices : _HostStateIndices
        Exact ordered state support.
    interaction_indices : _HostInteractionIndices
        Exact ordered interaction support.
    work_shape : Tuple[int, int, int]
        Product quotient used only for collision-checked lookup keys.

    Returns
    -------
    result : _DifferenceEvidence
        Lexicographic differences, their ordered ``Ichi`` positions and
        multiplicities, the row-major pair lookup, and the typed outcome.

    Raises
    ------
    AssertionError
        If the exact pair stream fails its internal coverage invariants.
    """
    moduli = np.asarray(work_shape, dtype=np.int64)
    residues = np.mod(interaction_indices, moduli[None, :])
    keys = (residues[:, 0] * work_shape[1] + residues[:, 1]) * work_shape[
        2
    ] + residues[:, 2]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    pair_count = state_indices.shape[0] * state_indices.shape[0]
    state_pair_positions = np.empty((pair_count,), dtype=np.int64)
    multiplicity_by_interaction = np.zeros(
        (interaction_indices.shape[0],), dtype=np.int64
    )
    flat_position = 0
    for left in state_indices:
        left_tuple = tuple(int(value) for value in left)
        for right in state_indices:
            difference: Tuple[int, int, int] = (
                left_tuple[0] - int(right[0]),
                left_tuple[1] - int(right[1]),
                left_tuple[2] - int(right[2]),
            )
            if any(
                value < _MINIMUM_SIGNED_INT64 or value > _MAXIMUM_SIGNED_INT64
                for value in difference
            ):
                result: _DifferenceEvidence = (
                    np.zeros((0, 3), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    GalerkinLocalCellCompressionFailure.DIFFERENCE_ARITHMETIC_RANGE_FAILURE,
                )
                return result
            residue_x = difference[0] % work_shape[0]
            residue_y = difference[1] % work_shape[1]
            residue_z = difference[2] % work_shape[2]
            key = (residue_x * work_shape[1] + residue_y) * work_shape[
                2
            ] + residue_z
            location = int(np.searchsorted(sorted_keys, key, side="left"))
            if (
                location >= sorted_keys.shape[0]
                or int(sorted_keys[location]) != key
            ):
                result: _DifferenceEvidence = (
                    np.zeros((0, 3), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    GalerkinLocalCellCompressionFailure.DIFFERENCE_COVERAGE_MISSING,
                )
                return result
            support_position = int(order[location])
            matched = interaction_indices[support_position]
            if not (
                int(matched[0]) == difference[0]
                and int(matched[1]) == difference[1]
                and int(matched[2]) == difference[2]
            ):
                result: _DifferenceEvidence = (
                    np.zeros((0, 3), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    GalerkinLocalCellCompressionFailure.DIFFERENCE_COVERAGE_MISSING,
                )
                return result
            state_pair_positions[flat_position] = support_position
            multiplicity_by_interaction[support_position] += 1
            flat_position += 1

    used_positions = np.flatnonzero(multiplicity_by_interaction)
    used_indices = interaction_indices[used_positions]
    lexicographic_order = np.lexsort(
        (used_indices[:, 2], used_indices[:, 1], used_indices[:, 0])
    )
    positions = np.asarray(used_positions[lexicographic_order], dtype=np.int64)
    differences = np.asarray(interaction_indices[positions], dtype=np.int64)
    multiplicities = np.asarray(
        multiplicity_by_interaction[positions], dtype=np.int64
    )
    if flat_position != pair_count:
        raise AssertionError("pair lookup did not fill its exact allocation")
    if int(sum(int(value) for value in multiplicities)) != pair_count:
        raise AssertionError("difference multiplicities do not sum to n^2")
    result: _DifferenceEvidence = (
        differences,
        positions,
        multiplicities,
        state_pair_positions,
        GalerkinLocalCellCompressionFailure.NONE,
    )
    return result


def _exact_sigma_fraction(voltage_kv: float) -> Fraction:
    r"""PRIVATE: Evaluate the exact stored-input SC.4 rational point.

    Parameters
    ----------
    voltage_kv : float
        Positive finite stored binary64 accelerating voltage in kilovolts.

    Returns
    -------
    sigma : Fraction
        Exact positive SC.4 value in inverse-square Angstroms per volt.

    Raises
    ------
    ValueError
        If the exact stored-input coupling is not positive.

    Notes
    -----
    Every physical input is interpreted as its exact stored dyadic rational.
    Kilovolt conversion uses exact integer ``1000`` and square-metre to
    square-Angstrom conversion uses mathematical ``10^-20`` exactly, not the
    rounded binary64 literal with that decimal spelling.
    """
    voltage = _fraction_from_float(voltage_kv)
    mass = _fraction_from_float(float(np.asarray(jax.device_get(M_E))))
    charge = _fraction_from_float(float(np.asarray(jax.device_get(E_CHARGE))))
    speed = _fraction_from_float(float(np.asarray(jax.device_get(C_LIGHT))))
    hbar = _fraction_from_float(float(np.asarray(jax.device_get(HBAR))))
    voltage_volts = 1000 * voltage
    prefactor = 2 * mass * charge / (hbar * hbar)
    correction = 1 + charge * voltage_volts / (mass * speed * speed)
    sigma: Fraction = prefactor * correction * Fraction(1, 10**20)
    if sigma <= 0:
        raise ValueError("exact SC.4 coupling must be positive")
    return sigma


def _point_rectangle_error(
    point: np.complex128,
    rectangle: Tuple[Fraction, Fraction, Fraction, Fraction],
) -> Fraction:
    """PRIVATE: Bound a complex point against every rectangle value.

    Parameters
    ----------
    point : np.complex128
        Stored rounded interaction coefficient.
    rectangle : Tuple[Fraction, Fraction, Fraction, Fraction]
        Exact rational real/imaginary rectangle.

    Returns
    -------
    error : Fraction
        Verified rational Euclidean farthest-corner radius.
    """
    real = _fraction_from_float(float(np.real(point)))
    imaginary = _fraction_from_float(float(np.imag(point)))
    real_gap = max(abs(real - rectangle[0]), abs(real - rectangle[1]))
    imag_gap = max(
        abs(imaginary - rectangle[2]),
        abs(imaginary - rectangle[3]),
    )
    error: Fraction = _sqrt_fraction_upper(
        real_gap * real_gap + imag_gap * imag_gap
    )
    return error


def _operator_digest(
    realization: GalerkinLocalCellPotentialRealization,
    support: GalerkinProductSupport,
    differences: Int64[NDArray, "d 3"],
    positions: Int64[NDArray, " d"],
    multiplicities: Int64[NDArray, " d"],
    state_pair_positions: Int64[NDArray, " s"],
    voltage: Float64[NDArray, ""],
    coupling: Float64[NDArray, ""],
    coefficients: _HostComplex,
) -> str:
    """PRIVATE: Digest operator identity without evidence or budgets.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Authenticated L2 parent.
    support : GalerkinProductSupport
        Fresh full product support.
    differences : Int64[NDArray, "d 3"]
        Ordered exact ``Du``.
    positions : Int64[NDArray, " d"]
        Ordered ``Du`` to ``Ichi`` positions.
    multiplicities : Int64[NDArray, " d"]
        Exact LVT.17 multiplicities.
    state_pair_positions : Int64[NDArray, " s"]
        Exact row-major state-pair to ``Ichi`` action map.
    voltage : Float64[NDArray, ""]
        Stored accelerating voltage.
    coupling : Float64[NDArray, ""]
        Canonical stored 50-bit coupling.
    coefficients : _HostComplex
        Canonical stored 50-bit interaction coefficients.

    Returns
    -------
    digest : str
        Reproducible fixed interaction identity.

    Raises
    ------
    ValueError
        If the authenticated parent lacks direct L2 evidence.

    Notes
    -----
    The nested L2 certificate, rectangles, errors, all work counts, all
    budgets, and every outcome field are deliberately absent.  The L2
    realization digest included here binds source, ordered ``Ichi``, stored
    voltage coefficients, and formula without binding its certificate.
    """
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("operator digest requires direct L2 evidence")
    digest: str = _sha256(
        {
            "domain": _OPERATOR_DIGEST_DOMAIN,
            "l2_realization_digest": certificate.realization_digest,
            "local_potential_digest": certificate.local_potential_digest,
            "coefficient_formula": realization.coefficient_formula,
            "interaction_realization_route": _INTERACTION_REALIZATION_ROUTE,
            "state_indices": _stored_value_payload(support.state_indices),
            "interaction_indices": _stored_value_payload(
                support.interaction_indices
            ),
            "difference_indices": _stored_value_payload(differences),
            "difference_interaction_positions": _stored_value_payload(
                positions
            ),
            "difference_multiplicities": _stored_value_payload(multiplicities),
            "state_pair_interaction_positions": _stored_value_payload(
                state_pair_positions
            ),
            "accelerating_voltage_kv": _stored_value_payload(voltage),
            "interaction_coupling": _stored_value_payload(coupling),
            "interaction_coefficients": _stored_value_payload(coefficients),
        }
    )
    return digest


def _certificate_digest(evidence: Dict[str, object]) -> str:
    """PRIVATE: Digest complete L3 evidence and outcome.

    Parameters
    ----------
    evidence : Dict[str, object]
        Canonical evidence payload excluding its own digest.

    Returns
    -------
    digest : str
        Complete LVT.14--LVT.18 certificate identity.
    """
    digest: str = _sha256(
        {"domain": _CERTIFICATE_DOMAIN, "evidence": evidence}
    )
    return digest


def _evidence_payload(  # noqa: PLR0913
    realization: GalerkinLocalCellPotentialRealization,
    support: GalerkinProductSupport,
    differences: Int64[NDArray, "d 3"],
    positions: Int64[NDArray, " d"],
    multiplicities: Int64[NDArray, " d"],
    state_pair_positions: Int64[NDArray, " s"],
    voltage: Float64[NDArray, ""],
    coupling: Float64[NDArray, ""],
    coefficients: _HostComplex,
    sigma_bounds: Tuple[Float64[NDArray, ""], Float64[NDArray, ""]],
    sigma_error: Float64[NDArray, ""],
    real_bounds: Tuple[_HostDifferenceFloat, _HostDifferenceFloat],
    imag_bounds: Tuple[_HostDifferenceFloat, _HostDifferenceFloat],
    coefficient_errors: _HostDifferenceFloat,
    operator_error: Float64[NDArray, ""],
    counts: Tuple[int, int, int, int],
    budgets: Tuple[int, int, int, int],
    failure: GalerkinLocalCellCompressionFailure,
    operator_digest: str,
) -> Dict[str, object]:
    """PRIVATE: Build the complete child certificate payload.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Authenticated direct L2 parent.
    support : GalerkinProductSupport
        Independently rebuilt product support.
    differences : Int64[NDArray, "d 3"]
        Lexicographically ordered exact state differences.
    positions : Int64[NDArray, " d"]
        Ordered difference-to-interaction positions.
    multiplicities : Int64[NDArray, " d"]
        Exact ordered-pair multiplicities.
    state_pair_positions : Int64[NDArray, " s"]
        Row-major pair-to-interaction lookup.
    voltage : Float64[NDArray, ""]
        Exact stored accelerating-voltage scalar.
    coupling : Float64[NDArray, ""]
        Canonical stored SC.4 coupling.
    coefficients : _HostComplex
        Canonical stored interaction coefficients.
    sigma_bounds : Tuple[Float64[NDArray, ""], Float64[NDArray, ""]]
        Outward binary64 exact-coupling enclosure.
    sigma_error : Float64[NDArray, ""]
        Stored-coupling audit error.
    real_bounds : Tuple[_HostDifferenceFloat, _HostDifferenceFloat]
        Outward real interaction rectangle endpoints.
    imag_bounds : Tuple[_HostDifferenceFloat, _HostDifferenceFloat]
        Outward imaginary interaction rectangle endpoints.
    coefficient_errors : _HostDifferenceFloat
        Direct interaction point-to-rectangle errors.
    operator_error : Float64[NDArray, ""]
        Fixed LVT.18 interaction error.
    counts : Tuple[int, int, int, int]
        Exact post-replay L3 resource counts.
    budgets : Tuple[int, int, int, int]
        Corresponding resource budgets.
    failure : GalerkinLocalCellCompressionFailure
        Typed child outcome.
    operator_digest : str
        Evidence-free fixed operator identity.

    Returns
    -------
    evidence : Dict[str, object]
        Complete canonical payload excluding its own certificate digest.

    Raises
    ------
    ValueError
        If the authenticated parent lacks direct L2 evidence.
    """
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("L3 evidence requires a direct L2 certificate")
    count_arrays = tuple(np.asarray(value, dtype=np.int64) for value in counts)
    budget_arrays = tuple(
        np.asarray(value, dtype=np.int64) for value in budgets
    )
    evidence: Dict[str, object] = {
        "parent_certificate_digest": certificate.certificate_digest,
        "operator_digest": operator_digest,
        "failure": failure.value,
        "finite_certificate": (
            failure is GalerkinLocalCellCompressionFailure.NONE
        ),
        "exact_target": _EXACT_TARGET,
        "coupling_target": _COUPLING_TARGET,
        "compression_claim": _COMPRESSION_CLAIM,
        "interaction_realization_route": _INTERACTION_REALIZATION_ROUTE,
        "difference_count_route": _DIFFERENCE_COUNT_ROUTE,
        "operator_error_scope": _OPERATOR_ERROR_SCOPE,
        "per_call_arithmetic_exclusion": _PER_CALL_ARITHMETIC_EXCLUSION,
        "host_transient_scalar_scope": _HOST_TRANSIENT_SCALAR_SCOPE,
        "product_support": _stored_value_payload(support),
        "difference_indices": _stored_value_payload(differences),
        "difference_interaction_positions": _stored_value_payload(positions),
        "difference_multiplicities": _stored_value_payload(multiplicities),
        "state_pair_interaction_positions": _stored_value_payload(
            state_pair_positions
        ),
        "accelerating_voltage_kv": _stored_value_payload(voltage),
        "interaction_coupling": _stored_value_payload(coupling),
        "interaction_coefficients": _stored_value_payload(coefficients),
        "exact_coupling_lower_bound": _stored_value_payload(sigma_bounds[0]),
        "exact_coupling_upper_bound": _stored_value_payload(sigma_bounds[1]),
        "coupling_error_bound": _stored_value_payload(sigma_error),
        "exact_interaction_real_lower_bounds": _stored_value_payload(
            real_bounds[0]
        ),
        "exact_interaction_real_upper_bounds": _stored_value_payload(
            real_bounds[1]
        ),
        "exact_interaction_imag_lower_bounds": _stored_value_payload(
            imag_bounds[0]
        ),
        "exact_interaction_imag_upper_bounds": _stored_value_payload(
            imag_bounds[1]
        ),
        "interaction_coefficient_error_bounds": _stored_value_payload(
            coefficient_errors
        ),
        "fixed_interaction_error_bound": _stored_value_payload(operator_error),
        "counts": _stored_value_payload(count_arrays),
        "budgets": _stored_value_payload(budget_arrays),
    }
    return evidence


def _make_failure_compression(  # noqa: PLR0913
    realization: GalerkinLocalCellPotentialRealization,
    support: GalerkinProductSupport,
    voltage: Float64[NDArray, ""],
    counts: Tuple[int, int, int, int],
    budgets: Tuple[int, int, int, int],
    failure: GalerkinLocalCellCompressionFailure,
    differences: Int64[NDArray, "d 3"] | None = None,
    positions: Int64[NDArray, " d"] | None = None,
    multiplicities: Int64[NDArray, " d"] | None = None,
    state_pair_positions: Int64[NDArray, " s"] | None = None,
) -> GalerkinLocalCellExactCompression:
    """PRIVATE: Create one typed all-infinite L3 noncertificate.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Authenticated direct L2 parent.
    support : GalerkinProductSupport
        Independently rebuilt product support.
    voltage : Float64[NDArray, ""]
        Exact stored accelerating-voltage scalar.
    counts : Tuple[int, int, int, int]
        Exact post-replay L3 resource counts.
    budgets : Tuple[int, int, int, int]
        Corresponding resource budgets.
    failure : GalerkinLocalCellCompressionFailure
        Typed non-success outcome.
    differences : Int64[NDArray, "d 3"] | None
        Partial exact differences, or an empty array if omitted. Default is
        ``None``.
    positions : Int64[NDArray, " d"] | None
        Partial interaction positions, or zeros if omitted. Default is
        ``None``.
    multiplicities : Int64[NDArray, " d"] | None
        Partial exact multiplicities, or zeros if omitted. Default is
        ``None``.
    state_pair_positions : Int64[NDArray, " s"] | None
        Partial pair lookup, or an empty array if omitted. Default is
        ``None``.

    Returns
    -------
    compression : GalerkinLocalCellExactCompression
        Typed all-infinite noncertificate storage.

    Raises
    ------
    ValueError
        If the outcome is successful or the parent lacks direct L2 evidence.
    """
    if failure is GalerkinLocalCellCompressionFailure.NONE:
        raise ValueError("failure compression requires a non-NONE outcome")
    if differences is None:
        differences = np.zeros((0, 3), dtype=np.int64)
    difference_count = differences.shape[0]
    if positions is None:
        positions = np.zeros((difference_count,), dtype=np.int64)
    if multiplicities is None:
        multiplicities = np.zeros((difference_count,), dtype=np.int64)
    if state_pair_positions is None:
        state_pair_positions = np.zeros((0,), dtype=np.int64)
    coefficient_count = support.interaction_indices.shape[0]
    coefficients = np.zeros((coefficient_count,), dtype=np.complex128)
    coupling = np.asarray(0.0, dtype=np.float64)
    sigma_bounds = (
        np.asarray(0.0, dtype=np.float64),
        np.asarray(0.0, dtype=np.float64),
    )
    sigma_error = np.asarray(np.inf, dtype=np.float64)
    lower = np.full((difference_count,), -np.inf, dtype=np.float64)
    upper = np.full((difference_count,), np.inf, dtype=np.float64)
    coefficient_errors = np.full((difference_count,), np.inf, dtype=np.float64)
    operator_error = np.asarray(np.inf, dtype=np.float64)
    evidence = _evidence_payload(
        realization,
        support,
        differences,
        positions,
        multiplicities,
        state_pair_positions,
        voltage,
        coupling,
        coefficients,
        sigma_bounds,
        sigma_error,
        (lower, upper),
        (lower, upper),
        coefficient_errors,
        operator_error,
        counts,
        budgets,
        failure,
        _ZERO_DIGEST,
    )
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("failure compression requires direct L2 evidence")
    compression: GalerkinLocalCellExactCompression = (
        _make_local_cell_exact_compression(
            realization,
            support,
            jnp.asarray(differences),
            jnp.asarray(positions),
            jnp.asarray(multiplicities),
            jnp.asarray(state_pair_positions),
            jnp.asarray(voltage),
            jnp.asarray(coupling),
            jnp.asarray(coefficients),
            (jnp.asarray(sigma_bounds[0]), jnp.asarray(sigma_bounds[1])),
            jnp.asarray(sigma_error),
            (jnp.asarray(lower), jnp.asarray(upper)),
            (jnp.asarray(lower), jnp.asarray(upper)),
            jnp.asarray(coefficient_errors),
            jnp.asarray(operator_error),
            jnp.asarray(False),
            tuple(jnp.asarray(value, dtype=jnp.int64) for value in counts),
            tuple(jnp.asarray(value, dtype=jnp.int64) for value in budgets),
            failure=failure,
            exact_target=_EXACT_TARGET,
            coupling_target=_COUPLING_TARGET,
            interaction_realization_route=_INTERACTION_REALIZATION_ROUTE,
            difference_count_route=_DIFFERENCE_COUNT_ROUTE,
            compression_claim=_COMPRESSION_CLAIM,
            operator_error_scope=_OPERATOR_ERROR_SCOPE,
            per_call_arithmetic_exclusion=_PER_CALL_ARITHMETIC_EXCLUSION,
            host_transient_scalar_scope=_HOST_TRANSIENT_SCALAR_SCOPE,
            parent_certificate_digest=certificate.certificate_digest,
            operator_digest=_ZERO_DIGEST,
            certificate_digest=_certificate_digest(evidence),
        )
    )
    jax.block_until_ready(compression)
    return compression


def _successful_compression(  # noqa: PLR0913
    realization: GalerkinLocalCellPotentialRealization,
    support: GalerkinProductSupport,
    voltage: Float64[NDArray, ""],
    differences: Int64[NDArray, "d 3"],
    positions: Int64[NDArray, " d"],
    multiplicities: Int64[NDArray, " d"],
    state_pair_positions: Int64[NDArray, " s"],
    coupling: Float64[NDArray, ""],
    coefficients: _HostComplex,
    sigma_bounds: Tuple[Float64[NDArray, ""], Float64[NDArray, ""]],
    sigma_error: Float64[NDArray, ""],
    real_bounds: Tuple[_HostDifferenceFloat, _HostDifferenceFloat],
    imag_bounds: Tuple[_HostDifferenceFloat, _HostDifferenceFloat],
    coefficient_errors: _HostDifferenceFloat,
    operator_error: Float64[NDArray, ""],
    counts: Tuple[int, int, int, int],
    budgets: Tuple[int, int, int, int],
) -> GalerkinLocalCellExactCompression:
    """PRIVATE: Store one finite exact-compression certificate.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Authenticated direct L2 parent.
    support : GalerkinProductSupport
        Independently rebuilt product support.
    voltage : Float64[NDArray, ""]
        Exact stored accelerating-voltage scalar.
    differences : Int64[NDArray, "d 3"]
        Lexicographically ordered exact state differences.
    positions : Int64[NDArray, " d"]
        Ordered difference-to-interaction positions.
    multiplicities : Int64[NDArray, " d"]
        Exact ordered-pair multiplicities.
    state_pair_positions : Int64[NDArray, " s"]
        Row-major pair-to-interaction lookup.
    coupling : Float64[NDArray, ""]
        Canonical stored SC.4 coupling.
    coefficients : _HostComplex
        Canonical stored interaction coefficients.
    sigma_bounds : Tuple[Float64[NDArray, ""], Float64[NDArray, ""]]
        Outward binary64 exact-coupling enclosure.
    sigma_error : Float64[NDArray, ""]
        Stored-coupling audit error.
    real_bounds : Tuple[_HostDifferenceFloat, _HostDifferenceFloat]
        Outward real interaction rectangle endpoints.
    imag_bounds : Tuple[_HostDifferenceFloat, _HostDifferenceFloat]
        Outward imaginary interaction rectangle endpoints.
    coefficient_errors : _HostDifferenceFloat
        Direct interaction point-to-rectangle errors.
    operator_error : Float64[NDArray, ""]
        Fixed LVT.18 interaction error.
    counts : Tuple[int, int, int, int]
        Exact post-replay L3 resource counts.
    budgets : Tuple[int, int, int, int]
        Corresponding resource budgets.

    Returns
    -------
    compression : GalerkinLocalCellExactCompression
        Finite replayable exact-compression evidence.

    Raises
    ------
    ValueError
        If the authenticated parent lacks direct L2 evidence.
    """
    digest = _operator_digest(
        realization,
        support,
        differences,
        positions,
        multiplicities,
        state_pair_positions,
        voltage,
        coupling,
        coefficients,
    )
    evidence = _evidence_payload(
        realization,
        support,
        differences,
        positions,
        multiplicities,
        state_pair_positions,
        voltage,
        coupling,
        coefficients,
        sigma_bounds,
        sigma_error,
        real_bounds,
        imag_bounds,
        coefficient_errors,
        operator_error,
        counts,
        budgets,
        GalerkinLocalCellCompressionFailure.NONE,
        digest,
    )
    certificate = realization.coefficient_certificate
    if certificate is None:
        raise ValueError("success compression requires direct L2 evidence")
    compression: GalerkinLocalCellExactCompression = (
        _make_local_cell_exact_compression(
            realization,
            support,
            jnp.asarray(differences),
            jnp.asarray(positions),
            jnp.asarray(multiplicities),
            jnp.asarray(state_pair_positions),
            jnp.asarray(voltage),
            jnp.asarray(coupling),
            jnp.asarray(coefficients),
            (jnp.asarray(sigma_bounds[0]), jnp.asarray(sigma_bounds[1])),
            jnp.asarray(sigma_error),
            (jnp.asarray(real_bounds[0]), jnp.asarray(real_bounds[1])),
            (jnp.asarray(imag_bounds[0]), jnp.asarray(imag_bounds[1])),
            jnp.asarray(coefficient_errors),
            jnp.asarray(operator_error),
            jnp.asarray(True),
            tuple(jnp.asarray(value, dtype=jnp.int64) for value in counts),
            tuple(jnp.asarray(value, dtype=jnp.int64) for value in budgets),
            failure=GalerkinLocalCellCompressionFailure.NONE,
            exact_target=_EXACT_TARGET,
            coupling_target=_COUPLING_TARGET,
            interaction_realization_route=_INTERACTION_REALIZATION_ROUTE,
            difference_count_route=_DIFFERENCE_COUNT_ROUTE,
            compression_claim=_COMPRESSION_CLAIM,
            operator_error_scope=_OPERATOR_ERROR_SCOPE,
            per_call_arithmetic_exclusion=_PER_CALL_ARITHMETIC_EXCLUSION,
            host_transient_scalar_scope=_HOST_TRANSIENT_SCALAR_SCOPE,
            parent_certificate_digest=certificate.certificate_digest,
            operator_digest=digest,
            certificate_digest=_certificate_digest(evidence),
        )
    )
    jax.block_until_ready(compression)
    return compression


def _certify_local_cell_exact_compression_impl(  # noqa: PLR0911,PLR0912,PLR0913,PLR0915
    realization: GalerkinLocalCellPotentialRealization,
    *,
    accelerating_voltage_kv: float | jax.Array | Shaped[NDArray, ""],
    maximum_state_pairs: int,
    maximum_interaction_modes: int,
    maximum_work_grid_points: int,
    maximum_host_array_working_set_bytes: int,
) -> GalerkinLocalCellExactCompression:
    """PRIVATE: Construct L3 evidence without recursive self-authentication.

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Submitted direct L2 realization to authenticate fully.
    accelerating_voltage_kv : float | jax.Array | Shaped[NDArray, ""]
        Exact float64 scalar accelerating voltage in kilovolts.
    maximum_state_pairs : int
        Positive ordered-pair budget.
    maximum_interaction_modes : int
        Positive ordered-interaction-mode budget.
    maximum_work_grid_points : int
        Positive product work-grid budget.
    maximum_host_array_working_set_bytes : int
        Positive post-replay L3 explicit host-array budget.

    Returns
    -------
    compression : GalerkinLocalCellExactCompression
        Finite evidence or one typed all-infinite noncertificate.

    Raises
    ------
    ValueError
        If scalar storage, raw structure, replay, or exact counts are invalid.
    """
    _assert_concrete((realization, accelerating_voltage_kv))
    voltage_host = np.asarray(jax.device_get(accelerating_voltage_kv))
    if (
        voltage_host.shape != ()
        or voltage_host.dtype != np.dtype(np.float64)
        or not np.isfinite(voltage_host)
        or float(voltage_host) <= 0.0
    ):
        raise ValueError(
            "accelerating_voltage_kv must be an exact finite positive "
            "float64 scalar"
        )
    voltage: Float64[NDArray, ""] = voltage_host
    voltage_array: Float64[Array, ""] = jnp.asarray(voltage)
    budgets = (
        maximum_state_pairs,
        maximum_interaction_modes,
        maximum_work_grid_points,
        maximum_host_array_working_set_bytes,
    )
    submitted_support = realization.support_eligibility.manifest.support
    raw_support_arrays = (
        submitted_support.state_indices,
        submitted_support.interaction_indices,
        submitted_support.absorber_indices,
        submitted_support.work_indices,
    )
    if any(
        values.ndim != _SUPPORT_RANK
        or values.shape[1:] != (_SPACE_DIMENSIONS,)
        or values.shape[0] == 0
        for values in raw_support_arrays
    ):
        raise ValueError(
            "submitted product-support arrays must have shape (n, 3)"
        )
    raw_work_shape = submitted_support.work_shape
    if (
        not isinstance(raw_work_shape, tuple)
        or len(raw_work_shape) != _SPACE_DIMENSIONS
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in raw_work_shape
        )
    ):
        raise ValueError(
            "submitted work_shape must be one positive integer tuple"
        )
    preflight_counts = _resource_counts(submitted_support)
    if any(value > _MAXIMUM_SIGNED_INT64 for value in preflight_counts):
        raise ValueError(
            "exact L3 resource counts must fit signed int64 storage"
        )
    preflight_failure = _first_resource_failure(preflight_counts, budgets)

    canonical = _authenticate_local_cell_certificate(realization)
    support = _rebuild_product_support(canonical)
    counts = _resource_counts(support)
    if any(value > _MAXIMUM_SIGNED_INT64 for value in counts):
        raise ValueError(
            "exact L3 resource counts must fit signed int64 storage"
        )
    if counts != preflight_counts:
        raise ValueError("raw and replayed exact resource counts differ")
    if _first_resource_failure(counts, budgets) is not preflight_failure:
        raise ValueError("raw and replayed resource outcomes differ")

    certificate = canonical.coefficient_certificate
    if certificate is None:
        raise ValueError("L3 requires a direct L2 coefficient certificate")
    finite_l2 = bool(
        _host_exact_scalar(
            certificate.finite_certificate,
            np.dtype(np.bool_),
            "L2 finite_certificate",
        )
    )
    if (
        not finite_l2
        or certificate.failure is not GalerkinLocalCellCertificateFailure.NONE
        or canonical.error_route
        is not GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
    ):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.L2_CERTIFICATE_NOT_FINITE,
            )
        )
        return compression

    resource_failure = preflight_failure
    if resource_failure is not GalerkinLocalCellCompressionFailure.NONE:
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                resource_failure,
            )
        )
        return compression
    if not _host_binary64_supported():
        compression: GalerkinLocalCellExactCompression
        compression = _make_failure_compression(
            canonical,
            support,
            voltage,
            counts,
            budgets,
            GalerkinLocalCellCompressionFailure.HOST_ARITHMETIC_UNSUPPORTED,
        )
        return compression

    state = _checked_support_matrix(support.state_indices, "state_indices")
    interaction = _checked_support_matrix(
        support.interaction_indices, "interaction_indices"
    )
    (
        differences,
        positions,
        multiplicities,
        state_pair_positions,
        difference_failure,
    ) = _stream_difference_evidence(
        state,
        interaction,
        support.work_shape,
    )
    if difference_failure is not GalerkinLocalCellCompressionFailure.NONE:
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                difference_failure,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression

    try:
        exact_sigma = _exact_sigma_fraction(float(voltage))
        sigma_lower_float = _fraction_lower_float(exact_sigma)
        sigma_upper_float = _fraction_upper_float(exact_sigma)
    except (ValueError, OverflowError):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.SIGMA_ENCLOSURE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression
    if (
        not math.isfinite(sigma_lower_float)
        or not math.isfinite(sigma_upper_float)
        or sigma_lower_float <= 0.0
    ):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.SIGMA_ENCLOSURE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression
    sigma_bounds = (
        np.asarray(sigma_lower_float, dtype=np.float64),
        np.asarray(sigma_upper_float, dtype=np.float64),
    )

    coupling_device, coefficients_device = coupled_interaction_value(
        canonical.voltage_coefficients,
        voltage_array,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    jax.block_until_ready((coupling_device, coefficients_device))
    coupling_host = _host_exact_array(
        coupling_device, np.dtype(np.float64), "interaction_coupling"
    )
    coefficients = _host_exact_array(
        coefficients_device,
        np.dtype(np.complex128),
        "interaction_coefficients",
    )
    if coupling_host.shape != () or coefficients.shape != (
        interaction.shape[0],
    ):
        raise ValueError("canonical interaction route returned invalid shapes")
    coupling: Float64[NDArray, ""] = coupling_host
    interaction_coefficients: _HostComplex = coefficients
    voltage_coefficients = _host_exact_array(
        canonical.voltage_coefficients,
        np.dtype(np.complex128),
        "voltage_coefficients",
    )
    lost_component = np.any(
        ((voltage_coefficients.real != 0.0) & (coefficients.real == 0.0))
        | ((voltage_coefficients.imag != 0.0) & (coefficients.imag == 0.0))
    )
    tiny = np.finfo(np.float64).tiny
    subnormal_input = np.any(
        (
            (np.abs(voltage_coefficients.real) < tiny)
            & (voltage_coefficients.real != 0.0)
        )
        | (
            (np.abs(voltage_coefficients.imag) < tiny)
            & (voltage_coefficients.imag != 0.0)
        )
    )
    if (
        not np.isfinite(coupling)
        or float(coupling) <= 0.0
        or not np.all(np.isfinite(interaction_coefficients))
        or lost_component
        or subnormal_input
    ):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.INTERACTION_RANGE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression

    sigma_hat_fraction = _fraction_from_float(float(coupling))
    sigma_error_fraction = abs(sigma_hat_fraction - exact_sigma)
    sigma_error_float = _fraction_upper_float(sigma_error_fraction)
    sigma_error = np.asarray(sigma_error_float, dtype=np.float64)
    if not np.isfinite(sigma_error):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.ARITHMETIC_RANGE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression

    real_lower_c = _host_exact_array(
        certificate.exact_coefficient_real_lower_bounds,
        np.dtype(np.float64),
        "L2 real lower bounds",
    )
    real_upper_c = _host_exact_array(
        certificate.exact_coefficient_real_upper_bounds,
        np.dtype(np.float64),
        "L2 real upper bounds",
    )
    imag_lower_c = _host_exact_array(
        certificate.exact_coefficient_imag_lower_bounds,
        np.dtype(np.float64),
        "L2 imag lower bounds",
    )
    imag_upper_c = _host_exact_array(
        certificate.exact_coefficient_imag_upper_bounds,
        np.dtype(np.float64),
        "L2 imag upper bounds",
    )
    sigma_interval = (
        _fraction_from_float(float(sigma_bounds[0])),
        _fraction_from_float(float(sigma_bounds[1])),
    )
    difference_count = positions.shape[0]
    real_lower = np.empty((difference_count,), dtype=np.float64)
    real_upper = np.empty((difference_count,), dtype=np.float64)
    imag_lower = np.empty((difference_count,), dtype=np.float64)
    imag_upper = np.empty((difference_count,), dtype=np.float64)
    coefficient_errors = np.empty((difference_count,), dtype=np.float64)
    radicand = Fraction(0)
    interaction_range_failure = False
    try:
        for difference_position, position in enumerate(positions):
            support_position = int(position)
            real_interval = _real_interval_product(
                sigma_interval,
                (
                    _fraction_from_float(
                        float(real_lower_c[support_position])
                    ),
                    _fraction_from_float(
                        float(real_upper_c[support_position])
                    ),
                ),
            )
            imag_interval = _real_interval_product(
                sigma_interval,
                (
                    _fraction_from_float(
                        float(imag_lower_c[support_position])
                    ),
                    _fraction_from_float(
                        float(imag_upper_c[support_position])
                    ),
                ),
            )
            rectangle = (
                real_interval[0],
                real_interval[1],
                imag_interval[0],
                imag_interval[1],
            )
            error_fraction = _point_rectangle_error(
                interaction_coefficients[support_position], rectangle
            )
            stored_values = (
                _fraction_lower_float(rectangle[0]),
                _fraction_upper_float(rectangle[1]),
                _fraction_lower_float(rectangle[2]),
                _fraction_upper_float(rectangle[3]),
                _fraction_upper_float(error_fraction),
            )
            if not all(math.isfinite(value) for value in stored_values):
                interaction_range_failure = True
                break
            real_lower[difference_position] = stored_values[0]
            real_upper[difference_position] = stored_values[1]
            imag_lower[difference_position] = stored_values[2]
            imag_upper[difference_position] = stored_values[3]
            coefficient_errors[difference_position] = stored_values[4]
            stored_error = _fraction_from_float(stored_values[4])
            radicand += (
                int(multiplicities[difference_position])
                * stored_error
                * stored_error
            )
    except _RootEnclosureError:
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.ROOT_ENCLOSURE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression

    if interaction_range_failure:
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.INTERACTION_RANGE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression

    try:
        operator_error_fraction = _sqrt_fraction_upper(radicand)
    except (ValueError, _RootEnclosureError):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.ROOT_ENCLOSURE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression
    operator_error = np.asarray(
        _fraction_upper_float(operator_error_fraction), dtype=np.float64
    )
    if not np.isfinite(operator_error):
        compression: GalerkinLocalCellExactCompression = (
            _make_failure_compression(
                canonical,
                support,
                voltage,
                counts,
                budgets,
                GalerkinLocalCellCompressionFailure.ARITHMETIC_RANGE_FAILURE,
                differences,
                positions,
                multiplicities,
            )
        )
        return compression
    compression: GalerkinLocalCellExactCompression = _successful_compression(
        canonical,
        support,
        voltage,
        differences,
        positions,
        multiplicities,
        state_pair_positions,
        coupling,
        interaction_coefficients,
        sigma_bounds,
        sigma_error,
        (real_lower, real_upper),
        (imag_lower, imag_upper),
        coefficient_errors,
        operator_error,
        counts,
        budgets,
    )
    return compression


def _validate_compression_storage(
    compression: GalerkinLocalCellExactCompression,
) -> Tuple[float, int, int, int, int]:
    """PRIVATE: Validate exact scalar storage before full mathematical replay.

    Parameters
    ----------
    compression : GalerkinLocalCellExactCompression
        Submitted public storage carrier.

    Returns
    -------
    voltage : float
        Stored binary64 accelerating voltage.
    maximum_state_pairs : int
        Stored exact pair budget.
    maximum_interaction_modes : int
        Stored exact mode budget.
    maximum_work_grid_points : int
        Stored exact work-grid budget.
    maximum_host_array_working_set_bytes : int
        Stored exact host-byte budget.

    Raises
    ------
    ValueError
        If exact dtype, shape, static declaration, or digest binding differs.
    """
    _assert_concrete(compression)
    voltage = float(
        _host_exact_scalar(
            compression.accelerating_voltage_kv,
            np.dtype(np.float64),
            "accelerating_voltage_kv",
        )
    )
    budget_values = tuple(
        int(
            _host_exact_scalar(
                value,
                np.dtype(np.int64),
                name,
            )
        )
        for value, name in (
            (compression.maximum_state_pairs, "maximum_state_pairs"),
            (
                compression.maximum_interaction_modes,
                "maximum_interaction_modes",
            ),
            (
                compression.maximum_work_grid_points,
                "maximum_work_grid_points",
            ),
            (
                compression.maximum_host_array_working_set_bytes,
                "maximum_host_array_working_set_bytes",
            ),
        )
    )
    for budget, name in zip(
        budget_values,
        (
            "maximum_state_pairs",
            "maximum_interaction_modes",
            "maximum_work_grid_points",
            "maximum_host_array_working_set_bytes",
        ),
        strict=True,
    ):
        _checked_budget(budget, name)
    certificate = compression.realization.coefficient_certificate
    if certificate is None:
        raise ValueError("compression requires nested direct L2 evidence")
    if compression.parent_certificate_digest != certificate.certificate_digest:
        raise ValueError("compression parent certificate digest is invalid")
    expected_static = (
        compression.exact_target == _EXACT_TARGET
        and compression.coupling_target == _COUPLING_TARGET
        and compression.interaction_realization_route
        == _INTERACTION_REALIZATION_ROUTE
        and compression.difference_count_route == _DIFFERENCE_COUNT_ROUTE
        and compression.compression_claim == _COMPRESSION_CLAIM
        and compression.operator_error_scope == _OPERATOR_ERROR_SCOPE
        and compression.per_call_arithmetic_exclusion
        == _PER_CALL_ARITHMETIC_EXCLUSION
        and compression.host_transient_scalar_scope
        == _HOST_TRANSIENT_SCALAR_SCOPE
    )
    if not expected_static:
        raise ValueError("compression static semantics are noncanonical")
    result: Tuple[float, int, int, int, int] = (
        voltage,
        budget_values[0],
        budget_values[1],
        budget_values[2],
        budget_values[3],
    )
    return result


def _authenticate_local_cell_exact_compression(
    compression: GalerkinLocalCellExactCompression,
) -> GalerkinLocalCellExactCompression:
    """PRIVATE: Fully replay one LVT.14--LVT.18 compression carrier.

    Parameters
    ----------
    compression : GalerkinLocalCellExactCompression
        Submitted public storage carrier.

    Returns
    -------
    canonical : GalerkinLocalCellExactCompression
        Fresh exact mathematical replay with identical stored payload.

    Raises
    ------
    ValueError
        If any nested input, support predicate, difference, physical input,
        rectangle, error, budget, outcome, or digest differs.
    """
    voltage, pair_budget, mode_budget, grid_budget, byte_budget = (
        _validate_compression_storage(compression)
    )
    canonical: GalerkinLocalCellExactCompression = (
        _certify_local_cell_exact_compression_impl(
            compression.realization,
            accelerating_voltage_kv=voltage,
            maximum_state_pairs=pair_budget,
            maximum_interaction_modes=mode_budget,
            maximum_work_grid_points=grid_budget,
            maximum_host_array_working_set_bytes=byte_budget,
        )
    )
    if _stored_value_payload(canonical) != _stored_value_payload(compression):
        raise ValueError(
            "local-cell exact compression does not match full host replay"
        )
    return canonical


@jaxtyped(typechecker=beartype)
def certify_local_cell_exact_compression(
    realization: GalerkinLocalCellPotentialRealization,
    *,
    accelerating_voltage_kv: float | jax.Array | Shaped[NDArray, ""],
    maximum_state_pairs: int = _DEFAULT_MAXIMUM_STATE_PAIRS,
    maximum_interaction_modes: int = _DEFAULT_MAXIMUM_INTERACTION_MODES,
    maximum_work_grid_points: int = _DEFAULT_MAXIMUM_WORK_GRID_POINTS,
    maximum_host_array_working_set_bytes: int = (
        _DEFAULT_MAXIMUM_HOST_ARRAY_WORKING_SET_BYTES
    ),
) -> GalerkinLocalCellExactCompression:
    """Certify exact local-cell compression and its fixed interaction error.

    :see: :func:`~.test_local_cell_interaction.\
test_complete_compression_has_exact_du_mapping_and_multiplicity`

    Parameters
    ----------
    realization : GalerkinLocalCellPotentialRealization
        Concrete direct LVT.13 realization to replay and authenticate.
    accelerating_voltage_kv : float or jax.Array or Shaped[NDArray, ""]
        Positive binary64 scalar accelerating voltage in kilovolts.
    maximum_state_pairs : int, optional
        Maximum ordered ``Iu x Iu`` pairs. Default: ``20_000_000``.
    maximum_interaction_modes : int, optional
        Maximum ordered ``Ichi`` modes. Default: ``2_000_000``.
    maximum_work_grid_points : int, optional
        Maximum product work-grid points. Default: ``200_000_000``.
    maximum_host_array_working_set_bytes : int, optional
        Maximum conservative post-replay L3 explicit host-array working-set
        bytes. Nested L2 authentication and product-support factory replay
        are outside this budget.
        Default: ``2_000_000_000``.

    Returns
    -------
    compression : GalerkinLocalCellExactCompression
        Finite replayable LVT.14--LVT.18 evidence or a typed all-infinite
        noncertificate.  A noncertificate cannot enter an interaction core.

    Raises
    ------
    ValueError
        If budgets, scalar structure, or submitted public storage is forged.
    equinox.EquinoxRuntimeError
        If full L2 or product-support replay rejects the submission.

    Notes
    -----
    The componentwise interaction error compares the canonical stored
    ``chi_hat`` directly with the exact rectangle ``S*C``.  It already
    includes coefficient, coupling, multiplication, and storage error;
    ``coupling_error_bound`` is audit evidence and is never added again.
    """
    budgets = (
        _checked_budget(maximum_state_pairs, "maximum_state_pairs"),
        _checked_budget(
            maximum_interaction_modes, "maximum_interaction_modes"
        ),
        _checked_budget(maximum_work_grid_points, "maximum_work_grid_points"),
        _checked_budget(
            maximum_host_array_working_set_bytes,
            "maximum_host_array_working_set_bytes",
        ),
    )
    compression: GalerkinLocalCellExactCompression = (
        _certify_local_cell_exact_compression_impl(
            realization,
            accelerating_voltage_kv=accelerating_voltage_kv,
            maximum_state_pairs=budgets[0],
            maximum_interaction_modes=budgets[1],
            maximum_work_grid_points=budgets[2],
            maximum_host_array_working_set_bytes=budgets[3],
        )
    )
    _validate_compression_storage(compression)
    return compression


def create_local_cell_interaction_core(
    compression: GalerkinLocalCellExactCompression,
) -> GalerkinLocalCellInteractionCore:
    """Create a non-solver-ready core from finite replayed LVT evidence.

    :see: :func:`~.test_local_cell_interaction.\
test_actions_match_dense_matrix_and_formal_adjoint`

    Parameters
    ----------
    compression : GalerkinLocalCellExactCompression
        Submitted exact-compression carrier.

    Returns
    -------
    core : GalerkinLocalCellInteractionCore
        Fixed rounded interaction action and formal matrix-adjoint identity.

    Raises
    ------
    ValueError
        If full replay differs or the compression is a noncertificate.
    """
    canonical = _authenticate_local_cell_exact_compression(compression)
    if (
        canonical.failure is not GalerkinLocalCellCompressionFailure.NONE
        or not bool(canonical.finite_certificate)
    ):
        raise ValueError("interaction core requires finite LVT.18 evidence")
    core: GalerkinLocalCellInteractionCore = _make_local_cell_interaction_core(
        canonical,
        action_route=_ACTION_ROUTE,
        adjoint_route=_ADJOINT_ROUTE,
        completion_scope=_COMPLETION_SCOPE,
        operator_digest=canonical.operator_digest,
    )
    return core


def _authenticate_local_cell_interaction_core(
    core: GalerkinLocalCellInteractionCore,
) -> GalerkinLocalCellInteractionCore:
    """PRIVATE: Fully replay one public interaction-core carrier.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Submitted public action carrier.

    Returns
    -------
    canonical : GalerkinLocalCellInteractionCore
        Fresh replayed core with exact stored-value identity.

    Raises
    ------
    ValueError
        If nested evidence, action semantics, or operator identity differs.
    """
    canonical: GalerkinLocalCellInteractionCore = (
        create_local_cell_interaction_core(core.compression)
    )
    if _stored_value_payload(canonical) != _stored_value_payload(core):
        raise ValueError("local-cell interaction core does not match replay")
    return canonical


def prepare_local_cell_interaction_core(
    core: GalerkinLocalCellInteractionCore,
) -> GalerkinLocalCellInteractionCore:
    """Replay-authenticate stored core data before transform-compatible use.

    :see: :func:`~.test_local_cell_interaction.\
test_prepare_rejects_self_rehashed_coefficient_forgery`

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Public stored core to validate at a concrete host boundary.

    Returns
    -------
    prepared : GalerkinLocalCellInteractionCore
        Fresh canonical core with exact stored-value identity to the input.

    Raises
    ------
    ValueError
        If nested evidence, action data, or identity differs from full replay.

    Notes
    -----
    Call this after deserialization or any untrusted storage boundary.  The
    transform-compatible action functions deliberately do not perform host
    replay inside JAX tracing.
    """
    prepared: GalerkinLocalCellInteractionCore = (
        _authenticate_local_cell_interaction_core(core)
    )
    return prepared


def _checked_action_field(
    core: GalerkinLocalCellInteractionCore,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """PRIVATE: Validate one rounded interaction-action input.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Fixed LVT interaction action.
    field : Complex[Array, "..."]
        Candidate state coefficient vector.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Finite normal-range complex128 state vector.

    Raises
    ------
    ValueError
        If static action semantics, input rank, or state size differ.
    """
    if (
        core.action_route != _ACTION_ROUTE
        or core.adjoint_route != _ADJOINT_ROUTE
        or core.completion_scope != _COMPLETION_SCOPE
        or core.operator_digest != core.compression.operator_digest
    ):
        raise ValueError("interaction core action semantics are noncanonical")
    values: Complex128[Array, " n"] = jnp.asarray(field, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("field must be 1D")
    if values.shape[0] != core.support.state_indices.shape[0]:
        raise ValueError("field must match the state support")
    checked: Complex128[Array, " n"] = eqx.error_if(
        values,
        (~core.compression.finite_certificate)
        | jnp.any(~jnp.isfinite(values))
        | has_subnormal_components(values),
        "interaction core and field must be finite and normal-range",
    )
    return checked


def _apply_core(
    core: GalerkinLocalCellInteractionCore,
    field: Complex128[Array, " n"],
    *,
    adjoint: bool,
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the forward or explicit reverse-conjugate pair map.

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Fixed accepted interaction action.
    field : Complex128[Array, " n"]
        Checked state coefficient vector.
    adjoint : bool
        Whether to apply the formal algebraic-matrix adjoint branch.

    Returns
    -------
    checked : Complex128[Array, " n"]
        Rounded action output in state order.
    """
    compression = core.compression
    state = core.support.state_indices
    pair_positions = compression.state_pair_interaction_positions
    coefficients = compression.interaction_coefficients
    state_count = state.shape[0]
    pair_count = state_count * state_count

    def add_pair(
        flat_position: Int64[Array, ""],
        accumulator: Complex128[Array, " n"],
    ) -> Complex128[Array, " n"]:
        """Accumulate one exact state pair without a dense matrix."""
        row = flat_position // state_count
        column = flat_position % state_count
        support_position = pair_positions[flat_position]
        coefficient = coefficients[support_position]
        if adjoint:
            updated: Complex128[Array, " n"] = accumulator.at[column].add(
                jnp.conj(coefficient) * field[row]
            )
        else:
            updated: Complex128[Array, " n"] = accumulator.at[row].add(
                coefficient * field[column]
            )
        return updated

    initial: Complex128[Array, " n"] = jnp.zeros(
        (state_count,), dtype=jnp.complex128
    )
    applied: Complex128[Array, " n"] = jax.lax.fori_loop(
        0, pair_count, add_pair, initial
    )
    checked: Complex128[Array, " n"] = eqx.error_if(
        applied,
        jnp.any(~jnp.isfinite(applied)),
        "interaction action left finite binary64 range",
    )
    return checked


@jaxtyped(typechecker=beartype)
def apply_local_cell_interaction(
    core: GalerkinLocalCellInteractionCore,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the frozen rounded LVT.16 interaction action.

    :see: :func:`~.test_local_cell_interaction.\
test_actions_match_dense_matrix_and_formal_adjoint`

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Accepted non-solver-ready interaction core.
    field : Complex[Array, "..."]
        Retained state coefficient vector.

    Returns
    -------
    interaction : Complex128[Array, " n"]
        Rounded compressed interaction output.

    Notes
    -----
    This is the transform-compatible algebraic action and does not certify
    public storage.  A core crossing a storage or trust boundary must first
    pass :func:`prepare_local_cell_interaction_core`; callers then close over
    that prepared core while transforming the field argument.
    LVT.18 bounds the frozen coefficient operator.  Per-call multiplication,
    accumulation, transform, and output rounding are explicitly excluded.
    """
    checked: Complex128[Array, " n"] = _checked_action_field(core, field)
    interaction: Complex128[Array, " n"] = _apply_core(
        core, checked, adjoint=False
    )
    return interaction


@jaxtyped(typechecker=beartype)
def apply_local_cell_interaction_adjoint(
    core: GalerkinLocalCellInteractionCore,
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply the formal matrix adjoint of the frozen rounded interaction.

    :see: :func:`~.test_local_cell_interaction.\
test_actions_match_dense_matrix_and_formal_adjoint`

    Parameters
    ----------
    core : GalerkinLocalCellInteractionCore
        Accepted non-solver-ready interaction core.
    field : Complex[Array, "..."]
        Retained adjoint-state coefficient vector.

    Returns
    -------
    interaction_adjoint : Complex128[Array, " n"]
        Explicit reverse-pair, conjugated algebraic-matrix action.

    Notes
    -----
    This is the formal adjoint of the frozen stored coefficient matrix, not a
    claim of bitwise adjointness for floating multiplication and accumulation.
    It excludes per-call arithmetic exactly as LVT.18 does.  Prepare any core
    crossing a trust boundary before closing over it in a JAX transform.
    """
    checked: Complex128[Array, " n"] = _checked_action_field(core, field)
    interaction_adjoint: Complex128[Array, " n"] = _apply_core(
        core, checked, adjoint=True
    )
    return interaction_adjoint


__all__: list[str] = [
    "apply_local_cell_interaction",
    "apply_local_cell_interaction_adjoint",
    "certify_local_cell_exact_compression",
    "create_local_cell_interaction_core",
    "prepare_local_cell_interaction_core",
]
