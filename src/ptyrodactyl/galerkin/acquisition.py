r"""Check one finite scalar acquisition against the closed RM-S1 predicates.

Extended Summary
----------------
This module checks exact integer support membership, signed endpoints,
coordinate-terminal fibers, represented direct transfers, physical-direction
evidence, complete sector masks, and explicit backward disposition for one
independently owned SC-1 carrier frame. Binary set relations use
constant-memory loops and fail closed above a fixed bounded ceiling instead
of expanding an unbounded Minkowski product.

Routine Listings
----------------
:func:`check_galerkin_acquisition_support`
    Check one bounded acquisition manifest against RM-S1.

Notes
-----
This checker establishes the finite support, physical-direction, geometry,
sector, and single-carrier ownership core needed before an RM-S1 invocation.
``SUPPORT_ELIGIBLE`` is deliberately narrower than full detector eligibility:
the raw-pixel map remains a separate RM-S4 artifact. The result also does not
establish repeated-scattering convergence or multicarrier quotient assembly.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float64, Int32, Int64, jaxtyped

from ptyrodactyl._tools import (
    downward_sqrt,
    interval_add,
    interval_divide_positive,
    interval_multiply,
    mathematical_pi_interval,
    point_interval,
    upward_sqrt,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionManifest,
    GalerkinAcquisitionSupportResult,
    GalerkinBackwardDisposition,
    GalerkinCarrierOverlapDisposition,
    GalerkinCarrierOwnership,
    GalerkinCarrierTargetRoute,
    GalerkinDirectionDisposition,
    GalerkinEndpointConvention,
    GalerkinProductSupport,
    GalerkinTerminalSide,
    scalar_int,
)

from ..types.acquisition_types import (
    _create_galerkin_acquisition_support_result,
)

_SPACE_DIMENSIONS: int = 3
_MAX_BINARY_PAIR_CHECKS: int = 20_000_000


def _transverse_axes(normal_axis: int) -> Tuple[int, int]:
    """PRIVATE: Return the two axes transverse to one coordinate normal.

    Parameters
    ----------
    normal_axis : int
        Coordinate-normal axis in ``{0, 1, 2}``.

    Returns
    -------
    first_axis : int
        First complementary coordinate axis.
    second_axis : int
        Second complementary coordinate axis.
    """
    if normal_axis == 0:
        first_axis: int = 1
        second_axis: int = 2
    elif normal_axis == 1:
        first_axis = 0
        second_axis = 2
    else:
        first_axis = 0
        second_axis = 1
    axes: Tuple[int, int] = (first_axis, second_axis)
    return axes


def _flat_residue_keys(
    indices: Int64[Array, "... 3"],
    work_shape: Tuple[int, int, int],
) -> Int64[Array, "..."]:
    """PRIVATE: Map exact indices to overflow-safe quotient keys.

    Parameters
    ----------
    indices : Int64[Array, "... 3"]
        Exact reciprocal indices.
    work_shape : Tuple[int, int, int]
        Three-dimensional quotient moduli.

    Returns
    -------
    keys : Int64[Array, "..."]
        Flattened non-negative work-grid residue keys.
    """
    moduli: Int64[Array, " 3"] = jnp.asarray(work_shape, dtype=jnp.int64)
    residues: Int64[Array, "... 3"] = jnp.mod(indices, moduli)
    keys: Int64[Array, "..."] = (
        residues[..., 0] * work_shape[1] + residues[..., 1]
    ) * work_shape[2] + residues[..., 2]
    return keys


def _contains_duplicates(
    indices: Int64[Array, "n d"],
) -> Bool[Array, ""]:
    """PRIVATE: Detect duplicate rows in an exact integer set.

    Parameters
    ----------
    indices : Int64[Array, "n d"]
        Exact integer rows.

    Returns
    -------
    duplicates : Bool[Array, ""]
        Whether any two rows are identical.
    """
    order: Int64[Array, " n"] = jnp.lexsort(
        tuple(indices[:, axis] for axis in range(indices.shape[1] - 1, -1, -1))
    )
    sorted_indices: Int64[Array, "n d"] = indices[order]
    duplicates: Bool[Array, ""] = jnp.any(
        jnp.all(sorted_indices[1:] == sorted_indices[:-1], axis=-1)
    )
    return duplicates


def _member_mask(
    candidates: Int64[Array, "m 3"],
    support: Int64[Array, "n 3"],
    work_shape: Tuple[int, int, int],
) -> Bool[Array, " m"]:
    """PRIVATE: Test exact membership with bounded working memory.

    Parameters
    ----------
    candidates : Int64[Array, "m 3"]
        Exact candidate reciprocal indices.
    support : Int64[Array, "n 3"]
        Nonempty exact support set.
    work_shape : Tuple[int, int, int]
        Three-dimensional quotient moduli.

    Returns
    -------
    members : Bool[Array, " m"]
        Per-candidate exact membership mask.
    """
    support_keys: Int64[Array, " n"] = _flat_residue_keys(support, work_shape)
    order: Int64[Array, " n"] = jnp.argsort(support_keys)
    sorted_keys: Int64[Array, " n"] = support_keys[order]
    candidate_keys: Int64[Array, " m"] = _flat_residue_keys(
        candidates, work_shape
    )
    locations: Int32[Array, " m"] = jnp.searchsorted(
        sorted_keys, candidate_keys, side="left"
    )
    clipped: Int32[Array, " m"] = jnp.clip(locations, 0, support.shape[0] - 1)
    members: Bool[Array, " m"] = (
        (locations < support.shape[0])
        & (sorted_keys[clipped] == candidate_keys)
        & jnp.all(support[order[clipped]] == candidates, axis=-1)
    )
    return members


def _all_members(
    candidates: Int64[Array, "m 3"],
    support: Int64[Array, "n 3"],
    work_shape: Tuple[int, int, int],
) -> Bool[Array, ""]:
    """PRIVATE: Require every candidate to belong to an exact support.

    Parameters
    ----------
    candidates : Int64[Array, "m 3"]
        Exact candidate reciprocal indices.
    support : Int64[Array, "n 3"]
        Nonempty exact support set.
    work_shape : Tuple[int, int, int]
        Three-dimensional quotient moduli.

    Returns
    -------
    contained : Bool[Array, ""]
        Whether every candidate is a support member.
    """
    contained: Bool[Array, ""] = jnp.all(
        _member_mask(candidates, support, work_shape)
    )
    return contained


def _binary_relation(  # noqa: PLR0913
    left: Int64[Array, "n 3"],
    right: Int64[Array, "p 3"],
    support: Int64[Array, "w 3"],
    work_shape: Tuple[int, int, int],
    right_sign: int,
    max_pair_checks: int,
) -> Tuple[Bool[Array, ""], Int64[Array, ""], Bool[Array, ""]]:
    """PRIVATE: Check an exact binary-set relation with bounded memory.

    Parameters
    ----------
    left : Int64[Array, "n 3"]
        Left exact operand set.
    right : Int64[Array, "p 3"]
        Right exact operand set.
    support : Int64[Array, "w 3"]
        Required result support.
    work_shape : Tuple[int, int, int]
        Three-dimensional quotient moduli.
    right_sign : int
        Sign applied to each right operand before addition.
    max_pair_checks : int
        Maximum admitted Cartesian-pair count.

    Returns
    -------
    contained : Bool[Array, ""]
        Whether every binary result belongs to ``support``.
    represented_count : Int64[Array, ""]
        Number of represented operand pairs.
    admitted : Bool[Array, ""]
        Whether the pair count stayed within the checker ceiling.
    """
    right_count: int = right.shape[0]
    pair_count: int = left.shape[0] * right_count
    if pair_count > max_pair_checks:
        contained: Bool[Array, ""] = jnp.asarray(False)
        represented_count: Int64[Array, ""] = jnp.asarray(0, dtype=jnp.int64)
        admitted: Bool[Array, ""] = jnp.asarray(False)
        result: Tuple[Bool[Array, ""], Int64[Array, ""], Bool[Array, ""]] = (
            contained,
            represented_count,
            admitted,
        )
        return result

    support_keys: Int64[Array, " w"] = _flat_residue_keys(support, work_shape)
    order: Int64[Array, " w"] = jnp.argsort(support_keys)
    sorted_keys: Int64[Array, " w"] = support_keys[order]

    def check_pair(
        position: scalar_int,
        carry: Tuple[Bool[Array, ""], Int64[Array, ""]],
    ) -> Tuple[Bool[Array, ""], Int64[Array, ""]]:
        """Accumulate exact pair membership and the represented count."""
        current, count = carry
        left_position: scalar_int = position // right_count
        right_position: scalar_int = position % right_count
        candidate: Int64[Array, " 3"] = (
            left[left_position] + right_sign * right[right_position]
        )
        candidate_key: Int64[Array, ""] = _flat_residue_keys(
            candidate[None, :], work_shape
        )[0]
        location: Int32[Array, ""] = jnp.searchsorted(
            sorted_keys, candidate_key, side="left"
        )
        clipped: Int32[Array, ""] = jnp.clip(location, 0, support.shape[0] - 1)
        member: Bool[Array, ""] = (
            (location < support.shape[0])
            & (sorted_keys[clipped] == candidate_key)
            & jnp.all(support[order[clipped]] == candidate)
        )
        updated: Tuple[Bool[Array, ""], Int64[Array, ""]] = (
            current & member,
            count + member.astype(jnp.int64),
        )
        return updated

    loop_result: Tuple[Bool[Array, ""], Int64[Array, ""]] = jax.lax.fori_loop(
        0,
        pair_count,
        check_pair,
        (
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int64),
        ),
    )
    contained, represented_count = loop_result
    admitted = jnp.asarray(True)
    result = (contained, represented_count, admitted)
    return result  # noqa: RET504


def _canonical_endpoint_predicate(
    indices: Int64[Array, "n d"],
    shape: Tuple[int, ...],
) -> Bool[Array, ""]:
    """PRIVATE: Check a signed half-open endpoint realization.

    Parameters
    ----------
    indices : Int64[Array, "n d"]
        Exact integer indices.
    shape : Tuple[int, ...]
        Realized grid lengths for each coordinate.

    Returns
    -------
    valid : Bool[Array, ""]
        Whether every index lies in the canonical signed interval.
    """
    shape_array: Int64[Array, " d"] = jnp.asarray(shape, dtype=jnp.int64)
    lower: Int64[Array, " d"] = -(shape_array // 2)
    upper: Int64[Array, " d"] = (shape_array - 1) // 2
    valid: Bool[Array, ""] = jnp.all(
        (indices >= lower[None, :]) & (indices <= upper[None, :])
    )
    return valid


def _two_pi_interval() -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Enclose mathematical ``2 pi`` in binary64.

    Returns
    -------
    lower : Float64[Array, ""]
        Downward-rounded lower endpoint.
    upper : Float64[Array, ""]
        Upward-rounded upper endpoint.

    Notes
    -----
    The shared primitive supplies a proved binary64 bracket for mathematical
    pi and fails closed when its required normal-operation probes fail.
    """
    pi_interval = mathematical_pi_interval()
    two: Float64[Array, ""] = jnp.asarray(2.0, dtype=jnp.float64)
    computed_lower, computed_upper = interval_multiply(
        pi_interval,
        point_interval(two),
    )
    lower: Float64[Array, ""] = computed_lower
    upper: Float64[Array, ""] = computed_upper
    result: Tuple[Float64[Array, ""], Float64[Array, ""]] = (lower, upper)
    return result


def _coefficient_wavevector_interval(
    manifest: GalerkinAcquisitionManifest,
    indices: Int64[Array, "n 3"],
) -> Tuple[Float64[Array, "n 3"], Float64[Array, "n 3"]]:
    """PRIVATE: Enclose wavevectors derived from exact lattice indices.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition carrier and physical box.
    indices : Int64[Array, "n 3"]
        Exact reciprocal lattice indices.

    Returns
    -------
    lower : Float64[Array, "n 3"]
        Componentwise lower wavevector endpoints in radians per Angstrom.
    upper : Float64[Array, "n 3"]
        Componentwise upper wavevector endpoints in radians per Angstrom.

    Notes
    -----
    The enclosed formula is ``carrier + 2 * pi * indices / box_lengths``.
    The manifest supplies positive finite box lengths and bounds each index
    by ``2**52``, so its conversion to binary64 is exact. The stored carrier
    is an exact singleton input. Outward interval operations make this a
    certification path rather than a differentiation path.
    """
    index_values: Float64[Array, "n 3"] = indices.astype(jnp.float64)
    box: Float64[Array, " 3"] = manifest.box_lengths
    ratio_lower, ratio_upper = interval_divide_positive(
        point_interval(index_values),
        point_interval(box[None, :]),
    )
    two_pi_lower, two_pi_upper = _two_pi_interval()
    offset_lower, offset_upper = interval_multiply(
        (ratio_lower, ratio_upper),
        (two_pi_lower, two_pi_upper),
    )
    computed_lower, computed_upper = interval_add(
        point_interval(manifest.carrier[None, :]),
        (offset_lower, offset_upper),
    )
    lower: Float64[Array, "n 3"] = computed_lower
    upper: Float64[Array, "n 3"] = computed_upper
    result: Tuple[Float64[Array, "n 3"], Float64[Array, "n 3"]] = (
        lower,
        upper,
    )
    return result


def _norm_upper_from_interval(
    lower: Float64[Array, "... d"],
    upper: Float64[Array, "... d"],
) -> Float64[Array, "..."]:
    """PRIVATE: Bound Euclidean norms of vector intervals outwardly.

    Parameters
    ----------
    lower : Float64[Array, "... d"]
        Componentwise interval lower endpoints.
    upper : Float64[Array, "... d"]
        Componentwise interval upper endpoints.

    Returns
    -------
    upper_root : Float64[Array, "..."]
        Upward-rounded Euclidean-norm upper bounds.

    Notes
    -----
    For each component, the larger endpoint magnitude bounds every value in
    that interval. Outward products and sums bound the squared norm. One
    upward step after ``sqrt`` bounds the norm and preserves an exact zero.
    This helper assumes finite ordered intervals. Its absolute values,
    maxima, and discrete rounding make the bound nondifferentiable.
    """
    component_upper: Float64[Array, "... d"] = jnp.maximum(
        jnp.abs(lower), jnp.abs(upper)
    )
    squared_lower, squared_upper = interval_multiply(
        point_interval(component_upper),
        point_interval(component_upper),
    )
    del squared_lower
    total: Float64[Array, "..."] = jnp.zeros(
        component_upper.shape[:-1], dtype=jnp.float64
    )
    for axis in range(component_upper.shape[-1]):
        _, total = interval_add(
            point_interval(total),
            point_interval(squared_upper[..., axis]),
        )
    upper_root: Float64[Array, "..."] = upward_sqrt(total)
    return upper_root


def _transverse_norm_upper_from_interval(
    lower: Float64[Array, "n 3"],
    upper: Float64[Array, "n 3"],
    carrier: Float64[Array, " 3"],
) -> Float64[Array, " n"]:
    """PRIVATE: Bound norms perpendicular to the carrier direction.

    Parameters
    ----------
    lower : Float64[Array, "n 3"]
        Componentwise cyclic-vector lower endpoints in inverse Angstroms.
    upper : Float64[Array, "n 3"]
        Componentwise cyclic-vector upper endpoints in inverse Angstroms.
    carrier : Float64[Array, " 3"]
        Nonzero angular-wavevector direction seed in radians per Angstrom.

    Returns
    -------
    bounds : Float64[Array, " n"]
        Outward bounds for ``|v cross carrier| / |carrier|`` in inverse
        Angstroms.

    Notes
    -----
    Scaling the seed by its largest component prevents norm underflow without
    changing its direction. Interval arithmetic encloses both the scaled seed
    and the cross-product quotient.
    """
    scale: Float64[Array, ""] = jnp.max(jnp.abs(carrier))
    scaled_lower, scaled_upper = interval_divide_positive(
        point_interval(carrier),
        point_interval(scale),
    )
    cross_lowers: list[Float64[Array, " n"]] = []
    cross_uppers: list[Float64[Array, " n"]] = []
    for left_axis, right_axis in ((1, 2), (2, 0), (0, 1)):
        first_lower, first_upper = interval_multiply(
            (lower[:, left_axis], upper[:, left_axis]),
            (scaled_lower[right_axis], scaled_upper[right_axis]),
        )
        second_lower, second_upper = interval_multiply(
            (lower[:, right_axis], upper[:, right_axis]),
            (scaled_lower[left_axis], scaled_upper[left_axis]),
        )
        component_lower, component_upper = interval_add(
            (first_lower, first_upper),
            (-second_upper, -second_lower),
        )
        cross_lowers.append(component_lower)
        cross_uppers.append(component_upper)
    cross_lower: Float64[Array, "n 3"] = jnp.stack(cross_lowers, axis=-1)
    cross_upper: Float64[Array, "n 3"] = jnp.stack(cross_uppers, axis=-1)
    cross_norm_upper: Float64[Array, " n"] = _norm_upper_from_interval(
        cross_lower, cross_upper
    )

    component_minimum: Float64[Array, " 3"] = jnp.where(
        (scaled_lower <= 0.0) & (scaled_upper >= 0.0),
        0.0,
        jnp.minimum(jnp.abs(scaled_lower), jnp.abs(scaled_upper)),
    )
    seed_square_lower, seed_square_upper = interval_multiply(
        point_interval(component_minimum),
        point_interval(component_minimum),
    )
    del seed_square_upper
    seed_norm_squared_lower: Float64[Array, ""] = jnp.asarray(
        0.0, dtype=jnp.float64
    )
    for axis in range(_SPACE_DIMENSIONS):
        seed_norm_squared_lower, _ = interval_add(
            point_interval(seed_norm_squared_lower),
            point_interval(seed_square_lower[axis]),
        )
    seed_norm_lower: Float64[Array, ""] = downward_sqrt(
        jnp.maximum(seed_norm_squared_lower, 0.0)
    )
    _, computed_bounds = interval_divide_positive(
        (jnp.zeros_like(cross_norm_upper), cross_norm_upper),
        point_interval(seed_norm_lower),
    )
    bounds: Float64[Array, " n"] = computed_bounds
    return bounds


def _shell_defect_upper_bounds(
    wavevectors: Float64[Array, "n 3"],
    wavenumber: Float64[Array, ""],
) -> Float64[Array, " n"]:
    r"""PRIVATE: Enclose represented shell defects outwardly.

    Parameters
    ----------
    wavevectors : Float64[Array, "n 3"]
        Stored physical wavevectors in radians per Angstrom.
    wavenumber : Float64[Array, ""]
        Stored carrier wavenumber in radians per Angstrom.

    Returns
    -------
    bounds : Float64[Array, " n"]
        Upper bounds for :math:`\lvert |k|^2-k_0^2\rvert` in inverse-square
        Angstroms.

    Notes
    -----
    This helper encloses ``abs(sum_j wavevectors[j]**2 - wavenumber**2)``
    for each represented mode. It treats the stored binary64 inputs as exact
    singletons and bounds their arithmetic rounding. It does not include
    projection error or physical-model uncertainty, and the outward bound is
    not a differentiation path.
    """
    vector_lower: Float64[Array, " n"] = jnp.zeros(
        (wavevectors.shape[0],), dtype=jnp.float64
    )
    vector_upper: Float64[Array, " n"] = vector_lower
    for axis in range(_SPACE_DIMENSIONS):
        square_lower, square_upper = interval_multiply(
            point_interval(wavevectors[:, axis]),
            point_interval(wavevectors[:, axis]),
        )
        vector_lower, vector_upper = interval_add(
            (vector_lower, vector_upper),
            (square_lower, square_upper),
        )
    shell_lower, shell_upper = interval_multiply(
        point_interval(wavenumber),
        point_interval(wavenumber),
    )
    defect_lower, defect_upper = interval_add(
        (vector_lower, vector_upper),
        (-shell_upper, -shell_lower),
    )
    bounds: Float64[Array, " n"] = jnp.maximum(
        jnp.abs(defect_lower), jnp.abs(defect_upper)
    )
    return bounds


def _projection_error_upper_bounds(
    manifest: GalerkinAcquisitionManifest,
    indices: Int64[Array, "n 3"],
    physical_wavevectors: Float64[Array, "n 3"],
) -> Float64[Array, " n"]:
    """PRIVATE: Enclose requested-to-coefficient wavevector errors.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition carrier and physical box.
    indices : Int64[Array, "n 3"]
        Exact reciprocal indices for the coefficients.
    physical_wavevectors : Float64[Array, "n 3"]
        Requested physical wavevectors in radians per Angstrom.

    Returns
    -------
    bounds : Float64[Array, " n"]
        Outward Euclidean discrepancy bounds in radians per Angstrom.

    Notes
    -----
    The bounded formula is the Euclidean norm of each requested wavevector
    minus ``carrier + 2 * pi * indices / box_lengths``. Requested binary64
    values enter as exact singletons. The result encloses arithmetic rounding
    but not uncertainty in the requested physical data, and it is not a
    differentiation path.
    """
    coefficient_lower, coefficient_upper = _coefficient_wavevector_interval(
        manifest, indices
    )
    difference_lower, difference_upper = interval_add(
        point_interval(physical_wavevectors),
        (-coefficient_upper, -coefficient_lower),
    )
    bounds: Float64[Array, " n"] = _norm_upper_from_interval(
        difference_lower, difference_upper
    )
    return bounds


def _canonical_coefficient_wavevectors(
    manifest: GalerkinAcquisitionManifest,
    indices: Int64[Array, "n 3"],
) -> Float64[Array, "n 3"]:
    """PRIVATE: Realize exact coefficient wavevectors canonically in binary64.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition carrier and physical box.
    indices : Int64[Array, "n 3"]
        Primary exact reciprocal-lattice indices.

    Returns
    -------
    wavevectors : Float64[Array, "n 3"]
        Canonical binary64 realization of ``k_c + 2 pi g`` in radians per
        Angstrom.

    Notes
    -----
    Exact-coefficient identity is symbolic in ``indices``. This realization
    is used only to require that redundant physical metadata round-trips
    canonically; it does not replace exact integer membership.
    """
    index_values: Float64[Array, "n 3"] = indices.astype(jnp.float64)
    wavevectors: Float64[Array, "n 3"] = (
        manifest.carrier[None, :]
        + (2.0 * jnp.asarray(jnp.pi, dtype=jnp.float64) * index_values)
        / manifest.box_lengths[None, :]
    )
    return wavevectors


def _sector_masks(
    manifest: GalerkinAcquisitionManifest,
    indices: Int64[Array, "n 3"],
) -> Tuple[
    Bool[Array, " n"],
    Bool[Array, " n"],
    Bool[Array, " n"],
    Bool[Array, " n"],
    Float64[Array, " n"],
    Float64[Array, " n"],
]:
    """PRIVATE: Partition modes by outward normal-component intervals.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition carrier, box, terminal axis, and terminal side.
    indices : Int64[Array, "n 3"]
        Exact reciprocal indices to classify.

    Returns
    -------
    forward : Bool[Array, " n"]
        Modes proved to have positive oriented normal component.
    grazing : Bool[Array, " n"]
        Modes proved to have exactly zero normal component.
    backward : Bool[Array, " n"]
        Modes proved to have negative oriented normal component.
    ambiguous : Bool[Array, " n"]
        Modes whose interval crosses or touches zero without equality proof.
    oriented_lower : Float64[Array, " n"]
        Outward lower nominal oriented normal-component endpoints in radians
        per Angstrom.
    oriented_upper : Float64[Array, " n"]
        Outward upper nominal oriented normal-component endpoints in radians
        per Angstrom.

    Notes
    -----
    A mode is forward only when its oriented lower endpoint is strictly
    positive, and backward only when its upper endpoint is strictly negative.
    Grazing requires both endpoints to equal zero exactly. Every other
    interval that touches or crosses zero is ambiguous; no tolerance assigns
    a sector at that boundary.
    """
    lower, upper = _coefficient_wavevector_interval(manifest, indices)
    normal_lower: Float64[Array, " n"] = lower[:, manifest.terminal_axis]
    normal_upper: Float64[Array, " n"] = upper[:, manifest.terminal_axis]
    if manifest.terminal_side == GalerkinTerminalSide.NEGATIVE:
        oriented_lower: Float64[Array, " n"] = -normal_upper
        oriented_upper: Float64[Array, " n"] = -normal_lower
    else:
        oriented_lower = normal_lower
        oriented_upper = normal_upper
    forward: Bool[Array, " n"] = oriented_lower > 0.0
    backward: Bool[Array, " n"] = oriented_upper < 0.0
    grazing: Bool[Array, " n"] = (oriented_lower == 0.0) & (
        oriented_upper == 0.0
    )
    ambiguous: Bool[Array, " n"] = ~(forward | backward | grazing)
    result: Tuple[
        Bool[Array, " n"],
        Bool[Array, " n"],
        Bool[Array, " n"],
        Bool[Array, " n"],
        Float64[Array, " n"],
        Float64[Array, " n"],
    ] = (
        forward,
        grazing,
        backward,
        ambiguous,
        oriented_lower,
        oriented_upper,
    )
    return result


def _cyclic_offset_norm_maxima(
    manifest: GalerkinAcquisitionManifest,
    wavevectors: Float64[Array, "n 3"],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Bound transverse and full cyclic offsets outwardly.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition carrier-direction convention.
    wavevectors : Float64[Array, "n 3"]
        Physical angular wavevectors in radians per Angstrom.

    Returns
    -------
    transverse_max : Float64[Array, ""]
        Maximum transverse cyclic-offset norm in inverse Angstroms.
    full_max : Float64[Array, ""]
        Maximum full cyclic-offset norm in inverse Angstroms.

    Notes
    -----
    Subtracting the carrier gives an angular offset. Division by ``2 * pi``
    converts that offset to cyclic inverse-Angstrom units. Outward interval
    arithmetic encloses the subtraction, normalization, and each norm before
    the maximum selects the largest per-mode upper bound.
    """
    angular_lower, angular_upper = interval_add(
        point_interval(wavevectors),
        point_interval(-manifest.carrier[None, :]),
    )
    two_pi_lower, two_pi_upper = _two_pi_interval()
    cyclic_lower, cyclic_upper = interval_divide_positive(
        (angular_lower, angular_upper),
        (two_pi_lower, two_pi_upper),
    )
    transverse_bounds: Float64[Array, " n"] = (
        _transverse_norm_upper_from_interval(
            cyclic_lower, cyclic_upper, manifest.carrier
        )
    )
    full_bounds: Float64[Array, " n"] = _norm_upper_from_interval(
        cyclic_lower, cyclic_upper
    )
    transverse_max: Float64[Array, ""] = jnp.max(transverse_bounds)
    full_max: Float64[Array, ""] = jnp.max(full_bounds)
    result: Tuple[Float64[Array, ""], Float64[Array, ""]] = (
        transverse_max,
        full_max,
    )
    return result


def _transfer_norm_maxima(
    manifest: GalerkinAcquisitionManifest,
    max_pair_checks: int,
) -> Tuple[Float64[Array, ""], Float64[Array, ""], Bool[Array, ""]]:
    """PRIVATE: Bound all requested direction transfers with fixed memory.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition incident and outgoing physical wavevectors.
    max_pair_checks : int
        Maximum admitted incident-outgoing pair count.

    Returns
    -------
    transverse_max : Float64[Array, ""]
        Maximum transverse cyclic-transfer norm in inverse Angstroms.
    full_max : Float64[Array, ""]
        Maximum full cyclic-transfer norm in inverse Angstroms.
    admitted : Bool[Array, ""]
        Whether the pair count stayed within the checker ceiling.

    Notes
    -----
    Each outgoing-minus-incident angular transfer is divided by ``2 * pi``
    to obtain a cyclic transfer in inverse Angstroms. Outward intervals
    enclose every admitted pair before the maxima are accumulated. If the
    pair count exceeds ``max_pair_checks``, both numeric outputs are sentinel
    zeros and ``admitted`` is false; the zeros are not certified maxima.
    """
    incident = manifest.incident_physical_wavevectors
    outgoing = manifest.outgoing_physical_wavevectors
    incident_count: int = incident.shape[0]
    pair_count: int = incident_count * outgoing.shape[0]
    if pair_count > max_pair_checks:
        zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
        transverse_max: Float64[Array, ""] = zero
        full_max: Float64[Array, ""] = zero
        admitted: Bool[Array, ""] = jnp.asarray(False)
        result: Tuple[
            Float64[Array, ""], Float64[Array, ""], Bool[Array, ""]
        ] = (transverse_max, full_max, admitted)
        return result

    two_pi_lower, two_pi_upper = _two_pi_interval()

    def check_pair(
        position: scalar_int,
        maxima: Tuple[Float64[Array, ""], Float64[Array, ""]],
    ) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
        """Accumulate transverse and full physical transfer maxima."""
        outgoing_position: scalar_int = position // incident_count
        incident_position: scalar_int = position % incident_count
        difference_lower, difference_upper = interval_add(
            point_interval(outgoing[outgoing_position]),
            point_interval(-incident[incident_position]),
        )
        cyclic_lower, cyclic_upper = interval_divide_positive(
            (difference_lower, difference_upper),
            (two_pi_lower, two_pi_upper),
        )
        transverse: Float64[Array, ""] = _transverse_norm_upper_from_interval(
            cyclic_lower[None, :],
            cyclic_upper[None, :],
            manifest.carrier,
        )[0]
        full: Float64[Array, ""] = _norm_upper_from_interval(
            cyclic_lower[None, :], cyclic_upper[None, :]
        )[0]
        updated: Tuple[Float64[Array, ""], Float64[Array, ""]] = (
            jnp.maximum(maxima[0], transverse),
            jnp.maximum(maxima[1], full),
        )
        return updated

    transverse_max, full_max = jax.lax.fori_loop(
        0,
        pair_count,
        check_pair,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
    )
    admitted = jnp.asarray(True)
    result = (transverse_max, full_max, admitted)
    return result  # noqa: RET504


def _terminal_fiber_predicate(
    manifest: GalerkinAcquisitionManifest,
) -> Bool[Array, ""]:
    """PRIVATE: Check complete selected preterminal state fibers.

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Acquisition state, transverse, and preterminal index sets.

    Returns
    -------
    complete : Bool[Array, ""]
        Whether ``K_d`` equals the selected complete ``K_u`` fibers.
    """
    state: Int64[Array, "n 3"] = manifest.support.state_indices
    if manifest.terminal_axis == 0:
        state_transverse: Int64[Array, "n 2"] = state[:, 1:]
    elif manifest.terminal_axis == 1:
        state_transverse = state[:, (0, 2)]
    else:
        state_transverse = state[:, :2]

    def check_state(
        position: scalar_int,
        current: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Compare selected-transverse and preterminal membership."""
        selected: Bool[Array, ""] = jnp.any(
            jnp.all(
                manifest.transverse_indices == state_transverse[position],
                axis=-1,
            )
        )
        retained: Bool[Array, ""] = _member_mask(
            state[position][None, :],
            manifest.preterminal_indices,
            manifest.support.work_shape,
        )[0]
        updated: Bool[Array, ""] = current & (selected == retained)
        return updated

    complete: Bool[Array, ""] = jax.lax.fori_loop(
        0,
        state.shape[0],
        check_state,
        jnp.asarray(True),
    )
    return complete


@jaxtyped(typechecker=beartype)
def check_galerkin_acquisition_support(  # noqa: PLR0915
    manifest: GalerkinAcquisitionManifest,
) -> GalerkinAcquisitionSupportResult:
    """Check one bounded acquisition manifest against RM-S1.

    :see: :class:`~.test_acquisition.TestGalerkinAcquisitionSupport`

    Parameters
    ----------
    manifest : GalerkinAcquisitionManifest
        Complete RM-S1 support-core submission for one independent carrier.

    Returns
    -------
    result : GalerkinAcquisitionSupportResult
        Fail-closed structural and RM-S1 support-core artifact.

    Notes
    -----
    Integer predicates are exact and use constant working memory. Physical
    shell, projection, sector, and geometry quantities are independently
    recomputed with outward binary64 intervals. An interval containing zero
    is reported as ambiguous, never assigned to grazing by a tolerance.
    """
    support: GalerkinProductSupport = manifest.support
    shape: Tuple[int, int, int] = support.work_shape
    all_sets: Tuple[Int64[Array, "..."], ...] = (
        support.state_indices,
        support.interaction_indices,
        support.absorber_indices,
        support.work_indices,
        manifest.incident_indices,
        manifest.elastic_outgoing_indices,
        manifest.preterminal_indices,
        manifest.transverse_indices,
        manifest.deliberately_omitted_indices,
    )
    unique: Bool[Array, ""] = jnp.asarray(True)
    for indices in all_sets:
        unique = unique & ~_contains_duplicates(indices)

    transverse_axis_pair: Tuple[int, int] = _transverse_axes(
        manifest.terminal_axis
    )
    transverse_shape: Tuple[int, int] = (
        shape[transverse_axis_pair[0]],
        shape[transverse_axis_pair[1]],
    )
    endpoint_valid: Bool[Array, ""] = jnp.asarray(
        manifest.endpoint_convention
        == GalerkinEndpointConvention.SIGNED_HALF_OPEN
    )
    for indices in all_sets[:7]:
        endpoint_valid = endpoint_valid & _canonical_endpoint_predicate(
            indices, shape
        )
    endpoint_valid = endpoint_valid & _canonical_endpoint_predicate(
        manifest.transverse_indices,
        transverse_shape,
    )
    incident_in_state: Bool[Array, ""] = _all_members(
        manifest.incident_indices,
        support.state_indices,
        shape,
    )
    outgoing_in_preterminal: Bool[Array, ""] = _all_members(
        manifest.elastic_outgoing_indices,
        manifest.preterminal_indices,
        shape,
    )
    preterminal_in_state: Bool[Array, ""] = _all_members(
        manifest.preterminal_indices,
        support.state_indices,
        shape,
    )

    (
        direct_transfers_represented,
        represented_transfer_count,
        direct_admitted,
    ) = _binary_relation(
        manifest.preterminal_indices,
        manifest.incident_indices,
        support.interaction_indices,
        shape,
        -1,
        _MAX_BINARY_PAIR_CHECKS,
    )
    (
        absorber_differences_represented,
        _,
        absorber_admitted,
    ) = _binary_relation(
        support.state_indices,
        support.state_indices,
        support.absorber_indices,
        shape,
        -1,
        _MAX_BINARY_PAIR_CHECKS,
    )
    (
        work_products_represented,
        _,
        work_admitted,
    ) = _binary_relation(
        support.state_indices,
        support.interaction_indices,
        support.work_indices,
        shape,
        1,
        _MAX_BINARY_PAIR_CHECKS,
    )
    interaction_sign_symmetric: Bool[Array, ""] = _all_members(
        -support.interaction_indices,
        support.interaction_indices,
        shape,
    )
    absorber_sign_symmetric: Bool[Array, ""] = _all_members(
        -support.absorber_indices,
        support.absorber_indices,
        shape,
    )
    terminal_fiber_complete: Bool[Array, ""] = _terminal_fiber_predicate(
        manifest
    )

    omitted_mask_valid: Bool[Array, ""] = ~_contains_duplicates(
        manifest.deliberately_omitted_indices
    ) & ~jnp.any(
        _member_mask(
            manifest.deliberately_omitted_indices,
            support.state_indices,
            shape,
        )
    )
    (
        state_forward_mask,
        state_grazing_mask,
        state_backward_mask,
        state_ambiguous_mask,
        state_oriented_normal_lower,
        state_oriented_normal_upper,
    ) = _sector_masks(manifest, support.state_indices)
    (
        omitted_forward_mask,
        omitted_grazing_mask,
        omitted_backward_mask,
        omitted_ambiguous_mask,
        omitted_oriented_normal_lower,
        omitted_oriented_normal_upper,
    ) = _sector_masks(manifest, manifest.deliberately_omitted_indices)
    state_partition_count: Int32[Array, " n"] = (
        state_forward_mask.astype(jnp.int32)
        + state_grazing_mask.astype(jnp.int32)
        + state_backward_mask.astype(jnp.int32)
        + state_ambiguous_mask.astype(jnp.int32)
    )
    omitted_partition_count: Int32[Array, " v"] = (
        omitted_forward_mask.astype(jnp.int32)
        + omitted_grazing_mask.astype(jnp.int32)
        + omitted_backward_mask.astype(jnp.int32)
        + omitted_ambiguous_mask.astype(jnp.int32)
    )
    sector_masks_valid: Bool[Array, ""] = (
        jnp.all(state_partition_count == 1)
        & jnp.all(omitted_partition_count == 1)
        & omitted_mask_valid
    )
    sector_classification_complete: Bool[Array, ""] = ~jnp.any(
        state_ambiguous_mask
    ) & ~jnp.any(omitted_ambiguous_mask)

    if manifest.backward_disposition == GalerkinBackwardDisposition.EXCLUDED:
        backward_disposition_valid: Bool[Array, ""] = (
            ~jnp.any(state_backward_mask)
            & ~jnp.any(state_ambiguous_mask)
            & jnp.asarray(bool(manifest.backward_exclusion_basis.strip()))
            & jnp.asarray(not manifest.claims_backscatter)
        )
    else:
        represented_claim_valid: Bool[Array, ""] = jnp.asarray(
            not manifest.claims_backscatter
        ) | (
            ~jnp.any(omitted_backward_mask) & ~jnp.any(omitted_ambiguous_mask)
        )
        backward_disposition_valid = (
            jnp.any(state_backward_mask)
            & ~jnp.any(state_ambiguous_mask)
            & jnp.asarray(not bool(manifest.backward_exclusion_basis.strip()))
            & represented_claim_valid
        )

    carrier_shell_defect: Float64[Array, ""] = _shell_defect_upper_bounds(
        manifest.carrier[None, :], manifest.wavenumber
    )[0]
    carrier_contract_valid: Bool[Array, ""] = (
        (carrier_shell_defect <= manifest.carrier_on_shell_defect_bound)
        & (
            manifest.carrier_on_shell_defect_bound
            <= manifest.on_shell_defect_tolerance
        )
        & jnp.asarray(
            manifest.carrier_ownership
            == GalerkinCarrierOwnership.INDEPENDENT_SINGLE_CARRIER
        )
        & jnp.asarray(
            manifest.carrier_overlap_disposition
            == GalerkinCarrierOverlapDisposition.NO_OTHER_CARRIER_BLOCKS
        )
        & jnp.asarray(
            manifest.carrier_target_route
            == GalerkinCarrierTargetRoute.NORMALIZE_FROM_ACCELERATING_VOLTAGE
        )
    )

    incident_shell_defects: Float64[Array, " i"] = _shell_defect_upper_bounds(
        manifest.incident_physical_wavevectors,
        manifest.wavenumber,
    )
    outgoing_shell_defects: Float64[Array, " o"] = _shell_defect_upper_bounds(
        manifest.outgoing_physical_wavevectors,
        manifest.wavenumber,
    )
    incident_projection_diagnostics: Float64[Array, " i"] = (
        _projection_error_upper_bounds(
            manifest,
            manifest.incident_indices,
            manifest.incident_physical_wavevectors,
        )
    )
    outgoing_projection_diagnostics: Float64[Array, " o"] = (
        _projection_error_upper_bounds(
            manifest,
            manifest.elastic_outgoing_indices,
            manifest.outgoing_physical_wavevectors,
        )
    )
    incident_codes_valid: Bool[Array, " i"] = (
        manifest.incident_direction_dispositions
        == GalerkinDirectionDisposition.EXACT_COEFFICIENT
    ) | (
        manifest.incident_direction_dispositions
        == GalerkinDirectionDisposition.PROJECTED
    )
    outgoing_codes_valid: Bool[Array, " o"] = (
        manifest.outgoing_direction_dispositions
        == GalerkinDirectionDisposition.EXACT_COEFFICIENT
    ) | (
        manifest.outgoing_direction_dispositions
        == GalerkinDirectionDisposition.PROJECTED
    )
    incident_exact: Bool[Array, " i"] = (
        manifest.incident_direction_dispositions
        == GalerkinDirectionDisposition.EXACT_COEFFICIENT
    )
    outgoing_exact: Bool[Array, " o"] = (
        manifest.outgoing_direction_dispositions
        == GalerkinDirectionDisposition.EXACT_COEFFICIENT
    )
    canonical_incident_wavevectors: Float64[Array, "i 3"] = (
        _canonical_coefficient_wavevectors(manifest, manifest.incident_indices)
    )
    canonical_outgoing_wavevectors: Float64[Array, "o 3"] = (
        _canonical_coefficient_wavevectors(
            manifest, manifest.elastic_outgoing_indices
        )
    )
    incident_exact_roundtrip: Bool[Array, " i"] = jnp.all(
        manifest.incident_physical_wavevectors
        == canonical_incident_wavevectors,
        axis=-1,
    )
    outgoing_exact_roundtrip: Bool[Array, " o"] = jnp.all(
        manifest.outgoing_physical_wavevectors
        == canonical_outgoing_wavevectors,
        axis=-1,
    )
    incident_exact_zero_index: Bool[Array, " i"] = jnp.all(
        manifest.incident_indices == 0, axis=-1
    )
    outgoing_exact_zero_index: Bool[Array, " o"] = jnp.all(
        manifest.elastic_outgoing_indices == 0, axis=-1
    )
    incident_projection_errors: Float64[Array, " i"] = jnp.where(
        incident_exact, 0.0, incident_projection_diagnostics
    )
    outgoing_projection_errors: Float64[Array, " o"] = jnp.where(
        outgoing_exact, 0.0, outgoing_projection_diagnostics
    )
    direction_evidence_valid: Bool[Array, ""] = (
        jnp.all(incident_codes_valid)
        & jnp.all(outgoing_codes_valid)
        & jnp.all(
            incident_shell_defects <= manifest.incident_on_shell_defect_bounds
        )
        & jnp.all(
            outgoing_shell_defects <= manifest.outgoing_on_shell_defect_bounds
        )
        & jnp.all(
            manifest.incident_on_shell_defect_bounds
            <= manifest.on_shell_defect_tolerance
        )
        & jnp.all(
            manifest.outgoing_on_shell_defect_bounds
            <= manifest.on_shell_defect_tolerance
        )
        & jnp.all(
            incident_projection_errors
            <= manifest.incident_projection_error_bounds
        )
        & jnp.all(
            outgoing_projection_errors
            <= manifest.outgoing_projection_error_bounds
        )
        & jnp.all(
            ~incident_exact
            | (
                incident_exact_roundtrip
                & incident_exact_zero_index
                & (manifest.incident_projection_error_bounds == 0.0)
            )
        )
        & jnp.all(
            ~outgoing_exact
            | (
                outgoing_exact_roundtrip
                & outgoing_exact_zero_index
                & (manifest.outgoing_projection_error_bounds == 0.0)
            )
        )
    )

    incident_transverse_max, incident_full_max = _cyclic_offset_norm_maxima(
        manifest, manifest.incident_physical_wavevectors
    )
    outgoing_transverse_max, outgoing_full_max = _cyclic_offset_norm_maxima(
        manifest, manifest.outgoing_physical_wavevectors
    )
    transfer_transverse_max, transfer_full_max, geometry_admitted = (
        _transfer_norm_maxima(manifest, _MAX_BINARY_PAIR_CHECKS)
    )
    check_capacity_admitted: Bool[Array, ""] = (
        direct_admitted & absorber_admitted & work_admitted & geometry_admitted
    )

    direct_pair_count: Int64[Array, ""] = jnp.asarray(
        manifest.preterminal_indices.shape[0]
        * manifest.incident_indices.shape[0],
        dtype=jnp.int64,
    )
    result: GalerkinAcquisitionSupportResult = (
        _create_galerkin_acquisition_support_result(
            manifest=manifest,
            unique=unique,
            endpoint_valid=endpoint_valid,
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
            sector_classification_complete=(sector_classification_complete),
            omitted_mask_valid=omitted_mask_valid,
            state_forward_mask=state_forward_mask,
            state_grazing_mask=state_grazing_mask,
            state_backward_mask=state_backward_mask,
            state_ambiguous_mask=state_ambiguous_mask,
            state_oriented_normal_interval_lower=(state_oriented_normal_lower),
            state_oriented_normal_interval_upper=(state_oriented_normal_upper),
            omitted_forward_mask=omitted_forward_mask,
            omitted_grazing_mask=omitted_grazing_mask,
            omitted_backward_mask=omitted_backward_mask,
            omitted_ambiguous_mask=omitted_ambiguous_mask,
            omitted_oriented_normal_interval_lower=(
                omitted_oriented_normal_lower
            ),
            omitted_oriented_normal_interval_upper=(
                omitted_oriented_normal_upper
            ),
            carrier_shell_defect_upper_bound=carrier_shell_defect,
            incident_shell_defect_upper_bounds=incident_shell_defects,
            outgoing_shell_defect_upper_bounds=outgoing_shell_defects,
            incident_projection_error_upper_bounds=(
                incident_projection_errors
            ),
            outgoing_projection_error_upper_bounds=(
                outgoing_projection_errors
            ),
            incident_transverse_offset_max=incident_transverse_max,
            incident_full_offset_max=incident_full_max,
            outgoing_transverse_offset_max=outgoing_transverse_max,
            outgoing_full_offset_max=outgoing_full_max,
            transfer_transverse_max=transfer_transverse_max,
            transfer_full_max=transfer_full_max,
            direct_transfer_pair_count=direct_pair_count,
            represented_direct_transfer_pair_count=(
                represented_transfer_count
            ),
            max_binary_pair_checks=_MAX_BINARY_PAIR_CHECKS,
        )
    )
    return result


__all__: list[str] = [
    "check_galerkin_acquisition_support",
]
