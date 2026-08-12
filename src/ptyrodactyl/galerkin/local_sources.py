r"""Realize and directly certify the disjoint LVT.20 additional source.

Extended Summary
----------------
This leaf implements only the additional local source

``S_add(m) = sqrt(|Omega|) c_q(m)``

on the completed ``LOCAL_CELL_LVT1`` target.  It does not build a represented
plane/focused incident field, the matched total source, a zero slab, or a
vacuum-terminal predicate.

Routine Listings
----------------
:func:`apply_local_cell_additional_source_map`
    Apply rounded LVT.20b--LVT.20c without Hermitian projection.
:func:`apply_local_cell_additional_source_metric_adjoint`
    Apply the frozen rounded linear factors' formal metric adjoint.
:func:`certify_local_additional_source`
    Full-replay the parent/map and directly certify exact LVT.20c.
:func:`prepare_local_additional_source_certificate`
    Full-replay target, map, q rectangles, budget, and all digests.
:func:`realize_local_cell_additional_source`
    Full-replay the target and realize complex LVT.20a--LVT.20c.
:func:`realize_zero_local_additional_source`
    Full-replay the target and build the empty-carrier ZERO route.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
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
    Bool,
    Complex,
    Complex128,
    Float64,
    Int64,
    jaxtyped,
)
from numpy.typing import NDArray

from ptyrodactyl._tools import (
    ComplexRectangle,
    RootEnclosureError,
    complex_rectangle_multiply,
    fraction_from_float,
    fraction_lower_float,
    fraction_upper_float,
    has_subnormal_components,
    host_binary64_supported,
    normalized_sinc_integer_ratio,
    pairwise_rectangle_sum,
    rational_turn_exponential,
    real_interval_product,
    scale_complex_rectangle,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.types import (
    GalerkinLocalAdditionalSource,
    GalerkinLocalAdditionalSourceCertificate,
    GalerkinLocalAdditionalSourceCertificateFailure,
    GalerkinLocalAdditionalSourceRoute,
    GalerkinLocalCellTargetManifest,
    _make_local_additional_source,
    _make_local_additional_source_certificate,
)

from .local_cell import (
    _local_cell_shape_factors,
    _negative_origin_phase,
    _normalized_fft_adjoint_gain,
    _normalized_fft_scales,
    _origin_cycle_fractions,
    _physical_cell_volume,
)
from .local_cell_system import prepare_local_cell_galerkin_target

_ADJOINT_FORMULA: str = (
    "formal transpose of the separately frozen algebraic_volume_sqrt, FFT "
    "scales, sinc/origin factors under the separately rounded physical-cell "
    "metric DeltaV; complex; targets exact LVT.20d without identifying the "
    "rounded factors with |Omega|^(-1/2); excludes actual per-call rounding"
)
_ARITHMETIC: str = (
    "guarded IEEE binary64 host; exact Fraction complex-cell direct mean "
    "DFT; exact unwrapped integer sinc arguments with symbolic zeros; "
    "Machin rational pi; alternating rational sin/cos; binary pairwise "
    "accumulation; verified rational square roots; outward binary64 endpoints"
)
_CELL_SUPPORT_CONVENTION: str = (
    "same centered half-open periodic cells, xyz physical axes, and zyx "
    "storage order as the bound LocalCellPotential3D"
)
_CELL_VALUE_UNITS: str = "envelope field per squared Angstrom"
_CERTIFICATE_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_source.lvt20c_direct_certificate.v1"
)
_COEFFICIENT_FORMULA: str = (
    "LVT.20a disjoint ZERO exact-zero vector or LVT.20b LOCAL_CELL complex "
    "mean DFT times centered-cell sinc and negative cell-center-origin "
    "phase; no Hermitian projection"
)
_COEFFICIENT_NORM: str = "LVT.20e Euclidean retained-coefficient norm"
_DEFAULT_MAXIMUM_DIRECT_TERMS: int = 2_000_000
_ERROR_SCOPE: str = (
    "LVT.20e complete sqrt(|Omega|) c_q component rectangles and Euclidean "
    "source-norm transfer only; excludes Dv, Bv, matched/total source, "
    "per-call arithmetic, solver, slab, projection-defect, and terminal errors"
)
_EXACT_TARGET: str = (
    "LVT.20c exact disjoint ZERO or LOCAL_CELL additional-source "
    "coefficients on ordered I_u"
)
_MAP_ARITHMETIC: str = (
    "fixed power-of-two normalized complex FFT, modular bin gather, "
    "unwrapped sinc/origin factors, one intentionally approximate frozen "
    "binary64 sqrt(|Omega|) multiplier, and no Hermitian projection"
)
_MAXIMUM_DIRECT_TERMS: int = np.iinfo(np.int64).max
_REALIZATION_DIGEST_DOMAIN: str = (
    "ptyrodactyl.local_source.lvt20c_realization_evidence.v1"
)
_SOURCE_DIGEST_DOMAIN: str = "ptyrodactyl.local_source.lvt20a_identity.v1"
_SOURCE_FORMULA: str = (
    "LVT.20c S_add[m] = 0 on ZERO or sqrt(|Omega|) c_q(m) on LOCAL_CELL "
    "for m in ordered I_u"
)
_TERM_COUNT_ROUTE: str = (
    "lvt20c-all-Iu-nonsymbolic-sinc-complex-cell-products-v1"
)
_ZERO_RECTANGLE: ComplexRectangle = (
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)


def _assert_concrete(value: object) -> None:
    """PRIVATE: Reject traced leaves at the direct host boundary.

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
            "direct local-source certification requires concrete host values"
        )


def _box_volume_fraction(target: GalerkinLocalCellTargetManifest) -> Fraction:
    """PRIVATE: Compute the exact stored-binary64 target box volume.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target that owns the local box geometry.

    Returns
    -------
    volume : Fraction
        Exact rational product of the three stored box lengths.

    Raises
    ------
    ValueError
        If the resulting box volume is not positive.
    """
    volume: Fraction = Fraction(1)
    for length in target.local_potential.box_size:
        volume *= fraction_from_float(length)
    if volume <= 0:
        raise ValueError("target box volume must be positive")
    return volume


def _rounded_box_volume_sqrt(
    target: GalerkinLocalCellTargetManifest,
) -> Float64[Array, ""]:
    """PRIVATE: Round the frozen algebraic volume-square-root multiplier.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target that owns the local box geometry.

    Returns
    -------
    volume_sqrt : Float64[Array, ""]
        Positive binary64 approximation frozen into the rounded map.

    Raises
    ------
    ValueError
        If the multiplier cannot be represented as a positive finite binary64
        value.
    """
    exact_upper = sqrt_fraction_upper(_box_volume_fraction(target))
    try:
        rounded = float(exact_upper)
    except OverflowError as error:
        raise ValueError(
            "sqrt target box volume must remain finite in binary64"
        ) from error
    if not math.isfinite(rounded) or rounded < float(
        np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "sqrt target box volume must remain positive, finite, and normal"
        )
    volume_sqrt: Float64[Array, ""] = jnp.asarray(rounded, dtype=jnp.float64)
    return volume_sqrt


def _checked_physical_cell_volume(
    target: GalerkinLocalCellTargetManifest,
) -> Float64[Array, ""]:
    """PRIVATE: Require a positive normal rounded physical-cell metric.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Target that owns the local grid and box geometry.

    Returns
    -------
    cell_volume : Float64[Array, ""]
        Positive finite normal binary64 physical-cell metric.
    """
    raw_cell_volume = _physical_cell_volume(target.local_potential)
    cell_volume: Float64[Array, ""] = eqx.error_if(
        raw_cell_volume,
        (~jnp.isfinite(raw_cell_volume))
        | (raw_cell_volume < jnp.finfo(jnp.float64).tiny),
        "physical cell volume must remain positive, finite, and normal in "
        "local-source arithmetic",
    )
    return cell_volume


def _grid_shape_xyz(
    target: GalerkinLocalCellTargetManifest,
) -> Tuple[int, int, int]:
    """PRIVATE: Read the source-grid shape in physical xyz order.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Target that owns the zyx-stored local grid.

    Returns
    -------
    nx : int
        Grid length along the physical x axis.
    ny : int
        Grid length along the physical y axis.
    nz : int
        Grid length along the physical z axis.
    """
    nz, ny, nx = target.local_potential.cell_values.shape
    shape_xyz: Tuple[int, int, int] = (nx, ny, nz)
    return shape_xyz


def _checked_target(
    target: GalerkinLocalCellTargetManifest,
) -> GalerkinLocalCellTargetManifest:
    """PRIVATE: Check the action type without replaying the parent proof.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Candidate local-cell Galerkin target.

    Returns
    -------
    checked_target : GalerkinLocalCellTargetManifest
        Target after the public carrier-type check.

    Raises
    ------
    TypeError
        If the input is not a local-cell Galerkin target manifest.
    """
    if not isinstance(target, GalerkinLocalCellTargetManifest):
        raise TypeError(
            "target must be GalerkinLocalCellTargetManifest; legacy targets "
            "cannot enter the LVT.20 local-source map"
        )
    checked_target: GalerkinLocalCellTargetManifest = target
    return checked_target


def _checked_source_cells(
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex[Array, "..."],
) -> Complex128[Array, "nz ny nx"]:
    """PRIVATE: Check one complex128 q field on the exact target grid.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Target that owns the required q-cell grid.
    source_cell_values : Complex[Array, "..."]
        Candidate complex q-cell values.

    Returns
    -------
    checked_cells : Complex128[Array, "nz ny nx"]
        Finite normal-or-zero complex128 values on the target grid.

    Raises
    ------
    ValueError
        If dtype or shape does not exactly match the LOCAL_CELL contract.
    """
    raw = jnp.asarray(source_cell_values)
    if raw.dtype != jnp.dtype(jnp.complex128):
        raise ValueError(
            "LOCAL_CELL source_cell_values must have complex128 dtype"
        )
    values: Complex128[Array, "nz ny nx"] = raw
    if values.shape != target.local_potential.cell_values.shape:
        raise ValueError(
            "LOCAL_CELL source_cell_values must match the target local grid"
        )
    checked_cells: Complex128[Array, "nz ny nx"] = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | has_subnormal_components(values),
        "LOCAL_CELL source_cell_values must be finite and normal-or-zero",
    )
    return checked_cells


def _prepared_local_cell_map(
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex128[Array, "nz ny nx"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply the rounded complete LVT.20c map to checked cells.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target that owns the exact retained support and geometry.
    source_cell_values : Complex128[Array, "nz ny nx"]
        Checked complex q values in zyx storage order.

    Returns
    -------
    additional_source : Complex128[Array, " n"]
        Rounded retained ``sqrt(|Omega|) c_q`` vector.
    """
    cell_volume: Float64[Array, ""] = _checked_physical_cell_volume(target)
    nz, ny, nx = source_cell_values.shape
    sample_count = nx * ny * nz
    scale_exponent, output_scale = _normalized_fft_scales(sample_count)
    input_scale = math.ldexp(1.0, -scale_exponent)
    full_coefficients = (
        jnp.fft.fftn(source_cell_values * input_scale) * output_scale
    )
    indices = target.state_indices
    grid_shape_xyz = jnp.asarray((nx, ny, nz), dtype=jnp.int64)
    residues = jnp.mod(indices, grid_shape_xyz[None, :])
    selected = full_coefficients[
        residues[:, 2], residues[:, 1], residues[:, 0]
    ]
    factors = _local_cell_shape_factors(indices, grid_shape_xyz)
    origin_cycles = _origin_cycle_fractions(target.local_potential)
    negative_phase = _negative_origin_phase(
        indices, origin_cycles, factors != 0.0
    )
    volume_sqrt = _rounded_box_volume_sqrt(target)
    result = volume_sqrt * selected * factors * negative_phase
    additional_source: Complex128[Array, " n"] = eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result))
        | has_subnormal_components(result)
        | (~jnp.isfinite(cell_volume)),
        "rounded LVT.20c source map left finite normal binary64 range",
    )
    return additional_source


@jaxtyped(typechecker=beartype)
def apply_local_cell_additional_source_map(
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    """Apply rounded LVT.20b--LVT.20c without Hermitian projection.

    :see: :func:`~.test_local_sources.\
test_complex_map_matches_direct_sum_without_hermitian_projection`

    ``source_cell_values`` must be complex128 on the exact target local grid
    and have units of envelope field per squared Angstrom.  The normalization
    targets ``sqrt(|Omega|) / N`` times the unnormalized complex DFT,
    centered-cell sinc, and physical-origin phase using separately frozen
    binary64 factors.

    A public target crossing a trust boundary must first be returned by
    :func:`prepare_local_cell_galerkin_target`.  Transform callers close over
    that prepared value; host certification independently performs the full
    replay itself.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared local-cell target that owns the retained support and grid.
    source_cell_values : Complex[Array, "..."]
        Complex128 q values on exactly the target local grid.

    Returns
    -------
    additional_source : Complex128[Array, " n"]
        Rounded retained ``sqrt(|Omega|) c_q`` vector.

    Raises
    ------
    TypeError
        If ``target`` has the wrong carrier type.
    ValueError
        If q dtype or shape violates the LOCAL_CELL contract.
    """
    checked_target = _checked_target(target)
    cells = _checked_source_cells(checked_target, source_cell_values)
    additional_source: Complex128[Array, " n"] = _prepared_local_cell_map(
        checked_target, cells
    )
    return additional_source


@jaxtyped(typechecker=beartype)
def apply_local_cell_additional_source_metric_adjoint(
    target: GalerkinLocalCellTargetManifest,
    coefficient_cotangent: Complex[Array, "..."],
) -> Complex128[Array, "nz ny nx"]:
    """Apply the frozen rounded linear factors' formal metric adjoint.

    :see: :func:`~.test_local_sources.\
test_complex_metric_adjoint_dense_dot_jit_and_realified_vjp`

    The returned complex grid satisfies the realified identity

    ``Re DeltaV sum conj(eta) * C* z = Re sum conj(C eta) * z``.

    No real part and no Hermitian projection is applied to the source-cell
    cotangent.  This transposes the declared frozen binary64 factors; it does
    not claim bitwise transposition through each rounded multiply/FFT event
    and supplies no actual per-call rounding-error certificate.

    A public target crossing a trust boundary must first be returned by
    :func:`prepare_local_cell_galerkin_target`; transform callers close over
    that prepared target.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared local-cell target that owns the retained support and grid.
    coefficient_cotangent : Complex[Array, "..."]
        Complex cotangent on the ordered retained support.

    Returns
    -------
    source_cell_gradient : Complex128[Array, "nz ny nx"]
        Formal adjoint grid under the physical-cell metric.

    Raises
    ------
    TypeError
        If ``target`` has the wrong carrier type.
    ValueError
        If the cotangent is not a matching one-dimensional vector.
    """
    checked_target = _checked_target(target)
    cotangent = jnp.asarray(coefficient_cotangent, dtype=jnp.complex128)
    if cotangent.ndim != 1:
        raise ValueError("coefficient_cotangent must be 1D")
    if cotangent.shape != (checked_target.state_indices.shape[0],):
        raise ValueError("coefficient_cotangent must match target I_u")
    checked_cotangent = eqx.error_if(
        cotangent,
        jnp.any(~jnp.isfinite(cotangent))
        | has_subnormal_components(cotangent),
        "coefficient_cotangent must be finite and normal-or-zero",
    )
    nx, ny, nz = _grid_shape_xyz(checked_target)
    indices = checked_target.state_indices
    grid_shape_xyz = jnp.asarray((nx, ny, nz), dtype=jnp.int64)
    residues = jnp.mod(indices, grid_shape_xyz[None, :])
    factors = _local_cell_shape_factors(indices, grid_shape_xyz)
    origin_cycles = _origin_cycle_fractions(checked_target.local_potential)
    negative_phase = _negative_origin_phase(
        indices, origin_cycles, factors != 0.0
    )
    volume_sqrt = _rounded_box_volume_sqrt(checked_target)
    weights = (
        jnp.conj(volume_sqrt * factors * negative_phase) * checked_cotangent
    )
    embedded = (
        jnp.zeros((nz, ny, nx), dtype=jnp.complex128)
        .at[residues[:, 2], residues[:, 1], residues[:, 0]]
        .add(weights)
    )
    sample_count = nx * ny * nz
    scale_exponent, output_scale = _normalized_fft_scales(sample_count)
    adjoint_gain = _normalized_fft_adjoint_gain(
        sample_count, scale_exponent, output_scale
    )
    euclidean_adjoint = jnp.fft.ifftn(embedded) * adjoint_gain
    cell_volume = _checked_physical_cell_volume(checked_target)
    result = euclidean_adjoint / cell_volume
    source_cell_gradient: Complex128[Array, "nz ny nx"] = eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)) | has_subnormal_components(result),
        "local-source complex metric adjoint left finite normal range",
    )
    return source_cell_gradient


def _source_digest(
    target: GalerkinLocalCellTargetManifest,
    route: GalerkinLocalAdditionalSourceRoute,
    source_cell_values: Complex128[Array, " ..."],
) -> str:
    """PRIVATE: Digest source identity against operator identity only.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Prepared target whose target digest defines operator identity.
    route : GalerkinLocalAdditionalSourceRoute
        Disjoint ZERO or LOCAL_CELL route.
    source_cell_values : Complex128[Array, " ..."]
        Exact stored q-cell payload for this route.

    Returns
    -------
    source_digest : str
        Canonical source identity digest.
    """
    source_digest: str = sha256(
        {
            "domain": _SOURCE_DIGEST_DOMAIN,
            "route": route.value,
            "target_digest": target.target_digest,
            "source_cell_values": stored_value_payload(source_cell_values),
            "cell_value_units": _CELL_VALUE_UNITS,
            "cell_support_convention": _CELL_SUPPORT_CONVENTION,
            "coefficient_formula": _COEFFICIENT_FORMULA,
            "source_formula": _SOURCE_FORMULA,
        }
    )
    return source_digest


def _realization_digest(
    target: GalerkinLocalCellTargetManifest,
    source_digest: str,
    algebraic_additional_source: Complex128[Array, " n"],
    algebraic_volume_sqrt: Float64[Array, ""],
) -> str:
    """PRIVATE: Digest rounded realization and full parent evidence.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared parent target.
    source_digest : str
        Bound source identity digest.
    algebraic_additional_source : Complex128[Array, " n"]
        Rounded retained source vector.
    algebraic_volume_sqrt : Float64[Array, ""]
        Frozen rounded volume-square-root multiplier.

    Returns
    -------
    realization_digest : str
        Complete rounded-realization evidence digest.
    """
    realization_digest: str = sha256(
        {
            "domain": _REALIZATION_DIGEST_DOMAIN,
            "source_digest": source_digest,
            "target_digest": target.target_digest,
            "parent_target_evidence_digest": target.manifest_evidence_digest,
            "full_prepared_parent_target": stored_value_payload(target),
            "algebraic_additional_source": stored_value_payload(
                algebraic_additional_source
            ),
            "algebraic_volume_sqrt": stored_value_payload(
                algebraic_volume_sqrt
            ),
            "map_arithmetic": _MAP_ARITHMETIC,
            "adjoint_formula": _ADJOINT_FORMULA,
        }
    )
    return realization_digest


def _make_source(
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex128[Array, " ..."],
    algebraic_additional_source: Complex128[Array, " n"],
    algebraic_volume_sqrt: Float64[Array, ""],
    route: GalerkinLocalAdditionalSourceRoute,
) -> GalerkinLocalAdditionalSource:
    """PRIVATE: Bind a canonical prepared target to a rounded source map.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.
    source_cell_values : Complex128[Array, " ..."]
        Exact stored route-specific q-cell payload.
    algebraic_additional_source : Complex128[Array, " n"]
        Rounded retained additional-source vector.
    algebraic_volume_sqrt : Float64[Array, ""]
        Frozen rounded square-root-of-volume multiplier.
    route : GalerkinLocalAdditionalSourceRoute
        Disjoint ZERO or LOCAL_CELL route.

    Returns
    -------
    source : GalerkinLocalAdditionalSource
        Digest-bound canonical source realization.
    """
    source_digest = _source_digest(target, route, source_cell_values)
    realization_digest = _realization_digest(
        target,
        source_digest,
        algebraic_additional_source,
        algebraic_volume_sqrt,
    )
    source: GalerkinLocalAdditionalSource = _make_local_additional_source(
        target,
        source_cell_values,
        algebraic_additional_source,
        algebraic_volume_sqrt,
        route=route,
        cell_value_units=_CELL_VALUE_UNITS,
        cell_support_convention=_CELL_SUPPORT_CONVENTION,
        coefficient_formula=_COEFFICIENT_FORMULA,
        source_formula=_SOURCE_FORMULA,
        map_arithmetic=_MAP_ARITHMETIC,
        target_digest=target.target_digest,
        parent_target_evidence_digest=target.manifest_evidence_digest,
        source_digest=source_digest,
        realization_digest=realization_digest,
    )
    return source


def _realize_zero_prepared(
    target: GalerkinLocalCellTargetManifest,
) -> GalerkinLocalAdditionalSource:
    """PRIVATE: Create the symbolic ZERO realization from a prepared target.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.

    Returns
    -------
    source : GalerkinLocalAdditionalSource
        ZERO route with an empty q carrier and exact zero map.
    """
    empty = jnp.empty((0,), dtype=jnp.complex128)
    vector = jnp.zeros((target.state_indices.shape[0],), dtype=jnp.complex128)
    source: GalerkinLocalAdditionalSource = _make_source(
        target,
        empty,
        vector,
        jnp.asarray(0.0, dtype=jnp.float64),
        GalerkinLocalAdditionalSourceRoute.ZERO,
    )
    return source


def _realize_local_cell_prepared(
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex[Array, "..."],
) -> GalerkinLocalAdditionalSource:
    """PRIVATE: Create one complex realization from a prepared target.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell target.
    source_cell_values : Complex[Array, "..."]
        Complex128 q-cell values on exactly the target grid.

    Returns
    -------
    source : GalerkinLocalAdditionalSource
        Canonical LOCAL_CELL rounded source realization.
    """
    cells = _checked_source_cells(target, source_cell_values)
    vector = _prepared_local_cell_map(target, cells)
    volume_sqrt = _rounded_box_volume_sqrt(target)
    source: GalerkinLocalAdditionalSource = _make_source(
        target,
        cells,
        vector,
        volume_sqrt,
        GalerkinLocalAdditionalSourceRoute.LOCAL_CELL,
    )
    return source


def realize_zero_local_additional_source(
    target: GalerkinLocalCellTargetManifest,
) -> GalerkinLocalAdditionalSource:
    """Full-replay the target and build the empty-carrier ZERO route.

    :see: :func:`~.test_local_sources.\
test_zero_route_has_empty_q_and_symbolic_zero_certificate`

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Public local-cell target to authenticate in full.

    Returns
    -------
    source : GalerkinLocalAdditionalSource
        Authenticated ZERO source with no q-cell payload or work.

    Raises
    ------
    TypeError
        If ``target`` has the wrong carrier type.
    ValueError
        If the parent target fails full replay.
    """
    prepared = prepare_local_cell_galerkin_target(_checked_target(target))
    source: GalerkinLocalAdditionalSource = _realize_zero_prepared(prepared)
    return source


@jaxtyped(typechecker=beartype)
def realize_local_cell_additional_source(
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex[Array, "..."],
) -> GalerkinLocalAdditionalSource:
    """Full-replay the target and realize complex LVT.20a--LVT.20c.

    :see: :func:`~.test_local_sources.\
test_complex_map_matches_direct_sum_without_hermitian_projection`

    The q-cell input must be complex128 on exactly the target's local grid.
    No real-valued or Hermitian reduction is performed.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Public local-cell target to authenticate in full.
    source_cell_values : Complex[Array, "..."]
        Complex128 q values on exactly the target local grid.

    Returns
    -------
    source : GalerkinLocalAdditionalSource
        Authenticated LOCAL_CELL rounded source realization.

    Raises
    ------
    TypeError
        If ``target`` has the wrong carrier type.
    ValueError
        If q dtype or shape is invalid or parent replay fails.
    """
    checked_target = _checked_target(target)
    checked_cells = _checked_source_cells(checked_target, source_cell_values)
    prepared = prepare_local_cell_galerkin_target(checked_target)
    source: GalerkinLocalAdditionalSource = _realize_local_cell_prepared(
        prepared, checked_cells
    )
    return source


def _canonical_source(
    source: GalerkinLocalAdditionalSource,
) -> GalerkinLocalAdditionalSource:
    """PRIVATE: Full-replay and compare one rounded source carrier.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Public source carrier whose parent and rounded map must be replayed.

    Returns
    -------
    canonical : GalerkinLocalAdditionalSource
        Fresh canonical realization reconstructed from authenticated inputs.

    Raises
    ------
    TypeError
        If ``source`` has the wrong carrier type.
    ValueError
        If the route is invalid or complete replay differs from submission.
    """
    if not isinstance(source, GalerkinLocalAdditionalSource):
        raise TypeError("source must be GalerkinLocalAdditionalSource")
    _assert_concrete(source)
    prepared_target = prepare_local_cell_galerkin_target(source.target)
    if source.route is GalerkinLocalAdditionalSourceRoute.ZERO:
        canonical: GalerkinLocalAdditionalSource = _realize_zero_prepared(
            prepared_target
        )
    elif source.route is GalerkinLocalAdditionalSourceRoute.LOCAL_CELL:
        canonical: GalerkinLocalAdditionalSource = (
            _realize_local_cell_prepared(
                prepared_target, source.source_cell_values
            )
        )
    else:
        raise ValueError("local additional-source route is noncanonical")
    if stored_value_payload(canonical) != stored_value_payload(source):
        raise ValueError(
            "local additional source does not match full target/map replay"
        )
    return canonical


def _symbolic_shape_zero(
    mode: Tuple[int, int, int], shape_xyz: Tuple[int, int, int]
) -> bool:
    """PRIVATE: Detect one exact centered-cell symbolic sinc zero.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Unwrapped integer Fourier mode in xyz order.
    shape_xyz : Tuple[int, int, int]
        Positive cell counts in xyz order.

    Returns
    -------
    symbolic_zero : bool
        Whether any nonzero mode component is a multiple of its cell count.
    """
    symbolic_zero: bool = any(
        component != 0 and component % count == 0
        for component, count in zip(mode, shape_xyz, strict=True)
    )
    return symbolic_zero


def _shape_factor_rectangle(
    mode: Tuple[int, int, int], shape_xyz: Tuple[int, int, int]
) -> Tuple[Fraction, Fraction]:
    """PRIVATE: Enclose one exact separable centered-cell sinc factor.

    Parameters
    ----------
    mode : Tuple[int, int, int]
        Unwrapped integer Fourier mode in xyz order.
    shape_xyz : Tuple[int, int, int]
        Positive cell counts in xyz order.

    Returns
    -------
    lower : Fraction
        Rational lower endpoint of the separable shape factor.
    upper : Fraction
        Rational upper endpoint of the separable shape factor.
    """
    factor: Tuple[Fraction, Fraction] = (Fraction(1), Fraction(1))
    for component, count in zip(mode, shape_xyz, strict=True):
        factor = real_interval_product(
            factor, normalized_sinc_integer_ratio(component, count)
        )
    return factor


def _axis_phase_rectangles(
    mode: int,
    size: int,
    cache: Dict[Fraction, ComplexRectangle],
) -> Tuple[ComplexRectangle, ...]:
    """PRIVATE: Enclose every exact DFT phase on one cell axis.

    Parameters
    ----------
    mode : int
        Unwrapped integer Fourier-mode component.
    size : int
        Positive cell count along this axis.
    cache : Dict[Fraction, ComplexRectangle]
        Mutable cache of exact rational-turn exponential rectangles.

    Returns
    -------
    phase_rectangles : Tuple[ComplexRectangle, ...]
        Ordered phase rectangle for every cell position on the axis.
    """
    phases: list[ComplexRectangle] = []
    for position in range(size):
        turn = Fraction(mode * position, size) % 1
        if turn not in cache:
            cache[turn] = rational_turn_exponential(turn)
        phases.append(cache[turn])
    phase_rectangles: Tuple[ComplexRectangle, ...] = tuple(phases)
    return phase_rectangles


def _exact_source_rectangle(  # noqa: PLR0913
    source_cells: Complex128[NDArray, "nz ny nx"],
    mode: Tuple[int, int, int],
    origin_xyz: Tuple[float, float, float],
    box_xyz: Tuple[float, float, float],
    volume_sqrt: Tuple[Fraction, Fraction],
    phase_cache: Dict[Fraction, ComplexRectangle],
) -> ComplexRectangle:
    """PRIVATE: Enclose one complete exact LVT.20c coefficient.

    Parameters
    ----------
    source_cells : Complex128[NDArray, "nz ny nx"]
        Exact stored complex128 q values in zyx order.
    mode : Tuple[int, int, int]
        Unwrapped integer Fourier mode in xyz order.
    origin_xyz : Tuple[float, float, float]
        Stored physical center of cell zero in xyz order.
    box_xyz : Tuple[float, float, float]
        Stored target box lengths in xyz order.
    volume_sqrt : Tuple[Fraction, Fraction]
        Rational enclosure of the exact square root of box volume.
    phase_cache : Dict[Fraction, ComplexRectangle]
        Mutable rational-turn phase-enclosure cache.

    Returns
    -------
    rectangle : ComplexRectangle
        Outward rational rectangle for exact ``sqrt(|Omega|) c_q(mode)``.
    """
    nz, ny, nx = source_cells.shape
    shape_xyz: Tuple[int, int, int] = (nx, ny, nz)
    if _symbolic_shape_zero(mode, shape_xyz):
        rectangle: ComplexRectangle = _ZERO_RECTANGLE
        return rectangle
    shape_factor = _shape_factor_rectangle(mode, shape_xyz)
    mode_x, mode_y, mode_z = mode
    x_phases = _axis_phase_rectangles(mode_x, nx, phase_cache)
    y_phases = _axis_phase_rectangles(mode_y, ny, phase_cache)
    z_phases = _axis_phase_rectangles(mode_z, nz, phase_cache)

    def direct_terms() -> Iterator[ComplexRectangle]:
        """Yield exact complex-cell times DFT-phase rectangles."""
        for z_position in range(nz):
            for y_position in range(ny):
                yz_phase = complex_rectangle_multiply(
                    z_phases[z_position], y_phases[y_position]
                )
                for x_position in range(nx):
                    phase = complex_rectangle_multiply(
                        yz_phase, x_phases[x_position]
                    )
                    cell = source_cells[z_position, y_position, x_position]
                    cell_rectangle: ComplexRectangle = (
                        fraction_from_float(float(np.real(cell))),
                        fraction_from_float(float(np.real(cell))),
                        fraction_from_float(float(np.imag(cell))),
                        fraction_from_float(float(np.imag(cell))),
                    )
                    yield complex_rectangle_multiply(cell_rectangle, phase)

    mean_dft = scale_complex_rectangle(
        pairwise_rectangle_sum(direct_terms()), Fraction(1, source_cells.size)
    )
    shaped = complex_rectangle_multiply(
        mean_dft,
        (shape_factor[0], shape_factor[1], Fraction(0), Fraction(0)),
    )
    origin_turn = sum(
        (
            component
            * fraction_from_float(origin)
            / fraction_from_float(length)
            for component, origin, length in zip(
                mode, origin_xyz, box_xyz, strict=True
            )
        ),
        start=Fraction(0),
    )
    reduced_origin_turn = origin_turn % 1
    if reduced_origin_turn not in phase_cache:
        phase_cache[reduced_origin_turn] = rational_turn_exponential(
            reduced_origin_turn
        )
    coefficient = complex_rectangle_multiply(
        shaped, phase_cache[reduced_origin_turn]
    )
    rectangle: ComplexRectangle = complex_rectangle_multiply(
        coefficient,
        (volume_sqrt[0], volume_sqrt[1], Fraction(0), Fraction(0)),
    )
    return rectangle


def _coefficient_error_fraction(
    coefficient: np.complex128, rectangle: ComplexRectangle
) -> Fraction:
    """PRIVATE: Bound one stored complex point against an exact rectangle.

    Parameters
    ----------
    coefficient : np.complex128
        Stored rounded complex coefficient.
    rectangle : ComplexRectangle
        Exact-source enclosure rectangle.

    Returns
    -------
    error_bound : Fraction
        Rational upper bound on complex Euclidean component error.
    """
    real = fraction_from_float(float(np.real(coefficient)))
    imaginary = fraction_from_float(float(np.imag(coefficient)))
    real_gap = max(abs(real - rectangle[0]), abs(real - rectangle[1]))
    imag_gap = max(
        abs(imaginary - rectangle[2]), abs(imaginary - rectangle[3])
    )
    error_bound: Fraction = sqrt_fraction_upper(
        real_gap * real_gap + imag_gap * imag_gap
    )
    return error_bound


def _certificate_digest(  # noqa: PLR0913
    source: GalerkinLocalAdditionalSource,
    real_lower: Float64[NDArray, " n"],
    real_upper: Float64[NDArray, " n"],
    imag_lower: Float64[NDArray, " n"],
    imag_upper: Float64[NDArray, " n"],
    errors: Float64[NDArray, " n"],
    norm_error: Float64[NDArray, ""],
    finite: Bool[NDArray, ""],
    term_count: Int64[NDArray, ""],
    term_budget: Int64[NDArray, ""],
    failure: GalerkinLocalAdditionalSourceCertificateFailure,
) -> str:
    """PRIVATE: Bind direct evidence and the full prepared parent target.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Canonical source realization.
    real_lower : Float64[NDArray, " n"]
        Exact-source real lower endpoints.
    real_upper : Float64[NDArray, " n"]
        Exact-source real upper endpoints.
    imag_lower : Float64[NDArray, " n"]
        Exact-source imaginary lower endpoints.
    imag_upper : Float64[NDArray, " n"]
        Exact-source imaginary upper endpoints.
    errors : Float64[NDArray, " n"]
        Outward component error bounds.
    norm_error : Float64[NDArray, ""]
        Outward Euclidean source-error bound.
    finite : Bool[NDArray, ""]
        Whether this is a finite success certificate.
    term_count : Int64[NDArray, ""]
        Direct cell-product work count.
    term_budget : Int64[NDArray, ""]
        Certified direct-work budget.
    failure : GalerkinLocalAdditionalSourceCertificateFailure
        Typed success or noncertificate outcome.

    Returns
    -------
    certificate_digest : str
        Complete direct-certificate evidence digest.
    """
    certificate_digest: str = sha256(
        {
            "domain": _CERTIFICATE_DIGEST_DOMAIN,
            "source_digest": source.source_digest,
            "realization_digest": source.realization_digest,
            "target_digest": source.target_digest,
            "parent_target_evidence_digest": (
                source.parent_target_evidence_digest
            ),
            "full_prepared_parent_target": stored_value_payload(source.target),
            "exact_target": _EXACT_TARGET,
            "arithmetic": _ARITHMETIC,
            "direct_term_count_route": _TERM_COUNT_ROUTE,
            "error_scope": _ERROR_SCOPE,
            "coefficient_norm": _COEFFICIENT_NORM,
            "failure": failure.value,
            "finite_certificate": stored_value_payload(finite),
            "direct_term_count": stored_value_payload(term_count),
            "maximum_direct_terms": stored_value_payload(term_budget),
            "real_lower": stored_value_payload(real_lower),
            "real_upper": stored_value_payload(real_upper),
            "imag_lower": stored_value_payload(imag_lower),
            "imag_upper": stored_value_payload(imag_upper),
            "component_errors": stored_value_payload(errors),
            "source_norm_error": stored_value_payload(norm_error),
        }
    )
    return certificate_digest


def _make_certificate(  # noqa: PLR0913
    source: GalerkinLocalAdditionalSource,
    real_lower: Float64[NDArray, " n"],
    real_upper: Float64[NDArray, " n"],
    imag_lower: Float64[NDArray, " n"],
    imag_upper: Float64[NDArray, " n"],
    errors: Float64[NDArray, " n"],
    norm_error: Float64[NDArray, ""],
    term_count: int,
    term_budget: int,
    failure: GalerkinLocalAdditionalSourceCertificateFailure,
) -> GalerkinLocalAdditionalSourceCertificate:
    """PRIVATE: Create one digest-bound success or typed noncertificate.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Canonical source realization.
    real_lower : Float64[NDArray, " n"]
        Exact-source real lower endpoints.
    real_upper : Float64[NDArray, " n"]
        Exact-source real upper endpoints.
    imag_lower : Float64[NDArray, " n"]
        Exact-source imaginary lower endpoints.
    imag_upper : Float64[NDArray, " n"]
        Exact-source imaginary upper endpoints.
    errors : Float64[NDArray, " n"]
        Outward component error bounds.
    norm_error : Float64[NDArray, ""]
        Outward Euclidean source-error bound.
    term_count : int
        Direct cell-product work count.
    term_budget : int
        Certified direct-work budget.
    failure : GalerkinLocalAdditionalSourceCertificateFailure
        Typed success or noncertificate outcome.

    Returns
    -------
    certificate : GalerkinLocalAdditionalSourceCertificate
        Digest-bound finite certificate or typed noncertificate.
    """
    finite = np.asarray(
        failure is GalerkinLocalAdditionalSourceCertificateFailure.NONE,
        dtype=np.bool_,
    )
    count = np.asarray(term_count, dtype=np.int64)
    budget = np.asarray(term_budget, dtype=np.int64)
    digest = _certificate_digest(
        source,
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        errors,
        norm_error,
        finite,
        count,
        budget,
        failure,
    )
    certificate: GalerkinLocalAdditionalSourceCertificate = (
        _make_local_additional_source_certificate(
            source,
            jax.lax.stop_gradient(jnp.asarray(real_lower, dtype=jnp.float64)),
            jax.lax.stop_gradient(jnp.asarray(real_upper, dtype=jnp.float64)),
            jax.lax.stop_gradient(jnp.asarray(imag_lower, dtype=jnp.float64)),
            jax.lax.stop_gradient(jnp.asarray(imag_upper, dtype=jnp.float64)),
            jax.lax.stop_gradient(jnp.asarray(errors, dtype=jnp.float64)),
            jax.lax.stop_gradient(jnp.asarray(norm_error, dtype=jnp.float64)),
            jnp.asarray(finite),
            jnp.asarray(count),
            jnp.asarray(budget),
            failure=failure,
            exact_target=_EXACT_TARGET,
            arithmetic=_ARITHMETIC,
            direct_term_count_route=_TERM_COUNT_ROUTE,
            error_scope=_ERROR_SCOPE,
            coefficient_norm=_COEFFICIENT_NORM,
            parent_source_digest=source.source_digest,
            parent_target_evidence_digest=source.parent_target_evidence_digest,
            certificate_digest=digest,
        )
    )
    return certificate


def _failure_certificate(
    source: GalerkinLocalAdditionalSource,
    term_count: int,
    term_budget: int,
    failure: GalerkinLocalAdditionalSourceCertificateFailure,
) -> GalerkinLocalAdditionalSourceCertificate:
    """PRIVATE: Create an all-infinite typed direct noncertificate.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Canonical source realization.
    term_count : int
        Direct cell-product count associated with the failure.
    term_budget : int
        Certified direct-work budget.
    failure : GalerkinLocalAdditionalSourceCertificateFailure
        Typed non-success outcome.

    Returns
    -------
    certificate : GalerkinLocalAdditionalSourceCertificate
        All-infinite typed direct noncertificate.
    """
    size = source.algebraic_additional_source.shape[0]
    lower = np.full((size,), -np.inf, dtype=np.float64)
    upper = np.full((size,), np.inf, dtype=np.float64)
    errors = np.full((size,), np.inf, dtype=np.float64)
    norm_error = np.asarray(np.inf, dtype=np.float64)
    certificate: GalerkinLocalAdditionalSourceCertificate = _make_certificate(
        source,
        lower,
        upper,
        lower.copy(),
        upper.copy(),
        errors,
        norm_error,
        term_count,
        term_budget,
        failure,
    )
    return certificate


def _certify_canonical_source(
    source: GalerkinLocalAdditionalSource,
    maximum_direct_terms: int,
) -> GalerkinLocalAdditionalSourceCertificate:
    """PRIVATE: Derive direct evidence from one canonical replayed source.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Canonical fully replayed source realization.
    maximum_direct_terms : int
        Positive signed-64-bit direct-work budget.

    Returns
    -------
    certificate : GalerkinLocalAdditionalSourceCertificate
        Finite direct certificate or typed noncertificate.

    Raises
    ------
    ValueError
        If retained support or direct-work count violates exact host storage
        contracts, or if a LOCAL_CELL q payload lacks complex128 dtype.
    """
    size = source.algebraic_additional_source.shape[0]
    if source.route is GalerkinLocalAdditionalSourceRoute.ZERO:
        zeros = np.zeros((size,), dtype=np.float64)
        certificate: GalerkinLocalAdditionalSourceCertificate = (
            _make_certificate(
                source,
                zeros,
                zeros.copy(),
                zeros.copy(),
                zeros.copy(),
                zeros.copy(),
                np.asarray(0.0, dtype=np.float64),
                0,
                maximum_direct_terms,
                GalerkinLocalAdditionalSourceCertificateFailure.NONE,
            )
        )
        return certificate

    target = source.target
    nx, ny, nz = _grid_shape_xyz(target)
    shape_xyz = (nx, ny, nz)
    indices = np.asarray(jax.device_get(target.state_indices))
    if indices.dtype != np.dtype(np.int64) or indices.shape != (size, 3):
        raise ValueError("target I_u must be one exact int64 (n, 3) array")
    modes: list[Tuple[int, int, int]] = [
        (int(row[0]), int(row[1]), int(row[2])) for row in indices
    ]
    term_count = source.source_cell_values.size * sum(
        not _symbolic_shape_zero(mode, shape_xyz) for mode in modes
    )
    if term_count > _MAXIMUM_DIRECT_TERMS:
        raise ValueError("direct_term_count must fit in signed 64-bit storage")
    if term_count > maximum_direct_terms:
        failed = _failure_certificate(
            source,
            term_count,
            maximum_direct_terms,
            GalerkinLocalAdditionalSourceCertificateFailure.WORK_BUDGET_EXCEEDED,
        )
        certificate: GalerkinLocalAdditionalSourceCertificate = failed
        return certificate
    if not host_binary64_supported():
        failed = _failure_certificate(
            source,
            term_count,
            maximum_direct_terms,
            GalerkinLocalAdditionalSourceCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED,
        )
        certificate: GalerkinLocalAdditionalSourceCertificate = failed
        return certificate
    source_cells = np.asarray(jax.device_get(source.source_cell_values))
    if source_cells.dtype != np.dtype(np.complex128):
        raise ValueError(
            "LOCAL_CELL source_cell_values must have complex128 dtype"
        )
    volume = _box_volume_fraction(target)
    phase_cache: Dict[Fraction, ComplexRectangle] = {}
    try:
        volume_upper = sqrt_fraction_upper(volume)
        volume_interval = (volume / volume_upper, volume_upper)
        rectangles = [
            _exact_source_rectangle(
                source_cells,
                mode,
                target.local_potential.cell_center_origin,
                target.local_potential.box_size,
                volume_interval,
                phase_cache,
            )
            for mode in modes
        ]
        algebraic = np.asarray(
            jax.device_get(source.algebraic_additional_source),
            dtype=np.complex128,
        )
        error_fractions = [
            _coefficient_error_fraction(value, rectangle)
            for value, rectangle in zip(algebraic, rectangles, strict=True)
        ]
        real_lower = np.asarray(
            [fraction_lower_float(value[0]) for value in rectangles],
            dtype=np.float64,
        )
        real_upper = np.asarray(
            [fraction_upper_float(value[1]) for value in rectangles],
            dtype=np.float64,
        )
        imag_lower = np.asarray(
            [fraction_lower_float(value[2]) for value in rectangles],
            dtype=np.float64,
        )
        imag_upper = np.asarray(
            [fraction_upper_float(value[3]) for value in rectangles],
            dtype=np.float64,
        )
        errors = np.asarray(
            [fraction_upper_float(value) for value in error_fractions],
            dtype=np.float64,
        )
        norm_fraction = sqrt_fraction_upper(
            sum(
                (
                    fraction_from_float(float(error))
                    * fraction_from_float(float(error))
                    for error in errors
                ),
                start=Fraction(0),
            )
        )
        norm_error = np.asarray(
            fraction_upper_float(norm_fraction), dtype=np.float64
        )
    except RootEnclosureError:
        failed = _failure_certificate(
            source,
            term_count,
            maximum_direct_terms,
            GalerkinLocalAdditionalSourceCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        )
        certificate: GalerkinLocalAdditionalSourceCertificate = failed
        return certificate
    arrays = (
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        errors,
        norm_error,
    )
    tiny = np.finfo(np.float64).tiny
    outside_normal_range = any(
        not np.all(np.isfinite(value)) for value in arrays
    ) or any(
        np.any((value != 0.0) & (np.abs(value) < tiny)) for value in arrays
    )
    if outside_normal_range:
        failed = _failure_certificate(
            source,
            term_count,
            maximum_direct_terms,
            GalerkinLocalAdditionalSourceCertificateFailure.ARITHMETIC_RANGE_FAILURE,
        )
        certificate: GalerkinLocalAdditionalSourceCertificate = failed
        return certificate
    certificate: GalerkinLocalAdditionalSourceCertificate = _make_certificate(
        source,
        real_lower,
        real_upper,
        imag_lower,
        imag_upper,
        errors,
        norm_error,
        term_count,
        maximum_direct_terms,
        GalerkinLocalAdditionalSourceCertificateFailure.NONE,
    )
    return certificate


def certify_local_additional_source(
    source: GalerkinLocalAdditionalSource,
    *,
    maximum_direct_terms: int = _DEFAULT_MAXIMUM_DIRECT_TERMS,
) -> GalerkinLocalAdditionalSourceCertificate:
    """Full-replay the parent/map and directly certify exact LVT.20c.

    :see: :func:`~.test_local_sources.\
test_direct_rectangles_enclose_sqrt_volume_and_lvt20e`

    Budget, host-capability, transcendental-root, and arithmetic-range
    failures return typed noncertificates.  Structural, dtype, parent, and
    replay mismatches raise instead of being converted into evidence.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Public rounded source realization to authenticate in full.
    maximum_direct_terms : int, optional
        Positive signed-64-bit direct-work budget. By default, 2,000,000.

    Returns
    -------
    certificate : GalerkinLocalAdditionalSourceCertificate
        Direct finite certificate or typed noncertificate.

    Raises
    ------
    TypeError
        If ``source`` has the wrong carrier type.
    ValueError
        If the budget is invalid or complete replay detects forgery.
    """
    if (
        isinstance(maximum_direct_terms, bool)
        or not isinstance(maximum_direct_terms, int)
        or maximum_direct_terms <= 0
        or maximum_direct_terms > _MAXIMUM_DIRECT_TERMS
    ):
        raise ValueError(
            "maximum_direct_terms must be a positive signed-64-bit integer"
        )
    canonical = _canonical_source(source)
    certificate: GalerkinLocalAdditionalSourceCertificate = (
        _certify_canonical_source(canonical, maximum_direct_terms)
    )
    return certificate


def prepare_local_additional_source_certificate(
    certificate: GalerkinLocalAdditionalSourceCertificate,
) -> GalerkinLocalAdditionalSourceCertificate:
    """Full-replay target, map, q rectangles, budget, and all digests.

    :see: :func:`~.test_local_sources.\
test_replay_rejects_parent_map_rectangle_and_digest_forgery`

    Parameters
    ----------
    certificate : GalerkinLocalAdditionalSourceCertificate
        Public certificate or typed noncertificate to replay in full.

    Returns
    -------
    canonical : GalerkinLocalAdditionalSourceCertificate
        Fresh canonical result reconstructed from source q bytes and parent.

    Raises
    ------
    TypeError
        If ``certificate`` has the wrong carrier type.
    ValueError
        If scalar storage or complete host replay differs from submission.
    """
    if not isinstance(certificate, GalerkinLocalAdditionalSourceCertificate):
        raise TypeError(
            "certificate must be GalerkinLocalAdditionalSourceCertificate"
        )
    _assert_concrete(certificate)
    budget_array = np.asarray(jax.device_get(certificate.maximum_direct_terms))
    if budget_array.dtype != np.dtype(np.int64) or budget_array.shape != ():
        raise ValueError("maximum_direct_terms must be an exact int64 scalar")
    canonical_source = _canonical_source(certificate.source)
    canonical: GalerkinLocalAdditionalSourceCertificate = (
        _certify_canonical_source(canonical_source, int(budget_array))
    )
    if stored_value_payload(canonical) != stored_value_payload(certificate):
        raise ValueError(
            "local-source certificate does not match complete host replay"
        )
    return canonical


__all__: list[str] = [
    "apply_local_cell_additional_source_map",
    "apply_local_cell_additional_source_metric_adjoint",
    "certify_local_additional_source",
    "prepare_local_additional_source_certificate",
    "realize_local_cell_additional_source",
    "realize_zero_local_additional_source",
]
