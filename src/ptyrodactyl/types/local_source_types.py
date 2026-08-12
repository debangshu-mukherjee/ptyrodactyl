r"""Define disjoint LVT.20 local additional-source carriers.

Extended Summary
----------------
The ``ZERO`` route owns an empty q-cell payload and an exact zero retained
vector.  The ``LOCAL_CELL`` route alone owns complex cell values on the
nested target grid.  Direct host certification encloses the complete
``sqrt(|Omega|) c_q`` expression; it does not impose Hermitian symmetry and
does not reuse a coefficient-only represented-source branch.

Routine Listings
----------------
:class:`GalerkinLocalAdditionalSource`
    Store one rounded ZERO or complex LOCAL_CELL LVT.20a--LVT.20c map.
:class:`GalerkinLocalAdditionalSourceCertificate`
    Store direct rectangles and the LVT.20e source-norm transfer.
:class:`GalerkinLocalAdditionalSourceCertificateFailure`
    Store one typed direct LVT.20c certificate outcome.
:class:`GalerkinLocalAdditionalSourceRoute`
    Select one exact LVT.20a additional-source carrier.
"""

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import has_subnormal_components

from .local_cell_target_types import GalerkinLocalCellTargetManifest
from .local_cell_types import LocalCellPotential3D

_SHA256_HEX_LENGTH: int = 64


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise for one structural carrier failure.

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


class GalerkinLocalAdditionalSourceRoute(str, Enum):
    """Select one exact LVT.20a additional-source carrier.

    :see: :func:`~.test_local_source_types.\
test_local_source_routes_and_failures_are_typed_and_disjoint`
    """

    ZERO = "zero"
    LOCAL_CELL = "local_cell"


class GalerkinLocalAdditionalSourceCertificateFailure(str, Enum):
    """Store one typed direct LVT.20c certificate outcome.

    :see: :func:`~.test_local_source_types.\
test_local_source_routes_and_failures_are_typed_and_disjoint`
    """

    NONE = "none"
    HOST_ARITHMETIC_UNSUPPORTED = "host_arithmetic_unsupported"
    WORK_BUDGET_EXCEEDED = "work_budget_exceeded"
    ROOT_ENCLOSURE_FAILURE = "root_enclosure_failure"
    ARITHMETIC_RANGE_FAILURE = "arithmetic_range_failure"


class GalerkinLocalAdditionalSource(eqx.Module):
    r"""Store one rounded ZERO or complex LOCAL_CELL LVT.20a--LVT.20c map.

    :see: :func:`~.test_local_source_types.\
test_local_source_carriers_own_only_lvt20c_evidence`

    ``source_cell_values`` has shape ``(0,)`` on ``ZERO``.  On
    ``LOCAL_CELL`` it has the target potential's storage shape ``(nz, ny,
    nx)`` and units of envelope field per squared length.  Its retained
    coefficient vector has the corresponding source-field units times the
    square root of volume.

    ``algebraic_volume_sqrt`` is the intentionally frozen binary64
    approximation used by the rounded map.  It is not relabelled as the
    exact square root; the direct child certificate compares the complete
    stored vector against an enclosure of the exact LVT.20c expression.

    The nested target is public forgeable storage.  Constructors in
    :mod:`ptyrodactyl.galerkin.local_sources` replay it in full before
    creating this carrier.
    """

    target: GalerkinLocalCellTargetManifest
    source_cell_values: Complex128[Array, " ..."]
    algebraic_additional_source: Complex128[Array, " n"]
    algebraic_volume_sqrt: Float64[Array, ""]
    route: GalerkinLocalAdditionalSourceRoute = eqx.field(static=True)
    cell_value_units: str = eqx.field(static=True)
    cell_support_convention: str = eqx.field(static=True)
    coefficient_formula: str = eqx.field(static=True)
    source_formula: str = eqx.field(static=True)
    map_arithmetic: str = eqx.field(static=True)
    target_digest: str = eqx.field(static=True)
    parent_target_evidence_digest: str = eqx.field(static=True)
    source_digest: str = eqx.field(static=True)
    realization_digest: str = eqx.field(static=True)

    @property
    def state_indices(self) -> Array:
        """Return the exact ordered target state support ``I_u``.

        Returns
        -------
        state_indices : Array
            Ordered target state indices.
        """
        state_indices: Array = self.target.state_indices
        return state_indices

    @property
    def local_potential(self) -> LocalCellPotential3D:
        """Return the target local potential that solely owns cell geometry.

        Returns
        -------
        local_potential : LocalCellPotential3D
            Target-owned local potential and cell geometry.
        """
        local_potential: LocalCellPotential3D = self.target.local_potential
        return local_potential


class GalerkinLocalAdditionalSourceCertificate(eqx.Module):
    r"""Store direct rectangles and the LVT.20e source-norm transfer.

    :see: :func:`~.test_local_source_types.\
test_local_source_carriers_own_only_lvt20c_evidence`

    The rectangles enclose the complete exact orthonormal coefficient
    ``sqrt(|Omega|) c_q(m)``.  Consequently each component error already
    includes the coefficient, square-root-of-volume, multiplication, and
    stored-vector discrepancy exactly once.
    """

    source: GalerkinLocalAdditionalSource
    exact_source_real_lower_bounds: Float64[Array, " n"]
    exact_source_real_upper_bounds: Float64[Array, " n"]
    exact_source_imag_lower_bounds: Float64[Array, " n"]
    exact_source_imag_upper_bounds: Float64[Array, " n"]
    component_error_bounds: Float64[Array, " n"]
    additional_source_error_upper_bound: Float64[Array, ""]
    finite_certificate: Bool[Array, ""]
    direct_term_count: Int64[Array, ""]
    maximum_direct_terms: Int64[Array, ""]
    failure: GalerkinLocalAdditionalSourceCertificateFailure = eqx.field(
        static=True
    )
    exact_target: str = eqx.field(static=True)
    arithmetic: str = eqx.field(static=True)
    direct_term_count_route: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)
    parent_source_digest: str = eqx.field(static=True)
    parent_target_evidence_digest: str = eqx.field(static=True)
    certificate_digest: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def _make_local_additional_source(  # noqa: PLR0913
    target: GalerkinLocalCellTargetManifest,
    source_cell_values: Complex[Array, "..."],
    algebraic_additional_source: Complex[Array, "..."],
    algebraic_volume_sqrt: Float[Array, ""],
    *,
    route: GalerkinLocalAdditionalSourceRoute,
    cell_value_units: str,
    cell_support_convention: str,
    coefficient_formula: str,
    source_formula: str,
    map_arithmetic: str,
    target_digest: str,
    parent_target_evidence_digest: str,
    source_digest: str,
    realization_digest: str,
) -> GalerkinLocalAdditionalSource:
    """PRIVATE: Store one validated rounded source realization.

    Parameters
    ----------
    target : GalerkinLocalCellTargetManifest
        Fully prepared local-cell Galerkin target.
    source_cell_values : Complex[Array, "..."]
        Empty ZERO payload or complex LOCAL_CELL q values.
    algebraic_additional_source : Complex[Array, "..."]
        Rounded retained additional-source vector.
    algebraic_volume_sqrt : Float[Array, ""]
        Frozen rounded square-root-of-volume multiplier.
    route : GalerkinLocalAdditionalSourceRoute
        Disjoint ZERO or LOCAL_CELL source route.
    cell_value_units : str
        Declared units of the q-cell values.
    cell_support_convention : str
        Declared periodic cell-support convention.
    coefficient_formula : str
        Declared coefficient-map formula.
    source_formula : str
        Declared retained-source formula.
    map_arithmetic : str
        Declared rounded map arithmetic.
    target_digest : str
        Bound parent target identity digest.
    parent_target_evidence_digest : str
        Bound full parent-target evidence digest.
    source_digest : str
        Source identity digest.
    realization_digest : str
        Rounded realization evidence digest.

    Returns
    -------
    source : GalerkinLocalAdditionalSource
        Validated canonical source carrier.

    Raises
    ------
    TypeError
        If ``target`` or ``route`` has the wrong carrier type.
    """
    if not isinstance(target, GalerkinLocalCellTargetManifest):
        raise TypeError("target must be GalerkinLocalCellTargetManifest")
    if not isinstance(route, GalerkinLocalAdditionalSourceRoute):
        raise TypeError("route must be GalerkinLocalAdditionalSourceRoute")
    cells = jnp.asarray(source_cell_values, dtype=jnp.complex128)
    vector = jnp.asarray(algebraic_additional_source, dtype=jnp.complex128)
    volume_sqrt = jnp.asarray(algebraic_volume_sqrt, dtype=jnp.float64)
    _raise_if(vector.ndim != 1, "algebraic_additional_source must be 1D")
    _raise_if(
        vector.shape != (target.state_indices.shape[0],),
        "algebraic_additional_source must match target I_u",
    )
    _raise_if(volume_sqrt.shape != (), "algebraic_volume_sqrt must be scalar")
    if route is GalerkinLocalAdditionalSourceRoute.ZERO:
        _raise_if(
            cells.shape != (0,),
            "ZERO source_cell_values must be the exact empty carrier",
        )
        invalid_route_values = (volume_sqrt != 0.0) | jnp.any(vector != 0.0)
    else:
        _raise_if(
            cells.shape != target.local_potential.cell_values.shape,
            "LOCAL_CELL source_cell_values must match the target local grid",
        )
        invalid_route_values = volume_sqrt <= 0.0
    for text, name in (
        (cell_value_units, "cell_value_units"),
        (cell_support_convention, "cell_support_convention"),
        (coefficient_formula, "coefficient_formula"),
        (source_formula, "source_formula"),
        (map_arithmetic, "map_arithmetic"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (target_digest, "target_digest"),
        (parent_target_evidence_digest, "parent_target_evidence_digest"),
        (source_digest, "source_digest"),
        (realization_digest, "realization_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    _raise_if(
        target_digest != target.target_digest,
        "target_digest must bind the nested target",
    )
    _raise_if(
        parent_target_evidence_digest != target.manifest_evidence_digest,
        "parent target evidence digest must bind the nested target",
    )
    invalid = (
        jnp.any(~jnp.isfinite(cells))
        | jnp.any(~jnp.isfinite(vector))
        | (~jnp.isfinite(volume_sqrt))
        | has_subnormal_components(cells)
        | has_subnormal_components(vector)
        | has_subnormal_components(volume_sqrt)
        | invalid_route_values
    )
    checked_cells = eqx.error_if(
        cells,
        invalid,
        "local additional-source realization must be finite and canonical",
    )
    source: GalerkinLocalAdditionalSource = GalerkinLocalAdditionalSource(
        target=target,
        source_cell_values=checked_cells,
        algebraic_additional_source=vector,
        algebraic_volume_sqrt=volume_sqrt,
        route=route,
        cell_value_units=cell_value_units.strip(),
        cell_support_convention=cell_support_convention.strip(),
        coefficient_formula=coefficient_formula.strip(),
        source_formula=source_formula.strip(),
        map_arithmetic=map_arithmetic.strip(),
        target_digest=target_digest,
        parent_target_evidence_digest=parent_target_evidence_digest,
        source_digest=source_digest,
        realization_digest=realization_digest,
    )
    return source


@jaxtyped(typechecker=beartype)
def _make_local_additional_source_certificate(  # noqa: PLR0913
    source: GalerkinLocalAdditionalSource,
    exact_source_real_lower_bounds: Float[Array, "..."],
    exact_source_real_upper_bounds: Float[Array, "..."],
    exact_source_imag_lower_bounds: Float[Array, "..."],
    exact_source_imag_upper_bounds: Float[Array, "..."],
    component_error_bounds: Float[Array, "..."],
    additional_source_error_upper_bound: Float[Array, ""],
    finite_certificate: Bool[Array, ""],
    direct_term_count: Int[Array, ""],
    maximum_direct_terms: Int[Array, ""],
    *,
    failure: GalerkinLocalAdditionalSourceCertificateFailure,
    exact_target: str,
    arithmetic: str,
    direct_term_count_route: str,
    error_scope: str,
    coefficient_norm: str,
    parent_source_digest: str,
    parent_target_evidence_digest: str,
    certificate_digest: str,
) -> GalerkinLocalAdditionalSourceCertificate:
    """PRIVATE: Store one validated direct LVT.20c certificate outcome.

    Parameters
    ----------
    source : GalerkinLocalAdditionalSource
        Canonical rounded source realization.
    exact_source_real_lower_bounds : Float[Array, "..."]
        Outward lower bounds on exact real components.
    exact_source_real_upper_bounds : Float[Array, "..."]
        Outward upper bounds on exact real components.
    exact_source_imag_lower_bounds : Float[Array, "..."]
        Outward lower bounds on exact imaginary components.
    exact_source_imag_upper_bounds : Float[Array, "..."]
        Outward upper bounds on exact imaginary components.
    component_error_bounds : Float[Array, "..."]
        Outward per-component complex error bounds.
    additional_source_error_upper_bound : Float[Array, ""]
        Outward Euclidean source-error bound.
    finite_certificate : Bool[Array, ""]
        Whether this outcome is a finite success certificate.
    direct_term_count : Int[Array, ""]
        Direct complex cell-product count.
    maximum_direct_terms : Int[Array, ""]
        Certified direct-work budget.
    failure : GalerkinLocalAdditionalSourceCertificateFailure
        Typed success or noncertificate outcome.
    exact_target : str
        Declared exact mathematical target.
    arithmetic : str
        Declared direct enclosure arithmetic.
    direct_term_count_route : str
        Declared work-count convention.
    error_scope : str
        Declared scope of the certified error.
    coefficient_norm : str
        Declared retained coefficient norm.
    parent_source_digest : str
        Bound source identity digest.
    parent_target_evidence_digest : str
        Bound full parent-target evidence digest.
    certificate_digest : str
        Complete certificate evidence digest.

    Returns
    -------
    certificate : GalerkinLocalAdditionalSourceCertificate
        Validated success certificate or typed noncertificate.

    Raises
    ------
    TypeError
        If ``source`` or ``failure`` has the wrong carrier type.
    """
    if not isinstance(source, GalerkinLocalAdditionalSource):
        raise TypeError("source must be GalerkinLocalAdditionalSource")
    if not isinstance(
        failure, GalerkinLocalAdditionalSourceCertificateFailure
    ):
        raise TypeError(
            "failure must be GalerkinLocalAdditionalSourceCertificateFailure"
        )
    real_lower = jnp.asarray(exact_source_real_lower_bounds, dtype=jnp.float64)
    real_upper = jnp.asarray(exact_source_real_upper_bounds, dtype=jnp.float64)
    imag_lower = jnp.asarray(exact_source_imag_lower_bounds, dtype=jnp.float64)
    imag_upper = jnp.asarray(exact_source_imag_upper_bounds, dtype=jnp.float64)
    errors = jnp.asarray(component_error_bounds, dtype=jnp.float64)
    norm_error = jnp.asarray(
        additional_source_error_upper_bound, dtype=jnp.float64
    )
    finite = jnp.asarray(finite_certificate, dtype=jnp.bool_)
    term_count = jnp.asarray(direct_term_count, dtype=jnp.int64)
    term_budget = jnp.asarray(maximum_direct_terms, dtype=jnp.int64)
    expected_shape = source.algebraic_additional_source.shape
    for values, name in (
        (real_lower, "exact_source_real_lower_bounds"),
        (real_upper, "exact_source_real_upper_bounds"),
        (imag_lower, "exact_source_imag_lower_bounds"),
        (imag_upper, "exact_source_imag_upper_bounds"),
        (errors, "component_error_bounds"),
    ):
        _raise_if(
            values.shape != expected_shape, f"{name} must match target I_u"
        )
    for value, name in (
        (norm_error, "additional_source_error_upper_bound"),
        (finite, "finite_certificate"),
        (term_count, "direct_term_count"),
        (term_budget, "maximum_direct_terms"),
    ):
        _raise_if(value.shape != (), f"{name} must be scalar")
    for text, name in (
        (exact_target, "exact_target"),
        (arithmetic, "arithmetic"),
        (direct_term_count_route, "direct_term_count_route"),
        (error_scope, "error_scope"),
        (coefficient_norm, "coefficient_norm"),
    ):
        _raise_if(not text.strip(), f"{name} must be nonempty")
    for digest, name in (
        (parent_source_digest, "parent_source_digest"),
        (parent_target_evidence_digest, "parent_target_evidence_digest"),
        (certificate_digest, "certificate_digest"),
    ):
        _raise_if(
            not _valid_digest(digest), f"{name} must be a SHA-256 digest"
        )
    _raise_if(
        parent_source_digest != source.source_digest,
        "parent_source_digest must bind the nested source",
    )
    _raise_if(
        parent_target_evidence_digest
        != source.target.manifest_evidence_digest,
        "certificate must bind full parent target evidence",
    )
    success = failure is GalerkinLocalAdditionalSourceCertificateFailure.NONE
    invalid_common = (
        (term_count < 0)
        | (term_budget <= 0)
        | (finite != success)
        | jnp.any(real_lower > real_upper)
        | jnp.any(imag_lower > imag_upper)
        | jnp.any(jnp.isnan(errors))
        | jnp.any(errors < 0.0)
        | jnp.isnan(norm_error)
        | (norm_error < 0.0)
    )
    if success:
        invalid_outcome = (
            jnp.any(~jnp.isfinite(real_lower))
            | jnp.any(~jnp.isfinite(real_upper))
            | jnp.any(~jnp.isfinite(imag_lower))
            | jnp.any(~jnp.isfinite(imag_upper))
            | jnp.any(~jnp.isfinite(errors))
            | (~jnp.isfinite(norm_error))
            | has_subnormal_components(real_lower)
            | has_subnormal_components(real_upper)
            | has_subnormal_components(imag_lower)
            | has_subnormal_components(imag_upper)
            | has_subnormal_components(errors)
            | has_subnormal_components(norm_error)
            | (term_count > term_budget)
            | (norm_error < jnp.linalg.norm(errors))
        )
    else:
        invalid_outcome = (
            ~jnp.all(jnp.isneginf(real_lower))
            | ~jnp.all(jnp.isposinf(real_upper))
            | ~jnp.all(jnp.isneginf(imag_lower))
            | ~jnp.all(jnp.isposinf(imag_upper))
            | ~jnp.all(jnp.isposinf(errors))
            | ~jnp.isposinf(norm_error)
        )
        failure_type = GalerkinLocalAdditionalSourceCertificateFailure
        work_failure = failure_type.WORK_BUDGET_EXCEEDED
        if failure is work_failure:
            invalid_outcome = invalid_outcome | (term_count <= term_budget)
        else:
            invalid_outcome = invalid_outcome | (term_count > term_budget)
    checked_lower = eqx.error_if(
        real_lower,
        invalid_common | invalid_outcome,
        "local additional-source certificate outcome is inconsistent",
    )
    certificate: GalerkinLocalAdditionalSourceCertificate = (
        GalerkinLocalAdditionalSourceCertificate(
            source=source,
            exact_source_real_lower_bounds=checked_lower,
            exact_source_real_upper_bounds=real_upper,
            exact_source_imag_lower_bounds=imag_lower,
            exact_source_imag_upper_bounds=imag_upper,
            component_error_bounds=errors,
            additional_source_error_upper_bound=norm_error,
            finite_certificate=finite,
            direct_term_count=term_count,
            maximum_direct_terms=term_budget,
            failure=failure,
            exact_target=exact_target.strip(),
            arithmetic=arithmetic.strip(),
            direct_term_count_route=direct_term_count_route.strip(),
            error_scope=error_scope.strip(),
            coefficient_norm=coefficient_norm.strip(),
            parent_source_digest=parent_source_digest,
            parent_target_evidence_digest=parent_target_evidence_digest,
            certificate_digest=certificate_digest,
        )
    )
    return certificate


__all__: list[str] = [
    "GalerkinLocalAdditionalSource",
    "GalerkinLocalAdditionalSourceCertificate",
    "GalerkinLocalAdditionalSourceCertificateFailure",
    "GalerkinLocalAdditionalSourceRoute",
]
