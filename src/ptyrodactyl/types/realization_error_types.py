r"""Define the RM-S2 fixed-linear realization-error ledger.

Extended Summary
----------------
This module owns the typed evidence that separates one frozen complex-linear
Galerkin matrix from the exact SC-1 finite target.  It deliberately contains
no source, per-call action, residual-formation, or solver-iteration error.

Routine Listings
----------------
:class:`GalerkinFixedLinearAbsorberRoute`
    Store the admitted exact absorber and CAP realization route.
:class:`GalerkinFixedLinearErrorLedger`
    Store componentwise RM-S2 S2.27--S2.43 evidence.
:func:`create_galerkin_fixed_linear_error_ledger`
    Create a structurally validated fixed-linear error ledger.

Notes
-----
Positive infinity is an explicit noncertificate rather than a structural
error.  The dynamic ``finite_certificate`` field records that distinction.
"""

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

    Parameters
    ----------
    condition : bool
        Dimensionless structural failure predicate.
    message : str
        Unit-free exception text used when the predicate is true.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


class GalerkinFixedLinearAbsorberRoute(str, Enum):
    """Store the admitted exact absorber and CAP realization route.

    :see: :class:`~.test_realization_error_types.TestFixedLinearErrorTypes`

    Attributes
    ----------
    ANALYTIC_COSINE_SHELL_EXACT_DYADIC : str
        The SC-1 cosine-shell coefficients are exact dyadic rationals and the
        binary64 CAP input is interpreted as the exact target scale.
    """

    ANALYTIC_COSINE_SHELL_EXACT_DYADIC = (
        "sc1_analytic_cosine_shell_exact_dyadic"
    )


class GalerkinFixedLinearErrorLedger(eqx.Module):
    """Store componentwise RM-S2 S2.27--S2.43 evidence.

    :see: :class:`~.test_realization_error_types.TestFixedLinearErrorTypes`

    Attributes
    ----------
    algebraic_free_diagonal : Float64[Array, " n"]
        Canonical frozen binary64 SC.22 diagonal used by ``H_alg``.
    exact_wavenumber_lower_bound : Float64[Array, ""]
        Lower endpoint enclosing the exact SC.2 positive wavenumber.
    exact_wavenumber_upper_bound : Float64[Array, ""]
        Upper endpoint enclosing the exact SC.2 positive wavenumber.
    wavenumber_error_bound : Float64[Array, ""]
        Outward error of the stored acquisition wavenumber.
    exact_carrier_lower_bounds : Float64[Array, " 3"]
        Lower endpoints for the exact normalized SC.8 carrier.
    exact_carrier_upper_bounds : Float64[Array, " 3"]
        Upper endpoints for the exact normalized SC.8 carrier.
    carrier_component_error_bounds : Float64[Array, " 3"]
        Outward componentwise errors of the stored carrier seed.
    box_length_error_bounds : Float64[Array, " 3"]
        Box errors. They are zero because the manifested binary lengths are
        the exact target box for this route.
    exact_free_diagonal_lower_bounds : Float64[Array, " n"]
        Lower endpoints for the exact on-shell SC.23 diagonal.
    exact_free_diagonal_upper_bounds : Float64[Array, " n"]
        Upper endpoints for the exact on-shell SC.23 diagonal.
    free_diagonal_error_bounds : Float64[Array, " n"]
        Outward componentwise diagonal errors.
    free_operator_error_bound : Float64[Array, ""]
        S2.28 bound ``delta_D``.
    exact_interaction_coupling_lower_bound : Float64[Array, ""]
        Lower endpoint for the exact SC.4 coupling.
    exact_interaction_coupling_upper_bound : Float64[Array, ""]
        Upper endpoint for the exact SC.4 coupling.
    interaction_coupling_error_bound : Float64[Array, ""]
        Outward error of the stored canonical coupling.
    interaction_rounding_error_bounds : Float64[Array, " p"]
        Errors between stored interaction coefficients and exact products of
        the stored coupling and stored voltage coefficients.
    interaction_coupling_transfer_error_bounds : Float64[Array, " p"]
        Errors caused by replacing the exact SC.4 coupling by the stored
        canonical coupling while retaining the stored voltage coefficient.
    interaction_voltage_transfer_error_bounds : Float64[Array, " p"]
        Exact-coupling multiples of the componentwise VC-1 coefficient errors.
    interaction_coefficient_error_bounds : Float64[Array, " p"]
        Complete componentwise ``e_chi`` bounds on the interaction support.
    difference_multiplicities : Int64[Array, " p"]
        S2.30 multiplicities aligned with the interaction support. Differences
        absent from that support have identically zero exact and algebraic
        coefficients under the projected target.
    interaction_row_error_bounds : Float64[Array, " n"]
        Outward absolute row sums before the S2.31 maximum.
    interaction_column_error_bounds : Float64[Array, " n"]
        Outward absolute column sums before the S2.32 maximum.
    interaction_max_row_error_bound : Float64[Array, ""]
        S2.31 maximum row-sum bound ``r_chi``.
    interaction_max_column_error_bound : Float64[Array, ""]
        S2.32 maximum column-sum bound ``c_chi``.
    interaction_schur_error_bound : Float64[Array, ""]
        Outward ``sqrt(r_chi c_chi)`` bound.
    interaction_frobenius_error_bound : Float64[Array, ""]
        Outward multiplicity-weighted Frobenius bound in S2.33a.
    interaction_operator_error_bound : Float64[Array, ""]
        S2.33a bound ``delta_R``, the minimum of Schur and Frobenius.
    absorber_operator_error_bound : Float64[Array, ""]
        S2.34 absorber error ``delta_A``.
    cap_scale_error_bound : Float64[Array, ""]
        S2.35 CAP scale error ``delta_epsilon``.
    cap_operator_error_bound : Float64[Array, ""]
        S2.37 CAP error ``delta_B``.
    fixed_linear_operator_error_bound : Float64[Array, ""]
        S2.43a total ``delta_H = delta_D + delta_R + delta_B``.
    finite_certificate : Bool[Array, ""]
        Whether every final fixed-linear component bound is finite.
    absorber_route : GalerkinFixedLinearAbsorberRoute
        Static absorber/CAP realization route. This value affects tracing.
    exact_geometry_target : str
        Static SC.2/SC.8 exact-target provenance. This value affects tracing.
    algebraic_geometry_realization : str
        Static frozen-diagonal provenance. This value affects tracing.
    exact_interaction_target : str
        Static SC.4 exact-target provenance. This value affects tracing.
    algebraic_interaction_realization : str
        Static canonical coupling provenance. This value affects tracing.
    absorber_target : str
        Static exact cosine-shell target provenance. This value affects
        tracing.
    cap_target : str
        Static exact CAP-scale provenance. This value affects tracing.
    error_scope : str
        Static declaration excluding non-fixed-linear errors.
    coefficient_norm : str
        Static norm in which all matrix bounds hold.

    Notes
    -----
    Coefficient errors outside the interaction support are exactly zero for
    both sides because the exact target is ``P_Kchi chi P_Ku``.  The
    multiplicity array therefore needs entries only for represented
    interaction coefficients.

    See Also
    --------
    :func:`create_galerkin_fixed_linear_error_ledger`
        Construct and structurally validate this carrier.
    """

    algebraic_free_diagonal: Float64[Array, " n"]
    exact_wavenumber_lower_bound: Float64[Array, ""]
    exact_wavenumber_upper_bound: Float64[Array, ""]
    wavenumber_error_bound: Float64[Array, ""]
    exact_carrier_lower_bounds: Float64[Array, " 3"]
    exact_carrier_upper_bounds: Float64[Array, " 3"]
    carrier_component_error_bounds: Float64[Array, " 3"]
    box_length_error_bounds: Float64[Array, " 3"]
    exact_free_diagonal_lower_bounds: Float64[Array, " n"]
    exact_free_diagonal_upper_bounds: Float64[Array, " n"]
    free_diagonal_error_bounds: Float64[Array, " n"]
    free_operator_error_bound: Float64[Array, ""]
    exact_interaction_coupling_lower_bound: Float64[Array, ""]
    exact_interaction_coupling_upper_bound: Float64[Array, ""]
    interaction_coupling_error_bound: Float64[Array, ""]
    interaction_rounding_error_bounds: Float64[Array, " p"]
    interaction_coupling_transfer_error_bounds: Float64[Array, " p"]
    interaction_voltage_transfer_error_bounds: Float64[Array, " p"]
    interaction_coefficient_error_bounds: Float64[Array, " p"]
    difference_multiplicities: Int64[Array, " p"]
    interaction_row_error_bounds: Float64[Array, " n"]
    interaction_column_error_bounds: Float64[Array, " n"]
    interaction_max_row_error_bound: Float64[Array, ""]
    interaction_max_column_error_bound: Float64[Array, ""]
    interaction_schur_error_bound: Float64[Array, ""]
    interaction_frobenius_error_bound: Float64[Array, ""]
    interaction_operator_error_bound: Float64[Array, ""]
    absorber_operator_error_bound: Float64[Array, ""]
    cap_scale_error_bound: Float64[Array, ""]
    cap_operator_error_bound: Float64[Array, ""]
    fixed_linear_operator_error_bound: Float64[Array, ""]
    finite_certificate: Bool[Array, ""]
    absorber_route: GalerkinFixedLinearAbsorberRoute = eqx.field(static=True)
    exact_geometry_target: str = eqx.field(static=True)
    algebraic_geometry_realization: str = eqx.field(static=True)
    exact_interaction_target: str = eqx.field(static=True)
    algebraic_interaction_realization: str = eqx.field(static=True)
    absorber_target: str = eqx.field(static=True)
    cap_target: str = eqx.field(static=True)
    error_scope: str = eqx.field(static=True)
    coefficient_norm: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_fixed_linear_error_ledger(  # noqa: PLR0913, PLR0915
    algebraic_free_diagonal: Float[Array, "..."],
    exact_wavenumber_lower_bound: Float[Array, ""],
    exact_wavenumber_upper_bound: Float[Array, ""],
    wavenumber_error_bound: Float[Array, ""],
    exact_carrier_lower_bounds: Float[Array, "..."],
    exact_carrier_upper_bounds: Float[Array, "..."],
    carrier_component_error_bounds: Float[Array, "..."],
    box_length_error_bounds: Float[Array, "..."],
    exact_free_diagonal_lower_bounds: Float[Array, "..."],
    exact_free_diagonal_upper_bounds: Float[Array, "..."],
    free_diagonal_error_bounds: Float[Array, "..."],
    free_operator_error_bound: Float[Array, ""],
    exact_interaction_coupling_lower_bound: Float[Array, ""],
    exact_interaction_coupling_upper_bound: Float[Array, ""],
    interaction_coupling_error_bound: Float[Array, ""],
    interaction_rounding_error_bounds: Float[Array, "..."],
    interaction_coupling_transfer_error_bounds: Float[Array, "..."],
    interaction_voltage_transfer_error_bounds: Float[Array, "..."],
    interaction_coefficient_error_bounds: Float[Array, "..."],
    difference_multiplicities: Int[Array, "..."],
    interaction_row_error_bounds: Float[Array, "..."],
    interaction_column_error_bounds: Float[Array, "..."],
    interaction_max_row_error_bound: Float[Array, ""],
    interaction_max_column_error_bound: Float[Array, ""],
    interaction_schur_error_bound: Float[Array, ""],
    interaction_frobenius_error_bound: Float[Array, ""],
    interaction_operator_error_bound: Float[Array, ""],
    absorber_operator_error_bound: Float[Array, ""],
    cap_scale_error_bound: Float[Array, ""],
    cap_operator_error_bound: Float[Array, ""],
    fixed_linear_operator_error_bound: Float[Array, ""],
    finite_certificate: Bool[Array, ""],
    *,
    absorber_route: GalerkinFixedLinearAbsorberRoute | str,
    exact_geometry_target: str,
    algebraic_geometry_realization: str,
    exact_interaction_target: str,
    algebraic_interaction_realization: str,
    absorber_target: str,
    cap_target: str,
    error_scope: str,
    coefficient_norm: str,
) -> GalerkinFixedLinearErrorLedger:
    """Create a structurally validated fixed-linear error ledger.

    :see: :class:`~.test_realization_error_types.TestFixedLinearErrorTypes`

    Parameters
    ----------
    algebraic_free_diagonal : Float[Array, "..."]
        Canonical frozen SC.22 diagonal.
    exact_wavenumber_lower_bound : Float[Array, ""]
        Lower exact-SC.2 wavenumber endpoint.
    exact_wavenumber_upper_bound : Float[Array, ""]
        Upper exact-SC.2 wavenumber endpoint.
    wavenumber_error_bound : Float[Array, ""]
        Stored-wavenumber absolute error bound.
    exact_carrier_lower_bounds : Float[Array, "..."]
        Lower exact-SC.8 carrier endpoints.
    exact_carrier_upper_bounds : Float[Array, "..."]
        Upper exact-SC.8 carrier endpoints.
    carrier_component_error_bounds : Float[Array, "..."]
        Stored-carrier component error bounds.
    box_length_error_bounds : Float[Array, "..."]
        Exact-target box length error bounds.
    exact_free_diagonal_lower_bounds : Float[Array, "..."]
        Lower exact-SC.23 diagonal endpoints.
    exact_free_diagonal_upper_bounds : Float[Array, "..."]
        Upper exact-SC.23 diagonal endpoints.
    free_diagonal_error_bounds : Float[Array, "..."]
        Componentwise free-diagonal error bounds.
    free_operator_error_bound : Float[Array, ""]
        RM-S2 ``delta_D``.
    exact_interaction_coupling_lower_bound : Float[Array, ""]
        Lower exact-SC.4 coupling endpoint.
    exact_interaction_coupling_upper_bound : Float[Array, ""]
        Upper exact-SC.4 coupling endpoint.
    interaction_coupling_error_bound : Float[Array, ""]
        Stored-coupling absolute error bound.
    interaction_rounding_error_bounds : Float[Array, "..."]
        Stored interaction-product rounding errors.
    interaction_coupling_transfer_error_bounds : Float[Array, "..."]
        Coupling-replacement coefficient errors.
    interaction_voltage_transfer_error_bounds : Float[Array, "..."]
        VC-1 coefficient-error transfer bounds.
    interaction_coefficient_error_bounds : Float[Array, "..."]
        Complete represented interaction coefficient errors.
    difference_multiplicities : Int[Array, "..."]
        S2.30 multiplicities.
    interaction_row_error_bounds : Float[Array, "..."]
        Unmaximized S2.31 row sums.
    interaction_column_error_bounds : Float[Array, "..."]
        Unmaximized S2.32 column sums.
    interaction_max_row_error_bound : Float[Array, ""]
        S2.31 maximum row sum.
    interaction_max_column_error_bound : Float[Array, ""]
        S2.32 maximum column sum.
    interaction_schur_error_bound : Float[Array, ""]
        S2.33a Schur bound.
    interaction_frobenius_error_bound : Float[Array, ""]
        S2.33a Frobenius bound.
    interaction_operator_error_bound : Float[Array, ""]
        RM-S2 ``delta_R``.
    absorber_operator_error_bound : Float[Array, ""]
        RM-S2 ``delta_A``.
    cap_scale_error_bound : Float[Array, ""]
        RM-S2 ``delta_epsilon``.
    cap_operator_error_bound : Float[Array, ""]
        RM-S2 ``delta_B``.
    fixed_linear_operator_error_bound : Float[Array, ""]
        RM-S2 ``delta_H``.
    finite_certificate : Bool[Array, ""]
        Whether all final fixed-linear component bounds are finite.
    absorber_route : GalerkinFixedLinearAbsorberRoute | str
        Exact absorber/CAP route.
    exact_geometry_target : str
        Exact geometry target provenance.
    algebraic_geometry_realization : str
        Algebraic geometry realization provenance.
    exact_interaction_target : str
        Exact interaction target provenance.
    algebraic_interaction_realization : str
        Algebraic interaction realization provenance.
    absorber_target : str
        Exact absorber target provenance.
    cap_target : str
        Exact CAP target provenance.
    error_scope : str
        Fixed-linear-only error scope.
    coefficient_norm : str
        Coefficient norm for every operator bound.

    Returns
    -------
    ledger : GalerkinFixedLinearErrorLedger
        Structurally validated exact-width carrier.

    Raises
    ------
    ValueError
        If a dynamic array shape or static identifier is invalid.
    equinox.EquinoxRuntimeError
        If a value is NaN, a bound is negative, an interval is reversed, or
        the finite/noncertificate flag contradicts the final bounds.
    """
    free_diagonal: Float64[Array, " n"] = jnp.asarray(
        algebraic_free_diagonal,
        dtype=jnp.float64,
    )
    wavenumber_lower: Float64[Array, ""] = jnp.asarray(
        exact_wavenumber_lower_bound,
        dtype=jnp.float64,
    )
    wavenumber_upper: Float64[Array, ""] = jnp.asarray(
        exact_wavenumber_upper_bound,
        dtype=jnp.float64,
    )
    wavenumber_error: Float64[Array, ""] = jnp.asarray(
        wavenumber_error_bound,
        dtype=jnp.float64,
    )
    carrier_lower: Float64[Array, " 3"] = jnp.asarray(
        exact_carrier_lower_bounds,
        dtype=jnp.float64,
    )
    carrier_upper: Float64[Array, " 3"] = jnp.asarray(
        exact_carrier_upper_bounds,
        dtype=jnp.float64,
    )
    carrier_errors: Float64[Array, " 3"] = jnp.asarray(
        carrier_component_error_bounds,
        dtype=jnp.float64,
    )
    box_errors: Float64[Array, " 3"] = jnp.asarray(
        box_length_error_bounds,
        dtype=jnp.float64,
    )
    free_lower: Float64[Array, " n"] = jnp.asarray(
        exact_free_diagonal_lower_bounds,
        dtype=jnp.float64,
    )
    free_upper: Float64[Array, " n"] = jnp.asarray(
        exact_free_diagonal_upper_bounds,
        dtype=jnp.float64,
    )
    free_errors: Float64[Array, " n"] = jnp.asarray(
        free_diagonal_error_bounds,
        dtype=jnp.float64,
    )
    delta_d: Float64[Array, ""] = jnp.asarray(
        free_operator_error_bound,
        dtype=jnp.float64,
    )
    coupling_lower: Float64[Array, ""] = jnp.asarray(
        exact_interaction_coupling_lower_bound,
        dtype=jnp.float64,
    )
    coupling_upper: Float64[Array, ""] = jnp.asarray(
        exact_interaction_coupling_upper_bound,
        dtype=jnp.float64,
    )
    coupling_error: Float64[Array, ""] = jnp.asarray(
        interaction_coupling_error_bound,
        dtype=jnp.float64,
    )
    rounding_errors: Float64[Array, " p"] = jnp.asarray(
        interaction_rounding_error_bounds,
        dtype=jnp.float64,
    )
    coupling_transfer_errors: Float64[Array, " p"] = jnp.asarray(
        interaction_coupling_transfer_error_bounds,
        dtype=jnp.float64,
    )
    voltage_transfer_errors: Float64[Array, " p"] = jnp.asarray(
        interaction_voltage_transfer_error_bounds,
        dtype=jnp.float64,
    )
    coefficient_errors: Float64[Array, " p"] = jnp.asarray(
        interaction_coefficient_error_bounds,
        dtype=jnp.float64,
    )
    multiplicities: Int64[Array, " p"] = jnp.asarray(
        difference_multiplicities,
        dtype=jnp.int64,
    )
    row_errors: Float64[Array, " n"] = jnp.asarray(
        interaction_row_error_bounds,
        dtype=jnp.float64,
    )
    column_errors: Float64[Array, " n"] = jnp.asarray(
        interaction_column_error_bounds,
        dtype=jnp.float64,
    )
    maximum_row: Float64[Array, ""] = jnp.asarray(
        interaction_max_row_error_bound,
        dtype=jnp.float64,
    )
    maximum_column: Float64[Array, ""] = jnp.asarray(
        interaction_max_column_error_bound,
        dtype=jnp.float64,
    )
    schur: Float64[Array, ""] = jnp.asarray(
        interaction_schur_error_bound,
        dtype=jnp.float64,
    )
    frobenius: Float64[Array, ""] = jnp.asarray(
        interaction_frobenius_error_bound,
        dtype=jnp.float64,
    )
    delta_r: Float64[Array, ""] = jnp.asarray(
        interaction_operator_error_bound,
        dtype=jnp.float64,
    )
    delta_a: Float64[Array, ""] = jnp.asarray(
        absorber_operator_error_bound,
        dtype=jnp.float64,
    )
    delta_epsilon: Float64[Array, ""] = jnp.asarray(
        cap_scale_error_bound,
        dtype=jnp.float64,
    )
    delta_b: Float64[Array, ""] = jnp.asarray(
        cap_operator_error_bound,
        dtype=jnp.float64,
    )
    delta_h: Float64[Array, ""] = jnp.asarray(
        fixed_linear_operator_error_bound,
        dtype=jnp.float64,
    )
    finite: Bool[Array, ""] = jnp.asarray(finite_certificate, dtype=jnp.bool_)
    route = GalerkinFixedLinearAbsorberRoute(absorber_route)

    _raise_if(free_diagonal.ndim != 1, "algebraic_free_diagonal must be 1D")
    _raise_if(free_diagonal.shape[0] == 0, "free diagonal must be nonempty")
    _raise_if(
        carrier_lower.shape != (3,)
        or carrier_upper.shape != (3,)
        or carrier_errors.shape != (3,)
        or box_errors.shape != (3,),
        "carrier and box evidence must have shape (3,)",
    )
    for values, name in (
        (free_lower, "exact_free_diagonal_lower_bounds"),
        (free_upper, "exact_free_diagonal_upper_bounds"),
        (free_errors, "free_diagonal_error_bounds"),
        (row_errors, "interaction_row_error_bounds"),
        (column_errors, "interaction_column_error_bounds"),
    ):
        _raise_if(
            values.shape != free_diagonal.shape,
            f"{name} must match algebraic_free_diagonal",
        )
    for values, name in (
        (
            coupling_transfer_errors,
            "interaction_coupling_transfer_error_bounds",
        ),
        (voltage_transfer_errors, "interaction_voltage_transfer_error_bounds"),
        (coefficient_errors, "interaction_coefficient_error_bounds"),
        (multiplicities, "difference_multiplicities"),
    ):
        _raise_if(
            values.shape != rounding_errors.shape,
            f"{name} must match interaction_rounding_error_bounds",
        )
    _raise_if(rounding_errors.ndim != 1, "interaction errors must be 1D")
    for scalar, name in (
        (wavenumber_lower, "exact_wavenumber_lower_bound"),
        (wavenumber_upper, "exact_wavenumber_upper_bound"),
        (wavenumber_error, "wavenumber_error_bound"),
        (delta_d, "free_operator_error_bound"),
        (coupling_lower, "exact_interaction_coupling_lower_bound"),
        (coupling_upper, "exact_interaction_coupling_upper_bound"),
        (coupling_error, "interaction_coupling_error_bound"),
        (maximum_row, "interaction_max_row_error_bound"),
        (maximum_column, "interaction_max_column_error_bound"),
        (schur, "interaction_schur_error_bound"),
        (frobenius, "interaction_frobenius_error_bound"),
        (delta_r, "interaction_operator_error_bound"),
        (delta_a, "absorber_operator_error_bound"),
        (delta_epsilon, "cap_scale_error_bound"),
        (delta_b, "cap_operator_error_bound"),
        (delta_h, "fixed_linear_operator_error_bound"),
        (finite, "finite_certificate"),
    ):
        _raise_if(scalar.shape != (), f"{name} must be a scalar")
    for value, name in (
        (exact_geometry_target, "exact_geometry_target"),
        (algebraic_geometry_realization, "algebraic_geometry_realization"),
        (exact_interaction_target, "exact_interaction_target"),
        (
            algebraic_interaction_realization,
            "algebraic_interaction_realization",
        ),
        (absorber_target, "absorber_target"),
        (cap_target, "cap_target"),
        (error_scope, "error_scope"),
        (coefficient_norm, "coefficient_norm"),
    ):
        _raise_if(not value.strip(), f"{name} must be nonempty")

    checked_free_diagonal: Float64[Array, " n"] = eqx.error_if(
        free_diagonal,
        jnp.any(~jnp.isfinite(free_diagonal)),
        "algebraic_free_diagonal must be finite",
    )
    checked_free_lower: Float64[Array, " n"] = eqx.error_if(
        free_lower,
        jnp.any(jnp.isnan(free_lower))
        | jnp.any(jnp.isnan(free_upper))
        | jnp.any(free_lower > free_upper),
        "exact free diagonal intervals must be ordered and not NaN",
    )
    checked_wavenumber_lower: Float64[Array, ""] = eqx.error_if(
        wavenumber_lower,
        jnp.isnan(wavenumber_lower)
        | jnp.isnan(wavenumber_upper)
        | (wavenumber_lower > wavenumber_upper),
        "exact wavenumber interval must be ordered and not NaN",
    )
    checked_carrier_lower: Float64[Array, " 3"] = eqx.error_if(
        carrier_lower,
        jnp.any(jnp.isnan(carrier_lower))
        | jnp.any(jnp.isnan(carrier_upper))
        | jnp.any(carrier_lower > carrier_upper),
        "exact carrier intervals must be ordered and not NaN",
    )
    checked_coupling_lower: Float64[Array, ""] = eqx.error_if(
        coupling_lower,
        jnp.isnan(coupling_lower)
        | jnp.isnan(coupling_upper)
        | (coupling_lower > coupling_upper),
        "exact interaction coupling interval must be ordered and not NaN",
    )
    checked_multiplicities: Int64[Array, " p"] = eqx.error_if(
        multiplicities,
        jnp.any(multiplicities < 0),
        "difference_multiplicities must be non-negative",
    )
    nonnegative_bounds = (
        wavenumber_error,
        carrier_errors,
        box_errors,
        free_errors,
        delta_d,
        coupling_error,
        rounding_errors,
        coupling_transfer_errors,
        voltage_transfer_errors,
        coefficient_errors,
        row_errors,
        column_errors,
        maximum_row,
        maximum_column,
        schur,
        frobenius,
        delta_r,
        delta_a,
        delta_epsilon,
        delta_b,
        delta_h,
    )
    invalid_bound: Bool[Array, ""] = jnp.asarray(False)
    for bound in nonnegative_bounds:
        invalid_bound = (
            invalid_bound | jnp.any(jnp.isnan(bound)) | jnp.any(bound < 0.0)
        )
    expected_finite: Bool[Array, ""] = (
        jnp.isfinite(delta_d)
        & jnp.isfinite(delta_r)
        & jnp.isfinite(delta_b)
        & jnp.isfinite(delta_h)
    )
    checked_total: Float64[Array, ""] = eqx.error_if(
        delta_h,
        invalid_bound
        | (finite != expected_finite)
        | (delta_a != 0.0)
        | (delta_epsilon != 0.0)
        | (delta_b != 0.0),
        "fixed-linear bounds, exact absorber route, or finite flag are "
        "invalid",
    )

    ledger: GalerkinFixedLinearErrorLedger = GalerkinFixedLinearErrorLedger(
        algebraic_free_diagonal=checked_free_diagonal,
        exact_wavenumber_lower_bound=checked_wavenumber_lower,
        exact_wavenumber_upper_bound=wavenumber_upper,
        wavenumber_error_bound=wavenumber_error,
        exact_carrier_lower_bounds=checked_carrier_lower,
        exact_carrier_upper_bounds=carrier_upper,
        carrier_component_error_bounds=carrier_errors,
        box_length_error_bounds=box_errors,
        exact_free_diagonal_lower_bounds=checked_free_lower,
        exact_free_diagonal_upper_bounds=free_upper,
        free_diagonal_error_bounds=free_errors,
        free_operator_error_bound=delta_d,
        exact_interaction_coupling_lower_bound=checked_coupling_lower,
        exact_interaction_coupling_upper_bound=coupling_upper,
        interaction_coupling_error_bound=coupling_error,
        interaction_rounding_error_bounds=rounding_errors,
        interaction_coupling_transfer_error_bounds=coupling_transfer_errors,
        interaction_voltage_transfer_error_bounds=voltage_transfer_errors,
        interaction_coefficient_error_bounds=coefficient_errors,
        difference_multiplicities=checked_multiplicities,
        interaction_row_error_bounds=row_errors,
        interaction_column_error_bounds=column_errors,
        interaction_max_row_error_bound=maximum_row,
        interaction_max_column_error_bound=maximum_column,
        interaction_schur_error_bound=schur,
        interaction_frobenius_error_bound=frobenius,
        interaction_operator_error_bound=delta_r,
        absorber_operator_error_bound=delta_a,
        cap_scale_error_bound=delta_epsilon,
        cap_operator_error_bound=delta_b,
        fixed_linear_operator_error_bound=checked_total,
        finite_certificate=finite,
        absorber_route=route,
        exact_geometry_target=exact_geometry_target.strip(),
        algebraic_geometry_realization=algebraic_geometry_realization.strip(),
        exact_interaction_target=exact_interaction_target.strip(),
        algebraic_interaction_realization=(
            algebraic_interaction_realization.strip()
        ),
        absorber_target=absorber_target.strip(),
        cap_target=cap_target.strip(),
        error_scope=error_scope.strip(),
        coefficient_norm=coefficient_norm.strip(),
    )
    return ledger


__all__: list[str] = [
    "GalerkinFixedLinearAbsorberRoute",
    "GalerkinFixedLinearErrorLedger",
    "create_galerkin_fixed_linear_error_ledger",
]
