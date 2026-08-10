"""Tests for :mod:`ptyrodactyl.types.realization_error_types`."""

import jax
import jax.numpy as jnp

from ptyrodactyl._physics import coupled_interaction_value
from ptyrodactyl.galerkin.enclosures import (
    build_galerkin_fixed_linear_error_ledger,
)
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    M_E,
    GalerkinFixedLinearAbsorberRoute,
    GalerkinFixedLinearErrorLedger,
    create_galerkin_fixed_linear_error_ledger,
)


def _ledger() -> GalerkinFixedLinearErrorLedger:
    """Build one singleton typed ledger."""
    zero = jnp.zeros((1, 3), dtype=jnp.int64)
    voltage = jnp.asarray([0.25 + 0.0j], dtype=jnp.complex128)
    voltage_kv = jnp.asarray(200.0, dtype=jnp.float64)
    coupling, interaction = coupled_interaction_value(
        voltage,
        voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    result = build_galerkin_fixed_linear_error_ledger(
        state_indices=zero,
        interaction_indices=zero,
        voltage_coefficients=voltage,
        voltage_coefficient_error_bounds=jnp.asarray([0.125]),
        interaction_coupling=coupling,
        interaction_coefficients=interaction,
        accelerating_voltage_kv=voltage_kv,
        carrier=jnp.asarray((0.0, 0.0, 250.0)),
        box_lengths=jnp.asarray((8.0, 9.0, 10.0)),
        wavenumber=jnp.asarray(250.0),
        cap_scale=jnp.asarray(0.5),
    )
    return result


class TestFixedLinearErrorTypes:
    """Verify the typed fixed-linear evidence vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinFixedLinearAbsorberRoute`
    :see: :class:`ptyrodactyl.types.GalerkinFixedLinearErrorLedger`
    :see: :func:`ptyrodactyl.types.create_galerkin_fixed_linear_error_ledger`
    """

    def test_exact_absorber_route_is_explicit(self) -> None:
        """Name the dyadic absorber argument behind every zero CAP error."""
        assert (
            GalerkinFixedLinearAbsorberRoute.ANALYTIC_COSINE_SHELL_EXACT_DYADIC.value
            == "sc1_analytic_cosine_shell_exact_dyadic"
        )

    def test_ledger_uses_exact_widths_and_excludes_other_error_classes(
        self,
    ) -> None:
        """Store typed S2 evidence without source or action-error fields."""
        ledger = _ledger()
        jax.block_until_ready(ledger)

        assert ledger.algebraic_free_diagonal.dtype == jnp.float64
        assert ledger.interaction_coefficient_error_bounds.dtype == jnp.float64
        assert ledger.difference_multiplicities.dtype == jnp.int64
        assert ledger.finite_certificate.dtype == jnp.bool_
        assert ledger.absorber_route is (
            GalerkinFixedLinearAbsorberRoute.ANALYTIC_COSINE_SHELL_EXACT_DYADIC
        )
        assert "SC.2" in ledger.exact_geometry_target
        assert "SC.4" in ledger.exact_interaction_target
        assert "per-call" in ledger.error_scope
        for forbidden in (
            "source_error_bound",
            "action_error_bound",
            "residual_error_bound",
            "solver_error_bound",
        ):
            assert (
                forbidden not in GalerkinFixedLinearErrorLedger.__annotations__
            )
        for field_name in (
            "free_operator_error_bound",
            "interaction_operator_error_bound",
            "cap_operator_error_bound",
            "fixed_linear_operator_error_bound",
        ):
            assert GalerkinFixedLinearErrorLedger.__annotations__[
                field_name
            ].dtypes == ("float64",)
        assert create_galerkin_fixed_linear_error_ledger.__name__ == (
            "create_galerkin_fixed_linear_error_ledger"
        )
