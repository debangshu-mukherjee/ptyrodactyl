"""Tests for :mod:`ptyrodactyl.types.terminal_types`."""

import jax.numpy as jnp

from ptyrodactyl.galerkin.terminal import enclose_galerkin_terminal_current
from ptyrodactyl.types.terminal_types import (
    GalerkinCoordinateCauchyCurrent,
    GalerkinCurrentOperatorCertificate,
    GalerkinCurrentOperatorFailure,
    GalerkinDetectorFailure,
    GalerkinTerminalCurrentActionEnclosure,
    GalerkinTerminalCurrentActionFailure,
    GalerkinTerminalCurrentFailure,
    GalerkinTerminalCurrentRoute,
    GalerkinTerminalCurrentScope,
    GalerkinVacuumBranchFailure,
    create_galerkin_coordinate_cauchy_current,
    create_galerkin_current_operator_certificate,
    create_galerkin_terminal_current_action_enclosure,
)
from tests._galerkin_target_fixture import production_target


class TestCoordinateTerminalTypes:
    """Bind the coordinate-current vocabulary to its carrier test.

    :see: :class:`ptyrodactyl.types.GalerkinCoordinateCauchyCurrent`
    :see: :class:`ptyrodactyl.types.GalerkinDetectorFailure`
    :see: :class:`ptyrodactyl.types.GalerkinTerminalCurrentFailure`
    :see: :class:`ptyrodactyl.types.GalerkinTerminalCurrentRoute`
    :see: :class:`ptyrodactyl.types.GalerkinTerminalCurrentScope`
    :see: :class:`ptyrodactyl.types.GalerkinVacuumBranchFailure`
    :see: :func:`ptyrodactyl.types.create_galerkin_coordinate_cauchy_current`
    """

    def test_carrier_keeps_three_eligibility_contracts_separate(self) -> None:
        """Keep finite current availability distinct from downstream claims."""
        target = production_target()
        field = jnp.asarray(
            (1.0 + 0.5j, -0.25 + 0.75j, 0.4 - 0.2j),
            dtype=jnp.complex128,
        )

        diagnostic = enclose_galerkin_terminal_current(target, field)

        assert isinstance(diagnostic, GalerkinCoordinateCauchyCurrent)
        assert bool(diagnostic.current_diagnostic_eligible)
        assert int(diagnostic.current_diagnostic_failure_mask) == int(
            GalerkinTerminalCurrentFailure.NONE
        )
        assert not bool(diagnostic.vacuum_branch_eligible)
        assert not bool(diagnostic.detector_eligible)
        assert "per-submitted-state" in diagnostic.eligibility_scope
        assert "no uniform exact-operator/action-error" in (
            diagnostic.eligibility_scope
        )
        assert diagnostic.route is (
            GalerkinTerminalCurrentRoute.FTZ_SAFE_EXACT_CARRIER_CAUCHY
        )
        assert diagnostic.current_scope is (
            GalerkinTerminalCurrentScope.SELECTED_ACQUISITION_FIBER_SECTOR
        )
        assert diagnostic.vacuum_branch_failure is (
            GalerkinVacuumBranchFailure.NO_COMPACT_LOCAL_VACUUM_SLAB_CONTRACT
        )
        assert diagnostic.detector_failure == (
            GalerkinDetectorFailure.NO_VACUUM_BRANCH
            | GalerkinDetectorFailure.NO_OUTGOING_EXTRACTION
            | GalerkinDetectorFailure.NO_PIXEL_RESPONSE
        )
        assert callable(create_galerkin_coordinate_cauchy_current)


class TestCurrentOperatorTypes:
    """Bind the uniform operator/action storage vocabulary to leaf tests.

    :see: :class:`ptyrodactyl.types.GalerkinCurrentOperatorCertificate`
    :see: :class:`ptyrodactyl.types.GalerkinCurrentOperatorFailure`
    :see: :class:`ptyrodactyl.types.\
GalerkinTerminalCurrentActionEnclosure`
    :see: :class:`ptyrodactyl.types.\
GalerkinTerminalCurrentActionFailure`
    :see: :func:`ptyrodactyl.types.\
create_galerkin_current_operator_certificate`
    :see: :func:`ptyrodactyl.types.\
create_galerkin_terminal_current_action_enclosure`
    """

    def test_operator_and_action_carriers_do_not_promote_downstream_claims(
        self,
    ) -> None:
        """Keep typed uniform evidence free of vacuum/detector booleans."""
        operator_fields = (
            GalerkinCurrentOperatorCertificate.__dataclass_fields__
        )
        action_fields = (
            GalerkinTerminalCurrentActionEnclosure.__dataclass_fields__
        )

        assert "current_operator_eligible" in operator_fields
        assert "finite_certificate" in action_fields
        assert "vacuum_branch_eligible" not in operator_fields
        assert "detector_eligible" not in operator_fields
        assert "vacuum_branch_eligible" not in action_fields
        assert "detector_eligible" not in action_fields
        assert int(GalerkinCurrentOperatorFailure.NONE) == 0
        assert int(GalerkinTerminalCurrentActionFailure.NONE) == 0
        assert callable(create_galerkin_current_operator_certificate)
        assert callable(create_galerkin_terminal_current_action_enclosure)
