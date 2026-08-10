"""Tests for :mod:`ptyrodactyl.types.action_error_types`."""

from ptyrodactyl.types.action_error_types import (
    GalerkinActionDirection,
    GalerkinActionErrorRoute,
    GalerkinResidualErrorEnclosure,
    GalerkinTargetActionEnclosure,
    create_galerkin_residual_error_enclosure,
    create_galerkin_target_action_enclosure,
)


class TestActionErrorTypes:
    """Freeze the submitted-state action-evidence vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinActionDirection`
    :see: :class:`ptyrodactyl.types.GalerkinActionErrorRoute`
    :see: :class:`ptyrodactyl.types.GalerkinResidualErrorEnclosure`
    :see: :class:`ptyrodactyl.types.GalerkinTargetActionEnclosure`
    :see: :func:`ptyrodactyl.types.create_galerkin_residual_error_enclosure`
    :see: :func:`ptyrodactyl.types.create_galerkin_target_action_enclosure`
    """

    def test_route_names_ftz_safe_direct_interval_bridge(self) -> None:
        """Freeze the independently recomputed finite enclosure route."""
        assert GalerkinActionDirection.FORWARD.value == "forward"
        assert GalerkinActionDirection.ADJOINT.value == "adjoint"
        assert (
            GalerkinActionErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE.value
            == "rm_s2_ftz_safe_direct_interval_bridge"
        )

    def test_carriers_exclude_fixed_and_source_error_terms(self) -> None:
        """Prevent submitted-state evidence from absorbing other ledgers."""
        for carrier in (
            GalerkinTargetActionEnclosure,
            GalerkinResidualErrorEnclosure,
        ):
            for forbidden in (
                "fixed_linear_operator_error_bound",
                "source_error_bound",
                "solver_error_bound",
            ):
                assert forbidden not in carrier.__annotations__
        assert (
            "action_error_bound"
            in GalerkinTargetActionEnclosure.__annotations__
        )
        assert (
            "algebraic_residual_norm_upper_bound"
            in GalerkinResidualErrorEnclosure.__annotations__
        )
        assert callable(create_galerkin_target_action_enclosure)
        assert callable(create_galerkin_residual_error_enclosure)
