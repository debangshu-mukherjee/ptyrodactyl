"""Tests for :mod:`ptyrodactyl.types.born_types`.

Extended Summary
----------------
These tests verify the structural and traced validation contracts of the
scalar Galerkin operator and solve-result carriers. They exercise sparse COO
structure without constructing a production dense operator.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any

from ptyrodactyl.types import (
    GalerkinCertificateReason,
    GalerkinOperator,
    GalerkinSolveMethod,
    GalerkinSolveResult,
    GalerkinSolveStatus,
    create_galerkin_operator,
    create_galerkin_solve_result,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _operator_inputs() -> dict[str, Any]:
    """Return one valid sparse scalar Galerkin operator input set."""
    inputs: dict[str, Any] = {
        "free_diagonal": jnp.array([-0.4, 0.2, 1.1], dtype=jnp.float64),
        "interaction_rows": jnp.array([0, 0, 1, 2], dtype=jnp.int32),
        "interaction_columns": jnp.array([0, 1, 0, 2], dtype=jnp.int32),
        "interaction_values": jnp.array(
            [0.7 + 0.0j, 0.1 + 0.03j, 0.1 - 0.03j, -0.2 + 0.0j],
            dtype=jnp.complex128,
        ),
        "absorber_factor_rows": jnp.array([0, 0, 1], dtype=jnp.int32),
        "absorber_factor_columns": jnp.array([0, 2, 1], dtype=jnp.int32),
        "absorber_factor_values": jnp.array(
            [0.8 + 0.0j, 0.1 - 0.2j, 0.6 + 0.0j],
            dtype=jnp.complex128,
        ),
        "cap_scale": jnp.array(0.25, dtype=jnp.float64),
        "absorber_factor_size": 2,
    }
    return inputs


def _result_inputs() -> dict[str, Any]:
    """Return one valid converged algebraic solve-result input set."""
    residual: jax.Array = jnp.array(
        [1.0e-9 + 2.0e-9j, -3.0e-9 + 0.0j, 2.0e-9 - 1.0e-9j],
        dtype=jnp.complex128,
    )
    inputs: dict[str, Any] = {
        "field": jnp.array(
            [0.4 + 0.2j, -0.1 + 0.5j, 0.7 - 0.3j],
            dtype=jnp.complex128,
        ),
        "residual": residual,
        "residual_norm": jnp.linalg.norm(residual),
        "normal_residual_norm": jnp.array(4.0e-9, dtype=jnp.float64),
        "recurrence_residual_norm": jnp.array(3.5e-9, dtype=jnp.float64),
        "iterations": jnp.array(7, dtype=jnp.int32),
        "operator_applications": jnp.array(16, dtype=jnp.int32),
        "status": GalerkinSolveStatus.CONVERGED,
        "converged": jnp.array(True),
        "method": GalerkinSolveMethod.CGLS,
        "certificate_reason": (
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND
        ),
    }
    return inputs


class TestGalerkinCarriers:
    """Verify the scalar Galerkin carrier and factory contracts.

    :see: :class:`ptyrodactyl.types.GalerkinCertificateReason`
    :see: :class:`ptyrodactyl.types.GalerkinOperator`
    :see: :class:`ptyrodactyl.types.GalerkinSolveMethod`
    :see: :class:`ptyrodactyl.types.GalerkinSolveResult`
    :see: :class:`ptyrodactyl.types.GalerkinSolveStatus`
    :see: :func:`ptyrodactyl.types.create_galerkin_operator`
    :see: :func:`ptyrodactyl.types.create_galerkin_solve_result`
    """

    def test_enum_values_freeze_public_status_vocabulary(self) -> None:
        """Freeze method, termination, and noncertificate enum values."""
        assert GalerkinSolveMethod.CGLS.value == "cgls"
        assert GalerkinSolveMethod.LSQR.value == "lsqr"
        assert int(GalerkinSolveStatus.CONVERGED) == 0
        assert int(GalerkinSolveStatus.MAX_ITERATIONS) == 1
        assert int(GalerkinSolveStatus.BREAKDOWN) == 2
        assert int(GalerkinSolveStatus.RESIDUAL_MISMATCH) == 3
        assert (
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND.value
            == "no_outward_residual_bound"
        )
        assert (
            GalerkinCertificateReason.NO_STABILITY_BOUND.value
            == "no_stability_bound"
        )
        assert (
            GalerkinCertificateReason.STATE_BUDGET_MISSED.value
            == "state_budget_missed"
        )
        assert (
            GalerkinCertificateReason.INVALID_OPERATOR_CONTRACT.value
            == "invalid_operator_contract"
        )

    def test_operator_factory_preserves_sparse_fields(self) -> None:
        """Preserve validated COO arrays without dense matrix construction."""
        inputs: dict[str, Any] = _operator_inputs()
        operator: GalerkinOperator = create_galerkin_operator(**inputs)
        jax.block_until_ready(operator)

        chex.assert_trees_all_close(
            operator.free_diagonal,
            inputs["free_diagonal"],
        )
        chex.assert_trees_all_equal(
            operator.interaction_rows,
            inputs["interaction_rows"],
        )
        chex.assert_trees_all_close(
            operator.interaction_values,
            inputs["interaction_values"],
        )
        chex.assert_trees_all_close(
            operator.absorber_factor_values,
            inputs["absorber_factor_values"],
        )
        assert operator.absorber_factor_size == 2

    def test_operator_factory_is_jittable_and_keeps_static_size(self) -> None:
        """Compile checks while retaining the factor row count as static."""
        inputs: dict[str, Any] = _operator_inputs()

        @jax.jit
        def compiled(cap_scale: jax.Array) -> GalerkinOperator:
            """Construct an operator with one traced CAP scale."""
            traced_inputs: dict[str, Any] = dict(inputs)
            traced_inputs["cap_scale"] = cap_scale
            operator: GalerkinOperator = create_galerkin_operator(
                **traced_inputs
            )
            return operator

        operator: GalerkinOperator = compiled(jnp.asarray(0.3))
        jax.block_until_ready(operator)
        assert operator.absorber_factor_size == 2
        chex.assert_trees_all_close(operator.cap_scale, jnp.asarray(0.3))

    def test_operator_factory_allows_zero_interaction_entries(self) -> None:
        """Allow the empty Hermitian interaction needed by free-space cases."""
        inputs: dict[str, Any] = _operator_inputs()
        inputs["interaction_rows"] = jnp.array([], dtype=jnp.int32)
        inputs["interaction_columns"] = jnp.array([], dtype=jnp.int32)
        inputs["interaction_values"] = jnp.array([], dtype=jnp.complex128)

        operator: GalerkinOperator = create_galerkin_operator(**inputs)
        jax.block_until_ready(operator)
        chex.assert_shape(operator.interaction_values, (0,))

    def test_operator_factory_uses_overflow_safe_coo_keys(self) -> None:
        """Keep distinct int32 coordinates unique beyond the key range."""
        state_size = 70_000
        aliasing_row = 61_356
        aliasing_column = 47_296
        assert aliasing_row * state_size + aliasing_column == 2**32
        interaction_rows = jnp.array(
            [0, aliasing_row, aliasing_column], dtype=jnp.int32
        )
        interaction_columns = jnp.array(
            [0, aliasing_column, aliasing_row], dtype=jnp.int32
        )
        absorber_rows = jnp.array([0, aliasing_row], dtype=jnp.int32)
        absorber_columns = jnp.array([0, aliasing_column], dtype=jnp.int32)

        operator = create_galerkin_operator(
            free_diagonal=jnp.ones(state_size, dtype=jnp.float64),
            interaction_rows=interaction_rows,
            interaction_columns=interaction_columns,
            interaction_values=jnp.array(
                [1.0 + 0.0j, 2.0 + 3.0j, 2.0 - 3.0j],
                dtype=jnp.complex128,
            ),
            absorber_factor_rows=absorber_rows,
            absorber_factor_columns=absorber_columns,
            absorber_factor_values=jnp.array(
                [1.0 + 0.0j, 0.5 - 0.25j], dtype=jnp.complex128
            ),
            cap_scale=jnp.asarray(0.25),
            absorber_factor_size=state_size,
        )
        jax.block_until_ready(operator)

        chex.assert_trees_all_equal(
            operator.interaction_rows, interaction_rows
        )
        chex.assert_trees_all_equal(
            operator.interaction_columns, interaction_columns
        )
        chex.assert_trees_all_equal(
            operator.absorber_factor_rows, absorber_rows
        )
        chex.assert_trees_all_equal(
            operator.absorber_factor_columns, absorber_columns
        )
        assert operator.interaction_rows.dtype == jnp.int32
        assert operator.absorber_factor_rows.dtype == jnp.int32

    def test_operator_factory_rejects_structural_errors(self) -> None:
        """Reject mismatched COO lengths and empty absorber factors eagerly."""
        mismatched: dict[str, Any] = _operator_inputs()
        mismatched["interaction_columns"] = jnp.array([0], dtype=jnp.int32)
        with pytest.raises(ValueError, match="matching shapes"):
            create_galerkin_operator(**mismatched)

        empty_factor: dict[str, Any] = _operator_inputs()
        empty_factor["absorber_factor_rows"] = jnp.array([], dtype=jnp.int32)
        empty_factor["absorber_factor_columns"] = jnp.array(
            [], dtype=jnp.int32
        )
        empty_factor["absorber_factor_values"] = jnp.array(
            [], dtype=jnp.complex128
        )
        with pytest.raises(ValueError, match="must be nonempty"):
            create_galerkin_operator(**empty_factor)

        invalid_size: dict[str, Any] = _operator_inputs()
        invalid_size["absorber_factor_size"] = 0
        with pytest.raises(ValueError, match="must be positive"):
            create_galerkin_operator(**invalid_size)

    @pytest.mark.parametrize(
        ("field_name", "bad_value", "message"),
        [
            ("cap_scale", -0.1, "cap_scale"),
            ("cap_scale", jnp.inf, "cap_scale"),
        ],
    )
    def test_operator_factory_rejects_invalid_traced_scalars(
        self,
        field_name: str,
        bad_value: float | jax.Array,
        message: str,
    ) -> None:
        """Reject non-positive or non-finite traced CAP scales."""
        inputs: dict[str, Any] = _operator_inputs()

        @jax.jit
        def compiled(value: jax.Array) -> GalerkinOperator:
            """Construct an operator with one invalid traced scalar."""
            traced_inputs: dict[str, Any] = dict(inputs)
            traced_inputs[field_name] = value
            operator: GalerkinOperator = create_galerkin_operator(
                **traced_inputs
            )
            return operator

        with pytest.raises(_RUNTIME_ERRORS, match=message):
            operator: GalerkinOperator = compiled(jnp.asarray(bad_value))
            jax.block_until_ready(operator)

    def test_operator_factory_rejects_nonhermitian_interaction_under_jit(
        self,
    ) -> None:
        """Reject a traced interaction whose reverse entry is not conjugate."""
        inputs: dict[str, Any] = _operator_inputs()

        @jax.jit
        def compiled(values: jax.Array) -> GalerkinOperator:
            """Construct an operator from traced interaction values."""
            traced_inputs: dict[str, Any] = dict(inputs)
            traced_inputs["interaction_values"] = values
            operator: GalerkinOperator = create_galerkin_operator(
                **traced_inputs
            )
            return operator

        bad_values: jax.Array = (
            inputs["interaction_values"].at[2].set(0.1 + 0.03j)
        )
        with pytest.raises(_RUNTIME_ERRORS, match="Hermitian"):
            operator: GalerkinOperator = compiled(bad_values)
            jax.block_until_ready(operator)

    def test_operator_factory_rejects_duplicate_and_out_of_range_coo(
        self,
    ) -> None:
        """Reject duplicate interaction keys and invalid factor indices."""
        duplicate: dict[str, Any] = _operator_inputs()
        duplicate["interaction_rows"] = jnp.array([0, 0], dtype=jnp.int32)
        duplicate["interaction_columns"] = jnp.array([0, 0], dtype=jnp.int32)
        duplicate["interaction_values"] = jnp.array(
            [0.5 + 0.0j, 0.5 + 0.0j],
            dtype=jnp.complex128,
        )
        with pytest.raises(_RUNTIME_ERRORS, match="unique"):
            operator: GalerkinOperator = create_galerkin_operator(**duplicate)
            jax.block_until_ready(operator)

        out_of_range: dict[str, Any] = _operator_inputs()
        out_of_range["absorber_factor_rows"] = jnp.array(
            [0, 0, 2], dtype=jnp.int32
        )
        with pytest.raises(_RUNTIME_ERRORS, match="matrix ranges"):
            operator = create_galerkin_operator(**out_of_range)
            jax.block_until_ready(operator)

    def test_solve_result_factory_preserves_algebraic_diagnostics(
        self,
    ) -> None:
        """Preserve separate algebraic, normal, and recurrence residuals."""
        inputs: dict[str, Any] = _result_inputs()
        result: GalerkinSolveResult = create_galerkin_solve_result(**inputs)
        jax.block_until_ready(result)

        chex.assert_trees_all_close(result.field, inputs["field"])
        chex.assert_trees_all_close(result.residual, inputs["residual"])
        chex.assert_trees_all_close(
            result.recurrence_residual_norm,
            inputs["recurrence_residual_norm"],
        )
        assert result.method is GalerkinSolveMethod.CGLS
        assert result.certificate_reason is (
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND
        )

    def test_solve_result_factory_normalizes_static_enums(self) -> None:
        """Normalize valid labels and reject invalid static enum labels."""
        inputs: dict[str, Any] = _result_inputs()
        inputs["method"] = "lsqr"
        inputs["certificate_reason"] = "no_stability_bound"
        result: GalerkinSolveResult = create_galerkin_solve_result(**inputs)

        assert result.method is GalerkinSolveMethod.LSQR
        assert (
            result.certificate_reason
            is GalerkinCertificateReason.NO_STABILITY_BOUND
        )
        for field, label in (
            ("method", "unknown-method"),
            ("certificate_reason", "not-a-reason"),
        ):
            invalid: dict[str, Any] = dict(inputs)
            invalid[field] = label
            with pytest.raises(ValueError):
                create_galerkin_solve_result(**invalid)

    def test_solve_result_factory_is_jittable(self) -> None:
        """Compile result validation for traced residual and status scalars."""
        inputs: dict[str, Any] = _result_inputs()

        @jax.jit
        def compiled(
            residual_norm: jax.Array,
            status: jax.Array,
            converged: jax.Array,
        ) -> GalerkinSolveResult:
            """Construct a result from traced diagnostic scalars."""
            traced_inputs: dict[str, Any] = dict(inputs)
            traced_inputs["residual_norm"] = residual_norm
            traced_inputs["status"] = status
            traced_inputs["converged"] = converged
            result: GalerkinSolveResult = create_galerkin_solve_result(
                **traced_inputs
            )
            return result

        result: GalerkinSolveResult = compiled(
            jnp.asarray(2.0e-9),
            jnp.asarray(GalerkinSolveStatus.CONVERGED),
            jnp.asarray(True),
        )
        jax.block_until_ready(result)
        chex.assert_trees_all_close(result.residual_norm, jnp.asarray(2.0e-9))
        assert bool(result.converged)

    def test_solve_result_factory_rejects_structural_errors(self) -> None:
        """Reject non-vector fields and mismatched residual shapes eagerly."""
        nonvector: dict[str, Any] = _result_inputs()
        nonvector["field"] = jnp.ones((1, 3), dtype=jnp.complex128)
        with pytest.raises(ValueError, match="field must be 1D"):
            create_galerkin_solve_result(**nonvector)

        mismatched: dict[str, Any] = _result_inputs()
        mismatched["residual"] = jnp.ones(2, dtype=jnp.complex128)
        with pytest.raises(ValueError, match="matching shapes"):
            create_galerkin_solve_result(**mismatched)

    @pytest.mark.parametrize(
        ("field_name", "bad_value", "message"),
        [
            ("residual_norm", -1.0, "residual_norm"),
            ("normal_residual_norm", jnp.nan, "normal_residual_norm"),
            ("iterations", -1, "iterations"),
            ("operator_applications", -1, "operator_applications"),
            ("status", 9, "GalerkinSolveStatus"),
        ],
    )
    def test_solve_result_factory_rejects_invalid_traced_scalars(
        self,
        field_name: str,
        bad_value: float | int | jax.Array,
        message: str,
    ) -> None:
        """Reject invalid traced scalar diagnostics with their field names."""
        inputs: dict[str, Any] = _result_inputs()

        @jax.jit
        def compiled(value: jax.Array) -> GalerkinSolveResult:
            """Construct a result with one invalid traced scalar."""
            traced_inputs: dict[str, Any] = dict(inputs)
            traced_inputs[field_name] = value
            result: GalerkinSolveResult = create_galerkin_solve_result(
                **traced_inputs
            )
            return result

        with pytest.raises(_RUNTIME_ERRORS, match=message):
            result: GalerkinSolveResult = compiled(jnp.asarray(bad_value))
            jax.block_until_ready(result)

    def test_solve_result_factory_rejects_status_flag_mismatch(self) -> None:
        """Reject a convergence flag that disagrees with solver status."""
        inputs: dict[str, Any] = _result_inputs()

        @jax.jit
        def compiled(
            status: jax.Array,
            converged: jax.Array,
        ) -> GalerkinSolveResult:
            """Construct a result from inconsistent traced status values."""
            traced_inputs: dict[str, Any] = dict(inputs)
            traced_inputs["status"] = status
            traced_inputs["converged"] = converged
            result: GalerkinSolveResult = create_galerkin_solve_result(
                **traced_inputs
            )
            return result

        with pytest.raises(_RUNTIME_ERRORS, match="agree with status"):
            result: GalerkinSolveResult = compiled(
                jnp.asarray(GalerkinSolveStatus.BREAKDOWN),
                jnp.asarray(True),
            )
            jax.block_until_ready(result)

    def test_carrier_pytrees_keep_only_declared_metadata_static(self) -> None:
        """Keep arrays dynamic while size, method, and reason remain static."""
        operator: GalerkinOperator = create_galerkin_operator(
            **_operator_inputs()
        )
        result: GalerkinSolveResult = create_galerkin_solve_result(
            **_result_inputs()
        )
        jax.block_until_ready((operator, result))

        operator_leaves: list[Any] = jax.tree_util.tree_leaves(operator)
        result_leaves: list[Any] = jax.tree_util.tree_leaves(result)
        assert len(operator_leaves) == 8
        assert len(result_leaves) == 9
        assert operator.absorber_factor_size == 2
        assert result.method is GalerkinSolveMethod.CGLS
        assert result.certificate_reason is (
            GalerkinCertificateReason.NO_OUTWARD_RESIDUAL_BOUND
        )
