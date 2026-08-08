"""Tests for :mod:`ptyrodactyl.types.born_potential_types`.

Extended Summary
----------------
These tests verify exact support inclusions, independent support ownership,
and endpoint-safe quotient validation for fixed scalar Galerkin products.
"""

import ast
import inspect
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from ptyrodactyl.types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _line_indices(values: tuple[int, ...]) -> jax.Array:
    """Place one-dimensional exact indices on the final work-grid axis."""
    indices: jax.Array = jnp.asarray(
        [[0, 0, value] for value in values],
        dtype=jnp.int32,
    )
    return indices


def _create_support() -> GalerkinProductSupport:
    """Create one endpoint-safe support with four distinct finite sets."""
    support: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=_line_indices((-1, 0, 1)),
        interaction_indices=_line_indices((-1, 0, 1)),
        absorber_indices=_line_indices((-2, -1, 0, 1, 2)),
        work_indices=_line_indices((-3, -2, -1, 0, 1, 2, 3)),
        work_shape=(1, 1, 7),
    )
    return support


class TestGalerkinProductSupport:
    """Verify fixed Galerkin support structure and exact set predicates.

    :see: :class:`ptyrodactyl.types.GalerkinProductSupport`
    :see: :func:`ptyrodactyl.types.create_galerkin_product_support`
    """

    def test_factory_keeps_four_independent_supports(self) -> None:
        """Retain each exact support and the odd endpoint-safe work grid."""
        support: GalerkinProductSupport = _create_support()

        assert support.state_indices.shape == (3, 3)
        assert support.interaction_indices.shape == (3, 3)
        assert support.absorber_indices.shape == (5, 3)
        assert support.work_indices.shape == (7, 3)
        assert support.work_shape == (1, 1, 7)
        assert support.state_indices is not support.interaction_indices
        assert support.absorber_indices is not support.work_indices

    def test_factory_is_jit_compatible_for_frozen_structure(self) -> None:
        """Compile support checks with the work shape fixed statically."""
        interaction: jax.Array = _line_indices((-1, 0, 1))
        absorber: jax.Array = _line_indices((-2, -1, 0, 1, 2))
        work: jax.Array = _line_indices((-3, -2, -1, 0, 1, 2, 3))

        @jax.jit
        def build(state: jax.Array) -> GalerkinProductSupport:
            return create_galerkin_product_support(
                state,
                interaction,
                absorber,
                work,
                (1, 1, 7),
            )

        support: GalerkinProductSupport = build(_line_indices((-1, 0, 1)))
        jax.block_until_ready(support.state_indices)
        assert support.work_shape == (1, 1, 7)

    def test_support_validation_does_not_materialize_pair_grids(self) -> None:
        """Keep exact support checks free of rank-three pair tensors."""
        module = sys.modules[create_galerkin_product_support.__module__]
        tree = ast.parse(inspect.getsource(module))
        guarded_functions = {
            "_all_binary_products_are_members",
            "_restricted_product_has_no_alias",
            "create_galerkin_product_support",
        }
        offenders: list[tuple[str, str]] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in guarded_functions:
                continue
            for assignment in ast.walk(node):
                if not isinstance(assignment, ast.AnnAssign):
                    continue
                annotation = ast.unparse(assignment.annotation)
                strings = [
                    value.value
                    for value in ast.walk(assignment.annotation)
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ]
                if any(len(value.split()) >= 3 for value in strings):
                    offenders.append((node.name, annotation))

        assert offenders == []

    def test_even_nyquist_endpoints_are_rejected(self) -> None:
        """Reject the collision of signed endpoints -3 and +3 modulo six."""
        with pytest.raises(_RUNTIME_ERRORS, match="quotient|no-alias"):
            support: GalerkinProductSupport = create_galerkin_product_support(
                state_indices=_line_indices((-1, 0, 1)),
                interaction_indices=_line_indices((-1, 0, 1)),
                absorber_indices=_line_indices((-2, -1, 0, 1, 2)),
                work_indices=_line_indices((-3, -2, -1, 0, 1, 2, 3)),
                work_shape=(1, 1, 6),
            )
            jax.block_until_ready(support.work_indices)

    def test_narrow_integer_negation_cannot_fake_sign_symmetry(self) -> None:
        """Canonicalize before negating the signed-dtype minimum."""
        invalid_interaction = jnp.array([[0, 0, -128]], dtype=jnp.int8)
        with pytest.raises(_RUNTIME_ERRORS, match="sign-symmetric"):
            support = create_galerkin_product_support(
                state_indices=_line_indices((-1, 0, 1)),
                interaction_indices=invalid_interaction,
                absorber_indices=_line_indices((-2, -1, 0, 1, 2)),
                work_indices=_line_indices((-129, -128, -127, -1, 0, 1)),
                work_shape=(1, 1, 263),
            )
            jax.block_until_ready(support.interaction_indices)

    @pytest.mark.parametrize(
        ("replacement", "message"),
        [
            ("work", "contain both product sets"),
            ("absorber", "contain K_u-K_u"),
            ("interaction", "sign-symmetric"),
        ],
    )
    def test_exact_support_contract_rejects_missing_sets(
        self,
        replacement: str,
        message: str,
    ) -> None:
        """Reject a missing sum, difference, or interaction inverse index."""
        state: jax.Array = _line_indices((-1, 0, 1))
        interaction: jax.Array = _line_indices((-1, 0, 1))
        absorber: jax.Array = _line_indices((-2, -1, 0, 1, 2))
        work: jax.Array = _line_indices((-3, -2, -1, 0, 1, 2, 3))
        if replacement == "work":
            work = _line_indices((-2, -1, 0, 1, 2, 3))
        elif replacement == "absorber":
            absorber = _line_indices((-1, 0, 1))
        else:
            interaction = _line_indices((-1, 0))

        with pytest.raises(_RUNTIME_ERRORS, match=message):
            support: GalerkinProductSupport = create_galerkin_product_support(
                state,
                interaction,
                absorber,
                work,
                (1, 1, 7),
            )
            jax.block_until_ready(support.state_indices)

    @pytest.mark.parametrize(
        "work_shape",
        [(1, 5), (1, True, 5), (1, 0, 5)],
    )
    def test_factory_rejects_invalid_static_work_shape(
        self,
        work_shape: tuple[int, ...],
    ) -> None:
        """Reject a wrong-rank, Boolean, or nonpositive work-grid shape."""
        with pytest.raises((ValueError, TypeCheckError), match="work_shape"):
            create_galerkin_product_support(
                _line_indices((-1, 0, 1)),
                _line_indices((-1, 0, 1)),
                _line_indices((-2, -1, 0, 1, 2)),
                _line_indices((-3, -2, -1, 0, 1, 2, 3)),
                work_shape,
            )

    def test_factory_rejects_wrong_index_shape(self) -> None:
        """Reject supports that do not contain three-component indices."""
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            create_galerkin_product_support(
                jnp.zeros((3, 2), dtype=jnp.int32),
                _line_indices((-1, 0, 1)),
                _line_indices((-2, -1, 0, 1, 2)),
                _line_indices((-3, -2, -1, 0, 1, 2, 3)),
                (1, 1, 7),
            )
