"""Tests for :mod:`ptyrodactyl.types.distributions`.

Extended Summary
----------------
These tests verify the generic distribution skeleton used for later
ensemble reducers. They pin structural validation, traced probability
weight checks, differentiability of probability normalization, constant
exports, and Equinox PyTree round-tripping.

Notes
-----
The gradient test is the TC5 gate: normalization must use
``jax.lax.cond`` rather than data-dependent Python control flow so
``jax.grad`` can pass through positive weights.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ptyrodactyl.types import (
    TRIVIAL,
    TRIVIAL_DISTRIBUTION,
    Distribution,
    ReductionMode,
    create_distribution,
    create_trivial_distribution,
)
from ptyrodactyl.types.distributions import _normalize_probability_weights


class TestDistributionFactories:
    """Verify distribution factories and constants.

    :see: :mod:`ptyrodactyl.types.distributions`

    Extended Summary
    ----------------
    This suite checks that distributions validate array structure with
    ``ValueError``, validate weight data with ``eqx.error_if``, normalize
    differentiably, and preserve static metadata across PyTree
    flatten/unflatten.
    """

    def test_create_distribution_rejects_nonmatrix_samples(self) -> None:
        """Raise ValueError when samples are not two-dimensional.

        Extended Summary
        ----------------
        Sample rank is a static structural property, so the factory must
        raise ``ValueError`` before any traced data-dependent checks run.
        """
        with pytest.raises(ValueError, match="samples must have shape"):
            create_distribution(
                samples=jnp.ones((3,), dtype=jnp.float64),
                weights=jnp.ones((3,), dtype=jnp.float64),
            )

    def test_create_distribution_rejects_length_mismatch(self) -> None:
        """Raise ValueError when samples and weights differ in length.

        Extended Summary
        ----------------
        The leading sample dimension and weight vector length must match
        before probability normalization can be defined.
        """
        with pytest.raises(ValueError, match="share leading dimension"):
            create_distribution(
                samples=jnp.ones((3, 2), dtype=jnp.float64),
                weights=jnp.ones((2,), dtype=jnp.float64),
            )

    def test_create_distribution_rejects_negative_weights(self) -> None:
        """Raise through eqx.error_if for negative weights.

        Extended Summary
        ----------------
        Negative weights keep the correct vector shape but violate the
        probability simplex contract. The check is data-dependent and
        must use ``eqx.error_if``.

        Notes
        -----
        ``block_until_ready`` materializes any deferred Equinox runtime
        error before leaving the assertion context.
        """
        with pytest.raises(Exception, match="weights"):
            distribution: Distribution = create_distribution(
                samples=jnp.ones((3, 2), dtype=jnp.float64),
                weights=jnp.array([0.5, -0.25, 0.75], dtype=jnp.float64),
            )
            jax.block_until_ready(distribution.weights)

    def test_normalize_probability_weights_has_finite_gradient(self) -> None:
        """Differentiate through probability weight normalization.

        Extended Summary
        ----------------
        Positive weights flow through the normalizer and into a scalar
        objective. The resulting gradient must be finite, verifying the
        differentiability gate for the ``jax.lax.cond`` implementation.
        """

        def objective(weights: jax.Array) -> jax.Array:
            """Return a scalar objective of normalized weights."""
            normalized: jax.Array = _normalize_probability_weights(weights)
            value: jax.Array = jnp.sum(normalized**2)
            return value

        weights: jax.Array = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float64)
        gradient: jax.Array = jax.grad(objective)(weights)

        chex.assert_shape(gradient, (3,))
        chex.assert_tree_all_finite(gradient)

    def test_trivial_constants_exist(self) -> None:
        """Check the trivial distribution constants.

        Extended Summary
        ----------------
        The package exposes both the long-form constant and short alias.
        They must point to the same one-sample identity distribution.
        """
        assert TRIVIAL is TRIVIAL_DISTRIBUTION
        assert isinstance(TRIVIAL, Distribution)
        assert isinstance(TRIVIAL, eqx.Module)
        assert TRIVIAL.reduction is ReductionMode.INCOHERENT
        chex.assert_shape(TRIVIAL.samples, (1, 1))
        chex.assert_shape(TRIVIAL.weights, (1,))
        chex.assert_trees_all_close(
            TRIVIAL.weights,
            jnp.ones((1,), dtype=jnp.float64),
        )

    def test_distribution_round_trips_tree_flatten_unflatten(self) -> None:
        """Round-trip Distribution through JAX tree utilities.

        Extended Summary
        ----------------
        ``samples`` and ``weights`` are dynamic leaves while ``reduction``
        and ``axis_id`` are static metadata preserved by the treedef.
        """
        distribution: Distribution = create_distribution(
            samples=jnp.arange(6, dtype=jnp.float64).reshape(3, 2),
            weights=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
            reduction=ReductionMode.COHERENT,
            axis_id="source",
        )
        leaves, treedef = jax.tree_util.tree_flatten(distribution)
        reconstructed: Distribution = jax.tree_util.tree_unflatten(
            treedef,
            leaves,
        )

        assert len(leaves) == 2
        assert isinstance(reconstructed, Distribution)
        assert reconstructed.reduction is ReductionMode.COHERENT
        assert reconstructed.axis_id == "source"
        chex.assert_trees_all_close(distribution, reconstructed)

    def test_create_trivial_distribution_validates_sample_dim(self) -> None:
        """Raise ValueError for non-positive trivial sample width.

        Extended Summary
        ----------------
        The identity distribution needs at least one latent coordinate
        column, so non-positive widths are rejected structurally.
        """
        with pytest.raises(ValueError, match="sample_dim must be positive"):
            create_trivial_distribution(sample_dim=0)
