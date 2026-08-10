"""Tests for :mod:`ptyrodactyl.types.recon_types`.

Extended Summary
----------------
These tests verify that reconstruction skeleton carriers exposed from
:mod:`ptyrodactyl.types` are Equinox modules with the expected static
metadata behavior, and that their factories enforce the two-tier
validation contract without implementing solver logic.

Notes
-----
The carrier tests construct each module both positionally and by keyword.
Factory tests distinguish structural ``ValueError`` checks from traced
``eqx.error_if`` checks that are materialized with ``block_until_ready``.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict, Tuple

from ptyrodactyl.types import (
    LaplaceUncertainty,
    PosteriorSamples,
    ReconProblem,
    ReconResult,
    create_laplace_uncertainty,
    create_posterior_samples,
    create_recon_problem,
    create_recon_result,
)


def _forward(params: jax.Array) -> jax.Array:
    """Return a placeholder forward-model output."""
    output: jax.Array = params
    return output


def _identity(params: jax.Array) -> jax.Array:
    """Return a placeholder identity transform."""
    output: jax.Array = params
    return output


def _residual(simulated: jax.Array, measured: jax.Array) -> jax.Array:
    """Return a placeholder residual."""
    output: jax.Array = simulated - measured
    return output


def _loss(residual: jax.Array) -> jax.Array:
    """Return a placeholder scalar loss."""
    output: jax.Array = jnp.sum(residual**2)
    return output


def _carrier_case(
    case_name: str,
) -> Tuple[type[eqx.Module], Tuple[Any, ...], Dict[str, Any], int]:
    """Return constructor data for one reconstruction carrier.

    Parameters
    ----------
    case_name : str
        Name of the carrier case.

    Returns
    -------
    case : Tuple[type[eqx.Module], Tuple[Any, ...], Dict[str, Any], int]
        Carrier class, positional arguments, keyword arguments, and
        expected dynamic leaf count.
    """
    measured: jax.Array = jnp.ones((2, 3), dtype=jnp.float64)
    vector: jax.Array = jnp.array([1.0, 2.0], dtype=jnp.float64)
    matrix: jax.Array = jnp.eye(2, dtype=jnp.float64)
    simulated: jax.Array = jnp.full((2, 3), 2.0, dtype=jnp.float64)
    residual: jax.Array = simulated - measured
    scalar_loss: jax.Array = jnp.array(6.0, dtype=jnp.float64)
    iterations: jax.Array = jnp.array(3, dtype=jnp.int32)
    converged: jax.Array = jnp.array(True, dtype=jnp.bool_)
    samples: jax.Array = jnp.arange(6, dtype=jnp.float64).reshape(3, 2)

    cases: Dict[
        str, Tuple[type[eqx.Module], Tuple[Any, ...], Dict[str, Any], int]
    ]
    cases = {
        "problem": (
            ReconProblem,
            (
                _forward,
                measured,
                _identity,
                _residual,
                _loss,
                "galerkin",
                "diffraction",
            ),
            {
                "forward": _forward,
                "measured": measured,
                "transform": _identity,
                "residual_fn": _residual,
                "loss_fn": _loss,
                "forward_family": "galerkin",
                "terminal": "diffraction",
            },
            1,
        ),
        "result": (
            ReconResult,
            (
                vector,
                vector + 1.0,
                simulated,
                residual,
                scalar_loss,
                iterations,
                converged,
                "success",
            ),
            {
                "params": vector,
                "latent_params": vector + 1.0,
                "simulated": simulated,
                "residual": residual,
                "loss": scalar_loss,
                "iterations": iterations,
                "converged": converged,
                "solver_status": "success",
            },
            7,
        ),
        "laplace": (
            LaplaceUncertainty,
            (
                matrix,
                matrix * 0.5,
                vector,
                matrix,
            ),
            {
                "fisher_information": matrix,
                "covariance": matrix * 0.5,
                "standard_deviation": vector,
                "correlation": matrix,
            },
            4,
        ),
        "posterior": (
            PosteriorSamples,
            (
                samples,
                vector,
                matrix,
                vector + 1.0,
                vector * 10.0,
                converged,
            ),
            {
                "samples": samples,
                "mean": vector,
                "covariance": matrix,
                "rhat": vector + 1.0,
                "ess": vector * 10.0,
                "converged": converged,
            },
            6,
        ),
    }
    case: Tuple[type[eqx.Module], Tuple[Any, ...], Dict[str, Any], int] = (
        cases[case_name]
    )
    return case


def _assert_static_fields_excluded(instance: eqx.Module) -> None:
    """Assert reconstruction static fields are absent from PyTree leaves.

    Parameters
    ----------
    instance : eqx.Module
        Carrier instance under test.
    """
    leaves: list[Any] = jax.tree_util.tree_leaves(instance)
    assert _forward not in leaves
    assert _identity not in leaves
    assert _residual not in leaves
    assert _loss not in leaves
    assert "galerkin" not in leaves
    assert "diffraction" not in leaves
    assert "success" not in leaves


class TestReconCarriers:
    """Verify reconstruction carrier construction and PyTree behavior.


    Extended Summary
    ----------------
    This suite covers all four public reconstruction carriers. Each case
    is constructed once positionally and once with keywords, then checked
    for Equinox inheritance, static-field exclusion, and dynamic leaf
    round-tripping.

    :see: :class:`ptyrodactyl.types.LaplaceUncertainty`
    :see: :class:`ptyrodactyl.types.PosteriorSamples`
    :see: :class:`ptyrodactyl.types.ReconProblem`
    :see: :class:`ptyrodactyl.types.ReconResult`
    """

    @pytest.mark.parametrize(
        "case_name",
        ("problem", "result", "laplace", "posterior"),
    )
    def test_constructs_keyword_positional_and_round_trips(
        self,
        case_name: str,
    ) -> None:
        """Check construction and tree round-tripping for one carrier.

        Extended Summary
        ----------------
        The selected carrier is instantiated through both supported
        constructor forms and verified as an Equinox module with the
        expected number of dynamic leaves.

        Notes
        -----
        The helper reconstructs the carrier from ``tree_flatten`` output
        and compares every dynamic leaf exactly.
        """
        carrier_type, args, kwargs, expected_num_leaves = _carrier_case(
            case_name,
        )
        positional: eqx.Module = carrier_type(*args)
        keyword: eqx.Module = carrier_type(**kwargs)

        for instance in (positional, keyword):
            assert issubclass(carrier_type, eqx.Module)
            assert isinstance(instance, carrier_type)
            assert isinstance(instance, eqx.Module)
            leaves, treedef = jax.tree_util.tree_flatten(instance)
            assert len(leaves) == expected_num_leaves
            reconstructed: eqx.Module = jax.tree_util.tree_unflatten(
                treedef,
                leaves,
            )
            assert isinstance(reconstructed, carrier_type)
            chex.assert_trees_all_equal(instance, reconstructed)
            _assert_static_fields_excluded(instance)


class TestCreateReconFactories:
    """Verify reconstruction factory validation.


    Extended Summary
    ----------------
    This suite covers structural failures reported as ``ValueError`` and
    representative data-dependent failures reported through
    ``eqx.error_if``.

    :see: :func:`ptyrodactyl.types.create_laplace_uncertainty`
    :see: :func:`ptyrodactyl.types.create_posterior_samples`
    :see: :func:`ptyrodactyl.types.create_recon_problem`
    :see: :func:`ptyrodactyl.types.create_recon_result`
    """

    def test_create_recon_problem_validates_callable(self) -> None:
        """Raise ValueError for non-callable hooks.

        Extended Summary
        ----------------
        The forward hook is static metadata and can be checked before any
        traced array checks run.

        Notes
        -----
        The factory reports this structural violation with ``ValueError``.
        """
        with pytest.raises(ValueError, match="forward must be callable"):
            create_recon_problem(
                forward=object(),
                measured=jnp.ones((2, 2), dtype=jnp.float64),
            )

    def test_create_recon_problem_rejects_nonfinite_measured(self) -> None:
        """Raise for non-finite measured data.

        Extended Summary
        ----------------
        The measured data keeps a valid structure but contains ``inf``,
        so the data-dependent check must be attached through
        ``eqx.error_if``.

        Notes
        -----
        ``block_until_ready`` materializes the Equinox runtime error.
        """
        with pytest.raises(Exception, match="measured"):
            problem: ReconProblem = create_recon_problem(
                forward=_forward,
                measured=jnp.array([jnp.inf], dtype=jnp.float64),
            )
            jax.block_until_ready(problem.measured)

    def test_create_recon_result_validates_shape_consistency(self) -> None:
        """Raise ValueError for simulated/residual shape mismatch.

        Extended Summary
        ----------------
        The simulated and residual terminal arrays must describe the same
        measurement surface. Mismatched shapes are structural and checked
        before traced data checks.
        """
        with pytest.raises(ValueError, match="matching shapes"):
            create_recon_result(
                params=jnp.ones(2, dtype=jnp.float64),
                latent_params=jnp.ones(2, dtype=jnp.float64),
                simulated=jnp.ones((2, 2), dtype=jnp.float64),
                residual=jnp.ones((2,), dtype=jnp.float64),
                loss=jnp.array(1.0, dtype=jnp.float64),
                iterations=jnp.array(0, dtype=jnp.int32),
                converged=jnp.array(False, dtype=jnp.bool_),
                solver_status="failed",
            )

    def test_create_recon_result_rejects_negative_iterations(self) -> None:
        """Raise for data-dependent invalid iteration counts.

        Extended Summary
        ----------------
        A negative iteration count has valid scalar structure but invalid
        data. The factory attaches this check with ``eqx.error_if``.
        """
        with pytest.raises(Exception, match="iterations"):
            result: ReconResult = create_recon_result(
                params=jnp.ones(2, dtype=jnp.float64),
                latent_params=jnp.ones(2, dtype=jnp.float64),
                simulated=jnp.ones((2,), dtype=jnp.float64),
                residual=jnp.zeros((2,), dtype=jnp.float64),
                loss=jnp.array(1.0, dtype=jnp.float64),
                iterations=jnp.array(-1, dtype=jnp.int32),
                converged=jnp.array(False, dtype=jnp.bool_),
                solver_status="failed",
            )
            jax.block_until_ready(result.iterations)

    def test_create_laplace_uncertainty_validates_square_covariance(
        self,
    ) -> None:
        """Raise ValueError for a non-square covariance matrix.

        Extended Summary
        ----------------
        Covariance matrix shape is statically available after array
        coercion and must be square before the carrier is constructed.
        """
        with pytest.raises(ValueError, match="covariance must be square"):
            create_laplace_uncertainty(
                fisher_information=jnp.eye(2, dtype=jnp.float64),
                covariance=jnp.ones((2, 3), dtype=jnp.float64),
                standard_deviation=jnp.ones(2, dtype=jnp.float64),
                correlation=jnp.eye(2, dtype=jnp.float64),
            )

    def test_create_laplace_uncertainty_rejects_nonfinite_values(
        self,
    ) -> None:
        """Raise for non-finite Laplace uncertainty arrays.

        Extended Summary
        ----------------
        The covariance structure is valid but contains ``nan``, so the
        invalid data is checked through ``eqx.error_if``.
        """
        covariance: jax.Array = jnp.array(
            [[1.0, jnp.nan], [jnp.nan, 1.0]],
            dtype=jnp.float64,
        )
        with pytest.raises(Exception, match="covariance"):
            uncertainty: LaplaceUncertainty = create_laplace_uncertainty(
                fisher_information=jnp.eye(2, dtype=jnp.float64),
                covariance=covariance,
                standard_deviation=jnp.ones(2, dtype=jnp.float64),
                correlation=jnp.eye(2, dtype=jnp.float64),
            )
            jax.block_until_ready(uncertainty.covariance)

    def test_create_posterior_samples_validates_shapes(self) -> None:
        """Raise ValueError for posterior diagnostic shape mismatch.

        Extended Summary
        ----------------
        The posterior mean length must match the parameter dimension
        carried by ``samples``.
        """
        with pytest.raises(ValueError, match="mean must have shape"):
            create_posterior_samples(
                samples=jnp.ones((3, 2), dtype=jnp.float64),
                mean=jnp.ones(3, dtype=jnp.float64),
                covariance=jnp.eye(2, dtype=jnp.float64),
                rhat=jnp.ones(2, dtype=jnp.float64),
                ess=jnp.ones(2, dtype=jnp.float64),
                converged=jnp.array(True, dtype=jnp.bool_),
            )

    def test_create_posterior_samples_rejects_nonfinite_diagnostics(
        self,
    ) -> None:
        """Raise for non-finite posterior diagnostics.

        Extended Summary
        ----------------
        The ESS vector has valid shape but contains ``inf``, so the
        invalid data is checked through ``eqx.error_if``.
        """
        with pytest.raises(Exception, match="ess"):
            posterior: PosteriorSamples = create_posterior_samples(
                samples=jnp.ones((3, 2), dtype=jnp.float64),
                mean=jnp.ones(2, dtype=jnp.float64),
                covariance=jnp.eye(2, dtype=jnp.float64),
                rhat=jnp.ones(2, dtype=jnp.float64),
                ess=jnp.array([1.0, jnp.inf], dtype=jnp.float64),
                converged=jnp.array(True, dtype=jnp.bool_),
            )
            jax.block_until_ready(posterior.ess)
