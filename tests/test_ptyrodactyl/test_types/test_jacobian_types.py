"""Tests for :mod:`ptyrodactyl.types.jacobian_types`.

Extended Summary
----------------
These tests verify that the jacobian parameter and solver-state
carriers exposed from :mod:`ptyrodactyl.types` are Equinox modules with
the expected dynamic PyTree leaf behavior. The factory tests pin the
field assembly contract while checking the new two-tier validation
behavior.

Notes
-----
The scan tests cover the TC4 gate: ``GNState`` and ``CGState`` must be
valid traced carries under ``jax.lax.scan`` and ``jax.jit``.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any
from jaxtyping import Array, Float

from ptyrodactyl.types import (
    CGState,
    GNState,
    LMState,
    AberrationParams,
    ExitWaveParams,
    FisherState,
    GeometryParams,
    LanczosState,
    PositionParams,
    ProbeModeParams,
    PtychoParams,
    create_ptycho_params,
)


def _ptycho_inputs(rng_key: jax.Array) -> dict[str, Any]:
    """Build deterministic valid ptychography inputs.

    Parameters
    ----------
    rng_key : jax.Array
        Random key supplied by the shared ``rng_key`` fixture.

    Returns
    -------
    inputs : dict[str, Any]
        Keyword arguments accepted by ``create_ptycho_params``.
    """
    wave_real_key, wave_imag_key, position_key, phase_key = jax.random.split(
        rng_key,
        4,
    )
    exit_wave: Array = (
        jax.random.normal(wave_real_key, (4, 5), dtype=jnp.float64)
        + 1j * jax.random.normal(wave_imag_key, (4, 5), dtype=jnp.float64)
    )
    inputs: dict[str, Any] = {
        "exit_wave": exit_wave,
        "zernike_coeffs": jnp.array([0.0, 0.1, -0.2], dtype=jnp.float64),
        "aperture_mrad": jnp.array(20.0, dtype=jnp.float64),
        "aperture_softness": jnp.array(0.5, dtype=jnp.float64),
        "rotation_rad": jnp.array(0.05, dtype=jnp.float64),
        "center_offset": jnp.array([1.0, -2.0], dtype=jnp.float64),
        "ellipticity": jnp.array([0.01, -0.02], dtype=jnp.float64),
        "position_offsets": jax.random.normal(
            position_key,
            (6, 2),
            dtype=jnp.float64,
        ),
        "mode_weights": jnp.array([0.6, 0.4], dtype=jnp.float64),
        "mode_phases": jax.random.normal(
            phase_key,
            (2, 4, 5),
            dtype=jnp.float64,
        ),
    }
    return inputs


def _parameter_blocks(
    inputs: dict[str, Any],
) -> tuple[
    ExitWaveParams,
    AberrationParams,
    GeometryParams,
    PositionParams,
    ProbeModeParams,
]:
    """Build parameter block carriers from valid inputs.

    Parameters
    ----------
    inputs : dict[str, Any]
        Valid ptychography inputs.

    Returns
    -------
    blocks : tuple[ExitWaveParams, AberrationParams, GeometryParams, \
PositionParams, ProbeModeParams]
        Individual parameter blocks in ``PtychoParams`` field order.
    """
    exit_wave: ExitWaveParams = ExitWaveParams(inputs["exit_wave"])
    aberrations: AberrationParams = AberrationParams(
        inputs["zernike_coeffs"],
        inputs["aperture_mrad"],
        inputs["aperture_softness"],
    )
    geometry: GeometryParams = GeometryParams(
        inputs["rotation_rad"],
        inputs["center_offset"],
        inputs["ellipticity"],
    )
    positions: PositionParams = PositionParams(inputs["position_offsets"])
    probe_modes: ProbeModeParams = ProbeModeParams(
        inputs["mode_weights"],
        inputs["mode_phases"],
    )
    blocks: tuple[
        ExitWaveParams,
        AberrationParams,
        GeometryParams,
        PositionParams,
        ProbeModeParams,
    ] = (exit_wave, aberrations, geometry, positions, probe_modes)
    return blocks


def _carrier_case(
    case_name: str,
    rng_key: jax.Array,
) -> tuple[type[eqx.Module], tuple[Any, ...], dict[str, Any], int]:
    """Return constructor data for one carrier case.

    Parameters
    ----------
    case_name : str
        Name of the carrier case.
    rng_key : jax.Array
        Random key supplied by the shared ``rng_key`` fixture.

    Returns
    -------
    case : tuple[type[eqx.Module], tuple[Any, ...], dict[str, Any], int]
        Carrier class, positional arguments, keyword arguments, and
        expected dynamic leaf count.
    """
    inputs: dict[str, Any] = _ptycho_inputs(rng_key)
    blocks: tuple[
        ExitWaveParams,
        AberrationParams,
        GeometryParams,
        PositionParams,
        ProbeModeParams,
    ] = _parameter_blocks(inputs)
    vector: Float[Array, " n"] = jnp.arange(4, dtype=jnp.float64)
    matrix: Float[Array, "n n"] = jnp.eye(3, dtype=jnp.float64)
    zeros: Float[Array, " n"] = jnp.zeros(4, dtype=jnp.float64)
    alpha_beta: Float[Array, " k"] = jnp.zeros(3, dtype=jnp.float64)

    cases: dict[str, tuple[type[eqx.Module], tuple[Any, ...], dict[str, Any], int]]
    cases = {
        "exit_wave": (
            ExitWaveParams,
            (inputs["exit_wave"],),
            {"wave": inputs["exit_wave"]},
            1,
        ),
        "aberrations": (
            AberrationParams,
            (
                inputs["zernike_coeffs"],
                inputs["aperture_mrad"],
                inputs["aperture_softness"],
            ),
            {
                "zernike_coeffs": inputs["zernike_coeffs"],
                "aperture_mrad": inputs["aperture_mrad"],
                "aperture_softness": inputs["aperture_softness"],
            },
            3,
        ),
        "geometry": (
            GeometryParams,
            (
                inputs["rotation_rad"],
                inputs["center_offset"],
                inputs["ellipticity"],
            ),
            {
                "rotation_rad": inputs["rotation_rad"],
                "center_offset": inputs["center_offset"],
                "ellipticity": inputs["ellipticity"],
            },
            3,
        ),
        "positions": (
            PositionParams,
            (inputs["position_offsets"],),
            {"position_offsets": inputs["position_offsets"]},
            1,
        ),
        "probe_modes": (
            ProbeModeParams,
            (inputs["mode_weights"], inputs["mode_phases"]),
            {
                "mode_weights": inputs["mode_weights"],
                "mode_phases": inputs["mode_phases"],
            },
            2,
        ),
        "ptycho": (
            PtychoParams,
            blocks,
            {
                "exit_wave": blocks[0],
                "aberrations": blocks[1],
                "geometry": blocks[2],
                "positions": blocks[3],
                "probe_modes": blocks[4],
            },
            10,
        ),
        "fisher": (
            FisherState,
            (matrix, jnp.array(0, dtype=jnp.int32)),
            {
                "fisher_matrix": matrix,
                "iteration": jnp.array(0, dtype=jnp.int32),
            },
            2,
        ),
        "cg": (
            CGState,
            (
                zeros,
                vector,
                vector,
                jnp.array(1.0, dtype=jnp.float64),
                jnp.array(0, dtype=jnp.int32),
            ),
            {
                "x": zeros,
                "r": vector,
                "p": vector,
                "r_dot_r": jnp.array(1.0, dtype=jnp.float64),
                "iteration": jnp.array(0, dtype=jnp.int32),
            },
            5,
        ),
        "gn": (
            GNState,
            (vector, jnp.array(1.0, dtype=jnp.float64), jnp.array(0)),
            {
                "params": vector,
                "residual_norm": jnp.array(1.0, dtype=jnp.float64),
                "iteration": jnp.array(0),
            },
            3,
        ),
        "lm": (
            LMState,
            (
                vector,
                jnp.array(1.0, dtype=jnp.float64),
                jnp.array(0.1, dtype=jnp.float64),
                jnp.array(0),
            ),
            {
                "params": vector,
                "residual_norm": jnp.array(1.0, dtype=jnp.float64),
                "damping": jnp.array(0.1, dtype=jnp.float64),
                "iteration": jnp.array(0),
            },
            4,
        ),
        "lanczos": (
            LanczosState,
            (
                zeros,
                vector,
                alpha_beta,
                alpha_beta,
                jnp.array(0, dtype=jnp.int32),
            ),
            {
                "v_prev": zeros,
                "v_curr": vector,
                "alpha": alpha_beta,
                "beta": alpha_beta,
                "iteration": jnp.array(0, dtype=jnp.int32),
            },
            5,
        ),
    }
    case: tuple[type[eqx.Module], tuple[Any, ...], dict[str, Any], int] = cases[
        case_name
    ]
    return case


def _assert_carrier_round_trip(
    instance: eqx.Module,
    carrier_type: type[eqx.Module],
    expected_num_leaves: int,
) -> None:
    """Assert Equinox module and PyTree round-trip behavior.

    Parameters
    ----------
    instance : eqx.Module
        Carrier instance under test.
    carrier_type : type[eqx.Module]
        Expected carrier class.
    expected_num_leaves : int
        Expected number of dynamic PyTree leaves.
    """
    assert issubclass(carrier_type, eqx.Module)
    assert isinstance(instance, carrier_type)
    assert isinstance(instance, eqx.Module)
    leaves, treedef = jax.tree_util.tree_flatten(instance)
    assert len(leaves) == expected_num_leaves
    reconstructed: eqx.Module = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, carrier_type)
    chex.assert_trees_all_equal(instance, reconstructed)


class TestJacobianCarriers:
    """Verify jacobian carrier construction and PyTree behavior.

    :see: :mod:`ptyrodactyl.types.jacobian_types`

    Extended Summary
    ----------------
    This suite covers all eleven public carriers. Each case is
    constructed once positionally and once with keywords, then checked for
    Equinox inheritance and dynamic leaf round-tripping.
    """

    @pytest.mark.parametrize(
        "case_name",
        (
            "exit_wave",
            "aberrations",
            "geometry",
            "positions",
            "probe_modes",
            "ptycho",
            "fisher",
            "cg",
            "gn",
            "lm",
            "lanczos",
        ),
    )
    def test_constructs_keyword_positional_and_round_trips(
        self,
        case_name: str,
        rng_key: jax.Array,
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
        and compares every leaf exactly.
        """
        carrier_type, args, kwargs, expected_num_leaves = _carrier_case(
            case_name,
            rng_key,
        )
        positional: eqx.Module = carrier_type(*args)
        keyword: eqx.Module = carrier_type(**kwargs)

        _assert_carrier_round_trip(
            positional,
            carrier_type,
            expected_num_leaves,
        )
        _assert_carrier_round_trip(
            keyword,
            carrier_type,
            expected_num_leaves,
        )


class TestCreatePtychoParams:
    """Verify the relocated ptychography factory.

    :see: :func:`ptyrodactyl.types.create_ptycho_params`

    Extended Summary
    ----------------
    This suite checks valid-output field assembly and verifies that
    structural errors are raised as ``ValueError`` while data-dependent
    violations raise through ``eqx.error_if``.
    """

    def test_assembles_parameter_blocks(
        self,
        rng_key: jax.Array,
    ) -> None:
        """Compare valid factory leaves with directly built blocks.

        Extended Summary
        ----------------
        A shared input dictionary is passed to the factory and to direct
        Equinox carrier construction. The factory tree must expose leaves
        numerically equal to the expected block tree in field order.

        Notes
        -----
        The comparison intentionally checks leaves so field ordering and
        scalar conversion behavior stay pinned.
        """
        inputs: dict[str, Any] = _ptycho_inputs(rng_key)
        expected_blocks = _parameter_blocks(inputs)
        params: PtychoParams = create_ptycho_params(**inputs)

        assert isinstance(params, PtychoParams)
        assert isinstance(params, eqx.Module)
        expected_leaves = jax.tree_util.tree_leaves(expected_blocks)
        new_leaves = jax.tree_util.tree_leaves(params)
        assert len(new_leaves) == len(expected_leaves)
        chex.assert_trees_all_close(new_leaves, expected_leaves)

    def test_rejects_wrong_rank_with_value_error(
        self,
        rng_key: jax.Array,
    ) -> None:
        """Raise ValueError for static structure violations.

        Extended Summary
        ----------------
        The exit wave rank is changed from two to one, which is a
        trace-time structural violation and must be reported before any
        data-dependent checks run.

        Notes
        -----
        The factory follows the two-tier validation pattern by using a
        plain Python ``ValueError`` for this check.
        """
        inputs: dict[str, Any] = _ptycho_inputs(rng_key)
        inputs["exit_wave"] = jnp.ones((4,), dtype=jnp.complex128)

        with pytest.raises(ValueError, match="exit_wave must be 2D"):
            create_ptycho_params(**inputs)

    def test_rejects_data_violation(
        self,
        rng_key: jax.Array,
    ) -> None:
        """Raise for data-dependent invalid mode weights.

        Extended Summary
        ----------------
        The mode weights keep the correct shape but include a negative
        entry. The violation depends on array data, so it is checked with
        ``eqx.error_if``.

        Notes
        -----
        ``block_until_ready`` forces any deferred Equinox runtime error to
        materialize before the assertion exits.
        """
        inputs: dict[str, Any] = _ptycho_inputs(rng_key)
        inputs["mode_weights"] = jnp.array([1.2, -0.2], dtype=jnp.float64)

        with pytest.raises(Exception, match="mode_weights"):
            params: PtychoParams = create_ptycho_params(**inputs)
            jax.block_until_ready(params.probe_modes.mode_weights)


class TestJacobianStateScans:
    """Verify solver states as traced scan carries.

    :see: :class:`ptyrodactyl.types.GNState`
    :see: :class:`ptyrodactyl.types.CGState`

    Extended Summary
    ----------------
    These tests pin the TC4 behavior that ``GNState`` and ``CGState`` are
    valid dynamic PyTree carries for ``jax.lax.scan`` under ``jax.jit``.
    """

    def test_gn_state_scan_traces_and_runs(self) -> None:
        """Trace and execute a scan carrying GNState.

        Extended Summary
        ----------------
        The scan increments the parameter vector, residual norm, and
        iteration counter for three steps under JIT compilation.

        Notes
        -----
        The final state and residual history are checked for shape and
        expected scalar values.
        """

        def run_scan(
            initial_state: GNState,
        ) -> tuple[GNState, Float[Array, " steps"]]:
            """Run a fixed-length GNState scan."""

            def step(
                state: GNState,
                _: None,
            ) -> tuple[GNState, Float[Array, ""]]:
                """Advance one GNState scan step."""
                new_state: GNState = GNState(
                    params=state.params + 1.0,
                    residual_norm=state.residual_norm + 0.5,
                    iteration=state.iteration + 1,
                )
                return new_state, new_state.residual_norm

            final_state, history = jax.lax.scan(
                step,
                initial_state,
                None,
                length=3,
            )
            return final_state, history

        initial_state: GNState = GNState(
            params=jnp.zeros(4, dtype=jnp.float64),
            residual_norm=jnp.array(1.0, dtype=jnp.float64),
            iteration=jnp.array(0, dtype=jnp.int32),
        )
        final_state, history = jax.jit(run_scan)(initial_state)

        assert isinstance(final_state, GNState)
        chex.assert_trees_all_close(
            final_state.params,
            jnp.full(4, 3.0, dtype=jnp.float64),
        )
        chex.assert_trees_all_close(
            final_state.residual_norm,
            jnp.array(2.5, dtype=jnp.float64),
        )
        chex.assert_trees_all_equal(final_state.iteration, jnp.array(3))
        chex.assert_shape(history, (3,))

    def test_cg_state_scan_traces_and_runs(self) -> None:
        """Trace and execute a scan carrying CGState.

        Extended Summary
        ----------------
        The scan updates conjugate-gradient-like vector leaves and the
        traced iteration scalar for three steps under JIT compilation.

        Notes
        -----
        This test exercises a solver state with multiple PyTree vector
        fields to catch static-field regressions.
        """

        def run_scan(initial_state: CGState) -> tuple[CGState, Float[Array, " steps"]]:
            """Run a fixed-length CGState scan."""

            def step(
                state: CGState,
                _: None,
            ) -> tuple[CGState, Float[Array, ""]]:
                """Advance one CGState scan step."""
                new_r: Float[Array, " n"] = state.r * 0.5
                new_p: Float[Array, " n"] = state.p + new_r
                new_state: CGState = CGState(
                    x=state.x + state.p,
                    r=new_r,
                    p=new_p,
                    r_dot_r=jnp.sum(new_r**2),
                    iteration=state.iteration + 1,
                )
                return new_state, new_state.r_dot_r

            final_state, history = jax.lax.scan(
                step,
                initial_state,
                None,
                length=3,
            )
            return final_state, history

        initial_vector: Float[Array, " n"] = jnp.ones(4, dtype=jnp.float64)
        initial_state: CGState = CGState(
            x=jnp.zeros(4, dtype=jnp.float64),
            r=initial_vector,
            p=initial_vector,
            r_dot_r=jnp.array(4.0, dtype=jnp.float64),
            iteration=jnp.array(0, dtype=jnp.int32),
        )
        final_state, history = jax.jit(run_scan)(initial_state)

        assert isinstance(final_state, CGState)
        chex.assert_shape(final_state.x, (4,))
        chex.assert_shape(final_state.r, (4,))
        chex.assert_shape(final_state.p, (4,))
        chex.assert_trees_all_equal(final_state.iteration, jnp.array(3))
        chex.assert_tree_all_finite(final_state)
        chex.assert_shape(history, (3,))
