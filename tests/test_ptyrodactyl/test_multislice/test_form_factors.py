"""Test independent form-factor physics and parameterization agreement.

:see: :func:`ptyrodactyl.multislice.atomic_form_factor`
:see: :func:`ptyrodactyl.multislice.lobato_bandlimited_peak`
:see: :func:`ptyrodactyl.multislice.projected_atom_potential`
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import EquinoxRuntimeError

from ptyrodactyl.inout import kirkland_potentials, lobato_potentials
from ptyrodactyl.multislice import (
    atomic_form_factor,
    kirkland_form_factor,
    kirkland_projected_potential,
    lobato_bandlimited_peak,
    lobato_form_factor,
    lobato_projected_potential,
    projected_atom_potential,
)
from ptyrodactyl.types import (
    A_BOHR,
    create_kirkland_parameters,
    create_lobato_parameters,
)

_FORM_FACTOR_RELATIVE_BOUND: float = 6e-3
_PROJECTED_RELATIVE_BOUND: float = 3e-2
_TRACED_ERROR_TYPES: tuple[type[Exception], ...] = (
    EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def test_hydrogen_zero_angle_form_factor_is_bohr_radius() -> None:
    """The neutral-H Bethe anchor is f_H(0) = a_0, not Z."""
    actual = atomic_form_factor(1, jnp.asarray(0.0, dtype=jnp.float64))
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(A_BOHR),
        rtol=1e-9,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("atomic_number", "g_max", "expected_volts"),
    [
        (6, 2.0, 417.0),
        (6, 3.0, 712.0),
        (6, 4.0, 1028.0),
        (14, 2.0, 823.0),
        (14, 3.0, 1513.0),
        (14, 4.0, 2238.0),
        (79, 2.0, 3327.0),
        (79, 3.0, 6534.0),
        (79, 4.0, 10125.0),
    ],
)
def test_bandlimited_peaks_match_published_table_p(
    atomic_number: int,
    g_max: float,
    expected_volts: float,
) -> None:
    """C, Si, and Au reproduce the independently verified rounded values."""
    actual = lobato_bandlimited_peak(atomic_number, g_max)
    np.testing.assert_allclose(
        np.asarray(actual),
        expected_volts,
        rtol=0.0,
        atol=0.51,
    )


@pytest.mark.parametrize("atomic_number", [6, 14, 79])
def test_dispatchers_default_to_explicit_lobato(atomic_number: int) -> None:
    """Omitting parameterization is exactly the explicit Lobato path."""
    q = 2.0 * jnp.pi * jnp.linspace(0.0, 4.0, 33, dtype=jnp.float64)
    r = jnp.geomspace(0.05, 1.5, 33)

    default_form_factor = atomic_form_factor(atomic_number, q)
    explicit_form_factor = atomic_form_factor(
        atomic_number,
        q,
        parameterization="lobato",
    )
    default_projected = projected_atom_potential(atomic_number, r)
    explicit_projected = projected_atom_potential(
        atomic_number,
        r,
        parameterization="lobato",
    )

    np.testing.assert_array_equal(
        np.asarray(default_form_factor),
        np.asarray(explicit_form_factor),
    )
    np.testing.assert_array_equal(
        np.asarray(default_projected),
        np.asarray(explicit_projected),
    )


def test_dispatchers_jit_with_traced_coordinates_and_atomic_number() -> None:
    """Both public dispatchers execute with traced numerical arguments."""
    q = 2.0 * jnp.pi * jnp.linspace(0.0, 4.0, 17, dtype=jnp.float64)
    r = jnp.geomspace(0.05, 1.5, 17)

    compiled_form_factor = jax.jit(
        lambda atom_no, values: atomic_form_factor(atom_no, values)
    )
    compiled_projected = jax.jit(
        lambda atom_no, values: projected_atom_potential(atom_no, values)
    )
    form_factor = compiled_form_factor(jnp.asarray(14), q)
    projected = compiled_projected(jnp.asarray(14), r)
    jax.block_until_ready((form_factor, projected))

    assert form_factor.shape == q.shape
    assert projected.shape == r.shape
    assert form_factor.dtype == projected.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(form_factor)))
    assert bool(jnp.all(jnp.isfinite(projected)))


@pytest.mark.parametrize(
    "function, coordinate",
    [
        (atomic_form_factor, jnp.asarray(1.0, dtype=jnp.float64)),
        (projected_atom_potential, jnp.asarray(0.25, dtype=jnp.float64)),
    ],
)
@pytest.mark.parametrize("parameterization", ["", "Kirkland", "unknown"])
def test_dispatchers_reject_unknown_parameterizations(
    function: Callable[..., jax.Array],
    coordinate: jax.Array,
    parameterization: str,
) -> None:
    """The public selection surface is the closed Lobato/Kirkland pair."""
    with pytest.raises(ValueError, match="parameterization"):
        function(14, coordinate, parameterization=parameterization)


@pytest.mark.parametrize(
    "function, coordinate",
    [
        (atomic_form_factor, jnp.asarray(1.0, dtype=jnp.float64)),
        (projected_atom_potential, jnp.asarray(0.25, dtype=jnp.float64)),
    ],
)
@pytest.mark.parametrize("atomic_number", [0, 104])
def test_dispatchers_reject_invalid_python_atomic_numbers(
    function: Callable[..., jax.Array],
    coordinate: jax.Array,
    atomic_number: int,
) -> None:
    """Python atomic numbers outside the complete H--Lr table fail closed."""
    with pytest.raises(ValueError, match="between 1 and 103"):
        function(atomic_number, coordinate)


@pytest.mark.parametrize(
    "function, coordinate",
    [
        (atomic_form_factor, jnp.asarray(1.0, dtype=jnp.float64)),
        (projected_atom_potential, jnp.asarray(0.25, dtype=jnp.float64)),
        (lobato_bandlimited_peak, 1.0),
    ],
)
@pytest.mark.parametrize("atomic_number", [False, True])
def test_public_form_factor_apis_reject_python_bool_atomic_numbers(
    function: Callable[..., jax.Array],
    coordinate: jax.Array | float,
    atomic_number: bool,
) -> None:
    """Python booleans cannot masquerade as atomic numbers zero and one."""
    with pytest.raises(ValueError, match="atom_no"):
        function(atomic_number, coordinate)


@pytest.mark.parametrize("atomic_number", [0, 104])
def test_traced_atomic_number_validation_executes(atomic_number: int) -> None:
    """JIT tracing cannot bypass the atomic-number range check."""
    compiled = jax.jit(
        lambda atom_no: atomic_form_factor(
            atom_no,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
    )

    with pytest.raises(_TRACED_ERROR_TYPES, match="atom_no"):
        result = compiled(jnp.asarray(atomic_number, dtype=jnp.int32))
        jax.block_until_ready(result)


@pytest.mark.parametrize("g_max", [0.0, -1.0])
def test_bandlimited_peak_rejects_nonpositive_python_limit(
    g_max: float,
) -> None:
    """A nonpositive physical band limit is invalid."""
    with pytest.raises(ValueError, match="g_max must be positive"):
        lobato_bandlimited_peak(6, g_max)


@pytest.mark.parametrize("g_max", [0.0, -1.0])
def test_bandlimited_peak_rejects_nonpositive_traced_limit(
    g_max: float,
) -> None:
    """JIT tracing cannot bypass the positive-band-limit check."""
    compiled = jax.jit(lambda value: lobato_bandlimited_peak(6, value))

    with pytest.raises(_TRACED_ERROR_TYPES, match="g_max must be positive"):
        result = compiled(jnp.asarray(g_max, dtype=jnp.float64))
        jax.block_until_ready(result)


@pytest.mark.parametrize("g_max", [jnp.nan, jnp.inf, -jnp.inf])
def test_bandlimited_peak_rejects_nonfinite_limit(g_max: jax.Array) -> None:
    """NaN and infinite band limits are not physical positive cutoffs."""
    compiled = jax.jit(lambda value: lobato_bandlimited_peak(6, value))

    with pytest.raises(_TRACED_ERROR_TYPES, match="g_max"):
        result = compiled(jnp.asarray(g_max, dtype=jnp.float64))
        jax.block_until_ready(result)


@pytest.mark.parametrize("radius", [-0.1, jnp.nan, jnp.inf, -jnp.inf])
def test_projected_dispatch_rejects_invalid_radius(radius: float) -> None:
    """A radial coordinate must be nonnegative and finite under tracing."""
    compiled = jax.jit(
        lambda value: projected_atom_potential(
            6,
            value,
            parameterization="lobato",
        )
    )

    with pytest.raises(_TRACED_ERROR_TYPES, match="radial distances"):
        result = compiled(jnp.asarray(radius, dtype=jnp.float64))
        jax.block_until_ready(result)


@pytest.mark.parametrize("parameterization", ["lobato", "kirkland"])
def test_dispatcher_coordinate_gradients_are_finite(
    parameterization: str,
) -> None:
    """Both models preserve finite reciprocal- and real-space gradients."""
    q_gradient = jax.grad(
        lambda value: atomic_form_factor(
            14,
            value,
            parameterization=parameterization,
        )
    )(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64))
    r_gradient = jax.grad(
        lambda value: projected_atom_potential(
            14,
            value,
            parameterization=parameterization,
        )
    )(jnp.asarray(0.3, dtype=jnp.float64))

    assert bool(jnp.isfinite(q_gradient))
    assert bool(jnp.isfinite(r_gradient))


def test_primitive_coefficient_gradients_are_finite() -> None:
    """Every validated coefficient leaf remains differentiable in physics.

    :see: :func:`ptyrodactyl.multislice.kirkland_form_factor`
    :see: :func:`ptyrodactyl.multislice.kirkland_projected_potential`
    :see: :func:`ptyrodactyl.multislice.lobato_form_factor`
    :see: :func:`ptyrodactyl.multislice.lobato_projected_potential`
    """
    lobato_row = lobato_potentials()[13]
    lobato_params = create_lobato_parameters(
        lobato_row[0::2],
        lobato_row[1::2],
    )
    kirkland_row = kirkland_potentials()[13]
    kirkland_params = create_kirkland_parameters(
        kirkland_row[:6:2],
        kirkland_row[1:6:2],
        kirkland_row[6::2],
        kirkland_row[7::2],
    )
    q = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
    r = jnp.asarray(0.3, dtype=jnp.float64)

    lobato_gradient = jax.grad(
        lambda params: lobato_form_factor(params, q)
        + lobato_projected_potential(params, r)
    )(lobato_params)
    kirkland_gradient = jax.grad(
        lambda params: kirkland_form_factor(params, q)
        + kirkland_projected_potential(params, r)
    )(kirkland_params)

    for gradient in jax.tree_util.tree_leaves(
        (lobato_gradient, kirkland_gradient)
    ):
        assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize("atomic_number", [6, 14, 79])
def test_form_factor_parameterizations_agree_through_four_inverse_angstroms(
    atomic_number: int,
) -> None:
    """C/Si/Au agree within 0.6% over g = 0--4 inverse Angstroms."""
    g = jnp.linspace(0.0, 4.0, 201, dtype=jnp.float64)
    q = 2.0 * jnp.pi * g
    lobato = atomic_form_factor(atomic_number, q, parameterization="lobato")
    kirkland = atomic_form_factor(
        atomic_number,
        q,
        parameterization="kirkland",
    )

    np.testing.assert_allclose(
        np.asarray(kirkland),
        np.asarray(lobato),
        rtol=_FORM_FACTOR_RELATIVE_BOUND,
        atol=0.0,
    )


@pytest.mark.parametrize("atomic_number", [6, 14, 79])
def test_projected_parameterizations_agree_over_reference_radii(
    atomic_number: int,
) -> None:
    """C/Si/Au agree within 3% over r = 0.05--1.5 Angstroms."""
    radius = jnp.geomspace(0.05, 1.5, 201)
    lobato = projected_atom_potential(
        atomic_number,
        radius,
        parameterization="lobato",
    )
    kirkland = projected_atom_potential(
        atomic_number,
        radius,
        parameterization="kirkland",
    )

    np.testing.assert_allclose(
        np.asarray(kirkland),
        np.asarray(lobato),
        rtol=_PROJECTED_RELATIVE_BOUND,
        atol=0.0,
    )
