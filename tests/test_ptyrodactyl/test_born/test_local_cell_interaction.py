r"""Tests for exact LVT compression and its fixed interaction core.

These tests cover full L2 replay, product-support reconstruction, exact
difference coverage and multiplicity, SC.4 coupling, sign-general interaction
rectangles, LVT.18, bounded preflight, operator identity, dense action, formal
matrix adjoint, JAX complex-VJP convention, and the explicit preparation trust
boundary.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.born.local_cell_interaction as interaction_module
from ptyrodactyl._host_interval import (
    _fraction_from_float,
    _fraction_upper_float,
    _sqrt_fraction_upper,
)
from ptyrodactyl.born.local_cell import (
    realize_local_cell_galerkin_potential,
)
from ptyrodactyl.born.local_cell_certification import (
    certify_local_cell_galerkin_potential,
)
from ptyrodactyl.born.local_cell_interaction import (
    _authenticate_local_cell_exact_compression,
    _stream_difference_evidence,
    apply_local_cell_interaction,
    apply_local_cell_interaction_adjoint,
    certify_local_cell_exact_compression,
    create_local_cell_interaction_core,
    prepare_local_cell_interaction_core,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.constants import C_LIGHT, E_CHARGE, HBAR, M_E
from ptyrodactyl.types.local_cell_interaction_types import (
    GalerkinLocalCellCompressionFailure,
    GalerkinLocalCellExactCompression,
    GalerkinLocalCellInteractionCore,
)
from ptyrodactyl.types.local_cell_types import (
    GalerkinLocalCellPotentialRealization,
    create_local_cell_potential_3d,
)
from tests._galerkin_target_fixture import checked_acquisition

_PROVENANCE = "4" * 64


def _line_support(
    *, complete_interaction: bool = True
) -> GalerkinProductSupport:
    """Build a three-state line support, optionally missing outer Du modes."""
    state = jnp.asarray(((-1, 0, 0), (0, 0, 0), (1, 0, 0)), dtype=jnp.int64)
    interaction_extent = 2 if complete_interaction else 1
    interaction = jnp.asarray(
        [
            (mode, 0, 0)
            for mode in range(-interaction_extent, interaction_extent + 1)
        ],
        dtype=jnp.int64,
    )
    absorber = jnp.asarray(
        [(mode, 0, 0) for mode in range(-2, 3)], dtype=jnp.int64
    )
    work_extent = 3
    work = jnp.asarray(
        [(mode, 0, 0) for mode in range(-work_extent, work_extent + 1)],
        dtype=jnp.int64,
    )
    return create_galerkin_product_support(
        state_indices=state,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(2 * work_extent + 3, 1, 1),
    )


def _realization(
    support: GalerkinProductSupport,
) -> GalerkinLocalCellPotentialRealization:
    """Build and directly certify one shifted three-cell local potential."""
    potential = create_local_cell_potential_3d(
        jnp.asarray([[[1.0, -0.5, 2.0]]], dtype=jnp.float64),
        cell_size=(1.0, 1.0, 1.0),
        box_size=(3.0, 1.0, 1.0),
        cell_center_origin=(0.1, 0.2, 0.3),
        reference_value=0.0,
        reference_semantics="L3 interaction test reference",
        producer="L3-interaction-test-v1",
        provenance_hash=_PROVENANCE,
        producer_coefficient_normalization="producer metadata only",
        producer_bandwidth=1.0,
    )
    eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=0,
    )
    rounded = realize_local_cell_galerkin_potential(potential, eligibility)
    canonical_mode_count = (support.interaction_indices.shape[0] + 1) // 2
    term_count = canonical_mode_count * potential.cell_values.size
    return certify_local_cell_galerkin_potential(
        rounded,
        maximum_direct_terms=term_count,
    )


@functools.lru_cache(maxsize=1)
def _successful_fixture() -> Tuple[
    GalerkinLocalCellPotentialRealization,
    GalerkinLocalCellExactCompression,
    GalerkinLocalCellInteractionCore,
]:
    """Build one fully replayed finite L3 fixture once per test process."""
    realization = _realization(_line_support())
    compression = certify_local_cell_exact_compression(
        realization,
        accelerating_voltage_kv=200.0,
    )
    core = create_local_cell_interaction_core(compression)
    return realization, compression, core


def _dense_matrix(core: GalerkinLocalCellInteractionCore) -> np.ndarray:
    """Build the independent dense matrix from exact state differences."""
    support = core.support
    state = np.asarray(support.state_indices)
    modes = np.asarray(support.interaction_indices)
    coefficients = np.asarray(core.compression.interaction_coefficients)
    lookup = {
        tuple(int(value) for value in mode): coefficient
        for mode, coefficient in zip(modes, coefficients, strict=True)
    }
    return np.asarray(
        [
            [
                lookup[tuple(int(value) for value in (row_mode - column_mode))]
                for column_mode in state
            ]
            for row_mode in state
        ],
        dtype=np.complex128,
    )


def _exact_sigma(voltage_kv: float) -> Fraction:
    """Compute SC.4 independently from exact stored dyadic inputs."""
    voltage = Fraction.from_float(voltage_kv) * 1000
    mass = Fraction.from_float(float(np.asarray(M_E)))
    charge = Fraction.from_float(float(np.asarray(E_CHARGE)))
    speed = Fraction.from_float(float(np.asarray(C_LIGHT)))
    hbar = Fraction.from_float(float(np.asarray(HBAR)))
    return (
        2
        * mass
        * charge
        / (hbar * hbar)
        * (1 + charge * voltage / (mass * speed * speed))
        * Fraction(1, 10**20)
    )


def test_complete_compression_has_exact_du_mapping_and_multiplicity() -> None:
    """Freeze lexicographic Du, pair lookup, and the LVT.17 sum n squared."""
    _, compression, _ = _successful_fixture()
    assert compression.failure is GalerkinLocalCellCompressionFailure.NONE
    assert bool(compression.finite_certificate)
    assert_array_equal(
        compression.difference_indices,
        np.asarray([[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    )
    assert_array_equal(compression.difference_multiplicities, [1, 2, 3, 2, 1])
    assert int(jnp.sum(compression.difference_multiplicities)) == 9
    assert_array_equal(
        compression.state_pair_interaction_positions,
        [2, 1, 0, 3, 2, 1, 4, 3, 2],
    )
    assert compression.compression_claim.startswith("LVT.15")
    assert "per-call" in compression.per_call_arithmetic_exclusion
    assert "O(1)" in compression.host_transient_scalar_scope


def test_exact_sigma_point_and_sign_general_chi_rectangles() -> None:
    """Enclose exact SC.4 and every signed L2 rectangle product once."""
    realization, compression, _ = _successful_fixture()
    sigma = _exact_sigma(200.0)
    sigma_lower = Fraction.from_float(
        float(compression.exact_coupling_lower_bound)
    )
    sigma_upper = Fraction.from_float(
        float(compression.exact_coupling_upper_bound)
    )
    assert sigma_lower <= sigma <= sigma_upper

    certificate = realization.coefficient_certificate
    assert certificate is not None
    for difference_position, raw_support_position in enumerate(
        np.asarray(compression.difference_interaction_positions)
    ):
        support_position = int(raw_support_position)
        for source_lower, source_upper, target_lower, target_upper in (
            (
                certificate.exact_coefficient_real_lower_bounds,
                certificate.exact_coefficient_real_upper_bounds,
                compression.exact_interaction_real_lower_bounds,
                compression.exact_interaction_real_upper_bounds,
            ),
            (
                certificate.exact_coefficient_imag_lower_bounds,
                certificate.exact_coefficient_imag_upper_bounds,
                compression.exact_interaction_imag_lower_bounds,
                compression.exact_interaction_imag_upper_bounds,
            ),
        ):
            source = (
                Fraction.from_float(float(source_lower[support_position])),
                Fraction.from_float(float(source_upper[support_position])),
            )
            products = (
                sigma_lower * source[0],
                sigma_lower * source[1],
                sigma_upper * source[0],
                sigma_upper * source[1],
            )
            assert Fraction.from_float(
                float(target_lower[difference_position])
            ) <= min(products)
            assert max(products) <= Fraction.from_float(
                float(target_upper[difference_position])
            )
    assert np.any(
        np.asarray(compression.exact_interaction_real_lower_bounds) < 0
    )


def test_lvt18_is_exact_fraction_sum_of_stored_component_errors() -> None:
    """Rebuild LVT.18 from exact stored binary64 errors and multiplicities."""
    _, compression, _ = _successful_fixture()
    radicand = sum(
        (
            int(multiplicity)
            * _fraction_from_float(float(error))
            * _fraction_from_float(float(error))
            for multiplicity, error in zip(
                np.asarray(compression.difference_multiplicities),
                np.asarray(compression.interaction_coefficient_error_bounds),
                strict=True,
            )
        ),
        start=Fraction(0),
    )
    expected = _fraction_upper_float(_sqrt_fraction_upper(radicand))
    assert float(compression.fixed_interaction_error_bound) == expected
    assert np.all(
        np.isfinite(compression.interaction_coefficient_error_bounds)
    )


def test_actions_match_dense_matrix_and_formal_adjoint() -> None:
    """Match dense R/R-star and JAX's conjugated complex-VJP convention."""
    _, _, stored_core = _successful_fixture()
    core = prepare_local_cell_interaction_core(stored_core)
    field = jnp.asarray((1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 0.2j))
    cotangent = jnp.asarray((-0.3 + 0.8j, 0.4 - 1.2j, 2.0 + 0.1j))
    dense = _dense_matrix(core)

    forward = apply_local_cell_interaction(core, field)
    formal_adjoint = apply_local_cell_interaction_adjoint(core, cotangent)
    assert_allclose(forward, dense @ np.asarray(field), rtol=2e-15, atol=2e-15)
    assert_allclose(
        formal_adjoint,
        dense.conj().T @ np.asarray(cotangent),
        rtol=2e-15,
        atol=2e-15,
    )
    assert_allclose(
        jnp.vdot(forward, cotangent),
        jnp.vdot(field, formal_adjoint),
        rtol=3e-15,
        atol=3e-15,
    )

    def closed_forward(value):
        return apply_local_cell_interaction(core, value)

    assert_allclose(
        jax.jit(closed_forward)(field), forward, rtol=0.0, atol=0.0
    )
    _, pullback = jax.vjp(closed_forward, field)
    vjp_formal_adjoint = jnp.conj(pullback(jnp.conj(cotangent))[0])
    assert_allclose(
        vjp_formal_adjoint,
        formal_adjoint,
        rtol=3e-15,
        atol=3e-15,
    )


def test_prepare_rejects_self_rehashed_coefficient_forgery() -> None:
    """Require mathematical replay rather than digest-string consistency."""
    _, _, core = _successful_fixture()
    forged_compression = eqx.tree_at(
        lambda value: value.interaction_coefficients,
        core.compression,
        core.compression.interaction_coefficients.at[2].add(1.0e-12),
    )
    forged_compression = eqx.tree_at(
        lambda value: (value.operator_digest, value.certificate_digest),
        forged_compression,
        ("d" * 64, "e" * 64),
    )
    forged_core = eqx.tree_at(
        lambda value: (value.compression, value.operator_digest),
        core,
        (forged_compression, "d" * 64),
    )
    with pytest.raises(ValueError, match="replay|match"):
        prepare_local_cell_interaction_core(forged_core)


def test_pair_budget_fails_before_pair_map_allocation(monkeypatch) -> None:
    """Return a typed budget noncertificate before streaming any pair."""
    realization, _, _ = _successful_fixture()

    def forbidden_stream(*args, **kwargs):
        raise AssertionError("pair stream must not run above its budget")

    monkeypatch.setattr(
        interaction_module,
        "_stream_difference_evidence",
        forbidden_stream,
    )
    failed = certify_local_cell_exact_compression(
        realization,
        accelerating_voltage_kv=200.0,
        maximum_state_pairs=8,
    )
    assert failed.failure is (
        GalerkinLocalCellCompressionFailure.STATE_PAIR_BUDGET_EXCEEDED
    )
    assert not bool(failed.finite_certificate)
    assert np.isinf(failed.fixed_interaction_error_bound)
    assert failed.state_pair_interaction_positions.shape == (0,)
    with pytest.raises(ValueError, match="finite"):
        create_local_cell_interaction_core(failed)


def test_missing_du_is_typed_even_without_coefficient_inspection() -> None:
    """Fail exact coverage from integer sets before coefficient shortcuts."""
    support = _line_support(complete_interaction=False)
    state = np.asarray(support.state_indices)
    interaction = np.asarray(support.interaction_indices)
    differences, positions, multiplicities, pair_map, failure = (
        _stream_difference_evidence(state, interaction, support.work_shape)
    )
    assert failure is (
        GalerkinLocalCellCompressionFailure.DIFFERENCE_COVERAGE_MISSING
    )
    assert differences.shape == (0, 3)
    assert positions.shape == multiplicities.shape == pair_map.shape == (0,)


def test_scalar_dtype_forgery_rejects_before_replay() -> None:
    """Reject a public scalar whose dtype changed despite retained values."""
    _, compression, _ = _successful_fixture()
    forged = eqx.tree_at(
        lambda value: value.maximum_state_pairs,
        compression,
        compression.maximum_state_pairs.astype(jnp.int32),
    )
    with pytest.raises(ValueError, match="int64"):
        _authenticate_local_cell_exact_compression(forged)


def test_host_boundary_rejects_traced_compression_inputs() -> None:
    """Keep certification and preparation outside traced transformations."""
    realization, _, _ = _successful_fixture()

    with pytest.raises(ValueError, match="host values"):
        jax.jit(
            lambda voltage: certify_local_cell_exact_compression(
                realization,
                accelerating_voltage_kv=voltage,
            )
        )(jnp.asarray(200.0))
