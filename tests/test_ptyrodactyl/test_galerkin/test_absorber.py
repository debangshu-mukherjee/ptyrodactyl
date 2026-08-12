r"""Tests for LVT.23--LVT.32 axial local-cell CAP evidence and actions."""

from __future__ import annotations

import dataclasses
import functools
from decimal import Decimal
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.absorber as absorber_module
from ptyrodactyl._tools import (
    fraction_from_float,
    fraction_lower_float,
    fraction_upper_float,
)
from ptyrodactyl.galerkin.absorber import (
    _checked_coefficients,
    _exact_rectangles,
    _gram_attempt,
    _signed_position_map,
    apply_axial_physical_cap,
    apply_axial_physical_cap_adjoint,
    certify_axial_cap_floor,
    certify_axial_cell_absorber,
    prepare_axial_cap_floor,
    realize_axial_cell_absorber,
)
from ptyrodactyl.galerkin.local_cell import (
    realize_local_cell_galerkin_potential,
)
from ptyrodactyl.galerkin.local_cell_certification import (
    certify_local_cell_galerkin_potential,
)
from ptyrodactyl.galerkin.local_cell_interaction import (
    certify_local_cell_exact_compression,
    create_local_cell_interaction_core,
)
from ptyrodactyl.types.absorber_types import (
    GalerkinAxialCapCoefficientCertificate,
    GalerkinAxialCapCoefficientFailure,
    GalerkinAxialCapExactFloorFailure,
    GalerkinAxialCapFloorProof,
    GalerkinAxialCapRealizedFloorFailure,
    GalerkinAxialCapRealizedFloorRoute,
    GalerkinAxialCellAbsorber,
)
from ptyrodactyl.types.born_potential_types import (
    create_galerkin_product_support,
)
from ptyrodactyl.types.local_cell_types import (
    create_local_cell_potential_3d,
)
from tests._galerkin_target_fixture import checked_acquisition
from tests.test_ptyrodactyl.test_galerkin.test_local_cell_interaction import (
    _successful_fixture,
)

type _SuccessfulCapFixture = Tuple[
    GalerkinAxialCellAbsorber,
    GalerkinAxialCapCoefficientCertificate,
    GalerkinAxialCapFloorProof,
]
type _AxisCapFixture = Tuple[
    GalerkinAxialCellAbsorber,
    GalerkinAxialCapCoefficientCertificate,
]

type _ReferenceRectangle = Tuple[Fraction, Fraction, Fraction, Fraction]

# Independent 120-digit Decimal/Taylor evaluation with every submitted input,
# including 0.1, interpreted as its exact stored binary64 rational.  The last
# reported decimal is widened outward to make each literal a rational bracket.
_REFERENCE_REAL_ONE = (
    Fraction(
        Decimal(
            "0.22704801859219709462754390731183945392639335717921065278880564016"
        )
    ),
    Fraction(
        Decimal(
            "0.22704801859219709462754390731183945392639335717921065278880564017"
        )
    ),
)
_REFERENCE_IMAG_ONE = (
    Fraction(
        Decimal(
            "0.07377237323125994491134948528434549181050631043182390881963529916"
        )
    ),
    Fraction(
        Decimal(
            "0.07377237323125994491134948528434549181050631043182390881963529917"
        )
    ),
)
_REFERENCE_REAL_TWO = (
    Fraction(
        Decimal(
            "0.07016169628414792524549569379120523135429627966177271429827810467"
        )
    ),
    Fraction(
        Decimal(
            "0.07016169628414792524549569379120523135429627966177271429827810469"
        )
    ),
)
_REFERENCE_IMAG_TWO = (
    Fraction(
        Decimal(
            "0.09656929027509072632029400812072856597120329063331986573730533258"
        )
    ),
    Fraction(
        Decimal(
            "0.09656929027509072632029400812072856597120329063331986573730533260"
        )
    ),
)
_HIGH_PRECISION_COEFFICIENT_BRACKETS: Dict[int, _ReferenceRectangle] = {
    -2: (*_REFERENCE_REAL_TWO, *_REFERENCE_IMAG_TWO),
    -1: (
        *_REFERENCE_REAL_ONE,
        -_REFERENCE_IMAG_ONE[1],
        -_REFERENCE_IMAG_ONE[0],
    ),
    0: (Fraction(1, 2), Fraction(1, 2), Fraction(0), Fraction(0)),
    1: (*_REFERENCE_REAL_ONE, *_REFERENCE_IMAG_ONE),
    2: (
        *_REFERENCE_REAL_TWO,
        -_REFERENCE_IMAG_TWO[1],
        -_REFERENCE_IMAG_TWO[0],
    ),
}


@functools.lru_cache(maxsize=1)
def _successful_cap_fixture() -> _SuccessfulCapFixture:
    """Build one fully replayed exact-scale L4 fixture once per process."""
    _, _, core = _successful_fixture()
    absorber = realize_axial_cell_absorber(
        core,
        jnp.asarray([1.0, 0.0, 0.5], dtype=jnp.float64),
        terminal_axis=0,
        plateau_start=0,
        plateau_count=1,
        plateau_floor=jnp.asarray(1.0, dtype=jnp.float64),
        zero_start=1,
        zero_count=1,
        exact_cap_scale=jnp.asarray(0.25, dtype=jnp.float64),
    )
    certificate = certify_axial_cell_absorber(absorber)
    proof = certify_axial_cap_floor(
        certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    prepared = prepare_axial_cap_floor(proof)
    return absorber, certificate, prepared


@functools.lru_cache(maxsize=1)
def _axis_two_cap_fixture() -> _AxisCapFixture:
    """Build one anisotropic z-axis hard plateau with wide Ia support."""
    state = jnp.asarray(
        [(0, 0, mode) for mode in range(-1, 2)], dtype=jnp.int64
    )
    interaction = jnp.asarray(
        [(0, 0, mode) for mode in range(-2, 3)], dtype=jnp.int64
    )
    absorber = jnp.asarray(
        [
            (0, -1, 0),
            *((0, 0, mode) for mode in range(-5, 6)),
            (0, 1, 0),
        ],
        dtype=jnp.int64,
    )
    work = jnp.asarray(
        [
            (0, transverse, normal)
            for transverse in range(-1, 2)
            for normal in range(-6, 7)
        ],
        dtype=jnp.int64,
    )
    support = create_galerkin_product_support(
        state_indices=state,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(1, 3, 13),
    )
    potential = create_local_cell_potential_3d(
        jnp.asarray([[[1.0]], [[-0.5]], [[2.0]], [[0.25]]], dtype=jnp.float64),
        cell_size=(1.25, 2.0, 0.75),
        box_size=(1.25, 2.0, 3.0),
        cell_center_origin=(0.125, -0.25, 0.2),
        reference_value=0.0,
        reference_semantics="L4 anisotropic axis-mapping test reference",
        producer="L4-axis-two-test-v1",
        provenance_hash="7" * 64,
        producer_coefficient_normalization="producer metadata only",
        producer_bandwidth=1.0,
    )
    eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
    )
    rounded = realize_local_cell_galerkin_potential(potential, eligibility)
    local_certificate = certify_local_cell_galerkin_potential(
        rounded,
        maximum_direct_terms=12,
    )
    compression = certify_local_cell_exact_compression(
        local_certificate,
        accelerating_voltage_kv=200.0,
    )
    core = create_local_cell_interaction_core(compression)
    axial = realize_axial_cell_absorber(
        core,
        jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64),
        terminal_axis=2,
        plateau_start=3,
        plateau_count=2,
        plateau_floor=jnp.asarray(1.0, dtype=jnp.float64),
        zero_start=1,
        zero_count=2,
        exact_cap_scale=jnp.asarray(0.25, dtype=jnp.float64),
    )
    certificate = certify_axial_cell_absorber(axial)
    return axial, certificate


def _analytic_coefficient(mode: int) -> complex:
    """Evaluate a noncertifying NumPy regression for the three-layer case."""
    values = np.asarray([1.0, 0.0, 0.5], dtype=np.float64)
    count = values.size
    if mode != 0 and mode % count == 0:
        return 0.0j
    series = sum(
        value * np.exp(-2j * np.pi * mode * row / count)
        for row, value in enumerate(values)
    )
    return (
        np.sinc(mode / count)
        * np.exp(-2j * np.pi * mode * 0.1 / 3.0)
        * series
        / count
    )


def _reference_coefficient(mode: int) -> complex:
    """Return the midpoint of one fixed 65-digit independent bracket."""
    real_lower, real_upper, imag_lower, imag_upper = (
        _HIGH_PRECISION_COEFFICIENT_BRACKETS[mode]
    )
    return complex(
        float((real_lower + real_upper) / 2),
        float((imag_lower + imag_upper) / 2),
    )


def _dense_interval_gram(
    normal_indices: np.ndarray, width: float
) -> np.ndarray:
    """Integrate one interval Gramian independently in dense binary64."""
    differences = normal_indices[None, :] - normal_indices[:, None]
    return np.asarray(
        width
        * np.sinc(differences * width)
        * np.exp(1j * np.pi * differences * width),
        dtype=np.complex128,
    )


def test_lvt24_rectangles_du_mapping_and_lvt31_transfer() -> None:
    """Enclose every coefficient and freeze the exact Du-to-Ia evidence.

    :see: :func:`ptyrodactyl.galerkin.certify_axial_cell_absorber`
    """
    absorber, certificate, _ = _successful_cap_fixture()
    assert certificate.failure is GalerkinAxialCapCoefficientFailure.NONE
    assert bool(certificate.finite_certificate)
    assert absorber.terminal_axis == 0
    assert absorber.layer_values.shape == (3,)
    assert_array_equal(absorber.signed_absorber_positions, [4, 3, 2, 1, 0])
    modes = np.asarray(absorber.support.absorber_indices)[:, 0]
    for position, mode in enumerate(modes):
        reference = _HIGH_PRECISION_COEFFICIENT_BRACKETS[int(mode)]
        stored = (
            Fraction.from_float(
                float(
                    certificate.exact_coefficient_real_lower_bounds[position]
                )
            ),
            Fraction.from_float(
                float(
                    certificate.exact_coefficient_real_upper_bounds[position]
                )
            ),
            Fraction.from_float(
                float(
                    certificate.exact_coefficient_imag_lower_bounds[position]
                )
            ),
            Fraction.from_float(
                float(
                    certificate.exact_coefficient_imag_upper_bounds[position]
                )
            ),
        )
        assert stored[0] <= reference[0] <= reference[1] <= stored[1]
        assert stored[2] <= reference[2] <= reference[3] <= stored[3]
        assert_allclose(
            _analytic_coefficient(int(mode)),
            _reference_coefficient(int(mode)),
            rtol=2.0e-15,
            atol=2.0e-17,
        )
    assert_array_equal(
        certificate.difference_indices,
        [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
    )
    assert_array_equal(certificate.difference_multiplicities, [1, 2, 3, 2, 1])
    assert_array_equal(
        certificate.state_pair_absorber_positions,
        [2, 1, 0, 3, 2, 1, 4, 3, 2],
    )
    radicand = sum(
        int(multiplicity)
        * float(certificate.coefficient_error_bounds[int(position)]) ** 2
        for position, multiplicity in zip(
            np.asarray(certificate.difference_absorber_positions),
            np.asarray(certificate.difference_multiplicities),
            strict=True,
        )
    )
    assert float(certificate.absorber_operator_error_bound) >= np.sqrt(
        radicand
    )
    state = np.asarray(absorber.support.state_indices)
    coefficient_lookup = {
        tuple(int(value) for value in index): coefficient
        for index, coefficient in zip(
            np.asarray(absorber.support.absorber_indices),
            np.asarray(absorber.absorber_coefficients),
            strict=True,
        )
    }
    dense_algebraic = np.asarray(
        [
            [
                coefficient_lookup[
                    tuple(int(value) for value in (left - right))
                ]
                for right in state
            ]
            for left in state
        ],
        dtype=np.complex128,
    )
    dense_exact = np.asarray(
        [
            [_reference_coefficient(int((left - right)[0])) for right in state]
            for left in state
        ],
        dtype=np.complex128,
    )
    spectral_error = np.linalg.norm(dense_algebraic - dense_exact, ord=2)
    assert spectral_error <= float(certificate.absorber_operator_error_bound)


def test_symbolic_transverse_zeros_and_nonhermitian_rejection() -> None:
    """Keep transverse LVT.24 zeros exact and reject asymmetric storage."""
    values = np.asarray([1.0, 0.0, 0.5], dtype=np.float64)
    indices = np.asarray([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int64)
    modes, signed_positions = _signed_position_map(indices)
    rectangles = _exact_rectangles(
        values,
        modes,
        signed_positions,
        origin=0.1,
        length=3.0,
        terminal_axis=0,
    )
    zero = (Fraction(0),) * 4
    assert rectangles[0] == zero
    assert rectangles[2] == zero
    with pytest.raises(ValueError, match="Hermitian pairs"):
        _checked_coefficients(
            jnp.asarray([1.0 + 1.0j, 0.5, 1.0 + 1.0j], dtype=jnp.complex128),
            modes,
            signed_positions,
        )


def test_axis_two_anisotropic_wrapped_hard_plateau_and_wide_modes() -> None:
    """Bind xyz/zyx, wrapped blocks, m=N zeros, and modes beyond Nyquist.

    :see: :func:`ptyrodactyl.galerkin.realize_axial_cell_absorber`
    """
    absorber, certificate = _axis_two_cap_fixture()
    assert certificate.failure is GalerkinAxialCapCoefficientFailure.NONE
    assert bool(certificate.finite_certificate)
    assert absorber.terminal_axis == 2
    assert absorber.layer_values.shape == (4,)
    assert (
        absorber.interaction_core.compression.realization.local_potential.cell_values.shape
        == (
            4,
            1,
            1,
        )
    )
    plateau_positions = {
        (absorber.plateau_start + offset) % absorber.layer_values.shape[0]
        for offset in range(absorber.plateau_count)
    }
    zero_positions = {
        (absorber.zero_start + offset) % absorber.layer_values.shape[0]
        for offset in range(absorber.zero_count)
    }
    assert plateau_positions == {0, 3}
    assert zero_positions == {1, 2}
    assert plateau_positions.isdisjoint(zero_positions)
    values = np.asarray(absorber.layer_values)
    assert_array_equal(values[list(sorted(plateau_positions))], [1.0, 1.0])
    assert_array_equal(values[list(sorted(zero_positions))], [0.0, 0.0])

    indices = np.asarray(absorber.support.absorber_indices)
    coefficients = np.asarray(absorber.absorber_coefficients)
    lookup = {
        tuple(int(value) for value in index): coefficient
        for index, coefficient in zip(indices, coefficients, strict=True)
    }
    assert lookup[(0, -1, 0)] == 0.0j
    assert lookup[(0, 1, 0)] == 0.0j
    assert lookup[(0, 0, -4)] == 0.0j
    assert lookup[(0, 0, 4)] == 0.0j
    assert lookup[(0, 0, 5)] != 0.0j

    mode = 5
    direct = (
        np.sinc(mode / values.size)
        * np.exp(-2j * np.pi * mode * 0.2 / 3.0)
        * sum(
            value * np.exp(-2j * np.pi * mode * row / values.size)
            for row, value in enumerate(values)
        )
        / values.size
    )
    assert_allclose(lookup[(0, 0, mode)], direct, rtol=3.0e-15, atol=2.0e-16)
    wide_position = int(
        np.flatnonzero(np.all(indices == np.asarray((0, 0, mode)), axis=1))[0]
    )
    assert np.isfinite(
        certificate.exact_coefficient_real_lower_bounds[wide_position]
    )
    assert np.isfinite(
        certificate.exact_coefficient_real_upper_bounds[wide_position]
    )


def test_verified_floor_routes_are_separate_and_exact_target_is_positive() -> (
    None
):
    """Accept the support-only LDL/Weyl proof and exactly one LVT.32 route.

    :see: :func:`ptyrodactyl.galerkin.certify_axial_cap_floor`
    """
    _, _, proof = _successful_cap_fixture()
    assert proof.exact_target_failure is GalerkinAxialCapExactFloorFailure.NONE
    assert (
        proof.realized_floor_failure
        is GalerkinAxialCapRealizedFloorFailure.NONE
    )
    assert proof.realized_floor_route is (
        GalerkinAxialCapRealizedFloorRoute.EXACT_FROZEN_SCALE_LVT32A
    )
    assert bool(proof.exact_target_floor_eligible)
    assert bool(proof.realized_floor_eligible)
    assert int(proof.gram_degree) == 2
    assert (
        proof.gram_subinterval_numerator,
        proof.gram_subinterval_denominator,
    ) == (
        "1",
        "6",
    )
    assert 0.0 < float(proof.plateau_gram_lower_bound) < 1.0 / 6.0
    assert float(proof.dimensionless_exact_floor_lower_bound) > 0.0
    assert float(proof.realized_physical_floor_lower_bound) > 0.0
    assert len(proof.gram_transcript_digest) == 64
    delta = Fraction(
        int(proof.gram_subinterval_numerator),
        int(proof.gram_subinterval_denominator),
    )
    consecutive = _dense_interval_gram(
        np.arange(int(proof.gram_degree) + 1, dtype=np.float64),
        float(delta),
    )
    certified = float(proof.plateau_gram_lower_bound)
    assert 0.0 < certified <= np.linalg.eigvalsh(consecutive)[0]

    absorber = proof.coefficient_certificate.absorber
    states = np.asarray(absorber.support.state_indices)
    transverse_axes = tuple(
        axis for axis in range(3) if axis != absorber.terminal_axis
    )
    fiber_keys = {
        tuple(int(row[axis]) for axis in transverse_axes) for row in states
    }
    plateau_width = absorber.plateau_count / absorber.layer_values.shape[0]
    dense_fiber_lowers = []
    for key in fiber_keys:
        normal = np.asarray(
            [
                int(row[absorber.terminal_axis])
                for row in states
                if tuple(int(row[axis]) for axis in transverse_axes) == key
            ],
            dtype=np.float64,
        )
        dense_fiber_lowers.append(
            float(
                np.linalg.eigvalsh(
                    _dense_interval_gram(normal, plateau_width)
                )[0]
            )
        )
    assert certified <= min(dense_fiber_lowers)


def test_degree_zero_gram_and_typed_preflight_failures() -> None:
    """Check dense degree zero and fail closed before expensive arithmetic."""
    degree_zero = _gram_attempt(
        0,
        Fraction(1, 6),
        32,
        40,
        0,
        1_000,
    )
    assert degree_zero["failure"] is GalerkinAxialCapExactFloorFailure.NONE
    degree_zero_lower = degree_zero["gram_lower"]
    assert isinstance(degree_zero_lower, Fraction)
    dense_zero = _dense_interval_gram(np.asarray([0.0]), 1.0 / 6.0)
    assert 0.0 < float(degree_zero_lower) <= np.linalg.eigvalsh(dense_zero)[0]

    degree_failed = _gram_attempt(
        2,
        Fraction(1, 6),
        32,
        40,
        1,
        50_000_000,
    )
    work_failed = _gram_attempt(
        2,
        Fraction(1, 6),
        32,
        40,
        2,
        1,
    )
    assert degree_failed["failure"] is (
        GalerkinAxialCapExactFloorFailure.GRAM_DEGREE_BUDGET_EXCEEDED
    )
    assert work_failed["failure"] is (
        GalerkinAxialCapExactFloorFailure.GRAM_WORK_BUDGET_EXCEEDED
    )
    assert degree_failed["gram_lower"] is None
    assert work_failed["gram_lower"] is None


def test_public_floor_budget_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose degree/work failures without repeating nested L3 replay."""
    _, certificate, _ = _successful_cap_fixture()
    monkeypatch.setattr(
        absorber_module,
        "_authenticate_coefficient_certificate",
        lambda submitted: certificate,
    )
    degree_failed = certify_axial_cap_floor(
        certificate,
        maximum_gram_degree=1,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    work_failed = certify_axial_cap_floor(
        certificate,
        maximum_gram_degree=2,
        gram_precision_bits=32,
        ldl_iteration_count=40,
        maximum_gram_work=1,
    )
    for proof, failure in (
        (
            degree_failed,
            GalerkinAxialCapExactFloorFailure.GRAM_DEGREE_BUDGET_EXCEEDED,
        ),
        (
            work_failed,
            GalerkinAxialCapExactFloorFailure.GRAM_WORK_BUDGET_EXCEEDED,
        ),
    ):
        assert proof.exact_target_failure is failure
        assert not bool(proof.exact_target_floor_eligible)
        assert proof.realized_floor_failure is (
            GalerkinAxialCapRealizedFloorFailure.EXACT_TARGET_FLOOR_NOT_FINITE
        )
        assert not bool(proof.realized_floor_eligible)


def test_physical_actions_match_dense_matrix_and_formal_adjoint() -> None:
    """Apply physical B_alg and its explicit reverse-conjugate adjoint.

    :see: :func:`ptyrodactyl.galerkin.apply_axial_physical_cap`
    :see: :func:`ptyrodactyl.galerkin.apply_axial_physical_cap_adjoint`
    """
    _, certificate, proof = _successful_cap_fixture()
    absorber = certificate.absorber
    state_count = absorber.support.state_indices.shape[0]
    pair_positions = np.asarray(certificate.state_pair_absorber_positions)
    coefficients = np.asarray(absorber.absorber_coefficients)
    scale = float(absorber.algebraic_cap_scale)
    dense = np.asarray(
        [
            [
                scale
                * coefficients[pair_positions[row * state_count + column]]
                for column in range(state_count)
            ]
            for row in range(state_count)
        ],
        dtype=np.complex128,
    )
    field = jnp.asarray(
        [0.25 + 0.5j, -1.0 + 0.2j, 0.7 - 0.3j], dtype=jnp.complex128
    )
    assert_allclose(
        apply_axial_physical_cap(proof, field),
        dense @ np.asarray(field),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert_allclose(
        apply_axial_physical_cap_adjoint(proof, field),
        dense.conj().T @ np.asarray(field),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    compiled = jax.jit(lambda value: apply_axial_physical_cap(proof, value))
    assert_allclose(
        compiled(field),
        dense @ np.asarray(field),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


@functools.lru_cache(maxsize=1)
def _positive_route_b_fixture() -> GalerkinAxialCapFloorProof:
    """Certify one positive unequal-scale LVT.32b result."""
    absorber, _, _ = _successful_cap_fixture()
    algebraic_scale = np.nextafter(
        float(absorber.exact_cap_scale), np.inf, dtype=np.float64
    )
    modified = eqx.tree_at(
        lambda value: value.algebraic_cap_scale,
        absorber,
        jnp.asarray(algebraic_scale, dtype=jnp.float64),
    )
    certificate = certify_axial_cell_absorber(modified)
    return certify_axial_cap_floor(
        certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )


def test_positive_unequal_scale_uses_lvt32b_once() -> None:
    """Rebuild delta_B and exclude an accidental duplicate scale charge."""
    proof = _positive_route_b_fixture()
    absorber = proof.coefficient_certificate.absorber
    assert proof.realized_floor_route is (
        GalerkinAxialCapRealizedFloorRoute.SCALE_TRANSFER_LVT32B
    )
    assert proof.realized_floor_failure is (
        GalerkinAxialCapRealizedFloorFailure.NONE
    )
    assert bool(proof.exact_target_floor_eligible)
    assert bool(proof.realized_floor_eligible)

    epsilon = fraction_from_float(float(absorber.exact_cap_scale))
    algebraic = fraction_from_float(float(absorber.algebraic_cap_scale))
    delta_epsilon = abs(algebraic - epsilon)
    delta_a = fraction_from_float(
        float(proof.coefficient_certificate.absorber_operator_error_bound)
    )
    expected_delta_b = (epsilon + delta_epsilon) * delta_a + delta_epsilon
    assert float(proof.scale_error_bound) == fraction_upper_float(
        delta_epsilon
    )
    assert float(proof.physical_operator_error_upper_bound) == (
        fraction_upper_float(expected_delta_b)
    )

    gram = fraction_from_float(float(proof.plateau_gram_lower_bound))
    plateau = fraction_from_float(float(absorber.plateau_floor))
    one_charge_proxy = epsilon * plateau * gram - expected_delta_b
    one_charge_lower = fraction_lower_float(one_charge_proxy)
    realized = float(proof.realized_physical_floor_lower_bound)
    ulp = np.spacing(abs(realized))
    assert abs(realized - one_charge_lower) <= 4.0 * ulp
    duplicate_charge_proxy = one_charge_proxy - expected_delta_b
    assert realized > fraction_upper_float(duplicate_charge_proxy)

    gram_budget_failed = certify_axial_cap_floor(
        proof.coefficient_certificate,
        maximum_gram_degree=2,
        gram_precision_bits=32,
        ldl_iteration_count=40,
        maximum_gram_work=1,
    )
    assert gram_budget_failed.exact_target_failure is (
        GalerkinAxialCapExactFloorFailure.GRAM_WORK_BUDGET_EXCEEDED
    )
    assert not bool(gram_budget_failed.exact_target_floor_eligible)
    assert not bool(gram_budget_failed.realized_floor_eligible)
    assert np.isfinite(gram_budget_failed.physical_operator_error_upper_bound)
    assert float(gram_budget_failed.physical_operator_error_upper_bound) == (
        fraction_upper_float(expected_delta_b)
    )


@functools.lru_cache(maxsize=1)
def _nonpositive_route_b_fixture() -> GalerkinAxialCapFloorProof:
    """Certify a zero approximant under a distinct algebraic scale."""
    absorber, _, _ = _successful_cap_fixture()
    modified = eqx.tree_at(
        lambda value: (
            value.absorber_coefficients,
            value.algebraic_cap_scale,
        ),
        absorber,
        (
            jnp.zeros_like(absorber.absorber_coefficients),
            jnp.asarray(0.5, dtype=jnp.float64),
        ),
    )
    certificate = certify_axial_cell_absorber(modified)
    return certify_axial_cap_floor(
        certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )


def test_nonpositive_realized_route_retains_exact_target_eligibility() -> None:
    """Fail LVT.32b without invalidating the exact LVT.29a target floor."""
    proof = _nonpositive_route_b_fixture()
    assert proof.realized_floor_route is (
        GalerkinAxialCapRealizedFloorRoute.SCALE_TRANSFER_LVT32B
    )
    assert proof.exact_target_failure is GalerkinAxialCapExactFloorFailure.NONE
    assert bool(proof.exact_target_floor_eligible)
    assert not bool(proof.realized_floor_eligible)
    assert proof.realized_floor_failure in {
        GalerkinAxialCapRealizedFloorFailure.REALIZED_DIMENSIONLESS_FLOOR_NONPOSITIVE,
        GalerkinAxialCapRealizedFloorFailure.REALIZED_PHYSICAL_FLOOR_NONPOSITIVE,
    }


@functools.lru_cache(maxsize=1)
def _coefficient_noncertificate_floor() -> GalerkinAxialCapFloorProof:
    """Prove the target floor above a typed coefficient noncertificate."""
    absorber, _, _ = _successful_cap_fixture()
    certificate = certify_axial_cell_absorber(
        absorber,
        maximum_direct_terms=1,
    )
    return certify_axial_cap_floor(
        certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )


def test_coefficient_noncertificate_does_not_erase_exact_target_floor() -> (
    None
):
    """Keep LVT.29a eligible when direct LVT.24 evidence is budget-failed."""
    proof = _coefficient_noncertificate_floor()
    certificate = proof.coefficient_certificate
    assert certificate.failure is (
        GalerkinAxialCapCoefficientFailure.DIRECT_TERM_BUDGET_EXCEEDED
    )
    assert not bool(certificate.finite_certificate)
    assert proof.exact_target_failure is GalerkinAxialCapExactFloorFailure.NONE
    assert bool(proof.exact_target_floor_eligible)
    assert proof.realized_floor_failure is (
        GalerkinAxialCapRealizedFloorFailure.COEFFICIENT_CERTIFICATE_NOT_FINITE
    )
    assert not bool(proof.realized_floor_eligible)


def test_prepare_rejects_forged_nested_pair_map() -> None:
    """Replay the coefficient certificate before trusting its pair map.

    :see: :func:`ptyrodactyl.galerkin.prepare_axial_cap_floor`
    """
    _, _, proof = _successful_cap_fixture()
    certificate = proof.coefficient_certificate
    pair_map = certificate.state_pair_absorber_positions
    altered = pair_map.at[0].set(
        (pair_map[0] + 1) % certificate.absorber.absorber_coefficients.shape[0]
    )
    forged_certificate = eqx.tree_at(
        lambda value: value.state_pair_absorber_positions,
        certificate,
        altered,
    )
    forged = eqx.tree_at(
        lambda value: value.coefficient_certificate,
        proof,
        forged_certificate,
    )
    with pytest.raises(ValueError, match="replay"):
        prepare_axial_cap_floor(forged)


def test_prepare_payload_comparison_covers_all_forgery_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover budgets, route/status, transcript, and nested digest storage."""
    _, _, proof = _successful_cap_fixture()
    certificate = proof.coefficient_certificate
    absorber = certificate.absorber

    certificate_budget = eqx.tree_at(
        lambda value: value.maximum_state_pairs,
        certificate,
        certificate.maximum_state_pairs + 1,
    )
    forged_certificate_budget = eqx.tree_at(
        lambda value: value.coefficient_certificate,
        proof,
        certificate_budget,
    )
    forged_gram_budget = eqx.tree_at(
        lambda value: value.maximum_gram_work,
        proof,
        proof.maximum_gram_work + 1,
    )
    forged_route = dataclasses.replace(
        proof,
        realized_floor_route=(
            GalerkinAxialCapRealizedFloorRoute.SCALE_TRANSFER_LVT32B
        ),
    )
    forged_status = dataclasses.replace(
        proof,
        realized_floor_failure=(
            GalerkinAxialCapRealizedFloorFailure.REALIZED_PHYSICAL_FLOOR_NONPOSITIVE
        ),
    )
    forged_transcript = dataclasses.replace(
        proof,
        gram_transcript_digest="8" * 64,
    )
    forged_proof_digest = dataclasses.replace(
        proof,
        proof_digest="9" * 64,
    )
    forged_core = dataclasses.replace(
        absorber.interaction_core,
        operator_digest="a" * 64,
    )
    forged_absorber = dataclasses.replace(
        absorber,
        interaction_core=forged_core,
    )
    forged_nested_certificate = dataclasses.replace(
        certificate,
        absorber=forged_absorber,
    )
    forged_nested_digest = dataclasses.replace(
        proof,
        coefficient_certificate=forged_nested_certificate,
    )

    monkeypatch.setattr(
        absorber_module,
        "_certify_axial_cap_floor_impl",
        lambda *args, **kwargs: proof,
    )
    for forged in (
        forged_certificate_budget,
        forged_gram_budget,
        forged_route,
        forged_status,
        forged_transcript,
        forged_proof_digest,
        forged_nested_digest,
    ):
        with pytest.raises(ValueError, match="full host replay"):
            prepare_axial_cap_floor(forged)
