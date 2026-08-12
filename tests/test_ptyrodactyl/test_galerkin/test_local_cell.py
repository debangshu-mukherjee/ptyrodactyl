r"""Tests for :mod:`ptyrodactyl.galerkin.local_cell`.

Extended Summary
----------------
These tests compare rounded LVT.7 coefficients with direct integrals of
half-open rectangular cells. They exercise unwrapped modes on odd and even
grids, exact symbolic sinc zeros, alias collisions, physical origin phases,
the rounded callable's physical-metric adjoint, and JAX transformations.
"""

import dataclasses
import functools
import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from numpy.testing import assert_allclose

from ptyrodactyl.galerkin.local_cell import (
    _authenticate_local_cell_tail,
    _coefficient_error_bounds,
    _local_cell_coefficients_from_full_grid,
    _origin_cycle_fractions,
    _outward_lvt9_subtraction,
    _physical_cell_volume,
    _tail_enclosure_digest,
    apply_local_cell_potential_metric_adjoint,
    enclose_local_cell_tail,
    realize_local_cell_galerkin_potential,
)
from ptyrodactyl.galerkin.local_cell_certification import (
    certify_local_cell_galerkin_potential,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.local_cell_types import (
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellPotentialRealization,
    GalerkinLocalCellTailEnclosure,
    GalerkinLocalCellTailFailure,
    LocalCellPotential3D,
    create_local_cell_potential_3d,
)
from tests._galerkin_target_fixture import checked_acquisition

_PROVENANCE = "7" * 64
_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _support(
    indices: Tuple[Tuple[int, int, int], ...],
) -> GalerkinProductSupport:
    """Create a product-valid support with a singleton state space."""
    interaction = jnp.asarray(indices, dtype=jnp.int64)
    zero = jnp.zeros((1, 3), dtype=jnp.int64)
    shell = tuple(
        (first, second, third)
        for first in range(-1, 2)
        for second in range(-1, 2)
        for third in range(-1, 2)
    )
    work_values = tuple(sorted(set(indices) | set(shell)))
    work = jnp.asarray(work_values, dtype=jnp.int64)
    absorber = jnp.asarray(shell, dtype=jnp.int64)
    maxima = np.max(np.abs(np.asarray(work)), axis=0)
    work_shape = tuple(
        1 if maximum == 0 else 2 * int(maximum) + 3 for maximum in maxima
    )
    support: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=zero,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=work_shape,
    )
    return support


def _potential(
    values: jax.Array | np.ndarray,
    *,
    cell_size: Tuple[float, float, float] = (0.5, 0.75, 1.25),
    cell_center_origin: Tuple[float, float, float] = (
        0.125,
        -0.375,
        0.625,
    ),
    producer_bandwidth: float = 1.0e6,
    reference_value: float = 0.0,
    reference_semantics: str = (
        "declared local-cell realization test reference"
    ),
) -> LocalCellPotential3D:
    """Create one complete periodic local-cell voltage field."""
    nz, ny, nx = values.shape
    box_size = (
        nx * cell_size[0],
        ny * cell_size[1],
        nz * cell_size[2],
    )
    potential: LocalCellPotential3D = create_local_cell_potential_3d(
        values,
        cell_size=cell_size,
        box_size=box_size,
        cell_center_origin=cell_center_origin,
        reference_value=reference_value,
        reference_semantics=reference_semantics,
        producer="local-cell-realization-test-v1",
        provenance_hash=_PROVENANCE,
        producer_coefficient_normalization="diagnostic producer transform",
        producer_bandwidth=producer_bandwidth,
    )
    return potential


def _realize(
    potential: LocalCellPotential3D,
    support: GalerkinProductSupport,
) -> GalerkinLocalCellPotentialRealization:
    """Realize through one independently checked support artifact."""
    eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
    )
    realization: GalerkinLocalCellPotentialRealization = (
        realize_local_cell_galerkin_potential(potential, eligibility)
    )
    return realization


def _axis_cell_integral(
    mode: int,
    lower: float,
    upper: float,
    length: float,
) -> complex:
    """Integrate one Fourier exponential over one physical cell interval."""
    if mode == 0:
        return complex(upper - lower)
    frequency = mode / length
    return (
        np.exp(-2.0j * np.pi * frequency * upper)
        - np.exp(-2.0j * np.pi * frequency * lower)
    ) / (-2.0j * np.pi * frequency)


def _direct_cell_integral_coefficients(
    potential: LocalCellPotential3D,
    indices: jax.Array,
) -> np.ndarray:
    """Integrate every rectangular cell without using a DFT or sinc."""
    values = np.asarray(potential.cell_values)
    nz, ny, nx = values.shape
    counts = (nx, ny, nz)
    lengths = potential.box_size
    widths = tuple(length / count for length, count in zip(lengths, counts))
    origin = potential.cell_center_origin
    box_volume = float(np.prod(lengths))
    coefficients: list[complex] = []
    for mode in np.asarray(indices, dtype=np.int64):
        coefficient = 0.0j
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    cell_index = (ix, iy, iz)
                    integral = 1.0 + 0.0j
                    for axis in range(3):
                        center = origin[axis] + widths[axis] * cell_index[axis]
                        integral *= _axis_cell_integral(
                            int(mode[axis]),
                            center - 0.5 * widths[axis],
                            center + 0.5 * widths[axis],
                            lengths[axis],
                        )
                    coefficient += values[iz, iy, ix] * integral
        coefficients.append(coefficient / box_volume)
    return np.asarray(coefficients, dtype=np.complex128)


def _direct_metric_adjoint(
    potential: LocalCellPotential3D,
    indices: jax.Array,
    cotangent: jax.Array,
) -> np.ndarray:
    """Evaluate the formal rounded-map formula at all cell centers."""
    nz, ny, nx = potential.cell_values.shape
    counts = np.asarray((nx, ny, nz), dtype=np.float64)
    lengths = np.asarray(potential.box_size, dtype=np.float64)
    widths = lengths / counts
    origin = np.asarray(potential.cell_center_origin, dtype=np.float64)
    modes = np.asarray(indices, dtype=np.float64)
    factors = np.prod(np.sinc(modes / counts[None, :]), axis=-1)
    # Enforce the mathematical values independently of libm near integer q.
    symbolic_zero = np.any(
        (modes != 0.0) & (np.mod(modes, counts[None, :]) == 0.0),
        axis=-1,
    )
    factors[symbolic_zero] = 0.0
    frequencies = modes / lengths[None, :]
    result = np.empty((nz, ny, nx), dtype=np.float64)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                center = origin + widths * np.asarray((ix, iy, iz))
                phase = np.exp(2.0j * np.pi * (frequencies @ center))
                result[iz, iy, ix] = np.real(
                    np.sum(np.asarray(cotangent) * factors * phase)
                ) / float(np.prod(lengths))
    return result


class TestLocalCellRealization:
    """Verify rounded LVT.7 and its formal physical-metric adjoint.

    :see: :func:`ptyrodactyl.galerkin.\
apply_local_cell_potential_metric_adjoint`
    :see: :func:`ptyrodactyl.galerkin.realize_local_cell_galerkin_potential`
    """

    def test_constant_field_and_symbolic_qn_modes_are_exact(self) -> None:
        """Preserve the mean and make every nonzero uniform-field mode zero."""
        values = jnp.full((2, 3, 4), 7.25, dtype=jnp.float64)
        indices = (
            (-8, 0, 0),
            (-4, 0, 0),
            (-1, 0, 0),
            (0, -6, 0),
            (0, -3, 0),
            (0, 0, -4),
            (0, 0, -2),
            (0, 0, 0),
            (0, 0, 2),
            (0, 0, 4),
            (0, 3, 0),
            (0, 6, 0),
            (1, 0, 0),
            (4, 0, 0),
            (8, 0, 0),
        )
        realization = _realize(_potential(values), _support(indices))
        jax.block_until_ready(realization)
        coefficients = np.asarray(realization.voltage_coefficients)
        zero_position = indices.index((0, 0, 0))

        assert coefficients[zero_position] == 7.25 + 0.0j
        assert np.array_equal(
            np.delete(coefficients, zero_position),
            np.zeros(len(indices) - 1, dtype=np.complex128),
        )
        assert realization.voltage_coefficients.dtype == jnp.complex128
        assert realization.coefficient_error_bounds.dtype == jnp.float64

    def test_prescaled_fft_keeps_max_finite_constant_zero_mode(self) -> None:
        """Avoid overflow in an unnormalized transform's zero-mode sum."""
        maximum = jnp.finfo(jnp.float64).max
        values = jnp.full((2, 3, 4), maximum, dtype=jnp.float64)
        realization = _realize(
            _potential(values),
            _support(((0, 0, 0),)),
        )
        jax.block_until_ready(realization)

        coefficient = realization.voltage_coefficients[0]
        assert jnp.isfinite(coefficient)
        assert 0.0 < jnp.real(coefficient) <= maximum
        assert jnp.imag(coefficient) == 0.0
        assert (
            jnp.abs(coefficient - maximum)
            <= (realization.coefficient_error_bounds[0])
        )

    def test_non_power_of_two_max_cotangent_adjoint_stays_finite(self) -> None:
        """Apply the combined fixed transpose gain without a large product."""
        potential = _potential(jnp.zeros((2, 3, 4), dtype=jnp.float64))
        realization = _realize(potential, _support(((0, 0, 0),)))
        maximum = jnp.finfo(jnp.float64).max
        gradient = apply_local_cell_potential_metric_adjoint(
            realization,
            jnp.asarray([maximum + 0.0j], dtype=jnp.complex128),
        )
        expected = float(maximum) / float(np.prod(potential.box_size))

        assert np.all(jnp.isfinite(gradient))
        assert_allclose(gradient, expected, rtol=4.0e-14, atol=0.0)

    def test_huge_whole_box_origin_shift_has_exact_canonical_phase(
        self,
    ) -> None:
        """Reduce a finite ``2**1023`` box shift before trigonometry."""
        huge_shift = float.fromhex("0x1.0000000000000p+1023")
        values = jnp.asarray([[[1.0, -2.0, 4.0]]], dtype=jnp.float64)
        indices = ((-1, 0, 0), (0, 0, 0), (1, 0, 0))
        shifted = _potential(
            values,
            cell_size=(1.0 / 3.0, 1.0, 1.0),
            cell_center_origin=(huge_shift, 0.0, 0.0),
        )
        canonical = _potential(
            values,
            cell_size=(1.0 / 3.0, 1.0, 1.0),
            cell_center_origin=(0.0, 0.0, 0.0),
        )
        shifted_result = _realize(shifted, _support(indices))
        canonical_result = _realize(canonical, _support(indices))

        assert shifted.cell_center_origin == (0.0, 0.0, 0.0)
        assert np.array_equal(
            np.asarray(shifted_result.voltage_coefficients),
            np.asarray(canonical_result.voltage_coefficients),
        )
        assert np.all(jnp.isfinite(shifted_result.voltage_coefficients))

    def test_tiny_box_symbolic_sinc_zero_bypasses_phase(self) -> None:
        """Avoid ``0 * NaN`` when reciprocal division would overflow."""
        subnormal_length = float.fromhex("0x0.0000000000001p-1022")
        potential = _potential(
            jnp.asarray([[[2.0]]], dtype=jnp.float64),
            cell_size=(subnormal_length, 1.0, 1.0),
            cell_center_origin=(subnormal_length, 0.0, 0.0),
        )
        indices = jnp.asarray(
            ((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
            dtype=jnp.int64,
        )
        coefficients = _local_cell_coefficients_from_full_grid(
            jnp.asarray([[[2.0 + 0.0j]]], dtype=jnp.complex128),
            indices,
            _origin_cycle_fractions(potential),
        )

        assert potential.cell_center_origin == (0.0, 0.0, 0.0)
        assert np.array_equal(
            np.asarray(coefficients),
            np.asarray([0.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j]),
        )

    def test_shifted_anisotropic_single_cell_matches_direct_integrals(
        self,
    ) -> None:
        """Carry one localized cell through all three physical coordinates."""
        values = jnp.zeros((2, 3, 4), dtype=jnp.float64).at[1, 2, 3].set(5.5)
        indices = (
            (-3, -2, -1),
            (-2, 1, -3),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (2, -1, 3),
            (3, 2, 1),
        )
        potential = _potential(
            values,
            cell_size=(0.375, 0.625, 1.125),
            cell_center_origin=(0.1875, -0.3125, 0.5625),
        )
        support = _support(indices)
        realization = _realize(potential, support)
        expected = _direct_cell_integral_coefficients(
            potential,
            support.interaction_indices,
        )

        assert_allclose(
            realization.voltage_coefficients,
            expected,
            rtol=4.0e-14,
            atol=4.0e-14,
        )
        assert np.all(
            np.abs(np.asarray(realization.voltage_coefficients) - expected)
            <= np.asarray(realization.coefficient_error_bounds)
        )

    def test_box_binding_rejects_one_binary64_bit_mismatch(self) -> None:
        """Compare stored box identities by bits rather than arithmetic."""
        potential = _potential(jnp.ones((2, 3, 4), dtype=jnp.float64))
        support = _support(((0, 0, 0),))
        mismatched_box = (
            float(np.nextafter(potential.box_size[0], np.inf)),
            *potential.box_size[1:],
        )
        eligibility = checked_acquisition(
            support,
            mismatched_box,
            terminal_axis=2,
        )

        with pytest.raises(_RUNTIME_ERRORS, match="exactly match"):
            realization = realize_local_cell_galerkin_potential(
                potential,
                eligibility,
            )
            jax.block_until_ready(realization)

    def test_odd_even_and_beyond_nyquist_modes_remain_unwrapped(self) -> None:
        """Use modular DFT bins without wrapping sinc or origin phases."""
        values = jnp.asarray(
            [
                [
                    [1.0, -2.0, 3.5, 0.25],
                    [4.0, 0.5, -1.0, 2.0],
                    [3.0, 1.0, -0.5, 5.0],
                ],
                [
                    [-1.5, 2.5, 0.75, 4.0],
                    [0.0, -3.0, 2.0, 1.0],
                    [1.25, 3.0, -2.5, 0.5],
                ],
            ],
            dtype=jnp.float64,
        )
        indices = (
            (-5, 0, 0),
            (-4, 0, 0),
            (-3, 0, 0),
            (-2, 0, 0),
            (-1, 0, 0),
            (0, -4, 0),
            (0, -3, 0),
            (0, -2, 0),
            (0, -1, 0),
            (0, 0, -3),
            (0, 0, -2),
            (0, 0, -1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
        )
        potential = _potential(values)
        support = _support(indices)
        realization = _realize(potential, support)
        expected = _direct_cell_integral_coefficients(
            potential,
            support.interaction_indices,
        )

        assert_allclose(
            realization.voltage_coefficients,
            expected,
            rtol=8.0e-14,
            atol=8.0e-14,
        )
        assert np.all(
            np.abs(np.asarray(realization.voltage_coefficients) - expected)
            <= np.asarray(realization.coefficient_error_bounds)
        )
        positions = {mode: position for position, mode in enumerate(indices)}
        assert realization.voltage_coefficients[positions[(4, 0, 0)]] == 0.0
        assert realization.voltage_coefficients[positions[(0, 3, 0)]] == 0.0
        assert realization.voltage_coefficients[positions[(0, 0, 2)]] == 0.0
        assert not np.isclose(
            realization.voltage_coefficients[positions[(1, 0, 0)]],
            realization.voltage_coefficients[positions[(5, 0, 0)]],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_stored_coefficients_are_exactly_hermitian(self) -> None:
        """Store each ordinary signed pair by exact conjugation."""
        values = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4) - 7.0
        indices = (
            (3, 2, 1),
            (0, 0, 0),
            (-1, 0, 2),
            (-3, -2, -1),
            (1, 0, -2),
        )
        realization = _realize(_potential(values), _support(indices))
        coefficients = np.asarray(realization.voltage_coefficients)
        positions = {mode: position for position, mode in enumerate(indices)}

        for mode, position in positions.items():
            opposite: Tuple[int, int, int] = (-mode[0], -mode[1], -mode[2])
            assert coefficients[position] == np.conj(
                coefficients[positions[opposite]]
            )
        assert coefficients[positions[(0, 0, 0)]].imag == 0.0

    def test_producer_bandwidth_does_not_change_coefficients(self) -> None:
        """Keep producer-band metadata out of support and coefficient math."""
        values = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4)
        indices = ((-5, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (5, 0, 0))
        support = _support(indices)
        narrow = _realize(
            _potential(values, producer_bandwidth=1.0e-12),
            support,
        )
        broad = _realize(
            _potential(values, producer_bandwidth=1.0e12),
            support,
        )

        assert narrow.local_potential.producer_bandwidth != (
            broad.local_potential.producer_bandwidth
        )
        assert np.array_equal(
            np.asarray(narrow.voltage_coefficients),
            np.asarray(broad.voltage_coefficients),
        )

    def test_reference_metadata_does_not_shift_cell_values(self) -> None:
        """Record the declared reference without subtracting any array mean."""
        values = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4) + 3.0
        support = _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0)))
        first_potential = _potential(
            values,
            reference_value=-12.5,
            reference_semantics="declared first physical reference",
        )
        second_potential = _potential(
            values,
            reference_value=43.0,
            reference_semantics="declared second physical reference",
        )
        first = _realize(first_potential, support)
        second = _realize(second_potential, support)

        assert np.array_equal(
            np.asarray(first.local_potential.cell_values),
            np.asarray(values),
        )
        assert np.array_equal(
            np.asarray(second.local_potential.cell_values),
            np.asarray(values),
        )
        assert np.array_equal(
            np.asarray(first.voltage_coefficients),
            np.asarray(second.voltage_coefficients),
        )

    def test_forged_acquisition_result_leaves_have_no_influence(self) -> None:
        """Rebuild from the manifest rather than aggregate claims."""
        potential = _potential(jnp.arange(24).reshape(2, 3, 4))
        support = _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0)))
        eligibility = checked_acquisition(
            support,
            potential.box_size,
            terminal_axis=2,
        )
        forged = eqx.tree_at(
            lambda result: result.status,
            eligibility,
            jnp.asarray(-123, dtype=jnp.int32),
        )
        forged = eqx.tree_at(
            lambda result: result.support_eligible,
            forged,
            jnp.asarray(False),
        )
        forged = eqx.tree_at(
            lambda result: result.direct_transfers_represented,
            forged,
            jnp.asarray(False),
        )
        canonical = realize_local_cell_galerkin_potential(
            potential,
            eligibility,
        )
        rebuilt = realize_local_cell_galerkin_potential(
            potential,
            forged,
        )

        assert np.array_equal(
            np.asarray(rebuilt.voltage_coefficients),
            np.asarray(canonical.voltage_coefficients),
        )
        assert bool(rebuilt.support_eligibility.support_eligible)

    def test_triangle_evidence_contains_stored_subnormal_cells(self) -> None:
        """Embed FTZ-sensitive exact cells before any backend reduction."""
        subnormal = float.fromhex("0x0.0000000000001p-1022")
        values = jnp.full((2, 3, 4), subnormal, dtype=jnp.float64)
        potential = _potential(values)
        realization = _realize(
            potential,
            _support(((0, 0, 0),)),
        )
        expected = _direct_cell_integral_coefficients(
            potential,
            realization.support.interaction_indices,
        )
        jax.block_until_ready(realization)

        assert (
            realization.coefficient_error_bounds[0]
            >= jnp.finfo(jnp.float64).tiny
        )
        assert (
            np.abs(
                np.asarray(realization.voltage_coefficients)[0] - expected[0]
            )
            <= np.asarray(realization.coefficient_error_bounds)[0]
        )

    def test_subnormal_signed_pair_is_hermitian_and_contained(self) -> None:
        """Keep an extreme nonzero pair sound across safe pair averaging."""
        tiny = jnp.finfo(jnp.float64).tiny
        values = jnp.asarray([[[tiny, 0.0]]], dtype=jnp.float64)
        potential = _potential(
            values,
            cell_size=(0.5, 1.0, 1.0),
            cell_center_origin=(0.25, 0.0, 0.0),
        )
        support = _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0)))
        realization = _realize(potential, support)
        expected = _direct_cell_integral_coefficients(
            potential,
            support.interaction_indices,
        )
        coefficients = np.asarray(realization.voltage_coefficients)

        assert abs(coefficients[0]) < np.finfo(np.float64).tiny
        assert coefficients[0] == np.conj(coefficients[2])
        assert np.all(
            np.abs(coefficients - expected)
            <= np.asarray(realization.coefficient_error_bounds)
        )

    def test_triangle_helper_embeds_subnormal_components_before_abs(
        self,
    ) -> None:
        """Prevent DAZ from erasing stored coefficient-side evidence."""
        subnormal = float.fromhex("0x0.0000000000001p-1022")
        potential = _potential(jnp.zeros((1, 1, 1), dtype=jnp.float64))
        bound = _coefficient_error_bounds(
            potential,
            jnp.asarray(
                [complex(subnormal, subnormal)],
                dtype=jnp.complex128,
            ),
        )[0]

        assert bound >= 2.0 * jnp.finfo(jnp.float64).tiny

    def test_forged_local_cell_sources_fail_independent_reconstruction(
        self,
    ) -> None:
        """Reject mutated dynamic and static fields before any FFT."""
        potential = _potential(jnp.ones((2, 3, 4), dtype=jnp.float64))
        eligibility = checked_acquisition(
            _support(((0, 0, 0),)),
            potential.box_size,
            terminal_axis=2,
        )
        complex_values = eqx.tree_at(
            lambda item: item.cell_values,
            potential,
            potential.cell_values.astype(jnp.complex128) + 1.0j,
        )
        forged_sources = (
            (complex_values, "real voltages"),
            (
                dataclasses.replace(
                    potential,
                    cell_size=(9.0, *potential.cell_size[1:]),
                ),
                "must equal",
            ),
            (
                dataclasses.replace(
                    potential,
                    cell_value_semantics="forged point samples",
                ),
                "noncanonical static structure",
            ),
            (
                dataclasses.replace(
                    potential,
                    cell_center_origin=(jnp.inf, 0.0, 0.0),
                ),
                "finite",
            ),
        )

        for forged, match in forged_sources:
            with pytest.raises(_RUNTIME_ERRORS, match=match):
                realization = realize_local_cell_galerkin_potential(
                    forged,
                    eligibility,
                )
                jax.block_until_ready(realization)

    def test_alias_collisions_preserve_the_physical_metric_adjoint(
        self,
    ) -> None:
        """Scatter-add all colliding unwrapped modes in the formal adjoint."""
        values = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4) - 9.0
        direction = jnp.asarray(
            [
                [
                    [0.5, -1.0, 0.25, 2.0],
                    [1.0, 0.0, -0.5, 0.75],
                    [-1.5, 1.25, 0.5, -0.25],
                ],
                [
                    [0.125, 0.375, -0.625, 1.5],
                    [0.25, -0.75, 2.0, -1.0],
                    [1.75, -0.125, 0.875, 0.625],
                ],
            ],
            dtype=jnp.float64,
        )
        indices = (
            (-5, 0, 0),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (5, 0, 0),
        )
        cotangent = jnp.asarray(
            [
                0.5 + 1.25j,
                -2.0 + 0.75j,
                0.625 - 0.5j,
                1.5 + 2.25j,
                -0.875 + 1.0j,
            ],
            dtype=jnp.complex128,
        )
        potential = _potential(values)
        support = _support(indices)
        eligibility = checked_acquisition(
            support,
            potential.box_size,
            terminal_axis=2,
        )

        def coefficient_map(dynamic_values: jax.Array) -> jax.Array:
            candidate = eqx.tree_at(
                lambda item: item.cell_values,
                potential,
                dynamic_values,
            )
            return realize_local_cell_galerkin_potential(
                candidate,
                eligibility,
            ).voltage_coefficients

        coefficients, differential = jax.jvp(
            coefficient_map,
            (values,),
            (direction,),
        )
        realization = _realize(potential, support)
        gradient = apply_local_cell_potential_metric_adjoint(
            realization,
            cotangent,
        )
        euclidean_gradient = jax.grad(
            lambda dynamic_values: jnp.real(
                jnp.vdot(cotangent, coefficient_map(dynamic_values))
            )
        )(values)
        cell_volume = np.prod(potential.box_size) / values.size
        physical_dot = cell_volume * jnp.sum(direction * gradient)
        coefficient_dot = jnp.real(jnp.vdot(cotangent, differential))
        expected_gradient = _direct_metric_adjoint(
            potential,
            support.interaction_indices,
            cotangent,
        )

        assert_allclose(coefficients, realization.voltage_coefficients)
        assert_allclose(
            gradient, expected_gradient, rtol=8.0e-14, atol=8.0e-14
        )
        assert_allclose(
            gradient,
            euclidean_gradient / cell_volume,
            rtol=8.0e-14,
            atol=8.0e-14,
        )
        compiled_gradient = jax.jit(
            lambda dynamic_cotangent: (
                apply_local_cell_potential_metric_adjoint(
                    realization,
                    dynamic_cotangent,
                )
            )
        )(cotangent)
        assert_allclose(
            compiled_gradient,
            gradient,
            rtol=8.0e-14,
            atol=8.0e-14,
        )
        compiled_realization = jax.jit(
            lambda dynamic_values: realize_local_cell_galerkin_potential(
                eqx.tree_at(
                    lambda item: item.cell_values,
                    potential,
                    dynamic_values,
                ),
                eligibility,
            )
        )(values)
        replay_gradient = apply_local_cell_potential_metric_adjoint(
            compiled_realization,
            cotangent,
        )
        assert_allclose(
            replay_gradient,
            expected_gradient,
            rtol=8.0e-14,
            atol=8.0e-14,
        )
        assert_allclose(
            physical_dot, coefficient_dot, rtol=8.0e-14, atol=8.0e-14
        )

    def test_anisotropic_metric_volume_avoids_intermediate_overflow(
        self,
    ) -> None:
        """Round the exact rational volume only after multiplying all axes."""
        potential = _potential(
            jnp.asarray([[[2.0]]], dtype=jnp.float64),
            cell_size=(1.0e200, 1.0e200, 1.0e-100),
            cell_center_origin=(0.0, 0.0, 0.0),
        )
        realization = _realize(potential, _support(((0, 0, 0),)))
        gradient = apply_local_cell_potential_metric_adjoint(
            realization,
            jnp.asarray([1.0 + 0.0j], dtype=jnp.complex128),
        )

        assert_allclose(gradient, [[[1.0e-300]]], rtol=2.0e-14, atol=0.0)

    @pytest.mark.parametrize(
        "cell_size",
        [
            (1.0e200, 1.0e200, 1.0e200),
            (1.0e-200, 1.0e-200, 1.0e-200),
        ],
    )
    def test_metric_volume_rejects_binary64_endpoints(
        self,
        cell_size: Tuple[float, float, float],
    ) -> None:
        """Fail closed when rounded ``DeltaV`` is infinity or zero."""
        potential = _potential(
            jnp.asarray([[[1.0]]], dtype=jnp.float64),
            cell_size=cell_size,
            cell_center_origin=(0.0, 0.0, 0.0),
        )

        with pytest.raises(ValueError, match="physical cell volume"):
            _physical_cell_volume(potential)

    def test_adjoint_ignores_forged_coefficient_storage_leaves(self) -> None:
        """Rebuild map inputs instead of trusting coefficient payloads."""
        realization = _realize(
            _potential(jnp.ones((2, 3, 4), dtype=jnp.float64)),
            _support(((0, 0, 0),)),
        )
        forged = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            realization,
            jnp.asarray([2.0 + 0.25j], dtype=jnp.complex128),
        )

        cotangent = jnp.ones((1,), dtype=jnp.complex128)
        canonical_gradient = apply_local_cell_potential_metric_adjoint(
            realization,
            cotangent,
        )
        forged_gradient = apply_local_cell_potential_metric_adjoint(
            forged,
            cotangent,
        )

        assert_allclose(
            forged_gradient, canonical_gradient, rtol=0.0, atol=0.0
        )

    def test_coefficient_map_is_jittable_and_jvp_linear(self) -> None:
        """Compile LVT.7 and retain its exact dynamic-value tangent map."""
        values = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4) - 4.0
        direction = jnp.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
        indices = ((-5, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (5, 0, 0))
        potential = _potential(values)
        support = _support(indices)
        eligibility = checked_acquisition(
            support,
            potential.box_size,
            terminal_axis=2,
        )

        def coefficient_map(dynamic_values: jax.Array) -> jax.Array:
            candidate = eqx.tree_at(
                lambda item: item.cell_values,
                potential,
                dynamic_values,
            )
            return realize_local_cell_galerkin_potential(
                candidate,
                eligibility,
            ).voltage_coefficients

        eager = coefficient_map(values)
        compiled = jax.jit(coefficient_map)(values)
        _, differential = jax.jvp(
            coefficient_map,
            (values,),
            (direction,),
        )
        _, error_differential = jax.jvp(
            lambda dynamic_values: (
                realize_local_cell_galerkin_potential(
                    eqx.tree_at(
                        lambda item: item.cell_values,
                        potential,
                        dynamic_values,
                    ),
                    eligibility,
                ).coefficient_error_bounds
            ),
            (values,),
            (direction,),
        )
        direction_coefficients = coefficient_map(direction)
        realization = _realize(potential, support)
        expected = _direct_cell_integral_coefficients(
            potential,
            support.interaction_indices,
        )

        assert compiled.dtype == jnp.complex128
        assert differential.dtype == jnp.complex128
        assert np.array_equal(
            np.asarray(error_differential),
            np.zeros_like(np.asarray(error_differential)),
        )
        assert_allclose(compiled, eager, rtol=3.0e-14, atol=3.0e-14)
        assert np.all(
            np.abs(np.asarray(eager) - expected)
            <= np.asarray(realization.coefficient_error_bounds)
        )
        assert np.all(
            np.abs(np.asarray(compiled) - expected)
            <= np.asarray(realization.coefficient_error_bounds)
        )
        assert_allclose(
            differential, direction_coefficients, rtol=3.0e-14, atol=3.0e-14
        )

    @pytest.mark.parametrize(
        ("cotangent", "match"),
        [
            (jnp.ones((1, 1), dtype=jnp.complex128), "must be 1D"),
            (jnp.ones((2,), dtype=jnp.complex128), "must match"),
            (jnp.asarray([jnp.nan + 0.0j]), "finite"),
        ],
    )
    def test_adjoint_rejects_invalid_cotangents(
        self,
        cotangent: jax.Array,
        match: str,
    ) -> None:
        """Reject malformed or nonfinite coefficient covectors."""
        realization = _realize(
            _potential(jnp.ones((2, 3, 4), dtype=jnp.float64)),
            _support(((0, 0, 0),)),
        )

        with pytest.raises(_RUNTIME_ERRORS, match=match):
            gradient = apply_local_cell_potential_metric_adjoint(
                realization,
                cotangent,
            )
            jax.block_until_ready(gradient)


class TestLocalCellTailEnclosure:
    """Verify authenticated outward LVT.9 full-tail evidence.

    :see: :func:`ptyrodactyl.galerkin.enclose_local_cell_tail`
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _certified_step(
        *,
        maximum_direct_terms: int = 4,
    ) -> GalerkinLocalCellPotentialRealization:
        """Certify a two-cell unit step on ``K = {-1, 0, 1}``."""
        potential = _potential(
            jnp.asarray([[[1.0, -1.0]]], dtype=jnp.float64),
            cell_size=(1.0, 1.0, 1.0),
            cell_center_origin=(0.0, 0.0, 0.0),
        )
        realization = _realize(
            potential,
            _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0))),
        )
        certified: GalerkinLocalCellPotentialRealization = (
            certify_local_cell_galerkin_potential(
                realization,
                maximum_direct_terms=maximum_direct_terms,
            )
        )
        return certified

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _enclosed_step() -> GalerkinLocalCellTailEnclosure:
        """Build and cache the common finite LVT.9 regression carrier."""
        enclosure: GalerkinLocalCellTailEnclosure = enclose_local_cell_tail(
            TestLocalCellTailEnclosure._certified_step()
        )
        return enclosure

    def test_two_cell_step_encloses_exact_infinite_parseval_tail(self) -> None:
        r"""Enclose ``2 - 16 / pi**2`` and count both signed modes."""
        enclosure = self._enclosed_step()
        pi_digits = 314159265358979323846264338327950288419716939937510
        pi_lower = Fraction(pi_digits, 10**50)
        pi_upper = Fraction(pi_digits + 1, 10**50)
        exact_squared_lower = Fraction(2) - Fraction(16) / (pi_lower**2)
        exact_squared_upper = Fraction(2) - Fraction(16) / (pi_upper**2)
        squared_lower = Fraction.from_float(
            float(enclosure.squared_tail_lower_bound)
        )
        squared_upper = Fraction.from_float(
            float(enclosure.squared_tail_upper_bound)
        )
        midpoint_norm = math.sqrt(
            float((exact_squared_lower + exact_squared_upper) / 2)
        )

        assert isinstance(enclosure, GalerkinLocalCellTailEnclosure)
        assert bool(enclosure.finite_enclosure)
        assert enclosure.failure is GalerkinLocalCellTailFailure.NONE
        assert enclosure.parent_certificate_failure is (
            GalerkinLocalCellCertificateFailure.NONE
        )
        assert squared_lower <= exact_squared_lower
        assert squared_upper >= exact_squared_upper
        assert float(enclosure.tail_l2_lower_bound) <= midpoint_norm
        assert float(enclosure.tail_l2_upper_bound) >= midpoint_norm
        assert len(enclosure.parent_certificate_digest) == 64
        assert len(enclosure.tail_enclosure_digest) == 64
        replay = _authenticate_local_cell_tail(enclosure)
        assert replay.tail_enclosure_digest == enclosure.tail_enclosure_digest

    def test_constant_field_retaining_zero_has_exact_zero_tail(self) -> None:
        """Return four exact zero endpoints for a globally constant field."""
        potential = _potential(
            jnp.full((2, 2, 2), 3.25, dtype=jnp.float64),
            cell_size=(0.5, 0.75, 1.25),
            cell_center_origin=(0.0, 0.0, 0.0),
        )
        realized = _realize(potential, _support(((0, 0, 0),)))
        certified = certify_local_cell_galerkin_potential(
            realized,
            maximum_direct_terms=8,
        )
        enclosure = enclose_local_cell_tail(certified)

        assert bool(enclosure.finite_enclosure)
        assert enclosure.squared_tail_lower_bound == 0.0
        assert enclosure.squared_tail_upper_bound == 0.0
        assert enclosure.tail_l2_lower_bound == 0.0
        assert enclosure.tail_l2_upper_bound == 0.0

    def test_mean_zero_step_retaining_zero_has_exact_total_energy_tail(
        self,
    ) -> None:
        """Enclose the independent exact-Fraction value two for ``K={0}``."""
        potential = _potential(
            jnp.asarray([[[1.0, -1.0]]], dtype=jnp.float64),
            cell_size=(1.0, 1.0, 1.0),
            cell_center_origin=(0.0, 0.0, 0.0),
        )
        realized = _realize(potential, _support(((0, 0, 0),)))
        certified = certify_local_cell_galerkin_potential(
            realized,
            maximum_direct_terms=2,
        )
        enclosure = enclose_local_cell_tail(certified)
        exact_squared = Fraction(2)

        assert (
            Fraction.from_float(float(enclosure.squared_tail_lower_bound))
            <= exact_squared
        )
        assert (
            Fraction.from_float(float(enclosure.squared_tail_upper_bound))
            >= exact_squared
        )
        assert float(enclosure.tail_l2_lower_bound) <= math.sqrt(2.0)
        assert float(enclosure.tail_l2_upper_bound) >= math.sqrt(2.0)

    def test_triangle_route_is_not_tail_evidence(self) -> None:
        """Reject stopped fallback errors before any LVT.9 arithmetic."""
        potential = _potential(
            jnp.asarray([[[1.0, -1.0]]], dtype=jnp.float64),
            cell_size=(1.0, 1.0, 1.0),
            cell_center_origin=(0.0, 0.0, 0.0),
        )
        triangle = _realize(potential, _support(((0, 0, 0),)))

        with pytest.raises(ValueError, match="direct local-cell evidence"):
            enclose_local_cell_tail(triangle)

    def test_host_boundary_rejects_traced_realization(self) -> None:
        """Keep certificate replay and exact Fraction arithmetic off traces."""
        certified = self._certified_step()

        with pytest.raises(ValueError, match="requires concrete host values"):
            result = jax.jit(enclose_local_cell_tail)(certified)
            jax.block_until_ready(result)

    def test_parent_typed_noncertificate_propagates_unbounded_tail(
        self,
    ) -> None:
        """Retain the parent's exact failure and never use fallback errors."""
        failed_parent = self._certified_step(maximum_direct_terms=3)
        parent_certificate = failed_parent.coefficient_certificate
        assert parent_certificate is not None
        enclosure = enclose_local_cell_tail(failed_parent)

        assert parent_certificate.failure is (
            GalerkinLocalCellCertificateFailure.WORK_BUDGET_EXCEEDED
        )
        assert not bool(enclosure.finite_enclosure)
        assert enclosure.failure is (
            GalerkinLocalCellTailFailure.PARENT_CERTIFICATE_NOT_FINITE
        )
        assert enclosure.parent_certificate_failure is (
            GalerkinLocalCellCertificateFailure.WORK_BUDGET_EXCEEDED
        )
        assert enclosure.squared_tail_lower_bound == 0.0
        assert jnp.isposinf(enclosure.squared_tail_upper_bound)
        assert enclosure.tail_l2_lower_bound == 0.0
        assert jnp.isposinf(enclosure.tail_l2_upper_bound)
        _authenticate_local_cell_tail(enclosure)

    def test_subtraction_intersects_only_after_exact_outward_difference(
        self,
    ) -> None:
        """Account for a negative raw lower and reject a negative raw upper."""
        interval = _outward_lvt9_subtraction(
            Fraction(1),
            Fraction(9, 10),
            Fraction(11, 10),
        )
        contradiction = _outward_lvt9_subtraction(
            Fraction(1),
            Fraction(11, 10),
            Fraction(6, 5),
        )

        assert interval == (Fraction(0), Fraction(1, 10), True)
        assert contradiction == (Fraction(0), Fraction(0), False)

    def test_replay_rejects_dynamic_and_static_tail_forgery(self) -> None:
        """Treat public Equinox storage and its checksum as untrusted input."""
        enclosure = self._enclosed_step()
        forged_bound = eqx.tree_at(
            lambda item: item.squared_tail_upper_bound,
            enclosure,
            enclosure.squared_tail_upper_bound + 1.0,
        )
        forged_digest = dataclasses.replace(
            enclosure,
            tail_enclosure_digest="0" * 64,
        )
        rehashed_digest = _tail_enclosure_digest(
            forged_bound.parent_certificate_digest,
            float(forged_bound.squared_tail_lower_bound),
            float(forged_bound.squared_tail_upper_bound),
            float(forged_bound.tail_l2_lower_bound),
            float(forged_bound.tail_l2_upper_bound),
            finite_enclosure=bool(forged_bound.finite_enclosure),
            failure=forged_bound.failure,
            parent_certificate_failure=(
                forged_bound.parent_certificate_failure
            ),
        )
        rehashed_bound = dataclasses.replace(
            forged_bound,
            tail_enclosure_digest=rehashed_digest,
        )

        with pytest.raises(ValueError, match="does not match host replay"):
            _authenticate_local_cell_tail(forged_bound)
        with pytest.raises(ValueError, match="does not match host replay"):
            _authenticate_local_cell_tail(forged_digest)
        with pytest.raises(ValueError, match="does not match host replay"):
            _authenticate_local_cell_tail(rehashed_bound)
