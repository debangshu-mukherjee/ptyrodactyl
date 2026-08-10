"""Tests for :mod:`ptyrodactyl.galerkin.coefficient_certification`."""

import dataclasses
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple

from ptyrodactyl.galerkin import coefficient_certification
from ptyrodactyl.galerkin.coefficient_certification import (
    _voltage_operator_error_fraction,
    certify_galerkin_potential_realization,
    host_binary64_supported,
    rational_turn_exponential,
)
from ptyrodactyl.galerkin.realization import realize_galerkin_potential
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.potential_types import (
    Potential3D,
    create_potential_3d,
)
from ptyrodactyl.types.realization_types import (
    GalerkinPotentialCertificateFailure,
    GalerkinPotentialErrorRoute,
    GalerkinPotentialRealization,
)
from tests._galerkin_target_fixture import checked_acquisition

_PROVENANCE: str = "e" * 64


def _support(
    indices: Tuple[Tuple[int, int, int], ...],
) -> GalerkinProductSupport:
    """Create one product-valid singleton-state support."""
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
    work_shape = tuple(2 * int(maximum) + 3 for maximum in maxima)
    return create_galerkin_product_support(
        state_indices=zero,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=work_shape,
    )


def _potential(
    volume: jax.Array,
    *,
    origin: Tuple[float, float, float] = (-0.375, 0.2, 0.125),
) -> Potential3D:
    """Create one anisotropic non-power-of-two VC-1 potential."""
    nz, ny, nx = volume.shape
    voxel_size = (0.5, 0.7, 1.1)
    return create_potential_3d(
        volume,
        voxel_size=voxel_size,
        box_size=(
            nx * voxel_size[0],
            ny * voxel_size[1],
            nz * voxel_size[2],
        ),
        origin=origin,
        producer="coefficient-certificate-test-v1",
        provenance_hash=_PROVENANCE,
        coefficient_normalization="VC-1 mean DFT",
        band_limit=0.44,
    )


def _realization(
    volume: jax.Array,
    indices: Tuple[Tuple[int, int, int], ...],
    *,
    origin: Tuple[float, float, float] = (-0.375, 0.2, 0.125),
) -> GalerkinPotentialRealization:
    """Build one canonical production realization before refinement."""
    potential = _potential(volume, origin=origin)
    support = _support(indices)
    eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
    )
    return realize_galerkin_potential(potential, eligibility)


def _dense_coefficient(
    realization: GalerkinPotentialRealization,
    mode: Tuple[int, int, int],
) -> complex:
    """Evaluate one independent long-double-sized dense complex sum."""
    volume = np.asarray(realization.potential.volume, dtype=np.float64)
    nz, ny, nx = volume.shape
    ox, oy, oz = realization.potential.origin
    lx, ly, lz = realization.potential.box_size
    total = 0.0 + 0.0j
    for z_position in range(nz):
        for y_position in range(ny):
            for x_position in range(nx):
                turn = (
                    mode[0] * (x_position / nx + ox / lx)
                    + mode[1] * (y_position / ny + oy / ly)
                    + mode[2] * (z_position / nz + oz / lz)
                )
                total += volume[z_position, y_position, x_position] * np.exp(
                    -2.0j * np.pi * turn
                )
    return complex(total / volume.size)


class TestDirectCoefficientCertification:
    """Verify the bounded direct VC.17 host checker.

    :see: :func:`ptyrodactyl.galerkin.\
certify_galerkin_potential_realization`
    """

    def test_translated_anisotropic_nonpower_grid_and_pair(self) -> None:
        """Certify origin phase and exact signed conjugacy on a 2x3x5 grid."""
        nz, ny, nx = 2, 3, 5
        x = jnp.arange(nx, dtype=jnp.float64)[None, None, :]
        volume = jnp.broadcast_to(
            3.0 + 2.0 * jnp.cos(2.0 * jnp.pi * x / nx),
            (nz, ny, nx),
        )
        modes = ((-1, 0, 0), (0, 0, 0), (1, 0, 0))
        base = _realization(volume, modes)
        refined = certify_galerkin_potential_realization(
            base,
            maximum_direct_terms=1_000,
        )
        certificate = refined.coefficient_certificate
        assert certificate is not None

        assert refined.potential is base.potential
        assert refined.voltage_coefficients is base.voltage_coefficients
        assert refined.error_route is (
            GalerkinPotentialErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
        )
        assert certificate.failure is GalerkinPotentialCertificateFailure.NONE
        assert bool(certificate.finite_certificate)
        assert np.all(np.asarray(refined.coefficient_error_bounds) < 1.0e-12)
        np.testing.assert_array_equal(
            certificate.exact_coefficient_real_lower_bounds[0],
            certificate.exact_coefficient_real_lower_bounds[2],
        )
        np.testing.assert_array_equal(
            certificate.exact_coefficient_imag_lower_bounds[0],
            -certificate.exact_coefficient_imag_upper_bounds[2],
        )
        assert refined.voltage_coefficients[0] == jnp.conj(
            refined.voltage_coefficients[2]
        )

    def test_exact_zero_has_zero_rectangles_and_error(self) -> None:
        """Retain algebraic zero identities through the entire certificate."""
        base = _realization(
            jnp.zeros((2, 3, 5), dtype=jnp.float64),
            ((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
        )
        refined = certify_galerkin_potential_realization(
            base,
            maximum_direct_terms=1_000,
        )
        certificate = refined.coefficient_certificate
        assert certificate is not None

        np.testing.assert_array_equal(refined.coefficient_error_bounds, 0.0)
        np.testing.assert_array_equal(
            certificate.exact_coefficient_real_lower_bounds,
            0.0,
        )
        np.testing.assert_array_equal(
            certificate.exact_coefficient_real_upper_bounds,
            0.0,
        )
        np.testing.assert_array_equal(
            certificate.exact_coefficient_imag_lower_bounds,
            0.0,
        )
        np.testing.assert_array_equal(
            certificate.exact_coefficient_imag_upper_bounds,
            0.0,
        )
        assert refined.voltage_operator_error_bound == 0.0

    def test_perturbed_production_coefficient_is_still_enclosed(self) -> None:
        """Bound an arbitrary stored point without trusting its FFT route."""
        volume = jnp.arange(30, dtype=jnp.float64).reshape(2, 3, 5) / 7.0 - 1.0
        modes = ((-1, 0, 0), (0, 0, 0), (1, 0, 0))
        base = _realization(volume, modes)
        perturbation = jnp.asarray(
            [0.25 - 0.125j, -0.4 + 0.2j, 0.25 + 0.125j],
            dtype=jnp.complex128,
        )
        perturbed = dataclasses.replace(
            base,
            voltage_coefficients=base.voltage_coefficients + perturbation,
        )
        refined = certify_galerkin_potential_realization(
            perturbed,
            maximum_direct_terms=1_000,
        )
        oracle = np.asarray(
            [_dense_coefficient(refined, mode) for mode in modes]
        )
        observed = np.abs(np.asarray(refined.voltage_coefficients) - oracle)

        assert np.all(observed <= np.asarray(refined.coefficient_error_bounds))
        assert np.all(np.asarray(refined.coefficient_error_bounds) > 0.1)

    def test_work_budget_returns_typed_infinite_noncertificate(self) -> None:
        """Fail closed before performing an over-budget direct sum."""
        base = _realization(
            jnp.ones((2, 3, 5), dtype=jnp.float64),
            ((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
        )
        refined = certify_galerkin_potential_realization(
            base,
            maximum_direct_terms=10,
        )
        certificate = refined.coefficient_certificate
        assert certificate is not None

        assert certificate.failure is (
            GalerkinPotentialCertificateFailure.WORK_BUDGET_EXCEEDED
        )
        assert not bool(certificate.finite_certificate)
        assert np.all(np.isinf(refined.coefficient_error_bounds))
        assert np.isinf(refined.voltage_operator_error_bound)

    def test_host_binary64_probe_and_typed_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recognize this host and fail closed when its probe is absent."""
        assert host_binary64_supported()
        base = _realization(
            jnp.ones((2, 3, 5), dtype=jnp.float64),
            ((0, 0, 0),),
        )
        monkeypatch.setattr(
            coefficient_certification,
            "host_binary64_supported",
            lambda: False,
        )

        refined = certify_galerkin_potential_realization(
            base,
            maximum_direct_terms=100,
        )
        certificate = refined.coefficient_certificate
        assert certificate is not None
        assert certificate.failure is (
            GalerkinPotentialCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
        assert not bool(certificate.finite_certificate)
        assert np.all(np.isinf(refined.coefficient_error_bounds))
        assert np.isinf(refined.voltage_operator_error_bound)

    def test_acquisition_box_must_match_potential_exactly(self) -> None:
        """Reject an eligible artifact borrowed from a different box."""
        base = _realization(
            jnp.ones((2, 3, 5), dtype=jnp.float64),
            ((0, 0, 0),),
        )
        forged_potential = dataclasses.replace(
            base.potential,
            box_size=(
                float(np.nextafter(base.potential.box_size[0], np.inf)),
                base.potential.box_size[1],
                base.potential.box_size[2],
            ),
        )
        forged = dataclasses.replace(base, potential=forged_potential)

        with pytest.raises(ValueError, match="exactly match"):
            certify_galerkin_potential_realization(
                forged,
                maximum_direct_terms=100,
            )

    def test_traced_volume_is_rejected_at_host_boundary(self) -> None:
        """Reject certificate construction from inside a JAX trace."""
        base = _realization(
            jnp.ones((2, 3, 5), dtype=jnp.float64),
            ((0, 0, 0),),
        )

        @jax.jit
        def attempted(values: jax.Array) -> jax.Array:
            traced_potential = dataclasses.replace(
                base.potential,
                volume=values,
            )
            traced_realization = dataclasses.replace(
                base,
                potential=traced_potential,
            )
            refined = certify_galerkin_potential_realization(
                traced_realization,
                maximum_direct_terms=100,
            )
            return refined.coefficient_error_bounds

        with pytest.raises(ValueError, match="requires concrete host values"):
            attempted(base.potential.volume)

    def test_pi_interval_crossing_quadrant_uses_analytic_extrema(self) -> None:
        """Remain sound closer to one quarter turn than the pi interval."""
        turn = Fraction(1, 4) - Fraction(1, 1 << 300)
        rectangle = rational_turn_exponential(turn)

        assert rectangle[0] == 0
        assert rectangle[1] > 0
        assert rectangle[2] == -1
        assert rectangle[3] < 0

    def test_operator_transfer_uses_frobenius_schur_minimum(self) -> None:
        """Count represented differences and ignore absent projected modes."""
        states = np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.int64)
        represented = np.asarray(
            ((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
            dtype=np.int64,
        )
        bound = _voltage_operator_error_fraction(
            states,
            represented,
            [Fraction(1, 5), Fraction(1, 10), Fraction(1, 5)],
        )
        expected_schur = Fraction(3, 10)

        assert bound >= expected_schur
        assert bound - expected_schur < Fraction(1, 1 << 120)

        zero_only_bound = _voltage_operator_error_fraction(
            states,
            represented[1:2],
            [Fraction(1, 10)],
        )
        assert zero_only_bound >= Fraction(1, 10)
        assert zero_only_bound - Fraction(1, 10) < Fraction(1, 1 << 120)
