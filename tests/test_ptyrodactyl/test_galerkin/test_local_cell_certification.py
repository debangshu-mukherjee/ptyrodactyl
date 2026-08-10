r"""Tests for direct host certification of local-cell LVT.7 coefficients.

Extended Summary
----------------
These tests exercise independent mean-DFT, unwrapped sinc, origin-phase, and
Euclidean LVT.13 enclosures. They also freeze the host work budget, exact
Hermitian approximant boundary, canonical digest ownership, typed infinite
noncertificates, and full replay required before downstream scientific use.
"""

import dataclasses
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from numpy.testing import assert_allclose

import ptyrodactyl.galerkin.local_cell_certification as certification_module
from ptyrodactyl._tools import RootEnclosureError
from ptyrodactyl.galerkin.local_cell import (
    realize_local_cell_galerkin_potential,
)
from ptyrodactyl.galerkin.local_cell_certification import (
    _authenticate_local_cell_certificate,
    _certificate_digest,
    _validate_local_cell_certificate_binding,
    certify_local_cell_galerkin_potential,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.local_cell_types import (
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellErrorRoute,
    GalerkinLocalCellPotentialRealization,
    LocalCellPotential3D,
    create_local_cell_potential_3d,
)
from tests._galerkin_target_fixture import checked_acquisition

_PROVENANCE = "8" * 64
_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _support(
    indices: Tuple[Tuple[int, int, int], ...],
) -> GalerkinProductSupport:
    """Create one product-valid support with a singleton state space."""
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
    cell_size: Tuple[float, float, float] = (0.4, 0.7, 1.1),
    cell_center_origin: Tuple[float, float, float] = (0.13, -0.22, 0.31),
    reference_value: float = 0.0,
    reference_semantics: str = "declared certificate-test reference",
) -> LocalCellPotential3D:
    """Create one shifted anisotropic periodic local-cell source."""
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
        producer="local-cell-certificate-test-v1",
        provenance_hash=_PROVENANCE,
        producer_coefficient_normalization="producer metadata only",
        producer_bandwidth=1.0e9,
    )
    return potential


def _realize(
    potential: LocalCellPotential3D,
    support: GalerkinProductSupport,
) -> GalerkinLocalCellPotentialRealization:
    """Realize through one independently checked acquisition artifact."""
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
    result: complex = (
        np.exp(-2.0j * np.pi * frequency * upper)
        - np.exp(-2.0j * np.pi * frequency * lower)
    ) / (-2.0j * np.pi * frequency)
    return result


def _direct_cell_integral_coefficients(
    potential: LocalCellPotential3D,
    indices: jax.Array,
) -> np.ndarray:
    """Integrate every physical cell independently of DFT and sinc code."""
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
    result = np.asarray(coefficients, dtype=np.complex128)
    return result


def _shifted_fixture() -> Tuple[
    LocalCellPotential3D,
    GalerkinProductSupport,
    GalerkinLocalCellPotentialRealization,
]:
    """Create one nontrivial beyond-Nyquist direct-certificate fixture."""
    values = jnp.asarray(
        [
            [[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]],
            [[-0.75, 0.5, 1.25], [2.25, -1.5, 0.75]],
        ],
        dtype=jnp.float64,
    )
    potential = _potential(values)
    support = _support(
        (
            (-4, -1, -1),
            (0, 0, 0),
            (4, 1, 1),
        )
    )
    realization = _realize(potential, support)
    result: Tuple[
        LocalCellPotential3D,
        GalerkinProductSupport,
        GalerkinLocalCellPotentialRealization,
    ] = (potential, support, realization)
    return result


class TestLocalCellDirectCertification:
    """Verify exact LVT.7 rectangles, errors, budgets, and identities.

    :see: :func:`ptyrodactyl.galerkin.\
certify_local_cell_galerkin_potential`
    """

    def test_shifted_anisotropic_beyond_nyquist_matches_direct_oracle(
        self,
    ) -> None:
        """Match an independent floating cell-integral regression."""
        potential, support, realization = _shifted_fixture()
        certified = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        certificate = certified.coefficient_certificate
        assert certificate is not None
        exact = _direct_cell_integral_coefficients(
            potential,
            support.interaction_indices,
        )
        real_lower = np.asarray(
            certificate.exact_coefficient_real_lower_bounds
        )
        real_upper = np.asarray(
            certificate.exact_coefficient_real_upper_bounds
        )
        imag_lower = np.asarray(
            certificate.exact_coefficient_imag_lower_bounds
        )
        imag_upper = np.asarray(
            certificate.exact_coefficient_imag_upper_bounds
        )

        assert certified.error_route is (
            GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
        )
        assert certificate.failure is GalerkinLocalCellCertificateFailure.NONE
        assert bool(certificate.finite_certificate)
        assert int(certificate.direct_term_count) == 24
        assert int(certificate.maximum_direct_terms) == 24
        midpoint = 0.5 * (real_lower + real_upper) + 0.5j * (
            imag_lower + imag_upper
        )
        assert_allclose(midpoint, exact, rtol=2.0e-13, atol=2.0e-14)
        assert_allclose(
            certified.voltage_coefficients,
            exact,
            rtol=2.0e-13,
            atol=2.0e-14,
        )
        assert np.all(real_lower <= real_upper)
        assert np.all(imag_lower <= imag_upper)
        assert np.all(np.isfinite(certified.coefficient_error_bounds))
        assert all(
            len(digest) == 64
            for digest in (
                certificate.local_potential_digest,
                certificate.requested_support_digest,
                certificate.stored_coefficients_digest,
                certificate.realization_digest,
                certificate.certificate_digest,
            )
        )
        replay = _authenticate_local_cell_certificate(certified)
        replay_certificate = replay.coefficient_certificate
        assert replay_certificate is not None
        assert certificate.certificate_digest == (
            replay_certificate.certificate_digest
        )

    def test_exact_beyond_nyquist_oracle_uses_rational_pi_bracket(
        self,
    ) -> None:
        """Enclose 8/(27*pi**3) without using a rounded trig oracle."""
        values = np.zeros((2, 2, 2), dtype=np.float64)
        values[0, 1, 1] = 8.0
        potential = _potential(
            jnp.asarray(values, dtype=jnp.float64),
            cell_size=(0.5, 0.75, 1.25),
            cell_center_origin=(0.5, 0.0, 0.0),
        )
        support = _support(((-3, -3, -3), (0, 0, 0), (3, 3, 3)))
        certified = certify_local_cell_galerkin_potential(
            _realize(potential, support),
            maximum_direct_terms=16,
        )
        certificate = certified.coefficient_certificate
        assert certificate is not None
        modes = np.asarray(support.interaction_indices)
        nonzero_positions = np.flatnonzero(np.any(modes != 0, axis=1))
        zero_position = int(np.flatnonzero(np.all(modes == 0, axis=1))[0])
        real_lower = np.asarray(
            certificate.exact_coefficient_real_lower_bounds
        )
        real_upper = np.asarray(
            certificate.exact_coefficient_real_upper_bounds
        )
        imag_lower = np.asarray(
            certificate.exact_coefficient_imag_lower_bounds
        )
        imag_upper = np.asarray(
            certificate.exact_coefficient_imag_upper_bounds
        )
        pi_numerator = 314159265358979323846264338327950288419716939937510
        pi_lower = Fraction(pi_numerator, 10**50)
        pi_upper = Fraction(pi_numerator + 1, 10**50)
        exact_lower = Fraction(8, 27) / (pi_upper**3)
        exact_upper = Fraction(8, 27) / (pi_lower**3)

        assert certificate.failure is GalerkinLocalCellCertificateFailure.NONE
        assert int(certificate.direct_term_count) == 16
        for position in nonzero_positions:
            assert Fraction.from_float(float(real_lower[position])) <= (
                exact_lower
            )
            assert Fraction.from_float(float(real_upper[position])) >= (
                exact_upper
            )
            assert imag_lower[position] <= 0.0
            assert imag_upper[position] >= 0.0
        assert real_lower[zero_position] == 1.0
        assert real_upper[zero_position] == 1.0
        assert imag_lower[zero_position] == 0.0
        assert imag_upper[zero_position] == 0.0

    def test_symbolic_qn_zero_skips_work_and_uses_euclidean_error(
        self,
    ) -> None:
        """Give a qN pair exact-zero rectangles and radius five."""
        potential = _potential(jnp.ones((1, 1, 1), dtype=jnp.float64))
        support = _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0)))
        realization = _realize(potential, support)
        modes = np.asarray(support.interaction_indices)
        negative_position = int(
            np.flatnonzero(np.all(modes == (-1, 0, 0), axis=1))[0]
        )
        positive_position = int(
            np.flatnonzero(np.all(modes == (1, 0, 0), axis=1))[0]
        )
        zero_position = int(np.flatnonzero(np.all(modes == 0, axis=1))[0])
        coefficients = np.asarray(realization.voltage_coefficients).copy()
        coefficients[negative_position] = 3.0 - 4.0j
        coefficients[positive_position] = 3.0 + 4.0j
        coefficients[zero_position] = 2.5 + 0.0j
        approximant = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            realization,
            jnp.asarray(coefficients, dtype=jnp.complex128),
        )
        certified = certify_local_cell_galerkin_potential(
            approximant,
            maximum_direct_terms=1,
        )
        certificate = certified.coefficient_certificate
        assert certificate is not None
        pair_positions = np.asarray(
            (negative_position, positive_position),
            dtype=np.int64,
        )

        assert certificate.failure is GalerkinLocalCellCertificateFailure.NONE
        assert int(certificate.direct_term_count) == 1
        assert np.all(
            np.asarray(certificate.exact_coefficient_real_lower_bounds)[
                pair_positions
            ]
            == 0.0
        )
        assert np.all(
            np.asarray(certificate.exact_coefficient_real_upper_bounds)[
                pair_positions
            ]
            == 0.0
        )
        assert np.all(
            np.asarray(certificate.exact_coefficient_imag_lower_bounds)[
                pair_positions
            ]
            == 0.0
        )
        assert np.all(
            np.asarray(certificate.exact_coefficient_imag_upper_bounds)[
                pair_positions
            ]
            == 0.0
        )
        assert np.all(
            np.asarray(certified.coefficient_error_bounds)[pair_positions]
            == 5.0
        )
        _authenticate_local_cell_certificate(certified)

    def test_budget_boundary_succeeds_and_one_less_is_typed_infinite(
        self,
    ) -> None:
        """Bind actual work at both sides of the budget boundary."""
        _, _, realization = _shifted_fixture()
        success = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        failed = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=23,
        )
        success_certificate = success.coefficient_certificate
        failed_certificate = failed.coefficient_certificate
        assert success_certificate is not None
        assert failed_certificate is not None

        assert success_certificate.failure is (
            GalerkinLocalCellCertificateFailure.NONE
        )
        assert failed_certificate.failure is (
            GalerkinLocalCellCertificateFailure.WORK_BUDGET_EXCEEDED
        )
        assert int(failed_certificate.direct_term_count) == 24
        assert not bool(failed_certificate.finite_certificate)
        assert np.all(np.isposinf(failed.coefficient_error_bounds))
        assert np.all(
            np.isneginf(failed_certificate.exact_coefficient_real_lower_bounds)
        )
        assert np.all(
            np.isposinf(failed_certificate.exact_coefficient_real_upper_bounds)
        )
        replay = _authenticate_local_cell_certificate(failed)
        replay_certificate = replay.coefficient_certificate
        assert replay_certificate is not None
        assert replay_certificate.failure is (
            GalerkinLocalCellCertificateFailure.WORK_BUDGET_EXCEEDED
        )

    def test_eager_and_materialized_jit_approximants_both_certify(
        self,
    ) -> None:
        """Avoid authenticating coefficient bytes by eager FFT replay."""
        potential, support, eager = _shifted_fixture()
        eligibility = eager.support_eligibility

        def coefficient_map(values: jax.Array) -> jax.Array:
            candidate = eqx.tree_at(
                lambda item: item.cell_values,
                potential,
                values,
            )
            result = realize_local_cell_galerkin_potential(
                candidate,
                eligibility,
            )
            return result.voltage_coefficients

        compiled_coefficients = jax.jit(coefficient_map)(potential.cell_values)
        compiled = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            eager,
            compiled_coefficients,
        )
        eager_certificate = certify_local_cell_galerkin_potential(
            eager,
            maximum_direct_terms=24,
        )
        compiled_certificate = certify_local_cell_galerkin_potential(
            compiled,
            maximum_direct_terms=24,
        )
        eager_evidence = eager_certificate.coefficient_certificate
        compiled_evidence = compiled_certificate.coefficient_certificate
        assert eager_evidence is not None
        assert compiled_evidence is not None

        assert eager_evidence.failure is (
            GalerkinLocalCellCertificateFailure.NONE
        )
        assert compiled_evidence.failure is (
            GalerkinLocalCellCertificateFailure.NONE
        )
        assert_allclose(
            compiled_certificate.voltage_coefficients,
            eager_certificate.voltage_coefficients,
            rtol=5.0e-14,
            atol=5.0e-14,
        )
        _authenticate_local_cell_certificate(eager_certificate)
        _authenticate_local_cell_certificate(compiled_certificate)

    def test_one_bit_mutation_changes_digest_without_forward_replay(
        self,
    ) -> None:
        """Bind the actual approximant rather than fallback errors."""
        _, support, realization = _shifted_fixture()
        certified = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        certificate = certified.coefficient_certificate
        assert certificate is not None
        coefficients = np.asarray(certified.voltage_coefficients).copy()
        zero_position = int(
            np.flatnonzero(
                np.all(np.asarray(support.interaction_indices) == 0, axis=1)
            )[0]
        )
        coefficients[zero_position] = np.nextafter(
            coefficients[zero_position].real,
            np.inf,
        )
        mutated = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            certified,
            jnp.asarray(coefficients, dtype=jnp.complex128),
        )

        with pytest.raises(ValueError, match="parent binding"):
            _validate_local_cell_certificate_binding(mutated)

        recertified = certify_local_cell_galerkin_potential(
            mutated,
            maximum_direct_terms=24,
        )
        recertificate = recertified.coefficient_certificate
        assert recertificate is not None
        assert recertificate.stored_coefficients_digest != (
            certificate.stored_coefficients_digest
        )
        assert recertificate.certificate_digest != (
            certificate.certificate_digest
        )

    def test_signed_zero_pairs_are_numeric_hermitian_but_byte_bound(
        self,
    ) -> None:
        """Accept either zero sign while distinguishing exact stored bytes."""
        potential = _potential(jnp.ones((1, 1, 3), dtype=jnp.float64))
        support = _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0)))
        realization = _realize(potential, support)
        modes = np.asarray(support.interaction_indices)
        coefficients = np.empty((3,), dtype=np.complex128)
        coefficients[np.all(modes == (-1, 0, 0), axis=1)] = complex(1.0, 0.0)
        coefficients[np.all(modes == (0, 0, 0), axis=1)] = complex(2.0, 0.0)
        coefficients[np.all(modes == (1, 0, 0), axis=1)] = complex(1.0, 0.0)
        positive_zero = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            realization,
            jnp.asarray(coefficients),
        )
        negative_bytes = coefficients.copy()
        negative_bytes[np.all(modes == (-1, 0, 0), axis=1)] = complex(
            1.0,
            -0.0,
        )
        mixed_zero = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            realization,
            jnp.asarray(negative_bytes),
        )

        first = certify_local_cell_galerkin_potential(
            positive_zero,
            maximum_direct_terms=6,
        )
        second = certify_local_cell_galerkin_potential(
            mixed_zero,
            maximum_direct_terms=6,
        )
        first_certificate = first.coefficient_certificate
        second_certificate = second.coefficient_certificate
        assert first_certificate is not None
        assert second_certificate is not None
        first_payload = np.asarray(first.voltage_coefficients)
        second_payload = np.asarray(second.voltage_coefficients)
        negative_position = int(
            np.flatnonzero(np.all(modes == (-1, 0, 0), axis=1))[0]
        )

        assert not np.signbit(first_payload[negative_position].imag)
        assert np.signbit(second_payload[negative_position].imag)
        assert np.array_equal(first_payload, second_payload)
        assert first_payload.tobytes() != second_payload.tobytes()
        assert first_certificate.failure is (
            GalerkinLocalCellCertificateFailure.NONE
        )
        assert second_certificate.failure is (
            GalerkinLocalCellCertificateFailure.NONE
        )
        assert first_certificate.stored_coefficients_digest != (
            second_certificate.stored_coefficients_digest
        )

    def test_self_consistent_forged_evidence_fails_full_host_replay(
        self,
    ) -> None:
        """Reject forged rectangles despite an updated public checksum."""
        _, _, realization = _shifted_fixture()
        certified = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        certificate = certified.coefficient_certificate
        assert certificate is not None
        size = certified.voltage_coefficients.shape[0]
        zeros = np.zeros((size,), dtype=np.float64)
        errors = np.zeros((size,), dtype=np.float64)
        finite_array = np.asarray(True, dtype=np.bool_)
        count_array = np.asarray(24, dtype=np.int64)
        budget_array = np.asarray(24, dtype=np.int64)
        forged_digest = _certificate_digest(
            certificate.realization_digest,
            certificate.local_potential_digest,
            certificate.requested_support_digest,
            certificate.stored_coefficients_digest,
            zeros,
            zeros,
            zeros,
            zeros,
            errors,
            finite_array,
            count_array,
            budget_array,
            certificate.failure,
            certificate.coefficient_formula,
        )
        forged_certificate = dataclasses.replace(
            certificate,
            exact_coefficient_real_lower_bounds=jnp.asarray(zeros),
            exact_coefficient_real_upper_bounds=jnp.asarray(zeros),
            exact_coefficient_imag_lower_bounds=jnp.asarray(zeros),
            exact_coefficient_imag_upper_bounds=jnp.asarray(zeros),
            certificate_digest=forged_digest,
        )
        forged = dataclasses.replace(
            certified,
            coefficient_error_bounds=jnp.asarray(errors),
            coefficient_certificate=forged_certificate,
        )

        _validate_local_cell_certificate_binding(forged)
        with pytest.raises(ValueError, match="host replay"):
            _authenticate_local_cell_certificate(forged)

    def test_swapped_certificate_rejects(self) -> None:
        """Prevent certificate replay across distinct source identities."""
        potential, support, realization = _shifted_fixture()
        first = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        changed_potential = _potential(
            potential.cell_values,
            reference_value=1.25,
            reference_semantics="declared shifted reference for swap test",
        )
        second = certify_local_cell_galerkin_potential(
            _realize(changed_potential, support),
            maximum_direct_terms=24,
        )
        swapped = dataclasses.replace(
            second,
            coefficient_certificate=first.coefficient_certificate,
        )

        with pytest.raises(ValueError, match="parent binding"):
            _validate_local_cell_certificate_binding(swapped)

    def test_certificate_scalar_dtype_and_shape_forgery_rejects(
        self,
    ) -> None:
        """Bind scalar certificate dtype, rank, and stored bytes exactly."""
        _, _, realization = _shifted_fixture()
        certified = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        certificate = certified.coefficient_certificate
        assert certificate is not None
        mutations = (
            ("finite_int8", "exact bool"),
            ("count_int32", "exact int64"),
            ("budget_int32", "exact int64"),
            ("budget_vector", "exact int64"),
        )
        for mutation, match in mutations:
            if mutation == "finite_int8":
                bad_certificate = eqx.tree_at(
                    lambda item: item.finite_certificate,
                    certificate,
                    jnp.asarray(1, dtype=jnp.int8),
                )
            elif mutation == "count_int32":
                bad_certificate = eqx.tree_at(
                    lambda item: item.direct_term_count,
                    certificate,
                    jnp.asarray(24, dtype=jnp.int32),
                )
            elif mutation == "budget_int32":
                bad_certificate = eqx.tree_at(
                    lambda item: item.maximum_direct_terms,
                    certificate,
                    jnp.asarray(24, dtype=jnp.int32),
                )
            else:
                bad_certificate = eqx.tree_at(
                    lambda item: item.maximum_direct_terms,
                    certificate,
                    jnp.asarray([24], dtype=jnp.int64),
                )
            bad_scalar = eqx.tree_at(
                lambda item: item.coefficient_certificate,
                certified,
                bad_certificate,
            )
            with pytest.raises(ValueError, match=match):
                _validate_local_cell_certificate_binding(bad_scalar)

    def test_reference_metadata_changes_source_not_coefficient_digest(
        self,
    ) -> None:
        """Bind references without adding them to complete cell values."""
        potential, support, realization = _shifted_fixture()
        shifted = _potential(
            potential.cell_values,
            reference_value=-2.5,
            reference_semantics="declared alternate complete-value reference",
        )
        first = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        second = certify_local_cell_galerkin_potential(
            _realize(shifted, support),
            maximum_direct_terms=24,
        )
        first_certificate = first.coefficient_certificate
        second_certificate = second.coefficient_certificate
        assert first_certificate is not None
        assert second_certificate is not None

        assert_allclose(
            first.voltage_coefficients,
            second.voltage_coefficients,
            rtol=0.0,
            atol=0.0,
        )
        assert first_certificate.stored_coefficients_digest == (
            second_certificate.stored_coefficients_digest
        )
        assert first_certificate.local_potential_digest != (
            second_certificate.local_potential_digest
        )

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ("nonhermitian", "Hermitian"),
            ("complex_zero", "zero-mode"),
            ("wrong_dtype", "complex128"),
        ],
    )
    def test_invalid_actual_approximants_reject(
        self,
        mutation: str,
        match: str,
    ) -> None:
        """Reject malformed coefficient payloads before interval work."""
        _, support, realization = _shifted_fixture()
        coefficients = np.asarray(realization.voltage_coefficients).copy()
        modes = np.asarray(support.interaction_indices)
        if mutation == "nonhermitian":
            coefficients[0] += 0.125j
            payload = jnp.asarray(coefficients, dtype=jnp.complex128)
        elif mutation == "complex_zero":
            zero_position = int(np.flatnonzero(np.all(modes == 0, axis=1))[0])
            coefficients[zero_position] += 0.125j
            payload = jnp.asarray(coefficients, dtype=jnp.complex128)
        else:
            payload = jnp.asarray(coefficients, dtype=jnp.complex64)
        invalid = eqx.tree_at(
            lambda item: item.voltage_coefficients,
            realization,
            payload,
        )

        with pytest.raises(ValueError, match=match):
            certify_local_cell_galerkin_potential(
                invalid,
                maximum_direct_terms=24,
            )

    def test_tracers_reject_at_the_host_boundary(self) -> None:
        """Do not let JIT tracing manufacture host certificate claims."""
        _, _, realization = _shifted_fixture()

        @jax.jit
        def traced(coefficients: jax.Array) -> jax.Array:
            candidate = eqx.tree_at(
                lambda item: item.voltage_coefficients,
                realization,
                coefficients,
            )
            result = certify_local_cell_galerkin_potential(
                candidate,
                maximum_direct_terms=24,
            )
            return result.coefficient_error_bounds

        with pytest.raises(ValueError, match="concrete host"):
            traced(realization.voltage_coefficients)

    @pytest.mark.parametrize(
        ("failure", "patch_name"),
        [
            (
                GalerkinLocalCellCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED,
                "host",
            ),
            (
                GalerkinLocalCellCertificateFailure.ROOT_ENCLOSURE_FAILURE,
                "root",
            ),
            (
                GalerkinLocalCellCertificateFailure.ARITHMETIC_RANGE_FAILURE,
                "range",
            ),
        ],
    )
    def test_host_root_and_range_failures_are_typed_infinite(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure: GalerkinLocalCellCertificateFailure,
        patch_name: str,
    ) -> None:
        """Fail closed without falling back to triangle evidence."""
        _, _, realization = _shifted_fixture()
        if patch_name == "host":
            monkeypatch.setattr(
                certification_module,
                "host_binary64_supported",
                lambda: False,
            )
        elif patch_name == "root":

            def fail_root(*args: object, **kwargs: object) -> object:
                """Raise the typed private root-enclosure failure."""
                raise RootEnclosureError("injected root failure")

            monkeypatch.setattr(
                certification_module,
                "_exact_coefficient_rectangles",
                fail_root,
            )
        else:
            monkeypatch.setattr(
                certification_module,
                "fraction_upper_float",
                lambda value: np.inf,
            )

        result = certify_local_cell_galerkin_potential(
            realization,
            maximum_direct_terms=24,
        )
        certificate = result.coefficient_certificate
        assert certificate is not None
        assert certificate.failure is failure
        assert not bool(certificate.finite_certificate)
        assert result.error_route is (
            GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL
        )
        assert np.all(np.isposinf(result.coefficient_error_bounds))
