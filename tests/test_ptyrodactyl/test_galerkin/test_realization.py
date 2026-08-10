"""Tests for :mod:`ptyrodactyl.galerkin.realization`.

Extended Summary
----------------
These tests verify the VC-1 voxel-to-Galerkin map against direct physical-
coordinate Fourier sums. They also distinguish the physical voxel metric
from the ordinary Euclidean array metric.
"""

import importlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from numpy.testing import assert_allclose

from ptyrodactyl.galerkin.realization import (
    apply_galerkin_potential_metric_adjoint,
    realize_galerkin_potential,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.potential_types import (
    Potential3D,
    create_potential_3d,
)
from ptyrodactyl.types.realization_types import GalerkinPotentialRealization
from tests._galerkin_target_fixture import checked_acquisition

_PROVENANCE = "b" * 64
_NORMALIZATION = "VC-1 periodic trigonometric mean DFT"
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
    support = create_galerkin_product_support(
        state_indices=zero,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=work_shape,
    )
    return support


def _realize(
    potential: Potential3D,
    support: GalerkinProductSupport,
) -> GalerkinPotentialRealization:
    """Realize through one independently checked support artifact."""
    eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
    )
    realization: GalerkinPotentialRealization = realize_galerkin_potential(
        potential,
        eligibility,
    )
    return realization


def _potential(
    volume: jax.Array | np.ndarray,
    *,
    voxel_size: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    band_limit: float = 0.9,
    boundary: str = "periodic",
) -> Potential3D:
    """Create one fully declared VC-1 source potential."""
    nz, ny, nx = volume.shape
    box_size = (
        nx * voxel_size[0],
        ny * voxel_size[1],
        nz * voxel_size[2],
    )
    potential = create_potential_3d(
        volume,
        voxel_size=voxel_size,
        box_size=box_size,
        origin=origin,
        boundary=boundary,
        producer="realization-test-fixture-v1",
        provenance_hash=_PROVENANCE,
        coefficient_normalization=_NORMALIZATION,
        band_limit=band_limit,
    )
    return potential


def _dense_physical_coefficients(
    potential: Potential3D,
    indices: jax.Array,
) -> np.ndarray:
    """Evaluate the VC-1 coefficients by direct physical-coordinate sums."""
    volume = np.asarray(potential.volume)
    nz, ny, nx = volume.shape
    ox, oy, oz = potential.origin
    lx, ly, lz = potential.box_size
    dx, dy, dz = (lx / nx, ly / ny, lz / nz)
    coordinates = np.stack(
        np.meshgrid(
            ox + dx * np.arange(nx),
            oy + dy * np.arange(ny),
            oz + dz * np.arange(nz),
            indexing="xy",
        ),
        axis=-1,
    )
    # ``meshgrid(indexing="xy")`` returns (ny, nx, nz); restore (z, y, x).
    coordinates = np.transpose(coordinates, (2, 0, 1, 3))
    frequencies = np.asarray(indices, dtype=np.float64) / np.asarray(
        (lx, ly, lz),
        dtype=np.float64,
    )
    coefficients = np.asarray(
        [
            np.sum(
                volume
                * np.exp(
                    -2.0j
                    * np.pi
                    * np.einsum("...d,d->...", coordinates, frequency)
                )
            )
            / volume.size
            for frequency in frequencies
        ],
        dtype=np.complex128,
    )
    return coefficients


class TestGalerkinPotentialRealization:
    """Verify the finite VC-1 realization and its metric adjoint.

    :see: :func:`ptyrodactyl.galerkin.apply_galerkin_potential_metric_adjoint`
    :see: :func:`ptyrodactyl.galerkin.realize_galerkin_potential`
    """

    def test_constant_field_preserves_zero_mode_and_exact_widths(self) -> None:
        """Keep the physical reference instead of silently mean-subtracting."""
        volume = jnp.full((2, 3, 4), 7.25, dtype=jnp.float32)
        realization = _realize(
            _potential(volume),
            _support(((0, 0, 0),)),
        )

        assert realization.potential.volume.dtype == jnp.float64
        assert realization.support.interaction_indices.dtype == jnp.int64
        assert realization.voltage_coefficients.dtype == jnp.complex128
        assert realization.coefficient_error_bounds.dtype == jnp.float64
        assert realization.voltage_operator_error_bound.dtype == jnp.float64
        assert realization.omitted_voltage_l2_diagnostic.dtype == jnp.float64
        assert realization.omitted_voltage_l2_upper_bound.dtype == jnp.float64
        assert_allclose(realization.voltage_coefficients, [7.25], atol=0.0)
        assert_allclose(
            realization.omitted_voltage_l2_diagnostic,
            0.0,
            atol=1.0e-15,
        )

    def test_realization_rejects_acquisition_box_mismatch(self) -> None:
        """Bind the checked reciprocal support to the potential's exact box."""
        potential = _potential(jnp.zeros((3, 3, 3)))
        support = _support(((0, 0, 0),))
        eligibility = checked_acquisition(
            support,
            (potential.box_size[0] + 0.125, *potential.box_size[1:]),
            terminal_axis=2,
        )

        with pytest.raises(_RUNTIME_ERRORS, match="box lengths"):
            realization = realize_galerkin_potential(potential, eligibility)
            jax.block_until_ready(realization.voltage_coefficients)

    def test_strict_band_check_rejects_rounded_norm_false_positive(
        self,
    ) -> None:
        """Use interval squares, not a rounded Euclidean norm, at VC.12."""
        shape = (65, 65, 65)
        box_size = (
            float.fromhex("0x1.c5e12a03afd24p+1"),
            float.fromhex("0x1.7024469110368p+2"),
            float.fromhex("0x1.1f22829157a18p+3"),
        )
        potential = create_potential_3d(
            jnp.zeros(shape, dtype=jnp.float64),
            voxel_size=tuple(length / 65 for length in box_size),
            box_size=box_size,
            origin=(0.0, 0.0, 0.0),
            boundary="periodic",
            producer="strict-band-adversarial-fixture-v1",
            provenance_hash=_PROVENANCE,
            coefficient_normalization=_NORMALIZATION,
            band_limit=float.fromhex("0x1.90cacfbbade59p+0"),
        )
        support = _support(
            ((-2, -8, -4), (0, 0, 0), (2, 8, 4)),
        )
        eligibility = checked_acquisition(
            support,
            potential.box_size,
            terminal_axis=2,
        )

        with pytest.raises(_RUNTIME_ERRORS, match="strict potential band"):
            realization = realize_galerkin_potential(potential, eligibility)
            jax.block_until_ready(realization.voltage_coefficients)

    @pytest.mark.parametrize("include_zero", [False, True])
    def test_triangle_bounds_embed_subnormal_voxels_before_reduction(
        self,
        include_zero: bool,
    ) -> None:
        """Do not let DAZ-sensitive reductions erase stored voxel evidence."""
        subnormal = float.fromhex("0x0.0000000000001p-1022")
        volume = jnp.full((3, 3, 3), subnormal, dtype=jnp.float64)
        if include_zero:
            volume = volume.at[0, 0, 0].set(0.0)
        realization = _realize(
            _potential(
                volume,
                voxel_size=(1.0, 1.0, 1.0),
                band_limit=0.4,
            ),
            _support(((0, 0, 0),)),
        )
        jax.block_until_ready(realization)

        assert (
            realization.coefficient_error_bounds[0]
            >= jnp.finfo(jnp.float64).tiny
        )
        assert (
            realization.omitted_voltage_l2_upper_bound
            >= jnp.finfo(jnp.float64).tiny
        )

    def test_translated_cosine_applies_physical_origin_phase(self) -> None:
        """Apply the origin phase and store one exact Hermitian pair."""
        nx = 6
        origin = (0.375, -0.2, 0.1)
        x_index = jnp.arange(nx, dtype=jnp.float64)
        line = 3.0 + 4.0 * jnp.cos(2.0 * jnp.pi * x_index / nx)
        volume = jnp.broadcast_to(line, (4, 5, nx))
        indices = ((-1, 0, 0), (0, 0, 0), (1, 0, 0))
        potential = _potential(volume, origin=origin)
        realization = _realize(
            potential,
            _support(indices),
        )

        positive = 2.0 * np.exp(
            -2.0j * np.pi * origin[0] / potential.box_size[0]
        )
        expected = np.asarray([np.conj(positive), 3.0, positive])
        assert_allclose(
            realization.voltage_coefficients,
            expected,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        assert realization.voltage_coefficients[0] == jnp.conj(
            realization.voltage_coefficients[2]
        )
        assert jnp.imag(realization.voltage_coefficients[1]) == 0.0

    def test_noncubic_storage_maps_xyz_indices_to_zyx_bins(self) -> None:
        """Distinguish all physical axes on one noncubic stored volume."""
        nz, ny, nx = 4, 5, 6
        z = jnp.arange(nz, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(nx, dtype=jnp.float64)[None, None, :]
        volume = (
            11.0
            + 2.0 * jnp.cos(2.0 * jnp.pi * x / nx)
            + 4.0 * jnp.cos(2.0 * jnp.pi * y / ny)
            + 6.0 * jnp.cos(2.0 * jnp.pi * z / nz)
        )
        indices = (
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
        realization = _realize(
            _potential(volume),
            _support(indices),
        )

        assert_allclose(
            realization.voltage_coefficients,
            [11.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    def test_dense_physical_dft_oracle_is_enclosed_by_error_bounds(
        self,
    ) -> None:
        """Match an independent dense sum and enclose every rounded result."""
        nz, ny, nx = 4, 5, 6
        volume = (
            jnp.arange(nz * ny * nx, dtype=jnp.float64).reshape(nz, ny, nx)
            / 17.0
            - 2.0
        )
        origin = (-0.37, 0.21, 0.44)
        indices = (
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 1),
            (0, -1, -1),
            (1, -1, 0),
            (-1, 1, 0),
        )
        potential = _potential(volume, origin=origin)
        support = _support(indices)
        realization = _realize(potential, support)
        oracle = _dense_physical_coefficients(
            potential,
            support.interaction_indices,
        )
        absolute_error = np.abs(
            np.asarray(realization.voltage_coefficients) - oracle
        )

        assert_allclose(
            realization.voltage_coefficients,
            oracle,
            rtol=5.0e-14,
            atol=5.0e-14,
        )
        assert np.all(
            absolute_error <= np.asarray(realization.coefficient_error_bounds)
        )
        assert np.all(np.isfinite(realization.coefficient_error_bounds))

    def test_even_nyquist_energy_is_reported_as_omitted(self) -> None:
        """Include an even-grid Nyquist mode in the Parseval diagnostic."""
        nz, ny, nx = 3, 2, 4
        alternating_x = (-1.0) ** jnp.arange(nx, dtype=jnp.float64)
        volume = jnp.broadcast_to(alternating_x, (nz, ny, nx))
        potential = _potential(
            volume,
            voxel_size=(0.5, 0.75, 1.0),
            band_limit=0.49,
        )
        realization = _realize(
            potential,
            _support(((0, 0, 0),)),
        )
        expected = np.sqrt(np.prod(potential.box_size))

        assert_allclose(realization.voltage_coefficients, [0.0], atol=0.0)
        assert_allclose(
            realization.omitted_voltage_l2_diagnostic,
            expected,
            rtol=2.0e-14,
        )
        assert (
            realization.omitted_voltage_l2_upper_bound
            >= realization.omitted_voltage_l2_diagnostic
        )

    def test_even_nyquist_interaction_endpoint_is_rejected(self) -> None:
        """Reject both signed representatives of an even physical endpoint."""
        potential = _potential(
            jnp.zeros((3, 3, 4), dtype=jnp.float64),
            voxel_size=(0.25, 0.25, 0.25),
            band_limit=2.0,
        )
        support = _support(((-2, 0, 0), (0, 0, 0), (2, 0, 0)))

        with pytest.raises(_RUNTIME_ERRORS, match="signed grid endpoints"):
            realization = _realize(potential, support)
            jax.block_until_ready(realization.voltage_coefficients)

    def test_strict_band_and_periodic_boundary_are_enforced(self) -> None:
        """Reject an on-or-outside-band mode and a nonperiodic source."""
        support = _support(((-1, 0, 0), (0, 0, 0), (1, 0, 0)))
        outside_band = _potential(
            jnp.zeros((5, 5, 5), dtype=jnp.float64),
            voxel_size=(1.0, 1.0, 1.0),
            band_limit=0.2,
        )
        with pytest.raises(_RUNTIME_ERRORS, match="strict potential band"):
            realization = _realize(outside_band, support)
            jax.block_until_ready(realization.voltage_coefficients)

        nonperiodic = _potential(
            jnp.zeros((4, 4, 4), dtype=jnp.float64),
            boundary="isolated",
        )
        with pytest.raises(ValueError, match="exactly 'periodic'"):
            _realize(nonperiodic, _support(((0, 0, 0),)))

    def test_forged_checked_support_artifact_is_rejected(self) -> None:
        """Compare every supplied evidence leaf with a fresh checker result."""
        potential = _potential(jnp.ones((3, 3, 3), dtype=jnp.float64))
        support = _support(((0, 0, 0),))
        eligibility = checked_acquisition(
            support,
            potential.box_size,
            terminal_axis=2,
        )
        forged = eqx.tree_at(
            lambda result: result.incident_full_offset_max,
            eligibility,
            eligibility.incident_full_offset_max + 1.0,
        )

        with pytest.raises(_RUNTIME_ERRORS, match="exactly match"):
            realization = realize_galerkin_potential(potential, forged)
            jax.block_until_ready(realization.voltage_coefficients)

    def test_metric_adjoint_obeys_realified_physical_dot_identity(
        self,
    ) -> None:
        """Satisfy VC.21 with the cell-volume factor exactly once."""
        shape = (5, 5, 6)
        direction = (
            jnp.arange(math.prod(shape), dtype=jnp.float64).reshape(shape)
            / 31.0
            - 1.0
        )
        indices = (
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 1),
            (0, -1, -1),
        )
        support = _support(indices)
        base = _realize(
            _potential(
                jnp.zeros(shape),
                voxel_size=(0.4, 0.3, 0.2),
                origin=(-0.19, 0.27, 0.08),
                band_limit=1.23,
            ),
            support,
        )
        direction_coefficients = _realize(
            _potential(
                direction,
                voxel_size=(0.4, 0.3, 0.2),
                origin=(-0.19, 0.27, 0.08),
                band_limit=1.23,
            ),
            support,
        ).voltage_coefficients
        cotangent = jnp.asarray(
            [
                0.2 - 0.7j,
                -0.4 + 0.1j,
                0.9 + 0.3j,
                -0.6 - 0.2j,
                0.5 + 0.8j,
            ],
            dtype=jnp.complex128,
        )
        voxel_adjoint = apply_galerkin_potential_metric_adjoint(
            base,
            cotangent,
        )
        voxel_volume = np.prod(base.potential.box_size) / np.prod(shape)
        physical_pairing = voxel_volume * np.sum(
            np.asarray(direction) * np.asarray(voxel_adjoint)
        )
        coefficient_pairing = np.real(
            np.vdot(
                np.asarray(cotangent),
                np.asarray(direction_coefficients),
            )
        )

        assert voxel_adjoint.dtype == jnp.float64
        assert_allclose(
            physical_pairing,
            coefficient_pairing,
            rtol=5.0e-14,
            atol=5.0e-14,
        )

    def test_coefficient_map_is_jit_jvp_and_centered_difference_clean(
        self,
    ) -> None:
        """Compile and differentiate the real voxel-to-coefficient map."""
        shape = (4, 5, 6)
        sample_count = math.prod(shape)
        volume = jnp.linspace(-1.0, 2.0, sample_count).reshape(shape)
        direction = jnp.cos(
            jnp.arange(sample_count, dtype=jnp.float64)
        ).reshape(shape)
        support = _support(
            (
                (0, 0, 0),
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 1),
                (0, -1, -1),
            )
        )
        eligibility = checked_acquisition(
            support,
            _potential(volume, origin=(-0.13, 0.22, 0.31)).box_size,
            terminal_axis=2,
        )

        def coefficient_map(values: jax.Array) -> jax.Array:
            """Return only the differentiable coefficient leaf."""
            realization = realize_galerkin_potential(
                _potential(values, origin=(-0.13, 0.22, 0.31)),
                eligibility,
            )
            return realization.voltage_coefficients

        compiled_map = jax.jit(coefficient_map)
        compiled = compiled_map(volume)
        eager = coefficient_map(volume)
        _, tangent = jax.jvp(coefficient_map, (volume,), (direction,))

        def coefficient_and_evidence_map(values: jax.Array):
            """Return differentiable coefficients and stopped evidence."""
            realization = realize_galerkin_potential(
                _potential(values, origin=(-0.13, 0.22, 0.31)),
                eligibility,
            )
            result = (
                realization.voltage_coefficients,
                realization.coefficient_error_bounds,
                realization.voltage_operator_error_bound,
                realization.omitted_voltage_l2_diagnostic,
                realization.omitted_voltage_l2_upper_bound,
            )
            return result

        _, evidence_tangents = jax.jvp(
            coefficient_and_evidence_map,
            (volume,),
            (direction,),
        )
        step = 1.0e-5
        finite_difference = (
            compiled_map(volume + step * direction)
            - compiled_map(volume - step * direction)
        ) / (2.0 * step)

        assert compiled.dtype == jnp.complex128
        assert tangent.dtype == jnp.complex128
        assert_allclose(compiled, eager, rtol=2.0e-14, atol=2.0e-14)
        assert_allclose(
            tangent,
            finite_difference,
            rtol=2.0e-9,
            atol=2.0e-10,
        )
        assert np.any(np.asarray(evidence_tangents[0]) != 0.0)
        for evidence_tangent in evidence_tangents[1:]:
            np.testing.assert_array_equal(
                evidence_tangent,
                jnp.zeros_like(evidence_tangent),
            )

    def test_unsupported_normal_arithmetic_rejects_realization(
        self,
        monkeypatch,
    ) -> None:
        """Reject realization when rechecked arithmetic is unsupported."""
        potential = _potential(jnp.ones((2, 3, 4), dtype=jnp.float64))
        support = _support(((0, 0, 0),))
        eligibility = checked_acquisition(
            support,
            potential.box_size,
            terminal_axis=2,
        )
        interval_core = importlib.import_module("ptyrodactyl._interval")

        def unsupported_normal_arithmetic() -> jax.Array:
            return jnp.asarray(False)

        monkeypatch.setattr(
            interval_core,
            "_all_normal_arithmetic_supported",
            unsupported_normal_arithmetic,
        )

        with pytest.raises(_RUNTIME_ERRORS):
            realization = realize_galerkin_potential(potential, eligibility)
            jax.block_until_ready(realization)
