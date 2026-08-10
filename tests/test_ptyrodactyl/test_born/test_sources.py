"""Tests for :mod:`ptyrodactyl.born.sources`.

Extended Summary
----------------
These tests verify the narrow represented stored-shell RM-S3 branch: physical
carrier flux, explicit phase conventions, matched ``H_0v`` injection with
CAP, total/scattered residual equivalence, transformations, and fail-closed
branch predicates.
"""

from decimal import Decimal, getcontext
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple

from ptyrodactyl._interval import _upward_add
from ptyrodactyl.born.potential import (
    apply_absorber_action,
    apply_interaction_product,
)
from ptyrodactyl.born.sources import (
    build_represented_focused_galerkin_source,
    build_represented_plane_galerkin_source,
)
from ptyrodactyl.born.system import create_galerkin_target
from ptyrodactyl.types.acquisition_types import GalerkinBackwardDisposition
from ptyrodactyl.types.born_potential_types import (
    create_galerkin_product_support,
)
from ptyrodactyl.types.galerkin_types import GalerkinTargetManifest
from ptyrodactyl.types.potential_types import create_potential_3d
from ptyrodactyl.types.source_types import (
    GalerkinRepresentedSource,
    GalerkinRepresentedSourceKind,
    GalerkinSourceAxis,
    GalerkinSourceErrorRoute,
    GalerkinSourcePhaseConvention,
    GalerkinSourceRepresentationRoute,
    GalerkinStoredShellRoute,
    create_represented_galerkin_source,
)
from tests._galerkin_target_fixture import (
    checked_acquisition,
    stored_wavenumber,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)
getcontext().prec = 120

type _RationalComplex = Tuple[Fraction, Fraction]


def _sorted_rows(rows: set[Tuple[int, int, int]]) -> jax.Array:
    """Convert one exact integer row set to canonical int64 storage."""
    return jnp.asarray(sorted(rows), dtype=jnp.int64)


def _step_negative(value: float, count: int) -> jax.Array:
    """Move one binary64 value ``count`` ULPs toward negative infinity."""
    stepped = np.float64(value)
    for _ in range(count):
        stepped = np.nextafter(stepped, -np.inf)
    return jnp.asarray(stepped, dtype=jnp.float64)


def _rational_float(value: float | np.float64) -> Fraction:
    """Return the exact rational represented by one binary64 value."""
    return Fraction.from_float(float(value))


def _rational_complex(value: complex | np.complex128) -> _RationalComplex:
    """Return exact rational components for one stored complex value."""
    stored = complex(value)
    return (_rational_float(stored.real), _rational_float(stored.imag))


def _complex_add(
    left: _RationalComplex,
    right: _RationalComplex,
) -> _RationalComplex:
    """Add two exact rational complex values."""
    return (left[0] + right[0], left[1] + right[1])


def _complex_subtract(
    left: _RationalComplex,
    right: _RationalComplex,
) -> _RationalComplex:
    """Subtract two exact rational complex values."""
    return (left[0] - right[0], left[1] - right[1])


def _complex_multiply(
    left: _RationalComplex,
    right: _RationalComplex,
) -> _RationalComplex:
    """Multiply two exact rational complex values."""
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _exact_multiplier_action(
    state_indices: jax.Array,
    multiplier_indices: jax.Array,
    coefficients: jax.Array,
    field: jax.Array,
) -> list[_RationalComplex]:
    """Apply one exact-real compressed multiplier with rational arithmetic."""
    state = np.asarray(state_indices, dtype=np.int64)
    coefficient_map = {
        tuple(index): _rational_complex(value)
        for index, value in zip(
            np.asarray(multiplier_indices),
            np.asarray(coefficients),
            strict=True,
        )
    }
    exact_field = [_rational_complex(value) for value in np.asarray(field)]
    zero = (Fraction(0), Fraction(0))
    result: list[_RationalComplex] = []
    for row in state:
        value = zero
        for column, field_value in zip(state, exact_field, strict=True):
            coefficient = coefficient_map.get(tuple(row - column), zero)
            value = _complex_add(
                value,
                _complex_multiply(coefficient, field_value),
            )
        result.append(value)
    return result


def _scale_exact_action(
    scalar: float | np.float64,
    action: list[_RationalComplex],
) -> list[_RationalComplex]:
    """Multiply an exact rational complex action by a real binary64 scalar."""
    factor = (_rational_float(scalar), Fraction(0))
    return [_complex_multiply(factor, value) for value in action]


def _exact_free_action(
    manifest: GalerkinTargetManifest,
    field: jax.Array,
) -> list[_RationalComplex]:
    """Apply the exact-real stored free diagonal with rational arithmetic."""
    return [
        _complex_multiply(
            (_rational_float(diagonal), Fraction(0)),
            _rational_complex(value),
        )
        for diagonal, value in zip(
            np.asarray(manifest.free_diagonal),
            np.asarray(field),
            strict=True,
        )
    ]


def _exact_error_norm(
    stored: jax.Array,
    exact: list[_RationalComplex],
) -> Decimal:
    """Return the high-precision norm of stored-minus-exact components."""
    squared = Decimal(0)
    for stored_value, exact_value in zip(
        np.asarray(stored), exact, strict=True
    ):
        difference = _complex_subtract(
            _rational_complex(stored_value), exact_value
        )
        for component in difference:
            decimal = Decimal(component.numerator) / Decimal(
                component.denominator
            )
            squared += decimal**2
    return squared.sqrt()


def _decimal_bound(value: jax.Array) -> Decimal:
    """Interpret one finite binary64 bound exactly."""
    return Decimal.from_float(float(value))


def _manifest() -> GalerkinTargetManifest:
    """Create forward, off-shell, backward, and duplicate-fiber test modes."""
    state_rows = {
        (0, 0, 0),
        (1, 0, -1),
        (0, 1, -1),
        (1, 1, -1),
        (0, 0, -10),
        (0, 0, -1),
    }
    interaction_rows = state_rows | {
        (-first, -second, -third) for first, second, third in state_rows
    }
    absorber_rows = {
        (first, second, third)
        for first in range(-1, 2)
        for second in range(-1, 2)
        for third in range(-1, 2)
    }
    absorber_rows.update(
        {
            (
                left[0] - right[0],
                left[1] - right[1],
                left[2] - right[2],
            )
            for left in state_rows
            for right in state_rows
        }
    )
    work_rows = {
        (
            state[0] + multiplier[0],
            state[1] + multiplier[1],
            state[2] + multiplier[2],
        )
        for state in state_rows
        for multiplier in interaction_rows | absorber_rows
    }
    maxima = tuple(
        max(abs(row[axis]) for row in work_rows | absorber_rows | state_rows)
        for axis in range(3)
    )
    work_shape = tuple(2 * maximum + 3 for maximum in maxima)
    support = create_galerkin_product_support(
        state_indices=_sorted_rows(state_rows),
        interaction_indices=_sorted_rows(interaction_rows),
        absorber_indices=_sorted_rows(absorber_rows),
        work_indices=_sorted_rows(work_rows),
        work_shape=work_shape,
    )
    voltage_kv = jnp.asarray(200.0, dtype=jnp.float64)
    wavenumber = stored_wavenumber(voltage_kv)
    transverse_length = _step_negative(
        float(2.0 * jnp.pi / (0.6 * wavenumber)),
        5,
    )
    normal_length = _step_negative(
        float(2.0 * jnp.pi / (0.2 * wavenumber)),
        10,
    )
    box_lengths = (
        float(transverse_length),
        float(transverse_length),
        float(normal_length),
    )
    nx, ny, nz = 7, 7, 23
    interaction_frequencies = (
        np.asarray(support.interaction_indices, dtype=np.float64)
        / np.asarray(box_lengths, dtype=np.float64)[None, :]
    )
    maximum_frequency = float(
        np.max(np.linalg.norm(interaction_frequencies, axis=-1))
    )
    common_nyquist = min(
        nx / (2.0 * box_lengths[0]),
        ny / (2.0 * box_lengths[1]),
        nz / (2.0 * box_lengths[2]),
    )
    potential = create_potential_3d(
        jnp.full((nz, ny, nx), 0.01, dtype=jnp.float64),
        voxel_size=(
            box_lengths[0] / nx,
            box_lengths[1] / ny,
            box_lengths[2] / nz,
        ),
        box_size=box_lengths,
        origin=(0.0, 0.0, 0.0),
        producer="represented-source-production-fixture-v1",
        provenance_hash="e" * 64,
        coefficient_normalization=("VC-1 periodic trigonometric mean DFT"),
        band_limit=0.5 * (maximum_frequency + common_nyquist),
    )
    eligibility = checked_acquisition(
        support,
        box_lengths,
        voltage_kv=float(voltage_kv),
        terminal_axis=2,
        carrier_direction=(0.0, 0.0, 1.0),
        backward_disposition=GalerkinBackwardDisposition.REPRESENTED,
        claims_backscatter=True,
    )
    return create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=voltage_kv,
        cap_scale=0.25,
        target_name="exact-represented-source-test",
    )


def _position(
    manifest: GalerkinTargetManifest, row: Tuple[int, int, int]
) -> int:
    """Return the canonical state-array position of one exact row."""
    matches = np.all(
        np.asarray(manifest.support.state_indices) == np.asarray(row),
        axis=1,
    )
    return int(np.flatnonzero(matches)[0])


def _source_kwargs() -> Dict[str, object]:
    """Return explicit no-phase exact-shell keyword arguments."""
    return {
        "normal_axis": GalerkinSourceAxis.Z,
        "phase_convention": GalerkinSourcePhaseConvention.PHYSICAL_WAVEVECTOR,
        "stored_shell_route": (GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL),
        "shell_defect_tolerance": jnp.asarray(0.0, dtype=jnp.float64),
        "source_plane_coordinate": jnp.asarray(0.0, dtype=jnp.float64),
        "scan_position": jnp.zeros((3,), dtype=jnp.float64),
    }


def _dense_actions(
    manifest: GalerkinTargetManifest,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Build independent dense ``B``, ``R``, and ``H_0`` action matrices."""
    size = manifest.support.state_indices.shape[0]
    basis = jnp.eye(size, dtype=jnp.complex128)
    cap = jnp.stack(
        [
            manifest.cap_scale
            * apply_absorber_action(
                manifest.support,
                manifest.absorber_coefficients,
                basis[:, column],
            )
            for column in range(size)
        ],
        axis=1,
    )
    interaction = jnp.stack(
        [
            apply_interaction_product(
                manifest.support,
                manifest.interaction_coefficients,
                basis[:, column],
            )
            for column in range(size)
        ],
        axis=1,
    )
    vacuum = jnp.diag(manifest.free_diagonal) - 1j * cap
    return cap, interaction, vacuum


class TestRepresentedSources:
    """Verify represented stored-shell plane and coherent sources.

    :see: :func:`ptyrodactyl.born.build_represented_focused_galerkin_source`
    :see: :func:`ptyrodactyl.born.build_represented_plane_galerkin_source`
    :see: :class:`ptyrodactyl.types.GalerkinRepresentedSource`
    :see: :class:`ptyrodactyl.types.GalerkinSourceActions`
    :see: :class:`ptyrodactyl.types.GalerkinSourceModes`
    :see: :func:`ptyrodactyl.types.create_galerkin_source_modes`
    """

    def test_plane_mode_matches_dense_h0_and_reproduces_empty_box(
        self,
    ) -> None:
        """Include both ``Dv`` and ``Bv`` in the unique matched injection."""
        manifest = _manifest()
        position = _position(manifest, (0, 0, 0))
        source = build_represented_plane_galerkin_source(
            manifest=manifest,
            state_position=position,
            aperture_weight=jnp.asarray(2.0 - 0.5j, dtype=jnp.complex128),
            target_reduced_flux=jnp.asarray(2.5, dtype=jnp.float64),
            aberration_phase=jnp.asarray(0.0, dtype=jnp.float64),
            **_source_kwargs(),
        )
        jax.block_until_ready(source)
        cap, _, vacuum = _dense_actions(manifest)

        expected_input_flux = (
            manifest.wavenumber * (2.0**2 + 0.5**2) / manifest.box_lengths[2]
        )
        np.testing.assert_allclose(
            source.modes.aperture_reduced_flux,
            expected_input_flux,
            rtol=2.0e-14,
        )
        np.testing.assert_allclose(
            source.modes.output_reduced_flux,
            source.modes.target_reduced_flux,
            rtol=2.0e-14,
        )
        np.testing.assert_array_equal(source.actions.free_action, 0.0)
        np.testing.assert_allclose(
            source.actions.cap_action,
            cap @ source.modes.incident_field,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            source.actions.incident_source,
            vacuum @ source.modes.incident_field,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        reproduced = jnp.linalg.solve(vacuum, source.actions.incident_source)
        np.testing.assert_allclose(
            reproduced,
            source.modes.incident_field,
            rtol=3.0e-12,
            atol=3.0e-12,
        )
        assert source.kind is GalerkinRepresentedSourceKind.PLANE_MODE
        assert source.error_enclosure.route is (
            GalerkinSourceErrorRoute.FTZ_SAFE_DIRECT_INTERVAL_BRIDGE
        )
        assert source.support_eligible
        assert source.declared_incident_eligible
        assert source.exact_shell_eligible
        assert source.exact_flux_eligible
        assert source.action_enclosures_eligible
        assert source.rm_s3_eligible
        assert source.error_enclosure.finite_certificate
        final_action_bounds = jnp.asarray(
            (
                source.error_enclosure.free_action_error_upper_bound,
                source.error_enclosure.cap_action_error_upper_bound,
                source.error_enclosure.matched_source_error_upper_bound,
                source.error_enclosure.interaction_action_error_upper_bound,
                source.error_enclosure.total_source_error_upper_bound,
                source.error_enclosure.scattered_source_error_upper_bound,
            )
        )
        assert jnp.all(jnp.isfinite(final_action_bounds))
        assert source.modes.exact_reduced_flux_lower_bound > 0.0
        assert jnp.isfinite(source.modes.exact_reduced_flux_upper_bound)
        for endpoint in (
            source.modes.exact_reduced_flux_lower_bound,
            source.modes.exact_reduced_flux_upper_bound,
        ):
            assert (
                jnp.abs(source.modes.target_reduced_flux - endpoint)
                <= source.modes.target_reduced_flux_discrepancy_upper_bound
            )

        exact_free = _exact_free_action(manifest, source.modes.incident_field)
        exact_cap = _scale_exact_action(
            float(np.asarray(manifest.cap_scale).item()),
            _exact_multiplier_action(
                manifest.support.state_indices,
                manifest.support.absorber_indices,
                manifest.absorber_coefficients,
                source.modes.incident_field,
            ),
        )
        exact_interaction = _exact_multiplier_action(
            manifest.support.state_indices,
            manifest.support.interaction_indices,
            manifest.interaction_coefficients,
            source.modes.incident_field,
        )
        minus_i = (Fraction(0), Fraction(-1))
        exact_matched = [
            _complex_add(free, _complex_multiply(minus_i, cap_value))
            for free, cap_value in zip(exact_free, exact_cap, strict=True)
        ]
        for stored, exact, bound in (
            (
                source.actions.free_action,
                exact_free,
                source.error_enclosure.free_action_error_upper_bound,
            ),
            (
                source.actions.cap_action,
                exact_cap,
                source.error_enclosure.cap_action_error_upper_bound,
            ),
            (
                source.actions.incident_source,
                exact_matched,
                source.error_enclosure.matched_source_error_upper_bound,
            ),
            (
                source.actions.interaction_action,
                exact_interaction,
                source.error_enclosure.interaction_action_error_upper_bound,
            ),
            (
                source.actions.total_source,
                exact_matched,
                source.error_enclosure.total_source_error_upper_bound,
            ),
            (
                source.actions.scattered_source,
                exact_interaction,
                source.error_enclosure.scattered_source_error_upper_bound,
            ),
        ):
            assert _exact_error_norm(stored, exact) <= _decimal_bound(bound)

        enclosure = source.error_enclosure
        np.testing.assert_array_equal(
            enclosure.exact_target_matched_source_error_upper_bound,
            _upward_add(
                enclosure.matched_source_error_upper_bound,
                _upward_add(
                    enclosure.free_target_transfer_error_upper_bound,
                    enclosure.cap_target_transfer_error_upper_bound,
                ),
            ),
        )
        np.testing.assert_array_equal(
            enclosure.exact_target_scattered_source_error_upper_bound,
            _upward_add(
                enclosure.scattered_source_error_upper_bound,
                enclosure.interaction_target_transfer_error_upper_bound,
            ),
        )
        assert "excludes full delta_H" in enclosure.error_scope

        ineligible_manifest = eqx.tree_at(
            lambda value: (
                value.realization.support_eligibility.support_eligible
            ),
            manifest,
            jnp.asarray(False, dtype=jnp.bool_),
        )
        support_rebound = create_represented_galerkin_source(
            manifest=ineligible_manifest,
            modes=source.modes,
            actions=source.actions,
            representation_ledger=source.representation_ledger,
            error_enclosure=source.error_enclosure,
            kind=source.kind,
        )
        assert not support_rebound.support_eligible
        assert not support_rebound.rm_s3_eligible

        nonshell_modes = eqx.tree_at(
            lambda value: value.exact_free_diagonal_lower_bounds,
            source.modes,
            source.modes.exact_free_diagonal_lower_bounds.at[position].set(
                -1.0e-12
            ),
        )
        nonshell_modes = eqx.tree_at(
            lambda value: value.exact_free_diagonal_upper_bounds,
            nonshell_modes,
            nonshell_modes.exact_free_diagonal_upper_bounds.at[position].set(
                1.0e-12
            ),
        )
        shell_rebound = create_represented_galerkin_source(
            manifest=manifest,
            modes=nonshell_modes,
            actions=source.actions,
            representation_ledger=source.representation_ledger,
            error_enclosure=source.error_enclosure,
            kind=source.kind,
        )
        assert not shell_rebound.exact_shell_eligible
        assert not shell_rebound.rm_s3_eligible

    def test_focused_phases_preserve_flux_and_use_one_common_factor(
        self,
    ) -> None:
        """Apply the declared phases without changing coherent flux weights."""
        manifest = _manifest()
        weights = jnp.zeros(
            (manifest.support.state_indices.shape[0],),
            dtype=jnp.complex128,
        )
        for row, value in (
            ((0, 0, 0), 1.0 + 0.25j),
            ((1, 0, -1), -0.5 + 0.75j),
            ((0, 1, -1), 0.125 - 0.25j),
        ):
            weights = weights.at[_position(manifest, row)].set(value)
        aberrations = jnp.linspace(
            -0.3,
            0.4,
            manifest.support.state_indices.shape[0],
            dtype=jnp.float64,
        )
        source = build_represented_focused_galerkin_source(
            manifest=manifest,
            aperture_weights=weights,
            target_reduced_flux=jnp.asarray(3.0, dtype=jnp.float64),
            normal_axis=GalerkinSourceAxis.Z,
            phase_convention=(
                GalerkinSourcePhaseConvention.PHYSICAL_WAVEVECTOR
            ),
            stored_shell_route=(
                GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL
            ),
            shell_defect_tolerance=jnp.asarray(0.0, dtype=jnp.float64),
            source_plane_coordinate=jnp.asarray(0.0125, dtype=jnp.float64),
            scan_position=jnp.asarray((0.004, -0.006, 0.0)),
            aberration_phases=aberrations,
        )
        jax.block_until_ready(source)

        active = np.asarray(source.modes.active_mask)
        ratios = (
            np.asarray(source.modes.incident_field)[active]
            / np.asarray(source.modes.phased_coefficients)[active]
        )
        np.testing.assert_allclose(
            ratios,
            source.modes.flux_normalization,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            source.modes.aperture_reduced_flux,
            source.modes.input_reduced_flux,
            rtol=2.0e-14,
        )
        np.testing.assert_allclose(
            source.modes.output_reduced_flux,
            3.0,
            rtol=2.0e-14,
        )
        assert source.kind is GalerkinRepresentedSourceKind.COHERENT_FOCUSED
        assert source.representation_ledger.route is (
            GalerkinSourceRepresentationRoute.EXACT_PERIODIC_FINITE_TARGET
        )
        exact_stages = (
            source.representation_ledger.box_error_upper_bound,
            source.representation_ledger.carrier_error_upper_bound,
            source.representation_ledger.window_error_upper_bound,
            source.representation_ledger.preband_error_upper_bound,
            source.representation_ledger.band_error_upper_bound,
            source.representation_ledger.algebraic_error_upper_bound,
        )
        np.testing.assert_array_equal(np.asarray(exact_stages), 0.0)
        assert source.support_eligible
        assert not source.declared_incident_eligible
        assert source.exact_flux_eligible
        assert source.action_enclosures_eligible
        assert not source.rm_s3_eligible

    def test_total_and_scattered_dense_residuals_are_identical(self) -> None:
        """Verify the exact RM-S3 total/scattered residual identity."""
        manifest = _manifest()
        position = _position(manifest, (1, 0, -1))
        additional = jnp.linspace(
            0.01,
            0.06,
            manifest.support.state_indices.shape[0],
            dtype=jnp.float64,
        ).astype(jnp.complex128)
        source = build_represented_plane_galerkin_source(
            manifest=manifest,
            state_position=position,
            aperture_weight=1.0 + 0.5j,
            target_reduced_flux=1.0,
            aberration_phase=0.0,
            additional_source=additional,
            **_source_kwargs(),
        )
        jax.block_until_ready(source)
        _, interaction, vacuum = _dense_actions(manifest)
        system = vacuum - interaction
        candidate = jnp.asarray(
            [0.01j * (index + 1) for index in range(system.shape[0])],
            dtype=jnp.complex128,
        )
        total_residual = source.actions.total_source - system @ (
            source.modes.incident_field + candidate
        )
        scattered_residual = (
            source.actions.scattered_source - system @ candidate
        )
        np.testing.assert_allclose(
            total_residual,
            scattered_residual,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        exact_free = _exact_free_action(manifest, source.modes.incident_field)
        exact_cap = _scale_exact_action(
            float(np.asarray(manifest.cap_scale).item()),
            _exact_multiplier_action(
                manifest.support.state_indices,
                manifest.support.absorber_indices,
                manifest.absorber_coefficients,
                source.modes.incident_field,
            ),
        )
        exact_interaction = _exact_multiplier_action(
            manifest.support.state_indices,
            manifest.support.interaction_indices,
            manifest.interaction_coefficients,
            source.modes.incident_field,
        )
        minus_i = (Fraction(0), Fraction(-1))
        exact_additional = [
            _rational_complex(value) for value in np.asarray(additional)
        ]
        exact_matched = [
            _complex_add(free, _complex_multiply(minus_i, cap_value))
            for free, cap_value in zip(exact_free, exact_cap, strict=True)
        ]
        exact_total = [
            _complex_add(matched, extra)
            for matched, extra in zip(
                exact_matched, exact_additional, strict=True
            )
        ]
        exact_scattered = [
            _complex_add(interaction_value, extra)
            for interaction_value, extra in zip(
                exact_interaction, exact_additional, strict=True
            )
        ]
        assert _exact_error_norm(
            source.actions.total_source, exact_total
        ) <= _decimal_bound(
            source.error_enclosure.total_source_error_upper_bound
        )
        assert _exact_error_norm(
            source.actions.scattered_source, exact_scattered
        ) <= _decimal_bound(
            source.error_enclosure.scattered_source_error_upper_bound
        )
        enclosure = source.error_enclosure
        np.testing.assert_array_equal(
            enclosure.exact_target_total_source_error_upper_bound,
            _upward_add(
                enclosure.total_source_error_upper_bound,
                _upward_add(
                    enclosure.free_target_transfer_error_upper_bound,
                    enclosure.cap_target_transfer_error_upper_bound,
                ),
            ),
        )

    def test_focused_builder_is_jittable_and_has_coefficient_jvp(self) -> None:
        """Trace the source and differentiate its common-normalized field."""
        manifest = _manifest()
        positions = (
            _position(manifest, (0, 0, 0)),
            _position(manifest, (1, 0, -1)),
        )
        base = jnp.zeros(
            (manifest.support.state_indices.shape[0],),
            dtype=jnp.complex128,
        )
        base = base.at[positions[0]].set(1.0 + 0.2j)
        base = base.at[positions[1]].set(0.4 - 0.3j)
        aberrations = jnp.zeros_like(jnp.real(base))
        kwargs = _source_kwargs()

        @jax.jit
        def field(weights: jax.Array) -> jax.Array:
            """Return one traced normalized incident coefficient vector."""
            result = build_represented_focused_galerkin_source(
                manifest=manifest,
                aperture_weights=weights,
                target_reduced_flux=1.5,
                aberration_phases=aberrations,
                **kwargs,
            )
            return result.modes.incident_field

        compiled = field(base)
        primal, tangent = jax.jvp(
            field,
            (base,),
            (0.1 * base,),
        )
        jax.block_until_ready((compiled, primal, tangent))
        np.testing.assert_allclose(compiled, primal, rtol=2.0e-14)
        assert jnp.all(jnp.isfinite(tangent))
        assert jnp.linalg.norm(tangent) < 1.0e-12

    @pytest.mark.parametrize(
        ("row", "axis", "message"),
        [
            ((1, 1, -1), GalerkinSourceAxis.Z, "exactly zero represented"),
            ((0, 0, 0), GalerkinSourceAxis.X, "forward with exactly"),
            ((0, 0, -10), GalerkinSourceAxis.Z, "forward with exactly"),
        ],
    )
    def test_plane_builder_rejects_off_shell_grazing_and_backward(
        self,
        row: Tuple[int, int, int],
        axis: GalerkinSourceAxis,
        message: str,
    ) -> None:
        """Fail closed outside the represented stored-shell branch."""
        manifest = _manifest()
        with pytest.raises(_RUNTIME_ERRORS, match=message):
            source = build_represented_plane_galerkin_source(
                manifest=manifest,
                state_position=_position(manifest, row),
                aperture_weight=1.0 + 0.0j,
                target_reduced_flux=1.0,
                normal_axis=axis,
                phase_convention=(
                    GalerkinSourcePhaseConvention.PHYSICAL_WAVEVECTOR
                ),
                stored_shell_route=(
                    GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL
                ),
                shell_defect_tolerance=0.0,
                source_plane_coordinate=0.0,
                scan_position=jnp.zeros((3,), dtype=jnp.float64),
                aberration_phase=0.0,
            )
            jax.block_until_ready(source)

    def test_focused_builder_rejects_duplicate_transverse_fiber(self) -> None:
        """Reject two active normal branches for one transverse harmonic."""
        manifest = _manifest()
        weights = jnp.zeros(
            (manifest.support.state_indices.shape[0],),
            dtype=jnp.complex128,
        )
        weights = weights.at[_position(manifest, (0, 0, 0))].set(1.0)
        weights = weights.at[_position(manifest, (0, 0, -1))].set(0.5)
        with pytest.raises(_RUNTIME_ERRORS, match="one normal branch"):
            source = build_represented_focused_galerkin_source(
                manifest=manifest,
                aperture_weights=weights,
                target_reduced_flux=1.0,
                aberration_phases=jnp.zeros_like(jnp.real(weights)),
                **_source_kwargs(),
            )
            jax.block_until_ready(source)

    def test_result_arrays_use_exact_public_dtypes(self) -> None:
        """Store every numerical carrier leaf in its declared exact dtype."""
        manifest = _manifest()
        source = build_represented_plane_galerkin_source(
            manifest=manifest,
            state_position=_position(manifest, (0, 0, 0)),
            aperture_weight=jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64),
            target_reduced_flux=jnp.asarray(1.0, dtype=jnp.float32),
            aberration_phase=jnp.asarray(0.0, dtype=jnp.float32),
            **_source_kwargs(),
        )
        jax.block_until_ready(source)
        assert isinstance(source, GalerkinRepresentedSource)
        for value in (
            source.modes.aperture_weights,
            source.modes.phased_coefficients,
            source.modes.incident_field,
            source.actions.free_action,
            source.actions.cap_action,
            source.actions.interaction_action,
            source.actions.incident_source,
            source.actions.total_source,
            source.actions.scattered_source,
            source.error_enclosure.independent_direct_cap_action,
            source.error_enclosure.independent_direct_interaction_action,
        ):
            assert value.dtype == jnp.complex128
        for value in (
            source.modes.physical_wavevectors,
            source.modes.shell_defects,
            source.modes.exact_free_diagonal_lower_bounds,
            source.modes.exact_free_diagonal_upper_bounds,
            source.modes.exact_normal_wavevector_lower_bounds,
            source.modes.exact_normal_wavevector_upper_bounds,
            source.modes.scan_position,
            source.modes.aberration_phases,
            source.modes.source_plane_coordinate,
            source.modes.shell_defect_tolerance,
            source.modes.aperture_reduced_flux,
            source.modes.input_reduced_flux,
            source.modes.target_reduced_flux,
            source.modes.output_reduced_flux,
            source.modes.flux_normalization,
            source.modes.exact_reduced_flux_lower_bound,
            source.modes.exact_reduced_flux_upper_bound,
            source.modes.target_reduced_flux_discrepancy_upper_bound,
            source.error_enclosure.free_action_error_upper_bound,
            source.error_enclosure.cap_action_error_upper_bound,
            source.error_enclosure.matched_source_error_upper_bound,
            source.error_enclosure.interaction_action_error_upper_bound,
            source.error_enclosure.total_source_error_upper_bound,
            source.error_enclosure.scattered_source_error_upper_bound,
            source.error_enclosure.incident_field_norm_upper_bound,
            source.error_enclosure.free_target_transfer_error_upper_bound,
            source.error_enclosure.cap_target_transfer_error_upper_bound,
            source.error_enclosure.interaction_target_transfer_error_upper_bound,
            source.error_enclosure.exact_target_matched_source_error_upper_bound,
            source.error_enclosure.exact_target_total_source_error_upper_bound,
            source.error_enclosure.exact_target_scattered_source_error_upper_bound,
        ):
            assert value.dtype == jnp.float64
        for value in (
            source.modes.active_mask,
            source.modes.forward_mask,
            source.modes.grazing_mask,
            source.modes.backward_mask,
            source.error_enclosure.arithmetic_environment_supported,
            source.error_enclosure.gradual_underflow_supported,
            source.error_enclosure.finite_certificate,
            source.support_eligible,
            source.declared_incident_eligible,
            source.exact_shell_eligible,
            source.exact_flux_eligible,
            source.action_enclosures_eligible,
            source.rm_s3_eligible,
        ):
            assert value.dtype == jnp.bool_
