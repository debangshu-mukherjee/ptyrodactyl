"""Tests for :mod:`ptyrodactyl.born.source_derivatives`.

Extended Summary
----------------
These tests compare the represented-source JVP and VJP with an independently
assembled dense ``H0_alg`` matrix and an analytic normalized-aperture
derivative.  Plane and coherent-focused routes are both exercised through
centered finite differences, dense realification, dot tests, exact gauge
families, eager/JIT parity, and mapped execution.  Dense ``H0_alg`` is an
exact-real oracle for the rounded production callable, not its executable
arithmetic identity.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from jaxtyping import TypeCheckError
from numpy.typing import NDArray

from ptyrodactyl.born import (
    build_represented_focused_galerkin_source,
    build_represented_plane_galerkin_source,
    represented_total_source_jvp,
    represented_total_source_vjp,
)
from ptyrodactyl.types import (
    GalerkinRepresentedSource,
    GalerkinSourceAxis,
    GalerkinSourcePhaseConvention,
    GalerkinStoredShellRoute,
)
from tests.test_ptyrodactyl.test_born.test_sources import (
    _manifest,
    _position,
)

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


class _SourceDirection(NamedTuple):
    """Store one direction in the fixed represented-source chart."""

    aperture: NDArray[np.complex128]
    log_flux: float
    scan: NDArray[np.float64]
    aberrations: NDArray[np.float64]
    source_plane: float


def _source_kwargs() -> Dict[str, object]:
    """Return the common exact-shell source-construction keywords."""
    return {
        "normal_axis": GalerkinSourceAxis.Z,
        "phase_convention": GalerkinSourcePhaseConvention.PHYSICAL_WAVEVECTOR,
        "stored_shell_route": GalerkinStoredShellRoute.EXACT_STORED_DIAGONAL,
        "shell_defect_tolerance": jnp.asarray(0.0, dtype=jnp.float64),
    }


@pytest.fixture(scope="module")
def represented_sources() -> Tuple[GalerkinRepresentedSource, ...]:
    """Build one plane and one coherent source on a shared target."""
    manifest = _manifest()
    size = manifest.support.state_indices.shape[0]
    additional = jnp.asarray(
        [0.004 * (index + 1) * (1.0 - 0.35j) for index in range(size)],
        dtype=jnp.complex128,
    )
    plane = build_represented_plane_galerkin_source(
        manifest=manifest,
        state_position=_position(manifest, (0, 0, 0)),
        aperture_weight=jnp.asarray(1.2 - 0.3j, dtype=jnp.complex128),
        target_reduced_flux=jnp.asarray(1.7, dtype=jnp.float64),
        source_plane_coordinate=jnp.asarray(0.017, dtype=jnp.float64),
        scan_position=jnp.asarray((0.006, -0.009, 0.0)),
        aberration_phase=jnp.asarray(0.23, dtype=jnp.float64),
        additional_source=additional,
        **_source_kwargs(),
    )
    weights = jnp.zeros((size,), dtype=jnp.complex128)
    for row, value in (
        ((0, 0, 0), 0.9 + 0.2j),
        ((1, 0, -1), -0.45 + 0.65j),
        ((0, 1, -1), 0.3 - 0.25j),
    ):
        weights = weights.at[_position(manifest, row)].set(value)
    aberrations = jnp.zeros((size,), dtype=jnp.float64)
    aberrations = aberrations.at[jnp.asarray(np.flatnonzero(weights))].set(
        jnp.asarray((0.11, -0.19, 0.07), dtype=jnp.float64)
    )
    focused = build_represented_focused_galerkin_source(
        manifest=manifest,
        aperture_weights=weights,
        target_reduced_flux=jnp.asarray(2.3, dtype=jnp.float64),
        source_plane_coordinate=jnp.asarray(-0.013, dtype=jnp.float64),
        scan_position=jnp.asarray((-0.004, 0.008, 0.0)),
        aberration_phases=aberrations,
        additional_source=-0.6j * additional,
        **_source_kwargs(),
    )
    jax.block_until_ready((plane, focused))
    return plane, focused


def _coefficient_matrix(
    state_indices: NDArray[np.int64],
    multiplier_indices: NDArray[np.int64],
    coefficients: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Assemble one compressed multiplier by exact index lookup."""
    coefficient_map = {
        tuple(int(component) for component in index): coefficient
        for index, coefficient in zip(
            multiplier_indices,
            coefficients,
            strict=True,
        )
    }
    return np.asarray(
        [
            [
                coefficient_map.get(
                    tuple(int(component) for component in row - column),
                    0.0j,
                )
                for column in state_indices
            ]
            for row in state_indices
        ],
        dtype=np.complex128,
    )


def _dense_h0(source: GalerkinRepresentedSource) -> NDArray[np.complex128]:
    """Assemble ``D_alg - i B_alg`` without a production action call."""
    manifest = source.manifest
    cap = float(manifest.cap_scale) * _coefficient_matrix(
        np.asarray(manifest.support.state_indices, dtype=np.int64),
        np.asarray(manifest.support.absorber_indices, dtype=np.int64),
        np.asarray(manifest.absorber_coefficients, dtype=np.complex128),
    )
    return np.diag(np.asarray(manifest.free_diagonal)) - 1.0j * cap


def _dense_interaction(
    source: GalerkinRepresentedSource,
) -> NDArray[np.complex128]:
    """Assemble the stored interaction independently by index lookup."""
    manifest = source.manifest
    return _coefficient_matrix(
        np.asarray(manifest.support.state_indices, dtype=np.int64),
        np.asarray(manifest.support.interaction_indices, dtype=np.int64),
        np.asarray(manifest.interaction_coefficients, dtype=np.complex128),
    )


def _incident(
    source: GalerkinRepresentedSource,
    aperture: NDArray[np.complex128],
    log_flux: float,
    scan: NDArray[np.float64],
    aberrations: NDArray[np.float64],
    source_plane: float,
) -> NDArray[np.complex128]:
    """Evaluate the normalized represented incident chart with NumPy."""
    wavevectors = np.asarray(source.modes.physical_wavevectors)
    normal_axis = int(source.modes.normal_axis)
    normal = wavevectors[:, normal_axis]
    length = float(source.manifest.box_lengths[normal_axis])
    aperture_flux = np.sum(normal * np.abs(aperture) ** 2) / length
    target_flux = float(source.modes.target_reduced_flux) * np.exp(log_flux)
    phase = -(wavevectors @ scan) - normal * source_plane + aberrations
    return (
        np.sqrt(target_flux / aperture_flux) * aperture * np.exp(1.0j * phase)
    )


def _incident_tangent(
    source: GalerkinRepresentedSource,
    direction: _SourceDirection,
) -> NDArray[np.complex128]:
    """Differentiate flux normalization and physical phases analytically."""
    aperture = np.asarray(source.modes.aperture_weights)
    wavevectors = np.asarray(source.modes.physical_wavevectors)
    normal_axis = int(source.modes.normal_axis)
    normal = wavevectors[:, normal_axis]
    length = float(source.manifest.box_lengths[normal_axis])
    aperture_flux = np.sum(normal * np.abs(aperture) ** 2) / length
    aperture_flux_tangent = (
        2.0
        * np.sum(normal * np.real(np.conj(aperture) * direction.aperture))
        / length
    )
    incident = _incident(
        source,
        aperture,
        0.0,
        np.asarray(source.modes.scan_position),
        np.asarray(source.modes.aberration_phases),
        float(source.modes.source_plane_coordinate),
    )
    phase = (
        -(wavevectors @ direction.scan)
        - normal * direction.source_plane
        + direction.aberrations
    )
    normalization = float(source.modes.target_reduced_flux) ** 0.5
    normalization /= aperture_flux**0.5
    base_phase = (
        -(wavevectors @ np.asarray(source.modes.scan_position))
        - normal * float(source.modes.source_plane_coordinate)
        + np.asarray(source.modes.aberration_phases)
    )
    return normalization * np.exp(
        1.0j * base_phase
    ) * direction.aperture + incident * (
        0.5 * direction.log_flux
        - 0.5 * aperture_flux_tangent / aperture_flux
        + 1.0j * phase
    )


def _dense_tangent(
    source: GalerkinRepresentedSource,
    direction: _SourceDirection,
) -> NDArray[np.complex128]:
    """Apply the independently assembled algebraic vacuum target."""
    return _dense_h0(source) @ _incident_tangent(source, direction)


def _dense_chart(
    source: GalerkinRepresentedSource,
    direction: _SourceDirection,
    step: float,
) -> NDArray[np.complex128]:
    """Evaluate the independently anchored finite source chart."""
    aperture = np.asarray(source.modes.aperture_weights)
    scan = np.asarray(source.modes.scan_position)
    aberrations = np.asarray(source.modes.aberration_phases)
    source_plane = float(source.modes.source_plane_coordinate)
    base = _incident(
        source,
        aperture,
        0.0,
        scan,
        aberrations,
        source_plane,
    )
    varied = _incident(
        source,
        aperture + step * direction.aperture,
        step * direction.log_flux,
        scan + step * direction.scan,
        aberrations + step * direction.aberrations,
        source_plane + step * direction.source_plane,
    )
    return np.asarray(source.actions.total_source) + _dense_h0(source) @ (
        varied - base
    )


def _direction(source: GalerkinRepresentedSource) -> _SourceDirection:
    """Return one mixed, fixed-active-set source direction."""
    active = np.flatnonzero(np.asarray(source.modes.active_mask))
    aperture = np.zeros_like(np.asarray(source.modes.aperture_weights))
    values = (0.13 + 0.07j, -0.08 + 0.11j, 0.05 - 0.09j)
    for position, value in zip(active, values, strict=False):
        aperture[position] = value
    aberrations = np.zeros_like(np.asarray(source.modes.aberration_phases))
    aberrations[active] = np.asarray((-0.08, 0.05, -0.03))[: active.size]
    return _SourceDirection(
        aperture=aperture,
        log_flux=0.21,
        scan=np.asarray((0.005, -0.006, 0.0), dtype=np.float64),
        aberrations=aberrations,
        source_plane=0.004,
    )


def _zero_direction(source: GalerkinRepresentedSource) -> _SourceDirection:
    """Return the zero direction in every admitted coordinate block."""
    return _SourceDirection(
        aperture=np.zeros_like(np.asarray(source.modes.aperture_weights)),
        log_flux=0.0,
        scan=np.zeros((3,), dtype=np.float64),
        aberrations=np.zeros_like(np.asarray(source.modes.aberration_phases)),
        source_plane=0.0,
    )


def _production_jvp(
    source: GalerkinRepresentedSource,
    direction: _SourceDirection,
) -> Tuple[jax.Array, jax.Array]:
    """Call the public JVP with exact binary64 coordinate arrays."""
    return represented_total_source_jvp(
        source,
        jnp.asarray(direction.aperture, dtype=jnp.complex128),
        jnp.asarray(direction.log_flux, dtype=jnp.float64),
        jnp.asarray(direction.scan, dtype=jnp.float64),
        jnp.asarray(direction.aberrations, dtype=jnp.float64),
        jnp.asarray(direction.source_plane, dtype=jnp.float64),
    )


def _realify(values: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return block-ordered real and imaginary coordinates."""
    return np.concatenate((np.real(values), np.imag(values)))


def _coordinate_directions(
    source: GalerkinRepresentedSource,
) -> Tuple[_SourceDirection, ...]:
    """Enumerate every admitted real coordinate of one fixed source stratum."""
    zero = _zero_direction(source)
    active = np.flatnonzero(np.asarray(source.modes.active_mask))
    directions: list[_SourceDirection] = []
    for position in active:
        for value in (1.0 + 0.0j, 0.0 + 1.0j):
            aperture = zero.aperture.copy()
            aperture[position] = value
            directions.append(zero._replace(aperture=aperture))
    directions.append(zero._replace(log_flux=1.0))
    normal_axis = int(source.modes.normal_axis)
    for axis in range(3):
        if axis != normal_axis:
            scan = zero.scan.copy()
            scan[axis] = 1.0
            directions.append(zero._replace(scan=scan))
    for position in active:
        phases = zero.aberrations.copy()
        phases[position] = 1.0
        directions.append(zero._replace(aberrations=phases))
    directions.append(zero._replace(source_plane=1.0))
    return tuple(directions)


class TestRepresentedSourceDerivatives:
    """Verify the fixed-stratum represented-source derivative contract.

    :see: :func:`ptyrodactyl.born.represented_total_source_jvp`
    :see: :func:`ptyrodactyl.born.represented_total_source_vjp`
    """

    def test_plane_and_focused_jvps_match_dense_and_centered_oracles(
        self,
        represented_sources: Tuple[GalerkinRepresentedSource, ...],
    ) -> None:
        """Match analytic dense derivatives and a pre-roundoff step sweep."""
        assert represented_sources[0].rm_s3_eligible
        assert not represented_sources[1].rm_s3_eligible
        for source in represented_sources:
            direction = _direction(source)
            primal, tangent = _production_jvp(source, direction)
            expected = _dense_tangent(source, direction)
            np.testing.assert_array_equal(
                primal,
                source.actions.total_source,
            )
            np.testing.assert_allclose(
                tangent,
                expected,
                rtol=2.0e-12,
                atol=2.0e-12,
            )
            errors = []
            for step in (2.0e-3, 1.0e-3, 5.0e-4):
                centered = (
                    _dense_chart(source, direction, step)
                    - _dense_chart(source, direction, -step)
                ) / (2.0 * step)
                errors.append(np.linalg.norm(centered - np.asarray(tangent)))
            assert errors[1] < 0.27 * errors[0]
            assert errors[2] < 0.27 * errors[1]
            assert errors[2] < 2.0e-7

    def test_plane_and_focused_vjps_match_dense_transposes_and_dot_tests(
        self,
        represented_sources: Tuple[GalerkinRepresentedSource, ...],
    ) -> None:
        """Match the full realified dense transpose under declared metrics."""
        for source in represented_sources:
            size = source.actions.total_source.shape[0]
            output_cotangent = np.asarray(
                [(0.17 - 0.09j) * (index + 1) for index in range(size)],
                dtype=np.complex128,
            )
            result = represented_total_source_vjp(
                source,
                jnp.asarray(output_cotangent),
            )
            primal, aperture, log_flux, scan, phases, source_plane = result
            np.testing.assert_array_equal(primal, source.actions.total_source)
            coordinate_directions = _coordinate_directions(source)
            dense_jacobian = np.stack(
                [
                    _realify(_dense_tangent(source, direction))
                    for direction in coordinate_directions
                ],
                axis=1,
            )
            expected = dense_jacobian.T @ _realify(output_cotangent)
            active = np.flatnonzero(np.asarray(source.modes.active_mask))
            returned: list[float] = []
            for position in active:
                returned.extend(
                    (
                        float(jnp.real(aperture[position])),
                        float(jnp.imag(aperture[position])),
                    )
                )
            returned.append(float(log_flux))
            normal_axis = int(source.modes.normal_axis)
            returned.extend(
                float(scan[axis]) for axis in range(3) if axis != normal_axis
            )
            returned.extend(float(phases[position]) for position in active)
            returned.append(float(source_plane))
            np.testing.assert_allclose(
                returned,
                expected,
                rtol=3.0e-12,
                atol=3.0e-12,
            )
            direction = _direction(source)
            _, tangent = _production_jvp(source, direction)
            left = np.real(np.vdot(output_cotangent, np.asarray(tangent)))
            right = (
                np.real(np.vdot(np.asarray(aperture), direction.aperture))
                + float(log_flux) * direction.log_flux
                + float(np.dot(np.asarray(scan), direction.scan))
                + float(np.dot(np.asarray(phases), direction.aberrations))
                + float(source_plane) * direction.source_plane
            )
            np.testing.assert_allclose(left, right, rtol=3.0e-12, atol=3.0e-12)

    def test_exact_redundant_gauge_families_are_null(
        self,
        represented_sources: Tuple[GalerkinRepresentedSource, ...],
    ) -> None:
        """Retain all four exact null families for later RM-I3 quotienting."""
        for source in represented_sources:
            zero = _zero_direction(source)
            aperture = np.asarray(source.modes.aperture_weights)
            active = np.asarray(source.modes.active_mask)
            wavevectors = np.asarray(source.modes.physical_wavevectors)
            normal_axis = int(source.modes.normal_axis)

            scale = zero._replace(aperture=0.37 * aperture)
            phase_values = np.zeros_like(zero.aberrations)
            phase_values[active] = np.linspace(0.13, 0.31, np.sum(active))
            aperture_phase = zero._replace(
                aperture=1.0j * phase_values * aperture,
                aberrations=-phase_values,
            )
            scan_value = np.asarray((0.003, -0.005, 0.0))
            scan_phases = np.zeros_like(zero.aberrations)
            scan_phases[active] = (wavevectors @ scan_value)[active]
            scan_phase = zero._replace(
                scan=scan_value,
                aberrations=scan_phases,
            )
            plane_value = 0.007
            plane_phases = np.zeros_like(zero.aberrations)
            plane_phases[active] = (wavevectors[:, normal_axis] * plane_value)[
                active
            ]
            plane_phase = zero._replace(
                aberrations=plane_phases,
                source_plane=plane_value,
            )
            for direction in (scale, aperture_phase, scan_phase, plane_phase):
                _, tangent = _production_jvp(source, direction)
                np.testing.assert_allclose(tangent, 0.0, atol=2.0e-11)
                np.testing.assert_allclose(
                    _dense_tangent(source, direction),
                    0.0,
                    atol=2.0e-11,
                )

    def test_jit_and_vmap_preserve_both_source_routes(
        self,
        represented_sources: Tuple[GalerkinRepresentedSource, ...],
    ) -> None:
        """Exercise compiled and mapped JVP/VJP paths for each source kind."""
        for source in represented_sources:
            direction = _direction(source)

            @jax.jit
            def compiled_jvp(
                aperture: jax.Array,
                log_flux: jax.Array,
                scan: jax.Array,
                phases: jax.Array,
                source_plane: jax.Array,
            ) -> Tuple[jax.Array, jax.Array]:
                """Compile the public represented-source JVP."""
                return represented_total_source_jvp(
                    source,
                    aperture,
                    log_flux,
                    scan,
                    phases,
                    source_plane,
                )

            arguments = (
                jnp.asarray(direction.aperture),
                jnp.asarray(direction.log_flux),
                jnp.asarray(direction.scan),
                jnp.asarray(direction.aberrations),
                jnp.asarray(direction.source_plane),
            )
            eager = represented_total_source_jvp(source, *arguments)
            compiled = compiled_jvp(*arguments)
            jax.block_until_ready(compiled)
            np.testing.assert_array_equal(eager[0], compiled[0])
            np.testing.assert_allclose(
                eager[1],
                compiled[1],
                rtol=3.0e-13,
                atol=3.0e-18,
            )

            factors = jnp.asarray((0.5, -0.75, 1.25), dtype=jnp.float64)
            mapped = jax.vmap(compiled_jvp)(
                factors[:, None] * arguments[0][None, :],
                factors * arguments[1],
                factors[:, None] * arguments[2][None, :],
                factors[:, None] * arguments[3][None, :],
                factors * arguments[4],
            )
            np.testing.assert_array_equal(
                mapped[0],
                jnp.broadcast_to(eager[0], mapped[0].shape),
            )
            np.testing.assert_allclose(
                mapped[1],
                factors[:, None] * eager[1][None, :],
                rtol=3.0e-13,
                atol=3.0e-13,
            )

            cotangent = jnp.asarray(
                np.linspace(0.1, 0.6, source.actions.total_source.shape[0])
                * (1.0 - 0.4j),
                dtype=jnp.complex128,
            )
            compiled_vjp = jax.jit(
                lambda value: represented_total_source_vjp(source, value)
            )
            eager_vjp = represented_total_source_vjp(source, cotangent)
            mapped_vjp = jax.vmap(compiled_vjp)(
                factors[:, None] * cotangent[None, :]
            )
            jax.block_until_ready(mapped_vjp)
            np.testing.assert_array_equal(
                mapped_vjp[0],
                jnp.broadcast_to(eager_vjp[0], mapped_vjp[0].shape),
            )
            for mapped_block, eager_block in zip(
                mapped_vjp[1:],
                eager_vjp[1:],
                strict=True,
            ):
                np.testing.assert_allclose(
                    mapped_block,
                    factors.reshape((-1,) + (1,) * eager_block.ndim)
                    * eager_block,
                    rtol=3.0e-13,
                    atol=3.0e-13,
                )

    def test_tangent_uses_rounded_h0_callable_without_exact_claim(
        self,
        represented_sources: Tuple[GalerkinRepresentedSource, ...],
    ) -> None:
        """Compare the rounded free-plus-CAP tangent with dense ``H0_alg``."""
        for source in represented_sources:
            direction = _direction(source)
            primal, tangent = _production_jvp(source, direction)
            incident_tangent = _incident_tangent(source, direction)
            h0_tangent = _dense_h0(source) @ incident_tangent
            full_target_tangent = (
                _dense_h0(source) - _dense_interaction(source)
            ) @ incident_tangent
            np.testing.assert_array_equal(primal, source.actions.total_source)
            assert (
                np.linalg.norm(np.asarray(source.actions.additional_source))
                > 0.0
            )
            np.testing.assert_allclose(tangent, h0_tangent, atol=2.0e-12)
            assert (
                np.linalg.norm(np.asarray(tangent) - full_target_tangent)
                > 1.0e-8
            )

    def test_fixed_stratum_checks_fail_closed(
        self,
        represented_sources: Tuple[GalerkinRepresentedSource, ...],
    ) -> None:
        """Reject inactive, normal, nonfinite, and wrong-shaped directions."""
        source = represented_sources[0]
        direction = _direction(source)
        inactive = int(
            np.flatnonzero(~np.asarray(source.modes.active_mask))[0]
        )
        invalid_aperture = direction.aperture.copy()
        invalid_aperture[inactive] = 1.0j
        with pytest.raises(_RUNTIME_ERRORS, match="fixed active aperture"):
            result = _production_jvp(
                source,
                direction._replace(aperture=invalid_aperture),
            )
            jax.block_until_ready(result)
        with pytest.raises(_RUNTIME_ERRORS, match="exactly transverse"):
            result = _production_jvp(
                source,
                direction._replace(scan=np.asarray((0.0, 0.0, 1.0))),
            )
            jax.block_until_ready(result)
        with pytest.raises(_RUNTIME_ERRORS, match="must be finite"):
            result = represented_total_source_vjp(
                source,
                jnp.full_like(source.actions.total_source, jnp.nan + 0.0j),
            )
            jax.block_until_ready(result)
        with pytest.raises(
            (TypeCheckError, ValueError),
            match="state shape|typechecking",
        ):
            represented_total_source_jvp(
                source,
                jnp.zeros((1,), dtype=jnp.complex128),
                jnp.asarray(direction.log_flux),
                jnp.asarray(direction.scan),
                jnp.asarray(direction.aberrations),
                jnp.asarray(direction.source_plane),
            )
