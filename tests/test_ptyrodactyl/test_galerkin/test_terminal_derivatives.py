"""Tests for :mod:`ptyrodactyl.galerkin.terminal_derivatives`.

The oracle independently assembles the tiny complex Galerkin matrix, its
realified physical-coordinate derivatives, and the selected-fiber terminal
matrix.  It does not call the production derivative leaf or a private
terminal action.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from numpy.typing import NDArray

from ptyrodactyl.galerkin.acquisition import (
    check_galerkin_acquisition_support,
)
from ptyrodactyl.galerkin.engine import apply_galerkin_operator
from ptyrodactyl.galerkin.system import create_galerkin_target
from ptyrodactyl.galerkin.terminal import (
    certify_galerkin_terminal_current_operator,
    enclose_galerkin_terminal_current,
)
from ptyrodactyl.galerkin.terminal_derivatives import (
    galerkin_terminal_number_current_jvp,
    galerkin_terminal_number_current_vjp,
)
from ptyrodactyl.types.acquisition_types import (
    GalerkinBackwardDisposition,
    GalerkinTerminalSide,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.galerkin_types import GalerkinTargetManifest
from ptyrodactyl.types.terminal_types import (
    GalerkinCurrentOperatorCertificate,
    GalerkinTerminalCurrentScope,
)
from tests._galerkin_target_fixture import (
    TARGET_CAP_SCALE,
    TARGET_VOLTAGE_KV,
    checked_acquisition,
    periodic_target_potential,
)

_SOURCE = np.asarray(
    (
        0.72 + 0.11j,
        -0.18 + 0.27j,
        0.09 - 0.16j,
        -0.31 + 0.08j,
        0.14 + 0.22j,
        -0.06 - 0.13j,
    ),
    dtype=np.complex128,
)
_CARRIER_DIRECTION = (1.0, -0.018, 0.031)
_SOLVER_ARGUMENTS = {
    "max_iterations": 96,
    "relative_tolerance": 2.0e-13,
    "absolute_tolerance": 2.0e-14,
}


class _Direction(NamedTuple):
    """Store one direction in the fixed physical derivative chart."""

    volume: NDArray[np.float64]
    rotation: NDArray[np.float64]
    source: NDArray[np.complex128]


def _build_target(
    terminal_side: GalerkinTerminalSide,
) -> GalerkinTargetManifest:
    """Build two retained fibers while selecting only the first one."""
    state = jnp.asarray(
        [
            (normal, transverse, 0)
            for transverse in (0, 1)
            for normal in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    interaction = jnp.asarray(
        [(normal, 0, 0) for normal in range(-1, 2)],
        dtype=jnp.int64,
    )
    absorber = jnp.asarray(
        [
            (normal, transverse, third)
            for normal in range(-2, 3)
            for transverse in range(-1, 2)
            for third in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    work = jnp.asarray(
        [
            (normal, transverse, third)
            for normal in range(-3, 4)
            for transverse in range(-1, 3)
            for third in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    support: GalerkinProductSupport = create_galerkin_product_support(
        state_indices=state,
        interaction_indices=interaction,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(7, 5, 3),
    )
    potential = periodic_target_potential()
    negative_side = terminal_side is GalerkinTerminalSide.NEGATIVE
    full = checked_acquisition(
        support,
        potential.box_size,
        terminal_side=terminal_side,
        carrier_direction=_CARRIER_DIRECTION,
        backward_disposition=(
            GalerkinBackwardDisposition.REPRESENTED
            if negative_side
            else GalerkinBackwardDisposition.EXCLUDED
        ),
        claims_backscatter=negative_side,
    )
    selected_preterminal = state[state[:, 1] == 0]
    selected_manifest = replace(
        full.manifest,
        preterminal_indices=selected_preterminal,
        transverse_indices=jnp.asarray(((0, 0),), dtype=jnp.int64),
    )
    selected = check_galerkin_acquisition_support(selected_manifest)
    assert bool(selected.support_eligible)
    target: GalerkinTargetManifest = create_galerkin_target(
        potential,
        selected,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name=f"tilted-terminal-derivative-{terminal_side.value}",
    )
    return target


def _certificate(
    target: GalerkinTargetManifest,
) -> GalerkinCurrentOperatorCertificate:
    """Create one canonical operator certificate independently of the root."""
    indices = np.asarray(target.support.state_indices)
    seed = (
        0.21
        + 0.04 * indices[:, 0]
        + 1.0j * (0.13 - 0.03 * indices[:, 0] + 0.02 * indices[:, 1])
    )
    diagnostic = enclose_galerkin_terminal_current(
        target, jnp.asarray(seed, dtype=jnp.complex128)
    )
    certificate = certify_galerkin_terminal_current_operator(diagnostic)
    assert bool(certificate.current_operator_eligible)
    return certificate


@pytest.fixture(scope="module")
def positive_certificate() -> GalerkinCurrentOperatorCertificate:
    """Return the positive-side tilted selected-sector certificate."""
    return _certificate(_build_target(GalerkinTerminalSide.POSITIVE))


@pytest.fixture(scope="module")
def negative_certificate() -> GalerkinCurrentOperatorCertificate:
    """Return the otherwise identical negative-side certificate."""
    return _certificate(_build_target(GalerkinTerminalSide.NEGATIVE))


def _coefficient_matrix(
    state_indices: NDArray[np.int64],
    support_indices: NDArray[np.int64],
    coefficients: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Dense-assemble one Fourier multiplier by exact index differences."""
    coefficient_map = {
        tuple(int(component) for component in index): coefficient
        for index, coefficient in zip(
            support_indices, coefficients, strict=True
        )
    }
    matrix = np.asarray(
        [
            [
                coefficient_map.get(
                    tuple(int(value) for value in row - column), 0.0j
                )
                for column in state_indices
            ]
            for row in state_indices
        ],
        dtype=np.complex128,
    )
    return matrix


def _vc1_coefficients(
    target: GalerkinTargetManifest,
    volume: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Evaluate the ordered VC-1 map by an independent NumPy DFT."""
    indices = np.asarray(target.support.interaction_indices, dtype=np.int64)
    transformed = np.fft.fftn(volume) / volume.size
    nz, ny, nx = volume.shape
    residues = np.mod(indices, np.asarray((nx, ny, nz), dtype=np.int64))
    selected = transformed[residues[:, 2], residues[:, 1], residues[:, 0]]
    box = np.asarray(target.potential.box_size, dtype=np.float64)
    origin = np.asarray(target.potential.origin, dtype=np.float64)
    phased = selected * np.exp(
        -2.0j * np.pi * ((indices / box[None, :]) @ origin)
    )
    positions = {
        tuple(int(component) for component in index): position
        for position, index in enumerate(indices)
    }
    partners = np.asarray(
        [
            positions[tuple(int(-value) for value in index)]
            for index in indices
        ],
        dtype=np.int64,
    )
    averaged = 0.5 * (phased + np.conj(phased[partners]))
    canonical = np.asarray(
        [
            (first > 0)
            or (first == 0 and (second > 0 or (second == 0 and third >= 0)))
            for first, second, third in indices
        ],
        dtype=np.bool_,
    )
    coefficients = np.where(canonical, averaged, np.conj(averaged[partners]))
    return coefficients


def _dense_operator(
    target: GalerkinTargetManifest,
    volume: NDArray[np.float64],
    carrier: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Assemble the anchored fixed-support SC-1 chart independently."""
    state = np.asarray(target.support.state_indices, dtype=np.int64)
    box = np.asarray(target.box_lengths, dtype=np.float64)
    frequencies = state / box[None, :]
    base_carrier = np.asarray(target.carrier, dtype=np.float64)
    shifted = carrier[None, :] + 2.0 * np.pi * frequencies
    base_shifted = base_carrier[None, :] + 2.0 * np.pi * frequencies
    raw_free = (
        np.sum(shifted * shifted, axis=1) - float(target.wavenumber) ** 2
    )
    raw_base_free = (
        np.sum(base_shifted * base_shifted, axis=1)
        - float(target.wavenumber) ** 2
    )
    free = np.asarray(target.free_diagonal) + raw_free - raw_base_free
    voltage_delta = _vc1_coefficients(target, volume) - _vc1_coefficients(
        target, np.asarray(target.potential.volume)
    )
    interaction_coefficients = (
        np.asarray(target.interaction_coefficients)
        + float(target.interaction_coupling) * voltage_delta
    )
    interaction = _coefficient_matrix(
        state,
        np.asarray(target.support.interaction_indices),
        interaction_coefficients,
    )
    absorber = _coefficient_matrix(
        state,
        np.asarray(target.support.absorber_indices),
        np.asarray(target.absorber_coefficients),
    )
    operator = (
        np.diag(free) - interaction - 1.0j * float(target.cap_scale) * absorber
    )
    return operator


def _terminal_rows(target: GalerkinTargetManifest) -> NDArray[np.int64]:
    """Map state positions to selected transverse rows or minus one."""
    axis = target.acquisition.terminal_axis
    transverse_axes = tuple(index for index in range(3) if index != axis)
    selected = {
        tuple(int(component) for component in index): position
        for position, index in enumerate(
            np.asarray(target.acquisition.transverse_indices)
        )
    }
    rows = np.full(target.support.state_indices.shape[0], -1, dtype=np.int64)
    for position, index in enumerate(np.asarray(target.support.state_indices)):
        transverse = tuple(int(index[axis_]) for axis_ in transverse_axes)
        rows[position] = selected.get(transverse, -1)
    return rows


def _dense_terminal_matrices(
    target: GalerkinTargetManifest,
    carrier: NDArray[np.float64],
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Independently assemble selected trace and normal matrices."""
    state = np.asarray(target.support.state_indices)
    rows = _terminal_rows(target)
    axis = target.acquisition.terminal_axis
    length = float(np.asarray(target.box_lengths)[axis])
    normalization = 1.0 / np.sqrt(length)
    sign = (
        1.0
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else -1.0
    )
    trace = np.zeros(
        (target.acquisition.transverse_indices.shape[0], state.shape[0]),
        dtype=np.complex128,
    )
    normal = np.zeros_like(trace)
    for position, row in enumerate(rows):
        if row < 0:
            continue
        wavevector = sign * (
            carrier[axis] + 2.0 * np.pi * float(state[position, axis]) / length
        )
        trace[row, position] = normalization
        normal[row, position] = 1.0j * normalization * wavevector
    return trace, normal


def _dense_current_matrix(
    target: GalerkinTargetManifest,
    carrier: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Assemble ``F=(T* N-N* T)/(2i)`` from independent dense rows."""
    trace, normal = _dense_terminal_matrices(target, carrier)
    current = (trace.conj().T @ normal - normal.conj().T @ trace) / (2.0j)
    return current


def _rotation_matrix(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Rodrigues matrix for one rotation vector."""
    angle = np.linalg.norm(rotation)
    if angle == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = rotation / angle
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def _volume_modes(
    shape: Tuple[int, int, int],
) -> Tuple[NDArray[np.float64], ...]:
    """Span the three retained VC-1 voltage coordinates."""
    nz, ny, nx = shape
    x = np.arange(nx, dtype=np.float64)
    return (
        np.ones(shape, dtype=np.float64),
        np.broadcast_to(np.cos(2.0 * np.pi * x / nx), (nz, ny, nx)).copy(),
        np.broadcast_to(np.sin(2.0 * np.pi * x / nx), (nz, ny, nx)).copy(),
    )


def _coordinate_directions(
    target: GalerkinTargetManifest,
) -> Tuple[_Direction, ...]:
    """Return a realified basis for every active tiny-chart coordinate."""
    nz, ny, nx = target.potential.volume.shape
    shape: Tuple[int, int, int] = (int(nz), int(ny), int(nx))
    zero_volume = np.zeros(shape, dtype=np.float64)
    zero_rotation = np.zeros(3, dtype=np.float64)
    zero_source = np.zeros(_SOURCE.shape, dtype=np.complex128)
    directions = [
        _Direction(
            zero_volume.copy(),
            np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
            zero_source.copy(),
        ),
        _Direction(
            zero_volume.copy(),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            zero_source.copy(),
        ),
    ]
    directions.extend(
        _Direction(mode, zero_rotation.copy(), zero_source.copy())
        for mode in _volume_modes(shape)
    )
    for position in range(_SOURCE.size):
        real_source = zero_source.copy()
        real_source[position] = 1.0
        imaginary_source = zero_source.copy()
        imaginary_source[position] = 1.0j
        directions.extend(
            (
                _Direction(
                    zero_volume.copy(), zero_rotation.copy(), real_source
                ),
                _Direction(
                    zero_volume.copy(), zero_rotation.copy(), imaginary_source
                ),
            )
        )
    return tuple(directions)


def _carrier_tangent(
    target: GalerkinTargetManifest,
    direction: _Direction,
) -> NDArray[np.float64]:
    """Map one rotation direction to the on-shell carrier tangent."""
    tangent = np.cross(direction.rotation, np.asarray(target.carrier))
    return tangent


def _dense_tangent(
    certificate: GalerkinCurrentOperatorCertificate,
    direction: _Direction,
) -> Tuple[NDArray[np.complex128], float, float]:
    """Evaluate the independent dense implicit and terminal differential."""
    target = certificate.diagnostic.target
    volume = np.asarray(target.potential.volume)
    carrier = np.asarray(target.carrier)
    operator = _dense_operator(target, volume, carrier)
    field = np.linalg.solve(operator, _SOURCE)
    carrier_tangent = _carrier_tangent(target, direction)
    state = np.asarray(target.support.state_indices)
    frequencies = state / np.asarray(target.box_lengths)[None, :]
    shifted = carrier[None, :] + 2.0 * np.pi * frequencies
    free_tangent = 2.0 * np.sum(shifted * carrier_tangent[None, :], axis=1)
    interaction_tangent = _coefficient_matrix(
        state,
        np.asarray(target.support.interaction_indices),
        float(target.interaction_coupling)
        * _vc1_coefficients(target, direction.volume),
    )
    operator_tangent = np.diag(free_tangent) - interaction_tangent
    field_tangent = np.linalg.solve(
        operator, direction.source - operator_tangent @ field
    )
    current_matrix = _dense_current_matrix(target, carrier)
    axis = target.acquisition.terminal_axis
    sign = (
        1.0
        if target.acquisition.terminal_side is GalerkinTerminalSide.POSITIVE
        else -1.0
    )
    trace, _ = _dense_terminal_matrices(target, carrier)
    current_tangent_matrix = (
        sign * carrier_tangent[axis] * (trace.conj().T @ trace)
    )
    scale = float(certificate.number_current_scale)
    current = scale * float(np.real(np.vdot(field, current_matrix @ field)))
    tangent = scale * float(
        np.real(
            np.vdot(field_tangent, current_matrix @ field)
            + np.vdot(field, current_matrix @ field_tangent)
            + np.vdot(field, current_tangent_matrix @ field)
        )
    )
    return field, current, tangent


def _aggregate_direction(
    directions: Tuple[_Direction, ...],
) -> _Direction:
    """Combine all coordinate families into one nontrivial direction."""
    weights = np.linspace(-0.19, 0.23, len(directions), dtype=np.float64)
    volume = sum(
        (
            weight * direction.volume
            for weight, direction in zip(weights, directions, strict=True)
        ),
        np.zeros_like(directions[0].volume),
    )
    rotation = sum(
        (
            weight * direction.rotation
            for weight, direction in zip(weights, directions, strict=True)
        ),
        np.zeros_like(directions[0].rotation),
    )
    source = sum(
        (
            weight * direction.source
            for weight, direction in zip(weights, directions, strict=True)
        ),
        np.zeros_like(directions[0].source),
    )
    return _Direction(volume, rotation, source)


def _dense_chart(
    certificate: GalerkinCurrentOperatorCertificate,
    direction: _Direction,
    step: float,
) -> float:
    """Evaluate one finite rotated dense current chart."""
    target = certificate.diagnostic.target
    volume = np.asarray(target.potential.volume) + step * direction.volume
    carrier = _rotation_matrix(step * direction.rotation) @ np.asarray(
        target.carrier
    )
    source = _SOURCE + step * direction.source
    field = np.linalg.solve(_dense_operator(target, volume, carrier), source)
    current_matrix = _dense_current_matrix(target, carrier)
    current = float(certificate.number_current_scale) * float(
        np.real(np.vdot(field, current_matrix @ field))
    )
    return current


class TestTerminalNumberCurrentDerivatives:
    """Bind the selected-sector terminal RM-I1 derivative slice.

    :see: :func:`ptyrodactyl.galerkin.\
galerkin_terminal_number_current_jvp`
    :see: :func:`ptyrodactyl.galerkin.\
galerkin_terminal_number_current_vjp`
    """

    def test_jvp_matches_independent_dense_and_centered_oracles(
        self,
        positive_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Match dense implicit/current math and second-order centered FD."""
        certificate = positive_certificate
        target = certificate.diagnostic.target
        directions = _coordinate_directions(target)
        direction = _aggregate_direction(directions)
        expected_field, expected_current, expected_tangent = _dense_tangent(
            certificate, direction
        )
        field, current, tangent = galerkin_terminal_number_current_jvp(
            certificate,
            jnp.asarray(_SOURCE),
            jnp.asarray(direction.volume),
            jnp.asarray(_carrier_tangent(target, direction)),
            jnp.asarray(direction.source),
            **_SOLVER_ARGUMENTS,
        )
        np.testing.assert_allclose(
            field, expected_field, rtol=4.0e-10, atol=4.0e-11
        )
        np.testing.assert_allclose(
            current, expected_current, rtol=5.0e-12, atol=2.0e-2
        )
        np.testing.assert_allclose(
            tangent, expected_tangent, rtol=8.0e-9, atol=2.0e-2
        )

        steps = (2.0e-3, 1.0e-3, 5.0e-4)
        centered = np.asarray(
            [
                (
                    _dense_chart(certificate, direction, step)
                    - _dense_chart(certificate, direction, -step)
                )
                / (2.0 * step)
                for step in steps
            ]
        )
        errors = np.abs(centered - float(tangent))
        assert errors[-1] < 4.0e10
        np.testing.assert_allclose(
            errors[1:] / errors[:-1], 0.25, rtol=0.08, atol=0.01
        )

    def test_vjp_matches_realified_dense_transpose_and_dot_identity(
        self,
        positive_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Pin complex conjugation and all three physical cotangent blocks."""
        certificate = positive_certificate
        target = certificate.diagnostic.target
        directions = _coordinate_directions(target)
        dense_jacobian = np.asarray(
            [
                _dense_tangent(certificate, direction)[2]
                for direction in directions
            ]
        )
        output_cotangent = np.asarray(0.37, dtype=np.float64)
        field, current, volume, carrier, source = (
            galerkin_terminal_number_current_vjp(
                certificate,
                jnp.asarray(_SOURCE),
                jnp.asarray(output_cotangent),
                **_SOLVER_ARGUMENTS,
            )
        )
        voxel_volume = np.prod(np.asarray(target.potential.box_size)) / (
            target.potential.volume.size
        )
        production_transpose = np.asarray(
            [
                voxel_volume * np.sum(np.asarray(volume) * direction.volume)
                + np.dot(
                    np.asarray(carrier), _carrier_tangent(target, direction)
                )
                + np.real(np.sum(np.asarray(source) * direction.source))
                for direction in directions
            ]
        )
        np.testing.assert_allclose(
            production_transpose,
            output_cotangent * dense_jacobian,
            rtol=1.2e-8,
            atol=5.0e4,
        )
        source_coordinate_pullback = np.asarray(
            [
                component
                for value in np.asarray(source)
                for component in (value.real, -value.imag)
            ]
        )
        np.testing.assert_allclose(
            source_coordinate_pullback,
            production_transpose[5:],
            rtol=1.2e-8,
            atol=5.0e4,
        )
        carrier_vector = np.asarray(target.carrier)
        carrier_cotangent = np.asarray(carrier)
        radial_residual = abs(np.vdot(carrier_vector, carrier_cotangent))
        # Allow 64 binary64 eps for the projection's two dot products,
        # division, scaling, and cancellation at the physical Cj magnitude.
        radial_tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * np.linalg.norm(carrier_vector)
            * np.linalg.norm(carrier_cotangent)
        )
        assert radial_residual <= radial_tolerance

        mixed = _aggregate_direction(directions)
        _, jvp_current, jvp_tangent = galerkin_terminal_number_current_jvp(
            certificate,
            jnp.asarray(_SOURCE),
            jnp.asarray(mixed.volume),
            jnp.asarray(_carrier_tangent(target, mixed)),
            jnp.asarray(mixed.source),
            **_SOLVER_ARGUMENTS,
        )
        right = (
            voxel_volume * np.sum(np.asarray(volume) * mixed.volume)
            + np.dot(np.asarray(carrier), _carrier_tangent(target, mixed))
            + np.real(np.sum(np.asarray(source) * mixed.source))
        )
        np.testing.assert_allclose(
            output_cotangent * float(jvp_tangent),
            right,
            rtol=5.0e-10,
            atol=2.0e3,
        )
        np.testing.assert_allclose(
            field, _dense_tangent(certificate, mixed)[0]
        )
        np.testing.assert_allclose(current, jvp_current, rtol=0.0, atol=0.0)

    def test_side_reversal_and_selected_scope_are_structural(
        self,
        positive_certificate: GalerkinCurrentOperatorCertificate,
        negative_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Reverse the oriented chart and annihilate an unselected fiber."""
        positive = positive_certificate
        negative = negative_certificate
        target = positive.diagnostic.target
        direction = _aggregate_direction(_coordinate_directions(target))
        arguments = (
            jnp.asarray(_SOURCE),
            jnp.asarray(direction.volume),
            jnp.asarray(_carrier_tangent(target, direction)),
            jnp.asarray(direction.source),
        )
        positive_jvp = galerkin_terminal_number_current_jvp(
            positive, *arguments, **_SOLVER_ARGUMENTS
        )
        negative_jvp = galerkin_terminal_number_current_jvp(
            negative, *arguments, **_SOLVER_ARGUMENTS
        )
        np.testing.assert_allclose(negative_jvp[0], positive_jvp[0])
        np.testing.assert_allclose(negative_jvp[1], -positive_jvp[1])
        np.testing.assert_allclose(negative_jvp[2], -positive_jvp[2])

        assert positive.current_scope is (
            GalerkinTerminalCurrentScope.SELECTED_ACQUISITION_FIBER_SECTOR
        )
        assert not hasattr(positive, "vacuum_branch_eligible")
        assert not hasattr(positive, "detector_eligible")
        rows = _terminal_rows(target)
        unselected = rows < 0
        desired_field = np.zeros(rows.shape[0], dtype=np.complex128)
        desired_field[unselected] = np.asarray(
            (0.23 - 0.17j, -0.09 + 0.31j, 0.14 + 0.08j),
            dtype=np.complex128,
        )
        source = apply_galerkin_operator(
            target, jnp.asarray(desired_field, dtype=jnp.complex128)
        )
        zero_volume = jnp.zeros_like(target.potential.volume)
        zero_carrier = jnp.zeros_like(target.carrier)
        zero_source = jnp.zeros_like(source)
        scoped = galerkin_terminal_number_current_jvp(
            positive,
            source,
            zero_volume,
            zero_carrier,
            zero_source,
            **_SOLVER_ARGUMENTS,
        )
        np.testing.assert_allclose(scoped[0], desired_field, atol=2.0e-11)
        np.testing.assert_allclose(scoped[1], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(scoped[2], 0.0, atol=1.0e-7)

    def test_public_boundary_and_chart_checks_fail_closed(
        self,
        positive_certificate: GalerkinCurrentOperatorCertificate,
    ) -> None:
        """Reject forgery, nonfinite data cotangents, and radial tangents."""
        certificate = positive_certificate
        forged = eqx.tree_at(
            lambda record: record.number_current_scale,
            certificate,
            certificate.number_current_scale * 1.001,
        )
        target = certificate.diagnostic.target
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="failed canonical replay"
        ):
            result = galerkin_terminal_number_current_jvp(
                forged,
                jnp.asarray(_SOURCE),
                jnp.zeros_like(target.potential.volume),
                jnp.zeros_like(target.carrier),
                jnp.zeros_like(jnp.asarray(_SOURCE)),
                **_SOLVER_ARGUMENTS,
            )
            jax.block_until_ready(result)

        runtime_errors = (
            eqx.EquinoxRuntimeError,
            jax.errors.JaxRuntimeError,
            ValueError,
        )
        with pytest.raises(runtime_errors, match="must be finite"):
            result = galerkin_terminal_number_current_vjp(
                certificate,
                jnp.asarray(_SOURCE),
                jnp.asarray(np.nan, dtype=jnp.float64),
                **_SOLVER_ARGUMENTS,
            )
            jax.block_until_ready(result)
        with pytest.raises(
            runtime_errors, match="tangent to the on-shell sphere"
        ):
            result = galerkin_terminal_number_current_jvp(
                certificate,
                jnp.asarray(_SOURCE),
                jnp.zeros_like(target.potential.volume),
                target.carrier,
                jnp.zeros_like(jnp.asarray(_SOURCE)),
                **_SOLVER_ARGUMENTS,
            )
            jax.block_until_ready(result)
