"""Tests for :mod:`ptyrodactyl.born.derivatives`.

Extended Summary
----------------
The tests build one canonical tilted target through
``Potential3D -> checked acquisition -> create_galerkin_target``.  Independent
NumPy DFT, dense-matrix, realified-Jacobian, rotation-chart, and centered-step
oracles never call a production action or private manifest factory.
"""

from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Tuple
from jaxtyping import Complex, Float, Int, TypeCheckError
from numpy.typing import NDArray

from ptyrodactyl.born import (
    create_galerkin_target,
    create_matched_galerkin_source,
    galerkin_state_jvp,
    galerkin_state_vjp,
)
from ptyrodactyl.types import GalerkinSource, GalerkinTargetManifest
from tests._galerkin_target_fixture import (
    TARGET_CAP_SCALE,
    TARGET_VOLTAGE_KV,
    checked_acquisition,
    periodic_target_potential,
    target_support,
)

_CARRIER_DIRECTION: Tuple[float, float, float] = (1.0, -0.018, 0.031)
_SOURCE: Complex[NDArray, "3"] = np.array(
    [0.8 + 0.2j, -0.25 + 0.35j, 0.12 - 0.18j],
    dtype=np.complex128,
)
_OUTPUT_COTANGENT: Complex[NDArray, "3"] = np.array(
    [0.2 + 0.3j, -0.1 + 0.05j, 0.15 - 0.2j],
    dtype=np.complex128,
)
_SOLVER_ARGUMENTS = {
    "max_iterations": 48,
    "relative_tolerance": 1e-13,
    "absolute_tolerance": 1e-14,
}


class _Direction(NamedTuple):
    """Store one real coordinate direction in the physical local chart."""

    name: str
    rotation: Float[NDArray, "3"]
    volume: Float[NDArray, "3 3 5"]
    source: Complex[NDArray, "3"]


def _create_target() -> GalerkinTargetManifest:
    """Create the canonical production target through the sole public path."""
    potential = periodic_target_potential()
    eligibility = checked_acquisition(
        target_support(),
        potential.box_size,
        carrier_direction=_CARRIER_DIRECTION,
    )
    target: GalerkinTargetManifest = create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name="tilted-voxel-derivative-target",
    )
    return target


def _coefficient_matrix(
    state_indices: Int[NDArray, "n 3"],
    support_indices: Int[NDArray, "p 3"],
    coefficients: Complex[NDArray, "p"],
) -> Complex[NDArray, "n n"]:
    """Assemble a multiplier directly from exact index differences."""
    coefficient_map = {
        tuple(int(component) for component in index): coefficient
        for index, coefficient in zip(
            support_indices,
            coefficients,
            strict=True,
        )
    }
    matrix = np.array(
        [
            [
                coefficient_map.get(
                    tuple(int(value) for value in row - column),
                    0.0j,
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
    volume: Float[NDArray, "3 3 5"],
) -> Complex[NDArray, "3"]:
    """Evaluate the ordered VC-1 DFT independently with NumPy."""
    indices = np.asarray(target.support.interaction_indices, dtype=np.int64)
    full = np.fft.fftn(volume) / volume.size
    nz, ny, nx = volume.shape
    residues = np.mod(indices, np.array([nx, ny, nz], dtype=np.int64))
    selected = full[residues[:, 2], residues[:, 1], residues[:, 0]]
    box = np.asarray(target.potential.box_size, dtype=np.float64)
    origin = np.asarray(target.potential.origin, dtype=np.float64)
    phase = np.exp(-2.0j * np.pi * ((indices / box[None, :]) @ origin))
    raw = selected * phase

    index_to_position = {
        tuple(int(component) for component in index): position
        for position, index in enumerate(indices)
    }
    pair_positions = np.array(
        [
            index_to_position[tuple(int(-value) for value in index)]
            for index in indices
        ],
        dtype=np.int64,
    )
    pair_average = 0.5 * (raw + np.conj(raw[pair_positions]))
    canonical = np.array(
        [
            (first > 0)
            or (first == 0 and (second > 0 or (second == 0 and third >= 0)))
            for first, second, third in indices
        ],
        dtype=np.bool_,
    )
    coefficients = np.where(
        canonical,
        pair_average,
        np.conj(pair_average[pair_positions]),
    )
    return coefficients


def _absorber_coefficients(
    target: GalerkinTargetManifest,
) -> Complex[NDArray, "q"]:
    """Evaluate the analytic cosine-shell coefficients independently."""
    indices = np.asarray(target.support.absorber_indices, dtype=np.int64)
    axis = np.where(
        indices == 0,
        0.5,
        np.where(np.abs(indices) == 1, 0.25, 0.0),
    )
    interior = np.prod(axis, axis=-1)
    coefficients = np.where(
        np.all(indices == 0, axis=-1),
        1.0 - interior,
        -interior,
    ).astype(np.complex128)
    return coefficients


def _dense_operator(
    target: GalerkinTargetManifest,
    volume: Float[NDArray, "3 3 5"],
    carrier: Float[NDArray, "3"],
) -> Complex[NDArray, "3 3"]:
    """Assemble SC-1 independently from physical voxel/carrier parameters."""
    state = np.asarray(target.support.state_indices, dtype=np.int64)
    box = np.asarray(target.box_lengths, dtype=np.float64)
    reciprocal_frequencies = state / box[None, :]
    shifted = carrier[None, :] + 2.0 * np.pi * reciprocal_frequencies
    free_diagonal = np.sum(shifted**2, axis=-1) - float(target.wavenumber) ** 2
    voltage_coefficients = _vc1_coefficients(target, volume)
    interaction_coefficients = (
        float(target.interaction_coupling) * voltage_coefficients
    )
    interaction = _coefficient_matrix(
        state,
        np.asarray(target.support.interaction_indices),
        interaction_coefficients,
    )
    absorber = _coefficient_matrix(
        state,
        np.asarray(target.support.absorber_indices),
        _absorber_coefficients(target),
    )
    operator = (
        np.diag(free_diagonal)
        - interaction
        - 1.0j * float(target.cap_scale) * absorber
    )
    return operator


def _create_conforming_source(
    target: GalerkinTargetManifest,
) -> GalerkinSource:
    """Represent the copied right-hand side by production matched injection."""
    state = np.asarray(target.support.state_indices, dtype=np.int64)
    absorber = _coefficient_matrix(
        state,
        np.asarray(target.support.absorber_indices),
        _absorber_coefficients(target),
    )
    free_target = np.diag(np.asarray(target.free_diagonal)) - (
        1.0j * float(target.cap_scale) * absorber
    )
    incident_field = np.linalg.solve(free_target, _SOURCE)
    source: GalerkinSource = create_matched_galerkin_source(
        target,
        jnp.asarray(incident_field),
    )
    return source


def _volume_directions(
    shape: Tuple[int, int, int],
) -> Tuple[Float[NDArray, "3 3 5"], ...]:
    """Return real voxel directions spanning all retained VC-1 coordinates."""
    nz, ny, nx = shape
    x = np.arange(nx, dtype=np.float64)
    ones = np.ones(shape, dtype=np.float64)
    cosine = np.broadcast_to(
        np.cos(2.0 * np.pi * x / nx),
        (nz, ny, nx),
    ).copy()
    sine = np.broadcast_to(
        np.sin(2.0 * np.pi * x / nx),
        (nz, ny, nx),
    ).copy()
    return ones, cosine, sine


def _directions(target: GalerkinTargetManifest) -> Tuple[_Direction, ...]:
    """Build two sphere rotations, three voxel modes, and complex sources."""
    nz: int
    ny: int
    nx: int
    nz, ny, nx = target.potential.volume.shape
    shape: Tuple[int, int, int] = (nz, ny, nx)
    zero_rotation = np.zeros(3, dtype=np.float64)
    zero_volume = np.zeros(shape, dtype=np.float64)
    zero_source = np.zeros(3, dtype=np.complex128)
    directions: list[_Direction] = [
        _Direction(
            "carrier_rotation_y",
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            zero_volume.copy(),
            zero_source.copy(),
        ),
        _Direction(
            "carrier_rotation_z",
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            zero_volume.copy(),
            zero_source.copy(),
        ),
    ]
    for name, volume in zip(
        ("volume_dc", "volume_cosine", "volume_sine"),
        _volume_directions(shape),
        strict=True,
    ):
        directions.append(
            _Direction(
                name,
                zero_rotation.copy(),
                volume,
                zero_source.copy(),
            )
        )
    for index in range(_SOURCE.size):
        real_source = zero_source.copy()
        real_source[index] = 1.0
        imaginary_source = zero_source.copy()
        imaginary_source[index] = 1.0j
        directions.extend(
            (
                _Direction(
                    f"source_{index}_real",
                    zero_rotation.copy(),
                    zero_volume.copy(),
                    real_source,
                ),
                _Direction(
                    f"source_{index}_imaginary",
                    zero_rotation.copy(),
                    zero_volume.copy(),
                    imaginary_source,
                ),
            )
        )
    return tuple(directions)


def _carrier_tangent(
    target: GalerkinTargetManifest,
    direction: _Direction,
) -> Float[NDArray, "3"]:
    """Pull one infinitesimal rotation into the carrier tangent plane."""
    tangent = np.cross(direction.rotation, np.asarray(target.carrier))
    return tangent


def _directional_dense_tangent(
    target: GalerkinTargetManifest,
    source: Complex[NDArray, "3"],
    direction: _Direction,
) -> Complex[NDArray, "3"]:
    """Solve one independently assembled implicit tangent equation."""
    base_volume = np.asarray(target.potential.volume)
    base_carrier = np.asarray(target.carrier)
    operator = _dense_operator(target, base_volume, base_carrier)
    field = np.linalg.solve(operator, source)
    state = np.asarray(target.support.state_indices, dtype=np.int64)
    reciprocal_frequencies = state / np.asarray(target.box_lengths)[None, :]
    shifted = base_carrier[None, :] + 2.0 * np.pi * reciprocal_frequencies
    carrier_tangent = _carrier_tangent(target, direction)
    diagonal_tangent = 2.0 * np.sum(
        shifted * carrier_tangent[None, :],
        axis=-1,
    )
    voltage_tangent = _vc1_coefficients(target, direction.volume)
    interaction_tangent = _coefficient_matrix(
        state,
        np.asarray(target.support.interaction_indices),
        float(target.interaction_coupling) * voltage_tangent,
    )
    operator_tangent = np.diag(diagonal_tangent) - interaction_tangent
    tangent_source = direction.source - operator_tangent @ field
    field_tangent = np.linalg.solve(operator, tangent_source)
    return field_tangent


def _realify_vector(
    values: Complex[NDArray, "3"],
) -> Float[NDArray, "6"]:
    """Map one complex state to the frozen block-ordered real chart."""
    return np.concatenate((values.real, values.imag))


def _dense_realified_jacobian(
    target: GalerkinTargetManifest,
    source: Complex[NDArray, "3"],
    directions: Tuple[_Direction, ...],
) -> Float[NDArray, "6 parameters"]:
    """Assemble every tangent column without production differentiation."""
    columns = [
        _realify_vector(_directional_dense_tangent(target, source, direction))
        for direction in directions
    ]
    return np.column_stack(columns)


def _rotation_matrix(rotation_vector: Float[NDArray, "3"]) -> np.ndarray:
    """Evaluate Rodrigues' map for one nonzero finite-difference rotation."""
    angle = float(np.linalg.norm(rotation_vector))
    if angle == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector / angle
    x, y, z = axis
    cross = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def _aggregate_direction(
    directions: Tuple[_Direction, ...],
    weights: Float[NDArray, "parameters"],
) -> _Direction:
    """Combine all chart families into one directional perturbation."""
    rotation = sum(
        (
            weight * direction.rotation
            for weight, direction in zip(weights, directions, strict=True)
        ),
        np.zeros(3, dtype=np.float64),
    )
    volume = sum(
        (
            weight * direction.volume
            for weight, direction in zip(weights, directions, strict=True)
        ),
        np.zeros_like(directions[0].volume),
    )
    source = sum(
        (
            weight * direction.source
            for weight, direction in zip(weights, directions, strict=True)
        ),
        np.zeros(3, dtype=np.complex128),
    )
    return _Direction("aggregate", rotation, volume, source)


def _parameterized_dense_field(
    target: GalerkinTargetManifest,
    source: Complex[NDArray, "3"],
    directions: Tuple[_Direction, ...],
    parameters: Float[NDArray, "parameters"],
) -> Complex[NDArray, "3"]:
    """Evaluate an independent dense root on the rotation/voxel chart."""
    aggregate = _aggregate_direction(directions, parameters)
    volume = np.asarray(target.potential.volume) + aggregate.volume
    carrier = _rotation_matrix(aggregate.rotation) @ np.asarray(target.carrier)
    candidate_source = source + aggregate.source
    field = np.linalg.solve(
        _dense_operator(target, volume, carrier),
        candidate_source,
    )
    return field


def _mixed_weights(size: int) -> Float[NDArray, "parameters"]:
    """Return a nonzero direction spanning every admitted parameter family."""
    template = np.array(
        [
            0.08,
            -0.04,
            0.03,
            -0.05,
            0.02,
            -0.01,
            0.04,
            0.03,
            -0.02,
            0.01,
            -0.03,
        ],
        dtype=np.float64,
    )
    if size != template.size:
        raise ValueError("direction fixture size changed")
    return template


class TestGalerkinDerivatives:
    """Verify the canonical voxel/carrier/source derivative seam.

    :see: :func:`ptyrodactyl.born.galerkin_state_jvp`
    :see: :func:`ptyrodactyl.born.galerkin_state_vjp`
    """

    def test_jvp_matches_dense_realified_jacobian_jit_and_vmap(self) -> None:
        """Match all rotation, voxel, and complex-source tangent columns."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        source_host = np.asarray(source)
        directions = _directions(target)
        volume_tangents = jnp.asarray(
            np.stack([direction.volume for direction in directions])
        )
        carrier_tangents = jnp.asarray(
            np.stack(
                [
                    _carrier_tangent(target, direction)
                    for direction in directions
                ]
            )
        )
        source_tangents = jnp.asarray(
            np.stack([direction.source for direction in directions])
        )

        def all_tangents(
            candidate_volume_tangents: Float[jax.Array, "parameters 3 3 5"],
            candidate_carrier_tangents: Float[jax.Array, "parameters 3"],
            candidate_source_tangents: Complex[jax.Array, "parameters 3"],
        ) -> Tuple[
            Complex[jax.Array, "parameters 3"],
            Complex[jax.Array, "parameters 3"],
        ]:
            """Vectorize the public JVP over physical chart directions."""
            fields, field_tangents = jax.vmap(
                lambda volume_tangent, carrier_tangent, source_tangent: (
                    galerkin_state_jvp(
                        target,
                        source,
                        volume_tangent,
                        carrier_tangent,
                        source_tangent,
                        **_SOLVER_ARGUMENTS,
                    )
                )
            )(
                candidate_volume_tangents,
                candidate_carrier_tangents,
                candidate_source_tangents,
            )
            return fields, field_tangents

        eager_fields, eager_tangents = all_tangents(
            volume_tangents,
            carrier_tangents,
            source_tangents,
        )
        compiled_fields, compiled_tangents = jax.jit(all_tangents)(
            volume_tangents,
            carrier_tangents,
            source_tangents,
        )
        dense_field = np.linalg.solve(
            _dense_operator(
                target,
                np.asarray(target.potential.volume),
                np.asarray(target.carrier),
            ),
            source_host,
        )
        dense_jacobian = _dense_realified_jacobian(
            target,
            source_host,
            directions,
        )
        production_jacobian = np.column_stack(
            [
                _realify_vector(np.asarray(tangent))
                for tangent in eager_tangents
            ]
        )

        np.testing.assert_allclose(
            eager_fields,
            np.broadcast_to(dense_field, eager_fields.shape),
            rtol=3e-10,
            atol=3e-11,
        )
        np.testing.assert_allclose(
            production_jacobian,
            dense_jacobian,
            rtol=5e-9,
            atol=5e-10,
        )
        chex.assert_trees_all_close(
            compiled_fields,
            eager_fields,
            rtol=2e-11,
            atol=2e-12,
        )
        chex.assert_trees_all_close(
            compiled_tangents,
            eager_tangents,
            rtol=2e-10,
            atol=2e-11,
        )
        assert eager_fields.dtype == jnp.complex128
        assert eager_tangents.dtype == jnp.complex128
        column_norms = np.linalg.norm(production_jacobian, axis=0)
        assert np.all(column_norms > 1e-7)

    def test_jvp_matches_independent_centered_rotation_voxel_sweep(
        self,
    ) -> None:
        """Show centered convergence for one mixed physical direction."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        source_host = np.asarray(source)
        directions = _directions(target)
        weights = _mixed_weights(len(directions))
        aggregate = _aggregate_direction(directions, weights)
        carrier_tangent = np.cross(
            aggregate.rotation,
            np.asarray(target.carrier),
        )
        _, production_tangent = galerkin_state_jvp(
            target,
            source,
            jnp.asarray(aggregate.volume),
            jnp.asarray(carrier_tangent),
            jnp.asarray(aggregate.source),
            **_SOLVER_ARGUMENTS,
        )
        steps = (2e-1, 1e-1, 5e-2, 2.5e-2, 1.25e-2)
        differences = np.stack(
            [
                (
                    _parameterized_dense_field(
                        target,
                        source_host,
                        directions,
                        step * weights,
                    )
                    - _parameterized_dense_field(
                        target,
                        source_host,
                        directions,
                        -step * weights,
                    )
                )
                / (2.0 * step)
                for step in steps
            ]
        )
        errors = np.linalg.norm(
            differences - np.asarray(production_tangent)[None, :],
            axis=1,
        )

        np.testing.assert_allclose(
            production_tangent,
            differences[-1],
            rtol=4e-7,
            atol=3e-9,
        )
        np.testing.assert_allclose(
            errors[1:4] / errors[:3],
            0.25,
            rtol=1e-1,
            atol=3e-3,
        )

    def test_vjp_matches_dense_transpose_and_centered_differences(
        self,
    ) -> None:
        """Match the physical VJP in eager, JIT, dense, and FD paths."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        source_host = np.asarray(source)
        directions = _directions(target)
        output_cotangent = jnp.asarray(_OUTPUT_COTANGENT)

        def pullback(
            candidate_output_cotangent: Complex[jax.Array, "3"],
        ) -> Tuple[
            Complex[jax.Array, "3"],
            Float[jax.Array, "3 3 5"],
            Float[jax.Array, "3"],
            Complex[jax.Array, "3"],
        ]:
            """Evaluate the public physical VJP for one state cotangent."""
            return galerkin_state_vjp(
                target,
                source,
                candidate_output_cotangent,
                **_SOLVER_ARGUMENTS,
            )

        eager = pullback(output_cotangent)
        compiled = jax.jit(pullback)(output_cotangent)
        field, volume_metric_cotangent, carrier_cotangent, source_cotangent = (
            eager
        )
        dense_field = np.linalg.solve(
            _dense_operator(
                target,
                np.asarray(target.potential.volume),
                np.asarray(target.carrier),
            ),
            source_host,
        )
        dense_jacobian = _dense_realified_jacobian(
            target,
            source_host,
            directions,
        )
        output_real_covector = np.concatenate(
            (_OUTPUT_COTANGENT.real, -_OUTPUT_COTANGENT.imag)
        )
        dense_pullback = dense_jacobian.T @ output_real_covector
        voxel_volume = np.prod(np.asarray(target.potential.box_size)) / (
            target.potential.volume.size
        )
        production_pullback = np.array(
            [
                voxel_volume
                * np.sum(
                    np.asarray(volume_metric_cotangent) * direction.volume
                )
                + np.sum(
                    np.asarray(carrier_cotangent)
                    * _carrier_tangent(target, direction)
                )
                + np.real(
                    np.sum(np.asarray(source_cotangent) * direction.source)
                )
                for direction in directions
            ]
        )

        np.testing.assert_allclose(field, dense_field, rtol=3e-10, atol=3e-11)
        np.testing.assert_allclose(
            production_pullback,
            dense_pullback,
            rtol=6e-9,
            atol=6e-10,
        )
        chex.assert_trees_all_close(compiled, eager, rtol=2e-10, atol=2e-11)
        assert field.dtype == jnp.complex128
        assert volume_metric_cotangent.dtype == jnp.float64
        assert carrier_cotangent.dtype == jnp.float64
        assert source_cotangent.dtype == jnp.complex128
        np.testing.assert_allclose(
            np.vdot(np.asarray(target.carrier), np.asarray(carrier_cotangent)),
            0.0,
            rtol=0.0,
            atol=2e-12,
        )

        def dense_loss(parameters: Float[NDArray, "parameters"]) -> float:
            """Contract an independent dense state with the JAX cotangent."""
            candidate_field = _parameterized_dense_field(
                target,
                source_host,
                directions,
                parameters,
            )
            return float(np.real(np.sum(_OUTPUT_COTANGENT * candidate_field)))

        steps = (2e-3, 1e-3, 5e-4)
        zero = np.zeros(len(directions), dtype=np.float64)
        finite_gradients = []
        for step in steps:
            gradient = np.zeros(len(directions), dtype=np.float64)
            for index in range(len(directions)):
                offset = zero.copy()
                offset[index] = step
                gradient[index] = (
                    dense_loss(offset) - dense_loss(-offset)
                ) / (2.0 * step)
            finite_gradients.append(gradient)
        finite_gradient_array = np.stack(finite_gradients)
        errors = np.max(
            np.abs(finite_gradient_array - production_pullback[None, :]),
            axis=1,
        )

        np.testing.assert_allclose(
            production_pullback,
            finite_gradient_array[-1],
            rtol=8e-7,
            atol=3e-9,
        )
        np.testing.assert_allclose(
            errors[1:] / errors[:-1],
            0.25,
            rtol=2e-3,
            atol=2e-4,
        )

    def test_volume_vjp_uses_physical_voxel_metric_dot_pairing(self) -> None:
        """Distinguish the returned physical gradient from JAX's raw one."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        directions = _directions(target)
        weights = _mixed_weights(len(directions))
        aggregate = _aggregate_direction(directions, weights)
        carrier_tangent = np.cross(
            aggregate.rotation,
            np.asarray(target.carrier),
        )
        _, field_tangent = galerkin_state_jvp(
            target,
            source,
            jnp.asarray(aggregate.volume),
            jnp.asarray(carrier_tangent),
            jnp.asarray(aggregate.source),
            **_SOLVER_ARGUMENTS,
        )
        _, volume_metric_cotangent, carrier_cotangent, source_cotangent = (
            galerkin_state_vjp(
                target,
                source,
                jnp.asarray(_OUTPUT_COTANGENT),
                **_SOLVER_ARGUMENTS,
            )
        )
        voxel_volume = np.prod(np.asarray(target.potential.box_size)) / (
            target.potential.volume.size
        )
        state_pairing = np.real(
            np.sum(_OUTPUT_COTANGENT * np.asarray(field_tangent))
        )
        parameter_pairing = (
            voxel_volume
            * np.sum(np.asarray(volume_metric_cotangent) * aggregate.volume)
            + np.sum(np.asarray(carrier_cotangent) * carrier_tangent)
            + np.real(np.sum(np.asarray(source_cotangent) * aggregate.source))
        )

        np.testing.assert_allclose(
            parameter_pairing,
            state_pairing,
            rtol=6e-9,
            atol=6e-10,
        )

    def test_rotation_chart_preserves_carrier_sphere(self) -> None:
        """Use a genuine two-direction rotation chart on the carrier sphere."""
        target = _create_target()
        carrier = np.asarray(target.carrier)
        directions = _directions(target)[:2]

        for direction in directions:
            tangent = _carrier_tangent(target, direction)
            rotated = _rotation_matrix(0.17 * direction.rotation) @ carrier
            np.testing.assert_allclose(
                np.vdot(carrier, tangent),
                0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.linalg.norm(rotated),
                np.linalg.norm(carrier),
                rtol=2e-15,
                atol=2e-13,
            )

    def test_derivative_harness_rejects_wrong_width_arrays(self) -> None:
        """Require the fixed binary64 physical chart."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        volume_tangent = jnp.zeros_like(target.potential.volume)
        carrier_tangent = jnp.zeros_like(target.carrier)
        source_tangent = jnp.zeros_like(source)

        wrong_jvp_arguments = (
            (
                source.astype(jnp.complex64),
                volume_tangent,
                carrier_tangent,
                source_tangent,
            ),
            (
                source,
                volume_tangent.astype(jnp.float32),
                carrier_tangent,
                source_tangent,
            ),
            (
                source,
                volume_tangent,
                carrier_tangent.astype(jnp.float32),
                source_tangent,
            ),
            (
                source,
                volume_tangent,
                carrier_tangent,
                source_tangent.astype(jnp.complex64),
            ),
        )
        for arguments in wrong_jvp_arguments:
            with pytest.raises(TypeCheckError):
                galerkin_state_jvp(target, *arguments, **_SOLVER_ARGUMENTS)

        with pytest.raises(TypeCheckError):
            galerkin_state_vjp(
                target,
                source,
                jnp.asarray(_OUTPUT_COTANGENT, dtype=jnp.complex64),
                **_SOLVER_ARGUMENTS,
            )

    def test_jvp_rejects_directions_outside_the_fixed_chart(self) -> None:
        """Reject wrong voxel shape, radial carrier, and non-finite source."""
        target = _create_target()
        source = _create_conforming_source(target).total_source
        zero_volume = jnp.zeros_like(target.potential.volume)
        zero_carrier = jnp.zeros_like(target.carrier)
        zero_source = jnp.zeros_like(source)

        with pytest.raises(ValueError, match="potential_volume_tangent"):
            galerkin_state_jvp(
                target,
                source,
                jnp.zeros((3, 3, 4), dtype=jnp.float64),
                zero_carrier,
                zero_source,
            )

        with pytest.raises(
            (
                eqx.EquinoxRuntimeError,
                jax.errors.JaxRuntimeError,
                ValueError,
            ),
            match="tangent to the on-shell sphere",
        ):
            radial_result = galerkin_state_jvp(
                target,
                source,
                zero_volume,
                target.carrier,
                zero_source,
            )
            jax.block_until_ready(radial_result[1])

        with pytest.raises(
            (
                eqx.EquinoxRuntimeError,
                jax.errors.JaxRuntimeError,
                ValueError,
            ),
            match="source_tangent must be finite",
        ):
            nonfinite_source = zero_source.at[0].set(jnp.inf + 0.0j)
            result = galerkin_state_jvp(
                target,
                source,
                zero_volume,
                zero_carrier,
                nonfinite_source,
            )
            jax.block_until_ready(result[1])
