r"""Tests for the disjoint LVT.20 local additional-source leaf."""

from __future__ import annotations

import functools
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.local_cell_system as local_system
import ptyrodactyl.galerkin.local_sources as local_sources
from ptyrodactyl.galerkin.local_sources import (
    apply_local_cell_additional_source_map,
    apply_local_cell_additional_source_metric_adjoint,
    certify_local_additional_source,
    prepare_local_additional_source_certificate,
    realize_local_cell_additional_source,
    realize_zero_local_additional_source,
)
from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellTargetManifest,
)
from ptyrodactyl.types.local_source_types import (
    GalerkinLocalAdditionalSourceCertificateFailure,
    GalerkinLocalAdditionalSourceRoute,
)
from tests._galerkin_target_fixture import production_target
from tests.test_ptyrodactyl.test_galerkin.test_absorber import (
    _successful_cap_fixture,
)


@functools.lru_cache(maxsize=4)
def _target(
    name: str = "local-source-test",
) -> GalerkinLocalCellTargetManifest:
    """Return one completed local-cell target with a selected evidence name."""
    proof = _successful_cap_fixture()[2]
    return local_system._compose_prepared(proof, name)


def _complex_cells() -> jax.Array:
    """Return one non-Hermitian complex source-cell field."""
    return jnp.asarray(
        [[[1.0 + 2.0j, -0.5 + 0.25j, 0.75 - 1.5j]]],
        dtype=jnp.complex128,
    )


@functools.lru_cache(maxsize=1)
def _zero_source():
    """Exercise and retain the public ZERO target-authentication path."""
    return realize_zero_local_additional_source(_target())


@functools.lru_cache(maxsize=1)
def _complex_source():
    """Exercise and retain the public LOCAL_CELL target-authentication path."""
    return realize_local_cell_additional_source(_target(), _complex_cells())


@functools.lru_cache(maxsize=1)
def _complex_certificate():
    """Exercise and retain the public direct-certification replay path."""
    return certify_local_additional_source(
        _complex_source(), maximum_direct_terms=9
    )


def _dense_map(
    target: GalerkinLocalCellTargetManifest,
    cells: np.ndarray,
) -> np.ndarray:
    """Evaluate LVT.20b--LVT.20c independently by direct cell summation."""
    nz, ny, nx = cells.shape
    shape_xyz = np.asarray((nx, ny, nz), dtype=np.int64)
    origin = np.asarray(target.local_potential.cell_center_origin)
    lengths = np.asarray(target.local_potential.box_size)
    result: list[complex] = []
    for row in np.asarray(target.state_indices):
        mode = np.asarray(row, dtype=np.int64)
        factor = float(np.prod(np.sinc(mode / shape_xyz)))
        if np.any((mode != 0) & (mode % shape_xyz == 0)):
            factor = 0.0
        direct = 0.0j
        for z_index in range(nz):
            for y_index in range(ny):
                for x_index in range(nx):
                    position = np.asarray((x_index, y_index, z_index))
                    direct += cells[z_index, y_index, x_index] * np.exp(
                        -2.0j * np.pi * np.sum(mode * position / shape_xyz)
                    )
        direct /= cells.size
        origin_phase = np.exp(-2.0j * np.pi * np.sum(mode * origin / lengths))
        result.append(
            np.sqrt(np.prod(lengths)) * factor * origin_phase * direct
        )
    return np.asarray(result, dtype=np.complex128)


def test_zero_route_has_empty_q_and_symbolic_zero_certificate() -> None:
    """ZERO owns no hidden grid and expands no direct source terms.

    :see: :func:`ptyrodactyl.galerkin.realize_zero_local_additional_source`
    """
    source = _zero_source()
    certificate = local_sources._certify_canonical_source(
        source, maximum_direct_terms=1
    )
    assert source.route is GalerkinLocalAdditionalSourceRoute.ZERO
    assert source.source_cell_values.shape == (0,)
    assert_array_equal(source.algebraic_additional_source, 0.0j)
    assert float(source.algebraic_volume_sqrt) == 0.0
    assert certificate.failure is (
        GalerkinLocalAdditionalSourceCertificateFailure.NONE
    )
    assert bool(certificate.finite_certificate)
    assert int(certificate.direct_term_count) == 0
    assert_array_equal(certificate.exact_source_real_lower_bounds, 0.0)
    assert_array_equal(certificate.exact_source_real_upper_bounds, 0.0)
    assert_array_equal(certificate.exact_source_imag_lower_bounds, 0.0)
    assert_array_equal(certificate.exact_source_imag_upper_bounds, 0.0)
    assert_array_equal(certificate.component_error_bounds, 0.0)
    assert float(certificate.additional_source_error_upper_bound) == 0.0
    assert "ZERO or LOCAL_CELL" in certificate.exact_target
    assert "ZERO exact-zero" in source.coefficient_formula
    assert "0 on ZERO" in source.source_formula


def test_complex_map_matches_direct_sum_without_hermitian_projection() -> None:
    """Match dense LVT.20c and retain genuinely complex q asymmetry.

    :see: :func:`ptyrodactyl.galerkin.apply_local_cell_additional_source_map`
    :see: :func:`ptyrodactyl.galerkin.realize_local_cell_additional_source`
    """
    target = _target()
    cells = _complex_cells()
    mapped = apply_local_cell_additional_source_map(target, cells)
    expected = _dense_map(target, np.asarray(cells))
    assert_allclose(mapped, expected, rtol=5.0e-15, atol=5.0e-15)
    source = _complex_source()
    assert source.route is GalerkinLocalAdditionalSourceRoute.LOCAL_CELL
    assert_allclose(
        source.algebraic_additional_source, mapped, rtol=0.0, atol=0.0
    )
    modes = np.asarray(target.state_indices)[:, 0]
    negative = int(np.flatnonzero(modes == -1)[0])
    positive = int(np.flatnonzero(modes == 1)[0])
    assert not np.isclose(
        complex(mapped[negative]), np.conj(complex(mapped[positive]))
    )


def test_complex_metric_adjoint_dense_dot_jit_and_realified_vjp() -> None:
    """Check the exact requested Re-DeltaV pairing and JAX convention.

    :see: :func:`ptyrodactyl.galerkin.\
apply_local_cell_additional_source_metric_adjoint`
    """
    target = _target()
    direction = _complex_cells() * (0.3 - 0.2j)
    cotangent = jnp.asarray(
        [0.7 - 0.4j, -0.2 + 0.9j, 1.1 + 0.3j],
        dtype=jnp.complex128,
    )
    mapped = apply_local_cell_additional_source_map(target, direction)
    adjoint = apply_local_cell_additional_source_metric_adjoint(
        target, cotangent
    )
    cell_volume = (
        np.prod(np.asarray(target.local_potential.box_size)) / direction.size
    )
    left = np.real(
        cell_volume * np.vdot(np.asarray(direction), np.asarray(adjoint))
    )
    right = np.real(np.vdot(np.asarray(mapped), np.asarray(cotangent)))
    assert_allclose(left, right, rtol=5.0e-15, atol=5.0e-14)

    def closed(values):
        """Close over the prepared target for transformation."""
        return apply_local_cell_additional_source_map(target, values)

    assert_allclose(
        jax.jit(closed)(direction), mapped, rtol=4.0e-15, atol=4.0e-15
    )
    _, pullback = jax.vjp(closed, direction)
    euclidean_adjoint = jnp.conj(pullback(jnp.conj(cotangent))[0])
    assert_allclose(
        euclidean_adjoint,
        cell_volume * adjoint,
        rtol=5.0e-15,
        atol=5.0e-14,
    )


def test_source_action_rejects_subnormal_frozen_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject FTZ-unstable volume-sqrt and physical-cell metric factors."""
    target = _target()
    largest_subnormal = np.nextafter(
        np.finfo(np.float64).tiny, 0.0, dtype=np.float64
    )
    exact_tiny_volume = Fraction.from_float(float(largest_subnormal)) ** 2
    with monkeypatch.context() as patch:
        patch.setattr(
            local_sources,
            "_box_volume_fraction",
            lambda submitted: exact_tiny_volume,
        )
        with pytest.raises(ValueError, match="positive, finite, and normal"):
            local_sources._rounded_box_volume_sqrt(target)

    with monkeypatch.context() as patch:
        patch.setattr(
            local_sources,
            "_physical_cell_volume",
            lambda potential: jnp.asarray(
                largest_subnormal, dtype=jnp.float64
            ),
        )
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="positive, finite, and normal",
        ):
            local_sources._prepared_local_cell_map(target, _complex_cells())

    with monkeypatch.context() as patch:
        patch.setattr(
            local_sources,
            "_physical_cell_volume",
            lambda potential: jnp.asarray(np.inf, dtype=jnp.float64),
        )
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match="positive, finite, and normal",
        ):
            apply_local_cell_additional_source_metric_adjoint(
                target,
                jnp.zeros(
                    (target.state_indices.shape[0],), dtype=jnp.complex128
                ),
            )


def test_anisotropic_complex_rectangle_matches_decimal_oracle() -> None:
    """Enclose one all-axis complex zyx oracle beyond a grid Nyquist index."""
    cells = np.asarray(
        [
            [
                [1.0 + 2.0j, -0.5 + 0.25j, 0.75 - 1.5j],
                [0.2 - 0.3j, 1.1 + 0.7j, -0.9 + 0.4j],
            ],
            [
                [-0.1 + 0.6j, 0.8 - 0.2j, 0.33 + 0.44j],
                [1.2 - 1.1j, -0.7 - 0.8j, 0.5 + 0.9j],
            ],
        ],
        dtype=np.complex128,
    )
    mode = (2, -1, 3)
    origin = (0.125, -0.25, 0.2)
    box = (1.25, 2.0, 3.0)
    volume = Fraction(1)
    for length in box:
        volume *= Fraction.from_float(length)
    volume_upper = local_sources.sqrt_fraction_upper(volume)
    rectangle = local_sources._exact_source_rectangle(
        cells,
        mode,
        origin,
        box,
        (volume / volume_upper, volume_upper),
        {},
    )
    exact_real = Decimal(
        "0.025430184996168795879492088749965108078351916359160469611812"
        "25591842580295235214710918259822160908600"
    )
    exact_imag = Decimal(
        "-0.035210145173661857391231984270918291580780272373661876704444"
        "48768391438584362009178034470794107598079"
    )
    with localcontext() as context:
        context.prec = 120
        endpoints = [
            Decimal(value.numerator) / Decimal(value.denominator)
            for value in rectangle
        ]
    assert endpoints[0] <= exact_real <= endpoints[1]
    assert endpoints[2] <= exact_imag <= endpoints[3]
    assert not local_sources._symbolic_shape_zero(mode, (3, 2, 2))
    assert (
        local_sources._exact_source_rectangle(
            cells,
            (3, 0, 0),
            origin,
            box,
            (volume / volume_upper, volume_upper),
            {},
        )
        == (Fraction(0),) * 4
    )


def test_symbolic_qn_modes_are_excluded_from_direct_term_count() -> None:
    """Count beyond-Nyquist aliases but no exact qN sinc-zero products."""
    target = _target()
    modes = jnp.asarray([[0, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=jnp.int64)
    modified_target = eqx.tree_at(
        lambda value: (
            value.cap_floor_proof.coefficient_certificate.absorber.interaction_core.support.state_indices
        ),
        target,
        modes,
    )
    source = local_sources._realize_local_cell_prepared(
        modified_target, _complex_cells()
    )
    certificate = local_sources._certify_canonical_source(
        source, maximum_direct_terms=6
    )
    assert certificate.failure is (
        GalerkinLocalAdditionalSourceCertificateFailure.NONE
    )
    assert int(certificate.direct_term_count) == 6
    assert_array_equal(source.algebraic_additional_source[1], 0.0j)


def test_direct_rectangles_enclose_sqrt_volume_and_lvt20e() -> None:
    """Enclose an independently evaluated complex constant-cell source.

    :see: :func:`ptyrodactyl.galerkin.certify_local_additional_source`
    """
    target = _target()
    cells = jnp.full(
        target.local_potential.cell_values.shape,
        1.0 + 2.0j,
        dtype=jnp.complex128,
    )
    source = local_sources._realize_local_cell_prepared(target, cells)
    certificate = local_sources._certify_canonical_source(
        source, maximum_direct_terms=9
    )
    assert certificate.failure is (
        GalerkinLocalAdditionalSourceCertificateFailure.NONE
    )
    assert bool(certificate.finite_certificate)
    assert int(certificate.direct_term_count) == 9
    modes = np.asarray(target.state_indices)
    zero_position = int(np.flatnonzero(np.all(modes == 0, axis=1))[0])
    with localcontext() as context:
        context.prec = 100
        volume = Decimal(1)
        for length in target.local_potential.box_size:
            volume *= Decimal.from_float(length)
        scale = volume.sqrt()
        exact_real = scale
        exact_imag = Decimal(2) * scale
    real_lower = Decimal.from_float(
        float(certificate.exact_source_real_lower_bounds[zero_position])
    )
    real_upper = Decimal.from_float(
        float(certificate.exact_source_real_upper_bounds[zero_position])
    )
    imag_lower = Decimal.from_float(
        float(certificate.exact_source_imag_lower_bounds[zero_position])
    )
    imag_upper = Decimal.from_float(
        float(certificate.exact_source_imag_upper_bounds[zero_position])
    )
    assert real_lower <= exact_real <= real_upper
    assert imag_lower <= exact_imag <= imag_upper
    for position, mode in enumerate(modes):
        if np.all(mode == 0):
            continue
        assert (
            float(certificate.exact_source_real_lower_bounds[position])
            <= 0.0
            <= float(certificate.exact_source_real_upper_bounds[position])
        )
        assert (
            float(certificate.exact_source_imag_lower_bounds[position])
            <= 0.0
            <= float(certificate.exact_source_imag_upper_bounds[position])
        )
    errors = np.asarray(certificate.component_error_bounds)
    algebraic = np.asarray(source.algebraic_additional_source)
    with localcontext() as context:
        context.prec = 100
        true_component_errors: list[Decimal] = []
        for position, mode in enumerate(modes):
            exact_component_real = scale if np.all(mode == 0) else Decimal(0)
            exact_component_imag = (
                Decimal(2) * scale if np.all(mode == 0) else Decimal(0)
            )
            real_difference = (
                Decimal.from_float(float(np.real(algebraic[position])))
                - exact_component_real
            )
            imag_difference = (
                Decimal.from_float(float(np.imag(algebraic[position])))
                - exact_component_imag
            )
            true_error = (
                real_difference * real_difference
                + imag_difference * imag_difference
            ).sqrt()
            true_component_errors.append(true_error)
            assert Decimal.from_float(float(errors[position])) >= true_error
        true_error_norm = sum(
            (error * error for error in true_component_errors),
            start=Decimal(0),
        ).sqrt()
        exact_stored_error_norm = sum(
            (
                Decimal.from_float(float(error))
                * Decimal.from_float(float(error))
                for error in errors
            ),
            start=Decimal(0),
        ).sqrt()
    assert (
        Decimal.from_float(
            float(certificate.additional_source_error_upper_bound)
        )
        >= exact_stored_error_norm
    )
    assert (
        Decimal.from_float(
            float(certificate.additional_source_error_upper_bound)
        )
        >= true_error_norm
    )


def test_budget_failure_and_zero_vs_all_zero_local_identity() -> None:
    """Return typed work failure and keep the two exact routes distinct."""
    target = _target()
    local_zero = local_sources._realize_local_cell_prepared(
        target,
        jnp.zeros(
            target.local_potential.cell_values.shape, dtype=jnp.complex128
        ),
    )
    symbolic_zero = _zero_source()
    assert local_zero.source_digest != symbolic_zero.source_digest
    assert local_zero.source_cell_values.shape != (0,)
    failed = local_sources._certify_canonical_source(
        local_zero, maximum_direct_terms=1
    )
    assert failed.failure is (
        GalerkinLocalAdditionalSourceCertificateFailure.WORK_BUDGET_EXCEEDED
    )
    assert not bool(failed.finite_certificate)
    assert int(failed.direct_term_count) == 9
    assert np.all(np.isinf(np.asarray(failed.component_error_bounds)))
    assert np.isposinf(float(failed.additional_source_error_upper_bound))


@pytest.mark.parametrize(
    ("failure_kind", "expected_failure"),
    [
        (
            "host",
            GalerkinLocalAdditionalSourceCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED,
        ),
        (
            "root-rectangle",
            GalerkinLocalAdditionalSourceCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        ),
        (
            "root-volume",
            GalerkinLocalAdditionalSourceCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        ),
        (
            "root-component",
            GalerkinLocalAdditionalSourceCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        ),
        (
            "root-norm",
            GalerkinLocalAdditionalSourceCertificateFailure.ROOT_ENCLOSURE_FAILURE,
        ),
        (
            "range",
            GalerkinLocalAdditionalSourceCertificateFailure.ARITHMETIC_RANGE_FAILURE,
        ),
        (
            "subnormal",
            GalerkinLocalAdditionalSourceCertificateFailure.ARITHMETIC_RANGE_FAILURE,
        ),
    ],
)
def test_typed_direct_failures_are_replay_shaped(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_failure: GalerkinLocalAdditionalSourceCertificateFailure,
) -> None:
    """Keep host, root, and arithmetic failures typed and all-infinite."""

    def fail_rectangle(*args, **kwargs):
        """Raise the typed internal root-enclosure signal."""
        del args, kwargs
        raise local_sources.RootEnclosureError("forced test root failure")

    source = _complex_source()
    with monkeypatch.context() as patch:
        if failure_kind == "host":
            patch.setattr(
                local_sources, "host_binary64_supported", lambda: False
            )
        elif failure_kind == "root-rectangle":
            patch.setattr(
                local_sources, "_exact_source_rectangle", fail_rectangle
            )
        elif failure_kind.startswith("root-"):
            root_stage = failure_kind.removeprefix("root-")
            failure_call = {
                "volume": 1,
                "component": 2,
                "norm": source.algebraic_additional_source.shape[0] + 2,
            }[root_stage]
            original_sqrt = local_sources.sqrt_fraction_upper
            calls = 0

            def fail_selected_root(value):
                """Raise at one selected verified-root operation."""
                nonlocal calls
                calls += 1
                if calls == failure_call:
                    raise local_sources.RootEnclosureError(
                        "forced selected root failure"
                    )
                return original_sqrt(value)

            patch.setattr(
                local_sources, "sqrt_fraction_upper", fail_selected_root
            )
        elif failure_kind == "range":
            patch.setattr(
                local_sources, "fraction_lower_float", lambda value: -np.inf
            )
        else:
            subnormal = np.nextafter(
                np.finfo(np.float64).tiny, 0.0, dtype=np.float64
            )
            patch.setattr(
                local_sources,
                "fraction_lower_float",
                lambda value: float(subnormal),
            )
        certificate = local_sources._certify_canonical_source(
            source, maximum_direct_terms=9
        )

    assert certificate.failure is expected_failure
    assert not bool(certificate.finite_certificate)
    assert int(certificate.direct_term_count) <= int(
        certificate.maximum_direct_terms
    )
    assert np.all(
        np.isneginf(np.asarray(certificate.exact_source_real_lower_bounds))
    )
    assert np.all(
        np.isposinf(np.asarray(certificate.exact_source_real_upper_bounds))
    )
    assert np.all(np.isposinf(np.asarray(certificate.component_error_bounds)))
    assert np.isposinf(float(certificate.additional_source_error_upper_bound))


def test_identity_splits_operator_source_from_full_parent_evidence() -> None:
    """Keep target name/proof context out of exact source identity only."""
    first = _target("local-source-name-a")
    second = _target("local-source-name-b")
    assert first.target_digest == second.target_digest
    assert first.manifest_evidence_digest != second.manifest_evidence_digest
    first_source = local_sources._realize_local_cell_prepared(
        first, _complex_cells()
    )
    second_source = local_sources._realize_local_cell_prepared(
        second, _complex_cells()
    )
    assert first_source.source_digest == second_source.source_digest
    assert first_source.realization_digest != second_source.realization_digest


def test_replay_rejects_parent_map_rectangle_and_digest_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recompute target, q map, rectangles, and evidence instead of hashes.

    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_additional_source_certificate`
    """
    source = _complex_source()
    forged_nested_target = replace(source.target, target_digest="a" * 64)
    forged_target = replace(
        source,
        target=forged_nested_target,
    )
    with pytest.raises(ValueError, match="full operator/evidence replay"):
        certify_local_additional_source(forged_target, maximum_direct_terms=9)

    certificate = _complex_certificate()
    forged_map = eqx.tree_at(
        lambda value: value.algebraic_additional_source,
        source,
        source.algebraic_additional_source.at[0].add(1.0e-6),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            local_sources,
            "prepare_local_cell_galerkin_target",
            lambda submitted: source.target,
        )
        with pytest.raises(ValueError, match="target/map replay"):
            certify_local_additional_source(forged_map, maximum_direct_terms=9)

    forged_rectangle = eqx.tree_at(
        lambda value: value.exact_source_real_lower_bounds,
        certificate,
        certificate.exact_source_real_lower_bounds.at[0].add(-1.0e-6),
    )
    forged_certificate = replace(
        forged_rectangle,
        certificate_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="complete host replay"):
        prepare_local_additional_source_certificate(forged_certificate)


def test_dtype_grid_budget_and_legacy_target_fail_closed() -> None:
    """Reject non-complex128 q, wrong grids, invalid budgets, and VC-1."""
    target = _target()
    shape = target.local_potential.cell_values.shape
    with pytest.raises(TypeError, match="source_cell_values"):
        realize_local_cell_additional_source(
            target, jnp.zeros(shape, dtype=jnp.float64)
        )
    with pytest.raises(ValueError, match="complex128 dtype"):
        realize_local_cell_additional_source(
            target, jnp.zeros(shape, dtype=jnp.complex64)
        )
    with pytest.raises(ValueError, match="match the target local grid"):
        realize_local_cell_additional_source(
            target, jnp.zeros((1, 1, 1), dtype=jnp.complex128)
        )
    source = _zero_source()
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive signed-64-bit"):
            certify_local_additional_source(
                source,
                maximum_direct_terms=invalid,
            )
    with pytest.raises(TypeError, match="parameter 'target'"):
        realize_zero_local_additional_source(production_target())  # type: ignore[arg-type]
