r"""Falsification tests for composed local vacuum-terminal evidence."""

from __future__ import annotations

import functools
import inspect
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.local_projection as projection_module
import ptyrodactyl.galerkin.local_vacuum_terminal as terminal
import ptyrodactyl.types.local_vacuum_terminal_types as terminal_types
from ptyrodactyl._tools import (
    ComplexRectangle,
    EntireEnclosureError,
    EntireEnclosureFailure,
    fraction_lower_float,
    fraction_upper_float,
    sha256,
    sqrt_fraction_upper,
    stored_value_payload,
)
from ptyrodactyl.galerkin.absorber import (
    certify_axial_cap_floor,
    certify_axial_cell_absorber,
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
from ptyrodactyl.galerkin.local_cell_system import (
    compose_local_cell_galerkin_target,
)
from ptyrodactyl.galerkin.local_represented_sources import (
    certify_local_represented_source,
    compose_local_represented_plane_source,
)
from ptyrodactyl.galerkin.local_sources import (
    certify_local_additional_source,
    realize_zero_local_additional_source,
)
from ptyrodactyl.galerkin.local_vacuum_propagation import (
    classify_local_vacuum_root,
    enclose_local_vacuum_propagator,
    make_local_vacuum_zero_witness,
)
from ptyrodactyl.galerkin.local_vacuum_terminal import (
    certify_local_vacuum_terminal,
    prepare_local_vacuum_terminal_certificate,
)
from ptyrodactyl.galerkin.local_zero_slab import certify_local_zero_slab
from ptyrodactyl.types.acquisition_types import GalerkinTerminalSide
from ptyrodactyl.types.born_potential_types import (
    create_galerkin_product_support,
)
from ptyrodactyl.types.local_cell_types import (
    create_local_cell_potential_3d,
)
from ptyrodactyl.types.local_projection_types import (
    GalerkinLocalProjectionDefectCertificate,
    GalerkinLocalProjectionDefectFailure,
)
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalRepresentedSourceFailure,
    GalerkinLocalRepresentedSourceKind,
    GalerkinLocalSourcePhaseConvention,
)
from ptyrodactyl.types.local_source_types import (
    GalerkinLocalAdditionalSourceRoute,
)
from ptyrodactyl.types.local_stability_types import (
    GalerkinLocalStabilityDisposition,
    GalerkinLocalStabilityFailure,
    GalerkinLocalStabilityResult,
)
from ptyrodactyl.types.local_terminal_types import (
    GalerkinLocalTerminalComplexRectangles,
    GalerkinLocalTerminalScope,
)
from ptyrodactyl.types.local_vacuum_propagation_types import (
    GalerkinLocalVacuumRootClass,
    GalerkinLocalVacuumZeroWitnessRoute,
)
from ptyrodactyl.types.local_vacuum_terminal_types import (
    GalerkinLocalVacuumHalfSpaceDisposition,
    GalerkinLocalVacuumTerminalCertificate,
    GalerkinLocalVacuumTerminalDisposition,
    GalerkinLocalVacuumTerminalFailure,
)
from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalVacuumReference,
    GalerkinLocalZeroSlabCertificate,
    GalerkinLocalZeroSlabFailure,
)
from tests._galerkin_target_fixture import checked_acquisition
from tests.test_ptyrodactyl.test_galerkin import (
    test_local_projection as projection_tests,
)

_FULL = GalerkinLocalTerminalScope.FULL_STATE_FIBERS
_SELECTED = GalerkinLocalTerminalScope.SELECTED_PRETERMINAL_FIBERS
_PLANE = GalerkinLocalVacuumTerminalDisposition.PLANE_DEFINED_FREE_CONTINUATION
_NATIVE_SELECTED = (
    GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_TERMINAL_SECTOR
)
_NATIVE_FULL = GalerkinLocalVacuumTerminalDisposition.NATIVE_ZERO_DEFECT_SLAB
_STATE_BUDGET = np.float64(np.finfo(np.float64).max)
_STABILITY_PAIRS = 21
_GRAM_PAIRS = 9
_STRUCTURAL_STABILITY_PAIRS = 3
_STRUCTURAL_GRAM_PAIRS = 1
_TERMINAL_PAIRS = 64
_BRANCH_TERMS = 256
_CUT_PAIRS = 64
_ROOT_WORK = 64
_PRECISION = 160
_ENTIRE_TERMS = 4096
_ENTIRE_WORK = 1_000_000
_RANGE_REDUCTIONS = 4096
_INTERVAL_WORK = 1_000_000
_RATIONAL_BITS = 262_144
_ENTIRE_POLICIES = (
    _PRECISION,
    _ENTIRE_TERMS,
    _ENTIRE_WORK,
    _RANGE_REDUCTIONS,
    _RATIONAL_BITS,
)


def _contains(
    rectangle: ComplexRectangle,
    value: tuple[Decimal, Decimal],
) -> bool:
    """Return whether one exact rectangle contains a Decimal complex point."""
    real = Fraction(value[0])
    imag = Fraction(value[1])
    return (
        rectangle[0] <= real <= rectangle[1]
        and rectangle[2] <= imag <= rectangle[3]
    )


def _stored_rectangle(
    rectangles: GalerkinLocalTerminalComplexRectangles,
    index: int,
) -> ComplexRectangle:
    """Interpret one stored complex rectangle exactly."""
    values = tuple(np.asarray(value) for value in rectangles)
    return tuple(Fraction.from_float(float(value[index])) for value in values)  # type: ignore[return-value]


def _interval_multiply(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    """Multiply two exact intervals without production helpers."""
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _rectangle_multiply(
    left: ComplexRectangle,
    right: ComplexRectangle,
) -> ComplexRectangle:
    """Multiply two exact complex rectangles independently."""
    ac = _interval_multiply(left[:2], right[:2])
    bd = _interval_multiply(left[2:], right[2:])
    ad = _interval_multiply(left[:2], right[2:])
    bc = _interval_multiply(left[2:], right[:2])
    return (
        ac[0] - bd[1],
        ac[1] - bd[0],
        ad[0] + bc[0],
        ad[1] + bc[1],
    )


def _rectangle_sum(values: list[ComplexRectangle]) -> ComplexRectangle:
    """Sum complex rectangles componentwise with exact Fractions."""
    columns = zip(*values, strict=True)
    return tuple(sum(column, Fraction(0)) for column in columns)  # type: ignore[return-value]


def _point_rectangle(value: complex) -> ComplexRectangle:
    """Return the singleton rectangle for one stored complex128 point."""
    real = Fraction.from_float(float(value.real))
    imag = Fraction.from_float(float(value.imag))
    return real, real, imag, imag


def _magnitude_upper(rectangle: ComplexRectangle) -> Fraction:
    """Bound a complex rectangle magnitude with an independent root."""
    real = max(abs(rectangle[0]), abs(rectangle[1]))
    imag = max(abs(rectangle[2]), abs(rectangle[3]))
    return sqrt_fraction_upper(real * real + imag * imag)


def _row_norms(
    rectangles: GalerkinLocalTerminalComplexRectangles,
    rows: np.ndarray,
    selected: np.ndarray,
    fiber_size: int,
) -> list[Fraction]:
    """Independently reduce exact coefficient rectangles by fiber."""
    squared = [Fraction(0) for _ in range(fiber_size)]
    for index in np.flatnonzero(selected):
        magnitude = _magnitude_upper(_stored_rectangle(rectangles, int(index)))
        squared[int(rows[index])] += magnitude * magnitude
    return [sqrt_fraction_upper(value) for value in squared]


def _point_rectangle_error(
    point: complex,
    rectangle: ComplexRectangle,
) -> float:
    """Independently bound a frozen point's maximum corner distance."""
    real = Fraction.from_float(float(point.real))
    imag = Fraction.from_float(float(point.imag))
    real_error = max(abs(real - rectangle[0]), abs(real - rectangle[1]))
    imag_error = max(abs(imag - rectangle[2]), abs(imag - rectangle[3]))
    squared = real_error * real_error + imag_error * imag_error
    return fraction_upper_float(sqrt_fraction_upper(squared))


def _complex_vector_norm(values: np.ndarray) -> float:
    """Independently reduce one stored complex vector to an l2 upper."""
    squared = Fraction(0)
    for value in values:
        point = complex(value)
        real = Fraction.from_float(float(point.real))
        imag = Fraction.from_float(float(point.imag))
        squared += real * real + imag * imag
    return fraction_upper_float(sqrt_fraction_upper(squared))


def _real_vector_norm(values: np.ndarray) -> float:
    """Independently reduce stored real upper bounds to an l2 upper."""
    squared = sum(
        (
            Fraction.from_float(float(value)) ** 2
            for value in np.asarray(values)
        ),
        Fraction(0),
    )
    return fraction_upper_float(sqrt_fraction_upper(squared))


def _assert_cut_oracle(
    certificate: GalerkinLocalVacuumTerminalCertificate,
) -> None:
    """Check literal nonsymmetrized exact cut arithmetic independently."""
    projection = certificate.projection_certificate
    cut = certificate.cut_balance
    rows = np.asarray(projection.state_to_fiber_rows)
    selected = np.asarray(projection.selected_state_mask)
    field = np.asarray(projection.stability_result.solve_result.field)
    gram_columns = tuple(
        np.asarray(value)
        for value in (
            projection.gram_real_lower_bounds,
            projection.gram_real_upper_bounds,
            projection.gram_imag_lower_bounds,
            projection.gram_imag_upper_bounds,
        )
    )
    free_lower = np.asarray(projection.exact_free_diagonal_lower_bounds)
    free_upper = np.asarray(projection.exact_free_diagonal_upper_bounds)
    terms: list[ComplexRectangle] = []
    pair_count = 0
    for left in np.flatnonzero(selected):
        conjugate = _point_rectangle(complex(field[left]))
        conjugate = (
            conjugate[0],
            conjugate[1],
            -conjugate[3],
            -conjugate[2],
        )
        for right in np.flatnonzero(selected & (rows == rows[left])):
            pair_count += 1
            gram: ComplexRectangle = tuple(
                Fraction.from_float(float(column[left, right]))
                for column in gram_columns
            )  # type: ignore[assignment]
            free: ComplexRectangle = (
                Fraction.from_float(float(free_lower[right])),
                Fraction.from_float(float(free_upper[right])),
                Fraction(0),
                Fraction(0),
            )
            term = _rectangle_multiply(conjugate, gram)
            term = _rectangle_multiply(term, free)
            term = _rectangle_multiply(
                term, _point_rectangle(complex(field[right]))
            )
            terms.append(term)
    total = _rectangle_sum(terms)
    exact_work = (-total[3], -total[2])
    assert int(cut.direct_work_count) == pair_count
    assert cut.direct_work_count_exact == str(pair_count)
    assert float(cut.negative_defect_work_lower_bound) == (
        fraction_lower_float(exact_work[0])
    )
    assert float(cut.negative_defect_work_upper_bound) == (
        fraction_upper_float(exact_work[1])
    )

    inner = certificate.inner_current_diagnostic
    outer = certificate.outer_current_diagnostic
    expected_current = (
        Fraction.from_float(float(outer.exact_reduced_current_lower_bound))
        - Fraction.from_float(float(inner.exact_reduced_current_upper_bound)),
        Fraction.from_float(float(outer.exact_reduced_current_upper_bound))
        - Fraction.from_float(float(inner.exact_reduced_current_lower_bound)),
    )
    assert float(cut.current_difference_lower_bound) == fraction_lower_float(
        expected_current[0]
    )
    assert float(cut.current_difference_upper_bound) == fraction_upper_float(
        expected_current[1]
    )


@functools.lru_cache(maxsize=2)
def _projection(
    scope: GalerkinLocalTerminalScope,
) -> GalerkinLocalProjectionDefectCertificate:
    """Reuse the projection wall's single prepared L5/L6 parent pair."""
    return projection_tests._certificate(scope)


@functools.lru_cache(maxsize=8)
def _terminal(
    scope: GalerkinLocalTerminalScope = _FULL,
    disposition: GalerkinLocalVacuumTerminalDisposition = _PLANE,
    *,
    structural: bool = False,
) -> GalerkinLocalVacuumTerminalCertificate:
    """Compose one L8 record behind the already prepared projection seam."""
    parent = (
        _structural_projection(scope) if structural else _projection(scope)
    )
    return terminal._certify_prepared_terminal(
        parent,
        disposition,
        _TERMINAL_PAIRS,
        _BRANCH_TERMS,
        _CUT_PAIRS,
        _ROOT_WORK,
        _ENTIRE_POLICIES,
        _INTERVAL_WORK,
        _RATIONAL_BITS,
    )


@functools.lru_cache(maxsize=1)
def _structural_parents() -> tuple[
    GalerkinLocalZeroSlabCertificate,
    GalerkinLocalStabilityResult,
]:
    """Build one genuine one-state on-shell L1--L6 parent pair."""
    zero_mode = jnp.asarray([[0, 0, 0]], dtype=jnp.int64)
    support = create_galerkin_product_support(
        state_indices=zero_mode,
        interaction_indices=zero_mode,
        absorber_indices=zero_mode,
        work_indices=zero_mode,
        work_shape=(1, 1, 1),
    )
    for indices in (
        support.state_indices,
        support.interaction_indices,
        support.absorber_indices,
        support.work_indices,
    ):
        assert_array_equal(indices, zero_mode)
    assert support.work_shape == (1, 1, 1)
    potential = create_local_cell_potential_3d(
        jnp.zeros((1, 1, 3), dtype=jnp.float64),
        cell_size=(1.0, 1.0, 1.0),
        box_size=(3.0, 1.0, 1.0),
        cell_center_origin=(0.1, 0.2, 0.3),
        reference_value=0.0,
        reference_semantics=(
            GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value
        ),
        producer="local-vacuum-terminal-native-fixture-v1",
        provenance_hash="6" * 64,
        producer_coefficient_normalization="producer metadata only",
        producer_bandwidth=1.0,
    )
    acquisition = checked_acquisition(support, potential.box_size)
    realization = realize_local_cell_galerkin_potential(potential, acquisition)
    realization = certify_local_cell_galerkin_potential(
        realization, maximum_direct_terms=3
    )
    compression = certify_local_cell_exact_compression(
        realization,
        accelerating_voltage_kv=200.0,
        maximum_state_pairs=1,
        maximum_interaction_modes=1,
        maximum_work_grid_points=1,
    )
    core = create_local_cell_interaction_core(compression)
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
    cap_certificate = certify_axial_cell_absorber(
        absorber,
        maximum_direct_terms=3,
        maximum_state_pairs=1,
    )
    cap_floor = certify_axial_cap_floor(
        cap_certificate,
        maximum_gram_degree=0,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    target = compose_local_cell_galerkin_target(
        cap_floor,
        target_name="local-vacuum-terminal-native-target",
    )
    ledger = target.fixed_linear_error_ledger
    for values in (
        target.free_diagonal,
        ledger.algebraic_free_diagonal,
        ledger.exact_free_diagonal_lower_bounds,
        ledger.exact_free_diagonal_upper_bounds,
        ledger.free_diagonal_error_bounds,
    ):
        assert_array_equal(values, [0.0])
    additional = certify_local_additional_source(
        realize_zero_local_additional_source(target),
        maximum_direct_terms=1,
    )
    assert additional.source.route is GalerkinLocalAdditionalSourceRoute.ZERO
    source = compose_local_represented_plane_source(
        target,
        additional,
        0,
        jnp.asarray(0.75 - 0.25j, dtype=jnp.complex128),
        jnp.asarray(1.25, dtype=jnp.float64),
        phase_convention=(
            GalerkinLocalSourcePhaseConvention.PHYSICAL_WAVEVECTOR
        ),
        source_plane_coordinate=jnp.asarray(0.125, dtype=jnp.float64),
        scan_position=jnp.asarray([0.0, 0.2, -0.1], dtype=jnp.float64),
        aberration_phase=jnp.asarray(0.3, dtype=jnp.float64),
        source_name="local-vacuum-terminal-native-plane",
    )
    represented = certify_local_represented_source(
        source, maximum_direct_pairs=_STRUCTURAL_STABILITY_PAIRS
    )
    assert source.kind is GalerkinLocalRepresentedSourceKind.PLANE_MODE
    assert int(np.count_nonzero(np.asarray(source.modes.active_mask))) == 1
    assert bool(represented.finite_certificate)
    assert represented.failure is GalerkinLocalRepresentedSourceFailure.NONE
    assert int(represented.direct_pair_count) == _STRUCTURAL_STABILITY_PAIRS
    assert int(represented.maximum_direct_pairs) == (
        _STRUCTURAL_STABILITY_PAIRS
    )
    assert stored_value_payload(represented.source.target) == (
        stored_value_payload(target)
    )
    assert represented.parent_source_evidence_digest == (
        represented.source.source_evidence_digest
    )
    zero_slab = certify_local_zero_slab(
        represented,
        slab_lower_coordinate=np.float64(0.75),
        slab_upper_coordinate=np.float64(1.25),
    )
    assert float(zero_slab.slab_lower_coordinate) == 0.75
    assert float(zero_slab.slab_upper_coordinate) == 1.25
    assert bool(zero_slab.terminal_zero_slab_eligible)
    assert (
        GalerkinLocalZeroSlabFailure(int(zero_slab.failure_mask))
        is GalerkinLocalZeroSlabFailure.NONE
    )
    assert stored_value_payload(zero_slab.represented_source_certificate) == (
        stored_value_payload(represented)
    )
    assert zero_slab.parent_represented_certificate_digest == (
        represented.certificate_digest
    )
    result = projection_tests._make_stability_result(
        zero_slab,
        maximum_direct_pairs=_STRUCTURAL_STABILITY_PAIRS,
    )
    assert bool(result.proof.state_radius_eligible)
    assert bool(result.proof.operational_state_eligible)
    assert result.proof.failure is GalerkinLocalStabilityFailure.NONE
    assert int(result.proof.direct_work_count) == _STRUCTURAL_STABILITY_PAIRS
    assert int(result.proof.maximum_direct_pairs) == (
        _STRUCTURAL_STABILITY_PAIRS
    )
    assert result.proof.disposition is (
        GalerkinLocalStabilityDisposition.OPERATIONAL_PASS
    )
    assert stored_value_payload(result.certificate) == stored_value_payload(
        represented
    )
    assert result.proof.certificate_digest == represented.certificate_digest
    assert result.proof.result_identity_digest == result.result_identity_digest
    return zero_slab, result


@functools.lru_cache(maxsize=2)
def _structural_projection(
    scope: GalerkinLocalTerminalScope,
) -> GalerkinLocalProjectionDefectCertificate:
    """Project the cached one-state exact-zero L1--L6 fixture."""
    zero_slab, result = _structural_parents()
    certificate = projection_module._certify_prepared(
        zero_slab,
        result,
        scope,
        _STATE_BUDGET,
        _STRUCTURAL_STABILITY_PAIRS,
        _STRUCTURAL_GRAM_PAIRS,
    )
    assert certificate.projection_scope is scope
    assert int(certificate.direct_pair_count) == _STRUCTURAL_GRAM_PAIRS
    assert int(certificate.maximum_gram_pairs) == _STRUCTURAL_GRAM_PAIRS
    assert int(certificate.maximum_stability_direct_pairs) == (
        _STRUCTURAL_STABILITY_PAIRS
    )
    assert_array_equal(certificate.selected_state_mask, [True])
    assert_array_equal(certificate.structural_exact_zero_state_mask, [True])
    assert_array_equal(certificate.structural_exact_zero_fiber_mask, [True])
    assert bool(certificate.finite_projection_bound_eligible)
    assert bool(certificate.operational_budget_eligible)
    assert bool(certificate.structural_exact_zero_eligible)
    assert (
        GalerkinLocalProjectionDefectFailure(int(certificate.failure_mask))
        is GalerkinLocalProjectionDefectFailure.NONE
    )
    assert stored_value_payload(certificate.zero_slab_certificate) == (
        stored_value_payload(zero_slab)
    )
    assert stored_value_payload(certificate.stability_result) == (
        stored_value_payload(result)
    )
    assert certificate.parent_zero_slab_certificate_digest == (
        zero_slab.certificate_digest
    )
    assert certificate.parent_stability_result_identity_digest == (
        result.result_identity_digest
    )
    assert certificate.parent_stability_result_evidence_digest == (
        result.result_evidence_digest
    )
    return certificate


@functools.lru_cache(maxsize=1)
def _negative_projection() -> GalerkinLocalProjectionDefectCertificate:
    """Flip only the prepared arithmetic fixture's terminal orientation."""
    parent = _projection(_FULL)
    zero_slab = parent.zero_slab_certificate
    represented = zero_slab.represented_source_certificate
    target = represented.source.target
    acquisition = replace(
        target.acquisition,
        terminal_side=GalerkinTerminalSide.NEGATIVE,
    )
    support_eligibility = replace(
        target.support_eligibility,
        manifest=acquisition,
    )
    realization = replace(
        target.realization,
        support_eligibility=support_eligibility,
    )
    compression = replace(target.compression, realization=realization)
    interaction_core = replace(
        target.interaction_core,
        compression=compression,
    )
    coefficient_certificate = target.cap_floor_proof.coefficient_certificate
    absorber = replace(
        coefficient_certificate.absorber,
        interaction_core=interaction_core,
    )
    coefficient_certificate = replace(
        coefficient_certificate,
        absorber=absorber,
    )
    cap_floor_proof = replace(
        target.cap_floor_proof,
        coefficient_certificate=coefficient_certificate,
    )
    replaced_target = replace(target, cap_floor_proof=cap_floor_proof)
    replaced_source = replace(represented.source, target=replaced_target)
    replaced_represented = replace(represented, source=replaced_source)
    replaced_zero = replace(
        zero_slab,
        represented_source_certificate=replaced_represented,
    )
    replaced_result = replace(
        parent.stability_result,
        certificate=replaced_represented,
    )
    return projection_module._certify_prepared(
        replaced_zero,
        replaced_result,
        _FULL,
        _STATE_BUDGET,
        _STABILITY_PAIRS,
        _GRAM_PAIRS,
    )


@functools.lru_cache(maxsize=1)
def _negative_terminal() -> GalerkinLocalVacuumTerminalCertificate:
    """Compose the negative-side arithmetic oracle without another parent."""
    return terminal._certify_prepared_terminal(
        _negative_projection(),
        _PLANE,
        _TERMINAL_PAIRS,
        _BRANCH_TERMS,
        _CUT_PAIRS,
        _ROOT_WORK,
        _ENTIRE_POLICIES,
        _INTERVAL_WORK,
        _RATIONAL_BITS,
    )


@functools.lru_cache(maxsize=1)
def _public_terminal() -> GalerkinLocalVacuumTerminalCertificate:
    """Cross the public projection replay boundary exactly once."""
    return certify_local_vacuum_terminal(
        _projection(_FULL),
        disposition=_PLANE,
        maximum_state_error=_STATE_BUDGET,
        maximum_stability_direct_pairs=_STABILITY_PAIRS,
        maximum_gram_pairs=_GRAM_PAIRS,
        maximum_terminal_direct_pairs=_TERMINAL_PAIRS,
        maximum_branch_direct_terms=_BRANCH_TERMS,
        maximum_cut_direct_pairs=_CUT_PAIRS,
        maximum_root_work=_ROOT_WORK,
        precision_bits=_PRECISION,
        maximum_terms=_ENTIRE_TERMS,
        maximum_entire_work=_ENTIRE_WORK,
        maximum_range_reductions=_RANGE_REDUCTIONS,
        maximum_interval_work=_INTERVAL_WORK,
    )


def _prepare(
    certificate: GalerkinLocalVacuumTerminalCertificate,
    *,
    disposition: GalerkinLocalVacuumTerminalDisposition = _PLANE,
    branch_terms: int = _BRANCH_TERMS,
) -> GalerkinLocalVacuumTerminalCertificate:
    """Replay one raw L8 record against independent caller policies."""
    return prepare_local_vacuum_terminal_certificate(
        certificate,
        disposition=disposition,
        maximum_state_error=_STATE_BUDGET,
        maximum_stability_direct_pairs=_STABILITY_PAIRS,
        maximum_gram_pairs=_GRAM_PAIRS,
        maximum_terminal_direct_pairs=_TERMINAL_PAIRS,
        maximum_branch_direct_terms=branch_terms,
        maximum_cut_direct_pairs=_CUT_PAIRS,
        maximum_root_work=_ROOT_WORK,
        precision_bits=_PRECISION,
        maximum_terms=_ENTIRE_TERMS,
        maximum_entire_work=_ENTIRE_WORK,
        maximum_range_reductions=_RANGE_REDUCTIONS,
        maximum_interval_work=_INTERVAL_WORK,
    )


def test_public_rational_bit_defaults_are_frozen() -> None:
    """Exercise the canonical public L8 retained-bit policy by omission."""
    for function in (
        certify_local_vacuum_terminal,
        prepare_local_vacuum_terminal_certificate,
    ):
        parameter = inspect.signature(function).parameters[
            "maximum_rational_bits"
        ]
        assert parameter.default == _RATIONAL_BITS


def test_exact_branch_maps_and_negative_side_entire_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the fixed hull, branch signs, negative-side J, and phi2."""
    hull = terminal._OutwardDyadicHullLedger(4096)
    thirds = hull.interval((Fraction(-1, 3), Fraction(1, 3)))
    assert thirds is not None
    assert thirds[0] <= Fraction(-1, 3) <= Fraction(1, 3) <= thirds[1]
    assert hull.normal_floor_count == 0

    minimum_normal = Fraction.from_float(float(np.finfo(np.float64).tiny))
    half_minimum_normal = minimum_normal / 2
    positive_tiny = hull.interval((half_minimum_normal, half_minimum_normal))
    negative_tiny = hull.interval((-half_minimum_normal, -half_minimum_normal))
    assert positive_tiny is not None
    assert negative_tiny is not None
    assert positive_tiny == (Fraction(0), minimum_normal)
    assert negative_tiny == (-minimum_normal, Fraction(0))
    assert hull.attempted_endpoint_count == 6
    assert hull.completed_endpoint_count == 6
    assert hull.normal_floor_count == 4
    assert hull.output_peak_bits <= 1024
    for endpoint in (*thirds, *positive_tiny, *negative_tiny):
        denominator = endpoint.denominator
        assert denominator & (denominator - 1) == 0
    assert hull.evidence_digest() == sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.hull.v1",
            "algorithm": "outward_binary64_normal_hull_v1",
            "maximum_rational_bits": 4096,
            "attempted_endpoints": 6,
            "completed_endpoints": 6,
            "input_peak_bits": hull.input_peak_bits,
            "output_peak_bits": hull.output_peak_bits,
            "normal_floor_count": 4,
            "range_failure": False,
        }
    )

    limited_hull = terminal._OutwardDyadicHullLedger(2)
    with pytest.raises(EntireEnclosureError) as captured:
        limited_hull.interval((Fraction(-1, 2), Fraction(1, 5)))
    assert captured.value.failure is EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    assert captured.value.exact_work_count == (
        limited_hull.completed_endpoint_count
    )
    assert limited_hull.attempted_endpoint_count == 2
    assert limited_hull.completed_endpoint_count == 1
    assert limited_hull.input_peak_bits == 3
    assert not limited_hull.range_failure

    huge_hull = terminal._OutwardDyadicHullLedger(16_384)
    huge = Fraction(1 << 4096)
    assert huge_hull.interval((huge, huge)) is None
    assert huge_hull.range_failure
    assert huge_hull.attempted_endpoint_count == 2
    assert huge_hull.completed_endpoint_count == 1

    subnormal = float.fromhex("0x0.0000000000001p-1022")
    subnormal_hull = terminal._OutwardDyadicHullLedger(4096)
    with monkeypatch.context() as context:
        context.setattr(
            terminal, "fraction_lower_float", lambda value: subnormal
        )
        context.setattr(
            terminal, "fraction_upper_float", lambda value: subnormal
        )
        assert subnormal_hull.interval((Fraction(1), Fraction(1))) is None
    assert subnormal_hull.range_failure
    assert subnormal_hull.attempted_endpoint_count == 2
    assert subnormal_hull.completed_endpoint_count == 0

    tiny_root = classify_local_vacuum_root(
        (half_minimum_normal**2, half_minimum_normal**2),
        maximum_rational_bits=8192,
    )
    raw_tiny_interval = tiny_root.root_interval
    assert raw_tiny_interval is not None
    tiny_root_hull = terminal._OutwardDyadicHullLedger(8192)
    assert terminal._hull_branch_root_intervals(
        (tiny_root,), tiny_root_hull
    ) == (None,)
    assert tiny_root.classification is GalerkinLocalVacuumRootClass.PROPAGATING
    assert tiny_root.root_interval == raw_tiny_interval
    assert tiny_root_hull.range_failure

    propagating = classify_local_vacuum_root((Fraction(4), Fraction(4)))
    evanescent = classify_local_vacuum_root((Fraction(-4), Fraction(-4)))
    witness = make_local_vacuum_zero_witness(
        (("1", Fraction(1)),),
        (("1", Fraction(1)),),
        route=GalerkinLocalVacuumZeroWitnessRoute.EXACT_RATIONAL_DIFFERENCE,
    )
    grazing = classify_local_vacuum_root(
        (Fraction(0), Fraction(0)), zero_witness=witness
    )
    point_one: ComplexRectangle = (
        Fraction(1),
        Fraction(1),
        Fraction(0),
        Fraction(0),
    )
    two_i: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(2),
        Fraction(2),
    )
    minus_two: ComplexRectangle = (
        Fraction(-2),
        Fraction(-2),
        Fraction(0),
        Fraction(0),
    )
    exact_zero: ComplexRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    propagating_interval = propagating.root_interval
    evanescent_interval = evanescent.root_interval
    assert propagating_interval is not None
    assert evanescent_interval is not None
    assert terminal._branch_transform(
        point_one,
        two_i,
        propagating,
        (propagating_interval.lower, propagating_interval.upper),
        terminal._DirectRationalLedger(1024),
    ) == (point_one, exact_zero)
    assert terminal._branch_transform(
        point_one,
        minus_two,
        evanescent,
        (evanescent_interval.lower, evanescent_interval.upper),
        terminal._DirectRationalLedger(1024),
    ) == (point_one, exact_zero)
    assert terminal._branch_transform(
        point_one,
        two_i,
        grazing,
        None,
        terminal._DirectRationalLedger(1024),
    ) == (point_one, two_i)

    helper_hull = terminal._OutwardDyadicHullLedger(8192)
    recorder = terminal._EntireRecorder(
        (192, 4096, 100_000, 4096, 8192), helper_hull
    )
    kernel = terminal._j_kernel(
        two_i,
        (Fraction(-3), Fraction(-3)),
        Fraction(1, 2),
        "negative_side.propagating_plus",
        recorder,
        terminal._DirectRationalLedger(8192),
    )
    assert kernel is not None
    with localcontext() as context:
        context.prec = 140
        exp_imag, exp_real = projection_tests._decimal_sin_cos(Decimal(1))
        phi_imag, phi_real = projection_tests._decimal_sin_cos(Decimal("-2.5"))
        numerator = (phi_real - Decimal(1), phi_imag)
        denominator = Decimal("6.25")
        phi1 = (
            numerator[1] * Decimal("-2.5") / denominator,
            -numerator[0] * Decimal("-2.5") / denominator,
        )
        oracle = (
            Decimal("0.5") * (exp_real * phi1[0] - exp_imag * phi1[1]),
            Decimal("0.5") * (exp_real * phi1[1] + exp_imag * phi1[0]),
        )
    assert _contains(kernel, oracle)
    phi2 = recorder.call(
        "negative_side.grazing_field.phi2",
        "phi2",
        exact_zero,
    )
    assert phi2 == (
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(0),
        Fraction(0),
    )
    assert bool(recorder.evidence().helper_eligible)

    irrational_root = classify_local_vacuum_root(
        (Fraction(2), Fraction(2)), maximum_rational_bits=_RATIONAL_BITS
    )
    raw_root_payload = stored_value_payload(irrational_root)
    raw_propagator = enclose_local_vacuum_propagator(
        irrational_root,
        Fraction(1, 2),
        maximum_rational_bits=_RATIONAL_BITS,
    )
    raw_propagator_payload = stored_value_payload(raw_propagator)
    consumption_hull = terminal._OutwardDyadicHullLedger(_RATIONAL_BITS)
    root_copies = terminal._hull_branch_root_intervals(
        (irrational_root,), consumption_hull
    )
    propagator_copies = terminal._hull_branch_propagator_entries(
        (raw_propagator,), consumption_hull
    )
    assert stored_value_payload(irrational_root) == raw_root_payload
    assert stored_value_payload(raw_propagator) == raw_propagator_payload
    assert root_copies[0] is not None
    assert propagator_copies[0] is not None
    assert irrational_root.root_interval is not None
    copied_root = root_copies[0]
    assert copied_root is not None
    assert (
        copied_root[0]
        <= irrational_root.root_interval.lower
        <= irrational_root.root_interval.upper
        <= copied_root[1]
    )
    for copied, raw in zip(
        propagator_copies[0], raw_propagator.entries, strict=True
    ):
        assert copied[0] <= raw.lower <= raw.upper <= copied[1]
    assert consumption_hull.input_peak_bits > consumption_hull.output_peak_bits
    assert consumption_hull.output_peak_bits <= 1024
    assert not consumption_hull.range_failure


def test_root_realization_is_canonical_and_overflow_is_typed() -> None:
    """Check nearest-float root evidence and typed conversion overflow."""
    irrational = classify_local_vacuum_root((Fraction(2), Fraction(2)))
    realization, error = terminal._root_realization(
        irrational, terminal._DirectRationalLedger(4096)
    )
    assert irrational.root_interval is not None
    exact = Fraction.from_float(realization)
    assert error == fraction_upper_float(
        max(
            abs(exact - irrational.root_interval.lower),
            abs(exact - irrational.root_interval.upper),
        )
    )
    huge = Fraction(1 << 4096)
    huge_root = classify_local_vacuum_root(
        (huge, huge), maximum_rational_bits=16_384
    )
    with pytest.raises(
        terminal._LocalArithmeticRangeError,
        match="finite float64",
    ):
        terminal._root_realization(
            huge_root, terminal._DirectRationalLedger(16_384)
        )


def test_availability_intersections_and_three_way_half_space_status() -> None:
    """Distinguish unavailable routes, empty intersections, and zero status."""
    lower = jnp.asarray([0.0, 2.0, -1.0], dtype=jnp.float64)
    upper = jnp.asarray([0.0, 3.0, 1.0], dtype=jnp.float64)
    zero_imag = jnp.zeros((3,), dtype=jnp.float64)
    rectangles = terminal.GalerkinLocalTerminalComplexRectangles(
        lower, upper, zero_imag, zero_imag
    )
    pair = (rectangles, rectangles)
    available = np.ones((3, 2), dtype=np.bool_)
    unavailable = available.copy()
    unavailable[2, 1] = False
    _, mask = terminal._intersect_rectangle_pairs(
        pair, pair, available, unavailable
    )
    assert_array_equal(mask, unavailable)

    roots = (
        classify_local_vacuum_root((Fraction(4), Fraction(4))),
        classify_local_vacuum_root((Fraction(-4), Fraction(-4))),
        classify_local_vacuum_root((Fraction(4), Fraction(4))),
    )
    propagators = (
        enclose_local_vacuum_propagator(roots[0], Fraction(1, 2)),
        enclose_local_vacuum_propagator(roots[1], Fraction(1, 2)),
        None,
    )
    dispositions = terminal._half_space_dispositions(
        roots, propagators, rectangles
    )
    assert dispositions == (
        GalerkinLocalVacuumHalfSpaceDisposition.PROPAGATING_INWARD_EXACT_ZERO,
        GalerkinLocalVacuumHalfSpaceDisposition.EVANESCENT_GROWING_PROVABLY_NONZERO,
        GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED,
    )


def test_full_selected_native_composition_binds_routes_and_work() -> None:
    """Exercise both scopes, native claims, and the full dependency DAG."""
    certificate = _terminal()
    branch = certificate.branch_evidence
    cut = certificate.cut_balance
    assert certificate.terminal_scope is _FULL
    assert certificate.terminal_side is GalerkinTerminalSide.POSITIVE
    assert certificate.disposition is _PLANE
    assert float(certificate.defining_plane_coordinate) == 1.25
    assert float(certificate.comparison_plane_coordinate) == 0.75
    assert bool(certificate.current_diagnostic_eligible)
    assert bool(certificate.current_operator_eligible)
    assert bool(certificate.current_action_eligible)
    assert bool(branch.branch_evidence_eligible)
    assert branch.maximum_rational_bits == 262_144
    assert branch.direct_rational_peak_bits <= branch.maximum_rational_bits
    assert branch.hull_algorithm == "outward_binary64_normal_hull_v1"
    assert branch.hull_attempted_endpoint_count > 0
    assert (
        branch.hull_completed_endpoint_count
        == branch.hull_attempted_endpoint_count
    )
    assert branch.hull_input_peak_bits <= branch.maximum_rational_bits
    assert branch.hull_input_peak_bits > branch.hull_output_peak_bits
    assert branch.hull_output_peak_bits <= 1024
    assert not branch.hull_range_failure
    assert branch.hull_evidence_digest == sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.hull.v1",
            "algorithm": branch.hull_algorithm,
            "maximum_rational_bits": branch.maximum_rational_bits,
            "attempted_endpoints": branch.hull_attempted_endpoint_count,
            "completed_endpoints": branch.hull_completed_endpoint_count,
            "input_peak_bits": branch.hull_input_peak_bits,
            "output_peak_bits": branch.hull_output_peak_bits,
            "normal_floor_count": branch.hull_normal_floor_count,
            "range_failure": branch.hull_range_failure,
        }
    )
    assert bool(np.all(branch.cauchy_crosscheck_mask))
    assert bool(np.all(branch.branch_crosscheck_mask))
    assert bool(cut.cut_balance_eligible)
    assert bool(certificate.vacuum_branch_eligible)
    assert (
        GalerkinLocalVacuumTerminalFailure(int(certificate.failure_mask))
        is GalerkinLocalVacuumTerminalFailure.NONE
    )
    assert branch.direct_work_count_exact == str(
        6
        * int(
            np.count_nonzero(
                certificate.projection_certificate.selected_state_mask
            )
        )
        + 32 * len(branch.root_certificates)
    )
    assert cut.direct_work_count_exact == "9"
    assert len(branch.root_certificates) == len(branch.propagators)
    for root, propagator in zip(
        branch.root_certificates, branch.propagators, strict=True
    ):
        assert root is not None
        assert (
            root.classification
            is not GalerkinLocalVacuumRootClass.UNCLASSIFIED
        )
        assert propagator is not None
        assert (
            root.root_evidence_digest
            == propagator.root_certificate.root_evidence_digest
        )
        if root.classification is GalerkinLocalVacuumRootClass.GRAZING:
            assert root.zero_witness is not None
        else:
            assert root.zero_witness is None
    assert len(branch.entire_evidence.kernel_labels) == len(
        branch.entire_evidence.transcripts
    )
    assert len(branch.entire_evidence.kernel_labels) == len(
        branch.entire_evidence.failure_reasons
    )
    assert (
        branch.physical_root_identity_digest != branch.branch_evidence_digest
    )
    assert certificate.parent_projection_certificate_digest == (
        certificate.projection_certificate.certificate_digest
    )
    assert "excludes positive-port" in certificate.completion_scope
    selected = _terminal(_SELECTED, _PLANE)
    assert selected.terminal_scope is _SELECTED
    assert (
        selected.terminal_identity_digest
        != _terminal().terminal_identity_digest
    )

    mismatched = _terminal(_FULL, _NATIVE_SELECTED)
    mismatch_failure = GalerkinLocalVacuumTerminalFailure(
        int(mismatched.failure_mask)
    )
    assert mismatch_failure & (
        GalerkinLocalVacuumTerminalFailure.DISPOSITION_SCOPE_MISMATCH
    )
    assert mismatch_failure & (
        GalerkinLocalVacuumTerminalFailure.NATIVE_STRUCTURAL_ZERO_UNAVAILABLE
    )
    assert not bool(mismatched.vacuum_branch_eligible)

    native_full = _terminal(_FULL, _NATIVE_FULL, structural=True)
    native_selected = _terminal(_SELECTED, _NATIVE_SELECTED, structural=True)
    assert native_full.projection_certificate.projection_scope is _FULL
    assert native_selected.projection_certificate.projection_scope is _SELECTED
    assert (
        native_full.projection_certificate.projection_identity_digest
        != native_selected.projection_certificate.projection_identity_digest
    )
    assert bool(
        native_full.projection_certificate.structural_exact_zero_eligible
    )
    assert bool(
        native_selected.projection_certificate.structural_exact_zero_eligible
    )
    for value in (native_full, native_selected):
        failure = GalerkinLocalVacuumTerminalFailure(int(value.failure_mask))
        failure_enum = GalerkinLocalVacuumTerminalFailure
        structural_failure = failure_enum.NATIVE_STRUCTURAL_ZERO_UNAVAILABLE
        native_failures = (
            GalerkinLocalVacuumTerminalFailure.DISPOSITION_SCOPE_MISMATCH
            | structural_failure
        )
        assert not failure & native_failures
        assert bool(value.vacuum_branch_eligible)


def test_lvt56_production_point_error_and_norm_dag_are_exact_once() -> None:
    """Verify the role-zero point and exact-once amplitude reductions."""
    certificate = _terminal()
    branch = certificate.branch_evidence
    outer = certificate.outer_current_diagnostic
    operator = outer.action_enclosure.certificate
    points = np.asarray(branch.frozen_defining_branch_points)
    production = np.asarray(
        branch.production_to_submitted_amplitude_error_bounds
    )
    state = np.asarray(branch.state_radius_amplitude_error_bounds)
    total = np.asarray(branch.exact_state_total_amplitude_error_bounds)
    point_norms = np.asarray(branch.production_amplitude_norm_upper_bounds)
    total_norms = np.asarray(branch.exact_state_amplitude_norm_upper_bounds)
    roots = np.asarray(branch.frozen_positive_root_realizations)
    root_errors = np.asarray(branch.frozen_positive_root_error_bounds)
    phases = np.asarray(branch.physical_phase_realizations)
    trace_points = phases[:, 1] * np.asarray(outer.trace_coefficients)
    normal_points = phases[:, 1] * np.asarray(
        outer.normal_derivative_coefficients
    )
    rows = np.asarray(operator.state_to_fiber_rows)
    selected = np.asarray(operator.selected_state_mask)
    trace_norms = _row_norms(
        operator.exact_trace_coefficient_rectangles,
        rows,
        selected,
        len(branch.root_certificates),
    )
    normal_norms = _row_norms(
        operator.exact_normal_coefficient_rectangles,
        rows,
        selected,
        len(branch.root_certificates),
    )
    state_radius = Fraction.from_float(
        float(certificate.projection_certificate.state_radius_upper_bound)
    )
    for fiber, root in enumerate(branch.root_certificates):
        assert root is not None and root.root_interval is not None
        if root.classification is GalerkinLocalVacuumRootClass.GRAZING:
            assert roots[fiber] == 0.0 and root_errors[fiber] == 0.0
            expected_points = (
                complex(trace_points[fiber]),
                complex(normal_points[fiber]),
            )
            state_fractions = (
                trace_norms[fiber] * state_radius,
                normal_norms[fiber] * state_radius,
            )
        else:
            midpoint = (
                root.root_interval.lower + root.root_interval.upper
            ) / 2
            assert roots[fiber] == float(midpoint)
            exact = Fraction.from_float(float(roots[fiber]))
            assert root_errors[fiber] == fraction_upper_float(
                max(
                    abs(exact - root.root_interval.lower),
                    abs(exact - root.root_interval.upper),
                )
            )
            if root.classification is GalerkinLocalVacuumRootClass.PROPAGATING:
                quotient = normal_points[fiber] / (1.0j * roots[fiber])
                expected_points = (
                    0.5 * (trace_points[fiber] + quotient),
                    0.5 * (trace_points[fiber] - quotient),
                )
            else:
                quotient = normal_points[fiber] / roots[fiber]
                expected_points = (
                    0.5 * (trace_points[fiber] - quotient),
                    0.5 * (trace_points[fiber] + quotient),
                )
            hulled_root_lower = Fraction.from_float(
                fraction_lower_float(root.root_interval.lower)
            )
            assert 0 < hulled_root_lower <= root.root_interval.lower
            map_norm = Fraction(1, 2) * (
                trace_norms[fiber] + normal_norms[fiber] / hulled_root_lower
            )
            shared = map_norm * state_radius
            state_fractions = (shared, shared)
        for role in range(2):
            assert points[fiber, role] == expected_points[role]
            rectangle = _stored_rectangle(
                branch.defining_branch_rectangles[role], fiber
            )
            expected_production = _point_rectangle_error(
                complex(points[fiber, role]), rectangle
            )
            assert production[fiber, role] == expected_production
            assert state[fiber, role] == fraction_upper_float(
                state_fractions[role]
            )
            expected_total = fraction_upper_float(
                Fraction.from_float(float(production[fiber, role]))
                + Fraction.from_float(float(state[fiber, role]))
            )
            assert total[fiber, role] == expected_total
            point = complex(points[fiber, role])
            point_real = Fraction.from_float(float(point.real))
            point_imag = Fraction.from_float(float(point.imag))
            expected_point_norm = fraction_upper_float(
                sqrt_fraction_upper(
                    point_real * point_real + point_imag * point_imag
                )
            )
            assert point_norms[fiber, role] == expected_point_norm
            expected_norm = fraction_upper_float(
                Fraction.from_float(expected_point_norm)
                + Fraction.from_float(float(total[fiber, role]))
            )
            assert total_norms[fiber, role] == expected_norm
    production_l2 = _complex_vector_norm(points[:, 0])
    error_l2 = _real_vector_norm(total[:, 0])
    assert float(branch.production_prediction_l2_norm_upper_bound) == (
        production_l2
    )
    assert float(branch.exact_state_prediction_error_l2_upper_bound) == (
        error_l2
    )
    assert float(branch.exact_state_prediction_l2_norm_upper_bound) == (
        fraction_upper_float(
            Fraction.from_float(production_l2) + Fraction.from_float(error_l2)
        )
    )
    submitted = np.asarray(branch.submitted_state_branch_mismatch_upper_bounds)
    transfer = np.asarray(
        branch.projection_state_transfer_branch_mismatch_upper_bounds
    )
    plane_total = np.asarray(
        branch.projection_total_branch_mismatch_upper_bounds
    )
    for index in np.ndindex(plane_total.shape):
        assert plane_total[index] == fraction_upper_float(
            Fraction.from_float(float(submitted[index]))
            + Fraction.from_float(float(transfer[index]))
        )
    assert branch.prediction_branch_role == 0
    assert (
        "exact terminal amplitude map" in branch.state_radius_amplitude_scope
    )
    assert "exactly once" in branch.exact_state_amplitude_scope
    assert (
        "neither root error nor dyadic hull widening is added separately "
        "to E_a" in branch.root_realization_scope
    )


def test_negative_side_endpoint_forced_and_cut_signs_are_oriented_once() -> (
    None
):
    """Check the negative-side seam against independent sign oracles."""
    positive = _terminal()
    negative = _negative_terminal()
    _assert_cut_oracle(positive)
    _assert_cut_oracle(negative)
    assert negative.terminal_side is GalerkinTerminalSide.NEGATIVE
    assert float(negative.defining_plane_coordinate) == 0.75
    assert float(negative.comparison_plane_coordinate) == 1.25

    positive_zero = positive.projection_certificate.zero_slab_certificate
    negative_zero = negative.projection_certificate.zero_slab_certificate
    positive_target = (
        positive_zero.represented_source_certificate.source.target
    )
    negative_target = (
        negative_zero.represented_source_certificate.source.target
    )
    positive_normals = terminal._oriented_normal_intervals(
        positive_target, terminal._DirectRationalLedger(_RATIONAL_BITS)
    )
    negative_normals = terminal._oriented_normal_intervals(
        negative_target, terminal._DirectRationalLedger(_RATIONAL_BITS)
    )
    for positive_interval, negative_interval in zip(
        positive_normals, negative_normals, strict=True
    ):
        assert negative_interval == (
            -positive_interval[1],
            -positive_interval[0],
        )
    assert_allclose(
        negative.cut_balance.current_difference_lower_bound,
        positive.cut_balance.current_difference_lower_bound,
    )
    assert_allclose(
        negative.cut_balance.current_difference_upper_bound,
        positive.cut_balance.current_difference_upper_bound,
    )
    assert_allclose(
        negative.cut_balance.negative_defect_work_lower_bound,
        positive.cut_balance.negative_defect_work_lower_bound,
    )
    negative_branch = negative.branch_evidence
    assert bool(np.all(negative_branch.cauchy_crosscheck_mask))
    assert bool(np.all(negative_branch.branch_crosscheck_mask))
    for pair in (
        negative_branch.endpoint_cauchy_mismatch_rectangles,
        negative_branch.forced_cauchy_mismatch_rectangles,
        negative_branch.endpoint_branch_mismatch_rectangles,
        negative_branch.forced_branch_mismatch_rectangles,
    ):
        assert all(
            np.all(np.isfinite(np.asarray(column)))
            for role in pair
            for column in role
        )
    endpoint_formula = negative.branch_evidence.endpoint_mismatch_formula
    assert "m=y_outer-E_h(ell)y_inner" in endpoint_formula
    assert "r=s*(xi-xi_inner)" in endpoint_formula
    assert "once-hulled prepared propagator entries" in endpoint_formula
    assert "physical inner phase" in (
        negative.branch_evidence.forced_mismatch_formula
    )
    current_formula = negative.cut_balance.current_difference_formula
    assert (
        "outer side-oriented current minus inner side-oriented current"
        in current_formula
    )
    assert (
        "positive-coordinate upper-cut minus lower-cut current on both sides"
        in current_formula
    )


def test_branch_and_cut_preflight_skip_prohibited_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove preflight failures issue no helper or Gram work."""
    canonical = _terminal()
    parent = canonical.projection_certificate

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("preflight issued prohibited work")

    with monkeypatch.context() as context:
        context.setattr(terminal, "_physical_q_intervals", forbidden)
        branch = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            1,
            _ROOT_WORK,
            _ENTIRE_POLICIES,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(branch.failure_mask))
    assert (
        failure
        & GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
    )
    assert not bool(branch.entire_evidence.helper_attempted)
    assert branch.entire_evidence.kernel_labels == ()

    with monkeypatch.context() as context:
        context.setattr(terminal, "_point_rectangle", forbidden)
        cut = terminal._cut_balance(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            1,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(cut.failure_mask))
    assert (
        failure
        & GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
    )
    assert np.isfinite(float(cut.current_difference_lower_bound))
    assert np.isneginf(float(cut.negative_defect_work_lower_bound))
    assert np.isposinf(float(cut.certified_balance_upper_bound))

    with monkeypatch.context() as context:
        context.setattr(terminal, "_MAXIMUM_SIGNED_INT64", 8)
        context.setattr(terminal_types, "_MAXIMUM_SIGNED_INT64", 8)
        context.setattr(terminal, "_physical_q_intervals", forbidden)
        context.setattr(terminal, "_point_rectangle", forbidden)
        overflow = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            8,
            1,
            (1, 1, 1, 0, 2),
            1,
            2,
        )
        cut_overflow = terminal._cut_balance(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            8,
            2,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(overflow.failure_mask))
    assert (
        failure & GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
    )
    assert int(overflow.direct_work_count) == 0
    cut_failure = GalerkinLocalVacuumTerminalFailure(
        int(cut_overflow.failure_mask)
    )
    assert cut_failure & (
        GalerkinLocalVacuumTerminalFailure.DIRECT_WORK_COUNT_OVERFLOW
    )
    assert int(cut_overflow.direct_work_count) == 0


def test_helper_rational_and_range_failures_are_typed_and_nonempty_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep zero-work, direct-bit, and float overflow routes distinct."""
    canonical = _terminal()
    parent = canonical.projection_certificate

    def zero_work_failure(*args: object, **kwargs: object) -> object:
        raise EntireEnclosureError(
            EntireEnclosureFailure.TERM_BUDGET_EXCEEDED,
            0,
            "forced zero-work helper failure",
        )

    with monkeypatch.context() as context:
        context.setattr(terminal, "enclose_complex_exp", zero_work_failure)
        failed = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            _BRANCH_TERMS,
            _ROOT_WORK,
            _ENTIRE_POLICIES,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(failed.failure_mask))
    assert (
        failure
        & GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
    )
    assert bool(failed.entire_evidence.helper_attempted)
    failed_calls = [
        (reason, work)
        for reason, work in zip(
            failed.entire_evidence.failure_reasons,
            failed.entire_evidence.failure_work_counts,
            strict=True,
        )
        if reason is not None
    ]
    assert failed_calls
    assert all(work == 0 for _, work in failed_calls)

    with monkeypatch.context() as context:
        context.setattr(
            terminal, "enclose_local_vacuum_propagator", zero_work_failure
        )
        propagator_failed = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            _BRANCH_TERMS,
            _ROOT_WORK,
            _ENTIRE_POLICIES,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(
        int(propagator_failed.failure_mask)
    )
    assert failure & GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.CAUCHY_CROSSCHECK_EMPTY
    )
    assert all(
        root is not None for root in propagator_failed.root_certificates
    )
    assert all(value is None for value in propagator_failed.propagators)
    assert all(
        reason is EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
        for reason in propagator_failed.propagator_failure_reasons
    )
    assert propagator_failed.propagator_failure_work_counts == (
        (0,) * len(propagator_failed.root_certificates)
    )
    assert all(
        disposition
        is GalerkinLocalVacuumHalfSpaceDisposition.ROOT_UNCLASSIFIED
        for disposition in propagator_failed.half_space_dispositions
    )

    with monkeypatch.context() as context:
        context.setattr(
            terminal, "classify_local_vacuum_root", zero_work_failure
        )
        root_failed = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            _BRANCH_TERMS,
            _ROOT_WORK,
            _ENTIRE_POLICIES,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(root_failed.failure_mask))
    assert failure & GalerkinLocalVacuumTerminalFailure.ROOT_PROPAGATOR_FAILURE
    assert all(root is None for root in root_failed.root_certificates)
    assert all(
        reason is EntireEnclosureFailure.TERM_BUDGET_EXCEEDED
        for reason in root_failed.root_failure_reasons
    )
    assert root_failed.root_failure_work_counts == (
        (0,) * len(root_failed.root_certificates)
    )

    bit_failed = terminal._branch_evidence(
        parent,
        canonical.inner_current_diagnostic,
        canonical.outer_current_diagnostic,
        np.float64(canonical.comparison_plane_coordinate),
        np.float64(canonical.defining_plane_coordinate),
        _BRANCH_TERMS,
        _ROOT_WORK,
        (_PRECISION, _ENTIRE_TERMS, _ENTIRE_WORK, _RANGE_REDUCTIONS, 2),
        _INTERVAL_WORK,
        2,
    )
    failure = GalerkinLocalVacuumTerminalFailure(int(bit_failed.failure_mask))
    assert (
        failure
        & GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    assert (
        bit_failed.direct_rational_failure
        is EntireEnclosureFailure.RATIONAL_SIZE_LIMIT
    )
    assert bit_failed.direct_rational_work_count_exact == "0"

    subnormal = float.fromhex("0x0.0000000000001p-1022")
    with monkeypatch.context() as context:
        context.setattr(
            terminal, "fraction_lower_float", lambda value: subnormal
        )
        context.setattr(
            terminal, "fraction_upper_float", lambda value: subnormal
        )
        helper_range_failed = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            _BRANCH_TERMS,
            _ROOT_WORK,
            _ENTIRE_POLICIES,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(
        int(helper_range_failed.failure_mask)
    )
    assert (
        failure & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert not failure & (
        GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    assert bool(helper_range_failed.entire_evidence.helper_eligible)
    assert helper_range_failed.hull_range_failure
    assert (
        helper_range_failed.hull_completed_endpoint_count
        < helper_range_failed.hull_attempted_endpoint_count
    )
    assert np.all(
        np.isinf(helper_range_failed.exact_state_total_amplitude_error_bounds)
    )

    with monkeypatch.context() as context:
        context.setattr(
            terminal,
            "_float_sum_upper",
            lambda *args: (_ for _ in ()).throw(
                terminal._LocalArithmeticRangeError("forced outward overflow")
            ),
        )
        range_failed = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            _BRANCH_TERMS,
            _ROOT_WORK,
            _ENTIRE_POLICIES,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(
        int(range_failed.failure_mask)
    )
    assert (
        failure & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert all(root is not None for root in range_failed.root_certificates)
    assert np.all(
        np.isinf(range_failed.exact_state_total_amplitude_error_bounds)
    )
    with pytest.raises(terminal._LocalArithmeticRangeError):
        terminal._float_sum_upper(
            float(np.finfo(np.float64).max),
            float(np.finfo(np.float64).max),
            terminal._DirectRationalLedger(_RATIONAL_BITS),
        )

    zero_slab, _ = projection_tests._parents()
    unavailable_result = projection_tests._make_stability_result(
        zero_slab,
        maximum_direct_pairs=1,
    )
    unavailable_projection = projection_module._certify_prepared(
        zero_slab,
        unavailable_result,
        _FULL,
        _STATE_BUDGET,
        1,
        _GRAM_PAIRS,
    )
    projection_failed = terminal._certify_prepared_terminal(
        unavailable_projection,
        _PLANE,
        _TERMINAL_PAIRS,
        _BRANCH_TERMS,
        _CUT_PAIRS,
        _ROOT_WORK,
        _ENTIRE_POLICIES,
        _INTERVAL_WORK,
        _RATIONAL_BITS,
    )
    failure = GalerkinLocalVacuumTerminalFailure(
        int(projection_failed.failure_mask)
    )
    assert (
        failure & GalerkinLocalVacuumTerminalFailure.PROJECTION_NONCERTIFICATE
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.BRANCH_CROSSCHECK_EMPTY
    )
    projection_branch = projection_failed.branch_evidence
    assert np.all(
        np.isinf(projection_branch.exact_state_total_amplitude_error_bounds)
    )

    current_failed = terminal._certify_prepared_terminal(
        parent,
        _PLANE,
        1,
        _BRANCH_TERMS,
        _CUT_PAIRS,
        _ROOT_WORK,
        _ENTIRE_POLICIES,
        _INTERVAL_WORK,
        _RATIONAL_BITS,
    )
    failure = GalerkinLocalVacuumTerminalFailure(
        int(current_failed.failure_mask)
    )
    assert failure & (
        GalerkinLocalVacuumTerminalFailure.CURRENT_DIAGNOSTIC_NONCERTIFICATE
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
    )
    assert np.isneginf(
        float(current_failed.cut_balance.certified_balance_lower_bound)
    )
    assert np.isposinf(
        float(current_failed.cut_balance.certified_balance_upper_bound)
    )

    with monkeypatch.context() as context:
        context.setattr(
            terminal, "fraction_lower_float", lambda value: -np.inf
        )
        context.setattr(terminal, "fraction_upper_float", lambda value: np.inf)
        cut_range = terminal._cut_balance(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            _CUT_PAIRS,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(cut_range.failure_mask))
    assert (
        failure & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert (
        not failure
        & GalerkinLocalVacuumTerminalFailure.CUT_BALANCE_CROSSCHECK_EMPTY
    )
    for lower, upper in (
        (
            cut_range.current_difference_lower_bound,
            cut_range.current_difference_upper_bound,
        ),
        (
            cut_range.negative_defect_work_lower_bound,
            cut_range.negative_defect_work_upper_bound,
        ),
        (
            cut_range.certified_balance_lower_bound,
            cut_range.certified_balance_upper_bound,
        ),
    ):
        assert np.isneginf(float(lower))
        assert np.isposinf(float(upper))

    huge = Fraction(1 << 4096)
    huge_root = classify_local_vacuum_root(
        (huge, huge), maximum_rational_bits=_RATIONAL_BITS
    )
    fiber_size = parent.scope_transverse_indices.shape[0]

    def overflowing_roots(*args: object, **kwargs: object) -> object:
        return (
            (huge_root,) * fiber_size,
            (None,) * fiber_size,
            (None,) * fiber_size,
            (None,) * fiber_size,
            (0,) * fiber_size,
            (EntireEnclosureFailure.WORK_BUDGET_EXCEEDED,) * fiber_size,
            (0,) * fiber_size,
        )

    overflow_policies = (
        _PRECISION,
        _ENTIRE_TERMS,
        _ENTIRE_WORK,
        _RANGE_REDUCTIONS,
        _RATIONAL_BITS,
    )
    with monkeypatch.context() as context:
        context.setattr(
            terminal, "_classify_physical_roots", overflowing_roots
        )
        root_range = terminal._branch_evidence(
            parent,
            canonical.inner_current_diagnostic,
            canonical.outer_current_diagnostic,
            np.float64(canonical.comparison_plane_coordinate),
            np.float64(canonical.defining_plane_coordinate),
            _BRANCH_TERMS,
            _ROOT_WORK,
            overflow_policies,
            _INTERVAL_WORK,
            _RATIONAL_BITS,
        )
    failure = GalerkinLocalVacuumTerminalFailure(int(root_range.failure_mask))
    assert (
        failure & GalerkinLocalVacuumTerminalFailure.ARITHMETIC_RANGE_FAILURE
    )
    assert not failure & (
        GalerkinLocalVacuumTerminalFailure.ENTIRE_HELPER_ENCLOSURE_FAILURE
        | GalerkinLocalVacuumTerminalFailure.DIRECT_RATIONAL_SIZE_FAILURE
    )
    assert root_range.hull_range_failure
    assert root_range.root_certificates[0] is not None
    assert stored_value_payload(root_range.root_certificates[0]) == (
        stored_value_payload(huge_root)
    )
    assert_array_equal(root_range.frozen_positive_root_realizations, 0.0)
    assert_array_equal(root_range.frozen_positive_root_error_bounds, 0.0)
    assert_array_equal(root_range.physical_phase_realizations, 0.0)


def test_public_replay_rejects_digest_disposition_policy_and_tracing() -> None:
    """Reject coherent trust, policy, and host-boundary forgeries.

    :see: :func:`ptyrodactyl.galerkin.certify_local_vacuum_terminal`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_vacuum_terminal_certificate`
    """
    certificate = _public_terminal()
    canonical = _prepare(certificate)
    assert bool(eqx.tree_equal(canonical, certificate))

    alternate_branch = _terminal(_SELECTED).branch_evidence
    forged_floor_count = alternate_branch.hull_normal_floor_count + 1
    assert forged_floor_count <= (
        alternate_branch.hull_completed_endpoint_count
    )
    forged_hull_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.hull.v1",
            "algorithm": alternate_branch.hull_algorithm,
            "maximum_rational_bits": alternate_branch.maximum_rational_bits,
            "attempted_endpoints": (
                alternate_branch.hull_attempted_endpoint_count
            ),
            "completed_endpoints": (
                alternate_branch.hull_completed_endpoint_count
            ),
            "input_peak_bits": alternate_branch.hull_input_peak_bits,
            "output_peak_bits": alternate_branch.hull_output_peak_bits,
            "normal_floor_count": forged_floor_count,
            "range_failure": alternate_branch.hull_range_failure,
        }
    )
    alternate_branch = replace(
        alternate_branch,
        hull_normal_floor_count=forged_floor_count,
        hull_evidence_digest=forged_hull_digest,
    )
    assert alternate_branch.hull_evidence_digest == (forged_hull_digest)
    projection_digest = certificate.projection_certificate.certificate_digest
    inner_digest = (
        certificate.inner_current_diagnostic.diagnostic_evidence_digest
    )
    outer_digest = (
        certificate.outer_current_diagnostic.diagnostic_evidence_digest
    )
    forged_digest = sha256(
        {
            "domain": "ptyrodactyl.local_vacuum_terminal.evidence.v1",
            "identity": certificate.terminal_identity_digest,
            "projection": projection_digest,
            "inner": inner_digest,
            "outer": outer_digest,
            "branch": alternate_branch.branch_evidence_digest,
            "cut": certificate.cut_balance.cut_balance_digest,
            "failure": int(certificate.failure_mask),
            "predicates": stored_value_payload(
                tuple(
                    jnp.asarray(value)
                    for value in (
                        certificate.current_diagnostic_eligible,
                        certificate.current_operator_eligible,
                        certificate.current_action_eligible,
                        certificate.vacuum_branch_eligible,
                    )
                )
            ),
            "terminal_direct_pairs": _TERMINAL_PAIRS,
            "branch_direct_terms": _BRANCH_TERMS,
            "cut_direct_pairs": _CUT_PAIRS,
            "root_work": _ROOT_WORK,
            "entire_policies": stored_value_payload(_ENTIRE_POLICIES),
            "interval_work": _INTERVAL_WORK,
            "rational_bits": _RATIONAL_BITS,
        }
    )
    forged = replace(
        certificate,
        branch_evidence=alternate_branch,
        branch_evidence_digest=alternate_branch.branch_evidence_digest,
        terminal_evidence_digest=forged_digest,
    )
    assert (
        forged.terminal_evidence_digest != certificate.terminal_evidence_digest
    )
    assert forged.branch_evidence.branch_evidence_digest == (
        forged.branch_evidence_digest
    )
    with pytest.raises(ValueError, match="complete replay"):
        _prepare(forged)
    with pytest.raises(ValueError, match="complete replay"):
        _prepare(certificate, disposition=_NATIVE_FULL)
    with pytest.raises(ValueError, match="complete replay"):
        _prepare(certificate, branch_terms=_BRANCH_TERMS - 1)

    with pytest.raises(ValueError, match="concrete host"):
        eqx.filter_jit(
            lambda value: certify_local_vacuum_terminal(
                value,
                disposition=_PLANE,
                maximum_state_error=_STATE_BUDGET,
            )
        )(_projection(_FULL))
    with pytest.raises(ValueError, match="greater than 1"):
        certify_local_vacuum_terminal(
            _projection(_FULL),
            disposition=_PLANE,
            maximum_state_error=_STATE_BUDGET,
            maximum_rational_bits=1,
        )


__all__: list[str] = []
