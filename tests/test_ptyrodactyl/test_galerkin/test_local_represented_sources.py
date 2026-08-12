r"""Tests for direct represented ``LOCAL_CELL_LVT1`` source evidence."""

from __future__ import annotations

import functools
import inspect
import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import ptyrodactyl.galerkin.local_represented_sources as represented
from ptyrodactyl._tools import RootEnclosureError
from ptyrodactyl.galerkin.absorber import (
    apply_axial_physical_cap,
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
    apply_local_cell_interaction,
    certify_local_cell_exact_compression,
    create_local_cell_interaction_core,
)
from ptyrodactyl.galerkin.local_cell_system import (
    compose_local_cell_galerkin_target,
)
from ptyrodactyl.galerkin.local_represented_sources import (
    certify_local_represented_source,
    compose_local_represented_focused_source,
    compose_local_represented_plane_source,
    prepare_local_represented_source,
    prepare_local_represented_source_certificate,
)
from ptyrodactyl.galerkin.local_sources import (
    certify_local_additional_source,
    realize_local_cell_additional_source,
)
from ptyrodactyl.types.local_cell_target_types import (
    GalerkinLocalCellTargetManifest,
)
from ptyrodactyl.types.local_cell_types import (
    create_local_cell_potential_3d,
)
from ptyrodactyl.types.local_represented_source_types import (
    GalerkinLocalComplexRectangles,
    GalerkinLocalRepresentedSource,
    GalerkinLocalRepresentedSourceCertificate,
    GalerkinLocalRepresentedSourceFailure,
    GalerkinLocalRepresentedSourceKind,
    GalerkinLocalSourceAxis,
    GalerkinLocalSourcePhaseConvention,
)
from ptyrodactyl.types.local_source_types import (
    GalerkinLocalAdditionalSourceCertificate,
)
from ptyrodactyl.types.local_zero_slab_types import (
    GalerkinLocalVacuumReference,
)
from tests._galerkin_target_fixture import checked_acquisition
from tests.test_ptyrodactyl.test_galerkin.test_absorber import (
    _HIGH_PRECISION_COEFFICIENT_BRACKETS,
)
from tests.test_ptyrodactyl.test_galerkin.test_local_cell_interaction import (
    _line_support,
)

_PHASE = GalerkinLocalSourcePhaseConvention.PHYSICAL_WAVEVECTOR
_PROVENANCE = "8" * 64

type _RationalRectangle = tuple[Fraction, Fraction, Fraction, Fraction]


@functools.lru_cache(maxsize=1)
def _cap_proof():
    """Build one canonical-reference zero-potential L1--L4 parent."""
    support = _line_support()
    potential = create_local_cell_potential_3d(
        jnp.zeros((1, 1, 3), dtype=jnp.float64),
        cell_size=(1.0, 1.0, 1.0),
        box_size=(3.0, 1.0, 1.0),
        cell_center_origin=(0.1, 0.2, 0.3),
        reference_value=0.0,
        reference_semantics=(
            GalerkinLocalVacuumReference.VACUUM_K0_CARRIER.value
        ),
        producer="represented-source-test-v1",
        provenance_hash=_PROVENANCE,
        producer_coefficient_normalization="producer metadata only",
        producer_bandwidth=1.0,
    )
    acquisition = checked_acquisition(support, potential.box_size)
    realization = realize_local_cell_galerkin_potential(potential, acquisition)
    certificate = certify_local_cell_galerkin_potential(
        realization, maximum_direct_terms=9
    )
    compression = certify_local_cell_exact_compression(
        certificate, accelerating_voltage_kv=200.0
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
    cap_certificate = certify_axial_cell_absorber(absorber)
    proof = certify_axial_cap_floor(
        cap_certificate,
        gram_precision_bits=32,
        ldl_iteration_count=40,
    )
    return proof


@functools.lru_cache(maxsize=4)
def _target(
    name: str = "represented-source-target",
) -> GalerkinLocalCellTargetManifest:
    """Compose one name-selected target from the shared canonical proof."""
    return compose_local_cell_galerkin_target(_cap_proof(), target_name=name)


def _complex_constant_cells() -> jax.Array:
    """Return a genuinely complex constant LVT.20 source cell field."""
    return jnp.full((1, 1, 3), 1.0 + 2.0j, dtype=jnp.complex128)


@functools.lru_cache(maxsize=4)
def _additional(
    name: str = "represented-source-target",
) -> GalerkinLocalAdditionalSourceCertificate:
    """Return one nonzero directly certified LOCAL_CELL source."""
    source = realize_local_cell_additional_source(
        _target(name), _complex_constant_cells()
    )
    return certify_local_additional_source(source, maximum_direct_terms=9)


@functools.lru_cache(maxsize=4)
def _source(
    name: str = "represented-source-target",
) -> GalerkinLocalRepresentedSource:
    """Return one exact-shell forward plane source on the zero mode."""
    target = _target(name)
    zero_position = int(
        np.flatnonzero(np.all(np.asarray(target.state_indices) == 0, axis=1))[
            0
        ]
    )
    return compose_local_represented_plane_source(
        target,
        _additional(name),
        zero_position,
        jnp.asarray(0.75 - 0.25j, dtype=jnp.complex128),
        jnp.asarray(1.25, dtype=jnp.float64),
        phase_convention=_PHASE,
        source_plane_coordinate=jnp.asarray(0.125, dtype=jnp.float64),
        scan_position=jnp.asarray([0.0, 0.2, -0.1], dtype=jnp.float64),
        aberration_phase=jnp.asarray(0.3, dtype=jnp.float64),
        source_name=f"{name}-plane",
    )


@functools.lru_cache(maxsize=4)
def _certificate(
    name: str = "represented-source-target",
) -> GalerkinLocalRepresentedSourceCertificate:
    """Return one finite direct represented-source certificate."""
    return certify_local_represented_source(
        _source(name), maximum_direct_pairs=21
    )


def _scale_interval(
    interval: tuple[Fraction, Fraction], scalar: Fraction
) -> tuple[Fraction, Fraction]:
    """Scale one rational interval by an exact rational scalar."""
    values = (interval[0] * scalar, interval[1] * scalar)
    return min(values), max(values)


def _add_interval(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    """Add two rational intervals exactly."""
    return left[0] + right[0], left[1] + right[1]


def _multiply_rectangle_by_point(
    rectangle: _RationalRectangle,
    point: complex,
) -> _RationalRectangle:
    """Multiply one rational complex rectangle by one binary64 point."""
    real = Fraction.from_float(float(point.real))
    imag = Fraction.from_float(float(point.imag))
    real_part = _add_interval(
        _scale_interval(rectangle[:2], real),
        _scale_interval(rectangle[2:], -imag),
    )
    imag_part = _add_interval(
        _scale_interval(rectangle[:2], imag),
        _scale_interval(rectangle[2:], real),
    )
    return (*real_part, *imag_part)


def _add_rectangles(
    left: _RationalRectangle,
    right: _RationalRectangle,
) -> _RationalRectangle:
    """Add two rational complex rectangles exactly."""
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
    )


def _minus_i_rectangle(
    value: _RationalRectangle,
) -> _RationalRectangle:
    """Multiply one rational complex rectangle by exact ``-i``."""
    return value[2], value[3], -value[1], -value[0]


def _oracle_rectangles(
    source: GalerkinLocalRepresentedSource,
) -> tuple[tuple[_RationalRectangle, ...], ...]:
    """Build independent high-precision D/B/R/S/M/T/C rectangles."""
    size = source.target.state_indices.shape[0]
    zero: _RationalRectangle = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    active = int(np.flatnonzero(np.asarray(source.modes.active_mask))[0])
    incident = complex(np.asarray(source.modes.incident_field)[active])
    state = np.asarray(source.target.state_indices)
    cap: list[_RationalRectangle] = []
    for row in range(size):
        mode = int((state[row] - state[active])[0])
        coefficient = _HIGH_PRECISION_COEFFICIENT_BRACKETS[mode]
        scaled_coefficient: _RationalRectangle = (
            coefficient[0] * Fraction(1, 4),
            coefficient[1] * Fraction(1, 4),
            coefficient[2] * Fraction(1, 4),
            coefficient[3] * Fraction(1, 4),
        )
        cap.append(_multiply_rectangle_by_point(scaled_coefficient, incident))
    with localcontext() as context:
        context.prec = 110
        root = Decimal(3).sqrt()
        width = Decimal("1e-100")
        root_interval = (
            Fraction(root - width),
            Fraction(root + width),
        )
    additional = [zero for _ in range(size)]
    additional[active] = (
        root_interval[0],
        root_interval[1],
        2 * root_interval[0],
        2 * root_interval[1],
    )
    free = tuple(zero for _ in range(size))
    interaction = tuple(zero for _ in range(size))
    cap_tuple = tuple(cap)
    additional_tuple = tuple(additional)
    matched = tuple(_minus_i_rectangle(value) for value in cap_tuple)
    total = tuple(
        _add_rectangles(left, right)
        for left, right in zip(matched, additional_tuple, strict=True)
    )
    scattered = additional_tuple
    return (
        free,
        cap_tuple,
        interaction,
        additional_tuple,
        matched,
        total,
        scattered,
    )


def _stored_rectangles(
    certificate: GalerkinLocalRepresentedSourceCertificate,
) -> tuple[GalerkinLocalComplexRectangles, ...]:
    """Return certificate rectangles in canonical D/B/R/S/M/T/C order."""
    return (
        certificate.free_rectangles,
        certificate.physical_cap_rectangles,
        certificate.interaction_rectangles,
        certificate.additional_source_rectangles,
        certificate.vacuum_matched_rectangles,
        certificate.total_source_rectangles,
        certificate.scattered_source_rectangles,
    )


def _component_errors(
    certificate: GalerkinLocalRepresentedSourceCertificate,
) -> tuple[jax.Array, ...]:
    """Return component errors in canonical D/B/R/S/M/T/C order."""
    return (
        certificate.free_component_error_bounds,
        certificate.physical_cap_component_error_bounds,
        certificate.interaction_component_error_bounds,
        certificate.additional_source_component_error_bounds,
        certificate.vacuum_matched_component_error_bounds,
        certificate.total_source_component_error_bounds,
        certificate.scattered_source_component_error_bounds,
    )


def _action_bounds(
    certificate: GalerkinLocalRepresentedSourceCertificate,
) -> tuple[jax.Array, ...]:
    """Return action bounds in canonical D/B/R/S/M/T/C order."""
    return (
        certificate.free_action_error_upper_bound,
        certificate.physical_cap_action_error_upper_bound,
        certificate.interaction_action_error_upper_bound,
        certificate.additional_source_error_upper_bound,
        certificate.vacuum_matched_source_error_upper_bound,
        certificate.total_source_error_upper_bound,
        certificate.scattered_source_error_upper_bound,
    )


def _decimal_sqrt_fraction(value: Fraction) -> Decimal:
    """Evaluate one non-negative rational square root at high precision."""
    with localcontext() as context:
        context.prec = 100
        result = (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()
    return result


def test_direct_complex_source_rectangles_errors_norms_and_actions() -> None:
    """Enclose an independent complex nonzero D/B/R/S/M/T/C oracle.

    :see: :func:`ptyrodactyl.galerkin.certify_local_represented_source`
    :see: :func:`ptyrodactyl.galerkin.\
compose_local_represented_plane_source`
    """
    source = _source()
    certificate = _certificate()
    assert source.failure is GalerkinLocalRepresentedSourceFailure.NONE
    assert bool(source.incident_eligible)
    assert certificate.failure is GalerkinLocalRepresentedSourceFailure.NONE
    assert bool(certificate.finite_certificate)
    assert source.normal_axis is GalerkinLocalSourceAxis.X
    assert_array_equal(
        source.modes.active_mask,
        (jnp.real(source.modes.phased_coefficients) != 0.0)
        | (jnp.imag(source.modes.phased_coefficients) != 0.0),
    )
    assert_array_equal(
        source.modes.active_mask,
        (jnp.real(source.modes.incident_field) != 0.0)
        | (jnp.imag(source.modes.incident_field) != 0.0),
    )
    assert_allclose(
        source.actions.vacuum_matched_source,
        source.actions.free_action - 1j * source.actions.physical_cap_action,
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        source.actions.total_source,
        source.actions.vacuum_matched_source
        + source.actions.additional_source,
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        source.actions.scattered_source,
        source.actions.interaction_action + source.actions.additional_source,
        rtol=0.0,
        atol=0.0,
    )
    assert np.any(np.real(np.asarray(source.actions.additional_source)) != 0.0)
    assert np.any(np.imag(np.asarray(source.actions.additional_source)) != 0.0)

    oracle = _oracle_rectangles(source)
    points = tuple(source.actions)
    for action_position, (stored, reference) in enumerate(
        zip(_stored_rectangles(certificate), oracle, strict=True)
    ):
        stored_arrays = tuple(np.asarray(values) for values in stored)
        component_bounds = np.asarray(
            _component_errors(certificate)[action_position]
        )
        for row, exact_rectangle in enumerate(reference):
            stored_rectangle = tuple(
                Fraction.from_float(float(values[row]))
                for values in stored_arrays
            )
            assert stored_rectangle[0] <= exact_rectangle[0]
            assert exact_rectangle[1] <= stored_rectangle[1]
            assert stored_rectangle[2] <= exact_rectangle[2]
            assert exact_rectangle[3] <= stored_rectangle[3]
            point = complex(np.asarray(points[action_position])[row])
            real = Fraction.from_float(float(point.real))
            imag = Fraction.from_float(float(point.imag))
            real_radius = max(
                abs(real - exact_rectangle[0]),
                abs(real - exact_rectangle[1]),
            )
            imag_radius = max(
                abs(imag - exact_rectangle[2]),
                abs(imag - exact_rectangle[3]),
            )
            radius = _decimal_sqrt_fraction(real_radius**2 + imag_radius**2)
            assert Decimal.from_float(float(component_bounds[row])) >= radius
    for components, bound in zip(
        _component_errors(certificate),
        _action_bounds(certificate),
        strict=True,
    ):
        squared = sum(
            (
                Fraction.from_float(float(value)) ** 2
                for value in np.asarray(components)
            ),
            start=Fraction(0),
        )
        assert Decimal.from_float(float(bound)) >= _decimal_sqrt_fraction(
            squared
        )
    incident_squared = sum(
        (
            Fraction.from_float(float(value.real)) ** 2
            + Fraction.from_float(float(value.imag)) ** 2
            for value in np.asarray(source.modes.incident_field)
        ),
        start=Fraction(0),
    )
    assert Decimal.from_float(
        float(certificate.incident_field_norm_upper_bound)
    ) >= _decimal_sqrt_fraction(incident_squared)
    assert prepare_local_represented_source(source).source_digest == (
        source.source_digest
    )
    assert (
        prepare_local_represented_source_certificate(
            certificate
        ).certificate_digest
        == certificate.certificate_digest
    )


def test_incident_v_action_static_and_certificate_rehash_forgery() -> None:
    """Reject fully replayed one-bit, action, static, and ledger forgeries.

    :see: :func:`ptyrodactyl.galerkin.prepare_local_represented_source`
    :see: :func:`ptyrodactyl.galerkin.\
prepare_local_represented_source_certificate`
    """
    source = _source()
    modes = source.modes
    active = int(np.flatnonzero(np.asarray(modes.active_mask))[0])
    changed_real = np.nextafter(
        float(jnp.real(modes.incident_field[active])), np.inf
    )
    changed_v = modes.incident_field.at[active].set(
        changed_real + 1j * jnp.imag(modes.incident_field[active])
    )
    assert changed_v[active] != modes.incident_field[active]
    changed_digest = represented._source_digest(
        source.target,
        source.additional_source_certificate,
        modes.aperture_weights,
        modes.target_reduced_flux,
        modes.scan_position,
        modes.aberration_phases,
        modes.source_plane_coordinate,
        changed_v,
        source.kind,
        source.normal_axis,
        source.phase_convention,
    )
    assert changed_digest != source.source_digest
    changed_modes = modes._replace(incident_field=changed_v)
    free = source.target.free_diagonal * changed_v
    cap = apply_axial_physical_cap(source.target.cap_floor_proof, changed_v)
    interaction = apply_local_cell_interaction(
        source.target.interaction_core, changed_v
    )
    additional = source.actions.additional_source
    changed_actions = source.actions._replace(
        free_action=free,
        physical_cap_action=cap,
        interaction_action=interaction,
        vacuum_matched_source=free - 1j * cap,
        total_source=free - 1j * cap + additional,
        scattered_source=interaction + additional,
    )
    changed_evidence = represented._source_evidence_digest(
        source.target,
        source.additional_source_certificate,
        changed_modes,
        changed_actions,
        source.failure,
        changed_digest,
        source.source_name,
    )
    forged_v = replace(
        source,
        modes=changed_modes,
        actions=changed_actions,
        source_digest=changed_digest,
        source_evidence_digest=changed_evidence,
    )
    with pytest.raises(ValueError, match="complete parent/source replay"):
        prepare_local_represented_source(forged_v)

    action_forgery = source.actions._replace(
        total_source=source.actions.total_source.at[0].set(
            source.actions.total_source[0]
            + jnp.asarray(1.0e-12j, dtype=jnp.complex128)
        )
    )
    action_evidence = represented._source_evidence_digest(
        source.target,
        source.additional_source_certificate,
        source.modes,
        action_forgery,
        source.failure,
        source.source_digest,
        source.source_name,
    )
    with pytest.raises(ValueError, match="complete parent/source replay"):
        prepare_local_represented_source(
            replace(
                source,
                actions=action_forgery,
                source_evidence_digest=action_evidence,
            )
        )
    with pytest.raises(ValueError, match="complete parent/source replay"):
        prepare_local_represented_source(
            replace(source, total_source_formula="forged T formula")
        )

    certificate = _certificate()
    changed_bound = jnp.nextafter(
        certificate.total_source_error_upper_bound,
        jnp.asarray(jnp.inf, dtype=jnp.float64),
    )
    rectangles = _stored_rectangles(certificate)
    errors = _component_errors(certificate)
    bounds = list(_action_bounds(certificate))
    bounds[5] = changed_bound
    digest = represented._certificate_digest(
        source,
        rectangles,
        errors,
        tuple(bounds),
        certificate.incident_field_norm_upper_bound,
        certificate.direct_pair_count,
        certificate.maximum_direct_pairs,
        certificate.failure,
    )
    with pytest.raises(ValueError, match="complete replay"):
        prepare_local_represented_source_certificate(
            replace(
                certificate,
                total_source_error_upper_bound=changed_bound,
                certificate_digest=digest,
            )
        )


def test_budget_host_root_range_and_exact_count_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return typed direct failures and reject an unrepresentable count."""
    source = _source()
    budget = represented._certify_canonical(source, 1)
    assert budget.failure is (
        GalerkinLocalRepresentedSourceFailure.DIRECT_WORK_BUDGET_EXCEEDED
    )
    with monkeypatch.context() as patch:
        patch.setattr(represented, "host_binary64_supported", lambda: False)
        host = represented._certify_canonical(source, 21)
    assert host.failure is (
        GalerkinLocalRepresentedSourceFailure.HOST_ARITHMETIC_UNSUPPORTED
    )

    def fail_root(value: Fraction) -> Fraction:
        """Force every verified rational root enclosure to fail."""
        del value
        raise RootEnclosureError("forced represented-source root failure")

    with monkeypatch.context() as patch:
        patch.setattr(represented, "sqrt_fraction_upper", fail_root)
        root = represented._certify_canonical(source, 21)
    assert (
        root.failure
        is GalerkinLocalRepresentedSourceFailure.ROOT_ENCLOSURE_FAILURE
    )

    infinite = represented._infinite_rectangles(
        source.target.state_indices.shape[0]
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            represented,
            "_exact_action_rectangles",
            lambda submitted: (infinite,) * 7,
        )
        ranged = represented._certify_canonical(source, 21)
    assert ranged.failure is (
        GalerkinLocalRepresentedSourceFailure.ARITHMETIC_RANGE_FAILURE
    )

    maximum = np.iinfo(np.int64).max
    boundary = (math.isqrt(1 + 8 * maximum) - 1) // 4
    while (boundary + 1) + 2 * (boundary + 1) ** 2 <= maximum:
        boundary += 1
    assert represented._direct_pair_count(boundary) <= maximum
    with pytest.raises(ValueError, match="fit signed int64"):
        represented._direct_pair_count(boundary + 1)


def test_incident_gates_parent_coherence_and_derived_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail disposition, shell, branch, fiber, and parent gates.

    :see: :func:`ptyrodactyl.galerkin.\
compose_local_represented_focused_source`
    """
    signature = inspect.signature(compose_local_represented_plane_source)
    assert "normal_axis" not in signature.parameters
    assert (
        "normal_axis"
        not in inspect.signature(
            compose_local_represented_focused_source
        ).parameters
    )
    canonical_source = _source()
    target = canonical_source.target
    additional = canonical_source.additional_source_certificate
    active = jnp.asarray([True, False])
    true = jnp.asarray([True, True])
    exact_flux = (
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )

    def outcome(
        *,
        declared: jax.Array = true,
        exact: jax.Array = true,
        shell: jax.Array = true,
        forward: jax.Array = true,
        duplicate: bool = False,
        flux: tuple[jax.Array, jax.Array] = exact_flux,
    ) -> GalerkinLocalRepresentedSourceFailure:
        """Evaluate one isolated represented incident gate."""
        return represented._incident_failure(
            target,
            additional,
            active,
            forward,
            declared,
            exact,
            shell,
            forward,
            jnp.asarray(duplicate),
            flux,
            GalerkinLocalSourceAxis.X,
        )

    assert outcome(declared=jnp.asarray([False, True])) is (
        GalerkinLocalRepresentedSourceFailure.UNDECLARED_INCIDENT_MODE
    )
    assert outcome(exact=jnp.asarray([False, True])) is (
        GalerkinLocalRepresentedSourceFailure.NONEXACT_INCIDENT_DISPOSITION
    )
    assert outcome(shell=jnp.asarray([False, True])) is (
        GalerkinLocalRepresentedSourceFailure.EXACT_SHELL_FAILURE
    )
    assert outcome(forward=jnp.asarray([False, True])) is (
        GalerkinLocalRepresentedSourceFailure.NONFORWARD_OR_GRAZING
    )
    assert outcome(duplicate=True) is (
        GalerkinLocalRepresentedSourceFailure.DUPLICATE_TRANSVERSE_FIBER
    )
    assert (
        outcome(flux=(jnp.asarray(0.0), jnp.asarray(1.0)))
        is GalerkinLocalRepresentedSourceFailure.NONPOSITIVE_EXACT_FLUX
    )
    with monkeypatch.context() as patch:
        patch.setattr(represented, "host_binary64_supported", lambda: False)
        assert outcome() is (
            GalerkinLocalRepresentedSourceFailure.HOST_ARITHMETIC_UNSUPPORTED
        )

    source = _source()
    focused_weights = source.modes.aperture_weights.at[0].set(0.2 + 0.1j)
    focused = compose_local_represented_focused_source(
        source.target,
        source.additional_source_certificate,
        focused_weights,
        source.modes.target_reduced_flux,
        phase_convention=_PHASE,
        source_plane_coordinate=source.modes.source_plane_coordinate,
        scan_position=source.modes.scan_position,
        aberration_phases=source.modes.aberration_phases,
        source_name="focused-fail-closed",
    )
    assert not bool(focused.incident_eligible)
    assert focused.failure is (
        GalerkinLocalRepresentedSourceFailure.UNDECLARED_INCIDENT_MODE
    )
    with pytest.raises(ValueError, match="identical target"):
        compose_local_represented_plane_source(
            _target("same-operator-different-evidence"),
            _additional(),
            1,
            1.0 + 0.0j,
            1.0,
            phase_convention=_PHASE,
            source_plane_coordinate=0.0,
            scan_position=jnp.zeros((3,), dtype=jnp.float64),
            aberration_phase=0.0,
            source_name="cross-pair-forgery",
        )
