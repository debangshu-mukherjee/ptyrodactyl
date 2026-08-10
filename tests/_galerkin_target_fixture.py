"""Shared checked scalar-target fixtures for direct Galerkin tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Tuple

from ptyrodactyl.born.acquisition import (
    check_galerkin_acquisition_support,
)
from ptyrodactyl.born.system import create_galerkin_target
from ptyrodactyl.types.acquisition_types import (
    GalerkinAcquisitionSupportResult,
    GalerkinBackwardDisposition,
    GalerkinCarrierOverlapDisposition,
    GalerkinCarrierOwnership,
    GalerkinCarrierTargetRoute,
    GalerkinDirectionDisposition,
    GalerkinEndpointConvention,
    GalerkinTerminalSide,
    create_galerkin_acquisition_manifest,
)
from ptyrodactyl.types.born_potential_types import (
    GalerkinProductSupport,
    create_galerkin_product_support,
)
from ptyrodactyl.types.constants import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    M_E,
)
from ptyrodactyl.types.galerkin_types import GalerkinTargetManifest
from ptyrodactyl.types.potential_types import Potential3D, create_potential_3d

TARGET_VOLTAGE_KV: float = 200.0
TARGET_CAP_SCALE: float = 0.25


def stored_wavenumber(voltage_kv: jax.Array | float) -> jax.Array:
    """Return the canonical algebraic Planck-form binary64 wavenumber."""
    voltage = jnp.asarray(voltage_kv, dtype=jnp.float64)
    energy = voltage * 1000.0 * jnp.asarray(E_CHARGE)
    wavelength_metre = jnp.sqrt(
        (jnp.asarray(H_PLANCK) * jnp.asarray(C_LIGHT)) ** 2
        / (
            energy
            * (2.0 * jnp.asarray(M_E) * jnp.asarray(C_LIGHT) ** 2 + energy)
        )
    )
    return 2.0 * jnp.pi / (1.0e10 * wavelength_metre)


def target_support() -> GalerkinProductSupport:
    """Return a three-state support with complete product and CAP sets."""
    state = jnp.asarray(
        ((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
        dtype=jnp.int64,
    )
    shell = [
        (first, second, third)
        for first in range(-1, 2)
        for second in range(-1, 2)
        for third in range(-1, 2)
    ]
    absorber = jnp.asarray(
        [(-2, 0, 0), *shell, (2, 0, 0)],
        dtype=jnp.int64,
    )
    work = jnp.asarray(
        [
            (first, second, third)
            for first in range(-3, 4)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    return create_galerkin_product_support(
        state_indices=state,
        interaction_indices=state,
        absorber_indices=absorber,
        work_indices=work,
        work_shape=(7, 3, 3),
    )


def checked_acquisition(
    support: GalerkinProductSupport,
    box_lengths: Tuple[float, float, float],
    *,
    voltage_kv: float = TARGET_VOLTAGE_KV,
    terminal_axis: int = 0,
    carrier_direction: Tuple[float, float, float] | None = None,
    projected_offset: Tuple[float, float, float] | None = None,
    backward_disposition: GalerkinBackwardDisposition = (
        GalerkinBackwardDisposition.EXCLUDED
    ),
    claims_backscatter: bool = False,
) -> GalerkinAcquisitionSupportResult:
    """Build a complete eligible artifact for a full coordinate fiber."""
    wavenumber = stored_wavenumber(voltage_kv)
    if carrier_direction is None:
        carrier = (
            jnp.zeros((3,), dtype=jnp.float64)
            .at[terminal_axis]
            .set(wavenumber)
        )
    else:
        direction = jnp.asarray(carrier_direction, dtype=jnp.float64)
        carrier = wavenumber * direction / jnp.linalg.norm(direction)
    zero = jnp.zeros((1, 3), dtype=jnp.int64)
    transverse_axes = tuple(axis for axis in range(3) if axis != terminal_axis)
    transverse = support.state_indices[:, transverse_axes]
    transverse = jnp.unique(transverse, axis=0)
    projected = projected_offset is not None
    offset = (
        jnp.zeros((3,), dtype=jnp.float64)
        if projected_offset is None
        else jnp.asarray(projected_offset, dtype=jnp.float64)
    )
    physical = (carrier + offset)[None, :]
    disposition = (
        GalerkinDirectionDisposition.PROJECTED
        if projected
        else GalerkinDirectionDisposition.EXACT_COEFFICIENT
    )
    shell_bound = 1.0 if projected else 1.0e-8
    shell_tolerance = 2.0 if projected else 1.0e-7
    projection_bound = 1.0 if projected else 0.0
    manifest = create_galerkin_acquisition_manifest(
        support,
        zero,
        zero,
        support.state_indices,
        transverse,
        jnp.zeros((0, 3), dtype=jnp.int64),
        incident_physical_wavevectors=physical,
        outgoing_physical_wavevectors=physical,
        incident_direction_dispositions=jnp.asarray(
            [disposition],
            dtype=jnp.int32,
        ),
        outgoing_direction_dispositions=jnp.asarray(
            [disposition],
            dtype=jnp.int32,
        ),
        incident_on_shell_defect_bounds=jnp.asarray(
            [shell_bound], dtype=jnp.float64
        ),
        outgoing_on_shell_defect_bounds=jnp.asarray(
            [shell_bound], dtype=jnp.float64
        ),
        incident_projection_error_bounds=jnp.asarray(
            [projection_bound], dtype=jnp.float64
        ),
        outgoing_projection_error_bounds=jnp.asarray(
            [projection_bound], dtype=jnp.float64
        ),
        carrier=carrier,
        box_lengths=jnp.asarray(box_lengths, dtype=jnp.float64),
        wavenumber=wavenumber,
        carrier_on_shell_defect_bound=jnp.asarray(1.0e-8, dtype=jnp.float64),
        on_shell_defect_tolerance=jnp.asarray(
            shell_tolerance,
            dtype=jnp.float64,
        ),
        terminal_axis=terminal_axis,
        terminal_side=GalerkinTerminalSide.POSITIVE,
        carrier_id="fixture-carrier-0",
        carrier_ownership=(
            GalerkinCarrierOwnership.INDEPENDENT_SINGLE_CARRIER
        ),
        carrier_overlap_disposition=(
            GalerkinCarrierOverlapDisposition.NO_OTHER_CARRIER_BLOCKS
        ),
        carrier_target_route=(
            GalerkinCarrierTargetRoute.NORMALIZE_FROM_ACCELERATING_VOLTAGE
        ),
        endpoint_convention=GalerkinEndpointConvention.SIGNED_HALF_OPEN,
        backward_disposition=backward_disposition,
        backward_exclusion_basis=(
            "forward-only fixture; no backscatter claim"
            if backward_disposition is GalerkinBackwardDisposition.EXCLUDED
            else ""
        ),
        claims_backscatter=claims_backscatter,
    )
    return check_galerkin_acquisition_support(manifest)


def periodic_target_potential() -> Potential3D:
    """Synthesize one real periodic volume with known retained harmonics."""
    nx, ny, nz = 5, 3, 3
    x = jnp.arange(nx, dtype=jnp.float64)
    line = (
        2.0
        + 0.4 * jnp.cos(2.0 * jnp.pi * x / nx)
        + 0.2 * jnp.sin(2.0 * jnp.pi * x / nx)
    )
    volume = jnp.broadcast_to(line, (nz, ny, nx))
    return create_potential_3d(
        volume,
        voxel_size=(1.0, 1.0, 1.0),
        box_size=(5.0, 3.0, 3.0),
        origin=(0.0, 0.0, 0.0),
        producer="shared-galerkin-target-fixture-v1",
        provenance_hash="d" * 64,
        coefficient_normalization="VC-1 periodic trigonometric mean DFT",
        band_limit=0.3,
    )


def production_target() -> GalerkinTargetManifest:
    """Build the shared end-to-end production target."""
    potential = periodic_target_potential()
    eligibility = checked_acquisition(target_support(), potential.box_size)
    return create_galerkin_target(
        potential,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name="shared-production-target",
    )


def production_vacuum_target() -> GalerkinTargetManifest:
    """Build the shared target with an exactly zero voxel potential."""
    potential = periodic_target_potential()
    vacuum: Potential3D = eqx.tree_at(
        lambda candidate: candidate.volume,
        potential,
        jnp.zeros_like(potential.volume),
    )
    eligibility = checked_acquisition(target_support(), vacuum.box_size)
    target: GalerkinTargetManifest = create_galerkin_target(
        vacuum,
        eligibility,
        accelerating_voltage_kv=TARGET_VOLTAGE_KV,
        cap_scale=TARGET_CAP_SCALE,
        target_name="shared-production-vacuum-target",
    )
    return target


__all__: list[str] = [
    "TARGET_CAP_SCALE",
    "TARGET_VOLTAGE_KV",
    "checked_acquisition",
    "periodic_target_potential",
    "production_target",
    "production_vacuum_target",
    "stored_wavenumber",
    "target_support",
]
