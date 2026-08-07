"""Tests for :mod:`ptyrodactyl.multislice.potential_volume`."""

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from ptyrodactyl.multislice import (
    atomic_form_factor,
    lobato_bandlimited_peak,
    projected_atom_potential,
)
from ptyrodactyl.multislice.potential_volume import (
    crystal_potential_volume,
    single_atom_potential_3d,
)
from ptyrodactyl.types import (
    MOTT_BETHE_VOLT_ANGSTROM_SQ,
    CrystalData,
    create_crystal_data,
    create_crystal_structure,
)


def test_single_atom_volume_preserves_zero_mode_and_translation() -> None:
    """The periodic field keeps its physical mean and continuous shift."""
    spacing = 0.4
    shape = (6, 6, 6)
    center = jnp.array([0.6, 0.8, 1.0], dtype=jnp.float64)
    volume = single_atom_potential_3d(
        14,
        spacing,
        shape,
        center,
        band_limit=0.9,
    )
    translated = single_atom_potential_3d(
        14,
        spacing,
        shape,
        center + jnp.array([spacing, 0.0, 0.0]),
        band_limit=0.9,
    )
    box_volume = (spacing * shape[0]) ** 3
    expected_mean = (
        MOTT_BETHE_VOLT_ANGSTROM_SQ
        * atomic_form_factor(14, jnp.array(0.0))
        / box_volume
    )

    assert jnp.all(jnp.isfinite(volume))
    assert jnp.allclose(jnp.mean(volume), expected_mean, rtol=2e-14)
    assert jnp.allclose(translated, jnp.roll(volume, 1, axis=2), atol=2e-12)


def test_crystal_volume_is_additive_and_carries_physical_metadata() -> None:
    """The crystal producer superimposes atoms without removing the mean."""
    positions = jnp.array(
        [[0.7, 0.8, 0.9], [1.3, 1.2, 1.1]],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.array([6, 14], dtype=jnp.int32)
    crystal = create_crystal_data(
        positions,
        atomic_numbers,
        lattice=jnp.eye(3) * 2.0,
    )
    potential = crystal_potential_volume(
        crystal,
        0.25,
        (8, 8, 8),
        band_limit=1.5,
    )
    expected = sum(
        (
            single_atom_potential_3d(
                int(atomic_number),
                0.25,
                (8, 8, 8),
                position,
                band_limit=1.5,
            )
            for atomic_number, position in zip(
                atomic_numbers,
                positions,
                strict=True,
            )
        ),
        start=jnp.zeros((8, 8, 8), dtype=jnp.float64),
    )

    assert jnp.allclose(potential.volume, expected, atol=2e-12)
    assert potential.units == "V"
    assert potential.voxel_size == (0.25, 0.25, 0.25)
    assert potential.box_size == (2.0, 2.0, 2.0)
    assert potential.reference_value == 0.0
    assert "vacuum zero" in potential.reference_semantics
    assert "zero Fourier mode retained" in potential.reference_semantics
    assert potential.boundary == "periodic orthogonal box"
    assert potential.band_limit == 1.5


def test_automatic_grid_inference_preserves_noncommensurate_cell_extent() -> (
    None
):
    """Inference refines spacing instead of enlarging the physical cell."""
    cell_lengths = (2.1, 2.3, 2.6)
    requested_spacing = 0.5
    crystal = create_crystal_data(
        jnp.array([[0.7, 0.8, 0.9]], dtype=jnp.float64),
        jnp.array([14], dtype=jnp.int32),
        lattice=jnp.diag(jnp.asarray(cell_lengths, dtype=jnp.float64)),
    )

    potential = crystal_potential_volume(
        crystal,
        requested_spacing,
        band_limit=0.75,
    )

    expected_shape = (6, 5, 5)
    expected_voxel_size = (
        cell_lengths[0] / expected_shape[2],
        cell_lengths[1] / expected_shape[1],
        cell_lengths[2] / expected_shape[0],
    )
    assert potential.volume.shape == expected_shape
    assert potential.box_size == pytest.approx(cell_lengths)
    assert potential.voxel_size == pytest.approx(expected_voxel_size)
    assert all(actual <= requested_spacing for actual in potential.voxel_size)


def test_volume_projection_matches_independent_bandlimited_2d_series() -> None:
    """Integrating z selects the same gz=0 coefficients as a 2D series."""
    voxel_size = (0.4, 0.5, 0.6)
    shape = (6, 7, 8)
    center = jnp.array([1.1, 1.2, 1.3], dtype=jnp.float64)
    band_limit = 0.7
    volume = single_atom_potential_3d(
        6,
        voxel_size,
        shape,
        center,
        band_limit=band_limit,
    )
    nz, ny, nx = shape
    del nz
    dx, dy, dz = voxel_size
    gx_1d = jnp.fft.fftfreq(nx, d=dx)
    gy_1d = jnp.fft.fftfreq(ny, d=dy)
    gy, gx = jnp.meshgrid(gy_1d, gx_1d, indexing="ij")
    magnitude = jnp.sqrt(gx * gx + gy * gy)
    phase = jnp.exp(-2.0j * jnp.pi * (gx * center[0] + gy * center[1]))
    coefficients = (
        MOTT_BETHE_VOLT_ANGSTROM_SQ
        * atomic_form_factor(6, 2.0 * jnp.pi * magnitude)
        * (magnitude < band_limit)
        * phase
    )
    direct_projection = jnp.real(jnp.fft.ifftn(coefficients)) / (dx * dy)

    assert jnp.allclose(
        jnp.sum(volume, axis=0) * dz,
        direct_projection,
        atol=3e-12,
    )


def test_volume_projection_converges_to_analytic_lobato_projection() -> None:
    """A refined 3D cutoff approaches the analytic real-space projection."""
    spacing = 0.05
    shape = (4, 192, 192)
    center = jnp.array([4.8, 4.8, 0.1], dtype=jnp.float64)
    pixel_offsets = jnp.array([5, 8, 10, 12, 15], dtype=jnp.int32)
    radii = spacing * pixel_offsets
    analytic = projected_atom_potential(6, radii)

    def projected_samples(band_limit: float) -> jax.Array:
        volume = single_atom_potential_3d(
            6,
            spacing,
            shape,
            center,
            band_limit=band_limit,
        )
        projected = jnp.sum(volume, axis=0) * spacing
        center_index = shape[1] // 2
        return projected[center_index, center_index + pixel_offsets]

    coarse = projected_samples(4.0)
    refined = projected_samples(8.0)
    coarse_relative_error = jnp.max(jnp.abs((coarse - analytic) / analytic))
    refined_relative_error = jnp.max(jnp.abs((refined - analytic) / analytic))

    assert jnp.allclose(refined, analytic, rtol=4e-2, atol=0.0)
    assert refined_relative_error < 0.5 * coarse_relative_error


def test_larger_box_refines_the_bandlimited_on_nucleus_peak() -> None:
    """Finer reciprocal sampling converges toward the analytic Lobato peak."""
    exact_peak = lobato_bandlimited_peak(6, 1.5)
    errors: list[jax.Array] = []
    for grid_size in (8, 16):
        spacing = 0.25
        box_size = grid_size * spacing
        center = jnp.full((3,), box_size / 2.0)
        volume = single_atom_potential_3d(
            6,
            spacing,
            (grid_size, grid_size, grid_size),
            center,
            band_limit=1.5,
        )
        sampled_peak = volume[grid_size // 2, grid_size // 2, grid_size // 2]
        errors.append(jnp.abs(sampled_peak - exact_peak))

    assert errors[1] < errors[0] / 10.0


def test_atom_position_directional_gradient_matches_finite_difference() -> (
    None
):
    """Analytic phase ramps retain a nonzero three-direction position seam."""
    weights = jnp.arange(512, dtype=jnp.float64).reshape(8, 8, 8)

    def loss(position: jax.Array) -> jax.Array:
        volume = single_atom_potential_3d(
            14,
            0.25,
            (8, 8, 8),
            position,
            band_limit=1.5,
        )
        return jnp.sum(weights * volume) / weights.size

    position = jnp.array([0.91, 0.83, 0.72], dtype=jnp.float64)
    direction = jnp.array([0.3, -0.4, 0.5], dtype=jnp.float64)
    gradient = jax.grad(loss)(position)
    step = 1e-5
    finite_difference = (
        loss(position + step * direction) - loss(position - step * direction)
    ) / (2.0 * step)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.abs(gradient) > 0.0)
    assert jnp.allclose(
        jnp.vdot(gradient, direction),
        finite_difference,
        rtol=2e-8,
    )


def test_crystal_volume_preserves_end_to_end_position_gradients() -> None:
    """The public crystal-to-carrier wrapper keeps every position seam."""
    positions = jnp.array(
        [[0.71, 0.82, 0.93], [1.21, 1.12, 1.03]],
        dtype=jnp.float64,
    )
    crystal = create_crystal_data(
        positions,
        jnp.array([6, 14], dtype=jnp.int32),
        lattice=jnp.eye(3, dtype=jnp.float64) * 2.0,
    )
    weights = jnp.arange(512, dtype=jnp.float64).reshape(8, 8, 8)

    def loss(candidate_positions: jax.Array) -> jax.Array:
        candidate = eqx.tree_at(
            lambda value: value.positions,
            crystal,
            candidate_positions,
        )
        potential = crystal_potential_volume(
            candidate,
            0.25,
            (8, 8, 8),
            band_limit=1.5,
        )
        return jnp.mean(weights * potential.volume)

    gradient = jax.jit(jax.grad(loss))(positions)

    assert gradient.shape == positions.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(jnp.abs(gradient) > 0.0)


def test_crystal_structure_input_and_parameterization_contract() -> None:
    """Both crystal carriers work and only explicit Kirkland selects it."""
    frac = jnp.array([[0.25, 0.25, 0.25, 6.0]])
    cart = jnp.array([[0.5, 0.5, 0.5, 6.0]])
    crystal = create_crystal_structure(
        frac,
        cart,
        jnp.array([2.0, 2.0, 2.0]),
        jnp.array([90.0, 90.0, 90.0]),
    )
    default = crystal_potential_volume(
        crystal,
        0.5,
        band_limit=0.75,
    )
    lobato = crystal_potential_volume(
        crystal,
        0.5,
        band_limit=0.75,
        parameterization="lobato",
    )
    kirkland = crystal_potential_volume(
        crystal,
        0.5,
        band_limit=0.75,
        parameterization="kirkland",
    )

    assert jnp.array_equal(default.volume, lobato.volume)
    assert not jnp.array_equal(default.volume, kirkland.volume)
    with pytest.raises(ValueError, match="parameterization"):
        crystal_potential_volume(
            crystal,
            0.5,
            band_limit=0.75,
            parameterization="invalid",
        )


def test_volume_builder_has_no_projection_or_slice_reduction() -> None:
    """The volumetric public builder contains no spatial-axis reduction."""
    source = inspect.getsource(crystal_potential_volume)
    assert "jnp.sum" not in source
    assert "axis=" not in source


@pytest.mark.parametrize("atom_no", [0, 104])
def test_single_atom_volume_rejects_out_of_table_species(atom_no: int) -> None:
    """Invalid atomic numbers fail rather than clipping into the table."""
    with pytest.raises(Exception, match=r"inclusive range \[1, 103\]"):
        volume = single_atom_potential_3d(atom_no, 0.5, (4, 4, 4))
        jax.block_until_ready(volume)


def test_volume_apis_reject_boolean_atomic_numbers() -> None:
    """Booleans are not silently interpreted as hydrogen atomic numbers."""
    with pytest.raises(ValueError, match="integer"):
        single_atom_potential_3d(True, 0.5, (4, 4, 4))

    valid_crystal = create_crystal_data(
        jnp.array([[0.5, 0.5, 0.5]], dtype=jnp.float64),
        jnp.array([1], dtype=jnp.int32),
        lattice=jnp.eye(3, dtype=jnp.float64) * 2.0,
    )
    invalid_crystal: CrystalData = eqx.tree_at(
        lambda crystal: crystal.atomic_numbers,
        valid_crystal,
        jnp.array([True]),
    )
    with pytest.raises(ValueError, match="integer"):
        crystal_potential_volume(
            invalid_crystal,
            0.5,
            (4, 4, 4),
            band_limit=0.75,
        )


@pytest.mark.parametrize(
    ("voxel_size", "origin", "message"),
    [
        (True, (0.0, 0.0, 0.0), "voxel_size"),
        ((True, 0.5, 0.5), (0.0, 0.0, 0.0), "voxel_size"),
        (0.5, (False, 0.0, 0.0), "origin"),
    ],
)
def test_volume_builders_reject_boolean_geometry(
    voxel_size: object,
    origin: tuple[object, object, object],
    message: str,
) -> None:
    """Boolean geometry cannot silently become zero or one Angstrom."""
    with pytest.raises((ValueError, TypeCheckError), match=message):
        single_atom_potential_3d(
            6,
            voxel_size,
            (4, 4, 4),
            origin=origin,
        )
