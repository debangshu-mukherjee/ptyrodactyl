r"""Realize a voxelized voltage field on one scalar Galerkin support.

Extended Summary
----------------
This module implements the VC-1 map from a periodic :class:`Potential3D` to
SC.13b voltage coefficients. It preserves the zero mode, applies the physical
origin phase, rejects signed Nyquist aliasing, and records coefficient and
potential-band errors separately.

Routine Listings
----------------
:func:`apply_galerkin_potential_metric_adjoint`
    Apply the VC-1 coefficient map adjoint in the physical voxel metric.
:func:`realize_galerkin_potential`
    Realize a periodic voxel potential on one interaction support.

Notes
-----
The coefficient-error route in this module uses a conservative triangle
bound. It is independent of the FFT backend but can be too large for a useful
RM-S2 perturbation margin. A tighter verified route can replace it without
changing the exact VC-1 target.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float64,
    Int64,
    jaxtyped,
)

from ptyrodactyl._interval import (
    _interval_add,
    _interval_divide_positive,
    _interval_multiply,
    _interval_sqrt,
    _interval_square,
    _point_interval,
    _RealInterval,
)
from ptyrodactyl.types import (
    GalerkinAcquisitionSupportResult,
    GalerkinAcquisitionSupportStatus,
    GalerkinPotentialErrorRoute,
    GalerkinPotentialRealization,
    GalerkinPotentialRealizationMethod,
    GalerkinProductSupport,
    Potential3D,
    create_galerkin_potential_realization,
)

from .acquisition import check_galerkin_acquisition_support

_ENDPOINT_CONVENTION: str = "vc1_signed_half_open_no_even_nyquist"
_OUTPUT_NORMALIZATION: str = "SC.13b_mean_DFT_with_physical_origin_phase"
_VOXEL_METRIC: str = "cell_volume_weighted_real_L2"
_SPACE_DIMENSIONS: int = 3


def _canonical_checked_acquisition_support(
    submitted: GalerkinAcquisitionSupportResult,
) -> GalerkinAcquisitionSupportResult:
    """PRIVATE: Recheck and exactly match one acquisition-support artifact.

    Parameters
    ----------
    submitted : GalerkinAcquisitionSupportResult
        Caller-supplied checked-support carrier.

    Returns
    -------
    checked : GalerkinAcquisitionSupportResult
        Fresh checker output with fail-closed equality and eligibility guards.

    Raises
    ------
    ValueError
        If static structure, leaf shapes, or leaf dtypes differ from the fresh
        checker output.
    equinox.EquinoxRuntimeError
        If any dynamic leaf differs or the fresh result is not eligible.

    Notes
    -----
    Checking only the submitted status would permit a forged aggregate. This
    comparison covers the manifest, every result leaf, and every static field.
    """
    canonical: GalerkinAcquisitionSupportResult = (
        check_galerkin_acquisition_support(submitted.manifest)
    )
    submitted_leaves, submitted_structure = jax.tree_util.tree_flatten(
        submitted
    )
    canonical_leaves, canonical_structure = jax.tree_util.tree_flatten(
        canonical
    )
    if submitted_structure != canonical_structure:
        raise ValueError(
            "submitted acquisition-support artifact has noncanonical static "
            "structure"
        )
    if len(submitted_leaves) != len(canonical_leaves):
        raise ValueError(
            "submitted acquisition-support artifact has noncanonical leaves"
        )
    exact_match: Bool[Array, ""] = jnp.asarray(True)
    for submitted_leaf, canonical_leaf in zip(
        submitted_leaves,
        canonical_leaves,
        strict=True,
    ):
        submitted_array = jnp.asarray(submitted_leaf)
        canonical_array = jnp.asarray(canonical_leaf)
        if (
            submitted_array.shape != canonical_array.shape
            or submitted_array.dtype != canonical_array.dtype
        ):
            raise ValueError(
                "submitted acquisition-support artifact has noncanonical "
                "leaf shape or dtype"
            )
        exact_match = exact_match & jnp.all(submitted_array == canonical_array)
    eligible: Bool[Array, ""] = (
        canonical.status
        == int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
    ) & canonical.support_eligible
    checked_status = eqx.error_if(
        canonical.status,
        (~exact_match) | (~eligible),
        "acquisition-support artifact must exactly match an independently "
        "rechecked SUPPORT_ELIGIBLE result",
    )
    checked: GalerkinAcquisitionSupportResult = eqx.tree_at(
        lambda result: result.status,
        canonical,
        checked_status,
    )
    return checked


def _outward_add(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose one exact-point nonnegative sum from above.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left exact stored nonnegative binary64 point.
    right : Float64[Array, "..."]
        Right exact stored nonnegative binary64 point.

    Returns
    -------
    result : Float64[Array, "..."]
        FTZ-safe upper endpoint for the exact-real sum.

    Notes
    -----
    Exact points first pass through the shared FTZ.2 embedding.  Certificate
    arithmetic is stopped before evaluation and has no evidence tangent.
    """
    result: Float64[Array, "..."] = _interval_add(
        _point_interval(left),
        _point_interval(right),
    )[1]
    return result


def _outward_multiply(
    left: Float64[Array, "..."],
    right: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose one exact-point nonnegative product from above.

    Parameters
    ----------
    left : Float64[Array, "..."]
        Left exact stored nonnegative binary64 point.
    right : Float64[Array, "..."]
        Right exact stored nonnegative binary64 point.

    Returns
    -------
    result : Float64[Array, "..."]
        FTZ-safe upper endpoint for the exact-real product.

    Notes
    -----
    Exact points first pass through the shared FTZ.2 embedding.  Certificate
    arithmetic is stopped before evaluation and has no evidence tangent.
    """
    result: Float64[Array, "..."] = _interval_multiply(
        _point_interval(left),
        _point_interval(right),
    )[1]
    return result


def _outward_sqrt(
    value: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose one exact-point nonnegative square root from above.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Exact stored nonnegative binary64 radicand.

    Returns
    -------
    result : Float64[Array, "..."]
        FTZ-safe upper endpoint for the exact-real square root.

    Notes
    -----
    The exact point first passes through the shared FTZ.2 embedding.
    Certificate arithmetic is stopped before evaluation and has no evidence
    tangent.
    """
    result: Float64[Array, "..."] = _interval_sqrt(_point_interval(value))[1]
    return result


def _checked_interaction_indices(
    potential: Potential3D,
    support: GalerkinProductSupport,
) -> Int64[Array, "p 3"]:
    """PRIVATE: Check VC-1 support predicates.

    Parameters
    ----------
    potential : Potential3D
        Canonical periodic voltage volume and band limit.
    support : GalerkinProductSupport
        Candidate finite interaction support.

    Returns
    -------
    checked_band : Int64[Array, "p 3"]
        Interaction indices after zero-mode, endpoint, and strict-band checks.
    """
    indices: Int64[Array, "p 3"] = support.interaction_indices
    nz: int
    ny: int
    nx: int
    nz, ny, nx = potential.volume.shape
    grid_shape_xyz: Int64[Array, " 3"] = jnp.asarray(
        (nx, ny, nz),
        dtype=jnp.int64,
    )
    contains_zero: Bool[Array, ""] = jnp.any(jnp.all(indices == 0, axis=-1))
    checked_zero: Int64[Array, "p 3"] = eqx.error_if(
        indices,
        ~contains_zero,
        "interaction support must retain the voltage zero mode",
    )
    outside_signed_grid: Bool[Array, ""] = jnp.any(
        2 * jnp.abs(checked_zero) >= grid_shape_xyz[None, :]
    )
    checked_grid: Int64[Array, "p 3"] = eqx.error_if(
        checked_zero,
        outside_signed_grid,
        "interaction support must stay inside VC-1 signed grid endpoints",
    )
    box_size: Float64[Array, " 3"] = jnp.asarray(
        potential.box_size,
        dtype=jnp.float64,
    )
    absolute_indices: Float64[Array, "p 3"] = jnp.abs(checked_grid).astype(
        jnp.float64
    )
    frequency_intervals: _RealInterval = _interval_divide_positive(
        _point_interval(absolute_indices),
        _point_interval(box_size[None, :]),
    )
    squared_frequency_intervals: _RealInterval = _interval_square(
        frequency_intervals
    )
    squared_frequency_sum: _RealInterval = _interval_add(
        _interval_add(
            (
                squared_frequency_intervals[0][:, 0],
                squared_frequency_intervals[1][:, 0],
            ),
            (
                squared_frequency_intervals[0][:, 1],
                squared_frequency_intervals[1][:, 1],
            ),
        ),
        (
            squared_frequency_intervals[0][:, 2],
            squared_frequency_intervals[1][:, 2],
        ),
    )
    band_interval: _RealInterval = _interval_square(
        _point_interval(jnp.asarray(potential.band_limit, dtype=jnp.float64))
    )
    certified_strictly_inside: Bool[Array, " p"] = (
        squared_frequency_sum[1] < band_interval[0]
    )
    outside_band: Bool[Array, ""] = jnp.any(~certified_strictly_inside)
    checked_band: Int64[Array, "p 3"] = eqx.error_if(
        checked_grid,
        outside_band,
        "interaction support must stay inside the strict potential band",
    )
    return checked_band


def _canonical_pair_mask(
    indices: Int64[Array, "p 3"],
) -> Bool[Array, " p"]:
    """PRIVATE: Choose one canonical member of each signed index pair.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Signed reciprocal indices.

    Returns
    -------
    canonical : Bool[Array, " p"]
        Mask selecting zero and lexicographically positive indices.
    """
    first: Int64[Array, " p"] = indices[:, 0]
    second: Int64[Array, " p"] = indices[:, 1]
    third: Int64[Array, " p"] = indices[:, 2]
    canonical: Bool[Array, " p"] = (first > 0) | (
        (first == 0) & ((second > 0) | ((second == 0) & (third >= 0)))
    )
    return canonical


def _hermitian_coefficients(
    indices: Int64[Array, "p 3"],
    raw_coefficients: Complex128[Array, " p"],
) -> Complex128[Array, " p"]:
    """PRIVATE: Project coefficients onto stored Hermitian symmetry.

    Parameters
    ----------
    indices : Int64[Array, "p 3"]
        Sign-symmetric interaction indices.
    raw_coefficients : Complex128[Array, " p"]
        Rounded voltage coefficients in volts before pair averaging.

    Returns
    -------
    coefficients : Complex128[Array, " p"]
        Pair-averaged voltage coefficients in volts with exact stored
        conjugate symmetry.

    Notes
    -----
    Each signed pair is averaged once with its conjugate partner. The
    canonical member is stored directly and the opposite member is copied by
    exact conjugation, including the self-paired zero mode.
    """
    pair_matches: Bool[Array, "p p"] = jnp.all(
        indices[:, None, :] == -indices[None, :, :],
        axis=-1,
    )
    pair_positions: Int64[Array, " p"] = jnp.argmax(
        pair_matches,
        axis=1,
    ).astype(jnp.int64)
    pair_average: Complex128[Array, " p"] = 0.5 * (
        raw_coefficients + jnp.conj(raw_coefficients[pair_positions])
    )
    canonical: Bool[Array, " p"] = _canonical_pair_mask(indices)
    coefficients: Complex128[Array, " p"] = jnp.where(
        canonical,
        pair_average,
        jnp.conj(pair_average[pair_positions]),
    )
    return coefficients


def _vc1_voltage_coefficients_from_full_grid(
    full_coefficients: Complex128[Array, "nz ny nx"],
    indices: Int64[Array, "p 3"],
    box_size: Float64[Array, " 3"],
    origin: Float64[Array, " 3"],
) -> Complex128[Array, " p"]:
    """PRIVATE: Restrict one mean DFT to the ordered VC-1 support.

    Parameters
    ----------
    full_coefficients : Complex128[Array, "nz ny nx"]
        Mean-normalized DFT coefficients in storage-axis order.
    indices : Int64[Array, "p 3"]
        Signed interaction indices in physical ``(x, y, z)`` order.
    box_size : Float64[Array, " 3"]
        Authoritative periodic box lengths in Angstroms.
    origin : Float64[Array, " 3"]
        Physical coordinate of the first voxel sample in Angstroms.

    Returns
    -------
    coefficients : Complex128[Array, " p"]
        Origin-shifted, exactly stored-Hermitian VC-1 coefficients.

    Notes
    -----
    This is the shared differentiable algebraic core of VC-1.  Structural,
    endpoint, and acquisition checks remain the responsibility of
    :func:`realize_galerkin_potential` before this helper is called.
    """
    nz: int
    ny: int
    nx: int
    nz, ny, nx = full_coefficients.shape
    grid_shape_xyz: Int64[Array, " 3"] = jnp.asarray(
        (nx, ny, nz),
        dtype=jnp.int64,
    )
    residues_xyz: Int64[Array, "p 3"] = jnp.mod(
        indices,
        grid_shape_xyz[None, :],
    )
    selected: Complex128[Array, " p"] = full_coefficients[
        residues_xyz[:, 2],
        residues_xyz[:, 1],
        residues_xyz[:, 0],
    ]
    frequencies: Float64[Array, "p 3"] = indices / box_size[None, :]
    phase: Complex128[Array, " p"] = jnp.exp(
        -2.0j * jnp.pi * (frequencies @ origin)
    )
    raw_coefficients: Complex128[Array, " p"] = selected * phase
    coefficients: Complex128[Array, " p"] = _hermitian_coefficients(
        indices,
        raw_coefficients,
    )
    return coefficients


def _coefficient_error_bounds(
    potential: Potential3D,
    coefficients: Complex128[Array, " p"],
) -> Float64[Array, " p"]:
    """PRIVATE: Bound each realized coefficient by an outward triangle rule.

    Parameters
    ----------
    potential : Potential3D
        Source voltage volume in volts.
    coefficients : Complex128[Array, " p"]
        Realized voltage coefficients in volts.

    Returns
    -------
    bounds : Float64[Array, " p"]
        Backend-independent non-negative coefficient error bounds in volts.

    Notes
    -----
    Outward-rounded additions combine each coefficient's complex L1
    magnitude with the maximum voxel voltage, preserving a conservative
    componentwise bound.
    """
    voltage_interval: _RealInterval = _point_interval(potential.volume)
    voltage_magnitude_upper: Float64[Array, "nz ny nx"] = jnp.maximum(
        jnp.abs(voltage_interval[0]),
        jnp.abs(voltage_interval[1]),
    )
    maximum_voltage: Float64[Array, ""] = jnp.max(voltage_magnitude_upper)
    coefficient_l1: Float64[Array, " p"] = _outward_add(
        jnp.abs(jnp.real(coefficients)),
        jnp.abs(jnp.imag(coefficients)),
    )
    bounds: Float64[Array, " p"] = _outward_add(
        coefficient_l1,
        maximum_voltage,
    )
    return bounds


def _omitted_band_evidence(
    potential: Potential3D,
    indices: Int64[Array, "p 3"],
    full_coefficients: Complex128[Array, "nz ny nx"],
) -> Tuple[Float64[Array, ""], Float64[Array, ""]]:
    """PRIVATE: Bound voltage omitted outside the retained interaction band.

    Parameters
    ----------
    potential : Potential3D
        Source voltage volume and physical box metadata.
    indices : Int64[Array, "p 3"]
        Retained interaction indices.
    full_coefficients : Complex128[Array, "nz ny nx"]
        Full mean-normalized Fourier coefficient grid in volts.

    Returns
    -------
    diagnostic : Float64[Array, ""]
        Floating Parseval norm of omitted represented coefficients in volt
        Angstrom to the power three-halves.
    upper_bound : Float64[Array, ""]
        Outward total-norm bound in volt Angstrom to the power three-halves,
        derived from box volume and peak voltage.

    Notes
    -----
    Mean-normalized Fourier coefficients make the box-volume-weighted sum the
    discrete Parseval norm. Signed retained indices map to modular grid
    residues, so coefficients on the admitted band boundary stay retained and
    only the complementary band contributes to the diagnostic.
    """
    nz: int
    ny: int
    nx: int
    nz, ny, nx = potential.volume.shape
    grid_shape_xyz: Int64[Array, " 3"] = jnp.asarray(
        (nx, ny, nz),
        dtype=jnp.int64,
    )
    residues_xyz: Int64[Array, "p 3"] = jnp.mod(
        indices,
        grid_shape_xyz[None, :],
    )
    retained_mask: Bool[Array, "nz ny nx"] = (
        jnp.zeros(
            potential.volume.shape,
            dtype=jnp.bool_,
        )
        .at[
            residues_xyz[:, 2],
            residues_xyz[:, 1],
            residues_xyz[:, 0],
        ]
        .set(True)
    )
    omitted_coefficients: Complex128[Array, "nz ny nx"] = jnp.where(
        retained_mask,
        0.0 + 0.0j,
        full_coefficients,
    )
    box_volume: Float64[Array, ""] = jnp.asarray(
        math.prod(potential.box_size),
        dtype=jnp.float64,
    )
    diagnostic: Float64[Array, ""] = jnp.sqrt(
        box_volume * jnp.sum(jnp.abs(omitted_coefficients) ** 2)
    )

    length_x: Float64[Array, ""] = jnp.asarray(
        potential.box_size[0], dtype=jnp.float64
    )
    length_y: Float64[Array, ""] = jnp.asarray(
        potential.box_size[1], dtype=jnp.float64
    )
    length_z: Float64[Array, ""] = jnp.asarray(
        potential.box_size[2], dtype=jnp.float64
    )
    volume_xy: Float64[Array, ""] = _outward_multiply(length_x, length_y)
    volume_upper: Float64[Array, ""] = _outward_multiply(
        volume_xy,
        length_z,
    )
    root_volume_upper: Float64[Array, ""] = _outward_sqrt(volume_upper)
    voltage_interval: _RealInterval = _point_interval(potential.volume)
    voltage_magnitude_upper: Float64[Array, "nz ny nx"] = jnp.maximum(
        jnp.abs(voltage_interval[0]),
        jnp.abs(voltage_interval[1]),
    )
    maximum_voltage: Float64[Array, ""] = jnp.max(voltage_magnitude_upper)
    upper_bound: Float64[Array, ""] = _outward_multiply(
        root_volume_upper,
        maximum_voltage,
    )
    result: Tuple[Float64[Array, ""], Float64[Array, ""]] = (
        diagnostic,
        upper_bound,
    )
    return result


@jaxtyped(typechecker=beartype)
def realize_galerkin_potential(
    potential: Potential3D,
    support_eligibility: GalerkinAcquisitionSupportResult,
) -> GalerkinPotentialRealization:
    r"""Realize a periodic voxel potential on one interaction support.

    :see: :class:`~.test_realization.TestGalerkinPotentialRealization`

    Implementation Logic
    --------------------
    1. Check periodic, zero-mode, signed-endpoint, and strict-band predicates.
    2. Evaluate the mean-normalized DFT and the physical-origin phase.
    3. Store each signed pair through one canonical Hermitian representative.
    4. Return distinct coefficient, operator, and omitted-band evidence.

    Parameters
    ----------
    potential : Potential3D
        Periodic voltage samples in `(z, y, x)` order.
    support_eligibility : GalerkinAcquisitionSupportResult
        Submitted support artifact. Its manifest is independently rechecked;
        caller-supplied aggregate eligibility fields are never trusted.

    Returns
    -------
    realization : GalerkinPotentialRealization
        Bound potential, support, SC.13b coefficients, and error evidence.

    Raises
    ------
    ValueError
        If the static boundary convention is not exactly periodic.
    equinox.EquinoxRuntimeError
        If the support omits zero, reaches a signed grid endpoint, exceeds the
        strict potential band, or produces invalid numeric evidence.

    Notes
    -----
    The exact finite target is the VC-1 periodic trigonometric interpolant of
    the stored binary64 voxels. The returned triangle coefficient bound is
    sound without an FFT implementation assumption, but it can be vacuous.
    """
    if potential.boundary != "periodic":
        raise ValueError("VC-1 potential boundary must be exactly 'periodic'")
    canonical_eligibility: GalerkinAcquisitionSupportResult = (
        _canonical_checked_acquisition_support(support_eligibility)
    )
    support: GalerkinProductSupport = canonical_eligibility.manifest.support
    potential_box: Float64[Array, " 3"] = jnp.asarray(
        potential.box_size,
        dtype=jnp.float64,
    )
    indices: Int64[Array, "p 3"] = _checked_interaction_indices(
        potential,
        support,
    )
    indices = eqx.error_if(
        indices,
        (
            canonical_eligibility.status
            != int(GalerkinAcquisitionSupportStatus.SUPPORT_ELIGIBLE)
        )
        | (~canonical_eligibility.support_eligible),
        "acquisition support must be independently SUPPORT_ELIGIBLE",
    )
    indices = eqx.error_if(
        indices,
        jnp.any(canonical_eligibility.manifest.box_lengths != potential_box),
        "Potential3D box lengths must exactly match acquisition support",
    )
    nz: int
    ny: int
    nx: int
    nz, ny, nx = potential.volume.shape
    sample_count: int = nx * ny * nz
    full_coefficients: Complex128[Array, "nz ny nx"] = (
        jnp.fft.fftn(potential.volume) / sample_count
    )
    box_size: Float64[Array, " 3"] = jnp.asarray(
        potential.box_size,
        dtype=jnp.float64,
    )
    origin: Float64[Array, " 3"] = jnp.asarray(
        potential.origin,
        dtype=jnp.float64,
    )
    coefficients: Complex128[Array, " p"] = (
        _vc1_voltage_coefficients_from_full_grid(
            full_coefficients,
            indices,
            box_size,
            origin,
        )
    )
    coefficient_errors: Float64[Array, " p"] = jax.lax.stop_gradient(
        _coefficient_error_bounds(potential, coefficients)
    )
    state_size: Float64[Array, ""] = jnp.asarray(
        support.state_indices.shape[0],
        dtype=jnp.float64,
    )
    operator_error: Float64[Array, ""] = jax.lax.stop_gradient(
        _outward_multiply(
            state_size,
            jnp.max(coefficient_errors),
        )
    )
    omitted_diagnostic: Float64[Array, ""]
    omitted_upper: Float64[Array, ""]
    omitted_diagnostic, omitted_upper = jax.tree.map(
        jax.lax.stop_gradient,
        _omitted_band_evidence(
            potential,
            indices,
            full_coefficients,
        ),
    )
    realization: GalerkinPotentialRealization = (
        create_galerkin_potential_realization(
            potential=potential,
            support_eligibility=canonical_eligibility,
            voltage_coefficients=coefficients,
            coefficient_error_bounds=coefficient_errors,
            voltage_operator_error_bound=operator_error,
            omitted_voltage_l2_diagnostic=omitted_diagnostic,
            omitted_voltage_l2_upper_bound=omitted_upper,
            method=(GalerkinPotentialRealizationMethod.PERIODIC_TRIGONOMETRIC),
            error_route=GalerkinPotentialErrorRoute.TRIANGLE_FALLBACK,
            output_coefficient_normalization=_OUTPUT_NORMALIZATION,
            endpoint_convention=_ENDPOINT_CONVENTION,
            voxel_metric=_VOXEL_METRIC,
        )
    )
    return realization


@jaxtyped(typechecker=beartype)
def apply_galerkin_potential_metric_adjoint(
    realization: GalerkinPotentialRealization,
    coefficient_cotangent: Complex[Array, "..."],
) -> Float64[Array, "nz ny nx"]:
    r"""Apply the VC-1 coefficient map adjoint in the physical voxel metric.

    :see: :class:`~.test_realization.TestGalerkinPotentialRealization`

    Implementation Logic
    --------------------
    1. Apply the positive physical-origin phase to coefficient cotangents.
    2. Embed the fixed interaction support in `(z, y, x)` DFT-bin order.
    3. Synthesize with the inverse DFT and divide by the voxel volume.

    Parameters
    ----------
    realization : GalerkinPotentialRealization
        Fixed VC-1 source potential, support, and coefficient ordering.
    coefficient_cotangent : Complex[Array, "..."]
        Coefficient covector under the realified Euclidean coefficient metric.

    Returns
    -------
    voxel_gradient : Float64[Array, "nz ny nx"]
        Real gradient under the cell-volume-weighted voxel metric.

    Raises
    ------
    ValueError
        If the coefficient cotangent is not a vector of the fixed size.
    equinox.EquinoxRuntimeError
        If a cotangent or derived voxel gradient is non-finite.

    Notes
    -----
    This function implements VC.20. A Euclidean-coordinate voxel cotangent is
    the returned array multiplied by the voxel volume.
    """
    cotangent: Complex128[Array, " p"] = jnp.asarray(
        coefficient_cotangent,
        dtype=jnp.complex128,
    )
    if cotangent.ndim != 1:
        raise ValueError("coefficient_cotangent must be 1D")
    expected_size: int = realization.support.interaction_indices.shape[0]
    if cotangent.shape[0] != expected_size:
        raise ValueError(
            "coefficient_cotangent must match the interaction support"
        )
    checked_cotangent: Complex128[Array, " p"] = eqx.error_if(
        cotangent,
        jnp.any(~jnp.isfinite(cotangent)),
        "coefficient_cotangent must be finite",
    )
    potential: Potential3D = realization.potential
    indices: Int64[Array, "p 3"] = realization.support.interaction_indices
    nz: int
    ny: int
    nx: int
    nz, ny, nx = potential.volume.shape
    grid_shape_xyz: Int64[Array, " 3"] = jnp.asarray(
        (nx, ny, nz),
        dtype=jnp.int64,
    )
    residues_xyz: Int64[Array, "p 3"] = jnp.mod(
        indices,
        grid_shape_xyz[None, :],
    )
    box_size: Float64[Array, " 3"] = jnp.asarray(
        potential.box_size,
        dtype=jnp.float64,
    )
    origin: Float64[Array, " 3"] = jnp.asarray(
        potential.origin,
        dtype=jnp.float64,
    )
    frequencies: Float64[Array, "p 3"] = indices / box_size[None, :]
    origin_phase: Complex128[Array, " p"] = jnp.exp(
        2.0j * jnp.pi * (frequencies @ origin)
    )
    embedded: Complex128[Array, "nz ny nx"] = (
        jnp.zeros(
            potential.volume.shape,
            dtype=jnp.complex128,
        )
        .at[
            residues_xyz[:, 2],
            residues_xyz[:, 1],
            residues_xyz[:, 0],
        ]
        .add(checked_cotangent * origin_phase)
    )
    sample_count: int = nx * ny * nz
    voxel_volume: Float64[Array, ""] = jnp.asarray(
        math.prod(potential.box_size) / sample_count,
        dtype=jnp.float64,
    )
    raw_gradient: Float64[Array, "nz ny nx"] = (
        jnp.real(jnp.fft.ifftn(embedded)) / voxel_volume
    )
    voxel_gradient: Float64[Array, "nz ny nx"] = eqx.error_if(
        raw_gradient,
        jnp.any(~jnp.isfinite(raw_gradient)),
        "voxel metric gradient must be finite",
    )
    return voxel_gradient


__all__: list[str] = [
    "apply_galerkin_potential_metric_adjoint",
    "realize_galerkin_potential",
]
