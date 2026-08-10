r"""Build SC-1 interaction and absorber products on fixed supports.

Extended Summary
----------------
This module converts voltage Fourier coefficients to the real SC-1
interaction and applies endpoint-safe Galerkin multiplier products. It also
provides a bounded dense diagnostic for one compressed absorber. Raw profile
samples are not treated as continuous-profile coefficients.

Routine Listings
----------------
:func:`apply_absorber_action`
    Apply the endpoint-safe fixed-support absorber product.
:func:`apply_interaction_product`
    Apply the endpoint-safe fixed-support interaction product.
:func:`build_absorber_factor`
    Build a bounded dense diagnostic factor for one absorber compression.
:func:`build_cosine_shell_absorber_coefficients`
    Build analytic coefficients of the bounded periodic shell absorber.
:func:`build_interaction_coefficients`
    Build SC-1 interaction coefficients from voltage coefficients.

Notes
-----
The interaction uses the positive SC-1 sign
``chi = sigma_H(voltage_kv) * phi``. All multiplier coefficients use the
normalization in SC.13b. The bounded absorber factor satisfies
``G.conj().T @ G = A`` up to input-dtype floating-point rounding. It can
falsify a candidate
compression, but it is not an exact positivity or stability proof.
"""

import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import (
    Array,
    Bool,
    Complex,
    Complex128,
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from ptyrodactyl._numeric import (
    has_lost_nonzero_components,
    has_subnormal_components,
)
from ptyrodactyl._physics import coupled_interaction_value
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    M_E,
    GalerkinProductSupport,
    scalar_num,
)

_MAX_DENSE_VALIDATION_SIZE: int = 32


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` when a structural condition is true.

    Parameters
    ----------
    condition : bool
        Structural condition that triggers rejection when true.
    message : str
        Error message for the rejected condition.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _checked_multiplier_coefficients(
    indices: Int[Array, "p 3"],
    coefficients: Complex[Array, " p"],
    name: str,
) -> Complex[Array, " p"]:
    """PRIVATE: Attach finite and exact Hermitian-symmetry checks.

    Parameters
    ----------
    indices : Int[Array, "p 3"]
        Reciprocal multiplier indices in fixed coefficient order.
    coefficients : Complex[Array, " p"]
        Candidate multiplier coefficients.
    name : str
        Multiplier name used in the traced error message.

    Returns
    -------
    checked : Complex[Array, " p"]
        Coefficients with traced range and Hermitian checks attached.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If coefficients are non-finite, subnormal, or not exactly Hermitian.

    Notes
    -----
    The inverse-index comparison uses lexicographic order and exact equality.
    """
    inverse_indices: Int[Array, "p 3"] = -indices
    forward_order: Int[Array, " p"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    inverse_order: Int[Array, " p"] = jnp.lexsort(
        (
            inverse_indices[:, 2],
            inverse_indices[:, 1],
            inverse_indices[:, 0],
        )
    )
    nonhermitian: Bool[Array, ""] = jnp.any(
        indices[forward_order] != inverse_indices[inverse_order]
    ) | jnp.any(
        coefficients[forward_order] != jnp.conj(coefficients[inverse_order])
    )
    checked: Complex[Array, " p"] = eqx.error_if(
        coefficients,
        jnp.any(~jnp.isfinite(coefficients))
        | has_subnormal_components(coefficients)
        | nonhermitian,
        f"{name} must be finite, normal-range, and exactly Hermitian",
    )
    return checked


def _flat_residues(
    indices: Int[Array, "n 3"],
    work_shape: Tuple[int, int, int],
) -> Int[Array, " n"]:
    """PRIVATE: Map exact indices to flat row-major work-grid positions.

    Parameters
    ----------
    indices : Int[Array, "n 3"]
        Signed reciprocal indices in ``(x, y, z)`` axis order.
    work_shape : Tuple[int, int, int]
        Positive endpoint-safe work-grid shape in ``(x, y, z)`` order.

    Returns
    -------
    flat : Int[Array, " n"]
        Periodic row-major positions in the flattened work grid.

    Notes
    -----
    Modular residues implement periodic index embedding without endpoint
    aliasing on the validated work shape.
    """
    moduli: Int[Array, " 3"] = jnp.asarray(work_shape, dtype=indices.dtype)
    residues: Int[Array, "n 3"] = jnp.mod(indices, moduli)
    flat: Int[Array, " n"] = (
        residues[:, 0] * work_shape[1] + residues[:, 1]
    ) * work_shape[2] + residues[:, 2]
    return flat


def _cosine_shell_coefficients(
    indices: Int64[Array, "q 3"],
) -> Complex128[Array, " q"]:
    """PRIVATE: Evaluate analytic cosine-shell coefficients independently.

    Parameters
    ----------
    indices : Int64[Array, "q 3"]
        Absorber reciprocal indices in ``(x, y, z)`` order.

    Returns
    -------
    coefficients : Complex128[Array, " q"]
        Analytic shell coefficients in the supplied index order.

    Notes
    -----
    The separable interior profile has one-dimensional coefficients
    ``(1/4, 1/2, 1/4)`` on modes ``(-1, 0, 1)``. The shell is one minus that
    profile.
    """
    axis_coefficients: Float64[Array, "q 3"] = jnp.where(
        indices == 0,
        0.5,
        jnp.where(jnp.abs(indices) == 1, 0.25, 0.0),
    )
    interior_coefficients: Float64[Array, " q"] = jnp.prod(
        axis_coefficients, axis=-1
    )
    zero_mode: Bool[Array, " q"] = jnp.all(indices == 0, axis=-1)
    coefficients: Complex128[Array, " q"] = jnp.where(
        zero_mode,
        1.0 - interior_coefficients,
        -interior_coefficients,
    ).astype(jnp.complex128)
    return coefficients


def _has_complete_cosine_shell_support(
    indices: Int64[Array, "q 3"],
) -> Bool[Array, ""]:
    """PRIVATE: Determine whether the absorber contains all 27 shell modes.

    Parameters
    ----------
    indices : Int64[Array, "q 3"]
        Absorber reciprocal indices in ``(x, y, z)`` order.

    Returns
    -------
    complete : Bool[Array, ""]
        True when every index in ``{-1, 0, 1}^3`` is present.

    Notes
    -----
    Extra absorber modes do not affect this completeness test.
    """
    axis: Int64[Array, " 3"] = jnp.asarray((-1, 0, 1), dtype=jnp.int64)
    mesh = jnp.meshgrid(axis, axis, axis, indexing="ij")
    required: Int64[Array, "27 3"] = jnp.stack(mesh, axis=-1).reshape((27, 3))
    matches: Bool[Array, "27 q"] = jnp.all(
        required[:, None, :] == indices[None, :, :], axis=-1
    )
    complete: Bool[Array, ""] = jnp.all(jnp.any(matches, axis=1))
    return complete


def _compressed_absorber(
    support: GalerkinProductSupport,
    absorber_coefficients: Complex[Array, " q"],
) -> Complex[Array, "n n"]:
    """PRIVATE: Assemble the exact direct-branch absorber compression.

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed state and absorber supports.
    absorber_coefficients : Complex[Array, " q"]
        Absorber multiplier coefficients in support order.

    Returns
    -------
    absorber : Complex[Array, "n n"]
        Dense state-space absorber compression.

    Notes
    -----
    Entry ``(i, j)`` is the coefficient at the exact state-index difference
    ``k_i - k_j`` or zero when that mode is absent.
    """
    differences: Int[Array, "n n 3"] = (
        support.state_indices[:, None, :] - support.state_indices[None, :, :]
    )
    coefficient_matches: Bool[Array, "n n q"] = jnp.all(
        differences[:, :, None, :]
        == support.absorber_indices[None, None, :, :],
        axis=-1,
    )
    absorber: Complex[Array, "n n"] = jnp.sum(
        jnp.where(
            coefficient_matches,
            absorber_coefficients[None, None, :],
            0.0,
        ),
        axis=-1,
    )
    return absorber


def _apply_multiplier_product(
    support: GalerkinProductSupport,
    multiplier_indices: Int[Array, "p 3"],
    multiplier_coefficients: Complex[Array, " p"],
    field: Complex[Array, " n"],
) -> Complex128[Array, " n"]:
    """PRIVATE: Apply one validated unitary-DFT multiplier product.

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed state support and endpoint-safe work quotient.
    multiplier_indices : Int[Array, "p 3"]
        Multiplier indices in ``(x, y, z)`` order.
    multiplier_coefficients : Complex[Array, " p"]
        Multiplier coefficients in the supplied index order.
    field : Complex[Array, " n"]
        State coefficients in fixed support order.

    Returns
    -------
    product : Complex128[Array, " n"]
        Retained coefficients of the multiplier-field product.

    Notes
    -----
    Unitary FFT normalization requires the explicit square-root work-size
    factor on the multiplier grid.
    """
    work_size: int = math.prod(support.work_shape)
    state_positions: Int[Array, " n"] = _flat_residues(
        support.state_indices,
        support.work_shape,
    )
    multiplier_positions: Int[Array, " p"] = _flat_residues(
        multiplier_indices,
        support.work_shape,
    )
    dtype: jnp.dtype = jnp.result_type(
        multiplier_coefficients.dtype,
        field.dtype,
    )
    embedded_state: Complex[Array, " work"] = (
        jnp.zeros((work_size,), dtype=dtype).at[state_positions].set(field)
    )
    embedded_multiplier: Complex[Array, " work"] = (
        jnp.zeros((work_size,), dtype=dtype)
        .at[multiplier_positions]
        .set(multiplier_coefficients)
    )
    state_grid: Complex[Array, "nw0 nw1 nw2"] = jnp.fft.ifftn(
        embedded_state.reshape(support.work_shape),
        norm="ortho",
    )
    multiplier_grid: Complex128[Array, "nw0 nw1 nw2"] = jnp.sqrt(
        jnp.asarray(work_size, dtype=jnp.float64)
    ) * jnp.fft.ifftn(
        embedded_multiplier.reshape(support.work_shape),
        norm="ortho",
    )
    product_coefficients: Complex128[Array, "nw0 nw1 nw2"] = jnp.fft.fftn(
        multiplier_grid * state_grid,
        norm="ortho",
    )
    product: Complex128[Array, " n"] = product_coefficients.reshape(
        (work_size,)
    )[state_positions]
    return product


@jaxtyped(typechecker=beartype)
def build_interaction_coefficients(
    support: GalerkinProductSupport,
    voltage_coefficients: Complex[Array, "..."],
    voltage_kv: scalar_num,
) -> Complex128[Array, " p"]:
    r"""Build SC-1 interaction coefficients from voltage coefficients.

    :see: :class:`~.test_potential.TestScalarPotentialProducts`

    Implementation Logic
    --------------------
    1. Validate SC.13b voltage coefficients on the interaction support.
    2. Compute the positive relativistic Helmholtz coupling from voltage.
    3. Multiply the voltage coefficients by that coupling.

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed independent integer supports and endpoint-safe work grid.
    voltage_coefficients : Complex[Array, "..."]
        Electrostatic-potential multiplier coefficients in volts. Their order
        matches ``support.interaction_indices``.
    voltage_kv : scalar_num
        Positive accelerating voltage in kilovolts.

    Returns
    -------
    interaction_coefficients : Complex128[Array, " p"]
        Canonical binary64-complex interaction coefficients in inverse-square
        Angstroms.

    Raises
    ------
    ValueError
        If the coefficient vector or voltage scalar has invalid structure.
    equinox.EquinoxRuntimeError
        If a coefficient or voltage is non-finite, the voltage is not
        positive, or Hermitian symmetry is absent during traced execution.

    Notes
    -----
    The formula is ``chi = sigma_H * phi`` with positive ``sigma_H``. A
    positive electrostatic potential therefore increases the local squared
    wavenumber in the SC-1 convention.

    See Also
    --------
    :func:`ptyrodactyl.types.helmholtz_coupling`
        Compute the voltage-dependent coupling used by this builder.
    """
    coefficient_array: Complex[Array, " p"] = jnp.asarray(voltage_coefficients)
    _raise_if(
        coefficient_array.ndim != 1,
        "voltage_coefficients must be 1D",
    )
    _raise_if(
        coefficient_array.shape[0] != support.interaction_indices.shape[0],
        "voltage_coefficients must match the interaction support",
    )
    voltage_array: Float64[Array, ""] = jnp.asarray(
        voltage_kv,
        dtype=jnp.float64,
    )
    _raise_if(voltage_array.shape != (), "voltage_kv must be a scalar")
    checked_voltage: Float64[Array, ""] = eqx.error_if(
        voltage_array,
        (~jnp.isfinite(voltage_array)) | (voltage_array <= 0.0),
        "voltage_kv must be finite and positive",
    )
    checked_coefficients: Complex[Array, " p"] = (
        _checked_multiplier_coefficients(
            support.interaction_indices,
            coefficient_array,
            "voltage_coefficients",
        )
    )
    _, raw_interaction_coefficients = coupled_interaction_value(
        coefficient_array,
        voltage_array,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    interaction_coefficients: Complex128[Array, " p"] = eqx.error_if(
        raw_interaction_coefficients,
        jnp.any(~jnp.isfinite(raw_interaction_coefficients))
        | jnp.any(~jnp.isfinite(checked_coefficients))
        | (~jnp.isfinite(checked_voltage))
        | has_subnormal_components(coefficient_array)
        | has_lost_nonzero_components(
            coefficient_array, raw_interaction_coefficients
        ),
        "derived interaction_coefficients must be finite and preserve every "
        "nonzero normal voltage component",
    )
    return interaction_coefficients


@jaxtyped(typechecker=beartype)
def build_cosine_shell_absorber_coefficients(
    support: GalerkinProductSupport,
) -> Complex128[Array, " q"]:
    r"""Build analytic coefficients of the bounded periodic shell absorber.

    :see: :class:`~.test_potential.TestScalarPotentialProducts`

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed support whose absorber band contains the 27 profile modes.

    Returns
    -------
    absorber_coefficients : Complex128[Array, " q"]
        Exact binary64-complex SC.13b coefficients of the declared
        cosine-shell profile.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the absorber support omits an analytic profile mode.

    Notes
    -----
    On the centered periodic box, the profile is
    ``a(x) = 1 - product_j cos(pi x_j / L_j)^2``. It is analytic,
    dimensionless, lies in ``[0, 1]``, vanishes at the box center, and equals
    one whenever any coordinate is on its boundary. Its only nonzero modes
    lie in ``{-1, 0, 1}^3``; coefficients on extra ``K_a`` modes are zero.
    """
    coefficients: Complex128[Array, " q"] = _cosine_shell_coefficients(
        support.absorber_indices
    )
    absorber_coefficients: Complex128[Array, " q"] = eqx.error_if(
        coefficients,
        ~_has_complete_cosine_shell_support(support.absorber_indices),
        "absorber support must contain all cosine-shell profile modes",
    )
    return absorber_coefficients


@jaxtyped(typechecker=beartype)
def apply_interaction_product(
    support: GalerkinProductSupport,
    interaction_coefficients: Complex[Array, "..."],
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    r"""Apply the endpoint-safe fixed-support interaction product.

    :see: :class:`~.test_potential.TestScalarPotentialProducts`

    Implementation Logic
    --------------------
    1. Zero-embed state and multiplier coefficients in the work quotient.
    2. Use the unitary positive-phase synthesis transform on both arrays.
    3. Multiply on the work grid, transform back, and restrict to the state.

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed supports whose factory proved the restricted no-alias property.
    interaction_coefficients : Complex[Array, "..."]
        SC.13b interaction coefficients in inverse-square Angstroms.
    field : Complex[Array, "..."]
        Retained orthonormal state coefficients.

    Returns
    -------
    interaction : Complex128[Array, " n"]
        Binary64-complex compressed interaction action in inverse-square
        Angstroms times field.

    Raises
    ------
    ValueError
        If a coefficient or field vector has invalid structure.
    equinox.EquinoxRuntimeError
        If either vector is non-finite or the interaction is not Hermitian
        during traced execution.

    Notes
    -----
    The unitary DFT normalization includes the required square-root work-size
    factor. The result equals ``sum_h c_chi(g-h) field[h]`` without circular
    endpoint aliasing.
    """
    coefficient_array: Complex[Array, " p"] = jnp.asarray(
        interaction_coefficients
    )
    field_array: Complex[Array, " n"] = jnp.asarray(field)
    _raise_if(
        coefficient_array.ndim != 1,
        "interaction_coefficients must be 1D",
    )
    _raise_if(
        coefficient_array.shape[0] != support.interaction_indices.shape[0],
        "interaction_coefficients must match the interaction support",
    )
    _raise_if(field_array.ndim != 1, "field must be 1D")
    _raise_if(
        field_array.shape[0] != support.state_indices.shape[0],
        "field must match the state support",
    )
    checked_coefficients: Complex[Array, " p"] = (
        _checked_multiplier_coefficients(
            support.interaction_indices,
            coefficient_array,
            "interaction_coefficients",
        )
    )
    checked_field: Complex[Array, " n"] = eqx.error_if(
        field_array,
        jnp.any(~jnp.isfinite(field_array))
        | has_subnormal_components(field_array),
        "field must be finite and contain no nonzero subnormal components",
    )

    raw_interaction: Complex128[Array, " n"] = _apply_multiplier_product(
        support,
        support.interaction_indices,
        checked_coefficients,
        checked_field,
    )
    interaction: Complex128[Array, " n"] = eqx.error_if(
        raw_interaction,
        jnp.any(~jnp.isfinite(raw_interaction))
        | has_subnormal_components(raw_interaction),
        "interaction action must be finite and contain no nonzero subnormal "
        "components",
    )
    return interaction


@jaxtyped(typechecker=beartype)
def build_absorber_factor(
    support: GalerkinProductSupport,
    absorber_coefficients: Complex[Array, "..."],
) -> Complex[Array, "n n"]:
    r"""Build a bounded dense diagnostic factor for one absorber compression.

    :see: :class:`~.test_potential.TestScalarPotentialProducts`

    Implementation Logic
    --------------------
    1. Validate exact SC.13b coefficients on the absorber support.
    2. Compress them directly as ``A[g, h] = c_a(g - h)``.
    3. Require positive definiteness and return a Cholesky-derived factor.

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed supports with ``K_u - K_u`` contained in ``K_a``.
    absorber_coefficients : Complex[Array, "..."]
        Dimensionless continuous-profile coefficients in SC.13b
        normalization. Their order matches ``support.absorber_indices``.

    Returns
    -------
    absorber_factor : Complex[Array, "n n"]
        Dense factor ``G`` satisfying ``G.conj().T @ G = A``.

    Raises
    ------
    ValueError
        If the coefficient vector is invalid or the state exceeds 32 modes.
    equinox.EquinoxRuntimeError
        If coefficients are non-finite or non-Hermitian, or if their exact
        compression is not positive definite during traced execution.

    Notes
    -----
    This helper is limited to 32 state modes and uses eigendecomposition and
    Cholesky factorization in the input coefficient dtype. It is validation
    and falsification evidence, not an exact positivity certificate or
    per-result stability proof. Production actions use
    :func:`apply_absorber_action` and do not store this square factor. This
    function does not infer continuous-profile coefficients from raw samples.
    The factor is dimensionless; the physical CAP scale remains separate.
    """
    coefficient_array: Complex[Array, " q"] = jnp.asarray(
        absorber_coefficients
    )
    _raise_if(
        support.state_indices.shape[0] > _MAX_DENSE_VALIDATION_SIZE,
        "dense absorber validation is limited to 32 state modes",
    )
    _raise_if(
        coefficient_array.ndim != 1,
        "absorber_coefficients must be 1D",
    )
    _raise_if(
        coefficient_array.shape[0] != support.absorber_indices.shape[0],
        "absorber_coefficients must match the absorber support",
    )
    checked_coefficients: Complex[Array, " q"] = (
        _checked_multiplier_coefficients(
            support.absorber_indices,
            coefficient_array,
            "absorber_coefficients",
        )
    )
    absorber: Complex[Array, "n n"] = _compressed_absorber(
        support,
        checked_coefficients,
    )
    eigenvalues: Float[Array, " n"] = jnp.linalg.eigvalsh(absorber)
    checked_absorber: Complex[Array, "n n"] = eqx.error_if(
        absorber,
        jnp.any(~jnp.isfinite(eigenvalues)) | (jnp.min(eigenvalues) <= 0.0),
        "compressed absorber must be finite and positive definite",
    )
    lower_factor: Complex[Array, "n n"] = jnp.linalg.cholesky(checked_absorber)
    absorber_factor: Complex[Array, "n n"] = jnp.conj(lower_factor.T)
    return absorber_factor


@jaxtyped(typechecker=beartype)
def apply_absorber_action(
    support: GalerkinProductSupport,
    absorber_coefficients: Complex[Array, "..."],
    field: Complex[Array, "..."],
) -> Complex128[Array, " n"]:
    r"""Apply the endpoint-safe fixed-support absorber product.

    :see: :class:`~.test_potential.TestScalarPotentialProducts`

    Parameters
    ----------
    support : GalerkinProductSupport
        Fixed supports whose factory proved absorber product no-aliasing.
    absorber_coefficients : Complex[Array, "..."]
        Exact dimensionless SC.13b coefficients ordered as ``K_a``.
    field : Complex[Array, "..."]
        Retained state coefficients with shape ``(n,)``.

    Returns
    -------
    absorber : Complex128[Array, " n"]
        Binary64-complex compressed absorber action
        ``P_u M_a P_u field``.

    Raises
    ------
    ValueError
        If the coefficient and field shapes are incompatible.
    equinox.EquinoxRuntimeError
        If a coefficient or field is non-finite or non-Hermitian during
        tracing.

    Notes
    -----
    This action validates shape, finiteness, and Hermitian symmetry. It does
    not establish positivity for arbitrary input coefficients. Production
    manifests derive the fixed cosine-shell profile and record its provenance.
    The exact bounded stability invocation then attempts a compression-floor
    proof for each retained result and fails closed. The
    :func:`build_absorber_factor` helper supplies only bounded floating-point
    validation and falsification evidence.
    """
    coefficient_array: Complex[Array, " q"] = jnp.asarray(
        absorber_coefficients
    )
    field_array: Complex[Array, " n"] = jnp.asarray(field)
    _raise_if(
        coefficient_array.ndim != 1,
        "absorber_coefficients must be 1D",
    )
    _raise_if(
        coefficient_array.shape[0] != support.absorber_indices.shape[0],
        "absorber_coefficients must match the absorber support",
    )
    _raise_if(field_array.ndim != 1, "field must be 1D")
    _raise_if(
        field_array.shape[0] != support.state_indices.shape[0],
        "field must match the state support",
    )
    checked_coefficients: Complex[Array, " q"] = (
        _checked_multiplier_coefficients(
            support.absorber_indices,
            coefficient_array,
            "absorber_coefficients",
        )
    )
    checked_field: Complex[Array, " n"] = eqx.error_if(
        field_array,
        jnp.any(~jnp.isfinite(field_array))
        | has_subnormal_components(field_array),
        "field must be finite and contain no nonzero subnormal components",
    )
    raw_absorber: Complex128[Array, " n"] = _apply_multiplier_product(
        support,
        support.absorber_indices,
        checked_coefficients,
        checked_field,
    )
    absorber: Complex128[Array, " n"] = eqx.error_if(
        raw_absorber,
        jnp.any(~jnp.isfinite(raw_absorber))
        | has_subnormal_components(raw_absorber),
        "absorber action must be finite and contain no nonzero subnormal "
        "components",
    )
    return absorber


__all__: list[str] = [
    "apply_absorber_action",
    "apply_interaction_product",
    "build_absorber_factor",
    "build_cosine_shell_absorber_coefficients",
    "build_interaction_coefficients",
]
