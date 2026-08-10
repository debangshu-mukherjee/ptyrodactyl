r"""Define fixed Fourier supports for scalar Galerkin products.

Extended Summary
----------------
This module owns the integer reciprocal-index supports used by the scalar
Galerkin interaction and absorber builders. The four supports remain
independent. The validated work grid represents every required multiplier
product without modular endpoint aliasing.

Routine Listings
----------------
:class:`GalerkinProductSupport`
    Store independent supports for fixed scalar Galerkin products.
:func:`create_galerkin_product_support`
    Create validated supports for fixed scalar Galerkin products.

Notes
-----
Index components follow the axes of ``work_shape``. They are exact integer
reciprocal-lattice coordinates, not rounded physical frequencies.
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
    Complex128,
    Float64,
    Int,
    Int32,
    Int64,
    jaxtyped,
)

from .custom_types import scalar_int

_SPACE_DIMENSIONS: int = 3
_SUPPORT_RANK: int = 2


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a true structural condition.

    Parameters
    ----------
    condition : bool
        Whether the structural contract is invalid.
    message : str
        Error message for the rejected contract.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _contains_duplicates(indices: Int64[Array, "n 3"]) -> Bool[Array, ""]:
    """PRIVATE: Detect repeated indices in one integer support.

    Parameters
    ----------
    indices : Int64[Array, "n 3"]
        Exact three-dimensional reciprocal indices.

    Returns
    -------
    duplicate : Bool[Array, ""]
        Whether at least one index occurs more than once.
    """
    order: Int64[Array, " n"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    sorted_indices: Int64[Array, "n 3"] = indices[order]
    duplicate: Bool[Array, ""] = jnp.any(
        jnp.all(sorted_indices[1:] == sorted_indices[:-1], axis=-1)
    )
    return duplicate


def _is_sign_symmetric(indices: Int64[Array, "n 3"]) -> Bool[Array, ""]:
    """PRIVATE: Check whether every index has its additive inverse.

    Parameters
    ----------
    indices : Int64[Array, "n 3"]
        Exact three-dimensional reciprocal indices.

    Returns
    -------
    symmetric : Bool[Array, ""]
        Whether the support is invariant under sign reversal.
    """
    inverse_indices: Int64[Array, "n 3"] = -indices
    forward_order: Int64[Array, " n"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    inverse_order: Int64[Array, " n"] = jnp.lexsort(
        (
            inverse_indices[:, 2],
            inverse_indices[:, 1],
            inverse_indices[:, 0],
        )
    )
    symmetric: Bool[Array, ""] = jnp.all(
        indices[forward_order] == inverse_indices[inverse_order]
    )
    return symmetric


def _residues(
    indices: Int64[Array, "... 3"],
    work_shape: Tuple[int, int, int],
) -> Int64[Array, "... 3"]:
    """PRIVATE: Reduce integer indices into the work-grid quotient.

    Parameters
    ----------
    indices : Int64[Array, "... 3"]
        Exact three-dimensional reciprocal indices.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.

    Returns
    -------
    residues : Int64[Array, "... 3"]
        Non-negative componentwise residues on the work grid.
    """
    moduli: Int64[Array, " 3"] = jnp.asarray(work_shape, dtype=indices.dtype)
    residues: Int64[Array, "... 3"] = jnp.mod(indices, moduli)
    return residues


def _flat_residues(
    indices: Int64[Array, "... 3"],
    work_shape: Tuple[int, int, int],
) -> Int64[Array, "..."]:
    """PRIVATE: Map exact indices to flat work-quotient keys.

    Parameters
    ----------
    indices : Int64[Array, "... 3"]
        Exact three-dimensional reciprocal indices.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.

    Returns
    -------
    flat : Int64[Array, "..."]
        Row-major flat keys for the componentwise residues.

    Notes
    -----
    Validated work shapes keep the mixed-radix calculation within signed
    64-bit range.
    """
    residues: Int64[Array, "... 3"] = _residues(indices, work_shape)
    flat: Int64[Array, "..."] = (
        residues[..., 0] * work_shape[1] + residues[..., 1]
    ) * work_shape[2] + residues[..., 2]
    return flat


def _all_members(
    candidates: Int64[Array, "... 3"],
    support: Int64[Array, "n 3"],
    work_shape: Tuple[int, int, int],
) -> Bool[Array, ""]:
    """PRIVATE: Check whether every candidate occurs in the exact support.

    Parameters
    ----------
    candidates : Int64[Array, "... 3"]
        Exact reciprocal indices to locate.
    support : Int64[Array, "n 3"]
        Unique exact reciprocal indices admitted by the support.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.

    Returns
    -------
    contained : Bool[Array, ""]
        Whether every candidate has an exact support match.

    Notes
    -----
    Quotient keys accelerate the lookup. The final coordinate comparison
    prevents a modular collision from counting as membership.
    """
    flat_candidates: Int64[Array, "m 3"] = candidates.reshape(
        (-1, _SPACE_DIMENSIONS)
    )
    candidate_keys: Int64[Array, " m"] = _flat_residues(
        flat_candidates, work_shape
    )
    support_keys: Int64[Array, " n"] = _flat_residues(support, work_shape)
    order: Int64[Array, " n"] = jnp.argsort(support_keys)
    sorted_keys: Int64[Array, " n"] = support_keys[order]
    locations: Int32[Array, " m"] = jnp.searchsorted(
        sorted_keys, candidate_keys, side="left"
    )
    clipped: Int32[Array, " m"] = jnp.clip(locations, 0, support.shape[0] - 1)
    key_matches: Bool[Array, " m"] = (locations < support.shape[0]) & (
        sorted_keys[clipped] == candidate_keys
    )
    exact_matches: Bool[Array, " m"] = jnp.all(
        support[order[clipped]] == flat_candidates, axis=-1
    )
    contained: Bool[Array, ""] = jnp.all(key_matches & exact_matches)
    return contained


def _quotient_is_injective(
    indices: Int64[Array, "n 3"],
    work_shape: Tuple[int, int, int],
) -> Bool[Array, ""]:
    """PRIVATE: Check work-quotient injectivity on one support.

    Parameters
    ----------
    indices : Int64[Array, "n 3"]
        Exact reciprocal indices in one support.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.

    Returns
    -------
    injective : Bool[Array, ""]
        Whether distinct exact indices have distinct quotient keys.
    """
    keys: Int64[Array, " n"] = _flat_residues(indices, work_shape)
    sorted_keys: Int64[Array, " n"] = jnp.sort(keys)
    injective: Bool[Array, ""] = ~jnp.any(sorted_keys[1:] == sorted_keys[:-1])
    return injective


def _restricted_product_has_no_alias(
    state_indices: Int64[Array, "n 3"],
    multiplier_indices: Int64[Array, "p 3"],
    work_shape: Tuple[int, int, int],
) -> Bool[Array, ""]:
    """PRIVATE: Check the RM-S2 restricted-product no-alias criterion.

    Parameters
    ----------
    state_indices : Int64[Array, "n 3"]
        Exact retained-state indices.
    multiplier_indices : Int64[Array, "p 3"]
        Exact multiplier indices added to every retained-state index.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.

    Returns
    -------
    no_alias : Bool[Array, ""]
        Whether every quotient collision is also an exact state match.

    Notes
    -----
    The loop checks each restricted product without materializing the full
    pair grid.
    """
    state_keys: Int64[Array, " n"] = _flat_residues(state_indices, work_shape)
    order: Int64[Array, " n"] = jnp.argsort(state_keys)
    sorted_keys: Int64[Array, " n"] = state_keys[order]
    multiplier_count: int = multiplier_indices.shape[0]
    product_count: int = state_indices.shape[0] * multiplier_count

    def check_product(
        flat_position: scalar_int,
        valid: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Accumulate the restricted no-alias predicate without a pair grid."""
        state_position: scalar_int = flat_position // multiplier_count
        multiplier_position: scalar_int = flat_position % multiplier_count
        product: Int64[Array, " 3"] = (
            state_indices[state_position]
            + multiplier_indices[multiplier_position]
        )
        product_key: Int64[Array, ""] = _flat_residues(
            product[None, :], work_shape
        )[0]
        location: Int32[Array, ""] = jnp.searchsorted(
            sorted_keys, product_key, side="left"
        )
        clipped: Int32[Array, ""] = jnp.clip(
            location, 0, state_indices.shape[0] - 1
        )
        collision: Bool[Array, ""] = (location < state_indices.shape[0]) & (
            sorted_keys[clipped] == product_key
        )
        exact_match: Bool[Array, ""] = jnp.all(
            product == state_indices[order[clipped]]
        )
        updated: Bool[Array, ""] = valid & ~(collision & ~exact_match)
        return updated

    no_alias: Bool[Array, ""] = jax.lax.fori_loop(
        0,
        product_count,
        check_product,
        jnp.asarray(True),
    )
    return no_alias


def _all_binary_products_are_members(
    left_indices: Int64[Array, "n 3"],
    right_indices: Int64[Array, "p 3"],
    support: Int64[Array, "w 3"],
    work_shape: Tuple[int, int, int],
    right_sign: int,
) -> Bool[Array, ""]:
    """PRIVATE: Check binary support inclusion with bounded memory.

    Parameters
    ----------
    left_indices : Int64[Array, "n 3"]
        Exact indices forming the left product operand.
    right_indices : Int64[Array, "p 3"]
        Exact indices forming the right product operand.
    support : Int64[Array, "w 3"]
        Exact support that must contain every product index.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.
    right_sign : int
        Sign applied to every right index before addition.

    Returns
    -------
    contained : Bool[Array, ""]
        Whether each signed binary product occurs exactly in ``support``.

    Notes
    -----
    The loop checks products individually to avoid allocating the full pair
    grid.
    """
    support_keys: Int64[Array, " w"] = _flat_residues(support, work_shape)
    order: Int64[Array, " w"] = jnp.argsort(support_keys)
    sorted_keys: Int64[Array, " w"] = support_keys[order]
    right_count: int = right_indices.shape[0]
    product_count: int = left_indices.shape[0] * right_count

    def check_product(
        flat_position: scalar_int,
        valid: Bool[Array, ""],
    ) -> Bool[Array, ""]:
        """Accumulate exact membership without materializing all products."""
        left_position: scalar_int = flat_position // right_count
        right_position: scalar_int = flat_position % right_count
        candidate: Int64[Array, " 3"] = (
            left_indices[left_position]
            + right_sign * right_indices[right_position]
        )
        candidate_key: Int64[Array, ""] = _flat_residues(
            candidate[None, :], work_shape
        )[0]
        location: Int32[Array, ""] = jnp.searchsorted(
            sorted_keys, candidate_key, side="left"
        )
        clipped: Int32[Array, ""] = jnp.clip(location, 0, support.shape[0] - 1)
        exact_match: Bool[Array, ""] = (
            (location < support.shape[0])
            & (sorted_keys[clipped] == candidate_key)
            & jnp.all(support[order[clipped]] == candidate)
        )
        updated: Bool[Array, ""] = valid & exact_match
        return updated

    contained: Bool[Array, ""] = jax.lax.fori_loop(
        0,
        product_count,
        check_product,
        jnp.asarray(True),
    )
    return contained


def _cosine_shell_coefficients(
    indices: Int64[Array, "q 3"],
) -> Complex128[Array, " q"]:
    r"""PRIVATE: Return coefficients of the bounded cosine-shell profile.

    Parameters
    ----------
    indices : Int64[Array, "q 3"]
        Exact absorber-support indices.

    Returns
    -------
    coefficients : Complex128[Array, " q"]
        Analytic complex128 Fourier coefficients at ``indices``.

    Notes
    -----
    The profile is :math:`1-\prod_j\cos^2(\pi x_j/L_j)`. Coefficients outside
    the tensor shell ``{-1, 0, 1}^3`` are zero.
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
    work_shape: Tuple[int, int, int],
) -> Bool[Array, ""]:
    """PRIVATE: Check whether all 27 cosine-shell modes are represented.

    Parameters
    ----------
    indices : Int64[Array, "q 3"]
        Exact absorber-support indices.
    work_shape : Tuple[int, int, int]
        Rectangular quotient moduli in reciprocal-axis order.

    Returns
    -------
    complete : Bool[Array, ""]
        Whether ``indices`` contains every mode in ``{-1, 0, 1}^3``.
    """
    axis: Int64[Array, " 3"] = jnp.asarray((-1, 0, 1), dtype=jnp.int64)
    mesh = jnp.meshgrid(axis, axis, axis, indexing="ij")
    required: Int64[Array, "27 3"] = jnp.stack(mesh, axis=-1).reshape((27, 3))
    complete: Bool[Array, ""] = _all_members(required, indices, work_shape)
    return complete


class GalerkinProductSupport(eqx.Module):
    """Store independent supports for fixed scalar Galerkin products.

    :see: :class:`~.test_born_potential_types.TestGalerkinProductSupport`

    Attributes
    ----------
    state_indices : Int64[Array, "n 3"]
        Exact integer reciprocal indices in the retained state support.
    interaction_indices : Int64[Array, "p 3"]
        Exact integer reciprocal indices in the real-interaction support.
    absorber_indices : Int64[Array, "q 3"]
        Exact integer reciprocal indices in the absorber support.
    work_indices : Int64[Array, "w 3"]
        Exact integer reciprocal indices in the product work support.
    work_shape : Tuple[int, int, int]
        Static rectangular work-grid shape. This value affects tracing.

    See Also
    --------
    :func:`create_galerkin_product_support`
        Create and validate a :class:`GalerkinProductSupport`.

    Notes
    -----
    The factory enforces ``K_u + K_chi <= K_w`` and
    ``K_u - K_u <= K_a`` as exact integer-set inclusions. It also enforces
    the RM-S2 quotient injectivity and restricted no-alias predicates for both
    multiplier products.
    """

    state_indices: Int64[Array, "n 3"]
    interaction_indices: Int64[Array, "p 3"]
    absorber_indices: Int64[Array, "q 3"]
    work_indices: Int64[Array, "w 3"]
    work_shape: Tuple[int, int, int] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def create_galerkin_product_support(
    state_indices: Int[Array, "..."],
    interaction_indices: Int[Array, "..."],
    absorber_indices: Int[Array, "..."],
    work_indices: Int[Array, "..."],
    work_shape: Tuple[int, int, int],
) -> GalerkinProductSupport:
    """Create validated supports for fixed scalar Galerkin products.

    :see: :class:`~.test_born_potential_types.TestGalerkinProductSupport`

    Implementation Logic
    --------------------
    1. Validate nonempty three-dimensional integer-index arrays.
    2. Enforce exact support uniqueness, symmetry, and inclusions.
    3. Enforce quotient injectivity and both product no-alias predicates.

    Parameters
    ----------
    state_indices : Int[Array, "..."]
        Exact retained-state reciprocal indices with shape ``(n, 3)``.
    interaction_indices : Int[Array, "..."]
        Exact interaction reciprocal indices with shape ``(p, 3)``.
    absorber_indices : Int[Array, "..."]
        Exact absorber reciprocal indices with shape ``(q, 3)``.
    work_indices : Int[Array, "..."]
        Exact product-work reciprocal indices with shape ``(w, 3)``.
    work_shape : Tuple[int, int, int]
        Static positive rectangular work-grid shape. Changing it retraces.

    Returns
    -------
    support : GalerkinProductSupport
        Validated independent Galerkin product supports.

    Raises
    ------
    ValueError
        If an array shape or static work-grid dimension is invalid.
    equinox.EquinoxRuntimeError
        If a support is duplicated or nonconforming during traced execution.

    Notes
    -----
    The work support evaluates products. It does not enlarge the physical
    state support. This factory requires the stronger full-product rule for
    both the interaction and absorber coefficient supports.
    """
    if len(work_shape) != _SPACE_DIMENSIONS:
        raise ValueError("work_shape must contain exactly three dimensions")
    if any(isinstance(size, bool) for size in work_shape):
        raise ValueError("work_shape dimensions must not be boolean")
    if any(size <= 0 for size in work_shape):
        raise ValueError("work_shape dimensions must be positive")
    if math.prod(work_shape) > jnp.iinfo(jnp.int64).max:
        raise ValueError(
            "work_shape product must fit in signed 64-bit indices"
        )

    state_array: Int64[Array, "n 3"] = jnp.asarray(
        state_indices, dtype=jnp.int64
    )
    interaction_array: Int64[Array, "p 3"] = jnp.asarray(
        interaction_indices, dtype=jnp.int64
    )
    absorber_array: Int64[Array, "q 3"] = jnp.asarray(
        absorber_indices, dtype=jnp.int64
    )
    work_array: Int64[Array, "w 3"] = jnp.asarray(
        work_indices, dtype=jnp.int64
    )

    for values, name in (
        (state_array, "state_indices"),
        (interaction_array, "interaction_indices"),
        (absorber_array, "absorber_indices"),
        (work_array, "work_indices"),
    ):
        _raise_if(values.ndim != _SUPPORT_RANK, f"{name} must be 2D")
        _raise_if(
            values.shape[1:] != (_SPACE_DIMENSIONS,),
            f"{name} must have shape (n, 3)",
        )
        _raise_if(values.shape[0] == 0, f"{name} must be nonempty")

    safe_index_limit: int = jnp.iinfo(jnp.int64).max // 4
    unsafe_index: Bool[Array, ""] = jnp.asarray(False)
    for values in (
        state_array,
        interaction_array,
        absorber_array,
        work_array,
    ):
        unsafe_index = unsafe_index | jnp.any(
            (values < -safe_index_limit) | (values > safe_index_limit)
        )

    minimum_integer: int = jnp.iinfo(jnp.int64).min
    contains_unnegatable_index: Bool[Array, ""] = jnp.any(
        interaction_array == minimum_integer
    ) | jnp.any(absorber_array == minimum_integer)

    checked_state: Int64[Array, "n 3"] = eqx.error_if(
        state_array,
        unsafe_index
        | _contains_duplicates(state_array)
        | ~_quotient_is_injective(state_array, work_shape),
        "state support must be unique in the work quotient",
    )
    checked_interaction: Int64[Array, "p 3"] = eqx.error_if(
        interaction_array,
        unsafe_index
        | _contains_duplicates(interaction_array)
        | contains_unnegatable_index
        | ~_is_sign_symmetric(interaction_array)
        | ~_quotient_is_injective(interaction_array, work_shape)
        | ~_restricted_product_has_no_alias(
            state_array,
            interaction_array,
            work_shape,
        ),
        "interaction support must be unique, sign-symmetric, and no-alias",
    )
    checked_absorber: Int64[Array, "q 3"] = eqx.error_if(
        absorber_array,
        unsafe_index
        | _contains_duplicates(absorber_array)
        | contains_unnegatable_index
        | ~_is_sign_symmetric(absorber_array)
        | ~_all_binary_products_are_members(
            state_array,
            state_array,
            absorber_array,
            work_shape,
            -1,
        )
        | ~_quotient_is_injective(absorber_array, work_shape)
        | ~_restricted_product_has_no_alias(
            state_array,
            absorber_array,
            work_shape,
        ),
        "absorber support must contain K_u-K_u and satisfy symmetry/no-alias",
    )
    checked_work: Int64[Array, "w 3"] = eqx.error_if(
        work_array,
        unsafe_index
        | _contains_duplicates(work_array)
        | ~_quotient_is_injective(work_array, work_shape)
        | ~_all_binary_products_are_members(
            state_array,
            interaction_array,
            work_array,
            work_shape,
            1,
        )
        | ~_all_binary_products_are_members(
            state_array,
            absorber_array,
            work_array,
            work_shape,
            1,
        ),
        "work support must contain both product sets and be quotient-unique",
    )

    support: GalerkinProductSupport = GalerkinProductSupport(
        state_indices=checked_state,
        interaction_indices=checked_interaction,
        absorber_indices=checked_absorber,
        work_indices=checked_work,
        work_shape=work_shape,
    )
    return support


__all__: list[str] = [
    "GalerkinProductSupport",
    "create_galerkin_product_support",
]
