r"""Build the RM-S2 fixed-linear Galerkin error enclosure.

Extended Summary
----------------
This module encloses the exact SC-1 free and interaction matrices by one
canonical frozen binary64 realization.  It implements the coefficient
multiplicity, row, column, Frobenius, Schur, and total component ledgers in
RM-S2 S2.27--S2.43.  Source, per-call action, residual-formation, and solver
errors are intentionally outside this module.

Routine Listings
----------------
:func:`build_galerkin_fixed_linear_error_ledger`
    Build a fixed-linear error ledger from manifested physical inputs.

Notes
-----
The exact target treats every supplied binary physical value and stored
physical constant as an exact real number.  Mathematical pi is enclosed by
the adjacent binary64 numbers that bracket it.  Exact SC.2 and SC.4 use the
stored reduced Planck constant; the current algebraic interaction realization
uses its separately frozen 50-mantissa-bit Planck/pi route, and that difference
is charged rather than silently identified.
"""

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
    Float,
    Float64,
    Int,
    Int64,
    jaxtyped,
)

from ptyrodactyl._tools import (
    coupled_interaction_value,
    interval_add,
    interval_divide_positive,
    interval_multiply,
    interval_sqrt,
    interval_square,
    interval_subtract,
    mathematical_pi_interval,
    point_interval,
    round_up,
    upward_add,
    upward_multiply,
    upward_sqrt,
)
from ptyrodactyl.types import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M_E,
    GalerkinFixedLinearAbsorberRoute,
    GalerkinFixedLinearErrorLedger,
    create_galerkin_fixed_linear_error_ledger,
    scalar_float,
)

_SPACE_DIMENSIONS: int = 3
_SUPPORT_RANK: int = 2
_MAX_EXACT_BINARY64_INTEGER: int = 1 << 53
_ANGSTROM_SQUARED_LOWER: float = float.fromhex("0x1.79ca10c924223p-67")
_EXACT_GEOMETRY_TARGET: str = (
    "SC.2 positive k0 from exact binary U0/M_E/E_CHARGE/C_LIGHT/HBAR; "
    "SC.8 carrier is k0 times the exact-real normalization of the stored "
    "nonzero binary direction seed; SC.23 exact on-shell diagonal; "
    "mathematical pi bracketed by adjacent binary64 values"
)
_ALGEBRAIC_GEOMETRY_REALIZATION: str = (
    "stored binary64 acquisition carrier/k0/box; canonical SC.22 JAX "
    "binary64 diagonal"
)
_EXACT_INTERACTION_TARGET: str = (
    "SC.4 sigma_H from exact binary U0/M_E/E_CHARGE/C_LIGHT/HBAR"
)
_ALGEBRAIC_INTERACTION_REALIZATION: str = (
    "stored coupling and interaction equal coupled_interaction_value: "
    "H_PLANCK/mathematical-pi binary evaluation with canonical "
    "50-mantissa-bit rounding; H_alg is the exact direct Toeplitz matrix of "
    "those stored coefficients, while evaluated FFT/reduction rounding is "
    "delta_fl"
)
_ABSORBER_TARGET: str = (
    "a(x)=1-product_j cos(pi*x_j/L_j)^2 with exact SC.13b dyadic "
    "coefficients on {-1,0,1}^3"
)
_CAP_TARGET: str = (
    "stored binary64 cap_scale interpreted as the exact target scalar; "
    "H_alg stores the exact factor product cap_scale*A and leaves evaluated "
    "multiplication rounding to delta_fl"
)
_ERROR_SCOPE: str = (
    "fixed_linear_H_alg_minus_exact_SC1_H_only; excludes source, per-call "
    "action, residual formation, solver recurrence, and model discrepancy"
)
_COEFFICIENT_NORM: str = "SC.12/SC.13 Euclidean complex coefficient norm"


def _raise_if(condition: bool, message: str) -> None:
    """PRIVATE: Raise ``ValueError`` for a structural contract failure.

    Parameters
    ----------
    condition : bool
        Structural failure predicate.
    message : str
        Exception message used when the predicate is true.

    Raises
    ------
    ValueError
        If ``condition`` is true.
    """
    if condition:
        raise ValueError(message)


def _absolute_interval_upper(
    value: Tuple[Float64[Array, "..."], Float64[Array, "..."]],
) -> Float64[Array, "..."]:
    """PRIVATE: Return an upper bound on absolute value over an interval.

    Parameters
    ----------
    value : Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        Inclusive endpoints for the input interval.

    Returns
    -------
    result : Float64[Array, "..."]
        Maximum absolute endpoint value.

    Notes
    -----
    The absolute value reaches its interval maximum at one endpoint.
    """
    result: Float64[Array, "..."] = jnp.maximum(
        jnp.abs(value[0]),
        jnp.abs(value[1]),
    )
    return result


def _point_interval_distance_upper(
    point: Float64[Array, "..."],
    interval: Tuple[Float64[Array, "..."], Float64[Array, "..."]],
) -> Float64[Array, "..."]:
    """PRIVATE: Enclose distance from one exact point to an interval value.

    Parameters
    ----------
    point : Float64[Array, "..."]
        Binary64 values interpreted as exact real points.
    interval : Tuple[Float64[Array, "..."], Float64[Array, "..."]]
        Inclusive endpoints for the comparison interval.

    Returns
    -------
    result : Float64[Array, "..."]
        Upper bounds on the absolute point-to-interval differences.

    Notes
    -----
    The result bounds every exact interval value, not only the nearest point.
    """
    difference = interval_subtract(point_interval(point), interval)
    result: Float64[Array, "..."] = _absolute_interval_upper(difference)
    return result


def _interval_sum(
    values: Tuple[Tuple[Float64[Array, "..."], Float64[Array, "..."]], ...],
) -> Tuple[Float64[Array, "..."], Float64[Array, "..."]]:
    """PRIVATE: Outward-sum a fixed tuple of intervals.

    Parameters
    ----------
    values : Tuple[Tuple[Float64[Array, "..."], Float64[Array, "..."]], ...]
        Nonempty tuple of aligned inclusive intervals.

    Returns
    -------
    lower : Float64[Array, "..."]
        Inclusive lower endpoints for the accumulated sum.
    upper : Float64[Array, "..."]
        Inclusive upper endpoints for the accumulated sum.

    Notes
    -----
    The fixed iteration order defines the binary64 accumulation route.
    """
    first: Float64[Array, "..."] = jnp.zeros_like(values[0][0])
    result: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
        point_interval(first)
    )
    for value in values:
        result = interval_add(result, value)
    lower: Float64[Array, "..."] = result[0]
    upper: Float64[Array, "..."] = result[1]
    summed: Tuple[Float64[Array, "..."], Float64[Array, "..."]] = (
        lower,
        upper,
    )
    return summed


def _exact_kinematic_intervals(
    accelerating_voltage_kv: Float64[Array, ""],
    direction_seed: Float64[Array, " 3"],
) -> Tuple[
    Tuple[Float64[Array, ""], Float64[Array, ""]],
    Tuple[Float64[Array, " 3"], Float64[Array, " 3"]],
    Tuple[Float64[Array, ""], Float64[Array, ""]],
]:
    r"""PRIVATE: Enclose exact SC.2, SC.8, and SC.4 quantities.

    Implementation Logic
    --------------------
    1. Enclose the relativistic SC.2 wavenumber from exact manifested inputs.
    2. Enclose the SC.4 interaction coupling with the same physical inputs.
    3. Normalize the binary direction seed and scale it by the wavenumber.

    Parameters
    ----------
    accelerating_voltage_kv : Float64[Array, ""]
        Exact manifested accelerating voltage in kilovolts.
    direction_seed : Float64[Array, " 3"]
        Nonzero binary64 acquisition direction seed in Cartesian axis order.

    Returns
    -------
    wavenumber : Tuple[Float64[Array, ""], Float64[Array, ""]]
        Inclusive SC.2 wavenumber endpoints in inverse Angstroms.
    exact_carrier : Tuple[Float64[Array, " 3"], Float64[Array, " 3"]]
        Inclusive SC.8 carrier endpoints in radians per Angstrom and
        Cartesian axis order.
    interaction_coupling : Tuple[Float64[Array, ""], Float64[Array, ""]]
        Inclusive SC.4 interaction-coupling endpoints in inverse-square
        Angstroms per volt.

    Notes
    -----
    Stored physical constants are interpreted as exact real numbers. The
    ``1.0e-20`` factor converts square metres to square Angstroms.
    """
    one = point_interval(jnp.asarray(1.0, dtype=jnp.float64))
    two = point_interval(jnp.asarray(2.0, dtype=jnp.float64))
    thousand = point_interval(jnp.asarray(1000.0, dtype=jnp.float64))
    angstrom_squared_lower: Float64[Array, ""] = jnp.asarray(
        _ANGSTROM_SQUARED_LOWER,
        dtype=jnp.float64,
    )
    angstrom_squared = (
        angstrom_squared_lower,
        round_up(angstrom_squared_lower),
    )
    mass = point_interval(jnp.asarray(M_E, dtype=jnp.float64))
    charge = point_interval(jnp.asarray(E_CHARGE, dtype=jnp.float64))
    speed = point_interval(jnp.asarray(C_LIGHT, dtype=jnp.float64))
    hbar = point_interval(jnp.asarray(HBAR, dtype=jnp.float64))
    voltage = point_interval(accelerating_voltage_kv)
    voltage_volts = interval_multiply(voltage, thousand)

    numerator = interval_multiply(
        interval_multiply(two, mass),
        charge,
    )
    hbar_squared = interval_square(hbar)
    prefactor = interval_divide_positive(numerator, hbar_squared)
    speed_squared = interval_square(speed)
    rest_energy = interval_multiply(mass, speed_squared)
    charge_voltage = interval_multiply(charge, voltage_volts)

    twice_rest_energy = interval_multiply(two, rest_energy)
    kinematic_ratio = interval_divide_positive(
        charge_voltage,
        twice_rest_energy,
    )
    kinematic_correction = interval_add(one, kinematic_ratio)
    wavenumber_squared = interval_multiply(
        interval_multiply(
            interval_multiply(prefactor, voltage_volts),
            kinematic_correction,
        ),
        angstrom_squared,
    )
    wavenumber: Tuple[Float64[Array, ""], Float64[Array, ""]] = interval_sqrt(
        wavenumber_squared
    )

    interaction_ratio = interval_divide_positive(
        charge_voltage,
        rest_energy,
    )
    interaction_correction = interval_add(one, interaction_ratio)
    interaction_coupling: Tuple[Float64[Array, ""], Float64[Array, ""]] = (
        interval_multiply(
            interval_multiply(prefactor, interaction_correction),
            angstrom_squared,
        )
    )

    scale: Float64[Array, ""] = jnp.max(jnp.abs(direction_seed))
    scaled_seed = interval_divide_positive(
        point_interval(direction_seed),
        point_interval(scale),
    )
    seed_norm_squared = _interval_sum(
        tuple(
            interval_square((scaled_seed[0][axis], scaled_seed[1][axis]))
            for axis in range(_SPACE_DIMENSIONS)
        )
    )
    seed_norm = interval_sqrt(seed_norm_squared)
    unit_direction = interval_divide_positive(scaled_seed, seed_norm)
    exact_carrier: Tuple[Float64[Array, " 3"], Float64[Array, " 3"]] = (
        interval_multiply(
            (
                jnp.broadcast_to(wavenumber[0], (_SPACE_DIMENSIONS,)),
                jnp.broadcast_to(wavenumber[1], (_SPACE_DIMENSIONS,)),
            ),
            unit_direction,
        )
    )
    result: Tuple[
        Tuple[Float64[Array, ""], Float64[Array, ""]],
        Tuple[Float64[Array, " 3"], Float64[Array, " 3"]],
        Tuple[Float64[Array, ""], Float64[Array, ""]],
    ] = (wavenumber, exact_carrier, interaction_coupling)
    return result


def _exact_free_diagonal_interval(
    state_indices: Int64[Array, "n 3"],
    box_lengths: Float64[Array, " 3"],
    exact_carrier: Tuple[Float64[Array, " 3"], Float64[Array, " 3"]],
) -> Tuple[Float64[Array, " n"], Float64[Array, " n"]]:
    r"""PRIVATE: Enclose the exact on-shell SC.23 free diagonal.

    Implementation Logic
    --------------------
    1. Enclose reciprocal offsets from exact integer indices and box lengths.
    2. Enclose mathematical pi with adjacent binary64 values.
    3. Evaluate ``2 k_carrier dot q + |q|^2`` with interval arithmetic.

    Parameters
    ----------
    state_indices : Int64[Array, "n 3"]
        Retained reciprocal indices in Cartesian axis order.
    box_lengths : Float64[Array, " 3"]
        Exact manifested box lengths in Angstroms and Cartesian axis order.
    exact_carrier : Tuple[Float64[Array, " 3"], Float64[Array, " 3"]]
        Inclusive SC.8 carrier endpoints in radians per Angstrom.

    Returns
    -------
    lower : Float64[Array, " n"]
        Inclusive lower endpoints for the SC.23 free diagonal in
        inverse-square Angstroms.
    upper : Float64[Array, " n"]
        Inclusive upper endpoints for the SC.23 free diagonal in
        inverse-square Angstroms.

    Notes
    -----
    The diagonal uses the on-shell cancellation form. This form avoids
    subtracting a separately evaluated squared wavenumber.
    """
    state_float: Float64[Array, "n 3"] = state_indices.astype(jnp.float64)
    reciprocal = interval_divide_positive(
        point_interval(state_float),
        point_interval(box_lengths[None, :]),
    )
    pi_interval = mathematical_pi_interval()
    two_pi = interval_multiply(
        point_interval(jnp.asarray(2.0, dtype=jnp.float64)),
        pi_interval,
    )
    wavevector_offset = interval_multiply(
        reciprocal,
        (
            jnp.broadcast_to(two_pi[0], reciprocal[0].shape),
            jnp.broadcast_to(two_pi[1], reciprocal[1].shape),
        ),
    )
    carrier_dot_offset = _interval_sum(
        tuple(
            interval_multiply(
                (
                    exact_carrier[0][axis],
                    exact_carrier[1][axis],
                ),
                (
                    wavevector_offset[0][:, axis],
                    wavevector_offset[1][:, axis],
                ),
            )
            for axis in range(_SPACE_DIMENSIONS)
        )
    )
    offset_norm_squared = _interval_sum(
        tuple(
            interval_square(
                (
                    wavevector_offset[0][:, axis],
                    wavevector_offset[1][:, axis],
                )
            )
            for axis in range(_SPACE_DIMENSIONS)
        )
    )
    twice_dot = interval_multiply(
        point_interval(jnp.asarray(2.0, dtype=jnp.float64)),
        carrier_dot_offset,
    )
    interval: Tuple[Float64[Array, " n"], Float64[Array, " n"]] = interval_add(
        twice_dot, offset_norm_squared
    )
    lower: Float64[Array, " n"] = interval[0]
    upper: Float64[Array, " n"] = interval[1]
    result: Tuple[Float64[Array, " n"], Float64[Array, " n"]] = (
        lower,
        upper,
    )
    return result


def _algebraic_free_diagonal(
    state_indices: Int64[Array, "n 3"],
    box_lengths: Float64[Array, " 3"],
    carrier: Float64[Array, " 3"],
    wavenumber: Float64[Array, ""],
) -> Float64[Array, " n"]:
    r"""PRIVATE: Build the canonical frozen binary64 SC.22 diagonal.

    Implementation Logic
    --------------------
    1. Convert integer state indices to reciprocal frequencies.
    2. Add each reciprocal offset to the stored carrier.
    3. Subtract the stored squared wavenumber from each squared wavevector.
    4. Freeze the evaluated diagonal behind an optimization barrier.

    Parameters
    ----------
    state_indices : Int64[Array, "n 3"]
        Retained reciprocal indices in Cartesian axis order.
    box_lengths : Float64[Array, " 3"]
        Stored binary64 box lengths in Angstroms and Cartesian axis order.
    carrier : Float64[Array, " 3"]
        Stored binary64 carrier in radians per Angstrom and Cartesian axis
        order.
    wavenumber : Float64[Array, ""]
        Stored binary64 wavenumber in radians per Angstrom.

    Returns
    -------
    diagonal : Float64[Array, " n"]
        Frozen binary64 SC.22 free-diagonal realization in inverse-square
        Angstroms.

    Notes
    -----
    This operation defines ``H_alg``. The independent interval route encloses
    its distance from the exact SC.23 target.
    """
    reciprocal_frequencies: Float64[Array, "n 3"] = (
        state_indices / box_lengths[None, :]
    )
    physical_wavevectors: Float64[Array, "n 3"] = carrier[None, :] + (
        2.0 * jnp.pi * reciprocal_frequencies
    )
    raw_diagonal: Float64[Array, " n"] = (
        jnp.sum(physical_wavevectors**2, axis=1) - wavenumber**2
    )
    diagonal: Float64[Array, " n"] = jax.lax.optimization_barrier(raw_diagonal)
    return diagonal


def _complex_product_discrepancy_upper(
    interaction_coefficients: Complex128[Array, " p"],
    interaction_coupling: Float64[Array, ""],
    voltage_coefficients: Complex128[Array, " p"],
) -> Float64[Array, " p"]:
    r"""PRIVATE: Enclose ``|chi_hat - sigma_hat*c_hat|`` componentwise.

    Implementation Logic
    --------------------
    1. Enclose the stored coupling times each voltage-coefficient component.
    2. Subtract each product interval from the stored interaction component.
    3. Add absolute real and imaginary discrepancy bounds.

    Parameters
    ----------
    interaction_coefficients : Complex128[Array, " p"]
        Stored binary64 complex interaction coefficients in inverse-square
        Angstroms.
    interaction_coupling : Float64[Array, ""]
        Stored binary64 interaction coupling in inverse-square Angstroms per
        volt.
    voltage_coefficients : Complex128[Array, " p"]
        Stored binary64 complex voltage coefficients in volts.

    Returns
    -------
    result : Float64[Array, " p"]
        Componentwise upper bounds on stored-product discrepancies in
        inverse-square Angstroms.

    Notes
    -----
    The real-plus-imaginary bound is conservative for the complex Euclidean
    magnitude and avoids an additional square-root rounding path.
    """
    coupling = point_interval(interaction_coupling)
    product_real = interval_multiply(
        coupling,
        point_interval(jnp.real(voltage_coefficients)),
    )
    product_imaginary = interval_multiply(
        coupling,
        point_interval(jnp.imag(voltage_coefficients)),
    )
    real_difference = interval_subtract(
        point_interval(jnp.real(interaction_coefficients)),
        product_real,
    )
    imaginary_difference = interval_subtract(
        point_interval(jnp.imag(interaction_coefficients)),
        product_imaginary,
    )
    result: Float64[Array, " p"] = upward_add(
        _absolute_interval_upper(real_difference),
        _absolute_interval_upper(imaginary_difference),
    )
    return result


def _complex_l1_upper(
    values: Complex128[Array, " p"],
) -> Float64[Array, " p"]:
    r"""PRIVATE: Bound each complex magnitude with an outward L1 norm.

    Parameters
    ----------
    values : Complex128[Array, " p"]
        Stored binary64 complex values.

    Returns
    -------
    result : Float64[Array, " p"]
        Componentwise upper bounds ``|real(values)| + |imag(values)|``.

    Notes
    -----
    The complex L1 norm bounds the complex Euclidean magnitude from above.
    """
    result: Float64[Array, " p"] = upward_add(
        jnp.abs(jnp.real(values)),
        jnp.abs(jnp.imag(values)),
    )
    return result


def _interaction_matrix_error_bounds(
    state_indices: Int64[Array, "n 3"],
    interaction_indices: Int64[Array, "p 3"],
    coefficient_errors: Float64[Array, " p"],
) -> Tuple[
    Int64[Array, " p"],
    Float64[Array, " n"],
    Float64[Array, " n"],
]:
    """PRIVATE: Derive S2.30 and the unmaximized S2.31--S2.32 sums.

    Implementation Logic
    --------------------
    1. Form every retained state-index difference.
    2. Match each represented interaction index against the differences.
    3. Accumulate multiplicities and outward row and column error sums.

    Parameters
    ----------
    state_indices : Int64[Array, "n 3"]
        Unique retained reciprocal indices in Cartesian axis order.
    interaction_indices : Int64[Array, "p 3"]
        Unique represented interaction indices in Cartesian axis order.
    coefficient_errors : Float64[Array, " p"]
        Non-negative componentwise interaction-coefficient error bounds in
        inverse-square Angstroms.

    Returns
    -------
    multiplicities : Int64[Array, " p"]
        S2.30 retained-difference multiplicities for represented coefficients.
    row_sums : Float64[Array, " n"]
        Unmaximized outward S2.31 row operator-error sums in inverse-square
        Angstroms.
    column_sums : Float64[Array, " n"]
        Unmaximized outward S2.32 column operator-error sums in inverse-square
        Angstroms.

    Notes
    -----
    Each represented difference contributes at most one coefficient error to
    each matching row and column.
    """
    differences: Int64[Array, "n n 3"] = (
        state_indices[:, None, :] - state_indices[None, :, :]
    )
    multiplicities: Int64[Array, " p"] = jnp.zeros(
        (interaction_indices.shape[0],),
        dtype=jnp.int64,
    )
    row_sums: Float64[Array, " n"] = jnp.zeros(
        (state_indices.shape[0],),
        dtype=jnp.float64,
    )
    column_sums: Float64[Array, " n"] = jnp.zeros(
        (state_indices.shape[0],),
        dtype=jnp.float64,
    )

    def add_coefficient(
        position: Int64[Array, ""],
        accumulator: Tuple[
            Int64[Array, " p"],
            Float64[Array, " n"],
            Float64[Array, " n"],
        ],
    ) -> Tuple[
        Int64[Array, " p"],
        Float64[Array, " n"],
        Float64[Array, " n"],
    ]:
        """Accumulate one represented difference coefficient."""
        counts, rows, columns = accumulator
        matches: Bool[Array, "n n"] = jnp.all(
            differences == interaction_indices[position],
            axis=-1,
        )
        multiplicity: Int64[Array, ""] = jnp.sum(
            matches,
            dtype=jnp.int64,
        )
        row_increment: Float64[Array, " n"] = jnp.where(
            jnp.any(matches, axis=1),
            coefficient_errors[position],
            0.0,
        )
        column_increment: Float64[Array, " n"] = jnp.where(
            jnp.any(matches, axis=0),
            coefficient_errors[position],
            0.0,
        )
        updated: Tuple[
            Int64[Array, " p"],
            Float64[Array, " n"],
            Float64[Array, " n"],
        ] = (
            counts.at[position].set(multiplicity),
            upward_add(rows, row_increment),
            upward_add(columns, column_increment),
        )
        return updated

    multiplicities, row_sums, column_sums = jax.lax.fori_loop(
        0,
        interaction_indices.shape[0],
        add_coefficient,
        (multiplicities, row_sums, column_sums),
    )
    result: Tuple[
        Int64[Array, " p"],
        Float64[Array, " n"],
        Float64[Array, " n"],
    ] = (multiplicities, row_sums, column_sums)
    return result


def _frobenius_error_upper(
    multiplicities: Int64[Array, " p"],
    coefficient_errors: Float64[Array, " p"],
) -> Float64[Array, ""]:
    r"""PRIVATE: Derive the outward multiplicity-weighted S2.33a bound.

    Implementation Logic
    --------------------
    1. Round each nonzero integer multiplicity upward after binary64 casting.
    2. Accumulate ``multiplicity * coefficient_error**2`` outward.
    3. Take an outward square root of the accumulated radicand.

    Parameters
    ----------
    multiplicities : Int64[Array, " p"]
        S2.30 retained-difference multiplicities.
    coefficient_errors : Float64[Array, " p"]
        Non-negative componentwise interaction-coefficient error bounds in
        inverse-square Angstroms.

    Returns
    -------
    result : Float64[Array, ""]
        Outward S2.33a Frobenius operator-error bound in inverse-square
        Angstroms.

    Notes
    -----
    Multiplicities need not be exactly representable after binary64 casting;
    the explicit upward step keeps the bound conservative.
    """
    radicand: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)

    def add_term(
        position: Int64[Array, ""],
        total: Float64[Array, ""],
    ) -> Float64[Array, ""]:
        """Accumulate one non-negative Frobenius-square term."""
        raw_multiplicity: Float64[Array, ""] = jnp.asarray(
            multiplicities[position],
            dtype=jnp.float64,
        )
        multiplicity_upper: Float64[Array, ""] = jnp.where(
            multiplicities[position] == 0,
            0.0,
            round_up(raw_multiplicity),
        )
        squared_error: Float64[Array, ""] = upward_multiply(
            coefficient_errors[position],
            coefficient_errors[position],
        )
        term: Float64[Array, ""] = upward_multiply(
            multiplicity_upper,
            squared_error,
        )
        updated: Float64[Array, ""] = upward_add(total, term)
        return updated

    radicand = jax.lax.fori_loop(
        0,
        coefficient_errors.shape[0],
        add_term,
        radicand,
    )
    result: Float64[Array, ""] = upward_sqrt(radicand)
    return result


def _contains_duplicates(indices: Int64[Array, "n 3"]) -> Bool[Array, ""]:
    """PRIVATE: Detect exact duplicate reciprocal indices.

    Parameters
    ----------
    indices : Int64[Array, "n 3"]
        Reciprocal indices in Cartesian axis order.

    Returns
    -------
    result : Bool[Array, ""]
        True when at least two rows are equal; otherwise, false.

    Notes
    -----
    Lexicographic sorting makes every duplicate pair adjacent.
    """
    order: Int64[Array, " n"] = jnp.lexsort(
        (indices[:, 2], indices[:, 1], indices[:, 0])
    )
    sorted_indices: Int64[Array, "n 3"] = indices[order]
    result: Bool[Array, ""] = jnp.any(
        jnp.all(sorted_indices[1:] == sorted_indices[:-1], axis=-1)
    )
    return result


@jaxtyped(typechecker=beartype)
def build_galerkin_fixed_linear_error_ledger(  # noqa: PLR0913, PLR0915
    state_indices: Int[Array, "..."],
    interaction_indices: Int[Array, "..."],
    voltage_coefficients: Complex[Array, "..."],
    voltage_coefficient_error_bounds: Float[Array, "..."],
    interaction_coupling: scalar_float,
    interaction_coefficients: Complex[Array, "..."],
    accelerating_voltage_kv: scalar_float,
    carrier: Float[Array, "..."],
    box_lengths: Float[Array, "..."],
    wavenumber: scalar_float,
    cap_scale: scalar_float,
) -> GalerkinFixedLinearErrorLedger:
    r"""Build a fixed-linear error ledger from manifested physical inputs.

    :see: :class:`~.test_enclosures.TestGalerkinFixedLinearEnclosure`

    Implementation Logic
    --------------------
    1. Validate the frozen supports, physical inputs, and canonical
       interaction realization.
    2. Enclose exact SC.2/SC.8 geometry and SC.4 coupling with real outward
       interval arithmetic.
    3. Transfer VC-1 coefficient errors through coupling and stored-product
       rounding exactly once.
    4. Derive S2.30 multiplicities, S2.31--S2.32 row/column sums, both S2.33a
       bounds, and S2.43.

    Parameters
    ----------
    state_indices : Int[Array, "..."]
        Unique retained reciprocal indices with shape ``(n, 3)``.
    interaction_indices : Int[Array, "..."]
        Unique represented interaction indices with shape ``(p, 3)``.
    voltage_coefficients : Complex[Array, "..."]
        Stored VC-1 voltage coefficients aligned with ``interaction_indices``.
    voltage_coefficient_error_bounds : Float[Array, "..."]
        Componentwise outward VC.17 errors. Infinity is a noncertificate.
    interaction_coupling : scalar_float
        Stored canonical 50-mantissa-bit Helmholtz coupling.
    interaction_coefficients : Complex[Array, "..."]
        Stored canonical interaction coefficients.
    accelerating_voltage_kv : scalar_float
        Exact manifested accelerating voltage in kilovolts.
    carrier : Float[Array, "..."]
        Stored binary acquisition carrier and nonzero exact direction seed.
    box_lengths : Float[Array, "..."]
        Exact manifested box lengths in Angstroms.
    wavenumber : scalar_float
        Stored binary acquisition wavenumber used by ``H_alg``.
    cap_scale : scalar_float
        Exact manifested CAP scale and frozen algebraic factor.

    Returns
    -------
    ledger : GalerkinFixedLinearErrorLedger
        Fixed-linear component evidence. ``algebraic_free_diagonal`` is the
        only diagonal that this ledger certifies as ``H_alg``.

    Raises
    ------
    ValueError
        If an input rank, shape, or support size is invalid.
    equinox.EquinoxRuntimeError
        If a dynamic value is invalid or the stored interaction is not the
        canonical frozen realization of the supplied voltage coefficients.

    Notes
    -----
    The exact carrier is not the stored vector interpreted as accidentally
    on-shell. It is exact SC.2 ``k0`` times the exact-real normalization of
    that nonzero binary vector. Thus tiny tilts retain their direction while
    all stored shell error is charged to ``delta_D``.

    The analytic cosine-shell coefficients are products and differences of
    ``1/2`` and ``1/4`` and are therefore exact dyadic binary values. With the
    stored CAP scalar interpreted as an exact real and ``H_alg`` defined by
    the exact factor product, ``delta_A = delta_epsilon = delta_B = 0``.
    Rounding in an evaluated CAP multiplication belongs only to the separate
    per-call S2.89 ledger.
    """
    state_array: Int64[Array, "n 3"] = jnp.asarray(
        state_indices,
        dtype=jnp.int64,
    )
    interaction_index_array: Int64[Array, "p 3"] = jnp.asarray(
        interaction_indices,
        dtype=jnp.int64,
    )
    voltage_array: Complex128[Array, " p"] = jnp.asarray(
        voltage_coefficients,
        dtype=jnp.complex128,
    )
    voltage_error_array: Float64[Array, " p"] = jnp.asarray(
        voltage_coefficient_error_bounds,
        dtype=jnp.float64,
    )
    stored_coupling: Float64[Array, ""] = jnp.asarray(
        interaction_coupling,
        dtype=jnp.float64,
    )
    interaction_array: Complex128[Array, " p"] = jnp.asarray(
        interaction_coefficients,
        dtype=jnp.complex128,
    )
    accelerating_voltage: Float64[Array, ""] = jnp.asarray(
        accelerating_voltage_kv,
        dtype=jnp.float64,
    )
    carrier_array: Float64[Array, " 3"] = jnp.asarray(
        carrier,
        dtype=jnp.float64,
    )
    box_array: Float64[Array, " 3"] = jnp.asarray(
        box_lengths,
        dtype=jnp.float64,
    )
    stored_wavenumber: Float64[Array, ""] = jnp.asarray(
        wavenumber,
        dtype=jnp.float64,
    )
    cap_array: Float64[Array, ""] = jnp.asarray(
        cap_scale,
        dtype=jnp.float64,
    )

    for indices, name in (
        (state_array, "state_indices"),
        (interaction_index_array, "interaction_indices"),
    ):
        _raise_if(
            indices.ndim != _SUPPORT_RANK
            or indices.shape[1:] != (_SPACE_DIMENSIONS,),
            f"{name} must have shape (n, 3)",
        )
        _raise_if(indices.shape[0] == 0, f"{name} must be nonempty")
    _raise_if(voltage_array.ndim != 1, "voltage_coefficients must be 1D")
    _raise_if(
        voltage_array.shape[0] != interaction_index_array.shape[0],
        "voltage_coefficients must match interaction_indices",
    )
    _raise_if(
        voltage_error_array.shape != voltage_array.shape,
        "voltage_coefficient_error_bounds must match voltage_coefficients",
    )
    _raise_if(
        interaction_array.shape != voltage_array.shape,
        "interaction_coefficients must match voltage_coefficients",
    )
    _raise_if(carrier_array.shape != (3,), "carrier must have shape (3,)")
    _raise_if(box_array.shape != (3,), "box_lengths must have shape (3,)")
    for value, name in (
        (stored_coupling, "interaction_coupling"),
        (accelerating_voltage, "accelerating_voltage_kv"),
        (stored_wavenumber, "wavenumber"),
        (cap_array, "cap_scale"),
    ):
        _raise_if(value.shape != (), f"{name} must be a scalar")

    checked_state: Int64[Array, "n 3"] = eqx.error_if(
        state_array,
        _contains_duplicates(state_array)
        | jnp.any(state_array > _MAX_EXACT_BINARY64_INTEGER)
        | jnp.any(state_array < -_MAX_EXACT_BINARY64_INTEGER),
        "state_indices must be unique and exactly representable in binary64",
    )
    checked_interaction_indices: Int64[Array, "p 3"] = eqx.error_if(
        interaction_index_array,
        _contains_duplicates(interaction_index_array),
        "interaction_indices must be unique",
    )
    checked_voltage: Complex128[Array, " p"] = eqx.error_if(
        voltage_array,
        jnp.any(~jnp.isfinite(voltage_array)),
        "voltage_coefficients must be finite",
    )
    checked_voltage_errors: Float64[Array, " p"] = eqx.error_if(
        voltage_error_array,
        jnp.any(jnp.isnan(voltage_error_array))
        | jnp.any(voltage_error_array < 0.0),
        "voltage_coefficient_error_bounds must be non-negative and not NaN",
    )
    checked_interaction: Complex128[Array, " p"] = eqx.error_if(
        interaction_array,
        jnp.any(~jnp.isfinite(interaction_array)),
        "interaction_coefficients must be finite",
    )
    checked_coupling: Float64[Array, ""] = eqx.error_if(
        stored_coupling,
        (~jnp.isfinite(stored_coupling)) | (stored_coupling <= 0.0),
        "interaction_coupling must be finite and positive",
    )
    checked_voltage_kv: Float64[Array, ""] = eqx.error_if(
        accelerating_voltage,
        (~jnp.isfinite(accelerating_voltage)) | (accelerating_voltage <= 0.0),
        "accelerating_voltage_kv must be finite and positive",
    )
    checked_carrier: Float64[Array, " 3"] = eqx.error_if(
        carrier_array,
        jnp.any(~jnp.isfinite(carrier_array))
        | (jnp.max(jnp.abs(carrier_array)) == 0.0),
        "carrier must be finite and nonzero",
    )
    checked_box: Float64[Array, " 3"] = eqx.error_if(
        box_array,
        jnp.any(~jnp.isfinite(box_array)) | jnp.any(box_array <= 0.0),
        "box_lengths must be finite and positive",
    )
    checked_wavenumber: Float64[Array, ""] = eqx.error_if(
        stored_wavenumber,
        (~jnp.isfinite(stored_wavenumber)) | (stored_wavenumber <= 0.0),
        "wavenumber must be finite and positive",
    )
    checked_cap: Float64[Array, ""] = eqx.error_if(
        cap_array,
        (~jnp.isfinite(cap_array)) | (cap_array <= 0.0),
        "cap_scale must be finite and positive",
    )

    canonical_coupling: Float64[Array, ""]
    canonical_interaction: Complex128[Array, " p"]
    canonical_coupling, canonical_interaction = coupled_interaction_value(
        checked_voltage,
        checked_voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    checked_coupling = eqx.error_if(
        checked_coupling,
        checked_coupling != canonical_coupling,
        "interaction_coupling must equal the canonical 50-bit realization",
    )
    checked_interaction = eqx.error_if(
        checked_interaction,
        jnp.any(checked_interaction != canonical_interaction),
        "interaction_coefficients must equal the canonical 50-bit realization",
    )

    algebraic_free_diagonal: Float64[Array, " n"] = _algebraic_free_diagonal(
        checked_state,
        checked_box,
        checked_carrier,
        checked_wavenumber,
    )
    algebraic_free_diagonal = eqx.error_if(
        algebraic_free_diagonal,
        jnp.any(~jnp.isfinite(algebraic_free_diagonal)),
        "canonical algebraic free diagonal must be finite",
    )

    evidence_state = jax.lax.stop_gradient(checked_state)
    evidence_interaction_indices = jax.lax.stop_gradient(
        checked_interaction_indices
    )
    evidence_voltage = jax.lax.stop_gradient(checked_voltage)
    evidence_voltage_errors = jax.lax.stop_gradient(checked_voltage_errors)
    evidence_coupling = jax.lax.stop_gradient(checked_coupling)
    evidence_interaction = jax.lax.stop_gradient(checked_interaction)
    evidence_voltage_kv = jax.lax.stop_gradient(checked_voltage_kv)
    evidence_carrier = jax.lax.stop_gradient(checked_carrier)
    evidence_box = jax.lax.stop_gradient(checked_box)
    evidence_wavenumber = jax.lax.stop_gradient(checked_wavenumber)
    evidence_free_diagonal = jax.lax.stop_gradient(algebraic_free_diagonal)
    del checked_cap

    exact_wavenumber, exact_carrier, exact_coupling = (
        _exact_kinematic_intervals(evidence_voltage_kv, evidence_carrier)
    )
    exact_free_diagonal = _exact_free_diagonal_interval(
        evidence_state,
        evidence_box,
        exact_carrier,
    )
    wavenumber_error: Float64[Array, ""] = _point_interval_distance_upper(
        evidence_wavenumber,
        exact_wavenumber,
    )
    carrier_errors: Float64[Array, " 3"] = _point_interval_distance_upper(
        evidence_carrier, exact_carrier
    )
    free_errors: Float64[Array, " n"] = _point_interval_distance_upper(
        evidence_free_diagonal,
        exact_free_diagonal,
    )
    delta_d: Float64[Array, ""] = jnp.max(free_errors)

    coupling_error: Float64[Array, ""] = _point_interval_distance_upper(
        evidence_coupling,
        exact_coupling,
    )
    interaction_rounding_errors: Float64[Array, " p"] = (
        _complex_product_discrepancy_upper(
            evidence_interaction,
            evidence_coupling,
            evidence_voltage,
        )
    )
    voltage_magnitudes: Float64[Array, " p"] = _complex_l1_upper(
        evidence_voltage
    )
    coupling_transfer_errors: Float64[Array, " p"] = upward_multiply(
        coupling_error,
        voltage_magnitudes,
    )
    exact_coupling_magnitude_upper: Float64[Array, ""] = jnp.maximum(
        jnp.abs(exact_coupling[0]),
        jnp.abs(exact_coupling[1]),
    )
    voltage_transfer_errors: Float64[Array, " p"] = upward_multiply(
        exact_coupling_magnitude_upper,
        evidence_voltage_errors,
    )
    interaction_errors: Float64[Array, " p"] = upward_add(
        upward_add(
            interaction_rounding_errors,
            coupling_transfer_errors,
        ),
        voltage_transfer_errors,
    )

    multiplicities: Int64[Array, " p"]
    row_error_bounds: Float64[Array, " n"]
    column_error_bounds: Float64[Array, " n"]
    multiplicities, row_error_bounds, column_error_bounds = (
        _interaction_matrix_error_bounds(
            evidence_state,
            evidence_interaction_indices,
            interaction_errors,
        )
    )
    maximum_row: Float64[Array, ""] = jnp.max(row_error_bounds)
    maximum_column: Float64[Array, ""] = jnp.max(column_error_bounds)
    schur_bound: Float64[Array, ""] = upward_sqrt(
        upward_multiply(maximum_row, maximum_column)
    )
    frobenius_bound: Float64[Array, ""] = _frobenius_error_upper(
        multiplicities,
        interaction_errors,
    )
    delta_r: Float64[Array, ""] = jnp.minimum(
        schur_bound,
        frobenius_bound,
    )

    zero: Float64[Array, ""] = jnp.asarray(0.0, dtype=jnp.float64)
    delta_a: Float64[Array, ""] = zero
    delta_epsilon: Float64[Array, ""] = zero
    delta_b: Float64[Array, ""] = zero
    delta_h: Float64[Array, ""] = upward_add(
        upward_add(delta_d, delta_r),
        delta_b,
    )
    finite_certificate: Bool[Array, ""] = (
        jnp.isfinite(delta_d)
        & jnp.isfinite(delta_r)
        & jnp.isfinite(delta_b)
        & jnp.isfinite(delta_h)
    )

    ledger_factory = create_galerkin_fixed_linear_error_ledger
    ledger: GalerkinFixedLinearErrorLedger = ledger_factory(
        algebraic_free_diagonal=algebraic_free_diagonal,
        exact_wavenumber_lower_bound=exact_wavenumber[0],
        exact_wavenumber_upper_bound=exact_wavenumber[1],
        wavenumber_error_bound=wavenumber_error,
        exact_carrier_lower_bounds=exact_carrier[0],
        exact_carrier_upper_bounds=exact_carrier[1],
        carrier_component_error_bounds=carrier_errors,
        box_length_error_bounds=jnp.zeros((3,), dtype=jnp.float64),
        exact_free_diagonal_lower_bounds=exact_free_diagonal[0],
        exact_free_diagonal_upper_bounds=exact_free_diagonal[1],
        free_diagonal_error_bounds=free_errors,
        free_operator_error_bound=delta_d,
        exact_interaction_coupling_lower_bound=exact_coupling[0],
        exact_interaction_coupling_upper_bound=exact_coupling[1],
        interaction_coupling_error_bound=coupling_error,
        interaction_rounding_error_bounds=interaction_rounding_errors,
        interaction_coupling_transfer_error_bounds=coupling_transfer_errors,
        interaction_voltage_transfer_error_bounds=voltage_transfer_errors,
        interaction_coefficient_error_bounds=interaction_errors,
        difference_multiplicities=multiplicities,
        interaction_row_error_bounds=row_error_bounds,
        interaction_column_error_bounds=column_error_bounds,
        interaction_max_row_error_bound=maximum_row,
        interaction_max_column_error_bound=maximum_column,
        interaction_schur_error_bound=schur_bound,
        interaction_frobenius_error_bound=frobenius_bound,
        interaction_operator_error_bound=delta_r,
        absorber_operator_error_bound=delta_a,
        cap_scale_error_bound=delta_epsilon,
        cap_operator_error_bound=delta_b,
        fixed_linear_operator_error_bound=delta_h,
        finite_certificate=finite_certificate,
        absorber_route=(
            GalerkinFixedLinearAbsorberRoute.ANALYTIC_COSINE_SHELL_EXACT_DYADIC
        ),
        exact_geometry_target=_EXACT_GEOMETRY_TARGET,
        algebraic_geometry_realization=_ALGEBRAIC_GEOMETRY_REALIZATION,
        exact_interaction_target=_EXACT_INTERACTION_TARGET,
        algebraic_interaction_realization=(_ALGEBRAIC_INTERACTION_REALIZATION),
        absorber_target=_ABSORBER_TARGET,
        cap_target=_CAP_TARGET,
        error_scope=_ERROR_SCOPE,
        coefficient_norm=_COEFFICIENT_NORM,
    )
    return ledger


__all__: list[str] = ["build_galerkin_fixed_linear_error_ledger"]
