"""PRIVATE: Provide shared private implementation infrastructure.

Extended Summary
----------------
This private package owns implementation primitives shared across public
operator families. The submodules are organized as follows:

- :mod:`canonical_digest`
    Canonical stored-value payloads and provenance digests.
- :mod:`host_interval`
    Exact-rational host interval arithmetic and outward conversion.
- :mod:`interval`
    FTZ-safe traced binary64 interval arithmetic.
- :mod:`numeric`
    Dtype-preserving floating-point range predicates.
- :mod:`physics`
    Canonically rounded scalar electron-physics formulas.

Unprefixed names form the package-internal shared seam. Underscored names
remain leaf-local implementation details, and ``ptyrodactyl`` does not
re-export this private package.

Routine Listings
----------------
:class:`RootEnclosureError`
    Identify an internal failure to enclose one rational-turn phase.
:func:`all_normal_arithmetic_supported`
    Combine every required normal-range arithmetic probe.
:func:`arithmetic_environment_probes`
    Probe normal primitives and gradual underflow separately.
:func:`array_payload`
    Build a canonical dtype-, shape-, and byte-bound payload.
:func:`coefficient_error_fraction`
    Bound one stored coefficient against an exact rectangle.
:func:`complex_rectangle_multiply`
    Multiply two exact rational complex rectangles.
:func:`conjugate_rectangle`
    Conjugate one exact complex rectangle.
:func:`coupled_interaction_value`
    Evaluate and canonically round the coupling and coupled coefficients.
:func:`downward_divide`
    Enclose one positive-denominator quotient from below.
:func:`downward_sqrt`
    Enclose one nonnegative square root from below.
:func:`fraction_from_float`
    Return the exact rational value of one finite binary64.
:func:`fraction_lower_float`
    Convert one rational endpoint toward minus infinity.
:func:`fraction_upper_float`
    Convert one rational endpoint toward plus infinity.
:func:`has_lost_nonzero_components`
    Detect a nonzero component mapped to zero or subnormal magnitude.
:func:`has_lost_subtraction`
    Detect a real or complex subtraction flushed from nonzero to zero.
:func:`has_nonzero_components`
    Detect any nonzero real or imaginary IEEE component bitwise.
:func:`has_subnormal_components`
    Return whether a real or complex array has a nonzero subnormal part.
:func:`helmholtz_coupling_value`
    Evaluate the 50-mantissa-bit canonical Helmholtz coupling.
:func:`host_array`
    Transfer one JAX array to a read-only host NumPy value.
:func:`host_binary64_supported`
    Probe every host-float property used by certificates.
:func:`interval_add`
    Outward-add two reusable real intervals.
:func:`interval_divide_positive`
    Outward-divide by one strictly positive real interval.
:func:`interval_multiply`
    Outward-multiply two reusable real intervals.
:func:`interval_sqrt`
    Outward-square-root one nonnegative real interval.
:func:`interval_square`
    Outward-square one reusable real interval.
:func:`interval_subtract`
    Outward-subtract two reusable real intervals.
:func:`mathematical_pi_interval`
    Enclose mathematical pi with guarded binary64 endpoints.
:func:`normalized_sinc_integer_ratio`
    Enclose ``sin(pi mode/count)/(pi mode/count)``.
:func:`pairwise_rectangle_sum`
    Sum rectangles through a deterministic binary reduction.
:func:`point_interval`
    Embed exact stored binary64 points through FTZ.2.
:func:`rational_turn_exponential`
    Enclose ``exp(-2 pi i turn)`` without library trig.
:func:`real_interval_product`
    Multiply two exact rational real intervals.
:func:`round_up`
    Widen a nearest binary64 point toward positive infinity.
:func:`scale_complex_rectangle`
    Multiply one complex rectangle by an exact real scalar.
:func:`sha256`
    Hash one canonical JSON payload as a provenance checksum.
:func:`sqrt_fraction_upper`
    Enclose one non-negative rational square root above.
:func:`stored_value_payload`
    Serialize one declared carrier value without properties.
:func:`upward_add`
    Enclose one exact-real endpoint addition from above.
:func:`upward_divide`
    Enclose one positive-denominator quotient from above.
:func:`upward_multiply`
    Enclose one exact-real endpoint product from above.
:func:`upward_sqrt`
    Enclose one nonnegative square root from above.
:obj:`ComplexRectangle`
    Represent exact rational bounds for both complex components.
:obj:`RationalInterval`
    Represent one exact rational interval by its lower and upper bounds.
:obj:`RealInterval`
    Represent one traced binary64 interval by its lower and upper arrays.
"""

from .canonical_digest import (
    array_payload,
    host_array,
    sha256,
    stored_value_payload,
)
from .host_interval import (
    ComplexRectangle,
    RationalInterval,
    RootEnclosureError,
    coefficient_error_fraction,
    complex_rectangle_multiply,
    conjugate_rectangle,
    fraction_from_float,
    fraction_lower_float,
    fraction_upper_float,
    host_binary64_supported,
    normalized_sinc_integer_ratio,
    pairwise_rectangle_sum,
    rational_turn_exponential,
    real_interval_product,
    scale_complex_rectangle,
    sqrt_fraction_upper,
)
from .interval import (
    RealInterval,
    all_normal_arithmetic_supported,
    arithmetic_environment_probes,
    downward_divide,
    downward_sqrt,
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
    upward_divide,
    upward_multiply,
    upward_sqrt,
)
from .numeric import (
    has_lost_nonzero_components,
    has_lost_subtraction,
    has_nonzero_components,
    has_subnormal_components,
)
from .physics import coupled_interaction_value, helmholtz_coupling_value

__all__: list[str] = [
    "all_normal_arithmetic_supported",
    "arithmetic_environment_probes",
    "array_payload",
    "coefficient_error_fraction",
    "complex_rectangle_multiply",
    "ComplexRectangle",
    "conjugate_rectangle",
    "coupled_interaction_value",
    "downward_divide",
    "downward_sqrt",
    "fraction_from_float",
    "fraction_lower_float",
    "fraction_upper_float",
    "has_lost_nonzero_components",
    "has_lost_subtraction",
    "has_nonzero_components",
    "has_subnormal_components",
    "helmholtz_coupling_value",
    "host_array",
    "host_binary64_supported",
    "interval_add",
    "interval_divide_positive",
    "interval_multiply",
    "interval_sqrt",
    "interval_square",
    "interval_subtract",
    "mathematical_pi_interval",
    "normalized_sinc_integer_ratio",
    "pairwise_rectangle_sum",
    "point_interval",
    "rational_turn_exponential",
    "RationalInterval",
    "real_interval_product",
    "RealInterval",
    "RootEnclosureError",
    "round_up",
    "scale_complex_rectangle",
    "sha256",
    "sqrt_fraction_upper",
    "stored_value_payload",
    "upward_add",
    "upward_divide",
    "upward_multiply",
    "upward_sqrt",
]
