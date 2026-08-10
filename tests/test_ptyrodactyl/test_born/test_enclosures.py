"""Tests for :mod:`ptyrodactyl.born.enclosures`.

Extended Summary
----------------
These tests compare the RM-S2 fixed-linear enclosure with independent
high-precision SC-1 kinematics and explicitly assembled dense matrices.
"""

import importlib
from decimal import Decimal, getcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import Dict, Tuple
from numpy.testing import assert_allclose, assert_array_equal

from ptyrodactyl._physics import coupled_interaction_value
from ptyrodactyl.born.enclosures import (
    _ANGSTROM_SQUARED_LOWER,
    _exact_kinematic_intervals,
    build_galerkin_fixed_linear_error_ledger,
)
from ptyrodactyl.types.constants import (
    C_LIGHT,
    E_CHARGE,
    H_PLANCK,
    HBAR,
    M_E,
)

getcontext().prec = 110

_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944"
    "5923078164062862089986280348253421170679"
)
_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _decimal(value: float | jax.Array) -> Decimal:
    """Return the exact decimal value of one stored binary64 scalar."""
    return Decimal.from_float(float(value))


def _stored_wavenumber(voltage_kv: jax.Array) -> jax.Array:
    """Reproduce the current frozen Planck-form binary64 wavenumber."""
    energy = voltage_kv * 1000.0 * jnp.asarray(E_CHARGE)
    wavelength_metre = jnp.sqrt(
        (jnp.asarray(H_PLANCK) * jnp.asarray(C_LIGHT)) ** 2
        / (
            energy
            * (2.0 * jnp.asarray(M_E) * jnp.asarray(C_LIGHT) ** 2 + energy)
        )
    )
    return 2.0 * jnp.pi / (1.0e10 * wavelength_metre)


def _factory_inputs(*, zero_potential: bool = False) -> Dict[str, jax.Array]:
    """Return one canonical one-dimensional interaction fixture."""
    state = jnp.asarray(
        ((-1, 0, 0), (0, 0, 0), (1, 0, 0)),
        dtype=jnp.int64,
    )
    interaction_indices = jnp.asarray(
        ((-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)),
        dtype=jnp.int64,
    )
    if zero_potential:
        voltage_coefficients = jnp.zeros((5,), dtype=jnp.complex128)
        coefficient_errors = jnp.zeros((5,), dtype=jnp.float64)
    else:
        voltage_coefficients = jnp.asarray(
            (
                0.015 + 0.002j,
                0.04 - 0.01j,
                0.3 + 0.0j,
                0.04 + 0.01j,
                0.015 - 0.002j,
            ),
            dtype=jnp.complex128,
        )
        coefficient_errors = jnp.asarray(
            (2.0e-13, 3.0e-13, 5.0e-13, 3.0e-13, 2.0e-13),
            dtype=jnp.float64,
        )
    voltage_kv = jnp.asarray(300.0, dtype=jnp.float64)
    coupling, interaction = coupled_interaction_value(
        voltage_coefficients,
        voltage_kv,
        M_E,
        E_CHARGE,
        C_LIGHT,
        H_PLANCK,
    )
    wavenumber = _stored_wavenumber(voltage_kv)
    direction = jnp.asarray((0.071, -0.113, 1.0), dtype=jnp.float64)
    carrier = wavenumber * direction / jnp.linalg.norm(direction)
    return {
        "state_indices": state,
        "interaction_indices": interaction_indices,
        "voltage_coefficients": voltage_coefficients,
        "voltage_coefficient_error_bounds": coefficient_errors,
        "interaction_coupling": coupling,
        "interaction_coefficients": interaction,
        "accelerating_voltage_kv": voltage_kv,
        "carrier": carrier,
        "box_lengths": jnp.asarray((11.0, 13.0, 17.0), dtype=jnp.float64),
        "wavenumber": wavenumber,
        "cap_scale": jnp.asarray(0.4, dtype=jnp.float64),
    }


def _exact_decimal_kinematics(
    inputs: Dict[str, jax.Array],
) -> Tuple[Decimal, Decimal, Tuple[Decimal, Decimal, Decimal]]:
    """Evaluate exact SC.2, SC.4, and normalized SC.8 with Decimal."""
    mass = _decimal(M_E)
    charge = _decimal(E_CHARGE)
    speed = _decimal(C_LIGHT)
    hbar = _decimal(HBAR)
    voltage = _decimal(inputs["accelerating_voltage_kv"]) * Decimal(1000)
    prefactor = Decimal(2) * mass * charge / (hbar * hbar)
    rest_energy = mass * speed * speed
    wavenumber_squared = (
        prefactor
        * voltage
        * (Decimal(1) + charge * voltage / (Decimal(2) * rest_energy))
        * Decimal("1e-20")
    )
    wavenumber = wavenumber_squared.sqrt()
    coupling = (
        prefactor
        * (Decimal(1) + charge * voltage / rest_energy)
        * Decimal("1e-20")
    )
    seed = tuple(_decimal(value) for value in inputs["carrier"])
    norm = sum(
        (value * value for value in seed),
        start=Decimal(0),
    ).sqrt()
    carrier = (
        wavenumber * seed[0] / norm,
        wavenumber * seed[1] / norm,
        wavenumber * seed[2] / norm,
    )
    return wavenumber, coupling, carrier


def _exact_decimal_diagonal(
    inputs: Dict[str, jax.Array],
    exact_carrier: Tuple[Decimal, Decimal, Decimal],
) -> list[Decimal]:
    """Evaluate the exact cancellation-safe on-shell SC.23 diagonal."""
    box = tuple(_decimal(value) for value in inputs["box_lengths"])
    result: list[Decimal] = []
    for index in np.asarray(inputs["state_indices"]):
        offset = tuple(
            Decimal(2) * _PI * Decimal(int(component)) / length
            for component, length in zip(index, box, strict=True)
        )
        value = Decimal(2) * sum(
            carrier * wavevector
            for carrier, wavevector in zip(
                exact_carrier,
                offset,
                strict=True,
            )
        ) + sum(wavevector * wavevector for wavevector in offset)
        result.append(value)
    return result


def _decimal_complex_error(
    stored: complex,
    exact_real: Decimal,
    exact_imaginary: Decimal,
) -> Decimal:
    """Return one high-precision complex absolute difference."""
    real_error = Decimal.from_float(stored.real) - exact_real
    imaginary_error = Decimal.from_float(stored.imag) - exact_imaginary
    return (real_error * real_error + imaginary_error * imaginary_error).sqrt()


class TestGalerkinFixedLinearEnclosure:
    """Verify independent RM-S2 component and matrix bounds.

    :see: :func:`ptyrodactyl.born.build_galerkin_fixed_linear_error_ledger`
    """

    def test_exact_si_to_angstrom_conversion_is_bracketed(self) -> None:
        """Treat mathematical ``10^-20`` as a conversion, not an input."""
        lower_float = _ANGSTROM_SQUARED_LOWER
        upper_float = np.nextafter(lower_float, np.inf)

        assert Decimal.from_float(lower_float) < Decimal("1e-20")
        assert Decimal.from_float(upper_float) > Decimal("1e-20")

    def test_high_precision_exact_physics_and_dense_matrix_are_enclosed(
        self,
    ) -> None:
        """Never underbound SC.2/SC.4/SC.8 or the dense fixed matrix."""
        inputs = _factory_inputs()
        ledger = build_galerkin_fixed_linear_error_ledger(**inputs)
        jax.block_until_ready(ledger)
        exact_k0, exact_sigma, exact_carrier = _exact_decimal_kinematics(
            inputs
        )
        exact_diagonal = _exact_decimal_diagonal(inputs, exact_carrier)

        assert _decimal(ledger.exact_wavenumber_lower_bound) <= exact_k0
        assert _decimal(ledger.exact_wavenumber_upper_bound) >= exact_k0
        assert _decimal(ledger.wavenumber_error_bound) >= abs(
            _decimal(inputs["wavenumber"]) - exact_k0
        )
        assert (
            _decimal(ledger.exact_interaction_coupling_lower_bound)
            <= exact_sigma
        )
        assert (
            _decimal(ledger.exact_interaction_coupling_upper_bound)
            >= exact_sigma
        )
        assert _decimal(ledger.interaction_coupling_error_bound) >= abs(
            _decimal(inputs["interaction_coupling"]) - exact_sigma
        )
        for axis, exact_value in enumerate(exact_carrier):
            assert (
                _decimal(ledger.exact_carrier_lower_bounds[axis])
                <= exact_value
            )
            assert (
                _decimal(ledger.exact_carrier_upper_bounds[axis])
                >= exact_value
            )
            assert _decimal(
                ledger.carrier_component_error_bounds[axis]
            ) >= abs(_decimal(inputs["carrier"][axis]) - exact_value)
        for position, exact_value in enumerate(exact_diagonal):
            assert (
                _decimal(ledger.exact_free_diagonal_lower_bounds[position])
                <= exact_value
            )
            assert (
                _decimal(ledger.exact_free_diagonal_upper_bounds[position])
                >= exact_value
            )
            assert _decimal(
                ledger.free_diagonal_error_bounds[position]
            ) >= abs(
                _decimal(ledger.algebraic_free_diagonal[position])
                - exact_value
            )

        stored_voltage = np.asarray(inputs["voltage_coefficients"])
        stored_interaction = np.asarray(inputs["interaction_coefficients"])
        exact_voltage: list[Tuple[Decimal, Decimal]] = [
            (
                Decimal.from_float(value.real),
                Decimal.from_float(value.imag),
            )
            for value in stored_voltage
        ]
        # Exercise a genuinely nonzero VC-1 error at the origin coefficient.
        exact_voltage[2] = (
            exact_voltage[2][0] + Decimal("2e-13"),
            exact_voltage[2][1],
        )
        component_errors: list[Decimal] = []
        exact_interaction: list[complex] = []
        for stored, (voltage_real, voltage_imaginary) in zip(
            stored_interaction,
            exact_voltage,
            strict=True,
        ):
            exact_real = exact_sigma * voltage_real
            exact_imaginary = exact_sigma * voltage_imaginary
            component_errors.append(
                _decimal_complex_error(stored, exact_real, exact_imaginary)
            )
            exact_interaction.append(
                complex(float(exact_real), float(exact_imaginary))
            )
        assert inputs["voltage_coefficient_error_bounds"][2] > 0.0
        for actual, upper in zip(
            component_errors,
            ledger.interaction_coefficient_error_bounds,
            strict=True,
        ):
            assert _decimal(upper) >= actual

        state = np.asarray(inputs["state_indices"])
        coefficient_map = {
            tuple(index): value
            for index, value in zip(
                np.asarray(inputs["interaction_indices"]),
                exact_interaction,
                strict=True,
            )
        }
        stored_map = {
            tuple(index): value
            for index, value in zip(
                np.asarray(inputs["interaction_indices"]),
                stored_interaction,
                strict=True,
            )
        }
        interaction_error_matrix = np.asarray(
            [
                [
                    stored_map[tuple(row - column)]
                    - coefficient_map[tuple(row - column)]
                    for column in state
                ]
                for row in state
            ],
            dtype=np.complex128,
        )
        free_error = np.asarray(
            [
                float(_decimal(value) - exact)
                for value, exact in zip(
                    ledger.algebraic_free_diagonal,
                    exact_diagonal,
                    strict=True,
                )
            ]
        )
        full_error = np.diag(free_error) - interaction_error_matrix
        dense_interaction_norm = np.linalg.norm(interaction_error_matrix, 2)
        dense_full_norm = np.linalg.norm(full_error, 2)

        assert_array_equal(ledger.difference_multiplicities, [1, 2, 3, 2, 1])
        assert float(
            ledger.interaction_frobenius_error_bound
        ) >= np.linalg.norm(
            interaction_error_matrix,
            "fro",
        )
        assert float(ledger.interaction_schur_error_bound) >= np.sqrt(
            np.linalg.norm(interaction_error_matrix, np.inf)
            * np.linalg.norm(interaction_error_matrix, 1)
        )
        assert (
            float(ledger.interaction_operator_error_bound)
            >= dense_interaction_norm
        )
        assert (
            float(ledger.fixed_linear_operator_error_bound) >= dense_full_norm
        )
        assert bool(ledger.finite_certificate)

    def test_zero_potential_has_exactly_zero_interaction_ledger(self) -> None:
        """Do not invent interaction error when both VC-1 terms are zero."""
        ledger = build_galerkin_fixed_linear_error_ledger(
            **_factory_inputs(zero_potential=True)
        )
        jax.block_until_ready(ledger)

        assert_allclose(
            ledger.interaction_coefficient_error_bounds,
            0.0,
            atol=0.0,
        )
        assert ledger.interaction_operator_error_bound == 0.0
        assert (
            ledger.fixed_linear_operator_error_bound
            == ledger.free_operator_error_bound
        )
        assert ledger.absorber_operator_error_bound == 0.0
        assert ledger.cap_scale_error_bound == 0.0
        assert ledger.cap_operator_error_bound == 0.0

    def test_infinite_used_vc_error_propagates_as_typed_noncertificate(
        self,
    ) -> None:
        """Propagate infinity without converting it into NaN or a pass."""
        inputs = _factory_inputs()
        inputs["voltage_coefficient_error_bounds"] = (
            inputs["voltage_coefficient_error_bounds"].at[2].set(jnp.inf)
        )
        ledger = build_galerkin_fixed_linear_error_ledger(**inputs)
        jax.block_until_ready(ledger)

        assert jnp.isinf(ledger.interaction_coefficient_error_bounds[2])
        assert jnp.isinf(ledger.interaction_max_row_error_bound)
        assert jnp.isinf(ledger.interaction_max_column_error_bound)
        assert jnp.isinf(ledger.interaction_frobenius_error_bound)
        assert jnp.isinf(ledger.interaction_operator_error_bound)
        assert jnp.isinf(ledger.fixed_linear_operator_error_bound)
        assert not jnp.any(jnp.isnan(ledger.interaction_row_error_bounds))
        assert not bool(ledger.finite_certificate)

    def test_certificate_tangents_are_zero_but_coefficients_remain_live(
        self,
    ) -> None:
        """Keep proof arithmetic outside the differentiable physical map."""
        inputs = _factory_inputs()
        voltage_kv = inputs["accelerating_voltage_kv"]

        def voltage_and_wavenumber_bound(value):
            wavenumber, _, _ = _exact_kinematic_intervals(
                value,
                inputs["carrier"],
            )
            return value, wavenumber[1]

        eager = voltage_and_wavenumber_bound(voltage_kv)
        compiled = jax.jit(voltage_and_wavenumber_bound)(voltage_kv)
        _, tangent = jax.jvp(
            voltage_and_wavenumber_bound,
            (voltage_kv,),
            (jnp.ones_like(voltage_kv),),
        )

        assert_allclose(compiled[0], eager[0], atol=0.0)
        assert compiled[1] == eager[1]
        assert_allclose(tangent[0], 1.0, atol=0.0)
        assert tangent[1] == 0.0

    def test_unsupported_normal_arithmetic_is_a_noncertificate(
        self,
        monkeypatch,
    ) -> None:
        """Expose required-probe failure as an infinite typed bound."""
        interval_core = importlib.import_module("ptyrodactyl._interval")

        def unsupported_normal_arithmetic() -> jax.Array:
            return jnp.asarray(False)

        monkeypatch.setattr(
            interval_core,
            "_all_normal_arithmetic_supported",
            unsupported_normal_arithmetic,
        )
        ledger = build_galerkin_fixed_linear_error_ledger(**_factory_inputs())
        jax.block_until_ready(ledger)

        assert jnp.isinf(ledger.fixed_linear_operator_error_bound)
        assert not bool(ledger.finite_certificate)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            (
                {"carrier": jnp.zeros((3,), dtype=jnp.float64)},
                "carrier must be finite and nonzero",
            ),
            (
                {
                    "voltage_coefficient_error_bounds": jnp.asarray(
                        (-1.0, 0.0, 0.0, 0.0, 0.0),
                        dtype=jnp.float64,
                    )
                },
                "must be non-negative and not NaN",
            ),
            (
                {"interaction_coupling": jnp.asarray(1.0)},
                "must equal the canonical 50-bit realization",
            ),
        ],
    )
    def test_invalid_or_noncanonical_inputs_fail_closed(
        self,
        override: Dict[str, jax.Array],
        message: str,
    ) -> None:
        """Reject structural invalidity and forged canonical coefficients."""
        inputs = _factory_inputs()
        inputs.update(override)
        with pytest.raises(_RUNTIME_ERRORS, match=message):
            result = build_galerkin_fixed_linear_error_ledger(**inputs)
            jax.block_until_ready(result)
