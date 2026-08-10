"""Tests for :mod:`ptyrodactyl.types.realization_types`.

Extended Summary
----------------
These tests verify canonical storage, explicit noncertificate values, and
structural and traced validation for the VC-1 realization carrier.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Dict

from ptyrodactyl.types import (
    GalerkinPotentialErrorRoute,
    GalerkinPotentialRealization,
    GalerkinPotentialRealizationMethod,
    create_galerkin_potential_realization,
    create_galerkin_product_support,
    create_potential_3d,
)
from ptyrodactyl.types.local_cell_types import GalerkinVoxelTargetRoute
from ptyrodactyl.types.realization_types import (
    GalerkinPotentialCertificateFailure,
    GalerkinPotentialCoefficientCertificate,
    create_galerkin_potential_coefficient_certificate,
)
from tests._galerkin_target_fixture import checked_acquisition

_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _factory_inputs() -> Dict[str, Any]:
    """Return one valid singleton-support realization input mapping."""
    zero = jnp.zeros((1, 3), dtype=jnp.int16)
    potential = create_potential_3d(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2),
        voxel_size=(1.0, 1.0, 1.0),
        box_size=(2.0, 2.0, 2.0),
        origin=(-0.2, 0.1, 0.3),
        producer="realization-carrier-test-v1",
        provenance_hash="c" * 64,
        coefficient_normalization="VC-1 mean DFT",
        band_limit=0.4,
    )
    shell = jnp.asarray(
        [
            (first, second, third)
            for first in range(-1, 2)
            for second in range(-1, 2)
            for third in range(-1, 2)
        ],
        dtype=jnp.int64,
    )
    support = create_galerkin_product_support(
        state_indices=zero,
        interaction_indices=zero,
        absorber_indices=shell,
        work_indices=shell,
        work_shape=(3, 3, 3),
    )
    support_eligibility = checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
    )
    inputs: Dict[str, Any] = {
        "potential": potential,
        "support_eligibility": support_eligibility,
        "voltage_coefficients": jnp.asarray(
            [1.5 - 0.25j],
            dtype=jnp.complex64,
        ),
        "coefficient_error_bounds": jnp.asarray(
            [0.125],
            dtype=jnp.float32,
        ),
        "voltage_operator_error_bound": jnp.asarray(
            0.25,
            dtype=jnp.float32,
        ),
        "omitted_voltage_l2_diagnostic": jnp.asarray(
            0.375,
            dtype=jnp.float32,
        ),
        "omitted_voltage_l2_upper_bound": jnp.asarray(
            0.5,
            dtype=jnp.float32,
        ),
        "method": (GalerkinPotentialRealizationMethod.PERIODIC_TRIGONOMETRIC),
        "error_route": GalerkinPotentialErrorRoute.TRIANGLE_FALLBACK,
        "output_coefficient_normalization": "SC.13b mean DFT",
        "endpoint_convention": "signed half-open without even Nyquist",
        "voxel_metric": "cell-volume-weighted real L2",
    }
    return inputs


def _coefficient_certificate(
    **overrides: object,
) -> GalerkinPotentialCoefficientCertificate:
    """Create one finite two-coefficient certificate fixture."""
    arguments: Dict[str, object] = {
        "exact_coefficient_real_lower_bounds": jnp.asarray([0.9, -0.2]),
        "exact_coefficient_real_upper_bounds": jnp.asarray([1.1, 0.3]),
        "exact_coefficient_imag_lower_bounds": jnp.asarray([-0.1, -0.4]),
        "exact_coefficient_imag_upper_bounds": jnp.asarray([0.2, 0.5]),
        "finite_certificate": jnp.asarray(True),
        "direct_term_count": jnp.asarray(64),
        "maximum_direct_terms": jnp.asarray(128),
        "failure": GalerkinPotentialCertificateFailure.NONE,
        "exact_target": "VC.8 periodic trigonometric interpolant",
        "arithmetic": "exact rational test arithmetic",
    }
    arguments.update(overrides)
    return create_galerkin_potential_coefficient_certificate(**arguments)


class TestCoefficientCertificateTypes:
    """Verify the direct certificate vocabulary and structural factory.

    :see: :class:`ptyrodactyl.types.GalerkinPotentialCertificateFailure`
    :see: :class:`ptyrodactyl.types.GalerkinPotentialCoefficientCertificate`
    :see: :func:`ptyrodactyl.types.\
create_galerkin_potential_coefficient_certificate`
    """

    def test_enum_values_and_exact_dtypes(self) -> None:
        """Freeze direct-route and typed-failure values."""
        certificate = _coefficient_certificate()
        jax.block_until_ready(certificate)

        assert (
            GalerkinPotentialErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL.value
            == "vc1_direct_pairwise_host_interval"
        )
        assert GalerkinPotentialCertificateFailure.NONE.value == "none"
        unsupported_failure = (
            GalerkinPotentialCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED
        )
        assert unsupported_failure.value == "host_arithmetic_unsupported"
        assert certificate.failure is GalerkinPotentialCertificateFailure.NONE
        assert certificate.exact_coefficient_real_lower_bounds.dtype == (
            jnp.float64
        )
        assert certificate.direct_term_count.dtype == jnp.int64
        assert certificate.finite_certificate.dtype == jnp.bool_

    def test_infinite_failure_is_a_typed_noncertificate(self) -> None:
        """Accept unbounded rectangles only with a non-none failure."""
        certificate = _coefficient_certificate(
            exact_coefficient_real_lower_bounds=jnp.asarray([-jnp.inf]),
            exact_coefficient_real_upper_bounds=jnp.asarray([jnp.inf]),
            exact_coefficient_imag_lower_bounds=jnp.asarray([-jnp.inf]),
            exact_coefficient_imag_upper_bounds=jnp.asarray([jnp.inf]),
            finite_certificate=jnp.asarray(False),
            failure=(GalerkinPotentialCertificateFailure.WORK_BUDGET_EXCEEDED),
        )
        jax.block_until_ready(certificate)

        assert not bool(certificate.finite_certificate)
        assert certificate.failure is (
            GalerkinPotentialCertificateFailure.WORK_BUDGET_EXCEEDED
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {
                "exact_coefficient_real_lower_bounds": jnp.asarray([2.0]),
                "exact_coefficient_real_upper_bounds": jnp.asarray([1.0]),
                "exact_coefficient_imag_lower_bounds": jnp.asarray([0.0]),
                "exact_coefficient_imag_upper_bounds": jnp.asarray([0.0]),
            },
            {
                "finite_certificate": jnp.asarray(True),
                "failure": (
                    GalerkinPotentialCertificateFailure.WORK_BUDGET_EXCEEDED
                ),
            },
            {"direct_term_count": jnp.asarray(129)},
            {"maximum_direct_terms": jnp.asarray(0)},
        ],
    )
    def test_factory_rejects_crossed_or_contradictory_evidence(
        self,
        overrides: Dict[str, object],
    ) -> None:
        """Reject crossed endpoints and inconsistent outcome metadata."""
        with pytest.raises(_RUNTIME_ERRORS, match="inconsistent"):
            certificate = _coefficient_certificate(**overrides)
            jax.block_until_ready(certificate)


class TestGalerkinPotentialRealization:
    """Verify the public VC-1 realization carrier vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinPotentialErrorRoute`
    :see: :class:`ptyrodactyl.types.GalerkinPotentialRealization`
    :see: :class:`ptyrodactyl.types.GalerkinPotentialRealizationMethod`
    :see: :func:`ptyrodactyl.types.create_galerkin_potential_realization`
    """

    def test_enum_values_freeze_method_and_error_route(self) -> None:
        """Freeze the finite-target and outward-error vocabulary."""
        assert (
            GalerkinPotentialRealizationMethod.PERIODIC_TRIGONOMETRIC.value
            == "vc1_periodic_trigonometric"
        )
        assert (
            GalerkinPotentialErrorRoute.TRIANGLE_FALLBACK.value
            == "vc1_triangle_fallback"
        )

    def test_factory_canonicalizes_arrays_and_preserves_bindings(self) -> None:
        """Store exact widths and retain the bound potential and support."""
        inputs = _factory_inputs()
        realization = create_galerkin_potential_realization(**inputs)
        jax.block_until_ready(realization)

        assert realization.potential is inputs["potential"]
        assert realization.support_eligibility is inputs["support_eligibility"]
        assert (
            realization.support
            is inputs["support_eligibility"].manifest.support
        )
        assert realization.voltage_coefficients.dtype == jnp.complex128
        assert realization.coefficient_error_bounds.dtype == jnp.float64
        assert realization.voltage_operator_error_bound.dtype == jnp.float64
        assert realization.omitted_voltage_l2_diagnostic.dtype == jnp.float64
        assert realization.omitted_voltage_l2_upper_bound.dtype == jnp.float64
        assert realization.voltage_coefficients.shape == (1,)
        assert realization.method is (
            GalerkinPotentialRealizationMethod.PERIODIC_TRIGONOMETRIC
        )
        assert realization.target_route is (
            GalerkinVoxelTargetRoute.TRIGONOMETRIC_VC1
        )
        assert realization.error_route is (
            GalerkinPotentialErrorRoute.TRIANGLE_FALLBACK
        )
        assert (
            realization.output_coefficient_normalization == "SC.13b mean DFT"
        )
        assert GalerkinPotentialRealization.__annotations__[
            "voltage_coefficients"
        ].dtypes == ("complex128",)
        for field_name in (
            "coefficient_error_bounds",
            "voltage_operator_error_bound",
            "omitted_voltage_l2_diagnostic",
            "omitted_voltage_l2_upper_bound",
        ):
            assert GalerkinPotentialRealization.__annotations__[
                field_name
            ].dtypes == ("float64",)

    def test_factory_is_jittable_for_fixed_source_and_support(self) -> None:
        """Compile dynamic coefficient and evidence leaves."""
        inputs = _factory_inputs()

        @jax.jit
        def build(
            coefficients: jax.Array,
            errors: jax.Array,
        ) -> GalerkinPotentialRealization:
            """Build one realization with two traced array inputs."""
            traced_inputs = dict(inputs)
            traced_inputs["voltage_coefficients"] = coefficients
            traced_inputs["coefficient_error_bounds"] = errors
            result = create_galerkin_potential_realization(**traced_inputs)
            return result

        realization = build(
            jnp.asarray([0.75 + 0.5j], dtype=jnp.complex64),
            jnp.asarray([0.0625], dtype=jnp.float32),
        )
        jax.block_until_ready(realization)

        assert realization.voltage_coefficients.dtype == jnp.complex128
        assert realization.coefficient_error_bounds.dtype == jnp.float64
        assert realization.voltage_coefficients[0] == 0.75 + 0.5j
        assert realization.coefficient_error_bounds[0] == 0.0625

    def test_infinite_bounds_remain_explicit_noncertificates(self) -> None:
        """Accept positive infinity only in fields documented as bounds."""
        inputs = _factory_inputs()
        inputs["coefficient_error_bounds"] = jnp.asarray([jnp.inf])
        inputs["voltage_operator_error_bound"] = jnp.asarray(jnp.inf)
        inputs["omitted_voltage_l2_upper_bound"] = jnp.asarray(jnp.inf)
        realization = create_galerkin_potential_realization(**inputs)
        jax.block_until_ready(realization)

        assert jnp.isinf(realization.coefficient_error_bounds[0])
        assert jnp.isinf(realization.voltage_operator_error_bound)
        assert jnp.isinf(realization.omitted_voltage_l2_upper_bound)
        assert jnp.isfinite(realization.omitted_voltage_l2_diagnostic)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            (
                {
                    "voltage_coefficients": jnp.ones(
                        (1, 1),
                        dtype=jnp.complex128,
                    )
                },
                "voltage_coefficients must be 1D",
            ),
            (
                {
                    "voltage_coefficients": jnp.ones(
                        (2,),
                        dtype=jnp.complex128,
                    )
                },
                "must match the interaction support",
            ),
            (
                {"coefficient_error_bounds": jnp.ones((2,))},
                "must match voltage_coefficients",
            ),
            ({"endpoint_convention": "  "}, "must be nonempty"),
            ({"voxel_metric": ""}, "must be nonempty"),
        ],
    )
    def test_factory_rejects_structural_mismatches(
        self,
        override: Dict[str, Any],
        message: str,
    ) -> None:
        """Reject inconsistent vector sizes and empty static identifiers."""
        inputs = _factory_inputs()
        inputs.update(override)
        with pytest.raises(ValueError, match=message):
            create_galerkin_potential_realization(**inputs)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "voltage_coefficients",
                jnp.asarray([jnp.nan + 0.0j]),
                "voltage_coefficients must be finite",
            ),
            (
                "coefficient_error_bounds",
                jnp.asarray([-0.1]),
                "coefficient_error_bounds must be non-negative",
            ),
            (
                "coefficient_error_bounds",
                jnp.asarray([jnp.nan]),
                "coefficient_error_bounds must be non-negative",
            ),
            (
                "voltage_operator_error_bound",
                jnp.asarray(-0.1),
                "voltage_operator_error_bound must be non-negative",
            ),
            (
                "omitted_voltage_l2_diagnostic",
                jnp.asarray(jnp.inf),
                "omitted_voltage_l2_diagnostic must be finite",
            ),
            (
                "omitted_voltage_l2_upper_bound",
                jnp.asarray(jnp.nan),
                "omitted_voltage_l2_upper_bound must be non-negative",
            ),
        ],
    )
    def test_factory_rejects_invalid_numeric_evidence(
        self,
        field: str,
        value: jax.Array,
        message: str,
    ) -> None:
        """Reject non-finite values except explicit infinite upper bounds."""
        inputs = _factory_inputs()
        inputs[field] = value
        with pytest.raises(_RUNTIME_ERRORS, match=message):
            realization = create_galerkin_potential_realization(**inputs)
            jax.block_until_ready(realization)
