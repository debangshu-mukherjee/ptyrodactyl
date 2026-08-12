"""Tests for :mod:`ptyrodactyl.types.local_cell_types`.

Extended Summary
----------------
These tests freeze the disjoint LVT-1 carrier vocabulary, exact binary64
storage, metadata-only producer bandwidth, and factory validation. They also
prevent the local-cell target from becoming a relabelled VC-1 point-sample
carrier.
"""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Dict

from ptyrodactyl.galerkin.local_cell import (
    realize_local_cell_galerkin_potential,
)
from ptyrodactyl.types.born_potential_types import (
    create_galerkin_product_support,
)
from ptyrodactyl.types.local_cell_types import (
    GalerkinLocalCellCertificateFailure,
    GalerkinLocalCellCoefficientCertificate,
    GalerkinLocalCellErrorRoute,
    GalerkinLocalCellPotentialRealization,
    GalerkinLocalCellTailEnclosure,
    GalerkinLocalCellTailFailure,
    GalerkinVoxelTargetRoute,
    LocalCellPotential3D,
    create_local_cell_potential_3d,
)
from ptyrodactyl.types.potential_types import Potential3D
from ptyrodactyl.types.realization_types import GalerkinPotentialRealization
from tests._galerkin_target_fixture import checked_acquisition

_PROVENANCE = "a" * 64
_RUNTIME_ERRORS = (
    eqx.EquinoxRuntimeError,
    jax.errors.JaxRuntimeError,
    ValueError,
)


def _local_potential(
    *,
    producer_bandwidth: float = 1.0e6,
) -> LocalCellPotential3D:
    """Create one anisotropic local-cell target with a shifted origin."""
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    potential: LocalCellPotential3D = create_local_cell_potential_3d(
        values,
        cell_size=(0.5, 0.75, 1.25),
        box_size=(2.0, 2.25, 2.5),
        cell_center_origin=(0.125, -0.375, 0.625),
        reference_semantics="declared local-cell test vacuum reference",
        producer="local-cell-carrier-test-v1",
        provenance_hash=f"sha256:{_PROVENANCE}",
        producer_coefficient_normalization="producer diagnostic only",
        producer_bandwidth=producer_bandwidth,
    )
    return potential


def _singleton_eligibility(potential: LocalCellPotential3D):
    """Create one eligible singleton interaction support."""
    zero = jnp.zeros((1, 3), dtype=jnp.int64)
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
    return checked_acquisition(
        support,
        potential.box_size,
        terminal_axis=2,
    )


class TestLocalCellRouteTypes:
    """Freeze the explicitly disjoint finite-target route vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinLocalCellErrorRoute`
    :see: :class:`ptyrodactyl.types.GalerkinVoxelTargetRoute`
    """

    def test_route_values_are_nonlegacy_and_nonoptional(self) -> None:
        """Give VC-1 and LVT-1 distinct mandatory identities."""
        assert GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1.value == (
            "local_cell_lvt1"
        )
        assert GalerkinVoxelTargetRoute.TRIGONOMETRIC_VC1.value == (
            "trigonometric_vc1"
        )
        assert GalerkinLocalCellErrorRoute.TRIANGLE_FALLBACK.value == (
            "lvt1_triangle_fallback"
        )
        assert (
            GalerkinLocalCellErrorRoute.DIRECT_PAIRWISE_HOST_INTERVAL.value
            == "lvt1_direct_pairwise_host_interval"
        )

    def test_no_public_raw_realization_factory_exists(self) -> None:
        """Reserve semantic minting for the checked numerical builder."""
        module = importlib.import_module("ptyrodactyl.types.local_cell_types")

        assert "create_galerkin_local_cell_potential_realization" not in (
            module.__all__
        )
        assert not hasattr(
            module,
            "create_galerkin_local_cell_potential_realization",
        )


class TestLocalCellPotential3D:
    """Verify the exact local-cell source carrier and factory.

    :see: :class:`ptyrodactyl.types.LocalCellPotential3D`
    :see: :func:`ptyrodactyl.types.create_local_cell_potential_3d`
    """

    def test_factory_stores_cell_semantics_and_binary64_values(self) -> None:
        """Canonicalize geometry while preserving an explicit cell target."""
        potential = _local_potential()
        jax.block_until_ready(potential)

        assert potential.cell_values.dtype == jnp.float64
        assert potential.cell_values.shape == (2, 3, 4)
        assert potential.cell_size == (0.5, 0.75, 1.25)
        assert potential.box_size == (2.0, 2.25, 2.5)
        assert potential.cell_center_origin == (0.125, 1.875, 0.625)
        assert potential.provenance_hash == _PROVENANCE
        assert potential.target_route is (
            GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1
        )
        assert "constant on each" in potential.cell_value_semantics
        assert "half-open" in potential.cell_support_convention
        assert "metadata only" in potential.producer_bandwidth_role
        assert "LVT.7" in potential.coefficient_formula
        assert LocalCellPotential3D.__annotations__["cell_values"].dtypes == (
            "float64",
        )

    def test_target_is_not_duck_compatible_with_vc1(self) -> None:
        """Prevent implicit point-sample relabelling at the API boundary."""
        potential = _local_potential()

        assert not isinstance(potential, Potential3D)
        assert not hasattr(potential, "volume")
        assert not hasattr(potential, "voxel_size")
        assert not hasattr(potential, "origin")
        assert not hasattr(potential, "band_limit")

    def test_producer_bandwidth_is_positive_metadata_not_a_gate(self) -> None:
        """Accept bandwidth far beyond every sampled-grid Nyquist value."""
        potential = _local_potential(producer_bandwidth=1.0e12)

        assert potential.producer_bandwidth == 1.0e12
        assert potential.producer_bandwidth_role == (
            "producer metadata only; not an LVT-1 coefficient cutoff"
        )

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"cell_values": jnp.ones((2, 2), dtype=jnp.float64)}, "shape"),
            (
                {"cell_values": jnp.ones((2, 2, 2), dtype=jnp.complex128)},
                "real voltages",
            ),
            ({"cell_size": (0.5, 0.5, 0.6)}, "must equal"),
            ({"units": "eV"}, "exactly 'V'"),
            ({"reference_semantics": "unknown"}, "physical reference"),
            ({"boundary": "open"}, "exactly 'periodic'"),
            ({"provenance_hash": "bad"}, "SHA-256"),
            ({"producer_bandwidth": 0.0}, "positive and finite"),
            ({"producer_bandwidth": jnp.inf}, "positive and finite"),
        ],
    )
    def test_factory_rejects_invalid_structure_or_metadata(
        self,
        overrides: Dict[str, object],
        match: str,
    ) -> None:
        """Reject ambiguous geometry and target identity declarations."""
        arguments: Dict[str, object] = {
            "cell_values": jnp.ones((2, 2, 2), dtype=jnp.float64),
            "cell_size": (0.5, 0.5, 0.5),
            "box_size": (1.0, 1.0, 1.0),
            "cell_center_origin": (0.0, 0.0, 0.0),
            "reference_semantics": "declared validation-test reference",
            "producer": "validation-test-v1",
            "provenance_hash": _PROVENANCE,
            "producer_coefficient_normalization": "producer metadata",
            "producer_bandwidth": 1.0,
        }
        arguments.update(overrides)

        with pytest.raises(_RUNTIME_ERRORS, match=match):
            potential = create_local_cell_potential_3d(**arguments)
            jax.block_until_ready(potential)

    def test_factory_rejects_nonfinite_dynamic_values_under_jit(self) -> None:
        """Keep runtime validation live after JAX tracing."""

        @jax.jit
        def build(values: jax.Array) -> jax.Array:
            potential = create_local_cell_potential_3d(
                values,
                cell_size=(0.5, 0.5, 0.5),
                box_size=(1.0, 1.0, 1.0),
                cell_center_origin=(0.0, 0.0, 0.0),
                reference_semantics="declared traced-test reference",
                producer="traced-validation-test-v1",
                provenance_hash=_PROVENANCE,
                producer_coefficient_normalization="producer metadata",
                producer_bandwidth=1.0,
            )
            return potential.cell_values

        with pytest.raises(_RUNTIME_ERRORS, match="non-finite"):
            result = build(jnp.full((2, 2, 2), jnp.nan))
            jax.block_until_ready(result)


class TestLocalCellRealizationTypes:
    """Verify the LVT-1 realization carrier remains disjoint and typed.

    :see: :class:`ptyrodactyl.types.GalerkinLocalCellPotentialRealization`
    """

    def test_checked_builder_mints_the_only_claimed_realization(self) -> None:
        """Obtain the route through recomputation, not arbitrary payloads."""
        potential = _local_potential()
        eligibility = _singleton_eligibility(potential)
        realization = realize_local_cell_galerkin_potential(
            potential,
            eligibility,
        )
        jax.block_until_ready(realization)

        assert isinstance(realization, GalerkinLocalCellPotentialRealization)
        assert not isinstance(realization, GalerkinPotentialRealization)
        assert jnp.array_equal(
            realization.local_potential.cell_values,
            potential.cell_values,
        )
        assert realization.local_potential.box_size == potential.box_size
        assert jnp.array_equal(
            realization.support.interaction_indices,
            eligibility.manifest.support.interaction_indices,
        )
        assert realization.voltage_coefficients.dtype == jnp.complex128
        assert realization.coefficient_error_bounds.dtype == jnp.float64
        assert realization.target_route is (
            GalerkinVoxelTargetRoute.LOCAL_CELL_LVT1
        )
        assert realization.error_route is (
            GalerkinLocalCellErrorRoute.TRIANGLE_FALLBACK
        )
        assert realization.coefficient_certificate is None
        assert not hasattr(realization, "potential")
        assert GalerkinLocalCellPotentialRealization.__annotations__[
            "voltage_coefficients"
        ].dtypes == ("complex128",)
        assert GalerkinLocalCellPotentialRealization.__annotations__[
            "coefficient_error_bounds"
        ].dtypes == ("float64",)


class TestLocalCellCoefficientCertificateTypes:
    """Freeze disjoint LVT.13 certificate vocabulary and exact dtypes.

    :see: :class:`ptyrodactyl.types.\
GalerkinLocalCellCertificateFailure`
    :see: :class:`ptyrodactyl.types.\
GalerkinLocalCellCoefficientCertificate`
    """

    def test_failure_values_and_evidence_dtypes_are_stable(self) -> None:
        """Keep typed noncertificates separate from finite evidence."""
        assert GalerkinLocalCellCertificateFailure.NONE.value == "none"
        assert (
            GalerkinLocalCellCertificateFailure.WORK_BUDGET_EXCEEDED.value
            == "work_budget_exceeded"
        )
        assert (
            GalerkinLocalCellCertificateFailure.HOST_ARITHMETIC_UNSUPPORTED.value
            == "host_arithmetic_unsupported"
        )
        annotations = GalerkinLocalCellCoefficientCertificate.__annotations__
        assert annotations["exact_coefficient_real_lower_bounds"].dtypes == (
            "float64",
        )
        assert annotations["finite_certificate"].dtypes == ("bool", "bool_")
        assert annotations["direct_term_count"].dtypes == ("int64",)

    def test_no_public_raw_certificate_factory_exists(self) -> None:
        """Reserve direct semantic minting for the host certifier."""
        module = importlib.import_module("ptyrodactyl.types.local_cell_types")

        assert "create_local_cell_coefficient_certificate" not in (
            module.__all__
        )
        assert not hasattr(module, "create_local_cell_coefficient_certificate")


class TestLocalCellTailEnclosureTypes:
    """Freeze the separate LVT.9 evidence carrier and failure vocabulary.

    :see: :class:`ptyrodactyl.types.GalerkinLocalCellTailEnclosure`
    :see: :class:`ptyrodactyl.types.GalerkinLocalCellTailFailure`
    """

    def test_tail_failure_values_and_interval_dtypes_are_stable(self) -> None:
        """Keep finite, parent-failed, and local-failed outcomes distinct."""
        assert GalerkinLocalCellTailFailure.NONE.value == "none"
        assert (
            GalerkinLocalCellTailFailure.PARENT_CERTIFICATE_NOT_FINITE.value
            == "parent_certificate_not_finite"
        )
        assert (
            GalerkinLocalCellTailFailure.PARSEVAL_CONTRADICTION.value
            == "parseval_contradiction"
        )
        annotations = GalerkinLocalCellTailEnclosure.__annotations__
        assert annotations["squared_tail_lower_bound"].dtypes == ("float64",)
        assert annotations["squared_tail_upper_bound"].dtypes == ("float64",)
        assert annotations["tail_l2_lower_bound"].dtypes == ("float64",)
        assert annotations["tail_l2_upper_bound"].dtypes == ("float64",)
        assert annotations["finite_enclosure"].dtypes == ("bool", "bool_")

    def test_no_public_raw_tail_factory_exists(self) -> None:
        """Reserve LVT.9 semantic minting for the authenticated host action."""
        module = importlib.import_module("ptyrodactyl.types.local_cell_types")

        assert "create_local_cell_tail_enclosure" not in module.__all__
        assert not hasattr(module, "create_local_cell_tail_enclosure")
