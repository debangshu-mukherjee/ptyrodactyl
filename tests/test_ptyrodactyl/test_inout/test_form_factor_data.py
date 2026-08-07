"""Test bundled atomic form-factor table provenance and loading.

:see: :func:`ptyrodactyl.inout.kirkland_potentials`
:see: :func:`ptyrodactyl.inout.lobato_potentials`
"""

import hashlib
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ptyrodactyl.inout.xyz as xyz
from ptyrodactyl.inout import kirkland_potentials, lobato_potentials
from ptyrodactyl.types import A_BOHR


@pytest.mark.parametrize(
    ("asset_name", "expected_sha256"),
    [
        (
            "Lobato_van_Dyck.csv",
            "84fb662f029dda26488c8f85a9bdecf2ffa9adafad4c3fd7a16f3112110938c0",
        ),
        (
            "Kirkland_Potentials.csv",
            "79d0dd198a688ee5c3009db7deb9963abbbea2974734193e2ceb1ea07c16b226",
        ),
    ],
)
def test_form_factor_assets_match_pinned_provenance(
    asset_name: str,
    expected_sha256: str,
) -> None:
    """The shipped coefficient bytes remain the audited source tables."""
    asset = files("ptyrodactyl.inout").joinpath("luggage", asset_name)
    actual_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()

    assert actual_sha256 == expected_sha256


def test_tables_have_canonical_shapes_dtypes_values_and_cache_identity() -> (
    None
):
    """Both complete H--Lr tables are finite float64 arrays loaded once."""
    lobato = lobato_potentials()
    kirkland = kirkland_potentials()

    assert lobato.shape == (103, 10)
    assert kirkland.shape == (103, 12)
    assert lobato.dtype == kirkland.dtype == jnp.float64
    assert isinstance(lobato, jax.Array)
    assert isinstance(kirkland, jax.Array)
    assert lobato_potentials() is lobato
    assert kirkland_potentials() is kirkland
    assert bool(jnp.all(jnp.isfinite(lobato)))
    assert bool(jnp.all(jnp.isfinite(kirkland)))
    assert bool(jnp.all(lobato[:, 1::2] > 0.0))
    assert bool(jnp.all(kirkland[:, 1::2] > 0.0))


@pytest.mark.parametrize(
    ("atomic_number", "expected"),
    [
        (
            1,
            [
                0.00647384848835291,
                2.78519885379148,
                -0.490192576780229,
                2.77620428330644,
                0.573284160390876,
                2.77538591050625,
                -0.37940330148399,
                2.76759302867258,
                0.554426474774079,
                2.76511897642927,
            ],
        ),
        (
            14,
            [
                2.87189142611612,
                5.08487103642989,
                -2.06173501195173,
                0.429178185305126,
                2.17114024204478,
                0.366485434192162,
                -0.0663073633058801,
                0.119710611296903,
                0.00301070709670513,
                0.0143994536128397,
            ],
        ),
        (
            79,
            [
                1.6759346706487,
                5.52231093211402,
                3.00486602969729,
                1.38007223007196,
                0.595340013161635,
                0.162229237655945,
                0.0117163186623094,
                0.00901814890416575,
                4.296782976398e-05,
                0.00037927766747767,
            ],
        ),
    ],
)
def test_lobato_rows_match_published_provenance_anchors(
    atomic_number: int,
    expected: list[float],
) -> None:
    """Hardcoded H, Si, and Au rows pin the cited 2014 coefficient table."""
    actual = np.asarray(lobato_potentials()[atomic_number - 1])
    np.testing.assert_array_equal(
        actual, np.asarray(expected, dtype=np.float64)
    )


@pytest.mark.parametrize(
    ("atomic_number", "expected"),
    [
        (
            1,
            [
                0.0355221981,
                0.225354459,
                0.0262782423,
                0.225354636,
                0.0352695173,
                0.225355749,
                0.0677755867,
                4.38850114,
                0.00356601775,
                0.40388115,
                0.0276131055,
                1.44488619,
            ],
        ),
        (
            79,
            [
                0.870155469,
                0.162787604,
                3.7298545,
                13.0045249,
                2.83853565,
                0.652969096,
                0.0148964674,
                0.0275909732,
                0.298381945,
                0.163263715,
                0.282816664,
                1.3849048,
            ],
        ),
    ],
)
def test_kirkland_rows_match_reference_provenance_anchors(
    atomic_number: int,
    expected: list[float],
) -> None:
    """Hardcoded H and Au rows pin the retained Kirkland reference table."""
    actual = np.asarray(kirkland_potentials()[atomic_number - 1])
    np.testing.assert_array_equal(
        actual, np.asarray(expected, dtype=np.float64)
    )


def test_lobato_all_element_bethe_normalization() -> None:
    """Every row has the required bare-Coulomb high-q coefficient."""
    table = np.asarray(lobato_potentials())
    amplitudes = table[:, 0::2]
    scales = table[:, 1::2]
    actual = np.sum(amplitudes / scales, axis=1)
    atomic_numbers = np.arange(1, 104, dtype=np.float64)
    expected = atomic_numbers / (2.0 * np.pi**2 * float(A_BOHR))

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=0.0)


def test_tables_support_jit_indexing_and_coefficient_gradients() -> None:
    """Moving table ownership preserves JAX indexing and gradients."""
    table = kirkland_potentials()
    selected = jax.jit(lambda values, index: values[index])(
        table,
        jnp.asarray(5, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(table[5]))

    weights = jnp.linspace(0.5, 1.5, 12, dtype=jnp.float64)
    gradient = jax.grad(lambda values: jnp.dot(values, table[5]))(weights)
    np.testing.assert_array_equal(np.asarray(gradient), np.asarray(table[5]))


def test_xyz_module_no_longer_owns_form_factor_data() -> None:
    """XYZ exposes neither form-factor loaders nor cached table aliases."""
    assert not hasattr(xyz, "kirkland_potentials")
    assert not hasattr(xyz, "_KIRKLAND_POTENTIALS")
    assert "kirkland_potentials" not in xyz.__all__
