"""Pytest configuration for the ptyrodactyl test suite.

Extended Summary
----------------
Suite-wide configuration only: float64 discipline verification and shared
random-key handling. Test layout mirrors ``src/``: every source module
``src/ptyrodactyl/<pkg>/<mod>.py`` has its counterpart
``tests/test_ptyrodactyl/test_<pkg>/test_<mod>.py``; ``tests/test_data/``
holds fixture data and pinned regression references.
"""

import jax
import pytest


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Assert the x64 discipline the package enables at import time.

    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object (unused beyond the hook contract).
    """
    import ptyrodactyl  # noqa: F401  (import side effect: enables x64)

    assert jax.config.read("jax_enable_x64"), (
        "ptyrodactyl tests require jax_enable_x64"
    )


@pytest.fixture
def rng_key() -> jax.Array:
    """Return a fixed PRNG key for deterministic tests.

    Returns
    -------
    key : jax.Array
        A ``jax.random.PRNGKey(0)`` key; split inside tests as needed.
    """
    key: jax.Array = jax.random.PRNGKey(0)
    return key
