"""Pytest configuration for the ptyrodactyl test suite.

Extended Summary
----------------
Suite-wide configuration only: the annotation pre-flight gate, float64
discipline verification, and shared random-key handling. Test layout mirrors
``src/``: every source module ``src/ptyrodactyl/<pkg>/<mod>.py`` has its
counterpart ``tests/test_ptyrodactyl/test_<pkg>/test_<mod>.py``;
``tests/test_data/`` holds fixture data and pinned regression references.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

_PREFLIGHT_SKIP_VARIABLE: str = "PTYRODACTYL_SKIP_PREFLIGHT"


def _run_annotation_preflight(config: pytest.Config) -> None:
    """Reject invalid annotations before pytest collects test modules.

    The gate runs ``tests/_preflight_types.py`` in a subprocess. A subprocess
    keeps the jaxtyping import hook out of this process. An in-process run
    leaves decorated modules in ``sys.modules``. Pytest then collects wrapped
    fixtures.

    The gate runs once for each session. It does not run on a pytest-xdist
    worker.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration. A worker configuration carries
        ``workerinput``.

    Raises
    ------
    pytest.UsageError
        If one module or more carries an invalid annotation.
    """
    if hasattr(config, "workerinput"):
        return
    if os.environ.get(_PREFLIGHT_SKIP_VARIABLE):
        return
    os.environ[_PREFLIGHT_SKIP_VARIABLE] = "1"
    script: Path = Path(__file__).resolve().parent / "_preflight_types.py"
    if not script.is_file():
        return
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message: str = (
            "annotation pre-flight failed; fix these before running tests:\n"
            + completed.stdout
            + completed.stderr
        )
        raise pytest.UsageError(message)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Assert the x64 discipline the package enables at import time.

    Parameters
    ----------
    config : pytest.Config
        The pytest configuration object (unused beyond the hook contract).
    """
    _run_annotation_preflight(config)
    import ptyrodactyl  # noqa: F401, PLC0415  (enables x64 after hooks)

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
