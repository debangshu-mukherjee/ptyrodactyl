"""Test :mod:`ptyrodactyl.tools.caching`.

:see: :func:`ptyrodactyl.tools.enable_compilation_cache`
"""

import os
import subprocess
import sys
import textwrap

_ENV_KEYS = (
    "EQX_ON_ERROR",
    "PTYRODACTYL_CACHE_DIR",
    "PTYRODACTYL_COMPILATION_CACHE",
    "PTYRODACTYL_COORDINATOR_ADDRESS",
    "PTYRODACTYL_DISABLE_RUNTIME_CHECKS",
    "PTYRODACTYL_DISTRIBUTED",
    "SLURM_NTASKS",
    "XLA_FLAGS",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _ENV_KEYS:
        env.pop(key, None)
    return env


def _run_script(script: str, env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def test_import_opt_in_configures_compilation_cache(tmp_path) -> None:
    """Prove import opt-in configures a usable cache below the requested path.

    Start a clean subprocess with a temporary cache root, run one JIT function,
    and inspect the resolved JAX cache path and public helper result.
    """
    env = _clean_env()
    env["PTYRODACTYL_CACHE_DIR"] = str(tmp_path)
    env["PTYRODACTYL_COMPILATION_CACHE"] = "1"

    result = _run_script(
        """
        import os
        import pathlib

        import jax
        import jax.numpy as jnp
        import ptyrodactyl

        root = pathlib.Path(os.environ["PTYRODACTYL_CACHE_DIR"]).resolve()
        cache_dir = pathlib.Path(jax.config.jax_compilation_cache_dir)
        cache_dir = cache_dir.resolve()
        assert cache_dir.exists()
        assert cache_dir == root or root in cache_dir.parents
        assert ptyrodactyl.tools.enable_compilation_cache() is True

        @jax.jit
        def add_one(value):
            return value + 1

        value = add_one(jnp.ones((4,), dtype=jnp.float64))
        jax.block_until_ready(value)
        assert jax.config.jax_compilation_cache_dir == str(cache_dir)
        print("cache-configured")
        """,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "cache-configured" in result.stdout


def test_enable_compilation_cache_returns_bool_and_is_idempotent(
    tmp_path,
) -> None:
    """Prove repeated cache enablement returns true Boolean results.

    Call the public helper twice with one temporary root in a clean subprocess.

    :see: :func:`ptyrodactyl.tools.enable_compilation_cache`
    """
    env = _clean_env()
    env["CACHE_UNDER_TEST"] = str(tmp_path)

    result = _run_script(
        """
        import os

        from ptyrodactyl.tools import enable_compilation_cache

        cache_root = os.environ["CACHE_UNDER_TEST"]
        first = enable_compilation_cache(cache_root)
        second = enable_compilation_cache(cache_root)
        assert isinstance(first, bool)
        assert isinstance(second, bool)
        assert first is True
        assert second is True
        print(first, second)
        """,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "True True" in result.stdout
