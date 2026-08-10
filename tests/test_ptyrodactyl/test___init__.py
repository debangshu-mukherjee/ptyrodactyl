import os
import subprocess
import sys
import textwrap

from beartype.typing import Dict

import ptyrodactyl

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


def _clean_env() -> Dict[str, str]:
    env = os.environ.copy()
    for key in _ENV_KEYS:
        env.pop(key, None)
    return env


def _run_script(script: str, env: Dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def test_double_import_reload_is_safe() -> None:
    """Prove a clean subprocess can import and reload the package once."""
    result = _run_script(
        """
        import importlib
        import ptyrodactyl

        importlib.reload(ptyrodactyl)
        """,
        env=_clean_env(),
    )

    assert result.returncode == 0, result.stderr


def test_xla_flags_are_merged_with_existing_operator_flags() -> None:
    """Prove CPU defaults preserve an operator-supplied XLA flag.

    Inspect the merged ``XLA_FLAGS`` value in a clean subprocess.
    """
    env = _clean_env()
    custom_flag = "--xla_force_host_platform_device_count=2"
    env["XLA_FLAGS"] = custom_flag

    result = _run_script(
        """
        import os
        import ptyrodactyl

        flags = os.environ["XLA_FLAGS"]
        assert "--xla_force_host_platform_device_count=2" in flags
        assert "--xla_cpu_multi_thread_eigen=true" in flags
        assert "intra_op_parallelism_threads=0" in flags
        """,
        env=env,
    )

    assert result.returncode == 0, result.stderr


def test_init_distributed_returns_false_without_env_opt_in(
    monkeypatch,
) -> None:
    """Prove distributed setup returns false without its opt-in variable.

    Remove the variable with ``monkeypatch`` before calling the public helper.

    :see: :func:`ptyrodactyl.init_distributed`
    """
    monkeypatch.delenv("PTYRODACTYL_DISTRIBUTED", raising=False)

    assert ptyrodactyl.init_distributed() is False


def test_distributed_init_warning_does_not_crash_import() -> None:
    """Prove a forced distributed failure warns without breaking import.

    Replace JAX setup in a subprocess and inspect its warning and exit.
    """
    env = _clean_env()
    env["PTYRODACTYL_COORDINATOR_ADDRESS"] = "127.0.0.1:1"
    env["PTYRODACTYL_DISTRIBUTED"] = "1"
    env["SLURM_NTASKS"] = "2"

    result = _run_script(
        """
        import warnings

        import jax

        def fail_initialize(*args, **kwargs):
            raise RuntimeError("forced distributed failure")

        jax.distributed.initialize = fail_initialize
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            import ptyrodactyl

        assert any(
            issubclass(item.category, RuntimeWarning)
            and "forced distributed failure" in str(item.message)
            for item in caught
        )
        print("runtime-warning")
        """,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime-warning" in result.stdout


def test_runtime_check_toggle_sets_eqx_on_error_and_disables_raise() -> None:
    """Prove the runtime-check opt-out permits a negative calibration.

    Compare subprocess exits and errors with runtime checks enabled and off.
    """
    script = """
        import os

        import jax
        import jax.numpy as jnp
        import ptyrodactyl
        from ptyrodactyl.types import create_calibrated_array

        assert os.environ["EQX_ON_ERROR"] == "off"
        array = create_calibrated_array(
            jnp.ones((1, 1), dtype=jnp.float64),
            jnp.array(-1.0, dtype=jnp.float64),
            jnp.array(1.0, dtype=jnp.float64),
            True,
        )
        jax.block_until_ready(array.calib_y)
    """
    off_env = _clean_env()
    off_env["PTYRODACTYL_DISABLE_RUNTIME_CHECKS"] = "1"
    off_result = _run_script(script, env=off_env)

    raise_result = _run_script(
        """
        import jax
        import jax.numpy as jnp
        import ptyrodactyl
        from ptyrodactyl.types import create_calibrated_array

        array = create_calibrated_array(
            jnp.ones((1, 1), dtype=jnp.float64),
            jnp.array(-1.0, dtype=jnp.float64),
            jnp.array(1.0, dtype=jnp.float64),
            True,
        )
        jax.block_until_ready(array.calib_y)
        """,
        env=_clean_env(),
    )

    assert off_result.returncode == 0, off_result.stderr
    assert raise_result.returncode != 0
    assert "calib_y must be positive" in raise_result.stderr


def test_init_distributed_is_public() -> None:
    """Prove distributed setup is a top-level public export.

    Inspect membership in the package's literal ``__all__`` list.
    """
    assert "init_distributed" in ptyrodactyl.__all__
