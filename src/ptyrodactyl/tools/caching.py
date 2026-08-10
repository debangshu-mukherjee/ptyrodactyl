"""Persistent XLA compilation cache configuration for ptyrodactyl.

Extended Summary
----------------
JAX specializes every compiled executable on the input shapes and dtypes it is
traced with, recompiling whenever a new shape signature appears. The XLA
persistent cache writes those compiled executables to disk so that a later
process can load them instead of paying the compilation cost again.

The cache must be configured before the first compilation. The top-level
:mod:`ptyrodactyl` package calls :func:`enable_compilation_cache` at import
time only when an environment opt-in is present; interactive users can call it
directly before their first JIT compilation.

Routine Listings
----------------
:func:`enable_compilation_cache`
    Enable JAX's persistent compilation cache.

Notes
-----
XLA:CPU executables are codegen-specialized on the host CPU feature set. By
default the cache directory is namespaced per architecture so heterogeneous
clusters never share compiled executables.
"""

import hashlib
import os
import pathlib
import platform

import jax
from beartype import beartype
from jaxtyping import jaxtyped

_DEFAULT_CACHE_ROOT: str = "~/.cache/ptyrodactyl/xla"


def _architecture_tag() -> str:
    """PRIVATE: Build a directory tag that distinguishes XLA targets.

    Returns
    -------
    result : str
        Host system, machine, and CPU-feature digest joined as one directory
        tag.

    Notes
    -----
    Linux hosts include a short SHA-1 digest of the ``/proc/cpuinfo`` feature
    flags. Other hosts use the literal ``noflags`` component.
    """
    system: str = platform.system()
    machine: str = platform.machine()
    flags: str = ""
    cpuinfo: pathlib.Path = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("flags"):
                flags = line.split(":", 1)[1].strip()
                break
    digest: str = (
        hashlib.sha1(flags.encode()).hexdigest()[:8]  # noqa: S324
        if flags
        else "noflags"
    )
    result: str = f"{system}-{machine}-{digest}"
    return result


@jaxtyped(typechecker=beartype)
def enable_compilation_cache(
    cache_dir: str | None = None,
    *,
    per_arch: bool = True,
    min_compile_time_secs: float = 0.0,
    min_entry_size_bytes: int = 0,
) -> bool:
    """Enable JAX's persistent compilation cache.

    Configures the XLA persistent cache so compiled executables are written to
    and reloaded from disk across processes. Call before the first compilation
    for the executables of interest to be cached.

    :see: :mod:`~.test_caching`

    Parameters
    ----------
    cache_dir : str, optional
        Cache root directory. If ``None``, falls back to
        ``PTYRODACTYL_CACHE_DIR`` and then to ``~/.cache/ptyrodactyl/xla``.
    per_arch : bool, optional
        If ``True`` (default), append an architecture tag to ``cache_dir`` so
        nodes with different CPUs never load each other's executables.
    min_compile_time_secs : float, optional
        Only cache executables whose compilation took at least this many
        seconds. ``0.0`` caches everything.
    min_entry_size_bytes : int, optional
        Only cache executables at least this many bytes in size. ``0`` caches
        everything.

    Returns
    -------
    enabled : bool
        ``True`` when the cache directory was created and JAX configuration was
        updated, ``False`` when setup failed.
    """
    enabled: bool = False
    try:
        root: str = (
            cache_dir
            or os.environ.get("PTYRODACTYL_CACHE_DIR")
            or _DEFAULT_CACHE_ROOT
        )
        resolved: pathlib.Path = pathlib.Path(root).expanduser()
        if per_arch:
            resolved = resolved / _architecture_tag()
        resolved.mkdir(parents=True, exist_ok=True)
        resolved_dir: str = str(resolved)

        jax.config.update("jax_compilation_cache_dir", resolved_dir)
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs",
            min_compile_time_secs,
        )
        jax.config.update(
            "jax_persistent_cache_min_entry_size_bytes",
            min_entry_size_bytes,
        )
    except Exception:
        return enabled
    enabled: bool = True
    return enabled


__all__: list[str] = [
    "enable_compilation_cache",
]
