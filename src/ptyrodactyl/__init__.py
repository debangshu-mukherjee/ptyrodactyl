"""Differentiable electron microscopy forward and inverse problems.

Extended Summary
----------------
A comprehensive toolkit for electron ptychography simulations
and reconstructions using JAX for automatic differentiation and
GPU acceleration. All functions are fully differentiable and
JIT-compilable, supporting ``jax.jit``, ``jax.grad``,
``jax.vmap``, and other JAX transformations. Complex-valued
optimization is handled via Wirtinger calculus, and distributed
computing is supported through device mesh parallelism. Type
safety is enforced with jaxtyping and beartype.

Import-time bootstrap is deliberately ordered. The package merges its
CPU XLA defaults into any operator-supplied ``XLA_FLAGS`` without
clobbering existing keys, optionally sets ``EQX_ON_ERROR=off`` when
``PTYRODACTYL_DISABLE_RUNTIME_CHECKS=1`` is present, imports JAX, and
then enables 64-bit precision before importing submodules. The early
64-bit setting ensures physical constants materialize as float64.

The persistent XLA compilation cache is opt-in. Set
``PTYRODACTYL_CACHE_DIR`` (or ``PTYRODACTYL_COMPILATION_CACHE=1`` to use
the default location) before import and the package enables it before
any compilation through :func:`~ptyrodactyl.tools.enable_compilation_cache`.
Interactive users can call that function directly before their first
compilation.

Runtime input validation (``equinox.error_if`` in factory functions and
checked simulators) is on by default. Equinox resolves ``EQX_ON_ERROR``
at import, so set ``PTYRODACTYL_DISABLE_RUNTIME_CHECKS=1`` before import
to request ``EQX_ON_ERROR=off`` for trusted data or export workflows. An
explicit ``EQX_ON_ERROR`` setting is always respected.

The submodules are organized as follows:

- :mod:`bloch`
    Bloch wave simulations.
- :mod:`born`
    Convergent Born series simulations.
- :mod:`inout`
    Crystal-structure ingestion and lookup exports.
- :mod:`invert`
    Electron microscopy reconstructions, ptychography and
    focal series.
- :mod:`jacobian`
    Jacobian computation submodule.
- :mod:`multislice`
    Multislice-family forward simulations including CBED and
    4D-STEM.
- :mod:`plots`
    Plotting and visualization helper exports.
- :mod:`tools`
    Utility tools for optimization, loss functions, and
    parallel processing including complex-valued optimizers
    with Wirtinger derivatives.
- :mod:`types`
    Single home for carriers, type aliases, physical constants,
    and validated create_* factories.
- :mod:`ucell`
    Unit-cell geometry and crystallographic helpers.
- :mod:`workflows`
    High-level workflows combining simulation steps for common
    use cases such as simulating 4D-STEM data from XYZ
    structure files.

Routine Listings
----------------
:func:`init_distributed`
    Initialize JAX multi-host execution, idempotently and safely.

Notes
-----
All functions are optimized for JAX transformations and support
both CPU and GPU execution. For best performance, use JIT
compilation and consider using the provided factory functions
for input validation and float64 casting.

Multi-node distributed execution is supported via
``jax.distributed.initialize()``. To enable, set the
environment variable ``PTYRODACTYL_DISTRIBUTED=1`` before
launching with ``srun`` or equivalent. An optional
``PTYRODACTYL_COORDINATOR_ADDRESS`` environment variable
overrides automatic SLURM coordinator detection, which is
required on some ROCm clusters.
"""

import os
import warnings
from importlib.metadata import version

_PTYRODACTYL_XLA_FLAGS: tuple[str, ...] = (
    "--xla_cpu_multi_thread_eigen=true",
    "intra_op_parallelism_threads=0",
)
_existing_xla: str = os.environ.get("XLA_FLAGS", "")
_xla_parts: list[str] = [_existing_xla] if _existing_xla else []
for _flag in _PTYRODACTYL_XLA_FLAGS:
    if _flag.split("=", 1)[0] not in _existing_xla:
        _xla_parts.append(_flag)
os.environ["XLA_FLAGS"] = " ".join(_xla_parts).strip()

if os.environ.get("PTYRODACTYL_DISABLE_RUNTIME_CHECKS", "0") == "1":
    os.environ.setdefault("EQX_ON_ERROR", "off")

import jax  # noqa: E402
from beartype import beartype  # noqa: E402
from jaxtyping import jaxtyped  # noqa: E402

jax.config.update("jax_enable_x64", True)

_cache_requested: bool = (
    os.environ.get("PTYRODACTYL_COMPILATION_CACHE", "0") == "1"
    or os.environ.get("PTYRODACTYL_CACHE_DIR") is not None
)
if _cache_requested:
    from .tools.caching import enable_compilation_cache  # noqa: E402

    enable_compilation_cache()


@jaxtyped(typechecker=beartype)
def init_distributed(
    coordinator_address: str | None = None,
    *,
    force: bool = False,
) -> bool:
    """Initialize JAX multi-host execution, idempotently and safely.

    Extended Summary
    ----------------
    Wraps ``jax.distributed.initialize`` with guards for import-time use. The
    call is skipped unless distributed execution is explicitly requested, it
    is not repeated when the JAX runtime is already initialized, and failures
    degrade to :class:`RuntimeWarning` instead of crashing package import.

    ``jax.distributed.initialize`` is a collective operation: every process in
    a multi-host job must reach it.

    Parameters
    ----------
    coordinator_address : str | None, optional
        Coordinator ``host:port``. If ``None``, falls back to
        ``PTYRODACTYL_COORDINATOR_ADDRESS`` and then to automatic SLURM
        detection.
    force : bool, optional
        If ``True``, attempt initialization even when the environment opt-in
        (``PTYRODACTYL_DISTRIBUTED`` / ``SLURM_NTASKS``) is not satisfied.

    Returns
    -------
    bool
        ``True`` if the runtime is initialized on return, ``False`` otherwise.
    """
    if not force:
        if os.environ.get("PTYRODACTYL_DISTRIBUTED", "0") != "1":
            return False
        if int(os.environ.get("SLURM_NTASKS") or "1") <= 1:
            return False

    is_initialized = getattr(jax.distributed, "is_initialized", None)
    if callable(is_initialized) and is_initialized():
        return True

    address: str | None = coordinator_address or os.environ.get(
        "PTYRODACTYL_COORDINATOR_ADDRESS"
    )
    try:
        if address is not None:
            jax.distributed.initialize(coordinator_address=address)
        else:
            jax.distributed.initialize()
    except (RuntimeError, ValueError) as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        return False
    return True


init_distributed()

from . import (  # noqa: E402, I001
    bloch,
    born,
    inout,
    invert,
    jacobian,
    multislice,
    plots,
    tools,
    types,
    ucell,
    workflows,
)

__version__: str = version("ptyrodactyl")

__all__: list[str] = [
    "bloch",
    "born",
    "init_distributed",
    "inout",
    "invert",
    "jacobian",
    "multislice",
    "plots",
    "tools",
    "types",
    "ucell",
    "workflows",
]
