"""Run the scalar-potential smoke test against an installed wheel."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np

import ptyrodactyl
from ptyrodactyl.inout import kirkland_potentials, lobato_potentials
from ptyrodactyl.multislice import (
    atomic_form_factor,
    single_atom_potential_3d,
)


def _main() -> None:
    """Check the packaged tables and one JIT-compiled voltage-field path."""
    if not ((3, 12) <= sys.version_info[:2] < (3, 15)):
        raise RuntimeError("wheel smoke requires supported Python 3.12--3.14")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("ptyrodactyl did not enable JAX 64-bit precision")

    checkout_root = Path(__file__).resolve().parents[1]
    package_path = Path(ptyrodactyl.__file__).resolve()
    if package_path.is_relative_to(checkout_root):
        raise RuntimeError(
            "smoke test imported the source checkout, not the wheel"
        )

    lobato = lobato_potentials()
    kirkland = kirkland_potentials()
    if lobato.shape != (103, 10) or kirkland.shape != (103, 12):
        raise RuntimeError(
            "the installed wheel has invalid coefficient tables"
        )

    compiled_form_factor = jax.jit(lambda value: atomic_form_factor(14, value))
    form_factor = compiled_form_factor(jnp.asarray(0.25, dtype=jnp.float64))
    volume = single_atom_potential_3d(
        14,
        0.5,
        (4, 4, 4),
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float64),
        band_limit=0.75,
    )
    jax.block_until_ready((form_factor, volume))
    if form_factor.dtype != jnp.float64 or volume.dtype != jnp.float64:
        raise RuntimeError(
            "the installed scalar path did not preserve float64"
        )
    if volume.shape != (4, 4, 4):
        raise RuntimeError(
            "the installed scalar path returned the wrong shape"
        )
    if not bool(jnp.isfinite(form_factor)) or not bool(
        jnp.all(jnp.isfinite(volume))
    ):
        raise RuntimeError(
            "the installed scalar path returned a non-finite value"
        )

    result = {
        "jax": jax.__version__,
        "jaxtyping": jaxtyping.__version__,
        "numpy": np.__version__,
        "package": str(package_path),
        "ptyrodactyl": ptyrodactyl.__version__,
        "units": "V",
        "volume_shape": list(volume.shape),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    _main()
