import jax
import jax.numpy as jnp

from ptyrodactyl.electrons.atom_potentials import bessel_kv

jax.config.update("jax_enable_x64", True)

# Test basic function call
x = jnp.array(0.1, dtype=jnp.float64)
v = jnp.array(0.0, dtype=jnp.float64)

# Without JIT
result_no_jit = bessel_kv(v, x)
print(f"Without JIT: K_0(0.1) = {result_no_jit}")

# With JIT
bessel_kv_jit = jax.jit(bessel_kv)
try:
    result_jit = bessel_kv_jit(v, x)
    print(f"With JIT: K_0(0.1) = {result_jit}")
except Exception as e:
    print(f"JIT failed with error: {type(e).__name__}: {e}")
