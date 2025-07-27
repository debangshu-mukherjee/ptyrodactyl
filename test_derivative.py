import jax
import jax.numpy as jnp

from ptyrodactyl.electrons.atom_potentials import bessel_kv

jax.config.update("jax_enable_x64", True)

# Test derivative computation
x = jnp.array(1.0, dtype=jnp.float64)
v0 = jnp.array(0.0, dtype=jnp.float64)
v1 = jnp.array(1.0, dtype=jnp.float64)

# K_0(x)
k0 = bessel_kv(v0, x)
print(f"K_0(1.0) = {k0}")

# K_1(x)
k1 = bessel_kv(v1, x)
print(f"K_1(1.0) = {k1}")

# Derivative of K_0
grad_fn = jax.grad(lambda y: bessel_kv(v0, y))
dk0_dx = grad_fn(x)
print(f"dK_0/dx(1.0) = {dk0_dx}")
print(f"-K_1(1.0) = {-k1}")
print(f"Difference: {dk0_dx - (-k1)}")
