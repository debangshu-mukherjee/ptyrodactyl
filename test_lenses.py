import jax.numpy as jnp
from ptyrodactyl.photons.lenses import plano_convex_lens, create_lens_phase

# 512×512 grid over +/- 1mm
t = jnp.linspace(-1e-3, 1e-3, 512)
xx, yy = jnp.meshgrid(t, t, indexing='xy')

# Create lens parameters
params = plano_convex_lens(
    focal_length=0.05,
    diameter=2e-3,
    n=1.5,
    center_thickness=1e-3,
)


# Compute phase and transmission
phase, trans = create_lens_phase(xx, yy, params, wavelength=500e-9)

# Print results
print("Grid shape:", xx.shape)
print("Phase min/max:", float(phase.min()), float(phase.max()))
print("Transmission sum:", int(trans.sum()))

