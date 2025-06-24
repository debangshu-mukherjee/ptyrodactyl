import jax.numpy as jnp
from ptyrodactyl.electrons.forward import make_probe, stem_4D

image_size = jnp.array([64, 64], dtype=int)
voltage_kV = 200.0
aperture = 0.01
calib_ang = 10.0
slice_thickness = 5.0

probe = make_probe(
    aperture=aperture,
    voltage=voltage_kV,
    image_size=image_size,
    calibration_pm=calib_ang,
)

pot_slice = jnp.ones((64, 64, 1), dtype=jnp.complex64)

positions = jnp.array(
    [
        [0.0, 0.0],
        [10.0, 10.0],
        [20.0, 20.0],
    ],
    dtype=jnp.float32,
)

beam = probe[..., None]

output = stem_4D(
    pot_slice=pot_slice,
    beam=beam,
    positions=positions,
    slice_thickness=slice_thickness,
    voltage_kV=voltage_kV,
    calib_ang=calib_ang,
)

print(output.shape)
