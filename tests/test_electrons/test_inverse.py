import jax.numpy as jnp
from ptyrodactyl.electrons.forward import make_probe, make_sample, stem_4D

image_size = (64, 64)
voltage = 200
aperture = 0.2
calib = 0.01

probe = make_probe(
    voltage=voltage,
    aperture=aperture,
    image_size=image_size,
    calibration_pm=calib * 1000,
)

sample_shape = (64, 64, 1)
sample = make_sample(sample_shape)

scan_positions = jnp.array([[32, 32], [16, 16], [48, 48]])  # shape: (3, 2)

pot_slice = sample[..., 0]  # shape: (64, 64)

output = stem_4D(
    pot_slice=pot_slice,
    beam=probe,
    positions=scan_positions,
    slice_thickness=5.0,
    voltage_kV=voltage,
    calib_ang=calib,
)

print("4D-STEM output shape:", output.shape)
