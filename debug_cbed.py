"""Debug CBED - look at raw pattern before clipping."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ptyrodactyl import simul, tools
from ptyrodactyl.simul.parallelized import _cbed_from_potential_slices
from ptyrodactyl.simul import make_probe, single_atom_potential

print(f"JAX devices: {jax.devices()}")

# Load crystal data
xyz_file = "data/mos2_234atoms_GB.xyz"
crystal_data: tools.CrystalData = simul.parse_crystal(xyz_file)
print(f"Crystal positions shape: {crystal_data.positions.shape}")

# Parameters
voltage_kv = 60
cbed_aperture_mrad = 5.0
grid_pixels = 512
pixel_size_ang = 0.05
slice_thickness = 1.0

# Get unique atom types
unique_types = np.unique(np.array(crystal_data.atomic_numbers))
print(f"Unique atomic numbers: {unique_types}")

# Precompute atom potentials
atom_potentials_list = []
for atomic_number in unique_types:
    pot = single_atom_potential(
        atomic_number=int(atomic_number),
        output_shape=(grid_pixels, grid_pixels),
        calib_ang=pixel_size_ang,
        voltage_kv=voltage_kv,
    )
    atom_potentials_list.append(pot)
atom_potentials = jnp.stack(atom_potentials_list, axis=0)
print(f"Atom potentials shape: {atom_potentials.shape}")

# Create type index mapping
type_to_idx = {int(t): i for i, t in enumerate(unique_types)}
atom_types_idx = jnp.array([type_to_idx[int(t)] for t in crystal_data.atomic_numbers])

# Create probe (not shifted)
probe = make_probe(
    output_shape=(grid_pixels, grid_pixels),
    calib_ang=pixel_size_ang,
    voltage_kv=voltage_kv,
    aperture_mrad=cbed_aperture_mrad,
)
print(f"Probe shape: {probe.shape}")

# Add mode dimension
beam = probe[..., jnp.newaxis]

# Get z range and create slices
z_coords = crystal_data.positions[:, 2]
z_min = float(jnp.min(z_coords))
z_max = float(jnp.max(z_coords))
num_slices = max(1, int(np.ceil((z_max - z_min) / slice_thickness)))
print(f"Z range: {z_min:.2f} to {z_max:.2f} Å, {num_slices} slices")

slice_z_bounds = []
for i in range(num_slices):
    s_z_min = z_min + i * slice_thickness
    s_z_max = z_min + (i + 1) * slice_thickness
    slice_z_bounds.append([s_z_min, s_z_max])
slice_z_bounds = jnp.array(slice_z_bounds)

# Shift atom coordinates to center beam at grid center
x_coords = crystal_data.positions[:, 0]
y_coords = crystal_data.positions[:, 1]
x_center = float((jnp.max(x_coords) + jnp.min(x_coords)) / 2)
y_center = float((jnp.max(y_coords) + jnp.min(y_coords)) / 2)

# The beam is centered at grid_center (grid_pixels * pixel_size_ang / 2)
# We want to simulate as if beam is at (y_center, x_center)
# So we shift atoms: atom_new = atom - beam_pos + grid_center
grid_center = grid_pixels * pixel_size_ang / 2
shifted_positions = crystal_data.positions.at[:, 0].add(grid_center - x_center)
shifted_positions = shifted_positions.at[:, 1].add(grid_center - y_center)

print(f"Sample center: ({y_center:.1f}, {x_center:.1f}) Å")
print(f"Grid center: {grid_center:.1f} Å")

# Compute CBED with centered beam
cbed = _cbed_from_potential_slices(
    beam=beam,
    atom_coords=shifted_positions,
    atom_types=atom_types_idx,
    slice_z_bounds=slice_z_bounds,
    atom_potentials=atom_potentials,
    voltage_kv=voltage_kv,
    calib_ang=pixel_size_ang,
)

print(f"\nCBED shape: {cbed.shape}")
print(f"CBED min: {float(jnp.min(cbed)):.6f}")
print(f"CBED max: {float(jnp.max(cbed)):.6f}")
print(f"CBED sum: {float(jnp.sum(cbed)):.6f}")

# Plot raw CBED
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Linear scale
im0 = axes[0].imshow(np.array(cbed), cmap='viridis')
axes[0].set_title('Raw CBED (linear)')
plt.colorbar(im0, ax=axes[0])

# Log scale
cbed_log = np.log10(np.array(cbed) + 1e-10)
im1 = axes[1].imshow(cbed_log, cmap='viridis')
axes[1].set_title('Raw CBED (log10)')
plt.colorbar(im1, ax=axes[1])

# Center crop (256x256)
center = grid_pixels // 2
crop = 128
cropped = np.array(cbed)[center-crop:center+crop, center-crop:center+crop]
im2 = axes[2].imshow(cropped, cmap='viridis')
axes[2].set_title(f'Center crop [{center-crop}:{center+crop}]')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('data/debug_cbed.png', dpi=150)
print("\n✓ Saved to data/debug_cbed.png")
