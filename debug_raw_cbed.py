"""Debug raw CBED pattern before clipping."""

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
print(f"Crystal: {crystal_data.positions.shape[0]} atoms")

# Parameters
voltage_kv = 60
cbed_aperture_mrad = 5.0
grid_pixels = 512
pixel_size_ang = 0.05
slice_thickness = 1.0
grid_size_ang = grid_pixels * pixel_size_ang

# Atom type mapping
unique_atoms = sorted({int(x) for x in crystal_data.atomic_numbers})
atom_type_map = {atom_num: idx for idx, atom_num in enumerate(unique_atoms)}
atom_types_idx = jnp.array(
    [atom_type_map[int(x)] for x in crystal_data.atomic_numbers],
    dtype=jnp.int32
)
print(f"Unique atoms: {unique_atoms}")

# Build atom potentials (padded for FFT convolution)
potential_extent_ang = 10.0
kernel_size = int(np.ceil(potential_extent_ang / pixel_size_ang))
if kernel_size % 2 == 0:
    kernel_size += 1

atom_potentials_list = []
for atom_num in unique_atoms:
    small_pot = single_atom_potential(
        atom_no=atom_num,
        pixel_size=pixel_size_ang,
        grid_shape=(kernel_size, kernel_size),
        center_coords=jnp.array([0.0, 0.0]),
        supersampling=4,
    )
    padded_pot = jnp.zeros((grid_pixels, grid_pixels), dtype=jnp.float64)
    half_k = kernel_size // 2
    padded_pot = padded_pot.at[:half_k + 1, :half_k + 1].set(small_pot[half_k:, half_k:])
    padded_pot = padded_pot.at[:half_k + 1, -half_k:].set(small_pot[half_k:, :half_k])
    padded_pot = padded_pot.at[-half_k:, :half_k + 1].set(small_pot[:half_k, half_k:])
    padded_pot = padded_pot.at[-half_k:, -half_k:].set(small_pot[:half_k, :half_k])
    atom_potentials_list.append(padded_pot)
atom_potentials = jnp.stack(atom_potentials_list, axis=0)
print(f"Atom potentials: {atom_potentials.shape}")

# Create probe
image_size = jnp.array([grid_pixels, grid_pixels])
probe = make_probe(
    aperture=cbed_aperture_mrad,
    voltage=voltage_kv,
    image_size=image_size,
    calibration_pm=pixel_size_ang * 100.0,
)
modes = probe[..., jnp.newaxis]  # Add mode dimension
print(f"Probe: {probe.shape}, sum={float(jnp.sum(jnp.abs(probe)**2)):.4f}")

# Z slicing
z_coords = crystal_data.positions[:, 2]
z_min = float(jnp.min(z_coords))
z_max = float(jnp.max(z_coords))
num_slices = max(1, int(np.ceil((z_max - z_min) / slice_thickness)))
slice_boundaries = [[z_min + i * slice_thickness, z_min + (i + 1) * slice_thickness] 
                    for i in range(num_slices)]
slice_z_bounds = jnp.array(slice_boundaries, dtype=jnp.float64)
print(f"Slices: {num_slices} from z={z_min:.2f} to {z_max:.2f} Å")

# Set up tile at sample center
x_coords = crystal_data.positions[:, 0]
y_coords = crystal_data.positions[:, 1]
x_center = float((jnp.max(x_coords) + jnp.min(x_coords)) / 2)
y_center = float((jnp.max(y_coords) + jnp.min(y_coords)) / 2)
print(f"Sample center: ({x_center:.1f}, {y_center:.1f}) Å")

tile_center_offset = grid_size_ang / 2.0
tile_min_x = x_center - tile_center_offset
tile_min_y = y_center - tile_center_offset

# Compute in_tile mask
in_tile = (
    (crystal_data.positions[:, 0] >= tile_min_x) &
    (crystal_data.positions[:, 0] < tile_min_x + grid_size_ang) &
    (crystal_data.positions[:, 1] >= tile_min_y) &
    (crystal_data.positions[:, 1] < tile_min_y + grid_size_ang)
).astype(jnp.float64)
print(f"Atoms in tile: {int(jnp.sum(in_tile))} / {len(in_tile)}")

# Local coordinates
local_x = crystal_data.positions[:, 0] - tile_min_x
local_y = crystal_data.positions[:, 1] - tile_min_y
local_z = crystal_data.positions[:, 2]
local_coords = jnp.stack([local_x, local_y, local_z], axis=-1)

# Shift beam to tile center
h, w = grid_pixels, grid_pixels
probe_k = jnp.fft.fft2(modes, axes=(0, 1))
qy = jnp.fft.fftfreq(h, d=pixel_size_ang)
qx = jnp.fft.fftfreq(w, d=pixel_size_ang)
qya, qxa = jnp.meshgrid(qy, qx, indexing="ij")
y_shift = tile_center_offset
x_shift = tile_center_offset
phase = -2.0 * jnp.pi * ((qya * y_shift) + (qxa * x_shift))
phase_shift = jnp.exp(1j * phase)
shifted_k = probe_k * phase_shift[..., None]
shifted_beam = jnp.fft.ifft2(shifted_k, axes=(0, 1))

print(f"Beam shifted to: ({y_shift:.1f}, {x_shift:.1f}) Å in tile coords")

# Compute raw CBED
print("Computing CBED...")
cbed = _cbed_from_potential_slices(
    beam=shifted_beam,
    atom_coords=local_coords,
    atom_types=atom_types_idx,
    slice_z_bounds=slice_z_bounds,
    atom_potentials=atom_potentials,
    voltage_kv=voltage_kv,
    calib_ang=pixel_size_ang,
    atom_mask=in_tile,
)

print(f"\nRaw CBED: {cbed.shape}")
print(f"  min: {float(jnp.min(cbed)):.6f}")
print(f"  max: {float(jnp.max(cbed)):.6f}")
print(f"  sum: {float(jnp.sum(cbed)):.6f}")
print(f"  center value: {float(cbed[grid_pixels//2, grid_pixels//2]):.6f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Full raw CBED
im0 = axes[0].imshow(np.array(cbed), cmap='viridis')
axes[0].set_title(f'Full raw CBED {cbed.shape}')
axes[0].axhline(grid_pixels//2, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(grid_pixels//2, color='r', linestyle='--', alpha=0.5)
plt.colorbar(im0, ax=axes[0])

# Log scale
cbed_np = np.array(cbed)
cbed_log = np.log10(cbed_np + 1e-10)
im1 = axes[1].imshow(cbed_log, cmap='viridis')
axes[1].set_title('Log10 scale')
plt.colorbar(im1, ax=axes[1])

# Center crop
center = grid_pixels // 2
crop = 64
cropped = cbed_np[center-crop:center+crop, center-crop:center+crop]
im2 = axes[2].imshow(cropped, cmap='viridis')
axes[2].set_title(f'Center {2*crop}x{2*crop} crop')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('data/debug_raw_cbed.png', dpi=150)
print("\n✓ Saved to data/debug_raw_cbed.png")
