# Changelog

## Plan 05 / Plan 14 — Stage-0 scalar HDF5 slice

- Added the versioned `ptyrodactyl.inout.save_to_h5` / `load_from_h5`
  boundary for lossless, validated `PotentialSlices` archives.
- Added required runtime HDF5 support through `h5py`; broader Plan-05 carrier,
  parser, DFT-interoperability, and electromagnetic schemas remain future work.

## Plans 01–04 — Revalidation repairs

The implemented foundation plans were revalidated against their strict gates.
The historical 46-test failure baseline is now fully retired instead of being
deferred to future work.

- Corrected the XYZ five-column parser and aligned its absent-lattice contract.
- Repaired the integer-order Bessel-K series, stabilized inactive singular
  branches for autodiff, improved the large-x K0/K1 approximations, and made
  Kirkland grid/repeat handling safe across eager and traced variants.
- Made complex Gauss–Newton/Levenberg–Marquardt paths use a real Hermitian tree
  inner product and primal-space adjoints; conjugate gradient now retains the
  step that first satisfies its tolerance.
- Routed atom-backed/sharded CBED through the common distribution reducer, so
  `stem4d_sharded` honors explicit `ProbeModes.weights` and keeps a complex
  amplitude seam until the single detector reduction.
- Tightened every positive factory scalar to reject NaN and infinity under JIT,
  restored the missing optimizer-oracle regression, and completed the public
  runtime-typecheck/static-type contract.
- Replaced stale `simul` documentation/tutorial calls with their owning
  `multislice`, `inout`, and `types` APIs and added missing package typing
  markers and API pages.

## Plan 04 — simul renamed to multislice

Moved: `src/ptyrodactyl/simul` -> `src/ptyrodactyl/multislice`
Moved: `tests/test_ptyrodactyl/test_simul` -> `tests/test_ptyrodactyl/test_multislice`
Retargeted: `ptyrodactyl.simul.*` imports -> `ptyrodactyl.multislice.*`

`ptyrodactyl.born` remains a sibling subpackage; the multislice family does
not absorb convergent Born series simulations.

The `checked_*` wrappers, reducer module, and producer module moved with the
package unchanged.

## Plan 04 — Ucell And Plots Relocation

Moved: `ptyrodactyl.multislice.geometry.reciprocal_lattice` → `ptyrodactyl.ucell.reciprocal_lattice`
Moved: `ptyrodactyl.multislice.geometry.rotate_structure` → `ptyrodactyl.ucell.rotate_structure`
Moved: `ptyrodactyl.multislice.geometry.rotmatrix_axis` → `ptyrodactyl.ucell.rotmatrix_axis`
Moved: `ptyrodactyl.multislice.geometry.rotmatrix_vectors` → `ptyrodactyl.ucell.rotmatrix_vectors`
Moved: `ptyrodactyl.multislice.geometry.tilt_crystal` → `ptyrodactyl.ucell.tilt_crystal`
Moved: `ptyrodactyl.multislice.atom_potentials.contrast_stretch` → `ptyrodactyl.plots.contrast_stretch`
Moved: `ptyrodactyl.multislice.parallelized.clip_cbed` → `ptyrodactyl.plots.clip_cbed`
Added: `ptyrodactyl.plots.create_phosphor_colormap`

## Plan 04 — Inout Parser Relocation

Moved: `ptyrodactyl.multislice.atomic_symbol` → `ptyrodactyl.inout.atomic_symbol`
Moved: `ptyrodactyl.multislice.kirkland_potentials` → `ptyrodactyl.inout.kirkland_potentials`
Moved: `ptyrodactyl.multislice.parse_crystal` → `ptyrodactyl.inout.parse_crystal`
Moved: `ptyrodactyl.multislice.parse_poscar` → `ptyrodactyl.inout.parse_poscar`
Moved: `ptyrodactyl.multislice.parse_xyz` → `ptyrodactyl.inout.parse_xyz`

## Plan 03 — IM7 Forward-Simulation Carrier Bundling

Forward simulator entry points now take types-owned Equinox carrier
configuration instead of loose voltage, calibration, scan, and detector
scalars. The ensemble injection point is `MicroscopeConfig.ensemble`, so
adding jitter/coherence/probe-mode axes no longer widens public signatures.

| Symbol | Old contract | New contract |
| --- | --- | --- |
| `make_probe` | `(aperture, voltage, image_size, calibration_pm, defocus=0, c3=0, c5=0)` | `(microscope: MicroscopeConfig, detector: DetectorConfig)` |
| `cbed_amplitude` | `(pot_slices, beam, voltage_kv)` | `(pot_slices, beam, microscope: MicroscopeConfig)` |
| `cbed_image` | `(pot_slices, beam, voltage_kv)` | `(pot_slices, beam, microscope: MicroscopeConfig)` |
| `stem_4d` | `(pot_slice, beam, positions, voltage_kv, calib_ang)` | `(pot_slice, beam, microscope: MicroscopeConfig, detector: DetectorConfig)` |
| `stem4d_sharded` | `(probe_modes, scan_positions_ang, atom_coords, atom_types, slice_z_bounds, atom_potentials, voltage_kv, calib_ang, mesh=None)` | `(probe_modes: ProbeModes, sample: AtomicSliceData, microscope: MicroscopeConfig, detector: DetectorConfig, mesh=None)` |
| `annular_detector` | `(stem4d_data, collection_angles, scan_shape)` | `(stem4d_data, detector: DetectorConfig)` |
| `checked_make_probe` | `(aperture, voltage, image_size, calibration_pm, defocus=0, c3=0, c5=0)` | `(microscope: MicroscopeConfig, detector: DetectorConfig)` |
| `checked_cbed_image` | `(pot_slices, beam, voltage_kv)` | `(pot_slices, beam, microscope: MicroscopeConfig)` |
| `checked_stem_4d` | `(pot_slice, beam, positions, voltage_kv, calib_ang)` | `(pot_slice, beam, microscope: MicroscopeConfig, detector: DetectorConfig)` |
| `checked_stem4d_sharded` | `(probe_modes, scan_positions_ang, atom_coords, atom_types, slice_z_bounds, atom_potentials, voltage_kv, calib_ang, mesh=None)` | `(probe_modes: ProbeModes, sample: AtomicSliceData, microscope: MicroscopeConfig, detector: DetectorConfig, mesh=None)` |

New factories: `create_microscope_config`, `create_detector_config`,
`create_ensemble_axes`, and `create_atomic_slice_data`. Existing sample-side
carriers remain in use; `AtomicSliceData` covers the sharded on-the-fly atom
slice inputs that had no existing carrier.

## Plan 03 — IM2+IM3 CBED Amplitude Split And Explicit Mode Weights

| Change | Symbols |
| --- | --- |
| Added | `cbed_amplitude`, `cbed_image`, `probe_modes_to_distribution`, `cbed_amplitude_from_atoms`, `cbed_image_from_atoms`, `checked_cbed_image` |
| Deleted | `cbed`, `_cbed_from_potential_slices` |
| Renamed | `checked_cbed` -> `checked_cbed_image` |

CBED now exposes Layer-0 complex detector amplitudes and forms public
intensities only through the distribution reducer. Probe modes returned by
`decompose_beam_to_modes` are no longer pre-scaled by `sqrt(weight)`;
`ProbeModes.weights` is the explicit incoherent mixture carrier.

## Plan 02 — JIT & Runtime-Validation Hardening (complete)

All seven phases landed (gates RJ1-RJ7): runtime typecheck stack
(`@jaxtyped(typechecker=beartype)`) on every public function in `jacobian/`,
`born/`, and `tools/`; duplicated PyTree helpers collapsed into
`jacobian/_treemath.py`; hard-coded `float32` purged; typed enums `LossType`
and `OptimizableBlock` replace string dispatch (invalid keys raise
`ValueError` at the boundary); `born/green.py` preconditions enforced
(`safety_factor > 1`, finiteness); the four `phase_recon` reconstructors run
`lax.scan` (no Python loops, no `print`; the `jnp.floor`-as-shape bug is
fixed, so `single_slice_ptychography` runs); `annular_detector` takes a
static `scan_shape`; `decompose_beam_to_modes` requires an explicit PRNG key;
additive `checked_*` validating wrappers (`checked_make_probe`,
`checked_cbed_image`, `checked_stem_4d`, `checked_stem4d_sharded`); the top-level
bootstrap merges `XLA_FLAGS` per-flag, honors
`PTYRODACTYL_DISABLE_RUNTIME_CHECKS=1 -> EQX_ON_ERROR=off`, enables an opt-in
compilation cache (`tools/caching.py`), and exposes an idempotent,
warn-degrading `init_distributed(..., force=False)`; and the test suite runs
under LIVE jaxtyping enforcement (`--jaxtyping-packages` in pytest addopts).

### Historical 46-failure baseline retired

The 27 Bessel-K failures, 17 `kirkland_potentials_crystal` shape/JIT failures,
and 2 XYZ-parser contract failures recorded when Plan 02 landed are fixed. The
suite is expected to remain fully green; the bit-level unification-gate
reference (`tests/test_data/plan01_ug_capture.py verify`) remains authoritative
for the preserved Plan-01 numerical path.

## Interaction-Parameter Physics Fix

`ptyrodactyl.tools.interaction_parameter` was removed. Its formula used
`hbar**2` where `h**2` is required, so it returned values inflated by
`(2*pi)**2 ~ 39.48` (0.0365 instead of 0.92440e-3 rad/(V·Angstrom) at
100 kV). The function had no consumers inside the package — the
multislice transmission function computes its own algebraically
equivalent sigma inline — so no simulation output changes (the
unification-gate reference remains bit-identical).

Two corrected functions replace it, both exported from
`ptyrodactyl.tools`:

- `phase_interaction_parameter(voltage_kv)` — the projected-potential
  phase coupling `sigma = 2*pi*m*e*lambda/h**2` in rad/(V·Angstrom);
  0.92440e-3 at 100 kV.
- `helmholtz_coupling(voltage_kv)` — the volumetric Helmholtz coupling
  `sigma_H = 2*m*e/hbar**2 = 8*pi**2*m0*e/h**2 * (1 + e*U0/(m0*c**2))`
  in 1/(V·Angstrom^2); 0.31383 at 100 kV. Satisfies
  `sigma_H = 2*k0*sigma` to machine precision. This is the coupling the
  convergent Born series forward model consumes.

Both are regression-pinned against Kirkland Table 2.1 reference values
in `tests/test_ptyrodactyl/test_tools/test_constants.py`.

## Plan-01 Phase P6

### Carrier, Alias, And Constant Imports

Shared carriers, scalar aliases, constants, and validated constructors now
live only under `ptyrodactyl.types`. Imports from `ptyrodactyl.tools` for
electron carriers, crystal carriers, jacobian carriers, scalar aliases, and
physical constants were removed.

### Factory Renames

The legacy `make_*` constructor names were removed. Use the canonical
`create_*` factories from `ptyrodactyl.types`, including
`create_calibrated_array`, `create_probe_modes`,
`create_potential_slices`, `create_crystal_structure`,
`create_crystal_data`, `create_stem4d`, and
`create_ptycho_params`.

### Scalar Alias Renames

PascalCase scalar aliases were removed in favor of snake_case aliases:
`ScalarFloat` -> `scalar_float`, `ScalarInt` -> `scalar_int`,
`ScalarNumeric` -> `scalar_num`, and `NonJaxNumber` ->
`non_jax_number`.

### Validation Behavior

Factories now raise on invalid input through two-tier validation:
static shape and structure violations raise `ValueError`, and traced
data-dependent violations raise through `eqx.error_if`. The previous
NaN-poisoning behavior was removed.

Silent data mutation during construction was also removed. Factories no
longer repair invalid values with `abs() + eps` or `jnp.clip`; callers
must pass valid inputs explicitly.

### Constants

Physical constants are now canonical 0-d weakly typed JAX arrays exported
from `ptyrodactyl.types`. The derived electron-optics functions
`relativistic_wavelength_ang`, `interaction_parameter`, and
`relativistic_mass` remain in `ptyrodactyl.tools.constants` and are still
exported from `ptyrodactyl.tools`.

### Deleted Files

Deleted source modules:

- `src/ptyrodactyl/tools/electron_types.py`
- `src/ptyrodactyl/tools/factory.py`

Deleted mirror tests:

- `tests/test_ptyrodactyl/test_tools/test_electron_types.py`
- `tests/test_ptyrodactyl/test_tools/test_factory.py`

### Documented Exception

Optimizer carriers and Wirtinger optimizer helpers remain in
`ptyrodactyl.tools.optimizers` until Plan 04.
