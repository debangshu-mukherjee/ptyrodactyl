# Changelog

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
