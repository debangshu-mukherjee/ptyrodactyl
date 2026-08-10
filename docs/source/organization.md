# Package organization

Ptyrodactyl is organized around explicit, differentiable operator boundaries.
Structured carriers and constructors have one owner, forward-model families keep
complex amplitudes available until an explicit detector reduction, and host-side
I/O is separated from JAX kernels.

## Public subpackages

- `ptyrodactyl.types` owns Equinox carriers, including the volts-based
  `Potential3D` interchange field, scalar aliases, physical constants,
  distributions, and validated `create_*` constructors.
- `ptyrodactyl.inout` owns crystal-file parsers, bundled Lobato--Van Dyck and
  Kirkland coefficient data, atomic lookup data, and the versioned HDF5
  ingest/emit boundary for canonical carriers.
- `ptyrodactyl.ucell` owns lattice, rotation, and crystal-tilt operations.
- `ptyrodactyl.multislice` owns the Lobato-default independent-atom potential
  producers, projected multislice amplitudes, detector reductions,
  distribution producers, 4D-STEM simulation, and multislice reconstruction.
  Its volumetric producer returns an unsliced `Potential3D`; it does not
  perform a beam-axis collapse.
- `ptyrodactyl.galerkin` owns globally coupled scalar Fourier-Galerkin
  Helmholtz operators, solvers, sources, certificates, derivatives, and
  terminals.
- `ptyrodactyl.born` owns the distinct convergent-Born-series
  Green-function helpers; `ptyrodactyl.bloch` is the sibling Bloch-wave
  forward family.
- `ptyrodactyl.plots` owns presentation-only image and colormap helpers.
- `ptyrodactyl.jacobian` owns derivative operations shared across forward
  families.
- `ptyrodactyl.workflows` composes the lower-level packages into end-to-end
  tasks.

## Source layout

```text
src/ptyrodactyl/
├── types/          # carriers, aliases, constants, validated constructors
├── inout/          # parsers, lookup assets, and HDF5 ingest/emit
├── ucell/          # unit-cell and rotation geometry
├── multislice/     # IAM potentials, forward models, and reconstruction
├── born/ # convergent-Born-series Green-function helpers
├── galerkin/       # scalar Fourier-Galerkin Helmholtz scattering
├── bloch/          # Bloch-wave utilities
├── plots/          # visualization helpers
├── jacobian/       # Jacobian, Fisher, gauge, and solver operations
└── workflows/      # high-level orchestration
```

## Design rules

1. Array computation is JAX-first and supports `jax.jit`, `jax.grad`, and
   `jax.vmap` where the public contract promises them.
2. Public array functions use jaxtyping and runtime checking.
3. Canonical carrier fields and post-conversion arrays use width-qualified
   jaxtyping dtypes. Coercing inputs and dtype-polymorphic kernels use broad
   dtype families.
4. Carriers are constructed through `ptyrodactyl.types.create_*` functions.
5. Coherent/incoherent averaging is represented by an explicit `Distribution`
   and reduced after the complex amplitude kernel.
6. Symbols are exported from one owning subpackage; removed
   `ptyrodactyl.simul`, `ptyrodactyl.tools`, and `ptyrodactyl.invert` paths
   have no compatibility aliases.
7. Filesystem and third-party file-format access belongs in
   `ptyrodactyl.inout`, outside differentiable JAX kernels.
