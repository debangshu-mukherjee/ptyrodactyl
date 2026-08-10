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
  distribution producers, and 4D-STEM simulation. Its volumetric producer
  returns an unsliced `Potential3D`; it does not perform a beam-axis collapse.
- `ptyrodactyl.born` owns scalar Galerkin scattering operators, solvers,
  fixed-support derivatives, and Green-function utilities;
  `ptyrodactyl.bloch` is the sibling Bloch-wave forward family.
- `ptyrodactyl.plots` owns presentation-only image and colormap helpers.
- `ptyrodactyl.invert` and `ptyrodactyl.jacobian` own reconstruction and
  derivative operations.
- `ptyrodactyl.tools` contains numerical utilities, optimizer helpers, caching,
  and sharding support; it does not own domain carriers.
- `ptyrodactyl.workflows` composes the lower-level packages into end-to-end
  tasks.

## Source layout

```text
src/ptyrodactyl/
├── types/          # carriers, aliases, constants, validated constructors
├── inout/          # parsers, lookup assets, and HDF5 ingest/emit
├── ucell/          # unit-cell and rotation geometry
├── multislice/     # IAM potentials, multislice amplitudes, and reducers
├── born/           # scalar Galerkin scattering and Green-function utilities
├── bloch/          # Bloch-wave utilities
├── plots/          # visualization helpers
├── invert/         # reconstruction algorithms
├── jacobian/       # Jacobian, Fisher, gauge, and solver operations
├── tools/          # shared numerical/runtime utilities
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
6. Symbols are exported from one owning subpackage; removed `ptyrodactyl.simul`
   and `ptyrodactyl.tools.make_*` paths have no compatibility aliases.
7. Filesystem and third-party file-format access belongs in
   `ptyrodactyl.inout`, outside differentiable JAX kernels.
