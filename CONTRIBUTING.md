# Contributing to ptyrodactyl

Thank you for contributing to ptyrodactyl. This guide defines the standards
for physics, differentiability, types, documentation, testing, and packaging.

The runtime repository and the planning repository are separate Git
repositories. A local `ptyrodactyl-plans` symlink can expose the plans beside
the code, but plan state does not belong in runtime names. Follow the current
authorized plan when a change is plan-driven. Follow this guide for every new
or changed code surface.

## Core Principle: Invertible Modularity

Ptyrodactyl composes electron-scattering forward models from differentiable
operators. The same operator boundaries are the boundaries of the inverse
problem. A loss can attach to a detector terminal or to an intermediate field
and optimize the potential, atom positions, probe, aberrations, scan positions,
or other declared parameters.

This design rests on one invariant:

> **Reductions stay explicit, late, and differentiable. No module collapses
> information it is not forced to.**

Apply this invariant as follows:

- Keep incident, intermediate, exit, and detector-plane wavefields complex.
  Apply `|psi|^2` only at an explicit detector reduction.
- Sum coherent samples as amplitudes before the modulus square. Sum
  incoherent samples as weighted intensities after the modulus square.
- Represent probe modes, position jitter, and partial coherence through an
  explicit `Distribution`. Do not hide these averages inside a propagation
  kernel.
- Keep a three-dimensional `Potential3D` unsliced when the downstream
  Galerkin, convergent-Born, or Bloch model needs a volume. Projection and
  multislice slicing are explicit model choices, not implicit producer
  behavior.
- Keep the detector reduction separate from reusable amplitude producers such
  as `cbed_amplitude`.
- Treat a gradient as part of the physics. A zero, NaN, wrong-sign, or
  conjugated gradient is a physics defect even when the forward value looks
  correct.

A premature reduction can preserve a forward image while destroying one
inverse seam. Review every non-differentiable operation and every reduction
with that failure mode in mind.

## Development Setup

### Prerequisites

- Python 3.12 through 3.14. The project metadata declares
  `>=3.12,<3.15`. Python 3.11 is not supported.
- [uv](https://docs.astral.sh/uv/) for environments, locking, builds, and
  publishing.
- Git.
- A CUDA-capable GPU is optional. CPU execution is the reference development
  path.

JAX 0.11 requires Python 3.12 or newer. Do not test a JAX 0.11 change only in
an old Python 3.11 environment. The supported Python matrix is 3.12, 3.13, and
3.14.

### Installation

Clone the repository and synchronize the development extra:

```bash
git clone https://github.com/debangshu-mukherjee/ptyrodactyl.git
cd ptyrodactyl
python --version
uv sync --extra dev
```

Use the CUDA development extra only on a supported Linux system:

```bash
uv sync --extra dev_cuda
```

The available extras are `cuda`, `docs`, `test`, `notebooks`, `dev`, and
`dev_cuda`. Do not document or invoke an `all` extra unless the project adds
one.

Use the lock file for normal development and CI-like checks:

```bash
uv lock --check
uv run --frozen python -c "import ptyrodactyl; print(ptyrodactyl.__version__)"
```

### Import-time environment contract

`src/ptyrodactyl/__init__.py` owns import-time JAX configuration. It performs
these actions before it imports the public subpackages:

- merges the package CPU defaults into an existing `XLA_FLAGS` value;
- honors `PTYRODACTYL_DISABLE_RUNTIME_CHECKS=1` through `EQX_ON_ERROR`;
- imports JAX and enables 64-bit precision;
- optionally initializes distributed execution through
  `PTYRODACTYL_DISTRIBUTED=1`.

Do not move physical-constant materialization ahead of the x64 configuration.
Do not overwrite operator-supplied XLA flags. Keep new import-time side
effects in the top-level package and justify them in review.

### Project structure

```text
ptyrodactyl/
├── src/ptyrodactyl/
│   ├── types/          # carriers, aliases, constants, create_* factories
│   ├── inout/          # parsers, coefficient data, and HDF5 ingest/emit
│   ├── ucell/          # lattice, rotation, and crystal geometry
│   ├── multislice/     # IAM potentials, forward models, and reconstruction
│   ├── born/ # convergent-Born-series helpers
│   ├── galerkin/       # scalar Fourier-Galerkin Helmholtz operators
│   ├── bloch/          # Bloch-wave operators
│   ├── jacobian/       # Jacobian, Fisher, gauge, and solver operations
│   ├── plots/          # presentation-only helpers
│   ├── _tools/         # private cross-family implementation seams
│   └── workflows/      # end-to-end compositions
├── tests/
│   ├── test_ptyrodactyl/  # mirrors the source package
│   └── test_data/         # fixtures and pinned regression artifacts
├── docs/source/           # Sphinx sources
└── tutorials/             # tutorial notebooks
```

The package boundaries have specific roles:

- `types` owns shared carriers and their validation contracts.
- `inout` is the host-world boundary. Filesystem access, NumPy parsing, HDF5,
  and packaged lookup data belong here.
- `multislice`, `galerkin`, `born`, and `bloch` are sibling forward-model
  families. Family-specific reconstruction belongs with its forward model;
  `multislice` therefore owns multislice reconstruction.
- `jacobian` owns derivative operations shared across forward families.
- `_tools` is the sole private cross-subpackage infrastructure boundary. Its
  owned leaves are `canonical_digest`, `host_interval`, `interval`, `numeric`,
  and `physics`. It is not a public package. Each leaf exports its unprefixed
  internal seams, and `_tools.__init__` re-exports their union for consumers;
  helpers local to one leaf remain `_`-prefixed. `_tools` stays
  dependency-neutral and does not import domain packages.
- `workflows` composes lower-level APIs. It does not become a second owner for
  their symbols.

Keep Python leaf modules out of `src/ptyrodactyl`. The root contains only
`__init__.py`; `py.typed` is the package-level PEP 561 marker. Put a shared
private seam in `_tools` only when multiple owning packages consume it.

Read `docs/source/organization.md` before changing a package boundary.

## Coding Standards

### JAX-first development

Ptyrodactyl uses JAX for traced, differentiable numerical work. Follow these
rules for array kernels:

- Keep functions pure. Do not mutate global state or perform I/O in a traced
  function.
- Use vectorization or `jax.lax.scan` for loops over traced array data.
  Python loops remain valid for small static structures and host-side I/O.
- Use `jax.lax.cond`, `jax.lax.switch`, or `jnp.where` for data-dependent
  traced control flow.
- Use `.at[...]` updates instead of in-place mutation.
- Keep array shapes stable under `jax.jit`, `jax.grad`, and `jax.vmap`.
- Mark only genuine compile-time structure as static. A value that should carry
  a gradient must not appear in `static_argnames`.
- Place JIT at a useful boundary. A tiny helper does not need its own JIT if a
  public caller already compiles the complete operation.
- Call `jax.block_until_ready` in timing tests and in tests that must force a
  deferred runtime error.

```python
# Wrong: Python flow depends on traced array values.
def positive_scale(values):
    output = []
    for value in values:
        if value > 0:
            output.append(2.0 * value)
    return jnp.asarray(output)


# Correct: the output shape is stable and the operation is traceable.
@jaxtyped(typechecker=beartype)
def positive_scale(
    values: Float[Array, " n"],
) -> Float[Array, " n"]:
    scaled: Float[Array, " n"] = jnp.where(
        values > 0.0,
        2.0 * values,
        values,
    )
    return scaled
```

### JAX 0.11 and jaxtyping 0.3 lessons

JAX 0.11 and jaxtyping 0.3.11 are compatible for dtype and shape
annotations. Do not add a dependency cap to avoid an invalid decorator stack.
Fix the stack.

Runtime type checking must wrap the original Python function. The JAX
transformation then wraps that checked function. Write the JAX decorator
outermost and the jaxtyping decorator immediately above the function:

```python
@jax.jit
@jaxtyped(typechecker=beartype)
def transmission(
    potential: Float[Array, "H W"],
) -> Complex[Array, "H W"]:
    result: Complex[Array, "H W"] = jnp.exp(1j * potential)
    return result
```

For static arguments, use the direct JAX decorator factory:

```python
@jax.jit(static_argnames=("grid_shape",))
@jaxtyped(typechecker=beartype)
def build_grid(
    grid_shape: tuple[int, int, int],
    spacing: Float[Array, ""],
) -> Float[Array, "Nx Ny Nz"]:
    """..."""
```

Do not write this stack:

```python
@jaxtyped(typechecker=beartype)
@jax.jit
def wrong_order(...): ...
```

Do not use `functools.partial(jax.jit, ...)` as a decorator factory. The direct
`@jax.jit(...)` form gives `ty` a usable callable signature and avoids an
unnecessary wrapper. Runtime type checking must receive the original Python
function, not a JAX `PjitFunction`. A wrong order can fail during import when
jaxtyping inspects `__globals__`. The AST guard in
`tests/test_ptyrodactyl/test_package_structure.py` rejects the reversed stack.

JAX 0.11 also makes the uninitialized contract of `jnp.empty` observable. Do
not assume that `jnp.empty` contains zeros or any deterministic value. Use
`jnp.zeros` when zero initialization matters. Use `jnp.full` when another
initial value matters. Use `jnp.empty` only when every element is definitely
written before any read, and test that property. The production source should
normally have no need for `jnp.empty`. The current production tree has no
`jnp.empty` call. Its one host-side `np.empty` allocation is fully populated
by the form-factor loader before use.

When changing JAX, jaxtyping, beartype, NumPy, or Python support, run a fresh
unconstrained wheel installation in addition to the locked development suite.
The fresh check is described under Packaging.

### Numerical and differentiability rules

- Prefer JAX, `jax.scipy`, Equinox, Optax, SciPy, and NumPy primitives already
  declared by the project over a new local implementation.
- Do not add a numerical dependency only because another repository uses it.
  Add dependencies in the owning, reviewed environment change.
- Require `jax.grad` to agree with a central finite difference for every new
  differentiable primitive. Check real and imaginary directions separately
  when a complex field is optimized.
- A finite gradient is not enough. Add a nonzero-sensitivity tripwire when the
  physics predicts sensitivity.
- Keep real-valued objectives real. State the complex-gradient convention for
  any optimizer-facing API.
- Preserve the captured convention of the historical complex optimizer path
  until its owning migration changes it. Do not silently mix a Wirtinger
  derivative, a conjugated gradient, and a real-coordinate optimizer.
- Guard unsafe branch inputs before evaluating operations that can create NaN
  or infinity. Sanitizing only the selected output of `jnp.where` does not
  make the unused branch gradient safe.
- Do not differentiate raw eigenvectors at a degeneracy. Differentiate a
  gauge-invariant projector, spectral quantity, or another declared invariant.

### Type hints with jaxtyping and beartype

Every public array function uses `@jaxtyped(typechecker=beartype)` and a
jaxtyping dtype/shape contract.

Pure-Python host functions use precise Python annotations, beartype where it
adds value, and explicit schema validation. Do not force a JAX array contract
onto the HDF5 or filesystem boundary.

```python
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from ptyrodactyl.types import ProbeModes, scalar_float


@jaxtyped(typechecker=beartype)
def weighted_probe(
    probe: ProbeModes,
    scale: scalar_float,
) -> Complex[Array, "H W M"]:
    scaled_modes: Complex[Array, "H W M"] = (
        probe.modes * jnp.asarray(scale)
    )
    return scaled_modes
```

Apply these rules:

- Annotate every parameter and return value.
- Annotate important intermediate arrays and every returned variable.
- Assign before returning. Bind the result to a type-annotated name, then
  return that name.
- Use descriptive dimension names. Examples include
  `Complex[Array, "H W M"]`, `Float[Array, "P H W"]`, and
  `Float[Array, "N 3"]`.
- Use width-qualified jaxtyping dtypes when an array has a canonical storage
  contract. Examples include `Float64`, `Complex128`, `Int64`, and `Int32`.
  Apply them to carrier fields, post-conversion locals, and returned arrays
  whose producer guarantees that width.
- Keep `Float`, `Complex`, and `Int` only at an explicit conversion boundary
  or in a genuinely dtype-polymorphic function. An exact dtype annotation is
  an assertion, not a cast. Convert first, then annotate the converted value.
- Use `Float32` or `Complex64` when single-precision storage is intentional.
  Do not describe a width-qualified annotation as proof of numerical accuracy
  or an outward rounding bound.
- Use `Float[Array, ""]` for a rank-zero float array. Follow the existing
  whitespace convention only where a touched module requires migration-free
  consistency.
- Prefer `scalar_float`, `scalar_int`, `scalar_bool`, and `scalar_num` from
  `ptyrodactyl.types` for public scalar arguments that accept Python and JAX
  scalars.
- Use capital `Tuple[...]` and `Dict[...]` hints throughout `src/` and
  `tests/`, importing both from `beartype.typing`. Do not use the built-in
  `tuple[...]` or `dict[...]` generic forms, and never import or qualify the
  capital forms from `typing` or `typing_extensions`. Import other runtime
  typing constructs such as `Optional`, `Union`, and `List` from
  `beartype.typing` when beartype must inspect them.
- Import shared carriers, aliases, and constants from `ptyrodactyl.types`.
  Do not redefine them in a consuming module.

For host-side NumPy arrays, never annotate a bare `np.ndarray`. Import
`NDArray` from `numpy.typing` and place it inside a jaxtyping specification:

```python
from jaxtyping import Float, Int
from numpy.typing import NDArray


def accumulate_counts(
    values: Float[NDArray, " n"],
    counts: Int[NDArray, " n"],
) -> Float[NDArray, " n"]:
    weighted: Float[NDArray, " n"] = values * counts
    return weighted
```

Do not alias `numpy.ndarray` as `NDArray`. Do not use a bare `NDArray`, which
leaves its scalar type variable unbound.

### Imports and ownership

New and touched cross-subpackage imports use the public owning subpackage. For
example, use `from ptyrodactyl.types import Potential3D`. Do not add a reach
into `ptyrodactyl.types.potential_types` from another subpackage. Some legacy
deep imports remain; migrate them only in an authorized owning change. The
only exception is an import from the private `ptyrodactyl._tools` aggregate.

A cross-subpackage name outside `ptyrodactyl._tools` is public. Its leaf
module, package `__init__.py`, `__all__`, and Routine Listings must expose it.
Imports from `_tools` form an unsupported internal seam: use an unprefixed
name for a seam consumed outside its leaf, re-export it through `_tools`, and
reserve `_` prefixes for helpers local to that leaf. Consumers import from
`ptyrodactyl._tools`, never from one of its leaves. Deep relative imports are
for private communication inside one subpackage.

Do not add renamed ptyrodactyl imports in new or touched code. Standard
ecosystem aliases such as `jnp`, `np`, `eqx`, and `plt` remain valid. Migrate
an existing project alias only when the change owns all affected call sites.

### Custom types and PyTrees

`ptyrodactyl.types` is the canonical home for shared carriers, PyTrees, type
aliases, physical constants, and validated `create_*` factories. A result
container is still a type. Do not define it beside its producer merely because
only one producer uses it today.

Use `eqx.Module` for structured numerical data. Dynamic array fields remain
PyTree leaves. Declare non-array metadata with `eqx.field(static=True)` only
when it must not participate in transformations.

```python
class Distribution(eqx.Module):
    """Store a weighted distribution over latent samples."""

    samples: Float[Array, "N D"]
    weights: Float[Array, "N"]
    reduction: ReductionMode = eqx.field(static=True)
    axis_id: Optional[str] = eqx.field(static=True, default=None)
```

Static metadata participates in compilation cache identity. Keep it small,
hashable, and scientifically explicit. Do not mark a physical fit parameter
static merely to make tracing easier.

Construct public carriers through their `create_*` factory. Direct Equinox
construction is for the factory implementation and narrowly justified tests.

### Validation in factories

Use two validation tiers:

- Use Python `ValueError` for static rank, shape, enum, string, and structure
  checks that tracing can resolve.
- Use `equinox.error_if` for finite, sign, bound, normalization, and other
  value checks that can receive tracers.

```python
import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Float64, jaxtyped

from ptyrodactyl.types import LobatoParameters


@jaxtyped(typechecker=beartype)
def create_lobato_parameters(
    amplitudes: Float[Array, "..."],
    scales: Float[Array, "..."],
) -> LobatoParameters:
    amplitudes_array: Float64[Array, " 5"] = jnp.asarray(
        amplitudes,
        dtype=jnp.float64,
    )
    scales_array: Float64[Array, " 5"] = jnp.asarray(
        scales,
        dtype=jnp.float64,
    )
    if amplitudes_array.shape != (5,) or scales_array.shape != (5,):
        raise ValueError("Lobato coefficients must have shape (5,)")

    checked_amplitudes: Float64[Array, " 5"] = eqx.error_if(
        amplitudes_array,
        jnp.any(~jnp.isfinite(amplitudes_array)),
        "Lobato amplitudes must be finite",
    )
    checked_scales: Float64[Array, " 5"] = eqx.error_if(
        scales_array,
        jnp.any(~jnp.isfinite(scales_array))
        | jnp.any(scales_array <= 0.0),
        "Lobato scales must be finite and positive",
    )
    parameters: LobatoParameters = LobatoParameters(
        amplitudes=checked_amplitudes,
        scales=checked_scales,
    )
    return parameters
```

Attach the checked value to the returned computation. An unused
`eqx.error_if` result can be removed by tracing. Reject Python booleans
explicitly when `bool` could masquerade as atomic number zero or one.

Do not replace rejection with NaN poisoning. Invalid scientific inputs must
fail closed in eager and compiled execution.

### Units, coordinates, and physical conventions

State units on every public physical quantity. Current canonical conventions
include:

- lengths and real-space coordinates in Angstroms;
- reciprocal spatial frequencies in inverse Angstroms;
- angular scattering-vector magnitude `q` in radians per Angstrom, with
  `g = q / (2*pi)` in cycles per Angstrom where required;
- accelerating voltage in kilovolts at microscope API boundaries;
- electrostatic `Potential3D.volume` in volts;
- projected `PotentialSlices` and projected atomic potentials in
  volt-Angstroms;
- aperture and tilt angles in milliradians unless a signature says otherwise;
- crystallographic cell angles in degrees;
- `Potential3D.volume` storage order `(z, y, x)`, while geometry tuples use
  physical `(x, y, z)` order.

Do not use `eV` as a synonym for an electrostatic voltage field. Preserve the
declared additive potential reference, boundary convention, coefficient
normalization, band limit, and provenance. Do not silently subtract a mean or
discard the Fourier zero mode.

Use standard zero-based, end-exclusive Python indexing. State axis order in
the type annotation and docstring when it is not evident.

Lobato--Van Dyck is the default independent-atom parameterization. Kirkland is
available only through explicit `parameterization="kirkland"`. Do not create a
second Kirkland-default entry point or duplicate either coefficient loader.

### Naming uses domain terms

Name code for electron microscopy, physics, mathematics, or software
structure. Do not put plan numbers, work-package labels, gates, stages,
sprints, or agent names in public APIs, modules, tests, or new data schemas.

```python
# Wrong: a tracking label hides the verified property.
def test_plan06_gate_lb3() -> None: ...


# Correct: the name states the physics property.
def test_projected_potential_matches_zero_frequency_slice() -> None: ...
```

Historical regression artifacts can retain stable names when a rename would
break their audit trail. New scientific tests and APIs use domain terms.

## Documentation Standards

### Numpydoc and technical prose

Docstrings use the NumPy/numpydoc convention. Ruff enforces the configured
docstring rules. `pyproject.toml` contains the current pydoclint settings; do
not weaken a correct jaxtyping annotation merely to satisfy a parser.

Write repository prose in Simplified Technical English:

- Keep instructions to 20 words when practical.
- Keep descriptions to 25 words when practical.
- Use active voice and identify the actor.
- Use present tense for descriptions and the imperative for instructions.
- Use one term for one concept.
- State units and axis order explicitly.
- Avoid idioms and long noun clusters.
- Keep technical names such as PyTree, 4D-STEM, Lobato--Van Dyck, and
  `jnp.where` consistent.

These rules apply to docstrings, Markdown, and notebook Markdown cells.

### Module docstrings and public exports

Every public module contains:

1. a one-line summary;
2. an `Extended Summary` when the module needs more context;
3. a `Routine Listings` section for every public symbol;
4. `Notes` or `References` when required by the physics.

A package `__init__.py` also lists its public submodules in its extended
summary. Keep submodules alphabetical. Group Routine Listings as classes,
functions, then objects. Sort each group alphabetically.

Use `:class:` for carriers and classes, `:func:` for functions, `:obj:` for
aliases and constants, and `:mod:` for submodules.

List every public symbol in four synchronized locations:

1. the leaf module Routine Listings;
2. the leaf module `__all__`;
3. the subpackage Routine Listings;
4. the subpackage `__all__`.

Copy the symbol's summary verbatim into both Routine Listings. Export each
symbol from exactly one owning leaf module and one owning subpackage. The
package-structure tests enforce this contract on the migrated surfaces.

When a symbol moves, update every consumer and delete the old export in the
same change. Do not add a forwarding alias, compatibility module, or
`DeprecationWarning`. Record the migration in `CHANGELOG.md`.

### Function docstrings

A public function docstring uses the following order, omitting sections that
do not apply:

1. summary line;
2. extended summary;
3. `:see:` test cross-reference;
4. `Implementation Logic`;
5. `Parameters`;
6. `Returns` or `Yields`;
7. `Raises`;
8. `Notes`;
9. `References`;
10. `See Also`;
11. `Examples`.

Use an imperative summary that fits on one line. Copy that summary verbatim
to both Routine Listings.

Document every parameter in signature order. Copy the annotated type. State
units and defaults. Mark a static argument and explain that changing it causes
retracing.

Name every returned value after the annotated local variable returned by the
body. For a tuple, document each element in order. Document every explicit
`ValueError`. Document traced runtime rejection as the relevant Equinox or JAX
runtime error when that detail helps callers.

Use `Implementation Logic` for multi-step algorithms. Keep the steps aligned
with the code. Use `Notes` for a direct formula and for approximation,
boundary, normalization, gauge, and differentiability limits.

Give every public object a `:see:` reference to its test class or test module.
Give the test a back-reference to the object. Land both sides together.

Use a raw docstring when LaTeX contains backslashes. Use unique reference
footnote labels across one module because Sphinx renders the module on one
page.

### Private function docstrings

A `_`-prefixed function in a source module receives a fully structured
numpydoc docstring. Use the public section order, including `Parameters`,
`Returns`, `Raises`, and `Notes` where they apply.

Apply these rules:

- Start the summary line with `PRIVATE:` and continue with the imperative
  summary.
- Keep private functions out of `__all__` and every Routine Listings section.
- Do not add a `:see:` test cross-reference. Verify private behavior through
  the public callers.

```python
def _outward_sqrt(
    value: Float64[Array, "..."],
) -> Float64[Array, "..."]:
    """PRIVATE: Round one non-negative binary64 square root upward.

    Parameters
    ----------
    value : Float64[Array, "..."]
        Non-negative radicand.

    Returns
    -------
    result : Float64[Array, "..."]
        Square root rounded one ULP toward positive infinity, with an
        exact zero preserved.
    """
```

### Class and factory docstrings

An `eqx.Module` docstring documents every field in declaration order under
`Attributes`. Mark every `eqx.field(static=True)` value as static. State its
units and interpretation.

Do not add a custom constructor docstring for an Equinox carrier. Put the
construction and validation contract on the `create_*` factory. Name the
factory in `See Also`.

The factory docstring separates structural `ValueError` checks from traced
value checks. Its `Returns` name matches the annotated carrier variable.

### Code style

Ruff configures a 79-character line length, Python 3.12 syntax, double quotes,
and import sorting. Follow these additional rules:

- Use descriptive `snake_case` names. Use a short symbol only when it matches
  a displayed physical formula.
- Put explanations in docstrings. Keep inline comments for necessary rationale
  or tool directives.
- Keep imports at module scope unless an optional dependency or a documented
  cycle requires a local import.
- Use no star imports.
- Keep `__all__` a literal list so the structure tests can inspect it.
- Preserve user-visible exceptions and messages when a change does not own an
  API break.

## Testing

The suite uses pytest, chex, NumPy testing helpers, pytest-cov, and
pytest-xdist. The pytest configuration activates live jaxtyping and beartype
checking with `--jaxtyping-packages=ptyrodactyl,beartype.beartype`.

Hypothesis is not currently a declared test dependency. Do not add an
untracked property-test dependency in an unrelated change. Use parameterized
deterministic cases until the environment change that owns Hypothesis lands.

### Layout

Tests mirror the source tree:

```text
tests/test_ptyrodactyl/
├── test_inout/
├── test_ucell/
├── test_multislice/
├── test_born/
├── test_galerkin/
├── test_bloch/
├── test_jacobian/
├── test_plots/
├── test_types/
└── test_workflows/
```

Name a test file `test_<module>.py`. Name a test class `Test<Symbol>` when a
class groups one public symbol. Name a test function for the property it
verifies.

Test modules do not define `__all__` or Routine Listings. They are not a
public import surface.

### Scientific evidence

Use the strongest available independent truth:

- a closed-form result;
- a conservation, symmetry, normalization, adjoint, or gauge invariant;
- a published table value with its units and convention;
- an independently generated and provenance-pinned artifact;
- systematic convergence under grid, band-limit, slice, or solver
  refinement.

A stored output from ptyrodactyl is a regression capture, not an independent
physics oracle. Use it to prove unchanged behavior for the same
parameterization. Pair it with physics evidence when correctness is at issue.

Keep external generators and heavyweight reference implementations outside
the runtime package and normal test environment unless their dependency is
explicitly declared. Commit immutable data, hashes, conventions, and
provenance. Tests consume the artifact; they do not silently regenerate it.

For atomic form factors, examples of independent anchors include the hydrogen
Bethe normalization, published band-limited Lobato values, analytic projection
identities, and refinement convergence. Kirkland-to-Lobato closeness is a
cross-check, not a proof that either model is exact.

### Required numerical tests

For each changed numerical primitive, test the applicable properties:

- values and units against independent evidence;
- shape, dtype, and finiteness;
- eager and `jax.jit` agreement;
- `jax.vmap` compatibility;
- `jax.grad` against central finite differences;
- a nonzero-gradient tripwire for sensitive parameters;
- rejection of invalid Python values;
- rejection of invalid traced values;
- translation, periodicity, additivity, symmetry, or gauge behavior;
- convergence under the discretization the function introduces.

Prefer `chex.assert_shape`, `chex.assert_tree_all_finite`, and
`chex.assert_trees_all_close` for arrays and PyTrees. NumPy's testing helpers
remain appropriate for scalar and independent reference comparisons. A plain
`assert` is appropriate for metadata and exact Boolean contracts.

### Two-tier rejection tests

Exercise both validation tiers. A traced error can be deferred until the
result is consumed:

```python
with pytest.raises(ValueError, match="must be positive"):
    create_example(jnp.ones((2, 2)), spacing=-1.0)

compiled = jax.jit(lambda spacing: create_example(jnp.ones((2, 2)), spacing))
with pytest.raises(
    (equinox.EquinoxRuntimeError, jax.errors.JaxRuntimeError, ValueError),
    match="must be positive",
):
    result = compiled(jnp.asarray(-1.0))
    jax.block_until_ready(result)
```

Do not accept a traced invalid value merely because the eager path rejects a
Python scalar.

### Test documentation

Treat a test docstring as a compact specification. State what property the
test proves and how it proves it. Include units, parameterization, and
tolerances. Add bidirectional `:see:` references on migrated documentation
surfaces.

Keep private helpers local and `_`-prefixed. Move a helper into shared test
support only after more than one module needs it.

### Test type annotations and the annotation pre-flight

Annotate every test function signature. Test functions return `None`.
Annotate fixture and parametrized arguments with the same jaxtyping
specifications as source code. Decorate array-typed helpers defined inside
tests with `@jaxtyped(typechecker=beartype)`. Body locals may stay
unannotated. Ruff does not enforce test annotations; keep them present and
correct in every new or touched test.

`pytest` runs an annotation gate before it collects one test. The gate is
`tests/_preflight_types.py`. It imports every module in `ptyrodactyl` and
`tests` while the jaxtyping import hook is active. The hook decorates every
function with `@jaxtyped(typechecker=beartype)`, and decoration evaluates
each annotation. An invalid annotation therefore fails the session before
collection.

The gate rejects three defect classes without running one test:

- a malformed jaxtyping specification, such as a wrong dtype name or a wrong
  shape string;
- a name that an annotation uses but the module does not import;
- a hint that beartype cannot use, such as a bare `NDArray`.

Run the gate alone during development:

```bash
uv run --frozen python tests/_preflight_types.py
```

Set `PTYRODACTYL_SKIP_PREFLIGHT=1` to skip the gate for one fast local run.
Do not skip it before you submit a pull request.

The gate does not detect a wrong dtype at runtime, because that defect
requires real array values. The test suite detects that defect for `src/`,
where the pytest jaxtyping hook checks every signature. Annotations in
`tests/` carry no runtime check, so keep them correct by inspection.

### Running tests

Run the full deterministic CPU suite:

```bash
MPLCONFIGDIR=/tmp/ptyrodactyl-mpl uv run --frozen pytest -q -n 0
```

Run a focused module or test:

```bash
uv run --frozen pytest -q -n 0 \
  tests/test_ptyrodactyl/test_multislice/test_form_factors.py
uv run --frozen pytest -q -n 0 \
  tests/test_ptyrodactyl/test_multislice/test_form_factors.py::test_hydrogen_zero_angle_form_factor_is_bohr_radius
```

Run coverage when a change affects a broad surface:

```bash
uv run --frozen pytest tests/ \
  --cov=src/ptyrodactyl --cov-report=term-missing
```

The historical Plan-01 numerical capture has its own verifier:

```bash
MPLCONFIGDIR=/tmp/ptyrodactyl-mpl \
  uv run --frozen python tests/test_data/plan01_ug_capture.py verify
```

Do not update a pinned capture merely to make a failure disappear. Explain the
intended behavior change and update its provenance in the same review.

## Tutorial Notebooks

The current tutorial tree contains committed `.ipynb` notebooks. Keep
explanations, motivation, units, and physics in Markdown cells. Keep code cells
small and reproducible. Remove outputs before committing unless an owning
documentation change explicitly requires rendered output.

The repository does not yet declare Jupytext or provide paired percent
scripts. Do not claim that notebooks are paired until the tooling and pairs
land together. If the project adopts Jupytext, update this section and commit
the `.ipynb` and `.py` pair in sync.

## Pull Request Process

### Before submitting

Run the gates that the current repository declares:

```bash
uv lock --check
uv run --frozen ruff check pyproject.toml src tests
uv run --frozen ruff format --check src tests
uv run --frozen ty check
git diff --check
MPLCONFIGDIR=/tmp/ptyrodactyl-mpl uv run --frozen pytest -q -n 0
```

Build documentation with warnings as errors after documentation changes:

```bash
uv sync --extra docs
uv run --frozen sphinx-build -a -E -W -b html \
  docs/source docs/build/html
```

`pyproject.toml` contains pydoclint and interrogate configuration, but the
current `dev` extra does not declare every associated executable. Run these
checks when the executable exists in the active environment. Do not describe
them as a reproducible required gate until their dependencies and baseline are
declared in the repository.

The repository declares pre-commit hooks in `.pre-commit-config.yaml`. The
hooks run the locked ruff check, ruff format, `ty check`, and the local
lines-of-code badge through `uv run`. Install them once per clone:

```bash
uv run pre-commit install
```

The hooks complement the explicit commands above; they do not replace the
full test suite or the documentation build.

### PR description and review

Use a descriptive branch name, such as `feature/lobato-volume` or
`fix/projected-potential-gradient`.

Write an imperative commit subject. In the PR description, state:

- the behavior and boundary changed;
- the scientific or software reason;
- the independent evidence and test commands;
- gradient, unit, sign, and parameterization effects;
- public API and schema changes;
- fresh-install evidence when dependencies or packaging changed.

Differentiability is an acceptance criterion. A change that breaks a touched
gradient seam has failed even if its forward regression tests pass.

## API Evolution: Zero Legacy

Ptyrodactyl is pre-1.0 and uses a zero-legacy policy:

- Add no compatibility shim, forwarding alias, duplicate re-export, or
  `DeprecationWarning` for a removed API.
- Update every call site and delete the old path in the same change.
- Keep one implementation and one canonical import path.
- Record the migration in `CHANGELOG.md`.
- Preserve an explicitly supported secondary physical model through a named
  option, not a second default entry point.

This policy does not authorize an unrelated API break. Scope the break to its
owning change and include migration notes.

## Packaging and Fresh-Environment Validation

`pyproject.toml` is the source of truth for package metadata and the CalVer
version. The build backend is `uv_build`. Use uv for builds and publication.

```bash
uv build
uv publish
```

Do not use an editable checkout as the only packaging test. A wheel can omit
coefficient tables, `py.typed` markers, or other package data even when source
tests pass.

For dependency, Python, JAX, jaxtyping, NumPy, package-data, or build changes,
create a disposable environment and let the wheel metadata resolve the newest
compatible dependencies. Do not reuse `uv.lock`, a constraint file, or the
project `.venv` for this check.

```bash
fresh_dir="$(mktemp -d)"
uv build --wheel --out-dir "$fresh_dir/dist"
uv venv --python 3.13 --no-project "$fresh_dir/venv"
uv pip install --python "$fresh_dir/venv/bin/python" \
  "$fresh_dir"/dist/ptyrodactyl-*.whl
(
  cd "$fresh_dir"
  "$fresh_dir/venv/bin/python" -c '
import jax, jaxtyping, numpy, ptyrodactyl
print(
    jax.__version__,
    jaxtyping.__version__,
    numpy.__version__,
    ptyrodactyl.__version__,
)
'
)
```

Run a small scientific smoke test from outside the source tree. Exercise a
public form-factor or forward-model path, not only `import ptyrodactyl`.
Confirm these properties:

- Python is in the supported 3.12--3.14 range;
- JAX 0.11 and jaxtyping 0.3.11 import together when they are the current
  unconstrained resolution;
- the top-level x64 setup takes effect;
- packaged Lobato and Kirkland assets load with the expected shapes;
- one JIT-compiled, runtime-typechecked public function executes;
- the result is finite and has the expected dtype and units.

Also build the source distribution for a release candidate. Build a wheel from
that source distribution and repeat the smoke test when package-data rules
changed.

Do not respond to a fresh-environment failure by adding arbitrary dependency
caps. Identify whether the defect is Python support, decorator order, changed
initialization semantics, removed API, or missing package data. Add a cap only
when an actual incompatibility requires one and the review records the reason.

## Issue Guidelines

A bug report includes a minimal reproducer, expected behavior, actual
behavior, Python version, JAX version, jaxtyping version, NumPy version,
platform, and accelerator type.

A wrong-gradient report also includes the differentiated parameter, scalar
loss, automatic derivative, finite-difference derivative, step size, and
tolerance.

A feature request states the scientific use case, required units and
conventions, intended module owner, differentiability requirements, and likely
performance effect.

## Getting Help

- Open a GitHub discussion or issue for questions.
- Read the Sphinx sources under `docs/source/`.
- Read `docs/source/organization.md` for package ownership.
- Consult the separate `ptyrodactyl-plans` repository for authorized roadmap
  scope and physics canons.

Thank you for advancing differentiable electron scattering and ptychography
with ptyrodactyl.
