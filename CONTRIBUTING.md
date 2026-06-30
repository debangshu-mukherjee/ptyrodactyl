# Contributing to ptyrodactyl

Thank you for your interest in contributing to ptyrodactyl! This guide describes how the
codebase is written — type hinting, documentation, validation, testing, and tooling — so
your contributions match the existing standards.

> **A note on target conventions.** This guide codifies the conventions the project is
> converging to under the `plans/` roadmap (plans 00–12). A few structural items — the
> single `ptyrodactyl.types` home, `eqx.Module` carriers, the `simul_multislice` /
> `simul_born` split, and the `recon` inverse-problem package — are **mid-rollout**: write
> *new* code to the target conventions below, and follow the migration in `plans/` when
> touching code that has not yet been moved.

## Core Principle: Invertible Modularity

ptyrodactyl's modules are differentiable operators, and the boundaries between them are the
boundaries at which the inverse problem is solved. A forward model built from clean but
*opaque* boxes can only be run forwards; one built from boxes that never discard a gradient
can be run backwards just as well — you attach a loss at any seam and solve for what produced
the data (the object transmission function, the probe, the 3D potential, the aberrations, the
scan positions, the sample orientation, the partial coherence) while freezing the rest. This
invertibility is the codebase's core asset.

It is also the headline: ptyrodactyl computes electron scattering with the **Convergent Born
Series** (exact 3D, no paraxial or slicing approximation), and because every seam of that
forward model keeps a gradient, **3D ptychography, 3D off-axis holography, and 3D focal-series
reconstruction are the same `solve()`** — one differentiable model, three measurement
terminals (see `plans/future/12_born_first_revision_plan.md`).

The principle rests on one invariant:

> **Reductions stay explicit, late, and differentiable. No module collapses information it
> is not forced to.**

Concretely, for electron ptychography / 4D-STEM:

- **Keep the wavefield complex; take `|ψ|²` once, at the detector plane only.** The exit
  wave, the propagated diffraction-plane field, and every intermediate must stay complex
  through the forward model. The single intensity reduction is the final detector step —
  never inside a reusable kernel that a downstream stage (a mixed-state sum, a coherent
  average, a multislice/Born propagation) might want to use coherently.
- **Express every average as an explicit weighted sum over a `Distribution`, summed late.**
  Mixed-state probe modes, scan-position jitter, partial spatial/temporal coherence, and
  finite source size are *incoherent distributions* over latent samples; their reduction is
  a weighted sum of intensities applied at the detector, not a convolution or quadrature
  baked into a kernel.
- **Prefer the analytic coherent-average limit** over a hard, irreversible collapse, and use
  `jnp.where` / `lax.cond` and continuous fields rather than discrete swaps or data-dependent
  Python control flow, so every parameter keeps a derivative.

The failure mode is silent: when a module performs a hard, non-differentiable, or premature
reduction, the forward model still looks correct — only invertibility breaks, and only at
that one seam. Treat any such reduction as a design smell to be justified explicitly in
review, not an implementation detail. The JAX-First rules below are the mechanics of
upholding this principle.

## Development Setup

### Prerequisites

- Python 3.13 (`requires-python = ">3.12, <3.15"`)
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Git
- CUDA-compatible GPU (optional, for acceleration)

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debangshu-mukherjee/ptyrodactyl.git
   cd ptyrodactyl
   ```

2. **Install in development mode:**
   ```bash
   # Everything (docs, tests, notebooks, dev tooling)
   uv sync --extra dev

   # With CUDA support as well
   uv sync --extra dev_cuda
   ```

   The dependency groups are defined in `pyproject.toml`.

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

The target module map (see `plans/future/00_rheedium_parity_roadmap.md` §2 and
`plans/future/12_born_first_revision_plan.md` §2):

```
ptyrodactyl/
├── src/ptyrodactyl/
│   ├── types/             # The single home for every carrier, alias, constant, create_* factory
│   ├── simul_multislice/  # Legacy multislice / projected-potential forward model (CBED, 4D-STEM) — faster
│   ├── simul_born/        # Convergent Born Series forward model (exact 3D) — the headline; TEM/CBED/4D-STEM terminals
│   ├── recon/             # Inverse problem: ReconProblem/ReconResult, solve(), losses, transforms, uncertainty, gauge, distributed Schwartz
│   ├── audit/             # Physics-invariant audit suite + reference benchmarking
│   ├── harness/           # Agent-runnable automaton process-boundary contract
│   ├── inout/             # The JAX↔world airlock: parsers, GPAW/DFT ingest, PyTree↔HDF5 codec
│   ├── ucell/             # Crystallography / unit-cell math (reciprocal lattice, rotations, wavelength)
│   ├── plots/             # Visualization utilities
│   └── tools/             # Numerical + parallel/distributed helpers only (no types, no optimizers)
├── tests/                 # Test suite (mirrors src layout, see below)
├── docs/                  # Sphinx documentation
├── data/                  # Sample data files
├── plans/                 # Planning documents (architecture + physics)
└── tutorials/             # Paired Jupyter notebooks (.ipynb + .py)
```

Each subpackage is a namespace package exposing its public API through `__init__.py` with an
explicit `__all__`. The top-level `src/ptyrodactyl/__init__.py` enables 64-bit precision
(`jax.config.update("jax_enable_x64", True)`) and sets CPU threading XLA flags **before** JAX
is imported, and optionally initializes multi-host distributed execution. Keep import-time
side effects confined to that module.

## Coding Standards

### JAX-First Development

ptyrodactyl is built on JAX for differentiable, high-performance computation. All new code
must follow JAX best practices:

**Required JAX Patterns:**
- Use `jax.lax.scan` instead of Python `for` loops over array data
- Use `jax.lax.cond` / `jnp.where` instead of data-dependent `if`/`else`
- Use `.at[].set()` for array updates instead of in-place modification
- Keep functions purely functional — no side effects, no global mutable state
- Decorate computational functions with `@jax.jit` where appropriate
- Code must remain traceable for `jit`, `grad`, `vmap`, and `pmap`

```python
# ❌ Wrong - Python loops and conditionals over array data
def bad_function(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i] * 2)
    return jnp.array(result)


# ✅ Correct - vectorized JAX
@jaxtyped(typechecker=beartype)
def good_function(x: Float[Array, " n"]) -> Float[Array, " n"]:
    return jnp.where(x > 0, x * 2, x)
```

### Prefer the ecosystem over hand-rolled numerics

**Before writing any numerical routine** — a solver, a fixed-point iteration, a special
function, an interpolation/rotation, a linear solve, a PyTree op — check for an
`equinox`-family / `jax.scipy` / ecosystem implementation **first**. Hand-roll only when none
exists, and **state the why in the PR**. The inversion engine is `optimistix` / `optax` /
`lineax` (with `blackjax` for posteriors); PyTrees are `equinox`; runtime typing is
`jaxtyping` / `beartype`; testing is `chex` / `hypothesis`. Re-implementing what the ecosystem
already solved is the anti-pattern this rule exists to stop.

### Optimizer & complex-parameter policy

Complex-valued parameters (object transmission, probe) are **real-ified** at the optimizer
boundary — carry `z = x + iy` as a stacked real pytree `(x, y)`, run *stock* `optax` /
`optimistix` on the resulting real problem, and re-complexify only inside the forward model.

- **Do not hand-roll Wirtinger derivatives or Wirtinger optimizers.** `jax.grad` of a
  real-valued loss already computes the Wirtinger gradient; on the real-ified representation
  the problem is an ordinary `ℝ^{2n} → ℝ` optimization where stock optimizers are correct by
  construction (no conjugation convention to get wrong, no `g²`-vs-`|g|²` second-moment trap).
- Real-ification is also what makes `optimistix`'s least-squares / minimisers correct — its
  complex support is explicitly "a work in progress, [that] may still return incorrect
  results."
- The 3D Born inverse problem recovers a **real** potential `φ(r)`; complex parameters appear
  only in the legacy 2D projected ptychography. See
  `plans/future/12_born_first_revision_plan.md` §1.

### Type Hinting with jaxtyping and beartype

Every public function is runtime-typechecked with the `@jaxtyped(typechecker=beartype)`
decorator stack and annotated with `jaxtyping` shape/dtype specs:

```python
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from ptyrodactyl.types import STEM4D, ReconProblem, scalar_float


@jaxtyped(typechecker=beartype)
def simulate_cbed(
    potential: Complex[Array, "H W"],
    beam: Complex[Array, "H W"],
    voltage_kV: scalar_float,
) -> Float[Array, "H W"]:
    """..."""
```

**Type Hinting Rules:**
- All parameters and return values are annotated; multiple returns use
  `beartype.typing.Tuple[...]`.
- Annotate intermediate variables inside function bodies too — e.g.
  `alpha_rad: Float[Array, ""] = jnp.deg2rad(alpha)`.
- **Assign before returning.** Bind a function's result to a type-annotated variable and
  return that name, rather than returning a bare expression — so the returned value carries an
  explicit type at its definition site.
- Use descriptive dimension names in shape specs: `Num[Array, "N 4"]`,
  `Float[Array, " n 3"]`, scalars as `Float[Array, ""]`.
- Prefer the scalar aliases from `ptyrodactyl.types` (`scalar_float`, `scalar_int`,
  `scalar_bool`, `scalar_num`) for scalar arguments; these are unions accepting both Python
  scalars and 0-d JAX arrays.
- Import shared types from `ptyrodactyl.types`, not by re-defining them.
- Import typing constructs (`Optional`, `Union`, `Tuple`, `List`, `Dict`, `TypeAlias`) from
  `beartype.typing`, not the stdlib `typing` module.

### Custom Types and PyTrees

**All types live in `ptyrodactyl.types` — no exceptions.** Every structured data type — every
`eqx.Module` PyTree, every container, every type alias, **and every `create_*` constructor
that builds one** — is defined under `src/ptyrodactyl/types/` and **nowhere else**. Every
other subpackage (`simul_multislice`, `simul_born`, `recon`, `ucell`, `inout`, `plots`,
`tools`, `audit`, `harness`) **imports** its types from `ptyrodactyl.types`; it must **not**
define its own PyTree, container, or `create_*` factory. Why: a single import surface, one
PyTree flatten/unflatten registration per type, one home for the validation (`create_*`)
contract, and no duplicate carriers drifting across modules. A result/parameter container that
"feels local" to a solver or producer (e.g. a `ReconResult`) is **still a type**: it goes in
`ptyrodactyl.types`, not beside the function that returns it.

Structured data types are **Equinox modules** (`eqx.Module`): immutable JAX PyTrees that flow
through `jit`/`grad`/`vmap`. Static, non-array metadata fields are declared with
`eqx.field(static=True)` so they are excluded from the differentiable leaves.

```python
import equinox as eqx
from jaxtyping import Array, Complex, Float


class ReconResult(eqx.Module):
    """JAX-compatible reconstruction result PyTree.

    :see: :class:`~.test_recon_types.TestReconResult`
    ...
    """

    params: Complex[Array, "H W"]
    loss: Float[Array, ""]
    iterations: int = eqx.field(static=True)
    solver_status: str = eqx.field(static=True)
```

### Validation Pattern for Factory Functions

Custom types are constructed through `create_*` factory functions that validate inputs. These
factories live in `ptyrodactyl.types` **next to the type they build** (never in the consuming
subpackage). Use a two-tier approach:

- **Static shape/structure checks** that can be resolved at trace time use plain Python
  `raise ValueError`.
- **Data-dependent (traced) checks** use `equinox.error_if`, which raises at runtime without
  breaking `jit`. **Never NaN-poison** invalid inputs via `lax.cond` — that silently corrupts
  downstream gradients; raise instead.

### Documentation Standards

Docstrings follow the **NumPy / numpydoc convention** (enforced by Ruff's `pydocstyle` rules
and a source-only `pydoclint` pass). Coverage is checked by `interrogate` (`fail-under = 90`).
Do **not** use ad-hoc section headers — stick to the numpydoc sections.

#### Module Docstrings

Each module starts with a one-line summary, an `Extended Summary`, a `Routine Listings`
section cross-referencing every public object, and a `Notes` section where relevant. Use the
correct Sphinx role in `Routine Listings`: `:func:` for functions, `:class:` for
classes/PyTrees, `:obj:` for type aliases and constants, and `:mod:` for submodules.

**Every public object is listed in three places, and all three must agree:**

1. In its own **module**, at the **top** in the docstring's `Routine Listings`, **and**
2. at the **bottom** in that module's `__all__`, **and**
3. in the **subpackage `__init__.py`** — repeated in `__init__.py`'s own `Routine Listings`
   *and* `__all__`.

A symbol missing from any of the three is a defect. When you add, rename, or remove a public
function, update **all three** in the same change, and keep the one-line summary
**verbatim-identical** across both `Routine Listings`.

**Export once, from the module that owns it — no compatibility re-exports.** Each public symbol
has exactly **one** canonical export path: the module that *defines* it, surfaced through *its
own* subpackage's `__init__.py`. Do **not** add a second export of the same symbol from any
other module or subpackage — not to preserve an old import location, not "for convenience," not
as a forwarding alias. When a symbol **moves or is renamed, it moves**: update every import
site and **delete** the old one in the *same* change — no shim, no alias, no
`DeprecationWarning`. The only migration record is a `CHANGELOG.md` note.

#### Function and Class Docstrings

- Open with a single imperative summary line.
- Add a `:see:` Sphinx cross-reference linking the object to its test class (e.g.
  `:see: :class:`~.test_unitcell.TestReciprocalLattice``). The test class carries the matching
  **back-reference**, so the link is **bidirectional** in the rendered docs.
- `Parameters` and `Returns` repeat the type and describe each item. Name the return values
  (numpydoc `name : type` form) so `pydoclint` is satisfied — and since functions **return a
  type-annotated variable rather than a bare expression**, the `Returns` name **must be that
  variable's name**.
- **Mark static (non-traced) parameters** — anything passed through
  `jax.jit(static_argnames=...)`, a Python `int`/`str`/`bool` flag driving shape/control flow,
  or an `eqx.field(static=True)` value — because changing it forces re-tracing.
- Use `Notes` (often a numbered list) for the algorithm/flow, `See Also` for related functions,
  `Attributes` for `eqx.Module` fields, `Raises` where relevant. Use a raw string (`r"""`) when
  the docstring contains LaTeX/backslashes.

### Code Style

Style is enforced by Ruff (`line-length = 79`, double quotes). Key conventions:

- **Variable Names**: descriptive `snake_case`; long names over abbreviations. Scientific
  single-letter symbols (`G`, `σ_e`, `ε`) are permitted where they mirror the physics.
- **No inline comments for explanation**: explanations belong in docstrings. Comments are
  reserved for non-obvious rationale (the *why*, not the *what*).
- **Pure functions**: no side effects; return new data.
- **Imports**: sorted by isort; `jax` and `jaxtyping` are treated as known third-party.
  Imports inside functions are used only to guard optional dependencies or platform branches.

## Testing

The test suite uses `pytest` with `chex`, `absl.testing.parameterized`, `hypothesis`, and
`pytest-xdist`. Runtime jaxtyping/beartype checking is active during tests via the
`--jaxtyping-packages=ptyrodactyl,beartype.beartype` pytest flag, so shape/dtype bugs surface
as test failures.

### Test Layout

Tests mirror the source layout under `tests/test_ptyrodactyl/`, with shared helpers in
`tests/_factories.py`, `tests/_assertions.py`, `tests/_types.py`. Test files are named
`test_<module>.py`; test classes `Test*` (typically `chex.TestCase`); test functions `test_*`.
Reuse the shared helpers instead of hand-rolling fixture data.

### Writing Tests

- Prefer `chex` assertions over bare `assert` for arrays: `chex.assert_shape`,
  `chex.assert_type`, `chex.assert_tree_all_finite`, `chex.assert_trees_all_close`.
- Use `absl.testing.parameterized` for table-driven cases, and `chex` variants
  (`@chex.variants`) to exercise functions under JIT and eager.
- Use `hypothesis` for property-based tests of numerical / physics invariants.
- Test both correctness and JAX compatibility (`jit`/`grad`/`vmap` where relevant) — in
  particular, **grad must flow** through every differentiable function.

### Test Code Conventions

Tests are first-class source: `tests/**/*.py` is in **both** the Ruff and `ty` scope and runs
under live jaxtyping/beartype checking, so the same style discipline as `src/` applies.

- **Type-hint test bodies and helpers exactly as in `src/`.**
- **Document *what* and *how* on every test, class, and module (numpydoc).** A test's docstring
  is its specification: open with an imperative summary, then an `Extended Summary` stating
  **what** is verified (with units and tolerances), and a `Notes` section describing **how**.
  Test docstrings are published to Read the Docs as a Testing / Validation reference.
- **Mirror the source layout, and make the `:see:` cross-reference bidirectional** (source →
  test *and* test → source).
- **No `__all__` or `Routine Listings` in test modules.**
- **Private helpers are `_`-prefixed and local; reused fixtures go in the shared helpers.**

## Tutorial Notebooks

Tutorials live in `tutorials/` as Jupyter notebooks paired with Jupytext percent scripts
(`.ipynb` plus `.py`). **Explanation lives in markdown cells, not code comments** — narrative,
motivation, and the physics belong in `# %% [markdown]` blocks. After editing a paired
notebook, run `uv run jupytext --sync tutorials/<notebook>.ipynb` and commit the synced `.py`
and output-stripped `.ipynb` together.

## Pull Request Process

### Before Submitting

```bash
ruff check src/ tests/
ruff format src/ tests/
pydoclint src/
ty check
pre-commit run --all-files
pytest
cd docs/ && uv run make html
```

### PR Guidelines

1. **Branch Naming:** descriptive, e.g. `feature/born-tem-terminal` or
   `fix/complex-optimizer-conjugation`.
2. **Commit Messages:** a clear imperative subject line followed by a bulleted body.
3. **PR Description:** include what the PR does, why it's needed, how to test it, and any
   breaking changes.

### Review Process

All PRs require passing CI tests, a code review approval, documentation updates where
applicable, and no merge conflicts. **Differentiability is an acceptance criterion**: a change
that breaks `jax.grad` through a touched seam has failed even if every test passes.

## API Evolution (zero-legacy)

The codebase carries **no compatibility layer**. When an API changes:

- **No shims, aliases, re-exports, or `DeprecationWarning`s** are kept alive for old import
  paths or signatures.
- Update every call site and **delete** the old path in the *same* change; two implementations
  or import paths never ship together.
- The **only** migration record is a `CHANGELOG.md` note documenting the rename/removal and the
  new path.
- Prefer getting the API right over preserving a wrong one — a clean break with a changelog
  entry beats a forwarding alias that quietly rots.

## Getting Help

- **Questions:** Open a discussion or issue
- **Documentation:** Check the [docs](https://github.com/debangshu-mukherjee/ptyrodactyl)

Thank you for contributing to ptyrodactyl and advancing differentiable electron microscopy!
