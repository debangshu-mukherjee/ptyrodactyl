"""Package export and Routine Listing structure tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import ptyrodactyl

_PUBLIC_PACKAGES = (
    "ptyrodactyl.bloch",
    "ptyrodactyl.born",
    "ptyrodactyl.inout",
    "ptyrodactyl.invert",
    "ptyrodactyl.jacobian",
    "ptyrodactyl.multislice",
    "ptyrodactyl.plots",
    "ptyrodactyl.tools",
    "ptyrodactyl.types",
    "ptyrodactyl.ucell",
    "ptyrodactyl.workflows",
)
_ROUTINE_KIND_ORDER = {"class": 0, "func": 1, "obj": 2}
_TEST_REFERENCE = re.compile(
    r":see:\s+:(?P<role>class|func|meth|mod):`"
    r"(?P<target>(?:tests\.test_ptyrodactyl\.|~\.test_)[^`]+)`"
)
_POTENTIAL_TYPE_EXPORTS = (
    "KirklandParameters",
    "LobatoParameters",
    "Potential3D",
    "create_kirkland_parameters",
    "create_lobato_parameters",
    "create_potential_3d",
)


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    if module.__file__ is None:
        raise AssertionError(f"{module_name} has no source file")
    return Path(module.__file__)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_all(tree: ast.Module, path: Path) -> list[str]:
    for node in tree.body:
        value: ast.expr | None = None
        if (
            isinstance(node, ast.AnnAssign)
            and _target_name(node.target) == "__all__"
        ):
            value = node.value
        elif isinstance(node, ast.Assign) and any(
            _target_name(target) == "__all__" for target in node.targets
        ):
            value = node.value
        if value is not None:
            if not isinstance(value, ast.List):
                raise AssertionError(f"{path}: __all__ must be a literal list")
            symbols: list[str] = []
            for element in value.elts:
                if not isinstance(element, ast.Constant) or not isinstance(
                    element.value,
                    str,
                ):
                    raise AssertionError(
                        f"{path}: __all__ entries must be literal strings",
                    )
                symbols.append(element.value)
            return symbols
    return []


def _target_name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _routine_listing_entries(
    tree: ast.Module,
    path: Path,
) -> list[tuple[str, str, str]]:
    docstring = ast.get_docstring(tree) or ""
    lines = docstring.splitlines()
    try:
        start = lines.index("Routine Listings") + 1
    except ValueError as exc:
        raise AssertionError(
            f"{path}: missing Routine Listings block"
        ) from exc

    entries: list[tuple[str, str, str]] = []
    current_kind: str | None = None
    current: str | None = None
    summary_parts: list[str] = []
    for raw_line in lines[start:]:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "" or set(stripped) == {"-"}:
            continue
        if line.startswith(":data:`"):
            raise AssertionError(f"{path}: use :obj: for public objects")
        if line.startswith((":class:`", ":func:`", ":obj:`")) and "`" in line:
            if current is not None:
                assert current_kind is not None
                entries.append(
                    (current_kind, current, " ".join(summary_parts))
                )
            current_kind = line.split(":", 2)[1]
            current = line.split("`", 2)[1]
            summary_parts = []
            continue
        if current is not None and line.startswith("    "):
            summary_parts.append(stripped)
            continue
        break
    if current is not None:
        assert current_kind is not None
        entries.append((current_kind, current, " ".join(summary_parts)))
    return entries


def _routine_listings(tree: ast.Module, path: Path) -> dict[str, str]:
    return {
        symbol: summary
        for _, symbol, summary in _routine_listing_entries(tree, path)
    }


def _definition_summaries(tree: ast.Module) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if docstring:
            summaries[node.name] = docstring.splitlines()[0]
    return summaries


class _ReturnBindingVisitor(ast.NodeVisitor):
    """Collect returns and annotations owned by one function scope."""

    def __init__(self, function: ast.AST) -> None:
        self.function = function
        self.returns: list[ast.Return] = []
        self.annotated_assignments: dict[str, list[int]] = {}
        self.annotated_values: dict[str, list[tuple[int, ast.expr]]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.function:
            for statement in node.body:
                self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.function:
            for statement in node.body:
                self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Do not descend into a nested class scope."""

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Do not descend into a nested lambda scope."""

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.annotated_assignments.setdefault(node.target.id, []).append(
                node.lineno
            )
            self.annotated_values.setdefault(node.target.id, []).append(
                (node.lineno, node.value)
            )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node)


def _returns_doc_names(docstring: str) -> list[str]:
    """Extract names from one NumPy-style ``Returns`` section."""
    lines = docstring.splitlines()
    try:
        index = (
            next(
                index
                for index, line in enumerate(lines)
                if line.strip() == "Returns"
            )
            + 2
        )
    except StopIteration:
        return []

    names: list[str] = []
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and lines[index].strip()
            and lines[index + 1].strip()
            and set(lines[index + 1].strip()) == {"-"}
        ):
            break
        line = lines[index]
        if line.strip() and not line.startswith((" ", ":")):
            names.append(line.split(" : ", maxsplit=1)[0].strip())
        index += 1
    return names


def _export_definition(
    tree: ast.Module,
    symbol: str,
    path: Path,
) -> tuple[ast.AST, str]:
    """Return one public definition node and its documentation text."""
    for index, node in enumerate(tree.body):
        if isinstance(
            node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)
        ):
            if node.name == symbol:
                return node, ast.get_docstring(node) or ""

        targets: tuple[ast.expr, ...] = ()
        if isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        elif isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        if not any(_target_name(target) == symbol for target in targets):
            continue
        if index + 1 < len(tree.body):
            adjacent = tree.body[index + 1]
            if (
                isinstance(adjacent, ast.Expr)
                and isinstance(adjacent.value, ast.Constant)
                and isinstance(adjacent.value.value, str)
            ):
                return node, adjacent.value.value
        return node, ""
    raise AssertionError(f"{path}: cannot locate public definition {symbol!r}")


def _public_reference_role(node: ast.AST) -> str:
    """Return the Sphinx role for one public definition node."""
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return "func"
    return "obj"


def _resolve_test_reference(
    role: str,
    target: str,
    expected_module: str,
) -> object:
    """Resolve one canonicalized test reference below its mirror module."""
    module = importlib.import_module(expected_module)
    short_module = f"~.{expected_module.rsplit('.', maxsplit=1)[1]}"
    if target == short_module or target.startswith(f"{short_module}."):
        target = f"{expected_module}{target.removeprefix(short_module)}"
    elif target.startswith("~."):
        target = f"{expected_module}.{target.removeprefix('~.')}"
    if target == expected_module:
        assert role == "mod", f"{target}: module target must use :mod:"
        return module

    prefix = f"{expected_module}."
    assert target.startswith(prefix), (
        f"{target}: test target must resolve below {expected_module}"
    )
    resolved: object = module
    for part in target.removeprefix(prefix).split("."):
        resolved = getattr(resolved, part)
    if role == "class":
        assert isinstance(resolved, type), f"{target}: :class: is not a class"
    elif role in {"func", "meth"}:
        assert callable(resolved), f"{target}: callable role is not callable"
    else:
        raise AssertionError(f"{target}: nested target cannot use :{role}:")
    return resolved


def _forward_test_references(docstring: str) -> list[tuple[str, str]]:
    """Parse test references after joining explicit source continuations."""
    normalized = docstring.replace("\\\n", "")
    return [
        (match.group("role"), match.group("target"))
        for match in _TEST_REFERENCE.finditer(normalized)
    ]


def _leaf_modules(package: str) -> list[Path]:
    package_path = _module_path(package).parent
    return sorted(
        path
        for path in package_path.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def test_public_exports_have_synchronized_routine_listings() -> None:
    """Prove each public export has one leaf and matching summaries.

    Parse literal export lists and Routine Listings from package and leaf
    modules, then compare ownership and summary text exactly.
    """
    for package in _PUBLIC_PACKAGES:
        package_module = importlib.import_module(package)
        init_path = _module_path(package)
        init_tree = _parse(init_path)
        init_all = _literal_all(init_tree, init_path)
        init_listings = _routine_listings(init_tree, init_path)
        init_entries = _routine_listing_entries(init_tree, init_path)

        expected_init_order = sorted(
            init_entries,
            key=lambda entry: (
                _ROUTINE_KIND_ORDER[entry[0]],
                entry[1].casefold(),
            ),
        )
        assert init_entries == expected_init_order, (
            f"{init_path}: Routine Listings must group classes, functions, "
            "and objects, then alphabetize each group"
        )
        assert set(init_all) == set(init_listings), (
            f"{package}: package __all__ and Routine Listings differ"
        )

        leaf_all: dict[str, list[Path]] = {}
        leaf_listings: dict[Path, dict[str, str]] = {}
        for leaf_path in _leaf_modules(package):
            leaf_tree = _parse(leaf_path)
            leaf_exports = _literal_all(leaf_tree, leaf_path)
            leaf_entries = _routine_listing_entries(leaf_tree, leaf_path)
            leaf_listings[leaf_path] = {
                symbol: summary for _, symbol, summary in leaf_entries
            }
            expected_leaf_order = sorted(
                leaf_entries,
                key=lambda entry: (
                    _ROUTINE_KIND_ORDER[entry[0]],
                    entry[1].casefold(),
                ),
            )
            assert leaf_entries == expected_leaf_order, (
                f"{leaf_path}: Routine Listings are not grouped and sorted"
            )
            assert set(leaf_exports) == set(leaf_listings[leaf_path]), (
                f"{leaf_path}: __all__ and Routine Listings differ"
            )
            definitions = _definition_summaries(leaf_tree)
            for symbol, summary in leaf_listings[leaf_path].items():
                assert not symbol.startswith("_"), (
                    f"{leaf_path}: private symbol {symbol!r} is public"
                )
                if symbol in definitions:
                    assert summary == definitions[symbol], (
                        f"{leaf_path}: {symbol!r} summary differs from its "
                        "docstring summary"
                    )
            for symbol in leaf_exports:
                leaf_all.setdefault(symbol, []).append(leaf_path)

        assert set(init_all) == set(leaf_all), (
            f"{package}: package and aggregate leaf exports differ"
        )
        for symbol in init_all:
            assert symbol in init_listings, (
                f"{package}.__init__ Routine Listings missing {symbol}"
            )
            assert symbol in leaf_all, (
                f"{package}: no leaf __all__ exports {symbol}"
            )
            assert len(leaf_all[symbol]) == 1, (
                f"{package}: {symbol} exported from multiple leaf __all__s: "
                f"{leaf_all[symbol]}"
            )
            leaf_path = leaf_all[symbol][0]
            leaf_summary = leaf_listings[leaf_path].get(symbol)
            assert leaf_summary is not None, (
                f"{leaf_path}: Routine Listings missing {symbol}"
            )
            assert init_listings[symbol] == leaf_summary, (
                f"{package}: summary mismatch for {symbol!r}: "
                f"{init_listings[symbol]!r} != {leaf_summary!r}"
            )
            leaf_module = importlib.import_module(
                f"{package}.{leaf_path.stem}"
            )
            assert getattr(package_module, symbol) is getattr(
                leaf_module,
                symbol,
            )


def test_public_symbols_are_exported_from_one_subpackage() -> None:
    """Prove public subpackages do not claim the same public symbol.

    Parse each package's literal export list and require one owner per name.
    """
    owners: dict[str, str] = {}
    for package in _PUBLIC_PACKAGES:
        package_all = _literal_all(
            _parse(_module_path(package)), _module_path(package)
        )
        for symbol in package_all:
            previous = owners.setdefault(symbol, package)
            assert previous == package, (
                f"{symbol!r} is exported by both {previous} and {package}"
            )


def test_public_exports_have_bidirectional_test_references() -> None:
    """Require one resolvable test target and its canonical back-reference.

    Parse every public leaf export, resolve its mirrored test target, and
    require that target to name the canonical public object.
    """
    for package in _PUBLIC_PACKAGES:
        package_name = package.rsplit(".", maxsplit=1)[1]
        for leaf_path in _leaf_modules(package):
            leaf_tree = _parse(leaf_path)
            test_module = (
                "tests.test_ptyrodactyl."
                f"test_{package_name}.test_{leaf_path.stem}"
            )
            for symbol in _literal_all(leaf_tree, leaf_path):
                definition, docstring = _export_definition(
                    leaf_tree,
                    symbol,
                    leaf_path,
                )
                references = _forward_test_references(docstring)
                assert len(references) == 1, (
                    f"{leaf_path}: {symbol} must have exactly one canonical "
                    f"test :see: reference, found {references}"
                )
                role, target = references[0]
                test_target = _resolve_test_reference(
                    role,
                    target,
                    test_module,
                )
                target_docstring = inspect.getdoc(test_target) or ""
                public_role = _public_reference_role(definition)
                reciprocal = f":see: :{public_role}:`{package}.{symbol}`"
                assert reciprocal in target_docstring, (
                    f"{target}: missing reciprocal reference {reciprocal}"
                )


def test_root_public_objects_have_bidirectional_test_references() -> None:
    """Apply the public test-reference contract to top-level objects."""
    package = "ptyrodactyl"
    package_path = _module_path(package)
    package_tree = _parse(package_path)
    test_module = "tests.test_ptyrodactyl.test___init__"
    for symbol in _literal_all(package_tree, package_path):
        try:
            definition, docstring = _export_definition(
                package_tree,
                symbol,
                package_path,
            )
        except AssertionError:
            continue
        references = _forward_test_references(docstring)
        assert len(references) == 1, (
            f"{package_path}: {symbol} must have exactly one canonical "
            f"test :see: reference, found {references}"
        )
        role, target = references[0]
        test_target = _resolve_test_reference(role, target, test_module)
        target_docstring = inspect.getdoc(test_target) or ""
        public_role = _public_reference_role(definition)
        reciprocal = f":see: :{public_role}:`{package}.{symbol}`"
        assert reciprocal in target_docstring, (
            f"{target}: missing reciprocal reference {reciprocal}"
        )


def test_public_submodules_are_listed_once_in_alphabetical_order() -> None:
    """Require each package summary to list every public leaf once."""
    for package in _PUBLIC_PACKAGES:
        init_path = _module_path(package)
        docstring = ast.get_docstring(_parse(init_path)) or ""
        extended_summary = docstring.partition("Routine Listings")[0]
        listed = re.findall(
            r"^- :mod:`([^`]+)`",
            extended_summary,
            flags=re.MULTILINE,
        )
        expected = [path.stem for path in _leaf_modules(package)]
        assert listed == expected, (
            f"{init_path}: submodules {listed} do not match {expected}"
        )


def test_potential_types_have_one_matching_public_owner() -> None:
    """Potential carriers and factories have one canonical type export."""
    package_name = "ptyrodactyl.types"
    package = importlib.import_module(package_name)
    init_path = _module_path(package_name)
    init_tree = _parse(init_path)
    init_all = _literal_all(init_tree, init_path)
    init_listings = _routine_listings(init_tree, init_path)

    for symbol in _POTENTIAL_TYPE_EXPORTS:
        owners: list[Path] = []
        for leaf_path in _leaf_modules(package_name):
            if symbol in _literal_all(_parse(leaf_path), leaf_path):
                owners.append(leaf_path)

        assert symbol in init_all
        assert symbol in init_listings
        assert len(owners) == 1, f"{symbol!r} leaf owners: {owners}"
        leaf_path = owners[0]
        leaf_module = importlib.import_module(
            f"{package_name}.{leaf_path.stem}"
        )
        leaf_listings = _routine_listings(_parse(leaf_path), leaf_path)
        assert leaf_listings.get(symbol) == init_listings[symbol]
        assert getattr(package, symbol) is getattr(leaf_module, symbol)


def test_public_equinox_carriers_use_owning_factory_modules() -> None:
    """Reject production carrier construction outside its owning type module.

    Derive public Equinox carrier names from the literal type exports, then
    inspect every production call without maintaining a second carrier list.
    """
    type_package = "ptyrodactyl.types"
    carrier_owners: dict[str, Path] = {}
    for leaf_path in _leaf_modules(type_package):
        leaf_tree = _parse(leaf_path)
        exports = set(_literal_all(leaf_tree, leaf_path))
        for definition in leaf_tree.body:
            if not isinstance(definition, ast.ClassDef):
                continue
            bases = {ast.unparse(base) for base in definition.bases}
            if definition.name in exports and bases.intersection(
                {"eqx.Module", "equinox.Module"}
            ):
                carrier_owners[definition.name] = leaf_path

    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        for call in (
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            call_name: str | None = None
            if isinstance(call.func, ast.Name):
                call_name = call.func.id
            elif isinstance(call.func, ast.Attribute):
                call_name = call.func.attr
            owner = carrier_owners.get(call_name or "")
            if owner is not None and path != owner:
                offenders.append(
                    f"{path}:{call.lineno} constructs {call_name}; "
                    f"owner is {owner}"
                )

    assert offenders == []


def test_source_uses_no_star_imports() -> None:
    """Prove production modules contain no star imports through AST walks."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_jax_jit_wraps_runtime_typechecking() -> None:
    """JIT transformations wrap runtime type checking on Python functions."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = [ast.unparse(item) for item in node.decorator_list]
            jit_indices = [
                index
                for index, decorator in enumerate(decorators)
                if decorator.startswith(("jax.jit", "partial(jax.jit"))
            ]
            type_indices = [
                index
                for index, decorator in enumerate(decorators)
                if decorator.startswith("jaxtyped(")
            ]
            if (
                jit_indices
                and type_indices
                and min(jit_indices) > min(type_indices)
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_public_function_signatures_are_fully_annotated() -> None:
    """Require annotations on every public module-level function."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            arguments = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            missing = [arg.arg for arg in arguments if arg.annotation is None]
            if (
                node.args.vararg is not None
                and node.args.vararg.annotation is None
            ):
                missing.append(f"*{node.args.vararg.arg}")
            if (
                node.args.kwarg is not None
                and node.args.kwarg.annotation is None
            ):
                missing.append(f"**{node.args.kwarg.arg}")
            if node.returns is None:
                missing.append("return")
            if missing:
                offenders.append(
                    f"{path}:{node.lineno} {node.name}: {', '.join(missing)}"
                )
    assert offenders == []


def test_production_returns_use_annotated_local_bindings() -> None:
    """Require every valued return to use an annotated local variable."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        functions = (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        )
        for function in functions:
            visitor = _ReturnBindingVisitor(function)
            visitor.visit(function)
            for returned in visitor.returns:
                if returned.value is None:
                    continue
                if not isinstance(returned.value, ast.Name):
                    offenders.append(
                        f"{path}:{returned.lineno} {function.name}: "
                        "return expression is not a local name"
                    )
                    continue
                assignment_lines = visitor.annotated_assignments.get(
                    returned.value.id,
                    [],
                )
                if not any(
                    line_number <= returned.lineno
                    for line_number in assignment_lines
                ):
                    offenders.append(
                        f"{path}:{returned.lineno} {function.name}: "
                        f"{returned.value.id} has no preceding annotated "
                        "assignment"
                    )
    assert offenders == []


def test_exported_returns_document_bound_local_names() -> None:
    """Match exported ``Returns`` names to annotated returned values."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        exports = set(_literal_all(tree, path))
        functions = (
            node
            for node in tree.body
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name in exports
        )
        for function in functions:
            visitor = _ReturnBindingVisitor(function)
            visitor.visit(function)
            expected: list[str] = []
            for returned in visitor.returns:
                if not isinstance(returned.value, ast.Name):
                    continue
                candidates = [
                    (line_number, value)
                    for line_number, value in visitor.annotated_values.get(
                        returned.value.id,
                        [],
                    )
                    if line_number <= returned.lineno
                ]
                if not candidates:
                    continue
                assigned_value = max(candidates, key=lambda item: item[0])[1]
                if isinstance(assigned_value, ast.Tuple) and all(
                    isinstance(element, ast.Name)
                    for element in assigned_value.elts
                ):
                    names = [
                        element.id
                        for element in assigned_value.elts
                        if isinstance(element, ast.Name)
                    ]
                else:
                    names = [returned.value.id]
                expected.extend(name for name in names if name not in expected)
            documented = _returns_doc_names(ast.get_docstring(function) or "")
            if documented != expected:
                offenders.append(
                    f"{path}:{function.lineno} {function.name}: "
                    f"documented {documented}, expected {expected}"
                )
    assert offenders == []


def test_source_uses_canonical_cross_subpackage_imports() -> None:
    """Reject renamed project imports and cross-package leaf reaches."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        owner = relative.parts[0] if len(relative.parts) > 1 else None
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("ptyrodactyl") and alias.asname:
                        offenders.append(
                            f"{path}:{node.lineno} renamed import"
                        )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("ptyrodactyl.")
            ):
                parts = node.module.split(".")
                imported_owner = parts[1]
                if (
                    owner is not None
                    and imported_owner != owner
                    and len(parts) > 2
                ):
                    offenders.append(
                        f"{path}:{node.lineno} deep import {node.module}"
                    )
                if any(alias.asname for alias in node.names):
                    offenders.append(f"{path}:{node.lineno} renamed import")
    assert offenders == []


def test_source_uses_no_uninitialized_jax_arrays() -> None:
    """Reject jnp.empty because production kernels require defined values."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "jnp"
                and node.func.attr == "empty"
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_numpy_annotations_bind_dtype_and_shape() -> None:
    """Reject bare NumPy ndarray annotations at host boundaries."""
    source_root = Path(ptyrodactyl.__file__).parent
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        annotations: list[ast.expr] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                annotations.append(node.annotation)
            elif isinstance(node, ast.arg) and node.annotation is not None:
                annotations.append(node.annotation)
            elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.returns is not None:
                    annotations.append(node.returns)
        for annotation in annotations:
            rendered = ast.unparse(annotation)
            if (
                rendered in {"NDArray", "np.ndarray", "numpy.ndarray"}
                or rendered.startswith("NDArray[")
                or "np.ndarray" in rendered
                or "numpy.ndarray" in rendered
            ):
                offenders.append(f"{path}:{annotation.lineno} {rendered}")
    assert offenders == []


def test_tests_mirror_public_source_modules() -> None:
    """Require one mirrored test module for every public source module."""
    source_root = Path(ptyrodactyl.__file__).parent
    test_root = Path(__file__).parent
    source_packages = {
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    test_packages = {
        path.name.removeprefix("test_")
        for path in test_root.glob("test_*")
        if path.is_dir()
    }
    assert test_packages == source_packages

    missing: list[str] = []
    for package_name in sorted(source_packages):
        source_package = source_root / package_name
        test_package = test_root / f"test_{package_name}"
        for source_path in sorted(source_package.glob("*.py")):
            is_private = source_path.name.startswith("_")
            if source_path.name == "__init__.py" or is_private:
                continue
            test_path = test_package / f"test_{source_path.name}"
            if not test_path.is_file():
                missing.append(str(test_path))
    assert missing == []


def test_test_functions_have_nonempty_docstrings() -> None:
    """Require a compact specification on every pytest test function."""
    tests_root = Path(__file__).parents[1]
    offenders: list[str] = []
    for path in sorted(tests_root.rglob("test_*.py")):
        for node in ast.walk(_parse(path)):
            if not isinstance(
                node,
                (ast.AsyncFunctionDef, ast.FunctionDef),
            ):
                continue
            if not node.name.startswith("test_"):
                continue
            if not (ast.get_docstring(node) or "").strip():
                offenders.append(f"{path}:{node.lineno} {node.name}")
    assert offenders == []
