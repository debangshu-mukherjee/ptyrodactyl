"""Package export and Routine Listing structure tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import ptyrodactyl

_SWEPT_PACKAGES = (
    "ptyrodactyl.inout",
    "ptyrodactyl.ucell",
    "ptyrodactyl.plots",
    "ptyrodactyl.multislice",
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
        if isinstance(node, ast.AnnAssign) and _target_name(node.target) == "__all__":
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


def _routine_listings(tree: ast.Module, path: Path) -> dict[str, str]:
    docstring = ast.get_docstring(tree) or ""
    lines = docstring.splitlines()
    try:
        start = lines.index("Routine Listings") + 1
    except ValueError as exc:
        raise AssertionError(f"{path}: missing Routine Listings block") from exc

    listings: dict[str, str] = {}
    current: str | None = None
    summary_parts: list[str] = []
    for raw_line in lines[start:]:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "" or set(stripped) == {"-"}:
            continue
        if line.startswith(":func:`") and "`" in line[len(":func:`") :]:
            if current is not None:
                listings[current] = " ".join(summary_parts)
            current = line.split("`", 2)[1]
            summary_parts = []
            continue
        if current is not None and line.startswith("    "):
            summary_parts.append(stripped)
            continue
        break
    if current is not None:
        listings[current] = " ".join(summary_parts)
    return listings


def _leaf_modules(package: str) -> list[Path]:
    package_path = _module_path(package).parent
    return sorted(
        path
        for path in package_path.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def test_public_exports_have_matching_routine_listing_triads() -> None:
    for package in _SWEPT_PACKAGES:
        init_path = _module_path(package)
        init_tree = _parse(init_path)
        init_all = _literal_all(init_tree, init_path)
        init_listings = _routine_listings(init_tree, init_path)

        leaf_all: dict[str, list[Path]] = {}
        leaf_listings: dict[Path, dict[str, str]] = {}
        for leaf_path in _leaf_modules(package):
            leaf_tree = _parse(leaf_path)
            leaf_listings[leaf_path] = _routine_listings(leaf_tree, leaf_path)
            for symbol in _literal_all(leaf_tree, leaf_path):
                leaf_all.setdefault(symbol, []).append(leaf_path)

        for symbol in init_all:
            assert symbol in init_listings, (
                f"{package}.__init__ Routine Listings missing {symbol}"
            )
            assert symbol in leaf_all, f"{package}: no leaf __all__ exports {symbol}"
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


def test_public_symbols_are_exported_from_one_swept_subpackage() -> None:
    owners: dict[str, str] = {}
    for package in _SWEPT_PACKAGES:
        package_all = _literal_all(_parse(_module_path(package)), _module_path(package))
        for symbol in package_all:
            previous = owners.setdefault(symbol, package)
            assert previous == package, (
                f"{symbol!r} is exported by both {previous} and {package}"
            )


def test_source_uses_no_star_imports() -> None:
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
