"""Validate every annotation before the test suite runs.

Extended Summary
----------------
This module imports each module in ``ptyrodactyl`` and ``tests`` while the
jaxtyping import hook is active. The hook decorates every function and class
with ``@jaxtyped(typechecker=beartype)``. Decoration evaluates each annotation.
An invalid annotation therefore raises at import time.

The gate reports three defect classes without running one test:

- a malformed jaxtyping specification, such as a wrong dtype name or a wrong
  shape string;
- a name that an annotation uses but the module does not import;
- a hint that beartype cannot use, such as a bare ``NDArray`` that carries an
  unbound ``_ScalarT`` type variable.

The gate does not detect a wrong dtype at runtime. Only the test suite detects
that defect, because it requires real array values.

The gate skips ``conftest`` modules. Pytest drives the fixtures in those
modules, and the jaxtyping wrapper hides ``inspect.isgeneratorfunction``. A
hooked yield fixture therefore loses its teardown.

Run the gate directly::

    uv run --frozen python tests/_preflight_types.py

The process exits with status 1 when it finds one defect or more.

Routine Listings
----------------
:func:`iter_module_names`
    Yield every importable module name below one package.
:func:`check_annotations`
    Import every target module and collect annotation defects.
:func:`main`
    Run the gate and return a process exit status.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from pathlib import Path
from types import ModuleType

from beartype.typing import Iterator, List, Tuple
from jaxtyping import install_import_hook

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

TARGET_PACKAGES: Tuple[str, ...] = ("ptyrodactyl", "tests")
TYPECHECKER: str = "beartype.beartype"


def iter_module_names(package_name: str) -> Iterator[str]:
    """Yield every importable module name below one package.

    Parameters
    ----------
    package_name : str
        Import path of the package to walk.

    Yields
    ------
    module_name : str
        Dotted import path of one module inside the package.
    """
    package: ModuleType
    try:
        package = importlib.import_module(package_name)
    except Exception:  # noqa: BLE001 -- report the failure as a defect.
        yield package_name
        return
    yield package_name
    if not hasattr(package, "__path__"):
        return
    info: pkgutil.ModuleInfo
    for info in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        yield info.name


def check_annotations() -> Tuple[int, List[Tuple[str, str]]]:
    """Import every target module and collect annotation defects.

    Returns
    -------
    checked : int
        Count of modules that the gate imported.
    failures : List[Tuple[str, str]]
        One ``(module_name, message)`` pair for each defect.
    """
    failures: List[Tuple[str, str]] = []
    checked: int = 0
    package_name: str
    module_name: str
    with install_import_hook(list(TARGET_PACKAGES), TYPECHECKER):
        for package_name in TARGET_PACKAGES:
            for module_name in iter_module_names(package_name):
                if module_name.endswith("conftest"):
                    continue
                checked += 1
                try:
                    importlib.import_module(module_name)
                except Exception:  # noqa: BLE001 -- collect every defect.
                    detail: str = traceback.format_exc().strip()
                    failures.append((module_name, detail.splitlines()[-1]))
    result: Tuple[int, List[Tuple[str, str]]] = (checked, failures)
    return result


def main() -> int:
    """Run the gate and return a process exit status.

    Returns
    -------
    status : int
        Zero when every annotation is valid, otherwise one.
    """
    checked: int
    failures: List[Tuple[str, str]]
    checked, failures = check_annotations()
    print(f"modules imported: {checked}")
    print(f"annotation defects: {len(failures)}")
    module_name: str
    message: str
    for module_name, message in failures:
        print("-" * 70)
        print(module_name)
        print(message)
    status: int = 1 if failures else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
