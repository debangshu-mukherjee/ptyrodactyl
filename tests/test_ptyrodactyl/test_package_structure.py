"""Package export and Routine Listing structure tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from importlib.util import find_spec
from pathlib import Path

from beartype.typing import Dict, Tuple

import ptyrodactyl

_PUBLIC_PACKAGES = (
    "ptyrodactyl.bloch",
    "ptyrodactyl.born",
    "ptyrodactyl.galerkin",
    "ptyrodactyl.inout",
    "ptyrodactyl.jacobian",
    "ptyrodactyl.multislice",
    "ptyrodactyl.plots",
    "ptyrodactyl.types",
    "ptyrodactyl.ucell",
    "ptyrodactyl.workflows",
)
_ROUTINE_KIND_ORDER = {"class": 0, "func": 1, "obj": 2}
_INTERNAL_TOOL_LEAVES: Tuple[str, ...] = (
    "canonical_digest",
    "censored_poisson_differential_interval",
    "entire_interval",
    "host_interval",
    "interval",
    "numeric",
    "physics",
    "poisson_interval",
)
_PRIVATE_FUNCTION_SECTION_ORDER = {
    "Extended Summary": 0,
    "Implementation Logic": 1,
    "Parameters": 2,
    "Returns": 3,
    "Yields": 3,
    "Raises": 4,
    "Notes": 5,
    "References": 6,
    "See Also": 7,
    "Examples": 8,
}
_TEST_REFERENCE = re.compile(
    r":see:\s+:(?P<role>class|func|meth|mod):`"
    r"(?P<target>(?:tests\.test_ptyrodactyl\.|~\.test_)[^`]+)`"
)
_POTENTIAL_TYPE_EXPORTS = (
    "KirklandParameters",
    "LobatoParameters",
    "LocalCellPotential3D",
    "Potential3D",
    "create_kirkland_parameters",
    "create_lobato_parameters",
    "create_local_cell_potential_3d",
    "create_potential_3d",
)
_CANONICAL_CARRIER_DTYPES: Tuple[Tuple[str, str, str, str], ...] = (
    ("potential_types.py", "Potential3D", "volume", "Float64"),
    (
        "local_cell_types.py",
        "LocalCellPotential3D",
        "cell_values",
        "Float64",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "exact_coefficient_real_lower_bounds",
        "Float64",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "exact_coefficient_real_upper_bounds",
        "Float64",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "exact_coefficient_imag_lower_bounds",
        "Float64",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "exact_coefficient_imag_upper_bounds",
        "Float64",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "finite_certificate",
        "Bool",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "direct_term_count",
        "Int64",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellCoefficientCertificate",
        "maximum_direct_terms",
        "Int64",
    ),
    (
        "born_potential_types.py",
        "GalerkinProductSupport",
        "state_indices",
        "Int64",
    ),
    (
        "born_potential_types.py",
        "GalerkinProductSupport",
        "interaction_indices",
        "Int64",
    ),
    (
        "born_potential_types.py",
        "GalerkinProductSupport",
        "absorber_indices",
        "Int64",
    ),
    (
        "born_potential_types.py",
        "GalerkinProductSupport",
        "work_indices",
        "Int64",
    ),
    (
        "acquisition_types.py",
        "GalerkinAcquisitionManifest",
        "preterminal_indices",
        "Int64",
    ),
    (
        "realization_types.py",
        "GalerkinPotentialRealization",
        "voltage_coefficients",
        "Complex128",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellPotentialRealization",
        "voltage_coefficients",
        "Complex128",
    ),
    (
        "local_cell_types.py",
        "GalerkinLocalCellPotentialRealization",
        "coefficient_error_bounds",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "interaction_coefficients",
        "Complex128",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "interaction_coupling",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "absorber_coefficients",
        "Complex128",
    ),
    (
        "realization_error_types.py",
        "GalerkinFixedLinearErrorLedger",
        "algebraic_free_diagonal",
        "Float64",
    ),
    (
        "acquisition_types.py",
        "GalerkinAcquisitionManifest",
        "carrier",
        "Float64",
    ),
    (
        "acquisition_types.py",
        "GalerkinAcquisitionManifest",
        "box_lengths",
        "Float64",
    ),
    (
        "acquisition_types.py",
        "GalerkinAcquisitionManifest",
        "wavenumber",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "exact_target_incident_full_offset_max",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "exact_target_outgoing_full_offset_max",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "exact_target_incident_shell_defect_bounds",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "exact_target_outgoing_shell_defect_bounds",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "exact_target_incident_projection_error_bounds",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "exact_target_outgoing_projection_error_bounds",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "accelerating_voltage_kv",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinTargetManifest",
        "cap_scale",
        "Float64",
    ),
    ("galerkin_types.py", "GalerkinSource", "incident_field", "Complex128"),
    (
        "galerkin_types.py",
        "GalerkinSource",
        "incident_source",
        "Complex128",
    ),
    (
        "galerkin_types.py",
        "GalerkinSource",
        "additional_source",
        "Complex128",
    ),
    ("galerkin_types.py", "GalerkinSource", "total_source", "Complex128"),
    (
        "galerkin_types.py",
        "GalerkinSource",
        "scattered_source",
        "Complex128",
    ),
    (
        "galerkin_types.py",
        "GalerkinPhysicalResidual",
        "residual",
        "Complex128",
    ),
    (
        "galerkin_types.py",
        "GalerkinPhysicalResidual",
        "residual_norm",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinStabilityResult",
        "lower_singular_bound",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinStabilityResult",
        "residual_upper_bound",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinStabilityResult",
        "state_error_upper_bound",
        "Float64",
    ),
    (
        "galerkin_types.py",
        "GalerkinStabilityResult",
        "state_budget",
        "Float64",
    ),
    ("born_types.py", "GalerkinOperator", "cap_scale", "Float64"),
    ("born_types.py", "GalerkinSolveResult", "residual_norm", "Float64"),
    (
        "born_types.py",
        "GalerkinSolveResult",
        "normal_residual_norm",
        "Float64",
    ),
    (
        "born_types.py",
        "GalerkinSolveResult",
        "recurrence_residual_norm",
        "Float64",
    ),
    ("born_types.py", "GalerkinSolveResult", "iterations", "Int32"),
    (
        "born_types.py",
        "GalerkinSolveResult",
        "operator_applications",
        "Int32",
    ),
    ("born_types.py", "GalerkinSolveResult", "status", "Int32"),
)


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    if module.__file__ is None:
        raise AssertionError(f"{module_name} has no source file")
    return Path(module.__file__)


def _parse(path: Path) -> ast.Module:
    return ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
        type_comments=True,
    )


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
) -> list[Tuple[str, str, str]]:
    docstring = ast.get_docstring(tree) or ""
    lines = docstring.splitlines()
    try:
        start = lines.index("Routine Listings") + 1
    except ValueError as exc:
        raise AssertionError(
            f"{path}: missing Routine Listings block"
        ) from exc

    entries: list[Tuple[str, str, str]] = []
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


def _routine_listings(tree: ast.Module, path: Path) -> Dict[str, str]:
    return {
        symbol: summary
        for _, symbol, summary in _routine_listing_entries(tree, path)
    }


def _definition_summaries(tree: ast.Module) -> Dict[str, str]:
    summaries: Dict[str, str] = {}
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
        self.annotated_assignments: Dict[str, list[int]] = {}
        self.annotated_values: Dict[str, list[Tuple[int, ast.expr]]] = {}
        self.annotated_types: Dict[str, list[Tuple[int, ast.expr]]] = {}

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
        if isinstance(node.target, ast.Name):
            self.annotated_types.setdefault(node.target.id, []).append(
                (node.lineno, node.annotation)
            )
            if node.value is not None:
                self.annotated_assignments.setdefault(
                    node.target.id,
                    [],
                ).append(node.lineno)
                self.annotated_values.setdefault(node.target.id, []).append(
                    (node.lineno, node.value)
                )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node)


class _PrivateFunctionBodyVisitor(ast.NodeVisitor):
    """Inspect direct control flow without entering nested function scopes."""

    def __init__(self, function: ast.AST) -> None:
        self.function = function
        self.has_value_return = False
        self.has_yield = False
        self.has_raise = False
        self.raised_exceptions: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.function:
            for statement in node.body:
                self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.function:
            for statement in node.body:
                self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            self.has_value_return = True

    def visit_Yield(self, node: ast.Yield) -> None:
        self.has_yield = True

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.has_yield = True

    def visit_Raise(self, node: ast.Raise) -> None:
        self.has_raise = True
        exception = node.exc
        if isinstance(exception, ast.Call):
            exception = exception.func
        if isinstance(exception, (ast.Attribute, ast.Name)):
            rendered = ast.unparse(exception)
            if rendered not in self.raised_exceptions:
                self.raised_exceptions.append(rendered)


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


def _numpydoc_parameter_names(docstring: str) -> list[str]:
    """Extract ordered names from one NumPy-style Parameters section."""
    lines = docstring.splitlines()
    try:
        index = (
            next(
                index
                for index, line in enumerate(lines)
                if line.strip() == "Parameters"
                and index + 1 < len(lines)
                and set(lines[index + 1].strip()) == {"-"}
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
            and set(lines[index + 1].strip()) == {"-"}
        ):
            break
        line = lines[index]
        if line and not line[0].isspace() and " : " in line:
            declaration = line.split(" : ", maxsplit=1)[0]
            names.extend(
                name.strip().lstrip("*") for name in declaration.split(",")
            )
        index += 1
    return names


def _numpydoc_declarations(
    docstring: str,
    section: str,
) -> list[Tuple[str, str]]:
    """Extract ordered ``name : type`` declarations from one section."""
    lines = docstring.splitlines()
    try:
        index = (
            next(
                index
                for index, line in enumerate(lines)
                if line.strip() == section
                and index + 1 < len(lines)
                and set(lines[index + 1].strip()) == {"-"}
            )
            + 2
        )
    except StopIteration:
        return []

    declarations: list[Tuple[str, str]] = []
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and lines[index].strip()
            and set(lines[index + 1].strip()) == {"-"}
        ):
            break
        line = lines[index]
        if line and not line[0].isspace() and " : " in line:
            names, documented_type = line.split(" : ", maxsplit=1)
            declarations.extend(
                (name.strip().lstrip("*"), documented_type.strip())
                for name in names.split(",")
            )
        index += 1
    return declarations


def _numpydoc_raise_names(docstring: str) -> list[str]:
    """Extract exception names from one NumPy-style ``Raises`` section."""
    lines = docstring.splitlines()
    try:
        index = (
            next(
                index
                for index, line in enumerate(lines)
                if line.strip() == "Raises"
                and index + 1 < len(lines)
                and set(lines[index + 1].strip()) == {"-"}
            )
            + 2
        )
    except StopIteration:
        return []

    exceptions: list[str] = []
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and lines[index].strip()
            and set(lines[index + 1].strip()) == {"-"}
        ):
            break
        line = lines[index]
        if line and not line[0].isspace():
            exception = line.split(" : ", maxsplit=1)[0].strip()
            if exception:
                exceptions.append(exception)
        index += 1
    return exceptions


def _numpydoc_descriptions(
    docstring: str,
    section: str,
) -> Dict[str, str]:
    """Extract declaration descriptions from one NumPy-style section."""
    lines = docstring.splitlines()
    try:
        index = (
            next(
                index
                for index, line in enumerate(lines)
                if line.strip() == section
                and index + 1 < len(lines)
                and set(lines[index + 1].strip()) == {"-"}
            )
            + 2
        )
    except StopIteration:
        return {}

    descriptions: Dict[str, str] = {}
    current_names: list[str] = []
    current_description: list[str] = []

    def save_current() -> None:
        description = " ".join(current_description).strip()
        for name in current_names:
            descriptions[name] = description

    while index < len(lines):
        if (
            index + 1 < len(lines)
            and lines[index].strip()
            and set(lines[index + 1].strip()) == {"-"}
        ):
            break
        line = lines[index]
        if line and not line[0].isspace():
            save_current()
            current_description = []
            declaration = line.split(" : ", maxsplit=1)[0]
            current_names = [
                name.strip().lstrip("*") for name in declaration.split(",")
            ]
        elif line.strip() and current_names:
            current_description.append(line.strip())
        index += 1
    save_current()
    return descriptions


def _private_default_parameter_names(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[str]:
    """Return parameters that own an explicit default value."""
    positional = [*function.args.posonlyargs, *function.args.args]
    default_count = len(function.args.defaults)
    positional_defaults = positional[-default_count:] if default_count else []
    names = [
        argument.arg
        for argument in positional_defaults
        if argument.arg not in {"self", "cls"}
    ]
    names.extend(
        argument.arg
        for argument, default in zip(
            function.args.kwonlyargs,
            function.args.kw_defaults,
            strict=True,
        )
        if default is not None and argument.arg not in {"self", "cls"}
    )
    return names


def _private_static_parameter_names(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[str]:
    """Return literal JIT-static parameter names from decorators."""
    names: list[str] = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "static_argnames":
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str):
                values = (value,)
            elif isinstance(value, (list, tuple)):
                values = tuple(value)
            else:
                continue
            names.extend(item for item in values if isinstance(item, str))
    return names


def _normalized_doc_type(value: str) -> str:
    """Normalize harmless formatting differences in one annotation."""
    try:
        expression: ast.Expression = ast.parse(value, mode="eval")
    except SyntaxError:
        normalized: str = value.strip()
    else:
        normalized = ast.dump(expression.body, include_attributes=False)
    return normalized


def _private_signature_declarations(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[Tuple[str, str]]:
    """Return private parameters and exact annotations in signature order."""
    arguments: list[ast.arg] = [
        *function.args.posonlyargs,
        *function.args.args,
    ]
    if function.args.vararg is not None:
        arguments.append(function.args.vararg)
    arguments.extend(function.args.kwonlyargs)
    if function.args.kwarg is not None:
        arguments.append(function.args.kwarg)
    declarations: list[Tuple[str, str]] = []
    for argument in arguments:
        if argument.arg in {"self", "cls"}:
            continue
        annotation = argument.annotation
        if annotation is None:
            continue
        declarations.append((argument.arg, ast.unparse(annotation)))
    return declarations


def _fixed_tuple_annotation_types(annotation: ast.expr) -> list[str] | None:
    """Return fixed positional types from one tuple annotation."""
    if not isinstance(annotation, ast.Subscript):
        return None
    if ast.unparse(annotation.value) not in {"Tuple", "tuple"}:
        return None
    slice_node = annotation.slice
    elements = (
        list(slice_node.elts)
        if isinstance(slice_node, ast.Tuple)
        else [slice_node]
    )
    if any(
        isinstance(element, ast.Constant) and element.value is Ellipsis
        for element in elements
    ):
        return None
    return [ast.unparse(element) for element in elements]


def _private_return_candidates(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[list[Tuple[str | None, str]]]:
    """Return valid documented declarations for each direct return path."""
    visitor = _ReturnBindingVisitor(function)
    visitor.visit(function)
    known_types: Dict[str, ast.expr] = {}
    for argument in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        if argument.annotation is not None:
            known_types[argument.arg] = argument.annotation
    if function.args.vararg is not None:
        annotation = function.args.vararg.annotation
        if annotation is not None:
            known_types[function.args.vararg.arg] = annotation
    if function.args.kwarg is not None:
        annotation = function.args.kwarg.annotation
        if annotation is not None:
            known_types[function.args.kwarg.arg] = annotation
    for name, entries in visitor.annotated_types.items():
        known_types[name] = max(entries, key=lambda item: item[0])[1]

    candidates: list[list[Tuple[str | None, str]]] = []
    for returned in visitor.returns:
        if returned.value is None:
            continue
        if isinstance(returned.value, ast.Name):
            return_name: str | None = returned.value.id
            types = [
                (line_number, annotation)
                for line_number, annotation in visitor.annotated_types.get(
                    returned.value.id,
                    [],
                )
                if line_number <= returned.lineno
            ]
            return_annotation = (
                max(types, key=lambda item: item[0])[1]
                if types
                else function.returns
            )
            values = [
                (line_number, value)
                for line_number, value in visitor.annotated_values.get(
                    returned.value.id,
                    [],
                )
                if line_number <= returned.lineno
            ]
            returned_value = (
                max(values, key=lambda item: item[0])[1]
                if values
                else returned.value
            )
        else:
            return_name = None
            return_annotation = function.returns
            returned_value = returned.value
        if return_annotation is None:
            candidates.append([])
            continue

        positional_types = _fixed_tuple_annotation_types(return_annotation)
        if positional_types is None:
            candidates.append([(return_name, ast.unparse(return_annotation))])
            continue

        if isinstance(returned_value, ast.Tuple) and len(
            returned_value.elts
        ) == len(positional_types):
            names = [
                element.id if isinstance(element, ast.Name) else None
                for element in returned_value.elts
            ]
        else:
            names = [None] * len(positional_types)
        candidates.append(list(zip(names, positional_types, strict=True)))
    return candidates


def _return_declarations_match(
    actual: list[Tuple[str, str]],
    candidates: list[list[Tuple[str | None, str]]],
) -> bool:
    """Return whether declarations match every direct return path exactly."""
    normalized_actual = [
        (name, _normalized_doc_type(annotation)) for name, annotation in actual
    ]

    def matches(candidate: list[Tuple[str | None, str]]) -> bool:
        """Return whether one direct return path matches the declarations."""
        if len(candidate) != len(normalized_actual):
            return False
        return all(
            (expected_name is None or actual_name == expected_name)
            and actual_type == _normalized_doc_type(expected_type)
            for (actual_name, actual_type), (
                expected_name,
                expected_type,
            ) in zip(
                normalized_actual,
                candidate,
                strict=True,
            )
        )

    return bool(candidates) and all(
        matches(candidate) for candidate in candidates
    )


def _export_definition(
    tree: ast.Module,
    symbol: str,
    path: Path,
) -> Tuple[ast.AST, str]:
    """Return one public definition node and its documentation text."""
    for index, node in enumerate(tree.body):
        if isinstance(
            node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)
        ):
            if node.name == symbol:
                return node, ast.get_docstring(node) or ""

        targets: Tuple[ast.expr, ...] = ()
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


def _forward_test_references(docstring: str) -> list[Tuple[str, str]]:
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


def test_removed_convergent_born_namespace_is_absent() -> None:
    """Require the superseded verbose Born package name to remain removed."""
    assert find_spec("ptyrodactyl.convergent_born") is None


def test_private_tools_owns_exact_internal_infrastructure_leaves() -> None:
    """Require _tools to own only its seven internal leaves and type marker."""
    assert find_spec("ptyrodactyl._tools") is not None
    tools_path = _module_path("ptyrodactyl._tools").parent
    expected_files = {"__init__.py", "py.typed"}
    expected_files.update(f"{name}.py" for name in _INTERNAL_TOOL_LEAVES)
    actual_files = {
        path.name for path in tools_path.iterdir() if path.is_file()
    }
    assert actual_files == expected_files
    tools_module = importlib.import_module("ptyrodactyl._tools")
    assert tools_module.__all__


def test_private_tools_exports_have_synchronized_routine_listings() -> None:
    """Synchronize every internal seam with its leaf and aggregate docs."""
    package = "ptyrodactyl._tools"
    package_module = importlib.import_module(package)
    init_path = _module_path(package)
    init_tree = _parse(init_path)
    init_all = _literal_all(init_tree, init_path)
    init_entries = _routine_listing_entries(init_tree, init_path)
    init_listings = {symbol: summary for _, symbol, summary in init_entries}
    expected_init_entries = sorted(
        init_entries,
        key=lambda entry: (
            _ROUTINE_KIND_ORDER[entry[0]],
            entry[1].casefold(),
        ),
    )
    assert init_entries == expected_init_entries
    assert init_all == sorted(init_all, key=str.casefold)
    assert set(init_all) == set(init_listings)

    owners: Dict[str, Path] = {}
    for leaf_path in _leaf_modules(package):
        leaf_tree = _parse(leaf_path)
        leaf_all = _literal_all(leaf_tree, leaf_path)
        leaf_entries = _routine_listing_entries(leaf_tree, leaf_path)
        leaf_listings = {
            symbol: summary for _, symbol, summary in leaf_entries
        }
        expected_leaf_entries = sorted(
            leaf_entries,
            key=lambda entry: (
                _ROUTINE_KIND_ORDER[entry[0]],
                entry[1].casefold(),
            ),
        )
        assert leaf_entries == expected_leaf_entries
        assert leaf_all == sorted(leaf_all, key=str.casefold)
        assert set(leaf_all) == set(leaf_listings)
        definitions = _definition_summaries(leaf_tree)
        leaf_module = importlib.import_module(f"{package}.{leaf_path.stem}")
        for symbol in leaf_all:
            assert not symbol.startswith("_")
            assert symbol not in owners, (
                f"{symbol!r} is owned by both {owners[symbol]} and {leaf_path}"
            )
            owners[symbol] = leaf_path
            if symbol in definitions:
                assert leaf_listings[symbol] == definitions[symbol]
            assert init_listings[symbol] == leaf_listings[symbol]
            assert getattr(package_module, symbol) is getattr(
                leaf_module, symbol
            )
    assert set(init_all) == set(owners)


def test_package_root_contains_only_its_python_initializer() -> None:
    """Forbid Python leaf modules at the ptyrodactyl package root."""
    source_root = Path(ptyrodactyl.__file__).parent
    root_python_files = {
        path.name for path in source_root.glob("*.py") if path.is_file()
    }
    assert root_python_files == {"__init__.py"}
    assert (source_root / "py.typed").is_file()


def test_private_infrastructure_has_no_root_compatibility_aliases() -> None:
    """Keep every former root private-module path undiscoverable."""
    assert find_spec("ptyrodactyl.tools") is None
    assert "tools" not in ptyrodactyl.__all__
    for module_name in _INTERNAL_TOOL_LEAVES:
        former_name = f"_{module_name}"
        assert find_spec(f"ptyrodactyl.{former_name}") is None
        assert not hasattr(ptyrodactyl, former_name)


def test_removed_invert_namespace_is_absent() -> None:
    """Require multislice reconstruction's former package to stay removed."""
    assert find_spec("ptyrodactyl.invert") is None


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

        leaf_all: Dict[str, list[Path]] = {}
        leaf_listings: Dict[Path, Dict[str, str]] = {}
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
    owners: Dict[str, str] = {}
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
    carrier_owners: Dict[str, Path] = {}
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


def test_canonical_carrier_fields_use_declared_storage_dtypes() -> None:
    """Pin exact-width annotations on canonical stored array fields.

    Parse the carrier declarations whose factories canonicalize storage and
    compare each outer jaxtyping dtype with the documented width.
    """
    types_root = Path(ptyrodactyl.__file__).parent / "types"
    for (
        module_name,
        class_name,
        field_name,
        expected_dtype,
    ) in _CANONICAL_CARRIER_DTYPES:
        path = types_root / module_name
        tree = _parse(path)
        carrier = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        field = next(
            node
            for node in carrier.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == field_name
        )
        annotation = field.annotation
        assert isinstance(annotation, ast.Subscript), (
            f"{path}: {class_name}.{field_name} must use jaxtyping"
        )
        actual_dtype = ast.unparse(annotation.value)
        assert actual_dtype == expected_dtype, (
            f"{path}: {class_name}.{field_name} uses {actual_dtype}; "
            f"expected {expected_dtype}"
        )


def test_private_functions_use_structured_numpydoc() -> None:
    """Enforce exact structured private documentation contracts.

    Match parameters, returns, raises, defaults, static arguments, and section
    order to each production scope, then reject private public-surface entries.
    """
    source_root = Path(ptyrodactyl.__file__).parent
    heading_pattern = re.compile(
        r"(?m)^(Extended Summary|Implementation Logic|Parameters|Returns|"
        r"Yields|Raises|Notes|References|See Also|Examples)\n-{3,}\n"
    )
    offenders: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = _parse(path)
        exports = set(_literal_all(tree, path))
        module_docstring = ast.get_docstring(tree) or ""
        listed_private = set()
        if "Routine Listings" in module_docstring:
            listed_private = {
                symbol
                for _, symbol, _ in _routine_listing_entries(tree, path)
                if symbol.rsplit(".", maxsplit=1)[-1].startswith("_")
            }
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name.startswith("_")
            and not node.name.startswith("__")
        ):
            location = f"{path}:{function.lineno} {function.name}"
            if function.name in exports:
                offenders.append(f"{location}: private export")
            if any(
                reference.rsplit(".", maxsplit=1)[-1] == function.name
                for reference in listed_private
            ):
                offenders.append(f"{location}: private Routine Listing")

            docstring = ast.get_docstring(function) or ""
            if not docstring:
                offenders.append(f"{location}: missing docstring")
                continue
            summary = docstring.splitlines()[0].strip()
            if not summary.startswith("PRIVATE: "):
                offenders.append(f"{location}: summary must start PRIVATE:")
            if ":see:" in docstring.lower():
                offenders.append(f"{location}: private :see: reference")

            headings = [
                match.group(1) for match in heading_pattern.finditer(docstring)
            ]
            ranks = [
                _PRIVATE_FUNCTION_SECTION_ORDER[heading]
                for heading in headings
            ]
            if ranks != sorted(ranks):
                offenders.append(f"{location}: section order {headings}")
            if len(headings) != len(set(headings)):
                offenders.append(f"{location}: duplicate section {headings}")
            if "Returns" in headings and "Yields" in headings:
                offenders.append(f"{location}: both Returns and Yields")

            expected_parameters = _private_signature_declarations(function)
            actual_parameters = _numpydoc_declarations(
                docstring,
                "Parameters",
            )
            normalized_expected_parameters = [
                (name, _normalized_doc_type(annotation))
                for name, annotation in expected_parameters
            ]
            normalized_actual_parameters = [
                (name, _normalized_doc_type(annotation))
                for name, annotation in actual_parameters
            ]
            if normalized_actual_parameters != normalized_expected_parameters:
                offenders.append(
                    f"{location}: parameters {actual_parameters}; "
                    f"expected {expected_parameters}"
                )
            parameter_descriptions = _numpydoc_descriptions(
                docstring,
                "Parameters",
            )
            missing_parameter_descriptions = [
                name
                for name, _ in actual_parameters
                if not parameter_descriptions.get(name)
            ]
            if missing_parameter_descriptions:
                offenders.append(
                    f"{location}: empty parameter descriptions "
                    f"{missing_parameter_descriptions}"
                )
            missing_defaults = [
                name
                for name in _private_default_parameter_names(function)
                if not any(
                    marker in parameter_descriptions.get(name, "").lower()
                    for marker in ("default", "if omitted", "when omitted")
                )
            ]
            if missing_defaults:
                offenders.append(
                    f"{location}: undocumented defaults {missing_defaults}"
                )
            missing_static_details = [
                name
                for name in _private_static_parameter_names(function)
                if not (
                    "static" in parameter_descriptions.get(name, "").lower()
                    and "retrac"
                    in parameter_descriptions.get(name, "").lower()
                )
            ]
            if missing_static_details:
                offenders.append(
                    f"{location}: static parameters lack retracing details "
                    f"{missing_static_details}"
                )
            body = _PrivateFunctionBodyVisitor(function)
            body.visit(function)
            required_sections = {
                "Parameters": bool(expected_parameters),
                "Returns": body.has_value_return and not body.has_yield,
                "Yields": body.has_yield,
                "Raises": body.has_raise,
            }
            for section, required in required_sections.items():
                if required and section not in headings:
                    offenders.append(f"{location}: missing {section}")

            expected_returns = _private_return_candidates(function)
            actual_returns = _numpydoc_declarations(docstring, "Returns")
            if body.has_value_return and not _return_declarations_match(
                actual_returns,
                expected_returns,
            ):
                offenders.append(
                    f"{location}: returns {actual_returns}; "
                    f"expected every path {expected_returns}"
                )
            return_descriptions = _numpydoc_descriptions(
                docstring,
                "Returns",
            )
            missing_return_descriptions = [
                name
                for name, _ in actual_returns
                if not return_descriptions.get(name)
            ]
            if missing_return_descriptions:
                offenders.append(
                    f"{location}: empty return descriptions "
                    f"{missing_return_descriptions}"
                )

            documented_exceptions = _numpydoc_raise_names(docstring)
            missing_exceptions = [
                exception
                for exception in body.raised_exceptions
                if exception not in documented_exceptions
            ]
            if missing_exceptions:
                offenders.append(
                    f"{location}: undocumented raises {missing_exceptions}"
                )
            raise_descriptions = _numpydoc_descriptions(docstring, "Raises")
            missing_raise_descriptions = [
                exception
                for exception in documented_exceptions
                if not raise_descriptions.get(exception)
            ]
            if missing_raise_descriptions:
                offenders.append(
                    f"{location}: empty raise descriptions "
                    f"{missing_raise_descriptions}"
                )
    assert offenders == [], "\n".join(offenders)


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
    """Reject renamed imports and reaches below an owning aggregate."""
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
                is_internal_tools_aggregate = (
                    imported_owner == "_tools" and len(parts) == 2
                )
                if imported_owner == "_tools" and len(parts) > 2:
                    offenders.append(
                        f"{path}:{node.lineno} deep _tools leaf import"
                    )
                elif (
                    owner is not None
                    and imported_owner != owner
                    and len(parts) > 2
                ):
                    offenders.append(
                        f"{path}:{node.lineno} deep import {node.module}"
                    )
                if is_internal_tools_aggregate and any(
                    alias.name.startswith("_") for alias in node.names
                ):
                    offenders.append(
                        f"{path}:{node.lineno} private _tools seam import"
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


def test_tuple_and_dict_hints_use_beartype_typing() -> None:
    """Require Tuple and Dict hints to use beartype's capital forms."""
    repository_root = Path(__file__).parents[2]
    roots = (repository_root / "src", repository_root / "tests")
    required_names = {"Dict", "Tuple"}
    offenders: list[str] = []

    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = _parse(path)
            beartype_names: set[str] = set()
            standard_typing_aliases: set[str] = set()
            builtins_module_aliases: set[str] = set()
            built_in_aliases: Dict[str, str] = {}
            alias_assignments: list[Tuple[str, ast.expr]] = []
            built_in_hint_locations: set[Tuple[int, str]] = set()
            hint_roots: list[Tuple[int, ast.AST]] = []

            for node in ast.walk(tree):
                raw_line_number = getattr(node, "lineno", 1)
                node_line_number: int = (
                    raw_line_number if isinstance(raw_line_number, int) else 1
                )
                if isinstance(node, ast.ImportFrom):
                    imported_names = {
                        alias.name
                        for alias in node.names
                        if alias.name in required_names
                    }
                    if node.module == "beartype.typing":
                        beartype_names.update(
                            alias.asname or alias.name
                            for alias in node.names
                            if alias.name in required_names
                        )
                    elif imported_names:
                        offenders.append(
                            f"{path}:{node.lineno} imports "
                            f"{sorted(imported_names)} from {node.module}"
                        )
                    if node.module == "builtins":
                        built_in_aliases.update(
                            {
                                alias.asname or alias.name: alias.name
                                for alias in node.names
                                if alias.name in {"dict", "tuple"}
                            }
                        )
                elif isinstance(node, ast.Import):
                    standard_typing_aliases.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name in {"typing", "typing_extensions"}
                    )
                    builtins_module_aliases.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name == "builtins"
                    )
                elif isinstance(node, ast.AnnAssign):
                    hint_roots.append((node.lineno, node.annotation))
                    if node.value is not None and ast.unparse(
                        node.annotation
                    ).endswith("TypeAlias"):
                        hint_roots.append((node.lineno, node.value))
                    if node.value is not None and isinstance(
                        node.target, ast.Name
                    ):
                        alias_assignments.append((node.target.id, node.value))
                elif isinstance(node, ast.Assign):
                    alias_assignments.extend(
                        (target.id, node.value)
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    )
                elif isinstance(node, ast.arg) and node.annotation is not None:
                    hint_roots.append((node.lineno, node.annotation))
                elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    if node.returns is not None:
                        hint_roots.append((node.lineno, node.returns))
                elif isinstance(node, ast.TypeAlias):
                    hint_roots.append((node.lineno, node.value))
                elif (
                    isinstance(node, ast.Call)
                    and node.args
                    and (
                        (
                            isinstance(node.func, ast.Name)
                            and node.func.id == "cast"
                        )
                        or (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "cast"
                        )
                    )
                ):
                    hint_roots.append((node.lineno, node.args[0]))

                if isinstance(node, ast.Call):
                    callable_name = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else ""
                    )
                    if callable_name in {
                        "ParamSpec",
                        "TypeVar",
                        "TypeVarTuple",
                    }:
                        hint_roots.extend(
                            (node.lineno, argument)
                            for argument in node.args[1:]
                        )
                        hint_roots.extend(
                            (node.lineno, keyword.value)
                            for keyword in node.keywords
                            if keyword.arg
                            in {"bound", "default", "default_value"}
                        )
                    elif callable_name == "NewType" and len(node.args) >= 2:
                        hint_roots.append((node.lineno, node.args[1]))

                for type_parameter in getattr(node, "type_params", ()):
                    for attribute in ("bound", "default_value"):
                        value = getattr(type_parameter, attribute, None)
                        if value is not None:
                            hint_roots.append((node_line_number, value))

                type_comment = getattr(node, "type_comment", None)
                if isinstance(type_comment, str):
                    try:
                        parsed_comment = ast.parse(
                            type_comment,
                            mode=(
                                "func_type"
                                if isinstance(
                                    node,
                                    (ast.AsyncFunctionDef, ast.FunctionDef),
                                )
                                else "eval"
                            ),
                        )
                    except SyntaxError:
                        continue
                    hint_roots.append((node_line_number, parsed_comment))

            unresolved_assignments = alias_assignments
            while unresolved_assignments:
                remaining_assignments: list[Tuple[str, ast.expr]] = []
                changed = False
                for target_name, value in unresolved_assignments:
                    built_in_name: str | None = None
                    if isinstance(value, ast.Name) and value.id in {
                        "dict",
                        "tuple",
                    }:
                        built_in_name = value.id
                    elif (
                        isinstance(value, ast.Name)
                        and value.id in built_in_aliases
                    ):
                        built_in_name = built_in_aliases[value.id]
                    elif (
                        isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Name)
                        and value.value.id in builtins_module_aliases
                        and value.attr in {"dict", "tuple"}
                    ):
                        built_in_name = value.attr
                    if built_in_name is None:
                        remaining_assignments.append((target_name, value))
                        continue
                    built_in_aliases[target_name] = built_in_name
                    changed = True
                if not changed:
                    break
                unresolved_assignments = remaining_assignments

            for line_number, hint_root in hint_roots:
                parsed_root = hint_root
                if isinstance(hint_root, ast.Constant) and isinstance(
                    hint_root.value,
                    str,
                ):
                    try:
                        parsed_root = ast.parse(
                            hint_root.value,
                            mode="eval",
                        ).body
                    except SyntaxError:
                        continue
                pending_nodes = [parsed_root]
                while pending_nodes:
                    hint_node = pending_nodes.pop()
                    if isinstance(hint_node, ast.Constant) and isinstance(
                        hint_node.value,
                        str,
                    ):
                        try:
                            nested_hint = ast.parse(
                                hint_node.value,
                                mode="eval",
                            ).body
                        except SyntaxError:
                            continue
                        pending_nodes.append(nested_hint)
                        continue
                    pending_nodes.extend(ast.iter_child_nodes(hint_node))
                    if isinstance(hint_node, ast.Name) and hint_node.id in {
                        "dict",
                        "tuple",
                    }:
                        built_in_hint_locations.add(
                            (line_number, hint_node.id)
                        )
                    elif (
                        isinstance(hint_node, ast.Name)
                        and hint_node.id in built_in_aliases
                    ):
                        built_in_hint_locations.add(
                            (line_number, built_in_aliases[hint_node.id])
                        )
                    elif (
                        isinstance(hint_node, ast.Attribute)
                        and isinstance(hint_node.value, ast.Name)
                        and hint_node.value.id in builtins_module_aliases
                        and hint_node.attr in {"dict", "tuple"}
                    ):
                        built_in_hint_locations.add(
                            (line_number, hint_node.attr)
                        )

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in required_names
                    and node.id not in beartype_names
                ):
                    offenders.append(
                        f"{path}:{node.lineno} {node.id} lacks a "
                        "beartype.typing import"
                    )
                elif (
                    isinstance(node, ast.Attribute)
                    and node.attr in required_names
                    and isinstance(node.value, ast.Name)
                    and node.value.id in standard_typing_aliases
                ):
                    offenders.append(
                        f"{path}:{node.lineno} qualifies "
                        f"{node.value.id}.{node.attr}"
                    )
                elif isinstance(node, ast.Subscript) and (
                    isinstance(node.value, ast.Name)
                    and (
                        node.value.id in {"dict", "tuple"}
                        or node.value.id in built_in_aliases
                    )
                    or isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id in builtins_module_aliases
                    and node.value.attr in {"dict", "tuple"}
                ):
                    built_in_name = (
                        node.value.id
                        if isinstance(node.value, ast.Name)
                        and node.value.id in {"dict", "tuple"}
                        else built_in_aliases[node.value.id]
                        if isinstance(node.value, ast.Name)
                        else node.value.attr
                    )
                    built_in_hint_locations.add((node.lineno, built_in_name))

            offenders.extend(
                f"{path}:{line_number} uses built-in {name} hint"
                for line_number, name in sorted(built_in_hint_locations)
            )

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
