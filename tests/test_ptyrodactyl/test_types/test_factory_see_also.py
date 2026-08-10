"""Enforce carrier-to-factory documentation links."""

from __future__ import annotations

import ast
from pathlib import Path

from beartype.typing import Dict

_TYPES_SOURCE = Path(__file__).parents[3] / "src" / "ptyrodactyl" / "types"
_FACTORY_BY_MODULE: Dict[str, Dict[str, str]] = {
    "born_types": {
        "GalerkinOperator": "create_galerkin_operator",
        "GalerkinSolveResult": "create_galerkin_solve_result",
    },
    "crystal_types": {
        "CrystalData": "create_crystal_data",
        "CrystalStructure": "create_crystal_structure",
    },
    "distributions": {
        "Distribution": "create_distribution",
    },
    "electron_types": {
        "AtomicSliceData": "create_atomic_slice_data",
        "AxisUpdate": "create_axis_update",
        "CalibratedArray": "create_calibrated_array",
        "DetectorConfig": "create_detector_config",
        "EnsembleAxes": "create_ensemble_axes",
        "MicroscopeConfig": "create_microscope_config",
        "PotentialSlices": "create_potential_slices",
        "ProbeModes": "create_probe_modes",
        "STEM4D": "create_stem4d",
    },
    "form_factor_types": {
        "KirklandParameters": "create_kirkland_parameters",
        "LobatoParameters": "create_lobato_parameters",
    },
    "galerkin_types": {
        "GalerkinPhysicalResidual": "create_galerkin_physical_residual",
        "GalerkinSource": "create_galerkin_source",
        "GalerkinStabilityProof": "create_galerkin_stability_proof",
        "GalerkinStabilityResult": "create_galerkin_stability_result",
    },
    "jacobian_types": {
        "AberrationParams": "create_ptycho_params",
        "CGState": "create_cg_state",
        "ExitWaveParams": "create_ptycho_params",
        "FisherState": "create_fisher_state",
        "GNState": "create_gn_state",
        "GeometryParams": "create_ptycho_params",
        "LMState": "create_lm_state",
        "LanczosState": "create_lanczos_state",
        "PositionParams": "create_ptycho_params",
        "ProbeModeParams": "create_ptycho_params",
        "PtychoParams": "create_ptycho_params",
    },
    "potential_types": {
        "Potential3D": "create_potential_3d",
    },
    "recon_types": {
        "LaplaceUncertainty": "create_laplace_uncertainty",
        "PosteriorSamples": "create_posterior_samples",
        "ReconProblem": "create_recon_problem",
        "ReconResult": "create_recon_result",
    },
}


def _literal_all(tree: ast.Module, path: Path) -> set[str]:
    """Return the literal public export set from one source module."""
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and node.value is not None
        ):
            exports = ast.literal_eval(node.value)
            if not isinstance(exports, list) or not all(
                isinstance(name, str) for name in exports
            ):
                raise AssertionError(f"{path}: __all__ must be a string list")
            return {name for name in exports if isinstance(name, str)}
    raise AssertionError(f"{path}: missing literal __all__")


def _is_eqx_module(node: ast.ClassDef) -> bool:
    """Return whether a class directly inherits from ``eqx.Module``."""
    return any(
        isinstance(base, ast.Attribute)
        and isinstance(base.value, ast.Name)
        and base.value.id == "eqx"
        and base.attr == "Module"
        for base in node.bases
    )


def test_public_carriers_name_their_validated_factories() -> None:
    """Require every public carrier to link its validated factory."""
    attributes_marker = "Attributes\n----------"
    see_also_marker = "See Also\n--------"

    for module_name, expected_factories in _FACTORY_BY_MODULE.items():
        path = _TYPES_SOURCE / f"{module_name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exports = _literal_all(tree, path)
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        public_carriers = {
            name
            for name, node in classes.items()
            if name in exports and _is_eqx_module(node)
        }

        assert public_carriers == set(expected_factories), path
        for class_name, factory_name in expected_factories.items():
            assert factory_name in exports, (
                f"{path}: {factory_name} not public"
            )
            assert factory_name in functions, f"{path}: {factory_name} missing"

            docstring = ast.get_docstring(classes[class_name]) or ""
            assert docstring.count(attributes_marker) == 1, (
                f"{path}: {class_name} needs one Attributes section"
            )
            assert docstring.count(see_also_marker) == 1, (
                f"{path}: {class_name} needs one See Also section"
            )
            assert docstring.index(attributes_marker) < docstring.index(
                see_also_marker
            ), f"{path}: {class_name} must put See Also after Attributes"

            see_also = docstring.split(see_also_marker, maxsplit=1)[1]
            factory_roles = [
                line.strip()
                for line in see_also.splitlines()
                if line.strip().startswith(":func:`")
            ]
            assert factory_roles == [f":func:`{factory_name}`"], (
                f"{path}: {class_name} must link only {factory_name}"
            )
