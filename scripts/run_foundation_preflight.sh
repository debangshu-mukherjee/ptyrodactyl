#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "usage: $0 EVIDENCE_RECORD [--validate-only]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

record_path=$1
validate_only=${2:-}
if [[ -n $validate_only && $validate_only != "--validate-only" ]]; then
    usage
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

python3 - "$record_path" <<'PY'
import json
import re
import sys
from pathlib import Path

record_path = Path(sys.argv[1])
if not record_path.is_file():
    raise SystemExit(f"evidence record does not exist: {record_path}")

with record_path.open(encoding="utf-8") as stream:
    record = json.load(stream)

expected_commands = {
    "frozen_install": "uv sync --frozen --extra dev",
    "lock_check": "uv lock --check",
    "cpu_tests": (
        "MPLCONFIGDIR=<preflight-work-directory>/matplotlib "
        "JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "
        "uv run --frozen pytest -q -n 0"
    ),
    "ruff_check": "uv run --frozen ruff check pyproject.toml src tests",
    "ruff_format": "uv run --frozen ruff format --check src tests",
    "ty": "uv run --frozen ty check",
    "docs": (
        "MPLCONFIGDIR=<preflight-work-directory>/matplotlib "
        "JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "
        "uv run --frozen sphinx-build -a -E -W -b html docs/source "
        "<preflight-work-directory>/docs-html"
    ),
    "wheel_build": (
        "uv build --wheel --out-dir <preflight-work-directory>/dist"
    ),
    "fresh_venv": (
        "uv venv --python 3.13 --no-project "
        "<preflight-work-directory>/venv"
    ),
    "wheel_install": (
        "uv pip install --python "
        "<preflight-work-directory>/venv/bin/python <built-wheel>"
    ),
    "wheel_scalar_smoke": (
        "MPLCONFIGDIR=<preflight-work-directory>/matplotlib "
        "JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "
        "<preflight-work-directory>/venv/bin/python "
        "scripts/scalar_wheel_smoke.py"
    ),
}

errors = []
if record.get("schema_version") != 1:
    errors.append("schema_version must equal 1")
if record.get("scope") != "minimal-foundation-preflight":
    errors.append("scope must equal 'minimal-foundation-preflight'")

code = record.get("code")
if not isinstance(code, dict):
    errors.append("code must be an object")
else:
    for field in ("location", "branch_policy"):
        if not isinstance(code.get(field), str) or not code[field].strip():
            errors.append(f"code.{field} must be non-empty text")
    start_commit = code.get("start_commit")
    if not isinstance(start_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", start_commit
    ) is None:
        errors.append("code.start_commit must be a 40-character Git hash")

environment = record.get("environment")
if not isinstance(environment, dict):
    errors.append("environment must be an object")
else:
    if environment.get("accelerator") != "cpu":
        errors.append("environment.accelerator must equal 'cpu'")
    if not isinstance(environment.get("python"), str) or not environment[
        "python"
    ].strip():
        errors.append("environment.python must be non-empty text")

if record.get("commands") != expected_commands:
    errors.append("commands must exactly match the bounded runner commands")

artifacts = record.get("artifacts")
if not isinstance(artifacts, dict):
    errors.append("artifacts must be an object")
else:
    for field in ("wheel", "documentation", "data_cache", "artifact_store"):
        if not isinstance(artifacts.get(field), str) or not artifacts[
            field
        ].strip():
            errors.append(f"artifacts.{field} must be non-empty text")

data_gates = record.get("data_gates")
if not isinstance(data_gates, dict):
    errors.append("data_gates must be an object")
else:
    for name in ("prsco3_ssmm_2j11", "paradim_g4wa_0j57"):
        gate = data_gates.get(name)
        if not isinstance(gate, dict) or gate.get("status") != "OPEN":
            errors.append(f"data_gates.{name}.status must equal 'OPEN'")

    ornl = data_gates.get("ornl_independent")
    if not isinstance(ornl, dict):
        errors.append("data_gates.ornl_independent must be an object")
    else:
        if ornl.get("status") != "OPEN_UNSELECTED":
            errors.append(
                "data_gates.ornl_independent.status must equal "
                "'OPEN_UNSELECTED'"
            )
        required_fields = ornl.get("required_fields")
        expected_fields = {
            "identity",
            "provenance",
            "reuse_terms",
            "calibration",
            "dose",
            "geometry",
            "thickness_orientation",
            "scientific_role",
        }
        if not isinstance(required_fields, list) or not expected_fields.issubset(
            set(required_fields)
        ):
            errors.append(
                "data_gates.ornl_independent.required_fields is incomplete"
            )
        excluded = ornl.get("excluded_candidates")
        excluded_text = " ".join(excluded) if isinstance(excluded, list) else ""
        if "NiFe" not in excluded_text or "14503519" not in excluded_text:
            errors.append(
                "the ORNL record must explicitly exclude NiFe Zenodo 14503519"
            )

if errors:
    for error in errors:
        print(f"record error: {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"validated foundation evidence record: {record_path.resolve()}")
PY

if [[ $validate_only == "--validate-only" ]]; then
    exit 0
fi

cd -- "$repo_root"
preflight_work_dir=$(mktemp -d "${TMPDIR:-/tmp}/ptyrodactyl-preflight.XXXXXX")
mkdir -p "$preflight_work_dir/dist"
mkdir -p "$preflight_work_dir/docs-html"
mkdir -p "$preflight_work_dir/matplotlib"

uv lock --check
uv sync --frozen --extra dev
uv run --frozen ruff check pyproject.toml src tests
uv run --frozen ruff format --check src tests
uv run --frozen ty check

MPLCONFIGDIR="$preflight_work_dir/matplotlib" \
JAX_PLATFORM_NAME=cpu \
JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES= \
uv run --frozen pytest -q -n 0

MPLCONFIGDIR="$preflight_work_dir/matplotlib" \
JAX_PLATFORM_NAME=cpu \
JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES= \
uv run --frozen sphinx-build -a -E -W -b html \
    docs/source "$preflight_work_dir/docs-html"

uv build --wheel --out-dir "$preflight_work_dir/dist"
uv venv --python 3.13 --no-project "$preflight_work_dir/venv"
wheel_path=$(find "$preflight_work_dir/dist" -maxdepth 1 -type f \
    -name 'ptyrodactyl-*.whl' -print -quit)
if [[ -z $wheel_path ]]; then
    echo "preflight did not produce a ptyrodactyl wheel" >&2
    exit 1
fi
uv pip install --python "$preflight_work_dir/venv/bin/python" "$wheel_path"
(
    cd -- "$preflight_work_dir"
    MPLCONFIGDIR="$preflight_work_dir/matplotlib" \
    JAX_PLATFORM_NAME=cpu \
    JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES= \
    "$preflight_work_dir/venv/bin/python" \
        "$repo_root/scripts/scalar_wheel_smoke.py"
)

echo "foundation preflight passed"
echo "wheel artifacts: $preflight_work_dir/dist"
echo "documentation artifact: $preflight_work_dir/docs-html"
