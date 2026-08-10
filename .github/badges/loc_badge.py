"""Regenerate the local lines-of-code badge JSON.

Extended Summary
----------------
Counts logical lines of code in ``src/ptyrodactyl`` with pygount and
writes ``.github/badges/loc.json`` in the shields.io endpoint schema.
Runs as a local pre-commit hook so the badge updates inside normal
commits — no CI job ever commits to the repository for this badge.
The file is rewritten only when the count changes, keeping the hook
silent on unrelated commits.

Routine Listings
----------------
:func:`main`
    Count lines of code and rewrite the badge JSON if it changed.
"""

import json
from pathlib import Path

import pygount.analysis
from pygments.lexers.python import PythonLexer

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_BADGE_PATH: Path = _REPO_ROOT / ".github" / "badges" / "loc.json"
_COUNT_TARGET: Path = _REPO_ROOT / "src" / "ptyrodactyl"


def _count_python_loc(count_target: Path) -> int:
    """Count Python source lines without Pygments language guessing.

    Parameters
    ----------
    count_target : Path
        Directory containing the Python source tree.

    Returns
    -------
    loc : int
        Pygount source-line total with every ``.py`` file parsed as Python.

    Notes
    -----
    Pygount delegates filename classification to Pygments. When IPython is
    installed, Pygments can classify ordinary ``.py`` files as either Python
    or IPython depending on entry-point discovery. Pygount treats Python
    docstrings as comments but IPython docstrings as code, so the badge count
    would otherwise depend on the invocation environment.
    """
    duplicate_pool = pygount.analysis.DuplicatePool()
    original_guess_lexer = pygount.analysis.guess_lexer

    def python_lexer(_source_path: str, _source_code: str) -> PythonLexer:
        return PythonLexer()

    pygount.analysis.guess_lexer = python_lexer
    try:
        analyses = (
            pygount.analysis.SourceAnalysis.from_file(
                str(source_path),
                group="ptyrodactyl",
                duplicate_pool=duplicate_pool,
            )
            for source_path in sorted(count_target.rglob("*.py"))
        )
        return sum(analysis.source_count for analysis in analyses)
    finally:
        pygount.analysis.guess_lexer = original_guess_lexer


def main() -> int:
    """Count lines of code and rewrite the badge JSON if it changed.

    Returns
    -------
    exit_code : int
        Zero on success. Pre-commit detects a modified badge file itself.
    """
    loc: str = str(_count_python_loc(_COUNT_TARGET))
    badge: str = (
        json.dumps(
            {
                "schemaVersion": 1,
                "label": "lines of code",
                "message": loc,
                "color": "blue",
            },
            indent=2,
        )
        + "\n"
    )
    if _BADGE_PATH.exists() and _BADGE_PATH.read_text() == badge:
        return 0
    _BADGE_PATH.write_text(badge)
    print(f"updated {_BADGE_PATH.relative_to(_REPO_ROOT)}: {loc} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
