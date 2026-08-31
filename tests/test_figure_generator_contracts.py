# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Figure generator contracts
"""Guard tracked figure generators and their static quality ownership."""

from __future__ import annotations

import ast
from pathlib import Path

from tools import preflight

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
PYPROJECT = ROOT / "pyproject.toml"
PREFLIGHT = ROOT / "tools" / "preflight.py"
STATIC_WORKFLOW = ROOT / ".github" / "workflows" / "ci-static-analysis.yml"
EXPECTED_GENERATOR_COUNT = 4


def _generator_scripts() -> tuple[Path, ...]:
    """Return every tracked Python figure generator in deterministic order."""
    return tuple(sorted(FIGURES.glob("generate_*.py")))


def _has_main_guard(tree: ast.Module) -> bool:
    """Return whether a generator protects direct execution behind a main guard."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


def test_tracked_figure_generators_have_documented_public_surfaces() -> None:
    """Require parseable, documented generators with guarded direct execution."""
    scripts = _generator_scripts()

    assert len(scripts) == EXPECTED_GENERATOR_COUNT
    for script in scripts:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        public_functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }
        assert public_functions, f"{script.name} must expose a public generator"
        for function_name, function in public_functions.items():
            assert ast.get_docstring(function), (
                f"{script.name}:{function_name} must have a native docstring"
            )
        assert _has_main_guard(tree), f"{script.name} must guard direct execution"


def test_figure_generators_remain_in_static_quality_ownership() -> None:
    """Pin figure generators into configured Ruff, preflight, and hosted CI."""
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    workflow = STATIC_WORKFLOW.read_text(encoding="utf-8")
    gates = dict(preflight.STATIC_GATES)

    assert '"figures/**" = ["D"]' not in pyproject
    assert "figures/" in gates["ruff check"]
    assert "figures/" in gates["ruff format"]
    assert "ruff check src/ tests/ examples/ figures/" in workflow
    assert "ruff format --check src/ tests/ examples/ figures/" in workflow


def test_repository_header_source_has_no_volatile_release_counts() -> None:
    """Prevent the generated repository header from embedding stale counters."""
    source = (FIGURES / "generate_header.py").read_text(encoding="utf-8")

    assert " TESTS |" not in source
    assert "SIMULATOR-FIRST | FAIL-CLOSED EVIDENCE | 16-LAYER UPDE" in source
