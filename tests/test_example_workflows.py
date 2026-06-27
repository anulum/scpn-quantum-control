# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — example workflow contracts
"""Static contracts for example workflow scripts.

Examples are intentionally not executed here because several are
demonstrations with optimisation loops or optional dependencies.  This
test constrains the workflow boundary that matters for import safety and
documentation freshness: scripts must parse, expose ``main()``, protect
execution behind a main guard, and be listed in the examples README.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
README = EXAMPLES / "README.md"
EXPECTED_EXAMPLE_SCRIPT_COUNT = 31


def _example_scripts() -> tuple[Path, ...]:
    """Return the committed example scripts in deterministic order."""
    return tuple(sorted(EXAMPLES.glob("[0-9][0-9]_*.py")))


def _has_main_guard(tree: ast.Module) -> bool:
    """Return True when a module has ``if __name__ == "__main__"``."""
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


def test_all_example_scripts_are_documented_and_import_safe() -> None:
    scripts = _example_scripts()
    readme = README.read_text(encoding="utf-8")

    assert len(scripts) == EXPECTED_EXAMPLE_SCRIPT_COUNT
    for script in scripts:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        public_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert "main" in public_functions, f"{script.name} must expose main()"
        assert _has_main_guard(tree), f"{script.name} must guard example execution"
        assert script.name in readme, f"{script.name} must be listed in examples/README.md"
