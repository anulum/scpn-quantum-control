# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Documentation-layout tests for the differentiable readiness ledger."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import scpn_quantum_control.phase.differentiable_readiness as readiness


def _function_nodes(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield every function definition from a parsed module tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


def _docstring_end_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    """Return the one-based end line for a function docstring."""
    if not node.body:
        return None
    first = node.body[0]
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return first.end_lineno


def test_phase_differentiable_readiness_docstrings_follow_contracts() -> None:
    """Readiness ledger docstrings should satisfy magic-method and D202 rules."""
    source_path = Path(readiness.__file__)
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(source_path))
    offenders: list[str] = []

    for node in _function_nodes(tree):
        end_line = _docstring_end_line(node)
        is_magic_method = node.name.startswith("__") and node.name.endswith("__")
        if end_line is None:
            if is_magic_method:
                offenders.append(f"missing:{node.name}:{node.lineno}")
            continue
        if end_line < len(lines) and lines[end_line].strip() == "":
            offenders.append(f"blank:{node.name}:{end_line + 1}")

    assert offenders == []
