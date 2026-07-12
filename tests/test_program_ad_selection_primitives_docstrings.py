# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD selection primitives docstrings tests
"""Documentation-layout tests for Program AD selection primitives."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import scpn_quantum_control.program_ad_selection_primitives as selection


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


def test_program_ad_selection_primitives_have_no_post_docstring_blanks() -> None:
    """Selection primitive docstrings should follow the repository D202 contract."""
    source_path = Path(selection.__file__)
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(source_path))
    offenders: list[str] = []

    for node in _function_nodes(tree):
        end_line = _docstring_end_line(node)
        if end_line is None or end_line >= len(lines):
            continue
        if lines[end_line].strip() == "":
            offenders.append(f"{node.name}:{end_line + 1}")

    assert offenders == []
