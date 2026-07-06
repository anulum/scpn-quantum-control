# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD Adjoint Docstring Tests
"""Documentation-layout tests for Program AD adjoint generation."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import scpn_quantum_control.program_ad_adjoint_generation as generation


def _function_nodes(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


def _docstring_end_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    if not node.body:
        return None
    first = node.body[0]
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return first.end_lineno


def test_program_adjoint_generation_docstrings_touch_code_immediately() -> None:
    """Program AD adjoint docstrings must be adjacent to executable code."""
    source_path = Path(generation.__file__)
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
