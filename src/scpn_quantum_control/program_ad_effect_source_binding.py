# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — parsed source identity admission
"""Compare parsed effect metadata with the actual lexical source definition."""

from __future__ import annotations

import ast
from types import CodeType
from typing import cast

_AST_FIELDS: dict[int, tuple[str, ...]] = {
    id(kind): kind._fields
    for kind in vars(ast).values()
    if isinstance(kind, type) and issubclass(kind, ast.AST)
}


def _same_source_tree(actual: ast.AST, observed: ast.AST) -> bool:
    """Compare native parser fields without representing foreign metadata.

    Parameters
    ----------
    actual
        Definition parsed from the admitted actual source file.
    observed
        Definition supplied by the caller's earlier effect inspection.

    Returns
    -------
    bool
        Whether node kinds and every semantic field match. Source coordinates
        may be relative in the observed tree. The actual parser tree controls
        traversal; exact field types reject foreign equality protocols.

    """
    pending: list[tuple[object, object]] = [(actual, observed)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        fields = _AST_FIELDS.get(id(type(left)))
        if fields is not None:
            try:
                pending.extend(
                    (ast.AST.__getattribute__(left, name), ast.AST.__getattribute__(right, name))
                    for name in fields
                )
            except AttributeError:
                return False
        elif type(left) is list:
            actual_items, observed_items = cast(list[object], left), cast(list[object], right)
            if len(actual_items) != len(observed_items):
                return False
            pending.extend(zip(actual_items, observed_items, strict=True))
        elif left != right:
            return False
    return True


def _source_ast_matches(code: CodeType, observed: ast.AST, scope: ast.stmt) -> bool:
    """Locate the loaded lexical definition and bind the earlier semantic tree.

    Parameters
    ----------
    code
        Loaded code whose native metadata has already been admitted.
    observed
        Source definition that will be used for public effect analysis.
    scope
        Actual enclosing module statement parsed without executing its code.

    Returns
    -------
    bool
        Whether the function or lambda at the loaded first line has identical
        semantic fields. Decorator lines retain their original code location.

    """
    for candidate in ast.walk(scope):
        if not isinstance(candidate, ast.FunctionDef | ast.Lambda):
            continue
        start = candidate.lineno
        if isinstance(candidate, ast.FunctionDef):
            start = min([start, *(item.lineno for item in candidate.decorator_list)])
        if start == code.co_firstlineno and _same_source_tree(candidate, observed):
            return True
    return False
