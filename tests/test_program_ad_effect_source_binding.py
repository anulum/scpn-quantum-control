# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — parsed source binding contracts
"""Bind public effect inspection to the actual source used by loaded code."""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path
from types import FrameType, FunctionType
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control import whole_program_value_and_grad
from scpn_quantum_control.differentiable import (
    compile_whole_program_frontend,
    program_adjoint_replay_gradient,
)
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects


@pytest.mark.parametrize("form", ["decorated", "lambda", "docstring", "literal"])
def test_parsed_binding_preserves_lexical_forms_literals_and_decorator_refusal(
    tmp_path: Path, form: str
) -> None:
    """Keep literal contents and the existing located unsupported-decorator contract.

    Parameters
    ----------
    tmp_path
        Owned location of the actual public objective module.
    form
        Decorated root, lambda or nested root with a multiline docstring or key.

    """
    sources = {
        "decorated": (
            "def identity(function):\n    return function\n\n"
            "@identity\ndef objective(values):\n    return values[0] ** 2\n"
        ),
        "lambda": "objective = lambda values: values[0] ** 2\n",
        "docstring": (
            "def build():\n    def objective(values):\n"
            '        """Native description.\n        Preserve this original margin.\n        """\n'
            "        return values[0] ** 2\n    return objective\n"
            "objective = build()\n"
        ),
        "literal": (
            "def build():\n    def objective(values):\n"
            '        options = {"""coefficient\n        preserved""": 2.0}\n'
            '        return values[0] * options["coefficient\\n        preserved"]\n'
            "    return objective\nobjective = build()\n"
        ),
    }
    source = sources[form]
    path = tmp_path / "lexical_form.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    function = cast(FunctionType, namespace["objective"])
    baseline = active_reserved_bytes()
    if form == "decorated":
        assert find_objective_effects(function, ast.parse(source)) == ()
        report = compile_whole_program_frontend(function)
        assert not report.frontend_ready
        assert any(
            item.semantic == "decorator" for item in report.unsupported_semantic_diagnostics
        )
        with pytest.raises(ValueError, match="semantic=decorator.*line="):
            whole_program_value_and_grad(function, [3.0], trace=False)
        assert active_reserved_bytes() == baseline
        return
    assert compile_whole_program_frontend(function).frontend_ready
    result = whole_program_value_and_grad(function, [3.0], trace=False)
    expected_value, expected_gradient = (6.0, [2.0]) if form == "literal" else (9.0, [6.0])
    assert result.value == expected_value
    np.testing.assert_array_equal(result.gradient, expected_gradient)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), expected_gradient)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("entry_point", ["compiler", "runtime"])
def test_source_and_code_replacement_cannot_borrow_an_already_read_tree(
    tmp_path: Path, entry_point: str
) -> None:
    """A real file/code transition at public admission refuses before callback execution.

    Parameters
    ----------
    tmp_path
        Owned location of the actual loaded objective and callback module.
    entry_point
        Public compiler or numerical runtime which has already read old source.

    """
    source = (
        "ledger = []\n"
        "def callback(values):\n    ledger.append('executed')\n"
        "    return values[0] * 3.0\n\n"
        "def objective(values):\n    return values[0] ** 2\n"
    )
    changed = source.replace("return values[0] ** 2", "return callback(values)")
    path = tmp_path / "source_transition.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    other: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    exec(compile(changed, str(path), "exec"), other)
    function = cast(FunctionType, namespace["objective"])
    original = function.__code__
    replacement = cast(FunctionType, other["objective"]).__code__
    transitions: list[str] = []

    def observe(frame: FrameType, event: str, argument: object) -> None:
        if (
            event == "call"
            and frame.f_code is find_objective_effects.__code__
            and frame.f_locals["objective"] is function
            and not transitions
        ):
            transitions.append("source and code changed")
            path.write_text(changed, encoding="utf-8")
            function.__code__ = replacement

    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        if entry_point == "compiler":
            report = compile_whole_program_frontend(function)
            assert not report.frontend_ready
            assert any(
                item.semantic == "external_callback"
                and item.detail == "external callback source does not match captured function"
                for item in report.unsupported_semantic_diagnostics
            )
        else:
            with pytest.raises(ValueError, match="source does not match captured function.*line="):
                whole_program_value_and_grad(function, [2.0], trace=False)
    finally:
        sys.setprofile(previous)
    assert transitions == ["source and code changed"]
    assert sys.getprofile() is previous
    assert namespace["ledger"] == []
    assert active_reserved_bytes() == baseline
    function.__code__ = original
    path.write_text(source, encoding="utf-8")
    assert compile_whole_program_frontend(function).frontend_ready
    result = whole_program_value_and_grad(function, [2.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])
    assert namespace["ledger"] == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("tree_change", ["constant", "length", "kind", "missing", "foreign"])
def test_public_effect_admission_refuses_a_different_or_malformed_tree_without_protocols(
    tmp_path: Path, tree_change: str
) -> None:
    """Parsed metadata cannot substitute another computation or invoke a constant protocol.

    Parameters
    ----------
    tmp_path
        Owned actual source for the unchanged loaded numerical objective.
    tree_change
        Changed literal, statement count or kind, missing field or foreign constant.

    """
    source = "def objective(values):\n    return values[0] ** 2\n"
    path = tmp_path / "parsed_binding.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    function = cast(FunctionType, namespace["objective"])
    tree = ast.parse(source)
    definition = cast(ast.FunctionDef, tree.body[0])
    calls: list[str] = []

    class ForeignConstant:
        def __repr__(self) -> str:
            calls.append("representation")
            raise AssertionError("foreign AST constant representation was called")

        def __eq__(self, other: object) -> bool:
            calls.append("equality")
            raise AssertionError("foreign AST constant equality was called")

    if tree_change == "length":
        definition.body.append(ast.Expr(value=ast.Constant(value="extra statement")))
    elif tree_change == "kind":
        definition.body[0] = ast.Expr(value=ast.Constant(value=2))
    elif tree_change == "missing":
        del definition.body
    else:
        expression = cast(ast.BinOp, cast(ast.Return, definition.body[0]).value)
        constant = ast.Constant(value=3)
        if tree_change == "foreign":
            vars(constant)["value"] = ForeignConstant()
        expression.right = constant
    baseline = active_reserved_bytes()
    findings = find_objective_effects(function, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback source does not match captured function"
    assert calls == []
    assert active_reserved_bytes() == baseline
    assert find_objective_effects(function, ast.parse(source)) == ()
    result = whole_program_value_and_grad(function, [3.0], trace=False)
    assert result.value == 9.0
    np.testing.assert_array_equal(result.gradient, [6.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [6.0])
    assert active_reserved_bytes() == baseline
