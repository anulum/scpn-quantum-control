# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native container call binding contracts
"""Qualify located container signatures through real compiler and native replay."""

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
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects


@pytest.mark.parametrize(
    ("kind", "call"),
    [
        ("list", "append()"),
        ("list", "append(1.0, 2.0)"),
        ("list", "extend()"),
        ("list", "extend([1.0], [2.0])"),
        ("list", "insert(0)"),
        ("list", "insert(0, 1.0, 2.0)"),
        ("list", "insert(index=0, object=1.0)"),
        ("list", "remove()"),
        ("list", "remove(1.0, 2.0)"),
        ("list", "pop(0, 1)"),
        ("list", "pop(index=0)"),
        ("list", "clear(1)"),
        ("list", "reverse(1)"),
        ("list", "sort(1)"),
        ("list", "sort(unregistered=True)"),
        ("list", "update()"),
        ("dict", "append(1.0)"),
        ("dict", "pop()"),
        ("dict", "pop('one', 2.0, 3.0)"),
        ("dict", "pop(key='one')"),
        ("dict", "clear(1)"),
        ("dict", "update({}, {})"),
        ("list", "append(*values, 1.0, 2.0)"),
        ("list", "insert(*values, 0, 1.0, 2.0)"),
        ("list", "pop(*values, 0, 1)"),
        ("list", "sort(*values, 1)"),
        ("dict", "update(*values, {}, {})"),
        ("list", "append(*values, *(1.0, 2.0))"),
        ("list", "insert(*values, *(), *(0, 1.0, 2.0))"),
        ("list", "clear(*values, *[1])"),
        ("dict", "pop(*values, *('one', 2.0, 3.0))"),
        ("dict", "update(*values, *({}, {}))"),
        ("list", "get('one')"),
        ("list", "copy(1)"),
        ("dict", "get()"),
        ("dict", "get('one', 2.0, 3.0)"),
        ("dict", "get(key='one')"),
        ("dict", "copy(1)"),
        ("tuple", "append(1.0)"),
        ("tuple", "insert(0)"),
        ("tuple", "pop()"),
        ("tuple", "reverse()"),
        ("tuple", "copy()"),
        ("tuple", "mean()"),
        ("generator", "append(1.0)"),
        ("generator", "insert(0, 1.0)"),
        ("generator", "copy()"),
    ],
)
def test_invalid_container_signature_refuses_before_execution_and_recovers(
    tmp_path: Path, kind: str, call: str
) -> None:
    """Malformed local calls refuse with a location and recover real native gradients.

    Parameters
    ----------
    tmp_path
        Owned path for the actual loaded module and its restored valid source.
    kind
        Native list, dictionary, tuple or generator owned by the objective.
    call
        Actual invalid native-method syntax, without protocol substitution.

    """
    container = {
        "list": "[1.0, 2.0]",
        "dict": "{'one': 1.0}",
        "tuple": "(1.0, 2.0)",
        "generator": "(item for item in (1.0, 2.0))",
    }[kind]
    binding = "_result = " if call.startswith("get(") else ""
    source = (
        "def objective(values):\n"
        f"    storage = {container}\n"
        f"    {binding}storage.{call}\n"
        "    return values[0] * 3.0\n"
    )
    path = tmp_path / "container_objective.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    findings = find_objective_effects(objective, ast.parse(source))
    assert any(
        finding.semantic == "external_callback"
        and finding.detail == "local container call signature is unsupported"
        and getattr(finding.node, "lineno", 0) == 3
        for finding in findings
    )
    inputs = np.array([2.0])
    baseline = active_reserved_bytes()
    calls: list[str] = []
    previous = sys.getprofile()

    def observe(frame: FrameType, event: str, argument: object) -> None:
        if event == "call" and frame.f_code is objective.__code__:
            calls.append("objective")

    try:
        sys.setprofile(observe)
        with pytest.raises(
            ValueError,
            match="local container call signature is unsupported.*line=.*absolute_line=",
        ):
            whole_program_value_and_grad(objective, inputs, trace=False)
        assert calls == []
        assert active_reserved_bytes() == baseline
        restored = "def objective(values):\n    return values[0] * 3.0\n"
        path.write_text(restored, encoding="utf-8")
        exec(compile(restored, str(path), "exec"), namespace)
        objective = cast(FunctionType, namespace["objective"])
        result = whole_program_value_and_grad(objective, inputs, trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
        assert calls == ["objective"]
    finally:
        sys.setprofile(previous)
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline
    assert sys.getprofile() is previous


@pytest.mark.parametrize(
    ("body", "gradient"),
    [
        ("storage = [1.0]\n    storage.append(2.0)\n    coefficient = sum(storage)", 3.0),
        ("storage = [1.0]\n    storage.extend([2.0])\n    coefficient = sum(storage)", 3.0),
        ("storage = [1.0]\n    storage.insert(0, 2.0)\n    coefficient = sum(storage)", 3.0),
        (
            "storage = [1.0, 2.0, 5.0]\n    storage.remove(5.0)\n    coefficient = sum(storage)",
            3.0,
        ),
        ("storage = [1.0, 2.0, 5.0]\n    storage.pop()\n    coefficient = sum(storage)", 3.0),
        ("storage = [1.0, 2.0]\n    storage.reverse()\n    coefficient = sum(storage)", 3.0),
        (
            "storage = [1.0, 2.0]\n    storage.sort(reverse=True)\n    coefficient = sum(storage)",
            3.0,
        ),
        (
            "storage = [1.0]\n    storage.clear()\n    storage.append(3.0)\n    coefficient = sum(storage)",
            3.0,
        ),
        (
            "storage = {'one': 1.0}\n    storage.update(two=2.0)\n    coefficient = storage['one'] + storage['two']",
            3.0,
        ),
        (
            "storage = {'three': 3.0}\n    storage.pop('missing', 2.0)\n    coefficient = storage['three']",
            3.0,
        ),
        (
            "storage = {'one': 1.0}\n    storage.clear()\n    storage.update(three=3.0)\n    coefficient = storage['three']",
            3.0,
        ),
        ("storage = [1.0]\n    storage.append(*values)\n    coefficient = sum(storage)", 5.0),
        ("storage = [1.0]\n    storage.extend(values)\n    coefficient = sum(storage)", 5.0),
        ("storage = [1.0]\n    storage.insert(0, *values)\n    coefficient = sum(storage)", 5.0),
        (
            "storage = [1.0]\n    storage.append(*values, *())\n    coefficient = sum(storage)",
            5.0,
        ),
        (
            "storage = [1.0]\n    storage.insert(0, *values, *())\n    coefficient = sum(storage)",
            5.0,
        ),
        (
            "storage = {'one': 1.0}\n    storage.update(**{'two': values[0]})\n    coefficient = storage['one'] + storage['two']",
            5.0,
        ),
        ("storage = [1.0, 2.0]\n    copied = storage.copy()\n    coefficient = sum(copied)", 3.0),
        (
            "storage = {'three': 3.0}\n    copied = storage.copy()\n    coefficient = copied['three']",
            3.0,
        ),
        ("storage = {'three': 3.0}\n    coefficient = storage.get('three')", 3.0),
        ("storage = {}\n    coefficient = storage.get('missing', 3.0)", 3.0),
        ("storage = (item for item in (1.0, 2.0))\n    coefficient = sum(storage)", 3.0),
        (
            "storage = (item for item in (1.0, 2.0))\n    copied = list(storage)\n    coefficient = sum(copied)",
            3.0,
        ),
    ],
)
def test_valid_container_signatures_preserve_ordinary_and_expanded_derivatives(
    tmp_path: Path, body: str, gradient: float
) -> None:
    """Supported signatures and unknown-arity expansion retain analytic native replay.

    Parameters
    ----------
    tmp_path
        Actual source-visible module path for the selected storage operation.
    body
        Ordinary native operation on objective-owned storage.
    gradient
        Independent derivative for the resulting scalar coefficient expression.

    """
    source = f"def objective(values):\n    {body}\n    return values[0] * coefficient\n"
    path = tmp_path / "valid_container_objective.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(FunctionType, namespace["objective"])
    assert find_objective_effects(objective, ast.parse(source)) == ()
    baseline = active_reserved_bytes()
    inputs = np.array([2.0])
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [gradient])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [gradient])
    np.testing.assert_array_equal(inputs, [2.0])
    assert active_reserved_bytes() == baseline
