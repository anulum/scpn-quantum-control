# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program AD contract tests
"""Fail-closed contract tests for whole-program automatic differentiation."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import whole_program_ad_result as result_module
from scpn_quantum_control.differentiable import (
    Parameter,
    ProgramADAdjointResult,
    ProgramADEffectIR,
    TraceADScalar,
    WholeProgramADResult,
    WholeProgramBytecodeInstruction,
    WholeProgramIRNode,
    WholeProgramSemanticsReport,
    WholeProgramSourceIRFeature,
    WholeProgramTraceEvent,
    compile_whole_program_frontend,
    whole_program_grad,
    whole_program_value_and_grad,
)


def _valid_whole_program_payload() -> dict[str, Any]:
    """Build a valid whole-program result payload for mutation checks."""

    return {
        "value": 1.0,
        "gradient": np.array([1.0, 0.0], dtype=np.float64),
        "method": "whole_program_ad",
        "step": 0.0,
        "evaluations": 1,
        "parameter_names": ("x", "y"),
        "trainable": (True, False),
        "trace_events": (),
        "ir_nodes": (),
        "source": None,
        "control_flow_observed": False,
        "numpy_observed": False,
        "polyglot_targets": {"python": "available"},
        "claim_boundary": "bounded claim",
    }


def test_whole_program_ad_records_ir_and_executed_branch_semantics() -> None:
    """Whole-program AD should differentiate the executed Python control-flow branch exactly."""

    def objective(values: Any) -> object:
        total = values[0] * values[0]
        for index, value in enumerate(values):
            if value > 0.0:
                total = total + np.sin(value) + index * value
            else:
                total = total + value**2
        return total

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, -0.5, 0.75], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("bias"), Parameter("phase")),
    )

    assert result.method == "whole_program_ad"
    assert result.step == 0.0
    assert result.control_flow_observed is True
    assert result.numpy_observed is True
    assert any(node.op.startswith("branch:") for node in result.ir_nodes)
    assert any(node.op == "sin" for node in result.ir_nodes)
    np.testing.assert_allclose(
        result.gradient,
        [2.0 * 0.25 + math.cos(0.25), -1.0, math.cos(0.75) + 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert result.frontend_report is not None
    assert result.frontend_report.frontend_ready is True
    assert (
        result.frontend_report.frontend_digest
        == compile_whole_program_frontend(objective).frontend_digest
    )


def test_whole_program_ad_frontend_gate_rejects_unsupported_semantics_before_execution() -> None:
    """Unsupported semantics should fail before execution with located frontend evidence."""

    calls = {"count": 0}

    def objective(values: Any) -> object:
        calls["count"] += 1
        return sum([item for item in values if item > 0.0])

    with pytest.raises(ValueError) as exc_info:
        whole_program_value_and_grad(objective, np.array([1.0, -2.0], dtype=np.float64))

    message = str(exc_info.value)
    assert calls == {"count": 0}
    assert "whole-program AD frontend execution gate rejected objective" in message
    assert "unsupported_python_semantics:filtered_comprehension" in message
    assert "semantic=filtered_comprehension" in message
    assert "line=" in message
    assert "absolute_line=" in message
    assert "regions=" in message
    assert "bytecode_offsets=" in message
    assert "frontend_digest=" in message


def test_whole_program_ad_frontend_gate_requires_source_coordinates() -> None:
    """Objectives without source coordinates should not execute under whole-program AD."""

    namespace: dict[str, Any] = {}
    exec(  # noqa: S102 - deliberate source-unavailable objective for frontend gate coverage.
        "def dynamic_objective(values):\n"
        "    dynamic_objective.calls += 1\n"
        "    return values[0] * values[0]\n",
        namespace,
    )
    dynamic_objective = namespace["dynamic_objective"]
    dynamic_objective.calls = 0

    with pytest.raises(ValueError, match="source_frontend_missing"):
        whole_program_value_and_grad(dynamic_objective, np.array([2.0], dtype=np.float64))

    assert dynamic_objective.calls == 0


def test_whole_program_grad_respects_trainable_mask_and_rejects_derivative_loss() -> None:
    """Whole-program AD should freeze masked parameters and reject float-cast derivative loss."""

    gradient = whole_program_grad(
        lambda values: values[0] ** 2 + values[1] ** 2,
        np.array([2.0, 3.0], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("frozen", trainable=False)),
        trace=False,
    )
    np.testing.assert_allclose(gradient, [4.0, 0.0], rtol=1.0e-12, atol=1.0e-12)

    with pytest.raises(ValueError, match="converted to float"):
        whole_program_value_and_grad(
            lambda values: float(values[0] ** 2),
            np.array([2.0], dtype=np.float64),
        )


def test_whole_program_ad_operator_surface_and_fail_closed_paths() -> None:
    """Whole-program AD should cover scalar operator interception and reject derivative loss."""

    def objective(values: Any) -> object:
        x, y = values
        branch = x if x >= y else y
        reverse_branch = y if y <= x else x
        return (
            np.sin(x)
            + np.cos(y)
            + np.exp(x - y)
            + np.log(x + 3.0)
            + (2.0 + x)
            + (5.0 - y)
            + (3.0 * x)
            + (12.0 / (x + 4.0))
            + (x**2.0)
            + (2.0 ** (y + 1.0))
            - branch
            + reverse_branch
            + (-y)
        )

    result = whole_program_value_and_grad(objective, [1.5, 0.25], trace=False)

    assert result.method == "whole_program_ad"
    assert result.evaluations == 1
    assert result.numpy_observed is True
    assert result.control_flow_observed is True
    assert result.gradient.shape == (2,)
    ops = {node.op.split(":", maxsplit=1)[0] for node in result.ir_nodes}
    assert {
        "parameter",
        "branch",
        "sin",
        "cos",
        "exp",
        "log",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "neg",
    }.issubset(ops)

    with pytest.raises(ValueError, match="converted to float"):
        whole_program_value_and_grad(lambda values: float(values[0]), [1.0])
    with pytest.raises(ValueError, match="denominator"):
        whole_program_value_and_grad(lambda values: values[0] / 0.0, [1.0])
    with pytest.raises(ValueError, match="log input"):
        whole_program_value_and_grad(lambda values: np.log(values[0]), [-1.0])
    with pytest.raises(ValueError, match="unsupported whole-program AD NumPy ufunc"):
        whole_program_value_and_grad(lambda values: np.arctan2(values[0], values[0]), [0.25])
    np.testing.assert_allclose(
        whole_program_grad(lambda values: np.add(values[0], values[0]), [1.0]),
        [2.0],
    )
    with pytest.raises(ValueError, match="direct NumPy scalar ufunc"):
        whole_program_value_and_grad(
            lambda values: values[0].__array_ufunc__(np.sin, "reduce", values[0]), [1.0]
        )
    with pytest.raises(ValueError, match="different traces"):
        left_context = SimpleNamespace(parameter_count=1)
        right_context = SimpleNamespace(parameter_count=1)
        left = TraceADScalar(
            1.0,
            np.array([1.0], dtype=np.float64),
            cast(Any, left_context),
            "%l",
        )
        right = TraceADScalar(
            2.0,
            np.array([1.0], dtype=np.float64),
            cast(Any, right_context),
            "%r",
        )
        _ = left + right


def test_whole_program_result_validation_fail_closed_paths() -> None:
    """Whole-program AD result contracts should reject malformed metadata."""

    assert WholeProgramADResult is result_module.WholeProgramADResult
    assert WholeProgramIRNode is result_module.WholeProgramIRNode
    assert WholeProgramTraceEvent is result_module.WholeProgramTraceEvent

    for key, value, message in (
        ("value", float("nan"), "finite"),
        ("gradient", np.array([[1.0]]), "one-dimensional"),
        ("gradient", np.array([float("nan")]), "finite"),
        ("step", cast(Any, "bad"), "real scalar"),
        ("step", float("nan"), "finite"),
        ("step", -1.0, "step"),
        ("evaluations", 0, "evaluations"),
        ("parameter_names", ("x",), "parameter_names"),
        ("trainable", (True,), "trainable"),
        ("trainable", (True, "no"), "booleans"),
        ("trace_events", (object(),), "trace_events"),
        ("ir_nodes", (object(),), "ir_nodes"),
        ("bytecode_instructions", (object(),), "bytecode_instructions"),
        ("source_ir_features", (object(),), "source_ir_features"),
        ("semantics_report", object(), "semantics_report"),
        ("program_ir", object(), "program_ir"),
        ("adjoint_result", object(), "adjoint_result"),
        ("frontend_report", object(), "frontend_report"),
        ("control_flow_observed", "yes", "control_flow_observed"),
        ("numpy_observed", "yes", "numpy_observed"),
        ("polyglot_targets", {}, "polyglot_targets"),
        ("polyglot_targets", {"": "available"}, "polyglot target"),
        ("polyglot_targets", {"python": ""}, "polyglot target"),
        ("claim_boundary", "", "claim_boundary"),
    ):
        payload = _valid_whole_program_payload()
        payload[key] = value
        with pytest.raises(ValueError, match=message):
            WholeProgramADResult(**payload)

    payload = _valid_whole_program_payload()
    payload["gradient"] = np.array([1.0, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="non-trainable parameters"):
        WholeProgramADResult(**payload)

    payload = _valid_whole_program_payload()
    payload["adjoint_result"] = ProgramADAdjointResult(
        gradient=np.array([1.0], dtype=np.float64),
        supported=True,
        unsupported_ops=(),
        method="program_adjoint_replay",
        claim_boundary="bounded adjoint",
    )
    with pytest.raises(ValueError, match="gradient shape"):
        WholeProgramADResult(**payload)

    payload = _valid_whole_program_payload()
    payload["adjoint_result"] = ProgramADAdjointResult(
        gradient=np.array([1.0, 0.25], dtype=np.float64),
        supported=True,
        unsupported_ops=(),
        method="program_adjoint_replay",
        claim_boundary="bounded adjoint",
    )
    with pytest.raises(ValueError, match="non-trainable parameters"):
        WholeProgramADResult(**payload)

    payload = _valid_whole_program_payload()
    payload["bytecode_instructions"] = (
        WholeProgramBytecodeInstruction(
            offset=0,
            opname="LOAD_FAST",
            argrepr="x",
            line_number=1,
        ),
    )
    payload["source_ir_features"] = (
        WholeProgramSourceIRFeature(kind="call", detail="np.sin", line_number=1),
    )
    payload["semantics_report"] = WholeProgramSemanticsReport(
        bytecode_frontend=True,
        source_frontend=True,
        graph_capture=True,
        aliasing_observed=False,
        mutation_observed=False,
        loop_observed=False,
        control_flow_observed=False,
        numpy_observed=True,
        differentiation_semantics="operator_intercepted",
        accepted_python_semantics=("plain_return",),
        unsupported_python_semantics=(),
    )
    payload["program_ir"] = ProgramADEffectIR(
        ssa_values=(),
        effects=(),
        alias_edges=(),
        control_regions=(),
        serialization='{"format":"program_ad_effect_ir.v1"}',
    )
    valid = WholeProgramADResult(**payload)
    assert valid.bytecode_instructions
    assert valid.source_ir_features
    assert valid.semantics_report is payload["semantics_report"]
    assert valid.program_ir is payload["program_ir"]


def test_whole_program_event_and_ir_node_validation_paths() -> None:
    """Trace event and IR node records should reject malformed trace metadata."""

    event = WholeProgramTraceEvent("objective.py", "loss", 7, "  return x  ")
    assert event.source == "return x"

    valid_tangent = np.array([1.0, 0.0], dtype=np.float64)
    node = WholeProgramIRNode(0, "parameter", ("x",), 1.5, valid_tangent)
    assert node.value == pytest.approx(1.5)
    np.testing.assert_allclose(node.tangent, valid_tangent)

    for kwargs, message in (
        ({"filename": "", "function_name": "loss", "line_number": 1, "source": ""}, "filename"),
        (
            {"filename": "objective.py", "function_name": "", "line_number": 1, "source": ""},
            "function_name",
        ),
        (
            {"filename": "objective.py", "function_name": "loss", "line_number": 0, "source": ""},
            "line_number",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            WholeProgramTraceEvent(**cast(Any, kwargs))

    for args, message in (
        ((-1, "parameter", ("x",), 1.0, valid_tangent), "index"),
        ((0, "", ("x",), 1.0, valid_tangent), "op"),
        ((0, "parameter", ("",), 1.0, valid_tangent), "inputs"),
        ((0, "parameter", ("x",), float("inf"), valid_tangent), "finite"),
        ((0, "parameter", ("x",), 1.0, np.array([[1.0]])), "one-dimensional"),
        ((0, "parameter", ("x",), 1.0, np.array([float("nan")])), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            WholeProgramIRNode(*args)


def test_whole_program_ad_rejects_unsupported_power_and_return_contracts() -> None:
    """Whole-program AD should fail closed for unsupported powers and non-scalar returns."""

    with pytest.raises(ValueError, match="positive base"):
        whole_program_value_and_grad(lambda values: (-values[0]) ** values[0], [1.0])
    with pytest.raises(ValueError, match="whole-program AD scalar"):
        whole_program_value_and_grad(lambda values: np.array([values[0]], dtype=object), [1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        TraceADScalar(
            1.0,
            np.array([[1.0]], dtype=np.float64),
            cast(Any, SimpleNamespace(parameter_count=1)),
            "%bad",
        )
