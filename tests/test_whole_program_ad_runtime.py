# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program AD runtime tests
# scpn-quantum-control -- whole-program AD runtime contracts
"""Runtime contracts for whole-program automatic differentiation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import Parameter, whole_program_value_and_grad

FloatArray = NDArray[np.float64]


def test_whole_program_value_and_grad_traces_numpy_control_flow() -> None:
    """Whole-program AD should handle ordinary Python control flow and NumPy calls."""

    def objective(values: Any) -> object:
        total = values[0] * 0.0
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
    assert result.control_flow_observed is True
    assert result.numpy_observed is True
    assert result.polyglot_targets["python"].startswith("operator-intercepted")
    assert result.polyglot_targets["rust"].startswith("blocked")
    assert result.polyglot_targets["llvm"].startswith("blocked")
    assert len(result.trace_events) >= 4
    np.testing.assert_allclose(
        result.gradient,
        [math.cos(0.25), -1.0, math.cos(0.75) + 2.0],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_whole_program_ad_captures_bytecode_source_alias_mutation_and_loop_semantics() -> None:
    """Arbitrary whole-program AD should expose frontend IR and semantic analysis."""

    def objective(values: Any) -> object:
        history = [values[0]]
        alias = history
        total = values[0] * 0.0
        for item in range(3):
            alias.append(values[1] * item)
            total = total + history[item]
        if total > 0.0:
            return np.sin(values[0]) + total
        return np.cos(values[1]) - total

    result = whole_program_value_and_grad(
        objective,
        np.array([0.5, 0.25], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("phi")),
    )

    assert result.semantics_report is not None
    assert result.semantics_report.bytecode_frontend is True
    assert result.semantics_report.source_frontend is True
    assert result.semantics_report.graph_capture is True
    assert result.semantics_report.aliasing_observed is True
    assert result.semantics_report.mutation_observed is True
    assert result.semantics_report.loop_observed is True
    assert result.semantics_report.control_flow_observed is True
    assert result.semantics_report.numpy_observed is True
    assert result.bytecode_instructions
    assert any(instruction.opname == "FOR_ITER" for instruction in result.bytecode_instructions)
    assert {feature.kind for feature in result.source_ir_features} >= {
        "alias_analysis",
        "control_flow",
        "loop",
        "mutation",
        "numpy",
    }
    assert any(node.op.startswith("branch:") for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, [math.cos(0.5) + 1.0, 1.0], atol=1.0e-12)


def test_whole_program_ad_reports_accepted_python_calling_semantics() -> None:
    """Whole-program AD should expose accepted closure/default/kwargs semantics."""

    scale = 2.5

    def objective(
        values: Any,
        bias: float = 0.25,
        **metadata: float,
    ) -> object:
        return sum(item for item in (scale * values[0], bias, metadata.get("offset", 0.0)))

    result = whole_program_value_and_grad(
        objective,
        np.array([3.0], dtype=np.float64),
        parameters=(Parameter("theta"),),
    )

    assert result.semantics_report is not None
    assert result.semantics_report.unsupported_python_semantics == ()
    assert set(result.semantics_report.accepted_python_semantics) >= {
        "closure",
        "default_argument",
        "generator_expression",
        "var_keyword_parameter",
    }
    assert any(
        feature.kind == "python_semantics" and feature.detail == "closure"
        for feature in result.source_ir_features
    )
    np.testing.assert_allclose(result.gradient, [scale], atol=1.0e-12)


def test_whole_program_ad_accepts_bounded_list_comprehension_semantics() -> None:
    """Plain list comprehensions should preserve derivative-carrying values."""

    def objective(values: Any) -> object:
        terms = [item * item + np.sin(item) for item in values]
        return sum(terms)

    values = np.array([0.25, -0.5, 1.25], dtype=np.float64)
    result = whole_program_value_and_grad(objective, values)

    assert result.semantics_report is not None
    assert result.semantics_report.unsupported_python_semantics == ()
    assert "list_comprehension" in result.semantics_report.accepted_python_semantics
    assert any(
        feature.kind == "python_semantics" and feature.detail == "list_comprehension"
        for feature in result.source_ir_features
    )
    np.testing.assert_allclose(result.gradient, 2.0 * values + np.cos(values), atol=1.0e-12)


def test_whole_program_ad_fails_closed_for_unsupported_python_semantics() -> None:
    """Unsupported Python constructs should be rejected before objective execution."""

    @dataclass(frozen=True)
    class ScaleHolder:
        scale: float

    holder = ScaleHolder(scale=2.0)

    def filtered_comprehension_objective(values: FloatArray) -> object:
        return sum([item for item in values if item > 0.0])

    def set_comprehension_objective(values: FloatArray) -> object:
        return sum({item for item in values})

    def dict_comprehension_objective(values: FloatArray) -> object:
        return sum({index: item for index, item in enumerate(values)}.values())

    def generator_objective(values: FloatArray) -> object:
        yield values[0]

    def context_manager_objective(values: FloatArray) -> object:
        with pytest.raises(RuntimeError):
            raise RuntimeError("sentinel")
        return values[0]

    def exception_objective(values: FloatArray) -> object:
        try:
            return values[0]
        except RuntimeError:
            return values[0] * 0.0

    def recursive_objective(values: FloatArray) -> object:
        if values[0] <= 0.0:
            return values[0]
        return recursive_objective(values - 1.0)

    def object_attribute_objective(values: FloatArray) -> object:
        return holder.scale * values[0]

    def passthrough(function: Callable[[FloatArray], object]) -> Callable[[FloatArray], object]:
        return function

    @passthrough
    def decorated_objective(values: FloatArray) -> object:
        return values[0]

    for objective, diagnostic in (
        (filtered_comprehension_objective, "filtered_comprehension"),
        (set_comprehension_objective, "set_or_dict_comprehension"),
        (dict_comprehension_objective, "set_or_dict_comprehension"),
        (generator_objective, "generator"),
        (context_manager_objective, "context_manager"),
        (exception_objective, "exception_control_flow"),
        (recursive_objective, "recursion"),
        (object_attribute_objective, "object_attribute"),
        (decorated_objective, "decorator"),
    ):
        with pytest.raises(ValueError, match=diagnostic):
            whole_program_value_and_grad(objective, np.array([1.0], dtype=np.float64))


def test_whole_program_ad_handles_vector_numpy_reductions_dot_and_array_mutation() -> None:
    """Whole-program AD should execute vector NumPy semantics with derivative-carrying arrays."""

    def objective(values: Any) -> object:
        working = values.copy()
        working[1] = working[1] + values[0] * 2.0
        vector_term = np.sum(np.sin(working) + working**2)
        mean_term = np.mean(working)
        dot_term = np.dot(working, np.array([1.0, -2.0, 0.5], dtype=np.float64))
        return vector_term + mean_term + dot_term

    result = whole_program_value_and_grad(
        objective,
        np.array([0.2, -0.4, 0.7], dtype=np.float64),
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )

    working = np.array([0.2, 0.0, 0.7], dtype=np.float64)
    base_grad = np.cos(working) + 2.0 * working + np.array([1.0, -2.0, 0.5]) + (1.0 / 3.0)
    expected = np.array(
        [base_grad[0] + 2.0 * base_grad[1], base_grad[1], base_grad[2]],
        dtype=np.float64,
    )
    assert result.method == "whole_program_ad"
    assert any(node.op == "mutation:setitem" for node in result.ir_nodes)
    assert any(node.op == "sin" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)


def test_whole_program_ad_handles_piecewise_vector_numpy_semantics() -> None:
    """Whole-program AD should differentiate executed vector piecewise NumPy branches."""

    def objective(values: Any) -> object:
        shifted = values + np.array([2.0, 3.0, 4.0], dtype=np.float64)
        smooth = np.sqrt(shifted) + np.tanh(values) + np.square(values)
        piecewise = np.where(values > 0.0, smooth, np.absolute(values - 1.0))
        clipped = np.maximum(piecewise, values + 0.5)
        return np.sum(np.minimum(clipped, piecewise + 2.0))

    values = np.array([0.25, -0.5, 1.2], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )

    expected = np.array(
        [
            0.5 / math.sqrt(2.25) + (1.0 - math.tanh(0.25) ** 2) + 0.5,
            -1.0,
            0.5 / math.sqrt(5.2) + (1.0 - math.tanh(1.2) ** 2) + 2.4,
        ],
        dtype=np.float64,
    )
    assert any(node.op == "where" for node in result.ir_nodes)
    assert any(node.op == "maximum" for node in result.ir_nodes)
    assert any(node.op == "minimum" for node in result.ir_nodes)
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)
