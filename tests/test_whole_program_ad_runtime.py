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

import importlib.util
import json
import linecache
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic
from types import FrameType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import (
    Parameter,
    whole_program_grad,
    whole_program_value_and_grad,
)
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
)

FloatArray = NDArray[np.float64]


def test_ad_initial_tangent_storage_refuses_before_objective() -> None:
    """Reject the initial parameter basis under a cap too small for its live storage."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    with pytest.raises(DenseAllocationError, match="execution memory"):
        whole_program_value_and_grad(objective, np.ones(64), trace=False, max_execution_gib=1e-6)


def test_ad_retained_tape_growth_refuses_under_explicit_cap(tmp_path: Path) -> None:
    """A valid initial basis cannot authorize unlimited retained operation tangents."""
    path = tmp_path / "tape_growth.py"
    path.write_text(
        "def objective(values):\n"
        "    total = values[0]\n"
        "    for _ in range(500):\n"
        "        total = total + values[0]\n"
        "    return total\n"
    )
    spec = importlib.util.spec_from_file_location("tape_growth", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(DenseAllocationError, match="adjoint tape"):
        whole_program_value_and_grad(
            module.objective, [1.0], trace=False, max_execution_gib=256 * 1024 / 1024**3
        )


def test_ad_bounded_tape_preserves_real_value_and_derivative() -> None:
    """A bounded real trace retains its analytic derivative under explicit admission."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    result = whole_program_value_and_grad(
        objective, [2.0, 3.0], trace=False, max_execution_gib=0.001
    )
    assert result.value == 7.0
    np.testing.assert_array_equal(result.gradient, [4.0, 1.0])


def test_whole_program_scope_releases_on_deadline_cancel_and_success() -> None:
    """Real public AD refuses interrupted entry and disposes completed numeric tape charge."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ExecutionCancelledError):
        whole_program_value_and_grad(objective, [2.0], trace=False, cancelled=cancelled)
    with pytest.raises(TimeoutError):
        whole_program_value_and_grad(
            objective, [2.0], trace=False, deadline_monotonic=monotonic() - 1
        )
    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


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


def test_gradient_only_entry_preserves_resource_and_lifecycle_policy() -> None:
    """The gradient facade cannot bypass the actual owned AD admission scope."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError, match="execution memory"):
        whole_program_grad(objective, [2.0], trace=False, max_execution_gib=1 / 1024**3)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ExecutionCancelledError):
        whole_program_grad(objective, [2.0], trace=False, cancelled=cancelled)
    with pytest.raises(TimeoutError):
        whole_program_grad(objective, [2.0], trace=False, deadline_monotonic=monotonic() - 1)
    np.testing.assert_array_equal(
        whole_program_grad(objective, [2.0], trace=False, max_execution_gib=0.001),
        [4.0],
    )
    assert active_reserved_bytes() == baseline


def test_alias_metadata_growth_cannot_bypass_tape_admission(tmp_path: Path) -> None:
    """Real repeated array views refuse metadata growth without new tangent nodes."""
    path = tmp_path / "alias_growth.py"
    path.write_text(
        "def objective(values):\n"
        "    working = values\n"
        "    for _ in range(500):\n"
        "        working = working.reshape(1)\n"
        "    return working[0] * working[0]\n"
    )
    spec = importlib.util.spec_from_file_location("alias_growth", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError, match="adjoint tape"):
        whole_program_value_and_grad(
            module.objective, [2.0], trace=False, max_execution_gib=256 * 1024 / 1024**3
        )
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        module.objective, [2.0], trace=False, max_execution_gib=0.01
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


def test_program_ir_encoding_preserves_canonical_public_result_wire() -> None:
    """Actual admitted IR encoding retains compact sorted-key JSON and analytic AD."""

    def objective(values: Any) -> object:
        working = values.reshape(1)
        return working[0] * working[0]

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [2.0], trace=False, max_execution_gib=0.01)
    assert result.program_ir is not None
    serialization = result.program_ir.serialization
    assert serialization == json.dumps(
        json.loads(serialization), sort_keys=True, separators=(",", ":")
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


def test_runtime_trace_restores_real_tracer_and_preserves_ad_result() -> None:
    """The actual public trace path restores prior instrumentation and disposes its charge."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    observed: list[str] = []

    def observer(frame: FrameType, event: str, arg: object) -> Any:
        del arg
        if frame.f_code is objective.__code__:
            observed.append(event)
        return observer

    previous = sys.gettrace()
    baseline = active_reserved_bytes()
    sys.settrace(observer)
    try:
        result = whole_program_value_and_grad(
            objective, [2.0, 3.0], trace=True, max_execution_gib=0.01
        )
        assert sys.gettrace() is observer
    finally:
        sys.settrace(previous)
    assert observed
    assert result.trace_events
    assert result.value == 7.0
    np.testing.assert_array_equal(result.gradient, [4.0, 1.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("refusal", ["memory", "cancelled", "deadline"])
def test_public_ad_refuses_before_loading_objective_source(tmp_path: Path, refusal: str) -> None:
    """An interrupted or inadmissible public call never populates its source cache."""
    source_path = tmp_path / "bounded_ad_objective.py"
    source_path.write_text("def objective(values):\n    return values[0] * values[0]\n")
    spec = importlib.util.spec_from_file_location("bounded_ad_objective", source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filename = str(source_path)
    assert filename not in linecache.cache
    baseline = active_reserved_bytes()
    cancelled = Event()
    if refusal == "cancelled":
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            whole_program_value_and_grad(module.objective, [2.0], cancelled=cancelled)
    elif refusal == "deadline":
        with pytest.raises(TimeoutError):
            whole_program_value_and_grad(
                module.objective, [2.0], deadline_monotonic=monotonic() - 1
            )
    else:
        with pytest.raises(DenseAllocationError):
            whole_program_value_and_grad(module.objective, [2.0], max_execution_gib=1e-9)
    assert filename not in linecache.cache
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        module.objective, [2.0], trace=False, max_execution_gib=0.01
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert filename not in linecache.cache
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("kind", ["float32", "integer", "strided", "tuple", "range"])
def test_public_ad_bounded_input_conversion_preserves_gradient(kind: str) -> None:
    """Real numeric input forms retain the analytic result and caller data unchanged."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    if kind == "float32":
        values: Any = np.array([2.0, 3.0], dtype=np.float32)
    elif kind == "integer":
        values = np.array([2, 3], dtype=np.int32)
    elif kind == "strided":
        values = np.array([2.0, 99.0, 3.0, 99.0])[::2]
    elif kind == "range":
        values = range(2, 4)
    else:
        values = (2.0, 3.0)
    original = np.asarray(values).copy()
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(objective, values, max_execution_gib=1e-9)
    result = whole_program_value_and_grad(objective, values, trace=False, max_execution_gib=0.01)
    assert result.value == 7.0
    np.testing.assert_array_equal(result.gradient, [4.0, 1.0])
    np.testing.assert_array_equal(np.asarray(values), original)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("values", [[True], [[1.0]], [1.0, float("nan")], [1.0, float("inf")]])
def test_public_ad_rejects_malformed_input_and_disposes_conversion(values: Any) -> None:
    """Actual input refusal leaves the shared ledger ready for a valid next call."""

    def objective(parameters: Any) -> object:
        return parameters[0] * parameters[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="parameters"):
        whole_program_value_and_grad(objective, values, max_execution_gib=0.01)
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [2.0], trace=False, max_execution_gib=0.01)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


def test_public_ad_absurd_parameter_range_refuses_before_materialisation() -> None:
    """Native-unaddressable metadata refuses without enumerating or converting a range."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError, match="native addressability"):
        whole_program_value_and_grad(objective, range(1 << 200), trace=False)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("dtype", ["complex128", "object", "bool"])
def test_public_ad_undeclared_input_dtype_refuses_without_coercion(dtype: str) -> None:
    """Native arrays with unsupported scalar storage cannot enter the conversion scope."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    values = np.array([1, 2], dtype=dtype)
    original = values.copy()
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="real numeric scalars"):
        whole_program_value_and_grad(objective, values, trace=False)
    np.testing.assert_array_equal(values, original)
    assert active_reserved_bytes() == baseline


def test_public_ad_large_source_file_refuses_before_source_cache_loading(tmp_path: Path) -> None:
    """An actual oversized source file is refused before inspect fills its global cache."""
    path = tmp_path / "oversized_objective_source.py"
    path.write_text(
        "#" + "x" * (2 * 1024**2) + "\ndef objective(values):\n    return values[0] * values[0]\n"
    )
    spec = importlib.util.spec_from_file_location("oversized_objective_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = active_reserved_bytes()
    assert str(path) not in linecache.cache
    with pytest.raises(DenseAllocationError, match="execution memory"):
        whole_program_value_and_grad(module.objective, [2.0], trace=False, max_execution_gib=0.001)
    assert str(path) not in linecache.cache
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        module.objective, [2.0], trace=False, max_execution_gib=0.1
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("refusal", ["memory", "cancelled", "deadline"])
def test_public_ad_checks_entry_policy_before_sequence_scalar_validation(refusal: str) -> None:
    """Real entry policy refuses before an invalid sequence is traversed or copied."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    values = cast(ArrayLike, [None] * 32)
    baseline = active_reserved_bytes()
    if refusal == "memory":
        with pytest.raises(DenseAllocationError):
            whole_program_value_and_grad(objective, values, max_execution_gib=1e-9)
    elif refusal == "cancelled":
        cancelled = Event()
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            whole_program_value_and_grad(objective, values, cancelled=cancelled)
    else:
        with pytest.raises(TimeoutError):
            whole_program_value_and_grad(objective, values, deadline_monotonic=monotonic() - 1)
    assert active_reserved_bytes() == baseline
    with pytest.raises(ValueError, match="real numeric scalars"):
        whole_program_value_and_grad(objective, values, max_execution_gib=0.01)
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        objective, [np.longdouble(2.0)], trace=False, max_execution_gib=0.01
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


def test_public_ad_retains_source_charge_during_objective_execution(tmp_path: Path) -> None:
    """Real AD keeps source data charged throughout its objective and disposes on return."""
    path = tmp_path / "retained_objective_source.py"
    path.write_text(
        "#" + "x" * (2 * 1024**2) + "\n"
        "from scpn_quantum_control.execution_reservations import active_reserved_bytes\n"
        "def objective(values):\n"
        "    return values[0] * 0.0 + active_reserved_bytes()\n"
    )
    spec = importlib.util.spec_from_file_location("retained_objective_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(
        module.objective, [2.0], trace=False, max_execution_gib=0.1
    )
    assert result.value >= path.stat().st_size
    np.testing.assert_array_equal(result.gradient, [0.0])
    assert active_reserved_bytes() == baseline


def test_public_ad_parameter_metadata_preserves_names_and_trainability() -> None:
    """The bounded public metadata snapshot retains names and the trainable mask."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    baseline = active_reserved_bytes()
    for parameters in (
        [Parameter("alpha"), Parameter("β", False)],
        (Parameter("alpha"), Parameter("β", False)),
    ):
        result = whole_program_value_and_grad(
            objective, [2.0, 3.0], parameters=parameters, trace=False
        )
        assert result.value == 7.0
        assert result.parameter_names == ("alpha", "β")
        assert result.trainable == (True, False)
        np.testing.assert_array_equal(result.gradient, [4.0, 0.0])
        assert active_reserved_bytes() == baseline


def test_public_ad_parameter_metadata_refuses_opaque_iteration_without_calling_it() -> None:
    """A metadata protocol cannot allocate before its public admission boundary."""

    class Opaque:
        observed = False

        def __iter__(self) -> object:
            self.observed = True
            raise AssertionError("opaque metadata must not be iterated")

        def __len__(self) -> int:
            self.observed = True
            raise AssertionError("opaque metadata length must not be queried")

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    metadata = Opaque()
    with pytest.raises(ValueError, match="plain list or tuple"):
        whole_program_value_and_grad(
            objective, [2.0], parameters=cast(Sequence[Parameter], metadata), trace=False
        )
    assert not metadata.observed
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        objective, [2.0], parameters=[Parameter("x")], trace=False
    )
    assert result.value == 4.0
    assert result.parameter_names == ("x",)
    np.testing.assert_array_equal(result.gradient, [4.0])


def test_public_ad_parameter_metadata_name_storage_refuses_and_recovers() -> None:
    """Long existing names must fit the live metadata budget before copying."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError):
        whole_program_value_and_grad(
            objective,
            [2.0],
            parameters=[Parameter("x" * 1024**2)],
            trace=False,
            max_execution_gib=0.0005,
        )
    assert active_reserved_bytes() == baseline
    with pytest.raises(ValueError, match="length"):
        whole_program_value_and_grad(objective, [2.0], parameters=[], trace=False)
    with pytest.raises(ValueError, match="unique"):
        whole_program_value_and_grad(
            objective, [2.0, 3.0], parameters=[Parameter("x"), Parameter("x")], trace=False
        )
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        objective, [2.0], parameters=[Parameter("x")], trace=False
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("grow", [False, True])
def test_public_ad_parameter_metadata_detects_list_change_and_recovers(grow: bool) -> None:
    """Mutation during actual snapshot construction cannot change admitted length.

    Parameters
    ----------
    grow
        Grow the source list rather than remove its last entry.

    """

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    baseline = active_reserved_bytes()
    parameters = [Parameter("x"), Parameter("y")]
    previous = sys.getprofile()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if observed or event != "call" or frame.f_code.co_name != "__post_init__":
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == "_bounded_parameter_metadata":
                observed = True
                if grow:
                    parameters.append(Parameter("extra"))
                else:
                    parameters.pop()
                return
            caller = caller.f_back

    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="metadata changed after admission"):
            whole_program_value_and_grad(objective, [2.0, 3.0], parameters=parameters, trace=False)
    finally:
        sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        objective, [2.0, 3.0], parameters=[Parameter("x"), Parameter("y")], trace=False
    )
    assert result.value == 7.0
    np.testing.assert_array_equal(result.gradient, [4.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_public_ad_parameter_metadata_refuses_record_and_name_subclasses() -> None:
    """Opaque record or string subclasses do not enter bounded metadata copying."""

    class Record(Parameter):
        pass

    class Name(str):
        pass

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    for parameter in (Record("x"), Parameter(Name("x"))):
        with pytest.raises(ValueError, match="exact Parameter|plain names"):
            whole_program_value_and_grad(objective, [2.0], parameters=[parameter], trace=False)
        assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(
        objective, [2.0], parameters=[Parameter("x")], trace=False
    )
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "invalid", ["array_subclass", "rank", "dtype", "sequence_subclass", "opaque"]
)
def test_public_ad_refuses_unbounded_parameter_protocol_and_recovers(invalid: str) -> None:
    """Unsupported storage protocols never enter conversion or retain a charge."""

    class ArraySubclass(np.ndarray[Any, Any]):
        """Expose a non-builtin array protocol at the public admission boundary."""

    class SequenceSubclass(list[float]):
        """Expose a non-builtin sequence at the public admission boundary."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    values: Any
    if invalid == "array_subclass":
        values = np.array([2.0]).view(ArraySubclass)
    elif invalid == "rank":
        values = np.array([[2.0]])
    elif invalid == "dtype":
        values = np.array([True])
    elif invalid == "sequence_subclass":
        values = SequenceSubclass([2.0])
    else:
        values = iter([2.0])
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="parameter"):
        whole_program_value_and_grad(objective, values, trace=False)
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("fault", ["array_shape", "initial_growth", "late_growth", "late_shrink"])
def test_public_ad_refuses_parameter_storage_changed_after_admission(fault: str) -> None:
    """Actual storage mutation at copy checkpoints cannot exceed the admitted input."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    values: Any = np.array([2.0, 3.0]) if fault == "array_shape" else [2.0, 3.0]
    observed: list[str] = []
    baseline = active_reserved_bytes()

    def profile(frame: FrameType, event: str, argument: object) -> None:
        del argument
        if observed or event != "call":
            return
        if frame.f_code.co_name == "_bounded_parameter_input":
            if fault == "array_shape":
                values.shape = (1, 2)
                observed.append(fault)
            elif fault == "initial_growth":
                values.append(5.0)
                observed.append(fault)
        elif frame.f_code.co_name == "checkpoint":
            caller = frame.f_back
            if caller is not None and caller.f_code.co_name == "_bounded_parameter_input":
                if caller.f_locals.get("index") == 1 and fault == "late_shrink":
                    values.pop()
                    observed.append(fault)
                elif caller.f_locals.get("index") == 0 and fault == "late_growth":
                    values.append(5.0)
                    observed.append(fault)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="input changed after admission"):
            whole_program_value_and_grad(objective, values, trace=False)
    finally:
        sys.setprofile(previous)
    assert observed == [fault]
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    assert result.value == 7.0
    np.testing.assert_array_equal(result.gradient, [4.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_public_ad_refuses_scalar_storage_wider_than_admitted_conversion() -> None:
    """A numeric scalar protocol cannot underdeclare its claimed storage width."""

    class OversizedScalar(np.float64):
        """Claim extra storage through a real numeric scalar subtype."""

        @property
        def dtype(self) -> np.dtype[Any]:
            """Report the unadmitted storage width at the conversion boundary."""
            return np.dtype("V64")

    def objective(values: Any) -> object:
        return values[0] * values[0]

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="input changed after admission"):
        whole_program_value_and_grad(objective, [OversizedScalar(2.0)], trace=False)
    assert active_reserved_bytes() == baseline
    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    assert active_reserved_bytes() == baseline
