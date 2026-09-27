# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public adjoint resource policy tests
"""Propagate execution-memory policy through real public adjoint entry points."""

import sys
from threading import Event
from time import monotonic
from types import FrameType
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable_parameter_contracts import Parameter
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
)
from scpn_quantum_control.program_ad_adjoint import (
    program_adjoint_grad,
    program_adjoint_gradient,
    program_adjoint_replay_gradient,
    program_adjoint_value_and_grad,
)
from scpn_quantum_control.whole_program_ad_api import whole_program_value_and_grad


def test_adjoint_entries_preserve_memory_refusal() -> None:
    """Both adjoint facades refuse an insufficient initial numeric-buffer cap."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    with pytest.raises(DenseAllocationError, match="execution memory"):
        program_adjoint_grad(objective, [2.0], trace=False, max_execution_gib=1 / 1024**3)
    with pytest.raises(DenseAllocationError, match="execution memory"):
        program_adjoint_value_and_grad(
            objective, [2.0], trace=False, max_execution_gib=1 / 1024**3
        )


def test_adjoint_entries_keep_analytic_value_and_gradient_under_cap() -> None:
    """Actual captured adjoint generation preserves the independent quadratic oracle."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    value, gradient = program_adjoint_value_and_grad(
        objective, [2.0, 3.0], trace=False, max_execution_gib=0.001
    )
    assert value == 7.0
    np.testing.assert_array_equal(gradient, [4.0, 1.0])
    np.testing.assert_array_equal(
        program_adjoint_grad(objective, [2.0, 3.0], trace=False, max_execution_gib=0.001),
        [4.0, 1.0],
    )


def test_attached_gradient_copy_owns_admission_and_disposal() -> None:
    """Retained result access cannot bypass numeric-copy or lifecycle admission."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError, match="execution memory"):
        program_adjoint_gradient(result, max_execution_gib=1 / 1024**3)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ExecutionCancelledError):
        program_adjoint_gradient(result, cancelled=cancelled)
    with pytest.raises(TimeoutError):
        program_adjoint_gradient(result, deadline_monotonic=monotonic() - 1)
    gradient = program_adjoint_gradient(result, max_execution_gib=0.001)
    np.testing.assert_array_equal(gradient, [4.0])
    assert result.adjoint_result is not None
    assert not np.shares_memory(gradient, result.adjoint_result.gradient)
    assert active_reserved_bytes() == baseline


def test_executable_replay_owns_workspace_policy_and_disposal() -> None:
    """Real replay refuses caps and interrupted entry without leaking its charge."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    baseline = active_reserved_bytes()
    with pytest.raises(DenseAllocationError, match="execution memory"):
        program_adjoint_replay_gradient(result, max_execution_gib=1 / 1024**3)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ExecutionCancelledError):
        program_adjoint_replay_gradient(result, cancelled=cancelled)
    with pytest.raises(TimeoutError):
        program_adjoint_replay_gradient(result, deadline_monotonic=monotonic() - 1)
    np.testing.assert_array_equal(
        program_adjoint_replay_gradient(result, max_execution_gib=0.001), [4.0, 1.0]
    )
    assert active_reserved_bytes() == baseline


def test_public_replay_inherits_active_parent_cancellation() -> None:
    """Calling replay with default policy cannot clear an enclosing cancellation."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    parent_plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    cancelled = Event()
    baseline = active_reserved_bytes()
    with reserve_execution_memory(parent_plan, cancelled=cancelled):
        cancelled.set()
        with pytest.raises(ExecutionCancelledError):
            program_adjoint_replay_gradient(result)
        assert active_reserved_bytes() == baseline + 16
        cancelled.clear()
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])
    assert active_reserved_bytes() == baseline


def test_public_generation_retains_real_step_stream_through_replay() -> None:
    """Public AD preserves generated steps, analytic gradients and scoped disposal."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1] * values[1]

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(
        objective, [2.0, 3.0], trace=True, max_execution_gib=0.01
    )
    assert result.adjoint_result is not None
    assert result.adjoint_result.supported
    assert len(result.adjoint_result.adjoint_steps) == len(result.ir_nodes)
    assert result.value == 13.0
    np.testing.assert_array_equal(result.adjoint_result.gradient, [4.0, 6.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0, 6.0])
    assert active_reserved_bytes() == baseline


def test_public_cumulative_adjoint_preserves_independent_shape_oracle() -> None:
    """Real cumulative primitive generation retains its closed-form scalar gradient."""

    def objective(values: Any) -> object:
        return np.sum(np.cumsum(values))

    values = np.array([1.0, 2.0, 3.0])
    baseline = active_reserved_bytes()
    value, gradient = program_adjoint_value_and_grad(
        objective, values, trace=False, max_execution_gib=0.01
    )
    expected_gradient = np.arange(values.size, 0, -1, dtype=np.float64)
    assert value == float(np.dot(values, expected_gradient))
    np.testing.assert_array_equal(gradient, expected_gradient)
    assert active_reserved_bytes() == baseline


def test_attached_gradient_accessor_refuses_post_capture_nonfinite_data_and_recovers() -> None:
    """Writable captured arrays cannot promote nonfinite gradient copies."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    assert result.adjoint_result is not None
    source = result.adjoint_result.gradient
    baseline = active_reserved_bytes()
    for value in (np.nan, np.inf, -np.inf):
        source[0] = value
        with pytest.raises(ValueError, match="finite values"):
            program_adjoint_gradient(result)
        assert active_reserved_bytes() == baseline
    source[:] = [4.0, 1.0]
    gradient = program_adjoint_gradient(result)
    np.testing.assert_array_equal(gradient, [4.0, 1.0])
    gradient[0] = 99.0
    np.testing.assert_array_equal(source, [4.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_attached_gradient_accessor_refuses_mutated_shape_dtype_and_alignment() -> None:
    """Captured typed shape metadata remains binding after ndarray mutations."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    assert result.adjoint_result is not None
    source = result.adjoint_result.gradient
    baseline = active_reserved_bytes()
    source.shape = (1, 2)
    with pytest.raises(ValueError, match="one-dimensional float64"):
        program_adjoint_gradient(result)
    source.shape = (2,)
    source.dtype = np.dtype(np.uint8)
    with pytest.raises(ValueError, match="one-dimensional float64"):
        program_adjoint_gradient(result)
    source.dtype = np.dtype(np.float64)
    source.resize(1, refcheck=False)
    with pytest.raises(ValueError, match="shape must match parameter names"):
        program_adjoint_gradient(result)
    source.resize(2, refcheck=False)
    source[:] = [4.0, 1.0]
    np.testing.assert_array_equal(program_adjoint_gradient(result), [4.0, 1.0])
    assert active_reserved_bytes() == baseline


def test_attached_gradient_accessor_detects_shape_change_after_capacity_admission() -> None:
    """A real reservation-entry mutation refuses before fixed-shape copying."""

    def objective(values: Any) -> object:
        return values[0] * values[0]

    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.adjoint_result is not None
    source = result.adjoint_result.gradient
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if observed or event != "call" or frame.f_code.co_name != "reserve_execution_memory":
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == "program_adjoint_gradient":
                observed = True
                source.shape = (1, 1)
                return
            caller = caller.f_back

    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="changed after admission"):
            program_adjoint_gradient(result)
    finally:
        sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    source.shape = (1,)
    np.testing.assert_array_equal(program_adjoint_gradient(result), [4.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("kind", ["value", "tangent"])
def test_public_ad_entries_refuse_nonfinite_computation_and_recover(kind: str) -> None:
    """Real arithmetic overflow cannot escape as a captured AD result.

    Parameters
    ----------
    kind
        Overflow the primal product or the division's derivative arithmetic.

    """
    if kind == "value":

        def objective(values: Any) -> object:
            return values[0] * values[0]

        bad_values = [1e308]
        good_values = [2.0]
        expected_value, expected_gradient = 4.0, [4.0]
    else:

        def objective(values: Any) -> object:
            return values[0] / values[1]

        bad_values = [1.0, 1e-308]
        good_values = [6.0, 2.0]
        expected_value, expected_gradient = 3.0, [0.5, -1.5]
    baseline = active_reserved_bytes()
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="finite"):
            whole_program_value_and_grad(objective, bad_values, trace=False)
        assert active_reserved_bytes() == baseline
        with pytest.raises(ValueError, match="finite"):
            program_adjoint_value_and_grad(objective, bad_values, trace=False)
        assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(objective, good_values, trace=False)
    assert value == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)
    assert active_reserved_bytes() == baseline


def test_attached_gradient_accessor_refuses_nonzero_frozen_entries_and_recovers() -> None:
    """Post-capture mutations cannot re-enable a frozen derivative coordinate."""

    def objective(values: Any) -> object:
        return values[0] * values[0] + values[1]

    result = whole_program_value_and_grad(
        objective, [2.0, 3.0], parameters=[Parameter("x"), Parameter("y", False)], trace=False
    )
    assert result.adjoint_result is not None
    source = result.adjoint_result.gradient
    baseline = active_reserved_bytes()
    np.testing.assert_array_equal(source, [4.0, 0.0])
    source[1] = 1.0
    with pytest.raises(ValueError, match="zero for non-trainable"):
        program_adjoint_gradient(result)
    assert active_reserved_bytes() == baseline
    source[1] = 0.0
    np.testing.assert_array_equal(program_adjoint_gradient(result), [4.0, 0.0])
    assert active_reserved_bytes() == baseline
