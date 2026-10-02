# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — captured state workspace tests
"""Exercise declared captured-state workspace through public AD access."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from threading import Event
from types import CodeType, FrameType, FunctionType
from typing import Protocol, cast

import numpy as np
import pytest

from scpn_quantum_control import TraceADArray, whole_program_value_and_grad
from scpn_quantum_control.dense_budget import GIB, DenseAllocationError
from scpn_quantum_control.differentiable import (
    program_adjoint_gradient,
    program_adjoint_replay_gradient,
    program_adjoint_result,
)
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
)


class _TraversalStorage(Protocol):
    seen: set[int]
    module_scopes: set[tuple[int, tuple[str, ...]]]
    code_references: list[CodeType]


@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
def test_callable_disassembly_workspace_obeys_parent_budget_and_recovers(access: str) -> None:
    """Source-line tables are admitted before materialising disassembly.

    Parameters
    ----------
    access
        Metadata access, attached gradient copy or executable replay.

    """
    namespace: dict[str, object] = {"coefficient": 2.0}
    source = "def dependency():\n" + "    coefficient\n" * 1800 + "    return 2.0\n"
    compiled = compile(source, __file__, "exec")
    code = next(value for value in compiled.co_consts if type(value) is CodeType)
    dependency = FunctionType(code, namespace)

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    with reserve_execution_memory(plan, max_gib=64 * 1024 / GIB):
        with pytest.raises(DenseAllocationError):
            getters[access](result)
        assert active_reserved_bytes() == baseline + 16
    with reserve_execution_memory(plan, max_gib=16 * 1024**2 / GIB):
        getters[access](result)
        assert active_reserved_bytes() == baseline + 16
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_opaque_code_constant_refuses_without_invoking_its_representation() -> None:
    """Disassembly never executes a captured object's formatting protocol."""
    calls: list[str] = []

    class Opaque:
        def __repr__(self) -> str:
            """Record the forbidden implicit callback."""
            calls.append("repr")
            return "opaque"

    def dependency() -> float:
        return 2.0

    dependency.__code__ = dependency.__code__.replace(co_consts=(None, Opaque()))

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="unsupported storage"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "constant",
    [None, Ellipsis, True, 17, -17, 1.25, 2.0 + 3.0j, "é", b"\x00", (1, "x"), frozenset({1, 2})],
)
def test_compiler_native_constants_preserve_the_live_numeric_derivative(constant: object) -> None:
    """Immutable code constants stay supported without executing a dependency.

    Parameters
    ----------
    constant
        A compiler-native immutable value stored in a captured callable's code.

    """

    def dependency() -> object:
        return 2.0

    dependency.__code__ = dependency.__code__.replace(co_consts=(None, constant))

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_nested_code_constant_preserves_derivative_and_rebinding_refusal() -> None:
    """A genuine nested definition stays immutable under captured code binding."""

    def dependency() -> object:
        def nested() -> float:
            return 2.0

        return nested

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    original = dependency.__code__
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    dependency.__code__ = original.replace(co_filename="changed-dependency.py")
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_result(result)
    dependency.__code__ = original
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_deep_code_constant_refuses_and_recapture_recovers() -> None:
    """Unsupported constant nesting is refused before recursive formatting."""

    def dependency() -> object:
        return 2.0

    original = dependency.__code__
    nested: object = 2.0
    for _ in range(66):
        nested = (nested,)
    dependency.__code__ = original.replace(co_consts=(None, nested))

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if dependency is not None else 5.0)

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="unsupported storage"):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    assert active_reserved_bytes() == baseline
    dependency.__code__ = original
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_sibling_callable_workspaces_preserve_each_code_identity() -> None:
    """Sequential sibling inspection shares capacity and retains both bindings."""

    def first() -> float:
        return 2.0

    def second() -> float:
        return 2.0

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if first is not None and second is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    original = second.__code__
    second.__code__ = original.replace(co_filename="changed-second.py")
    with pytest.raises(ValueError, match="captured program state changed"):
        program_adjoint_result(result)
    second.__code__ = original
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
def test_large_capture_identity_workspace_obeys_parent_budget_and_recovers(access: str) -> None:
    """Traversal tables cannot grow outside the enclosing execution allowance.

    Parameters
    ----------
    access
        Metadata access, attached gradient copy or executable replay.

    """
    state = [[2.0] for _ in range(1500)]

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    with reserve_execution_memory(plan, max_gib=64 * 1024 / GIB):
        with pytest.raises(DenseAllocationError):
            getters[access](result)
        assert active_reserved_bytes() == baseline + 16
    assert active_reserved_bytes() == baseline
    with reserve_execution_memory(plan, max_gib=512 * 1024 / GIB):
        getters[access](result)
        assert active_reserved_bytes() == baseline + 16
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_live_identity_tables_remain_inside_the_declared_workspace() -> None:
    """Real table allocations never exceed their active reservation charge."""
    state = [[2.0] for _ in range(1500)]

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    samples: list[tuple[int, int, int]] = []
    thresholds = [1, 32, 128, 512, 1500]

    def profile(frame: FrameType, event: str, argument: object) -> None:
        del argument
        if (
            event == "return"
            and frame.f_code.co_name == "aggregate"
            and frame.f_code.co_filename.endswith("/program_ad_captured_state.py")
        ):
            traversal = cast(_TraversalStorage, frame.f_locals["self"])
            if thresholds and len(traversal.seen) >= thresholds[0]:
                thresholds.pop(0)
                owned = (
                    sys.getsizeof(traversal.seen)
                    + sum(sys.getsizeof(item) for item in traversal.seen)
                    + sys.getsizeof(traversal.module_scopes)
                    + sum(
                        sys.getsizeof(scope) + sys.getsizeof(scope[0])
                        for scope in traversal.module_scopes
                    )
                    + sys.getsizeof(traversal.code_references)
                )
                samples.append((len(traversal.seen), owned, active_reserved_bytes()))

    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        assert program_adjoint_result(result).supported
    finally:
        sys.setprofile(previous)
    assert not thresholds
    assert all(charged >= baseline + owned for _, owned, charged in samples)
    assert samples[-1][1] > 64 * 1024
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


def test_module_scope_workspace_preserves_live_builtin_identity_and_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module-reference cycles stay admitted and retain referenced callables.

    Parameters
    ----------
    monkeypatch
        Test-local restoration of the real math module's sine attribute.

    """

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if math.sin is not None else 5.0)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    with monkeypatch.context() as changed:
        changed.setattr(math, "sin", math.cos)
        with pytest.raises(ValueError, match="captured program state changed"):
            program_adjoint_result(result)
        assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("access", ["metadata", "copy", "replay"])
def test_growing_identity_table_observes_parent_cancellation_and_recovers(access: str) -> None:
    """Cancellation during real table growth releases every child reservation.

    Parameters
    ----------
    access
        Metadata access, attached gradient copy or executable replay.

    """
    state = [[2.0] for _ in range(1500)]

    def objective(values: TraceADArray) -> object:
        return values[0] * (2.0 if state is not None else 5.0)

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    getters: dict[str, Callable[[object], object]] = {
        "metadata": program_adjoint_result,
        "copy": program_adjoint_gradient,
        "replay": program_adjoint_replay_gradient,
    }
    cancelled = Event()
    observed = False
    peak = 0

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal observed, peak
        del argument
        if (
            not observed
            and event == "return"
            and frame.f_code.co_name == "aggregate"
            and frame.f_code.co_filename.endswith("/program_ad_captured_state.py")
        ):
            traversal = cast(_TraversalStorage, frame.f_locals["self"])
            if len(traversal.seen) >= 128:
                observed = True
                peak = active_reserved_bytes()
                cancelled.set()

    baseline = active_reserved_bytes()
    plan = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (16,), "uint8"),))
    with reserve_execution_memory(plan, cancelled=cancelled):
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            with pytest.raises(ExecutionCancelledError):
                getters[access](result)
        finally:
            sys.setprofile(previous)
        assert observed and peak > baseline + 16
        assert active_reserved_bytes() == baseline + 16
        cancelled.clear()
        getters[access](result)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])
    assert active_reserved_bytes() == baseline
