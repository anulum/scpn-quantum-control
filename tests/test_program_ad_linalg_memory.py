# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Public matrix-power execution admission tests
"""Exercise real matrix-power callbacks, failure recovery and lifecycle owners."""

import sys
from threading import Event
from types import FrameType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.differentiable import PrimitiveIdentity, primitive_contract_for
from scpn_quantum_control.execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from scpn_quantum_control.execution_reservations import (
    ExecutionCancelledError,
    active_reserved_bytes,
    reserve_execution_memory,
)
from scpn_quantum_control.program_ad_adjoint import program_adjoint_value_and_grad
from scpn_quantum_control.program_ad_linalg_primitives import (
    program_ad_linalg_diag_derivative_rule,
    program_ad_linalg_diagflat_derivative_rule,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    program_ad_linalg_solve_derivative_rule,
)


@pytest.mark.parametrize("power", [0, 2, -1, -2])
def test_matrix_power_public_callbacks_match_independent_differentials(power: int) -> None:
    """Preserve explicit product/inverse differentials and dispose each scope.

    Parameters
    ----------
    power
        Identity, square, inverse or squared-inverse operation.

    """
    baseline = active_reserved_bytes()
    matrix = np.diag([2.0, 4.0])
    operand = np.array([[1.0, 2.0], [3.0, 4.0]])
    inverse = np.diag([0.5, 0.25])
    if power == 0:
        expected_value = np.eye(2)
        expected_derivative = np.zeros((2, 2))
    elif power == 2:
        expected_value = np.diag([4.0, 16.0])
        expected_derivative = matrix @ operand + operand @ matrix
    elif power == -1:
        expected_value = inverse
        expected_derivative = -(inverse @ operand @ inverse)
    else:
        expected_value = np.diag([0.25, 0.0625])
        expected_derivative = -(
            inverse @ inverse @ operand @ inverse + inverse @ operand @ inverse @ inverse
        )
    rule = program_ad_linalg_matrix_power_derivative_rule(power)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    np.testing.assert_array_equal(rule.value_fn(matrix.reshape(-1)), expected_value.reshape(-1))
    np.testing.assert_array_equal(
        rule.jvp_rule(matrix.reshape(-1), operand.reshape(-1)), expected_derivative.reshape(-1)
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(matrix.reshape(-1), operand.reshape(-1)), expected_derivative.reshape(-1)
    )
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("power", [10_000_000, -10_000_000, sys.maxsize])
def test_matrix_power_derivatives_refuse_retained_powers_and_recover(
    power: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Huge retained power lists refuse before creation under real environment caps.

    Parameters
    ----------
    power
        Exponent exceeding policy capacity or native byte addressability.
    monkeypatch
        Real process environment scope for the dense memory cap.

    """
    baseline = active_reserved_bytes()
    values = np.array([2.0, 0.0, 0.0, 4.0])
    operand = np.ones(4)
    rule = program_ad_linalg_matrix_power_derivative_rule(power)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        for callback in (rule.jvp_rule, rule.vjp_rule):
            with pytest.raises(DenseAllocationError):
                callback(values, operand)
            assert active_reserved_bytes() == baseline
        retry = program_ad_linalg_matrix_power_derivative_rule(2)
        assert retry.jvp_rule is not None
        np.testing.assert_array_equal(retry.jvp_rule(values, operand), [4.0, 6.0, 6.0, 8.0])
    assert active_reserved_bytes() == baseline


def test_matrix_power_value_refuses_before_expanding_virtual_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large zero-stride input is refused before float conversion or matrix power.

    Parameters
    ----------
    monkeypatch
        Real process environment scope for the dense memory cap.

    """
    baseline = active_reserved_bytes()
    source = np.ndarray((100_000_000,), dtype=np.float64, buffer=np.array([1.0]), strides=(0,))
    rule = program_ad_linalg_matrix_power_derivative_rule(2)
    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.001")
        with pytest.raises(DenseAllocationError):
            rule.value_fn(source)
        np.testing.assert_array_equal(rule.value_fn(np.array([2.0])), [4.0])
    assert active_reserved_bytes() == baseline


def test_matrix_power_callback_failure_restores_owner_and_allows_retry() -> None:
    """Real singular and malformed operands dispose their active execution scopes."""
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_matrix_power_derivative_rule(-1)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    for callback in (rule.jvp_rule, rule.vjp_rule):
        with pytest.raises(np.linalg.LinAlgError):
            callback(np.array([1.0, 2.0, 2.0, 4.0]), np.ones(4))
        with pytest.raises(ValueError, match="shape must match"):
            callback(np.ones(4), np.ones(9))
        assert active_reserved_bytes() == baseline
    with pytest.raises(ValueError, match="flattened square"):
        rule.value_fn(np.ones(3))
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0])), [0.5])
    assert active_reserved_bytes() == baseline


def test_matrix_power_loop_inherits_real_parent_cancellation() -> None:
    """A real NumPy power call cancels the parent and the next checkpoint refuses."""
    baseline = active_reserved_bytes()
    event = Event()
    seen = False
    parent = ExecutionMemoryPlan((ExecutionBuffer("parent", "forward", (1,), "float64"),))

    def observe(frame: FrameType, trace_event: str, argument: object) -> None:
        nonlocal seen
        if (
            trace_event == "call"
            and frame.f_code.co_name == "matrix_power"
            and str(frame.f_globals.get("__name__", "")).startswith("numpy.linalg")
        ):
            seen = True
            event.set()

    rule = program_ad_linalg_matrix_power_derivative_rule(3)
    assert rule.jvp_rule is not None
    previous = sys.getprofile()
    with reserve_execution_memory(parent, cancelled=event):
        try:
            sys.setprofile(observe)
            with pytest.raises(ExecutionCancelledError):
                rule.jvp_rule(np.array([2.0, 0.0, 0.0, 4.0]), np.ones(4))
        finally:
            sys.setprofile(previous)
    assert seen
    assert active_reserved_bytes() == baseline


def test_matrix_power_opaque_and_nonreal_inputs_refuse_without_conversion() -> None:
    """The public value callback rejects protocol-bearing and complex inputs early."""
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_matrix_power_derivative_rule(2)
    for invalid in (np.array([1.0j]), np.array([object()], dtype=object)):
        with pytest.raises(ValueError, match="plain real numeric"):
            rule.value_fn(cast(NDArray[np.float64], invalid))
        assert active_reserved_bytes() == baseline
    empty = np.empty(0, dtype=np.float64)
    identity = program_ad_linalg_matrix_power_derivative_rule(0)
    assert identity.jvp_rule is not None
    assert identity.vjp_rule is not None
    np.testing.assert_array_equal(identity.value_fn(empty), empty)
    np.testing.assert_array_equal(identity.jvp_rule(empty, empty), empty)
    np.testing.assert_array_equal(identity.vjp_rule(empty, empty), empty)
    assert active_reserved_bytes() == baseline


def _inverse_sum_objective(values: Any) -> object:
    """Return a composed inverse objective through the public trace surface."""
    return np.sum(np.linalg.inv(np.reshape(values, (2, 2))))


def _solve_vector_sum_objective(values: Any) -> object:
    """Return a vector RHS solve objective through the public trace surface."""
    return np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:]))


def _solve_matrix_sum_objective(values: Any) -> object:
    """Return a matrix RHS solve objective through the public trace surface."""
    matrix = np.reshape(values[:4], (2, 2))
    rhs = np.reshape(values[4:], (2, 2))
    return np.sum(np.linalg.solve(matrix, rhs))


@pytest.mark.parametrize("kind", ["inverse", "solve_vector", "solve_matrix"])
def test_public_adjoint_pullbacks_preserve_independent_linalg_oracles(kind: str) -> None:
    """Actual inverse and solve generation preserves independently derived gradients.

    Parameters
    ----------
    kind
        Inverse, vector RHS solve or matrix RHS solve objective.

    """
    baseline = active_reserved_bytes()
    if kind == "inverse":
        objective = _inverse_sum_objective
        values = [2.0, 0.0, 0.0, 4.0]
        expected_value = 0.75
        expected_gradient = [-0.25, -0.125, -0.125, -0.0625]
    elif kind == "solve_vector":
        objective = _solve_vector_sum_objective
        values = [2.0, 0.0, 0.0, 4.0, 6.0, 8.0]
        expected_value = 5.0
        expected_gradient = [-1.5, -1.0, -0.75, -0.5, 0.5, 0.25]
    else:
        objective = _solve_matrix_sum_objective
        values = [2.0, 0.0, 0.0, 4.0, 6.0, 10.0, 8.0, 12.0]
        expected_value = 13.0
        expected_gradient = [-4.0, -2.5, -2.0, -1.25, 0.5, 0.5, 0.25, 0.25]
    value, gradient = program_adjoint_value_and_grad(objective, values, trace=False)
    assert value == expected_value
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0.0, atol=1.0e-12)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("kind", ["inverse", "solve"])
def test_public_adjoint_pullback_refuses_its_own_capacity_and_recovers(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Change the real cap at pullback entry after the public primal trace completed.

    Parameters
    ----------
    kind
        Inverse or solve pullback.
    monkeypatch
        Restore the real process capacity setting after refusal.

    """
    objective = _inverse_sum_objective if kind == "inverse" else _solve_vector_sum_objective
    values = [2.0, 0.0, 0.0, 4.0] if kind == "inverse" else [2.0, 0.0, 0.0, 4.0, 6.0, 8.0]
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    entered = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal entered
        if event == "call" and frame.f_code.co_name == "linalg_pullback_execution_scope":
            entered = True
            environment.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))

    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.01")
        sys.setprofile(profile)
        try:
            with pytest.raises(DenseAllocationError):
                program_adjoint_value_and_grad(objective, values, trace=False)
        finally:
            sys.setprofile(previous)
    assert entered
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(objective, values, trace=False)
    assert value == (0.75 if kind == "inverse" else 5.0)
    assert np.all(np.isfinite(gradient))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("kind", ["inverse", "solve"])
def test_public_adjoint_pullback_observes_cancellation_after_native_operation(kind: str) -> None:
    """Cancel after a real reverse native operation and verify parent-owner cleanup.

    Parameters
    ----------
    kind
        Inverse or solve pullback whose native return triggers cancellation.

    """
    objective = _inverse_sum_objective if kind == "inverse" else _solve_vector_sum_objective
    values = [2.0, 0.0, 0.0, 4.0] if kind == "inverse" else [2.0, 0.0, 0.0, 4.0, 6.0, 8.0]
    operation = "inv" if kind == "inverse" else "solve"
    owner = f"_program_adjoint_{'inv' if kind == 'inverse' else 'solve'}_contributions"
    baseline = active_reserved_bytes()
    cancelled = Event()
    previous = sys.getprofile()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if event != "return" or frame.f_code.co_name != operation:
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == owner:
                observed = True
                cancelled.set()
                return
            caller = caller.f_back

    sys.setprofile(profile)
    try:
        with pytest.raises(ExecutionCancelledError):
            program_adjoint_value_and_grad(objective, values, trace=False, cancelled=cancelled)
    finally:
        sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(objective, values, trace=False)
    assert value == (0.75 if kind == "inverse" else 5.0)
    assert np.all(np.isfinite(gradient))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("size", [0, 1, 2])
def test_registered_determinant_callbacks_preserve_values_and_differentials(size: int) -> None:
    """Registered callbacks preserve independent empty/scalar/matrix oracles.

    Parameters
    ----------
    size
        Square dimension including the supported empty determinant.

    """
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "det", "1")
    ).derivative_rule
    assert rule is not None
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    if size == 0:
        values = np.empty(0, dtype=np.float64)
        tangent = np.empty(0, dtype=np.float64)
        value, derivative, pullback = 1.0, 0.0, np.empty(0, dtype=np.float64)
    elif size == 1:
        values = np.array([1.0])
        tangent = np.array([5.0])
        value, derivative, pullback = 1.0, 5.0, np.array([2.0])
    else:
        values = np.array([1.0, 0.0, 0.0, 1.0])
        tangent = np.array([1.0, 2.0, 3.0, 4.0])
        value, derivative, pullback = 1.0, 5.0, np.array([2.0, 0.0, 0.0, 2.0])
    np.testing.assert_array_equal(rule.value_fn(values), [value])
    np.testing.assert_array_equal(rule.jvp_rule(values, tangent), [derivative])
    np.testing.assert_array_equal(rule.vjp_rule(values, np.array([2.0])), pullback)
    assert active_reserved_bytes() == baseline


def test_registered_determinant_callbacks_refuse_virtual_storage_and_recover() -> None:
    """Value/JVP/VJP refuse before inspecting huge zero-stride input data."""
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "det", "1")
    ).derivative_rule
    assert rule is not None
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    huge = np.broadcast_to(np.array([1.0]), (sys.maxsize // 16 + 1,))
    for callback, args in (
        (rule.value_fn, (huge,)),
        (rule.jvp_rule, (huge, np.array([1.0]))),
        (rule.vjp_rule, (huge, np.array([1.0]))),
    ):
        with pytest.raises(DenseAllocationError):
            callback(*args)
        assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(rule.value_fn(np.array([1.0])), [1.0])
    assert active_reserved_bytes() == baseline


def _determinant_objective(values: Any) -> object:
    """Return a four-dimensional determinant through the public trace path."""
    return np.linalg.det(np.reshape(values, (4, 4)))


def test_public_determinant_trace_and_adjoint_preserve_independent_gradient() -> None:
    """Trace and generated pullback retain the diagonal product oracle."""
    baseline = active_reserved_bytes()
    values = np.diag([2.0, 3.0, 4.0, 5.0]).reshape(-1)
    value, gradient = program_adjoint_value_and_grad(_determinant_objective, values, trace=False)
    assert value == pytest.approx(120.0, rel=0.0, abs=1.0e-12)
    expected = np.diag([60.0, 40.0, 30.0, 24.0]).reshape(-1)
    np.testing.assert_allclose(gradient, expected, rtol=0.0, atol=1.0e-12)
    assert active_reserved_bytes() == baseline


def test_registered_determinant_cofactors_observe_native_return_cancellation() -> None:
    """A real minor determinant return cancels the inherited callback owner."""
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "det", "1")
    ).derivative_rule
    assert rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    cancelled = Event()
    previous = sys.getprofile()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if event != "return" or frame.f_code.co_name != "det":
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == "_program_ad_linalg_det_cofactor_matrix":
                observed = True
                cancelled.set()
                return
            caller = caller.f_back

    values = np.array([2.0, 0.0, 0.0, 4.0])
    parent = ExecutionMemoryPlan((ExecutionBuffer("det_parent", "forward", (1,), "float64"),))
    with reserve_execution_memory(parent, cancelled=cancelled):
        sys.setprofile(profile)
        try:
            with pytest.raises(ExecutionCancelledError):
                rule.vjp_rule(values, np.array([2.0]))
        finally:
            sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(rule.vjp_rule(values, np.array([2.0])), [8.0, 0.0, 0.0, 4.0])
    assert active_reserved_bytes() == baseline


def test_registered_determinant_callbacks_refuse_opaque_dtype_and_shape_then_recover() -> None:
    """Real registered callbacks reject malformed input before unsupported conversion."""
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "det", "1")
    ).derivative_rule
    assert rule is not None
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()

    class Opaque:
        """Array protocol that must never execute on the static admission boundary."""

        def __array__(self, dtype: object = None, copy: object = None) -> NDArray[np.float64]:
            """Fail if opaque conversion executes instead of metadata refusal."""
            raise AssertionError("opaque determinant conversion executed")

    for value in (
        Opaque(),
        np.array([1.0j]),
        np.array([object()], dtype=object),
        np.array([True]),
    ):
        with pytest.raises(ValueError, match="plain real numeric arrays"):
            rule.value_fn(cast(Any, value))
    with pytest.raises(ValueError, match="flattened square matrix"):
        rule.value_fn(np.ones(3))
    with pytest.raises(ValueError, match="tangent shape"):
        rule.jvp_rule(np.array([2.0, 0.0, 0.0, 4.0]), np.ones(1))
    with pytest.raises(ValueError, match="scalar cotangent"):
        rule.vjp_rule(np.array([2.0, 0.0, 0.0, 4.0]), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        rule.value_fn(np.array([np.nan]))
    np.testing.assert_array_equal(rule.value_fn(np.array([1.0])), [1.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("size", [0, 1, 2])
def test_registered_inverse_callbacks_preserve_independent_value_jvp_vjp(size: int) -> None:
    """Registered inverse callbacks preserve empty/scalar and nonsymmetric oracles.

    Parameters
    ----------
    size
        Empty, scalar or nonsymmetric square matrix dimension.

    """
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "inv", "1")
    ).derivative_rule
    assert rule is not None
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    if size == 0:
        values = np.empty(0, dtype=np.float64)
        operand = np.empty(0, dtype=np.float64)
        expected_value = expected_jvp = expected_vjp = np.empty(0, dtype=np.float64)
    elif size == 1:
        values, operand = np.array([2.0]), np.array([3.0])
        expected_value = np.array([0.5])
        expected_jvp = expected_vjp = np.array([-0.75])
    else:
        values, operand = np.array([2.0, 1.0, 0.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])
        expected_value = np.array([0.5, -0.125, 0.0, 0.25])
        expected_jvp = np.array([-0.0625, -0.109375, -0.375, -0.15625])
        expected_vjp = np.array([-0.125, -0.25, -0.21875, -0.1875])
    np.testing.assert_array_equal(rule.value_fn(values), expected_value)
    np.testing.assert_array_equal(rule.jvp_rule(values, operand), expected_jvp)
    np.testing.assert_array_equal(rule.vjp_rule(values, operand), expected_vjp)
    assert active_reserved_bytes() == baseline


def test_registered_inverse_callbacks_refuse_huge_and_malformed_inputs_then_recover() -> None:
    """Real callbacks refuse capacity/dtype/shape/finite errors and dispose scopes."""
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "inv", "1")
    ).derivative_rule
    assert rule is not None
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    huge = np.broadcast_to(np.array([1.0]), (sys.maxsize // 16 + 1,))
    for callback, args in (
        (rule.value_fn, (huge,)),
        (rule.jvp_rule, (huge, np.array([1.0]))),
        (rule.vjp_rule, (huge, np.array([1.0]))),
    ):
        with pytest.raises(DenseAllocationError):
            callback(*args)
    for value in (np.array([True]), np.array([1.0j]), np.array([object()], dtype=object)):
        with pytest.raises(ValueError, match="plain real numeric arrays"):
            rule.value_fn(cast(Any, value))
    with pytest.raises(ValueError, match="flattened square matrix"):
        rule.value_fn(np.ones(3))
    with pytest.raises(ValueError, match="tangent shape"):
        rule.jvp_rule(np.array([2.0, 0.0, 0.0, 4.0]), np.ones(1))
    with pytest.raises(ValueError, match="cotangent shape"):
        rule.vjp_rule(np.array([2.0, 0.0, 0.0, 4.0]), np.ones(1))
    for callback, args in (
        (rule.value_fn, (np.array([np.inf]),)),
        (rule.jvp_rule, (np.array([2.0]), np.array([np.nan]))),
        (rule.vjp_rule, (np.array([2.0]), np.array([np.inf]))),
    ):
        with pytest.raises(ValueError, match="finite"):
            callback(*args)
    with pytest.raises(np.linalg.LinAlgError):
        rule.value_fn(np.array([1.0, 2.0, 2.0, 4.0]))
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0])), [0.5])
    assert active_reserved_bytes() == baseline


def test_public_inverse_trace_refuses_workspace_before_native_call_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real trace entry capacity refuses before inverse, then preserves its oracle.

    Parameters
    ----------
    monkeypatch
        Restore the real process environment after observed trace-entry refusal.

    """
    values = [2.0, 1.0, 0.0, 4.0]
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    entered = False
    native_called = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal entered, native_called
        if event != "call":
            return
        if frame.f_code.co_name == "_trace_inv":
            entered = True
            environment.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
        elif frame.f_code.co_name == "inv":
            caller = frame.f_back
            while caller is not None:
                if caller.f_code.co_name == "_trace_inv":
                    native_called = True
                    return
                caller = caller.f_back

    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.01")
        sys.setprofile(profile)
        try:
            with pytest.raises(DenseAllocationError):
                program_adjoint_value_and_grad(_inverse_sum_objective, values, trace=False)
        finally:
            sys.setprofile(previous)
    assert entered
    assert not native_called
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(_inverse_sum_objective, values, trace=False)
    assert value == 0.625
    np.testing.assert_array_equal(gradient, [-0.1875, -0.125, -0.046875, -0.03125])
    assert active_reserved_bytes() == baseline


def test_registered_inverse_value_observes_cancellation_before_output_copy() -> None:
    """An actual native inverse return cancels its registered value callback."""
    rule = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "inv", "1")
    ).derivative_rule
    assert rule is not None
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    cancelled = Event()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if event != "return" or frame.f_code.co_name != "inv":
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == "_program_ad_linalg_inv_value":
                observed = True
                cancelled.set()
                return
            caller = caller.f_back

    parent = ExecutionMemoryPlan((ExecutionBuffer("inv_parent", "forward", (1,), "float64"),))
    with reserve_execution_memory(parent, cancelled=cancelled):
        sys.setprofile(profile)
        try:
            with pytest.raises(ExecutionCancelledError):
                rule.value_fn(np.array([2.0]))
        finally:
            sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0])), [0.5])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("static", [False, True])
def test_solve_vector_callbacks_preserve_independent_differentials(static: bool) -> None:
    """Exercise registered and fixed-shape solves against explicit derivatives.

    Parameters
    ----------
    static
        Select the fixed-shape factory or registered legacy vector signature.

    """
    rule = (
        program_ad_linalg_solve_derivative_rule((2, 2), (2,))
        if static
        else primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.linalg", "solve", "1")
        ).derivative_rule
    )
    assert rule is not None
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    values = np.array([2.0, 0.0, 0.0, 4.0, 6.0, 8.0])
    np.testing.assert_array_equal(rule.value_fn(values), [3.0, 2.0])
    np.testing.assert_array_equal(
        rule.jvp_rule(values, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])), [-1.0, -2.75]
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(values, np.array([1.0, 2.0])), [-1.5, -1.0, -1.5, -1.0, 0.5, 0.5]
    )
    with pytest.raises(ValueError, match="finite"):
        rule.value_fn(np.array([np.inf, 1.0]))
    with pytest.raises(np.linalg.LinAlgError):
        rule.value_fn(np.array([1.0, 2.0, 2.0, 4.0, 6.0, 8.0]))
    np.testing.assert_array_equal(rule.value_fn(values), [3.0, 2.0])
    assert active_reserved_bytes() == baseline


def test_solve_matrix_rhs_callbacks_preserve_independent_differentials() -> None:
    """Matrix RHS retains columnwise solutions and its summed matrix pullback."""
    rule = program_ad_linalg_solve_derivative_rule((2, 2), (2, 2))
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    values = np.array([2.0, 0.0, 0.0, 4.0, 6.0, 10.0, 8.0, 12.0])
    np.testing.assert_array_equal(rule.value_fn(values), [3.0, 5.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        rule.jvp_rule(values, np.array([0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 8.0, 12.0])),
        [1.0, 2.0, 2.0, 3.0],
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(values, np.ones(4)), [-4.0, -2.5, -2.0, -1.25, 0.5, 0.5, 0.25, 0.25]
    )
    with pytest.raises(ValueError, match="cotangent shape"):
        rule.vjp_rule(values, np.ones(1))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("matrix_rhs", [False, True])
def test_public_solve_trace_refuses_before_native_call_then_recovers(
    monkeypatch: pytest.MonkeyPatch,
    matrix_rhs: bool,
) -> None:
    """Exercise real trace admission with both RHS ranks and clean retries.

    Parameters
    ----------
    monkeypatch
        Restore the process capacity after trace-entry refusal.
    matrix_rhs
        Select vector or matrix RHS through the public adjoint facade.

    """
    objective = _solve_matrix_sum_objective if matrix_rhs else _solve_vector_sum_objective
    values = (
        [2.0, 0.0, 0.0, 4.0, 6.0, 10.0, 8.0, 12.0]
        if matrix_rhs
        else [2.0, 0.0, 0.0, 4.0, 6.0, 8.0]
    )
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    entered = False
    native_called = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal entered, native_called
        if event != "call":
            return
        if frame.f_code.co_name == "_trace_solve":
            entered = True
            environment.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
        elif frame.f_code.co_name == "solve":
            caller = frame.f_back
            while caller is not None:
                if caller.f_code.co_name == "_trace_solve":
                    native_called = True
                    return
                caller = caller.f_back

    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.01")
        sys.setprofile(profile)
        try:
            with pytest.raises(DenseAllocationError):
                program_adjoint_value_and_grad(objective, values, trace=False)
        finally:
            sys.setprofile(previous)
    assert entered
    assert not native_called
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(objective, values, trace=False)
    if matrix_rhs:
        assert value == 13.0
        np.testing.assert_array_equal(gradient, [-4.0, -2.5, -2.0, -1.25, 0.5, 0.5, 0.25, 0.25])
    else:
        assert value == 5.0
        np.testing.assert_array_equal(gradient, [-1.5, -1.0, -0.75, -0.5, 0.5, 0.25])
    assert active_reserved_bytes() == baseline


def _matrix_square_sum_objective(values: Any) -> object:
    """Return the sum of the squared matrix through the public trace facade."""
    return np.sum(np.linalg.matrix_power(np.reshape(values, (2, 2)), 2))


def _matrix_inverse_power_sum_objective(values: Any) -> object:
    """Return a negative-power matrix objective through the public trace facade."""
    return np.sum(np.linalg.matrix_power(np.reshape(values, (2, 2)), -1))


@pytest.mark.parametrize("inverse", [False, True])
def test_public_matrix_power_trace_preserves_composed_independent_oracles(inverse: bool) -> None:
    """Public trace and reverse generation retain positive/negative power results.

    Parameters
    ----------
    inverse
        Select a negative power rather than the square objective.

    """
    objective = _matrix_inverse_power_sum_objective if inverse else _matrix_square_sum_objective
    baseline = active_reserved_bytes()
    value, gradient = program_adjoint_value_and_grad(objective, [2.0, 0.0, 0.0, 4.0], trace=False)
    assert value == (0.75 if inverse else 20.0)
    np.testing.assert_array_equal(
        gradient, [-0.25, -0.125, -0.125, -0.0625] if inverse else [4.0, 6.0, 6.0, 8.0]
    )
    assert active_reserved_bytes() == baseline


def test_public_matrix_power_trace_refuses_before_value_callback_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actual trace-entry capacity change refuses before numeric power execution.

    Parameters
    ----------
    monkeypatch
        Restore the environment after the observed public boundary refusal.

    """
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    entered = False
    value_called = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal entered, value_called
        if event != "call":
            return
        if frame.f_code.co_name == "_trace_matrix_power":
            entered = True
            environment.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
        elif frame.f_code.co_name == "value_fn":
            caller = frame.f_back
            while caller is not None:
                if caller.f_code.co_name == "_trace_matrix_power":
                    value_called = True
                    return
                caller = caller.f_back

    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", "0.01")
        sys.setprofile(profile)
        try:
            with pytest.raises(DenseAllocationError):
                program_adjoint_value_and_grad(
                    _matrix_square_sum_objective, [2.0, 0.0, 0.0, 4.0], trace=False
                )
        finally:
            sys.setprofile(previous)
    assert entered
    assert not value_called
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(
        _matrix_square_sum_objective, [2.0, 0.0, 0.0, 4.0], trace=False
    )
    assert value == 20.0
    np.testing.assert_array_equal(gradient, [4.0, 6.0, 6.0, 8.0])
    assert active_reserved_bytes() == baseline


def test_public_matrix_power_trace_observes_real_jvp_return_cancellation() -> None:
    """Cancellation after an actual JVP unwinds trace and nested workspace owners."""
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    cancelled = Event()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if event != "return" or frame.f_code.co_name != "jvp_rule":
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == "_trace_matrix_power":
                observed = True
                cancelled.set()
                return
            caller = caller.f_back

    parent = ExecutionMemoryPlan(
        (ExecutionBuffer("power_trace_parent", "forward", (1,), "float64"),)
    )
    with reserve_execution_memory(parent, cancelled=cancelled):
        sys.setprofile(profile)
        try:
            with pytest.raises(ExecutionCancelledError):
                program_adjoint_value_and_grad(
                    _matrix_square_sum_objective, [2.0, 0.0, 0.0, 4.0], trace=False
                )
        finally:
            sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(
        _matrix_square_sum_objective, [2.0, 0.0, 0.0, 4.0], trace=False
    )
    assert value == 20.0
    np.testing.assert_array_equal(gradient, [4.0, 6.0, 6.0, 8.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("power", [0, 2, -1])
def test_matrix_power_callbacks_refuse_nonfinite_inputs_and_recover(power: int) -> None:
    """Finite validation covers all direct callbacks, including identity powers.

    Parameters
    ----------
    power
        Identity, positive or negative static exponent.

    """
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_matrix_power_derivative_rule(power)
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    values = np.array([2.0, 0.0, 0.0, 4.0])
    bad = np.array([np.inf, 0.0, 0.0, 4.0])
    nan_operand = np.array([np.nan, 0.0, 0.0, 1.0])
    for callback, arguments in (
        (rule.value_fn, (bad,)),
        (rule.jvp_rule, (bad, np.ones(4))),
        (rule.vjp_rule, (bad, np.ones(4))),
        (rule.jvp_rule, (values, nan_operand)),
        (rule.vjp_rule, (values, nan_operand)),
    ):
        with pytest.raises(ValueError, match="finite"):
            callback(*arguments)
        assert active_reserved_bytes() == baseline
    expected = (
        [1.0, 0.0, 0.0, 1.0]
        if power == 0
        else ([4.0, 0.0, 0.0, 16.0] if power == 2 else [0.5, 0.0, 0.0, 0.25])
    )
    np.testing.assert_array_equal(rule.value_fn(values), expected)
    assert active_reserved_bytes() == baseline


def test_multi_dot_matrix_callbacks_preserve_independent_differentials() -> None:
    """Two matrix callbacks retain product, one varied operand and explicit pullbacks."""
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_multi_dot_derivative_rule(((2, 2), (2, 2)))
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    values = np.array([2.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0])
    np.testing.assert_array_equal(rule.value_fn(values), [6.0, 0.0, 0.0, 20.0])
    np.testing.assert_array_equal(
        rule.jvp_rule(values, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
        [3.0, 0.0, 0.0, 5.0],
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(values, np.ones(4)), [3.0, 5.0, 3.0, 5.0, 2.0, 2.0, 4.0, 4.0]
    )
    with pytest.raises(ValueError, match="cotangent shape"):
        rule.vjp_rule(values, np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        rule.jvp_rule(values, np.full(8, np.nan))
    np.testing.assert_array_equal(rule.value_fn(values), [6.0, 0.0, 0.0, 20.0])
    assert active_reserved_bytes() == baseline


def test_multi_dot_vector_endpoints_preserve_scalar_value_and_derivatives() -> None:
    """Promoted endpoint vectors retain the independently expanded bilinear form."""
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_multi_dot_derivative_rule(((2,), (2, 2), (2,)))
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    values = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(rule.value_fn(values), [63.0])
    np.testing.assert_array_equal(
        rule.jvp_rule(values, np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])), [39.0]
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(values, np.ones(1)), [15.0, 24.0, 5.0, 6.0, 10.0, 12.0, 3.0, 8.0]
    )
    assert active_reserved_bytes() == baseline


def test_multi_dot_absurd_shapes_refuse_before_numeric_operands_and_recover() -> None:
    """Factory metadata does not allocate huge operands; callbacks refuse checked bytes."""
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_multi_dot_derivative_rule(((2**32, 2**32), (2**32, 2**32)))
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    empty = np.empty(0, dtype=np.float64)
    for callback, arguments in (
        (rule.value_fn, (empty,)),
        (rule.jvp_rule, (empty, empty)),
        (rule.vjp_rule, (empty, empty)),
    ):
        with pytest.raises(DenseAllocationError):
            callback(*arguments)
        assert active_reserved_bytes() == baseline
    with pytest.raises(ValueError, match="dimensions must align"):
        program_ad_linalg_multi_dot_derivative_rule(((2**32, 2), (3, 2**32)))
    valid = program_ad_linalg_multi_dot_derivative_rule(((1,), (1,)))
    np.testing.assert_array_equal(valid.value_fn(np.array([2.0, 3.0])), [6.0])
    assert active_reserved_bytes() == baseline


def test_multi_dot_value_observes_actual_native_return_cancellation() -> None:
    """A real multi-dot return observes inherited cancellation before output conversion."""
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_multi_dot_derivative_rule(((1,), (1,)))
    previous = sys.getprofile()
    cancelled = Event()
    observed = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal observed
        if event != "return" or frame.f_code.co_name != "multi_dot":
            return
        caller = frame.f_back
        while caller is not None:
            if caller.f_code.co_name == "value_fn":
                observed = True
                cancelled.set()
                return
            caller = caller.f_back

    parent = ExecutionMemoryPlan(
        (ExecutionBuffer("multi_dot_parent", "forward", (1,), "float64"),)
    )
    with reserve_execution_memory(parent, cancelled=cancelled):
        sys.setprofile(profile)
        try:
            with pytest.raises(ExecutionCancelledError):
                rule.value_fn(np.array([2.0, 3.0]))
        finally:
            sys.setprofile(previous)
    assert observed
    assert active_reserved_bytes() == baseline
    np.testing.assert_array_equal(rule.value_fn(np.array([2.0, 3.0])), [6.0])
    assert active_reserved_bytes() == baseline


def _multi_dot_scalar_objective(values: Any) -> object:
    """Expand a real vector-matrix-vector objective through the public AD facade."""
    return np.linalg.multi_dot((values[:2], np.reshape(values[2:6], (2, 2)), values[6:]))


def test_public_multi_dot_trace_preserves_scalar_chain_oracle() -> None:
    """Trace materialisation and reverse generation preserve the expanded scalar chain."""
    baseline = active_reserved_bytes()
    value, gradient = program_adjoint_value_and_grad(
        _multi_dot_scalar_objective, [1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0], trace=False
    )
    assert value == 63.0
    np.testing.assert_array_equal(gradient, [15.0, 24.0, 5.0, 6.0, 10.0, 12.0, 3.0, 8.0])
    assert active_reserved_bytes() == baseline


def test_public_multi_dot_trace_refuses_before_numeric_chain_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity refusal at actual trace entry precedes native numeric materialisation.

    Parameters
    ----------
    monkeypatch
        Restore the environment after the observed trace-entry refusal.

    """
    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    entered = False
    native_called = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal entered, native_called
        if event != "call":
            return
        if frame.f_code.co_name == "_trace_multi_dot":
            entered = True
            environment.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024**3))
        elif frame.f_code.co_name == "multi_dot":
            caller = frame.f_back
            while caller is not None:
                if caller.f_code.co_name == "_trace_multi_dot":
                    native_called = True
                    return
                caller = caller.f_back

    with monkeypatch.context() as environment:
        environment.setenv("SCPN_MAX_DENSE_GIB", ".01")
        sys.setprofile(profile)
        try:
            with pytest.raises(DenseAllocationError):
                program_adjoint_value_and_grad(
                    _multi_dot_scalar_objective,
                    [1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0],
                    trace=False,
                )
        finally:
            sys.setprofile(previous)
    assert entered
    assert not native_called
    assert active_reserved_bytes() == baseline
    value, gradient = program_adjoint_value_and_grad(
        _multi_dot_scalar_objective, [1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0], trace=False
    )
    assert value == 63.0
    np.testing.assert_array_equal(gradient, [15.0, 24.0, 5.0, 6.0, 10.0, 12.0, 3.0, 8.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("kind", ["matrix", "left_vector", "right_vector"])
def test_public_multi_dot_trace_preserves_array_output_oracles(kind: str) -> None:
    """Actual traced matrix and endpoint-vector outputs retain analytic gradients.

    Parameters
    ----------
    kind
        Matrix output, left endpoint vector or right endpoint vector signature.

    """
    if kind == "matrix":

        def objective(values: Any) -> object:
            return np.sum(
                np.linalg.multi_dot(
                    (np.reshape(values[:4], (2, 2)), np.reshape(values[4:], (2, 2)))
                )
            )

        values = [2.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0]
        expected_value = 26.0
        expected_gradient = [3.0, 5.0, 3.0, 5.0, 2.0, 2.0, 4.0, 4.0]
    elif kind == "left_vector":

        def objective(values: Any) -> object:
            return np.sum(np.linalg.multi_dot((values[:2], np.reshape(values[2:], (2, 2)))))

        values = [1.0, 2.0, 3.0, 0.0, 0.0, 4.0]
        expected_value = 11.0
        expected_gradient = [3.0, 4.0, 1.0, 1.0, 2.0, 2.0]
    else:

        def objective(values: Any) -> object:
            return np.sum(np.linalg.multi_dot((np.reshape(values[:4], (2, 2)), values[4:])))

        values = [3.0, 0.0, 0.0, 4.0, 1.0, 2.0]
        expected_value = 11.0
        expected_gradient = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    baseline = active_reserved_bytes()
    value, gradient = program_adjoint_value_and_grad(objective, values, trace=False)
    assert value == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("flat", [False, True])
def test_diagonal_construction_callbacks_preserve_offset_differentials(flat: bool) -> None:
    """Constructed off-diagonal matrices retain explicit insertion/extraction derivatives.

    Parameters
    ----------
    flat
        Flatten a ranked source instead of taking a vector source.

    """
    rule = (
        program_ad_linalg_diagflat_derivative_rule((1, 2), k=1)
        if flat
        else program_ad_linalg_diag_derivative_rule((2,), k=1)
    )
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    baseline = active_reserved_bytes()
    values = np.array([2.0, 3.0])
    np.testing.assert_array_equal(
        rule.value_fn(values), [0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        rule.jvp_rule(values, np.array([4.0, 5.0])), [0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(values, np.arange(9, dtype=np.float64)), [1.0, 5.0]
    )
    with pytest.raises(ValueError, match="finite"):
        rule.value_fn(np.array([np.nan, 3.0]))
    np.testing.assert_array_equal(
        rule.value_fn(values), [0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    )
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("negative", [False, True])
def test_diagonal_extraction_callbacks_preserve_rectangular_pullback(negative: bool) -> None:
    """Rectangular extraction and its sparse pullback match explicit coordinates.

    Parameters
    ----------
    negative
        Select a lower diagonal of a tall matrix rather than an upper diagonal.

    """
    baseline = active_reserved_bytes()
    rule = program_ad_linalg_diag_derivative_rule(
        (3, 2) if negative else (2, 3), k=-1 if negative else 1
    )
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    values = np.arange(1, 7, dtype=np.float64)
    np.testing.assert_array_equal(rule.value_fn(values), [3.0, 6.0] if negative else [2.0, 6.0])
    np.testing.assert_array_equal(
        rule.jvp_rule(values, values[::-1]), [4.0, 1.0] if negative else [5.0, 1.0]
    )
    np.testing.assert_array_equal(
        rule.vjp_rule(values, np.array([7.0, 8.0])),
        [0.0, 0.0, 7.0, 0.0, 0.0, 8.0] if negative else [0.0, 7.0, 0.0, 0.0, 0.0, 8.0],
    )
    with pytest.raises(ValueError, match="cotangent size"):
        rule.vjp_rule(values, np.ones(1))
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("flat", [False, True])
def test_diagonal_factory_absurd_shapes_refuse_before_coordinates_or_numeric_arrays(
    flat: bool,
) -> None:
    """Huge static metadata can be inspected without eagerly building coordinates.

    Parameters
    ----------
    flat
        Select diagflat rather than a vector diagonal constructor.

    """
    baseline = active_reserved_bytes()
    rule = (
        program_ad_linalg_diagflat_derivative_rule((2**32,))
        if flat
        else program_ad_linalg_diag_derivative_rule((2**32,))
    )
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    empty = np.empty(0, dtype=np.float64)
    for callback, arguments in (
        (rule.value_fn, (empty,)),
        (rule.jvp_rule, (empty, empty)),
        (rule.vjp_rule, (empty, empty)),
    ):
        with pytest.raises(DenseAllocationError):
            callback(*arguments)
        assert active_reserved_bytes() == baseline
    valid = program_ad_linalg_diag_derivative_rule((1,))
    np.testing.assert_array_equal(valid.value_fn(np.array([2.0])), [2.0])
    assert active_reserved_bytes() == baseline
