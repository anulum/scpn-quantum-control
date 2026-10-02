# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — unbuffered trace scatter tests
"""Exercise native NumPy scatter dispatch through the public AD runtime."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import FrameType
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control import TraceADArray, TraceADScalar, whole_program_value_and_grad
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
from scpn_quantum_control.execution_reservations import active_reserved_bytes


@pytest.mark.parametrize("index_operand", [False, True])
def test_scatter_storage_admission_does_not_call_opaque_type_protocols(
    index_operand: bool,
) -> None:
    """Malformed public protocol operands refuse without metaclass dispatch.

    Parameters
    ----------
    index_operand
        Whether the opaque storage is supplied as indices or updates.

    """
    calls: list[str] = []

    class ProtocolMeta(type):
        def __hash__(cls) -> int:
            calls.append("hash")
            raise AssertionError("opaque metaclass hash was called")

        def __eq__(cls, other: object) -> bool:
            calls.append("equality")
            raise AssertionError("opaque metaclass equality was called")

    class Opaque(metaclass=ProtocolMeta):
        def __getattribute__(self, name: str) -> object:
            calls.append(name)
            raise AssertionError("opaque attribute protocol was called")

    opaque = Opaque()

    def objective(values: TraceADArray) -> object:
        indices = opaque if index_operand else [0]
        updates = 2.0 if index_operand else opaque
        values.__array_ufunc__(np.add, "at", values, indices, updates)
        return values.sum()

    inputs = np.array([1.0, 2.0])
    reason = "static" if index_operand else "matching rank-one"
    with pytest.raises(ValueError, match=reason):
        whole_program_value_and_grad(objective, inputs, trace=False)
    assert calls == []
    np.testing.assert_array_equal(inputs, [1.0, 2.0])


@pytest.mark.parametrize(
    ("indices", "updates", "expected_value", "expected_gradient"),
    [
        ([0, 0], 2.0, 38.0, (10.0, 4.0, 6.0)),
        ((-1, -1), (1.0, 2.0), 41.0, (2.0, 4.0, 12.0)),
        ([], [], 14.0, (2.0, 4.0, 6.0)),
        ([], 2.0, 14.0, (2.0, 4.0, 6.0)),
        (np.int64(1), (2.0,), 26.0, (2.0, 8.0, 6.0)),
        (np.array([0, 2], dtype=np.int32), np.array([2.0, 1.0]), 29.0, (6.0, 4.0, 8.0)),
        ([1], np.array(2.0), 26.0, (2.0, 8.0, 6.0)),
        ([1], np.int16(2), 26.0, (2.0, 8.0, 6.0)),
    ],
)
def test_static_scatter_matches_native_values_and_analytic_derivatives(
    indices: object,
    updates: object,
    expected_value: float,
    expected_gradient: tuple[float, float, float],
) -> None:
    """Static indices, broadcasting and negative targets preserve native semantics.

    Parameters
    ----------
    indices
        Exact static index storage passed to native ``np.add.at`` dispatch.
    updates
        Static finite scalar or rank-one updates.
    expected_value
        Independent quadratic value after sequential accumulation.
    expected_gradient
        Analytic derivative with respect to original destination parameters.

    """

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        scatter = cast(Callable[[object, object, object], None], np.add.at)
        scatter(working, indices, updates)
        return cast(TraceADArray, working**2).sum()

    inputs = np.array([1.0, 2.0, 3.0])
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == pytest.approx(expected_value, abs=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result), expected_gradient, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_array_equal(inputs, [1.0, 2.0, 3.0])


def test_aliasing_update_array_is_snapshotted_before_destination_writes() -> None:
    """An overlapping RHS uses its original values for each repeated destination."""

    def objective(values: TraceADArray) -> object:
        np.add.at(values, [1, 0, 1], values)
        return cast(TraceADArray, values**2).sum()

    inputs = np.array([1.0, 2.0, 3.0])
    reference = inputs.copy()
    np.add.at(reference, [1, 0, 1], reference)
    result = whole_program_value_and_grad(objective, inputs, trace=False)
    assert result.value == float(np.sum(reference**2))
    np.testing.assert_array_equal(reference, [3.0, 6.0, 3.0])
    np.testing.assert_array_equal(result.gradient, [18.0, 18.0, 18.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), result.gradient)
    np.testing.assert_array_equal(inputs, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("scalar_array", [False, True])
def test_trace_scalar_broadcasts_without_detaching_its_derivative(scalar_array: bool) -> None:
    """Scalar trace values and zero-dimensional trace arrays accumulate twice.

    Parameters
    ----------
    scalar_array
        Whether the RHS is a scalar trace value or a zero-dimensional trace view.

    """

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        update = cast(TraceADArray, values[:1]).reshape(()) if scalar_array else values[0]
        scatter = cast(Callable[[object, object, object], None], np.add.at)
        scatter(working, [1, 1], update)
        return cast(TraceADArray, working**2).sum()

    result = whole_program_value_and_grad(objective, np.array([1.0, 2.0, 3.0]), trace=False)
    assert result.value == 26.0
    np.testing.assert_array_equal(result.gradient, [18.0, 8.0, 6.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), result.gradient)


@pytest.mark.parametrize(
    ("indices", "updates", "reason"),
    [
        ([0, 3], [1.0, 2.0], "out of bounds"),
        ([-4], [1.0], "out of bounds"),
        ([True], [1.0], "static integers"),
        ([0.5], [1.0], "static integers"),
        (np.array([0.0]), [1.0], "static rank-one integers"),
        (np.array([[0]]), [1.0], "static rank-one integers"),
        ([0, 1], [2.0], "matching rank-one"),
        ([0], [np.nan], "finite real scalars"),
        ([], np.nan, "finite real scalars"),
        ([0], [np.inf], "finite real scalars"),
        ([0], [True], "finite real scalars"),
        ([0], [1.0j], "finite real scalars"),
        ([0], np.array([1.0j]), "matching rank-one"),
        ([0], np.array([[1.0]]), "matching rank-one"),
        ([0], object(), "matching rank-one"),
        ([0], 10**1000, "finite real scalars"),
        ([0], [10**1000], "finite real scalars"),
        ([0], np.longdouble(np.finfo(np.longdouble).max), "finite real scalars"),
    ],
)
def test_invalid_scatter_operands_refuse_without_changing_caller_input(
    indices: object, updates: object, reason: str
) -> None:
    """Malformed storage, shapes and scalars fail at the public operation boundary.

    Parameters
    ----------
    indices
        Invalid or otherwise supported static target indices.
    updates
        Invalid or otherwise supported update storage.
    reason
        Authored refusal text for the rejected operand contract.

    """

    def objective(values: TraceADArray) -> object:
        working = values.copy()
        scatter = cast(Callable[[object, object, object], None], np.add.at)
        scatter(working, indices, updates)
        return working.sum()

    inputs = np.array([1.0, 2.0, 3.0])
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match=reason):
        whole_program_value_and_grad(objective, inputs, trace=False)
    np.testing.assert_array_equal(inputs, [1.0, 2.0, 3.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("signature", ["missing_update", "keyword", "rank_two"])
def test_scatter_protocol_refuses_invalid_call_shapes_before_writes(signature: str) -> None:
    """Invalid public ufunc signatures release reservations and preserve input.

    Parameters
    ----------
    signature
        Missing update, unsupported keyword or non-vector destination.

    """

    def objective(values: TraceADArray) -> object:
        if signature == "missing_update":
            values.__array_ufunc__(np.add, "at", values, [0])
        elif signature == "keyword":
            values.__array_ufunc__(np.add, "at", values, [0], 2.0, where=True)
        else:
            destination = values.reshape((1, 3))
            destination.__array_ufunc__(np.add, "at", destination, [0], 2.0)
        return values.sum()

    inputs = np.array([1.0, 2.0, 3.0])
    baseline = active_reserved_bytes()
    reason = "rank-one trace destination" if signature == "rank_two" else "without keywords"
    with pytest.raises(ValueError, match=reason):
        whole_program_value_and_grad(objective, inputs, trace=False)
    np.testing.assert_array_equal(inputs, [1.0, 2.0, 3.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("storage", ["scalar", "array", "nested_scalar", "rank_two"])
def test_trace_scatter_update_refusal_preserves_destination_and_historical_tape(
    storage: str,
) -> None:
    """Public scatter rejects foreign traces and wrong ranks before any write.

    Parameters
    ----------
    storage
        Foreign trace scalar, vector, nested scalar or same-trace matrix.

    """
    captured: list[TraceADArray] = []

    def objective(values: TraceADArray) -> object:
        return cast(TraceADArray, values**2).sum()

    def observe(frame: FrameType, event: str, argument: object) -> None:
        """Observe actual runtime operands through Python's public profiling API."""
        if event == "call" and frame.f_code is objective.__code__:
            values = frame.f_locals["values"]
            if type(values) is TraceADArray:
                captured.append(values)

    baseline = active_reserved_bytes()
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        first = whole_program_value_and_grad(objective, [1.0, 2.0, 3.0], trace=False)
        second = whole_program_value_and_grad(objective, [4.0, 5.0, 6.0], trace=False)
    finally:
        sys.setprofile(previous)
    assert sys.getprofile() is previous
    assert len(captured) == 2
    destination, foreign = captured
    updates: object
    indices: list[int]
    if storage == "array":
        updates, indices = foreign, [0, 0, 1]
    elif storage == "rank_two":
        updates, indices = destination.reshape((1, 3)), [0, 0, 1]
    elif storage == "nested_scalar":
        updates, indices = [destination[0], foreign[0]], [0, 1]
    else:
        updates, indices = foreign[0], [0]
    reason = "matching rank-one" if storage == "rank_two" else "different trace"
    with pytest.raises(ValueError, match=reason):
        destination.__array_ufunc__(np.add, "at", destination, indices, updates)
    assert tuple(cast(TraceADScalar, destination[i]).primal for i in range(3)) == (1.0, 2.0, 3.0)
    assert tuple(cast(TraceADScalar, foreign[i]).primal for i in range(3)) == (4.0, 5.0, 6.0)
    np.testing.assert_array_equal(program_adjoint_replay_gradient(first), [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(second), [8.0, 10.0, 12.0])
    assert active_reserved_bytes() == baseline


def test_other_inplace_ufuncs_remain_explicitly_unsupported() -> None:
    """Multiplicative in-place updates cannot inherit additive scatter semantics."""

    def objective(values: TraceADArray) -> object:
        np.multiply.at(values, [0], 2.0)
        return values.sum()

    with pytest.raises(ValueError, match="np.add.at"):
        whole_program_value_and_grad(objective, np.array([1.0, 2.0, 3.0]), trace=False)
