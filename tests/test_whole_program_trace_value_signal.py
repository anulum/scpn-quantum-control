# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value signal tests
# scpn-quantum-control -- trace-value signal production contracts
"""Real-surface contracts for trace-value signal and stencil operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import (
    TraceADArray,
    TraceADScalar,
    WholeProgramADResult,
    whole_program_value_and_grad,
)

FloatArray = NDArray[np.float64]
TraceObjective = Callable[[TraceADArray], object]
ArrayFunction = Callable[..., object]


def _differentiate(
    objective: TraceObjective,
    values: FloatArray,
) -> WholeProgramADResult:
    """Execute one signal contract through the public whole-program API."""

    def public_objective(raw_values: object) -> object:
        return objective(cast(TraceADArray, raw_values))

    return whole_program_value_and_grad(public_objective, values, trace=False)


def _dispatch(
    trace_values: TraceADArray,
    function: ArrayFunction,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    """Invoke one public NumPy array-function protocol contract."""
    return trace_values.__array_function__(
        function,
        (TraceADArray,),
        args,
        {} if kwargs is None else kwargs,
    )


def _capture_trace_array(values: FloatArray) -> TraceADArray:
    """Capture the production trace array injected by the public AD API."""
    captured: list[TraceADArray] = []

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(TraceADScalar, trace_values[0] * trace_values[0])

    _differentiate(objective, values)
    assert len(captured) == 1
    return captured[0]


def _capture_empty_trace_array() -> TraceADArray:
    """Capture a valid zero-parameter trace through the public AD API."""
    captured: list[TraceADArray] = []

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(
            TraceADScalar,
            _dispatch(trace_values, np.dot, (trace_values, trace_values)),
        )

    _differentiate(objective, np.array([], dtype=np.float64))
    assert len(captured) == 1
    return captured[0]


def _constant_array(template: TraceADArray, values: Sequence[float]) -> TraceADArray:
    """Build a public constant vector in an already captured trace context."""
    tangent = np.zeros(template.context.parameter_count, dtype=np.float64)
    items = tuple(
        TraceADScalar(float(value), tangent.copy(), template.context, f"constant:{index}")
        for index, value in enumerate(values)
    )
    return TraceADArray(items, (len(items),), template.context)


def _primals(value: object) -> tuple[float, ...]:
    """Return primal values from a public traced scalar or array result."""
    if isinstance(value, TraceADScalar):
        return (value.primal,)
    array = cast(TraceADArray, value)
    return tuple(cast(TraceADScalar, item).primal for item in array.flatten())


def test_trapezoid_supports_full_static_coordinate_arrays() -> None:
    """Trapezoid should differentiate integration over per-row static grids."""

    def objective(values: TraceADArray) -> TraceADScalar:
        matrix = values.reshape((2, 2))
        grid = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        integrated = cast(
            TraceADArray,
            _dispatch(values, np.trapezoid, (matrix, grid), {"axis": 1}),
        )
        return cast(TraceADScalar, integrated.sum())

    result = _differentiate(objective, np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float64))

    assert result.value == pytest.approx(8.0)
    np.testing.assert_array_equal(
        result.gradient,
        np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float64),
    )


def test_trapezoid_rejects_invalid_grids_and_widths() -> None:
    """Trapezoid should reject short axes, dynamic grids, and inconsistent widths."""
    values = _capture_trace_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

    with pytest.raises(ValueError, match="requires at least two samples"):
        _dispatch(values, np.trapezoid, (values[:1],))
    for grid in (values, values[0]):
        with pytest.raises(ValueError, match="grid x must be static real numeric"):
            _dispatch(values, np.trapezoid, (values, grid))
    with pytest.raises(ValueError, match="dx must be finite"):
        _dispatch(values, np.trapezoid, (values,), {"dx": np.inf})
    with pytest.raises(ValueError, match="either x or dx"):
        _dispatch(values, np.trapezoid, (values, (0.0, 1.0, 2.0, 3.0)), {"dx": 2.0})
    with pytest.raises(ValueError, match="only finite values"):
        _dispatch(values, np.trapezoid, (values, (0.0, 1.0, np.nan, 3.0)))
    with pytest.raises(ValueError, match="must match the integration axis"):
        _dispatch(values, np.trapezoid, (values, (0.0, 1.0)))
    with pytest.raises(ValueError, match="integration axis or full array shape"):
        _dispatch(
            values,
            np.trapezoid,
            (values.reshape((2, 2)), np.ones((1, 2), dtype=np.float64)),
            {"axis": 1},
        )


def test_interp_scalar_samples_use_compact_derivative_rules() -> None:
    """Scalar interpolation samples should produce exact compact-rule gradients."""

    def objective(values: TraceADArray) -> TraceADScalar:
        return cast(
            TraceADScalar,
            _dispatch(
                values,
                np.interp,
                (values[0], (0.0, 1.0, 2.0), (0.0, 1.0, 4.0)),
            ),
        )

    result = _differentiate(objective, np.array([0.5], dtype=np.float64))

    assert result.value == pytest.approx(0.5)
    np.testing.assert_array_equal(result.gradient, np.array([1.0], dtype=np.float64))
    assert any(node.op.startswith("interpolation:interp:") for node in result.ir_nodes)


def test_interp_accepts_static_left_and_right_boundaries() -> None:
    """Interpolation should normalize finite static boundary values."""
    values = _capture_trace_array(np.array([0.5], dtype=np.float64))

    interpolated = cast(
        TraceADScalar,
        _dispatch(
            values,
            np.interp,
            (values[0], (0.0, 1.0), (0.0, 2.0)),
            {"left": -1.0, "right": 3.0},
        ),
    )

    assert interpolated.primal == pytest.approx(1.0)


def test_interp_rejects_malformed_values_samples_and_boundaries() -> None:
    """Interpolation should reject malformed ordinates, samples, and dynamic boundaries."""
    values = _capture_trace_array(np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64))
    grid = (0.0, 1.0, 2.0)

    with pytest.raises(ValueError, match="fp values must match xp grid"):
        _dispatch(values, np.interp, (values, grid, values[:2]))
    with pytest.raises(ValueError, match="fp must be one-dimensional"):
        _dispatch(values, np.interp, (values, grid, values[0]))
    with pytest.raises(ValueError, match="fp values must match xp grid"):
        _dispatch(values, np.interp, (values, grid, (0.0, 1.0)))
    with pytest.raises(ValueError, match="fp values must contain only finite values"):
        _dispatch(values, np.interp, (values, grid, (0.0, np.nan, 4.0)))
    with pytest.raises(ValueError, match="x samples must contain only finite values"):
        _dispatch(values, np.interp, ((0.5, np.nan), grid, (0.0, 1.0, 4.0)))
    boundary_cases: tuple[tuple[str, dict[str, object]], ...] = (
        ("left", {"left": values[0]}),
        ("right", {"right": values[:1]}),
    )
    for name, kwargs in boundary_cases:
        with pytest.raises(ValueError, match=rf"{name} boundary must be static real numeric"):
            _dispatch(values, np.interp, (values, grid, (0.0, 1.0, 4.0)), kwargs)


def test_convolve_and_correlate_match_full_mode_calculus() -> None:
    """Full convolution and correlation should preserve exact input derivatives."""

    def objective(values: TraceADArray) -> TraceADScalar:
        kernel = np.array([1.0, 2.0], dtype=np.float64)
        convolved = cast(TraceADArray, _dispatch(values, np.convolve, (values, kernel)))
        correlated = cast(
            TraceADArray,
            _dispatch(values, np.correlate, (values, kernel, "full")),
        )
        return cast(TraceADScalar, convolved.sum() + correlated.sum())

    result = _differentiate(objective, np.array([1.0, 2.0, 3.0], dtype=np.float64))

    assert result.value == pytest.approx(36.0)
    np.testing.assert_array_equal(result.gradient, np.full(3, 6.0, dtype=np.float64))


def test_zero_parameter_signal_rules_preserve_empty_tangents_and_finiteness() -> None:
    """Constant signal rules should preserve zero-width tangents and reject overflow."""
    empty = _capture_empty_trace_array()
    left = _constant_array(empty, (1.0, 2.0))
    right = _constant_array(empty, (3.0, 4.0))

    convolved = cast(
        TraceADArray,
        _dispatch(empty, np.convolve, (left, right)),
    )
    assert _primals(convolved) == (3.0, 10.0, 8.0)
    assert all(cast(TraceADScalar, item).tangent.shape == (0,) for item in convolved)

    huge = _constant_array(empty, (1.0e200,))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="compact outputs must be finite"),
    ):
        _dispatch(empty, np.convolve, (huge, huge))


@pytest.mark.parametrize("function", [np.convolve, np.correlate])
def test_signal_operations_reject_invalid_operands(function: ArrayFunction) -> None:
    """Signal operations should reject scalar, ranked, empty, and non-finite operands."""
    values = _capture_trace_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
    operation = function.__name__

    for operand in (values[0], values.reshape((2, 2)), np.array([[1.0, 2.0]])):
        with pytest.raises(ValueError, match=rf"signal {operation} shape rule requires rank-1"):
            _dispatch(values, function, (values, operand))
    for operand in (values[:0], np.array([], dtype=np.float64)):
        with pytest.raises(ValueError, match=rf"signal {operation} shape rule requires non-empty"):
            _dispatch(values, function, (values, operand))
    with pytest.raises(
        ValueError, match=rf"np\.{operation} right operand must contain only finite"
    ):
        _dispatch(values, function, (values, np.array([1.0, np.nan])))


def test_diff_zero_and_exhausted_orders_preserve_static_shapes() -> None:
    """Diff should copy order zero and return a typed empty array beyond axis length."""
    values = _capture_trace_array(np.array([1.0, 2.0, 4.0], dtype=np.float64))

    unchanged = cast(TraceADArray, _dispatch(values, np.diff, (values, 0)))
    exhausted = cast(TraceADArray, _dispatch(values, np.diff, (values, 3)))

    assert unchanged.shape == (3,)
    assert _primals(unchanged) == (1.0, 2.0, 4.0)
    assert exhausted.shape == (0,)
    assert exhausted.size == 0
    for order in (1.5, -1):
        with pytest.raises(ValueError, match="requires non-negative integer n"):
            _dispatch(values, np.diff, (values, order))


def test_empty_cumulative_operations_fail_closed() -> None:
    """Cumulative sums and products should reject empty traced arrays explicitly."""
    empty = _capture_empty_trace_array()

    with pytest.raises(ValueError, match="cumulative scan requires at least one element"):
        _dispatch(empty, np.cumsum, (empty,))
    with pytest.raises(ValueError, match="cumulative scan requires at least one element"):
        _dispatch(empty, np.cumprod, (empty,))
