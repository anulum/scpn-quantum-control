# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value operator tests
# scpn-quantum-control -- trace-value operator production contracts
"""Real-surface contracts for whole-program trace-value operators."""

from __future__ import annotations

from collections.abc import Callable
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
NumpyReduction = Callable[[object], object]


def _where(
    dispatch: TraceADArray,
    condition: object,
    when_true: object,
    when_false: object,
) -> object:
    """Call NumPy where through the public trace-array function protocol."""
    return dispatch.__array_function__(
        np.where,
        (TraceADArray,),
        (condition, when_true, when_false),
        {},
    )


def _sum(values: object) -> object:
    """Call NumPy sum through its runtime array-function protocol."""
    return cast(NumpyReduction, np.sum)(values)


def _differentiate(
    objective: TraceObjective,
    values: FloatArray | None = None,
) -> WholeProgramADResult:
    """Execute one trace-value contract through the public whole-program API."""

    def public_objective(raw_values: object) -> object:
        return objective(cast(TraceADArray, raw_values))

    parameters = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64) if values is None else values
    return whole_program_value_and_grad(public_objective, parameters, trace=False)


def _capture_trace_array(values: FloatArray | None = None) -> TraceADArray:
    """Capture the real trace array injected by the public whole-program API."""
    captured: list[TraceADArray] = []

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(TraceADScalar, trace_values[0] * trace_values[0])

    _differentiate(objective, values)
    assert len(captured) == 1
    return captured[0]


def test_scalar_predicates_preserve_executed_derivatives() -> None:
    """Scalar equality, inequality, and ordering should select traced values."""

    def objective(values: TraceADArray) -> TraceADScalar:
        left = cast(TraceADScalar, values[0])
        right = cast(TraceADScalar, values[1])
        equal = cast(TraceADScalar, _where(values, left == left, left, right))
        unequal = cast(TraceADScalar, _where(values, left != right, left, right))
        ordered = cast(TraceADScalar, _where(values, left < right, right, left))
        return equal + unequal + ordered

    result = _differentiate(
        objective,
        np.array([2.0, 3.0], dtype=np.float64),
    )

    assert result.value == pytest.approx(7.0)
    np.testing.assert_array_equal(result.gradient, np.array([2.0, 1.0], dtype=np.float64))


def test_array_predicates_preserve_elementwise_derivatives() -> None:
    """Array equality and ordering should retain elementwise trace ownership."""

    def objective(values: TraceADArray) -> TraceADScalar:
        equal = cast(TraceADArray, _where(values, values == values, values, -values))
        unequal = cast(TraceADArray, _where(values, values != -values, values, -values))
        ordered = cast(
            TraceADArray,
            _where(values, values <= values + 1.0, values, -values),
        )
        return cast(TraceADScalar, _sum(equal + unequal + ordered))

    result = _differentiate(
        objective,
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    assert result.value == pytest.approx(18.0)
    np.testing.assert_array_equal(result.gradient, np.full(3, 3.0, dtype=np.float64))


def test_reflected_array_operators_match_numpy_calculus() -> None:
    """Reflected arithmetic and matrix multiplication should preserve gradients."""

    def objective(values: TraceADArray) -> TraceADScalar:
        reflected = 2.0 + values + (2.0 - values) + 2.0 / values + 2.0**values
        matrix_product = np.array([[1.0, 2.0]], dtype=np.float64) @ values.reshape((2, 1))
        reflected_sum = cast(TraceADScalar, _sum(reflected))
        matrix_sum = cast(TraceADScalar, _sum(matrix_product))
        return reflected_sum + matrix_sum

    values = np.array([1.0, 2.0], dtype=np.float64)
    result = _differentiate(objective, values)
    expected_gradient = (
        -2.0 / values**2
        + np.log(2.0) * np.power(2.0, values)
        + np.array([1.0, 2.0], dtype=np.float64)
    )

    assert result.value == pytest.approx(22.0)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1.0e-12)


def test_public_array_construction_rejects_invalid_metadata() -> None:
    """The public trace-array constructor should fail closed on inconsistent state."""
    first = _capture_trace_array()
    second = _capture_trace_array(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64))
    scalar = cast(TraceADScalar, first[0])
    foreign_scalar = cast(TraceADScalar, second[0])

    with pytest.raises(ValueError, match="scalar TraceADArray requires exactly one item"):
        TraceADArray((), (), first.context)
    with pytest.raises(ValueError, match="shape must match item count"):
        TraceADArray((scalar,), (2,), first.context)
    with pytest.raises(ValueError, match="items must belong to the same trace"):
        TraceADArray((foreign_scalar,), (1,), first.context)
    with pytest.raises(ValueError, match="source indices must match item count"):
        TraceADArray((scalar,), (1,), first.context, ())
    with pytest.raises(ValueError, match="source indices must be non-negative or None"):
        TraceADArray((scalar,), (1,), first.context, (-1,))


def test_array_container_protocols_preserve_rank_and_derivatives() -> None:
    """Length, iteration, transpose, flatten, and sum should preserve trace values."""
    values = _capture_trace_array()
    scalar_array = TraceADArray(
        (cast(TraceADScalar, values[0]),),
        (),
        values.context,
    )

    assert len(values) == 4
    assert [cast(TraceADScalar, item).primal for item in values] == [1.0, 2.0, 3.0, 4.0]
    rows = list(values.reshape((2, 2)))
    assert [cast(TraceADArray, row).shape for row in rows] == [(2,), (2,)]
    assert values.flatten().shape == (4,)
    assert values.T.shape == (4,)
    assert cast(TraceADScalar, values.sum()).primal == pytest.approx(10.0)
    assert scalar_array.item().primal == pytest.approx(1.0)

    with pytest.raises(TypeError, match=r"has no len\(\)"):
        len(scalar_array)
    with pytest.raises(ValueError, match="rank <= 2"):
        list(values.reshape((1, 2, 2)))
    with pytest.raises(ValueError, match="requires exactly one element"):
        values.item()
    with pytest.raises(ValueError, match="cannot be converted to a NumPy ndarray"):
        np.asarray(values)
    with pytest.raises(ValueError, match="index selection semantics"):
        values.argmax()


def test_ufunc_protocols_reject_derivative_losing_invocations() -> None:
    """Scalar and array ufunc protocols should reject methods, outputs, and rank drift."""
    values = _capture_trace_array()
    scalar = cast(TraceADScalar, values[0])

    with pytest.raises(ValueError, match="direct NumPy array ufunc calls"):
        values.__array_ufunc__(np.add, "reduce", values)
    with pytest.raises(ValueError, match="direct NumPy scalar ufunc calls"):
        scalar.__array_ufunc__(np.add, "__call__", scalar, out=None)
    with pytest.raises(ValueError, match="scalar ufunc returned a non-scalar result"):
        scalar.__array_ufunc__(np.add, "__call__", scalar, values)


def test_cross_trace_operators_and_predicates_fail_closed() -> None:
    """Values and predicates from independent public traces should never be combined."""
    first = _capture_trace_array()
    second = _capture_trace_array(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64))
    first_scalar = cast(TraceADScalar, first[0])
    second_scalar = cast(TraceADScalar, second[0])

    with pytest.raises(ValueError, match="scalars belong to different traces"):
        first_scalar + second_scalar
    with pytest.raises(ValueError, match="arrays belong to different traces"):
        first + second
    with pytest.raises(ValueError, match="scalars belong to different traces"):
        first + second_scalar
    with pytest.raises(ValueError, match="predicate belongs to a different trace"):
        _where(first, second_scalar > 0.0, first, first)
    with pytest.raises(ValueError, match="scalars belong to different traces"):
        first[:2] = second_scalar
    with pytest.raises(ValueError, match="arrays belong to different traces"):
        first[:2] = second[:2]


def test_slice_mutation_accepts_public_value_forms() -> None:
    """Slice mutation should accept traced scalars, arrays, and static numeric values."""
    values = _capture_trace_array()
    scalar_array = TraceADArray(
        (cast(TraceADScalar, values[2]),),
        (),
        values.context,
    )

    values[1:1] = values
    values[:2] = scalar_array
    assert [cast(TraceADScalar, values[index]).primal for index in range(2)] == [3.0, 3.0]
    values[:2] = values[2:]
    assert [cast(TraceADScalar, values[index]).primal for index in range(2)] == [3.0, 4.0]
    values[:2] = cast(TraceADScalar, values[2])
    assert [cast(TraceADScalar, values[index]).primal for index in range(2)] == [3.0, 3.0]
    values[:2] = 7.0
    assert [cast(TraceADScalar, values[index]).primal for index in range(2)] == [7.0, 7.0]
    values[:2] = np.array([8.0, 9.0], dtype=np.float64)
    assert [cast(TraceADScalar, values[index]).primal for index in range(2)] == [8.0, 9.0]


def test_mutation_rejects_invalid_rank_shape_and_index_contracts() -> None:
    """Mutation should reject unsupported ranks, mismatched values, and invalid indices."""
    values = _capture_trace_array()
    matrix = values.reshape((2, 2))
    tensor = values.reshape((1, 2, 2))

    with pytest.raises(ValueError, match="rank <= 2"):
        tensor[0] = values[0]
    with pytest.raises(ValueError, match="slice mutation supports rank-1 arrays"):
        matrix[:] = values
    with pytest.raises(ValueError, match="value length must match target length"):
        values[:3] = values[:2]
    with pytest.raises(ValueError, match="value length must match target length"):
        values[:3] = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="expects two integer indices"):
        matrix[(0,)] = values[0]
    with pytest.raises(ValueError, match="expects two integer indices"):
        values[(0, 0)] = values[0]
    with pytest.raises(ValueError, match="integer or slice indices"):
        values[object()] = values[0]
    with pytest.raises(ValueError, match="index out of bounds"):
        values[-5] = values[0]
    with pytest.raises(ValueError, match="index out of bounds"):
        values[4] = values[0]

    matrix[-1, -1] = values[0]
    assert cast(TraceADScalar, matrix[1, 1]).primal == pytest.approx(1.0)


def test_single_item_mutation_coerces_arrays_and_rejects_foreign_contexts() -> None:
    """Single-item mutation should coerce singleton arrays within exactly one trace."""
    first = _capture_trace_array()
    second = _capture_trace_array(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64))

    first[0] = first[1:2]
    assert cast(TraceADScalar, first[0]).primal == 2.0
    with pytest.raises(ValueError, match="arrays belong to different traces"):
        first[0] = second[:1]

    standalone = TraceADArray((cast(TraceADScalar, first[1]),), (1,), first.context)
    standalone[0] = first[2]
    assert cast(TraceADScalar, standalone[0]).primal == 3.0


def test_static_array_coercion_rejects_nonreal_and_nonfinite_operands() -> None:
    """Elementwise public calls should reject non-real and non-finite constants."""
    values = _capture_trace_array()

    with pytest.raises(ValueError, match="must be real numeric"):
        _ = values + np.array(["invalid"], dtype=np.str_)
    with pytest.raises(ValueError, match="must be finite"):
        _ = values + np.array([np.inf], dtype=np.float64)


def test_reflected_matmul_and_scalar_axis_reductions_preserve_calculus() -> None:
    """Reflected matmul and one-axis reductions should return traced scalars."""
    values = _capture_trace_array()
    vector = cast(TraceADArray, values[:2])

    reflected = cast(
        TraceADArray,
        vector.__rmatmul__(np.array([[2.0, 3.0]], dtype=np.float64)),
    )
    sum_value = vector.__array_function__(np.sum, (TraceADArray,), (vector,), {"axis": 0})
    product_value = vector.__array_function__(np.prod, (TraceADArray,), (vector,), {"axis": 0})
    variance_value = vector.__array_function__(np.var, (TraceADArray,), (vector,), {"axis": 0})
    assert cast(TraceADScalar, reflected[0]).primal == 8.0
    assert cast(TraceADScalar, sum_value).primal == 3.0
    assert cast(TraceADScalar, product_value).primal == 2.0
    assert cast(TraceADScalar, variance_value).primal == pytest.approx(0.25)


def test_sqrt_fails_closed_at_nonpositive_boundary() -> None:
    """Square root should reject the nonpositive derivative boundary."""
    values = _capture_trace_array(np.array([-1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="sqrt input must be positive"):
        np.sqrt(values)
