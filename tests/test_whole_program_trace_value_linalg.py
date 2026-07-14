# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program trace-value linalg tests
# scpn-quantum-control -- trace-value linalg production contracts
"""Real-surface contracts for whole-program trace-value linear algebra."""

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
    """Execute one linalg contract through the public whole-program API."""

    def public_objective(raw_values: object) -> object:
        return objective(cast(TraceADArray, raw_values))

    return whole_program_value_and_grad(public_objective, values, trace=False)


def _capture_trace_array(values: FloatArray) -> TraceADArray:
    """Capture the production trace array injected by the public AD API."""
    captured: list[TraceADArray] = []

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(TraceADScalar, trace_values[0])

    _differentiate(objective, values)
    assert len(captured) == 1
    return captured[0]


def _capture_empty_trace_array() -> TraceADArray:
    """Capture the valid zero-parameter trace context through the public API."""
    captured: list[TraceADArray] = []

    def objective(trace_values: TraceADArray) -> TraceADScalar:
        captured.append(trace_values)
        return cast(
            TraceADScalar,
            trace_values.__array_function__(
                np.dot,
                (TraceADArray,),
                (trace_values, trace_values),
                {},
            ),
        )

    _differentiate(objective, np.array([], dtype=np.float64))
    assert len(captured) == 1
    return captured[0]


def _constant_array(
    template: TraceADArray,
    values: Sequence[float],
    shape: tuple[int, ...],
) -> TraceADArray:
    """Build a public constant trace array in an already captured context."""
    flat_values = tuple(float(value) for value in values)
    assert len(flat_values) == int(np.prod(shape))
    tangent = np.zeros(template.context.parameter_count, dtype=np.float64)
    items = tuple(
        TraceADScalar(value, tangent.copy(), template.context, f"constant:{index}")
        for index, value in enumerate(flat_values)
    )
    return TraceADArray(items, shape, template.context)


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


def _primals(value: object) -> tuple[float, ...]:
    """Return primal values from a traced scalar or array result."""
    if isinstance(value, TraceADScalar):
        return (value.primal,)
    array = cast(TraceADArray, value)
    return tuple(cast(TraceADScalar, item).primal for item in array.flatten())


def test_scalar_inner_broadcasts_and_empty_vdot_returns_zero() -> None:
    """Inner should multiply scalar arrays and vdot should define the empty identity."""
    values = _capture_trace_array(np.array([2.0, 3.0], dtype=np.float64))
    scalar = TraceADArray((cast(TraceADScalar, values[0]),), (), values.context)

    assert _primals(_dispatch(values, np.inner, (scalar, values))) == (4.0, 6.0)
    assert _primals(_dispatch(values, np.inner, (values, scalar))) == (4.0, 6.0)

    empty = _capture_empty_trace_array()
    assert _primals(_dispatch(empty, np.vdot, (empty, empty))) == (0.0,)


def test_empty_square_det_inv_and_matrix_power_have_numpy_identities() -> None:
    """Empty square matrices should preserve NumPy determinant and power identities."""
    empty = _capture_empty_trace_array()
    matrix = TraceADArray((), (0, 0), empty.context)

    determinant = cast(TraceADScalar, _dispatch(empty, np.linalg.det, (matrix,)))
    inverse = cast(TraceADArray, _dispatch(empty, np.linalg.inv, (matrix,)))
    powered = cast(
        TraceADArray,
        _dispatch(empty, np.linalg.matrix_power, (matrix, 2)),
    )

    assert determinant.primal == 1.0
    assert determinant.tangent.shape == (0,)
    assert inverse.shape == (0, 0)
    assert powered.shape == (0, 0)


def test_zero_parameter_solve_and_multi_dot_preserve_empty_tangents() -> None:
    """Constant linalg calls should produce correctly shaped zero-width tangents."""
    empty = _capture_empty_trace_array()
    matrix = _constant_array(empty, (2.0, 0.0, 0.0, 4.0), (2, 2))
    vector = _constant_array(empty, (2.0, 8.0), (2,))
    rhs_matrix = _constant_array(empty, (2.0, 4.0, 8.0, 12.0), (2, 2))

    vector_solution = cast(
        TraceADArray,
        _dispatch(empty, np.linalg.solve, (matrix, vector)),
    )
    matrix_solution = cast(
        TraceADArray,
        _dispatch(empty, np.linalg.solve, (matrix, rhs_matrix)),
    )
    scalar_product = cast(
        TraceADScalar,
        _dispatch(empty, np.linalg.multi_dot, ((vector, matrix, vector),)),
    )
    matrix_product = cast(
        TraceADArray,
        _dispatch(empty, np.linalg.multi_dot, ((matrix, rhs_matrix),)),
    )

    assert _primals(vector_solution) == (1.0, 2.0)
    assert _primals(matrix_solution) == (1.0, 2.0, 2.0, 3.0)
    assert scalar_product.primal == 264.0
    assert _primals(matrix_product) == (4.0, 8.0, 32.0, 48.0)
    for result in (*vector_solution.flatten(), *matrix_solution.flatten(), scalar_product):
        assert cast(TraceADScalar, result).tangent.shape == (0,)


def test_matrix_power_and_multi_dot_match_public_calculus() -> None:
    """Matrix powers and chained products should expose exact public gradients."""

    def objective(values: TraceADArray) -> TraceADScalar:
        matrix = cast(TraceADArray, values[:4]).reshape((2, 2))
        vector = values[4:]
        squared = cast(
            TraceADArray,
            _dispatch(values, np.linalg.matrix_power, (matrix, 2)),
        )
        inverted = cast(
            TraceADArray,
            _dispatch(values, np.linalg.matrix_power, (matrix, -1)),
        )
        chained = cast(
            TraceADScalar,
            _dispatch(values, np.linalg.multi_dot, ((vector, matrix, vector),)),
        )
        return cast(TraceADScalar, squared.sum() + inverted.sum() + chained)

    parameters = np.array([2.0, 0.0, 0.0, 4.0, 1.0, 2.0], dtype=np.float64)
    result = _differentiate(objective, parameters)

    def reference(raw: FloatArray) -> float:
        matrix = raw[:4].reshape(2, 2)
        vector = raw[4:]
        return float(
            np.sum(np.linalg.matrix_power(matrix, 2))
            + np.sum(np.linalg.matrix_power(matrix, -1))
            + np.linalg.multi_dot((vector, matrix, vector))
        )

    epsilon = 1.0e-6
    expected = np.array(
        [
            (
                reference(parameters + np.eye(1, parameters.size, index).reshape(-1) * epsilon)
                - reference(parameters - np.eye(1, parameters.size, index).reshape(-1) * epsilon)
            )
            / (2.0 * epsilon)
            for index in range(parameters.size)
        ],
        dtype=np.float64,
    )

    assert result.value == pytest.approx(reference(parameters), rel=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected, rtol=1.0e-7, atol=1.0e-7)


@pytest.mark.parametrize(
    ("function", "values", "message"),
    [
        (np.linalg.det, (1.0e308, 0.0, 0.0, 1.0e308), "finite determinant"),
        (np.linalg.inv, (1.0e-320, 0.0, 0.0, 1.0e-320), "nonsingular matrix"),
    ],
)
def test_factorisations_fail_closed_on_nonfinite_numeric_results(
    function: ArrayFunction,
    values: tuple[float, float, float, float],
    message: str,
) -> None:
    """Factorisations should reject overflowed outputs from finite inputs."""
    trace_values = _capture_trace_array(np.array(values, dtype=np.float64))
    matrix = trace_values.reshape((2, 2))

    with (
        np.errstate(over="ignore", divide="ignore", invalid="ignore"),
        pytest.raises(ValueError, match=message),
    ):
        _dispatch(trace_values, function, (matrix,))


def test_solve_matrix_power_and_multi_dot_reject_nonfinite_outputs() -> None:
    """Linalg primitives should fail closed when finite operands overflow."""
    values = _capture_trace_array(np.array([1.0e-308, 0.0, 0.0, 1.0e-308], dtype=np.float64))
    matrix = values.reshape((2, 2))
    rhs = np.array([2.0, 2.0], dtype=np.float64)

    with (
        np.errstate(over="ignore", divide="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite solution"),
    ):
        _dispatch(values, np.linalg.solve, (matrix, rhs))

    large = _capture_trace_array(np.array([1.0e200, 0.0, 0.0, 1.0e200], dtype=np.float64))
    large_matrix = large.reshape((2, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite outputs"):
            _dispatch(large, np.linalg.matrix_power, (large_matrix, 2))
        with pytest.raises(ValueError, match="finite outputs"):
            _dispatch(large, np.linalg.multi_dot, ((large_matrix, large_matrix),))


def test_linalg_registry_rejects_invalid_shapes_before_execution() -> None:
    """The public registry should own linalg shape validation consistently."""
    values = _capture_trace_array(np.arange(1.0, 7.0, dtype=np.float64))
    matrix = cast(TraceADArray, values[:4]).reshape((2, 2))

    invalid_cases: tuple[tuple[ArrayFunction, tuple[object, ...], str], ...] = (
        (np.linalg.det, (values[:3],), "shape rule requires a rank-2 matrix"),
        (np.linalg.inv, (values.reshape((2, 3)),), "requires a square matrix"),
        (np.linalg.solve, (matrix, values[:3]), "vector length must match matrix"),
        (
            np.linalg.matrix_power,
            (matrix, 1.5),
            "requires an integer power",
        ),
        (
            np.linalg.multi_dot,
            ((values[:2], values[2:4], values[4:]),),
            "middle operands must be rank-2",
        ),
        (np.linalg.eig, (values[:3],), "shape rule requires a rank-2 matrix"),
        (np.linalg.eigvalsh, (values.reshape((2, 3)),), "requires a square matrix"),
        (np.linalg.svd, (values[:3], False, False), "requires a rank-2 matrix"),
        (np.linalg.pinv, (values[:3],), "requires a rank-2 matrix"),
    )

    for function, args, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _dispatch(values, function, args)


def test_multi_dot_registry_rejects_nonsequences_rank_and_alignment() -> None:
    """Multi-dot should fail closed at its static signature boundary."""
    values = _capture_trace_array(np.arange(1.0, 9.0, dtype=np.float64))
    matrix = cast(TraceADArray, values[:4]).reshape((2, 2))
    ranked = values.reshape((2, 2, 2))

    cases: tuple[tuple[object, str], ...] = (
        (values, "requires a static operand sequence"),
        (object(), "requires a static operand sequence"),
        ((matrix,), "requires at least two operands"),
        ((ranked, matrix), "supports rank-1 and rank-2 operands"),
        ((matrix, values[:3]), "dimensions must align"),
    )
    for operands, message in cases:
        with pytest.raises(ValueError, match=message):
            _dispatch(values, np.linalg.multi_dot, (operands,))


def test_direct_matmul_and_einsum_shortcuts_cover_ranked_products() -> None:
    """Direct protocol matmul and einsum shortcuts should preserve ranked products."""
    values = _capture_trace_array(np.arange(1.0, 9.0, dtype=np.float64))
    vector = cast(TraceADArray, values[:2])
    matrix = cast(TraceADArray, values[2:6]).reshape((2, 2))

    matrix_vector = _dispatch(values, np.matmul, (matrix, vector))
    vector_matrix = _dispatch(values, np.einsum, ("i,ij->j", vector, matrix))
    matrix_matrix = _dispatch(values, np.einsum, ("ij,jk->ik", matrix, matrix))
    diagonal = _dispatch(values, np.einsum, ("ii->i", matrix))

    assert _primals(matrix_vector) == (11.0, 17.0)
    assert _primals(vector_matrix) == (13.0, 16.0)
    assert _primals(matrix_matrix) == (29.0, 36.0, 45.0, 56.0)
    assert _primals(diagonal) == (3.0, 6.0)


def test_positional_spectral_uplo_forms_match_keyword_contracts() -> None:
    """Positional UPLO forms should execute symmetric spectral contracts."""
    values = _capture_trace_array(np.array([2.0, 0.0, 3.0], dtype=np.float64))
    diagonal_left = cast(TraceADScalar, values[0])
    off_diagonal = cast(TraceADScalar, values[1])
    diagonal_right = cast(TraceADScalar, values[2])
    matrix = TraceADArray(
        (diagonal_left, off_diagonal, off_diagonal, diagonal_right),
        (2, 2),
        values.context,
    )

    eigenvalues, eigenvectors = cast(
        tuple[TraceADArray, TraceADArray],
        _dispatch(values, np.linalg.eigh, (matrix, "L")),
    )
    values_only = cast(
        TraceADArray,
        _dispatch(values, np.linalg.eigvalsh, (matrix, "U")),
    )

    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)
    assert _primals(values_only) == (2.0, 3.0)


def test_norm_scalar_axis_and_invalid_public_options() -> None:
    """Norm should support scalar axis reductions and reject unsupported options."""
    values = _capture_trace_array(np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float64))
    matrix = values.reshape((2, 2))

    scalar_norm = cast(
        TraceADScalar,
        _dispatch(values, np.linalg.norm, (values,), {"axis": 0}),
    )
    assert scalar_norm.primal == pytest.approx(np.sqrt(86.0))

    invalid_cases: tuple[tuple[tuple[object, ...], dict[str, object], str], ...] = (
        ((values,), {"ord": 1}, "supports only Euclidean"),
        ((values,), {"ord": "fro"}, "matrix norms require rank"),
        ((matrix,), {"axis": (0,)}, "axes must have length two"),
        ((matrix,), {"axis": (True, 1)}, "axes must be static integers"),
    )
    for args, kwargs, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _dispatch(values, np.linalg.norm, args, kwargs)

    empty = _capture_empty_trace_array()
    with pytest.raises(ValueError, match="requires at least one element"):
        _dispatch(empty, np.linalg.norm, (empty,))
