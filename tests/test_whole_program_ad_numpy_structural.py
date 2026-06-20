# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- whole-program AD NumPy structural tests
"""Tests for whole-program AD NumPy structural semantics."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_whole_program_ad_numpy_linear_algebra_fail_closed_paths() -> None:
    """Unsupported NumPy linear algebra modes should fail closed with explicit diagnostics."""

    with pytest.raises(ValueError, match="output labels must appear"):
        whole_program_value_and_grad(
            lambda values: np.einsum("i->j", values),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="explicit output"):
        whole_program_value_and_grad(
            lambda values: np.einsum("i,i", values, values),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ellipsis"):
        whole_program_value_and_grad(
            lambda values: np.einsum("...i->i", values.reshape((1, 2))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axis count exceeds operand rank"):
        whole_program_value_and_grad(
            lambda values: np.tensordot(values, values, axes=2),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="argmax/argmin/argsort index selection semantics"):
        whole_program_value_and_grad(
            lambda values: np.argsort(values)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_whole_program_ad_handles_numpy_broadcasting_semantics() -> None:
    """Program AD should follow NumPy broadcasting for ufuncs, predicates, and where."""

    def objective(values: Any) -> object:
        column = np.reshape(values[:2], (2, 1))
        row = values[2:5]
        broadcast_product = column * row
        shifted = broadcast_product + values[5]
        selected = np.where(shifted > 0.0, shifted, -shifted)
        return np.sum(selected) + np.sum(row / (column + 3.0))

    values = np.array([0.5, 1.25, 0.2, -0.4, 0.75, 0.3], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(
            Parameter("c0"),
            Parameter("c1"),
            Parameter("r0"),
            Parameter("r1"),
            Parameter("r2"),
            Parameter("bias"),
        ),
    )

    column = values[:2].reshape(2, 1)
    row = values[2:5]
    shifted = column * row + values[5]
    signs = np.where(shifted > 0.0, 1.0, -1.0)
    expected = np.zeros(6, dtype=np.float64)
    expected[0:2] = np.sum(signs * row, axis=1) - np.sum(row) / (column[:, 0] + 3.0) ** 2
    expected[2:5] = np.sum(signs * column, axis=0) + np.sum(1.0 / (column[:, 0] + 3.0))
    expected[5] = np.sum(signs)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_whole_program_ad_broadcasting_rejects_incompatible_shapes() -> None:
    """Program AD broadcasting should fail closed on incompatible NumPy shapes."""

    with pytest.raises(ValueError, match="broadcasting rules"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values[:4], (2, 2)) + values[4:7]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64),
        )


def test_whole_program_ad_handles_rank_n_reductions_and_transpose() -> None:
    """Program AD should support rank-N reductions and explicit transpose axes."""

    weights_axis0 = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    weights_axis1 = np.array([[2.0, -1.0], [0.25, 1.5]], dtype=np.float64)
    weights_transpose = np.array(
        [[[0.5, -0.25], [1.0, 2.0]], [[-1.5, 0.75], [0.0, 1.25]]],
        dtype=np.float64,
    )

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 2, 2))
        axis0 = np.sum(tensor, axis=0)
        axis1 = np.mean(tensor, axis=1)
        transposed = np.transpose(tensor, axes=(2, 0, 1))
        reversed_axes = tensor.T
        return (
            np.sum(axis0 * weights_axis0)
            + np.sum(axis1 * weights_axis1)
            + np.sum(transposed * weights_transpose)
            + np.sum(reversed_axes)
        )

    values = np.linspace(-0.4, 0.9, 8, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(8)),
    )
    expected_tensor = np.zeros((2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected_tensor[i, j, k] = (
                    weights_axis0[j, k]
                    + 0.5 * weights_axis1[i, k]
                    + weights_transpose[k, i, j]
                    + 1.0
                )

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    _assert_allclose(result.gradient, expected_tensor.reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result),
        expected_tensor.reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_whole_program_ad_rank_n_axis_validation_paths() -> None:
    """Rank-N program AD array operations should reject invalid axes explicitly."""

    with pytest.raises(ValueError, match="axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, 2)), axis=3),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axes must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.transpose(np.reshape(values, (2, 2, 1)), axes=(0, 1))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axes must be a permutation"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.transpose(np.reshape(values, (2, 2)), axes=(0, 0))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_whole_program_ad_handles_matrix_indexing_reductions_and_products() -> None:
    """Program AD should cover rank-2 array control, mutation, and products."""

    def objective(values: Any) -> object:
        matrix = values.reshape(2, 2).copy()
        matrix[0, 1] = matrix[0, 1] + matrix[1, 0]
        column_sum = np.sum(matrix, axis=0)
        row_sum = np.sum(matrix, axis=1)
        matrix_vector = matrix @ np.array([2.0, -1.0], dtype=np.float64)
        vector_matrix = np.array([1.5, -0.5], dtype=np.float64) @ matrix
        return (
            np.sum(column_sum)
            + np.sum(row_sum)
            + np.sum(matrix_vector)
            + np.sum(vector_matrix)
            + np.sum(matrix.T)
        )

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )

    assert any(node.op == "mutation:setitem" for node in result.ir_nodes)
    assert result.semantics_report is not None
    assert result.semantics_report.mutation_observed is True
    _assert_allclose(result.gradient, [6.5, 3.5, 8.0, 1.5], atol=1.0e-12)


def test_whole_program_ad_handles_numpy_composition_and_norms() -> None:
    """Program AD should cover common NumPy shape composition and norm workflows."""

    def objective(values: Any) -> object:
        left = values[:2]
        right = values[2:4]
        stacked = np.stack((left, right), axis=0)
        flat = np.concatenate((stacked[0], stacked[1]))
        reshaped = np.reshape(flat, (2, 2))
        transposed = np.transpose(reshaped)
        clipped = np.clip(transposed, -0.25, 1.5)
        return np.linalg.norm(clipped) + np.sum(np.ravel(transposed))

    values = np.array([0.5, -0.1, 2.0, -2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )

    norm = math.sqrt(0.5**2 + (-0.1) ** 2 + 1.5**2 + (-0.25) ** 2)
    expected = np.array([1.0 + 0.5 / norm, 1.0 - 0.1 / norm, 1.0, 1.0], dtype=np.float64)
    assert any(node.op == "clip" for node in result.ir_nodes)
    assert any(node.op == "sqrt" for node in result.ir_nodes)
    _assert_allclose(result.gradient, expected, atol=1.0e-12)


def test_whole_program_ad_handles_numpy_linear_algebra_primitives() -> None:
    """Program AD should cover bounded NumPy linear algebra forms exactly."""

    def objective(values: Any) -> object:
        left = values[:2]
        right = values[2:4]
        matrix = np.reshape(values, (2, 2))
        return (
            np.inner(left, right)
            + np.sum(np.outer(left, right))
            + np.trace(matrix)
            + np.sum(np.diag(matrix))
            + np.tensordot(left, right, axes=1)
            + np.sum(np.tensordot(left, right, axes=0))
            + np.einsum("i,i->", left, right)
            + np.sum(np.einsum("i,j->ij", left, right))
            + np.sum(np.einsum("ij,j->i", matrix, left))
            + np.einsum("ii->", matrix)
        )

    values = np.array([0.5, -0.25, 1.5, -2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")),
    )
    expected = np.array(
        [
            7.0 * values[2] + 3.0 * values[3] + 2.0 * values[0] + 3.0,
            3.0 * values[2] + 7.0 * values[3] + 2.0 * values[1],
            7.0 * values[0] + 3.0 * values[1],
            3.0 * values[0] + 7.0 * values[1] + 3.0,
        ],
        dtype=np.float64,
    )

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)
