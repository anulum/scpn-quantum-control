# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD core linalg tests
"""Tests for Program AD determinant, inverse, and solve adjoints."""

from __future__ import annotations

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


def test_program_ad_linalg_det_matches_cofactor_adjoint() -> None:
    """Program AD determinant should expose exact cofactor derivatives."""

    def objective(values: Any) -> object:
        two_by_two = np.reshape(values[:4], (2, 2))
        three_by_three = np.reshape(values[4:], (3, 3))
        return np.linalg.det(two_by_two) + 0.5 * np.linalg.det(three_by_three)

    values = np.array(
        [
            1.5,
            -0.25,
            0.75,
            2.0,
            1.0,
            0.5,
            -0.25,
            0.0,
            1.25,
            0.75,
            0.5,
            -0.5,
            1.5,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    two_by_two = values[:4].reshape(2, 2)
    three_by_three = values[4:].reshape(3, 3)
    expected = np.zeros_like(values)
    expected[:4] = np.array(
        [two_by_two[1, 1], -two_by_two[1, 0], -two_by_two[0, 1], two_by_two[0, 0]],
        dtype=np.float64,
    )
    for row in range(3):
        for col in range(3):
            minor = np.delete(np.delete(three_by_three, row, axis=0), col, axis=1)
            expected[4 + row * 3 + col] = 0.5 * ((-1.0) ** (row + col)) * np.linalg.det(minor)

    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:det:")] == [
        "linalg:det:2x2",
        "linalg:det:3x3",
    ]
    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    assert result.adjoint_result.unsupported_ops == ()
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_det_fails_closed_invalid_matrix_contracts() -> None:
    """Program AD determinant should reject non-rank-2 and non-square inputs."""

    with pytest.raises(ValueError, match="shape rule requires a rank-2 matrix"):
        whole_program_value_and_grad(
            lambda values: np.linalg.det(values),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.linalg.det(np.reshape(values, (2, 3))),
            np.arange(1.0, 7.0, dtype=np.float64),
        )


def test_program_ad_linalg_inv_uses_primitive_adjoint_replay_ir() -> None:
    """Program AD inverse should emit compact primitive nodes for adjoint replay."""

    weights = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.sum(np.linalg.inv(matrix) * weights)

    values = np.array([2.0, -0.5, 0.25, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"a{index}") for index in range(values.size)),
    )

    inverse = np.linalg.inv(values.reshape(2, 2))
    expected = -(inverse.T @ weights @ inverse.T).reshape(-1)
    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:inv:")] == [
        "linalg:inv:2x2:0:0",
        "linalg:inv:2x2:0:1",
        "linalg:inv:2x2:1:0",
        "linalg:inv:2x2:1:1",
    ]
    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    assert result.adjoint_result.unsupported_ops == ()
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_linalg_inv_matches_inverse_differential() -> None:
    """Program AD inverse should match the exact matrix inverse differential."""

    weights_two = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    weights_three = np.array(
        [[0.25, -0.5, 1.5], [2.0, 0.75, -1.0], [-0.25, 1.25, 0.5]],
        dtype=np.float64,
    )

    def objective(values: Any) -> object:
        two_by_two = np.reshape(values[:4], (2, 2))
        three_by_three = np.reshape(values[4:], (3, 3))
        return np.sum(np.linalg.inv(two_by_two) * weights_two) + np.sum(
            np.linalg.inv(three_by_three) * weights_three
        )

    values = np.array(
        [
            2.0,
            -0.5,
            0.75,
            1.5,
            1.5,
            0.25,
            -0.5,
            0.0,
            1.25,
            0.5,
            -0.25,
            0.75,
            1.75,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    two_by_two = values[:4].reshape(2, 2)
    three_by_three = values[4:].reshape(3, 3)
    expected = np.zeros_like(values)
    inv_two = np.linalg.inv(two_by_two)
    inv_three = np.linalg.inv(three_by_three)
    expected[:4] = (-(inv_two.T @ weights_two @ inv_two.T)).reshape(-1)
    expected[4:] = (-(inv_three.T @ weights_three @ inv_three.T)).reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_inv_fails_closed_invalid_matrix_contracts() -> None:
    """Program AD inverse should reject non-rank-2, non-square, and singular inputs."""

    with pytest.raises(ValueError, match="shape rule requires a rank-2 matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.inv(values)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.inv(np.reshape(values, (2, 3)))),
            np.arange(1.0, 7.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.inv(np.reshape(values, (2, 2)))),
            np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64),
        )


def test_program_ad_linalg_solve_uses_primitive_adjoint_replay_ir() -> None:
    """Program AD solve should emit compact primitive nodes for adjoint replay."""

    vector_weights = np.array([0.4, -1.1], dtype=np.float64)
    matrix_weights = np.array([[0.25, -0.75], [1.5, 0.5]], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:4], (2, 2))
        vector_rhs = values[4:6]
        matrix_rhs = np.reshape(values[6:], (2, 2))
        vector_solution = np.linalg.solve(matrix, vector_rhs)
        matrix_solution = np.linalg.solve(matrix, matrix_rhs)
        return np.sum(vector_solution * vector_weights) + np.sum(matrix_solution * matrix_weights)

    values = np.array(
        [2.0, -0.5, 0.25, 1.5, 1.25, -0.75, 0.5, 1.0, -1.5, 0.25],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"s{index}") for index in range(values.size)),
    )

    matrix = values[:4].reshape(2, 2)
    vector_rhs = values[4:6]
    matrix_rhs = values[6:].reshape(2, 2)
    vector_solution = np.linalg.solve(matrix, vector_rhs)
    matrix_solution = np.linalg.solve(matrix, matrix_rhs)
    vector_adjoint = np.linalg.solve(matrix.T, vector_weights)
    matrix_adjoint = np.linalg.solve(matrix.T, matrix_weights)
    expected = np.zeros_like(values)
    expected[:4] = (
        -np.outer(vector_adjoint, vector_solution) - matrix_adjoint @ matrix_solution.T
    ).reshape(-1)
    expected[4:6] = vector_adjoint
    expected[6:] = matrix_adjoint.reshape(-1)

    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:solve:")] == [
        "linalg:solve:2x2:rhs:2:0",
        "linalg:solve:2x2:rhs:2:1",
        "linalg:solve:2x2:rhs:2x2:0:0",
        "linalg:solve:2x2:rhs:2x2:0:1",
        "linalg:solve:2x2:rhs:2x2:1:0",
        "linalg:solve:2x2:rhs:2x2:1:1",
    ]
    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    assert result.adjoint_result.unsupported_ops == ()
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_linalg_solve_matches_implicit_linear_system_differential() -> None:
    """Program AD solve should match exact linear-system differential semantics."""

    vector_weights = np.array([0.5, -1.25], dtype=np.float64)
    matrix_weights = np.array([[1.0, -0.5], [0.25, 1.5], [-1.25, 0.75]], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix_two = np.reshape(values[:4], (2, 2))
        rhs_two = values[4:6]
        matrix_three = np.reshape(values[6:15], (3, 3))
        rhs_three = np.reshape(values[15:], (3, 2))
        return np.sum(np.linalg.solve(matrix_two, rhs_two) * vector_weights) + np.sum(
            np.linalg.solve(matrix_three, rhs_three) * matrix_weights
        )

    values = np.array(
        [
            2.0,
            -0.5,
            0.75,
            1.5,
            0.25,
            -1.0,
            1.5,
            0.25,
            -0.5,
            0.0,
            1.25,
            0.5,
            -0.25,
            0.75,
            1.75,
            0.5,
            -1.0,
            1.25,
            0.75,
            -0.5,
            1.5,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    matrix_two = values[:4].reshape(2, 2)
    rhs_two = values[4:6]
    matrix_three = values[6:15].reshape(3, 3)
    rhs_three = values[15:].reshape(3, 2)
    solution_two = np.linalg.solve(matrix_two, rhs_two)
    solution_three = np.linalg.solve(matrix_three, rhs_three)
    expected = np.zeros_like(values)
    adjoint_rhs_two = np.linalg.solve(matrix_two.T, vector_weights)
    expected[:4] = (-np.outer(adjoint_rhs_two, solution_two)).reshape(-1)
    expected[4:6] = adjoint_rhs_two
    adjoint_rhs_three = np.linalg.solve(matrix_three.T, matrix_weights)
    expected[6:15] = (-(adjoint_rhs_three @ solution_three.T)).reshape(-1)
    expected[15:] = adjoint_rhs_three.reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_solve_fails_closed_invalid_contracts() -> None:
    """Program AD solve should reject invalid matrix and right-hand side contracts."""

    with pytest.raises(ValueError, match="shape rule requires a rank-2 matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(values[:2], values[2:4])),
            np.arange(1.0, 5.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="shape rule requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(np.reshape(values[:6], (2, 3)), values[6:8])),
            np.arange(1.0, 9.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="vector length must match matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:7])),
            np.arange(1.0, 8.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.solve(np.reshape(values[:4], (2, 2)), values[4:6])),
            np.array([1.0, 2.0, 2.0, 4.0, 1.0, 0.0], dtype=np.float64),
        )
