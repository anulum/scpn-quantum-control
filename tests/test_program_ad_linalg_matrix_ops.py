# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD matrix linalg tests
"""Tests for Program AD matrix-power and matrix-chain adjoints."""

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


def test_program_ad_linalg_matrix_power_matches_exact_differential() -> None:
    """Program AD matrix_power should compose exact matrix products and inverses."""

    weights_square = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    weights_inverse = np.array([[1.0, -0.5], [0.25, 1.5]], dtype=np.float64)
    weights_identity = np.array([[0.75, -0.25], [1.25, 0.5]], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return (
            np.sum(np.linalg.matrix_power(matrix, 2) * weights_square)
            + np.sum(np.linalg.matrix_power(matrix, -1) * weights_inverse)
            + np.sum(np.linalg.matrix_power(matrix, 0) * weights_identity)
        )

    values = np.array([2.0, -0.5, 0.75, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    matrix = values.reshape(2, 2)
    inv_matrix = np.linalg.inv(matrix)
    expected_matrix = (
        weights_square @ matrix.T
        + matrix.T @ weights_square
        - inv_matrix.T @ weights_inverse @ inv_matrix.T
    )
    _assert_allclose(result.gradient, expected_matrix.reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_matrix.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_linalg_matrix_power_uses_primitive_adjoint_replay_ir() -> None:
    """Program AD matrix_power should emit compact primitive nodes for adjoint replay."""

    weights_square = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    weights_inverse = np.array([[1.0, -0.5], [0.25, 1.5]], dtype=np.float64)
    weights_identity = np.array([[0.75, -0.25], [1.25, 0.5]], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return (
            np.sum(np.linalg.matrix_power(matrix, 2) * weights_square)
            + np.sum(np.linalg.matrix_power(matrix, -1) * weights_inverse)
            + np.sum(np.linalg.matrix_power(matrix, 0) * weights_identity)
        )

    values = np.array([2.0, -0.5, 0.75, 1.5], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"p{index}") for index in range(values.size)),
    )

    matrix = values.reshape(2, 2)
    inv_matrix = np.linalg.inv(matrix)
    expected = (
        weights_square @ matrix.T
        + matrix.T @ weights_square
        - inv_matrix.T @ weights_inverse @ inv_matrix.T
    ).reshape(-1)

    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:matrix_power:")] == [
        "linalg:matrix_power:2x2:power:2:0:0",
        "linalg:matrix_power:2x2:power:2:0:1",
        "linalg:matrix_power:2x2:power:2:1:0",
        "linalg:matrix_power:2x2:power:2:1:1",
        "linalg:matrix_power:2x2:power:-1:0:0",
        "linalg:matrix_power:2x2:power:-1:0:1",
        "linalg:matrix_power:2x2:power:-1:1:0",
        "linalg:matrix_power:2x2:power:-1:1:1",
        "linalg:matrix_power:2x2:power:0:0:0",
        "linalg:matrix_power:2x2:power:0:0:1",
        "linalg:matrix_power:2x2:power:0:1:0",
        "linalg:matrix_power:2x2:power:0:1:1",
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


def test_program_ad_linalg_matrix_power_fails_closed_invalid_contracts() -> None:
    """Program AD matrix_power should reject invalid matrices and powers."""

    with pytest.raises(ValueError, match="shape rule requires a rank-2 matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(values, 2)),
            np.arange(1.0, 5.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a square matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(np.reshape(values, (2, 3)), 2)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="static rule requires an integer power"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.matrix_power(np.reshape(values, (2, 2)), cast(Any, 1.5))
            ),
            np.arange(1.0, 5.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires a nonsingular matrix"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.matrix_power(np.reshape(values, (2, 2)), -1)),
            np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64),
        )


def test_program_ad_linalg_multi_dot_matches_exact_chain_differential() -> None:
    """Program AD multi_dot should compose exact static matrix-chain semantics."""

    matrix_weights = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    scalar_weight = 1.75

    def objective(values: Any) -> object:
        left = np.reshape(values[:4], (2, 2))
        middle = np.reshape(values[4:8], (2, 2))
        right = np.reshape(values[8:12], (2, 2))
        vector_left = values[12:14]
        vector_right = values[14:16]
        return np.sum(np.linalg.multi_dot((left, middle, right)) * matrix_weights) + (
            scalar_weight * np.linalg.multi_dot((vector_left, middle, vector_right))
        )

    values = np.array(
        [
            1.0,
            -0.5,
            0.75,
            1.5,
            0.25,
            -1.0,
            1.25,
            0.5,
            -0.75,
            2.0,
            0.5,
            -1.5,
            1.25,
            -0.25,
            0.75,
            -1.0,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    left = values[:4].reshape(2, 2)
    middle = values[4:8].reshape(2, 2)
    right = values[8:12].reshape(2, 2)
    vector_left = values[12:14]
    vector_right = values[14:16]
    expected = np.zeros_like(values)
    expected[:4] = (matrix_weights @ right.T @ middle.T).reshape(-1)
    expected[4:8] = (
        left.T @ matrix_weights @ right.T + scalar_weight * np.outer(vector_left, vector_right)
    ).reshape(-1)
    expected[8:12] = (middle.T @ left.T @ matrix_weights).reshape(-1)
    expected[12:14] = scalar_weight * (middle @ vector_right)
    expected[14:16] = scalar_weight * (vector_left @ middle)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_multi_dot_uses_primitive_adjoint_replay_ir() -> None:
    """Program AD multi_dot should emit compact primitive nodes for adjoint replay."""

    matrix_weights = np.array([[0.5, -1.25], [2.0, 0.75]], dtype=np.float64)
    scalar_weight = 1.75

    def objective(values: Any) -> object:
        left = np.reshape(values[:4], (2, 2))
        middle = np.reshape(values[4:8], (2, 2))
        right = np.reshape(values[8:12], (2, 2))
        vector_left = values[12:14]
        vector_right = values[14:16]
        return np.sum(np.linalg.multi_dot((left, middle, right)) * matrix_weights) + (
            scalar_weight * np.linalg.multi_dot((vector_left, middle, vector_right))
        )

    values = np.array(
        [
            1.0,
            -0.5,
            0.75,
            1.5,
            0.25,
            -1.0,
            1.25,
            0.5,
            -0.75,
            2.0,
            0.5,
            -1.5,
            1.25,
            -0.25,
            0.75,
            -1.0,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"m{index}") for index in range(values.size)),
    )

    left = values[:4].reshape(2, 2)
    middle = values[4:8].reshape(2, 2)
    right = values[8:12].reshape(2, 2)
    vector_left = values[12:14]
    vector_right = values[14:16]
    expected = np.zeros_like(values)
    expected[:4] = (matrix_weights @ right.T @ middle.T).reshape(-1)
    expected[4:8] = (
        left.T @ matrix_weights @ right.T + scalar_weight * np.outer(vector_left, vector_right)
    ).reshape(-1)
    expected[8:12] = (middle.T @ left.T @ matrix_weights).reshape(-1)
    expected[12:14] = scalar_weight * (middle @ vector_right)
    expected[14:16] = scalar_weight * (vector_left @ middle)

    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:multi_dot:")] == [
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:0",
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:1",
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:2",
        "linalg:multi_dot:2x2__2x2__2x2:out:2x2:3",
        "linalg:multi_dot:2__2x2__2:out:scalar",
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


def test_program_ad_linalg_multi_dot_fails_closed_invalid_contracts() -> None:
    """Program AD multi_dot should reject dynamic or invalid matrix-chain contracts."""

    with pytest.raises(ValueError, match="requires at least two operands"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.multi_dot((values,))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="supports rank-1 and rank-2 operands"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.multi_dot((np.reshape(values[:8], (2, 2, 2)), values[8:10]))
            ),
            np.arange(1.0, 11.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="middle operands must be rank-2"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.multi_dot((np.reshape(values[:4], (2, 2)), values[4:6], values[6:8]))
            ),
            np.arange(1.0, 9.0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dimensions must align"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.linalg.multi_dot(
                    (np.reshape(values[:4], (2, 2)), np.reshape(values[4:], (3, 2)))
                )
            ),
            np.arange(1.0, 11.0, dtype=np.float64),
        )
