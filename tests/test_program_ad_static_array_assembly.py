# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD static array assembly tests
"""Tests for Program AD static gather, scatter, and array assembly semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(actual: object, expected: object, *, atol: float = 0.0) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, atol=atol)


def test_program_ad_static_take_accumulates_gather_adjoint() -> None:
    """Static NumPy take gathers should preserve exact adjoint accumulation."""

    def objective(values: Any) -> object:
        vector_gather = np.take(values, [2, 0, 2])
        matrix = np.reshape(values, (2, 3))
        column_gather = matrix.take([1, 0], axis=1)
        return np.sum(vector_gather) + column_gather[0, 0] - 2.0 * column_gather[1, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(1.0)
    _assert_allclose(
        result.gradient,
        [1.0, 1.0, 2.0, -2.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_take_modes_accumulate_gather_adjoint() -> None:
    """Static NumPy take wrap and clip modes should preserve exact scatter adjoints."""

    wrap_weights = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    clip_weights = np.array([-0.25, 1.5, 0.75], dtype=np.float64)

    def objective(values: Any) -> object:
        wrapped = np.take(values, [-1, 6, 0], mode="wrap")
        clipped = np.take(values, [-3, 2, 20], mode="clip")
        return np.sum(wrapped * wrap_weights) + np.sum(clipped * clip_weights)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(12.75)
    _assert_allclose(result.gradient, [0.75, 0.0, 1.5, 0.0, 0.0, 1.25])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_advanced_indexing_accumulates_gather_adjoint() -> None:
    """Static integer and boolean advanced indexes should preserve exact scatter adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        integer_gather = matrix[[1, 0, 1], [2, 0, 2]]
        boolean_rows = matrix[np.array([True, False])]
        repeated_columns = boolean_rows[:, np.array([2, 0, 2])]
        return np.sum(integer_gather) + 2.0 * np.sum(repeated_columns)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(27.0)
    _assert_allclose(
        result.gradient,
        [3.0, 0.0, 4.0, 0.0, 0.0, 2.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_take_along_axis_accumulates_gather_adjoint() -> None:
    """Static take_along_axis gathers should preserve exact repeated-index adjoints."""

    indices = np.array([[2, 0, 2], [1, 1, 0]], dtype=np.int64)
    weights = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.25]], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        gathered = np.take_along_axis(matrix, indices, axis=1)
        return np.sum(gathered * weights)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(12.25)
    _assert_allclose(
        result.gradient,
        [-0.5, 0.0, 3.0, -1.25, 1.75, 0.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_delete_preserves_gather_adjoint() -> None:
    """Static NumPy delete should preserve exact gather/scatter adjoints."""

    axis_weights = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    flat_weights = np.array([0.25, -0.75, 1.25, -1.5], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        axis_deleted = np.delete(matrix, [1], axis=1)
        flat_deleted = np.delete(values, [1, 4])
        return np.sum(axis_deleted * axis_weights) + np.sum(flat_deleted * flat_weights)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=tuple(Parameter(f"x{index}") for index in range(6)),
    )

    assert result.value == pytest.approx(9.0)
    _assert_allclose(result.gradient, [1.25, 0.0, -2.75, 1.75, 0.0, 1.5])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_constant_pad_preserves_scatter_adjoint() -> None:
    """Static constant padding should preserve exact source scatter adjoints."""

    matrix_weights = np.array(
        [
            [0.5, -1.0, 2.0, -0.5, 1.0],
            [1.25, -0.75, 1.5, -2.0, 0.25],
            [-1.5, 2.5, 0.75, 3.0, -0.25],
        ],
        dtype=np.float64,
    )
    flat_weights = np.array([0.1, -0.5, 2.25, -1.25, 0.75, 1.0, -2.0], dtype=np.float64)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 2))
        matrix_padded = np.pad(
            matrix,
            ((1, 0), (2, 1)),
            mode="constant",
            constant_values=-2.0,
        )
        flat_padded = np.pad(
            trace_values,
            (1, 2),
            mode="constant",
            constant_values=(0.5, -1.0),
        )
        return np.sum(matrix_padded * matrix_weights) + np.sum(flat_padded * flat_weights)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected_value = float(
        np.sum(
            np.pad(values.reshape(2, 2), ((1, 0), (2, 1)), constant_values=-2.0) * matrix_weights
        )
        + np.sum(np.pad(values, (1, 2), constant_values=(0.5, -1.0)) * flat_weights)
    )
    assert result.value == pytest.approx(expected_value)
    _assert_allclose(result.gradient, [1.0, 0.25, -0.5, 3.75])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_constant_insert_preserves_scatter_adjoint() -> None:
    """Static constant insertions should preserve exact source scatter adjoints."""

    axis_weights = np.array([[1.0, -10.0, 2.0], [3.0, 20.0, 4.0]], dtype=np.float64)
    flat_weights = np.array([0.25, -0.5, 1.25, 2.0, -0.75, 3.0], dtype=np.float64)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 2))
        axis_inserted = np.insert(matrix, 1, np.array([-2.0, 3.0]), axis=1)
        flat_inserted = np.insert(trace_values, [1, 3], np.array([0.5, -1.0]))
        return np.sum(axis_inserted * axis_weights) + np.sum(flat_inserted * flat_weights)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected_value = float(
        np.sum(np.insert(values.reshape(2, 2), 1, np.array([-2.0, 3.0]), axis=1) * axis_weights)
        + np.sum(np.insert(values, [1, 3], np.array([0.5, -1.0])) * flat_weights)
    )
    assert result.value == pytest.approx(expected_value)
    _assert_allclose(result.gradient, [1.25, 3.25, 5.0, 7.0])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_append_preserves_traceable_assembly_adjoint() -> None:
    """NumPy append should preserve exact flat and axis-aware assembly adjoints."""

    values = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    axis_weights = np.array([[1.5, -0.25], [2.0, -1.0]], dtype=np.float64)
    flat_weights = np.array([0.75, -1.5, 2.5, -0.5, 4.0, -2.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 2))
        axis_appended = np.append(matrix[:, :1], np.array([[7.0], [-3.0]]), axis=1)
        flat_appended = np.append(trace_values[:2], trace_values[2:])
        constant_appended = np.append(trace_values[:2], np.array([5.0, -4.0]))
        return (
            np.sum(axis_appended * axis_weights)
            + np.sum(flat_appended * flat_weights[:4])
            + np.sum(constant_appended * flat_weights[2:])
        )

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected_value = float(
        np.sum(
            np.append(values.reshape(2, 2)[:, :1], np.array([[7.0], [-3.0]]), axis=1)
            * axis_weights
        )
        + np.sum(np.append(values[:2], values[2:]) * flat_weights[:4])
        + np.sum(np.append(values[:2], np.array([5.0, -4.0])) * flat_weights[2:])
    )
    assert result.value == pytest.approx(expected_value)
    _assert_allclose(result.gradient, [4.75, -2.0, 4.5, -0.5])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_static_take_rejects_dynamic_indices_and_modes() -> None:
    """Program AD take should fail closed outside static integer gather semantics."""

    with pytest.raises(ValueError, match="static integer indices"):
        whole_program_value_and_grad(
            lambda values: np.take(values, values[0]),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="mode"):
        whole_program_value_and_grad(
            lambda values: cast(Any, np.take)(values, [0], mode="not_a_mode")[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_static_delete_rejects_dynamic_indices() -> None:
    """Program AD delete should fail closed on derivative-carrying deletion indices."""

    with pytest.raises(ValueError, match="static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.delete(values, values[0])),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_static_constant_pad_rejects_dynamic_parameters() -> None:
    """Program AD pad should fail closed on derivative-carrying pad parameters."""

    with pytest.raises(ValueError, match="static non-negative integer pad widths"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.pad(values, (values[0], 1))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="static finite real constant_values"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.pad(values, (1, 0), constant_values=values[0])),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="constant mode"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.pad(values, (1, 0), mode="edge")),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_static_constant_insert_rejects_dynamic_parameters() -> None:
    """Program AD insert should fail closed on derivative-carrying insert parameters."""

    with pytest.raises(ValueError, match="static integer insertion"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.insert(values, values[0], 1.0)),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="static finite real insert values"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.insert(values, 1, values[0])),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_append_rejects_dynamic_or_incompatible_axes() -> None:
    """Program AD append should fail closed outside static axis-compatible assembly."""

    with pytest.raises(ValueError, match="static integer axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.append(values[:1], values[1:], axis=values[0])),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="equal operand ranks"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.append(np.reshape(values, (2, 2)), values[:2], axis=0)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
