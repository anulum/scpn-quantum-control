# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD selection order statistics tests
# scpn-quantum-control -- Program AD selection order-statistic tests
"""Tests for Program AD strict-order selection and order-statistic semantics."""

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


def test_program_ad_sort_routes_strict_static_order_adjoint_semantics() -> None:
    """Program AD np.sort should route adjoints through strict sorted order."""

    weights_flat = np.array([1.5, -2.0, 0.25, 3.0], dtype=np.float64)
    weights_axis = np.array(
        [[0.5, -1.0, 2.0], [1.25, -0.75, 3.5]],
        dtype=np.float64,
    )

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:6], (2, 3))
        flat_sorted = np.sort(values[6:10], axis=None)
        axis_sorted = np.sort(matrix, axis=1)
        return np.sum(flat_sorted * weights_flat) + np.sum(axis_sorted * weights_axis)

    values = np.array(
        [3.0, 1.0, 2.0, -1.0, 4.0, 0.5, 0.25, -2.0, 1.5, -0.75],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    flat_indices = np.argsort(values[6:10])
    flat_expected = np.zeros(4, dtype=np.float64)
    flat_expected[flat_indices] = weights_flat
    expected[6:10] = flat_expected
    matrix = values[:6].reshape(2, 3)
    axis_indices = np.argsort(matrix, axis=1)
    matrix_expected = np.zeros_like(matrix)
    np.put_along_axis(matrix_expected, axis_indices, weights_axis, axis=1)
    expected[:6] = matrix_expected.reshape(-1)

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_sort_fails_closed_on_nondifferentiable_boundaries() -> None:
    """Program AD np.sort should reject ties, invalid axes, and integer policies."""

    sort = cast(Any, np.sort)
    with pytest.raises(ValueError, match="strictly ordered"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.sort(values)),
            np.array([1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axis must be a static integer or None"):
        whole_program_value_and_grad(
            lambda values: np.sum(sort(np.reshape(values, (2, 2)), axis=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axis out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.sort(np.reshape(values, (2, 2)), axis=2)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="argmax/argmin/argsort index selection semantics"):
        whole_program_value_and_grad(
            lambda values: np.argsort(values)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_order_statistic_reductions_match_selection_adjoint() -> None:
    """Program AD order-statistic reductions should route exact strict-order adjoints."""

    quantile_weights = np.array([1.2, -0.4], dtype=np.float64)
    percentile_weights = np.array([0.75, -1.1, 0.5], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        return (
            0.7 * np.median(values)
            + np.sum(np.quantile(matrix, 0.25, axis=1) * quantile_weights)
            + np.sum(np.percentile(matrix, 75.0, axis=0) * percentile_weights)
        )

    values = np.array([3.0, -2.0, 0.5, 1.0, -1.5, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    median_order = np.argsort(values)
    expected[median_order[2]] += 0.7 * 0.5
    expected[median_order[3]] += 0.7 * 0.5

    matrix = values.reshape(2, 3)
    matrix_expected = np.zeros_like(matrix)
    for row in range(matrix.shape[0]):
        order = np.argsort(matrix[row])
        matrix_expected[row, order[0]] += quantile_weights[row] * 0.5
        matrix_expected[row, order[1]] += quantile_weights[row] * 0.5
    for column in range(matrix.shape[1]):
        order = np.argsort(matrix[:, column])
        matrix_expected[order[0], column] += percentile_weights[column] * 0.25
        matrix_expected[order[1], column] += percentile_weights[column] * 0.75
    expected += matrix_expected.reshape(-1)

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_order_statistic_reductions_fail_closed_on_boundaries() -> None:
    """Program AD order-statistic reductions should reject unstable selection contracts."""

    quantile = cast(Any, np.quantile)
    percentile = cast(Any, np.percentile)
    with pytest.raises(ValueError, match="strictly ordered"):
        whole_program_value_and_grad(
            lambda values: np.median(values),
            np.array([1.0, 1.0, 2.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="q must be static"):
        whole_program_value_and_grad(
            lambda values: np.quantile(values, values[0]),
            np.array([0.25, 1.0, 2.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="scalar q"):
        whole_program_value_and_grad(
            lambda values: np.quantile(values, np.array([0.25, 0.75])),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="q must be in \\[0, 1\\]"):
        whole_program_value_and_grad(
            lambda values: np.quantile(values, 1.5),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="only supports method='linear'"):
        whole_program_value_and_grad(
            lambda values: quantile(values, 0.5, method="nearest"),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axis must be a static integer or None"):
        whole_program_value_and_grad(
            lambda values: percentile(np.reshape(values, (2, 2)), 50.0, axis=True),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
