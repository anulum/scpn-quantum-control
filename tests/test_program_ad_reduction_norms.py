# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD reduction and norm tests
"""Tests for Program AD product, statistical, norm, and cumulative reductions."""

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


def test_program_ad_product_reductions_match_product_rule_adjoint() -> None:
    """Program AD product reductions should preserve exact product-rule adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        row_products = np.prod(matrix, axis=1)
        return np.prod(values[:3]) + row_products[0] - 2.0 * row_products[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, -3.0, 4.0, 5.0, -2.0, 0.5], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-38.0)
    _assert_allclose(
        result.gradient,
        [-24.0, 16.0, -12.0, 2.0, -5.0, 20.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_product_reduction_methods_handle_zero_factor() -> None:
    """Trace-array prod methods should handle single-zero factors without finite differences."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return matrix.prod(axis=0)[0] + matrix.prod()

    result = whole_program_value_and_grad(
        objective,
        np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(0.0)
    _assert_allclose(result.gradient, [-21.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_program_ad_variance_and_std_reductions_match_analytic_gradients() -> None:
    """Program AD variance and standard deviation should use exact differentiable reductions."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.var(values) + matrix.var(axis=0)[1] + np.std(values[:2])

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
        ),
    )

    assert result.value == pytest.approx(10.0)
    _assert_allclose(
        result.gradient,
        [-2.0, -2.0, 0.5, 3.5],
        atol=1.0e-12,
    )


def test_program_ad_variance_and_std_reject_invalid_ddof() -> None:
    """Program AD variance/std should fail closed on unsupported or singular ddof."""

    with pytest.raises(ValueError, match="integer ddof"):
        whole_program_value_and_grad(
            lambda values: np.var(values, ddof=0.5),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="ddof must leave"):
        whole_program_value_and_grad(
            lambda values: np.std(values, ddof=2),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_axis_norms_match_euclidean_adjoint() -> None:
    """Program AD axis-aware Euclidean norms should replay exact vector adjoints."""

    row_weights = np.array([1.25, -0.5], dtype=np.float64)
    column_weights = np.array([0.75, -1.5, 0.25], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        row_norms = np.linalg.norm(matrix, 2, axis=1)
        column_norms = np.linalg.norm(matrix, None, 0)
        flat_norm = np.linalg.norm(values)
        return (
            np.sum(row_norms * row_weights)
            + np.sum(column_norms * column_weights)
            + 0.125 * flat_norm
        )

    values = np.array([1.0, 2.0, 2.0, 4.0, -1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    matrix = values.reshape(2, 3)
    expected = np.zeros_like(matrix)
    expected += row_weights[:, None] * matrix / np.linalg.norm(matrix, axis=1)[:, None]
    expected += column_weights[None, :] * matrix / np.linalg.norm(matrix, axis=0)[None, :]
    expected += 0.125 * matrix / np.linalg.norm(values)

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected.reshape(-1), atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected.reshape(-1), atol=1.0e-12)


def test_program_ad_axis_norms_fail_closed_on_unsupported_contracts() -> None:
    """Program AD axis norms should reject non-Euclidean, dynamic, and singular contracts."""

    with pytest.raises(ValueError, match="Euclidean norm"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.norm(np.reshape(values, (2, 2)), ord=1, axis=1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axis must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.norm(np.reshape(values, (2, 2)), axis=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires non-zero Euclidean norms"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.linalg.norm(np.reshape(values, (2, 2)), axis=1)),
            np.array([0.0, 0.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_frobenius_matrix_norms_match_exact_adjoint() -> None:
    """Program AD Frobenius matrix norms should replay exact static two-axis adjoints."""

    batch_weights = np.array([0.75, -1.25], dtype=np.float64)

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 2, 3))
        batched_norms = np.linalg.norm(tensor, "fro", axis=(1, 2))
        leading_matrix_norm = np.linalg.norm(tensor[0], None, axis=(-2, -1))
        return np.sum(batched_norms * batch_weights) + 0.5 * leading_matrix_norm

    values = np.array(
        [1.0, -2.0, 2.0, 0.5, -1.5, 2.5, 3.0, -1.0, 4.0, -2.0, 0.75, 1.25],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    tensor = values.reshape(2, 2, 3)
    norms = np.linalg.norm(tensor, ord="fro", axis=(1, 2))
    expected = batch_weights[:, None, None] * tensor / norms[:, None, None]
    expected[0] += 0.5 * tensor[0] / norms[0]

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected.reshape(-1), atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected.reshape(-1), atol=1.0e-12)


def test_program_ad_frobenius_matrix_norms_fail_closed_on_unsupported_contracts() -> None:
    """Program AD Frobenius norms should reject unsupported matrix-norm boundaries."""

    with pytest.raises(ValueError, match="matrix norms support only Frobenius"):
        whole_program_value_and_grad(
            lambda values: np.linalg.norm(np.reshape(values, (2, 2)), ord=1, axis=(0, 1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="axes must be distinct"):
        whole_program_value_and_grad(
            lambda values: np.linalg.norm(np.reshape(values, (2, 2)), ord="fro", axis=(1, 1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="requires non-zero Frobenius norms"):
        whole_program_value_and_grad(
            lambda values: np.linalg.norm(np.reshape(values, (2, 2)), ord="fro", axis=(0, 1)),
            np.zeros(4, dtype=np.float64),
        )


def test_program_ad_cumulative_sum_matches_prefix_adjoint() -> None:
    """Program AD cumulative sums should accumulate prefix adjoints exactly."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        flat_prefix = np.cumsum(values)
        row_prefix = matrix.cumsum(axis=1)
        return flat_prefix[3] + row_prefix[1, 2] - 2.0 * row_prefix[0, 1]

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

    assert result.value == pytest.approx(19.0)
    _assert_allclose(
        result.gradient,
        [-1.0, -1.0, 1.0, 2.0, 1.0, 1.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_cumulative_product_matches_prefix_product_adjoint() -> None:
    """Program AD cumulative products should preserve exact product-rule adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        flat_prefix = np.cumprod(values)
        row_prefix = matrix.cumprod(axis=1)
        return flat_prefix[2] + row_prefix[1, 2] - row_prefix[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([2.0, -3.0, 4.0, 5.0, -2.0, 0.5], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-23.0)
    _assert_allclose(
        result.gradient,
        [-9.0, 6.0, -6.0, -1.0, 2.5, -10.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_cumulative_product_method_handles_zero_factor() -> None:
    """Trace-array cumulative product methods should differentiate single-zero prefixes."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 2))
        return np.cumprod(values)[3] + matrix.cumprod(axis=1)[0, 1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float64),
        parameters=(
            Parameter("x00"),
            Parameter("x01"),
            Parameter("x10"),
            Parameter("x11"),
        ),
    )

    assert result.value == pytest.approx(0.0)
    _assert_allclose(result.gradient, [-22.0, 0.0, 0.0, 0.0], atol=1.0e-12)
