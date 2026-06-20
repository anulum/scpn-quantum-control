# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD finite-difference stencil tests
"""Tests for Program AD finite-difference and static-gradient stencil semantics."""

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
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_finite_differences_match_linear_adjoint() -> None:
    """Program AD finite differences should preserve exact linear adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        first_order = np.diff(values)
        second_order_rows = np.diff(matrix, n=2, axis=1)
        return first_order[2] - 2.0 * first_order[4] + second_order_rows[0, 0]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(-7.0)
    _assert_allclose(
        result.gradient,
        [1.0, -2.0, 0.0, 1.0, 2.0, -2.0],
        atol=1.0e-12,
    )
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_finite_differences_reject_boundary_extensions() -> None:
    """Program AD finite differences should fail closed for boundary-extension modes."""

    with pytest.raises(ValueError, match="non-negative integer n"):
        whole_program_value_and_grad(
            lambda values: np.diff(values, n=-1)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="prepend/append"):
        whole_program_value_and_grad(
            lambda values: np.diff(values, prepend=0.0)[0],
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_gradient_matches_static_spacing_adjoint() -> None:
    """Program AD np.gradient should replay exact static finite-difference adjoints."""

    row_grid = np.array([0.0, 0.5, 1.5], dtype=np.float64)
    column_grid = np.array([-0.25, 0.0, 0.75, 1.25], dtype=np.float64)
    row_weights = np.linspace(-1.5, 2.0, 12, dtype=np.float64).reshape(3, 4)
    column_weights = np.linspace(0.5, -2.5, 12, dtype=np.float64).reshape(3, 4)
    flat_weights = np.array([0.25, -0.5, 1.0, -1.5], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:12], (3, 4))
        row_gradient, column_gradient = np.gradient(
            matrix,
            row_grid,
            column_grid,
            axis=(0, 1),
            edge_order=2,
        )
        flat_gradient = np.gradient(values[12:], 0.5, edge_order=1)
        return (
            np.sum(row_gradient * row_weights)
            + np.sum(column_gradient * column_weights)
            + np.sum(flat_gradient * flat_weights)
        )

    values = np.array(
        [
            1.0,
            -2.0,
            0.5,
            3.0,
            -1.5,
            2.0,
            4.0,
            -0.25,
            0.75,
            -3.0,
            1.5,
            2.5,
            0.5,
            -1.0,
            2.0,
            4.0,
        ],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    for source_index in range(12):
        basis = np.zeros((3, 4), dtype=np.float64)
        basis.reshape(-1)[source_index] = 1.0
        basis_row_gradient, basis_column_gradient = np.gradient(
            basis,
            row_grid,
            column_grid,
            axis=(0, 1),
            edge_order=2,
        )
        expected[source_index] = np.sum(basis_row_gradient * row_weights) + np.sum(
            basis_column_gradient * column_weights
        )
    for source_index in range(4):
        flat_basis = np.zeros(4, dtype=np.float64)
        flat_basis[source_index] = 1.0
        expected[12 + source_index] = np.sum(
            np.gradient(flat_basis, 0.5, edge_order=1) * flat_weights
        )

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, atol=1.0e-12)


def test_program_ad_gradient_fails_closed_invalid_static_contracts() -> None:
    """Program AD np.gradient should reject unsupported dynamic or singular grids."""

    with pytest.raises(ValueError, match="spacing must be static"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(values, values)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="edge_order must be 1 or 2"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.gradient)(values, edge_order=3)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axis must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(np.reshape(values, (2, 2)), axis=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(np.reshape(values, (2, 2)), axis=(0, 0))[0]),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="strictly monotonic"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(values, np.array([0.0, 1.0, 1.0]))),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="at least 3 samples"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.gradient(values, edge_order=2)),
            np.array([1.0, 2.0], dtype=np.float64),
        )
