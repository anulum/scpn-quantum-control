# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- sparse differentiable derivative tests
"""Tests for sparse derivative containers and sparse derivative converters."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    HessianResult,
    JacobianResult,
    SparseMatrixResult,
    dense_to_sparse_matrix,
    sparse_empirical_fisher_metric,
    sparse_hessian,
    sparse_jacobian,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_sparse_matrix_result_round_trips_dense_derivatives() -> None:
    """Sparse coordinate derivatives should preserve dense values and metadata."""

    dense = np.array([[1.0, 0.0, 2.0e-8], [0.0, 0.0, -3.0]])
    sparse = dense_to_sparse_matrix(
        dense,
        parameter_names=("a", "b", "c"),
        trainable=(True, False, True),
        tolerance=1.0e-7,
    )

    assert isinstance(sparse, SparseMatrixResult)
    assert sparse.nnz == 2
    assert sparse.shape == dense.shape
    assert sparse.parameter_names == ("a", "b", "c")
    assert sparse.trainable == (True, False, True)
    _assert_allclose(sparse.to_dense(), [[1.0, 0.0, 0.0], [0.0, 0.0, -3.0]])


def test_sparse_jacobian_hessian_and_fisher_preserve_provenance() -> None:
    """Sparse helpers should convert derivative result objects without metadata loss."""

    jacobian_result = JacobianResult(
        value=np.array([1.0, -2.0]),
        jacobian=np.array([[1.0, 0.0], [2.0, 0.0]]),
        method="analytic",
        step=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )
    hessian_result = HessianResult(
        value=1.0,
        hessian=np.array([[2.0, 0.0], [0.0, 0.0]]),
        method="analytic",
        step=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )

    sparse_j = sparse_jacobian(jacobian_result)
    sparse_h = sparse_hessian(hessian_result)
    sparse_fisher = sparse_empirical_fisher_metric(jacobian_result)

    assert sparse_j.method == "sparse:analytic"
    assert sparse_h.method == "sparse:analytic"
    assert sparse_fisher.method == "sparse:empirical_fisher"
    assert sparse_j.parameter_names == ("x", "y")
    assert sparse_h.trainable == (True, False)
    _assert_allclose(sparse_j.to_dense(), jacobian_result.jacobian)
    _assert_allclose(sparse_h.to_dense(), hessian_result.hessian)
    _assert_allclose(sparse_fisher.to_dense(), [[5.0, 0.0], [0.0, 0.0]])


def test_sparse_matrix_result_rejects_invalid_contracts() -> None:
    """Sparse derivative containers must fail closed on malformed coordinates."""

    with pytest.raises(ValueError, match="duplicate"):
        SparseMatrixResult(
            row_indices=np.array([0, 0]),
            column_indices=np.array([1, 1]),
            values=np.array([1.0, 2.0]),
            shape=(2, 2),
            method="bad",
            parameter_names=("x", "y"),
            trainable=(True, True),
        )
    with pytest.raises(ValueError, match="inside matrix shape"):
        SparseMatrixResult(
            row_indices=np.array([2]),
            column_indices=np.array([0]),
            values=np.array([1.0]),
            shape=(2, 2),
            method="bad",
            parameter_names=("x", "y"),
            trainable=(True, True),
        )
    with pytest.raises(ValueError, match="parameter_names"):
        dense_to_sparse_matrix(np.eye(2), parameter_names=("x",))
    with pytest.raises(ValueError, match="sparse tolerance"):
        dense_to_sparse_matrix(np.eye(2), tolerance=-1.0)
