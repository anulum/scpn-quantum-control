# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable sparse derivatives tests
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
    empirical_fisher_metric,
    sparse_empirical_fisher_metric,
    sparse_hessian,
    sparse_jacobian,
)
from scpn_quantum_control.differentiable_sparse_derivatives import (
    dense_to_sparse_matrix as direct_dense_to_sparse_matrix,
)
from scpn_quantum_control.differentiable_sparse_derivatives import (
    empirical_fisher_metric as direct_empirical_fisher_metric,
)
from scpn_quantum_control.differentiable_sparse_derivatives import (
    sparse_empirical_fisher_metric as direct_sparse_empirical_fisher_metric,
)
from scpn_quantum_control.differentiable_sparse_derivatives import (
    sparse_hessian as direct_sparse_hessian,
)
from scpn_quantum_control.differentiable_sparse_derivatives import (
    sparse_jacobian as direct_sparse_jacobian,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_sparse_derivative_helpers_preserve_facade_identity() -> None:
    """Extracted sparse helpers should keep facade and package-root imports stable."""

    import scpn_quantum_control as scpn
    from scpn_quantum_control import differentiable as differentiable_facade

    assert differentiable_facade.dense_to_sparse_matrix is direct_dense_to_sparse_matrix
    assert differentiable_facade.empirical_fisher_metric is direct_empirical_fisher_metric
    assert (
        differentiable_facade.sparse_empirical_fisher_metric
        is direct_sparse_empirical_fisher_metric
    )
    assert differentiable_facade.sparse_hessian is direct_sparse_hessian
    assert differentiable_facade.sparse_jacobian is direct_sparse_jacobian
    assert scpn.dense_to_sparse_matrix is direct_dense_to_sparse_matrix
    assert scpn.empirical_fisher_metric is direct_empirical_fisher_metric
    assert scpn.sparse_empirical_fisher_metric is direct_sparse_empirical_fisher_metric
    assert scpn.sparse_hessian is direct_sparse_hessian
    assert scpn.sparse_jacobian is direct_sparse_jacobian


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


def test_dense_to_sparse_matrix_defaults_metadata() -> None:
    """Dense sparse conversion should supply deterministic default metadata."""

    sparse = dense_to_sparse_matrix([[0.0, 2.0], [3.0, 0.0]])

    assert sparse.parameter_names == ("p0", "p1")
    assert sparse.trainable == (True, True)
    assert sparse.method == "dense_to_sparse"
    _assert_allclose(sparse.to_dense(), [[0.0, 2.0], [3.0, 0.0]])


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


def test_sparse_empirical_fisher_metric_accepts_dense_jacobian() -> None:
    """Sparse empirical Fisher conversion should work without result metadata."""

    sparse = sparse_empirical_fisher_metric(
        np.array([[1.0, 2.0], [0.5, 0.0]], dtype=np.float64),
        weights=np.array([2.0, 4.0], dtype=np.float64),
        damping=0.25,
        tolerance=1.0e-12,
    )

    assert sparse.parameter_names == ("p0", "p1")
    assert sparse.trainable == (True, True)
    _assert_allclose(sparse.to_dense(), [[3.25, 4.0], [4.0, 8.25]])


def test_empirical_fisher_metric_rejects_invalid_dense_inputs() -> None:
    """Dense Fisher metrics should fail closed on malformed arrays."""

    with pytest.raises(ValueError, match="two-dimensional"):
        empirical_fisher_metric(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite values"):
        empirical_fisher_metric(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="weights"):
        empirical_fisher_metric(np.eye(2), weights=np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        empirical_fisher_metric(np.eye(2), weights=np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="fisher damping"):
        empirical_fisher_metric(np.eye(2), damping=-1.0)


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
    with pytest.raises(ValueError, match="two-dimensional"):
        dense_to_sparse_matrix(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite values"):
        dense_to_sparse_matrix(np.array([[1.0, np.inf]]))
    with pytest.raises(ValueError, match="JacobianResult"):
        sparse_jacobian(cast(JacobianResult, np.eye(2)))
    with pytest.raises(ValueError, match="HessianResult"):
        sparse_hessian(cast(HessianResult, np.eye(2)))
