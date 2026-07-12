# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable sparse derivatives module
# scpn-quantum-control -- sparse differentiable derivative helpers
"""Sparse derivative conversion helpers for native differentiable transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    HessianResult,
    JacobianResult,
    SparseMatrixResult,
)


def dense_to_sparse_matrix(
    matrix: ArrayLike,
    *,
    parameter_names: Sequence[str] | None = None,
    trainable: Sequence[bool] | None = None,
    method: str = "dense_to_sparse",
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Convert a dense derivative matrix to validated coordinate form.

    Parameters
    ----------
    matrix:
        Dense two-dimensional derivative matrix.
    parameter_names:
        Optional column names. Defaults to generated ``p{index}`` names.
    trainable:
        Optional trainable mask aligned to matrix columns. Defaults to all
        trainable.
    method:
        Provenance label stored in the sparse result.
    tolerance:
        Non-negative absolute-value threshold below which entries are dropped.

    Returns
    -------
    SparseMatrixResult
        Validated coordinate sparse derivative matrix with metadata preserved.
    """
    matrix_arr = _as_real_numeric_array("sparse source matrix", matrix)
    if matrix_arr.ndim != 2:
        raise ValueError("sparse source matrix must be two-dimensional")
    if not np.all(np.isfinite(matrix_arr)):
        raise ValueError("sparse source matrix must contain only finite values")
    tolerance_value = _as_real_scalar("sparse tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("sparse tolerance must be finite and non-negative")
    names = (
        tuple(f"p{index}" for index in range(matrix_arr.shape[1]))
        if parameter_names is None
        else tuple(parameter_names)
    )
    trainable_mask = (
        tuple(True for _ in range(matrix_arr.shape[1])) if trainable is None else tuple(trainable)
    )
    row_indices, column_indices = np.nonzero(np.abs(matrix_arr) > tolerance_value)
    values = matrix_arr[row_indices, column_indices]
    return SparseMatrixResult(
        row_indices=cast(NDArray[np.int64], row_indices.astype(np.int64)),
        column_indices=cast(NDArray[np.int64], column_indices.astype(np.int64)),
        values=cast(NDArray[np.float64], values.astype(np.float64)),
        shape=(int(matrix_arr.shape[0]), int(matrix_arr.shape[1])),
        method=method,
        parameter_names=names,
        trainable=trainable_mask,
    )


def sparse_jacobian(
    jacobian_result: JacobianResult,
    *,
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Return a coordinate sparse representation of a Jacobian result.

    Parameters
    ----------
    jacobian_result:
        Validated dense Jacobian result to convert.
    tolerance:
        Non-negative absolute-value threshold below which entries are dropped.

    Returns
    -------
    SparseMatrixResult
        Sparse Jacobian preserving parameter names, trainable mask, and method
        provenance.
    """
    if not isinstance(jacobian_result, JacobianResult):
        raise ValueError("sparse_jacobian requires a JacobianResult")
    return dense_to_sparse_matrix(
        jacobian_result.jacobian,
        parameter_names=jacobian_result.parameter_names,
        trainable=jacobian_result.trainable,
        method=f"sparse:{jacobian_result.method}",
        tolerance=tolerance,
    )


def sparse_hessian(
    hessian_result: HessianResult,
    *,
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Return a coordinate sparse representation of a Hessian result.

    Parameters
    ----------
    hessian_result:
        Validated dense Hessian result to convert.
    tolerance:
        Non-negative absolute-value threshold below which entries are dropped.

    Returns
    -------
    SparseMatrixResult
        Sparse Hessian preserving parameter names, trainable mask, and method
        provenance.
    """
    if not isinstance(hessian_result, HessianResult):
        raise ValueError("sparse_hessian requires a HessianResult")
    return dense_to_sparse_matrix(
        hessian_result.hessian,
        parameter_names=hessian_result.parameter_names,
        trainable=hessian_result.trainable,
        method=f"sparse:{hessian_result.method}",
        tolerance=tolerance,
    )


def empirical_fisher_metric(
    jacobian: JacobianResult | ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
) -> NDArray[np.float64]:
    """Return a dense empirical Fisher/Gauss-Newton metric.

    Parameters
    ----------
    jacobian:
        Dense residual Jacobian or a validated ``JacobianResult``.
    weights:
        Optional non-negative residual-row weights.
    damping:
        Optional non-negative diagonal damping.

    Returns
    -------
    numpy.ndarray
        Dense ``J.T @ W @ J + damping * I`` metric.
    """
    jacobian_arr = (
        jacobian.jacobian
        if isinstance(jacobian, JacobianResult)
        else _as_real_numeric_array("jacobian", jacobian)
    )
    if jacobian_arr.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    if not np.all(np.isfinite(jacobian_arr)):
        raise ValueError("jacobian must contain only finite values")
    if weights is None:
        weighted = jacobian_arr
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != jacobian_arr.shape[0]:
            raise ValueError("weights must be a one-dimensional array matching jacobian rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted = jacobian_arr * weight_arr[:, None]
    damping_value = _as_real_scalar("fisher damping", damping)
    if damping_value < 0.0:
        raise ValueError("fisher damping must be finite and non-negative")
    metric = jacobian_arr.T @ weighted
    if damping_value > 0.0:
        metric = metric + damping_value * np.eye(metric.shape[0], dtype=np.float64)
    typed_metric: NDArray[np.float64] = metric
    return typed_metric


def sparse_empirical_fisher_metric(
    jacobian: JacobianResult | ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    tolerance: float = 0.0,
) -> SparseMatrixResult:
    """Return a coordinate sparse empirical Fisher/Gauss-Newton metric.

    Parameters
    ----------
    jacobian:
        Dense residual Jacobian or a validated ``JacobianResult``.
    weights:
        Optional non-negative residual-row weights.
    damping:
        Optional non-negative diagonal damping.
    tolerance:
        Non-negative sparse conversion threshold.

    Returns
    -------
    SparseMatrixResult
        Sparse empirical Fisher metric with parameter metadata preserved when a
        ``JacobianResult`` is supplied.
    """
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    if isinstance(jacobian, JacobianResult):
        parameter_names = jacobian.parameter_names
        trainable = jacobian.trainable
    else:
        parameter_count = metric.shape[1]
        parameter_names = tuple(f"p{index}" for index in range(parameter_count))
        trainable = tuple(True for _ in range(parameter_count))
    return dense_to_sparse_matrix(
        metric,
        parameter_names=parameter_names,
        trainable=trainable,
        method="sparse:empirical_fisher",
        tolerance=tolerance,
    )


__all__ = [
    "dense_to_sparse_matrix",
    "empirical_fisher_metric",
    "sparse_empirical_fisher_metric",
    "sparse_hessian",
    "sparse_jacobian",
]
