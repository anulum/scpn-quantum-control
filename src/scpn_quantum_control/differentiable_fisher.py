# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable fisher module
# scpn-quantum-control -- empirical-Fisher linear algebra helpers
"""Empirical-Fisher matrix-free solves for residual-map derivatives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    FisherConjugateGradientResult,
    FisherVectorProductResult,
    JacobianResult,
    LeastSquaresCovarianceResult,
)
from .differentiable_sparse_derivatives import empirical_fisher_metric


def empirical_fisher_vector_product(
    jacobian: JacobianResult,
    tangent: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
) -> FisherVectorProductResult:
    """Return a weighted empirical-Fisher product without materialising a solve.

    Parameters
    ----------
    jacobian
        Finite-difference or custom Jacobian result for a residual map.
    tangent
        Candidate parameter-space vector to multiply by the Fisher metric.
    weights
        Optional non-negative residual weights with one entry per residual row.
    damping
        Non-negative diagonal damping applied only on trainable parameters.

    Returns
    -------
    FisherVectorProductResult
        Matrix-free product result with frozen parameter entries zeroed.
    """
    if not isinstance(jacobian, JacobianResult):
        raise ValueError("empirical_fisher_vector_product requires a JacobianResult")
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape[0] != jacobian.jacobian.shape[1]:
        raise ValueError("Fisher-vector tangent length must match Jacobian parameter dimension")
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    damping_value = _as_real_scalar("Fisher-vector damping", damping)
    if damping_value < 0.0:
        raise ValueError("Fisher-vector damping must be finite and non-negative")
    projection = jacobian.jacobian @ masked_tangent
    if weights is None:
        weighted_projection = projection
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != projection.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_projection = projection * weight_arr
    product = jacobian.jacobian.T @ weighted_projection
    if damping_value > 0.0:
        product[trainable] += damping_value * masked_tangent[trainable]
    product[~trainable] = 0.0
    return FisherVectorProductResult(
        value=jacobian.value,
        tangent=masked_tangent,
        product=product,
        residual_projection=projection,
        damping=damping_value,
        method=f"fisher_vector_product:{jacobian.method}",
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def empirical_fisher_conjugate_gradient(
    jacobian: JacobianResult,
    rhs: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-8,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
) -> FisherConjugateGradientResult:
    """Solve an empirical-Fisher system with matrix-free conjugate gradients.

    Parameters
    ----------
    jacobian
        Residual-map Jacobian whose trainable columns define the solve space.
    rhs
        Right-hand side vector with one entry per parameter.
    weights
        Optional non-negative residual weights.
    damping
        Non-negative diagonal damping for the empirical-Fisher operator.
    tolerance
        Non-negative residual-norm convergence tolerance.
    max_iterations
        Optional positive iteration cap. Defaults to ten passes over trainable
        parameters.

    Returns
    -------
    FisherConjugateGradientResult
        Solution and residual history for the trainable-subspace solve.
    """
    if not isinstance(jacobian, JacobianResult):
        raise ValueError("empirical_fisher_conjugate_gradient requires a JacobianResult")
    rhs_values = _as_parameter_array(rhs)
    if rhs_values.shape[0] != jacobian.jacobian.shape[1]:
        raise ValueError("Fisher-CG rhs length must match Jacobian parameter dimension")
    damping_value = _as_real_scalar("Fisher-CG damping", damping)
    if damping_value < 0.0:
        raise ValueError("Fisher-CG damping must be finite and non-negative")
    tolerance_value = _as_real_scalar("Fisher-CG tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("Fisher-CG tolerance must be finite and non-negative")
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    if max_iterations is None:
        max_iter = max(1, int(np.count_nonzero(trainable)) * 10)
    else:
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("Fisher-CG max_iterations must be a positive integer")
        max_iter = max_iterations
    solution = np.zeros_like(rhs_values)
    masked_rhs = rhs_values.copy()
    masked_rhs[~trainable] = 0.0
    residual = masked_rhs.copy()
    residual_norm = float(np.linalg.norm(residual[trainable], ord=2))
    residual_history: list[float] = [residual_norm]
    if residual_norm <= tolerance_value or not np.any(trainable):
        return FisherConjugateGradientResult(
            solution=solution,
            residual_norm_history=tuple(residual_history),
            iterations=0,
            converged=True,
            tolerance=tolerance_value,
            damping=damping_value,
            parameter_names=jacobian.parameter_names,
            trainable=jacobian.trainable,
        )

    direction = residual.copy()
    residual_sq = float(residual[trainable] @ residual[trainable])
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        product_result = empirical_fisher_vector_product(
            jacobian,
            direction,
            weights=weights,
            damping=damping_value,
        )
        product = product_result.product
        denom = float(direction[trainable] @ product[trainable])
        if denom <= 0.0 or not np.isfinite(denom):
            raise ValueError(
                "Fisher-CG operator must be positive definite on trainable parameters"
            )
        alpha = residual_sq / denom
        solution[trainable] += alpha * direction[trainable]
        residual[trainable] -= alpha * product[trainable]
        new_residual_sq = float(residual[trainable] @ residual[trainable])
        residual_norm = float(np.sqrt(max(new_residual_sq, 0.0)))
        residual_history.append(residual_norm)
        iterations = iteration
        if residual_norm <= tolerance_value:
            converged = True
            break
        beta = new_residual_sq / residual_sq
        direction[trainable] = residual[trainable] + beta * direction[trainable]
        direction[~trainable] = 0.0
        residual_sq = new_residual_sq
    solution[~trainable] = 0.0
    return FisherConjugateGradientResult(
        solution=solution,
        residual_norm_history=tuple(residual_history),
        iterations=iterations,
        converged=converged,
        tolerance=tolerance_value,
        damping=damping_value,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


def least_squares_covariance(
    jacobian: JacobianResult,
    *,
    weights: ArrayLike | None = None,
    residual_variance: float | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> LeastSquaresCovarianceResult:
    """Estimate residual-map parameter covariance from the empirical Fisher.

    Parameters
    ----------
    jacobian
        Residual-map Jacobian result whose value contains residuals.
    weights
        Optional non-negative residual weights.
    residual_variance
        Optional externally estimated residual variance. When omitted, the
        weighted residual norm is divided by residual degrees of freedom.
    damping
        Non-negative diagonal damping passed to the empirical-Fisher metric.
    rcond
        Positive reciprocal-condition threshold in ``(0, 1)``.

    Returns
    -------
    LeastSquaresCovarianceResult
        Full parameter covariance matrix, standard errors, and conditioning
        metadata with frozen parameter rows and columns zeroed.
    """
    if not isinstance(jacobian, JacobianResult):
        raise ValueError("least_squares_covariance requires a JacobianResult")
    residual = jacobian.value
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    active_count = int(np.count_nonzero(trainable))
    if active_count == 0:
        raise ValueError("least_squares_covariance requires at least one trainable parameter")
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    active_metric = metric[np.ix_(trainable, trainable)]
    eigenvalues = np.linalg.eigvalsh(active_metric)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    rcond_value = _as_real_scalar("least-squares rcond", rcond)
    if not 0.0 < rcond_value < 1.0:
        raise ValueError("rcond must be finite and between 0 and 1")
    if min_eigenvalue <= 0.0:
        raise ValueError("least-squares Fisher metric must be positive definite")
    condition_number = max_eigenvalue / min_eigenvalue
    if condition_number > 1.0 / rcond_value:
        raise ValueError("least-squares Fisher metric is ill-conditioned")
    degrees_of_freedom = max(1, residual.size - active_count)
    if residual_variance is None:
        if weights is None:
            weighted_residual = residual
        else:
            weight_arr = _as_real_numeric_array("weights", weights)
            weighted_residual = residual * weight_arr
        variance = float(residual @ weighted_residual) / degrees_of_freedom
    else:
        variance = _as_real_scalar("least-squares residual_variance", residual_variance)
        if variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")
    active_covariance = np.linalg.inv(active_metric) * variance
    covariance: NDArray[np.float64] = np.zeros_like(metric)
    covariance[np.ix_(trainable, trainable)] = active_covariance
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return LeastSquaresCovarianceResult(
        covariance=covariance,
        standard_errors=standard_errors,
        residual_variance=variance,
        degrees_of_freedom=degrees_of_freedom,
        condition_number=condition_number,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )


__all__ = [
    "empirical_fisher_conjugate_gradient",
    "empirical_fisher_vector_product",
    "least_squares_covariance",
]
