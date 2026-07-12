# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable implicit sensitivity module
# scpn-quantum-control -- implicit sensitivity solvers
"""Implicit stationary and fixed-point sensitivity solvers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .differentiable_parameter_contracts import (
    Parameter,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    FixedPointSensitivityResult,
    ImplicitSensitivityResult,
)
from .differentiable_transform_helpers import _normalise_parameters


def implicit_stationary_sensitivity(
    hessian: ArrayLike,
    cross_derivative: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    hyperparameter_names: Sequence[str] | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> ImplicitSensitivityResult:
    """Return sensitivities for an implicit stationary optimum.

    Parameters
    ----------
    hessian
        Symmetric positive-definite Hessian ``H`` of the stationarity
        equations with respect to trainable parameters.
    cross_derivative
        Cross derivative matrix ``B`` with one row per parameter and one
        column per hyperparameter. One-dimensional inputs are treated as a
        single hyperparameter column.
    parameters
        Optional parameter metadata. Frozen parameters receive zero
        sensitivity and are excluded from the trainable linear solve.
    hyperparameter_names
        Optional names for the cross-derivative columns. Defaults to
        ``alpha0``, ``alpha1``, and so on.
    damping
        Non-negative diagonal damping applied to the active Hessian before the
        solve.
    rcond
        Positive reciprocal-condition threshold used to reject ill-conditioned
        trainable Hessian blocks.

    Returns
    -------
    ImplicitSensitivityResult
        Validated ``dx*/dalpha = -H^-1 B`` sensitivity metadata.
    """
    hessian_arr = _as_real_numeric_array("implicit hessian", hessian)
    cross = _as_real_numeric_array("implicit cross_derivative", cross_derivative)
    if hessian_arr.ndim != 2 or hessian_arr.shape[0] != hessian_arr.shape[1]:
        raise ValueError("implicit hessian must be a square matrix")
    if cross.ndim == 1:
        cross = cross.reshape((-1, 1))
    if cross.ndim != 2 or cross.shape[0] != hessian_arr.shape[0]:
        raise ValueError("implicit cross_derivative row count must match hessian dimension")
    if not np.all(np.isfinite(hessian_arr)) or not np.all(np.isfinite(cross)):
        raise ValueError("implicit operands must contain only finite values")
    if not np.allclose(hessian_arr, hessian_arr.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("implicit hessian must be symmetric")
    damping_value = _as_real_scalar("implicit damping", damping)
    if damping_value < 0.0:
        raise ValueError("implicit damping must be finite and non-negative")
    rcond_value = _as_real_scalar("implicit rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("implicit rcond must be finite and positive")
    parameter_values = np.zeros(hessian_arr.shape[0], dtype=np.float64)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    hyper_names = (
        tuple(f"alpha{index}" for index in range(cross.shape[1]))
        if hyperparameter_names is None
        else tuple(hyperparameter_names)
    )
    if len(hyper_names) != cross.shape[1]:
        raise ValueError("hyperparameter_names length must match cross_derivative columns")
    trainable = np.asarray([parameter.trainable for parameter in parameter_meta], dtype=bool)
    sensitivity = np.zeros_like(cross)
    condition_number = 1.0
    if np.any(trainable):
        active_hessian = hessian_arr[np.ix_(trainable, trainable)].copy()
        if damping_value > 0.0:
            active_hessian += damping_value * np.eye(active_hessian.shape[0], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(active_hessian)
        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))
        if min_eigenvalue <= 0.0:
            raise ValueError("implicit hessian must be positive definite on trainable parameters")
        condition_number = max_eigenvalue / min_eigenvalue
        if condition_number > 1.0 / rcond_value:
            raise ValueError("implicit hessian is ill-conditioned")
        sensitivity[trainable, :] = -np.linalg.solve(active_hessian, cross[trainable, :])
    return ImplicitSensitivityResult(
        sensitivity=sensitivity,
        hessian=hessian_arr,
        cross_derivative=cross,
        damping=damping_value,
        condition_number=condition_number,
        method="implicit_stationary_sensitivity",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        hyperparameter_names=hyper_names,
    )


def implicit_fixed_point_sensitivity(
    state_jacobian: ArrayLike,
    parameter_jacobian: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    hyperparameter_names: Sequence[str] | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> FixedPointSensitivityResult:
    """Return sensitivities for a fixed point ``x* = T(x*, alpha)``.

    Parameters
    ----------
    state_jacobian
        Square Jacobian ``dT/dx`` evaluated at the fixed point.
    parameter_jacobian
        Jacobian ``dT/dalpha`` with one row per state entry and one column per
        hyperparameter. One-dimensional inputs are treated as a single
        hyperparameter column.
    parameters
        Optional state-parameter metadata. Frozen entries receive zero
        sensitivity and are excluded from the active linear solve.
    hyperparameter_names
        Optional names for parameter-Jacobian columns. Defaults to ``alpha0``,
        ``alpha1``, and so on.
    damping
        Non-negative diagonal damping added to ``I - dT/dx``.
    rcond
        Positive reciprocal-condition threshold used to reject ill-conditioned
        active systems.

    Returns
    -------
    FixedPointSensitivityResult
        Validated ``(I - dT/dx)^-1 dT/dalpha`` sensitivity metadata.
    """
    state = _as_real_numeric_array("fixed-point state_jacobian", state_jacobian)
    parameter = _as_real_numeric_array("fixed-point parameter_jacobian", parameter_jacobian)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("fixed-point state_jacobian must be a square matrix")
    if parameter.ndim == 1:
        parameter = parameter.reshape((-1, 1))
    if parameter.ndim != 2 or parameter.shape[0] != state.shape[0]:
        raise ValueError("fixed-point parameter_jacobian row count must match state dimension")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(parameter)):
        raise ValueError("fixed-point operands must contain only finite values")
    damping_value = _as_real_scalar("fixed-point damping", damping)
    if damping_value < 0.0:
        raise ValueError("fixed-point damping must be finite and non-negative")
    rcond_value = _as_real_scalar("fixed-point rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("fixed-point rcond must be finite and positive")
    parameter_values = np.zeros(state.shape[0], dtype=np.float64)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    hyper_names = (
        tuple(f"alpha{index}" for index in range(parameter.shape[1]))
        if hyperparameter_names is None
        else tuple(hyperparameter_names)
    )
    if len(hyper_names) != parameter.shape[1]:
        raise ValueError(
            "hyperparameter_names length must match fixed-point parameter_jacobian columns"
        )
    system_matrix = np.eye(state.shape[0], dtype=np.float64) - state
    if damping_value > 0.0:
        system_matrix = system_matrix + damping_value * np.eye(state.shape[0], dtype=np.float64)
    trainable = np.asarray(
        [parameter_info.trainable for parameter_info in parameter_meta], dtype=bool
    )
    sensitivity = np.zeros_like(parameter)
    condition_number = 1.0
    if np.any(trainable):
        active_system = system_matrix[np.ix_(trainable, trainable)]
        condition_number = float(np.linalg.cond(active_system))
        if not np.isfinite(condition_number) or condition_number > 1.0 / rcond_value:
            raise ValueError("fixed-point system is ill-conditioned")
        sensitivity[trainable, :] = np.linalg.solve(active_system, parameter[trainable, :])
    return FixedPointSensitivityResult(
        sensitivity=sensitivity,
        state_jacobian=state,
        parameter_jacobian=parameter,
        system_matrix=system_matrix,
        damping=damping_value,
        condition_number=condition_number,
        method="implicit_fixed_point_sensitivity",
        parameter_names=tuple(parameter_info.name for parameter_info in parameter_meta),
        trainable=tuple(parameter_info.trainable for parameter_info in parameter_meta),
        hyperparameter_names=hyper_names,
    )
