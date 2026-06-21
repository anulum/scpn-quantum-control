# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- natural-gradient and line-search helpers
"""Natural-gradient preconditioning and scalar line-search helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    ParameterBounds,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    ArmijoLineSearchResult,
    GradientResult,
    NaturalGradientResult,
    WeightedGradientResult,
)
from .differentiable_transform_helpers import (
    _as_scalar,
    _normalise_bounds,
    _project_bounds,
)

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]


def armijo_backtracking_line_search(
    objective: ScalarObjective,
    values: ArrayLike,
    gradient_result: GradientResult,
    direction: ArrayLike,
    *,
    bounds: Sequence[ParameterBounds] | None = None,
    initial_step: float = 1.0,
    contraction: float = 0.5,
    sufficient_decrease: float = 1.0e-4,
    max_steps: int = 20,
) -> ArmijoLineSearchResult:
    """Return a bounded Armijo backtracking step for a scalar objective.

    Parameters
    ----------
    objective
        Scalar objective evaluated at candidate parameter vectors.
    values
        Current parameter vector.
    gradient_result
        Gradient metadata at ``values``.
    direction
        Candidate descent direction. Frozen parameter entries are masked out.
    bounds
        Optional closed-interval parameter bounds used to project candidates.
    initial_step
        Positive first trial step length.
    contraction
        Multiplicative step shrinkage in ``(0, 1)``.
    sufficient_decrease
        Armijo sufficient-decrease coefficient in ``(0, 1)``.
    max_steps
        Positive trial-step cap.

    Returns
    -------
    ArmijoLineSearchResult
        Accepted candidate or fail-closed rejection metadata.
    """

    if not isinstance(gradient_result, GradientResult):
        raise ValueError("line search requires a GradientResult")
    parameter_values = _as_parameter_array(values)
    if parameter_values.size != gradient_result.gradient.size:
        raise ValueError("line-search values length must match gradient length")
    direction_values = _as_parameter_array(direction)
    if direction_values.shape != parameter_values.shape:
        raise ValueError("line-search direction length must match values length")
    initial_step_value = _as_real_scalar("line-search initial_step", initial_step)
    contraction_value = _as_real_scalar("line-search contraction", contraction)
    sufficient_decrease_value = _as_real_scalar(
        "line-search sufficient_decrease",
        sufficient_decrease,
    )
    if initial_step_value <= 0.0:
        raise ValueError("line-search initial_step must be finite and positive")
    if not 0.0 < contraction_value < 1.0:
        raise ValueError("line-search contraction must be finite and between 0 and 1")
    if not 0.0 < sufficient_decrease_value < 1.0:
        raise ValueError("line-search sufficient_decrease must be finite and between 0 and 1")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("line-search max_steps must be a positive integer")
    trainable = np.asarray(gradient_result.trainable, dtype=bool)
    bounds_meta = _normalise_bounds(parameter_values, bounds)
    masked_direction = direction_values.copy()
    masked_direction[~trainable] = 0.0
    directional_derivative = float(gradient_result.gradient @ masked_direction)
    start_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1
    history: list[float] = [start_value]
    if directional_derivative >= 0.0 or not np.any(masked_direction[trainable]):
        return ArmijoLineSearchResult(
            values=parameter_values,
            value=start_value,
            step_size=0.0,
            direction=masked_direction,
            directional_derivative=directional_derivative,
            accepted=False,
            evaluations=evaluations,
            value_history=tuple(history),
            reason="non_descent_direction",
            parameter_names=gradient_result.parameter_names,
            trainable=gradient_result.trainable,
        )
    step_size = initial_step_value
    for _ in range(max_steps):
        candidate = _project_bounds(parameter_values + step_size * masked_direction, bounds_meta)
        actual_step = candidate - parameter_values
        actual_derivative = float(gradient_result.gradient @ actual_step)
        candidate_value = _as_scalar(objective(candidate.copy()))
        evaluations += 1
        history.append(candidate_value)
        if candidate_value <= start_value + sufficient_decrease_value * actual_derivative:
            return ArmijoLineSearchResult(
                values=candidate,
                value=candidate_value,
                step_size=step_size,
                direction=masked_direction,
                directional_derivative=directional_derivative,
                accepted=True,
                evaluations=evaluations,
                value_history=tuple(history),
                reason="accepted",
                parameter_names=gradient_result.parameter_names,
                trainable=gradient_result.trainable,
            )
        step_size *= contraction_value
    return ArmijoLineSearchResult(
        values=parameter_values,
        value=start_value,
        step_size=0.0,
        direction=masked_direction,
        directional_derivative=directional_derivative,
        accepted=False,
        evaluations=evaluations,
        value_history=tuple(history),
        reason="max_steps",
        parameter_names=gradient_result.parameter_names,
        trainable=gradient_result.trainable,
    )


def weighted_gradient_sum(
    components: Sequence[GradientResult],
    weights: ArrayLike,
    *,
    method: str = "weighted_sum",
) -> WeightedGradientResult:
    """Combine compatible scalar gradient results by an explicit weight vector.

    Parameters
    ----------
    components
        Non-empty gradient results with matching shape and parameter metadata.
    weights
        One finite scalar weight per component.
    method
        Provenance label stored on the result.

    Returns
    -------
    WeightedGradientResult
        Weighted scalar value, gradient, component provenance, and metadata.
    """

    component_tuple = tuple(components)
    if not component_tuple:
        raise ValueError("components must contain at least one GradientResult")
    if any(not isinstance(component, GradientResult) for component in component_tuple):
        raise ValueError("components must contain GradientResult instances")
    weight_arr = _as_real_numeric_array("weights", weights)
    if weight_arr.ndim != 1 or weight_arr.size != len(component_tuple):
        raise ValueError("weights length must match components length")
    if not np.all(np.isfinite(weight_arr)):
        raise ValueError("weights must contain only finite values")
    reference = component_tuple[0]
    for component in component_tuple[1:]:
        if component.gradient.shape != reference.gradient.shape:
            raise ValueError("all component gradients must have matching shapes")
        if component.parameter_names != reference.parameter_names:
            raise ValueError("all component parameter_names must match")
        if component.trainable != reference.trainable:
            raise ValueError("all component trainable masks must match")
    value = float(
        sum(
            float(weight) * component.value
            for weight, component in zip(weight_arr, component_tuple)
        )
    )
    gradient = np.zeros_like(reference.gradient)
    evaluations = 0
    for weight, component in zip(weight_arr, component_tuple):
        gradient += float(weight) * component.gradient
        evaluations += component.evaluations
    return WeightedGradientResult(
        value=value,
        gradient=gradient,
        components=component_tuple,
        weights=weight_arr,
        method=method,
        evaluations=evaluations,
        parameter_names=reference.parameter_names,
        trainable=reference.trainable,
    )


def natural_gradient(
    gradient_result: GradientResult,
    metric: ArrayLike,
    *,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Solve a trainable-subspace natural-gradient linear system.

    Parameters
    ----------
    gradient_result
        Scalar-objective gradient and parameter metadata.
    metric
        Symmetric positive-definite metric matrix over all parameters.
    damping
        Non-negative diagonal damping added on the trainable block.
    rcond
        Positive reciprocal-condition threshold.

    Returns
    -------
    NaturalGradientResult
        Preconditioned gradient with frozen parameter entries zeroed.
    """

    metric_arr = _as_real_numeric_array("natural-gradient metric", metric)
    if metric_arr.ndim != 2 or metric_arr.shape != (
        gradient_result.gradient.size,
        gradient_result.gradient.size,
    ):
        raise ValueError("natural-gradient metric must have shape (n_parameters, n_parameters)")
    if not np.all(np.isfinite(metric_arr)):
        raise ValueError("natural-gradient metric must contain only finite values")
    if not np.allclose(metric_arr, metric_arr.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("natural-gradient metric must be symmetric")
    damping_value = _as_real_scalar("natural-gradient damping", damping)
    if damping_value < 0.0:
        raise ValueError("natural-gradient damping must be finite and non-negative")
    rcond_value = _as_real_scalar("natural-gradient rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("natural-gradient rcond must be finite and positive")

    trainable = np.asarray(gradient_result.trainable, dtype=bool)
    result = np.zeros_like(gradient_result.gradient)
    if not np.any(trainable):
        return NaturalGradientResult(
            base_gradient=gradient_result,
            metric=metric_arr,
            natural_gradient=result,
            damping=damping_value,
            condition_number=1.0,
        )

    active_metric = metric_arr[np.ix_(trainable, trainable)].copy()
    if damping_value > 0.0:
        active_metric += damping_value * np.eye(active_metric.shape[0], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(active_metric)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    if min_eigenvalue <= 0.0:
        raise ValueError(
            "natural-gradient metric must be positive definite on trainable parameters"
        )
    condition_number = max_eigenvalue / min_eigenvalue
    if condition_number > 1.0 / rcond_value:
        raise ValueError("natural-gradient metric is ill-conditioned")
    result[trainable] = np.linalg.solve(active_metric, gradient_result.gradient[trainable])
    return NaturalGradientResult(
        base_gradient=gradient_result,
        metric=metric_arr,
        natural_gradient=result,
        damping=damping_value,
        condition_number=condition_number,
    )


__all__ = [
    "armijo_backtracking_line_search",
    "natural_gradient",
    "weighted_gradient_sum",
]
