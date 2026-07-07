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
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_finite_difference import value_and_finite_difference_grad
from .differentiable_parameter_contracts import (
    Parameter,
    ParameterBounds,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_parameter_shift import value_and_parameter_shift_grad
from .differentiable_result_contracts import (
    ArmijoLineSearchResult,
    GradientResult,
    NaturalGradientOptimizationResult,
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


@dataclass(frozen=True)
class NaturalGradientOptimizer:
    """Bounded natural-gradient optimizer for scalar objectives with explicit metrics.

    Parameters
    ----------
    learning_rate
        Non-negative scale applied to each natural-gradient step.
    damping
        Non-negative diagonal damping added to the trainable metric block.
    rcond
        Positive reciprocal-condition threshold for metric solves.
    max_step_norm
        Optional positive L2 cap for trainable natural-gradient steps.
    """

    learning_rate: float = 0.01
    damping: float = 0.0
    rcond: float = 1.0e-12
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        """Validate and canonicalize natural-gradient optimizer controls."""
        learning_rate = _as_real_scalar("natural-gradient learning_rate", self.learning_rate)
        damping = _as_real_scalar("natural-gradient damping", self.damping)
        rcond = _as_real_scalar("natural-gradient rcond", self.rcond)
        if learning_rate < 0.0:
            raise ValueError("natural-gradient learning_rate must be finite and non-negative")
        if damping < 0.0:
            raise ValueError("natural-gradient damping must be finite and non-negative")
        if rcond <= 0.0:
            raise ValueError("natural-gradient rcond must be finite and positive")
        max_step_norm = (
            None
            if self.max_step_norm is None
            else _as_real_scalar("natural-gradient max_step_norm", self.max_step_norm)
        )
        if max_step_norm is not None and max_step_norm <= 0.0:
            raise ValueError("natural-gradient max_step_norm must be finite and positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "rcond", rcond)
        object.__setattr__(self, "max_step_norm", max_step_norm)

    def minimize(
        self,
        objective: ScalarObjective,
        initial_values: ArrayLike,
        metric_fn: Callable[[GradientResult, NDArray[np.float64]], ArrayLike],
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        gradient_method: str = "parameter_shift",
        finite_difference_step: float = 1.0e-6,
        bounds: Sequence[ParameterBounds] | None = None,
        max_steps: int = 100,
        gradient_tolerance: float = 1.0e-8,
        step_tolerance: float = 1.0e-8,
        value_tolerance: float | None = None,
    ) -> NaturalGradientOptimizationResult:
        """Run a bounded natural-gradient descent loop with metric provenance.

        Parameters
        ----------
        objective
            Scalar objective evaluated at real parameter vectors.
        initial_values
            Initial real parameter vector before optional bound projection.
        metric_fn
            Callback returning the metric matrix for the current gradient and
            parameter vector.
        parameters
            Optional parameter metadata controlling names and trainable masks.
        rule
            Optional parameter-shift rule for the parameter-shift backend.
        gradient_method
            Either ``"parameter_shift"`` or ``"finite_difference"``.
        finite_difference_step
            Positive central-difference step for finite-difference gradients.
        bounds
            Optional per-parameter box or periodic bounds.
        max_steps
            Non-negative maximum number of descent steps.
        gradient_tolerance
            Non-negative trainable-gradient convergence tolerance.
        step_tolerance
            Non-negative trainable-step convergence tolerance.
        value_tolerance
            Optional non-negative objective-change convergence tolerance.

        Returns
        -------
        NaturalGradientOptimizationResult
            Final values, gradient and natural-gradient records, histories,
            convergence state, and best-observed iterate.
        """
        if gradient_method not in {"parameter_shift", "finite_difference"}:
            raise ValueError("gradient_method must be 'parameter_shift' or 'finite_difference'")
        finite_difference_step_value = _as_real_scalar(
            "finite_difference_step", finite_difference_step
        )
        if finite_difference_step_value <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        gradient_tolerance_value = _as_real_scalar("gradient_tolerance", gradient_tolerance)
        step_tolerance_value = _as_real_scalar("step_tolerance", step_tolerance)
        if gradient_tolerance_value < 0.0 or step_tolerance_value < 0.0:
            raise ValueError("natural-gradient tolerances must be finite and non-negative")
        value_tolerance_value = (
            None
            if value_tolerance is None
            else _as_real_scalar("value_tolerance", value_tolerance)
        )
        if value_tolerance_value is not None and value_tolerance_value < 0.0:
            raise ValueError("value_tolerance must be finite and non-negative")

        values = _as_parameter_array(initial_values).copy()
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        value_history: list[float] = []
        gradient_norm_history: list[float] = []
        step_norm_history: list[float] = []
        best_values = values.copy()
        best_value = float("inf")
        previous_value: float | None = None

        for step_index in range(max_steps + 1):
            gradient_result = self._gradient(
                objective,
                values,
                parameters=parameters,
                rule=rule,
                gradient_method=gradient_method,
                finite_difference_step=finite_difference_step_value,
            )
            metric = metric_fn(gradient_result, values.copy())
            natural_result = natural_gradient(
                gradient_result,
                metric,
                damping=self.damping,
                rcond=self.rcond,
            )
            trainable = np.asarray(gradient_result.trainable, dtype=bool)
            gradient_norm = float(np.linalg.norm(gradient_result.gradient[trainable], ord=2))
            value_history.append(gradient_result.value)
            gradient_norm_history.append(gradient_norm)
            if gradient_result.value < best_value:
                best_value = gradient_result.value
                best_values = values.copy()
            if gradient_norm <= gradient_tolerance_value:
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=True,
                    reason="gradient_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if (
                value_tolerance_value is not None
                and previous_value is not None
                and abs(previous_value - gradient_result.value) <= value_tolerance_value
            ):
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=True,
                    reason="value_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if step_index == max_steps:
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=False,
                    reason="max_steps",
                    best_values=best_values,
                    best_value=best_value,
                )
            step_vector = self._bounded_step(natural_result.natural_gradient, trainable)
            step_norm = float(np.linalg.norm(step_vector[trainable], ord=2))
            if step_norm <= step_tolerance_value:
                return NaturalGradientOptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    final_natural_gradient=natural_result,
                    value_history=tuple(value_history),
                    gradient_norm_history=tuple(gradient_norm_history),
                    natural_step_norm_history=tuple(step_norm_history),
                    steps=step_index,
                    converged=True,
                    reason="step_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            step_norm_history.append(step_norm)
            previous_value = gradient_result.value
            values = _project_bounds(values - step_vector, bounds_meta)

        raise RuntimeError("unreachable natural-gradient optimizer state")  # pragma: no cover

    def _bounded_step(
        self,
        natural_gradient_value: NDArray[np.float64],
        trainable: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return one trainable-mask-aware bounded natural-gradient step."""
        step_vector = self.learning_rate * natural_gradient_value.copy()
        if self.max_step_norm is not None and np.any(trainable):
            norm = float(np.linalg.norm(step_vector[trainable], ord=2))
            if norm > self.max_step_norm:
                step_vector[trainable] *= self.max_step_norm / norm
        step_vector[~trainable] = 0.0
        typed_step: NDArray[np.float64] = step_vector
        return typed_step

    @staticmethod
    def _gradient(
        objective: ScalarObjective,
        values: NDArray[np.float64],
        *,
        parameters: Sequence[Parameter] | None,
        rule: ParameterShiftRule | None,
        gradient_method: str,
        finite_difference_step: float,
    ) -> GradientResult:
        """Return a gradient record from the requested scalar-gradient backend."""
        if gradient_method == "finite_difference":
            return value_and_finite_difference_grad(
                objective,
                values,
                parameters=parameters,
                step=finite_difference_step,
            )
        return value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )


__all__ = [
    "NaturalGradientOptimizer",
    "armijo_backtracking_line_search",
    "natural_gradient",
    "weighted_gradient_sum",
]
