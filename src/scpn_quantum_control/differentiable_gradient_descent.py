# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable gradient descent module
# scpn-quantum-control -- native gradient-descent optimizer
"""Native gradient-descent optimizer for differentiable scalar objectives."""

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
    _as_real_scalar,
)
from .differentiable_parameter_shift import value_and_parameter_shift_grad
from .differentiable_result_contracts import GradientResult, OptimizationResult
from .differentiable_transform_helpers import (
    _clip_gradient,
    _normalise_bounds,
    _project_bounds,
    _validate_max_gradient_norm,
)

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]


@dataclass(frozen=True)
class DifferentiableOptimizer:
    """Native gradient-descent optimizer for differentiable parameters.

    Parameters
    ----------
    learning_rate:
        Non-negative step size applied to trainable gradient components before
        optional bound projection.
    """

    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        """Validate and canonicalize the optimizer step size."""
        learning_rate = _as_real_scalar("learning_rate", self.learning_rate)
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be finite and non-negative")
        object.__setattr__(self, "learning_rate", learning_rate)

    def step(
        self,
        values: ArrayLike,
        gradient_result: GradientResult,
        *,
        bounds: Sequence[ParameterBounds] | None = None,
        max_gradient_norm: float | None = None,
    ) -> NDArray[np.float64]:
        """Return one projected gradient-descent update.

        Parameters
        ----------
        values:
            Current real parameter vector.
        gradient_result:
            Gradient and trainable-mask metadata for the current point.
        bounds:
            Optional per-parameter box or periodic bounds.
        max_gradient_norm:
            Optional L2 clipping threshold applied to trainable components.

        Returns
        -------
        numpy.ndarray
            Updated real parameter vector after trainable-mask filtering and
            optional bound projection.
        """
        parameter_values = _as_parameter_array(values)
        bounds_meta = _normalise_bounds(parameter_values, bounds)
        if parameter_values.size != gradient_result.gradient.size:
            raise ValueError("values length must match gradient length")
        trainable = np.asarray(gradient_result.trainable, dtype=bool)
        if trainable.size != parameter_values.size:
            raise ValueError("trainable mask length must match values length")
        gradient = _clip_gradient(
            gradient_result.gradient,
            trainable,
            max_gradient_norm=max_gradient_norm,
        )
        updated: NDArray[np.float64] = parameter_values.copy()
        updated[trainable] -= self.learning_rate * gradient[trainable]
        return _project_bounds(updated, bounds_meta)

    def minimize(
        self,
        objective: ScalarObjective,
        initial_values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        gradient_method: str = "parameter_shift",
        finite_difference_step: float = 1.0e-6,
        bounds: Sequence[ParameterBounds] | None = None,
        max_gradient_norm: float | None = None,
        max_steps: int = 100,
        gradient_tolerance: float = 1.0e-8,
        value_tolerance: float | None = None,
    ) -> OptimizationResult:
        """Run bounded gradient descent with native gradient backends.

        Parameters
        ----------
        objective:
            Scalar-valued objective evaluated on real parameter vectors.
        initial_values:
            Initial real parameter vector before bound projection.
        parameters:
            Optional metadata controlling names and trainable masks.
        rule:
            Optional parameter-shift rule for the parameter-shift backend.
        gradient_method:
            Either ``"parameter_shift"`` or ``"finite_difference"``.
        finite_difference_step:
            Positive central-difference step used by the finite-difference
            backend.
        bounds:
            Optional per-parameter box or periodic bounds.
        max_gradient_norm:
            Optional L2 clipping threshold applied to trainable components.
        max_steps:
            Non-negative maximum number of descent steps.
        gradient_tolerance:
            Non-negative convergence tolerance for the trainable gradient norm.
        value_tolerance:
            Optional non-negative convergence tolerance for objective changes.

        Returns
        -------
        OptimizationResult
            Final values, gradient record, value history, convergence status,
            and best-observed iterate.
        """
        if gradient_method not in {"parameter_shift", "finite_difference"}:
            raise ValueError("gradient_method must be 'parameter_shift' or 'finite_difference'")
        finite_difference_step_value = _as_real_scalar(
            "finite_difference_step", finite_difference_step
        )
        if finite_difference_step_value <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        _validate_max_gradient_norm(max_gradient_norm)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        gradient_tolerance_value = _as_real_scalar("gradient_tolerance", gradient_tolerance)
        if gradient_tolerance_value < 0.0:
            raise ValueError("gradient_tolerance must be finite and non-negative")
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
        history: list[float] = []
        best_values = values.copy()
        best_value = float("inf")
        previous_value: float | None = None

        for step_index in range(max_steps + 1):
            if gradient_method == "finite_difference":
                gradient_result = value_and_finite_difference_grad(
                    objective,
                    values,
                    parameters=parameters,
                    step=finite_difference_step_value,
                )
            else:
                gradient_result = value_and_parameter_shift_grad(
                    objective,
                    values,
                    parameters=parameters,
                    rule=rule,
                )
            history.append(gradient_result.value)
            if gradient_result.value < best_value:
                best_value = gradient_result.value
                best_values = values.copy()
            trainable = np.asarray(gradient_result.trainable, dtype=bool)
            gradient_norm = float(np.linalg.norm(gradient_result.gradient[trainable], ord=2))
            if gradient_norm <= gradient_tolerance_value:
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
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
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=True,
                    reason="value_tolerance",
                    best_values=best_values,
                    best_value=best_value,
                )
            if step_index == max_steps:
                return OptimizationResult(
                    values=values,
                    final_gradient=gradient_result,
                    value_history=tuple(history),
                    steps=step_index,
                    converged=False,
                    reason="max_steps",
                    best_values=best_values,
                    best_value=best_value,
                )
            previous_value = gradient_result.value
            values = self.step(
                values,
                gradient_result,
                bounds=bounds_meta,
                max_gradient_norm=max_gradient_norm,
            )

        raise RuntimeError("unreachable optimizer state")  # pragma: no cover


__all__ = ["DifferentiableOptimizer", "ScalarObjective"]
