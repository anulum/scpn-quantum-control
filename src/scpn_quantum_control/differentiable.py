# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- native differentiable programming primitives
"""Native differentiable-programming primitives for SCPN quantum objectives.

The base layer is backend-neutral parameter-shift differentiation for scalar
objectives. Optional JAX support is exposed as an adapter without making JAX a
runtime dependency of the core package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]
VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    """Return a real numeric vector without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must contain real numeric scalars")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars") from exc
    return cast(NDArray[np.float64], array)


def _as_real_scalar(name: str, value: object) -> float:
    """Return an explicit real numeric scalar without implicit coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real numeric scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclass(frozen=True)
class Parameter:
    """One differentiable scalar parameter in an SCPN objective."""

    name: str
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be non-empty")
        if not isinstance(self.trainable, bool):
            raise ValueError("parameter trainable flag must be a boolean")


@dataclass(frozen=True)
class ParameterBounds:
    """Closed interval constraint for one differentiable scalar parameter."""

    lower: float | None = None
    upper: float | None = None
    periodic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.periodic, bool):
            raise ValueError("periodic flag must be a boolean")
        lower = None if self.lower is None else _as_real_scalar("lower bound", self.lower)
        upper = None if self.upper is None else _as_real_scalar("upper bound", self.upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower bound must be less than or equal to upper bound")
        if self.periodic:
            if lower is None or upper is None:
                raise ValueError("periodic bounds require finite lower and upper bounds")
            if lower == upper:
                raise ValueError("periodic bounds require lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class ParameterShiftRule:
    """Two-point parameter-shift rule for one-generator rotation parameters."""

    shift: float = float(np.pi / 2.0)
    coefficient: float = 0.5

    def __post_init__(self) -> None:
        shift = _as_real_scalar("shift", self.shift)
        coefficient = _as_real_scalar("coefficient", self.coefficient)
        if shift <= 0.0:
            raise ValueError("shift must be finite and positive")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class GradientResult:
    """Value, gradient, and provenance returned by a differentiable backend."""

    value: float
    gradient: NDArray[np.float64]
    method: str
    shift: float | None
    coefficient: float | None
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("gradient result value", self.value)
        gradient = _as_real_numeric_array("gradient", self.gradient)
        if gradient.ndim != 1:
            raise ValueError("gradient must be a one-dimensional array")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values")
        if not self.method:
            raise ValueError("gradient method must be non-empty")
        shift = None if self.shift is None else _as_real_scalar("gradient shift", self.shift)
        coefficient = (
            None
            if self.coefficient is None
            else _as_real_scalar("gradient coefficient", self.coefficient)
        )
        if shift is not None and shift <= 0.0:
            raise ValueError("gradient shift must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class OptimizationResult:
    """Bounded gradient-descent result with convergence provenance."""

    values: NDArray[np.float64]
    final_gradient: GradientResult
    value_history: tuple[float, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64] | None = None
    best_value: float | None = None

    def __post_init__(self) -> None:
        values = _as_parameter_array(self.values)
        if values.size != self.final_gradient.gradient.size:
            raise ValueError("optimized values length must match gradient length")
        if not self.value_history:
            raise ValueError("value_history must contain at least one value")
        history = tuple(_as_real_scalar("value_history item", item) for item in self.value_history)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("optimization steps must be a non-negative integer")
        if not isinstance(self.converged, bool):
            raise ValueError("optimization converged flag must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("optimization reason must be non-empty")
        best_values = values if self.best_values is None else _as_parameter_array(self.best_values)
        if best_values.size != values.size:
            raise ValueError("best_values length must match optimized values length")
        best_value = (
            min(history)
            if self.best_value is None
            else _as_real_scalar("best_value", self.best_value)
        )
        if best_value > min(history) + 1.0e-12:
            raise ValueError("best_value must not exceed the minimum value_history entry")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_history", history)
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class GradientCheckResult:
    """Consistency check between two differentiable gradient estimators."""

    reference: GradientResult
    candidate: GradientResult
    max_abs_error: float
    l2_error: float
    value_delta: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        if self.reference.gradient.shape != self.candidate.gradient.shape:
            raise ValueError("gradient check operands must have matching shapes")
        max_abs_error = _as_real_scalar("max_abs_error", self.max_abs_error)
        l2_error = _as_real_scalar("l2_error", self.l2_error)
        value_delta = _as_real_scalar("value_delta", self.value_delta)
        tolerance = _as_real_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if l2_error < 0.0:
            raise ValueError("l2_error must be non-negative")
        if value_delta < 0.0:
            raise ValueError("value_delta must be non-negative")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("gradient check passed flag must be a boolean")
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "l2_error", l2_error)
        object.__setattr__(self, "value_delta", value_delta)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class JacobianResult:
    """Value, Jacobian, and provenance for a vector-valued objective."""

    value: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_numeric_array("jacobian value", self.value)
        jacobian = _as_real_numeric_array("jacobian", self.jacobian)
        if value.ndim != 1:
            raise ValueError("jacobian value must be a one-dimensional array")
        if jacobian.ndim != 2:
            raise ValueError("jacobian must be a two-dimensional array")
        if jacobian.shape[0] != value.size:
            raise ValueError("jacobian row count must match value length")
        if not np.all(np.isfinite(value)):
            raise ValueError("jacobian value must contain only finite values")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian must contain only finite values")
        if not self.method:
            raise ValueError("jacobian method must be non-empty")
        step = _as_real_scalar("jacobian step", self.step)
        if step <= 0.0:
            raise ValueError("jacobian step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("jacobian evaluations must be non-negative")
        if len(self.parameter_names) != jacobian.shape[1]:
            raise ValueError("parameter_names length must match jacobian column count")
        if len(self.trainable) != jacobian.shape[1]:
            raise ValueError("trainable mask length must match jacobian column count")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class HessianResult:
    """Value, Hessian, and provenance for a scalar objective."""

    value: float
    hessian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        value = _as_real_scalar("hessian value", self.value)
        hessian = _as_real_numeric_array("hessian", self.hessian)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("hessian must be a square two-dimensional array")
        if not np.all(np.isfinite(hessian)):
            raise ValueError("hessian must contain only finite values")
        if not self.method:
            raise ValueError("hessian method must be non-empty")
        step = _as_real_scalar("hessian step", self.step)
        if step <= 0.0:
            raise ValueError("hessian step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("hessian evaluations must be non-negative")
        if len(self.parameter_names) != hessian.shape[1]:
            raise ValueError("parameter_names length must match hessian dimension")
        if len(self.trainable) != hessian.shape[1]:
            raise ValueError("trainable mask length must match hessian dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        if not np.allclose(hessian, hessian.T, atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("hessian must be symmetric")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "step", step)


@dataclass(frozen=True)
class NaturalGradientResult:
    """Metric-preconditioned gradient with solve provenance."""

    base_gradient: GradientResult
    metric: NDArray[np.float64]
    natural_gradient: NDArray[np.float64]
    damping: float
    condition_number: float

    def __post_init__(self) -> None:
        metric = _as_real_numeric_array("natural-gradient metric", self.metric)
        natural_gradient = _as_real_numeric_array("natural_gradient", self.natural_gradient)
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            raise ValueError("natural-gradient metric must be a square matrix")
        if metric.shape[0] != self.base_gradient.gradient.size:
            raise ValueError("natural-gradient metric dimension must match gradient length")
        if natural_gradient.shape != self.base_gradient.gradient.shape:
            raise ValueError("natural_gradient shape must match gradient shape")
        if not np.all(np.isfinite(metric)):
            raise ValueError("natural-gradient metric must contain only finite values")
        if not np.all(np.isfinite(natural_gradient)):
            raise ValueError("natural_gradient must contain only finite values")
        if not np.allclose(metric, metric.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("natural-gradient metric must be symmetric")
        damping = _as_real_scalar("natural-gradient damping", self.damping)
        if damping < 0.0:
            raise ValueError("natural-gradient damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "natural-gradient condition_number", self.condition_number
        )
        if condition_number < 1.0:
            raise ValueError("natural-gradient condition_number must be at least 1")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "natural_gradient", natural_gradient)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class DifferentiableOptimizer:
    """Small native gradient-descent optimizer for differentiable SCPN parameters."""

    learning_rate: float = 0.01

    def __post_init__(self) -> None:
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
        """Return one gradient-descent update respecting the trainable mask."""

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
        """Run bounded gradient descent with parameter-shift gradients."""

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

        raise RuntimeError("unreachable optimizer state")


def _as_parameter_array(values: ArrayLike) -> NDArray[np.float64]:
    array = _as_real_numeric_array("parameters", values)
    if array.ndim != 1:
        raise ValueError("parameters must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("parameters must contain only finite values")
    return array


def _as_scalar(value: float | int | np.floating[Any] | NDArray[np.float64]) -> float:
    try:
        scalar = _as_real_scalar("differentiable objective", value)
    except ValueError as exc:
        if "scalar" in str(exc):
            raise ValueError("differentiable objective must return a scalar") from exc
        raise
    if not np.isfinite(scalar):
        raise ValueError("differentiable objective returned a non-finite scalar")
    return scalar


def _as_vector_output(value: ArrayLike) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("differentiable vector objective", value)
    if vector.ndim != 1:
        raise ValueError("differentiable vector objective must return a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("differentiable vector objective returned non-finite values")
    return vector


def _normalise_parameters(
    values: NDArray[np.float64],
    parameters: Sequence[Parameter] | None,
) -> tuple[Parameter, ...]:
    if parameters is None:
        return tuple(Parameter(f"theta_{index}") for index in range(values.size))
    normalised = tuple(parameters)
    if len(normalised) != values.size:
        raise ValueError("parameters length must match values length")
    if len({parameter.name for parameter in normalised}) != len(normalised):
        raise ValueError("parameter names must be unique")
    return normalised


def _normalise_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds] | None,
) -> tuple[ParameterBounds, ...]:
    if bounds is None:
        return tuple(ParameterBounds() for _ in range(values.size))
    normalised = tuple(bounds)
    if len(normalised) != values.size:
        raise ValueError("bounds length must match values length")
    if any(not isinstance(item, ParameterBounds) for item in normalised):
        raise ValueError("bounds must contain ParameterBounds instances")
    return normalised


def _project_bounds(
    values: NDArray[np.float64],
    bounds: Sequence[ParameterBounds],
) -> NDArray[np.float64]:
    projected = values.copy()
    for index, bound in enumerate(bounds):
        if bound.periodic:
            lower = cast(float, bound.lower)
            upper = cast(float, bound.upper)
            width = upper - lower
            projected[index] = ((projected[index] - lower) % width) + lower
            continue
        if bound.lower is not None and projected[index] < bound.lower:
            projected[index] = bound.lower
        if bound.upper is not None and projected[index] > bound.upper:
            projected[index] = bound.upper
    return cast(NDArray[np.float64], projected)


def _validate_max_gradient_norm(max_gradient_norm: float | None) -> float | None:
    if max_gradient_norm is None:
        return None
    max_norm = _as_real_scalar("max_gradient_norm", max_gradient_norm)
    if max_norm <= 0.0:
        raise ValueError("max_gradient_norm must be finite and positive")
    return max_norm


def _clip_gradient(
    gradient: NDArray[np.float64],
    trainable: NDArray[np.bool_],
    *,
    max_gradient_norm: float | None,
) -> NDArray[np.float64]:
    max_norm = _validate_max_gradient_norm(max_gradient_norm)
    clipped = gradient.copy()
    if max_norm is None or not np.any(trainable):
        return cast(NDArray[np.float64], clipped)
    trainable_norm = float(np.linalg.norm(clipped[trainable], ord=2))
    if trainable_norm > max_norm:
        clipped[trainable] *= max_norm / trainable_norm
    return cast(NDArray[np.float64], clipped)


def parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return the parameter-shift gradient of a scalar objective."""

    result = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    return result.gradient


def batch_parameter_shift_gradient(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return stacked parameter-shift gradients for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    rows = [
        parameter_shift_gradient(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    ]
    return np.vstack(rows)


def batch_value_and_parameter_shift_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> tuple[GradientResult, ...]:
    """Return full parameter-shift results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    )


def value_and_parameter_shift_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and its native parameter-shift gradient."""

    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    shift_rule = rule or ParameterShiftRule()
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += shift_rule.shift
        minus[index] -= shift_rule.shift
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = shift_rule.coefficient * (plus_value - minus_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="parameter_shift",
        shift=shift_rule.shift,
        coefficient=shift_rule.coefficient,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference gradient for scalar diagnostics."""

    result = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return result.gradient


def batch_value_and_finite_difference_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[GradientResult, ...]:
    """Return full finite-difference results for multiple scalar objectives."""

    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_finite_difference_grad(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    )


def value_and_finite_difference_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> GradientResult:
    """Evaluate a scalar objective and central finite-difference gradient."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = (plus_value - minus_value) / (2.0 * step_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="finite_difference_central",
        shift=step_value,
        coefficient=1.0 / (2.0 * step_value),
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian for vector objectives."""

    return value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).jacobian


def value_and_finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and its central finite-difference Jacobian."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    base_value = _as_vector_output(objective(parameter_values.copy()))
    jacobian = np.zeros((base_value.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        evaluations += 2
        jacobian[:, index] = (plus_value - minus_value) / (2.0 * step_value)

    return JacobianResult(
        value=base_value,
        jacobian=jacobian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian for scalar objectives."""

    return value_and_finite_difference_hessian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).hessian


def value_and_finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> HessianResult:
    """Evaluate a scalar objective and central finite-difference Hessian."""

    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    base_value = _as_scalar(objective(parameter_values.copy()))
    hessian = np.zeros((parameter_values.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for row in range(parameter_values.size):
        if not trainable[row]:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[row] += step_value
        minus[row] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        hessian[row, row] = (plus_value - 2.0 * base_value + minus_value) / (step_value**2)

        for column in range(row + 1, parameter_values.size):
            if not trainable[column]:
                continue
            plus_plus = parameter_values.copy()
            plus_minus = parameter_values.copy()
            minus_plus = parameter_values.copy()
            minus_minus = parameter_values.copy()
            plus_plus[row] += step_value
            plus_plus[column] += step_value
            plus_minus[row] += step_value
            plus_minus[column] -= step_value
            minus_plus[row] -= step_value
            minus_plus[column] += step_value
            minus_minus[row] -= step_value
            minus_minus[column] -= step_value
            mixed = (
                _as_scalar(objective(plus_plus))
                - _as_scalar(objective(plus_minus))
                - _as_scalar(objective(minus_plus))
                + _as_scalar(objective(minus_minus))
            ) / (4.0 * step_value**2)
            evaluations += 4
            hessian[row, column] = mixed
            hessian[column, row] = mixed

    return HessianResult(
        value=base_value,
        hessian=hessian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def natural_gradient(
    gradient_result: GradientResult,
    metric: ArrayLike,
    *,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Solve ``metric @ natural_gradient = gradient`` on trainable parameters."""

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


def check_parameter_shift_consistency(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> GradientCheckResult:
    """Compare parameter-shift gradients against central finite differences."""

    tolerance_value = _as_real_scalar("gradient check tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("gradient check tolerance must be finite and non-negative")
    candidate = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    reference = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=finite_difference_step,
    )
    delta = candidate.gradient - reference.gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    value_delta = float(abs(candidate.value - reference.value))
    return GradientCheckResult(
        reference=reference,
        candidate=candidate,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        value_delta=value_delta,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
    )


def is_jax_autodiff_available() -> bool:
    """Return whether JAX autodiff can be imported in the active environment."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ImportError:
        return False
    return True


def jax_value_and_grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate a JAX scalar objective and return ``(value, gradient)``."""

    parameter_values = _as_parameter_array(values)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("JAX autodiff is unavailable; install the [jax] extra") from exc

    def wrapped(raw_values: Any) -> Any:
        return objective(raw_values)

    value, gradient = jax.value_and_grad(wrapped)(jnp.asarray(parameter_values))
    result_value = _as_real_scalar("JAX objective value", value)
    result_gradient = _as_real_numeric_array("JAX gradient", gradient)
    if result_gradient.shape != parameter_values.shape:
        raise ValueError("JAX gradient shape must match parameter shape")
    if not np.all(np.isfinite(result_gradient)):
        raise ValueError("JAX gradient must contain only finite values")
    return result_value, result_gradient


__all__ = [
    "DifferentiableOptimizer",
    "GradientCheckResult",
    "GradientResult",
    "HessianResult",
    "JacobianResult",
    "NaturalGradientResult",
    "OptimizationResult",
    "Parameter",
    "ParameterBounds",
    "ParameterShiftRule",
    "batch_parameter_shift_gradient",
    "batch_value_and_finite_difference_grad",
    "batch_value_and_parameter_shift_grad",
    "check_parameter_shift_consistency",
    "finite_difference_gradient",
    "finite_difference_hessian",
    "finite_difference_jacobian",
    "is_jax_autodiff_available",
    "jax_value_and_grad",
    "natural_gradient",
    "parameter_shift_gradient",
    "value_and_finite_difference_grad",
    "value_and_finite_difference_hessian",
    "value_and_finite_difference_jacobian",
    "value_and_parameter_shift_grad",
]
