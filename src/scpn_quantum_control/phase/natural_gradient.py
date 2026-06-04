# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-Shift Natural Gradient
"""Metric-aware parameter-shift optimisation for supported phase objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import GradientResult, Parameter, ParameterShiftRule
from ..differentiable import value_and_parameter_shift_grad as _value_and_grad
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
MetricTensorProvider = Callable[[FloatArray], ArrayLike]
MetricTensorInput = ArrayLike | MetricTensorProvider | None


@dataclass(frozen=True)
class NaturalGradientDirection:
    """Damped metric solve used for one natural-gradient step."""

    direction: FloatArray
    metric: FloatArray
    regularized_metric: FloatArray
    damping: float
    condition_number: float
    euclidean_gradient_norm: float
    natural_gradient_norm: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready direction metadata."""
        return {
            "direction": self.direction.tolist(),
            "metric": self.metric.tolist(),
            "regularized_metric": self.regularized_metric.tolist(),
            "damping": self.damping,
            "condition_number": self.condition_number,
            "euclidean_gradient_norm": self.euclidean_gradient_norm,
            "natural_gradient_norm": self.natural_gradient_norm,
        }


@dataclass(frozen=True)
class ParameterShiftNaturalGradientStep:
    """One accepted or rejected metric-aware parameter-shift step."""

    index: int
    value: float
    gradient_norm: float
    natural_gradient_norm: float
    direction_norm: float
    step_size: float
    accepted: bool
    backtracks: int
    evaluations: int
    metric_condition_number: float
    damping: float
    method: str
    shift_terms: int
    params: FloatArray

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready step metadata."""
        return {
            "index": self.index,
            "value": self.value,
            "gradient_norm": self.gradient_norm,
            "natural_gradient_norm": self.natural_gradient_norm,
            "direction_norm": self.direction_norm,
            "step_size": self.step_size,
            "accepted": self.accepted,
            "backtracks": self.backtracks,
            "evaluations": self.evaluations,
            "metric_condition_number": self.metric_condition_number,
            "damping": self.damping,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "params": self.params.tolist(),
        }


@dataclass(frozen=True)
class ParameterShiftNaturalGradientResult:
    """Auditable result for parameter-shift natural-gradient optimisation."""

    initial_value: float
    final_value: float
    best_value: float
    initial_params: FloatArray
    final_params: FloatArray
    best_params: FloatArray
    final_gradient: FloatArray
    final_gradient_norm: float
    final_natural_gradient_norm: float
    steps: tuple[ParameterShiftNaturalGradientStep, ...]
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    backend_plan: QuantumGradientPlan
    method: str
    shift_terms: int
    metric_source: str
    damping: float
    max_condition_number: float
    converged: bool
    reason: str
    claim_boundary: str

    @property
    def value_history(self) -> tuple[float, ...]:
        """Return the initial value plus every recorded step value."""
        return (self.initial_value, *(step.value for step in self.steps))

    @property
    def accepted_value_history(self) -> tuple[float, ...]:
        """Return the initial value plus accepted-step values."""
        return (
            self.initial_value,
            *(step.value for step in self.steps if step.accepted),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready optimisation provenance."""
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "best_value": self.best_value,
            "initial_params": self.initial_params.tolist(),
            "final_params": self.final_params.tolist(),
            "best_params": self.best_params.tolist(),
            "final_gradient": self.final_gradient.tolist(),
            "final_gradient_norm": self.final_gradient_norm,
            "final_natural_gradient_norm": self.final_natural_gradient_norm,
            "steps": [step.to_dict() for step in self.steps],
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "evaluations": self.evaluations,
            "backend": self.backend_plan.backend,
            "plan_method": self.backend_plan.method,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "metric_source": self.metric_source,
            "damping": self.damping,
            "max_condition_number": self.max_condition_number,
            "converged": self.converged,
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ParameterShiftNaturalGradientCertificate:
    """Machine-checkable certificate for a natural-gradient training result."""

    initial_value: float
    final_value: float
    best_value: float
    value_decrease: float
    monotone_accepted_values: bool
    converged: bool
    within_gradient_tolerance: bool | None
    within_natural_gradient_tolerance: bool | None
    within_target_value_tolerance: bool | None
    min_decrease_satisfied: bool | None
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    final_gradient_norm: float
    final_natural_gradient_norm: float
    metric_source: str
    method: str
    shift_terms: int
    reason: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready certificate metadata."""
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "best_value": self.best_value,
            "value_decrease": self.value_decrease,
            "monotone_accepted_values": self.monotone_accepted_values,
            "converged": self.converged,
            "within_gradient_tolerance": self.within_gradient_tolerance,
            "within_natural_gradient_tolerance": self.within_natural_gradient_tolerance,
            "within_target_value_tolerance": self.within_target_value_tolerance,
            "min_decrease_satisfied": self.min_decrease_satisfied,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "evaluations": self.evaluations,
            "final_gradient_norm": self.final_gradient_norm,
            "final_natural_gradient_norm": self.final_natural_gradient_norm,
            "metric_source": self.metric_source,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


def _as_parameter_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_objective_value(value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("objective must return a finite scalar value")
    return scalar


def _positive_float(name: str, value: float) -> float:
    scalar = float(value)
    if scalar <= 0.0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar


def _non_negative_float(name: str, value: float) -> float:
    scalar = float(value)
    if scalar < 0.0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shift_terms(rule: ParameterShiftRule | None) -> int:
    if rule is None:
        return 1
    return len(rule.terms)


def _ensure_supported(plan: QuantumGradientPlan) -> None:
    if plan.fail_closed:
        reasons = "; ".join(plan.reasons)
        raise ValueError(f"natural-gradient backend is unsupported: {reasons}")


def _metric_source(metric_tensor: MetricTensorInput) -> str:
    if metric_tensor is None:
        return "identity"
    if callable(metric_tensor):
        return "callable"
    return "array"


def _metric_at(metric_tensor: MetricTensorInput, params: FloatArray) -> FloatArray:
    width = params.size
    raw_metric = np.eye(width, dtype=np.float64)
    if metric_tensor is not None:
        raw_metric = np.asarray(
            metric_tensor(params.copy()) if callable(metric_tensor) else metric_tensor,
            dtype=float,
        )
    if raw_metric.shape != (width, width):
        raise ValueError(f"metric tensor must have shape ({width}, {width})")
    if not np.all(np.isfinite(raw_metric)):
        raise ValueError("metric tensor must contain only finite values")
    metric = raw_metric.astype(np.float64, copy=True)
    if not np.allclose(metric, metric.T, rtol=1e-10, atol=1e-12):
        raise ValueError("metric tensor must be symmetric")
    return metric


def _value_and_parameter_shift_grad(
    objective: ScalarObjective,
    params: FloatArray,
    *,
    parameters: Sequence[Parameter] | None,
    rule: ParameterShiftRule | None,
) -> GradientResult:
    result = _value_and_grad(
        objective,
        params,
        parameters=parameters,
        rule=rule,
    )
    if result.gradient.shape != params.shape:
        raise ValueError("gradient shape must match parameter shape")
    if not np.all(np.isfinite(result.gradient)):
        raise ValueError("gradient must contain only finite values")
    return result


def solve_natural_gradient_direction(
    gradient: ArrayLike,
    metric_tensor: ArrayLike,
    *,
    damping: float = 1e-8,
    max_condition_number: float = 1e12,
) -> NaturalGradientDirection:
    """Solve ``(metric + damping I) direction = gradient`` fail-closed.

    The returned direction is a preconditioned descent direction for
    minimisation steps of the form ``params - step_size * direction``. The
    metric must be symmetric and the regularised solve must produce positive
    natural-gradient energy ``gradient @ direction``.
    """
    grad = _as_parameter_vector("gradient", gradient)
    damp = _non_negative_float("damping", damping)
    max_condition = _positive_float("max_condition_number", max_condition_number)
    metric = _metric_at(metric_tensor, grad)
    regularized = metric + damp * np.eye(grad.size, dtype=np.float64)
    condition_number = float(np.linalg.cond(regularized))
    if not np.isfinite(condition_number):
        raise ValueError("regularized metric condition number must be finite")
    if condition_number > max_condition:
        raise ValueError("regularized metric condition number exceeds max_condition_number")
    eigenvalues = np.linalg.eigvalsh(regularized)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("regularized metric eigenvalues must be finite")
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("regularized metric tensor must be positive definite")
    try:
        direction = np.linalg.solve(regularized, grad).astype(np.float64, copy=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("regularized metric tensor must be invertible") from exc
    if not np.all(np.isfinite(direction)):
        raise ValueError("natural-gradient direction must contain only finite values")
    energy = float(np.dot(grad, direction))
    if energy < -1e-12:
        raise ValueError("regularized metric tensor must define a positive descent direction")
    natural_norm = float(np.sqrt(max(energy, 0.0)))
    return NaturalGradientDirection(
        direction=direction,
        metric=metric,
        regularized_metric=regularized,
        damping=damp,
        condition_number=condition_number,
        euclidean_gradient_norm=float(np.linalg.norm(grad)),
        natural_gradient_norm=natural_norm,
    )


def parameter_shift_natural_gradient_descent(
    objective: ScalarObjective,
    initial_params: ArrayLike,
    *,
    metric_tensor: MetricTensorInput = None,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    backend: str = "statevector",
    learning_rate: float = 0.1,
    max_steps: int = 100,
    gradient_tolerance: float = 1e-8,
    natural_gradient_tolerance: float = 1e-8,
    value_tolerance: float | None = None,
    damping: float = 1e-8,
    max_condition_number: float = 1e12,
    sufficient_decrease: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 12,
    allow_hardware: bool = False,
) -> ParameterShiftNaturalGradientResult:
    """Minimise a phase objective with parameter-shift natural gradients.

    The function composes native parameter-shift gradients with an explicit
    metric tensor supplied by the caller. The default identity metric is an
    auditable preconditioner baseline, not a claim of quantum Fisher metric
    extraction. Hardware backends remain fail-closed through the backend
    planner unless an explicit future policy enables them.
    """
    params = _as_parameter_vector("initial_params", initial_params)
    rate = _positive_float("learning_rate", learning_rate)
    steps_limit = _positive_int("max_steps", max_steps)
    grad_tol = _non_negative_float("gradient_tolerance", gradient_tolerance)
    natural_tol = _non_negative_float("natural_gradient_tolerance", natural_gradient_tolerance)
    value_tol = None
    if value_tolerance is not None:
        value_tol = _non_negative_float("value_tolerance", value_tolerance)
    damp = _non_negative_float("damping", damping)
    max_condition = _positive_float("max_condition_number", max_condition_number)
    decrease = _positive_float("sufficient_decrease", sufficient_decrease)
    shrink = _positive_float("backtracking_factor", backtracking_factor)
    if shrink >= 1.0:
        raise ValueError("backtracking_factor must be smaller than one")
    backtrack_limit = _positive_int("max_backtracks", max_backtracks)
    terms = _shift_terms(rule)
    plan = plan_quantum_gradient_backend(
        backend,
        n_params=params.size,
        shots=None,
        shift_terms=terms,
        allow_hardware=allow_hardware,
    )
    _ensure_supported(plan)

    current = _value_and_parameter_shift_grad(
        objective,
        params,
        parameters=parameters,
        rule=rule,
    )
    current_value = _as_objective_value(current.value)
    current_gradient = current.gradient.astype(np.float64, copy=True)
    evaluations = current.evaluations
    initial_value = current_value
    initial_vector = params.copy()
    best_value = current_value
    best_params = params.copy()
    records: list[ParameterShiftNaturalGradientStep] = []
    accepted_steps = 0
    rejected_steps = 0
    reason = "max_steps"
    converged = False
    final_direction = solve_natural_gradient_direction(
        current_gradient,
        _metric_at(metric_tensor, params),
        damping=damp,
        max_condition_number=max_condition,
    )

    for index in range(steps_limit):
        gradient_norm = float(np.linalg.norm(current_gradient))
        direction = solve_natural_gradient_direction(
            current_gradient,
            _metric_at(metric_tensor, params),
            damping=damp,
            max_condition_number=max_condition,
        )
        final_direction = direction
        if gradient_norm <= grad_tol:
            converged = True
            reason = "gradient_tolerance"
            break
        if direction.natural_gradient_norm <= natural_tol:
            converged = True
            reason = "natural_gradient_tolerance"
            break
        if value_tol is not None and best_value <= value_tol:
            converged = True
            reason = "value_tolerance"
            break

        step_size = rate
        accepted = False
        trial_value = current_value
        trial_params = params.copy()
        backtracks = 0
        descent_energy = max(float(np.dot(current_gradient, direction.direction)), 0.0)
        for attempt in range(backtrack_limit + 1):
            trial_params = params - step_size * direction.direction
            trial_value = _as_objective_value(objective(trial_params.copy()))
            evaluations += 1
            threshold = current_value - decrease * step_size * descent_energy
            if trial_value <= threshold:
                accepted = True
                backtracks = attempt
                break
            step_size *= shrink
            backtracks = attempt + 1

        if not accepted:
            rejected_steps += 1
            records.append(
                ParameterShiftNaturalGradientStep(
                    index=index,
                    value=current_value,
                    gradient_norm=gradient_norm,
                    natural_gradient_norm=direction.natural_gradient_norm,
                    direction_norm=float(np.linalg.norm(direction.direction)),
                    step_size=step_size,
                    accepted=False,
                    backtracks=backtracks,
                    evaluations=evaluations,
                    metric_condition_number=direction.condition_number,
                    damping=damp,
                    method=current.method,
                    shift_terms=terms,
                    params=params.copy(),
                )
            )
            reason = "line_search_failed"
            break

        accepted_steps += 1
        params = trial_params.astype(np.float64, copy=True)
        current = _value_and_parameter_shift_grad(
            objective,
            params,
            parameters=parameters,
            rule=rule,
        )
        evaluations += current.evaluations
        current_value = _as_objective_value(current.value)
        current_gradient = current.gradient.astype(np.float64, copy=True)
        if current_value < best_value:
            best_value = current_value
            best_params = params.copy()
        records.append(
            ParameterShiftNaturalGradientStep(
                index=index,
                value=current_value,
                gradient_norm=float(np.linalg.norm(current_gradient)),
                natural_gradient_norm=direction.natural_gradient_norm,
                direction_norm=float(np.linalg.norm(direction.direction)),
                step_size=step_size,
                accepted=True,
                backtracks=backtracks,
                evaluations=evaluations,
                metric_condition_number=direction.condition_number,
                damping=damp,
                method=current.method,
                shift_terms=terms,
                params=params.copy(),
            )
        )
    else:
        gradient_norm = float(np.linalg.norm(current_gradient))
        if gradient_norm <= grad_tol:
            converged = True
            reason = "gradient_tolerance"
        elif final_direction.natural_gradient_norm <= natural_tol:
            converged = True
            reason = "natural_gradient_tolerance"
        elif value_tol is not None and best_value <= value_tol:
            converged = True
            reason = "value_tolerance"

    final_direction = solve_natural_gradient_direction(
        current_gradient,
        _metric_at(metric_tensor, params),
        damping=damp,
        max_condition_number=max_condition,
    )
    return ParameterShiftNaturalGradientResult(
        initial_value=initial_value,
        final_value=current_value,
        best_value=best_value,
        initial_params=initial_vector,
        final_params=params.copy(),
        best_params=best_params,
        final_gradient=current_gradient.copy(),
        final_gradient_norm=float(np.linalg.norm(current_gradient)),
        final_natural_gradient_norm=final_direction.natural_gradient_norm,
        steps=tuple(records),
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        evaluations=evaluations,
        backend_plan=plan,
        method=current.method,
        shift_terms=terms,
        metric_source=_metric_source(metric_tensor),
        damping=damp,
        max_condition_number=max_condition,
        converged=converged,
        reason=reason,
        claim_boundary=(
            "local parameter-shift gradients with caller-supplied metric; "
            "identity metric is preconditioner baseline only; no hardware or "
            "arbitrary-circuit quantum Fisher extraction is implied"
        ),
    )


def validate_natural_gradient_training(
    result: ParameterShiftNaturalGradientResult,
    *,
    gradient_tolerance: float | None = None,
    natural_gradient_tolerance: float | None = None,
    target_value: float | None = None,
    target_value_tolerance: float | None = None,
    min_decrease: float | None = None,
) -> ParameterShiftNaturalGradientCertificate:
    """Validate natural-gradient descent provenance against explicit gates."""
    grad_ok = None
    if gradient_tolerance is not None:
        grad_ok = result.final_gradient_norm <= _non_negative_float(
            "gradient_tolerance",
            gradient_tolerance,
        )
    natural_ok = None
    if natural_gradient_tolerance is not None:
        natural_ok = result.final_natural_gradient_norm <= _non_negative_float(
            "natural_gradient_tolerance",
            natural_gradient_tolerance,
        )
    target_ok = None
    if target_value is not None:
        tolerance = 0.0
        if target_value_tolerance is not None:
            tolerance = _non_negative_float("target_value_tolerance", target_value_tolerance)
        target_ok = result.best_value <= float(target_value) + tolerance
    elif target_value_tolerance is not None:
        raise ValueError("target_value_tolerance requires target_value")
    decrease_ok = None
    value_decrease = result.initial_value - result.best_value
    if min_decrease is not None:
        decrease_ok = value_decrease >= _non_negative_float("min_decrease", min_decrease)
    accepted_history = result.accepted_value_history
    monotone = all(
        later <= earlier + 1e-12
        for earlier, later in zip(accepted_history, accepted_history[1:], strict=False)
    )
    return ParameterShiftNaturalGradientCertificate(
        initial_value=result.initial_value,
        final_value=result.final_value,
        best_value=result.best_value,
        value_decrease=value_decrease,
        monotone_accepted_values=monotone,
        converged=result.converged,
        within_gradient_tolerance=grad_ok,
        within_natural_gradient_tolerance=natural_ok,
        within_target_value_tolerance=target_ok,
        min_decrease_satisfied=decrease_ok,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        evaluations=result.evaluations,
        final_gradient_norm=result.final_gradient_norm,
        final_natural_gradient_norm=result.final_natural_gradient_norm,
        metric_source=result.metric_source,
        method=result.method,
        shift_terms=result.shift_terms,
        reason=result.reason,
        claim_boundary=result.claim_boundary,
    )
