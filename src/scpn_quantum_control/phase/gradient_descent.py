# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-Shift Gradient Descent
"""Auditable parameter-shift gradient-descent training for phase objectives."""

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


@dataclass(frozen=True)
class ParameterShiftTrainingStep:
    """One accepted or rejected parameter-shift optimisation step."""

    index: int
    value: float
    gradient_norm: float
    step_size: float
    accepted: bool
    backtracks: int
    evaluations: int
    method: str
    shift_terms: int
    params: FloatArray

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready step metadata."""
        return {
            "index": self.index,
            "value": self.value,
            "gradient_norm": self.gradient_norm,
            "step_size": self.step_size,
            "accepted": self.accepted,
            "backtracks": self.backtracks,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "params": self.params.tolist(),
        }


@dataclass(frozen=True)
class ParameterShiftTrainingResult:
    """Auditable parameter-shift gradient-descent result."""

    initial_value: float
    final_value: float
    best_value: float
    initial_params: FloatArray
    final_params: FloatArray
    best_params: FloatArray
    final_gradient: FloatArray
    final_gradient_norm: float
    steps: tuple[ParameterShiftTrainingStep, ...]
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    backend_plan: QuantumGradientPlan
    method: str
    shift_terms: int
    converged: bool
    reason: str

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
        """Return JSON-ready training provenance."""
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "best_value": self.best_value,
            "initial_params": self.initial_params.tolist(),
            "final_params": self.final_params.tolist(),
            "best_params": self.best_params.tolist(),
            "final_gradient": self.final_gradient.tolist(),
            "final_gradient_norm": self.final_gradient_norm,
            "steps": [step.to_dict() for step in self.steps],
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "evaluations": self.evaluations,
            "backend": self.backend_plan.backend,
            "plan_method": self.backend_plan.method,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "converged": self.converged,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ParameterShiftTrainingCertificate:
    """Machine-checkable convergence certificate for a training result."""

    initial_value: float
    final_value: float
    best_value: float
    value_decrease: float
    monotone_accepted_values: bool
    converged: bool
    within_gradient_tolerance: bool | None
    within_target_value_tolerance: bool | None
    min_decrease_satisfied: bool | None
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    final_gradient_norm: float
    method: str
    shift_terms: int
    reason: str

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
            "within_target_value_tolerance": self.within_target_value_tolerance,
            "min_decrease_satisfied": self.min_decrease_satisfied,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "evaluations": self.evaluations,
            "final_gradient_norm": self.final_gradient_norm,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "reason": self.reason,
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
        raise ValueError(f"parameter-shift training backend is unsupported: {reasons}")


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


def parameter_shift_gradient_descent(
    objective: ScalarObjective,
    initial_params: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    backend: str = "statevector",
    learning_rate: float = 0.1,
    max_steps: int = 100,
    gradient_tolerance: float = 1e-8,
    value_tolerance: float | None = None,
    sufficient_decrease: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 12,
    allow_hardware: bool = False,
) -> ParameterShiftTrainingResult:
    """Minimise a scalar phase objective with native parameter-shift gradients.

    The optimizer is intentionally bounded and fail-closed. It supports local
    deterministic parameter-shift execution with Armijo backtracking and records
    enough metadata to audit convergence, shifted-evaluation cost, and
    multi-frequency rule provenance.
    """
    params = _as_parameter_vector("initial_params", initial_params)
    rate = _positive_float("learning_rate", learning_rate)
    steps_limit = _positive_int("max_steps", max_steps)
    grad_tol = _non_negative_float("gradient_tolerance", gradient_tolerance)
    value_tol = None
    if value_tolerance is not None:
        value_tol = _non_negative_float("value_tolerance", value_tolerance)
    decrease = _positive_float("sufficient_decrease", sufficient_decrease)
    shrink = _positive_float("backtracking_factor", backtracking_factor)
    if shrink >= 1.0:
        raise ValueError("backtracking_factor must be smaller than one")
    backtrack_limit = _positive_int("max_backtracks", max_backtracks)
    term_count = _shift_terms(rule)

    plan = plan_quantum_gradient_backend(
        backend,
        n_params=params.size,
        shift_terms=term_count,
        method="parameter_shift",
        allow_hardware=allow_hardware,
    )
    _ensure_supported(plan)

    current = params.copy()
    initial_result = _value_and_parameter_shift_grad(
        objective,
        current,
        parameters=parameters,
        rule=rule,
    )
    initial_value = _as_objective_value(initial_result.value)
    best_value = initial_value
    best_params = current.copy()
    total_evaluations = initial_result.evaluations
    step_records: list[ParameterShiftTrainingStep] = []
    accepted_steps = 0
    rejected_steps = 0
    reason = "max_steps"
    converged = False

    for step_index in range(1, steps_limit + 1):
        gradient_result = (
            initial_result
            if step_index == 1
            else _value_and_parameter_shift_grad(
                objective,
                current,
                parameters=parameters,
                rule=rule,
            )
        )
        if step_index != 1:
            total_evaluations += gradient_result.evaluations
        base_value = _as_objective_value(gradient_result.value)
        gradient = gradient_result.gradient.astype(np.float64, copy=True)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= grad_tol:
            reason = "gradient_tolerance"
            converged = True
            break

        step_size = rate
        accepted = False
        candidate = current.copy()
        candidate_value = base_value
        backtracks = 0
        trial_evaluations = 0
        directional_decrease = decrease * gradient_norm * gradient_norm
        for trial_index in range(backtrack_limit + 1):
            trial = current - step_size * gradient
            trial_value = _as_objective_value(objective(trial.copy()))
            trial_evaluations += 1
            if trial_value <= base_value - step_size * directional_decrease:
                candidate = trial
                candidate_value = trial_value
                accepted = True
                backtracks = trial_index
                break
            step_size *= shrink

        total_evaluations += trial_evaluations
        if accepted:
            current = candidate.astype(np.float64, copy=True)
            accepted_steps += 1
            if candidate_value < best_value:
                best_value = candidate_value
                best_params = current.copy()
            step_records.append(
                ParameterShiftTrainingStep(
                    index=step_index,
                    value=candidate_value,
                    gradient_norm=gradient_norm,
                    step_size=step_size,
                    accepted=True,
                    backtracks=backtracks,
                    evaluations=gradient_result.evaluations + trial_evaluations,
                    method=gradient_result.method,
                    shift_terms=term_count,
                    params=current.copy(),
                )
            )
            if value_tol is not None and abs(base_value - candidate_value) <= value_tol:
                reason = "value_tolerance"
                converged = True
                break
        else:
            rejected_steps += 1
            reason = "line_search_failed"
            step_records.append(
                ParameterShiftTrainingStep(
                    index=step_index,
                    value=base_value,
                    gradient_norm=gradient_norm,
                    step_size=0.0,
                    accepted=False,
                    backtracks=backtrack_limit + 1,
                    evaluations=gradient_result.evaluations + trial_evaluations,
                    method=gradient_result.method,
                    shift_terms=term_count,
                    params=current.copy(),
                )
            )
            break

    final_result = _value_and_parameter_shift_grad(
        objective,
        current,
        parameters=parameters,
        rule=rule,
    )
    total_evaluations += final_result.evaluations
    final_value = _as_objective_value(final_result.value)
    final_gradient = final_result.gradient.astype(np.float64, copy=True)
    final_gradient_norm = float(np.linalg.norm(final_gradient))
    if final_value < best_value:
        best_value = final_value
        best_params = current.copy()
    if not converged and final_gradient_norm <= grad_tol:
        reason = "gradient_tolerance"
        converged = True

    return ParameterShiftTrainingResult(
        initial_value=initial_value,
        final_value=final_value,
        best_value=best_value,
        initial_params=params.copy(),
        final_params=current.copy(),
        best_params=best_params.copy(),
        final_gradient=final_gradient,
        final_gradient_norm=final_gradient_norm,
        steps=tuple(step_records),
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        evaluations=total_evaluations,
        backend_plan=plan,
        method=final_result.method,
        shift_terms=term_count,
        converged=converged,
        reason=reason,
    )


def validate_parameter_shift_training(
    result: ParameterShiftTrainingResult,
    *,
    gradient_tolerance: float | None = None,
    target_value: float | None = None,
    target_value_tolerance: float = 1e-8,
    min_decrease: float | None = None,
) -> ParameterShiftTrainingCertificate:
    """Return a machine-checkable certificate for a training trace."""
    grad_tol = None
    if gradient_tolerance is not None:
        grad_tol = _non_negative_float("gradient_tolerance", gradient_tolerance)
    target_tol = _non_negative_float("target_value_tolerance", target_value_tolerance)
    decrease_floor = None
    if min_decrease is not None:
        decrease_floor = _non_negative_float("min_decrease", min_decrease)

    accepted_values = result.accepted_value_history
    monotone = all(
        later <= earlier + 1e-10
        for earlier, later in zip(accepted_values, accepted_values[1:], strict=False)
    )
    value_decrease = float(result.initial_value - result.best_value)
    within_gradient = None if grad_tol is None else result.final_gradient_norm <= grad_tol
    within_target = None
    if target_value is not None:
        target = _as_objective_value(target_value)
        within_target = abs(result.best_value - target) <= target_tol
    min_decrease_satisfied = None if decrease_floor is None else value_decrease >= decrease_floor

    return ParameterShiftTrainingCertificate(
        initial_value=result.initial_value,
        final_value=result.final_value,
        best_value=result.best_value,
        value_decrease=value_decrease,
        monotone_accepted_values=monotone,
        converged=result.converged,
        within_gradient_tolerance=within_gradient,
        within_target_value_tolerance=within_target,
        min_decrease_satisfied=min_decrease_satisfied,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        evaluations=result.evaluations,
        final_gradient_norm=result.final_gradient_norm,
        method=result.method,
        shift_terms=result.shift_terms,
        reason=result.reason,
    )


__all__ = [
    "ParameterShiftTrainingCertificate",
    "ParameterShiftTrainingResult",
    "ParameterShiftTrainingStep",
    "parameter_shift_gradient_descent",
    "validate_parameter_shift_training",
]
