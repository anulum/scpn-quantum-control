# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Phase Objectives
"""Composable differentiable objectives for phase-control training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TermValueFn = Callable[[FloatArray], float]
TermGradientFn = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class ObjectiveTermValue:
    """One weighted term contribution in a composed objective evaluation."""

    name: str
    kind: str
    raw_value: float
    weight: float
    weighted_value: float
    gradient_mode: str
    parameter_shift_compatible: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready term contribution metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "raw_value": self.raw_value,
            "weight": self.weight,
            "weighted_value": self.weighted_value,
            "gradient_mode": self.gradient_mode,
            "parameter_shift_compatible": self.parameter_shift_compatible,
        }


@dataclass(frozen=True)
class ObjectiveGradientEvaluation:
    """Value, gradient, and term breakdown for a composed objective."""

    value: float
    gradient: FloatArray
    terms: tuple[ObjectiveTermValue, ...]
    parameter_shift_compatible: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready objective-gradient evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "terms": [term.to_dict() for term in self.terms],
            "parameter_shift_compatible": self.parameter_shift_compatible,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ObjectiveTerm:
    """Weighted differentiable objective term with explicit gradient semantics."""

    name: str
    kind: str
    weight: float
    value_fn: TermValueFn
    gradient_fn: TermGradientFn
    gradient_mode: str
    parameter_shift_compatible: bool
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("term name must be non-empty")
        if not self.kind:
            raise ValueError("term kind must be non-empty")
        weight = _as_finite_scalar("weight", self.weight)
        if weight < 0.0:
            raise ValueError("term weight must be non-negative")
        if self.gradient_mode not in {"parameter_shift", "analytic"}:
            raise ValueError("gradient_mode must be 'parameter_shift' or 'analytic'")
        if self.gradient_mode == "analytic" and self.parameter_shift_compatible:
            raise ValueError("analytic-only terms must not be parameter-shift compatible")

    def evaluate(self, params: ArrayLike) -> ObjectiveTermValue:
        """Evaluate one weighted term at ``params``."""
        vector = _as_vector("params", params)
        raw_value = _as_finite_scalar("term value", self.value_fn(vector.copy()))
        weighted_value = float(self.weight * raw_value)
        return ObjectiveTermValue(
            name=self.name,
            kind=self.kind,
            raw_value=raw_value,
            weight=float(self.weight),
            weighted_value=weighted_value,
            gradient_mode=self.gradient_mode,
            parameter_shift_compatible=self.parameter_shift_compatible,
        )

    def gradient(self, params: ArrayLike) -> FloatArray:
        """Evaluate one weighted term gradient at ``params``."""
        vector = _as_vector("params", params)
        gradient = _as_vector("term gradient", self.gradient_fn(vector.copy()), width=vector.size)
        return (float(self.weight) * gradient).astype(np.float64, copy=False)

    def to_dict(self) -> dict[str, object]:
        """Return serialisable static term metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "weight": float(self.weight),
            "gradient_mode": self.gradient_mode,
            "parameter_shift_compatible": self.parameter_shift_compatible,
            "description": self.description,
        }


@dataclass(frozen=True)
class ComposedPhaseObjective:
    """Named weighted sum of differentiable phase-control objective terms."""

    terms: tuple[ObjectiveTerm, ...]
    name: str = "phase_control_objective"
    claim_boundary: str = (
        "local differentiable phase-control objective; analytic classical "
        "penalties are not silently promoted to parameter-shift quantum terms"
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("objective name must be non-empty")
        if not self.terms:
            raise ValueError("objective must contain at least one term")
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("objective term names must be unique")

    @property
    def parameter_shift_compatible(self) -> bool:
        """Return whether every term is compatible with parameter-shift."""
        return all(term.parameter_shift_compatible for term in self.terms)

    @property
    def term_names(self) -> tuple[str, ...]:
        """Return objective term names."""
        return tuple(term.name for term in self.terms)

    def evaluate(self, params: ArrayLike) -> ObjectiveGradientEvaluation:
        """Evaluate objective value, exact term gradients, and term breakdown."""
        vector = _as_vector("params", params)
        terms = tuple(term.evaluate(vector) for term in self.terms)
        gradient = np.zeros_like(vector)
        for term in self.terms:
            gradient += term.gradient(vector)
        total_value = _as_finite_scalar(
            "objective value",
            sum(term.weighted_value for term in terms),
        )
        return ObjectiveGradientEvaluation(
            value=total_value,
            gradient=gradient,
            terms=terms,
            parameter_shift_compatible=self.parameter_shift_compatible,
            claim_boundary=self.claim_boundary,
        )

    def __call__(self, params: ArrayLike) -> float:
        """Return the scalar objective value."""
        return self.evaluate(params).value

    def require_parameter_shift_compatible(self) -> None:
        """Fail closed if any term is not parameter-shift compatible."""
        incompatible = [term.name for term in self.terms if not term.parameter_shift_compatible]
        if incompatible:
            names = ", ".join(incompatible)
            raise ValueError(f"objective contains non-parameter-shift terms: {names}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready static objective metadata."""
        return {
            "name": self.name,
            "terms": [term.to_dict() for term in self.terms],
            "parameter_shift_compatible": self.parameter_shift_compatible,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ComposedObjectiveTrainingStep:
    """One accepted or rejected composed-objective optimisation step."""

    index: int
    value: float
    gradient_norm: float
    step_size: float
    accepted: bool
    backtracks: int
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
            "params": self.params.tolist(),
        }


@dataclass(frozen=True)
class ComposedObjectiveTrainingResult:
    """Auditable training result for a composed phase objective."""

    initial_value: float
    final_value: float
    best_value: float
    initial_params: FloatArray
    final_params: FloatArray
    best_params: FloatArray
    final_gradient: FloatArray
    final_gradient_norm: float
    steps: tuple[ComposedObjectiveTrainingStep, ...]
    accepted_steps: int
    rejected_steps: int
    converged: bool
    reason: str
    objective: ComposedPhaseObjective
    claim_boundary: str

    @property
    def accepted_value_history(self) -> tuple[float, ...]:
        """Return initial value plus accepted-step values."""
        return (
            self.initial_value,
            *(step.value for step in self.steps if step.accepted),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready training evidence."""
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
            "converged": self.converged,
            "reason": self.reason,
            "objective": self.objective.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ComposedObjectiveTrainingCertificate:
    """Machine-checkable certificate for composed-objective training."""

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
    final_gradient_norm: float
    parameter_shift_compatible: bool
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
            "within_target_value_tolerance": self.within_target_value_tolerance,
            "min_decrease_satisfied": self.min_decrease_satisfied,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "final_gradient_norm": self.final_gradient_norm,
            "parameter_shift_compatible": self.parameter_shift_compatible,
            "claim_boundary": self.claim_boundary,
        }


def _as_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_float(name: str, value: float) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _non_negative_float(name: str, value: float) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_vector(name: str, values: ArrayLike, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _broadcast_vector(name: str, values: ArrayLike | float, width: int) -> FloatArray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim == 0:
        return np.full(width, float(raw), dtype=np.float64)
    return _as_vector(name, raw, width=width)


def _validate_pairs(pairs: Sequence[tuple[int, int]], width: int) -> tuple[tuple[int, int], ...]:
    if not pairs:
        raise ValueError("pairs must contain at least one index pair")
    validated: list[tuple[int, int]] = []
    for left, right in pairs:
        if left == right:
            raise ValueError("symmetry pairs must reference distinct parameters")
        if left < 0 or right < 0 or left >= width or right >= width:
            raise ValueError("symmetry pair index out of bounds")
        validated.append((int(left), int(right)))
    return tuple(validated)


def _softplus(values: FloatArray) -> FloatArray:
    return cast(FloatArray, np.logaddexp(values, 0.0).astype(np.float64, copy=False))


def _sigmoid(values: FloatArray) -> FloatArray:
    return cast(FloatArray, (1.0 / (1.0 + np.exp(-values))).astype(np.float64, copy=False))


def phase_energy_term(
    width: int,
    *,
    weights: ArrayLike | float = 1.0,
    term_weight: float = 1.0,
    name: str = "phase_energy",
) -> ObjectiveTerm:
    """Build a parameter-shift-compatible ``sum(w_i * (1 - cos(theta_i)))`` term."""
    if width <= 0:
        raise ValueError("width must be positive")
    local_weights = _broadcast_vector("weights", weights, width)

    def value(params: FloatArray) -> float:
        vector = _as_vector("params", params, width=width)
        return float(np.sum(local_weights * (1.0 - np.cos(vector))))

    def gradient(params: FloatArray) -> FloatArray:
        vector = _as_vector("params", params, width=width)
        return cast(
            FloatArray,
            (local_weights * np.sin(vector)).astype(np.float64, copy=False),
        )

    return ObjectiveTerm(
        name=name,
        kind="energy",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="parameter_shift",
        parameter_shift_compatible=True,
        description="periodic phase energy term with exact sinusoidal gradient",
    )


def phase_fidelity_target_term(
    target: ArrayLike,
    *,
    term_weight: float = 1.0,
    name: str = "phase_fidelity_target",
) -> ObjectiveTerm:
    """Build a periodic infidelity term ``mean(1 - cos(theta - target))``."""
    target_vector = _as_vector("target", target)
    width = target_vector.size

    def value(params: FloatArray) -> float:
        vector = _as_vector("params", params, width=width)
        return float(np.mean(1.0 - np.cos(vector - target_vector)))

    def gradient(params: FloatArray) -> FloatArray:
        vector = _as_vector("params", params, width=width)
        return cast(
            FloatArray,
            (np.sin(vector - target_vector) / width).astype(np.float64, copy=False),
        )

    return ObjectiveTerm(
        name=name,
        kind="fidelity",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="parameter_shift",
        parameter_shift_compatible=True,
        description="periodic target-phase infidelity term",
    )


def periodic_regularization_term(
    center: ArrayLike,
    *,
    term_weight: float = 1.0,
    name: str = "periodic_regularization",
) -> ObjectiveTerm:
    """Build a periodic regularizer around a reference phase vector."""
    center_vector = _as_vector("center", center)
    width = center_vector.size

    def value(params: FloatArray) -> float:
        vector = _as_vector("params", params, width=width)
        return float(np.mean(1.0 - np.cos(vector - center_vector)))

    def gradient(params: FloatArray) -> FloatArray:
        vector = _as_vector("params", params, width=width)
        return cast(
            FloatArray,
            (np.sin(vector - center_vector) / width).astype(np.float64, copy=False),
        )

    return ObjectiveTerm(
        name=name,
        kind="regularization",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="parameter_shift",
        parameter_shift_compatible=True,
        description="periodic phase regularization compatible with parameter-shift",
    )


def phase_symmetry_penalty_term(
    width: int,
    pairs: Sequence[tuple[int, int]],
    *,
    offsets: ArrayLike | float = 0.0,
    term_weight: float = 1.0,
    name: str = "phase_symmetry_penalty",
) -> ObjectiveTerm:
    """Build a periodic pair-symmetry penalty over selected phase pairs."""
    if width <= 0:
        raise ValueError("width must be positive")
    pair_tuple = _validate_pairs(pairs, width)
    offset_vector = _broadcast_vector("offsets", offsets, len(pair_tuple))

    def value(params: FloatArray) -> float:
        vector = _as_vector("params", params, width=width)
        total = 0.0
        for index, (left, right) in enumerate(pair_tuple):
            total += 1.0 - np.cos(vector[left] - vector[right] - offset_vector[index])
        return float(total / len(pair_tuple))

    def gradient(params: FloatArray) -> FloatArray:
        vector = _as_vector("params", params, width=width)
        grad = np.zeros(width, dtype=np.float64)
        scale = 1.0 / len(pair_tuple)
        for index, (left, right) in enumerate(pair_tuple):
            diff = vector[left] - vector[right] - offset_vector[index]
            contribution = float(np.sin(diff) * scale)
            grad[left] += contribution
            grad[right] -= contribution
        return grad

    return ObjectiveTerm(
        name=name,
        kind="symmetry",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="parameter_shift",
        parameter_shift_compatible=True,
        description="periodic pair-symmetry penalty compatible with parameter-shift",
    )


def smooth_box_safety_penalty_term(
    lower: ArrayLike | float,
    upper: ArrayLike | float,
    *,
    width: int,
    sharpness: float = 8.0,
    term_weight: float = 1.0,
    name: str = "smooth_box_safety_penalty",
) -> ObjectiveTerm:
    """Build a smooth analytic penalty for excursions outside a safe box."""
    if width <= 0:
        raise ValueError("width must be positive")
    slope = _positive_float("sharpness", sharpness)
    lower_vector = _broadcast_vector("lower", lower, width)
    upper_vector = _broadcast_vector("upper", upper, width)
    if np.any(lower_vector >= upper_vector):
        raise ValueError("lower bounds must be smaller than upper bounds")

    def value(params: FloatArray) -> float:
        vector = _as_vector("params", params, width=width)
        upper_gap = _softplus(slope * (vector - upper_vector)) / slope
        lower_gap = _softplus(slope * (lower_vector - vector)) / slope
        return float(np.mean(upper_gap**2 + lower_gap**2))

    def gradient(params: FloatArray) -> FloatArray:
        vector = _as_vector("params", params, width=width)
        upper_scaled = slope * (vector - upper_vector)
        lower_scaled = slope * (lower_vector - vector)
        upper_gap = _softplus(upper_scaled) / slope
        lower_gap = _softplus(lower_scaled) / slope
        grad = 2.0 * upper_gap * _sigmoid(upper_scaled)
        grad -= 2.0 * lower_gap * _sigmoid(lower_scaled)
        return cast(FloatArray, (grad / width).astype(np.float64, copy=False))

    return ObjectiveTerm(
        name=name,
        kind="safety",
        weight=term_weight,
        value_fn=value,
        gradient_fn=gradient,
        gradient_mode="analytic",
        parameter_shift_compatible=False,
        description=("smooth analytic box-safety penalty; not parameter-shift compatible"),
    )


def build_phase_control_objective(
    width: int,
    *,
    energy_weight: float = 1.0,
    fidelity_target: ArrayLike | None = None,
    fidelity_weight: float = 0.0,
    regularization_center: ArrayLike | None = None,
    regularization_weight: float = 0.0,
    symmetry_pairs: Sequence[tuple[int, int]] | None = None,
    symmetry_weight: float = 0.0,
    safety_bounds: tuple[ArrayLike | float, ArrayLike | float] | None = None,
    safety_weight: float = 0.0,
) -> ComposedPhaseObjective:
    """Build a standard differentiable phase-control objective."""
    if width <= 0:
        raise ValueError("width must be positive")
    terms: list[ObjectiveTerm] = []
    if energy_weight > 0.0:
        terms.append(phase_energy_term(width, term_weight=energy_weight))
    if fidelity_target is not None and fidelity_weight > 0.0:
        terms.append(phase_fidelity_target_term(fidelity_target, term_weight=fidelity_weight))
    if regularization_center is not None and regularization_weight > 0.0:
        terms.append(
            periodic_regularization_term(
                regularization_center,
                term_weight=regularization_weight,
            )
        )
    if symmetry_pairs is not None and symmetry_weight > 0.0:
        terms.append(
            phase_symmetry_penalty_term(
                width,
                symmetry_pairs,
                term_weight=symmetry_weight,
            )
        )
    if safety_bounds is not None and safety_weight > 0.0:
        lower, upper = safety_bounds
        terms.append(
            smooth_box_safety_penalty_term(
                lower,
                upper,
                width=width,
                term_weight=safety_weight,
            )
        )
    return ComposedPhaseObjective(terms=tuple(terms))


def train_composed_phase_objective(
    objective: ComposedPhaseObjective,
    initial_params: ArrayLike,
    *,
    learning_rate: float = 0.2,
    max_steps: int = 100,
    gradient_tolerance: float = 1e-8,
    sufficient_decrease: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 12,
) -> ComposedObjectiveTrainingResult:
    """Minimise a composed objective with exact term-wise gradients."""
    params = _as_vector("initial_params", initial_params)
    rate = _positive_float("learning_rate", learning_rate)
    steps_limit = _positive_int("max_steps", max_steps)
    grad_tol = _non_negative_float("gradient_tolerance", gradient_tolerance)
    decrease = _positive_float("sufficient_decrease", sufficient_decrease)
    shrink = _positive_float("backtracking_factor", backtracking_factor)
    if shrink >= 1.0:
        raise ValueError("backtracking_factor must be smaller than one")
    backtrack_limit = _positive_int("max_backtracks", max_backtracks)

    current = objective.evaluate(params)
    current_value = current.value
    current_gradient = current.gradient.copy()
    initial_value = current_value
    initial_vector = params.copy()
    best_value = current_value
    best_params = params.copy()
    records: list[ComposedObjectiveTrainingStep] = []
    accepted_steps = 0
    rejected_steps = 0
    converged = False
    reason = "max_steps"

    for index in range(steps_limit):
        gradient_norm = float(np.linalg.norm(current_gradient))
        if gradient_norm <= grad_tol:
            converged = True
            reason = "gradient_tolerance"
            break
        step_size = rate
        accepted = False
        trial_value = current_value
        trial_params = params.copy()
        backtracks = 0
        descent_energy = gradient_norm * gradient_norm
        for attempt in range(backtrack_limit + 1):
            trial_params = params - step_size * current_gradient
            trial_value = objective(trial_params)
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
                ComposedObjectiveTrainingStep(
                    index=index,
                    value=current_value,
                    gradient_norm=gradient_norm,
                    step_size=step_size,
                    accepted=False,
                    backtracks=backtracks,
                    params=params.copy(),
                )
            )
            reason = "line_search_failed"
            break
        accepted_steps += 1
        params = trial_params.astype(np.float64, copy=True)
        current = objective.evaluate(params)
        current_value = current.value
        current_gradient = current.gradient.copy()
        if current_value < best_value:
            best_value = current_value
            best_params = params.copy()
        records.append(
            ComposedObjectiveTrainingStep(
                index=index,
                value=current_value,
                gradient_norm=float(np.linalg.norm(current_gradient)),
                step_size=step_size,
                accepted=True,
                backtracks=backtracks,
                params=params.copy(),
            )
        )
    else:
        if float(np.linalg.norm(current_gradient)) <= grad_tol:
            converged = True
            reason = "gradient_tolerance"

    return ComposedObjectiveTrainingResult(
        initial_value=initial_value,
        final_value=current_value,
        best_value=best_value,
        initial_params=initial_vector,
        final_params=params.copy(),
        best_params=best_params,
        final_gradient=current_gradient.copy(),
        final_gradient_norm=float(np.linalg.norm(current_gradient)),
        steps=tuple(records),
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        converged=converged,
        reason=reason,
        objective=objective,
        claim_boundary=(
            "local exact term-gradient descent for composed phase objectives; "
            "no hardware execution or global optimality guarantee is implied"
        ),
    )


def validate_composed_objective_training(
    result: ComposedObjectiveTrainingResult,
    *,
    gradient_tolerance: float | None = None,
    target_value: float | None = None,
    target_value_tolerance: float | None = None,
    min_decrease: float | None = None,
) -> ComposedObjectiveTrainingCertificate:
    """Validate composed-objective training against explicit gates."""
    grad_ok = None
    if gradient_tolerance is not None:
        grad_ok = result.final_gradient_norm <= _non_negative_float(
            "gradient_tolerance",
            gradient_tolerance,
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
    return ComposedObjectiveTrainingCertificate(
        initial_value=result.initial_value,
        final_value=result.final_value,
        best_value=result.best_value,
        value_decrease=value_decrease,
        monotone_accepted_values=monotone,
        converged=result.converged,
        within_gradient_tolerance=grad_ok,
        within_target_value_tolerance=target_ok,
        min_decrease_satisfied=decrease_ok,
        accepted_steps=result.accepted_steps,
        rejected_steps=result.rejected_steps,
        final_gradient_norm=result.final_gradient_norm,
        parameter_shift_compatible=result.objective.parameter_shift_compatible,
        claim_boundary=result.claim_boundary,
    )
