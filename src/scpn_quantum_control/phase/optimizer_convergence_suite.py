# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ground-State Optimizer Convergence Suite
"""Convergence certificates for small phase ground-state objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from ..differentiable import Parameter, ParameterShiftRule, value_and_parameter_shift_grad
from .natural_gradient import (
    MetricTensorInput,
    parameter_shift_natural_gradient_descent,
    validate_natural_gradient_training,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective: TypeAlias = Callable[[FloatArray], float]
OptimizerName = Literal["natural_gradient", "adam", "lbfgs", "spsa", "cobyla"]
BoundaryStatus = Literal["hard_gap"]

GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS = "functional_non_isolated"
GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY = (
    "local functional optimizer-convergence evidence on deterministic small "
    "phase ground-state objectives; not isolated-core timing evidence, not "
    "hardware execution, and not a global optimality claim"
)


@dataclass(frozen=True)
class KnownGroundStateObjective:
    """One deterministic small phase objective with a known ground state."""

    case_id: str
    ground_state_label: str
    initial_params: FloatArray
    target_params: FloatArray
    weights: FloatArray
    exact_ground_energy: float
    target_energy_tolerance: float
    target_parameter_tolerance: float

    def __post_init__(self) -> None:
        """Validate and freeze the ground-state objective arrays."""
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.ground_state_label.strip():
            raise ValueError("ground_state_label must be non-empty")
        initial = _as_vector("initial_params", self.initial_params)
        target = _as_vector("target_params", self.target_params)
        weights = _as_vector("weights", self.weights)
        if initial.shape != target.shape or initial.shape != weights.shape:
            raise ValueError("initial_params, target_params, and weights must share shape")
        if np.any(weights <= 0.0):
            raise ValueError("weights must be strictly positive")
        energy = float(self.exact_ground_energy)
        if not np.isfinite(energy):
            raise ValueError("exact_ground_energy must be finite")
        energy_tolerance = _non_negative_float(
            "target_energy_tolerance",
            self.target_energy_tolerance,
        )
        parameter_tolerance = _non_negative_float(
            "target_parameter_tolerance",
            self.target_parameter_tolerance,
        )
        object.__setattr__(self, "initial_params", initial)
        object.__setattr__(self, "target_params", target)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "exact_ground_energy", energy)
        object.__setattr__(self, "target_energy_tolerance", energy_tolerance)
        object.__setattr__(self, "target_parameter_tolerance", parameter_tolerance)

    @property
    def width(self) -> int:
        """Return the number of trainable phase parameters."""
        return int(self.initial_params.size)

    def value(self, params: FloatArray) -> float:
        """Return the exact phase objective value at ``params``."""
        vector = _as_vector("params", params)
        if vector.shape != self.target_params.shape:
            raise ValueError("params must match the ground-state objective shape")
        phase_error = vector - self.target_params
        return float(self.exact_ground_energy + np.sum(self.weights * (1.0 - np.cos(phase_error))))

    def metric_tensor(self, params: FloatArray) -> FloatArray:
        """Return the diagonal local metric used for natural-gradient runs."""
        vector = _as_vector("params", params)
        if vector.shape != self.target_params.shape:
            raise ValueError("params must match the ground-state objective shape")
        return np.diag(self.weights).astype(np.float64, copy=True)

    def wrapped_parameter_distance(self, params: ArrayLike) -> float:
        """Return the Euclidean distance to the target modulo ``2π``."""
        vector = _as_vector("params", params)
        if vector.shape != self.target_params.shape:
            raise ValueError("params must match the ground-state objective shape")
        wrapped = np.angle(np.exp(1j * (vector - self.target_params)))
        return float(np.linalg.norm(wrapped))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready objective metadata."""
        return {
            "case_id": self.case_id,
            "ground_state_label": self.ground_state_label,
            "initial_params": self.initial_params.tolist(),
            "target_params": self.target_params.tolist(),
            "weights": self.weights.tolist(),
            "exact_ground_energy": self.exact_ground_energy,
            "target_energy_tolerance": self.target_energy_tolerance,
            "target_parameter_tolerance": self.target_parameter_tolerance,
        }


@dataclass(frozen=True)
class GroundStateConvergenceCertificate:
    """Machine-checkable convergence certificate for one optimizer run."""

    passed: bool
    energy_error: float
    parameter_distance: float
    energy_within_tolerance: bool
    parameter_within_tolerance: bool
    value_decreased: bool
    finite_values: bool
    monotone_best_history: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready certificate evidence."""
        return {
            "passed": self.passed,
            "energy_error": self.energy_error,
            "parameter_distance": self.parameter_distance,
            "energy_within_tolerance": self.energy_within_tolerance,
            "parameter_within_tolerance": self.parameter_within_tolerance,
            "value_decreased": self.value_decreased,
            "finite_values": self.finite_values,
            "monotone_best_history": self.monotone_best_history,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GroundStateOptimizerRunRecord:
    """One optimizer run reduced to benchmark-row evidence."""

    case_id: str
    optimizer: OptimizerName
    method: str
    initial_value: float
    final_value: float
    best_value: float
    exact_ground_energy: float
    initial_params: FloatArray
    final_params: FloatArray
    best_params: FloatArray
    iterations: int
    evaluations: int
    wall_time_seconds: float
    certificate: GroundStateConvergenceCertificate
    evidence_class: str
    claim_boundary: str

    @property
    def passed(self) -> bool:
        """Return whether the convergence certificate passed."""
        return self.certificate.passed

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready optimizer row evidence."""
        return {
            "case_id": self.case_id,
            "optimizer": self.optimizer,
            "method": self.method,
            "status": "success" if self.passed else "certificate_failed",
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "best_value": self.best_value,
            "exact_ground_energy": self.exact_ground_energy,
            "initial_params": self.initial_params.tolist(),
            "final_params": self.final_params.tolist(),
            "best_params": self.best_params.tolist(),
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "wall_time_seconds": self.wall_time_seconds,
            "certificate": self.certificate.to_dict(),
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GroundStateOptimizerBoundaryRow:
    """Fail-closed optimizer boundary row for unsupported comparison routes."""

    case_id: str
    optimizer: str
    status: BoundaryStatus
    failure_class: str
    setup_instructions: str
    evidence_class: str
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate fail-closed boundary metadata."""
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.optimizer.strip():
            raise ValueError("optimizer must be non-empty")
        if self.status != "hard_gap":
            raise ValueError("boundary status must be hard_gap")
        if not self.failure_class.strip():
            raise ValueError("failure_class must be non-empty")
        if not self.setup_instructions.strip():
            raise ValueError("setup_instructions must be non-empty")
        if self.evidence_class != GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS:
            raise ValueError("boundary evidence_class must be functional_non_isolated")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready boundary evidence."""
        return {
            "case_id": self.case_id,
            "optimizer": self.optimizer,
            "status": self.status,
            "failure_class": self.failure_class,
            "setup_instructions": self.setup_instructions,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GroundStateOptimizerConvergenceSuiteResult:
    """Ground-state optimizer comparison suite with benchmark rows."""

    objectives: tuple[KnownGroundStateObjective, ...]
    records: tuple[GroundStateOptimizerRunRecord, ...]
    boundary_rows: tuple[GroundStateOptimizerBoundaryRow, ...]
    evidence_class: str
    claim_boundary: str

    @property
    def passed(self) -> bool:
        """Return whether every executable optimizer certificate passed."""
        return bool(self.records) and all(record.passed for record in self.records)

    @property
    def optimizer_names(self) -> tuple[str, ...]:
        """Return optimizer names in first-seen order."""
        names: list[str] = []
        for record in self.records:
            if record.optimizer not in names:
                names.append(record.optimizer)
        return tuple(names)

    @property
    def case_count(self) -> int:
        """Return the number of ground-state objective cases."""
        return len(self.objectives)

    @property
    def record_count(self) -> int:
        """Return the number of executable optimizer rows."""
        return len(self.records)

    def records_for_case(self, case_id: str) -> tuple[GroundStateOptimizerRunRecord, ...]:
        """Return executable records for one objective case."""
        return tuple(record for record in self.records if record.case_id == case_id)

    def best_record_for_case(self, case_id: str) -> GroundStateOptimizerRunRecord:
        """Return the lowest-energy executable record for one objective case."""
        records = self.records_for_case(case_id)
        if not records:
            raise KeyError(f"unknown ground-state optimizer case: {case_id}")
        return min(records, key=lambda record: record.best_value)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "record_count": self.record_count,
            "optimizer_names": list(self.optimizer_names),
            "objectives": [objective.to_dict() for objective in self.objectives],
            "records": [record.to_dict() for record in self.records],
            "boundary_rows": [row.to_dict() for row in self.boundary_rows],
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class _OptimizerState:
    params: FloatArray
    value_history: tuple[float, ...]
    best_params: FloatArray
    best_value: float
    iterations: int
    evaluations: int
    method: str


def default_ground_state_optimizer_objectives() -> tuple[KnownGroundStateObjective, ...]:
    """Return deterministic small ground-state objectives used by BL-15."""
    return (
        KnownGroundStateObjective(
            case_id="single_qubit_z_rotation_ground",
            ground_state_label="single-qubit Z ground state under one phase rotation",
            initial_params=np.array([0.24], dtype=np.float64),
            target_params=np.array([np.pi], dtype=np.float64),
            weights=np.array([1.0], dtype=np.float64),
            exact_ground_energy=-1.0,
            target_energy_tolerance=2.5e-5,
            target_parameter_tolerance=8.0e-3,
        ),
        KnownGroundStateObjective(
            case_id="two_qubit_product_ising_ground",
            ground_state_label="two-qubit product Ising ground state under phase rotations",
            initial_params=np.array([0.9, 2.2], dtype=np.float64),
            target_params=np.array([0.0, np.pi], dtype=np.float64),
            weights=np.array([1.0, 0.7], dtype=np.float64),
            exact_ground_energy=-1.7,
            target_energy_tolerance=8.0e-5,
            target_parameter_tolerance=1.6e-2,
        ),
    )


def run_ground_state_optimizer_convergence_suite(
    objectives: Sequence[KnownGroundStateObjective] | None = None,
    *,
    optimizers: Sequence[OptimizerName] = ("natural_gradient", "adam", "lbfgs", "spsa", "cobyla"),
    learning_rate: float = 0.45,
    max_steps: int = 96,
    spsa_perturbation: float = 0.12,
    spsa_seed: int = 17,
    include_qng_qjit_boundary: bool = True,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> GroundStateOptimizerConvergenceSuiteResult:
    """Run BL-15 optimizer convergence evidence on known small ground states."""
    objective_cases = (
        tuple(objectives)
        if objectives is not None
        else default_ground_state_optimizer_objectives()
    )
    if not objective_cases:
        raise ValueError("at least one ground-state objective is required")
    names = _optimizer_names(optimizers)
    rate = _positive_float("learning_rate", learning_rate)
    steps = _positive_int("max_steps", max_steps)
    perturbation = _positive_float("spsa_perturbation", spsa_perturbation)
    seed = _non_negative_int("spsa_seed", spsa_seed)

    records: list[GroundStateOptimizerRunRecord] = []
    for objective in objective_cases:
        for offset, name in enumerate(names):
            records.append(
                _run_optimizer(
                    objective,
                    name,
                    learning_rate=rate,
                    max_steps=steps,
                    spsa_perturbation=perturbation,
                    spsa_seed=seed + offset,
                    parameters=parameters,
                    rule=rule,
                )
            )
    boundary_rows = (_qng_qjit_boundary_row(),) if include_qng_qjit_boundary else ()
    return GroundStateOptimizerConvergenceSuiteResult(
        objectives=objective_cases,
        records=tuple(records),
        boundary_rows=boundary_rows,
        evidence_class=GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
        claim_boundary=GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
    )


def _run_optimizer(
    objective: KnownGroundStateObjective,
    optimizer: OptimizerName,
    *,
    learning_rate: float,
    max_steps: int,
    spsa_perturbation: float,
    spsa_seed: int,
    parameters: Sequence[Parameter] | None,
    rule: ParameterShiftRule | None,
) -> GroundStateOptimizerRunRecord:
    started = perf_counter()
    if optimizer == "natural_gradient":
        state = _natural_gradient_state(
            objective,
            learning_rate=learning_rate,
            max_steps=max_steps,
            parameters=parameters,
            rule=rule,
        )
    elif optimizer == "adam":
        state = _adam_state(
            objective,
            learning_rate=learning_rate,
            max_steps=max_steps,
            parameters=parameters,
            rule=rule,
        )
    elif optimizer == "lbfgs":
        state = _lbfgs_state(objective, max_steps=max_steps, parameters=parameters, rule=rule)
    elif optimizer == "spsa":
        state = _spsa_state(
            objective,
            learning_rate=learning_rate,
            max_steps=max_steps,
            perturbation=spsa_perturbation,
            seed=spsa_seed,
        )
    else:
        state = _cobyla_state(objective, max_steps=max_steps)
    final_value = objective.value(state.params)
    certificate = _certificate(objective, state, final_value=final_value)
    return GroundStateOptimizerRunRecord(
        case_id=objective.case_id,
        optimizer=optimizer,
        method=state.method,
        initial_value=objective.value(objective.initial_params),
        final_value=final_value,
        best_value=state.best_value,
        exact_ground_energy=objective.exact_ground_energy,
        initial_params=objective.initial_params.copy(),
        final_params=state.params.copy(),
        best_params=state.best_params.copy(),
        iterations=state.iterations,
        evaluations=state.evaluations,
        wall_time_seconds=max(0.0, perf_counter() - started),
        certificate=certificate,
        evidence_class=GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
        claim_boundary=GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
    )


def _natural_gradient_state(
    objective: KnownGroundStateObjective,
    *,
    learning_rate: float,
    max_steps: int,
    parameters: Sequence[Parameter] | None,
    rule: ParameterShiftRule | None,
) -> _OptimizerState:
    result = parameter_shift_natural_gradient_descent(
        objective.value,
        objective.initial_params,
        metric_tensor=cast(MetricTensorInput, objective.metric_tensor),
        parameters=parameters,
        rule=rule,
        learning_rate=learning_rate,
        max_steps=max_steps,
        gradient_tolerance=1e-10,
        natural_gradient_tolerance=1e-10,
    )
    validate_natural_gradient_training(
        result,
        target_value=objective.exact_ground_energy,
        target_value_tolerance=objective.target_energy_tolerance,
        min_decrease=0.0,
    )
    return _OptimizerState(
        params=result.best_params.copy(),
        value_history=result.value_history,
        best_params=result.best_params.copy(),
        best_value=result.best_value,
        iterations=result.accepted_steps + result.rejected_steps,
        evaluations=result.evaluations,
        method="parameter_shift_natural_gradient_descent",
    )


def _adam_state(
    objective: KnownGroundStateObjective,
    *,
    learning_rate: float,
    max_steps: int,
    parameters: Sequence[Parameter] | None,
    rule: ParameterShiftRule | None,
) -> _OptimizerState:
    params = objective.initial_params.copy()
    current_value = objective.value(params)
    best_value = current_value
    best_params = params.copy()
    values = [current_value]
    evaluations = 1
    first_moment = np.zeros_like(params)
    second_moment = np.zeros_like(params)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1.0e-8
    iterations = 0
    for index in range(1, max_steps + 1):
        gradient_result = value_and_parameter_shift_grad(
            objective.value,
            params,
            parameters=parameters,
            rule=rule,
        )
        gradient = _validated_gradient(gradient_result.gradient, params)
        evaluations += gradient_result.evaluations
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**index)
        corrected_second = second_moment / (1.0 - beta2**index)
        step = learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        candidate, candidate_value, extra_evaluations = _monotone_candidate(
            objective.value,
            params,
            step,
            current_value=current_value,
        )
        evaluations += extra_evaluations
        params = candidate
        current_value = candidate_value
        iterations = index
        values.append(current_value)
        if current_value < best_value:
            best_value = current_value
            best_params = params.copy()
        if objective.exact_ground_energy + objective.target_energy_tolerance >= best_value:
            break
    return _OptimizerState(
        params=params,
        value_history=tuple(values),
        best_params=best_params,
        best_value=best_value,
        iterations=iterations,
        evaluations=evaluations,
        method="parameter_shift_adam_with_monotone_acceptance",
    )


def _lbfgs_state(
    objective: KnownGroundStateObjective,
    *,
    max_steps: int,
    parameters: Sequence[Parameter] | None,
    rule: ParameterShiftRule | None,
) -> _OptimizerState:
    values = [objective.value(objective.initial_params)]
    evaluations = 1

    def scipy_objective(candidate: FloatArray) -> float:
        nonlocal evaluations
        value = objective.value(candidate)
        values.append(value)
        evaluations += 1
        return value

    def scipy_jacobian(candidate: FloatArray) -> FloatArray:
        nonlocal evaluations
        result = value_and_parameter_shift_grad(
            objective.value,
            np.asarray(candidate, dtype=np.float64),
            parameters=parameters,
            rule=rule,
        )
        evaluations += result.evaluations
        return _validated_gradient(result.gradient, np.asarray(candidate, dtype=np.float64))

    result = minimize(
        scipy_objective,
        objective.initial_params.copy(),
        jac=scipy_jacobian,
        method="L-BFGS-B",
        options={"maxiter": max_steps, "ftol": objective.target_energy_tolerance * 0.1},
    )
    best_index = int(np.argmin(values))
    params = np.asarray(result.x, dtype=np.float64)
    return _OptimizerState(
        params=params,
        value_history=tuple(values),
        best_params=params.copy(),
        best_value=float(min(values)),
        iterations=max(int(getattr(result, "nit", 0)), best_index),
        evaluations=evaluations,
        method="scipy_l_bfgs_b_with_parameter_shift_jacobian",
    )


def _spsa_state(
    objective: KnownGroundStateObjective,
    *,
    learning_rate: float,
    max_steps: int,
    perturbation: float,
    seed: int,
) -> _OptimizerState:
    rng = np.random.default_rng(seed)
    params = objective.initial_params.copy()
    current_value = objective.value(params)
    best_value = current_value
    best_params = params.copy()
    values = [current_value]
    evaluations = 1
    iterations = 0
    for index in range(max_steps):
        delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=params.size)
        radius = perturbation / np.sqrt(index + 1.0)
        forward = params + radius * delta
        backward = params - radius * delta
        forward_value = objective.value(forward)
        backward_value = objective.value(backward)
        evaluations += 2
        gradient = ((forward_value - backward_value) / (2.0 * radius)) * delta
        step = (learning_rate / np.sqrt(index + 1.0)) * gradient
        candidate, candidate_value, extra_evaluations = _monotone_candidate(
            objective.value,
            params,
            step,
            current_value=current_value,
        )
        evaluations += extra_evaluations
        params = candidate
        current_value = candidate_value
        iterations = index + 1
        values.append(current_value)
        if current_value < best_value:
            best_value = current_value
            best_params = params.copy()
        if objective.exact_ground_energy + objective.target_energy_tolerance >= best_value:
            break
    return _OptimizerState(
        params=params,
        value_history=tuple(values),
        best_params=best_params,
        best_value=best_value,
        iterations=iterations,
        evaluations=evaluations,
        method="seeded_spsa_with_monotone_acceptance",
    )


def _cobyla_state(objective: KnownGroundStateObjective, *, max_steps: int) -> _OptimizerState:
    values = [objective.value(objective.initial_params)]
    evaluations = 1

    def scipy_objective(candidate: FloatArray) -> float:
        nonlocal evaluations
        value = objective.value(candidate)
        values.append(value)
        evaluations += 1
        return value

    result = minimize(
        scipy_objective,
        objective.initial_params.copy(),
        method="COBYLA",
        options={"maxiter": max_steps, "rhobeg": 0.5, "tol": objective.target_parameter_tolerance},
    )
    params = np.asarray(result.x, dtype=np.float64)
    best_value = float(min(values))
    return _OptimizerState(
        params=params,
        value_history=tuple(values),
        best_params=params.copy(),
        best_value=best_value,
        iterations=int(getattr(result, "nfev", max(0, len(values) - 1))),
        evaluations=evaluations,
        method="scipy_cobyla_derivative_free",
    )


def _certificate(
    objective: KnownGroundStateObjective,
    state: _OptimizerState,
    *,
    final_value: float,
) -> GroundStateConvergenceCertificate:
    finite_values = all(np.isfinite(value) for value in state.value_history)
    energy_error = abs(state.best_value - objective.exact_ground_energy)
    parameter_distance = objective.wrapped_parameter_distance(state.best_params)
    energy_ok = energy_error <= objective.target_energy_tolerance
    parameter_ok = parameter_distance <= objective.target_parameter_tolerance
    value_decreased = state.best_value <= state.value_history[0]
    monotone_best = bool(state.value_history) and finite_values
    passed = bool(
        finite_values and energy_ok and parameter_ok and value_decreased and monotone_best
    )
    reason = "target_ground_state_reached" if passed else "certificate_failed"
    if not np.isfinite(
        final_value
    ):  # pragma: no cover - public objective validation keeps this unreachable.
        reason = "non_finite_final_value"
    return GroundStateConvergenceCertificate(
        passed=passed,
        energy_error=energy_error,
        parameter_distance=parameter_distance,
        energy_within_tolerance=energy_ok,
        parameter_within_tolerance=parameter_ok,
        value_decreased=value_decreased,
        finite_values=finite_values,
        monotone_best_history=monotone_best,
        reason=reason,
    )


def _qng_qjit_boundary_row() -> GroundStateOptimizerBoundaryRow:
    return GroundStateOptimizerBoundaryRow(
        case_id="qng_qjit_class_metric_fusion",
        optimizer="qng_qjit_class_boundary",
        status="hard_gap",
        failure_class="unsupported_qjit_metric_fusion",
        setup_instructions=(
            "The local BL-15 suite exposes Python parameter-shift natural-gradient "
            "evidence only. A QNG-QJIT-class route needs a compiler-owned metric "
            "fusion and executable lowering contract before it can be compared."
        ),
        evidence_class=GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS,
        claim_boundary=GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY,
    )


def _monotone_candidate(
    objective: ScalarObjective,
    params: FloatArray,
    step: FloatArray,
    *,
    current_value: float,
) -> tuple[FloatArray, float, int]:
    scale = 1.0
    evaluations = 0
    candidate = params.copy()
    candidate_value = current_value
    for _ in range(10):
        candidate = params - scale * step
        candidate_value = objective(candidate)
        evaluations += 1
        if np.isfinite(candidate_value) and candidate_value <= current_value:
            return candidate.astype(np.float64, copy=True), float(candidate_value), evaluations
        scale *= 0.5
    return params.copy(), current_value, evaluations


def _validated_gradient(gradient: ArrayLike, params: FloatArray) -> FloatArray:
    vector = np.asarray(gradient, dtype=float)
    if vector.shape != params.shape:
        raise ValueError("gradient shape must match parameter shape")
    if not np.all(np.isfinite(vector)):
        raise ValueError("gradient must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


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


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optimizer_names(names: Sequence[OptimizerName]) -> tuple[OptimizerName, ...]:
    if not names:
        raise ValueError("at least one optimizer is required")
    allowed = {"natural_gradient", "adam", "lbfgs", "spsa", "cobyla"}
    result: list[OptimizerName] = []
    for name in names:
        if name not in allowed:
            raise ValueError(f"unknown optimizer: {name}")
        if name not in result:
            result.append(name)
    return tuple(result)


__all__ = [
    "GROUND_STATE_OPTIMIZER_CLAIM_BOUNDARY",
    "GROUND_STATE_OPTIMIZER_EVIDENCE_CLASS",
    "GroundStateConvergenceCertificate",
    "GroundStateOptimizerBoundaryRow",
    "GroundStateOptimizerConvergenceSuiteResult",
    "GroundStateOptimizerRunRecord",
    "KnownGroundStateObjective",
    "default_ground_state_optimizer_objectives",
    "run_ground_state_optimizer_convergence_suite",
]
