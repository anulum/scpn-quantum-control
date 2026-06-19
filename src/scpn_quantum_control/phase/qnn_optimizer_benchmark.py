# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNN Optimizer Benchmark
"""Functional optimizer benchmarks for bounded phase-QNN training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .natural_gradient import solve_natural_gradient_direction
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    predict_parameter_shift_qnn_classifier,
    train_parameter_shift_qnn_classifier,
)

FloatArray: TypeAlias = NDArray[np.float64]
DerivativeFreeCandidateMap: TypeAlias = Mapping[str, Sequence[ArrayLike]]
OptimizerTrainingFactory: TypeAlias = Callable[[], "_OptimizerTrainingResult"]

EVIDENCE_CLASS = "functional_non_isolated"
CLAIM_BOUNDARY = (
    "local deterministic optimizer comparison; not a throughput benchmark, "
    "not isolated-core evidence, and not a hardware performance claim"
)


@dataclass(frozen=True)
class _QNNOptimizerBenchmarkCase:
    name: str
    features: FloatArray
    labels: FloatArray
    initial_params: FloatArray
    derivative_free_candidates: tuple[FloatArray, ...]


@dataclass(frozen=True)
class _FiniteDifferenceTrainingResult:
    best_params: FloatArray
    best_loss: float
    evaluations: int
    steps: int
    converged: bool
    method: str


@dataclass(frozen=True)
class _DerivativeFreeTrainingResult:
    best_params: FloatArray
    best_loss: float
    evaluations: int
    steps: int
    converged: bool
    method: str


@dataclass(frozen=True)
class _OptimizerTrainingResult:
    best_params: FloatArray
    best_loss: float
    evaluations: int
    steps: int
    converged: bool
    method: str


@dataclass(frozen=True)
class QNNOptimizerBaselineResult:
    """Serializable bounded-QNN optimizer baseline evidence."""

    name: str
    best_loss: float
    accuracy: float | None
    evaluations: int
    steps: int
    converged: bool
    method: str
    wall_time_seconds: float
    evidence_class: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready optimizer baseline evidence."""
        return {
            "name": self.name,
            "best_loss": self.best_loss,
            "accuracy": self.accuracy,
            "evaluations": self.evaluations,
            "steps": self.steps,
            "converged": self.converged,
            "method": self.method,
            "wall_time_seconds": self.wall_time_seconds,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ParameterShiftQNNOptimizerBenchmarkCaseResult:
    """One bounded QNN optimizer-comparison case."""

    name: str
    n_samples: int
    n_features: int
    parameter_shift_best_loss: float
    finite_difference_best_loss: float
    derivative_free_best_loss: float
    parameter_shift_accuracy: float | None
    finite_difference_accuracy: float | None
    derivative_free_accuracy: float | None
    parameter_shift_evaluations: int
    finite_difference_evaluations: int
    derivative_free_evaluations: int
    optimizer_results: tuple[QNNOptimizerBaselineResult, ...]
    tolerance: float
    evidence_class: str
    claim_boundary: str

    @property
    def optimizer_names(self) -> tuple[str, ...]:
        """Return optimizer names in suite order."""
        return tuple(result.name for result in self.optimizer_results)

    def optimizer_by_name(self, name: str) -> QNNOptimizerBaselineResult:
        """Return one optimizer baseline result by name."""
        for result in self.optimizer_results:
            if result.name == name:
                return result
        raise KeyError(f"unknown QNN optimizer baseline: {name}")

    @property
    def parameter_shift_not_worse_than_finite_difference(self) -> bool:
        """Return whether parameter shift is no worse than finite differences."""
        return self.parameter_shift_best_loss <= self.finite_difference_best_loss + self.tolerance

    @property
    def derivative_free_wins(self) -> bool:
        """Return whether the derivative-free baseline beats parameter shift."""
        return self.derivative_free_best_loss + self.tolerance < self.parameter_shift_best_loss

    @property
    def passed(self) -> bool:
        """Return whether this optimizer benchmark case passed."""
        return bool(
            self.parameter_shift_not_worse_than_finite_difference
            and not self.derivative_free_wins
            and all(result.evaluations > 0 for result in self.optimizer_results)
            and all(np.isfinite(result.best_loss) for result in self.optimizer_results)
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark evidence."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "parameter_shift_best_loss": self.parameter_shift_best_loss,
            "finite_difference_best_loss": self.finite_difference_best_loss,
            "derivative_free_best_loss": self.derivative_free_best_loss,
            "parameter_shift_accuracy": self.parameter_shift_accuracy,
            "finite_difference_accuracy": self.finite_difference_accuracy,
            "derivative_free_accuracy": self.derivative_free_accuracy,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "finite_difference_evaluations": self.finite_difference_evaluations,
            "derivative_free_evaluations": self.derivative_free_evaluations,
            "optimizer_names": list(self.optimizer_names),
            "optimizer_results": [result.to_dict() for result in self.optimizer_results],
            "tolerance": self.tolerance,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "parameter_shift_not_worse_than_finite_difference": (
                self.parameter_shift_not_worse_than_finite_difference
            ),
            "derivative_free_wins": self.derivative_free_wins,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ParameterShiftQNNOptimizerBenchmarkSuiteResult:
    """Bounded QNN optimizer-comparison suite."""

    cases: tuple[ParameterShiftQNNOptimizerBenchmarkCaseResult, ...]
    evidence_class: str
    claim_boundary: str
    production_benchmark: bool

    @property
    def passed(self) -> bool:
        """Return whether every benchmark case passed."""
        return all(case.passed for case in self.cases)

    @property
    def case_count(self) -> int:
        """Return the number of benchmark cases."""
        return len(self.cases)

    @property
    def parameter_shift_not_worse_count(self) -> int:
        """Return cases where parameter shift is no worse than finite differences."""
        return sum(
            1 for case in self.cases if case.parameter_shift_not_worse_than_finite_difference
        )

    @property
    def derivative_free_win_count(self) -> int:
        """Return cases where the derivative-free baseline wins."""
        return sum(1 for case in self.cases if case.derivative_free_wins)

    @property
    def optimizer_names(self) -> tuple[str, ...]:
        """Return optimizer names in first-case order."""
        if not self.cases:
            return ()
        return self.cases[0].optimizer_names

    @property
    def optimizer_count(self) -> int:
        """Return the number of optimizer baselines per case."""
        return len(self.optimizer_names)

    def case_by_name(self, name: str) -> ParameterShiftQNNOptimizerBenchmarkCaseResult:
        """Return a benchmark case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown QNN optimizer benchmark case: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "parameter_shift_not_worse_count": self.parameter_shift_not_worse_count,
            "derivative_free_win_count": self.derivative_free_win_count,
            "optimizer_names": list(self.optimizer_names),
            "optimizer_count": self.optimizer_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "production_benchmark": self.production_benchmark,
            "cases": [case.to_dict() for case in self.cases],
        }


def _default_cases() -> tuple[_QNNOptimizerBenchmarkCase, ...]:
    single_initial = np.array([0.8], dtype=np.float64)
    mixed_initial = np.array([0.4, -0.2], dtype=np.float64)
    return (
        _QNNOptimizerBenchmarkCase(
            name="phase_separable_single_feature",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            initial_params=single_initial,
            derivative_free_candidates=(
                single_initial.copy(),
                single_initial + np.array([0.1], dtype=np.float64),
                single_initial - np.array([0.1], dtype=np.float64),
            ),
        ),
        _QNNOptimizerBenchmarkCase(
            name="two_feature_mixed_phase",
            features=np.array(
                [[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]],
                dtype=np.float64,
            ),
            labels=np.array([0.0, 1.0, 0.25], dtype=np.float64),
            initial_params=mixed_initial,
            derivative_free_candidates=(
                mixed_initial.copy(),
                mixed_initial + np.array([0.05, -0.05], dtype=np.float64),
                mixed_initial - np.array([0.05, -0.05], dtype=np.float64),
            ),
        ),
    )


def _as_positive_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


def _as_non_negative_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _as_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _as_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _case_map() -> dict[str, _QNNOptimizerBenchmarkCase]:
    return {case.name: case for case in _default_cases()}


def _selected_cases(case_names: Sequence[str] | None) -> tuple[_QNNOptimizerBenchmarkCase, ...]:
    cases = _case_map()
    if case_names is None:
        return tuple(cases.values())
    selected: list[_QNNOptimizerBenchmarkCase] = []
    for name in case_names:
        if name not in cases:
            raise ValueError(f"unknown QNN optimizer benchmark case: {name}")
        selected.append(cases[name])
    return tuple(selected)


def _loss(case: _QNNOptimizerBenchmarkCase, params: FloatArray) -> float:
    return parameter_shift_qnn_classifier_loss(case.features, case.labels, params)


def _accuracy(case: _QNNOptimizerBenchmarkCase, params: FloatArray) -> float | None:
    prediction = predict_parameter_shift_qnn_classifier(
        case.features,
        params,
        labels=case.labels,
    )
    return prediction.accuracy


def _parameter_shift_gradient(case: _QNNOptimizerBenchmarkCase, params: FloatArray) -> FloatArray:
    return parameter_shift_qnn_classifier_gradient(case.features, case.labels, params)


def _loss_decreased(
    initial_loss: float,
    best_loss: float,
    *,
    tolerance: float,
) -> bool:
    return bool(best_loss <= initial_loss - tolerance or best_loss <= tolerance)


def _parameter_shift_evaluation_count(steps: int, width: int) -> int:
    return max(1, steps) * 4 * width


def _record_wall_time(
    *,
    name: str,
    case: _QNNOptimizerBenchmarkCase,
    result: _OptimizerTrainingResult
    | _FiniteDifferenceTrainingResult
    | _DerivativeFreeTrainingResult,
    started: float,
) -> QNNOptimizerBaselineResult:
    return QNNOptimizerBaselineResult(
        name=name,
        best_loss=result.best_loss,
        accuracy=_accuracy(case, result.best_params),
        evaluations=result.evaluations,
        steps=result.steps,
        converged=result.converged,
        method=result.method,
        wall_time_seconds=max(0.0, perf_counter() - started),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
    )


def _sgd_training(
    case: _QNNOptimizerBenchmarkCase,
    *,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
) -> _OptimizerTrainingResult:
    params = case.initial_params.copy()
    best_params = params.copy()
    initial_loss = _loss(case, params)
    best_loss = initial_loss
    evaluations = 1
    steps = 0
    for _ in range(max_steps):
        gradient = _parameter_shift_gradient(case, params)
        evaluations += 4 * params.size
        candidate = params - learning_rate * gradient
        candidate_loss = _loss(case, candidate)
        evaluations += 1
        if not np.isfinite(candidate_loss):
            break
        params = candidate
        steps += 1
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_params = candidate.copy()
    return _OptimizerTrainingResult(
        best_params=best_params,
        best_loss=best_loss,
        evaluations=evaluations,
        steps=steps,
        converged=_loss_decreased(initial_loss, best_loss, tolerance=tolerance),
        method="full_batch_parameter_shift_sgd",
    )


def _adam_training(
    case: _QNNOptimizerBenchmarkCase,
    *,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
) -> _OptimizerTrainingResult:
    params = case.initial_params.copy()
    best_params = params.copy()
    initial_loss = _loss(case, params)
    best_loss = initial_loss
    first_moment = np.zeros_like(params, dtype=np.float64)
    second_moment = np.zeros_like(params, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    evaluations = 1
    steps = 0
    for index in range(1, max_steps + 1):
        gradient = _parameter_shift_gradient(case, params)
        evaluations += 4 * params.size
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**index)
        corrected_second = second_moment / (1.0 - beta2**index)
        candidate = params - learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        candidate_loss = _loss(case, candidate)
        evaluations += 1
        if not np.isfinite(candidate_loss):
            break
        params = candidate
        steps += 1
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_params = candidate.copy()
    return _OptimizerTrainingResult(
        best_params=best_params,
        best_loss=best_loss,
        evaluations=evaluations,
        steps=steps,
        converged=_loss_decreased(initial_loss, best_loss, tolerance=tolerance),
        method="full_batch_parameter_shift_adam",
    )


def _lbfgs_training(
    case: _QNNOptimizerBenchmarkCase,
    *,
    max_steps: int,
    tolerance: float,
) -> _OptimizerTrainingResult:
    initial_loss = _loss(case, case.initial_params)
    loss_evaluations = 1
    gradient_evaluations = 0

    def objective(candidate: FloatArray) -> float:
        nonlocal loss_evaluations
        loss_evaluations += 1
        return _loss(case, candidate)

    def jacobian(candidate: FloatArray) -> FloatArray:
        nonlocal gradient_evaluations
        gradient_evaluations += 4 * candidate.size
        return _parameter_shift_gradient(case, candidate)

    result = minimize(
        objective,
        case.initial_params.copy(),
        jac=jacobian,
        method="L-BFGS-B",
        options={"maxiter": max_steps, "ftol": tolerance},
    )
    best_params = np.asarray(result.x, dtype=np.float64)
    best_loss = float(result.fun)
    steps = int(getattr(result, "nit", 0))
    return _OptimizerTrainingResult(
        best_params=best_params,
        best_loss=best_loss,
        evaluations=loss_evaluations + gradient_evaluations,
        steps=steps,
        converged=bool(result.success)
        or _loss_decreased(initial_loss, best_loss, tolerance=tolerance),
        method="scipy_l_bfgs_b_with_parameter_shift_jacobian",
    )


def _qnn_diagonal_fisher_metric(
    case: _QNNOptimizerBenchmarkCase,
    params: FloatArray,
    *,
    damping: float,
) -> FloatArray:
    probabilities = 0.5 * (1.0 - np.cos(case.features + params[None, :]))
    derivatives = 0.5 * np.sin(case.features + params[None, :])
    variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
    diagonal = np.mean((derivatives * derivatives) / variance, axis=0)
    return np.diag(diagonal + damping).astype(np.float64, copy=False)


def _natural_gradient_training(
    case: _QNNOptimizerBenchmarkCase,
    *,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
) -> _OptimizerTrainingResult:
    params = case.initial_params.copy()
    best_params = params.copy()
    initial_loss = _loss(case, params)
    best_loss = initial_loss
    evaluations = 1
    steps = 0
    damping = 1e-6
    for _ in range(max_steps):
        gradient = _parameter_shift_gradient(case, params)
        evaluations += 4 * params.size
        metric = _qnn_diagonal_fisher_metric(case, params, damping=damping)
        direction = solve_natural_gradient_direction(gradient, metric, damping=damping)
        candidate = params - learning_rate * direction.direction
        candidate_loss = _loss(case, candidate)
        evaluations += 1
        if not np.isfinite(candidate_loss):
            break
        params = candidate
        steps += 1
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_params = candidate.copy()
    return _OptimizerTrainingResult(
        best_params=best_params,
        best_loss=best_loss,
        evaluations=evaluations,
        steps=steps,
        converged=_loss_decreased(initial_loss, best_loss, tolerance=tolerance),
        method="parameter_shift_diagonal_fisher_natural_gradient",
    )


def _spsa_training(
    case: _QNNOptimizerBenchmarkCase,
    *,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
    perturbation: float,
    seed: int,
) -> _OptimizerTrainingResult:
    rng = np.random.default_rng(seed)
    params = case.initial_params.copy()
    best_params = params.copy()
    initial_loss = _loss(case, params)
    best_loss = initial_loss
    evaluations = 1
    steps = 0
    for index in range(max_steps):
        delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=params.size)
        radius = perturbation / np.sqrt(index + 1.0)
        forward = params + radius * delta
        backward = params - radius * delta
        loss_forward = _loss(case, forward)
        loss_backward = _loss(case, backward)
        evaluations += 2
        gradient = ((loss_forward - loss_backward) / (2.0 * radius)) * delta
        step_rate = learning_rate / np.sqrt(index + 1.0)
        candidate = params - step_rate * gradient
        candidate_loss = _loss(case, candidate)
        evaluations += 1
        if not np.isfinite(candidate_loss):
            break
        params = candidate
        steps += 1
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_params = candidate.copy()
    return _OptimizerTrainingResult(
        best_params=best_params,
        best_loss=best_loss,
        evaluations=evaluations,
        steps=steps,
        converged=_loss_decreased(initial_loss, best_loss, tolerance=tolerance),
        method="seeded_spsa_finite_difference_estimator",
    )


def _finite_difference_gradient(
    case: _QNNOptimizerBenchmarkCase,
    params: FloatArray,
    *,
    step: float,
) -> tuple[FloatArray, int]:
    gradient = np.zeros_like(params, dtype=np.float64)
    evaluations = 0
    for index in range(params.size):
        forward = params.copy()
        backward = params.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (_loss(case, forward) - _loss(case, backward)) / (2.0 * step)
        evaluations += 2
    return gradient, evaluations


def _finite_difference_training(
    case: _QNNOptimizerBenchmarkCase,
    *,
    learning_rate: float,
    max_steps: int,
    finite_difference_step: float,
) -> _FiniteDifferenceTrainingResult:
    params = case.initial_params.copy()
    best_params = params.copy()
    best_loss = _loss(case, params)
    evaluations = 1
    steps = 0
    for _ in range(max_steps):
        gradient, gradient_evaluations = _finite_difference_gradient(
            case,
            params,
            step=finite_difference_step,
        )
        evaluations += gradient_evaluations
        candidate = params - learning_rate * gradient
        candidate_loss = _loss(case, candidate)
        evaluations += 1
        if not np.isfinite(candidate_loss):
            break
        params = candidate
        steps += 1
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_params = candidate.copy()
    return _FiniteDifferenceTrainingResult(
        best_params=best_params,
        best_loss=best_loss,
        evaluations=evaluations,
        steps=steps,
        converged=_loss_decreased(_loss(case, case.initial_params), best_loss, tolerance=0.0),
        method="central_finite_difference_gradient_descent",
    )


def _candidate_vector(raw: ArrayLike, *, width: int, case_name: str) -> FloatArray:
    vector = np.asarray(raw, dtype=float)
    if vector.ndim != 1 or vector.shape != (width,):
        raise ValueError(f"derivative-free candidate for {case_name!r} must have shape ({width},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"derivative-free candidate for {case_name!r} must contain only finite values"
        )
    return vector.astype(np.float64, copy=True)


def _derivative_free_training(
    case: _QNNOptimizerBenchmarkCase,
    derivative_free_candidates: DerivativeFreeCandidateMap | None,
) -> _DerivativeFreeTrainingResult:
    raw_candidates = (
        case.derivative_free_candidates
        if derivative_free_candidates is None or case.name not in derivative_free_candidates
        else derivative_free_candidates[case.name]
    )
    candidates = tuple(
        _candidate_vector(candidate, width=case.initial_params.size, case_name=case.name)
        for candidate in raw_candidates
    )
    if not candidates:
        raise ValueError(f"derivative-free candidates for {case.name!r} must be non-empty")
    losses = tuple(_loss(case, candidate) for candidate in candidates)
    best_index = int(np.argmin(losses))
    return _DerivativeFreeTrainingResult(
        best_params=candidates[best_index].copy(),
        best_loss=float(losses[best_index]),
        evaluations=len(candidates),
        steps=len(candidates),
        converged=bool(float(losses[best_index]) <= _loss(case, case.initial_params)),
        method="deterministic_declared_candidate_scan",
    )


def _baseline_results(
    case: _QNNOptimizerBenchmarkCase,
    *,
    parameter_shift_best_loss: float,
    parameter_shift_accuracy: float | None,
    parameter_shift_evaluations: int,
    parameter_shift_steps: int,
    parameter_shift_converged: bool,
    parameter_shift_wall_time_seconds: float,
    finite_difference: _FiniteDifferenceTrainingResult,
    derivative_free: _DerivativeFreeTrainingResult,
    learning_rate: float,
    max_steps: int,
    tolerance: float,
    spsa_perturbation: float,
    spsa_seed: int,
) -> tuple[QNNOptimizerBaselineResult, ...]:
    results: list[QNNOptimizerBaselineResult] = [
        QNNOptimizerBaselineResult(
            name="parameter_shift",
            best_loss=parameter_shift_best_loss,
            accuracy=parameter_shift_accuracy,
            evaluations=parameter_shift_evaluations,
            steps=parameter_shift_steps,
            converged=parameter_shift_converged,
            method="production_parameter_shift_gradient_descent",
            wall_time_seconds=parameter_shift_wall_time_seconds,
            evidence_class=EVIDENCE_CLASS,
            claim_boundary=CLAIM_BOUNDARY,
        )
    ]
    started = perf_counter()
    results.append(
        _record_wall_time(
            name="finite_difference",
            case=case,
            result=finite_difference,
            started=started,
        )
    )
    optimizer_training_factories: tuple[tuple[str, OptimizerTrainingFactory], ...] = (
        (
            "sgd",
            lambda: _sgd_training(
                case,
                learning_rate=learning_rate,
                max_steps=max_steps,
                tolerance=tolerance,
            ),
        ),
        (
            "adam",
            lambda: _adam_training(
                case,
                learning_rate=learning_rate,
                max_steps=max_steps,
                tolerance=tolerance,
            ),
        ),
        ("lbfgs", lambda: _lbfgs_training(case, max_steps=max_steps, tolerance=tolerance)),
        (
            "natural_gradient",
            lambda: _natural_gradient_training(
                case,
                learning_rate=learning_rate,
                max_steps=max_steps,
                tolerance=tolerance,
            ),
        ),
        (
            "spsa",
            lambda: _spsa_training(
                case,
                learning_rate=learning_rate,
                max_steps=max_steps,
                tolerance=tolerance,
                perturbation=spsa_perturbation,
                seed=spsa_seed,
            ),
        ),
    )
    for name, training in optimizer_training_factories:
        started = perf_counter()
        results.append(
            _record_wall_time(
                name=name,
                case=case,
                result=training(),
                started=started,
            )
        )
    started = perf_counter()
    results.append(
        _record_wall_time(
            name="derivative_free_grid",
            case=case,
            result=derivative_free,
            started=started,
        )
    )
    return tuple(results)


def run_parameter_shift_qnn_optimizer_benchmark_suite(
    *,
    case_names: Sequence[str] | None = None,
    learning_rate: float = 0.7,
    max_steps: int = 80,
    finite_difference_step: float = 1e-6,
    tolerance: float = 1e-6,
    spsa_perturbation: float = 0.08,
    spsa_seed: int = 7,
    derivative_free_candidates: DerivativeFreeCandidateMap | None = None,
) -> ParameterShiftQNNOptimizerBenchmarkSuiteResult:
    """Run local functional optimizer comparisons for bounded QNN training.

    The suite compares the production parameter-shift trainer with a central
    finite-difference gradient baseline and a deterministic derivative-free
    candidate scan. Results are explicitly marked as functional, non-isolated
    evidence; they are not suitable for throughput, hardware, or isolated-core
    performance claims.
    """
    lr = _as_positive_float("learning_rate", learning_rate)
    steps = _as_positive_int("max_steps", max_steps)
    fd_step = _as_positive_float("finite_difference_step", finite_difference_step)
    tolerance_value = _as_non_negative_float("tolerance", tolerance)
    spsa_radius = _as_positive_float("spsa_perturbation", spsa_perturbation)
    seed = _as_non_negative_int("spsa_seed", spsa_seed)

    results: list[ParameterShiftQNNOptimizerBenchmarkCaseResult] = []
    for case in _selected_cases(case_names):
        parameter_shift_started = perf_counter()
        parameter_shift_training = train_parameter_shift_qnn_classifier(
            case.features,
            case.labels,
            initial_params=case.initial_params,
            learning_rate=lr,
            max_steps=steps,
            target_loss=0.0,
        )
        parameter_shift_wall_time = max(0.0, perf_counter() - parameter_shift_started)
        finite_difference = _finite_difference_training(
            case,
            learning_rate=lr,
            max_steps=steps,
            finite_difference_step=fd_step,
        )
        derivative_free = _derivative_free_training(case, derivative_free_candidates)
        gradient_width = parameter_shift_qnn_classifier_gradient(
            case.features,
            case.labels,
            case.initial_params,
        ).size
        parameter_shift_evaluations = _parameter_shift_evaluation_count(
            len(parameter_shift_training.loss_history),
            gradient_width,
        )
        optimizer_results = _baseline_results(
            case,
            parameter_shift_best_loss=parameter_shift_training.best_loss,
            parameter_shift_accuracy=parameter_shift_training.prediction.accuracy,
            parameter_shift_evaluations=parameter_shift_evaluations,
            parameter_shift_steps=max(0, len(parameter_shift_training.loss_history) - 1),
            parameter_shift_converged=parameter_shift_training.training.converged,
            parameter_shift_wall_time_seconds=parameter_shift_wall_time,
            finite_difference=finite_difference,
            derivative_free=derivative_free,
            learning_rate=lr,
            max_steps=steps,
            tolerance=tolerance_value,
            spsa_perturbation=spsa_radius,
            spsa_seed=seed,
        )
        results.append(
            ParameterShiftQNNOptimizerBenchmarkCaseResult(
                name=case.name,
                n_samples=int(case.features.shape[0]),
                n_features=int(case.features.shape[1]),
                parameter_shift_best_loss=parameter_shift_training.best_loss,
                finite_difference_best_loss=finite_difference.best_loss,
                derivative_free_best_loss=derivative_free.best_loss,
                parameter_shift_accuracy=parameter_shift_training.prediction.accuracy,
                finite_difference_accuracy=_accuracy(case, finite_difference.best_params),
                derivative_free_accuracy=_accuracy(case, derivative_free.best_params),
                parameter_shift_evaluations=parameter_shift_evaluations,
                finite_difference_evaluations=finite_difference.evaluations,
                derivative_free_evaluations=derivative_free.evaluations,
                optimizer_results=optimizer_results,
                tolerance=tolerance_value,
                evidence_class=EVIDENCE_CLASS,
                claim_boundary=CLAIM_BOUNDARY,
            )
        )

    return ParameterShiftQNNOptimizerBenchmarkSuiteResult(
        cases=tuple(results),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        production_benchmark=False,
    )


__all__ = [
    "DerivativeFreeCandidateMap",
    "ParameterShiftQNNOptimizerBenchmarkCaseResult",
    "ParameterShiftQNNOptimizerBenchmarkSuiteResult",
    "QNNOptimizerBaselineResult",
    "run_parameter_shift_qnn_optimizer_benchmark_suite",
]
