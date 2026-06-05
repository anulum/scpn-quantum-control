# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNN Finite-Shot Evidence
"""Seeded finite-shot evidence for bounded phase-QNN gradients and training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .param_shift import multi_frequency_parameter_shift_rule
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    predict_parameter_shift_qnn_classifier,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

GRADIENT_EVIDENCE_CLASS = "seeded_finite_shot_qnn_gradient"
CONVERGENCE_EVIDENCE_CLASS = "seeded_finite_shot_qnn_convergence"
CLAIM_BOUNDARY = (
    "seeded local finite-shot simulator evidence for bounded phase-QNN gradients; "
    "not hardware execution, not provider-job evidence, and not arbitrary "
    "QNN/QGNN/QSNN architecture evidence"
)


@dataclass(frozen=True)
class _FiniteShotConvergenceCase:
    name: str
    features: FloatArray
    labels: FloatArray
    initial_params: FloatArray
    shots_per_sample: int
    seed: int
    learning_rate: float
    max_steps: int
    target_loss_tolerance: float
    min_loss_drop: float


@dataclass(frozen=True)
class ParameterShiftQNNFiniteShotProbeRecord:
    """One plus/minus finite-shot loss-probe pair for a QNN parameter."""

    parameter_index: int
    term_index: int
    shift: float
    coefficient: float
    plus_loss: float
    minus_loss: float
    plus_loss_variance: float
    minus_loss_variance: float
    plus_seed: int
    minus_seed: int
    shots_per_sample: int
    n_samples: int

    @property
    def total_shots(self) -> int:
        """Return total Bernoulli shots consumed by this plus/minus pair."""
        return 2 * self.n_samples * self.shots_per_sample

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready probe evidence."""
        return {
            "parameter_index": self.parameter_index,
            "term_index": self.term_index,
            "shift": self.shift,
            "coefficient": self.coefficient,
            "plus_loss": self.plus_loss,
            "minus_loss": self.minus_loss,
            "plus_loss_variance": self.plus_loss_variance,
            "minus_loss_variance": self.minus_loss_variance,
            "plus_seed": self.plus_seed,
            "minus_seed": self.minus_seed,
            "shots_per_sample": self.shots_per_sample,
            "n_samples": self.n_samples,
            "total_shots": self.total_shots,
        }


@dataclass(frozen=True)
class ParameterShiftQNNFiniteShotGradientResult:
    """Seeded finite-shot QNN gradient with uncertainty evidence."""

    loss: float
    deterministic_gradient: FloatArray
    finite_shot_gradient: FloatArray
    standard_error: FloatArray
    confidence_radius: FloatArray
    max_abs_error: float
    max_standard_error: float
    max_confidence_radius: float
    confidence_z: float
    shots_per_sample: int
    seed: int
    records: tuple[ParameterShiftQNNFiniteShotProbeRecord, ...]
    evidence_class: str
    claim_boundary: str
    hardware_execution: bool

    @property
    def passed(self) -> bool:
        """Return whether finite-shot errors fit inside confidence radii."""
        return self.max_abs_error <= self.max_confidence_radius

    @property
    def probe_count(self) -> int:
        """Return the number of shifted plus/minus loss probes."""
        return 2 * len(self.records)

    @property
    def parameter_shift_evaluations(self) -> int:
        """Return parameter-shift loss evaluations used by the estimator."""
        return self.probe_count

    @property
    def total_shots(self) -> int:
        """Return total Bernoulli shots consumed by the estimator."""
        return sum(record.total_shots for record in self.records)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready finite-shot gradient evidence."""
        return {
            "loss": self.loss,
            "deterministic_gradient": self.deterministic_gradient.tolist(),
            "finite_shot_gradient": self.finite_shot_gradient.tolist(),
            "standard_error": self.standard_error.tolist(),
            "confidence_radius": self.confidence_radius.tolist(),
            "max_abs_error": self.max_abs_error,
            "max_standard_error": self.max_standard_error,
            "max_confidence_radius": self.max_confidence_radius,
            "confidence_z": self.confidence_z,
            "shots_per_sample": self.shots_per_sample,
            "seed": self.seed,
            "probe_count": self.probe_count,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "total_shots": self.total_shots,
            "passed": self.passed,
            "records": [record.to_dict() for record in self.records],
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class ParameterShiftQNNFiniteShotConvergenceCaseResult:
    """Seeded noisy-gradient convergence evidence for one bounded QNN case."""

    name: str
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_drop: float
    target_loss_tolerance: float
    min_loss_drop: float
    learning_rate: float
    max_steps: int
    accepted_steps: int
    shots_per_sample: int
    seed: int
    total_shots: int
    total_parameter_shift_evaluations: int
    max_gradient_standard_error: float
    evidence_class: str
    claim_boundary: str
    hardware_execution: bool
    production_benchmark: bool

    @property
    def converged(self) -> bool:
        """Return whether the best deterministic replay loss met tolerance."""
        return self.best_loss <= self.target_loss_tolerance

    @property
    def loss_drop_passed(self) -> bool:
        """Return whether noisy-gradient training reduced loss enough."""
        return self.loss_drop >= self.min_loss_drop

    @property
    def passed(self) -> bool:
        """Return whether this finite-shot convergence case passed."""
        return bool(self.converged and self.loss_drop_passed)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready convergence-case evidence."""
        return {
            "name": self.name,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_drop": self.loss_drop,
            "target_loss_tolerance": self.target_loss_tolerance,
            "min_loss_drop": self.min_loss_drop,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "accepted_steps": self.accepted_steps,
            "shots_per_sample": self.shots_per_sample,
            "seed": self.seed,
            "total_shots": self.total_shots,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "max_gradient_standard_error": self.max_gradient_standard_error,
            "converged": self.converged,
            "loss_drop_passed": self.loss_drop_passed,
            "passed": self.passed,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
            "production_benchmark": self.production_benchmark,
        }


@dataclass(frozen=True)
class ParameterShiftQNNFiniteShotUnsupportedScenario:
    """Unsupported finite-shot QNN scenario and mitigation."""

    name: str
    status: str
    reason: str
    mitigation: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready unsupported-scenario evidence."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "mitigation": self.mitigation,
        }


@dataclass(frozen=True)
class ParameterShiftQNNFiniteShotConvergenceSuiteResult:
    """Bundled seeded finite-shot convergence evidence."""

    cases: tuple[ParameterShiftQNNFiniteShotConvergenceCaseResult, ...]
    unsuitable_scenarios: tuple[ParameterShiftQNNFiniteShotUnsupportedScenario, ...]
    evidence_class: str
    claim_boundary: str
    hardware_execution: bool
    production_benchmark: bool

    @property
    def passed(self) -> bool:
        """Return whether every finite-shot convergence case passed."""
        return all(case.passed for case in self.cases)

    @property
    def case_count(self) -> int:
        """Return the number of convergence cases."""
        return len(self.cases)

    @property
    def passed_count(self) -> int:
        """Return the number of passing cases."""
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed_count(self) -> int:
        """Return the number of failing cases."""
        return self.case_count - self.passed_count

    @property
    def total_shots(self) -> int:
        """Return total Bernoulli shots consumed across cases."""
        return sum(case.total_shots for case in self.cases)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Return total shifted loss evaluations across cases."""
        return sum(case.total_parameter_shift_evaluations for case in self.cases)

    def case_by_name(self, name: str) -> ParameterShiftQNNFiniteShotConvergenceCaseResult:
        """Return a finite-shot convergence case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown QNN finite-shot convergence case: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready finite-shot suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_shots": self.total_shots,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
            "production_benchmark": self.production_benchmark,
            "cases": [case.to_dict() for case in self.cases],
            "unsuitable_scenarios": [scenario.to_dict() for scenario in self.unsuitable_scenarios],
        }


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must contain at least one sample and one feature")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(labels: ArrayLike, *, n_samples: int) -> FloatArray:
    vector = np.asarray(labels, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1 or vector.shape != (n_samples,):
        raise ValueError("labels must be a one-dimensional array matching features")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("labels must lie in [0, 1]")
    return vector.astype(np.float64, copy=True)


def _as_parameter_vector(params: ArrayLike, *, n_features: int) -> FloatArray:
    vector = np.asarray(params, dtype=float)
    if vector.ndim != 1 or vector.shape != (n_features,):
        raise ValueError("params must be a one-dimensional vector matching feature width")
    if not np.all(np.isfinite(vector)):
        raise ValueError("params must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_positive_int(name: str, value: int) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


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


def _phase_qnn_probabilities(features: FloatArray, params: FloatArray) -> FloatArray:
    probabilities = 0.5 * (1.0 - np.cos(features + params[None, :]))
    averaged = np.mean(probabilities, axis=1)
    return cast(FloatArray, np.clip(averaged, 0.0, 1.0).astype(np.float64, copy=False))


def _sampled_loss(
    features: FloatArray,
    labels: FloatArray,
    params: FloatArray,
    *,
    shots_per_sample: int,
    seed: int,
) -> tuple[float, float]:
    probabilities = _phase_qnn_probabilities(features, params)
    rng = np.random.default_rng(seed)
    counts = rng.binomial(shots_per_sample, probabilities).astype(np.float64, copy=False)
    sampled_probabilities = counts / float(shots_per_sample)
    residual = sampled_probabilities - labels
    loss = float(np.mean(residual * residual))
    probability_variance = sampled_probabilities * (1.0 - sampled_probabilities)
    contribution_variance = 4.0 * residual * residual * probability_variance
    loss_variance = float(
        np.sum(contribution_variance) / (features.shape[0] ** 2 * shots_per_sample)
    )
    return loss, max(0.0, loss_variance)


def estimate_parameter_shift_qnn_finite_shot_gradient(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    shots_per_sample: int = 8192,
    seed: int = 20260605,
    confidence_z: float = 1.959963984540054,
) -> ParameterShiftQNNFiniteShotGradientResult:
    """Estimate bounded-QNN MSE gradients from seeded finite-shot samples."""
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_vector = _as_parameter_vector(params, n_features=feature_matrix.shape[1])
    shots = _as_positive_int("shots_per_sample", shots_per_sample)
    root_seed = _as_positive_int("seed", seed)
    z_value = _as_positive_float("confidence_z", confidence_z)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    gradient = np.zeros_like(parameter_vector, dtype=np.float64)
    gradient_variance = np.zeros_like(parameter_vector, dtype=np.float64)
    records: list[ParameterShiftQNNFiniteShotProbeRecord] = []
    rng = np.random.default_rng(root_seed)

    for parameter_index in range(parameter_vector.size):
        for term_index, (shift, coefficient) in enumerate(rule.terms):
            plus = parameter_vector.copy()
            minus = parameter_vector.copy()
            plus[parameter_index] += shift
            minus[parameter_index] -= shift
            plus_seed = int(rng.integers(1, np.iinfo(np.int32).max))
            minus_seed = int(rng.integers(1, np.iinfo(np.int32).max))
            plus_loss, plus_loss_variance = _sampled_loss(
                feature_matrix,
                label_vector,
                plus,
                shots_per_sample=shots,
                seed=plus_seed,
            )
            minus_loss, minus_loss_variance = _sampled_loss(
                feature_matrix,
                label_vector,
                minus,
                shots_per_sample=shots,
                seed=minus_seed,
            )
            gradient[parameter_index] += coefficient * (plus_loss - minus_loss)
            gradient_variance[parameter_index] += (
                coefficient * coefficient * (plus_loss_variance + minus_loss_variance)
            )
            records.append(
                ParameterShiftQNNFiniteShotProbeRecord(
                    parameter_index=parameter_index,
                    term_index=term_index,
                    shift=shift,
                    coefficient=coefficient,
                    plus_loss=plus_loss,
                    minus_loss=minus_loss,
                    plus_loss_variance=plus_loss_variance,
                    minus_loss_variance=minus_loss_variance,
                    plus_seed=plus_seed,
                    minus_seed=minus_seed,
                    shots_per_sample=shots,
                    n_samples=feature_matrix.shape[0],
                )
            )

    deterministic = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_vector,
    )
    standard_error = np.sqrt(np.maximum(gradient_variance, 0.0))
    confidence_radius = z_value * standard_error
    delta = gradient - deterministic
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    max_standard_error = float(np.max(standard_error)) if standard_error.size else 0.0
    max_confidence_radius = float(np.max(confidence_radius)) if confidence_radius.size else 0.0
    return ParameterShiftQNNFiniteShotGradientResult(
        loss=parameter_shift_qnn_classifier_loss(feature_matrix, label_vector, parameter_vector),
        deterministic_gradient=deterministic,
        finite_shot_gradient=gradient.astype(np.float64, copy=False),
        standard_error=standard_error.astype(np.float64, copy=False),
        confidence_radius=confidence_radius.astype(np.float64, copy=False),
        max_abs_error=max_abs_error,
        max_standard_error=max_standard_error,
        max_confidence_radius=max_confidence_radius,
        confidence_z=z_value,
        shots_per_sample=shots,
        seed=root_seed,
        records=tuple(records),
        evidence_class=GRADIENT_EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        hardware_execution=False,
    )


def _default_convergence_cases() -> tuple[_FiniteShotConvergenceCase, ...]:
    return (
        _FiniteShotConvergenceCase(
            name="single_feature_finite_shot_phase_flip",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            initial_params=np.array([0.8], dtype=np.float64),
            shots_per_sample=8192,
            seed=31,
            learning_rate=0.7,
            max_steps=60,
            target_loss_tolerance=2e-3,
            min_loss_drop=0.015,
        ),
        _FiniteShotConvergenceCase(
            name="two_feature_finite_shot_phase_flip",
            features=np.array([[0.0, 0.0], [np.pi, np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            initial_params=np.array([0.8, 0.8], dtype=np.float64),
            shots_per_sample=8192,
            seed=53,
            learning_rate=0.7,
            max_steps=70,
            target_loss_tolerance=4e-3,
            min_loss_drop=0.015,
        ),
    )


def _convergence_case_map() -> dict[str, _FiniteShotConvergenceCase]:
    return {case.name: case for case in _default_convergence_cases()}


def _selected_convergence_cases(
    case_names: Sequence[str] | None,
) -> tuple[_FiniteShotConvergenceCase, ...]:
    cases = _convergence_case_map()
    if case_names is None:
        return tuple(cases.values())
    selected: list[_FiniteShotConvergenceCase] = []
    for name in case_names:
        if name not in cases:
            raise ValueError(f"unknown QNN finite-shot convergence case: {name}")
        selected.append(cases[name])
    return tuple(selected)


def _run_convergence_case(
    case: _FiniteShotConvergenceCase,
    *,
    min_loss_drop: float | None,
) -> ParameterShiftQNNFiniteShotConvergenceCaseResult:
    required_loss_drop = case.min_loss_drop if min_loss_drop is None else min_loss_drop
    params = case.initial_params.copy()
    initial_loss = parameter_shift_qnn_classifier_loss(case.features, case.labels, params)
    best_loss = initial_loss
    final_loss = initial_loss
    total_shots = 0
    total_evaluations = 0
    max_standard_error = 0.0
    accepted_steps = 0
    for step in range(case.max_steps):
        gradient_result = estimate_parameter_shift_qnn_finite_shot_gradient(
            case.features,
            case.labels,
            params,
            shots_per_sample=case.shots_per_sample,
            seed=case.seed + step,
            confidence_z=3.0,
        )
        params = params - case.learning_rate * gradient_result.finite_shot_gradient
        final_loss = parameter_shift_qnn_classifier_loss(case.features, case.labels, params)
        best_loss = min(best_loss, final_loss)
        total_shots += gradient_result.total_shots
        total_evaluations += gradient_result.parameter_shift_evaluations
        max_standard_error = max(max_standard_error, gradient_result.max_standard_error)
        accepted_steps += 1
        if best_loss <= case.target_loss_tolerance:
            break
    prediction = predict_parameter_shift_qnn_classifier(case.features, params, labels=case.labels)
    if prediction.accuracy is None or prediction.accuracy < 1.0:
        best_loss = min(best_loss, final_loss)
    return ParameterShiftQNNFiniteShotConvergenceCaseResult(
        name=case.name,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        loss_drop=float(initial_loss - best_loss),
        target_loss_tolerance=case.target_loss_tolerance,
        min_loss_drop=required_loss_drop,
        learning_rate=case.learning_rate,
        max_steps=case.max_steps,
        accepted_steps=accepted_steps,
        shots_per_sample=case.shots_per_sample,
        seed=case.seed,
        total_shots=total_shots,
        total_parameter_shift_evaluations=total_evaluations,
        max_gradient_standard_error=max_standard_error,
        evidence_class=CONVERGENCE_EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        hardware_execution=False,
        production_benchmark=False,
    )


def summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios() -> tuple[
    ParameterShiftQNNFiniteShotUnsupportedScenario, ...
]:
    """Return unsupported finite-shot QNN scenarios and mitigations."""
    status = "fail_closed_or_staged"
    return (
        ParameterShiftQNNFiniteShotUnsupportedScenario(
            name="hardware_provider_jobs",
            status=status,
            reason=(
                "the finite-shot QNN route samples a local seeded Bernoulli simulator "
                "and does not submit provider jobs or certify queue/hardware drift"
            ),
            mitigation=(
                "route through provider-gradient readiness with shifted-job lineage, "
                "raw counts, queue metadata, and hardware-result ledger promotion"
            ),
        ),
        ParameterShiftQNNFiniteShotUnsupportedScenario(
            name="unseeded_stochastic_training",
            status=status,
            reason="unseeded stochastic training cannot be replayed or audited deterministically",
            mitigation="provide explicit seeds, shot counts, and recorded shifted-loss probes",
        ),
        ParameterShiftQNNFiniteShotUnsupportedScenario(
            name="low_shot_gradient_promotion",
            status=status,
            reason=(
                "low-shot gradients can pass by chance unless confidence radii, "
                "repeat seeds, and promotion thresholds are recorded"
            ),
            mitigation=(
                "increase shots, require multi-seed envelopes, and keep low-shot runs "
                "as exploratory evidence"
            ),
        ),
        ParameterShiftQNNFiniteShotUnsupportedScenario(
            name="arbitrary_qnn_architecture",
            status=status,
            reason=(
                "finite-shot evidence covers the bounded phase classifier only, not "
                "arbitrary QNN/QGNN/QSNN architectures"
            ),
            mitigation=(
                "add architecture-specific finite-shot estimators, tests, and claim "
                "boundaries before promotion"
            ),
        ),
    )


def run_parameter_shift_qnn_finite_shot_convergence_suite(
    *,
    case_names: Sequence[str] | None = None,
    min_loss_drop: float | None = None,
) -> ParameterShiftQNNFiniteShotConvergenceSuiteResult:
    """Run seeded finite-shot noisy-gradient convergence evidence."""
    checked_min_loss_drop = (
        None if min_loss_drop is None else _as_non_negative_float("min_loss_drop", min_loss_drop)
    )
    cases = tuple(
        _run_convergence_case(case, min_loss_drop=checked_min_loss_drop)
        for case in _selected_convergence_cases(case_names)
    )
    return ParameterShiftQNNFiniteShotConvergenceSuiteResult(
        cases=cases,
        unsuitable_scenarios=summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios(),
        evidence_class=CONVERGENCE_EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        hardware_execution=False,
        production_benchmark=False,
    )
