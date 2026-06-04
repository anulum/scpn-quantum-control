# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNN Convergence
"""Deterministic convergence evidence for bounded phase-QNN training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_loss,
    train_parameter_shift_qnn_classifier,
)

FloatArray: TypeAlias = NDArray[np.float64]

EVIDENCE_CLASS = "local_deterministic_qnn_convergence"
CLAIM_BOUNDARY = (
    "local deterministic bounded phase-QNN convergence evidence; not hardware, "
    "not finite-shot training, and not arbitrary QNN/QGNN/QSNN architecture evidence"
)


@dataclass(frozen=True)
class _QNNConvergenceCase:
    name: str
    features: FloatArray
    labels: FloatArray
    initial_params: FloatArray
    learning_rate: float
    max_steps: int
    target_loss_tolerance: float
    min_loss_drop: float
    min_accuracy: float


@dataclass(frozen=True)
class ParameterShiftQNNConvergenceUnsuitableScenario:
    """Unsupported convergence scenario and the current fail-closed boundary."""

    name: str
    status: str
    reason: str
    mitigation: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready unsuitable-scenario evidence."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "mitigation": self.mitigation,
        }


@dataclass(frozen=True)
class ParameterShiftQNNConvergenceCaseResult:
    """Convergence evidence for one bounded phase-QNN training case."""

    name: str
    n_samples: int
    n_features: int
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_drop: float
    accuracy: float | None
    target_loss_tolerance: float
    min_loss_drop: float
    min_accuracy: float
    learning_rate: float
    max_steps: int
    accepted_steps: int
    gradient_evaluations: int
    parameter_shift_evaluations: int
    evidence_class: str
    claim_boundary: str
    production_benchmark: bool

    @property
    def converged(self) -> bool:
        """Return whether the best loss reached the requested tolerance."""
        return self.best_loss <= self.target_loss_tolerance

    @property
    def loss_drop_passed(self) -> bool:
        """Return whether the training run reduced loss enough."""
        return self.loss_drop >= self.min_loss_drop

    @property
    def accuracy_passed(self) -> bool:
        """Return whether prediction accuracy met the requested threshold."""
        return self.accuracy is not None and self.accuracy >= self.min_accuracy

    @property
    def passed(self) -> bool:
        """Return whether this convergence case passed all thresholds."""
        return bool(self.converged and self.loss_drop_passed and self.accuracy_passed)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready convergence-case evidence."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_drop": self.loss_drop,
            "accuracy": self.accuracy,
            "target_loss_tolerance": self.target_loss_tolerance,
            "min_loss_drop": self.min_loss_drop,
            "min_accuracy": self.min_accuracy,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "accepted_steps": self.accepted_steps,
            "gradient_evaluations": self.gradient_evaluations,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "converged": self.converged,
            "loss_drop_passed": self.loss_drop_passed,
            "accuracy_passed": self.accuracy_passed,
            "passed": self.passed,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "production_benchmark": self.production_benchmark,
        }


@dataclass(frozen=True)
class ParameterShiftQNNConvergenceSuiteResult:
    """Bundled deterministic convergence evidence for bounded phase QNNs."""

    cases: tuple[ParameterShiftQNNConvergenceCaseResult, ...]
    unsuitable_scenarios: tuple[ParameterShiftQNNConvergenceUnsuitableScenario, ...]
    evidence_class: str
    claim_boundary: str
    production_benchmark: bool

    @property
    def passed(self) -> bool:
        """Return whether every convergence case passed."""
        return all(case.passed for case in self.cases)

    @property
    def case_count(self) -> int:
        """Return the number of convergence cases."""
        return len(self.cases)

    @property
    def passed_count(self) -> int:
        """Return the number of passing convergence cases."""
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed_count(self) -> int:
        """Return the number of failing convergence cases."""
        return self.case_count - self.passed_count

    @property
    def total_gradient_evaluations(self) -> int:
        """Return total gradient-evaluation accounting across cases."""
        return sum(case.gradient_evaluations for case in self.cases)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Return total parameter-shift objective evaluations across cases."""
        return sum(case.parameter_shift_evaluations for case in self.cases)

    @property
    def unsuitable_scenario_count(self) -> int:
        """Return the number of documented unsuitable scenarios."""
        return len(self.unsuitable_scenarios)

    def case_by_name(self, name: str) -> ParameterShiftQNNConvergenceCaseResult:
        """Return a convergence case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown QNN convergence case: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready convergence-suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_gradient_evaluations": self.total_gradient_evaluations,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "unsuitable_scenario_count": self.unsuitable_scenario_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "production_benchmark": self.production_benchmark,
            "cases": [case.to_dict() for case in self.cases],
            "unsuitable_scenarios": [scenario.to_dict() for scenario in self.unsuitable_scenarios],
        }


def _default_cases() -> tuple[_QNNConvergenceCase, ...]:
    return (
        _QNNConvergenceCase(
            name="single_feature_phase_flip",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            initial_params=np.array([0.8], dtype=np.float64),
            learning_rate=0.7,
            max_steps=80,
            target_loss_tolerance=1e-4,
            min_loss_drop=0.02,
            min_accuracy=1.0,
        ),
        _QNNConvergenceCase(
            name="two_feature_phase_flip",
            features=np.array([[0.0, 0.0], [np.pi, np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            initial_params=np.array([0.8, 0.8], dtype=np.float64),
            learning_rate=0.7,
            max_steps=90,
            target_loss_tolerance=5e-4,
            min_loss_drop=0.02,
            min_accuracy=1.0,
        ),
        _QNNConvergenceCase(
            name="three_feature_phase_flip",
            features=np.array(
                [[0.0, 0.0, 0.0], [np.pi, np.pi, np.pi]],
                dtype=np.float64,
            ),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            initial_params=np.array([0.8, 0.8, 0.8], dtype=np.float64),
            learning_rate=0.7,
            max_steps=100,
            target_loss_tolerance=5e-4,
            min_loss_drop=0.02,
            min_accuracy=1.0,
        ),
    )


def _case_map() -> dict[str, _QNNConvergenceCase]:
    return {case.name: case for case in _default_cases()}


def _selected_cases(case_names: Sequence[str] | None) -> tuple[_QNNConvergenceCase, ...]:
    cases = _case_map()
    if case_names is None:
        return tuple(cases.values())
    selected: list[_QNNConvergenceCase] = []
    for name in case_names:
        if name not in cases:
            raise ValueError(f"unknown QNN convergence case: {name}")
        selected.append(cases[name])
    return tuple(selected)


def _as_non_negative_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _as_probability_threshold(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return scalar


def _training_result_for_case(
    case: _QNNConvergenceCase,
    *,
    min_loss_drop: float | None,
    min_accuracy: float | None,
) -> ParameterShiftQNNConvergenceCaseResult:
    required_loss_drop = case.min_loss_drop if min_loss_drop is None else min_loss_drop
    required_accuracy = case.min_accuracy if min_accuracy is None else min_accuracy
    initial_loss = parameter_shift_qnn_classifier_loss(
        case.features,
        case.labels,
        case.initial_params,
    )
    training = train_parameter_shift_qnn_classifier(
        case.features,
        case.labels,
        initial_params=case.initial_params,
        learning_rate=case.learning_rate,
        max_steps=case.max_steps,
        gradient_tolerance=1e-7,
        target_loss=0.0,
        target_loss_tolerance=case.target_loss_tolerance,
    )
    loss_history = training.loss_history
    final_loss = float(loss_history[-1]) if loss_history else training.best_loss
    accepted_steps = max(1, len(loss_history) - 1)
    gradient_evaluations = accepted_steps
    parameter_shift_evaluations = gradient_evaluations * 4 * case.features.shape[1]
    loss_drop = float(initial_loss - training.best_loss)
    return ParameterShiftQNNConvergenceCaseResult(
        name=case.name,
        n_samples=case.features.shape[0],
        n_features=case.features.shape[1],
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=training.best_loss,
        loss_drop=loss_drop,
        accuracy=training.prediction.accuracy,
        target_loss_tolerance=case.target_loss_tolerance,
        min_loss_drop=required_loss_drop,
        min_accuracy=required_accuracy,
        learning_rate=case.learning_rate,
        max_steps=case.max_steps,
        accepted_steps=accepted_steps,
        gradient_evaluations=gradient_evaluations,
        parameter_shift_evaluations=parameter_shift_evaluations,
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        production_benchmark=False,
    )


def summarize_parameter_shift_qnn_convergence_unsuitable_scenarios() -> tuple[
    ParameterShiftQNNConvergenceUnsuitableScenario, ...
]:
    """Return unsuitable scenarios for bounded QNN convergence evidence."""
    status = "fail_closed_or_staged"
    return (
        ParameterShiftQNNConvergenceUnsuitableScenario(
            name="hardware_backend_convergence",
            status=status,
            reason=(
                "this convergence suite is local deterministic evidence and does not "
                "allocate provider shots, track queue variance, or certify hardware jobs"
            ),
            mitigation=(
                "promote through provider-gradient readiness, shifted-job lineage, "
                "finite-shot variance, and hardware-result ledger gates"
            ),
        ),
        ParameterShiftQNNConvergenceUnsuitableScenario(
            name="finite_shot_noisy_training",
            status=status,
            reason=(
                "finite-shot convergence needs stochastic confidence intervals and "
                "repeat-seed variance envelopes before pass/fail claims are meaningful"
            ),
            mitigation=(
                "add a stochastic convergence module with shot budgets, random seeds, "
                "confidence bounds, and non-isolated versus isolated evidence labels"
            ),
        ),
        ParameterShiftQNNConvergenceUnsuitableScenario(
            name="arbitrary_qnn_architecture",
            status=status,
            reason=(
                "the implemented training lane covers a bounded data-reuploading "
                "phase classifier, not arbitrary QNN, QGNN, or QSNN architectures"
            ),
            mitigation=(
                "add one architecture-specific module at a time with analytic gradient "
                "contracts, module-specific tests, and convergence certificates"
            ),
        ),
        ParameterShiftQNNConvergenceUnsuitableScenario(
            name="native_framework_autodiff_training",
            status=status,
            reason=(
                "the suite can compare caller-supplied framework-style gradients but "
                "does not train through native framework autodiff simulator kernels"
            ),
            mitigation=(
                "route native framework training through dedicated JAX, PyTorch, "
                "TensorFlow, or PennyLane adapters with round-trip device metadata"
            ),
        ),
    )


def run_parameter_shift_qnn_convergence_suite(
    *,
    case_names: Sequence[str] | None = None,
    min_loss_drop: float | None = None,
    min_accuracy: float | None = None,
) -> ParameterShiftQNNConvergenceSuiteResult:
    """Run deterministic convergence evidence for bounded phase-QNN cases."""
    checked_min_loss_drop = (
        None if min_loss_drop is None else _as_non_negative_float("min_loss_drop", min_loss_drop)
    )
    checked_min_accuracy = (
        None if min_accuracy is None else _as_probability_threshold("min_accuracy", min_accuracy)
    )
    cases = tuple(
        _training_result_for_case(
            case,
            min_loss_drop=checked_min_loss_drop,
            min_accuracy=checked_min_accuracy,
        )
        for case in _selected_cases(case_names)
    )
    return ParameterShiftQNNConvergenceSuiteResult(
        cases=cases,
        unsuitable_scenarios=summarize_parameter_shift_qnn_convergence_unsuitable_scenarios(),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        production_benchmark=False,
    )
