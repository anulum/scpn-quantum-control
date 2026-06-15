# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNN Conformance Suite
"""Conformance evidence for bounded phase-QNN differentiable workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    train_parameter_shift_qnn_classifier,
    verify_parameter_shift_qnn_classifier_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
GradientCallable: TypeAlias = Callable[[FloatArray], ArrayLike]
ExternalGradientMap: TypeAlias = Mapping[str, Mapping[str, GradientCallable]]


@dataclass(frozen=True)
class _ParameterShiftQNNSuitableCase:
    """Internal deterministic QNN conformance case definition."""

    name: str
    features: FloatArray
    labels: FloatArray
    params: FloatArray
    training_initial_params: FloatArray | None = None
    training_learning_rate: float = 0.7
    training_max_steps: int = 80
    training_target_loss_tolerance: float = 1e-4


@dataclass(frozen=True)
class ParameterShiftQNNUnsupportedScenario:
    """Explicit unsuitable QNN scenario and its current fail-closed boundary."""

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
class ParameterShiftQNNConformanceCaseResult:
    """Evidence for one bounded phase-QNN conformance case."""

    name: str
    n_samples: int
    n_features: int
    loss: float
    finite_difference_passed: bool
    max_abs_error: float
    l2_error: float
    tolerance: float
    finite_difference_step: float
    method: str
    shift_terms: int
    parameter_shift_evaluations: int
    training_required: bool
    training_passed: bool
    training_best_loss: float | None
    training_accuracy: float | None
    external_agreement_names: tuple[str, ...]
    external_source_classes: tuple[str, ...]
    external_native_framework_autodiff: bool
    external_passed: bool

    @property
    def external_agreement_count(self) -> int:
        """Return the number of external gradient agreements recorded."""
        return len(self.external_agreement_names)

    @property
    def passed(self) -> bool:
        """Return whether this conformance case passed every required check."""
        training_ok = (not self.training_required) or self.training_passed
        return bool(self.finite_difference_passed and training_ok and self.external_passed)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready conformance case evidence."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "loss": self.loss,
            "finite_difference_passed": self.finite_difference_passed,
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "finite_difference_step": self.finite_difference_step,
            "method": self.method,
            "shift_terms": self.shift_terms,
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "training_required": self.training_required,
            "training_passed": self.training_passed,
            "training_best_loss": self.training_best_loss,
            "training_accuracy": self.training_accuracy,
            "external_agreement_names": list(self.external_agreement_names),
            "external_source_classes": list(self.external_source_classes),
            "external_native_framework_autodiff": self.external_native_framework_autodiff,
            "external_agreement_count": self.external_agreement_count,
            "external_passed": self.external_passed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ParameterShiftQNNConformanceSuiteResult:
    """Bundled bounded-QNN conformance evidence."""

    cases: tuple[ParameterShiftQNNConformanceCaseResult, ...]
    unsuitable_scenarios: tuple[ParameterShiftQNNUnsupportedScenario, ...]

    @property
    def passed(self) -> bool:
        """Return whether every conformance case passed."""
        return all(case.passed for case in self.cases)

    @property
    def case_count(self) -> int:
        """Return the number of conformance cases."""
        return len(self.cases)

    @property
    def gradient_passed_count(self) -> int:
        """Return the number of cases with finite-difference gradient agreement."""
        return sum(1 for case in self.cases if case.finite_difference_passed)

    @property
    def training_passed_count(self) -> int:
        """Return the number of required training cases that passed."""
        return sum(1 for case in self.cases if case.training_required and case.training_passed)

    @property
    def external_agreement_count(self) -> int:
        """Return the total number of external gradient agreements recorded."""
        return sum(case.external_agreement_count for case in self.cases)

    @property
    def unsuitable_scenario_count(self) -> int:
        """Return the number of documented unsuitable scenarios."""
        return len(self.unsuitable_scenarios)

    def case_by_name(self, name: str) -> ParameterShiftQNNConformanceCaseResult:
        """Return a conformance case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown QNN conformance case: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "gradient_passed_count": self.gradient_passed_count,
            "training_passed_count": self.training_passed_count,
            "external_agreement_count": self.external_agreement_count,
            "unsuitable_scenario_count": self.unsuitable_scenario_count,
            "cases": [case.to_dict() for case in self.cases],
            "unsuitable_scenarios": [scenario.to_dict() for scenario in self.unsuitable_scenarios],
        }


def _case_definitions() -> tuple[_ParameterShiftQNNSuitableCase, ...]:
    return (
        _ParameterShiftQNNSuitableCase(
            name="phase_separable_single_feature",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            params=np.array([0.45], dtype=np.float64),
            training_initial_params=np.array([0.8], dtype=np.float64),
        ),
        _ParameterShiftQNNSuitableCase(
            name="two_feature_mixed_phase",
            features=np.array(
                [[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]],
                dtype=np.float64,
            ),
            labels=np.array([0.0, 1.0, 0.25], dtype=np.float64),
            params=np.array([0.4, -0.2], dtype=np.float64),
        ),
        _ParameterShiftQNNSuitableCase(
            name="balanced_threshold_two_feature",
            features=np.array([[0.0, np.pi], [np.pi, 0.0]], dtype=np.float64),
            labels=np.array([0.5, 0.5], dtype=np.float64),
            params=np.array([0.0, 0.0], dtype=np.float64),
        ),
    )


def summarize_parameter_shift_qnn_unsuitable_scenarios() -> tuple[
    ParameterShiftQNNUnsupportedScenario, ...
]:
    """Return explicit unsuitable scenarios for the bounded phase-QNN route."""
    status = "fail_closed_or_staged"
    return (
        ParameterShiftQNNUnsupportedScenario(
            name="hardware_backend",
            status=status,
            reason=(
                "bounded QNN conformance is local deterministic evidence; hardware "
                "gradient jobs need explicit provider policy, shifted-job lineage, "
                "shot allocation, and variance accounting"
            ),
            mitigation=(
                "use local conformance first, then promote through provider-gradient "
                "readiness and finite-shot provenance gates"
            ),
        ),
        ParameterShiftQNNUnsupportedScenario(
            name="arbitrary_qnn_architecture",
            status=status,
            reason=(
                "the implemented classifier has one trainable phase per feature and "
                "does not represent arbitrary QNN/QGNN/QSNN architectures"
            ),
            mitigation=(
                "add a dedicated architecture module, analytic gradient contract, "
                "module-specific tests, and convergence evidence before promotion"
            ),
        ),
        ParameterShiftQNNUnsupportedScenario(
            name="nonfinite_training_data",
            status=status,
            reason=(
                "non-finite features, labels, parameters, or external gradients make "
                "loss and gradient evidence non-reviewable"
            ),
            mitigation=(
                "clean or reject the dataset before calling the QNN training or verification APIs"
            ),
        ),
        ParameterShiftQNNUnsupportedScenario(
            name="native_framework_autodiff",
            status=status,
            reason=(
                "external-gradient agreements record caller-supplied references and "
                "do not claim native autodiff-through-simulator kernels"
            ),
            mitigation=(
                "route native framework kernels through a separate adapter with "
                "round-trip tests and explicit host/device boundary metadata"
            ),
        ),
    )


def _validate_external_case_names(
    external_gradients: ExternalGradientMap | None,
    *,
    known_names: set[str],
) -> None:
    if external_gradients is None:
        return
    unknown = set(external_gradients) - known_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"external gradient case not found: {names}")


def _run_training_case(
    case: _ParameterShiftQNNSuitableCase,
) -> tuple[bool, float | None, float | None]:
    if case.training_initial_params is None:
        return False, None, None
    training = train_parameter_shift_qnn_classifier(
        case.features,
        case.labels,
        initial_params=case.training_initial_params,
        learning_rate=case.training_learning_rate,
        max_steps=case.training_max_steps,
        target_loss=0.0,
        target_loss_tolerance=case.training_target_loss_tolerance,
    )
    accuracy = training.prediction.accuracy
    passed = bool(
        training.certificate.monotone_accepted_values
        and training.certificate.within_target_value_tolerance
        and accuracy == 1.0
    )
    return passed, training.best_loss, accuracy


def run_parameter_shift_qnn_conformance_suite(
    *,
    finite_difference_step: float = 1e-6,
    tolerance: float = 2e-6,
    external_gradients: ExternalGradientMap | None = None,
    external_tolerance: float | None = None,
) -> ParameterShiftQNNConformanceSuiteResult:
    """Run deterministic bounded phase-QNN conformance checks.

    The suite intentionally covers the production-bounded QNN surface:
    multi-frequency parameter-shift gradients for the QNN MSE loss, independent
    central finite-difference replay, one convergence case, optional named
    external-gradient agreements, and explicit records for unsuitable scenarios.
    """
    cases = _case_definitions()
    _validate_external_case_names(
        external_gradients,
        known_names={case.name for case in cases},
    )
    results: list[ParameterShiftQNNConformanceCaseResult] = []

    for case in cases:
        case_external_gradients = (
            None if external_gradients is None else external_gradients.get(case.name)
        )
        verification = verify_parameter_shift_qnn_classifier_gradient(
            case.features,
            case.labels,
            case.params,
            finite_difference_step=finite_difference_step,
            tolerance=tolerance,
            external_gradients=case_external_gradients,
            external_tolerance=external_tolerance,
        )
        training_required = case.training_initial_params is not None
        training_passed, training_best_loss, training_accuracy = _run_training_case(case)
        external_passed = all(agreement.passed for agreement in verification.external_agreements)
        gradient = parameter_shift_qnn_classifier_gradient(
            case.features,
            case.labels,
            case.params,
        )
        results.append(
            ParameterShiftQNNConformanceCaseResult(
                name=case.name,
                n_samples=int(case.features.shape[0]),
                n_features=int(case.features.shape[1]),
                loss=verification.loss,
                finite_difference_passed=verification.passed
                if case_external_gradients is None
                else verification.max_abs_error <= verification.tolerance,
                max_abs_error=verification.max_abs_error,
                l2_error=verification.l2_error,
                tolerance=verification.tolerance,
                finite_difference_step=verification.finite_difference_step,
                method=verification.method,
                shift_terms=verification.shift_terms,
                parameter_shift_evaluations=2 * verification.shift_terms * gradient.size,
                training_required=training_required,
                training_passed=training_passed,
                training_best_loss=training_best_loss,
                training_accuracy=training_accuracy,
                external_agreement_names=tuple(
                    agreement.name for agreement in verification.external_agreements
                ),
                external_source_classes=tuple(
                    agreement.source_class for agreement in verification.external_agreements
                ),
                external_native_framework_autodiff=any(
                    agreement.native_framework_autodiff
                    for agreement in verification.external_agreements
                ),
                external_passed=external_passed,
            )
        )

    return ParameterShiftQNNConformanceSuiteResult(
        cases=tuple(results),
        unsuitable_scenarios=summarize_parameter_shift_qnn_unsuitable_scenarios(),
    )


__all__ = [
    "ExternalGradientMap",
    "ParameterShiftQNNConformanceCaseResult",
    "ParameterShiftQNNConformanceSuiteResult",
    "ParameterShiftQNNUnsupportedScenario",
    "run_parameter_shift_qnn_conformance_suite",
    "summarize_parameter_shift_qnn_unsuitable_scenarios",
]
