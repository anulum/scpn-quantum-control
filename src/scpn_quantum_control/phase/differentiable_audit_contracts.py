# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Gradient Audit Contracts
"""Immutable reports and serializers for differentiable gradient audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import ShotAllocationResult, StochasticGradientResult
from .coupling_learning import CouplingGradientVerificationResult, CouplingLearningResult
from .gradient_descent import ParameterShiftTrainingCertificate, ParameterShiftTrainingResult
from .param_shift import GradientVerificationResult

FloatArray: TypeAlias = NDArray[np.float64]


def _as_finite_vector(name: str, values: ArrayLike) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must contain finite real numeric values")
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite real numeric values")
    return vector.astype(np.float64, copy=True)


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(raw.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _stochastic_gradient_to_dict(result: StochasticGradientResult) -> dict[str, object]:
    return {
        "value": result.value,
        "gradient": result.gradient.tolist(),
        "standard_error": result.standard_error.tolist(),
        "covariance": result.covariance.tolist(),
        "confidence_radius": result.confidence_radius.tolist(),
        "shots": np.asarray(result.shots, dtype=float).tolist(),
        "confidence_level": result.confidence_level,
        "method": result.method,
        "shift": result.shift,
        "coefficient": result.coefficient,
        "evaluations": result.evaluations,
        "parameter_names": list(result.parameter_names),
        "trainable": list(result.trainable),
    }


def _shot_allocation_to_dict(result: ShotAllocationResult) -> dict[str, object]:
    return {
        "shots": np.asarray(result.shots, dtype=float).tolist(),
        "predicted_standard_error": result.predicted_standard_error.tolist(),
        "covariance": result.covariance.tolist(),
        "target_standard_error": result.target_standard_error,
        "total_shots": result.total_shots,
        "method": result.method,
        "parameter_names": list(result.parameter_names),
        "trainable": list(result.trainable),
    }


@dataclass(frozen=True)
class ParameterShiftAnalyticAgreement:
    """Agreement certificate between parameter-shift and analytic gradients."""

    parameters: FloatArray
    parameter_shift_gradient: FloatArray
    analytic_gradient: FloatArray
    abs_error: FloatArray
    max_abs_error: float
    tolerance: float
    passed: bool
    method: str
    evaluations: int
    claim_boundary: str

    def __post_init__(self) -> None:
        parameters = _as_finite_vector("parameters", self.parameters)
        parameter_shift = _as_finite_vector(
            "parameter_shift_gradient",
            self.parameter_shift_gradient,
        )
        analytic = _as_finite_vector("analytic_gradient", self.analytic_gradient)
        abs_error = _as_finite_vector("abs_error", self.abs_error)
        if parameter_shift.shape != parameters.shape:
            raise ValueError("parameter_shift_gradient shape must match parameters")
        if analytic.shape != parameters.shape:
            raise ValueError("analytic_gradient shape must match parameters")
        if abs_error.shape != parameters.shape:
            raise ValueError("abs_error shape must match parameters")
        if np.any(abs_error < 0.0):
            raise ValueError("abs_error must be non-negative")
        max_abs_error = _as_finite_scalar("max_abs_error", self.max_abs_error)
        tolerance = _as_finite_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.method:
            raise ValueError("method must be non-empty")
        if isinstance(self.evaluations, bool) or self.evaluations < 1:
            raise ValueError("evaluations must be a positive integer")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "parameter_shift_gradient", parameter_shift)
        object.__setattr__(self, "analytic_gradient", analytic)
        object.__setattr__(self, "abs_error", abs_error)
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready analytic agreement evidence."""
        return {
            "parameters": self.parameters.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "analytic_gradient": self.analytic_gradient.tolist(),
            "abs_error": self.abs_error.tolist(),
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "method": self.method,
            "evaluations": self.evaluations,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableQuantumAuditReport:
    """Composite evidence report for parameter-shift differentiable workflows."""

    finite_difference: GradientVerificationResult
    analytic: ParameterShiftAnalyticAgreement
    training: ParameterShiftTrainingResult
    training_certificate: ParameterShiftTrainingCertificate
    passed: bool
    task: str
    claim_boundary: str

    @property
    def best_value(self) -> float:
        """Return the best scalar objective value reached during training."""
        return self.training.best_value

    @property
    def max_gradient_error(self) -> float:
        """Return the largest independent gradient-check error."""
        return max(self.finite_difference.max_abs_error, self.analytic.max_abs_error)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready differentiable-programming audit evidence."""
        return {
            "finite_difference": self.finite_difference.to_dict(),
            "analytic": self.analytic.to_dict(),
            "training": self.training.to_dict(),
            "training_certificate": self.training_certificate.to_dict(),
            "passed": self.passed,
            "task": self.task,
            "best_value": self.best_value,
            "max_gradient_error": self.max_gradient_error,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseGradientBenchmarkSuiteResult:
    """Aggregate result for the built-in phase-gradient conformance suite."""

    benchmark_names: tuple[str, ...]
    reports: tuple[DifferentiableQuantumAuditReport, ...]
    unsupported_scenarios: tuple[str, ...]
    passed: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.benchmark_names:
            raise ValueError("benchmark_names must be non-empty")
        if len(self.benchmark_names) != len(self.reports):
            raise ValueError("benchmark_names length must match reports length")
        if any(not name for name in self.benchmark_names):
            raise ValueError("benchmark_names must contain non-empty names")
        if not self.unsupported_scenarios:
            raise ValueError("unsupported_scenarios must be non-empty")
        if any(not item for item in self.unsupported_scenarios):
            raise ValueError("unsupported_scenarios must contain non-empty items")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @property
    def worst_gradient_error(self) -> float:
        """Return the largest gradient error across all benchmark reports."""
        return max(report.max_gradient_error for report in self.reports)

    @property
    def best_values(self) -> tuple[float, ...]:
        """Return best objective values for each benchmark case."""
        return tuple(report.best_value for report in self.reports)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark-suite evidence."""
        return {
            "benchmark_names": list(self.benchmark_names),
            "reports": [
                {"name": name, "report": report.to_dict()}
                for name, report in zip(self.benchmark_names, self.reports, strict=True)
            ],
            "unsupported_scenarios": list(self.unsupported_scenarios),
            "passed": self.passed,
            "worst_gradient_error": self.worst_gradient_error,
            "best_values": list(self.best_values),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableWorkflowAuditSuiteResult:
    """Aggregate evidence across supported differentiable quantum workflows."""

    phase_benchmarks: PhaseGradientBenchmarkSuiteResult
    finite_shot: FiniteShotGradientAuditResult
    coupling_gradient: CouplingGradientVerificationResult
    coupling_learning: CouplingLearningResult
    workflow_names: tuple[str, ...]
    unsupported_scenarios: tuple[str, ...]
    passed: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.workflow_names:
            raise ValueError("workflow_names must be non-empty")
        if any(not name for name in self.workflow_names):
            raise ValueError("workflow_names must contain non-empty names")
        if not self.unsupported_scenarios:
            raise ValueError("unsupported_scenarios must be non-empty")
        if any(not item for item in self.unsupported_scenarios):
            raise ValueError("unsupported_scenarios must contain non-empty items")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @property
    def worst_gradient_error(self) -> float:
        """Return the largest gradient error across all audit workflows."""
        return max(
            self.phase_benchmarks.worst_gradient_error,
            self.finite_shot.max_abs_error,
            self.coupling_gradient.max_abs_error,
        )

    @property
    def best_training_values(self) -> tuple[float, float]:
        """Return best training values for phase and coupling-training lanes."""
        return (
            min(self.phase_benchmarks.best_values),
            self.coupling_learning.best_loss,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready workflow-audit evidence."""
        return {
            "phase_benchmarks": self.phase_benchmarks.to_dict(),
            "finite_shot": self.finite_shot.to_dict(),
            "coupling_gradient": self.coupling_gradient.to_dict(),
            "coupling_learning": self.coupling_learning.to_dict(),
            "workflow_names": list(self.workflow_names),
            "unsupported_scenarios": list(self.unsupported_scenarios),
            "passed": self.passed,
            "worst_gradient_error": self.worst_gradient_error,
            "best_training_values": list(self.best_training_values),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class FiniteShotGradientAuditResult:
    """Finite-shot uncertainty containment certificate for parameter-shift gradients."""

    deterministic_gradient: FloatArray
    stochastic: StochasticGradientResult
    shot_allocation: ShotAllocationResult
    abs_error: FloatArray
    within_confidence: tuple[bool, ...]
    target_standard_error: float
    max_abs_error: float
    max_confidence_radius: float
    max_standard_error: float
    executed_total_shots: int
    passed: bool
    method: str
    claim_boundary: str

    def __post_init__(self) -> None:
        deterministic = _as_finite_vector(
            "deterministic_gradient",
            self.deterministic_gradient,
        )
        abs_error = _as_finite_vector("abs_error", self.abs_error)
        if deterministic.shape != self.stochastic.gradient.shape:
            raise ValueError("deterministic_gradient shape must match stochastic gradient")
        if abs_error.shape != deterministic.shape:
            raise ValueError("abs_error shape must match deterministic_gradient")
        if len(self.within_confidence) != deterministic.size:
            raise ValueError("within_confidence length must match gradient width")
        if np.any(abs_error < 0.0):
            raise ValueError("abs_error must be non-negative")
        target_standard_error = _as_finite_scalar(
            "target_standard_error",
            self.target_standard_error,
        )
        max_abs_error = _as_finite_scalar("max_abs_error", self.max_abs_error)
        max_confidence_radius = _as_finite_scalar(
            "max_confidence_radius",
            self.max_confidence_radius,
        )
        max_standard_error = _as_finite_scalar("max_standard_error", self.max_standard_error)
        if target_standard_error <= 0.0:
            raise ValueError("target_standard_error must be finite and positive")
        if max_abs_error < 0.0 or max_confidence_radius < 0.0 or max_standard_error < 0.0:
            raise ValueError("finite-shot audit maxima must be non-negative")
        if isinstance(self.executed_total_shots, bool) or self.executed_total_shots < 1:
            raise ValueError("executed_total_shots must be a positive integer")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not self.method:
            raise ValueError("method must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "deterministic_gradient", deterministic)
        object.__setattr__(self, "abs_error", abs_error)
        object.__setattr__(self, "target_standard_error", target_standard_error)
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "max_confidence_radius", max_confidence_radius)
        object.__setattr__(self, "max_standard_error", max_standard_error)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready finite-shot audit evidence."""
        return {
            "deterministic_gradient": self.deterministic_gradient.tolist(),
            "stochastic": _stochastic_gradient_to_dict(self.stochastic),
            "shot_allocation": _shot_allocation_to_dict(self.shot_allocation),
            "abs_error": self.abs_error.tolist(),
            "within_confidence": list(self.within_confidence),
            "target_standard_error": self.target_standard_error,
            "max_abs_error": self.max_abs_error,
            "max_confidence_radius": self.max_confidence_radius,
            "max_standard_error": self.max_standard_error,
            "executed_total_shots": self.executed_total_shots,
            "passed": self.passed,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class MLFrameworkGradientAuditRecord:
    """Per-framework parity evidence for optional ML gradient adapters."""

    framework: str
    available: bool
    executed: bool
    status: str
    reason: str
    value: float | None
    gradient: FloatArray | None
    reference_gradient: FloatArray
    abs_error: FloatArray | None
    max_abs_error: float | None
    tolerance: float
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.framework:
            raise ValueError("framework must be non-empty")
        if not isinstance(self.available, bool) or not isinstance(self.executed, bool):
            raise ValueError("available and executed must be booleans")
        if self.status not in {"passed", "failed", "unavailable", "blocked"}:
            raise ValueError("status must be passed, failed, unavailable, or blocked")
        if not self.reason:
            raise ValueError("reason must be non-empty")
        reference = _as_finite_vector("reference_gradient", self.reference_gradient)
        tolerance = _as_finite_scalar("tolerance", self.tolerance)
        if tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        value = None if self.value is None else _as_finite_scalar("adapter value", self.value)
        gradient = (
            None if self.gradient is None else _as_finite_vector("adapter gradient", self.gradient)
        )
        abs_error = (
            None if self.abs_error is None else _as_finite_vector("abs_error", self.abs_error)
        )
        if gradient is not None and gradient.shape != reference.shape:
            raise ValueError("adapter gradient shape must match reference_gradient")
        if abs_error is not None and abs_error.shape != reference.shape:
            raise ValueError("abs_error shape must match reference_gradient")
        max_abs_error = (
            None
            if self.max_abs_error is None
            else _as_finite_scalar("max_abs_error", self.max_abs_error)
        )
        if max_abs_error is not None and max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "reference_gradient", reference)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "abs_error", abs_error)
        object.__setattr__(self, "max_abs_error", max_abs_error)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready per-framework ML parity evidence."""
        return {
            "framework": self.framework,
            "available": self.available,
            "executed": self.executed,
            "status": self.status,
            "reason": self.reason,
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "reference_gradient": self.reference_gradient.tolist(),
            "abs_error": None if self.abs_error is None else self.abs_error.tolist(),
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class MLFrameworkGradientAuditSuiteResult:
    """Fail-closed parity report for optional ML gradient adapters."""

    records: tuple[MLFrameworkGradientAuditRecord, ...]
    audit_passed: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("records must be non-empty")
        frameworks = [record.framework for record in self.records]
        if len(frameworks) != len(set(frameworks)):
            raise ValueError("framework records must be unique")
        if not isinstance(self.audit_passed, bool):
            raise ValueError("audit_passed must be a boolean")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @property
    def executed_frameworks(self) -> tuple[str, ...]:
        """Return frameworks whose adapters were executed."""
        return tuple(record.framework for record in self.records if record.executed)

    @property
    def unavailable_frameworks(self) -> tuple[str, ...]:
        """Return frameworks whose optional dependencies were unavailable."""
        return tuple(record.framework for record in self.records if record.status == "unavailable")

    @property
    def blocked_frameworks(self) -> tuple[str, ...]:
        """Return frameworks available but not executable without caller-owned objects."""
        return tuple(record.framework for record in self.records if record.status == "blocked")

    @property
    def failed_frameworks(self) -> tuple[str, ...]:
        """Return frameworks that executed and failed parity."""
        return tuple(record.framework for record in self.records if record.status == "failed")

    @property
    def worst_executed_error(self) -> float:
        """Return the largest error across executed ML adapters."""
        errors = [
            record.max_abs_error
            for record in self.records
            if record.executed and record.max_abs_error is not None
        ]
        return max(errors) if errors else 0.0

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready ML framework parity evidence."""
        return {
            "records": [record.to_dict() for record in self.records],
            "audit_passed": self.audit_passed,
            "executed_frameworks": list(self.executed_frameworks),
            "unavailable_frameworks": list(self.unavailable_frameworks),
            "blocked_frameworks": list(self.blocked_frameworks),
            "failed_frameworks": list(self.failed_frameworks),
            "worst_executed_error": self.worst_executed_error,
            "claim_boundary": self.claim_boundary,
        }


__all__ = [
    "DifferentiableQuantumAuditReport",
    "DifferentiableWorkflowAuditSuiteResult",
    "FiniteShotGradientAuditResult",
    "FloatArray",
    "MLFrameworkGradientAuditRecord",
    "MLFrameworkGradientAuditSuiteResult",
    "ParameterShiftAnalyticAgreement",
    "PhaseGradientBenchmarkSuiteResult",
]
