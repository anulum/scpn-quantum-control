# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Gradient Audit
"""Reviewer-facing differentiable quantum gradient audit reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import ParameterShiftRule, ShotAllocationResult, StochasticGradientResult
from .coupling_learning import (
    CouplingGradientVerificationResult,
    CouplingLearningResult,
    learn_couplings_from_observations,
    verify_coupling_parameter_shift_gradient,
)
from .gradient_descent import (
    ParameterShiftTrainingCertificate,
    ParameterShiftTrainingResult,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)
from .jax_bridge import is_phase_jax_available, jax_parameter_shift_value_and_grad
from .param_shift import (
    GradientVerificationResult,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient,
    parameter_shift_gradient_with_uncertainty,
    plan_parameter_shift_shots,
    verify_parameter_shift_gradient,
)
from .pennylane_bridge import is_phase_pennylane_available
from .tensorflow_bridge import (
    is_phase_tensorflow_available,
    tensorflow_parameter_shift_value_and_grad,
)
from .torch_bridge import is_phase_torch_available, torch_parameter_shift_value_and_grad

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
AnalyticGradient = Callable[[FloatArray], ArrayLike]


class _GradientAdapterResult(Protocol):
    @property
    def value(self) -> float: ...

    @property
    def gradient(self) -> FloatArray: ...


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


def _normalise_target_value(
    target_value: float | None,
    target_value_tolerance: float,
) -> tuple[float | None, float]:
    tolerance = _as_finite_scalar("target_value_tolerance", target_value_tolerance)
    if tolerance < 0.0:
        raise ValueError("target_value_tolerance must be finite and non-negative")
    if target_value is None:
        return None, tolerance
    return _as_finite_scalar("target_value", target_value), tolerance


def _as_non_negative_shift_matrix(
    name: str,
    values: ArrayLike,
    *,
    term_count: int,
    width: int,
) -> FloatArray:
    raw = np.asarray(values)
    matrix: FloatArray
    if raw.shape == ():
        scalar = _as_finite_scalar(name, values)
        matrix = cast(FloatArray, np.full((term_count, width), scalar, dtype=np.float64))
    elif raw.ndim == 1:
        vector = _as_finite_vector(name, values)
        if vector.size != width:
            raise ValueError(f"{name} width must match parameter count")
        matrix = cast(
            FloatArray,
            np.asarray(np.tile(vector, (term_count, 1)), dtype=np.float64),
        )
    elif raw.ndim == 2:
        matrix = cast(FloatArray, np.asarray(values, dtype=np.float64))
        if matrix.shape != (term_count, width):
            raise ValueError(f"{name} shape must be (shift_terms, parameter_count)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain finite real numeric values")
        matrix = matrix.astype(np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be scalar, vector, or shift-term matrix")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return matrix


def _shifted_objective_values(
    objective: ScalarObjective,
    values: FloatArray,
    rule: ParameterShiftRule,
) -> tuple[FloatArray, FloatArray]:
    terms = rule.terms
    plus_values = np.zeros((len(terms), values.size), dtype=np.float64)
    minus_values = np.zeros_like(plus_values)
    for term_index, (shift, _coefficient) in enumerate(terms):
        for param_index in range(values.size):
            plus = values.copy()
            minus = values.copy()
            plus[param_index] += shift
            minus[param_index] -= shift
            plus_values[term_index, param_index] = _as_finite_scalar(
                "plus shifted objective",
                objective(plus),
            )
            minus_values[term_index, param_index] = _as_finite_scalar(
                "minus shifted objective",
                objective(minus),
            )
    return plus_values, minus_values


def _stochastic_input(array: FloatArray) -> FloatArray:
    if array.shape[0] == 1:
        return cast(FloatArray, array[0].copy())
    return cast(FloatArray, array.copy())


def _balanced_shots_from_allocation(allocation: ShotAllocationResult) -> FloatArray:
    shots = np.asarray(allocation.shots, dtype=float)
    if shots.ndim == 2:
        return cast(FloatArray, np.maximum(shots[0], shots[1]).astype(np.float64, copy=False))
    if shots.ndim == 3:
        return cast(FloatArray, np.max(shots, axis=1).astype(np.float64, copy=False))
    raise ValueError("shot allocation shape must contain plus/minus shot counts")


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


def _result_value_and_gradient(
    result: _GradientAdapterResult,
    *,
    framework: str,
) -> tuple[float, FloatArray]:
    value = _as_finite_scalar(f"{framework} adapter value", result.value)
    gradient = _as_finite_vector(
        f"{framework} adapter gradient",
        result.gradient,
    )
    return value, gradient


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


def verify_parameter_shift_analytic_gradient(
    objective: ScalarObjective,
    analytic_gradient: AnalyticGradient,
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    tolerance: float = 1.0e-8,
) -> ParameterShiftAnalyticAgreement:
    """Verify parameter-shift gradients against a supplied analytic gradient."""
    params = _as_finite_vector("values", values)
    tolerance_value = _as_finite_scalar("tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    parameter_shift = parameter_shift_gradient(objective, params, rule=rule)
    analytic = _as_finite_vector("analytic_gradient", analytic_gradient(params.copy()))
    if analytic.shape != params.shape:
        raise ValueError("analytic_gradient shape must match values")
    abs_error = np.abs(parameter_shift - analytic)
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    return ParameterShiftAnalyticAgreement(
        parameters=params,
        parameter_shift_gradient=parameter_shift,
        analytic_gradient=analytic,
        abs_error=abs_error,
        max_abs_error=max_abs_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        method="parameter_shift_vs_analytic_gradient",
        evaluations=2 * params.size * len(rule.terms) if rule is not None else 2 * params.size,
        claim_boundary=(
            "analytic-gradient agreement for smooth parameter-shift-compatible "
            "objectives with caller-supplied closed-form gradients"
        ),
    )


def _ml_record_unavailable(
    framework: str,
    reference_gradient: FloatArray,
    tolerance: float,
    reason: str,
) -> MLFrameworkGradientAuditRecord:
    return MLFrameworkGradientAuditRecord(
        framework=framework,
        available=False,
        executed=False,
        status="unavailable",
        reason=reason,
        value=None,
        gradient=None,
        reference_gradient=reference_gradient,
        abs_error=None,
        max_abs_error=None,
        tolerance=tolerance,
        claim_boundary=(
            "optional ML-framework dependency was absent; audit records a "
            "fail-closed unavailable status instead of pretending parity"
        ),
    )


def _ml_record_blocked(
    framework: str,
    reference_gradient: FloatArray,
    tolerance: float,
    reason: str,
) -> MLFrameworkGradientAuditRecord:
    return MLFrameworkGradientAuditRecord(
        framework=framework,
        available=True,
        executed=False,
        status="blocked",
        reason=reason,
        value=None,
        gradient=None,
        reference_gradient=reference_gradient,
        abs_error=None,
        max_abs_error=None,
        tolerance=tolerance,
        claim_boundary=(
            "framework is importable but this audit requires caller-owned "
            "objects before parity execution can be meaningful"
        ),
    )


def _ml_record_executed(
    framework: str,
    reference_gradient: FloatArray,
    tolerance: float,
    value: float,
    gradient: FloatArray,
) -> MLFrameworkGradientAuditRecord:
    if gradient.shape != reference_gradient.shape:
        raise ValueError(f"{framework} adapter gradient shape must match reference")
    abs_error = np.abs(gradient - reference_gradient)
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    passed = max_abs_error <= tolerance
    return MLFrameworkGradientAuditRecord(
        framework=framework,
        available=True,
        executed=True,
        status="passed" if passed else "failed",
        reason="adapter gradient matched native parameter-shift reference"
        if passed
        else "adapter gradient differed from native parameter-shift reference",
        value=value,
        gradient=gradient,
        reference_gradient=reference_gradient,
        abs_error=abs_error,
        max_abs_error=max_abs_error,
        tolerance=tolerance,
        claim_boundary=(
            "adapter parity for a smooth local parameter-shift objective; not "
            "a full training-loop, accelerator, or graph-compilation certificate"
        ),
    )


def run_ml_framework_gradient_audit(
    objective: ScalarObjective | None = None,
    initial_values: ArrayLike | None = None,
    *,
    rule: ParameterShiftRule | None = None,
    tolerance: float = 1.0e-8,
    pennylane_gradient: AnalyticGradient | None = None,
) -> MLFrameworkGradientAuditSuiteResult:
    """Run fail-closed parity checks for optional ML gradient adapters."""
    values = (
        np.array([0.3, -0.2], dtype=np.float64)
        if initial_values is None
        else _as_finite_vector("initial_values", initial_values)
    )
    tolerance_value = _as_finite_scalar("tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    def default_objective(theta: FloatArray) -> float:
        return float(np.mean(1.0 - np.cos(theta)))

    selected_objective = default_objective if objective is None else objective
    reference_gradient = parameter_shift_gradient(selected_objective, values, rule=rule)
    records: list[MLFrameworkGradientAuditRecord] = []

    if is_phase_jax_available():
        value, gradient = _result_value_and_gradient(
            jax_parameter_shift_value_and_grad(selected_objective, values, rule=rule),
            framework="jax",
        )
        records.append(
            _ml_record_executed("jax", reference_gradient, tolerance_value, value, gradient)
        )
    else:
        records.append(
            _ml_record_unavailable(
                "jax",
                reference_gradient,
                tolerance_value,
                "JAX is not importable in this environment",
            )
        )

    if is_phase_torch_available():
        value, gradient = _result_value_and_gradient(
            torch_parameter_shift_value_and_grad(selected_objective, values, rule=rule),
            framework="torch",
        )
        records.append(
            _ml_record_executed("torch", reference_gradient, tolerance_value, value, gradient)
        )
    else:
        records.append(
            _ml_record_unavailable(
                "torch",
                reference_gradient,
                tolerance_value,
                "PyTorch is not importable in this environment",
            )
        )

    if is_phase_tensorflow_available():
        value, gradient = _result_value_and_gradient(
            tensorflow_parameter_shift_value_and_grad(selected_objective, values, rule=rule),
            framework="tensorflow",
        )
        records.append(
            _ml_record_executed("tensorflow", reference_gradient, tolerance_value, value, gradient)
        )
    else:
        records.append(
            _ml_record_unavailable(
                "tensorflow",
                reference_gradient,
                tolerance_value,
                "TensorFlow is not importable in this environment",
            )
        )

    if not is_phase_pennylane_available():
        records.append(
            _ml_record_unavailable(
                "pennylane",
                reference_gradient,
                tolerance_value,
                "PennyLane is not importable in this environment",
            )
        )
    elif pennylane_gradient is None:
        records.append(
            _ml_record_blocked(
                "pennylane",
                reference_gradient,
                tolerance_value,
                "PennyLane parity requires a caller-supplied QNode gradient callable",
            )
        )
    else:
        gradient = _as_finite_vector("pennylane_gradient", pennylane_gradient(values.copy()))
        value = _as_finite_scalar("pennylane objective value", selected_objective(values.copy()))
        records.append(
            _ml_record_executed("pennylane", reference_gradient, tolerance_value, value, gradient)
        )

    audit_passed = all(record.status != "failed" for record in records)
    return MLFrameworkGradientAuditSuiteResult(
        records=tuple(records),
        audit_passed=audit_passed,
        claim_boundary=(
            "optional ML-framework parity audit for smooth local "
            "parameter-shift objectives; unavailable dependencies are recorded "
            "fail-closed, and this is not a full accelerator, autograd graph, "
            "or framework-native training-loop certificate"
        ),
    )


def run_parameter_shift_audit_suite(
    objective: ScalarObjective,
    analytic_gradient: AnalyticGradient,
    initial_values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    finite_difference_step: float = 1.0e-6,
    finite_difference_tolerance: float = 1.0e-5,
    analytic_tolerance: float = 1.0e-8,
    learning_rate: float = 0.35,
    max_steps: int = 80,
    gradient_tolerance: float = 1.0e-8,
    target_value: float | None = 0.0,
    target_value_tolerance: float = 1.0e-8,
    min_loss_decrease: float | None = None,
) -> DifferentiableQuantumAuditReport:
    """Run finite-difference, analytic, and convergence checks as one report."""
    values = _as_finite_vector("initial_values", initial_values)
    target, target_tolerance = _normalise_target_value(target_value, target_value_tolerance)
    finite_difference = verify_parameter_shift_gradient(
        objective,
        values,
        rule=rule,
        finite_difference_step=finite_difference_step,
        absolute_tolerance=finite_difference_tolerance,
    )
    analytic = verify_parameter_shift_analytic_gradient(
        objective,
        analytic_gradient,
        values,
        rule=rule,
        tolerance=analytic_tolerance,
    )
    training = parameter_shift_gradient_descent(
        objective,
        values,
        rule=rule,
        learning_rate=learning_rate,
        max_steps=max_steps,
        gradient_tolerance=gradient_tolerance,
    )
    training_certificate = validate_parameter_shift_training(
        training,
        gradient_tolerance=gradient_tolerance,
        target_value=target,
        target_value_tolerance=target_tolerance,
        min_decrease=min_loss_decrease,
    )
    target_passed = (
        True if target is None else abs(training.best_value - target) <= target_tolerance
    )
    training_passed = training.best_value <= training.initial_value and target_passed
    return DifferentiableQuantumAuditReport(
        finite_difference=finite_difference,
        analytic=analytic,
        training=training,
        training_certificate=training_certificate,
        passed=finite_difference.passed and analytic.passed and training_passed,
        task="parameter_shift_gradient_audit_suite",
        claim_boundary=(
            "deterministic local parameter-shift audit for smooth objectives; "
            "not a proof for discontinuous objectives, stochastic hardware "
            "shots, arbitrary regressors, or undeclared generator spectra"
        ),
    )


def run_known_phase_gradient_audit(
    initial_values: ArrayLike | None = None,
    *,
    learning_rate: float = 0.35,
    max_steps: int = 80,
    finite_difference_step: float = 1.0e-6,
    finite_difference_tolerance: float = 1.0e-5,
    analytic_tolerance: float = 1.0e-10,
    target_value_tolerance: float = 1.0e-8,
) -> DifferentiableQuantumAuditReport:
    """Run the built-in smooth phase-rotation audit benchmark.

    The benchmark objective is mean(1 - cos(theta_i)), whose exact gradient is
    mean-scaled sin(theta_i). It models the single-frequency expectation losses
    used by local parameter-shift phase-gradient diagnostics.
    """
    values = (
        np.array([0.8, -0.5, 0.3], dtype=np.float64)
        if initial_values is None
        else _as_finite_vector("initial_values", initial_values)
    )

    def objective(params: FloatArray) -> float:
        return float(np.mean(1.0 - np.cos(params)))

    def analytic_gradient(params: FloatArray) -> FloatArray:
        return (np.sin(params) / params.size).astype(np.float64, copy=False)

    return run_parameter_shift_audit_suite(
        objective,
        analytic_gradient,
        values,
        learning_rate=learning_rate,
        max_steps=max_steps,
        finite_difference_step=finite_difference_step,
        finite_difference_tolerance=finite_difference_tolerance,
        analytic_tolerance=analytic_tolerance,
        gradient_tolerance=1.0e-8,
        target_value=0.0,
        target_value_tolerance=target_value_tolerance,
        min_loss_decrease=0.1,
    )


def run_finite_shot_gradient_uncertainty_audit(
    objective: ScalarObjective,
    initial_values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    plus_variances: ArrayLike = 0.04,
    minus_variances: ArrayLike = 0.04,
    target_standard_error: float = 0.02,
    min_shots: int = 64,
    max_shots_per_evaluation: int | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
) -> FiniteShotGradientAuditResult:
    """Audit finite-shot parameter-shift uncertainty propagation.

    The shifted expectation values are evaluated deterministically, while the
    supplied variances and planned shots define the stochastic uncertainty
    envelope. This certifies propagation and containment semantics; it does not
    claim live hardware sampling correctness.
    """
    values = _as_finite_vector("initial_values", initial_values)
    shift_rule = rule or ParameterShiftRule()
    term_count = len(shift_rule.terms)
    target = _as_finite_scalar("target_standard_error", target_standard_error)
    if target <= 0.0:
        raise ValueError("target_standard_error must be finite and positive")
    plus_var = _as_non_negative_shift_matrix(
        "plus_variances",
        plus_variances,
        term_count=term_count,
        width=values.size,
    )
    minus_var = _as_non_negative_shift_matrix(
        "minus_variances",
        minus_variances,
        term_count=term_count,
        width=values.size,
    )
    plus_values, minus_values = _shifted_objective_values(objective, values, shift_rule)
    shot_allocation = plan_parameter_shift_shots(
        _stochastic_input(plus_var),
        _stochastic_input(minus_var),
        target_standard_error=target,
        rule=shift_rule,
        min_shots=min_shots,
        max_shots_per_evaluation=max_shots_per_evaluation,
    )
    balanced_shots = _balanced_shots_from_allocation(shot_allocation)
    stochastic = parameter_shift_gradient_with_uncertainty(
        _stochastic_input(plus_values),
        _stochastic_input(minus_values),
        _stochastic_input(plus_var),
        _stochastic_input(minus_var),
        shots=balanced_shots,
        rule=shift_rule,
        confidence_level=confidence_level,
        confidence_z=confidence_z,
    )
    deterministic = parameter_shift_gradient(objective, values, rule=shift_rule)
    abs_error = np.abs(stochastic.gradient - deterministic)
    within_confidence = tuple(
        bool(error <= radius + 1.0e-15)
        for error, radius in zip(abs_error, stochastic.confidence_radius, strict=True)
    )
    max_standard_error = (
        float(np.max(stochastic.standard_error)) if stochastic.standard_error.size else 0.0
    )
    return FiniteShotGradientAuditResult(
        deterministic_gradient=deterministic,
        stochastic=stochastic,
        shot_allocation=shot_allocation,
        abs_error=abs_error,
        within_confidence=within_confidence,
        target_standard_error=target,
        max_abs_error=float(np.max(abs_error)) if abs_error.size else 0.0,
        max_confidence_radius=float(np.max(stochastic.confidence_radius))
        if stochastic.confidence_radius.size
        else 0.0,
        max_standard_error=max_standard_error,
        executed_total_shots=int(2 * np.sum(balanced_shots)),
        passed=all(within_confidence) and max_standard_error <= target + 1.0e-12,
        method="finite_shot_parameter_shift_uncertainty_audit",
        claim_boundary=(
            "finite-shot uncertainty propagation for deterministic shifted "
            "expectation values with declared variances and shot budgets; not "
            "a live hardware sampling, detector-drift, or queue-calibration certificate"
        ),
    )


def run_differentiable_workflow_audit_suite(
    *,
    finite_shot_target_standard_error: float = 0.02,
    coupling_learning_rate: float = 0.35,
    coupling_max_steps: int = 80,
    gradient_tolerance: float = 1.0e-7,
) -> DifferentiableWorkflowAuditSuiteResult:
    """Run the built-in cross-workflow differentiable-programming audit suite."""

    def phase_objective(theta: FloatArray) -> float:
        return float(np.mean(1.0 - np.cos(theta)))

    phase_benchmarks = run_phase_gradient_benchmark_suite()
    finite_shot = run_finite_shot_gradient_uncertainty_audit(
        phase_objective,
        np.array([0.7, -0.4, 0.2], dtype=np.float64),
        target_standard_error=finite_shot_target_standard_error,
        plus_variances=np.array([0.04, 0.03, 0.02], dtype=np.float64),
        minus_variances=np.array([0.04, 0.03, 0.02], dtype=np.float64),
    )

    def observations(couplings: FloatArray) -> FloatArray:
        return np.array([np.sin(couplings[0, 1])], dtype=np.float64)

    coupling_rule = multi_frequency_parameter_shift_rule([2.0])
    initial_couplings = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64)
    coupling_gradient = verify_coupling_parameter_shift_gradient(
        observations,
        np.array([0.0], dtype=np.float64),
        initial_couplings,
        rule=coupling_rule,
        finite_difference_step=1.0e-6,
        tolerance=1.0e-5,
    )
    coupling_learning = learn_couplings_from_observations(
        observations,
        np.array([0.0], dtype=np.float64),
        initial_couplings,
        rule=coupling_rule,
        learning_rate=coupling_learning_rate,
        max_steps=coupling_max_steps,
        gradient_tolerance=gradient_tolerance,
        min_loss_decrease=0.1,
    )
    return DifferentiableWorkflowAuditSuiteResult(
        phase_benchmarks=phase_benchmarks,
        finite_shot=finite_shot,
        coupling_gradient=coupling_gradient,
        coupling_learning=coupling_learning,
        workflow_names=(
            "phase_gradient_conformance",
            "finite_shot_uncertainty_containment",
            "coupling_gradient_verification",
            "coupling_learning_training",
        ),
        unsupported_scenarios=(
            "arbitrary Python program reverse-mode AD",
            "live hardware sampling and provider queue calibration",
            "dynamic circuit topology with unstable parameter identity",
            "classical regressors without declared generator spectra",
            "mutation-heavy or aliasing program IR semantics",
        ),
        passed=(
            phase_benchmarks.passed
            and finite_shot.passed
            and coupling_gradient.passed
            and coupling_learning.certificate.monotone_accepted_values
            and coupling_learning.best_loss <= 1.0e-8
        ),
        claim_boundary=(
            "cross-workflow deterministic differentiable quantum audit for "
            "supported phase, finite-shot uncertainty, and coupling-learning "
            "paths; not a full arbitrary-program AD, live hardware, or "
            "complete ML-framework integration certificate"
        ),
    )


def run_phase_gradient_benchmark_suite(
    *,
    learning_rate: float = 0.35,
    max_steps: int = 100,
    finite_difference_step: float = 1.0e-6,
    finite_difference_tolerance: float = 1.0e-5,
    analytic_tolerance: float = 1.0e-9,
    target_value_tolerance: float = 1.0e-8,
) -> PhaseGradientBenchmarkSuiteResult:
    """Run built-in deterministic phase-gradient conformance benchmarks."""

    def single_frequency_objective(params: FloatArray) -> float:
        return float(np.mean(1.0 - np.cos(params)))

    def single_frequency_gradient(params: FloatArray) -> FloatArray:
        return (np.sin(params) / params.size).astype(np.float64, copy=False)

    def multi_frequency_objective(params: FloatArray) -> float:
        return float(np.mean((1.0 - np.cos(params)) + 0.05 * (1.0 - np.cos(2.0 * params))))

    def multi_frequency_gradient(params: FloatArray) -> FloatArray:
        return ((np.sin(params) + 0.1 * np.sin(2.0 * params)) / params.size).astype(
            np.float64,
            copy=False,
        )

    def coupled_phase_objective(params: FloatArray) -> float:
        return float(
            0.5
            * (
                (1.0 - np.cos(params[0]))
                + (1.0 - np.cos(params[1]))
                + 0.25 * (1.0 - np.cos(params[0] - params[1]))
            )
        )

    def coupled_phase_gradient(params: FloatArray) -> FloatArray:
        coupling = np.sin(params[0] - params[1])
        return np.array(
            [
                0.5 * (np.sin(params[0]) + 0.25 * coupling),
                0.5 * (np.sin(params[1]) - 0.25 * coupling),
            ],
            dtype=np.float64,
        )

    report_single = run_parameter_shift_audit_suite(
        single_frequency_objective,
        single_frequency_gradient,
        np.array([0.8, -0.5, 0.3], dtype=np.float64),
        learning_rate=learning_rate,
        max_steps=max_steps,
        finite_difference_step=finite_difference_step,
        finite_difference_tolerance=finite_difference_tolerance,
        analytic_tolerance=analytic_tolerance,
        target_value_tolerance=target_value_tolerance,
        min_loss_decrease=0.1,
    )
    report_multi = run_parameter_shift_audit_suite(
        multi_frequency_objective,
        multi_frequency_gradient,
        np.array([0.45, -0.35, 0.25], dtype=np.float64),
        rule=multi_frequency_parameter_shift_rule([1.0, 2.0]),
        learning_rate=learning_rate,
        max_steps=max_steps,
        finite_difference_step=finite_difference_step,
        finite_difference_tolerance=finite_difference_tolerance,
        analytic_tolerance=analytic_tolerance,
        target_value_tolerance=target_value_tolerance,
        min_loss_decrease=0.1,
    )
    report_coupled = run_parameter_shift_audit_suite(
        coupled_phase_objective,
        coupled_phase_gradient,
        np.array([0.55, -0.35], dtype=np.float64),
        learning_rate=learning_rate,
        max_steps=max_steps,
        finite_difference_step=finite_difference_step,
        finite_difference_tolerance=finite_difference_tolerance,
        analytic_tolerance=analytic_tolerance,
        target_value_tolerance=target_value_tolerance,
        min_loss_decrease=0.1,
    )
    reports = (report_single, report_multi, report_coupled)
    return PhaseGradientBenchmarkSuiteResult(
        benchmark_names=(
            "single_frequency_phase_rotation",
            "multi_frequency_phase_rotation",
            "coupled_pair_phase_rotation",
        ),
        reports=reports,
        unsupported_scenarios=(
            "discontinuous objective surfaces",
            "shot-noisy hardware gradients without uncertainty certificates",
            "arbitrary classical regressors without declared generator spectra",
            "objectives with non-finite values or shape-drifting gradients",
        ),
        passed=all(report.passed for report in reports),
        claim_boundary=(
            "deterministic local phase-gradient conformance for smooth "
            "parameter-shift-compatible objectives; not a blanket hardware, "
            "stochastic, or arbitrary-program AD certificate"
        ),
    )


__all__ = [
    "AnalyticGradient",
    "DifferentiableQuantumAuditReport",
    "DifferentiableWorkflowAuditSuiteResult",
    "FiniteShotGradientAuditResult",
    "MLFrameworkGradientAuditRecord",
    "MLFrameworkGradientAuditSuiteResult",
    "ParameterShiftAnalyticAgreement",
    "PhaseGradientBenchmarkSuiteResult",
    "ScalarObjective",
    "run_differentiable_workflow_audit_suite",
    "run_finite_shot_gradient_uncertainty_audit",
    "run_known_phase_gradient_audit",
    "run_ml_framework_gradient_audit",
    "run_parameter_shift_audit_suite",
    "run_phase_gradient_benchmark_suite",
    "verify_parameter_shift_analytic_gradient",
]
