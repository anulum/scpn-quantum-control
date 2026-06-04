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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import ParameterShiftRule
from .gradient_descent import (
    ParameterShiftTrainingCertificate,
    ParameterShiftTrainingResult,
    parameter_shift_gradient_descent,
    validate_parameter_shift_training,
)
from .param_shift import (
    GradientVerificationResult,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient,
    verify_parameter_shift_gradient,
)

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
AnalyticGradient = Callable[[FloatArray], ArrayLike]


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
    "ParameterShiftAnalyticAgreement",
    "PhaseGradientBenchmarkSuiteResult",
    "ScalarObjective",
    "run_known_phase_gradient_audit",
    "run_parameter_shift_audit_suite",
    "run_phase_gradient_benchmark_suite",
    "verify_parameter_shift_analytic_gradient",
]
