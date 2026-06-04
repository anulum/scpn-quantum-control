# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Gradient Audits
"""Tests for phase/differentiable_audit.py audit certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import scpn_quantum_control.phase.differentiable_audit as audit_module
from scpn_quantum_control.phase import (
    DifferentiableQuantumAuditReport,
    DifferentiableWorkflowAuditSuiteResult,
    FiniteShotGradientAuditResult,
    MLFrameworkGradientAuditSuiteResult,
    ParameterShiftAnalyticAgreement,
    PhaseGradientBenchmarkSuiteResult,
    run_differentiable_workflow_audit_suite,
    run_finite_shot_gradient_uncertainty_audit,
    run_known_phase_gradient_audit,
    run_ml_framework_gradient_audit,
    run_parameter_shift_audit_suite,
    run_phase_gradient_benchmark_suite,
    verify_parameter_shift_analytic_gradient,
)


@dataclass(frozen=True)
class _FakeAdapterResult:
    value: float
    gradient: np.ndarray


def test_run_known_phase_gradient_audit_passes_all_evidence_checks() -> None:
    report = run_known_phase_gradient_audit(np.array([0.8, -0.5, 0.3], dtype=float))

    assert isinstance(report, DifferentiableQuantumAuditReport)
    assert report.passed
    assert report.finite_difference.passed
    assert report.analytic.passed
    assert report.best_value < 1e-8
    assert report.max_gradient_error < 1e-5
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["task"] == "parameter_shift_gradient_audit_suite"
    assert "finite_difference" in payload
    assert "analytic" in payload
    assert "training_certificate" in payload


def test_verify_parameter_shift_analytic_gradient_matches_closed_form() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))

    def analytic(values: np.ndarray) -> np.ndarray:
        return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)

    certificate = verify_parameter_shift_analytic_gradient(
        objective,
        analytic,
        np.array([0.2, -0.4], dtype=float),
    )

    assert isinstance(certificate, ParameterShiftAnalyticAgreement)
    assert certificate.passed
    assert certificate.max_abs_error < 1e-12
    assert certificate.evaluations == 4
    assert certificate.to_dict()["method"] == "parameter_shift_vs_analytic_gradient"


def test_run_finite_shot_gradient_uncertainty_audit_contains_errors() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.mean(1.0 - np.cos(values)))

    certificate = run_finite_shot_gradient_uncertainty_audit(
        objective,
        np.array([0.7, -0.4, 0.2], dtype=float),
        target_standard_error=0.02,
        plus_variances=np.array([0.04, 0.03, 0.02], dtype=float),
        minus_variances=np.array([0.04, 0.03, 0.02], dtype=float),
    )

    assert isinstance(certificate, FiniteShotGradientAuditResult)
    assert certificate.passed
    assert all(certificate.within_confidence)
    assert certificate.max_abs_error <= certificate.max_confidence_radius
    assert certificate.max_standard_error <= certificate.target_standard_error + 1e-12
    assert certificate.executed_total_shots >= certificate.shot_allocation.total_shots
    payload = certificate.to_dict()
    assert payload["passed"] is True
    assert payload["method"] == "finite_shot_parameter_shift_uncertainty_audit"


def test_run_finite_shot_gradient_uncertainty_audit_rejects_negative_variance() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.mean(1.0 - np.cos(values)))

    with pytest.raises(ValueError, match="plus_variances"):
        run_finite_shot_gradient_uncertainty_audit(
            objective,
            np.array([0.7, -0.4], dtype=float),
            plus_variances=np.array([0.04, -0.03], dtype=float),
        )


def test_run_differentiable_workflow_audit_suite_passes_supported_lanes() -> None:
    suite = run_differentiable_workflow_audit_suite()

    assert isinstance(suite, DifferentiableWorkflowAuditSuiteResult)
    assert suite.passed
    assert suite.workflow_names == (
        "phase_gradient_conformance",
        "finite_shot_uncertainty_containment",
        "coupling_gradient_verification",
        "coupling_learning_training",
    )
    assert suite.phase_benchmarks.passed
    assert suite.finite_shot.passed
    assert suite.coupling_gradient.passed
    assert suite.coupling_learning.best_loss < 1e-8
    assert suite.worst_gradient_error < 1e-5
    payload = suite.to_dict()
    assert payload["passed"] is True
    assert "arbitrary Python program reverse-mode AD" in payload["unsupported_scenarios"]
    assert "coupling_learning" in payload


def test_run_ml_framework_gradient_audit_records_unavailable_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_module, "is_phase_jax_available", lambda: False)
    monkeypatch.setattr(audit_module, "is_phase_torch_available", lambda: False)
    monkeypatch.setattr(audit_module, "is_phase_tensorflow_available", lambda: False)
    monkeypatch.setattr(audit_module, "is_phase_pennylane_available", lambda: False)

    suite = run_ml_framework_gradient_audit()

    assert isinstance(suite, MLFrameworkGradientAuditSuiteResult)
    assert suite.audit_passed
    assert suite.executed_frameworks == ()
    assert suite.unavailable_frameworks == ("jax", "torch", "tensorflow", "pennylane")
    assert suite.failed_frameworks == ()
    payload = suite.to_dict()
    assert payload["audit_passed"] is True
    assert payload["unavailable_frameworks"] == ["jax", "torch", "tensorflow", "pennylane"]


def test_run_ml_framework_gradient_audit_executes_available_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_module, "is_phase_jax_available", lambda: True)
    monkeypatch.setattr(audit_module, "is_phase_torch_available", lambda: False)
    monkeypatch.setattr(audit_module, "is_phase_tensorflow_available", lambda: False)
    monkeypatch.setattr(audit_module, "is_phase_pennylane_available", lambda: False)

    def fake_jax_adapter(
        objective: audit_module.ScalarObjective,
        values: np.ndarray,
        **_kwargs: object,
    ) -> _FakeAdapterResult:
        params = np.asarray(values, dtype=float)
        return _FakeAdapterResult(
            value=float(objective(params)),
            gradient=(np.sin(params) / params.size).astype(float),
        )

    monkeypatch.setattr(audit_module, "jax_parameter_shift_value_and_grad", fake_jax_adapter)

    suite = run_ml_framework_gradient_audit()

    assert suite.audit_passed
    assert suite.executed_frameworks == ("jax",)
    assert suite.failed_frameworks == ()
    assert suite.records[0].status == "passed"
    assert suite.records[0].max_abs_error is not None
    assert suite.records[0].max_abs_error < 1e-12


def test_run_phase_gradient_benchmark_suite_passes_all_cases() -> None:
    suite = run_phase_gradient_benchmark_suite()

    assert isinstance(suite, PhaseGradientBenchmarkSuiteResult)
    assert suite.passed
    assert suite.benchmark_names == (
        "single_frequency_phase_rotation",
        "multi_frequency_phase_rotation",
        "coupled_pair_phase_rotation",
    )
    assert len(suite.reports) == 3
    assert suite.worst_gradient_error < 1e-5
    assert all(value < 1e-8 for value in suite.best_values)
    payload = suite.to_dict()
    assert payload["passed"] is True
    assert payload["benchmark_names"] == list(suite.benchmark_names)
    assert len(payload["reports"]) == 3
    assert (
        "shot-noisy hardware gradients without uncertainty certificates"
        in payload["unsupported_scenarios"]
    )


def test_run_parameter_shift_audit_suite_rejects_bad_analytic_shape() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.mean(1.0 - np.cos(values)))

    def bad_analytic(values: np.ndarray) -> np.ndarray:
        return np.array([values[0], values[0]], dtype=float)

    with pytest.raises(ValueError, match="analytic_gradient shape"):
        run_parameter_shift_audit_suite(
            objective,
            bad_analytic,
            np.array([0.4], dtype=float),
            max_steps=4,
        )
