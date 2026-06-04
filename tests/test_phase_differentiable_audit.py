# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Gradient Audits
"""Tests for phase/differentiable_audit.py audit certificates."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    DifferentiableQuantumAuditReport,
    ParameterShiftAnalyticAgreement,
    PhaseGradientBenchmarkSuiteResult,
    run_known_phase_gradient_audit,
    run_parameter_shift_audit_suite,
    run_phase_gradient_benchmark_suite,
    verify_parameter_shift_analytic_gradient,
)


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
