# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Optimizer Benchmark
"""Tests for phase/qnn_optimizer_benchmark.py bounded QNN optimizer evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNOptimizerBenchmarkSuiteResult,
    run_parameter_shift_qnn_optimizer_benchmark_suite,
)


def test_qnn_optimizer_benchmark_records_non_isolated_functional_evidence() -> None:
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite()

    assert isinstance(suite, ParameterShiftQNNOptimizerBenchmarkSuiteResult)
    assert suite.passed
    assert suite.case_count == 2
    assert suite.parameter_shift_not_worse_count == 2
    assert suite.derivative_free_win_count == 0
    assert suite.evidence_class == "functional_non_isolated"
    assert not suite.production_benchmark
    assert "not a throughput benchmark" in suite.claim_boundary

    for case in suite.cases:
        assert case.passed
        assert case.parameter_shift_best_loss <= case.finite_difference_best_loss + 1e-6
        assert case.parameter_shift_best_loss <= case.derivative_free_best_loss
        assert case.parameter_shift_evaluations > 0
        assert case.finite_difference_evaluations > 0
        assert case.derivative_free_evaluations > 0
        assert case.evidence_class == "functional_non_isolated"
        assert case.to_dict()["passed"] is True

    payload = suite.to_dict()
    assert payload["passed"] is True
    assert payload["case_count"] == 2
    assert payload["production_benchmark"] is False


def test_qnn_optimizer_benchmark_supports_external_case_selection() -> None:
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite(
        case_names=("phase_separable_single_feature",)
    )

    assert suite.passed
    assert suite.case_count == 1
    assert suite.cases[0].name == "phase_separable_single_feature"
    assert suite.cases[0].parameter_shift_accuracy == 1.0
    assert suite.case_by_name("phase_separable_single_feature") is suite.cases[0]


def test_qnn_optimizer_benchmark_rejects_unknown_case_names() -> None:
    with pytest.raises(ValueError, match="unknown QNN optimizer benchmark case"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(case_names=("missing",))


def test_qnn_optimizer_benchmark_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(learning_rate=0.0)

    with pytest.raises(ValueError, match="max_steps"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(max_steps=0)

    with pytest.raises(ValueError, match="tolerance"):
        run_parameter_shift_qnn_optimizer_benchmark_suite(tolerance=-1.0)


def test_qnn_optimizer_benchmark_derivative_free_baseline_uses_declared_candidates() -> None:
    candidates = (np.array([0.8], dtype=float), np.array([0.9], dtype=float))
    suite = run_parameter_shift_qnn_optimizer_benchmark_suite(
        case_names=("phase_separable_single_feature",),
        derivative_free_candidates={"phase_separable_single_feature": candidates},
    )

    case = suite.cases[0]
    assert case.derivative_free_evaluations == len(candidates)
    assert case.parameter_shift_best_loss < case.derivative_free_best_loss
