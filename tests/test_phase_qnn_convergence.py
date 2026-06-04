# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Convergence
"""Tests for phase/qnn_convergence.py bounded QNN convergence evidence."""

from __future__ import annotations

import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNConvergenceSuiteResult,
    run_parameter_shift_qnn_convergence_suite,
    summarize_parameter_shift_qnn_convergence_unsuitable_scenarios,
)


def test_qnn_convergence_suite_records_deterministic_training_evidence() -> None:
    suite = run_parameter_shift_qnn_convergence_suite()

    assert isinstance(suite, ParameterShiftQNNConvergenceSuiteResult)
    assert suite.passed
    assert suite.case_count == 3
    assert suite.passed_count == 3
    assert suite.failed_count == 0
    assert suite.total_parameter_shift_evaluations > 0
    assert suite.evidence_class == "local_deterministic_qnn_convergence"
    assert not suite.production_benchmark
    assert "not hardware" in suite.claim_boundary
    assert tuple(case.name for case in suite.cases) == (
        "single_feature_phase_flip",
        "two_feature_phase_flip",
        "three_feature_phase_flip",
    )

    for case in suite.cases:
        assert case.passed
        assert case.converged
        assert case.best_loss <= case.target_loss_tolerance
        assert case.loss_drop >= case.min_loss_drop
        assert case.accuracy is not None
        assert case.accuracy >= case.min_accuracy
        assert case.parameter_shift_evaluations > 0
        assert case.gradient_evaluations > 0
        assert case.to_dict()["passed"] is True


def test_qnn_convergence_suite_supports_case_selection() -> None:
    suite = run_parameter_shift_qnn_convergence_suite(case_names=("two_feature_phase_flip",))

    assert suite.passed
    assert suite.case_count == 1
    assert suite.cases[0].name == "two_feature_phase_flip"
    assert suite.case_by_name("two_feature_phase_flip") is suite.cases[0]


def test_qnn_convergence_suite_records_threshold_failures_without_hiding_them() -> None:
    suite = run_parameter_shift_qnn_convergence_suite(
        case_names=("single_feature_phase_flip",),
        min_loss_drop=1.0,
    )

    assert not suite.passed
    assert suite.case_count == 1
    assert suite.passed_count == 0
    assert suite.failed_count == 1
    case = suite.cases[0]
    assert not case.passed
    assert case.loss_drop < case.min_loss_drop
    assert case.to_dict()["passed"] is False


def test_qnn_convergence_suite_fails_closed_on_invalid_controls() -> None:
    with pytest.raises(ValueError, match="unknown QNN convergence case"):
        run_parameter_shift_qnn_convergence_suite(case_names=("missing",))

    with pytest.raises(ValueError, match="min_loss_drop"):
        run_parameter_shift_qnn_convergence_suite(min_loss_drop=-1.0)

    with pytest.raises(ValueError, match="min_accuracy"):
        run_parameter_shift_qnn_convergence_suite(min_accuracy=1.5)


def test_qnn_convergence_unsuitable_scenarios_are_documented() -> None:
    scenarios = summarize_parameter_shift_qnn_convergence_unsuitable_scenarios()

    assert len(scenarios) >= 4
    assert {scenario.name for scenario in scenarios} >= {
        "hardware_backend_convergence",
        "finite_shot_noisy_training",
        "arbitrary_qnn_architecture",
        "native_framework_autodiff_training",
    }
    assert all(scenario.status == "fail_closed_or_staged" for scenario in scenarios)
    assert all(scenario.mitigation for scenario in scenarios)
