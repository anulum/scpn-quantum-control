# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Finite-Shot Evidence
"""Tests for phase/qnn_finite_shot.py seeded finite-shot QNN evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNFiniteShotConvergenceSuiteResult,
    ParameterShiftQNNFiniteShotGradientResult,
    estimate_parameter_shift_qnn_finite_shot_gradient,
    parameter_shift_qnn_classifier_gradient,
    run_parameter_shift_qnn_finite_shot_convergence_suite,
    summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios,
)


def test_qnn_finite_shot_gradient_records_seeded_uncertainty() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)
    deterministic = parameter_shift_qnn_classifier_gradient(features, labels, params)

    result = estimate_parameter_shift_qnn_finite_shot_gradient(
        features,
        labels,
        params,
        shots_per_sample=8192,
        seed=17,
        confidence_z=3.0,
    )

    assert isinstance(result, ParameterShiftQNNFiniteShotGradientResult)
    assert result.passed
    assert result.seed == 17
    assert result.shots_per_sample == 8192
    assert result.probe_count == 4
    assert result.total_shots == 4 * 2 * 8192
    assert result.evidence_class == "seeded_finite_shot_qnn_gradient"
    assert not result.hardware_execution
    assert "not hardware" in result.claim_boundary
    np.testing.assert_allclose(result.deterministic_gradient, deterministic)
    assert result.max_abs_error <= result.max_confidence_radius
    assert result.max_standard_error >= 0.0
    assert result.to_dict()["passed"] is True


def test_qnn_finite_shot_gradient_replays_with_same_seed() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)

    first = estimate_parameter_shift_qnn_finite_shot_gradient(
        features,
        labels,
        params,
        shots_per_sample=4096,
        seed=91,
    )
    second = estimate_parameter_shift_qnn_finite_shot_gradient(
        features,
        labels,
        params,
        shots_per_sample=4096,
        seed=91,
    )

    np.testing.assert_allclose(first.finite_shot_gradient, second.finite_shot_gradient)
    assert first.to_dict() == second.to_dict()


def test_qnn_finite_shot_convergence_suite_records_seeded_noisy_training() -> None:
    suite = run_parameter_shift_qnn_finite_shot_convergence_suite()

    assert isinstance(suite, ParameterShiftQNNFiniteShotConvergenceSuiteResult)
    assert suite.passed
    assert suite.case_count == 2
    assert suite.passed_count == 2
    assert suite.failed_count == 0
    assert suite.total_shots > 0
    assert suite.total_parameter_shift_evaluations > 0
    assert suite.evidence_class == "seeded_finite_shot_qnn_convergence"
    assert not suite.hardware_execution
    assert not suite.production_benchmark
    assert "not hardware" in suite.claim_boundary
    assert tuple(case.name for case in suite.cases) == (
        "single_feature_finite_shot_phase_flip",
        "two_feature_finite_shot_phase_flip",
    )

    for case in suite.cases:
        assert case.passed
        assert case.best_loss <= case.target_loss_tolerance
        assert case.loss_drop >= case.min_loss_drop
        assert case.max_gradient_standard_error >= 0.0
        assert case.total_shots > 0
        assert case.to_dict()["passed"] is True


def test_qnn_finite_shot_convergence_records_threshold_failures() -> None:
    suite = run_parameter_shift_qnn_finite_shot_convergence_suite(
        case_names=("single_feature_finite_shot_phase_flip",),
        min_loss_drop=1.0,
    )

    assert not suite.passed
    assert suite.passed_count == 0
    assert suite.failed_count == 1
    assert suite.cases[0].loss_drop < suite.cases[0].min_loss_drop


def test_qnn_finite_shot_fails_closed_on_invalid_controls() -> None:
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)

    with pytest.raises(ValueError, match="shots_per_sample"):
        estimate_parameter_shift_qnn_finite_shot_gradient(
            features,
            labels,
            params,
            shots_per_sample=0,
        )

    with pytest.raises(ValueError, match="confidence_z"):
        estimate_parameter_shift_qnn_finite_shot_gradient(
            features,
            labels,
            params,
            confidence_z=0.0,
        )

    with pytest.raises(ValueError, match="unknown QNN finite-shot convergence case"):
        run_parameter_shift_qnn_finite_shot_convergence_suite(case_names=("missing",))


def test_qnn_finite_shot_unsuitable_scenarios_are_documented() -> None:
    scenarios = summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios()

    assert len(scenarios) >= 4
    assert {scenario.name for scenario in scenarios} >= {
        "hardware_provider_jobs",
        "unseeded_stochastic_training",
        "low_shot_gradient_promotion",
        "arbitrary_qnn_architecture",
    }
    assert all(scenario.status == "fail_closed_or_staged" for scenario in scenarios)
    assert all(scenario.mitigation for scenario in scenarios)
