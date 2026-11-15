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
    """A seeded finite-shot gradient records its probes and stays within its confidence radius."""
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
    """The same seed replays the same finite-shot gradient and probe records."""
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
    """The default suite trains both seeded cases to tolerance without hardware claims."""
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
    """An unreachable required loss drop is recorded as a failed case."""
    suite = run_parameter_shift_qnn_finite_shot_convergence_suite(
        case_names=("single_feature_finite_shot_phase_flip",),
        min_loss_drop=1.0,
    )

    assert not suite.passed
    assert suite.passed_count == 0
    assert suite.failed_count == 1
    assert suite.cases[0].loss_drop < suite.cases[0].min_loss_drop


def test_qnn_finite_shot_fails_closed_on_invalid_controls() -> None:
    """Zero shots, a zero z multiplier and an unknown case name are refused."""
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
    """Every unsupported scenario is named, staged and has a mitigation."""
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


_FEATURES = np.array([[0.0], [np.pi]], dtype=float)
_LABELS = np.array([0.0, 1.0], dtype=float)
_PARAMS = np.array([0.45], dtype=float)


@pytest.mark.parametrize(
    ("features", "labels", "params", "message"),
    [
        pytest.param(
            np.array([0.0, np.pi]), _LABELS, _PARAMS, "two-dimensional", id="1d-features"
        ),
        pytest.param(
            np.empty((0, 1)), _LABELS[:0], _PARAMS, "at least one sample", id="no-samples"
        ),
        pytest.param(np.empty((2, 0)), _LABELS, _PARAMS[:0], "and one feature", id="no-features"),
        pytest.param(
            np.array([[0.0], [np.inf]]),
            _LABELS,
            _PARAMS,
            "features must contain only finite",
            id="inf",
        ),
        pytest.param(
            _FEATURES, np.array([0.0]), _PARAMS, "labels must be a one-dimensional", id="short"
        ),
        pytest.param(
            _FEATURES,
            np.array([0.0, np.nan]),
            _PARAMS,
            "labels must contain only finite",
            id="nan",
        ),
        pytest.param(
            _FEATURES, np.array([0.0, 1.5]), _PARAMS, r"labels must lie in \[0, 1\]", id="range"
        ),
        pytest.param(
            _FEATURES, _LABELS, np.array([0.1, 0.2]), "params must be a one-dimensional", id="wide"
        ),
        pytest.param(
            _FEATURES,
            _LABELS,
            np.array([np.nan]),
            "params must contain only finite",
            id="nan-params",
        ),
    ],
)
def test_qnn_finite_shot_gradient_refuses_malformed_data(
    features: np.ndarray, labels: np.ndarray, params: np.ndarray, message: str
) -> None:
    """Features, labels and parameters must be finite, shaped alike and labels within [0, 1]."""
    with pytest.raises(ValueError, match=message):
        estimate_parameter_shift_qnn_finite_shot_gradient(features, labels, params)


def test_qnn_finite_shot_gradient_accepts_a_label_column() -> None:
    """A single-column label matrix is the same as the label vector, seed for seed."""
    column = estimate_parameter_shift_qnn_finite_shot_gradient(
        _FEATURES, _LABELS[:, None], _PARAMS
    )
    vector = estimate_parameter_shift_qnn_finite_shot_gradient(_FEATURES, _LABELS, _PARAMS)

    np.testing.assert_array_equal(column.finite_shot_gradient, vector.finite_shot_gradient)


@pytest.mark.parametrize("min_loss_drop", [-0.1, float("nan")])
def test_qnn_finite_shot_suite_refuses_an_invalid_loss_drop(min_loss_drop: float) -> None:
    """The required loss drop must be finite and non-negative."""
    with pytest.raises(ValueError, match="min_loss_drop must be finite and non-negative"):
        run_parameter_shift_qnn_finite_shot_convergence_suite(min_loss_drop=min_loss_drop)


def test_qnn_finite_shot_suite_looks_up_cases_and_serialises_its_evidence() -> None:
    """A suite finds its cases by name, refuses unknown names and serialises every part."""
    suite = run_parameter_shift_qnn_finite_shot_convergence_suite(
        case_names=("single_feature_finite_shot_phase_flip",)
    )

    case = suite.case_by_name("single_feature_finite_shot_phase_flip")
    assert case is suite.cases[0]
    with pytest.raises(KeyError, match="unknown QNN finite-shot convergence case: missing"):
        suite.case_by_name("missing")
    payload = suite.to_dict()
    assert payload["case_count"] == 1
    assert payload["passed"] is suite.passed
    assert payload["total_shots"] == case.total_shots
    assert payload["cases"] == [case.to_dict()]
    scenarios = summarize_parameter_shift_qnn_finite_shot_unsuitable_scenarios()
    assert payload["unsuitable_scenarios"] == [scenario.to_dict() for scenario in scenarios]
    assert scenarios[0].to_dict() == {
        "name": "hardware_provider_jobs",
        "status": "fail_closed_or_staged",
        "reason": scenarios[0].reason,
        "mitigation": scenarios[0].mitigation,
    }
