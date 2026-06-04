# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Conformance
"""Tests for phase/qnn_conformance.py bounded QNN audit evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNConformanceSuiteResult,
    parameter_shift_qnn_classifier_gradient,
    run_parameter_shift_qnn_conformance_suite,
    summarize_parameter_shift_qnn_unsuitable_scenarios,
)


def test_qnn_conformance_suite_records_training_and_gradient_evidence() -> None:
    suite = run_parameter_shift_qnn_conformance_suite()

    assert isinstance(suite, ParameterShiftQNNConformanceSuiteResult)
    assert suite.passed
    assert suite.case_count == 3
    assert suite.gradient_passed_count == 3
    assert suite.training_passed_count == 1
    assert suite.external_agreement_count == 0
    assert suite.unsuitable_scenario_count >= 3
    assert tuple(case.name for case in suite.cases) == (
        "phase_separable_single_feature",
        "two_feature_mixed_phase",
        "balanced_threshold_two_feature",
    )

    training_case = suite.cases[0]
    assert training_case.training_required
    assert training_case.training_passed
    assert training_case.training_accuracy == 1.0
    assert training_case.training_best_loss is not None
    assert training_case.training_best_loss < 1e-4

    for case in suite.cases:
        assert case.finite_difference_passed
        assert case.max_abs_error <= case.tolerance
        assert case.shift_terms == 2
        assert case.method == "multi_frequency_parameter_shift_qnn_gradient"
        assert case.parameter_shift_evaluations == 4 * case.n_features
        assert case.to_dict()["finite_difference_passed"] is True

    payload = suite.to_dict()
    assert payload["passed"] is True
    assert payload["case_count"] == 3
    assert payload["gradient_passed_count"] == 3


def test_qnn_conformance_suite_records_named_external_gradient_agreements() -> None:
    features = np.array(
        [[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]],
        dtype=float,
    )
    labels = np.array([0.0, 1.0, 0.25], dtype=float)
    params = np.array([0.4, -0.2], dtype=float)
    expected = parameter_shift_qnn_classifier_gradient(features, labels, params)

    suite = run_parameter_shift_qnn_conformance_suite(
        external_gradients={
            "two_feature_mixed_phase": {
                "jax_manual_reference": lambda _values: expected.copy(),
                "pennylane_manual_reference": lambda _values: (
                    expected + np.array([2e-8, -2e-8], dtype=float)
                ),
            }
        },
        external_tolerance=1e-6,
    )

    case = suite.case_by_name("two_feature_mixed_phase")
    assert case.external_agreement_count == 2
    assert case.external_passed
    assert suite.external_agreement_count == 2
    assert suite.passed
    assert tuple(case.external_agreement_names) == (
        "jax_manual_reference",
        "pennylane_manual_reference",
    )


def test_qnn_conformance_suite_fails_closed_on_bad_external_gradient() -> None:
    with pytest.raises(ValueError, match="external gradient"):
        run_parameter_shift_qnn_conformance_suite(
            external_gradients={
                "phase_separable_single_feature": {
                    "bad_shape": lambda _values: np.array([1.0, 2.0], dtype=float)
                }
            }
        )


def test_qnn_conformance_unsuitable_scenarios_are_explicit() -> None:
    scenarios = summarize_parameter_shift_qnn_unsuitable_scenarios()

    assert len(scenarios) >= 3
    assert {scenario.name for scenario in scenarios} >= {
        "hardware_backend",
        "arbitrary_qnn_architecture",
        "nonfinite_training_data",
    }
    assert all(scenario.status == "fail_closed_or_staged" for scenario in scenarios)
    assert all(scenario.mitigation for scenario in scenarios)
