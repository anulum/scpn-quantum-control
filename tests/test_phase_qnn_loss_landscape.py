# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Loss Landscapes
"""Tests for phase/qnn_loss_landscape.py bounded QNN loss-landscape evidence."""

from __future__ import annotations

import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNLossLandscapeSuiteResult,
    run_parameter_shift_qnn_loss_landscape_suite,
)


def test_qnn_loss_landscape_suite_records_grid_and_gradient_evidence() -> None:
    suite = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("single_feature_phase_flip",),
        grid_radius=0.2,
        points_per_axis=5,
    )

    assert isinstance(suite, ParameterShiftQNNLossLandscapeSuiteResult)
    assert suite.passed
    assert suite.case_count == 1
    assert suite.total_point_count == 5
    assert suite.evidence_class == "local_deterministic_qnn_loss_landscape"
    assert not suite.production_benchmark
    assert "not hardware" in suite.claim_boundary

    case = suite.case_by_name("single_feature_phase_flip")
    assert case.passed
    assert case.n_features == 1
    assert case.points_per_axis == 5
    assert case.point_count == 5
    assert case.loss_span > 0.0
    assert case.min_loss <= case.center_loss <= case.max_loss
    assert case.min_gradient_norm >= 0.0
    assert case.max_gradient_norm >= case.min_gradient_norm
    assert len(case.axis_values[0]) == 5
    assert all(point.gradient_norm >= 0.0 for point in case.points)
    assert case.to_dict()["passed"] is True
    assert suite.to_dict()["total_point_count"] == 5


def test_qnn_loss_landscape_suite_supports_two_feature_grid() -> None:
    suite = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("two_feature_phase_flip",),
        grid_radius=0.15,
        points_per_axis=3,
    )

    case = suite.case_by_name("two_feature_phase_flip")
    assert suite.passed
    assert case.n_features == 2
    assert case.point_count == 9
    assert len(case.axis_values) == 2
    assert all(len(axis) == 3 for axis in case.axis_values)
    assert len(case.argmin_params) == 2


def test_qnn_loss_landscape_records_failure_thresholds_without_hiding_them() -> None:
    suite = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("single_feature_phase_flip",),
        min_loss_span=100.0,
    )

    assert not suite.passed
    assert suite.failed_count == 1
    case = suite.cases[0]
    assert not case.passed
    assert not case.loss_span_passed
    assert case.to_dict()["passed"] is False


def test_qnn_loss_landscape_fails_closed_on_invalid_controls() -> None:
    with pytest.raises(ValueError, match="unknown QNN loss landscape case"):
        run_parameter_shift_qnn_loss_landscape_suite(case_names=("missing",))

    with pytest.raises(ValueError, match="grid_radius"):
        run_parameter_shift_qnn_loss_landscape_suite(grid_radius=0.0)

    with pytest.raises(ValueError, match="points_per_axis"):
        run_parameter_shift_qnn_loss_landscape_suite(points_per_axis=1)

    with pytest.raises(ValueError, match="max_features"):
        run_parameter_shift_qnn_loss_landscape_suite(max_features=0)
