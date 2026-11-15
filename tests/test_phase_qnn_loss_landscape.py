# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNN Loss Landscapes
"""Tests for phase/qnn_loss_landscape.py bounded QNN loss-landscape evidence."""

from __future__ import annotations

import dataclasses

import pytest

from scpn_quantum_control.phase import (
    ParameterShiftQNNLossLandscapeSuiteResult,
    run_parameter_shift_qnn_loss_landscape_suite,
)


def test_qnn_loss_landscape_suite_records_grid_and_gradient_evidence() -> None:
    """A one-feature grid records its points, losses, gradient norms and claim boundary."""
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
    """A two-feature case samples the full product grid of both axes."""
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
    """An unmet loss-span threshold marks the case and the suite as failed."""
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
    """Unknown cases, a zero radius, too few points and a zero feature cap are refused."""
    with pytest.raises(ValueError, match="unknown QNN loss landscape case"):
        run_parameter_shift_qnn_loss_landscape_suite(case_names=("missing",))

    with pytest.raises(ValueError, match="grid_radius"):
        run_parameter_shift_qnn_loss_landscape_suite(grid_radius=0.0)

    with pytest.raises(ValueError, match="points_per_axis"):
        run_parameter_shift_qnn_loss_landscape_suite(points_per_axis=1)

    with pytest.raises(ValueError, match="max_features"):
        run_parameter_shift_qnn_loss_landscape_suite(max_features=0)


@pytest.mark.parametrize(
    ("grid_radius", "points_per_axis"), [(1.7, 11), (100.3, 7), (12345.6, 11)]
)
def test_qnn_loss_landscape_grid_keeps_its_center_point(
    grid_radius: float, points_per_axis: int
) -> None:
    """A radius whose linspace middle misses 0.0 still samples the exact centre."""
    suite = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("single_feature_phase_flip",),
        grid_radius=grid_radius,
        points_per_axis=points_per_axis,
    )

    case = suite.cases[0]
    assert case.axis_values[0][points_per_axis // 2] == 0.8
    assert case.axis_values[0][0] == pytest.approx(0.8 - grid_radius)
    assert case.axis_values[0][-1] == pytest.approx(0.8 + grid_radius)
    center = next(point for point in case.points if point.params == (0.8,))
    assert case.center_loss == center.loss
    assert suite.to_dict()["cases"] == [case.to_dict()]


def test_qnn_loss_landscape_default_suite_runs_every_case_in_order() -> None:
    """Without case names the suite runs both declared cases and finds each by name."""
    suite = run_parameter_shift_qnn_loss_landscape_suite(grid_radius=0.1, points_per_axis=3)

    assert [case.name for case in suite.cases] == [
        "single_feature_phase_flip",
        "two_feature_phase_flip",
    ]
    assert suite.passed_count == 2
    assert suite.total_point_count == 3 + 9
    assert suite.case_by_name("two_feature_phase_flip") is suite.cases[1]
    with pytest.raises(KeyError, match="^'unknown QNN loss landscape case: missing'$"):
        suite.case_by_name("missing")


def test_qnn_loss_landscape_refuses_bad_span_and_an_oversized_case() -> None:
    """A negative or NaN span threshold and a case wider than max_features are refused."""
    for span in (-1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="^min_loss_span must be finite and non-negative$"):
            run_parameter_shift_qnn_loss_landscape_suite(min_loss_span=span)
    zero_span = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("single_feature_phase_flip",), min_loss_span=0.0
    )
    assert zero_span.cases[0].min_loss_span == 0.0
    message = (
        "^QNN loss landscape case 'two_feature_phase_flip' has 2 features; "
        "max_features=1 would create an unreviewed grid$"
    )
    with pytest.raises(ValueError, match=message):
        run_parameter_shift_qnn_loss_landscape_suite(
            case_names=("two_feature_phase_flip",), max_features=1
        )
    single = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("single_feature_phase_flip",), max_features=1
    )
    assert single.case_count == 1


def test_qnn_loss_landscape_result_without_its_center_refuses_center_loss() -> None:
    """A result assembled without its centre point cannot report a centre loss."""
    suite = run_parameter_shift_qnn_loss_landscape_suite(
        case_names=("single_feature_phase_flip",), grid_radius=0.2, points_per_axis=5
    )
    case = suite.cases[0]
    off_center = dataclasses.replace(
        case, points=tuple(point for point in case.points if point.params != (0.8,))
    )

    with pytest.raises(
        RuntimeError, match="^loss landscape grid does not contain its center point$"
    ):
        _ = off_center.center_loss
