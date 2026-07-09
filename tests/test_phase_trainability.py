# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Trainability Diagnostics
"""Tests for phase/trainability.py barren-plateau diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import Parameter, ParameterShiftRule
from scpn_quantum_control.phase import (
    TRAINABILITY_CLAIM_BOUNDARY,
    BarrenPlateauTrainabilityReport,
    run_barren_plateau_trainability_report,
)

ScalarObjective = Callable[[NDArray[np.float64]], float]


def _curved_objective(params: NDArray[np.float64]) -> float:
    return float((1.0 - np.cos(params[0])) + 0.2 * (1.0 - np.cos(params[1])))


def _flat_objective(params: NDArray[np.float64]) -> float:
    del params
    return 1.0


def _linear_objective(params: NDArray[np.float64]) -> float:
    return float(params[0] + 0.5 * params[1])


def test_trainability_report_marks_trainable_landscape_and_dry_run_cost() -> None:
    report = run_barren_plateau_trainability_report(
        _curved_objective,
        np.array([[0.2, -0.3], [0.7, 0.4], [-0.5, 0.6]], dtype=np.float64),
        parameters=(Parameter("theta0"), Parameter("theta1")),
        plus_variances=np.array([0.04, 0.09], dtype=np.float64),
        minus_variances=np.array([0.05, 0.10], dtype=np.float64),
        target_standard_error=0.02,
        min_shots=10,
        cost_per_shot=0.001,
        cost_unit="credits",
    )

    payload = report.to_dict()
    dry_run = report.shot_dry_run

    assert isinstance(report, BarrenPlateauTrainabilityReport)
    assert report.status == "trainable"
    assert report.sample_count == 3
    assert report.claim_boundary == TRAINABILITY_CLAIM_BOUNDARY
    assert not report.barren_plateau_detected
    assert report.warnings == ()
    assert dry_run.variance_source == "caller_supplied"
    assert dry_run.backend_plan.backend == "finite_shot_simulator"
    assert dry_run.estimated_shift_evaluations == 4
    assert dry_run.estimated_quantum_shots == dry_run.allocation.total_shots
    assert dry_run.estimated_cost == pytest.approx(dry_run.allocation.total_shots * 0.001)
    assert dry_run.cost_unit == "credits"
    assert not dry_run.hardware_execution
    assert cast(str, payload["status"]) == "trainable"
    assert cast(dict[str, object], payload["shot_dry_run"])["hardware_execution"] is False


def test_trainability_report_detects_flat_objective_from_low_gradient_samples() -> None:
    report = run_barren_plateau_trainability_report(
        _flat_objective,
        np.array([[0.0, 0.0], [0.4, -0.2]], dtype=np.float64),
        target_standard_error=0.05,
        min_shots=4,
    )

    assert report.barren_plateau_detected
    assert report.status == "flat_objective"
    assert report.shot_dry_run.variance_source == "gradient_sample_variance_floor"
    assert report.shot_dry_run.estimated_shift_evaluations == 4
    assert "mean_gradient_norm_below_threshold" in report.warnings
    assert "gradient_variance_below_threshold" in report.warnings
    np.testing.assert_allclose(report.gradient_mean, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(report.gradient_variance, np.zeros(2, dtype=np.float64))


def test_trainability_report_handles_multi_frequency_and_frozen_parameter() -> None:
    rule = ParameterShiftRule(
        shifts=(float(np.pi / 2.0), float(np.pi / 4.0)),
        coefficients=(0.5, 0.25),
    )
    report = run_barren_plateau_trainability_report(
        _curved_objective,
        np.array([[0.2, -0.3], [0.7, 0.4], [-0.5, 0.6]], dtype=np.float64),
        parameters=(Parameter("theta0"), Parameter("theta1", trainable=False)),
        rule=rule,
        plus_variances=np.array([[0.04, 0.09], [0.02, 0.03]], dtype=np.float64),
        minus_variances=np.array([[0.05, 0.10], [0.02, 0.03]], dtype=np.float64),
        target_standard_error=0.03,
        min_shots=5,
    )

    assert report.samples[0].method == "multi_frequency_parameter_shift"
    assert report.samples[0].evaluations == 5
    assert report.shot_dry_run.allocation.method == "multi_frequency_parameter_shift_target_se"
    assert report.shot_dry_run.allocation.shots.shape == (2, 2, 2)
    assert report.shot_dry_run.estimated_shift_evaluations == 8
    assert report.shot_dry_run.allocation.parameter_names == ("theta0", "theta1")
    assert report.shot_dry_run.allocation.trainable == (True, False)
    assert report.shot_dry_run.allocation.predicted_standard_error[1] == 0.0

    derived_variance_report = run_barren_plateau_trainability_report(
        _curved_objective,
        np.array([[0.2, -0.3], [0.7, 0.4], [-0.5, 0.6]], dtype=np.float64),
        rule=rule,
        target_standard_error=0.03,
        min_shots=5,
    )
    assert derived_variance_report.shot_dry_run.variance_source == (
        "gradient_sample_variance_floor"
    )
    assert derived_variance_report.shot_dry_run.allocation.shots.shape == (2, 2, 2)


def test_trainability_report_marks_shot_limited_when_cap_prevents_target() -> None:
    report = run_barren_plateau_trainability_report(
        _curved_objective,
        np.array([[0.2, -0.3], [0.7, 0.4], [-0.5, 0.6]], dtype=np.float64),
        plus_variances=np.array([1.0, 1.0], dtype=np.float64),
        minus_variances=np.array([1.0, 1.0], dtype=np.float64),
        target_standard_error=1.0e-4,
        min_shots=5,
        max_shots_per_evaluation=10,
    )

    assert report.status == "shot_limited"
    assert report.shot_dry_run.capped
    assert "shot_allocation_cap_exceeded_target" in report.warnings


def test_trainability_report_marks_low_gradient_variance_without_flat_norm() -> None:
    report = run_barren_plateau_trainability_report(
        _linear_objective,
        np.array([[0.0, 0.0], [0.4, -0.2], [0.8, 0.3]], dtype=np.float64),
        gradient_variance_threshold=1.0e-10,
        gradient_norm_threshold=1.0e-9,
        target_standard_error=0.05,
        min_shots=3,
    )

    assert report.status == "low_gradient_variance"
    assert not report.barren_plateau_detected
    assert report.mean_gradient_norm > 1.0e-9
    assert "gradient_variance_below_threshold" in report.warnings
    assert "mean_gradient_norm_below_threshold" not in report.warnings


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("one_sample", "sample_params"),
        ("non_finite_sample", "finite"),
        ("bad_target", "target_standard_error"),
        ("bad_variance_threshold", "gradient_variance_threshold"),
        ("bad_norm_threshold", "gradient_norm_threshold"),
        ("bad_min_shots", "min_shots"),
        ("bad_shot_cap", "max_shots_per_evaluation"),
        ("bad_cost", "cost_per_shot"),
        ("bad_cost_unit", "cost_unit"),
        ("one_sided_variances", "plus_variances"),
        ("hardware_backend", "unsupported"),
    ),
)
def test_trainability_report_rejects_invalid_inputs(case: str, match: str) -> None:
    sample_params = np.array([[0.2, -0.3], [0.7, 0.4]], dtype=np.float64)
    objective = cast(ScalarObjective, _curved_objective)

    with pytest.raises(ValueError, match=match):
        if case == "one_sample":
            run_barren_plateau_trainability_report(
                objective,
                np.array([0.2], dtype=np.float64),
            )
        elif case == "non_finite_sample":
            run_barren_plateau_trainability_report(
                objective,
                np.array([[0.2], [np.nan]], dtype=np.float64),
            )
        elif case == "bad_target":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                target_standard_error=0.0,
            )
        elif case == "bad_variance_threshold":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                gradient_variance_threshold=-1.0,
            )
        elif case == "bad_norm_threshold":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                gradient_norm_threshold=-1.0,
            )
        elif case == "bad_min_shots":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                min_shots=0,
            )
        elif case == "bad_shot_cap":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                max_shots_per_evaluation=0,
            )
        elif case == "bad_cost":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                cost_per_shot=-1.0,
            )
        elif case == "bad_cost_unit":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                cost_unit="   ",
            )
        elif case == "one_sided_variances":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                plus_variances=np.array([0.1, 0.2], dtype=np.float64),
            )
        elif case == "hardware_backend":
            run_barren_plateau_trainability_report(
                objective,
                sample_params,
                backend="ibm_quantum",
            )
        else:
            raise AssertionError(f"unhandled invalid-input case {case}")
