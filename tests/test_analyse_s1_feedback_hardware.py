# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S1 feedback analysis
"""Tests for S1 hardware raw-count analysis."""

from __future__ import annotations

import pytest

from scripts.analyse_s1_feedback_hardware import analyse_package


def _package():
    return {
        "experiment_id": "s1",
        "target_r": 0.7,
        "job_ids": ["job-feedback", "job-control"],
        "arms": [
            {
                "label": "feedback",
                "records": [
                    {"r_live": 0.62, "counts": {"000": 8, "111": 8}},
                    {"r_live": 0.68, "counts": {"000": 10, "111": 6}},
                ],
            },
            {
                "label": "matched_open_loop_control",
                "records": [
                    {"r_live": 0.50, "counts": {"000": 7, "111": 9}},
                    {"r_live": 0.55, "counts": {"000": 6, "111": 10}},
                ],
            },
        ],
    }


def test_analyse_package_reports_feedback_target_error_improvement() -> None:
    summary = analyse_package(_package())

    assert summary["experiment_id"] == "s1"
    assert summary["decision"] == "positive"
    assert summary["target_error_improvement"] == pytest.approx(0.125)
    assert summary["relative_target_error_improvement"] == pytest.approx(0.125 / 0.175)
    assert summary["feedback_minus_control_mean_r_live"] == pytest.approx(0.125)
    assert summary["arm_summaries"][0]["total_shots"] == 32


def test_analyse_package_rejects_missing_required_arms() -> None:
    package = _package()
    package["arms"] = [package["arms"][0]]

    with pytest.raises(ValueError, match="at least two arms"):
        analyse_package(package)


def test_analyse_package_rejects_invalid_counts() -> None:
    package = _package()
    package["arms"][0]["records"][0]["counts"] = {"000": -1}

    with pytest.raises(ValueError, match="non-negative integer"):
        analyse_package(package)
