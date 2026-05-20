# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for configurable S1b/S1c XY runner
"""Tests for the configurable S1 direct-XY IBM runner."""

from __future__ import annotations

from scripts.submit_s1b_ibm_xy_observable_pair import (
    _analysis_summary,
    _controller,
    _is_policy_sweep,
    _parse_args,
    _policy_variants,
)


def test_s1c_arguments_can_prepare_shallow_gain_tuned_lane() -> None:
    args = _parse_args(
        [
            "--lane",
            "s1c",
            "--experiment-id",
            "s1c_shallow_gain_tuned_xy_extension_2026-05-20",
            "--n-rounds",
            "1",
            "--correction-angle",
            "0.06",
            "--base-gain",
            "0.4",
            "--repetitions",
            "3",
        ]
    )

    assert args.lane == "s1c"
    assert args.experiment_id == "s1c_shallow_gain_tuned_xy_extension_2026-05-20"
    assert args.n_rounds == 1
    assert args.correction_angle == 0.06
    assert args.base_gain == 0.4
    assert args.repetitions == 3


def test_configurable_controller_applies_s1c_policy_parameters() -> None:
    args = _parse_args(
        [
            "--n-rounds",
            "1",
            "--correction-angle",
            "0.06",
            "--base-gain",
            "0.4",
        ]
    )

    controller = _controller(args)

    assert controller.config.correction_angle == 0.06
    assert controller.config.base_gain == 0.4


def test_s1d_policy_sweep_expands_to_preregistered_variants() -> None:
    args = _parse_args(
        [
            "--lane",
            "s1d",
            "--experiment-id",
            "s1d_policy_direction_sweep_2026-05-20",
            "--policy-sweep",
            "s1d",
        ]
    )

    variants = _policy_variants(args)

    assert [variant["policy_variant"] for variant in variants] == [
        "current_shallow_positive",
        "polarity_flipped",
        "weak_positive",
    ]
    assert [variant["correction_angle"] for variant in variants] == [0.06, -0.06, 0.03]
    assert [variant["base_gain"] for variant in variants] == [0.4, 0.4, 0.2]
    assert all(variant["n_rounds"] == 1 for variant in variants)


def test_s1e_confirmatory_repeat_reuses_policy_sweep_with_five_repetitions() -> None:
    args = _parse_args(
        [
            "--lane",
            "s1e",
            "--experiment-id",
            "s1e_policy_sweep_confirmatory_repeat_2026-05-20",
            "--policy-sweep",
            "s1e",
            "--repetitions",
            "5",
        ]
    )

    variants = _policy_variants(args)

    assert args.policy_sweep == "s1e"
    assert args.repetitions == 5
    assert _is_policy_sweep(args)
    assert [variant["policy_variant"] for variant in variants] == [
        "current_shallow_positive",
        "polarity_flipped",
        "weak_positive",
    ]
    assert all(variant["n_rounds"] == 1 for variant in variants)


def test_polarity_flipped_s1d_controller_preserves_signed_policy_metadata() -> None:
    args = _parse_args(
        [
            "--lane",
            "s1d",
            "--policy-sweep",
            "s1d",
        ]
    )
    polarity_flipped = _policy_variants(args)[1]

    controller = _controller(
        _parse_args(
            [
                "--correction-angle",
                str(polarity_flipped["correction_angle"]),
                "--base-gain",
                str(polarity_flipped["base_gain"]),
            ]
        )
    )

    assert controller.config.correction_angle == 0.06
    assert controller.config.feedback_correction_sign == 1.0


def test_s1d_analysis_summary_preserves_policy_variant_outcomes() -> None:
    package = {
        "experiment_id": "s1d_policy_direction_sweep_2026-05-20",
        "parent_experiment_id": "s1_dynamic_feedback_preregistration_2026-05-06",
        "lane": "s1d",
        "job_ids": ["job-a", "job-b"],
        "observable_family": "direct_xy_pauli_correlators",
        "policy_variants": [
            {
                "policy_variant": "current_shallow_positive",
                "correction_angle": 0.06,
                "base_gain": 0.4,
                "n_rounds": 1,
                "observables": [
                    {"basis": "XXI", "feedback_minus_control": -0.03},
                    {"basis": "YYI", "feedback_minus_control": -0.01},
                ],
            },
            {
                "policy_variant": "polarity_flipped",
                "correction_angle": -0.06,
                "base_gain": 0.4,
                "n_rounds": 1,
                "observables": [
                    {"basis": "XXI", "feedback_minus_control": 0.04},
                    {"basis": "YYI", "feedback_minus_control": 0.02},
                ],
            },
        ],
    }

    summary = _analysis_summary(package)

    assert summary["n_policy_variants"] == 2
    assert summary["best_variant_by_mean_signed_delta"] == "polarity_flipped"
    assert summary["policy_variants"][0]["mean_signed_feedback_minus_control"] == -0.02
    assert summary["policy_variants"][1]["mean_abs_feedback_minus_control"] == 0.03
