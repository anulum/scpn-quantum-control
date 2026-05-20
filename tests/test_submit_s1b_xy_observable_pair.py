# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for configurable S1b/S1c XY runner
"""Tests for the configurable S1 direct-XY IBM runner."""

from __future__ import annotations

from scripts.submit_s1b_ibm_xy_observable_pair import _controller, _parse_args


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
