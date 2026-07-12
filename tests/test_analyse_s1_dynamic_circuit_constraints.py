# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analyse s1 dynamic circuit constraints tests
# SCPN Quantum Control -- Tests for S1 provider-constraint analysis
"""Tests for S1 dynamic-circuit provider-constraint helpers."""

from __future__ import annotations

from scripts.analyse_s1_dynamic_circuit_constraints import (
    _hamming_distance,
    _independent_multi_flip_probability,
    _readout_leakage_summary,
)


def test_hamming_distance_requires_equal_length_bitstrings() -> None:
    assert _hamming_distance("000", "101") == 2


def test_independent_multi_flip_probability_for_three_bits() -> None:
    probability = _independent_multi_flip_probability((0.1, 0.2, 0.3))

    assert abs(probability - 0.098) < 1e-12


def test_readout_leakage_summary_reports_multi_bit_proxy() -> None:
    readout_model = {
        "condition_number": 1.25,
        "calibration_rows": [
            {
                "prepared": "000",
                "total_shots": 100,
                "retention": 0.90,
                "counts": {"000": 90, "001": 5, "011": 5},
            },
            {
                "prepared": "111",
                "total_shots": 100,
                "retention": 0.80,
                "counts": {"111": 80, "110": 10, "000": 10},
            },
        ],
    }

    summary = _readout_leakage_summary(readout_model)

    assert summary["condition_number"] == 1.25
    assert abs(summary["mean_retention"] - 0.85) < 1e-12
    assert abs(summary["mean_nonretention"] - 0.15) < 1e-12
    assert abs(summary["max_single_state_leakage"] - 0.2) < 1e-12
    assert abs(summary["mean_multi_bit_flip_probability"] - 0.075) < 1e-12
    assert "mean_excess_multi_bit_flip_over_independent_proxy" in summary
