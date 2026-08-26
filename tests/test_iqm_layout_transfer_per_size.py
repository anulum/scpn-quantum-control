# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM Garnet per-size layout-transfer readiness tests
"""Tests for the frozen IQM Garnet per-size layout-transfer readiness package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.iqm_layout_transfer_per_size import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CAMPAIGN,
    REPETITIONS,
    analyse_per_size_counts,
    build_per_size_layout_transfer_plan,
    holm_adjusted_p_values,
)
from scpn_quantum_control.hardware.iqm_lattice_calibration import LatticeCalibration


def _grid_calibration(rows: int = 3, cols: int = 4) -> LatticeCalibration:
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(cols):
            qubit = row * cols + column
            if column + 1 < cols:
                edges.append((qubit, qubit + 1))
            if row + 1 < rows:
                edges.append((qubit, qubit + cols))
    return LatticeCalibration(
        num_qubits=rows * cols,
        edges=tuple(sorted(edges)),
        edge_fidelity={edge: 0.995 for edge in edges},
        readout_error={qubit: 0.01 for qubit in range(rows * cols)},
    )


def _counts_for_plan(plan: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for block in plan["blocks"]:
        n = int(block["n"])
        readout_width = len(block["readout_qubits"])
        counts[f"readout_n{n}_zeros"] = {"0" * readout_width: 1024}
        counts[f"readout_n{n}_ones"] = {"1" * readout_width: 1024}
        arm_payloads = {arm["arm"]: arm for arm in block["arms"]}
        for arm, outcomes in {
            "optimised": {"0" * (n - 1) + "1": 2048},
            "default": {"0" * n: 1024, "0" * (n - 1) + "1": 1024},
            "naive": {"0" * n: 2048},
        }.items():
            assert len(arm_payloads[arm]["measured_qubits"]) == n
            for repetition in REPETITIONS:
                counts[f"main_n{n}_{arm}_rep{repetition}"] = outcomes
    return counts


def test_frozen_constants_and_full_budget() -> None:
    plan = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4, 8, 12))
    assert REPETITIONS == (1, 2, 3, 4)
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert BOOTSTRAP_SEED == 20260722
    assert plan.circuit_count == 42
    assert plan.total_shots == 79_872
    assert plan.all_gates_pass


def test_manifest_has_unique_execution_order_repetitions() -> None:
    plan = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4,))
    labels = [label for label, _ in plan.circuit_manifest()]
    assert labels == [
        "main_n4_optimised_rep1",
        "main_n4_optimised_rep2",
        "main_n4_optimised_rep3",
        "main_n4_optimised_rep4",
        "main_n4_default_rep1",
        "main_n4_default_rep2",
        "main_n4_default_rep3",
        "main_n4_default_rep4",
        "main_n4_naive_rep1",
        "main_n4_naive_rep2",
        "main_n4_naive_rep3",
        "main_n4_naive_rep4",
        "readout_n4_zeros",
        "readout_n4_ones",
    ]
    assert len(labels) == len(set(labels))


def test_plan_payload_records_frozen_campaign_and_single_snapshot_layouts() -> None:
    plan = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4,))
    payload = plan.to_dict()
    assert payload["campaign"] == CAMPAIGN
    assert payload["main_shots_per_arm_size"] == 8192
    assert payload["circuit_count"] == 14
    assert payload["all_gates_pass"] is True
    assert len(payload["blocks"][0]["arms"]) == 3


def test_holm_adjustment_is_monotone_in_rank() -> None:
    adjusted = holm_adjusted_p_values({8: 0.01, 12: 0.03, 16: 0.02})
    assert adjusted == pytest.approx({8: 0.03, 16: 0.04, 12: 0.04})


def test_frozen_analysis_is_deterministic_and_complete() -> None:
    built = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4, 8, 12))
    plan = built.to_dict()
    counts = _counts_for_plan(plan)
    first = analyse_per_size_counts(plan, counts, n_resamples=200, seed=7)
    second = analyse_per_size_counts(plan, counts, n_resamples=200, seed=7)
    assert first == second
    assert first["matrix_complete"] is True
    assert first["bootstrap"] == {"resamples": 200, "seed": 7}
    assert set(first["per_size"]) == {"4", "8", "12"}
    assert first["s2_pooled_default_minus_optimised"]["point"] > 0
    assert first["s3_cochran_q"]["analysable"] is True
    for payload in first["per_size"].values():
        assert payload["primary_default_minus_optimised"]["point"] > 0
        assert np.isfinite(payload["primary_default_minus_optimised"]["two_sided_p"])


def test_analysis_fails_closed_on_matrix_or_gate_mismatch() -> None:
    built = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4,))
    plan = built.to_dict()
    counts = _counts_for_plan(plan)
    counts.pop("main_n4_default_rep4")
    with pytest.raises(ValueError, match="count matrix mismatch"):
        analyse_per_size_counts(plan, counts, n_resamples=10)
    plan["all_gates_pass"] = False
    with pytest.raises(ValueError, match="depth-parity gate failed"):
        analyse_per_size_counts(plan, {}, n_resamples=10)
