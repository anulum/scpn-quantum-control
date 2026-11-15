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
    """Return a rows-by-cols grid calibration with uniform edge fidelity and readout error."""
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
    """Return deterministic readout and main-arm counts for every circuit of the plan."""
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
    """The frozen repetitions, bootstrap settings, circuit count and shot budget hold."""
    plan = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4, 8, 12))
    assert REPETITIONS == (1, 2, 3, 4)
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert BOOTSTRAP_SEED == 20260722
    assert plan.circuit_count == 42
    assert plan.total_shots == 79_872
    assert plan.all_gates_pass


def test_manifest_has_unique_execution_order_repetitions() -> None:
    """The manifest lists every repetition once, in execution order."""
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
    """The plan payload records the campaign, budget and the three arms."""
    plan = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4,))
    payload = plan.to_dict()
    assert payload["campaign"] == CAMPAIGN
    assert payload["main_shots_per_arm_size"] == 8192
    assert payload["circuit_count"] == 14
    assert payload["all_gates_pass"] is True
    assert len(payload["blocks"][0]["arms"]) == 3


def test_holm_adjustment_is_monotone_in_rank() -> None:
    """Holm-adjusted p-values are monotone in rank."""
    adjusted = holm_adjusted_p_values({8: 0.01, 12: 0.03, 16: 0.02})
    assert adjusted == pytest.approx({8: 0.03, 16: 0.04, 12: 0.04})


def test_frozen_analysis_is_deterministic_and_complete() -> None:
    """The same counts and seed give the same complete per-size analysis."""
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
    """A missing circuit or a failed depth-parity gate blocks the analysis."""
    built = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4,))
    plan = built.to_dict()
    counts = _counts_for_plan(plan)
    counts.pop("main_n4_default_rep4")
    with pytest.raises(ValueError, match="count matrix mismatch"):
        analyse_per_size_counts(plan, counts, n_resamples=10)
    plan["all_gates_pass"] = False
    with pytest.raises(ValueError, match="depth-parity gate failed"):
        analyse_per_size_counts(plan, {}, n_resamples=10)


def _single_size_plan() -> dict[str, Any]:
    """Return the payload of a four-qubit transfer plan on the grid calibration."""
    return build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4,)).to_dict()


def test_analysis_refuses_another_campaign_and_a_single_resample() -> None:
    """The plan must be this campaign's, and the bootstrap needs at least two resamples."""
    plan = _single_size_plan()
    counts = _counts_for_plan(plan)

    with pytest.raises(ValueError, match="^bootstrap requires at least two resamples$"):
        analyse_per_size_counts(plan, counts, n_resamples=1)
    plan["campaign"] = "another_campaign"
    with pytest.raises(ValueError, match="^expected campaign"):
        analyse_per_size_counts(plan, counts, n_resamples=10)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        pytest.param({"0000": -1}, "^counts must be non-negative$", id="negative"),
        pytest.param(
            {"0000": 0}, "^counts must contain a positive number of", id="empty-repetition"
        ),
    ],
)
def test_analysis_refuses_negative_or_empty_repetition_counts(
    replacement: dict[str, int], message: str
) -> None:
    """A negative count, or a repetition that recorded no shots, is refused."""
    plan = _single_size_plan()
    counts = _counts_for_plan(plan)
    counts["main_n4_default_rep2"] = replacement

    with pytest.raises(ValueError, match=message):
        analyse_per_size_counts(plan, counts, n_resamples=10)


@pytest.mark.parametrize(
    "recorded",
    [pytest.param({}, id="no-outcomes"), pytest.param({"0000": 0}, id="zero-counts")],
)
def test_analysis_refuses_a_pooled_arm_without_shots(recorded: dict[str, int]) -> None:
    """An arm whose repetitions recorded no shots at all is refused, keyed or not."""
    plan = _single_size_plan()
    counts = _counts_for_plan(plan)
    for repetition in REPETITIONS:
        counts[f"main_n4_naive_rep{repetition}"] = recorded

    with pytest.raises(ValueError, match="^pooled counts must contain at least one shot$"):
        analyse_per_size_counts(plan, counts, n_resamples=10)


def test_analysis_marks_heterogeneity_unanalysable_for_a_deterministic_endpoint() -> None:
    """When every arm is deterministic the bootstrap variance is zero and Cochran's Q is withheld."""
    plan = build_per_size_layout_transfer_plan(_grid_calibration(), sizes=(4, 8)).to_dict()
    counts = _counts_for_plan(plan)
    for n in (4, 8):
        for repetition in REPETITIONS:
            counts[f"main_n{n}_default_rep{repetition}"] = {"0" * (n - 1) + "1": 2048}

    result = analyse_per_size_counts(plan, counts, n_resamples=50, seed=3)

    assert result["s3_cochran_q"]["analysable"] is False
    assert result["s3_cochran_q"]["reason"] == "at least one bootstrap variance is non-positive"
