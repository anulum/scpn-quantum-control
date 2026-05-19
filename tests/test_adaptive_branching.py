# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- adaptive branching tests
"""Tests for the S8 adaptive-branching readiness layer."""

from __future__ import annotations

import pytest

from scpn_quantum_control.control.adaptive_branching import (
    ADAPTIVE_BRANCHING_SCHEMA,
    AdaptiveBranchingConfig,
    build_adaptive_branch_table,
    classify_branch_state,
    estimate_branching_readiness,
    s8_adaptive_branching_markdown,
    s8_adaptive_branching_payload,
)


def test_threshold_branch_triggers_only_below_target_deadband() -> None:
    config = AdaptiveBranchingConfig(target_r=0.75, deadband=0.05)

    low = classify_branch_state(
        local_r=0.62, parity_leakage=0.01, cluster_imbalance=0.05, config=config
    )
    hold = classify_branch_state(
        local_r=0.72,
        parity_leakage=0.01,
        cluster_imbalance=0.05,
        config=config,
    )

    assert low.action == "corrective_kick"
    assert low.triggered_policy == "local_order_threshold"
    assert hold.action == "hold"


def test_parity_leakage_has_priority_over_local_order() -> None:
    config = AdaptiveBranchingConfig(target_r=0.75, max_parity_leakage=0.08)

    decision = classify_branch_state(
        local_r=0.40,
        parity_leakage=0.2,
        cluster_imbalance=0.01,
        config=config,
    )

    assert decision.triggered_policy == "dla_parity_leakage"
    assert decision.action == "sector_rebalance"
    assert decision.claim_boundary == "branch-planning decision only; not hardware evidence"


def test_chimera_branch_detects_clustered_desynchronisation() -> None:
    config = AdaptiveBranchingConfig(target_r=0.8, chimera_imbalance_threshold=0.25)

    decision = classify_branch_state(
        local_r=0.68,
        parity_leakage=0.01,
        cluster_imbalance=0.5,
        config=config,
    )

    assert decision.triggered_policy == "chimera_cluster_detector"
    assert decision.action == "topology_aware_pulse"
    assert decision.correction_angle > 0.0


def test_branch_table_is_deterministic_and_non_submitting() -> None:
    table = build_adaptive_branch_table(
        AdaptiveBranchingConfig(target_r=0.75),
        local_r_grid=(0.5, 0.75),
        parity_leakage_grid=(0.0, 0.2),
        cluster_imbalance_grid=(0.0,),
    )

    assert len(table) == 4
    assert {row.to_dict()["hardware_submission_allowed"] for row in table} == {False}
    assert table[0].to_dict()["schema"] == "s8_adaptive_branch_row_v1"


def test_readiness_blocks_without_backend_dynamic_circuit_support() -> None:
    readiness = estimate_branching_readiness(
        AdaptiveBranchingConfig(n_oscillators=4, n_rounds=3),
        backend_features=("cross_shot_batches",),
    )

    assert readiness.ready is False
    assert "mid_circuit_measurement" in readiness.missing_features
    assert "conditional_control" in readiness.missing_features
    assert readiness.hardware_submission_allowed is False


def test_payload_keeps_s8_as_no_submit_readiness_artifact() -> None:
    payload = s8_adaptive_branching_payload()

    assert payload["schema"] == ADAPTIVE_BRANCHING_SCHEMA
    assert payload["no_qpu_submission"] is True
    assert payload["hardware_submission_allowed"] is False
    assert payload["adaptive_advantage_claim_allowed"] is False
    assert payload["branch_table_count"] > 0
    assert {row["triggered_policy"] for row in payload["branch_table"]} >= {
        "local_order_threshold",
        "dla_parity_leakage",
        "chimera_cluster_detector",
    }


def test_markdown_records_gate_and_falsifier() -> None:
    markdown = s8_adaptive_branching_markdown(s8_adaptive_branching_payload())

    assert "scpn-bench s8-adaptive-branching-readiness" in markdown
    assert "win-rate <= 50%" in markdown
    assert "no hardware submission" in markdown


def test_invalid_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="target_r"):
        AdaptiveBranchingConfig(target_r=1.5)
