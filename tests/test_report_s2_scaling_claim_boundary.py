# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S2 claim-boundary report
"""Tests for S2 scaling claim-boundary reporting."""

from __future__ import annotations

from scpn_quantum_control.benchmarks.advantage_protocol import default_s2_scaling_protocol
from scripts.report_s2_scaling_claim_boundary import build_claim_boundary_report


def _rows():
    protocol = default_s2_scaling_protocol()
    rows = []
    for baseline in protocol.required_baselines:
        row = {key: None for key in protocol.output_schema["row_keys"]}
        row["protocol_id"] = protocol.protocol_id
        row["n_qubits"] = 4
        row["baseline"] = baseline
        row["status"] = "skipped"
        row["metric_payload"] = {}
        row["command"] = "test"
        row["machine"] = "test"
        row["dependencies"] = {}
        row["git_commit"] = "test"
        row["notes"] = ["size-gated fixture row"]
        rows.append(row)
    rows[0]["status"] = "ok"
    rows[0]["wall_time_ms"] = 1.0
    rows[0]["memory_bytes"] = 1024
    return rows


def test_claim_boundary_report_forbids_advantage_from_lite_rows() -> None:
    report = build_claim_boundary_report(_rows())

    assert report["validation"]["valid"] is True
    assert report["advantage_claim"] is False
    assert report["ibm_readiness"]["decision"] == "blocked_no_qpu_advantage_spend"
    assert report["ibm_readiness"]["ready_for_meaningful_ibm_advantage_run"] is False
    assert report["ibm_readiness"]["hardware_ok_rows"] == 0
    assert any(
        "Do not claim broad quantum advantage" in item for item in report["forbidden_claims"]
    )
    assert any("IBM time" in item for item in report["forbidden_claims"])
    assert any("MPS/TN" in item for item in report["remaining_blockers"])


def test_ibm_readiness_requires_full_required_matrix_and_hardware_row() -> None:
    protocol = default_s2_scaling_protocol()
    rows = []
    for size in protocol.sizes:
        for baseline in protocol.required_baselines:
            row = {key: None for key in protocol.output_schema["row_keys"]}
            row["protocol_id"] = protocol.protocol_id
            row["n_qubits"] = size
            row["baseline"] = baseline
            row["status"] = "ok"
            row["wall_time_ms"] = 1.0
            row["memory_bytes"] = 1024
            row["metric_payload"] = {}
            row["command"] = "test"
            row["machine"] = "test"
            row["dependencies"] = {}
            row["git_commit"] = "test"
            row["notes"] = []
            rows.append(row)
    hardware = {key: None for key in protocol.output_schema["row_keys"]}
    hardware["protocol_id"] = protocol.protocol_id
    hardware["n_qubits"] = 4
    hardware["baseline"] = "qpu_hardware"
    hardware["status"] = "ok"
    hardware["wall_time_ms"] = 1.0
    hardware["memory_bytes"] = 0
    hardware["metric_payload"] = {
        "job_ids": ["ibm-run-fixture"],
        "raw_counts_path": "data/raw.json",
    }
    hardware["command"] = "test"
    hardware["machine"] = "ibm-fixture"
    hardware["dependencies"] = {}
    hardware["git_commit"] = "test"
    hardware["notes"] = ["fixture preregistered hardware row"]
    rows.append(hardware)

    report = build_claim_boundary_report(rows)

    assert report["validation"]["valid"] is True
    assert (
        report["ibm_readiness"]["decision"] == "ready_for_preregistered_ibm_advantage_comparison"
    )
    assert report["ibm_readiness"]["ready_for_meaningful_ibm_advantage_run"] is True
    assert report["ibm_readiness"]["required_matrix"]["full_required_matrix_ok"] is True
    assert report["ibm_readiness"]["hardware_ok_rows"] == 1
