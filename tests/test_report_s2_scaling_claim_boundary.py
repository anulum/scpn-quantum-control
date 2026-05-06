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
        row["notes"] = []
        rows.append(row)
    rows[0]["status"] = "ok"
    rows[0]["wall_time_ms"] = 1.0
    return rows


def test_claim_boundary_report_forbids_advantage_from_lite_rows() -> None:
    report = build_claim_boundary_report(_rows())

    assert report["validation"]["valid"] is True
    assert report["advantage_claim"] is False
    assert any(
        "Do not claim broad quantum advantage" in item for item in report["forbidden_claims"]
    )
    assert any("MPS/TN" in item for item in report["remaining_blockers"])
