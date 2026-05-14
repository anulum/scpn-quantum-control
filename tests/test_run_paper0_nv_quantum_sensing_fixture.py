# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 NV quantum sensing runner tests
"""Tests for the Paper 0 NV-center quantum sensing protocol runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_nv_quantum_sensing_fixture import render_report, write_outputs


def test_nv_quantum_sensing_fixture_runner_writes_auditable_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "nv_quantum_sensing_result.json"
    report_path = tmp_path / "nv_quantum_sensing_result.md"

    payload = write_outputs(output_path=json_path, report_path=report_path)
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    report = render_report(payload)

    assert payload["hardware_status"] == "protocol_design_no_lab_execution"
    assert payload["source_ledger_span"] == ["P0R06677", "P0R06729"]
    assert payload["delta_gamma"] > 0.0
    assert 0.05 <= payload["effect_size_ratio"] <= 0.15
    assert payload["falsification_rejected"] is False
    assert payload["null_controls"]["unsupported_empirical_protocol_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in payload["claim_boundary"]
    assert persisted == payload
    assert "# Paper 0 NV-Center Quantum Sensing Protocol Fixture" in report
    assert report_path.read_text(encoding="utf-8") == report
