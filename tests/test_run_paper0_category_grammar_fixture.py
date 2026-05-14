# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category grammar runner tests
"""Tests for the Paper 0 category grammar fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_category_grammar_fixture import render_report, write_outputs


def test_category_grammar_fixture_runner_writes_auditable_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "category_grammar_result.json"
    report_path = tmp_path / "category_grammar_result.md"

    payload = write_outputs(output_path=json_path, report_path=report_path)
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    report = render_report(payload)

    assert payload["hardware_status"] == "formal_consistency_fixture_no_execution"
    assert payload["source_ledger_span"] == ["P0R06815", "P0R06877"]
    assert payload["layer_count"] == 16
    assert payload["category_laws"]["identity_law"] is True
    assert payload["category_laws"]["associativity_law"] is True
    assert payload["functorial_mapping_valid"] is True
    assert payload["null_controls"]["unsupported_empirical_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in payload["claim_boundary"]
    assert persisted == payload
    assert "# Paper 0 Category Grammar Fixture" in report
    assert report_path.read_text(encoding="utf-8") == report
