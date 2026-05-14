# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 front matter context runner tests
"""Tests for the Paper 0 front matter context fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_front_matter_context_fixture import write_outputs


def test_run_front_matter_context_fixture_writes_result_and_report(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"

    payload = write_outputs(output_path=output, report_path=report)

    written = json.loads(output.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00018", "P0R00104"]
    assert written["context_classification"] == "fragmented_context_boundary"
    assert written["blank_placeholder_count"] == 45
    assert written["hardware_status"] == "source_context_no_experiment"
    assert "Paper 0 Front Matter Context Fixture" in report_text
    assert "not validation evidence" in report_text
