# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 terminal boundary runner tests
"""Tests for the Paper 0 EBS and terminal boundary fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_terminal_boundary_fixture import write_outputs


def test_run_terminal_boundary_fixture_writes_result_and_report(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"

    payload = write_outputs(output_path=output, report_path=report)

    written = json.loads(output.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R07073", "P0R07080"]
    assert written["terminal_count"] == 7
    assert written["hardware_status"] == "boundary_protocol_no_device_execution"
    assert "Paper 0 Terminal Boundary Fixture" in report_text
    assert "no unbound empirical claim" in report_text
