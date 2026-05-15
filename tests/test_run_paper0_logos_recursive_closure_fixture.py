# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Logos recursive closure runner tests
"""Tests for the Paper 0 Logos recursive-closure fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_logos_recursive_closure_fixture import write_outputs


def test_run_logos_recursive_closure_fixture_writes_result_and_report(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"

    payload = write_outputs(output_path=output, report_path=report)

    written = json.loads(output.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R00545", "P0R00577"]
    assert written["axiom_count"] == 3
    assert written["hint_role_count"] == 3
    assert written["hardware_status"] == "source_methodology_no_experiment"
    assert "Paper 0 Logos Recursive Closure Fixture" in report_text
    assert "defines_lambda_sigma_information_geometry" in report_text
