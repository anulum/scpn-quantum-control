# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry cascade runner tests
"""Tests for the Paper 0 symmetry-cascade fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_symmetry_cascade_fixture import render_report, write_outputs


def test_symmetry_cascade_fixture_runner_writes_json_and_report(tmp_path: Path) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json",
        report_path=tmp_path / "fixture.md",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")

    assert payload["source_ledger_span"] == ["P0R01582", "P0R01596"]
    assert payload["hardware_status"] == "source_methodology_no_experiment"
    assert payload["source_record_count"] == 15
    assert payload["component_count"] == 4
    assert payload["next_source_boundary"] == "P0R01597"
    assert "Paper 0 Symmetry Cascade Fixture" in report
    assert "geometric_informational_world_interface_summary_boundary" in report
    assert "source_symmetry_cascade_only_no_experiment" in render_report(payload)
