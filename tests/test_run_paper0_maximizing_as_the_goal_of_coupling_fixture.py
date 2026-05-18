# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Maximizing  as the Goal of Coupling: runner tests
"""Tests for the Paper 0 Maximizing  as the Goal of Coupling: fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_maximizing_as_the_goal_of_coupling_fixture import (
    render_report,
    write_outputs,
)


def test_run_maximizing_as_the_goal_of_coupling_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R03539", "P0R03554"]
    assert payload["source_record_count"] == 16
    assert payload["component_count"] == 4
    assert payload["next_source_boundary"] == "P0R03555"
    assert (
        payload["claim_boundary"]
        == "source-bounded maximizing as the goal of coupling source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "Maximizing  as the Goal of Coupling:" + " Fixture" in report
    assert "source_maximizing_as_the_goal_of_coupling_only_no_experiment" in render_report(payload)
