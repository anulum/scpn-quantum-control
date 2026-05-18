# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation runner tests
"""Tests for the Paper 0 Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_fixture import (
    render_report,
    write_outputs,
)


def test_run_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R05273", "P0R05284"]
    assert payload["source_record_count"] == 12
    assert payload["component_count"] == 3
    assert payload["next_source_boundary"] == "P0R05285"
    assert (
        payload["claim_boundary"]
        == "source-bounded table 1 predicted nths phase characteristics in multi agent active infer source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation"
        + " Fixture"
        in report
    )
    assert (
        "source_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_only_no_experiment"
        in render_report(payload)
    )
