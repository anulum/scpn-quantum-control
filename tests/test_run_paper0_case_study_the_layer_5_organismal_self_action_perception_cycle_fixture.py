# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle runner tests
"""Tests for the Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture import (
    render_report,
    write_outputs,
)


def test_run_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R02177", "P0R02188"]
    assert payload["source_record_count"] == 12
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R02189"
    assert (
        payload["claim_boundary"]
        == "source-bounded case study the layer 5 organismal self action perception cycle source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle"
        + " Fixture"
        in report
    )
    assert (
        "source_case_study_the_layer_5_organismal_self_action_perception_cycle_only_no_experiment"
        in render_report(payload)
    )
