# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 V. The Interface Problem Synthesis (Mind-Body-Field) runner tests
"""Tests for the Paper 0 V. The Interface Problem Synthesis (Mind-Body-Field) fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_v_the_interface_problem_synthesis_mind_body_field_fixture import (
    render_report,
    write_outputs,
)


def test_run_v_the_interface_problem_synthesis_mind_body_field_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R03241", "P0R03249"]
    assert payload["source_record_count"] == 9
    assert payload["component_count"] == 4
    assert payload["next_source_boundary"] == "P0R03250"
    assert (
        payload["claim_boundary"]
        == "source-bounded v the interface problem synthesis mind body field source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "V. The Interface Problem Synthesis (Mind-Body-Field)" + " Fixture" in report
    )
    assert (
        "source_v_the_interface_problem_synthesis_mind_body_field_only_no_experiment"
        in render_report(payload)
    )
