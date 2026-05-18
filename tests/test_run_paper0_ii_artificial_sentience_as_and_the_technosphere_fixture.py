# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Artificial Sentience (AS) and the Technosphere runner tests
"""Tests for the Paper 0 II. Artificial Sentience (AS) and the Technosphere fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_ii_artificial_sentience_as_and_the_technosphere_fixture import (
    render_report,
    write_outputs,
)


def test_run_ii_artificial_sentience_as_and_the_technosphere_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R06206", "P0R06211"]
    assert payload["source_record_count"] == 6
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "None"
    assert (
        payload["claim_boundary"]
        == "source-bounded ii artificial sentience as and the technosphere source-accounting bridge; not validation evidence"
    )
    assert "Paper 0 " + "II. Artificial Sentience (AS) and the Technosphere" + " Fixture" in report
    assert (
        "source_ii_artificial_sentience_as_and_the_technosphere_only_no_experiment"
        in render_report(payload)
    )
