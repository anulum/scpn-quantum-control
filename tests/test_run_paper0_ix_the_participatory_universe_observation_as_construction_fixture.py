# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IX. The Participatory Universe: Observation as Construction runner tests
"""Tests for the Paper 0 IX. The Participatory Universe: Observation as Construction fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_ix_the_participatory_universe_observation_as_construction_fixture import (
    render_report,
    write_outputs,
)


def test_run_ix_the_participatory_universe_observation_as_construction_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R06047", "P0R06056"]
    assert payload["source_record_count"] == 10
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R06057"
    assert (
        payload["claim_boundary"]
        == "source-bounded ix the participatory universe observation as construction source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "IX. The Participatory Universe: Observation as Construction" + " Fixture"
        in report
    )
    assert (
        "source_ix_the_participatory_universe_observation_as_construction_only_no_experiment"
        in render_report(payload)
    )
