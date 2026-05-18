# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.2 The Proof of Equivalence and the Emergence of PELA runner tests
"""Tests for the Paper 0 4.2 The Proof of Equivalence and the Emergence of PELA fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_fixture import (
    render_report,
    write_outputs,
)


def test_run_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R03826", "P0R03847"]
    assert payload["source_record_count"] == 22
    assert payload["component_count"] == 2
    assert payload["next_source_boundary"] == "P0R03848"
    assert (
        payload["claim_boundary"]
        == "source-bounded section 4 2 the proof of equivalence and the emergence of pela source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 " + "4.2 The Proof of Equivalence and the Emergence of PELA" + " Fixture"
        in report
    )
    assert (
        "source_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_only_no_experiment"
        in render_report(payload)
    )
