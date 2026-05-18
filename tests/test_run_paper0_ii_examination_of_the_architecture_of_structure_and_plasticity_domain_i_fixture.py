# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) runner tests
"""Tests for the Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture import (
    render_report,
    write_outputs,
)


def test_run_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R04560", "P0R04571"]
    assert payload["source_record_count"] == 12
    assert payload["component_count"] == 3
    assert payload["next_source_boundary"] == "P0R04572"
    assert (
        payload["claim_boundary"]
        == "source-bounded ii examination of the architecture of structure and plasticity domain i source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)"
        + " Fixture"
        in report
    )
    assert (
        "source_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_only_no_experiment"
        in render_report(payload)
    )
