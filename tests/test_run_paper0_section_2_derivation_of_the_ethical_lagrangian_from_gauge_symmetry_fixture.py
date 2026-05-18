# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Section 2: Derivation of the Ethical Lagrangian from Gauge Symmetry runner tests
"""Tests for the Paper 0 Section 2: Derivation of the Ethical Lagrangian from Gauge Symmetry fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_fixture import (
    render_report,
    write_outputs,
)


def test_run_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R03612", "P0R03621"]
    assert payload["source_record_count"] == 10
    assert payload["component_count"] == 3
    assert payload["next_source_boundary"] == "P0R03622"
    assert (
        payload["claim_boundary"]
        == "source-bounded section 2 derivation of the ethical lagrangian from gauge symmetry source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Section 2: Derivation of the Ethical Lagrangian from Gauge Symmetry"
        + " Fixture"
        in report
    )
    assert (
        "source_section_2_derivation_of_the_ethical_lagrangian_from_gauge_symmetry_only_no_experiment"
        in render_report(payload)
    )
