# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway runner tests
"""Tests for the Paper 0 Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_fixture import (
    render_report,
    write_outputs,
)


def test_run_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_fixture_writes_json_and_report(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(
        output_path=tmp_path / "fixture.json", report_path=tmp_path / "fixture.md"
    )
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["source_ledger_span"] == ["P0R02088", "P0R02097"]
    assert payload["source_record_count"] == 10
    assert payload["component_count"] == 1
    assert payload["next_source_boundary"] == "P0R02098"
    assert (
        payload["claim_boundary"]
        == "source-bounded case study the layer 3 genomic morphogenetic transduction pathway source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway"
        + " Fixture"
        in report
    )
    assert (
        "source_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_only_no_experiment"
        in render_report(payload)
    )
