# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention builder tests
"""Tests for Paper 0 Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05050", "P0R05057"]
    assert bundle.summary["source_record_count"] == 8
    assert bundle.summary["consumed_source_record_count"] == 8
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 3
    assert bundle.summary["next_source_boundary"] == "P0R05058"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == ["IMG0128"]
    assert bundle.summary["table_ids"] == []


def test_build_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "1_l1_disruption_the_decoherence_cascade",
        "i_the_traumatised_brain_catastrophic_disruption_of_the_scpn_architecture",
        "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded clinical scenario analysis traumatic brain injury tbi and pharmacologica source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded clinical scenario analysis traumatic brain injury tbi and pharmacologica source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention"
        + " Specs"
        in report
    )
    assert "P0R05050 - P0R05057" in report
