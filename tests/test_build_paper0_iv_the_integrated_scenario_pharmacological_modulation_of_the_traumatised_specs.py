# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain builder tests
"""Tests for Paper 0 IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain source-accounting specs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper0_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_specs import (
    build_from_ledger,
    write_outputs,
)


def test_build_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_specs_preserves_source_slice() -> (
    None
):
    bundle = build_from_ledger()
    assert bundle.summary["source_ledger_span"] == ["P0R05091", "P0R05101"]
    assert bundle.summary["source_record_count"] == 11
    assert bundle.summary["consumed_source_record_count"] == 11
    assert bundle.summary["coverage_match"] is True
    assert bundle.summary["unconsumed_source_ledger_ids"] == []
    assert bundle.summary["spec_count"] == 4
    assert bundle.summary["next_source_boundary"] == "P0R05102"
    assert bundle.summary["math_ids"] == []
    assert bundle.summary["image_ids"] == ["IMG0133"]
    assert bundle.summary["table_ids"] == []


def test_build_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_specs_preserves_component_source_formulae() -> (
    None
):
    bundle = build_from_ledger()
    by_context = {spec.context_id: spec for spec in bundle.specs}
    assert set(by_context) == {
        "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",
        "1_the_therapeutic_effects_stabilisation_and_f_minimisation",
        "2_the_synergistic_risks_the_dangers_of_dyscritia",
        "v_intervention_adjunct_agents",
    }
    for spec in bundle.specs:
        assert spec.source_formulae
        assert (
            spec.claim_boundary
            == "source-bounded iv the integrated scenario pharmacological modulation of the traumatised source-accounting bridge; not validation evidence"
        )
        assert spec.hardware_status == "source_methodology_no_experiment"


def test_write_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_outputs(
    tmp_path: Path,
) -> None:
    outputs = write_outputs(build_from_ledger(), output_dir=tmp_path, date_tag="2099-01-02")
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    report = outputs["report"].read_text(encoding="utf-8")
    assert payload["summary"]["coverage_match"] is True
    assert (
        payload["summary"]["claim_boundary"]
        == "source-bounded iv the integrated scenario pharmacological modulation of the traumatised source-accounting bridge; not validation evidence"
    )
    assert (
        "Paper 0 "
        + "IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain"
        + " Specs"
        in report
    )
    assert "P0R05091 - P0R05101" in report
