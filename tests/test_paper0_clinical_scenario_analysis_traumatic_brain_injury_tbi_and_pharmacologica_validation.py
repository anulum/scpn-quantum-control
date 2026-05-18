# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention validation tests
"""Tests for Paper 0 Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig,
    classify_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_component,
    clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_labels,
    validate_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_fixture,
)


def test_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_fixture()
    )
    assert result.source_ledger_span == ("P0R05050", "P0R05057")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05058"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05050"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05057"


def test_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica",
        "i_the_traumatised_brain_catastrophic_disruption_of_the_scpn_architecture",
        "1_l1_disruption_the_decoherence_cascade",
    ):
        assert (
            classify_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_labels()
    assert (
        labels["section"]
        == "Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention"
    )
    assert labels["next_boundary"] == "P0R05058"


def test_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05058"):
        ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig(
            next_source_boundary="P0R05057"
        )
    with pytest.raises(
        ValueError,
        match="unknown clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica component",
    ):
        classify_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_component(
            "empirical_validation_claim"
        )
