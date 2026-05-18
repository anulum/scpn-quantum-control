# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Applied SCPN: Pathology, Technology, and Anomalies validation tests
"""Tests for Paper 0 Applied SCPN: Pathology, Technology, and Anomalies source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.applied_scpn_pathology_technology_and_anomalies_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    AppliedScpnPathologyTechnologyAndAnomaliesConfig,
    applied_scpn_pathology_technology_and_anomalies_labels,
    classify_applied_scpn_pathology_technology_and_anomalies_component,
    validate_applied_scpn_pathology_technology_and_anomalies_fixture,
)


def test_applied_scpn_pathology_technology_and_anomalies_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_applied_scpn_pathology_technology_and_anomalies_fixture()
    assert result.source_ledger_span == ("P0R06197", "P0R06205")
    assert result.source_record_count == 9
    assert result.component_count == 6
    assert result.next_source_boundary == "P0R06206"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_applied_scpn_pathology_technology_and_anomalies_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06197"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06205"


def test_applied_scpn_pathology_technology_and_anomalies_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "applied_scpn_pathology_technology_and_anomalies",
        "i_pathology_and_therapeutics",
        "aetiology_of_disorder",
        "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction",
        "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig",
        "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
    ):
        assert (
            classify_applied_scpn_pathology_technology_and_anomalies_component(component)
            == f"{component}_source_boundary"
        )
    labels = applied_scpn_pathology_technology_and_anomalies_labels()
    assert labels["section"] == "Applied SCPN: Pathology, Technology, and Anomalies"
    assert labels["next_boundary"] == "P0R06206"


def test_applied_scpn_pathology_technology_and_anomalies_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        AppliedScpnPathologyTechnologyAndAnomaliesConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 6"):
        AppliedScpnPathologyTechnologyAndAnomaliesConfig(expected_component_count=7)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06206"):
        AppliedScpnPathologyTechnologyAndAnomaliesConfig(next_source_boundary="P0R06205")
    with pytest.raises(
        ValueError, match="unknown applied_scpn_pathology_technology_and_anomalies component"
    ):
        classify_applied_scpn_pathology_technology_and_anomalies_component(
            "empirical_validation_claim"
        )
