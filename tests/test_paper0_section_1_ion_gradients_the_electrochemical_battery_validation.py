# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. Ion Gradients: The Electrochemical Battery validation tests
"""Tests for Paper 0 1. Ion Gradients: The Electrochemical Battery source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_ion_gradients_the_electrochemical_battery_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1IonGradientsTheElectrochemicalBatteryConfig,
    classify_section_1_ion_gradients_the_electrochemical_battery_component,
    section_1_ion_gradients_the_electrochemical_battery_labels,
    validate_section_1_ion_gradients_the_electrochemical_battery_fixture,
)


def test_section_1_ion_gradients_the_electrochemical_battery_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_ion_gradients_the_electrochemical_battery_fixture()
    assert result.source_ledger_span == ("P0R04711", "P0R04719")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04720"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_ion_gradients_the_electrochemical_battery_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04711"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04719"


def test_section_1_ion_gradients_the_electrochemical_battery_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_ion_gradients_the_electrochemical_battery",
        "2_the_aqueous_substrate_qed_water_and_coherence_domains_l1",
        "iii_the_neuronal_membrane_the_critical_interface_l2_l3",
    ):
        assert (
            classify_section_1_ion_gradients_the_electrochemical_battery_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_ion_gradients_the_electrochemical_battery_labels()
    assert labels["section"] == "1. Ion Gradients: The Electrochemical Battery"
    assert labels["next_boundary"] == "P0R04720"


def test_section_1_ion_gradients_the_electrochemical_battery_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section1IonGradientsTheElectrochemicalBatteryConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1IonGradientsTheElectrochemicalBatteryConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04720"):
        Section1IonGradientsTheElectrochemicalBatteryConfig(next_source_boundary="P0R04719")
    with pytest.raises(
        ValueError, match="unknown section_1_ion_gradients_the_electrochemical_battery component"
    ):
        classify_section_1_ion_gradients_the_electrochemical_battery_component(
            "empirical_validation_claim"
        )
