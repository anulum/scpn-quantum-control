# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mechanisms of Criticality and Control (Layers 1-4) validation tests
"""Tests for Paper 0 Mechanisms of Criticality and Control (Layers 1-4) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mechanisms_of_criticality_and_control_layers_1_4_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MechanismsOfCriticalityAndControlLayers14Config,
    classify_mechanisms_of_criticality_and_control_layers_1_4_component,
    mechanisms_of_criticality_and_control_layers_1_4_labels,
    validate_mechanisms_of_criticality_and_control_layers_1_4_fixture,
)


def test_mechanisms_of_criticality_and_control_layers_1_4_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_mechanisms_of_criticality_and_control_layers_1_4_fixture()
    assert result.source_ledger_span == ("P0R05113", "P0R05123")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05124"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mechanisms_of_criticality_and_control_layers_1_4_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05113"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05123"


def test_mechanisms_of_criticality_and_control_layers_1_4_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("mechanisms_of_criticality_and_control_layers_1_4", "p0r05120"):
        assert (
            classify_mechanisms_of_criticality_and_control_layers_1_4_component(component)
            == f"{component}_source_boundary"
        )
    labels = mechanisms_of_criticality_and_control_layers_1_4_labels()
    assert labels["section"] == "Mechanisms of Criticality and Control (Layers 1-4)"
    assert labels["next_boundary"] == "P0R05124"


def test_mechanisms_of_criticality_and_control_layers_1_4_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        MechanismsOfCriticalityAndControlLayers14Config(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        MechanismsOfCriticalityAndControlLayers14Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05124"):
        MechanismsOfCriticalityAndControlLayers14Config(next_source_boundary="P0R05123")
    with pytest.raises(
        ValueError, match="unknown mechanisms_of_criticality_and_control_layers_1_4 component"
    ):
        classify_mechanisms_of_criticality_and_control_layers_1_4_component(
            "empirical_validation_claim"
        )
