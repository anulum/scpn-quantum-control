# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation validation tests
"""Tests for Paper 0 Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferConfig,
    classify_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_component,
    table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_labels,
    validate_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_fixture,
)


def test_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_fixture()
    )
    assert result.source_ledger_span == ("P0R05273", "P0R05284")
    assert result.source_record_count == 12
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05285"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05273"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05284"


def test_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
        "synthesis_implications_and_consequent_trajectories",
        "section_8_the_role_of_cybernetic_closure_and_the_anulum",
    ):
        assert (
            classify_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_labels()
    assert (
        labels["section"]
        == "Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation"
    )
    assert labels["next_boundary"] == "P0R05285"


def test_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferConfig(
            expected_source_record_count=11
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05285"):
        Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferConfig(
            next_source_boundary="P0R05284"
        )
    with pytest.raises(
        ValueError,
        match="unknown table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer component",
    ):
        classify_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_component(
            "empirical_validation_claim"
        )
