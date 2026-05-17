# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle validation tests
"""Tests for Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.case_study_the_layer_5_organismal_self_action_perception_cycle_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig,
    case_study_the_layer_5_organismal_self_action_perception_cycle_labels,
    classify_case_study_the_layer_5_organismal_self_action_perception_cycle_component,
    validate_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture,
)


def test_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture()
    assert result.source_ledger_span == ("P0R02177", "P0R02188")
    assert result.source_record_count == 12
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02189"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_case_study_the_layer_5_organismal_self_action_perception_cycle_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02177"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02188"


def test_case_study_the_layer_5_organismal_self_action_perception_cycle_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("case_study_the_layer_5_organismal_self_action_perception_cycle",):
        assert (
            classify_case_study_the_layer_5_organismal_self_action_perception_cycle_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = case_study_the_layer_5_organismal_self_action_perception_cycle_labels()
    assert labels["section"] == "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle"
    assert labels["next_boundary"] == "P0R02189"


def test_case_study_the_layer_5_organismal_self_action_perception_cycle_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig(
            expected_source_record_count=11
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02189"):
        CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig(
            next_source_boundary="P0R02188"
        )
    with pytest.raises(
        ValueError,
        match="unknown case_study_the_layer_5_organismal_self_action_perception_cycle component",
    ):
        classify_case_study_the_layer_5_organismal_self_action_perception_cycle_component(
            "empirical_validation_claim"
        )
