# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Bridge Between Mind and Matter: How Consciousness Influences the Brain's Electricity validation tests
"""Tests for Paper 0 The Bridge Between Mind and Matter: How Consciousness Influences the Brain's Electricity source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheBridgeBetweenMindAndMatterHowConsciousnessInfluencesTheBraiConfig,
    classify_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_component,
    the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_labels,
    validate_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_fixture,
)


def test_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_fixture()
    )
    assert result.source_ledger_span == ("P0R04359", "P0R04371")
    assert result.source_record_count == 13
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04372"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04359"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04371"


def test_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai",):
        assert (
            classify_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_labels()
    assert (
        labels["section"]
        == "The Bridge Between Mind and Matter: How Consciousness Influences the Brain's Electricity"
    )
    assert labels["next_boundary"] == "P0R04372"


def test_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        TheBridgeBetweenMindAndMatterHowConsciousnessInfluencesTheBraiConfig(
            expected_source_record_count=12
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheBridgeBetweenMindAndMatterHowConsciousnessInfluencesTheBraiConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04372"):
        TheBridgeBetweenMindAndMatterHowConsciousnessInfluencesTheBraiConfig(
            next_source_boundary="P0R04371"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai component",
    ):
        classify_the_bridge_between_mind_and_matter_how_consciousness_influences_the_brai_component(
            "empirical_validation_claim"
        )
