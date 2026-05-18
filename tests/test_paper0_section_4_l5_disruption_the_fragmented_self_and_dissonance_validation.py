# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4. L5 Disruption (The Fragmented Self and Dissonance): validation tests
"""Tests for Paper 0 4. L5 Disruption (The Fragmented Self and Dissonance): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_l5_disruption_the_fragmented_self_and_dissonance_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section4L5DisruptionTheFragmentedSelfAndDissonanceConfig,
    classify_section_4_l5_disruption_the_fragmented_self_and_dissonance_component,
    section_4_l5_disruption_the_fragmented_self_and_dissonance_labels,
    validate_section_4_l5_disruption_the_fragmented_self_and_dissonance_fixture,
)


def test_section_4_l5_disruption_the_fragmented_self_and_dissonance_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_l5_disruption_the_fragmented_self_and_dissonance_fixture()
    assert result.source_ledger_span == ("P0R05066", "P0R05074")
    assert result.source_record_count == 9
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05075"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_l5_disruption_the_fragmented_self_and_dissonance_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05066"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05074"


def test_section_4_l5_disruption_the_fragmented_self_and_dissonance_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("4_l5_disruption_the_fragmented_self_and_dissonance",):
        assert (
            classify_section_4_l5_disruption_the_fragmented_self_and_dissonance_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_l5_disruption_the_fragmented_self_and_dissonance_labels()
    assert labels["section"] == "4. L5 Disruption (The Fragmented Self and Dissonance):"
    assert labels["next_boundary"] == "P0R05075"


def test_section_4_l5_disruption_the_fragmented_self_and_dissonance_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section4L5DisruptionTheFragmentedSelfAndDissonanceConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section4L5DisruptionTheFragmentedSelfAndDissonanceConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05075"):
        Section4L5DisruptionTheFragmentedSelfAndDissonanceConfig(next_source_boundary="P0R05074")
    with pytest.raises(
        ValueError,
        match="unknown section_4_l5_disruption_the_fragmented_self_and_dissonance component",
    ):
        classify_section_4_l5_disruption_the_fragmented_self_and_dissonance_component(
            "empirical_validation_claim"
        )
