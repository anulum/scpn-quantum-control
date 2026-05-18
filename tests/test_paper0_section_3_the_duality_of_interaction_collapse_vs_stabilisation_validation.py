# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Duality of Interaction: Collapse vs. Stabilisation validation tests
"""Tests for Paper 0 3. The Duality of Interaction: Collapse vs. Stabilisation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_the_duality_of_interaction_collapse_vs_stabilisation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3TheDualityOfInteractionCollapseVsStabilisationConfig,
    classify_section_3_the_duality_of_interaction_collapse_vs_stabilisation_component,
    section_3_the_duality_of_interaction_collapse_vs_stabilisation_labels,
    validate_section_3_the_duality_of_interaction_collapse_vs_stabilisation_fixture,
)


def test_section_3_the_duality_of_interaction_collapse_vs_stabilisation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_the_duality_of_interaction_collapse_vs_stabilisation_fixture()
    assert result.source_ledger_span == ("P0R05994", "P0R06014")
    assert result.source_record_count == 21
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R06015"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_the_duality_of_interaction_collapse_vs_stabilisation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05994"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06014"


def test_section_3_the_duality_of_interaction_collapse_vs_stabilisation_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_the_duality_of_interaction_collapse_vs_stabilisation",
        "the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
    ):
        assert (
            classify_section_3_the_duality_of_interaction_collapse_vs_stabilisation_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_3_the_duality_of_interaction_collapse_vs_stabilisation_labels()
    assert labels["section"] == "3. The Duality of Interaction: Collapse vs. Stabilisation"
    assert labels["next_boundary"] == "P0R06015"


def test_section_3_the_duality_of_interaction_collapse_vs_stabilisation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 21"):
        Section3TheDualityOfInteractionCollapseVsStabilisationConfig(
            expected_source_record_count=20
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section3TheDualityOfInteractionCollapseVsStabilisationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06015"):
        Section3TheDualityOfInteractionCollapseVsStabilisationConfig(
            next_source_boundary="P0R06014"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_3_the_duality_of_interaction_collapse_vs_stabilisation component",
    ):
        classify_section_3_the_duality_of_interaction_collapse_vs_stabilisation_component(
            "empirical_validation_claim"
        )
