# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Section 4: Justification for a Teleological Least-Action Principle validation tests
"""Tests for Paper 0 Section 4: Justification for a Teleological Least-Action Principle source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_justification_for_a_teleological_least_action_principle_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section4JustificationForATeleologicalLeastActionPrincipleConfig,
    classify_section_4_justification_for_a_teleological_least_action_principle_component,
    section_4_justification_for_a_teleological_least_action_principle_labels,
    validate_section_4_justification_for_a_teleological_least_action_principle_fixture,
)


def test_section_4_justification_for_a_teleological_least_action_principle_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_justification_for_a_teleological_least_action_principle_fixture()
    assert result.source_ledger_span == ("P0R03638", "P0R03652")
    assert result.source_record_count == 15
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03653"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_justification_for_a_teleological_least_action_principle_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03638"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03652"


def test_section_4_justification_for_a_teleological_least_action_principle_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "section_4_justification_for_a_teleological_least_action_principle",
        "4_1_the_problem_of_teleology_in_physics",
        "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
    ):
        assert (
            classify_section_4_justification_for_a_teleological_least_action_principle_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_justification_for_a_teleological_least_action_principle_labels()
    assert (
        labels["section"] == "Section 4: Justification for a Teleological Least-Action Principle"
    )
    assert labels["next_boundary"] == "P0R03653"


def test_section_4_justification_for_a_teleological_least_action_principle_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        Section4JustificationForATeleologicalLeastActionPrincipleConfig(
            expected_source_record_count=14
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section4JustificationForATeleologicalLeastActionPrincipleConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03653"):
        Section4JustificationForATeleologicalLeastActionPrincipleConfig(
            next_source_boundary="P0R03652"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_justification_for_a_teleological_least_action_principle component",
    ):
        classify_section_4_justification_for_a_teleological_least_action_principle_component(
            "empirical_validation_claim"
        )
