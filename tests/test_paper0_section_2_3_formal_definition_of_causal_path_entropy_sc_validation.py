# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2.3 Formal Definition of Causal Path Entropy (SC) validation tests
"""Tests for Paper 0 2.3 Formal Definition of Causal Path Entropy (SC) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_3_formal_definition_of_causal_path_entropy_sc_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section23FormalDefinitionOfCausalPathEntropyScConfig,
    classify_section_2_3_formal_definition_of_causal_path_entropy_sc_component,
    section_2_3_formal_definition_of_causal_path_entropy_sc_labels,
    validate_section_2_3_formal_definition_of_causal_path_entropy_sc_fixture,
)


def test_section_2_3_formal_definition_of_causal_path_entropy_sc_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_3_formal_definition_of_causal_path_entropy_sc_fixture()
    assert result.source_ledger_span == ("P0R03762", "P0R03771")
    assert result.source_record_count == 10
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03772"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_3_formal_definition_of_causal_path_entropy_sc_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03762"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03771"


def test_section_2_3_formal_definition_of_causal_path_entropy_sc_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_3_formal_definition_of_causal_path_entropy_sc",
        "2_4_the_causal_entropic_force",
    ):
        assert (
            classify_section_2_3_formal_definition_of_causal_path_entropy_sc_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_3_formal_definition_of_causal_path_entropy_sc_labels()
    assert labels["section"] == "2.3 Formal Definition of Causal Path Entropy (SC)"
    assert labels["next_boundary"] == "P0R03772"


def test_section_2_3_formal_definition_of_causal_path_entropy_sc_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        Section23FormalDefinitionOfCausalPathEntropyScConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section23FormalDefinitionOfCausalPathEntropyScConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03772"):
        Section23FormalDefinitionOfCausalPathEntropyScConfig(next_source_boundary="P0R03771")
    with pytest.raises(
        ValueError,
        match="unknown section_2_3_formal_definition_of_causal_path_entropy_sc component",
    ):
        classify_section_2_3_formal_definition_of_causal_path_entropy_sc_component(
            "empirical_validation_claim"
        )
