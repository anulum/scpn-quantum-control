# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3.1 Complexity (K) and the Cardinality of the State Space validation tests
"""Tests for Paper 0 3.1 Complexity (K) and the Cardinality of the State Space source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_1_complexity_k_and_the_cardinality_of_the_state_space_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section31ComplexityKAndTheCardinalityOfTheStateSpaceConfig,
    classify_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_component,
    section_3_1_complexity_k_and_the_cardinality_of_the_state_space_labels,
    validate_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_fixture,
)


def test_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_fixture()
    assert result.source_ledger_span == ("P0R03781", "P0R03788")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03789"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03781"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03788"


def test_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("3_1_complexity_k_and_the_cardinality_of_the_state_space",):
        assert (
            classify_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_3_1_complexity_k_and_the_cardinality_of_the_state_space_labels()
    assert labels["section"] == "3.1 Complexity (K) and the Cardinality of the State Space"
    assert labels["next_boundary"] == "P0R03789"


def test_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section31ComplexityKAndTheCardinalityOfTheStateSpaceConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section31ComplexityKAndTheCardinalityOfTheStateSpaceConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03789"):
        Section31ComplexityKAndTheCardinalityOfTheStateSpaceConfig(next_source_boundary="P0R03788")
    with pytest.raises(
        ValueError,
        match="unknown section_3_1_complexity_k_and_the_cardinality_of_the_state_space component",
    ):
        classify_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_component(
            "empirical_validation_claim"
        )
