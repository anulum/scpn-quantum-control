# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Guided Variation (Non-Random Mutation): validation tests
"""Tests for Paper 0 2. Guided Variation (Non-Random Mutation): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_guided_variation_non_random_mutation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2GuidedVariationNonRandomMutationConfig,
    classify_section_2_guided_variation_non_random_mutation_component,
    section_2_guided_variation_non_random_mutation_labels,
    validate_section_2_guided_variation_non_random_mutation_fixture,
)


def test_section_2_guided_variation_non_random_mutation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_guided_variation_non_random_mutation_fixture()
    assert result.source_ledger_span == ("P0R06031", "P0R06038")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R06039"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_guided_variation_non_random_mutation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06031"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06038"


def test_section_2_guided_variation_non_random_mutation_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_guided_variation_non_random_mutation",
        "3_teleological_dynamics_rg_flow",
        "4_the_co_evolutionary_spiral",
    ):
        assert (
            classify_section_2_guided_variation_non_random_mutation_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_guided_variation_non_random_mutation_labels()
    assert labels["section"] == "2. Guided Variation (Non-Random Mutation):"
    assert labels["next_boundary"] == "P0R06039"


def test_section_2_guided_variation_non_random_mutation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2GuidedVariationNonRandomMutationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2GuidedVariationNonRandomMutationConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06039"):
        Section2GuidedVariationNonRandomMutationConfig(next_source_boundary="P0R06038")
    with pytest.raises(
        ValueError, match="unknown section_2_guided_variation_non_random_mutation component"
    ):
        classify_section_2_guided_variation_non_random_mutation_component(
            "empirical_validation_claim"
        )
