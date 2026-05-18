# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Bioelectric Code in Neurogenesis and Regeneration: validation tests
"""Tests for Paper 0 1. The Bioelectric Code in Neurogenesis and Regeneration: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig,
    classify_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_component,
    section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_labels,
    validate_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_fixture,
)


def test_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_fixture()
    assert result.source_ledger_span == ("P0R04657", "P0R04665")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04666"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04657"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04665"


def test_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_bioelectric_code_in_neurogenesis_and_regeneration",
        "2_the_optimised_connectome",
        "3_the_active_role_of_glia_the_tripartite_synapse",
    ):
        assert (
            classify_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_labels()
    assert labels["section"] == "1. The Bioelectric Code in Neurogenesis and Regeneration:"
    assert labels["next_boundary"] == "P0R04666"


def test_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04666"):
        Section1TheBioelectricCodeInNeurogenesisAndRegenerationConfig(
            next_source_boundary="P0R04665"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_1_the_bioelectric_code_in_neurogenesis_and_regeneration component",
    ):
        classify_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_component(
            "empirical_validation_claim"
        )
