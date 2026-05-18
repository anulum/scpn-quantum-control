# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Cross-Frequency Coupling (CFC) and Hierarchical Processing: validation tests
"""Tests for Paper 0 2. Cross-Frequency Coupling (CFC) and Hierarchical Processing: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingConfig,
    classify_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_component,
    section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_labels,
    validate_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_fixture,
)


def test_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_fixture()
    assert result.source_ledger_span == ("P0R04488", "P0R04498")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04499"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04488"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04498"


def test_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_cross_frequency_coupling_cfc_and_hierarchical_processing",
        "the_chemoarchitectural_basis_of_network_dynamics",
    ):
        assert (
            classify_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_labels()
    assert labels["section"] == "2. Cross-Frequency Coupling (CFC) and Hierarchical Processing:"
    assert labels["next_boundary"] == "P0R04499"


def test_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingConfig(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04499"):
        Section2CrossFrequencyCouplingCfcAndHierarchicalProcessingConfig(
            next_source_boundary="P0R04498"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_cross_frequency_coupling_cfc_and_hierarchical_processing component",
    ):
        classify_section_2_cross_frequency_coupling_cfc_and_hierarchical_processing_component(
            "empirical_validation_claim"
        )
