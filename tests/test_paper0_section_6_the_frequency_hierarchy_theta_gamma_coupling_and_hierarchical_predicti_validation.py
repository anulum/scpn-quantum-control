# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 6. The Frequency Hierarchy: Theta-Gamma Coupling and Hierarchical Predictive Coding validation tests
"""Tests for Paper 0 6. The Frequency Hierarchy: Theta-Gamma Coupling and Hierarchical Predictive Coding source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section6TheFrequencyHierarchyThetaGammaCouplingAndHierarchicalPredictiConfig,
    classify_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_component,
    section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_labels,
    validate_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_fixture,
)


def test_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_fixture()
    assert result.source_ledger_span == ("P0R04507", "P0R04516")
    assert result.source_record_count == 10
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04517"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04507"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04516"


def test_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti",
        "v_the_architecture_of_cognition_and_self_l5",
    ):
        assert (
            classify_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = (
        section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_labels()
    )
    assert (
        labels["section"]
        == "6. The Frequency Hierarchy: Theta-Gamma Coupling and Hierarchical Predictive Coding"
    )
    assert labels["next_boundary"] == "P0R04517"


def test_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        Section6TheFrequencyHierarchyThetaGammaCouplingAndHierarchicalPredictiConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section6TheFrequencyHierarchyThetaGammaCouplingAndHierarchicalPredictiConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04517"):
        Section6TheFrequencyHierarchyThetaGammaCouplingAndHierarchicalPredictiConfig(
            next_source_boundary="P0R04516"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti component",
    ):
        classify_section_6_the_frequency_hierarchy_theta_gamma_coupling_and_hierarchical_predicti_component(
            "empirical_validation_claim"
        )
