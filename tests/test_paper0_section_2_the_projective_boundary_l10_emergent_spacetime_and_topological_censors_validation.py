# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Projective Boundary (L10): Emergent Spacetime and Topological Censorship validation tests
"""Tests for Paper 0 2. The Projective Boundary (L10): Emergent Spacetime and Topological Censorship source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheProjectiveBoundaryL10EmergentSpacetimeAndTopologicalCensorsConfig,
    classify_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_component,
    section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_labels,
    validate_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_fixture,
)


def test_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_fixture()
    assert result.source_ledger_span == ("P0R04454", "P0R04461")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04462"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04454"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04461"


def test_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors",
        "vii_synthesis_the_scpn_torus",
        "the_neurobiological_architecture_of_the_scpn",
    ):
        assert (
            classify_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = (
        section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_labels()
    )
    assert (
        labels["section"]
        == "2. The Projective Boundary (L10): Emergent Spacetime and Topological Censorship"
    )
    assert labels["next_boundary"] == "P0R04462"


def test_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2TheProjectiveBoundaryL10EmergentSpacetimeAndTopologicalCensorsConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2TheProjectiveBoundaryL10EmergentSpacetimeAndTopologicalCensorsConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04462"):
        Section2TheProjectiveBoundaryL10EmergentSpacetimeAndTopologicalCensorsConfig(
            next_source_boundary="P0R04461"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors component",
    ):
        classify_section_2_the_projective_boundary_l10_emergent_spacetime_and_topological_censors_component(
            "empirical_validation_claim"
        )
