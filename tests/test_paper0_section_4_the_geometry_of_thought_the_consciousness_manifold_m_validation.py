# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4. The Geometry of Thought (The Consciousness Manifold M): validation tests
"""Tests for Paper 0 4. The Geometry of Thought (The Consciousness Manifold M): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_the_geometry_of_thought_the_consciousness_manifold_m_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig,
    classify_section_4_the_geometry_of_thought_the_consciousness_manifold_m_component,
    section_4_the_geometry_of_thought_the_consciousness_manifold_m_labels,
    validate_section_4_the_geometry_of_thought_the_consciousness_manifold_m_fixture,
)


def test_section_4_the_geometry_of_thought_the_consciousness_manifold_m_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_the_geometry_of_thought_the_consciousness_manifold_m_fixture()
    assert result.source_ledger_span == ("P0R04526", "P0R04533")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04534"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_the_geometry_of_thought_the_consciousness_manifold_m_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04526"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04533"


def test_section_4_the_geometry_of_thought_the_consciousness_manifold_m_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_the_geometry_of_thought_the_consciousness_manifold_m",
        "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",
        "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
    ):
        assert (
            classify_section_4_the_geometry_of_thought_the_consciousness_manifold_m_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_the_geometry_of_thought_the_consciousness_manifold_m_labels()
    assert labels["section"] == "4. The Geometry of Thought (The Consciousness Manifold M):"
    assert labels["next_boundary"] == "P0R04534"


def test_section_4_the_geometry_of_thought_the_consciousness_manifold_m_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04534"):
        Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig(
            next_source_boundary="P0R04533"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_the_geometry_of_thought_the_consciousness_manifold_m component",
    ):
        classify_section_4_the_geometry_of_thought_the_consciousness_manifold_m_component(
            "empirical_validation_claim"
        )
