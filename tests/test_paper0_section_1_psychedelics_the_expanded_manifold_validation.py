# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. Psychedelics (The Expanded Manifold): validation tests
"""Tests for Paper 0 1. Psychedelics (The Expanded Manifold): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_psychedelics_the_expanded_manifold_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1PsychedelicsTheExpandedManifoldConfig,
    classify_section_1_psychedelics_the_expanded_manifold_component,
    section_1_psychedelics_the_expanded_manifold_labels,
    validate_section_1_psychedelics_the_expanded_manifold_fixture,
)


def test_section_1_psychedelics_the_expanded_manifold_fixture_preserves_source_boundary() -> None:
    result = validate_section_1_psychedelics_the_expanded_manifold_fixture()
    assert result.source_ledger_span == ("P0R05026", "P0R05038")
    assert result.source_record_count == 13
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R05039"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_psychedelics_the_expanded_manifold_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05026"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05038"


def test_section_1_psychedelics_the_expanded_manifold_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_psychedelics_the_expanded_manifold",
        "2_meditation_and_flow_states_optimised_criticality",
        "3_anaesthesia_the_decoupling",
        "vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
    ):
        assert (
            classify_section_1_psychedelics_the_expanded_manifold_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_psychedelics_the_expanded_manifold_labels()
    assert labels["section"] == "1. Psychedelics (The Expanded Manifold):"
    assert labels["next_boundary"] == "P0R05039"


def test_section_1_psychedelics_the_expanded_manifold_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        Section1PsychedelicsTheExpandedManifoldConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section1PsychedelicsTheExpandedManifoldConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05039"):
        Section1PsychedelicsTheExpandedManifoldConfig(next_source_boundary="P0R05038")
    with pytest.raises(
        ValueError, match="unknown section_1_psychedelics_the_expanded_manifold component"
    ):
        classify_section_1_psychedelics_the_expanded_manifold_component(
            "empirical_validation_claim"
        )
