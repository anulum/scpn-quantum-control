# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Locus of the Interaction: validation tests
"""Tests for Paper 0 The Locus of the Interaction: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_locus_of_the_interaction_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheLocusOfTheInteractionConfig,
    classify_the_locus_of_the_interaction_component,
    the_locus_of_the_interaction_labels,
    validate_the_locus_of_the_interaction_fixture,
)


def test_the_locus_of_the_interaction_fixture_preserves_source_boundary() -> None:
    result = validate_the_locus_of_the_interaction_fixture()
    assert result.source_ledger_span == ("P0R02551", "P0R02565")
    assert result.source_record_count == 15
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R02566"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_locus_of_the_interaction_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02551"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02565"


def test_the_locus_of_the_interaction_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_locus_of_the_interaction",
        "holonomy_as_memory_of_coupling",
        "the_dynamic_visualisation_the_scpn_torus",
        "conceptual_specification_of_the_scpn_torus",
        "1_the_geometry_and_flow",
    ):
        assert (
            classify_the_locus_of_the_interaction_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_locus_of_the_interaction_labels()
    assert labels["section"] == "The Locus of the Interaction:"
    assert labels["next_boundary"] == "P0R02566"


def test_the_locus_of_the_interaction_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        TheLocusOfTheInteractionConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        TheLocusOfTheInteractionConfig(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02566"):
        TheLocusOfTheInteractionConfig(next_source_boundary="P0R02565")
    with pytest.raises(ValueError, match="unknown the_locus_of_the_interaction component"):
        classify_the_locus_of_the_interaction_component("empirical_validation_claim")
