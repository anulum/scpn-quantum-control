# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Mechanism of Interaction: validation tests
"""Tests for Paper 0 The Mechanism of Interaction: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_mechanism_of_interaction_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheMechanismOfInteractionConfig,
    classify_the_mechanism_of_interaction_component,
    the_mechanism_of_interaction_labels,
    validate_the_mechanism_of_interaction_fixture,
)


def test_the_mechanism_of_interaction_fixture_preserves_source_boundary() -> None:
    result = validate_the_mechanism_of_interaction_fixture()
    assert result.source_ledger_span == ("P0R03148", "P0R03173")
    assert result.source_record_count == 26
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03174"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_mechanism_of_interaction_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03148"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03173"


def test_the_mechanism_of_interaction_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_mechanism_of_interaction",
        "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
    ):
        assert (
            classify_the_mechanism_of_interaction_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_mechanism_of_interaction_labels()
    assert labels["section"] == "The Mechanism of Interaction:"
    assert labels["next_boundary"] == "P0R03174"


def test_the_mechanism_of_interaction_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 26"):
        TheMechanismOfInteractionConfig(expected_source_record_count=25)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheMechanismOfInteractionConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03174"):
        TheMechanismOfInteractionConfig(next_source_boundary="P0R03173")
    with pytest.raises(ValueError, match="unknown the_mechanism_of_interaction component"):
        classify_the_mechanism_of_interaction_component("empirical_validation_claim")
