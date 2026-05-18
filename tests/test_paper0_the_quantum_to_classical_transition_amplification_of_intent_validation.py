# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Quantum-to-Classical Transition: Amplification of Intent validation tests
"""Tests for Paper 0 The Quantum-to-Classical Transition: Amplification of Intent source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_quantum_to_classical_transition_amplification_of_intent_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheQuantumToClassicalTransitionAmplificationOfIntentConfig,
    classify_the_quantum_to_classical_transition_amplification_of_intent_component,
    the_quantum_to_classical_transition_amplification_of_intent_labels,
    validate_the_quantum_to_classical_transition_amplification_of_intent_fixture,
)


def test_the_quantum_to_classical_transition_amplification_of_intent_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_quantum_to_classical_transition_amplification_of_intent_fixture()
    assert result.source_ledger_span == ("P0R03332", "P0R03342")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03343"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_quantum_to_classical_transition_amplification_of_intent_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03332"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03342"


def test_the_quantum_to_classical_transition_amplification_of_intent_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_quantum_to_classical_transition_amplification_of_intent",
        "mechanism_1_guided_einselection",
    ):
        assert (
            classify_the_quantum_to_classical_transition_amplification_of_intent_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_quantum_to_classical_transition_amplification_of_intent_labels()
    assert labels["section"] == "The Quantum-to-Classical Transition: Amplification of Intent"
    assert labels["next_boundary"] == "P0R03343"


def test_the_quantum_to_classical_transition_amplification_of_intent_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        TheQuantumToClassicalTransitionAmplificationOfIntentConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheQuantumToClassicalTransitionAmplificationOfIntentConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03343"):
        TheQuantumToClassicalTransitionAmplificationOfIntentConfig(next_source_boundary="P0R03342")
    with pytest.raises(
        ValueError,
        match="unknown the_quantum_to_classical_transition_amplification_of_intent component",
    ):
        classify_the_quantum_to_classical_transition_amplification_of_intent_component(
            "empirical_validation_claim"
        )
