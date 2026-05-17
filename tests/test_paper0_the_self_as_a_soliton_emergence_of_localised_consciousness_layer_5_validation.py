# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) validation tests
"""Tests for Paper 0 The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheSelfAsASolitonEmergenceOfLocalisedConsciousnessLayer5Config,
    classify_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_component,
    the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_labels,
    validate_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_fixture,
)


def test_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_fixture()
    assert result.source_ledger_span == ("P0R01792", "P0R01802")
    assert result.source_record_count == 11
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R01803"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01792"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01802"


def test_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5",):
        assert (
            classify_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_labels()
    assert (
        labels["section"]
        == 'The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5)'
    )
    assert labels["next_boundary"] == "P0R01803"


def test_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        TheSelfAsASolitonEmergenceOfLocalisedConsciousnessLayer5Config(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheSelfAsASolitonEmergenceOfLocalisedConsciousnessLayer5Config(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01803"):
        TheSelfAsASolitonEmergenceOfLocalisedConsciousnessLayer5Config(
            next_source_boundary="P0R01802"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5 component",
    ):
        classify_the_self_as_a_soliton_emergence_of_localised_consciousness_layer_5_component(
            "empirical_validation_claim"
        )
