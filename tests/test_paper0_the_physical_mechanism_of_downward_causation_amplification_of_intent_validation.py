# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physical Mechanism of Downward Causation: Amplification of Intent validation tests
"""Tests for Paper 0 The Physical Mechanism of Downward Causation: Amplification of Intent source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_physical_mechanism_of_downward_causation_amplification_of_intent_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentConfig,
    classify_the_physical_mechanism_of_downward_causation_amplification_of_intent_component,
    the_physical_mechanism_of_downward_causation_amplification_of_intent_labels,
    validate_the_physical_mechanism_of_downward_causation_amplification_of_intent_fixture,
)


def test_the_physical_mechanism_of_downward_causation_amplification_of_intent_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_physical_mechanism_of_downward_causation_amplification_of_intent_fixture()
    )
    assert result.source_ledger_span == ("P0R03307", "P0R03314")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03315"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_physical_mechanism_of_downward_causation_amplification_of_intent_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03307"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03314"


def test_the_physical_mechanism_of_downward_causation_amplification_of_intent_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_physical_mechanism_of_downward_causation_amplification_of_intent",):
        assert (
            classify_the_physical_mechanism_of_downward_causation_amplification_of_intent_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_physical_mechanism_of_downward_causation_amplification_of_intent_labels()
    assert (
        labels["section"]
        == "The Physical Mechanism of Downward Causation: Amplification of Intent"
    )
    assert labels["next_boundary"] == "P0R03315"


def test_the_physical_mechanism_of_downward_causation_amplification_of_intent_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03315"):
        ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentConfig(
            next_source_boundary="P0R03314"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_physical_mechanism_of_downward_causation_amplification_of_intent component",
    ):
        classify_the_physical_mechanism_of_downward_causation_amplification_of_intent_component(
            "empirical_validation_claim"
        )
