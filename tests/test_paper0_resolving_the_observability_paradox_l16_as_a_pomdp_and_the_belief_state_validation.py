# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB validation tests
"""Tests for Paper 0 Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateConfig,
    classify_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_component,
    resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_labels,
    validate_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_fixture,
)


def test_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_fixture()
    )
    assert result.source_ledger_span == ("P0R05603", "P0R05624")
    assert result.source_record_count == 22
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05625"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05603"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05624"


def test_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",):
        assert (
            classify_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_labels()
    assert (
        labels["section"]
        == "Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB"
    )
    assert labels["next_boundary"] == "P0R05625"


def test_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 22"):
        ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateConfig(
            expected_source_record_count=21
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05625"):
        ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateConfig(
            next_source_boundary="P0R05624"
        )
    with pytest.raises(
        ValueError,
        match="unknown resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state component",
    ):
        classify_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_component(
            "empirical_validation_claim"
        )
