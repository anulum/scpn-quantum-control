# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary validation tests
"""Tests for Paper 0 The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundConfig,
    classify_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_component,
    the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_labels,
    validate_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_fixture,
)


def test_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_fixture()
    )
    assert result.source_ledger_span == ("P0R03122", "P0R03138")
    assert result.source_record_count == 17
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03139"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03122"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03138"


def test_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",):
        assert (
            classify_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_labels()
    assert (
        labels["section"]
        == "The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary"
    )
    assert labels["next_boundary"] == "P0R03139"


def test_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundConfig(
            expected_source_record_count=16
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03139"):
        TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundConfig(
            next_source_boundary="P0R03138"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound component",
    ):
        classify_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_component(
            "empirical_validation_claim"
        )
