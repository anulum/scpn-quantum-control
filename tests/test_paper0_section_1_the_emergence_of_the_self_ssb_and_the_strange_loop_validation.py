# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Emergence of the Self (SSB and the Strange Loop): validation tests
"""Tests for Paper 0 1. The Emergence of the Self (SSB and the Strange Loop): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig,
    classify_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_component,
    section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_labels,
    validate_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_fixture,
)


def test_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_fixture()
    assert result.source_ledger_span == ("P0R04517", "P0R04525")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04526"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04517"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04525"


def test_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_emergence_of_the_self_ssb_and_the_strange_loop",
        "2_hierarchical_predictive_coding_hpc_in_the_cortex",
        "3_mapping_major_cognitive_networks",
    ):
        assert (
            classify_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_labels()
    assert labels["section"] == "1. The Emergence of the Self (SSB and the Strange Loop):"
    assert labels["next_boundary"] == "P0R04526"


def test_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04526"):
        Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig(next_source_boundary="P0R04525")
    with pytest.raises(
        ValueError,
        match="unknown section_1_the_emergence_of_the_self_ssb_and_the_strange_loop component",
    ):
        classify_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_component(
            "empirical_validation_claim"
        )
