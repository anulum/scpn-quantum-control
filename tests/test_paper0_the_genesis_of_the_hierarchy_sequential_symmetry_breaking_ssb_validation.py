# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) validation tests
"""Tests for Paper 0 The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbConfig,
    classify_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_component,
    the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_labels,
    validate_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_fixture,
)


def test_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_fixture()
    assert result.source_ledger_span == ("P0R01727", "P0R01754")
    assert result.source_record_count == 28
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01755"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01727"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01754"


def test_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb",
        "1_the_primordial_state_l13",
        "2_the_ssb_cascade_the_projection",
        "key_ssb_events",
    ):
        assert (
            classify_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_labels()
    assert labels["section"] == "The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB)"
    assert labels["next_boundary"] == "P0R01755"


def test_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 28"):
        TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbConfig(
            expected_source_record_count=27
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01755"):
        TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbConfig(
            next_source_boundary="P0R01754"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb component",
    ):
        classify_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_component(
            "empirical_validation_claim"
        )
