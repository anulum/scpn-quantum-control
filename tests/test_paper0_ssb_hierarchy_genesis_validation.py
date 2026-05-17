# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SSB hierarchy genesis validation tests
"""Tests for Paper 0 SSB hierarchy-genesis validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ssb_hierarchy_genesis_validation import (
    SSBHierarchyGenesisConfig,
    classify_ssb_hierarchy_genesis_component,
    ssb_hierarchy_genesis_labels,
    validate_ssb_hierarchy_genesis_fixture,
)


def test_ssb_hierarchy_genesis_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 21"):
        SSBHierarchyGenesisConfig(expected_source_record_count=20)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        SSBHierarchyGenesisConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01714"):
        SSBHierarchyGenesisConfig(next_source_boundary="P0R01713")


def test_ssb_hierarchy_genesis_classifiers_are_source_bounded() -> None:
    assert (
        classify_ssb_hierarchy_genesis_component("architecture_cascade")
        == "ssb_architecture_cascade_source_boundary"
    )
    assert (
        classify_ssb_hierarchy_genesis_component("conformal_torsion_seeding")
        == "conformal_torsion_seeding_source_boundary"
    )
    assert (
        classify_ssb_hierarchy_genesis_component("three_strike_explanation")
        == "three_strike_explanatory_analogy_boundary"
    )
    with pytest.raises(ValueError, match="unknown SSB hierarchy-genesis component"):
        classify_ssb_hierarchy_genesis_component("meta_framework")


def test_ssb_hierarchy_genesis_fixture_preserves_claim_boundary() -> None:
    result = validate_ssb_hierarchy_genesis_fixture()

    assert result.source_ledger_span == ("P0R01693", "P0R01713")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 21
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01714"
    assert result.null_controls == {
        "architecture_cascade_is_not_measured_layer_hierarchy": 1.0,
        "sec_torsion_seeding_is_not_observational_cosmology": 1.0,
        "three_strike_analogy_is_not_physical_derivation": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ssb_hierarchy_genesis_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01693"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01713"


def test_ssb_hierarchy_genesis_labels_name_next_meta_framework_boundary() -> None:
    labels = ssb_hierarchy_genesis_labels()

    assert (
        labels["section"]
        == "The Genesis of the Hierarchy: A Cascade of Sequential Symmetry Breaking"
    )
    assert labels["architecture"] == "15-layer SCPN as sequential SSB remnant"
    assert labels["breaks"] == "laws, individuals, actuality"
    assert labels["next_boundary"] == "Meta-Framework Integrations"
