# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9) validation tests
"""Tests for Paper 0 4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Config,
    classify_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_component,
    section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_labels,
    validate_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_fixture,
)


def test_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_fixture()
    assert result.source_ledger_span == ("P0R04802", "P0R04812")
    assert result.source_record_count == 11
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04813"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04802"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04812"


def test_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9",
        "vi_neuro_metabolism_and_energetics_l1_l4",
        "the_geometric_scaffold_of_the_brain_the_architecture_of_consciousness",
        "i_introduction_the_brain_as_a_geometric_engine",
    ):
        assert (
            classify_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_labels()
    assert (
        labels["section"]
        == "4. Nuclear Spin and Posner Clusters: The Quantum Memory Substrate (L1/L9)"
    )
    assert labels["next_boundary"] == "P0R04813"


def test_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Config(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Config(
            expected_component_count=5
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04813"):
        Section4NuclearSpinAndPosnerClustersTheQuantumMemorySubstrateL1L9Config(
            next_source_boundary="P0R04812"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9 component",
    ):
        classify_section_4_nuclear_spin_and_posner_clusters_the_quantum_memory_substrate_l1_l9_component(
            "empirical_validation_claim"
        )
