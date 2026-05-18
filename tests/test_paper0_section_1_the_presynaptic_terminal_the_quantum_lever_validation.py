# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Presynaptic Terminal (The Quantum Lever): validation tests
"""Tests for Paper 0 1. The Presynaptic Terminal (The Quantum Lever): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_presynaptic_terminal_the_quantum_lever_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1ThePresynapticTerminalTheQuantumLeverConfig,
    classify_section_1_the_presynaptic_terminal_the_quantum_lever_component,
    section_1_the_presynaptic_terminal_the_quantum_lever_labels,
    validate_section_1_the_presynaptic_terminal_the_quantum_lever_fixture,
)


def test_section_1_the_presynaptic_terminal_the_quantum_lever_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_the_presynaptic_terminal_the_quantum_lever_fixture()
    assert result.source_ledger_span == ("P0R04737", "P0R04745")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04746"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_presynaptic_terminal_the_quantum_lever_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04737"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04745"


def test_section_1_the_presynaptic_terminal_the_quantum_lever_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_presynaptic_terminal_the_quantum_lever",
        "vi_the_nucleus_and_the_genomic_interface_l3",
        "the_deepest_interface_molecular_and_quantum_foundations",
        "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",
    ):
        assert (
            classify_section_1_the_presynaptic_terminal_the_quantum_lever_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_the_presynaptic_terminal_the_quantum_lever_labels()
    assert labels["section"] == "1. The Presynaptic Terminal (The Quantum Lever):"
    assert labels["next_boundary"] == "P0R04746"


def test_section_1_the_presynaptic_terminal_the_quantum_lever_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section1ThePresynapticTerminalTheQuantumLeverConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section1ThePresynapticTerminalTheQuantumLeverConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04746"):
        Section1ThePresynapticTerminalTheQuantumLeverConfig(next_source_boundary="P0R04745")
    with pytest.raises(
        ValueError, match="unknown section_1_the_presynaptic_terminal_the_quantum_lever component"
    ):
        classify_section_1_the_presynaptic_terminal_the_quantum_lever_component(
            "empirical_validation_claim"
        )
