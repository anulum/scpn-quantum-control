# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Cytoskeleton: The L1 Quantum Scaffold validation tests
"""Tests for Paper 0 1. The Cytoskeleton: The L1 Quantum Scaffold source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_cytoskeleton_the_l1_quantum_scaffold_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheCytoskeletonTheL1QuantumScaffoldConfig,
    classify_section_1_the_cytoskeleton_the_l1_quantum_scaffold_component,
    section_1_the_cytoskeleton_the_l1_quantum_scaffold_labels,
    validate_section_1_the_cytoskeleton_the_l1_quantum_scaffold_fixture,
)


def test_section_1_the_cytoskeleton_the_l1_quantum_scaffold_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_1_the_cytoskeleton_the_l1_quantum_scaffold_fixture()
    assert result.source_ledger_span == ("P0R04728", "P0R04736")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04737"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_cytoskeleton_the_l1_quantum_scaffold_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04728"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04736"


def test_section_1_the_cytoskeleton_the_l1_quantum_scaffold_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_cytoskeleton_the_l1_quantum_scaffold",
        "2_mitochondria_the_bioenergetic_generators_l1_l3",
        "3_endoplasmic_reticulum_er_calcium_dynamics",
    ):
        assert (
            classify_section_1_the_cytoskeleton_the_l1_quantum_scaffold_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_the_cytoskeleton_the_l1_quantum_scaffold_labels()
    assert labels["section"] == "1. The Cytoskeleton: The L1 Quantum Scaffold"
    assert labels["next_boundary"] == "P0R04737"


def test_section_1_the_cytoskeleton_the_l1_quantum_scaffold_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section1TheCytoskeletonTheL1QuantumScaffoldConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheCytoskeletonTheL1QuantumScaffoldConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04737"):
        Section1TheCytoskeletonTheL1QuantumScaffoldConfig(next_source_boundary="P0R04736")
    with pytest.raises(
        ValueError, match="unknown section_1_the_cytoskeleton_the_l1_quantum_scaffold component"
    ):
        classify_section_1_the_cytoskeleton_the_l1_quantum_scaffold_component(
            "empirical_validation_claim"
        )
