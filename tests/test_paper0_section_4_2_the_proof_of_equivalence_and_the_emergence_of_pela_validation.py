# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.2 The Proof of Equivalence and the Emergence of PELA validation tests
"""Tests for Paper 0 4.2 The Proof of Equivalence and the Emergence of PELA source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section42TheProofOfEquivalenceAndTheEmergenceOfPelaConfig,
    classify_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_component,
    section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_labels,
    validate_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_fixture,
)


def test_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_fixture()
    assert result.source_ledger_span == ("P0R03826", "P0R03847")
    assert result.source_record_count == 22
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03848"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03826"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03847"


def test_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_2_the_proof_of_equivalence_and_the_emergence_of_pela",
        "4_3_resolving_the_category_error_ethics_as_an_attractor_viability_metric",
    ):
        assert (
            classify_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_labels()
    assert labels["section"] == "4.2 The Proof of Equivalence and the Emergence of PELA"
    assert labels["next_boundary"] == "P0R03848"


def test_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 22"):
        Section42TheProofOfEquivalenceAndTheEmergenceOfPelaConfig(expected_source_record_count=21)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section42TheProofOfEquivalenceAndTheEmergenceOfPelaConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03848"):
        Section42TheProofOfEquivalenceAndTheEmergenceOfPelaConfig(next_source_boundary="P0R03847")
    with pytest.raises(
        ValueError,
        match="unknown section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela component",
    ):
        classify_section_4_2_the_proof_of_equivalence_and_the_emergence_of_pela_component(
            "empirical_validation_claim"
        )
