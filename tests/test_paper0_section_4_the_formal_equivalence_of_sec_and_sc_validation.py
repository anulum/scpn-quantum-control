# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4. The Formal Equivalence of SEC and SC validation tests
"""Tests for Paper 0 4. The Formal Equivalence of SEC and SC source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_the_formal_equivalence_of_sec_and_sc_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section4TheFormalEquivalenceOfSecAndScConfig,
    classify_section_4_the_formal_equivalence_of_sec_and_sc_component,
    section_4_the_formal_equivalence_of_sec_and_sc_labels,
    validate_section_4_the_formal_equivalence_of_sec_and_sc_fixture,
)


def test_section_4_the_formal_equivalence_of_sec_and_sc_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_the_formal_equivalence_of_sec_and_sc_fixture()
    assert result.source_ledger_span == ("P0R03804", "P0R03817")
    assert result.source_record_count == 14
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03818"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_the_formal_equivalence_of_sec_and_sc_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03804"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03817"


def test_section_4_the_formal_equivalence_of_sec_and_sc_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("4_the_formal_equivalence_of_sec_and_sc",):
        assert (
            classify_section_4_the_formal_equivalence_of_sec_and_sc_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_4_the_formal_equivalence_of_sec_and_sc_labels()
    assert labels["section"] == "4. The Formal Equivalence of SEC and SC"
    assert labels["next_boundary"] == "P0R03818"


def test_section_4_the_formal_equivalence_of_sec_and_sc_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        Section4TheFormalEquivalenceOfSecAndScConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section4TheFormalEquivalenceOfSecAndScConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03818"):
        Section4TheFormalEquivalenceOfSecAndScConfig(next_source_boundary="P0R03817")
    with pytest.raises(
        ValueError, match="unknown section_4_the_formal_equivalence_of_sec_and_sc component"
    ):
        classify_section_4_the_formal_equivalence_of_sec_and_sc_component(
            "empirical_validation_claim"
        )
