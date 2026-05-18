# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Retrocausality via the Two-State Vector Formalism (TSVF): validation tests
"""Tests for Paper 0 3. Retrocausality via the Two-State Vector Formalism (TSVF): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig,
    classify_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_component,
    section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_labels,
    validate_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_fixture,
)


def test_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_fixture()
    assert result.source_ledger_span == ("P0R05936", "P0R05943")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05944"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05936"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05943"


def test_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_retrocausality_via_the_two_state_vector_formalism_tsvf",
        "ii_the_thermodynamics_of_consciousness_negentropy_and_information",
    ):
        assert (
            classify_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_labels()
    assert labels["section"] == "3. Retrocausality via the Two-State Vector Formalism (TSVF):"
    assert labels["next_boundary"] == "P0R05944"


def test_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05944"):
        Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig(
            next_source_boundary="P0R05943"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_3_retrocausality_via_the_two_state_vector_formalism_tsvf component",
    ):
        classify_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_component(
            "empirical_validation_claim"
        )
