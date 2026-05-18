# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.1 The Composite Functional for Causal Path Entropy validation tests
"""Tests for Paper 0 4.1 The Composite Functional for Causal Path Entropy source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_1_the_composite_functional_for_causal_path_entropy_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section41TheCompositeFunctionalForCausalPathEntropyConfig,
    classify_section_4_1_the_composite_functional_for_causal_path_entropy_component,
    section_4_1_the_composite_functional_for_causal_path_entropy_labels,
    validate_section_4_1_the_composite_functional_for_causal_path_entropy_fixture,
)


def test_section_4_1_the_composite_functional_for_causal_path_entropy_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_1_the_composite_functional_for_causal_path_entropy_fixture()
    assert result.source_ledger_span == ("P0R03818", "P0R03825")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03826"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_1_the_composite_functional_for_causal_path_entropy_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03818"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03825"


def test_section_4_1_the_composite_functional_for_causal_path_entropy_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("4_1_the_composite_functional_for_causal_path_entropy",):
        assert (
            classify_section_4_1_the_composite_functional_for_causal_path_entropy_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_1_the_composite_functional_for_causal_path_entropy_labels()
    assert labels["section"] == "4.1 The Composite Functional for Causal Path Entropy"
    assert labels["next_boundary"] == "P0R03826"


def test_section_4_1_the_composite_functional_for_causal_path_entropy_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section41TheCompositeFunctionalForCausalPathEntropyConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section41TheCompositeFunctionalForCausalPathEntropyConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03826"):
        Section41TheCompositeFunctionalForCausalPathEntropyConfig(next_source_boundary="P0R03825")
    with pytest.raises(
        ValueError,
        match="unknown section_4_1_the_composite_functional_for_causal_path_entropy component",
    ):
        classify_section_4_1_the_composite_functional_for_causal_path_entropy_component(
            "empirical_validation_claim"
        )
