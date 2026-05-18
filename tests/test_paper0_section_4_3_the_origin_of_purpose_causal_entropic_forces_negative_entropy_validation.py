# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy validation tests
"""Tests for Paper 0 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig,
    classify_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_component,
    section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_labels,
    validate_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_fixture,
)


def test_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_fixture()
    assert result.source_ledger_span == ("P0R03653", "P0R03663")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03664"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03653"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03663"


def test_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy",
        "causal_entropic_forces_cef",
    ):
        assert (
            classify_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_labels()
    assert (
        labels["section"] == "4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy"
    )
    assert labels["next_boundary"] == "P0R03664"


def test_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03664"):
        Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig(
            next_source_boundary="P0R03663"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy component",
    ):
        classify_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_component(
            "empirical_validation_claim"
        )
