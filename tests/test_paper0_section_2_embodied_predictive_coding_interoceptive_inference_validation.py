# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Embodied Predictive Coding (Interoceptive Inference) validation tests
"""Tests for Paper 0 2. Embodied Predictive Coding (Interoceptive Inference) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_embodied_predictive_coding_interoceptive_inference_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig,
    classify_section_2_embodied_predictive_coding_interoceptive_inference_component,
    section_2_embodied_predictive_coding_interoceptive_inference_labels,
    validate_section_2_embodied_predictive_coding_interoceptive_inference_fixture,
)


def test_section_2_embodied_predictive_coding_interoceptive_inference_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_embodied_predictive_coding_interoceptive_inference_fixture()
    assert result.source_ledger_span == ("P0R04967", "P0R04974")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04975"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_embodied_predictive_coding_interoceptive_inference_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04967"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04974"


def test_section_2_embodied_predictive_coding_interoceptive_inference_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_embodied_predictive_coding_interoceptive_inference",
        "3_distributed_criticality",
    ):
        assert (
            classify_section_2_embodied_predictive_coding_interoceptive_inference_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_embodied_predictive_coding_interoceptive_inference_labels()
    assert labels["section"] == "2. Embodied Predictive Coding (Interoceptive Inference)"
    assert labels["next_boundary"] == "P0R04975"


def test_section_2_embodied_predictive_coding_interoceptive_inference_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04975"):
        Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig(
            next_source_boundary="P0R04974"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_embodied_predictive_coding_interoceptive_inference component",
    ):
        classify_section_2_embodied_predictive_coding_interoceptive_inference_component(
            "empirical_validation_claim"
        )
