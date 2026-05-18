# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Section 9: Concluding Assessment and Future Directions validation tests
"""Tests for Paper 0 Section 9: Concluding Assessment and Future Directions source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_9_concluding_assessment_and_future_directions_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section9ConcludingAssessmentAndFutureDirectionsConfig,
    classify_section_9_concluding_assessment_and_future_directions_component,
    section_9_concluding_assessment_and_future_directions_labels,
    validate_section_9_concluding_assessment_and_future_directions_fixture,
)


def test_section_9_concluding_assessment_and_future_directions_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_9_concluding_assessment_and_future_directions_fixture()
    assert result.source_ledger_span == ("P0R05285", "P0R05292")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05293"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_9_concluding_assessment_and_future_directions_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05285"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05292"


def test_section_9_concluding_assessment_and_future_directions_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "section_9_concluding_assessment_and_future_directions",
        "structural_overview_by_domain",
        "domain_i_the_biological_substrate_layers_1_4",
    ):
        assert (
            classify_section_9_concluding_assessment_and_future_directions_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_9_concluding_assessment_and_future_directions_labels()
    assert labels["section"] == "Section 9: Concluding Assessment and Future Directions"
    assert labels["next_boundary"] == "P0R05293"


def test_section_9_concluding_assessment_and_future_directions_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section9ConcludingAssessmentAndFutureDirectionsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section9ConcludingAssessmentAndFutureDirectionsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05293"):
        Section9ConcludingAssessmentAndFutureDirectionsConfig(next_source_boundary="P0R05292")
    with pytest.raises(
        ValueError, match="unknown section_9_concluding_assessment_and_future_directions component"
    ):
        classify_section_9_concluding_assessment_and_future_directions_component(
            "empirical_validation_claim"
        )
