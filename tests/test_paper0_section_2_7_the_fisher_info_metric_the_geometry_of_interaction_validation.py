# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2.7 The Fisher Info Metric: The Geometry of Interaction validation tests
"""Tests for Paper 0 2.7 The Fisher Info Metric: The Geometry of Interaction source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_7_the_fisher_info_metric_the_geometry_of_interaction_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section27TheFisherInfoMetricTheGeometryOfInteractionConfig,
    classify_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_component,
    section_2_7_the_fisher_info_metric_the_geometry_of_interaction_labels,
    validate_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_fixture,
)


def test_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_fixture()
    assert result.source_ledger_span == ("P0R01993", "P0R02010")
    assert result.source_record_count == 18
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02011"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01993"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02010"


def test_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("2_7_the_fisher_info_metric_the_geometry_of_interaction",):
        assert (
            classify_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_7_the_fisher_info_metric_the_geometry_of_interaction_labels()
    assert labels["section"] == "2.7 The Fisher Info Metric: The Geometry of Interaction"
    assert labels["next_boundary"] == "P0R02011"


def test_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 18"):
        Section27TheFisherInfoMetricTheGeometryOfInteractionConfig(expected_source_record_count=17)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section27TheFisherInfoMetricTheGeometryOfInteractionConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02011"):
        Section27TheFisherInfoMetricTheGeometryOfInteractionConfig(next_source_boundary="P0R02010")
    with pytest.raises(
        ValueError,
        match="unknown section_2_7_the_fisher_info_metric_the_geometry_of_interaction component",
    ):
        classify_section_2_7_the_fisher_info_metric_the_geometry_of_interaction_component(
            "empirical_validation_claim"
        )
